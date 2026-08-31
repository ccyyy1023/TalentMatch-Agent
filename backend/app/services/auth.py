from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

from sqlalchemy import delete, insert, select
from sqlalchemy.exc import IntegrityError

from app.config import settings
from app.schemas import AuditRecord, LoginResponse, UserCreate, UserView
from app.services.relational import audit_log, auth_sessions, get_engine, users


PBKDF2_ITERATIONS = 310_000


class AuthService:
    """Database-independent identity, bearer sessions and append-only audit records."""

    def __init__(self, path: Path | None = None, session_hours: int | None = None) -> None:
        self.engine = get_engine(path)
        self.session_hours = session_hours or settings.auth_session_hours

    def create_user(self, request: UserCreate) -> UserView:
        username = self._normalize_username(request.username)
        now = self._now()
        user_id = f"usr-{secrets.token_hex(6)}"
        try:
            with self.engine.begin() as connection:
                connection.execute(insert(users).values(
                    id=user_id, username=username, password_hash=self.hash_password(request.password),
                    role=request.role, active=True, created_at=now,
                ))
        except IntegrityError as exc:
            raise ValueError("用户名已存在") from exc
        return UserView(id=user_id, username=username, role=request.role, active=True, created_at=now)

    def login(self, username: str, password: str) -> LoginResponse | None:
        normalized = self._normalize_username(username)
        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    users.c.id, users.c.username, users.c.password_hash,
                    users.c.role, users.c.active, users.c.created_at,
                ).where(users.c.username == normalized)
            ).mappings().first()
        if row is None or not row["active"] or not self.verify_password(password, row["password_hash"]):
            return None
        token = secrets.token_urlsafe(32)
        created_at = datetime.now(timezone.utc)
        expires_at = created_at + timedelta(hours=self.session_hours)
        with self.engine.begin() as connection:
            connection.execute(insert(auth_sessions).values(
                token_hash=self._token_hash(token), user_id=row["id"],
                expires_at=expires_at.isoformat(), created_at=created_at.isoformat(),
            ))
        return LoginResponse(access_token=token, expires_at=expires_at.isoformat(), user=self._user_from_row(row))

    def authenticate(self, token: str) -> UserView | None:
        token_hash = self._token_hash(token)
        statement = (
            select(
                users.c.id, users.c.username, users.c.role, users.c.active,
                users.c.created_at, auth_sessions.c.expires_at,
            )
            .select_from(auth_sessions.join(users, users.c.id == auth_sessions.c.user_id))
            .where(auth_sessions.c.token_hash == token_hash)
        )
        with self.engine.begin() as connection:
            row = connection.execute(statement).mappings().first()
            if row is None:
                return None
            if datetime.fromisoformat(row["expires_at"]) <= datetime.now(timezone.utc) or not row["active"]:
                connection.execute(delete(auth_sessions).where(auth_sessions.c.token_hash == token_hash))
                return None
        return self._user_from_row(row)

    def logout(self, token: str) -> None:
        with self.engine.begin() as connection:
            connection.execute(delete(auth_sessions).where(auth_sessions.c.token_hash == self._token_hash(token)))

    def list_users(self) -> list[UserView]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(users.c.id, users.c.username, users.c.role, users.c.active, users.c.created_at)
                .order_by(users.c.created_at)
            ).mappings().all()
        return [self._user_from_row(row) for row in rows]

    def record_audit(
        self,
        actor: UserView,
        action: str,
        resource_type: str,
        resource_id: str | None = None,
        detail: dict | None = None,
    ) -> None:
        with self.engine.begin() as connection:
            connection.execute(insert(audit_log).values(
                actor_user_id=actor.id, actor_username=actor.username, action=action,
                resource_type=resource_type, resource_id=resource_id,
                detail_json=json.dumps(detail or {}, ensure_ascii=False, separators=(",", ":")),
                created_at=self._now(),
            ))

    def list_audit(self, limit: int = 100) -> list[AuditRecord]:
        statement = (
            select(
                audit_log.c.id, audit_log.c.actor_user_id, audit_log.c.actor_username,
                audit_log.c.action, audit_log.c.resource_type, audit_log.c.resource_id,
                audit_log.c.detail_json, audit_log.c.created_at,
            )
            .order_by(audit_log.c.id.desc())
            .limit(min(max(limit, 1), 500))
        )
        with self.engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return [
            AuditRecord(
                id=row["id"], actor_user_id=row["actor_user_id"], actor_username=row["actor_username"],
                action=row["action"], resource_type=row["resource_type"], resource_id=row["resource_id"],
                detail=json.loads(row["detail_json"]), created_at=row["created_at"],
            )
            for row in rows
        ]

    @staticmethod
    def hash_password(password: str) -> str:
        salt = secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
        return "pbkdf2_sha256${}${}${}".format(
            PBKDF2_ITERATIONS,
            base64.urlsafe_b64encode(salt).decode("ascii"),
            base64.urlsafe_b64encode(digest).decode("ascii"),
        )

    @staticmethod
    def verify_password(password: str, encoded: str) -> bool:
        try:
            algorithm, iterations, salt, expected = encoded.split("$", 3)
            if algorithm != "pbkdf2_sha256":
                return False
            digest = hashlib.pbkdf2_hmac(
                "sha256", password.encode("utf-8"), base64.urlsafe_b64decode(salt), int(iterations),
            )
            return hmac.compare_digest(digest, base64.urlsafe_b64decode(expected))
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_username(username: str) -> str:
        return username.strip().lower()

    @staticmethod
    def _user_from_row(row: Mapping) -> UserView:
        return UserView(
            id=row["id"], username=row["username"], role=row["role"],
            active=bool(row["active"]), created_at=row["created_at"],
        )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
