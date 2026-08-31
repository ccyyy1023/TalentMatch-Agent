from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func, insert, select, update

from app.services.relational import get_engine, model_cache


class ModelResponseCache:
    """Persistent cache keyed by model, prompt, input and options."""

    def __init__(self, path: Path | None = None):
        self.engine = get_engine(path)

    @staticmethod
    def build_key(
        model_identity: str, namespace: str, prompt_version: str,
        system: str, user: str, options: dict[str, Any],
    ) -> str:
        canonical = json.dumps(
            {
                "model_identity": model_identity,
                "namespace": namespace,
                "prompt_version": prompt_version,
                "system": system,
                "user": user,
                "options": options,
            },
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> dict[str, Any] | None:
        now = datetime.now(timezone.utc).isoformat()
        with self.engine.begin() as connection:
            row = connection.execute(
                select(model_cache.c.response_json).where(model_cache.c.cache_key == cache_key)
            ).mappings().first()
            if not row:
                return None
            connection.execute(
                update(model_cache)
                .where(model_cache.c.cache_key == cache_key)
                .values(hit_count=model_cache.c.hit_count + 1, last_accessed_at=now)
            )
        return json.loads(row["response_json"])

    def put(self, cache_key: str, model_identity: str, namespace: str, response: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        values = {
            "cache_key": cache_key,
            "model_identity": model_identity,
            "namespace": namespace,
            "response_json": json.dumps(response, ensure_ascii=False),
            "created_at": now,
            "last_accessed_at": now,
            "hit_count": 0,
        }
        with self.engine.begin() as connection:
            exists = connection.execute(
                select(model_cache.c.cache_key).where(model_cache.c.cache_key == cache_key)
            ).first()
            if exists:
                connection.execute(
                    update(model_cache)
                    .where(model_cache.c.cache_key == cache_key)
                    .values(response_json=values["response_json"], last_accessed_at=now)
                )
            else:
                connection.execute(insert(model_cache).values(**values))

    def stats(self) -> dict[str, int]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(func.count().label("entries"), func.coalesce(func.sum(model_cache.c.hit_count), 0).label("hits"))
            ).mappings().one()
        return {"entries": int(row["entries"]), "hits": int(row["hits"])}
