from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from sqlalchemy import Boolean, CheckConstraint, Column, ForeignKey, Index, Integer, MetaData, String, Table, Text, UniqueConstraint, create_engine, event, text
from sqlalchemy.engine import Engine

from app.config import settings


metadata = MetaData()

runs = Table(
    "runs", metadata,
    Column("run_id", String(64), primary_key=True),
    Column("created_at", String(64), nullable=False),
    Column("mode", String(32), nullable=False),
    Column("job_title", String(300), nullable=False),
    Column("candidate_count", Integer, nullable=False),
    Column("response_json", Text, nullable=False),
    Column("metrics_json", Text),
)

human_decisions = Table(
    "human_decisions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("candidate_id", String(128), nullable=False),
    Column("decision", String(32), nullable=False),
    Column("note", Text, nullable=False),
    Column("reviewer", String(80), nullable=False),
    Column("created_at", String(64), nullable=False),
)
Index("uq_decision_run_candidate", human_decisions.c.run_id, human_decisions.c.candidate_id, unique=True)

model_cache = Table(
    "model_cache", metadata,
    Column("cache_key", String(64), primary_key=True),
    Column("model_identity", String(300), nullable=False),
    Column("namespace", String(80), nullable=False),
    Column("response_json", Text, nullable=False),
    Column("created_at", String(64), nullable=False),
    Column("last_accessed_at", String(64), nullable=False),
    Column("hit_count", Integer, nullable=False, default=0),
)

users = Table(
    "users", metadata,
    Column("id", String(64), primary_key=True),
    Column("username", String(80), nullable=False, unique=True),
    Column("password_hash", Text, nullable=False),
    Column("role", String(20), nullable=False),
    Column("active", Boolean, nullable=False, default=True),
    Column("created_at", String(64), nullable=False),
    CheckConstraint("role IN ('admin','recruiter')", name="ck_users_role"),
)

auth_sessions = Table(
    "auth_sessions", metadata,
    Column("token_hash", String(64), primary_key=True),
    Column("user_id", String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("expires_at", String(64), nullable=False),
    Column("created_at", String(64), nullable=False),
)
Index("idx_sessions_user_id", auth_sessions.c.user_id)

audit_log = Table(
    "audit_log", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("actor_user_id", String(64), nullable=False),
    Column("actor_username", String(80), nullable=False),
    Column("action", String(80), nullable=False),
    Column("resource_type", String(80), nullable=False),
    Column("resource_id", String(128)),
    Column("detail_json", Text, nullable=False),
    Column("created_at", String(64), nullable=False),
)
Index("idx_audit_created_at", audit_log.c.created_at.desc())

jobs = Table(
    "jobs", metadata,
    Column("id", String(64), primary_key=True),
    Column("title", String(300), nullable=False),
    Column("description", Text, nullable=False),
    Column("status", String(20), nullable=False),
    Column("created_by", String(64), ForeignKey("users.id"), nullable=False),
    Column("created_at", String(64), nullable=False),
    Column("updated_at", String(64), nullable=False),
    CheckConstraint("status IN ('draft','active','closed')", name="ck_jobs_status"),
)
Index("idx_jobs_status_updated", jobs.c.status, jobs.c.updated_at.desc())

candidates = Table(
    "candidates", metadata,
    Column("id", String(64), primary_key=True),
    Column("external_ref", String(128)),
    Column("display_name", String(160), nullable=False),
    Column("resume_text", Text, nullable=False),
    Column("status", String(20), nullable=False),
    Column("created_by", String(64), ForeignKey("users.id"), nullable=False),
    Column("created_at", String(64), nullable=False),
    Column("updated_at", String(64), nullable=False),
    CheckConstraint("status IN ('new','reviewing','archived')", name="ck_candidates_status"),
)
Index("idx_candidates_status_updated", candidates.c.status, candidates.c.updated_at.desc())

screening_batches = Table(
    "screening_batches", metadata,
    Column("id", String(64), primary_key=True),
    Column("job_id", String(64), ForeignKey("jobs.id"), nullable=False),
    Column("task_id", String(64), nullable=False, unique=True),
    Column("run_id", String(64), ForeignKey("runs.run_id")),
    Column("status", String(24), nullable=False),
    Column("mode", String(32), nullable=False),
    Column("error", Text),
    Column("created_by", String(64), ForeignKey("users.id"), nullable=False),
    Column("created_at", String(64), nullable=False),
    Column("updated_at", String(64), nullable=False),
    CheckConstraint(
        "status IN ('queued','running','awaiting_review','reviewed','failed')",
        name="ck_screening_batches_status",
    ),
)
Index("idx_screening_batches_status_updated", screening_batches.c.status, screening_batches.c.updated_at.desc())

screening_batch_items = Table(
    "screening_batch_items", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("batch_id", String(64), ForeignKey("screening_batches.id", ondelete="CASCADE"), nullable=False),
    Column("candidate_id", String(64), ForeignKey("candidates.id"), nullable=False),
    Column("stage", String(24), nullable=False),
    Column("decision", String(24)),
    Column("note", Text, nullable=False, default=""),
    Column("reviewer", String(80)),
    Column("decided_at", String(64)),
    CheckConstraint(
        "stage IN ('pending','analyzed','advanced','held','not_advanced')",
        name="ck_screening_batch_items_stage",
    ),
    CheckConstraint(
        "decision IS NULL OR decision IN ('advance','hold','not_advance')",
        name="ck_screening_batch_items_decision",
    ),
    UniqueConstraint("batch_id", "candidate_id", name="uq_screening_batch_candidate"),
)
Index("idx_screening_items_candidate", screening_batch_items.c.candidate_id)


def database_url(path: Path | None = None) -> str:
    return f"sqlite:///{path.resolve().as_posix()}" if path is not None else settings.database_url


@lru_cache(maxsize=16)
def engine_for_url(url: str) -> Engine:
    connect_args = {"timeout": 10, "check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, pool_pre_ping=True, connect_args=connect_args)
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode = WAL")
            cursor.execute("PRAGMA busy_timeout = 10000")
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.close()
    return engine


def get_engine(path: Path | None = None) -> Engine:
    url = database_url(path)
    engine = engine_for_url(url)
    if url.startswith("sqlite"):
        metadata.create_all(engine)
    return engine


def database_health() -> dict:
    engine = engine_for_url(settings.database_url)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {"available": True, "dialect": engine.dialect.name}
    except Exception as exc:
        return {"available": False, "dialect": engine.dialect.name, "error": str(exc)[:200]}
