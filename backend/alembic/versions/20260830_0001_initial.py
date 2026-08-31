"""Initial production relational schema.

Revision ID: 20260830_0001
Revises:
Create Date: 2026-08-30
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260830_0001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "runs",
        sa.Column("run_id", sa.String(64), primary_key=True),
        sa.Column("created_at", sa.String(64), nullable=False),
        sa.Column("mode", sa.String(32), nullable=False),
        sa.Column("job_title", sa.String(300), nullable=False),
        sa.Column("candidate_count", sa.Integer(), nullable=False),
        sa.Column("response_json", sa.Text(), nullable=False),
        sa.Column("metrics_json", sa.Text()),
    )
    op.create_table(
        "human_decisions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(64), sa.ForeignKey("runs.run_id"), nullable=False),
        sa.Column("candidate_id", sa.String(128), nullable=False),
        sa.Column("decision", sa.String(32), nullable=False),
        sa.Column("note", sa.Text(), nullable=False),
        sa.Column("reviewer", sa.String(80), nullable=False),
        sa.Column("created_at", sa.String(64), nullable=False),
    )
    op.create_index("uq_decision_run_candidate", "human_decisions", ["run_id", "candidate_id"], unique=True)
    op.create_table(
        "model_cache",
        sa.Column("cache_key", sa.String(64), primary_key=True),
        sa.Column("model_identity", sa.String(300), nullable=False),
        sa.Column("namespace", sa.String(80), nullable=False),
        sa.Column("response_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.String(64), nullable=False),
        sa.Column("last_accessed_at", sa.String(64), nullable=False),
        sa.Column("hit_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_table(
        "users",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("username", sa.String(80), nullable=False, unique=True),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.String(64), nullable=False),
        sa.CheckConstraint("role IN ('admin','recruiter')", name="ck_users_role"),
    )
    op.create_table(
        "auth_sessions",
        sa.Column("token_hash", sa.String(64), primary_key=True),
        sa.Column("user_id", sa.String(64), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("expires_at", sa.String(64), nullable=False),
        sa.Column("created_at", sa.String(64), nullable=False),
    )
    op.create_index("idx_sessions_user_id", "auth_sessions", ["user_id"])
    op.create_table(
        "audit_log",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("actor_user_id", sa.String(64), nullable=False),
        sa.Column("actor_username", sa.String(80), nullable=False),
        sa.Column("action", sa.String(80), nullable=False),
        sa.Column("resource_type", sa.String(80), nullable=False),
        sa.Column("resource_id", sa.String(128)),
        sa.Column("detail_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.String(64), nullable=False),
    )
    op.create_index("idx_audit_created_at", "audit_log", ["created_at"])


def downgrade() -> None:
    op.drop_index("idx_audit_created_at", table_name="audit_log")
    op.drop_table("audit_log")
    op.drop_index("idx_sessions_user_id", table_name="auth_sessions")
    op.drop_table("auth_sessions")
    op.drop_table("users")
    op.drop_table("model_cache")
    op.drop_index("uq_decision_run_candidate", table_name="human_decisions")
    op.drop_table("human_decisions")
    op.drop_table("runs")
