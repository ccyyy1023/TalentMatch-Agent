"""Add managed jobs and candidates.

Revision ID: 20260830_0002
Revises: 20260830_0001
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260830_0002"
down_revision: Union[str, Sequence[str], None] = "20260830_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("created_by", sa.String(64), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("created_at", sa.String(64), nullable=False),
        sa.Column("updated_at", sa.String(64), nullable=False),
        sa.CheckConstraint("status IN ('draft','active','closed')", name="ck_jobs_status"),
    )
    op.create_index("idx_jobs_status_updated", "jobs", ["status", "updated_at"])
    op.create_table(
        "candidates",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("external_ref", sa.String(128)),
        sa.Column("display_name", sa.String(160), nullable=False),
        sa.Column("resume_text", sa.Text(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("created_by", sa.String(64), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("created_at", sa.String(64), nullable=False),
        sa.Column("updated_at", sa.String(64), nullable=False),
        sa.CheckConstraint("status IN ('new','reviewing','archived')", name="ck_candidates_status"),
    )
    op.create_index("idx_candidates_status_updated", "candidates", ["status", "updated_at"])


def downgrade() -> None:
    op.drop_index("idx_candidates_status_updated", table_name="candidates")
    op.drop_table("candidates")
    op.drop_index("idx_jobs_status_updated", table_name="jobs")
    op.drop_table("jobs")
