"""Add persistent screening batches and candidate workflow items.

Revision ID: 20260831_0003
Revises: 20260830_0002
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260831_0003"
down_revision: Union[str, Sequence[str], None] = "20260830_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "screening_batches",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("job_id", sa.String(64), sa.ForeignKey("jobs.id"), nullable=False),
        sa.Column("task_id", sa.String(64), nullable=False, unique=True),
        sa.Column("run_id", sa.String(64), sa.ForeignKey("runs.run_id")),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("mode", sa.String(32), nullable=False),
        sa.Column("error", sa.Text()),
        sa.Column("created_by", sa.String(64), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("created_at", sa.String(64), nullable=False),
        sa.Column("updated_at", sa.String(64), nullable=False),
        sa.CheckConstraint(
            "status IN ('queued','running','awaiting_review','reviewed','failed')",
            name="ck_screening_batches_status",
        ),
    )
    op.create_index("idx_screening_batches_status_updated", "screening_batches", ["status", "updated_at"])
    op.create_table(
        "screening_batch_items",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("batch_id", sa.String(64), sa.ForeignKey("screening_batches.id", ondelete="CASCADE"), nullable=False),
        sa.Column("candidate_id", sa.String(64), sa.ForeignKey("candidates.id"), nullable=False),
        sa.Column("stage", sa.String(24), nullable=False),
        sa.Column("decision", sa.String(24)),
        sa.Column("note", sa.Text(), nullable=False, server_default=""),
        sa.Column("reviewer", sa.String(80)),
        sa.Column("decided_at", sa.String(64)),
        sa.CheckConstraint(
            "stage IN ('pending','analyzed','advanced','held','not_advanced')",
            name="ck_screening_batch_items_stage",
        ),
        sa.CheckConstraint(
            "decision IS NULL OR decision IN ('advance','hold','not_advance')",
            name="ck_screening_batch_items_decision",
        ),
        sa.UniqueConstraint("batch_id", "candidate_id", name="uq_screening_batch_candidate"),
    )
    op.create_index("idx_screening_items_candidate", "screening_batch_items", ["candidate_id"])


def downgrade() -> None:
    op.drop_index("idx_screening_items_candidate", table_name="screening_batch_items")
    op.drop_table("screening_batch_items")
    op.drop_index("idx_screening_batches_status_updated", table_name="screening_batches")
    op.drop_table("screening_batches")
