"""Add post_only column to execution_session_legs.

Revision ID: 202602220003
Revises: 202602220002
Create Date: 2026-02-22 01:00:00.000000
"""

import sqlalchemy as sa
from alembic import op
from alembic_helpers import column_names


revision = "202602220003"
down_revision = "202602220002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    existing = column_names("execution_session_legs")
    if not existing:
        return
    if "post_only" not in existing:
        op.add_column(
            "execution_session_legs",
            sa.Column("post_only", sa.Boolean(), nullable=False, server_default="false"),
        )


def downgrade() -> None:
    existing = column_names("execution_session_legs")
    if "post_only" in existing:
        op.drop_column("execution_session_legs", "post_only")
