"""Persist scanner opportunity counts on scanner_snapshot.

Revision ID: 202603140002
Revises: 202603140001
Create Date: 2026-03-14 03:35:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

from alembic_helpers import column_names


revision = "202603140002"
down_revision = "202603140001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    scanner_snapshot_columns = column_names("scanner_snapshot")
    if "opportunities_count" not in scanner_snapshot_columns:
        op.add_column(
            "scanner_snapshot",
            sa.Column("opportunities_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        )

    op.execute(
        sa.text(
            """
            UPDATE scanner_snapshot
            SET opportunities_count = COALESCE(json_array_length(opportunities_json), 0)
            WHERE id = 'latest'
            """
        )
    )


def downgrade() -> None:
    pass
