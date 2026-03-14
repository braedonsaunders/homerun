"""Persist pending live-trading reconciliation payloads.

Revision ID: 202603140001
Revises: 202603130001
Create Date: 2026-03-14 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

from alembic_helpers import column_names


revision = "202603140001"
down_revision = "202603130001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    runtime_columns = column_names("live_trading_runtime_state")
    if "pending_reconciliation_json" not in runtime_columns:
        op.add_column(
            "live_trading_runtime_state",
            sa.Column("pending_reconciliation_json", sa.JSON(), nullable=True),
        )


def downgrade() -> None:
    pass
