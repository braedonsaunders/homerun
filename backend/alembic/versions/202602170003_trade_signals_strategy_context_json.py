"""Add strategy_context_json column to trade_signals.

Workers now persist per-signal strategy context for runtime evaluation and exit
logic. Older databases can be at head without this column, so keep this
migration idempotent.

Revision ID: 202602170003
Revises: 9069a6422cfd
Create Date: 2026-02-17 01:50:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from alembic_helpers import column_names, table_names


# revision identifiers, used by Alembic.
revision = "202602170003"
down_revision = "9069a6422cfd"
branch_labels = None
depends_on = None


def upgrade() -> None:
    table_name = "trade_signals"
    if table_name not in table_names():
        return
    existing_columns = column_names(table_name)
    if "strategy_context_json" not in existing_columns:
        op.add_column(table_name, sa.Column("strategy_context_json", sa.JSON(), nullable=True))


def downgrade() -> None:
    # Explicit downgrade support omitted for SQLite safety.
    pass
