"""create strategy_persistent_state table

Revision ID: 202604300001
Revises: 202604290001
Create Date: 2026-04-30

Backs ``StrategySDK.PersistentState`` — a durable key/value store
keyed by ``(strategy_slug, key)`` so custom strategies can persist
state (rolling stats, last seen timestamps, multi-window snapshots)
across worker restarts without inventing per-strategy tables.

``BaseStrategy.state`` remains in-memory; this table is the durable
counterpart that strategies opt into via the SDK helper. Values are
JSON to accept any JSON-serialisable Python value.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "202604300001"
down_revision = "202604290001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "strategy_persistent_state",
        sa.Column("strategy_slug", sa.String(), primary_key=True),
        sa.Column("key", sa.String(), primary_key=True),
        sa.Column("value", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "idx_strategy_persistent_state_slug",
        "strategy_persistent_state",
        ["strategy_slug"],
    )
    op.create_index(
        "idx_strategy_persistent_state_updated",
        "strategy_persistent_state",
        ["updated_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_strategy_persistent_state_updated", table_name="strategy_persistent_state")
    op.drop_index("idx_strategy_persistent_state_slug", table_name="strategy_persistent_state")
    op.drop_table("strategy_persistent_state")
