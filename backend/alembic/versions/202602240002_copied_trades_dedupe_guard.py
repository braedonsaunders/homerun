"""Enforce copy-trader dedupe identity per config/source trade.

Revision ID: 202602240002
Revises: 202602240001
Create Date: 2026-02-24 00:15:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from alembic_helpers import index_names, table_names


revision = "202602240002"
down_revision = "202602240001"
branch_labels = None
depends_on = None


def _unique_constraint_names(table_name: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if table_name not in set(inspector.get_table_names()):
        return set()
    return {
        constraint.get("name") for constraint in inspector.get_unique_constraints(table_name) if constraint.get("name")
    }


def upgrade() -> None:
    table_name = "copied_trades"
    guard_name = "uq_copied_trades_config_source_trade"

    if table_name not in table_names():
        return

    # Keep the newest row for each (config_id, source_trade_id) pair.
    op.execute(
        sa.text(
            """
            DELETE FROM copied_trades
            WHERE id IN (
                SELECT id
                FROM (
                    SELECT
                        id,
                        ROW_NUMBER() OVER (
                            PARTITION BY config_id, source_trade_id
                            ORDER BY copied_at DESC, id DESC
                        ) AS row_num
                    FROM copied_trades
                ) ranked
                WHERE ranked.row_num > 1
            )
            """
        )
    )

    unique_names = _unique_constraint_names(table_name)
    idx_names = index_names(table_name)
    if guard_name not in unique_names and guard_name not in idx_names:
        op.create_index(
            guard_name,
            table_name,
            ["config_id", "source_trade_id"],
            unique=True,
        )


def downgrade() -> None:
    return
