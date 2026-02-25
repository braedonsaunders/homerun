"""Scope traders to a single execution mode.

Revision ID: 202602250001
Revises: 202602240002
Create Date: 2026-02-25 00:10:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from alembic_helpers import column_names, table_names


revision = "202602250001"
down_revision = "202602240002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if "traders" not in table_names():
        return

    existing_cols = column_names("traders")
    if "mode" not in existing_cols:
        op.add_column(
            "traders",
            sa.Column("mode", sa.String(), nullable=False, server_default=sa.text("'paper'")),
        )

    op.execute(
        sa.text(
            """
            UPDATE traders
            SET mode = LOWER(TRIM(COALESCE(mode, 'paper')))
            """
        )
    )
    op.execute(
        sa.text(
            """
            UPDATE traders
            SET mode = 'paper'
            WHERE mode NOT IN ('paper', 'live')
            """
        )
    )

    if "trader_orders" in table_names():
        op.execute(
            sa.text(
                """
                WITH mode_rollup AS (
                    SELECT
                        trader_id,
                        LOWER(TRIM(mode)) AS mode_key,
                        COUNT(*) AS mode_count,
                        MAX(COALESCE(updated_at, executed_at, created_at)) AS latest_ts
                    FROM trader_orders
                    WHERE LOWER(TRIM(COALESCE(mode, ''))) IN ('paper', 'live')
                    GROUP BY trader_id, LOWER(TRIM(mode))
                ),
                ranked AS (
                    SELECT
                        trader_id,
                        mode_key,
                        ROW_NUMBER() OVER (
                            PARTITION BY trader_id
                            ORDER BY mode_count DESC, latest_ts DESC, mode_key ASC
                        ) AS rn
                    FROM mode_rollup
                )
                UPDATE traders AS t
                SET mode = ranked.mode_key
                FROM ranked
                WHERE ranked.rn = 1
                  AND ranked.trader_id = t.id
                """
            )
        )

    op.execute(
        sa.text(
            """
            UPDATE traders
            SET mode = 'paper'
            WHERE mode IS NULL OR mode = ''
            """
        )
    )


def downgrade() -> None:
    return
