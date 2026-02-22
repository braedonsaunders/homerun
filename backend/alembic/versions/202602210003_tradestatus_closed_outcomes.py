"""Add closed win/loss simulation statuses and backfill non-resolution exits.

Revision ID: 202602210003
Revises: 202602210002
Create Date: 2026-02-21 17:10:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from alembic_helpers import table_names


revision = "202602210003"
down_revision = "202602210002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    with op.get_context().autocommit_block():
        op.execute(sa.text("ALTER TYPE tradestatus ADD VALUE IF NOT EXISTS 'CLOSED_WIN'"))
        op.execute(sa.text("ALTER TYPE tradestatus ADD VALUE IF NOT EXISTS 'CLOSED_LOSS'"))

    existing_tables = table_names()

    if "trader_orders" in existing_tables:
        op.execute(
            sa.text(
                """
                UPDATE trader_orders
                SET status = CASE
                    WHEN status = 'resolved_win' THEN 'closed_win'
                    WHEN status = 'resolved_loss' THEN 'closed_loss'
                    ELSE status
                END
                WHERE status IN ('resolved_win', 'resolved_loss')
                  AND COALESCE(payload_json->'position_close'->>'close_trigger', '') <> ''
                  AND LOWER(payload_json->'position_close'->>'close_trigger') NOT IN (
                    'resolution',
                    'resolution_inferred'
                  )
                """
            )
        )
        op.execute(
            sa.text(
                """
                UPDATE trader_orders
                SET payload_json = jsonb_set(
                    payload_json::jsonb,
                    '{position_close,simulation_close,trade_status}',
                    to_jsonb(status),
                    true
                )::json
                WHERE status IN ('closed_win', 'closed_loss')
                  AND COALESCE(payload_json->'position_close'->'simulation_close'->>'trade_status', '') IN (
                    'resolved_win',
                    'resolved_loss'
                  )
                  AND COALESCE(payload_json->'position_close'->>'close_trigger', '') <> ''
                  AND LOWER(payload_json->'position_close'->>'close_trigger') NOT IN (
                    'resolution',
                    'resolution_inferred'
                  )
                """
            )
        )

    if "simulation_trades" in existing_tables:
        op.execute(
            sa.text(
                """
                UPDATE simulation_trades
                SET status = CASE
                    WHEN status::text = 'RESOLVED_WIN' THEN 'CLOSED_WIN'::tradestatus
                    WHEN status::text = 'RESOLVED_LOSS' THEN 'CLOSED_LOSS'::tradestatus
                    ELSE status
                END
                WHERE status::text IN ('RESOLVED_WIN', 'RESOLVED_LOSS')
                  AND COALESCE(positions_data->0->>'close_trigger', '') <> ''
                  AND LOWER(positions_data->0->>'close_trigger') NOT IN (
                    'resolution',
                    'resolution_inferred'
                  )
                """
            )
        )
        if "trader_orders" in existing_tables:
            op.execute(
                sa.text(
                    """
                    UPDATE simulation_trades AS t
                    SET status = CASE
                        WHEN t.status::text = 'RESOLVED_WIN' THEN 'CLOSED_WIN'::tradestatus
                        WHEN t.status::text = 'RESOLVED_LOSS' THEN 'CLOSED_LOSS'::tradestatus
                        ELSE t.status
                    END
                    FROM trader_orders AS o
                    WHERE t.status::text IN ('RESOLVED_WIN', 'RESOLVED_LOSS')
                      AND COALESCE(o.payload_json->'simulation_ledger'->>'trade_id', '') = t.id
                      AND COALESCE(o.payload_json->'position_close'->>'close_trigger', '') <> ''
                      AND LOWER(o.payload_json->'position_close'->>'close_trigger') NOT IN (
                        'resolution',
                        'resolution_inferred'
                      )
                    """
                )
            )

def downgrade() -> None:
    pass
