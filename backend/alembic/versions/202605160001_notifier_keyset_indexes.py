"""Notifier keyset cursor indexes.

Revision ID: 202605160001
Revises: 202605150001
Create Date: 2026-05-16

Adds composite ``(created_at, id)`` / ``(updated_at, id)`` indexes on the
two tables the autotrader notifier scans every poll cycle.  The notifier
uses row-value keyset pagination — ``WHERE (created_at, id) > ($1, $2)
ORDER BY created_at ASC, id ASC LIMIT N`` — which needs a leading-keyed
composite index to seek-and-stop at the LIMIT.  Without these the
planner falls back to a sort over the whole 15M-row heap (~25 s).

Indexes are built CONCURRENTLY so the migration can run on a live DB
without blocking writers on these hot append tables.
"""

from __future__ import annotations

from alembic import op


revision = "202605160001"
down_revision = "202605150001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS
                idx_trader_events_created_at_id
            ON trader_events (created_at, id)
            """
        )
        op.execute(
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS
                idx_trader_orders_created_at_id
            ON trader_orders (created_at, id)
            """
        )
        op.execute(
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS
                idx_trader_orders_updated_at_id
            ON trader_orders (updated_at, id)
            WHERE updated_at IS NOT NULL
            """
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_trader_orders_updated_at_id")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_trader_orders_created_at_id")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_trader_events_created_at_id")
