"""Functional index for live_execution_service._derive_pnl_counters_from_orders

Revision ID: 202605020001
Revises: 202605010004
Create Date: 2026-05-02

Cycle 6 of the perf-harness loop pinpointed
``_derive_pnl_counters_from_orders`` as the surviving offender for
``slow_persist_runtime_state`` — the 30s TTL cache absorbs most calls
but cache-miss aggregations average 3.5s under DB pressure.

The query filters by
``lower(coalesce(execution_wallet_address, '')) = :wallet_lower``
combined with ``actual_profit IS NOT NULL``.  The functional predicate
defeats the existing btree on ``execution_wallet_address`` (writers
across the codebase are inconsistent — some write the wallet
lower-cased, some only ``.strip()``, so the application can't drop
the function defensively).

Add a partial functional index that EXACTLY matches the query
predicate.  PG can then satisfy the SUM/COUNT-with-CASE aggregates
via an Index Only Scan over the (wallet_lower, actual_profit) leaves
without touching the heap.

Partial-index ``WHERE actual_profit IS NOT NULL`` keeps the index
small (only verified-profit rows; pending/open orders are excluded
from both the index and the aggregate, so this is exact).
"""
from __future__ import annotations

from alembic import op


# revision identifiers, used by Alembic.
revision = "202605020001"
down_revision = "202605010004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_trader_orders_wallet_lower_profit
            ON trader_orders (
                lower(coalesce(execution_wallet_address, '')),
                actual_profit,
                coalesce(executed_at, created_at)
            )
            WHERE actual_profit IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_trader_orders_wallet_lower_profit")
