"""Drop redundant idx_trader_orders_wallet_lower_profit (cycle 7 finding)

Revision ID: 202605020002
Revises: 202605020001
Create Date: 2026-05-02

Cycle 7 of the perf-harness loop verified the assumption behind
202605020001 was wrong.  EXPLAIN ANALYZE on the
``_derive_pnl_counters_from_orders`` aggregate shows postgres is
already choosing the pre-existing partial index
``idx_trader_orders_wallet_realized_pnl`` (which covers the
wallet predicate via ``lower(coalesce(execution_wallet_address))``
and partial WHERE ``actual_profit IS NOT NULL``):

    Index Scan using idx_trader_orders_wallet_realized_pnl
        Index Cond: (lower(coalesce(execution_wallet_address)) = ...)
    Execution Time: 0.999 ms

The query is already 1ms in isolation.  The 3.5s cache-miss
latency observed in production is DB POOL CONTENTION (waiting for
a free connection while writers are flushing), not query speed.

Index ``idx_trader_orders_wallet_lower_profit`` is therefore pure
write-amplification cost on every UPDATE/INSERT to trader_orders
without query benefit.  Drop it.

The right next-step fix is extending the cache TTL (handled in the
service layer, not a migration) so cache misses are rarer.
"""
from __future__ import annotations

from alembic import op


# revision identifiers, used by Alembic.
revision = "202605020002"
down_revision = "202605020001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_trader_orders_wallet_lower_profit")


def downgrade() -> None:
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
