"""Make high-volume derived/cache/telemetry tables UNLOGGED.

The 2026-05-23 soak showed the single Postgres instance saturated on WAL:
``IO/WALSync`` / ``LWLock/WALWrite`` waits, ``dirty_rows=0`` empty commits
taking 2-5s, and every trading transaction (incl. the already-UNLOGGED
``trade_signal_emissions``) slow purely from systemic contention.  The
instance is the shared write bus for discovery + market-data + news +
scanner + trading, and the non-trading streams dominate WAL:

    discovered_wallets       69.0M writes
    book_delta_events         4.9 GB   (market-data ingestor)
    search_index              2.5M writes
    opportunity_state         1.2M writes
    scanner_market_history    0.89M writes
    ...

These tables are all DERIVED / CACHE / TELEMETRY that rebuild from their
source (the venue, the scanner, the discovery worker), so they do not
need crash durability.  UNLOGGED removes them from the WAL critical path
entirely (same lever that fixed trade_signal_emissions).  Tradeoff: each
is TRUNCATED on an unclean crash and re-accumulates — acceptable given
the relaxed durability posture (synchronous_commit=off, Redis no
persistence) and that none hold orders/positions/financial state.

NOTE: ``ALTER TABLE ... SET UNLOGGED`` rewrites the table once under a
brief ACCESS EXCLUSIVE lock.  ``book_delta_events`` (~4.9GB) is the slow
one; if its lock window matters, ``TRUNCATE book_delta_events`` first
(pure replayable market-data deltas) to make the rewrite instant.
"""

from alembic import op


revision = "202605230001"
down_revision = "202605220001"
branch_labels = None
depends_on = None


_UNLOGGED_TABLES = (
    "book_delta_events",
    "scanner_market_history",
    "cached_markets",
    "market_tags_seen",
    "search_index",
    "discovered_wallets",
    "opportunity_state",
    "market_confluence_signals",
    "wallet_activity_rollups",
)


def upgrade() -> None:
    for table in _UNLOGGED_TABLES:
        op.execute(f"ALTER TABLE {table} SET UNLOGGED")


def downgrade() -> None:
    for table in _UNLOGGED_TABLES:
        op.execute(f"ALTER TABLE {table} SET LOGGED")
