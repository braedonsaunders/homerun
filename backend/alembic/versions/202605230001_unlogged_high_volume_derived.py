"""Make high-volume derived/cache/telemetry tables UNLOGGED.

The 2026-05-23 soak showed the single Postgres instance saturated on WAL
(IO/WALSync + LWLock/WALWrite waits; dirty_rows=0 empty commits taking
2-5s; every trading txn slow purely from systemic contention — incl. the
already-UNLOGGED ``trade_signal_emissions``). The instance is the shared
write bus for discovery + market-data + news + scanner + trading, and
the non-trading streams dominate WAL (discovered_wallets 69M writes,
book_delta_events 4.9GB, ...).

These tables are all DERIVED / CACHE / TELEMETRY that rebuild from their
source, so they don't need crash durability. UNLOGGED removes them from
the WAL critical path (same lever that fixed trade_signal_emissions).
Tradeoff: truncated on unclean crash, re-accumulated; none hold orders/
positions/financial state.

IDEMPOTENT + LOCK-SAFE (2026-05-23 boot fix): the first version of this
migration ran a bare ``ALTER TABLE ... SET UNLOGGED`` per table. That
takes ACCESS EXCLUSIVE, which can't be granted within the server's
``lock_timeout`` (5s) while the high-churn tables are being written
during a live boot — so the migration aborted and ``init_database``
retried it ~30x, then the backend exited (never booted). Fix:

  * Skip any table already in the target persistence via a cheap
    ``pg_class`` read (AccessShareLock — never blocked by writers), so a
    table already converted (e.g. out-of-band) costs zero locks and the
    migration is a pure no-op + version bump.
  * For tables still needing conversion, drop ``lock_timeout`` /
    ``statement_timeout`` for this migration so the ALTER WAITS for the
    lock + rewrite instead of failing fast and looping the boot. (On a
    fresh install these tables are created UNLOGGED via the model
    ``prefixes`` + ``create_all``, so this branch is only hit by an
    existing LOGGED DB.)
"""

import sqlalchemy as sa
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


def _flip_persistence(target_char: str, alter_clause: str) -> None:
    bind = op.get_bind()
    pending = []
    for table in _UNLOGGED_TABLES:
        current = bind.execute(
            sa.text(
                "SELECT relpersistence FROM pg_class "
                "WHERE relname = :t AND relkind = 'r'"
            ),
            {"t": table},
        ).scalar()
        # None = table doesn't exist (skip); already-target = skip (no lock).
        if current is not None and current != target_char:
            pending.append(table)
    if not pending:
        return
    # Only reached when an existing LOGGED DB still needs conversion. Wait
    # for the lock/rewrite rather than failing fast and looping the boot.
    op.execute("SET LOCAL lock_timeout = 0")
    op.execute("SET LOCAL statement_timeout = 0")
    for table in pending:
        op.execute(f"ALTER TABLE {table} {alter_clause}")


def upgrade() -> None:
    _flip_persistence("u", "SET UNLOGGED")


def downgrade() -> None:
    _flip_persistence("p", "SET LOGGED")
