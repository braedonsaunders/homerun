"""Aggressive autovacuum on tiny hot UPSERT tables (cycle 13 finding)

Revision ID: 202605030001
Revises: 202605020002
Create Date: 2026-05-03

Cycle 13 of the perf-harness loop captured a 2735 ms UPSERT slow
event on ``live_trading_runtime_state`` with no lock contention,
no derive_pnl penalty (Fix R held the cache hot), and
``session_checkout=0`` (pool fast).  ``pg_stat_user_tables``
shows the cause:

    relname                    | n_live | n_dead | dead_pct
    live_trading_runtime_state |   2    |   45   | 2250 %
    live_trading_positions     |   8    |   46   |  575 %

Both tables are wallet-scoped: the live row count is bounded by
the number of active wallets (typically 1-3 in this deployment).
Every successful order UPSERTs the runtime-state row again,
producing one dead tuple per UPDATE.

Postgres' default autovacuum threshold is::

    threshold = autovacuum_vacuum_threshold +
                autovacuum_vacuum_scale_factor * n_live_tup
              = 50 + 0.20 * n_live_tup

For a 2-row table that's 50.4 dead tuples before autovacuum
fires.  Order flow churns dead-tuple count from 0 → 50 in a
few minutes, so we sit near the threshold permanently — the
PK index has 25-50 dead tuple versions to skip on every
UPSERT, and HOT prune isn't enough to rescue the page.

Per-table autovacuum tuning fixes this without touching
indexes.  Aggressive thresholds keep the dead-tuple count near
zero so PK lookups stay clean::

    autovacuum_vacuum_threshold       = 5
    autovacuum_vacuum_scale_factor    = 0
    autovacuum_analyze_threshold      = 5
    autovacuum_analyze_scale_factor   = 0

Same pattern applied to ``live_trading_positions`` (also small
and UPSERT-heavy) and ``live_trading_orders`` (359 live, 56 dead
= 15.6 % — borderline; tightening is cheap insurance, the
table is still small enough that aggressive vacuum is fast).

These are runtime tunings — they take effect immediately on
existing tables without a rewrite.  The downgrade restores
defaults (``RESET``).
"""
from __future__ import annotations

from alembic import op


# revision identifiers, used by Alembic.
revision = "202605030001"
down_revision = "202605020002"
branch_labels = None
depends_on = None


_AGGRESSIVE_TABLES = (
    "live_trading_runtime_state",
    "live_trading_positions",
    "live_trading_orders",
)


def upgrade() -> None:
    for table in _AGGRESSIVE_TABLES:
        op.execute(
            f"""
            ALTER TABLE {table} SET (
                autovacuum_vacuum_threshold = 5,
                autovacuum_vacuum_scale_factor = 0.0,
                autovacuum_analyze_threshold = 5,
                autovacuum_analyze_scale_factor = 0.0
            )
            """
        )
    # NOTE: ``VACUUM`` cannot run inside a transaction block, so
    # the catch-up vacuum must be done outside the migration.  The
    # new autovacuum thresholds take effect immediately for future
    # operations; existing dead tuples will be cleaned on the next
    # autovacuum cycle (typically within a minute under default
    # ``autovacuum_naptime``).  For an immediate manual catch-up
    # run, see ``scripts/db/vacuum_hot_tables.sql``.


def downgrade() -> None:
    for table in _AGGRESSIVE_TABLES:
        op.execute(
            f"""
            ALTER TABLE {table} RESET (
                autovacuum_vacuum_threshold,
                autovacuum_vacuum_scale_factor,
                autovacuum_analyze_threshold,
                autovacuum_analyze_scale_factor
            )
            """
        )
