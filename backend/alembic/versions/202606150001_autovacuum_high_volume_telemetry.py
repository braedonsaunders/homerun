"""Aggressive per-table autovacuum/analyze on high-volume telemetry tables

Revision ID: 202606150001
Revises: 202606130001
Create Date: 2026-06-15

The 20h soak captured ``trade_signal_emissions`` at ~1.1M rows and
``trader_decision_checks`` at ~3.2M rows, both append-only and pruned only by
the daily ``full_cleanup``. A dedicated frequent retention loop now trims them
(see ``MaintenanceService.start_high_volume_retention``), which produces periodic
batched-DELETE dead tuples that must be reclaimed promptly, and these tables grow
fast enough between analyzes that the planner's row estimates can lag under the
default ``autovacuum_analyze_scale_factor = 0.20`` (a 1M-row table needs ~200k
new rows before a re-analyze).

Unlike the tiny hot UPSERT tables in 202605030001 (threshold = 5, scale = 0,
appropriate for 2-8 live rows), these are large append tables — a zero
scale-factor would trigger near-constant full-table vacuums. We instead set a
low-but-non-zero scale factor plus a row-count floor:

    autovacuum_vacuum_scale_factor   = 0.05   # reclaim post-prune dead tuples
    autovacuum_vacuum_threshold      = 10000
    autovacuum_analyze_scale_factor  = 0.02   # keep planner stats fresh as they grow
    autovacuum_analyze_threshold     = 5000

``trade_signal_emissions`` is UNLOGGED; autovacuum still applies to it (UNLOGGED
removes WAL/crash-safety, not dead-tuple maintenance).

Runtime tunings — take effect immediately on the existing tables without a
rewrite. The downgrade restores defaults (``RESET``).
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


def _is_plain_table(table: str) -> bool:
    """True only for an ordinary (relkind='r') table.

    On the fresh-install path ``Base.metadata.create_all`` builds these tables
    PARTITIONED (relkind='p'), where storage/autovacuum reloptions are invalid
    (no storage) — the partition-lifecycle job tunes each daily partition
    instead, so this migration must be a no-op there. On the pre-partition live
    DB they are still plain and get tuned here.
    """
    # Cast to text: asyncpg returns pg's "char" type as bytes (b'r'), which would
    # never == the str "r".
    relkind = (
        op.get_bind()
        .execute(sa.text("SELECT relkind::text FROM pg_class WHERE relname = :t"), {"t": table})
        .scalar()
    )
    return relkind == "r"


# revision identifiers, used by Alembic.
revision = "202606150001"
down_revision = "202606130001"
branch_labels = None
depends_on = None


_HIGH_VOLUME_TABLES = (
    "trade_signal_emissions",
    "trader_decision_checks",
)


def upgrade() -> None:
    for table in _HIGH_VOLUME_TABLES:
        if not _is_plain_table(table):
            continue
        op.execute(
            f"""
            ALTER TABLE {table} SET (
                autovacuum_vacuum_threshold = 10000,
                autovacuum_vacuum_scale_factor = 0.05,
                autovacuum_analyze_threshold = 5000,
                autovacuum_analyze_scale_factor = 0.02
            )
            """
        )


def downgrade() -> None:
    for table in _HIGH_VOLUME_TABLES:
        if not _is_plain_table(table):
            continue
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
