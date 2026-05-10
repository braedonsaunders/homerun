"""Tune trade_signals fillfactor + autovacuum on the hot tables.

Diagnosis from the 2026-05-09 15:55-16:04 soak (SLOW COMMIT DIAGNOSTIC
captures via worker_state._capture_slow_commit_diagnostic):

  * ``trade_signals.hot_pct = 0.0%`` (121 HOT updates / 557,989 total
    updates). Every UPDATE rebuilds B-tree entries in 3 of 4 indexes
    (status is in idx_trade_signals_source_status,
    idx_trade_signals_source_status_sequence, idx_trade_signals_market_status).
    Orchestrator status updates and producer UPSERTs both touch indexed
    columns -> never HOT -> ~1500 ms/row UPDATE cost vs ~90 ms baseline.

  * ``commit_ms=2077`` with ``dirty_rows=0`` -- a no-op commit takes 2 s.
    That's pure WAL fsync latency under concurrent write load.

  * ``trader_events`` autovacuum age 5.7 h on a 4.9M-row table.

This migration applies three table-level knobs:

  1. ``trade_signals fillfactor=85``: leaves 15% headroom per page so
     that UPDATEs that *don't* touch indexed columns can be HOT (heap-
     only tuple). Combined with the upsert active-refresh skip-if-equal
     fix (signal_bus.py 2026-05-09), payload-only refreshes from
     producers should now be HOT, eliminating per-update index
     rewrites for that path. Status changes still require non-HOT
     because status is indexed -- structural follow-up below.

  2. ``trade_signals`` aggressive autovacuum: scale_factor=0.05 (was
     default 0.20) so vacuum fires every ~300 dead tuples on the
     6k-row table instead of every ~1200. Keeps page bloat low so
     fillfactor headroom stays available.

  3. ``trader_events`` aggressive autovacuum: scale_factor=0.02 on the
     4.9M-row append-mostly table. Default 0.20 means vacuum waits
     for ~980k dead tuples -- explains the 5.7 h lag.

What this migration does NOT fix (call out for follow-up):

  * WAL fsync latency. ``synchronous_commit`` is currently ``on``
    (default). Each commit waits for fsync to complete: 500-2000 ms
    per the soak. Postgres-config decision: setting
    ``synchronous_commit=local`` (or ``off``) trades worst-case
    durability (lose last few transactions on crash) for ~5x commit
    latency. Worth evaluating against the system's actual recovery
    model.

  * Status updates still non-HOT. ``status`` is in 3 indexes. Real
    fix needs either: (a) drop one redundant index
    (idx_trade_signals_source_status is a prefix of
    idx_trade_signals_source_status_sequence -- queries that use the
    short index will use the long one), (b) make status indexes
    partial (WHERE status IN ('pending','selected','submitted')) so
    terminal-state rows aren't indexed at all, or (c) move status to
    a separate non-indexed column with a background-synced
    materialized status. Defer to a follow-up migration after we
    verify query plans don't regress.

Revision ID: 202605090003
Revises: 202605090002
Create Date: 2026-05-09
"""

from alembic import op


revision = "202605090003"
down_revision = "202605090002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Per-table storage parameters. ``ALTER TABLE ... SET`` is a metadata-
    # only operation; existing rows keep their current page layout but
    # any subsequent UPDATE that has to move a tuple will respect the
    # new fillfactor. Run a manual VACUUM (not VACUUM FULL) afterwards
    # if you want to reclaim immediately; otherwise the next autovacuum
    # cycle picks it up naturally.
    op.execute("ALTER TABLE trade_signals SET (fillfactor = 85)")

    # Aggressive autovacuum on trade_signals. The 6k-row hot table
    # is the orchestrator's commit path; keeping page bloat low is
    # the highest leverage point.
    op.execute(
        "ALTER TABLE trade_signals SET ("
        "autovacuum_vacuum_scale_factor = 0.05, "
        "autovacuum_analyze_scale_factor = 0.02, "
        "autovacuum_vacuum_cost_delay = 0"
        ")"
    )

    # trader_events is append-heavy (4.9M rows, mostly inserts). The
    # default 0.20 scale_factor means autovacuum waits for ~980k dead
    # tuples. The 2026-05-09 soak observed 5.7h since last autovacuum.
    # Lower threshold so it fires more often on a smaller backlog.
    op.execute(
        "ALTER TABLE trader_events SET ("
        "autovacuum_vacuum_scale_factor = 0.02, "
        "autovacuum_analyze_scale_factor = 0.01"
        ")"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE trade_signals RESET (fillfactor)")
    op.execute(
        "ALTER TABLE trade_signals RESET ("
        "autovacuum_vacuum_scale_factor, "
        "autovacuum_analyze_scale_factor, "
        "autovacuum_vacuum_cost_delay"
        ")"
    )
    op.execute(
        "ALTER TABLE trader_events RESET ("
        "autovacuum_vacuum_scale_factor, "
        "autovacuum_analyze_scale_factor"
        ")"
    )
