"""Drop redundant trade_signals index + convert remaining status indexes to partial.

The 2026-05-09 SLOW COMMIT DIAGNOSTIC captures show
``trade_signals.hot_pct = 0.0%`` (121 HOT updates / 557,989 total).
Every UPDATE rebuilds B-tree entries because ``status`` is in three
indexes -- making most UPDATEs non-HOT regardless of fillfactor.

Two structural fixes here:

1. **Drop ``idx_trade_signals_source_status``**. This was the strict
   prefix ``(source, status)`` of ``idx_trade_signals_source_status_sequence``
   ``(source, status, runtime_sequence)``. Postgres can serve any query
   that used the shorter index from the longer one (same prefix). The
   redundant index just added one more B-tree write per UPDATE.

2. **Convert remaining status indexes to PARTIAL** -- only index rows
   where ``status IN ('pending', 'selected', 'submitted')``. Hot-path
   queries (orchestrator's list_unconsumed_trade_signals,
   signal_bus.expire_stale_signals, signal_cache.recent_pending,
   intent_runtime active-status reads) all filter on these statuses
   exclusively. Terminal-state rows (executed / skipped / expired /
   failed) are ~99% of the table; they're queried via
   ``created_at`` (maintenance cleanup) or ``updated_at`` (intent_runtime
   compound query) -- not via (source, status). API filter by terminal
   status (routes_signals.py) is non-hot-path; seq scan acceptable.

   Effect: index size shrinks ~100×, and any UPDATE that transitions a
   row OUT of an active status (e.g. 'selected' -> 'executed') REMOVES
   the entry from the index instead of rewriting it.

Combined with the fillfactor=85 from migration 202605090003 and the
producer skip-if-equal fix in signal_bus.py, this should bring
``ps_db_commit`` per-row cost back toward the 90 ms baseline observed
during the startup burst.

Revision ID: 202605090004
Revises: 202605090003
Create Date: 2026-05-09
"""

import sqlalchemy as sa
from alembic import op


revision = "202605090004"
down_revision = "202605090003"
branch_labels = None
depends_on = None


_ACTIVE_STATUS_FILTER = "status IN ('pending', 'selected', 'submitted')"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "trade_signals" not in set(inspector.get_table_names()):
        # Fresh-DB bootstrap path: Base.metadata.create_all already built
        # the partial indexes from the current ORM model. Nothing to do.
        return

    existing = {ix["name"] for ix in inspector.get_indexes("trade_signals")}

    # (1) Drop the redundant short-prefix index. Idempotent: skip if
    # the create_all path on a fresh DB already omitted it.
    if "idx_trade_signals_source_status" in existing:
        op.drop_index("idx_trade_signals_source_status", table_name="trade_signals")

    # (2) Recreate the source-status-sequence index as partial. We can't
    # change postgresql_where on an existing index, so drop + recreate.
    if "idx_trade_signals_source_status_sequence" in existing:
        op.drop_index(
            "idx_trade_signals_source_status_sequence",
            table_name="trade_signals",
        )
    op.create_index(
        "idx_trade_signals_source_status_sequence",
        "trade_signals",
        ["source", "status", "runtime_sequence"],
        postgresql_where=sa.text(_ACTIVE_STATUS_FILTER),
    )

    # (3) Same for market-status.
    if "idx_trade_signals_market_status" in existing:
        op.drop_index("idx_trade_signals_market_status", table_name="trade_signals")
    op.create_index(
        "idx_trade_signals_market_status",
        "trade_signals",
        ["market_id", "status"],
        postgresql_where=sa.text(_ACTIVE_STATUS_FILTER),
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = {ix["name"] for ix in inspector.get_indexes("trade_signals")}

    if "idx_trade_signals_market_status" in existing:
        op.drop_index("idx_trade_signals_market_status", table_name="trade_signals")
    op.create_index(
        "idx_trade_signals_market_status",
        "trade_signals",
        ["market_id", "status"],
    )

    if "idx_trade_signals_source_status_sequence" in existing:
        op.drop_index(
            "idx_trade_signals_source_status_sequence",
            table_name="trade_signals",
        )
    op.create_index(
        "idx_trade_signals_source_status_sequence",
        "trade_signals",
        ["source", "status", "runtime_sequence"],
    )

    op.create_index(
        "idx_trade_signals_source_status",
        "trade_signals",
        ["source", "status"],
    )
