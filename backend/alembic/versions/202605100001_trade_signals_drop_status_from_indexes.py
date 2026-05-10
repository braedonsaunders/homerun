"""Drop ``status`` from every index on ``trade_signals`` to recover HOT updates.

The 2026-05-09/10 soak captured ``trade_signals.hot_pct`` stuck at
4.8% across 601,573 UPDATEs even AFTER migration 202605090005 had
already pulled ``runtime_sequence`` out of the active-status partial
index.  Looking at the actual indexes in production:

    idx_trade_signals_status_lower         lower(coalesce(status,''))      14,257 scans
    idx_trade_signals_source_status_active (source, status) PARTIAL         4,789 scans
    idx_trade_signals_market_status        (market_id, status) PARTIAL         71 scans

``status`` lives in the indexed key set of all three.  Per Postgres
HOT semantics, an UPDATE that touches *any* indexed column
disqualifies the heap-only-tuple optimization, regardless of
fillfactor headroom.  Every status transition (pending→selected,
selected→submitted, submitted→executed/skipped/expired/failed) hits
all three indexes, so producer status updates ARE the dominant
non-HOT UPDATE source.  The functional ``idx_trade_signals_status_lower``
is the worst offender because, unlike the partial indexes, it is
NOT scoped to active states — it covers ALL 4,068 rows, and so every
status change anywhere in the lifecycle triggers a B-tree update.

This migration removes ``status`` from every index on the table:

  ``idx_trade_signals_status_lower``         → DROP outright.
       Two callsites used it:
         * ``services/trader_orchestrator_state.py:6840``
           ``func.lower(func.coalesce(TradeSignal.status, "")) == "pending"``
         * ``services/trader_orchestrator_state.py:6930``
           ``func.lower(func.coalesce(TradeSignal.status, "")).in_(normalized_statuses)``
       Verified on the live database that every existing
       ``trade_signals.status`` value is already lowercase
       (``executed`` 7, ``expired`` 3883, ``failed`` 4, ``filtered``
       35, ``pending`` 119, ``skipped`` 8 — bool_and(status =
       lower(status)) = true), and the producer paths
       (``set_trade_signal_status`` / model default ``"pending"``)
       only ever write lowercase.  The ``lower()``/``coalesce()``
       wrappers in those queries are defensive remnants and are
       being dropped in the companion code change so the simpler
       ``TradeSignal.status == "pending"`` /
       ``TradeSignal.status.in_(...)`` predicates use the
       PRIMARY-KEY scan + status filter (sub-millisecond on 4 K
       rows), or — for queries also filtered by source/market —
       the surviving partial indexes below.

  ``idx_trade_signals_source_status_active`` (source, status) PARTIAL
       → REPLACED by ``idx_trade_signals_source_active`` (source) PARTIAL,
       same WHERE clause.  4,789 scans / cycle is real query
       traffic, but the (source, status) two-column key is overkill
       — the partial WHERE already restricts to the active set, and
       a status filter on the result is sub-ms over the few hundred
       active rows that match a source.

  ``idx_trade_signals_market_status`` (market_id, status) PARTIAL
       → REPLACED by ``idx_trade_signals_market_active`` (market_id) PARTIAL,
       same WHERE clause.  71 scans is essentially noise — the
       partial-index existence is more about correctness than perf
       — but we keep it to avoid changing query plans that *do*
       depend on (market_id) lookups landing on the active subset.

Net effect: status transitions on any TradeSignal row no longer
touch any index (status is no longer indexed at all), so they
become HOT-eligible.  ``hot_pct`` should rise from the current
4.8% toward 60-80% once status-update volume cycles through.
WAL volume on the busiest table in the system drops proportionally,
which is the headline win — the slow_commit_diagnostic identified
``trade_signals: 601,573 UPDATEs / 4.8% HOT`` as the primary WAL
faucet feeding the 6-8s ``ps_db_commit`` stalls on the trader
cycle.

Companion code change (NOT in this migration, but landed in the
same commit):
  ``services/trader_orchestrator_state.py`` lines 6840 and 6930 —
  drop the ``func.lower(func.coalesce(...))`` wrapper around
  ``TradeSignal.status`` since the column is uniformly lowercase
  by producer contract.

Migration 202604270001 added the ``_lower`` indexes; we are
explicitly reversing that decision because the assumption (mixed-
case data) didn't materialize in practice and the index cost is no
longer free now that we can quantify the HOT% impact.

Revision ID: 202605100001
Revises: 202605090008
Create Date: 2026-05-10
"""

import sqlalchemy as sa
from alembic import op


revision = "202605100001"
down_revision = "202605090008"
branch_labels = None
depends_on = None


_ACTIVE_STATUS_FILTER = "status IN ('pending', 'selected', 'submitted')"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "trade_signals" not in set(inspector.get_table_names()):
        return

    existing = {ix["name"] for ix in inspector.get_indexes("trade_signals")}

    # ---- Drop the functional lower(status) index ---------------------------
    # Heaviest HOT killer; non-partial; touches all 4 K rows on every
    # status transition.  Companion code change drops the lower() wrapper
    # in the two query callsites that exercised it.
    if "idx_trade_signals_status_lower" in existing:
        op.drop_index("idx_trade_signals_status_lower", table_name="trade_signals")

    # ---- (source, status) PARTIAL → (source) PARTIAL -----------------------
    if "idx_trade_signals_source_status_active" in existing:
        op.drop_index(
            "idx_trade_signals_source_status_active", table_name="trade_signals"
        )
    if "idx_trade_signals_source_active" not in existing:
        op.create_index(
            "idx_trade_signals_source_active",
            "trade_signals",
            ["source"],
            postgresql_where=sa.text(_ACTIVE_STATUS_FILTER),
        )

    # ---- (market_id, status) PARTIAL → (market_id) PARTIAL -----------------
    if "idx_trade_signals_market_status" in existing:
        op.drop_index("idx_trade_signals_market_status", table_name="trade_signals")
    if "idx_trade_signals_market_active" not in existing:
        op.create_index(
            "idx_trade_signals_market_active",
            "trade_signals",
            ["market_id"],
            postgresql_where=sa.text(_ACTIVE_STATUS_FILTER),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "trade_signals" not in set(inspector.get_table_names()):
        return

    existing = {ix["name"] for ix in inspector.get_indexes("trade_signals")}

    # Recreate the functional lower(status) index that 202604270001
    # introduced.  Mirrors that migration's CREATE.
    if "idx_trade_signals_status_lower" not in existing:
        op.execute(
            sa.text(
                "CREATE INDEX idx_trade_signals_status_lower "
                "ON trade_signals (lower(coalesce(status, '')))"
            )
        )

    # Restore (market_id, status) PARTIAL.
    if "idx_trade_signals_market_active" in existing:
        op.drop_index("idx_trade_signals_market_active", table_name="trade_signals")
    if "idx_trade_signals_market_status" not in existing:
        op.create_index(
            "idx_trade_signals_market_status",
            "trade_signals",
            ["market_id", "status"],
            postgresql_where=sa.text(_ACTIVE_STATUS_FILTER),
        )

    # Restore (source, status) PARTIAL.
    if "idx_trade_signals_source_active" in existing:
        op.drop_index("idx_trade_signals_source_active", table_name="trade_signals")
    if "idx_trade_signals_source_status_active" not in existing:
        op.create_index(
            "idx_trade_signals_source_status_active",
            "trade_signals",
            ["source", "status"],
            postgresql_where=sa.text(_ACTIVE_STATUS_FILTER),
        )
