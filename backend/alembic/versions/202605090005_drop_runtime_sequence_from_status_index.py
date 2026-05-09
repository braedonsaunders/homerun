"""Drop runtime_sequence from the active-status partial index to enable HOT updates.

The 2026-05-09 16:55 SLOW COMMIT DIAGNOSTIC -- captured AFTER the
fillfactor=85 + VACUUM FULL + partial-index migration was applied
(202605090003 + 202605090004) -- still showed
``trade_signals.hot_pct = 0.0%`` (121 of 562,538 UPDATEs HOT).
The fillfactor headroom was wasted because every producer UPSERT
(``signal_bus.upsert_trade_signal`` else-branch, line 1642) sets
``runtime_sequence`` in the SET clause, and ``runtime_sequence`` is
the third column of ``idx_trade_signals_source_status_sequence``.
Per postgres HOT semantics, an UPDATE that touches *any* indexed
column disqualifies the heap-only-tuple optimisation regardless of
page-fullness.

This migration drops ``runtime_sequence`` from the indexed key set:

  Before: ``(source, status, runtime_sequence) WHERE status IN active``
  After:  ``(source, status) WHERE status IN active``

Renamed for clarity to ``idx_trade_signals_source_status_active``
(was ``idx_trade_signals_source_status_sequence`` -- which now
misrepresents the columns).

Query impact verification:
  * ``list_unconsumed_trade_signals`` (services/trader_orchestrator_state.py:6739)
    has ``ORDER BY runtime_sequence ... LIMIT 200`` over ~3,800 active
    rows. Postgres can no longer stream-sort via the index; it adds a
    ``Sort`` node above the index scan. Sort cost on 3,800 rows is
    sub-millisecond -- orders of magnitude smaller than the per-UPDATE
    cost we save by enabling HOT.
  * ``upsert_trade_signal`` conditional UPDATE filters by
    ``WHERE id = ? AND (runtime_sequence < ? OR runtime_sequence IS NULL)``.
    The PK index on ``id`` covers the lookup; runtime_sequence is just
    a SARGable filter, not used for index navigation.

Net effect: producer UPSERTs that touch only non-indexed columns
(payload_json, strategy_context_json, edge_percent, entry_price,
runtime_sequence, etc.) now become HOT updates -- no index B-tree
writes at all. ``hot_pct`` should climb to 80%+ within minutes after
this migration applies. Status changes by the orchestrator
(``set_trade_signal_status``) still touch ``status`` and remain
non-HOT, but they're a small minority of UPDATE volume.

Revision ID: 202605090005
Revises: 202605090004
Create Date: 2026-05-09
"""

import sqlalchemy as sa
from alembic import op


revision = "202605090005"
down_revision = "202605090004"
branch_labels = None
depends_on = None


_ACTIVE_STATUS_FILTER = "status IN ('pending', 'selected', 'submitted')"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "trade_signals" not in set(inspector.get_table_names()):
        return

    existing = {ix["name"] for ix in inspector.get_indexes("trade_signals")}

    if "idx_trade_signals_source_status_sequence" in existing:
        op.drop_index(
            "idx_trade_signals_source_status_sequence",
            table_name="trade_signals",
        )

    if "idx_trade_signals_source_status_active" not in existing:
        op.create_index(
            "idx_trade_signals_source_status_active",
            "trade_signals",
            ["source", "status"],
            postgresql_where=sa.text(_ACTIVE_STATUS_FILTER),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = {ix["name"] for ix in inspector.get_indexes("trade_signals")}

    if "idx_trade_signals_source_status_active" in existing:
        op.drop_index(
            "idx_trade_signals_source_status_active",
            table_name="trade_signals",
        )

    if "idx_trade_signals_source_status_sequence" not in existing:
        op.create_index(
            "idx_trade_signals_source_status_sequence",
            "trade_signals",
            ["source", "status", "runtime_sequence"],
            postgresql_where=sa.text(_ACTIVE_STATUS_FILTER),
        )
