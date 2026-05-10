"""Drop standalone runtime_sequence + expires_at indexes on trade_signals.

The 17:03 verification cycle (after migration 202605090005 dropped
runtime_sequence from the active-status composite index) STILL showed
``hot_pct=0.02`` -- 14 new UPDATEs since baseline, zero new HOT.

Diagnosis: missed two ``index=True`` decorators on the column
definitions in models/database.py:

    runtime_sequence = Column(BigInteger, nullable=True, index=True)
    expires_at = Column(DateTime, nullable=True, index=True)

These create ``ix_trade_signals_runtime_sequence`` and
``ix_trade_signals_expires_at`` -- standalone B-trees on those single
columns. Producer UPSERTs touch BOTH columns on every emit (sequence
is bumped, expires_at TTL is refreshed), so HOT remains impossible
even with the composite index fix.

Both indexes are dead weight at the query layer:

* ``runtime_sequence`` is read via the PK lookup in
  ``upsert_trade_signal`` (``WHERE id = ? AND runtime_sequence < ?``)
  -- no index navigation needed, just a SARGable filter post-PK fetch.
  ORDER BY runtime_sequence in ``list_unconsumed_trade_signals`` is
  served by the Sort node above the (source, status) partial scan
  (sub-ms over ~3,800 active rows -- see migration 202605090005).

* ``expires_at`` is used only as an additional WHERE filter
  (``expires_at IS NULL OR expires_at >= now``) inside queries that
  already use (source, status) or (market_id, status) for primary
  access. The standalone index is never the planner's chosen path
  because the queries don't have an indexable expires_at predicate
  on its own.

Dropping both lets producer UPSERTs that touch only non-indexed
columns become HOT updates. Status changes by the orchestrator still
touch ``status`` (in the partial composite indexes) and remain
non-HOT, but they're a small minority of UPDATE volume.

Revision ID: 202605090006
Revises: 202605090005
Create Date: 2026-05-09
"""

import sqlalchemy as sa
from alembic import op


revision = "202605090006"
down_revision = "202605090005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "trade_signals" not in set(inspector.get_table_names()):
        return

    existing = {ix["name"] for ix in inspector.get_indexes("trade_signals")}

    # Single-column auto-named indexes from index=True. Names follow
    # SQLAlchemy's convention: ``ix_<table>_<column>``.
    for name in ("ix_trade_signals_runtime_sequence", "ix_trade_signals_expires_at"):
        if name in existing:
            op.drop_index(name, table_name="trade_signals")


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = {ix["name"] for ix in inspector.get_indexes("trade_signals")}

    if "ix_trade_signals_runtime_sequence" not in existing:
        op.create_index(
            "ix_trade_signals_runtime_sequence",
            "trade_signals",
            ["runtime_sequence"],
        )
    if "ix_trade_signals_expires_at" not in existing:
        op.create_index(
            "ix_trade_signals_expires_at",
            "trade_signals",
            ["expires_at"],
        )
