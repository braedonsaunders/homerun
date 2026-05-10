"""Drop idx_trade_signals_runtime_sequence (correct name; previous migration used wrong name).

Migration 202605090006 attempted to drop ``ix_trade_signals_runtime_sequence``
(the SQLAlchemy auto-naming convention ``ix_<table>_<col>``), but the
actual index in production was named ``idx_trade_signals_runtime_sequence``
(the project's preferred ``idx_<table>_<col>`` convention). The DROP
silently no-op'd because the name didn't match. pg_indexes verification
at 17:05 still showed the index present, and ``hot_pct`` remained at
0.02% with only 3 of 182 new UPDATEs becoming HOT.

This migration drops the actually-named index. Same rationale as
202605090006: ``runtime_sequence`` is bumped on every producer UPSERT,
and any indexed-column UPDATE disqualifies HOT regardless of fillfactor
headroom. The standalone single-column index serves no query path that
the PK index + the partial composite index don't already cover.

Revision ID: 202605090007
Revises: 202605090006
Create Date: 2026-05-09
"""

import sqlalchemy as sa
from alembic import op


revision = "202605090007"
down_revision = "202605090006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "trade_signals" not in set(inspector.get_table_names()):
        return

    existing = {ix["name"] for ix in inspector.get_indexes("trade_signals")}

    # Both naming conventions seen in the wild: ``idx_`` (project
    # convention via explicit Index() in __table_args__ — historical)
    # and ``ix_`` (SQLAlchemy auto-naming via ``index=True`` on Column).
    # Drop whichever survives.
    for name in ("idx_trade_signals_runtime_sequence",
                 "ix_trade_signals_runtime_sequence"):
        if name in existing:
            op.drop_index(name, table_name="trade_signals")


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = {ix["name"] for ix in inspector.get_indexes("trade_signals")}

    if "idx_trade_signals_runtime_sequence" not in existing:
        op.create_index(
            "idx_trade_signals_runtime_sequence",
            "trade_signals",
            ["runtime_sequence"],
        )
