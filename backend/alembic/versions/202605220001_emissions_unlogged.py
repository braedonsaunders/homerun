"""Make trade_signal_emissions UNLOGGED to drop it from the WAL path.

``trade_signal_emissions`` is immutable, append-only history of signal
upserts and status transitions — ~1.3M rows/cycle under live load.  It
is the single largest WAL producer in the system, and the 2026-05-22
soak showed it doubling the WAL volume of every intent-runtime
projection commit (the emission INSERT rode the same transaction as the
``trade_signals`` UPDATE), driving 8s+ empty commits and
``Long transaction held origin=intent-runtime-projection`` warnings.

No live trading path reads this table.  Only the offline backtester
(``strategy_backtester``) and execution simulator
(``execution_simulator``) consume it, plus maintenance retention
pruning.  That makes it the same data-class as the firehose telemetry:
loss-tolerant for live trading.

UNLOGGED tables generate NO WAL for inserts/updates — removing this
table from the WAL critical path entirely.  The tradeoff: on an unclean
shutdown / crash, an UNLOGGED table is TRUNCATED during recovery, so
backtests run shortly after a crash will see a gap until history
re-accumulates.  Acceptable given the relaxed durability posture already
in place (``synchronous_commit=off``, Redis persistence off) and the
absence of any live-trading reader.

``ALTER TABLE ... SET UNLOGGED`` rewrites the table once (brief lock);
on the next restart the worker stops double-writing WAL for every
emission.
"""

from alembic import op


revision = "202605220001"
down_revision = "202605200001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE trade_signal_emissions SET UNLOGGED")


def downgrade() -> None:
    op.execute("ALTER TABLE trade_signal_emissions SET LOGGED")
