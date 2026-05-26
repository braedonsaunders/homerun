"""Add retention-window settings for previously-unbounded high-volume tables.

The 2026-05-26 disk-saturation incident (C: drive at 2.3 GB free -> WAL
fsync stalls -> 5-6s commits on empty transactions) was triggered by the
host disk filling.  Root cause of the fill: several high-volume tables had
NO DELETE retention and grew without bound:

  * market_microstructure_snapshots  (real per-tick L2 snapshots)
  * book_delta_events                 (already UNLOGGED; still on disk)
  * wallet_monitor_events             (smart-money wallet activity)
  * trader_decision_checks            (per-decision gate audit, 4.2M rows)
  * trader_decisions                  (decision records; pruned only when
                                       not referenced by orders/sessions)
  * opportunity_history               (opportunity snapshots)

This migration adds the AppSettings columns that drive the new
``MaintenanceService`` sweeps for those tables (wired into the daily
``full_cleanup``).  Defaults are conservative and operator-overridable via
the Settings UI / API.  ``0`` disables a given sweep.

Idempotent + boot-safe: each column is added only if absent (cheap
information_schema read; no locks on missing columns), mirroring the rest
of the alembic chain which must survive ``init_database`` retry loops.
"""

import sqlalchemy as sa
from alembic import op


revision = "202605260001"
down_revision = "202605230001"
branch_labels = None
depends_on = None


_COLUMNS = {
    "cleanup_market_microstructure_days": 7,
    "cleanup_book_delta_events_days": 7,
    "cleanup_wallet_monitor_events_days": 14,
    "cleanup_trader_decision_checks_days": 14,
    "cleanup_trader_decisions_days": 30,
    "cleanup_opportunity_history_days": 30,
}


def _existing_columns(bind) -> set[str]:
    rows = bind.execute(
        sa.text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'app_settings'"
        )
    ).scalars().all()
    return {str(r) for r in rows}


def upgrade() -> None:
    bind = op.get_bind()
    have = _existing_columns(bind)
    for name, default in _COLUMNS.items():
        if name in have:
            continue
        op.add_column(
            "app_settings",
            sa.Column(name, sa.Integer(), nullable=True, server_default=str(default)),
        )


def downgrade() -> None:
    bind = op.get_bind()
    have = _existing_columns(bind)
    for name in _COLUMNS:
        if name in have:
            op.drop_column("app_settings", name)
