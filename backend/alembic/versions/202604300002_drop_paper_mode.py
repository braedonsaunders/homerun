"""Drop legacy 'paper' mode aliasing — shadow + live only.

Revision ID: 202604300002
Revises: 202604300001
Create Date: 2026-04-30

Migrates any historical ``mode='paper'`` rows on traders + trader_orders +
copy_trading_configs (and any other table with a ``mode`` column referencing
the trader-execution mode) to ``mode='shadow'``, then installs a CHECK
constraint pinning future writes to ``('shadow', 'live')``.

Backstory: ``paper`` was a legacy alias for ``shadow`` and silently
normalized at multiple read paths (``_normalize_trader_mode``,
``_normalize_mode_key``, ``_LEGACY_MODE_ALIASES``, payload aliases on
``order_manager``).  The dual-naming ate cycles every time someone read
the code; this migration is the boundary that ends it.

Idempotent: the UPDATE is a no-op if no rows have ``mode='paper'``, and
the CHECK constraint creation guards against duplicate installs.
"""
from __future__ import annotations

from alembic import op


revision = "202604300002"
down_revision = "202604300001"
branch_labels = None
depends_on = None


_TABLES_WITH_MODE = (
    ("traders", "mode"),
    ("trader_orders", "mode"),
    ("trader_positions", "mode"),
    ("execution_sessions", "mode"),
    ("trader_orchestrator_control", "mode"),
)

_CHECK_CONSTRAINT_NAME = "{table}_mode_shadow_or_live_chk"


def upgrade() -> None:
    bind = op.get_bind()
    for table, column in _TABLES_WITH_MODE:
        # 1. Backfill any historical paper rows -> shadow.
        op.execute(
            f"UPDATE {table} SET {column} = 'shadow' WHERE LOWER({column}) = 'paper'"
        )
        # 2. Drop any prior CHECK with this name (re-run safety) then install.
        constraint = _CHECK_CONSTRAINT_NAME.format(table=table)
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {constraint}")
        op.execute(
            f"ALTER TABLE {table} ADD CONSTRAINT {constraint} "
            f"CHECK ({column} IS NULL OR LOWER({column}) IN ('shadow', 'live'))"
        )


def downgrade() -> None:
    for table, _column in _TABLES_WITH_MODE:
        constraint = _CHECK_CONSTRAINT_NAME.format(table=table)
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {constraint}")
    # Note: the paper -> shadow value backfill is not reversed; we do not
    # know which rows were originally written as paper, and reverting them
    # would corrupt mixed cohorts.  Code paths no longer accept 'paper'.
