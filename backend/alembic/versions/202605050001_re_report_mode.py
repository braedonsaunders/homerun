"""Add report_mode to strategy_reverse_engineer_jobs

Revision ID: 202605050001
Revises: 202605040001
Create Date: 2026-05-05

Adds the ``report_mode`` column that selects between the analytical
report deliverable ('report') and the legacy LLM-synthesized strategy
seed deliverable ('strategy_seed').

Default is 'report' for new jobs since that's the higher-value
deliverable for most operators.  Existing rows are backfilled to
'strategy_seed' to preserve their original mode.

Idempotent: probes information_schema before adding the column so the
migration succeeds even if a previous run created the column directly
via ALTER TABLE (e.g. during dev iteration).  Without this guard the
migration would crash on the second invocation and the launcher would
restart-loop forever.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "202605050001"
down_revision = "202605040001"
branch_labels = None
depends_on = None


def _column_exists(bind, table: str, column: str) -> bool:
    inspector = sa.inspect(bind)
    try:
        cols = inspector.get_columns(table)
    except Exception:
        return False
    return any(c.get("name") == column for c in cols)


def upgrade() -> None:
    bind = op.get_bind()
    if not _column_exists(bind, "strategy_reverse_engineer_jobs", "report_mode"):
        op.add_column(
            "strategy_reverse_engineer_jobs",
            sa.Column(
                "report_mode",
                sa.String(),
                nullable=False,
                server_default="strategy_seed",
            ),
        )
    # Always re-assert the new-row default — cheap and idempotent.
    # New jobs default to 'report'; existing rows keep whatever value
    # they have (usually 'strategy_seed' from the backfill above, or
    # 'report' if added directly via ALTER TABLE in dev).
    op.alter_column(
        "strategy_reverse_engineer_jobs",
        "report_mode",
        server_default="report",
    )


def downgrade() -> None:
    bind = op.get_bind()
    if _column_exists(bind, "strategy_reverse_engineer_jobs", "report_mode"):
        op.drop_column("strategy_reverse_engineer_jobs", "report_mode")
