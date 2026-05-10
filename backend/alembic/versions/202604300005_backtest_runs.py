"""create backtest_runs table

Revision ID: 202604300005
Revises: 202604300004
Create Date: 2026-04-30

Persisted Backtest Studio runs.  Replaces the prior process-local
LRU so runs survive worker restarts.

Summary columns are denormalized for the run-history list query
(sort by started_at, render the row sparkline + return %) while
the full augmented result blob lives in ``result_json`` for the
per-run detail view.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from alembic_helpers import safe_create_table, safe_create_index


revision = "202604300005"
down_revision = "202604300004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    safe_create_table(
        "backtest_runs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("strategy_slug", sa.String(), nullable=True),
        sa.Column("strategy_name", sa.String(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("total_time_ms", sa.Float(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(), nullable=False, server_default="ok"),
        sa.Column("trade_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_return_pct", sa.Float(), nullable=False, server_default="0"),
        sa.Column("sparkline_pct_json", sa.JSON(), nullable=True),
        sa.Column("result_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )
    safe_create_index(
        "idx_btr_strategy_started",
        "backtest_runs",
        ["strategy_slug", "started_at"],
    )
    safe_create_index("idx_btr_started", "backtest_runs", ["started_at"])
    safe_create_index("idx_btr_status", "backtest_runs", ["status"])


def downgrade() -> None:
    op.drop_index("idx_btr_status", table_name="backtest_runs")
    op.drop_index("idx_btr_started", table_name="backtest_runs")
    op.drop_index("idx_btr_strategy_started", table_name="backtest_runs")
    op.drop_table("backtest_runs")
