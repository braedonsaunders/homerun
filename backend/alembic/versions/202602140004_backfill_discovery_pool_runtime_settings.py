"""Backfill discovery pool runtime settings columns on stamped databases.

Some local DBs were stamped to 202602140003 while missing the
`discovery_pool_*` runtime columns due to a revision-id reuse incident.
This migration is intentionally idempotent and only adds missing columns.

Revision ID: 202602140004
Revises: 202602140003
Create Date: 2026-02-14 23:59:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "202602140004"
down_revision = "202602140003"
branch_labels = None
depends_on = None


def _column_names(table_name: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    table_names = set(inspector.get_table_names())
    if table_name not in table_names:
        return set()
    return {col["name"] for col in inspector.get_columns(table_name)}


def upgrade() -> None:
    table_name = "app_settings"
    existing = _column_names(table_name)

    additions: list[sa.Column] = [
        sa.Column("discovery_pool_target_size", sa.Integer(), nullable=True, server_default=sa.text("500")),
        sa.Column("discovery_pool_min_size", sa.Integer(), nullable=True, server_default=sa.text("400")),
        sa.Column("discovery_pool_max_size", sa.Integer(), nullable=True, server_default=sa.text("600")),
        sa.Column("discovery_pool_active_window_hours", sa.Integer(), nullable=True, server_default=sa.text("72")),
        sa.Column("discovery_pool_selection_score_floor", sa.Float(), nullable=True, server_default=sa.text("0.55")),
        sa.Column(
            "discovery_pool_max_hourly_replacement_rate",
            sa.Float(),
            nullable=True,
            server_default=sa.text("0.15"),
        ),
        sa.Column(
            "discovery_pool_replacement_score_cutoff",
            sa.Float(),
            nullable=True,
            server_default=sa.text("0.05"),
        ),
        sa.Column("discovery_pool_max_cluster_share", sa.Float(), nullable=True, server_default=sa.text("0.08")),
        sa.Column(
            "discovery_pool_high_conviction_threshold",
            sa.Float(),
            nullable=True,
            server_default=sa.text("0.72"),
        ),
        sa.Column(
            "discovery_pool_insider_priority_threshold",
            sa.Float(),
            nullable=True,
            server_default=sa.text("0.62"),
        ),
        sa.Column("discovery_pool_min_eligible_trades", sa.Integer(), nullable=True, server_default=sa.text("50")),
        sa.Column("discovery_pool_max_eligible_anomaly", sa.Float(), nullable=True, server_default=sa.text("0.5")),
        sa.Column("discovery_pool_core_min_win_rate", sa.Float(), nullable=True, server_default=sa.text("0.6")),
        sa.Column("discovery_pool_core_min_sharpe", sa.Float(), nullable=True, server_default=sa.text("1.0")),
        sa.Column(
            "discovery_pool_core_min_profit_factor",
            sa.Float(),
            nullable=True,
            server_default=sa.text("1.5"),
        ),
        sa.Column("discovery_pool_rising_min_win_rate", sa.Float(), nullable=True, server_default=sa.text("0.55")),
        sa.Column("discovery_pool_slo_min_analyzed_pct", sa.Float(), nullable=True, server_default=sa.text("95.0")),
        sa.Column(
            "discovery_pool_slo_min_profitable_pct",
            sa.Float(),
            nullable=True,
            server_default=sa.text("80.0"),
        ),
        sa.Column(
            "discovery_pool_leaderboard_wallet_trade_sample",
            sa.Integer(),
            nullable=True,
            server_default=sa.text("160"),
        ),
        sa.Column(
            "discovery_pool_incremental_wallet_trade_sample",
            sa.Integer(),
            nullable=True,
            server_default=sa.text("80"),
        ),
        sa.Column(
            "discovery_pool_full_sweep_interval_seconds",
            sa.Integer(),
            nullable=True,
            server_default=sa.text("1800"),
        ),
        sa.Column(
            "discovery_pool_incremental_refresh_interval_seconds",
            sa.Integer(),
            nullable=True,
            server_default=sa.text("120"),
        ),
        sa.Column(
            "discovery_pool_activity_reconciliation_interval_seconds",
            sa.Integer(),
            nullable=True,
            server_default=sa.text("120"),
        ),
        sa.Column(
            "discovery_pool_recompute_interval_seconds",
            sa.Integer(),
            nullable=True,
            server_default=sa.text("60"),
        ),
    ]

    for column in additions:
        if column.name not in existing:
            op.add_column(table_name, column)


def downgrade() -> None:
    # Explicit downgrade support is intentionally omitted for SQLite safety.
    pass

