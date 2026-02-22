"""Add durable tables for live trading state, orders, and position snapshots.

Revision ID: 202602220001
Revises: 202602210003
Create Date: 2026-02-22 00:01:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from alembic_helpers import index_names, table_names


revision = "202602220001"
down_revision = "202602210003"
branch_labels = None
depends_on = None


def _ensure_runtime_indexes() -> None:
    existing_indexes = index_names("live_trading_runtime_state")
    if "idx_live_trading_runtime_wallet" not in existing_indexes:
        op.create_index(
            "idx_live_trading_runtime_wallet",
            "live_trading_runtime_state",
            ["wallet_address"],
            unique=False,
        )
    if "idx_live_trading_runtime_updated" not in existing_indexes:
        op.create_index(
            "idx_live_trading_runtime_updated",
            "live_trading_runtime_state",
            ["updated_at"],
            unique=False,
        )


def _ensure_order_indexes() -> None:
    existing_indexes = index_names("live_trading_orders")
    if "idx_live_trading_orders_wallet_created" not in existing_indexes:
        op.create_index(
            "idx_live_trading_orders_wallet_created",
            "live_trading_orders",
            ["wallet_address", "created_at"],
            unique=False,
        )
    if "idx_live_trading_orders_wallet_status" not in existing_indexes:
        op.create_index(
            "idx_live_trading_orders_wallet_status",
            "live_trading_orders",
            ["wallet_address", "status"],
            unique=False,
        )
    if "idx_live_trading_orders_wallet_clob" not in existing_indexes:
        op.create_index(
            "idx_live_trading_orders_wallet_clob",
            "live_trading_orders",
            ["wallet_address", "clob_order_id"],
            unique=False,
        )


def _ensure_position_indexes() -> None:
    existing_indexes = index_names("live_trading_positions")
    if "idx_live_trading_positions_wallet_market" not in existing_indexes:
        op.create_index(
            "idx_live_trading_positions_wallet_market",
            "live_trading_positions",
            ["wallet_address", "market_id"],
            unique=False,
        )


def upgrade() -> None:
    existing_tables = table_names()

    if "live_trading_runtime_state" not in existing_tables:
        op.create_table(
            "live_trading_runtime_state",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("wallet_address", sa.String(), nullable=False),
            sa.Column("total_trades", sa.Integer(), nullable=False, server_default=sa.text("0")),
            sa.Column("winning_trades", sa.Integer(), nullable=False, server_default=sa.text("0")),
            sa.Column("losing_trades", sa.Integer(), nullable=False, server_default=sa.text("0")),
            sa.Column("total_volume", sa.Float(), nullable=False, server_default=sa.text("0")),
            sa.Column("total_pnl", sa.Float(), nullable=False, server_default=sa.text("0")),
            sa.Column("daily_volume", sa.Float(), nullable=False, server_default=sa.text("0")),
            sa.Column("daily_pnl", sa.Float(), nullable=False, server_default=sa.text("0")),
            sa.Column("open_positions", sa.Integer(), nullable=False, server_default=sa.text("0")),
            sa.Column("last_trade_at", sa.DateTime(), nullable=True),
            sa.Column("daily_volume_reset_at", sa.DateTime(), nullable=True),
            sa.Column("market_positions_json", sa.JSON(), nullable=True),
            sa.Column("balance_signature_type", sa.Integer(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("wallet_address", name="uq_live_trading_runtime_state_wallet"),
        )
    if "live_trading_runtime_state" in table_names():
        _ensure_runtime_indexes()

    if "live_trading_orders" not in existing_tables:
        op.create_table(
            "live_trading_orders",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("wallet_address", sa.String(), nullable=False),
            sa.Column("clob_order_id", sa.String(), nullable=True),
            sa.Column("token_id", sa.String(), nullable=False),
            sa.Column("side", sa.String(), nullable=False),
            sa.Column("price", sa.Float(), nullable=False, server_default=sa.text("0")),
            sa.Column("size", sa.Float(), nullable=False, server_default=sa.text("0")),
            sa.Column("order_type", sa.String(), nullable=False, server_default=sa.text("'GTC'")),
            sa.Column("status", sa.String(), nullable=False, server_default=sa.text("'pending'")),
            sa.Column("filled_size", sa.Float(), nullable=False, server_default=sa.text("0")),
            sa.Column("average_fill_price", sa.Float(), nullable=False, server_default=sa.text("0")),
            sa.Column("market_question", sa.Text(), nullable=True),
            sa.Column("opportunity_id", sa.String(), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
    if "live_trading_orders" in table_names():
        _ensure_order_indexes()

    if "live_trading_positions" not in existing_tables:
        op.create_table(
            "live_trading_positions",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("wallet_address", sa.String(), nullable=False),
            sa.Column("token_id", sa.String(), nullable=False),
            sa.Column("market_id", sa.String(), nullable=False),
            sa.Column("market_question", sa.Text(), nullable=True),
            sa.Column("outcome", sa.String(), nullable=True),
            sa.Column("size", sa.Float(), nullable=False, server_default=sa.text("0")),
            sa.Column("average_cost", sa.Float(), nullable=False, server_default=sa.text("0")),
            sa.Column("current_price", sa.Float(), nullable=False, server_default=sa.text("0")),
            sa.Column("unrealized_pnl", sa.Float(), nullable=False, server_default=sa.text("0")),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "wallet_address",
                "token_id",
                name="uq_live_trading_positions_wallet_token",
            ),
        )
    if "live_trading_positions" in table_names():
        _ensure_position_indexes()


def downgrade() -> None:
    pass
