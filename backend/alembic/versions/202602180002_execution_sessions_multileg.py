"""Add persistent multi-leg execution session tables.

Revision ID: 202602180002
Revises: 202602180001
Create Date: 2026-02-18 00:00:01.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from alembic_helpers import index_names, table_names


revision = "202602180002"
down_revision = "202602180001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    existing_tables = table_names()

    if "execution_sessions" not in existing_tables:
        op.create_table(
            "execution_sessions",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("trader_id", sa.String(), nullable=False),
            sa.Column("signal_id", sa.String(), nullable=True),
            sa.Column("decision_id", sa.String(), nullable=True),
            sa.Column("source", sa.String(), nullable=False),
            sa.Column("strategy_key", sa.String(), nullable=True),
            sa.Column("mode", sa.String(), nullable=False, server_default=sa.text("'paper'")),
            sa.Column("status", sa.String(), nullable=False, server_default=sa.text("'pending'")),
            sa.Column("policy", sa.String(), nullable=True),
            sa.Column("plan_id", sa.String(), nullable=True),
            sa.Column("market_ids_json", sa.JSON(), nullable=True),
            sa.Column("legs_total", sa.Integer(), nullable=False, server_default=sa.text("0")),
            sa.Column("legs_completed", sa.Integer(), nullable=False, server_default=sa.text("0")),
            sa.Column("legs_failed", sa.Integer(), nullable=False, server_default=sa.text("0")),
            sa.Column("legs_open", sa.Integer(), nullable=False, server_default=sa.text("0")),
            sa.Column("requested_notional_usd", sa.Float(), nullable=True),
            sa.Column("executed_notional_usd", sa.Float(), nullable=False, server_default=sa.text("0")),
            sa.Column("max_unhedged_notional_usd", sa.Float(), nullable=False, server_default=sa.text("0")),
            sa.Column("unhedged_notional_usd", sa.Float(), nullable=False, server_default=sa.text("0")),
            sa.Column("trace_id", sa.String(), nullable=True),
            sa.Column("started_at", sa.DateTime(), nullable=True),
            sa.Column("completed_at", sa.DateTime(), nullable=True),
            sa.Column("expires_at", sa.DateTime(), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("payload_json", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(["trader_id"], ["traders.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["signal_id"], ["trade_signals.id"], ondelete="SET NULL"),
            sa.ForeignKeyConstraint(["decision_id"], ["trader_decisions.id"], ondelete="SET NULL"),
        )

    execution_session_indexes = index_names("execution_sessions")
    if "idx_execution_sessions_created" not in execution_session_indexes:
        op.create_index("idx_execution_sessions_created", "execution_sessions", ["created_at"], unique=False)
    if "idx_execution_sessions_status" not in execution_session_indexes:
        op.create_index("idx_execution_sessions_status", "execution_sessions", ["status"], unique=False)
    if "idx_execution_sessions_trader_status" not in execution_session_indexes:
        op.create_index(
            "idx_execution_sessions_trader_status",
            "execution_sessions",
            ["trader_id", "status"],
            unique=False,
        )
    if "ix_execution_sessions_trader_id" not in execution_session_indexes:
        op.create_index("ix_execution_sessions_trader_id", "execution_sessions", ["trader_id"], unique=False)
    if "ix_execution_sessions_signal_id" not in execution_session_indexes:
        op.create_index("ix_execution_sessions_signal_id", "execution_sessions", ["signal_id"], unique=False)
    if "ix_execution_sessions_decision_id" not in execution_session_indexes:
        op.create_index("ix_execution_sessions_decision_id", "execution_sessions", ["decision_id"], unique=False)
    if "ix_execution_sessions_source" not in execution_session_indexes:
        op.create_index("ix_execution_sessions_source", "execution_sessions", ["source"], unique=False)
    if "ix_execution_sessions_strategy_key" not in execution_session_indexes:
        op.create_index("ix_execution_sessions_strategy_key", "execution_sessions", ["strategy_key"], unique=False)
    if "ix_execution_sessions_plan_id" not in execution_session_indexes:
        op.create_index("ix_execution_sessions_plan_id", "execution_sessions", ["plan_id"], unique=False)
    if "ix_execution_sessions_trace_id" not in execution_session_indexes:
        op.create_index("ix_execution_sessions_trace_id", "execution_sessions", ["trace_id"], unique=False)

    if "execution_session_legs" not in existing_tables:
        op.create_table(
            "execution_session_legs",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("session_id", sa.String(), nullable=False),
            sa.Column("leg_index", sa.Integer(), nullable=False),
            sa.Column("leg_id", sa.String(), nullable=False),
            sa.Column("market_id", sa.String(), nullable=False),
            sa.Column("market_question", sa.Text(), nullable=True),
            sa.Column("token_id", sa.String(), nullable=True),
            sa.Column("side", sa.String(), nullable=False, server_default=sa.text("'buy'")),
            sa.Column("outcome", sa.String(), nullable=True),
            sa.Column("price_policy", sa.String(), nullable=False, server_default=sa.text("'maker_limit'")),
            sa.Column("time_in_force", sa.String(), nullable=False, server_default=sa.text("'GTC'")),
            sa.Column("target_price", sa.Float(), nullable=True),
            sa.Column("requested_notional_usd", sa.Float(), nullable=True),
            sa.Column("requested_shares", sa.Float(), nullable=True),
            sa.Column("filled_notional_usd", sa.Float(), nullable=False, server_default=sa.text("0")),
            sa.Column("filled_shares", sa.Float(), nullable=False, server_default=sa.text("0")),
            sa.Column("avg_fill_price", sa.Float(), nullable=True),
            sa.Column("status", sa.String(), nullable=False, server_default=sa.text("'pending'")),
            sa.Column("last_error", sa.Text(), nullable=True),
            sa.Column("metadata_json", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(["session_id"], ["execution_sessions.id"], ondelete="CASCADE"),
            sa.UniqueConstraint("session_id", "leg_index", name="uq_execution_session_leg_index"),
        )

    execution_leg_indexes = index_names("execution_session_legs")
    if "idx_execution_session_legs_session_status" not in execution_leg_indexes:
        op.create_index(
            "idx_execution_session_legs_session_status",
            "execution_session_legs",
            ["session_id", "status"],
            unique=False,
        )
    if "ix_execution_session_legs_session_id" not in execution_leg_indexes:
        op.create_index("ix_execution_session_legs_session_id", "execution_session_legs", ["session_id"], unique=False)
    if "ix_execution_session_legs_market_id" not in execution_leg_indexes:
        op.create_index("ix_execution_session_legs_market_id", "execution_session_legs", ["market_id"], unique=False)

    if "execution_session_orders" not in existing_tables:
        op.create_table(
            "execution_session_orders",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("session_id", sa.String(), nullable=False),
            sa.Column("leg_id", sa.String(), nullable=False),
            sa.Column("trader_order_id", sa.String(), nullable=True),
            sa.Column("provider_order_id", sa.String(), nullable=True),
            sa.Column("provider_clob_order_id", sa.String(), nullable=True),
            sa.Column("action", sa.String(), nullable=False, server_default=sa.text("'submit'")),
            sa.Column("side", sa.String(), nullable=False),
            sa.Column("price", sa.Float(), nullable=True),
            sa.Column("size", sa.Float(), nullable=True),
            sa.Column("notional_usd", sa.Float(), nullable=True),
            sa.Column("status", sa.String(), nullable=False, server_default=sa.text("'submitted'")),
            sa.Column("reason", sa.Text(), nullable=True),
            sa.Column("payload_json", sa.JSON(), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(["session_id"], ["execution_sessions.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["leg_id"], ["execution_session_legs.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["trader_order_id"], ["trader_orders.id"], ondelete="SET NULL"),
        )

    execution_order_indexes = index_names("execution_session_orders")
    if "idx_execution_session_orders_session_created" not in execution_order_indexes:
        op.create_index(
            "idx_execution_session_orders_session_created",
            "execution_session_orders",
            ["session_id", "created_at"],
            unique=False,
        )
    if "idx_execution_session_orders_leg_created" not in execution_order_indexes:
        op.create_index(
            "idx_execution_session_orders_leg_created",
            "execution_session_orders",
            ["leg_id", "created_at"],
            unique=False,
        )
    if "ix_execution_session_orders_session_id" not in execution_order_indexes:
        op.create_index(
            "ix_execution_session_orders_session_id", "execution_session_orders", ["session_id"], unique=False
        )
    if "ix_execution_session_orders_leg_id" not in execution_order_indexes:
        op.create_index("ix_execution_session_orders_leg_id", "execution_session_orders", ["leg_id"], unique=False)
    if "ix_execution_session_orders_trader_order_id" not in execution_order_indexes:
        op.create_index(
            "ix_execution_session_orders_trader_order_id",
            "execution_session_orders",
            ["trader_order_id"],
            unique=False,
        )
    if "ix_execution_session_orders_provider_order_id" not in execution_order_indexes:
        op.create_index(
            "ix_execution_session_orders_provider_order_id",
            "execution_session_orders",
            ["provider_order_id"],
            unique=False,
        )
    if "ix_execution_session_orders_provider_clob_order_id" not in execution_order_indexes:
        op.create_index(
            "ix_execution_session_orders_provider_clob_order_id",
            "execution_session_orders",
            ["provider_clob_order_id"],
            unique=False,
        )

    if "execution_session_events" not in existing_tables:
        op.create_table(
            "execution_session_events",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("session_id", sa.String(), nullable=False),
            sa.Column("leg_id", sa.String(), nullable=True),
            sa.Column("event_type", sa.String(), nullable=False),
            sa.Column("severity", sa.String(), nullable=False, server_default=sa.text("'info'")),
            sa.Column("message", sa.Text(), nullable=True),
            sa.Column("payload_json", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["session_id"], ["execution_sessions.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["leg_id"], ["execution_session_legs.id"], ondelete="SET NULL"),
        )

    execution_event_indexes = index_names("execution_session_events")
    if "idx_execution_session_events_session_created" not in execution_event_indexes:
        op.create_index(
            "idx_execution_session_events_session_created",
            "execution_session_events",
            ["session_id", "created_at"],
            unique=False,
        )
    if "ix_execution_session_events_session_id" not in execution_event_indexes:
        op.create_index(
            "ix_execution_session_events_session_id", "execution_session_events", ["session_id"], unique=False
        )
    if "ix_execution_session_events_leg_id" not in execution_event_indexes:
        op.create_index("ix_execution_session_events_leg_id", "execution_session_events", ["leg_id"], unique=False)
    if "ix_execution_session_events_event_type" not in execution_event_indexes:
        op.create_index(
            "ix_execution_session_events_event_type",
            "execution_session_events",
            ["event_type"],
            unique=False,
        )
    if "ix_execution_session_events_created_at" not in execution_event_indexes:
        op.create_index(
            "ix_execution_session_events_created_at",
            "execution_session_events",
            ["created_at"],
            unique=False,
        )


def downgrade() -> None:
    pass
