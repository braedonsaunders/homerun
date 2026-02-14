"""Add trader_positions inventory table.

Revision ID: 202602140003
Revises: 202602140002
Create Date: 2026-02-14 23:25:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "202602140003"
down_revision = "202602140002"
branch_labels = None
depends_on = None


def _table_names() -> set[str]:
    inspector = sa.inspect(op.get_bind())
    return set(inspector.get_table_names())


def _index_names(table_name: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if table_name not in set(inspector.get_table_names()):
        return set()
    return {idx.get("name") for idx in inspector.get_indexes(table_name) if idx.get("name")}


def _unique_constraint_names(table_name: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if table_name not in set(inspector.get_table_names()):
        return set()
    return {
        constraint.get("name")
        for constraint in inspector.get_unique_constraints(table_name)
        if constraint.get("name")
    }


def upgrade() -> None:
    table_name = "trader_positions"
    tables = _table_names()

    if table_name not in tables:
        op.create_table(
            table_name,
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("trader_id", sa.String(), nullable=False),
            sa.Column("mode", sa.String(), nullable=False, server_default=sa.text("'paper'")),
            sa.Column("market_id", sa.String(), nullable=False),
            sa.Column("market_question", sa.Text(), nullable=True),
            sa.Column("direction", sa.String(), nullable=True),
            sa.Column("status", sa.String(), nullable=False, server_default=sa.text("'open'")),
            sa.Column("open_order_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
            sa.Column("total_notional_usd", sa.Float(), nullable=False, server_default=sa.text("0")),
            sa.Column("avg_entry_price", sa.Float(), nullable=True),
            sa.Column("first_order_at", sa.DateTime(), nullable=True),
            sa.Column("last_order_at", sa.DateTime(), nullable=True),
            sa.Column("closed_at", sa.DateTime(), nullable=True),
            sa.Column("payload_json", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(["trader_id"], ["traders.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "trader_id",
                "mode",
                "market_id",
                "direction",
                name="uq_trader_position_identity",
            ),
        )

    index_names = _index_names(table_name)
    unique_names = _unique_constraint_names(table_name)
    if "uq_trader_position_identity" not in unique_names and "uq_trader_position_identity" not in index_names:
        op.create_index(
            "uq_trader_position_identity",
            table_name,
            ["trader_id", "mode", "market_id", "direction"],
            unique=True,
        )

    if "idx_trader_positions_status" not in index_names:
        op.create_index("idx_trader_positions_status", table_name, ["status"])
    if "idx_trader_positions_trader_status" not in index_names:
        op.create_index("idx_trader_positions_trader_status", table_name, ["trader_id", "status"])
    if "ix_trader_positions_trader_id" not in index_names:
        op.create_index("ix_trader_positions_trader_id", table_name, ["trader_id"])
    if "ix_trader_positions_market_id" not in index_names:
        op.create_index("ix_trader_positions_market_id", table_name, ["market_id"])


def downgrade() -> None:
    # Explicit downgrade support is intentionally omitted for SQLite safety.
    pass
