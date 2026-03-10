"""Add composite indexes for trader reconciliation hot paths.

Revision ID: 202603070002
Revises: 202603070001
"""

from alembic import op

from alembic_helpers import index_names


revision = "202603070002"
down_revision = "202603070001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    trader_order_indexes = index_names("trader_orders")
    if "idx_trader_orders_trader_mode_status" not in trader_order_indexes:
        op.create_index(
            "idx_trader_orders_trader_mode_status",
            "trader_orders",
            ["trader_id", "mode", "status"],
            unique=False,
        )

    trader_position_indexes = index_names("trader_positions")
    if "idx_trader_positions_trader_mode_status" not in trader_position_indexes:
        op.create_index(
            "idx_trader_positions_trader_mode_status",
            "trader_positions",
            ["trader_id", "mode", "status"],
            unique=False,
        )


def downgrade() -> None:
    pass
