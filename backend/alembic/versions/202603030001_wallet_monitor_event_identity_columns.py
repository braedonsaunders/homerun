"""Add wallet monitor event identity columns for per-log replay/dedupe.

Revision ID: 202603030001
Revises: 202602280008
Create Date: 2026-03-03
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from alembic_helpers import column_names, index_names, table_names


revision = "202603030001"
down_revision = "202602280008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    table_name = "wallet_monitor_events"
    if table_name not in table_names():
        return

    existing_columns = column_names(table_name)
    if "order_hash" not in existing_columns:
        op.add_column(table_name, sa.Column("order_hash", sa.String(), nullable=True))
    if "log_index" not in existing_columns:
        op.add_column(table_name, sa.Column("log_index", sa.Integer(), nullable=True))

    existing_indexes = index_names(table_name)
    if "idx_wme_tx_hash" not in existing_indexes:
        op.create_index("idx_wme_tx_hash", table_name, ["tx_hash"], unique=False)
    if "idx_wme_order_hash" not in existing_indexes:
        op.create_index("idx_wme_order_hash", table_name, ["order_hash"], unique=False)
    if "idx_wme_tx_log" not in existing_indexes:
        op.create_index("idx_wme_tx_log", table_name, ["tx_hash", "log_index"], unique=False)


def downgrade() -> None:
    return
