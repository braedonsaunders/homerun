"""Add missing search filter columns for strategy settings.

Revision ID: 202602130003
Revises: 202602130002
Create Date: 2026-02-13 01:15:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from alembic_helpers import column_names


# revision identifiers, used by Alembic.
revision = "202602130003"
down_revision = "202602130002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    table_name = "app_settings"
    existing = column_names(table_name)

    additions: list[sa.Column] = [
        sa.Column("min_liquidity_per_leg", sa.Float(), nullable=True),
        sa.Column("btc_eth_hf_maker_mode", sa.Boolean(), nullable=True),
    ]

    for column in additions:
        if column.name not in existing:
            op.add_column(table_name, column)


def downgrade() -> None:
    # Explicit downgrade support is intentionally omitted for migration safety.
    pass
