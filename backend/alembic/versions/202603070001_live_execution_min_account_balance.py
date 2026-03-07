"""Add min_account_balance_usd live execution setting.

Revision ID: 202603070001
Revises: 202603060001
"""

from alembic import op
import sqlalchemy as sa

from alembic_helpers import column_names


revision = "202603070001"
down_revision = "202603060001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    existing = column_names("app_settings")
    if "min_account_balance_usd" not in existing:
        op.add_column(
            "app_settings",
            sa.Column("min_account_balance_usd", sa.Float(), nullable=False, server_default=sa.text("0")),
        )


def downgrade() -> None:
    pass
