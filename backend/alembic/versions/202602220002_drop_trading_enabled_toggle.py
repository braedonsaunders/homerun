"""Drop deprecated trading_enabled toggle from app settings.

Revision ID: 202602220002
Revises: 202602220001
Create Date: 2026-02-22 00:45:00.000000
"""

from alembic import op
from alembic_helpers import column_names


revision = "202602220002"
down_revision = "202602220001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    existing = column_names("app_settings")
    if "trading_enabled" in existing:
        op.drop_column("app_settings", "trading_enabled")


def downgrade() -> None:
    pass
