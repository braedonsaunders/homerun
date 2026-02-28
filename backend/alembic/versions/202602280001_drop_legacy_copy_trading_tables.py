"""Drop legacy global copy-trading tables.

Revision ID: 202602280001
Revises: 202602250003
Create Date: 2026-02-28 10:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from alembic_helpers import table_names


revision = "202602280001"
down_revision = "202602250003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    tables = table_names()

    if "copied_trades" in tables:
        op.drop_table("copied_trades")

    if "copy_trading_configs" in tables:
        op.drop_table("copy_trading_configs")

    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(sa.text("DROP TYPE IF EXISTS copytradingmode"))


def downgrade() -> None:
    return
