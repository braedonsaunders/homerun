"""Add scanner opportunity cap settings columns.

Revision ID: 202602180004
Revises: 202602180003
Create Date: 2026-02-18
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from alembic_helpers import column_names


revision = "202602180004"
down_revision = "202602180003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    table_name = "app_settings"
    existing = column_names(table_name)

    additions: list[sa.Column] = [
        sa.Column(
            "scanner_max_opportunities_total",
            sa.Integer(),
            nullable=True,
            server_default=sa.text("500"),
        ),
        sa.Column(
            "scanner_max_opportunities_per_strategy",
            sa.Integer(),
            nullable=True,
            server_default=sa.text("120"),
        ),
    ]

    for column in additions:
        if column.name not in existing:
            op.add_column(table_name, column)


def downgrade() -> None:
    pass
