"""Add pool inactive rising retention setting to app_settings.

Revision ID: 202602180010
Revises: 202602180009
Create Date: 2026-02-18 16:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from alembic_helpers import column_names


revision = "202602180010"
down_revision = "202602180009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    table_name = "app_settings"
    existing = column_names(table_name)

    new_column = sa.Column(
        "discovery_pool_inactive_rising_retention_hours",
        sa.Integer(),
        nullable=True,
        server_default=sa.text("336"),
    )

    if new_column.name not in existing:
        op.add_column(table_name, new_column)


def downgrade() -> None:
    # Explicit downgrade support is intentionally omitted for migration safety.
    return
