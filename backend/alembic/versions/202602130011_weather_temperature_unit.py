"""Add weather_workflow_temperature_unit column to app_settings.

Stores user preference for temperature display unit (F or C).

Revision ID: 202602130011
Revises: 202602130010
Create Date: 2026-02-13 22:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from alembic_helpers import column_names


# revision identifiers, used by Alembic.
revision = "202602130011"
down_revision = "202602130010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    table_name = "app_settings"
    existing = column_names(table_name)

    additions: list[sa.Column] = [
        sa.Column(
            "weather_workflow_temperature_unit",
            sa.String(),
            nullable=True,
            server_default=sa.text("'F'"),
        ),
    ]

    for column in additions:
        if column.name not in existing:
            op.add_column(table_name, column)


def downgrade() -> None:
    # Explicit downgrade support is intentionally omitted for migration safety.
    pass
