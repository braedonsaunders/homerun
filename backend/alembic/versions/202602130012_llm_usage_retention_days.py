"""Add llm_usage_retention_days column to app_settings.

Controls automatic retention of raw llm_usage_log rows.

Revision ID: 202602130012
Revises: 202602130011
Create Date: 2026-02-14 10:30:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from alembic_helpers import column_names


# revision identifiers, used by Alembic.
revision = "202602130012"
down_revision = "202602130011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    table_name = "app_settings"
    existing = column_names(table_name)

    additions: list[sa.Column] = [
        sa.Column(
            "llm_usage_retention_days",
            sa.Integer(),
            nullable=True,
            server_default=sa.text("30"),
        ),
    ]

    for column in additions:
        if column.name not in existing:
            op.add_column(table_name, column)


def downgrade() -> None:
    # Explicit downgrade support is intentionally omitted for SQLite safety.
    pass
