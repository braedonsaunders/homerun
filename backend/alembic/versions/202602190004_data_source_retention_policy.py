"""Add datasource-level retention policy storage.

Revision ID: 202602190004
Revises: 202602190003
Create Date: 2026-02-19 05:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from alembic_helpers import column_names, table_names


revision = "202602190004"
down_revision = "202602190003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if "data_sources" not in table_names():
        return

    existing_columns = set(column_names("data_sources"))
    if "retention" not in existing_columns:
        op.add_column(
            "data_sources",
            sa.Column("retention", sa.JSON(), nullable=True, server_default=sa.text("'{}'::json")),
        )
        existing_columns.add("retention")

    if "retention" in existing_columns:
        bind = op.get_bind()
        bind.execute(sa.text("UPDATE data_sources SET retention = '{}'::json WHERE retention IS NULL"))


def downgrade() -> None:
    return
