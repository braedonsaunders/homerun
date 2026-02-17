"""Add tombstones for permanently deleted system opportunity strategies.

Revision ID: 202602170002
Revises: 202602170001
Create Date: 2026-02-17 00:00:02.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from alembic_helpers import table_names, index_names


# revision identifiers, used by Alembic.
revision = "202602170002"
down_revision = "202602170001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if "strategy_plugin_tombstones" not in table_names():
        op.create_table(
            "strategy_plugin_tombstones",
            sa.Column("slug", sa.String(), nullable=False),
            sa.Column("deleted_at", sa.DateTime(), nullable=False),
            sa.Column("reason", sa.String(), nullable=True),
            sa.PrimaryKeyConstraint("slug"),
        )

    indexes = index_names("strategy_plugin_tombstones")
    if "idx_strategy_plugin_tombstones_deleted_at" not in indexes:
        op.create_index(
            "idx_strategy_plugin_tombstones_deleted_at",
            "strategy_plugin_tombstones",
            ["deleted_at"],
            unique=False,
        )


def downgrade() -> None:
    if "strategy_plugin_tombstones" in table_names():
        op.drop_table("strategy_plugin_tombstones")
