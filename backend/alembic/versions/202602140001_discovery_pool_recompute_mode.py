"""Add discovery_pool_recompute_mode to app_settings.

Stores the desired smart-wallet pool recompute mode for tracked-traders worker.

Revision ID: 202602140001
Revises: 202602130012
Create Date: 2026-02-14 12:20:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "202602140001"
down_revision = "202602130012"
branch_labels = None
depends_on = None


def _column_names(table_name: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    table_names = set(inspector.get_table_names())
    if table_name not in table_names:
        return set()
    return {col["name"] for col in inspector.get_columns(table_name)}


def upgrade() -> None:
    table_name = "app_settings"
    existing = _column_names(table_name)

    additions: list[sa.Column] = [
        sa.Column(
            "discovery_pool_recompute_mode",
            sa.String(),
            nullable=True,
            server_default=sa.text("'quality_only'"),
        ),
    ]

    for column in additions:
        if column.name not in existing:
            op.add_column(table_name, column)


def downgrade() -> None:
    # Explicit downgrade support is intentionally omitted for SQLite safety.
    pass

