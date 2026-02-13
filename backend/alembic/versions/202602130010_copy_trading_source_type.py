"""Add source_type column to copy_trading_configs.

Supports pool/tracked_group/individual copy trading modes.

Revision ID: 202602130010
Revises: 202602130009
Create Date: 2026-02-13 18:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "202602130010"
down_revision = "202602130009"
branch_labels = None
depends_on = None


def _column_names(table_name: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    table_names = set(inspector.get_table_names())
    if table_name not in table_names:
        return set()
    return {col["name"] for col in inspector.get_columns(table_name)}


def upgrade() -> None:
    table_name = "copy_trading_configs"
    existing = _column_names(table_name)

    additions: list[sa.Column] = [
        sa.Column(
            "source_type",
            sa.String(),
            nullable=True,
            server_default=sa.text("'individual'"),
        ),
    ]

    for column in additions:
        if column.name not in existing:
            op.add_column(table_name, column)


def downgrade() -> None:
    # Explicit downgrade support is intentionally omitted for SQLite safety.
    pass
