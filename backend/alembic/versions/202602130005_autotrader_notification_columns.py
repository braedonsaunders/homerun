"""Add autotrader notification settings columns.

Revision ID: 202602130005
Revises: 202602130004
Create Date: 2026-02-13 12:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "202602130005"
down_revision = "202602130004"
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
            "notify_autotrader_orders",
            sa.Boolean(),
            nullable=True,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "notify_autotrader_issues",
            sa.Boolean(),
            nullable=True,
            server_default=sa.text("1"),
        ),
        sa.Column(
            "notify_autotrader_timeline",
            sa.Boolean(),
            nullable=True,
            server_default=sa.text("1"),
        ),
        sa.Column(
            "notify_autotrader_summary_interval_minutes",
            sa.Integer(),
            nullable=True,
            server_default=sa.text("60"),
        ),
        sa.Column(
            "notify_autotrader_summary_per_trader",
            sa.Boolean(),
            nullable=True,
            server_default=sa.text("0"),
        ),
    ]

    for column in additions:
        if column.name not in existing:
            op.add_column(table_name, column)


def downgrade() -> None:
    # Explicit downgrade support is intentionally omitted for SQLite safety.
    pass
