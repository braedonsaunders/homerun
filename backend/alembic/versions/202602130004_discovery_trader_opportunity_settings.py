"""Add persisted Opportunities -> Traders settings fields.

Revision ID: 202602130004
Revises: 202602130003
Create Date: 2026-02-13 02:10:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "202602130004"
down_revision = "202602130003"
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
        sa.Column("discovery_trader_opps_source_filter", sa.String(), nullable=True),
        sa.Column("discovery_trader_opps_min_tier", sa.String(), nullable=True),
        sa.Column("discovery_trader_opps_side_filter", sa.String(), nullable=True),
        sa.Column("discovery_trader_opps_confluence_limit", sa.Integer(), nullable=True),
        sa.Column("discovery_trader_opps_insider_limit", sa.Integer(), nullable=True),
        sa.Column(
            "discovery_trader_opps_insider_min_confidence",
            sa.Float(),
            nullable=True,
        ),
        sa.Column(
            "discovery_trader_opps_insider_max_age_minutes",
            sa.Integer(),
            nullable=True,
        ),
    ]

    for column in additions:
        if column.name not in existing:
            op.add_column(table_name, column)


def downgrade() -> None:
    # Explicit downgrade support is intentionally omitted for SQLite safety.
    pass
