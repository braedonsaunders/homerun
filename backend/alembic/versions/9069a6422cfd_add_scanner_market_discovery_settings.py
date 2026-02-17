"""Add scanner market discovery settings

Revision ID: 9069a6422cfd
Revises: 202602170002
Create Date: 2026-02-16 23:28:55.104313

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '9069a6422cfd'
down_revision = '202602170002'
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
        sa.Column("max_events_to_scan", sa.Integer(), nullable=True),
        sa.Column("market_fetch_page_size", sa.Integer(), nullable=True),
        sa.Column("market_fetch_order", sa.String(), nullable=True),
    ]

    for column in additions:
        if column.name not in existing:
            op.add_column(table_name, column)


def downgrade() -> None:
    with op.batch_alter_table('app_settings', schema=None) as batch_op:
        batch_op.drop_column('market_fetch_order')
        batch_op.drop_column('market_fetch_page_size')
        batch_op.drop_column('max_events_to_scan')
