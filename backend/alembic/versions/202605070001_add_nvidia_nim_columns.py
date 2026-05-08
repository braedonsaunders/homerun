"""Add nvidia_api_key and nvidia_base_url columns to app_settings.

Revision ID: 202605070001
Revises: 202605060001
Create Date: 2026-05-07
"""

import sqlalchemy as sa
from alembic import op

revision = "202605070001"
down_revision = "202605060001"
branch_labels = None
depends_on = None


def _column_names(table_name: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if table_name not in set(inspector.get_table_names()):
        return set()
    return {col["name"] for col in inspector.get_columns(table_name)}


def upgrade() -> None:
    existing = _column_names("app_settings")
    columns = [
        sa.Column("nvidia_api_key", sa.String(), nullable=True),
        sa.Column("nvidia_base_url", sa.String(), nullable=True),
    ]
    for col in columns:
        if col.name not in existing:
            op.add_column("app_settings", col)


def downgrade() -> None:
    pass
