"""Add local UI lock settings to app_settings.

Revision ID: 202602210001
Revises: 202602200001
"""

from alembic import op
import sqlalchemy as sa

from alembic_helpers import column_names


revision = "202602210001"
down_revision = "202602200001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    existing = column_names("app_settings")
    columns = [
        sa.Column("ui_lock_enabled", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("ui_lock_password_hash", sa.String(), nullable=True),
        sa.Column("ui_lock_idle_timeout_minutes", sa.Integer(), nullable=False, server_default=sa.text("15")),
    ]
    for col in columns:
        if col.name not in existing:
            op.add_column("app_settings", col)


def downgrade() -> None:
    pass
