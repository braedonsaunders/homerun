"""Drop legacy strategy backup tables.

All strategies are now in the unified `strategies` table.
The legacy tables were kept as backups by migration 202602170004 and are no
longer referenced by any code.

Revision ID: 202602170005
Revises: 202602170004
Create Date: 2026-02-17
"""

from alembic import op
import sqlalchemy as sa

revision = "202602170005"
down_revision = "202602170004"
branch_labels = None
depends_on = None


def _table_exists(name: str) -> bool:
    bind = op.get_bind()
    result = bind.execute(
        sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name=:n"),
        {"n": name},
    )
    return result.scalar() is not None


def upgrade():
    if _table_exists("_legacy_strategy_plugins"):
        op.drop_table("_legacy_strategy_plugins")
    if _table_exists("_legacy_trader_strategy_definitions"):
        op.drop_table("_legacy_trader_strategy_definitions")


def downgrade():
    # Legacy tables cannot be restored — data was migrated into strategies.
    pass
