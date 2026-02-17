"""DB-native opportunity strategies (system seeds + metadata columns).

Revision ID: 202602160002
Revises: 202602160001
Create Date: 2026-02-16 00:00:02.000000
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from alembic import op
import sqlalchemy as sa
from alembic_helpers import column_names, table_names, index_names

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.opportunity_strategy_catalog import build_system_opportunity_strategy_rows  # noqa: E402


# revision identifiers, used by Alembic.
revision = "202602160002"
down_revision = "202602160001"
branch_labels = None
depends_on = None


def _ensure_columns() -> None:
    if "strategy_plugins" not in table_names():
        return

    cols = column_names("strategy_plugins")
    if "source_key" not in cols:
        op.add_column(
            "strategy_plugins",
            sa.Column("source_key", sa.String(), nullable=True, server_default="scanner"),
        )
    if "is_system" not in cols:
        op.add_column(
            "strategy_plugins",
            sa.Column("is_system", sa.Boolean(), nullable=True, server_default=sa.false()),
        )

    bind = op.get_bind()
    bind.execute(sa.text("UPDATE strategy_plugins SET source_key='scanner' WHERE source_key IS NULL OR source_key=''"))
    bind.execute(sa.text("UPDATE strategy_plugins SET is_system=0 WHERE is_system IS NULL"))

    indexes = index_names("strategy_plugins")
    if "idx_strategy_plugin_source_key" not in indexes:
        op.create_index(
            "idx_strategy_plugin_source_key",
            "strategy_plugins",
            ["source_key"],
            unique=False,
        )
    if "idx_strategy_plugin_is_system" not in indexes:
        op.create_index(
            "idx_strategy_plugin_is_system",
            "strategy_plugins",
            ["is_system"],
            unique=False,
        )


def _seed_system_rows() -> None:
    if "strategy_plugins" not in table_names():
        return

    bind = op.get_bind()
    table = sa.table(
        "strategy_plugins",
        sa.column("id", sa.String()),
        sa.column("slug", sa.String()),
        sa.column("source_key", sa.String()),
        sa.column("name", sa.String()),
        sa.column("description", sa.Text()),
        sa.column("source_code", sa.Text()),
        sa.column("class_name", sa.String()),
        sa.column("is_system", sa.Boolean()),
        sa.column("enabled", sa.Boolean()),
        sa.column("status", sa.String()),
        sa.column("error_message", sa.Text()),
        sa.column("config", sa.JSON()),
        sa.column("version", sa.Integer()),
        sa.column("sort_order", sa.Integer()),
        sa.column("created_at", sa.DateTime()),
        sa.column("updated_at", sa.DateTime()),
    )

    existing_slugs = set(bind.execute(sa.text("SELECT slug FROM strategy_plugins")).scalars().all())
    rows = build_system_opportunity_strategy_rows(now=datetime.utcnow())
    missing = [row for row in rows if row["slug"] not in existing_slugs]
    if missing:
        op.bulk_insert(table, missing)


def upgrade() -> None:
    _ensure_columns()
    _seed_system_rows()


def downgrade() -> None:
    if "strategy_plugins" not in table_names():
        return

    bind = op.get_bind()
    slugs = [row["slug"] for row in build_system_opportunity_strategy_rows(now=datetime.utcnow())]
    if slugs:
        strategy_table = sa.table(
            "strategy_plugins",
            sa.column("slug", sa.String()),
        )
        bind.execute(sa.delete(strategy_table).where(strategy_table.c.slug.in_(slugs)))

    indexes = index_names("strategy_plugins")
    if "idx_strategy_plugin_is_system" in indexes:
        op.drop_index("idx_strategy_plugin_is_system", table_name="strategy_plugins")
    if "idx_strategy_plugin_source_key" in indexes:
        op.drop_index("idx_strategy_plugin_source_key", table_name="strategy_plugins")

    cols = column_names("strategy_plugins")
    if "is_system" in cols:
        op.drop_column("strategy_plugins", "is_system")
    if "source_key" in cols:
        op.drop_column("strategy_plugins", "source_key")
