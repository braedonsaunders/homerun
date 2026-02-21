"""Merge strategy_plugins + trader_strategy_definitions into unified strategies table.

Revision ID: 202602170004
Revises: 202602170003
Create Date: 2026-02-17
"""

from alembic import op
import sqlalchemy as sa

revision = "202602170004"
down_revision = "202602170003"
branch_labels = None
depends_on = None


def _table_names() -> set[str]:
    inspector = sa.inspect(op.get_bind())
    return set(inspector.get_table_names())


def _index_names(table_name: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if table_name not in set(inspector.get_table_names()):
        return set()
    return {idx["name"] for idx in inspector.get_indexes(table_name)}


def upgrade():
    tables = _table_names()

    # 1. Create the new unified strategies table
    if "strategies" not in tables:
        op.create_table(
            "strategies",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("slug", sa.String(), unique=True, nullable=False),
            sa.Column("source_key", sa.String(), nullable=False, server_default="scanner"),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("source_code", sa.Text(), nullable=False),
            sa.Column("class_name", sa.String(), nullable=True),
            sa.Column("is_system", sa.Boolean(), server_default=sa.false()),
            sa.Column("enabled", sa.Boolean(), server_default=sa.true()),
            sa.Column("status", sa.String(), server_default="unloaded"),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("config", sa.JSON(), server_default=sa.text("'{}'::json")),
            sa.Column("config_schema", sa.JSON(), server_default=sa.text("'{}'::json")),
            sa.Column("aliases", sa.JSON(), server_default=sa.text("'[]'::json")),
            sa.Column("version", sa.Integer(), server_default="1"),
            sa.Column("sort_order", sa.Integer(), server_default="0"),
            sa.Column("created_at", sa.DateTime()),
            sa.Column("updated_at", sa.DateTime()),
        )
        tables = _table_names()

    strategy_indexes = _index_names("strategies")
    if "idx_strategy_slug" not in strategy_indexes:
        op.create_index("idx_strategy_slug", "strategies", ["slug"])
    if "idx_strategy_source_key" not in strategy_indexes:
        op.create_index("idx_strategy_source_key", "strategies", ["source_key"])
    if "idx_strategy_enabled" not in strategy_indexes:
        op.create_index("idx_strategy_enabled", "strategies", ["enabled"])
    if "idx_strategy_is_system" not in strategy_indexes:
        op.create_index("idx_strategy_is_system", "strategies", ["is_system"])
    if "idx_strategy_status" not in strategy_indexes:
        op.create_index("idx_strategy_status", "strategies", ["status"])

    # 2. Migrate data from strategy_plugins
    if "strategy_plugins" in tables:
        op.execute(
            """
            INSERT INTO strategies (id, slug, source_key, name, description, source_code,
                                    class_name, is_system, enabled, status, error_message,
                                    config, config_schema, aliases, version, sort_order,
                                    created_at, updated_at)
            SELECT id, slug, source_key, name, description, source_code,
                   class_name, is_system, enabled, status, error_message,
                   config, '{}', '[]', version, sort_order,
                   created_at, updated_at
            FROM strategy_plugins
            ON CONFLICT (slug) DO NOTHING
            """
        )

    # 3. Migrate data from trader_strategy_definitions (skip duplicates by slug)
    if "trader_strategy_definitions" in tables:
        op.execute(
            """
            INSERT INTO strategies (id, slug, source_key, name, description, source_code,
                                    class_name, is_system, enabled, status, error_message,
                                    config, config_schema, aliases, version, sort_order,
                                    created_at, updated_at)
            SELECT id, strategy_key, source_key, label, description, source_code,
                   class_name, is_system, enabled, status, error_message,
                   default_params_json, param_schema_json, aliases_json, version, 0,
                   created_at, updated_at
            FROM trader_strategy_definitions
            ON CONFLICT (slug) DO NOTHING
            """
        )

    # 4. Rename tombstone table for consistency
    tables = _table_names()
    if "strategy_plugin_tombstones" in tables and "strategy_tombstones" not in tables:
        op.rename_table("strategy_plugin_tombstones", "strategy_tombstones")

    # 5. Rename old tables (keep as backup, don't drop yet)
    tables = _table_names()
    if "strategy_plugins" in tables and "_legacy_strategy_plugins" not in tables:
        op.rename_table("strategy_plugins", "_legacy_strategy_plugins")
    tables = _table_names()
    if "trader_strategy_definitions" in tables and "_legacy_trader_strategy_definitions" not in tables:
        op.rename_table("trader_strategy_definitions", "_legacy_trader_strategy_definitions")


def downgrade():
    tables = _table_names()
    if "_legacy_strategy_plugins" in tables and "strategy_plugins" not in tables:
        op.rename_table("_legacy_strategy_plugins", "strategy_plugins")
    tables = _table_names()
    if "_legacy_trader_strategy_definitions" in tables and "trader_strategy_definitions" not in tables:
        op.rename_table("_legacy_trader_strategy_definitions", "trader_strategy_definitions")
    tables = _table_names()
    if "strategy_tombstones" in tables and "strategy_plugin_tombstones" not in tables:
        op.rename_table("strategy_tombstones", "strategy_plugin_tombstones")
    if "strategies" in _table_names():
        op.drop_table("strategies")
