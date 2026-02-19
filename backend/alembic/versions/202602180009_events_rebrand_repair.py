"""Repair events rebrand schema and worker keys.

Revision ID: 202602180009
Revises: 202602180008
Create Date: 2026-02-18 14:15:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from alembic_helpers import column_names, table_names


# revision identifiers, used by Alembic.
revision = "202602180009"
down_revision = "202602180008"
branch_labels = None
depends_on = None


EVENTS_SETTINGS_COLUMNS: list[tuple[str, sa.Column]] = [
    ("events_settings_json", sa.Column("events_settings_json", sa.JSON(), nullable=True)),
    ("events_acled_api_key", sa.Column("events_acled_api_key", sa.String(), nullable=True)),
    ("events_acled_email", sa.Column("events_acled_email", sa.String(), nullable=True)),
    ("events_opensky_username", sa.Column("events_opensky_username", sa.String(), nullable=True)),
    ("events_opensky_password", sa.Column("events_opensky_password", sa.String(), nullable=True)),
    ("events_aisstream_api_key", sa.Column("events_aisstream_api_key", sa.String(), nullable=True)),
    ("events_cloudflare_radar_token", sa.Column("events_cloudflare_radar_token", sa.String(), nullable=True)),
    ("events_country_reference_json", sa.Column("events_country_reference_json", sa.JSON(), nullable=True)),
    ("events_country_reference_source", sa.Column("events_country_reference_source", sa.String(), nullable=True)),
    (
        "events_country_reference_synced_at",
        sa.Column("events_country_reference_synced_at", sa.DateTime(), nullable=True),
    ),
    ("events_ucdp_active_wars_json", sa.Column("events_ucdp_active_wars_json", sa.JSON(), nullable=True)),
    (
        "events_ucdp_minor_conflicts_json",
        sa.Column("events_ucdp_minor_conflicts_json", sa.JSON(), nullable=True),
    ),
    ("events_ucdp_source", sa.Column("events_ucdp_source", sa.String(), nullable=True)),
    ("events_ucdp_year", sa.Column("events_ucdp_year", sa.Integer(), nullable=True)),
    ("events_ucdp_synced_at", sa.Column("events_ucdp_synced_at", sa.DateTime(), nullable=True)),
    ("events_mid_iso3_json", sa.Column("events_mid_iso3_json", sa.JSON(), nullable=True)),
    ("events_mid_source", sa.Column("events_mid_source", sa.String(), nullable=True)),
    ("events_mid_synced_at", sa.Column("events_mid_synced_at", sa.DateTime(), nullable=True)),
    ("events_trade_dependencies_json", sa.Column("events_trade_dependencies_json", sa.JSON(), nullable=True)),
    (
        "events_trade_dependency_source",
        sa.Column("events_trade_dependency_source", sa.String(), nullable=True),
    ),
    ("events_trade_dependency_year", sa.Column("events_trade_dependency_year", sa.Integer(), nullable=True)),
    (
        "events_trade_dependency_synced_at",
        sa.Column("events_trade_dependency_synced_at", sa.DateTime(), nullable=True),
    ),
    ("events_chokepoints_json", sa.Column("events_chokepoints_json", sa.JSON(), nullable=True)),
    ("events_chokepoints_source", sa.Column("events_chokepoints_source", sa.String(), nullable=True)),
    ("events_chokepoints_synced_at", sa.Column("events_chokepoints_synced_at", sa.DateTime(), nullable=True)),
    ("events_gdelt_news_enabled", sa.Column("events_gdelt_news_enabled", sa.Boolean(), nullable=True)),
    ("events_gdelt_news_queries_json", sa.Column("events_gdelt_news_queries_json", sa.JSON(), nullable=True)),
    (
        "events_gdelt_news_timespan_hours",
        sa.Column("events_gdelt_news_timespan_hours", sa.Integer(), nullable=True),
    ),
    ("events_gdelt_news_max_records", sa.Column("events_gdelt_news_max_records", sa.Integer(), nullable=True)),
    ("events_gdelt_news_source", sa.Column("events_gdelt_news_source", sa.String(), nullable=True)),
    ("events_gdelt_news_synced_at", sa.Column("events_gdelt_news_synced_at", sa.DateTime(), nullable=True)),
]


EVENTS_SETTINGS_RENAMES: list[tuple[str, str]] = [
    ("world_intel_settings_json", "events_settings_json"),
    ("world_intel_acled_api_key", "events_acled_api_key"),
    ("world_intel_acled_email", "events_acled_email"),
    ("world_intel_opensky_username", "events_opensky_username"),
    ("world_intel_opensky_password", "events_opensky_password"),
    ("world_intel_aisstream_api_key", "events_aisstream_api_key"),
    ("world_intel_cloudflare_radar_token", "events_cloudflare_radar_token"),
    ("world_intel_country_reference_json", "events_country_reference_json"),
    ("world_intel_country_reference_source", "events_country_reference_source"),
    ("world_intel_country_reference_synced_at", "events_country_reference_synced_at"),
    ("world_intel_ucdp_active_wars_json", "events_ucdp_active_wars_json"),
    ("world_intel_ucdp_minor_conflicts_json", "events_ucdp_minor_conflicts_json"),
    ("world_intel_ucdp_source", "events_ucdp_source"),
    ("world_intel_ucdp_year", "events_ucdp_year"),
    ("world_intel_ucdp_synced_at", "events_ucdp_synced_at"),
    ("world_intel_mid_iso3_json", "events_mid_iso3_json"),
    ("world_intel_mid_source", "events_mid_source"),
    ("world_intel_mid_synced_at", "events_mid_synced_at"),
    ("world_intel_trade_dependencies_json", "events_trade_dependencies_json"),
    ("world_intel_trade_dependency_source", "events_trade_dependency_source"),
    ("world_intel_trade_dependency_year", "events_trade_dependency_year"),
    ("world_intel_trade_dependency_synced_at", "events_trade_dependency_synced_at"),
    ("world_intel_chokepoints_json", "events_chokepoints_json"),
    ("world_intel_chokepoints_source", "events_chokepoints_source"),
    ("world_intel_chokepoints_synced_at", "events_chokepoints_synced_at"),
    ("world_intel_gdelt_news_enabled", "events_gdelt_news_enabled"),
    ("world_intel_gdelt_news_queries_json", "events_gdelt_news_queries_json"),
    ("world_intel_gdelt_news_timespan_hours", "events_gdelt_news_timespan_hours"),
    ("world_intel_gdelt_news_max_records", "events_gdelt_news_max_records"),
    ("world_intel_gdelt_news_source", "events_gdelt_news_source"),
    ("world_intel_gdelt_news_synced_at", "events_gdelt_news_synced_at"),
]


def _rename_table_if_needed(old_name: str, new_name: str) -> None:
    existing = table_names()
    if old_name in existing and new_name not in existing:
        op.rename_table(old_name, new_name)


def _rename_column_if_needed(table_name: str, old_name: str, new_name: str) -> None:
    if table_name not in table_names():
        return
    existing = column_names(table_name)
    if old_name not in existing or new_name in existing:
        return
    op.execute(sa.text(f"ALTER TABLE {table_name} RENAME COLUMN {old_name} TO {new_name}"))


def _ensure_events_columns() -> None:
    if "app_settings" not in table_names():
        return
    existing = column_names("app_settings")
    for col_name, column in EVENTS_SETTINGS_COLUMNS:
        if col_name not in existing:
            op.add_column("app_settings", column)
            existing.add(col_name)


def _normalize_worker_key(table_name: str) -> None:
    if table_name not in table_names():
        return
    if "worker_name" not in column_names(table_name):
        return
    bind = op.get_bind()
    rows = bind.execute(
        sa.text(f"SELECT worker_name FROM {table_name} WHERE worker_name IN ('world_intelligence', 'events')")
    ).fetchall()
    values = {str(row[0]) for row in rows if row and row[0] is not None}
    if "world_intelligence" not in values:
        return
    if "events" in values:
        bind.execute(sa.text(f"DELETE FROM {table_name} WHERE worker_name = 'world_intelligence'"))
        return
    bind.execute(sa.text(f"UPDATE {table_name} SET worker_name = 'events' WHERE worker_name = 'world_intelligence'"))


def upgrade() -> None:
    _rename_table_if_needed("world_intelligence_signals", "events_signals")
    _rename_table_if_needed("world_intelligence_snapshots", "events_snapshots")

    for old_name, new_name in EVENTS_SETTINGS_RENAMES:
        _rename_column_if_needed("app_settings", old_name, new_name)
    _ensure_events_columns()

    _normalize_worker_key("worker_control")
    _normalize_worker_key("worker_snapshot")


def downgrade() -> None:
    return
