"""Rebrand world-intelligence storage and worker/source keys to events.

Revision ID: 202602180008
Revises: 202602180007
Create Date: 2026-02-18 13:05:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from alembic_helpers import column_names, table_names


# revision identifiers, used by Alembic.
revision = "202602180008"
down_revision = "202602180007"
branch_labels = None
depends_on = None


def _rename_table_if_needed(old_name: str, new_name: str) -> None:
    existing = table_names()
    if old_name in existing and new_name not in existing:
        op.rename_table(old_name, new_name)


def _rewrite_value_if_present(
    table_name: str,
    column_name: str,
    old_value: str,
    new_value: str,
) -> None:
    if table_name not in table_names():
        return
    if column_name not in column_names(table_name):
        return
    bind = op.get_bind()
    table = sa.table(table_name, sa.column(column_name, sa.String()))
    bind.execute(table.update().where(table.c[column_name] == old_value).values({column_name: new_value}))


def _rename_column_if_needed(table_name: str, old_name: str, new_name: str) -> None:
    if table_name not in table_names():
        return
    existing_columns = column_names(table_name)
    if old_name not in existing_columns or new_name in existing_columns:
        return
    op.execute(sa.text(f"ALTER TABLE {table_name} RENAME COLUMN {old_name} TO {new_name}"))


def upgrade() -> None:
    _rename_table_if_needed("world_intelligence_signals", "events_signals")
    _rename_table_if_needed("world_intelligence_snapshots", "events_snapshots")

    _rewrite_value_if_present("worker_control", "worker_name", "world_intelligence", "events")
    _rewrite_value_if_present("worker_snapshot", "worker_name", "world_intelligence", "events")

    _rewrite_value_if_present("strategies", "source_key", "world_intelligence", "events")
    _rewrite_value_if_present("data_sources", "source_key", "world_intelligence", "events")
    _rewrite_value_if_present("execution_sim_runs", "source_key", "world_intelligence", "events")

    settings_column_renames = [
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
    for old_column, new_column in settings_column_renames:
        _rename_column_if_needed("app_settings", old_column, new_column)


def downgrade() -> None:
    return
