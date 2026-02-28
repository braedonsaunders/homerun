"""Drop legacy trader and weather workflow compatibility columns.

Revision ID: 202602280002
Revises: 202602280001
Create Date: 2026-02-28 16:00:00.000000
"""

from __future__ import annotations

from alembic import op
from alembic_helpers import column_names, table_names


revision = "202602280002"
down_revision = "202602280001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    tables = table_names()

    if "traders" in tables:
        existing = column_names("traders")
        for column_name in (
            "strategy_key",
            "strategy_version",
            "sources_json",
            "params_json",
        ):
            if column_name in existing:
                op.drop_column("traders", column_name)

    if "app_settings" in tables:
        existing = column_names("app_settings")
        for column_name in (
            "weather_workflow_orchestrator_enabled",
            "weather_workflow_orchestrator_min_edge",
            "weather_workflow_orchestrator_max_age_minutes",
        ):
            if column_name in existing:
                op.drop_column("app_settings", column_name)


def downgrade() -> None:
    pass
