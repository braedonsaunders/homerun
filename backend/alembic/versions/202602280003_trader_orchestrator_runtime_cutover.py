"""Strip removed trader orchestrator runtime compatibility keys.

Revision ID: 202602280003
Revises: 202602280002
Create Date: 2026-02-28
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "202602280003"
down_revision = "202602280002"
branch_labels = None
depends_on = None


def _column_names(table_name: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if table_name not in set(inspector.get_table_names()):
        return set()
    return {col["name"] for col in inspector.get_columns(table_name)}


def upgrade() -> None:
    columns = _column_names("trader_orchestrator_control")
    if "settings_json" not in columns:
        return

    op.execute(
        sa.text(
            """
            UPDATE trader_orchestrator_control
            SET settings_json = (
                (
                    COALESCE(settings_json, '{}'::json)::jsonb
                    - 'enable_live_market_context'
                    - 'live_market_history_window_seconds'
                    - 'live_market_history_fidelity_seconds'
                    - 'live_market_history_max_points'
                    - 'live_market_context_timeout_seconds'
                    - 'trader_cycle_timeout_seconds'
                )::json
            )
            WHERE settings_json IS NOT NULL
            """
        )
    )


def downgrade() -> None:
    pass
