"""editable latency fallback defaults on app_settings

Revision ID: 202605010001
Revises: 202604300005
Create Date: 2026-05-01

The fill simulator and BacktestStudio fall back to hardcoded
200/600/1500 ms latency quantiles when no real measurements are
available (fresh deployment, paused traders, etc.).  These three
columns let an operator override the fallbacks through the settings
UI without redeploying.  NULL means use the module-level defaults.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from alembic_helpers import safe_add_column


# revision identifiers, used by Alembic.
revision = "202605010001"
down_revision = "202604300005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    safe_add_column("app_settings", sa.Column("latency_fallback_p50_ms", sa.Float(), nullable=True))
    safe_add_column("app_settings", sa.Column("latency_fallback_p95_ms", sa.Float(), nullable=True))
    safe_add_column("app_settings", sa.Column("latency_fallback_p99_ms", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("app_settings", "latency_fallback_p99_ms")
    op.drop_column("app_settings", "latency_fallback_p95_ms")
    op.drop_column("app_settings", "latency_fallback_p50_ms")
