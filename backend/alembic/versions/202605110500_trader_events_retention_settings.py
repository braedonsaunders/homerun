"""App settings: trader_events two-tier retention knobs.

Revision ID: 202605110500
Revises: 202605110400
Create Date: 2026-05-11

Adds ``trader_events_firehose_retention_days`` and
``trader_events_other_retention_days`` to ``app_settings`` so the
MaintenanceService's full_cleanup can prune firehose rows on a
shorter cadence than audit/decision rows.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic_helpers import safe_add_column


revision = "202605110500"
down_revision = "202605110400"
branch_labels = None
depends_on = None


def upgrade() -> None:
    safe_add_column(
        "app_settings",
        sa.Column(
            "trader_events_firehose_retention_days",
            sa.Integer(),
            nullable=True,
            server_default=sa.text("7"),
        ),
    )
    safe_add_column(
        "app_settings",
        sa.Column(
            "trader_events_other_retention_days",
            sa.Integer(),
            nullable=True,
            server_default=sa.text("90"),
        ),
    )


def downgrade() -> None:
    pass
