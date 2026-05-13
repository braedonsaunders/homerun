"""Add indexes for log-observed read hot paths.

Revision ID: 202605130001
Revises: 202605110500
Create Date: 2026-05-13
"""
from __future__ import annotations

from alembic_helpers import safe_create_index


revision = "202605130001"
down_revision = "202605110500"
branch_labels = None
depends_on = None


def upgrade() -> None:
    safe_create_index(
        "idx_position_account_status",
        "simulation_positions",
        ["account_id", "status"],
        unique=False,
    )
    safe_create_index(
        "idx_anomaly_detected_at",
        "detected_anomalies",
        ["detected_at"],
        unique=False,
    )
    safe_create_index(
        "idx_data_source_records_source_ordering",
        "data_source_records",
        ["data_source_id", "observed_at", "ingested_at", "id"],
        unique=False,
    )


def downgrade() -> None:
    pass
