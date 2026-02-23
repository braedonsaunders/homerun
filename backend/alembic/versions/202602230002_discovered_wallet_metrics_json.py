"""Add metrics_json column to discovered_wallets for timing skill and execution quality.

Revision ID: 202602230002
Revises: 202602230001
Create Date: 2026-02-23 00:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from alembic_helpers import column_names


revision = "202602230002"
down_revision = "202602230001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    existing = column_names("discovered_wallets")
    if "metrics_json" not in existing:
        op.add_column("discovered_wallets", sa.Column("metrics_json", sa.JSON(), nullable=True))


def downgrade() -> None:
    pass
