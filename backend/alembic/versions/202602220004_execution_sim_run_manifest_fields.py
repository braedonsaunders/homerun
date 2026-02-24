"""Add execution simulation run manifest fields.

Revision ID: 202602220004
Revises: 202602220003
Create Date: 2026-02-22 03:20:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from alembic_helpers import column_names


revision = "202602220004"
down_revision = "202602220003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    existing = column_names("execution_sim_runs")
    if not existing:
        return

    additions: list[sa.Column] = [
        sa.Column("run_seed", sa.String(), nullable=True),
        sa.Column("dataset_hash", sa.String(), nullable=True),
        sa.Column("config_hash", sa.String(), nullable=True),
        sa.Column("code_sha", sa.String(), nullable=True),
    ]
    for col in additions:
        if col.name not in existing:
            op.add_column("execution_sim_runs", col)


def downgrade() -> None:
    pass
