"""Add market_catalog table for persisted upstream market/event catalog.

Revision ID: 202602190007
Revises: 202602190006
Create Date: 2026-02-19 20:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from alembic_helpers import table_names


revision = "202602190007"
down_revision = "202602190006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    existing = table_names()
    if "market_catalog" not in existing:
        op.create_table(
            "market_catalog",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.Column("events_json", sa.JSON(), server_default=sa.text("'[]'::json")),
            sa.Column("markets_json", sa.JSON(), server_default=sa.text("'[]'::json")),
            sa.Column("event_count", sa.Integer(), server_default="0"),
            sa.Column("market_count", sa.Integer(), server_default="0"),
            sa.Column("fetch_duration_seconds", sa.Float(), nullable=True),
            sa.Column("error", sa.Text(), nullable=True),
        )


def downgrade() -> None:
    pass
