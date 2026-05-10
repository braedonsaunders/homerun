"""trader_events.verbosity column for the firehose volume control

Revision ID: 202605010003
Revises: 202605010002
Create Date: 2026-05-01

Adds a nullable ``verbosity`` field to ``trader_events`` so per-strategy
firehose emissions can be filtered by volume tier in the trader Terminal
UI.  Orthogonal to ``severity`` — severity stays info|warn|error and is
used by alert rollups; verbosity is whisper|murmur|voice|shout and is
only meaningful for ``severity='info'`` rows.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from alembic_helpers import safe_add_column, safe_create_index


revision = "202605010003"
down_revision = "202605010002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    safe_add_column(
        "trader_events",
        sa.Column("verbosity", sa.String(), nullable=True),
    )
    safe_create_index(
        "idx_trader_events_verbosity",
        "trader_events",
        ["verbosity"],
    )


def downgrade() -> None:
    op.drop_index("idx_trader_events_verbosity", table_name="trader_events")
    op.drop_column("trader_events", "verbosity")
