"""Create the golden-source market settlement store (market_settlements).

The backtester must settle a held position at the binary outcome
($1.00 winner / $0.00 loser), not the last observed mid.  The winner
ground-truth is not present in recorded/imported book data (the
polybacktest importer deliberately excludes it as hindsight), so it lives
here: a per-market resolution record keyed by ``condition_id``, populated
OFFLINE (polybacktest import + the resolution resolver) and read only at
settlement time — so it can never leak look-ahead into a decision.

Records the winning TOKEN id so settlement is a robust token-id equality
check rather than outcome-label matching.

Revision ID: 202606010001
Revises: 202605310001
Create Date: 2026-06-01
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from alembic_helpers import safe_create_index, safe_create_table


revision = "202606010001"
down_revision = "202605310001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    safe_create_table(
        "market_settlements",
        sa.Column("condition_id", sa.String(), primary_key=True, nullable=False),
        sa.Column("slug", sa.String(), nullable=True),
        sa.Column("winning_token_id", sa.String(), nullable=True),
        sa.Column("winning_outcome", sa.String(), nullable=True),
        sa.Column("token_ids_json", sa.JSON(), nullable=True),
        sa.Column("coin_price_start", sa.Float(), nullable=True),
        sa.Column("coin_price_end", sa.Float(), nullable=True),
        sa.Column("resolution_time", sa.DateTime(), nullable=True),
        sa.Column("resolved", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("source", sa.String(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    safe_create_index("ix_market_settlements_slug", "market_settlements", ["slug"])
    safe_create_index(
        "ix_market_settlements_resolution_time",
        "market_settlements",
        ["resolution_time"],
    )


def downgrade() -> None:
    op.drop_index("ix_market_settlements_resolution_time", table_name="market_settlements")
    op.drop_index("ix_market_settlements_slug", table_name="market_settlements")
    op.drop_table("market_settlements")
