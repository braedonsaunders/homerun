"""Strategy-scoped autoresearch experiments.

Revision ID: 202604260003
Revises: 202604260002
Create Date: 2026-04-26

Code experiments belong on the backtest data plane and don't need a
bot context — they evolve a strategy's source code and are kept on the
Strategy record itself. Make ``trader_id`` nullable on
``autoresearch_experiments`` and add an index on ``strategy_id`` so the
UI can list / stream experiments without joining through traders.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from alembic_helpers import safe_create_index


revision = "202604260003"
down_revision = "202604260002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "autoresearch_experiments",
        "trader_id",
        existing_type=sa.String(),
        nullable=True,
    )
    safe_create_index(
        "idx_arx_strategy_id",
        "autoresearch_experiments",
        ["strategy_id"],
    )


def downgrade() -> None:
    op.drop_index("idx_arx_strategy_id", table_name="autoresearch_experiments")
    # Repopulating trader_id for any rows that lost it is out of scope
    # for this downgrade — only safe to revert when no strategy-scoped
    # experiments exist.
    op.alter_column(
        "autoresearch_experiments",
        "trader_id",
        existing_type=sa.String(),
        nullable=False,
    )
