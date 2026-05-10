"""Per-bot block_new_orders flag.

Revision ID: 202604260002
Revises: 202604260001
Create Date: 2026-04-26

Adds a per-trader analogue of the global kill switch on
trader_orchestrator_control. When True, the trader keeps running
(reconciliation, exits, manage-only mode) but the entry-decision path
short-circuits before opening any new positions.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from alembic_helpers import safe_add_column


revision = "202604260002"
down_revision = "202604260001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    safe_add_column(
        "traders",
        sa.Column(
            "block_new_orders",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )


def downgrade() -> None:
    op.drop_column("traders", "block_new_orders")
