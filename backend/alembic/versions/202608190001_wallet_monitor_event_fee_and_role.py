"""Persist the on-chain taker fee (and book side) on wallet_monitor_events.

``_parse_order_filled_log`` has always decoded the ``fee`` word out of the
CLOB V2 ``OrderFilled`` event, but nothing stored it — the value was
computed and dropped on every fill.

That number matters because it is the only *ground truth* fee available
anywhere in the pipeline:

  * Polymarket's REST ``/trades`` payload carries no fee field at all
    (verified 2026-08-19: proxyWallet/side/asset/conditionId/size/price/
    timestamp/title/slug/outcome/name/transactionHash and nothing else),
    and no maker-vs-taker flag either.
  * Everything else in the tree is a *model* of the published fee schedule
    (``utils.kelly.polymarket_taker_fee``), which is an estimate keyed on
    price and market category.

``role`` is stored alongside because ``fee_usd`` is uninterpretable without
it: only takers are charged a fee, and only makers accrue rebates (a
percentage of collected taker fees), so a fee figure with no side-of-book
attached cannot be attributed to either.

Both columns are nullable — historical rows predate the capture and stay
NULL rather than being back-filled with a misleading 0.0, which would be
indistinguishable from a genuine zero-fee (geopolitics) fill.

Idempotent + boot-safe: columns are added only when absent, matching the
rest of the chain, which must survive ``init_database`` retry loops.

Revision ID: 202608190001
Revises: 202606160003
Create Date: 2026-08-19
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from alembic_helpers import column_names, safe_add_column


revision = "202608190001"
down_revision = "202606160003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    safe_add_column("wallet_monitor_events", sa.Column("fee_usd", sa.Float(), nullable=True))
    safe_add_column("wallet_monitor_events", sa.Column("role", sa.String(), nullable=True))


def downgrade() -> None:
    existing = column_names("wallet_monitor_events")
    if "role" in existing:
        op.drop_column("wallet_monitor_events", "role")
    if "fee_usd" in existing:
        op.drop_column("wallet_monitor_events", "fee_usd")
