"""create trader_order_verification side table + backfill

Revision ID: 202604290001
Revises: 202604280002
Create Date: 2026-04-29

Splits the verification-related columns off TraderOrder onto a 1-to-1
child table so the Polymarket verifier and the orchestrator no longer
share row-level locks on trader_orders.

Today both writers UPDATE the same trader_orders row:
  * verifier sets verification_status / verification_source / verification_reason
    / verification_tx_hash / verified_at / actual_profit
  * orchestrator/lifecycle sets status / payload_json / updated_at

PG row-locks are at the row level, so even though they touch disjoint
columns, the two writers serialize on the same TransactionID lock
(captured in the 24h log as ``LOCK CONTENTION ... UPDATE trader_orders
SET status=...verification_status=...``).

This migration is STEP 1 of a staged cutover:

  * step 1 (this commit): create table + backfill.  No application
    code reads or writes from the new table yet — schema is purely
    additive, safe to deploy + roll back.
  * step 2: route verifier writes to the new table.  Readers remain on
    the old columns; the new table is the source of truth for
    new verifier writes only.
  * step 3: switch readers to LEFT JOIN against the new table,
    falling back to old columns when missing.
  * step 4: drop the old verification columns from trader_orders.

Until step 4, both the original columns and the new table coexist.
"""
from __future__ import annotations

from alembic import op
from alembic_helpers import safe_create_table, safe_create_index
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "202604290001"
down_revision = "202604280002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    safe_create_table(
        "trader_order_verification",
        sa.Column(
            "trader_order_id",
            sa.String(),
            sa.ForeignKey("trader_orders.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("verification_status", sa.String(), nullable=False, server_default="local"),
        sa.Column("verification_source", sa.String(), nullable=True),
        sa.Column("verification_reason", sa.Text(), nullable=True),
        sa.Column("verification_tx_hash", sa.String(), nullable=True),
        sa.Column("verified_at", sa.DateTime(), nullable=True),
        sa.Column("actual_profit", sa.Float(), nullable=True),
        sa.Column("execution_wallet_address", sa.String(), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    safe_create_index(
        "idx_trader_order_verification_status",
        "trader_order_verification",
        ["verification_status"],
    )
    safe_create_index(
        "idx_trader_order_verification_tx_hash",
        "trader_order_verification",
        ["verification_tx_hash"],
    )
    safe_create_index(
        "idx_trader_order_verification_wallet",
        "trader_order_verification",
        ["execution_wallet_address"],
    )
    # Partial index mirroring idx_trader_orders_wallet_realized_pnl on
    # the original table — supports the live P&L SUM(actual_profit)
    # aggregations once readers are cut over.
    op.execute(
        "CREATE INDEX IF NOT EXISTS "
        "idx_trader_order_verification_wallet_realized_pnl "
        "ON trader_order_verification (execution_wallet_address) "
        "WHERE actual_profit IS NOT NULL"
    )

    # Backfill from existing trader_orders rows.  This copies the
    # current verification state so the new table is initially in sync.
    # Use a single bulk INSERT for speed (millions of rows possible on
    # a long-lived install — this app is at ~10k rows in production
    # right now, well within a single statement).
    op.execute(
        """
        INSERT INTO trader_order_verification (
            trader_order_id,
            verification_status,
            verification_source,
            verification_reason,
            verification_tx_hash,
            verified_at,
            actual_profit,
            execution_wallet_address,
            updated_at
        )
        SELECT
            id,
            COALESCE(verification_status, 'local'),
            verification_source,
            verification_reason,
            verification_tx_hash,
            verified_at,
            actual_profit,
            execution_wallet_address,
            COALESCE(updated_at, NOW())
        FROM trader_orders
        ON CONFLICT (trader_order_id) DO NOTHING
        """
    )


def downgrade() -> None:
    op.drop_index("idx_trader_order_verification_wallet_realized_pnl", table_name="trader_order_verification")
    op.drop_index("idx_trader_order_verification_wallet", table_name="trader_order_verification")
    op.drop_index("idx_trader_order_verification_tx_hash", table_name="trader_order_verification")
    op.drop_index("idx_trader_order_verification_status", table_name="trader_order_verification")
    op.drop_table("trader_order_verification")
