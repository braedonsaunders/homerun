"""Drop the duplicate (strategy_key) index on trader_orders.

pg_indexes shows two IDENTICAL btree indexes on trader_orders(strategy_key):

    ix_trader_orders_strategy_key   (strategy_key)   <- SQLAlchemy-named, owned
                                                        by the model's
                                                        ``index=True`` column arg
    idx_trader_orders_strategy_key  (strategy_key)   <- orphan twin from an old
                                                        hand-written migration

Every INSERT/UPDATE on trader_orders — the hottest money table (orders,
fills, verification updates, PnL writes) — maintains both. The 2026-06-11
soak shows the trading plane under sustained commit pressure; this is pure
write amplification for zero read benefit (Postgres only ever uses one).
Drop the orphan ``idx_`` twin and keep the model-owned ``ix_`` so the ORM
metadata stays consistent. FK-enforcement coverage is unaffected: the
surviving twin indexes the same column.

Revision ID: 202606120001
Revises: 202606010001
Create Date: 2026-06-12
"""

import sqlalchemy as sa
from alembic import op


revision = "202606120001"
down_revision = "202606010001"
branch_labels = None
depends_on = None

_DUPLICATE_INDEX = "idx_trader_orders_strategy_key"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "trader_orders" not in set(inspector.get_table_names()):
        return

    existing = {ix["name"] for ix in inspector.get_indexes("trader_orders")}
    # Only drop the twin while the model-owned ix_ remains, so the column
    # never goes unindexed even if one side was already pruned by hand.
    if _DUPLICATE_INDEX in existing and "ix_trader_orders_strategy_key" in existing:
        op.drop_index(_DUPLICATE_INDEX, table_name="trader_orders")


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "trader_orders" not in set(inspector.get_table_names()):
        return
    existing = {ix["name"] for ix in inspector.get_indexes("trader_orders")}
    if _DUPLICATE_INDEX not in existing:
        op.create_index(_DUPLICATE_INDEX, "trader_orders", ["strategy_key"])
