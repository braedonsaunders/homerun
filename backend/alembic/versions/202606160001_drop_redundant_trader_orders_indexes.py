"""Drop three redundant/unused indexes on trader_orders.

trader_orders is the hottest money table (orders, fills, verification
updates, PnL writes). The 2026-06-15/16 soak shows it as the dominant
WAL/write-amplification source (~543 MB WAL over ~51 K updates, 23
indexes maintained per row write). Three of those indexes carry zero
read benefit and are pure write tax:

  idx_trader_orders_strategy_version
      Duplicate of the model-owned ``ix_trader_orders_strategy_version``
      (created by the ``strategy_version`` column's ``index=True``).
      pg_stat_user_indexes: 0 scans. Postgres only ever uses one twin.

  idx_trader_orders_created  (created_at)
      Prefix-redundant with ``idx_trader_orders_created_at_id``
      (created_at, id) which serves the same ORDER BY created_at access
      path plus keyset pagination. pg_stat: 84 scans vs the composite's
      hot usage — the planner already prefers the composite.

  idx_trader_orders_trader_created  (trader_id, created_at)
      0 scans. Redundant with ``idx_trader_orders_trader_mode_status``
      (trader_id, mode, status) for trader-scoped lookups, which IS hot
      (620 K scans).

None of the three back a foreign key (only ``decision_id`` is an FK on
this table, covered by its own index), so dropping them does not regress
FK-enforcement scans. The model-owned ``ix_`` twin for strategy_version
survives, so that column never goes unindexed. The two model-defined
indexes (idx_trader_orders_created, idx_trader_orders_trader_created)
are removed from the ORM ``__table_args__`` in the same change so that
create_all() == migration-chain (test_alembic_roundtrip invariant).

Revision ID: 202606160001
Revises: 202606150002
Create Date: 2026-06-16
"""

import sqlalchemy as sa
from alembic import op


revision = "202606160001"
down_revision = "202606150002"
branch_labels = None
depends_on = None

# Indexes that are simply removed (no surviving twin needed — the access
# path is covered by another, hotter composite index).
_REMOVED = ("idx_trader_orders_created", "idx_trader_orders_trader_created")

# Duplicate twin: only drop while the model-owned ix_ survives, so the
# column never goes unindexed even if one side was already pruned by hand.
_DUP = "idx_trader_orders_strategy_version"
_DUP_TWIN = "ix_trader_orders_strategy_version"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "trader_orders" not in set(inspector.get_table_names()):
        return
    existing = {ix["name"] for ix in inspector.get_indexes("trader_orders")}

    for name in _REMOVED:
        if name in existing:
            op.drop_index(name, table_name="trader_orders")

    if _DUP in existing and _DUP_TWIN in existing:
        op.drop_index(_DUP, table_name="trader_orders")


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "trader_orders" not in set(inspector.get_table_names()):
        return
    existing = {ix["name"] for ix in inspector.get_indexes("trader_orders")}

    if "idx_trader_orders_created" not in existing:
        op.create_index(
            "idx_trader_orders_created", "trader_orders", ["created_at"]
        )
    if "idx_trader_orders_trader_created" not in existing:
        op.create_index(
            "idx_trader_orders_trader_created",
            "trader_orders",
            ["trader_id", "created_at"],
        )
    if _DUP not in existing:
        op.create_index(_DUP, "trader_orders", ["strategy_version"])
