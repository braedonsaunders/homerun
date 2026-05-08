"""Add market_tags_seen table and market_filter_tags columns to app_settings.

Adds an operator-managed tag whitelist applied at the Polymarket / Kalshi
ingest layer. Markets whose ``(market.tags ∪ event.tags)`` intersection
with the whitelist is empty are dropped before the catalog is written.

The companion ``market_tags_seen`` table records every distinct tag
observed on the raw stream so the operator-facing chooser has data to
render.

Revision ID: 202605070002
Revises: 202605070001
Create Date: 2026-05-07
"""

import sqlalchemy as sa
from alembic import op


revision = "202605070002"
down_revision = "202605070001"
branch_labels = None
depends_on = None


def _table_names() -> set[str]:
    inspector = sa.inspect(op.get_bind())
    return set(inspector.get_table_names())


def _column_names(table_name: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if table_name not in set(inspector.get_table_names()):
        return set()
    return {col["name"] for col in inspector.get_columns(table_name)}


def _index_names(table_name: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if table_name not in set(inspector.get_table_names()):
        return set()
    return {idx["name"] for idx in inspector.get_indexes(table_name)}


def upgrade() -> None:
    if "market_tags_seen" not in _table_names():
        op.create_table(
            "market_tags_seen",
            sa.Column("tag", sa.String(), primary_key=True),
            sa.Column("first_seen", sa.DateTime(timezone=False), nullable=False),
            sa.Column("last_seen", sa.DateTime(timezone=False), nullable=False),
            sa.Column(
                "occurrences",
                sa.BigInteger(),
                nullable=False,
                server_default=sa.text("1"),
            ),
        )

    if "idx_market_tags_seen_last_seen" not in _index_names("market_tags_seen"):
        op.create_index(
            "idx_market_tags_seen_last_seen",
            "market_tags_seen",
            ["last_seen"],
        )

    existing = _column_names("app_settings")
    new_columns = [
        sa.Column("market_filter_tags", sa.JSON(), nullable=True),
        sa.Column("market_filter_updated_at", sa.DateTime(timezone=False), nullable=True),
    ]
    for col in new_columns:
        if col.name not in existing:
            op.add_column("app_settings", col)


def downgrade() -> None:
    pass
