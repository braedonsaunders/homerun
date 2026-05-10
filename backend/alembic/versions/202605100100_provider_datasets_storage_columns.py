"""Provider datasets storage_type/storage_uri for parquet datasets

Revision ID: 202605100100
Revises: 202605100001
Create Date: 2026-05-10

Backtester gains a parquet replay path (see services/backtest/parquet_replay.py
and services/external_data/parquet_*).  ``provider_datasets`` rows can now
point at on-disk parquet files instead of always projecting through
``market_microstructure_snapshots``:

  * ``storage_type``  ∈ {'postgres', 'parquet'}.  'postgres' (default)
    means snapshots live in mms keyed by provider+token_id (legacy
    polybacktest behaviour).  'parquet' means a file at ``storage_uri``
    is the canonical record; the backtester's ParquetBookReplay reads
    it directly.
  * ``storage_uri`` — fs:// or s3:// URI of the parquet file (for
    parquet rows; null for postgres rows).

Filesystem auto-discovery (services/external_data/parquet_scanner.py)
walks ``HOMERUN_PARQUET_ROOT`` and inserts/updates the provider_datasets
rows; nothing's UPLOADED — the operator just drops files into the root.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision = "202605100100"
down_revision = "202605100001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "provider_datasets",
        sa.Column(
            "storage_type",
            sa.String(),
            nullable=False,
            server_default="postgres",
        ),
    )
    op.add_column(
        "provider_datasets",
        sa.Column("storage_uri", sa.String(), nullable=True),
    )
    # Index on (storage_type, provider) so the backtester's "find
    # parquet datasets covering token X in window W" lookup is cheap.
    op.create_index(
        "ix_provider_datasets_storage_type_provider",
        "provider_datasets",
        ["storage_type", "provider"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_provider_datasets_storage_type_provider",
        table_name="provider_datasets",
    )
    op.drop_column("provider_datasets", "storage_uri")
    op.drop_column("provider_datasets", "storage_type")
