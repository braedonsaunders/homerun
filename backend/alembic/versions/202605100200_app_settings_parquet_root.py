"""App settings parquet_root_override for UI-editable parquet storage path.

Revision ID: 202605100200
Revises: 202605100100
Create Date: 2026-05-10

Operators were forced to set ``HOMERUN_PARQUET_ROOT`` in the env file
to point the backtester's parquet ingestion at a directory.  This
migration adds a ``parquet_root_override`` column to ``app_settings``
so the path is editable from Data Lab → Providers → Parquet without
restarting the backend.

Resolution order (now):
  1. ``app_settings.parquet_root_override`` (UI-set, persists)
  2. ``HOMERUN_PARQUET_ROOT`` env var (legacy / fallback)
  3. ``<repo>/data/parquet`` (default for fresh installs)
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from alembic_helpers import safe_add_column


# revision identifiers, used by Alembic.
revision = "202605100200"
down_revision = "202605100100"
branch_labels = None
depends_on = None


def upgrade() -> None:
    safe_add_column(
        "app_settings",
        sa.Column("parquet_root_override", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("app_settings", "parquet_root_override")
