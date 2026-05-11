"""App settings: replace parquet_root_override String with parquet_root_overrides JSON array.

Revision ID: 202605110200
Revises: 202605110100
Create Date: 2026-05-11

The previous schema (202605100200) added a single ``parquet_root_override``
String column.  Operators wanted to point the parquet ingest at MULTIPLE
directories (e.g. one for vendor A's data, one for vendor B's, one
shared mount), which a single string can't represent.

This migration:
  1. Adds ``parquet_root_overrides`` JSON column (a list of absolute
     paths the scanner will walk in order).
  2. Backfills it from the old ``parquet_root_override`` value if
     non-null (becomes a single-element list).
  3. Drops the old ``parquet_root_override`` column.

The HOMERUN_PARQUET_ROOT environment variable is also no longer
consulted by ``parquet_root()`` — see services/external_data/parquet_schema.py.
The DB-backed list is the single source of truth.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from alembic_helpers import safe_add_column
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "202605110200"
down_revision = "202605110100"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Add the new JSON column.  Use safe_add_column because the two
    # dynamic-sync migrations (202602130009, 202603190004) may have
    # already added this column when running from a fresh DB against the
    # current ORM model, which only carries parquet_root_overrides.
    safe_add_column(
        "app_settings",
        sa.Column(
            "parquet_root_overrides",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    # 2. Backfill: convert any existing single-string value into a
    #    one-element JSON array.  Use NULLIF + jsonb_build_array so
    #    a null/empty old value leaves the new column as null.
    op.execute(
        sa.text(
            """
            UPDATE app_settings
            SET parquet_root_overrides = jsonb_build_array(parquet_root_override)
            WHERE parquet_root_override IS NOT NULL
              AND length(trim(parquet_root_override)) > 0
            """
        )
    )
    # 3. Drop the old single-value column.
    op.drop_column("app_settings", "parquet_root_override")


def downgrade() -> None:
    # Reverse: re-add the String column, populate from the first
    # element of the JSON array (operator loses additional roots),
    # drop the JSON column.
    op.add_column(
        "app_settings",
        sa.Column("parquet_root_override", sa.String(), nullable=True),
    )
    op.execute(
        sa.text(
            """
            UPDATE app_settings
            SET parquet_root_override = parquet_root_overrides->>0
            WHERE parquet_root_overrides IS NOT NULL
              AND jsonb_array_length(parquet_root_overrides) > 0
            """
        )
    )
    op.drop_column("app_settings", "parquet_root_overrides")
