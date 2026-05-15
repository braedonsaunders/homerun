"""Add soak-warning hot path indexes.

Revision ID: 202605150001
Revises: 202605130001
Create Date: 2026-05-15
"""

from __future__ import annotations

from alembic import op

from alembic_helpers import index_names, table_names


revision = "202605150001"
down_revision = "202605130001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if "data_source_records" not in table_names():
        return

    context = op.get_context()

    data_source_record_indexes = index_names("data_source_records")
    if "idx_data_source_records_source_ordering_desc" not in data_source_record_indexes:
        with context.autocommit_block():
            op.execute(
                """
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_data_source_records_source_ordering_desc
                ON data_source_records (
                    data_source_id,
                    (coalesce(observed_at, ingested_at)) DESC,
                    ingested_at DESC,
                    id DESC
                )
                """
            )


def downgrade() -> None:
    pass
