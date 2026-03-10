"""Add composite indexes for data source record hot paths.

Revision ID: 202603090001
Revises: 202603070002
"""

from alembic import op

from alembic_helpers import index_names


revision = "202603090001"
down_revision = "202603070002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    record_indexes = index_names("data_source_records")

    if "idx_data_source_records_source_ordering" not in record_indexes:
        op.create_index(
            "idx_data_source_records_source_ordering",
            "data_source_records",
            ["data_source_id", "observed_at", "ingested_at", "id"],
            unique=False,
        )

    if "idx_data_source_records_slug_ingested" not in record_indexes:
        op.create_index(
            "idx_data_source_records_slug_ingested",
            "data_source_records",
            ["source_slug", "ingested_at", "observed_at"],
            unique=False,
        )


def downgrade() -> None:
    pass
