"""Add unified data source tables and execution history.

Revision ID: 202602180003
Revises: 202602180002
Create Date: 2026-02-18
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from alembic_helpers import index_names, table_names


revision = "202602180003"
down_revision = "202602180002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    existing_tables = table_names()

    if "data_source_tombstones" not in existing_tables:
        op.create_table(
            "data_source_tombstones",
            sa.Column("slug", sa.String(), primary_key=True),
            sa.Column("deleted_at", sa.DateTime(), nullable=False),
            sa.Column("reason", sa.String(), nullable=True),
        )
        existing_tables.add("data_source_tombstones")

    tombstone_indexes = index_names("data_source_tombstones")
    if "idx_data_source_tombstones_deleted_at" not in tombstone_indexes:
        op.create_index(
            "idx_data_source_tombstones_deleted_at",
            "data_source_tombstones",
            ["deleted_at"],
            unique=False,
        )

    if "data_sources" not in existing_tables:
        op.create_table(
            "data_sources",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("slug", sa.String(), nullable=False, unique=True),
            sa.Column("source_key", sa.String(), nullable=False, server_default=sa.text("'custom'")),
            sa.Column("source_kind", sa.String(), nullable=False, server_default=sa.text("'python'")),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("source_code", sa.Text(), nullable=False, server_default=sa.text("''")),
            sa.Column("class_name", sa.String(), nullable=True),
            sa.Column("is_system", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("enabled", sa.Boolean(), nullable=True, server_default=sa.true()),
            sa.Column("status", sa.String(), nullable=True, server_default=sa.text("'unloaded'")),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("config", sa.JSON(), nullable=True, server_default=sa.text("'{}'::json")),
            sa.Column("config_schema", sa.JSON(), nullable=True, server_default=sa.text("'{}'::json")),
            sa.Column("version", sa.Integer(), nullable=True, server_default=sa.text("1")),
            sa.Column("sort_order", sa.Integer(), nullable=True, server_default=sa.text("0")),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
        )
        existing_tables.add("data_sources")

    source_indexes = index_names("data_sources")
    if "idx_data_source_slug" not in source_indexes:
        op.create_index("idx_data_source_slug", "data_sources", ["slug"], unique=False)
    if "idx_data_source_source_key" not in source_indexes:
        op.create_index("idx_data_source_source_key", "data_sources", ["source_key"], unique=False)
    if "idx_data_source_source_kind" not in source_indexes:
        op.create_index("idx_data_source_source_kind", "data_sources", ["source_kind"], unique=False)
    if "idx_data_source_enabled" not in source_indexes:
        op.create_index("idx_data_source_enabled", "data_sources", ["enabled"], unique=False)
    if "idx_data_source_is_system" not in source_indexes:
        op.create_index("idx_data_source_is_system", "data_sources", ["is_system"], unique=False)
    if "idx_data_source_status" not in source_indexes:
        op.create_index("idx_data_source_status", "data_sources", ["status"], unique=False)

    if "data_source_runs" not in existing_tables:
        op.create_table(
            "data_source_runs",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("data_source_id", sa.String(), nullable=False),
            sa.Column("source_slug", sa.String(), nullable=False),
            sa.Column("status", sa.String(), nullable=False, server_default=sa.text("'success'")),
            sa.Column("fetched_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
            sa.Column("transformed_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
            sa.Column("upserted_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
            sa.Column("skipped_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("metadata_json", sa.JSON(), nullable=True),
            sa.Column("started_at", sa.DateTime(), nullable=False),
            sa.Column("completed_at", sa.DateTime(), nullable=True),
            sa.Column("duration_ms", sa.Integer(), nullable=True),
            sa.ForeignKeyConstraint(
                ["data_source_id"],
                ["data_sources.id"],
                ondelete="CASCADE",
            ),
        )
        existing_tables.add("data_source_runs")

    run_indexes = index_names("data_source_runs")
    if "idx_data_source_runs_source_slug" not in run_indexes:
        op.create_index("idx_data_source_runs_source_slug", "data_source_runs", ["source_slug"], unique=False)
    if "idx_data_source_runs_started_at" not in run_indexes:
        op.create_index("idx_data_source_runs_started_at", "data_source_runs", ["started_at"], unique=False)
    if "idx_data_source_runs_status" not in run_indexes:
        op.create_index("idx_data_source_runs_status", "data_source_runs", ["status"], unique=False)
    if "ix_data_source_runs_data_source_id" not in run_indexes:
        op.create_index(
            "ix_data_source_runs_data_source_id",
            "data_source_runs",
            ["data_source_id"],
            unique=False,
        )

    if "data_source_records" not in existing_tables:
        op.create_table(
            "data_source_records",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("data_source_id", sa.String(), nullable=False),
            sa.Column("source_slug", sa.String(), nullable=False),
            sa.Column("external_id", sa.String(), nullable=True),
            sa.Column("title", sa.Text(), nullable=True),
            sa.Column("summary", sa.Text(), nullable=True),
            sa.Column("category", sa.String(), nullable=True),
            sa.Column("source", sa.String(), nullable=True),
            sa.Column("url", sa.Text(), nullable=True),
            sa.Column("geotagged", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("country_iso3", sa.String(), nullable=True),
            sa.Column("latitude", sa.Float(), nullable=True),
            sa.Column("longitude", sa.Float(), nullable=True),
            sa.Column("observed_at", sa.DateTime(), nullable=True),
            sa.Column("ingested_at", sa.DateTime(), nullable=False),
            sa.Column("payload_json", sa.JSON(), nullable=True),
            sa.Column("transformed_json", sa.JSON(), nullable=True),
            sa.Column("tags_json", sa.JSON(), nullable=True),
            sa.ForeignKeyConstraint(
                ["data_source_id"],
                ["data_sources.id"],
                ondelete="CASCADE",
            ),
        )
        existing_tables.add("data_source_records")

    record_indexes = index_names("data_source_records")
    if "idx_data_source_records_source_slug" not in record_indexes:
        op.create_index(
            "idx_data_source_records_source_slug",
            "data_source_records",
            ["source_slug"],
            unique=False,
        )
    if "idx_data_source_records_data_source_id" not in record_indexes:
        op.create_index(
            "idx_data_source_records_data_source_id",
            "data_source_records",
            ["data_source_id"],
            unique=False,
        )
    if "idx_data_source_records_observed_at" not in record_indexes:
        op.create_index(
            "idx_data_source_records_observed_at",
            "data_source_records",
            ["observed_at"],
            unique=False,
        )
    if "idx_data_source_records_ingested_at" not in record_indexes:
        op.create_index(
            "idx_data_source_records_ingested_at",
            "data_source_records",
            ["ingested_at"],
            unique=False,
        )
    if "idx_data_source_records_geotagged" not in record_indexes:
        op.create_index(
            "idx_data_source_records_geotagged",
            "data_source_records",
            ["geotagged"],
            unique=False,
        )
    if "idx_data_source_records_country" not in record_indexes:
        op.create_index(
            "idx_data_source_records_country",
            "data_source_records",
            ["country_iso3"],
            unique=False,
        )
    if "idx_data_source_records_external" not in record_indexes:
        op.create_index(
            "idx_data_source_records_external",
            "data_source_records",
            ["source_slug", "external_id"],
            unique=False,
        )
    if "ix_data_source_records_data_source_id" not in record_indexes:
        op.create_index(
            "ix_data_source_records_data_source_id",
            "data_source_records",
            ["data_source_id"],
            unique=False,
        )


def downgrade() -> None:
    pass
