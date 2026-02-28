"""Add durable scanner batch queue for aggregation cutover.

Revision ID: 202602280004
Revises: 202602280003
Create Date: 2026-02-28
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from alembic_helpers import index_names, table_names


revision = "202602280004"
down_revision = "202602280003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    tables = table_names()
    if "scanner_batch_queue" not in tables:
        op.create_table(
            "scanner_batch_queue",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("source", sa.String(), nullable=False, server_default=sa.text("'scanner'")),
            sa.Column("batch_kind", sa.String(), nullable=False, server_default=sa.text("'scan_cycle'")),
            sa.Column("opportunities_json", sa.JSON(), nullable=False, server_default=sa.text("'[]'::json")),
            sa.Column("status_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
            sa.Column("emitted_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
            sa.Column("lease_owner", sa.String(), nullable=True),
            sa.Column("lease_expires_at", sa.DateTime(), nullable=True),
            sa.Column("attempt_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
            sa.Column("processed_at", sa.DateTime(), nullable=True),
            sa.Column("error", sa.Text(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )

    existing_indexes = index_names("scanner_batch_queue")
    if "idx_scanner_batch_queue_pending" not in existing_indexes:
        op.create_index(
            "idx_scanner_batch_queue_pending",
            "scanner_batch_queue",
            ["processed_at", "emitted_at"],
            unique=False,
        )
    if "idx_scanner_batch_queue_emitted" not in existing_indexes:
        op.create_index(
            "idx_scanner_batch_queue_emitted",
            "scanner_batch_queue",
            ["emitted_at"],
            unique=False,
        )
    if "idx_scanner_batch_queue_lease" not in existing_indexes:
        op.create_index(
            "idx_scanner_batch_queue_lease",
            "scanner_batch_queue",
            ["lease_expires_at"],
            unique=False,
        )


def downgrade() -> None:
    return
