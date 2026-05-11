"""Add rotation/pruning controls for the recorded-event bus.

Revision ID: 202605110400
Revises: 202605110300
Create Date: 2026-05-11

Adds:
  * ``topic_catalog.max_bytes`` — per-topic hard size cap.  Pruner
    deletes oldest partition files when a topic exceeds this.
  * ``app_settings.recorded_event_bus_global_max_bytes`` — global cap
    across all parquet topics.  Pruner enforces both per-topic and
    global caps.
  * ``app_settings.recorded_event_bus_pruner_enabled`` — master
    kill switch for the periodic pruner task.

The pruner runs in the live process (lifespan-managed task), so the
operator can flip the switch from the UI at any time without a deploy.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from alembic_helpers import safe_add_column


revision = "202605110400"
down_revision = "202605110300"
branch_labels = None
depends_on = None


def upgrade() -> None:
    safe_add_column(
        "topic_catalog",
        sa.Column("max_bytes", sa.BigInteger(), nullable=True),
    )
    safe_add_column(
        "app_settings",
        sa.Column("recorded_event_bus_global_max_bytes", sa.BigInteger(), nullable=True),
    )
    safe_add_column(
        "app_settings",
        sa.Column(
            "recorded_event_bus_pruner_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
    )


def downgrade() -> None:
    op.drop_column("app_settings", "recorded_event_bus_pruner_enabled")
    op.drop_column("app_settings", "recorded_event_bus_global_max_bytes")
    op.drop_column("topic_catalog", "max_bytes")
