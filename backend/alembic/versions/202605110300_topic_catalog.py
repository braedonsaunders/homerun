"""Add topic_catalog table — single source of truth for the
recorded-event-bus architecture.

Revision ID: 202605110300
Revises: 202605110200
Create Date: 2026-05-11

The recorded-event bus (services/recorded_event_bus) is the unified
data plane every recorder / strategy / backtest replay reads through.
This table is its registry: every topic the bus knows about lives
here.  Recorders fail-closed if they try to publish to an unregistered
topic; backtest replay refuses to read unknown topics; the UI surfaces
this list in Backtest Studio's data picker and Data Lab's source list.

Without this table the bus has no contract — anyone can publish any
topic name and there's no way to know what shape the payload takes,
where it's stored, or who reads from it.  That's exactly the situation
the architecture is replacing.

The seeding pass (run by ``services/recorded_event_bus/catalog.py``
on first import) registers the well-known topics that wrap existing
SQL tables: ``polymarket.book.snapshot``, ``polymarket.book.delta``,
``wallet.trade``, ``opportunity.detected``.  New topics
(``crypto.update.btc_eth``, ``news.gdelt.story``, etc.) get registered
either by the recorder at startup or by the operator via the UI.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "202605110300"
down_revision = "202605110200"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "topic_catalog",
        sa.Column("slug", sa.String(), primary_key=True),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("storage_kind", sa.String(), nullable=False),
        sa.Column("storage_uri", sa.String(), nullable=True),
        sa.Column(
            "payload_schema_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("schema_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("retention_days", sa.Integer(), nullable=True),
        sa.Column(
            "publishers_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "subscribers_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("is_replayable", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("last_published_at", sa.DateTime(), nullable=True),
        sa.Column("last_replayed_at", sa.DateTime(), nullable=True),
        sa.Column(
            "event_count",
            sa.BigInteger(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "bytes_on_disk",
            sa.BigInteger(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_topic_catalog_storage_kind",
        "topic_catalog",
        ["storage_kind"],
    )
    op.create_index(
        "ix_topic_catalog_enabled_replayable",
        "topic_catalog",
        ["enabled", "is_replayable"],
    )


def downgrade() -> None:
    op.drop_index("ix_topic_catalog_enabled_replayable", table_name="topic_catalog")
    op.drop_index("ix_topic_catalog_storage_kind", table_name="topic_catalog")
    op.drop_table("topic_catalog")
