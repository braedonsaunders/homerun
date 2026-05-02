"""Global search infrastructure: pg_trgm, search_index, search_query_log

Revision ID: 202605010004
Revises: 202605010003
Create Date: 2026-05-01

Adds the substrate for institutional-grade global search:

* ``pg_trgm`` extension for fuzzy / typo-tolerant matching via similarity
* ``search_index``: a single materialized index over every searchable
  entity in the system (markets, events, traders, strategies, data
  sources, news, wallets, alerts, research sessions, ...).  One row per
  entity, with a generated ``tsvector`` derived from
  ``title (A) | subtitle (B) | body (C)`` so we get BM25-ish ranking
  via ``ts_rank_cd`` for free.  Finance-specific signals
  (``liquidity``, ``volume``, ``recency``) live as regular columns and
  are folded into the composite ranking expression at query time.
* ``search_query_log``: lightweight telemetry on every query (query
  text, result count, top entity type, latency).  Powers the "recent
  searches" suggestions and surfaces zero-result queries for tuning.

Indexes:

* ``idx_search_index_tsv`` — GIN over the tsvector (full-text)
* ``idx_search_index_title_trgm`` — GIN trigram on ``title`` for
  fuzzy / prefix matching when the user mistypes
* ``idx_search_index_entity_type`` — B-tree for type-faceted filtering
* ``idx_search_index_recency`` — B-tree for recency boosts
* ``idx_search_index_updated_at`` — B-tree for the sweep-delete pass
  during periodic reindex (delete rows with stale ``updated_at``)
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "202605010004"
down_revision = "202605010003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    op.execute(
        """
        CREATE TABLE search_index (
            entity_type    VARCHAR(64)  NOT NULL,
            entity_id      VARCHAR(255) NOT NULL,
            title          TEXT         NOT NULL,
            subtitle       TEXT         NULL,
            body           TEXT         NULL,
            category       VARCHAR(128) NULL,
            tags           JSONB        NOT NULL DEFAULT '[]'::jsonb,
            metadata_json  JSONB        NOT NULL DEFAULT '{}'::jsonb,
            liquidity      DOUBLE PRECISION NULL,
            volume         DOUBLE PRECISION NULL,
            recency        TIMESTAMP    NULL,
            updated_at     TIMESTAMP    NOT NULL DEFAULT (now() AT TIME ZONE 'utc'),
            tsv            TSVECTOR     GENERATED ALWAYS AS (
                setweight(to_tsvector('english', coalesce(title, '')),    'A') ||
                setweight(to_tsvector('english', coalesce(subtitle, '')), 'B') ||
                setweight(to_tsvector('english', coalesce(body, '')),     'C')
            ) STORED,
            PRIMARY KEY (entity_type, entity_id)
        )
        """
    )

    op.execute("CREATE INDEX idx_search_index_tsv ON search_index USING GIN (tsv)")
    op.execute(
        "CREATE INDEX idx_search_index_title_trgm "
        "ON search_index USING GIN (title gin_trgm_ops)"
    )
    op.create_index(
        "idx_search_index_entity_type",
        "search_index",
        ["entity_type"],
    )
    op.create_index(
        "idx_search_index_recency",
        "search_index",
        ["recency"],
    )
    op.create_index(
        "idx_search_index_updated_at",
        "search_index",
        ["updated_at"],
    )

    op.create_table(
        "search_query_log",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("result_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("top_entity_type", sa.String(length=64), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("(now() AT TIME ZONE 'utc')"),
        ),
    )
    op.create_index(
        "idx_search_query_log_created_at",
        "search_query_log",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_search_query_log_created_at", table_name="search_query_log")
    op.drop_table("search_query_log")
    op.drop_index("idx_search_index_updated_at", table_name="search_index")
    op.drop_index("idx_search_index_recency", table_name="search_index")
    op.drop_index("idx_search_index_entity_type", table_name="search_index")
    op.execute("DROP INDEX IF EXISTS idx_search_index_title_trgm")
    op.execute("DROP INDEX IF EXISTS idx_search_index_tsv")
    op.execute("DROP TABLE IF EXISTS search_index")
    # Leave pg_trgm in place — other migrations may depend on it later.
