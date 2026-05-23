"""SQLAlchemy models for the global search subsystem."""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Column,
    Float,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB

from models.database import Base, DateTime, _utcnow


class SearchIndex(Base):
    """One row per searchable entity across the whole system.

    The ``tsv`` column is a Postgres ``GENERATED ALWAYS AS STORED``
    tsvector that the migration creates with weighted sections
    (title=A, subtitle=B, body=C).  Python never writes ``tsv``
    directly — it's derived from the text columns automatically.

    ``recency``, ``liquidity`` and ``volume`` are blended into the
    ranking expression at query time so results respect the financial
    context (a more liquid Trump market beats a stale, illiquid one
    of the same name).
    """

    __tablename__ = "search_index"

    entity_type = Column(String(64), primary_key=True)
    entity_id = Column(String(255), primary_key=True)
    title = Column(Text, nullable=False)
    subtitle = Column(Text, nullable=True)
    body = Column(Text, nullable=True)
    category = Column(String(128), nullable=True)
    tags = Column(JSONB, default=list, nullable=False)
    metadata_json = Column(JSONB, default=dict, nullable=False)
    liquidity = Column(Float, nullable=True)
    volume = Column(Float, nullable=True)
    recency = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=_utcnow, nullable=False)
    # ``tsv`` is a generated column managed by Postgres; expose it as a
    # read-only opaque column on the ORM side so SQLAlchemy doesn't try
    # to bind it on INSERT/UPDATE.
    # We deliberately don't map ``tsv`` here — all FTS predicates are
    # built via ``sa.text()`` in the service layer.

    __table_args__ = (
        Index("idx_search_index_entity_type", "entity_type"),
        Index("idx_search_index_recency", "recency"),
        Index("idx_search_index_updated_at", "updated_at"),
        {"prefixes": ["UNLOGGED"]},  # rebuildable search index; see migration 202605230001
    )


class SearchQueryLog(Base):
    """Telemetry: one row per ``/search/global`` invocation.

    Powers "recent searches" suggestions in the UI and surfaces
    zero-result queries for tuning.  Kept lightweight on purpose —
    no per-result rows, just the query and an aggregate.
    """

    __tablename__ = "search_query_log"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False)
    result_count = Column(Integer, default=0, nullable=False)
    top_entity_type = Column(String(64), nullable=True)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=_utcnow, nullable=False)

    __table_args__ = (
        Index("idx_search_query_log_created_at", "created_at"),
    )
