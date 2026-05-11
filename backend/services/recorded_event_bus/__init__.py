"""Canonical recorded-event bus for Homerun's data plane.

This module is the institutional-grade backbone that the existing
recorders / data sources / backtest engine should converge on over
time.  Today the system has FIVE parallel pipelines (live ws→sql,
external→parquet, opportunity recorder→sql, news ingest→sql, in-memory
crypto-update broadcast) with no shared contract — each one invented
its own schema, timestamps, and replay glue.  Adding a sixth source
means writing all of it over again.  Backtesting the news lane and
the crypto-strategy lane is impossible because those topics were
never assigned replayable storage.

This package defines the missing layer:

  * :class:`RecordedEvent` — one envelope every observation in the
    system flows through (live + recorded).  Carries topic, entity,
    bitemporal stamps (``observed_at`` for truth-time, ``ingested_at``
    for knowledge-time), source identity, schema version, payload.

  * :class:`TopicCatalog` — single source of truth for "what topics
    exist, where they're stored, what their schema is, who publishes
    them, who subscribes."  Backtest Studio's data picker, Data Lab's
    source list, the strategy editor's autocomplete, the replay
    engine — all read from here.

  * :class:`RecordedEventBus` — one publish / subscribe API.  Live
    delivery uses in-memory fan-out (same call site as today).
    Backtest delivery reads from the topic's storage backend over
    ``(window, topics)`` and time-merges across topics so a strategy
    sees the same envelope sequence it would have seen live.

  * Storage adapters — parquet for hot time-series (the default for
    new topics), SQL wrappers for the existing tables that already
    hold years of data (``market_microstructure_snapshots``,
    ``book_delta_events``, ``wallet_monitor_events``).  Same bus,
    different backings.

Design discipline (institutional-grade — not negotiable):

  * **Single envelope.**  Every observation has the same wrapper.  No
    per-topic table shapes.
  * **Bitemporal.**  ``observed_at`` for replay-by-truth-time;
    ``ingested_at`` for replay-by-knowledge-time so we can detect
    backtest leakage ("strategy knew at observed_at=T but we only
    actually recorded at ingested_at=T+δ").
  * **Topic naming is structured.**  Dotted, lowercased, three-part
    minimum: ``{venue}.{kind}.{subkind}``.  Examples:
    ``polymarket.book.snapshot``, ``polymarket.book.delta``,
    ``polymarket.trade``, ``crypto.update.btc_eth``,
    ``news.gdelt.story``, ``wallet.trade``.
  * **Versioned payloads.**  Every payload schema carries an integer
    version; replay knows how to up-convert older payloads on the
    way out so strategies always see the latest shape.
  * **Catalog is authoritative.**  No topic exists until it's
    registered.  Recorders fail-closed if they try to publish to an
    unregistered topic.  This catches drift early.

This module is intentionally a thin coordination layer — storage
adapters live in :mod:`services.recorded_event_bus.storage`, the
catalog table is plain SQLAlchemy ORM in ``models.database``, and
adapters around existing SQL tables live in
:mod:`services.recorded_event_bus.adapters`.  Everything composes;
nothing duplicates.
"""
from __future__ import annotations

from services.recorded_event_bus.envelope import (
    RecordedEvent,
    EnvelopeValidationError,
    parse_topic,
)
from services.recorded_event_bus.catalog import (
    TopicSpec,
    TopicNotRegisteredError,
    register_topic,
    get_topic,
    list_topics,
    delete_topic,
)
from services.recorded_event_bus.bus import (
    RecordedEventBus,
    Subscription,
    ReplayWindow,
    bus,
)

__all__ = [
    # Envelope
    "RecordedEvent",
    "EnvelopeValidationError",
    "parse_topic",
    # Catalog
    "TopicSpec",
    "TopicNotRegisteredError",
    "register_topic",
    "get_topic",
    "list_topics",
    "delete_topic",
    # Bus
    "RecordedEventBus",
    "Subscription",
    "ReplayWindow",
    "bus",
]
