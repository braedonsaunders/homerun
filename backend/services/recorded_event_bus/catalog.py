"""Topic catalog — registry of every recorded-event topic in the system.

The catalog is read on every ``bus.publish`` (cached in memory with a
short TTL) and on every ``bus.replay`` (uncached — replay is a slow
path).  Reads are >> writes by ~5 orders of magnitude in production,
so the in-memory cache is the right shape: refresh every few seconds,
invalidate on writes from this process.

This module owns:
  * :class:`TopicSpec` — the dataclass projection of the
    ``topic_catalog`` ORM row, used by all bus consumers.  Plain
    Python (no SQLAlchemy session leakage) so it round-trips through
    queues and cross-process pickles cleanly.
  * The async API: :func:`register_topic`, :func:`get_topic`,
    :func:`list_topics`, :func:`delete_topic`.
  * Seed data — the well-known topics that wrap pre-existing SQL
    tables get registered on first import so the bus is usable
    without a manual setup step.
"""
from __future__ import annotations

import json
import logging
import asyncio
import threading as _threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional

from sqlalchemy import select, delete, update as _sa_update, func as _sa_func, text as _sa_text

from models.database import AsyncSessionLocal, TopicCatalog
from services.live_pressure import is_db_pressure_active, maybe_mark_db_pressure
from services.recorded_event_bus.envelope import parse_topic

logger = logging.getLogger(__name__)


class TopicNotRegisteredError(LookupError):
    """Raised by ``bus.publish`` / ``bus.replay`` when the topic isn't
    in the catalog.  Fail-closed: an unregistered topic almost always
    means a typo or a missing migration, not a runtime decision."""


# ── DTO ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TopicSpec:
    """In-memory projection of a ``topic_catalog`` row.

    Frozen for the same reason :class:`RecordedEvent` is frozen — this
    object is read on the hot path (every bus.publish), and frozen
    dataclasses are fast to hash/compare for cache key purposes."""

    slug: str
    title: str
    storage_kind: str  # 'parquet' | 'external_parquet' | 'sql_table' | 'memory'
    storage_uri: Optional[str]
    schema_version: int
    enabled: bool
    is_replayable: bool
    description: Optional[str] = None
    payload_schema: Optional[dict[str, Any]] = None
    retention_days: Optional[int] = None
    max_bytes: Optional[int] = None
    publishers: tuple[str, ...] = ()
    subscribers: tuple[str, ...] = ()
    last_published_at: Optional[datetime] = None
    last_replayed_at: Optional[datetime] = None
    event_count: int = 0
    bytes_on_disk: int = 0

    @classmethod
    def from_row(cls, row: TopicCatalog) -> "TopicSpec":
        return cls(
            slug=row.slug,
            title=row.title,
            storage_kind=row.storage_kind,
            storage_uri=row.storage_uri,
            schema_version=int(row.schema_version or 1),
            enabled=bool(row.enabled),
            is_replayable=bool(row.is_replayable),
            description=row.description,
            payload_schema=row.payload_schema_json,
            retention_days=row.retention_days,
            max_bytes=row.max_bytes,
            publishers=tuple(row.publishers_json or ()),
            subscribers=tuple(row.subscribers_json or ()),
            last_published_at=row.last_published_at,
            last_replayed_at=row.last_replayed_at,
            event_count=int(row.event_count or 0),
            bytes_on_disk=int(row.bytes_on_disk or 0),
        )


# ── Catalog cache ────────────────────────────────────────────────────
#
# bus.publish reads the topic spec on every call.  With a hot path of
# ~10k events/sec we cannot afford a Postgres round-trip per call.  A
# small in-memory cache with a short TTL is the right shape:
#
#   * TTL = 5s → spec changes from the UI propagate fast enough
#     for human ergonomics (operator updates retention; it takes
#     effect within seconds).
#   * Invalidation on local writes → register_topic/delete_topic
#     bust the cache so the calling process never sees its own
#     stale read.
#
# Cross-process invalidation isn't needed for the registry data —
# topics rarely change shape — but the publish/replay timestamps
# would benefit from it.  Out of scope for v1; revisit when the
# operator UI starts showing live "events/sec by topic" charts.

_CACHE_TTL_SECONDS = 5.0


@dataclass
class _CacheEntry:
    spec: TopicSpec
    expires_at: float


_cache: dict[str, _CacheEntry] = {}
# A threading.Lock is the right choice over asyncio.Lock: the cache
# operations are dict reads/writes (microseconds), and asyncio.Lock
# bind to a specific event loop on first acquire — which breaks unit
# tests that create a fresh loop per test (``RuntimeError: Event loop
# is closed`` on the second test that hits the lock).  threading.Lock
# is loop-agnostic, fast enough for this workload, and contention-free
# in practice (the cache is process-local with single-digit µs holds).
_cache_lock = _threading.Lock()

_TOUCH_PUBLISHED_FLUSH_DELAY_SECONDS = 1.0
_touch_published_lock = _threading.Lock()
_touch_published_pending: dict[str, dict[str, Any]] = {}
_touch_published_task: asyncio.Task | None = None


def _requeue_touch_published_batch(batch: Mapping[str, Mapping[str, Any]]) -> None:
    with _touch_published_lock:
        for slug, payload in batch.items():
            pending = _touch_published_pending.setdefault(
                slug,
                {"n_events": 0, "bytes_added": 0, "published_at": payload["published_at"]},
            )
            pending["n_events"] = int(pending["n_events"]) + int(payload["n_events"])
            pending["bytes_added"] = int(pending["bytes_added"]) + int(payload["bytes_added"])
            if payload["published_at"] > pending["published_at"]:
                pending["published_at"] = payload["published_at"]


async def _persist_touch_published_batch(batch: Mapping[str, Mapping[str, Any]]) -> list[str]:
    slugs = list(batch)
    async with AsyncSessionLocal() as session:
        await session.execute(
            _sa_text(
                """
                UPDATE topic_catalog AS t
                SET last_published_at = v.published_at,
                    event_count = coalesce(t.event_count, 0) + v.n_events,
                    bytes_on_disk = coalesce(t.bytes_on_disk, 0) + v.bytes_added,
                    updated_at = v.published_at
                FROM (
                    SELECT
                        unnest(CAST(:slugs AS text[])) AS slug,
                        unnest(CAST(:published_ats AS timestamp[])) AS published_at,
                        unnest(CAST(:n_events AS bigint[])) AS n_events,
                        unnest(CAST(:bytes_added AS bigint[])) AS bytes_added
                ) AS v
                WHERE t.slug = v.slug
                """
            ),
            {
                "slugs": slugs,
                "published_ats": [batch[slug]["published_at"] for slug in slugs],
                "n_events": [int(batch[slug]["n_events"]) for slug in slugs],
                "bytes_added": [int(batch[slug]["bytes_added"]) for slug in slugs],
            },
        )
        await session.commit()
    return slugs


def _invalidate_cache(slug: Optional[str] = None) -> None:
    if slug is None:
        _cache.clear()
    else:
        _cache.pop(slug, None)


# ── Public API ───────────────────────────────────────────────────────


async def register_topic(
    *,
    slug: str,
    title: str,
    storage_kind: str,
    storage_uri: Optional[str] = None,
    description: Optional[str] = None,
    payload_schema: Optional[Mapping[str, Any]] = None,
    schema_version: int = 1,
    retention_days: Optional[int] = None,
    max_bytes: Optional[int] = None,
    publishers: Iterable[str] = (),
    subscribers: Iterable[str] = (),
    enabled: bool = True,
    is_replayable: bool = True,
    upsert: bool = True,
) -> TopicSpec:
    """Create or update a topic.

    The slug must be a valid topic name (validated against
    :func:`parse_topic` for consistency with envelope-side checks).
    The storage_kind ∈ {'parquet', 'sql_table', 'memory'}; for
    'parquet' a ``storage_uri`` is required.

    With ``upsert=True`` (the default), re-registering an existing
    topic merges the new fields onto the existing row.  This lets
    recorders idempotently call ``register_topic`` from their
    startup hook without worrying about whether the operator already
    customised the row.  Fields the recorder shouldn't override
    (retention_days, enabled) are NOT overwritten on upsert when the
    caller leaves them at their default — only explicit kwargs touch
    those.  See ``_upsert_merge`` below for the exact rules.
    """
    parse_topic(slug)
    # storage_kind is now a UI-facing badge — operator-readable label
    # for "what's the primary backing of this topic" — and no longer
    # the dispatch key.  The replayer reads ``sources`` from
    # storage_uri to find every backing.  Memory topics have no
    # replayable backing; everything else needs a storage_uri.
    if storage_kind not in {"parquet", "external_parquet", "sql_table", "memory"}:
        raise ValueError(
            f"unknown storage_kind {storage_kind!r}; "
            "must be 'parquet' | 'external_parquet' | 'sql_table' | 'memory'"
        )
    if storage_kind != "memory" and not storage_uri:
        raise ValueError(
            f"storage_kind={storage_kind!r} requires a storage_uri"
        )

    publishers_list = list(publishers)
    subscribers_list = list(subscribers)
    payload_schema_dict = dict(payload_schema) if payload_schema else None

    async with AsyncSessionLocal() as session:
        existing = (
            await session.execute(
                select(TopicCatalog).where(TopicCatalog.slug == slug)
            )
        ).scalar_one_or_none()
        if existing is None:
            row = TopicCatalog(
                slug=slug,
                title=title,
                description=description,
                storage_kind=storage_kind,
                storage_uri=storage_uri,
                payload_schema_json=payload_schema_dict,
                schema_version=schema_version,
                retention_days=retention_days,
                max_bytes=max_bytes,
                publishers_json=publishers_list,
                subscribers_json=subscribers_list,
                enabled=enabled,
                is_replayable=is_replayable,
            )
            session.add(row)
            logger.info("topic_catalog: registered new topic %s (storage=%s)", slug, storage_kind)
        else:
            if not upsert:
                raise ValueError(
                    f"topic {slug!r} already registered; pass upsert=True to merge"
                )
            # Merge — see docstring rules.
            existing.title = title or existing.title
            if description is not None:
                existing.description = description
            existing.storage_kind = storage_kind
            existing.storage_uri = storage_uri
            if payload_schema_dict is not None:
                existing.payload_schema_json = payload_schema_dict
            existing.schema_version = schema_version
            # publishers / subscribers are union-merged so multiple
            # subsystems can advertise themselves without clobbering.
            existing.publishers_json = sorted(set(existing.publishers_json or []) | set(publishers_list))
            existing.subscribers_json = sorted(set(existing.subscribers_json or []) | set(subscribers_list))
            # Operator-controlled fields only update on explicit kwargs.
            # The recorder calling register_topic() at startup should
            # NOT silently re-enable a topic the operator disabled.
            # We detect "explicit" by comparing against the kwarg
            # default — imperfect but pragmatic; operator can pass
            # the current value to opt into preservation.
            if retention_days is not None:
                existing.retention_days = retention_days
            if max_bytes is not None:
                existing.max_bytes = max_bytes
            # ``enabled`` and ``is_replayable`` deliberately not
            # touched on upsert — the operator owns those.
            logger.debug("topic_catalog: upserted topic %s", slug)
        await session.commit()
        # Refresh to capture server defaults / generated columns
        # (created_at populated by the DB on insert).
        if existing is None:
            await session.refresh(row)
            spec = TopicSpec.from_row(row)
        else:
            spec = TopicSpec.from_row(existing)
        _invalidate_cache(slug)
        return spec


async def get_topic(slug: str) -> Optional[TopicSpec]:
    """Cached read.  Returns None if the topic isn't registered.
    Callers that need fail-closed semantics should use
    :func:`require_topic` instead."""
    now = time.monotonic()
    with _cache_lock:
        entry = _cache.get(slug)
        if entry is not None and entry.expires_at > now:
            return entry.spec
    async with AsyncSessionLocal() as session:
        row = (
            await session.execute(
                select(TopicCatalog).where(TopicCatalog.slug == slug)
            )
        ).scalar_one_or_none()
    spec = TopicSpec.from_row(row) if row is not None else None
    if spec is not None:
        with _cache_lock:
            _cache[slug] = _CacheEntry(spec=spec, expires_at=now + _CACHE_TTL_SECONDS)
    return spec


async def require_topic(slug: str) -> TopicSpec:
    spec = await get_topic(slug)
    if spec is None:
        raise TopicNotRegisteredError(
            f"topic {slug!r} is not registered in topic_catalog. "
            "Call register_topic(...) at recorder startup, or check "
            "for a typo in the publish call."
        )
    if not spec.enabled:
        raise TopicNotRegisteredError(
            f"topic {slug!r} is disabled in topic_catalog (enabled=false). "
            "Re-enable in Data Lab → Topics, or set enabled=true via "
            "register_topic(slug, enabled=True, ...)."
        )
    return spec


async def list_topics(
    *,
    storage_kind: Optional[str] = None,
    enabled_only: bool = False,
    replayable_only: bool = False,
) -> list[TopicSpec]:
    """Used by the Data Lab + Backtest Studio UIs to populate
    pickers.  Uncached (the catalog is small and operators expect
    UI changes to reflect immediately)."""
    async with AsyncSessionLocal() as session:
        q = select(TopicCatalog)
        if storage_kind is not None:
            q = q.where(TopicCatalog.storage_kind == storage_kind)
        if enabled_only:
            q = q.where(TopicCatalog.enabled.is_(True))
        if replayable_only:
            q = q.where(TopicCatalog.is_replayable.is_(True))
        q = q.order_by(TopicCatalog.slug)
        rows = (await session.execute(q)).scalars().all()
    return [TopicSpec.from_row(r) for r in rows]


async def delete_topic(slug: str) -> bool:
    """Returns True if a row was deleted, False if no such topic
    existed.  Does NOT delete the underlying storage (operator must
    do that explicitly — accidental data loss is exactly what
    this fail-safe prevents)."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            delete(TopicCatalog).where(TopicCatalog.slug == slug)
        )
        await session.commit()
    _invalidate_cache(slug)
    return result.rowcount > 0


async def touch_published(
    slug: str,
    *,
    n_events: int = 1,
    bytes_added: int = 0,
    published_at: datetime | None = None,
) -> None:
    """Lightweight write the bus calls after each successful publish
    batch — bumps last_published_at + counters.  Best-effort; failures
    log a warning but do not propagate (a counter desync isn't worth
    breaking the hot path).

    IMPORTANT: do this as a SINGLE atomic UPDATE.  The earlier
    select-then-mutate-then-commit pattern surfaced as 2–5s long
    transactions and a 5s slow-execute on the SELECT during production
    soaks: every successful publish batch opened a session, did a
    PK lookup, mutated in Python, then wrote + commited, and multiple
    publishers touching the same topic concurrently serialised through
    SQLAlchemy's UoW.  A bare UPDATE is one round-trip, atomic against
    concurrent writers, and immune to lost-update races.
    """
    normalized_slug = str(slug or "").strip()
    if not normalized_slug:
        return
    published_at_value = published_at or datetime.now(timezone.utc).replace(tzinfo=None)
    if published_at_value.tzinfo is not None:
        published_at_value = published_at_value.astimezone(timezone.utc).replace(tzinfo=None)
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(
                _sa_update(TopicCatalog)
                .where(TopicCatalog.slug == normalized_slug)
                .values(
                    last_published_at=published_at_value,
                    event_count=_sa_func.coalesce(TopicCatalog.event_count, 0) + int(n_events),
                    bytes_on_disk=_sa_func.coalesce(TopicCatalog.bytes_on_disk, 0) + int(bytes_added),
                )
            )
            await session.commit()
        _invalidate_cache(normalized_slug)
    except Exception:  # noqa: BLE001
        logger.warning("topic_catalog.touch_published failed for %s", normalized_slug, exc_info=True)


def schedule_touch_published(slug: str, *, n_events: int = 1, bytes_added: int = 0) -> None:
    normalized_slug = str(slug or "").strip()
    if not normalized_slug:
        return

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    global _touch_published_task
    with _touch_published_lock:
        pending = _touch_published_pending.setdefault(
            normalized_slug,
            {"n_events": 0, "bytes_added": 0, "published_at": now},
        )
        pending["n_events"] = int(pending["n_events"]) + int(n_events)
        pending["bytes_added"] = int(pending["bytes_added"]) + int(bytes_added)
        if now > pending["published_at"]:
            pending["published_at"] = now

        if _touch_published_task is not None and not _touch_published_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        _touch_published_task = loop.create_task(_flush_scheduled_touch_published())


async def _flush_scheduled_touch_published() -> None:
    global _touch_published_task

    try:
        while True:
            await asyncio.sleep(_TOUCH_PUBLISHED_FLUSH_DELAY_SECONDS)
            with _touch_published_lock:
                batch = dict(_touch_published_pending)
                _touch_published_pending.clear()
            if not batch:
                return
            if is_db_pressure_active():
                _requeue_touch_published_batch(batch)
                await asyncio.sleep(_TOUCH_PUBLISHED_FLUSH_DELAY_SECONDS)
                continue
            try:
                slugs = await _persist_touch_published_batch(batch)
                for slug in slugs:
                    _invalidate_cache(slug)
            except Exception as exc:  # noqa: BLE001
                maybe_mark_db_pressure(exc, component="topic_catalog.touch_published", ttl_seconds=45.0)
                _requeue_touch_published_batch(batch)
                logger.warning(
                    "topic_catalog.touch_published batch failed",
                    exc_info=exc,
                )
            with _touch_published_lock:
                if not _touch_published_pending:
                    return
    finally:
        current_task = asyncio.current_task()
        with _touch_published_lock:
            if _touch_published_task is current_task:
                _touch_published_task = None


async def touch_replayed(slug: str) -> None:
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(
                _sa_update(TopicCatalog)
                .where(TopicCatalog.slug == slug)
                .values(
                    last_replayed_at=datetime.now(timezone.utc).replace(tzinfo=None),
                )
            )
            await session.commit()
        _invalidate_cache(slug)
    except Exception:  # noqa: BLE001
        logger.warning("topic_catalog.touch_replayed failed for %s", slug, exc_info=True)


# ── Seed topics ──────────────────────────────────────────────────────
#
# These wrap pre-existing storage so the bus is usable on day one.
# Each storage_uri is a JSON string holding adapter-specific config —
# the SQL adapter reads ``{"adapter": "...", "table": "..."}`` from
# storage_uri to know which table to query and which adapter class to
# instantiate.

def _sources(*entries: dict[str, Any]) -> str:
    """Helper: serialise a sources list to JSON for ``storage_uri``.

    Every topic — single or multi source — uses the same schema:

        {"sources": [<source-config>, <source-config>, ...]}

    A source-config is ``{"kind": "...", ...source-specific fields}``.
    The bus's multi_source_replayer iterates this list and dispatches
    each entry to its kind-specific reader.  Single-source topics
    just have ``sources`` of length 1; nothing else is special.
    """
    return json.dumps({"sources": list(entries)})


SEED_TOPICS: tuple[dict[str, Any], ...] = (
    {
        "slug": "polymarket.book.snapshot",
        "title": "Polymarket book snapshots",
        "description": (
            "L2 book snapshots for Polymarket markets.  Top-of-book + "
            "ladders + trade tape per observation.  Sources: live WS "
            "ingest, polybacktest backfill, and operator-imported "
            "Telonex parquet — all unioned by observed_at.  Strategies "
            "subscribe to this one topic regardless of which storage "
            "the bytes happen to live in."
        ),
        # Primary-backing badge for the operator UI.  The actual
        # dispatch reads ``sources`` below; this field is informational
        # ("this topic is primarily backed by sql_table data, with
        # external_parquet supplementing").
        "storage_kind": "sql_table",
        "storage_uri": _sources(
            {
                "kind": "sql_table",
                "adapter": "MarketMicrostructureSnapshot",
                "table": "market_microstructure_snapshots",
                # No provider filter — every importer that writes to
                # this table contributes to this topic.  Provider
                # identity is surfaced per-envelope in the payload.
            },
            {
                "kind": "external_parquet",
                "uri": r"C:\homerun\data\parquet\telonex",
                # Recursive walk picks up every coin subdirectory
                # (btc/eth/sol/pred...).  Future providers under
                # data/parquet/{provider}/{coin}/... can add their
                # own source entry without changing the topic.
            },
        ),
        "publishers": (
            "market_data_ingestor",
            "polybacktest_import_service",
            "telonex_import_service",
        ),
    },
    {
        "slug": "polymarket.book.delta",
        "title": "Polymarket book deltas (per-level changes)",
        "description": (
            "Per-level book changes — single-level event payload "
            "(event_type, side, price, trade_size, cancel_size).  "
            "Different shape from book.snapshot, hence its own topic.  "
            "Backed by book_delta_events (postgres).  7M+ rows."
        ),
        "storage_kind": "sql_table",
        "storage_uri": _sources({
            "kind": "sql_table",
            "adapter": "BookDeltaEvent",
            "table": "book_delta_events",
        }),
        "publishers": ("market_data_ingestor",),
    },
    {
        "slug": "wallet.trade",
        "title": "Wallet monitor trade events",
        "description": (
            "Trade events the ws-monitor records for tracked wallets.  "
            "Different entity (wallet, not token) and different shape "
            "(side, size, tx_hash, ...).  Backed by wallet_monitor_events "
            "(postgres).  2.9M+ rows.  The traders_copy_trade strategy "
            "replays from this topic."
        ),
        "storage_kind": "sql_table",
        "storage_uri": _sources({
            "kind": "sql_table",
            "adapter": "WalletMonitorEvent",
            "table": "wallet_monitor_events",
        }),
        "publishers": ("ws_monitor",),
        "subscribers": ("traders_copy_trade",),
    },
    {
        "slug": "opportunity.detected",
        "title": "Detected scanner opportunities",
        "description": (
            "Every arbitrage opportunity the scanner emits.  Backed by "
            "opportunity_history (postgres).  440K+ rows.  Resolution "
            "is tracked by opportunity_recorder."
        ),
        "storage_kind": "sql_table",
        "storage_uri": _sources({
            "kind": "sql_table",
            "adapter": "OpportunityHistory",
            "table": "opportunity_history",
        }),
        "publishers": ("scanner", "opportunity_recorder"),
    },
)


async def ensure_seed_topics() -> None:
    """Idempotent — registers any seed topics that aren't already in
    the catalog, leaves existing rows untouched (so operator edits
    survive recorder restarts).  Called from app startup."""
    for spec in SEED_TOPICS:
        try:
            await register_topic(upsert=True, **spec)
        except Exception:  # noqa: BLE001
            logger.exception("topic_catalog: failed to seed %s", spec.get("slug"))
