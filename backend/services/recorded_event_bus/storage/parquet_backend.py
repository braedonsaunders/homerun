"""Parquet writer + reader for bus-native topics.

The parquet backend is the default storage for new topics — cheap to
keep years of history, fast to scan (window, topic) ranges, easy to
back up.  Layout on disk:

    {storage_uri}/                         # configured per topic in catalog
      {entity_id}/                         # token_id, asset_id, wallet, etc.
        {YYYY-MM-DD}/                      # date partition
          events__{batch_id}.parquet

Writes are batched and asynchronous.  The publish hot path enqueues
into an in-memory ring; a background task flushes batches of up to
``_FLUSH_BATCH_SIZE`` events every ``_FLUSH_INTERVAL_SECONDS`` to
parquet.  This trades a tiny window of crash-vulnerability (the
in-memory ring before flush) for hot-path throughput — the same
trade-off the existing :mod:`services.market_data_ingestor` makes
for snapshots / deltas.  Operators who want durability over
throughput can lower ``_FLUSH_INTERVAL_SECONDS`` toward 0 (per-event
flush) at the cost of one parquet write per envelope.

File-level schema:

    topic            string
    entity_id        string
    observed_at_us   int64
    ingested_at_us   int64
    source           string
    sequence         int64 (nullable)
    schema_version   int32
    payload_json     string   -- compact JSON; topic-specific shape

Why JSON for payload rather than per-topic columnar layouts:

  * Payload shapes evolve.  Locking each topic into a fixed
    parquet schema would mean a migration on every payload bump.
  * Replay reads the envelope on every row regardless of payload
    use, so the payload deserialisation is amortised against the
    iteration cost.
  * The system's hottest topic (polymarket.book.snapshot) is NOT
    backed by this writer — its hot path lives in the existing
    market_data_ingestor.  Bus-native topics are the smaller-volume
    ones (crypto.update, news.gdelt, fills, decisions) where JSON
    decode cost is negligible.

Topics whose volume eventually demands columnar payload storage
can opt into a per-topic columnar layout later — the catalog's
``payload_schema_json`` field is the hook.  Out of scope for v1;
the JSON-payload default covers every topic we have today.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

if TYPE_CHECKING:
    from services.recorded_event_bus.bus import ReplayWindow

import pyarrow as pa
import pyarrow.parquet as pq

from services.recorded_event_bus.catalog import TopicSpec, schedule_touch_published
from services.recorded_event_bus.envelope import RecordedEvent

logger = logging.getLogger(__name__)


def _entity_order_key(ev: "RecordedEvent", time_attr: str) -> tuple:
    """Sequence-aware total order for the cross-entity replay merge, so
    same-timestamp events from different partitions yield in the source's
    own order rather than heap-insertion order."""
    return (
        getattr(ev, time_attr),
        ev.sequence if ev.sequence is not None else 0,
        ev.topic,
        ev.entity_id,
    )


# ── Tunables ─────────────────────────────────────────────────────────

# Hot-path → flush cadence trade-offs:
#   * batch size 500 = ~half a second of book events at peak →
#     parquet row-group lower bound is ~5k rows; 500 batches into
#     a single file across a date partition fine.
#   * flush interval 1s = at most 1s of in-memory backlog if the
#     producer stops emitting.  Tighter helps recovery; looser
#     amortises write cost over more events.
#   * max ring size 5000 = drop-oldest on overflow.  Same back-pressure
#     discipline as market_data_ingestor.

_FLUSH_BATCH_SIZE = 500
_FLUSH_INTERVAL_SECONDS = 1.0
_MAX_RING_SIZE = 5000


# ── Bus event schema ─────────────────────────────────────────────────

# One row = one RecordedEvent.  Payload is JSON-encoded so the schema
# stays fixed across topic payload-shape changes.
RECORDED_EVENT_SCHEMA: pa.Schema = pa.schema(
    [
        ("topic",          pa.string()),
        ("entity_id",      pa.string()),
        ("observed_at_us", pa.int64()),
        ("ingested_at_us", pa.int64()),
        ("source",         pa.string()),
        ("sequence",       pa.int64()),
        ("schema_version", pa.int32()),
        ("payload_json",   pa.string()),
    ]
)


# ── Writer ───────────────────────────────────────────────────────────


class _WriteRing:
    """Per-process write ring.  Bounded, drop-oldest on overflow.

    Indexed by ``(topic, entity_id, date_str)`` so each flush batch
    lands in the right partition without re-scanning the ring.
    """

    def __init__(self) -> None:
        # buckets keyed by (topic, entity_id, date_str)
        self._buckets: dict[tuple[str, str, str], list[RecordedEvent]] = defaultdict(list)
        self._total = 0
        # Per-topic storage_uri cache so flushes don't re-query the
        # catalog for every batch.
        self._uri_cache: dict[str, str] = {}

    def offer(self, event: RecordedEvent, *, storage_uri: str) -> None:
        date_str = (
            datetime.fromtimestamp(event.observed_at_us / 1e6, tz=timezone.utc)
            .strftime("%Y-%m-%d")
        )
        key = (event.topic, event.entity_id, date_str)
        self._buckets[key].append(event)
        self._uri_cache.setdefault(event.topic, storage_uri)
        self._total += 1
        # Drop-oldest if we exceed the cap.  Affects whatever bucket
        # currently holds the oldest event — we shrink it by one.
        while self._total > _MAX_RING_SIZE:
            self._drop_one_oldest()

    def _drop_one_oldest(self) -> None:
        if not self._buckets:
            return
        # Cheapest: drop from the largest bucket.  Not strictly
        # FIFO but in practice the largest bucket IS the hot one
        # being produced into right now.
        largest_key = max(self._buckets.keys(), key=lambda k: len(self._buckets[k]))
        bucket = self._buckets[largest_key]
        if bucket:
            bucket.pop(0)
            self._total -= 1
            if not bucket:
                del self._buckets[largest_key]

    def drain_full_batches(self) -> list[tuple[tuple[str, str, str], list[RecordedEvent], str]]:
        """Returns batches ready to flush (any bucket that hit
        _FLUSH_BATCH_SIZE).  Removes them from the ring."""
        out: list[tuple[tuple[str, str, str], list[RecordedEvent], str]] = []
        for key, bucket in list(self._buckets.items()):
            if len(bucket) >= _FLUSH_BATCH_SIZE:
                out.append((key, bucket, self._uri_cache[key[0]]))
                del self._buckets[key]
                self._total -= len(bucket)
        return out

    def drain_all(self) -> list[tuple[tuple[str, str, str], list[RecordedEvent], str]]:
        """Flush every pending bucket regardless of size.  Called on
        the interval timer and at shutdown."""
        out: list[tuple[tuple[str, str, str], list[RecordedEvent], str]] = []
        for key, bucket in list(self._buckets.items()):
            if bucket:
                out.append((key, bucket, self._uri_cache[key[0]]))
        self._buckets.clear()
        self._total = 0
        return out


_ring = _WriteRing()
# Both locks are LAZY-INITIALISED on first await so they bind to the
# active event loop rather than whichever loop was alive at module
# import time.  Without this, pytest's fresh-loop-per-test pattern
# trips ``RuntimeError: Event loop is closed`` on the second flush
# (the lock object holds a dead loop reference).  The lazy pattern
# costs one ``if is None`` per acquire — invisible in profiles.
_flush_lock: Optional[asyncio.Lock] = None
_flush_task: Optional[asyncio.Task] = None
_flush_task_lock: Optional[asyncio.Lock] = None


def _get_flush_lock() -> asyncio.Lock:
    global _flush_lock
    if _flush_lock is None:
        _flush_lock = asyncio.Lock()
    return _flush_lock


def _get_flush_task_lock() -> asyncio.Lock:
    global _flush_task_lock
    if _flush_task_lock is None:
        _flush_task_lock = asyncio.Lock()
    return _flush_task_lock


async def parquet_writer(event: RecordedEvent, spec: TopicSpec) -> None:
    """The bus calls this on every publish() for parquet-backed topics.

    Fast path: enqueue + maybe schedule the flush task.  No disk I/O,
    no pyarrow allocations, no catalog round-trips.
    """
    if not spec.storage_uri:
        raise ValueError(
            f"parquet topic {spec.slug!r} has no storage_uri in catalog"
        )
    _ring.offer(event, storage_uri=spec.storage_uri)
    await _ensure_flush_task_running()


async def _ensure_flush_task_running() -> None:
    """Lazy-start the background flush task.  Idempotent — no harm if
    multiple publishers race to start it."""
    global _flush_task
    if _flush_task is not None and not _flush_task.done():
        return
    async with _get_flush_task_lock():
        if _flush_task is not None and not _flush_task.done():
            return
        _flush_task = asyncio.create_task(_flush_loop(), name="rec-event-bus-flusher")


async def _flush_loop() -> None:
    """Background flusher.  Cancellable via shutdown_parquet_backend()."""
    logger.info("recorded_event_bus parquet flush loop started")
    try:
        while True:
            await asyncio.sleep(_FLUSH_INTERVAL_SECONDS)
            await _flush_once()
    except asyncio.CancelledError:
        # On shutdown, flush everything we have before exiting.
        await _flush_once(drain_all=True)
        raise


async def _flush_once(*, drain_all: bool = False) -> int:
    """Flush full batches (or everything, when drain_all).  Returns the
    number of events written."""
    async with _get_flush_lock():
        batches = (
            _ring.drain_all() if drain_all else _ring.drain_full_batches()
        )
        if not batches:
            return 0
        n_written = 0
        bytes_written = 0
        # Group by topic for the catalog bookkeeping; events still go
        # to their own partition file.
        per_topic_counts: dict[str, tuple[int, int]] = {}
        for (topic, entity_id, date_str), events, storage_uri in batches:
            try:
                written_bytes = await asyncio.to_thread(
                    _write_parquet_batch,
                    storage_uri=storage_uri,
                    topic=topic,
                    entity_id=entity_id,
                    date_str=date_str,
                    events=events,
                )
                n_written += len(events)
                bytes_written += written_bytes
                cur_n, cur_b = per_topic_counts.get(topic, (0, 0))
                per_topic_counts[topic] = (cur_n + len(events), cur_b + written_bytes)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "parquet flush failed for %s/%s/%s (%d events lost)",
                    topic, entity_id, date_str, len(events),
                )
        # Update catalog counters.
        for topic, (_n, b) in per_topic_counts.items():
            schedule_touch_published(topic, n_events=0, bytes_added=b)
        return n_written


def _write_parquet_batch(
    *,
    storage_uri: str,
    topic: str,
    entity_id: str,
    date_str: str,
    events: list[RecordedEvent],
) -> int:
    """Sync writer (runs in thread).  One parquet file per call,
    appended to the partition directory.

    Files are immutable per call — each flush writes a new file with
    a uuid suffix.  Replay reads every file in the partition's
    directory (sorted by name for determinism)."""
    partition_dir = Path(storage_uri) / _safe_segment(entity_id) / date_str
    partition_dir.mkdir(parents=True, exist_ok=True)
    fname = f"events__{int(time.time()*1000):013d}_{uuid.uuid4().hex[:8]}.parquet"
    fpath = partition_dir / fname

    rows = {
        "topic":          [e.topic for e in events],
        "entity_id":      [e.entity_id for e in events],
        "observed_at_us": [e.observed_at_us for e in events],
        "ingested_at_us": [e.ingested_at_us for e in events],
        "source":         [e.source for e in events],
        "sequence":       [e.sequence for e in events],
        "schema_version": [e.schema_version for e in events],
        "payload_json":   [json.dumps(dict(e.payload), separators=(",", ":")) for e in events],
    }
    table = pa.table(rows, schema=RECORDED_EVENT_SCHEMA)
    # Atomic-ish: write to .tmp then rename so a concurrent reader
    # doesn't see a half-written file.
    tmp = fpath.with_suffix(".parquet.tmp")
    pq.write_table(table, str(tmp), compression="snappy")
    os.replace(tmp, fpath)
    return fpath.stat().st_size


def _safe_segment(value: str, max_len: int = 64) -> str:
    """Filesystem-safe segment.  Mirrors the policy in parquet_schema
    so the layouts compose."""
    import hashlib
    import re
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "-", value or "")
    if len(cleaned) <= max_len:
        return cleaned
    h = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[:max_len-9]}-{h}"


# ── Reader (replay path) ─────────────────────────────────────────────


async def parquet_replayer(
    spec: TopicSpec,
    window: "ReplayWindow",  # noqa: F821  -- forward ref (bus.py)
) -> AsyncIterator[RecordedEvent]:
    """Yield events from disk in time order for one topic.

    Layout on disk:
        ``{storage_uri}/{entity_id}/{YYYY-MM-DD}/events__*.parquet``

    Within a single partition file events are time-sorted by the
    writer.  Across files in the same (entity, date) we sort by
    filename (writer encodes a timestamp prefix, so filename order =
    write order = time order modulo the per-batch interleave, which
    is small enough that the in-file re-sort handles it).  Across
    entities we MUST heap-merge — yielding one entity's whole stream
    before another would violate the bus's "events in time order"
    contract.

    Memory bounded: each entity has at most one open partition file
    in memory at a time.  Total open files = number of matching
    entities, which the entity_filter typically caps.
    """
    import heapq

    from services.recorded_event_bus.bus import ReplayWindow  # local import to break cycle

    if not isinstance(window, ReplayWindow):
        raise TypeError("window must be ReplayWindow")
    if not spec.storage_uri:
        return

    base = Path(spec.storage_uri)
    if not base.exists():
        return

    # Date partitions touching the window.  Pad by one day on each
    # side to cover events whose observed_at_us straddles a day
    # boundary in another timezone (defensive — writer always uses
    # UTC, but readers might be asked for events with a non-UTC
    # window stamp).
    start_day = datetime.fromtimestamp(window.start_us / 1e6, tz=timezone.utc).date()
    end_day = datetime.fromtimestamp(window.end_us / 1e6, tz=timezone.utc).date()
    from datetime import timedelta as _td
    days: list[str] = []
    d = start_day - _td(days=1)
    while d <= end_day + _td(days=1):
        days.append(d.strftime("%Y-%m-%d"))
        d = d + _td(days=1)

    # Resolve entity dirs.
    entity_set: Optional[frozenset[str]] = None
    if window.entity_filter is not None:
        entity_set = window.entity_filter.get(spec.slug)
    if entity_set is not None:
        entity_dirs = [
            base / _safe_segment(ent) for ent in entity_set
        ]
        entity_dirs = [p for p in entity_dirs if p.exists()]
    else:
        entity_dirs = [p for p in sorted(base.iterdir()) if p.is_dir()]

    # Build one async iterator per entity that walks its date
    # partitions and files in order, yielding RecordedEvent.  These
    # streams are individually time-sorted (writer guarantee +
    # per-file sort), so the bus-level heap-merge across them gives
    # a globally time-sorted stream within the topic.
    async def _entity_stream(ed: Path) -> AsyncIterator[RecordedEvent]:
        for day in days:
            day_dir = ed / day
            if not day_dir.exists():
                continue
            for fp in sorted(day_dir.glob("events__*.parquet")):
                try:
                    rows = await asyncio.to_thread(_read_file_rows, fp, window)
                except Exception:  # noqa: BLE001
                    logger.exception("parquet replay: failed to read %s", fp)
                    continue
                for ev in rows:
                    yield ev

    if not entity_dirs:
        return

    iterators: dict[str, AsyncIterator[RecordedEvent]] = {
        str(ed): _entity_stream(ed) for ed in entity_dirs
    }
    head: dict[str, Optional[RecordedEvent]] = {}
    heap: list[tuple[Any, str]] = []
    time_attr = window.time_field

    # Prime each entity stream's first event.
    for key, it in iterators.items():
        try:
            first = await it.__anext__()
        except StopAsyncIteration:
            first = None
        head[key] = first
        if first is not None:
            heapq.heappush(heap, (_entity_order_key(first, time_attr), key))

    while heap:
        _ord, key = heapq.heappop(heap)
        ev = head[key]
        if ev is None:
            continue
        yield ev
        try:
            nxt = await iterators[key].__anext__()
            head[key] = nxt
            heapq.heappush(heap, (_entity_order_key(nxt, time_attr), key))
        except StopAsyncIteration:
            head[key] = None


def _read_file_rows(fp: Path, window: "ReplayWindow") -> list[RecordedEvent]:
    """Sync helper: read one parquet file and project rows whose
    timestamp falls in the window.  Returns a list (small per file
    in practice — _FLUSH_BATCH_SIZE = 500 rows) so the async iterator
    can yield one-by-one upstream."""
    try:
        table = pq.read_table(str(fp))
    except Exception:
        return []
    cols = table.to_pydict()
    n = len(cols.get("topic", []))
    if n == 0:
        return []
    time_field = window.time_field
    if time_field not in cols:
        # Old file lacking ingested_at — fall back to observed_at.
        time_field = "observed_at_us"
    times = cols[time_field]
    out: list[RecordedEvent] = []
    for i in range(n):
        t = int(times[i])
        if t < window.start_us or t >= window.end_us:
            continue
        try:
            payload = json.loads(cols["payload_json"][i] or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = {}
        seq = cols["sequence"][i]
        try:
            ev = RecordedEvent(
                topic=cols["topic"][i],
                entity_id=cols["entity_id"][i],
                observed_at_us=int(cols["observed_at_us"][i]),
                ingested_at_us=int(cols["ingested_at_us"][i]),
                source=cols["source"][i] or "unknown",
                sequence=int(seq) if seq is not None else None,
                schema_version=int(cols["schema_version"][i] or 1),
                payload=payload,
            )
        except Exception:  # noqa: BLE001  -- defensive on bad rows
            logger.debug("parquet replay: skipped invalid row #%d in %s", i, fp)
            continue
        out.append(ev)
    # Per-file sort (defensive — writes are already ordered).
    out.sort(key=lambda e: (getattr(e, time_field), e.sequence or 0))
    return out


# ── Lifecycle ────────────────────────────────────────────────────────


async def flush_pending_writes() -> int:
    """Operator-callable forced flush (e.g. from a /healthz endpoint
    that wants to verify durability before failing the pod over).
    Returns the number of events written.
    """
    return await _flush_once(drain_all=True)


async def shutdown_parquet_backend() -> None:
    """Stop the background flusher and write everything in the ring.
    Called from app shutdown hooks."""
    global _flush_task
    if _flush_task is None:
        return
    _flush_task.cancel()
    try:
        await _flush_task
    except (asyncio.CancelledError, BaseException):
        pass
    _flush_task = None
