"""The unified recorded-event bus.

Three responsibilities, in order of hot-path frequency:

  1. **Live publish** (10k+ events/sec at peak).  Recorders call
     ``await bus.publish(event)``; the bus validates against the
     topic catalog (cached), fans out to in-process subscribers,
     and queues the event for storage if the topic is persistent.
     Hot path is mostly cache hits + memory allocation; the slow
     stuff (persistence, cross-process delivery) happens out-of-band.

  2. **Live subscribe** (a handful per process at startup).
     Strategies / runtime services call ``bus.subscribe(topic, handler)``
     once at boot.  Handlers run as tasks so a slow handler doesn't
     back up the publish hot path.  Same delivery-once semantics
     the existing ``services.event_dispatcher`` provides — easy
     migration path.

  3. **Backtest replay** (slow, batched).  The runner calls
     ``bus.replay(window, topics)`` to get an async iterator of
     ``RecordedEvent`` time-merged across topics.  Each topic
     resolves to its storage backend (parquet / sql_table / memory)
     via the catalog; the bus heap-merges the per-topic streams so
     subscribers see one ordered sequence.

The bus is intentionally a thin coordination layer.  All the
heavy lifting (parquet writes/reads, SQL adapters) lives in the
``storage`` module so the bus stays small enough to reason about.

The bus is process-local.  Cross-process pub/sub goes through the
existing :mod:`services.event_bus` (the WebSocket fanout) for live
fan-out, and through the storage backend for persistent topics —
process B replays from disk what process A wrote.  No Redis, no
Kafka; this is an institutional-grade trading system, not an event-
sourcing platform, and adding a new infra dep would dwarf the cost
of any latency we'd save.
"""
from __future__ import annotations

import asyncio
import heapq
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import (
    Any, AsyncIterator, Awaitable, Callable, Optional,
    Iterable, Sequence,
)

from services.recorded_event_bus.envelope import (
    RecordedEvent,
    EnvelopeValidationError,
)
from services.recorded_event_bus.catalog import (
    TopicSpec,
    require_topic,
    touch_published,
    touch_replayed,
)

logger = logging.getLogger(__name__)


# ── Subscription handles ────────────────────────────────────────────


SubscriberHandler = Callable[[RecordedEvent], Awaitable[None]]


@dataclass
class Subscription:
    """Returned by ``bus.subscribe``.  Calling ``unsubscribe`` removes
    the handler — call it from teardown so a restarted strategy
    doesn't end up double-subscribed (the same leak class the existing
    services.event_bus had to defend against)."""
    topic: str
    handler: SubscriberHandler
    _bus: "RecordedEventBus"

    def unsubscribe(self) -> None:
        self._bus._unsubscribe_handle(self)


# ── Replay window ───────────────────────────────────────────────────


@dataclass(frozen=True)
class ReplayWindow:
    """A closed-open ``[start_us, end_us)`` time window over a set of
    topics, optionally filtered to specific entity_ids.

    The closed-open convention matches every other range API in the
    codebase (Postgres ``[start, end)`` semantics, parquet row-group
    pruning) so callers don't have to remember a different rule per
    layer.
    """
    start_us: int
    end_us: int
    topics: tuple[str, ...]
    # Optional per-topic entity_id allowlist.  When None, all entities
    # in the topic are streamed.  When provided, only events whose
    # entity_id is in the set are yielded.  Maps topic → frozenset of
    # entity_ids.
    entity_filter: Optional[dict[str, frozenset[str]]] = None
    # Time semantics — either replay-by-truth-time (the default; book
    # snapshots stream in observed_at order) or replay-by-knowledge-time
    # (book snapshots stream in ingested_at order, catching leakage).
    time_field: str = "observed_at_us"

    def __post_init__(self) -> None:
        if self.start_us >= self.end_us:
            raise ValueError(
                f"ReplayWindow start_us={self.start_us} must precede end_us={self.end_us}"
            )
        if not self.topics:
            raise ValueError("ReplayWindow must specify at least one topic")
        if self.time_field not in {"observed_at_us", "ingested_at_us"}:
            raise ValueError(
                f"time_field={self.time_field!r} must be observed_at_us or ingested_at_us"
            )


# ── The bus ─────────────────────────────────────────────────────────


class RecordedEventBus:
    """Process-local recorded-event bus.

    Get the singleton via ``services.recorded_event_bus.bus`` —
    instantiating directly is fine for tests but in production every
    publisher / subscriber should converge on the one instance.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Subscription]] = {}
        self._dispatch_tasks: set[asyncio.Task] = set()
        # Optional storage writer hook — set by services.recorded_event_bus.storage
        # at import time so the bus doesn't have a hard dep on parquet
        # / pyarrow at module-load (which would slow every cold start
        # and prevent strategies from importing the bus without the
        # full data plane).  See storage.attach_writer().
        self._storage_writer: Optional[Callable[[RecordedEvent, TopicSpec], Awaitable[None]]] = None
        self._storage_replayer: Optional[
            Callable[[TopicSpec, ReplayWindow], AsyncIterator[RecordedEvent]]
        ] = None

    # ── Wiring storage backends (lazy import) ──────────────────────

    def _attach_storage(
        self,
        *,
        writer: Callable[[RecordedEvent, TopicSpec], Awaitable[None]],
        replayer: Callable[[TopicSpec, ReplayWindow], AsyncIterator[RecordedEvent]],
    ) -> None:
        """Called by :mod:`services.recorded_event_bus.storage` on
        import.  Keeps the bus free of pyarrow/SQLAlchemy direct
        dependencies."""
        self._storage_writer = writer
        self._storage_replayer = replayer

    # ── Live subscribe ─────────────────────────────────────────────

    def subscribe(
        self,
        topic: str,
        handler: SubscriberHandler,
    ) -> Subscription:
        """Register a handler for live events on ``topic``.

        Idempotent: registering the same (topic, handler) twice
        returns the existing subscription handle without dual-firing
        (same leak guard the WebSocket bus has).
        """
        existing = self._subscribers.setdefault(topic, [])
        for sub in existing:
            if sub.handler == handler:
                logger.debug(
                    "RecordedEventBus: skipped duplicate subscribe to %s",
                    topic,
                )
                return sub
        sub = Subscription(topic=topic, handler=handler, _bus=self)
        existing.append(sub)
        logger.debug(
            "RecordedEventBus: subscribed to %s (n=%d)",
            topic, len(existing),
        )
        return sub

    def _unsubscribe_handle(self, sub: Subscription) -> None:
        bucket = self._subscribers.get(sub.topic, [])
        self._subscribers[sub.topic] = [s for s in bucket if s is not sub]
        if not self._subscribers[sub.topic]:
            self._subscribers.pop(sub.topic, None)

    # ── Live publish ───────────────────────────────────────────────

    async def publish(self, event: RecordedEvent) -> None:
        """Validate, fan out to live subscribers, persist if backing.

        Hot path:
          * topic catalog lookup (cache hit ~1µs)
          * envelope validation already happened at construction
          * subscriber fan-out launches background tasks (no await
            on slow handlers)
          * storage write enqueues (parquet writer is a queued
            flush; SQL-table topics are no-op for writes — the
            existing recorders own those tables)

        Returns once subscribers have been DISPATCHED (not awaited).
        That keeps the producer's loop unblocked at the cost of
        delivery-after-return for slow handlers — the same trade-off
        the existing services.event_bus makes.
        """
        # Validate the envelope is real (defensive — the dataclass
        # validates at construction, but a caller might have monkeyed
        # with the frozen instance via ``object.__setattr__``).
        if not isinstance(event, RecordedEvent):
            raise EnvelopeValidationError(
                f"event must be RecordedEvent, got {type(event).__name__}"
            )

        # Catalog lookup — fail-closed on unregistered topics.
        spec = await require_topic(event.topic)

        # Optional payload-schema validation (currently presence-only;
        # full JSON-Schema validation lands when a topic actually
        # opts in by setting payload_schema_json).  TODO when needed.

        # Fan out to live subscribers (background tasks).
        await self._dispatch(event)

        # Persist if the topic has a writable parquet source.  We
        # resolve the writable source from the topic's sources[] list
        # rather than dispatching on spec.storage_kind, so a topic
        # whose primary badge is sql_table can still have an
        # auxiliary parquet source for new-data archival.  When the
        # writer needs a single-source spec (plain path uri) we hand
        # it the ephemeral source spec, not the parent.
        if self._storage_writer is not None:
            from services.recorded_event_bus.storage.multi_source import resolve_writable_parquet_source
            writable = resolve_writable_parquet_source(spec)
            if writable is not None:
                try:
                    await self._storage_writer(event, writable)
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "RecordedEventBus: storage_writer failed for %s/%s",
                        event.topic, event.entity_id,
                    )

        # Bookkeeping — best-effort, runs as a fire-and-forget task so
        # the publish hot path doesn't block on the catalog UPDATE.
        # Counter semantics:
        #   * parquet topics: the bus owns the storage path, so
        #     ``event_count`` accurately reflects rows written.
        #     ``bytes_on_disk`` is touched by the parquet writer
        #     after the actual flush so it reflects real bytes.
        #   * sql_table topics: the underlying recorder owns the row
        #     writes and we'd double-count if we bumped here.  We do
        #     update ``last_published_at`` so operators can see the
        #     bus is observing live publishes, but we leave
        #     ``event_count`` to come from the catalog's storage-
        #     adapter row count (a future ``recount`` cron) rather
        #     than guess from publish() calls.
        #   * memory topics: nothing to bookkeep — pure fan-out.
        if spec.storage_kind == "parquet":
            t = asyncio.create_task(touch_published(event.topic, n_events=1))
            self._dispatch_tasks.add(t)
            t.add_done_callback(self._dispatch_tasks.discard)
        elif spec.storage_kind == "sql_table":
            # touch_published with n_events=0 only updates the
            # last_published_at timestamp — the count stays accurate
            # against the underlying table.
            t = asyncio.create_task(touch_published(event.topic, n_events=0))
            self._dispatch_tasks.add(t)
            t.add_done_callback(self._dispatch_tasks.discard)

    async def publish_many(self, events: Iterable[RecordedEvent]) -> None:
        """Batch publish — same semantics as a loop over ``publish``
        but the catalog lookup is amortised over the batch."""
        # Group by topic so we hit the catalog cache once per topic
        # rather than once per event.
        by_topic: dict[str, list[RecordedEvent]] = {}
        for e in events:
            by_topic.setdefault(e.topic, []).append(e)
        for topic, batch in by_topic.items():
            spec = await require_topic(topic)
            for e in batch:
                await self._dispatch(e)
            if self._storage_writer is not None:
                from services.recorded_event_bus.storage.multi_source import resolve_writable_parquet_source
                writable = resolve_writable_parquet_source(spec)
                if writable is not None:
                    for e in batch:
                        try:
                            await self._storage_writer(e, writable)
                        except Exception:  # noqa: BLE001
                            logger.exception(
                                "RecordedEventBus.publish_many: storage failed for %s/%s",
                                e.topic, e.entity_id,
                            )
            # Same counter semantics as publish() — see that docstring.
            if spec.storage_kind == "parquet":
                t = asyncio.create_task(
                    touch_published(topic, n_events=len(batch))
                )
                self._dispatch_tasks.add(t)
                t.add_done_callback(self._dispatch_tasks.discard)
            elif spec.storage_kind == "sql_table":
                t = asyncio.create_task(touch_published(topic, n_events=0))
                self._dispatch_tasks.add(t)
                t.add_done_callback(self._dispatch_tasks.discard)

    async def _dispatch(self, event: RecordedEvent) -> None:
        """Fan out to live subscribers as tasks — never awaits a
        handler.  ``*`` wildcard subscribers receive every topic."""
        callbacks: list[SubscriberHandler] = [
            sub.handler for sub in self._subscribers.get(event.topic, [])
        ]
        callbacks.extend(sub.handler for sub in self._subscribers.get("*", []))
        if not callbacks:
            return
        for cb in callbacks:
            task = asyncio.create_task(_safe_invoke(cb, event))
            self._dispatch_tasks.add(task)
            task.add_done_callback(self._dispatch_tasks.discard)

    # ── Replay (backtest path) ─────────────────────────────────────

    async def replay(
        self,
        window: ReplayWindow,
    ) -> AsyncIterator[RecordedEvent]:
        """Time-merge events across the requested topics in
        ``window.time_field`` order.  Yields one envelope at a time
        so callers can interleave with their own work without
        materialising the full window in memory.

        Per topic, the bus delegates to the storage backend to
        produce that topic's stream; the bus heap-merges them.  This
        is exactly the design of :class:`HybridBookSource` generalised
        from "books only" to "every topic in the system".
        """
        if self._storage_replayer is None:
            raise RuntimeError(
                "RecordedEventBus: no storage replayer attached — "
                "import services.recorded_event_bus.storage first"
            )

        # Resolve every topic's spec once before opening any streams
        # so a typo'd topic fails fast rather than after 30s of book
        # streaming.
        specs: dict[str, TopicSpec] = {}
        for topic in window.topics:
            specs[topic] = await require_topic(topic)
            if not specs[topic].is_replayable:
                raise ValueError(
                    f"topic {topic!r} has is_replayable=false in catalog "
                    "(memory-only or ops-disabled); cannot replay"
                )

        # Open per-topic async iterators.  Each one yields envelopes
        # filtered to the window + entity_filter; the bus is
        # responsible only for the merge.
        per_topic_iters: dict[str, AsyncIterator[RecordedEvent]] = {}
        for topic, spec in specs.items():
            per_topic_iters[topic] = self._storage_replayer(spec, window)

        # Heap-merge: prime each iterator's first event, then
        # yield-the-min / refill loop.  Same algorithm as
        # services.backtest.book_replay.HybridBookSource — copying it
        # rather than importing because that one is bound to the
        # BookSnapshot dataclass.
        heap: list[tuple[Any, str]] = []
        head: dict[str, Optional[RecordedEvent]] = {}
        time_attr = window.time_field

        for topic, it in per_topic_iters.items():
            try:
                first = await it.__anext__()
            except StopAsyncIteration:
                first = None
            head[topic] = first
            if first is not None:
                heapq.heappush(
                    heap,
                    (getattr(first, time_attr), topic),
                )

        while heap:
            _ts, topic = heapq.heappop(heap)
            event = head[topic]
            if event is None:
                continue
            yield event
            try:
                nxt = await per_topic_iters[topic].__anext__()
                head[topic] = nxt
                heapq.heappush(
                    heap, (getattr(nxt, time_attr), topic),
                )
            except StopAsyncIteration:
                head[topic] = None

        # Bookkeeping — touch each topic's last_replayed_at.  Done
        # AFTER the iterator drains so partial-replay (caller broke
        # out of the loop early) still updates correctly.  Fire-and-
        # forget; the caller doesn't see this latency.
        for topic in window.topics:
            t = asyncio.create_task(touch_replayed(topic))
            self._dispatch_tasks.add(t)
            t.add_done_callback(self._dispatch_tasks.discard)


async def _safe_invoke(cb: SubscriberHandler, event: RecordedEvent) -> None:
    try:
        await cb(event)
    except Exception:  # noqa: BLE001
        logger.exception(
            "RecordedEventBus subscriber failed for topic=%s",
            event.topic,
        )


# ── Singleton ───────────────────────────────────────────────────────


bus = RecordedEventBus()
"""Process-wide bus instance.  Recorders / strategies / backtest
runner converge on this one object.  Tests can construct their own
``RecordedEventBus()`` for isolation."""
