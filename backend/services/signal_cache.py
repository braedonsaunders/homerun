"""Process-wide in-memory signal cache hydrated from Redis.

Architecture
------------
The fast trader runtime previously fetched unconsumed trade signals from
Postgres on EVERY cycle (``list_unconsumed_trade_signals``).  Under
contention the query took 1-7s, blowing the 3s fast-tier budget and
serializing the entire hot path on the database.

This module replaces that DB poll with a Redis-pushed in-memory cache:

  scanner / news / strategy bridge
        │
        ▼
  signal_bus.upsert_trade_signal
        ├─ DB INSERT (audit, source of truth)
        └─ Redis publish "signal_payloads" (full snapshot, ~1ms)
                          │
                          ▼
                    SignalCache subscriber
                       (this module)
                          │
                          ▼
                  Process-wide LRU cache
                          │
                          ▼
              fast_trader_runtime reads via
              get_unconsumed_signals(...)
              (no DB roundtrip)

The cache is the FAST PATH; DB is the SLOW PATH safety net.  When the
cache has the signals (which is the case under steady state), the fast
trader skips the DB query entirely.  When the cache misses (cold start,
Redis down, race with publisher), the caller falls back to the DB.

Per-trader consumption tracking
-------------------------------
``list_unconsumed_trade_signals``'s NOT EXISTS subquery is replaced by
an in-memory per-trader consumed-signal-id ring.  Hydrated lazily from
the DB on first use (last 24h of consumptions), updated on every
``mark_consumed`` call.  Bounded ring per trader so memory is O(N
traders × 1000) — kilobytes.

Soft-fail contract
------------------
* Redis down: subscriber sleeps and retries; cache stops growing but
  existing entries remain valid for the freshness window.  Fast trader
  detects empty/stale cache and falls back to DB.
* Cache miss for a specific signal_id: caller falls back to DB.
* Process restart: cache is empty until the subscriber reconnects and
  the publisher emits new signals; bootstrap hydration fills it from
  DB in the meantime.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from services import redis_client
from utils.logger import get_logger
from utils.utcnow import utcnow

logger = get_logger("signal_cache")

# Channel name used by both publisher and subscriber.  See
# ``services/signal_bus.py::_publish_signal_payload`` for the publisher.
SIGNAL_PAYLOADS_CHANNEL = "signal_payloads"

# Bounded LRU caps.  At 10 K signals × ~500 bytes each, cache memory
# stays under ~5 MB — fine for a worker process.  Per-trader consumed
# rings hold at most 1 K recent signal_ids each, ~50 bytes per entry.
_MAX_SIGNAL_CACHE_ENTRIES = 10_000
_MAX_CONSUMED_RING_PER_TRADER = 1_000


@dataclass(slots=True)
class SignalSnapshot:
    """Duck-typed mirror of ``TradeSignal`` for fast-trader consumption.

    All fields the fast trader reads via ``getattr(signal, ...)`` are
    present.  Heavy JSON columns (``payload_json``,
    ``strategy_context_json``, ``quality_rejection_reasons``) are NOT
    included — the fast trader doesn't read them, and shipping them
    over Redis on every signal is wasted bandwidth.
    """

    id: str
    source: str
    source_item_id: Optional[str]
    signal_type: str
    strategy_type: Optional[str]
    market_id: str
    market_question: Optional[str]
    direction: Optional[str]
    entry_price: Optional[float]
    effective_price: Optional[float]
    edge_percent: Optional[float]
    confidence: Optional[float]
    liquidity: Optional[float]
    expires_at: Optional[datetime]
    status: str
    quality_passed: Optional[bool]
    dedupe_key: str
    runtime_sequence: Optional[int]
    created_at: datetime
    updated_at: Optional[datetime]
    # Time the snapshot was placed in the cache (monotonic seconds).
    # Used for freshness checks and LRU eviction.
    _cached_at_mono: float = field(default_factory=time.monotonic)

    @classmethod
    def from_redis_payload(cls, payload: dict[str, Any]) -> Optional["SignalSnapshot"]:
        """Build a snapshot from the JSON dict published over Redis.

        Returns None if the payload is missing required fields — the
        caller should drop and continue.  Datetime strings are parsed
        back into ``datetime`` instances so consumers can compare them
        against ``utcnow()`` directly.
        """
        try:
            signal_id = str(payload.get("id") or "").strip()
            if not signal_id:
                return None
            source = str(payload.get("source") or "").strip()
            if not source:
                return None
            return cls(
                id=signal_id,
                source=source,
                source_item_id=_str_or_none(payload.get("source_item_id")),
                signal_type=str(payload.get("signal_type") or "").strip(),
                strategy_type=_str_or_none(payload.get("strategy_type")),
                market_id=str(payload.get("market_id") or "").strip(),
                market_question=_str_or_none(payload.get("market_question")),
                direction=_str_or_none(payload.get("direction")),
                entry_price=_float_or_none(payload.get("entry_price")),
                effective_price=_float_or_none(payload.get("effective_price")),
                edge_percent=_float_or_none(payload.get("edge_percent")),
                confidence=_float_or_none(payload.get("confidence")),
                liquidity=_float_or_none(payload.get("liquidity")),
                expires_at=_dt_or_none(payload.get("expires_at")),
                status=str(payload.get("status") or "pending").strip().lower(),
                quality_passed=_bool_or_none(payload.get("quality_passed")),
                dedupe_key=str(payload.get("dedupe_key") or "").strip(),
                runtime_sequence=_int_or_none(payload.get("runtime_sequence")),
                created_at=_dt_or_none(payload.get("created_at")) or utcnow(),
                updated_at=_dt_or_none(payload.get("updated_at")),
            )
        except Exception as exc:
            logger.debug("SignalSnapshot.from_redis_payload failed: %s", exc)
            return None


def _str_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float_or_none(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _dt_or_none(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, ValueError, OSError):
            return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# The cache itself.
# ---------------------------------------------------------------------------


class SignalCache:
    """LRU cache of signals + per-trader consumed-id rings.

    All mutation goes through ``self._lock`` (a threading.RLock) so the
    Redis subscriber task and the fast trader's read path can call it
    concurrently.  Read-side operations are sub-microsecond dict
    lookups.
    """

    def __init__(self, max_entries: int = _MAX_SIGNAL_CACHE_ENTRIES) -> None:
        self._lock = threading.RLock()
        # OrderedDict for O(1) LRU semantics.
        self._signals: OrderedDict[str, SignalSnapshot] = OrderedDict()
        self._max_entries = max_entries
        # Per-trader consumed signal_id ring.  ``deque(maxlen=N)`` for
        # bounded memory; companion set for O(1) "is consumed" check.
        self._consumed_ids: dict[str, deque[str]] = {}
        self._consumed_set: dict[str, set[str]] = {}
        # Diagnostic counters.
        self._signals_added: int = 0
        self._signals_evicted: int = 0
        self._consumptions_recorded: int = 0
        self._lookups_total: int = 0
        self._lookups_hit: int = 0
        # Timestamp of the last subscriber message — surfaced in
        # ``status_snapshot()`` so operators can see if the cache is
        # actively being fed.
        self._last_received_mono: Optional[float] = None
        # Whether the per-trader consumption sets have been hydrated
        # from DB.  Bootstrap path sets this to True once done.
        self._consumed_hydrated: set[str] = set()

    # ---------- Mutation (subscriber side) ----------

    def upsert(self, snapshot: SignalSnapshot) -> None:
        """Insert or refresh a signal in the cache."""
        with self._lock:
            self._signals[snapshot.id] = snapshot
            # Move-to-end so LRU eviction picks the oldest unused entry.
            self._signals.move_to_end(snapshot.id)
            self._signals_added += 1
            self._last_received_mono = time.monotonic()
            # Bounded eviction.
            while len(self._signals) > self._max_entries:
                self._signals.popitem(last=False)
                self._signals_evicted += 1

    def mark_consumed(self, trader_id: str, signal_id: str) -> None:
        """Record that a trader consumed a signal.

        Hot-path callers SHOULD call this after a successful consumption
        write to the DB so subsequent ``get_unconsumed_signals`` calls
        skip the signal without hitting the DB.
        """
        if not trader_id or not signal_id:
            return
        with self._lock:
            ring = self._consumed_ids.get(trader_id)
            consumed = self._consumed_set.get(trader_id)
            if ring is None:
                ring = deque(maxlen=_MAX_CONSUMED_RING_PER_TRADER)
                self._consumed_ids[trader_id] = ring
                consumed = set()
                self._consumed_set[trader_id] = consumed
            if signal_id in consumed:
                return
            if len(ring) >= ring.maxlen:
                old = ring[0]
                consumed.discard(old)
            ring.append(signal_id)
            consumed.add(signal_id)
            self._consumptions_recorded += 1

    def hydrate_trader_consumed_ids(
        self,
        trader_id: str,
        signal_ids: Iterable[str],
    ) -> None:
        """Bulk-load consumed signal_ids for a trader from a DB query.

        Idempotent: re-hydrating a trader simply refreshes the ring.
        """
        if not trader_id:
            return
        with self._lock:
            ring = deque(
                (str(sid) for sid in signal_ids if sid),
                maxlen=_MAX_CONSUMED_RING_PER_TRADER,
            )
            self._consumed_ids[trader_id] = ring
            self._consumed_set[trader_id] = set(ring)
            self._consumed_hydrated.add(trader_id)

    def is_trader_hydrated(self, trader_id: str) -> bool:
        with self._lock:
            return trader_id in self._consumed_hydrated

    # ---------- Read (fast-trader side) ----------

    def get_unconsumed_signals(
        self,
        *,
        trader_id: str,
        sources: Optional[Iterable[str]] = None,
        cursor_runtime_sequence: Optional[int] = None,
        statuses: Optional[Iterable[str]] = None,
        limit: int = 200,
    ) -> list[SignalSnapshot]:
        """Return signals matching the filter that this trader hasn't consumed.

        Filters applied in order (cheapest first):
          * status in ``statuses`` (default: ``{"pending"}``)
          * source in ``sources`` (if provided)
          * runtime_sequence > ``cursor_runtime_sequence`` (if provided)
          * not in this trader's consumed-set
          * expires_at not in the past (if set)

        Sorted by ``(runtime_sequence asc, created_at asc)`` so the
        fast trader processes signals in arrival order.
        """
        normalized_statuses = (
            {str(s).strip().lower() for s in statuses}
            if statuses is not None
            else {"pending"}
        )
        normalized_sources = (
            {str(s).strip().lower() for s in sources}
            if sources is not None
            else None
        )
        now = utcnow()
        results: list[SignalSnapshot] = []
        with self._lock:
            self._lookups_total += 1
            consumed = self._consumed_set.get(trader_id) or frozenset()
            for snapshot in self._signals.values():
                if snapshot.status not in normalized_statuses:
                    continue
                if normalized_sources is not None and snapshot.source.lower() not in normalized_sources:
                    continue
                if cursor_runtime_sequence is not None:
                    seq = snapshot.runtime_sequence
                    if seq is not None and seq <= cursor_runtime_sequence:
                        continue
                if snapshot.id in consumed:
                    continue
                if snapshot.expires_at is not None and snapshot.expires_at < now:
                    continue
                results.append(snapshot)
            if results:
                self._lookups_hit += 1
        # Sort outside the lock — cheap, ascending by sequence then time.
        results.sort(
            key=lambda s: (
                s.runtime_sequence if s.runtime_sequence is not None else 0,
                s.created_at,
                s.id,
            )
        )
        return results[: max(1, limit)]

    def get_signal(self, signal_id: str) -> Optional[SignalSnapshot]:
        with self._lock:
            return self._signals.get(signal_id)

    # ---------- Diagnostics ----------

    def status_snapshot(self) -> dict[str, Any]:
        with self._lock:
            age = (
                None
                if self._last_received_mono is None
                else round(time.monotonic() - self._last_received_mono, 3)
            )
            hit_rate = (
                round(self._lookups_hit / self._lookups_total, 3)
                if self._lookups_total > 0
                else None
            )
            return {
                "size": len(self._signals),
                "max_entries": self._max_entries,
                "signals_added_total": self._signals_added,
                "signals_evicted_total": self._signals_evicted,
                "consumptions_recorded_total": self._consumptions_recorded,
                "lookups_total": self._lookups_total,
                "lookups_hit": self._lookups_hit,
                "hit_rate": hit_rate,
                "last_received_age_seconds": age,
                "traders_hydrated": len(self._consumed_hydrated),
                "channel": SIGNAL_PAYLOADS_CHANNEL,
            }


_cache: Optional[SignalCache] = None
_cache_lock = threading.Lock()


def get_signal_cache() -> SignalCache:
    """Process-wide singleton."""
    global _cache
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                _cache = SignalCache()
    return _cache


# ---------------------------------------------------------------------------
# Subscriber task (trading plane).
# ---------------------------------------------------------------------------


class _Subscriber:
    def __init__(self) -> None:
        self._task: Optional[asyncio.Task] = None
        self._stop_event: Optional[asyncio.Event] = None

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="signal_cache_subscriber")

    async def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._task is not None and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except (asyncio.CancelledError, Exception):
                    pass
        self._task = None
        self._stop_event = None

    async def _run(self) -> None:
        stop_event = self._stop_event
        assert stop_event is not None
        cache = get_signal_cache()
        backoff = 1.0
        while not stop_event.is_set():
            client = redis_client.get_client_or_none()
            if client is None:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass
                continue
            pubsub = None
            try:
                pubsub = client.pubsub()
                channel = redis_client.namespaced(SIGNAL_PAYLOADS_CHANNEL)
                await pubsub.subscribe(channel)
                logger.info("signal_cache subscribed", channel=channel)
                backoff = 1.0
                while not stop_event.is_set():
                    try:
                        message = await pubsub.get_message(
                            ignore_subscribe_messages=True,
                            timeout=2.0,
                        )
                    except asyncio.TimeoutError:
                        message = None
                    if message is None:
                        continue
                    if message.get("type") != "message":
                        continue
                    data = message.get("data")
                    if isinstance(data, (bytes, bytearray)):
                        data = data.decode("utf-8", errors="replace")
                    try:
                        payload = json.loads(data) if data else None
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
                    if not isinstance(payload, dict):
                        continue
                    snapshot = SignalSnapshot.from_redis_payload(payload)
                    if snapshot is not None:
                        cache.upsert(snapshot)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("signal_cache subscriber error: %s", exc)
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2.0, 15.0)
            finally:
                if pubsub is not None:
                    try:
                        await pubsub.aclose()
                    except Exception:
                        pass


_subscriber = _Subscriber()


async def start_subscriber() -> None:
    """Start the Redis subscriber.  Trading plane only."""
    await _subscriber.start()


async def stop_subscriber() -> None:
    await _subscriber.stop()


def status_snapshot() -> dict[str, Any]:
    return get_signal_cache().status_snapshot()
