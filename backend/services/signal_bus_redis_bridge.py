"""Cross-plane → in-process bridge for trade-signal arrival events.

Why
---
The trader orchestrator (and fast_trader_runtime) wake up on
``event_bus`` events emitted by ``services.signal_bus`` whenever a new
trade signal is upserted in the same process.  Signals upserted in a
*different* process (e.g. the news plane writes a signal via
``strategy_signal_bridge``) never reach this in-process bus, so the
orchestrator must wait for its 60s polling fallback to discover the
DB row.

This bridge subscribes to the Redis pub/sub channels that
``signal_bus._publish_redis`` writes on every emission, and re-publishes
them onto the local ``event_bus`` so local subscribers wake immediately.

Idempotency / dedup
-------------------
The trading plane is BOTH a producer and a consumer of these Redis
events.  Without dedup, every locally-emitted signal would fire its
in-process event_bus event AND, ~600 us later, fire it again from
the Redis bridge — doubling work.  We dedup by signal_id over a small
FIFO ring of recently-seen IDs.

Lifecycle
---------
* Trading plane only — not started on the API or news plane.  Started
  from ``workers/host.py::_initialize_services`` after Redis comes up.
* Soft-fail: if Redis is down, ``redis_client.get_client_or_none()``
  returns None on subsequent reconnect attempts; the bridge sleeps and
  retries with exponential backoff.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from typing import Any, Optional

from services import redis_client
from services.event_bus import event_bus
from services.runtime_signal_queue import publish_signal_batch
from services.signal_bus import SIGNAL_BATCH_CHANNEL, SIGNAL_EMISSION_CHANNEL
from utils.logger import get_logger

logger = get_logger("signal_bus_redis_bridge")

# Bounded ring of signal_ids the local plane already published in the
# last few seconds.  A signal_id seen here is dropped from the Redis
# bridge to prevent double-firing.
_DEDUP_WINDOW_SIZE = 2048


class _SignalDedup:
    __slots__ = ("_ids", "_set", "_lock")

    def __init__(self) -> None:
        self._ids: deque[str] = deque(maxlen=_DEDUP_WINDOW_SIZE)
        self._set: set[str] = set()
        self._lock = asyncio.Lock()

    async def add(self, signal_id: str) -> None:
        if not signal_id:
            return
        async with self._lock:
            if signal_id in self._set:
                return
            if len(self._ids) >= _DEDUP_WINDOW_SIZE:
                old = self._ids.popleft()
                self._set.discard(old)
            self._ids.append(signal_id)
            self._set.add(signal_id)

    async def seen(self, signal_id: str) -> bool:
        if not signal_id:
            return False
        async with self._lock:
            return signal_id in self._set


_dedup = _SignalDedup()


async def mark_locally_emitted(signal_ids: list[str] | str) -> None:
    """Record signal_ids as locally-emitted so the Redis bridge skips them.

    Subscribers in the same plane that publish to event_bus locally
    should call this so the cross-plane bridge doesn't re-fire the same
    signal a few microseconds later.
    """
    if isinstance(signal_ids, str):
        await _dedup.add(signal_ids)
        return
    for sid in signal_ids:
        await _dedup.add(sid)


# ---------------------------------------------------------------------------
# Bridge task.
# ---------------------------------------------------------------------------


class _Bridge:
    def __init__(self) -> None:
        self._task: Optional[asyncio.Task] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._messages_received: int = 0
        self._messages_bridged: int = 0
        self._messages_skipped_dup: int = 0
        self._last_message_mono: Optional[float] = None

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="signal_bus_redis_bridge")

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

    def status_snapshot(self) -> dict[str, Any]:
        age = (
            None
            if self._last_message_mono is None
            else round(time.monotonic() - self._last_message_mono, 3)
        )
        return {
            "running": self._task is not None and not self._task.done(),
            "messages_received": self._messages_received,
            "messages_bridged": self._messages_bridged,
            "messages_skipped_dup": self._messages_skipped_dup,
            "last_message_age_seconds": age,
        }

    async def _run(self) -> None:
        stop_event = self._stop_event
        assert stop_event is not None
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
                emission_ch = redis_client.namespaced(SIGNAL_EMISSION_CHANNEL)
                batch_ch = redis_client.namespaced(SIGNAL_BATCH_CHANNEL)
                await pubsub.subscribe(emission_ch, batch_ch)
                logger.info(
                    "signal_bus_redis_bridge subscribed",
                    emission_channel=emission_ch,
                    batch_channel=batch_ch,
                )
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
                    self._messages_received += 1
                    self._last_message_mono = time.monotonic()
                    channel = message.get("channel")
                    if isinstance(channel, (bytes, bytearray)):
                        channel = channel.decode("utf-8", errors="replace")
                    data = message.get("data")
                    if isinstance(data, (bytes, bytearray)):
                        data = data.decode("utf-8", errors="replace")
                    try:
                        payload = json.loads(data) if data else None
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
                    if not isinstance(payload, dict):
                        continue
                    if channel == emission_ch:
                        await self._bridge_emission(payload)
                    elif channel == batch_ch:
                        await self._bridge_batch(payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("signal_bus_redis_bridge error: %s", exc)
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

    async def _wake_runtime_queue(
        self,
        *,
        event_type: Any,
        source: Any,
        signal_ids: list[str],
        trigger: Any,
        reason: Any = None,
        emitted_at: Any = None,
    ) -> None:
        """Wake the in-process ``runtime_signal_queue`` so the trader
        orchestrator cycles immediately on a cross-plane signal.

        The orchestrator awaits the lane queue (``wait_for_signal_batch``),
        not the ``event_bus``, so re-injecting onto the bus alone never wakes
        it — it only wakes ``event_bus`` subscribers (e.g. the fast trader).
        Before detection/scanner moved to their own plane, the orchestrator
        was woken in-process by that plane's ``publish_signal_batch``; with
        the producer in another process its queue had no wake source.

        ``publish_signal_batch``'s ``_is_actionable_event`` gate filters
        non-actionable emissions (``status_update``/``status_expired``), so
        only ``upsert_*``/``reverse_entry`` batches actually trigger a cycle.
        Trading plane only — the bridge runs nowhere else, and only the
        trading plane consumes the queue.
        """
        ids = [str(s) for s in signal_ids if s]
        if not ids:
            return
        try:
            await publish_signal_batch(
                event_type=str(event_type or ""),
                source=source,
                signal_ids=ids,
                trigger=str(trigger or "signal_bus"),
                reason=reason,
                emitted_at=emitted_at,
            )
        except Exception as exc:
            logger.debug("runtime_signal_queue wake from bridge failed: %s", exc)

    async def _bridge_emission(self, payload: dict[str, Any]) -> None:
        signal_id = str(payload.get("signal_id") or "").strip()
        if signal_id and await _dedup.seen(signal_id):
            self._messages_skipped_dup += 1
            return
        if signal_id:
            await _dedup.add(signal_id)
        try:
            await event_bus.publish("trade_signal_emission", payload)
            self._messages_bridged += 1
        except Exception as exc:
            logger.debug("event_bus.publish for emission bridge failed: %s", exc)
        await self._wake_runtime_queue(
            event_type=payload.get("event_type"),
            source=payload.get("source"),
            signal_ids=[signal_id] if signal_id else [],
            trigger=payload.get("trigger"),
            reason=payload.get("reason"),
            emitted_at=payload.get("updated_at"),
        )

    async def _bridge_batch(self, payload: dict[str, Any]) -> None:
        signal_ids_raw = payload.get("signal_ids") or []
        signal_ids = [str(s) for s in signal_ids_raw if s]
        if signal_ids:
            unseen: list[str] = []
            for sid in signal_ids:
                if not await _dedup.seen(sid):
                    unseen.append(sid)
            if not unseen:
                self._messages_skipped_dup += 1
                return
            for sid in unseen:
                await _dedup.add(sid)
            payload = dict(payload)
            payload["signal_ids"] = unseen
            payload["signal_count"] = len(unseen)
        try:
            await event_bus.publish("trade_signal_batch", payload)
            self._messages_bridged += 1
        except Exception as exc:
            logger.debug("event_bus.publish for batch bridge failed: %s", exc)


_bridge = _Bridge()


async def start() -> None:
    """Start the bridge task.  Trading plane only."""
    await _bridge.start()


async def stop() -> None:
    await _bridge.stop()


def status_snapshot() -> dict[str, Any]:
    return _bridge.status_snapshot()
