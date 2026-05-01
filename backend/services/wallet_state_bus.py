"""Cross-process wallet-state visibility via Redis pub/sub.

Architecture
------------
Only the trading plane runs the Polymarket user-channel WS feed (see
``services/polymarket_user_feed.py``); the API plane and the news plane
have no direct view of WS-driven wallet deltas.  Before Redis was
reintroduced, the API exposed wallet-cache freshness as **its own**
``WalletStateCache`` instance — which was always cold (no WS feed in
that process), so the surface lied.

This module fixes that with a TWO-CHANNEL design:

  1. **Delta channel** (event-driven, sub-millisecond):
     ``wallet_state.changed`` events fire on every ``apply_trade`` /
     ``apply_order`` / ``seed_from_rest`` mutation.  The cache calls
     ``publish_delta(...)`` directly; the API plane subscribes and
     updates its mirror immediately.  This is the path that drives
     real-time visibility — no polling, no cadence.

  2. **Heartbeat channel** (periodic liveness, 2s cadence):
     ``stats_snapshot()`` published every 2s.  Used solely as a
     *liveness* signal: if the API plane stops receiving heartbeats,
     the trading plane is presumed down.  Not a freshness mechanism.

Soft-fail contract
------------------
* If Redis is disabled or unreachable, every publish is a no-op; the
  subscriber sleeps and retries; ``get_latest_heartbeat()`` /
  ``get_latest_delta()`` return ``None``.  No path can wedge the
  trading plane or the API.
* All publishes run as ``loop.create_task`` so the wallet-cache lock
  is held for nanoseconds, not for the duration of a Redis round trip.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional

from services import redis_client
from services.wallet_state_cache import get_wallet_state_cache
from utils.logger import get_logger
from utils.utcnow import utcnow

logger = get_logger("wallet_state_bus")

# Channel + key names.  Use a single channel per topic — namespacing is
# applied by ``redis_client.namespaced()`` so a multi-tenant Redis is safe.
#
# DELTA channel: pushed on every cache mutation; sub-millisecond delivery.
# HEARTBEAT channel: periodic liveness probe; NOT a freshness mechanism.
WALLET_STATE_DELTA_CHANNEL = "wallet_state:delta"
WALLET_STATE_HEARTBEAT_CHANNEL = "wallet_state:heartbeat"
WALLET_STATE_HEARTBEAT_LAST_KEY = "wallet_state:heartbeat:last"

# Heartbeat cadence.  Liveness signal only — actual wallet deltas flow on
# the delta channel within microseconds of the WS event.  Slow enough that
# even a degraded Redis can keep up.
_HEARTBEAT_INTERVAL_SECONDS = 2.0

# Heartbeat staleness threshold for the API-side ``is_stale()`` helper.
# A heartbeat older than this is treated as "trading plane down".
_HEARTBEAT_STALE_AFTER_SECONDS = 15.0


class _DeltaCounter:
    """Process-local counter of deltas emitted by ``publish_delta``.

    Surfaced in the heartbeat payload so the subscriber can detect
    pub/sub message loss (e.g. heartbeat says 50 deltas emitted but
    subscriber only consumed 48).
    """

    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value: int = 0

    def bump(self) -> None:
        self.value += 1


_delta_counter = _DeltaCounter()


# ---------------------------------------------------------------------------
# Delta publish — called inline from WalletStateCache mutations.
# ---------------------------------------------------------------------------


async def publish_delta(payload: dict[str, Any]) -> None:
    """Publish a wallet-state delta to subscribers across processes.

    Called from ``WalletStateCache._emit_change`` on every apply_trade /
    apply_order / seed_from_rest / ws_state mutation.  Soft-fail: if
    Redis is down, the publish is silently dropped (the trading plane's
    in-process event_bus already woke local subscribers; cross-process
    visibility just degrades to the heartbeat fallback).
    """
    _delta_counter.bump()
    client = redis_client.get_client_or_none()
    if client is None:
        return
    body = {
        "type": "wallet_state_delta",
        "ts": utcnow().isoformat(),
        "monotonic": time.monotonic(),
        "seq": _delta_counter.value,
        "payload": payload,
    }
    try:
        await client.publish(
            redis_client.namespaced(WALLET_STATE_DELTA_CHANNEL),
            json.dumps(body, default=str),
        )
    except Exception as exc:
        logger.debug("publish_delta failed: %s", exc)


# ---------------------------------------------------------------------------
# Publisher (runs in the trading plane).
# ---------------------------------------------------------------------------


class _Publisher:
    def __init__(self) -> None:
        self._task: Optional[asyncio.Task] = None
        self._stop_event: Optional[asyncio.Event] = None

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(
            self._run(),
            name="wallet_state_bus_publisher",
        )

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
        cache = get_wallet_state_cache()
        stop_event = self._stop_event
        assert stop_event is not None
        try:
            while not stop_event.is_set():
                try:
                    payload = self._build_payload(cache.stats_snapshot())
                    await self._publish(payload)
                except Exception as exc:
                    logger.debug("wallet_state_bus publish cycle failed: %s", exc)
                try:
                    await asyncio.wait_for(
                        stop_event.wait(),
                        timeout=_HEARTBEAT_INTERVAL_SECONDS,
                    )
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise

    def _build_payload(self, stats: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "wallet_state_heartbeat",
            "ts": utcnow().isoformat(),
            "monotonic": time.monotonic(),
            "stats": stats,
            # Aggregate delta counters so the subscriber can detect
            # missed events (e.g. trading plane published 50 deltas but
            # subscriber only saw 48 → 2 lost in transit).
            "deltas_emitted": _delta_counter.value,
        }

    async def _publish(self, payload: dict[str, Any]) -> None:
        client = redis_client.get_client_or_none()
        if client is None:
            return
        message = json.dumps(payload, default=str)
        # Two writes: PUBLISH for live subscribers, SET for late joiners
        # (so a freshly-booted API plane gets the most-recent state
        # without waiting for the next publish tick).
        try:
            await client.publish(
                redis_client.namespaced(WALLET_STATE_HEARTBEAT_CHANNEL),
                message,
            )
            await client.set(
                redis_client.namespaced(WALLET_STATE_HEARTBEAT_LAST_KEY),
                message,
                ex=60,
            )
        except Exception as exc:
            logger.debug("wallet_state_bus publish failed: %s", exc)


_publisher = _Publisher()


async def start_publisher() -> None:
    """Start the periodic publisher loop.  Call from the trading plane only."""
    await _publisher.start()


async def stop_publisher() -> None:
    await _publisher.stop()


# ---------------------------------------------------------------------------
# Subscriber (runs in the API plane).
# ---------------------------------------------------------------------------


class _Subscriber:
    def __init__(self) -> None:
        self._task: Optional[asyncio.Task] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._latest: Optional[dict[str, Any]] = None
        self._latest_received_mono: Optional[float] = None
        # Bounded ring of recent deltas — the API plane uses these to
        # populate UI streams and to detect gaps in delta sequence.
        self._latest_delta: Optional[dict[str, Any]] = None
        self._latest_delta_received_mono: Optional[float] = None
        self._delta_count: int = 0
        # Asyncio event the API-side consumers can await to wake on the
        # next delta.  Cleared when consumed.
        self._delta_event: Optional[asyncio.Event] = None

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(
            self._run(),
            name="wallet_state_bus_subscriber",
        )

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

    def latest(self) -> Optional[dict[str, Any]]:
        return self._latest

    def latest_age_seconds(self) -> Optional[float]:
        if self._latest_received_mono is None:
            return None
        return time.monotonic() - self._latest_received_mono

    def latest_delta(self) -> Optional[dict[str, Any]]:
        return self._latest_delta

    def latest_delta_age_seconds(self) -> Optional[float]:
        if self._latest_delta_received_mono is None:
            return None
        return time.monotonic() - self._latest_delta_received_mono

    def delta_count(self) -> int:
        return self._delta_count

    def delta_event(self) -> Optional[asyncio.Event]:
        """Event consumers can await to be notified of the next delta.

        Lazily-constructed in the subscriber's loop; safe to read but
        treat as None if the subscriber hasn't started yet.
        """
        return self._delta_event

    async def _run(self) -> None:
        stop_event = self._stop_event
        assert stop_event is not None
        # Lazy-construct the wakeup event in the subscriber's own loop so
        # it binds to the right loop for thread-safe waiters.
        if self._delta_event is None:
            self._delta_event = asyncio.Event()
        backoff = 1.0
        # Seed from the LAST key on first run so a freshly-booted API
        # plane has state before the next publish tick.
        await self._seed_from_last_key()
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
                heartbeat_channel = redis_client.namespaced(
                    WALLET_STATE_HEARTBEAT_CHANNEL,
                )
                delta_channel = redis_client.namespaced(
                    WALLET_STATE_DELTA_CHANNEL,
                )
                await pubsub.subscribe(heartbeat_channel, delta_channel)
                backoff = 1.0
                while not stop_event.is_set():
                    try:
                        message = await pubsub.get_message(
                            ignore_subscribe_messages=True,
                            timeout=2.0,
                        )
                    except (asyncio.TimeoutError,):
                        message = None
                    if message is None:
                        continue
                    if message.get("type") != "message":
                        continue
                    channel = message.get("channel")
                    if isinstance(channel, (bytes, bytearray)):
                        channel = channel.decode("utf-8", errors="replace")
                    if channel == delta_channel:
                        self._consume_delta(message.get("data"))
                    else:
                        self._consume_message(message.get("data"))
            except Exception as exc:
                logger.debug("wallet_state_bus subscriber error: %s", exc)
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

    async def _seed_from_last_key(self) -> None:
        client = redis_client.get_client_or_none()
        if client is None:
            return
        try:
            value = await client.get(
                redis_client.namespaced(WALLET_STATE_HEARTBEAT_LAST_KEY),
            )
        except Exception:
            return
        self._consume_message(value)

    def _consume_message(self, data: Any) -> None:
        if not data:
            return
        try:
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8", errors="replace")
            payload = json.loads(data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        self._latest = payload
        self._latest_received_mono = time.monotonic()

    def _consume_delta(self, data: Any) -> None:
        if not data:
            return
        try:
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8", errors="replace")
            payload = json.loads(data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        self._latest_delta = payload
        self._latest_delta_received_mono = time.monotonic()
        self._delta_count += 1
        # Wake any local task awaiting the next delta.  Set+clear is the
        # standard "edge-trigger from background loop" pattern.
        ev = self._delta_event
        if ev is not None and not ev.is_set():
            ev.set()


_subscriber = _Subscriber()


async def start_subscriber() -> None:
    """Start the API-side subscriber loop."""
    await _subscriber.start()


async def stop_subscriber() -> None:
    await _subscriber.stop()


def get_latest_heartbeat() -> Optional[dict[str, Any]]:
    """Return the most-recent heartbeat payload, or None if never received."""
    return _subscriber.latest()


def get_latest_delta() -> Optional[dict[str, Any]]:
    """Return the most-recent delta event, or None if never received."""
    return _subscriber.latest_delta()


def get_delta_event() -> Optional[asyncio.Event]:
    """Return the asyncio.Event that fires on every new delta.

    Consumers can ``await event.wait()`` to be woken on the next
    cross-process wallet mutation, then call ``event.clear()`` to arm
    the next wakeup.  Returns ``None`` if the subscriber hasn't started.
    """
    return _subscriber.delta_event()


def is_stale(threshold_seconds: float = _HEARTBEAT_STALE_AFTER_SECONDS) -> bool:
    """True iff no fresh heartbeat has been seen within the threshold."""
    age = _subscriber.latest_age_seconds()
    if age is None:
        return True
    return age > threshold_seconds


def status_snapshot() -> dict[str, Any]:
    """Lightweight status dict for /health/* endpoints."""
    age = _subscriber.latest_age_seconds()
    delta_age = _subscriber.latest_delta_age_seconds()
    return {
        "heartbeat_channel": WALLET_STATE_HEARTBEAT_CHANNEL,
        "delta_channel": WALLET_STATE_DELTA_CHANNEL,
        "last_heartbeat_age_seconds": (None if age is None else round(age, 2)),
        "last_delta_age_seconds": (None if delta_age is None else round(delta_age, 3)),
        "deltas_received": _subscriber.delta_count(),
        "deltas_emitted": _delta_counter.value,
        "stale": is_stale(),
        "latest_heartbeat": _subscriber.latest(),
        "latest_delta": _subscriber.latest_delta(),
    }
