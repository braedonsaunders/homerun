"""Cross-plane → in-process bridge for Telegram operator alerts.

Why
---
The ``TelegramNotifier`` used to run inside the trading plane alongside
the trader-orchestrator.  Its ``_autotrader_monitor_loop`` polls the DB
every 5 s and its queue worker holds HTTP state — both fine in
isolation, but under sustained load the monitor's single session
wrapped multiple awaits and held asyncpg connections for 30+ s,
starving the trader hot path.  Pool-hold warnings from the previous
soak:

    Connection held for 30.6s before return to pool (task=Task-536,
    coro=TelegramNotifier._autotrader_monitor_loop)

R10-B.1 moves the notifier to the **discovery** plane so its DB work
and HTTP latency can never steal budget from the trader orchestrator.
The notifier is still a single-singleton service (running it twice
would double-deliver every Telegram message), so callers on other
planes (e.g. ``stuck_position_monitor`` on trading, settings routes on
the API plane) that need to surface an operator alert publish to a
Redis channel; the notifier host subscribes and enqueues locally.

Architecture
------------
* Producers (any plane): ``await notifier_bridge.publish_alert(text,
  category)``.  Soft-fails — if Redis is unavailable, the alert is
  dropped and a debug log is emitted.  The caller is expected to log
  the condition separately.
* Consumer (discovery plane): ``await notifier_bridge.start_subscriber()``
  subscribes to the channel and pushes every payload into
  ``notifier.send_operator_alert_local``.
* Idempotency: the notifier itself also runs ``send_operator_alert``
  directly on the host plane (bypassing Redis), so local callers on the
  discovery plane don't round-trip through Redis either.

Lifecycle
---------
* Subscriber: started by ``workers/host.py::_initialize_services`` on
  the discovery plane only (the plane that owns the notifier
  singleton), stopped on shutdown.
* Publish API: importable from any plane; becomes a no-op when Redis
  is disabled.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional

from services import redis_client
from utils.logger import get_logger

logger = get_logger("notifier_bridge")

# Channel name — pre-namespace so callers outside this file see the
# public name.  ``redis_client.safe_publish`` namespaces internally.
NOTIFIER_ALERT_CHANNEL = "notifier:operator_alerts"


async def publish_alert(text: str, *, category: str = "operator") -> bool:
    """Publish an operator alert to the notifier host plane.

    Returns True on successful publish, False if Redis was unavailable
    or the payload was empty (soft-fail).  Callers MUST NOT rely on the
    return value for delivery confirmation — a True here only means the
    message was accepted by Redis; actual Telegram delivery happens on
    the subscriber side.
    """
    if not text:
        return False
    normalized = str(text).strip()
    if not normalized:
        return False
    payload = {
        "text": normalized,
        "category": str(category or "operator"),
        "ts": time.time(),
    }
    try:
        body = json.dumps(payload)
    except (TypeError, ValueError):
        return False
    return await redis_client.safe_publish(NOTIFIER_ALERT_CHANNEL, body)


# ---------------------------------------------------------------------------
# Subscriber.
# ---------------------------------------------------------------------------


class _Subscriber:
    def __init__(self) -> None:
        self._task: Optional[asyncio.Task] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._messages_received: int = 0
        self._messages_dispatched: int = 0
        self._last_message_mono: Optional[float] = None

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="notifier_bridge_subscriber")

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
            "messages_dispatched": self._messages_dispatched,
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
                channel = redis_client.namespaced(NOTIFIER_ALERT_CHANNEL)
                await pubsub.subscribe(channel)
                logger.info("notifier_bridge subscribed", channel=channel)
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
                    data = message.get("data")
                    if isinstance(data, (bytes, bytearray)):
                        data = data.decode("utf-8", errors="replace")
                    try:
                        payload = json.loads(data) if data else None
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
                    if not isinstance(payload, dict):
                        continue
                    text = str(payload.get("text") or "").strip()
                    category = str(payload.get("category") or "operator")
                    if not text:
                        continue
                    try:
                        # Lazy import to avoid circular: notifier imports
                        # nothing from us, but we import from it.
                        from services.notifier import notifier as _notifier

                        await _notifier.send_operator_alert_local(
                            text, category=category
                        )
                        self._messages_dispatched += 1
                    except Exception as exc:
                        logger.debug("notifier_bridge dispatch failed: %s", exc)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("notifier_bridge error: %s", exc)
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
    """Start the bridge subscriber.  Discovery plane only."""
    await _subscriber.start()


async def stop_subscriber() -> None:
    await _subscriber.stop()


def status_snapshot() -> dict[str, Any]:
    return _subscriber.status_snapshot()
