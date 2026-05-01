"""Redis → WebSocket bridge for cross-process trader_event streaming.

Architecture
------------
Bot worker plane fires ``create_trader_event(...)`` (or any code path
that emits a trader_event), which now publishes to the Redis pub/sub
channel ``trader_events`` in addition to the in-process ``event_bus``.
The API plane is a SEPARATE process — it can't see the in-process
event_bus from the worker plane — so before this bridge existed, the
UI's only way to get bot terminal events was to poll
``/api/traders/events/all`` (DB-bound, N-second stale).

This bridge runs in the API plane lifespan, subscribes to the Redis
``trader_events`` channel, and hands each event to a caller-supplied
callback.  ``main.py`` registers the WebSocket-broadcast callback so
events fan out to every connected UI client in <5ms.

Soft-fail contract
------------------
* If Redis is disabled or unreachable, the bridge sleeps and retries
  with exponential backoff.  ``get_latest_event()`` returns ``None``
  until the first message lands.  No path here can wedge the API.
* Subscriber failures (decode errors, callback exceptions) are logged
  and swallowed — one bad event must not stop the stream.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Awaitable, Callable, Optional

from services import redis_client
from utils.logger import get_logger

logger = get_logger("trader_events_bridge")

# Channel name MUST match the publisher in
# ``services/trader_orchestrator_state.py::create_trader_event``.
TRADER_EVENTS_CHANNEL = "trader_events"


# Callback signature: async function that takes the deserialized event
# payload (a dict) and broadcasts it to UI WebSocket clients.
TraderEventCallback = Callable[[dict[str, Any]], Awaitable[None]]


class _Bridge:
    def __init__(self) -> None:
        self._task: Optional[asyncio.Task] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._on_event: Optional[TraderEventCallback] = None
        self._messages_received: int = 0
        self._messages_dispatched: int = 0
        self._messages_dropped: int = 0
        self._last_message_mono: Optional[float] = None
        self._latest_event: Optional[dict[str, Any]] = None

    async def start(self, on_event: TraderEventCallback) -> None:
        if self._task is not None and not self._task.done():
            return
        self._on_event = on_event
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(
            self._run(), name="trader_events_bridge"
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
        self._on_event = None

    def status_snapshot(self) -> dict[str, Any]:
        age = (
            None
            if self._last_message_mono is None
            else round(time.monotonic() - self._last_message_mono, 3)
        )
        return {
            "channel": TRADER_EVENTS_CHANNEL,
            "running": self._task is not None and not self._task.done(),
            "messages_received": self._messages_received,
            "messages_dispatched": self._messages_dispatched,
            "messages_dropped": self._messages_dropped,
            "last_message_age_seconds": age,
            "latest_event": self._latest_event,
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
                channel = redis_client.namespaced(TRADER_EVENTS_CHANNEL)
                await pubsub.subscribe(channel)
                logger.info(
                    "trader_events_bridge subscribed", channel=channel
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
                    data = message.get("data")
                    if isinstance(data, (bytes, bytearray)):
                        data = data.decode("utf-8", errors="replace")
                    try:
                        payload = json.loads(data) if data else None
                    except (TypeError, ValueError, json.JSONDecodeError):
                        self._messages_dropped += 1
                        continue
                    if not isinstance(payload, dict):
                        self._messages_dropped += 1
                        continue
                    self._latest_event = payload
                    if self._on_event is not None:
                        try:
                            await self._on_event(payload)
                            self._messages_dispatched += 1
                        except Exception as exc:
                            self._messages_dropped += 1
                            logger.debug(
                                "trader_events_bridge callback raised: %s",
                                exc,
                            )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("trader_events_bridge error: %s", exc)
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


_bridge = _Bridge()


async def start(on_event: TraderEventCallback) -> None:
    """Start the bridge.  Call from the API-plane lifespan only.

    ``on_event`` is invoked for every trader_event payload received from
    Redis.  Typical implementation: forward the event to all connected
    WebSocket clients via the existing ``manager.broadcast`` API.
    """
    await _bridge.start(on_event)


async def stop() -> None:
    await _bridge.stop()


def status_snapshot() -> dict[str, Any]:
    return _bridge.status_snapshot()


def get_latest_event() -> Optional[dict[str, Any]]:
    return _bridge._latest_event  # noqa: SLF001 — internal read for diagnostics
