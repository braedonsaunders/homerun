"""WebSocket broadcast bus with Redis stream fan-out for cross-process delivery."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional

from config import settings
from services.redis_streams import redis_streams
from utils.logger import get_logger

logger = get_logger("event_bus")

EventCallback = Callable[[str, Dict[str, Any]], Awaitable[None]]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: Dict[str, List[EventCallback]] = {}
        self._stream_key = str(settings.EVENT_BUS_STREAM_KEY)
        self._stream_maxlen = int(settings.REDIS_EVENT_STREAM_MAXLEN)
        self._read_block_ms = int(settings.REDIS_STREAM_BLOCK_MS)
        self._read_count = int(settings.REDIS_STREAM_READ_COUNT)
        self._instance_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self._listener_task: Optional[asyncio.Task] = None
        self._start_lock = asyncio.Lock()
        self._running = False
        self._last_stream_id = "$"

    def subscribe(self, event_type: str, callback: EventCallback) -> None:
        self._subscribers.setdefault(event_type, []).append(callback)
        logger.debug("Subscribed to event_type=%s", event_type)
        self._ensure_listener_started()

    def unsubscribe(self, event_type: str, callback: EventCallback) -> None:
        callbacks = self._subscribers.get(event_type)
        if callbacks is None:
            return
        try:
            callbacks.remove(callback)
        except ValueError:
            pass

    def _ensure_listener_started(self) -> None:
        if self._running:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self.start())

    async def start(self) -> None:
        if self._running:
            return
        async with self._start_lock:
            if self._running:
                return
            self._running = True
            self._last_stream_id = "$"
            self._listener_task = asyncio.create_task(
                self._run_stream_listener(),
                name="event-bus-redis-listener",
            )
            logger.info(
                "Event bus Redis listener started",
                stream=self._stream_key,
                instance=self._instance_id,
            )

    async def stop(self) -> None:
        self._running = False
        task = self._listener_task
        self._listener_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def publish(self, event_type: str, data: Dict[str, Any]) -> None:
        self._dispatch_local(event_type, data)
        await self._publish_remote(event_type, data)

    def _dispatch_local(self, event_type: str, data: Dict[str, Any]) -> None:
        callbacks: List[EventCallback] = []
        callbacks.extend(self._subscribers.get(event_type, []))
        callbacks.extend(self._subscribers.get("*", []))
        if not callbacks:
            return
        for cb in callbacks:
            try:
                asyncio.create_task(_safe_invoke(cb, event_type, data))
            except Exception:
                logger.debug(
                    "Failed to schedule event callback",
                    event_type=event_type,
                )

    async def _publish_remote(self, event_type: str, data: Dict[str, Any]) -> None:
        payload = {
            "instance_id": self._instance_id,
            "event_type": event_type,
            "data": data,
        }
        await redis_streams.append_json(
            self._stream_key,
            payload,
            maxlen=self._stream_maxlen,
        )

    async def _run_stream_listener(self) -> None:
        own_marker = f'"instance_id":"{self._instance_id}"'
        while self._running:
            messages = await redis_streams.read_raw(
                self._stream_key,
                last_id=self._last_stream_id,
                block_ms=self._read_block_ms,
                count=self._read_count,
            )
            if not messages:
                continue
            for entry_id, raw in messages:
                self._last_stream_id = entry_id
                if own_marker in raw:
                    continue
                try:
                    payload = json.loads(raw)
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                event_type = payload.get("event_type")
                data = payload.get("data")
                if not isinstance(event_type, str) or not isinstance(data, dict):
                    continue
                self._dispatch_local(event_type, data)


async def _safe_invoke(
    cb: EventCallback,
    event_type: str,
    data: Dict[str, Any],
) -> None:
    try:
        await cb(event_type, data)
    except Exception as exc:
        logger.debug(
            "Event subscriber raised: %s",
            exc,
            event_type=event_type,
        )


event_bus = EventBus()
