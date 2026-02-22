"""Strategy DataEvent routing with Redis stream fan-out for cross-process delivery."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Awaitable, Callable, Set, Optional, Any

from config import settings
from models.database import AsyncSessionLocal
from services.data_events import DataEvent, EventType
from services.redis_streams import redis_streams
from services.strategy_signal_bridge import bridge_opportunities_to_signals
from utils.logger import get_logger
from utils.utcnow import utcnow

logger = get_logger("event_dispatcher")

EventHandler = Callable[[DataEvent], Awaitable[list]]

_REDIS_FANOUT_EVENT_TYPES = frozenset(
    {
        EventType.PRICE_CHANGE,
        EventType.MARKET_RESOLVED,
        EventType.CRYPTO_UPDATE,
        EventType.WEATHER_UPDATE,
        EventType.TRADER_ACTIVITY,
        EventType.NEWS_UPDATE,
        EventType.NEWS_EVENT,
        EventType.EVENTS_UPDATE,
        EventType.DATA_SOURCE_UPDATE,
        EventType.TRADE_EXECUTION,
    }
)

_REMOTE_BRIDGE_SOURCE_BY_EVENT: dict[str, str] = {
    EventType.EVENTS_UPDATE: "events",
    EventType.DATA_SOURCE_UPDATE: "data_source",
}


def _parse_timestamp(value: str | None) -> datetime:
    if not value:
        return utcnow()
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except Exception:
        return utcnow()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _sanitize_payload(event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    if event_type == EventType.TRADER_ACTIVITY:
        sanitized = dict(payload)
        sanitized.pop("opportunities", None)
        return sanitized
    return payload


class EventDispatcher:
    def __init__(self) -> None:
        self._handlers: dict[str, list[tuple[str, EventHandler]]] = defaultdict(list)
        self._subscriptions: dict[str, set[str]] = defaultdict(set)
        self._stream_key = str(settings.DATA_EVENT_STREAM_KEY)
        self._stream_maxlen = int(settings.REDIS_EVENT_STREAM_MAXLEN)
        self._read_block_ms = int(settings.REDIS_STREAM_BLOCK_MS)
        self._read_count = int(settings.REDIS_STREAM_READ_COUNT)
        self._instance_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self._listener_task: Optional[asyncio.Task] = None
        self._start_lock = asyncio.Lock()
        self._cursor_lock = asyncio.Lock()
        self._running = False
        self._last_stream_id = "$"
        self._handler_timeout_seconds = float(
            max(
                5.0,
                float(getattr(settings, "EVENT_HANDLER_TIMEOUT_SECONDS", 60.0) or 60.0),
            )
        )

    def subscribe(self, strategy_slug: str, event_type: str, handler: EventHandler) -> None:
        if event_type != "*" and event_type not in EventType._ALL:
            raise ValueError(
                f"Unknown event_type '{event_type}' for strategy '{strategy_slug}'. "
                f"Valid types: {sorted(EventType._ALL)}. "
                f"Use EventType.* constants from services.data_events — "
                f"e.g. EventType.MARKET_DATA_REFRESH instead of 'market_data_refresh'."
            )
        self._handlers[event_type].append((strategy_slug, handler))
        self._subscriptions[strategy_slug].add(event_type)
        self._ensure_listener_started()

    def unsubscribe_all(self, strategy_slug: str) -> None:
        for event_type in list(self._subscriptions.get(strategy_slug, [])):
            self._handlers[event_type] = [(slug, h) for slug, h in self._handlers[event_type] if slug != strategy_slug]
        self._subscriptions.pop(strategy_slug, None)

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
                name="event-dispatcher-redis-listener",
            )
            logger.info(
                "Event dispatcher Redis listener started",
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

    async def dispatch(
        self,
        event: DataEvent,
        include_strategies: Set[str] | None = None,
        handler_timeout_seconds: Optional[float] = None,
    ) -> list:
        all_opportunities = await self._dispatch_local(
            event,
            include_strategies=include_strategies,
            handler_timeout_seconds=handler_timeout_seconds,
        )
        if include_strategies is None:
            await self._publish_remote(event)
        return all_opportunities

    async def _dispatch_local(
        self,
        event: DataEvent,
        *,
        include_strategies: Set[str] | None = None,
        handler_timeout_seconds: Optional[float] = None,
    ) -> list:
        handlers = list(self._handlers.get(event.event_type, []))
        handlers.extend(self._handlers.get("*", []))
        if include_strategies is not None:
            handlers = [(slug, handler) for slug, handler in handlers if slug in include_strategies]

        if not handlers:
            return []

        timeout_seconds = (
            self._handler_timeout_seconds
            if handler_timeout_seconds is None
            else max(1.0, float(handler_timeout_seconds))
        )
        tasks = [
            asyncio.create_task(
                self._safe_invoke(
                    slug,
                    handler,
                    event,
                    timeout_seconds=timeout_seconds,
                )
            )
            for slug, handler in handlers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_opportunities = []
        for result in results:
            if isinstance(result, Exception):
                continue
            if isinstance(result, list):
                all_opportunities.extend(result)
        return all_opportunities

    async def _safe_invoke(
        self,
        slug: str,
        handler: EventHandler,
        event: DataEvent,
        *,
        timeout_seconds: float,
    ) -> list:
        try:
            result = await asyncio.wait_for(
                handler(event),
                timeout=timeout_seconds,
            )
            return result if isinstance(result, list) else []
        except asyncio.TimeoutError:
            logger.warning(
                "Strategy event handler timed out",
                strategy=slug,
                event_type=event.event_type,
                timeout_seconds=timeout_seconds,
            )
            return []
        except Exception as exc:
            logger.warning(
                "Strategy event handler failed",
                strategy=slug,
                event_type=event.event_type,
                exc_info=exc,
            )
            return []

    async def _publish_remote(self, event: DataEvent) -> None:
        if event.event_type not in _REDIS_FANOUT_EVENT_TYPES:
            return
        payload: dict[str, Any] = {
            "instance_id": self._instance_id,
            "event": {
                "event_type": event.event_type,
                "source": event.source,
                "timestamp": event.timestamp.isoformat(),
                "market_id": event.market_id,
                "token_id": event.token_id,
                "payload": _sanitize_payload(
                    event.event_type,
                    event.payload if isinstance(event.payload, dict) else {},
                ),
                "old_price": event.old_price,
                "new_price": event.new_price,
                "scan_mode": event.scan_mode,
                "changed_token_ids": event.changed_token_ids,
                "changed_market_ids": event.changed_market_ids,
                "affected_market_ids": event.affected_market_ids,
            },
        }
        await redis_streams.append_json(
            self._stream_key,
            payload,
            maxlen=self._stream_maxlen,
        )

    def _deserialize_event(self, payload: dict[str, Any]) -> Optional[DataEvent]:
        event_data = payload.get("event")
        if not isinstance(event_data, dict):
            return None
        event_type = event_data.get("event_type")
        source = event_data.get("source")
        if not isinstance(event_type, str) or not isinstance(source, str):
            return None
        return DataEvent(
            event_type=event_type,
            source=source,
            timestamp=_parse_timestamp(event_data.get("timestamp")),
            market_id=event_data.get("market_id"),
            token_id=event_data.get("token_id"),
            payload=event_data.get("payload") if isinstance(event_data.get("payload"), dict) else {},
            old_price=event_data.get("old_price"),
            new_price=event_data.get("new_price"),
            scan_mode=event_data.get("scan_mode"),
            changed_token_ids=event_data.get("changed_token_ids"),
            changed_market_ids=event_data.get("changed_market_ids"),
            affected_market_ids=event_data.get("affected_market_ids"),
        )

    async def _run_stream_listener(self) -> None:
        own_marker = f'"instance_id":"{self._instance_id}"'
        while self._running:
            async with self._cursor_lock:
                last_stream_id = self._last_stream_id
            messages = await redis_streams.read_raw(
                self._stream_key,
                last_id=last_stream_id,
                block_ms=self._read_block_ms,
                count=self._read_count,
            )
            if not messages:
                continue
            for entry_id, raw in messages:
                async with self._cursor_lock:
                    self._last_stream_id = entry_id
                if own_marker in raw:
                    continue
                try:
                    payload = json.loads(raw)
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                event = self._deserialize_event(payload)
                if event is None:
                    continue
                opportunities = await self._dispatch_local(event)
                bridge_source = _REMOTE_BRIDGE_SOURCE_BY_EVENT.get(event.event_type)
                if bridge_source and opportunities:
                    try:
                        async with AsyncSessionLocal() as session:
                            await bridge_opportunities_to_signals(
                                session,
                                opportunities,
                                source=bridge_source,
                            )
                    except Exception as exc:
                        logger.warning(
                            "Remote DataEvent signal bridge failed",
                            event_type=event.event_type,
                            source=bridge_source,
                            exc_info=exc,
                        )

    @property
    def subscription_count(self) -> int:
        return sum(len(handlers) for handlers in self._handlers.values())

    def get_subscriptions(self, strategy_slug: str) -> set[str]:
        return set(self._subscriptions.get(strategy_slug, set()))


event_dispatcher = EventDispatcher()
