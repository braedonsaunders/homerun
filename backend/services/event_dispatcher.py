from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Awaitable, Callable, Set

from utils.logger import get_logger
from services.data_events import DataEvent

logger = get_logger("event_dispatcher")

EventHandler = Callable[[DataEvent], Awaitable[list]]


class EventDispatcher:
    """Routes DataEvents to subscribed strategy handlers.

    Strategies declare what event types they care about via their
    ``subscriptions`` class attribute. The strategy loader registers
    handlers when loading strategies and unregisters on unload.

    Dispatch is concurrent -- all handlers for an event type run
    via asyncio.gather with error isolation (one handler failing
    doesn't affect others).
    """

    def __init__(self):
        # event_type -> [(strategy_slug, handler)]
        self._handlers: dict[str, list[tuple[str, EventHandler]]] = defaultdict(list)
        # strategy_slug -> set of subscribed event_types
        self._subscriptions: dict[str, set[str]] = defaultdict(set)

    def subscribe(self, strategy_slug: str, event_type: str, handler: EventHandler) -> None:
        self._handlers[event_type].append((strategy_slug, handler))
        self._subscriptions[strategy_slug].add(event_type)

    def unsubscribe_all(self, strategy_slug: str) -> None:
        for event_type in list(self._subscriptions.get(strategy_slug, [])):
            self._handlers[event_type] = [
                (slug, h) for slug, h in self._handlers[event_type]
                if slug != strategy_slug
            ]
        self._subscriptions.pop(strategy_slug, None)

    async def dispatch(self, event: DataEvent) -> list:
        """Dispatch an event to all subscribed handlers.

        Returns a flat list of all results (ArbitrageOpportunity objects)
        from all handlers. Handlers that raise exceptions are logged
        and skipped -- they don't affect other handlers.
        """
        handlers = list(self._handlers.get(event.event_type, []))
        # Also dispatch to wildcard subscribers
        handlers.extend(self._handlers.get("*", []))

        if not handlers:
            return []

        tasks = [
            asyncio.create_task(
                self._safe_invoke(slug, handler, event)
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

    async def _safe_invoke(self, slug: str, handler: EventHandler, event: DataEvent) -> list:
        try:
            result = await handler(event)
            return result if isinstance(result, list) else []
        except Exception as exc:
            logger.warning("Strategy event handler failed", strategy=slug, event_type=event.event_type, error=str(exc))
            return []

    @property
    def subscription_count(self) -> int:
        return sum(len(handlers) for handlers in self._handlers.values())

    def get_subscriptions(self, strategy_slug: str) -> set[str]:
        return set(self._subscriptions.get(strategy_slug, set()))


event_dispatcher = EventDispatcher()
