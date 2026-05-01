"""Strategy DataEvent routing for the single-process runtime."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Awaitable, Callable, Set, Optional, Any

from services.data_events import DataEvent, EventType
from utils.logger import get_logger
logger = get_logger("event_dispatcher")

EventHandler = Callable[[DataEvent], Awaitable[list]]


class EventDispatcher:
    def __init__(self) -> None:
        self._handlers: dict[str, list[tuple[str, EventHandler]]] = defaultdict(list)
        self._subscriptions: dict[str, set[str]] = defaultdict(set)
        self._start_lock = asyncio.Lock()
        self._running = False
        self._handler_timeout_seconds = float(
            max(
                5.0,
                60.0,
            )
        )
        self._handler_cancel_grace_seconds = float(
            max(
                0.1,
                5.0,
            )
        )
        self._timed_out_handler_tasks: set[asyncio.Task[Any]] = set()
        self._force_kill_tasks: set[asyncio.Task[Any]] = set()

    def _consume_timed_out_handler_result(
        self,
        task: asyncio.Task[Any],
        *,
        strategy: str | None = None,
        event_type: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self._timed_out_handler_tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.warning(
                "Strategy event handler failed after timeout",
                strategy=strategy,
                event_type=event_type,
                timeout_seconds=timeout_seconds,
                exc_info=exc,
            )

    def _track_timed_out_handler_task(
        self,
        task: asyncio.Task[Any],
        *,
        strategy: str,
        event_type: str,
        timeout_seconds: float,
    ) -> None:
        self._timed_out_handler_tasks.add(task)

        async def _force_kill() -> None:
            """Give the runaway handler 10s more, then hard-cancel again."""
            await asyncio.sleep(10)
            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait({task}, timeout=self._handler_cancel_grace_seconds)
                except Exception:
                    pass
                logger.warning(
                    "Force-cancelled runaway handler task",
                    strategy=strategy,
                    event_type=event_type,
                    finished=task.done(),
                )

        kill_task = asyncio.create_task(_force_kill(), name=f"force-kill-{strategy}")
        self._force_kill_tasks.add(kill_task)
        kill_task.add_done_callback(self._force_kill_tasks.discard)
        task.add_done_callback(
            lambda done_task: self._consume_timed_out_handler_result(
                done_task,
                strategy=strategy,
                event_type=event_type,
                timeout_seconds=timeout_seconds,
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

    def unsubscribe_all(self, strategy_slug: str) -> None:
        for event_type in list(self._subscriptions.get(strategy_slug, [])):
            self._handlers[event_type] = [(slug, h) for slug, h in self._handlers[event_type] if slug != strategy_slug]
        self._subscriptions.pop(strategy_slug, None)

    async def start(self) -> None:
        if self._running:
            return
        async with self._start_lock:
            if self._running:
                return
            self._running = True

    async def stop(self) -> None:
        self._running = False
        pending_cancellations: list[asyncio.Task[Any]] = []
        for timed_out_task in list(self._timed_out_handler_tasks):
            if timed_out_task.done():
                self._consume_timed_out_handler_result(timed_out_task)
                continue
            timed_out_task.cancel()
            pending_cancellations.append(timed_out_task)
        if pending_cancellations:
            shutdown_grace_seconds = max(self._handler_cancel_grace_seconds, 2.0)
            try:
                done, pending = await asyncio.wait(
                    pending_cancellations,
                    timeout=shutdown_grace_seconds,
                )
            except Exception as exc:
                logger.warning(
                    "Failed while awaiting timed-out strategy handlers during shutdown",
                    tasks=len(pending_cancellations),
                    exc_info=exc,
                )
                done = set()
                pending = set(pending_cancellations)
            for done_task in done:
                self._consume_timed_out_handler_result(done_task)
            if pending:
                logger.warning(
                    "Timed-out strategy handlers still pending during shutdown; cancelling force-kill tasks",
                    tasks=len(pending),
                    shutdown_grace_seconds=shutdown_grace_seconds,
                )
        for kill_task in list(self._force_kill_tasks):
            kill_task.cancel()
        self._force_kill_tasks.clear()

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
        # Bounded worker pool when many strategies are registered.
        # Production saw 16-19 strategies fan out per event with each
        # strategy spawning its own HTTP/DB sub-tasks, multiplying loop
        # pressure into the hundreds.  At small N (<=6) keep the simple
        # gather form so the typical case has zero scheduling overhead.
        if len(handlers) <= 6:
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
        else:
            queue: asyncio.Queue = asyncio.Queue()
            for item in handlers:
                queue.put_nowait(item)
            results: list = []

            async def _dispatch_worker() -> None:
                while True:
                    try:
                        slug, handler = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    try:
                        result = await self._safe_invoke(
                            slug,
                            handler,
                            event,
                            timeout_seconds=timeout_seconds,
                        )
                        results.append(result)
                    except Exception as exc:
                        results.append(exc)
                    finally:
                        queue.task_done()
                        await asyncio.sleep(0)

            # Concurrency=8 keeps the loop healthy even when 20+
            # strategies are registered.  Each strategy still gets
            # the same per-handler timeout; we just run at most 8 at
            # a time instead of all in parallel.
            workers = [
                asyncio.create_task(
                    _dispatch_worker(), name=f"event-dispatch-worker-{i}"
                )
                for i in range(min(8, len(handlers)))
            ]
            try:
                await asyncio.gather(*workers, return_exceptions=True)
            finally:
                for w in workers:
                    if not w.done():
                        w.cancel()

        all_opportunities = []
        for result in results:
            if isinstance(result, Exception):
                continue
            if isinstance(result, list):
                all_opportunities.extend(result)
        logger.debug(
            "DataEvent dispatched locally",
            event_type=event.event_type,
            source=event.source,
            handlers=len(handlers),
            opportunities=len(all_opportunities),
            targeted=include_strategies is not None,
        )
        return all_opportunities

    async def _safe_invoke(
        self,
        slug: str,
        handler: EventHandler,
        event: DataEvent,
        *,
        timeout_seconds: float,
    ) -> list:
        handler_task = asyncio.create_task(handler(event))

        try:
            done, _pending = await asyncio.wait({handler_task}, timeout=timeout_seconds)
            if done:
                result = handler_task.result()
                return result if isinstance(result, list) else []
            handler_task.cancel()
            done_after_cancel, _pending_after_cancel = await asyncio.shield(
                asyncio.wait(
                    {handler_task},
                    timeout=self._handler_cancel_grace_seconds,
                )
            )
            if done_after_cancel:
                try:
                    handler_task.result()
                    logger.warning(
                        "Strategy event handler timed out but completed during cancellation grace period: %s (%s)",
                        slug,
                        event.event_type,
                        strategy=slug,
                        event_type=event.event_type,
                        timeout_seconds=timeout_seconds,
                        cancel_grace_seconds=self._handler_cancel_grace_seconds,
                    )
                except asyncio.CancelledError:
                    logger.warning(
                        "Strategy event handler timed out and was cancelled: %s (%s)",
                        slug,
                        event.event_type,
                        strategy=slug,
                        event_type=event.event_type,
                        timeout_seconds=timeout_seconds,
                        cancel_grace_seconds=self._handler_cancel_grace_seconds,
                    )
                except Exception as exc:
                    logger.warning(
                        "Strategy event handler timed out and failed during cancellation: %s (%s)",
                        slug,
                        event.event_type,
                        strategy=slug,
                        event_type=event.event_type,
                        timeout_seconds=timeout_seconds,
                        cancel_grace_seconds=self._handler_cancel_grace_seconds,
                        exc_info=exc,
                    )
                return []
            else:
                self._track_timed_out_handler_task(
                    handler_task,
                    strategy=slug,
                    event_type=event.event_type,
                    timeout_seconds=timeout_seconds,
                )
                logger.warning(
                    "Strategy event handler timed out; cancellation grace exceeded: %s (%s)",
                    slug,
                    event.event_type,
                    strategy=slug,
                    event_type=event.event_type,
                    timeout_seconds=timeout_seconds,
                    cancel_grace_seconds=self._handler_cancel_grace_seconds,
                )
                return []
        except asyncio.CancelledError:
            # Parent (_dispatch_local / gather) was cancelled — ensure the
            # handler task is cleaned up so it doesn't leak a DB connection.
            if not handler_task.done():
                handler_task.cancel()
                try:
                    await asyncio.shield(asyncio.wait({handler_task}, timeout=2.0))
                except (asyncio.CancelledError, Exception):
                    pass
                if not handler_task.done():
                    self._track_timed_out_handler_task(
                        handler_task,
                        strategy=slug,
                        event_type=event.event_type,
                        timeout_seconds=timeout_seconds,
                    )
            raise
        except Exception as exc:
            logger.warning(
                "Strategy event handler failed: %s (%s): %s",
                slug,
                event.event_type,
                exc,
                strategy=slug,
                event_type=event.event_type,
                exc_info=exc,
            )
            return []

    @property
    def subscription_count(self) -> int:
        return sum(len(handlers) for handlers in self._handlers.values())

    def get_subscriptions(self, strategy_slug: str) -> set[str]:
        return set(self._subscriptions.get(strategy_slug, set()))


event_dispatcher = EventDispatcher()
