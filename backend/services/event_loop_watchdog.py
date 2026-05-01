"""Asyncio event-loop stall detector.

Runs as a low-overhead background task that periodically schedules a
short ``asyncio.sleep`` and measures how late the wakeup actually
arrives.  If the wakeup is delayed beyond a threshold the loop has
been monopolized by something — either a long sync function on the
loop, a C-extension that didn't release the GIL, or excessive CPU work
inside a coroutine without yielding.

When a stall is detected we dump the active task list with each task's
short stack so operators can see exactly which coroutine was running
when the loop ground to a halt.  That is the single most valuable
diagnostic for "everything just got slow" symptoms in production.

Soft-fail: any exception in the watchdog itself is swallowed.  The
watchdog must NEVER crash the event loop it monitors.
"""

from __future__ import annotations

import asyncio
import time
import traceback
from typing import Optional

from utils.logger import get_logger

logger = get_logger("event_loop_watchdog")

# A 50 ms sleep is short enough to catch sub-second stalls but long
# enough that the watchdog's own overhead (~one wakeup per 50 ms) is
# negligible — under 0.1 % of one CPU.
_PROBE_SLEEP_SECONDS = 0.05
# Threshold for "the loop was stalled."  250 ms is well above any
# normal jitter from a healthy loop and catches the multi-second
# stalls we've been hunting in production.  Tunable per-environment.
_STALL_THRESHOLD_SECONDS = 0.25
# Cap how many task stacks we dump per stall to keep logs readable.
_MAX_TASKS_DUMPED = 20
# Don't spam logs: ignore a stall if we already logged one in the
# last 5 seconds.  Better one warning per real incident than 100
# warnings per stall window.
_WARN_COOLDOWN_SECONDS = 5.0


class _Watchdog:
    def __init__(self) -> None:
        self._task: Optional[asyncio.Task] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._last_warn_mono: float = 0.0
        self._stalls_observed: int = 0
        self._max_stall_seconds: float = 0.0

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="event-loop-watchdog")
        logger.info(
            "Event-loop watchdog started",
            probe_sleep_ms=int(_PROBE_SLEEP_SECONDS * 1000),
            stall_threshold_ms=int(_STALL_THRESHOLD_SECONDS * 1000),
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
        stop_event = self._stop_event
        assert stop_event is not None
        while not stop_event.is_set():
            scheduled_at = time.monotonic()
            try:
                await asyncio.sleep(_PROBE_SLEEP_SECONDS)
            except asyncio.CancelledError:
                raise
            actual_elapsed = time.monotonic() - scheduled_at
            stall_seconds = actual_elapsed - _PROBE_SLEEP_SECONDS
            if stall_seconds < _STALL_THRESHOLD_SECONDS:
                continue
            self._stalls_observed += 1
            if stall_seconds > self._max_stall_seconds:
                self._max_stall_seconds = stall_seconds
            now_mono = time.monotonic()
            if now_mono - self._last_warn_mono < _WARN_COOLDOWN_SECONDS:
                continue
            self._last_warn_mono = now_mono
            try:
                self._dump_tasks(stall_seconds=stall_seconds)
            except Exception as exc:
                # Watchdog must never crash; soft-fail and keep running.
                logger.debug("Event-loop watchdog dump failed", exc_info=exc)

    def _dump_tasks(self, *, stall_seconds: float) -> None:
        """Log running tasks with short stacks so we can see who's hogging the loop.

        Only the top ``_MAX_TASKS_DUMPED`` tasks are listed; for each
        we capture the topmost frame (the function actually executing)
        — which is what we need to identify the culprit, without
        flooding logs with deep traces.
        """
        try:
            tasks = list(asyncio.all_tasks())
        except RuntimeError:
            return
        # Don't include the watchdog itself in the dump.
        watchdog_self = self._task
        tasks = [t for t in tasks if t is not watchdog_self and not t.done()]
        # Sort by name so logs are stable and diff-able across stalls.
        tasks.sort(key=lambda t: (t.get_name() or "", id(t)))
        sampled = tasks[:_MAX_TASKS_DUMPED]
        running_summary: list[str] = []
        for task in sampled:
            name = task.get_name() or "<unnamed>"
            frame_summary = "<no_stack>"
            try:
                stack = task.get_stack(limit=3)
                if stack:
                    # Topmost frame (innermost) is the one currently
                    # executing.  Format as "file:lineno in func".
                    top = stack[-1]
                    frame_summary = (
                        f"{top.f_code.co_filename}:{top.f_lineno}"
                        f" in {top.f_code.co_name}"
                    )
            except Exception:
                pass
            running_summary.append(f"{name} -> {frame_summary}")
        logger.warning(
            "Event-loop stall detected",
            stall_seconds=round(stall_seconds, 3),
            active_tasks=len(tasks),
            tasks_sampled=len(sampled),
            stalls_observed=self._stalls_observed,
            max_stall_seconds=round(self._max_stall_seconds, 3),
            running_tasks=running_summary,
        )

    def status_snapshot(self) -> dict:
        return {
            "running": self._task is not None and not self._task.done(),
            "stalls_observed": self._stalls_observed,
            "max_stall_seconds": round(self._max_stall_seconds, 3),
            "stall_threshold_seconds": _STALL_THRESHOLD_SECONDS,
        }


_watchdog = _Watchdog()


async def start() -> None:
    await _watchdog.start()


async def stop() -> None:
    await _watchdog.stop()


def status_snapshot() -> dict:
    return _watchdog.status_snapshot()
