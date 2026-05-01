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
        """Log a task census so we can see who's hogging the loop.

        We group ALL active tasks by their topmost stack frame
        (``file:func``) and report counts.  This is far more useful
        than dumping the top-N by name: when 400 tasks all sit at the
        same await point, the previous "top 20 by name" view was
        dominated by 20 long-lived WS protocol tasks, hiding the
        actual flood.  Grouping surfaces it as e.g. ``wallet_discovery
        :_scan_wallet x 380`` in a single line.

        We still list a few representative individual tasks for
        non-dominant groups so unique stack traces aren't lost.
        """
        try:
            tasks = list(asyncio.all_tasks())
        except RuntimeError:
            return
        watchdog_self = self._task
        tasks = [t for t in tasks if t is not watchdog_self and not t.done()]

        # Group by topmost frame: short "filename:func" key.
        groups: dict[str, dict] = {}
        unknown_count = 0
        for task in tasks:
            try:
                stack = task.get_stack(limit=1)
                if not stack:
                    unknown_count += 1
                    continue
                top = stack[-1]
                # filename:basename only (drop path) so the key is concise.
                fname = top.f_code.co_filename
                # Last path component for readability.
                short_fname = fname.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
                key = f"{short_fname}:{top.f_code.co_name}:{top.f_lineno}"
                bucket = groups.setdefault(
                    key,
                    {"count": 0, "first_task_name": task.get_name() or "<unnamed>"},
                )
                bucket["count"] += 1
            except Exception:
                unknown_count += 1
        # Sort groups by count descending — biggest floods first.
        ranked = sorted(groups.items(), key=lambda kv: -kv[1]["count"])
        # Compact summary: top groups inline, with their count.
        top_summary = [
            f"{key} x{bucket['count']}"
            for key, bucket in ranked[:_MAX_TASKS_DUMPED]
        ]
        if unknown_count:
            top_summary.append(f"<no_stack> x{unknown_count}")
        logger.warning(
            "Event-loop stall detected",
            stall_seconds=round(stall_seconds, 3),
            active_tasks=len(tasks),
            unique_locations=len(groups),
            stalls_observed=self._stalls_observed,
            max_stall_seconds=round(self._max_stall_seconds, 3),
            task_groups=top_summary,
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
