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

from services.live_pressure import publish_backpressure
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


# Cap on cr_await traversal depth — guards against cycles or pathological
# deep coroutine chains.  Real firehose chains are rarely > 6 deep.
_MAX_CORO_DEPTH = 32


def _short_frame_key(frame: object) -> str | None:
    """Render ``filename:func:lineno`` for a Python frame object."""
    try:
        code = frame.f_code  # type: ignore[attr-defined]
        fname = code.co_filename
        short_fname = fname.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
        return f"{short_fname}:{code.co_name}:{frame.f_lineno}"  # type: ignore[attr-defined]
    except Exception:
        return None


# Frame filters: when the deepest suspended frame sits inside an
# asyncio/stdlib primitive (Semaphore.acquire, Lock.acquire, Queue.get,
# tasks.sleep, etc.), its file:func:lineno is the same for every parked
# task and gives ZERO signal about WHO is actually waiting.  We walk
# back up the chain to the first frame that sits in USER code and key
# by that frame instead, while appending a short tag so operators
# still see "parked on a Semaphore at caller_X:42".
#
# Simple substring match on the filename — robust across Python
# versions and avoids importing asyncio internals.
_STDLIB_FILENAME_MARKERS = (
    # Windows + POSIX path separators both end up here after rsplit.
    "asyncio/locks.py",
    "asyncio\\locks.py",
    "asyncio/queues.py",
    "asyncio\\queues.py",
    "asyncio/tasks.py",
    "asyncio\\tasks.py",
    "asyncio/base_events.py",
    "asyncio\\base_events.py",
    "asyncio/futures.py",
    "asyncio\\futures.py",
    "asyncio/streams.py",
    "asyncio\\streams.py",
)


def _frame_is_stdlib_asyncio(frame: object) -> bool:
    try:
        fname = frame.f_code.co_filename  # type: ignore[attr-defined]
    except Exception:
        return False
    # Normalize forward slashes for cross-platform matching.
    return any(marker in fname for marker in _STDLIB_FILENAME_MARKERS)


def _innermost_coro_frame_key(task: asyncio.Task) -> str | None:
    """Walk the cr_await chain to the deepest coroutine still suspended.

    Returns the ``filename:func:lineno`` for the frame where that inner
    coroutine is actually parked — i.e. the true blocking await point.
    Falls back to the outer ``task.get_stack(limit=1)`` frame if
    traversal isn't possible (e.g. C-extension coro, non-coroutine awaitable).

    If the deepest frame sits inside an asyncio stdlib primitive
    (Semaphore.acquire, Lock.acquire, Queue.get, sleep, etc.) we walk
    BACK UP the chain to the first user-code frame and append a short
    ``[via <primitive>:<lineno>]`` tag so operators can see who is
    parked on which primitive — ``Semaphore.acquire:386 x30`` is
    structurally useless but ``trader_hot_state.py:flush_audit:1452
    [via locks.py:acquire:386] x30`` identifies the real caller.
    """
    # Start from the Task's coroutine.  In CPython this is ``_coro``.
    coro = getattr(task, "_coro", None)
    if coro is None:
        # Fallback: use the regular stack.
        try:
            stack = task.get_stack(limit=1)
            if not stack:
                return None
            return _short_frame_key(stack[-1])
        except Exception:
            return None

    # Walk cr_await as far as it points to another coroutine.  Collect
    # EVERY frame along the chain so we can pick a non-stdlib one to
    # report when the innermost is stdlib-primitive noise.
    frames: list[object] = []
    current = coro
    for _ in range(_MAX_CORO_DEPTH):
        if current is None:
            break
        # Prefer cr_* (coroutine), then gi_* (generator-based coroutine),
        # then ag_* (async generator).
        frame = (
            getattr(current, "cr_frame", None)
            or getattr(current, "gi_frame", None)
            or getattr(current, "ag_frame", None)
        )
        if frame is not None:
            frames.append(frame)
        next_awaited = (
            getattr(current, "cr_await", None)
            or getattr(current, "gi_yieldfrom", None)
            or getattr(current, "ag_await", None)
        )
        # If the next awaited object isn't another coroutine/generator,
        # stop — we've reached the "leaf" that's actually suspended on
        # a Future or socket.
        if next_awaited is None:
            break
        if not (
            hasattr(next_awaited, "cr_frame")
            or hasattr(next_awaited, "gi_frame")
            or hasattr(next_awaited, "ag_frame")
        ):
            break
        current = next_awaited

    if not frames:
        # Fallback: use the Task's regular stack.
        try:
            stack = task.get_stack(limit=1)
            if not stack:
                return None
            return _short_frame_key(stack[-1])
        except Exception:
            return None

    leaf_frame = frames[-1]
    leaf_key = _short_frame_key(leaf_frame)

    # If the leaf sits in asyncio stdlib, walk back up the chain to
    # the first USER frame and report that, tagged with the primitive.
    if _frame_is_stdlib_asyncio(leaf_frame):
        for candidate in reversed(frames[:-1]):
            if not _frame_is_stdlib_asyncio(candidate):
                caller_key = _short_frame_key(candidate)
                if caller_key is not None and leaf_key is not None:
                    return f"{caller_key} [via {leaf_key}]"
                return caller_key or leaf_key
        # Every frame was asyncio stdlib — fall through and emit the
        # leaf as-is (rare; typically means a bare stdlib task).
    return leaf_key


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

        # Group by innermost frame of the task's coroutine chain.
        #
        # ``task.get_stack()`` only returns frames of the OUTER coroutine,
        # not coroutines that outer is awaiting.  When a task does
        # ``await inner_coro()`` and inner_coro suspends on I/O, the
        # Python stack says the outer task is parked at the ``await``
        # statement, which is useless for root-causing — we see "x30 at
        # _tracked_emission:393" (the wrapper) instead of the actual
        # blocking call inside buffer_trader_event or wherever.
        #
        # To drill through, we walk the ``cr_await`` chain from the
        # outer coroutine down to whatever coroutine is ACTUALLY
        # suspended on a non-coroutine awaitable (Future, asyncio.sleep,
        # socket read, etc.).  That gives us the real stall location.
        groups: dict[str, dict] = {}
        unknown_count = 0
        for task in tasks:
            try:
                key = _innermost_coro_frame_key(task)
                if key is None:
                    unknown_count += 1
                    continue
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
        publish_backpressure(
            "event_loop_watchdog",
            level=min(1.0, max(0.5, stall_seconds / 2.0)),
            reason=f"stall:{round(stall_seconds, 3)}s",
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
