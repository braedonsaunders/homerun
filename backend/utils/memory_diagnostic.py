"""In-process memory diagnostic for hunting heap-growth leaks.

Triggered via a filesystem marker — drop a file at ``backend/.runtime/
memory_dump_request`` and the next time the diagnostic loop wakes up
(every 30 s), it will write a snapshot to ``backend/.runtime/
memory_dump.txt``.  No HTTP endpoint, no live-attach debugger, no
restart needed — only a 30 s file-poll loop in the worker.

Snapshot contents:

  1. ``psutil`` process memory breakdown (RSS / private / shared).
  2. ``tracemalloc`` top-50 sources of allocated bytes by file:line —
     this is the ground truth for which code path is allocating.
  3. Top-50 module-level dict / list / set / deque containers sorted by
     ``len()`` — answers the "which singleton is huge" question.
  4. ``gc`` statistics — generation counts and pause history.

Tracemalloc adds 5–10 % overhead per allocation, which is acceptable
for a worker that has a 12 GB leak; we'd rather have observability
than the perf.  The diagnostic loop itself does no work between
dumps so the steady-state cost is one ``Path.exists()`` call every
30 s.
"""

from __future__ import annotations

import asyncio
import gc
import linecache
import logging
import os
import sys
import time
import tracemalloc
from collections import deque
from pathlib import Path

logger = logging.getLogger("memory_diagnostic")

_RUNTIME_DIR = Path(__file__).resolve().parents[1] / ".runtime"
_REQUEST_PATH = _RUNTIME_DIR / "memory_dump_request"
_OUTPUT_PATH = _RUNTIME_DIR / "memory_dump.txt"
_POLL_INTERVAL_SECONDS = 30.0
_TOP_N = 50


def start_tracemalloc() -> None:
    """Enable tracemalloc with a deep-enough frame budget that the
    leaking allocation site is identifiable.

    Frame=10 is enough to find the actual source through asyncio
    callback wrappers + SQLAlchemy + asyncpg layers.  Cost: ~10 %
    runtime overhead and a few hundred MB of trace metadata at
    steady state — acceptable for diagnostic runs.
    """
    if not tracemalloc.is_tracing():
        tracemalloc.start(10)
        logger.info("tracemalloc enabled (frame=10)")


def _format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    if n < 1024 ** 3:
        return f"{n / 1024 ** 2:.1f} MB"
    return f"{n / 1024 ** 3:.2f} GB"


def _process_memory_section() -> str:
    """Process-level memory using the same Windows/Linux probes the
    worker already uses for ``rss_bytes`` in worker_snapshot.  Avoids
    a psutil dependency since the project doesn't already require it.
    """
    pid = os.getpid()
    lines = ["=== PROCESS MEMORY ===", f"PID: {pid}"]
    try:
        from services.worker_state import _read_process_rss_bytes, _read_peak_rss_bytes

        rss = _read_process_rss_bytes(pid)
        if rss is not None:
            lines.append(f"RSS / WorkingSet:   {_format_bytes(rss)}")
        peak = _read_peak_rss_bytes()
        if peak is not None:
            lines.append(f"Peak RSS:           {_format_bytes(peak)}")
    except Exception as exc:
        lines.append(f"(rss probe failed: {exc})")

    if tracemalloc.is_tracing():
        current, peak = tracemalloc.get_traced_memory()
        lines.append(f"tracemalloc.current: {_format_bytes(current)}")
        lines.append(f"tracemalloc.peak:    {_format_bytes(peak)}")
    lines.append("")
    return "\n".join(lines)


def _tracemalloc_section() -> str:
    if not tracemalloc.is_tracing():
        return "=== TRACEMALLOC ===\nNot enabled.\n\n"
    snapshot = tracemalloc.take_snapshot()
    snapshot = snapshot.filter_traces(
        (
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<frozen importlib._bootstrap_external>"),
            tracemalloc.Filter(False, tracemalloc.__file__),
        )
    )
    top = snapshot.statistics("lineno")[:_TOP_N]
    total = sum(stat.size for stat in snapshot.statistics("filename"))
    lines = [
        "=== TRACEMALLOC TOP ALLOCATIONS BY FILE:LINE ===",
        f"Tracked total: {_format_bytes(total)}",
        "",
    ]
    for i, stat in enumerate(top, 1):
        frame = stat.traceback[0]
        filename = Path(frame.filename).as_posix()
        lines.append(
            f"{i:>3}. {_format_bytes(stat.size):>10} | "
            f"{stat.count:>7} blocks | {filename}:{frame.lineno}"
        )
        line_src = linecache.getline(frame.filename, frame.lineno).strip()
        if line_src:
            lines.append(f"      {line_src}")
    lines.append("")
    return "\n".join(lines)


def _enumerate_singleton_containers() -> list[tuple[str, int, str]]:
    """Walk ``sys.modules`` for module-level dict/list/set/deque
    attributes whose len() looks suspicious.  Returns
    ``(qualified_name, length, type_name)`` tuples, sorted by length.

    Skips the standard library and a few known-noisy third-party
    libs to keep the output focused on application code.
    """
    skip_prefixes = (
        "_",
        "asyncio",
        "logging",
        "sqlalchemy",
        "alembic",
        "pydantic",
        "pkg_resources",
        "importlib",
        "concurrent",
        "encodings",
        "fastapi",
        "starlette",
        "uvicorn",
        "h11",
        "anyio",
        "websockets",
        "httpx",
        "httpcore",
        "_pytest",
        "pytest",
        "tracemalloc",
        "linecache",
    )
    # App-code prefixes get a much lower noise floor — a 50-entry
    # leak in services/workers code is more interesting than a
    # 1000-entry stdlib constant table.
    app_prefixes = ("services.", "workers.", "models.", "api.", "utils.", "strategies.", "main", "config")
    # Snapshot module list once.  Iterating ``sys.modules.items()``
    # while modules import (background tasks may import on the fly)
    # would otherwise dispatch to the dict's hot path and slow the
    # walk dramatically.
    snapshot = tuple(sys.modules.items())
    out: list[tuple[str, int, str]] = []
    for mod_name, mod in snapshot:
        if not mod or any(mod_name.startswith(p) for p in skip_prefixes):
            continue
        try:
            mod_dict = vars(mod)
        except TypeError:
            continue
        is_app = any(mod_name == p.rstrip(".") or mod_name.startswith(p) for p in app_prefixes)
        floor = 10 if is_app else 100
        # Snapshot the dict items so live mutation during the walk
        # doesn't corrupt iteration.
        for attr_name, value in tuple(mod_dict.items()):
            if attr_name.startswith("__"):
                continue
            t = type(value)
            if t not in (dict, list, set, frozenset, deque):
                continue
            try:
                n = len(value)
            except Exception:
                continue
            if n >= floor:
                out.append((f"{mod_name}.{attr_name}", n, t.__name__))
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def _module_singletons_section() -> str:
    items = _enumerate_singleton_containers()
    lines = [
        "=== MODULE-LEVEL CONTAINERS (len >= 100, sorted by len) ===",
        f"Found {len(items)} containers above the noise floor",
        "",
    ]
    for qualname, n, typ in items[:_TOP_N]:
        lines.append(f"{n:>10,} | {typ:<10} | {qualname}")
    lines.append("")
    return "\n".join(lines)


def _enumerate_instance_containers() -> list[tuple[str, int, str]]:
    """Walk every live object via ``gc.get_objects()`` and find instance
    attributes that are dict/list/set/deque with len() >= 100.

    Module-level dicts ONLY surface stuff like
    ``somemodule._global_cache``.  The application's actual hot state
    is on class instances:
    ``MarketCacheService._market_cache``, ``IntentRuntime._signals_by_id``,
    ``PriceCache._entries``, etc.  This walker finds those.

    Reports ``ClassName.attribute`` keyed by the instance class so a
    hundred-instance class with one big attr each shows up as one
    aggregate row, not one hundred.
    """
    aggregate: dict[tuple[str, str, str], int] = {}
    skip_class_prefixes = (
        "builtins.",
        "typing.",
        "_collections_abc.",
        "collections.",
        "weakref.",
        "asyncio.",
        "concurrent.",
        "logging.",
        "pydantic.",
        "sqlalchemy.",
        "google.",
    )
    objs = gc.get_objects()
    for obj in objs:
        cls = type(obj)
        try:
            cls_name = f"{cls.__module__}.{cls.__name__}"
        except Exception:
            continue
        if any(cls_name.startswith(p) for p in skip_class_prefixes):
            continue
        try:
            slots = getattr(obj, "__dict__", None)
        except Exception:
            continue
        if not isinstance(slots, dict):
            continue
        for attr, value in tuple(slots.items()):
            t = type(value)
            if t not in (dict, list, set, frozenset, deque):
                continue
            try:
                n = len(value)
            except Exception:
                continue
            if n < 100:
                continue
            key = (cls_name, attr, t.__name__)
            current = aggregate.get(key, 0)
            if n > current:
                aggregate[key] = n
    out = [
        (f"{cls_name}.{attr}", n, typ)
        for (cls_name, attr, typ), n in aggregate.items()
    ]
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def _instance_containers_section() -> str:
    items = _enumerate_instance_containers()
    lines = [
        "=== CLASS-INSTANCE CONTAINERS (len >= 100, max len per class.attr) ===",
        f"Found {len(items)} containers above the noise floor",
        "",
    ]
    for qualname, n, typ in items[:_TOP_N]:
        lines.append(f"{n:>10,} | {typ:<10} | {qualname}")
    lines.append("")
    return "\n".join(lines)


def _gc_section() -> str:
    counts = gc.get_count()
    stats = gc.get_stats()
    n_objects = len(gc.get_objects())
    lines = [
        "=== GARBAGE COLLECTOR ===",
        f"gc.get_count() (gen0/gen1/gen2): {counts}",
        f"len(gc.get_objects()): {n_objects:,}",
        f"gc.get_threshold(): {gc.get_threshold()}",
        f"gc enabled: {gc.isenabled()}",
        "",
        "Per-generation stats (collections / collected / uncollectable):",
    ]
    for i, s in enumerate(stats):
        lines.append(
            f"  gen{i}: collections={s.get('collections', 0):>10,} "
            f"collected={s.get('collected', 0):>12,} "
            f"uncollectable={s.get('uncollectable', 0)}"
        )
    lines.append("")
    return "\n".join(lines)


def _write_dump() -> None:
    started = time.monotonic()
    sections = [
        f"Memory dump generated at {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n",
        _process_memory_section(),
        _gc_section(),
        _module_singletons_section(),
        _instance_containers_section(),
        _tracemalloc_section(),
    ]
    body = "\n".join(sections)
    body += f"\n(dump generated in {time.monotonic() - started:.2f}s)\n"
    try:
        _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _OUTPUT_PATH.write_text(body, encoding="utf-8")
        logger.warning(
            "memory_diagnostic dump written to %s (%s)",
            _OUTPUT_PATH,
            _format_bytes(len(body.encode("utf-8"))),
        )
    except Exception as exc:
        logger.error("memory_diagnostic dump write failed: %s", exc)


async def memory_diagnostic_loop() -> None:
    """Background loop: poll the request file every 30 s, dump on hit.

    Designed to be added to the worker's background task list.  Runs
    forever; exits cleanly on cancellation.
    """
    logger.info(
        "memory_diagnostic loop started — touch %s to trigger a dump",
        _REQUEST_PATH,
    )
    def _poll_and_dump_sync() -> None:
        # Synchronous filesystem work (stat + dump-write + unlink) runs in a
        # worker thread: the 16h soak's event-loop watchdog caught the bare
        # ``_REQUEST_PATH.exists()`` os.stat blocking the SERVICES plane's
        # loop for 1.0-1.2s when the disk was busy with WAL-fsync storms.
        if not _REQUEST_PATH.exists():
            return
        _write_dump()
        try:
            _REQUEST_PATH.unlink()
        except Exception:
            pass

    while True:
        try:
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
            await asyncio.to_thread(_poll_and_dump_sync)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("memory_diagnostic loop iteration failed: %s", exc)
