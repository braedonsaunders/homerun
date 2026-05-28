"""Perf monitoring harness for the homerun trading worker plane.

Captures every WARNING / ERROR line emitted by the worker
processes, parses the structured event-loop-watchdog stalls,
``_persist_*`` slow-logs, ``Trader cycle slow`` reports, and
``LOCK CONTENTION`` warnings, and writes a JSON report.

Two modes:

1. **Tail mode** (recommended — runs alongside the normal GUI):
   Launch homerun with the new ``-Debug`` flag, which writes per-plane
   JSON log files to ``tools/.harness/<plane>.log``.  Then point the
   harness at those files::

       Homerun.bat -Debug
       python tools/perf_harness.py --tail --duration 1200 --planes trading,discovery

   The harness starts at EOF on each file and only aggregates NEW
   lines, so it can be started/stopped without disrupting the GUI.

2. **Spawn mode** (CI / headless): the harness spawns the worker
   planes itself.  Postgres + Redis must already be up::

       docker compose -f scripts/infra/docker-compose.infra.yml up -d
       python tools/perf_harness.py --duration 1200 --planes trading,discovery

After the duration elapses the harness:
  - In spawn mode, sends SIGTERM to all spawned workers (clean shutdown)
  - Writes ``tools/.harness/report.json`` with aggregated metrics
  - Writes ``tools/.harness/live.log`` with the raw last-N WARNING /
    ERROR lines

The report is the canonical input for the fix-iteration loop:
the operator (or AI agent driving the harness) reads the top
issues, edits code to address them, restarts the harness, and
repeats until the report shows no actionable signal.
"""

from __future__ import annotations

import argparse
import asyncio
import ast
import json
import os
import re
import signal
import sys
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HARNESS_DIR = PROJECT_ROOT / "tools" / ".harness"
DEFAULT_REPORT_PATH = HARNESS_DIR / "report.json"
DEFAULT_LIVE_TAIL_PATH = HARNESS_DIR / "live.log"


# ---------------------------------------------------------------------------
# Log line parsers
# ---------------------------------------------------------------------------

# The host process logs structured JSON.  When run via the launcher
# the output is prefixed with ``[WORKERS]``/``[NEWS]``/``[DISCOVERY]``
# but when we spawn it ourselves we can either get raw stdout (no
# prefix) or reuse the prefixing.  The harness handles both.
LOG_PREFIX_RE = re.compile(
    r"^\[(?P<plane>WORKERS|NEWS|DISCOVERY)\]\s+"
    r"(?P<rest>.*)$"
)
LOG_LEVEL_RE = re.compile(
    r"(?P<time>\d{2}:\d{2}:\d{2})\s+\[(?P<level>WARNING|ERROR|INFO|DEBUG|CRITICAL)\]\s+(?P<body>.*)$"
)
STALL_RE = re.compile(
    r"Event-loop stall detected\s+stall_seconds=(?P<stall>[\d.]+)\s+"
    r"active_tasks=(?P<tasks>\d+)\s+"
    r"unique_locations=(?P<unique>\d+)\s+"
    r"stalls_observed=(?P<n>\d+)\s+"
    r"max_stall_seconds=(?P<max>[\d.]+)\s+"
    r"task_groups=\[(?P<groups>.+)\]\s*$"
)
TASK_GROUP_RE = re.compile(r"'([^']+)\s+x(\d+)'")
SLOW_PERSIST_RE = re.compile(
    r"_persist_(?P<kind>orders|runtime_state) slow\s+breakdown=(?P<bd>\{.+\})\s*$"
)
PLACE_ORDER_SLOW_RE = re.compile(
    r"place_order slow\s+token_id=(?P<tok>\S+)\s+side=(?P<side>\w+)\s+"
    r"total_ms=(?P<total>[\d.]+)\s+status=(?P<status>\w+)\s+"
    r"breakdown=(?P<bd>\{.+\})\s*$"
)
SLOW_CYCLE_RE = re.compile(
    r"Trader cycle slow\s+trader_id=(?P<tid>\w+).*?duration_s=(?P<dur>[\d.]+).*?"
    r"stage_timings_ms=(?P<stages>\{.+\})\s*$"
)
LOCK_CONTENTION_RE = re.compile(
    r"LOCK CONTENTION\s+blocked_pid=(?P<pid>\d+)\s+age=(?P<age>\d+)s.*?wait=(?P<wait>\S+).*?"
    r"query='(?P<q>[^']{0,120})"
)
PER_SIGNAL_SLOW_RE = re.compile(
    r"Per-signal iteration slow\s+trader_id=(?P<tid>\w+).*?elapsed_ms=(?P<ms>[\d.]+).*?last_stage=(?P<stage>\S+)"
)
HEARTBEAT_STALE_RE = re.compile(
    r"Worker heartbeat stale.*worker=(?P<worker>\w+)\s+age_seconds=(?P<age>[\d.]+)"
)


def _safe_eval_dict(text: str) -> dict[str, Any]:
    """Best-effort parse of a structured dict-shaped log payload.

    Logs use Python repr (e.g. ``{'orders': 4, 'commit': 47.0}``);
    ``ast.literal_eval`` is the safe parser for that shape.  Returns
    ``{}`` on any failure — the harness must never crash on a
    malformed log line.
    """
    try:
        result = ast.literal_eval(text)
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


class Aggregator:
    """Collects per-line metrics and produces a JSON report.

    All buckets are bounded — a long-running harness can't blow up
    memory.  The recent-warnings deque caps at 200 to keep the live
    tail file readable.
    """

    def __init__(self) -> None:
        self.start_mono = time.monotonic()
        self.start_wall = datetime.now(timezone.utc)
        self.lines_seen = 0
        self.by_level: Counter = Counter()
        self.by_plane: Counter = Counter()
        # Stall metrics
        self.stall_seconds: list[float] = []
        self.stall_active_tasks: list[int] = []
        self.task_group_freq: Counter = Counter()  # pattern → # of stalls it appeared in
        # ``self.stall_top5_seen`` tracks how often each stack frame
        # appears in the top-5 of any stall — the most actionable
        # leaderboard.  Anything appearing in 50 %+ of dumps is a
        # chronic background offender.
        self.stall_top5_seen: Counter = Counter()
        # Slow-log captures
        self.slow_persist_orders: list[dict[str, Any]] = []
        self.slow_persist_runtime_state: list[dict[str, Any]] = []
        self.slow_place_order: list[dict[str, Any]] = []
        self.slow_cycles: list[dict[str, Any]] = []
        self.per_signal_slow: list[dict[str, Any]] = []
        # Misc
        self.lock_contentions: int = 0
        self.heartbeat_stale: list[dict[str, Any]] = []
        self.errors: Counter = Counter()
        self.error_samples: dict[str, str] = {}  # bucket → first-seen full line
        self.warnings_other: Counter = Counter()
        self.warnings_other_samples: dict[str, str] = {}  # bucket → first-seen full line
        # Live tail
        self.recent: deque = deque(maxlen=200)

    # ---- Feed ----

    def feed_line(self, line: str) -> None:
        self.lines_seen += 1
        line = line.rstrip()
        if not line:
            return
        plane = "WORKERS"  # default — direct stdout has no prefix
        m = LOG_PREFIX_RE.match(line)
        if m:
            plane = m.group("plane")
            line_body = m.group("rest")
        else:
            line_body = line
        self.by_plane[plane] += 1
        m = LOG_LEVEL_RE.match(line_body)
        if not m:
            return
        level = m.group("level")
        body = m.group("body")
        self.by_level[level] += 1
        if level == "WARNING":
            self._bucket_warning(plane, body, line)
        elif level == "ERROR":
            self._bucket_error(plane, body, line)
        elif level == "CRITICAL":
            self._bucket_error(plane, body, line)

    # ---- Warning bucketing ----

    def _bucket_warning(self, plane: str, body: str, raw: str) -> None:
        # Stalls — biggest signal
        m = STALL_RE.search(body)
        if m:
            stall = float(m.group("stall"))
            tasks = int(m.group("tasks"))
            groups = TASK_GROUP_RE.findall(m.group("groups"))
            top5 = [(g, int(n)) for g, n in groups[:5]]
            self.stall_seconds.append(stall)
            self.stall_active_tasks.append(tasks)
            for g, _ in groups:
                self.task_group_freq[g] += 1
            for g, _ in top5:
                self.stall_top5_seen[g] += 1
            self.recent.append({
                "kind": "stall",
                "stall_seconds": stall,
                "active_tasks": tasks,
                "top_groups": top5,
            })
            return
        # _persist_orders / _persist_runtime_state slow
        m = SLOW_PERSIST_RE.search(body)
        if m:
            kind = m.group("kind")
            bd = _safe_eval_dict(m.group("bd"))
            entry = {"kind": kind, "breakdown": bd}
            (self.slow_persist_orders if kind == "orders" else self.slow_persist_runtime_state).append(entry)
            self.recent.append({"kind": f"slow_persist_{kind}", "breakdown": bd})
            return
        # place_order slow
        m = PLACE_ORDER_SLOW_RE.search(body)
        if m:
            entry = {
                "side": m.group("side"),
                "total_ms": float(m.group("total")),
                "status": m.group("status"),
                "breakdown": _safe_eval_dict(m.group("bd")),
            }
            self.slow_place_order.append(entry)
            self.recent.append({"kind": "place_order_slow", **entry})
            return
        # Trader cycle slow
        m = SLOW_CYCLE_RE.search(body)
        if m:
            entry = {
                "trader_id": m.group("tid"),
                "duration_s": float(m.group("dur")),
                "stages": _safe_eval_dict(m.group("stages")),
            }
            self.slow_cycles.append(entry)
            self.recent.append({"kind": "slow_cycle", **entry})
            return
        # LOCK CONTENTION
        m = LOCK_CONTENTION_RE.search(body)
        if m:
            self.lock_contentions += 1
            self.recent.append({
                "kind": "lock_contention",
                "wait": m.group("wait"),
                "query_excerpt": m.group("q"),
            })
            return
        # Per-signal iteration slow (orchestrator stage diag)
        m = PER_SIGNAL_SLOW_RE.search(body)
        if m:
            self.per_signal_slow.append({
                "trader_id": m.group("tid"),
                "elapsed_ms": float(m.group("ms")),
                "last_stage": m.group("stage"),
            })
            return
        # Worker heartbeat stale (forced restart)
        m = HEARTBEAT_STALE_RE.search(body)
        if m:
            self.heartbeat_stale.append({
                "worker": m.group("worker"),
                "age_seconds": float(m.group("age")),
            })
            return
        # Fallback — bucket by first 100 chars of the body
        bucket = body[:100]
        self.warnings_other[bucket] += 1
        # Keep the first FULL sample for each bucket so the report
        # preserves dynamic context (task names, coro names, ids) that
        # the 100-char bucket key truncates.  Cycle 3 lost the
        # ``Connection held for 97.8s before return to pool
        # (task=..., coro=...)`` task names this way.
        if bucket not in self.warnings_other_samples:
            self.warnings_other_samples[bucket] = raw[:600]

    def _bucket_error(self, plane: str, body: str, raw: str) -> None:
        bucket = body[:100]
        self.errors[bucket] += 1
        if bucket not in self.error_samples:
            self.error_samples[bucket] = raw[:400]
        self.recent.append({"kind": "error", "msg": body[:300]})

    # ---- Report ----

    def report(self) -> dict[str, Any]:
        elapsed_s = round(time.monotonic() - self.start_mono, 1)
        n_stalls = len(self.stall_seconds)
        sorted_stalls = sorted(self.stall_seconds)

        def pct(p: float) -> float:
            if not sorted_stalls:
                return 0.0
            idx = max(0, min(n_stalls - 1, int(n_stalls * p)))
            return round(sorted_stalls[idx], 3)

        def topn(c: Counter, n: int = 25) -> list[dict[str, Any]]:
            return [{"key": k, "count": v} for k, v in c.most_common(n)]

        def topn_with_samples(
            c: Counter,
            samples: dict[str, str],
            n: int = 25,
        ) -> list[dict[str, Any]]:
            return [
                {"key": k, "count": v, "sample": samples.get(k, "")}
                for k, v in c.most_common(n)
            ]

        # Per-trader slow cycle summary
        cycles_by_trader: dict[str, list[float]] = defaultdict(list)
        for c in self.slow_cycles:
            cycles_by_trader[c["trader_id"]].append(c["duration_s"])
        cycles_summary = {
            tid: {
                "count": len(durs),
                "max_s": round(max(durs), 2),
                "p50_s": round(sorted(durs)[len(durs) // 2], 2),
            }
            for tid, durs in cycles_by_trader.items()
        }
        # Top stages contributing to slow cycles
        stage_offenders: Counter = Counter()
        for c in self.slow_cycles:
            for stage, ms in (c.get("stages") or {}).items():
                if isinstance(ms, (int, float)) and ms >= 1000:
                    stage_offenders[stage] += int(ms)

        # Top sub-stages from place_order slow logs
        place_order_stage_total_ms: Counter = Counter()
        for entry in self.slow_place_order:
            for stage, ms in (entry.get("breakdown") or {}).items():
                if isinstance(ms, (int, float)) and stage != "total_ms" and stage != "attempts":
                    place_order_stage_total_ms[stage] += int(ms)

        # Top sub-stages from persist slow logs
        persist_orders_stage_total_ms: Counter = Counter()
        for entry in self.slow_persist_orders:
            for stage, ms in (entry.get("breakdown") or {}).items():
                if isinstance(ms, (int, float)) and stage not in {"total_ms", "attempts", "orders"}:
                    persist_orders_stage_total_ms[stage] += int(ms)
        persist_state_stage_total_ms: Counter = Counter()
        for entry in self.slow_persist_runtime_state:
            for stage, ms in (entry.get("breakdown") or {}).items():
                if isinstance(ms, (int, float)) and stage not in {"total_ms", "attempts"}:
                    persist_state_stage_total_ms[stage] += int(ms)

        return {
            "harness": {
                "started_at_utc": self.start_wall.isoformat(),
                "duration_seconds": elapsed_s,
                "lines_seen": self.lines_seen,
                "by_level": dict(self.by_level),
                "by_plane": dict(self.by_plane),
            },
            "stalls": {
                "count": n_stalls,
                "p50_s": pct(0.5),
                "p90_s": pct(0.9),
                "p99_s": pct(0.99),
                "max_s": round(max(sorted_stalls), 3) if sorted_stalls else 0.0,
                "active_tasks_p50": (
                    sorted(self.stall_active_tasks)[len(self.stall_active_tasks) // 2]
                    if self.stall_active_tasks else 0
                ),
                "active_tasks_p99": (
                    sorted(self.stall_active_tasks)[int(len(self.stall_active_tasks) * 0.99)]
                    if self.stall_active_tasks else 0
                ),
                "active_tasks_max": max(self.stall_active_tasks) if self.stall_active_tasks else 0,
            },
            "stall_top5_offenders": topn(self.stall_top5_seen, 25),
            "stall_all_groups": topn(self.task_group_freq, 25),
            "slow_cycles": {
                "count": len(self.slow_cycles),
                "by_trader": cycles_summary,
                "stage_total_ms": [{"stage": k, "total_ms": v} for k, v in stage_offenders.most_common(20)],
            },
            "slow_place_order": {
                "count": len(self.slow_place_order),
                "samples": self.slow_place_order[-5:],
                "stage_total_ms": [{"stage": k, "total_ms": v} for k, v in place_order_stage_total_ms.most_common(15)],
            },
            "slow_persist_orders": {
                "count": len(self.slow_persist_orders),
                "samples": self.slow_persist_orders[-5:],
                "stage_total_ms": [{"stage": k, "total_ms": v} for k, v in persist_orders_stage_total_ms.most_common(10)],
            },
            "slow_persist_runtime_state": {
                "count": len(self.slow_persist_runtime_state),
                "samples": self.slow_persist_runtime_state[-5:],
                "stage_total_ms": [{"stage": k, "total_ms": v} for k, v in persist_state_stage_total_ms.most_common(10)],
            },
            "per_signal_slow": {
                "count": len(self.per_signal_slow),
                "samples": self.per_signal_slow[-10:],
            },
            "lock_contentions": self.lock_contentions,
            "heartbeat_stale": self.heartbeat_stale,
            "top_errors": [
                {"key": k, "count": v, "sample": self.error_samples.get(k, "")[:300]}
                for k, v in self.errors.most_common(20)
            ],
            "top_other_warnings": topn_with_samples(
                self.warnings_other, self.warnings_other_samples, 25
            ),
        }


# ---------------------------------------------------------------------------
# Worker spawning + IO
# ---------------------------------------------------------------------------


async def _stream_to_aggregator(
    stream: asyncio.StreamReader,
    plane: str,
    agg: Aggregator,
    live_tail: Path,
) -> None:
    """Read stdout/stderr line by line, prefix with plane tag for the
    live tail, and feed every line to the aggregator."""
    while True:
        try:
            raw = await stream.readline()
        except (asyncio.CancelledError, Exception):
            return
        if not raw:
            return
        try:
            line = raw.decode("utf-8", errors="replace")
        except Exception:
            continue
        # Re-prefix so the aggregator's plane bucket works regardless
        # of whether the worker emitted a prefix or not.  We always
        # write the prefixed form to the live tail.
        prefixed = _normalize_line_for_aggregator(plane, line)
        agg.feed_line(prefixed)
        try:
            with live_tail.open("a", encoding="utf-8") as fh:
                fh.write(prefixed)
        except Exception:
            pass


def _normalize_line_for_aggregator(plane: str, line: str) -> str:
    """Convert a raw worker line to the bracketed text format the
    aggregator's regexes expect.

    Workers emit JSON via ``utils.logger.JSONFormatter``.  When a JSON
    line is detected, parse it and reshape into ``[<plane>] HH:MM:SS
    [<level>] <logger> <function>() <message> [extra=value ...]``
    (matches ``gui.py:format_log_line``).  Plain lines pass through
    with the plane prefix added if missing.
    """
    line = line.rstrip("\r\n")
    if not line:
        return line + "\n"
    if line.startswith("{"):
        try:
            data = json.loads(line)
        except Exception:
            data = None
        if isinstance(data, dict):
            level = str(data.get("level", "INFO")).upper()
            msg = str(data.get("message", line))
            logger_name = str(data.get("logger", ""))
            func = str(data.get("function", ""))
            ts = str(data.get("timestamp", ""))
            ts_short = ""
            if ts:
                try:
                    ts_short = ts.split("T")[1].split(".")[0] if "T" in ts else ts[-8:]
                except Exception:
                    ts_short = ts[:19]
            parts = [f"[{plane}]"]
            if ts_short:
                parts.append(ts_short)
            parts.append(f"[{level}]")
            if logger_name:
                parts.append(logger_name)
            if func:
                parts.append(f"{func}()")
            parts.append(msg)
            extra = data.get("data")
            if isinstance(extra, dict):
                for k, v in extra.items():
                    if v is not None:
                        parts.append(f"{k}={v}")
            return " ".join(parts) + "\n"
    # Plain text — add plane prefix if not already there.
    if line.startswith("["):
        return line + "\n"
    return f"[{plane}] {line}\n"


async def _tail_file(
    file_path: Path,
    plane: str,
    agg: Aggregator,
    live_tail: Path,
    stop_event: asyncio.Event,
) -> None:
    """Tail a JSON log file from EOF until ``stop_event`` is set.

    Uses simple line-buffered polling (50 ms) — sufficient for the
    multi-second WARNING cadence, and avoids the platform-specific
    complexity of inotify / ReadDirectoryChangesW.  Re-opens the
    file if it gets truncated (e.g. the launcher recreates it on
    plane spawn).
    """
    fh = None
    while not stop_event.is_set():
        try:
            if fh is None:
                if not file_path.exists():
                    await asyncio.sleep(0.5)
                    continue
                fh = file_path.open("r", encoding="utf-8", errors="replace")
                fh.seek(0, 2)  # EOF — only read NEW lines
            line = fh.readline()
            if not line:
                # Detect truncation — if file shrunk, reopen at start.
                try:
                    cur_size = file_path.stat().st_size
                    if cur_size < fh.tell():
                        fh.close()
                        fh = None
                        continue
                except Exception:
                    pass
                await asyncio.sleep(0.05)
                continue
            prefixed = _normalize_line_for_aggregator(plane, line)
            agg.feed_line(prefixed)
            try:
                with live_tail.open("a", encoding="utf-8") as out:
                    out.write(prefixed)
            except Exception:
                pass
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(0.5)
    if fh is not None:
        try:
            fh.close()
        except Exception:
            pass


async def _spawn_plane(
    plane_name: str,
    backend_dir: Path,
    venv_python: Path,
) -> asyncio.subprocess.Process:
    """Spawn ``python -m workers.host <plane>`` with stdout+stderr captured."""
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HOMERUN_PROCESS_ROLE", "worker")
    env.setdefault("LOG_LEVEL", "INFO")
    return await asyncio.create_subprocess_exec(
        str(venv_python),
        "-m",
        "workers.host",
        plane_name,
        cwd=str(backend_dir),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )


def _resolve_venv_python() -> Path:
    """Locate the homerun backend venv's python.

    Homerun ships its venv at ``backend/venv/Scripts/python.exe`` (or
    ``backend/venv/bin/python`` on POSIX) — that's the only env where
    the workers' deps are installed.  Falls back to a few other
    common venv layouts, then to the current interpreter as a last
    resort.
    """
    candidates = [
        PROJECT_ROOT / "backend" / "venv" / "Scripts" / "python.exe",
        PROJECT_ROOT / "backend" / "venv" / "bin" / "python",
        PROJECT_ROOT / "backend" / ".venv" / "Scripts" / "python.exe",
        PROJECT_ROOT / "backend" / ".venv" / "bin" / "python",
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
        PROJECT_ROOT / ".venv" / "bin" / "python",
        PROJECT_ROOT / "venv" / "Scripts" / "python.exe",
        PROJECT_ROOT / "venv" / "bin" / "python",
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path(sys.executable)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def _run_tail(
    *,
    duration: int,
    log_files: list[tuple[str, Path]],
    report_path: Path,
    live_tail: Path,
    write_interval: int,
) -> dict[str, Any]:
    """Tail-mode: consume existing JSON log files written by an
    already-running launcher with ``-Debug`` enabled.

    No worker subprocesses are spawned.  ``log_files`` is a list of
    ``(plane_tag, path)`` tuples — typically the trading / discovery
    / news files written under ``tools/.harness/``.  The harness
    starts at EOF on each (only NEW lines are aggregated) so multiple
    overlapping harness windows don't double-count.
    """
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    live_tail.write_text("", encoding="utf-8")
    agg = Aggregator()
    stop_event = asyncio.Event()
    print(f"[harness] tail mode — files={[str(p) for _, p in log_files]} duration={duration}s")
    readers: list[asyncio.Task] = []
    for plane, path in log_files:
        readers.append(asyncio.create_task(
            _tail_file(path, plane, agg, live_tail, stop_event),
            name=f"harness-tail-{plane}",
        ))

    async def _periodic_writer() -> None:
        try:
            while not stop_event.is_set():
                await asyncio.sleep(write_interval)
                _write_report(agg.report(), report_path)
        except asyncio.CancelledError:
            return

    writer_task = asyncio.create_task(_periodic_writer(), name="harness-writer")
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=duration)
    except asyncio.TimeoutError:
        pass
    except asyncio.CancelledError:
        pass
    finally:
        stop_event.set()
        for r in readers:
            r.cancel()
        writer_task.cancel()

    report = agg.report()
    _write_report(report, report_path)
    return report


async def _run(
    *,
    duration: int,
    planes: list[str],
    report_path: Path,
    live_tail: Path,
    write_interval: int,
) -> dict[str, Any]:
    backend_dir = PROJECT_ROOT / "backend"
    venv_python = _resolve_venv_python()
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    # Truncate live tail at the start of each run.
    live_tail.write_text("", encoding="utf-8")

    agg = Aggregator()
    procs: dict[str, asyncio.subprocess.Process] = {}
    readers: list[asyncio.Task] = []

    print(f"[harness] starting planes={planes} duration={duration}s python={venv_python}")
    for plane in planes:
        proc = await _spawn_plane(plane, backend_dir, venv_python)
        procs[plane] = proc
        plane_tag = {"trading": "WORKERS", "discovery": "DISCOVERY", "news": "NEWS"}.get(plane, plane.upper())
        if proc.stdout is not None:
            readers.append(asyncio.create_task(
                _stream_to_aggregator(proc.stdout, plane_tag, agg, live_tail),
                name=f"harness-reader-{plane}",
            ))
        print(f"[harness] spawned {plane} pid={proc.pid}")

    # Periodic snapshot writer — useful when the harness is killed early.
    async def _periodic_writer() -> None:
        try:
            while True:
                await asyncio.sleep(write_interval)
                _write_report(agg.report(), report_path)
        except asyncio.CancelledError:
            return

    writer_task = asyncio.create_task(_periodic_writer(), name="harness-writer")

    deadline = time.monotonic() + duration
    try:
        while time.monotonic() < deadline:
            # Bail early if a worker died unexpectedly — surface it
            # in the report rather than silently sitting on a stalled
            # stream.
            for plane, proc in procs.items():
                if proc.returncode is not None:
                    print(f"[harness] {plane} exited prematurely with rc={proc.returncode}")
                    deadline = time.monotonic()  # exit the outer loop
                    break
            await asyncio.sleep(2.0)
    except asyncio.CancelledError:
        pass
    finally:
        # Clean shutdown — workers will flush their buffered audit
        # writes and trigger their own cleanup paths.
        print("[harness] sending SIGTERM to workers")
        for plane, proc in procs.items():
            if proc.returncode is None:
                try:
                    if sys.platform == "win32":
                        # No SIGTERM on Windows; CTRL_BREAK_EVENT isn't
                        # propagated to subprocesses we didn't spawn
                        # with CREATE_NEW_PROCESS_GROUP.  Use terminate()
                        # which sends TerminateProcess — graceful enough
                        # for ~5s shutdown.
                        proc.terminate()
                    else:
                        proc.send_signal(signal.SIGTERM)
                except Exception as exc:
                    print(f"[harness] failed to signal {plane}: {exc}")
        # Give workers up to 30 s to drain.
        end_drain = time.monotonic() + 30.0
        while time.monotonic() < end_drain:
            still_alive = [p for p in procs.values() if p.returncode is None]
            if not still_alive:
                break
            await asyncio.sleep(0.5)
        # Force-kill any holdouts.
        for plane, proc in procs.items():
            if proc.returncode is None:
                print(f"[harness] force-killing {plane} pid={proc.pid}")
                try:
                    proc.kill()
                except Exception:
                    pass
        for r in readers:
            r.cancel()
        writer_task.cancel()

    report = agg.report()
    _write_report(report, report_path)
    return report


def _write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _summarize_report(report: dict[str, Any]) -> str:
    """Compact text summary suitable for stdout / chat output."""
    h = report.get("harness", {})
    s = report.get("stalls", {})
    sc = report.get("slow_cycles", {})
    spo = report.get("slow_persist_orders", {})
    sps = report.get("slow_persist_runtime_state", {})
    splo = report.get("slow_place_order", {})
    lines = [
        f"=== harness summary ({h.get('duration_seconds', 0):.0f}s, {h.get('lines_seen', 0)} lines) ===",
        f"levels: {h.get('by_level', {})}",
        f"stalls: count={s.get('count', 0)} p50={s.get('p50_s', 0)}s p90={s.get('p90_s', 0)}s "
        f"p99={s.get('p99_s', 0)}s max={s.get('max_s', 0)}s "
        f"active_tasks p50/p99/max={s.get('active_tasks_p50', 0)}/{s.get('active_tasks_p99', 0)}/{s.get('active_tasks_max', 0)}",
        f"slow_cycles={sc.get('count', 0)}  "
        f"slow_place_order={splo.get('count', 0)}  "
        f"slow_persist_orders={spo.get('count', 0)}  "
        f"slow_persist_runtime_state={sps.get('count', 0)}  "
        f"lock_contentions={report.get('lock_contentions', 0)}",
        "",
        "top stall offenders (top-5 of any stall):",
    ]
    for entry in (report.get("stall_top5_offenders") or [])[:10]:
        lines.append(f"  {entry['count']:>4}  {entry['key']}")
    if sc.get("by_trader"):
        lines.append("\nslow cycles by trader:")
        for tid, info in sorted(sc["by_trader"].items(), key=lambda x: -x[1]["count"])[:5]:
            lines.append(f"  {info['count']:>3}  trader={tid}  max={info['max_s']}s  p50={info['p50_s']}s")
    if sc.get("stage_total_ms"):
        lines.append("\nslow-cycle stage time (cumulative ms across all slow cycles):")
        for entry in sc["stage_total_ms"][:10]:
            lines.append(f"  {entry['total_ms']:>8}ms  {entry['stage']}")
    if splo.get("stage_total_ms"):
        lines.append("\nplace_order sub-stage time (cumulative ms):")
        for entry in splo["stage_total_ms"][:10]:
            lines.append(f"  {entry['total_ms']:>8}ms  {entry['stage']}")
    if spo.get("stage_total_ms"):
        lines.append("\n_persist_orders sub-stage time (cumulative ms):")
        for entry in spo["stage_total_ms"][:8]:
            lines.append(f"  {entry['total_ms']:>8}ms  {entry['stage']}")
    if sps.get("stage_total_ms"):
        lines.append("\n_persist_runtime_state sub-stage time (cumulative ms):")
        for entry in sps["stage_total_ms"][:8]:
            lines.append(f"  {entry['total_ms']:>8}ms  {entry['stage']}")
    if report.get("top_errors"):
        lines.append("\ntop ERRORS:")
        for entry in report["top_errors"][:10]:
            lines.append(f"  {entry['count']:>4}  {entry['key'][:90]}")
    if report.get("heartbeat_stale"):
        lines.append(f"\nworker heartbeat stale events: {len(report['heartbeat_stale'])}")
        for h in report["heartbeat_stale"][:5]:
            lines.append(f"  worker={h['worker']} age={h['age_seconds']}s")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration",
        type=int,
        default=1200,
        help="Capture window in seconds (default: 1200 = 20 min).",
    )
    parser.add_argument(
        "--planes",
        type=str,
        default="trading",
        help="Comma-separated plane names to spawn (default: trading).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=f"Where to write the JSON report (default: {DEFAULT_REPORT_PATH}).",
    )
    parser.add_argument(
        "--live-tail",
        type=Path,
        default=DEFAULT_LIVE_TAIL_PATH,
        help=f"Where to mirror the raw log lines (default: {DEFAULT_LIVE_TAIL_PATH}).",
    )
    parser.add_argument(
        "--write-interval",
        type=int,
        default=60,
        help="How often to checkpoint the report mid-run (seconds, default: 60).",
    )
    parser.add_argument(
        "--summarize",
        type=Path,
        default=None,
        help="Skip running; just print a text summary of an existing JSON report.",
    )
    parser.add_argument(
        "--tail",
        action="store_true",
        help=(
            "Tail-mode: consume existing log files (written by Homerun.bat -Debug) "
            "instead of spawning workers.  Recommended when running alongside the "
            "normal GUI-based launcher.  See --tail-files."
        ),
    )
    parser.add_argument(
        "--tail-files",
        type=str,
        default="",
        help=(
            "Comma-separated <plane>:<path> pairs of log files to tail.  Default "
            "infers from --planes: 'trading:tools/.harness/trading.log,...'"
        ),
    )
    args = parser.parse_args()

    if args.summarize is not None:
        report = json.loads(args.summarize.read_text(encoding="utf-8"))
        print(_summarize_report(report))
        return 0

    planes = [p.strip() for p in args.planes.split(",") if p.strip()]
    if not planes:
        print("error: --planes must contain at least one plane name", file=sys.stderr)
        return 2

    try:
        if args.tail:
            log_files = _resolve_tail_files(args.tail_files, planes)
            report = asyncio.run(_run_tail(
                duration=args.duration,
                log_files=log_files,
                report_path=args.report,
                live_tail=args.live_tail,
                write_interval=args.write_interval,
            ))
        else:
            report = asyncio.run(_run(
                duration=args.duration,
                planes=planes,
                report_path=args.report,
                live_tail=args.live_tail,
                write_interval=args.write_interval,
            ))
    except KeyboardInterrupt:
        print("[harness] interrupted; report should be at " + str(args.report))
        return 130

    print(_summarize_report(report))
    return 0


def _resolve_tail_files(spec: str, planes: list[str]) -> list[tuple[str, Path]]:
    """Resolve the ``--tail-files`` argument to a list of (plane_tag, path).

    Empty spec → derive from ``--planes`` using the default path
    template ``tools/.harness/<plane>.log``.  Plane → tag mapping
    matches ``gui.py:_WORKER_PLANES``.
    """
    plane_to_tag = {"trading": "WORKERS", "discovery": "DISCOVERY", "news": "NEWS"}
    if spec.strip():
        out: list[tuple[str, Path]] = []
        for entry in spec.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if ":" in entry:
                plane, path = entry.split(":", 1)
                tag = plane_to_tag.get(plane.strip().lower(), plane.strip().upper())
                out.append((tag, Path(path.strip())))
            else:
                out.append(("WORKERS", Path(entry)))
        return out
    # Default — derive from planes.
    return [
        (plane_to_tag.get(p, p.upper()), HARNESS_DIR / f"{p}.log")
        for p in planes
    ]


if __name__ == "__main__":
    raise SystemExit(_main())
