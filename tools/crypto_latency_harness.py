"""Sub-second end-to-end crypto trading latency harness.

Aggregates existing structured log lines emitted by the live worker plane
into per-stage p50/p90/p99 reports for the crypto-trading hot path.  This
tool is *purely read-side* — it does not modify any production behaviour
or instrument any new code paths.  It works against:

  - ``[WORKERS] HH:MM:SS [WARNING] trader_orchestrator_worker
    _run_trader_once_inner() Trader cycle slow ... mode=live ...
    stage_timings_ms={'maintenance': ..., 'setup': ..., 'signal_loop': ...,
    'ps_submit_order': ..., 'ps_decision_writes': ...,
    'submit_validate_reserve': ..., 'submit_create_market_order': ...,
    'submit_post_order': ..., 'submit_persist_runtime_state_inner': ...,
    'submit_persist_orders': ..., 'submit_leg_execute_live_order': ...}``

  - ``[WORKERS] HH:MM:SS [WARNING] services.live_execution_service
    place_order() place_order slow token_id=... side=... total_ms=...
    status=... breakdown={'vpn_check': ..., 'validate_reserve': ...,
    'sync_transport': ..., 'refresh_signature_type': ..., 'io_lock_wait':
    ..., 'create_market_order': ..., 'post_order': ...,
    'post_placement_fill_fetch': ..., 'persist_orders': ...,
    'persist_runtime_state_inner': ..., 'total_ms': ...}``

  - ``[WORKERS] HH:MM:SS [WARNING] trader_orchestrator_worker
    _maybe_log_execution_latency_sla_breach() Execution latency SLA
    breached lane=crypto stage_key=ws_release_to_submit_start_ms ...
    p95=... p99=... worst_strategy=...``

  - (After Commit 2) ``[WORKERS] HH:MM:SS [INFO] crypto_latency_trace
    signal_id=... breakdown_ms={...}`` — the canonical per-trade trace.

Modes:

  - ``--tail``  : tail the existing per-plane log files written by the
                  launcher's ``-Debug`` mode (default
                  ``tools/.harness/<plane>.log``).
  - ``--from-file FILE``: read a static log file (e.g. a copy of live.log
                  from a prior run).
  - default     : read from stdin.

Output:

  - ``--report PATH`` writes a JSON report with per-stage percentiles
    and the worst-N samples (default
    ``tools/.harness/crypto_latency_report.json``).
  - Stdout: a compact human-readable summary at the end of each capture
    window.

Re-running on a long log file is idempotent — every parse re-reads from
the start of the file.  Re-running in tail mode resumes at EOF.

Usage::

    # Live tail of the trading plane (assumes -Debug mode launcher):
    python tools/crypto_latency_harness.py --tail --duration 600

    # Offline analysis of the rolling tail file:
    python tools/crypto_latency_harness.py --from-file tools/.harness/live.log
"""
from __future__ import annotations

import argparse
import asyncio
import ast
import json
import re
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HARNESS_DIR = PROJECT_ROOT / "tools" / ".harness"
DEFAULT_REPORT_PATH = HARNESS_DIR / "crypto_latency_report.json"
DEFAULT_LOG_FILES = (
    ("WORKERS", HARNESS_DIR / "trading.log"),
    ("DISCOVERY", HARNESS_DIR / "discovery.log"),
    ("NEWS", HARNESS_DIR / "news.log"),
)


# ---------------------------------------------------------------------------
# Log line parsers
# ---------------------------------------------------------------------------

# Bracketed plane prefix — same shape the launcher and perf_harness emit.
LOG_PREFIX_RE = re.compile(
    r"^\[(?P<plane>WORKERS|NEWS|DISCOVERY)\]\s+(?P<rest>.*)$"
)
# Free-form level / timestamp — same shape ``perf_harness.py`` parses.
LOG_LEVEL_RE = re.compile(
    r"(?P<time>\d{2}:\d{2}:\d{2})\s+\[(?P<level>WARNING|ERROR|INFO|DEBUG|CRITICAL)\]\s+(?P<body>.*)$"
)

TRADER_CYCLE_SLOW_RE = re.compile(
    r"Trader cycle slow\s+trader_id=(?P<tid>\w+)\s+mode=(?P<mode>\w+)\s+"
    r"duration_s=(?P<dur>[\d.]+).*?stage_timings_ms=(?P<stages>\{.+\})\s*$"
)
PLACE_ORDER_SLOW_RE = re.compile(
    r"place_order slow\s+token_id=(?P<tok>\S+)\s+side=(?P<side>\w+)\s+"
    r"total_ms=(?P<total>[\d.]+)\s+status=(?P<status>\w+)\s+"
    r"breakdown=(?P<bd>\{.+\})\s*$"
)
SLA_BREACH_RE = re.compile(
    r"Execution latency SLA breached\s+lane=(?P<lane>\w+)\s+"
    r"stage_key=(?P<stage>\S+)\s+target_ms=(?P<target>[\d.]+)\s+"
    r"p95=(?P<p95>[\d.]+)\s+p99=(?P<p99>[\d.]+).*?"
    r"worst_strategy=(?P<strategy>\S+).*?worst_trader=(?P<trader>\w+)"
)
# Commit 2 will emit this line after every successful crypto place_order.
CRYPTO_TRACE_RE = re.compile(
    r"crypto_latency_trace\s+(?P<kv>.+)$"
)
# Fast-tier (workers/fast_trader_runtime.py) emits these whenever a
# fast-trader cycle or submit blows its budget.  Both carry
# ``stage_timings_ms`` so they're aggregator-friendly.  These are the
# actual measurements for the path the crypto traders run on (they
# have ``latency_class='fast'``); the original ``Trader cycle slow``
# regex captures the SLOW orchestrator only.
FAST_TRADER_CYCLE_BUDGET_RE = re.compile(
    r"Fast trader cycle exceeded hard budget\s+trader_id=(?P<tid>\w+)\s+"
    r"duration_s=(?P<dur>[\d.]+)\s+budget_s=(?P<budget>[\d.]+)\s+"
    r"stage_timings_ms=(?P<stages>\{.+\})\s*$"
)
FAST_TRADER_SUBMIT_BUDGET_RE = re.compile(
    r"Fast trader submit exceeded budget\s+trader_id=(?P<tid>\w+)\s+"
    r"signal_id=(?P<sid>\w+)\s+duration_s=(?P<dur>[\d.]+)\s+"
    r"budget_s=(?P<budget>[\d.]+)"
)


def _safe_eval_dict(text: str) -> dict[str, Any]:
    """Best-effort parse of a Python-repr dict literal from a log line.

    Same approach perf_harness uses — never raise on a malformed line.
    """
    try:
        result = ast.literal_eval(text)
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


def _parse_kv_blob(text: str) -> dict[str, Any]:
    """Parse ``key=value key2=value2`` log-trailer pairs into a dict.

    Handles values with commas inside braces (a dict-shaped value).  Values
    are returned as strings unless they look like a Python literal, in
    which case ``ast.literal_eval`` is attempted.
    """
    out: dict[str, Any] = {}
    i = 0
    n = len(text)
    while i < n:
        # Skip whitespace.
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        # Read key.
        key_start = i
        while i < n and text[i] not in "= \t":
            i += 1
        if i >= n or text[i] != "=":
            break
        key = text[key_start:i]
        i += 1  # skip "="
        # Read value: either {..} with brace matching, or a non-space token.
        if i < n and text[i] == "{":
            depth = 0
            val_start = i
            while i < n:
                ch = text[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
            value = text[val_start:i]
        else:
            val_start = i
            while i < n and not text[i].isspace():
                i += 1
            value = text[val_start:i]
        # Best-effort literal eval.
        try:
            out[key] = ast.literal_eval(value)
        except Exception:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


@dataclass
class StageSamples:
    """Per-stage sample bag with bounded memory."""

    name: str
    samples: list[float] = field(default_factory=list)

    def add(self, ms: float) -> None:
        if ms is None:
            return
        try:
            v = float(ms)
        except (TypeError, ValueError):
            return
        if v < 0:
            return
        self.samples.append(v)
        # Bounded — keep last 5k samples per stage to bound memory across
        # multi-hour captures.
        if len(self.samples) > 5000:
            del self.samples[: len(self.samples) - 5000]

    def percentiles(self) -> dict[str, float]:
        if not self.samples:
            return {"count": 0}
        s = sorted(self.samples)
        n = len(s)

        def pct(p: float) -> float:
            idx = max(0, min(n - 1, int(round(n * p)) - 1))
            return round(s[idx], 1)

        return {
            "count": n,
            "min": round(s[0], 1),
            "p50": round(median(s), 1),
            "p90": pct(0.90),
            "p95": pct(0.95),
            "p99": pct(0.99),
            "max": round(s[-1], 1),
        }


class CryptoLatencyAggregator:
    """Collects per-stage latency samples from log lines.

    Three sources merge into one report:

    1. ``Trader cycle slow`` — orchestrator-side stages, including
       ``ps_submit_order`` (orchestrator → place_order entry) and
       ``submit_leg_execute_live_order`` (place_order full duration as
       seen from the orchestrator).

    2. ``place_order slow`` — place_order's own per-stage breakdown
       (validate_reserve, io_lock_wait, create_order, post_order,
       persist_orders, persist_runtime_state_inner).  This is the
       authoritative source for the wire-side stages.

    3. ``Execution latency SLA breached lane=crypto`` — release-to-submit
       p95/p99 snapshots.  These are summary metrics, not raw samples,
       but they capture the lane-level latency picture between
       slow-event firings.

    4. (Commit 2) ``crypto_latency_trace`` — canonical per-trade trace
       with full wire→ack coverage.  When present, takes precedence over
       the slow-event derivations above.
    """

    def __init__(self) -> None:
        self.start_wall = datetime.now(timezone.utc)
        self.start_mono = time.monotonic()
        self.lines_seen = 0
        self.matched_trader_cycle_slow = 0
        self.matched_place_order_slow = 0
        self.matched_sla_breach = 0
        self.matched_crypto_trace = 0
        self.matched_fast_cycle_budget = 0
        self.matched_fast_submit_budget = 0

        self.stages: dict[str, StageSamples] = defaultdict(
            lambda: StageSamples(name="<unset>")
        )
        # Recent worst-N samples for forensic detail.
        self.worst_traces: deque = deque(maxlen=20)
        self.worst_place_orders: deque = deque(maxlen=20)
        self.sla_breaches: deque = deque(maxlen=50)
        self.recent_lines: deque = deque(maxlen=200)

    def _stage(self, name: str) -> StageSamples:
        bag = self.stages.get(name)
        if bag is None:
            bag = StageSamples(name=name)
            self.stages[name] = bag
        return bag

    def feed_line(self, line: str) -> None:
        self.lines_seen += 1
        line = line.rstrip()
        if not line:
            return
        m = LOG_PREFIX_RE.match(line)
        body = m.group("rest") if m else line
        m2 = LOG_LEVEL_RE.match(body)
        if not m2:
            return
        body = m2.group("body")

        # 1. Trader cycle slow — per-cycle stage_timings_ms.
        m = TRADER_CYCLE_SLOW_RE.search(body)
        if m and m.group("mode") == "live":
            self.matched_trader_cycle_slow += 1
            stages = _safe_eval_dict(m.group("stages"))
            if stages:
                # Keep a compact record for forensic dump.
                self.worst_traces.append(
                    {
                        "trader": m.group("tid"),
                        "duration_s": float(m.group("dur")),
                        "stages": stages,
                    }
                )
                # Map cycle stages into the aggregate bag.  Names are
                # preserved verbatim so the report shows the exact log
                # field names.
                for stage_name, ms in stages.items():
                    self._stage(f"cycle.{stage_name}").add(ms)
            return

        # 2. place_order slow — wire-side per-stage breakdown.
        m = PLACE_ORDER_SLOW_RE.search(body)
        if m:
            self.matched_place_order_slow += 1
            bd = _safe_eval_dict(m.group("bd"))
            if bd:
                self.worst_place_orders.append(
                    {
                        "token_id": m.group("tok"),
                        "side": m.group("side"),
                        "total_ms": float(m.group("total")),
                        "status": m.group("status"),
                        "breakdown": bd,
                    }
                )
                for stage_name, ms in bd.items():
                    self._stage(f"place_order.{stage_name}").add(ms)
            return

        # 3. SLA breach — lane-level p95/p99 snapshots.
        m = SLA_BREACH_RE.search(body)
        if m and m.group("lane") == "crypto":
            self.matched_sla_breach += 1
            self.sla_breaches.append(
                {
                    "stage": m.group("stage"),
                    "target_ms": float(m.group("target")),
                    "p95": float(m.group("p95")),
                    "p99": float(m.group("p99")),
                    "worst_strategy": m.group("strategy"),
                    "worst_trader": m.group("trader"),
                }
            )
            return

        # 4. crypto_latency_trace (Commit 2) — canonical per-trade trace.
        m = CRYPTO_TRACE_RE.search(body)
        if m:
            self.matched_crypto_trace += 1
            kv = _parse_kv_blob(m.group("kv"))
            breakdown = kv.get("breakdown_ms")
            if isinstance(breakdown, dict):
                for stage_name, ms in breakdown.items():
                    self._stage(f"trace.{stage_name}").add(ms)
                # Keep a copy for forensic detail.
                self.recent_lines.append({"kind": "crypto_trace", "trace": kv})
            return

        # 5. Fast-tier (workers/fast_trader_runtime.py) cycle budget
        # exceeded — carries the canonical fast-cycle stage_timings_ms.
        # The crypto traders run on this path (latency_class='fast'),
        # so this is where their actual cycle latency surfaces.
        m = FAST_TRADER_CYCLE_BUDGET_RE.search(body)
        if m:
            self.matched_fast_cycle_budget += 1
            stages = _safe_eval_dict(m.group("stages"))
            if stages:
                self.worst_traces.append(
                    {
                        "kind": "fast_cycle",
                        "trader": m.group("tid"),
                        "duration_s": float(m.group("dur")),
                        "budget_s": float(m.group("budget")),
                        "stages": stages,
                    }
                )
                for stage_name, ms in stages.items():
                    self._stage(f"fast_cycle.{stage_name}").add(ms)
            return

        # 6. Fast-tier submit budget exceeded — bare duration_s, no
        # stage_timings_ms (the per-trade detail comes from the
        # crypto_latency_trace line that fires inside place_order).
        m = FAST_TRADER_SUBMIT_BUDGET_RE.search(body)
        if m:
            self.matched_fast_submit_budget += 1
            self._stage("fast_submit.duration_ms").add(
                float(m.group("dur")) * 1000.0
            )
            return

    def report(self) -> dict[str, Any]:
        elapsed_s = round(time.monotonic() - self.start_mono, 1)
        stage_summary = {
            name: bag.percentiles()
            for name, bag in sorted(self.stages.items())
        }
        return {
            "harness": {
                "started_at_utc": self.start_wall.isoformat(),
                "duration_seconds": elapsed_s,
                "lines_seen": self.lines_seen,
                "matched": {
                    "trader_cycle_slow_live": self.matched_trader_cycle_slow,
                    "place_order_slow": self.matched_place_order_slow,
                    "sla_breach_crypto": self.matched_sla_breach,
                    "crypto_latency_trace": self.matched_crypto_trace,
                    "fast_cycle_budget": self.matched_fast_cycle_budget,
                    "fast_submit_budget": self.matched_fast_submit_budget,
                },
            },
            "stages": stage_summary,
            "headline": self._headline(),
            "worst_traces_recent": list(self.worst_traces)[-10:],
            "worst_place_orders_recent": list(self.worst_place_orders)[-10:],
            "sla_breaches_recent": list(self.sla_breaches)[-10:],
        }

    def _headline(self) -> dict[str, Any]:
        """Pick the most-actionable single number per layer."""
        end_to_end_p90 = None
        end_to_end_p99 = None
        # If Commit 2's traces are present, prefer the canonical end_to_end.
        bag = self.stages.get("trace.wire_to_ack_ms") or self.stages.get(
            "trace.total_ms"
        )
        if bag and bag.samples:
            pcts = bag.percentiles()
            end_to_end_p90 = pcts.get("p90")
            end_to_end_p99 = pcts.get("p99")

        place_order_p90 = None
        bag = self.stages.get("place_order.total_ms")
        if bag and bag.samples:
            place_order_p90 = bag.percentiles().get("p90")

        post_order_p90 = None
        bag = self.stages.get("place_order.post_order")
        if bag and bag.samples:
            post_order_p90 = bag.percentiles().get("p90")

        return {
            "end_to_end_p90_ms": end_to_end_p90,
            "end_to_end_p99_ms": end_to_end_p99,
            "place_order_total_p90_ms": place_order_p90,
            "post_order_p90_ms": post_order_p90,
        }


# ---------------------------------------------------------------------------
# IO loops
# ---------------------------------------------------------------------------


def _normalize_for_aggregator(plane: str, line: str) -> str:
    """Convert a raw worker JSON log line to the bracketed text shape the
    parser expects.  Plain bracketed lines pass through.
    """
    line = line.rstrip("\r\n")
    if not line:
        return line
    if line.startswith("["):
        return line
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
                    ts_short = (
                        ts.split("T")[1].split(".")[0] if "T" in ts else ts[-8:]
                    )
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
            return " ".join(parts)
    # Bare line — re-prefix with plane so the aggregator's plane bucket
    # works.
    return f"[{plane}] {line}"


async def _tail_file(
    file_path: Path,
    plane: str,
    agg: CryptoLatencyAggregator,
    stop_event: asyncio.Event,
) -> None:
    """Tail a JSON log file from EOF until ``stop_event`` is set."""
    fh = None
    while not stop_event.is_set():
        try:
            if fh is None:
                if not file_path.exists():
                    await asyncio.sleep(0.5)
                    continue
                fh = file_path.open("r", encoding="utf-8", errors="replace")
                fh.seek(0, 2)  # EOF — only NEW lines.
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
            agg.feed_line(_normalize_for_aggregator(plane, line))
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(0.5)
    if fh is not None:
        try:
            fh.close()
        except Exception:
            pass


def _read_static_file(path: Path, agg: CryptoLatencyAggregator) -> None:
    """Read every line of a static log file into the aggregator."""
    if not path.exists():
        print(f"[harness] file not found: {path}", file=sys.stderr)
        return
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            # Static files are already bracketed (live.log shape) — pass
            # through directly.  If they're JSON-shaped, the normalizer
            # handles it.
            stripped = line.lstrip()
            if stripped.startswith("["):
                agg.feed_line(line)
            else:
                agg.feed_line(_normalize_for_aggregator("WORKERS", line))


def _read_stdin(agg: CryptoLatencyAggregator) -> None:
    for line in sys.stdin:
        stripped = line.lstrip()
        if stripped.startswith("["):
            agg.feed_line(line)
        else:
            agg.feed_line(_normalize_for_aggregator("WORKERS", line))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _format_pcts(pcts: dict[str, float]) -> str:
    if not pcts.get("count"):
        return "(no samples)"
    return (
        f"n={pcts['count']:>4} "
        f"min={pcts['min']:>7.1f} "
        f"p50={pcts['p50']:>7.1f} "
        f"p90={pcts['p90']:>7.1f} "
        f"p95={pcts['p95']:>7.1f} "
        f"p99={pcts['p99']:>7.1f} "
        f"max={pcts['max']:>7.1f}"
    )


def _print_summary(agg: CryptoLatencyAggregator) -> None:
    report = agg.report()
    h = report["harness"]
    head = report["headline"]
    print(
        f"\n=== crypto latency harness summary "
        f"({h['duration_seconds']:.0f}s, {h['lines_seen']} lines) ==="
    )
    print(
        f"matches: trader_cycle_slow_live={h['matched']['trader_cycle_slow_live']} "
        f"place_order_slow={h['matched']['place_order_slow']} "
        f"sla_breach_crypto={h['matched']['sla_breach_crypto']} "
        f"crypto_latency_trace={h['matched']['crypto_latency_trace']} "
        f"fast_cycle_budget={h['matched'].get('fast_cycle_budget', 0)} "
        f"fast_submit_budget={h['matched'].get('fast_submit_budget', 0)}"
    )
    print(
        f"headline: end_to_end p90={head['end_to_end_p90_ms']}ms "
        f"p99={head['end_to_end_p99_ms']}ms; "
        f"place_order_total p90={head['place_order_total_p90_ms']}ms; "
        f"post_order p90={head['post_order_p90_ms']}ms"
    )
    print("\nper-stage breakdown:")
    for name, pcts in sorted(report["stages"].items()):
        print(f"  {name:55s}  {_format_pcts(pcts)}")
    if report["sla_breaches_recent"]:
        print("\nrecent SLA breaches (lane=crypto):")
        for b in report["sla_breaches_recent"][-5:]:
            print(
                f"  stage={b['stage']:36s} target={b['target_ms']:>5.0f}ms "
                f"p95={b['p95']:>7.0f}ms p99={b['p99']:>7.0f}ms "
                f"strategy={b['worst_strategy']}"
            )


def _write_report(agg: CryptoLatencyAggregator, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(agg.report(), indent=2), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _run_tail(
    duration: int,
    log_files: Iterable[tuple[str, Path]],
    report_path: Path,
    write_interval: int,
) -> CryptoLatencyAggregator:
    agg = CryptoLatencyAggregator()
    stop_event = asyncio.Event()
    print(
        f"[harness] tail mode — files={[str(p) for _, p in log_files]} "
        f"duration={duration}s report={report_path}"
    )
    readers = [
        asyncio.create_task(
            _tail_file(p, plane, agg, stop_event),
            name=f"crypto-harness-tail-{plane}",
        )
        for plane, p in log_files
    ]

    async def _periodic_writer() -> None:
        try:
            while not stop_event.is_set():
                await asyncio.sleep(write_interval)
                _write_report(agg, report_path)
        except asyncio.CancelledError:
            return

    writer_task = asyncio.create_task(_periodic_writer(), name="crypto-harness-writer")
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
    _write_report(agg, report_path)
    return agg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration",
        type=int,
        default=600,
        help="Capture window in seconds when tailing (default 600).",
    )
    parser.add_argument(
        "--tail",
        action="store_true",
        help="Tail the per-plane debug log files written by the launcher's "
        "-Debug mode.",
    )
    parser.add_argument(
        "--from-file",
        type=Path,
        default=None,
        help="Read a single static log file (e.g. tools/.harness/live.log) "
        "instead of tailing.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Where to write the JSON report.",
    )
    parser.add_argument(
        "--write-interval",
        type=int,
        default=30,
        help="Seconds between mid-run report checkpoints (tail mode only).",
    )
    parser.add_argument(
        "--planes",
        type=str,
        default="WORKERS,DISCOVERY,NEWS",
        help="Comma-separated planes to tail (default all three).",
    )
    args = parser.parse_args()

    if args.tail:
        wanted = {p.strip().upper() for p in args.planes.split(",") if p.strip()}
        files = [
            (plane, path)
            for plane, path in DEFAULT_LOG_FILES
            if plane in wanted
        ]
        if not files:
            print("[harness] no plane log files matched", file=sys.stderr)
            return 2
        agg = asyncio.run(
            _run_tail(args.duration, files, args.report, args.write_interval)
        )
    elif args.from_file:
        agg = CryptoLatencyAggregator()
        _read_static_file(args.from_file, agg)
        _write_report(agg, args.report)
    else:
        agg = CryptoLatencyAggregator()
        _read_stdin(agg)
        _write_report(agg, args.report)

    _print_summary(agg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
