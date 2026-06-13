"""Backtest job-queue runner.

Companion to ``services/strategy_backtester.py`` and the unified
runner.  Where ``run_unified_backtest`` is the synchronous entrypoint
(returns the full result blob), this module is the *async-by-default*
path: an operator enqueues a job, the dedicated worker process picks
it up, and the engine runs entirely off the API event loop with full
GIL + crash isolation.

Design parallels ``provider_import_service``:

  * ``enqueue_run(payload)`` — write a ``BacktestRun`` row with
    ``status='queued'``.  Returns the row immediately; no engine work
    happens yet.

  * ``claim_next_queued_run()`` — atomically claim the oldest queued
    row via ``UPDATE ... FOR UPDATE SKIP LOCKED``.  Returns the
    claimed row to the worker.

  * ``run_job(run_id)`` — execute the run.  Reconstructs the call
    args from ``payload_json``, runs the engine with a progress
    callback that updates the row, writes the final result on
    completion (or error message on crash, or ``cancelled`` if the
    operator hit stop).

  * ``request_cancel(run_id)`` — flip ``cancel_requested=True``.  The
    progress callback checks this on every yield and bails out.

The progress callback writes back at most once per second (debounced)
so the worker doesn't spam the DB during a fast replay.
"""
from __future__ import annotations

import logging
import socket
import time
import traceback
import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import select, update

from models.database import BacktestAsyncSessionLocal, BacktestRun
from utils.utcnow import utcnow

logger = logging.getLogger(__name__)


# How often the progress callback writes back to the DB.  The engine
# fires the callback every ``progress_every`` snapshots (~1k); we
# debounce DB writes so we don't make a 1ms-per-snapshot operation
# DB-bound.
_PROGRESS_DB_DEBOUNCE_SECONDS = 1.0

# Worker identity surfaced on the BacktestRun row for diagnostics.
# host:pid keeps it stable across restarts AND distinguishable when
# we eventually run multiple workers in parallel.
_WORKER_ID = f"{socket.gethostname()}:{__import__('os').getpid()}"


# ── Operator-facing payload shape ─────────────────────────────────


_PAYLOAD_KEYS = (
    "source_code",
    "slug",
    "config",
    "token_ids",
    "start",
    "end",
    "session_id",
    "provider_dataset_ids",
    "initial_capital_usd",
    "submit_p50_ms",
    "submit_p95_ms",
    "cancel_p50_ms",
    "cancel_p95_ms",
    "seed",
    "counterfactual_sample_size",
    "ensemble_sample_size",
    "fills_sample_size",
    "impact_strength_bps",
    "maker_rebate_bps",
    "maker_rebate_max_spread_bps",
    "n_trials",
    "discovery_sample_interval_seconds",
    "discovery_max_ticks",
)


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Filter to known keys + JSON-coerce datetimes so the row is
    self-contained (the worker doesn't need any state outside the row).
    """
    out: dict[str, Any] = {}
    for k in _PAYLOAD_KEYS:
        if k in payload:
            v = payload[k]
            if isinstance(v, datetime):
                out[k] = v.isoformat()
            else:
                out[k] = v
    return out


# ── Enqueue ───────────────────────────────────────────────────────


async def enqueue_run(payload: dict[str, Any]) -> BacktestRun:
    """Persist a queued run and return its row (with the allocated id).

    The id is a 16-hex token — same shape the legacy sync path uses
    via ``unified_runner._store_run``.  This means the operator gets a
    canonical run_id immediately and can poll for its status.
    """
    run_id = uuid.uuid4().hex[:16]
    now = utcnow()
    slug = (payload.get("slug") or "").strip() or None

    async with BacktestAsyncSessionLocal() as session:
        row = BacktestRun(
            id=run_id,
            strategy_slug=slug,
            strategy_name=None,  # set by the worker on first progress
            started_at=now,  # nominal "submitted at" until worker claims
            status="queued",
            payload_json=_normalize_payload(payload),
            progress=0.0,
            message="Queued; waiting for backtest worker",
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return row


# ── Claim ─────────────────────────────────────────────────────────


async def claim_next_queued_run() -> Optional[BacktestRun]:
    """Atomically claim the oldest ``queued`` row (FIFO).

    Uses Postgres' ``UPDATE ... WHERE id IN (SELECT ... FOR UPDATE
    SKIP LOCKED)`` so multiple workers can poll the queue safely
    without serializing.  Falls back to a non-locking SELECT+UPDATE
    on SQLite (tests).

    Returns the claimed row, or None when the queue is empty.
    """
    async with BacktestAsyncSessionLocal() as session:
        try:
            stmt = (
                update(BacktestRun)
                .where(
                    BacktestRun.id.in_(
                        select(BacktestRun.id)
                        .where(BacktestRun.status == "queued")
                        .order_by(BacktestRun.created_at.asc())
                        .limit(1)
                        .with_for_update(skip_locked=True)
                    )
                )
                .values(
                    status="running",
                    claimed_at=utcnow(),
                    worker_id=_WORKER_ID,
                    message="Worker claimed run",
                )
                .returning(BacktestRun)
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            await session.commit()
            return row
        except Exception:
            await session.rollback()

        # SQLite fallback.
        candidate = (
            await session.execute(
                select(BacktestRun)
                .where(BacktestRun.status == "queued")
                .order_by(BacktestRun.created_at.asc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if candidate is None:
            return None
        candidate.status = "running"
        candidate.claimed_at = utcnow()
        candidate.worker_id = _WORKER_ID
        candidate.message = "Worker claimed run"
        await session.commit()
        await session.refresh(candidate)
        return candidate


# ── Cancel ────────────────────────────────────────────────────────


async def request_cancel(run_id: str) -> bool:
    """Flip ``cancel_requested=True`` on a queued or running row.

    The worker's progress callback checks this flag on every yield
    and raises ``BacktestCancelled`` to short-circuit the run cleanly.
    Returns False if the row doesn't exist or has already finished.
    """
    async with BacktestAsyncSessionLocal() as session:
        row = (
            await session.execute(
                select(BacktestRun).where(BacktestRun.id == run_id)
            )
        ).scalar_one_or_none()
        if row is None:
            return False
        if row.status in ("completed", "failed", "cancelled", "ok"):
            return False
        row.cancel_requested = True
        # If still queued, complete the cancel right here — no worker
        # will ever pick this up.
        if row.status == "queued":
            row.status = "cancelled"
            row.completed_at = utcnow()
            row.message = "Cancelled before worker pickup"
        await session.commit()
        return True


class BacktestCancelled(Exception):
    """Raised by the progress callback when ``cancel_requested`` is
    set so the engine bails out cleanly."""


# ── Crash recovery ────────────────────────────────────────────────


def _pid_alive(pid: int) -> bool:
    import sys as _sys

    if _sys.platform == "win32":
        # os.kill(pid, 0) TERMINATES on Windows — probe via OpenProcess.
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            return bool(ok) and exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    import os as _os

    try:
        _os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


# A run that has been "running" longer than this is orphaned regardless
# of pid state (pid-reuse can make a dead worker look alive).  No
# legitimate single backtest run approaches this — the engine bounds
# windows and tick counts well below it.
_ORPHAN_MAX_RUNNING_SECONDS = 24 * 3600.0


async def requeue_orphaned_runs() -> int:
    """Requeue ``running`` rows whose claiming worker is provably dead.

    Worker ids are ``hostname:pid``.  Rows claimed on THIS host by a pid
    that no longer exists were orphaned by a worker crash / host restart
    and would otherwise sit ``running`` forever (the queue has no
    heartbeat).  Called by the worker on startup, before polling.

    Rows claimed on other hosts are only requeued past the 24h hard cap
    — pid liveness is unprovable across hosts.
    """
    hostname = socket.gethostname()
    my_pid = __import__("os").getpid()
    now = utcnow()
    requeued = 0
    async with BacktestAsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(BacktestRun).where(BacktestRun.status == "running")
            )
        ).scalars().all()
        for row in rows:
            worker = str(row.worker_id or "")
            host, _, pid_s = worker.rpartition(":")
            claimed_at = row.claimed_at
            if claimed_at is not None and claimed_at.tzinfo is None:
                from datetime import timezone as _tz

                claimed_at = claimed_at.replace(tzinfo=_tz.utc)
            age_s = (
                (now - claimed_at).total_seconds() if claimed_at is not None else None
            )
            same_host_dead = False
            if host == hostname and pid_s.isdigit():
                pid = int(pid_s)
                same_host_dead = pid != my_pid and not _pid_alive(pid)
            stale = age_s is not None and age_s > _ORPHAN_MAX_RUNNING_SECONDS
            if not (same_host_dead or stale):
                continue
            row.status = "queued"
            row.worker_id = None
            row.claimed_at = None
            row.progress = 0.0
            row.snapshots_processed = 0
            row.message = (
                "Requeued: claiming worker died"
                if same_host_dead
                else "Requeued: run exceeded 24h running cap"
            )
            requeued += 1
        if requeued:
            await session.commit()
    return requeued


# ── Run ───────────────────────────────────────────────────────────


async def run_job(run_id: str) -> dict[str, Any]:
    """Execute a previously-claimed run.  Idempotent within a worker
    lifecycle — calling twice on the same id is a no-op once status
    transitions out of ``running``.
    """
    # Reload the row fresh inside the worker so we have the canonical
    # payload + cancel state.
    async with BacktestAsyncSessionLocal() as session:
        row = (
            await session.execute(
                select(BacktestRun).where(BacktestRun.id == run_id)
            )
        ).scalar_one_or_none()
        if row is None:
            raise ValueError(f"BacktestRun '{run_id}' not found")
        if row.status != "running":
            logger.warning(
                "run_job called on row '%s' with status='%s'; skipping",
                run_id, row.status,
            )
            return {"run_id": run_id, "status": row.status, "skipped": True}
        payload = dict(row.payload_json or {})

    # Reconstruct datetimes from ISO strings.
    for k in ("start", "end"):
        v = payload.get(k)
        if isinstance(v, str) and v:
            try:
                payload[k] = datetime.fromisoformat(v)
            except ValueError:
                payload[k] = None

    # Progress writer (debounced).  Closure over run_id + last-write
    # time so multiple yields within a 1s window collapse to one DB
    # write.  Also re-reads ``cancel_requested`` so the worker can
    # honor an operator stop without an extra DB round-trip.
    state = {"last_write_ts": 0.0}

    async def _on_progress(processed: int, equity_usd: float, open_count: int) -> None:
        now = time.monotonic()
        if now - state["last_write_ts"] < _PROGRESS_DB_DEBOUNCE_SECONDS:
            return
        state["last_write_ts"] = now
        try:
            async with BacktestAsyncSessionLocal() as session:
                fresh = (
                    await session.execute(
                        select(BacktestRun).where(BacktestRun.id == run_id)
                    )
                ).scalar_one_or_none()
                if fresh is None:
                    return
                if fresh.cancel_requested:
                    raise BacktestCancelled()
                # Progress is a soft estimate; the engine doesn't know
                # the snapshot total upfront, so we use snapshots_total_
                # estimate when set, otherwise leave progress at 0 and
                # show snapshots_processed in the message.
                est = fresh.snapshots_total_estimate or 0
                if est > 0:
                    fresh.progress = min(1.0, processed / float(est))
                else:
                    fresh.progress = 0.0
                fresh.snapshots_processed = int(processed)
                if processed <= 0:
                    # Pre-engine phases (data load + discovery replay) probe
                    # this callback for cancellation before any snapshot is
                    # replayed — show the real phase instead of a fake $0 mark.
                    fresh.message = "Loading historical data · discovery replay…"
                else:
                    fresh.message = (
                        f"Replaying snapshots: {processed:,}"
                        + (f" / {est:,}" if est else "")
                        + f" · open={open_count} · equity=${equity_usd:,.2f}"
                    )
                await session.commit()
        except BacktestCancelled:
            raise
        except Exception as exc:
            logger.warning("Progress write failed for %s: %s", run_id, exc)

    # Run the unified pipeline.  The runner already returns the
    # augmented result blob (execution + fill_model + decomposition +
    # latency etc.); we just persist it on the row.
    from services.backtest.unified_runner import run_unified_backtest

    started_perf = time.perf_counter()
    error: Optional[str] = None
    error_tb: Optional[str] = None
    final_status = "completed"
    result_blob: dict[str, Any] = {}
    try:
        result_blob = await run_unified_backtest(
            **{
                k: v for k, v in payload.items()
                if k in (
                    "source_code", "slug", "config", "token_ids",
                    "start", "end", "session_id", "provider_dataset_ids",
                    "initial_capital_usd", "submit_p50_ms", "submit_p95_ms",
                    "cancel_p50_ms", "cancel_p95_ms", "seed",
                    "counterfactual_sample_size", "ensemble_sample_size",
                    "fills_sample_size",
                    "impact_strength_bps", "maker_rebate_bps",
                    "maker_rebate_max_spread_bps",
                    # n_trials drives the López de Prado deflated-Sharpe
                    # correction; the iteration loop passes the search
                    # size so over-fitting deflation actually applies.
                    "n_trials",
                    "discovery_sample_interval_seconds", "discovery_max_ticks",
                )
            },
            progress_callback=_on_progress,
            run_id=run_id,
        )
    except BacktestCancelled:
        final_status = "cancelled"
        error = "Cancelled by operator"
    except Exception as exc:
        final_status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        error_tb = traceback.format_exc()
        logger.exception("Backtest run %s crashed", run_id)

    elapsed_ms = (time.perf_counter() - started_perf) * 1000.0

    # Final write — replace the in-progress row with the canonical
    # result.  Use the unified result's run_id only as a fallback;
    # we already allocated our own.
    exec_block = (result_blob.get("execution") if isinstance(result_blob, dict) else {}) or {}
    async with BacktestAsyncSessionLocal() as session:
        row = (
            await session.execute(
                select(BacktestRun).where(BacktestRun.id == run_id)
            )
        ).scalar_one_or_none()
        if row is None:
            return {"run_id": run_id, "status": final_status}
        row.status = final_status
        row.completed_at = utcnow()
        row.total_time_ms = elapsed_ms
        row.progress = 1.0 if final_status == "completed" else row.progress
        row.error = error
        if final_status == "completed":
            row.result_json = result_blob
            row.strategy_slug = exec_block.get("strategy_slug") or row.strategy_slug
            row.strategy_name = exec_block.get("strategy_name") or row.strategy_name
            row.trade_count = int(exec_block.get("trade_count") or 0)
            row.total_return_pct = float(exec_block.get("total_return_pct") or 0.0)
            row.message = (
                f"Done · trades={row.trade_count} · "
                f"return={row.total_return_pct:.2f}% · "
                f"{elapsed_ms / 1000.0:.1f}s"
            )
        elif final_status == "cancelled":
            row.message = "Cancelled by operator"
        else:
            row.message = f"Failed: {error[:200] if error else 'unknown'}"
            if error_tb:
                # Stash traceback on result_json for the UI's error pane.
                row.result_json = {"runtime_traceback": error_tb}
        await session.commit()

    return {
        "run_id": run_id,
        "status": final_status,
        "elapsed_ms": elapsed_ms,
        "trade_count": exec_block.get("trade_count"),
        "total_return_pct": exec_block.get("total_return_pct"),
        "error": error,
    }


__all__ = [
    "enqueue_run",
    "claim_next_queued_run",
    "request_cancel",
    "requeue_orphaned_runs",
    "run_job",
    "BacktestCancelled",
]
