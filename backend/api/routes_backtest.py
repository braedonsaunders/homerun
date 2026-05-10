"""Unified backtest API.

Single entry point for the new BacktestStudio UI.  Wraps the
unified_runner, exposes recent-runs history (in-memory LRU, 32 deep)
and per-run retrieval.

Old code-backtest endpoints in routes_validation.py are unchanged
for back-compat.  The new UI uses these.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from services.backtest.unified_runner import (
    get_recent_run,
    list_recent_runs,
    run_unified_backtest,
)


logger = logging.getLogger("routes.backtest")
router = APIRouter(prefix="/backtest", tags=["Backtest"])


def _sanitize_floats(value: Any) -> Any:
    """Recursively replace non-finite floats (inf/-inf/NaN) with None.

    FastAPI's default encoder rejects these and 500s the entire
    response.  metrics.py uses _NO_DENOM_SENTINEL for legitimate
    "no denominator" cases; this catches anything that slips through
    (a future bug in a metric, a third-party lib that emits NaN, etc.).
    Belt-and-suspenders so a single misbehaving field can't take down
    the whole backtest result.
    """
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _sanitize_floats(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_floats(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_floats(v) for v in value)
    return value


class UnifiedBacktestRequest(BaseModel):
    source_code: str = Field(..., min_length=10)
    slug: str = Field(default="_backtest_unified", min_length=1)
    config: dict[str, Any] | None = None
    token_ids: list[str] | None = None
    start: datetime | None = None
    end: datetime | None = None
    # When set, the backtester scopes its replay to the recording
    # session's target tokens × time window — overrides token_ids /
    # start / end if provided here OR in the request.
    session_id: str | None = None
    # Imported provider datasets (polybacktest, etc.).  Resolves to the
    # union of token_ids and the [min(start), max(end)] window across
    # the selected datasets.  Mutually exclusive with session_id; when
    # both are present, session_id wins (recording sessions are the
    # higher-fidelity local capture).
    provider_dataset_ids: list[str] | None = None
    initial_capital_usd: float = Field(default=1000.0, gt=0.0)
    submit_p50_ms: float | None = Field(default=None, ge=0.0)
    submit_p95_ms: float | None = Field(default=None, ge=0.0)
    cancel_p50_ms: float | None = Field(default=None, ge=0.0)
    cancel_p95_ms: float | None = Field(default=None, ge=0.0)
    seed: int | None = None
    counterfactual_sample_size: int = Field(default=8, ge=0, le=64)
    ensemble_sample_size: int = Field(default=8, ge=0, le=64)
    # Phase 12f/g operator overrides — default None means "use the
    # built-in defaults" (impact disabled, no maker rebate).
    impact_strength_bps: float | None = Field(default=None, ge=0.0, le=500.0)
    maker_rebate_bps: float | None = Field(default=None, ge=0.0, le=20.0)
    maker_rebate_max_spread_bps: float | None = Field(default=None, ge=0.0, le=500.0)


@router.post("/run")
async def run_backtest(req: UnifiedBacktestRequest):
    """Run the unified pipeline and return the augmented result.

    The result includes the execution-realistic backtest plus the
    Cox PH fill model snapshot, empirical constants, latency
    distribution, trade-vs-cancel decomposition, sample
    counterfactual replays, and sample ensemble bands.  All fields
    are JSON-safe; the UI consumes the response directly.
    """
    # Resolve session_id (if any) into concrete token_ids + window.
    # Session scoping wins over per-request token_ids / start / end so
    # the operator can hit the same backtest button against a session
    # without re-typing window bounds.
    token_ids = req.token_ids
    start = req.start
    end = req.end
    session_meta: dict[str, Any] | None = None
    provider_dataset_meta: dict[str, Any] | None = None
    if req.session_id:
        from services.recording_session_service import session_backtest_scope

        scope = await session_backtest_scope(req.session_id)
        if scope is None:
            raise HTTPException(
                status_code=404,
                detail=f"Recording session '{req.session_id}' not found or has no captured data",
            )
        token_ids = scope["token_ids"]
        start = scope["start"]
        end = scope["end"]
        session_meta = {"session_id": scope["session_id"], "session_name": scope["session_name"]}
    elif req.provider_dataset_ids:
        from services.external_data.provider_import_service import resolve_dataset_scope

        scope = await resolve_dataset_scope(list(req.provider_dataset_ids))
        if scope is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "None of the requested provider_dataset_ids resolved to "
                    "an imported dataset. Import them via Data Lab → Providers first."
                ),
            )
        token_ids = scope["token_ids"]
        start = scope["start"]
        end = scope["end"]
        provider_dataset_meta = {
            "dataset_ids": scope["dataset_ids"],
            "labels": scope["labels"],
            "token_ids": scope["token_ids"],
        }

    try:
        result = await run_unified_backtest(
            source_code=req.source_code,
            slug=req.slug,
            config=req.config,
            token_ids=token_ids,
            start=start,
            end=end,
            initial_capital_usd=req.initial_capital_usd,
            submit_p50_ms=req.submit_p50_ms,
            submit_p95_ms=req.submit_p95_ms,
            cancel_p50_ms=req.cancel_p50_ms,
            cancel_p95_ms=req.cancel_p95_ms,
            seed=req.seed,
            counterfactual_sample_size=req.counterfactual_sample_size,
            ensemble_sample_size=req.ensemble_sample_size,
            impact_strength_bps=req.impact_strength_bps,
            maker_rebate_bps=req.maker_rebate_bps,
            maker_rebate_max_spread_bps=req.maker_rebate_max_spread_bps,
        )
    except Exception as exc:
        logger.exception("Unified backtest failed")
        raise HTTPException(status_code=500, detail=f"backtest failed: {exc}") from exc

    if isinstance(result, dict):
        if session_meta is not None:
            result["recording_session"] = session_meta
        if provider_dataset_meta is not None:
            result["provider_datasets"] = provider_dataset_meta
    return _sanitize_floats(result)


@router.get("/runs")
async def list_runs(limit: int = 32) -> dict[str, list[dict[str, Any]]]:
    runs = await list_recent_runs(limit=int(max(1, min(200, limit))))
    return {"runs": _sanitize_floats(runs)}


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    run = await get_recent_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return _sanitize_floats(run)


# ── Async-job-queue endpoints ───────────────────────────────────────────
#
# The default UX flow goes through these now:
#
#   POST /backtest/runs/enqueue        → 202 Accepted with run_id
#   GET  /backtest/runs/{run_id}/status → poll for progress (1s cadence)
#   POST /backtest/runs/{run_id}/cancel → operator stop
#
# Engine work runs in the dedicated backtest worker process (discovery
# plane, see workers/backtest_worker.py).  Full GIL + crash isolation
# from the API + orchestrator.
#
# The legacy ``POST /backtest/run`` (sync) path stays for backwards
# compatibility but should be considered deprecated — running a 1M+
# snapshot replay on the API process makes the entire backend
# unresponsive for the run's duration.


@router.post("/runs/enqueue", status_code=202)
async def enqueue_run_route(req: UnifiedBacktestRequest) -> dict[str, Any]:
    """Enqueue a backtest run for the dedicated worker.

    Returns 202 Accepted with the allocated ``run_id`` immediately.
    The operator polls ``/runs/{run_id}/status`` for progress.

    Session/provider-dataset resolution happens at enqueue time so
    the worker doesn't need to re-resolve.
    """
    from services.backtest.job_runner import enqueue_run

    payload: dict[str, Any] = {
        "source_code": req.source_code,
        "slug": req.slug,
        "config": req.config,
        "token_ids": req.token_ids,
        "start": req.start,
        "end": req.end,
        "session_id": req.session_id,
        "provider_dataset_ids": req.provider_dataset_ids,
        "initial_capital_usd": req.initial_capital_usd,
        "submit_p50_ms": req.submit_p50_ms,
        "submit_p95_ms": req.submit_p95_ms,
        "cancel_p50_ms": req.cancel_p50_ms,
        "cancel_p95_ms": req.cancel_p95_ms,
        "seed": req.seed,
        "counterfactual_sample_size": req.counterfactual_sample_size,
        "ensemble_sample_size": req.ensemble_sample_size,
        "impact_strength_bps": req.impact_strength_bps,
        "maker_rebate_bps": req.maker_rebate_bps,
        "maker_rebate_max_spread_bps": req.maker_rebate_max_spread_bps,
    }
    row = await enqueue_run(payload)
    return {
        "run_id": row.id,
        "status": row.status,
        "message": row.message,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


@router.get("/runs/{run_id}/status")
async def get_run_status(run_id: str) -> dict[str, Any]:
    """Lightweight status for polling.  Distinct from /runs/{id} which
    returns the full result blob (heavy)."""
    from sqlalchemy import select as sa_select
    from models.database import AsyncSessionLocal as _Sess, BacktestRun

    async with _Sess() as session:
        row = (
            await session.execute(
                sa_select(BacktestRun).where(BacktestRun.id == run_id)
            )
        ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return {
        "run_id": row.id,
        "status": row.status,
        "progress": float(row.progress or 0.0),
        "message": row.message,
        "snapshots_processed": int(row.snapshots_processed or 0),
        "snapshots_total_estimate": (
            int(row.snapshots_total_estimate)
            if row.snapshots_total_estimate is not None
            else None
        ),
        "trade_count": int(row.trade_count or 0),
        "total_return_pct": float(row.total_return_pct or 0.0),
        "error": row.error,
        "claimed_at": row.claimed_at.isoformat() if row.claimed_at else None,
        "completed_at": row.completed_at.isoformat() if row.completed_at else None,
        "worker_id": row.worker_id,
        "cancel_requested": bool(row.cancel_requested),
    }


@router.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> dict[str, Any]:
    """Request cancel.  The worker honors this on the next progress
    yield (within ~1s) and writes ``status='cancelled'`` to the row.
    Returns 404 if the run doesn't exist or is already terminal.
    """
    from services.backtest.job_runner import request_cancel

    ok = await request_cancel(run_id)
    if not ok:
        raise HTTPException(
            status_code=404,
            detail=f"Run '{run_id}' not found or already finished",
        )
    return {"run_id": run_id, "cancel_requested": True}


@router.get("/runs/{run_id}/report.pdf")
async def get_run_pdf(run_id: str) -> Response:
    """Render an executive PDF for a completed backtest run.

    Mirrors the reverse-engineer ``/jobs/{id}/report.pdf`` route —
    same WeasyPrint pipeline, same Jinja env, separate template
    (``backtest_run_report.html.j2``).  Returns 404 if the run
    doesn't exist, 503 if WeasyPrint can't be loaded (with a
    platform-specific install hint).
    """
    from sqlalchemy import select
    from models.database import AsyncSessionLocal, BacktestRun

    async with AsyncSessionLocal() as session:
        row = (
            await session.execute(select(BacktestRun).where(BacktestRun.id == run_id))
        ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    try:
        from services.reports.wallet_strategy_report import (
            ReportRenderError,
            render_backtest_run_report,
        )
        result_blob = row.result_json or {}
        pdf_bytes = render_backtest_run_report(run_row=row, result=result_blob)
    except ReportRenderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("PDF report render failed for backtest run %s", run_id)
        raise HTTPException(status_code=500, detail=f"PDF render failed: {exc}") from exc

    short = run_id[:8]
    slug = (row.strategy_slug or "backtest").replace(" ", "_")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="backtest_{slug}_{short}.pdf"',
        },
    )


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str) -> dict[str, Any]:
    """Delete a single backtest run row.

    Refuses to delete runs that are still actively executing
    (status ∈ {queued, running}) — the operator should cancel
    those first via POST /runs/{id}/cancel; once they reach a
    terminal state (completed / failed / cancelled / ok) the row
    is safe to remove.

    Returns 404 if the run doesn't exist, 409 if it's still alive.
    """
    from sqlalchemy import delete as sql_delete, select
    from models.database import AsyncSessionLocal, BacktestRun

    async with AsyncSessionLocal() as session:
        row = (
            await session.execute(select(BacktestRun).where(BacktestRun.id == run_id))
        ).scalar_one_or_none()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        if row.status in ("queued", "running"):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Run '{run_id}' is still {row.status}. "
                    f"Cancel it first via POST /runs/{run_id}/cancel, "
                    f"then delete once it reaches a terminal state."
                ),
            )
        await session.execute(sql_delete(BacktestRun).where(BacktestRun.id == run_id))
        await session.commit()
    return {"deleted": True, "run_id": run_id}


class BulkDeleteRequest(BaseModel):
    """List of run IDs to delete.  Active runs are skipped silently
    and reported in the response so the UI can surface which ones
    needed cancellation first."""
    run_ids: list[str] = Field(..., min_length=1, max_length=500)


@router.post("/runs/bulk-delete")
async def bulk_delete_runs(req: BulkDeleteRequest) -> dict[str, Any]:
    """Delete multiple terminal runs in one call.

    Active rows (queued / running) are skipped and listed under
    ``skipped_active`` so the UI can show "3 of 7 deleted, 4 still
    running — cancel them first" without needing per-row error
    handling.
    """
    from sqlalchemy import delete as sql_delete, select
    from models.database import AsyncSessionLocal, BacktestRun

    deleted: list[str] = []
    skipped_active: list[str] = []
    not_found: list[str] = []
    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(BacktestRun).where(BacktestRun.id.in_(req.run_ids))
            )
        ).scalars().all()
        seen_ids = {r.id for r in rows}
        for run_id in req.run_ids:
            if run_id not in seen_ids:
                not_found.append(run_id)
        deletable_ids: list[str] = []
        for r in rows:
            if r.status in ("queued", "running"):
                skipped_active.append(r.id)
            else:
                deletable_ids.append(r.id)
        if deletable_ids:
            await session.execute(
                sql_delete(BacktestRun).where(BacktestRun.id.in_(deletable_ids))
            )
            await session.commit()
            deleted.extend(deletable_ids)
    return {
        "deleted_count": len(deleted),
        "deleted": deleted,
        "skipped_active": skipped_active,
        "not_found": not_found,
    }


class WalkForwardRequest(BaseModel):
    source_code: str = Field(..., min_length=10)
    slug: str = Field(default="_backtest_walk_forward", min_length=1)
    config: dict[str, Any] | None = None
    token_ids: list[str] | None = None
    start: datetime
    end: datetime
    initial_capital_usd: float = Field(default=1000.0, gt=0.0)
    mode: str = Field(default="anchored", pattern="^(anchored|rolling)$")
    n_folds: int = Field(default=6, ge=2, le=24)
    train_ratio: float = Field(default=0.5, ge=0.1, le=0.95)
    embargo_seconds: float = Field(default=0.0, ge=0.0, le=86_400.0)
    submit_p50_ms: float | None = Field(default=None, ge=0.0)
    submit_p95_ms: float | None = Field(default=None, ge=0.0)
    cancel_p50_ms: float | None = Field(default=None, ge=0.0)
    cancel_p95_ms: float | None = Field(default=None, ge=0.0)
    seed: int | None = None
    concurrency: int = Field(default=2, ge=1, le=8)


@router.get("/drift")
async def drift_monitor_route(window_days: int = 30):
    """Live-vs-backtest drift report.

    For each strategy, compares the most recent backtest's Sharpe and
    trade rate against live realized performance over the last
    ``window_days``.  Flags strategies whose live behavior has
    materially diverged — the cleanest single signal of model decay
    or regime shift.
    """
    from services.backtest.drift import compute_drift

    try:
        result = await compute_drift(window_days=int(max(1, min(180, window_days))))
    except Exception as exc:
        logger.exception("Drift monitor failed")
        raise HTTPException(status_code=500, detail=f"drift failed: {exc}") from exc
    return _sanitize_floats(result.to_dict())


@router.get("/portfolio-correlation")
async def portfolio_correlation_route(
    window_days: int = 30,
    min_strategy_trades: int = 5,
):
    """Cross-strategy daily-PnL correlation matrix over the last
    ``window_days``.  Surfaces the portfolio-level question: even if
    each strategy looks fine alone, do they drown together?"""
    from services.backtest.portfolio_correlation import compute_portfolio_correlation

    try:
        result = await compute_portfolio_correlation(
            window_days=int(max(1, min(365, window_days))),
            min_strategy_trades=int(max(1, min_strategy_trades)),
        )
    except Exception as exc:
        logger.exception("Portfolio correlation failed")
        raise HTTPException(status_code=500, detail=f"correlation failed: {exc}") from exc
    return _sanitize_floats(result.to_dict())


class CPCVRequest(BaseModel):
    source_code: str = Field(..., min_length=10)
    slug: str = Field(default="_backtest_cpcv", min_length=1)
    config: dict[str, Any] | None = None
    token_ids: list[str] | None = None
    start: datetime
    end: datetime
    initial_capital_usd: float = Field(default=1000.0, gt=0.0)
    n_folds: int = Field(default=6, ge=3, le=12)
    k_test_folds: int = Field(default=2, ge=1, le=6)
    embargo_seconds: float = Field(default=3600.0, ge=0.0, le=86_400.0)
    submit_p50_ms: float | None = Field(default=None, ge=0.0)
    submit_p95_ms: float | None = Field(default=None, ge=0.0)
    cancel_p50_ms: float | None = Field(default=None, ge=0.0)
    cancel_p95_ms: float | None = Field(default=None, ge=0.0)
    seed: int | None = None
    concurrency: int = Field(default=2, ge=1, le=8)
    max_paths: int = Field(default=64, ge=4, le=200)


@router.post("/cpcv")
async def run_cpcv_route(req: CPCVRequest):
    """Combinatorial Purged Cross-Validation (Lopez de Prado).

    Evaluates the strategy on every C(n_folds, k_test_folds) combination
    of test windows, producing a distribution of out-of-sample Sharpes
    plus a Probability of Backtest Overfitting (PBO) estimate.  More
    rigorous than walk-forward: catches edges that hold up against an
    arbitrary subset of history, not just a single chronological path.
    """
    from services.backtest.cpcv import run_cpcv as _run

    try:
        result = await _run(
            source_code=req.source_code,
            slug=req.slug,
            config=req.config,
            token_ids=req.token_ids,
            start=req.start,
            end=req.end,
            initial_capital_usd=req.initial_capital_usd,
            n_folds=req.n_folds,
            k_test_folds=req.k_test_folds,
            embargo_seconds=req.embargo_seconds,
            submit_p50_ms=req.submit_p50_ms,
            submit_p95_ms=req.submit_p95_ms,
            cancel_p50_ms=req.cancel_p50_ms,
            cancel_p95_ms=req.cancel_p95_ms,
            seed=req.seed,
            concurrency=req.concurrency,
            max_paths=req.max_paths,
        )
    except Exception as exc:
        logger.exception("CPCV run failed")
        raise HTTPException(status_code=500, detail=f"cpcv failed: {exc}") from exc
    return _sanitize_floats(result.to_dict())


class MonteCarloLatencyRequest(BaseModel):
    source_code: str = Field(..., min_length=10)
    slug: str = Field(default="_backtest_mc_latency", min_length=1)
    config: dict[str, Any] | None = None
    token_ids: list[str] | None = None
    start: datetime
    end: datetime
    initial_capital_usd: float = Field(default=1000.0, gt=0.0)
    base_submit_p50_ms: float = Field(default=350.0, ge=0.0)
    base_submit_p95_ms: float = Field(default=900.0, ge=0.0)
    base_cancel_p50_ms: float = Field(default=200.0, ge=0.0)
    base_cancel_p95_ms: float = Field(default=600.0, ge=0.0)
    multipliers: list[float] = Field(default=[0.5, 0.75, 1.0, 1.5, 2.0])
    seed: int | None = 42
    concurrency: int = Field(default=2, ge=1, le=8)


@router.post("/monte-carlo-latency")
async def run_monte_carlo_latency_route(req: MonteCarloLatencyRequest):
    """Run the same backtest under multiple latency regimes.

    Returns a Sharpe-vs-latency curve so the operator can see how
    much their edge depends on the latency assumption.  A strategy
    with sharpe_slope < 0 erodes under worse network conditions
    (typical maker behavior); sharpe_slope ≈ 0 is latency-insensitive.
    """
    from services.backtest.monte_carlo import run_monte_carlo_latency as _run

    try:
        result = await _run(
            source_code=req.source_code,
            slug=req.slug,
            config=req.config,
            token_ids=req.token_ids,
            start=req.start,
            end=req.end,
            initial_capital_usd=req.initial_capital_usd,
            base_submit_p50_ms=req.base_submit_p50_ms,
            base_submit_p95_ms=req.base_submit_p95_ms,
            base_cancel_p50_ms=req.base_cancel_p50_ms,
            base_cancel_p95_ms=req.base_cancel_p95_ms,
            multipliers=tuple(req.multipliers),
            seed=req.seed,
            concurrency=req.concurrency,
        )
    except Exception as exc:
        logger.exception("Monte-carlo latency run failed")
        raise HTTPException(status_code=500, detail=f"monte-carlo failed: {exc}") from exc
    return _sanitize_floats(result.to_dict())


@router.post("/walk-forward")
async def run_walk_forward_route(req: WalkForwardRequest):
    """Run walk-forward analysis: split [start, end] into n_folds and
    backtest the strategy on each test window in chronological order.
    Returns per-fold metrics + cross-fold summary."""
    from services.backtest.walk_forward import run_walk_forward as _run

    try:
        result = await _run(
            source_code=req.source_code,
            slug=req.slug,
            config=req.config,
            token_ids=req.token_ids,
            start=req.start,
            end=req.end,
            initial_capital_usd=req.initial_capital_usd,
            mode=req.mode,  # type: ignore[arg-type]
            n_folds=req.n_folds,
            train_ratio=req.train_ratio,
            embargo_seconds=req.embargo_seconds,
            submit_p50_ms=req.submit_p50_ms,
            submit_p95_ms=req.submit_p95_ms,
            cancel_p50_ms=req.cancel_p50_ms,
            cancel_p95_ms=req.cancel_p95_ms,
            seed=req.seed,
            concurrency=req.concurrency,
        )
    except Exception as exc:
        logger.exception("Walk-forward run failed")
        raise HTTPException(status_code=500, detail=f"walk-forward failed: {exc}") from exc
    return _sanitize_floats(result.to_dict())
