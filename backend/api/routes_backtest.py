"""Unified backtest API.

Single entry point for the new BacktestStudio UI.  Wraps the
unified_runner, exposes recent-runs history (in-memory LRU, 32 deep)
and per-run retrieval.

Old code-backtest endpoints in routes_validation.py are unchanged
for back-compat.  The new UI uses these.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.backtest.unified_runner import (
    get_recent_run,
    list_recent_runs,
    run_unified_backtest,
)


logger = logging.getLogger("routes.backtest")
router = APIRouter(prefix="/backtest", tags=["Backtest"])


class UnifiedBacktestRequest(BaseModel):
    source_code: str = Field(..., min_length=10)
    slug: str = Field(default="_backtest_unified", min_length=1)
    config: dict[str, Any] | None = None
    token_ids: list[str] | None = None
    start: datetime | None = None
    end: datetime | None = None
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
    try:
        result = await run_unified_backtest(
            source_code=req.source_code,
            slug=req.slug,
            config=req.config,
            token_ids=req.token_ids,
            start=req.start,
            end=req.end,
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
    return result


@router.get("/runs")
async def list_runs(limit: int = 32) -> dict[str, list[dict[str, Any]]]:
    return {"runs": await list_recent_runs(limit=int(max(1, min(200, limit))))}


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    run = await get_recent_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return run


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
    return result.to_dict()


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
    return result.to_dict()


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
    return result.to_dict()


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
    return result.to_dict()


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
    return result.to_dict()
