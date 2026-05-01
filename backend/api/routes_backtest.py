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
        )
    except Exception as exc:
        logger.exception("Unified backtest failed")
        raise HTTPException(status_code=500, detail=f"backtest failed: {exc}") from exc
    return result


@router.get("/runs")
async def list_runs() -> dict[str, list[dict[str, Any]]]:
    return {"runs": list_recent_runs()}


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    run = get_recent_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return run
