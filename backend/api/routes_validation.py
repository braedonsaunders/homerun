"""Validation, async job queue, and guardrail routes."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from services.opportunity_recorder import opportunity_recorder
from services.param_optimizer import param_optimizer
from services.validation_service import validation_service
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/validation", tags=["Validation"])


class BacktestRequest(BaseModel):
    params: Optional[dict[str, Any]] = None
    save_parameter_set: bool = False
    parameter_set_name: Optional[str] = None
    activate_saved_set: bool = False


class OptimizeRequest(BaseModel):
    method: str = Field(default="grid", pattern="^(grid|random)$")
    param_ranges: Optional[dict[str, Any]] = None
    n_random_samples: int = Field(default=100, ge=5, le=2000)
    random_seed: int = Field(default=42)
    walk_forward: bool = True
    n_windows: int = Field(default=5, ge=2, le=20)
    train_ratio: float = Field(default=0.7, gt=0.1, lt=0.95)
    top_k: int = Field(default=20, ge=1, le=200)
    save_best_as_active: bool = False
    best_set_name: Optional[str] = None


class GuardrailConfigPatch(BaseModel):
    enabled: Optional[bool] = None
    min_samples: Optional[int] = Field(default=None, ge=1, le=100000)
    min_directional_accuracy: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_mae_roi: Optional[float] = Field(default=None, ge=0.0, le=1000.0)
    lookback_days: Optional[int] = Field(default=None, ge=7, le=3650)
    auto_promote: Optional[bool] = None


class ExecutionSimulationRequest(BaseModel):
    strategy_key: str = Field(min_length=2, max_length=128)
    source_key: str = Field(min_length=2, max_length=64)
    run_seed: Optional[str] = Field(default=None, min_length=4, max_length=128)
    market_provider: str = Field(default="polymarket")
    market_ref: Optional[str] = None
    market_id: Optional[str] = None
    timeframe: str = Field(default="15m")
    start_at: Optional[str] = None
    end_at: Optional[str] = None
    strategy_params: dict[str, Any] = Field(default_factory=dict)
    market_scope: dict[str, Any] = Field(default_factory=dict)
    default_notional_usd: float = Field(default=50.0, gt=0.0, le=1_000_000.0)
    slippage_bps: float = Field(default=5.0, ge=0.0, le=5000.0)
    fee_bps: float = Field(default=200.0, ge=0.0, le=10000.0)


class LiveTruthMonitorRequest(BaseModel):
    trader_id: Optional[str] = Field(default=None, min_length=2, max_length=128)
    trader_name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    duration_seconds: int = Field(
        default=300,
        ge=10,
        le=7200,
        description="How long to run the monitor script for this job.",
    )
    poll_seconds: float = Field(
        default=1.0,
        ge=0.2,
        le=10.0,
        description="Polling cadence passed to the monitor script.",
    )
    run_llm_analysis: bool = Field(default=False)
    llm_model: Optional[str] = Field(default=None, min_length=2, max_length=200)
    include_strategy_source: bool = Field(default=True)
    max_alerts_for_llm: int = Field(default=80, ge=1, le=400)
    enable_provider_checks: bool = Field(
        default=False,
        description="Enable provider/wallet API reconciliation checks (higher runtime/API load).",
    )


class CodeBacktestRequest(BaseModel):
    source_code: str = Field(min_length=10)
    slug: str = Field(default="_backtest_preview", min_length=1, max_length=128)
    config: Optional[dict[str, Any]] = None
    use_ohlc_replay: bool = True
    replay_lookback_hours: int = Field(default=24, ge=1, le=720)
    replay_timeframe: str = Field(default="30m", min_length=2, max_length=8)
    replay_max_markets: int = Field(default=80, ge=1, le=300)
    replay_max_steps: int = Field(default=72, ge=1, le=500)


class CodeBacktestOptimizeRequest(BaseModel):
    source_code: str = Field(min_length=10)
    slug: str = Field(default="_sweep_preview", min_length=1, max_length=128)
    param_grid: dict[str, list[Any]] = Field(
        ...,
        description="Dict mapping config param names to lists of values to sweep",
    )
    train_ratio: float = Field(default=0.75, gt=0.1, lt=0.95)
    top_k: int = Field(default=10, ge=1, le=200)

_LIVE_TRUTH_EXPORT_ARTIFACTS = {
    "summary_json",
    "report_jsonl",
    "llm_analysis_json",
    "bundle_json",
}


@router.get("/overview")
async def get_validation_overview():
    try:
        current_params = param_optimizer.get_current_params()
        param_specs = param_optimizer.get_param_specs()
        optimization_results = param_optimizer.get_optimization_results()
        active_set = await param_optimizer.load_active_parameter_set()
        all_sets = await param_optimizer.list_parameter_sets()
        jobs = await validation_service.list_jobs(limit=25)

        opportunity_stats = await opportunity_recorder.get_opportunity_stats()
        strategy_accuracy = await opportunity_recorder.get_strategy_accuracy()
        roi_30d = await opportunity_recorder.get_historical_roi(days=30)
        decay_30d = await opportunity_recorder.get_decay_analysis(days=30)
        calibration = await validation_service.compute_calibration_metrics(days=90)
        calibration_trend = await validation_service.compute_calibration_trend(days=90, bucket_days=7)
        strategy_health = await validation_service.get_strategy_health()
        guardrail_config = await validation_service.get_guardrail_config()
        trader_orchestrator_execution = await validation_service.compute_trader_orchestrator_execution_metrics(days=30)
        events_resolver = await validation_service.compute_events_resolver_metrics(days=7)

        latest_optimization = optimization_results[0] if optimization_results else None
        return {
            "current_params": current_params,
            "active_parameter_set": active_set,
            "parameter_spec_count": len(param_specs),
            "parameter_set_count": len(all_sets),
            "latest_optimization": latest_optimization,
            "opportunity_stats": opportunity_stats,
            "strategy_accuracy": strategy_accuracy,
            "roi_30d": roi_30d,
            "decay_30d": decay_30d,
            "calibration_90d": calibration,
            "calibration_trend_90d": calibration_trend,
            "strategy_health": strategy_health,
            "guardrail_config": guardrail_config,
            "trader_orchestrator_execution_30d": trader_orchestrator_execution,
            "events_resolver_7d": events_resolver,
            "jobs": jobs,
        }
    except Exception as e:
        logger.error("Failed to get validation overview", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/backtest")
async def enqueue_backtest(request: BacktestRequest):
    try:
        job_id = await validation_service.enqueue_job("backtest", payload=request.model_dump())
        return {"status": "queued", "job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/optimize")
async def enqueue_optimize(request: OptimizeRequest):
    try:
        job_id = await validation_service.enqueue_job("optimize", payload=request.model_dump())
        return {"status": "queued", "job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Backward-compatible aliases (now queue async jobs)
@router.post("/backtest")
async def run_backtest(request: BacktestRequest):
    return await enqueue_backtest(request)


@router.post("/optimize")
async def run_optimization(request: OptimizeRequest):
    return await enqueue_optimize(request)


@router.get("/jobs")
async def get_jobs(limit: int = 50):
    return {"jobs": await validation_service.list_jobs(limit=limit)}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    item = await validation_service.get_job(job_id)
    if not item:
        raise HTTPException(status_code=404, detail="Job not found")
    return item


@router.post("/jobs/live-truth-monitor")
async def enqueue_live_truth_monitor(request: LiveTruthMonitorRequest):
    try:
        payload = request.model_dump(exclude_none=True)
        job_id = await validation_service.enqueue_job("live_truth_monitor", payload=payload)
        return {"status": "queued", "job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/live-truth-monitor/raw")
async def get_live_truth_monitor_raw(
    job_id: str,
    max_alerts: int = Query(default=2000, ge=1, le=10000),
):
    item = await validation_service.get_job(job_id)
    if not item:
        raise HTTPException(status_code=404, detail="Job not found")
    if str(item.get("job_type") or "").strip().lower() != "live_truth_monitor":
        raise HTTPException(status_code=400, detail="Job is not a live truth monitor job")
    payload = await validation_service.get_live_truth_monitor_raw(job_id, max_alerts=max_alerts)
    if payload is None:
        raise HTTPException(status_code=404, detail="Live truth monitor payload unavailable")
    return payload


@router.get("/jobs/{job_id}/live-truth-monitor/export")
async def export_live_truth_monitor_artifact(
    job_id: str,
    artifact: str = Query(default="bundle_json"),
):
    normalized_artifact = str(artifact or "").strip().lower()
    if normalized_artifact not in _LIVE_TRUTH_EXPORT_ARTIFACTS:
        raise HTTPException(status_code=400, detail=f"Unsupported artifact '{artifact}'")

    item = await validation_service.get_job(job_id)
    if not item:
        raise HTTPException(status_code=404, detail="Job not found")
    if str(item.get("job_type") or "").strip().lower() != "live_truth_monitor":
        raise HTTPException(status_code=400, detail="Job is not a live truth monitor job")

    exported = await validation_service.export_live_truth_monitor_artifact(job_id, artifact=normalized_artifact)
    if exported is None:
        raise HTTPException(status_code=404, detail="Requested artifact is unavailable")
    filename, media_type, body = exported
    return Response(
        content=body,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    ok = await validation_service.cancel_job(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "cancelled", "job_id": job_id}


@router.post("/simulator/jobs")
async def enqueue_execution_simulation(request: ExecutionSimulationRequest):
    try:
        job_id = await validation_service.enqueue_job(
            "execution_simulation",
            payload=request.model_dump(),
        )
        return {"status": "queued", "job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/simulator/runs")
async def list_execution_sim_runs(limit: int = 50):
    return {"runs": await validation_service.list_execution_sim_runs(limit=limit)}


@router.get("/simulator/runs/{run_id}")
async def get_execution_sim_run(run_id: str):
    item = await validation_service.get_execution_sim_run(run_id)
    if not item:
        raise HTTPException(status_code=404, detail="Execution simulation run not found")
    return item


@router.get("/simulator/runs/{run_id}/events")
async def get_execution_sim_events(run_id: str, limit: int = 2000, offset: int = 0):
    run = await validation_service.get_execution_sim_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Execution simulation run not found")
    events = await validation_service.list_execution_sim_events(
        run_id,
        limit=limit,
        offset=offset,
    )
    return {"events": events}


@router.post("/code-backtest")
async def run_code_backtest(req: CodeBacktestRequest):
    """Run a strategy's source code against current market data.

    This compiles the strategy, loads it in a sandbox, runs detect()
    against the live market snapshot, and returns what opportunities
    it would find right now.
    """
    from services.strategy_backtester import run_strategy_backtest

    result = await run_strategy_backtest(
        source_code=req.source_code,
        slug=req.slug,
        config=req.config,
        use_ohlc_replay=req.use_ohlc_replay,
        replay_lookback_hours=req.replay_lookback_hours,
        replay_timeframe=req.replay_timeframe,
        replay_max_markets=req.replay_max_markets,
        replay_max_steps=req.replay_max_steps,
    )
    return result.to_dict()


@router.post("/code-backtest/evaluate")
async def run_evaluate_backtest_endpoint(req: CodeBacktestRequest):
    """Run a strategy's evaluate() against recent trade signals.

    Compiles the strategy, loads it in a sandbox, fetches recent signals,
    and runs evaluate() on each to show which would be selected/skipped.
    """
    from services.strategy_backtester import run_evaluate_backtest

    result = await run_evaluate_backtest(
        source_code=req.source_code,
        slug=req.slug,
        config=req.config,
    )
    return result.to_dict()


@router.post("/code-backtest/exit")
async def run_exit_backtest_endpoint(req: CodeBacktestRequest):
    """Run a strategy's should_exit() against current open positions.

    Compiles the strategy, loads it in a sandbox, fetches open positions,
    and runs should_exit() on each to show which would be closed.
    """
    from services.strategy_backtester import run_exit_backtest

    result = await run_exit_backtest(
        source_code=req.source_code,
        slug=req.slug,
        config=req.config,
    )
    return result.to_dict()


@router.post("/code-backtest/optimize")
async def run_code_backtest_optimize(req: CodeBacktestOptimizeRequest):
    from services.strategy_backtester import run_parameter_sweep

    result = await run_parameter_sweep(
        source_code=req.source_code,
        slug=req.slug,
        param_grid=req.param_grid,
        train_ratio=req.train_ratio,
        top_k=req.top_k,
    )
    return result.to_dict()


@router.get("/guardrails/config")
async def get_guardrail_config():
    return await validation_service.get_guardrail_config()


@router.put("/guardrails/config")
async def update_guardrail_config(patch: GuardrailConfigPatch):
    update = {k: v for k, v in patch.model_dump().items() if v is not None}
    return await validation_service.update_guardrail_config(update)


@router.post("/guardrails/evaluate")
async def evaluate_guardrails():
    return await validation_service.evaluate_guardrails()


@router.get("/strategy-health")
async def get_strategy_health():
    return {"strategy_health": await validation_service.get_strategy_health()}


@router.post("/strategy-health/{strategy_type}/override")
async def set_strategy_override(strategy_type: str, status: str = "active", note: Optional[str] = None):
    if status not in ("active", "demoted"):
        raise HTTPException(status_code=400, detail="status must be active or demoted")
    return await validation_service.set_strategy_override(
        strategy_type=strategy_type,
        status=status,
        manual_override=True,
        note=note,
    )


@router.delete("/strategy-health/{strategy_type}/override")
async def clear_strategy_override(strategy_type: str):
    return await validation_service.clear_strategy_override(strategy_type)


@router.get("/optimization-results")
async def get_optimization_results(top_k: int = 50):
    results = param_optimizer.get_optimization_results()
    return {"count": len(results), "results": results[: max(1, min(top_k, 500))]}


@router.get("/parameter-sets")
async def list_parameter_sets():
    sets = await param_optimizer.list_parameter_sets()
    return {"count": len(sets), "parameter_sets": sets}


@router.post("/parameter-sets/{set_id}/activate")
async def activate_parameter_set(set_id: str):
    item = await param_optimizer.load_parameter_set(set_id)
    if not item:
        raise HTTPException(status_code=404, detail="Parameter set not found")
    ok = await param_optimizer.activate_parameter_set(set_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Parameter set not found")
    params = item.get("parameters") or {}
    param_optimizer.set_params(params)
    return {"status": "success", "active_set_id": set_id}
