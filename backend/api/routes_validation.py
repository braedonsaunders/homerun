"""Validation, async job queue, and guardrail routes."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.opportunity_recorder import opportunity_recorder
from services.param_optimizer import param_optimizer
from services.scanner import scanner
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


def _get_combinatorial_validation_stats() -> dict[str, Any]:
    for strategy in scanner.strategies:
        st = getattr(strategy, "strategy_type", None)
        st_value = getattr(st, "value", st)
        if st_value == "combinatorial" and hasattr(strategy, "get_validation_stats"):
            try:
                return strategy.get_validation_stats()
            except Exception:
                logger.warning("Failed to get combinatorial validation stats")
                return {}
    return {}


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
        combinatorial_validation = _get_combinatorial_validation_stats()
        strategy_health = await validation_service.get_strategy_health()
        guardrail_config = await validation_service.get_guardrail_config()
        trader_orchestrator_execution = await validation_service.compute_trader_orchestrator_execution_metrics(days=30)
        world_intel_resolver = await validation_service.compute_world_intel_resolver_metrics(days=7)

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
            "combinatorial_validation": combinatorial_validation,
            "strategy_health": strategy_health,
            "guardrail_config": guardrail_config,
            "trader_orchestrator_execution_30d": trader_orchestrator_execution,
            "world_intel_resolver_7d": world_intel_resolver,
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


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    ok = await validation_service.cancel_job(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "cancelled", "job_id": job_id}


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
