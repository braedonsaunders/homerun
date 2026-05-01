from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from services.machine_learning_sdk import get_machine_learning_sdk
from services.ml import list_ml_capabilities

router = APIRouter(prefix="/ml", tags=["Machine Learning"])


@router.get("/tasks")
async def list_tasks_route() -> dict[str, list[dict[str, Any]]]:
    """List every registered MLCapability — built-in + strategy-attached.

    Frontend reads this to populate task_key dropdowns instead of
    hardcoding ``crypto_directional`` strings.  Adding a new ML task
    means declaring ``ml_capability = MLCapability(...)`` on a
    strategy class — it shows up here automatically.
    """
    return {"tasks": [cap.to_summary() for cap in list_ml_capabilities()]}


def _translate_ml_error(exc: Exception) -> HTTPException:
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=503, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


class RecorderConfigUpdate(BaseModel):
    is_recording: bool | None = None
    interval_seconds: int | None = Field(None, ge=5, le=3600)
    retention_days: int | None = Field(None, ge=1, le=365)
    assets: list[str] | None = None
    timeframes: list[str] | None = None
    schedule_enabled: bool | None = None
    schedule_start_utc: str | None = None
    schedule_end_utc: str | None = None


class ImportModelRequest(BaseModel):
    source_uri: str = Field(..., min_length=1)
    backend: str = Field(..., min_length=1)
    task_key: str = Field(default="crypto_directional", min_length=1)
    name: str | None = None
    version: str | None = None
    manifest_uri: str | None = None
    metadata: dict[str, Any] | None = None
    options: dict[str, Any] | None = None


class TrainAdapterRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    task_key: str = Field(default="crypto_directional", min_length=1)
    base_model_id: str = Field(..., min_length=1)
    adapter_kind: str = Field(..., min_length=1)
    name: str | None = None
    training_window_days: int = Field(default=90, ge=7, le=365)
    holdout_days: int = Field(default=7, ge=1, le=90)
    params: dict[str, Any] | None = None


class PruneDataRequest(BaseModel):
    older_than_days: int | None = Field(default=None, ge=1, le=365)
    confirm: bool = False


class UpdateDeploymentRequest(BaseModel):
    base_model_id: str | None = None
    adapter_id: str | None = None
    is_active: bool


class TriggerEvaluationRequest(BaseModel):
    target_type: str = Field(..., pattern=r"^(model|adapter|deployment)$")
    target_id: str = Field(..., min_length=1)


@router.get("/capabilities")
async def get_capabilities() -> Any:
    return await get_machine_learning_sdk().get_capabilities()


@router.get("/recorder/config")
async def get_recorder_config() -> Any:
    return await get_machine_learning_sdk().get_recorder_config()


@router.put("/recorder/config")
async def update_recorder_config(body: RecorderConfigUpdate) -> Any:
    try:
        return await get_machine_learning_sdk().update_recorder_config(body.model_dump(exclude_none=True))
    except Exception as exc:
        raise _translate_ml_error(exc) from exc


@router.get("/data/stats")
async def get_data_stats() -> Any:
    return await get_machine_learning_sdk().get_data_stats()


@router.post("/data/prune")
async def prune_data(body: PruneDataRequest) -> Any:
    sdk = get_machine_learning_sdk()
    if body.confirm:
        return await sdk.prune_data(body.older_than_days)
    return await sdk.preview_prune_data(body.older_than_days)


@router.delete("/data")
async def delete_data(confirm: bool = Query(default=False)) -> Any:
    if not confirm:
        raise HTTPException(status_code=400, detail="Pass confirm=true to delete recorded data")
    return await get_machine_learning_sdk().delete_all_data()


@router.post("/models/import")
async def import_model(body: ImportModelRequest) -> Any:
    try:
        return await get_machine_learning_sdk().import_model(body.model_dump(exclude_none=True))
    except Exception as exc:
        raise _translate_ml_error(exc) from exc


@router.get("/models")
async def list_models(status: str | None = Query(default=None), limit: int = Query(default=100, ge=1, le=500)) -> Any:
    return {"models": await get_machine_learning_sdk().list_models(status=status, limit=limit)}


@router.get("/models/{model_id}")
async def get_model(model_id: str) -> Any:
    model = await get_machine_learning_sdk().get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"model": model}


@router.post("/models/{model_id}/archive")
async def archive_model(model_id: str) -> Any:
    try:
        result = await get_machine_learning_sdk().archive_model(model_id)
    except Exception as exc:
        raise _translate_ml_error(exc) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return result


@router.delete("/models/{model_id}")
async def delete_model(model_id: str, confirm: bool = Query(default=False)) -> Any:
    if not confirm:
        raise HTTPException(status_code=400, detail="Pass confirm=true to delete the model")
    try:
        result = await get_machine_learning_sdk().delete_model(model_id)
    except Exception as exc:
        raise _translate_ml_error(exc) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return result


@router.post("/adapters/train")
async def train_adapter(body: TrainAdapterRequest) -> Any:
    try:
        return await get_machine_learning_sdk().start_adapter_training(body.model_dump(exclude_none=True))
    except Exception as exc:
        raise _translate_ml_error(exc) from exc


@router.get("/adapters")
async def list_adapters(status: str | None = Query(default=None), limit: int = Query(default=100, ge=1, le=500)) -> Any:
    return {"adapters": await get_machine_learning_sdk().list_adapters(status=status, limit=limit)}


@router.get("/adapters/{adapter_id}")
async def get_adapter(adapter_id: str) -> Any:
    adapter = await get_machine_learning_sdk().get_adapter(adapter_id)
    if adapter is None:
        raise HTTPException(status_code=404, detail="Adapter not found")
    return {"adapter": adapter}


@router.post("/adapters/{adapter_id}/archive")
async def archive_adapter(adapter_id: str) -> Any:
    try:
        result = await get_machine_learning_sdk().archive_adapter(adapter_id)
    except Exception as exc:
        raise _translate_ml_error(exc) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Adapter not found")
    return result


@router.delete("/adapters/{adapter_id}")
async def delete_adapter(adapter_id: str, confirm: bool = Query(default=False)) -> Any:
    if not confirm:
        raise HTTPException(status_code=400, detail="Pass confirm=true to delete the adapter")
    try:
        result = await get_machine_learning_sdk().delete_adapter(adapter_id)
    except Exception as exc:
        raise _translate_ml_error(exc) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Adapter not found")
    return result


@router.get("/deployments")
async def list_deployments() -> Any:
    return {"deployments": await get_machine_learning_sdk().list_deployments()}


@router.get("/deployments/{task_key}")
async def get_deployment(task_key: str) -> Any:
    return {"deployment": await get_machine_learning_sdk().get_deployment(task_key)}


@router.put("/deployments/{task_key}")
async def update_deployment(task_key: str, body: UpdateDeploymentRequest) -> Any:
    try:
        deployment = await get_machine_learning_sdk().update_deployment(task_key, body.model_dump(exclude_none=True))
    except Exception as exc:
        raise _translate_ml_error(exc) from exc
    return {"deployment": deployment}


@router.get("/jobs")
async def list_jobs(kind: str | None = Query(default=None), status: str | None = Query(default=None), limit: int = Query(default=100, ge=1, le=500)) -> Any:
    return {"jobs": await get_machine_learning_sdk().list_jobs(kind=kind, status=status, limit=limit)}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> Any:
    job = await get_machine_learning_sdk().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job": job}


@router.post("/evaluations")
async def trigger_evaluation(body: TriggerEvaluationRequest) -> Any:
    try:
        return await get_machine_learning_sdk().trigger_evaluation(body.model_dump())
    except Exception as exc:
        raise _translate_ml_error(exc) from exc
