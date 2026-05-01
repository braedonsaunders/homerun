from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import math
import pickle
import shutil
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, select

from models.database import (
    AsyncSessionLocal,
    MLRecorderConfig,
    MLTrainingSnapshot,
    MachineLearningAdapterArtifact,
    MachineLearningDeployment,
    MachineLearningJob,
    MachineLearningModelArtifact,
)
from services.machine_learning_tasks import get_machine_learning_task, list_machine_learning_tasks
from utils.logger import get_logger
from utils.utcnow import utcnow

logger = get_logger(__name__)

_BACKEND_LABELS = {
    "sklearn_joblib": "scikit-learn Joblib",
    "sklearn_pickle": "scikit-learn Pickle",
    "onnx": "ONNX Runtime",
    "xgboost_json": "XGBoost Booster",
    "lightgbm_booster": "LightGBM Booster",
}
_ADAPTER_LABELS = {
    "platt_scaler": "Platt Scaler",
    "residual_logistic": "Residual Logistic",
}
_RECORDER_CONFIG_CACHE_TTL_SECONDS = 15.0
_ACTIVE_DEPLOYMENT_CACHE_TTL_SECONDS = 5.0


def _to_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _clamp_probability(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return max(0.000001, min(0.999999, float(value)))


def _sigmoid(value: float) -> float:
    bounded = max(-60.0, min(60.0, float(value)))
    return 1.0 / (1.0 + math.exp(-bounded))


def _logit(probability: float) -> float:
    clipped = max(0.000001, min(0.999999, float(probability)))
    return math.log(clipped / (1.0 - clipped))


def _iso(value: datetime | None) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value is not None else None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _artifacts_root() -> Path:
    root = _repo_root() / "data" / "machine_learning"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _ensure_json_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _ensure_json_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_availability_payload(backend: str) -> tuple[bool, str | None]:
    normalized = str(backend or "").strip().lower()
    def _require(module_name: str) -> tuple[bool, str | None]:
        if importlib.util.find_spec(module_name) is None:
            return False, f"Missing Python module '{module_name}'"
        return True, None

    if normalized == "sklearn_joblib":
        available, detail = _require("joblib")
        if not available:
            return available, detail
        return _require("sklearn")
    if normalized == "sklearn_pickle":
        return _require("sklearn")
    if normalized == "onnx":
        return _require("onnxruntime")
    if normalized == "xgboost_json":
        return _require("xgboost")
    if normalized == "lightgbm_booster":
        return _require("lightgbm")
    return False, f"Unsupported backend '{backend}'"


def _copy_into_artifact_dir(source_path: Path, destination_dir: Path) -> tuple[Path, str]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / source_path.name
    shutil.copy2(source_path, destination_path)
    return destination_path, _hash_file(destination_path)


def _load_manifest_file(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Manifest file must contain a JSON object")
    return dict(loaded)


def _serialize_evaluation(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_count": int(metrics.get("sample_count") or 0),
        "accuracy": _to_float(metrics.get("accuracy")),
        "auc": _to_float(metrics.get("auc")),
        "log_loss": _to_float(metrics.get("log_loss")),
        "brier_score": _to_float(metrics.get("brier_score")),
        "updated_at": _iso(metrics.get("updated_at")) if isinstance(metrics.get("updated_at"), datetime) else metrics.get("updated_at"),
    }


class MachineLearningSDK:
    def __init__(self) -> None:
        self._recorder_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._last_record_mono_by_task: dict[str, float] = {}
        self._active_deployment_cache: dict[str, tuple[float, dict[str, Any] | None]] = {}
        self._loaded_deployment_cache: dict[str, tuple[str, str, dict[str, Any]]] = {}
        self._background_tasks: set[asyncio.Task[Any]] = set()

    async def get_capabilities(self) -> dict[str, Any]:
        async with AsyncSessionLocal() as session:
            active_count = (
                await session.execute(select(func.count(MachineLearningDeployment.id)))
            ).scalar() or 0
        return {
            "available": True,
            "runtime_active": bool(active_count),
            "active_deployment_count": int(active_count),
            "import_backends": [
                {
                    "backend": backend,
                    "label": _BACKEND_LABELS[backend],
                    "available": available,
                    "reason": detail,
                    "lazy": True,
                    "supported_formats": [backend],
                }
                for backend, (available, detail) in {
                    backend: _runtime_availability_payload(backend)
                    for backend in _BACKEND_LABELS
                }.items()
            ],
            "tasks": [
                {
                    "task_key": task.task_key,
                    "label": task.label,
                    "description": task.description,
                    "supports_recording": True,
                    "supports_import": True,
                    "supports_adapters": True,
                    "supports_evaluation": True,
                }
                for task in list_machine_learning_tasks()
            ],
            "adapter_kinds": [
                {"kind": adapter_kind, "label": _ADAPTER_LABELS[adapter_kind], "description": _ADAPTER_LABELS[adapter_kind]}
                for adapter_kind in _ADAPTER_LABELS
            ],
            "notes": [
                "Model backends load lazily only when an active deployment is used.",
                "Recording can run without an active deployment when the recorder is enabled.",
            ],
        }

    async def _get_recorder_row(self) -> MLRecorderConfig | None:
        async with AsyncSessionLocal() as session:
            return (
                await session.execute(select(MLRecorderConfig).where(MLRecorderConfig.id == "default"))
            ).scalar_one_or_none()

    async def get_recorder_config(self, task_key: str = "crypto_directional") -> dict[str, Any]:
        task = get_machine_learning_task(task_key)
        cache_key = task.task_key
        cached = self._recorder_cache.get(cache_key)
        now_mono = time.monotonic()
        if cached is not None and (now_mono - cached[0]) < _RECORDER_CONFIG_CACHE_TTL_SECONDS:
            return dict(cached[1])

        row = await self._get_recorder_row()
        if row is None:
            config = {
                "task_key": task.task_key,
                "is_recording": False,
                "interval_seconds": 60,
                "retention_days": 90,
                "assets": list(task.allowed_assets),
                "timeframes": list(task.allowed_timeframes),
                "schedule_enabled": False,
                "schedule_start_utc": None,
                "schedule_end_utc": None,
            }
        else:
            config = {
                "task_key": task.task_key,
                "is_recording": bool(row.is_recording),
                "interval_seconds": max(5, int(row.interval_seconds or 60)),
                "retention_days": max(1, int(row.retention_days or 90)),
                "assets": task.normalize_assets(_ensure_json_list(row.assets)),
                "timeframes": task.normalize_timeframes(_ensure_json_list(row.timeframes)),
                "schedule_enabled": bool(row.schedule_enabled),
                "schedule_start_utc": row.schedule_start_utc,
                "schedule_end_utc": row.schedule_end_utc,
            }

        async with AsyncSessionLocal() as session:
            snapshot_rollup = (
                await session.execute(
                    select(
                        func.count(MLTrainingSnapshot.id),
                        func.min(MLTrainingSnapshot.timestamp),
                        func.max(MLTrainingSnapshot.timestamp),
                    ).where(MLTrainingSnapshot.task_key == task.task_key)
                )
            ).one()
            total_snapshots = snapshot_rollup[0] or 0
            oldest = snapshot_rollup[1]
            newest = snapshot_rollup[2]

        payload = {
            "config": config,
            "total_snapshots": int(total_snapshots),
            "oldest_snapshot": _iso(oldest),
            "newest_snapshot": _iso(newest),
        }
        self._recorder_cache[cache_key] = (now_mono, dict(payload))
        return payload

    async def update_recorder_config(self, payload: dict[str, Any], task_key: str = "crypto_directional") -> dict[str, Any]:
        task = get_machine_learning_task(task_key)
        async with AsyncSessionLocal() as session:
            row = (
                await session.execute(select(MLRecorderConfig).where(MLRecorderConfig.id == "default"))
            ).scalar_one_or_none()
            if row is None:
                row = MLRecorderConfig(id="default")
                session.add(row)

            if "is_recording" in payload:
                row.is_recording = bool(payload.get("is_recording"))
            if "interval_seconds" in payload and payload.get("interval_seconds") is not None:
                row.interval_seconds = max(5, int(payload["interval_seconds"]))
            if "retention_days" in payload and payload.get("retention_days") is not None:
                row.retention_days = max(1, int(payload["retention_days"]))
            if "assets" in payload and payload.get("assets") is not None:
                row.assets = task.normalize_assets(list(payload["assets"]))
            if "timeframes" in payload and payload.get("timeframes") is not None:
                row.timeframes = task.normalize_timeframes(list(payload["timeframes"]))
            if "schedule_enabled" in payload:
                row.schedule_enabled = bool(payload.get("schedule_enabled"))
            if "schedule_start_utc" in payload:
                row.schedule_start_utc = str(payload.get("schedule_start_utc") or "").strip() or None
            if "schedule_end_utc" in payload:
                row.schedule_end_utc = str(payload.get("schedule_end_utc") or "").strip() or None
            row.updated_at = utcnow()
            await session.commit()

        self._recorder_cache.pop(task.task_key, None)
        return await self.get_recorder_config(task.task_key)

    def _in_schedule(self, config: dict[str, Any]) -> bool:
        if not bool(config.get("schedule_enabled")):
            return True
        now = utcnow()
        current_minutes = now.hour * 60 + now.minute
        start_text = str(config.get("schedule_start_utc") or "00:00").strip()
        end_text = str(config.get("schedule_end_utc") or "23:59").strip()
        try:
            start_hour, start_minute = (int(part) for part in start_text.split(":"))
            end_hour, end_minute = (int(part) for part in end_text.split(":"))
        except (TypeError, ValueError):
            return True
        start_total = start_hour * 60 + start_minute
        end_total = end_hour * 60 + end_minute
        if start_total <= end_total:
            return start_total <= current_minutes <= end_total
        return current_minutes >= start_total or current_minutes <= end_total

    async def is_recording_enabled(self, task_key: str = "crypto_directional") -> bool:
        recorder = await self.get_recorder_config(task_key)
        config = _ensure_json_dict(recorder.get("config"))
        return bool(config.get("is_recording")) and self._in_schedule(config)

    async def get_runtime_state(self, task_key: str = "crypto_directional") -> dict[str, Any]:
        task = get_machine_learning_task(task_key)
        recorder = await self.get_recorder_config(task.task_key)
        config = _ensure_json_dict(recorder.get("config"))
        recording_enabled = bool(config.get("is_recording")) and self._in_schedule(config)
        deployment = await self.get_active_deployment(task.task_key)
        deployment_active = deployment is not None
        return {
            "task_key": task.task_key,
            "recording_enabled": recording_enabled,
            "deployment_active": deployment_active,
            "engaged": recording_enabled or deployment_active,
        }

    async def record_market_batch(self, *, task_key: str, markets: list[dict[str, Any]]) -> dict[str, Any]:
        task = get_machine_learning_task(task_key)
        recorder = await self.get_recorder_config(task.task_key)
        config = _ensure_json_dict(recorder.get("config"))
        if not bool(config.get("is_recording")) or not self._in_schedule(config):
            return {"recorded": 0}

        now_mono = time.monotonic()
        last_recorded = self._last_record_mono_by_task.get(task.task_key, 0.0)
        interval_seconds = max(5, int(config.get("interval_seconds") or 60))
        if (now_mono - last_recorded) < interval_seconds:
            return {"recorded": 0}

        allowed_assets = set(task.normalize_assets(list(config.get("assets") or [])))
        allowed_timeframes = set(task.normalize_timeframes(list(config.get("timeframes") or [])))
        recorded_at = utcnow()
        snapshot_rows: list[MLTrainingSnapshot] = []
        for market in markets:
            if not isinstance(market, dict):
                continue
            record = task.build_snapshot_record(market, recorded_at=recorded_at)
            if record is None:
                continue
            if record["asset"] not in allowed_assets or record["timeframe"] not in allowed_timeframes:
                continue
            snapshot_rows.append(MLTrainingSnapshot(id=str(uuid.uuid4()), **record))

        if not snapshot_rows:
            return {"recorded": 0}

        async with AsyncSessionLocal() as session:
            session.add_all(snapshot_rows)
            await session.commit()

        self._last_record_mono_by_task[task.task_key] = now_mono
        self._recorder_cache.pop(task.task_key, None)
        return {"recorded": len(snapshot_rows), "recorded_at": _iso(recorded_at)}

    async def get_data_stats(self, task_key: str = "crypto_directional") -> dict[str, Any]:
        task = get_machine_learning_task(task_key)
        async with AsyncSessionLocal() as session:
            total_snapshots = (
                await session.execute(
                    select(func.count(MLTrainingSnapshot.id)).where(MLTrainingSnapshot.task_key == task.task_key)
                )
            ).scalar() or 0
            groups_raw = (
                await session.execute(
                    select(
                        MLTrainingSnapshot.task_key,
                        MLTrainingSnapshot.asset,
                        MLTrainingSnapshot.timeframe,
                        func.count(MLTrainingSnapshot.id),
                        func.max(MLTrainingSnapshot.timestamp),
                    )
                    .where(MLTrainingSnapshot.task_key == task.task_key)
                    .group_by(
                        MLTrainingSnapshot.task_key,
                        MLTrainingSnapshot.asset,
                        MLTrainingSnapshot.timeframe,
                    )
                    .order_by(MLTrainingSnapshot.asset, MLTrainingSnapshot.timeframe)
                )
            ).all()
        groups = [
            {
                "task_key": row[0],
                "asset": row[1],
                "timeframe": row[2],
                "count": int(row[3] or 0),
                "newest_at": _iso(row[4]),
            }
            for row in groups_raw
        ]
        return {
            "total_snapshots": int(total_snapshots),
            "groups": groups,
        }

    async def preview_prune_data(self, older_than_days: int | None = None, task_key: str = "crypto_directional") -> dict[str, Any]:
        recorder = await self.get_recorder_config(task_key)
        retention_days = max(1, int(older_than_days or _ensure_json_dict(recorder.get("config")).get("retention_days") or 90))
        cutoff = utcnow() - timedelta(days=retention_days)
        async with AsyncSessionLocal() as session:
            would_delete = (
                await session.execute(
                    select(func.count(MLTrainingSnapshot.id)).where(
                        MLTrainingSnapshot.task_key == task_key,
                        MLTrainingSnapshot.timestamp < cutoff,
                    )
                )
            ).scalar() or 0
        return {
            "status": "preview",
            "would_delete": int(would_delete),
            "older_than_days": retention_days,
            "cutoff": _iso(cutoff),
        }

    async def prune_data(self, older_than_days: int | None = None, task_key: str = "crypto_directional") -> dict[str, Any]:
        recorder = await self.get_recorder_config(task_key)
        retention_days = max(1, int(older_than_days or _ensure_json_dict(recorder.get("config")).get("retention_days") or 90))
        cutoff = utcnow() - timedelta(days=retention_days)
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                delete(MLTrainingSnapshot).where(
                    MLTrainingSnapshot.task_key == task_key,
                    MLTrainingSnapshot.timestamp < cutoff,
                )
            )
            await session.commit()
        self._recorder_cache.pop(task_key, None)
        return {
            "status": "pruned",
            "deleted_rows": int(result.rowcount or 0),
            "older_than_days": retention_days,
        }

    async def delete_all_data(self, task_key: str = "crypto_directional") -> dict[str, Any]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                delete(MLTrainingSnapshot).where(MLTrainingSnapshot.task_key == task_key)
            )
            await session.commit()
        self._recorder_cache.pop(task_key, None)
        return {"status": "deleted", "deleted_rows": int(result.rowcount or 0)}

    async def _create_job(
        self,
        *,
        task_key: str,
        job_type: str,
        payload: dict[str, Any] | None = None,
        target_id: str | None = None,
        status: str = "queued",
        message: str | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> MachineLearningJob:
        row = MachineLearningJob(
            id=str(uuid.uuid4()),
            task_key=task_key,
            job_type=job_type,
            status=status,
            target_id=target_id,
            message=message,
            error=error,
            payload_json=payload or {},
            result_json=result or {},
            created_at=utcnow(),
            started_at=utcnow() if status in {"running", "completed", "failed"} else None,
            finished_at=utcnow() if status in {"completed", "failed"} else None,
        )
        async with AsyncSessionLocal() as session:
            session.add(row)
            await session.commit()
            await session.refresh(row)
        return row

    async def _update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        message: str | None = None,
        error: str | None = None,
        target_id: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        async with AsyncSessionLocal() as session:
            row = (
                await session.execute(select(MachineLearningJob).where(MachineLearningJob.id == job_id))
            ).scalar_one_or_none()
            if row is None:
                return
            if status is not None:
                row.status = status
                if status == "running" and row.started_at is None:
                    row.started_at = utcnow()
                if status in {"completed", "failed"}:
                    row.finished_at = utcnow()
            if message is not None:
                row.message = message
            if error is not None:
                row.error = error
            if target_id is not None:
                row.target_id = target_id
            if result is not None:
                row.result_json = result
            await session.commit()

    def _spawn_background_job(self, coroutine: Any) -> None:
        task = asyncio.create_task(coroutine)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def _normalize_model_manifest(
        self,
        *,
        task_key: str,
        backend: str,
        name: str | None,
        version: str | None,
        source_uri: str,
        manifest_uri: str | None,
        metadata: dict[str, Any] | None,
        options: dict[str, Any] | None,
    ) -> dict[str, Any]:
        task = get_machine_learning_task(task_key)
        manifest_path = Path(manifest_uri).expanduser().resolve() if manifest_uri else None
        manifest = _load_manifest_file(manifest_path) if manifest_path else {}
        manifest.update(_ensure_json_dict(metadata))
        manifest.update(_ensure_json_dict(options))
        manifest["task_key"] = task.task_key
        manifest["backend"] = backend
        manifest["name"] = str(name or manifest.get("name") or Path(source_uri).stem).strip()
        manifest["version"] = str(version or manifest.get("version") or "1").strip()
        feature_names = manifest.get("feature_names")
        if not isinstance(feature_names, list) or not feature_names:
            feature_names = task.default_feature_names()
        manifest["feature_names"] = [str(item).strip() for item in feature_names if str(item).strip()]
        manifest["assets"] = task.normalize_assets(manifest.get("assets"))
        manifest["timeframes"] = task.normalize_timeframes(manifest.get("timeframes"))
        manifest["model_kind"] = str(manifest.get("model_kind") or backend).strip()
        manifest["input_name"] = str(manifest.get("input_name") or "").strip() or None
        manifest["output_name"] = str(manifest.get("output_name") or "").strip() or None
        manifest["output_kind"] = str(manifest.get("output_kind") or "probability").strip().lower()
        return manifest

    def _load_base_runner(self, descriptor: dict[str, Any]) -> dict[str, Any]:
        manifest = _ensure_json_dict(descriptor.get("manifest"))
        backend = str(descriptor.get("backend") or "").strip().lower()
        artifact_path = Path(str(descriptor.get("artifact_path") or "")).resolve()
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact file does not exist: {artifact_path}")
        available, reason = _runtime_availability_payload(backend)
        if not available:
            raise RuntimeError(reason or f"Backend '{backend}' is unavailable")

        if backend == "sklearn_joblib":
            import joblib

            model = joblib.load(artifact_path)
            return {"backend": backend, "model": model}
        if backend == "sklearn_pickle":
            with artifact_path.open("rb") as handle:
                model = pickle.load(handle)
            return {"backend": backend, "model": model}
        if backend == "onnx":
            import onnxruntime

            session = onnxruntime.InferenceSession(str(artifact_path), providers=["CPUExecutionProvider"])
            input_name = manifest.get("input_name") or session.get_inputs()[0].name
            output_name = manifest.get("output_name") or session.get_outputs()[0].name
            return {
                "backend": backend,
                "session": session,
                "input_name": input_name,
                "output_name": output_name,
                "output_kind": manifest.get("output_kind") or "probability",
            }
        if backend == "xgboost_json":
            import xgboost

            model = xgboost.Booster()
            model.load_model(str(artifact_path))
            return {
                "backend": backend,
                "model": model,
                "feature_names": list(manifest.get("feature_names") or []),
                "output_kind": manifest.get("output_kind") or "probability",
            }
        if backend == "lightgbm_booster":
            import lightgbm

            model = lightgbm.Booster(model_file=str(artifact_path))
            return {
                "backend": backend,
                "model": model,
                "output_kind": manifest.get("output_kind") or "probability",
            }
        raise RuntimeError(f"Unsupported backend '{backend}'")

    def _predict_base_probability(self, runner: dict[str, Any], feature_vector: Any) -> float:
        import numpy as np

        backend = str(runner.get("backend") or "")
        array = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
        probability_yes: float | None = None

        if backend in {"sklearn_joblib", "sklearn_pickle"}:
            model = runner["model"]
            if hasattr(model, "predict_proba"):
                values = model.predict_proba(array)
                if getattr(values, "ndim", 0) == 2 and values.shape[1] >= 2:
                    probability_yes = _to_float(values[0][1])
                else:
                    probability_yes = _to_float(values[0][0])
            elif hasattr(model, "decision_function"):
                decision = model.decision_function(array)
                scalar = _to_float(decision[0] if hasattr(decision, "__len__") else decision)
                probability_yes = _sigmoid(float(scalar or 0.0))
            elif hasattr(model, "predict"):
                predicted = model.predict(array)
                probability_yes = _to_float(predicted[0] if hasattr(predicted, "__len__") else predicted)
        elif backend == "onnx":
            session = runner["session"]
            output_name = str(runner["output_name"])
            result = session.run([output_name], {str(runner["input_name"]): array.astype("float32")})[0]
            values = np.asarray(result)
            output_kind = str(runner.get("output_kind") or "probability")
            if values.ndim == 2 and values.shape[1] >= 2:
                probability_yes = _to_float(values[0][1])
            elif values.ndim >= 1 and values.size >= 1:
                scalar = _to_float(values.reshape(-1)[0])
                if scalar is not None:
                    probability_yes = scalar if output_kind == "probability" else _sigmoid(scalar)
        elif backend == "xgboost_json":
            import xgboost

            dmatrix = xgboost.DMatrix(array, feature_names=runner.get("feature_names") or None)
            values = runner["model"].predict(dmatrix)
            scalar = _to_float(values[0] if hasattr(values, "__len__") else values)
            if scalar is not None:
                probability_yes = scalar if str(runner.get("output_kind")) == "probability" else _sigmoid(scalar)
        elif backend == "lightgbm_booster":
            values = runner["model"].predict(array)
            scalar = _to_float(values[0] if hasattr(values, "__len__") else values)
            if scalar is not None:
                probability_yes = scalar if str(runner.get("output_kind")) == "probability" else _sigmoid(scalar)

        resolved = _clamp_probability(probability_yes)
        if resolved is None:
            raise RuntimeError("Unable to derive probability_yes from model output")
        return resolved

    async def _get_model_row(self, model_id: str) -> MachineLearningModelArtifact | None:
        async with AsyncSessionLocal() as session:
            return (
                await session.execute(
                    select(MachineLearningModelArtifact).where(MachineLearningModelArtifact.id == model_id)
                )
            ).scalar_one_or_none()

    async def _get_adapter_row(self, adapter_id: str) -> MachineLearningAdapterArtifact | None:
        async with AsyncSessionLocal() as session:
            return (
                await session.execute(
                    select(MachineLearningAdapterArtifact).where(MachineLearningAdapterArtifact.id == adapter_id)
                )
            ).scalar_one_or_none()

    async def import_model(self, payload: dict[str, Any]) -> dict[str, Any]:
        task_key = str(payload.get("task_key") or "").strip().lower()
        backend = str(payload.get("backend") or "").strip().lower()
        source_uri = str(payload.get("source_uri") or payload.get("artifact_path") or "").strip()
        if not task_key or not source_uri or not backend:
            raise ValueError("task_key, backend, and source_uri are required")
        task = get_machine_learning_task(task_key)
        source_path = Path(source_uri).expanduser().resolve()
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"Model artifact file does not exist: {source_path}")

        manifest = self._normalize_model_manifest(
            task_key=task.task_key,
            backend=backend,
            name=payload.get("name"),
            version=payload.get("version"),
            source_uri=source_uri,
            manifest_uri=payload.get("manifest_uri"),
            metadata=_ensure_json_dict(payload.get("metadata")),
            options=_ensure_json_dict(payload.get("options")),
        )

        model_id = str(uuid.uuid4())
        artifact_dir = _artifacts_root() / "models" / task.task_key / model_id
        copied_artifact_path, artifact_sha256 = _copy_into_artifact_dir(source_path, artifact_dir)
        (artifact_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        self._load_base_runner(
            {
                "backend": backend,
                "artifact_path": str(copied_artifact_path),
                "manifest": manifest,
            }
        )

        created_at = utcnow()
        row = MachineLearningModelArtifact(
            id=model_id,
            task_key=task.task_key,
            name=str(manifest["name"]),
            backend=backend,
            status="ready",
            artifact_path=str(copied_artifact_path),
            artifact_sha256=artifact_sha256,
            manifest_json=manifest,
            metrics_json={},
            source_json={
                "source_uri": str(source_path),
                "manifest_uri": str(payload.get("manifest_uri") or ""),
            },
            created_at=created_at,
            updated_at=created_at,
        )
        async with AsyncSessionLocal() as session:
            session.add(row)
            await session.commit()

        await self._create_job(
            task_key=task.task_key,
            job_type="import_model",
            status="completed",
            message=f"Imported {manifest['name']}",
            target_id=model_id,
            payload=payload,
            result={"model_id": model_id},
        )
        return {
            "status": "completed",
            "job_id": None,
            "model_id": model_id,
            "message": f"Imported {manifest['name']}",
        }

    def _serialize_model_row(self, row: MachineLearningModelArtifact, *, deployment_count: int = 0) -> dict[str, Any]:
        manifest = _ensure_json_dict(row.manifest_json)
        metrics = _ensure_json_dict(row.metrics_json)
        available, reason = _runtime_availability_payload(row.backend)
        artifact_exists = Path(str(row.artifact_path)).exists()
        runtime_ready = bool(available and artifact_exists and row.status == "ready")
        return {
            "id": row.id,
            "name": row.name,
            "version": str(manifest.get("version") or "1"),
            "task_key": row.task_key,
            "backend": row.backend,
            "model_kind": str(manifest.get("model_kind") or row.backend),
            "status": row.status,
            "source_uri": _ensure_json_dict(row.source_json).get("source_uri"),
            "manifest_uri": _ensure_json_dict(row.source_json).get("manifest_uri"),
            "runtime_ready": runtime_ready,
            "availability_reason": None if runtime_ready else (reason or ("Artifact file missing" if not artifact_exists else None)),
            "evaluation": _serialize_evaluation(_ensure_json_dict(metrics.get("evaluation")) or metrics) if metrics else None,
            "deployment_count": int(deployment_count),
            "imported_at": _iso(row.created_at),
            "created_at": _iso(row.created_at),
            "archived_at": _iso(row.archived_at),
            "metadata": {
                "feature_names": _ensure_json_list(manifest.get("feature_names")),
                "assets": _ensure_json_list(manifest.get("assets")),
                "timeframes": _ensure_json_list(manifest.get("timeframes")),
                "output_kind": manifest.get("output_kind"),
                "artifact_sha256": row.artifact_sha256,
            },
        }

    async def list_models(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            model_rows = (
                await session.execute(
                    select(MachineLearningModelArtifact)
                    .order_by(MachineLearningModelArtifact.created_at.desc())
                    .limit(max(1, int(limit)))
                )
            ).scalars().all()
            deployment_counts_raw = (
                await session.execute(
                    select(MachineLearningDeployment.base_model_id, func.count(MachineLearningDeployment.id))
                    .group_by(MachineLearningDeployment.base_model_id)
                )
            ).all()
        deployment_counts = {row[0]: int(row[1] or 0) for row in deployment_counts_raw}
        items = [self._serialize_model_row(row, deployment_count=deployment_counts.get(row.id, 0)) for row in model_rows]
        if status:
            normalized_status = str(status).strip().lower()
            items = [item for item in items if str(item.get("status") or "").lower() == normalized_status]
        return items

    async def get_model(self, model_id: str) -> dict[str, Any] | None:
        row = await self._get_model_row(model_id)
        if row is None:
            return None
        async with AsyncSessionLocal() as session:
            deployment_count = (
                await session.execute(
                    select(func.count(MachineLearningDeployment.id)).where(MachineLearningDeployment.base_model_id == model_id)
                )
            ).scalar() or 0
        return self._serialize_model_row(row, deployment_count=int(deployment_count))

    async def archive_model(self, model_id: str) -> dict[str, Any] | None:
        async with AsyncSessionLocal() as session:
            deployment = (
                await session.execute(
                    select(MachineLearningDeployment).where(MachineLearningDeployment.base_model_id == model_id)
                )
            ).scalar_one_or_none()
            if deployment is not None:
                raise ValueError("Cannot archive an active deployment model. Disable the deployment first.")
            row = (
                await session.execute(
                    select(MachineLearningModelArtifact).where(MachineLearningModelArtifact.id == model_id)
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            row.status = "archived"
            row.archived_at = utcnow()
            row.updated_at = utcnow()
            await session.commit()
        self._invalidate_runtime_for_task(row.task_key)
        return {"status": "archived", "model_id": model_id}

    async def delete_model(self, model_id: str) -> dict[str, Any] | None:
        async with AsyncSessionLocal() as session:
            deployment = (
                await session.execute(
                    select(MachineLearningDeployment).where(MachineLearningDeployment.base_model_id == model_id)
                )
            ).scalar_one_or_none()
            if deployment is not None:
                raise ValueError("Cannot delete an active deployment model. Disable the deployment first.")

            row = (
                await session.execute(
                    select(MachineLearningModelArtifact).where(MachineLearningModelArtifact.id == model_id)
                )
            ).scalar_one_or_none()
            if row is None:
                return None

            adapters = (
                await session.execute(
                    select(MachineLearningAdapterArtifact).where(MachineLearningAdapterArtifact.base_model_id == model_id)
                )
            ).scalars().all()
            for adapter in adapters:
                await session.delete(adapter)

            artifact_path = Path(str(row.artifact_path)).resolve()
            await session.delete(row)
            await session.commit()

        if artifact_path.parent.exists():
            shutil.rmtree(artifact_path.parent, ignore_errors=True)
        self._invalidate_runtime_for_task(row.task_key)
        return {"status": "deleted", "model_id": model_id}

    def _serialize_adapter_row(
        self,
        row: MachineLearningAdapterArtifact,
        *,
        base_model_name: str | None,
        runtime_ready: bool,
        availability_reason: str | None,
    ) -> dict[str, Any]:
        metrics = _ensure_json_dict(row.metrics_json)
        manifest = _ensure_json_dict(row.manifest_json)
        training_source = _ensure_json_dict(row.training_source_json)
        return {
            "id": row.id,
            "name": row.name,
            "version": str(manifest.get("version") or "1"),
            "task_key": row.task_key,
            "adapter_kind": row.adapter_type,
            "base_model_id": row.base_model_id,
            "base_model_name": base_model_name,
            "status": row.status,
            "runtime_ready": runtime_ready,
            "availability_reason": availability_reason,
            "evaluation": _serialize_evaluation(_ensure_json_dict(metrics.get("evaluation")) or metrics) if metrics else None,
            "training_window_days": training_source.get("days_lookback"),
            "holdout_days": training_source.get("holdout_days"),
            "created_at": _iso(row.created_at),
            "archived_at": _iso(row.archived_at),
            "metadata": {
                "feature_names": _ensure_json_list(manifest.get("feature_names")),
                "artifact_sha256": row.artifact_sha256,
            },
        }

    async def _load_training_rows(
        self,
        *,
        task_key: str,
        days_lookback: int,
        assets: list[str],
        timeframes: list[str],
        start_at: datetime | None = None,
        end_at: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Load training rows.

        ``start_at`` / ``end_at`` override the rolling ``days_lookback``
        window when provided — used by recording-session-scoped
        training so the adapter sees exactly the captured slice.
        """
        cutoff = (
            start_at if start_at is not None
            else utcnow() - timedelta(days=max(1, int(days_lookback)))
        )
        async with AsyncSessionLocal() as session:
            stmt = (
                select(MLTrainingSnapshot)
                .where(
                    MLTrainingSnapshot.task_key == task_key,
                    MLTrainingSnapshot.asset.in_(assets),
                    MLTrainingSnapshot.timeframe.in_(timeframes),
                    MLTrainingSnapshot.timestamp >= cutoff,
                )
                .order_by(MLTrainingSnapshot.timestamp.asc())
            )
            if end_at is not None:
                stmt = stmt.where(MLTrainingSnapshot.timestamp <= end_at)
            rows = (await session.execute(stmt)).scalars().all()
        return [
            {
                "task_key": row.task_key,
                "asset": row.asset,
                "timeframe": row.timeframe,
                "timestamp": row.timestamp,
                "mid_price": row.mid_price,
                "up_price": row.up_price,
                "down_price": row.down_price,
                "best_bid": row.best_bid,
                "best_ask": row.best_ask,
                "spread": row.spread,
                "combined": row.combined,
                "liquidity": row.liquidity,
                "volume": row.volume,
                "volume_24h": row.volume_24h,
                "oracle_price": row.oracle_price,
                "price_to_beat": row.price_to_beat,
                "seconds_left": row.seconds_left,
                "is_live": row.is_live,
            }
            for row in rows
        ]

    def _evaluate_probabilities(self, y_true: Any, probabilities: Any) -> dict[str, Any]:
        import numpy as np
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

        y_true_array = np.asarray(y_true, dtype=np.int32)
        probabilities_array = np.asarray(probabilities, dtype=np.float64)
        predicted = (probabilities_array >= 0.5).astype(np.int32)
        auc = 0.5
        if len(np.unique(y_true_array)) > 1:
            auc = float(roc_auc_score(y_true_array, probabilities_array))
        return {
            "sample_count": int(len(y_true_array)),
            "accuracy": float(np.mean(predicted == y_true_array)) if len(y_true_array) else 0.0,
            "auc": auc,
            "log_loss": float(log_loss(y_true_array, probabilities_array, labels=[0, 1])) if len(y_true_array) else 0.0,
            "brier_score": float(brier_score_loss(y_true_array, probabilities_array)) if len(y_true_array) else 0.0,
            "updated_at": utcnow(),
        }

    def _load_adapter_payload(self, adapter: dict[str, Any] | None) -> dict[str, Any] | None:
        if adapter is None:
            return None
        artifact_path = Path(str(adapter.get("artifact_path") or "")).resolve()
        if not artifact_path.exists():
            raise FileNotFoundError(f"Adapter artifact does not exist: {artifact_path}")
        with artifact_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise ValueError("Adapter artifact must be a JSON object")
        return loaded

    def _apply_adapter(
        self,
        adapter_payload: dict[str, Any] | None,
        *,
        base_probability_yes: float,
        feature_vector: Any,
    ) -> float:
        import numpy as np

        if adapter_payload is None:
            return base_probability_yes
        adapter_type = str(adapter_payload.get("adapter_type") or adapter_payload.get("format") or "").strip().lower()
        if adapter_type in {"platt_scaler", "platt_scaler_v1"}:
            coef = _to_float(adapter_payload.get("coef")) or 1.0
            intercept = _to_float(adapter_payload.get("intercept")) or 0.0
            return _clamp_probability(_sigmoid((_logit(base_probability_yes) * coef) + intercept)) or base_probability_yes
        if adapter_type in {"residual_logistic", "residual_logistic_v1"}:
            coef = [_to_float(value) or 0.0 for value in _ensure_json_list(adapter_payload.get("coef"))]
            intercept = _to_float(adapter_payload.get("intercept")) or 0.0
            means = [_to_float(value) or 0.0 for value in _ensure_json_list(adapter_payload.get("means"))]
            scales = [max(1e-9, _to_float(value) or 1.0) for value in _ensure_json_list(adapter_payload.get("scales"))]
            vector = np.asarray(feature_vector, dtype=np.float64).reshape(-1)
            combined = np.concatenate([[float(_logit(base_probability_yes))], vector])
            if len(coef) != len(combined):
                raise RuntimeError("Residual adapter coefficient shape does not match feature vector")
            standardized = (combined - np.asarray(means, dtype=np.float64)) / np.asarray(scales, dtype=np.float64)
            z_value = float(np.dot(standardized, np.asarray(coef, dtype=np.float64)) + intercept)
            return _clamp_probability(_sigmoid(z_value)) or base_probability_yes
        raise RuntimeError(f"Unsupported adapter type '{adapter_type}'")

    async def start_adapter_training(self, payload: dict[str, Any]) -> dict[str, Any]:
        task_key = str(payload.get("task_key") or "").strip().lower()
        adapter_kind = str(payload.get("adapter_kind") or "").strip().lower()
        base_model_id = str(payload.get("base_model_id") or "").strip()
        if adapter_kind not in _ADAPTER_LABELS:
            raise ValueError(f"Unsupported adapter kind '{adapter_kind}'")
        base_model = await self._get_model_row(base_model_id)
        if base_model is None:
            raise FileNotFoundError("Base model not found")
        if base_model.task_key != task_key:
            raise ValueError("Base model task_key does not match adapter task_key")

        job = await self._create_job(
            task_key=task_key,
            job_type="train_adapter",
            payload=payload,
            target_id=base_model_id,
            status="queued",
            message=f"Queued {adapter_kind} adapter training",
        )
        self._spawn_background_job(self._run_adapter_training(job.id, payload))
        return {
            "status": "queued",
            "job_id": job.id,
            "message": f"Queued {adapter_kind} adapter training",
        }

    async def _run_adapter_training(self, job_id: str, payload: dict[str, Any]) -> None:
        import numpy as np

        task_key = str(payload.get("task_key") or "").strip().lower()
        task = get_machine_learning_task(task_key)
        adapter_kind = str(payload.get("adapter_kind") or "").strip().lower()
        base_model_id = str(payload.get("base_model_id") or "").strip()
        days_lookback = max(7, int(payload.get("training_window_days") or payload.get("days_lookback") or 90))
        holdout_days = max(1, int(payload.get("holdout_days") or 7))
        name = str(payload.get("name") or f"{adapter_kind}_{base_model_id[:8]}").strip()
        await self._update_job(job_id, status="running", message="Loading recorded data")

        base_model = await self._get_model_row(base_model_id)
        if base_model is None:
            await self._update_job(job_id, status="failed", error="Base model not found", message="Base model not found")
            return

        manifest = _ensure_json_dict(base_model.manifest_json)
        feature_names = [str(item) for item in _ensure_json_list(manifest.get("feature_names")) if str(item).strip()]
        assets = task.normalize_assets(_ensure_json_list(manifest.get("assets")))
        timeframes = task.normalize_timeframes(_ensure_json_list(manifest.get("timeframes")))

        # Optional recording-session scoping.  When set, the adapter
        # trains on exactly the (asset × timeframe × time-window) slice
        # the session captured.  The session metadata is preserved in
        # the adapter manifest so downstream tooling can show
        # "trained from session X" and re-derive the slice on demand.
        session_scope: dict[str, Any] | None = None
        recording_session_id = str(payload.get("recording_session_id") or "").strip()
        session_start: datetime | None = None
        session_end: datetime | None = None
        if recording_session_id:
            try:
                from services.recording_session_service import session_training_scope
                session_scope = await session_training_scope(recording_session_id)
            except Exception:
                session_scope = None
            if session_scope is None:
                await self._update_job(
                    job_id,
                    status="failed",
                    error=f"Recording session '{recording_session_id}' not found or has no captured rows",
                    message="Session scope unavailable",
                )
                return
            session_start = session_scope.get("start")
            session_end = session_scope.get("end")
            await self._update_job(
                job_id,
                status="running",
                message=(
                    f"Loading session '{session_scope.get('session_name')}' "
                    f"({session_scope.get('rows_captured', 0)} rows captured)"
                ),
            )

        training_rows = await self._load_training_rows(
            task_key=task.task_key,
            days_lookback=days_lookback,
            assets=assets,
            timeframes=timeframes,
            start_at=session_start,
            end_at=session_end,
        )
        x_all, y_all, _, meta_rows = task.build_training_dataset(training_rows, feature_names=feature_names)
        if len(y_all) < 50:
            await self._update_job(job_id, status="failed", error="Not enough recorded data", message="Not enough recorded data")
            return

        runner = self._load_base_runner(
            {
                "backend": base_model.backend,
                "artifact_path": base_model.artifact_path,
                "manifest": manifest,
            }
        )
        base_probabilities = np.asarray(
            [self._predict_base_probability(runner, feature_vector) for feature_vector in x_all],
            dtype=np.float64,
        )

        timestamps = [meta.get("timestamp") if isinstance(meta.get("timestamp"), datetime) else None for meta in meta_rows]
        holdout_cutoff = utcnow() - timedelta(days=holdout_days)
        train_indices = [index for index, ts in enumerate(timestamps) if ts is None or ts < holdout_cutoff]
        test_indices = [index for index, ts in enumerate(timestamps) if ts is not None and ts >= holdout_cutoff]
        if len(test_indices) < 10:
            split_index = max(1, int(len(y_all) * 0.8))
            train_indices = list(range(split_index))
            test_indices = list(range(split_index, len(y_all)))
        if len(train_indices) < 20 or len(test_indices) < 10:
            await self._update_job(job_id, status="failed", error="Not enough train/test samples", message="Not enough train/test samples")
            return

        x_train = x_all[train_indices]
        x_test = x_all[test_indices]
        y_train = y_all[train_indices]
        y_test = y_all[test_indices]
        base_train = base_probabilities[train_indices]
        base_test = base_probabilities[test_indices]

        await self._update_job(job_id, message="Fitting adapter")

        if adapter_kind == "platt_scaler":
            from sklearn.linear_model import LogisticRegression

            train_features = np.asarray([[_logit(probability)] for probability in base_train], dtype=np.float64)
            test_features = np.asarray([[_logit(probability)] for probability in base_test], dtype=np.float64)
            model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=0)
            model.fit(train_features, y_train)
            adapted_test = model.predict_proba(test_features)[:, 1]
            adapter_payload = {
                "adapter_type": "platt_scaler",
                "format": "platt_scaler_v1",
                "coef": float(model.coef_[0][0]),
                "intercept": float(model.intercept_[0]),
            }
        else:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            train_features = np.column_stack([np.asarray([_logit(probability) for probability in base_train]), x_train])
            test_features = np.column_stack([np.asarray([_logit(probability) for probability in base_test]), x_test])
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_features)
            test_scaled = scaler.transform(test_features)
            model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=0)
            model.fit(train_scaled, y_train)
            adapted_test = model.predict_proba(test_scaled)[:, 1]
            adapter_payload = {
                "adapter_type": "residual_logistic",
                "format": "residual_logistic_v1",
                "coef": [float(value) for value in model.coef_[0]],
                "intercept": float(model.intercept_[0]),
                "means": [float(value) for value in scaler.mean_],
                "scales": [max(1e-9, float(value)) for value in scaler.scale_],
                "feature_names": ["base_logit", *feature_names],
            }

        metrics = self._evaluate_probabilities(y_test, adapted_test)
        adapter_id = str(uuid.uuid4())
        adapter_dir = _artifacts_root() / "adapters" / task.task_key / adapter_id
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_path = adapter_dir / "adapter.json"
        adapter_path.write_text(json.dumps(adapter_payload, indent=2, sort_keys=True), encoding="utf-8")
        adapter_sha256 = _hash_file(adapter_path)
        manifest_payload = {
            "version": "1",
            "feature_names": feature_names,
            "adapter_type": adapter_kind,
        }

        row = MachineLearningAdapterArtifact(
            id=adapter_id,
            task_key=task.task_key,
            base_model_id=base_model_id,
            name=name,
            adapter_type=adapter_kind,
            status="ready",
            artifact_path=str(adapter_path),
            artifact_sha256=adapter_sha256,
            manifest_json=manifest_payload,
            metrics_json={"evaluation": _serialize_evaluation(metrics)},
            training_source_json={
                "days_lookback": days_lookback,
                "holdout_days": holdout_days,
                "sample_count": int(len(y_all)),
                "recording_session_id": (session_scope or {}).get("session_id"),
                "recording_session_name": (session_scope or {}).get("session_name"),
                "session_start": session_start.isoformat() if session_start else None,
                "session_end": session_end.isoformat() if session_end else None,
            },
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        async with AsyncSessionLocal() as session:
            session.add(row)
            await session.commit()

        await self._update_job(
            job_id,
            status="completed",
            message=f"Trained {name}",
            target_id=adapter_id,
            result={"adapter_id": adapter_id, "evaluation": _serialize_evaluation(metrics)},
        )
        self._invalidate_runtime_for_task(task.task_key)

    async def list_adapters(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            rows = (
                await session.execute(
                    select(MachineLearningAdapterArtifact)
                    .order_by(MachineLearningAdapterArtifact.created_at.desc())
                    .limit(max(1, int(limit)))
                )
            ).scalars().all()
            model_names_raw = (
                await session.execute(select(MachineLearningModelArtifact.id, MachineLearningModelArtifact.name))
            ).all()
        model_names = {row[0]: row[1] for row in model_names_raw}
        items = []
        for row in rows:
            artifact_exists = Path(str(row.artifact_path)).exists()
            runtime_ready = bool(row.status == "ready" and artifact_exists and row.base_model_id in model_names)
            availability_reason = None if runtime_ready else ("Artifact file missing" if not artifact_exists else "Base model missing")
            items.append(
                self._serialize_adapter_row(
                    row,
                    base_model_name=model_names.get(row.base_model_id),
                    runtime_ready=runtime_ready,
                    availability_reason=availability_reason,
                )
            )
        if status:
            normalized_status = str(status).strip().lower()
            items = [item for item in items if str(item.get("status") or "").lower() == normalized_status]
        return items

    async def get_adapter(self, adapter_id: str) -> dict[str, Any] | None:
        async with AsyncSessionLocal() as session:
            row = (
                await session.execute(
                    select(MachineLearningAdapterArtifact).where(MachineLearningAdapterArtifact.id == adapter_id)
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            base_model_name = (
                await session.execute(
                    select(MachineLearningModelArtifact.name).where(MachineLearningModelArtifact.id == row.base_model_id)
                )
            ).scalar_one_or_none()
        artifact_exists = Path(str(row.artifact_path)).exists()
        runtime_ready = bool(row.status == "ready" and artifact_exists and base_model_name)
        availability_reason = None if runtime_ready else ("Artifact file missing" if not artifact_exists else "Base model missing")
        return self._serialize_adapter_row(
            row,
            base_model_name=base_model_name,
            runtime_ready=runtime_ready,
            availability_reason=availability_reason,
        )

    async def archive_adapter(self, adapter_id: str) -> dict[str, Any] | None:
        async with AsyncSessionLocal() as session:
            deployment = (
                await session.execute(
                    select(MachineLearningDeployment).where(MachineLearningDeployment.adapter_id == adapter_id)
                )
            ).scalar_one_or_none()
            if deployment is not None:
                raise ValueError("Cannot archive an active deployment adapter. Disable or change the deployment first.")
            row = (
                await session.execute(
                    select(MachineLearningAdapterArtifact).where(MachineLearningAdapterArtifact.id == adapter_id)
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            row.status = "archived"
            row.archived_at = utcnow()
            row.updated_at = utcnow()
            await session.commit()
        self._invalidate_runtime_for_task(row.task_key)
        return {"status": "archived", "adapter_id": adapter_id}

    async def delete_adapter(self, adapter_id: str) -> dict[str, Any] | None:
        async with AsyncSessionLocal() as session:
            deployment = (
                await session.execute(
                    select(MachineLearningDeployment).where(MachineLearningDeployment.adapter_id == adapter_id)
                )
            ).scalar_one_or_none()
            if deployment is not None:
                raise ValueError("Cannot delete an active deployment adapter. Disable or change the deployment first.")
            row = (
                await session.execute(
                    select(MachineLearningAdapterArtifact).where(MachineLearningAdapterArtifact.id == adapter_id)
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            artifact_path = Path(str(row.artifact_path)).resolve()
            await session.delete(row)
            await session.commit()
        if artifact_path.parent.exists():
            shutil.rmtree(artifact_path.parent, ignore_errors=True)
        self._invalidate_runtime_for_task(row.task_key)
        return {"status": "deleted", "adapter_id": adapter_id}

    async def _serialize_deployment(self, task_key: str, deployment: MachineLearningDeployment | None) -> dict[str, Any]:
        if deployment is None:
            task = get_machine_learning_task(task_key)
            return {
                "task_key": task.task_key,
                "name": task.label,
                "is_active": False,
                "base_model_id": None,
                "base_model_name": None,
                "adapter_id": None,
                "adapter_name": None,
                "runtime_state": "inactive",
                "runtime_message": "No active deployment",
                "activated_at": None,
                "updated_at": None,
                "evaluation": None,
            }

        async with AsyncSessionLocal() as session:
            base_model = (
                await session.execute(
                    select(MachineLearningModelArtifact).where(MachineLearningModelArtifact.id == deployment.base_model_id)
                )
            ).scalar_one_or_none()
            adapter = None
            if deployment.adapter_id:
                adapter = (
                    await session.execute(
                        select(MachineLearningAdapterArtifact).where(MachineLearningAdapterArtifact.id == deployment.adapter_id)
                    )
                ).scalar_one_or_none()
        evaluation = None
        if adapter is not None and isinstance(adapter.metrics_json, dict):
            evaluation = _serialize_evaluation(_ensure_json_dict(adapter.metrics_json.get("evaluation")) or adapter.metrics_json)
        elif base_model is not None and isinstance(base_model.metrics_json, dict):
            evaluation = _serialize_evaluation(_ensure_json_dict(base_model.metrics_json.get("evaluation")) or base_model.metrics_json)
        return {
            "task_key": deployment.task_key,
            "name": get_machine_learning_task(deployment.task_key).label,
            "is_active": True,
            "base_model_id": deployment.base_model_id,
            "base_model_name": base_model.name if base_model is not None else None,
            "adapter_id": deployment.adapter_id,
            "adapter_name": adapter.name if adapter is not None else None,
            "runtime_state": "active",
            "runtime_message": "Deployment active",
            "activated_at": _iso(deployment.activated_at),
            "updated_at": _iso(deployment.updated_at),
            "evaluation": evaluation,
        }

    async def list_deployments(self) -> list[dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            deployments = (
                await session.execute(select(MachineLearningDeployment))
            ).scalars().all()
        deployment_by_task = {deployment.task_key: deployment for deployment in deployments}
        return [
            await self._serialize_deployment(task.task_key, deployment_by_task.get(task.task_key))
            for task in list_machine_learning_tasks()
        ]

    async def get_deployment(self, task_key: str) -> dict[str, Any]:
        async with AsyncSessionLocal() as session:
            deployment = (
                await session.execute(
                    select(MachineLearningDeployment).where(MachineLearningDeployment.task_key == task_key)
                )
            ).scalar_one_or_none()
        return await self._serialize_deployment(task_key, deployment)

    async def update_deployment(self, task_key: str, payload: dict[str, Any]) -> dict[str, Any]:
        task = get_machine_learning_task(task_key)
        is_active = bool(payload.get("is_active"))
        async with AsyncSessionLocal() as session:
            deployment = (
                await session.execute(
                    select(MachineLearningDeployment).where(MachineLearningDeployment.task_key == task.task_key)
                )
            ).scalar_one_or_none()
            if not is_active:
                if deployment is not None:
                    await session.delete(deployment)
                    await session.commit()
                self._invalidate_runtime_for_task(task.task_key)
                return await self._serialize_deployment(task.task_key, None)

            base_model_id = str(payload.get("base_model_id") or "").strip()
            if not base_model_id:
                raise ValueError("base_model_id is required when activating a deployment")
            base_model = (
                await session.execute(
                    select(MachineLearningModelArtifact).where(MachineLearningModelArtifact.id == base_model_id)
                )
            ).scalar_one_or_none()
            if base_model is None:
                raise FileNotFoundError("Base model not found")
            if base_model.task_key != task.task_key:
                raise ValueError("Base model task_key does not match deployment task_key")

            adapter_id = str(payload.get("adapter_id") or "").strip() or None
            if adapter_id:
                adapter = (
                    await session.execute(
                        select(MachineLearningAdapterArtifact).where(MachineLearningAdapterArtifact.id == adapter_id)
                    )
                ).scalar_one_or_none()
                if adapter is None:
                    raise FileNotFoundError("Adapter not found")
                if adapter.base_model_id != base_model_id:
                    raise ValueError("Adapter base model does not match selected deployment base model")

            now = utcnow()
            if deployment is None:
                deployment = MachineLearningDeployment(
                    id=str(uuid.uuid4()),
                    task_key=task.task_key,
                    base_model_id=base_model_id,
                    adapter_id=adapter_id,
                    notes=None,
                    created_at=now,
                    updated_at=now,
                    activated_at=now,
                )
                session.add(deployment)
            else:
                deployment.base_model_id = base_model_id
                deployment.adapter_id = adapter_id
                deployment.updated_at = now
                deployment.activated_at = now
            await session.commit()

        self._invalidate_runtime_for_task(task.task_key)
        async with AsyncSessionLocal() as session:
            deployment = (
                await session.execute(
                    select(MachineLearningDeployment).where(MachineLearningDeployment.task_key == task.task_key)
                )
            ).scalar_one_or_none()
        return await self._serialize_deployment(task.task_key, deployment)

    def _serialize_job_row(self, row: MachineLearningJob) -> dict[str, Any]:
        return {
            "id": row.id,
            "kind": row.job_type,
            "status": row.status,
            "task_key": row.task_key,
            "model_id": row.target_id if row.job_type == "import_model" else None,
            "adapter_id": row.target_id if row.job_type == "train_adapter" else None,
            "deployment_task_key": row.task_key if row.job_type == "evaluate_deployment" else None,
            "progress": None,
            "message": row.message,
            "error": row.error,
            "created_at": _iso(row.created_at),
            "updated_at": _iso(row.finished_at or row.started_at or row.created_at),
            "completed_at": _iso(row.finished_at),
            "result": _ensure_json_dict(row.result_json),
        }

    async def list_jobs(self, kind: str | None = None, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            query = select(MachineLearningJob).order_by(MachineLearningJob.created_at.desc()).limit(max(1, int(limit)))
            if kind:
                query = query.where(MachineLearningJob.job_type == str(kind).strip())
            if status:
                query = query.where(MachineLearningJob.status == str(status).strip())
            rows = (await session.execute(query)).scalars().all()
        return [self._serialize_job_row(row) for row in rows]

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        async with AsyncSessionLocal() as session:
            row = (
                await session.execute(select(MachineLearningJob).where(MachineLearningJob.id == job_id))
            ).scalar_one_or_none()
        return self._serialize_job_row(row) if row is not None else None

    async def trigger_evaluation(self, payload: dict[str, Any]) -> dict[str, Any]:
        target_type = str(payload.get("target_type") or "").strip().lower()
        target_id = str(payload.get("target_id") or "").strip()
        if target_type not in {"model", "adapter", "deployment"} or not target_id:
            raise ValueError("target_type and target_id are required")
        task_key = "crypto_directional"
        if target_type == "model":
            row = await self._get_model_row(target_id)
            if row is None:
                raise FileNotFoundError("Model not found")
            task_key = row.task_key
        elif target_type == "adapter":
            row = await self._get_adapter_row(target_id)
            if row is None:
                raise FileNotFoundError("Adapter not found")
            task_key = row.task_key
        else:
            task_key = target_id

        job = await self._create_job(
            task_key=task_key,
            job_type=f"evaluate_{target_type}",
            payload=payload,
            target_id=target_id,
            status="queued",
            message=f"Queued {target_type} evaluation",
        )
        self._spawn_background_job(self._run_evaluation(job.id, target_type=target_type, target_id=target_id))
        return {"status": "queued", "job_id": job.id, "message": f"Queued {target_type} evaluation"}

    async def _run_evaluation(self, job_id: str, *, target_type: str, target_id: str) -> None:
        import numpy as np

        await self._update_job(job_id, status="running", message="Loading evaluation dataset")

        if target_type == "model":
            model = await self._get_model_row(target_id)
            if model is None:
                await self._update_job(job_id, status="failed", message="Model not found", error="Model not found")
                return
            manifest = _ensure_json_dict(model.manifest_json)
            task = get_machine_learning_task(model.task_key)
            feature_names = [str(item) for item in _ensure_json_list(manifest.get("feature_names")) if str(item).strip()]
            assets = task.normalize_assets(_ensure_json_list(manifest.get("assets")))
            timeframes = task.normalize_timeframes(_ensure_json_list(manifest.get("timeframes")))
            rows = await self._load_training_rows(task_key=task.task_key, days_lookback=90, assets=assets, timeframes=timeframes)
            x_all, y_all, _, _ = task.build_training_dataset(rows, feature_names=feature_names)
            if len(y_all) < 20:
                await self._update_job(job_id, status="failed", message="Not enough evaluation data", error="Not enough evaluation data")
                return
            runner = self._load_base_runner({"backend": model.backend, "artifact_path": model.artifact_path, "manifest": manifest})
            probabilities = np.asarray([self._predict_base_probability(runner, feature_row) for feature_row in x_all], dtype=np.float64)
            metrics = self._evaluate_probabilities(y_all, probabilities)
            async with AsyncSessionLocal() as session:
                row = (
                    await session.execute(select(MachineLearningModelArtifact).where(MachineLearningModelArtifact.id == target_id))
                ).scalar_one_or_none()
                if row is not None:
                    row.metrics_json = {"evaluation": _serialize_evaluation(metrics)}
                    row.updated_at = utcnow()
                    await session.commit()
            await self._update_job(job_id, status="completed", message="Model evaluation completed", result={"evaluation": _serialize_evaluation(metrics)})
            return

        if target_type == "adapter":
            adapter = await self._get_adapter_row(target_id)
            if adapter is None:
                await self._update_job(job_id, status="failed", message="Adapter not found", error="Adapter not found")
                return
            base_model = await self._get_model_row(adapter.base_model_id)
            if base_model is None:
                await self._update_job(job_id, status="failed", message="Base model not found", error="Base model not found")
                return
            task = get_machine_learning_task(adapter.task_key)
            manifest = _ensure_json_dict(base_model.manifest_json)
            feature_names = [str(item) for item in _ensure_json_list(manifest.get("feature_names")) if str(item).strip()]
            assets = task.normalize_assets(_ensure_json_list(manifest.get("assets")))
            timeframes = task.normalize_timeframes(_ensure_json_list(manifest.get("timeframes")))
            rows = await self._load_training_rows(task_key=task.task_key, days_lookback=90, assets=assets, timeframes=timeframes)
            x_all, y_all, _, _ = task.build_training_dataset(rows, feature_names=feature_names)
            if len(y_all) < 20:
                await self._update_job(job_id, status="failed", message="Not enough evaluation data", error="Not enough evaluation data")
                return
            runner = self._load_base_runner({"backend": base_model.backend, "artifact_path": base_model.artifact_path, "manifest": manifest})
            adapter_payload = self._load_adapter_payload(await self.get_adapter_runtime(target_id))
            probabilities = np.asarray(
                [
                    self._apply_adapter(
                        adapter_payload,
                        base_probability_yes=self._predict_base_probability(runner, feature_row),
                        feature_vector=feature_row,
                    )
                    for feature_row in x_all
                ],
                dtype=np.float64,
            )
            metrics = self._evaluate_probabilities(y_all, probabilities)
            async with AsyncSessionLocal() as session:
                row = (
                    await session.execute(select(MachineLearningAdapterArtifact).where(MachineLearningAdapterArtifact.id == target_id))
                ).scalar_one_or_none()
                if row is not None:
                    row.metrics_json = {"evaluation": _serialize_evaluation(metrics)}
                    row.updated_at = utcnow()
                    await session.commit()
            await self._update_job(job_id, status="completed", message="Adapter evaluation completed", result={"evaluation": _serialize_evaluation(metrics)})
            return

        deployment = await self.get_active_deployment(target_id)
        if deployment is None:
            await self._update_job(job_id, status="failed", message="Deployment not found", error="Deployment not found")
            return
        if deployment.get("adapter_id"):
            await self._run_evaluation(job_id, target_type="adapter", target_id=str(deployment["adapter_id"]))
            return
        await self._run_evaluation(job_id, target_type="model", target_id=str(deployment["base_model_id"]))

    async def has_active_deployment(self, task_key: str) -> bool:
        return await self.get_active_deployment(task_key) is not None

    async def get_active_deployment(self, task_key: str) -> dict[str, Any] | None:
        normalized_task_key = str(task_key or "").strip().lower()
        cached = self._active_deployment_cache.get(normalized_task_key)
        now_mono = time.monotonic()
        if cached is not None and (now_mono - cached[0]) < _ACTIVE_DEPLOYMENT_CACHE_TTL_SECONDS:
            return dict(cached[1]) if cached[1] is not None else None

        async with AsyncSessionLocal() as session:
            row = (
                await session.execute(
                    select(MachineLearningDeployment).where(MachineLearningDeployment.task_key == normalized_task_key)
                )
            ).scalar_one_or_none()
        if row is None:
            self._active_deployment_cache[normalized_task_key] = (now_mono, None)
            return None
        payload = {
            "id": row.id,
            "task_key": row.task_key,
            "base_model_id": row.base_model_id,
            "adapter_id": row.adapter_id,
            "updated_at": _iso(row.updated_at),
        }
        self._active_deployment_cache[normalized_task_key] = (now_mono, dict(payload))
        return payload

    async def get_adapter_runtime(self, adapter_id: str) -> dict[str, Any] | None:
        row = await self._get_adapter_row(adapter_id)
        if row is None:
            return None
        return {
            "id": row.id,
            "task_key": row.task_key,
            "base_model_id": row.base_model_id,
            "adapter_type": row.adapter_type,
            "artifact_path": row.artifact_path,
            "manifest": _ensure_json_dict(row.manifest_json),
        }

    def _invalidate_runtime_for_task(self, task_key: str) -> None:
        normalized_task_key = str(task_key or "").strip().lower()
        self._active_deployment_cache.pop(normalized_task_key, None)
        cache_keys = [key for key in self._loaded_deployment_cache if key.startswith(f"{normalized_task_key}:")]
        for cache_key in cache_keys:
            self._loaded_deployment_cache.pop(cache_key, None)

    async def _get_loaded_deployment(self, task_key: str) -> dict[str, Any] | None:
        deployment = await self.get_active_deployment(task_key)
        if deployment is None:
            return None
        cache_key = f"{task_key}:{deployment['id']}:{deployment.get('updated_at') or ''}"
        cached = self._loaded_deployment_cache.get(cache_key)
        if cached is not None and cached[0] == str(deployment["id"]) and cached[1] == str(deployment.get("updated_at") or ""):
            return dict(cached[2])

        base_model = await self._get_model_row(str(deployment["base_model_id"]))
        if base_model is None:
            raise FileNotFoundError("Active deployment base model not found")
        manifest = _ensure_json_dict(base_model.manifest_json)
        feature_names = [str(item) for item in _ensure_json_list(manifest.get("feature_names")) if str(item).strip()]
        base_runner = self._load_base_runner(
            {"backend": base_model.backend, "artifact_path": base_model.artifact_path, "manifest": manifest}
        )
        adapter_payload = None
        adapter_name = None
        adapter_type = None
        if deployment.get("adapter_id"):
            adapter_row = await self._get_adapter_row(str(deployment["adapter_id"]))
            if adapter_row is None:
                raise FileNotFoundError("Active deployment adapter not found")
            adapter_payload = self._load_adapter_payload(await self.get_adapter_runtime(adapter_row.id))
            adapter_name = adapter_row.name
            adapter_type = adapter_row.adapter_type
        loaded = {
            "deployment_id": str(deployment["id"]),
            "task_key": task_key,
            "feature_names": feature_names,
            "assets": get_machine_learning_task(task_key).normalize_assets(_ensure_json_list(manifest.get("assets"))),
            "timeframes": get_machine_learning_task(task_key).normalize_timeframes(_ensure_json_list(manifest.get("timeframes"))),
            "base_model_id": base_model.id,
            "base_model_name": base_model.name,
            "backend": base_model.backend,
            "base_runner": base_runner,
            "adapter_id": deployment.get("adapter_id"),
            "adapter_name": adapter_name,
            "adapter_type": adapter_type,
            "adapter_payload": adapter_payload,
        }
        self._loaded_deployment_cache[cache_key] = (
            str(deployment["id"]),
            str(deployment.get("updated_at") or ""),
            dict(loaded),
        )
        return loaded

    async def annotate_market_batch(self, *, task_key: str, markets: list[dict[str, Any]]) -> dict[str, Any]:
        task = get_machine_learning_task(task_key)
        if not markets:
            return {"annotated": 0}
        loaded = await self._get_loaded_deployment(task.task_key)
        if loaded is None:
            return {"annotated": 0}
        annotated = 0
        for market in markets:
            if not isinstance(market, dict):
                continue
            if not task.scope_matches(
                market,
                assets=list(loaded["assets"]),
                timeframes=list(loaded["timeframes"]),
            ):
                market.pop("machine_learning", None)
                continue
            feature_vector = task.feature_vector_from_market(
                market,
                feature_names=list(loaded["feature_names"]),
            )
            if feature_vector is None:
                market.pop("machine_learning", None)
                continue
            try:
                base_probability_yes = self._predict_base_probability(loaded["base_runner"], feature_vector)
                probability_yes = self._apply_adapter(
                    loaded.get("adapter_payload"),
                    base_probability_yes=base_probability_yes,
                    feature_vector=feature_vector,
                )
            except Exception as exc:
                logger.warning("Machine learning prediction failed for market", exc_info=exc)
                market.pop("machine_learning", None)
                continue

            confidence = abs(probability_yes - 0.5) * 2.0
            market["machine_learning"] = {
                "task_key": task.task_key,
                "deployment_id": loaded["deployment_id"],
                "base_model_id": loaded["base_model_id"],
                "base_model_name": loaded["base_model_name"],
                "backend": loaded["backend"],
                "adapter_id": loaded.get("adapter_id"),
                "adapter_name": loaded.get("adapter_name"),
                "adapter_type": loaded.get("adapter_type"),
                "prediction": {
                    "probability_yes": round(float(probability_yes), 6),
                    "probability_no": round(float(1.0 - probability_yes), 6),
                    "base_probability_yes": round(float(base_probability_yes), 6),
                    "confidence": round(float(confidence), 6),
                    "predicted_at": _iso(utcnow()),
                },
            }
            annotated += 1
        return {"annotated": annotated}

    def extract_probability(self, payload: dict[str, Any] | None, *, direction: str | None = None) -> float | None:
        candidate_payload = _ensure_json_dict(payload)
        machine_learning = candidate_payload.get("machine_learning")
        if isinstance(machine_learning, dict):
            prediction = _ensure_json_dict(machine_learning.get("prediction"))
            probability_yes = _to_float(prediction.get("probability_yes"))
            if probability_yes is None:
                probability_yes = _to_float(machine_learning.get("probability_yes"))
            if probability_yes is not None:
                if str(direction or "").strip().lower() == "buy_no":
                    return max(0.0, min(1.0, 1.0 - probability_yes))
                return max(0.0, min(1.0, probability_yes))
        return None


machine_learning_sdk = MachineLearningSDK()


def get_machine_learning_sdk() -> MachineLearningSDK:
    return machine_learning_sdk
