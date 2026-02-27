"""ML Model Trainer — trains XGBoost/LightGBM models from recorded snapshots.

Reads from ``ml_training_snapshots``, engineers features, trains a gradient-boosted
model with walk-forward cross-validation, and stores the result in ``ml_trained_models``.

Can be triggered via API (manual) or scheduled (future ML worker).
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from datetime import timedelta
from typing import Any

import numpy as np
from sqlalchemy import func, select

from models.database import (
    AsyncSessionLocal,
    MLTrainedModel,
    MLTrainingJob,
    MLTrainingSnapshot,
)
from utils.utcnow import utcnow

logger = logging.getLogger(__name__)


# ── Feature engineering ────────────────────────────────────────────────────


def _compute_features(rows: list[dict], lookback: int = 5) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build feature matrix and binary labels from snapshot rows.

    Features per row:
    - mid_price, spread, combined, liquidity, volume_24h
    - oracle_price, price_to_beat, seconds_left
    - price_change_1 .. price_change_{lookback} (lagged returns)
    - spread_change_1 (lagged spread delta)
    - oracle_distance (mid - oracle/close)
    - ptb_distance (oracle - price_to_beat)

    Label: 1 if next row's mid_price > current mid_price, else 0
    """
    if len(rows) < lookback + 2:
        return np.array([]), np.array([]), []

    feature_names = [
        "mid_price",
        "spread",
        "combined",
        "liquidity_log",
        "volume_24h_log",
        "seconds_left_norm",
        "oracle_distance",
        "ptb_distance",
    ]
    for i in range(1, lookback + 1):
        feature_names.append(f"return_{i}")
    feature_names.append("spread_change_1")

    X_rows: list[list[float]] = []
    y_rows: list[int] = []

    for idx in range(lookback, len(rows) - 1):
        current = rows[idx]
        next_row = rows[idx + 1]

        mid = current.get("mid_price")
        next_mid = next_row.get("mid_price")
        if mid is None or next_mid is None or mid == 0:
            continue

        spread = current.get("spread") or 0.0
        combined = current.get("combined") or 1.0
        liq = max(1.0, current.get("liquidity") or 1.0)
        vol = max(1.0, current.get("volume_24h") or 1.0)
        secs = current.get("seconds_left") or 900
        oracle = current.get("oracle_price") or 0.0
        ptb = current.get("price_to_beat") or 0.0

        # Normalized features
        liq_log = math.log(liq)
        vol_log = math.log(vol)
        secs_norm = min(1.0, secs / 3600.0)  # normalize to [0, 1] range (1h max)
        oracle_dist = mid - (oracle / max(oracle, 1.0)) if oracle > 0 else 0.0
        ptb_dist = (oracle - ptb) / max(abs(ptb), 1.0) if ptb > 0 else 0.0

        # Lagged returns
        returns = []
        for lag in range(1, lookback + 1):
            prev_idx = idx - lag
            if prev_idx >= 0:
                prev_mid = rows[prev_idx].get("mid_price") or mid
                ret = (mid - prev_mid) / max(abs(prev_mid), 1e-6)
            else:
                ret = 0.0
            returns.append(ret)

        # Spread change
        if idx > 0:
            prev_spread = rows[idx - 1].get("spread") or 0.0
            spread_chg = spread - prev_spread
        else:
            spread_chg = 0.0

        features = [mid, spread, combined, liq_log, vol_log, secs_norm, oracle_dist, ptb_dist] + returns + [spread_chg]

        # Label: next price up = 1, down = 0
        label = 1 if next_mid > mid else 0

        X_rows.append(features)
        y_rows.append(label)

    if not X_rows:
        return np.array([]), np.array([]), feature_names

    return np.array(X_rows, dtype=np.float64), np.array(y_rows, dtype=np.int32), feature_names


# ── Model training ────────────────────────────────────────────────────────


def _train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    hyperparams: dict | None = None,
) -> dict:
    """Train XGBoost classifier.  Returns serializable model dict."""
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError(
            "xgboost is required for ML training. Install with: pip install xgboost"
        )

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbosity": 0,
    }
    if hyperparams:
        params.update(hyperparams)

    n_estimators = params.pop("n_estimators", 200)
    clf = xgb.XGBClassifier(n_estimators=n_estimators, **params)
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Metrics
    train_proba = clf.predict_proba(X_train)[:, 1]
    test_proba = clf.predict_proba(X_test)[:, 1]
    train_pred = (train_proba >= 0.5).astype(int)
    test_pred = (test_proba >= 0.5).astype(int)

    train_acc = float(np.mean(train_pred == y_train))
    test_acc = float(np.mean(test_pred == y_test))

    # AUC
    try:
        from sklearn.metrics import roc_auc_score
        test_auc = float(roc_auc_score(y_test, test_proba))
    except Exception:
        test_auc = _manual_auc(y_test, test_proba)

    # Feature importance
    importance = dict(zip(feature_names, [float(v) for v in clf.feature_importances_]))

    # Serialize model to JSON
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name
    try:
        clf.save_model(tmp_path)
        with open(tmp_path, "r") as f:
            model_json = json.load(f)
    finally:
        os.unlink(tmp_path)

    return {
        "model_json": model_json,
        "params": {**params, "n_estimators": n_estimators},
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "test_auc": test_auc,
        "feature_importance": importance,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
    }


def _train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    hyperparams: dict | None = None,
) -> dict:
    """Train LightGBM classifier.  Returns serializable model dict."""
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError(
            "lightgbm is required for ML training. Install with: pip install lightgbm"
        )

    params = {
        "objective": "binary",
        "metric": "auc",
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbosity": -1,
    }
    if hyperparams:
        params.update(hyperparams)

    n_estimators = params.pop("n_estimators", 200)
    clf = lgb.LGBMClassifier(n_estimators=n_estimators, **params)
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    # Metrics
    train_proba = clf.predict_proba(X_train)[:, 1]
    test_proba = clf.predict_proba(X_test)[:, 1]
    train_pred = (train_proba >= 0.5).astype(int)
    test_pred = (test_proba >= 0.5).astype(int)

    train_acc = float(np.mean(train_pred == y_train))
    test_acc = float(np.mean(test_pred == y_test))

    try:
        from sklearn.metrics import roc_auc_score
        test_auc = float(roc_auc_score(y_test, test_proba))
    except Exception:
        test_auc = _manual_auc(y_test, test_proba)

    importance = dict(zip(feature_names, [float(v) for v in clf.feature_importances_]))

    # Serialize model
    model_str = clf.booster_.model_to_string()

    return {
        "model_json": {"lgbm_model_string": model_str},
        "params": {**params, "n_estimators": n_estimators},
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "test_auc": test_auc,
        "feature_importance": importance,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
    }


def _manual_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Simple AUC calculation without sklearn."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    total = 0
    for p in pos:
        total += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(total / (len(pos) * len(neg)))


# ── Walk-forward validation ───────────────────────────────────────────────


def _walk_forward_train(
    all_rows: list[dict],
    model_type: str,
    n_folds: int = 5,
    min_train_size: int = 200,
    hyperparams: dict | None = None,
) -> tuple[dict | None, list[dict]]:
    """Walk-forward cross-validation: train on expanding window, test on next fold.

    Returns (best_result, fold_summaries).
    """
    X_all, y_all, feature_names = _compute_features(all_rows)
    if len(X_all) < min_train_size + 50:
        return None, [{"error": f"Not enough data: {len(X_all)} rows (need {min_train_size + 50})"}]

    fold_size = len(X_all) // n_folds
    if fold_size < 20:
        n_folds = max(2, len(X_all) // 50)
        fold_size = len(X_all) // n_folds

    train_fn = _train_xgboost if model_type == "xgboost" else _train_lightgbm
    fold_results: list[dict] = []
    best_result = None
    best_auc = -1.0

    for fold in range(1, n_folds):
        train_end = fold * fold_size
        test_end = min((fold + 1) * fold_size, len(X_all))

        if train_end < min_train_size:
            continue

        X_train = X_all[:train_end]
        y_train = y_all[:train_end]
        X_test = X_all[train_end:test_end]
        y_test = y_all[train_end:test_end]

        if len(X_test) < 10:
            continue

        try:
            result = train_fn(X_train, y_train, X_test, y_test, feature_names, hyperparams)
            fold_summary = {
                "fold": fold,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "train_accuracy": result["train_accuracy"],
                "test_accuracy": result["test_accuracy"],
                "test_auc": result["test_auc"],
            }
            fold_results.append(fold_summary)

            if result["test_auc"] > best_auc:
                best_auc = result["test_auc"]
                best_result = result
        except Exception as exc:
            fold_results.append({"fold": fold, "error": str(exc)})

    return best_result, fold_results


# ── Main training entry point ─────────────────────────────────────────────


async def run_training_job(
    job_id: str,
    model_type: str = "xgboost",
    assets: list[str] | None = None,
    timeframes: list[str] | None = None,
    hyperparams: dict | None = None,
    days_lookback: int = 30,
) -> dict:
    """Execute a full training run.  Updates the MLTrainingJob row with progress.

    Steps:
    1. Load snapshots from DB
    2. Group by asset+timeframe
    3. Walk-forward train per group
    4. Save best model to ml_trained_models
    5. Return summary
    """
    assets = assets or ["btc", "eth", "sol", "xrp"]
    timeframes = timeframes or ["5m", "15m", "1h", "4h"]

    async def _update_job(status: str, progress: float, message: str, **extra: Any) -> None:
        try:
            async with AsyncSessionLocal() as session:
                job = (await session.execute(select(MLTrainingJob).where(MLTrainingJob.id == job_id))).scalar_one_or_none()
                if job:
                    job.status = status
                    job.progress = progress
                    job.message = message
                    if status == "running" and job.started_at is None:
                        job.started_at = utcnow()
                    if status in ("completed", "failed"):
                        job.finished_at = utcnow()
                    for k, v in extra.items():
                        if hasattr(job, k):
                            setattr(job, k, v)
                    await session.commit()
        except Exception:
            pass

    await _update_job("running", 0.05, "Loading snapshots from database...")

    # 1. Load snapshots
    cutoff = utcnow() - timedelta(days=days_lookback)
    try:
        async with AsyncSessionLocal() as session:
            rows_raw = (
                await session.execute(
                    select(MLTrainingSnapshot)
                    .where(
                        MLTrainingSnapshot.asset.in_(assets),
                        MLTrainingSnapshot.timeframe.in_(timeframes),
                        MLTrainingSnapshot.timestamp >= cutoff,
                    )
                    .order_by(MLTrainingSnapshot.timestamp)
                )
            ).scalars().all()
    except Exception as exc:
        await _update_job("failed", 0.0, f"DB query failed: {exc}", error=str(exc))
        return {"error": str(exc)}

    if not rows_raw:
        await _update_job("failed", 0.0, "No training snapshots found", error="No data available")
        return {"error": "No training snapshots found"}

    await _update_job("running", 0.15, f"Loaded {len(rows_raw)} snapshots. Grouping...")

    # 2. Group by asset+timeframe
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows_raw:
        key = (row.asset, row.timeframe)
        if key not in groups:
            groups[key] = []
        groups[key].append({
            "mid_price": row.mid_price,
            "spread": row.spread,
            "combined": row.combined,
            "liquidity": row.liquidity,
            "volume_24h": row.volume_24h,
            "seconds_left": row.seconds_left,
            "oracle_price": row.oracle_price,
            "price_to_beat": row.price_to_beat,
        })

    await _update_job("running", 0.25, f"Training across {len(groups)} asset/timeframe groups...")

    # 3. Train per group with walk-forward validation
    all_fold_results: list[dict] = []
    best_overall_result = None
    best_overall_auc = -1.0
    step = 0
    total_steps = len(groups)

    for (asset, timeframe), group_rows in groups.items():
        step += 1
        progress = 0.25 + 0.6 * (step / max(total_steps, 1))
        await _update_job("running", progress, f"Training {asset}/{timeframe} ({len(group_rows)} rows)...")

        result, folds = _walk_forward_train(group_rows, model_type, hyperparams=hyperparams)
        for fold in folds:
            fold["asset"] = asset
            fold["timeframe"] = timeframe
        all_fold_results.extend(folds)

        if result and result.get("test_auc", 0) > best_overall_auc:
            best_overall_auc = result["test_auc"]
            best_overall_result = result
            best_overall_result["_best_group"] = f"{asset}_{timeframe}"

    if best_overall_result is None:
        await _update_job("failed", 0.9, "All training folds failed or insufficient data", error="No successful training")
        return {"error": "All training folds failed", "fold_results": all_fold_results}

    # 4. For final model, train on ALL data with 80/20 split
    await _update_job("running", 0.88, "Training final model on all data...")

    all_rows_flat = []
    for group_rows in groups.values():
        all_rows_flat.extend(group_rows)

    X_all, y_all, feature_names = _compute_features(all_rows_flat)
    if len(X_all) >= 50:
        split_idx = int(len(X_all) * 0.8)
        X_train, X_test = X_all[:split_idx], X_all[split_idx:]
        y_train, y_test = y_all[:split_idx], y_all[split_idx:]

        train_fn = _train_xgboost if model_type == "xgboost" else _train_lightgbm
        try:
            final_result = train_fn(X_train, y_train, X_test, y_test, feature_names, hyperparams)
        except Exception as exc:
            await _update_job("failed", 0.9, f"Final model training failed: {exc}", error=str(exc))
            return {"error": str(exc), "fold_results": all_fold_results}
    else:
        final_result = best_overall_result

    # 5. Save to DB
    await _update_job("running", 0.93, "Saving model to database...")

    model_id = str(uuid.uuid4())
    # Determine next version
    try:
        async with AsyncSessionLocal() as session:
            max_version = (
                await session.execute(
                    select(func.max(MLTrainedModel.version)).where(
                        MLTrainedModel.name == f"crypto_directional_{model_type}"
                    )
                )
            ).scalar() or 0
    except Exception:
        max_version = 0

    new_version = max_version + 1

    date_range = None
    if rows_raw:
        date_range = {
            "start": rows_raw[0].timestamp.isoformat() if rows_raw[0].timestamp else None,
            "end": rows_raw[-1].timestamp.isoformat() if rows_raw[-1].timestamp else None,
        }

    trained_model = MLTrainedModel(
        id=model_id,
        name=f"crypto_directional_{model_type}",
        model_type=model_type,
        version=new_version,
        status="trained",
        weights_json=final_result["model_json"],
        feature_names=feature_names,
        hyperparams=final_result.get("params"),
        assets=assets,
        timeframes=timeframes,
        train_accuracy=final_result["train_accuracy"],
        test_accuracy=final_result["test_accuracy"],
        test_auc=final_result["test_auc"],
        feature_importance=final_result.get("feature_importance"),
        train_samples=final_result["train_samples"],
        test_samples=final_result["test_samples"],
        training_date_range=date_range,
        walkforward_results=all_fold_results,
        created_at=utcnow(),
    )

    try:
        async with AsyncSessionLocal() as session:
            session.add(trained_model)
            await session.commit()
    except Exception as exc:
        await _update_job("failed", 0.95, f"Failed to save model: {exc}", error=str(exc))
        return {"error": str(exc)}

    summary = {
        "model_id": model_id,
        "model_type": model_type,
        "version": new_version,
        "train_accuracy": final_result["train_accuracy"],
        "test_accuracy": final_result["test_accuracy"],
        "test_auc": final_result["test_auc"],
        "train_samples": final_result["train_samples"],
        "test_samples": final_result["test_samples"],
        "groups_trained": len(groups),
        "total_snapshots": len(rows_raw),
        "walkforward_folds": len(all_fold_results),
        "date_range": date_range,
    }

    await _update_job(
        "completed",
        1.0,
        f"Model v{new_version}: test_acc={final_result['test_accuracy']:.3f} AUC={final_result['test_auc']:.3f}",
        trained_model_id=model_id,
        result_summary=summary,
    )

    logger.info(
        "ML training complete: model=%s v%d acc=%.3f auc=%.3f samples=%d",
        model_type,
        new_version,
        final_result["test_accuracy"],
        final_result["test_auc"],
        len(rows_raw),
    )

    return summary


async def promote_model(model_id: str) -> dict:
    """Promote a trained model to 'active' status, demoting any previously active model."""
    async with AsyncSessionLocal() as session:
        model = (await session.execute(select(MLTrainedModel).where(MLTrainedModel.id == model_id))).scalar_one_or_none()
        if not model:
            return {"error": "Model not found"}

        # Demote all currently active models of same type
        active_models = (
            await session.execute(
                select(MLTrainedModel).where(
                    MLTrainedModel.model_type == model.model_type,
                    MLTrainedModel.status == "active",
                )
            )
        ).scalars().all()
        for m in active_models:
            m.status = "archived"

        model.status = "active"
        model.promoted_at = utcnow()
        await session.commit()

        return {
            "model_id": model.id,
            "name": model.name,
            "version": model.version,
            "status": "active",
            "promoted_at": model.promoted_at.isoformat(),
        }


async def archive_model(model_id: str) -> dict:
    """Archive a model (remove from active rotation)."""
    async with AsyncSessionLocal() as session:
        model = (await session.execute(select(MLTrainedModel).where(MLTrainedModel.id == model_id))).scalar_one_or_none()
        if not model:
            return {"error": "Model not found"}
        model.status = "archived"
        await session.commit()
        return {"model_id": model.id, "status": "archived"}
