"""Validation orchestration service.

Responsibilities:
- Persistent async queue for backtest/optimization jobs
- Guardrail evaluation for strategy demotion/promotion
- Shared validation analytics used by API and scanner
"""

from __future__ import annotations

import asyncio
import math
import uuid
from datetime import datetime, timedelta
from utils.utcnow import utcnow
from typing import Any, Optional

from sqlalchemy import and_, select

from models.database import (
    AppSettings,
    AutoTraderTrade,
    AsyncSessionLocal,
    OpportunityHistory,
    StrategyValidationProfile,
    TradeSignal,
    ValidationJob,
    WorldIntelligenceSignal,
)
from services.param_optimizer import param_optimizer
from utils.logger import get_logger

logger = get_logger(__name__)


class ValidationService:
    def __init__(self) -> None:
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        self._guardrail_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        await self._recover_incomplete_jobs()
        self._worker_task = asyncio.create_task(self._worker_loop())
        self._guardrail_task = asyncio.create_task(self._guardrail_loop())
        logger.info("Validation service started")

    async def stop(self) -> None:
        self._running = False
        for task in (self._worker_task, self._guardrail_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        logger.info("Validation service stopped")

    async def enqueue_job(self, job_type: str, payload: dict[str, Any]) -> str:
        job_id = str(uuid.uuid4())
        async with AsyncSessionLocal() as session:
            session.add(
                ValidationJob(
                    id=job_id,
                    job_type=job_type,
                    status="queued",
                    payload=payload,
                    progress=0.0,
                    message="Queued",
                )
            )
            await session.commit()
        return job_id

    async def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            rows = (
                (
                    await session.execute(
                        select(ValidationJob)
                        .order_by(ValidationJob.created_at.desc())
                        .limit(max(1, min(limit, 500)))
                    )
                )
                .scalars()
                .all()
            )
        return [self._job_to_dict(r) for r in rows]

    async def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            row = await session.get(ValidationJob, job_id)
            return self._job_to_dict(row) if row else None

    async def cancel_job(self, job_id: str) -> bool:
        async with AsyncSessionLocal() as session:
            row = await session.get(ValidationJob, job_id)
            if not row:
                return False
            if row.status in ("completed", "failed", "cancelled"):
                return True
            row.status = "cancelled"
            row.finished_at = utcnow()
            row.progress = 1.0
            row.message = "Cancelled"
            await session.commit()
            return True

    async def get_guardrail_config(self) -> dict[str, Any]:
        s = await self._get_or_create_settings()
        return {
            "enabled": bool(s.validation_guardrails_enabled),
            "min_samples": int(s.validation_min_samples or 25),
            "min_directional_accuracy": float(
                s.validation_min_directional_accuracy or 0.52
            ),
            "max_mae_roi": float(s.validation_max_mae_roi or 12.0),
            "lookback_days": int(s.validation_lookback_days or 90),
            "auto_promote": bool(s.validation_auto_promote),
        }

    async def update_guardrail_config(self, patch: dict[str, Any]) -> dict[str, Any]:
        async with AsyncSessionLocal() as session:
            row = await session.get(AppSettings, "default")
            if row is None:
                row = AppSettings(id="default")
                session.add(row)
            if "enabled" in patch:
                row.validation_guardrails_enabled = bool(patch["enabled"])
            if "min_samples" in patch:
                row.validation_min_samples = max(1, int(patch["min_samples"]))
            if "min_directional_accuracy" in patch:
                row.validation_min_directional_accuracy = float(
                    patch["min_directional_accuracy"]
                )
            if "max_mae_roi" in patch:
                row.validation_max_mae_roi = float(patch["max_mae_roi"])
            if "lookback_days" in patch:
                row.validation_lookback_days = max(7, int(patch["lookback_days"]))
            if "auto_promote" in patch:
                row.validation_auto_promote = bool(patch["auto_promote"])
            row.updated_at = utcnow()
            await session.commit()
        return await self.get_guardrail_config()

    async def compute_calibration_metrics(self, days: int = 90) -> dict[str, Any]:
        cutoff = utcnow() - timedelta(days=days)
        filters = [
            OpportunityHistory.detected_at >= cutoff,
            OpportunityHistory.actual_roi.isnot(None),
            OpportunityHistory.expected_roi.isnot(None),
        ]
        async with AsyncSessionLocal() as session:
            rows = (
                (
                    await session.execute(
                        select(
                            OpportunityHistory.strategy_type,
                            OpportunityHistory.detected_at,
                            OpportunityHistory.expected_roi,
                            OpportunityHistory.actual_roi,
                        ).where(and_(*filters))
                    )
                )
                .all()
            )

        by_strategy: dict[str, list[tuple[float, float, datetime]]] = {}
        for strategy, detected_at, expected_roi, actual_roi in rows:
            if expected_roi is None or actual_roi is None:
                continue
            by_strategy.setdefault(strategy or "unknown", []).append(
                (float(expected_roi), float(actual_roi), detected_at)
            )

        def summarize(vals: list[tuple[float, float, datetime]]) -> dict[str, Any]:
            n = len(vals)
            if n == 0:
                return {"sample_size": 0}
            errors = [abs(e - a) for e, a, _ in vals]
            sq_errors = [(e - a) ** 2 for e, a, _ in vals]
            directional_hits = sum(
                1
                for e, a, _ in vals
                if ((e >= 0 and a >= 0) or (e < 0 and a < 0))
            )
            optimism_bias = sum((e - a) for e, a, _ in vals) / n
            expected_mean = sum(e for e, _, _ in vals) / n
            actual_mean = sum(a for _, a, _ in vals) / n
            rmse = math.sqrt(sum(sq_errors) / n)
            return {
                "sample_size": n,
                "expected_roi_mean": round(expected_mean, 4),
                "actual_roi_mean": round(actual_mean, 4),
                "mae_roi": round(sum(errors) / n, 4),
                "rmse_roi": round(rmse, 4),
                "directional_accuracy": round(directional_hits / n, 4),
                "optimism_bias_roi": round(optimism_bias, 4),
            }

        all_vals = [v for arr in by_strategy.values() for v in arr]
        by_strategy_summary = {k: summarize(v) for k, v in by_strategy.items()}

        return {
            "window_days": days,
            "sample_size": len(all_vals),
            "overall": summarize(all_vals),
            "by_strategy": by_strategy_summary,
        }

    async def compute_calibration_trend(
        self, days: int = 90, bucket_days: int = 7
    ) -> list[dict[str, Any]]:
        """Time-bucketed trend of calibration quality."""
        cutoff = utcnow() - timedelta(days=days)
        async with AsyncSessionLocal() as session:
            rows = (
                (
                    await session.execute(
                        select(
                            OpportunityHistory.detected_at,
                            OpportunityHistory.expected_roi,
                            OpportunityHistory.actual_roi,
                        ).where(
                            and_(
                                OpportunityHistory.detected_at >= cutoff,
                                OpportunityHistory.expected_roi.isnot(None),
                                OpportunityHistory.actual_roi.isnot(None),
                            )
                        )
                    )
                )
                .all()
            )

        buckets: dict[str, list[tuple[float, float]]] = {}
        for detected_at, expected_roi, actual_roi in rows:
            if expected_roi is None or actual_roi is None:
                continue
            day = detected_at.date()
            bucket_start = day - timedelta(days=(day.toordinal() % bucket_days))
            key = bucket_start.isoformat()
            buckets.setdefault(key, []).append((float(expected_roi), float(actual_roi)))

        trend = []
        for key in sorted(buckets.keys()):
            vals = buckets[key]
            n = len(vals)
            mae = sum(abs(e - a) for e, a in vals) / n
            acc = sum(1 for e, a in vals if ((e >= 0 and a >= 0) or (e < 0 and a < 0)))
            trend.append(
                {
                    "bucket_start": key,
                    "sample_size": n,
                    "mae_roi": round(mae, 4),
                    "directional_accuracy": round(acc / n, 4),
                }
            )
        return trend

    async def compute_autotrader_execution_metrics(self, days: int = 30) -> dict[str, Any]:
        cutoff = utcnow() - timedelta(days=days)
        async with AsyncSessionLocal() as session:
            rows = (
                (
                    await session.execute(
                        select(
                            AutoTraderTrade.source,
                            TradeSignal.strategy_type,
                            AutoTraderTrade.status,
                            AutoTraderTrade.mode,
                            AutoTraderTrade.notional_usd,
                            AutoTraderTrade.actual_profit,
                            AutoTraderTrade.edge_percent,
                            AutoTraderTrade.confidence,
                        )
                        .join(TradeSignal, AutoTraderTrade.signal_id == TradeSignal.id, isouter=True)
                        .where(AutoTraderTrade.created_at >= cutoff)
                    )
                )
                .all()
            )

        summary = {
            "window_days": int(days),
            "sample_size": len(rows),
            "executed_or_open": 0,
            "failed": 0,
            "resolved": 0,
            "failure_rate": 0.0,
            "avg_edge_percent": 0.0,
            "avg_confidence": 0.0,
            "notional_total_usd": 0.0,
            "realized_pnl_total": 0.0,
            "by_source": {},
            "by_strategy": {},
        }
        if not rows:
            return summary

        edges: list[float] = []
        confidences: list[float] = []
        by_source: dict[str, dict[str, Any]] = {}
        by_strategy: dict[str, dict[str, Any]] = {}

        for source, strategy_type, status, mode, notional, actual_profit, edge, confidence in rows:
            source_key = str(source or "unknown")
            strategy_key = str(strategy_type or "unknown")
            status_key = str(status or "unknown")

            if edge is not None:
                edges.append(float(edge))
            if confidence is not None:
                confidences.append(float(confidence))

            notional_value = float(notional or 0.0)
            pnl_value = float(actual_profit or 0.0)
            is_failed = status_key == "failed"
            is_executed_or_open = status_key in {"submitted", "executed", "open"}
            is_resolved = status_key in {"resolved_win", "resolved_loss"}

            if is_executed_or_open:
                summary["executed_or_open"] += 1
            if is_failed:
                summary["failed"] += 1
            if is_resolved:
                summary["resolved"] += 1
            summary["notional_total_usd"] += notional_value
            summary["realized_pnl_total"] += pnl_value

            source_row = by_source.setdefault(
                source_key,
                {
                    "source": source_key,
                    "total": 0,
                    "failed": 0,
                    "executed_or_open": 0,
                    "resolved": 0,
                    "notional_total_usd": 0.0,
                    "realized_pnl_total": 0.0,
                    "modes": {},
                },
            )
            source_row["total"] += 1
            source_row["failed"] += int(is_failed)
            source_row["executed_or_open"] += int(is_executed_or_open)
            source_row["resolved"] += int(is_resolved)
            source_row["notional_total_usd"] += notional_value
            source_row["realized_pnl_total"] += pnl_value
            mode_key = str(mode or "unknown")
            source_row["modes"][mode_key] = int(source_row["modes"].get(mode_key, 0)) + 1

            strategy_row = by_strategy.setdefault(
                strategy_key,
                {
                    "strategy_type": strategy_key,
                    "total": 0,
                    "failed": 0,
                    "executed_or_open": 0,
                    "resolved": 0,
                    "notional_total_usd": 0.0,
                    "realized_pnl_total": 0.0,
                },
            )
            strategy_row["total"] += 1
            strategy_row["failed"] += int(is_failed)
            strategy_row["executed_or_open"] += int(is_executed_or_open)
            strategy_row["resolved"] += int(is_resolved)
            strategy_row["notional_total_usd"] += notional_value
            strategy_row["realized_pnl_total"] += pnl_value

        sample_size = int(summary["sample_size"])
        summary["failure_rate"] = (
            float(summary["failed"]) / sample_size if sample_size > 0 else 0.0
        )
        summary["avg_edge_percent"] = (
            sum(edges) / len(edges) if edges else 0.0
        )
        summary["avg_confidence"] = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )
        summary["by_source"] = sorted(
            by_source.values(),
            key=lambda row: (
                float(row["notional_total_usd"]),
                int(row["total"]),
            ),
            reverse=True,
        )
        summary["by_strategy"] = sorted(
            by_strategy.values(),
            key=lambda row: (
                float(row["notional_total_usd"]),
                int(row["total"]),
            ),
            reverse=True,
        )
        return summary

    async def compute_world_intel_resolver_metrics(
        self,
        days: int = 7,
        max_signals: int = 500,
    ) -> dict[str, Any]:
        cutoff = utcnow() - timedelta(days=days)
        async with AsyncSessionLocal() as session:
            signals = (
                (
                    await session.execute(
                        select(WorldIntelligenceSignal)
                        .where(WorldIntelligenceSignal.detected_at >= cutoff)
                        .order_by(WorldIntelligenceSignal.detected_at.desc())
                        .limit(max(1, min(max_signals, 5000)))
                    )
                )
                .scalars()
                .all()
            )

            if not signals:
                return {
                    "window_days": int(days),
                    "signals_sampled": 0,
                    "candidates": 0,
                    "tradable": 0,
                    "tradable_rate": 0.0,
                    "by_signal_type": {},
                }

            from services.world_intelligence.resolver import resolve_world_signal_opportunities

            candidates = await resolve_world_signal_opportunities(
                session,
                signals,
                max_markets_per_signal=5,
            )

        tradable = [row for row in candidates if bool(row.get("tradable"))]
        by_type: dict[str, dict[str, Any]] = {}
        for row in candidates:
            key = str(row.get("signal_type") or "unknown")
            entry = by_type.setdefault(
                key,
                {"signal_type": key, "candidates": 0, "tradable": 0, "tradable_rate": 0.0},
            )
            entry["candidates"] += 1
            if bool(row.get("tradable")):
                entry["tradable"] += 1
        for row in by_type.values():
            total = int(row["candidates"] or 0)
            row["tradable_rate"] = (
                float(row["tradable"]) / total if total > 0 else 0.0
            )

        total_candidates = len(candidates)
        return {
            "window_days": int(days),
            "signals_sampled": len(signals),
            "candidates": total_candidates,
            "tradable": len(tradable),
            "tradable_rate": (
                len(tradable) / total_candidates if total_candidates > 0 else 0.0
            ),
            "by_signal_type": sorted(
                by_type.values(),
                key=lambda row: int(row["candidates"]),
                reverse=True,
            ),
        }

    async def evaluate_guardrails(self) -> dict[str, Any]:
        cfg = await self.get_guardrail_config()
        if not cfg["enabled"]:
            return {"enabled": False, "updated": 0, "demoted": [], "restored": []}

        metrics = await self.compute_calibration_metrics(days=cfg["lookback_days"])
        by_strategy = metrics.get("by_strategy", {})
        demoted: list[str] = []
        restored: list[str] = []
        updated = 0

        async with AsyncSessionLocal() as session:
            for strategy_type, m in by_strategy.items():
                sample_size = int(m.get("sample_size") or 0)
                directional_accuracy = m.get("directional_accuracy")
                mae_roi = m.get("mae_roi")
                rmse_roi = m.get("rmse_roi")
                optimism_bias = m.get("optimism_bias_roi")

                row = await session.get(StrategyValidationProfile, strategy_type)
                if row is None:
                    row = StrategyValidationProfile(strategy_type=strategy_type)
                    session.add(row)

                row.sample_size = sample_size
                row.directional_accuracy = directional_accuracy
                row.mae_roi = mae_roi
                row.rmse_roi = rmse_roi
                row.optimism_bias_roi = optimism_bias

                # Respect manual override lock.
                if row.manual_override:
                    row.updated_at = utcnow()
                    updated += 1
                    continue

                weak_accuracy = (
                    directional_accuracy is not None
                    and directional_accuracy < cfg["min_directional_accuracy"]
                )
                weak_error = mae_roi is not None and mae_roi > cfg["max_mae_roi"]
                enough_samples = sample_size >= cfg["min_samples"]

                now = utcnow()
                if enough_samples and (weak_accuracy or weak_error):
                    if row.status != "demoted":
                        row.status = "demoted"
                        row.demoted_at = now
                        if weak_accuracy and weak_error:
                            row.last_reason = (
                                f"Low directional accuracy ({directional_accuracy:.3f}) "
                                f"and high MAE ({mae_roi:.2f})"
                            )
                        elif weak_accuracy:
                            row.last_reason = (
                                f"Low directional accuracy ({directional_accuracy:.3f})"
                            )
                        else:
                            row.last_reason = f"High MAE ({mae_roi:.2f})"
                        demoted.append(strategy_type)
                elif cfg["auto_promote"]:
                    promoted_ok = (
                        enough_samples
                        and directional_accuracy is not None
                        and directional_accuracy
                        >= cfg["min_directional_accuracy"] + 0.03
                        and mae_roi is not None
                        and mae_roi <= cfg["max_mae_roi"] * 0.9
                    )
                    if row.status == "demoted" and promoted_ok:
                        row.status = "active"
                        row.restored_at = now
                        row.last_reason = "Auto-restored after sustained calibration recovery"
                        restored.append(strategy_type)
                row.updated_at = now
                updated += 1
            await session.commit()

        return {
            "enabled": True,
            "updated": updated,
            "demoted": demoted,
            "restored": restored,
            "config": cfg,
        }

    async def get_strategy_health(self) -> list[dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            rows = (
                (
                    await session.execute(
                        select(StrategyValidationProfile).order_by(
                            StrategyValidationProfile.strategy_type.asc()
                        )
                    )
                )
                .scalars()
                .all()
            )
        return [
            {
                "strategy_type": r.strategy_type,
                "status": r.status,
                "sample_size": r.sample_size,
                "directional_accuracy": r.directional_accuracy,
                "mae_roi": r.mae_roi,
                "rmse_roi": r.rmse_roi,
                "optimism_bias_roi": r.optimism_bias_roi,
                "last_reason": r.last_reason,
                "manual_override": bool(r.manual_override),
                "manual_override_note": r.manual_override_note,
                "demoted_at": r.demoted_at.isoformat() if r.demoted_at else None,
                "restored_at": r.restored_at.isoformat() if r.restored_at else None,
                "updated_at": r.updated_at.isoformat() if r.updated_at else None,
            }
            for r in rows
        ]

    async def set_strategy_override(
        self,
        strategy_type: str,
        status: str,
        manual_override: bool = True,
        note: Optional[str] = None,
    ) -> dict[str, Any]:
        async with AsyncSessionLocal() as session:
            row = await session.get(StrategyValidationProfile, strategy_type)
            if row is None:
                row = StrategyValidationProfile(strategy_type=strategy_type)
                session.add(row)
            row.status = status
            row.manual_override = manual_override
            row.manual_override_note = note
            row.last_reason = note or row.last_reason
            row.updated_at = utcnow()
            await session.commit()
        return {"strategy_type": strategy_type, "status": status}

    async def clear_strategy_override(self, strategy_type: str) -> dict[str, Any]:
        async with AsyncSessionLocal() as session:
            row = await session.get(StrategyValidationProfile, strategy_type)
            if row is None:
                row = StrategyValidationProfile(strategy_type=strategy_type)
                session.add(row)
            row.manual_override = False
            row.manual_override_note = None
            row.updated_at = utcnow()
            await session.commit()
        return {"strategy_type": strategy_type, "manual_override": False}

    async def get_demoted_strategy_types(self) -> set[str]:
        async with AsyncSessionLocal() as session:
            rows = (
                (
                    await session.execute(
                        select(StrategyValidationProfile.strategy_type).where(
                            StrategyValidationProfile.status == "demoted"
                        )
                    )
                )
                .all()
            )
        return {r[0] for r in rows}

    async def _recover_incomplete_jobs(self) -> None:
        async with AsyncSessionLocal() as session:
            rows = (
                (
                    await session.execute(
                        select(ValidationJob).where(ValidationJob.status == "running")
                    )
                )
                .scalars()
                .all()
            )
            for row in rows:
                row.status = "queued"
                row.progress = 0.0
                row.message = "Recovered after restart"
                row.started_at = None
                row.finished_at = None
            await session.commit()

    async def _worker_loop(self) -> None:
        while self._running:
            try:
                job = await self._claim_next_job()
                if not job:
                    await asyncio.sleep(1.0)
                    continue
                await self._run_job(job.id)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.error("Validation worker loop error", exc_info=True)
                await asyncio.sleep(1.0)

    async def _guardrail_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(900)  # 15 min
                if not self._running:
                    break
                await self.evaluate_guardrails()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.error("Guardrail loop error", exc_info=True)

    async def _claim_next_job(self) -> Optional[ValidationJob]:
        async with AsyncSessionLocal() as session:
            row = (
                (
                    await session.execute(
                        select(ValidationJob)
                        .where(ValidationJob.status == "queued")
                        .order_by(ValidationJob.created_at.asc())
                        .limit(1)
                    )
                )
                .scalars()
                .first()
            )
            if row is None:
                return None
            row.status = "running"
            row.started_at = utcnow()
            row.message = "Running"
            row.progress = 0.01
            await session.commit()
            return row

    async def _set_job_progress(
        self, job_id: str, progress: float, message: Optional[str] = None
    ) -> bool:
        async with AsyncSessionLocal() as session:
            row = await session.get(ValidationJob, job_id)
            if row is None:
                return False
            if row.status == "cancelled":
                return False
            row.progress = max(0.0, min(progress, 1.0))
            if message is not None:
                row.message = message
            await session.commit()
            return True

    async def _run_job(self, job_id: str) -> None:
        async with AsyncSessionLocal() as session:
            row = await session.get(ValidationJob, job_id)
            if row is None:
                return
            payload = row.payload or {}
            job_type = row.job_type

        try:
            if job_type == "backtest":
                await self._set_job_progress(job_id, 0.1, "Loading history")
                result = await param_optimizer.run_backtest(params=payload.get("params"))
                await self._set_job_progress(job_id, 0.8, "Persisting results")

                saved_parameter_set_id = None
                if payload.get("save_parameter_set"):
                    params_to_save = payload.get("params") or param_optimizer.get_current_params()
                    saved_parameter_set_id = await param_optimizer.save_parameter_set(
                        name=payload.get("parameter_set_name")
                        or f"Backtest {utcnow().isoformat(timespec='seconds')}",
                        params=params_to_save,
                        backtest_results=result,
                        is_active=bool(payload.get("activate_saved_set")),
                    )
                    if payload.get("activate_saved_set"):
                        param_optimizer.set_params(params_to_save)

                final_result = {
                    "status": "success",
                    "result": result,
                    "saved_parameter_set_id": saved_parameter_set_id,
                }
            elif job_type == "optimize":
                await self._set_job_progress(job_id, 0.1, "Preparing optimization")
                async def _progress(i: int, t: int):
                    ok = await self._set_job_progress(
                        job_id,
                        0.1 + 0.7 * (i / max(t, 1)),
                        f"Evaluated {i}/{t} candidates",
                    )
                    if not ok:
                        raise RuntimeError("Job cancelled")

                results = await param_optimizer.run_optimization(
                    method=payload.get("method", "grid"),
                    param_ranges=payload.get("param_ranges"),
                    n_random_samples=int(payload.get("n_random_samples", 100)),
                    random_seed=int(payload.get("random_seed", 42)),
                    walk_forward=bool(payload.get("walk_forward", True)),
                    n_windows=int(payload.get("n_windows", 5)),
                    train_ratio=float(payload.get("train_ratio", 0.7)),
                    progress_hook=_progress,
                )
                top_k = int(payload.get("top_k", 20))
                top = results[: max(1, min(top_k, 200))]
                saved_set_id = None
                if payload.get("save_best_as_active") and top:
                    best = top[0]
                    best_params = best.get("params", {})
                    saved_set_id = await param_optimizer.save_parameter_set(
                        name=payload.get("best_set_name")
                        or f"Optimized Best {utcnow().isoformat(timespec='seconds')}",
                        params=best_params,
                        backtest_results=best.get("test_result"),
                        is_active=True,
                    )
                    param_optimizer.set_params(best_params)

                final_result = {
                    "status": "success",
                    "candidate_count": len(results),
                    "top_results": top,
                    "best_saved_parameter_set_id": saved_set_id,
                }
            else:
                raise ValueError(f"Unsupported job type: {job_type}")

            await self.evaluate_guardrails()
            async with AsyncSessionLocal() as session:
                row = await session.get(ValidationJob, job_id)
                if row is None:
                    return
                if row.status == "cancelled":
                    return
                row.status = "completed"
                row.result = final_result
                row.progress = 1.0
                row.message = "Completed"
                row.finished_at = utcnow()
                await session.commit()
        except Exception as e:
            async with AsyncSessionLocal() as session:
                row = await session.get(ValidationJob, job_id)
                if row is None:
                    return
                if str(e) == "Job cancelled":
                    row.status = "cancelled"
                    row.error = None
                    row.progress = 1.0
                    row.message = "Cancelled"
                else:
                    row.status = "failed"
                    row.error = str(e)
                    row.progress = 1.0
                    row.message = "Failed"
                row.finished_at = utcnow()
                await session.commit()
            if str(e) != "Job cancelled":
                logger.error("Validation job failed", job_id=job_id, error=str(e))

    async def _get_or_create_settings(self) -> AppSettings:
        async with AsyncSessionLocal() as session:
            row = await session.get(AppSettings, "default")
            if row is None:
                row = AppSettings(id="default")
                session.add(row)
                await session.commit()
                await session.refresh(row)
            return row

    @staticmethod
    def _job_to_dict(row: ValidationJob) -> dict[str, Any]:
        return {
            "id": row.id,
            "job_type": row.job_type,
            "status": row.status,
            "payload": row.payload,
            "result": row.result,
            "error": row.error,
            "progress": row.progress,
            "message": row.message,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "finished_at": row.finished_at.isoformat() if row.finished_at else None,
        }


validation_service = ValidationService()
