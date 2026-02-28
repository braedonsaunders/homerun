"""Scanner SLO worker: evaluate freshness/coverage SLOs and trigger actions."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any

from config import apply_runtime_settings_overrides, settings
from models.database import AsyncSessionLocal
from services.event_bus import event_bus
from services.shared_state import (
    list_open_scanner_slo_incidents,
    read_scanner_control,
    read_scanner_status,
    set_scanner_heavy_lane_degraded,
    upsert_scanner_slo_incident,
)
from services.worker_state import clear_worker_run_request, read_worker_control, write_worker_snapshot
from utils.logger import setup_logging
from utils.utcnow import utcnow

setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"), json_format=False)
logger = logging.getLogger("scanner_slo_worker")


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


async def _run_loop() -> None:
    worker_name = "scanner_slo"
    heartbeat_interval = max(
        1.0,
        float(getattr(settings, "SCANNER_HEARTBEAT_INTERVAL_SECONDS", 5.0) or 5.0),
    )
    default_interval = max(
        1,
        int(getattr(settings, "SCANNER_SLO_WORKER_INTERVAL_SECONDS", 5) or 5),
    )

    state: dict[str, Any] = {
        "enabled": True,
        "interval_seconds": default_interval,
        "activity": "Scanner SLO worker started; awaiting first evaluation.",
        "last_error": None,
        "last_run_at": None,
        "run_id": None,
        "phase": "idle",
        "progress": 0.0,
        "open_incidents": 0,
        "breached_metrics": 0,
        "auto_degrade_enabled": bool(getattr(settings, "SCANNER_SLO_DEGRADE_HEAVY_ENABLED", True)),
        "heavy_lane_forced_degraded": False,
    }
    heartbeat_stop_event = asyncio.Event()

    async def _heartbeat_loop() -> None:
        while not heartbeat_stop_event.is_set():
            try:
                async with AsyncSessionLocal() as session:
                    await write_worker_snapshot(
                        session,
                        worker_name,
                        running=True,
                        enabled=bool(state.get("enabled", True)),
                        current_activity=str(state.get("activity") or "Idle"),
                        interval_seconds=int(state.get("interval_seconds") or default_interval),
                        last_run_at=state.get("last_run_at"),
                        last_error=(str(state["last_error"]) if state.get("last_error") is not None else None),
                        stats={
                            "run_id": state.get("run_id"),
                            "phase": state.get("phase"),
                            "progress": float(state.get("progress", 0.0) or 0.0),
                            "open_incidents": int(state.get("open_incidents", 0) or 0),
                            "breached_metrics": int(state.get("breached_metrics", 0) or 0),
                            "auto_degrade_enabled": bool(state.get("auto_degrade_enabled", False)),
                            "heavy_lane_forced_degraded": bool(
                                state.get("heavy_lane_forced_degraded", False)
                            ),
                        },
                    )
            except Exception as exc:
                state["last_error"] = str(exc)
                logger.warning("Scanner SLO heartbeat snapshot write failed: %s", exc)
            try:
                await asyncio.wait_for(heartbeat_stop_event.wait(), timeout=heartbeat_interval)
            except asyncio.TimeoutError:
                continue

    heartbeat_task = asyncio.create_task(_heartbeat_loop(), name="scanner-slo-heartbeat")
    logger.info("Scanner SLO worker started")

    try:
        while True:
            try:
                await apply_runtime_settings_overrides()
            except Exception as exc:
                logger.warning("Scanner SLO runtime settings refresh failed: %s", exc)

            async with AsyncSessionLocal() as session:
                control = await read_worker_control(
                    session,
                    worker_name,
                    default_interval=default_interval,
                )

            interval_seconds = max(1, int(control.get("interval_seconds") or default_interval))
            paused = bool(control.get("is_paused", False))
            requested = control.get("requested_run_at") is not None
            enabled = bool(control.get("is_enabled", True)) and not paused

            state["enabled"] = enabled
            state["interval_seconds"] = interval_seconds
            state["run_id"] = uuid.uuid4().hex[:16]

            threshold_fast_age = max(
                1.0,
                float(getattr(settings, "SCANNER_SLO_MAX_FAST_SCAN_AGE_SECONDS", 120.0) or 120.0),
            )
            threshold_full_coverage_completion = max(
                1.0,
                float(getattr(settings, "SCANNER_SLO_MAX_FULL_COVERAGE_COMPLETION_SECONDS", 900.0) or 900.0),
            )
            threshold_price_age_p95 = max(
                1.0,
                float(getattr(settings, "SCANNER_SLO_MAX_OPPORTUNITY_PRICE_AGE_P95_SECONDS", 60.0) or 60.0),
            )
            threshold_detected_age_p95 = max(
                1.0,
                float(getattr(settings, "SCANNER_SLO_MAX_OPPORTUNITY_LAST_DETECTED_AGE_P95_SECONDS", 180.0) or 180.0),
            )
            threshold_coverage_ratio_min = max(
                0.0,
                min(1.0, float(getattr(settings, "SCANNER_SLO_MIN_COVERAGE_RATIO", 0.95) or 0.95)),
            )
            auto_degrade_enabled = bool(getattr(settings, "SCANNER_SLO_DEGRADE_HEAVY_ENABLED", True))
            auto_degrade_duration_seconds = max(
                10,
                int(getattr(settings, "SCANNER_SLO_DEGRADE_HEAVY_DURATION_SECONDS", 180) or 180),
            )
            state["auto_degrade_enabled"] = auto_degrade_enabled

            if not enabled and not requested:
                state["phase"] = "idle"
                state["progress"] = 0.0
                state["activity"] = "Paused"
                await asyncio.sleep(min(heartbeat_interval, float(interval_seconds)))
                continue

            state["phase"] = "evaluate"
            state["progress"] = 0.2
            state["activity"] = "Evaluating scanner freshness and coverage SLOs..."

            try:
                async with AsyncSessionLocal() as session:
                    status = await read_scanner_status(
                        session,
                        include_opportunity_count=False,
                        include_slo_metrics=True,
                    )

                tiered = status.get("tiered_scanning") if isinstance(status.get("tiered_scanning"), dict) else {}
                metrics = {
                    "last_fast_scan_age_seconds": _as_float(status.get("last_fast_scan_age_seconds")),
                    "full_coverage_completion_time": _as_float(
                        status.get("full_coverage_completion_time")
                        or tiered.get("full_coverage_completion_time")
                    ),
                    "opportunity_price_age_p95": _as_float(status.get("opportunity_price_age_p95")),
                    "opportunity_last_detected_age_p95": _as_float(
                        status.get("opportunity_last_detected_age_p95")
                    ),
                    "coverage_ratio": _as_float(status.get("coverage_ratio") or tiered.get("full_snapshot_coverage_ratio")),
                }

                checks = {
                    "last_fast_scan_age_seconds": {
                        "threshold": threshold_fast_age,
                        "severity": "critical",
                        "breached": (
                            metrics["last_fast_scan_age_seconds"] is not None
                            and metrics["last_fast_scan_age_seconds"] > threshold_fast_age
                        ),
                    },
                    "full_coverage_completion_time": {
                        "threshold": threshold_full_coverage_completion,
                        "severity": "warning",
                        "breached": (
                            metrics["full_coverage_completion_time"] is not None
                            and metrics["full_coverage_completion_time"] > threshold_full_coverage_completion
                        ),
                    },
                    "opportunity_price_age_p95": {
                        "threshold": threshold_price_age_p95,
                        "severity": "critical",
                        "breached": (
                            metrics["opportunity_price_age_p95"] is not None
                            and metrics["opportunity_price_age_p95"] > threshold_price_age_p95
                        ),
                    },
                    "opportunity_last_detected_age_p95": {
                        "threshold": threshold_detected_age_p95,
                        "severity": "warning",
                        "breached": (
                            metrics["opportunity_last_detected_age_p95"] is not None
                            and metrics["opportunity_last_detected_age_p95"] > threshold_detected_age_p95
                        ),
                    },
                    "coverage_ratio": {
                        "threshold": threshold_coverage_ratio_min,
                        "severity": "critical",
                        "breached": (
                            metrics["coverage_ratio"] is not None
                            and metrics["coverage_ratio"] < threshold_coverage_ratio_min
                        ),
                    },
                }

                incident_actions: list[dict[str, Any]] = []
                breached_metric_names: list[str] = []
                degrade_trigger_metrics = {"coverage_ratio", "full_coverage_completion_time"}
                alert_payload: dict[str, Any] | None = None

                async with AsyncSessionLocal() as session:
                    for metric_name, check in checks.items():
                        breached = bool(check.get("breached", False))
                        if breached:
                            breached_metric_names.append(metric_name)
                        action = await upsert_scanner_slo_incident(
                            session,
                            metric=metric_name,
                            breached=breached,
                            observed_value=metrics.get(metric_name),
                            threshold_value=check.get("threshold"),
                            severity=str(check.get("severity") or "warning"),
                            details={
                                "metric": metric_name,
                                "value": metrics.get(metric_name),
                                "threshold": check.get("threshold"),
                                "scanner_running": bool(status.get("running", False)),
                                "scanner_enabled": bool(status.get("enabled", False)),
                            },
                        )
                        if action.get("action") in {"opened", "resolved"}:
                            incident_actions.append(action)

                    scanner_control = await read_scanner_control(session)
                    forced_degraded = bool(scanner_control.get("heavy_lane_forced_degraded", False))
                    forced_reason = str(scanner_control.get("heavy_lane_degraded_reason") or "")
                    breached_for_degrade = [
                        metric for metric in breached_metric_names if metric in degrade_trigger_metrics
                    ]

                    degrade_action = "none"
                    if auto_degrade_enabled and breached_for_degrade and (
                        not forced_degraded or not forced_reason.startswith("slo_breach:")
                    ):
                        reason = "slo_breach:" + ",".join(sorted(breached_for_degrade)[:5])
                        await set_scanner_heavy_lane_degraded(
                            session,
                            enabled=True,
                            reason=reason,
                            duration_seconds=auto_degrade_duration_seconds,
                        )
                        degrade_action = "enabled"
                        forced_degraded = True
                    elif auto_degrade_enabled and forced_degraded and forced_reason.startswith("slo_breach:"):
                        await set_scanner_heavy_lane_degraded(session, enabled=False)
                        degrade_action = "disabled"
                        forced_degraded = False

                    open_incidents = await list_open_scanner_slo_incidents(session)
                    state["open_incidents"] = len(open_incidents)
                    state["breached_metrics"] = len(breached_metric_names)
                    state["heavy_lane_forced_degraded"] = bool(forced_degraded)

                    should_publish = bool(incident_actions) or degrade_action in {"enabled", "disabled"}
                    if should_publish:
                        alert_payload = {
                            "run_id": state.get("run_id"),
                            "at": utcnow().isoformat(),
                            "breached_metrics": sorted(breached_metric_names),
                            "breached_for_degrade": sorted(breached_for_degrade),
                            "open_incidents": open_incidents,
                            "incident_actions": incident_actions,
                            "degrade_action": degrade_action,
                            "heavy_lane_forced_degraded": bool(forced_degraded),
                        }

                if alert_payload is not None:
                    await event_bus.publish("scanner_slo_alert", alert_payload)

                state["last_error"] = None
                state["last_run_at"] = utcnow()
                state["phase"] = "idle"
                state["progress"] = 1.0
                state["activity"] = (
                    f"Scanner SLO evaluated: {state['breached_metrics']} breached metrics, "
                    f"{state['open_incidents']} open incidents."
                )

                async with AsyncSessionLocal() as session:
                    await clear_worker_run_request(session, worker_name)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                state["last_error"] = str(exc)
                state["phase"] = "error"
                state["progress"] = 1.0
                state["activity"] = f"Scanner SLO evaluation error: {exc}"
                logger.exception("Scanner SLO evaluation cycle failed")

            await asyncio.sleep(float(interval_seconds))
    finally:
        heartbeat_stop_event.set()
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass


async def start_loop() -> None:
    try:
        await _run_loop()
    except asyncio.CancelledError:
        logger.info("Scanner SLO worker shutting down")
