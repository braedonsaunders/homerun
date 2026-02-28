"""Validation orchestration service.

Responsibilities:
- Persistent async queue for backtest/optimization jobs
- Guardrail evaluation for strategy demotion/promotion
- Shared validation analytics used by API and scanner
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
import re
import sys
import uuid
from datetime import datetime, timedelta, timezone
from utils.utcnow import utcnow
from typing import Any, Optional

from sqlalchemy import and_, func, select

from models.database import (
    AppSettings,
    AsyncSessionLocal,
    ExecutionSimEvent,
    ExecutionSimRun,
    OpportunityHistory,
    Strategy,
    StrategyValidationProfile,
    TradeSignal,
    Trader,
    TraderOrder,
    ValidationJob,
    EventsSignal,
)
from services.param_optimizer import param_optimizer
from services.simulation.execution_simulator import execution_simulator
from utils.logger import get_logger

logger = get_logger(__name__)

_LIVE_TRUTH_MONITOR_STDOUT_EVENT_LIMIT = 600
_LIVE_TRUTH_MONITOR_DEFAULT_MAX_ALERTS = 200
_LIVE_TRUTH_MONITOR_MAX_EXPORT_ALERTS = 10000
_LIVE_TRUTH_MONITOR_MIN_DURATION_SECONDS = 10
_LIVE_TRUTH_MONITOR_DEFAULT_DURATION_SECONDS = 300
_LIVE_TRUTH_MONITOR_MAX_DURATION_SECONDS = 7200
_LIVE_TRUTH_MONITOR_MIN_POLL_SECONDS = 0.2
_LIVE_TRUTH_MONITOR_DEFAULT_POLL_SECONDS = 1.0
_LIVE_TRUTH_MONITOR_MAX_POLL_SECONDS = 10.0
_LIVE_TRUTH_MONITOR_ALERTS_FOR_LLM_DEFAULT = 80
_LIVE_TRUTH_MONITOR_ALERTS_FOR_LLM_MAX = 400
_LIVE_TRUTH_MONITOR_LLM_TIMEOUT_SECONDS = 120.0
_LIVE_TRUTH_MONITOR_STDERR_TAIL_CHARS = 4000
_LIVE_TRUTH_MONITOR_STRATEGY_SOURCE_MAX_CHARS = 24000
_LIVE_TRUTH_MONITOR_SUMMARY_ARTIFACT = "summary_json"
_LIVE_TRUTH_MONITOR_REPORT_ARTIFACT = "report_jsonl"
_LIVE_TRUTH_MONITOR_LLM_ARTIFACT = "llm_analysis_json"
_LIVE_TRUTH_MONITOR_BUNDLE_ARTIFACT = "bundle_json"
_LIVE_TRUTH_MONITOR_ARTIFACTS = {
    _LIVE_TRUTH_MONITOR_SUMMARY_ARTIFACT,
    _LIVE_TRUTH_MONITOR_REPORT_ARTIFACT,
    _LIVE_TRUTH_MONITOR_LLM_ARTIFACT,
    _LIVE_TRUTH_MONITOR_BUNDLE_ARTIFACT,
}
_LIVE_TRUTH_MONITOR_LLM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "assessment",
        "priority",
        "root_causes",
        "strategy_changes",
        "runtime_changes",
        "tests_to_run",
    ],
    "properties": {
        "assessment": {"type": "string"},
        "priority": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
        "root_causes": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
        "strategy_changes": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "rationale", "target", "suggested_change"],
                "properties": {
                    "title": {"type": "string"},
                    "rationale": {"type": "string"},
                    "target": {"type": "string"},
                    "suggested_change": {"type": "string"},
                    "expected_impact": {"type": "string"},
                },
            },
        },
        "runtime_changes": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
        "tests_to_run": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    except Exception:
        return None


def _clamp_int(value: Any, default: int, lo: int, hi: int) -> int:
    try:
        parsed = int(float(value))
    except Exception:
        parsed = default
    return max(lo, min(hi, parsed))


def _clamp_float(value: Any, default: float, lo: float, hi: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        parsed = default
    return max(lo, min(hi, parsed))


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _monitor_output_root() -> Path:
    return (_repo_root() / "output").resolve()


def _resolve_monitor_output_path(raw: Any) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = (_repo_root() / candidate).resolve()
    else:
        candidate = candidate.resolve()
    output_root = _monitor_output_root()
    if candidate != output_root and output_root not in candidate.parents:
        return None
    return candidate


def _extract_json_payload(raw_text: str) -> dict[str, Any] | None:
    text = str(raw_text or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    fenced = re.search(r"```json\s*(\{.*\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
    return None


def _read_json_file(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists() or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _read_live_truth_report(path: Path | None, *, max_alerts: int) -> dict[str, Any]:
    report = {
        "path": str(path) if path is not None else "",
        "line_count": 0,
        "alert_count": 0,
        "heartbeat_count": 0,
        "transition_count": 0,
        "alerts_by_rule": {},
        "alert_samples": [],
    }
    if path is None or not path.exists() or not path.is_file():
        return report

    safe_alert_limit = _clamp_int(
        max_alerts, _LIVE_TRUTH_MONITOR_DEFAULT_MAX_ALERTS, 1, _LIVE_TRUTH_MONITOR_MAX_EXPORT_ALERTS
    )
    alerts_by_rule: dict[str, int] = {}
    alert_samples: list[dict[str, Any]] = []
    heartbeat_count = 0
    transition_count = 0
    line_count = 0
    alert_count = 0

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line_count += 1
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue

            event_name = str(payload.get("event") or "").strip().lower()
            if event_name == "heartbeat":
                heartbeat_count += 1
                continue
            if event_name == "transition":
                transition_count += 1
                continue

            rule = str(payload.get("rule") or "").strip()
            verdict = str(payload.get("lifecycle_verdict") or "").strip()
            if rule and verdict:
                alert_count += 1
                alerts_by_rule[rule] = alerts_by_rule.get(rule, 0) + 1
                if len(alert_samples) < safe_alert_limit:
                    alert_samples.append(payload)

    report["line_count"] = line_count
    report["alert_count"] = alert_count
    report["heartbeat_count"] = heartbeat_count
    report["transition_count"] = transition_count
    report["alerts_by_rule"] = alerts_by_rule
    report["alert_samples"] = alert_samples
    return report


def _compact_live_truth_alert_for_llm(raw: dict[str, Any]) -> dict[str, Any]:
    provider = dict(raw.get("provider") or {})
    return {
        "timestamp_utc": str(raw.get("timestamp_utc") or ""),
        "rule": str(raw.get("rule") or ""),
        "lifecycle_verdict": str(raw.get("lifecycle_verdict") or ""),
        "order_id": str(raw.get("order_id") or ""),
        "signal_id": str(raw.get("signal_id") or ""),
        "local_status_reason": str(raw.get("local_status_reason") or ""),
        "provider_status": str(provider.get("status") or ""),
        "provider_clob_order_id": str(provider.get("clob_order_id") or ""),
        "root_cause": str(raw.get("root_cause") or ""),
        "required_fix": str(raw.get("required_fix") or ""),
    }


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
                        select(ValidationJob).order_by(ValidationJob.created_at.desc()).limit(max(1, min(limit, 500)))
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

    async def get_live_truth_monitor_raw(
        self,
        job_id: str,
        *,
        max_alerts: int = _LIVE_TRUTH_MONITOR_MAX_EXPORT_ALERTS,
    ) -> Optional[dict[str, Any]]:
        safe_max_alerts = _clamp_int(
            max_alerts, _LIVE_TRUTH_MONITOR_MAX_EXPORT_ALERTS, 1, _LIVE_TRUTH_MONITOR_MAX_EXPORT_ALERTS
        )
        async with AsyncSessionLocal() as session:
            row = await session.get(ValidationJob, job_id)
            if row is None:
                return None
            if str(row.job_type or "").strip().lower() != "live_truth_monitor":
                return None
            payload = dict(row.payload or {})
            result = dict(row.result or {})

        monitor = dict(result.get("monitor") or {})
        summary_path = _resolve_monitor_output_path(monitor.get("summary_path"))
        report_path = _resolve_monitor_output_path(monitor.get("report_path"))
        summary_json = _read_json_file(summary_path)
        if report_path is None:
            report_path = _resolve_monitor_output_path(summary_json.get("report_path"))
        report_json = _read_live_truth_report(report_path, max_alerts=safe_max_alerts)

        return {
            "job_id": job_id,
            "job_status": str(row.status),
            "payload": payload,
            "monitor": {
                "summary": summary_json if summary_json else dict(monitor.get("summary") or {}),
                "summary_path": str(summary_path)
                if summary_path is not None
                else str(monitor.get("summary_path") or ""),
                "report_path": str(report_path) if report_path is not None else str(monitor.get("report_path") or ""),
                "stdout_events": list(monitor.get("stdout_events") or []),
                "report": report_json,
            },
            "llm_analysis": dict(result.get("llm_analysis") or {}),
        }

    async def export_live_truth_monitor_artifact(
        self,
        job_id: str,
        *,
        artifact: str = _LIVE_TRUTH_MONITOR_BUNDLE_ARTIFACT,
    ) -> Optional[tuple[str, str, bytes]]:
        selected_artifact = str(artifact or "").strip().lower()
        if selected_artifact not in _LIVE_TRUTH_MONITOR_ARTIFACTS:
            return None

        async with AsyncSessionLocal() as session:
            row = await session.get(ValidationJob, job_id)
            if row is None:
                return None
            if str(row.job_type or "").strip().lower() != "live_truth_monitor":
                return None
            payload = dict(row.payload or {})
            result = dict(row.result or {})

        monitor = dict(result.get("monitor") or {})
        summary_path = _resolve_monitor_output_path(monitor.get("summary_path"))
        report_path = _resolve_monitor_output_path(monitor.get("report_path"))
        summary_json = _read_json_file(summary_path)
        if report_path is None:
            report_path = _resolve_monitor_output_path(summary_json.get("report_path"))
        llm_analysis = dict(result.get("llm_analysis") or {})

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = f"live_truth_monitor_{job_id[:8]}_{stamp}"

        if selected_artifact == _LIVE_TRUTH_MONITOR_SUMMARY_ARTIFACT:
            body: bytes
            if summary_path is not None and summary_path.exists() and summary_path.is_file():
                body = summary_path.read_bytes()
            else:
                body = json.dumps(summary_json, indent=2, ensure_ascii=True).encode("utf-8")
            return (f"{base}_summary.json", "application/json", body)

        if selected_artifact == _LIVE_TRUTH_MONITOR_REPORT_ARTIFACT:
            if report_path is None or not report_path.exists() or not report_path.is_file():
                return None
            return (f"{base}_report.jsonl", "application/x-ndjson", report_path.read_bytes())

        if selected_artifact == _LIVE_TRUTH_MONITOR_LLM_ARTIFACT:
            body = json.dumps(llm_analysis, indent=2, ensure_ascii=True).encode("utf-8")
            return (f"{base}_llm_analysis.json", "application/json", body)

        report_json = _read_live_truth_report(report_path, max_alerts=_LIVE_TRUTH_MONITOR_MAX_EXPORT_ALERTS)
        bundle = {
            "job_id": job_id,
            "job_type": str(row.job_type),
            "job_status": str(row.status),
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "finished_at": row.finished_at.isoformat() if row.finished_at else None,
            "payload": payload,
            "monitor": {
                "summary": summary_json if summary_json else dict(monitor.get("summary") or {}),
                "summary_path": str(summary_path)
                if summary_path is not None
                else str(monitor.get("summary_path") or ""),
                "report_path": str(report_path) if report_path is not None else str(monitor.get("report_path") or ""),
                "stdout_events": list(monitor.get("stdout_events") or []),
                "report": report_json,
            },
            "llm_analysis": llm_analysis,
        }
        body = json.dumps(bundle, indent=2, ensure_ascii=True).encode("utf-8")
        return (f"{base}_bundle.json", "application/json", body)

    @staticmethod
    async def _terminate_subprocess(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

    async def _load_live_truth_strategy_context(
        self,
        *,
        trader_id: str,
        strategy_key_hint: str,
        include_strategy_source: bool,
    ) -> dict[str, Any]:
        trader_key = str(trader_id or "").strip()
        strategy_key = str(strategy_key_hint or "").strip().lower()
        context: dict[str, Any] = {
            "trader_id": trader_key,
            "trader_name": "",
            "strategy_key": strategy_key,
            "strategy_id": "",
            "strategy_name": "",
            "strategy_source": "",
            "strategy_source_included": False,
        }

        async with AsyncSessionLocal() as session:
            trader_row = await session.get(Trader, trader_key) if trader_key else None
            if trader_row is not None:
                context["trader_id"] = str(trader_row.id or "")
                context["trader_name"] = str(trader_row.name or "")
                if not context["strategy_key"]:
                    source_configs = trader_row.source_configs_json if isinstance(trader_row.source_configs_json, list) else []
                    if source_configs:
                        first = source_configs[0] if isinstance(source_configs[0], dict) else {}
                        context["strategy_key"] = str(first.get("strategy_key") or "").strip().lower()

            if include_strategy_source and context["strategy_key"]:
                strategy_row = (
                    (
                        await session.execute(
                            select(Strategy)
                            .where(func.lower(Strategy.slug) == str(context["strategy_key"]).lower())
                            .limit(1)
                        )
                    )
                    .scalars()
                    .first()
                )
                if strategy_row is not None:
                    context["strategy_id"] = str(strategy_row.id or "")
                    context["strategy_name"] = str(strategy_row.name or "")
                    source_text = str(strategy_row.source_code or "")
                    if len(source_text) > _LIVE_TRUTH_MONITOR_STRATEGY_SOURCE_MAX_CHARS:
                        source_text = (
                            source_text[:_LIVE_TRUTH_MONITOR_STRATEGY_SOURCE_MAX_CHARS]
                            + "\n# ... truncated for monitor LLM context ..."
                        )
                    context["strategy_source"] = source_text
                    context["strategy_source_included"] = bool(source_text)

        return context

    async def _run_live_truth_monitor_llm_analysis(
        self,
        *,
        payload: dict[str, Any],
        monitor_result: dict[str, Any],
        job_id: str,
    ) -> dict[str, Any]:
        enabled = _coerce_bool(payload.get("run_llm_analysis"), False)
        requested_model = str(payload.get("llm_model") or "").strip() or None
        include_strategy_source = _coerce_bool(payload.get("include_strategy_source"), True)
        max_alerts_for_llm = _clamp_int(
            payload.get("max_alerts_for_llm"),
            _LIVE_TRUTH_MONITOR_ALERTS_FOR_LLM_DEFAULT,
            1,
            _LIVE_TRUTH_MONITOR_ALERTS_FOR_LLM_MAX,
        )
        if not enabled:
            return {
                "enabled": False,
                "status": "skipped",
                "requested_model": requested_model,
                "alerts_considered": 0,
                "strategy_source_included": False,
                "analysis": {},
            }

        summary = dict(monitor_result.get("summary") or {})
        report = dict(monitor_result.get("report") or {})
        target_trader_id = str(
            summary.get("target_trader_id") or monitor_result.get("trader_id") or payload.get("trader_id") or ""
        ).strip()
        strategy_key_hint = str(
            monitor_result.get("strategy_key") or payload.get("strategy_key") or summary.get("strategy_key") or ""
        ).strip()

        strategy_context = await self._load_live_truth_strategy_context(
            trader_id=target_trader_id,
            strategy_key_hint=strategy_key_hint,
            include_strategy_source=include_strategy_source,
        )

        alert_samples = list(report.get("alert_samples") or [])
        compact_alerts = [
            _compact_live_truth_alert_for_llm(sample)
            for sample in alert_samples[:max_alerts_for_llm]
            if isinstance(sample, dict)
        ]

        try:
            from services.ai import get_llm_manager
            from services.ai.llm_provider import LLMMessage

            manager = get_llm_manager()
            if not manager.is_available():
                return {
                    "enabled": True,
                    "status": "unavailable",
                    "requested_model": requested_model,
                    "alerts_considered": len(compact_alerts),
                    "strategy_source_included": bool(strategy_context.get("strategy_source_included")),
                    "error": "No AI provider configured in Settings.",
                    "analysis": {},
                }

            system_prompt = (
                "You are a live trading reliability engineer for Homerun. "
                "Analyze monitor diagnostics and propose concrete strategy/runtime fixes. "
                "Prioritize changes that reduce false terminal closes, stale state drift, "
                "and missed provider reconciliation. Return only valid JSON."
            )
            prompt_payload = {
                "job_id": job_id,
                "summary": summary,
                "alerts_by_rule": report.get("alerts_by_rule") or {},
                "alert_samples": compact_alerts,
                "strategy_context": {
                    "trader_id": strategy_context.get("trader_id") or "",
                    "trader_name": strategy_context.get("trader_name") or "",
                    "strategy_key": strategy_context.get("strategy_key") or "",
                    "strategy_name": strategy_context.get("strategy_name") or "",
                    "strategy_source": strategy_context.get("strategy_source") or "",
                },
                "response_contract": {
                    "assessment": "overall monitoring assessment",
                    "priority": "critical|high|medium|low",
                    "root_causes": "ordered list of root causes",
                    "strategy_changes": "specific strategy-level changes with target and suggested_change",
                    "runtime_changes": "runtime or worker changes outside strategy code",
                    "tests_to_run": "follow-up tests to validate fixes",
                    "confidence": "0-1 confidence score",
                },
            }
            result = await asyncio.wait_for(
                manager.structured_output(
                    messages=[
                        LLMMessage(role="system", content=system_prompt),
                        LLMMessage(
                            role="user",
                            content=(
                                "Analyze this live-trading monitor payload and return JSON per schema.\n"
                                + json.dumps(prompt_payload, ensure_ascii=True)
                            ),
                        ),
                    ],
                    schema=_LIVE_TRUTH_MONITOR_LLM_SCHEMA,
                    model=requested_model,
                    purpose="live_truth_monitor",
                    session_id=job_id,
                ),
                timeout=_LIVE_TRUTH_MONITOR_LLM_TIMEOUT_SECONDS,
            )
            return {
                "enabled": True,
                "status": "completed",
                "requested_model": requested_model,
                "alerts_considered": len(compact_alerts),
                "strategy_source_included": bool(strategy_context.get("strategy_source_included")),
                "trader_id": strategy_context.get("trader_id") or "",
                "trader_name": strategy_context.get("trader_name") or "",
                "strategy_key": strategy_context.get("strategy_key") or "",
                "strategy_name": strategy_context.get("strategy_name") or "",
                "analysis": dict(result or {}),
            }
        except Exception as exc:
            logger.warning("Live truth monitor LLM analysis failed", job_id=job_id, exc_info=exc)
            return {
                "enabled": True,
                "status": "failed",
                "requested_model": requested_model,
                "alerts_considered": len(compact_alerts),
                "strategy_source_included": bool(strategy_context.get("strategy_source_included")),
                "trader_id": strategy_context.get("trader_id") or "",
                "trader_name": strategy_context.get("trader_name") or "",
                "strategy_key": strategy_context.get("strategy_key") or "",
                "strategy_name": strategy_context.get("strategy_name") or "",
                "error": str(exc),
                "analysis": {},
            }

    async def _run_live_truth_monitor_job(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        duration_seconds = _clamp_int(
            payload.get("duration_seconds"),
            _LIVE_TRUTH_MONITOR_DEFAULT_DURATION_SECONDS,
            _LIVE_TRUTH_MONITOR_MIN_DURATION_SECONDS,
            _LIVE_TRUTH_MONITOR_MAX_DURATION_SECONDS,
        )
        poll_seconds = _clamp_float(
            payload.get("poll_seconds"),
            _LIVE_TRUTH_MONITOR_DEFAULT_POLL_SECONDS,
            _LIVE_TRUTH_MONITOR_MIN_POLL_SECONDS,
            _LIVE_TRUTH_MONITOR_MAX_POLL_SECONDS,
        )
        trader_id = str(payload.get("trader_id") or "").strip()
        trader_name = str(payload.get("trader_name") or "").strip()
        enable_provider_checks = _coerce_bool(payload.get("enable_provider_checks"), False)
        run_llm_analysis = _coerce_bool(payload.get("run_llm_analysis"), False)

        script_path = (_repo_root() / "scripts" / "live_truth_monitor.py").resolve()
        if not script_path.exists() or not script_path.is_file():
            raise RuntimeError(f"Missing monitor script: {script_path}")

        command = [
            sys.executable,
            str(script_path),
            "--duration-seconds",
            str(duration_seconds),
            "--poll-seconds",
            f"{poll_seconds:.3f}",
        ]
        if trader_id:
            command.extend(["--trader-id", trader_id])
        elif trader_name:
            command.extend(["--trader-name", trader_name])
        if enable_provider_checks:
            command.append("--enable-provider-checks")

        ok = await self._set_job_progress(job_id, 0.05, "Launching live truth monitor")
        if not ok:
            raise RuntimeError("Job cancelled")

        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(_repo_root()),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_pipe = process.stdout
        if stdout_pipe is None:
            raise RuntimeError("Failed to capture monitor output.")
        stderr_task = asyncio.create_task(process.stderr.read()) if process.stderr is not None else None

        stdout_events: list[dict[str, Any]] = []
        monitor_start: dict[str, Any] = {}
        monitor_complete: dict[str, Any] = {}
        summary_path: Path | None = None
        report_path: Path | None = None
        run_started_at = utcnow()

        while True:
            try:
                raw_line = await asyncio.wait_for(stdout_pipe.readline(), timeout=1.0)
            except asyncio.TimeoutError:
                if process.returncode is not None:
                    break
                elapsed = max(0.0, (utcnow() - run_started_at).total_seconds())
                progress = 0.1 + (0.65 * min(1.0, elapsed / max(float(duration_seconds), 1.0)))
                ok = await self._set_job_progress(
                    job_id,
                    progress,
                    f"Monitoring trader ({int(elapsed)}s/{duration_seconds}s)",
                )
                if not ok:
                    await self._terminate_subprocess(process)
                    raise RuntimeError("Job cancelled")
                continue

            if raw_line == b"":
                break

            text = raw_line.decode("utf-8", errors="replace").strip()
            if not text:
                continue

            entry: dict[str, Any] = {"line": text}
            try:
                payload_obj = json.loads(text)
            except Exception:
                payload_obj = None

            if isinstance(payload_obj, dict):
                event_name = str(payload_obj.get("event") or "").strip()
                entry["event"] = event_name or "unknown"
                entry["payload"] = payload_obj
                if event_name == "monitor_start":
                    monitor_start = dict(payload_obj)
                    resolved_report = _resolve_monitor_output_path(payload_obj.get("report_path"))
                    if resolved_report is not None:
                        report_path = resolved_report
                elif event_name == "monitor_complete":
                    monitor_complete = dict(payload_obj)
                    resolved_report = _resolve_monitor_output_path(payload_obj.get("report_path"))
                    if resolved_report is not None:
                        report_path = resolved_report
                elif event_name == "summary_path":
                    resolved_summary = _resolve_monitor_output_path(payload_obj.get("path"))
                    if resolved_summary is not None:
                        summary_path = resolved_summary

            if len(stdout_events) < _LIVE_TRUTH_MONITOR_STDOUT_EVENT_LIMIT:
                stdout_events.append(entry)

            elapsed = max(0.0, (utcnow() - run_started_at).total_seconds())
            progress = 0.1 + (0.65 * min(1.0, elapsed / max(float(duration_seconds), 1.0)))
            ok = await self._set_job_progress(
                job_id,
                progress,
                f"Monitoring trader ({int(elapsed)}s/{duration_seconds}s)",
            )
            if not ok:
                await self._terminate_subprocess(process)
                raise RuntimeError("Job cancelled")

        return_code = await process.wait()
        stderr_bytes = await stderr_task if stderr_task is not None else b""
        stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
        if len(stderr_text) > _LIVE_TRUTH_MONITOR_STDERR_TAIL_CHARS:
            stderr_text = "..." + stderr_text[-_LIVE_TRUTH_MONITOR_STDERR_TAIL_CHARS:]

        if return_code != 0:
            detail = stderr_text or "No stderr output."
            raise RuntimeError(f"live_truth_monitor failed with exit code {return_code}: {detail}")

        if summary_path is None:
            for event in reversed(stdout_events):
                payload_obj = event.get("payload")
                if not isinstance(payload_obj, dict):
                    continue
                if str(payload_obj.get("event") or "").strip() != "summary_path":
                    continue
                resolved_summary = _resolve_monitor_output_path(payload_obj.get("path"))
                if resolved_summary is not None:
                    summary_path = resolved_summary
                    break

        summary_json = _read_json_file(summary_path)
        if not summary_json and monitor_complete:
            summary_json = dict(monitor_complete)
        if not summary_json:
            raise RuntimeError("Monitor completed but no summary output was produced.")

        if report_path is None:
            report_path = _resolve_monitor_output_path(summary_json.get("report_path"))
        report_json = _read_live_truth_report(report_path, max_alerts=_LIVE_TRUTH_MONITOR_DEFAULT_MAX_ALERTS)

        monitor_result = {
            "duration_seconds": duration_seconds,
            "poll_seconds": poll_seconds,
            "trader_id": str(
                summary_json.get("target_trader_id") or monitor_start.get("trader_id") or trader_id or ""
            ).strip(),
            "trader_name": str(
                summary_json.get("target_trader_name") or monitor_start.get("trader_name") or trader_name or ""
            ).strip(),
            "strategy_key": str(monitor_start.get("strategy_key") or "").strip(),
            "summary_path": str(summary_path) if summary_path is not None else "",
            "report_path": str(report_path) if report_path is not None else "",
            "summary": summary_json,
            "report": report_json,
            "stdout_events": stdout_events,
            "stderr_tail": stderr_text,
            "exit_code": int(return_code),
        }

        llm_progress_message = "Monitor completed; preparing LLM analysis" if run_llm_analysis else "Monitor completed"
        ok = await self._set_job_progress(job_id, 0.85, llm_progress_message)
        if not ok:
            raise RuntimeError("Job cancelled")

        llm_analysis = await self._run_live_truth_monitor_llm_analysis(
            payload=payload,
            monitor_result=monitor_result,
            job_id=job_id,
        )

        return {
            "status": "success",
            "monitor": monitor_result,
            "llm_analysis": llm_analysis,
        }

    async def list_execution_sim_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            rows = (
                (
                    await session.execute(
                        select(ExecutionSimRun)
                        .order_by(ExecutionSimRun.created_at.desc())
                        .limit(max(1, min(limit, 500)))
                    )
                )
                .scalars()
                .all()
            )
        return [self._execution_run_to_dict(row) for row in rows]

    async def get_execution_sim_run(self, run_id: str) -> Optional[dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            row = await session.get(ExecutionSimRun, run_id)
            return self._execution_run_to_dict(row) if row else None

    async def list_execution_sim_events(
        self,
        run_id: str,
        *,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            rows = (
                (
                    await session.execute(
                        select(ExecutionSimEvent)
                        .where(ExecutionSimEvent.run_id == run_id)
                        .order_by(ExecutionSimEvent.sequence.asc())
                        .offset(max(0, offset))
                        .limit(max(1, min(limit, 5000)))
                    )
                )
                .scalars()
                .all()
            )
        return [self._execution_event_to_dict(row) for row in rows]

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
            "min_directional_accuracy": float(s.validation_min_directional_accuracy or 0.52),
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
                row.validation_min_directional_accuracy = float(patch["min_directional_accuracy"])
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
                await session.execute(
                    select(
                        OpportunityHistory.strategy_type,
                        OpportunityHistory.detected_at,
                        OpportunityHistory.expected_roi,
                        OpportunityHistory.actual_roi,
                    ).where(and_(*filters))
                )
            ).all()

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
            directional_hits = sum(1 for e, a, _ in vals if ((e >= 0 and a >= 0) or (e < 0 and a < 0)))
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

    async def compute_calibration_trend(self, days: int = 90, bucket_days: int = 7) -> list[dict[str, Any]]:
        """Time-bucketed trend of calibration quality."""
        cutoff = utcnow() - timedelta(days=days)
        async with AsyncSessionLocal() as session:
            rows = (
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
            ).all()

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

    async def compute_trader_orchestrator_execution_metrics(self, days: int = 30) -> dict[str, Any]:
        cutoff = utcnow() - timedelta(days=days)
        async with AsyncSessionLocal() as session:
            rows = (
                await session.execute(
                    select(
                        TraderOrder.source,
                        TradeSignal.strategy_type,
                        TraderOrder.status,
                        TraderOrder.mode,
                        TraderOrder.notional_usd,
                        TraderOrder.actual_profit,
                        TraderOrder.edge_percent,
                        TraderOrder.confidence,
                    )
                    .join(TradeSignal, TraderOrder.signal_id == TradeSignal.id, isouter=True)
                    .where(TraderOrder.created_at >= cutoff)
                )
            ).all()

        summary = {
            "window_days": int(days),
            "sample_size": len(rows),
            "executed_or_open": 0,
            "failed": 0,
            "closed": 0,
            "resolved": 0,
            "terminal": 0,
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
            status_key = str(status or "unknown").strip().lower()

            if edge is not None:
                edges.append(float(edge))
            if confidence is not None:
                confidences.append(float(confidence))

            notional_value = float(notional or 0.0)
            pnl_value = float(actual_profit or 0.0)
            is_failed = status_key == "failed"
            is_executed_or_open = status_key in {"submitted", "executed", "open"}
            is_resolved = status_key in {"resolved_win", "resolved_loss"}
            is_closed = status_key in {"closed_win", "closed_loss"}
            is_terminal = is_resolved or is_closed

            if is_executed_or_open:
                summary["executed_or_open"] += 1
            if is_failed:
                summary["failed"] += 1
            if is_closed:
                summary["closed"] += 1
            if is_resolved:
                summary["resolved"] += 1
            if is_terminal:
                summary["terminal"] += 1
            summary["notional_total_usd"] += notional_value
            summary["realized_pnl_total"] += pnl_value

            source_row = by_source.setdefault(
                source_key,
                {
                    "source": source_key,
                    "total": 0,
                    "failed": 0,
                    "executed_or_open": 0,
                    "closed": 0,
                    "resolved": 0,
                    "terminal": 0,
                    "notional_total_usd": 0.0,
                    "realized_pnl_total": 0.0,
                    "modes": {},
                },
            )
            source_row["total"] += 1
            source_row["failed"] += int(is_failed)
            source_row["executed_or_open"] += int(is_executed_or_open)
            source_row["closed"] += int(is_closed)
            source_row["resolved"] += int(is_resolved)
            source_row["terminal"] += int(is_terminal)
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
                    "closed": 0,
                    "resolved": 0,
                    "terminal": 0,
                    "notional_total_usd": 0.0,
                    "realized_pnl_total": 0.0,
                },
            )
            strategy_row["total"] += 1
            strategy_row["failed"] += int(is_failed)
            strategy_row["executed_or_open"] += int(is_executed_or_open)
            strategy_row["closed"] += int(is_closed)
            strategy_row["resolved"] += int(is_resolved)
            strategy_row["terminal"] += int(is_terminal)
            strategy_row["notional_total_usd"] += notional_value
            strategy_row["realized_pnl_total"] += pnl_value

        sample_size = int(summary["sample_size"])
        summary["failure_rate"] = float(summary["failed"]) / sample_size if sample_size > 0 else 0.0
        summary["avg_edge_percent"] = sum(edges) / len(edges) if edges else 0.0
        summary["avg_confidence"] = sum(confidences) / len(confidences) if confidences else 0.0
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

    async def compute_events_resolver_metrics(
        self,
        days: int = 7,
        max_signals: int = 500,
    ) -> dict[str, Any]:
        cutoff = utcnow() - timedelta(days=days)
        async with AsyncSessionLocal() as session:
            signals = (
                (
                    await session.execute(
                        select(EventsSignal)
                        .where(EventsSignal.detected_at >= cutoff)
                        .order_by(EventsSignal.detected_at.desc())
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

            candidates: list[dict[str, Any]] = []
            for signal in signals:
                metadata = signal.metadata_json if isinstance(signal.metadata_json, dict) else {}
                related_market_ids = [
                    str(item).strip() for item in list(signal.related_market_ids or []) if str(item).strip()
                ]
                market_relevance_score = signal.market_relevance_score
                if market_relevance_score is None:
                    try:
                        market_relevance_score = float(metadata.get("market_relevance_score"))
                    except Exception:
                        market_relevance_score = 0.0
                relevance_value = float(market_relevance_score or 0.0)
                tradable = bool(related_market_ids) and relevance_value >= 0.3
                candidates.append(
                    {
                        "signal_type": str(signal.signal_type or "unknown"),
                        "tradable": bool(tradable),
                    }
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
            row["tradable_rate"] = float(row["tradable"]) / total if total > 0 else 0.0

        total_candidates = len(candidates)
        return {
            "window_days": int(days),
            "signals_sampled": len(signals),
            "candidates": total_candidates,
            "tradable": len(tradable),
            "tradable_rate": (len(tradable) / total_candidates if total_candidates > 0 else 0.0),
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
                    directional_accuracy is not None and directional_accuracy < cfg["min_directional_accuracy"]
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
                                f"Low directional accuracy ({directional_accuracy:.3f}) and high MAE ({mae_roi:.2f})"
                            )
                        elif weak_accuracy:
                            row.last_reason = f"Low directional accuracy ({directional_accuracy:.3f})"
                        else:
                            row.last_reason = f"High MAE ({mae_roi:.2f})"
                        demoted.append(strategy_type)
                elif cfg["auto_promote"]:
                    promoted_ok = (
                        enough_samples
                        and directional_accuracy is not None
                        and directional_accuracy >= cfg["min_directional_accuracy"] + 0.03
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
                        select(StrategyValidationProfile).order_by(StrategyValidationProfile.strategy_type.asc())
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
                await session.execute(
                    select(StrategyValidationProfile.strategy_type).where(StrategyValidationProfile.status == "demoted")
                )
            ).all()
        return {r[0] for r in rows}

    async def _recover_incomplete_jobs(self) -> None:
        async with AsyncSessionLocal() as session:
            rows = (
                (await session.execute(select(ValidationJob).where(ValidationJob.status == "running"))).scalars().all()
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

    async def _set_job_progress(self, job_id: str, progress: float, message: Optional[str] = None) -> bool:
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
            payload = dict(row.payload or {})
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
                        name=payload.get("parameter_set_name") or f"Backtest {utcnow().isoformat(timespec='seconds')}",
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
                        name=payload.get("best_set_name") or f"Optimized Best {utcnow().isoformat(timespec='seconds')}",
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
            elif job_type == "execution_simulation":
                await self._set_job_progress(job_id, 0.1, "Preparing execution simulation")
                strategy_key = str(payload.get("strategy_key") or "").strip().lower()
                source_key = str(payload.get("source_key") or "").strip().lower()
                if not strategy_key:
                    raise ValueError("execution_simulation requires payload.strategy_key")
                if not source_key:
                    raise ValueError("execution_simulation requires payload.source_key")

                async with AsyncSessionLocal() as sim_session:
                    run_row = ExecutionSimRun(
                        id=str(uuid.uuid4()),
                        job_id=job_id,
                        strategy_key=strategy_key,
                        source_key=source_key,
                        status="queued",
                        run_seed=str(payload.get("run_seed") or "").strip() or None,
                        market_scope_json=dict(payload.get("market_scope") or {}),
                        params_json=dict(payload),
                        requested_start_at=_parse_datetime(payload.get("start_at")),
                        requested_end_at=_parse_datetime(payload.get("end_at")),
                        created_at=utcnow(),
                        updated_at=utcnow(),
                    )
                    sim_session.add(run_row)
                    await sim_session.commit()
                    await sim_session.refresh(run_row)

                    await self._set_job_progress(job_id, 0.35, "Running execution simulation")
                    try:
                        summary = await execution_simulator.run(
                            sim_session,
                            run_row=run_row,
                            payload=payload,
                        )
                        await sim_session.commit()
                    except Exception as exc:
                        run_row.status = "failed"
                        run_row.error_message = str(exc)
                        run_row.finished_at = utcnow()
                        run_row.summary_json = {
                            "status": "failed",
                            "error": str(exc),
                        }
                        run_row.updated_at = utcnow()
                        await sim_session.commit()
                        raise

                await self._set_job_progress(job_id, 0.9, "Execution simulation completed")
                final_result = {
                    "status": "success",
                    "run_id": run_row.id,
                    "summary": summary,
                }
            elif job_type == "live_truth_monitor":
                final_result = await self._run_live_truth_monitor_job(job_id, payload)
            else:
                raise ValueError(f"Unsupported job type: {job_type}")

            if job_type in {"backtest", "optimize", "execution_simulation"}:
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
    def _execution_run_to_dict(row: ExecutionSimRun) -> dict[str, Any]:
        return {
            "id": row.id,
            "job_id": row.job_id,
            "strategy_key": row.strategy_key,
            "source_key": row.source_key,
            "status": row.status,
            "run_seed": row.run_seed,
            "dataset_hash": row.dataset_hash,
            "config_hash": row.config_hash,
            "code_sha": row.code_sha,
            "market_scope": row.market_scope_json or {},
            "params": row.params_json or {},
            "requested_start_at": row.requested_start_at.isoformat() if row.requested_start_at else None,
            "requested_end_at": row.requested_end_at.isoformat() if row.requested_end_at else None,
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "finished_at": row.finished_at.isoformat() if row.finished_at else None,
            "summary": row.summary_json or {},
            "error_message": row.error_message,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }

    @staticmethod
    def _execution_event_to_dict(row: ExecutionSimEvent) -> dict[str, Any]:
        return {
            "id": row.id,
            "run_id": row.run_id,
            "sequence": int(row.sequence or 0),
            "event_type": row.event_type,
            "event_at": row.event_at.isoformat() if row.event_at else None,
            "signal_id": row.signal_id,
            "market_id": row.market_id,
            "direction": row.direction,
            "price": row.price,
            "quantity": row.quantity,
            "notional_usd": row.notional_usd,
            "fees_usd": row.fees_usd,
            "slippage_bps": row.slippage_bps,
            "realized_pnl_usd": row.realized_pnl_usd,
            "unrealized_pnl_usd": row.unrealized_pnl_usd,
            "payload": row.payload_json or {},
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }

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
