from __future__ import annotations

import json
from typing import Any

from sqlalchemy import desc, select

from models.database import AsyncSessionLocal, TradeSignal, Trader, TraderDecision, TraderOrder
from services.ai import get_llm_manager
from services.ai.agent import AgentTool, run_agent_to_completion
from services.strategy_experiments import (
    create_strategy_experiment,
    list_strategy_experiments,
    serialize_strategy_experiment,
)
from services.strategy_loader import StrategyValidationError, strategy_loader, validate_strategy_source
from services.strategy_runtime import bump_strategy_runtime_revisions
from services.strategy_versioning import (
    create_strategy_version_snapshot,
    ensure_strategy_version_seeded,
    get_strategy_by_id_or_slug,
    list_strategy_versions,
    normalize_strategy_version,
    serialize_strategy_version,
)
from services.validation_service import validation_service
from utils.utcnow import utcnow


def _normalize_source_key(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_strategy_key(value: Any) -> str:
    return str(value or "").strip().lower()


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
    return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _serialize_trader_source_configs(source_configs: Any) -> list[dict[str, Any]]:
    rows = source_configs if isinstance(source_configs, list) else []
    out: list[dict[str, Any]] = []
    for raw in rows:
        item = dict(raw) if isinstance(raw, dict) else {}
        out.append(
            {
                "source_key": _normalize_source_key(item.get("source_key")),
                "strategy_key": _normalize_strategy_key(item.get("strategy_key")),
                "strategy_version": normalize_strategy_version(item.get("strategy_version")),
                "strategy_params": dict(item.get("strategy_params") or {}),
            }
        )
    return out


async def _tool_get_trader_context(args: dict[str, Any]) -> dict[str, Any]:
    trader_id = str(args.get("trader_id") or "").strip()
    if not trader_id:
        raise RuntimeError("trader_id is required")
    decision_limit = max(1, min(int(args.get("decision_limit") or 30), 200))
    order_limit = max(1, min(int(args.get("order_limit") or 30), 200))

    async with AsyncSessionLocal() as session:
        trader_row = await session.get(Trader, trader_id)
        if trader_row is None:
            raise RuntimeError(f"Trader '{trader_id}' not found")

        decisions = (
            (
                await session.execute(
                    select(TraderDecision)
                    .where(TraderDecision.trader_id == trader_id)
                    .order_by(desc(TraderDecision.created_at))
                    .limit(decision_limit)
                )
            )
            .scalars()
            .all()
        )
        orders = (
            (
                await session.execute(
                    select(TraderOrder)
                    .where(TraderOrder.trader_id == trader_id)
                    .order_by(desc(TraderOrder.created_at))
                    .limit(order_limit)
                )
            )
            .scalars()
            .all()
        )
        signals = (
            (
                await session.execute(
                    select(TradeSignal)
                    .where(TradeSignal.source.in_([cfg["source_key"] for cfg in _serialize_trader_source_configs(trader_row.source_configs_json)]))
                    .order_by(desc(TradeSignal.created_at))
                    .limit(30)
                )
            )
            .scalars()
            .all()
        )

    return {
        "trader": {
            "id": str(trader_row.id),
            "name": str(trader_row.name or ""),
            "mode": str(trader_row.mode or "paper"),
            "is_enabled": bool(trader_row.is_enabled),
            "is_paused": bool(trader_row.is_paused),
            "interval_seconds": int(trader_row.interval_seconds or 60),
            "source_configs": _serialize_trader_source_configs(trader_row.source_configs_json),
            "risk_limits": dict(trader_row.risk_limits_json or {}),
            "metadata": dict(trader_row.metadata_json or {}),
            "updated_at": trader_row.updated_at.isoformat() if trader_row.updated_at else None,
        },
        "decisions": [
            {
                "id": str(row.id),
                "source": str(row.source or ""),
                "strategy_key": str(row.strategy_key or ""),
                "strategy_version": int(row.strategy_version) if row.strategy_version is not None else None,
                "decision": str(row.decision or ""),
                "reason": row.reason,
                "score": float(row.score) if row.score is not None else None,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in decisions
        ],
        "orders": [
            {
                "id": str(row.id),
                "decision_id": str(row.decision_id or ""),
                "source": str(row.source or ""),
                "strategy_key": str(row.strategy_key or ""),
                "strategy_version": int(row.strategy_version) if row.strategy_version is not None else None,
                "status": str(row.status or ""),
                "mode": str(row.mode or ""),
                "market_id": str(row.market_id or ""),
                "notional_usd": float(row.notional_usd or 0.0),
                "effective_price": float(row.effective_price) if row.effective_price is not None else None,
                "actual_profit": float(row.actual_profit) if row.actual_profit is not None else None,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in orders
        ],
        "signals": [
            {
                "id": str(row.id),
                "source": str(row.source or ""),
                "strategy_type": str(row.strategy_type or ""),
                "market_id": str(row.market_id or ""),
                "market_question": str(row.market_question or ""),
                "edge_percent": float(row.edge_percent) if row.edge_percent is not None else None,
                "confidence": float(row.confidence) if row.confidence is not None else None,
                "status": str(row.status or ""),
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in signals
        ],
    }


async def _tool_get_strategy_context(args: dict[str, Any]) -> dict[str, Any]:
    strategy_key = _normalize_strategy_key(args.get("strategy_key"))
    if not strategy_key:
        raise RuntimeError("strategy_key is required")
    version = normalize_strategy_version(args.get("version"))
    include_source = bool(args.get("include_source", True))
    limit = max(1, min(int(args.get("limit") or 100), 1000))

    async with AsyncSessionLocal() as session:
        strategy_row = await get_strategy_by_id_or_slug(session, strategy_key=strategy_key)
        if strategy_row is None:
            raise RuntimeError(f"Strategy '{strategy_key}' not found")
        await ensure_strategy_version_seeded(
            session,
            strategy=strategy_row,
            reason="monitor_agent_seed_context",
            created_by="monitor_agent",
            commit=False,
        )
        version_rows = await list_strategy_versions(session, strategy_id=str(strategy_row.id), limit=limit)
        selected = next((row for row in version_rows if int(row.version or 0) == int(version or -1)), None)
        if selected is None and version_rows:
            selected = version_rows[0]

    return {
        "strategy": {
            "id": str(strategy_row.id),
            "slug": str(strategy_row.slug or ""),
            "name": str(strategy_row.name or ""),
            "description": strategy_row.description,
            "source_key": str(strategy_row.source_key or ""),
            "current_version": int(strategy_row.version or 1),
            "enabled": bool(strategy_row.enabled),
            "status": str(strategy_row.status or ""),
            "config": dict(strategy_row.config or {}),
            "config_schema": dict(strategy_row.config_schema or {}),
        },
        "selected_version": serialize_strategy_version(selected, include_source=include_source)
        if selected is not None
        else None,
        "versions": [serialize_strategy_version(row, include_source=False) for row in version_rows],
    }


async def _tool_patch_strategy_params(args: dict[str, Any]) -> dict[str, Any]:
    strategy_key = _normalize_strategy_key(args.get("strategy_key"))
    reason = str(args.get("reason") or "monitor_agent_param_tune").strip() or "monitor_agent_param_tune"
    params_patch = args.get("params_patch")
    if not strategy_key:
        raise RuntimeError("strategy_key is required")
    if not isinstance(params_patch, dict):
        raise RuntimeError("params_patch must be an object")
    created_by = str(args.get("created_by") or "monitor_agent").strip() or "monitor_agent"

    async with AsyncSessionLocal() as session:
        row = await get_strategy_by_id_or_slug(session, strategy_key=strategy_key)
        if row is None:
            raise RuntimeError(f"Strategy '{strategy_key}' not found")

        previous_config = dict(row.config or {})
        merged_config = {**previous_config, **dict(params_patch)}
        if merged_config == previous_config:
            return {
                "status": "no_change",
                "strategy_key": strategy_key,
                "version": int(row.version or 1),
                "config": merged_config,
            }

        previous_version = int(row.version or 1)
        row.config = merged_config
        restored = await create_strategy_version_snapshot(
            session,
            strategy=row,
            reason=reason,
            created_by=created_by,
            parent_version=previous_version,
            commit=False,
        )

        if row.enabled:
            try:
                strategy_loader.load(row.slug, row.source_code, row.config or None)
                row.status = "loaded"
                row.error_message = None
            except StrategyValidationError as exc:
                row.status = "error"
                row.error_message = str(exc)

        await bump_strategy_runtime_revisions(
            session,
            source_keys=[str(row.source_key or "").strip().lower()],
            commit=False,
        )
        await session.commit()
        await session.refresh(row)

    return {
        "status": "updated",
        "strategy_key": strategy_key,
        "version": int(row.version or 1),
        "snapshot": serialize_strategy_version(restored, include_source=False),
        "config": dict(row.config or {}),
    }


async def _tool_update_strategy_source(args: dict[str, Any]) -> dict[str, Any]:
    strategy_key = _normalize_strategy_key(args.get("strategy_key"))
    source_code = str(args.get("source_code") or "")
    reason = str(args.get("reason") or "monitor_agent_source_edit").strip() or "monitor_agent_source_edit"
    created_by = str(args.get("created_by") or "monitor_agent").strip() or "monitor_agent"
    if not strategy_key:
        raise RuntimeError("strategy_key is required")
    if len(source_code.strip()) < 10:
        raise RuntimeError("source_code must be at least 10 characters")

    validation = validate_strategy_source(source_code)
    if not validation.get("valid"):
        raise RuntimeError(
            "Source validation failed: "
            + "; ".join(str(item) for item in list(validation.get("errors") or [])[:6])
        )

    async with AsyncSessionLocal() as session:
        row = await get_strategy_by_id_or_slug(session, strategy_key=strategy_key)
        if row is None:
            raise RuntimeError(f"Strategy '{strategy_key}' not found")

        if source_code == str(row.source_code or ""):
            return {
                "status": "no_change",
                "strategy_key": strategy_key,
                "version": int(row.version or 1),
            }

        previous_version = int(row.version or 1)
        row.source_code = source_code
        row.class_name = validation.get("class_name")
        if not row.name and validation.get("strategy_name"):
            row.name = str(validation.get("strategy_name"))
        if not row.description and validation.get("strategy_description"):
            row.description = str(validation.get("strategy_description"))

        snapshot = await create_strategy_version_snapshot(
            session,
            strategy=row,
            reason=reason,
            created_by=created_by,
            parent_version=previous_version,
            commit=False,
        )

        if row.enabled:
            try:
                strategy_loader.load(row.slug, row.source_code, row.config or None)
                row.status = "loaded"
                row.error_message = None
            except StrategyValidationError as exc:
                row.status = "error"
                row.error_message = str(exc)

        await bump_strategy_runtime_revisions(
            session,
            source_keys=[str(row.source_key or "").strip().lower()],
            commit=False,
        )
        await session.commit()
        await session.refresh(row)

    return {
        "status": "updated",
        "strategy_key": strategy_key,
        "version": int(row.version or 1),
        "snapshot": serialize_strategy_version(snapshot, include_source=False),
        "class_name": row.class_name,
    }


async def _tool_set_trader_source_strategy_version(args: dict[str, Any]) -> dict[str, Any]:
    trader_id = str(args.get("trader_id") or "").strip()
    source_key = _normalize_source_key(args.get("source_key"))
    strategy_key = _normalize_strategy_key(args.get("strategy_key"))
    requested_version = normalize_strategy_version(args.get("strategy_version"))
    if not trader_id:
        raise RuntimeError("trader_id is required")
    if not source_key:
        raise RuntimeError("source_key is required")

    async with AsyncSessionLocal() as session:
        trader_row = await session.get(Trader, trader_id)
        if trader_row is None:
            raise RuntimeError(f"Trader '{trader_id}' not found")
        source_configs = _serialize_trader_source_configs(trader_row.source_configs_json)
        matched = False
        for config in source_configs:
            if config["source_key"] != source_key:
                continue
            if strategy_key and config["strategy_key"] != strategy_key:
                continue
            config["strategy_version"] = int(requested_version) if requested_version is not None else None
            matched = True
        if not matched:
            raise RuntimeError("No matching source configuration found on trader")
        trader_row.source_configs_json = source_configs
        trader_row.updated_at = utcnow()
        await session.commit()
        await session.refresh(trader_row)
    return {
        "status": "updated",
        "trader_id": trader_id,
        "source_configs": _serialize_trader_source_configs(trader_row.source_configs_json),
    }


async def _tool_create_ab_experiment(args: dict[str, Any]) -> dict[str, Any]:
    source_key = _normalize_source_key(args.get("source_key"))
    strategy_key = _normalize_strategy_key(args.get("strategy_key"))
    control_version = normalize_strategy_version(args.get("control_version"))
    candidate_version = normalize_strategy_version(args.get("candidate_version"))
    if control_version is None or candidate_version is None:
        raise RuntimeError("control_version and candidate_version must be explicit integers")
    name = str(args.get("name") or "").strip() or f"{strategy_key}: v{control_version} vs v{candidate_version}"
    allocation = float(args.get("candidate_allocation_pct") or 50.0)
    notes = str(args.get("notes") or "").strip() or None
    scope = args.get("scope")
    if not isinstance(scope, dict):
        scope = {}

    async with AsyncSessionLocal() as session:
        row = await create_strategy_experiment(
            session,
            name=name,
            source_key=source_key,
            strategy_key=strategy_key,
            control_version=int(control_version),
            candidate_version=int(candidate_version),
            candidate_allocation_pct=allocation,
            scope=scope,
            notes=notes,
            created_by="monitor_agent",
            commit=True,
        )
    return {"status": "created", "experiment": serialize_strategy_experiment(row)}


async def _tool_list_experiments(args: dict[str, Any]) -> dict[str, Any]:
    source_key = _normalize_source_key(args.get("source_key")) or None
    strategy_key = _normalize_strategy_key(args.get("strategy_key")) or None
    status = str(args.get("status") or "").strip().lower() or None
    limit = max(1, min(int(args.get("limit") or 50), 500))

    async with AsyncSessionLocal() as session:
        rows = await list_strategy_experiments(
            session,
            source_key=source_key,
            strategy_key=strategy_key,
            status=status,
            limit=limit,
        )
    return {"items": [serialize_strategy_experiment(row) for row in rows], "total": len(rows)}


async def _tool_get_monitor_job_context(args: dict[str, Any]) -> dict[str, Any]:
    job_id = str(args.get("job_id") or "").strip()
    max_alerts = max(1, min(int(args.get("max_alerts") or 120), 1000))
    if not job_id:
        raise RuntimeError("job_id is required")
    payload = await validation_service.get_live_truth_monitor_raw(job_id, max_alerts=max_alerts)
    if payload is None:
        raise RuntimeError(f"Monitor job '{job_id}' not found or unavailable")
    report = dict((payload.get("monitor") or {}).get("report") or {})
    return {
        "job_id": job_id,
        "job_status": payload.get("job_status"),
        "summary": dict((payload.get("monitor") or {}).get("summary") or {}),
        "alerts_by_rule": dict(report.get("alerts_by_rule") or {}),
        "alert_count": int(report.get("alert_count") or 0),
        "alert_samples": list(report.get("alert_samples") or [])[:max_alerts],
        "llm_analysis": dict(payload.get("llm_analysis") or {}),
    }


def _monitor_agent_tools() -> list[AgentTool]:
    return [
        AgentTool(
            name="get_trader_context",
            description="Fetch trader runtime context, recent decisions, orders, and signals.",
            parameters={
                "type": "object",
                "properties": {
                    "trader_id": {"type": "string"},
                    "decision_limit": {"type": "integer"},
                    "order_limit": {"type": "integer"},
                },
                "required": ["trader_id"],
                "additionalProperties": False,
            },
            handler=_tool_get_trader_context,
            max_calls=6,
        ),
        AgentTool(
            name="get_strategy_context",
            description="Fetch strategy source/config and version history.",
            parameters={
                "type": "object",
                "properties": {
                    "strategy_key": {"type": "string"},
                    "version": {"type": ["string", "integer", "null"]},
                    "include_source": {"type": "boolean"},
                    "limit": {"type": "integer"},
                },
                "required": ["strategy_key"],
                "additionalProperties": False,
            },
            handler=_tool_get_strategy_context,
            max_calls=8,
        ),
        AgentTool(
            name="patch_strategy_params",
            description="Apply config parameter patch to a strategy and create a new immutable version.",
            parameters={
                "type": "object",
                "properties": {
                    "strategy_key": {"type": "string"},
                    "params_patch": {"type": "object"},
                    "reason": {"type": "string"},
                    "created_by": {"type": "string"},
                },
                "required": ["strategy_key", "params_patch"],
                "additionalProperties": False,
            },
            handler=_tool_patch_strategy_params,
            max_calls=4,
        ),
        AgentTool(
            name="update_strategy_source",
            description="Replace strategy Python source, validate it, and create a new immutable version.",
            parameters={
                "type": "object",
                "properties": {
                    "strategy_key": {"type": "string"},
                    "source_code": {"type": "string"},
                    "reason": {"type": "string"},
                    "created_by": {"type": "string"},
                },
                "required": ["strategy_key", "source_code"],
                "additionalProperties": False,
            },
            handler=_tool_update_strategy_source,
            max_calls=3,
        ),
        AgentTool(
            name="set_trader_source_strategy_version",
            description="Pin a trader source config to a strategy version (or latest).",
            parameters={
                "type": "object",
                "properties": {
                    "trader_id": {"type": "string"},
                    "source_key": {"type": "string"},
                    "strategy_key": {"type": "string"},
                    "strategy_version": {"type": ["integer", "string", "null"]},
                },
                "required": ["trader_id", "source_key"],
                "additionalProperties": False,
            },
            handler=_tool_set_trader_source_strategy_version,
            max_calls=5,
        ),
        AgentTool(
            name="create_ab_experiment",
            description="Create a strategy version A/B experiment for orchestrator assignment.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "source_key": {"type": "string"},
                    "strategy_key": {"type": "string"},
                    "control_version": {"type": "integer"},
                    "candidate_version": {"type": "integer"},
                    "candidate_allocation_pct": {"type": "number"},
                    "scope": {"type": "object"},
                    "notes": {"type": "string"},
                },
                "required": ["source_key", "strategy_key", "control_version", "candidate_version"],
                "additionalProperties": False,
            },
            handler=_tool_create_ab_experiment,
            max_calls=4,
        ),
        AgentTool(
            name="list_experiments",
            description="List strategy A/B experiments and status.",
            parameters={
                "type": "object",
                "properties": {
                    "source_key": {"type": "string"},
                    "strategy_key": {"type": "string"},
                    "status": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": [],
                "additionalProperties": False,
            },
            handler=_tool_list_experiments,
            max_calls=4,
        ),
        AgentTool(
            name="get_monitor_job_context",
            description="Load live-truth monitor summary, alerts, and LLM analysis for a completed monitor job.",
            parameters={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "max_alerts": {"type": "integer"},
                },
                "required": ["job_id"],
                "additionalProperties": False,
            },
            handler=_tool_get_monitor_job_context,
            max_calls=4,
        ),
    ]


async def run_strategy_monitor_agent(
    *,
    trader_id: str,
    prompt: str,
    model: str | None = None,
    max_iterations: int = 12,
    monitor_job_id: str | None = None,
) -> dict[str, Any]:
    normalized_trader = str(trader_id or "").strip()
    if not normalized_trader:
        raise ValueError("trader_id is required")
    normalized_prompt = str(prompt or "").strip()
    if not normalized_prompt:
        raise ValueError("prompt is required")

    manager = get_llm_manager()
    if not manager.is_available():
        raise RuntimeError("No AI provider configured in settings.")

    base_context: dict[str, Any] = {
        "trader_id": normalized_trader,
        "monitor_job_id": str(monitor_job_id or "").strip() or None,
        "requested_at": utcnow().isoformat(),
    }

    system_prompt = (
        "You are the Homerun monitor/iterate trading strategist. "
        "You must inspect live trading decisions/orders and strategy code using tools, then apply concrete improvements. "
        "When a change is requested, call the appropriate mutation tool to create immutable versions or experiments. "
        "Never fabricate IDs or versions. Return final JSON with keys: summary, actions_taken, suggested_next_steps."
    )
    query_payload = {
        "goal": normalized_prompt,
        "context": base_context,
        "instructions": [
            "Use get_trader_context first unless context is already provided.",
            "Use get_strategy_context before modifying strategy code/config.",
            "Use patch_strategy_params for parameter tuning.",
            "Use update_strategy_source for direct source edits.",
            "Use create_ab_experiment for A/B tests.",
            "Use set_trader_source_strategy_version to pin versions on a trader source config.",
        ],
    }

    result = await run_agent_to_completion(
        system_prompt=system_prompt,
        query=json.dumps(query_payload, ensure_ascii=True),
        tools=_monitor_agent_tools(),
        model=model,
        max_iterations=max(1, min(int(max_iterations or 12), 24)),
        session_type="strategy_monitor_agent",
        market_id=None,
        opportunity_id=None,
    )

    answer = str(((result.get("result") or {}).get("answer")) or "").strip()
    return {
        "session_id": str(result.get("session_id") or ""),
        "answer": answer,
        "parsed": _extract_json_payload(answer),
        "raw": _json_safe(result),
    }
