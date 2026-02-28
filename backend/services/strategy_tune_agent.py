from __future__ import annotations

import json
from typing import Any

from sqlalchemy import desc, select

from models.database import AsyncSessionLocal, TradeSignal, Trader, TraderDecision, TraderOrder
from services.ai import get_llm_manager
from services.ai.agent import AgentTool, run_agent_to_completion
from services.strategy_versioning import (
    ensure_strategy_version_seeded,
    get_strategy_by_id_or_slug,
    list_strategy_versions,
    normalize_strategy_version,
    serialize_strategy_version,
)
from services.trader_orchestrator_state import update_trader
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
    candidates: list[str] = [text]

    if "```" in text:
        parts = text.split("```")
        for index in range(1, len(parts), 2):
            block = parts[index].strip()
            if not block:
                continue
            lowered = block.lower()
            if lowered.startswith("json"):
                block = block[4:].lstrip()
            if block:
                candidates.append(block)

    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)

        try:
            parsed = json.loads(normalized)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        start = normalized.find("{")
        end = normalized.rfind("}")
        if start >= 0 and end > start:
            fragment = normalized[start : end + 1].strip()
            if fragment and fragment not in seen:
                seen.add(fragment)
                try:
                    parsed_fragment = json.loads(fragment)
                    if isinstance(parsed_fragment, dict):
                        return parsed_fragment
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


def _normalize_param_updates(raw_updates: Any) -> list[dict[str, Any]]:
    updates = raw_updates if isinstance(raw_updates, list) else []
    out: list[dict[str, Any]] = []
    for raw in updates:
        item = raw if isinstance(raw, dict) else None
        if item is None:
            continue
        source_key = _normalize_source_key(item.get("source_key"))
        strategy_key = _normalize_strategy_key(item.get("strategy_key")) or None
        params_patch = item.get("params_patch")
        if not source_key or not isinstance(params_patch, dict) or not params_patch:
            continue
        out.append(
            {
                "source_key": source_key,
                "strategy_key": strategy_key,
                "params_patch": dict(params_patch),
                "reason": str(item.get("reason") or "llm_tune").strip() or "llm_tune",
            }
        )
    return out


def _summarize_trader_context(payload: dict[str, Any]) -> dict[str, Any]:
    trader = dict(payload.get("trader") or {})
    orders = payload.get("orders") if isinstance(payload.get("orders"), list) else []
    decisions = payload.get("decisions") if isinstance(payload.get("decisions"), list) else []
    signals = payload.get("signals") if isinstance(payload.get("signals"), list) else []

    resolved = 0
    open_orders = 0
    wins = 0
    losses = 0
    pnl = 0.0
    notional = 0.0
    edge_sum = 0.0
    edge_count = 0

    for row in orders:
        item = row if isinstance(row, dict) else {}
        status = str(item.get("status") or "").strip().lower()
        actual_profit = float(item.get("actual_profit") or 0.0)
        trade_notional = abs(float(item.get("notional_usd") or 0.0))
        notional += trade_notional
        if status in {"open", "pending", "submitted", "partially_filled"}:
            open_orders += 1
            continue
        if status in {"resolved_win", "resolved_loss"}:
            resolved += 1
            pnl += actual_profit
            if actual_profit > 0:
                wins += 1
            elif actual_profit < 0:
                losses += 1
        edge = item.get("edge_percent")
        if edge is not None:
            edge_sum += float(edge)
            edge_count += 1

    decision_counts: dict[str, int] = {"selected": 0, "blocked": 0, "skipped": 0, "failed": 0}
    for row in decisions:
        item = row if isinstance(row, dict) else {}
        decision = str(item.get("decision") or "").strip().lower()
        if decision in decision_counts:
            decision_counts[decision] += 1

    signals_by_source: dict[str, int] = {}
    for row in signals:
        item = row if isinstance(row, dict) else {}
        source = str(item.get("source") or "unknown").strip().lower() or "unknown"
        signals_by_source[source] = signals_by_source.get(source, 0) + 1

    return {
        "trader": {
            "id": str(trader.get("id") or ""),
            "name": str(trader.get("name") or ""),
            "mode": str(trader.get("mode") or "paper"),
            "is_enabled": bool(trader.get("is_enabled")),
            "is_paused": bool(trader.get("is_paused")),
            "source_count": len(trader.get("source_configs") or []),
        },
        "performance": {
            "orders_total": len(orders),
            "orders_open": open_orders,
            "orders_resolved": resolved,
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / resolved) if resolved > 0 else 0.0,
            "realized_pnl": pnl,
            "total_notional": notional,
            "avg_edge_percent": (edge_sum / edge_count) if edge_count > 0 else 0.0,
        },
        "decisions": {
            "total": len(decisions),
            "counts": decision_counts,
        },
        "signals": {
            "total": len(signals),
            "by_source": signals_by_source,
        },
    }


async def _tool_get_trader_context(args: dict[str, Any]) -> dict[str, Any]:
    trader_id = str(args.get("trader_id") or "").strip()
    if not trader_id:
        raise RuntimeError("trader_id is required")
    decision_limit = max(1, min(int(args.get("decision_limit") or 40), 200))
    order_limit = max(1, min(int(args.get("order_limit") or 80), 300))
    signal_limit = max(1, min(int(args.get("signal_limit") or 60), 300))

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
                    .where(
                        TradeSignal.source.in_(
                            [
                                cfg["source_key"]
                                for cfg in _serialize_trader_source_configs(trader_row.source_configs_json)
                            ]
                        )
                    )
                    .order_by(desc(TradeSignal.created_at))
                    .limit(signal_limit)
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
                "edge_percent": float(row.edge_percent) if row.edge_percent is not None else None,
                "confidence": float(row.confidence) if row.confidence is not None else None,
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
    limit = max(1, min(int(args.get("limit") or 80), 500))

    async with AsyncSessionLocal() as session:
        strategy_row = await get_strategy_by_id_or_slug(session, strategy_key=strategy_key)
        if strategy_row is None:
            raise RuntimeError(f"Strategy '{strategy_key}' not found")
        await ensure_strategy_version_seeded(
            session,
            strategy=strategy_row,
            reason="tune_agent_seed_context",
            created_by="tune_agent",
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


async def _apply_tuned_param_updates(
    *,
    trader_id: str,
    updates: list[dict[str, Any]],
) -> dict[str, Any]:
    if not updates:
        return {
            "applied_count": 0,
            "applied_param_updates": [],
            "updated_trader": None,
        }

    async with AsyncSessionLocal() as session:
        trader_row = await session.get(Trader, trader_id)
        if trader_row is None:
            raise RuntimeError(f"Trader '{trader_id}' not found")

        source_configs = _serialize_trader_source_configs(trader_row.source_configs_json)
        applied_rows: list[dict[str, Any]] = []

        for update in updates:
            source_key = _normalize_source_key(update.get("source_key"))
            strategy_key = _normalize_strategy_key(update.get("strategy_key")) or ""
            params_patch = dict(update.get("params_patch") or {})
            if not source_key or not params_patch:
                continue

            matched_indexes: list[int] = []
            for index, config in enumerate(source_configs):
                if config["source_key"] != source_key:
                    continue
                if strategy_key and config["strategy_key"] != strategy_key:
                    continue
                matched_indexes.append(index)

            if not matched_indexes:
                continue

            for index in matched_indexes:
                config = source_configs[index]
                before_params = dict(config.get("strategy_params") or {})
                after_params = {**before_params, **params_patch}
                changed_keys = [
                    key
                    for key, value in params_patch.items()
                    if before_params.get(key) != value
                ]
                if not changed_keys:
                    continue
                source_configs[index] = {
                    **config,
                    "strategy_params": after_params,
                }
                applied_rows.append(
                    {
                        "source_key": source_key,
                        "strategy_key": config["strategy_key"],
                        "changed_keys": changed_keys,
                        "params_patch": params_patch,
                        "reason": str(update.get("reason") or "llm_tune").strip() or "llm_tune",
                    }
                )

        if not applied_rows:
            return {
                "applied_count": 0,
                "applied_param_updates": [],
                "updated_trader": None,
            }

        updated_trader = await update_trader(
            session,
            trader_id,
            {
                "source_configs": source_configs,
            },
        )
        if updated_trader is None:
            raise RuntimeError(f"Trader '{trader_id}' not found")

    return {
        "applied_count": len(applied_rows),
        "applied_param_updates": applied_rows,
        "updated_trader": updated_trader,
    }


def _tune_agent_tools() -> list[AgentTool]:
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
                    "signal_limit": {"type": "integer"},
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
            max_calls=3,
        ),
    ]


async def run_strategy_tune_agent(
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

    seeded_context = await _tool_get_trader_context(
        {
            "trader_id": normalized_trader,
            "decision_limit": 60,
            "order_limit": 120,
            "signal_limit": 80,
        }
    )
    seeded_summary = _summarize_trader_context(seeded_context)

    monitor_context = None
    normalized_monitor_job_id = str(monitor_job_id or "").strip() or None
    if normalized_monitor_job_id:
        try:
            monitor_context = await _tool_get_monitor_job_context(
                {
                    "job_id": normalized_monitor_job_id,
                    "max_alerts": 120,
                }
            )
        except Exception:
            monitor_context = {
                "job_id": normalized_monitor_job_id,
                "status": "unavailable",
            }

    base_context: dict[str, Any] = {
        "trader_id": normalized_trader,
        "monitor_job_id": normalized_monitor_job_id,
        "requested_at": utcnow().isoformat(),
        "seeded_summary": seeded_summary,
        "seeded_trader_context": seeded_context,
        "monitor_context": monitor_context,
    }

    system_prompt = (
        "You are the Homerun strategy tuning agent. "
        "Your job is to optimize trader strategy parameters for higher risk-adjusted PnL while preserving execution safety. "
        "Use tools to inspect trader context and strategies before recommending changes. "
        "Return strict JSON with keys: summary, issues_identified, actions_taken, suggested_next_steps, param_updates. "
        "param_updates must be an array of objects with keys: source_key, strategy_key (optional), params_patch, reason. "
        "Only include params_patch values that should be overwritten on the trader source strategy_params. "
        "Do not include markdown in the final answer."
    )
    query_payload = {
        "goal": normalized_prompt,
        "context": base_context,
        "instructions": [
            "Always inspect trader context and current source_configs before proposing updates.",
            "Prefer small high-confidence parameter moves over large speculative rewrites.",
            "When uncertain, keep updates minimal and explain why in suggested_next_steps.",
        ],
    }

    result = await run_agent_to_completion(
        system_prompt=system_prompt,
        query=json.dumps(query_payload, ensure_ascii=True),
        tools=_tune_agent_tools(),
        model=model,
        max_iterations=max(1, min(int(max_iterations or 12), 24)),
        session_type="strategy_tune_agent",
        market_id=None,
        opportunity_id=None,
    )

    answer = str(((result.get("result") or {}).get("answer")) or "").strip()
    parsed = _extract_json_payload(answer)
    normalized_updates = _normalize_param_updates(parsed.get("param_updates") if isinstance(parsed, dict) else None)
    applied = await _apply_tuned_param_updates(
        trader_id=normalized_trader,
        updates=normalized_updates,
    )

    return {
        "session_id": str(result.get("session_id") or ""),
        "answer": answer,
        "parsed": parsed,
        "applied_param_updates": applied["applied_param_updates"],
        "applied_param_update_count": int(applied["applied_count"]),
        "updated_trader": applied["updated_trader"],
        "raw": _json_safe(result),
    }
