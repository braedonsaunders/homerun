"""DB-backed state for trader orchestrator runtime and APIs."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import and_, desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from models.database import (
    AppSettings,
    TradeSignal,
    Trader,
    TraderConfigRevision,
    TraderDecision,
    TraderDecisionCheck,
    TraderEvent,
    TraderOrder,
    TraderPosition,
    TraderOrchestratorControl,
    TraderOrchestratorSnapshot,
    TraderSignalConsumption,
)
from services.trader_orchestrator.sources.registry import normalize_sources
from services.trader_orchestrator.strategies import list_strategy_keys
from services.trader_orchestrator.templates import (
    DEFAULT_GLOBAL_RISK,
    TRADER_TEMPLATES,
    get_template,
)
from utils.utcnow import utcnow
from utils.secrets import decrypt_secret

ORCHESTRATOR_CONTROL_ID = "default"
ORCHESTRATOR_SNAPSHOT_ID = "latest"
OPEN_ORDER_STATUSES = {"submitted", "executed", "open"}
PAPER_ACTIVE_ORDER_STATUSES = {"submitted", "executed", "open"}
LIVE_ACTIVE_ORDER_STATUSES = {"submitted", "executed", "open"}
CLEANUP_ELIGIBLE_ORDER_STATUSES = {"submitted", "executed", "open"}
RESOLVED_ORDER_STATUSES = {"resolved_win", "resolved_loss"}
ACTIVE_POSITION_STATUS = "open"
INACTIVE_POSITION_STATUS = "closed"
_STRATEGY_KEY_ALIASES = {
    "strategy.default": "crypto_15m",
    "default": "crypto_15m",
}
_RESUME_POLICY_VALUES = {"resume_full", "manage_only", "flatten_then_start"}


def _now() -> datetime:
    return utcnow()


def _new_id() -> str:
    return uuid.uuid4().hex


def _to_iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.replace(tzinfo=None).isoformat() + "Z"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _normalize_mode_key(mode: Any) -> str:
    key = str(mode or "").strip().lower()
    if key in {"paper", "live"}:
        return key
    return "other"


def _normalize_status_key(status: Any) -> str:
    return str(status or "").strip().lower()


def _active_statuses_for_mode(mode: Any) -> set[str]:
    mode_key = _normalize_mode_key(mode)
    if mode_key == "paper":
        return PAPER_ACTIVE_ORDER_STATUSES
    if mode_key == "live":
        return LIVE_ACTIVE_ORDER_STATUSES
    return OPEN_ORDER_STATUSES


def _is_active_order_status(mode: Any, status: Any) -> bool:
    return _normalize_status_key(status) in _active_statuses_for_mode(mode)


def _normalize_position_status(status: Any) -> str:
    value = str(status or "").strip().lower()
    if value == ACTIVE_POSITION_STATUS:
        return ACTIVE_POSITION_STATUS
    return INACTIVE_POSITION_STATUS


def _position_identity_key(mode: Any, market_id: Any, direction: Any) -> tuple[str, str, str]:
    return (
        _normalize_mode_key(mode),
        str(market_id or "").strip(),
        str(direction or "").strip().lower(),
    )


def _normalize_confidence_fraction(value: Any, default: float = 0.0) -> float:
    parsed = _safe_float(value, default)
    if parsed > 1.0:
        parsed = parsed / 100.0
    return max(0.0, min(1.0, parsed))


def _normalize_resume_policy(value: Any) -> str:
    policy = str(value or "").strip().lower()
    if policy in _RESUME_POLICY_VALUES:
        return policy
    return "resume_full"


def _normalize_strategy_key(value: Any) -> str:
    key = str(value or "").strip().lower()
    return _STRATEGY_KEY_ALIASES.get(key, key)


def _validate_strategy_key(strategy_key: str) -> None:
    valid_keys = set(list_strategy_keys())
    if not strategy_key:
        raise ValueError("strategy_key is required")
    if strategy_key not in valid_keys:
        allowed = ", ".join(sorted(valid_keys))
        raise ValueError(f"Unknown strategy_key '{strategy_key}'. Valid strategy keys: {allowed}")


def _default_control_settings() -> dict[str, Any]:
    return {
        "global_risk": dict(DEFAULT_GLOBAL_RISK),
        "trading_domains": ["event_markets", "crypto"],
        "enabled_strategies": [t["strategy_key"] for t in TRADER_TEMPLATES],
        "llm_verify_trades": False,
        "paper_account_id": None,
    }


def _serialize_control(row: TraderOrchestratorControl) -> dict[str, Any]:
    return {
        "id": row.id,
        "is_enabled": bool(row.is_enabled),
        "is_paused": bool(row.is_paused),
        "mode": str(row.mode or "paper"),
        "run_interval_seconds": int(row.run_interval_seconds or 2),
        "requested_run_at": _to_iso(row.requested_run_at),
        "kill_switch": bool(row.kill_switch),
        "settings": row.settings_json or {},
        "updated_at": _to_iso(row.updated_at),
    }


def _serialize_snapshot(row: TraderOrchestratorSnapshot) -> dict[str, Any]:
    return {
        "id": row.id,
        "updated_at": _to_iso(row.updated_at),
        "last_run_at": _to_iso(row.last_run_at),
        "running": bool(row.running),
        "enabled": bool(row.enabled),
        "current_activity": row.current_activity,
        "interval_seconds": int(row.interval_seconds or 2),
        "traders_total": int(row.traders_total or 0),
        "traders_running": int(row.traders_running or 0),
        "decisions_count": int(row.decisions_count or 0),
        "orders_count": int(row.orders_count or 0),
        "open_orders": int(row.open_orders or 0),
        "gross_exposure_usd": float(row.gross_exposure_usd or 0.0),
        "daily_pnl": float(row.daily_pnl or 0.0),
        "last_error": row.last_error,
        "stats": row.stats_json or {},
    }


def _serialize_trader(row: Trader) -> dict[str, Any]:
    metadata = dict(row.metadata_json or {})
    metadata["resume_policy"] = _normalize_resume_policy(metadata.get("resume_policy"))
    return {
        "id": row.id,
        "name": row.name,
        "description": row.description,
        "strategy_key": row.strategy_key,
        "strategy_version": row.strategy_version,
        "sources": list(row.sources_json or []),
        "params": row.params_json or {},
        "risk_limits": row.risk_limits_json or {},
        "metadata": metadata,
        "is_enabled": bool(row.is_enabled),
        "is_paused": bool(row.is_paused),
        "interval_seconds": int(row.interval_seconds or 60),
        "requested_run_at": _to_iso(row.requested_run_at),
        "last_run_at": _to_iso(row.last_run_at),
        "next_run_at": _to_iso(row.next_run_at),
        "created_at": _to_iso(row.created_at),
        "updated_at": _to_iso(row.updated_at),
    }


def _serialize_decision(row: TraderDecision) -> dict[str, Any]:
    return {
        "id": row.id,
        "trader_id": row.trader_id,
        "signal_id": row.signal_id,
        "source": row.source,
        "strategy_key": row.strategy_key,
        "decision": row.decision,
        "reason": row.reason,
        "score": row.score,
        "event_id": row.event_id,
        "trace_id": row.trace_id,
        "checks_summary": row.checks_summary_json or {},
        "risk_snapshot": row.risk_snapshot_json or {},
        "payload": row.payload_json or {},
        "created_at": _to_iso(row.created_at),
    }


def _serialize_order(row: TraderOrder) -> dict[str, Any]:
    return {
        "id": row.id,
        "trader_id": row.trader_id,
        "signal_id": row.signal_id,
        "decision_id": row.decision_id,
        "source": row.source,
        "market_id": row.market_id,
        "market_question": row.market_question,
        "direction": row.direction,
        "mode": row.mode,
        "status": row.status,
        "notional_usd": row.notional_usd,
        "entry_price": row.entry_price,
        "effective_price": row.effective_price,
        "edge_percent": row.edge_percent,
        "confidence": row.confidence,
        "actual_profit": row.actual_profit,
        "reason": row.reason,
        "payload": row.payload_json or {},
        "error_message": row.error_message,
        "event_id": row.event_id,
        "trace_id": row.trace_id,
        "created_at": _to_iso(row.created_at),
        "executed_at": _to_iso(row.executed_at),
        "updated_at": _to_iso(row.updated_at),
    }


def _serialize_event(row: TraderEvent) -> dict[str, Any]:
    return {
        "id": row.id,
        "trader_id": row.trader_id,
        "event_type": row.event_type,
        "severity": row.severity,
        "source": row.source,
        "operator": row.operator,
        "message": row.message,
        "trace_id": row.trace_id,
        "payload": row.payload_json or {},
        "created_at": _to_iso(row.created_at),
    }


async def ensure_orchestrator_control(session: AsyncSession) -> TraderOrchestratorControl:
    row = await session.get(TraderOrchestratorControl, ORCHESTRATOR_CONTROL_ID)
    if row is None:
        row = TraderOrchestratorControl(
            id=ORCHESTRATOR_CONTROL_ID,
            is_enabled=False,
            is_paused=True,
            mode="paper",
            run_interval_seconds=2,
            kill_switch=False,
            settings_json=_default_control_settings(),
            updated_at=_now(),
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
    elif not isinstance(row.settings_json, dict):
        row.settings_json = _default_control_settings()
        row.updated_at = _now()
        await session.commit()
        await session.refresh(row)
    return row


async def ensure_orchestrator_snapshot(session: AsyncSession) -> TraderOrchestratorSnapshot:
    row = await session.get(TraderOrchestratorSnapshot, ORCHESTRATOR_SNAPSHOT_ID)
    if row is None:
        row = TraderOrchestratorSnapshot(
            id=ORCHESTRATOR_SNAPSHOT_ID,
            running=False,
            enabled=False,
            current_activity="Waiting for trader orchestrator worker.",
            interval_seconds=2,
            stats_json={},
            updated_at=_now(),
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
    return row


async def read_orchestrator_control(session: AsyncSession) -> dict[str, Any]:
    return _serialize_control(await ensure_orchestrator_control(session))


async def read_orchestrator_snapshot(session: AsyncSession) -> dict[str, Any]:
    return _serialize_snapshot(await ensure_orchestrator_snapshot(session))


async def update_orchestrator_control(session: AsyncSession, **updates: Any) -> dict[str, Any]:
    row = await ensure_orchestrator_control(session)
    for field in ("is_enabled", "is_paused", "mode", "run_interval_seconds", "requested_run_at", "kill_switch"):
        if field in updates and updates[field] is not None:
            setattr(row, field, updates[field])
    if isinstance(updates.get("settings_json"), dict):
        merged = dict(row.settings_json or {})
        merged.update(updates["settings_json"])
        row.settings_json = merged
    row.updated_at = _now()
    await session.commit()
    await session.refresh(row)
    return _serialize_control(row)


async def write_orchestrator_snapshot(
    session: AsyncSession,
    *,
    running: bool,
    enabled: bool,
    current_activity: Optional[str],
    interval_seconds: Optional[int],
    last_run_at: Optional[datetime] = None,
    last_error: Optional[str] = None,
    stats: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    row = await ensure_orchestrator_snapshot(session)
    row.updated_at = _now()
    row.running = bool(running)
    row.enabled = bool(enabled)
    row.current_activity = current_activity
    if interval_seconds is not None:
        row.interval_seconds = max(1, int(interval_seconds))
    if last_run_at is not None:
        row.last_run_at = last_run_at
    row.last_error = last_error
    if isinstance(stats, dict):
        row.stats_json = stats
        row.traders_total = int(stats.get("traders_total", row.traders_total or 0) or 0)
        row.traders_running = int(stats.get("traders_running", row.traders_running or 0) or 0)
        row.decisions_count = int(stats.get("decisions_count", row.decisions_count or 0) or 0)
        row.orders_count = int(stats.get("orders_count", row.orders_count or 0) or 0)
        row.open_orders = int(stats.get("open_orders", row.open_orders or 0) or 0)
        row.gross_exposure_usd = float(stats.get("gross_exposure_usd", row.gross_exposure_usd or 0.0) or 0.0)
        row.daily_pnl = float(stats.get("daily_pnl", row.daily_pnl or 0.0) or 0.0)
    await session.commit()
    await session.refresh(row)
    return _serialize_snapshot(row)


def list_trader_templates() -> list[dict[str, Any]]:
    return [
        {
            "id": template["id"],
            "name": template["name"],
            "description": template.get("description"),
            "strategy_key": template["strategy_key"],
            "sources": template.get("sources", []),
            "interval_seconds": int(template.get("interval_seconds", 60) or 60),
            "params": template.get("params", {}),
            "risk_limits": template.get("risk_limits", {}),
        }
        for template in TRADER_TEMPLATES
    ]


def _normalize_trader_payload(payload: dict[str, Any]) -> dict[str, Any]:
    params = payload.get("params") or {}
    if not isinstance(params, dict):
        params = {}
    if "min_confidence" in params:
        params = dict(params)
        params["min_confidence"] = _normalize_confidence_fraction(
            params.get("min_confidence"),
            0.0,
        )
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = dict(metadata)
    metadata["resume_policy"] = _normalize_resume_policy(metadata.get("resume_policy"))

    return {
        "name": str(payload.get("name") or "").strip(),
        "description": payload.get("description"),
        "strategy_key": _normalize_strategy_key(payload.get("strategy_key")),
        "sources": normalize_sources(payload.get("sources")),
        "params": params,
        "risk_limits": payload.get("risk_limits") or {},
        "metadata": metadata,
        "is_enabled": bool(payload.get("is_enabled", True)),
        "is_paused": bool(payload.get("is_paused", False)),
        "interval_seconds": max(1, min(86400, _safe_int(payload.get("interval_seconds"), 60))),
    }


async def list_traders(session: AsyncSession) -> list[dict[str, Any]]:
    rows = (await session.execute(select(Trader).order_by(Trader.name.asc()))).scalars().all()
    return [_serialize_trader(row) for row in rows]


async def get_trader(session: AsyncSession, trader_id: str) -> Optional[dict[str, Any]]:
    row = await session.get(Trader, trader_id)
    return _serialize_trader(row) if row else None


async def seed_default_traders(session: AsyncSession) -> None:
    count = int((await session.execute(select(func.count(Trader.id)))).scalar() or 0)
    if count > 0:
        return

    for template in TRADER_TEMPLATES:
        session.add(
            Trader(
                id=_new_id(),
                name=template["name"],
                description=template.get("description"),
                strategy_key=template["strategy_key"],
                strategy_version="v1",
                sources_json=normalize_sources(template.get("sources") or []),
                params_json=template.get("params") or {},
                risk_limits_json=template.get("risk_limits") or {},
                metadata_json={"template_id": template["id"]},
                is_enabled=True,
                is_paused=False,
                interval_seconds=int(template.get("interval_seconds", 60) or 60),
                created_at=_now(),
                updated_at=_now(),
            )
        )
    await session.commit()


async def create_trader(session: AsyncSession, payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_trader_payload(payload)
    if not normalized["name"]:
        raise ValueError("Trader name is required")
    _validate_strategy_key(normalized["strategy_key"])

    existing = (
        (await session.execute(select(Trader).where(func.lower(Trader.name) == normalized["name"].lower())))
        .scalars()
        .first()
    )
    if existing is not None:
        raise ValueError("Trader name already exists")

    row = Trader(
        id=_new_id(),
        name=normalized["name"],
        description=normalized["description"],
        strategy_key=normalized["strategy_key"],
        strategy_version="v1",
        sources_json=normalized["sources"],
        params_json=normalized["params"],
        risk_limits_json=normalized["risk_limits"],
        metadata_json=normalized["metadata"],
        is_enabled=normalized["is_enabled"],
        is_paused=normalized["is_paused"],
        interval_seconds=normalized["interval_seconds"],
        created_at=_now(),
        updated_at=_now(),
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return _serialize_trader(row)


async def create_trader_from_template(
    session: AsyncSession,
    template_id: str,
    overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    template = get_template(template_id)
    if template is None:
        raise ValueError("Unknown template")

    payload: dict[str, Any] = {
        "name": template["name"],
        "description": template.get("description"),
        "strategy_key": template["strategy_key"],
        "sources": template.get("sources", []),
        "interval_seconds": template.get("interval_seconds", 60),
        "params": template.get("params", {}),
        "risk_limits": template.get("risk_limits", {}),
        "metadata": {"template_id": template_id},
        "is_enabled": True,
        "is_paused": False,
    }
    if isinstance(overrides, dict):
        payload.update(overrides)
    return await create_trader(session, payload)


async def update_trader(
    session: AsyncSession,
    trader_id: str,
    payload: dict[str, Any],
) -> Optional[dict[str, Any]]:
    row = await session.get(Trader, trader_id)
    if row is None:
        return None

    normalized = _normalize_trader_payload({**_serialize_trader(row), **payload})
    if "name" in payload:
        row.name = normalized["name"]
    if "description" in payload:
        row.description = normalized["description"]
    if "strategy_key" in payload:
        _validate_strategy_key(normalized["strategy_key"])
        row.strategy_key = normalized["strategy_key"]
    if "sources" in payload:
        row.sources_json = normalized["sources"]
    if "params" in payload:
        row.params_json = normalized["params"]
    if "risk_limits" in payload:
        row.risk_limits_json = normalized["risk_limits"]
    if "metadata" in payload:
        row.metadata_json = normalized["metadata"]
    if "is_enabled" in payload:
        row.is_enabled = bool(payload.get("is_enabled"))
    if "is_paused" in payload:
        row.is_paused = bool(payload.get("is_paused"))
    if "interval_seconds" in payload:
        row.interval_seconds = normalized["interval_seconds"]

    row.updated_at = _now()
    await session.commit()
    await session.refresh(row)
    return _serialize_trader(row)


async def delete_trader(session: AsyncSession, trader_id: str) -> bool:
    row = await session.get(Trader, trader_id)
    if row is None:
        return False
    await session.delete(row)
    await session.commit()
    return True


async def set_trader_paused(session: AsyncSession, trader_id: str, paused: bool) -> Optional[dict[str, Any]]:
    row = await session.get(Trader, trader_id)
    if row is None:
        return None
    row.is_paused = bool(paused)
    row.updated_at = _now()
    await session.commit()
    await session.refresh(row)
    return _serialize_trader(row)


async def request_trader_run(session: AsyncSession, trader_id: str) -> Optional[dict[str, Any]]:
    row = await session.get(Trader, trader_id)
    if row is None:
        return None
    row.requested_run_at = _now()
    row.updated_at = _now()
    await session.commit()
    await session.refresh(row)
    return _serialize_trader(row)


async def clear_trader_run_request(session: AsyncSession, trader_id: str) -> None:
    row = await session.get(Trader, trader_id)
    if row is None:
        return
    row.requested_run_at = None
    row.updated_at = _now()
    await session.commit()


async def create_config_revision(
    session: AsyncSession,
    *,
    trader_id: Optional[str],
    operator: Optional[str],
    reason: Optional[str],
    orchestrator_before: Optional[dict[str, Any]],
    orchestrator_after: Optional[dict[str, Any]],
    trader_before: Optional[dict[str, Any]],
    trader_after: Optional[dict[str, Any]],
) -> None:
    session.add(
        TraderConfigRevision(
            id=_new_id(),
            trader_id=trader_id,
            operator=operator,
            reason=reason,
            orchestrator_before_json=orchestrator_before or {},
            orchestrator_after_json=orchestrator_after or {},
            trader_before_json=trader_before or {},
            trader_after_json=trader_after or {},
            created_at=_now(),
        )
    )
    await session.commit()


async def create_trader_event(
    session: AsyncSession,
    *,
    event_type: str,
    severity: str = "info",
    trader_id: Optional[str] = None,
    source: Optional[str] = None,
    operator: Optional[str] = None,
    message: Optional[str] = None,
    trace_id: Optional[str] = None,
    payload: Optional[dict[str, Any]] = None,
) -> TraderEvent:
    row = TraderEvent(
        id=_new_id(),
        trader_id=trader_id,
        event_type=str(event_type),
        severity=str(severity or "info"),
        source=source,
        operator=operator,
        message=message,
        trace_id=trace_id,
        payload_json=payload or {},
        created_at=_now(),
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row


async def list_trader_events(
    session: AsyncSession,
    *,
    trader_id: Optional[str] = None,
    limit: int = 200,
    cursor: Optional[str] = None,
    event_types: Optional[list[str]] = None,
) -> tuple[list[TraderEvent], Optional[str]]:
    query = select(TraderEvent).order_by(desc(TraderEvent.created_at), desc(TraderEvent.id))
    if trader_id:
        query = query.where(TraderEvent.trader_id == trader_id)
    if event_types:
        query = query.where(TraderEvent.event_type.in_(event_types))
    if cursor:
        cursor_row = await session.get(TraderEvent, cursor)
        if cursor_row is not None:
            query = query.where(
                or_(
                    TraderEvent.created_at < cursor_row.created_at,
                    and_(TraderEvent.created_at == cursor_row.created_at, TraderEvent.id < cursor_row.id),
                )
            )
    query = query.limit(max(1, min(limit, 500)) + 1)
    rows = list((await session.execute(query)).scalars().all())
    next_cursor = None
    if len(rows) > limit:
        next_cursor = rows[-1].id
        rows = rows[:limit]
    return rows, next_cursor


async def list_trader_decisions(
    session: AsyncSession,
    *,
    trader_id: Optional[str] = None,
    decision: Optional[str] = None,
    limit: int = 200,
) -> list[TraderDecision]:
    query = select(TraderDecision).order_by(desc(TraderDecision.created_at))
    if trader_id:
        query = query.where(TraderDecision.trader_id == trader_id)
    if decision:
        query = query.where(TraderDecision.decision == decision)
    query = query.limit(max(1, min(limit, 1000)))
    return list((await session.execute(query)).scalars().all())


async def list_trader_orders(
    session: AsyncSession,
    *,
    trader_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 200,
) -> list[TraderOrder]:
    query = select(TraderOrder).order_by(desc(TraderOrder.created_at))
    if trader_id:
        query = query.where(TraderOrder.trader_id == trader_id)
    if status:
        query = query.where(TraderOrder.status == status)
    query = query.limit(max(1, min(limit, 1000)))
    return list((await session.execute(query)).scalars().all())


async def get_trader_decision_detail(session: AsyncSession, decision_id: str) -> Optional[dict[str, Any]]:
    row = await session.get(TraderDecision, decision_id)
    if row is None:
        return None

    checks = (
        (
            await session.execute(
                select(TraderDecisionCheck)
                .where(TraderDecisionCheck.decision_id == decision_id)
                .order_by(TraderDecisionCheck.created_at.asc())
            )
        )
        .scalars()
        .all()
    )
    orders = (
        (
            await session.execute(
                select(TraderOrder).where(TraderOrder.decision_id == decision_id).order_by(desc(TraderOrder.created_at))
            )
        )
        .scalars()
        .all()
    )

    return {
        "decision": _serialize_decision(row),
        "checks": [
            {
                "id": check.id,
                "check_key": check.check_key,
                "check_label": check.check_label,
                "passed": bool(check.passed),
                "score": check.score,
                "detail": check.detail,
                "payload": check.payload_json or {},
                "created_at": _to_iso(check.created_at),
            }
            for check in checks
        ],
        "orders": [_serialize_order(order) for order in orders],
    }


async def create_trader_decision(
    session: AsyncSession,
    *,
    trader_id: str,
    signal: TradeSignal,
    strategy_key: str,
    decision: str,
    reason: Optional[str] = None,
    score: Optional[float] = None,
    checks_summary: Optional[dict[str, Any]] = None,
    risk_snapshot: Optional[dict[str, Any]] = None,
    payload: Optional[dict[str, Any]] = None,
    trace_id: Optional[str] = None,
) -> TraderDecision:
    row = TraderDecision(
        id=_new_id(),
        trader_id=trader_id,
        signal_id=signal.id,
        source=str(signal.source),
        strategy_key=str(strategy_key),
        decision=str(decision),
        reason=reason,
        score=score,
        trace_id=trace_id,
        checks_summary_json=checks_summary or {},
        risk_snapshot_json=risk_snapshot or {},
        payload_json=payload or {},
        created_at=_now(),
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row


async def create_trader_decision_checks(
    session: AsyncSession,
    *,
    decision_id: str,
    checks: list[dict[str, Any]],
) -> None:
    if not checks:
        return
    for check in checks:
        session.add(
            TraderDecisionCheck(
                id=_new_id(),
                decision_id=decision_id,
                check_key=str(check.get("check_key") or check.get("key") or "check"),
                check_label=str(check.get("check_label") or check.get("label") or "Check"),
                passed=bool(check.get("passed", False)),
                score=check.get("score"),
                detail=check.get("detail"),
                payload_json=check.get("payload") or {},
                created_at=_now(),
            )
        )
    await session.commit()


async def create_trader_order(
    session: AsyncSession,
    *,
    trader_id: str,
    signal: TradeSignal,
    decision_id: Optional[str],
    mode: str,
    status: str,
    notional_usd: Optional[float],
    effective_price: Optional[float],
    reason: Optional[str],
    payload: Optional[dict[str, Any]],
    error_message: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> TraderOrder:
    row = TraderOrder(
        id=_new_id(),
        trader_id=trader_id,
        signal_id=signal.id,
        decision_id=decision_id,
        source=str(signal.source),
        market_id=str(signal.market_id),
        market_question=signal.market_question,
        direction=signal.direction,
        mode=str(mode),
        status=str(status),
        notional_usd=notional_usd,
        entry_price=signal.entry_price,
        effective_price=effective_price,
        edge_percent=signal.edge_percent,
        confidence=signal.confidence,
        reason=reason,
        payload_json=payload or {},
        error_message=error_message,
        trace_id=trace_id,
        created_at=_now(),
        executed_at=_now() if status in {"executed", "open"} else None,
        updated_at=_now(),
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    if _is_active_order_status(mode, status):
        await sync_trader_position_inventory(
            session,
            trader_id=trader_id,
            mode=str(mode),
        )
    return row


async def record_signal_consumption(
    session: AsyncSession,
    *,
    trader_id: str,
    signal_id: str,
    outcome: str,
    reason: Optional[str] = None,
    decision_id: Optional[str] = None,
    payload: Optional[dict[str, Any]] = None,
) -> None:
    existing = (
        (
            await session.execute(
                select(TraderSignalConsumption).where(
                    TraderSignalConsumption.trader_id == trader_id,
                    TraderSignalConsumption.signal_id == signal_id,
                )
            )
        )
        .scalars()
        .first()
    )
    if existing is not None:
        return

    session.add(
        TraderSignalConsumption(
            id=_new_id(),
            trader_id=trader_id,
            signal_id=signal_id,
            decision_id=decision_id,
            outcome=outcome,
            reason=reason,
            payload_json=payload or {},
            consumed_at=_now(),
        )
    )
    await session.commit()


async def list_unconsumed_trade_signals(
    session: AsyncSession,
    *,
    trader_id: str,
    sources: Optional[list[str]] = None,
    limit: int = 200,
) -> list[TradeSignal]:
    now = _now()
    consumed = (
        select(TraderSignalConsumption.signal_id).where(TraderSignalConsumption.trader_id == trader_id).subquery()
    )
    query = (
        select(TradeSignal)
        .where(~TradeSignal.id.in_(select(consumed.c.signal_id)))
        .where(or_(TradeSignal.expires_at.is_(None), TradeSignal.expires_at >= now))
        .order_by(TradeSignal.created_at.asc())
        .limit(max(1, min(limit, 1000)))
    )
    if sources is not None:
        normalized_sources = [str(source).strip().lower() for source in sources if str(source).strip()]
        if not normalized_sources:
            return []
        query = query.where(TradeSignal.source.in_(normalized_sources))
    return list((await session.execute(query)).scalars().all())


async def sync_trader_position_inventory(
    session: AsyncSession,
    *,
    trader_id: str,
    mode: Optional[str] = None,
) -> dict[str, Any]:
    query = select(TraderOrder).where(TraderOrder.trader_id == trader_id)
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")).not_in(["paper", "live"]))
        else:
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key)

    order_rows = list((await session.execute(query)).scalars().all())

    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in order_rows:
        mode_key = _normalize_mode_key(row.mode)
        if not _is_active_order_status(mode_key, row.status):
            continue
        identity = _position_identity_key(mode_key, row.market_id, row.direction)
        if not identity[1]:
            continue

        bucket = grouped.get(identity)
        if bucket is None:
            bucket = {
                "mode": mode_key,
                "market_id": str(row.market_id or ""),
                "market_question": row.market_question,
                "direction": str(row.direction or ""),
                "open_order_count": 0,
                "total_notional_usd": 0.0,
                "weighted_entry_numerator": 0.0,
                "weighted_entry_denominator": 0.0,
                "first_order_at": row.created_at,
                "last_order_at": row.updated_at or row.executed_at or row.created_at,
            }
            grouped[identity] = bucket

        notional = abs(_safe_float(row.notional_usd, 0.0))
        entry_price = _safe_float(row.effective_price, 0.0) or _safe_float(row.entry_price, 0.0)
        bucket["open_order_count"] = int(bucket["open_order_count"]) + 1
        bucket["total_notional_usd"] = float(bucket["total_notional_usd"]) + max(0.0, notional)
        if entry_price and entry_price > 0 and notional > 0:
            bucket["weighted_entry_numerator"] = float(bucket["weighted_entry_numerator"]) + (entry_price * notional)
            bucket["weighted_entry_denominator"] = float(bucket["weighted_entry_denominator"]) + notional
        if row.created_at and (bucket["first_order_at"] is None or row.created_at < bucket["first_order_at"]):
            bucket["first_order_at"] = row.created_at
        last_order_at = row.updated_at or row.executed_at or row.created_at
        if last_order_at and (bucket["last_order_at"] is None or last_order_at > bucket["last_order_at"]):
            bucket["last_order_at"] = last_order_at

    existing_query = select(TraderPosition).where(TraderPosition.trader_id == trader_id)
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            existing_query = existing_query.where(func.lower(func.coalesce(TraderPosition.mode, "")).not_in(["paper", "live"]))
        else:
            existing_query = existing_query.where(func.lower(func.coalesce(TraderPosition.mode, "")) == mode_key)
    existing_rows = list((await session.execute(existing_query)).scalars().all())
    existing_by_identity = {
        _position_identity_key(row.mode, row.market_id, row.direction): row
        for row in existing_rows
    }

    now = _now()
    updates = 0
    inserts = 0
    closures = 0

    for identity, bucket in grouped.items():
        row = existing_by_identity.get(identity)
        avg_entry_price = None
        weighted_den = float(bucket.get("weighted_entry_denominator") or 0.0)
        if weighted_den > 0:
            avg_entry_price = float(bucket.get("weighted_entry_numerator") or 0.0) / weighted_den

        if row is None:
            row = TraderPosition(
                id=_new_id(),
                trader_id=trader_id,
                mode=str(bucket["mode"]),
                market_id=str(bucket["market_id"]),
                market_question=bucket.get("market_question"),
                direction=str(bucket.get("direction") or ""),
                status=ACTIVE_POSITION_STATUS,
                open_order_count=int(bucket.get("open_order_count") or 0),
                total_notional_usd=float(bucket.get("total_notional_usd") or 0.0),
                avg_entry_price=avg_entry_price,
                first_order_at=bucket.get("first_order_at"),
                last_order_at=bucket.get("last_order_at"),
                closed_at=None,
                payload_json={"sync_source": "order_inventory"},
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            inserts += 1
            continue

        row.market_question = bucket.get("market_question")
        row.status = ACTIVE_POSITION_STATUS
        row.open_order_count = int(bucket.get("open_order_count") or 0)
        row.total_notional_usd = float(bucket.get("total_notional_usd") or 0.0)
        row.avg_entry_price = avg_entry_price
        row.first_order_at = bucket.get("first_order_at")
        row.last_order_at = bucket.get("last_order_at")
        row.closed_at = None
        row.updated_at = now
        updates += 1

    grouped_keys = set(grouped.keys())
    for identity, row in existing_by_identity.items():
        if identity in grouped_keys:
            continue
        if _normalize_position_status(row.status) != ACTIVE_POSITION_STATUS:
            continue
        row.status = INACTIVE_POSITION_STATUS
        row.open_order_count = 0
        row.total_notional_usd = 0.0
        row.closed_at = now
        row.updated_at = now
        closures += 1

    if inserts > 0 or updates > 0 or closures > 0:
        await session.commit()

    return {
        "trader_id": trader_id,
        "mode": _normalize_mode_key(mode) if mode is not None else "all",
        "open_positions": len(grouped),
        "inserts": inserts,
        "updates": updates,
        "closures": closures,
    }


async def get_open_position_count_for_trader(
    session: AsyncSession,
    trader_id: str,
    mode: Optional[str] = None,
) -> int:
    query = (
        select(func.count(TraderPosition.id))
        .where(TraderPosition.trader_id == trader_id)
        .where(func.lower(func.coalesce(TraderPosition.status, "")) == ACTIVE_POSITION_STATUS)
    )
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            query = query.where(func.lower(func.coalesce(TraderPosition.mode, "")).not_in(["paper", "live"]))
        else:
            query = query.where(func.lower(func.coalesce(TraderPosition.mode, "")) == mode_key)
    return int((await session.execute(query)).scalar() or 0)


async def get_open_position_summary_for_trader(session: AsyncSession, trader_id: str) -> dict[str, int]:
    rows = (
        await session.execute(
            select(
                TraderPosition.mode,
                func.count(TraderPosition.id).label("count"),
            )
            .where(TraderPosition.trader_id == trader_id)
            .where(func.lower(func.coalesce(TraderPosition.status, "")) == ACTIVE_POSITION_STATUS)
            .group_by(TraderPosition.mode)
        )
    ).all()

    summary = {"live": 0, "paper": 0, "other": 0, "total": 0}
    for row in rows:
        mode_key = _normalize_mode_key(row.mode)
        count = int(row.count or 0)
        if mode_key == "live":
            summary["live"] += count
        elif mode_key == "paper":
            summary["paper"] += count
        else:
            summary["other"] += count
        summary["total"] += count
    return summary


async def get_open_order_count_for_trader(
    session: AsyncSession,
    trader_id: str,
    mode: Optional[str] = None,
) -> int:
    query = (
        select(
            TraderOrder.mode,
            TraderOrder.status,
            func.count(TraderOrder.id).label("count"),
        )
        .where(TraderOrder.trader_id == trader_id)
        .group_by(TraderOrder.mode, TraderOrder.status)
    )
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == "")
        else:
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key)

    rows = (await session.execute(query)).all()
    total = 0
    for row in rows:
        mode_key = _normalize_mode_key(mode if mode is not None else row.mode)
        if _is_active_order_status(mode_key, row.status):
            total += int(row.count or 0)
    return total


async def get_open_order_summary_for_trader(session: AsyncSession, trader_id: str) -> dict[str, int]:
    rows = (
        await session.execute(
            select(
                TraderOrder.mode,
                TraderOrder.status,
                func.count(TraderOrder.id).label("count"),
            )
            .where(TraderOrder.trader_id == trader_id)
            .group_by(TraderOrder.mode, TraderOrder.status)
        )
    ).all()

    summary = {"live": 0, "paper": 0, "other": 0, "total": 0}
    for row in rows:
        mode = _normalize_mode_key(row.mode)
        if not _is_active_order_status(mode, row.status):
            continue
        count = int(row.count or 0)
        if mode == "live":
            summary["live"] += count
        elif mode == "paper":
            summary["paper"] += count
        else:
            summary["other"] += count
        summary["total"] += count
    return summary


async def cleanup_trader_open_orders(
    session: AsyncSession,
    *,
    trader_id: str,
    scope: str = "paper",
    max_age_hours: Optional[int] = None,
    dry_run: bool = False,
    target_status: str = "cancelled",
    reason: Optional[str] = None,
) -> dict[str, Any]:
    scope_key = str(scope or "paper").strip().lower()
    if scope_key == "all":
        allowed_modes = {"paper", "live", "other"}
    elif scope_key in {"paper", "live"}:
        allowed_modes = {scope_key}
    else:
        raise ValueError("scope must be one of: paper, live, all")

    query = select(TraderOrder).where(TraderOrder.trader_id == trader_id)
    rows = list((await session.execute(query)).scalars().all())
    cutoff: Optional[datetime] = None
    if max_age_hours is not None:
        cutoff = _now() - timedelta(hours=max(1, int(max_age_hours)))

    candidates: list[TraderOrder] = []
    for row in rows:
        mode_key = _normalize_mode_key(row.mode)
        if mode_key not in allowed_modes:
            continue

        status_key = _normalize_status_key(row.status)
        if status_key not in CLEANUP_ELIGIBLE_ORDER_STATUSES:
            continue

        if cutoff is not None:
            age_anchor = row.executed_at or row.updated_at or row.created_at
            if age_anchor is None or age_anchor > cutoff:
                continue

        candidates.append(row)

    mode_breakdown = {"paper": 0, "live": 0, "other": 0}
    status_breakdown: dict[str, int] = {}
    for row in candidates:
        mode_key = _normalize_mode_key(row.mode)
        status_key = _normalize_status_key(row.status)
        mode_breakdown[mode_key] = int(mode_breakdown.get(mode_key, 0)) + 1
        status_breakdown[status_key] = int(status_breakdown.get(status_key, 0)) + 1

    updated = 0
    if not dry_run and candidates:
        now = _now()
        note_reason = str(reason or "manual_position_cleanup").strip()
        for row in candidates:
            previous_status = str(row.status or "")
            row.status = target_status
            row.updated_at = now
            existing_payload = dict(row.payload_json or {})
            existing_payload["cleanup"] = {
                "previous_status": previous_status,
                "target_status": target_status,
                "reason": note_reason,
                "performed_at": _to_iso(now),
            }
            row.payload_json = existing_payload
            if note_reason:
                if row.reason:
                    row.reason = f"{row.reason} | cleanup:{note_reason}"
                else:
                    row.reason = f"cleanup:{note_reason}"
            updated += 1
        await session.commit()
        await sync_trader_position_inventory(session, trader_id=trader_id)

    return {
        "trader_id": trader_id,
        "scope": scope_key,
        "max_age_hours": max_age_hours,
        "dry_run": bool(dry_run),
        "target_status": target_status,
        "matched": len(candidates),
        "updated": updated,
        "by_mode": mode_breakdown,
        "by_status": status_breakdown,
    }


async def get_market_exposure(session: AsyncSession, market_id: str, mode: Optional[str] = None) -> float:
    query = (
        select(
            TraderOrder.mode,
            TraderOrder.status,
            func.coalesce(func.sum(TraderOrder.notional_usd), 0.0).label("notional"),
        )
        .where(TraderOrder.market_id == market_id)
        .group_by(TraderOrder.mode, TraderOrder.status)
    )
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == "")
        else:
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key)

    rows = (await session.execute(query)).all()
    total = 0.0
    for row in rows:
        mode_key = _normalize_mode_key(mode if mode is not None else row.mode)
        if _is_active_order_status(mode_key, row.status):
            total += float(row.notional or 0.0)
    return total


async def get_gross_exposure(session: AsyncSession, mode: Optional[str] = None) -> float:
    query = (
        select(
            TraderOrder.mode,
            TraderOrder.status,
            func.coalesce(func.sum(func.abs(TraderOrder.notional_usd)), 0.0).label("notional_abs"),
        )
        .group_by(TraderOrder.mode, TraderOrder.status)
    )
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == "")
        else:
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key)

    rows = (await session.execute(query)).all()
    total = 0.0
    for row in rows:
        mode_key = _normalize_mode_key(mode if mode is not None else row.mode)
        if _is_active_order_status(mode_key, row.status):
            total += float(row.notional_abs or 0.0)
    return total


async def get_realized_pnl(
    session: AsyncSession,
    *,
    trader_id: Optional[str] = None,
    mode: Optional[str] = None,
    since: Optional[datetime] = None,
) -> float:
    query = select(func.coalesce(func.sum(TraderOrder.actual_profit), 0.0)).where(
        TraderOrder.status.in_(tuple(RESOLVED_ORDER_STATUSES))
    )
    if trader_id:
        query = query.where(TraderOrder.trader_id == trader_id)
    if since is not None:
        query = query.where(TraderOrder.updated_at >= since)
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == "")
        else:
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key)
    return float((await session.execute(query)).scalar() or 0.0)


async def get_daily_realized_pnl(
    session: AsyncSession,
    *,
    trader_id: Optional[str] = None,
    mode: Optional[str] = None,
) -> float:
    today_start = _now().replace(hour=0, minute=0, second=0, microsecond=0)
    return await get_realized_pnl(
        session,
        trader_id=trader_id,
        mode=mode,
        since=today_start,
    )


async def get_consecutive_loss_count(
    session: AsyncSession,
    *,
    trader_id: str,
    mode: Optional[str] = None,
    limit: int = 100,
) -> int:
    query = (
        select(TraderOrder.status, TraderOrder.updated_at)
        .where(TraderOrder.trader_id == trader_id)
        .where(TraderOrder.status.in_(tuple(RESOLVED_ORDER_STATUSES)))
        .order_by(desc(TraderOrder.updated_at), desc(TraderOrder.id))
        .limit(max(1, min(int(limit or 100), 1000)))
    )
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == "")
        else:
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key)

    rows = (await session.execute(query)).all()
    losses = 0
    for row in rows:
        status = _normalize_status_key(row.status)
        if status == "resolved_loss":
            losses += 1
            continue
        if status == "resolved_win":
            break
    return losses


async def get_last_resolved_loss_at(
    session: AsyncSession,
    *,
    trader_id: str,
    mode: Optional[str] = None,
) -> Optional[datetime]:
    query = select(func.max(TraderOrder.updated_at)).where(
        TraderOrder.trader_id == trader_id,
        TraderOrder.status == "resolved_loss",
    )
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == "")
        else:
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key)
    return (await session.execute(query)).scalar_one_or_none()


async def compute_orchestrator_metrics(session: AsyncSession) -> dict[str, Any]:
    traders_total = int((await session.execute(select(func.count(Trader.id)))).scalar() or 0)
    traders_running = int(
        (
            await session.execute(
                select(func.count(Trader.id)).where(
                    Trader.is_enabled == True,  # noqa: E712
                    Trader.is_paused == False,  # noqa: E712
                )
            )
        ).scalar()
        or 0
    )
    decisions_count = int((await session.execute(select(func.count(TraderDecision.id)))).scalar() or 0)
    orders_count = int((await session.execute(select(func.count(TraderOrder.id)))).scalar() or 0)
    open_rows = (
        await session.execute(
            select(
                TraderOrder.mode,
                TraderOrder.status,
                func.count(TraderOrder.id).label("count"),
            ).group_by(TraderOrder.mode, TraderOrder.status)
        )
    ).all()
    open_orders = 0
    for row in open_rows:
        if _is_active_order_status(row.mode, row.status):
            open_orders += int(row.count or 0)
    daily_pnl = await get_daily_realized_pnl(session)

    return {
        "traders_total": traders_total,
        "traders_running": traders_running,
        "decisions_count": decisions_count,
        "orders_count": orders_count,
        "open_orders": open_orders,
        "gross_exposure_usd": await get_gross_exposure(session),
        "daily_pnl": daily_pnl,
    }


async def compose_trader_orchestrator_config(session: AsyncSession) -> dict[str, Any]:
    control = await read_orchestrator_control(session)
    settings_json = control.get("settings") or {}
    global_risk = settings_json.get("global_risk") or dict(DEFAULT_GLOBAL_RISK)
    return {
        "mode": control.get("mode", "paper"),
        "kill_switch": bool(control.get("kill_switch", False)),
        "run_interval_seconds": int(control.get("run_interval_seconds") or 2),
        "global_risk": {
            "max_gross_exposure_usd": _safe_float(global_risk.get("max_gross_exposure_usd"), 5000.0),
            "max_daily_loss_usd": _safe_float(global_risk.get("max_daily_loss_usd"), 500.0),
            "max_orders_per_cycle": _safe_int(global_risk.get("max_orders_per_cycle"), 50),
        },
        "trading_domains": settings_json.get("trading_domains") or ["event_markets", "crypto"],
        "enabled_strategies": settings_json.get("enabled_strategies") or [],
        "llm_verify_trades": bool(settings_json.get("llm_verify_trades", False)),
        "paper_account_id": settings_json.get("paper_account_id"),
    }


async def get_orchestrator_overview(session: AsyncSession) -> dict[str, Any]:
    return {
        "control": await read_orchestrator_control(session),
        "worker": await read_orchestrator_snapshot(session),
        "config": await compose_trader_orchestrator_config(session),
        "metrics": await compute_orchestrator_metrics(session),
        "traders": await list_traders(session),
    }


async def _build_preflight_checks(
    session: AsyncSession,
    control: dict[str, Any],
    trader_count: int,
) -> list[dict[str, Any]]:
    app_settings = await session.get(AppSettings, "default")
    db_trading_enabled = bool(app_settings.trading_enabled) if app_settings is not None else False
    polymarket_ready = bool(
        app_settings is not None
        and decrypt_secret(app_settings.polymarket_api_key)
        and decrypt_secret(app_settings.polymarket_api_secret)
        and decrypt_secret(app_settings.polymarket_api_passphrase)
    )
    kalshi_ready = bool(
        app_settings is not None
        and (app_settings.kalshi_email or "").strip()
        and decrypt_secret(app_settings.kalshi_password)
        and decrypt_secret(app_settings.kalshi_api_key)
    )
    live_creds_ready = polymarket_ready or kalshi_ready

    return [
        {
            "id": "trading_enabled_env",
            "ok": bool(settings.TRADING_ENABLED),
            "message": "TRADING_ENABLED must be true in environment config",
        },
        {
            "id": "trading_enabled_setting",
            "ok": db_trading_enabled,
            "message": "trading_enabled must be true in app settings",
        },
        {
            "id": "live_credentials_configured",
            "ok": live_creds_ready,
            "message": "At least one live venue credential set must be configured",
            "polymarket_ready": polymarket_ready,
            "kalshi_ready": kalshi_ready,
        },
        {
            "id": "kill_switch_clear",
            "ok": not bool(control.get("kill_switch")),
            "message": "Kill switch must be disabled",
        },
        {
            "id": "traders_configured",
            "ok": trader_count > 0,
            "message": "At least one trader must exist",
            "trader_count": trader_count,
        },
    ]


async def create_live_preflight(
    session: AsyncSession,
    *,
    requested_mode: str,
    requested_by: Optional[str],
) -> dict[str, Any]:
    control_row = await ensure_orchestrator_control(session)
    control = _serialize_control(control_row)
    trader_count = int((await session.execute(select(func.count(Trader.id)))).scalar() or 0)
    checks = await _build_preflight_checks(session, control, trader_count)
    failed = [check for check in checks if not check["ok"]]
    status = "passed" if not failed else "failed"

    preflight = {
        "preflight_id": _new_id(),
        "requested_mode": requested_mode,
        "requested_by": requested_by,
        "status": status,
        "checks": checks,
        "failed_checks": failed,
        "created_at": _to_iso(_now()),
    }

    settings_json = dict(control_row.settings_json or {})
    settings_json["live_preflight"] = preflight
    control_row.settings_json = settings_json
    control_row.updated_at = _now()
    await session.commit()

    await create_trader_event(
        session,
        event_type="live_preflight",
        severity="info" if status == "passed" else "warn",
        source="trader_orchestrator",
        operator=requested_by,
        message=f"Live preflight {status}",
        payload=preflight,
    )
    return preflight


async def arm_live_start(
    session: AsyncSession,
    *,
    preflight_id: str,
    ttl_seconds: int,
    requested_by: Optional[str],
) -> dict[str, Any]:
    control_row = await ensure_orchestrator_control(session)
    settings_json = dict(control_row.settings_json or {})
    preflight = settings_json.get("live_preflight") or {}

    if str(preflight.get("preflight_id")) != str(preflight_id):
        raise ValueError("Unknown preflight_id")
    if str(preflight.get("status")) != "passed":
        raise ValueError("Preflight did not pass")

    arm_token = _new_id()
    expires_at = _now() + timedelta(seconds=max(30, min(ttl_seconds, 1800)))
    arm_data = {
        "arm_token": arm_token,
        "expires_at": _to_iso(expires_at),
        "consumed_at": None,
        "requested_by": requested_by,
    }

    settings_json["live_arm"] = arm_data
    control_row.settings_json = settings_json
    control_row.updated_at = _now()
    await session.commit()

    await create_trader_event(
        session,
        event_type="live_arm",
        source="trader_orchestrator",
        operator=requested_by,
        message="Live start token issued",
        payload={"preflight_id": preflight_id, "expires_at": arm_data["expires_at"]},
    )

    return {
        "preflight_id": preflight_id,
        "arm_token": arm_token,
        "expires_at": arm_data["expires_at"],
    }


async def consume_live_arm_token(session: AsyncSession, arm_token: str) -> bool:
    control_row = await ensure_orchestrator_control(session)
    settings_json = dict(control_row.settings_json or {})
    arm_data = settings_json.get("live_arm") or {}

    if str(arm_data.get("arm_token")) != str(arm_token):
        return False
    if arm_data.get("consumed_at"):
        return False

    expires_at_raw = arm_data.get("expires_at")
    expires_at = None
    if isinstance(expires_at_raw, str):
        try:
            expires_at = datetime.fromisoformat(expires_at_raw.replace("Z", "+00:00"))
        except Exception:
            expires_at = None

    if expires_at is not None and _now().astimezone(timezone.utc) > expires_at.astimezone(timezone.utc):
        return False

    arm_data["consumed_at"] = _to_iso(_now())
    settings_json["live_arm"] = arm_data
    control_row.settings_json = settings_json
    control_row.updated_at = _now()
    await session.commit()
    return True


async def list_serialized_trader_decisions(
    session: AsyncSession,
    *,
    trader_id: Optional[str] = None,
    decision: Optional[str] = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    return [
        _serialize_decision(row)
        for row in await list_trader_decisions(
            session,
            trader_id=trader_id,
            decision=decision,
            limit=limit,
        )
    ]


async def list_serialized_trader_orders(
    session: AsyncSession,
    *,
    trader_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    return [
        _serialize_order(row)
        for row in await list_trader_orders(
            session,
            trader_id=trader_id,
            status=status,
            limit=limit,
        )
    ]


async def list_serialized_trader_events(
    session: AsyncSession,
    *,
    trader_id: Optional[str] = None,
    limit: int = 200,
    cursor: Optional[str] = None,
    event_types: Optional[list[str]] = None,
) -> tuple[list[dict[str, Any]], Optional[str]]:
    rows, next_cursor = await list_trader_events(
        session,
        trader_id=trader_id,
        limit=limit,
        cursor=cursor,
        event_types=event_types,
    )
    return ([_serialize_event(row) for row in rows], next_cursor)
