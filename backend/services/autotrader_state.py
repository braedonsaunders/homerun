"""DB-backed state for the dedicated autotrader worker and API."""

from __future__ import annotations

from collections import defaultdict
import uuid
from datetime import datetime, timedelta, timezone
from utils.utcnow import utcnow
from typing import Any, Optional

from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    AutoTraderControl,
    AutoTraderDecision,
    AutoTraderPolicy,
    AutoTraderSnapshot,
    AutoTraderTrade,
    TradeSignal,
)


AUTOTRADER_CONTROL_ID = "default"
AUTOTRADER_SNAPSHOT_ID = "latest"

DEFAULT_SOURCE_POLICIES: dict[str, dict[str, Any]] = {
    "scanner": {
        "enabled": True,
        "weight": 1.0,
        "daily_budget_usd": 100.0,
        "max_open_positions": 8,
        "min_signal_score": 0.15,
        "size_multiplier": 1.0,
        "cooldown_seconds": 30,
    },
    "news": {
        "enabled": True,
        "weight": 1.1,
        "daily_budget_usd": 75.0,
        "max_open_positions": 6,
        "min_signal_score": 0.2,
        "size_multiplier": 1.0,
        "cooldown_seconds": 30,
    },
    "weather": {
        "enabled": True,
        "weight": 1.0,
        "daily_budget_usd": 75.0,
        "max_open_positions": 6,
        "min_signal_score": 0.2,
        "size_multiplier": 1.0,
        "cooldown_seconds": 60,
    },
    "crypto": {
        "enabled": True,
        "weight": 1.3,
        "daily_budget_usd": 150.0,
        "max_open_positions": 12,
        "min_signal_score": 0.25,
        "size_multiplier": 1.2,
        "cooldown_seconds": 5,
    },
    "tracked_traders": {
        "enabled": True,
        "weight": 1.2,
        "daily_budget_usd": 120.0,
        "max_open_positions": 8,
        "min_signal_score": 0.25,
        "size_multiplier": 1.0,
        "cooldown_seconds": 15,
    },
    "copy": {
        "enabled": True,
        "weight": 1.0,
        "daily_budget_usd": 100.0,
        "max_open_positions": 8,
        "min_signal_score": 0.25,
        "size_multiplier": 1.0,
        "cooldown_seconds": 10,
    },
    "insider": {
        "enabled": True,
        "weight": 1.25,
        "daily_budget_usd": 60.0,
        "max_open_positions": 4,
        "min_signal_score": 0.55,
        "size_multiplier": 0.8,
        "cooldown_seconds": 120,
        "metadata_json": {"live_enabled": False},
    },
}

DEFAULT_GLOBAL_POLICY: dict[str, Any] = {
    "enabled": True,
    "weight": 1.0,
    "daily_budget_usd": 1000.0,
    "max_open_positions": 50,
    "min_signal_score": 0.0,
    "size_multiplier": 1.0,
    "cooldown_seconds": 0,
    "max_daily_loss": 200.0,
    "max_total_open_positions": 20,
    "max_per_market_exposure": 100.0,
    "max_per_event_exposure": 150.0,
    "kill_switch": False,
}


def _now() -> datetime:
    return utcnow()


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(tzinfo=None).isoformat() + "Z"


async def ensure_autotrader_control(session: AsyncSession) -> AutoTraderControl:
    result = await session.execute(
        select(AutoTraderControl).where(AutoTraderControl.id == AUTOTRADER_CONTROL_ID)
    )
    row = result.scalar_one_or_none()
    if row is None:
        row = AutoTraderControl(
            id=AUTOTRADER_CONTROL_ID,
            is_enabled=False,
            is_paused=True,
            mode="paper",
            run_interval_seconds=2,
            requested_run_at=None,
            kill_switch=False,
            settings_json={},
            updated_at=_now(),
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
    return row


async def read_autotrader_control(session: AsyncSession) -> dict[str, Any]:
    row = await ensure_autotrader_control(session)
    return {
        "id": row.id,
        "is_enabled": bool(row.is_enabled),
        "is_paused": bool(row.is_paused),
        "mode": row.mode or "paper",
        "run_interval_seconds": int(row.run_interval_seconds or 2),
        "requested_run_at": _iso(row.requested_run_at),
        "kill_switch": bool(row.kill_switch),
        "settings": row.settings_json or {},
        "updated_at": _iso(row.updated_at),
    }


async def update_autotrader_control(
    session: AsyncSession,
    *,
    is_enabled: Optional[bool] = None,
    is_paused: Optional[bool] = None,
    mode: Optional[str] = None,
    run_interval_seconds: Optional[int] = None,
    requested_run: Optional[bool] = None,
    kill_switch: Optional[bool] = None,
    settings: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    row = await ensure_autotrader_control(session)
    if is_enabled is not None:
        row.is_enabled = bool(is_enabled)
    if is_paused is not None:
        row.is_paused = bool(is_paused)
    if mode is not None:
        row.mode = mode
    if run_interval_seconds is not None:
        row.run_interval_seconds = max(1, min(60, int(run_interval_seconds)))
    if requested_run is not None:
        row.requested_run_at = _now() if requested_run else None
    if kill_switch is not None:
        row.kill_switch = bool(kill_switch)
    if settings is not None:
        row.settings_json = settings
    row.updated_at = _now()
    await session.commit()
    return await read_autotrader_control(session)


async def clear_autotrader_run_request(session: AsyncSession) -> None:
    row = await ensure_autotrader_control(session)
    row.requested_run_at = None
    row.updated_at = _now()
    await session.commit()


async def reset_autotrader_for_manual_start(session: AsyncSession) -> dict[str, Any]:
    """Reset control flags so the worker always starts in manual mode on launch."""
    row = await ensure_autotrader_control(session)
    row.is_enabled = False
    row.is_paused = True
    row.requested_run_at = None
    row.updated_at = _now()
    await session.commit()
    return await read_autotrader_control(session)


async def ensure_default_autotrader_policies(session: AsyncSession) -> None:
    # Global row
    result = await session.execute(
        select(AutoTraderPolicy).where(AutoTraderPolicy.policy_key == "global")
    )
    global_row = result.scalar_one_or_none()
    if global_row is None:
        global_row = AutoTraderPolicy(policy_key="global", source=None)
        session.add(global_row)
        for key, value in DEFAULT_GLOBAL_POLICY.items():
            setattr(global_row, key, value)

    for source, default_policy in DEFAULT_SOURCE_POLICIES.items():
        policy_key = f"source:{source}"
        result = await session.execute(
            select(AutoTraderPolicy).where(AutoTraderPolicy.policy_key == policy_key)
        )
        row = result.scalar_one_or_none()
        if row is None:
            row = AutoTraderPolicy(policy_key=policy_key, source=source)
            session.add(row)
            for key, value in default_policy.items():
                setattr(row, key, value)

    await session.commit()


async def read_autotrader_policies(session: AsyncSession) -> dict[str, Any]:
    await ensure_default_autotrader_policies(session)
    result = await session.execute(
        select(AutoTraderPolicy).order_by(AutoTraderPolicy.policy_key.asc())
    )
    rows = list(result.scalars().all())

    global_policy: dict[str, Any] = {}
    sources: dict[str, dict[str, Any]] = {}

    for row in rows:
        payload = {
            "enabled": bool(row.enabled),
            "weight": float(row.weight or 0.0),
            "daily_budget_usd": float(row.daily_budget_usd or 0.0),
            "max_open_positions": int(row.max_open_positions or 0),
            "min_signal_score": float(row.min_signal_score or 0.0),
            "size_multiplier": float(row.size_multiplier or 1.0),
            "cooldown_seconds": int(row.cooldown_seconds or 0),
            "max_daily_loss": row.max_daily_loss,
            "max_total_open_positions": row.max_total_open_positions,
            "max_per_market_exposure": row.max_per_market_exposure,
            "max_per_event_exposure": row.max_per_event_exposure,
            "kill_switch": row.kill_switch,
            "metadata": row.metadata_json or {},
            "updated_at": _iso(row.updated_at),
        }

        if row.policy_key == "global":
            global_policy = payload
        elif row.source:
            sources[row.source] = payload

    return {"global": global_policy, "sources": sources}


async def upsert_autotrader_policies(
    session: AsyncSession,
    payload: dict[str, Any],
) -> dict[str, Any]:
    await ensure_default_autotrader_policies(session)

    global_update = payload.get("global") if isinstance(payload, dict) else None
    if isinstance(global_update, dict):
        result = await session.execute(
            select(AutoTraderPolicy).where(AutoTraderPolicy.policy_key == "global")
        )
        row = result.scalar_one()
        for key, value in global_update.items():
            if key == "metadata":
                row.metadata_json = value
            elif hasattr(row, key):
                setattr(row, key, value)
        row.updated_at = _now()

    source_updates = payload.get("sources") if isinstance(payload, dict) else None
    if isinstance(source_updates, dict):
        for source, update in source_updates.items():
            if not isinstance(update, dict):
                continue
            policy_key = f"source:{source}"
            result = await session.execute(
                select(AutoTraderPolicy).where(AutoTraderPolicy.policy_key == policy_key)
            )
            row = result.scalar_one_or_none()
            if row is None:
                row = AutoTraderPolicy(policy_key=policy_key, source=source)
                session.add(row)
            for key, value in update.items():
                if key == "metadata":
                    row.metadata_json = value
                elif hasattr(row, key):
                    setattr(row, key, value)
            row.updated_at = _now()

    await session.commit()
    return await read_autotrader_policies(session)


async def write_autotrader_snapshot(
    session: AsyncSession,
    *,
    running: bool,
    enabled: bool,
    current_activity: Optional[str],
    interval_seconds: int,
    last_run_at: Optional[datetime],
    signals_seen: int,
    signals_selected: int,
    decisions_count: int,
    trades_count: int,
    open_positions: int,
    daily_pnl: float,
    last_error: Optional[str] = None,
    stats: Optional[dict[str, Any]] = None,
) -> None:
    result = await session.execute(
        select(AutoTraderSnapshot).where(AutoTraderSnapshot.id == AUTOTRADER_SNAPSHOT_ID)
    )
    row = result.scalar_one_or_none()
    if row is None:
        row = AutoTraderSnapshot(id=AUTOTRADER_SNAPSHOT_ID)
        session.add(row)

    row.updated_at = _now()
    row.last_run_at = last_run_at or row.last_run_at
    row.running = bool(running)
    row.enabled = bool(enabled)
    row.current_activity = current_activity
    row.interval_seconds = max(1, int(interval_seconds))
    row.signals_seen = int(signals_seen)
    row.signals_selected = int(signals_selected)
    row.decisions_count = int(decisions_count)
    row.trades_count = int(trades_count)
    row.open_positions = int(open_positions)
    row.daily_pnl = float(daily_pnl)
    row.last_error = last_error
    row.stats_json = stats or {}

    await session.commit()


async def read_autotrader_snapshot(session: AsyncSession) -> dict[str, Any]:
    result = await session.execute(
        select(AutoTraderSnapshot).where(AutoTraderSnapshot.id == AUTOTRADER_SNAPSHOT_ID)
    )
    row = result.scalar_one_or_none()
    if row is None:
        return {
            "running": False,
            "enabled": False,
            "current_activity": "Waiting for autotrader worker.",
            "interval_seconds": 2,
            "last_run_at": None,
            "signals_seen": 0,
            "signals_selected": 0,
            "decisions_count": 0,
            "trades_count": 0,
            "open_positions": 0,
            "daily_pnl": 0.0,
            "last_error": None,
            "stats": {},
            "updated_at": None,
        }

    return {
        "running": bool(row.running),
        "enabled": bool(row.enabled),
        "current_activity": row.current_activity,
        "interval_seconds": int(row.interval_seconds or 2),
        "last_run_at": _iso(row.last_run_at),
        "signals_seen": int(row.signals_seen or 0),
        "signals_selected": int(row.signals_selected or 0),
        "decisions_count": int(row.decisions_count or 0),
        "trades_count": int(row.trades_count or 0),
        "open_positions": int(row.open_positions or 0),
        "daily_pnl": float(row.daily_pnl or 0.0),
        "last_error": row.last_error,
        "stats": row.stats_json or {},
        "updated_at": _iso(row.updated_at),
    }


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _event_key_from_payload(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    event_id = payload.get("event_id")
    if event_id:
        return str(event_id)
    event_slug = payload.get("event_slug")
    if event_slug:
        return str(event_slug)
    return None


async def get_autotrader_exposure(session: AsyncSession) -> dict[str, Any]:
    """Return budget/exposure state used by command-center risk surfaces."""
    policies = await read_autotrader_policies(session)
    global_policy = policies.get("global", {}) if isinstance(policies, dict) else {}
    source_policies = policies.get("sources", {}) if isinstance(policies, dict) else {}

    today_start = utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    source_rows = (
        (
            await session.execute(
                select(
                    AutoTraderTrade.source,
                    func.coalesce(func.sum(AutoTraderTrade.notional_usd), 0.0),
                    func.count(AutoTraderTrade.id),
                ).where(
                    and_(
                        AutoTraderTrade.created_at >= today_start,
                        AutoTraderTrade.status.in_(("submitted", "executed")),
                    )
                ).group_by(AutoTraderTrade.source)
            )
        )
        .all()
    )
    source_budget_used = {str(row[0]): _to_float(row[1]) for row in source_rows if row[0]}
    source_open_positions = {str(row[0]): _to_int(row[2]) for row in source_rows if row[0]}

    market_rows = (
        (
            await session.execute(
                select(
                    AutoTraderTrade.market_id,
                    AutoTraderTrade.direction,
                    func.coalesce(func.sum(AutoTraderTrade.notional_usd), 0.0),
                    func.count(AutoTraderTrade.id),
                ).where(
                    AutoTraderTrade.status.in_(("submitted", "executed"))
                ).group_by(AutoTraderTrade.market_id, AutoTraderTrade.direction)
            )
        )
        .all()
    )

    market_exposure: dict[str, float] = defaultdict(float)
    market_open_positions: dict[str, int] = defaultdict(int)
    market_directions: dict[str, set[str]] = defaultdict(set)
    for market_id, direction, notional, positions in market_rows:
        if not market_id:
            continue
        key = str(market_id)
        market_exposure[key] += _to_float(notional)
        market_open_positions[key] += _to_int(positions)
        if direction:
            market_directions[key].add(str(direction))

    event_rows = (
        (
            await session.execute(
                select(
                    AutoTraderTrade.payload_json,
                    func.coalesce(func.sum(AutoTraderTrade.notional_usd), 0.0),
                    func.count(AutoTraderTrade.id),
                )
                .where(AutoTraderTrade.status.in_(("submitted", "executed")))
                .group_by(AutoTraderTrade.payload_json)
            )
        )
        .all()
    )

    event_exposure: dict[str, float] = defaultdict(float)
    event_open_positions: dict[str, int] = defaultdict(int)
    for payload, notional, positions in event_rows:
        event_key = _event_key_from_payload(payload)
        if not event_key:
            continue
        event_exposure[event_key] += _to_float(notional)
        event_open_positions[event_key] += _to_int(positions)

    global_budget_used = _to_float(sum(source_budget_used.values()))
    global_open_positions = _to_int(sum(source_open_positions.values()))
    global_budget = _to_float(global_policy.get("daily_budget_usd"))

    known_sources = set(source_policies.keys()) | set(source_budget_used.keys())
    source_details = []
    for source in sorted(known_sources):
        policy = source_policies.get(source, {})
        budget = _to_float(policy.get("daily_budget_usd"))
        used = _to_float(source_budget_used.get(source))
        remaining = max(0.0, budget - used)
        open_positions = _to_int(source_open_positions.get(source))
        max_open = _to_int(policy.get("max_open_positions"))
        source_details.append(
            {
                "source": source,
                "enabled": bool(policy.get("enabled", True)),
                "daily_budget_usd": budget,
                "budget_used_usd": used,
                "budget_remaining_usd": remaining,
                "budget_utilization_pct": (used / budget) if budget > 0 else 0.0,
                "max_open_positions": max_open,
                "open_positions": open_positions,
                "open_position_utilization_pct": (
                    open_positions / max_open if max_open > 0 else 0.0
                ),
                "weight": _to_float(policy.get("weight"), 1.0),
                "min_signal_score": _to_float(policy.get("min_signal_score"), 0.0),
                "size_multiplier": _to_float(policy.get("size_multiplier"), 1.0),
                "cooldown_seconds": _to_int(policy.get("cooldown_seconds")),
                "metadata": policy.get("metadata", {}) if isinstance(policy, dict) else {},
            }
        )

    market_details = [
        {
            "market_id": market_id,
            "notional_usd": _to_float(notional),
            "open_positions": _to_int(market_open_positions.get(market_id)),
            "directions": sorted(list(market_directions.get(market_id, set()))),
        }
        for market_id, notional in sorted(
            market_exposure.items(), key=lambda item: item[1], reverse=True
        )
    ]

    event_details = [
        {
            "event_key": event_key,
            "notional_usd": _to_float(notional),
            "open_positions": _to_int(event_open_positions.get(event_key)),
        }
        for event_key, notional in sorted(
            event_exposure.items(), key=lambda item: item[1], reverse=True
        )
    ]

    return {
        "as_of": _iso(_now()),
        "global": {
            "daily_budget_usd": global_budget,
            "budget_used_usd": global_budget_used,
            "budget_remaining_usd": max(0.0, global_budget - global_budget_used),
            "budget_utilization_pct": (
                global_budget_used / global_budget if global_budget > 0 else 0.0
            ),
            "max_total_open_positions": _to_int(global_policy.get("max_total_open_positions")),
            "open_positions": global_open_positions,
            "max_per_market_exposure": _to_float(global_policy.get("max_per_market_exposure")),
            "max_per_event_exposure": _to_float(global_policy.get("max_per_event_exposure")),
            "kill_switch": bool(global_policy.get("kill_switch", False)),
        },
        "sources": source_details,
        "markets": market_details,
        "events": event_details,
    }


async def get_autotrader_metrics(session: AsyncSession) -> dict[str, Any]:
    """Return decision quality/performance metrics grouped by source."""
    policies = await read_autotrader_policies(session)
    source_policies = policies.get("sources", {}) if isinstance(policies, dict) else {}

    decision_rows = (
        (
            await session.execute(
                select(
                    AutoTraderDecision.source,
                    AutoTraderDecision.decision,
                    func.count(AutoTraderDecision.id),
                    func.max(AutoTraderDecision.created_at),
                ).group_by(AutoTraderDecision.source, AutoTraderDecision.decision)
            )
        )
        .all()
    )
    decision_counts: dict[str, dict[str, int]] = defaultdict(dict)
    decision_last_seen: dict[str, datetime] = {}
    for source, decision, count, last_seen in decision_rows:
        source_key = str(source)
        decision_counts[source_key][str(decision)] = _to_int(count)
        if last_seen and (
            source_key not in decision_last_seen or last_seen > decision_last_seen[source_key]
        ):
            decision_last_seen[source_key] = last_seen

    avg_score_rows = (
        (
            await session.execute(
                select(
                    AutoTraderDecision.source,
                    func.avg(AutoTraderDecision.score),
                    func.count(AutoTraderDecision.score),
                )
                .where(AutoTraderDecision.score.is_not(None))
                .group_by(AutoTraderDecision.source)
            )
        )
        .all()
    )
    avg_scores = {
        str(source): _to_float(avg_score) for source, avg_score, _count in avg_score_rows if source
    }

    skip_reason_rows = (
        (
            await session.execute(
                select(
                    AutoTraderDecision.source,
                    AutoTraderDecision.reason,
                    func.count(AutoTraderDecision.id),
                )
                .where(
                    and_(
                        AutoTraderDecision.decision == "skipped",
                        AutoTraderDecision.reason.is_not(None),
                    )
                )
                .group_by(AutoTraderDecision.source, AutoTraderDecision.reason)
            )
        )
        .all()
    )
    skip_reasons_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skip_reasons_global: dict[str, int] = defaultdict(int)
    for source, reason, count in skip_reason_rows:
        if not source or not reason:
            continue
        item = {"reason": str(reason), "count": _to_int(count)}
        skip_reasons_by_source[str(source)].append(item)
        skip_reasons_global[str(reason)] += _to_int(count)

    trade_rows = (
        (
            await session.execute(
                select(
                    AutoTraderTrade.source,
                    AutoTraderTrade.status,
                    func.count(AutoTraderTrade.id),
                    func.max(AutoTraderTrade.created_at),
                ).group_by(AutoTraderTrade.source, AutoTraderTrade.status)
            )
        )
        .all()
    )
    trade_counts: dict[str, dict[str, int]] = defaultdict(dict)
    trade_last_seen: dict[str, datetime] = {}
    for source, status, count, last_seen in trade_rows:
        if not source:
            continue
        source_key = str(source)
        trade_counts[source_key][str(status)] = _to_int(count)
        if last_seen and (
            source_key not in trade_last_seen or last_seen > trade_last_seen[source_key]
        ):
            trade_last_seen[source_key] = last_seen

    pending_rows = (
        (
            await session.execute(
                select(
                    TradeSignal.source,
                    func.count(TradeSignal.id),
                    func.max(TradeSignal.created_at),
                )
                .where(TradeSignal.status == "pending")
                .group_by(TradeSignal.source)
            )
        )
        .all()
    )
    pending_counts = {
        str(source): _to_int(count) for source, count, _last_seen in pending_rows if source
    }
    pending_last_seen = {
        str(source): last_seen for source, _count, last_seen in pending_rows if source and last_seen
    }

    one_hour_ago = _now() - timedelta(hours=1)
    recent_rows = (
        (
            await session.execute(
                select(AutoTraderDecision.source, func.count(AutoTraderDecision.id))
                .where(AutoTraderDecision.created_at >= one_hour_ago)
                .group_by(AutoTraderDecision.source)
            )
        )
        .all()
    )
    recent_counts = {str(source): _to_int(count) for source, count in recent_rows if source}

    latency_rows = (
        (
            await session.execute(
                select(
                    AutoTraderTrade.source,
                    AutoTraderTrade.created_at,
                    AutoTraderDecision.created_at,
                )
                .join(
                    AutoTraderDecision,
                    AutoTraderTrade.decision_id == AutoTraderDecision.id,
                )
                .where(AutoTraderTrade.decision_id.is_not(None))
                .order_by(AutoTraderTrade.created_at.desc())
                .limit(5000)
            )
        )
        .all()
    )
    latency_values: dict[str, list[float]] = defaultdict(list)
    for source, trade_created_at, decision_created_at in latency_rows:
        if not source or trade_created_at is None or decision_created_at is None:
            continue
        latency_sec = (trade_created_at - decision_created_at).total_seconds()
        if latency_sec >= 0:
            latency_values[str(source)].append(float(latency_sec))
    avg_latency_by_source = {
        source: (sum(values) / len(values)) if values else 0.0
        for source, values in latency_values.items()
    }

    all_sources = (
        set(source_policies.keys())
        | set(decision_counts.keys())
        | set(trade_counts.keys())
        | set(pending_counts.keys())
    )

    source_metrics: list[dict[str, Any]] = []
    for source in sorted(all_sources):
        source_decisions = decision_counts.get(source, {})
        source_trades = trade_counts.get(source, {})
        skipped = _to_int(source_decisions.get("skipped"))
        executed = _to_int(source_decisions.get("executed"))
        submitted = _to_int(source_decisions.get("submitted"))
        failed = _to_int(source_decisions.get("failed"))
        selected = executed + submitted + failed
        total_decisions = _to_int(sum(source_decisions.values()))

        source_skip_reasons = sorted(
            skip_reasons_by_source.get(source, []),
            key=lambda item: item["count"],
            reverse=True,
        )[:5]

        source_metrics.append(
            {
                "source": source,
                "policy_enabled": bool(source_policies.get(source, {}).get("enabled", True)),
                "pending_signals": _to_int(pending_counts.get(source)),
                "decisions_total": total_decisions,
                "selected": selected,
                "skipped": skipped,
                "executed": executed,
                "submitted": submitted,
                "failed": failed,
                "skip_rate": (skipped / total_decisions) if total_decisions > 0 else 0.0,
                "success_rate": (
                    executed / selected if selected > 0 else 0.0
                ),
                "trades": source_trades,
                "avg_decision_score": _to_float(avg_scores.get(source)),
                "decisions_last_hour": _to_int(recent_counts.get(source)),
                "throughput_per_minute": _to_float(recent_counts.get(source)) / 60.0,
                "avg_decision_to_trade_latency_seconds": _to_float(
                    avg_latency_by_source.get(source)
                ),
                "top_skip_reasons": source_skip_reasons,
                "last_decision_at": _iso(decision_last_seen.get(source)),
                "last_trade_at": _iso(trade_last_seen.get(source)),
                "last_pending_signal_at": _iso(pending_last_seen.get(source)),
            }
        )

    source_metrics.sort(
        key=lambda item: (
            item["pending_signals"] + item["decisions_last_hour"],
            item["decisions_total"],
        ),
        reverse=True,
    )

    pending_total = _to_int(sum(pending_counts.values()))
    skipped_total = _to_int(
        sum(source_counts.get("skipped", 0) for source_counts in decision_counts.values())
    )
    executed_total = _to_int(
        sum(source_counts.get("executed", 0) for source_counts in decision_counts.values())
    )
    submitted_total = _to_int(
        sum(source_counts.get("submitted", 0) for source_counts in decision_counts.values())
    )
    failed_total = _to_int(
        sum(source_counts.get("failed", 0) for source_counts in decision_counts.values())
    )
    selected_total = executed_total + submitted_total + failed_total

    skip_reasons = [
        {"reason": reason, "count": count}
        for reason, count in sorted(
            skip_reasons_global.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:10]
    ]

    return {
        "as_of": _iso(_now()),
        "summary": {
            "sources_tracked": len(source_metrics),
            "active_sources": sum(
                1
                for source in source_metrics
                if source["pending_signals"] > 0
                or source["decisions_last_hour"] > 0
                or source["decisions_total"] > 0
            ),
            "decisions_last_hour": _to_int(sum(recent_counts.values())),
        },
        "decision_funnel": {
            "seen": pending_total + selected_total + skipped_total,
            "pending": pending_total,
            "selected": selected_total,
            "skipped": skipped_total,
            "submitted": submitted_total,
            "executed": executed_total,
            "failed": failed_total,
        },
        "skip_reasons": skip_reasons,
        "sources": source_metrics,
    }


async def create_autotrader_decision(
    session: AsyncSession,
    *,
    signal_id: Optional[str],
    source: str,
    decision: str,
    reason: Optional[str],
    score: Optional[float],
    policy_snapshot: Optional[dict[str, Any]] = None,
    risk_snapshot: Optional[dict[str, Any]] = None,
    payload: Optional[dict[str, Any]] = None,
) -> AutoTraderDecision:
    row = AutoTraderDecision(
        id=uuid.uuid4().hex,
        signal_id=signal_id,
        source=source,
        decision=decision,
        reason=reason,
        score=score,
        policy_snapshot_json=policy_snapshot or {},
        risk_snapshot_json=risk_snapshot or {},
        payload_json=payload or {},
        created_at=_now(),
    )
    session.add(row)
    await session.commit()
    return row


async def create_autotrader_trade(
    session: AsyncSession,
    *,
    signal_id: str,
    decision_id: Optional[str],
    source: str,
    market_id: str,
    market_question: Optional[str],
    direction: Optional[str],
    mode: str,
    status: str,
    notional_usd: Optional[float],
    entry_price: Optional[float],
    effective_price: Optional[float],
    edge_percent: Optional[float],
    confidence: Optional[float],
    reason: Optional[str],
    payload: Optional[dict[str, Any]],
    error_message: Optional[str] = None,
) -> AutoTraderTrade:
    now = _now()
    row = AutoTraderTrade(
        id=uuid.uuid4().hex,
        signal_id=signal_id,
        decision_id=decision_id,
        source=source,
        market_id=market_id,
        market_question=market_question,
        direction=direction,
        mode=mode,
        status=status,
        notional_usd=notional_usd,
        entry_price=entry_price,
        effective_price=effective_price,
        edge_percent=edge_percent,
        confidence=confidence,
        reason=reason,
        payload_json=payload or {},
        error_message=error_message,
        created_at=now,
        executed_at=now if status == "executed" else None,
        updated_at=now,
    )
    session.add(row)
    await session.commit()
    return row


async def list_autotrader_trades(
    session: AsyncSession,
    *,
    source: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 200,
) -> list[AutoTraderTrade]:
    query = select(AutoTraderTrade).order_by(desc(AutoTraderTrade.created_at))
    if source:
        query = query.where(AutoTraderTrade.source == source)
    if status:
        query = query.where(AutoTraderTrade.status == status)
    query = query.limit(max(1, min(limit, 2000)))
    result = await session.execute(query)
    return list(result.scalars().all())


async def list_autotrader_decisions(
    session: AsyncSession,
    *,
    source: Optional[str] = None,
    decision: Optional[str] = None,
    limit: int = 500,
) -> list[AutoTraderDecision]:
    query = select(AutoTraderDecision).order_by(desc(AutoTraderDecision.created_at))
    if source:
        query = query.where(AutoTraderDecision.source == source)
    if decision:
        query = query.where(AutoTraderDecision.decision == decision)
    query = query.limit(max(1, min(limit, 5000)))
    result = await session.execute(query)
    return list(result.scalars().all())
