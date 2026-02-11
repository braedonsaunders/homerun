"""Dedicated autotrader worker consuming normalized trade signals."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import and_, func, select

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
if os.getcwd() != _BACKEND:
    os.chdir(_BACKEND)

from models.database import (
    AsyncSessionLocal,
    AutoTraderDecision,
    AutoTraderTrade,
    TradeSignal,
    init_database,
)
from services.autotrader_state import (
    clear_autotrader_run_request,
    create_autotrader_decision,
    create_autotrader_trade,
    ensure_default_autotrader_policies,
    read_autotrader_control,
    read_autotrader_policies,
    reset_autotrader_for_manual_start,
    write_autotrader_snapshot,
)
from services.signal_bus import (
    expire_stale_signals,
    list_pending_trade_signals,
    set_trade_signal_status,
)
from services.trading import OrderSide, OrderStatus, OrderType, trading_service
from services.worker_state import write_worker_snapshot

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("autotrader_worker")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _to_iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.isoformat() + "Z"


def _source_live_enabled(mode: str, source_policy: dict[str, Any]) -> bool:
    if mode != "live":
        return True
    metadata = source_policy.get("metadata") if isinstance(source_policy, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    return bool(metadata.get("live_enabled", True))


def _extract_event_key(signal: TradeSignal) -> Optional[str]:
    payload = signal.payload_json if isinstance(signal.payload_json, dict) else {}
    event_id = payload.get("event_id")
    if event_id:
        return str(event_id)
    event_slug = payload.get("event_slug")
    if event_slug:
        return str(event_slug)
    return None


def _extract_token_id(signal: TradeSignal) -> Optional[str]:
    payload = signal.payload_json if isinstance(signal.payload_json, dict) else {}
    direct = payload.get("token_id")
    if direct:
        return str(direct)

    positions = payload.get("positions_to_take") if isinstance(payload, dict) else None
    if isinstance(positions, list):
        for pos in positions:
            if not isinstance(pos, dict):
                continue
            token_id = pos.get("token_id")
            outcome = str(pos.get("outcome") or "").lower()
            if signal.direction == "buy_yes" and outcome == "yes" and token_id:
                return str(token_id)
            if signal.direction == "buy_no" and outcome == "no" and token_id:
                return str(token_id)
        for pos in positions:
            if isinstance(pos, dict) and pos.get("token_id"):
                return str(pos.get("token_id"))

    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    if isinstance(metadata, dict):
        if metadata.get("token_id"):
            return str(metadata.get("token_id"))

    return None


def _score_signal(
    signal: TradeSignal,
    *,
    now: datetime,
    source_weight: float,
    risk_penalty: float,
) -> float:
    confidence = _clamp(_safe_float(signal.confidence, 0.5), 0.01, 1.0)
    edge_percent = max(0.0, _safe_float(signal.edge_percent, 0.0))
    edge_factor = _clamp(edge_percent / 15.0, 0.05, 2.0)

    created_at = signal.created_at or now
    age_seconds = max(0.0, (now - created_at).total_seconds())
    if signal.expires_at:
        ttl = max(1.0, (signal.expires_at - created_at).total_seconds())
    else:
        ttl = 3600.0
    freshness_factor = _clamp(1.0 - (age_seconds / ttl), 0.05, 1.0)

    liquidity = max(0.0, _safe_float(signal.liquidity, 0.0))
    liquidity_factor = _clamp(math.log10(liquidity + 10.0) / 4.0, 0.05, 1.0)

    return (
        max(0.01, source_weight)
        * confidence
        * edge_factor
        * freshness_factor
        * liquidity_factor
        * _clamp(risk_penalty, 0.05, 1.0)
    )


def _size_for_signal(
    signal: TradeSignal,
    *,
    source_policy: dict[str, Any],
    source_budget_remaining: float,
    global_budget_remaining: float,
) -> float:
    edge = max(0.0, _safe_float(signal.edge_percent, 0.0))
    confidence = _clamp(_safe_float(signal.confidence, 0.5), 0.1, 1.0)
    size_multiplier = _clamp(_safe_float(source_policy.get("size_multiplier"), 1.0), 0.1, 5.0)

    base = 10.0 * size_multiplier
    edge_boost = _clamp(edge / 10.0, 0.5, 3.0)
    confidence_boost = _clamp(confidence, 0.25, 1.0)
    target = base * edge_boost * confidence_boost

    cap = min(max(0.0, source_budget_remaining), max(0.0, global_budget_remaining))
    return max(0.0, min(target, cap))


async def _load_exposure_state(session) -> dict[str, Any]:
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

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
    source_budget_used = {row[0]: float(row[1] or 0.0) for row in source_rows}
    source_open_positions = {row[0]: int(row[2] or 0) for row in source_rows}

    global_budget_used = float(sum(source_budget_used.values()))
    global_open_positions = int(sum(source_open_positions.values()))

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
    market_directions: dict[str, set[str]] = defaultdict(set)
    for market_id, direction, notional, _count in market_rows:
        if not market_id:
            continue
        market_exposure[str(market_id)] += float(notional or 0.0)
        if direction:
            market_directions[str(market_id)].add(str(direction))

    event_rows = (
        (
            await session.execute(
                select(AutoTraderTrade.payload_json, func.coalesce(func.sum(AutoTraderTrade.notional_usd), 0.0))
                .where(AutoTraderTrade.status.in_(("submitted", "executed")))
                .group_by(AutoTraderTrade.payload_json)
            )
        )
        .all()
    )

    event_exposure: dict[str, float] = defaultdict(float)
    for payload, notional in event_rows:
        if isinstance(payload, dict):
            key = payload.get("event_id") or payload.get("event_slug")
            if key:
                event_exposure[str(key)] += float(notional or 0.0)

    last_trade_rows = (
        (
            await session.execute(
                select(AutoTraderTrade.source, func.max(AutoTraderTrade.created_at)).group_by(AutoTraderTrade.source)
            )
        )
        .all()
    )
    last_trade_at = {str(source): ts for source, ts in last_trade_rows if source}

    return {
        "source_budget_used": source_budget_used,
        "source_open_positions": source_open_positions,
        "global_budget_used": global_budget_used,
        "global_open_positions": global_open_positions,
        "market_exposure": market_exposure,
        "market_directions": market_directions,
        "event_exposure": event_exposure,
        "last_trade_at": last_trade_at,
    }


async def _execute_live_trade(
    signal: TradeSignal,
    notional_usd: float,
) -> tuple[str, Optional[float], Optional[str], dict[str, Any]]:
    token_id = _extract_token_id(signal)
    if not token_id:
        return "failed", None, "Missing token_id for live execution", {}

    entry_price = _safe_float(signal.effective_price, _safe_float(signal.entry_price, 0.0))
    if entry_price <= 0:
        return "failed", None, "Missing entry price for live execution", {"token_id": token_id}

    if not trading_service.is_ready():
        initialized = await trading_service.initialize()
        if not initialized:
            return "failed", None, "Trading service initialization failed", {"token_id": token_id}

    shares = max(1.0, notional_usd / max(entry_price, 0.01))
    order = await trading_service.place_order(
        token_id=token_id,
        side=OrderSide.BUY,
        price=entry_price,
        size=shares,
        order_type=OrderType.GTC,
        market_question=signal.market_question,
        opportunity_id=signal.source_item_id,
    )

    if order.status == OrderStatus.FAILED:
        return "failed", None, order.error_message, {
            "order_id": order.id,
            "token_id": token_id,
            "shares": shares,
        }

    final_status = "executed" if order.status == OrderStatus.FILLED else "submitted"
    return final_status, order.price, None, {
        "order_id": order.id,
        "token_id": token_id,
        "shares": shares,
        "order_status": order.status.value,
    }


async def _run_cycle() -> dict[str, Any]:
    worker_name = "autotrader"
    now = datetime.utcnow()

    async with AsyncSessionLocal() as session:
        await ensure_default_autotrader_policies(session)
        control = await read_autotrader_control(session)
        policies = await read_autotrader_policies(session)

    mode = str(control.get("mode") or "paper")
    run_interval = max(1, min(60, int(control.get("run_interval_seconds") or 2)))
    enabled = bool(control.get("is_enabled", True))
    paused = bool(control.get("is_paused", False))
    kill_switch = bool(control.get("kill_switch", False))
    requested_run = bool(control.get("requested_run_at"))

    if (not enabled or paused or kill_switch) and not requested_run:
        return {
            "idle": True,
            "interval": run_interval,
            "mode": mode,
            "reason": "kill_switch" if kill_switch else ("paused" if paused else "disabled"),
            "signals_seen": 0,
            "signals_selected": 0,
            "decisions_count": 0,
            "trades_count": 0,
            "open_positions": 0,
            "daily_pnl": 0.0,
            "errors": [],
        }

    source_policies = policies.get("sources", {}) if isinstance(policies, dict) else {}
    global_policy = policies.get("global", {}) if isinstance(policies, dict) else {}

    enabled_sources = [
        source for source, cfg in source_policies.items() if bool((cfg or {}).get("enabled", False))
    ]

    # If explicit source policies exist and every source is disabled, the worker
    # must not fall back to trading all sources.
    if source_policies and not enabled_sources:
        if requested_run:
            async with AsyncSessionLocal() as session:
                await clear_autotrader_run_request(session)
        return {
            "idle": True,
            "interval": run_interval,
            "mode": mode,
            "reason": "no_enabled_sources",
            "signals_seen": 0,
            "signals_selected": 0,
            "decisions_count": 0,
            "trades_count": 0,
            "open_positions": 0,
            "daily_pnl": 0.0,
            "errors": [],
        }

    errors: list[str] = []
    decisions_count = 0
    trades_count = 0
    signals_selected = 0

    async with AsyncSessionLocal() as session:
        await expire_stale_signals(session)
        pending_signals = await list_pending_trade_signals(
            session,
            sources=enabled_sources if enabled_sources else None,
            limit=2000,
        )
        exposure = await _load_exposure_state(session)

    source_budget_used = dict(exposure["source_budget_used"])
    source_open_positions = dict(exposure["source_open_positions"])
    global_budget_used = float(exposure["global_budget_used"])
    global_open_positions = int(exposure["global_open_positions"])
    market_exposure = dict(exposure["market_exposure"])
    market_directions = {k: set(v) for k, v in exposure["market_directions"].items()}
    event_exposure = dict(exposure["event_exposure"])
    last_trade_at = dict(exposure["last_trade_at"])

    max_total_open = int(global_policy.get("max_total_open_positions") or 0)
    risk_penalty = 1.0
    if max_total_open > 0:
        risk_penalty = 1.0 / (1.0 + (global_open_positions / max_total_open))

    scored: list[tuple[TradeSignal, float]] = []
    for signal in pending_signals:
        source_policy = source_policies.get(signal.source) or {}
        source_weight = _safe_float(source_policy.get("weight"), 1.0)
        score = _score_signal(signal, now=now, source_weight=source_weight, risk_penalty=risk_penalty)
        scored.append((signal, score))

    scored.sort(key=lambda item: item[1], reverse=True)

    seen_dedupe: set[str] = set()

    for signal, score in scored:
        source_policy = source_policies.get(signal.source) or {}
        source_metadata = source_policy.get("metadata") if isinstance(source_policy, dict) else {}
        if not isinstance(source_metadata, dict):
            source_metadata = {}
        min_score = _safe_float(source_policy.get("min_signal_score"), 0.0)
        signal_payload = signal.payload_json if isinstance(signal.payload_json, dict) else {}

        if not _source_live_enabled(mode, source_policy):
            async with AsyncSessionLocal() as session:
                await set_trade_signal_status(session, signal.id, "skipped")
                await create_autotrader_decision(
                    session,
                    signal_id=signal.id,
                    source=signal.source,
                    decision="skipped",
                    reason="Source policy blocks LIVE trading (metadata.live_enabled=false)",
                    score=score,
                    policy_snapshot=source_policy,
                    risk_snapshot={"rule": "source_live_gate"},
                    payload={"mode": mode, "metadata": source_metadata},
                )
            decisions_count += 1
            continue

        # Deduplicate cross-source duplicates by dedupe_key.
        if signal.dedupe_key in seen_dedupe:
            async with AsyncSessionLocal() as session:
                await set_trade_signal_status(session, signal.id, "skipped")
                await create_autotrader_decision(
                    session,
                    signal_id=signal.id,
                    source=signal.source,
                    decision="skipped",
                    reason="Duplicate dedupe_key in selection set",
                    score=score,
                    policy_snapshot=source_policy,
                    risk_snapshot={"rule": "dedupe"},
                    payload={"dedupe_key": signal.dedupe_key},
                )
            decisions_count += 1
            continue

        if score < min_score:
            async with AsyncSessionLocal() as session:
                await set_trade_signal_status(session, signal.id, "skipped")
                await create_autotrader_decision(
                    session,
                    signal_id=signal.id,
                    source=signal.source,
                    decision="skipped",
                    reason=f"Signal score {score:.4f} below min_signal_score {min_score:.4f}",
                    score=score,
                    policy_snapshot=source_policy,
                    risk_snapshot={"rule": "min_signal_score"},
                    payload={"computed_score": score},
                )
            decisions_count += 1
            continue

        # Cooldown
        cooldown = int(source_policy.get("cooldown_seconds") or 0)
        last_ts = last_trade_at.get(signal.source)
        if cooldown > 0 and last_ts is not None and (now - last_ts).total_seconds() < cooldown:
            async with AsyncSessionLocal() as session:
                await set_trade_signal_status(session, signal.id, "skipped")
                await create_autotrader_decision(
                    session,
                    signal_id=signal.id,
                    source=signal.source,
                    decision="skipped",
                    reason=f"Cooldown active ({cooldown}s)",
                    score=score,
                    policy_snapshot=source_policy,
                    risk_snapshot={"rule": "cooldown"},
                    payload={"last_trade_at": _to_iso(last_ts)},
                )
            decisions_count += 1
            continue

        # Source/global budget + open position checks.
        source_budget = _safe_float(source_policy.get("daily_budget_usd"), 0.0)
        source_used = _safe_float(source_budget_used.get(signal.source), 0.0)
        source_remaining = max(0.0, source_budget - source_used)

        global_budget = _safe_float(global_policy.get("daily_budget_usd"), 0.0)
        global_remaining = max(0.0, global_budget - global_budget_used)

        source_open = int(source_open_positions.get(signal.source, 0))
        source_max_open = int(source_policy.get("max_open_positions") or 0)

        if source_max_open > 0 and source_open >= source_max_open:
            async with AsyncSessionLocal() as session:
                await set_trade_signal_status(session, signal.id, "skipped")
                await create_autotrader_decision(
                    session,
                    signal_id=signal.id,
                    source=signal.source,
                    decision="skipped",
                    reason="Source max_open_positions reached",
                    score=score,
                    policy_snapshot=source_policy,
                    risk_snapshot={"rule": "source_max_open_positions"},
                    payload={"open_positions": source_open},
                )
            decisions_count += 1
            continue

        if max_total_open > 0 and global_open_positions >= max_total_open:
            async with AsyncSessionLocal() as session:
                await set_trade_signal_status(session, signal.id, "skipped")
                await create_autotrader_decision(
                    session,
                    signal_id=signal.id,
                    source=signal.source,
                    decision="skipped",
                    reason="Global max_total_open_positions reached",
                    score=score,
                    policy_snapshot=global_policy,
                    risk_snapshot={"rule": "global_max_open_positions"},
                    payload={"open_positions": global_open_positions},
                )
            decisions_count += 1
            continue

        market_id = str(signal.market_id)
        existing_dirs = market_directions.get(market_id, set())
        if existing_dirs and signal.direction and signal.direction not in existing_dirs:
            async with AsyncSessionLocal() as session:
                await set_trade_signal_status(session, signal.id, "skipped")
                await create_autotrader_decision(
                    session,
                    signal_id=signal.id,
                    source=signal.source,
                    decision="skipped",
                    reason="Market direction conflict suppression",
                    score=score,
                    policy_snapshot=source_policy,
                    risk_snapshot={"rule": "market_direction_conflict"},
                    payload={"existing_directions": sorted(existing_dirs)},
                )
            decisions_count += 1
            continue

        max_market_exposure = _safe_float(global_policy.get("max_per_market_exposure"), 0.0)
        if max_market_exposure > 0 and _safe_float(market_exposure.get(market_id), 0.0) >= max_market_exposure:
            async with AsyncSessionLocal() as session:
                await set_trade_signal_status(session, signal.id, "skipped")
                await create_autotrader_decision(
                    session,
                    signal_id=signal.id,
                    source=signal.source,
                    decision="skipped",
                    reason="Per-market exposure cap reached",
                    score=score,
                    policy_snapshot=global_policy,
                    risk_snapshot={"rule": "max_per_market_exposure"},
                    payload={"market_exposure": market_exposure.get(market_id, 0.0)},
                )
            decisions_count += 1
            continue

        event_key = _extract_event_key(signal)
        max_event_exposure = _safe_float(global_policy.get("max_per_event_exposure"), 0.0)
        if (
            event_key
            and max_event_exposure > 0
            and _safe_float(event_exposure.get(event_key), 0.0) >= max_event_exposure
        ):
            async with AsyncSessionLocal() as session:
                await set_trade_signal_status(session, signal.id, "skipped")
                await create_autotrader_decision(
                    session,
                    signal_id=signal.id,
                    source=signal.source,
                    decision="skipped",
                    reason="Per-event exposure cap reached",
                    score=score,
                    policy_snapshot=global_policy,
                    risk_snapshot={"rule": "max_per_event_exposure"},
                    payload={"event_key": event_key, "event_exposure": event_exposure.get(event_key, 0.0)},
                )
            decisions_count += 1
            continue

        size_usd = _size_for_signal(
            signal,
            source_policy=source_policy,
            source_budget_remaining=source_remaining,
            global_budget_remaining=global_remaining,
        )
        if size_usd <= 0.0:
            async with AsyncSessionLocal() as session:
                await set_trade_signal_status(session, signal.id, "skipped")
                await create_autotrader_decision(
                    session,
                    signal_id=signal.id,
                    source=signal.source,
                    decision="skipped",
                    reason="No remaining source/global budget",
                    score=score,
                    policy_snapshot=source_policy,
                    risk_snapshot={"rule": "budget"},
                    payload={
                        "source_remaining": source_remaining,
                        "global_remaining": global_remaining,
                    },
                )
            decisions_count += 1
            continue

        # Selected -> execute/submit.
        async with AsyncSessionLocal() as session:
            await set_trade_signal_status(session, signal.id, "selected")

        trade_status = "executed"
        effective_price = signal.effective_price or signal.entry_price
        error_message = None
        execution_payload: dict[str, Any] = {}

        if mode == "live":
            try:
                trade_status, live_price, err, execution_payload = await _execute_live_trade(signal, size_usd)
                if live_price is not None:
                    effective_price = live_price
                error_message = err
            except Exception as exc:
                trade_status = "failed"
                error_message = str(exc)
                execution_payload = {"error": str(exc)}
                errors.append(str(exc))

        async with AsyncSessionLocal() as session:
            decision = await create_autotrader_decision(
                session,
                signal_id=signal.id,
                source=signal.source,
                decision=trade_status,
                reason=(
                    "Executed in paper mode"
                    if mode != "live" and trade_status == "executed"
                    else (error_message or "Submitted live order")
                ),
                score=score,
                policy_snapshot=source_policy,
                risk_snapshot={
                    "source_budget_remaining": source_remaining,
                    "global_budget_remaining": global_remaining,
                    "source_open_positions": source_open,
                    "global_open_positions": global_open_positions,
                },
                payload={
                    "mode": mode,
                    "size_usd": size_usd,
                    "execution": execution_payload,
                },
            )

            await create_autotrader_trade(
                session,
                signal_id=signal.id,
                decision_id=decision.id,
                source=signal.source,
                market_id=signal.market_id,
                market_question=signal.market_question,
                direction=signal.direction,
                mode=mode,
                status=trade_status,
                notional_usd=size_usd,
                entry_price=signal.entry_price,
                effective_price=effective_price,
                edge_percent=signal.edge_percent,
                confidence=signal.confidence,
                reason=decision.reason,
                payload={
                    "event_id": _extract_event_key(signal),
                    "signal_payload": signal_payload,
                    "execution": execution_payload,
                },
                error_message=error_message,
            )

            if trade_status == "submitted":
                final_signal_status = "submitted"
            elif trade_status == "executed":
                final_signal_status = "executed"
            else:
                final_signal_status = "failed"
            await set_trade_signal_status(
                session,
                signal.id,
                final_signal_status,
                effective_price=effective_price,
            )

        decisions_count += 1
        trades_count += 1
        signals_selected += 1

        # Update in-memory exposure after placement.
        source_budget_used[signal.source] = source_used + size_usd
        source_open_positions[signal.source] = source_open + 1
        global_budget_used += size_usd
        global_open_positions += 1
        market_exposure[market_id] = _safe_float(market_exposure.get(market_id), 0.0) + size_usd
        if signal.direction:
            market_directions.setdefault(market_id, set()).add(signal.direction)
        if event_key:
            event_exposure[event_key] = _safe_float(event_exposure.get(event_key), 0.0) + size_usd
        last_trade_at[signal.source] = now
        seen_dedupe.add(signal.dedupe_key)

    # Clear one-shot run request if present.
    if requested_run:
        async with AsyncSessionLocal() as session:
            await clear_autotrader_run_request(session)

    # Aggregate decision/trade counts for snapshot.
    async with AsyncSessionLocal() as session:
        total_decisions = int(
            (
                await session.execute(select(func.count(AutoTraderDecision.id)))
            ).scalar()
            or 0
        )
        open_positions = int(
            (
                await session.execute(
                    select(func.count(AutoTraderTrade.id)).where(
                        AutoTraderTrade.status.in_(("submitted", "executed"))
                    )
                )
            ).scalar()
            or 0
        )

    return {
        "idle": False,
        "interval": run_interval,
        "mode": mode,
        "signals_seen": len(pending_signals),
        "signals_selected": signals_selected,
        "decisions_count": decisions_count,
        "trades_count": trades_count,
        "open_positions": open_positions,
        "daily_pnl": 0.0,
        "errors": errors,
        "total_decisions": total_decisions,
    }


async def _run_loop() -> None:
    worker_name = "autotrader"
    logger.info("Autotrader worker started")

    # Create baseline snapshots.
    async with AsyncSessionLocal() as session:
        # Safety default: never auto-start trading on worker process launch.
        await reset_autotrader_for_manual_start(session)
        await write_worker_snapshot(
            session,
            worker_name,
            running=True,
            enabled=True,
            current_activity="Autotrader worker started; first cycle pending.",
            interval_seconds=2,
            last_run_at=None,
            last_error=None,
            stats={
                "mode": "paper",
                "signals_seen": 0,
                "signals_selected": 0,
                "decisions_count": 0,
                "trades_count": 0,
            },
        )

    while True:
        cycle_started = datetime.utcnow()
        cycle = await _run_cycle()
        interval = int(cycle.get("interval") or 2)

        activity = "Idle"
        if cycle.get("idle"):
            reason = cycle.get("reason") or "paused"
            activity = f"Autotrader idle ({reason})"
        else:
            activity = (
                f"Processed {cycle.get('signals_seen', 0)} signals, "
                f"selected {cycle.get('signals_selected', 0)}"
            )

        errors = cycle.get("errors") or []
        last_error = errors[-1] if errors else None

        async with AsyncSessionLocal() as session:
            await write_autotrader_snapshot(
                session,
                running=True,
                enabled=not cycle.get("idle", False),
                current_activity=activity,
                interval_seconds=interval,
                last_run_at=cycle_started,
                signals_seen=int(cycle.get("signals_seen", 0)),
                signals_selected=int(cycle.get("signals_selected", 0)),
                decisions_count=int(cycle.get("decisions_count", 0)),
                trades_count=int(cycle.get("trades_count", 0)),
                open_positions=int(cycle.get("open_positions", 0)),
                daily_pnl=float(cycle.get("daily_pnl", 0.0)),
                last_error=last_error,
                stats={
                    "mode": cycle.get("mode"),
                    "total_decisions": int(cycle.get("total_decisions", 0)),
                    "errors": errors[-10:],
                },
            )

            await write_worker_snapshot(
                session,
                worker_name,
                running=True,
                enabled=not cycle.get("idle", False),
                current_activity=activity,
                interval_seconds=interval,
                last_run_at=cycle_started,
                last_error=last_error,
                stats={
                    "mode": cycle.get("mode"),
                    "signals_seen": int(cycle.get("signals_seen", 0)),
                    "signals_selected": int(cycle.get("signals_selected", 0)),
                    "decisions_count": int(cycle.get("decisions_count", 0)),
                    "trades_count": int(cycle.get("trades_count", 0)),
                },
            )

        await asyncio.sleep(max(1, interval))


async def main() -> None:
    await init_database()
    logger.info("Database initialized")
    try:
        await _run_loop()
    except asyncio.CancelledError:
        logger.info("Autotrader worker shutting down")


if __name__ == "__main__":
    asyncio.run(main())
