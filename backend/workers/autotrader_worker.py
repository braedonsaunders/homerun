"""Dedicated autotrader worker consuming normalized trade signals."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import and_, func, select

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
if os.getcwd() != _BACKEND:
    os.chdir(_BACKEND)

from utils.utcnow import utcnow
from models.database import (
    AsyncSessionLocal,
    AutoTraderDecision,
    AutoTraderTrade,
    SimulationPosition,
    SimulationTrade,
    TradeSignal,
    TradeStatus,
    init_database,
)
from models.opportunity import ArbitrageOpportunity
from services.autotrader_state import (
    SOURCE_DOMAIN_MAP,
    clear_autotrader_run_request,
    create_autotrader_decision,
    create_autotrader_trade,
    ensure_default_autotrader_policies,
    normalize_trading_domains,
    read_autotrader_control,
    read_autotrader_policies,
    read_autotrader_snapshot,
    reset_autotrader_for_manual_start,
    write_autotrader_snapshot,
)
from services.signal_bus import (
    expire_stale_signals,
    list_pending_trade_signals,
    set_trade_signal_status,
)
from services.trading import OrderSide, OrderStatus, OrderType, trading_service
from services.simulation import simulation_service
from services.worker_state import write_worker_snapshot

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("autotrader_worker")
OPEN_AUTOTRADER_STATUSES = ("submitted", "executed", "open")


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


def _normalize_strategy(value: Any) -> str:
    return str(value or "").strip().lower()


def _strategy_enabled(signal: TradeSignal, enabled_strategies: set[str]) -> bool:
    if not enabled_strategies:
        return True
    strategy = _normalize_strategy(signal.strategy_type) or _normalize_strategy(signal.signal_type)
    return strategy in enabled_strategies


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


def _enum_text(value: Any) -> str:
    if hasattr(value, "value"):
        return str(value.value).lower()
    return str(value).lower()


def _build_positions_to_take(signal: TradeSignal) -> list[dict[str, Any]]:
    payload = signal.payload_json if isinstance(signal.payload_json, dict) else {}
    positions_raw = payload.get("positions_to_take") if isinstance(payload, dict) else None
    positions: list[dict[str, Any]] = []

    if isinstance(positions_raw, list):
        for pos in positions_raw:
            if not isinstance(pos, dict):
                continue

            price = _safe_float(
                pos.get("price"),
                _safe_float(signal.effective_price, _safe_float(signal.entry_price, 0.0)),
            )
            if price <= 0:
                continue

            outcome_raw = str(pos.get("outcome") or "").strip().lower()
            if outcome_raw in {"yes", "buy_yes", "y"}:
                outcome = "YES"
            elif outcome_raw in {"no", "buy_no", "n"}:
                outcome = "NO"
            else:
                outcome = "NO" if signal.direction == "buy_no" else "YES"

            market_ref = str(pos.get("market_id") or pos.get("market") or signal.market_id)
            normalized: dict[str, Any] = {
                "action": str(pos.get("action") or "buy").lower(),
                "outcome": outcome,
                "market": market_ref,
                "market_id": market_ref,
                "market_question": str(pos.get("market_question") or signal.market_question or market_ref),
                "price": price,
            }
            token_id = pos.get("token_id")
            if token_id:
                normalized["token_id"] = str(token_id)
            positions.append(normalized)

    if positions:
        return positions

    entry_price = _safe_float(signal.effective_price, _safe_float(signal.entry_price, 0.0))
    if entry_price <= 0:
        return []

    fallback_position: dict[str, Any] = {
        "action": "buy",
        "outcome": "NO" if signal.direction == "buy_no" else "YES",
        "market": str(signal.market_id),
        "market_id": str(signal.market_id),
        "market_question": str(signal.market_question or signal.market_id),
        "price": entry_price,
    }
    token_id = _extract_token_id(signal)
    if token_id:
        fallback_position["token_id"] = token_id
    return [fallback_position]


def _build_simulation_opportunity(signal: TradeSignal) -> Optional[ArbitrageOpportunity]:
    payload = signal.payload_json if isinstance(signal.payload_json, dict) else {}
    positions = _build_positions_to_take(signal)
    if not positions:
        return None

    total_cost = _safe_float(payload.get("total_cost"), 0.0)
    if total_cost <= 0:
        total_cost = sum(_safe_float(pos.get("price"), 0.0) for pos in positions)
    if total_cost <= 0:
        return None

    expected_payout = _safe_float(payload.get("expected_payout"), 1.0)
    gross_profit = _safe_float(payload.get("gross_profit"), max(0.0, expected_payout - total_cost))
    fee = _safe_float(payload.get("fee"), 0.0)
    net_profit = _safe_float(payload.get("net_profit"), gross_profit - fee)
    roi_percent = _safe_float(
        payload.get("roi_percent"),
        (net_profit / total_cost * 100.0) if total_cost > 0 else 0.0,
    )
    risk_score = _clamp(_safe_float(payload.get("risk_score"), 0.5), 0.0, 1.0)

    resolution_date = signal.expires_at
    if resolution_date is None and isinstance(payload, dict):
        raw_resolution = payload.get("resolution_date") or payload.get("expires_at")
        if isinstance(raw_resolution, str):
            try:
                parsed = datetime.fromisoformat(raw_resolution.replace("Z", "+00:00"))
                resolution_date = parsed.astimezone(timezone.utc).replace(tzinfo=None)
            except Exception:
                resolution_date = None

    return ArbitrageOpportunity(
        id=str(signal.id),
        strategy=str(signal.strategy_type or signal.source),
        title=str(signal.market_question or signal.market_id),
        description=f"AutoTrader {signal.source} signal",
        total_cost=total_cost,
        expected_payout=expected_payout,
        gross_profit=gross_profit,
        fee=fee,
        net_profit=net_profit,
        roi_percent=roi_percent,
        risk_score=risk_score,
        risk_factors=list(payload.get("risk_factors") or []),
        markets=list(payload.get("markets") or []),
        event_id=_extract_event_key(signal),
        min_liquidity=max(0.0, _safe_float(signal.liquidity, 0.0)),
        max_position_size=max(0.0, _safe_float(payload.get("max_position_size"), 0.0)),
        resolution_date=resolution_date,
        positions_to_take=positions,
    )


def _paper_exit_prices(
    positions: list[dict[str, Any]],
    settings: dict[str, Any],
) -> tuple[Optional[float], Optional[float]]:
    enabled = bool(settings.get("paper_enable_spread_exits", True))
    if not enabled or not positions:
        return None, None

    prices = [_safe_float(pos.get("price"), 0.0) for pos in positions]
    prices = [price for price in prices if price > 0]
    if not prices:
        return None, None

    avg_entry = sum(prices) / len(prices)
    take_profit_pct = max(0.0, _safe_float(settings.get("paper_take_profit_pct"), 5.0))
    stop_loss_pct = max(0.0, _safe_float(settings.get("paper_stop_loss_pct"), 10.0))

    take_profit_price = None
    stop_loss_price = None
    if take_profit_pct > 0 and avg_entry > 0:
        take_profit_price = min(avg_entry * (1.0 + take_profit_pct / 100.0), 0.99)
    if stop_loss_pct > 0 and avg_entry > 0:
        stop_loss_price = max(avg_entry * (1.0 - stop_loss_pct / 100.0), 0.01)
    return take_profit_price, stop_loss_price


async def _execute_paper_trade(
    signal: TradeSignal,
    notional_usd: float,
    paper_settings: dict[str, Any],
) -> tuple[str, Optional[float], Optional[str], dict[str, Any]]:
    account_id = str(paper_settings.get("paper_account_id") or "").strip()
    if not account_id:
        return "failed", None, "Paper account is not configured", {}

    account = await simulation_service.get_account(account_id)
    if account is None:
        return "failed", None, f"Paper account not found: {account_id}", {}

    opportunity = _build_simulation_opportunity(signal)
    if opportunity is None:
        return "failed", None, "Unable to build simulation opportunity", {"account_id": account_id}

    quantity = max(1.0, notional_usd / max(opportunity.total_cost, 0.01))
    take_profit_price, stop_loss_price = _paper_exit_prices(
        opportunity.positions_to_take,
        paper_settings,
    )

    sim_trade = await simulation_service.execute_opportunity(
        account_id=account_id,
        opportunity=opportunity,
        position_size=quantity,
        take_profit_price=take_profit_price,
        stop_loss_price=stop_loss_price,
    )

    effective_price = _safe_float(signal.effective_price, _safe_float(signal.entry_price, 0.0))
    if effective_price <= 0:
        prices = [
            _safe_float(pos.get("price"), 0.0)
            for pos in opportunity.positions_to_take
            if _safe_float(pos.get("price"), 0.0) > 0
        ]
        if prices:
            effective_price = sum(prices) / len(prices)
        else:
            effective_price = None

    payload: dict[str, Any] = {
        "simulation_trade_id": sim_trade.id,
        "simulation_account_id": account_id,
        "simulation_opportunity_id": sim_trade.opportunity_id,
        "simulation_status": _enum_text(sim_trade.status),
        "shares": quantity,
        "take_profit_price": take_profit_price,
        "stop_loss_price": stop_loss_price,
        "unrealized_pnl": 0.0,
    }
    return "open", effective_price, None, payload


async def _reconcile_paper_trades(session) -> dict[str, Any]:
    rows = list(
        (
            await session.execute(
                select(AutoTraderTrade).where(
                    and_(
                        AutoTraderTrade.mode == "paper",
                        AutoTraderTrade.status.in_(OPEN_AUTOTRADER_STATUSES),
                    )
                )
            )
        )
        .scalars()
        .all()
    )
    if not rows:
        return {"updated": 0, "resolved": 0, "unrealized_pnl": 0.0}

    trades_by_sim_id: dict[str, AutoTraderTrade] = {}
    for row in rows:
        payload = row.payload_json if isinstance(row.payload_json, dict) else {}
        sim_trade_id = payload.get("simulation_trade_id")
        if sim_trade_id:
            trades_by_sim_id[str(sim_trade_id)] = row

    if not trades_by_sim_id:
        return {"updated": 0, "resolved": 0, "unrealized_pnl": 0.0}

    sim_rows = list(
        (
            await session.execute(
                select(SimulationTrade).where(SimulationTrade.id.in_(list(trades_by_sim_id.keys())))
            )
        )
        .scalars()
        .all()
    )
    sim_by_id = {str(row.id): row for row in sim_rows}

    account_ids = {str(row.account_id) for row in sim_rows if row.account_id}
    opportunity_ids = {str(row.opportunity_id) for row in sim_rows if row.opportunity_id}
    unrealized_by_key: dict[tuple[str, str], float] = {}
    if account_ids and opportunity_ids:
        pnl_rows = (
            (
                await session.execute(
                    select(
                        SimulationPosition.account_id,
                        SimulationPosition.opportunity_id,
                        func.coalesce(func.sum(SimulationPosition.unrealized_pnl), 0.0),
                    )
                    .where(
                        and_(
                            SimulationPosition.status == TradeStatus.OPEN,
                            SimulationPosition.account_id.in_(list(account_ids)),
                            SimulationPosition.opportunity_id.in_(list(opportunity_ids)),
                        )
                    )
                    .group_by(SimulationPosition.account_id, SimulationPosition.opportunity_id)
                )
            )
            .all()
        )
        for account_id, opportunity_id, unrealized in pnl_rows:
            if account_id and opportunity_id:
                unrealized_by_key[(str(account_id), str(opportunity_id))] = float(unrealized or 0.0)

    updated = 0
    resolved = 0
    for sim_trade_id, auto_trade in trades_by_sim_id.items():
        sim_trade = sim_by_id.get(sim_trade_id)
        if sim_trade is None:
            continue

        existing_payload = auto_trade.payload_json if isinstance(auto_trade.payload_json, dict) else {}
        payload = dict(existing_payload)
        sim_status = _enum_text(sim_trade.status)
        account_id = str(sim_trade.account_id or payload.get("simulation_account_id") or "")
        opportunity_id = str(sim_trade.opportunity_id or payload.get("simulation_opportunity_id") or "")
        unrealized = 0.0
        if sim_status == "open" and account_id and opportunity_id:
            unrealized = float(unrealized_by_key.get((account_id, opportunity_id), 0.0))

        changed = False
        payload["simulation_status"] = sim_status
        payload["simulation_trade_id"] = sim_trade_id
        payload["simulation_account_id"] = account_id or payload.get("simulation_account_id")
        payload["simulation_opportunity_id"] = opportunity_id or payload.get("simulation_opportunity_id")
        payload["unrealized_pnl"] = unrealized if sim_status == "open" else 0.0

        if sim_status == "open":
            if auto_trade.status != "open":
                auto_trade.status = "open"
                changed = True
            if auto_trade.executed_at is None:
                auto_trade.executed_at = sim_trade.executed_at or auto_trade.created_at
                changed = True
        elif sim_status in {"resolved_win", "resolved_loss"}:
            pnl = _safe_float(sim_trade.actual_pnl, 0.0)
            payload["actual_profit"] = pnl
            payload["resolved_at"] = _to_iso(sim_trade.resolved_at)
            if auto_trade.status != sim_status:
                auto_trade.status = sim_status
                changed = True
            if auto_trade.actual_profit != pnl:
                auto_trade.actual_profit = pnl
                changed = True
            if auto_trade.executed_at is None:
                auto_trade.executed_at = sim_trade.executed_at or auto_trade.created_at
                changed = True
            resolved += 1
        elif sim_status in {"cancelled", "failed"}:
            if auto_trade.status != "failed":
                auto_trade.status = "failed"
                changed = True

        if payload != existing_payload:
            auto_trade.payload_json = payload
            changed = True

        if changed:
            auto_trade.updated_at = utcnow()
            updated += 1

    if updated > 0:
        await session.commit()

    open_payload_rows = (
        (
            await session.execute(
                select(AutoTraderTrade.payload_json).where(
                    and_(
                        AutoTraderTrade.mode == "paper",
                        AutoTraderTrade.status.in_(OPEN_AUTOTRADER_STATUSES),
                    )
                )
            )
        )
        .all()
    )
    unrealized_pnl_total = 0.0
    for (payload,) in open_payload_rows:
        if not isinstance(payload, dict):
            continue
        try:
            unrealized_pnl_total += float(payload.get("unrealized_pnl") or 0.0)
        except Exception:
            continue

    return {
        "updated": updated,
        "resolved": resolved,
        "unrealized_pnl": unrealized_pnl_total,
    }


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
                        AutoTraderTrade.status.in_(OPEN_AUTOTRADER_STATUSES),
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
                    AutoTraderTrade.status.in_(OPEN_AUTOTRADER_STATUSES)
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
                .where(AutoTraderTrade.status.in_(OPEN_AUTOTRADER_STATUSES))
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
    now = utcnow()

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
    control_settings = control.get("settings") if isinstance(control.get("settings"), dict) else {}
    enabled_strategies_raw = (
        control_settings.get("enabled_strategies")
        if isinstance(control_settings.get("enabled_strategies"), list)
        else []
    )
    enabled_strategies = {
        _normalize_strategy(item)
        for item in enabled_strategies_raw
        if isinstance(item, str) and _normalize_strategy(item)
    }
    active_domains = set(normalize_trading_domains(control_settings.get("trading_domains")))

    if (not enabled or paused or kill_switch) and not requested_run:
        return {
            "idle": True,
            "carry_snapshot": True,
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
            "carry_snapshot": True,
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

    paper_reconcile: dict[str, Any] = {"updated": 0, "resolved": 0, "unrealized_pnl": 0.0}
    async with AsyncSessionLocal() as session:
        if mode == "paper":
            paper_reconcile = await _reconcile_paper_trades(session)
        await expire_stale_signals(session)
        pending_signals = await list_pending_trade_signals(
            session,
            sources=enabled_sources if enabled_sources else None,
            limit=2000,
        )
        exposure = await _load_exposure_state(session)

    eligible_signals: list[TradeSignal] = []
    blocked_signals: list[TradeSignal] = []
    for signal in pending_signals:
        source_domain = SOURCE_DOMAIN_MAP.get(str(signal.source), "event_markets")
        if source_domain in active_domains:
            eligible_signals.append(signal)
        else:
            blocked_signals.append(signal)

    if blocked_signals:
        for signal in blocked_signals:
            source_policy = source_policies.get(signal.source) or {}
            async with AsyncSessionLocal() as session:
                await set_trade_signal_status(session, signal.id, "skipped")
                await create_autotrader_decision(
                    session,
                    signal_id=signal.id,
                    source=signal.source,
                    decision="skipped",
                    reason=f"Domain {SOURCE_DOMAIN_MAP.get(str(signal.source), 'event_markets')} disabled by settings",
                    score=None,
                    policy_snapshot=source_policy,
                    risk_snapshot={"rule": "domain_disabled"},
                    payload={
                        "active_domains": sorted(active_domains),
                        "source_domain": SOURCE_DOMAIN_MAP.get(str(signal.source), "event_markets"),
                    },
                )
            decisions_count += 1

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
    for signal in eligible_signals:
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
        strategy_key = _normalize_strategy(signal.strategy_type) or _normalize_strategy(signal.signal_type)

        if not _strategy_enabled(signal, enabled_strategies):
            async with AsyncSessionLocal() as session:
                await set_trade_signal_status(session, signal.id, "skipped")
                await create_autotrader_decision(
                    session,
                    signal_id=signal.id,
                    source=signal.source,
                    decision="skipped",
                    reason=f"Strategy {strategy_key or 'unknown'} disabled by settings",
                    score=score,
                    policy_snapshot=source_policy,
                    risk_snapshot={"rule": "strategy_disabled"},
                    payload={
                        "strategy_type": signal.strategy_type,
                        "signal_type": signal.signal_type,
                        "enabled_strategies": sorted(enabled_strategies),
                    },
                )
            decisions_count += 1
            continue

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
        else:
            try:
                trade_status, paper_price, err, execution_payload = await _execute_paper_trade(
                    signal,
                    size_usd,
                    control_settings,
                )
                if paper_price is not None:
                    effective_price = paper_price
                error_message = err
            except Exception as exc:
                trade_status = "failed"
                error_message = str(exc)
                execution_payload = {"error": str(exc)}
                errors.append(str(exc))

        decision_outcome = "executed" if trade_status == "open" else trade_status
        if trade_status == "failed":
            decision_reason = error_message or "Execution failed"
        elif mode == "live":
            decision_reason = "Submitted live order" if trade_status == "submitted" else "Executed live order"
        else:
            decision_reason = "Opened paper simulation position"

        async with AsyncSessionLocal() as session:
            decision = await create_autotrader_decision(
                session,
                signal_id=signal.id,
                source=signal.source,
                decision=decision_outcome,
                reason=decision_reason,
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
                    "execution_status": trade_status,
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
                    "unrealized_pnl": float(execution_payload.get("unrealized_pnl") or 0.0),
                },
                error_message=error_message,
            )

            if trade_status == "submitted":
                final_signal_status = "submitted"
            elif trade_status in {"executed", "open"}:
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
        if trade_status in OPEN_AUTOTRADER_STATUSES:
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
        total_trades = int(
            (
                await session.execute(select(func.count(AutoTraderTrade.id)))
            ).scalar()
            or 0
        )
        open_positions = int(
            (
                await session.execute(
                    select(func.count(AutoTraderTrade.id)).where(
                        AutoTraderTrade.status.in_(OPEN_AUTOTRADER_STATUSES)
                    )
                )
            ).scalar()
            or 0
        )
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        realized_daily_pnl = float(
            (
                await session.execute(
                    select(func.coalesce(func.sum(AutoTraderTrade.actual_profit), 0.0)).where(
                        and_(
                            AutoTraderTrade.updated_at >= today_start,
                            AutoTraderTrade.status.in_(("resolved_win", "resolved_loss")),
                        )
                    )
                )
            ).scalar()
            or 0.0
        )

    return {
        "idle": False,
        "carry_snapshot": False,
        "interval": run_interval,
        "mode": mode,
        "signals_seen": len(eligible_signals),
        "signals_blocked_by_domain": len(blocked_signals),
        "signals_selected": signals_selected,
        "signals_seen_total": total_decisions,
        "signals_selected_total": total_trades,
        "decisions_count": decisions_count,
        "trades_count": trades_count,
        "decisions_total": total_decisions,
        "trades_total": total_trades,
        "open_positions": open_positions,
        "daily_pnl": realized_daily_pnl + (paper_reconcile.get("unrealized_pnl", 0.0) if mode == "paper" else 0.0),
        "errors": errors,
        "total_decisions": total_decisions,
        "active_domains": sorted(active_domains),
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
        cycle_started = utcnow()
        cycle = await _run_cycle()
        interval = int(cycle.get("interval") or 2)

        activity = "Idle"
        if cycle.get("idle"):
            reason = cycle.get("reason") or "paused"
            activity = f"Autotrader idle ({reason})"
        else:
            cycle_seen = int(cycle.get("signals_seen", 0))
            cycle_selected = int(cycle.get("signals_selected", 0))
            total_seen = int(cycle.get("signals_seen_total", cycle_seen))
            total_selected = int(cycle.get("signals_selected_total", cycle_selected))
            activity = (
                f"Cycle processed {cycle_seen} pending signals, selected {cycle_selected} "
                f"(totals seen {total_seen}, selected {total_selected})"
            )

        errors = cycle.get("errors") or []
        last_error = errors[-1] if errors else None

        async with AsyncSessionLocal() as session:
            existing_snapshot = await read_autotrader_snapshot(session)
            snapshot_signals_seen = int(
                cycle.get("signals_seen_total", cycle.get("signals_seen", existing_snapshot.get("signals_seen", 0)))
            )
            snapshot_signals_selected = int(
                cycle.get(
                    "signals_selected_total",
                    cycle.get("signals_selected", existing_snapshot.get("signals_selected", 0)),
                )
            )
            snapshot_decisions_count = int(
                cycle.get("decisions_total", cycle.get("decisions_count", existing_snapshot.get("decisions_count", 0)))
            )
            snapshot_trades_count = int(
                cycle.get("trades_total", cycle.get("trades_count", existing_snapshot.get("trades_count", 0)))
            )
            snapshot_open_positions = int(cycle.get("open_positions", existing_snapshot.get("open_positions", 0)))
            snapshot_daily_pnl = float(cycle.get("daily_pnl", existing_snapshot.get("daily_pnl", 0.0)))
            if cycle.get("carry_snapshot"):
                snapshot_signals_seen = int(existing_snapshot.get("signals_seen", 0))
                snapshot_signals_selected = int(existing_snapshot.get("signals_selected", 0))
                snapshot_decisions_count = int(existing_snapshot.get("decisions_count", 0))
                snapshot_trades_count = int(existing_snapshot.get("trades_count", 0))
                snapshot_open_positions = int(existing_snapshot.get("open_positions", 0))
                snapshot_daily_pnl = float(existing_snapshot.get("daily_pnl", 0.0))

            await write_autotrader_snapshot(
                session,
                running=True,
                enabled=not cycle.get("idle", False),
                current_activity=activity,
                interval_seconds=interval,
                last_run_at=cycle_started,
                signals_seen=snapshot_signals_seen,
                signals_selected=snapshot_signals_selected,
                decisions_count=snapshot_decisions_count,
                trades_count=snapshot_trades_count,
                open_positions=snapshot_open_positions,
                daily_pnl=snapshot_daily_pnl,
                last_error=last_error,
                stats={
                    "mode": cycle.get("mode"),
                    "total_decisions": int(cycle.get("total_decisions", 0)),
                    "cycle_signals_seen": int(cycle.get("signals_seen", 0)),
                    "cycle_signals_blocked_by_domain": int(cycle.get("signals_blocked_by_domain", 0)),
                    "cycle_signals_selected": int(cycle.get("signals_selected", 0)),
                    "cycle_decisions_count": int(cycle.get("decisions_count", 0)),
                    "cycle_trades_count": int(cycle.get("trades_count", 0)),
                    "active_domains": list(cycle.get("active_domains") or []),
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
                    "active_domains": list(cycle.get("active_domains") or []),
                    "signals_seen_cycle": int(cycle.get("signals_seen", 0)),
                    "signals_blocked_by_domain_cycle": int(cycle.get("signals_blocked_by_domain", 0)),
                    "signals_selected_cycle": int(cycle.get("signals_selected", 0)),
                    "signals_seen_total": snapshot_signals_seen,
                    "signals_selected_total": snapshot_signals_selected,
                    "decisions_count_cycle": int(cycle.get("decisions_count", 0)),
                    "trades_count_cycle": int(cycle.get("trades_count", 0)),
                    "decisions_count_total": snapshot_decisions_count,
                    "trades_count_total": snapshot_trades_count,
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
