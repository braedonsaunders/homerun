from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import TradeSignal, TraderOrder
from services.polymarket import polymarket_client
from services.simulation import simulation_service
from utils.utcnow import utcnow
from utils.converters import safe_float

logger = logging.getLogger("position_lifecycle")

PAPER_ACTIVE_STATUSES = {"submitted", "executed", "open"}
LIVE_ACTIVE_STATUSES = {"submitted", "executed", "open"}


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _direction_outcome_index(direction: Any) -> Optional[int]:
    normalized = str(direction or "").strip().lower()
    if normalized in {"buy_yes", "yes", "up", "long_yes"}:
        return 0
    if normalized in {"buy_no", "no", "down", "long_no"}:
        return 1
    return None


def _extract_signal_side_price(payload: dict[str, Any], outcome_idx: int) -> Optional[float]:
    side_keys = ("yes", "up") if outcome_idx == 0 else ("no", "down")
    for prefix in side_keys:
        for key in (
            f"{prefix}_price",
            f"{prefix}Price",
            f"best_{prefix}",
            f"best{prefix.title()}",
            f"{prefix}_mid",
            f"{prefix}Mid",
        ):
            parsed = safe_float(payload.get(key))
            if parsed is not None and parsed >= 0:
                return parsed

    prices = payload.get("outcome_prices")
    if not isinstance(prices, list):
        prices = payload.get("outcomePrices")
    if isinstance(prices, list) and len(prices) > outcome_idx:
        parsed = safe_float(prices[outcome_idx])
        if parsed is not None and parsed >= 0:
            return parsed
    return None


def _extract_market_side_price(market_info: Optional[dict[str, Any]], outcome_idx: int) -> Optional[float]:
    if not isinstance(market_info, dict):
        return None
    key = "yes_price" if outcome_idx == 0 else "no_price"
    parsed = safe_float(market_info.get(key))
    if parsed is not None and parsed >= 0:
        return parsed
    prices = market_info.get("outcome_prices")
    if isinstance(prices, list) and len(prices) > outcome_idx:
        parsed = safe_float(prices[outcome_idx])
        if parsed is not None and parsed >= 0:
            return parsed
    return None


def _extract_winning_outcome_index(market_info: Optional[dict[str, Any]]) -> Optional[int]:
    if not isinstance(market_info, dict):
        return None

    winner_raw = (
        market_info.get("winning_outcome")
        if market_info.get("winning_outcome") not in (None, "")
        else market_info.get("winner")
    )
    if winner_raw in (None, ""):
        return None

    outcomes_raw = market_info.get("outcomes")
    outcomes: list[str] = []
    if isinstance(outcomes_raw, list):
        outcomes = [str(item or "").strip().lower() for item in outcomes_raw if str(item or "").strip()]

    try:
        idx = int(winner_raw)
        if idx in (0, 1):
            return idx
    except Exception:
        pass

    winner_text = str(winner_raw).strip().lower()
    if winner_text in {"yes", "up", "true"}:
        return 0
    if winner_text in {"no", "down", "false"}:
        return 1
    if outcomes:
        for idx, label in enumerate(outcomes):
            if label == winner_text and idx in (0, 1):
                return idx
    return None


def _extract_winning_outcome_index_from_prices(
    market_info: Optional[dict[str, Any]],
    *,
    market_tradable: bool,
    settle_floor: float,
) -> Optional[int]:
    if not isinstance(market_info, dict):
        return None
    if market_tradable:
        return None

    settle_floor = min(1.0, max(0.5, settle_floor))
    settle_ceiling = max(0.0, 1.0 - settle_floor)

    yes_price = safe_float(market_info.get("yes_price"))
    no_price = safe_float(market_info.get("no_price"))
    if yes_price is None or no_price is None:
        prices = market_info.get("outcome_prices")
        if isinstance(prices, list) and len(prices) >= 2:
            outcomes = market_info.get("outcomes")
            if isinstance(outcomes, list) and len(outcomes) == len(prices):
                for idx, raw_label in enumerate(outcomes):
                    label = str(raw_label or "").strip().lower()
                    parsed = safe_float(prices[idx])
                    if parsed is None:
                        continue
                    if yes_price is None and label in {"yes", "up", "true"}:
                        yes_price = parsed
                    if no_price is None and label in {"no", "down", "false"}:
                        no_price = parsed
            if yes_price is None:
                yes_price = safe_float(prices[0])
            if no_price is None:
                no_price = safe_float(prices[1])

    yes_price = _state_price_floor(yes_price)
    no_price = _state_price_floor(no_price)
    if yes_price is None or no_price is None:
        return None
    if yes_price >= settle_floor and no_price <= settle_ceiling:
        return 0
    if no_price >= settle_floor and yes_price <= settle_ceiling:
        return 1
    return None


def _state_price_floor(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _extract_position_state(payload: dict[str, Any]) -> dict[str, Any]:
    state = payload.get("position_state")
    return state if isinstance(state, dict) else {}


async def load_market_info_for_orders(orders: list[TraderOrder]) -> dict[str, Optional[dict[str, Any]]]:
    market_ids = sorted({str(order.market_id or "").strip() for order in orders if str(order.market_id or "").strip()})
    if not market_ids:
        return {}

    async def _fetch(market_id: str) -> tuple[str, Optional[dict[str, Any]]]:
        info: Optional[dict[str, Any]] = None
        if market_id.startswith("0x"):
            info = await polymarket_client.get_market_by_condition_id(market_id)
        if info is None:
            info = await polymarket_client.get_market_by_token_id(market_id)
        return market_id, info

    pairs = await asyncio.gather(*[_fetch(market_id) for market_id in market_ids], return_exceptions=True)
    out: dict[str, Optional[dict[str, Any]]] = {}
    for item in pairs:
        if isinstance(item, Exception):
            continue
        market_id, info = item
        out[market_id] = info
    return out


async def reconcile_paper_positions(
    session: AsyncSession,
    *,
    trader_id: str,
    trader_params: Optional[dict[str, Any]] = None,
    dry_run: bool = False,
    force_mark_to_market: bool = False,
    max_age_hours: Optional[int] = None,
    reason: str = "paper_position_lifecycle",
) -> dict[str, Any]:
    params = dict(trader_params or {})
    take_profit_pct = safe_float(params.get("paper_take_profit_pct"))
    stop_loss_pct = safe_float(params.get("paper_stop_loss_pct"))
    max_hold_minutes = safe_float(params.get("paper_max_hold_minutes"))
    min_hold_minutes = max(0.0, safe_float(params.get("paper_min_hold_minutes")) or 0.0)
    trailing_stop_pct = safe_float(params.get("paper_trailing_stop_pct"))
    resolve_only = _safe_bool(params.get("paper_resolve_only"), False)
    close_on_inactive_market = _safe_bool(params.get("paper_close_on_inactive_market"), False)
    resolution_infer_from_prices = _safe_bool(params.get("paper_resolution_infer_from_prices"), True)
    resolution_settle_floor = min(
        1.0,
        max(0.5, safe_float(params.get("paper_resolution_settle_floor")) or 0.98),
    )

    candidates = list(
        (
            await session.execute(
                select(TraderOrder).where(
                    TraderOrder.trader_id == trader_id,
                    func.lower(func.coalesce(TraderOrder.mode, "")) == "paper",
                    func.lower(func.coalesce(TraderOrder.status, "")).in_(tuple(PAPER_ACTIVE_STATUSES)),
                )
            )
        )
        .scalars()
        .all()
    )

    if max_age_hours is not None:
        cutoff = utcnow() - timedelta(hours=max(1, int(max_age_hours)))
        candidates = [
            row
            for row in candidates
            if (row.executed_at or row.updated_at or row.created_at) is not None
            and (row.executed_at or row.updated_at or row.created_at) <= cutoff
        ]

    signal_ids = [str(row.signal_id) for row in candidates if row.signal_id]
    signal_payloads: dict[str, dict[str, Any]] = {}
    if signal_ids:
        signal_rows = (
            await session.execute(
                select(TradeSignal.id, TradeSignal.payload_json).where(TradeSignal.id.in_(signal_ids))
            )
        ).all()
        signal_payloads = {str(row.id): dict(row.payload_json or {}) for row in signal_rows}

    market_info_by_id = await load_market_info_for_orders(candidates)

    now = utcnow()
    would_close = 0
    closed = 0
    held = 0
    skipped = 0
    total_realized_pnl = 0.0
    by_status = {"resolved_win": 0, "resolved_loss": 0}
    skipped_reasons: dict[str, int] = {}
    details: list[dict[str, Any]] = []
    state_updates = 0

    for row in candidates:
        entry_price = safe_float(row.effective_price)
        if entry_price is None or entry_price <= 0:
            entry_price = safe_float(row.entry_price)
        notional = safe_float(row.notional_usd) or 0.0
        outcome_idx = _direction_outcome_index(row.direction)
        if outcome_idx is None or entry_price is None or entry_price <= 0 or notional <= 0:
            skipped += 1
            skipped_reasons["invalid_entry"] = int(skipped_reasons.get("invalid_entry", 0)) + 1
            continue

        signal_payload = signal_payloads.get(str(row.signal_id), {})
        market_info = market_info_by_id.get(str(row.market_id or ""))
        market_tradable = polymarket_client.is_market_tradable(market_info, now=now)
        winning_idx = _extract_winning_outcome_index(market_info)
        winning_idx_inferred = False
        if winning_idx is None and resolution_infer_from_prices:
            inferred_idx = _extract_winning_outcome_index_from_prices(
                market_info,
                market_tradable=market_tradable,
                settle_floor=resolution_settle_floor,
            )
            if inferred_idx is not None:
                winning_idx = inferred_idx
                winning_idx_inferred = True
        market_side_price = _extract_market_side_price(market_info, outcome_idx)
        snapshot_side_price = _extract_signal_side_price(signal_payload, outcome_idx)

        close_price: Optional[float] = None
        close_trigger: Optional[str] = None
        price_source: Optional[str] = None
        trailing_trigger_price: Optional[float] = None

        current_price = market_side_price if market_side_price is not None else snapshot_side_price
        current_price = _state_price_floor(current_price)
        current_price_source = (
            "market_mark"
            if market_side_price is not None
            else ("signal_snapshot_mark" if snapshot_side_price is not None else None)
        )

        age_anchor = row.executed_at or row.updated_at or row.created_at
        age_minutes = None
        if age_anchor is not None:
            age_minutes = max(0.0, (now - age_anchor).total_seconds() / 60.0)
        min_hold_passed = age_minutes is None or age_minutes >= min_hold_minutes

        payload = dict(row.payload_json or {})
        position_state = _extract_position_state(payload)
        prev_high = safe_float(position_state.get("highest_price"))
        prev_low = safe_float(position_state.get("lowest_price"))
        prev_last_mark = safe_float(position_state.get("last_mark_price"))
        prev_mark_source = str(position_state.get("last_mark_source") or "")
        highest_price = prev_high
        lowest_price = prev_low
        if current_price is not None:
            if highest_price is None:
                highest_price = current_price
            else:
                highest_price = max(highest_price, current_price)
            if lowest_price is None:
                lowest_price = current_price
            else:
                lowest_price = min(lowest_price, current_price)

        next_state = {
            "highest_price": _state_price_floor(highest_price),
            "lowest_price": _state_price_floor(lowest_price),
            "last_mark_price": current_price,
            "last_mark_source": current_price_source,
            "last_marked_at": now.isoformat() + "Z",
        }

        if winning_idx is not None:
            close_price = 1.0 if winning_idx == outcome_idx else 0.0
            close_trigger = "resolution_inferred" if winning_idx_inferred else "resolution"
            price_source = "resolved_settlement"
        else:
            pnl_pct = None
            if current_price is not None and entry_price > 0:
                pnl_pct = ((current_price - entry_price) / entry_price) * 100.0

            if force_mark_to_market and current_price is not None:
                close_price = current_price
                close_trigger = "manual_mark_to_market"
                price_source = current_price_source
            else:
                # ── Strategy-based exit check ──────────────────────────
                # If the strategy that opened this position has a
                # should_exit() method, call it first and respect its
                # decision before falling through to default TP/SL/etc.
                strategy_slug = (payload.get("strategy_type") or "").strip().lower()
                strategy_exit = None
                _exit_instance = None
                if strategy_slug:
                    from services.strategy_loader import strategy_loader

                    loaded = strategy_loader.get_strategy(strategy_slug)
                    if loaded and hasattr(loaded.instance, "should_exit"):
                        _exit_instance = loaded.instance
                if _exit_instance is not None:
                    try:

                        class _PaperPositionView:
                            pass

                        pos_view = _PaperPositionView()
                        pos_view.entry_price = entry_price
                        pos_view.current_price = current_price
                        pos_view.highest_price = highest_price
                        pos_view.lowest_price = lowest_price
                        pos_view.age_minutes = age_minutes
                        pos_view.pnl_percent = pnl_pct
                        pos_view.strategy_context = payload.get("strategy_context", {})
                        pos_view.config = payload.get("strategy_exit_config", {})
                        pos_view.outcome_idx = outcome_idx

                        market_state_dict = {
                            "current_price": current_price,
                            "market_tradable": market_tradable,
                            "is_resolved": False,
                            "winning_outcome": None,
                        }

                        exit_decision = _exit_instance.should_exit(pos_view, market_state_dict)
                        if exit_decision is not None and getattr(exit_decision, "action", None) == "close":
                            strategy_exit = exit_decision
                    except Exception as exc:
                        logger.warning(
                            "Strategy should_exit() error for %s: %s",
                            strategy_slug,
                            exc,
                        )

                if strategy_exit is not None:
                    close_price = strategy_exit.close_price if strategy_exit.close_price is not None else current_price
                    close_trigger = f"strategy:{strategy_exit.reason}"
                    price_source = current_price_source
                elif (
                    not resolve_only
                    and take_profit_pct is not None
                    and pnl_pct is not None
                    and min_hold_passed
                    and pnl_pct >= take_profit_pct
                ):
                    close_price = current_price
                    close_trigger = "take_profit"
                    price_source = current_price_source
                elif (
                    not resolve_only
                    and stop_loss_pct is not None
                    and pnl_pct is not None
                    and min_hold_passed
                    and pnl_pct <= -abs(stop_loss_pct)
                ):
                    close_price = current_price
                    close_trigger = "stop_loss"
                    price_source = current_price_source
                elif (
                    not resolve_only
                    and trailing_stop_pct is not None
                    and trailing_stop_pct > 0
                    and current_price is not None
                    and highest_price is not None
                    and min_hold_passed
                ):
                    trailing_trigger_price = highest_price * (1.0 - (trailing_stop_pct / 100.0))
                    if highest_price > entry_price and current_price <= trailing_trigger_price:
                        close_price = current_price
                        close_trigger = "trailing_stop"
                        price_source = current_price_source
                elif (
                    not resolve_only
                    and max_hold_minutes is not None
                    and age_minutes is not None
                    and age_minutes >= max_hold_minutes
                ):
                    if current_price is not None:
                        close_price = current_price
                        close_trigger = "max_hold"
                        price_source = current_price_source
                elif (
                    not resolve_only
                    and close_on_inactive_market
                    and not market_tradable
                    and current_price is not None
                    and min_hold_passed
                ):
                    close_price = current_price
                    close_trigger = "market_inactive"
                    price_source = current_price_source

        if close_price is None:
            state_changed = False
            if current_price is not None:
                state_changed = (
                    prev_last_mark is None
                    or abs(prev_last_mark - current_price) > 1e-9
                    or prev_high is None
                    or prev_low is None
                    or abs((prev_high or 0.0) - (highest_price or 0.0)) > 1e-9
                    or abs((prev_low or 0.0) - (lowest_price or 0.0)) > 1e-9
                    or prev_mark_source != str(current_price_source or "")
                )
            if not dry_run and state_changed:
                payload["position_state"] = next_state
                row.payload_json = payload
                row.updated_at = now
                state_updates += 1
            held += 1
            continue

        quantity = notional / entry_price
        proceeds = quantity * close_price
        pnl = proceeds - notional
        next_status = "resolved_win" if pnl >= 0 else "resolved_loss"

        simulation_close: dict[str, Any] | None = None
        simulation_ledger = payload.get("simulation_ledger")
        if not dry_run and isinstance(simulation_ledger, dict):
            sim_account_id = str(simulation_ledger.get("account_id") or "").strip()
            sim_trade_id = str(simulation_ledger.get("trade_id") or "").strip()
            sim_position_id = str(simulation_ledger.get("position_id") or "").strip()
            if sim_account_id and sim_trade_id and sim_position_id:
                try:
                    simulation_close = await simulation_service.close_orchestrator_paper_fill(
                        account_id=sim_account_id,
                        trade_id=sim_trade_id,
                        position_id=sim_position_id,
                        close_price=float(close_price),
                        close_trigger=close_trigger,
                        price_source=price_source,
                        reason=reason,
                        session=session,
                        commit=False,
                    )
                    if simulation_close.get("closed"):
                        proceeds = float(simulation_close.get("actual_payout", proceeds))
                        pnl = float(simulation_close.get("actual_pnl", pnl))
                        next_status = str(simulation_close.get("trade_status") or next_status)
                    elif simulation_close.get("already_closed"):
                        existing_status = str(simulation_close.get("trade_status") or "")
                        if existing_status:
                            next_status = existing_status
                        pnl = float(simulation_close.get("actual_pnl", pnl))
                        proceeds = float(simulation_close.get("actual_payout", proceeds))
                except Exception as exc:
                    skipped += 1
                    skipped_reasons["simulation_close_error"] = (
                        int(skipped_reasons.get("simulation_close_error", 0)) + 1
                    )
                    details.append(
                        {
                            "order_id": row.id,
                            "market_id": row.market_id,
                            "direction": row.direction,
                            "close_trigger": close_trigger,
                            "reason": "simulation_close_error",
                            "error": str(exc),
                        }
                    )
                    continue

        total_realized_pnl += pnl
        by_status[next_status] = int(by_status.get(next_status, 0)) + 1

        detail = {
            "order_id": row.id,
            "market_id": row.market_id,
            "direction": row.direction,
            "entry_price": entry_price,
            "close_price": close_price,
            "price_source": price_source,
            "close_trigger": close_trigger,
            "market_tradable": market_tradable,
            "notional_usd": notional,
            "quantity": quantity,
            "realized_pnl": pnl,
            "next_status": next_status,
            "age_minutes": age_minutes,
            "min_hold_minutes": min_hold_minutes,
            "trailing_stop_trigger_price": trailing_trigger_price,
            "highest_price_seen": _state_price_floor(highest_price),
            "lowest_price_seen": _state_price_floor(lowest_price),
            "simulation_close": simulation_close,
        }
        details.append(detail)
        would_close += 1

        if dry_run:
            continue

        row.status = next_status
        row.actual_profit = pnl
        row.updated_at = now
        payload["position_state"] = next_state
        payload["position_close"] = {
            "close_price": close_price,
            "price_source": price_source,
            "close_trigger": close_trigger,
            "realized_pnl": pnl,
            "market_tradable": market_tradable,
            "age_minutes": age_minutes,
            "closed_at": now.isoformat() + "Z",
            "reason": reason,
        }
        if simulation_close is not None:
            payload["position_close"]["simulation_close"] = simulation_close
        row.payload_json = payload
        if reason:
            if row.reason:
                row.reason = f"{row.reason} | {reason}:{close_trigger}"
            else:
                row.reason = f"{reason}:{close_trigger}"
        closed += 1

    if not dry_run and (closed > 0 or state_updates > 0):
        await session.commit()

    return {
        "trader_id": trader_id,
        "dry_run": bool(dry_run),
        "matched": len(candidates),
        "would_close": would_close,
        "closed": closed,
        "held": held,
        "skipped": skipped,
        "state_updates": state_updates,
        "total_realized_pnl": total_realized_pnl,
        "by_status": by_status,
        "skipped_reasons": skipped_reasons,
        "details": details,
    }


async def reconcile_live_positions(
    session: AsyncSession,
    *,
    trader_id: str,
    trader_params: Optional[dict[str, Any]] = None,
    dry_run: bool = False,
    force_mark_to_market: bool = False,
    max_age_hours: Optional[int] = None,
    reason: str = "live_position_lifecycle",
) -> dict[str, Any]:
    """Lifecycle management for live positions.

    Mirrors reconcile_paper_positions but operates on mode='live' orders.
    Handles: stop-loss, take-profit, trailing stop, max hold, market
    inactivity, and resolution detection.  Does NOT interact with the
    simulation ledger (that is paper-only).
    """
    params = dict(trader_params or {})
    take_profit_pct = safe_float(params.get("live_take_profit_pct"))
    stop_loss_pct = safe_float(params.get("live_stop_loss_pct"))
    max_hold_minutes = safe_float(params.get("live_max_hold_minutes"))
    min_hold_minutes = max(0.0, safe_float(params.get("live_min_hold_minutes")) or 0.0)
    trailing_stop_pct = safe_float(params.get("live_trailing_stop_pct"))
    resolve_only = _safe_bool(params.get("live_resolve_only"), False)
    close_on_inactive_market = _safe_bool(params.get("live_close_on_inactive_market"), False)
    resolution_infer_from_prices = _safe_bool(params.get("live_resolution_infer_from_prices"), True)
    resolution_settle_floor = min(
        1.0,
        max(0.5, safe_float(params.get("live_resolution_settle_floor")) or 0.98),
    )

    candidates = list(
        (
            await session.execute(
                select(TraderOrder).where(
                    TraderOrder.trader_id == trader_id,
                    func.lower(func.coalesce(TraderOrder.mode, "")) == "live",
                    func.lower(func.coalesce(TraderOrder.status, "")).in_(tuple(LIVE_ACTIVE_STATUSES)),
                )
            )
        )
        .scalars()
        .all()
    )

    if max_age_hours is not None:
        cutoff = utcnow() - timedelta(hours=max(1, int(max_age_hours)))
        candidates = [
            row
            for row in candidates
            if (row.executed_at or row.updated_at or row.created_at) is not None
            and (row.executed_at or row.updated_at or row.created_at) <= cutoff
        ]

    signal_ids = [str(row.signal_id) for row in candidates if row.signal_id]
    signal_payloads: dict[str, dict[str, Any]] = {}
    if signal_ids:
        signal_rows = (
            await session.execute(
                select(TradeSignal.id, TradeSignal.payload_json).where(TradeSignal.id.in_(signal_ids))
            )
        ).all()
        signal_payloads = {str(row.id): dict(row.payload_json or {}) for row in signal_rows}

    market_info_by_id = await load_market_info_for_orders(candidates)

    now = utcnow()
    would_close = 0
    closed = 0
    held = 0
    skipped = 0
    total_realized_pnl = 0.0
    by_status: dict[str, int] = {"resolved_win": 0, "resolved_loss": 0}
    skipped_reasons: dict[str, int] = {}
    details: list[dict[str, Any]] = []
    state_updates = 0

    for row in candidates:
        entry_price = safe_float(row.effective_price)
        if entry_price is None or entry_price <= 0:
            entry_price = safe_float(row.entry_price)
        notional = safe_float(row.notional_usd) or 0.0
        outcome_idx = _direction_outcome_index(row.direction)
        if outcome_idx is None or entry_price is None or entry_price <= 0 or notional <= 0:
            skipped += 1
            skipped_reasons["invalid_entry"] = int(skipped_reasons.get("invalid_entry", 0)) + 1
            continue

        signal_payload = signal_payloads.get(str(row.signal_id), {})
        market_info = market_info_by_id.get(str(row.market_id or ""))
        market_tradable = polymarket_client.is_market_tradable(market_info, now=now)
        winning_idx = _extract_winning_outcome_index(market_info)
        winning_idx_inferred = False
        if winning_idx is None and resolution_infer_from_prices:
            inferred_idx = _extract_winning_outcome_index_from_prices(
                market_info,
                market_tradable=market_tradable,
                settle_floor=resolution_settle_floor,
            )
            if inferred_idx is not None:
                winning_idx = inferred_idx
                winning_idx_inferred = True
        market_side_price = _extract_market_side_price(market_info, outcome_idx)
        snapshot_side_price = _extract_signal_side_price(signal_payload, outcome_idx)

        close_price: Optional[float] = None
        close_trigger: Optional[str] = None
        price_source: Optional[str] = None
        trailing_trigger_price: Optional[float] = None

        current_price = market_side_price if market_side_price is not None else snapshot_side_price
        current_price = _state_price_floor(current_price)
        current_price_source = (
            "market_mark"
            if market_side_price is not None
            else ("signal_snapshot_mark" if snapshot_side_price is not None else None)
        )

        age_anchor = row.executed_at or row.updated_at or row.created_at
        age_minutes = None
        if age_anchor is not None:
            age_minutes = max(0.0, (now - age_anchor).total_seconds() / 60.0)
        min_hold_passed = age_minutes is None or age_minutes >= min_hold_minutes

        payload = dict(row.payload_json or {})
        position_state = _extract_position_state(payload)
        prev_high = safe_float(position_state.get("highest_price"))
        prev_low = safe_float(position_state.get("lowest_price"))
        prev_last_mark = safe_float(position_state.get("last_mark_price"))
        prev_mark_source = str(position_state.get("last_mark_source") or "")
        highest_price = prev_high
        lowest_price = prev_low
        if current_price is not None:
            if highest_price is None:
                highest_price = current_price
            else:
                highest_price = max(highest_price, current_price)
            if lowest_price is None:
                lowest_price = current_price
            else:
                lowest_price = min(lowest_price, current_price)

        next_state = {
            "highest_price": _state_price_floor(highest_price),
            "lowest_price": _state_price_floor(lowest_price),
            "last_mark_price": current_price,
            "last_mark_source": current_price_source,
            "last_marked_at": now.isoformat() + "Z",
        }

        if winning_idx is not None:
            close_price = 1.0 if winning_idx == outcome_idx else 0.0
            close_trigger = "resolution_inferred" if winning_idx_inferred else "resolution"
            price_source = "resolved_settlement"
        else:
            pnl_pct = None
            if current_price is not None and entry_price > 0:
                pnl_pct = ((current_price - entry_price) / entry_price) * 100.0

            if force_mark_to_market and current_price is not None:
                close_price = current_price
                close_trigger = "manual_mark_to_market"
                price_source = current_price_source
            else:
                # ── Strategy-based exit check ──────────────────────────
                # If the strategy that opened this position has a
                # should_exit() method, call it first and respect its
                # decision before falling through to default TP/SL/etc.
                strategy_slug = (payload.get("strategy_type") or "").strip().lower()
                strategy_exit = None
                if strategy_slug:
                    from services.strategy_loader import strategy_loader

                    loaded = strategy_loader.get_strategy(strategy_slug)
                    if loaded and hasattr(loaded.instance, "should_exit"):
                        try:

                            class _LivePositionView:
                                pass

                            pos_view = _LivePositionView()
                            pos_view.entry_price = entry_price
                            pos_view.current_price = current_price
                            pos_view.highest_price = highest_price
                            pos_view.lowest_price = lowest_price
                            pos_view.age_minutes = age_minutes
                            pos_view.pnl_percent = pnl_pct
                            pos_view.strategy_context = payload.get("strategy_context", {})
                            pos_view.config = payload.get("strategy_exit_config", {})
                            pos_view.outcome_idx = outcome_idx

                            market_state_dict = {
                                "current_price": current_price,
                                "market_tradable": market_tradable,
                                "is_resolved": False,
                                "winning_outcome": None,
                            }

                            exit_decision = loaded.instance.should_exit(pos_view, market_state_dict)
                            if exit_decision is not None and getattr(exit_decision, "action", None) == "close":
                                strategy_exit = exit_decision
                        except Exception as exc:
                            logger.warning(
                                "Strategy should_exit() error for %s: %s",
                                strategy_slug,
                                exc,
                            )

                if strategy_exit is not None:
                    close_price = strategy_exit.close_price if strategy_exit.close_price is not None else current_price
                    close_trigger = f"strategy:{strategy_exit.reason}"
                    price_source = current_price_source
                elif (
                    not resolve_only
                    and take_profit_pct is not None
                    and pnl_pct is not None
                    and min_hold_passed
                    and pnl_pct >= take_profit_pct
                ):
                    close_price = current_price
                    close_trigger = "take_profit"
                    price_source = current_price_source
                elif (
                    not resolve_only
                    and stop_loss_pct is not None
                    and pnl_pct is not None
                    and min_hold_passed
                    and pnl_pct <= -abs(stop_loss_pct)
                ):
                    close_price = current_price
                    close_trigger = "stop_loss"
                    price_source = current_price_source
                elif (
                    not resolve_only
                    and trailing_stop_pct is not None
                    and trailing_stop_pct > 0
                    and current_price is not None
                    and highest_price is not None
                    and min_hold_passed
                ):
                    trailing_trigger_price = highest_price * (1.0 - (trailing_stop_pct / 100.0))
                    if highest_price > entry_price and current_price <= trailing_trigger_price:
                        close_price = current_price
                        close_trigger = "trailing_stop"
                        price_source = current_price_source
                elif (
                    not resolve_only
                    and max_hold_minutes is not None
                    and age_minutes is not None
                    and age_minutes >= max_hold_minutes
                ):
                    if current_price is not None:
                        close_price = current_price
                        close_trigger = "max_hold"
                        price_source = current_price_source
                elif (
                    not resolve_only
                    and close_on_inactive_market
                    and not market_tradable
                    and current_price is not None
                    and min_hold_passed
                ):
                    close_price = current_price
                    close_trigger = "market_inactive"
                    price_source = current_price_source

        if close_price is None:
            state_changed = False
            if current_price is not None:
                state_changed = (
                    prev_last_mark is None
                    or abs(prev_last_mark - current_price) > 1e-9
                    or prev_high is None
                    or prev_low is None
                    or abs((prev_high or 0.0) - (highest_price or 0.0)) > 1e-9
                    or abs((prev_low or 0.0) - (lowest_price or 0.0)) > 1e-9
                    or prev_mark_source != str(current_price_source or "")
                )
            if not dry_run and state_changed:
                payload["position_state"] = next_state
                row.payload_json = payload
                row.updated_at = now
                state_updates += 1
            held += 1
            continue

        quantity = notional / entry_price
        proceeds = quantity * close_price
        pnl = proceeds - notional
        next_status = "resolved_win" if pnl >= 0 else "resolved_loss"

        # NOTE: No simulation ledger interaction for live positions.
        # Live positions settle against real exchange state.

        total_realized_pnl += pnl
        by_status[next_status] = int(by_status.get(next_status, 0)) + 1

        detail = {
            "order_id": row.id,
            "market_id": row.market_id,
            "direction": row.direction,
            "entry_price": entry_price,
            "close_price": close_price,
            "price_source": price_source,
            "close_trigger": close_trigger,
            "market_tradable": market_tradable,
            "notional_usd": notional,
            "quantity": quantity,
            "realized_pnl": pnl,
            "next_status": next_status,
            "age_minutes": age_minutes,
            "min_hold_minutes": min_hold_minutes,
            "trailing_stop_trigger_price": trailing_trigger_price,
            "highest_price_seen": _state_price_floor(highest_price),
            "lowest_price_seen": _state_price_floor(lowest_price),
        }
        details.append(detail)
        would_close += 1

        if dry_run:
            continue

        row.status = next_status
        row.actual_profit = pnl
        row.updated_at = now
        payload["position_state"] = next_state
        payload["position_close"] = {
            "close_price": close_price,
            "price_source": price_source,
            "close_trigger": close_trigger,
            "realized_pnl": pnl,
            "market_tradable": market_tradable,
            "age_minutes": age_minutes,
            "closed_at": now.isoformat() + "Z",
            "reason": reason,
        }
        row.payload_json = payload
        if reason:
            if row.reason:
                row.reason = f"{row.reason} | {reason}:{close_trigger}"
            else:
                row.reason = f"{reason}:{close_trigger}"
        closed += 1

    if not dry_run and (closed > 0 or state_updates > 0):
        await session.commit()

    return {
        "trader_id": trader_id,
        "mode": "live",
        "dry_run": bool(dry_run),
        "matched": len(candidates),
        "would_close": would_close,
        "closed": closed,
        "held": held,
        "skipped": skipped,
        "state_updates": state_updates,
        "total_realized_pnl": total_realized_pnl,
        "by_status": by_status,
        "skipped_reasons": skipped_reasons,
        "details": details,
    }
