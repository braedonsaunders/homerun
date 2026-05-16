from __future__ import annotations

import asyncio
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import ExecutionSessionEvent, ExecutionSessionLeg, ExecutionSessionOrder, TraderOrder, release_conn
from services.event_bus import event_bus
from services.live_execution_adapter import execute_live_order
from services.live_execution_service import live_execution_service
from services.signal_bus import set_trade_signal_status
from services.strategy_sdk import StrategySDK
from services.strategy_loader import strategy_loader
from services.trader_order_verification import apply_trader_order_verification, derive_trader_order_verification
from services.trader_orchestrator.execution_policies import (
    allocate_leg_notionals,
    execution_waves,
    normalize_execution_constraints,
    normalize_execution_policy,
    reprice_limit_price,
    requires_pair_lock,
    supports_reprice,
)
from services.trader_orchestrator.order_manager import (
    LegSubmitResult,
    cancel_live_provider_order,
    submit_execution_wave,
)
from services.trader_orchestrator_state import (
    _extract_copy_source_wallet_from_payload,
    _extract_live_fill_metrics,
    _serialize_execution_event,
    _serialize_execution_leg,
    _serialize_execution_order,
    _serialize_execution_session,
    _serialize_order,
    _sync_order_runtime_payload,
    _apply_execution_session_rollups_from_rows,
    _new_id,
    build_execution_session_rows,
    build_trader_order_row,
    create_execution_session_event,
    get_execution_session_detail,
    get_execution_session_leg_rollups,
    list_active_execution_sessions,
    recover_missing_live_trader_orders,
    sync_trader_position_inventory,
    update_execution_leg,
    update_execution_session_status,
)
from services.trader_orchestrator.fast_idempotency import derive_clob_idempotency_key
from utils.converters import safe_float, safe_int
from utils.logger import get_logger
from utils.signal_helpers import normalize_position_side
from utils.utcnow import utcnow
import services.trader_hot_state as hot_state

logger = get_logger(__name__)


_MIN_BUNDLE_EXECUTION_SHARES = 5.0
_PRE_SUBMIT_ORDER_STATUS = "placing"
_STALE_PRE_SUBMIT_SESSION_TIMEOUT_SECONDS = 45.0
_LIVE_PRE_SUBMIT_AUTHORITY_RECOVERY_MAX_AGE_SECONDS = 900.0
_STALE_SESSION_PROVIDER_IO_SKIP_SECONDS = 3600.0
_PROVIDER_SNAPSHOT_TIMEOUT_SECONDS = 5.0
_PROVIDER_CANCEL_TIMEOUT_SECONDS = 5.0


def _iso_utc(value: datetime) -> str:
    dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso_utc(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _resolve_leg_market_question(signal_payload: dict[str, Any], leg: dict[str, Any], fallback_question: str) -> str:
    explicit_question = str(leg.get("market_question") or "").strip()
    if explicit_question:
        return explicit_question

    leg_market_id = str(leg.get("market_id") or "").strip()
    raw_markets = signal_payload.get("markets")
    if isinstance(raw_markets, list):
        normalized_leg_market_id = leg_market_id.lower()
        for raw_market in raw_markets:
            if not isinstance(raw_market, dict):
                continue
            candidate_ids = (
                raw_market.get("question"),
                raw_market.get("market_question"),
                raw_market.get("id"),
                raw_market.get("market_id"),
                raw_market.get("condition_id"),
                raw_market.get("conditionId"),
                raw_market.get("slug"),
                raw_market.get("market_slug"),
                raw_market.get("ticker"),
            )
            if normalized_leg_market_id and any(
                str(candidate or "").strip().lower() == normalized_leg_market_id for candidate in candidate_ids
            ):
                resolved_question = str(
                    raw_market.get("question") or raw_market.get("market_question") or leg_market_id
                ).strip()
                if resolved_question:
                    return resolved_question

    return leg_market_id or fallback_question


def _resolve_leg_direction(leg: dict[str, Any], fallback_direction: str) -> str:
    explicit_direction = str(leg.get("direction") or "").strip().lower()
    if explicit_direction:
        return explicit_direction

    side = str(leg.get("side") or "").strip().lower()
    outcome = str(leg.get("outcome") or "").strip().lower()
    if side in {"buy", "sell"} and outcome in {"yes", "no"}:
        return f"{side}_{outcome}"
    if side in {"buy", "sell"}:
        return side
    return str(fallback_direction or "").strip().lower()


def _strategy_instance_for_execution(strategy_key: str) -> Any | None:
    key = str(strategy_key or "").strip().lower()
    if not key:
        return None
    return strategy_loader.get_instance(key)


def _strategy_supports_entry_take_profit_exit(strategy_instance: Any) -> bool:
    if strategy_instance is None:
        return False
    declared = getattr(strategy_instance, "supports_entry_take_profit_exit", None)
    if isinstance(declared, bool):
        return declared
    if declared is not None:
        return bool(declared)
    defaults = getattr(strategy_instance, "default_config", None)
    if not isinstance(defaults, dict):
        return False
    return "preplace_take_profit_exit" in defaults or "live_preplace_take_profit_exit" in defaults


def _resolve_preplace_take_profit_exit(
    *,
    strategy_instance: Any,
    params: dict[str, Any],
) -> bool:
    if not _strategy_supports_entry_take_profit_exit(strategy_instance):
        return False
    return StrategySDK.should_preplace_take_profit_exit(dict(params or {}), default_enabled=False)


def _execution_profile_for_signal(
    *,
    signal: Any,
    strategy_key: str,
    legs_count: int,
) -> dict[str, Any]:
    source_key = str(getattr(signal, "source", "") or "").strip().lower()
    signal_strategy_type = str(getattr(signal, "strategy_type", "") or "").strip().lower()
    strategy_slug = str(strategy_key or "").strip().lower()

    payload = getattr(signal, "payload_json", None)
    payload = payload if isinstance(payload, dict) else {}
    strategy_context = payload.get("strategy_context")
    if not isinstance(strategy_context, dict):
        strategy_context = payload.get("strategy_context_json")
    context_source = (
        str(strategy_context.get("source_key") or "").strip().lower() if isinstance(strategy_context, dict) else ""
    )
    market_roster = payload.get("market_roster")
    market_roster = market_roster if isinstance(market_roster, dict) else {}
    is_guaranteed_bundle = (
        bool(payload.get("is_guaranteed"))
        and legs_count > 1
        and str(market_roster.get("scope") or "").strip().lower() == "event"
    )

    is_traders = (
        source_key == "traders"
        or context_source == "traders"
        or signal_strategy_type.startswith("traders")
        or strategy_slug.startswith("traders")
    )
    if is_traders:
        return {
            "policy": "REPRICE_LOOP" if legs_count <= 1 else "SEQUENTIAL_HEDGE",
            "price_policy": "taker_limit",
            "time_in_force": "IOC",
            "constraints": {
                "max_unhedged_notional_usd": 0.0,
                "hedge_timeout_seconds": 12,
                "session_timeout_seconds": 240,
                "max_reprice_attempts": 2,
                "pair_lock": legs_count > 1,
                "leg_fill_tolerance_ratio": 0.02,
            },
        }
    if is_guaranteed_bundle:
        return {
            "policy": "PAIR_LOCK",
            "price_policy": "taker_limit",
            "time_in_force": "IOC",
            "constraints": {
                "max_unhedged_notional_usd": 0.0,
                "hedge_timeout_seconds": 3,
                "session_timeout_seconds": 90,
                "max_reprice_attempts": 0,
                "pair_lock": True,
                "leg_fill_tolerance_ratio": 0.02,
            },
        }
    return {
        "policy": "PARALLEL_MAKER" if legs_count > 1 else "SINGLE_LEG",
        "price_policy": "maker_limit",
        "time_in_force": "GTC",
        "constraints": {
            "max_unhedged_notional_usd": 0.0,
            "hedge_timeout_seconds": 20,
            "session_timeout_seconds": 300,
            "max_reprice_attempts": 3,
            "pair_lock": legs_count > 1,
            "leg_fill_tolerance_ratio": 0.02,
        },
    }


def _signal_payload(signal: Any) -> dict[str, Any]:
    payload = getattr(signal, "payload_json", None)
    return payload if isinstance(payload, dict) else {}


def _requires_full_bundle_execution(signal: Any, legs: list[dict[str, Any]]) -> bool:
    if len(legs) < 2:
        return False
    payload = _signal_payload(signal)
    strategy_type = str(
        payload.get("strategy_type")
        or getattr(signal, "strategy_type", "")
        or ""
    ).strip().lower()
    if strategy_type == "prob_surface_arb":
        return True
    if not bool(payload.get("is_guaranteed")):
        return False
    selected_market_ids = _selected_market_ids(legs)
    if len(selected_market_ids) == 1:
        return True
    roster = payload.get("market_roster")
    if isinstance(roster, dict) and str(roster.get("scope") or "").strip().lower() == "event":
        return len(_required_roster_market_ids(signal)) > 1
    execution_plan = payload.get("execution_plan")
    execution_plan = execution_plan if isinstance(execution_plan, dict) else {}
    metadata = execution_plan.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    market_coverage = metadata.get("market_coverage")
    market_coverage = market_coverage if isinstance(market_coverage, dict) else {}
    return bool(market_coverage.get("requires_full_market_coverage"))


def _required_roster_market_ids(signal: Any) -> list[str]:
    payload = _signal_payload(signal)
    roster = payload.get("market_roster")
    if not isinstance(roster, dict):
        return []
    if str(roster.get("scope") or "").strip().lower() != "event":
        return []

    required_ids: list[str] = []
    for market in roster.get("markets") or []:
        if not isinstance(market, dict):
            continue
        market_id = str(market.get("id") or market.get("market_id") or "").strip()
        if market_id and market_id not in required_ids:
            required_ids.append(market_id)
    return required_ids


def _selected_market_ids(legs: list[dict[str, Any]]) -> list[str]:
    selected_ids: list[str] = []
    for leg in legs:
        market_id = str(leg.get("market_id") or "").strip()
        if market_id and market_id not in selected_ids:
            selected_ids.append(market_id)
    return selected_ids


def _normalize_plan_leg_sides(payload: dict[str, Any], legs: list[dict[str, Any]]) -> None:
    positions = payload.get("positions_to_take") if isinstance(payload.get("positions_to_take"), list) else []
    for index, leg in enumerate(legs):
        metadata = leg.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        position_index = safe_int(metadata.get("position_index"), index)
        position = positions[position_index] if 0 <= position_index < len(positions) and isinstance(positions[position_index], dict) else {}
        action = (
            metadata.get("raw_action")
            or position.get("action")
            or position.get("side")
            or leg.get("side")
        )
        leg["side"] = normalize_position_side(action, fallback=str(leg.get("side") or "buy"))


def _self_crossing_plan_violation(legs: list[dict[str, Any]]) -> dict[str, Any] | None:
    by_instrument: dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]] = {}
    for leg in legs:
        market_id = str(leg.get("market_id") or "").strip()
        token_id = str(leg.get("token_id") or "").strip()
        outcome = str(leg.get("outcome") or "").strip().lower()
        if not market_id or not (token_id or outcome):
            continue
        side = normalize_position_side(leg.get("side"), fallback="buy")
        limit_price = safe_float(leg.get("limit_price"), None)
        if limit_price is None or limit_price <= 0.0:
            continue
        bucket = by_instrument.setdefault((market_id, token_id, outcome), {"buy": [], "sell": []})
        bucket[side].append(
            {
                "leg_id": str(leg.get("leg_id") or ""),
                "limit_price": float(limit_price),
            }
        )

    for (market_id, token_id, outcome), sides in by_instrument.items():
        buys = sides["buy"]
        sells = sides["sell"]
        if not buys or not sells:
            continue
        max_bid = max(float(item["limit_price"]) for item in buys)
        min_ask = min(float(item["limit_price"]) for item in sells)
        if max_bid >= min_ask:
            return {
                "reason": "self_crossing_quote",
                "market_id": market_id,
                "token_id": token_id or None,
                "outcome": outcome or None,
                "max_buy_limit_price": max_bid,
                "min_sell_limit_price": min_ask,
                "buy_legs": buys,
                "sell_legs": sells,
            }
    return None


def _entry_fill_metrics(result: Any, *, normalized_status: str) -> tuple[float, float, float | None]:
    payload = dict(getattr(result, "payload", None) or {})
    filled_shares = safe_float(payload.get("filled_size"), None)
    avg_fill_price = safe_float(payload.get("average_fill_price"), None)
    filled_notional = safe_float(payload.get("filled_notional_usd"), None)

    if filled_shares is None and normalized_status == "executed":
        filled_shares = safe_float(getattr(result, "shares", None), None)
    if avg_fill_price is None:
        avg_fill_price = safe_float(getattr(result, "effective_price", None), None)
    if filled_notional is None:
        if filled_shares is not None and filled_shares > 0.0 and avg_fill_price is not None and avg_fill_price > 0.0:
            filled_notional = filled_shares * avg_fill_price
        elif normalized_status == "executed":
            filled_notional = safe_float(getattr(result, "notional_usd", None), None)

    return (
        max(0.0, float(filled_notional or 0.0)),
        max(0.0, float(filled_shares or 0.0)),
        avg_fill_price if avg_fill_price is not None and avg_fill_price > 0.0 else None,
    )


def _bundle_preflight_violation(
    *,
    signal: Any,
    legs: list[dict[str, Any]],
    strategy_params: dict[str, Any],
    mode: str,
    size_usd: float,
) -> dict[str, Any] | None:
    if str(mode or "").strip().lower() != "live":
        return None
    if not _requires_full_bundle_execution(signal, legs):
        return None

    required_market_ids = _required_roster_market_ids(signal)
    if required_market_ids:
        selected_market_ids = _selected_market_ids(legs)
        if set(selected_market_ids) != set(required_market_ids):
            return {
                "reason": "incomplete_market_roster",
                "required_market_ids": required_market_ids,
                "selected_market_ids": selected_market_ids,
            }

    min_order_size_usd = StrategySDK.resolve_min_order_size_usd(strategy_params, mode=mode, fallback=1.0)
    bundle_scale_required = 1.0
    leg_shortfalls: list[dict[str, Any]] = []
    for leg in legs:
        leg_id = str(leg.get("leg_id") or "").strip() or "leg"
        market_id = str(leg.get("market_id") or "").strip()
        requested_notional = safe_float(leg.get("requested_notional_usd"), 0.0)
        requested_shares = safe_float(leg.get("requested_shares"), 0.0)
        limit_price = safe_float(leg.get("limit_price"), None)

        if limit_price is None or limit_price <= 0.0:
            return {
                "reason": "missing_limit_price",
                "leg_id": leg_id,
                "market_id": market_id,
            }
        if requested_notional <= 0.0 or requested_shares <= 0.0:
            return {
                "reason": "invalid_leg_size",
                "leg_id": leg_id,
                "market_id": market_id,
                "requested_notional_usd": requested_notional,
                "requested_shares": requested_shares,
            }

        leg_scale_required = 1.0
        if requested_shares < _MIN_BUNDLE_EXECUTION_SHARES:
            leg_scale_required = max(leg_scale_required, _MIN_BUNDLE_EXECUTION_SHARES / requested_shares)
        if requested_notional < min_order_size_usd:
            leg_scale_required = max(leg_scale_required, min_order_size_usd / requested_notional)
        if leg_scale_required > 1.0 + 1e-9:
            bundle_scale_required = max(bundle_scale_required, leg_scale_required)
            leg_shortfalls.append(
                {
                    "leg_id": leg_id,
                    "market_id": market_id,
                    "requested_notional_usd": requested_notional,
                    "requested_shares": requested_shares,
                    "minimum_notional_usd": float(min_order_size_usd),
                    "minimum_shares": float(_MIN_BUNDLE_EXECUTION_SHARES),
                    "required_scale": float(leg_scale_required),
                }
            )

    if leg_shortfalls:
        return {
            "reason": "bundle_below_minimum_executable_size",
            "requested_total_notional_usd": float(max(0.0, size_usd)),
            "minimum_total_notional_usd": float(max(0.0, size_usd) * bundle_scale_required),
            "minimum_order_size_usd": float(min_order_size_usd),
            "leg_shortfalls": leg_shortfalls,
        }
    return None


def _enforce_live_full_bundle_plan(
    *,
    signal: Any,
    plan: dict[str, Any],
    legs: list[dict[str, Any]],
    constraints: dict[str, Any],
) -> None:
    if not _requires_full_bundle_execution(signal, legs):
        return

    plan["policy"] = "PAIR_LOCK"
    plan["time_in_force"] = "IOC"
    constraints["max_unhedged_notional_usd"] = 0.0
    constraints["pair_lock"] = True
    constraints["max_reprice_attempts"] = 0
    constraints["hedge_timeout_seconds"] = min(max(1, safe_int(constraints.get("hedge_timeout_seconds"), 20)), 10)
    plan["constraints"] = dict(constraints)

    metadata = dict(plan.get("metadata") or {})
    metadata["full_bundle_execution_required"] = True
    metadata["full_bundle_execution_mode"] = "live_ioc_complete_or_flatten"
    plan["metadata"] = metadata

    for leg in legs:
        leg["time_in_force"] = "IOC"
        leg["price_policy"] = "taker_limit"
        leg["post_only"] = False


def _is_live_position_cap_skip(*, mode: str, status: str, error_message: str | None) -> bool:
    if str(mode or "").strip().lower() != "live":
        return False
    if str(status or "").strip().lower() != "failed":
        return False
    error_text = str(error_message or "").strip().lower()
    if not error_text:
        return False
    return "maximum open positions" in error_text and "reached" in error_text


_BUNDLE_FLATTEN_PRICE_BPS = 500.0


def _result_payload(result: Any) -> dict[str, Any]:
    payload = getattr(result, "payload", None)
    return payload if isinstance(payload, dict) else {}


def _result_fill_price(result: Any) -> float | None:
    payload = _result_payload(result)
    average_fill_price = safe_float(payload.get("average_fill_price"), None)
    if average_fill_price is not None and average_fill_price > 0.0:
        return float(average_fill_price)
    effective_price = safe_float(getattr(result, "effective_price", None), None)
    if effective_price is not None and effective_price > 0.0:
        return float(effective_price)
    return None


def _result_filled_shares(result: Any) -> float:
    payload = _result_payload(result)
    filled_size = max(0.0, safe_float(payload.get("filled_size"), 0.0) or 0.0)
    if filled_size > 0.0:
        return float(filled_size)
    status_key = str(getattr(result, "status", "") or "").strip().lower()
    if status_key == "executed":
        return max(0.0, safe_float(getattr(result, "shares", None), 0.0) or 0.0)
    return 0.0


def _result_filled_notional(result: Any) -> float:
    payload = _result_payload(result)
    filled_notional = max(0.0, safe_float(payload.get("filled_notional_usd"), 0.0) or 0.0)
    if filled_notional > 0.0:
        return float(filled_notional)
    filled_shares = _result_filled_shares(result)
    fill_price = _result_fill_price(result)
    if filled_shares > 0.0 and fill_price is not None and fill_price > 0.0:
        return float(filled_shares * fill_price)
    status_key = str(getattr(result, "status", "") or "").strip().lower()
    if status_key == "executed":
        return max(0.0, safe_float(getattr(result, "notional_usd", None), 0.0) or 0.0)
    return 0.0


def _remaining_requested_shares(*, requested_shares: Any, filled_shares: Any, tolerance_ratio: Any) -> float:
    requested = max(0.0, safe_float(requested_shares, 0.0) or 0.0)
    if requested <= 0.0:
        return 0.0
    filled = max(0.0, safe_float(filled_shares, 0.0) or 0.0)
    remaining = max(0.0, requested - filled)
    tolerance = max(0.0, min(1.0, safe_float(tolerance_ratio, 0.0) or 0.0))
    if remaining <= requested * tolerance:
        return 0.0
    return float(remaining)


def _event_bundle_coverage(signal_payload: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any] | None:
    roster = signal_payload.get("market_roster")
    if not isinstance(roster, dict):
        return None
    if str(roster.get("scope") or "").strip().lower() != "event":
        return None
    if not bool(signal_payload.get("is_guaranteed")):
        return None

    roster_market_ids: list[str] = []
    seen_roster_market_ids: set[str] = set()
    for raw_market in roster.get("markets") or []:
        if not isinstance(raw_market, dict):
            continue
        market_id = str(raw_market.get("id") or raw_market.get("market_id") or "").strip()
        if not market_id or market_id in seen_roster_market_ids:
            continue
        seen_roster_market_ids.add(market_id)
        roster_market_ids.append(market_id)
    if len(roster_market_ids) <= 1:
        return None

    plan_market_ids: list[str] = []
    seen_plan_market_ids: set[str] = set()
    for raw_leg in plan.get("legs") or []:
        if not isinstance(raw_leg, dict):
            continue
        market_id = str(raw_leg.get("market_id") or "").strip()
        if not market_id or market_id in seen_plan_market_ids:
            continue
        seen_plan_market_ids.add(market_id)
        plan_market_ids.append(market_id)
    if len(plan_market_ids) <= 1:
        return None

    roster_market_id_set = set(roster_market_ids)
    event_internal_bundle = all(market_id in roster_market_id_set for market_id in plan_market_ids)
    if not event_internal_bundle:
        return None

    missing_market_ids = [market_id for market_id in roster_market_ids if market_id not in seen_plan_market_ids]
    return {
        "event_id": str(roster.get("event_id") or "").strip() or None,
        "event_slug": str(roster.get("event_slug") or "").strip() or None,
        "roster_hash": str(roster.get("roster_hash") or "").strip() or None,
        "roster_market_ids": roster_market_ids,
        "plan_market_ids": plan_market_ids,
        "missing_market_ids": missing_market_ids,
    }


def _recovery_price(*, side: str, base_price: float | None, price_bps: float) -> float | None:
    if base_price is None or base_price <= 0.0:
        return None
    bps = max(0.0, float(price_bps))
    side_key = str(side or "").strip().upper()
    if side_key == "SELL":
        adjusted = float(base_price) * (1.0 - (bps / 10_000.0))
    else:
        adjusted = float(base_price) * (1.0 + (bps / 10_000.0))
    return max(0.01, min(0.99, adjusted))


@dataclass
class SessionExecutionResult:
    session_id: str
    status: str
    effective_price: float | None
    error_message: str | None
    orders_written: int
    payload: dict[str, Any]
    created_orders: list[dict[str, Any]] = field(default_factory=list)


class ExecutionSessionEngine:
    def __init__(self, db: AsyncSession):
        self.db = db

    def _build_plan(
        self,
        signal: Any,
        *,
        strategy_key: str,
        size_usd: float,
        risk_limits: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        payload = getattr(signal, "payload_json", None)
        payload = payload if isinstance(payload, dict) else {}
        execution_plan = payload.get("execution_plan")
        if not isinstance(execution_plan, dict):
            execution_plan = {}
        legs = [leg for leg in (execution_plan.get("legs") or []) if isinstance(leg, dict)]
        profile = _execution_profile_for_signal(
            signal=signal,
            strategy_key=strategy_key,
            legs_count=max(1, len(legs)),
        )

        if not legs:
            signal_payload = payload if isinstance(payload, dict) else {}
            fallback_market_id = str(getattr(signal, "market_id", "") or "")
            fallback_market_question = str(getattr(signal, "market_question", "") or "")
            legs = [
                {
                    "leg_id": "leg_1",
                    "market_id": fallback_market_id,
                    "market_question": fallback_market_question,
                    "token_id": str(
                        signal_payload.get("selected_token_id") or signal_payload.get("token_id") or ""
                    ).strip()
                    or None,
                    "side": "buy",
                    "outcome": None,
                    "limit_price": safe_float(getattr(signal, "entry_price", None), None),
                    "price_policy": str(profile["price_policy"]),
                    "time_in_force": str(profile["time_in_force"]),
                    "notional_weight": 1.0,
                    "min_fill_ratio": 0.0,
                    "metadata": {},
                }
            ]
            profile = _execution_profile_for_signal(
                signal=signal,
                strategy_key=strategy_key,
                legs_count=1,
            )

        fallback_market_question = str(getattr(signal, "market_question", "") or "")
        for leg in legs:
            resolved_market_question = _resolve_leg_market_question(payload, leg, fallback_market_question)
            if resolved_market_question:
                leg["market_question"] = resolved_market_question
        _normalize_plan_leg_sides(payload, legs)

        policy = normalize_execution_policy(execution_plan.get("policy") or profile["policy"], legs_count=len(legs))
        raw_constraints = execution_plan.get("constraints")
        raw_constraints = raw_constraints if isinstance(raw_constraints, dict) else {}
        constraints = normalize_execution_constraints(raw_constraints)
        profile_constraints = dict(profile["constraints"])
        if safe_float(constraints.get("max_unhedged_notional_usd"), 0.0) <= 0:
            constraints["max_unhedged_notional_usd"] = max(
                0.0,
                safe_float(
                    risk_limits.get("max_unhedged_notional_usd"),
                    safe_float(profile_constraints.get("max_unhedged_notional_usd"), 0.0),
                ),
            )
        if "pair_lock" not in raw_constraints:
            constraints["pair_lock"] = bool(
                risk_limits.get("pair_lock", profile_constraints.get("pair_lock", constraints.get("pair_lock", True)))
            )
        if "hedge_timeout_seconds" not in raw_constraints:
            constraints["hedge_timeout_seconds"] = max(
                1,
                safe_int(
                    risk_limits.get("hedge_timeout_seconds"),
                    safe_int(profile_constraints.get("hedge_timeout_seconds"), constraints["hedge_timeout_seconds"]),
                ),
            )
        if "session_timeout_seconds" not in raw_constraints:
            constraints["session_timeout_seconds"] = max(
                1,
                safe_int(
                    risk_limits.get("session_timeout_seconds"),
                    safe_int(
                        profile_constraints.get("session_timeout_seconds"), constraints["session_timeout_seconds"]
                    ),
                ),
            )
        if "max_reprice_attempts" not in raw_constraints:
            constraints["max_reprice_attempts"] = max(
                0,
                safe_int(
                    risk_limits.get("max_reprice_attempts"),
                    safe_int(profile_constraints.get("max_reprice_attempts"), constraints["max_reprice_attempts"]),
                ),
            )
        if "leg_fill_tolerance_ratio" not in raw_constraints:
            constraints["leg_fill_tolerance_ratio"] = max(
                0.0,
                min(
                    1.0,
                    safe_float(
                        risk_limits.get("leg_fill_tolerance_ratio"),
                        safe_float(
                            profile_constraints.get("leg_fill_tolerance_ratio"), constraints["leg_fill_tolerance_ratio"]
                        ),
                    ),
                ),
            )
        notionals = allocate_leg_notionals(size_usd, legs)
        for index, leg in enumerate(legs):
            leg["requested_notional_usd"] = float(max(0.0, notionals[index]))
            limit_price = safe_float(leg.get("limit_price"), None)
            if limit_price is not None and limit_price > 0:
                leg["requested_shares"] = float(leg["requested_notional_usd"] / max(0.0001, limit_price))
            else:
                leg["requested_shares"] = None

        plan = {
            "plan_id": str(execution_plan.get("plan_id") or ""),
            "policy": policy,
            "time_in_force": str(execution_plan.get("time_in_force") or profile["time_in_force"]),
            "legs": legs,
            "constraints": constraints,
            "metadata": dict(execution_plan.get("metadata") or {}),
        }
        return plan, legs, constraints

    def _effective_price_from_leg_results(self, leg_results: list[dict[str, Any]]) -> float | None:
        numerator = 0.0
        denominator = 0.0
        for result in leg_results:
            price = safe_float(result.get("effective_price"), None)
            notional = safe_float(result.get("notional_usd"), 0.0)
            if price is None or price <= 0 or notional <= 0:
                continue
            numerator += price * notional
            denominator += notional
        if denominator <= 0:
            return None
        return numerator / denominator

    async def _publish_hot_signal_status(
        self,
        *,
        signal_id: str,
        status: str,
        effective_price: float | None = None,
    ) -> None:
        normalized_signal_id = str(signal_id or "").strip()
        if not normalized_signal_id:
            return
        from services.intent_runtime import get_intent_runtime

        await get_intent_runtime().update_signal_status(
            signal_id=normalized_signal_id,
            status=str(status or "").strip().lower(),
            effective_price=effective_price,
        )

    async def execute_signal(
        self,
        *,
        trader_id: str,
        signal: Any,
        decision_id: str,
        strategy_key: str,
        strategy_version: int | None,
        strategy_params: dict[str, Any],
        risk_limits: dict[str, Any],
        mode: str,
        size_usd: float,
        reason: str | None,
        explicit_strategy_params: dict[str, Any] | None = None,
    ) -> SessionExecutionResult:
        _execution_started_at = _time.monotonic()
        _execution_timing_ms: dict[str, float] = {}

        def _record_execution_timing(stage: str, started_at: float) -> None:
            elapsed_ms = max(0.0, (_time.monotonic() - started_at) * 1000.0)
            previous_ms = safe_float(_execution_timing_ms.get(stage), 0.0) or 0.0
            _execution_timing_ms[stage] = round(previous_ms + elapsed_ms, 3)

        def _payload_with_execution_timing(payload: dict[str, Any]) -> dict[str, Any]:
            enriched = dict(payload)
            timing = dict(_execution_timing_ms)
            timing["total_ms"] = round(max(0.0, (_time.monotonic() - _execution_started_at) * 1000.0), 3)
            enriched["execution_timing_ms"] = timing
            return enriched

        _session_build_started_at = _time.monotonic()
        plan, legs, constraints = self._build_plan(
            signal,
            strategy_key=strategy_key,
            size_usd=size_usd,
            risk_limits=risk_limits,
        )
        signal_payload = getattr(signal, "payload_json", None)
        signal_payload = signal_payload if isinstance(signal_payload, dict) else {}
        _enforce_live_full_bundle_plan(signal=signal, plan=plan, legs=legs, constraints=constraints)
        if isinstance(getattr(signal, "payload_json", None), dict):
            signal.payload_json["execution_plan"] = plan
        session_timeout_seconds = safe_int(constraints.get("session_timeout_seconds"), 300)
        expires_at = utcnow() + timedelta(seconds=max(1, session_timeout_seconds))
        session_row, built_leg_rows = build_execution_session_rows(
            trader_id=trader_id,
            signal=signal,
            decision_id=decision_id,
            strategy_key=strategy_key,
            strategy_version=strategy_version,
            mode=mode,
            policy=plan["policy"],
            plan_id=plan["plan_id"] or None,
            legs=legs,
            requested_notional_usd=size_usd,
            max_unhedged_notional_usd=safe_float(constraints.get("max_unhedged_notional_usd"), 0.0),
            expires_at=expires_at,
            payload={
                "execution_plan": plan,
                "strategy_key": strategy_key,
                "strategy_version": int(strategy_version) if strategy_version is not None else None,
                "reason": reason,
            },
            trace_id=str(getattr(signal, "trace_id", "") or "") or None,
        )
        leg_rows = {str(row.leg_id): row for row in built_leg_rows}
        execution_orders: list[ExecutionSessionOrder] = []
        execution_events: list[ExecutionSessionEvent] = []
        trader_orders: list[TraderOrder] = []
        leg_execution_records: list[dict[str, Any]] = []
        # Collected across every submit wave (initial + retry + rescue
        # waves) so the orchestrator's slow log can name which sub-stage
        # of place_order owned the time.  Each entry is the breakdown
        # dict stashed by ``live_execution_service.place_order`` and
        # propagated via the live_execution_adapter's leg payload.
        _submit_wave_breakdowns: list[dict[str, Any]] = []
        order_write_inputs: list[dict[str, Any]] = []
        recovery_order_write_inputs: list[dict[str, Any]] = []
        created_order_records: list[dict[str, Any]] = []
        orders_written = 0
        failed_legs = 0
        skipped_legs = 0
        open_legs = 0
        completed_legs = 0
        skip_reasons: list[str] = []
        bundle_recovery_outcome: dict[str, Any] | None = None
        entry_submit_placeholders: dict[str, tuple[TraderOrder, ExecutionSessionOrder]] = {}
        _record_execution_timing("session_build", _session_build_started_at)

        def _append_event(
            *,
            event_type: str,
            severity: str = "info",
            leg_id: str | None = None,
            message: str | None = None,
            payload: dict[str, Any] | None = None,
        ) -> None:
            execution_events.append(
                ExecutionSessionEvent(
                    id=_new_id(),
                    session_id=session_row.id,
                    leg_id=leg_id,
                    event_type=str(event_type),
                    severity=str(severity or "info"),
                    message=message,
                    payload_json=payload or {},
                    created_at=utcnow(),
                )
            )

        def _update_session_status(
            *,
            status: str,
            error_message: str | None = None,
            payload_patch: dict[str, Any] | None = None,
        ) -> None:
            now = utcnow()
            session_row.status = str(status)
            session_row.error_message = error_message
            if payload_patch:
                merged = dict(session_row.payload_json or {})
                merged.update(payload_patch)
                session_row.payload_json = merged
            if str(status).strip().lower() in {"completed", "failed", "cancelled", "expired", "skipped"}:
                session_row.completed_at = now
            session_row.updated_at = now
            _apply_execution_session_rollups_from_rows(session_row, list(leg_rows.values()))

        def _update_leg_row(
            *,
            leg_row: ExecutionSessionLeg,
            status: str | None = None,
            filled_notional_usd: float | None = None,
            filled_shares: float | None = None,
            avg_fill_price: float | None = None,
            last_error: str | None = None,
            metadata_patch: dict[str, Any] | None = None,
        ) -> None:
            if status is not None:
                leg_row.status = str(status)
            if filled_notional_usd is not None:
                leg_row.filled_notional_usd = float(max(0.0, filled_notional_usd))
            if filled_shares is not None:
                leg_row.filled_shares = float(max(0.0, filled_shares))
            if avg_fill_price is not None:
                leg_row.avg_fill_price = float(avg_fill_price)
            if last_error is not None:
                leg_row.last_error = last_error
            if metadata_patch:
                merged_metadata = dict(leg_row.metadata_json or {})
                merged_metadata.update(metadata_patch)
                leg_row.metadata_json = merged_metadata
            leg_row.updated_at = utcnow()
            _apply_execution_session_rollups_from_rows(session_row, list(leg_rows.values()))

        def _build_entry_order_seed(
            *,
            leg_id: str,
            leg_row_id: str,
            leg_payload: dict[str, Any],
            result_payload: dict[str, Any] | None = None,
            fill_price: float | None = None,
        ) -> dict[str, Any]:
            order_payload = dict(result_payload or {})
            order_payload["execution_session"] = {
                "session_id": session_row.id,
                "leg_id": leg_row_id,
                "leg_ref": leg_id,
                "policy": plan["policy"],
            }
            order_payload["strategy_type"] = str(getattr(signal, "strategy_type", "") or "").strip().lower()
            order_payload["strategy_context"] = (
                getattr(signal, "strategy_context_json", None) or getattr(signal, "strategy_context", None) or {}
            )

            runtime_params = dict(strategy_params or {})
            explicit_params = dict(explicit_strategy_params or {})
            order_payload["strategy_params"] = dict(runtime_params)
            exit_config: dict[str, Any] = {}
            for param_key, param_value in explicit_params.items():
                key_text = str(param_key or "").strip()
                if not key_text:
                    continue
                exit_config[key_text] = param_value
                if key_text.startswith("live_") and len(key_text) > 5:
                    canonical_key = key_text[5:]
                    if canonical_key and canonical_key not in exit_config:
                        exit_config[canonical_key] = param_value
            exit_key_aliases = {
                "rapid_take_profit_pct": ("live_rapid_take_profit_pct", "rapid_take_profit_pct"),
                "rapid_exit_window_minutes": ("live_rapid_exit_window_minutes", "rapid_exit_window_minutes"),
                "rapid_exit_min_increase_pct": (
                    "live_rapid_exit_min_increase_pct",
                    "rapid_exit_min_increase_pct",
                ),
                "rapid_exit_breakeven_buffer_pct": (
                    "live_rapid_exit_breakeven_buffer_pct",
                    "rapid_exit_breakeven_buffer_pct",
                ),
                "take_profit_pct": ("live_take_profit_pct", "take_profit_pct"),
                "stop_loss_pct": ("live_stop_loss_pct", "stop_loss_pct"),
                "stop_loss_policy": ("live_stop_loss_policy", "stop_loss_policy", "stop_loss_mode"),
                "stop_loss_activation_seconds": (
                    "live_stop_loss_activation_seconds",
                    "stop_loss_activation_seconds",
                    "live_stop_loss_near_close_seconds",
                    "stop_loss_near_close_seconds",
                ),
                "trailing_stop_pct": ("live_trailing_stop_pct", "trailing_stop_pct"),
                "trailing_stop_activation_profit_pct": (
                    "live_trailing_stop_activation_profit_pct",
                    "trailing_stop_activation_profit_pct",
                ),
                "max_hold_minutes": ("live_max_hold_minutes", "max_hold_minutes"),
                "min_hold_minutes": ("live_min_hold_minutes", "min_hold_minutes"),
                "resolve_only": ("live_resolve_only", "resolve_only"),
                "close_on_inactive_market": ("live_close_on_inactive_market", "close_on_inactive_market"),
            }
            for base_key in (
                "rapid_take_profit_pct",
                "rapid_exit_window_minutes",
                "take_profit_pct",
                "stop_loss_pct",
                "stop_loss_policy",
                "stop_loss_activation_seconds",
                "trailing_stop_pct",
                "trailing_stop_activation_profit_pct",
                "max_hold_minutes",
                "min_hold_minutes",
            ):
                for suffix in ("5m", "15m", "1h", "4h"):
                    scoped_key = f"{base_key}_{suffix}"
                    exit_key_aliases[scoped_key] = (f"live_{scoped_key}", scoped_key)
            for target_key, aliases in exit_key_aliases.items():
                selected_value = None
                for alias_key in aliases:
                    if alias_key in explicit_params:
                        selected_value = explicit_params.get(alias_key)
                        break
                if selected_value is not None:
                    exit_config[target_key] = selected_value

            resolved_signal_payload = signal_payload if isinstance(signal_payload, dict) else {}
            leg_market_id = str(leg_payload.get("market_id") or getattr(signal, "market_id", "") or "").strip()
            leg_market_question = _resolve_leg_market_question(
                resolved_signal_payload,
                leg_payload,
                str(getattr(signal, "market_question", "") or "").strip(),
            )
            leg_direction = _resolve_leg_direction(leg_payload, str(getattr(signal, "direction", "") or ""))
            leg_entry_price = safe_float(leg_payload.get("limit_price"), safe_float(fill_price, None))
            if leg_market_id:
                order_payload["market_id"] = leg_market_id
            if leg_market_question:
                order_payload["market_question"] = leg_market_question
            if leg_direction:
                order_payload["direction"] = leg_direction
            if leg_entry_price is not None and leg_entry_price > 0.0:
                order_payload["entry_price"] = float(leg_entry_price)

            if isinstance(signal_payload, dict):
                live_market_payload = signal_payload.get("live_market")
                if isinstance(live_market_payload, dict):
                    signal_selected_token_id = str(live_market_payload.get("selected_token_id") or "").strip()
                    signal_market_key = str(
                        live_market_payload.get("market_id") or live_market_payload.get("condition_id") or ""
                    ).strip().lower()
                    leg_token_id = str(leg_payload.get("token_id") or "").strip()
                    leg_market_key = leg_market_id.strip().lower() if leg_market_id else ""
                    if (
                        signal_selected_token_id
                        and leg_token_id
                        and signal_selected_token_id == leg_token_id
                    ) or (
                        signal_market_key
                        and leg_market_key
                        and signal_market_key == leg_market_key
                    ):
                        order_payload["live_market"] = dict(live_market_payload)

            live_market_payload = order_payload.get("live_market")
            live_market = dict(live_market_payload) if isinstance(live_market_payload, dict) else {}
            if leg_market_id:
                live_market["market_id"] = leg_market_id
            if leg_market_question:
                live_market["market_question"] = leg_market_question
            selected_token_id = str(order_payload.get("token_id") or leg_payload.get("token_id") or "").strip()
            if selected_token_id:
                order_payload["token_id"] = selected_token_id
                live_market["selected_token_id"] = selected_token_id
            selected_outcome = str(leg_payload.get("outcome") or "").strip().lower()
            if selected_outcome:
                live_market["selected_outcome"] = selected_outcome
            live_selected_price = safe_float(fill_price, leg_entry_price)
            if live_selected_price is not None and live_selected_price > 0.0:
                live_market["live_selected_price"] = float(live_selected_price)
            if live_market:
                order_payload["live_market"] = live_market
            order_payload["strategy_exit_config"] = exit_config
            return {
                "order_payload": order_payload,
                "exit_config": exit_config,
                "market_id": leg_market_id,
                "market_question": leg_market_question,
                "direction": leg_direction,
                "entry_price": leg_entry_price,
            }

        async def _commit_pre_submit_projection() -> None:
            _started_at = _time.monotonic()
            try:
                self.db.add(session_row)
                for leg_row in leg_rows.values():
                    self.db.add(leg_row)
                for trader_order in trader_orders:
                    self.db.add(trader_order)
                await self.db.flush()
                for execution_order in execution_orders:
                    self.db.add(execution_order)
                if execution_orders:
                    await self.db.flush()
                for execution_event in execution_events:
                    self.db.add(execution_event)
                if execution_events:
                    await self.db.flush()
                await self.db.commit()
            finally:
                _record_execution_timing("pre_submit_projection", _started_at)

        def _create_entry_submit_placeholder(
            *,
            leg_id: str,
            leg_payload: dict[str, Any],
        ) -> tuple[TraderOrder, ExecutionSessionOrder] | None:
            leg_row = leg_rows.get(leg_id)
            if leg_row is None:
                return None
            now = utcnow()
            order_seed = _build_entry_order_seed(
                leg_id=leg_id,
                leg_row_id=leg_row.id,
                leg_payload=leg_payload,
                result_payload={},
                fill_price=None,
            )
            requested_notional = safe_float(leg_payload.get("requested_notional_usd"), None)
            requested_shares = safe_float(leg_payload.get("requested_shares"), None)
            submission_intent = {
                "state": "pre_submit_persisted",
                "persisted_at": _iso_utc(now),
                "requested_notional_usd": float(requested_notional) if requested_notional is not None and requested_notional > 0.0 else None,
                "requested_shares": float(requested_shares) if requested_shares is not None and requested_shares > 0.0 else None,
                "time_in_force": str(leg_payload.get("time_in_force") or "").strip().upper() or None,
                "post_only": bool(leg_payload.get("post_only", False)),
                "price_policy": str(leg_payload.get("price_policy") or "").strip().lower() or None,
            }
            placeholder_payload = dict(order_seed["order_payload"])
            placeholder_payload["submission_intent"] = submission_intent
            # Persist the venue idempotency key into the pre-submit row so
            # ``reconcile`` can locate this row by querying Polymarket for
            # orders whose ``metadata`` field matches this key.  Without
            # this we'd have to fuzzy-match by (token, side, price, size,
            # timestamp), which is unreliable on busy markets.
            clob_idempotency_key = leg_payload.get("clob_idempotency_key")
            if isinstance(clob_idempotency_key, str) and clob_idempotency_key.strip():
                placeholder_payload["clob_idempotency_key"] = clob_idempotency_key.strip()
            placeholder_payload = _sync_order_runtime_payload(
                payload=placeholder_payload,
                status=_PRE_SUBMIT_ORDER_STATUS,
                now=now,
                provider_snapshot_status="placing",
                mapped_status="placing",
            )
            trader_order = build_trader_order_row(
                trader_id=trader_id,
                signal=signal,
                decision_id=decision_id,
                strategy_key=strategy_key,
                strategy_version=strategy_version,
                mode=mode,
                status=_PRE_SUBMIT_ORDER_STATUS,
                notional_usd=None,
                effective_price=None,
                reason=reason,
                payload=placeholder_payload,
                error_message=None,
                market_id=order_seed["market_id"],
                market_question=order_seed["market_question"],
                direction=order_seed["direction"],
                entry_price=order_seed["entry_price"],
                created_at=now,
            )
            session_order = ExecutionSessionOrder(
                id=_new_id(),
                session_id=session_row.id,
                leg_id=leg_row.id,
                trader_order_id=trader_order.id,
                provider_order_id=None,
                provider_clob_order_id=None,
                action="submit",
                side=str(leg_payload.get("side") or "buy"),
                price=order_seed["entry_price"],
                size=float(requested_shares) if requested_shares is not None and requested_shares > 0.0 else None,
                notional_usd=float(requested_notional) if requested_notional is not None and requested_notional > 0.0 else None,
                status="placing",
                reason=reason,
                payload_json={"submission_intent": dict(submission_intent)},
                error_message=None,
                created_at=now,
                updated_at=now,
            )
            trader_orders.append(trader_order)
            execution_orders.append(session_order)
            return trader_order, session_order

        async def _persist_execution_projection(
            *,
            signal_status: str,
            effective_price: float | None,
        ) -> None:
            _projection_started_at = _time.monotonic()
            normalized_signal_id = str(getattr(signal, "id", "") or "").strip()
            normalized_signal_status = str(signal_status or "").strip().lower()
            # Flush trader_orders (and session/leg parents) before
            # execution_orders so the FK trader_order_id is satisfied at
            # flush time for any newly-created rows in shadow mode.
            self.db.add(session_row)
            for leg_row in leg_rows.values():
                self.db.add(leg_row)
            for trader_order in trader_orders:
                self.db.add(trader_order)
            await self.db.flush()
            for execution_order in execution_orders:
                self.db.add(execution_order)
            if execution_orders:
                await self.db.flush()
            for execution_event in execution_events:
                self.db.add(execution_event)
            if execution_events:
                await self.db.flush()
            if normalized_signal_id:
                await set_trade_signal_status(
                    self.db,
                    normalized_signal_id,
                    normalized_signal_status,
                    effective_price=effective_price,
                    commit=False,
                )
                await self._publish_hot_signal_status(
                    signal_id=normalized_signal_id,
                    status=normalized_signal_status,
                    effective_price=effective_price,
                )
            sync_targets = {
                (
                    str(order.trader_id or "").strip(),
                    str(order.mode or "").strip(),
                )
                for order in trader_orders
                if str(order.trader_id or "").strip() and str(order.mode or "").strip()
            }
            if mode != "live":
                for trader_id_key, mode_key in sorted(sync_targets):
                    await sync_trader_position_inventory(
                        self.db,
                        trader_id=trader_id_key,
                        mode=mode_key,
                        commit=False,
                    )
            # Persist the flushed session/leg/order/event rows to Postgres
            # before returning.  Without this commit, when ``submit_order``
            # uses a per-call ``AsyncSessionLocal`` (the shadow path —
            # see workers/trader_orchestrator_worker.py::submit_order),
            # ``async with`` exit calls ``session.close()`` which
            # implicitly rolls back every flush above.  The caller would
            # still see ``status="executed"`` and mutate ``hot_state``
            # via ``upsert_active_order``, leaving phantom open positions
            # against an empty database — exactly the race that was
            # blocking shadow Copy Trade with ``next=14 max=12``.  Commit
            # here is also safe when the worker reuses a shared session
            # (fast/live paths): a no-op if there's nothing pending.
            try:
                await self.db.commit()
                _record_execution_timing("projection_db_persist", _projection_started_at)
            except DBAPIError as commit_exc:
                try:
                    await self.db.rollback()
                except Exception:
                    pass
                logger.error(
                    "execution_session persist commit failed",
                    session_id=str(getattr(session_row, "id", "") or ""),
                    trader_id=str(getattr(session_row, "trader_id", "") or ""),
                    mode=str(getattr(session_row, "mode", "") or ""),
                    signal_status=normalized_signal_status,
                    trader_order_count=len(trader_orders),
                    error_class=type(commit_exc).__name__,
                    error=str(getattr(commit_exc, "orig", commit_exc))[:200],
                )
                raise
            if trader_orders:
                logger.info(
                    "execution_session persisted",
                    session_id=str(getattr(session_row, "id", "") or ""),
                    trader_id=str(getattr(session_row, "trader_id", "") or ""),
                    mode=str(getattr(session_row, "mode", "") or ""),
                    signal_status=normalized_signal_status,
                    trader_order_count=len(trader_orders),
                    execution_order_count=len(execution_orders),
                    leg_count=len(leg_rows),
                )
            # 2026-05-10: parallelize the post-commit event_bus
            # publishes.  Sequential awaits compound any Redis latency
            # (the same path that emitted ``trader_event Redis publish
            # batch failed ... Timeout writing to socket`` in the soak)
            # — for a typical session with 1 session_row + 1 leg + 1
            # trader_order + 1 execution_order + 2 events, that's 6
            # sequential awaits inside the submit_round_trip_ms window.
            # ``asyncio.gather(return_exceptions=True)`` fires them
            # concurrently and preserves the per-publish ``except
            # Exception: pass`` swallowing semantics — exceptions from
            # any publish surface as elements in the results list and
            # are then ignored, exactly like before.
            _publish_started_at = _time.monotonic()
            _publish_tasks = [event_bus.publish("execution_session", _serialize_execution_session(session_row))]
            _publish_tasks.extend(
                event_bus.publish("execution_leg", _serialize_execution_leg(leg_row))
                for leg_row in leg_rows.values()
            )
            _publish_tasks.extend(
                event_bus.publish("trader_order", _serialize_order(trader_order))
                for trader_order in trader_orders
            )
            _publish_tasks.extend(
                event_bus.publish("execution_order", _serialize_execution_order(execution_order))
                for execution_order in execution_orders
            )
            _publish_tasks.extend(
                event_bus.publish("execution_session_event", _serialize_execution_event(execution_event))
                for execution_event in execution_events
            )
            if _publish_tasks:
                await asyncio.gather(*_publish_tasks, return_exceptions=True)
            _record_execution_timing("projection_publish", _publish_started_at)
            _record_execution_timing("projection_persist", _projection_started_at)

        async def _persist_execution_projection_safely(
            *,
            signal_status: str,
            effective_price: float | None,
        ) -> None:
            persist = _persist_execution_projection(
                signal_status=signal_status,
                effective_price=effective_price,
            )
            if mode == "live" and entry_submit_placeholders:
                await asyncio.shield(persist)
                return
            await persist

        async def _finalize_cancelled_live_submit(
            *,
            error_message: str,
        ) -> None:
            if mode != "live" or not entry_submit_placeholders:
                return
            cancelled_at = utcnow()
            for leg_id, (trader_order, execution_order) in entry_submit_placeholders.items():
                leg_row = leg_rows.get(leg_id)
                if leg_row is not None:
                    _update_leg_row(
                        leg_row=leg_row,
                        status="failed",
                        last_error=error_message,
                        metadata_patch={"submit_cancelled": True},
                    )
                persisted_payload = dict(trader_order.payload_json or {})
                submission_intent = dict(persisted_payload.get("submission_intent") or {})
                submission_intent["state"] = "submit_cancelled"
                submission_intent["last_submit_result_at"] = _iso_utc(cancelled_at)
                submission_intent["submit_status"] = "cancelled"
                submission_intent["error_message"] = error_message
                persisted_payload["submission_intent"] = submission_intent
                persisted_payload = _sync_order_runtime_payload(
                    payload=persisted_payload,
                    status="failed",
                    now=cancelled_at,
                    provider_snapshot_status="cancelled",
                    mapped_status="failed",
                )
                trader_order.status = "failed"
                trader_order.notional_usd = 0.0
                trader_order.error_message = error_message
                trader_order.payload_json = persisted_payload
                trader_order.updated_at = cancelled_at
                verification_fields = derive_trader_order_verification(
                    mode=mode,
                    status="failed",
                    reason=reason,
                    payload=persisted_payload,
                )
                apply_trader_order_verification(
                    trader_order,
                    verification_status=str(verification_fields.get("verification_status") or ""),
                    verification_source=str(verification_fields.get("verification_source") or "").strip() or None,
                    verification_reason=str(verification_fields.get("verification_reason") or "").strip() or None,
                    provider_order_id=str(verification_fields.get("provider_order_id") or "").strip() or None,
                    provider_clob_order_id=str(verification_fields.get("provider_clob_order_id") or "").strip() or None,
                    execution_wallet_address=str(verification_fields.get("execution_wallet_address") or "").strip() or None,
                    verification_tx_hash=str(verification_fields.get("verification_tx_hash") or "").strip() or None,
                    verified_at=cancelled_at,
                    force=True,
                )
                execution_payload = dict(execution_order.payload_json or {})
                execution_payload["submission_intent"] = dict(submission_intent)
                execution_order.status = "failed"
                execution_order.payload_json = execution_payload
                execution_order.error_message = error_message
                execution_order.updated_at = cancelled_at
            _update_session_status(
                status="failed",
                error_message=error_message,
                payload_patch={
                    "orders_written": 0,
                    "submit_cancelled": True,
                },
            )
            _append_event(
                event_type="session_failed",
                severity="error",
                message=error_message,
                payload={
                    "submit_cancelled": True,
                    "failed_legs": len(entry_submit_placeholders),
                },
            )
            await _persist_execution_projection_safely(signal_status="failed", effective_price=None)

        async def _submit_execution_wave_with_cancellation_protection(
            *,
            legs_with_notionals: list[tuple[dict[str, Any], float]],
            wave_error_message: str,
        ) -> list[LegSubmitResult]:
            _started_at = _time.monotonic()
            try:
                async with release_conn(self.db):
                    return await submit_execution_wave(
                        mode=mode,
                        signal=signal,
                        legs_with_notionals=legs_with_notionals,
                        strategy_params=strategy_params,
                        risk_limits=risk_limits,
                        trader_id=trader_id,
                    )
            except asyncio.CancelledError:
                await asyncio.shield(
                    _finalize_cancelled_live_submit(error_message=wave_error_message)
                )
                raise
            finally:
                _record_execution_timing("venue_submit_wave", _started_at)

        _append_event(
            event_type="session_created",
            message=f"Execution session created with {len(legs)} leg(s)",
            payload={"policy": plan["policy"], "mode": mode},
        )

        bundle_coverage = _event_bundle_coverage(signal_payload, plan)
        if bundle_coverage is not None and list(bundle_coverage.get("missing_market_ids") or []):
            rejection_reason = "Guaranteed event bundle does not cover the full event market roster."
            _append_event(
                event_type="session_rejected",
                severity="error",
                message=rejection_reason,
                payload=bundle_coverage,
            )
            _update_session_status(
                status="failed",
                error_message=rejection_reason,
                payload_patch={
                    "orders_written": 0,
                    "bundle_coverage": bundle_coverage,
                },
            )
            await _persist_execution_projection_safely(signal_status="failed", effective_price=None)
            return SessionExecutionResult(
                session_id=session_row.id,
                status="failed",
                effective_price=None,
                error_message=rejection_reason,
                orders_written=0,
                payload=_payload_with_execution_timing({
                    "execution_plan": plan,
                    "bundle_coverage": bundle_coverage,
                    "legs": [],
                }),
                created_orders=[],
            )

        self_crossing_violation = _self_crossing_plan_violation(legs)
        if self_crossing_violation is not None:
            rejection_reason = "Execution plan self-crosses: buy limit is at or above sell limit for the same token."
            _append_event(
                event_type="session_rejected",
                severity="error",
                message=rejection_reason,
                payload=self_crossing_violation,
            )
            _update_session_status(
                status="failed",
                error_message=rejection_reason,
                payload_patch={
                    "orders_written": 0,
                    "self_crossing_plan": self_crossing_violation,
                },
            )
            await _persist_execution_projection_safely(signal_status="failed", effective_price=None)
            return SessionExecutionResult(
                session_id=session_row.id,
                status="failed",
                effective_price=None,
                error_message=rejection_reason,
                orders_written=0,
                payload=_payload_with_execution_timing({
                    "execution_plan": plan,
                    "self_crossing_plan": self_crossing_violation,
                    "legs": [],
                }),
                created_orders=[],
            )

        _update_session_status(status="placing")

        preflight_violation = _bundle_preflight_violation(
            signal=signal,
            legs=legs,
            strategy_params=strategy_params,
            mode=mode,
            size_usd=size_usd,
        )
        if preflight_violation is not None:
            rejection_reason = "Bundle cannot be executed safely at the selected size."
            _append_event(
                event_type="session_rejected",
                severity="warn",
                message=rejection_reason,
                payload=preflight_violation,
            )
            _update_session_status(
                status="failed",
                error_message=rejection_reason,
                payload_patch={
                    "orders_written": 0,
                    "bundle_preflight": preflight_violation,
                },
            )
            await _persist_execution_projection_safely(signal_status="failed", effective_price=None)
            return SessionExecutionResult(
                session_id=session_row.id,
                status="failed",
                effective_price=None,
                error_message=rejection_reason,
                orders_written=0,
                payload=_payload_with_execution_timing({
                    "execution_plan": plan,
                    "bundle_preflight": preflight_violation,
                    "legs": [],
                }),
                created_orders=[],
            )

        waves = execution_waves(plan["policy"], legs)
        reprice_enabled = supports_reprice(plan["policy"])
        max_reprice_attempts = safe_int(constraints.get("max_reprice_attempts"), 3)
        strategy_instance = _strategy_instance_for_execution(strategy_key)
        preplace_take_profit_exit = _resolve_preplace_take_profit_exit(
            strategy_instance=strategy_instance,
            params=dict(explicit_strategy_params or {}),
        )

        # Stamp a deterministic CLOB idempotency key onto every live leg
        # before either the pre-submit row write or the venue submission.
        # The key is derived from (trader_id, signal_id, leg_id) so:
        #
        #   * a retry of the same leg produces the same key — reconcile
        #     can attach to a previously-issued venue order rather than
        #     duplicating it,
        #   * the key gets recorded on both the venue order
        #     (OrderArgsV2.metadata) and the pre-submit TraderOrder row's
        #     payload_json, giving reconcile an exact match path when
        #     the post-submit DB update is lost.
        #
        # Mirrors the fast-tier guarantee from fa4d823 — the orchestrator
        # path used to fall back to fuzzy attribute matching for orphan
        # recovery; with the key in place every multi-leg session is
        # precisely recoverable too.
        signal_id_for_key = str(getattr(signal, "id", "") or "").strip()
        for wave_index, wave in enumerate(waves):
            if mode == "live" and signal_id_for_key:
                for leg_in_wave in wave:
                    leg_id_for_key = str(leg_in_wave.get("leg_id") or "").strip()
                    if not leg_id_for_key:
                        continue
                    if leg_in_wave.get("clob_idempotency_key"):
                        continue
                    leg_in_wave["clob_idempotency_key"] = derive_clob_idempotency_key(
                        trader_id=trader_id,
                        signal_id=signal_id_for_key,
                        leg_id=leg_id_for_key,
                    )
            wave_with_notionals = [(leg, safe_float(leg.get("requested_notional_usd"), 0.0)) for leg in wave]
            if mode == "live":
                created_pre_submit_placeholders = False
                for leg_payload, _leg_notional in wave_with_notionals:
                    leg_id = str(leg_payload.get("leg_id") or "").strip()
                    if not leg_id or leg_id in entry_submit_placeholders:
                        continue
                    placeholder_pair = _create_entry_submit_placeholder(
                        leg_id=leg_id,
                        leg_payload=dict(leg_payload),
                    )
                    if placeholder_pair is None:
                        continue
                    entry_submit_placeholders[leg_id] = placeholder_pair
                    created_pre_submit_placeholders = True
                if created_pre_submit_placeholders:
                    await _commit_pre_submit_projection()
            wave_results = await _submit_execution_wave_with_cancellation_protection(
                legs_with_notionals=wave_with_notionals,
                wave_error_message="Execution session cancelled during live order submission.",
            )

            # Append each leg's place_order sub-stage breakdown to the
            # function-scope ``_submit_wave_breakdowns`` collector so the
            # orchestrator's cycle-slow log can surface which sub-stage
            # owned the submit time.  Was opaque before — ps_submit_order
            # = 39.6s with no path to the offender.
            for _result in wave_results:
                _result_payload = getattr(_result, "payload", None)
                if not isinstance(_result_payload, dict):
                    continue
                _breakdown = _result_payload.get("submit_breakdown")
                if isinstance(_breakdown, dict) and _breakdown:
                    _submit_wave_breakdowns.append(dict(_breakdown))

            for result in wave_results:
                leg_id = str(result.leg_id)
                leg_payload = (
                    next((candidate for candidate in wave if str(candidate.get("leg_id")) == leg_id), None) or {}
                )
                leg_row = leg_rows.get(leg_id)
                if leg_row is None:
                    continue

                mapped_leg_status = "failed"
                normalized_status = str(result.status or "").strip().lower()
                if _is_live_position_cap_skip(
                    mode=mode,
                    status=normalized_status,
                    error_message=result.error_message,
                ):
                    normalized_status = "skipped"
                if normalized_status == "executed":
                    mapped_leg_status = "completed"
                    completed_legs += 1
                elif normalized_status in {"open", "submitted"}:
                    mapped_leg_status = "open"
                    open_legs += 1
                elif normalized_status == "skipped":
                    mapped_leg_status = "skipped"
                    skipped_legs += 1
                    skip_reason = str(result.error_message or "").strip()
                    if skip_reason:
                        skip_reasons.append(skip_reason)
                else:
                    failed_legs += 1

                if mapped_leg_status == "failed" and reprice_enabled and max_reprice_attempts > 0:
                    reprice_price = reprice_limit_price(
                        safe_float(leg_payload.get("limit_price"), None),
                        str(leg_payload.get("side") or "buy"),
                        1,
                    )
                    if reprice_price is not None:
                        # Cancel the failed order before repricing to prevent double-fills
                        prev_clob_id = str(getattr(result, "clob_order_id", "") or "").strip()
                        if prev_clob_id:
                            try:
                                from services.live_execution_service import live_execution_service
                                await live_execution_service.cancel_order(prev_clob_id)
                            except Exception:
                                pass
                        leg_payload["limit_price"] = reprice_price
                        retry_result = (
                            await _submit_execution_wave_with_cancellation_protection(
                                legs_with_notionals=[
                                    (leg_payload, safe_float(leg_payload.get("requested_notional_usd"), 0.0))
                                ],
                                wave_error_message="Execution session cancelled during live order repricing.",
                            )
                        )[0]
                        # Surface the retry wave's breakdown too.
                        _retry_payload = getattr(retry_result, "payload", None)
                        if isinstance(_retry_payload, dict):
                            _retry_breakdown = _retry_payload.get("submit_breakdown")
                            if isinstance(_retry_breakdown, dict) and _retry_breakdown:
                                _submit_wave_breakdowns.append(dict(_retry_breakdown))
                        normalized_retry = str(retry_result.status or "").strip().lower()
                        if normalized_retry in {"executed", "open", "submitted"}:
                            result = retry_result
                            normalized_status = normalized_retry
                            mapped_leg_status = "completed" if normalized_retry == "executed" else "open"
                            failed_legs -= 1
                            if mapped_leg_status == "completed":
                                completed_legs += 1
                            else:
                                open_legs += 1
                            _append_event(
                                event_type="reprice_success",
                                leg_id=leg_row.id,
                                message="Leg succeeded after reprice attempt",
                                payload={"new_limit_price": reprice_price},
                            )

                actual_filled_shares = _result_filled_shares(result)
                actual_filled_notional = _result_filled_notional(result)
                actual_fill_price = _result_fill_price(result)

                _update_leg_row(
                    leg_row=leg_row,
                    status=mapped_leg_status,
                    filled_notional_usd=actual_filled_notional,
                    filled_shares=actual_filled_shares,
                    avg_fill_price=actual_fill_price,
                    last_error=result.error_message if mapped_leg_status in {"failed", "skipped"} else None,
                    metadata_patch={
                        "wave_index": wave_index,
                        "provider_order_id": result.provider_order_id,
                        "provider_clob_order_id": result.provider_clob_order_id,
                    },
                )

                order_payload = dict(result.payload or {})
                # Persist provider IDs at the top level so venue authority
                # recovery can match this order and avoid creating duplicates.
                if result.provider_order_id:
                    order_payload["provider_order_id"] = result.provider_order_id
                if result.provider_clob_order_id:
                    order_payload["provider_clob_order_id"] = result.provider_clob_order_id
                order_payload["execution_session"] = {
                    "session_id": session_row.id,
                    "leg_id": leg_row.id,
                    "leg_ref": leg_id,
                    "policy": plan["policy"],
                }
                order_payload["strategy_type"] = str(getattr(signal, "strategy_type", "") or "").strip().lower()
                order_payload["strategy_context"] = (
                    getattr(signal, "strategy_context_json", None) or getattr(signal, "strategy_context", None) or {}
                )
                signal_payload = getattr(signal, "payload_json", None)
                runtime_params = dict(strategy_params or {})
                explicit_params = dict(explicit_strategy_params or {})
                order_payload["strategy_params"] = dict(runtime_params)
                exit_config: dict[str, Any] = {}
                for param_key, param_value in explicit_params.items():
                    key_text = str(param_key or "").strip()
                    if not key_text:
                        continue
                    exit_config[key_text] = param_value
                    if key_text.startswith("live_") and len(key_text) > 5:
                        canonical_key = key_text[5:]
                        if canonical_key and canonical_key not in exit_config:
                            exit_config[canonical_key] = param_value
                exit_key_aliases = {
                    "rapid_take_profit_pct": ("live_rapid_take_profit_pct", "rapid_take_profit_pct"),
                    "rapid_exit_window_minutes": ("live_rapid_exit_window_minutes", "rapid_exit_window_minutes"),
                    "rapid_exit_min_increase_pct": (
                        "live_rapid_exit_min_increase_pct",
                        "rapid_exit_min_increase_pct",
                    ),
                    "rapid_exit_breakeven_buffer_pct": (
                        "live_rapid_exit_breakeven_buffer_pct",
                        "rapid_exit_breakeven_buffer_pct",
                    ),
                    "take_profit_pct": ("live_take_profit_pct", "take_profit_pct"),
                    "stop_loss_pct": ("live_stop_loss_pct", "stop_loss_pct"),
                    "stop_loss_policy": ("live_stop_loss_policy", "stop_loss_policy", "stop_loss_mode"),
                    "stop_loss_activation_seconds": (
                        "live_stop_loss_activation_seconds",
                        "stop_loss_activation_seconds",
                        "live_stop_loss_near_close_seconds",
                        "stop_loss_near_close_seconds",
                    ),
                    "trailing_stop_pct": ("live_trailing_stop_pct", "trailing_stop_pct"),
                    "trailing_stop_activation_profit_pct": (
                        "live_trailing_stop_activation_profit_pct",
                        "trailing_stop_activation_profit_pct",
                    ),
                    "max_hold_minutes": ("live_max_hold_minutes", "max_hold_minutes"),
                    "min_hold_minutes": ("live_min_hold_minutes", "min_hold_minutes"),
                    "resolve_only": ("live_resolve_only", "resolve_only"),
                    "close_on_inactive_market": ("live_close_on_inactive_market", "close_on_inactive_market"),
                }
                for base_key in (
                    "rapid_take_profit_pct",
                    "rapid_exit_window_minutes",
                    "take_profit_pct",
                    "stop_loss_pct",
                    "stop_loss_policy",
                    "stop_loss_activation_seconds",
                    "trailing_stop_pct",
                    "trailing_stop_activation_profit_pct",
                    "max_hold_minutes",
                    "min_hold_minutes",
                ):
                    for suffix in ("5m", "15m", "1h", "4h"):
                        scoped_key = f"{base_key}_{suffix}"
                        exit_key_aliases[scoped_key] = (f"live_{scoped_key}", scoped_key)
                for target_key, aliases in exit_key_aliases.items():
                    selected_value = None
                    for alias_key in aliases:
                        if alias_key in explicit_params:
                            selected_value = explicit_params.get(alias_key)
                            break
                    if selected_value is not None:
                        exit_config[target_key] = selected_value

                resolved_signal_payload = signal_payload if isinstance(signal_payload, dict) else {}
                leg_market_id = str(leg_payload.get("market_id") or getattr(signal, "market_id", "") or "").strip()
                leg_market_question = _resolve_leg_market_question(
                    resolved_signal_payload,
                    leg_payload,
                    str(getattr(signal, "market_question", "") or "").strip(),
                )
                leg_direction = _resolve_leg_direction(leg_payload, str(getattr(signal, "direction", "") or ""))
                leg_entry_price = safe_float(leg_payload.get("limit_price"), safe_float(result.effective_price, None))
                if leg_market_id:
                    order_payload["market_id"] = leg_market_id
                if leg_market_question:
                    order_payload["market_question"] = leg_market_question
                if leg_direction:
                    order_payload["direction"] = leg_direction
                if leg_entry_price is not None and leg_entry_price > 0.0:
                    order_payload["entry_price"] = float(leg_entry_price)
                if isinstance(signal_payload, dict):
                    live_market_payload = signal_payload.get("live_market")
                    if isinstance(live_market_payload, dict):
                        signal_selected_token_id = str(live_market_payload.get("selected_token_id") or "").strip()
                        signal_market_key = str(
                            live_market_payload.get("market_id") or live_market_payload.get("condition_id") or ""
                        ).strip().lower()
                        leg_token_id = str(leg_payload.get("token_id") or "").strip()
                        leg_market_key = leg_market_id.strip().lower() if leg_market_id else ""
                        if (
                            signal_selected_token_id
                            and leg_token_id
                            and signal_selected_token_id == leg_token_id
                        ) or (
                            signal_market_key
                            and leg_market_key
                            and signal_market_key == leg_market_key
                        ):
                            order_payload["live_market"] = dict(live_market_payload)
                live_market_payload = order_payload.get("live_market")
                live_market = dict(live_market_payload) if isinstance(live_market_payload, dict) else {}
                if leg_market_id:
                    live_market["market_id"] = leg_market_id
                if leg_market_question:
                    live_market["market_question"] = leg_market_question
                selected_token_id = str(order_payload.get("token_id") or leg_payload.get("token_id") or "").strip()
                if selected_token_id:
                    live_market["selected_token_id"] = selected_token_id
                selected_outcome = str(leg_payload.get("outcome") or "").strip().lower()
                if selected_outcome:
                    live_market["selected_outcome"] = selected_outcome
                live_selected_price = safe_float(result.effective_price, leg_entry_price)
                if live_selected_price is not None and live_selected_price > 0.0:
                    live_market["live_selected_price"] = float(live_selected_price)
                if live_market:
                    order_payload["live_market"] = live_market
                order_payload["strategy_exit_config"] = exit_config
                placeholder_pair = entry_submit_placeholders.get(leg_id)
                if normalized_status != "skipped" or placeholder_pair is not None:
                    order_write_inputs.append(
                        {
                            "leg_id": leg_id,
                            "leg_row_id": leg_row.id,
                            "leg_payload": dict(leg_payload),
                            "market_id": leg_market_id,
                            "market_question": leg_market_question,
                            "direction": leg_direction,
                            "entry_price": leg_entry_price,
                            "normalized_status": normalized_status,
                            "result": result,
                            "order_payload": order_payload,
                            "exit_config": exit_config,
                            "requested_shares": safe_float(leg_payload.get("requested_shares"), None),
                            "actual_filled_shares": actual_filled_shares,
                            "actual_filled_notional_usd": actual_filled_notional,
                            "actual_fill_price": actual_fill_price,
                            "existing_trader_order": placeholder_pair[0] if placeholder_pair is not None else None,
                            "existing_execution_order": placeholder_pair[1] if placeholder_pair is not None else None,
                        }
                    )

                leg_execution_records.append(
                    {
                        "leg_id": leg_id,
                        "status": normalized_status,
                        "effective_price": actual_fill_price,
                        "notional_usd": actual_filled_notional,
                        "filled_shares": actual_filled_shares,
                    }
                )

        pair_lock_enabled = requires_pair_lock(plan["policy"], constraints)
        max_unhedged = safe_float(constraints.get("max_unhedged_notional_usd"), 0.0)
        if pair_lock_enabled and max_unhedged > 0:
            current_unhedged = safe_float(getattr(session_row, "unhedged_notional_usd", None), 0.0)
            if current_unhedged <= 0.0:
                session_detail = await get_execution_session_detail(self.db, session_row.id)
                session_view = session_detail.get("session") if isinstance(session_detail, dict) else {}
                if isinstance(session_view, dict):
                    current_unhedged = safe_float(session_view.get("unhedged_notional_usd"), current_unhedged)
            if current_unhedged > max_unhedged:
                violation_reason = (
                    f"Pair lock violation: unhedged notional {current_unhedged:.2f} exceeded {max_unhedged:.2f}."
                )
                _append_event(
                    event_type="pair_lock_violation",
                    severity="warn",
                    message=violation_reason,
                    payload={
                        "current_unhedged_notional_usd": current_unhedged,
                        "max_unhedged_notional_usd": max_unhedged,
                    },
                )
                cancel_attempted_ids: set[str] = set()
                cancelled_provider_orders: list[str] = []
                cancel_failed_provider_orders: list[str] = []
                for item in order_write_inputs:
                    normalized_status = str(item.get("normalized_status") or "").strip().lower()
                    if normalized_status not in {"open", "submitted"}:
                        continue
                    result = item.get("result")
                    candidate_ids = [
                        str(getattr(result, "provider_clob_order_id", "") or "").strip(),
                        str(getattr(result, "provider_order_id", "") or "").strip(),
                    ]
                    cancelled = False
                    for candidate_id in candidate_ids:
                        if not candidate_id or candidate_id in cancel_attempted_ids:
                            continue
                        cancel_attempted_ids.add(candidate_id)
                        async with release_conn(self.db):
                            cancelled_via_provider = await cancel_live_provider_order(candidate_id)
                        if cancelled_via_provider:
                            cancelled = True
                            cancelled_provider_orders.append(candidate_id)
                            break
                    if not cancelled and any(candidate_ids):
                        cancel_failed_provider_orders.extend([cid for cid in candidate_ids if cid])
                    leg_row_id = str(item.get("leg_row_id") or "").strip()
                    if leg_row_id:
                        leg_row = next(
                            (candidate for candidate in leg_rows.values() if str(candidate.id or "") == leg_row_id),
                            None,
                        )
                        if leg_row is not None:
                            _update_leg_row(
                                leg_row=leg_row,
                                status="cancelled",
                                last_error=violation_reason,
                                metadata_patch={"pair_lock_violation": True},
                            )
                    leg_id = str(item.get("leg_id") or "").strip()
                    for record in leg_execution_records:
                        if str(record.get("leg_id") or "") != leg_id:
                            continue
                        if str(record.get("status") or "").strip().lower() in {"open", "submitted"}:
                            record["status"] = "cancelled"
                        break

                effective_price = self._effective_price_from_leg_results(leg_execution_records)
                _update_session_status(
                    status="failed",
                    error_message=violation_reason,
                    payload_patch={
                        "orders_written": 0,
                        "pair_lock_violation": {
                            "current_unhedged_notional_usd": current_unhedged,
                            "max_unhedged_notional_usd": max_unhedged,
                            "cancelled_provider_orders": cancelled_provider_orders,
                            "cancel_failed_provider_orders": cancel_failed_provider_orders,
                        },
                    },
                )
                _append_event(
                    event_type="session_aborted",
                    severity="error",
                    message="Execution session aborted after pair lock violation",
                    payload={
                        "current_unhedged_notional_usd": current_unhedged,
                        "max_unhedged_notional_usd": max_unhedged,
                        "cancelled_provider_orders": cancelled_provider_orders,
                        "cancel_failed_provider_orders": cancel_failed_provider_orders,
                    },
                )
                await _persist_execution_projection_safely(signal_status="failed", effective_price=effective_price)
                return SessionExecutionResult(
                    session_id=session_row.id,
                    status="failed",
                    effective_price=effective_price,
                    error_message=violation_reason,
                    orders_written=0,
                    payload=_payload_with_execution_timing({
                        "execution_plan": plan,
                        "legs": leg_execution_records,
                        "pair_lock_violation": {
                            "current_unhedged_notional_usd": current_unhedged,
                            "max_unhedged_notional_usd": max_unhedged,
                            "cancelled_provider_orders": cancelled_provider_orders,
                            "cancel_failed_provider_orders": cancel_failed_provider_orders,
                        },
                    }),
                    created_orders=created_order_records,
                )

        tolerance_ratio = max(0.0, min(1.0, safe_float(constraints.get("leg_fill_tolerance_ratio"), 0.02) or 0.02))
        has_partial_bundle_fill = len(legs) > 1 and any(
            safe_float(item.get("actual_filled_shares"), 0.0) > 0.0 for item in order_write_inputs
        ) and any(
            _remaining_requested_shares(
                requested_shares=item.get("requested_shares"),
                filled_shares=item.get("actual_filled_shares"),
                tolerance_ratio=tolerance_ratio,
            )
            > 0.0
            for item in order_write_inputs
        )
        if has_partial_bundle_fill:
            _append_event(
                event_type="bundle_recovery_started",
                severity="warn",
                message="Multi-leg entry partially filled; attempting completion rescue before flatten.",
                payload={"leg_fill_tolerance_ratio": tolerance_ratio},
            )

            cancelled_provider_orders: list[str] = []
            cancel_failed_provider_orders: list[str] = []
            for item in order_write_inputs:
                remaining_shares = _remaining_requested_shares(
                    requested_shares=item.get("requested_shares"),
                    filled_shares=item.get("actual_filled_shares"),
                    tolerance_ratio=tolerance_ratio,
                )
                if remaining_shares <= 0.0:
                    continue
                normalized_status = str(item.get("normalized_status") or "").strip().lower()
                if normalized_status not in {"open", "submitted"}:
                    continue
                result = item.get("result")
                candidate_ids = [
                    str(getattr(result, "provider_clob_order_id", "") or "").strip(),
                    str(getattr(result, "provider_order_id", "") or "").strip(),
                ]
                for candidate_id in candidate_ids:
                    if not candidate_id:
                        continue
                    async with release_conn(self.db):
                        cancelled = await cancel_live_provider_order(candidate_id)
                    if cancelled:
                        cancelled_provider_orders.append(candidate_id)
                        break
                else:
                    cancel_failed_provider_orders.extend([candidate_id for candidate_id in candidate_ids if candidate_id])

            rescue_inputs: list[tuple[dict[str, Any], float, dict[str, Any], float, float]] = []
            for item in order_write_inputs:
                remaining_shares = _remaining_requested_shares(
                    requested_shares=item.get("requested_shares"),
                    filled_shares=item.get("actual_filled_shares"),
                    tolerance_ratio=tolerance_ratio,
                )
                if remaining_shares <= 0.0:
                    continue
                leg_payload = dict(item.get("leg_payload") or {})
                side_key = str(leg_payload.get("side") or "buy").strip().lower()
                base_price = safe_float(
                    leg_payload.get("limit_price"),
                    safe_float(item.get("actual_fill_price"), safe_float(getattr(item.get("result"), "effective_price", None), None)),
                )
                rescue_price = reprice_limit_price(base_price, side_key, 2) if base_price is not None else None
                if rescue_price is None or rescue_price <= 0.0:
                    rescue_price = base_price
                leg_payload["price_policy"] = "taker_limit"
                leg_payload["time_in_force"] = "IOC"
                leg_payload["post_only"] = False
                if rescue_price is not None and rescue_price > 0.0:
                    leg_payload["limit_price"] = rescue_price
                rescue_notional = remaining_shares * max(0.0001, safe_float(leg_payload.get("limit_price"), 0.01) or 0.01)
                rescue_inputs.append((leg_payload, rescue_notional, item, rescue_price or 0.0, remaining_shares))

            rescue_results: list[Any] = []
            if rescue_inputs:
                rescue_results = await _submit_execution_wave_with_cancellation_protection(
                    legs_with_notionals=[
                        (leg_payload, rescue_notional)
                        for leg_payload, rescue_notional, _, _, _ in rescue_inputs
                    ],
                    wave_error_message="Execution session cancelled during bundle rescue submission.",
                )
                # Surface rescue wave breakdowns too.
                for _rescue_result in rescue_results:
                    _rescue_payload = getattr(_rescue_result, "payload", None)
                    if not isinstance(_rescue_payload, dict):
                        continue
                    _rescue_breakdown = _rescue_payload.get("submit_breakdown")
                    if isinstance(_rescue_breakdown, dict) and _rescue_breakdown:
                        _submit_wave_breakdowns.append(dict(_rescue_breakdown))

            for rescue_result, (_, _, item, rescue_price, remaining_shares) in zip(rescue_results, rescue_inputs):
                rescue_filled_shares = _result_filled_shares(rescue_result)
                rescue_filled_notional = _result_filled_notional(rescue_result)
                rescue_fill_price = _result_fill_price(rescue_result)
                item["actual_filled_shares"] = float(max(0.0, safe_float(item.get("actual_filled_shares"), 0.0) + rescue_filled_shares))
                item["actual_filled_notional_usd"] = float(
                    max(0.0, safe_float(item.get("actual_filled_notional_usd"), 0.0) + rescue_filled_notional)
                )
                if rescue_fill_price is not None and rescue_fill_price > 0.0:
                    item["actual_fill_price"] = rescue_fill_price

                leg_row_id = str(item.get("leg_row_id") or "").strip()
                leg_row = next(
                    (candidate for candidate in leg_rows.values() if str(candidate.id or "") == leg_row_id),
                    None,
                )
                if leg_row is not None:
                    post_rescue_remaining = _remaining_requested_shares(
                        requested_shares=item.get("requested_shares"),
                        filled_shares=item.get("actual_filled_shares"),
                        tolerance_ratio=tolerance_ratio,
                    )
                    rescue_status_key = str(getattr(rescue_result, "status", "") or "").strip().lower()
                    if post_rescue_remaining <= 0.0:
                        updated_leg_status = "completed"
                    elif rescue_status_key in {"open", "submitted"}:
                        updated_leg_status = "failed"
                    else:
                        updated_leg_status = "failed"
                    _update_leg_row(
                        leg_row=leg_row,
                        status=updated_leg_status,
                        filled_notional_usd=safe_float(item.get("actual_filled_notional_usd"), 0.0),
                        filled_shares=safe_float(item.get("actual_filled_shares"), 0.0),
                        avg_fill_price=safe_float(item.get("actual_fill_price"), None),
                        last_error=(
                            None
                            if updated_leg_status == "completed"
                            else str(getattr(rescue_result, "error_message", "") or "completion_rescue_incomplete")
                        ),
                        metadata_patch={
                            "completion_rescue": {
                                "requested_remaining_shares": remaining_shares,
                                "fallback_price": rescue_price if rescue_price > 0.0 else None,
                                "status": rescue_status_key,
                                "filled_shares": rescue_filled_shares,
                                "effective_price": rescue_fill_price,
                            }
                        },
                    )

                for record in leg_execution_records:
                    if str(record.get("leg_id") or "") != str(item.get("leg_id") or ""):
                        continue
                    record["status"] = str(getattr(rescue_result, "status", "") or "").strip().lower() or record.get("status")
                    record["effective_price"] = safe_float(item.get("actual_fill_price"), record.get("effective_price"))
                    record["notional_usd"] = safe_float(item.get("actual_filled_notional_usd"), record.get("notional_usd"))
                    record["filled_shares"] = safe_float(item.get("actual_filled_shares"), record.get("filled_shares"))
                    break

                recovery_payload = {
                    **dict(item.get("order_payload") or {}),
                    "bundle_recovery": {
                        "recovery_action": "completion_rescue",
                        "requested_remaining_shares": float(remaining_shares),
                        "filled_shares": float(rescue_filled_shares),
                        "effective_price": rescue_fill_price,
                    },
                }
                recovery_order_write_inputs.append(
                    {
                        "leg_id": str(item.get("leg_id") or ""),
                        "leg_row_id": leg_row_id,
                        "leg_payload": dict(item.get("leg_payload") or {}),
                        "market_id": str(item.get("market_id") or ""),
                        "market_question": str(item.get("market_question") or ""),
                        "direction": str(item.get("direction") or ""),
                        "entry_price": safe_float(rescue_price, safe_float(rescue_fill_price, None)),
                        "normalized_status": str(getattr(rescue_result, "status", "") or "").strip().lower() or "failed",
                        "result": rescue_result,
                        "order_payload": recovery_payload,
                        "exit_config": {},
                        "requested_shares": float(max(0.0, remaining_shares)),
                        "actual_filled_shares": float(max(0.0, rescue_filled_shares)),
                        "actual_filled_notional_usd": float(max(0.0, rescue_filled_notional)),
                        "actual_fill_price": rescue_fill_price,
                        "session_order_action": "hedge_rescue",
                    }
                )
                _append_event(
                    event_type="bundle_completion_rescue",
                    severity="warn" if rescue_filled_shares <= 0.0 else "info",
                    leg_id=leg_row_id or None,
                    message="Completion rescue attempted for incomplete leg.",
                    payload={
                        "requested_remaining_shares": remaining_shares,
                        "rescue_status": str(getattr(rescue_result, "status", "") or "").strip().lower() or "failed",
                        "rescue_filled_shares": rescue_filled_shares,
                        "rescue_filled_notional_usd": rescue_filled_notional,
                        "rescue_effective_price": rescue_fill_price,
                    },
                )

            residual_bundle_items = [
                item
                for item in order_write_inputs
                if _remaining_requested_shares(
                    requested_shares=item.get("requested_shares"),
                    filled_shares=item.get("actual_filled_shares"),
                    tolerance_ratio=tolerance_ratio,
                )
                > 0.0
            ]
            if residual_bundle_items:
                flatten_attempts: list[dict[str, Any]] = []
                for item in order_write_inputs:
                    filled_shares = max(0.0, safe_float(item.get("actual_filled_shares"), 0.0) or 0.0)
                    if filled_shares <= 0.0:
                        continue
                    leg_payload = dict(item.get("leg_payload") or {})
                    flatten_side = "SELL" if str(leg_payload.get("side") or "buy").strip().lower() == "buy" else "BUY"
                    token_id = str(item.get("order_payload", {}).get("token_id") or leg_payload.get("token_id") or "").strip()
                    base_price = safe_float(
                        item.get("actual_fill_price"),
                        safe_float(leg_payload.get("limit_price"), safe_float(getattr(item.get("result"), "effective_price", None), None)),
                    )
                    fallback_price = _recovery_price(
                        side=flatten_side,
                        base_price=base_price,
                        price_bps=_BUNDLE_FLATTEN_PRICE_BPS,
                    )
                    flatten_execution = await execute_live_order(
                        token_id=token_id,
                        side=flatten_side,
                        size=float(filled_shares),
                        fallback_price=fallback_price,
                        market_question=str(item.get("market_question") or ""),
                        opportunity_id=str(getattr(signal, "id", "") or ""),
                        time_in_force="IOC",
                        post_only=False,
                        resolve_live_price=True,
                        prefer_cached_price=True,
                    )
                    flatten_filled_shares = max(
                        0.0,
                        safe_float(flatten_execution.payload.get("filled_size"), 0.0) or 0.0,
                    )
                    if flatten_filled_shares <= 0.0 and flatten_execution.status == "executed":
                        flatten_filled_shares = filled_shares
                    flatten_price = safe_float(
                        flatten_execution.payload.get("average_fill_price"),
                        safe_float(flatten_execution.effective_price, None),
                    )
                    flatten_notional = 0.0
                    if flatten_filled_shares > 0.0 and flatten_price is not None and flatten_price > 0.0:
                        flatten_notional = flatten_filled_shares * flatten_price
                    flatten_residual = max(0.0, filled_shares - flatten_filled_shares)
                    flatten_attempts.append(
                        {
                            "leg_id": str(item.get("leg_id") or ""),
                            "leg_row_id": str(item.get("leg_row_id") or ""),
                            "token_id": token_id or None,
                            "status": str(flatten_execution.status or "").strip().lower() or "failed",
                            "filled_shares": flatten_filled_shares,
                            "requested_shares": filled_shares,
                            "residual_shares": flatten_residual,
                            "effective_price": flatten_price,
                            "error_message": flatten_execution.error_message,
                        }
                    )
                    item["bundle_recovery"] = {
                        "recovery_action": "flatten_after_incomplete_entry",
                        "flatten_status": str(flatten_execution.status or "").strip().lower() or "failed",
                        "flatten_filled_shares": flatten_filled_shares,
                        "residual_shares": flatten_residual,
                        "effective_price": flatten_price,
                    }
                    flatten_payload = {
                        **dict(item.get("order_payload") or {}),
                        "bundle_recovery": {
                            "recovery_action": "flatten_after_incomplete_entry",
                            "entry_filled_shares": float(filled_shares),
                            "flatten_filled_shares": float(flatten_filled_shares),
                            "residual_shares": float(flatten_residual),
                            "effective_price": flatten_price,
                        },
                    }
                    if flatten_filled_shares > 0.0 and flatten_price is not None and flatten_price > 0.0:
                        flatten_payload["position_close"] = {
                            "close_price": float(flatten_price),
                            "price_source": "bundle_recovery_flatten",
                            "close_trigger": "bundle_partial_fill_flatten",
                            "filled_size": float(flatten_filled_shares),
                            "filled_notional_usd": float(flatten_notional),
                            "closed_at": _iso_utc(utcnow()),
                            "reason": "Bundle entry failed; flatten executed.",
                        }
                    flatten_leg_payload = dict(leg_payload)
                    flatten_leg_payload["side"] = flatten_side.lower()
                    flatten_leg_payload["time_in_force"] = "IOC"
                    flatten_leg_payload["post_only"] = False
                    if fallback_price is not None and fallback_price > 0.0:
                        flatten_leg_payload["limit_price"] = fallback_price
                    recovery_order_write_inputs.append(
                        {
                            "leg_id": str(item.get("leg_id") or ""),
                            "leg_row_id": str(item.get("leg_row_id") or ""),
                            "leg_payload": flatten_leg_payload,
                            "market_id": str(item.get("market_id") or ""),
                            "market_question": str(item.get("market_question") or ""),
                            "direction": _resolve_leg_direction(
                                flatten_leg_payload,
                                str(item.get("direction") or ""),
                            ),
                            "entry_price": safe_float(fallback_price, flatten_price),
                            "normalized_status": str(flatten_execution.status or "").strip().lower() or "failed",
                            "result": LegSubmitResult(
                                leg_id=str(item.get("leg_id") or ""),
                                status=str(flatten_execution.status or "").strip().lower() or "failed",
                                effective_price=flatten_price,
                                error_message=flatten_execution.error_message,
                                payload=dict(flatten_execution.payload or {}),
                                provider_order_id=flatten_execution.order_id,
                                provider_clob_order_id=str(flatten_execution.payload.get("clob_order_id") or "").strip() or None,
                                shares=float(filled_shares),
                                notional_usd=float(flatten_notional),
                            ),
                            "order_payload": flatten_payload,
                            "exit_config": {},
                            "requested_shares": float(max(0.0, filled_shares)),
                            "actual_filled_shares": float(max(0.0, flatten_filled_shares)),
                            "actual_filled_notional_usd": float(max(0.0, flatten_notional)),
                            "actual_fill_price": flatten_price,
                            "session_order_action": "flatten",
                            "persisted_order_status": (
                                "executed"
                                if flatten_filled_shares > 0.0
                                else str(flatten_execution.status or "").strip().lower() or "failed"
                            ),
                        }
                    )
                    _append_event(
                        event_type="bundle_flatten",
                        severity="warn" if flatten_residual > 0.0 else "info",
                        leg_id=str(item.get("leg_row_id") or "") or None,
                        message="Flatten order submitted after incomplete multi-leg entry.",
                        payload={
                            "flatten_status": str(flatten_execution.status or "").strip().lower() or "failed",
                            "requested_shares": filled_shares,
                            "flatten_filled_shares": flatten_filled_shares,
                            "residual_shares": flatten_residual,
                            "effective_price": flatten_price,
                        },
                    )

                residual_exposure = any(
                    max(0.0, safe_float(attempt.get("residual_shares"), 0.0) or 0.0) > 0.0 for attempt in flatten_attempts
                )
                bundle_recovery_outcome = {
                    "cancelled_provider_orders": cancelled_provider_orders,
                    "cancel_failed_provider_orders": cancel_failed_provider_orders,
                    "flatten_attempts": flatten_attempts,
                    "residual_exposure": residual_exposure,
                    "recovered_to_flat": not residual_exposure,
                }
                # When the flatten itself fails to flatten — i.e. there's
                # leftover wallet exposure on a leg that filled — the
                # entry order(s) become "zombie" rows: status='executed'
                # but no real position the lifecycle can manage with a
                # normal exit.  Tag the filled entry orders so the
                # position_lifecycle can terminalize them on the next
                # reconcile pass instead of burning cycles re-attempting
                # a flatten the session_engine already proved cannot
                # complete in this session window.
                if residual_exposure:
                    residual_marker = {
                        "status": "unrecoverable",
                        "marked_at": _iso_utc(utcnow()),
                        "flatten_attempts": flatten_attempts,
                    }
                    for entry_item in order_write_inputs:
                        if safe_float(entry_item.get("actual_filled_shares"), 0.0) > 0.0:
                            entry_payload = entry_item.get("order_payload") or {}
                            if not isinstance(entry_payload, dict):
                                entry_payload = {}
                            entry_payload["bundle_recovery_residual"] = residual_marker
                            entry_item["order_payload"] = entry_payload
                _append_event(
                    event_type="bundle_recovery_completed",
                    severity="error" if residual_exposure else "warn",
                    message=(
                        "Bundle recovery left residual exposure after flatten attempts."
                        if residual_exposure
                        else "Bundle recovery flattened filled legs after incomplete entry."
                    ),
                    payload=bundle_recovery_outcome,
                )
            else:
                bundle_recovery_outcome = {
                    "cancelled_provider_orders": cancelled_provider_orders,
                    "cancel_failed_provider_orders": cancel_failed_provider_orders,
                    "residual_exposure": False,
                    "recovered_to_flat": False,
                    "completion_rescue_succeeded": True,
                }
                _append_event(
                    event_type="bundle_recovery_completed",
                    severity="info",
                    message="Bundle completion rescue filled the remaining legs.",
                    payload=bundle_recovery_outcome,
                )
            failed_legs = sum(
                1 for leg_row in leg_rows.values() if str(leg_row.status or "").strip().lower() in {"failed", "cancelled"}
            )
            open_legs = sum(1 for leg_row in leg_rows.values() if str(leg_row.status or "").strip().lower() == "open")
            completed_legs = sum(
                1 for leg_row in leg_rows.values() if str(leg_row.status or "").strip().lower() == "completed"
            )

        all_order_write_inputs = list(order_write_inputs) + list(recovery_order_write_inputs)
        for item in all_order_write_inputs:
            result = item["result"]
            leg_payload = dict(item.get("leg_payload") or {})
            order_payload = dict(item.get("order_payload") or {})
            normalized_status = str(item.get("normalized_status") or "").strip().lower()
            exit_config = dict(item.get("exit_config") or {})
            actual_filled_shares = max(0.0, safe_float(item.get("actual_filled_shares"), 0.0) or 0.0)
            actual_filled_notional = max(0.0, safe_float(item.get("actual_filled_notional_usd"), 0.0) or 0.0)
            actual_fill_price = safe_float(item.get("actual_fill_price"), None)
            leg_row_id = str(item.get("leg_row_id") or "").strip()
            leg_market_id = str(item.get("market_id") or leg_payload.get("market_id") or getattr(signal, "market_id", "") or "").strip()
            leg_market_question = (
                str(item.get("market_question") or "").strip()
                or str(leg_payload.get("market_question") or getattr(signal, "market_question", "") or "").strip()
            )
            leg_direction = str(item.get("direction") or "").strip().lower() or _resolve_leg_direction(
                leg_payload,
                str(getattr(signal, "direction", "") or ""),
            )
            leg_entry_price = safe_float(
                item.get("entry_price"),
                safe_float(leg_payload.get("limit_price"), safe_float(result.effective_price, None)),
            )

            take_profit_pct = safe_float(exit_config.get("take_profit_pct"), None)
            entry_side = str(leg_payload.get("side") or "buy").strip().lower()
            if (
                mode == "live"
                and normalized_status == "executed"
                and entry_side == "buy"
                and preplace_take_profit_exit
                and take_profit_pct is not None
                and take_profit_pct > 0.0
            ):
                execution_payload = dict(result.payload or {})
                entry_fill_price = safe_float(result.effective_price, None)
                if entry_fill_price is None or entry_fill_price <= 0.0:
                    entry_fill_price = safe_float(execution_payload.get("average_fill_price"), None)
                exit_size = safe_float(execution_payload.get("filled_size"), None)
                if exit_size is None or exit_size <= 0.0:
                    exit_size = safe_float(result.shares, None)
                token_id = str(
                    execution_payload.get("token_id")
                    or (execution_payload.get("leg") or {}).get("token_id")
                    or leg_payload.get("token_id")
                    or ""
                ).strip()
                if (
                    token_id
                    and entry_fill_price is not None
                    and entry_fill_price > 0.0
                    and exit_size is not None
                    and exit_size > 0.0
                ):
                    target_price = min(0.999, max(0.001, entry_fill_price * (1.0 + (take_profit_pct / 100.0))))
                    async with release_conn(self.db):
                        try:
                            await live_execution_service.prepare_sell_balance_allowance(token_id)
                        except Exception:
                            pass
                        tp_submit = await execute_live_order(
                            token_id=token_id,
                            side="SELL",
                            size=float(exit_size),
                            fallback_price=target_price,
                            market_question=leg_market_question,
                            opportunity_id=str(getattr(signal, "id", "") or ""),
                            time_in_force="GTC",
                            post_only=True,
                            resolve_live_price=False,
                        )
                    bracket_timestamp = _iso_utc(utcnow())
                    pending_exit: dict[str, Any] = {
                        "kind": "take_profit_limit",
                        "triggered_at": bracket_timestamp,
                        "last_attempt_at": bracket_timestamp,
                        "close_trigger": "take_profit_limit",
                        "close_price": float(target_price),
                        "price_source": "entry_bracket_limit",
                        "target_take_profit_pct": float(take_profit_pct),
                        "entry_fill_price": float(entry_fill_price),
                        "market_tradable": True,
                        "exit_size": float(exit_size),
                        "post_only": True,
                        "retry_count": 0,
                        "reason": "entry_bracket_take_profit",
                    }
                    if tp_submit.status in {"executed", "open", "submitted"}:
                        pending_exit["status"] = "submitted"
                        pending_exit["exit_order_id"] = str(tp_submit.order_id or "")
                        pending_exit["provider_clob_order_id"] = str(tp_submit.payload.get("clob_order_id") or "")
                        pending_exit["provider_status"] = str(tp_submit.payload.get("trading_status") or "")
                    else:
                        pending_exit["status"] = "failed"
                        pending_exit["retry_count"] = 1
                        pending_exit["last_error"] = str(tp_submit.error_message or "failed_to_place_take_profit_exit")
                    order_payload["pending_live_exit"] = pending_exit

            bundle_recovery = item.get("bundle_recovery")
            if isinstance(bundle_recovery, dict):
                order_payload["bundle_recovery"] = dict(bundle_recovery)

            persisted_order_status = (
                "open" if mode in {"live", "shadow"} and normalized_status == "executed" else normalized_status
            )
            explicit_persisted_status = str(item.get("persisted_order_status") or "").strip().lower()
            if explicit_persisted_status:
                persisted_order_status = explicit_persisted_status
            persisted_at = utcnow()
            existing_trader_order = item.get("existing_trader_order")
            existing_execution_order = item.get("existing_execution_order")
            if isinstance(existing_trader_order, TraderOrder) and isinstance(existing_execution_order, ExecutionSessionOrder):
                persisted_payload = dict(existing_trader_order.payload_json or {})
                persisted_payload.update(order_payload)
                submission_intent = dict(persisted_payload.get("submission_intent") or {})
                if submission_intent:
                    if normalized_status in {"executed", "open", "submitted"}:
                        submission_intent["state"] = "provider_acknowledged"
                    elif normalized_status == "skipped":
                        submission_intent["state"] = "submit_skipped"
                    else:
                        submission_intent["state"] = "submit_failed"
                    submission_intent["last_submit_result_at"] = _iso_utc(persisted_at)
                    submission_intent["submit_status"] = normalized_status or None
                    if result.provider_order_id:
                        submission_intent["provider_order_id"] = str(result.provider_order_id)
                    if result.provider_clob_order_id:
                        submission_intent["provider_clob_order_id"] = str(result.provider_clob_order_id)
                    if actual_filled_shares > 0.0:
                        submission_intent["filled_shares"] = float(actual_filled_shares)
                    if actual_filled_notional > 0.0:
                        submission_intent["filled_notional_usd"] = float(actual_filled_notional)
                    if actual_fill_price is not None and actual_fill_price > 0.0:
                        submission_intent["effective_price"] = float(actual_fill_price)
                    if result.error_message:
                        submission_intent["error_message"] = str(result.error_message)
                    else:
                        submission_intent.pop("error_message", None)
                    persisted_payload["submission_intent"] = submission_intent
                if result.provider_order_id:
                    persisted_payload["provider_order_id"] = str(result.provider_order_id)
                if result.provider_clob_order_id:
                    persisted_payload["provider_clob_order_id"] = str(result.provider_clob_order_id)
                execution_session_payload = dict(persisted_payload.get("execution_session") or {})
                if result.provider_order_id:
                    execution_session_payload["provider_order_id"] = str(result.provider_order_id)
                if result.provider_clob_order_id:
                    execution_session_payload["provider_clob_order_id"] = str(result.provider_clob_order_id)
                if execution_session_payload:
                    persisted_payload["execution_session"] = execution_session_payload
                persisted_payload = _sync_order_runtime_payload(
                    payload=persisted_payload,
                    status=persisted_order_status,
                    now=persisted_at,
                    provider_snapshot_status=normalized_status or None,
                    mapped_status=normalized_status or persisted_order_status,
                )
                existing_trader_order.status = persisted_order_status
                existing_trader_order.notional_usd = (
                    float(actual_filled_notional)
                    if actual_filled_notional > 0.0
                    else (0.0 if persisted_order_status in {"failed", "cancelled", "rejected", "error", "skipped"} else None)
                )
                if actual_fill_price is not None and actual_fill_price > 0.0:
                    existing_trader_order.effective_price = float(actual_fill_price)
                existing_trader_order.reason = reason
                existing_trader_order.payload_json = persisted_payload
                existing_trader_order.error_message = result.error_message
                if leg_market_id:
                    existing_trader_order.market_id = leg_market_id
                if leg_market_question:
                    existing_trader_order.market_question = leg_market_question
                if leg_direction:
                    existing_trader_order.direction = leg_direction
                if leg_entry_price is not None and leg_entry_price > 0.0 and safe_float(existing_trader_order.entry_price, 0.0) <= 0.0:
                    existing_trader_order.entry_price = float(leg_entry_price)
                if actual_filled_notional > 0.0 and existing_trader_order.executed_at is None:
                    existing_trader_order.executed_at = persisted_at
                verification_fields = derive_trader_order_verification(
                    mode=mode,
                    status=persisted_order_status,
                    reason=reason,
                    payload=persisted_payload,
                )
                apply_trader_order_verification(
                    existing_trader_order,
                    verification_status=str(verification_fields.get("verification_status") or ""),
                    verification_source=str(verification_fields.get("verification_source") or "").strip() or None,
                    verification_reason=str(verification_fields.get("verification_reason") or "").strip() or None,
                    provider_order_id=str(verification_fields.get("provider_order_id") or "").strip() or None,
                    provider_clob_order_id=str(verification_fields.get("provider_clob_order_id") or "").strip() or None,
                    execution_wallet_address=str(verification_fields.get("execution_wallet_address") or "").strip() or None,
                    verification_tx_hash=str(verification_fields.get("verification_tx_hash") or "").strip() or None,
                    verified_at=persisted_at,
                    force=True,
                )
                existing_trader_order.updated_at = persisted_at
                trader_order = existing_trader_order
                if normalized_status != "skipped":
                    orders_written += 1
                    created_order_records.append(
                        {
                            "order_id": trader_order.id,
                            "market_id": str(trader_order.market_id or ""),
                            "direction": str(trader_order.direction or ""),
                            "source": str(getattr(signal, "source", "") or ""),
                            "notional_usd": actual_filled_notional,
                            "entry_price": safe_float(trader_order.entry_price, actual_fill_price),
                            "token_id": str(leg_payload.get("token_id") or ""),
                            "filled_shares": actual_filled_shares,
                            "status": persisted_order_status,
                            "payload": persisted_payload,
                        }
                    )
                execution_payload = dict(existing_execution_order.payload_json or {})
                execution_payload.update(dict(result.payload or {}))
                if submission_intent:
                    execution_payload["submission_intent"] = dict(submission_intent)
                existing_execution_order.provider_order_id = (
                    str(result.provider_order_id or "").strip() or existing_execution_order.provider_order_id
                )
                existing_execution_order.provider_clob_order_id = (
                    str(result.provider_clob_order_id or "").strip() or existing_execution_order.provider_clob_order_id
                )
                existing_execution_order.action = str(item.get("session_order_action") or existing_execution_order.action or "submit")
                existing_execution_order.side = str(leg_payload.get("side") or existing_execution_order.side or "buy")
                if actual_fill_price is not None and actual_fill_price > 0.0:
                    existing_execution_order.price = float(actual_fill_price)
                elif (
                    leg_entry_price is not None
                    and leg_entry_price > 0.0
                    and safe_float(existing_execution_order.price, 0.0) <= 0.0
                ):
                    existing_execution_order.price = float(leg_entry_price)
                existing_execution_order.size = float(actual_filled_shares) if actual_filled_shares > 0.0 else None
                existing_execution_order.notional_usd = (
                    float(actual_filled_notional) if actual_filled_notional > 0.0 else None
                )
                existing_execution_order.status = normalized_status
                existing_execution_order.reason = reason
                existing_execution_order.payload_json = execution_payload
                existing_execution_order.error_message = result.error_message
                existing_execution_order.updated_at = persisted_at
            else:
                trader_order = build_trader_order_row(
                    trader_id=trader_id,
                    signal=signal,
                    decision_id=decision_id,
                    strategy_key=strategy_key,
                    strategy_version=strategy_version,
                    mode=mode,
                    status=persisted_order_status,
                    notional_usd=actual_filled_notional,
                    effective_price=actual_fill_price,
                    reason=reason,
                    payload=order_payload,
                    error_message=result.error_message,
                    market_id=leg_market_id,
                    market_question=leg_market_question,
                    direction=leg_direction,
                    entry_price=leg_entry_price,
                )
                trader_orders.append(trader_order)
                orders_written += 1
                created_order_records.append(
                    {
                        "order_id": trader_order.id,
                        "market_id": str(trader_order.market_id or ""),
                        "direction": str(trader_order.direction or ""),
                        "source": str(getattr(signal, "source", "") or ""),
                        "notional_usd": actual_filled_notional,
                        "entry_price": safe_float(trader_order.entry_price, actual_fill_price),
                        "token_id": str(leg_payload.get("token_id") or ""),
                        "filled_shares": actual_filled_shares,
                        "status": persisted_order_status,
                        "payload": order_payload,
                    }
                )

                execution_orders.append(
                    ExecutionSessionOrder(
                        id=_new_id(),
                        session_id=session_row.id,
                        leg_id=leg_row_id,
                        trader_order_id=trader_order.id,
                        provider_order_id=result.provider_order_id,
                        provider_clob_order_id=result.provider_clob_order_id,
                        action=str(item.get("session_order_action") or "submit"),
                        side=str(leg_payload.get("side") or "buy"),
                        price=actual_fill_price,
                        size=actual_filled_shares if actual_filled_shares > 0.0 else None,
                        notional_usd=actual_filled_notional if actual_filled_notional > 0.0 else None,
                        status=normalized_status,
                        reason=reason,
                        payload_json=dict(result.payload or {}),
                        error_message=result.error_message,
                        created_at=persisted_at,
                        updated_at=persisted_at,
                    )
                )

        final_status = "completed"
        signal_status = "executed"
        error_message = None
        hedging_timeout_seconds: int | None = None
        hedging_started_at: datetime | None = None
        hedging_deadline_at: datetime | None = None
        if bundle_recovery_outcome is not None:
            residual_exposure = bool(bundle_recovery_outcome.get("residual_exposure"))
            recovered_to_flat = bool(bundle_recovery_outcome.get("recovered_to_flat"))
            completion_rescue_succeeded = bool(bundle_recovery_outcome.get("completion_rescue_succeeded"))
            if residual_exposure or recovered_to_flat:
                final_status = "failed"
                signal_status = "executed" if residual_exposure else "failed"
                error_message = (
                    "Partial fill: attempted bundle recovery, but residual exposure remains."
                    if residual_exposure
                    else "Partial fill: bundle recovery flattened filled legs after incomplete entry."
                )
            elif completion_rescue_succeeded:
                bundle_recovery_outcome = dict(bundle_recovery_outcome)
                bundle_recovery_outcome["recovered_to_flat"] = False
        elif failed_legs == 0 and completed_legs == 0 and open_legs == 0 and skipped_legs > 0:
            final_status = "skipped"
            signal_status = "skipped"
            error_message = skip_reasons[0] if skip_reasons else "Execution skipped before order submission."
        elif failed_legs > 0 and completed_legs == 0 and open_legs == 0:
            final_status = "failed"
            signal_status = "failed"
            error_message = "All execution legs failed."
        elif failed_legs > 0 and requires_pair_lock(plan["policy"], constraints):
            final_status = "failed"
            signal_status = "failed"
            error_message = "Full-bundle execution failed: not all legs completed."
        elif failed_legs > 0:
            final_status = "hedging"
            signal_status = "submitted"
            error_message = "Partial fill: session in hedging state."
            hedging_timeout_seconds = max(1, safe_int(constraints.get("hedge_timeout_seconds"), 20))
            hedging_started_at = utcnow()
            hedging_deadline_at = hedging_started_at + timedelta(seconds=hedging_timeout_seconds)
        elif open_legs > 0:
            final_status = "working"
            signal_status = "submitted"

        effective_price = self._effective_price_from_leg_results(leg_execution_records)
        session_payload_patch: dict[str, Any] = {"orders_written": orders_written}
        if skipped_legs > 0:
            session_payload_patch["skipped_legs"] = skipped_legs
            if skip_reasons:
                session_payload_patch["skip_reasons"] = skip_reasons[:5]
        if bundle_recovery_outcome is not None:
            session_payload_patch["bundle_recovery"] = bundle_recovery_outcome
        if final_status == "hedging" and hedging_started_at is not None and hedging_deadline_at is not None:
            session_payload_patch.update(
                {
                    "hedge_timeout_seconds": hedging_timeout_seconds,
                    "hedging_started_at": _iso_utc(hedging_started_at),
                    "hedging_deadline_at": _iso_utc(hedging_deadline_at),
                    "hedging_escalation": "auto_fail_on_timeout",
                }
            )
        _update_session_status(
            status=final_status,
            error_message=error_message,
            payload_patch=session_payload_patch,
        )
        event_type = "session_completed"
        event_severity = "info"
        event_message = f"Execution session ended with status={final_status}"
        if final_status in {"working", "hedging"}:
            event_type = "session_progress"
            event_severity = "info"
            event_message = f"Execution session remains {final_status}; awaiting leg completion"
        elif final_status == "failed":
            event_type = "session_failed"
            event_severity = "error"
            event_message = error_message or "Execution session failed."
        elif final_status == "skipped":
            event_type = "session_skipped"
            event_severity = "info"
            event_message = error_message or "Execution session skipped."
        _append_event(
            event_type=event_type,
            severity=event_severity,
            message=event_message,
            payload={
                "failed_legs": failed_legs,
                "completed_legs": completed_legs,
                "open_legs": open_legs,
                "skipped_legs": skipped_legs,
                "hedge_timeout_seconds": hedging_timeout_seconds,
                "hedging_deadline_at": (_iso_utc(hedging_deadline_at) if hedging_deadline_at is not None else None),
            },
        )
        await _persist_execution_projection_safely(signal_status=signal_status, effective_price=effective_price)

        return SessionExecutionResult(
            session_id=session_row.id,
            status=final_status,
            effective_price=effective_price,
            error_message=error_message,
            orders_written=orders_written,
            payload=_payload_with_execution_timing({
                "execution_plan": plan,
                "legs": leg_execution_records,
                "submit_breakdowns": list(_submit_wave_breakdowns),
            }),
            created_orders=created_order_records,
        )

    async def reconcile_active_sessions(
        self,
        *,
        mode: str,
        trader_id: str | None = None,
    ) -> dict[str, int]:
        sessions = await list_active_execution_sessions(
            self.db,
            trader_id=trader_id,
            mode=mode,
            limit=1000,
        )
        now = utcnow()
        expired = 0
        completed = 0
        cancelled = 0
        session_ids = [str(row.id) for row in sessions if str(row.id or "").strip()]
        leg_rollups = await get_execution_session_leg_rollups(self.db, session_ids)
        provider_reads_unavailable = mode == "live" and live_execution_service.clob_read_circuit_open()
        for row in sessions:
            row_id = str(row.id or "").strip()
            row_type = type(row)
            status_key = str(row.status or "").strip().lower()
            row_created_at = row.created_at
            row_updated_at = row.updated_at
            row_expires_at = row.expires_at
            row_payload = dict(row.payload_json or {}) if isinstance(row.payload_json, dict) else {}
            if not row_id:
                continue
            if status_key == "paused":
                continue
            if status_key == "placing":
                baseline_ts = row_updated_at or row_created_at or now
                if baseline_ts.tzinfo is None:
                    baseline_ts = baseline_ts.replace(tzinfo=timezone.utc)
                else:
                    baseline_ts = baseline_ts.astimezone(timezone.utc)
                placing_age_seconds = max(0.0, (now - baseline_ts).total_seconds())
                if placing_age_seconds >= _STALE_PRE_SUBMIT_SESSION_TIMEOUT_SECONDS:
                    resolved_trader_id = str(getattr(row, "trader_id", None) or trader_id or "").strip()
                    if (
                        mode == "live"
                        and resolved_trader_id
                        and hasattr(self.db, "execute")
                        and not provider_reads_unavailable
                        and placing_age_seconds <= _LIVE_PRE_SUBMIT_AUTHORITY_RECOVERY_MAX_AGE_SECONDS
                    ):
                        try:
                            await recover_missing_live_trader_orders(
                                self.db,
                                trader_ids=[resolved_trader_id],
                                commit=False,
                                broadcast=False,
                            )
                        except Exception:
                            pass
                        refreshed_row = None
                        if hasattr(self.db, "get"):
                            try:
                                refreshed_row = await self.db.get(row_type, row_id)
                            except Exception:
                                refreshed_row = None
                        refreshed_status = str(getattr(refreshed_row, "status", "") or "").strip().lower()
                        if refreshed_row is not None and refreshed_status != "placing":
                            if refreshed_status == "completed":
                                completed += 1
                            elif refreshed_status in {"cancelled", "failed", "expired", "skipped"}:
                                cancelled += 1
                            continue
                    timeout_reason = (
                        "Live order submission did not persist provider acknowledgement within "
                        f"{int(_STALE_PRE_SUBMIT_SESSION_TIMEOUT_SECONDS)}s."
                    )
                    await self.cancel_session(
                        session_id=row_id,
                        reason=timeout_reason,
                        terminal_status="failed",
                        skip_provider_io=(
                            provider_reads_unavailable
                            or placing_age_seconds > _LIVE_PRE_SUBMIT_AUTHORITY_RECOVERY_MAX_AGE_SECONDS
                        ),
                    )
                    cancelled += 1
                    continue
            if row_expires_at is not None and now > row_expires_at:
                expires_at = row_expires_at
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=timezone.utc)
                else:
                    expires_at = expires_at.astimezone(timezone.utc)
                await self.cancel_session(
                    session_id=row_id,
                    reason="Session timed out before all legs completed.",
                    terminal_status="expired",
                    skip_provider_io=(now - expires_at).total_seconds() > _STALE_SESSION_PROVIDER_IO_SKIP_SECONDS,
                )
                expired += 1
                continue

            if status_key == "hedging":
                session_payload = row_payload
                hedge_timeout_seconds = max(1, safe_int(session_payload.get("hedge_timeout_seconds"), 20))
                hedging_started_at = _parse_iso_utc(session_payload.get("hedging_started_at"))
                if hedging_started_at is None:
                    baseline_ts = row_updated_at or row_created_at or now
                    if baseline_ts.tzinfo is None:
                        hedging_started_at = baseline_ts.replace(tzinfo=timezone.utc)
                    else:
                        hedging_started_at = baseline_ts.astimezone(timezone.utc)
                hedging_deadline_at = _parse_iso_utc(session_payload.get("hedging_deadline_at"))
                if hedging_deadline_at is None:
                    hedging_deadline_at = hedging_started_at + timedelta(seconds=hedge_timeout_seconds)
                if now >= hedging_deadline_at:
                    timeout_reason = f"Hedge timeout exceeded ({hedge_timeout_seconds}s); escalating session to failed."
                    await create_execution_session_event(
                        self.db,
                        session_id=row_id,
                        event_type="hedge_timeout",
                        severity="error",
                        message=timeout_reason,
                        payload={
                            "hedge_timeout_seconds": hedge_timeout_seconds,
                            "hedging_started_at": _iso_utc(hedging_started_at),
                            "hedging_deadline_at": _iso_utc(hedging_deadline_at),
                        },
                        commit=False,
                    )
                    await self.cancel_session(
                        session_id=row_id,
                        reason=timeout_reason,
                        terminal_status="failed",
                    )
                    cancelled += 1
                    continue

            leg_rollup = leg_rollups.get(row_id, {})
            legs_total = safe_int(leg_rollup.get("legs_total"), 0)
            legs_completed = safe_int(leg_rollup.get("legs_completed"), 0)
            legs_failed = safe_int(leg_rollup.get("legs_failed"), 0)
            legs_open = safe_int(leg_rollup.get("legs_open"), 0)
            if legs_total > 0 and legs_completed >= legs_total:
                await update_execution_session_status(
                    self.db,
                    session_id=row_id,
                    status="completed",
                    commit=False,
                )
                completed += 1
            elif legs_failed > 0 and legs_open == 0 and legs_completed == 0:
                await update_execution_session_status(
                    self.db,
                    session_id=row_id,
                    status="failed",
                    error_message="All execution legs failed.",
                    commit=False,
                )
                cancelled += 1
        if expired > 0 or completed > 0 or cancelled > 0:
            await self.db.commit()
        return {
            "active_seen": len(sessions),
            "expired": expired,
            "completed": completed,
            "failed": cancelled,
        }

    async def cancel_session(
        self,
        *,
        session_id: str,
        reason: str,
        terminal_status: str = "cancelled",
        skip_provider_io: bool = False,
    ) -> bool:
        detail = await get_execution_session_detail(self.db, session_id)
        if detail is None:
            return False
        session_view = detail["session"]
        terminal_key = str(terminal_status or "cancelled").strip().lower()
        if terminal_key not in {"cancelled", "expired", "failed"}:
            terminal_key = "cancelled"
        if str(session_view.get("status") or "").strip().lower() in {"cancelled", "completed", "failed", "expired"}:
            return True

        now = utcnow()

        # Batch-fetch live provider snapshots so we have up-to-date fill data
        # before deciding whether each order was filled or not.  Without this,
        # a session timeout can race with provider reconciliation and treat a
        # 99%-filled order as unfilled/cancelled.
        clob_ids_to_fetch: set[str] = set()
        for order in detail.get("orders", []):
            clob_id = str(order.get("provider_clob_order_id") or "").strip()
            if clob_id:
                clob_ids_to_fetch.add(clob_id)
        provider_snapshots: dict[str, dict[str, Any]] = {}
        if clob_ids_to_fetch and not skip_provider_io:
            async with release_conn(self.db):
                try:
                    provider_snapshots = await asyncio.wait_for(
                        live_execution_service.get_order_snapshots_by_clob_ids(sorted(clob_ids_to_fetch)),
                        timeout=_PROVIDER_SNAPSHOT_TIMEOUT_SECONDS,
                    )
                except (asyncio.TimeoutError, Exception):
                    provider_snapshots = {}

        trader_orders_by_id: dict[str, TraderOrder] = {}
        trader_order_ids = sorted(
            {
                str(order.get("trader_order_id") or "").strip()
                for order in detail.get("orders", [])
                if str(order.get("status") or "").strip().lower() in {"open", "submitted", "placing", "cancelled", "failed"}
                and str(order.get("trader_order_id") or "").strip()
            }
        )
        if trader_order_ids:
            trader_order_rows = (
                await self.db.execute(
                    select(TraderOrder).where(TraderOrder.id.in_(trader_order_ids))
                )
            ).scalars().all()
            trader_orders_by_id = {str(row.id): row for row in trader_order_rows}

        updated_trader_orders: list[TraderOrder] = []
        filled_leg_ids: set[str] = set()
        total_orders_processed = 0
        total_orders_filled = 0
        for order in detail.get("orders", []):
            status_key = str(order.get("status") or "").strip().lower()
            # Also process session_orders already marked cancelled/failed so we can sync
            # a still-placing trader_order to its terminal state — otherwise the trader_order
            # gets stranded when a prior cleanup (e.g. authority recovery) marked the
            # session_order terminal but did not write the trader_order row.
            if status_key not in {"open", "submitted", "placing", "cancelled", "failed"}:
                continue
            trader_order_id = str(order.get("trader_order_id") or "").strip()
            trader_order = trader_orders_by_id.get(trader_order_id) if trader_order_id else None
            if trader_order is None:
                continue
            local_status_key = str(trader_order.status or "").strip().lower()
            if local_status_key not in {"open", "submitted", "placing", "executed"}:
                continue

            provider_order_id = str(order.get("provider_order_id") or "").strip()
            provider_clob_order_id = str(order.get("provider_clob_order_id") or "").strip()
            cancel_targets: list[str] = []
            if provider_clob_order_id:
                cancel_targets.append(provider_clob_order_id)
            if provider_order_id and provider_order_id not in cancel_targets:
                cancel_targets.append(provider_order_id)
            cancel_success = False
            if not skip_provider_io:
                for target in cancel_targets:
                    async with release_conn(self.db):
                        try:
                            cancel_success = await asyncio.wait_for(
                                cancel_live_provider_order(target),
                                timeout=_PROVIDER_CANCEL_TIMEOUT_SECONDS,
                            )
                        except asyncio.TimeoutError:
                            cancel_success = False
                    if cancel_success:
                        cancel_success = True
                        break

            if trader_order is None:
                continue
            total_orders_processed += 1
            execution_order_row = None
            execution_order_id = str(order.get("id") or "").strip()
            if execution_order_id and hasattr(self.db, "get"):
                try:
                    execution_order_row = await self.db.get(ExecutionSessionOrder, execution_order_id)
                except Exception:
                    execution_order_row = None
            payload = dict(trader_order.payload_json or {})
            submission_intent = payload.get("submission_intent")
            submission_intent = dict(submission_intent) if isinstance(submission_intent, dict) else {}
            is_pre_submit_placeholder = status_key == "placing" or str(trader_order.status or "").strip().lower() == "placing"

            # Merge live provider snapshot into payload so _extract_live_fill_metrics
            # sees the latest fill data even if reconciliation hasn't written it yet.
            snapshot = provider_snapshots.get(provider_clob_order_id) if provider_clob_order_id else None
            if isinstance(snapshot, dict):
                existing_recon = payload.get("provider_reconciliation")
                if not isinstance(existing_recon, dict):
                    existing_recon = {}
                existing_recon = dict(existing_recon)
                existing_recon["snapshot"] = snapshot
                filled_size_snap = max(0.0, safe_float(snapshot.get("filled_size"), 0.0) or 0.0)
                avg_price_snap = safe_float(snapshot.get("average_fill_price"))
                filled_notional_snap = max(0.0, safe_float(snapshot.get("filled_notional_usd"), 0.0) or 0.0)
                if filled_notional_snap <= 0.0 and filled_size_snap > 0.0:
                    ref_price = avg_price_snap if avg_price_snap and avg_price_snap > 0 else safe_float(snapshot.get("limit_price"))
                    if ref_price and ref_price > 0:
                        filled_notional_snap = filled_size_snap * ref_price
                if filled_size_snap > 0.0:
                    existing_recon["filled_size"] = filled_size_snap
                if avg_price_snap is not None and avg_price_snap > 0:
                    existing_recon["average_fill_price"] = avg_price_snap
                if filled_notional_snap > 0.0:
                    existing_recon["filled_notional_usd"] = filled_notional_snap
                payload["provider_reconciliation"] = existing_recon

            filled_notional_usd, filled_shares, avg_fill_price = _extract_live_fill_metrics(payload)
            has_fill = filled_notional_usd > 0.0
            if has_fill:
                next_status = "executed"
                next_notional = max(0.0, filled_notional_usd)
                if next_notional <= 0.0:
                    next_notional = max(0.0, abs(safe_float(trader_order.notional_usd, 0.0) or 0.0))
                trader_order.status = next_status
                trader_order.notional_usd = float(next_notional)
                if avg_fill_price is not None and avg_fill_price > 0:
                    trader_order.effective_price = float(avg_fill_price)
                if trader_order.executed_at is None and next_notional > 0.0:
                    trader_order.executed_at = now
                total_orders_filled += 1
                leg_id = str(order.get("leg_id") or "").strip()
                if leg_id:
                    filled_leg_ids.add(leg_id)
            else:
                next_status = "failed" if terminal_key == "failed" or is_pre_submit_placeholder else "cancelled"
                trader_order.status = next_status
                trader_order.notional_usd = 0.0
                trader_order.error_message = reason
                if is_pre_submit_placeholder:
                    submission_intent["state"] = "submit_timeout"
                    submission_intent["submit_status"] = "timed_out"
                    submission_intent["error_message"] = reason
                    submission_intent["last_submit_result_at"] = _iso_utc(now)
                    payload["submission_intent"] = submission_intent

            payload["session_cancel"] = {
                "cancelled_at": _iso_utc(now),
                "session_id": session_id,
                "terminal_status": terminal_key,
                "reason": reason,
                "provider_cancel_targets": cancel_targets,
                "provider_cancel_success": bool(cancel_success),
            }
            trader_order.payload_json = _sync_order_runtime_payload(
                payload=payload,
                status=next_status,
                now=now,
                provider_snapshot_status=next_status if next_status in {"cancelled", "failed"} else None,
                mapped_status="failed" if next_status == "failed" else "cancelled" if next_status == "cancelled" else "executed",
            )
            trader_order.updated_at = now
            if reason:
                if trader_order.reason:
                    trader_order.reason = f"{trader_order.reason} | session:{terminal_key}:{reason}"
                else:
                    trader_order.reason = f"session:{terminal_key}:{reason}"
            if isinstance(execution_order_row, ExecutionSessionOrder):
                execution_payload = dict(execution_order_row.payload_json or {})
                if submission_intent:
                    execution_payload["submission_intent"] = dict(submission_intent)
                execution_payload["session_cancel"] = dict(payload["session_cancel"])
                execution_order_row.status = next_status
                execution_order_row.error_message = None if has_fill else reason
                if has_fill:
                    if filled_shares > 0.0:
                        execution_order_row.size = float(filled_shares)
                    if filled_notional_usd > 0.0:
                        execution_order_row.notional_usd = float(filled_notional_usd)
                    if avg_fill_price is not None and avg_fill_price > 0.0:
                        execution_order_row.price = float(avg_fill_price)
                execution_order_row.payload_json = execution_payload
                execution_order_row.updated_at = now
            if next_status in {"cancelled", "failed"}:
                hot_state.record_order_cancelled(
                    trader_id=str(trader_order.trader_id or ""),
                    mode=str(trader_order.mode or ""),
                    order_id=str(trader_order.id or ""),
                    market_id=str(trader_order.market_id or ""),
                    source=str(trader_order.source or ""),
                    copy_source_wallet=_extract_copy_source_wallet_from_payload(payload),
                )
            updated_trader_orders.append(trader_order)

        any_filled = total_orders_filled > 0
        all_filled = total_orders_processed > 0 and total_orders_filled >= total_orders_processed

        for leg in detail.get("legs", []):
            leg_status = str(leg.get("status") or "").strip().lower()
            if leg_status in {"completed", "failed", "cancelled", "expired"}:
                continue
            leg_id = str(leg.get("id") or "").strip()
            if leg_id in filled_leg_ids:
                await update_execution_leg(
                    self.db,
                    leg_row_id=leg_id,
                    status="completed",
                    commit=False,
                )
            else:
                await update_execution_leg(
                    self.db,
                    leg_row_id=leg_id,
                    status="failed" if terminal_key == "failed" else "cancelled",
                    last_error=reason,
                    commit=False,
                )

        effective_session_status = "completed" if all_filled else terminal_key
        await update_execution_session_status(
            self.db,
            session_id=session_id,
            status=effective_session_status,
            error_message=None if all_filled else reason,
            commit=False,
        )
        # Any fill (even partial) means we hold a real position that must be
        # managed by the position lifecycle.  Mark the signal "executed" so the
        # downstream exit/P&L tracking picks it up.
        signal_id = str(session_view.get("signal_id") or "").strip()
        if signal_id:
            await set_trade_signal_status(
                self.db,
                signal_id=signal_id,
                status="executed" if any_filled else "failed",
                commit=False,
            )
        if all_filled:
            event_type = "session_completed"
            event_severity = "info"
        elif terminal_key == "expired":
            event_type = "session_expired"
            event_severity = "warn"
        elif terminal_key == "failed":
            event_type = "session_failed"
            event_severity = "error"
        else:
            event_type = "session_cancelled"
            event_severity = "warn"
        await create_execution_session_event(
            self.db,
            session_id=session_id,
            event_type=event_type,
            severity=event_severity,
            message=reason,
            commit=False,
        )
        await self.db.commit()

        # Trigger position inventory sync so filled orders are immediately
        # visible to the position lifecycle for exit management, rather than
        # waiting for the next worker loop iteration.
        if any_filled:
            trader_id = str(session_view.get("trader_id") or "").strip()
            if trader_id:
                try:
                    await sync_trader_position_inventory(
                        self.db,
                        trader_id=trader_id,
                        mode=str(session_view.get("mode") or ""),
                    )
                except Exception:
                    pass

        for row in updated_trader_orders:
            try:
                await event_bus.publish("trader_order", _serialize_order(row))
            except Exception:
                continue
        return True

    async def pause_session(self, *, session_id: str) -> bool:
        updated = await update_execution_session_status(
            self.db,
            session_id=session_id,
            status="paused",
            commit=False,
        )
        if updated is None:
            return False
        await create_execution_session_event(
            self.db,
            session_id=session_id,
            event_type="session_paused",
            message="Execution session paused",
            commit=False,
        )
        await self.db.commit()
        return True

    async def resume_session(self, *, session_id: str) -> bool:
        updated = await update_execution_session_status(
            self.db,
            session_id=session_id,
            status="working",
            commit=False,
        )
        if updated is None:
            return False
        await create_execution_session_event(
            self.db,
            session_id=session_id,
            event_type="session_resumed",
            message="Execution session resumed",
            commit=False,
        )
        await self.db.commit()
        return True

