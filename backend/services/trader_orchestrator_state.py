"""DB-backed state for trader orchestrator runtime and APIs."""

from __future__ import annotations

import uuid
import asyncio
import copy
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from config import settings
from sqlalchemy import and_, desc, func, or_, select, update as sa_update
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    AppSettings,
    ExecutionSession,
    ExecutionSessionEvent,
    ExecutionSessionLeg,
    ExecutionSessionOrder,
    TradeSignal,
    Strategy,
    Trader,
    TraderConfigRevision,
    TraderDecision,
    TraderDecisionCheck,
    TraderEvent,
    TraderOrder,
    TraderPosition,
    TraderOrchestratorControl,
    TraderOrchestratorSnapshot,
    TraderSignalCursor,
    TraderSignalConsumption,
    release_conn,
)
from utils.logger import get_logger
from services.event_bus import event_bus
from services.market_cache import CachedMarket
from services.redis_streams import redis_streams
from services.worker_state import (
    DB_RETRY_ATTEMPTS,
    _commit_with_retry,
    _is_retryable_db_error,
    _db_retry_delay,
    read_worker_snapshot,
)
from services.live_execution_adapter import execute_live_order
from services.live_execution_service import live_execution_service
from services.polymarket import polymarket_client
from services.trader_orchestrator.sources.registry import normalize_source_key
from services.opportunity_strategy_catalog import (
    build_system_opportunity_strategy_rows,
    list_system_strategy_keys,
)
from services.strategy_versioning import normalize_strategy_version, resolve_strategy_version
from services.strategy_sdk import StrategySDK
from services.strategies.news_edge import validate_news_edge_config
from services.strategies.traders_copy_trade import validate_traders_copy_trade_config
from services.trader_orchestrator.templates import (
    DEFAULT_GLOBAL_RISK,
    TRADER_TEMPLATES,
    get_template,
)
from utils.utcnow import utcnow
from utils.secrets import decrypt_secret
from utils.converters import safe_float, safe_int, to_iso
from utils.market_urls import build_polymarket_market_url

logger = get_logger(__name__)

ORCHESTRATOR_CONTROL_ID = "default"
ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS = 5
_UNSET = object()  # Sentinel: distinguish "not provided" from explicit None
ORCHESTRATOR_SNAPSHOT_ID = "latest"
OPEN_ORDER_STATUSES = {"submitted", "executed", "open"}
SHADOW_ACTIVE_ORDER_STATUSES = {"submitted", "executed", "open"}
LIVE_ACTIVE_ORDER_STATUSES = {"submitted", "executed", "open"}
CLEANUP_ELIGIBLE_ORDER_STATUSES = {"submitted", "executed", "open"}
PENDING_LIVE_EXIT_TERMINAL_STATUSES = {
    "filled",
    "superseded_resolution",
    "superseded_external",
    "cancelled",
}
DEFAULT_PENDING_LIVE_EXIT_GUARD = {
    "max_pending_exits": 0,
    "identity_guard_enabled": True,
    "terminal_statuses": sorted(PENDING_LIVE_EXIT_TERMINAL_STATUSES),
}
DEFAULT_LIVE_RISK_CLAMPS = {
    "enforce_allow_averaging_off": True,
    "min_cooldown_seconds": 90,
    "max_consecutive_losses_cap": 3,
    "max_open_orders_cap": 6,
    "max_open_positions_cap": 4,
    "max_trade_notional_usd_cap": 200.0,
    "max_orders_per_cycle_cap": 4,
    "enforce_halt_on_consecutive_losses": True,
}
DEFAULT_LIVE_MARKET_CONTEXT = {
    "enabled": True,
    "history_window_seconds": 7200,
    "history_fidelity_seconds": 300,
    "max_history_points": 120,
    "timeout_seconds": 4.0,
    "strict_ws_pricing_only": True,
    "allow_redis_strict_prices": True,
    "max_market_data_age_ms": 100,
}
DEFAULT_LIVE_PROVIDER_HEALTH = {
    "window_seconds": 180,
    "min_errors": 2,
    "block_seconds": 120,
}
LIVE_PROVIDER_CANCEL_TIMEOUT_SECONDS = 8.0
DEFAULT_TIMEOUT_TAKER_RESCUE_PRICE_BPS = 35.0
DEFAULT_TIMEOUT_TAKER_RESCUE_TIME_IN_FORCE = "IOC"
REALIZED_WIN_ORDER_STATUSES = {"resolved_win", "closed_win", "win"}
REALIZED_LOSS_ORDER_STATUSES = {"resolved_loss", "closed_loss", "loss"}
REALIZED_ORDER_STATUSES = REALIZED_WIN_ORDER_STATUSES | REALIZED_LOSS_ORDER_STATUSES
ACTIVE_POSITION_STATUS = "open"
INACTIVE_POSITION_STATUS = "closed"
ACTIVE_EXECUTION_SESSION_STATUSES = {"pending", "placing", "working", "partial", "hedging", "paused"}
TERMINAL_EXECUTION_SESSION_STATUSES = {"completed", "failed", "cancelled", "expired", "skipped"}
TRADER_ORDER_STATUS_HASH_KEY_PREFIX = "trader_order_status:"
TRADER_ORDER_UPDATES_STREAM = "trader_order_updates"
TRADER_ORDER_UPDATES_STREAM_MAXLEN = 200000
TRADER_ORDER_STATUS_TTL_SECONDS = 14 * 24 * 60 * 60
_TRADER_SCOPE_MODES = set(StrategySDK.TRADER_SCOPE_MODE_CANONICAL)
_ORCHESTRATOR_SNAPSHOT_STALE_MULTIPLIER = 30.0
_ORCHESTRATOR_SNAPSHOT_STALE_MIN_SECONDS = 120.0
_TRADER_MODES = {"shadow", "live"}
_MANUAL_LIVE_POSITION_SOURCE = "manual_wallet_position"
_LEGACY_CRYPTO_STRATEGY_TIMEFRAME_ALIASES: dict[str, str] = {
    "crypto_5m": "5m",
    "crypto_15m": "15m",
    "crypto_1h": "1h",
    "crypto_4h": "4h",
}
_LEGACY_STRATEGY_ALIASES: dict[str, str] = {
    **{legacy: "btc_eth_highfreq" for legacy in _LEGACY_CRYPTO_STRATEGY_TIMEFRAME_ALIASES.keys()},
    "news_reaction": "news_edge",
}


def _now() -> datetime:
    return utcnow()


def _new_id() -> str:
    return uuid.uuid4().hex


def _normalize_mode_key(mode: Any) -> str:
    key = str(mode or "").strip().lower()
    if key == "paper":
        return "shadow"
    if key in _TRADER_MODES:
        return key
    return "other"


def _normalize_status_key(status: Any) -> str:
    return str(status or "").strip().lower()


def _coerce_string_list(value: Any, fallback: list[str]) -> list[str]:
    if not isinstance(value, list):
        return list(fallback)
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out or list(fallback)


def _normalize_pending_live_exit_terminal_statuses(value: Any) -> list[str]:
    if not isinstance(value, list):
        return list(DEFAULT_PENDING_LIVE_EXIT_GUARD["terminal_statuses"])
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        status = _normalize_status_key(item)
        if not status or status in seen:
            continue
        seen.add(status)
        out.append(status)
    return out or list(DEFAULT_PENDING_LIVE_EXIT_GUARD["terminal_statuses"])


def _normalize_pending_live_exit_guard(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    return {
        "max_pending_exits": max(0, min(1000, safe_int(source.get("max_pending_exits"), 0))),
        "identity_guard_enabled": bool(
            source.get("identity_guard_enabled", DEFAULT_PENDING_LIVE_EXIT_GUARD["identity_guard_enabled"])
        ),
        "terminal_statuses": _normalize_pending_live_exit_terminal_statuses(source.get("terminal_statuses")),
    }


def _normalize_live_risk_clamps(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    return {
        "enforce_allow_averaging_off": bool(
            source.get("enforce_allow_averaging_off", DEFAULT_LIVE_RISK_CLAMPS["enforce_allow_averaging_off"])
        ),
        "min_cooldown_seconds": max(
            0,
            min(
                86400,
                safe_int(source.get("min_cooldown_seconds"), DEFAULT_LIVE_RISK_CLAMPS["min_cooldown_seconds"]),
            ),
        ),
        "max_consecutive_losses_cap": max(
            1,
            min(
                1000,
                safe_int(
                    source.get("max_consecutive_losses_cap"),
                    DEFAULT_LIVE_RISK_CLAMPS["max_consecutive_losses_cap"],
                ),
            ),
        ),
        "max_open_orders_cap": max(
            1,
            min(1000, safe_int(source.get("max_open_orders_cap"), DEFAULT_LIVE_RISK_CLAMPS["max_open_orders_cap"])),
        ),
        "max_open_positions_cap": max(
            1,
            min(
                1000,
                safe_int(source.get("max_open_positions_cap"), DEFAULT_LIVE_RISK_CLAMPS["max_open_positions_cap"]),
            ),
        ),
        "max_trade_notional_usd_cap": max(
            1.0,
            min(
                1_000_000.0,
                safe_float(
                    source.get("max_trade_notional_usd_cap"),
                    DEFAULT_LIVE_RISK_CLAMPS["max_trade_notional_usd_cap"],
                ),
            ),
        ),
        "max_orders_per_cycle_cap": max(
            1,
            min(
                1000,
                safe_int(source.get("max_orders_per_cycle_cap"), DEFAULT_LIVE_RISK_CLAMPS["max_orders_per_cycle_cap"]),
            ),
        ),
        "enforce_halt_on_consecutive_losses": bool(
            source.get(
                "enforce_halt_on_consecutive_losses",
                DEFAULT_LIVE_RISK_CLAMPS["enforce_halt_on_consecutive_losses"],
            )
        ),
    }


def _normalize_live_market_context(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    return {
        "enabled": bool(source.get("enabled", DEFAULT_LIVE_MARKET_CONTEXT["enabled"])),
        "history_window_seconds": max(
            300,
            min(
                21600,
                safe_int(source.get("history_window_seconds"), DEFAULT_LIVE_MARKET_CONTEXT["history_window_seconds"]),
            ),
        ),
        "history_fidelity_seconds": max(
            30,
            min(
                1800,
                safe_int(
                    source.get("history_fidelity_seconds"),
                    DEFAULT_LIVE_MARKET_CONTEXT["history_fidelity_seconds"],
                ),
            ),
        ),
        "max_history_points": max(
            20,
            min(240, safe_int(source.get("max_history_points"), DEFAULT_LIVE_MARKET_CONTEXT["max_history_points"])),
        ),
        "timeout_seconds": max(
            1.0,
            min(12.0, safe_float(source.get("timeout_seconds"), DEFAULT_LIVE_MARKET_CONTEXT["timeout_seconds"]) or 4.0),
        ),
        "strict_ws_pricing_only": bool(
            source.get("strict_ws_pricing_only", DEFAULT_LIVE_MARKET_CONTEXT["strict_ws_pricing_only"])
        ),
        "allow_redis_strict_prices": bool(
            source.get("allow_redis_strict_prices", DEFAULT_LIVE_MARKET_CONTEXT["allow_redis_strict_prices"])
        ),
        "max_market_data_age_ms": max(
            25,
            min(
                10_000,
                safe_int(source.get("max_market_data_age_ms"), DEFAULT_LIVE_MARKET_CONTEXT["max_market_data_age_ms"]),
            ),
        ),
    }


def _normalize_live_provider_health(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    return {
        "window_seconds": max(
            30,
            min(
                900,
                safe_int(source.get("window_seconds"), DEFAULT_LIVE_PROVIDER_HEALTH["window_seconds"]),
            ),
        ),
        "min_errors": max(1, min(20, safe_int(source.get("min_errors"), DEFAULT_LIVE_PROVIDER_HEALTH["min_errors"]))),
        "block_seconds": max(
            15,
            min(3600, safe_int(source.get("block_seconds"), DEFAULT_LIVE_PROVIDER_HEALTH["block_seconds"])),
        ),
    }


def _normalize_global_runtime_settings(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    timeout_override = source.get("trader_cycle_timeout_seconds")
    timeout_value = safe_float(timeout_override, 0.0)
    trader_cycle_timeout_seconds = None if timeout_value <= 0 else max(3.0, min(120.0, float(timeout_value)))
    return {
        "pending_live_exit_guard": _normalize_pending_live_exit_guard(source.get("pending_live_exit_guard")),
        "live_risk_clamps": _normalize_live_risk_clamps(source.get("live_risk_clamps")),
        "live_market_context": _normalize_live_market_context(source.get("live_market_context")),
        "live_provider_health": _normalize_live_provider_health(source.get("live_provider_health")),
        "trader_cycle_timeout_seconds": trader_cycle_timeout_seconds,
    }


def _normalize_global_risk(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    return {
        "max_gross_exposure_usd": max(
            1.0,
            min(
                1_000_000.0,
                safe_float(source.get("max_gross_exposure_usd"), DEFAULT_GLOBAL_RISK["max_gross_exposure_usd"]),
            ),
        ),
        "max_daily_loss_usd": max(
            0.0,
            min(1_000_000.0, safe_float(source.get("max_daily_loss_usd"), DEFAULT_GLOBAL_RISK["max_daily_loss_usd"])),
        ),
        "max_orders_per_cycle": max(
            1,
            min(1000, safe_int(source.get("max_orders_per_cycle"), DEFAULT_GLOBAL_RISK["max_orders_per_cycle"])),
        ),
    }


def _normalize_control_settings(value: Any) -> dict[str, Any]:
    settings = dict(value) if isinstance(value, dict) else {}
    removed_keys = {
        "paper_account_id",
        "enable_live_market_context",
        "live_market_history_window_seconds",
        "live_market_history_fidelity_seconds",
        "live_market_history_max_points",
        "live_market_context_timeout_seconds",
        "trader_cycle_timeout_seconds",
    }
    normalized_keys = {
        "global_risk",
        "global_runtime",
        "trading_domains",
        "enabled_strategies",
        "llm_verify_trades",
    }

    normalized = {
        "global_risk": _normalize_global_risk(settings.get("global_risk")),
        "global_runtime": _normalize_global_runtime_settings(settings.get("global_runtime")),
        "trading_domains": _coerce_string_list(settings.get("trading_domains"), ["event_markets", "crypto"]),
        "enabled_strategies": _coerce_string_list(settings.get("enabled_strategies"), list_system_strategy_keys()),
        "llm_verify_trades": bool(settings.get("llm_verify_trades", False)),
    }
    for key, raw_value in settings.items():
        if key in normalized_keys or key in removed_keys:
            continue
        normalized[key] = raw_value
    return normalized


def _normalize_condition_id(value: Any) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 66 or not text.startswith("0x"):
        return ""
    body = text[2:]
    if not body:
        return ""
    for char in body:
        if char not in "0123456789abcdef":
            return ""
    return text


def _extract_order_condition_id(row: TraderOrder) -> str:
    payload = row.payload_json if isinstance(row.payload_json, dict) else {}
    return (
        _normalize_condition_id(payload.get("condition_id"))
        or _normalize_condition_id(payload.get("conditionId"))
        or _normalize_condition_id(payload.get("market_id"))
        or _normalize_condition_id(payload.get("marketId"))
        or _normalize_condition_id(row.market_id)
    )


def _extract_order_token_id(row: TraderOrder) -> str:
    payload = dict(row.payload_json or {})
    token_id = str(payload.get("token_id") or payload.get("selected_token_id") or "").strip()
    if token_id:
        return token_id
    provider_reconciliation = payload.get("provider_reconciliation")
    if isinstance(provider_reconciliation, dict):
        snapshot = provider_reconciliation.get("snapshot")
        if isinstance(snapshot, dict):
            token_id = str(snapshot.get("asset_id") or snapshot.get("asset") or snapshot.get("token_id") or "").strip()
            if token_id:
                return token_id
    return ""


def _extract_copy_source_wallet_from_payload(payload: Any) -> str:
    data = payload if isinstance(payload, dict) else {}
    copy_attribution = data.get("copy_attribution")
    copy_attribution = copy_attribution if isinstance(copy_attribution, dict) else {}
    source_trade = data.get("source_trade")
    source_trade = source_trade if isinstance(source_trade, dict) else {}
    strategy_context = data.get("strategy_context")
    strategy_context = strategy_context if isinstance(strategy_context, dict) else {}
    copy_event = strategy_context.get("copy_event")
    copy_event = copy_event if isinstance(copy_event, dict) else {}
    execution_plan = data.get("execution_plan")
    execution_plan = execution_plan if isinstance(execution_plan, dict) else {}
    legs = execution_plan.get("legs")
    legs = legs if isinstance(legs, list) else []

    for raw_wallet in (
        copy_attribution.get("source_wallet"),
        copy_event.get("wallet_address"),
        copy_event.get("source_wallet"),
        source_trade.get("wallet_address"),
        source_trade.get("source_wallet"),
        data.get("wallet_address"),
        data.get("source_wallet"),
    ):
        normalized = StrategySDK.normalize_trader_wallet(raw_wallet)
        if normalized:
            return normalized

    for leg in legs:
        if not isinstance(leg, dict):
            continue
        metadata = leg.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        normalized = StrategySDK.normalize_trader_wallet(metadata.get("source_wallet"))
        if normalized:
            return normalized
    return ""


def _extract_copy_side_from_payload(payload: Any) -> str:
    data = payload if isinstance(payload, dict) else {}
    copy_attribution = data.get("copy_attribution")
    copy_attribution = copy_attribution if isinstance(copy_attribution, dict) else {}
    source_trade = data.get("source_trade")
    source_trade = source_trade if isinstance(source_trade, dict) else {}
    strategy_context = data.get("strategy_context")
    strategy_context = strategy_context if isinstance(strategy_context, dict) else {}
    copy_event = strategy_context.get("copy_event")
    copy_event = copy_event if isinstance(copy_event, dict) else {}
    raw_side = (
        copy_attribution.get("side")
        or copy_event.get("side")
        or source_trade.get("side")
        or data.get("side")
    )
    side = str(raw_side or "").strip().upper()
    if side in {"BUY", "SELL"}:
        return side
    return ""


async def _resolve_execution_wallet_address() -> str:
    wallet = ""
    try:
        wallet = str(live_execution_service.get_execution_wallet_address() or "").strip()
    except Exception:
        wallet = ""
    if wallet:
        return wallet

    try:
        await live_execution_service.ensure_initialized()
    except Exception:
        pass

    try:
        wallet = str(live_execution_service.get_execution_wallet_address() or "").strip()
    except Exception:
        wallet = ""
    if wallet:
        return wallet

    try:
        runtime_signature_type = getattr(live_execution_service, "_balance_signature_type", None)
        signature_type = (
            int(runtime_signature_type)
            if isinstance(runtime_signature_type, int)
            else int(getattr(settings, "POLYMARKET_SIGNATURE_TYPE", 1))
        )
    except Exception:
        signature_type = 1

    try:
        wallet = str(live_execution_service._funder_for_signature_type(signature_type) or "").strip()
    except Exception:
        wallet = ""
    if wallet:
        return wallet

    try:
        wallet = str(live_execution_service._get_wallet_address() or "").strip()
    except Exception:
        wallet = ""
    return wallet


def _wallet_position_token_key(token_id: Any) -> str:
    return str(token_id or "").strip().lower()


def _read_wallet_text(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    return ""


def _read_wallet_float(payload: dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        parsed = safe_float(payload.get(key))
        if parsed is not None:
            return float(parsed)
    return None


def _extract_market_token_ids(market_info: dict[str, Any]) -> list[str]:
    for key in ("token_ids", "tokenIds", "clob_token_ids", "clobTokenIds"):
        raw = market_info.get(key)
        if not isinstance(raw, list):
            continue
        out = [str(token_id or "").strip() for token_id in raw if str(token_id or "").strip()]
        if out:
            return out
    return []


def _extract_market_outcomes(market_info: dict[str, Any]) -> list[str]:
    raw = market_info.get("outcomes")
    if not isinstance(raw, list):
        return []
    return [str(value or "").strip().lower() for value in raw if str(value or "").strip()]


def _extract_binary_market_prices(market_info: dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    yes_price = safe_float(market_info.get("yes_price"))
    no_price = safe_float(market_info.get("no_price"))
    if yes_price is not None and no_price is not None:
        return yes_price, no_price

    outcome_prices = market_info.get("outcome_prices")
    if not isinstance(outcome_prices, list):
        outcome_prices = market_info.get("outcomePrices")
    if not isinstance(outcome_prices, list) or len(outcome_prices) < 2:
        return yes_price, no_price

    outcomes = _extract_market_outcomes(market_info)
    if outcomes and len(outcomes) == len(outcome_prices):
        for index, label in enumerate(outcomes):
            parsed_price = safe_float(outcome_prices[index])
            if parsed_price is None:
                continue
            if label == "yes":
                yes_price = parsed_price
            elif label == "no":
                no_price = parsed_price

    if yes_price is None:
        yes_price = safe_float(outcome_prices[0])
    if no_price is None:
        no_price = safe_float(outcome_prices[1])
    return yes_price, no_price


def _normalize_wallet_outcome_and_direction(
    wallet_position: dict[str, Any],
    *,
    market_info: dict[str, Any],
    token_id: str,
) -> tuple[str, str]:
    outcome_text = _read_wallet_text(wallet_position, "outcome", "position_side", "side").lower()
    if outcome_text in {"yes", "buy_yes"}:
        return "yes", "buy_yes"
    if outcome_text in {"no", "buy_no"}:
        return "no", "buy_no"

    outcome_index = safe_int(
        wallet_position.get("outcomeIndex") if wallet_position.get("outcomeIndex") is not None else wallet_position.get("outcome_index"),
        None,
    )
    token_ids = _extract_market_token_ids(market_info)
    if outcome_index is None and token_ids:
        normalized_token = _wallet_position_token_key(token_id)
        for index, known_token in enumerate(token_ids):
            if _wallet_position_token_key(known_token) == normalized_token:
                outcome_index = index
                break

    if outcome_index is None:
        return "", ""

    outcomes = _extract_market_outcomes(market_info)
    if outcomes and 0 <= outcome_index < len(outcomes):
        label = outcomes[outcome_index]
        if label == "yes":
            return "yes", "buy_yes"
        if label == "no":
            return "no", "buy_no"

    if outcome_index == 0:
        return "yes", "buy_yes"
    if outcome_index == 1:
        return "no", "buy_no"
    return "", ""


def _normalize_wallet_position_row(
    wallet_position: dict[str, Any],
    *,
    market_info: dict[str, Any],
) -> Optional[dict[str, Any]]:
    token_id = _read_wallet_text(wallet_position, "asset", "asset_id", "assetId", "token_id", "tokenId")
    if not token_id:
        return None

    size = _read_wallet_float(wallet_position, "size", "positionSize", "shares", "amount")
    if size is None or size <= 0.0:
        return None

    condition_id = _normalize_condition_id(
        _read_wallet_text(wallet_position, "conditionId", "condition_id", "market", "market_id", "marketId")
    )
    market_id = condition_id or _read_wallet_text(wallet_position, "market", "market_id", "marketId") or token_id

    avg_price = _read_wallet_float(wallet_position, "avgPrice", "avg_price", "avgCost", "average_cost")
    initial_value = _read_wallet_float(wallet_position, "initialValue", "initial_value")
    current_price = _read_wallet_float(
        wallet_position,
        "currentPrice",
        "current_price",
        "curPrice",
        "cur_price",
        "markPrice",
        "mark_price",
        "price",
    )
    current_value = _read_wallet_float(wallet_position, "currentValue", "current_value")
    unrealized_pnl = _read_wallet_float(wallet_position, "cashPnl", "cash_pnl", "unrealizedPnl", "unrealized_pnl")

    if (avg_price is None or avg_price <= 0.0) and initial_value is not None and initial_value > 0.0 and size > 0.0:
        avg_price = initial_value / size
    if (initial_value is None or initial_value <= 0.0) and avg_price is not None and avg_price > 0.0:
        initial_value = size * avg_price
    if (current_price is None or current_price <= 0.0) and current_value is not None and current_value > 0.0 and size > 0.0:
        current_price = current_value / size
    if current_value is None and current_price is not None and current_price > 0.0:
        current_value = current_price * size
    if unrealized_pnl is None and current_value is not None and initial_value is not None:
        unrealized_pnl = current_value - initial_value

    outcome, direction = _normalize_wallet_outcome_and_direction(
        wallet_position,
        market_info=market_info,
        token_id=token_id,
    )

    token_ids = _extract_market_token_ids(market_info)
    outcomes = _extract_market_outcomes(market_info)
    yes_token_id = ""
    no_token_id = ""
    if token_ids:
        yes_index = 0
        no_index = 1 if len(token_ids) > 1 else 0
        for index, label in enumerate(outcomes):
            if label == "yes":
                yes_index = index
            elif label == "no":
                no_index = index
        if 0 <= yes_index < len(token_ids):
            yes_token_id = str(token_ids[yes_index] or "").strip()
        if 0 <= no_index < len(token_ids):
            no_token_id = str(token_ids[no_index] or "").strip()

    yes_price, no_price = _extract_binary_market_prices(market_info)
    market_slug = _read_wallet_text(market_info, "slug")
    event_slug = _read_wallet_text(market_info, "event_slug", "eventSlug")
    market_question = (
        _read_wallet_text(wallet_position, "title", "market_question", "marketQuestion", "question")
        or _read_wallet_text(market_info, "question", "market_question", "title")
    )
    market_url = (
        build_polymarket_market_url(
            event_slug=event_slug or None,
            market_slug=market_slug or None,
            market_id=market_id,
            condition_id=condition_id or market_id,
        )
        or ""
    ).strip()

    return {
        "token_id": token_id,
        "market_id": market_id,
        "condition_id": condition_id or None,
        "market_question": market_question or None,
        "outcome": outcome or None,
        "direction": direction or None,
        "size": float(size),
        "avg_price": float(avg_price) if avg_price is not None and avg_price > 0.0 else None,
        "current_price": float(current_price) if current_price is not None and current_price > 0.0 else None,
        "initial_value": float(initial_value) if initial_value is not None and initial_value > 0.0 else None,
        "current_value": float(current_value) if current_value is not None and current_value > 0.0 else None,
        "unrealized_pnl": float(unrealized_pnl) if unrealized_pnl is not None else None,
        "yes_token_id": yes_token_id or None,
        "no_token_id": no_token_id or None,
        "yes_price": float(yes_price) if yes_price is not None else None,
        "no_price": float(no_price) if no_price is not None else None,
        "market_slug": market_slug or None,
        "event_slug": event_slug or None,
        "market_url": market_url or None,
    }


def _active_statuses_for_mode(mode: Any) -> set[str]:
    mode_key = _normalize_mode_key(mode)
    if mode_key == "shadow":
        return SHADOW_ACTIVE_ORDER_STATUSES
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


def _normalize_crypto_asset(value: Any) -> str:
    asset = str(value or "").strip().upper()
    if asset == "XBT":
        return "BTC"
    return asset


def _normalize_crypto_timeframe(value: Any) -> str:
    raw = str(value or "").strip().lower()
    compact = raw.replace("_", "").replace("-", "").replace(" ", "")
    if compact in {"5m", "5min", "5minute", "5minutes"}:
        return "5m"
    if compact in {"15m", "15min", "15minute", "15minutes"}:
        return "15m"
    if compact in {"1h", "1hr", "1hour", "60m", "60min"}:
        return "1h"
    if compact in {"4h", "4hr", "4hour", "240m", "240min"}:
        return "4h"
    return raw


def _extract_asset_timeframe_from_payload(payload: Any) -> tuple[str, str]:
    payload_dict = payload if isinstance(payload, dict) else {}
    strategy_context = payload_dict.get("strategy_context")
    strategy_context = strategy_context if isinstance(strategy_context, dict) else {}
    live_market = payload_dict.get("live_market")
    live_market = live_market if isinstance(live_market, dict) else {}
    position_state = payload_dict.get("position_state")
    position_state = position_state if isinstance(position_state, dict) else {}

    candidates = [payload_dict, strategy_context, live_market, position_state]

    asset = ""
    timeframe = ""
    for candidate in candidates:
        if not asset:
            asset = _normalize_crypto_asset(candidate.get("asset") or candidate.get("coin") or candidate.get("symbol"))
        if not timeframe:
            timeframe = _normalize_crypto_timeframe(
                candidate.get("timeframe") or candidate.get("cadence") or candidate.get("interval")
            )
        if asset and timeframe:
            break
    return asset, timeframe


def _position_cap_scope_key(
    *,
    position_cap_scope: str,
    mode: Any,
    market_id: Any,
    direction: Any,
    payload: Any,
) -> tuple[str, str, str] | None:
    mode_key = _normalize_mode_key(mode)
    market_key = str(market_id or "").strip()
    if not market_key:
        return None
    direction_key = str(direction or "").strip().lower() or "__unknown__"
    scope = str(position_cap_scope or "market_direction").strip().lower()
    if scope == "market":
        return mode_key, market_key, "__market__"
    if scope == "asset_timeframe":
        asset, timeframe = _extract_asset_timeframe_from_payload(payload)
        asset_key = asset or f"market:{market_key}"
        timeframe_key = timeframe or "__unknown_timeframe__"
        return mode_key, asset_key, timeframe_key
    return mode_key, market_key, direction_key


def _normalize_direction_side(value: Any) -> Optional[str]:
    raw = str(value or "").strip().lower()
    if not raw:
        return None
    token = raw.replace("-", "_").replace(" ", "_")
    if token in {"yes", "buy_yes", "sell_yes", "buy", "long", "up"}:
        return "YES"
    if token in {"no", "buy_no", "sell_no", "sell", "short", "down"}:
        return "NO"
    if token.startswith("buy_yes") or token.startswith("sell_yes"):
        return "YES"
    if token.startswith("buy_no") or token.startswith("sell_no"):
        return "NO"
    return None


def _extract_outcome_labels(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    labels: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            label = str(
                item.get("label")
                or item.get("outcome")
                or item.get("name")
                or item.get("title")
                or item.get("value")
                or ""
            ).strip()
        else:
            label = str(item or "").strip()
        if label and label not in labels:
            labels.append(label)
    return labels


def _extract_market_outcome_labels(market: dict[str, Any]) -> list[str]:
    for key in ("outcome_labels", "outcomeLabels", "outcomes", "labels"):
        labels = _extract_outcome_labels(market.get(key))
        if labels:
            return labels
    return _extract_outcome_labels(market.get("tokens"))


def _find_signal_market(signal_payload: dict[str, Any], market_id: str) -> Optional[dict[str, Any]]:
    raw_markets = signal_payload.get("markets")
    if not isinstance(raw_markets, list):
        return None
    normalized_market_id = str(market_id or "").strip().lower()
    first_market: Optional[dict[str, Any]] = None
    market_dict_count = 0
    for raw_market in raw_markets:
        if not isinstance(raw_market, dict):
            continue
        market_dict_count += 1
        if first_market is None:
            first_market = raw_market
        if not normalized_market_id:
            continue
        for candidate in (
            raw_market.get("id"),
            raw_market.get("market_id"),
            raw_market.get("condition_id"),
            raw_market.get("conditionId"),
            raw_market.get("slug"),
            raw_market.get("market_slug"),
            raw_market.get("ticker"),
        ):
            candidate_text = str(candidate or "").strip().lower()
            if candidate_text and candidate_text == normalized_market_id:
                return raw_market
    if market_dict_count == 1:
        return first_market
    return None


def _resolve_direction_label_from_signal(
    signal_payload: dict[str, Any],
    *,
    market_id: str,
    direction_side: str,
) -> Optional[str]:
    if direction_side not in {"YES", "NO"}:
        return None
    labels: list[str] = []
    matched_market = _find_signal_market(signal_payload, market_id)
    if isinstance(matched_market, dict):
        labels = _extract_market_outcome_labels(matched_market)
    if not labels:
        for key in ("outcome_labels", "outcomeLabels", "outcomes", "labels"):
            labels = _extract_outcome_labels(signal_payload.get(key))
            if labels:
                break
    index = 0 if direction_side == "YES" else 1
    if len(labels) <= index:
        return None
    label = str(labels[index] or "").strip()
    return label or None


def _resolve_binary_labels_from_signal(
    signal_payload: dict[str, Any],
    *,
    market_id: str,
) -> tuple[Optional[str], Optional[str]]:
    labels: list[str] = []
    matched_market = _find_signal_market(signal_payload, market_id)
    if isinstance(matched_market, dict):
        labels = _extract_market_outcome_labels(matched_market)
    if not labels:
        for key in ("outcome_labels", "outcomeLabels", "outcomes", "labels"):
            labels = _extract_outcome_labels(signal_payload.get(key))
            if labels:
                break
    yes_label = str(labels[0] or "").strip() if len(labels) > 0 else ""
    no_label = str(labels[1] or "").strip() if len(labels) > 1 else ""
    return (yes_label or None, no_label or None)


def _resolve_binary_labels_from_market_payload(
    market_payload: dict[str, Any],
) -> tuple[Optional[str], Optional[str]]:
    labels = _extract_market_outcome_labels(market_payload)
    yes_label = str(labels[0] or "").strip() if len(labels) > 0 else ""
    no_label = str(labels[1] or "").strip() if len(labels) > 1 else ""
    return (yes_label or None, no_label or None)


def _resolve_direction_label_from_payload(payload: dict[str, Any]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    leg = payload.get("leg")
    if isinstance(leg, dict):
        for key in ("outcome_label", "outcomeLabel", "outcome_name", "outcomeName"):
            label = str(leg.get(key) or "").strip()
            if label:
                return label
        outcome = str(leg.get("outcome") or "").strip()
        if outcome and _normalize_direction_side(outcome) is None:
            return outcome
    for key in ("outcome_label", "outcomeLabel"):
        label = str(payload.get(key) or "").strip()
        if label:
            return label
    outcome = str(payload.get("outcome") or "").strip()
    if outcome and _normalize_direction_side(outcome) is None:
        return outcome
    return None


def _resolve_direction_presentation(
    *,
    direction: Any,
    payload: Optional[dict[str, Any]] = None,
    signal_payload: Optional[dict[str, Any]] = None,
    market_payload: Optional[dict[str, Any]] = None,
    market_id: Any = None,
) -> tuple[Optional[str], str, Optional[str], Optional[str]]:
    payload_obj = payload if isinstance(payload, dict) else {}
    signal_payload_obj = signal_payload if isinstance(signal_payload, dict) else {}
    market_payload_obj = market_payload if isinstance(market_payload, dict) else {}
    direction_side = _normalize_direction_side(direction)
    leg = payload_obj.get("leg")
    leg = leg if isinstance(leg, dict) else {}
    leg_outcome_side = _normalize_direction_side(leg.get("outcome"))
    if leg_outcome_side:
        direction_side = leg_outcome_side
    resolved_market_id = str(market_id or leg.get("market_id") or payload_obj.get("market_id") or "").strip()
    yes_label, no_label = _resolve_binary_labels_from_signal(
        signal_payload_obj,
        market_id=resolved_market_id,
    )
    if not yes_label or not no_label:
        market_yes_label, market_no_label = _resolve_binary_labels_from_market_payload(market_payload_obj)
        yes_label = yes_label or market_yes_label
        no_label = no_label or market_no_label
    direction_label = _resolve_direction_label_from_payload(payload_obj)
    if not direction_label and direction_side:
        direction_label = yes_label if direction_side == "YES" else no_label
    if not direction_label and direction_side:
        direction_label = _resolve_direction_label_from_signal(
            signal_payload_obj, market_id=resolved_market_id, direction_side=direction_side
        )
    if not direction_label and direction_side:
        direction_label = direction_side
    if not direction_label:
        raw_direction = str(direction or "").strip()
        direction_label = raw_direction.upper() if raw_direction else "N/A"
    return direction_side, direction_label, yes_label, no_label


def _first_float_from_dicts(candidates: list[Any], keys: tuple[str, ...]) -> Optional[float]:
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for key in keys:
            parsed = safe_float(candidate.get(key))
            if parsed is not None:
                return float(parsed)
    return None


def _extract_live_provider_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    reconciliation = payload.get("provider_reconciliation")
    if not isinstance(reconciliation, dict):
        reconciliation = {}
    snapshot = reconciliation.get("snapshot")
    if not isinstance(snapshot, dict):
        snapshot = {}
    provider_snapshot = payload.get("provider_snapshot")
    if not isinstance(provider_snapshot, dict):
        provider_snapshot = {}
    return {
        "reconciliation": reconciliation,
        "snapshot": snapshot,
        "provider_snapshot": provider_snapshot,
        "payload": payload,
    }


def _extract_live_fill_metrics(payload: dict[str, Any]) -> tuple[float, float, Optional[float]]:
    snapshots = _extract_live_provider_snapshot(payload)
    candidates: list[Any] = [
        snapshots["reconciliation"],
        snapshots["snapshot"],
        snapshots["provider_snapshot"],
        payload,
    ]
    filled_shares = max(
        0.0,
        _first_float_from_dicts(
            candidates,
            (
                "filled_size",
                "size_matched",
                "sizeMatched",
                "matched_size",
                "filled_shares",
                "executed_size",
            ),
        )
        or 0.0,
    )
    average_fill_price = _first_float_from_dicts(
        candidates,
        (
            "average_fill_price",
            "avg_fill_price",
            "avg_price",
            "avgFillPrice",
            "matched_price",
            "price",
            "limit_price",
        ),
    )
    filled_notional_usd = max(
        0.0,
        _first_float_from_dicts(
            candidates,
            (
                "filled_notional_usd",
                "filled_notional",
                "matched_notional",
                "matched_amount",
                "executed_notional",
            ),
        )
        or 0.0,
    )
    if filled_notional_usd <= 0.0 and filled_shares > 0.0 and average_fill_price is not None and average_fill_price > 0:
        filled_notional_usd = filled_shares * average_fill_price
    if filled_shares <= 0.0 and filled_notional_usd > 0.0 and average_fill_price is not None and average_fill_price > 0:
        filled_shares = filled_notional_usd / average_fill_price
    return filled_notional_usd, filled_shares, average_fill_price


def _live_active_notional(mode: Any, status: Any, row_notional: float, payload: dict[str, Any]) -> float:
    mode_key = _normalize_mode_key(mode)
    if mode_key not in {"live", "shadow"}:
        return max(0.0, abs(row_notional))

    status_key = _normalize_status_key(status)
    filled_notional_usd, _, _ = _extract_live_fill_metrics(payload)
    if status_key == "executed":
        if filled_notional_usd > 0.0:
            return filled_notional_usd
        return max(0.0, abs(row_notional))
    if status_key in {"open", "submitted"}:
        return max(0.0, filled_notional_usd)
    return 0.0


def _resolve_provider_order_ids(
    *,
    order_payload: dict[str, Any],
    session_order: Optional[ExecutionSessionOrder],
) -> tuple[Optional[str], Optional[str]]:
    provider_order_candidates: list[str] = []
    provider_clob_candidates: list[str] = []
    execution_payload = order_payload.get("execution_session")
    if not isinstance(execution_payload, dict):
        execution_payload = {}

    for value in (
        order_payload.get("order_id"),
        order_payload.get("provider_order_id"),
        execution_payload.get("provider_order_id"),
        getattr(session_order, "provider_order_id", None),
    ):
        text = str(value or "").strip()
        if text:
            provider_order_candidates.append(text)

    for value in (
        order_payload.get("clob_order_id"),
        order_payload.get("provider_clob_order_id"),
        execution_payload.get("provider_clob_order_id"),
        getattr(session_order, "provider_clob_order_id", None),
    ):
        text = str(value or "").strip()
        if text:
            provider_clob_candidates.append(text)

    provider_order_id = provider_order_candidates[0] if provider_order_candidates else None
    provider_clob_order_id = provider_clob_candidates[0] if provider_clob_candidates else None
    return provider_order_id, provider_clob_order_id


def _cleanup_timeout_reason(note_reason: str) -> bool:
    normalized = str(note_reason or "").strip().lower()
    return normalized.startswith("max_open_order_timeout")


def _cleanup_rescue_side_from_direction(direction: Any) -> str:
    direction_key = str(direction or "").strip().lower()
    if direction_key.startswith("sell"):
        return "SELL"
    return "BUY"


def _cleanup_rescue_token_id(
    *,
    order_payload: dict[str, Any],
    session_order: Optional[ExecutionSessionOrder],
) -> str:
    candidates: list[str] = []
    execution_payload = order_payload.get("execution_session")
    if not isinstance(execution_payload, dict):
        execution_payload = {}
    leg_payload = order_payload.get("leg")
    if not isinstance(leg_payload, dict):
        leg_payload = {}
    session_payload = dict(session_order.payload_json or {}) if session_order is not None else {}
    session_leg_payload = session_payload.get("leg")
    if not isinstance(session_leg_payload, dict):
        session_leg_payload = {}

    for value in (
        order_payload.get("token_id"),
        order_payload.get("selected_token_id"),
        leg_payload.get("token_id"),
        execution_payload.get("token_id"),
        session_payload.get("token_id"),
        session_leg_payload.get("token_id"),
        order_payload.get("yes_token_id"),
        order_payload.get("no_token_id"),
    ):
        text = str(value or "").strip()
        if text:
            candidates.append(text)
    return candidates[0] if candidates else ""


def _cleanup_rescue_shares(
    *,
    row: TraderOrder,
    order_payload: dict[str, Any],
    session_order: Optional[ExecutionSessionOrder],
) -> float:
    session_payload = dict(session_order.payload_json or {}) if session_order is not None else {}
    requested_shares = safe_float(order_payload.get("requested_shares"), None)
    if requested_shares is None:
        requested_shares = safe_float((order_payload.get("leg") or {}).get("requested_shares"), None)
    if requested_shares is None:
        requested_shares = safe_float(session_payload.get("requested_shares"), None)
    if requested_shares is None:
        requested_shares = safe_float(order_payload.get("shares"), None)
    if requested_shares is None:
        requested_shares = safe_float(order_payload.get("size"), None)
    if requested_shares is None:
        requested_notional = safe_float(row.notional_usd, 0.0) or 0.0
        entry_price = safe_float(row.entry_price, None)
        if entry_price is None or entry_price <= 0.0:
            entry_price = safe_float(order_payload.get("resolved_price"), None)
        if entry_price is not None and entry_price > 0.0 and requested_notional > 0.0:
            requested_shares = requested_notional / entry_price
    return max(0.0, float(requested_shares or 0.0))


def _cleanup_rescue_price(
    *,
    side: str,
    row: TraderOrder,
    order_payload: dict[str, Any],
    rescue_price_bps: float,
) -> float | None:
    base_price = safe_float(row.entry_price, None)
    if base_price is None or base_price <= 0.0:
        base_price = safe_float(row.effective_price, None)
    if base_price is None or base_price <= 0.0:
        base_price = safe_float(order_payload.get("resolved_price"), None)
    if base_price is None or base_price <= 0.0:
        base_price = safe_float(order_payload.get("fallback_price"), None)
    if base_price is None or base_price <= 0.0:
        return None
    bps = max(0.0, float(rescue_price_bps))
    if side == "SELL":
        adjusted = float(base_price) * (1.0 - (bps / 10_000.0))
    else:
        adjusted = float(base_price) * (1.0 + (bps / 10_000.0))
    return min(0.99, max(0.01, adjusted))


async def _attempt_timeout_taker_rescue(
    *,
    row: TraderOrder,
    order_payload: dict[str, Any],
    session_order: Optional[ExecutionSessionOrder],
    rescue_price_bps: float,
    rescue_time_in_force: str,
) -> dict[str, Any]:
    token_id = _cleanup_rescue_token_id(order_payload=order_payload, session_order=session_order)
    side = _cleanup_rescue_side_from_direction(row.direction)
    shares = _cleanup_rescue_shares(row=row, order_payload=order_payload, session_order=session_order)
    fallback_price = _cleanup_rescue_price(
        side=side,
        row=row,
        order_payload=order_payload,
        rescue_price_bps=rescue_price_bps,
    )
    result: dict[str, Any] = {
        "attempted": False,
        "success": False,
        "status": "failed",
        "token_id": token_id or None,
        "side": side,
        "requested_shares": shares,
        "fallback_price": fallback_price,
        "time_in_force": rescue_time_in_force,
        "price_bps": float(rescue_price_bps),
        "order_id": None,
        "clob_order_id": None,
        "effective_price": None,
        "error": None,
        "payload": {},
    }
    if not token_id:
        result["error"] = "missing_token_id"
        return result
    if shares <= 0.0:
        result["error"] = "missing_shares"
        return result
    if fallback_price is None or fallback_price <= 0.0:
        result["error"] = "missing_price"
        return result

    result["attempted"] = True
    try:
        execution = await execute_live_order(
            token_id=token_id,
            side=side,
            size=float(shares),
            fallback_price=float(fallback_price),
            market_question=str(row.market_question or ""),
            opportunity_id=str(row.signal_id or row.id or ""),
            time_in_force=str(rescue_time_in_force or DEFAULT_TIMEOUT_TAKER_RESCUE_TIME_IN_FORCE).strip().upper() or "IOC",
            post_only=False,
            resolve_live_price=True,
            prefer_cached_price=True,
            enforce_fallback_bound=True,
        )
    except Exception as exc:
        result["error"] = str(exc)
        return result

    status_key = _normalize_status_key(execution.status)
    result["status"] = status_key or "failed"
    result["success"] = status_key in {"executed", "open", "submitted"}
    result["order_id"] = str(execution.order_id or "").strip() or None
    result["clob_order_id"] = str((execution.payload or {}).get("clob_order_id") or "").strip() or None
    result["effective_price"] = safe_float(execution.effective_price, None)
    result["error"] = str(execution.error_message or "").strip() or None
    result["payload"] = dict(execution.payload or {})
    return result


def _map_provider_snapshot_status(snapshot_status: str, filled_notional_usd: float) -> str:
    status_key = str(snapshot_status or "").strip().lower()
    if status_key in {"filled"}:
        return "executed"
    if status_key in {"partially_filled", "open"}:
        return "open"
    if status_key in {"pending"}:
        return "submitted"
    if status_key in {"cancelled", "expired", "failed"}:
        if filled_notional_usd > 0.0:
            return "executed"
        return "cancelled" if status_key in {"cancelled", "expired"} else "failed"
    return ""


def _map_live_order_status_to_leg_status(order_status: str) -> str:
    status_key = _normalize_status_key(order_status)
    if status_key == "executed":
        return "completed"
    if status_key in {"open", "submitted"}:
        return "open"
    if status_key in {"cancelled", "failed", "expired"}:
        return "failed"
    return "open"


def _is_terminal_order_status(status: Any) -> bool:
    return _normalize_status_key(status) in {
        "cancelled",
        "failed",
        "expired",
        "resolved_win",
        "resolved_loss",
        "closed_win",
        "closed_loss",
        "win",
        "loss",
    }


def _sync_order_runtime_payload(
    *,
    payload: dict[str, Any],
    status: Any,
    now: datetime,
    provider_snapshot_status: Optional[str] = None,
    mapped_status: Optional[str] = None,
) -> dict[str, Any]:
    status_key = _normalize_status_key(status)
    if status_key:
        payload["trading_status"] = status_key

    provider_reconciliation = payload.get("provider_reconciliation")
    if not isinstance(provider_reconciliation, dict):
        provider_reconciliation = {}
    provider_snapshot = provider_reconciliation.get("snapshot")
    if not isinstance(provider_snapshot, dict):
        provider_snapshot = {}

    snapshot_status_key = _normalize_status_key(
        provider_snapshot_status
        or provider_reconciliation.get("snapshot_status")
        or provider_snapshot.get("normalized_status")
        or provider_snapshot.get("status")
    )
    if snapshot_status_key:
        provider_reconciliation["snapshot_status"] = snapshot_status_key

    mapped_status_key = _normalize_status_key(mapped_status or provider_reconciliation.get("mapped_status"))
    if mapped_status_key:
        provider_reconciliation["mapped_status"] = mapped_status_key

    if _is_terminal_order_status(status_key):
        if not snapshot_status_key:
            snapshot_status_key = mapped_status_key or status_key
            provider_reconciliation["snapshot_status"] = snapshot_status_key
        if not mapped_status_key:
            mapped_status_key = "cancelled" if status_key in {"cancelled", "failed", "expired"} else "executed"
            provider_reconciliation["mapped_status"] = mapped_status_key
        provider_reconciliation["reconciled_at"] = to_iso(now)

    if snapshot_status_key:
        provider_snapshot["normalized_status"] = snapshot_status_key
        provider_reconciliation["snapshot"] = provider_snapshot
    elif mapped_status_key:
        provider_snapshot["normalized_status"] = mapped_status_key
        provider_reconciliation["snapshot"] = provider_snapshot

    if provider_reconciliation:
        payload["provider_reconciliation"] = provider_reconciliation

    return payload


def _active_order_notional_for_metrics(order: TraderOrder) -> float:
    mode_key = _normalize_mode_key(order.mode)
    if not _is_active_order_status(mode_key, order.status):
        return 0.0
    payload = dict(order.payload_json or {})
    row_notional = safe_float(order.notional_usd, 0.0) or 0.0
    return _live_active_notional(mode_key, order.status, row_notional, payload)


def _normalize_confidence_fraction(value: Any, default: float = 0.0) -> float:
    parsed = safe_float(value, default)
    if parsed < 0.0:
        parsed = 0.0
    elif parsed <= 1.0:
        # Already a 0-1 fraction; use as-is.
        pass
    elif parsed <= 100.0:
        # Percentage scale (e.g. 75 -> 0.75).
        parsed = parsed / 100.0
    else:
        # Values above 100 are nonsensical; clamp to 1.0.
        parsed = 1.0
    return parsed


def _normalize_resume_policy(value: Any) -> str:
    return StrategySDK.validate_trader_runtime_metadata({"resume_policy": value}).get("resume_policy", "resume_full")


def _normalize_strategy_key(value: Any) -> str:
    return str(value or "").strip().lower()


def _canonical_strategy_key(value: Any) -> str:
    key = _normalize_strategy_key(value)
    return _LEGACY_STRATEGY_ALIASES.get(key, key)


def _normalize_trader_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode == "paper":
        return "shadow"
    if not mode:
        return "shadow"
    if mode not in _TRADER_MODES:
        raise ValueError("mode must be 'shadow' or 'live'")
    return mode


def _normalize_strategy_for_source(source_key: str, strategy_key: str) -> str:
    del source_key
    return _canonical_strategy_key(strategy_key)


def _normalize_timeframe_scope(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        values = value
    elif isinstance(value, str):
        values = [part.strip() for part in value.split(",")]
    else:
        values = []
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        token = str(raw or "").strip().lower()
        if token in {"5m", "5min", "5"}:
            token = "5m"
        elif token in {"15m", "15min", "15"}:
            token = "15m"
        elif token in {"1h", "1hr", "60m", "60min"}:
            token = "1h"
        elif token in {"4h", "4hr", "240m", "240min"}:
            token = "4h"
        else:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


async def _fetch_enabled_strategy_catalog(
    session: AsyncSession,
) -> tuple[set[str], dict[str, set[str]]]:
    rows = list(
        (
            await session.execute(
                select(
                    Strategy.slug,
                    Strategy.source_key,
                ).where(Strategy.enabled == True)  # noqa: E712
            )
        ).all()
    )
    if not rows:
        fallback_rows = build_system_opportunity_strategy_rows()
        valid_keys: set[str] = set()
        by_source: dict[str, set[str]] = {}
        for row in fallback_rows:
            slug = str(row.get("slug") or "").strip().lower()
            src = str(row.get("source_key") or "").strip().lower()
            if not slug or not src:
                continue
            valid_keys.add(slug)
            by_source.setdefault(src, set()).add(slug)
        return valid_keys, by_source

    valid_keys: set[str] = set()
    by_source: dict[str, set[str]] = {}
    for strategy_key, source_key in rows:
        skey = str(strategy_key or "").strip().lower()
        src = str(source_key or "").strip().lower()
        if not skey or not src:
            continue
        valid_keys.add(skey)
        by_source.setdefault(src, set()).add(skey)
    return valid_keys, by_source


async def _validate_strategy_key(session: AsyncSession, strategy_key: str) -> None:
    valid_keys, _ = await _fetch_enabled_strategy_catalog(session)
    canonical_strategy_key = _canonical_strategy_key(strategy_key)
    if not canonical_strategy_key:
        raise ValueError("strategy_key is required")
    if canonical_strategy_key not in valid_keys:
        allowed = ", ".join(sorted(valid_keys))
        raise ValueError(f"Unknown strategy_key '{strategy_key}'. Valid strategy keys: {allowed}")


def _normalize_strategy_params(value: Any, source_key: str, strategy_key: str = "") -> dict[str, Any]:
    params = value if isinstance(value, dict) else {}
    out = dict(params)
    normalized_source = str(source_key or "").strip().lower()
    normalized_strategy_key = str(strategy_key or "").strip().lower()
    if normalized_source == "traders":
        if normalized_strategy_key == "traders_copy_trade":
            return validate_traders_copy_trade_config(out)
        return StrategySDK.validate_trader_filter_config(out)
    if normalized_source == "news":
        return validate_news_edge_config(out)
    if "min_confidence" in out:
        out["min_confidence"] = _normalize_confidence_fraction(out.get("min_confidence"), 0.0)
    return StrategySDK.normalize_strategy_retention_config(out)


def _validate_traders_scope(scope: dict[str, Any]) -> None:
    modes = list(scope.get("modes") or [])
    if not modes:
        raise ValueError("traders_scope.modes must include at least one mode")
    invalid = [mode for mode in modes if mode not in _TRADER_SCOPE_MODES]
    if invalid:
        raise ValueError(
            f"Invalid traders_scope.modes: {', '.join(invalid)}. Allowed: {', '.join(sorted(_TRADER_SCOPE_MODES))}"
        )
    if "individual" in modes and not list(scope.get("individual_wallets") or []):
        raise ValueError("traders_scope.individual_wallets is required when mode 'individual' is selected")
    if "group" in modes and not list(scope.get("group_ids") or []):
        raise ValueError("traders_scope.group_ids is required when mode 'group' is selected")


def _normalize_source_config(raw: Any) -> dict[str, Any]:
    item = raw if isinstance(raw, dict) else {}
    source_key = normalize_source_key(item.get("source_key"))
    requested_strategy_key = _normalize_strategy_key(item.get("strategy_key"))
    raw_strategy_params = dict(item.get("strategy_params") or {})
    strategy_key = _normalize_strategy_for_source(
        source_key,
        requested_strategy_key,
    )
    if source_key == "crypto" and requested_strategy_key in _LEGACY_CRYPTO_STRATEGY_TIMEFRAME_ALIASES:
        timeframe = _LEGACY_CRYPTO_STRATEGY_TIMEFRAME_ALIASES[requested_strategy_key]
        include_timeframes = _normalize_timeframe_scope(raw_strategy_params.get("include_timeframes"))
        raw_strategy_params["include_timeframes"] = [timeframe] if not include_timeframes else include_timeframes
        exclude_timeframes = [
            token for token in _normalize_timeframe_scope(raw_strategy_params.get("exclude_timeframes")) if token != timeframe
        ]
        if exclude_timeframes:
            raw_strategy_params["exclude_timeframes"] = exclude_timeframes
        elif "exclude_timeframes" in raw_strategy_params:
            raw_strategy_params.pop("exclude_timeframes", None)
        raw_strategy_params.setdefault("enable_live_market_context", False)
        raw_strategy_params.setdefault("require_live_market_revalidation", False)
        raw_strategy_params.setdefault("require_live_revalidation_for_sources", [])
        raw_strategy_params.setdefault("enforce_market_data_freshness", False)
        raw_strategy_params.setdefault("require_market_data_age_for_sources", [])
    if requested_strategy_key and requested_strategy_key != strategy_key:
        accepted = raw_strategy_params.get("accepted_signal_strategy_types")
        if isinstance(accepted, list):
            accepted_list = [str(item).strip().lower() for item in accepted if str(item or "").strip()]
        elif isinstance(accepted, str):
            accepted_list = [part.strip().lower() for part in accepted.split(",") if part.strip()]
        else:
            accepted_list = []
        if requested_strategy_key not in accepted_list:
            accepted_list.append(requested_strategy_key)
        raw_strategy_params["accepted_signal_strategy_types"] = accepted_list
    strategy_version = normalize_strategy_version(item.get("strategy_version"))
    strategy_params = _normalize_strategy_params(raw_strategy_params, source_key, strategy_key)
    normalized: dict[str, Any] = {
        "source_key": source_key,
        "strategy_key": strategy_key,
        "strategy_version": strategy_version,
        "strategy_params": strategy_params,
    }
    return normalized


def _normalize_source_configs(value: Any) -> list[dict[str, Any]]:
    items = value if isinstance(value, list) else []
    out: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for raw_item in items:
        source_config = _normalize_source_config(raw_item)
        source_key = source_config.get("source_key", "")
        if not source_key:
            continue
        if source_key in seen_sources:
            raise ValueError(f"Duplicate source_key '{source_key}' in source_configs")
        seen_sources.add(source_key)
        out.append(source_config)
    return out


async def _validate_source_strategy_pair(
    session: AsyncSession,
    source_key: str,
    strategy_key: str,
    strategy_version: int | None = None,
) -> None:
    _, by_source = await _fetch_enabled_strategy_catalog(session)
    valid_strategies = by_source.get(source_key)
    if not valid_strategies:
        allowed_sources = ", ".join(sorted(by_source.keys()))
        raise ValueError(f"Unknown source_key '{source_key}'. Allowed source keys: {allowed_sources}")
    await _validate_strategy_key(session, strategy_key)
    if strategy_key not in valid_strategies:
        allowed = ", ".join(sorted(valid_strategies))
        raise ValueError(
            f"Invalid strategy_key '{strategy_key}' for source_key '{source_key}'. Allowed strategies: {allowed}"
        )
    if strategy_version is not None:
        try:
            await resolve_strategy_version(
                session,
                strategy_key=strategy_key,
                requested_version=int(strategy_version),
            )
        except ValueError as exc:
            raise ValueError(str(exc))


async def _validate_source_configs(
    session: AsyncSession,
    source_configs: list[dict[str, Any]],
) -> None:
    if not source_configs:
        raise ValueError("source_configs must include at least one source")
    for source_config in source_configs:
        source_key = str(source_config.get("source_key") or "").strip().lower()
        strategy_key = _normalize_strategy_key(source_config.get("strategy_key"))
        strategy_version = normalize_strategy_version(source_config.get("strategy_version"))
        await _validate_source_strategy_pair(
            session,
            source_key,
            strategy_key,
            strategy_version=strategy_version,
        )
        if source_key == "traders":
            strategy_params = dict(source_config.get("strategy_params") or {})
            _validate_traders_scope(StrategySDK.validate_trader_scope_config(strategy_params.get("traders_scope")))


def _default_control_settings() -> dict[str, Any]:
    return _normalize_control_settings({})


def _serialize_control(row: TraderOrchestratorControl) -> dict[str, Any]:
    return {
        "id": row.id,
        "is_enabled": bool(row.is_enabled),
        "is_paused": bool(row.is_paused),
        "mode": _normalize_trader_mode(row.mode),
        "run_interval_seconds": int(row.run_interval_seconds or ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS),
        "requested_run_at": to_iso(row.requested_run_at),
        "kill_switch": bool(row.kill_switch),
        "settings": _normalize_control_settings(row.settings_json),
        "updated_at": to_iso(row.updated_at),
    }


def _serialize_snapshot(row: TraderOrchestratorSnapshot) -> dict[str, Any]:
    interval_seconds = int(row.interval_seconds or ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS)
    lag_seconds: Optional[float] = None
    if row.last_run_at is not None:
        try:
            lag_seconds = max(0.0, (_now() - row.last_run_at).total_seconds())
        except Exception:
            lag_seconds = None

    running = bool(row.running)
    if running and lag_seconds is not None:
        stale_after_seconds = max(
            _ORCHESTRATOR_SNAPSHOT_STALE_MIN_SECONDS,
            float(max(1, interval_seconds)) * _ORCHESTRATOR_SNAPSHOT_STALE_MULTIPLIER,
        )
        if lag_seconds > stale_after_seconds:
            running = False

    return {
        "id": row.id,
        "updated_at": to_iso(row.updated_at),
        "last_run_at": to_iso(row.last_run_at),
        "running": running,
        "enabled": bool(row.enabled),
        "current_activity": row.current_activity,
        "interval_seconds": interval_seconds,
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
    source_configs = _normalize_source_configs(row.source_configs_json)
    return {
        "id": row.id,
        "name": row.name,
        "description": row.description,
        "mode": _normalize_trader_mode(row.mode),
        "source_configs": source_configs,
        "risk_limits": row.risk_limits_json or {},
        "metadata": metadata,
        "is_enabled": bool(row.is_enabled),
        "is_paused": bool(row.is_paused),
        "interval_seconds": int(row.interval_seconds or 60),
        "requested_run_at": to_iso(row.requested_run_at),
        "last_run_at": to_iso(row.last_run_at),
        "next_run_at": to_iso(row.next_run_at),
        "created_at": to_iso(row.created_at),
        "updated_at": to_iso(row.updated_at),
    }


def _serialize_decision(row: TraderDecision) -> dict[str, Any]:
    return {
        "id": row.id,
        "trader_id": row.trader_id,
        "signal_id": row.signal_id,
        "source": row.source,
        "strategy_key": row.strategy_key,
        "strategy_version": int(row.strategy_version) if row.strategy_version is not None else None,
        "decision": row.decision,
        "reason": row.reason,
        "score": row.score,
        "event_id": row.event_id,
        "trace_id": row.trace_id,
        "checks_summary": row.checks_summary_json or {},
        "risk_snapshot": row.risk_snapshot_json or {},
        "payload": row.payload_json or {},
        "created_at": to_iso(row.created_at),
    }


def _serialize_decision_with_signal(
    row: TraderDecision,
    signal: TradeSignal | None,
) -> dict[str, Any]:
    serialized = _serialize_decision(row)

    signal_payload_full: dict[str, Any] = {}
    signal_payload: dict[str, Any] = {}
    signal_strategy_context: dict[str, Any] = {}
    market_id: str | None = None
    market_question: str | None = None
    direction: str | None = None
    market_price: float | None = None
    edge_percent: float | None = None
    confidence: float | None = None
    model_probability: float | None = None

    if signal is not None:
        market_id = str(signal.market_id or "").strip() or None
        market_question = str(signal.market_question or "").strip() or None
        direction = str(signal.direction or "").strip() or None
        market_price = signal.effective_price if signal.effective_price is not None else signal.entry_price
        edge_percent = signal.edge_percent
        confidence = signal.confidence
        if isinstance(signal.payload_json, dict):
            signal_payload_full = signal.payload_json
            for key in ("execution_plan", "markets", "positions_to_take"):
                value = signal_payload_full.get(key)
                if value is not None:
                    signal_payload[key] = value
        if isinstance(signal.strategy_context_json, dict):
            signal_strategy_context = signal.strategy_context_json

    ai_analysis = signal_payload_full.get("ai_analysis") if isinstance(signal_payload_full, dict) else None
    if isinstance(ai_analysis, dict):
        model_probability = safe_float(
            ai_analysis.get("model_probability")
            or ai_analysis.get("predicted_probability")
            or ai_analysis.get("estimated_probability")
            or ai_analysis.get("probability"),
            None,
        )
    if model_probability is None and isinstance(signal_payload_full, dict):
        model_probability = safe_float(
            signal_payload_full.get("model_probability")
            or signal_payload_full.get("predicted_probability")
            or signal_payload_full.get("probability"),
            None,
        )

    direction_side, direction_label, yes_label, no_label = _resolve_direction_presentation(
        direction=direction,
        payload=row.payload_json if isinstance(row.payload_json, dict) else {},
        signal_payload=signal_payload_full if isinstance(signal_payload_full, dict) else {},
        market_id=market_id,
    )

    serialized.update(
        {
            "market_id": market_id,
            "market_question": market_question,
            "direction": direction,
            "direction_side": direction_side,
            "direction_label": direction_label,
            "yes_label": yes_label,
            "no_label": no_label,
            "market_price": market_price,
            "model_probability": model_probability,
            "edge_percent": edge_percent,
            "confidence": confidence,
            "signal_score": row.score,
            "signal_payload": signal_payload,
            "signal_strategy_context": signal_strategy_context,
        }
    )
    return serialized


def _serialize_order(
    row: TraderOrder,
    signal: TradeSignal | None = None,
    cached_market: CachedMarket | None = None,
) -> dict[str, Any]:
    payload = dict(row.payload_json or {})
    provider_reconciliation = payload.get("provider_reconciliation")
    if not isinstance(provider_reconciliation, dict):
        provider_reconciliation = {}
    provider_snapshot = provider_reconciliation.get("snapshot")
    if not isinstance(provider_snapshot, dict):
        provider_snapshot = {}
    position_state = payload.get("position_state")
    if not isinstance(position_state, dict):
        position_state = {}
    position_close = payload.get("position_close")
    if not isinstance(position_close, dict):
        position_close = {}
    pending_live_exit = payload.get("pending_live_exit")
    if not isinstance(pending_live_exit, dict):
        pending_live_exit = {}
    close_trigger = str(position_close.get("close_trigger") or pending_live_exit.get("close_trigger") or "").strip()
    close_reason = str(position_close.get("reason") or pending_live_exit.get("reason") or row.reason or "").strip()
    filled_notional_usd, filled_shares, average_fill_price = _extract_live_fill_metrics(payload)
    price_anchor = average_fill_price
    if price_anchor is None or price_anchor <= 0:
        price_anchor = safe_float(row.effective_price) or safe_float(row.entry_price)
    current_price = safe_float(position_state.get("last_mark_price"))
    if current_price is None or current_price <= 0:
        current_price = safe_float(
            payload.get("market_price")
            or payload.get("resolved_price")
            or provider_snapshot.get("price")
            or provider_snapshot.get("limit_price")
        )
    filled_notional_display = filled_notional_usd
    if filled_notional_display <= 0:
        filled_notional_display = max(0.0, safe_float(row.notional_usd, 0.0) or 0.0)
    quantity = filled_shares
    if quantity <= 0 and price_anchor is not None and price_anchor > 0 and filled_notional_display > 0:
        quantity = filled_notional_display / price_anchor

    status_key = _normalize_status_key(row.status)
    provider_snapshot_status = _normalize_status_key(
        provider_reconciliation.get("snapshot_status")
        or provider_snapshot.get("normalized_status")
        or provider_snapshot.get("status")
        or payload.get("trading_status")
    )
    if (
        status_key
        and status_key not in LIVE_ACTIVE_ORDER_STATUSES
        and provider_snapshot_status
        in {
            "open",
            "submitted",
            "pending",
            "placing",
            "working",
            "partially_filled",
            "partial",
        }
    ):
        provider_snapshot_status = status_key
    if status_key and status_key not in LIVE_ACTIVE_ORDER_STATUSES and not provider_snapshot_status:
        provider_snapshot_status = status_key

    serialized_payload = dict(payload)
    serialized_payload["trading_status"] = status_key
    if provider_reconciliation:
        serialized_provider_reconciliation = dict(provider_reconciliation)
        if provider_snapshot_status:
            serialized_provider_reconciliation["snapshot_status"] = provider_snapshot_status
        serialized_payload["provider_reconciliation"] = serialized_provider_reconciliation

    cached_market_payload = (
        cached_market.extra_data if cached_market is not None and isinstance(cached_market.extra_data, dict) else {}
    )
    cached_market_slug = str(
        cached_market_payload.get("slug") or (cached_market.slug if cached_market is not None else "") or ""
    ).strip()
    cached_event_slug = str(
        cached_market_payload.get("event_slug") or cached_market_payload.get("eventSlug") or ""
    ).strip()
    cached_condition_id = str(
        cached_market_payload.get("condition_id")
        or cached_market_payload.get("conditionId")
        or (cached_market.condition_id if cached_market is not None else "")
        or ""
    ).strip()
    cached_market_id = str(cached_market_payload.get("id") or "").strip()

    existing_condition_id = str(
        serialized_payload.get("condition_id") or serialized_payload.get("conditionId") or ""
    ).strip()
    condition_id = existing_condition_id or cached_condition_id or _extract_order_condition_id(row)
    if condition_id:
        serialized_payload["condition_id"] = condition_id

    market_slug = str(
        serialized_payload.get("market_slug")
        or serialized_payload.get("marketSlug")
        or serialized_payload.get("slug")
        or ""
    ).strip()
    if not market_slug:
        market_slug = cached_market_slug
    if market_slug:
        serialized_payload["market_slug"] = market_slug
        if not str(serialized_payload.get("slug") or "").strip():
            serialized_payload["slug"] = market_slug

    event_slug = str(serialized_payload.get("event_slug") or serialized_payload.get("eventSlug") or "").strip()
    if not event_slug:
        event_slug = cached_event_slug
    if event_slug:
        serialized_payload["event_slug"] = event_slug

    existing_market_url = str(
        serialized_payload.get("market_url")
        or serialized_payload.get("marketUrl")
        or serialized_payload.get("url")
        or ""
    ).strip()
    existing_polymarket_url = str(
        serialized_payload.get("polymarket_url") or serialized_payload.get("polymarketUrl") or ""
    ).strip()
    market_url = existing_market_url or existing_polymarket_url
    if not market_url:
        market_url = (
            build_polymarket_market_url(
                event_slug=event_slug or None,
                market_slug=market_slug or None,
                market_id=cached_market_id or row.market_id,
                condition_id=condition_id or row.market_id,
            )
            or ""
        ).strip()
    if market_url:
        if not existing_market_url:
            serialized_payload["market_url"] = market_url
            serialized_payload["url"] = market_url
        if not existing_polymarket_url and "polymarket.com" in market_url.lower():
            serialized_payload["polymarket_url"] = market_url

    mark_updated_at = str(position_state.get("last_marked_at") or provider_reconciliation.get("reconciled_at") or "")

    unrealized_pnl = None
    if (
        status_key in {"submitted", "open", "executed"}
        and current_price is not None
        and current_price > 0
        and quantity > 0
    ):
        unrealized_pnl = (quantity * current_price) - filled_notional_display

    signal_payload = signal.payload_json if signal is not None and isinstance(signal.payload_json, dict) else {}
    direction_side, direction_label, yes_label, no_label = _resolve_direction_presentation(
        direction=row.direction,
        payload=payload,
        signal_payload=signal_payload,
        market_payload=cached_market_payload,
        market_id=row.market_id,
    )

    source_trade_payload = serialized_payload.get("source_trade")
    source_trade_payload = source_trade_payload if isinstance(source_trade_payload, dict) else {}
    strategy_context_payload = serialized_payload.get("strategy_context")
    strategy_context_payload = strategy_context_payload if isinstance(strategy_context_payload, dict) else {}
    copy_event_payload = strategy_context_payload.get("copy_event")
    copy_event_payload = copy_event_payload if isinstance(copy_event_payload, dict) else {}
    copy_attribution = serialized_payload.get("copy_attribution")
    copy_attribution = dict(copy_attribution) if isinstance(copy_attribution, dict) else {}
    source_wallet = _extract_copy_source_wallet_from_payload(serialized_payload)
    if source_wallet and not str(copy_attribution.get("source_wallet") or "").strip():
        copy_attribution["source_wallet"] = source_wallet

    source_price = safe_float(
        copy_attribution.get("source_price"),
        safe_float(source_trade_payload.get("price"), safe_float(copy_event_payload.get("price"), None)),
    )
    follower_effective_price = safe_float(
        copy_attribution.get("follower_effective_price"),
        average_fill_price if average_fill_price is not None and average_fill_price > 0 else safe_float(row.effective_price),
    )
    if source_price is not None and source_price > 0.0:
        copy_attribution["source_price"] = float(source_price)
    if follower_effective_price is not None and follower_effective_price > 0.0:
        copy_attribution["follower_effective_price"] = float(follower_effective_price)

    side = _extract_copy_side_from_payload(serialized_payload)
    if side and not str(copy_attribution.get("side") or "").strip():
        copy_attribution["side"] = side
    if (
        source_price is not None
        and source_price > 0.0
        and follower_effective_price is not None
        and follower_effective_price > 0.0
    ):
        slippage_bps = ((follower_effective_price - source_price) / source_price) * 10_000.0
        adverse_slippage_bps = slippage_bps
        if side == "BUY":
            adverse_slippage_bps = max(0.0, slippage_bps)
        elif side == "SELL":
            adverse_slippage_bps = max(0.0, -slippage_bps)
        copy_attribution["slippage_bps"] = float(slippage_bps)
        copy_attribution["adverse_slippage_bps"] = float(adverse_slippage_bps)

    if not str(copy_attribution.get("source_tx_hash") or "").strip():
        source_tx_hash = str(copy_event_payload.get("tx_hash") or source_trade_payload.get("tx_hash") or "").strip()
        if source_tx_hash:
            copy_attribution["source_tx_hash"] = source_tx_hash
    if "source_size" not in copy_attribution:
        source_size = safe_float(source_trade_payload.get("size"), safe_float(copy_event_payload.get("size"), None))
        if source_size is not None and source_size > 0.0:
            copy_attribution["source_size"] = float(source_size)
    if "source_notional_usd" not in copy_attribution:
        source_notional = safe_float(source_trade_payload.get("source_notional_usd"), None)
        if source_notional is not None and source_notional > 0.0:
            copy_attribution["source_notional_usd"] = float(source_notional)
    if "source_detection_latency_ms" not in copy_attribution:
        source_detection_latency_ms = safe_float(copy_event_payload.get("latency_ms"), None)
        if source_detection_latency_ms is not None and source_detection_latency_ms >= 0.0:
            copy_attribution["source_detection_latency_ms"] = float(source_detection_latency_ms)
    detected_at_text = str(
        copy_attribution.get("detected_at")
        or copy_event_payload.get("detected_at")
        or copy_event_payload.get("timestamp")
        or source_trade_payload.get("detected_at")
        or ""
    ).strip()
    if detected_at_text and "detected_at" not in copy_attribution:
        copy_attribution["detected_at"] = detected_at_text
    if detected_at_text and "copy_latency_ms" not in copy_attribution and row.created_at is not None:
        try:
            parsed_detected_at = datetime.fromisoformat(detected_at_text.replace("Z", "+00:00"))
        except Exception:
            parsed_detected_at = None
        if parsed_detected_at is not None:
            if parsed_detected_at.tzinfo is None:
                parsed_detected_at = parsed_detected_at.replace(tzinfo=timezone.utc)
            created_at_utc = row.created_at if row.created_at.tzinfo is not None else row.created_at.replace(tzinfo=timezone.utc)
            copy_latency_ms = max(0.0, (created_at_utc.astimezone(timezone.utc) - parsed_detected_at.astimezone(timezone.utc)).total_seconds() * 1000.0)
            copy_attribution["copy_latency_ms"] = float(copy_latency_ms)

    if copy_attribution:
        serialized_payload["copy_attribution"] = copy_attribution

    return {
        "id": row.id,
        "trader_id": row.trader_id,
        "signal_id": row.signal_id,
        "decision_id": row.decision_id,
        "source": row.source,
        "strategy_key": row.strategy_key,
        "strategy_version": int(row.strategy_version) if row.strategy_version is not None else None,
        "market_id": row.market_id,
        "market_question": row.market_question,
        "direction": row.direction,
        "direction_side": direction_side,
        "direction_label": direction_label,
        "yes_label": yes_label,
        "no_label": no_label,
        "mode": row.mode,
        "status": row.status,
        "notional_usd": row.notional_usd,
        "entry_price": row.entry_price,
        "effective_price": row.effective_price,
        "provider_order_id": str(payload.get("provider_order_id") or ""),
        "provider_clob_order_id": str(payload.get("provider_clob_order_id") or ""),
        "provider_snapshot_status": provider_snapshot_status,
        "filled_shares": float(max(0.0, filled_shares)),
        "filled_notional_usd": float(max(0.0, filled_notional_display)),
        "average_fill_price": float(average_fill_price)
        if average_fill_price is not None and average_fill_price > 0
        else None,
        "current_price": float(current_price) if current_price is not None and current_price > 0 else None,
        "mark_source": str(position_state.get("last_mark_source") or ""),
        "mark_updated_at": mark_updated_at,
        "unrealized_pnl": float(unrealized_pnl) if unrealized_pnl is not None else None,
        "edge_percent": row.edge_percent,
        "confidence": row.confidence,
        "actual_profit": row.actual_profit,
        "reason": row.reason,
        "close_trigger": close_trigger or None,
        "close_reason": close_reason or None,
        "payload": serialized_payload,
        "copy_attribution": copy_attribution,
        "error_message": row.error_message,
        "event_id": row.event_id,
        "trace_id": row.trace_id,
        "created_at": to_iso(row.created_at),
        "executed_at": to_iso(row.executed_at),
        "updated_at": to_iso(row.updated_at),
    }


def _serialize_execution_session(row: ExecutionSession) -> dict[str, Any]:
    return {
        "id": row.id,
        "trader_id": row.trader_id,
        "signal_id": row.signal_id,
        "decision_id": row.decision_id,
        "source": row.source,
        "strategy_key": row.strategy_key,
        "strategy_version": int(row.strategy_version) if row.strategy_version is not None else None,
        "mode": row.mode,
        "status": row.status,
        "policy": row.policy,
        "plan_id": row.plan_id,
        "market_ids": row.market_ids_json or [],
        "legs_total": int(row.legs_total or 0),
        "legs_completed": int(row.legs_completed or 0),
        "legs_failed": int(row.legs_failed or 0),
        "legs_open": int(row.legs_open or 0),
        "requested_notional_usd": float(row.requested_notional_usd or 0.0),
        "executed_notional_usd": float(row.executed_notional_usd or 0.0),
        "max_unhedged_notional_usd": float(row.max_unhedged_notional_usd or 0.0),
        "unhedged_notional_usd": float(row.unhedged_notional_usd or 0.0),
        "trace_id": row.trace_id,
        "started_at": to_iso(row.started_at),
        "completed_at": to_iso(row.completed_at),
        "expires_at": to_iso(row.expires_at),
        "error_message": row.error_message,
        "payload": row.payload_json or {},
        "created_at": to_iso(row.created_at),
        "updated_at": to_iso(row.updated_at),
    }


def _serialize_execution_leg(row: ExecutionSessionLeg) -> dict[str, Any]:
    return {
        "id": row.id,
        "session_id": row.session_id,
        "leg_index": int(row.leg_index or 0),
        "leg_id": row.leg_id,
        "market_id": row.market_id,
        "market_question": row.market_question,
        "token_id": row.token_id,
        "side": row.side,
        "outcome": row.outcome,
        "price_policy": row.price_policy,
        "time_in_force": row.time_in_force,
        "target_price": row.target_price,
        "requested_notional_usd": row.requested_notional_usd,
        "requested_shares": row.requested_shares,
        "filled_notional_usd": float(row.filled_notional_usd or 0.0),
        "filled_shares": float(row.filled_shares or 0.0),
        "avg_fill_price": row.avg_fill_price,
        "status": row.status,
        "last_error": row.last_error,
        "metadata": row.metadata_json or {},
        "created_at": to_iso(row.created_at),
        "updated_at": to_iso(row.updated_at),
    }


def _serialize_execution_order(row: ExecutionSessionOrder) -> dict[str, Any]:
    return {
        "id": row.id,
        "session_id": row.session_id,
        "leg_id": row.leg_id,
        "trader_order_id": row.trader_order_id,
        "provider_order_id": row.provider_order_id,
        "provider_clob_order_id": row.provider_clob_order_id,
        "action": row.action,
        "side": row.side,
        "price": row.price,
        "size": row.size,
        "notional_usd": row.notional_usd,
        "status": row.status,
        "reason": row.reason,
        "payload": row.payload_json or {},
        "error_message": row.error_message,
        "created_at": to_iso(row.created_at),
        "updated_at": to_iso(row.updated_at),
    }


def _serialize_execution_event(row: ExecutionSessionEvent) -> dict[str, Any]:
    return {
        "id": row.id,
        "session_id": row.session_id,
        "leg_id": row.leg_id,
        "event_type": row.event_type,
        "severity": row.severity,
        "message": row.message,
        "payload": row.payload_json or {},
        "created_at": to_iso(row.created_at),
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
        "created_at": to_iso(row.created_at),
    }


async def ensure_orchestrator_control(session: AsyncSession) -> TraderOrchestratorControl:
    row = await session.get(TraderOrchestratorControl, ORCHESTRATOR_CONTROL_ID)
    if row is None:
        row = TraderOrchestratorControl(
            id=ORCHESTRATOR_CONTROL_ID,
            is_enabled=False,
            is_paused=True,
            mode="shadow",
            run_interval_seconds=ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS,
            kill_switch=False,
            settings_json=_default_control_settings(),
            updated_at=_now(),
        )
        session.add(row)
        await _commit_with_retry(session)
        await session.refresh(row)
    else:
        normalized_settings = _normalize_control_settings(row.settings_json)
        if row.settings_json != normalized_settings:
            row.settings_json = normalized_settings
            row.updated_at = _now()
            await _commit_with_retry(session)
            await session.refresh(row)
    if not isinstance(row.settings_json, dict):
        row.settings_json = _default_control_settings()
        row.updated_at = _now()
        await _commit_with_retry(session)
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
            interval_seconds=ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS,
            stats_json={},
            updated_at=_now(),
        )
        session.add(row)
        await _commit_with_retry(session)
        await session.refresh(row)
    return row


async def read_orchestrator_control(session: AsyncSession) -> dict[str, Any]:
    return _serialize_control(await ensure_orchestrator_control(session))


async def read_orchestrator_snapshot(session: AsyncSession) -> dict[str, Any]:
    return _serialize_snapshot(await ensure_orchestrator_snapshot(session))


async def enforce_manual_start_on_startup(session: AsyncSession) -> dict[str, Any]:
    control = await update_orchestrator_control(
        session,
        is_enabled=False,
        is_paused=True,
        mode="shadow",
        requested_run_at=None,
    )
    await write_orchestrator_snapshot(
        session,
        running=False,
        enabled=False,
        current_activity="Stopped on startup; manual start required",
        interval_seconds=int(control.get("run_interval_seconds") or ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS),
        last_error=None,
    )
    return control


async def update_orchestrator_control(session: AsyncSession, **updates: Any) -> dict[str, Any]:
    row = await ensure_orchestrator_control(session)
    payload: dict[str, Any] = {}

    for field in ("is_enabled", "is_paused", "mode", "run_interval_seconds", "kill_switch"):
        if field in updates and updates[field] is not None:
            payload[field] = updates[field]

    if "requested_run_at" in updates:
        payload["requested_run_at"] = updates["requested_run_at"]

    if isinstance(updates.get("settings_json"), dict):
        merged = dict(_normalize_control_settings(row.settings_json))
        merged.update(updates["settings_json"])
        payload["settings_json"] = _normalize_control_settings(merged)

    payload["updated_at"] = _now()

    for attempt in range(DB_RETRY_ATTEMPTS):
        try:
            await session.execute(
                sa_update(TraderOrchestratorControl)
                .where(TraderOrchestratorControl.id == ORCHESTRATOR_CONTROL_ID)
                .values(**payload)
            )
            await session.commit()
            break
        except OperationalError as exc:
            await session.rollback()
            is_locked = _is_retryable_db_error(exc)
            is_last = attempt >= DB_RETRY_ATTEMPTS - 1
            if not is_locked or is_last:
                raise
            await asyncio.sleep(_db_retry_delay(attempt))

    refreshed = await session.get(TraderOrchestratorControl, ORCHESTRATOR_CONTROL_ID)
    if refreshed is None:
        refreshed = await ensure_orchestrator_control(session)
    return _serialize_control(refreshed)


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
    await _commit_with_retry(
        session,
        retry_attempts=2,
        base_delay_seconds=0.05,
        max_delay_seconds=0.1,
    )
    await session.refresh(row)
    snapshot_data = _serialize_snapshot(row)

    # Publish orchestrator status event.
    try:
        await event_bus.publish("trader_orchestrator_status", snapshot_data)
    except Exception:
        pass  # fire-and-forget

    return snapshot_data


def list_trader_templates() -> list[dict[str, Any]]:
    return [
        {
            "id": template["id"],
            "name": template["name"],
            "description": template.get("description"),
            "source_configs": _normalize_source_configs(template.get("source_configs") or []),
            "interval_seconds": int(template.get("interval_seconds", 60) or 60),
            "risk_limits": template.get("risk_limits", {}),
        }
        for template in TRADER_TEMPLATES
    ]


async def _normalize_trader_payload(
    session: AsyncSession,
    payload: dict[str, Any],
) -> dict[str, Any]:
    source_configs = _normalize_source_configs(payload.get("source_configs"))
    await _validate_source_configs(session, source_configs)

    risk_limits = StrategySDK.validate_trader_risk_config(payload.get("risk_limits"))
    metadata = StrategySDK.validate_trader_runtime_metadata(payload.get("metadata"))

    return {
        "name": str(payload.get("name") or "").strip(),
        "description": payload.get("description"),
        "mode": _normalize_trader_mode(payload.get("mode")),
        "source_configs": source_configs,
        "risk_limits": risk_limits,
        "metadata": metadata,
        "is_enabled": bool(payload.get("is_enabled", True)),
        "is_paused": bool(payload.get("is_paused", False)),
        "interval_seconds": max(1, min(86400, safe_int(payload.get("interval_seconds"), 60))),
    }


async def list_traders(session: AsyncSession, mode: Optional[str] = None) -> list[dict[str, Any]]:
    query = select(Trader)
    if mode is not None:
        mode_key = _normalize_trader_mode(mode)
        if mode_key == "shadow":
            query = query.where(func.lower(func.coalesce(Trader.mode, "")).in_(("shadow", "paper", "")))
        else:
            query = query.where(func.lower(func.coalesce(Trader.mode, "")) == mode_key)
    rows = (await session.execute(query.order_by(Trader.name.asc()))).scalars().all()
    return [_serialize_trader(row) for row in rows]


async def get_trader(session: AsyncSession, trader_id: str) -> Optional[dict[str, Any]]:
    row = await session.get(Trader, trader_id)
    return _serialize_trader(row) if row else None


async def seed_default_traders(session: AsyncSession) -> None:
    count = int((await session.execute(select(func.count(Trader.id)))).scalar() or 0)
    if count > 0:
        return

    for template in TRADER_TEMPLATES:
        source_configs = _normalize_source_configs(template.get("source_configs") or [])
        session.add(
            Trader(
                id=_new_id(),
                name=template["name"],
                description=template.get("description"),
                source_configs_json=source_configs,
                risk_limits_json=template.get("risk_limits") or {},
                metadata_json={"template_id": template["id"]},
                mode="shadow",
                is_enabled=True,
                is_paused=False,
                interval_seconds=int(template.get("interval_seconds", 60) or 60),
                created_at=_now(),
                updated_at=_now(),
            )
        )
    await _commit_with_retry(session)


async def create_trader(session: AsyncSession, payload: dict[str, Any]) -> dict[str, Any]:
    create_payload = dict(payload or {})
    copy_from_trader_id = str(create_payload.pop("copy_from_trader_id", "") or "").strip()
    if copy_from_trader_id:
        source_row = await session.get(Trader, copy_from_trader_id)
        if source_row is None:
            raise ValueError("Source trader not found")
        source_trader = _serialize_trader(source_row)
        copied_payload: dict[str, Any] = {
            "description": source_trader.get("description"),
            "mode": source_trader.get("mode", "shadow"),
            "source_configs": copy.deepcopy(source_trader.get("source_configs", [])),
            "interval_seconds": int(source_trader.get("interval_seconds", 60) or 60),
            "risk_limits": copy.deepcopy(source_trader.get("risk_limits", {})),
            "metadata": copy.deepcopy(source_trader.get("metadata", {})),
            "is_enabled": bool(source_trader.get("is_enabled", True)),
            "is_paused": bool(source_trader.get("is_paused", False)),
        }
        copied_payload.update(create_payload)
        create_payload = copied_payload

    normalized = await _normalize_trader_payload(session, create_payload)
    if not normalized["name"]:
        raise ValueError("Trader name is required")

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
        source_configs_json=normalized["source_configs"],
        risk_limits_json=normalized["risk_limits"],
        metadata_json=normalized["metadata"],
        mode=normalized["mode"],
        is_enabled=normalized["is_enabled"],
        is_paused=normalized["is_paused"],
        interval_seconds=normalized["interval_seconds"],
        created_at=_now(),
        updated_at=_now(),
    )
    session.add(row)
    await _commit_with_retry(session)
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
        "source_configs": template.get("source_configs", []),
        "interval_seconds": template.get("interval_seconds", 60),
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

    normalized = await _normalize_trader_payload(session, {**_serialize_trader(row), **payload})
    if "name" in payload:
        row.name = normalized["name"]
    if "description" in payload:
        row.description = normalized["description"]
    if "source_configs" in payload:
        row.source_configs_json = normalized["source_configs"]
    if "risk_limits" in payload:
        row.risk_limits_json = normalized["risk_limits"]
    if "metadata" in payload:
        row.metadata_json = normalized["metadata"]
    if "mode" in payload:
        row.mode = normalized["mode"]
    if "is_enabled" in payload:
        row.is_enabled = bool(payload.get("is_enabled"))
    if "is_paused" in payload:
        row.is_paused = bool(payload.get("is_paused"))
    if "interval_seconds" in payload:
        row.interval_seconds = normalized["interval_seconds"]

    row.updated_at = _now()
    await _commit_with_retry(session)
    await session.refresh(row)
    return _serialize_trader(row)


async def delete_trader(session: AsyncSession, trader_id: str, *, force: bool = False) -> bool:
    row = await session.get(Trader, trader_id)
    if row is None:
        return False

    await sync_trader_position_inventory(session, trader_id=trader_id)
    open_position_summary = await get_open_position_summary_for_trader(session, trader_id)
    open_order_summary = await get_open_order_summary_for_trader(session, trader_id)

    open_live_positions = int(open_position_summary.get("live", 0))
    open_shadow_positions = int(open_position_summary.get("shadow", 0))
    open_other_positions = int(open_position_summary.get("other", 0))
    open_total_positions = int(open_position_summary.get("total", 0))
    open_live_orders = int(open_order_summary.get("live", 0))
    open_shadow_orders = int(open_order_summary.get("shadow", 0))
    open_other_orders = int(open_order_summary.get("other", 0))
    open_total_orders = int(open_order_summary.get("total", 0))

    if (
        open_live_positions > 0
        or open_shadow_positions > 0
        or open_other_positions > 0
        or open_live_orders > 0
        or open_shadow_orders > 0
        or open_other_orders > 0
    ) and not force:
        raise ValueError(
            f"Trader {trader_id} has active exposure: "
            f"{open_live_positions} live open position(s), "
            f"{open_shadow_positions} shadow open position(s), "
            f"{open_other_positions} unknown-mode open position(s), "
            f"{open_live_orders} live active order(s), and "
            f"{open_shadow_orders} shadow active order(s), and "
            f"{open_other_orders} unknown-mode active order(s). "
            "Flatten active exposure first, or pass force=True to delete anyway."
        )

    if force and open_total_orders > 0:
        # Force-delete is an explicit operator override. Skip provider calls and
        # close local active orders immediately so manual Polymarket cleanup does
        # not block account hygiene.
        now = _now()
        active_rows = list(
            (
                await session.execute(
                    select(TraderOrder).where(
                        TraderOrder.trader_id == trader_id,
                        func.lower(func.coalesce(TraderOrder.status, "")).in_(tuple(OPEN_ORDER_STATUSES)),
                    )
                )
            )
            .scalars()
            .all()
        )
        for active_row in active_rows:
            payload = dict(active_row.payload_json or {})
            payload["cleanup"] = {
                "previous_status": str(active_row.status or ""),
                "target_status": "cancelled",
                "reason": "force_delete_override",
                "performed_at": to_iso(now),
            }
            if _normalize_mode_key(active_row.mode) == "live":
                payload["provider_cancel"] = {
                    "skipped": True,
                    "reason": "force_delete_override",
                    "performed_at": to_iso(now),
                }
            active_row.status = "cancelled"
            active_row.updated_at = now
            active_row.payload_json = payload
            existing_reason = str(active_row.reason or "").strip()
            active_row.reason = (
                f"{existing_reason} | cleanup:force_delete_override"
                if existing_reason
                else "cleanup:force_delete_override"
            )
            import services.trader_hot_state as _hs
            _hs.record_order_cancelled(
                trader_id=str(active_row.trader_id or ""),
                mode=str(active_row.mode or ""),
                order_id=str(active_row.id or ""),
                market_id=str(active_row.market_id or ""),
                source=str(active_row.source or ""),
                copy_source_wallet=_extract_copy_source_wallet_from_payload(payload),
            )
        await _commit_with_retry(session)
        await sync_trader_position_inventory(session, trader_id=trader_id)
    elif force and open_total_positions > 0:
        await sync_trader_position_inventory(session, trader_id=trader_id)

    await session.delete(row)
    await _commit_with_retry(session)
    return True


async def set_trader_paused(session: AsyncSession, trader_id: str, paused: bool) -> Optional[dict[str, Any]]:
    row = await session.get(Trader, trader_id)
    if row is None:
        return None
    row.is_paused = bool(paused)
    row.updated_at = _now()
    await _commit_with_retry(session)
    await session.refresh(row)
    return _serialize_trader(row)


async def request_trader_run(session: AsyncSession, trader_id: str) -> Optional[dict[str, Any]]:
    row = await session.get(Trader, trader_id)
    if row is None:
        return None
    row.requested_run_at = _now()
    row.updated_at = _now()
    await _commit_with_retry(session)
    await session.refresh(row)
    return _serialize_trader(row)


async def clear_trader_run_request(session: AsyncSession, trader_id: str) -> None:
    row = await session.get(Trader, trader_id)
    if row is None:
        return
    row.requested_run_at = None
    row.updated_at = _now()
    await _commit_with_retry(session)


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
    await _commit_with_retry(session)


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
    commit: bool = True,
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
    serialized_row = _serialize_event(row)
    if not commit:
        try:
            await event_bus.publish("trader_event", serialized_row)
        except Exception:
            pass
    if commit:
        await _commit_with_retry(session)
        try:
            await event_bus.publish("trader_event", serialized_row)
        except Exception:
            pass
    else:
        await session.flush()

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
    trader_ids: Optional[list[str]] = None,
    decision: Optional[str] = None,
    limit: int = 200,
    per_trader_limit: Optional[int] = None,
) -> list[TraderDecision]:
    normalized_trader_ids: list[str] = []
    if trader_ids:
        seen: set[str] = set()
        for raw in trader_ids:
            value = str(raw or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            normalized_trader_ids.append(value)

    if trader_id:
        query = select(TraderDecision).where(TraderDecision.trader_id == trader_id)
        if decision:
            query = query.where(TraderDecision.decision == decision)
        query = query.order_by(desc(TraderDecision.created_at), desc(TraderDecision.id))
        query = query.limit(max(1, min(limit, 1000)))
        return list((await session.execute(query)).scalars().all())

    max_limit = 5000 if normalized_trader_ids else 1000
    capped_limit = max(1, min(limit, max_limit))
    capped_per_trader_limit = (
        max(1, min(int(per_trader_limit or 1), 1000))
        if per_trader_limit is not None and normalized_trader_ids
        else None
    )

    if capped_per_trader_limit is not None:
        ranked_decisions = (
            select(
                TraderDecision.id.label("id"),
                func.row_number()
                .over(
                    partition_by=TraderDecision.trader_id,
                    order_by=(TraderDecision.created_at.desc(), TraderDecision.id.desc()),
                )
                .label("rn"),
            )
            .where(TraderDecision.trader_id.in_(normalized_trader_ids))
        )
        if decision:
            ranked_decisions = ranked_decisions.where(TraderDecision.decision == decision)
        ranked_subquery = ranked_decisions.subquery()
        query = (
            select(TraderDecision)
            .join(ranked_subquery, TraderDecision.id == ranked_subquery.c.id)
            .where(ranked_subquery.c.rn <= capped_per_trader_limit)
            .order_by(desc(TraderDecision.created_at), desc(TraderDecision.id))
            .limit(capped_limit)
        )
        return list((await session.execute(query)).scalars().all())

    query = select(TraderDecision).order_by(desc(TraderDecision.created_at), desc(TraderDecision.id))
    if normalized_trader_ids:
        query = query.where(TraderDecision.trader_id.in_(normalized_trader_ids))
    if decision:
        query = query.where(TraderDecision.decision == decision)
    query = query.limit(capped_limit)
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
    query = query.limit(max(1, min(limit, 5000)))
    return list((await session.execute(query)).scalars().all())


async def get_trader_decision_detail(session: AsyncSession, decision_id: str) -> Optional[dict[str, Any]]:
    row = await session.get(TraderDecision, decision_id)
    if row is None:
        return None
    signal_row = None
    if row.signal_id:
        signal_row = await session.get(TradeSignal, row.signal_id)

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
        "decision": _serialize_decision_with_signal(row, signal_row),
        "checks": [
            {
                "id": check.id,
                "check_key": check.check_key,
                "check_label": check.check_label,
                "passed": bool(check.passed),
                "score": check.score,
                "detail": check.detail,
                "payload": check.payload_json or {},
                "created_at": to_iso(check.created_at),
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
    strategy_version: int | None = None,
    decision: str,
    reason: Optional[str] = None,
    score: Optional[float] = None,
    checks_summary: Optional[dict[str, Any]] = None,
    risk_snapshot: Optional[dict[str, Any]] = None,
    payload: Optional[dict[str, Any]] = None,
    trace_id: Optional[str] = None,
    commit: bool = True,
) -> TraderDecision:
    row = TraderDecision(
        id=_new_id(),
        trader_id=trader_id,
        signal_id=signal.id,
        source=str(signal.source),
        strategy_key=str(strategy_key),
        strategy_version=int(strategy_version) if strategy_version is not None else None,
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
    serialized_row = _serialize_decision(row)
    if not commit:
        try:
            await event_bus.publish("trader_decision", serialized_row)
        except Exception:
            pass
    if commit:
        await _commit_with_retry(session)
        await session.refresh(row)
        try:
            await event_bus.publish("trader_decision", serialized_row)
        except Exception:
            pass
    else:
        await session.flush()

    return row


async def update_trader_decision(
    session: AsyncSession,
    *,
    decision_id: str,
    decision: str | None = None,
    reason: str | None = None,
    payload_patch: dict[str, Any] | None = None,
    commit: bool = True,
) -> TraderDecision | None:
    row = await session.get(TraderDecision, decision_id)
    if row is None:
        return None

    if decision is not None:
        row.decision = str(decision)
    if reason is not None:
        row.reason = reason
    if payload_patch:
        merged_payload = dict(row.payload_json or {})
        merged_payload.update(payload_patch)
        row.payload_json = merged_payload

    serialized_row = _serialize_decision(row)
    if not commit:
        try:
            await event_bus.publish("trader_decision", serialized_row)
        except Exception:
            pass
    if commit:
        await _commit_with_retry(session)
        await session.refresh(row)
        try:
            await event_bus.publish("trader_decision", serialized_row)
        except Exception:
            pass
    else:
        await session.flush()

    return row


async def create_trader_decision_checks(
    session: AsyncSession,
    *,
    decision_id: str,
    checks: list[dict[str, Any]],
    commit: bool = True,
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
    if commit:
        await _commit_with_retry(session)
    else:
        await session.flush()


async def create_trader_order(
    session: AsyncSession,
    *,
    trader_id: str,
    signal: TradeSignal,
    decision_id: Optional[str],
    strategy_key: str | None,
    strategy_version: int | None,
    mode: str,
    status: str,
    notional_usd: Optional[float],
    effective_price: Optional[float],
    reason: Optional[str],
    payload: Optional[dict[str, Any]],
    error_message: Optional[str] = None,
    trace_id: Optional[str] = None,
    commit: bool = True,
) -> TraderOrder:
    now = _now()
    order_payload = _sync_order_runtime_payload(
        payload=dict(payload or {}),
        status=status,
        now=now,
    )
    signal_payload = signal.payload_json if isinstance(getattr(signal, "payload_json", None), dict) else {}
    source_trade_payload = signal_payload.get("source_trade")
    source_trade_payload = source_trade_payload if isinstance(source_trade_payload, dict) else {}
    strategy_context_payload = signal_payload.get("strategy_context")
    strategy_context_payload = strategy_context_payload if isinstance(strategy_context_payload, dict) else {}
    copy_event_payload = strategy_context_payload.get("copy_event")
    copy_event_payload = copy_event_payload if isinstance(copy_event_payload, dict) else {}

    if source_trade_payload and not isinstance(order_payload.get("source_trade"), dict):
        order_payload["source_trade"] = dict(source_trade_payload)

    strategy_context_order_payload = (
        dict(order_payload.get("strategy_context"))
        if isinstance(order_payload.get("strategy_context"), dict)
        else {}
    )
    if strategy_context_payload and not strategy_context_order_payload:
        strategy_context_order_payload = dict(strategy_context_payload)
    elif copy_event_payload and not isinstance(strategy_context_order_payload.get("copy_event"), dict):
        strategy_context_order_payload["copy_event"] = dict(copy_event_payload)
    if strategy_context_order_payload:
        order_payload["strategy_context"] = strategy_context_order_payload

    signal_source_key = str(getattr(signal, "source", "") or "").strip().lower()
    strategy_key_normalized = str(strategy_key or "").strip().lower()
    has_copy_payload = bool(source_trade_payload) or bool(copy_event_payload)
    if signal_source_key == "traders" and (strategy_key_normalized == "traders_copy_trade" or has_copy_payload):
        copy_attribution = dict(order_payload.get("copy_attribution") or {})
        source_wallet = _extract_copy_source_wallet_from_payload(order_payload)
        if source_wallet and not str(copy_attribution.get("source_wallet") or "").strip():
            copy_attribution["source_wallet"] = source_wallet
        if "side" not in copy_attribution:
            copy_side = _extract_copy_side_from_payload(order_payload)
            if copy_side:
                copy_attribution["side"] = copy_side

        source_price = safe_float(
            copy_attribution.get("source_price"),
            safe_float(source_trade_payload.get("price"), safe_float(copy_event_payload.get("price"), None)),
        )
        if source_price is not None and source_price > 0.0:
            copy_attribution["source_price"] = float(source_price)

        source_size = safe_float(
            source_trade_payload.get("size"),
            safe_float(copy_event_payload.get("size"), None),
        )
        if source_size is not None and source_size > 0.0 and "source_size" not in copy_attribution:
            copy_attribution["source_size"] = float(source_size)
        source_notional_usd = safe_float(source_trade_payload.get("source_notional_usd"), None)
        if source_notional_usd is not None and source_notional_usd > 0.0 and "source_notional_usd" not in copy_attribution:
            copy_attribution["source_notional_usd"] = float(source_notional_usd)

        source_tx_hash = str(copy_event_payload.get("tx_hash") or source_trade_payload.get("tx_hash") or "").strip()
        if source_tx_hash and not str(copy_attribution.get("source_tx_hash") or "").strip():
            copy_attribution["source_tx_hash"] = source_tx_hash
        source_order_hash = str(copy_event_payload.get("order_hash") or source_trade_payload.get("order_hash") or "").strip()
        if source_order_hash and not str(copy_attribution.get("source_order_hash") or "").strip():
            copy_attribution["source_order_hash"] = source_order_hash

        source_detection_latency_ms = safe_float(copy_event_payload.get("latency_ms"), None)
        if source_detection_latency_ms is not None and source_detection_latency_ms >= 0.0:
            copy_attribution["source_detection_latency_ms"] = float(source_detection_latency_ms)

        detected_at_text = str(
            copy_event_payload.get("detected_at")
            or source_trade_payload.get("detected_at")
            or copy_event_payload.get("timestamp")
            or ""
        ).strip()
        if detected_at_text and "detected_at" not in copy_attribution:
            copy_attribution["detected_at"] = detected_at_text
        if detected_at_text and "copy_latency_ms" not in copy_attribution:
            try:
                parsed_detected_at = datetime.fromisoformat(detected_at_text.replace("Z", "+00:00"))
            except Exception:
                parsed_detected_at = None
            if parsed_detected_at is not None:
                if parsed_detected_at.tzinfo is None:
                    parsed_detected_at = parsed_detected_at.replace(tzinfo=timezone.utc)
                copy_latency_ms = max(
                    0.0,
                    (now.astimezone(timezone.utc) - parsed_detected_at.astimezone(timezone.utc)).total_seconds()
                    * 1000.0,
                )
                copy_attribution["copy_latency_ms"] = float(copy_latency_ms)

        follower_effective_price = safe_float(
            effective_price,
            safe_float(getattr(signal, "effective_price", None), safe_float(getattr(signal, "entry_price", None), None)),
        )
        if follower_effective_price is not None and follower_effective_price > 0.0:
            copy_attribution["follower_effective_price"] = float(follower_effective_price)
        if (
            source_price is not None
            and source_price > 0.0
            and follower_effective_price is not None
            and follower_effective_price > 0.0
        ):
            slippage_bps = ((follower_effective_price - source_price) / source_price) * 10_000.0
            side = str(copy_attribution.get("side") or "").strip().upper()
            adverse_slippage_bps = slippage_bps
            if side == "BUY":
                adverse_slippage_bps = max(0.0, slippage_bps)
            elif side == "SELL":
                adverse_slippage_bps = max(0.0, -slippage_bps)
            copy_attribution["slippage_bps"] = float(slippage_bps)
            copy_attribution["adverse_slippage_bps"] = float(adverse_slippage_bps)

        order_payload["copy_attribution"] = copy_attribution

    row = TraderOrder(
        id=_new_id(),
        trader_id=trader_id,
        signal_id=signal.id,
        decision_id=decision_id,
        source=str(signal.source),
        strategy_key=str(strategy_key or "").strip().lower() or None,
        strategy_version=int(strategy_version) if strategy_version is not None else None,
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
        payload_json=order_payload,
        error_message=error_message,
        trace_id=trace_id,
        created_at=now,
        executed_at=now if status in {"executed", "open"} else None,
        updated_at=now,
    )
    session.add(row)
    serialized_row = _serialize_order(row)
    if not commit:
        try:
            await event_bus.publish("trader_order", serialized_row)
        except Exception:
            pass
    await session.flush()
    if _is_active_order_status(mode, status):
        await sync_trader_position_inventory(
            session,
            trader_id=trader_id,
            mode=str(mode),
            commit=False,
        )
    if commit:
        await _commit_with_retry(session)
        await session.refresh(row)
        try:
            await event_bus.publish("trader_order", serialized_row)
        except Exception:
            pass

    return row


async def _refresh_execution_session_rollups(
    session: AsyncSession,
    *,
    session_id: str,
) -> ExecutionSession | None:
    row = await session.get(ExecutionSession, session_id)
    if row is None:
        return None

    legs = (
        (await session.execute(select(ExecutionSessionLeg).where(ExecutionSessionLeg.session_id == session_id)))
        .scalars()
        .all()
    )

    total = len(legs)
    completed = 0
    failed = 0
    open_count = 0
    buy_notional = 0.0
    sell_notional = 0.0
    executed_notional = 0.0

    for leg in legs:
        status_key = str(leg.status or "").strip().lower()
        leg_notional = abs(safe_float(leg.filled_notional_usd, 0.0))
        executed_notional += leg_notional
        side_key = str(leg.side or "").strip().lower()
        if side_key == "sell":
            sell_notional += leg_notional
        else:
            buy_notional += leg_notional

        if status_key == "completed":
            completed += 1
        elif status_key in {"failed", "cancelled", "expired"}:
            failed += 1
        elif status_key in {"open", "submitted", "placing", "working", "partial", "hedging", "pending"}:
            open_count += 1

    row.legs_total = total
    row.legs_completed = completed
    row.legs_failed = failed
    row.legs_open = open_count
    row.executed_notional_usd = executed_notional
    row.unhedged_notional_usd = abs(buy_notional - sell_notional)
    row.updated_at = _now()
    await session.flush()
    return row


async def create_execution_session(
    session: AsyncSession,
    *,
    trader_id: str,
    signal: TradeSignal,
    decision_id: str | None,
    strategy_key: str | None,
    strategy_version: int | None,
    mode: str,
    policy: str,
    plan_id: str | None,
    legs: list[dict[str, Any]],
    requested_notional_usd: float,
    max_unhedged_notional_usd: float,
    expires_at: datetime | None,
    payload: dict[str, Any] | None = None,
    trace_id: str | None = None,
    commit: bool = True,
) -> ExecutionSession:
    row = ExecutionSession(
        id=_new_id(),
        trader_id=trader_id,
        signal_id=signal.id,
        decision_id=decision_id,
        source=str(signal.source),
        strategy_key=str(strategy_key or "") or None,
        strategy_version=int(strategy_version) if strategy_version is not None else None,
        mode=str(mode),
        status="pending",
        policy=str(policy or "SINGLE_LEG"),
        plan_id=str(plan_id or "") or None,
        market_ids_json=sorted(
            {str(leg.get("market_id") or "").strip() for leg in legs if str(leg.get("market_id") or "").strip()}
        ),
        legs_total=0,
        legs_completed=0,
        legs_failed=0,
        legs_open=0,
        requested_notional_usd=float(max(0.0, requested_notional_usd)),
        executed_notional_usd=0.0,
        max_unhedged_notional_usd=float(max(0.0, max_unhedged_notional_usd)),
        unhedged_notional_usd=0.0,
        trace_id=trace_id,
        started_at=_now(),
        completed_at=None,
        expires_at=expires_at,
        payload_json=payload or {},
        created_at=_now(),
        updated_at=_now(),
    )
    session.add(row)
    await session.flush()

    for index, leg in enumerate(legs):
        target_price = safe_float(leg.get("limit_price"), None)
        requested_notional = safe_float(leg.get("requested_notional_usd"), None)
        requested_shares = safe_float(leg.get("requested_shares"), None)
        session.add(
            ExecutionSessionLeg(
                id=_new_id(),
                session_id=row.id,
                leg_index=index,
                leg_id=str(leg.get("leg_id") or f"leg_{index + 1}"),
                market_id=str(leg.get("market_id") or signal.market_id),
                market_question=str(leg.get("market_question") or signal.market_question or ""),
                token_id=str(leg.get("token_id") or "").strip() or None,
                side=str(leg.get("side") or "buy"),
                outcome=str(leg.get("outcome") or "").strip() or None,
                price_policy=str(leg.get("price_policy") or "maker_limit"),
                time_in_force=str(leg.get("time_in_force") or "GTC"),
                post_only=bool(leg.get("post_only", False)),
                target_price=target_price,
                requested_notional_usd=requested_notional,
                requested_shares=requested_shares,
                filled_notional_usd=0.0,
                filled_shares=0.0,
                avg_fill_price=None,
                status="pending",
                last_error=None,
                metadata_json=dict(leg.get("metadata") or {}),
                created_at=_now(),
                updated_at=_now(),
            )
        )

    await session.flush()
    await _refresh_execution_session_rollups(session, session_id=row.id)
    if commit:
        await _commit_with_retry(session)
        await session.refresh(row)

    try:
        await event_bus.publish("execution_session", _serialize_execution_session(row))
    except Exception:
        pass

    return row


async def list_execution_sessions(
    session: AsyncSession,
    *,
    trader_id: str | None = None,
    status: str | None = None,
    limit: int = 200,
) -> list[ExecutionSession]:
    query = select(ExecutionSession).order_by(desc(ExecutionSession.created_at))
    if trader_id:
        query = query.where(ExecutionSession.trader_id == trader_id)
    if status:
        query = query.where(func.lower(func.coalesce(ExecutionSession.status, "")) == str(status).strip().lower())
    query = query.limit(max(1, min(limit, 1000)))
    return list((await session.execute(query)).scalars().all())


async def list_active_execution_sessions(
    session: AsyncSession,
    *,
    trader_id: str | None = None,
    mode: str | None = None,
    limit: int = 500,
) -> list[ExecutionSession]:
    query = select(ExecutionSession).where(
        func.lower(func.coalesce(ExecutionSession.status, "")).in_(tuple(ACTIVE_EXECUTION_SESSION_STATUSES))
    )
    if trader_id:
        query = query.where(ExecutionSession.trader_id == trader_id)
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            query = query.where(func.lower(func.coalesce(ExecutionSession.mode, "")).not_in(list(_TRADER_MODES)))
        else:
            query = query.where(func.lower(func.coalesce(ExecutionSession.mode, "")) == mode_key)
    query = query.order_by(ExecutionSession.created_at.asc()).limit(max(1, min(limit, 2000)))
    return list((await session.execute(query)).scalars().all())


async def get_execution_session_detail(
    session: AsyncSession,
    session_id: str,
) -> dict[str, Any] | None:
    row = await session.get(ExecutionSession, session_id)
    if row is None:
        return None
    legs = (
        (
            await session.execute(
                select(ExecutionSessionLeg)
                .where(ExecutionSessionLeg.session_id == session_id)
                .order_by(ExecutionSessionLeg.leg_index.asc())
            )
        )
        .scalars()
        .all()
    )
    orders = (
        (
            await session.execute(
                select(ExecutionSessionOrder)
                .where(ExecutionSessionOrder.session_id == session_id)
                .order_by(desc(ExecutionSessionOrder.created_at))
            )
        )
        .scalars()
        .all()
    )
    events = (
        (
            await session.execute(
                select(ExecutionSessionEvent)
                .where(ExecutionSessionEvent.session_id == session_id)
                .order_by(desc(ExecutionSessionEvent.created_at))
            )
        )
        .scalars()
        .all()
    )
    return {
        "session": _serialize_execution_session(row),
        "legs": [_serialize_execution_leg(leg) for leg in legs],
        "orders": [_serialize_execution_order(order) for order in orders],
        "events": [_serialize_execution_event(event) for event in events],
    }


async def update_execution_session_status(
    session: AsyncSession,
    *,
    session_id: str,
    status: str,
    error_message: str | None = None,
    payload_patch: dict[str, Any] | None = None,
    commit: bool = True,
) -> ExecutionSession | None:
    row = await session.get(ExecutionSession, session_id)
    if row is None:
        return None
    now = _now()
    row.status = str(status)
    row.error_message = error_message
    if payload_patch:
        merged = dict(row.payload_json or {})
        merged.update(payload_patch)
        row.payload_json = merged
    if str(status).strip().lower() in TERMINAL_EXECUTION_SESSION_STATUSES:
        row.completed_at = now
    row.updated_at = now
    await _refresh_execution_session_rollups(session, session_id=session_id)
    if commit:
        await _commit_with_retry(session)
        await session.refresh(row)

    try:
        await event_bus.publish("execution_session", _serialize_execution_session(row))
    except Exception:
        pass

    return row


async def update_execution_leg(
    session: AsyncSession,
    *,
    leg_row_id: str,
    status: str | None = None,
    filled_notional_usd: float | None = None,
    filled_shares: float | None = None,
    avg_fill_price: float | None = None,
    last_error: str | None = None,
    metadata_patch: dict[str, Any] | None = None,
    commit: bool = True,
) -> ExecutionSessionLeg | None:
    row = await session.get(ExecutionSessionLeg, leg_row_id)
    if row is None:
        return None
    if status is not None:
        row.status = str(status)
    if filled_notional_usd is not None:
        row.filled_notional_usd = float(max(0.0, filled_notional_usd))
    if filled_shares is not None:
        row.filled_shares = float(max(0.0, filled_shares))
    if avg_fill_price is not None:
        row.avg_fill_price = float(avg_fill_price)
    if last_error is not None:
        row.last_error = last_error
    if metadata_patch:
        merged = dict(row.metadata_json or {})
        merged.update(metadata_patch)
        row.metadata_json = merged
    row.updated_at = _now()

    session_row = await _refresh_execution_session_rollups(session, session_id=row.session_id)
    if commit:
        await _commit_with_retry(session)
        await session.refresh(row)
        if session_row is not None:
            await session.refresh(session_row)

    try:
        await event_bus.publish("execution_leg", _serialize_execution_leg(row))
    except Exception:
        pass
    if session_row is not None:
        try:
            await event_bus.publish("execution_session", _serialize_execution_session(session_row))
        except Exception:
            pass

    return row


async def create_execution_session_order(
    session: AsyncSession,
    *,
    session_id: str,
    leg_id: str,
    trader_order_id: str | None,
    provider_order_id: str | None,
    provider_clob_order_id: str | None,
    action: str,
    side: str,
    price: float | None,
    size: float | None,
    notional_usd: float | None,
    status: str,
    reason: str | None = None,
    payload: dict[str, Any] | None = None,
    error_message: str | None = None,
    commit: bool = True,
) -> ExecutionSessionOrder:
    row = ExecutionSessionOrder(
        id=_new_id(),
        session_id=session_id,
        leg_id=leg_id,
        trader_order_id=trader_order_id,
        provider_order_id=provider_order_id,
        provider_clob_order_id=provider_clob_order_id,
        action=str(action),
        side=str(side),
        price=price,
        size=size,
        notional_usd=notional_usd,
        status=str(status),
        reason=reason,
        payload_json=payload or {},
        error_message=error_message,
        created_at=_now(),
        updated_at=_now(),
    )
    session.add(row)
    if commit:
        await _commit_with_retry(session)
        await session.refresh(row)
    else:
        await session.flush()

    try:
        await event_bus.publish("execution_order", _serialize_execution_order(row))
    except Exception:
        pass

    return row


async def create_execution_session_event(
    session: AsyncSession,
    *,
    session_id: str,
    event_type: str,
    severity: str = "info",
    leg_id: str | None = None,
    message: str | None = None,
    payload: dict[str, Any] | None = None,
    commit: bool = True,
) -> ExecutionSessionEvent:
    row = ExecutionSessionEvent(
        id=_new_id(),
        session_id=session_id,
        leg_id=leg_id,
        event_type=str(event_type),
        severity=str(severity or "info"),
        message=message,
        payload_json=payload or {},
        created_at=_now(),
    )
    session.add(row)
    if commit:
        await _commit_with_retry(session)
        await session.refresh(row)
    else:
        await session.flush()

    try:
        await event_bus.publish("execution_session_event", _serialize_execution_event(row))
    except Exception:
        pass

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
    commit: bool = True,
) -> None:
    now = _now()
    consumed_at = now
    signal_row = await session.get(TradeSignal, signal_id)
    if signal_row is not None:
        signal_ts = signal_row.updated_at or signal_row.created_at
        if signal_ts is not None and signal_ts > consumed_at:
            consumed_at = signal_ts
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
        existing.decision_id = decision_id
        existing.outcome = outcome
        existing.reason = reason
        existing.payload_json = payload or {}
        existing.consumed_at = consumed_at
        if commit:
            await _commit_with_retry(session)
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
            consumed_at=consumed_at,
        )
    )
    if commit:
        await _commit_with_retry(session)


async def list_unconsumed_trade_signals(
    session: AsyncSession,
    *,
    trader_id: str,
    sources: Optional[list[str]] = None,
    statuses: Optional[list[str]] = None,
    strategy_types_by_source: Optional[dict[str, Any]] = None,
    cursor_created_at: Optional[datetime] = None,
    cursor_signal_id: Optional[str] = None,
    limit: int = 200,
) -> list[TradeSignal]:
    now = _now().replace(tzinfo=None)
    normalized_cursor_created_at = cursor_created_at
    if isinstance(normalized_cursor_created_at, datetime) and normalized_cursor_created_at.tzinfo is not None:
        normalized_cursor_created_at = normalized_cursor_created_at.astimezone(timezone.utc).replace(tzinfo=None)
    normalized_cursor_signal_id = str(cursor_signal_id or "").strip()
    signal_sort_ts = func.coalesce(TradeSignal.updated_at, TradeSignal.created_at)
    latest_consumed_at = (
        select(func.max(TraderSignalConsumption.consumed_at))
        .where(
            TraderSignalConsumption.trader_id == trader_id,
            TraderSignalConsumption.signal_id == TradeSignal.id,
        )
        .correlate(TradeSignal)
        .scalar_subquery()
    )
    query = (
        select(TradeSignal)
        .where(
            or_(
                latest_consumed_at.is_(None),
                signal_sort_ts > latest_consumed_at,
            )
        )
        .where(or_(TradeSignal.expires_at.is_(None), TradeSignal.expires_at >= now))
        .order_by(signal_sort_ts.asc(), TradeSignal.id.asc())
        .limit(max(1, min(limit, 5000)))
    )
    if normalized_cursor_created_at is not None:
        if normalized_cursor_signal_id:
            query = query.where(
                or_(
                    signal_sort_ts > normalized_cursor_created_at,
                    and_(signal_sort_ts == normalized_cursor_created_at, TradeSignal.id > normalized_cursor_signal_id),
                )
            )
        else:
            query = query.where(signal_sort_ts > normalized_cursor_created_at)
    normalized_statuses = (
        [str(status or "").strip().lower() for status in statuses if str(status or "").strip()]
        if statuses is not None
        else ["pending"]
    )
    if normalized_statuses:
        query = query.where(func.lower(func.coalesce(TradeSignal.status, "")).in_(normalized_statuses))
    else:
        return []

    if sources is not None:
        normalized_sources = [str(source).strip().lower() for source in sources if str(source).strip()]
        if not normalized_sources:
            return []
        query = query.where(func.lower(func.coalesce(TradeSignal.source, "")).in_(normalized_sources))

    normalized_strategy_types_by_source: dict[str, list[str]] = {}
    if isinstance(strategy_types_by_source, dict):
        for raw_source, raw_strategy_types in strategy_types_by_source.items():
            source_key = str(raw_source or "").strip().lower()
            if not source_key:
                continue
            if isinstance(raw_strategy_types, str):
                candidates = [raw_strategy_types]
            elif isinstance(raw_strategy_types, (list, tuple, set)):
                candidates = list(raw_strategy_types)
            else:
                candidates = []
            normalized: list[str] = []
            seen: set[str] = set()
            for raw_strategy in candidates:
                strategy_type = str(raw_strategy or "").strip().lower()
                if not strategy_type or strategy_type in seen:
                    continue
                seen.add(strategy_type)
                normalized.append(strategy_type)
            if normalized:
                normalized_strategy_types_by_source[source_key] = normalized
    if normalized_strategy_types_by_source:
        source_strategy_clauses = []
        for source_key, allowed_strategy_types in normalized_strategy_types_by_source.items():
            source_strategy_clauses.append(
                and_(
                    func.lower(func.coalesce(TradeSignal.source, "")) == source_key,
                    func.lower(func.coalesce(TradeSignal.strategy_type, "")).in_(allowed_strategy_types),
                )
            )
        if not source_strategy_clauses:
            return []
        query = query.where(or_(*source_strategy_clauses))
    return list((await session.execute(query)).scalars().all())


async def get_trader_signal_cursor(
    session: AsyncSession,
    *,
    trader_id: str,
) -> tuple[Optional[datetime], Optional[str]]:
    row = await session.get(TraderSignalCursor, trader_id)
    if row is None:
        return None, None
    return row.last_signal_created_at, row.last_signal_id


async def upsert_trader_signal_cursor(
    session: AsyncSession,
    *,
    trader_id: str,
    last_signal_created_at: Optional[datetime],
    last_signal_id: Optional[str],
    commit: bool = True,
) -> None:
    row = await session.get(TraderSignalCursor, trader_id)
    if row is None:
        row = TraderSignalCursor(
            trader_id=trader_id,
            last_signal_created_at=last_signal_created_at,
            last_signal_id=last_signal_id,
            updated_at=_now(),
        )
        session.add(row)
    else:
        row.last_signal_created_at = last_signal_created_at
        row.last_signal_id = str(last_signal_id or "") or None
        row.updated_at = _now()

    if commit:
        await _commit_with_retry(session)


async def reconcile_live_provider_orders(
    session: AsyncSession,
    *,
    trader_id: str,
    commit: bool = True,
    broadcast: bool = True,
) -> dict[str, Any]:
    active_order_ids = [
        str(order_id)
        for order_id in (
            await session.execute(
                select(TraderOrder.id).where(
                    TraderOrder.trader_id == trader_id,
                    TraderOrder.mode == "live",
                    TraderOrder.status.in_(tuple(LIVE_ACTIVE_ORDER_STATUSES)),
                )
            )
        )
        .scalars()
        .all()
        if str(order_id).strip()
    ]
    if not active_order_ids:
        return {
            "trader_id": trader_id,
            "provider_ready": True,
            "active_seen": 0,
            "provider_tagged": 0,
            "snapshots_found": 0,
            "updated_orders": 0,
            "updated_session_orders": 0,
            "updated_legs": 0,
            "status_changes": 0,
            "notional_updates": 0,
            "price_updates": 0,
        }

    # Release the DB connection while initializing the external trading provider.
    if commit:
        await session.commit()
    provider_ready = False
    async with release_conn(session):
        try:
            provider_ready = bool(await live_execution_service.ensure_initialized())
        except Exception as exc:
            logger.warning(
                "Live provider reconciliation failed to initialize trading service", trader_id=trader_id, exc_info=exc
            )
            provider_ready = False
    if not provider_ready:
        return {
            "trader_id": trader_id,
            "provider_ready": False,
            "active_seen": len(active_order_ids),
            "provider_tagged": 0,
            "snapshots_found": 0,
            "updated_orders": 0,
            "updated_session_orders": 0,
            "updated_legs": 0,
            "status_changes": 0,
            "notional_updates": 0,
            "price_updates": 0,
        }

    order_reconciliation_targets: dict[str, dict[str, Any]] = {}
    provider_clob_ids: set[str] = set()
    provider_tagged = 0
    tagged_payload_updates = 0
    active_order_rows = list(
        (
            await session.execute(select(TraderOrder).where(TraderOrder.id.in_(active_order_ids)))
        )
        .scalars()
        .all()
    )
    session_order_rows = list(
        (
            await session.execute(
                select(ExecutionSessionOrder)
                .where(ExecutionSessionOrder.trader_order_id.in_(active_order_ids))
                .order_by(desc(ExecutionSessionOrder.created_at))
            )
        )
        .scalars()
        .all()
    )
    session_orders_by_order: dict[str, list[ExecutionSessionOrder]] = {}
    for row in session_order_rows:
        key = str(row.trader_order_id or "").strip()
        if key:
            session_orders_by_order.setdefault(key, []).append(row)

    for order in active_order_rows:
        order_id = str(order.id)
        payload = dict(order.payload_json or {})
        previous_provider_order_id = str(payload.get("provider_order_id") or "").strip()
        previous_provider_clob_order_id = str(payload.get("provider_clob_order_id") or "").strip()
        session_orders = session_orders_by_order.get(order_id, [])
        session_order = session_orders[0] if session_orders else None
        provider_order_id, provider_clob_order_id = _resolve_provider_order_ids(
            order_payload=payload,
            session_order=session_order,
        )
        if provider_order_id:
            payload.setdefault("provider_order_id", provider_order_id)
        if provider_clob_order_id:
            payload.setdefault("provider_clob_order_id", provider_clob_order_id)
            provider_clob_ids.add(provider_clob_order_id)
            provider_tagged += 1
        if (
            str(payload.get("provider_order_id") or "").strip() != previous_provider_order_id
            or str(payload.get("provider_clob_order_id") or "").strip() != previous_provider_clob_order_id
        ):
            tagged_payload_updates += 1
        order_reconciliation_targets[order_id] = {
            "provider_order_id": provider_order_id,
            "provider_clob_order_id": provider_clob_order_id,
        }

    snapshots: dict[str, dict[str, Any]] = {}
    if provider_clob_ids:
        async with release_conn(session):
            try:
                snapshots = await live_execution_service.get_order_snapshots_by_clob_ids(sorted(provider_clob_ids))
            except Exception as exc:
                logger.warning("Live provider reconciliation failed to fetch snapshots", trader_id=trader_id, exc_info=exc)
                snapshots = {}

    now = _now()
    updated_orders = 0
    updated_session_orders = 0
    updated_legs = 0
    updated_sessions = 0
    terminal_session_cancels = 0
    status_changes = 0
    session_status_changes = 0
    notional_updates = 0
    price_updates = 0
    updated_order_rows: dict[str, TraderOrder] = {}
    updated_execution_order_rows: dict[str, ExecutionSessionOrder] = {}
    updated_execution_session_rows: dict[str, ExecutionSession] = {}
    touched_session_ids: set[str] = set()
    leg_state_updates: dict[str, dict[str, Any]] = {}

    active_rows = list(
        (
            await session.execute(select(TraderOrder).where(TraderOrder.id.in_(active_order_ids)))
        )
        .scalars()
        .all()
    )
    session_order_rows = list(
        (
            await session.execute(
                select(ExecutionSessionOrder)
                .where(ExecutionSessionOrder.trader_order_id.in_(active_order_ids))
                .order_by(desc(ExecutionSessionOrder.created_at))
            )
        )
        .scalars()
        .all()
    )
    session_orders_by_order = {}
    linked_session_ids: set[str] = set()
    for row in session_order_rows:
        key = str(row.trader_order_id or "").strip()
        if not key:
            continue
        session_orders_by_order.setdefault(key, []).append(row)
        session_id_key = str(row.session_id or "").strip()
        if session_id_key:
            linked_session_ids.add(session_id_key)

    session_status_by_id: dict[str, str] = {}
    if linked_session_ids:
        linked_sessions = list(
            (
                await session.execute(
                    select(ExecutionSession.id, ExecutionSession.status).where(
                        ExecutionSession.id.in_(list(linked_session_ids))
                    )
                )
            ).all()
        )
        for linked_session in linked_sessions:
            session_status_by_id[str(linked_session.id)] = str(linked_session.status or "")

    for order in active_rows:
        with session.no_autoflush:
            await session.refresh(
                order,
                attribute_names=[
                    "status",
                    "payload_json",
                    "notional_usd",
                    "effective_price",
                    "entry_price",
                    "updated_at",
                    "executed_at",
                ],
            )
        if _normalize_status_key(order.status) not in LIVE_ACTIVE_ORDER_STATUSES:
            continue

        payload = dict(order.payload_json or {})
        order_target = order_reconciliation_targets.get(str(order.id), {})
        session_orders = session_orders_by_order.get(str(order.id), [])
        session_order = session_orders[0] if session_orders else None
        provider_order_id = str(order_target.get("provider_order_id") or "").strip()
        provider_clob_order_id = str(order_target.get("provider_clob_order_id") or "").strip()
        if not provider_order_id and session_order is not None:
            provider_order_id = str(session_order.provider_order_id or "").strip()
        if not provider_clob_order_id and session_order is not None:
            provider_clob_order_id = str(session_order.provider_clob_order_id or "").strip()
        if not provider_order_id or not provider_clob_order_id:
            resolved_provider_order_id, resolved_provider_clob_order_id = _resolve_provider_order_ids(
                order_payload=payload,
                session_order=session_order,
            )
            if not provider_order_id:
                provider_order_id = resolved_provider_order_id
            if not provider_clob_order_id:
                provider_clob_order_id = resolved_provider_clob_order_id
        if not provider_clob_order_id:
            continue
        snapshot = snapshots.get(provider_clob_order_id)
        if not isinstance(snapshot, dict):
            continue

        snapshot_status = str(snapshot.get("normalized_status") or "").strip().lower()
        filled_size = max(0.0, safe_float(snapshot.get("filled_size"), 0.0) or 0.0)
        avg_fill_price = safe_float(snapshot.get("average_fill_price"))
        filled_notional_usd = max(0.0, safe_float(snapshot.get("filled_notional_usd"), 0.0) or 0.0)
        if filled_notional_usd <= 0.0 and filled_size > 0.0:
            reference_price = (
                avg_fill_price
                if avg_fill_price is not None and avg_fill_price > 0
                else safe_float(snapshot.get("limit_price"))
            )
            if reference_price is None or reference_price <= 0:
                reference_price = safe_float(order.effective_price, 0.0) or safe_float(order.entry_price, 0.0)
            if reference_price and reference_price > 0:
                filled_notional_usd = filled_size * reference_price

        mapped_status = _map_provider_snapshot_status(snapshot_status, filled_notional_usd)
        if not mapped_status:
            continue

        linked_session_terminal = False
        for session_order_row in session_orders:
            session_status = _normalize_status_key(session_status_by_id.get(str(session_order_row.session_id or "")))
            if session_status in TERMINAL_EXECUTION_SESSION_STATUSES:
                linked_session_terminal = True
                break

        if (
            linked_session_terminal
            and mapped_status in {"submitted", "open"}
            and filled_notional_usd <= 0.0
            and filled_size <= 0.0
        ):
            cancel_targets: list[str] = []
            if provider_clob_order_id:
                cancel_targets.append(provider_clob_order_id)
            if provider_order_id and provider_order_id not in cancel_targets:
                cancel_targets.append(provider_order_id)
            cancel_success = False
            for cancel_target in cancel_targets:
                try:
                    if commit:
                        # Release the connection before provider cancellation I/O.
                        await session.commit()
                    if await live_execution_service.cancel_order(cancel_target):
                        cancel_success = True
                        break
                except Exception as exc:
                    logger.warning(
                        "Failed to cancel terminal-session live provider order",
                        trader_id=trader_id,
                        order_id=order.id,
                        cancel_target=cancel_target,
                        exc_info=exc,
                    )
            if cancel_success:
                terminal_session_cancels += 1
                mapped_status = "cancelled"
                snapshot_status = "cancelled"
                snapshot = dict(snapshot)
                snapshot["normalized_status"] = "cancelled"
                snapshot["raw_status"] = "cancelled_session_terminal"

        previous_status = _normalize_status_key(order.status)
        previous_notional = max(0.0, abs(safe_float(order.notional_usd, 0.0) or 0.0))
        previous_effective_price = safe_float(order.effective_price)
        order_status_to_persist = mapped_status

        if mapped_status in {"submitted", "open"}:
            next_notional = max(0.0, filled_notional_usd)
        elif mapped_status == "executed":
            next_notional = filled_notional_usd if filled_notional_usd > 0.0 else previous_notional
        else:
            next_notional = 0.0

        next_effective_price = previous_effective_price
        if avg_fill_price is not None and avg_fill_price > 0:
            next_effective_price = avg_fill_price

        order.status = order_status_to_persist
        order.notional_usd = float(next_notional)
        if next_effective_price is not None:
            order.effective_price = float(next_effective_price)
        if next_notional > 0.0 and order.executed_at is None:
            order.executed_at = now
        order.updated_at = now
        payload["provider_order_id"] = provider_order_id
        payload["provider_clob_order_id"] = provider_clob_order_id
        payload["provider_reconciliation"] = {
            "reconciled_at": to_iso(now),
            "provider_order_id": provider_order_id,
            "provider_clob_order_id": provider_clob_order_id,
            "snapshot": snapshot,
            "snapshot_status": snapshot_status,
            "mapped_status": mapped_status,
            "filled_size": filled_size,
            "average_fill_price": avg_fill_price,
            "filled_notional_usd": filled_notional_usd,
        }
        if linked_session_terminal:
            payload["provider_reconciliation"]["terminal_session"] = True
        order.payload_json = _sync_order_runtime_payload(
            payload=payload,
            status=order_status_to_persist,
            now=now,
            provider_snapshot_status=snapshot_status,
            mapped_status=mapped_status,
        )
        if order_status_to_persist in {"cancelled", "failed"}:
            import services.trader_hot_state as _hs
            _hs.record_order_cancelled(
                trader_id=trader_id,
                mode=str(order.mode or ""),
                order_id=str(order.id or ""),
                market_id=str(order.market_id or ""),
                source=str(order.source or ""),
                copy_source_wallet=_extract_copy_source_wallet_from_payload(payload),
            )
        updated_orders += 1
        updated_order_rows[str(order.id)] = order
        if previous_status != _normalize_status_key(order_status_to_persist):
            status_changes += 1
        if abs(previous_notional - next_notional) > 1e-9:
            notional_updates += 1
        if (
            previous_effective_price is None
            and next_effective_price is not None
            or previous_effective_price is not None
            and next_effective_price is not None
            and abs(previous_effective_price - next_effective_price) > 1e-9
        ):
            price_updates += 1

        for session_order_row in session_orders:
            session_order_row.status = mapped_status
            if avg_fill_price is not None and avg_fill_price > 0:
                session_order_row.price = float(avg_fill_price)
            if filled_size > 0.0:
                session_order_row.size = float(filled_size)
            if mapped_status in {"submitted", "open", "executed"}:
                session_order_row.notional_usd = float(next_notional)
            else:
                session_order_row.notional_usd = float(filled_notional_usd)
            if provider_order_id:
                session_order_row.provider_order_id = provider_order_id
            if provider_clob_order_id:
                session_order_row.provider_clob_order_id = provider_clob_order_id
            session_payload = dict(session_order_row.payload_json or {})
            session_payload["provider_reconciliation"] = {
                "reconciled_at": to_iso(now),
                "snapshot_status": snapshot_status,
                "mapped_status": mapped_status,
                "filled_size": filled_size,
                "average_fill_price": avg_fill_price,
                "filled_notional_usd": filled_notional_usd,
                "provider_order_id": provider_order_id,
                "provider_clob_order_id": provider_clob_order_id,
            }
            session_order_row.payload_json = session_payload
            session_order_row.updated_at = now
            updated_session_orders += 1
            updated_execution_order_rows[str(session_order_row.id)] = session_order_row
            leg_state_updates[str(session_order_row.leg_id)] = {
                "status": mapped_status,
                "filled_notional_usd": float(
                    next_notional if mapped_status in {"submitted", "open", "executed"} else 0.0
                ),
                "filled_shares": float(filled_size),
                "avg_fill_price": float(avg_fill_price) if avg_fill_price is not None and avg_fill_price > 0 else None,
                "provider_order_id": provider_order_id,
                "provider_clob_order_id": provider_clob_order_id,
                "snapshot_status": snapshot_status,
            }
            touched_session_ids.add(str(session_order_row.session_id))

    if leg_state_updates:
        await session.flush()
        leg_rows = list(
            (
                await session.execute(
                    select(ExecutionSessionLeg).where(ExecutionSessionLeg.id.in_(list(leg_state_updates.keys())))
                )
            )
            .scalars()
            .all()
        )
        for leg_row in leg_rows:
            leg_update = leg_state_updates.get(str(leg_row.id))
            if leg_update is None:
                continue
            mapped_status = str(leg_update["status"])
            leg_row.status = _map_live_order_status_to_leg_status(mapped_status)
            leg_row.filled_notional_usd = float(max(0.0, safe_float(leg_update.get("filled_notional_usd"), 0.0) or 0.0))
            leg_row.filled_shares = float(max(0.0, safe_float(leg_update.get("filled_shares"), 0.0) or 0.0))
            avg_fill_price = safe_float(leg_update.get("avg_fill_price"))
            if avg_fill_price is not None and avg_fill_price > 0:
                leg_row.avg_fill_price = float(avg_fill_price)
            if mapped_status in {"failed", "cancelled"}:
                leg_row.last_error = f"provider_reconcile:{mapped_status}"
            else:
                leg_row.last_error = None
            metadata_json = dict(leg_row.metadata_json or {})
            metadata_json["provider_reconciliation"] = {
                "reconciled_at": to_iso(now),
                "provider_order_id": leg_update.get("provider_order_id"),
                "provider_clob_order_id": leg_update.get("provider_clob_order_id"),
                "snapshot_status": leg_update.get("snapshot_status"),
                "mapped_status": mapped_status,
            }
            leg_row.metadata_json = metadata_json
            leg_row.updated_at = now
            updated_legs += 1
            touched_session_ids.add(str(leg_row.session_id))

    if touched_session_ids:
        await session.flush()
    for session_id in touched_session_ids:
        session_row = await _refresh_execution_session_rollups(session, session_id=session_id)
        if session_row is None:
            continue
        previous_session_status = _normalize_status_key(session_row.status)
        next_session_status = previous_session_status
        legs_open = int(session_row.legs_open or 0)
        legs_completed = int(session_row.legs_completed or 0)
        legs_failed = int(session_row.legs_failed or 0)
        if legs_open <= 0:
            if legs_completed > 0 and legs_failed == 0:
                next_session_status = "completed"
                session_row.error_message = None
            elif legs_completed == 0 and legs_failed > 0:
                next_session_status = "failed"
                if not session_row.error_message:
                    session_row.error_message = "All execution legs failed during provider reconciliation."
            elif legs_completed > 0 and legs_failed > 0:
                next_session_status = "completed"
        elif previous_session_status in {"pending", "placing", "partial", "hedging"}:
            next_session_status = "working"

        if next_session_status in TERMINAL_EXECUTION_SESSION_STATUSES and session_row.completed_at is None:
            session_row.completed_at = now
        session_row.status = next_session_status
        session_row.updated_at = now
        updated_sessions += 1
        if next_session_status != previous_session_status:
            session_status_changes += 1
        updated_execution_session_rows[str(session_row.id)] = session_row

    if commit and (
        updated_orders > 0
        or updated_session_orders > 0
        or updated_legs > 0
        or updated_sessions > 0
        or tagged_payload_updates > 0
    ):
        await _commit_with_retry(session)
    elif not commit and (
        updated_orders > 0
        or updated_session_orders > 0
        or updated_legs > 0
        or updated_sessions > 0
        or tagged_payload_updates > 0
    ):
        await session.flush()

    if broadcast and (updated_order_rows or updated_execution_order_rows or updated_execution_session_rows):
        order_payloads = [_serialize_order(row) for row in updated_order_rows.values()]
        execution_order_payloads = [_serialize_execution_order(row) for row in updated_execution_order_rows.values()]
        execution_session_payloads = [
            _serialize_execution_session(row) for row in updated_execution_session_rows.values()
        ]

        for payload in order_payloads:
            try:
                await event_bus.publish("trader_order", payload)
            except Exception:
                continue
        for payload in execution_order_payloads:
            try:
                await event_bus.publish("execution_order", payload)
            except Exception:
                continue
        for payload in execution_session_payloads:
            try:
                await event_bus.publish("execution_session", payload)
            except Exception:
                continue

        redis_rows: dict[str, dict[str, str]] = {}
        for payload in order_payloads:
            order_id = str(payload.get("id") or "").strip()
            if not order_id:
                continue
            redis_rows[f"{TRADER_ORDER_STATUS_HASH_KEY_PREFIX}{order_id}"] = {
                "id": order_id,
                "trader_id": str(payload.get("trader_id") or ""),
                "mode": str(payload.get("mode") or ""),
                "status": str(payload.get("status") or ""),
                "direction": str(payload.get("direction") or ""),
                "market_id": str(payload.get("market_id") or ""),
                "updated_at": str(payload.get("updated_at") or ""),
                "effective_price": str(payload.get("effective_price") or ""),
                "notional_usd": str(payload.get("notional_usd") or ""),
                "provider_order_id": str(payload.get("provider_order_id") or ""),
                "provider_clob_order_id": str(payload.get("provider_clob_order_id") or ""),
            }

        if redis_rows:
            await redis_streams.hset_many(
                redis_rows,
                expire_seconds=TRADER_ORDER_STATUS_TTL_SECONDS,
            )
            for payload in order_payloads:
                await redis_streams.append_json(
                    TRADER_ORDER_UPDATES_STREAM,
                    payload,
                    maxlen=TRADER_ORDER_UPDATES_STREAM_MAXLEN,
                )

    return {
        "trader_id": trader_id,
        "provider_ready": True,
        "active_seen": len(active_order_ids),
        "provider_tagged": provider_tagged,
        "payload_tag_updates": tagged_payload_updates,
        "snapshots_found": len(snapshots),
        "updated_orders": updated_orders,
        "updated_session_orders": updated_session_orders,
        "updated_legs": updated_legs,
        "updated_sessions": updated_sessions,
        "terminal_session_cancels": terminal_session_cancels,
        "status_changes": status_changes,
        "session_status_changes": session_status_changes,
        "notional_updates": notional_updates,
        "price_updates": price_updates,
    }


async def list_live_wallet_positions_for_trader(
    session: AsyncSession,
    *,
    trader_id: str,
    include_managed: bool = True,
) -> dict[str, Any]:
    trader_row = await session.get(Trader, trader_id)
    if trader_row is None:
        raise ValueError("Trader not found")

    wallet_address = await _resolve_execution_wallet_address()
    if not wallet_address:
        return {
            "trader_id": trader_id,
            "wallet_address": None,
            "positions": [],
            "managed_token_ids": [],
            "managed_order_ids": [],
            "summary": {
                "total_positions": 0,
                "managed_positions": 0,
                "unmanaged_positions": 0,
                "returned_positions": 0,
            },
        }

    async with release_conn(session):
        try:
            wallet_positions_raw = await polymarket_client.get_wallet_positions_with_prices(wallet_address)
        except Exception as exc:
            raise ValueError(f"Failed to fetch execution wallet positions: {exc}") from exc
    if not isinstance(wallet_positions_raw, list):
        wallet_positions_raw = []

    active_live_rows = list(
        (
            await session.execute(
                select(TraderOrder).where(
                    TraderOrder.trader_id == trader_id,
                    func.lower(func.coalesce(TraderOrder.mode, "")) == "live",
                    func.lower(func.coalesce(TraderOrder.status, "")).in_(tuple(LIVE_ACTIVE_ORDER_STATUSES)),
                )
            )
        )
        .scalars()
        .all()
    )
    managed_by_token: dict[str, str] = {}
    for row in active_live_rows:
        token_id = _wallet_position_token_key(_extract_order_token_id(row))
        if token_id and token_id not in managed_by_token:
            managed_by_token[token_id] = str(row.id)

    condition_lookups: dict[str, Any] = {}
    token_lookups: dict[str, Any] = {}
    for wallet_position in wallet_positions_raw:
        if not isinstance(wallet_position, dict):
            continue
        token_id = _read_wallet_text(wallet_position, "asset", "asset_id", "assetId", "token_id", "tokenId")
        if not token_id:
            continue
        condition_id = _normalize_condition_id(
            _read_wallet_text(wallet_position, "conditionId", "condition_id", "market", "market_id", "marketId")
        )
        if condition_id:
            if condition_id not in condition_lookups:
                condition_lookups[condition_id] = polymarket_client.get_market_by_condition_id(condition_id)
        token_key = _wallet_position_token_key(token_id)
        if token_key and token_key not in token_lookups:
            token_lookups[token_key] = polymarket_client.get_market_by_token_id(token_id)

    markets_by_condition: dict[str, dict[str, Any]] = {}
    markets_by_token: dict[str, dict[str, Any]] = {}
    async with release_conn(session):
        if condition_lookups:
            condition_results = await asyncio.gather(*condition_lookups.values(), return_exceptions=True)
            for condition_id, result in zip(condition_lookups.keys(), condition_results):
                if isinstance(result, dict):
                    markets_by_condition[condition_id] = result
        if token_lookups:
            token_results = await asyncio.gather(*token_lookups.values(), return_exceptions=True)
            for token_key, result in zip(token_lookups.keys(), token_results):
                if isinstance(result, dict):
                    markets_by_token[token_key] = result

    total_positions = 0
    managed_positions = 0
    normalized_positions: list[dict[str, Any]] = []
    seen_tokens: set[str] = set()
    for wallet_position in wallet_positions_raw:
        if not isinstance(wallet_position, dict):
            continue
        token_id = _read_wallet_text(wallet_position, "asset", "asset_id", "assetId", "token_id", "tokenId")
        token_key = _wallet_position_token_key(token_id)
        if not token_key or token_key in seen_tokens:
            continue
        seen_tokens.add(token_key)

        condition_id = _normalize_condition_id(
            _read_wallet_text(wallet_position, "conditionId", "condition_id", "market", "market_id", "marketId")
        )
        market_info = dict(markets_by_condition.get(condition_id) or {})
        if not market_info:
            market_info = dict(markets_by_token.get(token_key) or {})

        normalized = _normalize_wallet_position_row(
            wallet_position,
            market_info=market_info,
        )
        if normalized is None:
            continue

        total_positions += 1
        managed_order_id = managed_by_token.get(token_key)
        is_managed = managed_order_id is not None
        if is_managed:
            managed_positions += 1
        normalized["is_managed"] = is_managed
        normalized["managed_order_id"] = managed_order_id
        if include_managed or not is_managed:
            normalized_positions.append(normalized)

    normalized_positions.sort(
        key=lambda item: (
            bool(item.get("is_managed", False)),
            -float(item.get("current_value") or item.get("initial_value") or 0.0),
        )
    )

    return {
        "trader_id": trader_id,
        "wallet_address": wallet_address,
        "positions": normalized_positions,
        "managed_token_ids": sorted(managed_by_token.keys()),
        "managed_order_ids": sorted(set(managed_by_token.values())),
        "summary": {
            "total_positions": total_positions,
            "managed_positions": managed_positions,
            "unmanaged_positions": max(0, total_positions - managed_positions),
            "returned_positions": len(normalized_positions),
        },
    }


async def adopt_live_wallet_position(
    session: AsyncSession,
    *,
    trader_id: str,
    token_id: str,
    reason: Optional[str] = None,
) -> dict[str, Any]:
    trader_row = await session.get(Trader, trader_id)
    if trader_row is None:
        raise ValueError("Trader not found")

    trader_mode = _normalize_trader_mode(trader_row.mode)
    if trader_mode != "live":
        raise ValueError("Trader mode must be live to adopt wallet positions")

    requested_token_key = _wallet_position_token_key(token_id)
    if not requested_token_key:
        raise ValueError("token_id is required")
    metadata_json = trader_row.metadata_json if isinstance(trader_row.metadata_json, dict) else {}
    assigned_strategy_key = _normalize_strategy_key(metadata_json.get("manual_position_strategy_key"))
    if not assigned_strategy_key:
        source_configs = _normalize_source_configs(trader_row.source_configs_json)
        for source_config in source_configs:
            candidate = _normalize_strategy_key(source_config.get("strategy_key"))
            if candidate:
                assigned_strategy_key = candidate
                break
    if not assigned_strategy_key:
        raise ValueError("Trader must have a configured strategy before adopting live wallet positions")
    await _validate_strategy_key(session, assigned_strategy_key)

    wallet_snapshot = await list_live_wallet_positions_for_trader(
        session,
        trader_id=trader_id,
        include_managed=True,
    )
    wallet_address = str(wallet_snapshot.get("wallet_address") or "").strip()
    if not wallet_address:
        raise ValueError("Execution wallet address is not configured")

    positions = list(wallet_snapshot.get("positions") or [])
    target_position = next(
        (
            position
            for position in positions
            if _wallet_position_token_key(position.get("token_id")) == requested_token_key
        ),
        None,
    )
    if target_position is None:
        raise ValueError(f"Token '{token_id}' not found in execution wallet positions")
    if bool(target_position.get("is_managed")):
        managed_order_id = str(target_position.get("managed_order_id") or "").strip()
        if managed_order_id:
            raise ValueError(f"Token '{token_id}' is already managed by order {managed_order_id}")
        raise ValueError(f"Token '{token_id}' is already managed by this trader")

    direction = str(target_position.get("direction") or "").strip().lower()
    if direction not in {"buy_yes", "buy_no"}:
        raise ValueError("Unable to infer binary direction for wallet position")

    size = max(0.0, safe_float(target_position.get("size"), 0.0) or 0.0)
    if size <= 0.0:
        raise ValueError("Wallet position size must be greater than zero")

    entry_price = safe_float(target_position.get("avg_price"))
    initial_value = safe_float(target_position.get("initial_value"))
    if (entry_price is None or entry_price <= 0.0) and initial_value is not None and initial_value > 0.0:
        entry_price = initial_value / size
    if entry_price is None or entry_price <= 0.0:
        raise ValueError("Unable to infer average entry price for wallet position")

    notional_usd = initial_value if initial_value is not None and initial_value > 0.0 else (size * entry_price)
    if notional_usd <= 0.0:
        raise ValueError("Unable to infer position notional value for wallet position")

    current_price = safe_float(target_position.get("current_price"))
    if current_price is None or current_price <= 0.0:
        current_price = entry_price

    resolved_token_id = str(target_position.get("token_id") or "").strip()
    condition_id = _normalize_condition_id(target_position.get("condition_id") or target_position.get("market_id"))
    market_id = str(target_position.get("market_id") or condition_id or resolved_token_id).strip()
    if not market_id:
        raise ValueError("Unable to resolve market id for wallet position")

    selected_outcome = "yes" if direction == "buy_yes" else "no"
    yes_token_id = str(target_position.get("yes_token_id") or "").strip()
    no_token_id = str(target_position.get("no_token_id") or "").strip()
    yes_price = safe_float(target_position.get("yes_price"))
    no_price = safe_float(target_position.get("no_price"))
    market_slug = str(target_position.get("market_slug") or "").strip()
    event_slug = str(target_position.get("event_slug") or "").strip()
    market_url = str(target_position.get("market_url") or "").strip()
    market_question = str(target_position.get("market_question") or "").strip() or None

    now = _now()
    reason_text = str(reason or "manual_wallet_position_adopt").strip() or "manual_wallet_position_adopt"
    mark_price = float(current_price)
    provider_snapshot = {
        "normalized_status": "filled",
        "status": "filled",
        "asset_id": resolved_token_id,
        "token_id": resolved_token_id,
        "filled_size": float(size),
        "average_fill_price": float(entry_price),
        "filled_notional_usd": float(notional_usd),
    }
    payload_json: dict[str, Any] = {
        "source": _MANUAL_LIVE_POSITION_SOURCE,
        "strategy_type": assigned_strategy_key,
        "market_id": market_id,
        "condition_id": condition_id or None,
        "token_id": resolved_token_id,
        "selected_token_id": resolved_token_id,
        "selected_outcome": selected_outcome,
        "yes_token_id": yes_token_id or None,
        "no_token_id": no_token_id or None,
        "market_slug": market_slug or None,
        "event_slug": event_slug or None,
        "market_url": market_url or None,
        "live_market": {
            "market_data_source": "wallet_position_import",
            "condition_id": condition_id or None,
            "selected_token_id": resolved_token_id,
            "selected_outcome": selected_outcome,
            "yes_token_id": yes_token_id or None,
            "no_token_id": no_token_id or None,
            "live_selected_price": float(mark_price),
            "live_yes_price": float(yes_price) if yes_price is not None else None,
            "live_no_price": float(no_price) if no_price is not None else None,
            "live_market_fetched_at": to_iso(now),
            "fetched_at": to_iso(now),
        },
        "provider_reconciliation": {
            "source": "wallet_position_import",
            "snapshot_status": "filled",
            "mapped_status": "executed",
            "reconciled_at": to_iso(now),
            "filled_size": float(size),
            "average_fill_price": float(entry_price),
            "filled_notional_usd": float(notional_usd),
            "snapshot": provider_snapshot,
        },
        "position_state": {
            "highest_price": float(mark_price),
            "lowest_price": float(mark_price),
            "last_mark_price": float(mark_price),
            "last_mark_source": "wallet_mark",
            "last_marked_at": to_iso(now),
        },
        "wallet_position_import": {
            "wallet_address": wallet_address,
            "token_id": resolved_token_id,
            "market_id": market_id,
            "size": float(size),
            "avg_price": float(entry_price),
            "current_price": float(mark_price),
            "initial_value": float(notional_usd),
            "current_value": safe_float(target_position.get("current_value")),
            "imported_at": to_iso(now),
            "reason": reason_text,
        },
    }

    order_row = TraderOrder(
        id=_new_id(),
        trader_id=trader_id,
        signal_id=None,
        decision_id=None,
        source=_MANUAL_LIVE_POSITION_SOURCE,
        strategy_key=assigned_strategy_key,
        strategy_version=None,
        market_id=market_id,
        market_question=market_question,
        direction=direction,
        mode="live",
        status="open",
        notional_usd=float(notional_usd),
        entry_price=float(entry_price),
        effective_price=float(entry_price),
        edge_percent=None,
        confidence=None,
        actual_profit=None,
        reason=reason_text,
        payload_json=payload_json,
        error_message=None,
        created_at=now,
        executed_at=now,
        updated_at=now,
    )
    session.add(order_row)
    await _commit_with_retry(session)
    await session.refresh(order_row)

    position_inventory = await sync_trader_position_inventory(
        session,
        trader_id=trader_id,
        mode="live",
    )
    return {
        "status": "adopted",
        "trader_id": trader_id,
        "wallet_address": wallet_address,
        "strategy_key": assigned_strategy_key,
        "token_id": resolved_token_id,
        "market_id": market_id,
        "direction": direction,
        "order": _serialize_order(order_row),
        "position_inventory": position_inventory,
    }


async def sync_trader_position_inventory(
    session: AsyncSession,
    *,
    trader_id: str,
    mode: Optional[str] = None,
    commit: bool = True,
) -> dict[str, Any]:
    query = select(TraderOrder).where(
        TraderOrder.trader_id == trader_id,
        TraderOrder.status.in_(tuple(OPEN_ORDER_STATUSES)),
    )
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            query = query.where(TraderOrder.mode.not_in(list(_TRADER_MODES)))
        else:
            query = query.where(TraderOrder.mode == mode_key)

    order_rows = list((await session.execute(query)).scalars().all())

    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in order_rows:
        mode_key = _normalize_mode_key(row.mode)
        if not _is_active_order_status(mode_key, row.status):
            continue
        identity = _position_identity_key(mode_key, row.market_id, row.direction)
        if not identity[1]:
            continue

        payload = dict(row.payload_json or {})
        position_state = payload.get("position_state")
        if not isinstance(position_state, dict):
            position_state = {}
        row_notional = safe_float(row.notional_usd, 0.0) or 0.0
        notional = _live_active_notional(row.mode, row.status, row_notional, payload)
        _, _, fill_price = _extract_live_fill_metrics(payload)
        entry_price = (
            fill_price
            if fill_price is not None and fill_price > 0
            else (safe_float(row.effective_price, 0.0) or safe_float(row.entry_price, 0.0))
        )
        mark_price = safe_float(position_state.get("last_mark_price"))
        mark_source = str(position_state.get("last_mark_source") or "")
        mark_updated_at = str(position_state.get("last_marked_at") or "")
        if notional <= 0.0:
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
                "open_order_ids": [],
                "mark_anchor": None,
                "last_mark_price": None,
                "last_mark_source": "",
                "last_marked_at": "",
            }
            grouped[identity] = bucket

        bucket["open_order_count"] = int(bucket["open_order_count"]) + 1
        bucket["total_notional_usd"] = float(bucket["total_notional_usd"]) + max(0.0, notional)
        open_order_ids = bucket.get("open_order_ids")
        if isinstance(open_order_ids, list):
            open_order_ids.append(str(row.id))
        if entry_price and entry_price > 0 and notional > 0:
            bucket["weighted_entry_numerator"] = float(bucket["weighted_entry_numerator"]) + (entry_price * notional)
            bucket["weighted_entry_denominator"] = float(bucket["weighted_entry_denominator"]) + notional
        if row.created_at and (bucket["first_order_at"] is None or row.created_at < bucket["first_order_at"]):
            bucket["first_order_at"] = row.created_at
        last_order_at = row.updated_at or row.executed_at or row.created_at
        if last_order_at and (bucket["last_order_at"] is None or last_order_at > bucket["last_order_at"]):
            bucket["last_order_at"] = last_order_at
        if mark_price is not None and mark_price > 0:
            mark_anchor = last_order_at
            previous_anchor = bucket.get("mark_anchor")
            if previous_anchor is None or (mark_anchor is not None and mark_anchor >= previous_anchor):
                bucket["mark_anchor"] = mark_anchor
                bucket["last_mark_price"] = float(mark_price)
                bucket["last_mark_source"] = mark_source
                bucket["last_marked_at"] = mark_updated_at

    existing_query = select(TraderPosition).where(TraderPosition.trader_id == trader_id)
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            existing_query = existing_query.where(TraderPosition.mode.not_in(list(_TRADER_MODES)))
        else:
            existing_query = existing_query.where(TraderPosition.mode == mode_key)
    existing_rows = list((await session.execute(existing_query)).scalars().all())
    existing_by_identity = {
        _position_identity_key(row.mode, row.market_id, row.direction): row for row in existing_rows
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
        position_payload = {
            "sync_source": "order_inventory",
            "open_order_ids": list(
                dict.fromkeys([str(order_id) for order_id in list(bucket.get("open_order_ids") or [])])
            ),
            "last_mark_price": bucket.get("last_mark_price"),
            "last_mark_source": str(bucket.get("last_mark_source") or ""),
            "last_marked_at": str(bucket.get("last_marked_at") or ""),
        }

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
                payload_json=position_payload,
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
        existing_payload = dict(row.payload_json or {})
        existing_payload.update(position_payload)
        row.payload_json = existing_payload
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
        existing_payload = dict(row.payload_json or {})
        existing_payload["sync_source"] = "order_inventory"
        existing_payload["open_order_ids"] = []
        row.payload_json = existing_payload
        row.updated_at = now
        closures += 1

    if commit and (inserts > 0 or updates > 0 or closures > 0):
        await _commit_with_retry(session)

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
    position_cap_scope: str | None = None,
) -> int:
    scope = str(position_cap_scope or "market_direction").strip().lower()
    if scope not in {"market_direction", "market", "asset_timeframe"}:
        scope = "market_direction"

    position_query = (
        select(
            TraderPosition.mode,
            TraderPosition.market_id,
            TraderPosition.direction,
            TraderPosition.payload_json,
        )
        .where(TraderPosition.trader_id == trader_id)
        .where(func.lower(func.coalesce(TraderPosition.status, "")) == ACTIVE_POSITION_STATUS)
    )
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            position_query = position_query.where(
                func.lower(func.coalesce(TraderPosition.mode, "")).not_in(list(_TRADER_MODES))
            )
        else:
            position_query = position_query.where(func.lower(func.coalesce(TraderPosition.mode, "")) == mode_key)
    position_rows = (await session.execute(position_query)).all()
    active_position_keys: set[tuple[str, str, str]] = set()
    for row in position_rows:
        key = _position_cap_scope_key(
            position_cap_scope=scope,
            mode=(mode if mode is not None else row.mode),
            market_id=row.market_id,
            direction=row.direction,
            payload=row.payload_json,
        )
        if key is not None:
            active_position_keys.add(key)

    order_status_expr = func.lower(func.trim(func.coalesce(TraderOrder.status, "")))
    order_query = select(
        TraderOrder.mode,
        TraderOrder.status,
        TraderOrder.market_id,
        TraderOrder.direction,
        TraderOrder.notional_usd,
        TraderOrder.payload_json,
     ).where(TraderOrder.trader_id == trader_id).where(order_status_expr.in_(tuple(OPEN_ORDER_STATUSES)))
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            order_query = order_query.where(func.lower(func.coalesce(TraderOrder.mode, "")).not_in(list(_TRADER_MODES)))
        else:
            order_query = order_query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key)

    order_rows = (await session.execute(order_query)).all()
    active_order_position_keys: set[tuple[str, str, str]] = set()
    for row in order_rows:
        row_mode = _normalize_mode_key(mode if mode is not None else row.mode)
        if not _is_active_order_status(row_mode, row.status):
            continue
        row_payload = dict(row.payload_json or {})
        row_notional = safe_float(row.notional_usd, 0.0) or 0.0
        active_notional = _live_active_notional(row_mode, row.status, row_notional, row_payload)
        if active_notional <= 0.0:
            continue
        key = _position_cap_scope_key(
            position_cap_scope=scope,
            mode=row_mode,
            market_id=row.market_id,
            direction=row.direction,
            payload=row_payload,
        )
        if key is not None:
            active_order_position_keys.add(key)

    return max(len(active_position_keys), len(active_order_position_keys))


async def get_open_market_ids_for_trader(
    session: AsyncSession,
    trader_id: str,
    mode: Optional[str] = None,
) -> set[str]:
    position_query = (
        select(TraderPosition.market_id)
        .where(TraderPosition.trader_id == trader_id)
        .where(func.lower(func.coalesce(TraderPosition.status, "")) == ACTIVE_POSITION_STATUS)
    )
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            position_query = position_query.where(
                func.lower(func.coalesce(TraderPosition.mode, "")).not_in(list(_TRADER_MODES))
            )
        else:
            position_query = position_query.where(func.lower(func.coalesce(TraderPosition.mode, "")) == mode_key)

    rows = (await session.execute(position_query)).all()
    open_market_ids: set[str] = set()
    for row in rows:
        market_id = str(row.market_id or "").strip()
        if market_id:
            open_market_ids.add(market_id)

    order_status_expr = func.lower(func.trim(func.coalesce(TraderOrder.status, "")))
    order_query = select(
        TraderOrder.mode,
        TraderOrder.status,
        TraderOrder.market_id,
     ).where(TraderOrder.trader_id == trader_id).where(order_status_expr.in_(tuple(OPEN_ORDER_STATUSES)))
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            order_query = order_query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == "")
        else:
            order_query = order_query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key)

    order_rows = (await session.execute(order_query)).all()
    for row in order_rows:
        row_mode = _normalize_mode_key(mode if mode is not None else row.mode)
        if not _is_active_order_status(row_mode, row.status):
            continue
        market_id = str(row.market_id or "").strip()
        if market_id:
            open_market_ids.add(market_id)
    return open_market_ids


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

    summary = {"live": 0, "shadow": 0, "other": 0, "total": 0}
    for row in rows:
        mode_key = _normalize_mode_key(row.mode)
        count = int(row.count or 0)
        if mode_key == "live":
            summary["live"] += count
        elif mode_key == "shadow":
            summary["shadow"] += count
        else:
            summary["other"] += count
        summary["total"] += count
    return summary


async def get_open_order_count_for_trader(
    session: AsyncSession,
    trader_id: str,
    mode: Optional[str] = None,
) -> int:
    status_key_expr = func.lower(func.trim(func.coalesce(TraderOrder.status, "")))
    query = (
        select(
            TraderOrder.mode,
            TraderOrder.status,
            func.count(TraderOrder.id).label("count"),
        )
        .where(TraderOrder.trader_id == trader_id)
        .where(status_key_expr.in_(tuple(OPEN_ORDER_STATUSES)))
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

    summary = {"live": 0, "shadow": 0, "other": 0, "total": 0}
    for row in rows:
        mode = _normalize_mode_key(row.mode)
        if not _is_active_order_status(mode, row.status):
            continue
        count = int(row.count or 0)
        if mode == "live":
            summary["live"] += count
        elif mode == "shadow":
            summary["shadow"] += count
        else:
            summary["other"] += count
        summary["total"] += count
    return summary


def _pending_live_exit_is_non_terminal(
    value: Any,
    terminal_statuses: set[str] | None = None,
) -> bool:
    if not isinstance(value, dict):
        return False
    status_key = _normalize_status_key(value.get("status"))
    terminal_set = terminal_statuses or set(DEFAULT_PENDING_LIVE_EXIT_GUARD["terminal_statuses"])
    return status_key not in terminal_set


async def get_pending_live_exit_summary_for_trader(
    session: AsyncSession,
    trader_id: str,
    mode: str = "live",
    terminal_statuses: list[str] | set[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    mode_key = _normalize_mode_key(mode)
    if mode_key != "live":
        return {
            "count": 0,
            "order_ids": [],
            "market_ids": [],
            "signal_ids": [],
            "statuses": {},
            "terminal_statuses": list(DEFAULT_PENDING_LIVE_EXIT_GUARD["terminal_statuses"]),
            "identities": [],
            "identity_keys": [],
        }

    rows = (
        await session.execute(
            select(
                TraderOrder.id,
                TraderOrder.status,
                TraderOrder.market_id,
                TraderOrder.direction,
                TraderOrder.signal_id,
                TraderOrder.payload_json,
            )
            .where(TraderOrder.trader_id == trader_id)
            .where(func.lower(func.coalesce(TraderOrder.mode, "")) == "live")
        )
    ).all()

    configured_terminal_statuses = _normalize_pending_live_exit_terminal_statuses(
        list(terminal_statuses) if isinstance(terminal_statuses, (list, set, tuple)) else None
    )
    terminal_status_set = set(configured_terminal_statuses)
    order_ids: list[str] = []
    market_ids: set[str] = set()
    signal_ids: set[str] = set()
    statuses: dict[str, int] = {}
    identities: list[dict[str, Any]] = []
    identity_keys: set[str] = set()
    for row in rows:
        if not _is_active_order_status("live", row.status):
            continue
        payload = dict(row.payload_json or {})
        pending_live_exit = payload.get("pending_live_exit")
        if not _pending_live_exit_is_non_terminal(pending_live_exit, terminal_status_set):
            continue
        order_id = str(row.id or "").strip()
        market_id = str(row.market_id or "").strip()
        direction = str(row.direction or "").strip().lower()
        signal_id = str(row.signal_id or "").strip()
        status_key = _normalize_status_key((pending_live_exit or {}).get("status")) or "unknown"
        statuses[status_key] = int(statuses.get(status_key, 0)) + 1
        if order_id:
            order_ids.append(order_id)
        if market_id:
            market_ids.add(market_id)
        if signal_id:
            signal_ids.add(signal_id)
        identities.append(
            {
                "order_id": order_id or None,
                "market_id": market_id or None,
                "direction": direction or None,
                "signal_id": signal_id or None,
                "status": status_key,
            }
        )
        if market_id and direction:
            identity_keys.add(f"{market_id}|{direction}|{signal_id or '*'}")

    return {
        "count": len(order_ids),
        "order_ids": order_ids,
        "market_ids": sorted(market_ids),
        "signal_ids": sorted(signal_ids),
        "statuses": statuses,
        "terminal_statuses": configured_terminal_statuses,
        "identities": identities,
        "identity_keys": sorted(identity_keys),
    }


def _order_has_fill(row: TraderOrder) -> bool:
    mode_key = _normalize_mode_key(row.mode)
    status_key = _normalize_status_key(row.status)
    payload = dict(row.payload_json or {})
    row_notional = max(0.0, abs(safe_float(row.notional_usd, 0.0) or 0.0))

    if mode_key in {"live", "shadow"}:
        filled_notional_usd, filled_shares, _ = _extract_live_fill_metrics(payload)
        if filled_notional_usd > 0.0 or filled_shares > 0.0:
            return True
        return status_key == "executed" and row_notional > 0.0

    return row_notional > 0.0


async def cleanup_trader_open_orders(
    session: AsyncSession,
    *,
    trader_id: str,
    scope: str = "shadow",
    max_age_hours: Optional[int] = None,
    max_age_seconds: Optional[float] = None,
    source: Optional[str] = None,
    require_unfilled: bool = False,
    dry_run: bool = False,
    target_status: str = "cancelled",
    reason: Optional[str] = None,
    attempt_live_taker_rescue: bool = False,
    live_taker_rescue_price_bps: Optional[float] = None,
    live_taker_rescue_time_in_force: str = DEFAULT_TIMEOUT_TAKER_RESCUE_TIME_IN_FORCE,
) -> dict[str, Any]:
    scope_key = str(scope or "shadow").strip().lower()
    if scope_key == "paper":
        scope_key = "shadow"
    if scope_key == "all":
        allowed_modes = {"shadow", "live", "other"}
    elif scope_key in {"shadow", "live"}:
        allowed_modes = {scope_key}
    else:
        raise ValueError("scope must be one of: shadow, live, all")

    query = select(TraderOrder).where(TraderOrder.trader_id == trader_id)
    rows = list((await session.execute(query)).scalars().all())
    source_filter = normalize_source_key(source) if source is not None else ""
    cutoff: Optional[datetime] = None
    if max_age_seconds is not None:
        cutoff = _now() - timedelta(seconds=max(1.0, float(max_age_seconds)))
    elif max_age_hours is not None:
        cutoff = _now() - timedelta(hours=max(1, int(max_age_hours)))

    candidates: list[TraderOrder] = []
    for row in rows:
        mode_key = _normalize_mode_key(row.mode)
        if mode_key not in allowed_modes:
            continue
        if source_filter and normalize_source_key(row.source) != source_filter:
            continue

        status_key = _normalize_status_key(row.status)
        if status_key not in CLEANUP_ELIGIBLE_ORDER_STATUSES:
            continue
        if require_unfilled and _order_has_fill(row):
            continue

        if cutoff is not None:
            if require_unfilled:
                age_anchor = row.created_at or row.updated_at or row.executed_at
            else:
                age_anchor = row.executed_at or row.updated_at or row.created_at
            if age_anchor is None or age_anchor > cutoff:
                continue

        candidates.append(row)

    mode_breakdown = {"shadow": 0, "live": 0, "other": 0}
    status_breakdown: dict[str, int] = {}
    for row in candidates:
        mode_key = _normalize_mode_key(row.mode)
        status_key = _normalize_status_key(row.status)
        mode_breakdown[mode_key] = int(mode_breakdown.get(mode_key, 0)) + 1
        status_breakdown[status_key] = int(status_breakdown.get(status_key, 0)) + 1

    updated = 0
    provider_cancel_attempted = 0
    provider_cancelled = 0
    provider_cancel_failed = 0
    taker_rescue_attempted = 0
    taker_rescue_succeeded = 0
    taker_rescue_failed = 0
    execution_order_updates = 0
    execution_leg_updates = 0
    execution_session_updates = 0
    if not dry_run and candidates:
        now = _now()
        note_reason = str(reason or "manual_position_cleanup").strip()
        rescue_price_bps = safe_float(live_taker_rescue_price_bps, DEFAULT_TIMEOUT_TAKER_RESCUE_PRICE_BPS)
        rescue_price_bps = max(0.0, float(rescue_price_bps or 0.0))
        rescue_time_in_force = str(live_taker_rescue_time_in_force or DEFAULT_TIMEOUT_TAKER_RESCUE_TIME_IN_FORCE).strip().upper()
        if rescue_time_in_force not in {"IOC", "FOK", "GTC"}:
            rescue_time_in_force = DEFAULT_TIMEOUT_TAKER_RESCUE_TIME_IN_FORCE
        allow_timeout_taker_rescue = bool(attempt_live_taker_rescue and require_unfilled and _cleanup_timeout_reason(note_reason))
        candidate_ids = [str(row.id) for row in candidates]
        execution_order_rows = list(
            (
                await session.execute(
                    select(ExecutionSessionOrder)
                    .where(ExecutionSessionOrder.trader_order_id.in_(candidate_ids))
                    .order_by(desc(ExecutionSessionOrder.created_at))
                )
            )
            .scalars()
            .all()
        )
        execution_order_by_trader_order: dict[str, ExecutionSessionOrder] = {}
        for execution_order in execution_order_rows:
            key = str(execution_order.trader_order_id or "").strip()
            if not key or key in execution_order_by_trader_order:
                continue
            execution_order_by_trader_order[key] = execution_order
        touched_leg_ids: set[str] = set()
        touched_session_ids: set[str] = set()

        for row in candidates:
            mode_key = _normalize_mode_key(row.mode)
            previous_status = str(row.status or "")
            existing_payload = dict(row.payload_json or {})
            execution_order = execution_order_by_trader_order.get(str(row.id))

            if mode_key == "live" and not _is_active_order_status(mode_key, target_status):
                provider_cancel_attempted += 1
                provider_order_id, provider_clob_order_id = _resolve_provider_order_ids(
                    order_payload=existing_payload,
                    session_order=execution_order,
                )
                cancel_targets: list[str] = []
                if provider_clob_order_id:
                    cancel_targets.append(provider_clob_order_id)
                if provider_order_id and provider_order_id not in cancel_targets:
                    cancel_targets.append(provider_order_id)
                if not cancel_targets:
                    provider_cancel_failed += 1
                    raise ValueError(
                        f"Cannot cancel live order {row.id}: missing provider order identifiers in payload/session state."
                    )

                provider_cancel_success = False
                last_target = ""
                cancel_failure_reason = ""
                async with release_conn(session):
                    for target in cancel_targets:
                        last_target = target
                        try:
                            cancelled = await asyncio.wait_for(
                                live_execution_service.cancel_order(target),
                                timeout=LIVE_PROVIDER_CANCEL_TIMEOUT_SECONDS,
                            )
                        except asyncio.TimeoutError:
                            cancelled = False
                            cancel_failure_reason = f"timeout after {LIVE_PROVIDER_CANCEL_TIMEOUT_SECONDS:.1f}s"
                        except Exception as exc:
                            cancelled = False
                            cancel_failure_reason = f"error: {exc}"
                        if cancelled:
                            provider_cancel_success = True
                            cancel_failure_reason = ""
                            break
                        if not cancel_failure_reason:
                            cancel_failure_reason = "provider returned unsuccessful cancellation"

                existing_payload["provider_cancel"] = {
                    "attempted_at": to_iso(now),
                    "provider_order_id": provider_order_id,
                    "provider_clob_order_id": provider_clob_order_id,
                    "targets": cancel_targets,
                    "success": provider_cancel_success,
                    "failure_reason": cancel_failure_reason or None,
                    "timeout_seconds": LIVE_PROVIDER_CANCEL_TIMEOUT_SECONDS,
                }
                if not provider_cancel_success:
                    provider_cancel_failed += 1
                    raise ValueError(
                        f"Provider cancellation failed for live order {row.id} "
                        f"(target={last_target or 'unknown'}, reason={cancel_failure_reason or 'unknown'})."
                    )
                provider_cancelled += 1

                if allow_timeout_taker_rescue and source_filter == "crypto":
                    taker_rescue_attempted += 1
                    rescue = await _attempt_timeout_taker_rescue(
                        row=row,
                        order_payload=existing_payload,
                        session_order=execution_order,
                        rescue_price_bps=rescue_price_bps,
                        rescue_time_in_force=rescue_time_in_force,
                    )
                    existing_payload["timeout_taker_rescue"] = {
                        **rescue,
                        "attempted_at": to_iso(now),
                    }
                    if bool(rescue.get("success")):
                        taker_rescue_succeeded += 1
                        rescue_payload = dict(rescue.get("payload") or {})
                        rescue_status = _normalize_status_key(rescue.get("status"))
                        row.status = "open" if rescue_status in {"executed", "open", "submitted"} else "failed"
                        row.updated_at = now
                        if row.status == "failed":
                            import services.trader_hot_state as _hs
                            _hs.record_order_cancelled(
                                trader_id=str(row.trader_id or ""),
                                mode=str(row.mode or ""),
                                order_id=str(row.id or ""),
                                market_id=str(row.market_id or ""),
                                source=str(row.source or ""),
                                copy_source_wallet=_extract_copy_source_wallet_from_payload(existing_payload),
                            )
                        existing_payload["cleanup"] = {
                            "previous_status": previous_status,
                            "target_status": row.status,
                            "reason": "timeout_taker_rescue",
                            "performed_at": to_iso(now),
                        }
                        if rescue_payload:
                            existing_payload.update(
                                {
                                    "order_id": str(rescue.get("order_id") or rescue_payload.get("order_id") or ""),
                                    "clob_order_id": str(rescue.get("clob_order_id") or rescue_payload.get("clob_order_id") or ""),
                                    "provider_order_id": str(rescue.get("order_id") or rescue_payload.get("order_id") or ""),
                                    "provider_clob_order_id": str(
                                        rescue.get("clob_order_id") or rescue_payload.get("clob_order_id") or ""
                                    ),
                                    "effective_price": safe_float(rescue.get("effective_price"), None),
                                    "filled_size": safe_float(rescue_payload.get("filled_size"), 0.0) or 0.0,
                                    "average_fill_price": safe_float(rescue_payload.get("average_fill_price"), None),
                                    "submission": "timeout_taker_rescue",
                                    "post_only": False,
                                    "time_in_force": rescue_time_in_force,
                                }
                            )
                        row.payload_json = existing_payload
                        if row.reason:
                            row.reason = f"{row.reason} | rescue:timeout_taker_conversion"
                        else:
                            row.reason = "rescue:timeout_taker_conversion"

                        if execution_order is not None:
                            execution_order.status = str(row.status)
                            execution_order.updated_at = now
                            execution_order.provider_order_id = str(rescue.get("order_id") or "") or None
                            execution_order.provider_clob_order_id = str(rescue.get("clob_order_id") or "") or None
                            execution_payload = dict(execution_order.payload_json or {})
                            execution_payload["timeout_taker_rescue"] = {
                                **rescue,
                                "attempted_at": to_iso(now),
                            }
                            execution_order.payload_json = execution_payload
                            execution_order.error_message = None
                            execution_order.reason = "timeout_taker_rescue"
                            execution_order_updates += 1
                        updated += 1
                        continue

                    taker_rescue_failed += 1

            row.status = target_status
            row.updated_at = now
            existing_payload["cleanup"] = {
                "previous_status": previous_status,
                "target_status": target_status,
                "reason": note_reason,
                "performed_at": to_iso(now),
            }
            row.payload_json = existing_payload
            if note_reason:
                if row.reason:
                    row.reason = f"{row.reason} | cleanup:{note_reason}"
                else:
                    row.reason = f"cleanup:{note_reason}"
            import services.trader_hot_state as _hs
            _hs.record_order_cancelled(
                trader_id=str(row.trader_id or ""),
                mode=str(row.mode or ""),
                order_id=str(row.id or ""),
                market_id=str(row.market_id or ""),
                source=str(row.source or ""),
                copy_source_wallet=_extract_copy_source_wallet_from_payload(existing_payload),
            )

            if execution_order is not None and not _is_active_order_status(mode_key, target_status):
                execution_order.status = str(target_status)
                execution_order.updated_at = now
                if note_reason:
                    if execution_order.reason:
                        execution_order.reason = f"{execution_order.reason} | cleanup:{note_reason}"
                    else:
                        execution_order.reason = f"cleanup:{note_reason}"
                    execution_order.error_message = note_reason
                execution_payload = dict(execution_order.payload_json or {})
                execution_payload["cleanup"] = {
                    "target_status": str(target_status),
                    "reason": note_reason,
                    "performed_at": to_iso(now),
                }
                execution_order.payload_json = execution_payload
                execution_order_updates += 1
                leg_id = str(execution_order.leg_id or "").strip()
                if leg_id:
                    touched_leg_ids.add(leg_id)
                session_id = str(execution_order.session_id or "").strip()
                if session_id:
                    touched_session_ids.add(session_id)
            updated += 1

        if touched_leg_ids and not _is_active_order_status(scope_key, target_status):
            leg_target_status = str(target_status).strip().lower()
            if leg_target_status not in {"cancelled", "failed", "expired"}:
                leg_target_status = "cancelled"
            for leg_id in sorted(touched_leg_ids):
                leg_row = await session.get(ExecutionSessionLeg, leg_id)
                if leg_row is None:
                    continue
                leg_status_key = _normalize_status_key(leg_row.status)
                if leg_status_key in {"completed", "failed", "cancelled", "expired"}:
                    continue
                leg_row.status = leg_target_status
                if note_reason:
                    leg_row.last_error = note_reason
                leg_row.updated_at = now
                execution_leg_updates += 1
                session_id = str(leg_row.session_id or "").strip()
                if session_id:
                    touched_session_ids.add(session_id)

        if touched_session_ids and not _is_active_order_status(scope_key, target_status):
            for session_id in sorted(touched_session_ids):
                if not session_id:
                    continue
                session_row = await _refresh_execution_session_rollups(session, session_id=session_id)
                if session_row is None:
                    continue
                session_status_key = _normalize_status_key(session_row.status)
                if session_status_key not in ACTIVE_EXECUTION_SESSION_STATUSES:
                    continue
                legs_open = int(session_row.legs_open or 0)
                legs_failed = int(session_row.legs_failed or 0)
                legs_completed = int(session_row.legs_completed or 0)
                if legs_open <= 0 and legs_failed > 0 and legs_completed == 0:
                    session_row.status = "failed"
                    session_row.error_message = (
                        f"Order cleanup transitioned all active legs to {target_status} ({note_reason})."
                    )
                    session_row.completed_at = now
                    session_row.updated_at = now
                    execution_session_updates += 1
        await _commit_with_retry(session)
        await sync_trader_position_inventory(session, trader_id=trader_id)

    return {
        "trader_id": trader_id,
        "scope": scope_key,
        "max_age_hours": max_age_hours,
        "max_age_seconds": max_age_seconds,
        "source": source_filter or None,
        "require_unfilled": bool(require_unfilled),
        "dry_run": bool(dry_run),
        "target_status": target_status,
        "matched": len(candidates),
        "updated": updated,
        "by_mode": mode_breakdown,
        "by_status": status_breakdown,
        "provider_cancel_attempted": provider_cancel_attempted,
        "provider_cancelled": provider_cancelled,
        "provider_cancel_failed": provider_cancel_failed,
        "taker_rescue_attempted": taker_rescue_attempted,
        "taker_rescue_succeeded": taker_rescue_succeeded,
        "taker_rescue_failed": taker_rescue_failed,
        "execution_order_updates": execution_order_updates,
        "execution_leg_updates": execution_leg_updates,
        "execution_session_updates": execution_session_updates,
    }


async def get_market_exposure(session: AsyncSession, market_id: str, mode: Optional[str] = None) -> float:
    query = select(TraderOrder).where(TraderOrder.market_id == market_id)
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == "")
        else:
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key)
    rows = list((await session.execute(query)).scalars().all())
    return float(sum(_active_order_notional_for_metrics(row) for row in rows))


async def get_trader_source_exposure(
    session: AsyncSession,
    *,
    trader_id: str,
    source: str,
    mode: Optional[str] = None,
) -> float:
    source_key = str(source or "").strip().lower()
    if not source_key:
        return 0.0

    query = select(TraderOrder).where(
        TraderOrder.trader_id == trader_id,
        func.lower(func.coalesce(TraderOrder.source, "")) == source_key,
    )
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == "")
        else:
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key)
    rows = list((await session.execute(query)).scalars().all())
    return float(sum(_active_order_notional_for_metrics(row) for row in rows))


async def get_trader_copy_leader_exposure(
    session: AsyncSession,
    *,
    trader_id: str,
    source_wallet: str,
    mode: Optional[str] = None,
) -> float:
    normalized_wallet = StrategySDK.normalize_trader_wallet(source_wallet)
    if not normalized_wallet:
        return 0.0

    query = select(TraderOrder).where(
        TraderOrder.trader_id == trader_id,
        func.lower(func.coalesce(TraderOrder.source, "")) == "traders",
    )
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == "")
        else:
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key)

    rows = list((await session.execute(query)).scalars().all())
    exposure = 0.0
    for row in rows:
        row_payload = dict(row.payload_json or {})
        row_source_wallet = _extract_copy_source_wallet_from_payload(row_payload)
        if row_source_wallet != normalized_wallet:
            continue
        exposure += _active_order_notional_for_metrics(row)
    return float(exposure)


async def get_trader_copy_analytics(
    session: AsyncSession,
    *,
    trader_id: str,
    mode: Optional[str] = None,
    limit: int = 2000,
    leader_limit: int = 20,
) -> dict[str, Any]:
    capped_limit = max(1, min(int(limit or 2000), 10_000))
    query = (
        select(TraderOrder)
        .where(TraderOrder.trader_id == trader_id)
        .order_by(desc(TraderOrder.created_at), desc(TraderOrder.id))
        .limit(capped_limit)
    )
    mode_key: str | None = None
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == "")
        else:
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key)

    rows = list((await session.execute(query)).scalars().all())
    filtered_rows: list[tuple[TraderOrder, dict[str, Any], str]] = []
    for row in rows:
        if str(row.source or "").strip().lower() != "traders":
            continue
        row_payload = dict(row.payload_json or {})
        source_wallet = _extract_copy_source_wallet_from_payload(row_payload)
        strategy_key = str(row.strategy_key or "").strip().lower()
        if not source_wallet and strategy_key != "traders_copy_trade":
            continue
        filtered_rows.append((row, row_payload, source_wallet or "unknown"))

    total_orders = len(filtered_rows)
    total_notional_usd = 0.0
    open_exposure_usd = 0.0
    realized_orders = 0
    realized_pnl_usd = 0.0
    slippage_sum = 0.0
    slippage_count = 0
    adverse_slippage_sum = 0.0
    adverse_slippage_count = 0
    latency_sum = 0.0
    latency_count = 0

    per_leader: dict[str, dict[str, Any]] = {}
    for row, row_payload, leader_wallet in filtered_rows:
        status_key = _normalize_status_key(row.status)
        row_mode = _normalize_mode_key(row.mode)
        row_notional_usd = abs(safe_float(row.notional_usd, 0.0))
        row_active_exposure = _active_order_notional_for_metrics(row)
        total_notional_usd += row_notional_usd
        open_exposure_usd += row_active_exposure

        if status_key in REALIZED_ORDER_STATUSES:
            realized_orders += 1
            realized_pnl_usd += safe_float(row.actual_profit, 0.0)

        copy_attribution = row_payload.get("copy_attribution")
        copy_attribution = copy_attribution if isinstance(copy_attribution, dict) else {}
        row_slippage_bps = safe_float(copy_attribution.get("slippage_bps"), None)
        row_adverse_slippage_bps = safe_float(copy_attribution.get("adverse_slippage_bps"), None)
        row_copy_latency_ms = safe_float(copy_attribution.get("copy_latency_ms"), None)

        if row_slippage_bps is not None:
            slippage_sum += row_slippage_bps
            slippage_count += 1
        if row_adverse_slippage_bps is not None:
            adverse_slippage_sum += row_adverse_slippage_bps
            adverse_slippage_count += 1
        if row_copy_latency_ms is not None:
            latency_sum += row_copy_latency_ms
            latency_count += 1

        leader = per_leader.get(leader_wallet)
        if leader is None:
            leader = {
                "source_wallet": leader_wallet,
                "orders": 0,
                "active_orders": 0,
                "realized_orders": 0,
                "realized_pnl_usd": 0.0,
                "gross_notional_usd": 0.0,
                "open_exposure_usd": 0.0,
                "slippage_sum": 0.0,
                "slippage_count": 0,
                "adverse_slippage_sum": 0.0,
                "adverse_slippage_count": 0,
                "latency_sum": 0.0,
                "latency_count": 0,
            }
            per_leader[leader_wallet] = leader

        leader["orders"] += 1
        leader["gross_notional_usd"] += row_notional_usd
        leader["open_exposure_usd"] += row_active_exposure
        if _is_active_order_status(row_mode, row.status):
            leader["active_orders"] += 1
        if status_key in REALIZED_ORDER_STATUSES:
            leader["realized_orders"] += 1
            leader["realized_pnl_usd"] += safe_float(row.actual_profit, 0.0)
        if row_slippage_bps is not None:
            leader["slippage_sum"] += row_slippage_bps
            leader["slippage_count"] += 1
        if row_adverse_slippage_bps is not None:
            leader["adverse_slippage_sum"] += row_adverse_slippage_bps
            leader["adverse_slippage_count"] += 1
        if row_copy_latency_ms is not None:
            leader["latency_sum"] += row_copy_latency_ms
            leader["latency_count"] += 1

    resolved_orders = sum(1 for row, _, _ in filtered_rows if _normalize_status_key(row.status) in REALIZED_ORDER_STATUSES)
    win_orders = sum(1 for row, _, _ in filtered_rows if _normalize_status_key(row.status) in REALIZED_WIN_ORDER_STATUSES)
    loss_orders = sum(1 for row, _, _ in filtered_rows if _normalize_status_key(row.status) in REALIZED_LOSS_ORDER_STATUSES)
    win_rate_pct = (win_orders / resolved_orders * 100.0) if resolved_orders > 0 else None

    sorted_leaders = sorted(
        per_leader.values(),
        key=lambda item: (
            -float(item.get("gross_notional_usd") or 0.0),
            -int(item.get("orders") or 0),
            str(item.get("source_wallet") or ""),
        ),
    )
    leader_rows: list[dict[str, Any]] = []
    for leader in sorted_leaders[: max(1, min(int(leader_limit or 20), 500))]:
        leader_rows.append(
            {
                "source_wallet": leader["source_wallet"],
                "orders": int(leader["orders"]),
                "active_orders": int(leader["active_orders"]),
                "realized_orders": int(leader["realized_orders"]),
                "realized_pnl_usd": float(leader["realized_pnl_usd"]),
                "gross_notional_usd": float(leader["gross_notional_usd"]),
                "open_exposure_usd": float(leader["open_exposure_usd"]),
                "avg_slippage_bps": (
                    float(leader["slippage_sum"] / leader["slippage_count"])
                    if int(leader["slippage_count"] or 0) > 0
                    else None
                ),
                "avg_adverse_slippage_bps": (
                    float(leader["adverse_slippage_sum"] / leader["adverse_slippage_count"])
                    if int(leader["adverse_slippage_count"] or 0) > 0
                    else None
                ),
                "avg_copy_latency_ms": (
                    float(leader["latency_sum"] / leader["latency_count"])
                    if int(leader["latency_count"] or 0) > 0
                    else None
                ),
            }
        )

    return {
        "trader_id": trader_id,
        "mode": mode_key or "all",
        "as_of": to_iso(_now()),
        "sample_size": total_orders,
        "scanned_orders": len(rows),
        "summary": {
            "total_orders": total_orders,
            "resolved_orders": resolved_orders,
            "win_orders": win_orders,
            "loss_orders": loss_orders,
            "win_rate_pct": win_rate_pct,
            "realized_pnl_usd": float(realized_pnl_usd),
            "gross_notional_usd": float(total_notional_usd),
            "open_exposure_usd": float(open_exposure_usd),
            "avg_slippage_bps": (float(slippage_sum / slippage_count) if slippage_count > 0 else None),
            "avg_adverse_slippage_bps": (
                float(adverse_slippage_sum / adverse_slippage_count) if adverse_slippage_count > 0 else None
            ),
            "avg_copy_latency_ms": (float(latency_sum / latency_count) if latency_count > 0 else None),
            "distinct_leaders": len(per_leader),
        },
        "leaders": leader_rows,
    }


async def get_gross_exposure(session: AsyncSession, mode: Optional[str] = None) -> float:
    query = select(TraderOrder)
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == "")
        else:
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key)
    rows = list((await session.execute(query)).scalars().all())
    return float(sum(_active_order_notional_for_metrics(row) for row in rows))


async def get_realized_pnl(
    session: AsyncSession,
    *,
    trader_id: Optional[str] = None,
    mode: Optional[str] = None,
    since: Optional[datetime] = None,
) -> float:
    query = select(func.coalesce(func.sum(TraderOrder.actual_profit), 0.0)).where(
        TraderOrder.status.in_(tuple(REALIZED_ORDER_STATUSES))
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


async def get_unrealized_pnl(
    session: AsyncSession,
    *,
    trader_id: Optional[str] = None,
    mode: Optional[str] = None,
) -> float:
    """Compute mark-to-market unrealized PnL for all active orders.

    For each open position with an entry price and notional, fetches the
    current live mid-price and computes unrealized PnL as:
        quantity * (current_price - entry_price)
    where quantity = notional / entry_price.

    Returns the total unrealized PnL in USD (negative means the open
    positions are currently losing money).
    """
    from services.live_price_snapshot import get_live_mid_prices

    # Gather active orders with entry prices and token IDs
    query = select(TraderOrder)
    if trader_id:
        query = query.where(TraderOrder.trader_id == trader_id)
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == "")
        else:
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key)

    rows = list((await session.execute(query)).scalars().all())

    # Filter to active orders only and collect token IDs for price lookup
    active_orders = []
    token_ids: list[str] = []
    for order in rows:
        order_mode = _normalize_mode_key(order.mode)
        if not _is_active_order_status(order_mode, order.status):
            continue

        payload = dict(order.payload_json or {})
        position_state = payload.get("position_state")
        position_state = position_state if isinstance(position_state, dict) else {}
        row_notional = safe_float(order.notional_usd, 0.0) or 0.0
        notional = _live_active_notional(order_mode, order.status, row_notional, payload)
        _, filled_shares, fill_price = _extract_live_fill_metrics(payload)
        entry_price = (
            fill_price
            if fill_price is not None and fill_price > 0
            else safe_float(order.effective_price, 0.0) or safe_float(order.entry_price, 0.0) or 0.0
        )
        if entry_price <= 0 or notional <= 0:
            continue
        quantity = (
            float(filled_shares)
            if order_mode in {"live", "shadow"} and filled_shares > 0.0
            else float(notional / entry_price if entry_price > 0 else 0.0)
        )
        if quantity <= 0:
            continue

        token_id = str(payload.get("token_id") or payload.get("selected_token_id") or "").strip()
        if not token_id:
            continue

        active_orders.append(
            (
                order,
                entry_price,
                quantity,
                token_id,
                safe_float(position_state.get("last_mark_price"), 0.0) or 0.0,
            )
        )
        token_ids.append(token_id)

    if not active_orders:
        return 0.0

    # Fetch current prices in batch
    try:
        prices = await get_live_mid_prices(token_ids)
    except Exception:
        prices = {}

    total_unrealized = 0.0
    for order, entry_price, quantity, token_id, last_mark_price in active_orders:
        current_price = prices.get(token_id)
        if (current_price is None or current_price <= 0) and last_mark_price > 0:
            current_price = last_mark_price
        if current_price is None or current_price <= 0:
            continue
        direction = str(order.direction or "").strip().lower()
        if direction in ("sell", "no", "short"):
            # Short position: profit when price drops
            total_unrealized += quantity * (entry_price - current_price)
        else:
            # Long position (default): profit when price rises
            total_unrealized += quantity * (current_price - entry_price)

    return total_unrealized


async def get_consecutive_loss_count(
    session: AsyncSession,
    *,
    trader_id: str,
    mode: Optional[str] = None,
    limit: int = 100,
    since: Optional[datetime] = None,
) -> int:
    query = (
        select(TraderOrder.status, TraderOrder.updated_at)
        .where(TraderOrder.trader_id == trader_id)
        .where(TraderOrder.status.in_(tuple(REALIZED_ORDER_STATUSES)))
        .order_by(desc(TraderOrder.updated_at), desc(TraderOrder.id))
        .limit(max(1, min(int(limit or 100), 1000)))
    )
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == "")
        else:
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key)
    if since is not None:
        since_utc = since.replace(tzinfo=timezone.utc) if since.tzinfo is None else since.astimezone(timezone.utc)
        query = query.where(TraderOrder.updated_at >= since_utc)

    rows = (await session.execute(query)).all()
    losses = 0
    for row in rows:
        status = _normalize_status_key(row.status)
        if status in REALIZED_LOSS_ORDER_STATUSES:
            losses += 1
            continue
        if status in REALIZED_WIN_ORDER_STATUSES:
            break
    return losses


async def get_last_resolved_loss_at(
    session: AsyncSession,
    *,
    trader_id: str,
    mode: Optional[str] = None,
    since: Optional[datetime] = None,
) -> Optional[datetime]:
    query = select(func.max(TraderOrder.updated_at)).where(
        TraderOrder.trader_id == trader_id,
        TraderOrder.status.in_(tuple(REALIZED_LOSS_ORDER_STATUSES)),
    )
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == "")
        else:
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key)
    if since is not None:
        since_utc = since.replace(tzinfo=timezone.utc) if since.tzinfo is None else since.astimezone(timezone.utc)
        query = query.where(TraderOrder.updated_at >= since_utc)
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
    execution_sessions_count = int((await session.execute(select(func.count(ExecutionSession.id)))).scalar() or 0)
    active_execution_sessions = int(
        (
            await session.execute(
                select(func.count(ExecutionSession.id)).where(
                    func.lower(func.coalesce(ExecutionSession.status, "")).in_(tuple(ACTIVE_EXECUTION_SESSION_STATUSES))
                )
            )
        ).scalar()
        or 0
    )
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
        "execution_sessions_count": execution_sessions_count,
        "active_execution_sessions": active_execution_sessions,
        "open_orders": open_orders,
        "gross_exposure_usd": await get_gross_exposure(session),
        "daily_pnl": daily_pnl,
    }


async def compose_trader_orchestrator_config(session: AsyncSession) -> dict[str, Any]:
    control = await read_orchestrator_control(session)
    settings_json = _normalize_control_settings(control.get("settings") or {})
    global_risk = dict(settings_json.get("global_risk") or _normalize_global_risk({}))
    global_runtime = dict(settings_json.get("global_runtime") or _normalize_global_runtime_settings({}))
    pending_live_exit_guard = dict(
        global_runtime.get("pending_live_exit_guard") or _normalize_pending_live_exit_guard({})
    )
    live_risk_clamps = dict(global_runtime.get("live_risk_clamps") or _normalize_live_risk_clamps({}))
    live_market_context = dict(global_runtime.get("live_market_context") or _normalize_live_market_context({}))
    live_provider_health = dict(global_runtime.get("live_provider_health") or _normalize_live_provider_health({}))
    return {
        "mode": control.get("mode", "shadow"),
        "kill_switch": bool(control.get("kill_switch", False)),
        "run_interval_seconds": int(control.get("run_interval_seconds") or ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS),
        "global_risk": {
            "max_gross_exposure_usd": safe_float(global_risk.get("max_gross_exposure_usd"), 5000.0),
            "max_daily_loss_usd": safe_float(global_risk.get("max_daily_loss_usd"), 500.0),
            "max_orders_per_cycle": safe_int(global_risk.get("max_orders_per_cycle"), 50),
        },
        "trading_domains": settings_json.get("trading_domains") or ["event_markets", "crypto"],
        "enabled_strategies": settings_json.get("enabled_strategies") or [],
        "llm_verify_trades": bool(settings_json.get("llm_verify_trades", False)),
        "global_runtime": {
            "pending_live_exit_guard": {
                "max_pending_exits": int(pending_live_exit_guard.get("max_pending_exits", 0) or 0),
                "identity_guard_enabled": bool(pending_live_exit_guard.get("identity_guard_enabled", True)),
                "terminal_statuses": list(
                    pending_live_exit_guard.get("terminal_statuses")
                    or DEFAULT_PENDING_LIVE_EXIT_GUARD["terminal_statuses"]
                ),
            },
            "live_risk_clamps": {
                "enforce_allow_averaging_off": bool(live_risk_clamps.get("enforce_allow_averaging_off", True)),
                "min_cooldown_seconds": int(live_risk_clamps.get("min_cooldown_seconds", 0) or 0),
                "max_consecutive_losses_cap": int(live_risk_clamps.get("max_consecutive_losses_cap", 1) or 1),
                "max_open_orders_cap": int(live_risk_clamps.get("max_open_orders_cap", 1) or 1),
                "max_open_positions_cap": int(live_risk_clamps.get("max_open_positions_cap", 1) or 1),
                "max_trade_notional_usd_cap": float(live_risk_clamps.get("max_trade_notional_usd_cap", 1.0) or 1.0),
                "max_orders_per_cycle_cap": int(live_risk_clamps.get("max_orders_per_cycle_cap", 1) or 1),
                "enforce_halt_on_consecutive_losses": bool(
                    live_risk_clamps.get("enforce_halt_on_consecutive_losses", True)
                ),
            },
            "live_market_context": {
                "enabled": bool(live_market_context.get("enabled", True)),
                "history_window_seconds": int(live_market_context.get("history_window_seconds", 7200) or 7200),
                "history_fidelity_seconds": int(live_market_context.get("history_fidelity_seconds", 300) or 300),
                "max_history_points": int(live_market_context.get("max_history_points", 120) or 120),
                "timeout_seconds": float(live_market_context.get("timeout_seconds", 4.0) or 4.0),
                "strict_ws_pricing_only": bool(
                    live_market_context.get(
                        "strict_ws_pricing_only",
                        DEFAULT_LIVE_MARKET_CONTEXT["strict_ws_pricing_only"],
                    )
                ),
                "allow_redis_strict_prices": bool(
                    live_market_context.get(
                        "allow_redis_strict_prices",
                        DEFAULT_LIVE_MARKET_CONTEXT["allow_redis_strict_prices"],
                    )
                ),
                "max_market_data_age_ms": int(
                    live_market_context.get(
                        "max_market_data_age_ms",
                        DEFAULT_LIVE_MARKET_CONTEXT["max_market_data_age_ms"],
                    )
                    or DEFAULT_LIVE_MARKET_CONTEXT["max_market_data_age_ms"]
                ),
            },
            "live_provider_health": {
                "window_seconds": int(live_provider_health.get("window_seconds", 180) or 180),
                "min_errors": int(live_provider_health.get("min_errors", 2) or 2),
                "block_seconds": int(live_provider_health.get("block_seconds", 120) or 120),
            },
            "trader_cycle_timeout_seconds": (
                float(global_runtime.get("trader_cycle_timeout_seconds"))
                if global_runtime.get("trader_cycle_timeout_seconds") is not None
                else None
            ),
        },
    }


async def get_orchestrator_overview(session: AsyncSession) -> dict[str, Any]:
    return {
        "control": await read_orchestrator_control(session),
        "worker": await read_orchestrator_snapshot(session),
        "reconciliation_worker": await read_worker_snapshot(session, "trader_reconciliation"),
        "config": await compose_trader_orchestrator_config(session),
        "metrics": await compute_orchestrator_metrics(session),
        "traders": await list_traders(session),
        "execution_sessions": await list_serialized_execution_sessions(session, limit=200),
    }


async def _build_preflight_checks(
    session: AsyncSession,
    control: dict[str, Any],
    trader_count: int,
) -> list[dict[str, Any]]:
    app_settings = await session.get(AppSettings, "default")
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
        "created_at": to_iso(_now()),
    }

    settings_json = dict(control_row.settings_json or {})
    settings_json["live_preflight"] = preflight
    control_row.settings_json = settings_json
    control_row.updated_at = _now()
    await _commit_with_retry(session)

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
        "expires_at": to_iso(expires_at),
        "consumed_at": None,
        "requested_by": requested_by,
    }

    settings_json["live_arm"] = arm_data
    control_row.settings_json = settings_json
    control_row.updated_at = _now()
    await _commit_with_retry(session)

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

    arm_data["consumed_at"] = to_iso(_now())
    settings_json["live_arm"] = arm_data
    control_row.settings_json = settings_json
    control_row.updated_at = _now()
    await _commit_with_retry(session)
    return True


async def list_serialized_trader_decisions(
    session: AsyncSession,
    *,
    trader_id: Optional[str] = None,
    trader_ids: Optional[list[str]] = None,
    decision: Optional[str] = None,
    limit: int = 200,
    per_trader_limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    rows = await list_trader_decisions(
        session,
        trader_id=trader_id,
        trader_ids=trader_ids,
        decision=decision,
        limit=limit,
        per_trader_limit=per_trader_limit,
    )
    decision_ids = sorted({str(row.id).strip() for row in rows if str(row.id or "").strip()})
    signal_ids = sorted({str(row.signal_id).strip() for row in rows if str(row.signal_id or "").strip()})
    signals_by_id: dict[str, TradeSignal] = {}
    if signal_ids:
        signal_rows = (await session.execute(select(TradeSignal).where(TradeSignal.id.in_(signal_ids)))).scalars().all()
        signals_by_id = {str(signal_row.id): signal_row for signal_row in signal_rows}

    failed_checks_by_decision: dict[str, list[dict[str, Any]]] = {}
    if decision_ids:
        failed_check_rows = (
            (
                await session.execute(
                    select(TraderDecisionCheck)
                    .where(
                        TraderDecisionCheck.decision_id.in_(decision_ids),
                        TraderDecisionCheck.passed.is_(False),
                    )
                    .order_by(TraderDecisionCheck.created_at.asc())
                )
            )
            .scalars()
            .all()
        )
        for check_row in failed_check_rows:
            decision_key = str(check_row.decision_id or "").strip()
            if not decision_key:
                continue
            failed_checks_by_decision.setdefault(decision_key, []).append(
                {
                    "check_key": str(check_row.check_key or ""),
                    "check_label": str(check_row.check_label or ""),
                    "detail": check_row.detail,
                    "score": check_row.score,
                    "payload": check_row.payload_json or {},
                    "created_at": to_iso(check_row.created_at),
                }
            )

    serialized_rows: list[dict[str, Any]] = []
    for row in rows:
        serialized = _serialize_decision_with_signal(row, signals_by_id.get(str(row.signal_id or "")))
        failed_checks = failed_checks_by_decision.get(str(row.id or ""), [])
        serialized["failed_checks"] = failed_checks
        serialized["failed_check_count"] = len(failed_checks)
        serialized_rows.append(serialized)

    return serialized_rows


async def list_serialized_trader_orders(
    session: AsyncSession,
    *,
    trader_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    rows = await list_trader_orders(
        session,
        trader_id=trader_id,
        status=status,
        limit=limit,
    )
    signal_ids = sorted({str(row.signal_id).strip() for row in rows if str(row.signal_id or "").strip()})
    signals_by_id: dict[str, TradeSignal] = {}
    if signal_ids:
        signal_rows = (await session.execute(select(TradeSignal).where(TradeSignal.id.in_(signal_ids)))).scalars().all()
        signals_by_id = {str(signal_row.id): signal_row for signal_row in signal_rows}
    condition_ids_for_rows = [_extract_order_condition_id(row) for row in rows]
    condition_ids = sorted({condition_id for condition_id in condition_ids_for_rows if condition_id})
    cached_markets_by_condition_id: dict[str, CachedMarket] = {}
    if condition_ids:
        cached_market_rows = (
            (
                await session.execute(
                    select(CachedMarket).where(func.lower(CachedMarket.condition_id).in_(condition_ids))
                )
            )
            .scalars()
            .all()
        )
        for cached_market in cached_market_rows:
            key = _normalize_condition_id(cached_market.condition_id)
            if key:
                cached_markets_by_condition_id[key] = cached_market

    serialized_rows: list[dict[str, Any]] = []
    for row, condition_id in zip(rows, condition_ids_for_rows):
        serialized_rows.append(
            _serialize_order(
                row,
                signals_by_id.get(str(row.signal_id or "")),
                cached_markets_by_condition_id.get(condition_id),
            )
        )
    return serialized_rows


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


async def list_serialized_execution_sessions(
    session: AsyncSession,
    *,
    trader_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    return [
        _serialize_execution_session(row)
        for row in await list_execution_sessions(
            session,
            trader_id=trader_id,
            status=status,
            limit=limit,
        )
    ]


async def get_serialized_execution_session_detail(
    session: AsyncSession,
    session_id: str,
) -> dict[str, Any] | None:
    return await get_execution_session_detail(session, session_id)
