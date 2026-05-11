"""Normalized trade signal bus helpers.

Worker pipelines emit into ``trade_signals`` and consumers (trader orchestrator/UI)
read from one normalized contract.

# Helpers for WRITING/UPSERTING signals to DB from opportunities.
# For helpers that READ signal data back from DB TradeSignal rows (e.g. for
# strategy evaluate/should_exit), see utils/signal_helpers.py (signal_payload).
"""

from __future__ import annotations

import copy
import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from utils.utcnow import utcnow
from typing import Any, Optional

from config import settings
from sqlalchemy import case, func, or_, select, update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    TradeSignal,
    TradeSignalEmission,
)
from services.event_bus import event_bus
from services.market_roster import build_market_roster, ensure_market_roster_payload
from models.opportunity import Opportunity
from services.market_tradability import get_market_tradability_map
from utils.logger import get_logger
from utils.signal_helpers import normalize_position_side

logger = get_logger("signal_bus")


SIGNAL_TERMINAL_STATUSES = {"executed", "skipped", "expired", "failed"}
SIGNAL_ACTIVE_STATUSES = {"pending", "selected", "submitted"}
SIGNAL_REACTIVATABLE_STATUSES = {"selected", "submitted", "executed", "skipped", "expired", "failed"}


def _scanner_skipped_reactivation_cooldown_seconds() -> float:
    raw = getattr(settings, "SCANNER_SKIPPED_SIGNAL_REACTIVATION_COOLDOWN_SECONDS", 0.0)
    return max(
        0.0,
        float(0.0 if raw is None else raw),
    )
_SKIPPED_REACTIVATION_VOLATILE_KEYS = {
    "bridge_run_at",
    "bridge_source",
    "book_updated_at",
    "best_ask",
    "best_bid",
    "current_no_price",
    "current_yes_price",
    "ingested_at",
    "last_price_update",
    "last_trade_price",
    "liquidity",
    "market_data_age_ms",
    "mid_price",
    "observed_at",
    "no_price",
    "outcome_prices",
    "price_age_seconds",
    "price_updated_at",
    "published_at",
    "quote_age_seconds",
    "quote_updated_at",
    "runtime_sequence",
    "roster_hash",
    "scan_completed_at",
    "scan_started_at",
    "sequence",
    "signal_emitted_at",
    "updated_at",
    "volume",
    "yes_price",
}
_SKIPPED_REACTIVATION_FLAG_KEYS = {
    "asset",
    "condition_id",
    "direction",
    "event_slug",
    "timeframe",
    "market_id",
    "market_slug",
    "mode",
    "no_token_id",
    "outcome",
    "price_policy",
    "regime",
    "selected_market_id",
    "selected_direction",
    "selected_outcome",
    "selected_token_id",
    "side",
    "strategy_mode",
    "time_in_force",
    "token_id",
    "yes_token_id",
    "oracle_available",
    "oracle_source",
    "is_live",
    "is_current",
}
_ACTIVE_SIGNAL_PRICE_DELTA = 0.002
_ACTIVE_SIGNAL_EDGE_DELTA = 0.10
_ACTIVE_SIGNAL_CONFIDENCE_DELTA = 0.005
_ACTIVE_SIGNAL_LIQUIDITY_DELTA_ABS = 100.0
_ACTIVE_SIGNAL_LIQUIDITY_DELTA_REL = 0.05
_SKIPPED_REACTIVATION_PRICE_DELTA = 0.005
_SKIPPED_REACTIVATION_EDGE_DELTA = 0.25
_SKIPPED_REACTIVATION_CONFIDENCE_DELTA = 0.01
_SKIPPED_REACTIVATION_LIQUIDITY_DELTA_ABS = 250.0
_SKIPPED_REACTIVATION_LIQUIDITY_DELTA_REL = 0.10
_SKIPPED_REACTIVATION_EXPIRY_DELTA_SECONDS = 15.0
_SKIPPED_REACTIVATION_LIQUIDITY_BANDS = (250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 32000.0)
_ACTIVE_REFRESH_SOURCES = {"crypto", "scanner", "traders", "weather"}
_RUNTIME_SEQUENCE_UNSET = object()

_EXPIRE_SOURCE_GRACE_SECONDS: float = 60.0


def _utc_now() -> datetime:
    return utcnow()


def _to_utc_naive(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _safe_json(value: Any) -> Any:
    """Ensure payload is JSON-serializable."""
    if value is None:
        return None
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return {"raw": str(value)}


def _normalize_reactivation_value(value: Any) -> Any:
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for raw_key, raw_item in value.items():
            key = str(raw_key)
            if key in _SKIPPED_REACTIVATION_VOLATILE_KEYS:
                continue
            normalized[key] = _normalize_reactivation_value(raw_item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_reactivation_value(item) for item in value]
    if isinstance(value, float):
        return round(float(value), 4)
    return value


_PAYLOAD_AUTO_EMITTED_KEYS = frozenset({"signal_emitted_at"})


def _strip_auto_emitted_keys(payload: Any) -> Any:
    """Strip only keys auto-stamped by upsert_trade_signal on every call.

    Used for payload refresh comparisons in the active-refresh path —
    strips just ``signal_emitted_at`` so volatile data fields like
    ``price_age_seconds`` are still detected as changes, while avoiding
    spurious dirty-row writes when only the timestamp differs.
    """
    if isinstance(payload, dict):
        return {k: v for k, v in payload.items() if k not in _PAYLOAD_AUTO_EMITTED_KEYS}
    return payload


def _normalize_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return None
    return round(parsed, 10)


def _normalize_reason_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned = [str(item).strip() for item in values if str(item).strip()]
    cleaned.sort()
    return cleaned


def _should_reactivate_unchanged_skipped_scanner_signal(row: TradeSignal, *, source: str) -> bool:
    normalized_source = str(source or "").strip().lower()
    existing_source = str(row.source or "").strip().lower()
    if normalized_source != "scanner" or existing_source != "scanner":
        return False
    updated_at = _to_utc_naive(row.updated_at) or _to_utc_naive(row.created_at)
    if updated_at is None:
        return True
    now = _to_utc_naive(_utc_now())
    if now is None:
        return False
    cooldown_seconds = _scanner_skipped_reactivation_cooldown_seconds()
    if cooldown_seconds <= 0.0:
        return False
    return (now - updated_at).total_seconds() >= cooldown_seconds


def _has_signal_material_change(
    row: TradeSignal,
    *,
    source_item_id: Optional[str],
    signal_type: str,
    strategy_type: Optional[str],
    market_id: str,
    market_question: Optional[str],
    direction: Optional[str],
    entry_price: Optional[float],
    edge_percent: Optional[float],
    confidence: Optional[float],
    liquidity: Optional[float],
    payload_json: Optional[dict[str, Any]],
    strategy_context_json: Optional[dict[str, Any]],
    quality_passed: Optional[bool],
    quality_rejection_reasons: Optional[list],
) -> bool:
    if str(row.source_item_id or "") != str(source_item_id or ""):
        return True
    if str(row.signal_type or "") != str(signal_type or ""):
        return True
    if str(row.strategy_type or "") != str(strategy_type or ""):
        return True
    if str(row.market_id or "") != str(market_id or ""):
        return True
    if str(row.market_question or "") != str(market_question or ""):
        return True
    if str(row.direction or "") != str(direction or ""):
        return True
    if _delta_crosses_threshold(row.entry_price, entry_price, abs_threshold=_ACTIVE_SIGNAL_PRICE_DELTA):
        return True
    if _delta_crosses_threshold(row.edge_percent, edge_percent, abs_threshold=_ACTIVE_SIGNAL_EDGE_DELTA):
        return True
    if _delta_crosses_threshold(
        row.confidence,
        confidence,
        abs_threshold=_ACTIVE_SIGNAL_CONFIDENCE_DELTA,
    ):
        return True
    if _delta_crosses_threshold(
        row.liquidity,
        liquidity,
        abs_threshold=_ACTIVE_SIGNAL_LIQUIDITY_DELTA_ABS,
        rel_threshold=_ACTIVE_SIGNAL_LIQUIDITY_DELTA_REL,
    ):
        return True
    existing_raw = _safe_json(row.payload_json)
    existing_normalized = ensure_market_roster_payload(
        existing_raw, market_id=market_id, market_question=market_question
    )
    existing_payload = _normalize_reactivation_value(existing_normalized)
    incoming_payload = _normalize_reactivation_value(_safe_json(payload_json))
    if existing_payload != incoming_payload:
        return True
    existing_context = _normalize_reactivation_value(_safe_json(row.strategy_context_json))
    incoming_context = _normalize_reactivation_value(_safe_json(strategy_context_json))
    if existing_context != incoming_context:
        return True
    if quality_passed is not None:
        incoming_quality_passed = bool(quality_passed)
        if bool(row.quality_passed) != incoming_quality_passed:
            return True
        existing_reasons = _normalize_reason_list(row.quality_rejection_reasons)
        incoming_reasons = _normalize_reason_list(quality_rejection_reasons)
        if existing_reasons != incoming_reasons:
            return True
    return False


def _delta_crosses_threshold(
    existing: Any,
    incoming: Any,
    *,
    abs_threshold: float,
    rel_threshold: float = 0.0,
) -> bool:
    existing_value = _normalize_number(existing)
    incoming_value = _normalize_number(incoming)
    if existing_value is None or incoming_value is None:
        return existing_value != incoming_value
    delta = abs(incoming_value - existing_value)
    if delta >= abs_threshold:
        return True
    if rel_threshold <= 0.0:
        return False
    baseline = max(abs(existing_value), abs(incoming_value), 1.0)
    return (delta / baseline) >= rel_threshold


def _liquidity_band(value: Any) -> int:
    parsed = _normalize_number(value)
    if parsed is None:
        return -1
    band = 0
    for threshold in _SKIPPED_REACTIVATION_LIQUIDITY_BANDS:
        if parsed >= threshold:
            band += 1
    return band


def _extract_reactivation_flags(payload: Any, context: Any) -> dict[str, Any]:
    flags: dict[str, Any] = {}
    for raw_scope, source in (("payload", payload), ("context", context)):
        if not isinstance(source, dict):
            continue
        for key in _SKIPPED_REACTIVATION_FLAG_KEYS:
            if key in source:
                value = source.get(key)
                if isinstance(value, (str, int, float, bool)) or value is None:
                    flags[f"{raw_scope}.{key}"] = _normalize_reactivation_value(value)
    oracle_status = payload.get("oracle_status") if isinstance(payload, dict) else None
    if isinstance(oracle_status, dict):
        for key in ("availability_state", "freshness_state", "directional_state", "source", "available"):
            if key in oracle_status:
                flags[f"payload.oracle_status.{key}"] = _normalize_reactivation_value(oracle_status.get(key))
    return flags


def _has_skipped_signal_material_change(
    row: TradeSignal,
    *,
    source_item_id: Optional[str],
    signal_type: str,
    strategy_type: Optional[str],
    market_id: str,
    market_question: Optional[str],
    direction: Optional[str],
    entry_price: Optional[float],
    edge_percent: Optional[float],
    confidence: Optional[float],
    liquidity: Optional[float],
    payload_json: Optional[dict[str, Any]],
    strategy_context_json: Optional[dict[str, Any]],
    expires_at: Optional[datetime],
    quality_passed: Optional[bool],
    quality_rejection_reasons: Optional[list],
) -> bool:
    if str(row.source_item_id or "") != str(source_item_id or ""):
        return True
    if str(row.signal_type or "") != str(signal_type or ""):
        return True
    if str(row.strategy_type or "") != str(strategy_type or ""):
        return True
    if str(row.market_id or "") != str(market_id or ""):
        return True
    if str(row.market_question or "") != str(market_question or ""):
        return True
    if str(row.direction or "") != str(direction or ""):
        return True

    if _delta_crosses_threshold(row.entry_price, entry_price, abs_threshold=_SKIPPED_REACTIVATION_PRICE_DELTA):
        return True
    if _delta_crosses_threshold(row.edge_percent, edge_percent, abs_threshold=_SKIPPED_REACTIVATION_EDGE_DELTA):
        return True
    if _delta_crosses_threshold(
        row.confidence,
        confidence,
        abs_threshold=_SKIPPED_REACTIVATION_CONFIDENCE_DELTA,
    ):
        return True
    if _delta_crosses_threshold(
        row.liquidity,
        liquidity,
        abs_threshold=_SKIPPED_REACTIVATION_LIQUIDITY_DELTA_ABS,
        rel_threshold=_SKIPPED_REACTIVATION_LIQUIDITY_DELTA_REL,
    ):
        return True
    if _liquidity_band(row.liquidity) != _liquidity_band(liquidity):
        return True

    existing_payload = _normalize_reactivation_value(_safe_json(row.payload_json))
    existing_context = _normalize_reactivation_value(_safe_json(row.strategy_context_json))
    incoming_payload = _normalize_reactivation_value(_safe_json(payload_json))
    incoming_context = _normalize_reactivation_value(_safe_json(strategy_context_json))
    if _extract_reactivation_flags(existing_payload, existing_context) != _extract_reactivation_flags(
        incoming_payload,
        incoming_context,
    ):
        return True

    existing_expires_at = _to_utc_naive(row.expires_at)
    incoming_expires_at = _to_utc_naive(expires_at)
    if existing_expires_at != incoming_expires_at:
        if existing_expires_at is None or incoming_expires_at is None:
            return True
        expiry_delta = abs((incoming_expires_at - existing_expires_at).total_seconds())
        if expiry_delta >= _SKIPPED_REACTIVATION_EXPIRY_DELTA_SECONDS:
            return True

    if quality_passed is not None:
        incoming_quality_passed = bool(quality_passed)
        if bool(row.quality_passed) != incoming_quality_passed:
            return True
        existing_reasons = _normalize_reason_list(row.quality_rejection_reasons)
        incoming_reasons = _normalize_reason_list(quality_rejection_reasons)
        if existing_reasons != incoming_reasons:
            return True
    return False


def _parse_price_timestamp(value: Any) -> Optional[datetime]:
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 10_000_000_000:
            ts /= 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        ts = float(text)
        if ts > 10_000_000_000:
            ts /= 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _payload_contains_stale_market_prices(payload: Any, now: datetime, max_age_seconds: float) -> bool:
    if not isinstance(payload, dict):
        return False
    markets = payload.get("markets")
    if not isinstance(markets, list) or not markets:
        return False
    for market in markets:
        if not isinstance(market, dict):
            continue
        token_ids = market.get("clob_token_ids")
        if isinstance(token_ids, str):
            text = token_ids.strip()
            if text.startswith("[") and text.endswith("]"):
                try:
                    token_ids = json.loads(text)
                except Exception:
                    token_ids = []
        if not isinstance(token_ids, list) and not isinstance(token_ids, tuple):
            continue
        if len(token_ids) < 2:
            continue
        ts = _parse_price_timestamp(market.get("price_updated_at"))
        if ts is None:
            return True
        age = (now - ts).total_seconds()
        if age > max_age_seconds:
            return True
    return False


def _signal_recently_refreshed(row: TradeSignal, now: datetime, max_age_seconds: float) -> bool:
    refreshed_at = row.updated_at or row.created_at
    if refreshed_at is None:
        return False
    if refreshed_at.tzinfo is None:
        refreshed_at = refreshed_at.replace(tzinfo=timezone.utc)
    else:
        refreshed_at = refreshed_at.astimezone(timezone.utc)
    return (now - refreshed_at).total_seconds() <= max_age_seconds


def _uses_runtime_price_revalidation(payload: Any, strategy_context: Any) -> bool:
    payload_json = payload if isinstance(payload, dict) else {}
    strategy_context_json = strategy_context if isinstance(strategy_context, dict) else {}
    runtime = payload_json.get("strategy_runtime")
    runtime_json = runtime if isinstance(runtime, dict) else {}
    source_key = str(
        runtime_json.get("source_key")
        or strategy_context_json.get("source_key")
        or payload_json.get("source_key")
        or ""
    ).strip().lower()
    activation = str(
        runtime_json.get("execution_activation")
        or strategy_context_json.get("execution_activation")
        or ""
    ).strip().lower()
    if source_key == "scanner":
        return True
    return activation in {"ws_current", "ws_post_arm_tick"}


def _strategy_runtime_metadata(opportunity: Opportunity) -> dict[str, Any]:
    strategy_slug = str(getattr(opportunity, "strategy", "") or "").strip().lower()
    if not strategy_slug:
        return {}
    try:
        from services.strategy_loader import strategy_loader
    except Exception:
        return {}
    loaded = strategy_loader.get_strategy(strategy_slug)
    if loaded is None:
        return {}
    instance = loaded.instance
    source_key = str(getattr(instance, "source_key", "") or "").strip().lower()
    subscriptions = sorted(
        {
            str(subscription or "").strip().lower()
            for subscription in (getattr(instance, "subscriptions", None) or [])
            if str(subscription or "").strip()
        }
    )
    if source_key == "crypto":
        execution_activation = "immediate"
    elif source_key == "scanner":
        execution_activation = "ws_current"
    elif source_key == "traders":
        # Copy-trade leader-wallet tokens are usually not on the CLOB feed,
        # so ws_post_arm_tick would defer the signal indefinitely.
        execution_activation = "immediate"
    else:
        execution_activation = "ws_post_arm_tick"
    return {
        "strategy_slug": strategy_slug,
        "source_key": source_key,
        "subscriptions": subscriptions,
        "execution_activation": execution_activation,
    }


def _execution_profile_for_opportunity(opportunity: Opportunity, legs_count: int) -> dict[str, Any]:
    strategy_key = str(getattr(opportunity, "strategy", "") or "").strip().lower()
    runtime_metadata = _strategy_runtime_metadata(opportunity)
    strategy_context = getattr(opportunity, "strategy_context", None)
    source_key = str(runtime_metadata.get("source_key") or "").strip().lower()
    if isinstance(strategy_context, dict):
        source_key = source_key or str(strategy_context.get("source_key") or "").strip().lower()
    is_traders = source_key == "traders" or strategy_key.startswith("traders")
    is_guaranteed_bundle = bool(getattr(opportunity, "is_guaranteed", False)) and legs_count > 1

    if is_traders:
        return {
            "policy": "REPRICE_LOOP" if legs_count <= 1 else "SEQUENTIAL_HEDGE",
            "price_policy": "taker_limit",
            "time_in_force": "IOC",
            "constraints": {
                "max_unhedged_notional_usd": 0.0,
                "hedge_timeout_seconds": 12,
                "session_timeout_seconds": 90,
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


def _merge_position_execution_contract(
    plan: dict[str, Any],
    positions: list[dict[str, Any]],
) -> dict[str, Any]:
    legs = [dict(leg) for leg in (plan.get("legs") or []) if isinstance(leg, dict)]
    if not legs or not positions:
        return dict(plan)
    for index, position in enumerate(positions):
        if index >= len(legs) or not isinstance(position, dict):
            continue
        leg = legs[index]
        for key in (
            "max_execution_price",
            "max_entry_price",
            "min_execution_price",
            "min_exit_price",
            "allow_taker_limit_buy_above_signal",
            "aggressive_limit_buy_submit_as_gtc",
            "price_policy",
            "time_in_force",
        ):
            if key in position:
                leg[key] = position.get(key)
        legs[index] = leg
    merged = dict(plan)
    merged["legs"] = legs
    return merged


def _normalize_execution_plan(opportunity: Opportunity) -> dict[str, Any] | None:
    positions = [position for position in (getattr(opportunity, "positions_to_take", None) or []) if isinstance(position, dict)]
    expected_leg_count = len(positions)
    existing = getattr(opportunity, "execution_plan", None)
    if existing is not None:
        if hasattr(existing, "model_dump"):
            dumped = existing.model_dump(mode="json")
            if isinstance(dumped, dict) and list(dumped.get("legs") or []):
                existing_legs = [leg for leg in (dumped.get("legs") or []) if isinstance(leg, dict)]
                if expected_leg_count <= 1 or len(existing_legs) >= expected_leg_count:
                    return _merge_position_execution_contract(dumped, positions)
        elif isinstance(existing, dict) and list(existing.get("legs") or []):
            existing_legs = [leg for leg in (existing.get("legs") or []) if isinstance(leg, dict)]
            if expected_leg_count <= 1 or len(existing_legs) >= expected_leg_count:
                return _merge_position_execution_contract(dict(existing), positions)

    if not positions:
        return None

    markets = list(getattr(opportunity, "markets", None) or [])
    single_market = (
        markets[0]
        if len(markets) == 1 and isinstance(markets[0], dict) and str(markets[0].get("id") or "").strip()
        else None
    )
    market_by_id: dict[str, dict[str, Any]] = {}
    for market in markets:
        if not isinstance(market, dict):
            continue
        market_id = str(market.get("id") or "").strip()
        if market_id:
            market_by_id[market_id] = market

    profile = _execution_profile_for_opportunity(opportunity, len(positions))
    legs: list[dict[str, Any]] = []
    for index, position in enumerate(positions):
        market_id = str(
            position.get("market_id")
            or position.get("id")
            or position.get("market")
            or (markets[index].get("id") if index < len(markets) and isinstance(markets[index], dict) else "")
            or (single_market.get("id") if isinstance(single_market, dict) else "")
            or ""
        ).strip()
        if not market_id:
            continue
        market = market_by_id.get(market_id) or (
            single_market
            if isinstance(single_market, dict) and str(single_market.get("id") or "").strip() == market_id
            else (markets[index] if index < len(markets) and isinstance(markets[index], dict) else {})
        )
        action = str(position.get("action") or position.get("side") or "").strip().lower()
        limit_price = position.get("price")
        try:
            parsed_price = float(limit_price)
            if parsed_price <= 0:
                parsed_price = None
        except Exception:
            parsed_price = None

        legs.append(
            {
                "leg_id": str(position.get("leg_id") or f"leg_{index + 1}"),
                "market_id": market_id,
                "market_question": str(
                    position.get("market_question")
                    or market.get("question")
                    or (single_market.get("question") if isinstance(single_market, dict) else "")
                    or opportunity.title
                ),
                "token_id": str(position.get("token_id") or "").strip() or None,
                "side": normalize_position_side(action),
                "outcome": str(position.get("outcome") or "").strip().lower() or None,
                "limit_price": parsed_price,
                "max_execution_price": position.get("max_execution_price"),
                "max_entry_price": position.get("max_entry_price"),
                "min_execution_price": position.get("min_execution_price"),
                "min_exit_price": position.get("min_exit_price"),
                "price_policy": str(position.get("price_policy") or profile["price_policy"]),
                "time_in_force": str(position.get("time_in_force") or profile["time_in_force"]),
                "allow_taker_limit_buy_above_signal": position.get("allow_taker_limit_buy_above_signal"),
                "aggressive_limit_buy_submit_as_gtc": position.get("aggressive_limit_buy_submit_as_gtc"),
                "notional_weight": max(0.0001, float(position.get("notional_weight") or 1.0)),
                "min_fill_ratio": max(
                    0.0,
                    min(1.0, float(position.get("min_fill_ratio") or 0.0)),
                ),
                "metadata": {
                    "position_index": index,
                },
            }
        )

    if not legs:
        return None
    profile = _execution_profile_for_opportunity(opportunity, len(legs))
    roster_payload = getattr(opportunity, "market_roster", None)
    roster_scope = str((roster_payload or {}).get("scope") or "").strip().lower() if isinstance(roster_payload, dict) else ""
    roster_market_ids: list[str] = []
    if isinstance(roster_payload, dict):
        seen_roster_market_ids: set[str] = set()
        for raw_market in list(roster_payload.get("markets") or []):
            if not isinstance(raw_market, dict):
                continue
            market_id = str(raw_market.get("id") or raw_market.get("market_id") or "").strip()
            if not market_id or market_id in seen_roster_market_ids:
                continue
            seen_roster_market_ids.add(market_id)
            roster_market_ids.append(market_id)
    planned_market_ids: list[str] = []
    seen_planned_market_ids: set[str] = set()
    for leg in legs:
        market_id = str(leg.get("market_id") or "").strip()
        if not market_id or market_id in seen_planned_market_ids:
            continue
        seen_planned_market_ids.add(market_id)
        planned_market_ids.append(market_id)
    roster_market_id_set = set(roster_market_ids)
    event_internal_bundle = (
        roster_scope == "event"
        and len(planned_market_ids) > 1
        and bool(planned_market_ids)
        and all(market_id in roster_market_id_set for market_id in planned_market_ids)
    )
    missing_market_ids = [market_id for market_id in roster_market_ids if market_id not in seen_planned_market_ids]

    packed = "|".join(
        sorted(
            f"{leg['market_id']}:{leg.get('token_id') or ''}:{leg.get('outcome') or ''}:{leg.get('side') or ''}"
            for leg in legs
        )
    )
    plan_hash = hashlib.sha256(packed.encode("utf-8")).hexdigest()[:16]

    return {
        "plan_id": f"plan_{plan_hash}",
        "policy": str(profile["policy"]),
        "time_in_force": str(profile["time_in_force"]),
        "legs": legs,
        "constraints": dict(profile["constraints"]),
        "metadata": {
            "strategy_type": str(getattr(opportunity, "strategy", "") or ""),
            "source_item_id": str(getattr(opportunity, "stable_id", "") or ""),
            "is_guaranteed": bool(getattr(opportunity, "is_guaranteed", False)),
            "market_coverage": {
                "scope": roster_scope or None,
                "market_roster_hash": (
                    str(roster_payload.get("roster_hash") or "").strip() or None
                    if isinstance(roster_payload, dict)
                    else None
                ),
                "roster_market_count": len(roster_market_ids),
                "planned_market_count": len(planned_market_ids),
                "planned_market_ids": planned_market_ids,
                "missing_market_ids": missing_market_ids,
                "event_internal_bundle": event_internal_bundle,
                "requires_full_market_coverage": bool(
                    getattr(opportunity, "is_guaranteed", False) and event_internal_bundle
                ),
            },
        },
    }


def _primary_plan_leg(plan: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(plan, dict):
        return None
    legs = [leg for leg in (plan.get("legs") or []) if isinstance(leg, dict)]
    if not legs:
        return None
    buy_legs = [leg for leg in legs if str(leg.get("side") or "").strip().lower() == "buy"]
    if buy_legs:
        return buy_legs[0]
    return legs[0]


def _opportunity_field(opportunity: Opportunity, key: str) -> Any:
    if isinstance(opportunity, dict):
        return opportunity.get(key)
    return getattr(opportunity, key, None)


def _market_field(market: Any, key: str) -> Any:
    if isinstance(market, dict):
        return market.get(key)
    return getattr(market, key, None)


def _safe_list_copy(value: Any) -> list[Any]:
    if isinstance(value, list):
        return copy.deepcopy(value)
    if isinstance(value, tuple):
        return copy.deepcopy(list(value))
    return []


def _normalize_signal_market_entry(market: Any) -> dict[str, Any] | None:
    market_id = str(_market_field(market, "id") or _market_field(market, "market_id") or "").strip()
    if not market_id:
        return None

    token_ids = _safe_list_copy(_market_field(market, "clob_token_ids"))
    outcome_prices = _safe_list_copy(_market_field(market, "outcome_prices"))
    outcome_labels = _safe_list_copy(_market_field(market, "outcome_labels"))
    tags = _market_field(market, "tags")
    if isinstance(tags, list):
        normalized_tags = [str(tag or "").strip() for tag in tags if str(tag or "").strip()]
    else:
        single_tag = str(tags or "").strip()
        normalized_tags = [single_tag] if single_tag else []

    entry = {
        "id": market_id,
        "condition_id": str(_market_field(market, "condition_id") or _market_field(market, "conditionId") or "").strip() or None,
        "question": str(_market_field(market, "question") or _market_field(market, "market_question") or "").strip() or None,
        "slug": str(_market_field(market, "slug") or _market_field(market, "market_slug") or "").strip() or None,
        "group_item_title": str(_market_field(market, "group_item_title") or "").strip() or None,
        "event_slug": str(_market_field(market, "event_slug") or "").strip() or None,
        "event_ticker": str(_market_field(market, "event_ticker") or "").strip() or None,
        "sports_market_type": str(_market_field(market, "sports_market_type") or "").strip() or None,
        "game_start_time": _market_field(market, "game_start_time"),
        "line": _market_field(market, "line"),
        "platform": str(_market_field(market, "platform") or "polymarket").strip().lower() or "polymarket",
        "neg_risk": bool(_market_field(market, "neg_risk")),
        "active": bool(_market_field(market, "active")) if _market_field(market, "active") is not None else None,
        "closed": bool(_market_field(market, "closed")) if _market_field(market, "closed") is not None else None,
        "accepting_orders": (
            bool(_market_field(market, "accepting_orders"))
            if _market_field(market, "accepting_orders") is not None
            else None
        ),
        "resolved": bool(_market_field(market, "resolved")) if _market_field(market, "resolved") is not None else None,
        "winner": bool(_market_field(market, "winner")) if _market_field(market, "winner") is not None else None,
        "winning_outcome": str(_market_field(market, "winning_outcome") or "").strip() or None,
        "status": str(_market_field(market, "status") or "").strip() or None,
        "clob_token_ids": [str(token_id or "").strip() for token_id in token_ids if str(token_id or "").strip()],
        "outcome_prices": outcome_prices,
        "outcome_labels": outcome_labels,
        "yes_price": _market_field(market, "yes_price"),
        "no_price": _market_field(market, "no_price"),
        "current_yes_price": _market_field(market, "current_yes_price"),
        "current_no_price": _market_field(market, "current_no_price"),
        "liquidity": _market_field(market, "liquidity"),
        "volume": _market_field(market, "volume"),
        "price_updated_at": _market_field(market, "price_updated_at"),
        "price_age_seconds": _market_field(market, "price_age_seconds"),
        "is_price_fresh": _market_field(market, "is_price_fresh"),
    }
    if normalized_tags:
        entry["tags"] = normalized_tags
    return entry


def _select_signal_markets(
    opportunity: Opportunity,
    *,
    primary_market_id: str,
    plan: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[Any], bool]:
    raw_markets = list(_opportunity_field(opportunity, "markets") or [])
    raw_roster = _opportunity_field(opportunity, "market_roster")
    roster_markets = list(raw_roster.get("markets") or []) if isinstance(raw_roster, dict) else []
    candidate_markets = raw_markets or roster_markets
    full_roster_markets = roster_markets or raw_markets

    planned_market_ids: list[str] = []
    seen_market_ids: set[str] = set()
    for leg in list((plan or {}).get("legs") or []):
        if not isinstance(leg, dict):
            continue
        market_id = str(leg.get("market_id") or "").strip()
        if not market_id or market_id in seen_market_ids:
            continue
        seen_market_ids.add(market_id)
        planned_market_ids.append(market_id)
    if not planned_market_ids and primary_market_id:
        planned_market_ids.append(primary_market_id)

    selected_raw_markets: list[Any] = []
    selected_market_ids = set(planned_market_ids)
    if selected_market_ids:
        for market in candidate_markets:
            market_id = str(_market_field(market, "id") or _market_field(market, "market_id") or "").strip()
            if market_id and market_id in selected_market_ids:
                selected_raw_markets.append(market)
    if not selected_raw_markets and candidate_markets:
        selected_raw_markets = [candidate_markets[0]]

    payload_markets: list[dict[str, Any]] = []
    seen_payload_market_ids: set[str] = set()
    for market in selected_raw_markets:
        normalized = _normalize_signal_market_entry(market)
        if normalized is None:
            continue
        market_id = str(normalized.get("id") or "").strip()
        if not market_id or market_id in seen_payload_market_ids:
            continue
        seen_payload_market_ids.add(market_id)
        payload_markets.append(normalized)

    keep_full_event_roster = bool(_opportunity_field(opportunity, "is_guaranteed")) and len(planned_market_ids) > 1
    roster_source_markets = full_roster_markets if keep_full_event_roster else selected_raw_markets
    return payload_markets, roster_source_markets, keep_full_event_roster


def build_signal_contract_from_opportunity(
    opportunity: Opportunity,
) -> tuple[str, str | None, float | None, str | None, dict[str, Any], dict[str, Any] | None]:
    plan = _normalize_execution_plan(opportunity)
    primary_leg = _primary_plan_leg(plan) or {}
    first_market = (opportunity.markets or [{}])[0]

    market_id = str(primary_leg.get("market_id") or first_market.get("id") or opportunity.event_id or opportunity.id)
    market_question = str(primary_leg.get("market_question") or first_market.get("question") or opportunity.title)
    direction = _direction_from_outcome(primary_leg.get("outcome"))
    entry_price = primary_leg.get("limit_price")
    if entry_price is None:
        first_position = (opportunity.positions_to_take or [{}])[0]
        entry_price = first_position.get("price")

    payload_markets, roster_source_markets, keep_full_event_roster = _select_signal_markets(
        opportunity,
        primary_market_id=market_id,
        plan=plan,
    )
    first_payload_market = payload_markets[0] if payload_markets else {}
    strategy_context = dict(opportunity.strategy_context or {})
    event_id = str(getattr(opportunity, "event_id", "") or "") or None
    event_slug = str(getattr(opportunity, "event_slug", "") or "") or None
    event_title = str(getattr(opportunity, "event_title", "") or "") or None
    category = str(getattr(opportunity, "category", "") or "") or None

    payload: dict[str, Any] = {
        "id": str(getattr(opportunity, "id", "") or ""),
        "stable_id": str(getattr(opportunity, "stable_id", "") or ""),
        "strategy": str(getattr(opportunity, "strategy", "") or ""),
        "title": str(getattr(opportunity, "title", "") or ""),
        "description": str(getattr(opportunity, "description", "") or ""),
        "market_id": market_id,
        "market_question": market_question,
        "market_slug": str(first_payload_market.get("slug") or "") or None,
        "event_id": event_id,
        "event_slug": event_slug,
        "event_title": event_title,
        "category": category,
        "detected_at": getattr(opportunity, "detected_at", None),
        "first_detected_at": getattr(opportunity, "first_detected_at", None),
        "last_detected_at": getattr(opportunity, "last_detected_at", None),
        "last_priced_at": getattr(opportunity, "last_priced_at", None),
        "last_seen_at": getattr(opportunity, "last_seen_at", None),
        "last_updated_at": getattr(opportunity, "last_updated_at", None),
        "resolution_date": getattr(opportunity, "resolution_date", None),
        "total_cost": getattr(opportunity, "total_cost", None),
        "expected_payout": getattr(opportunity, "expected_payout", None),
        "gross_profit": getattr(opportunity, "gross_profit", None),
        "fee": getattr(opportunity, "fee", None),
        "net_profit": getattr(opportunity, "net_profit", None),
        "guaranteed_profit": getattr(opportunity, "guaranteed_profit", None),
        "roi_percent": getattr(opportunity, "roi_percent", None),
        "roi_type": getattr(opportunity, "roi_type", None),
        "is_guaranteed": bool(getattr(opportunity, "is_guaranteed", False)),
        "mispricing_type": getattr(opportunity, "mispricing_type", None),
        "risk_score": getattr(opportunity, "risk_score", None),
        "risk_factors": copy.deepcopy(list(getattr(opportunity, "risk_factors", None) or [])),
        "confidence": getattr(opportunity, "confidence", None),
        "max_position_size": getattr(opportunity, "max_position_size", None),
        "min_liquidity": getattr(opportunity, "min_liquidity", None),
        "capture_ratio": getattr(opportunity, "capture_ratio", None),
        "revision": getattr(opportunity, "revision", None),
        "polymarket_url": getattr(opportunity, "polymarket_url", None),
        "kalshi_url": getattr(opportunity, "kalshi_url", None),
        "markets": payload_markets,
        "positions_to_take": copy.deepcopy(list(getattr(opportunity, "positions_to_take", None) or [])),
        "strategy_context": copy.deepcopy(strategy_context),
    }
    if plan is not None:
        payload["execution_plan"] = plan

    roster_scope = "event" if keep_full_event_roster and (event_id or event_slug) else "signal"
    roster = build_market_roster(
        list(roster_source_markets or []),
        scope=roster_scope,
        event_id=event_id,
        event_slug=event_slug,
        event_title=event_title,
        category=category,
    )
    if roster is None:
        payload = ensure_market_roster_payload(
            payload,
            markets=payload_markets,
            market_id=market_id,
            market_question=market_question,
            event_id=event_id,
            event_slug=event_slug,
            event_title=event_title,
            category=category,
        )
    else:
        payload["market_roster"] = roster
    if event_id or event_slug or event_title or category:
        payload["event"] = {
            "id": event_id,
            "slug": event_slug,
            "title": event_title,
            "category": category,
        }
    runtime_metadata = _strategy_runtime_metadata(opportunity)
    if runtime_metadata:
        payload["strategy_runtime"] = dict(runtime_metadata)
        strategy_context.setdefault("source_key", runtime_metadata.get("source_key"))
        strategy_context.setdefault("subscriptions", list(runtime_metadata.get("subscriptions") or []))
        strategy_context.setdefault("execution_activation", runtime_metadata.get("execution_activation"))
    if plan is not None:
        strategy_context["execution_plan"] = plan
    selected_token_id = str(
        primary_leg.get("token_id")
        or (payload.get("positions_to_take") or [{}])[0].get("token_id")
        or ""
    ).strip()
    if selected_token_id:
        payload["selected_token_id"] = selected_token_id
        payload["token_id"] = selected_token_id
    first_market_token_ids = list(first_payload_market.get("clob_token_ids") or [])
    if first_market_token_ids:
        payload["yes_token_id"] = str(first_market_token_ids[0] or "").strip() or None
    if len(first_market_token_ids) > 1:
        payload["no_token_id"] = str(first_market_token_ids[1] or "").strip() or None

    return (
        market_id,
        direction,
        entry_price,
        market_question,
        payload,
        strategy_context or None,
    )


async def _record_signal_emission(
    session: AsyncSession,
    row: TradeSignal,
    *,
    event_type: str,
    reason: str | None = None,
) -> None:
    session.add(
        TradeSignalEmission(
            id=uuid.uuid4().hex,
            signal_id=row.id,
            source=str(row.source or ""),
            source_item_id=row.source_item_id,
            signal_type=str(row.signal_type or ""),
            strategy_type=row.strategy_type,
            market_id=str(row.market_id or ""),
            direction=row.direction,
            entry_price=row.entry_price,
            effective_price=row.effective_price,
            edge_percent=row.edge_percent,
            confidence=row.confidence,
            liquidity=row.liquidity,
            status=str(row.status or ""),
            dedupe_key=str(row.dedupe_key or ""),
            event_type=str(event_type),
            reason=reason,
            payload_json=None,
            snapshot_json=None,
            created_at=_utc_now(),
        )
    )


# Redis channel names for cross-process signal arrival notifications.
# See ``services/wallet_state_bus.py`` for the analogous pattern.
SIGNAL_EMISSION_CHANNEL = "trade_signals:emission"
SIGNAL_BATCH_CHANNEL = "trade_signals:batch"
# Channel for full signal payloads (vs the lightweight notifications
# above).  Subscribed by ``services/signal_cache.py`` in the trading
# plane to populate an in-memory cache, eliminating the fast trader's
# per-cycle ``list_unconsumed_trade_signals`` DB query.
SIGNAL_PAYLOADS_CHANNEL = "signal_payloads"


async def _publish_redis(channel: str, payload: dict[str, Any]) -> None:
    """Best-effort Redis publish for cross-plane signal arrival.

    Emissions are published in-process via ``event_bus`` (which wakes the
    fast_trader_runtime + main orchestrator running in the trading
    plane).  But when a signal is emitted from a *different* process —
    e.g. the news plane writes a signal via the strategy bridge — the
    trading plane's in-process event_bus never sees it, so without this
    Redis hop the trading plane has to wait for the orchestrator's 60s
    polling fallback.  This publish closes that gap to <1ms.
    """
    try:
        from services import redis_client

        client = redis_client.get_client_or_none()
        if client is None:
            return
        import json as _json

        await client.publish(
            redis_client.namespaced(channel),
            _json.dumps(payload, default=str),
        )
    except Exception:
        # Soft-fail: in-process event_bus already woke local subscribers.
        # Cross-plane is a nice-to-have but not load-bearing.
        pass


def _serialize_signal_payload(row: TradeSignal) -> dict[str, Any]:
    """Build the full-fidelity signal snapshot for Redis transmission.

    Mirrors the columns ``services/signal_cache.SignalSnapshot`` reads
    on the consumer side.  Heavy JSON columns are EXCLUDED — fast
    trader doesn't read them and shipping ~3 KB of unused JSON over
    the bus on every signal would waste bandwidth.
    """
    return {
        "id": str(row.id or ""),
        "source": str(row.source or ""),
        "source_item_id": row.source_item_id,
        "signal_type": str(row.signal_type or ""),
        "strategy_type": row.strategy_type,
        "market_id": str(row.market_id or ""),
        "market_question": row.market_question,
        "direction": row.direction,
        "entry_price": float(row.entry_price) if row.entry_price is not None else None,
        "effective_price": float(row.effective_price) if row.effective_price is not None else None,
        "edge_percent": float(row.edge_percent) if row.edge_percent is not None else None,
        "confidence": float(row.confidence) if row.confidence is not None else None,
        "liquidity": float(row.liquidity) if row.liquidity is not None else None,
        "expires_at": row.expires_at.isoformat() if row.expires_at else None,
        "status": str(row.status or "pending"),
        "quality_passed": row.quality_passed,
        "dedupe_key": str(row.dedupe_key or ""),
        "runtime_sequence": int(row.runtime_sequence) if row.runtime_sequence is not None else None,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


async def _publish_trade_signal_emission(
    *,
    row: TradeSignal,
    event_type: str,
    reason: str | None = None,
) -> None:
    payload = {
        "signal_id": str(row.id or ""),
        "source": str(row.source or ""),
        "status": str(row.status or ""),
        "event_type": str(event_type),
        "reason": reason,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }
    # Record this signal_id as locally-emitted BEFORE publishing to Redis
    # so the trading plane's bridge subscriber correctly dedups when the
    # Redis broadcast loops back.
    try:
        from services import signal_bus_redis_bridge

        await signal_bus_redis_bridge.mark_locally_emitted(payload["signal_id"])
    except Exception:
        pass
    try:
        await event_bus.publish("trade_signal_emission", payload)
    except Exception:
        pass
    # Cross-plane wakeup (lightweight notification).
    await _publish_redis(SIGNAL_EMISSION_CHANNEL, payload)
    # Full payload for the trading-plane signal cache.  This is what
    # eliminates the fast trader's per-cycle DB poll: the cache hydrates
    # from this channel and serves reads in microseconds.  Soft-fail at
    # the publish layer; the cache also has a DB-fallback safety net.
    try:
        full_snapshot = _serialize_signal_payload(row)
        await _publish_redis(SIGNAL_PAYLOADS_CHANNEL, full_snapshot)
    except Exception:
        pass


async def _publish_trade_signal_batch(
    *,
    event_type: str,
    signal_ids: list[str],
    source: str | None = None,
    reason: str | None = None,
) -> None:
    if not signal_ids:
        return
    normalized_event_type = str(event_type or "").strip().lower()
    normalized_source = str(source or "").strip().lower()
    emitted_at_iso = _utc_now().isoformat()
    payload = {
        "event_type": normalized_event_type,
        "source": normalized_source,
        "reason": reason,
        "signal_count": int(len(signal_ids)),
        "signal_ids": signal_ids[:500],
        "emitted_at": emitted_at_iso,
        "trigger": "signal_bus",
    }
    # Pre-mark these IDs so the cross-plane bridge dedups correctly.
    try:
        from services import signal_bus_redis_bridge

        await signal_bus_redis_bridge.mark_locally_emitted(payload["signal_ids"])
    except Exception:
        pass
    try:
        await event_bus.publish("trade_signal_batch", payload)
    except Exception:
        pass
    # Cross-plane wakeup.
    await _publish_redis(SIGNAL_BATCH_CHANNEL, payload)


def make_dedupe_key(*parts: Any) -> str:
    packed = "|".join(str(p or "") for p in parts)
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()[:32]


async def upsert_trade_signal(
    session: AsyncSession,
    *,
    source: str,
    source_item_id: Optional[str],
    signal_type: str,
    strategy_type: Optional[str],
    market_id: str,
    market_question: Optional[str],
    direction: Optional[str],
    entry_price: Optional[float],
    edge_percent: Optional[float],
    confidence: Optional[float],
    liquidity: Optional[float],
    expires_at: Optional[datetime],
    payload_json: Optional[dict[str, Any]],
    strategy_context_json: Optional[dict[str, Any]] = None,
    quality_passed: Optional[bool] = None,
    quality_rejection_reasons: Optional[list] = None,
    dedupe_key: str,
    signal_id: Optional[str] = None,
    runtime_sequence: Any = _RUNTIME_SEQUENCE_UNSET,
    commit: bool = True,
) -> TradeSignal:
    """Idempotently upsert a normalized trade signal by ``(source, dedupe_key)``.

    Architectural note: every signal published here must have a live
    user-channel WS subscription in place BEFORE the orchestrator can
    pick it up — otherwise the orchestrator's freshness gate will
    refuse to trade.  We ensure that subscription synchronously at
    publish time below.  ``ensure_user_subscribed()`` is idempotent
    and sub-millisecond when the condition is already in the set.
    """
    # Strategy-side subscription handoff.  The market_id passed here
    # is the Polymarket condition_id (hex address).  Adding it to the
    # user-channel WS subscription set guarantees that by the time the
    # orchestrator consumes this signal, the wallet's order/fill
    # events for this market will flow into ``WalletStateCache``.
    # Failures are logged but never blocking — the freshness gate
    # provides the safety net downstream.
    if market_id and isinstance(market_id, str) and market_id.startswith("0x"):
        try:
            from services.ws_feeds import get_feed_manager

            await get_feed_manager().ensure_user_subscribed([market_id])
        except Exception as _sub_exc:
            logger.debug(
                "User-channel WS subscribe at signal publish failed (non-fatal)",
                extra={"market_id": market_id, "error": str(_sub_exc)},
            )

    normalized_payload_json = ensure_market_roster_payload(
        payload_json,
        market_id=market_id,
        market_question=market_question,
    )
    if isinstance(normalized_payload_json, dict):
        normalized_payload_json = dict(normalized_payload_json)
        normalized_payload_json["signal_emitted_at"] = _utc_now().isoformat()
    row: Optional[TradeSignal] = None
    emission_event_type = "upsert_insert"
    emission_reason: str | None = None
    publish_signal_emission = True

    # Prefer pending in-session rows first so we can avoid query-invoked
    # autoflush while still preserving same-transaction dedupe behavior.
    for pending in session.new:
        if not isinstance(pending, TradeSignal):
            continue
        if pending.source == source and pending.dedupe_key == dedupe_key:
            row = pending
            break

    if row is None:
        with session.no_autoflush:
            result = await session.execute(
                select(TradeSignal).where(
                    TradeSignal.source == source,
                    TradeSignal.dedupe_key == dedupe_key,
                )
            )
            row = result.scalar_one_or_none()

    # Only used for the UPDATE path below — never drop fresh inserts, because
    # the scanner may emit slightly-after-TTL signals under pipeline load and
    # dropping those silently starves traders of any signals at all.  The
    # decision-path staleness gate can still reject them, but they must at
    # least get emitted so the trader sees a trace.
    incoming_expires_naive = _to_utc_naive(expires_at) if expires_at is not None else None
    now_naive_guard = _to_utc_naive(_utc_now())
    incoming_already_expired = (
        incoming_expires_naive is not None and incoming_expires_naive <= now_naive_guard
    )

    if row is None:
        row = TradeSignal(
            id=str(signal_id or uuid.uuid4().hex),
            source=source,
            source_item_id=source_item_id,
            signal_type=signal_type,
            strategy_type=strategy_type,
            market_id=market_id,
            market_question=market_question,
            direction=direction,
            entry_price=entry_price,
            edge_percent=edge_percent,
            confidence=confidence,
            liquidity=liquidity,
            expires_at=incoming_expires_naive,
            payload_json=_safe_json(normalized_payload_json),
            strategy_context_json=_safe_json(strategy_context_json),
            quality_passed=quality_passed,
            quality_rejection_reasons=quality_rejection_reasons if quality_rejection_reasons else None,
            dedupe_key=dedupe_key,
            runtime_sequence=(
                int(runtime_sequence)
                if runtime_sequence is not _RUNTIME_SEQUENCE_UNSET and runtime_sequence is not None
                else None
            ),
            status="pending",
            created_at=_utc_now(),
            updated_at=_utc_now(),
        )
        session.add(row)
        await _record_signal_emission(
            session,
            row,
            event_type="upsert_insert",
        )
    else:
        # Cross-process row-lock contention guard.  When the caller
        # supplies a ``runtime_sequence`` AND the existing row already
        # has an equal-or-greater sequence, this incoming write is
        # stale relative to a write another worker has already committed.
        # Skip the entire UPDATE path so we don't compete for the row
        # lock — the work is duplicate and the lock wait stretches the
        # holder's transaction, cascading into ``LOCK CONTENTION ...
        # UPDATE trade_signals SET edge_percent, expires_at,
        # payload_json, strategy_context_json, runtime_sequence,
        # updated_at`` events and growing the projection_queue.
        #
        # The intent_runtime projection loop already does an in-memory
        # sequence gate at projection-pop time
        # (``intent_runtime.py:2662``), but that snapshot was captured
        # before this transaction started; by the time we get here
        # another process may have committed a fresher state.  This
        # DB-level gate catches the cross-process race the in-memory
        # gate cannot see.
        #
        # ``runtime_sequence`` callers that don't pass a sequence (the
        # ``_RUNTIME_SEQUENCE_UNSET`` sentinel) are unaffected — they
        # fall through to the existing logic.  Legacy rows with a NULL
        # sequence also fall through, since we have no basis for
        # comparison.
        incoming_seq: int | None = None
        if runtime_sequence is not _RUNTIME_SEQUENCE_UNSET and runtime_sequence is not None:
            try:
                incoming_seq = int(runtime_sequence)
            except (TypeError, ValueError):
                incoming_seq = None
        existing_seq = getattr(row, "runtime_sequence", None)
        if (
            incoming_seq is not None
            and existing_seq is not None
            and incoming_seq <= existing_seq
        ):
            return row

        previous_status = str(row.status or "").strip().lower()
        can_update_row = previous_status in SIGNAL_ACTIVE_STATUSES or previous_status in SIGNAL_REACTIVATABLE_STATUSES
        if can_update_row:
            has_material_change = True
            skipped_cooldown_reactivation = False
            if previous_status in SIGNAL_ACTIVE_STATUSES:
                has_material_change = _has_signal_material_change(
                    row,
                    source_item_id=source_item_id,
                    signal_type=signal_type,
                    strategy_type=strategy_type,
                    market_id=market_id,
                    market_question=market_question,
                    direction=direction,
                    entry_price=entry_price,
                    edge_percent=edge_percent,
                    confidence=confidence,
                    liquidity=liquidity,
                    payload_json=normalized_payload_json,
                    strategy_context_json=strategy_context_json,
                    quality_passed=quality_passed,
                    quality_rejection_reasons=quality_rejection_reasons,
                )
            elif previous_status == "skipped":
                has_material_change = _has_skipped_signal_material_change(
                    row,
                    source_item_id=source_item_id,
                    signal_type=signal_type,
                    strategy_type=strategy_type,
                    market_id=market_id,
                    market_question=market_question,
                    direction=direction,
                    entry_price=entry_price,
                    edge_percent=edge_percent,
                    confidence=confidence,
                    liquidity=liquidity,
                    payload_json=normalized_payload_json,
                    strategy_context_json=strategy_context_json,
                    expires_at=expires_at,
                    quality_passed=quality_passed,
                    quality_rejection_reasons=quality_rejection_reasons,
                )
                if not has_material_change:
                    skipped_cooldown_reactivation = _should_reactivate_unchanged_skipped_scanner_signal(
                        row,
                        source=source,
                    )
            # Respect the existing row's expires_at: if the decision-path has
            # already forced it to the past (staleness gate, expired_on_block,
            # etc.), do not let the scanner's re-emit resurrect it with a new
            # far-future TTL.  Without this check, PSA was caught in an
            # infinite loop: decision rejects signal for staleness, sets
            # expires_at=now, scanner re-emits 30s later with a fresh TTL,
            # trader rejects again for staleness (age from created_at is 30h),
            # and so on — 106000s+ old signals re-decided every cycle.
            existing_expires_naive = (
                _to_utc_naive(row.expires_at) if row.expires_at is not None else None
            )
            existing_already_expired = (
                existing_expires_naive is not None
                and existing_expires_naive <= now_naive_guard
            )
            if previous_status in SIGNAL_ACTIVE_STATUSES and not has_material_change:
                emission_event_type = "upsert_active_unchanged"
                emission_reason = "suppressed:active_unchanged"
                publish_signal_emission = False
                if str(source or "").strip().lower() in _ACTIVE_REFRESH_SOURCES:
                    # Cross-process row-lock contention guard: only assign
                    # when the incoming value actually differs from the
                    # existing one.  SQLAlchemy marks the row dirty on
                    # EVERY attribute set, so re-assigning equal values
                    # still emits an UPDATE — which competes for the row
                    # lock with concurrent writers in other workers
                    # (DISCOVERY/NEWS/WORKERS) doing the same active-
                    # refresh.  Soak (post-runtime_sequence-gate) showed
                    # this as ``LOCK CONTENTION ... UPDATE trade_signals
                    # SET expires_at, payload_json, updated_at WHERE
                    # trade_signals.id = $4`` — a 3-field UPDATE with no
                    # ``runtime_sequence`` because that gate already
                    # short-circuited the runtime_sequence-bearing path,
                    # leaving these volatile-payload refreshes as the
                    # last contention source.  Skipping equal-value
                    # assignments removes the row from the dirty set
                    # entirely; no UPDATE is emitted, no row lock taken.
                    #
                    # 2026-05-09: tighten the JSON comparisons so a
                    # fresh ``signal_emitted_at`` timestamp doesn't
                    # itself defeat the skip. ``_has_signal_material_change``
                    # already runs ``_normalize_reactivation_value`` over
                    # both sides (strips ``signal_emitted_at``,
                    # ``runtime_sequence``, ``scan_completed_at``,
                    # ``updated_at`` etc.) before declaring no material
                    # change. We were re-comparing the RAW JSON here,
                    # which always reported "differs" because every call
                    # stamps a new ``signal_emitted_at`` at line ~1296.
                    # Result: every active-unchanged refresh dirtied the
                    # row and emitted an UPDATE — exactly the contention
                    # the surrounding block was meant to eliminate.
                    # Match the comparison: if the normalized forms are
                    # equal, the stored RAW with stale ``signal_emitted_at``
                    # is functionally identical and we keep it untouched.
                    # (The volatile keys are not load-bearing for any
                    # downstream consumer — they're audit timestamps,
                    # superseded by trade_signal_emissions rows.)
                    new_payload = _safe_json(normalized_payload_json)
                    new_strategy_ctx = _safe_json(strategy_context_json)
                    if existing_expires_naive != incoming_expires_naive:
                        row.expires_at = incoming_expires_naive
                    existing_payload_for_cmp = _strip_auto_emitted_keys(
                        ensure_market_roster_payload(
                            _safe_json(row.payload_json),
                            market_id=market_id,
                            market_question=market_question,
                        )
                    )
                    incoming_payload_for_cmp = _strip_auto_emitted_keys(new_payload)
                    if existing_payload_for_cmp != incoming_payload_for_cmp:
                        row.payload_json = new_payload
                    existing_ctx_normalized = _normalize_reactivation_value(
                        _safe_json(row.strategy_context_json)
                    )
                    incoming_ctx_normalized = _normalize_reactivation_value(new_strategy_ctx)
                    if existing_ctx_normalized != incoming_ctx_normalized:
                        row.strategy_context_json = new_strategy_ctx
                    if quality_passed is not None:
                        new_rejection_reasons = (
                            quality_rejection_reasons if quality_rejection_reasons else None
                        )
                        if row.quality_passed != quality_passed:
                            row.quality_passed = quality_passed
                        if row.quality_rejection_reasons != new_rejection_reasons:
                            row.quality_rejection_reasons = new_rejection_reasons
            elif incoming_already_expired and previous_status in SIGNAL_ACTIVE_STATUSES:
                # Scanner sent a fresh emit with an already-past expires_at.
                # Mark the existing row expired; do not reactivate or emit.
                row.status = "expired"
                row.expires_at = incoming_expires_naive
                row.updated_at = _utc_now()
                emission_event_type = "upsert_expired_on_reemit"
                emission_reason = "incoming_already_expired"
                publish_signal_emission = False
                await _record_signal_emission(
                    session,
                    row,
                    event_type=emission_event_type,
                    reason=emission_reason,
                )
            elif existing_already_expired and previous_status in ("skipped", "pending"):
                # Existing row's expires_at was previously forced into the past
                # (by the decision-gate staleness path or similar).  Keep it
                # expired; do not let a fresh scanner emit override it.
                row.status = "expired"
                row.updated_at = _utc_now()
                emission_event_type = "upsert_existing_expires_at_past"
                emission_reason = "existing_already_expired"
                publish_signal_emission = False
                await _record_signal_emission(
                    session,
                    row,
                    event_type=emission_event_type,
                    reason=emission_reason,
                )
            else:
                should_reactivate = previous_status in SIGNAL_REACTIVATABLE_STATUSES
                if previous_status == "skipped":
                    should_reactivate = has_material_change or skipped_cooldown_reactivation
                if previous_status == "skipped" and not should_reactivate:
                    emission_event_type = "upsert_skipped_unchanged"
                    emission_reason = "reactivation_suppressed:skipped_unchanged"
                    publish_signal_emission = False
                else:
                    emission_event_type = "upsert_update"
                    emission_reason = None
                    if should_reactivate:
                        emission_event_type = "upsert_reactivated"
                        emission_reason = f"reactivated_from:{previous_status}"

                    # Compute the values dict once so both code paths
                    # (SQL-level conditional UPDATE for the cross-process
                    # race, ORM mutate for callers without a sequence)
                    # write the same shape.
                    now_utc = _utc_now()
                    new_values: dict[str, Any] = {
                        "source_item_id": source_item_id,
                        "signal_type": signal_type,
                        "strategy_type": strategy_type,
                        "market_id": market_id,
                        "market_question": market_question,
                        "direction": direction,
                        "entry_price": entry_price,
                        "edge_percent": edge_percent,
                        "confidence": confidence,
                        "liquidity": liquidity,
                        "expires_at": _to_utc_naive(expires_at),
                        "payload_json": _safe_json(normalized_payload_json),
                        "strategy_context_json": _safe_json(strategy_context_json),
                        "updated_at": now_utc,
                    }
                    if quality_passed is not None:
                        new_values["quality_passed"] = quality_passed
                        new_values["quality_rejection_reasons"] = (
                            quality_rejection_reasons if quality_rejection_reasons else None
                        )
                    if should_reactivate:
                        new_values["status"] = "pending"
                        new_values["effective_price"] = None

                    incoming_seq_int_for_update: int | None = None
                    if (
                        runtime_sequence is not _RUNTIME_SEQUENCE_UNSET
                        and runtime_sequence is not None
                    ):
                        try:
                            incoming_seq_int_for_update = int(runtime_sequence)
                        except (TypeError, ValueError):
                            incoming_seq_int_for_update = None

                    if incoming_seq_int_for_update is not None:
                        # SQL-level optimistic concurrency.  The Python
                        # gate above already short-circuited the cases
                        # where ``incoming <= existing`` at SELECT time,
                        # but two writers from different worker processes
                        # can both have ``incoming > existing`` when they
                        # SELECTed the same snapshot — both then proceed
                        # to UPDATE and serialize on the row lock for
                        # multi-second waits, growing the projection
                        # queue.  This conditional UPDATE makes Postgres
                        # the arbiter: only one writer's row passes the
                        # ``runtime_sequence < incoming`` predicate, the
                        # loser's UPDATE returns 0 rows and we bail
                        # without emitting an audit row or holding the
                        # lock for the SET clause.
                        #
                        # ``runtime_sequence IS NULL`` covers legacy
                        # rows that pre-date sequence tracking — they
                        # accept any incoming write.
                        new_values["runtime_sequence"] = incoming_seq_int_for_update
                        with session.no_autoflush:
                            update_result = await session.execute(
                                sa_update(TradeSignal)
                                .where(TradeSignal.id == row.id)
                                .where(
                                    or_(
                                        TradeSignal.runtime_sequence < incoming_seq_int_for_update,
                                        TradeSignal.runtime_sequence.is_(None),
                                    )
                                )
                                .values(**new_values)
                            )
                        if update_result.rowcount == 0:
                            # Lost the race.  Another writer committed a
                            # >= sequence between our SELECT and our
                            # UPDATE.  No audit emission; no further
                            # mutation.  The Python ``row`` object still
                            # holds SELECT-time attribute values, but
                            # the caller's contract is that the row
                            # exists in DB at >= our sequence — the
                            # winning writer's data is now authoritative.
                            return row
                        # Won.  Refresh the in-memory row so callers
                        # that inspect attributes after this function
                        # see the values we just wrote.
                        await session.refresh(row)
                    else:
                        # Caller did not supply a runtime_sequence (or
                        # supplied None to clear it).  Fall back to the
                        # original ORM mutate-and-commit pattern; no
                        # SQL-level race gate applies here.  This path
                        # is a small minority of upsert calls — the
                        # contention sources (intent_runtime projection)
                        # always pass an integer sequence.
                        row.source_item_id = source_item_id
                        row.signal_type = signal_type
                        row.strategy_type = strategy_type
                        row.market_id = market_id
                        row.market_question = market_question
                        row.direction = direction
                        row.entry_price = entry_price
                        row.edge_percent = edge_percent
                        row.confidence = confidence
                        row.liquidity = liquidity
                        row.expires_at = _to_utc_naive(expires_at)
                        row.payload_json = _safe_json(normalized_payload_json)
                        row.strategy_context_json = _safe_json(strategy_context_json)
                        if quality_passed is not None:
                            row.quality_passed = quality_passed
                            row.quality_rejection_reasons = (
                                quality_rejection_reasons if quality_rejection_reasons else None
                            )
                        if should_reactivate:
                            row.status = "pending"
                            row.effective_price = None
                        if runtime_sequence is not _RUNTIME_SEQUENCE_UNSET:
                            row.runtime_sequence = (
                                int(runtime_sequence) if runtime_sequence is not None else None
                            )
                        row.updated_at = now_utc

                    await _record_signal_emission(
                        session,
                        row,
                        event_type=emission_event_type,
                        reason=emission_reason,
                    )
        else:
            emission_event_type = "upsert_ignored_nonreactivable"
            emission_reason = f"nonreactivable_status:{previous_status}"
            publish_signal_emission = False

    if row is not None and runtime_sequence is not _RUNTIME_SEQUENCE_UNSET:
        row.runtime_sequence = int(runtime_sequence) if runtime_sequence is not None else None

    if commit:
        await session.commit()
        if publish_signal_emission:
            await _publish_trade_signal_emission(
                row=row,
                event_type=emission_event_type,
                reason=emission_reason,
            )
    return row


async def set_trade_signal_status(
    session: AsyncSession,
    signal_id: str,
    status: str,
    *,
    effective_price: Optional[float] = None,
    commit: bool = True,
) -> bool:
    result = await session.execute(select(TradeSignal).where(TradeSignal.id == signal_id))
    row = result.scalar_one_or_none()
    if row is None:
        return False
    normalized_status = str(status or "").strip().lower()
    previous_status = str(row.status or "").strip().lower()
    existing_effective_price = _normalize_number(row.effective_price)
    incoming_effective_price = _normalize_number(effective_price) if effective_price is not None else None
    if previous_status == normalized_status and (
        effective_price is None or existing_effective_price == incoming_effective_price
    ):
        return True
    row.status = normalized_status
    row.updated_at = _utc_now()
    if effective_price is not None:
        row.effective_price = effective_price
    await _record_signal_emission(
        session,
        row,
        event_type="status_update",
        reason=f"status:{normalized_status}",
    )
    if commit:
        await session.commit()
        await _publish_trade_signal_emission(
            row=row,
            event_type="status_update",
            reason=f"status:{normalized_status}",
        )
    return True


async def expire_stale_signals(session: AsyncSession, *, commit: bool = True) -> int:
    now = _utc_now()
    now_naive = _to_utc_naive(now)
    max_price_age_seconds = float(getattr(settings, "SCANNER_MARKET_PRICE_MAX_AGE_SECONDS", 0) or 0)
    if max_price_age_seconds <= 0:
        max_price_age_seconds = max(30.0, float(getattr(settings, "WS_PRICE_STALE_SECONDS", 30.0) or 30.0) * 2.0)

    # Use FOR UPDATE SKIP LOCKED so we only process rows that are not
    # currently locked by in-flight trader cycles.  This eliminates the
    # 20+ consecutive LockNotAvailableError failures that occur when
    # trader cycles hold row locks during signal processing.  Skipped
    # rows are simply caught on the next expiry pass.
    result = await session.execute(
        select(TradeSignal)
        .where(TradeSignal.status.in_(tuple(SIGNAL_ACTIVE_STATUSES)))
        .with_for_update(skip_locked=True)
    )
    rows = list(result.scalars().all())
    expired_signal_ids: list[str] = []
    for row in rows:
        expire_reason = None
        expires_at = row.expires_at
        if expires_at is not None and expires_at.tzinfo is not None:
            expires_at = expires_at.astimezone(timezone.utc).replace(tzinfo=None)
        if expires_at is not None and expires_at < now_naive:
            expire_reason = "expires_at_passed"
        elif (
            str(row.source or "").strip().lower() == "scanner"
            and not _uses_runtime_price_revalidation(row.payload_json, row.strategy_context_json)
            and _payload_contains_stale_market_prices(row.payload_json, now, max_price_age_seconds)
            and not _signal_recently_refreshed(row, now, max_price_age_seconds)
        ):
            expire_reason = "market_price_stale"
        if expire_reason is None:
            continue
        row.status = "expired"
        row.updated_at = now_naive
        expired_signal_ids.append(str(row.id))
        await _record_signal_emission(
            session,
            row,
            event_type="status_expired",
            reason=expire_reason,
        )
    if rows and commit:
        await session.commit()
        await _publish_trade_signal_batch(
            event_type="status_expired",
            signal_ids=expired_signal_ids,
            reason="expire_stale_signals",
        )
    elif commit:
        await session.commit()
    return len(expired_signal_ids)


async def expire_source_signals_except(
    session: AsyncSession,
    *,
    source: str,
    keep_dedupe_keys: set[str],
    signal_types: Optional[list[str]] = None,
    strategy_types: Optional[list[str]] = None,
    commit: bool = True,
    min_signal_age_seconds: Optional[float] = None,
) -> int:
    """Expire pending source signals not present in the current emission set.

    Rows younger than ``min_signal_age_seconds`` (default 60 s via
    ``_EXPIRE_SOURCE_GRACE_SECONDS``) are skipped to protect the
    writer/projection race for event-driven sources (crypto, traders,
    weather). Pass ``0.0`` to opt out of the grace window.
    """
    grace_seconds = (
        float(_EXPIRE_SOURCE_GRACE_SECONDS)
        if min_signal_age_seconds is None
        else max(0.0, float(min_signal_age_seconds))
    )
    now = _to_utc_naive(_utc_now())
    keep = {str(key) for key in keep_dedupe_keys if str(key).strip()}

    query = select(TradeSignal).where(
        TradeSignal.source == str(source),
        TradeSignal.status == "pending",
    )
    if grace_seconds > 0.0:
        cutoff = now - timedelta(seconds=grace_seconds)
        query = query.where(TradeSignal.created_at <= cutoff)
    if signal_types:
        normalized_signal_types = [
            str(value or "").strip().lower() for value in signal_types if str(value or "").strip()
        ]
        if normalized_signal_types:
            query = query.where(func.lower(func.coalesce(TradeSignal.signal_type, "")).in_(normalized_signal_types))
    if strategy_types:
        normalized_strategy_types = [
            str(value or "").strip().lower() for value in strategy_types if str(value or "").strip()
        ]
        if normalized_strategy_types:
            query = query.where(func.lower(func.coalesce(TradeSignal.strategy_type, "")).in_(normalized_strategy_types))
    if keep:
        query = query.where(~TradeSignal.dedupe_key.in_(list(keep)))
    query = query.with_for_update(skip_locked=True)

    rows = list((await session.execute(query)).scalars().all())
    expired_signal_ids: list[str] = []
    for row in rows:
        row.status = "expired"
        row.expires_at = now
        row.updated_at = now
        expired_signal_ids.append(str(row.id))
        await _record_signal_emission(
            session,
            row,
            event_type="status_expired",
            reason="source_sweep",
        )

    if rows and commit:
        await session.commit()
        await _publish_trade_signal_batch(
            event_type="status_expired",
            signal_ids=expired_signal_ids,
            source=str(source),
            reason="source_sweep",
        )
    elif commit:
        await session.commit()

    return len(rows)


async def list_pending_source_dedupe_keys(
    session: AsyncSession,
    *,
    source: str,
    signal_types: Optional[list[str]] = None,
    strategy_types: Optional[list[str]] = None,
) -> set[str]:
    query = select(TradeSignal.dedupe_key).where(
        TradeSignal.source == str(source),
        TradeSignal.status == "pending",
    )
    if signal_types:
        normalized_signal_types = [
            str(value or "").strip().lower() for value in signal_types if str(value or "").strip()
        ]
        if normalized_signal_types:
            query = query.where(func.lower(func.coalesce(TradeSignal.signal_type, "")).in_(normalized_signal_types))
    if strategy_types:
        normalized_strategy_types = [
            str(value or "").strip().lower() for value in strategy_types if str(value or "").strip()
        ]
        if normalized_strategy_types:
            query = query.where(func.lower(func.coalesce(TradeSignal.strategy_type, "")).in_(normalized_strategy_types))

    rows = await session.execute(query)
    return {
        str(value).strip()
        for value in rows.scalars().all()
        if str(value).strip()
    }


async def list_trade_signals(
    session: AsyncSession,
    *,
    source: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
) -> list[TradeSignal]:
    query = select(TradeSignal).order_by(TradeSignal.created_at.desc())
    if source:
        query = query.where(TradeSignal.source == source)
    if status:
        query = query.where(TradeSignal.status == status)
    query = query.offset(max(0, offset)).limit(max(1, min(limit, 1000)))
    result = await session.execute(query)
    rows = list(result.scalars().all())
    if status in {None, "pending", "selected", "submitted"}:
        rows = await _expire_non_tradable_pending_signals(session, rows)
    return rows


async def list_pending_trade_signals(
    session: AsyncSession,
    *,
    sources: Optional[list[str]] = None,
    limit: int = 500,
) -> list[TradeSignal]:
    now = _to_utc_naive(_utc_now())
    query = (
        select(TradeSignal)
        .where(TradeSignal.status == "pending")
        .where(or_(TradeSignal.expires_at.is_(None), TradeSignal.expires_at >= now))
        .order_by(TradeSignal.created_at.asc())
        .limit(max(1, min(limit, 5000)))
    )
    if sources:
        query = query.where(TradeSignal.source.in_(sources))
    result = await session.execute(query)
    rows = list(result.scalars().all())
    rows = await _expire_non_tradable_pending_signals(session, rows)
    return rows


async def _expire_non_tradable_pending_signals(
    session: AsyncSession,
    rows: list[TradeSignal],
) -> list[TradeSignal]:
    if not rows:
        return rows

    pending_rows = [row for row in rows if row.status == "pending" and row.market_id]
    if not pending_rows:
        return rows

    tradability = await get_market_tradability_map([str(row.market_id) for row in pending_rows])
    now = _utc_now()
    max_price_age_seconds = float(getattr(settings, "SCANNER_MARKET_PRICE_MAX_AGE_SECONDS", 0) or 0)
    if max_price_age_seconds <= 0:
        max_price_age_seconds = max(30.0, float(getattr(settings, "WS_PRICE_STALE_SECONDS", 30.0) or 30.0) * 2.0)
    changed = False
    output: list[TradeSignal] = []
    for row in rows:
        if row.status != "pending":
            output.append(row)
            continue
        source_key = str(row.source or "").strip().lower()
        if (
            source_key == "scanner"
            and not _uses_runtime_price_revalidation(row.payload_json, row.strategy_context_json)
            and _payload_contains_stale_market_prices(row.payload_json, now, max_price_age_seconds)
            and not _signal_recently_refreshed(row, now, max_price_age_seconds)
        ):
            row.status = "expired"
            row.updated_at = now
            await _record_signal_emission(
                session,
                row,
                event_type="status_expired",
                reason="market_price_stale",
            )
            changed = True
            continue
        market_key = str(row.market_id or "").strip().lower()
        if tradability.get(market_key, True):
            output.append(row)
            continue
        row.status = "expired"
        row.updated_at = now
        await _record_signal_emission(
            session,
            row,
            event_type="status_expired",
            reason="market_not_tradable",
        )
        changed = True

    if changed:
        await session.commit()

    return output


async def list_trade_signal_source_stats(
    session: AsyncSession,
) -> list[tuple[str, str, int, datetime | None, datetime | None]]:
    return list(
        (
            await session.execute(
                select(
                    TradeSignal.source,
                    TradeSignal.status,
                    func.count(TradeSignal.id),
                    func.max(TradeSignal.created_at),
                    func.min(
                        case(
                            (TradeSignal.status == "pending", TradeSignal.created_at),
                            else_=None,
                        )
                    ),
                ).group_by(TradeSignal.source, TradeSignal.status)
            )
        ).all()
    )


def _serialize_trade_signal_source_stats(
    rows: list[tuple[str, str, int, datetime | None, datetime | None]],
) -> list[dict[str, Any]]:
    source_stats: dict[str, dict[str, Any]] = {}
    for source, status, count, latest_created, oldest_pending in rows:
        stats = source_stats.setdefault(
            source,
            {
                "source": source,
                "pending_count": 0,
                "selected_count": 0,
                "submitted_count": 0,
                "executed_count": 0,
                "skipped_count": 0,
                "expired_count": 0,
                "failed_count": 0,
                "latest_signal_at": None,
                "oldest_pending_at": None,
            },
        )
        key = f"{status}_count"
        if key in stats:
            stats[key] = int(count or 0)
        if latest_created and (stats["latest_signal_at"] is None or latest_created > stats["latest_signal_at"]):
            stats["latest_signal_at"] = latest_created
        if oldest_pending and (stats["oldest_pending_at"] is None or oldest_pending < stats["oldest_pending_at"]):
            stats["oldest_pending_at"] = oldest_pending

    now = _utc_now()
    out: list[dict[str, Any]] = []
    for source in sorted(source_stats.keys()):
        stats = source_stats[source]
        out.append(
            {
                "source": source,
                "pending_count": int(stats.get("pending_count", 0) or 0),
                "selected_count": int(stats.get("selected_count", 0) or 0),
                "submitted_count": int(stats.get("submitted_count", 0) or 0),
                "executed_count": int(stats.get("executed_count", 0) or 0),
                "skipped_count": int(stats.get("skipped_count", 0) or 0),
                "expired_count": int(stats.get("expired_count", 0) or 0),
                "failed_count": int(stats.get("failed_count", 0) or 0),
                "latest_signal_at": stats.get("latest_signal_at"),
                "oldest_pending_at": stats.get("oldest_pending_at"),
                "freshness_seconds": (
                    (now - stats["latest_signal_at"]).total_seconds() if stats.get("latest_signal_at") else None
                ),
                "updated_at": now,
            }
        )
    return out


async def read_trade_signal_source_stats(session: AsyncSession) -> list[dict[str, Any]]:
    rows = await list_trade_signal_source_stats(session)
    return _serialize_trade_signal_source_stats(rows)


def _direction_from_outcome(outcome: Optional[str]) -> Optional[str]:
    val = (outcome or "").lower().strip()
    if val == "yes":
        return "buy_yes"
    if val == "no":
        return "buy_no"
    return None
