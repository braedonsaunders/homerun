"""Normalized trade signal bus helpers.

Worker pipelines emit into ``trade_signals`` and consumers (trader orchestrator/UI)
read from one normalized contract.

# Helpers for WRITING/UPSERTING signals to DB from opportunities.
# For helpers that READ signal data back from DB TradeSignal rows (e.g. for
# strategy evaluate/should_exit), see utils/signal_helpers.py (signal_payload).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from utils.utcnow import utcnow
from typing import Any, Optional

from config import settings
from sqlalchemy import case, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    TradeSignal,
    TradeSignalEmission,
    TradeSignalSnapshot,
)
from services.event_bus import event_bus
from services.trade_signal_stream import publish_trade_signal_batch as publish_trade_signal_stream_batch
from models.opportunity import Opportunity
from services.market_tradability import get_market_tradability_map


SIGNAL_TERMINAL_STATUSES = {"executed", "skipped", "expired", "failed"}
SIGNAL_ACTIVE_STATUSES = {"pending", "selected", "submitted"}
SIGNAL_REACTIVATABLE_STATUSES = {"selected", "submitted", "executed", "skipped", "expired", "failed"}
_SKIPPED_REACTIVATION_VOLATILE_KEYS = {
    "bridge_run_at",
    "bridge_source",
    "ingested_at",
    "market_data_age_ms",
    "signal_emitted_at",
}
_SKIPPED_REACTIVATION_FLAG_KEYS = {
    "asset",
    "timeframe",
    "regime",
    "selected_direction",
    "selected_outcome",
    "oracle_available",
    "oracle_source",
    "is_live",
    "is_current",
}
_SKIPPED_REACTIVATION_PRICE_DELTA = 0.005
_SKIPPED_REACTIVATION_EDGE_DELTA = 0.25
_SKIPPED_REACTIVATION_CONFIDENCE_DELTA = 0.01
_SKIPPED_REACTIVATION_LIQUIDITY_DELTA_ABS = 250.0
_SKIPPED_REACTIVATION_LIQUIDITY_DELTA_REL = 0.10
_SKIPPED_REACTIVATION_EXPIRY_DELTA_SECONDS = 15.0
_SKIPPED_REACTIVATION_LIQUIDITY_BANDS = (250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 32000.0)


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
        return round(float(value), 10)
    return value


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
    if _normalize_number(row.entry_price) != _normalize_number(entry_price):
        return True
    if _normalize_number(row.edge_percent) != _normalize_number(edge_percent):
        return True
    if _normalize_number(row.confidence) != _normalize_number(confidence):
        return True
    if _normalize_number(row.liquidity) != _normalize_number(liquidity):
        return True
    existing_payload = _normalize_reactivation_value(_safe_json(row.payload_json))
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


def _execution_profile_for_opportunity(opportunity: Opportunity, legs_count: int) -> dict[str, Any]:
    strategy_key = str(getattr(opportunity, "strategy", "") or "").strip().lower()
    strategy_context = getattr(opportunity, "strategy_context", None)
    source_key = ""
    if isinstance(strategy_context, dict):
        source_key = str(strategy_context.get("source_key") or "").strip().lower()
    is_traders = source_key == "traders" or strategy_key.startswith("traders")

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


def _normalize_execution_plan(opportunity: Opportunity) -> dict[str, Any] | None:
    existing = getattr(opportunity, "execution_plan", None)
    if existing is not None:
        if hasattr(existing, "model_dump"):
            dumped = existing.model_dump(mode="json")
            if isinstance(dumped, dict) and list(dumped.get("legs") or []):
                return dumped
        elif isinstance(existing, dict) and list(existing.get("legs") or []):
            return dict(existing)

    positions = list(getattr(opportunity, "positions_to_take", None) or [])
    if not positions:
        return None

    markets = list(getattr(opportunity, "markets", None) or [])
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
        if not isinstance(position, dict):
            continue
        market_id = str(
            position.get("market_id")
            or position.get("id")
            or position.get("market")
            or (markets[index].get("id") if index < len(markets) and isinstance(markets[index], dict) else "")
            or ""
        ).strip()
        if not market_id:
            continue
        market = market_by_id.get(market_id) or (
            markets[index] if index < len(markets) and isinstance(markets[index], dict) else {}
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
                "market_question": str(position.get("market_question") or market.get("question") or opportunity.title),
                "token_id": str(position.get("token_id") or "").strip() or None,
                "side": "sell" if action.startswith("sell") else "buy",
                "outcome": str(position.get("outcome") or "").strip().lower() or None,
                "limit_price": parsed_price,
                "price_policy": str(position.get("price_policy") or profile["price_policy"]),
                "time_in_force": str(position.get("time_in_force") or profile["time_in_force"]),
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

    payload = opportunity.model_dump(mode="json")
    if plan is not None:
        payload["execution_plan"] = plan
    strategy_context = dict(opportunity.strategy_context or {})
    if plan is not None:
        strategy_context["execution_plan"] = plan

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
    snapshot = {
        "signal_id": row.id,
        "source": row.source,
        "source_item_id": row.source_item_id,
        "signal_type": row.signal_type,
        "strategy_type": row.strategy_type,
        "market_id": row.market_id,
        "direction": row.direction,
        "entry_price": row.entry_price,
        "effective_price": row.effective_price,
        "edge_percent": row.edge_percent,
        "confidence": row.confidence,
        "liquidity": row.liquidity,
        "status": row.status,
        "dedupe_key": row.dedupe_key,
        "expires_at": row.expires_at.isoformat() if row.expires_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }
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
            payload_json=_safe_json(row.payload_json),
            snapshot_json=snapshot,
            created_at=_utc_now(),
        )
    )


async def _publish_trade_signal_emission(
    *,
    row: TradeSignal,
    event_type: str,
    reason: str | None = None,
) -> None:
    try:
        await event_bus.publish(
            "trade_signal_emission",
            {
                "signal_id": str(row.id or ""),
                "source": str(row.source or ""),
                "status": str(row.status or ""),
                "event_type": str(event_type),
                "reason": reason,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            },
        )
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
    try:
        await event_bus.publish("trade_signal_batch", payload)
    except Exception:
        pass
    try:
        await publish_trade_signal_stream_batch(
            event_type=normalized_event_type,
            source=normalized_source,
            signal_ids=signal_ids,
            trigger="signal_bus",
            reason=reason,
            emitted_at=emitted_at_iso,
        )
    except Exception:
        pass


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
    commit: bool = True,
) -> TradeSignal:
    """Idempotently upsert a normalized trade signal by ``(source, dedupe_key)``."""
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

    if row is None:
        row = TradeSignal(
            id=uuid.uuid4().hex,
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
            expires_at=_to_utc_naive(expires_at),
            payload_json=_safe_json(payload_json),
            strategy_context_json=_safe_json(strategy_context_json),
            quality_passed=quality_passed,
            quality_rejection_reasons=quality_rejection_reasons if quality_rejection_reasons else None,
            dedupe_key=dedupe_key,
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
        previous_status = str(row.status or "").strip().lower()
        can_update_row = previous_status in SIGNAL_ACTIVE_STATUSES or previous_status in SIGNAL_REACTIVATABLE_STATUSES
        if can_update_row:
            has_material_change = True
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
                    payload_json=payload_json,
                    strategy_context_json=strategy_context_json,
                    quality_passed=quality_passed,
                    quality_rejection_reasons=quality_rejection_reasons,
                )
                if not has_material_change:
                    existing_expires_at = _to_utc_naive(row.expires_at)
                    incoming_expires_at = _to_utc_naive(expires_at)
                    if existing_expires_at != incoming_expires_at:
                        has_material_change = True
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
                    payload_json=payload_json,
                    strategy_context_json=strategy_context_json,
                    expires_at=expires_at,
                    quality_passed=quality_passed,
                    quality_rejection_reasons=quality_rejection_reasons,
                )
            if previous_status in SIGNAL_ACTIVE_STATUSES and not has_material_change:
                emission_event_type = "upsert_active_unchanged"
                emission_reason = "suppressed:active_unchanged"
                publish_signal_emission = False
            else:
                should_reactivate = previous_status in SIGNAL_REACTIVATABLE_STATUSES
                if previous_status == "skipped":
                    should_reactivate = has_material_change
                if previous_status == "skipped" and not should_reactivate:
                    emission_event_type = "upsert_skipped_unchanged"
                    emission_reason = "reactivation_suppressed:skipped_unchanged"
                    publish_signal_emission = False
                else:
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
                    row.payload_json = _safe_json(payload_json)
                    row.strategy_context_json = _safe_json(strategy_context_json)
                    if quality_passed is not None:
                        row.quality_passed = quality_passed
                        row.quality_rejection_reasons = quality_rejection_reasons if quality_rejection_reasons else None
                    emission_event_type = "upsert_update"
                    emission_reason = None
                    if should_reactivate:
                        row.status = "pending"
                        row.effective_price = None
                        emission_event_type = "upsert_reactivated"
                        emission_reason = f"reactivated_from:{previous_status}"
                    row.updated_at = _utc_now()
                    await _record_signal_emission(
                        session,
                        row,
                        event_type=emission_event_type,
                        reason=emission_reason,
                    )
        else:
            emission_event_type = "upsert_ignored_terminal"
            emission_reason = f"terminal_status:{previous_status}"
            await _record_signal_emission(
                session,
                row,
                event_type="upsert_ignored_terminal",
                reason=f"terminal_status:{previous_status}",
            )

    if commit:
        await session.commit()
        await refresh_trade_signal_snapshots(session)
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
        await refresh_trade_signal_snapshots(session)
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
    result = await session.execute(
        select(TradeSignal).where(
            TradeSignal.status.in_(tuple(SIGNAL_ACTIVE_STATUSES)),
        )
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
        await refresh_trade_signal_snapshots(session)
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
    commit: bool = True,
) -> int:
    """Expire pending source signals not present in the current emission set."""
    now = _to_utc_naive(_utc_now())
    keep = {str(key) for key in keep_dedupe_keys if str(key).strip()}

    query = select(TradeSignal).where(
        TradeSignal.source == str(source),
        TradeSignal.status == "pending",
    )
    if signal_types:
        normalized_signal_types = [
            str(value or "").strip().lower() for value in signal_types if str(value or "").strip()
        ]
        if normalized_signal_types:
            query = query.where(func.lower(func.coalesce(TradeSignal.signal_type, "")).in_(normalized_signal_types))
    if keep:
        query = query.where(~TradeSignal.dedupe_key.in_(list(keep)))

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
        await refresh_trade_signal_snapshots(session)
        await _publish_trade_signal_batch(
            event_type="status_expired",
            signal_ids=expired_signal_ids,
            source=str(source),
            reason="source_sweep",
        )
    elif commit:
        await session.commit()

    return len(rows)


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
        await refresh_trade_signal_snapshots(session)

    return output


async def refresh_trade_signal_snapshots(session: AsyncSession) -> list[dict[str, Any]]:
    """Recompute per-source snapshot rows from ``trade_signals``."""

    rows = (
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
    updated_sources: set[str] = set()

    for source, stats in source_stats.items():
        updated_sources.add(source)
        payload = {
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
            "stats_json": {
                "total": sum(
                    int(stats.get(k, 0) or 0)
                    for k in (
                        "pending_count",
                        "selected_count",
                        "submitted_count",
                        "executed_count",
                        "skipped_count",
                        "expired_count",
                        "failed_count",
                    )
                )
            },
        }
        stmt = pg_insert(TradeSignalSnapshot).values(**payload)
        stmt = stmt.on_conflict_do_update(
            index_elements=[TradeSignalSnapshot.source],
            set_={
                "pending_count": payload["pending_count"],
                "selected_count": payload["selected_count"],
                "submitted_count": payload["submitted_count"],
                "executed_count": payload["executed_count"],
                "skipped_count": payload["skipped_count"],
                "expired_count": payload["expired_count"],
                "failed_count": payload["failed_count"],
                "latest_signal_at": payload["latest_signal_at"],
                "oldest_pending_at": payload["oldest_pending_at"],
                "freshness_seconds": payload["freshness_seconds"],
                "updated_at": payload["updated_at"],
                "stats_json": payload["stats_json"],
            },
        )
        await session.execute(stmt)

    zero_payload = {
        "pending_count": 0,
        "selected_count": 0,
        "submitted_count": 0,
        "executed_count": 0,
        "skipped_count": 0,
        "expired_count": 0,
        "failed_count": 0,
        "latest_signal_at": None,
        "oldest_pending_at": None,
        "freshness_seconds": None,
        "updated_at": now,
        "stats_json": {"total": 0},
    }
    if updated_sources:
        await session.execute(
            update(TradeSignalSnapshot)
            .where(~TradeSignalSnapshot.source.in_(sorted(updated_sources)))
            .values(**zero_payload)
        )
    else:
        await session.execute(update(TradeSignalSnapshot).values(**zero_payload))

    await session.commit()

    out: list[dict[str, Any]] = []
    snapshot_rows = (
        (await session.execute(select(TradeSignalSnapshot).order_by(TradeSignalSnapshot.source.asc()))).scalars().all()
    )
    for row in snapshot_rows:
        out.append(
            {
                "source": row.source,
                "pending_count": int(row.pending_count or 0),
                "selected_count": int(row.selected_count or 0),
                "submitted_count": int(row.submitted_count or 0),
                "executed_count": int(row.executed_count or 0),
                "skipped_count": int(row.skipped_count or 0),
                "expired_count": int(row.expired_count or 0),
                "failed_count": int(row.failed_count or 0),
                "latest_signal_at": row.latest_signal_at,
                "oldest_pending_at": row.oldest_pending_at,
                "freshness_seconds": row.freshness_seconds,
                "updated_at": row.updated_at,
            }
        )
    return out


def _direction_from_outcome(outcome: Optional[str]) -> Optional[str]:
    val = (outcome or "").lower().strip()
    if val == "yes":
        return "buy_yes"
    if val == "no":
        return "buy_no"
    return None
