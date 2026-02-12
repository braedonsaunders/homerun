"""Resolver for mapping world-intelligence signals to tradable market intents."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from services.market_tradability import get_market_tradability_map
from services.shared_state import read_scanner_snapshot


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_outcome(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    if text in {"yes", "buy_yes", "y"}:
        return "yes"
    if text in {"no", "buy_no", "n"}:
        return "no"
    return None


def _normalize_direction(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    if text in {"buy_yes", "yes"}:
        return "buy_yes"
    if text in {"buy_no", "no"}:
        return "buy_no"
    return None


def _signal_value(signal: Any, field: str, default: Any = None) -> Any:
    if isinstance(signal, dict):
        return signal.get(field, default)
    return getattr(signal, field, default)


def _signal_metadata(signal: Any) -> dict[str, Any]:
    if isinstance(signal, dict):
        raw = signal.get("metadata") or signal.get("metadata_json") or {}
    else:
        raw = (
            getattr(signal, "metadata", None)
            or getattr(signal, "metadata_json", None)
            or {}
        )
    return raw if isinstance(raw, dict) else {}


def _signal_related_market_ids(signal: Any) -> list[str]:
    if isinstance(signal, dict):
        raw = signal.get("related_market_ids") or []
    else:
        raw = getattr(signal, "related_market_ids", None) or []
    if not isinstance(raw, list):
        return []
    ids: list[str] = []
    for item in raw:
        key = str(item or "").strip()
        if key and key not in ids:
            ids.append(key)
    return ids


def estimate_edge_percent(signal: Any) -> float:
    signal_type = str(_signal_value(signal, "signal_type") or "").strip().lower()
    severity = _safe_float(_signal_value(signal, "severity"), 0.0)
    edge = max(0.0, severity) * 15.0
    if signal_type == "convergence":
        edge *= 1.3
    elif signal_type == "anomaly":
        edge *= 1.2
    return min(25.0, max(5.0, edge))


def infer_direction(signal: Any) -> Optional[str]:
    metadata = _signal_metadata(signal)
    explicit = _normalize_direction(metadata.get("direction"))
    if explicit:
        return explicit

    signal_type = str(_signal_value(signal, "signal_type") or "").strip().lower()
    severity = _safe_float(_signal_value(signal, "severity"), 0.0)

    if signal_type in {"conflict", "military", "instability", "convergence", "tension"}:
        if severity >= 0.6:
            return "buy_yes"

    if signal_type == "anomaly":
        z_score = _safe_float(metadata.get("z_score"), 0.0)
        if z_score >= 2.0:
            return "buy_yes"
        if z_score <= -2.0:
            return "buy_no"

    return None


def map_signal_to_strategy(signal_type: str) -> str:
    mapping = {
        "conflict": "event_driven",
        "tension": "bayesian_cascade",
        "instability": "event_driven",
        "convergence": "event_driven",
        "anomaly": "stat_arb",
        "military": "event_driven",
        "infrastructure": "event_driven",
    }
    return mapping.get(str(signal_type or "").strip().lower(), "event_driven")


def _to_iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.replace(tzinfo=None).isoformat() + "Z"


def _build_market_lookup(opportunities: list[Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}

    for opp in opportunities:
        event_id = str(getattr(opp, "event_id", "") or "")
        event_slug = str(getattr(opp, "event_slug", "") or "")
        event_title = str(getattr(opp, "event_title", "") or "")
        category = str(getattr(opp, "category", "") or "")

        market_rows = getattr(opp, "markets", []) or []
        positions = getattr(opp, "positions_to_take", []) or []

        for market in market_rows:
            if not isinstance(market, dict):
                continue
            market_id = str(market.get("id") or market.get("condition_id") or "").strip()
            if not market_id:
                continue
            row = lookup.setdefault(
                market_id,
                {
                    "market_id": market_id,
                    "market_question": str(
                        market.get("question")
                        or market.get("title")
                        or getattr(opp, "title", "")
                        or market_id
                    ),
                    "event_id": event_id or None,
                    "event_slug": event_slug or None,
                    "event_title": event_title or None,
                    "category": category or None,
                    "liquidity": _safe_float(market.get("liquidity"), _safe_float(getattr(opp, "min_liquidity", 0.0))),
                    "yes": {"token_id": None, "price": None},
                    "no": {"token_id": None, "price": None},
                },
            )
            yes_price = _safe_float(
                market.get("yes_price"),
                _safe_float(market.get("best_bid"), 0.0),
            )
            no_price = _safe_float(
                market.get("no_price"),
                _safe_float(market.get("best_ask"), 0.0),
            )
            if yes_price > 0 and row["yes"]["price"] is None:
                row["yes"]["price"] = yes_price
            if no_price > 0 and row["no"]["price"] is None:
                row["no"]["price"] = no_price

        for position in positions:
            if not isinstance(position, dict):
                continue
            market_id = str(position.get("market_id") or position.get("market") or "").strip()
            if not market_id:
                continue
            outcome = _normalize_outcome(position.get("outcome"))
            if outcome is None:
                continue
            row = lookup.setdefault(
                market_id,
                {
                    "market_id": market_id,
                    "market_question": str(getattr(opp, "title", "") or market_id),
                    "event_id": event_id or None,
                    "event_slug": event_slug or None,
                    "event_title": event_title or None,
                    "category": category or None,
                    "liquidity": _safe_float(getattr(opp, "min_liquidity", 0.0)),
                    "yes": {"token_id": None, "price": None},
                    "no": {"token_id": None, "price": None},
                },
            )
            side = row[outcome]
            token_id = str(position.get("token_id") or "").strip() or None
            price = _safe_float(position.get("price"), 0.0)
            if token_id and not side["token_id"]:
                side["token_id"] = token_id
            if price > 0 and side["price"] is None:
                side["price"] = price
            if str(position.get("market_question") or "").strip():
                row["market_question"] = str(position.get("market_question")).strip()

    return lookup


def _resolve_candidate(
    signal: Any,
    market_id: str,
    market_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    signal_id = str(_signal_value(signal, "signal_id", _signal_value(signal, "id", "")) or "")
    signal_type = str(_signal_value(signal, "signal_type", "unknown") or "unknown")
    severity = _safe_float(_signal_value(signal, "severity"), 0.0)
    market_relevance_score = _safe_float(_signal_value(signal, "market_relevance_score"), 0.0)
    direction = infer_direction(signal)
    market = market_lookup.get(market_id, {})

    entry_price = None
    token_id = None
    outcome = None

    if direction == "buy_yes":
        outcome = "YES"
        token_id = market.get("yes", {}).get("token_id")
        entry_price = market.get("yes", {}).get("price")
    elif direction == "buy_no":
        outcome = "NO"
        token_id = market.get("no", {}).get("token_id")
        entry_price = market.get("no", {}).get("price")

    entry_price_value = _safe_float(entry_price, 0.0)
    if entry_price_value <= 0:
        entry_price = None
    else:
        entry_price = entry_price_value

    missing_fields: list[str] = []
    if not direction:
        missing_fields.append("direction")
    if not token_id:
        missing_fields.append("token_id")
    if entry_price is None:
        missing_fields.append("entry_price")

    confidence = min(1.0, max(0.0, severity * 0.6 + market_relevance_score * 0.4))
    tradable = len(missing_fields) == 0

    position = None
    if tradable:
        position = {
            "action": "buy",
            "outcome": outcome,
            "market": market_id,
            "market_id": market_id,
            "market_question": market.get("market_question") or market_id,
            "price": entry_price,
            "token_id": token_id,
        }

    detected_at_raw = _signal_value(signal, "detected_at")
    if isinstance(detected_at_raw, datetime):
        detected_at = _to_iso(detected_at_raw)
    elif isinstance(detected_at_raw, str):
        detected_at = detected_at_raw
    else:
        detected_at = None

    return {
        "id": f"wi:{signal_id}:{market_id}",
        "signal_id": signal_id,
        "signal_type": signal_type,
        "strategy_type": map_signal_to_strategy(signal_type),
        "severity": severity,
        "country": _signal_value(signal, "country"),
        "source": _signal_value(signal, "source"),
        "title": _signal_value(signal, "title"),
        "description": _signal_value(signal, "description"),
        "detected_at": detected_at,
        "market_id": market_id,
        "market_question": market.get("market_question") or market_id,
        "event_id": market.get("event_id"),
        "event_slug": market.get("event_slug"),
        "event_title": market.get("event_title"),
        "category": market.get("category"),
        "liquidity": _safe_float(market.get("liquidity"), 0.0),
        "direction": direction,
        "outcome": outcome,
        "token_id": token_id,
        "entry_price": entry_price,
        "confidence": round(confidence, 4),
        "edge_percent": round(estimate_edge_percent(signal), 3),
        "market_relevance_score": round(market_relevance_score, 4),
        "related_market_ids": _signal_related_market_ids(signal),
        "metadata": _signal_metadata(signal),
        "resolver_status": "tradable" if tradable else "incomplete",
        "missing_fields": missing_fields,
        "tradable": tradable,
        "positions_to_take": [position] if position else [],
    }


async def resolve_world_signal_opportunities(
    session: AsyncSession,
    signals: list[Any],
    *,
    max_markets_per_signal: int = 5,
) -> list[dict[str, Any]]:
    opportunities, _ = await read_scanner_snapshot(session)
    market_lookup = _build_market_lookup(opportunities)

    resolved: list[dict[str, Any]] = []
    for signal in signals:
        market_ids = _signal_related_market_ids(signal)
        if not market_ids:
            continue
        for market_id in market_ids[: max(1, int(max_markets_per_signal))]:
            if not market_id:
                continue
            resolved.append(
                _resolve_candidate(signal, market_id, market_lookup)
            )

    if resolved:
        tradability = await get_market_tradability_map(
            [str(item.get("market_id") or "") for item in resolved]
        )
        for item in resolved:
            market_key = str(item.get("market_id") or "").strip().lower()
            if tradability.get(market_key, True):
                continue
            missing = item.get("missing_fields")
            if not isinstance(missing, list):
                missing = []
            if "market_not_tradable" not in missing:
                missing.append("market_not_tradable")
            item["missing_fields"] = missing
            item["tradable"] = False
            item["resolver_status"] = "market_not_tradable"
            item["positions_to_take"] = []

    resolved.sort(
        key=lambda item: (
            bool(item.get("tradable")),
            _safe_float(item.get("market_relevance_score"), 0.0),
            _safe_float(item.get("severity"), 0.0),
        ),
        reverse=True,
    )
    return resolved
