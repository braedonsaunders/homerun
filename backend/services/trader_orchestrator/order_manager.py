from __future__ import annotations

import asyncio
import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any

from services.live_execution_adapter import execute_live_order
from services.polymarket import polymarket_client
from services.trading import trading_service
from utils.converters import safe_float


_MIN_EXECUTION_PRICE = 0.001
_NUMERIC_TOKEN_ID_RE = re.compile(r"^\d{18,}$")
_HEX_TOKEN_ID_RE = re.compile(r"^(?:0x)?[0-9a-f]{40,}$")
_CONDITION_ID_RE = re.compile(r"^0x[0-9a-f]{64}$")


@dataclass
class LegSubmitResult:
    leg_id: str
    status: str
    effective_price: float | None
    error_message: str | None
    payload: dict[str, Any]
    provider_order_id: str | None = None
    provider_clob_order_id: str | None = None
    shares: float | None = None
    notional_usd: float | None = None


@dataclass
class PaperExecutionResult:
    status: str
    effective_price: float | None
    error_message: str | None
    payload: dict[str, Any]
    shares: float | None
    notional_usd: float | None


def _normalize_id(value: Any) -> str:
    return str(value or "").strip().lower()


def _looks_like_token_id(value: Any) -> bool:
    normalized = _normalize_id(value)
    if not normalized:
        return False
    if _CONDITION_ID_RE.fullmatch(normalized):
        return False
    return bool(_NUMERIC_TOKEN_ID_RE.fullmatch(normalized) or _HEX_TOKEN_ID_RE.fullmatch(normalized))


def _safe_signal_payload(signal: Any) -> dict[str, Any]:
    payload = getattr(signal, "payload_json", None)
    return payload if isinstance(payload, dict) else {}


def _safe_live_context(signal: Any, payload: dict[str, Any]) -> dict[str, Any]:
    context = getattr(signal, "live_context", None)
    if isinstance(context, dict):
        return context
    from_payload = payload.get("live_market")
    if isinstance(from_payload, dict):
        return from_payload
    return {}


def _resolve_token_id_for_leg(
    *,
    leg: dict[str, Any],
    payload: dict[str, Any],
    live_context: dict[str, Any],
) -> tuple[str | None, str | None, list[str]]:
    candidates: list[tuple[str, str]] = []

    def _append(source: str, value: Any) -> None:
        normalized = _normalize_id(value)
        if normalized:
            candidates.append((source, normalized))

    _append("leg.token_id", leg.get("token_id"))
    _append("live_context.selected_token_id", live_context.get("selected_token_id"))
    _append("payload.selected_token_id", payload.get("selected_token_id"))

    outcome = str(leg.get("outcome") or "").strip().lower()
    side = str(leg.get("side") or "buy").strip().lower()
    if outcome == "yes":
        _append("live_context.yes_token_id", live_context.get("yes_token_id"))
        _append("payload.yes_token_id", payload.get("yes_token_id"))
    elif outcome == "no":
        _append("live_context.no_token_id", live_context.get("no_token_id"))
        _append("payload.no_token_id", payload.get("no_token_id"))
    elif side == "buy":
        _append("live_context.selected_token_id", live_context.get("selected_token_id"))

    token_ids = live_context.get("token_ids")
    if not isinstance(token_ids, list):
        token_ids = payload.get("token_ids")
    if isinstance(token_ids, list):
        for index, token in enumerate(token_ids):
            _append(f"token_ids[{index}]", token)

    _append("payload.token_id", payload.get("token_id"))

    for source, candidate in candidates:
        if _looks_like_token_id(candidate):
            return candidate, source, [entry[0] for entry in candidates]
    return None, None, [entry[0] for entry in candidates]


def _resolve_leg_price(leg: dict[str, Any], signal: Any, live_context: dict[str, Any]) -> float | None:
    limit_price = safe_float(leg.get("limit_price"), None)
    if limit_price is not None and limit_price > 0:
        return limit_price
    live_price = safe_float(live_context.get("live_selected_price"), None)
    if live_price is not None and live_price > 0:
        return live_price
    signal_price = safe_float(getattr(signal, "entry_price", None), None)
    if signal_price is not None and signal_price > 0:
        return signal_price
    return None


def _hash_ratio(*parts: Any) -> float:
    packed = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(packed.encode("utf-8")).hexdigest()
    numerator = int(digest[:15], 16)
    denominator = float(0xFFFFFFFFFFFFFFF)
    return numerator / denominator if denominator > 0 else 0.5


def _clamp(value: float, *, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _extract_market_liquidity(
    *,
    signal: Any,
    leg: dict[str, Any],
    payload: dict[str, Any],
    live_context: dict[str, Any],
) -> float:
    candidates: list[float] = []
    for value in (
        leg.get("liquidity"),
        ((leg.get("metadata") or {}).get("liquidity") if isinstance(leg.get("metadata"), dict) else None),
        payload.get("liquidity"),
        payload.get("min_liquidity"),
        payload.get("max_position_size"),
        getattr(signal, "liquidity", None),
        live_context.get("liquidity"),
    ):
        parsed = safe_float(value, None)
        if parsed is not None and parsed > 0:
            candidates.append(parsed)

    market_id = str(leg.get("market_id") or getattr(signal, "market_id", "") or "").strip()
    markets = payload.get("markets")
    if isinstance(markets, list) and market_id:
        for market in markets:
            if not isinstance(market, dict):
                continue
            if str(market.get("id") or "").strip() != market_id:
                continue
            for key in ("liquidity", "min_liquidity", "volume", "volume24h", "volume_24h"):
                parsed = safe_float(market.get(key), None)
                if parsed is not None and parsed > 0:
                    candidates.append(parsed)

    if not candidates:
        return 5_000.0
    return max(100.0, min(5_000_000.0, max(candidates)))


def _extract_volatility_factor(live_context: dict[str, Any]) -> float:
    summary = live_context.get("history_summary")
    if not isinstance(summary, dict):
        return 0.0

    move_5m_bucket = summary.get("move_5m")
    move_5m = safe_float((move_5m_bucket or {}).get("percent"), 0.0) if isinstance(move_5m_bucket, dict) else 0.0
    move_30m = (
        safe_float((summary.get("move_30m") or {}).get("percent"), 0.0)
        if isinstance(summary.get("move_30m"), dict)
        else 0.0
    )
    move_2h_bucket = summary.get("move_2h")
    move_2h = safe_float((move_2h_bucket or {}).get("percent"), 0.0) if isinstance(move_2h_bucket, dict) else 0.0
    intensity = (abs(move_5m) * 2.0) + abs(move_30m) + (abs(move_2h) * 0.5)
    return _clamp(intensity / 18.0, lo=0.0, hi=1.0)


def _paper_slippage_profile(price_policy: str, time_in_force: str) -> tuple[float, float]:
    policy = str(price_policy or "").strip().lower()
    tif = str(time_in_force or "").strip().upper()

    if policy in {"market", "marketable", "aggressive", "taker_limit"}:
        base_fill = 0.975
        base_slippage_bps = 14.0
    elif policy in {"maker_limit", "maker", "post_only"}:
        base_fill = 0.93
        base_slippage_bps = 3.5
    else:
        base_fill = 0.95
        base_slippage_bps = 8.0

    if tif == "IOC":
        base_fill += 0.01
        base_slippage_bps += 2.0
    elif tif == "FOK":
        base_fill -= 0.04
        base_slippage_bps += 1.0

    return _clamp(base_fill, lo=0.05, hi=0.995), max(0.0, base_slippage_bps)


def _simulate_paper_execution(
    *,
    signal: Any,
    leg: dict[str, Any],
    payload: dict[str, Any],
    live_context: dict[str, Any],
    notional_usd: float,
    target_price: float,
) -> PaperExecutionResult:
    leg_id = str(leg.get("leg_id") or "leg")
    side_key = str(leg.get("side") or "buy").strip().lower()
    is_buy = side_key != "sell"
    price_policy = str(leg.get("price_policy") or "maker_limit")
    time_in_force = str(leg.get("time_in_force") or "GTC").upper()
    min_fill_ratio = _clamp(safe_float(leg.get("min_fill_ratio"), 0.0), lo=0.0, hi=1.0)

    liquidity = _extract_market_liquidity(
        signal=signal,
        leg=leg,
        payload=payload,
        live_context=live_context,
    )
    participation = _clamp(notional_usd / max(1.0, liquidity), lo=0.0, hi=5.0)
    volatility = _extract_volatility_factor(live_context)
    confidence = _clamp(safe_float(getattr(signal, "confidence", None), 0.5), lo=0.0, hi=1.0)
    edge_percent = abs(safe_float(getattr(signal, "edge_percent", None), 0.0))

    entry_delta_pct = safe_float(live_context.get("entry_price_delta_pct"), 0.0)
    if is_buy:
        adverse_pressure = _clamp(max(0.0, entry_delta_pct) / 5.0, lo=0.0, hi=1.0)
    else:
        adverse_pressure = _clamp(max(0.0, -entry_delta_pct) / 5.0, lo=0.0, hi=1.0)

    base_fill_probability, base_slippage_bps = _paper_slippage_profile(price_policy, time_in_force)
    fill_probability = base_fill_probability
    fill_probability -= min(0.55, (participation * 0.12) + (math.sqrt(participation) * 0.03))
    fill_probability -= volatility * 0.12
    fill_probability -= adverse_pressure * 0.15
    fill_probability += min(0.06, confidence * 0.05)
    fill_probability += min(0.06, edge_percent / 250.0)
    fill_probability = _clamp(fill_probability, lo=0.02, hi=0.995)

    seed_prefix = (
        str(getattr(signal, "id", "") or ""),
        str(getattr(signal, "market_id", "") or ""),
        leg_id,
        str(leg.get("market_id") or ""),
        f"{notional_usd:.6f}",
        f"{target_price:.6f}",
        str(price_policy),
        str(time_in_force),
    )
    decision_roll = _hash_ratio(*seed_prefix, "decision")
    fill_roll = _hash_ratio(*seed_prefix, "fill_ratio")
    jitter_roll = _hash_ratio(*seed_prefix, "jitter")
    latency_roll = _hash_ratio(*seed_prefix, "latency")

    if decision_roll > fill_probability:
        return PaperExecutionResult(
            status="failed",
            effective_price=target_price,
            error_message="Paper simulation: order was not filled within modeled queue/latency window.",
            payload={
                "mode": "paper",
                "submission": "simulated",
                "paper_simulation": {
                    "filled": False,
                    "fill_probability": fill_probability,
                    "fill_ratio": 0.0,
                    "participation": participation,
                    "liquidity_usd": liquidity,
                    "volatility_factor": volatility,
                    "adverse_pressure": adverse_pressure,
                    "price_policy": price_policy,
                    "time_in_force": time_in_force,
                },
                "leg": dict(leg),
            },
            shares=None,
            notional_usd=0.0,
        )

    fill_ratio = 1.0 - min(0.90, (participation * 0.45) + (volatility * 0.22) + (adverse_pressure * 0.12))
    fill_ratio += (fill_roll - 0.5) * 0.28
    fill_ratio = _clamp(fill_ratio, lo=0.05, hi=1.0)
    fill_ratio = max(fill_ratio, min_fill_ratio)

    if time_in_force == "FOK" and fill_ratio < 0.999:
        return PaperExecutionResult(
            status="failed",
            effective_price=target_price,
            error_message="Paper simulation: FOK order could not be fully filled.",
            payload={
                "mode": "paper",
                "submission": "simulated",
                "paper_simulation": {
                    "filled": False,
                    "fill_probability": fill_probability,
                    "fill_ratio": fill_ratio,
                    "required_fill_ratio": 1.0,
                    "participation": participation,
                    "liquidity_usd": liquidity,
                    "volatility_factor": volatility,
                    "adverse_pressure": adverse_pressure,
                    "price_policy": price_policy,
                    "time_in_force": time_in_force,
                },
                "leg": dict(leg),
            },
            shares=None,
            notional_usd=0.0,
        )

    impact_bps = (math.sqrt(participation) * 22.0) + (participation * 38.0)
    volatility_bps = volatility * 18.0
    adverse_bps = adverse_pressure * 15.0
    jitter_bps = (jitter_roll - 0.5) * 8.0
    slippage_bps = _clamp(
        base_slippage_bps + impact_bps + volatility_bps + adverse_bps + jitter_bps,
        lo=0.0,
        hi=420.0,
    )
    slippage_factor = slippage_bps / 10_000.0

    effective_price = (
        target_price * (1.0 + slippage_factor if is_buy else max(0.0, 1.0 - slippage_factor))
    )
    effective_price = _clamp(effective_price, lo=_MIN_EXECUTION_PRICE, hi=0.9999)

    filled_notional = max(0.0, notional_usd * fill_ratio)
    if filled_notional <= 0.0:
        return PaperExecutionResult(
            status="failed",
            effective_price=effective_price,
            error_message="Paper simulation produced zero filled notional.",
            payload={
                "mode": "paper",
                "submission": "simulated",
                "paper_simulation": {
                    "filled": False,
                    "fill_probability": fill_probability,
                    "fill_ratio": fill_ratio,
                    "participation": participation,
                    "liquidity_usd": liquidity,
                    "volatility_factor": volatility,
                    "adverse_pressure": adverse_pressure,
                    "price_policy": price_policy,
                    "time_in_force": time_in_force,
                },
                "leg": dict(leg),
            },
            shares=None,
            notional_usd=0.0,
        )

    shares = filled_notional / effective_price if effective_price > 0 else 0.0
    if shares <= 0:
        return PaperExecutionResult(
            status="failed",
            effective_price=effective_price,
            error_message="Paper simulation produced zero shares.",
            payload={"mode": "paper", "submission": "simulated", "leg": dict(leg)},
            shares=None,
            notional_usd=0.0,
        )

    slippage_usd = abs(effective_price - target_price) * shares
    simulated_latency_ms = int(round(120.0 + (latency_roll * 900.0) + (participation * 140.0)))

    return PaperExecutionResult(
        status="executed",
        effective_price=effective_price,
        error_message=None,
        payload={
            "mode": "paper",
            "submission": "simulated",
            "filled_notional_usd": filled_notional,
            "shares": shares,
            "paper_simulation": {
                "filled": True,
                "fill_probability": fill_probability,
                "fill_ratio": fill_ratio,
                "price_policy": price_policy,
                "time_in_force": time_in_force,
                "target_price": target_price,
                "effective_price": effective_price,
                "slippage_bps": slippage_bps,
                "slippage_usd": slippage_usd,
                "estimated_fee_usd": 0.0,
                "simulated_latency_ms": simulated_latency_ms,
                "participation": participation,
                "liquidity_usd": liquidity,
                "volatility_factor": volatility,
                "adverse_pressure": adverse_pressure,
                "confidence": confidence,
                "edge_percent": edge_percent,
            },
            "leg": dict(leg),
        },
        shares=shares,
        notional_usd=filled_notional,
    )


async def _fetch_token_id_from_market(market_id: str, outcome: str) -> str | None:
    """Live fallback: query the Polymarket CLOB/Gamma API for token IDs by market condition_id."""
    condition_id = market_id.strip()
    if not condition_id:
        return None
    try:
        market_info = await polymarket_client.get_market_by_condition_id(condition_id)
    except Exception:
        return None
    if not isinstance(market_info, dict):
        return None

    for key in ("clobTokenIds", "clob_token_ids", "token_ids", "tokenIds"):
        raw = market_info.get(key)
        if isinstance(raw, list) and raw:
            token_ids = [str(t).strip() for t in raw if str(t).strip() and len(str(t).strip()) > 20]
            if not token_ids:
                continue
            outcome_lower = outcome.strip().lower()
            if outcome_lower == "yes" and len(token_ids) >= 1:
                return token_ids[0]
            if outcome_lower == "no" and len(token_ids) >= 2:
                return token_ids[1]
            return token_ids[0]
    return None


async def submit_execution_leg(
    *,
    mode: str,
    signal: Any,
    leg: dict[str, Any],
    notional_usd: float,
) -> LegSubmitResult:
    mode_key = str(mode or "").strip().lower()
    leg_id = str(leg.get("leg_id") or "leg")
    notional = float(max(0.0, notional_usd))
    payload = _safe_signal_payload(signal)
    live_context = _safe_live_context(signal, payload)
    price = _resolve_leg_price(leg, signal, live_context)
    side_key = str(leg.get("side") or "buy").strip().lower()
    order_side = "SELL" if side_key == "sell" else "BUY"

    if price is None or price <= 0:
        return LegSubmitResult(
            leg_id=leg_id,
            status="failed",
            effective_price=None,
            error_message="No valid price resolved for execution leg.",
            payload={"mode": mode_key, "leg": dict(leg), "reason": "missing_price"},
            shares=None,
            notional_usd=notional,
        )

    if price > 1.0:
        return LegSubmitResult(
            leg_id=leg_id,
            status="failed",
            effective_price=price,
            error_message="Execution price must be <= 1.0 for binary contracts.",
            payload={"mode": mode_key, "leg": dict(leg), "reason": "invalid_price_range"},
            shares=None,
            notional_usd=notional,
        )

    if price < _MIN_EXECUTION_PRICE:
        return LegSubmitResult(
            leg_id=leg_id,
            status="failed",
            effective_price=price,
            error_message=f"Execution price below minimum allowed ({_MIN_EXECUTION_PRICE:.4f}).",
            payload={"mode": mode_key, "leg": dict(leg), "reason": "invalid_price_too_small"},
            shares=None,
            notional_usd=notional,
        )

    if mode_key != "live":
        paper_result = _simulate_paper_execution(
            signal=signal,
            leg=leg,
            payload=payload,
            live_context=live_context,
            notional_usd=notional,
            target_price=price,
        )
        return LegSubmitResult(
            leg_id=leg_id,
            status=paper_result.status,
            effective_price=paper_result.effective_price,
            error_message=paper_result.error_message,
            payload=paper_result.payload,
            shares=paper_result.shares,
            notional_usd=paper_result.notional_usd,
        )

    shares = notional / price
    if shares <= 0:
        return LegSubmitResult(
            leg_id=leg_id,
            status="failed",
            effective_price=price,
            error_message="Computed leg size is zero.",
            payload={"mode": mode_key, "leg": dict(leg), "reason": "invalid_size"},
            shares=shares,
            notional_usd=notional,
        )

    token_id, token_source, token_attempts = _resolve_token_id_for_leg(
        leg=leg,
        payload=payload,
        live_context=live_context,
    )
    if not token_id:
        market_id_for_lookup = str(leg.get("market_id") or "").strip()
        outcome_for_lookup = str(leg.get("outcome") or "").strip()
        if market_id_for_lookup and outcome_for_lookup:
            token_id = await _fetch_token_id_from_market(market_id_for_lookup, outcome_for_lookup)
            if token_id:
                token_source = "polymarket_api_fallback"
                token_attempts.append(token_source)
    if not token_id:
        return LegSubmitResult(
            leg_id=leg_id,
            status="failed",
            effective_price=price,
            error_message="No executable token_id resolved for execution leg.",
            payload={
                "mode": "live",
                "submission": "rejected",
                "reason": "missing_token_id",
                "token_resolution_attempts": token_attempts,
                "leg": dict(leg),
            },
            shares=shares,
            notional_usd=notional,
        )

    execution = await execute_live_order(
        token_id=token_id,
        side=order_side,
        size=shares,
        fallback_price=price,
        market_question=str(leg.get("market_question") or getattr(signal, "market_question", "") or ""),
        opportunity_id=str(getattr(signal, "id", "") or ""),
    )

    return LegSubmitResult(
        leg_id=leg_id,
        status=execution.status,
        effective_price=execution.effective_price,
        error_message=execution.error_message,
        payload={
            **execution.payload,
            "mode": "live",
            "leg": dict(leg),
            "token_id_source": token_source,
            "shares": shares,
            "requested_notional_usd": notional,
        },
        provider_order_id=execution.order_id,
        provider_clob_order_id=str(execution.payload.get("clob_order_id") or "").strip() or None,
        shares=shares,
        notional_usd=notional,
    )


async def submit_execution_wave(
    *,
    mode: str,
    signal: Any,
    legs_with_notionals: list[tuple[dict[str, Any], float]],
) -> list[LegSubmitResult]:
    if not legs_with_notionals:
        return []
    tasks = [
        submit_execution_leg(
            mode=mode,
            signal=signal,
            leg=leg,
            notional_usd=notional,
        )
        for leg, notional in legs_with_notionals
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    normalized: list[LegSubmitResult] = []
    for index, result in enumerate(results):
        leg, notional = legs_with_notionals[index]
        leg_id = str(leg.get("leg_id") or f"leg_{index + 1}")
        if isinstance(result, Exception):
            normalized.append(
                LegSubmitResult(
                    leg_id=leg_id,
                    status="failed",
                    effective_price=safe_float(leg.get("limit_price"), None),
                    error_message=str(result),
                    payload={"mode": str(mode or "").lower(), "submission": "exception", "leg": dict(leg)},
                    shares=None,
                    notional_usd=notional,
                )
            )
            continue
        normalized.append(result)
    return normalized


async def cancel_live_provider_order(provider_order_id: str) -> bool:
    order_id = str(provider_order_id or "").strip()
    if not order_id:
        return False
    try:
        return bool(await trading_service.cancel_order(order_id))
    except Exception:
        return False
