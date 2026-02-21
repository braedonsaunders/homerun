from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

from services.live_execution_adapter import execute_live_order
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

    if mode_key != "live":
        return LegSubmitResult(
            leg_id=leg_id,
            status="executed",
            effective_price=price,
            error_message=None,
            payload={
                "mode": "paper",
                "submission": "simulated",
                "filled_notional_usd": notional,
                "shares": shares,
                "leg": dict(leg),
            },
            shares=shares,
            notional_usd=notional,
        )

    token_id, token_source, token_attempts = _resolve_token_id_for_leg(
        leg=leg,
        payload=payload,
        live_context=live_context,
    )
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
