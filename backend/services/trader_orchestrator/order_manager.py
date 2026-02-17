from __future__ import annotations

import re
from typing import Any

from services.live_execution_adapter import execute_live_order


_NUMERIC_TOKEN_ID_RE = re.compile(r"^\d{18,}$")
_HEX_TOKEN_ID_RE = re.compile(r"^(?:0x)?[0-9a-f]{40,}$")
_CONDITION_ID_RE = re.compile(r"^0x[0-9a-f]{64}$")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_id(value: Any) -> str:
    return str(value or "").strip().lower()


def _looks_like_token_id(value: Any) -> bool:
    normalized = _normalize_id(value)
    if not normalized:
        return False
    if _CONDITION_ID_RE.fullmatch(normalized):
        return False
    return bool(_NUMERIC_TOKEN_ID_RE.fullmatch(normalized) or _HEX_TOKEN_ID_RE.fullmatch(normalized))


def _safe_payload(signal: Any) -> dict[str, Any]:
    payload = getattr(signal, "payload_json", None)
    return payload if isinstance(payload, dict) else {}


def _live_context(signal: Any, payload: dict[str, Any]) -> dict[str, Any]:
    context = getattr(signal, "live_context", None)
    if isinstance(context, dict):
        return context
    from_payload = payload.get("live_market")
    if isinstance(from_payload, dict):
        return from_payload
    return {}


def _pick_token_id(signal: Any, *, payload: dict[str, Any], live_context: dict[str, Any]) -> str | None:
    direction = str(getattr(signal, "direction", "") or "").strip().lower()
    candidates: list[str] = []

    selected_token = _normalize_id(live_context.get("selected_token_id") or payload.get("selected_token_id"))
    if selected_token:
        candidates.append(selected_token)

    if direction == "buy_yes":
        candidates.extend(
            [
                _normalize_id(live_context.get("yes_token_id")),
                _normalize_id(payload.get("yes_token_id")),
            ]
        )
    elif direction == "buy_no":
        candidates.extend(
            [
                _normalize_id(live_context.get("no_token_id")),
                _normalize_id(payload.get("no_token_id")),
            ]
        )

    token_ids = live_context.get("token_ids")
    if not isinstance(token_ids, list):
        token_ids = payload.get("token_ids")
    if isinstance(token_ids, list):
        candidates.extend([_normalize_id(token_id) for token_id in token_ids])

    direct_payload_token = _normalize_id(payload.get("token_id") or payload.get("asset") or payload.get("assetId"))
    if direct_payload_token:
        candidates.append(direct_payload_token)

    market_id = _normalize_id(getattr(signal, "market_id", ""))
    if market_id:
        candidates.append(market_id)

    for candidate in candidates:
        if _looks_like_token_id(candidate):
            return candidate
    return None


async def submit_order(
    *,
    mode: str,
    signal: Any,
    size_usd: float,
) -> tuple[str, float | None, str | None, dict[str, Any]]:
    """Submit an order for paper/live orchestrator modes."""

    entry_price = _safe_float(getattr(signal, "entry_price", None), 0.0) or None
    mode_key = str(mode or "").strip().lower()

    if mode_key == "live":
        payload = _safe_payload(signal)
        live_context = _live_context(signal, payload)
        token_id = _pick_token_id(signal, payload=payload, live_context=live_context)
        if not token_id:
            return (
                "failed",
                entry_price,
                "No executable token_id resolved from signal payload/live context.",
                {
                    "mode": "live",
                    "submission": "rejected",
                    "reason": "missing_token_id",
                    "market_id": str(getattr(signal, "market_id", "") or ""),
                    "direction": str(getattr(signal, "direction", "") or ""),
                },
            )

        execution_price = _safe_float(live_context.get("live_selected_price"), entry_price or 0.0) or entry_price
        if execution_price is None or execution_price <= 0:
            return (
                "failed",
                None,
                "No valid execution price for live order.",
                {
                    "mode": "live",
                    "submission": "rejected",
                    "reason": "missing_price",
                    "token_id": token_id,
                },
            )

        shares = float(max(0.0, size_usd) / max(0.0001, execution_price))
        if shares <= 0:
            return (
                "failed",
                execution_price,
                "Computed live order size is zero.",
                {
                    "mode": "live",
                    "submission": "rejected",
                    "reason": "invalid_size",
                    "token_id": token_id,
                    "size_usd": float(max(0.0, size_usd)),
                    "execution_price": execution_price,
                },
            )

        side = "SELL" if str(getattr(signal, "direction", "") or "").strip().lower().startswith("sell_") else "BUY"
        execution = await execute_live_order(
            token_id=token_id,
            side=side,
            size=shares,
            fallback_price=execution_price,
            market_question=str(getattr(signal, "market_question", "") or ""),
        )
        return (
            execution.status,
            execution.effective_price,
            execution.error_message,
            {
                **execution.payload,
                "mode": "live",
                "size_usd": float(max(0.0, size_usd)),
                "shares": shares,
                "direction": str(getattr(signal, "direction", "") or ""),
                "market_id": str(getattr(signal, "market_id", "") or ""),
            },
        )

    return (
        "executed",
        entry_price,
        None,
        {
            "mode": "paper",
            "submission": "simulated",
            "filled_notional_usd": float(max(0.0, size_usd)),
        },
    )
