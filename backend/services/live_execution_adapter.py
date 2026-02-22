from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.polymarket import polymarket_client
from services.trading import OrderSide, OrderType, trading_service
from utils.converters import safe_float


def _normalize_side(value: Any) -> OrderSide | None:
    if isinstance(value, OrderSide):
        return value
    text = str(value or "").strip().upper()
    if text in {"BUY", "B"}:
        return OrderSide.BUY
    if text in {"SELL", "S"}:
        return OrderSide.SELL
    return None


def _map_trading_status(status: Any) -> str:
    key = str(getattr(status, "value", status) or "").strip().lower()
    if key == "filled":
        return "executed"
    if key in {"open", "partially_filled"}:
        return "open"
    if key == "pending":
        return "submitted"
    return "failed"


@dataclass
class LiveOrderExecution:
    status: str
    effective_price: float | None
    error_message: str | None
    payload: dict[str, Any]
    order_id: str | None = None


async def execute_live_order(
    *,
    token_id: str,
    side: Any,
    size: float,
    fallback_price: float | None = None,
    market_question: str | None = None,
    opportunity_id: str | None = None,
    time_in_force: str = "GTC",
    post_only: bool = False,
) -> LiveOrderExecution:
    normalized_token_id = str(token_id or "").strip()
    normalized_side = _normalize_side(side)
    requested_size = max(0.0, safe_float(size, 0.0) or 0.0)
    fallback = safe_float(fallback_price)

    base_payload = {
        "adapter": "live_execution_adapter_v1",
        "token_id": normalized_token_id,
        "side": str(getattr(normalized_side, "value", normalized_side) or ""),
        "requested_size": requested_size,
        "fallback_price": fallback,
    }

    if not normalized_token_id:
        return LiveOrderExecution(
            status="failed",
            effective_price=None,
            error_message="Missing token_id for live order.",
            payload={**base_payload, "submission": "rejected"},
        )
    if normalized_side is None:
        return LiveOrderExecution(
            status="failed",
            effective_price=None,
            error_message="Invalid side for live order.",
            payload={**base_payload, "submission": "rejected"},
        )
    if requested_size <= 0:
        return LiveOrderExecution(
            status="failed",
            effective_price=None,
            error_message="Order size must be greater than zero.",
            payload={**base_payload, "submission": "rejected"},
        )
    if not await trading_service.ensure_initialized():
        return LiveOrderExecution(
            status="failed",
            effective_price=None,
            error_message="Trading service is not initialized.",
            payload={**base_payload, "submission": "not_ready"},
        )

    resolved_price = fallback
    try:
        if normalized_side == OrderSide.BUY:
            resolved_price = safe_float(
                await polymarket_client.get_price(normalized_token_id, side="BUY"), resolved_price
            )
        else:
            resolved_price = safe_float(
                await polymarket_client.get_price(normalized_token_id, side="SELL"), resolved_price
            )
    except Exception:
        pass

    if resolved_price is None or resolved_price <= 0:
        return LiveOrderExecution(
            status="failed",
            effective_price=None,
            error_message="Could not resolve a valid live price.",
            payload={**base_payload, "submission": "rejected", "resolved_price": resolved_price},
        )

    try:
        try:
            order_type = OrderType(time_in_force.strip().upper())
        except ValueError:
            order_type = OrderType.GTC

        order = await trading_service.place_order(
            token_id=normalized_token_id,
            side=normalized_side,
            price=resolved_price,
            size=requested_size,
            order_type=order_type,
            post_only=post_only,
            market_question=market_question,
            opportunity_id=opportunity_id,
        )
    except Exception as exc:
        return LiveOrderExecution(
            status="failed",
            effective_price=resolved_price,
            error_message=str(exc),
            payload={
                **base_payload,
                "submission": "exception",
                "resolved_price": resolved_price,
            },
        )

    mapped_status = _map_trading_status(getattr(order, "status", None))
    error_message = getattr(order, "error_message", None) if mapped_status == "failed" else None
    average_fill = safe_float(getattr(order, "average_fill_price", None))
    effective_price = average_fill if average_fill and average_fill > 0 else resolved_price
    order_id = str(getattr(order, "id", "") or "") or None
    clob_order_id = str(getattr(order, "clob_order_id", "") or "") or None

    return LiveOrderExecution(
        status=mapped_status,
        effective_price=effective_price,
        error_message=error_message,
        order_id=order_id,
        payload={
            **base_payload,
            "submission": "live",
            "resolved_price": resolved_price,
            "order_id": order_id,
            "clob_order_id": clob_order_id,
            "trading_status": str(getattr(getattr(order, "status", None), "value", getattr(order, "status", "")) or ""),
            "filled_size": safe_float(getattr(order, "filled_size", None), 0.0) or 0.0,
            "average_fill_price": average_fill,
        },
    )
