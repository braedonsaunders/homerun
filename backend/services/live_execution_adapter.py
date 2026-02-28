from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.polymarket import polymarket_client
from services.strategy_sdk import StrategySDK
from services.live_execution_service import OrderSide, OrderType, live_execution_service
from utils.converters import safe_float
from utils.logger import get_logger

logger = get_logger(__name__)


_POST_ONLY_REPRICE_TICK = 0.01


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


def _is_post_only_cross_reject(error_message: str | None) -> bool:
    text = str(error_message or "").strip().lower()
    if not text:
        return False
    return "post-only" in text and "crosses book" in text


def _clamp_binary_price(value: float) -> float:
    return max(_POST_ONLY_REPRICE_TICK, min(0.99, float(value)))


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
    min_order_size_usd: float | None = None,
    market_question: str | None = None,
    opportunity_id: str | None = None,
    time_in_force: str = "GTC",
    post_only: bool = False,
    resolve_live_price: bool = True,
    prefer_cached_price: bool = True,
) -> LiveOrderExecution:
    normalized_token_id = str(token_id or "").strip()
    normalized_side = _normalize_side(side)
    requested_size = max(0.0, safe_float(size, 0.0) or 0.0)
    fallback = safe_float(fallback_price)
    min_order_size = StrategySDK.resolve_min_order_size_usd(
        {"min_order_size_usd": min_order_size_usd} if min_order_size_usd is not None else {},
        fallback=1.0,
    )

    base_payload = {
        "adapter": "live_execution_adapter_v1",
        "token_id": normalized_token_id,
        "side": str(getattr(normalized_side, "value", normalized_side) or ""),
        "requested_size": requested_size,
        "fallback_price": fallback,
        "post_only": bool(post_only),
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
    if not await live_execution_service.ensure_initialized():
        return LiveOrderExecution(
            status="failed",
            effective_price=None,
            error_message="Trading service is not initialized.",
            payload={**base_payload, "submission": "not_ready"},
        )

    resolved_price = fallback
    price_resolution = "explicit_limit"
    if resolve_live_price:
        price_resolution = "fallback_price"
        try:
            live_quote = None

            # Fast path: try cached prices before HTTP
            if prefer_cached_price:
                # Try WS in-memory cache first (sub-10ms)
                try:
                    from services.ws_feeds import get_feed_manager
                    fm = get_feed_manager()
                    if fm.cache.is_fresh(normalized_token_id):
                        cached = fm.cache.get_mid_price(normalized_token_id)
                        if cached is not None and cached > 0:
                            cached_notional = float(cached) * requested_size
                            if cached_notional + 1e-9 >= min_order_size:
                                live_quote = cached
                                price_resolution = "ws_cache"
                except Exception:
                    pass

                # Try Redis cross-process cache (~5ms)
                if live_quote is None:
                    try:
                        from services.redis_price_cache import redis_price_cache
                        redis_result = await redis_price_cache.read_prices([normalized_token_id])
                        redis_entry = redis_result.get(normalized_token_id)
                        if redis_entry is not None:
                            mid = safe_float(redis_entry.get("mid"))
                            if mid is not None and mid > 0:
                                redis_notional = float(mid) * requested_size
                                if redis_notional + 1e-9 >= min_order_size:
                                    live_quote = mid
                                    price_resolution = "redis_cache"
                    except Exception:
                        pass

            # Slow path: HTTP API calls (50-500ms)
            if live_quote is None:
                quote_buy = None
                quote_sell = None
                try:
                    quote_buy = safe_float(await polymarket_client.get_price(normalized_token_id, side="BUY"))
                except Exception:
                    quote_buy = None
                try:
                    quote_sell = safe_float(await polymarket_client.get_price(normalized_token_id, side="SELL"))
                except Exception:
                    quote_sell = None

                quote_candidates = [q for q in (quote_buy, quote_sell) if q is not None and q > 0]
                if quote_candidates:
                    if post_only:
                        if normalized_side == OrderSide.BUY:
                            if quote_buy is not None and quote_buy > 0:
                                live_quote = quote_buy
                                price_resolution = "live_quote_post_only_bid"
                            elif quote_sell is not None and quote_sell > 0:
                                live_quote = _clamp_binary_price(float(quote_sell) - _POST_ONLY_REPRICE_TICK)
                                price_resolution = "live_quote_post_only_from_ask"
                        else:
                            if quote_sell is not None and quote_sell > 0:
                                live_quote = quote_sell
                                price_resolution = "live_quote_post_only_ask"
                            elif quote_buy is not None and quote_buy > 0:
                                live_quote = _clamp_binary_price(float(quote_buy) + _POST_ONLY_REPRICE_TICK)
                                price_resolution = "live_quote_post_only_from_bid"
                    if live_quote is None:
                        live_quote = max(quote_candidates) if normalized_side == OrderSide.BUY else min(quote_candidates)
                        price_resolution = "live_quote"

            # Apply the resolved price with min notional guard
            if live_quote is not None and live_quote > 0:
                if post_only and fallback is not None and fallback > 0:
                    if normalized_side == OrderSide.BUY:
                        bounded_quote = min(float(live_quote), float(fallback))
                    else:
                        bounded_quote = max(float(live_quote), float(fallback))
                    if abs(bounded_quote - float(live_quote)) >= 1e-9:
                        price_resolution = f"{price_resolution}|bounded_by_fallback"
                    live_quote = _clamp_binary_price(bounded_quote)
                live_notional = float(live_quote) * requested_size
                fallback_notional = (float(fallback) * requested_size) if fallback is not None and fallback > 0 else 0.0
                if (
                    live_notional + 1e-9 < min_order_size
                    and fallback is not None
                    and fallback > 0
                    and fallback_notional + 1e-9 >= min_order_size
                ):
                    resolved_price = fallback
                    price_resolution = "fallback_min_notional_guard"
                else:
                    resolved_price = live_quote
        except Exception as exc:
            logger.warning(
                "Live quote resolution failed; using fallback price",
                token_id=normalized_token_id,
                side=str(normalized_side.value),
                fallback_price=fallback,
                exc_info=exc,
            )

    if resolved_price is None or resolved_price <= 0:
        return LiveOrderExecution(
            status="failed",
            effective_price=None,
            error_message="Could not resolve a valid live price.",
            payload={
                **base_payload,
                "submission": "rejected",
                "resolved_price": resolved_price,
                "price_resolution": price_resolution,
            },
        )

    try:
        try:
            order_type = OrderType(time_in_force.strip().upper())
        except ValueError:
            order_type = OrderType.GTC

        order = await live_execution_service.place_order(
            token_id=normalized_token_id,
            side=normalized_side,
            price=resolved_price,
            size=requested_size,
            order_type=order_type,
            post_only=post_only,
            min_order_size_usd=min_order_size,
            market_question=market_question,
            opportunity_id=opportunity_id,
        )

        order_status = _map_trading_status(getattr(order, "status", None))
        order_error_message = getattr(order, "error_message", None) if order_status == "failed" else None
        if post_only and order_status == "failed" and _is_post_only_cross_reject(order_error_message):
            retry_price = _clamp_binary_price(
                resolved_price - _POST_ONLY_REPRICE_TICK
                if normalized_side == OrderSide.BUY
                else resolved_price + _POST_ONLY_REPRICE_TICK
            )
            if abs(retry_price - resolved_price) >= 1e-9:
                retry_order = await live_execution_service.place_order(
                    token_id=normalized_token_id,
                    side=normalized_side,
                    price=retry_price,
                    size=requested_size,
                    order_type=order_type,
                    post_only=post_only,
                    min_order_size_usd=min_order_size,
                    market_question=market_question,
                    opportunity_id=opportunity_id,
                )
                retry_status = _map_trading_status(getattr(retry_order, "status", None))
                if retry_status != "failed":
                    order = retry_order
                    resolved_price = retry_price
                    price_resolution = f"{price_resolution}|post_only_retry_1tick"
                else:
                    retry_error_message = getattr(retry_order, "error_message", None)
                    if retry_error_message:
                        order.error_message = (
                            f"{str(order.error_message or '')} | post_only_retry_price={retry_price:.4f} failed: {retry_error_message}"
                        ).strip(" |")
    except Exception as exc:
        logger.error(
            "Live order placement failed",
            token_id=normalized_token_id,
            side=str(normalized_side.value),
            resolved_price=resolved_price,
            requested_size=requested_size,
            exc_info=exc,
        )
        return LiveOrderExecution(
            status="failed",
            effective_price=resolved_price,
            error_message=str(exc),
            payload={
                **base_payload,
                "submission": "exception",
                "resolved_price": resolved_price,
                "price_resolution": price_resolution,
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
            "price_resolution": price_resolution,
            "order_id": order_id,
            "clob_order_id": clob_order_id,
            "trading_status": str(getattr(getattr(order, "status", None), "value", getattr(order, "status", "")) or ""),
            "filled_size": safe_float(getattr(order, "filled_size", None), 0.0) or 0.0,
            "average_fill_price": average_fill,
        },
    )
