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
    quote_aggressively: bool | None = None,
    enforce_fallback_bound: bool = False,
    max_execution_price: float | None = None,
    min_execution_price: float | None = None,
    allow_taker_limit_buy_above_signal: bool = False,
    aggressive_limit_buy_submit_as_gtc: bool = False,
    skip_buy_pre_submit_gate: bool = False,
    metadata: str | None = None,
) -> LiveOrderExecution:
    normalized_token_id = str(token_id or "").strip()
    normalized_side = _normalize_side(side)
    requested_size = max(0.0, safe_float(size, 0.0) or 0.0)
    fallback = safe_float(fallback_price)
    max_execution = safe_float(max_execution_price)
    min_execution = safe_float(min_execution_price)
    min_order_size = StrategySDK.resolve_min_order_size_usd(
        {"min_order_size_usd": min_order_size_usd} if min_order_size_usd is not None else {},
        fallback=1.0,
    )
    if normalized_side == OrderSide.BUY and max_execution is not None and max_execution > 0:
        if fallback is None or fallback <= 0:
            fallback = float(max_execution)
        else:
            fallback = min(float(fallback), float(max_execution))
    if normalized_side == OrderSide.SELL and min_execution is not None and min_execution > 0:
        if fallback is None or fallback <= 0:
            fallback = float(min_execution)
        else:
            fallback = max(float(fallback), float(min_execution))
    aggressive_quote = (
        bool(quote_aggressively)
        if quote_aggressively is not None
        else (not enforce_fallback_bound and not post_only)
    )

    base_payload = {
        "adapter": "live_execution_adapter_v1",
        "token_id": normalized_token_id,
        "side": str(getattr(normalized_side, "value", normalized_side) or ""),
        "requested_size": requested_size,
        "fallback_price": fallback,
        "post_only": bool(post_only),
        "quote_aggressively": bool(aggressive_quote),
        "enforce_fallback_bound": bool(enforce_fallback_bound),
        "max_execution_price": max_execution,
        "min_execution_price": min_execution,
        "allow_taker_limit_buy_above_signal": bool(allow_taker_limit_buy_above_signal),
        "aggressive_limit_buy_submit_as_gtc": bool(aggressive_limit_buy_submit_as_gtc),
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
    if live_execution_service.clob_auth_circuit_open():
        reason = live_execution_service.clob_auth_circuit_reason() or "Trading service authentication failed."
        return LiveOrderExecution(
            status="failed",
            effective_price=None,
            error_message=reason,
            payload={**base_payload, "submission": "auth_unavailable"},
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
                        is_taker = aggressive_quote and not post_only
                        if is_taker:
                            bid_ask = fm.cache.get_best_bid_ask(normalized_token_id)
                            if bid_ask is not None:
                                best_bid, best_ask = bid_ask
                                if normalized_side == OrderSide.BUY and best_ask > 0:
                                    cached = best_ask
                                elif normalized_side == OrderSide.SELL and best_bid > 0:
                                    cached = best_bid
                                else:
                                    cached = None
                            else:
                                cached = None
                        else:
                            cached = fm.cache.get_mid_price(normalized_token_id)
                        if cached is not None and cached > 0:
                            cached_notional = float(cached) * requested_size
                            if cached_notional + 1e-9 >= min_order_size:
                                live_quote = cached
                                price_resolution = "ws_cache_ask" if (is_taker and normalized_side == OrderSide.BUY) else "ws_cache_bid" if (is_taker and normalized_side == OrderSide.SELL) else "ws_cache"
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
                if normalized_side == OrderSide.BUY and max_execution is not None and max_execution > 0:
                    bounded_quote = min(float(live_quote), float(max_execution))
                    if abs(bounded_quote - float(live_quote)) >= 1e-9:
                        price_resolution = f"{price_resolution}|bounded_by_max_execution"
                    live_quote = bounded_quote
                elif normalized_side == OrderSide.SELL and min_execution is not None and min_execution > 0:
                    bounded_quote = max(float(live_quote), float(min_execution))
                    if abs(bounded_quote - float(live_quote)) >= 1e-9:
                        price_resolution = f"{price_resolution}|bounded_by_min_execution"
                    live_quote = bounded_quote
                should_apply_fallback_bound = (post_only or enforce_fallback_bound) and fallback is not None and fallback > 0
                if should_apply_fallback_bound:
                    skip_buy_fallback_bound = (
                        normalized_side == OrderSide.BUY
                        and bool(allow_taker_limit_buy_above_signal)
                        and bool(aggressive_quote)
                        and not post_only
                    )
                    if not skip_buy_fallback_bound:
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
    normalized_resolved_price = _clamp_binary_price(float(resolved_price))
    if abs(normalized_resolved_price - float(resolved_price)) >= 1e-9:
        resolved_price = normalized_resolved_price
        price_resolution = f"{price_resolution}|binary_price_clamped"

    try:
        try:
            order_type = OrderType(time_in_force.strip().upper())
        except ValueError:
            order_type = OrderType.GTC
        if (
            normalized_side == OrderSide.BUY
            and bool(aggressive_quote)
            and bool(aggressive_limit_buy_submit_as_gtc)
            and not post_only
            and order_type in {OrderType.IOC, OrderType.FAK, OrderType.FOK}
        ):
            order_type = OrderType.GTC
            price_resolution = f"{price_resolution}|aggressive_buy_submit_as_gtc"

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
            skip_buy_pre_submit_gate=skip_buy_pre_submit_gate,
            metadata=metadata,
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
                # Cancel the first order before retrying to prevent double-fills
                first_clob_id = str(getattr(order, "clob_order_id", "") or "").strip()
                if first_clob_id:
                    try:
                        await live_execution_service.cancel_order(first_clob_id)
                    except Exception:
                        logger.warning("Post-only retry: cancel of first order failed", clob_order_id=first_clob_id)
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
                    skip_buy_pre_submit_gate=skip_buy_pre_submit_gate,
                    metadata=metadata,
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
        auth_failure = live_execution_service.is_auth_failure_message(exc)
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
                "submission": "auth_unavailable" if auth_failure else "exception",
                "resolved_price": resolved_price,
                "price_resolution": price_resolution,
                "submitted_order_type": str(getattr(order_type, "value", order_type) or ""),
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
            "submitted_order_type": str(getattr(order_type, "value", order_type) or ""),
            "trading_status": str(getattr(getattr(order, "status", None), "value", getattr(order, "status", "")) or ""),
            "filled_size": safe_float(getattr(order, "filled_size", None), 0.0) or 0.0,
            "average_fill_price": average_fill,
            "provider_order_type_sent": str(getattr(order, "_provider_order_type_sent", "") or "") or None,
            "submit_method": str(getattr(order, "_submit_method", "") or "") or None,
            # Sub-stage breakdown stashed by ``place_order`` — surfaced
            # so the orchestrator's slow log can name which step (lock
            # wait vs venue round-trip vs balance gate) owned the time.
            "submit_breakdown": dict(getattr(order, "_submit_breakdown", {}) or {}),
        },
    )
