from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

from services.live_execution_adapter import execute_live_order
from services.polymarket import polymarket_client
from services.live_execution_service import live_execution_service
from services.trader_orchestrator.hot_path import allow_polymarket_rest_call
from services.optimization.execution_estimator import (
    ExecutionEstimate,
    ExecutionEstimator,
    ExecutionEstimatorConfig,
)
from services.fill_simulator import build_survival_features
from services.strategy_sdk import StrategySDK
from utils.converters import safe_float


_MIN_EXECUTION_PRICE = 0.001
_MIN_LIVE_SHARES = 5.0
_LEG_SUBMIT_TIMEOUT_SECONDS = 35.0
_NUMERIC_TOKEN_ID_RE = re.compile(r"^\d{18,}$")
_HEX_TOKEN_ID_RE = re.compile(r"^(?:0x)?[0-9a-f]{40,}$")
_CONDITION_ID_RE = re.compile(r"^0x[0-9a-f]{64}$")
_execution_estimator = ExecutionEstimator()


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


def _clob_metadata_from_leg(leg: dict[str, Any]) -> str | None:
    """Extract the CLOB ``OrderArgsV2.metadata`` value from a leg dict.

    The venue metadata key lives in ``leg["clob_idempotency_key"]`` —
    a dedicated field separate from ``leg["metadata"]`` (which carries
    ExecutionPlan bookkeeping consumed by readers like
    ``_resolve_execution_price_bounds``). The two used to share the
    same key, which produced a recurring production crash where a
    bookkeeping dict got stringified into ``bytes.fromhex`` deep in
    the CLOB SDK; splitting them out is the structural fix.

    For backward compatibility we still accept a string-shaped
    ``leg["metadata"]`` so any in-flight legs constructed under the
    old shape continue to submit with their correct idempotency key
    on retry. Dict-shaped metadata is ignored (it was always
    bookkeeping).
    """
    primary = leg.get("clob_idempotency_key")
    if isinstance(primary, str):
        text = primary.strip()
        if text:
            return text
    legacy = leg.get("metadata")
    if isinstance(legacy, str):
        text = legacy.strip()
        if text:
            return text
    return None


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

    outcome = str(leg.get("outcome") or "").strip().lower()
    side = str(leg.get("side") or "buy").strip().lower()
    if outcome == "yes":
        _append("live_context.yes_token_id", live_context.get("yes_token_id"))
        _append("payload.yes_token_id", payload.get("yes_token_id"))
    elif outcome == "no":
        _append("live_context.no_token_id", live_context.get("no_token_id"))
        _append("payload.no_token_id", payload.get("no_token_id"))

    if side == "buy":
        _append("live_context.selected_token_id", live_context.get("selected_token_id"))
        _append("payload.selected_token_id", payload.get("selected_token_id"))

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


def _resolve_live_price_for_leg(leg: dict[str, Any], live_context: dict[str, Any]) -> float | None:
    if not isinstance(live_context, dict):
        return None

    selected_price = safe_float(live_context.get("live_selected_price"), None)
    yes_price = safe_float(live_context.get("live_yes_price"), None)
    no_price = safe_float(live_context.get("live_no_price"), None)

    def _valid(price: float | None) -> bool:
        return price is not None and price > 0

    leg_token = _normalize_id(leg.get("token_id"))
    selected_token = _normalize_id(live_context.get("selected_token_id"))
    yes_token = _normalize_id(live_context.get("yes_token_id"))
    no_token = _normalize_id(live_context.get("no_token_id"))
    token_ids = live_context.get("token_ids")
    context_token_ids = {
        _normalize_id(token)
        for token in (token_ids if isinstance(token_ids, list) else [])
        if _normalize_id(token)
    }
    leg_market_id = _normalize_id(leg.get("market_id"))
    context_market_id = _normalize_id(live_context.get("market_id"))
    context_condition_id = _normalize_id(live_context.get("condition_id"))
    outcome = str(leg.get("outcome") or "").strip().lower()
    selected_outcome = str(live_context.get("selected_outcome") or "").strip().lower()

    if leg_token:
        if selected_token and leg_token == selected_token and _valid(selected_price):
            return selected_price
        if yes_token and leg_token == yes_token and _valid(yes_price):
            return yes_price
        if no_token and leg_token == no_token and _valid(no_price):
            return no_price
        if leg_token not in context_token_ids:
            return None

    if leg_market_id and context_market_id and leg_market_id != context_market_id and leg_market_id != context_condition_id:
        return None

    if outcome == "yes":
        if _valid(yes_price):
            return yes_price
        if selected_outcome == "yes" and _valid(selected_price):
            return selected_price
    if outcome == "no":
        if _valid(no_price):
            return no_price
        if selected_outcome == "no" and _valid(selected_price):
            return selected_price

    if not outcome and not leg_token and _valid(selected_price):
        return selected_price
    return None


def _resolve_leg_price(leg: dict[str, Any], signal: Any, live_context: dict[str, Any]) -> float | None:
    live_price = _resolve_live_price_for_leg(leg, live_context)
    if live_price is not None and live_price > 0:
        return live_price

    limit_price = safe_float(leg.get("limit_price"), None)
    if limit_price is not None and limit_price > 0:
        return limit_price

    signal_price = safe_float(getattr(signal, "entry_price", None), None)
    if signal_price is not None and signal_price > 0:
        return signal_price
    return None


def _valid_execution_bound(value: Any) -> float | None:
    bound = safe_float(value, None)
    if bound is None or bound <= 0.0 or bound > 1.0:
        return None
    return float(bound)


def _derive_min_upside_price_cap(min_upside_percent: Any) -> float | None:
    upside = safe_float(min_upside_percent, None)
    if upside is None or upside <= 0.0:
        return None
    return _valid_execution_bound(100.0 / (100.0 + float(upside)))


def _allow_taker_limit_buy_above_signal(strategy_params: dict[str, Any] | None) -> bool:
    return StrategySDK.allow_taker_limit_buy_above_signal_price(strategy_params or {}, default=False)


def _aggressive_limit_buy_submit_as_gtc(strategy_params: dict[str, Any] | None) -> bool:
    return StrategySDK.aggressive_limit_buy_submit_as_gtc(strategy_params or {}, default=False)


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _resolve_leg_execution_bool(
    *,
    leg: dict[str, Any],
    key: str,
    strategy_default: bool,
) -> bool:
    leg_value = _coerce_optional_bool(leg.get(key))
    if leg_value is not None:
        return leg_value
    metadata = leg.get("metadata")
    if isinstance(metadata, dict):
        metadata_value = _coerce_optional_bool(metadata.get(key))
        if metadata_value is not None:
            return metadata_value
    return strategy_default


def _resolve_execution_price_bounds(
    *,
    leg: dict[str, Any],
    strategy_params: dict[str, Any],
    fallback_price: float | None,
    allow_taker_limit_buy_above_signal: bool = False,
) -> tuple[float | None, float | None]:
    side_key = str(leg.get("side") or "buy").strip().lower()
    price_policy = str(leg.get("price_policy") or "").strip().lower()
    metadata = leg.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    fallback_bound = _valid_execution_bound(fallback_price)

    if side_key == "buy":
        candidates = [
            _valid_execution_bound(leg.get("max_execution_price")),
            _valid_execution_bound(metadata.get("max_execution_price")),
            _valid_execution_bound(strategy_params.get("max_execution_price")),
            _valid_execution_bound(strategy_params.get("max_entry_price")),
            _valid_execution_bound(strategy_params.get("max_probability")),
            _derive_min_upside_price_cap(strategy_params.get("min_upside_percent")),
        ]
        has_explicit_cap = any(candidate is not None for candidate in candidates)
        if price_policy == "taker_limit" and fallback_bound is not None:
            if allow_taker_limit_buy_above_signal:
                if not has_explicit_cap:
                    candidates.append(fallback_bound)
            else:
                candidates.append(fallback_bound)
        resolved = min((candidate for candidate in candidates if candidate is not None), default=None)
        return resolved, None

    if side_key == "sell":
        candidates = [
            _valid_execution_bound(leg.get("min_execution_price")),
            _valid_execution_bound(metadata.get("min_execution_price")),
            _valid_execution_bound(strategy_params.get("min_execution_price")),
            _valid_execution_bound(strategy_params.get("min_exit_price")),
            _valid_execution_bound(strategy_params.get("min_sell_price")),
        ]
        if price_policy == "taker_limit" and fallback_bound is not None:
            candidates.append(fallback_bound)
        resolved = max((candidate for candidate in candidates if candidate is not None), default=None)
        return None, resolved

    return None, None


def _order_book_payload(order_book: Any) -> dict[str, Any] | None:
    if order_book is None:
        return None
    if isinstance(order_book, dict):
        bids = order_book.get("bids")
        asks = order_book.get("asks")
        if isinstance(bids, list) and isinstance(asks, list):
            return {"bids": bids, "asks": asks}
        return None

    def _levels(side_name: str) -> list[dict[str, float]]:
        levels = []
        for level in list(getattr(order_book, side_name, []) or []):
            price = safe_float(getattr(level, "price", None), None)
            size = safe_float(getattr(level, "size", None), None)
            if price is not None and size is not None and price > 0 and size > 0:
                levels.append({"price": float(price), "size": float(size)})
        return levels

    return {"bids": _levels("bids"), "asks": _levels("asks")}


def _trades_payload(trades: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trade in trades:
        if isinstance(trade, dict):
            price = safe_float(trade.get("price"), None)
            size = safe_float(trade.get("size"), None)
            side = str(trade.get("side") or "").strip().upper()
            timestamp = safe_float(trade.get("timestamp"), None)
        else:
            price = safe_float(getattr(trade, "price", None), None)
            size = safe_float(getattr(trade, "size", None), None)
            side = str(getattr(trade, "side", "") or "").strip().upper()
            timestamp = safe_float(getattr(trade, "timestamp", None), None)
        if price is None or size is None or timestamp is None or price <= 0 or size <= 0:
            continue
        rows.append(
            {
                "price": float(price),
                "size": float(size),
                "side": side if side in {"BUY", "SELL"} else "BUY",
                "timestamp": float(timestamp),
            }
        )
    return rows


def _context_order_book(live_context: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("execution_order_book", "order_book", "book"):
        payload = live_context.get(key)
        normalized = _order_book_payload(payload)
        if normalized is not None:
            return normalized
    return None


def _context_trades(live_context: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("execution_recent_trades", "recent_trades", "trades"):
        raw = live_context.get(key)
        if isinstance(raw, list):
            return _trades_payload(raw)
    return []


async def _resolve_shadow_book_and_tape(
    *,
    token_id: str | None,
    live_context: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], float | None, str, str | None]:
    context_book = _context_order_book(live_context)
    context_trades = _context_trades(live_context)
    context_age_ms = safe_float(live_context.get("execution_order_book_age_ms"), None)
    if context_book is not None:
        return context_book, context_trades, context_age_ms, "signal_microstructure_context", None

    if not token_id:
        return None, [], None, "missing_token_id", None

    try:
        from services.ws_feeds import get_feed_manager

        feed_manager = get_feed_manager()
        book = await feed_manager.get_order_book(token_id)
        book_payload = _order_book_payload(book)
        trades = _trades_payload(feed_manager.cache.get_recent_trades(token_id, max_trades=200))
        staleness = feed_manager.cache.staleness(token_id)
        book_age_ms = staleness * 1000.0 if staleness is not None else None
        if book_payload is not None:
            return book_payload, trades, book_age_ms, "ws_order_book", None
        return None, trades, book_age_ms, "ws_order_book_missing", None
    except Exception as exc:
        return None, [], None, "ws_order_book_error", repr(exc)


def _shadow_status_for_estimate(estimate: ExecutionEstimate) -> str:
    if estimate.filled_shares <= 0:
        return "skipped"
    return "executed"


def _resolve_condition_id_for_leg(
    *,
    leg: dict[str, Any],
    payload: dict[str, Any],
    live_context: dict[str, Any],
) -> str | None:
    metadata = leg.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    strategy_context = payload.get("strategy_context")
    strategy_context = strategy_context if isinstance(strategy_context, dict) else {}

    candidates = (
        leg.get("condition_id"),
        metadata.get("condition_id"),
        live_context.get("condition_id"),
        strategy_context.get("condition_id"),
        payload.get("condition_id"),
        leg.get("market_id"),
    )
    for raw in candidates:
        normalized = _normalize_id(raw)
        if normalized and _CONDITION_ID_RE.fullmatch(normalized):
            return normalized
    return None


async def _fetch_token_id_from_market(market_id: str, outcome: str) -> str | None:
    """Live fallback: query the Polymarket CLOB/Gamma API for token IDs by market condition_id.

    Gated by ``allow_polymarket_rest_call`` — if the orchestrator hot
    path reaches this function the call short-circuits.  Strategies are
    supposed to provide token_ids in the signal payload; we should
    never hit this path with a healthy emit pipeline.
    """
    condition_id = market_id.strip()
    if not condition_id:
        return None
    if not allow_polymarket_rest_call("token_id_lookup"):
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
    strategy_params: dict[str, Any] | None = None,
) -> LegSubmitResult:
    mode_key = str(mode or "").strip().lower()
    if mode_key not in {"live", "shadow"}:
        return LegSubmitResult(
            leg_id=str(leg.get("leg_id") or "leg"),
            status="failed",
            effective_price=None,
            error_message=f"Unsupported execution mode '{mode_key or 'unknown'}'.",
            payload={"mode": mode_key or "unknown", "submission": "rejected", "reason": "unsupported_mode"},
            shares=None,
            notional_usd=float(max(0.0, notional_usd)),
        )
    leg_id = str(leg.get("leg_id") or "leg")
    notional = float(max(0.0, notional_usd))
    payload = _safe_signal_payload(signal)
    live_context = _safe_live_context(signal, payload)
    metadata = leg.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    side_key = str(leg.get("side") or "buy").strip().lower()
    ctf_action = str(metadata.get("ctf_action") or side_key).strip().lower()
    order_side = "SELL" if side_key == "sell" else "BUY"

    if ctf_action in {"split", "merge", "redeem"}:
        condition_id = _resolve_condition_id_for_leg(leg=leg, payload=payload, live_context=live_context)
        if not condition_id:
            return LegSubmitResult(
                leg_id=leg_id,
                status="failed",
                effective_price=None,
                error_message="Missing condition_id for CTF execution leg.",
                payload={
                    "mode": mode_key,
                    "leg": dict(leg),
                    "reason": "missing_condition_id",
                    "ctf_action": ctf_action,
                },
                shares=None,
                notional_usd=notional,
            )

        if mode_key != "live":
            return LegSubmitResult(
                leg_id=leg_id,
                status="executed",
                effective_price=None,
                error_message=None,
                payload={
                    "mode": mode_key,
                    "submission": "shadow_ctf_simulated",
                    "ctf_action": ctf_action,
                    "condition_id": condition_id,
                    "requested_notional_usd": notional,
                    "leg": dict(leg),
                },
                shares=None,
                notional_usd=notional,
            )

        from services.ctf_execution import ctf_execution_service

        if ctf_action == "split":
            amount_usd = max(
                0.0,
                safe_float(
                    metadata.get("amount_usd", leg.get("amount_usd")),
                    notional,
                )
                or 0.0,
            )
            ctf_result = await ctf_execution_service.split_position(
                condition_id=condition_id,
                amount_usd=amount_usd,
            )
            result_notional = amount_usd
        elif ctf_action == "merge":
            shares_per_side = max(
                0.0,
                safe_float(
                    metadata.get("shares_per_side", leg.get("shares_per_side")),
                    notional,
                )
                or 0.0,
            )
            ctf_result = await ctf_execution_service.merge_positions(
                condition_id=condition_id,
                shares_per_side=shares_per_side,
            )
            result_notional = shares_per_side
        else:
            raw_index_sets = metadata.get("index_sets", leg.get("index_sets"))
            index_sets: list[int] = []
            if isinstance(raw_index_sets, list):
                for value in raw_index_sets:
                    parsed = safe_float(value, None)
                    if parsed is None:
                        continue
                    as_int = int(parsed)
                    if as_int > 0:
                        index_sets.append(as_int)
            ctf_result = await ctf_execution_service.redeem_positions(
                condition_id=condition_id,
                index_sets=index_sets or None,
            )
            result_notional = notional

        normalized_status = "executed" if ctf_result.status == "executed" else "failed"
        return LegSubmitResult(
            leg_id=leg_id,
            status=normalized_status,
            effective_price=None,
            error_message=ctf_result.error_message,
            payload={
                "mode": mode_key,
                "submission": "ctf_execution",
                "ctf_action": ctf_action,
                "condition_id": condition_id,
                "tx_hash": ctf_result.tx_hash,
                "payload": dict(ctf_result.payload or {}),
                "leg": dict(leg),
            },
            provider_order_id=ctf_result.tx_hash,
            provider_clob_order_id=None,
            shares=None,
            notional_usd=result_notional,
        )

    price = _resolve_leg_price(leg, signal, live_context)

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

    requested_shares = notional / price
    if requested_shares <= 0:
        return LegSubmitResult(
            leg_id=leg_id,
            status="failed",
            effective_price=price,
            error_message="Computed leg size is zero.",
            payload={"mode": mode_key, "leg": dict(leg), "reason": "invalid_size"},
            shares=requested_shares,
            notional_usd=notional,
        )
    shares = requested_shares
    if shares < _MIN_LIVE_SHARES:
        shares = _MIN_LIVE_SHARES
    effective_notional = shares * price

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
        if mode_key == "live":
            return LegSubmitResult(
                leg_id=leg_id,
                status="failed",
                effective_price=price,
                error_message="No executable token_id resolved for execution leg.",
                payload={
                    "mode": mode_key,
                    "submission": "rejected",
                    "reason": "missing_token_id",
                    "token_resolution_attempts": token_attempts,
                    "leg": dict(leg),
                },
                shares=shares,
                notional_usd=notional,
            )

    skip_buy_pre_submit_gate = False
    if mode_key == "live" and order_side == "BUY":
        buy_gate_ok, buy_gate_error = await live_execution_service.check_buy_pre_submit_gate(
            token_id=token_id,
            required_notional_usd=effective_notional,
        )
        if not buy_gate_ok:
            return LegSubmitResult(
                leg_id=leg_id,
                status="skipped",
                effective_price=price,
                error_message=buy_gate_error or "BUY pre-submit gate failed.",
                payload={
                    "mode": mode_key,
                    "submission": "skipped",
                    "reason": "buy_pre_submit_gate",
                    "token_id": token_id,
                    "token_id_source": token_source,
                    "token_resolution_attempts": token_attempts,
                    "leg": dict(leg),
                    "shares": shares,
                    "requested_shares": requested_shares,
                    "min_live_shares": _MIN_LIVE_SHARES,
                    "requested_notional_usd": notional,
                    "effective_notional_usd": effective_notional,
                },
                shares=shares,
                notional_usd=effective_notional,
            )
        skip_buy_pre_submit_gate = True

    time_in_force = str(leg.get("time_in_force") or "GTC").strip().upper()
    post_only = bool(leg.get("post_only", False))
    params = dict(strategy_params or {})
    allow_taker_limit_buy_above_signal = _resolve_leg_execution_bool(
        leg=leg,
        key="allow_taker_limit_buy_above_signal",
        strategy_default=_allow_taker_limit_buy_above_signal(params),
    )
    aggressive_limit_buy_submit_as_gtc = _resolve_leg_execution_bool(
        leg=leg,
        key="aggressive_limit_buy_submit_as_gtc",
        strategy_default=_aggressive_limit_buy_submit_as_gtc(params),
    )

    if mode_key == "shadow":
        price_policy = str(leg.get("price_policy") or "").strip().lower()
        order_type = "taker_limit" if price_policy == "taker_limit" else "maker_limit"
        book_payload, recent_trades, book_age_ms, quote_source, quote_error = await _resolve_shadow_book_and_tape(
            token_id=token_id,
            live_context=live_context,
        )
        payload_mode = "shadow"
        submission_label = "shadow_microstructure_simulated"
        if book_payload is None:
            return LegSubmitResult(
                leg_id=leg_id,
                status="skipped",
                effective_price=price,
                error_message="No order book available for shadow execution leg.",
                payload={
                    "mode": payload_mode,
                    "submission": "skipped",
                    "reason": "missing_order_book",
                    "token_id": token_id,
                    "token_id_source": token_source,
                    "token_resolution_attempts": token_attempts,
                    "quote_source": quote_source,
                    "quote_error": quote_error,
                    "leg": dict(leg),
                    "shares": shares,
                    "requested_shares": requested_shares,
                    "min_live_shares": _MIN_LIVE_SHARES,
                    "requested_notional_usd": notional,
                    "effective_notional_usd": 0.0,
                },
                shares=shares,
                notional_usd=0.0,
            )

        estimate = _execution_estimator.estimate_order(
            order_book=book_payload,
            side=order_side,
            size_shares=shares,
            limit_price=price,
            order_type=order_type,
            recent_trades=recent_trades,
            book_age_ms=book_age_ms,
            config=ExecutionEstimatorConfig(
                fee_bps=0.0,
                latency_ms=350.0,
                time_in_force_seconds=6.0,
                displayed_depth_factor=0.88,
                min_depth_factor=0.20,
                max_book_age_ms=10_000.0,
                stale_depth_decay=0.55,
                maker_queue_ahead_fraction=0.65,
                maker_trade_flow_multiplier=1.20,
                adverse_selection_multiplier=0.70,
            ),
        )
        quote_price = estimate.average_price
        effective_shadow_notional = estimate.filled_notional_usd
        shadow_status = _shadow_status_for_estimate(estimate)
        survival_features = build_survival_features(
            estimate=estimate,
            order_book=book_payload,
            recent_trades=recent_trades,
            book_age_ms=book_age_ms,
            payload=payload,
            side=order_side,
            limit_price=float(price or 0.0),
            notional_usd=float(notional or 0.0),
            latency_p95_ms=None,
            recent_trade_lookback_seconds=30.0,
        )
        shadow_simulation_payload = {
            "filled": estimate.filled_shares > 0,
            "fill_ratio": estimate.fill_ratio,
            "estimated_fee_usd": estimate.fees_usd,
            "slippage_usd": abs(estimate.slippage_bps) / 10_000.0 * effective_shadow_notional,
            "slippage_bps": estimate.slippage_bps,
            "price_impact_bps": estimate.price_impact_bps,
            "adverse_selection_bps": estimate.adverse_selection_bps,
            "adverse_selection_cost_usd": estimate.adverse_selection_cost_usd,
            "fill_probability": estimate.fill_probability,
            "queue_ahead_shares": estimate.queue_ahead_shares,
            "levels_consumed": estimate.levels_consumed,
            "execution_estimate": estimate.to_dict(),
            "survival_features": survival_features.to_payload(),
        }
        if estimate.filled_shares <= 0:
            return LegSubmitResult(
                leg_id=leg_id,
                status=shadow_status,
                effective_price=price,
                error_message=f"Shadow execution did not fill: {estimate.reason}.",
                payload={
                    "mode": payload_mode,
                    "submission": "skipped",
                    "reason": estimate.reason,
                    "token_id": token_id,
                    "token_id_source": token_source,
                    "token_resolution_attempts": token_attempts,
                    "quote_source": quote_source,
                    "quote_error": quote_error,
                    "leg": dict(leg),
                    "shares": shares,
                    "filled_size": 0.0,
                    "average_fill_price": None,
                    "filled_notional_usd": 0.0,
                    "requested_shares": requested_shares,
                    "min_live_shares": _MIN_LIVE_SHARES,
                    "requested_notional_usd": notional,
                    "effective_notional_usd": 0.0,
                    "time_in_force": time_in_force,
                    "post_only": post_only,
                    "shadow_simulation": shadow_simulation_payload,
                },
                provider_order_id=None,
                provider_clob_order_id=None,
                shares=0.0,
                notional_usd=0.0,
            )

        return LegSubmitResult(
            leg_id=leg_id,
            status=shadow_status,
            effective_price=quote_price,
            error_message=None,
            payload={
                "mode": payload_mode,
                "submission": submission_label,
                "token_id": token_id,
                "token_id_source": token_source,
                "token_resolution_attempts": token_attempts,
                "quote_source": quote_source,
                "quote_error": quote_error,
                "quote_price": quote_price,
                "leg": dict(leg),
                "shares": shares,
                "filled_size": estimate.filled_shares,
                "average_fill_price": quote_price,
                "filled_notional_usd": effective_shadow_notional,
                "requested_shares": requested_shares,
                "min_live_shares": _MIN_LIVE_SHARES,
                "requested_notional_usd": notional,
                "effective_notional_usd": effective_shadow_notional,
                "time_in_force": time_in_force,
                "post_only": post_only,
                "shadow_simulation": shadow_simulation_payload,
            },
            provider_order_id=None,
            provider_clob_order_id=None,
            shares=estimate.filled_shares,
            notional_usd=effective_shadow_notional,
        )

    price_policy = str(leg.get("price_policy") or "").strip().lower()
    enforce_fallback = price_policy != "taker_limit"
    quote_aggressively = price_policy == "taker_limit"
    max_execution_price, min_execution_price = _resolve_execution_price_bounds(
        leg=leg,
        strategy_params=params,
        fallback_price=price,
        allow_taker_limit_buy_above_signal=allow_taker_limit_buy_above_signal,
    )

    execution = await execute_live_order(
        token_id=token_id,
        side=order_side,
        size=shares,
        fallback_price=price,
        market_question=str(leg.get("market_question") or getattr(signal, "market_question", "") or ""),
        opportunity_id=str(getattr(signal, "id", "") or ""),
        time_in_force=time_in_force,
        post_only=post_only,
        quote_aggressively=quote_aggressively,
        enforce_fallback_bound=enforce_fallback,
        max_execution_price=max_execution_price,
        min_execution_price=min_execution_price,
        allow_taker_limit_buy_above_signal=allow_taker_limit_buy_above_signal,
        aggressive_limit_buy_submit_as_gtc=aggressive_limit_buy_submit_as_gtc,
        skip_buy_pre_submit_gate=skip_buy_pre_submit_gate,
        metadata=_clob_metadata_from_leg(leg),
    )

    execution_error_text = str(execution.error_message or "").lower()
    if (
        execution.status == "failed"
        and "orderbook" in execution_error_text
        and "does not exist" in execution_error_text
    ):
        market_id_for_lookup = str(leg.get("market_id") or "").strip()
        outcome_for_lookup = str(leg.get("outcome") or "").strip()
        if market_id_for_lookup and outcome_for_lookup:
            fallback_token_id = await _fetch_token_id_from_market(market_id_for_lookup, outcome_for_lookup)
            if fallback_token_id and fallback_token_id != token_id:
                retry_execution = await execute_live_order(
                    token_id=fallback_token_id,
                    side=order_side,
                    size=shares,
                    fallback_price=price,
                    market_question=str(leg.get("market_question") or getattr(signal, "market_question", "") or ""),
                    opportunity_id=str(getattr(signal, "id", "") or ""),
                    time_in_force=time_in_force,
                    post_only=post_only,
                    quote_aggressively=quote_aggressively,
                    enforce_fallback_bound=enforce_fallback,
                    max_execution_price=max_execution_price,
                    min_execution_price=min_execution_price,
                    allow_taker_limit_buy_above_signal=allow_taker_limit_buy_above_signal,
                    aggressive_limit_buy_submit_as_gtc=aggressive_limit_buy_submit_as_gtc,
                    skip_buy_pre_submit_gate=skip_buy_pre_submit_gate,
                )
                if retry_execution.status != "failed":
                    execution = retry_execution
                    token_id = fallback_token_id
                    token_source = "polymarket_api_retry"
                else:
                    retry_error_text = str(retry_execution.error_message or "")
                    execution = retry_execution
                    execution.error_message = (
                        f"{str(execution.error_message or '')} | retry_token={fallback_token_id} failed: {retry_error_text}"
                    ).strip(" |")

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
            "requested_shares": requested_shares,
            "min_live_shares": _MIN_LIVE_SHARES,
            "requested_notional_usd": notional,
            "effective_notional_usd": effective_notional,
        },
        provider_order_id=execution.order_id,
        provider_clob_order_id=str(execution.payload.get("clob_order_id") or "").strip() or None,
        shares=shares,
        notional_usd=effective_notional,
    )


async def submit_execution_wave(
    *,
    mode: str,
    signal: Any,
    legs_with_notionals: list[tuple[dict[str, Any], float]],
    strategy_params: dict[str, Any] | None = None,
) -> list[LegSubmitResult]:
    if not legs_with_notionals:
        return []
    tasks = [
        asyncio.wait_for(
            submit_execution_leg(
                mode=mode,
                signal=signal,
                leg=leg,
                notional_usd=notional,
                strategy_params=strategy_params,
            ),
            timeout=_LEG_SUBMIT_TIMEOUT_SECONDS,
        )
        for leg, notional in legs_with_notionals
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    normalized: list[LegSubmitResult] = []
    for index, result in enumerate(results):
        leg, notional = legs_with_notionals[index]
        leg_id = str(leg.get("leg_id") or f"leg_{index + 1}")
        if isinstance(result, Exception):
            error_message = "Order submission timed out."
            if not isinstance(result, asyncio.TimeoutError):
                error_message = str(result)
            normalized.append(
                LegSubmitResult(
                    leg_id=leg_id,
                    status="failed",
                    effective_price=safe_float(leg.get("limit_price"), None),
                    error_message=error_message,
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
        return bool(await live_execution_service.cancel_order(order_id))
    except Exception:
        return False
