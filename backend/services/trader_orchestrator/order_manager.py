from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from services.live_execution_adapter import execute_live_order
from services.polymarket import polymarket_client
from services.live_execution_service import live_execution_service
from services.trader_orchestrator.hot_path import allow_polymarket_rest_call
from services.optimization.execution_estimator import (
    ExecutionEstimate,
    ExecutionEstimator,
)
from services.fill_simulator import (
    build_survival_features,
    ensemble_estimate,
    get_empirical_constants,
    measured_latency_cached,
)
from services.fill_simulator.cox_inference import _cache as _cox_cache  # for lookup hint
from services.strategy_sdk import StrategySDK
from utils.converters import safe_float

logger = logging.getLogger(__name__)


_MIN_EXECUTION_PRICE = 0.001
_MIN_LIVE_SHARES = 5.0
# Round 4: tightened from 35s -> 10s.  Under degraded CLOB health the
# previous 35s ceiling let a single stuck leg hold the session engine
# (and, by extension, the orchestrator cycle that's awaiting it)
# hostage for the full 35 seconds.  The fast lane proves 5s is
# survivable: on timeout we mark the leg failed, the pre-submit /
# execution_session row unwinds the cap, and the orphan-reconcile
# sweep reconciles any order that actually reached the venue against
# the venue snapshot on its next pass (~60s).  10s is the conservative
# middle ground for multi-leg slow-lane orders (which gather
# concurrently, so a wave of 2 legs races at 10s each, not 20s).
_LEG_SUBMIT_TIMEOUT_SECONDS = 10.0
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


def _chase_up_execution_caps(
    *,
    leg: dict[str, Any],
    metadata: dict[str, Any],
    params: dict[str, Any],
) -> list[float]:
    """Return only the execution-price caps eligible for chase-up.

    Excludes entry-band guards (``max_probability``,
    derived-from-``min_upside_percent``), which apply at
    signal-emission, not at submit.  Mixing the two collapses the
    chase-up ceiling to the entry-band ceiling and produces the
    Plan 0033 / 0035 cancellation cluster where the simulator was
    handed ``shadow_limit_price = max_probability`` and (correctly)
    rejected every ask above it.  See Plan 0035.
    """
    candidates = [
        _valid_execution_bound(leg.get("max_execution_price")),
        _valid_execution_bound(metadata.get("max_execution_price")),
        _valid_execution_bound(params.get("max_execution_price")),
        _valid_execution_bound(params.get("max_entry_price")),
    ]
    return [cap for cap in candidates if cap is not None]


_ALLOW_TAKER_LIMIT_BUY_ABOVE_SIGNAL_ALIASES = (
    "allow_taker_limit_buy_above_signal",
    "allow_taker_limit_pay_up",
    "allow_taker_limit_to_exceed_signal_price",
    "allow_buy_above_signal_price",
)
_AGGRESSIVE_LIMIT_BUY_SUBMIT_AS_GTC_ALIASES = (
    "aggressive_limit_buy_submit_as_gtc",
    "submit_aggressive_buy_limits_as_gtc",
    "submit_taker_limit_buy_as_gtc",
)


def _strategy_params_have_alias(params: dict[str, Any] | None, aliases: tuple[str, ...]) -> bool:
    if not isinstance(params, dict):
        return False
    if any(alias in params for alias in aliases):
        return True
    execution_policy = params.get("execution_policy")
    if isinstance(execution_policy, dict) and any(alias in execution_policy for alias in aliases):
        return True
    return False


def _allow_taker_limit_buy_above_signal(
    strategy_params: dict[str, Any] | None,
    risk_limits: dict[str, Any] | None = None,
) -> bool:
    if _strategy_params_have_alias(strategy_params, _ALLOW_TAKER_LIMIT_BUY_ABOVE_SIGNAL_ALIASES):
        return StrategySDK.allow_taker_limit_buy_above_signal_price(strategy_params or {}, default=False)
    if isinstance(risk_limits, dict):
        return bool(risk_limits.get("allow_taker_limit_buy_above_signal", False))
    return False


def _aggressive_limit_buy_submit_as_gtc(
    strategy_params: dict[str, Any] | None,
    risk_limits: dict[str, Any] | None = None,
) -> bool:
    if _strategy_params_have_alias(strategy_params, _AGGRESSIVE_LIMIT_BUY_SUBMIT_AS_GTC_ALIASES):
        return StrategySDK.aggressive_limit_buy_submit_as_gtc(strategy_params or {}, default=False)
    if isinstance(risk_limits, dict):
        return bool(risk_limits.get("aggressive_limit_buy_submit_as_gtc", False))
    return False


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
        candidates = _chase_up_execution_caps(
            leg=leg,
            metadata=metadata,
            params=strategy_params,
        )
        has_explicit_cap = bool(candidates)
        if price_policy == "taker_limit" and fallback_bound is not None:
            if allow_taker_limit_buy_above_signal:
                if not has_explicit_cap:
                    candidates.append(fallback_bound)
            else:
                candidates.append(fallback_bound)
        resolved = min(candidates, default=None)
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


def _compute_book_spread_bps(book_payload: dict[str, Any] | None) -> float | None:
    """Best-bid / best-ask spread in basis points from a normalized book payload.

    Returns None when either side is empty or both legs aren't strictly positive.
    Matches the convention used elsewhere (services/ai/market_analyzer.py and
    services/ai/tools/market_tools.py): best_bid = max(bid prices),
    best_ask = min(ask prices) — defensive against unsorted level lists.
    """
    if not isinstance(book_payload, dict):
        return None
    bids = book_payload.get("bids") or []
    asks = book_payload.get("asks") or []
    bid_prices = [float(b.get("price") or 0.0) for b in bids if isinstance(b, dict)]
    ask_prices = [float(a.get("price") or 0.0) for a in asks if isinstance(a, dict)]
    bid_prices = [p for p in bid_prices if p > 0.0]
    ask_prices = [p for p in ask_prices if p > 0.0]
    if not bid_prices or not ask_prices:
        return None
    best_bid = max(bid_prices)
    best_ask = min(ask_prices)
    if best_ask <= best_bid:
        return 0.0
    mid = (best_bid + best_ask) / 2.0
    if mid <= 0.0:
        return None
    return (best_ask - best_bid) / mid * 10_000.0


def _check_slippage_bps(
    *,
    signal_price: float | None,
    intended_price: float | None,
    risk_limits: dict[str, Any] | None,
) -> tuple[bool, float | None, float | None]:
    """Returns (rejected, drift_bps, configured_cap_bps).

    Compares intended fill price against signal.entry_price symmetrically.
    Captures the "market moved between gate-time and submit-time and the
    move exceeded my tolerance" case — distinct from max_entry_drift_pct
    which fires at gate-time against live mid in percent.

    Rejected only when configured cap > 0 AND both prices are positive AND
    the absolute drift exceeds the cap. Signal-emitted strategies that
    don't carry an entry_price → no-op (we can't measure drift).
    """
    if not isinstance(risk_limits, dict):
        return False, None, None
    cap_raw = risk_limits.get("slippage_bps")
    if cap_raw is None:
        return False, None, None
    try:
        cap = float(cap_raw)
    except (TypeError, ValueError):
        return False, None, None
    if cap <= 0.0:
        return False, None, cap
    if signal_price is None or signal_price <= 0.0:
        return False, None, cap
    if intended_price is None or intended_price <= 0.0:
        return False, None, cap
    drift_bps = abs(intended_price - signal_price) / signal_price * 10_000.0
    return (drift_bps > cap), drift_bps, cap


def _check_max_spread_bps(
    *,
    book_payload: dict[str, Any] | None,
    risk_limits: dict[str, Any] | None,
) -> tuple[bool, float | None, float | None]:
    """Returns (rejected, spread_bps, configured_cap_bps).

    Rejected when configured cap is > 0 AND book yields a measurable spread
    that exceeds the cap. Returns rejected=False when cap is unset/zero (knob
    off) or when no book is available (we don't reject for missing data — that
    is a separate skip path).
    """
    if not isinstance(risk_limits, dict):
        return False, None, None
    cap_raw = risk_limits.get("max_spread_bps")
    if cap_raw is None:
        return False, None, None
    try:
        cap = float(cap_raw)
    except (TypeError, ValueError):
        return False, None, None
    if cap <= 0.0:
        return False, None, cap
    spread_bps = _compute_book_spread_bps(book_payload)
    if spread_bps is None:
        return False, None, cap
    return (spread_bps > cap), spread_bps, cap


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
    risk_limits: dict[str, Any] | None = None,
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
    # Per-leg sub-stage breakdown — captures the work that wraps
    # ``execute_live_order`` (token-id resolve, orchestrator-level buy
    # gate, shadow book/tape resolve).  Production soak (5/2026/05)
    # showed a ~3.6 s gap between the orchestrator's ps_submit_order
    # bucket and the inner place_order.total_ms; without these buckets
    # the gap was opaque.  Merged into the LegSubmitResult.payload's
    # ``submit_breakdown`` dict alongside the inner place_order
    # breakdown so the orchestrator's cycle-slow log surfaces both.
    _leg_breakdown: dict[str, float] = {}
    _leg_started_total = time.monotonic()

    def _leg_record(stage: str, started_mono: float) -> None:
        elapsed_ms = (time.monotonic() - started_mono) * 1000.0
        _leg_breakdown[f"leg_{stage}"] = round(
            _leg_breakdown.get(f"leg_{stage}", 0.0) + elapsed_ms, 1
        )
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

    # Pre-trade slippage gate (per-trader risk_limits.slippage_bps).
    # Distinct from max_entry_drift_pct: this fires at submit-time against
    # the intended fill price (which may have drifted from the signal via
    # chase-up, limit-policy resolution, etc.), in bps for finer precision.
    # No-op when knob unset/zero or signal carries no entry_price.
    _signal_entry_price = safe_float(getattr(signal, "entry_price", None), None)
    _slip_rejected, _slip_drift_bps, _slip_cap = _check_slippage_bps(
        signal_price=_signal_entry_price,
        intended_price=price,
        risk_limits=risk_limits,
    )
    if _slip_rejected:
        return LegSubmitResult(
            leg_id=leg_id,
            status="skipped" if mode_key == "shadow" else "failed",
            effective_price=price,
            error_message=(
                f"Intended price {price:.4f} drifted {_slip_drift_bps:.1f} bps from "
                f"signal entry {_signal_entry_price:.4f} (cap {_slip_cap:.1f} bps)."
            ),
            payload={
                "mode": mode_key,
                "submission": "rejected",
                "reason": "slippage_bps_exceeded",
                "drift_bps": round(float(_slip_drift_bps), 2),
                "slippage_bps": float(_slip_cap),
                "signal_entry_price": float(_signal_entry_price),
                "intended_price": float(price),
                "leg": dict(leg),
                "requested_notional_usd": notional,
                "effective_notional_usd": 0.0,
            },
            shares=None,
            notional_usd=0.0,
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
            _stage_started = time.monotonic()
            token_id = await _fetch_token_id_from_market(market_id_for_lookup, outcome_for_lookup)
            _leg_record("token_id_fetch", _stage_started)
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

    # 2026-05-05: defense-in-depth pre-submit safety floor.
    #
    # Strategies are loaded from ``Strategy.source_code`` in the database
    # (see services.strategy_loader). A fix made in the .py file under
    # services/strategies/ does NOT propagate to the running process
    # unless the corresponding row is re-seeded. The catalog explicitly
    # preserves user edits across reseeds. So a strategy-level safety
    # gate (e.g. min_entry_price floor on contrarian maker bets) can
    # be silently absent from the live system.
    #
    # The check below runs at the submit boundary on EVERY live order,
    # regardless of which path constructed it. Per-strategy floors live
    # in services.execution_safety.py and can only be changed by
    # deploying new code — operators cannot loosen them via the UI.
    # Violations are reported as a SKIPPED result with reason
    # ``execution_safety_floor`` so the cap accounting is unaffected
    # and the operator console clearly shows the rejection cause.
    if mode_key == "live" and order_side == "BUY":
        from services.execution_safety import (
            assert_buy_entry_price_within_safety_bounds,
        )

        safety_strategy_slug = str(
            payload.get("strategy_key")
            or getattr(signal, "strategy_key", "")
            or ""
        ).strip().lower() or None
        _stage_started = time.monotonic()
        safety_assessment = assert_buy_entry_price_within_safety_bounds(
            strategy_slug=safety_strategy_slug,
            entry_price=price,
        )
        _leg_record("orchestrator_safety_floor", _stage_started)
        if not safety_assessment.passed:
            logger.warning(
                "Pre-submit safety floor refused order: %s",
                safety_assessment.message,
                extra={
                    "strategy_slug": safety_strategy_slug,
                    "entry_price": price,
                    "floor": safety_assessment.floor,
                    "ceiling": safety_assessment.ceiling,
                    "reason": safety_assessment.reason,
                    "leg_id": leg_id,
                },
            )
            return LegSubmitResult(
                leg_id=leg_id,
                status="skipped",
                effective_price=price,
                error_message=safety_assessment.message,
                payload={
                    "mode": mode_key,
                    "submission": "skipped",
                    "reason": "execution_safety_floor",
                    "safety_reason": safety_assessment.reason,
                    "strategy_slug": safety_strategy_slug,
                    "entry_price": price,
                    "safety_floor": safety_assessment.floor,
                    "safety_ceiling": safety_assessment.ceiling,
                    "token_id": token_id,
                    "leg": dict(leg),
                    "shares": shares,
                    "requested_notional_usd": notional,
                },
                shares=shares,
                notional_usd=effective_notional,
            )

    skip_buy_pre_submit_gate = False
    if mode_key == "live" and order_side == "BUY":
        _stage_started = time.monotonic()
        buy_gate_ok, buy_gate_error = await live_execution_service.check_buy_pre_submit_gate(
            token_id=token_id,
            required_notional_usd=effective_notional,
        )
        _leg_record("orchestrator_buy_gate", _stage_started)
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
        strategy_default=_allow_taker_limit_buy_above_signal(params, risk_limits),
    )
    aggressive_limit_buy_submit_as_gtc = _resolve_leg_execution_bool(
        leg=leg,
        key="aggressive_limit_buy_submit_as_gtc",
        strategy_default=_aggressive_limit_buy_submit_as_gtc(params, risk_limits),
    )

    # Compute execution-price bounds once and reuse for both shadow and
    # live paths.  In shadow we lift the estimator's `limit_price` ceiling
    # to `max_execution_price` when chase-up is enabled — without this,
    # `ensemble_estimate` breaks on the first ask above mid and returns
    # `limit_price_not_executable`, neutralizing the toggle in shadow mode.
    price_policy = str(leg.get("price_policy") or "").strip().lower()
    enforce_fallback = price_policy != "taker_limit"
    quote_aggressively = price_policy == "taker_limit"
    max_execution_price, min_execution_price = _resolve_execution_price_bounds(
        leg=leg,
        strategy_params=params,
        fallback_price=price,
        allow_taker_limit_buy_above_signal=allow_taker_limit_buy_above_signal,
    )

    if mode_key == "shadow":
        order_type = "taker_limit" if price_policy == "taker_limit" else "maker_limit"
        book_payload, recent_trades, book_age_ms, quote_source, quote_error = await _resolve_shadow_book_and_tape(
            token_id=token_id,
            live_context=live_context,
        )
        payload_mode = "shadow"
        submission_label = "shadow_microstructure_simulated"
        # Pre-trade max-spread gate (per-trader risk_limits.max_spread_bps).
        # Applies symmetrically to shadow + live so the shadow preview
        # matches what live would do. No-op when knob is unset/zero.
        _spread_rejected, _spread_bps, _spread_cap = _check_max_spread_bps(
            book_payload=book_payload,
            risk_limits=risk_limits,
        )
        if _spread_rejected:
            return LegSubmitResult(
                leg_id=leg_id,
                status="skipped",
                effective_price=price,
                error_message=f"Book spread {_spread_bps:.1f} bps exceeds max_spread_bps {_spread_cap:.1f}.",
                payload={
                    "mode": payload_mode,
                    "submission": "rejected",
                    "reason": "max_spread_bps_exceeded",
                    "spread_bps": round(float(_spread_bps), 2),
                    "max_spread_bps": float(_spread_cap),
                    "token_id": token_id,
                    "leg": dict(leg),
                    "requested_notional_usd": notional,
                    "effective_notional_usd": 0.0,
                },
                shares=shares,
                notional_usd=0.0,
            )
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

        # Build the ensemble: pessimistic / realistic / optimistic.
        # The Cox model is loaded lazily — if an active model has been
        # cached in-process we evaluate; otherwise the ensemble falls
        # back to the heuristic estimator's fill_probability output.
        # Cox loading is fire-and-forget on the cache: subsequent
        # orders pick up the loaded model on cache hit.
        constants = get_empirical_constants()
        latency = measured_latency_cached()
        survival_features_struct = build_survival_features(
            estimate=None,
            order_book=book_payload,
            recent_trades=recent_trades,
            book_age_ms=book_age_ms,
            payload=payload,
            side=order_side,
            limit_price=float(price or 0.0),
            notional_usd=float(notional or 0.0),
            latency_p95_ms=latency.p95_ms,
            recent_trade_lookback_seconds=30.0,
        )
        survival_covariates = survival_features_struct.to_payload()
        cox_snapshot = None
        cached_entry = _cox_cache.get(survival_covariates.get("market_type_strata") or "pooled")
        if cached_entry is not None:
            _ts, cox_snapshot = cached_entry

        # Effective ceiling fed into the simulator.  Default = live mid /
        # signal entry price (current behaviour).  When chase-up is on
        # for a BUY, lift the ceiling to the strongest explicit
        # execution-price cap from strategy_params / leg / metadata; if
        # none, use 1.0 (the natural market boundary).  Entry-band
        # guards (``max_probability``, ``min_upside_percent``-derived)
        # are explicitly excluded — they apply at signal-emission, not
        # at chase-up; mixing them collapses the chase-up ceiling to
        # the entry-band ceiling (Plan 0033 / 0035).  Note:
        # ``max_execution_price`` from ``_resolve_execution_price_bounds``
        # includes the signal-price fallback even when chase=True, so
        # it cannot be used here — the whole point of chase-up is to
        # ignore that fallback.
        shadow_limit_price = float(price or 0.0)
        if (
            allow_taker_limit_buy_above_signal
            and order_side == "BUY"
            and shadow_limit_price > 0.0
        ):
            metadata_for_caps = leg.get("metadata") if isinstance(leg.get("metadata"), dict) else {}
            explicit_buy_caps = _chase_up_execution_caps(
                leg=leg,
                metadata=metadata_for_caps,
                params=params,
            )
            tightest_explicit_cap = min(explicit_buy_caps, default=None)
            if tightest_explicit_cap is not None and tightest_explicit_cap > shadow_limit_price:
                shadow_limit_price = float(tightest_explicit_cap)
            elif tightest_explicit_cap is None:
                shadow_limit_price = 1.0

        ensemble = ensemble_estimate(
            order_book=book_payload,
            side=order_side,
            size_shares=shares,
            limit_price=shadow_limit_price,
            order_type=order_type,
            recent_trades=recent_trades,
            book_age_ms=book_age_ms,
            time_in_force_seconds=6.0,
            fee_bps=0.0,
            cox_snapshot=cox_snapshot,
            survival_covariates=survival_covariates,
            latency=latency,
            constants=constants,
        )
        # The realistic scenario is the canonical fill estimate that
        # decides status/notional/etc.  Pessimistic + optimistic are
        # persisted alongside for the UI's PnL-band display.
        estimate = ensemble.realistic.estimate
        quote_price = estimate.average_price
        effective_shadow_notional = estimate.filled_notional_usd
        shadow_status = _shadow_status_for_estimate(estimate)
        # Update the persisted snapshot's ``queue_ahead_shares`` with the
        # realized estimate's value so the Cox trainer ETL sees the
        # realized queue position.  All other features are identical to
        # the at-placement snapshot — ``dataclasses.replace`` avoids the
        # cost of rebuilding the full feature struct (book traversal,
        # recent-trade intensity scan, ttr bucketing) just to swap one
        # field.
        from dataclasses import replace as _dc_replace

        survival_features = _dc_replace(
            survival_features_struct,
            queue_ahead_shares=(
                float(estimate.queue_ahead_shares)
                if estimate.queue_ahead_shares is not None
                else None
            ),
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
            "fill_probability": ensemble.realistic.fill_probability,
            "queue_ahead_shares": estimate.queue_ahead_shares,
            "levels_consumed": estimate.levels_consumed,
            "execution_estimate": estimate.to_dict(),
            "survival_features": survival_features.to_payload(),
            "ensemble": ensemble.to_dict(),
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

        # Strategy hook: notify on the shadow fill so a strategy can
        # update its own priors / calibrate its fill model.  Fire-and-
        # forget — failures are logged in the dispatcher, never block
        # the leg result.
        if estimate.filled_shares > 0:
            try:
                strategy_slug = str(payload.get("strategy_key") or getattr(signal, "strategy_key", "") or "")
                if strategy_slug:
                    from services.strategy_callbacks import dispatch_on_fill

                    asyncio.create_task(
                        dispatch_on_fill(
                            strategy_slug=strategy_slug,
                            order=signal,
                            mode="shadow",
                            filled_shares=estimate.filled_shares,
                            average_price=quote_price or 0.0,
                            notional_usd=effective_shadow_notional,
                            ensemble_snapshot=ensemble.to_dict(),
                        )
                    )
            except Exception:
                pass  # never raise from a fill notification

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

    # Capture the at-submit-time market microstructure snapshot so the
    # Cox PH trainer has labeled training rows for live orders too.
    # Read once BEFORE the (possibly slow) live submit so the snapshot
    # reflects the book the strategy actually decided against, not the
    # post-fill book.  Survival features are best-effort — failures are
    # logged via shadow_simulation snapshot path and don't block the order.
    _stage_started = time.monotonic()
    live_book_payload, live_recent_trades, live_book_age_ms, live_quote_source, _live_quote_err = (
        await _resolve_shadow_book_and_tape(token_id=token_id, live_context=live_context)
    )
    _leg_record("book_tape_resolve", _stage_started)
    # Pre-trade max-spread gate for live path. Mirrors the shadow-branch
    # gate above so a knob configured in the per-trader Risk Limits flyout
    # actually blocks submission when the book is too wide.
    _live_spread_rejected, _live_spread_bps, _live_spread_cap = _check_max_spread_bps(
        book_payload=live_book_payload,
        risk_limits=risk_limits,
    )
    if _live_spread_rejected:
        return LegSubmitResult(
            leg_id=leg_id,
            status="failed",
            effective_price=price,
            error_message=f"Book spread {_live_spread_bps:.1f} bps exceeds max_spread_bps {_live_spread_cap:.1f}.",
            payload={
                "mode": "live",
                "submission": "rejected",
                "reason": "max_spread_bps_exceeded",
                "spread_bps": round(float(_live_spread_bps), 2),
                "max_spread_bps": float(_live_spread_cap),
                "token_id": token_id,
                "leg": dict(leg),
                "requested_notional_usd": notional,
                "effective_notional_usd": 0.0,
            },
            shares=shares,
            notional_usd=0.0,
        )
    live_latency = measured_latency_cached()
    live_survival_features = build_survival_features(
        estimate=None,
        order_book=live_book_payload,
        recent_trades=live_recent_trades,
        book_age_ms=live_book_age_ms,
        payload=payload,
        side=order_side,
        limit_price=float(price or 0.0),
        notional_usd=float(notional or 0.0),
        latency_p95_ms=live_latency.p95_ms,
        recent_trade_lookback_seconds=30.0,
    )

    _stage_started = time.monotonic()
    # Transient-failure retry policy from per-trader risk_limits.
    # retry_limit=N means up to N RETRIES after the initial attempt (i.e. N+1
    # total tries). Only retry when execute_live_order returns
    # status="failed" with payload.submission == "exception" — that is the
    # transport/SDK exception path in live_execution_adapter.py. Logic
    # rejections ("rejected", "not_ready", "not_executable") are NOT retried.
    _retry_limit_raw = (risk_limits or {}).get("retry_limit")
    _retry_backoff_raw = (risk_limits or {}).get("retry_backoff_ms")
    try:
        _retry_limit = max(0, min(50, int(_retry_limit_raw))) if _retry_limit_raw is not None else 0
    except (TypeError, ValueError):
        _retry_limit = 0
    try:
        _retry_backoff_ms = max(0, min(60_000, int(_retry_backoff_raw))) if _retry_backoff_raw is not None else 0
    except (TypeError, ValueError):
        _retry_backoff_ms = 0
    _attempts_used = 0
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
    while (
        _retry_limit > 0
        and _attempts_used < _retry_limit
        and execution.status == "failed"
        and isinstance(execution.payload, dict)
        and str(execution.payload.get("submission") or "") == "exception"
    ):
        _attempts_used += 1
        if _retry_backoff_ms > 0:
            await asyncio.sleep(_retry_backoff_ms / 1000.0)
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
    if _attempts_used > 0 and isinstance(execution.payload, dict):
        execution.payload["transient_retry_attempts"] = _attempts_used
    _leg_record("execute_live_order", _stage_started)

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

    # Strategy hook: notify on the live fill (synchronous case — taker
    # cross or post-only that immediately matched).  Mirrors the shadow
    # branch dispatch above.  Async fills detected later via wallet
    # reconciliation are hooked separately in position_lifecycle.py.
    _live_filled = float(execution.payload.get("filled_size") or 0.0)
    if execution.status in {"executed", "completed"} and _live_filled > 0:
        try:
            strategy_slug = str(payload.get("strategy_key") or getattr(signal, "strategy_key", "") or "")
            if strategy_slug:
                from services.strategy_callbacks import dispatch_on_fill

                asyncio.create_task(
                    dispatch_on_fill(
                        strategy_slug=strategy_slug,
                        order=signal,
                        mode="live",
                        filled_shares=_live_filled,
                        average_price=float(execution.effective_price or 0.0),
                        notional_usd=effective_notional,
                        ensemble_snapshot=None,  # live has no simulator ensemble
                    )
                )
        except Exception:
            pass  # never raise from a fill notification

    # Stamp the wrapper-level total + merge our leg sub-stages INTO
    # the inner place_order ``submit_breakdown`` dict so the
    # orchestrator's slow log surfaces both.  ``submit_breakdown``
    # already exists in execution.payload (set by live_execution
    # _adapter); we add ``leg_*`` keys to it.
    _leg_breakdown["leg_total_ms"] = round(
        (time.monotonic() - _leg_started_total) * 1000.0, 1
    )
    _merged_payload: dict[str, Any] = {**execution.payload}
    _existing_breakdown = _merged_payload.get("submit_breakdown")
    if not isinstance(_existing_breakdown, dict):
        _existing_breakdown = {}
    _merged_breakdown = dict(_existing_breakdown)
    _merged_breakdown.update(_leg_breakdown)
    _merged_payload["submit_breakdown"] = _merged_breakdown

    return LegSubmitResult(
        leg_id=leg_id,
        status=execution.status,
        effective_price=execution.effective_price,
        error_message=execution.error_message,
        payload={
            **_merged_payload,
            "mode": "live",
            "leg": dict(leg),
            "token_id_source": token_source,
            "shares": shares,
            "requested_shares": requested_shares,
            "min_live_shares": _MIN_LIVE_SHARES,
            "requested_notional_usd": notional,
            "effective_notional_usd": effective_notional,
            # Cox PH trainer reads from this key on either shadow or
            # live orders.  The "shadow_simulation" naming is a slight
            # misnomer for live (no simulation happened) but keeping it
            # uniform means the trainer doesn't need a mode branch.
            "shadow_simulation": {
                "mode": "live",
                "survival_features": live_survival_features.to_payload(),
                "live_book_quote_source": live_quote_source,
                "live_book_age_ms": live_book_age_ms,
            },
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
    risk_limits: dict[str, Any] | None = None,
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
                risk_limits=risk_limits,
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
