"""Build the at-placement-time covariate snapshot for survival analysis.

Every shadow order persists a SurvivalFeatures dict into its
``TraderOrder.payload_json["survival_features"]`` key.  The Cox PH
trainer ETL later joins this against the order's terminal event
(``executed`` / ``expired`` / ``cancelled``) to produce one labeled
training row per order:

    duration_seconds, event_observed (1 if filled, 0 if right-censored),
    + the covariates below.

Covariates are the ones outlined in the design:

* ``queue_ahead_shares``: from ExecutionEstimate
* ``depth_behind_shares``: depth past your level on your own side
* ``spread_bps``: wide spread = thin market = different fill dynamics
* ``mid_distance_bps``: your limit price relative to mid (signed)
* ``recent_trade_intensity_per_sec``: opposing-side trade flow rate
* ``time_to_resolution_seconds``: the Polymarket-specific covariate
  that nobody else models — fill hazard rate spikes near close on
  short-window crypto binaries
* ``side_imbalance``: bid_depth / (bid_depth + ask_depth)
* ``underlying_volatility_bps_per_min``: for crypto markets, realized
  vol of the underlying.  Pulled from strategy context if available;
  None otherwise.
* ``latency_p95_ms``: from ExecutionLatencyMetrics rolling window
* ``book_age_ms``: how stale was the book at placement
* ``notional_usd``: order size

Strata key (``market_type_strata``) is derived from the market's
timeframe metadata (e.g. ``crypto_15m``, ``crypto_60m``,
``event_resolved``) so the trainer can later split or treat as a
covariate depending on per-strata sample size.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from services.optimization.execution_estimator import ExecutionEstimate
from utils.converters import safe_float


# Time-to-resolution bucket edges (seconds).  Empirical observation on
# Polymarket 15-minute crypto binaries: 60-90% of taker flow concentrates
# in the last 5 minutes, with a sharp acceleration in the last 60s.  A
# linear-in-log-hazard Cox covariate underfits this — a piecewise-
# constant hazard via dummy indicators captures the spike.  The omitted
# baseline is "ttr >= 1800s" (≥ 30 min — the relaxed regime where fill
# hazard is approximately stationary).
TTR_BUCKET_EDGES_SECONDS: tuple[tuple[float, float, str], ...] = (
    (0.0, 30.0, "ttr_bucket_lt_30s"),
    (30.0, 60.0, "ttr_bucket_30_60s"),
    (60.0, 300.0, "ttr_bucket_60_300s"),
    (300.0, 1800.0, "ttr_bucket_300_1800s"),
)


def _time_to_resolution_buckets(ttr_seconds: float | None) -> dict[str, float]:
    """Emit one binary indicator per bucket; the >= 1800s baseline is
    encoded by all four indicators being 0 (omitted to avoid the
    dummy-variable trap)."""
    out: dict[str, float] = {name: 0.0 for _, _, name in TTR_BUCKET_EDGES_SECONDS}
    if ttr_seconds is None or not (ttr_seconds > 0):
        return out
    for lo, hi, name in TTR_BUCKET_EDGES_SECONDS:
        if lo <= float(ttr_seconds) < hi:
            out[name] = 1.0
            break
    return out


@dataclass
class SurvivalFeatures:
    queue_ahead_shares: float | None
    depth_behind_shares: float | None
    spread_bps: float | None
    mid_distance_bps: float | None
    recent_trade_intensity_per_sec: float | None
    time_to_resolution_seconds: float | None
    # Piecewise hazard buckets — one per non-baseline ttr region.  These
    # ride alongside the continuous time_to_resolution_seconds covariate
    # so Cox PH can fit both a linear-in-log-hazard slope AND a
    # non-linear spike near resolution.  Missing ttr ⇒ all zero (the
    # "no information" baseline).
    ttr_bucket_lt_30s: float | None
    ttr_bucket_30_60s: float | None
    ttr_bucket_60_300s: float | None
    ttr_bucket_300_1800s: float | None
    side_imbalance: float | None
    underlying_volatility_bps_per_min: float | None
    latency_p95_ms: float | None
    book_age_ms: float | None
    notional_usd: float | None
    market_type_strata: str

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def _market_type_strata(payload: dict[str, Any]) -> str:
    """Derive a strata bucket like ``crypto_15m`` from signal payload."""
    market_payload = payload.get("market") if isinstance(payload, dict) else None
    market_payload = market_payload if isinstance(market_payload, dict) else {}
    timeframe = (
        str(market_payload.get("timeframe") or "").strip().lower()
        or str((payload or {}).get("timeframe") or "").strip().lower()
    )
    category = str((market_payload or {}).get("category") or "").strip().lower()
    if "crypto" in category and timeframe:
        return f"crypto_{timeframe}"
    if timeframe:
        return f"event_{timeframe}"
    return "pooled"


def _depth_behind_for_side(
    order_book: Any,
    *,
    side: str,
    limit_price: float,
) -> float | None:
    """Sum depth on your side at WORSE prices than your limit (i.e. the
    queue *behind* you)."""
    levels = None
    if isinstance(order_book, dict):
        # On a BUY at price P, you sit on the bid side at P; "behind"
        # means depth at lower bid prices (i.e. someone bidding less,
        # who would be reached after you).
        if side.upper() in {"BUY", "BUY_YES", "BUY_NO"}:
            levels = order_book.get("bids")
            cmp = lambda lvl_price: lvl_price < limit_price  # noqa: E731
        else:
            levels = order_book.get("asks")
            cmp = lambda lvl_price: lvl_price > limit_price  # noqa: E731
    if not levels:
        return None
    total = 0.0
    for level in levels[:25]:
        if isinstance(level, dict):
            lp = safe_float(level.get("price"))
            ls = safe_float(level.get("size"))
        else:
            lp = safe_float(getattr(level, "price", None))
            ls = safe_float(getattr(level, "size", None))
        if lp is None or ls is None or lp <= 0 or ls <= 0:
            continue
        if cmp(lp):
            total += float(ls)
    return total


def _side_imbalance(order_book: Any) -> float | None:
    """bid_depth / (bid_depth + ask_depth) over top-N levels."""
    if not isinstance(order_book, dict):
        return None
    bid_total = 0.0
    ask_total = 0.0
    for level in (order_book.get("bids") or [])[:5]:
        size = safe_float(level.get("size") if isinstance(level, dict) else getattr(level, "size", None))
        if size:
            bid_total += float(size)
    for level in (order_book.get("asks") or [])[:5]:
        size = safe_float(level.get("size") if isinstance(level, dict) else getattr(level, "size", None))
        if size:
            ask_total += float(size)
    total = bid_total + ask_total
    if total <= 0:
        return None
    return bid_total / total


def _recent_trade_intensity(recent_trades: list[Any] | None, lookback_seconds: float) -> float | None:
    if not recent_trades:
        return 0.0
    if lookback_seconds <= 0:
        return None
    return float(len(recent_trades)) / float(lookback_seconds)


def _mid_distance_bps(order_book: Any, *, limit_price: float, side: str) -> float | None:
    if not isinstance(order_book, dict):
        return None
    bids = order_book.get("bids") or []
    asks = order_book.get("asks") or []
    best_bid = None
    best_ask = None
    if bids:
        bid_lvl = bids[0]
        best_bid = safe_float(bid_lvl.get("price") if isinstance(bid_lvl, dict) else getattr(bid_lvl, "price", None))
    if asks:
        ask_lvl = asks[0]
        best_ask = safe_float(ask_lvl.get("price") if isinstance(ask_lvl, dict) else getattr(ask_lvl, "price", None))
    if not best_bid or not best_ask or best_bid <= 0 or best_ask <= 0:
        return None
    mid = (best_bid + best_ask) / 2.0
    if mid <= 0:
        return None
    sign = 1.0 if side.upper() in {"BUY", "BUY_YES", "BUY_NO"} else -1.0
    return sign * (limit_price - mid) / mid * 10_000.0


def _spread_bps(order_book: Any) -> float | None:
    if not isinstance(order_book, dict):
        return None
    bids = order_book.get("bids") or []
    asks = order_book.get("asks") or []
    if not bids or not asks:
        return None
    bid_lvl = bids[0]
    ask_lvl = asks[0]
    best_bid = safe_float(bid_lvl.get("price") if isinstance(bid_lvl, dict) else getattr(bid_lvl, "price", None))
    best_ask = safe_float(ask_lvl.get("price") if isinstance(ask_lvl, dict) else getattr(ask_lvl, "price", None))
    if not best_bid or not best_ask or best_bid <= 0 or best_ask <= 0:
        return None
    mid = (best_bid + best_ask) / 2.0
    if mid <= 0:
        return None
    return (best_ask - best_bid) / mid * 10_000.0


def _time_to_resolution_seconds(payload: dict[str, Any]) -> float | None:
    """Pull seconds_to_resolution from signal payload if available."""
    if not isinstance(payload, dict):
        return None
    candidates = (
        payload.get("seconds_to_resolution"),
        payload.get("time_to_resolution_seconds"),
    )
    for c in candidates:
        v = safe_float(c)
        if v is not None and v > 0:
            return float(v)
    market_payload = payload.get("market") if isinstance(payload, dict) else None
    if isinstance(market_payload, dict):
        for key in ("seconds_to_resolution", "time_to_resolution_seconds"):
            v = safe_float(market_payload.get(key))
            if v is not None and v > 0:
                return float(v)
    return None


def _underlying_vol(payload: dict[str, Any]) -> float | None:
    """Pull recent realized vol of the underlying from strategy context.

    Strategies that compute it (e.g. crypto binaries) drop it under
    ``payload['live_market']['underlying_volatility_bps_per_min']`` or
    ``payload['strategy_context']['underlying_volatility_bps_per_min']``.
    Returns None if not available; the Cox trainer treats None as
    missing-at-random and imputes the cohort mean.
    """
    if not isinstance(payload, dict):
        return None
    for parent_key in ("live_market", "strategy_context"):
        parent = payload.get(parent_key)
        if isinstance(parent, dict):
            v = safe_float(parent.get("underlying_volatility_bps_per_min"))
            if v is not None:
                return float(v)
    return None


def build_survival_features(
    *,
    estimate: ExecutionEstimate | None,
    order_book: Any,
    recent_trades: list[Any] | None,
    book_age_ms: float | None,
    payload: dict[str, Any] | None,
    side: str,
    limit_price: float,
    notional_usd: float,
    latency_p95_ms: float | None,
    recent_trade_lookback_seconds: float,
) -> SurvivalFeatures:
    """Assemble the covariate snapshot.  All fields nullable — the
    trainer handles missingness."""
    payload = payload if isinstance(payload, dict) else {}
    ttr_seconds = _time_to_resolution_seconds(payload)
    ttr_buckets = _time_to_resolution_buckets(ttr_seconds)
    return SurvivalFeatures(
        queue_ahead_shares=(
            float(estimate.queue_ahead_shares) if estimate and estimate.queue_ahead_shares is not None else None
        ),
        depth_behind_shares=_depth_behind_for_side(
            order_book, side=side, limit_price=float(limit_price)
        ),
        spread_bps=_spread_bps(order_book),
        mid_distance_bps=_mid_distance_bps(
            order_book, limit_price=float(limit_price), side=side
        ),
        recent_trade_intensity_per_sec=_recent_trade_intensity(
            recent_trades, recent_trade_lookback_seconds
        ),
        time_to_resolution_seconds=ttr_seconds,
        ttr_bucket_lt_30s=ttr_buckets["ttr_bucket_lt_30s"],
        ttr_bucket_30_60s=ttr_buckets["ttr_bucket_30_60s"],
        ttr_bucket_60_300s=ttr_buckets["ttr_bucket_60_300s"],
        ttr_bucket_300_1800s=ttr_buckets["ttr_bucket_300_1800s"],
        side_imbalance=_side_imbalance(order_book),
        underlying_volatility_bps_per_min=_underlying_vol(payload),
        latency_p95_ms=float(latency_p95_ms) if latency_p95_ms is not None else None,
        book_age_ms=float(book_age_ms) if book_age_ms is not None else None,
        notional_usd=float(notional_usd) if notional_usd > 0 else None,
        market_type_strata=_market_type_strata(payload),
    )
