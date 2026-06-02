# SEED TEMPLATE — not the live strategy. The DB `strategies.source_code` is the
# runtime master (loaded by the orchestrator + backtester); edits here only seed
# fresh installs / reset-to-factory. See services/opportunity_strategy_catalog.py.
"""Crypto distance-edge continuation strategy.

Rides the heavily-favored side of a Polymarket crypto over-under cycle
when two confirmations line up at once:

1. **Distance** — the Chainlink oracle is far enough from the cycle's
   reference price (``price_to_beat``), measured in *dollars* of the
   underlying. A BTC cycle that's $300 in the money is far less likely
   to flip than one that's only $40 in the money.
2. **Cost** — the live Polymarket book lets us buy that favored side at
   or above a cost floor, where the cost floor *loosens* as distance
   grows. The book pricing the side richly is the market agreeing with
   the oracle; combining both is stronger than either alone.

The two are encoded as a tiered table (the defaults below come from a
community-reported BTC 15m configuration)::

    distance ≥ $150  →  cost ≥ 90¢
    distance ≥ $200  →  cost ≥ 85¢
    distance ≥ $300  →  cost ≥ 80¢

The applicable tier is the one with the largest ``distance_usd`` that the
current distance clears; the favored side must then cost at least that
tier's ``min_cost_cents`` (and no more than ``max_cost_cents``, since
buying at 99¢ leaves no edge after fees). Below the smallest tier we
don't trade.

Distance is in DOLLARS, so the default tiers are calibrated for BTC.
Other assets (ETH/SOL/XRP) trade at very different absolute prices and
would need their own tiers — that's why the shipped default enables BTC
only.

Resolution truth source
-----------------------
Polymarket resolves these crypto markets against Chainlink Data Streams
(BTC/USD etc.), NOT Binance spot. We track Chainlink via
``pick_oracle_source(prefer="chainlink")`` so the "which side is winning?"
check matches what Polymarket itself uses at resolution.

Expected value
--------------
There is no validated win rate for this configuration. By default the
strategy treats the live cost as the market-implied win probability
(``edge_uplift_pct = 0``), which after Polymarket taker fees is slightly
negative EV — i.e. it tells the truth: the distance signal must actually
lift the true win probability above the market price for this to profit.
Users who have backtested an edge can dial ``edge_uplift_pct`` up in the
UI to reflect it. Fees are always modeled via ``taker_fee_pct``.

All gates are user-editable via the strategy-manager UI — see the seed
entry in ``opportunity_strategy_catalog.py`` for the ``config_schema``.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from models import Market, Opportunity
from services.data_events import DataEvent
from services.strategies._firehose import (
    GateResult,
    MURMUR,
    WHISPER,
    emit_emit_nowait,
    emit_evaluation_nowait,
)
from services.strategies.base import BaseStrategy
from services.strategy_helpers.crypto_strategy_utils import (
    build_binary_crypto_market,
    normalize_timeframe,
    pick_oracle_source,
    taker_fee_pct,
)
from services.strategy_sdk import StrategySDK
from utils.converters import to_float
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Defaults — every value is overridable via the DB ``config`` column.
# ---------------------------------------------------------------------------

_ALL_ASSETS: tuple[str, ...] = ("BTC", "ETH", "SOL", "XRP")

# Default tiers from the community-reported BTC 15m configuration. Each
# tier is (min distance from reference in USD, min favored-side cost in ¢).
# Cost floors loosen as distance grows — distance is the confirmation.
_DEFAULT_TIERS: list[dict[str, float]] = [
    {"distance_usd": 150.0, "min_cost_cents": 90.0},
    {"distance_usd": 200.0, "min_cost_cents": 85.0},
    {"distance_usd": 300.0, "min_cost_cents": 80.0},
]

DEFAULT_CONFIG: dict[str, Any] = {
    # Per-asset enable list. Distance tiers are in DOLLARS, so the shipped
    # defaults only make sense for BTC. Re-enable other assets only after
    # re-tuning the tiers for that asset's price scale.
    "assets": ["BTC"],
    # Tiered distance($)→min-cost(¢) table. Sorted ascending by distance
    # in configure(); the applicable tier is the highest distance cleared.
    "distance_cost_tiers": [dict(t) for t in _DEFAULT_TIERS],
    # Don't buy above this — at the top of the book edge is fee-dominated.
    "max_cost_cents": 98.0,
    # Notional per trade (USD). Also sizes the VWAP depth probe so the cost
    # we gate on is the cost we'd actually pay for this fill.
    "bet_size_usd": 15.0,
    # Don't enter with less than this much time left (stale-book / fill-risk
    # guard right at resolution).
    "min_seconds_to_resolution": 10.0,
    # Reject Chainlink readings older than this.
    "max_oracle_age_ms": 5000,
    # Belief that the distance signal lifts true win probability above the
    # market-implied price, in percentage points. Default 0 = no assumed
    # edge (honest, slightly-negative EV after fees). Raise once backtested.
    "edge_uplift_pct": 0.0,
    # Master switch (also exposed at the row level via Strategy.enabled).
    "enabled": True,
}


def crypto_distance_edge_config_schema() -> dict[str, Any]:
    """Return the param-fields schema for the strategy-manager UI."""
    return {
        "param_fields": [
            {
                "key": "enabled",
                "label": "Enabled",
                "type": "boolean",
                "default": True,
                "phase": "signal",
            },
            {
                "key": "assets",
                "label": "Assets (distance tiers are USD — BTC-calibrated)",
                "type": "list",
                "options": list(_ALL_ASSETS),
                "default": ["BTC"],
                "phase": "signal",
            },
            {
                "key": "distance_cost_tiers",
                "label": "Distance($)→Min-Cost(¢) Tiers",
                "type": "json",
                "default": [dict(t) for t in _DEFAULT_TIERS],
                "phase": "signal",
            },
            {
                "key": "max_cost_cents",
                "label": "Max Cost (¢)",
                "type": "number",
                "min": 0.0,
                "max": 100.0,
                "default": 98.0,
                "phase": "execution",
            },
            {
                "key": "bet_size_usd",
                "label": "Bet Size (USD)",
                "type": "number",
                "min": 1.0,
                "max": 10_000.0,
                "default": 15.0,
                "phase": "execution",
            },
            {
                "key": "min_seconds_to_resolution",
                "label": "Min Seconds to Resolution",
                "type": "number",
                "min": 0.0,
                "max": 900.0,
                "default": 10.0,
                "phase": "signal",
            },
            {
                "key": "max_oracle_age_ms",
                "label": "Max Oracle Age (ms)",
                "type": "integer",
                "min": 0,
                "max": 60_000,
                "default": 5_000,
                "phase": "signal",
            },
            {
                "key": "edge_uplift_pct",
                "label": "Assumed Edge Uplift (pct points over implied)",
                "type": "number",
                "min": 0.0,
                "max": 50.0,
                "default": 0.0,
                "phase": "signal",
            },
        ]
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_asset(value: Any) -> str:
    asset = str(value or "").strip().upper()
    if asset == "XBT":
        return "BTC"
    return asset


# Word-boundary anchored so "5min" / "115min" don't match "15min".
_FIFTEEN_MIN_SLUG_RE = re.compile(r"(?:^|[-_])15(?:min|m|-minute)(?:[-_]|$)")


def _detect_15m(market: dict[str, Any]) -> bool:
    """True when the market dict represents a 15-minute crypto cycle."""
    if normalize_timeframe(market.get("timeframe")) == "15m":
        return True
    slug = str(market.get("slug") or "").lower()
    return bool(_FIFTEEN_MIN_SLUG_RE.search(slug))


def _sanitize_tiers(raw: Any) -> list[dict[str, float]]:
    """Coerce a tier config into a clean, ascending-by-distance list."""
    tiers: list[dict[str, float]] = []
    if not isinstance(raw, (list, tuple)):
        return [dict(t) for t in _DEFAULT_TIERS]
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        distance = to_float(entry.get("distance_usd"), None)
        cost = to_float(entry.get("min_cost_cents"), None)
        if distance is None or cost is None:
            continue
        if distance < 0.0 or cost < 0.0:
            continue
        tiers.append({"distance_usd": float(distance), "min_cost_cents": float(cost)})
    if not tiers:
        return [dict(t) for t in _DEFAULT_TIERS]
    tiers.sort(key=lambda t: t["distance_usd"])
    return tiers


def _select_tier(
    distance_usd: float, tiers: list[dict[str, float]]
) -> Optional[dict[str, float]]:
    """Return the tier with the largest ``distance_usd`` the move clears.

    ``tiers`` must be sorted ascending by ``distance_usd``. Epsilon admits
    exact-threshold hits that come in a hair short due to float arithmetic.
    """
    chosen: Optional[dict[str, float]] = None
    for tier in tiers:
        if distance_usd + 1e-9 >= tier["distance_usd"]:
            chosen = tier
        else:
            break
    return chosen


def _build_market(d: dict[str, Any]) -> Market | None:
    """Crypto-worker market dict -> typed ``Market`` via the canonical shared
    reconstructor (``build_binary_crypto_market``).  Returns ``None`` for rows
    with no usable price/id — callers reject those (the strategy's gates do too)."""
    return build_binary_crypto_market(d)


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------


class CryptoDistanceEdgeStrategy(BaseStrategy):
    """Buy the favored side when distance and cost both confirm it."""

    strategy_type = "crypto_distance_edge"
    name = "Crypto Distance Edge"
    description = (
        "Buys the favored side of a 15-minute crypto over-under cycle when "
        "the Chainlink oracle is far enough (in dollars) from the cycle "
        "reference AND the Polymarket book prices that side richly enough — "
        "a tiered distance($)→cost(¢) table where the cost floor loosens as "
        "distance grows. Holds to resolution."
    )
    source_key = "crypto"
    market_categories = ["crypto"]
    requires_historical_prices = False
    subscriptions = ["crypto_update"]
    supports_entry_take_profit_exit = False
    default_open_order_timeout_seconds = 30.0

    default_config = dict(DEFAULT_CONFIG)

    def __init__(self) -> None:
        super().__init__()
        self.min_profit = 0.0
        self.fee = 0.0
        # Per-market guard: the cycle end_ts we've already emitted for, so a
        # condition that stays true across many ticks emits at most once per
        # cycle. Resets implicitly when the market rolls to a new end_ts.
        self._emitted_cycle_end_ms: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def configure(self, config: dict) -> None:
        merged = dict(self.default_config)
        if config:
            merged.update(config)
        # Sanitize: assets list normalized to upper-case canonical names.
        raw_assets = merged.get("assets") or []
        if isinstance(raw_assets, str):
            raw_assets = [a.strip() for a in raw_assets.split(",")]
        normalized_assets: list[str] = []
        seen: set[str] = set()
        for a in raw_assets:
            n = _normalize_asset(a)
            if n and n in _ALL_ASSETS and n not in seen:
                seen.add(n)
                normalized_assets.append(n)
        merged["assets"] = normalized_assets
        merged["distance_cost_tiers"] = _sanitize_tiers(
            merged.get("distance_cost_tiers")
        )
        self.config = merged

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------

    async def on_event(self, event: DataEvent) -> list[Opportunity]:
        if event.event_type != "crypto_update":
            return []
        if not self.config.get("enabled", True):
            return []

        markets = event.payload.get("markets") or []
        if not markets:
            return []

        opportunities: list[Opportunity] = []
        now_ms = self.now_ms()
        for market in markets:
            if not isinstance(market, dict):
                continue
            opp = self._evaluate_market(market, now_ms=now_ms)
            if opp is not None:
                opportunities.append(opp)
        return opportunities

    # ------------------------------------------------------------------
    # Per-market gate
    # ------------------------------------------------------------------

    def _evaluate_market(
        self,
        market: dict[str, Any],
        *,
        now_ms: int,
    ) -> Optional[Opportunity]:
        gates: list[GateResult] = []
        tiers: list[dict[str, float]] = self.config.get("distance_cost_tiers") or []
        min_left_cfg = float(self.config.get("min_seconds_to_resolution", 10.0))
        max_age_ms = float(self.config.get("max_oracle_age_ms", 5000))
        max_cost_cents = float(self.config.get("max_cost_cents", 98.0))
        bet_size_usd = float(self.config.get("bet_size_usd", 15.0))
        edge_uplift_pct = float(self.config.get("edge_uplift_pct", 0.0))

        def _emit_reject(verbosity: str = MURMUR) -> None:
            emit_evaluation_nowait(
                strategy_slug="crypto_distance_edge",
                market=market,
                gates=gates,
                outcome="rejected",
                verbosity=verbosity,
            )

        # Gate 1: 15-minute timeframe.
        is_15m = _detect_15m(market)
        gates.append(GateResult(
            "timeframe", "15-minute timeframe", is_15m,
            detail=f"timeframe={market.get('timeframe') or '?'} slug={market.get('slug') or '?'}",
        ))
        if not is_15m:
            _emit_reject(WHISPER)
            return None

        market_id = str(market.get("condition_id") or market.get("id") or "")
        gates.append(GateResult(
            "market_id", "Market id present", bool(market_id),
            detail="condition_id or id required",
        ))
        if not market_id:
            _emit_reject(WHISPER)
            return None

        # Gate 2: per-asset enable list.
        asset = _normalize_asset(
            market.get("asset") or market.get("symbol") or market.get("coin")
        )
        configured_assets = self.config.get("assets") or []
        asset_passed = bool(asset) and asset in configured_assets
        gates.append(GateResult(
            "asset_enabled", "Asset in config list", asset_passed,
            detail=f"asset={asset or '?'} configured={','.join(configured_assets) or 'none'}",
        ))
        if not asset_passed:
            _emit_reject(WHISPER)
            return None

        end_ms_value = StrategySDK._coerce_end_ts_ms(market)
        gates.append(GateResult(
            "end_timestamp", "Cycle end timestamp parseable", end_ms_value is not None,
            detail=f"end_ts_ms={end_ms_value}",
        ))
        if end_ms_value is None:
            _emit_reject(WHISPER)
            return None

        # Gate 3: one emit per cycle. Once we've fired for this end_ts we
        # stay quiet until the market rolls to the next cycle. WHISPER —
        # most ticks after the first emit rest here.
        already_emitted = self._emitted_cycle_end_ms.get(market_id) == end_ms_value
        gates.append(GateResult(
            "not_yet_emitted", "Not already emitted this cycle", not already_emitted,
            detail=f"emitted_end_ms={self._emitted_cycle_end_ms.get(market_id)}",
        ))
        if already_emitted:
            _emit_reject(WHISPER)
            return None

        seconds_left = (end_ms_value - now_ms) / 1000.0
        min_left_passed = seconds_left >= min_left_cfg
        gates.append(GateResult(
            "min_seconds_to_resolution", "Min seconds to resolution",
            min_left_passed, score=seconds_left,
            detail=f"left={seconds_left:.1f}s min={min_left_cfg:.1f}s",
        ))
        if not min_left_passed:
            _emit_reject(WHISPER)
            return None

        reference = to_float(market.get("price_to_beat"), None)
        ref_passed = reference is not None and reference > 0.0
        gates.append(GateResult(
            "reference_price", "Reference price available", ref_passed,
            score=reference,
            detail=f"price_to_beat={reference}",
        ))
        if not ref_passed:
            _emit_reject(MURMUR)
            return None

        chainlink = pick_oracle_source(
            market, prefer="chainlink", max_age_ms=max_age_ms, now_ms=now_ms
        )
        oracle_source = str(chainlink.get("source", "")).lower() if chainlink else None
        oracle_age = float(chainlink.get("age_ms", 0.0)) if chainlink else None
        oracle_passed = chainlink is not None and oracle_source == "chainlink"
        gates.append(GateResult(
            "fresh_chainlink", "Fresh Chainlink oracle", oracle_passed,
            score=oracle_age,
            detail=f"source={oracle_source or 'none'} age_ms={oracle_age} max_age_ms={max_age_ms:.0f}",
        ))
        if not oracle_passed:
            _emit_reject(MURMUR)
            return None
        spot = float(chainlink["price"])
        spot_passed = spot > 0.0
        gates.append(GateResult(
            "spot_price", "Spot price > 0", spot_passed, score=spot,
            detail=f"spot={spot}",
        ))
        if not spot_passed:
            _emit_reject(MURMUR)
            return None

        # Distance + tier selection. Distance is in DOLLARS of underlying.
        distance_usd = abs(spot - reference)
        side = "YES" if spot >= reference else "NO"
        tier = _select_tier(distance_usd, tiers)
        tier_passed = tier is not None
        smallest_distance = tiers[0]["distance_usd"] if tiers else float("inf")
        gates.append(GateResult(
            "distance_tier", "Distance clears a tier", tier_passed,
            score=distance_usd,
            detail=(
                f"distance=${distance_usd:.2f} "
                f"tier={'$%.0f→%.0f¢' % (tier['distance_usd'], tier['min_cost_cents']) if tier else 'none'} "
                f"min_tier=${smallest_distance:.0f}"
            ),
        ))
        if tier is None:
            _emit_reject(MURMUR)
            return None
        min_cost_cents = float(tier["min_cost_cents"])

        typed_market = _build_market(market)
        if typed_market is None:
            _emit_reject(MURMUR)
            return None
        token_ids_present = bool(typed_market.clob_token_ids)
        gates.append(GateResult(
            "clob_tokens", "CLOB token ids present", token_ids_present,
            detail=f"count={len(typed_market.clob_token_ids)}",
        ))
        if not token_ids_present:
            _emit_reject(MURMUR)
            return None

        depth = StrategySDK.get_order_book_depth(
            typed_market, side=side, size_usd=bet_size_usd
        )
        depth_present = depth is not None
        gates.append(GateResult(
            "book_depth", "Order book depth available", depth_present,
            detail=f"side={side} size_usd={bet_size_usd:.2f}",
        ))
        if not depth_present:
            _emit_reject(MURMUR)
            return None
        is_fresh = bool(depth.get("is_fresh", False))
        gates.append(GateResult(
            "book_fresh", "Order book fresh", is_fresh,
            score=float(depth.get("staleness_ms") or 0.0),
            detail=f"staleness_ms={depth.get('staleness_ms')}",
        ))
        if not is_fresh:
            _emit_reject(MURMUR)
            return None

        vwap_price = float(depth.get("vwap_price") or 0.0)
        cost_cents = vwap_price * 100.0
        cost_passed = (
            vwap_price > 0.0
            and cost_cents + 1e-9 >= min_cost_cents
            and cost_cents <= max_cost_cents + 1e-9
        )
        gates.append(GateResult(
            "cost_in_range", "Cost within tier floor and ceiling", cost_passed,
            score=cost_cents,
            detail=(
                f"cost={cost_cents:.2f}¢ "
                f"range=[{min_cost_cents:.2f},{max_cost_cents:.2f}]¢"
            ),
        ))
        if not cost_passed:
            _emit_reject(MURMUR)
            return None

        # All gates passed — build the Opportunity. EV is modeled net of
        # Polymarket taker fees; win prob is the market-implied price plus
        # the user's (default 0) assumed edge uplift.
        fee_frac = taker_fee_pct(vwap_price)
        win_prob = min(0.9999, max(0.0001, vwap_price + edge_uplift_pct / 100.0))
        ev_per_share = (
            win_prob * (1.0 - vwap_price - fee_frac) - (1.0 - win_prob) * vwap_price
        )
        roi_percent = (ev_per_share / vwap_price) * 100.0 if vwap_price > 0 else 0.0
        token_id = typed_market.clob_token_ids[0 if side == "YES" else 1]

        slug = str(market.get("slug") or market_id)
        title = f"Crypto distance edge: {slug} {side}"
        description = (
            f"distance edge | {asset} 15m | "
            f"distance=${distance_usd:.2f} | "
            f"cost={cost_cents:.2f}¢ (tier floor {min_cost_cents:.0f}¢) | "
            f"oracle_age={chainlink.get('age_ms', 0):.0f}ms"
        )

        opp = self.create_opportunity(
            title=title,
            description=description,
            total_cost=vwap_price,
            expected_payout=win_prob,
            markets=[typed_market],
            positions=[
                {
                    "action": "BUY",
                    "outcome": side,
                    "price": vwap_price,
                    "token_id": token_id,
                    "_distance_edge_context": {
                        "asset": asset,
                        "timeframe": "15min",
                        "reference_price": reference,
                        "spot_price": spot,
                        "distance_usd": distance_usd,
                        "tier_distance_usd": tier["distance_usd"],
                        "tier_min_cost_cents": min_cost_cents,
                        "cost_cents": cost_cents,
                        "vwap_price": vwap_price,
                        "vwap_slippage_bps": float(depth.get("slippage_bps") or 0.0),
                        "taker_fee_frac": fee_frac,
                        "oracle_source": chainlink.get("source"),
                        "oracle_age_ms": float(chainlink.get("age_ms") or 0.0),
                        "seconds_left": seconds_left,
                        "side": side,
                        "bet_size_usd": bet_size_usd,
                        "win_prob_estimate": win_prob,
                    },
                }
            ],
            is_guaranteed=False,
            skip_fee_model=True,
            custom_roi_percent=roi_percent,
            custom_risk_score=1.0 - win_prob,
            confidence=win_prob,
        )
        if opp is None:
            emit_evaluation_nowait(
                strategy_slug="crypto_distance_edge",
                market=market,
                gates=gates,
                outcome="rejected",
                verbosity=MURMUR,
                extra={"reason": "create_opportunity returned None"},
            )
            return None

        # Emitted — record the cycle so we don't re-fire until rollover.
        self._emitted_cycle_end_ms[market_id] = end_ms_value

        emit_emit_nowait(
            strategy_slug="crypto_distance_edge",
            market=market,
            detail=(
                f"{asset} {side} • dist=${distance_usd:.2f} • "
                f"cost={cost_cents:.2f}¢ • oracle_age={chainlink.get('age_ms', 0):.0f}ms"
            ),
            extra={
                "side": side,
                "asset": asset,
                "distance_usd": distance_usd,
                "cost_cents": cost_cents,
                "bet_size_usd": bet_size_usd,
            },
        )
        emit_evaluation_nowait(
            strategy_slug="crypto_distance_edge",
            market=market,
            gates=gates,
            outcome="emitted",
            verbosity=WHISPER,
        )

        opp.risk_factors = [
            f"Crypto distance-edge continuation ({asset} 15m)",
            f"Distance from strike: ${distance_usd:.2f}",
            f"Entry cost: {cost_cents:.2f}¢ (tier floor {min_cost_cents:.0f}¢, ceiling {max_cost_cents:.0f}¢)",
            f"Taker fee: {fee_frac * 100.0:.3f}% of notional",
            f"Chainlink age: {chainlink.get('age_ms', 0):.0f}ms",
        ]
        opp.strategy_context = {
            "source_key": "crypto",
            "strategy": "crypto_distance_edge",
            "asset": asset,
            "timeframe": "15min",
            "reference_price": reference,
            "spot_price": spot,
            "distance_usd": distance_usd,
            "tier_distance_usd": tier["distance_usd"],
            "tier_min_cost_cents": min_cost_cents,
            "cost_cents": cost_cents,
            "vwap_price": vwap_price,
            "taker_fee_frac": fee_frac,
            "side": side,
            "oracle_source": chainlink.get("source"),
            "oracle_age_ms": float(chainlink.get("age_ms") or 0.0),
            "seconds_left": seconds_left,
            "bet_size_usd": bet_size_usd,
            "win_prob_estimate": win_prob,
        }
        return opp
