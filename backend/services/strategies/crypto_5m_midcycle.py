"""Crypto 5m midcycle continuation strategy.

At the 2.5-minute mark of each Polymarket 5-minute crypto over-under
cycle (BTC / ETH / SOL / XRP), this strategy observes which side is
currently winning by checking the Chainlink oracle's distance from
the cycle's reference price. If the move is large enough (default
≥ 5 bps) AND the live Polymarket book lets us buy the winning side
cheaply enough (default ≤ 70¢ VWAP entry), we enter a $15 shadow /
live position and hold to resolution.

This is "observe, don't predict" — by 150s in, the price has had
half a cycle to establish a direction, and large existing moves tend
to persist. Empirically validated at ~80% win rate in a 78-trade
session, profitable when entry prices are ≤ 70¢ (the cheap-entry zone).

All gates are user-editable via the strategy-manager UI — see the
seed entry in ``opportunity_strategy_catalog.py`` for the
``config_schema``.

Resolution truth source
-----------------------
Polymarket resolves these 5-minute markets against Chainlink
BTC/USD (and ETH/USD, SOL/USD, XRP/USD) Data Streams — NOT Binance
spot. We deliberately track Chainlink (via
``StrategySDK.crypto.pick_oracle_source(prefer="chainlink")``) so the
"is the price above strike?" check matches what Polymarket itself
will do at resolution.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from utils.utcnow import utcnow  # replay-clock-aware "now" (honors backtest sim time)
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
from services.strategy_helpers.cycle_tracker import CycleTracker
from services.strategy_helpers.crypto_strategy_utils import (
    pick_oracle_source,
)
from services.strategy_sdk import StrategySDK
from utils.converters import to_float
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Defaults — every value is overridable via the DB ``config`` column.
# ---------------------------------------------------------------------------

_ALL_ASSETS: tuple[str, ...] = ("BTC", "ETH", "SOL", "XRP")

DEFAULT_CONFIG: dict[str, Any] = {
    # Per-asset enable list. Default ships SOL + XRP only — the report
    # this strategy is based on found BTC midcycle was -$15 and ETH
    # was -$179 over comparable trade counts. Users can re-enable BTC
    # / ETH in the UI to test.
    "assets": ["SOL", "XRP"],
    # Minimum |distance from reference| in bps required to fire.
    # Below this the direction is too uncertain and entries sit in the
    # "70-80¢ trap" where wins are too small to cover the -$15 losses.
    #
    # 2026-05-05 live data: 5-bps threshold produced 5 trades, 3 of which
    # resolved against (-$17.28) vs 2 wins (+$3.74) — net -$13.53. 5 bps
    # at typical asset prices is too small to be a directional signal:
    #   BTC @ $80k → 5 bps = $40 (noise)
    #   ETH @ $3k  → 5 bps = $1.50 (well within tick noise)
    # Raised to 15 bps so the strategy fires only on more decisive
    # mid-cycle moves: BTC $120, ETH $4.50, SOL $0.30. Tunable via UI.
    "min_distance_bps": 15.0,
    # VWAP entry-price ceiling. The report's data is unambiguous: 60-70¢
    # entries are +$35.87 (8 trades), 70-80¢ are -$45.24 (23 trades).
    "max_entry_price": 0.70,
    # Skip degenerate fills.
    "min_entry_price": 0.05,
    # Notional per trade (USD).
    "bet_size_usd": 15.0,
    # The midcycle milestone, in seconds since cycle start. 150s = 2:30
    # into a 5:00 cycle. Configurable so users can experiment with
    # earlier (e.g. 120s) or later (e.g. 180s) entries.
    "midcycle_seconds": 150.0,
    # Don't trade if the cycle has less than this much time remaining.
    # Belt-and-suspenders against firing the milestone late.
    "min_seconds_to_resolution": 90.0,
    # Reject Chainlink readings older than this. Chainlink heartbeats
    # at ~250ms-2s; we want a fresh reading at the decision moment.
    "max_oracle_age_ms": 5000,
    # Master switch (also exposed at the row level via Strategy.enabled).
    "enabled": True,
}


def crypto_5m_midcycle_config_schema() -> dict[str, Any]:
    """Return the param-fields schema for the strategy-manager UI.

    Mirrors the convention used by other crypto strategies — see
    the per-strategy ``config_schema`` declared in opportunity_strategy_catalog for reference. Each field becomes
    an editable input in the strategy detail panel.
    """
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
                "label": "Assets",
                "type": "list",
                "options": list(_ALL_ASSETS),
                "default": ["SOL", "XRP"],
                "phase": "signal",
            },
            {
                "key": "min_distance_bps",
                "label": "Min |Distance from Reference| (bps)",
                "type": "number",
                "min": 0.0,
                "max": 1000.0,
                "default": 15.0,
                "phase": "signal",
            },
            {
                "key": "max_entry_price",
                "label": "Max VWAP Entry Price",
                "type": "number",
                "min": 0.0,
                "max": 1.0,
                "default": 0.70,
                "phase": "execution",
            },
            {
                "key": "min_entry_price",
                "label": "Min VWAP Entry Price",
                "type": "number",
                "min": 0.0,
                "max": 1.0,
                "default": 0.05,
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
                "key": "midcycle_seconds",
                "label": "Midcycle Milestone (sec since cycle start)",
                "type": "number",
                "min": 1.0,
                "max": 299.0,
                "default": 150.0,
                "phase": "signal",
            },
            {
                "key": "min_seconds_to_resolution",
                "label": "Min Seconds to Resolution",
                "type": "number",
                "min": 0.0,
                "max": 300.0,
                "default": 90.0,
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


_FIVE_MIN_SLUG_RE = re.compile(r"(?:^|[-_])5(?:min|m|-minute)(?:[-_]|$)")


def _detect_5m(market: dict[str, Any]) -> bool:
    """True when the market dict represents a 5-minute crypto cycle."""
    timeframe = str(market.get("timeframe") or "").strip().lower()
    if timeframe in ("5m", "5min", "5-minute", "5 minute", "5 min"):
        return True
    # Fall back to slug regex — Polymarket's slugs encode the cadence.
    # Word-boundary anchored so "15min" / "55min" don't match "5min".
    slug = str(market.get("slug") or "").lower()
    return bool(_FIVE_MIN_SLUG_RE.search(slug))


def _build_market(d: dict[str, Any]) -> Market:
    """Deserialize a crypto-worker market dict to a typed ``Market``.

    Mirrors ``BtcEthConvergenceStrategy._market_from_crypto_dict`` so
    every downstream code path sees a typed Market — the SDK orderbook
    helpers expect ``.clob_token_ids`` and ``.outcome_prices`` shapes.
    """
    market_id = str(d.get("condition_id") or d.get("id") or "")
    up_price = float(d.get("up_price") or 0.0)
    down_price = float(d.get("down_price") or 0.0)
    liquidity = max(0.0, float(d.get("liquidity") or 0.0))
    slug = d.get("slug") or market_id
    question = d.get("question") or slug

    end_date = None
    end_time_raw = d.get("end_time")
    if isinstance(end_time_raw, str) and end_time_raw.strip():
        try:
            from datetime import datetime as _dt
            end_date = _dt.fromisoformat(end_time_raw.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass

    raw_token_ids = d.get("clob_token_ids") or []
    clob_token_ids = [
        str(t).strip()
        for t in raw_token_ids
        if str(t).strip() and len(str(t).strip()) > 20
    ]

    return Market(
        id=market_id,
        condition_id=market_id,
        question=question,
        slug=slug,
        outcome_prices=[up_price, down_price],
        liquidity=liquidity,
        end_date=end_date,
        platform="polymarket",
        clob_token_ids=clob_token_ids,
    )


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------


class Crypto5mMidcycleStrategy(BaseStrategy):
    """Observe-and-continue at the 2:30 mark of each 5m crypto cycle."""

    strategy_type = "crypto_5m_midcycle"
    name = "Crypto 5m Midcycle"
    description = (
        "At the 2:30 mark of each 5-minute crypto over-under cycle, bet on "
        "continuation when the Chainlink oracle has moved at least N bps "
        "from the cycle reference and the Polymarket book lets us buy the "
        "winning side at or below the configured VWAP entry ceiling."
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
        # Per-market CycleTracker — fires the midcycle milestone exactly
        # once per cycle and self-resets on cycle rollover.
        self._cycle_trackers: dict[str, CycleTracker] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def configure(self, config: dict) -> None:
        merged = dict(self.default_config)
        if config:
            merged.update(config)
        # Sanitize: assets list normalized to upper-case canonical names
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
        now_ms = int(utcnow().timestamp() * 1000)
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
        # Collect every gate so we can emit one structured evaluation
        # summary at the end (or on early reject).  Gates short-circuit
        # — once one fails, later ones are recorded as ``passed=None``
        # ("not evaluated").
        gates: list[GateResult] = []
        midcycle_s = float(self.config.get("midcycle_seconds", 150.0))
        min_left_cfg = float(self.config.get("min_seconds_to_resolution", 90.0))
        min_distance_bps = float(self.config.get("min_distance_bps", 15.0))
        max_age_ms = float(self.config.get("max_oracle_age_ms", 5000))
        max_entry = float(self.config.get("max_entry_price", 0.70))
        min_entry = float(self.config.get("min_entry_price", 0.05))

        def _emit_reject(verbosity: str = MURMUR) -> None:
            emit_evaluation_nowait(
                strategy_slug="crypto_5m_midcycle",
                market=market,
                gates=gates,
                outcome="rejected",
                verbosity=verbosity,
            )

        # Gate 1: 5-minute timeframe.  WHISPER tier — every non-5m
        # market that walks past the door.
        is_5m = _detect_5m(market)
        gates.append(GateResult(
            "timeframe", "5-minute timeframe", is_5m,
            detail=f"timeframe={market.get('timeframe') or '?'} slug={market.get('slug') or '?'}",
        ))
        if not is_5m:
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

        # Gate 3: midcycle milestone crossed.  This fires once per cycle
        # — most ticks of crypto_update don't cross it, so most
        # evaluations rest here.  WHISPER only.
        tracker = self._cycle_trackers.get(market_id)
        if tracker is None or tracker.cycle_seconds != 300.0:
            tracker = CycleTracker(cycle_seconds=300.0, milestones_s=(midcycle_s,))
            self._cycle_trackers[market_id] = tracker
        crossed = tracker.crossed(end_ms_value, now_ms=now_ms)
        seconds_into_cycle = max(0.0, 300.0 - (end_ms_value - now_ms) / 1000.0)
        milestone_passed = midcycle_s in crossed
        gates.append(GateResult(
            "midcycle_crossed", "Midcycle milestone crossed", milestone_passed,
            score=seconds_into_cycle,
            detail=f"milestone={midcycle_s:.0f}s elapsed={seconds_into_cycle:.1f}s",
        ))
        if not milestone_passed:
            _emit_reject(WHISPER)
            return None

        # Past the milestone — every gate from here is MURMUR or higher
        # because we have a real candidate.
        seconds_left = (end_ms_value - now_ms) / 1000.0
        min_left_passed = seconds_left >= min_left_cfg
        gates.append(GateResult(
            "min_seconds_to_resolution", "Min seconds to resolution",
            min_left_passed, score=seconds_left,
            detail=f"left={seconds_left:.1f}s min={min_left_cfg:.1f}s",
        ))
        if not min_left_passed:
            _emit_reject(MURMUR)
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

        # Distance gate.  Epsilon admits exact-threshold hits that come
        # in a hair short due to float arithmetic.
        distance_bps = (spot - reference) / reference * 10_000.0
        distance_passed = abs(distance_bps) + 1e-9 >= min_distance_bps
        gates.append(GateResult(
            "min_distance", "Min distance from reference", distance_passed,
            score=distance_bps,
            detail=f"distance={distance_bps:+.2f}bps min={min_distance_bps:.2f}bps",
        ))
        if not distance_passed:
            _emit_reject(MURMUR)
            return None

        side = "YES" if distance_bps > 0 else "NO"

        typed_market = _build_market(market)
        token_ids_present = bool(typed_market.clob_token_ids)
        gates.append(GateResult(
            "clob_tokens", "CLOB token ids present", token_ids_present,
            detail=f"count={len(typed_market.clob_token_ids)}",
        ))
        if not token_ids_present:
            _emit_reject(MURMUR)
            return None

        bet_size_usd = float(self.config.get("bet_size_usd", 15.0))
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
        vwap_passed = vwap_price > 0.0 and min_entry <= vwap_price <= max_entry
        gates.append(GateResult(
            "vwap_in_range", "VWAP within entry range", vwap_passed,
            score=vwap_price,
            detail=f"vwap={vwap_price:.4f} range=[{min_entry:.2f},{max_entry:.2f}]",
        ))
        if not vwap_passed:
            _emit_reject(MURMUR)
            return None

        # All gates passed — build the Opportunity.
        edge_per_share = 1.0 - vwap_price  # max possible profit per share
        edge_percent = (edge_per_share / vwap_price) * 100.0 if vwap_price > 0 else 0.0
        # Expected payout reflects the report's empirical 80% win rate
        # at this filter strength, NOT a guaranteed $1.
        win_prob_estimate = 0.80
        expected_payout = win_prob_estimate * 1.0  # Polymarket pays $1 per winning share
        token_id = typed_market.clob_token_ids[0 if side == "YES" else 1]

        slug = str(market.get("slug") or market_id)
        title = f"Crypto 5m midcycle: {slug} {side}"
        description = (
            f"midcycle continuation | {asset} 5m | "
            f"distance={distance_bps:+.2f}bps | "
            f"VWAP entry={vwap_price:.4f} | "
            f"oracle_age={chainlink.get('age_ms', 0):.0f}ms"
        )

        opp = self.create_opportunity(
            title=title,
            description=description,
            total_cost=vwap_price,
            expected_payout=expected_payout,
            markets=[typed_market],
            positions=[
                {
                    "action": "BUY",
                    "outcome": side,
                    "price": vwap_price,
                    "token_id": token_id,
                    "_midcycle_context": {
                        "asset": asset,
                        "timeframe": "5min",
                        "reference_price": reference,
                        "spot_price": spot,
                        "distance_bps": distance_bps,
                        "vwap_price": vwap_price,
                        "vwap_slippage_bps": float(depth.get("slippage_bps") or 0.0),
                        "oracle_source": chainlink.get("source"),
                        "oracle_age_ms": float(chainlink.get("age_ms") or 0.0),
                        "seconds_left": seconds_left,
                        "side": side,
                        "bet_size_usd": bet_size_usd,
                        "win_prob_estimate": win_prob_estimate,
                    },
                }
            ],
            is_guaranteed=False,
            skip_fee_model=True,
            custom_roi_percent=edge_percent * win_prob_estimate
            - (1.0 - win_prob_estimate) * 100.0,
            custom_risk_score=1.0 - win_prob_estimate,
            confidence=win_prob_estimate,
        )
        if opp is None:
            emit_evaluation_nowait(
                strategy_slug="crypto_5m_midcycle",
                market=market,
                gates=gates,
                outcome="rejected",
                verbosity=MURMUR,
                extra={"reason": "create_opportunity returned None"},
            )
            return None

        # All gates passed and Opportunity built — VOICE tier.
        emit_emit_nowait(
            strategy_slug="crypto_5m_midcycle",
            market=market,
            detail=(
                f"{asset} {side} • dist={distance_bps:+.2f}bps • "
                f"vwap={vwap_price:.4f} • oracle_age={chainlink.get('age_ms', 0):.0f}ms"
            ),
            extra={
                "side": side,
                "asset": asset,
                "distance_bps": distance_bps,
                "vwap_price": vwap_price,
                "bet_size_usd": bet_size_usd,
            },
        )
        emit_evaluation_nowait(
            strategy_slug="crypto_5m_midcycle",
            market=market,
            gates=gates,
            outcome="emitted",
            verbosity=WHISPER,
        )

        opp.risk_factors = [
            f"Crypto 5m midcycle continuation ({asset})",
            f"Distance from strike: {distance_bps:+.2f} bps",
            f"VWAP entry: {vwap_price:.4f} (max allowed: {max_entry:.2f})",
            f"Chainlink age: {chainlink.get('age_ms', 0):.0f}ms",
        ]
        opp.strategy_context = {
            "source_key": "crypto",
            "strategy": "crypto_5m_midcycle",
            "asset": asset,
            "timeframe": "5min",
            "reference_price": reference,
            "spot_price": spot,
            "distance_bps": distance_bps,
            "vwap_price": vwap_price,
            "side": side,
            "oracle_source": chainlink.get("source"),
            "oracle_age_ms": float(chainlink.get("age_ms") or 0.0),
            "seconds_left": seconds_left,
            "bet_size_usd": bet_size_usd,
            "win_prob_estimate": win_prob_estimate,
        }
        return opp
