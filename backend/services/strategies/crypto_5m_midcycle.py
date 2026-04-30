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
import time
from typing import Any, Optional

from models import Market, Opportunity
from services.data_events import DataEvent
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
    "min_distance_bps": 5.0,
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
                "default": 5.0,
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
        now_ms = int(time.time() * 1000)
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
        # Filter: 5-minute timeframe only.
        if not _detect_5m(market):
            return None

        market_id = str(market.get("condition_id") or market.get("id") or "")
        if not market_id:
            return None

        # Filter: per-asset enable list.
        asset = _normalize_asset(
            market.get("asset") or market.get("symbol") or market.get("coin")
        )
        if not asset:
            return None
        if asset not in (self.config.get("assets") or []):
            return None

        end_ms_value = StrategySDK._coerce_end_ts_ms(market)
        if end_ms_value is None:
            return None

        # Trigger: did we just cross the midcycle milestone?
        midcycle_s = float(self.config.get("midcycle_seconds", 150.0))
        tracker = self._cycle_trackers.get(market_id)
        if tracker is None or tracker.cycle_seconds != 300.0:
            tracker = CycleTracker(cycle_seconds=300.0, milestones_s=(midcycle_s,))
            self._cycle_trackers[market_id] = tracker
        crossed = tracker.crossed(end_ms_value, now_ms=now_ms)
        if midcycle_s not in crossed:
            return None

        # Belt-and-suspenders: reject if cycle is too close to resolution.
        seconds_left = (end_ms_value - now_ms) / 1000.0
        min_left = float(self.config.get("min_seconds_to_resolution", 90.0))
        if seconds_left < min_left:
            logger.debug(
                "crypto_5m_midcycle: %s skipped — only %.1fs left (min=%.1f)",
                market.get("slug") or market_id, seconds_left, min_left,
            )
            return None

        # Reference + current price — Chainlink only. Polymarket resolves
        # against Chainlink, so any other source could pick the wrong side
        # at the boundary.
        reference = to_float(market.get("price_to_beat"), None)
        if reference is None or reference <= 0.0:
            return None

        max_age_ms = float(self.config.get("max_oracle_age_ms", 5000))
        chainlink = pick_oracle_source(
            market, prefer="chainlink", max_age_ms=max_age_ms, now_ms=now_ms
        )
        # ``pick_oracle_source`` falls back to the freshest non-preferred
        # source when chainlink isn't available — but Polymarket resolves
        # specifically on Chainlink, so we treat any non-chainlink pick
        # as "no usable price." Better to skip than to pick the wrong
        # side at the resolution boundary.
        if chainlink is None or str(chainlink.get("source", "")).lower() != "chainlink":
            logger.debug(
                "crypto_5m_midcycle: %s skipped — no fresh Chainlink price (got %s)",
                market.get("slug") or market_id,
                chainlink.get("source") if chainlink else None,
            )
            return None
        spot = float(chainlink["price"])
        if spot <= 0.0:
            return None

        # Distance gate. Use a small epsilon to admit values that hit the
        # threshold exactly but fall a hair short due to float arithmetic
        # (e.g. (100.05 - 100.0) / 100.0 * 10000 ≈ 4.99999... rather than 5.0).
        distance_bps = (spot - reference) / reference * 10_000.0
        min_distance_bps = float(self.config.get("min_distance_bps", 5.0))
        if abs(distance_bps) + 1e-9 < min_distance_bps:
            logger.debug(
                "crypto_5m_midcycle: %s skipped — distance %.2f bps < min %.2f",
                market.get("slug") or market_id, distance_bps, min_distance_bps,
            )
            return None

        # Pick winning side: spot > reference → UP wins → buy YES (UP token).
        side = "YES" if distance_bps > 0 else "NO"

        # VWAP fill check against the live Polymarket book.
        typed_market = _build_market(market)
        if not typed_market.clob_token_ids:
            return None

        bet_size_usd = float(self.config.get("bet_size_usd", 15.0))
        depth = StrategySDK.get_order_book_depth(
            typed_market, side=side, size_usd=bet_size_usd
        )
        if depth is None:
            logger.debug(
                "crypto_5m_midcycle: %s skipped — no book depth for %s",
                market.get("slug") or market_id, side,
            )
            return None
        if not depth.get("is_fresh", False):
            logger.debug(
                "crypto_5m_midcycle: %s skipped — stale book (age=%sms)",
                market.get("slug") or market_id, depth.get("staleness_ms"),
            )
            return None

        vwap_price = float(depth.get("vwap_price") or 0.0)
        max_entry = float(self.config.get("max_entry_price", 0.70))
        min_entry = float(self.config.get("min_entry_price", 0.05))
        if vwap_price <= 0.0 or vwap_price > max_entry or vwap_price < min_entry:
            logger.debug(
                "crypto_5m_midcycle: %s skipped — VWAP %.4f outside [%.2f, %.2f]",
                market.get("slug") or market_id, vwap_price, min_entry, max_entry,
            )
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
            return None

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
