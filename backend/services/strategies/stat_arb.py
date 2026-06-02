"""
Strategy: Statistical Arbitrage / Information Edge

Compares Polymarket prices against external probability signals
to find markets where the crowd is wrong via a weighted ensemble of
weak signals (anchoring, category base rate, multi-market consensus,
momentum, volume-price divergence, favorite-longshot, liquidity
imbalance) producing a composite fair-probability estimate.

NOT risk-free. This is informed speculation with statistical edge.

Two ex-sub-strategies (certainty_shock, temporal_decay) used to live
inline in this file. They have been split into their own strategy
modules so their performance can be tracked independently — both were
weak performers in production.
"""

from __future__ import annotations

import re
import statistics
from typing import Any, Optional

import logging

from models import Market, Event, Opportunity
from config import settings
from .base import BaseStrategy, DecisionCheck, ExitDecision, ScoringWeights, SizingConfig, make_aware, utcnow
from services.quality_filter import QualityFilterOverrides
from services.strategy_sdk import StrategySDK
from utils.kelly import polymarket_taker_fee, kalshi_taker_fee

logger = logging.getLogger(__name__)


# Round numbers where human anchoring bias is common
ANCHOR_PRICES = [0.10, 0.25, 0.33, 0.50, 0.67, 0.75, 0.90]

# Anchoring detection tolerance: if price is within this of a round number,
# it is considered anchored
# Category base rates are sourced from
# ``StrategySDK.get_category_yes_resolve_rates()``, which combines an
# empirical JSON cache (refreshed by
# scripts/refresh_category_yes_rates.py) with a hardcoded baseline.
#
# Anchoring tolerance and signal weights live in ``default_config`` so
# they're tunable from the UI, params-mode autoresearch, and code-mode
# autoresearch. The list of round-number anchor targets above stays
# module-level because adding/removing entries changes the algorithm's
# shape, not its calibration.

_PLAYER_STAT_RE = re.compile(r"[A-Z][a-z]+ [A-Z][a-z]+:\s*\d+\+", re.IGNORECASE)

# Categories on the parent Event whose markets are bookmaker-efficient and
# unsuitable for ensemble-statistical alpha. Bookmakers price these tighter
# than our category-base-rate / anchoring / longshot signals can detect.
_SPORTS_CATEGORIES = frozenset({
    "sports", "soccer", "football", "tennis", "basketball", "baseball", "hockey",
    "esports", "e-sports", "mma", "boxing", "golf", "cricket", "racing",
    "horse racing", "motorsports", "motorsport", "formula 1", "f1", "nascar",
    "ufc", "rugby", "darts", "snooker", "volleyball", "handball",
})

# Polymarket/Kalshi slug prefixes for sport leagues + event series. The slug
# carries league context even when the question text doesn't — e.g.
# "Miami Marlins vs. Los Angeles Dodgers" has slug "mlb-mia-lad-2026-04-29".
_SPORTS_SLUG_PREFIXES = (
    "epl-", "ucl-", "uel-", "uefa-", "fifwc-", "fifa-", "wc-", "world-cup-",
    "euro-", "copa-", "la-liga-", "laliga-", "bundesliga-", "bl1-", "bl2-",
    "serie-a-", "seria-", "ligue-1-", "ligue1-", "ere-", "eredivisie-",
    "premier-league-", "champions-league-", "mls-", "usl-",
    "mlb-", "nba-", "nfl-", "nhl-", "ncaa-", "ncaab-", "ncaaf-",
    "march-madness-", "atp-", "wta-", "itf-", "ufc-", "pfl-", "bellator-",
    "lol-", "lpl-", "lck-", "lec-", "lcs-", "msi-", "worlds-",
    "kbo-", "npb-", "cpbl-", "kbl-",
    "valorant-", "csgo-", "cs2-", "dota-", "rocket-league-", "rl-",
    "f1-", "indycar-", "nascar-", "motogp-",
    "set-1-", "set-2-", "set-3-", "game-1-", "game-2-", "game-3-",
    "map-1-", "map-2-", "map-3-",
)

# Question text patterns that scream "match-up market". " vs. " / " vs " is
# the strongest single signal — true cross-event statistical alpha would
# almost never come through a head-to-head question.
_SPORTS_QUESTION_PATTERNS = (
    " vs. ", " vs ", "end in a draw", "game 1 winner", "game 2 winner",
    "game 3 winner", "game 4 winner", "game 5 winner",
    "map 1 winner", "map 2 winner", "map 3 winner",
    "set 1 winner", "set 2 winner", "set 3 winner",
    "first half", "second half", "halftime", "half-time", "1st half", "2nd half",
    "first set", "first game", "first map", "first round",
    "match winner", "moneyline", "handicap", "spread",
    "total goals", "total points", "total kills", "total sets", "total runs",
    "anytime scorer", "first scorer", "last scorer", "btts", "both teams",
    "to lift the trophy", "to win the trophy", "race to ",
    "o/u ", "over/under", "win on 20", "win on 21",  # "Will X win on 2026-..."
    # Tournament-bracket / sport-tournament Y/N markets — bookmakers price
    # these tightly even when there's no head-to-head "vs." in the question.
    "reach the final", "reach the semifinal", "reach the semi-final",
    "reach the semi finals", "reach the quarterfinal", "reach the quarter-final",
    "reach the quarter finals", "advance to", "advance from",
    "world cup", "fifa", "euro 2026", "uefa euro",
    "champions league", "conference league", "europa league",
    "stanley cup", "world series", "super bowl", "ncaa",
    "olympics", "olympic", "wimbledon", "us open", "french open", "australian open",
    "masters", "the masters",
)

# Open-ended / award markets where the outcome universe is unbounded
# (Nobel, Eurovision, Oscars, anime awards, etc.). The strategy's
# category-base-rate and consensus signals can't calibrate against an
# unbounded outcome set.
_OPEN_ENDED_QUESTION_PATTERNS = (
    "eurovision", "oscar", "grammy", "emmy", "ballon d'or", "mvp",
    "nobel", "pulitzer", "anime award", "crunchyroll",
    "best picture", "best actor", "best actress", "best director",
    "song of the year", "album of the year",
    "person of the year", "time person",
    # M&A / acquisition markets - any company could acquire
    "will acquire", "will buy", "who will purchase", "takeover bid",
)

_MULTILEG_MARKET_PREFIXES = ("KXMVESPORTSMULTIGAMEEXTENDED-",)


class StatArbStrategy(BaseStrategy):
    """
    Ensemble statistical mispricing strategy.

    Combines seven weak signals (anchoring, category base rate, multi-market
    consensus, momentum, volume-price divergence, favorite-longshot,
    liquidity imbalance) into a composite fair-probability estimate, then
    trades when |fair − market| > min_edge.

    Sports / head-to-head match markets are rejected up front because
    bookmakers price these tighter than the ensemble can detect.
    """

    strategy_type = "stat_arb"
    name = "Statistical Arbitrage"
    description = "Ensemble fair-probability vs market mispricing"
    mispricing_type = "within_market"
    subscriptions = ["market_data_refresh"]

    quality_filter_overrides = QualityFilterOverrides(
        min_roi=2.0,
        max_resolution_months=3.0,
    )

    default_config = {
        "min_edge_percent": 5.0,
        "min_confidence": 0.45,
        "max_risk_score": 0.75,
        "enable_stat_signals": True,
        "exclude_market_keywords": [
            "bitcoin",
            "btc",
            "ethereum",
            "eth",
            "solana",
            "sol",
            "xrp",
            "crypto",
            "up or down",
            "doge",
            "dogecoin",
            "bnb",
            "cardano",
            "ada",
            "polygon",
            "matic",
            "avax",
            "avalanche",
            "chainlink",
            "link",
            "litecoin",
            "ltc",
        ],
        # Stop-loss must be tighter than take-profit. The previous setting
        # (TP=12 / SL=25) demanded a >67% hit rate to break even, which the
        # ensemble does not deliver. Flipping the ratio so a 6% adverse move
        # exits before a 12% favorable move is required to keep R:R ≥ 1.
        "take_profit_pct": 12.0,
        "stop_loss_pct": 6.0,
        "trailing_stop_pct": 8.0,
        # Anchoring tolerance: prices within this distance of a round number
        # (10/25/33/50/67/75/90 ¢) count as "anchored" for the anchoring
        # signal. 2¢ is conservative — wider tolerance fires the signal more
        # often.
        "anchor_tolerance": 0.02,
        # Ensemble signal weights. Each of the 7 weak signals is in [-1, 1];
        # the composite is sum(weight × signal) / sum(weights), then scaled
        # by ``signal_adjustment_scale`` to a max ±0.15 price-space shift.
        # Tuned via UI / params-mode autoresearch / code-mode autoresearch.
        # Names must match the keys produced by ``_compute_fair_probability``;
        # any missing key is treated as weight 0.
        "signal_weights": {
            "anchoring": 0.13,
            "category_base_rate": 0.18,
            "consensus": 0.20,
            "momentum": 0.12,
            "volume_price": 0.15,
            "favorite_longshot": 0.10,
            "liquidity_imbalance": 0.12,
        },
        # Maximum composite adjustment to the market price, after weighted
        # ensemble. 0.15 = ±15¢ in price space.
        "signal_adjustment_scale": 0.15,
        # Momentum window (rolling) and lookback for the log-return.
        "momentum_window_seconds": 600.0,
        "momentum_lookback_seconds": 300.0,
        # Confidence-derivation knobs (drive
        # StrategySDK.confidence_from_signal_agreement).
        "confidence_floor": 0.30,
        "confidence_ceiling": 0.85,
        "confidence_agreement_weight": 0.6,
    }

    pipeline_defaults = {
        "min_edge_percent": 3.5,
        "min_confidence": 0.45,
        "max_risk_score": 0.75,
    }

    # Composable evaluate pipeline: score = edge*0.58 + conf*32 - risk*8
    scoring_weights = ScoringWeights(
        edge_weight=0.58,
        confidence_weight=32.0,
        risk_penalty=8.0,
    )
    # size = base*(1+edge/100)*(0.75+conf), no risk/market scaling
    sizing_config = SizingConfig(
        base_divisor=100.0,
        confidence_offset=0.75,
        risk_scale_factor=0.0,
        risk_floor=1.0,
        market_scale_factor=0.0,
        market_scale_cap=0,
    )

    # Per-market PriceWindow registry — populated lazily as detect() observes
    # markets. The window/lookback in seconds are read from default_config so
    # they're tunable without editing source.
    def __init__(self):
        super().__init__()
        # One PriceWindow per market for the momentum signal. Persisted in
        # the strategy instance's state so windows survive across scans.
        self._momentum_windows: dict[str, StrategySDK.PriceWindow] = {}

    @staticmethod
    def _normalize_excluded_keywords(value: Any) -> list[str]:
        if isinstance(value, str):
            candidates = [token.strip() for token in value.split(",")]
        elif isinstance(value, list):
            candidates = list(value)
        else:
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in candidates:
            token = str(raw or "").strip().lower()
            if not token or token in seen:
                continue
            seen.add(token)
            normalized.append(token)
        return normalized

    @staticmethod
    def _keyword_in_text(keyword: str, text: str) -> bool:
        if not keyword or not text:
            return False
        if len(keyword) <= 4 and keyword.replace("-", "").replace("_", "").isalnum():
            return re.search(rf"\b{re.escape(keyword)}\b", text) is not None
        return keyword in text

    @classmethod
    def _first_blocked_keyword(cls, text: str, excluded_keywords: list[str]) -> Optional[str]:
        for keyword in excluded_keywords:
            if cls._keyword_in_text(keyword, text):
                return keyword
        return None

    @classmethod
    def _market_text(cls, market: Market) -> str:
        chunks: list[str] = []
        for value in (
            market.id,
            market.question,
            getattr(market, "slug", None),
            getattr(market, "group_item_title", None),
            getattr(market, "event_slug", None),
        ):
            text = str(value or "").strip().lower()
            if text:
                chunks.append(text)
        for tag in list(getattr(market, "tags", []) or []):
            text = str(tag or "").strip().lower()
            if text:
                chunks.append(text)
        return " | ".join(chunks)

    @classmethod
    def _signal_market_text(cls, signal: Any, payload: dict) -> str:
        chunks: list[str] = []
        for value in (
            getattr(signal, "market_question", None),
            payload.get("market_question"),
            payload.get("title"),
            payload.get("description"),
            payload.get("market_id"),
        ):
            text = str(value or "").strip().lower()
            if text:
                chunks.append(text)
        markets = payload.get("markets")
        if isinstance(markets, list):
            for raw_market in markets[:2]:
                if not isinstance(raw_market, dict):
                    continue
                for value in (raw_market.get("id"), raw_market.get("question"), raw_market.get("slug")):
                    text = str(value or "").strip().lower()
                    if text:
                        chunks.append(text)
        return " | ".join(chunks)

    # ------------------------------------------------------------------
    # Signal 1: Anchoring Detection
    # ------------------------------------------------------------------
    def _signal_anchoring(self, yes_price: float, base_rate_hint: Optional[float] = None) -> float:
        """Detect whether a market price is anchored to a round number, AND
        infer the direction the price should drift toward.

        Markets at exactly 50%, 25%, 75% etc. are often anchored because
        humans round to convenient numbers. The previous implementation
        returned 1.0 (anchored) or 0.0 (not) — which tells the ensemble
        "anchored" but never tells it whether YES should be priced higher
        or lower. The result was that the anchoring weight contributed
        magnitude but no direction.

        Now: if anchored, return a SIGNED magnitude in [-1, 1].
          - positive => price is anchored low; expect drift up (buy YES)
          - negative => price is anchored high; expect drift down (buy NO)
          - 0.0 => not anchored

        Direction is inferred from the category base rate when available
        (price below base rate → underpriced, expect upward drift) and
        from a 50% midpoint otherwise.
        """
        tolerance = float(self.config.get("anchor_tolerance", 0.02) or 0.02)
        for anchor in ANCHOR_PRICES:
            if abs(yes_price - anchor) < tolerance:
                # Direction: drift toward base rate / mid.
                target = base_rate_hint if base_rate_hint is not None else 0.5
                if abs(target - yes_price) < tolerance:
                    # Anchor and base rate agree: no directional signal.
                    return 0.0
                # Sign: positive when price is below the anchor's "true"
                # level (suggesting upward drift) and vice versa.
                return 1.0 if target > yes_price else -1.0
        return 0.0

    # ------------------------------------------------------------------
    # Signal 2: Category Base Rate
    # ------------------------------------------------------------------
    def _signal_category_base_rate(self, yes_price: float, category: Optional[str]) -> float:
        """Compare market price to category base rate.

        If the market price is significantly above the category's
        historical YES-resolution rate the YES side may be overpriced
        (and vice versa).

        The base rate comes from
        ``StrategySDK.get_category_yes_resolve_rates()``, which prefers
        empirical numbers from the JSON cache (refreshed by
        ``scripts/refresh_category_yes_rates.py``) and falls back to a
        conservative hardcoded baseline. Strategies should not maintain
        their own ``CATEGORY_BASE_RATES`` dict — base rates are a
        cross-strategy concern.

        Returns a value in [-1, 1]:
          positive => market is under-priced (YES should be higher)
          negative => market is over-priced  (YES should be lower)
        """
        rates = StrategySDK.get_category_yes_resolve_rates()
        base_rate = rates.get("_default", 0.45)
        if category:
            base_rate = rates.get(category.lower().strip(), base_rate)
        deviation = base_rate - yes_price
        return max(-1.0, min(1.0, deviation))

    # ------------------------------------------------------------------
    # Signal 3: Multi-Market Consensus (within same event)
    # ------------------------------------------------------------------
    def _signal_consensus(
        self,
        market: Market,
        event_markets: list[Market],
        prices: dict[str, dict],
    ) -> Optional[float]:
        """Detect price outliers within an event using Median Absolute
        Deviation (MAD), excluding the target market from the consensus.

        For markets in the same event, check consistency. If event has 5
        markets and 4 imply X should be ~60% but market 5 prices X at 40%,
        market 5 is the outlier.

        Critical: the target market must NOT be part of its own consensus.
        Including it pulls the median toward the target's own price and
        understates the deviation. The previous implementation included
        the target — which made every market look closer to the median
        than it actually was. The fix is a proper leave-one-out median.

        Returns a value in [-1, 1] representing deviation from peer
        consensus, or None if consensus cannot be determined (fewer than 3
        SIBLING markets — not counting the target).
        """
        # Gather YES prices for all OTHER binary markets in the event,
        # plus the target's own price separately.
        sibling_prices: list[float] = []
        target_price: Optional[float] = None
        for m in event_markets:
            if len(m.outcome_prices) != 2:
                continue
            if m.closed or not m.active:
                continue
            p = self._live_yes_price(m, prices)
            if m.id == market.id:
                target_price = p
                continue  # leave-one-out: target is not in its own consensus
            sibling_prices.append(p)

        if target_price is None or len(sibling_prices) < 3:
            return None

        median = statistics.median(sibling_prices)
        mad = statistics.median([abs(p - median) for p in sibling_prices])
        if mad < 0.01:
            # All peer prices are essentially identical; no outlier signal
            return 0.0

        # Modified Z-score: how many MADs away is the target from the
        # peer median?
        z = (target_price - median) / mad
        # Positive z means target is above peer consensus (possibly
        # overpriced). Negative z means target is below peer consensus
        # (possibly underpriced). Normalize to [-1, 1] by capping at 3
        # MADs and flipping sign so that "underpriced" -> positive signal.
        signal = max(-1.0, min(1.0, -z / 3.0))
        return signal

    # ------------------------------------------------------------------
    # Signal 4: Momentum
    # ------------------------------------------------------------------
    def _signal_momentum(self, market_id: str, current_price: float) -> float:
        """Detect drift via a rolling 5-min log-return window.

        Backed by ``StrategySDK.PriceWindow`` so the rolling history is
        managed through the SDK's standard primitive rather than a hand-
        rolled dict-of-floats. The window persists across scans within
        the strategy instance.

        Returns a value in [-1, 1]. Positive ⇒ price has been rising
        (upward drift); negative ⇒ falling.
        """
        if not market_id or current_price <= 0:
            return 0.0
        window_seconds = float(self.config.get("momentum_window_seconds", 600.0) or 600.0)
        lookback_seconds = float(self.config.get("momentum_lookback_seconds", 300.0) or 300.0)
        window = self._momentum_windows.get(market_id)
        if window is None:
            window = StrategySDK.PriceWindow(window_seconds=window_seconds)
            self._momentum_windows[market_id] = window
        elif window.window_seconds != window_seconds:
            window.window_seconds = window_seconds
        window.record(current_price)
        log_ret = window.log_return(lookback_seconds)
        if log_ret is None:
            return 0.0
        # Scale: a 5% log-return over 5 minutes is a strong signal. Cap
        # at full magnitude when |log_ret| ≥ 0.10.
        return max(-1.0, min(1.0, log_ret * 10.0))

    # ------------------------------------------------------------------
    # Signal 5: Volume-Price Divergence
    # ------------------------------------------------------------------
    def _signal_volume_price(self, yes_price: float, volume: float, liquidity: float) -> float:
        """Detect volume-price divergence.

        Returns a continuous signal in [-1, 1] based on two factors:

          * **Stale-extreme-price**: low volume + extreme price suggests
            the price is stuck and should mean-revert toward 0.50.
          * **Active-efficient-market**: high volume + extreme price
            suggests the crowd has concluded; small fade (anti-momentum)
            in case retail is over-reacting.

        Sign:
          Positive => YES underpriced, expected to drift up (buy YES).
          Negative => YES overpriced, expected to drift down (buy NO).

        The previous implementation returned 0.0 in three of its four
        branches, so the volume-price weight contributed almost nothing
        to the ensemble. Now: a continuous output across the whole range
        of (vol_liq, price_extremity), bounded to [-1, +1].
        """
        if volume <= 0 or liquidity <= 0:
            return 0.0

        # Volume-to-liquidity ratio as a proxy for trading activity.
        vol_liq = volume / liquidity if liquidity > 0 else 0.0
        # Price distance from 0.50, scaled to [0, 1].
        price_extremity = abs(yes_price - 0.50) / 0.50  # 0 at mid, 1 at edge
        # Direction toward mid-reversion: from extreme prices, expect drift
        # toward 0.50 — buy NO when YES is high, buy YES when YES is low.
        direction = -1.0 if yes_price > 0.50 else 1.0

        # Liquidity-adjusted activity level in [0, 1]. <1 (thin trading)
        # gives high stale-score; >5 gives near-zero stale-score.
        if vol_liq <= 0.5:
            stale_score = 1.0
        elif vol_liq >= 5.0:
            stale_score = 0.0
        else:
            # Linear interpolation between the two cutoffs.
            stale_score = max(0.0, 1.0 - (vol_liq - 0.5) / 4.5)

        # Mean-reversion signal: thin volume + extreme price → strong
        # stale signal (full magnitude when both are saturated).
        # Active markets at extremes → small fade toward mid (0.20 of full).
        active_extreme_fade = 0.0
        if vol_liq > 5.0 and price_extremity > 0.5:
            active_extreme_fade = 0.20 * price_extremity

        magnitude = max(stale_score * price_extremity, active_extreme_fade)
        return max(-1.0, min(1.0, direction * magnitude))

    def _signal_favorite_longshot(self, yes_price: float) -> float:
        """Capture favorite-longshot skew as one weak component.

        Longshots (very low YES prices) are frequently overbid by retail.
        Favorites are frequently underbid. Positive values imply YES should be
        priced higher; negative values imply YES should be priced lower.
        """
        if yes_price <= 0.25:
            return -min(1.0, (0.25 - yes_price) / 0.25) * 0.8
        if yes_price >= 0.75:
            return min(1.0, (yes_price - 0.75) / 0.25) * 0.6
        return 0.0

    def _signal_liquidity_imbalance(self, market: Market, prices: dict[str, dict]) -> float:
        """Use bid/ask depth imbalance as an additional directional input."""
        token_ids = list(getattr(market, "clob_token_ids", []) or [])
        if len(token_ids) < 2:
            return 0.0

        yes_payload = prices.get(token_ids[0]) if token_ids[0] in prices else None
        no_payload = prices.get(token_ids[1]) if token_ids[1] in prices else None

        def _depth(payload: dict | None, *keys: str) -> float:
            if not isinstance(payload, dict):
                return 0.0
            for key in keys:
                value = payload.get(key)
                if isinstance(value, (int, float)) and value > 0:
                    return float(value)
            return 0.0

        yes_bid_depth = _depth(yes_payload, "bid_depth", "best_bid_size", "bid_size")
        yes_ask_depth = _depth(yes_payload, "ask_depth", "best_ask_size", "ask_size")
        no_bid_depth = _depth(no_payload, "bid_depth", "best_bid_size", "bid_size")
        no_ask_depth = _depth(no_payload, "ask_depth", "best_ask_size", "ask_size")

        total_yes = yes_bid_depth + yes_ask_depth
        total_no = no_bid_depth + no_ask_depth
        if total_yes <= 0 and total_no <= 0:
            return 0.0

        yes_pressure = 0.0
        if total_yes > 0:
            yes_pressure = (yes_bid_depth - yes_ask_depth) / total_yes
        no_pressure = 0.0
        if total_no > 0:
            no_pressure = (no_bid_depth - no_ask_depth) / total_no

        # If NO book has stronger bid pressure, YES should be lower.
        imbalance = (yes_pressure - no_pressure) / 2.0
        return max(-1.0, min(1.0, imbalance))

    # ------------------------------------------------------------------
    # Temporal awareness
    # ------------------------------------------------------------------
    _YEAR_PATTERN = re.compile(r"\b(20[0-9]{2})\b")

    def _is_likely_already_resolved(self, market: Market) -> bool:
        """Detect if a market's subject matter has likely already occurred.

        Markets about past events (e.g., "U.S. tariff revenue in 2025"
        scanned in February 2026) should not be traded on statistical
        signals, because the market price reflects actual known outcomes
        while our base-rate model uses stale historical averages.

        Also rejects markets whose resolution date has already passed.
        """
        now = utcnow()

        # Check 1: Resolution date already passed (should be filtered
        # upstream but double-check)
        if market.end_date:
            end_aware = make_aware(market.end_date)
            if end_aware < now:
                return True

        # Check 2: Question references a specific past year.
        # E.g., "How much revenue will the U.S. raise from tariffs in 2025?"
        # scanned in 2026 means the 2025 data is already known.
        question = market.question or ""
        current_year = now.year
        for match in self._YEAR_PATTERN.finditer(question):
            referenced_year = int(match.group(1))
            if referenced_year < current_year:
                return True

        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _live_yes_price(self, market: Market, prices: dict[str, dict]) -> float:
        """Return the best available YES price (live > static)."""
        yes_price = market.yes_price
        if market.clob_token_ids and len(market.clob_token_ids) > 0:
            yes_token = market.clob_token_ids[0]
            if yes_token in prices:
                yes_price = prices[yes_token].get("mid", yes_price)
        return yes_price

    def _live_no_price(self, market: Market, prices: dict[str, dict]) -> float:
        """Return the best available NO price (live > static)."""
        no_price = market.no_price
        if market.clob_token_ids and len(market.clob_token_ids) > 1:
            no_token = market.clob_token_ids[1]
            if no_token in prices:
                no_price = prices[no_token].get("mid", no_price)
        return no_price

    def _is_sports_or_match_market(self, market: Market, event: Optional[Event] = None) -> bool:
        """Reject markets where the ensemble has no edge.

        Three classes of market that fall in this bucket:

        1. **Sports / head-to-head match markets**: bookmakers price these
           tighter than the seven-signal ensemble can detect. Detected via
           event.category, slug-prefix (epl-, mlb-, lol-, ...), or question
           patterns (" vs ", "end in a draw", "moneyline", "spread", ...).

        2. **Tournament-bracket Y/N markets**: "Will Argentina reach the
           2026 FIFA World Cup final?" — no head-to-head phrasing but
           still a sports market. Detected via question patterns
           ("reach the final", "world cup", ...).

        3. **Open-ended outcome universes**: Eurovision, Nobel, Oscars,
           M&A. The ensemble's category-base-rate signal has no calibration
           target when the outcome set is unbounded.
        """
        # Layer 1: parent event category
        if event is not None:
            category = str(getattr(event, "category", "") or "").strip().lower()
            if category in _SPORTS_CATEGORIES:
                return True

        # Layer 2: slug prefix
        slug = str(getattr(market, "slug", "") or "").strip().lower()
        event_slug = str(getattr(market, "event_slug", "") or "").strip().lower()
        for prefix in _SPORTS_SLUG_PREFIXES:
            if slug.startswith(prefix) or event_slug.startswith(prefix):
                return True

        # Layer 3: question-text patterns
        question_raw = market.question or ""
        question = question_raw.lower()
        for pattern in _SPORTS_QUESTION_PATTERNS:
            if pattern in question:
                return True

        # Layer 4: open-ended outcome universes (awards, M&A, etc.) — these
        # aren't strictly "sports" but share the property that the
        # ensemble's category-base-rate signal has no calibration target.
        for pattern in _OPEN_ENDED_QUESTION_PATTERNS:
            if pattern in question:
                return True

        # Multi-leg parlay shapes: "Yes Yes No" stat lines, comma-heavy parlays,
        # "Player Name: 25+" stat thresholds.
        if question.count("yes ") + question.count("no ") >= 2:
            return True
        if question.count(",") >= 2:
            return True
        if _PLAYER_STAT_RE.search(question_raw):
            return True

        return False

    # Back-compat shim — older callers still reference _is_sports_parlay.
    def _is_sports_parlay(self, market: Market) -> bool:
        return self._is_sports_or_match_market(market, event=None)

    def _get_category(self, market: Market, events: list[Event]) -> Optional[str]:
        """Look up the category for a market via its parent event."""
        for event in events:
            for em in event.markets:
                if em.id == market.id:
                    return event.category
        return None

    def _get_event_for_market(self, market: Market, events: list[Event]) -> Optional[Event]:
        """Find the parent event for a given market."""
        for event in events:
            for em in event.markets:
                if em.id == market.id:
                    return event
        return None

    # ------------------------------------------------------------------
    # Composite Fair Probability
    # ------------------------------------------------------------------
    def _compute_fair_probability(
        self,
        market: Market,
        yes_price: float,
        events: list[Event],
        event_markets: list[Market],
        prices: dict[str, dict],
        category: Optional[str],
    ) -> tuple[float, dict[str, float]]:
        """Ensemble all signals into a composite fair probability estimate.

        Returns (fair_probability, signal_breakdown).
        """
        signals: dict[str, float] = {}

        # Compute the category base rate first so signal 1 can use it as a
        # directional hint when the price is anchored to a round number.
        # Source from the SDK (DB-backed JSON cache + hardcoded baseline)
        # rather than this strategy's local copy.
        rates = StrategySDK.get_category_yes_resolve_rates()
        base_rate = rates.get("_default", 0.45)
        if category:
            base_rate = rates.get(category.lower().strip(), base_rate)

        # Signal 1: Anchoring (now signed: directional)
        anchor = self._signal_anchoring(yes_price, base_rate_hint=base_rate)
        signals["anchoring"] = anchor

        # Signal 2: Category base rate
        cat_signal = self._signal_category_base_rate(yes_price, category)
        signals["category_base_rate"] = cat_signal

        # Signal 3: Consensus
        consensus = self._signal_consensus(market, event_markets, prices)
        if consensus is None:
            consensus = 0.0
        signals["consensus"] = consensus

        # Signal 4: Momentum
        momentum = self._signal_momentum(market.id, yes_price)
        signals["momentum"] = momentum

        # Signal 5: Volume-price divergence
        vol_price = self._signal_volume_price(yes_price, market.volume, market.liquidity)
        signals["volume_price"] = vol_price

        # Signal 6: Favorite/longshot skew
        flb_signal = self._signal_favorite_longshot(yes_price)
        signals["favorite_longshot"] = flb_signal

        # Signal 7: Liquidity imbalance (depth/pressure)
        liquidity_imbalance = self._signal_liquidity_imbalance(market, prices)
        signals["liquidity_imbalance"] = liquidity_imbalance

        # Weighted composite adjustment
        # Each signal contributes a directional shift from the market price.
        # Positive signal => fair prob higher than market (buy YES),
        # Negative signal => fair prob lower than market (buy NO).
        weights_cfg = self.config.get("signal_weights", {}) or {}
        if not isinstance(weights_cfg, dict):
            weights_cfg = {}
        scale = float(self.config.get("signal_adjustment_scale", 0.15) or 0.15)
        adjustment = 0.0
        total_weight = 0.0
        for sig_name, sig_value in signals.items():
            try:
                w = float(weights_cfg.get(sig_name, 0.0) or 0.0)
            except (TypeError, ValueError):
                w = 0.0
            adjustment += w * sig_value
            total_weight += w

        if total_weight > 0:
            adjustment /= total_weight
            # Scale: full signal moves price by up to ``signal_adjustment_scale``.
            adjustment *= scale

        fair_prob = yes_price + adjustment
        # Clamp to [0.01, 0.99]
        fair_prob = max(0.01, min(0.99, fair_prob))

        return fair_prob, signals

    # ------------------------------------------------------------------
    # detect()
    # ------------------------------------------------------------------
    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[Opportunity]:
        """Detect statistical arbitrage opportunities.

        For each binary market, calculate all signals, compute composite
        fair probability, and if the edge exceeds the configured minimum
        create an opportunity.
        """
        config = dict(self.default_config)
        config.update(getattr(self, "config", {}) or {})
        min_edge = max(0.0, float(config.get("min_edge_percent", 5.0) or 5.0) / 100.0)
        enable_stat_signals = bool(config.get("enable_stat_signals", True))
        excluded_keywords = self._normalize_excluded_keywords(config.get("exclude_market_keywords"))
        opportunities: list[Opportunity] = []

        # Pre-build market-id -> event mapping for fast lookup
        market_to_event: dict[str, Event] = {}
        for event in events:
            for em in event.markets:
                market_to_event[em.id] = event

        for market in markets:
            # Only binary markets
            if len(market.outcome_prices) != 2:
                continue

            # Skip inactive or closed
            if market.closed or not market.active:
                continue

            if excluded_keywords:
                market_text = self._market_text(market)
                if self._first_blocked_keyword(market_text, excluded_keywords) is not None:
                    continue

            market_keys = [
                str(getattr(market, "id", "") or "").upper(),
                str(getattr(market, "condition_id", "") or "").upper(),
            ]
            if any(key.startswith(prefix) for key in market_keys for prefix in _MULTILEG_MARKET_PREFIXES):
                continue

            # Look up parent event up-front so the sports/match filter and
            # downstream signals can both use it without redundant scans.
            event = market_to_event.get(market.id)

            # Reject sports / head-to-head match markets — bookmakers price
            # these tighter than this strategy's ensemble can detect, and they
            # were the dominant loss source in live trading.
            if self._is_sports_or_match_market(market, event=event):
                continue

            # Skip markets whose subject matter has already occurred.
            # Statistical base-rate signals are meaningless when the market
            # price reflects actual known outcomes (e.g., "2025 revenue"
            # scanned in 2026).
            if self._is_likely_already_resolved(market):
                continue

            # Get live prices
            yes_price = self._live_yes_price(market, prices)
            no_price = self._live_no_price(market, prices)

            # Skip extreme prices (nearly resolved, nothing to trade)
            if yes_price < 0.03 or yes_price > 0.97:
                continue

            if not enable_stat_signals:
                continue

            # event is already resolved up-front (above the sports filter)
            category = event.category if event else None
            event_markets = event.markets if event else [market]

            # Compute fair probability and signal breakdown
            fair_prob, signal_breakdown = self._compute_fair_probability(
                market=market,
                yes_price=yes_price,
                events=events,
                event_markets=event_markets,
                prices=prices,
                category=category,
            )

            # Edge = fair probability minus market price
            edge = fair_prob - yes_price

            # Only trade if edge exceeds threshold
            if abs(edge) < min_edge:
                continue

            # Determine direction
            if edge > 0:
                # Fair prob > market price => YES is underpriced, buy YES
                outcome = "YES"
                buy_price = yes_price
                token_id = (
                    market.clob_token_ids[0] if market.clob_token_ids and len(market.clob_token_ids) > 0 else None
                )
            else:
                # Fair prob < market price => YES is overpriced, buy NO
                outcome = "NO"
                buy_price = no_price
                token_id = (
                    market.clob_token_ids[1] if market.clob_token_ids and len(market.clob_token_ids) > 1 else None
                )

            # Profit metrics
            # expected_payout is 1.0 if our prediction is correct
            total_cost = buy_price
            expected_payout = 1.0

            # Two distinct quantities, distinguished:
            #
            # edge_pct = absolute price disagreement, scaled to cents/share.
            #   This is the conviction signal the trader scores on.
            #   Naturally bounded by [0, 100] — a 9¢ disagreement on any
            #   contract scores as 9, regardless of the contract's price.
            #
            # roi_pct  = capital efficiency: net profit per dollar staked,
            #   if our directional view is correct. Used by the QF gates
            #   and position sizing. On a $0.10 contract a 9¢ disagreement
            #   is a 90% capital-efficiency move; we DON'T want to feed
            #   that 90 number into the trader's score formula as if it
            #   were a 90-point conviction signal.
            edge_pct = abs(edge) * 100.0
            platform = str(getattr(market, "platform", "polymarket") or "polymarket")
            # ROI net of taker fees, computed from the absolute edge as a
            # fraction of the entry price. (price_disagreement / buy_price)
            # × 100 = realized capital efficiency on a directional bet.
            fee = polymarket_taker_fee(buy_price) if platform == "polymarket" else (
                kalshi_taker_fee(buy_price) if platform == "kalshi" else 0.0
            )
            net_edge = max(0.0, abs(edge) - fee)
            roi = (net_edge / max(buy_price, 1e-6)) * 100.0
            net_profit = total_cost * (roi / 100.0)

            # Skip if negative expected profit after fees
            if net_profit <= 0:
                continue

            # Liquidity & position sizing (moved up for hard filter checks)
            min_liquidity = market.liquidity
            max_position = min_liquidity * 0.05  # Conservative: 5% of liquidity

            # --- Hard filters (matching create_opportunity gate) ---
            if roi > settings.MAX_PLAUSIBLE_ROI:
                continue
            if market.liquidity < settings.MIN_LIQUIDITY_HARD:
                continue
            if max_position < settings.MIN_POSITION_SIZE:
                continue
            absolute_profit = max_position * (net_profit / total_cost) if total_cost > 0 else 0
            if absolute_profit < settings.MIN_ABSOLUTE_PROFIT:
                continue
            if market.end_date:
                resolution_aware = make_aware(market.end_date)
                days_until = (resolution_aware - utcnow()).days
                if days_until > settings.MAX_RESOLUTION_MONTHS * 30:
                    continue
                annualized_roi = self._calculate_annualized_roi(roi, market.end_date) or 0.0
                if annualized_roi < settings.MIN_ANNUALIZED_ROI:
                    continue

            # Risk assessment: stat arb is uncertain by nature
            # Risk score between 0.40 and 0.60 depending on edge strength
            # Stronger edge = slightly lower risk
            edge_strength = min(abs(edge) / 0.20, 1.0)  # Normalize to [0, 1]
            risk_score = 0.60 - (edge_strength * 0.20)  # Range: 0.40 - 0.60
            risk_score = max(0.40, min(0.60, risk_score))

            # Confidence derived from signal-agreement + edge-magnitude via
            # the SDK helper (broadly applicable to any ensemble strategy).
            # Replaces the previous hardcoded 0.5 which gave every stat_arb
            # signal the same conviction regardless of how well the seven
            # sub-signals actually agreed.
            edge_sign = 1.0 if edge > 0 else (-1.0 if edge < 0 else 0.0)
            confidence, agreeing, total_voting = StrategySDK.confidence_from_signal_agreement(
                signal_breakdown,
                edge_sign=edge_sign,
                edge_magnitude=abs(edge),
                magnitude_full_scale=float(config.get("signal_adjustment_scale", 0.15) or 0.15),
                floor=float(config.get("confidence_floor", 0.30) or 0.30),
                ceiling=float(config.get("confidence_ceiling", 0.85) or 0.85),
                agreement_weight=float(config.get("confidence_agreement_weight", 0.6) or 0.6),
            )

            risk_factors = [
                "Statistical edge, not guaranteed profit",
                f"Composite edge: {abs(edge):.1%} ({outcome} side)",
                f"Fair probability estimate: {fair_prob:.1%} vs market {yes_price:.1%}",
            ]

            # Add signal-specific risk factors
            if signal_breakdown.get("anchoring", 0) > 0:
                risk_factors.append("Anchoring bias detected at round number")
            if abs(signal_breakdown.get("category_base_rate", 0)) > 0.10:
                risk_factors.append(f"Category base rate deviation: {signal_breakdown['category_base_rate']:+.2f}")
            if abs(signal_breakdown.get("consensus", 0)) > 0.3:
                risk_factors.append("Price is an outlier within its event")
            if abs(signal_breakdown.get("momentum", 0)) > 0.3:
                direction_word = "upward" if signal_breakdown["momentum"] > 0 else "downward"
                risk_factors.append(f"Strong {direction_word} momentum detected")
            if abs(signal_breakdown.get("volume_price", 0)) > 0.3:
                risk_factors.append("Volume-price divergence detected")
            if abs(signal_breakdown.get("favorite_longshot", 0)) > 0.25:
                risk_factors.append("Favorite-longshot skew signal contributed")
            if abs(signal_breakdown.get("liquidity_imbalance", 0)) > 0.25:
                risk_factors.append("Order-book liquidity imbalance contributed")

            # Build position
            positions = [
                {
                    "action": "BUY",
                    "outcome": outcome,
                    "price": buy_price,
                    "token_id": token_id,
                    "edge": round(edge, 4),
                    "fair_probability": round(fair_prob, 4),
                    "signals": {k: round(v, 4) for k, v in signal_breakdown.items()},
                }
            ]

            opp = self.create_opportunity(
                title=f"Stat Arb: {market.question[:60]}",
                description=(
                    f"Buy {outcome} @ ${buy_price:.3f} | "
                    f"Fair prob {fair_prob:.1%} vs market {yes_price:.1%} | "
                    f"Edge {abs(edge):.1%} | "
                    f"Conf {confidence:.0%} ({agreeing}/{total_voting} signals agree)"
                ),
                total_cost=total_cost,
                expected_payout=expected_payout,
                markets=[market],
                positions=positions,
                event=event,
                is_guaranteed=False,
                custom_roi_percent=roi,
                custom_risk_score=risk_score,
                confidence=confidence,
            )
            if opp is not None:
                opp.risk_factors = risk_factors
                # Decouple conviction from capital efficiency: edge_percent is
                # the absolute price disagreement (cents/share); roi_percent
                # is the realized capital efficiency on a directional bet.
                # Route through strategy_context so the value reaches the
                # trader's signal pipeline regardless of whether the running
                # Opportunity model has the edge_percent field declared.
                conviction_edge = round(edge_pct, 3)
                try:
                    opp.edge_percent = conviction_edge
                except (AttributeError, ValueError):
                    pass
                opp.strategy_context["edge_percent"] = conviction_edge
                opp.strategy_context["fair_probability"] = round(fair_prob, 4)
                opp.strategy_context["signal_agreement"] = (
                    f"{agreeing}/{total_voting}" if total_voting else "0/0"
                )

            opportunities.append(opp)

        # Sort by absolute edge (strongest signals first)
        opportunities.sort(
            key=lambda opportunity: abs(
                float(opportunity.positions_to_take[0].get("edge", 0.0))
                if opportunity.positions_to_take and isinstance(opportunity.positions_to_take[0], dict)
                else float(opportunity.roi_percent or 0.0)
            ),
            reverse=True,
        )

        return opportunities

    def compute_size(
        self, base_size: float, max_size: float, edge: float, confidence: float, risk_score: float, market_count: int
    ) -> float:
        """Quarter-Kelly sizing via the SDK helper.

        ``edge`` here is the conviction signal (price-cents × 100, set on
        the opportunity as ``edge_percent`` and routed through the trader's
        signal pipeline). The SDK helper converts back to price space and
        computes ``f* = edge_price / (1 - edge_price)`` capped at
        quarter-Kelly, then damps by confidence and risk.

        The previous in-line implementation used
        ``p_estimated = 0.5 + edge/200`` with ``p_market = 0.5``, which
        structurally assumed the market price was always 0.50 regardless
        of the actual contract being traded.
        """
        edge_price = abs(edge) / 100.0
        return StrategySDK.fractional_kelly_size(
            base_size=base_size,
            max_size=max_size,
            edge_price=edge_price,
            confidence=confidence,
            risk_score=risk_score,
        )

    def custom_checks(self, signal, context, params, payload):
        source = str(getattr(signal, "source", "") or "").strip().lower()
        excluded_keywords = self._normalize_excluded_keywords(
            params.get("exclude_market_keywords", self.config.get("exclude_market_keywords", ""))
        )
        blocked_keyword = self._first_blocked_keyword(self._signal_market_text(signal, payload), excluded_keywords)
        return [
            DecisionCheck("source", "Signal source", source == "scanner", detail=f"got={source}"),
            DecisionCheck(
                "keyword_exclusion",
                "Market keyword exclusion",
                blocked_keyword is None,
                detail=f"blocked by '{blocked_keyword}'" if blocked_keyword else "no excluded keywords matched",
            ),
        ]

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        """Stat arb: standard TP/SL exit with temporal defaults applied."""
        if market_state.get("is_resolved"):
            return self.default_exit_check(position, market_state)
        config = getattr(position, "config", None) or {}
        config = dict(config)
        for key, fallback in (
            ("take_profit_pct", 12.0),
            ("stop_loss_pct", 25.0),
            ("trailing_stop_pct", 15.0),
        ):
            try:
                default_value = float((getattr(self, "config", None) or {}).get(key, fallback))
            except (TypeError, ValueError):
                default_value = fallback
            config.setdefault(key, default_value)
        position.config = config
        return self.default_exit_check(position, market_state)

    # ------------------------------------------------------------------
    # Platform gate hooks
    # ------------------------------------------------------------------

    def on_blocked(self, signal, reason: str, context: dict) -> None:
        logger.info("%s: signal blocked — %s (market=%s)", self.name, reason, getattr(signal, "market_id", "?"))

    def on_size_capped(self, original_size: float, capped_size: float, reason: str) -> None:
        logger.info("%s: size capped $%.0f → $%.0f — %s", self.name, original_size, capped_size, reason)
