"""
Strategy: BTC/ETH High-Frequency Arbitrage

Specialized strategy for Bitcoin and Ethereum binary markets on Polymarket,
targeting the highly liquid 15-minute and 1-hour "up or down" markets.

These markets are the most liquid arbitrage venue on Polymarket. The "gabagool"
bot reportedly earns ~$58 every 15 minutes by exploiting inefficiencies in
BTC 15-min markets alone.

This strategy uses dynamic sub-strategy selection (Option C):
  A. Pure Arbitrage   -- Buy YES + NO when combined < $1.00
  B. Dump-Hedge       -- Buy the dumped side after a >5% drop, then hedge
  C. Pre-Placed Limits -- Pre-place limit orders at $0.45-$0.47 on new markets

The selector scores each sub-strategy against current market conditions
(price levels, volatility, time to expiry, liquidity, order book state) and
returns opportunities from the best-fitting sub-strategy.
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from models import Market, Event, ArbitrageOpportunity, StrategyType
from config import settings as _cfg
from .base import BaseStrategy
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Question / slug patterns used to identify BTC/ETH high-frequency markets
_ASSET_PATTERNS: dict[str, list[str]] = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
    "SOL": ["solana", "sol"],
    "XRP": ["ripple", "xrp"],
}

_TIMEFRAME_PATTERNS: dict[str, list[str]] = {
    "15min": [
        "15 min",
        "15-min",
        "15min",
        "fifteen min",
        "15 minute",
        "15-minute",
        "15 minutes",
        "15-minutes",
        "quarter hour",
        "quarter-hour",
    ],
    "1hr": [
        "1 hour",
        "1-hour",
        "1hr",
        "one hour",
        "60 min",
        "60-min",
        "60 minute",
        "60-minute",
        "60 minutes",
        "60-minutes",
        "hourly",
        "next hour",
    ],
}

_DIRECTION_KEYWORDS: list[str] = [
    "up or down",
    "higher or lower",
    "go up",
    "go down",
    "above or below",
    "increase or decrease",
    "up",
    "down",
    "higher",
    "lower",
    "price",
    "beat",
    "price to beat",
]

# Slug regex: matches slugs where asset and timeframe may be separated by
# other words.  Allows "bitcoin-15-minute-up-or-down", "btc-price-15min",
# "ethereum-1-hour-up-down", etc.
_SLUG_REGEX = re.compile(
    r"(btc|eth|sol|xrp|bitcoin|ethereum|solana|ripple)"
    r".*?"  # allow intervening words (non-greedy)
    r"(15[\s_-]?min(?:ute)?s?|1[\s_-]?h(?:ou)?r|60[\s_-]?min(?:ute)?s?"
    r"|quarter[\s_-]?hour|hourly)",
    re.IGNORECASE,
)

# Gamma API crypto tags to search for high-freq markets
_GAMMA_CRYPTO_TAGS = ["crypto", "btc", "bitcoin", "eth", "ethereum"]

# Strategy selector thresholds — read from config (persisted in DB via Settings UI)


def _pure_arb_max_combined():
    return _cfg.BTC_ETH_HF_PURE_ARB_MAX_COMBINED


def _dump_hedge_drop_pct():
    return _cfg.BTC_ETH_HF_DUMP_THRESHOLD


def _thin_liquidity_usd():
    return _cfg.BTC_ETH_HF_THIN_LIQUIDITY_USD


# Dump-hedge combined cost target raised from 0.97 to 0.98: after 2% fee,
# you need combined < 0.98 to profit. 0.97 left only 1% gross before fees.
_DUMP_HEDGE_MAX_COMBINED = 0.98  # Combined cost target after dump-hedge
_LIMIT_ORDER_TARGET_LOW = 0.45  # Lower limit order price
_LIMIT_ORDER_TARGET_HIGH = 0.47  # Upper limit order price
_NEW_MARKET_VOLUME_THRESHOLD = 5000.0  # Markets with volume below this are "new"

# Price history defaults
_DEFAULT_HISTORY_WINDOW_SEC = 300  # 5 minutes for 15-min markets
_1HR_HISTORY_WINDOW_SEC = 600  # 10 minutes for 1-hr markets
_MAX_HISTORY_ENTRIES = 200  # Maximum price snapshots per market


# ---------------------------------------------------------------------------
# Gamma API crypto market fetcher
# ---------------------------------------------------------------------------


class _CryptoMarketFetcher:
    """Sync HTTP fetcher that queries Polymarket's Gamma API for crypto markets.

    BTC/ETH 15-minute markets are high-frequency and may not appear in the
    top 500 general markets.  This fetcher queries specific tags/slugs to
    ensure they are always included in the candidate pool.

    Uses a sync client (like the Kalshi cache) because detect() is sync.
    Results are cached for ``ttl_seconds`` to avoid hammering the API.
    """

    def __init__(self, gamma_url: str = "", ttl_seconds: int = 30):
        self._gamma_url = gamma_url or _cfg.GAMMA_API_URL
        self._ttl = ttl_seconds
        self._markets: list[Market] = []
        self._last_fetch: float = 0.0

    @property
    def is_stale(self) -> bool:
        return (time.monotonic() - self._last_fetch) > self._ttl

    def get_markets(self) -> list[Market]:
        """Return cached crypto markets, refreshing if stale."""
        if self.is_stale:
            fetched = self._fetch()
            if fetched:
                self._markets = fetched
                self._last_fetch = time.monotonic()
        return self._markets

    def _fetch(self) -> list[Market]:
        """Fetch crypto markets from Gamma API by tag and slug patterns."""
        import httpx

        all_markets: list[Market] = []
        seen_ids: set[str] = set()

        try:
            with httpx.Client(timeout=15.0) as client:
                # Strategy 1: Search by crypto tags
                for tag in _GAMMA_CRYPTO_TAGS:
                    try:
                        resp = client.get(
                            f"{self._gamma_url}/events",
                            params={"tag": tag, "closed": "false", "limit": 50},
                        )
                        if resp.status_code == 200:
                            for event_data in resp.json():
                                for mkt_data in event_data.get("markets", []):
                                    mid = mkt_data.get("condition_id") or mkt_data.get(
                                        "id", ""
                                    )
                                    if mid and mid not in seen_ids:
                                        try:
                                            m = Market.from_gamma_response(mkt_data)
                                            all_markets.append(m)
                                            seen_ids.add(mid)
                                        except Exception:
                                            pass
                        time.sleep(0.1)  # Rate limit
                    except Exception:
                        pass

                # Strategy 2: Search for specific slug patterns
                slug_searches = [
                    "bitcoin-15-minute",
                    "btc-15-min",
                    "btc-15min",
                    "ethereum-15-minute",
                    "eth-15-min",
                    "bitcoin-1-hour",
                    "btc-1-hour",
                    "ethereum-1-hour",
                    "eth-1-hour",
                    "bitcoin-up-or-down",
                    "ethereum-up-or-down",
                    "btc-up-or-down",
                    "eth-up-or-down",
                    "bitcoin-price",
                    "ethereum-price",
                ]
                for slug_query in slug_searches:
                    try:
                        resp = client.get(
                            f"{self._gamma_url}/markets",
                            params={
                                "slug": slug_query,
                                "closed": "false",
                                "active": "true",
                                "limit": 20,
                            },
                        )
                        if resp.status_code == 200:
                            for mkt_data in resp.json():
                                mid = mkt_data.get("condition_id") or mkt_data.get(
                                    "id", ""
                                )
                                if mid and mid not in seen_ids:
                                    try:
                                        m = Market.from_gamma_response(mkt_data)
                                        all_markets.append(m)
                                        seen_ids.add(mid)
                                    except Exception:
                                        pass
                        time.sleep(0.1)
                    except Exception:
                        pass

        except Exception as exc:
            logger.warning("Crypto market fetch failed", error=str(exc))

        if all_markets:
            logger.info(
                "BtcEthHighFreq: fetched %d crypto markets via Gamma API",
                len(all_markets),
            )
        return all_markets


# Module-level crypto market fetcher (lazy-initialized)
_crypto_fetcher: Optional[_CryptoMarketFetcher] = None


def _get_crypto_fetcher() -> _CryptoMarketFetcher:
    """Get or create the singleton crypto market fetcher."""
    global _crypto_fetcher
    if _crypto_fetcher is None:
        _crypto_fetcher = _CryptoMarketFetcher()
    return _crypto_fetcher


# ---------------------------------------------------------------------------
# Sub-strategy enum
# ---------------------------------------------------------------------------


class SubStrategy(str, Enum):
    PURE_ARB = "pure_arb"
    DUMP_HEDGE = "dump_hedge"
    PRE_PLACED_LIMITS = "pre_placed_limits"


# ---------------------------------------------------------------------------
# Price history tracker
# ---------------------------------------------------------------------------


@dataclass
class PriceSnapshot:
    """A single price observation at a point in time."""

    timestamp: float  # time.monotonic()
    yes_price: float
    no_price: float


@dataclass
class MarketPriceHistory:
    """Rolling window of price snapshots for a single market."""

    window_seconds: float = _DEFAULT_HISTORY_WINDOW_SEC
    snapshots: deque[PriceSnapshot] = field(default_factory=deque)

    def record(self, yes_price: float, no_price: float) -> None:
        """Append a snapshot and evict stale entries."""
        now = time.monotonic()
        self.snapshots.append(
            PriceSnapshot(
                timestamp=now,
                yes_price=yes_price,
                no_price=no_price,
            )
        )
        self._evict(now)

    def _evict(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self.snapshots and self.snapshots[0].timestamp < cutoff:
            self.snapshots.popleft()

    @property
    def has_data(self) -> bool:
        return len(self.snapshots) >= 2

    def max_drop_yes(self) -> float:
        """Return the largest drop (positive value) in YES price over the window."""
        if not self.has_data:
            return 0.0
        peak = max(s.yes_price for s in self.snapshots)
        current = self.snapshots[-1].yes_price
        return max(peak - current, 0.0)

    def max_drop_no(self) -> float:
        """Return the largest drop (positive value) in NO price over the window."""
        if not self.has_data:
            return 0.0
        peak = max(s.no_price for s in self.snapshots)
        current = self.snapshots[-1].no_price
        return max(peak - current, 0.0)

    def recent_volatility(self) -> float:
        """Simple volatility proxy: max price range over the window (YES side)."""
        if not self.has_data:
            return 0.0
        prices = [s.yes_price for s in self.snapshots]
        return max(prices) - min(prices)


# ---------------------------------------------------------------------------
# Candidate detection helper
# ---------------------------------------------------------------------------


@dataclass
class HighFreqCandidate:
    """A market identified as a BTC/ETH high-frequency binary market."""

    market: Market
    asset: str  # "BTC" or "ETH"
    timeframe: str  # "15min" or "1hr"
    yes_price: float
    no_price: float


# ---------------------------------------------------------------------------
# Sub-strategy scoring
# ---------------------------------------------------------------------------


@dataclass
class SubStrategyScore:
    """Score and metadata for a candidate sub-strategy."""

    strategy: SubStrategy
    score: float  # Higher is better (0-100 scale)
    reason: str
    params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main strategy class
# ---------------------------------------------------------------------------


class BtcEthHighFreqStrategy(BaseStrategy):
    """
    High-frequency arbitrage strategy for BTC and ETH binary markets.

    Dynamically selects among three sub-strategies based on current market
    conditions:
      A. Pure Arbitrage   -- guaranteed profit when YES + NO < $1.00
      B. Dump-Hedge       -- buy a dumped side, hedge with opposite
      C. Pre-Placed Limits -- limit orders on new/thin markets

    Designed for Polymarket's 15-min and 1-hr BTC/ETH up-or-down markets.
    """

    strategy_type = StrategyType.BTC_ETH_HIGHFREQ
    name = "BTC/ETH High-Frequency"
    description = (
        "Dynamic high-frequency arbitrage on BTC/ETH 15-min and 1-hr binary markets"
    )

    def __init__(self) -> None:
        super().__init__()
        # Per-market price history keyed by market ID
        self._price_histories: dict[str, MarketPriceHistory] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[ArbitrageOpportunity]:
        """Detect arbitrage opportunities across BTC/ETH high-freq markets.

        1. Filter markets to find BTC/ETH high-freq candidates.
        2. Update price history for each candidate.
        3. Run the dynamic strategy selector on each candidate.
        4. Return all detected opportunities.
        """
        opportunities: list[ArbitrageOpportunity] = []

        candidates = self._find_candidates(markets, prices)
        if not candidates:
            logger.debug("BtcEthHighFreq: no BTC/ETH high-freq candidates found")
            return opportunities

        logger.info(
            "BtcEthHighFreq: found %d candidate market(s) — evaluating sub-strategies",
            len(candidates),
        )

        for candidate in candidates:
            # Update price history
            self._update_price_history(candidate)

            # Dynamic strategy selection
            selected, all_scores = self._select_sub_strategy(candidate)
            if selected is None:
                logger.debug(
                    "BtcEthHighFreq: no viable sub-strategy for market %s (%s %s)",
                    candidate.market.id,
                    candidate.asset,
                    candidate.timeframe,
                )
                continue

            logger.info(
                "BtcEthHighFreq: market %s (%s %s) — selected %s (score=%.1f). "
                "All scores: %s",
                candidate.market.id,
                candidate.asset,
                candidate.timeframe,
                selected.strategy.value,
                selected.score,
                ", ".join(f"{s.strategy.value}={s.score:.1f}" for s in all_scores),
            )

            # Generate opportunity from the selected sub-strategy
            opp = self._generate_opportunity(candidate, selected)
            if opp is not None:
                opportunities.append(opp)
                logger.info(
                    "BtcEthHighFreq: opportunity detected — %s | ROI %.2f%% | "
                    "sub-strategy=%s | market=%s",
                    opp.title,
                    opp.roi_percent,
                    selected.strategy.value,
                    candidate.market.id,
                )

        logger.info(
            "BtcEthHighFreq: scan complete — %d opportunity(ies) found",
            len(opportunities),
        )
        return opportunities

    # ------------------------------------------------------------------
    # Market identification
    # ------------------------------------------------------------------

    def _find_candidates(
        self,
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[HighFreqCandidate]:
        """Filter the full market list to BTC/ETH high-freq binary markets.

        Also queries the Gamma API directly for crypto markets by tag/slug
        to catch BTC/ETH 15-min markets that may not be in the top 500.
        """
        candidates: list[HighFreqCandidate] = []
        seen_ids: set[str] = set()

        # Combine scanner markets with directly-fetched crypto markets
        all_markets = list(markets)
        try:
            fetcher = _get_crypto_fetcher()
            extra = fetcher.get_markets()
            for m in extra:
                if m.id not in seen_ids:
                    all_markets.append(m)
            # Also mark scanner-provided markets so we don't double-count
            for m in markets:
                seen_ids.add(m.id)
        except Exception:
            pass  # Non-fatal: fall back to scanner markets only

        for market in all_markets:
            if market.closed or not market.active:
                continue
            if len(market.outcome_prices) != 2:
                continue
            if market.id in seen_ids and market not in markets:
                continue  # Already processed from scanner list
            seen_ids.add(market.id)

            asset = self._detect_asset(market)
            if asset is None:
                continue

            timeframe = self._detect_timeframe(market)
            if timeframe is None:
                continue

            # Resolve live prices
            yes_price, no_price = self._resolve_prices(market, prices)

            candidates.append(
                HighFreqCandidate(
                    market=market,
                    asset=asset,
                    timeframe=timeframe,
                    yes_price=yes_price,
                    no_price=no_price,
                )
            )

        return candidates

    @staticmethod
    def _detect_asset(market: Market) -> Optional[str]:
        """Return 'BTC' or 'ETH' if the market targets one of those assets."""
        text = f"{market.question} {market.slug}".lower()
        for asset, keywords in _ASSET_PATTERNS.items():
            if any(kw in text for kw in keywords):
                return asset
        return None

    @staticmethod
    def _detect_timeframe(market: Market) -> Optional[str]:
        """Return '15min' or '1hr' if a matching timeframe is detected."""
        text = f"{market.question} {market.slug}".lower()

        # Try slug regex first — now allows words between asset and timeframe
        slug_text = f"{market.slug} {market.question}".lower()
        slug_match = _SLUG_REGEX.search(slug_text)
        if slug_match:
            raw_tf = slug_match.group(2).lower().replace("-", "").replace("_", "")
            if "15" in raw_tf or "quarter" in raw_tf:
                return "15min"
            if "1h" in raw_tf or "60" in raw_tf or "hourly" in raw_tf:
                return "1hr"

        # Fallback: question-text keyword matching (broadened patterns)
        for tf_key, patterns in _TIMEFRAME_PATTERNS.items():
            if any(p in text for p in patterns):
                return tf_key

        return None

    @staticmethod
    def _is_direction_market(market: Market) -> bool:
        """Check if the market is a directional up/down style question."""
        text = market.question.lower()
        return any(kw in text for kw in _DIRECTION_KEYWORDS)

    @staticmethod
    def _resolve_prices(
        market: Market,
        prices: dict[str, dict],
    ) -> tuple[float, float]:
        """Return (yes_price, no_price) using live CLOB prices when available."""
        yes_price = market.yes_price
        no_price = market.no_price

        if market.clob_token_ids:
            if len(market.clob_token_ids) > 0:
                token = market.clob_token_ids[0]
                if token in prices:
                    yes_price = prices[token].get("mid", yes_price)
            if len(market.clob_token_ids) > 1:
                token = market.clob_token_ids[1]
                if token in prices:
                    no_price = prices[token].get("mid", no_price)

        return yes_price, no_price

    # ------------------------------------------------------------------
    # Price history
    # ------------------------------------------------------------------

    def _update_price_history(self, candidate: HighFreqCandidate) -> None:
        """Record the latest prices into the rolling window for this market."""
        mid = candidate.market.id
        if mid not in self._price_histories:
            window = (
                _1HR_HISTORY_WINDOW_SEC
                if candidate.timeframe == "1hr"
                else _DEFAULT_HISTORY_WINDOW_SEC
            )
            self._price_histories[mid] = MarketPriceHistory(window_seconds=window)

        self._price_histories[mid].record(
            candidate.yes_price,
            candidate.no_price,
        )

    def _get_history(self, market_id: str) -> Optional[MarketPriceHistory]:
        return self._price_histories.get(market_id)

    # ------------------------------------------------------------------
    # Dynamic strategy selector
    # ------------------------------------------------------------------

    def _select_sub_strategy(
        self,
        candidate: HighFreqCandidate,
    ) -> tuple[Optional[SubStrategyScore], list[SubStrategyScore]]:
        """Score all three sub-strategies and return the best one.

        Returns (best_score_or_None, all_scores).
        A sub-strategy with score <= 0 is considered non-viable.
        """
        scores: list[SubStrategyScore] = [
            self._score_pure_arb(candidate),
            self._score_dump_hedge(candidate),
            self._score_pre_placed_limits(candidate),
        ]

        # Sort descending by score
        scores.sort(key=lambda s: s.score, reverse=True)

        best = scores[0] if scores[0].score > 0 else None
        return best, scores

    # -- Sub-strategy A: Pure Arbitrage scoring --

    def _score_pure_arb(self, c: HighFreqCandidate) -> SubStrategyScore:
        """Score pure arbitrage opportunity (YES + NO < $1.00).

        Higher score when combined cost is lower (larger guaranteed spread).
        Select when combined < 0.98.
        """
        combined = c.yes_price + c.no_price
        fee_cost = self.fee  # winner pays this fraction

        # Net profit per $1 payout after fees
        net_profit = 1.0 - combined - fee_cost
        if net_profit <= 0:
            return SubStrategyScore(
                strategy=SubStrategy.PURE_ARB,
                score=0.0,
                reason=f"No spread after fees (combined={combined:.4f}, fee={fee_cost:.4f})",
            )

        if combined >= _pure_arb_max_combined():
            return SubStrategyScore(
                strategy=SubStrategy.PURE_ARB,
                score=0.0,
                reason=f"Combined cost {combined:.4f} >= {_pure_arb_max_combined()} threshold",
            )

        # Base score proportional to net profit (scale: 1 cent = 10 points)
        base_score = net_profit * 1000.0  # e.g. 0.02 net profit -> 20 pts

        # Bonus for high liquidity (confidence we can fill)
        liquidity = c.market.liquidity
        if liquidity >= 5000:
            base_score += 15.0
        elif liquidity >= 2000:
            base_score += 8.0
        elif liquidity >= 1000:
            base_score += 3.0

        # Bonus for balanced prices (both sides near 0.49-0.50 = most liquid)
        balance = 1.0 - abs(c.yes_price - c.no_price)
        base_score += balance * 5.0

        return SubStrategyScore(
            strategy=SubStrategy.PURE_ARB,
            score=base_score,
            reason=(
                f"Pure arb: combined={combined:.4f}, net_profit={net_profit:.4f}, "
                f"liquidity=${liquidity:.0f}"
            ),
            params={
                "combined_cost": combined,
                "net_profit": net_profit,
                "yes_price": c.yes_price,
                "no_price": c.no_price,
            },
        )

    # -- Sub-strategy B: Dump-Hedge scoring --

    def _score_dump_hedge(self, c: HighFreqCandidate) -> SubStrategyScore:
        """Score dump-hedge opportunity.

        Triggered when one side drops > 5% in the recent window. Buy the dumped
        side (now cheap), wait for partial recovery, then hedge with the other
        side. If the combined cost after hedge < target, there is profit.
        """
        history = self._get_history(c.market.id)
        if history is None or not history.has_data:
            return SubStrategyScore(
                strategy=SubStrategy.DUMP_HEDGE,
                score=0.0,
                reason="No price history available yet",
            )

        yes_drop = history.max_drop_yes()
        no_drop = history.max_drop_no()
        max_drop = max(yes_drop, no_drop)
        dumped_side = "YES" if yes_drop >= no_drop else "NO"

        if max_drop < _dump_hedge_drop_pct():
            return SubStrategyScore(
                strategy=SubStrategy.DUMP_HEDGE,
                score=0.0,
                reason=(
                    f"Insufficient dump: max drop {max_drop:.4f} "
                    f"< {_dump_hedge_drop_pct()} threshold"
                ),
            )

        # Estimate combined cost if we buy the dumped side now and hedge
        combined = c.yes_price + c.no_price
        if combined >= _DUMP_HEDGE_MAX_COMBINED:
            return SubStrategyScore(
                strategy=SubStrategy.DUMP_HEDGE,
                score=0.0,
                reason=(
                    f"Combined {combined:.4f} too high for dump-hedge "
                    f"(target < {_DUMP_HEDGE_MAX_COMBINED})"
                ),
            )

        net_profit = 1.0 - combined - self.fee

        # Score: larger drop and larger profit = better
        base_score = max_drop * 200.0  # 5% drop -> 10 pts, 10% drop -> 20 pts
        base_score += max(net_profit, 0) * 500.0  # reward profitable combined

        # Volatility bonus: higher volatility means more dump-hedge opportunities
        volatility = history.recent_volatility()
        base_score += volatility * 50.0

        # Liquidity matters: need to be able to fill quickly
        if c.market.liquidity >= 3000:
            base_score += 5.0
        elif c.market.liquidity < 1000:
            base_score *= 0.5  # penalize low liquidity heavily

        return SubStrategyScore(
            strategy=SubStrategy.DUMP_HEDGE,
            score=base_score,
            reason=(
                f"Dump-hedge: {dumped_side} dropped {max_drop:.4f}, "
                f"combined={combined:.4f}, volatility={volatility:.4f}"
            ),
            params={
                "dumped_side": dumped_side,
                "drop_amount": max_drop,
                "combined_cost": combined,
                "net_profit": net_profit,
                "volatility": volatility,
                "yes_price": c.yes_price,
                "no_price": c.no_price,
            },
        )

    # -- Sub-strategy C: Pre-Placed Limits scoring --

    def _calculate_dynamic_limit_prices(
        self, c: HighFreqCandidate
    ) -> tuple[float, float]:
        """Calculate optimal limit order prices based on current market state.

        Instead of fixed $0.45-$0.47, adjusts based on:
        - Current market prices (bid closer to current for fill probability)
        - Liquidity level (thinner book = can be more aggressive)
        - Required minimum profit margin
        """
        # Base targets
        min_target = 0.42  # Absolute minimum (most aggressive)

        # Start from the aggressive end
        base_price = 0.45

        # Adjust based on liquidity: thinner book = can be more aggressive
        if c.market.liquidity < 100:
            base_price = min_target  # Very thin, be aggressive
        elif c.market.liquidity < 250:
            base_price = 0.44
        elif c.market.liquidity < 400:
            base_price = 0.45
        else:
            base_price = 0.46  # Moderate book, bid closer to fill

        # Ensure combined still profits: need combined < 1.0 - fee
        # If we bid base_price on both sides, combined = 2 * base_price
        # Need: 2 * base_price + fee < 1.0
        max_combined = 1.0 - self.fee - 0.01  # Leave 1% minimum profit
        max_per_side = max_combined / 2.0

        target_price = min(base_price, max_per_side)
        target_price = max(target_price, min_target)  # Never go below absolute min

        return target_price, target_price

    def _score_pre_placed_limits(self, c: HighFreqCandidate) -> SubStrategyScore:
        """Score pre-placed limit order opportunity.

        For markets about to open or with very thin order books, pre-place
        limit orders at $0.45-$0.47 on both sides. If both fill, combined cost
        is $0.90-$0.94 for guaranteed $1.00 payout.

        Select when: new market detected with thin order book.
        """
        liquidity = c.market.liquidity

        # This sub-strategy targets thin/new markets
        if liquidity > _thin_liquidity_usd():
            return SubStrategyScore(
                strategy=SubStrategy.PRE_PLACED_LIMITS,
                score=0.0,
                reason=(
                    f"Market too liquid (${liquidity:.0f}) for pre-placed limits "
                    f"(threshold=${_thin_liquidity_usd():.0f})"
                ),
            )

        # Check if prices are near the sweet spot (0.45-0.55 per side = new market)
        both_near_half = _LIMIT_ORDER_TARGET_LOW <= c.yes_price <= (
            1.0 - _LIMIT_ORDER_TARGET_LOW
        ) and _LIMIT_ORDER_TARGET_LOW <= c.no_price <= (1.0 - _LIMIT_ORDER_TARGET_LOW)

        # Calculate dynamic limit prices based on market conditions
        target_yes, target_no = self._calculate_dynamic_limit_prices(c)
        target_combined = target_yes + target_no
        target_profit = 1.0 - target_combined - self.fee

        if target_profit <= 0:
            return SubStrategyScore(
                strategy=SubStrategy.PRE_PLACED_LIMITS,
                score=0.0,
                reason=f"No profit at target prices (combined=${target_combined:.4f})",
            )

        # Base score: thin book is a strong signal
        base_score = 10.0

        # Lower liquidity = more opportunity for limit fills
        if liquidity < 100:
            base_score += 20.0
        elif liquidity < 250:
            base_score += 10.0
        else:
            base_score += 3.0

        # Near-half prices suggest a freshly opened market (ideal)
        if both_near_half:
            base_score += 15.0

        # Market age proxy: very low volume = likely just opened
        if c.market.volume < 100:
            base_score += 20.0  # Almost certainly a new market
        elif c.market.volume < 500:
            base_score += 12.0  # Very new
        elif c.market.volume < _NEW_MARKET_VOLUME_THRESHOLD:
            base_score += 5.0   # Relatively new

        # Bonus for the expected profit
        base_score += target_profit * 300.0

        # Penalty: if the market already has significant volume, limits are
        # less likely to fill at our targets
        if c.market.volume > 10000:
            base_score *= 0.4

        return SubStrategyScore(
            strategy=SubStrategy.PRE_PLACED_LIMITS,
            score=base_score,
            reason=(
                f"Pre-placed limits: liquidity=${liquidity:.0f}, "
                f"target_combined=${target_combined:.4f}, "
                f"target_profit=${target_profit:.4f}, "
                f"prices_near_half={both_near_half}"
            ),
            params={
                "target_yes_price": target_yes,
                "target_no_price": target_no,
                "target_combined": target_combined,
                "target_profit": target_profit,
                "current_yes_price": c.yes_price,
                "current_no_price": c.no_price,
                "liquidity": liquidity,
            },
        )

    # ------------------------------------------------------------------
    # Opportunity generation
    # ------------------------------------------------------------------

    def _generate_opportunity(
        self,
        candidate: HighFreqCandidate,
        selected: SubStrategyScore,
    ) -> Optional[ArbitrageOpportunity]:
        """Turn a scored sub-strategy into an ArbitrageOpportunity via the base
        class ``create_opportunity`` (which applies all hard filters)."""

        market = candidate.market
        sub = selected.strategy
        params = selected.params

        if sub == SubStrategy.PURE_ARB:
            return self._generate_pure_arb(candidate, params)
        elif sub == SubStrategy.DUMP_HEDGE:
            return self._generate_dump_hedge(candidate, params)
        elif sub == SubStrategy.PRE_PLACED_LIMITS:
            return self._generate_pre_placed_limits(candidate, params)

        logger.warning(
            "BtcEthHighFreq: unknown sub-strategy %s for market %s",
            sub,
            market.id,
        )
        return None

    def _generate_pure_arb(
        self,
        c: HighFreqCandidate,
        params: dict,
    ) -> Optional[ArbitrageOpportunity]:
        """Generate opportunity for sub-strategy A: Pure Arbitrage."""
        market = c.market
        yes_price = params["yes_price"]
        no_price = params["no_price"]
        combined = params["combined_cost"]

        positions = self._build_both_sides_positions(market, yes_price, no_price)

        opp = self.create_opportunity(
            title=(
                f"BTC/ETH HF Pure Arb: {c.asset} {c.timeframe} ({market.question[:40]})"
            ),
            description=(
                f"Pure arbitrage on {c.asset} {c.timeframe} market. "
                f"Buy YES (${yes_price:.4f}) + NO (${no_price:.4f}) = "
                f"${combined:.4f} for guaranteed $1.00 payout."
            ),
            total_cost=combined,
            markets=[market],
            positions=positions,
        )

        if opp is not None:
            self._attach_highfreq_metadata(opp, c, SubStrategy.PURE_ARB, params)
        return opp

    def _generate_dump_hedge(
        self,
        c: HighFreqCandidate,
        params: dict,
    ) -> Optional[ArbitrageOpportunity]:
        """Generate opportunity for sub-strategy B: Dump-Hedge."""
        market = c.market
        dumped_side = params["dumped_side"]
        drop_amount = params["drop_amount"]
        yes_price = params["yes_price"]
        no_price = params["no_price"]
        combined = params["combined_cost"]

        # Primary position: buy the dumped side
        # Hedge position: buy the opposite side
        positions = self._build_both_sides_positions(market, yes_price, no_price)

        # Mark which side was dumped for execution ordering
        for pos in positions:
            if pos["outcome"] == dumped_side:
                pos["role"] = "primary"
                pos["note"] = f"Dumped side (dropped {drop_amount:.4f})"
            else:
                pos["role"] = "hedge"
                pos["note"] = "Hedge after partial recovery of primary"

        opp = self.create_opportunity(
            title=(
                f"BTC/ETH HF Dump-Hedge: {c.asset} {c.timeframe} "
                f"({dumped_side} dropped)"
            ),
            description=(
                f"Dump-hedge on {c.asset} {c.timeframe} market. "
                f"{dumped_side} dropped {drop_amount:.4f} — buy dumped side, "
                f"then hedge. Combined=${combined:.4f}."
            ),
            total_cost=combined,
            markets=[market],
            positions=positions,
        )

        if opp is not None:
            self._attach_highfreq_metadata(opp, c, SubStrategy.DUMP_HEDGE, params)
            opp.risk_factors.insert(
                0,
                f"Dump-hedge requires fast execution: {dumped_side} may recover "
                f"before hedge is placed",
            )
        return opp

    def _generate_pre_placed_limits(
        self,
        c: HighFreqCandidate,
        params: dict,
    ) -> Optional[ArbitrageOpportunity]:
        """Generate opportunity for sub-strategy C: Pre-Placed Limits."""
        market = c.market
        target_yes = params["target_yes_price"]
        target_no = params["target_no_price"]
        target_combined = params["target_combined"]

        positions = []
        if market.clob_token_ids and len(market.clob_token_ids) >= 2:
            positions = [
                {
                    "action": "LIMIT_BUY",
                    "outcome": "YES",
                    "price": target_yes,
                    "token_id": market.clob_token_ids[0],
                    "note": f"Limit order at ${target_yes:.2f}",
                },
                {
                    "action": "LIMIT_BUY",
                    "outcome": "NO",
                    "price": target_no,
                    "token_id": market.clob_token_ids[1],
                    "note": f"Limit order at ${target_no:.2f}",
                },
            ]

        opp = self.create_opportunity(
            title=(f"BTC/ETH HF Pre-Limits: {c.asset} {c.timeframe} (thin book)"),
            description=(
                f"Pre-placed limit orders on {c.asset} {c.timeframe} market "
                f"(liquidity=${params.get('liquidity', 0):.0f}). "
                f"Target: YES@${target_yes:.2f} + NO@${target_no:.2f} = "
                f"${target_combined:.4f} for $1.00 payout."
            ),
            total_cost=target_combined,
            markets=[market],
            positions=positions,
        )

        if opp is not None:
            self._attach_highfreq_metadata(
                opp,
                c,
                SubStrategy.PRE_PLACED_LIMITS,
                params,
            )
            opp.risk_factors.insert(
                0,
                "Pre-placed limits: profit only if BOTH sides fill at target prices",
            )
        return opp

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_both_sides_positions(
        market: Market,
        yes_price: float,
        no_price: float,
    ) -> list[dict]:
        """Build standard BUY YES + BUY NO position list."""
        positions: list[dict] = []
        if market.clob_token_ids and len(market.clob_token_ids) >= 2:
            positions = [
                {
                    "action": "BUY",
                    "outcome": "YES",
                    "price": yes_price,
                    "token_id": market.clob_token_ids[0],
                },
                {
                    "action": "BUY",
                    "outcome": "NO",
                    "price": no_price,
                    "token_id": market.clob_token_ids[1],
                },
            ]
        return positions

    @staticmethod
    def _attach_highfreq_metadata(
        opp: ArbitrageOpportunity,
        candidate: HighFreqCandidate,
        sub_strategy: SubStrategy,
        params: dict,
    ) -> None:
        """Attach BTC/ETH high-freq metadata to the opportunity for
        downstream consumers (execution engine, dashboard, logging)."""
        # Store in the existing positions_to_take metadata (which is a list
        # of dicts). We append a metadata entry at the end.
        opp.positions_to_take.append(
            {
                "_highfreq_metadata": True,
                "asset": candidate.asset,
                "timeframe": candidate.timeframe,
                "sub_strategy": sub_strategy.value,
                "sub_strategy_params": params,
            }
        )
