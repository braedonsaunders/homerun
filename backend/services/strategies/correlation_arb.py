"""
Strategy: Correlation / Cointegration Arbitrage

Finds pairs of markets that should be correlated and trades
the spread when they diverge. Mean-reversion on the spread.

Example:
- "Fed raises rates March" and "USD strengthens March"
- Usually move together (correlation ~0.8)
- If one moves up but other doesn't, bet on convergence

Uses rolling correlation calculation and z-score of spread.
"""

import math
import time
from collections import deque
from typing import Optional

from models import Market, Event, ArbitrageOpportunity, StrategyType
from config import settings
from .base import BaseStrategy
from utils.logger import get_logger

logger = get_logger(__name__)

# Minimum data points before calculating correlation
_MIN_DATA_POINTS = 10

# Z-score threshold for divergence signal
_ZSCORE_THRESHOLD = 2.0

# Maximum number of price history entries per market
_MAX_HISTORY = 200

# Minimum price variance to consider a series informative.
# If a price hasn't moved (variance near zero), correlation is meaningless.
_MIN_PRICE_VARIANCE = 1e-6

# Maximum plausible correlation threshold.
# r > 0.98 with sparse data almost always indicates two static series
# that happen not to move, not a genuine economic relationship.
_MAX_PLAUSIBLE_CORRELATION = 0.98

# Minimum keyword overlap for cross-event (category-based) pairing.
# Prevents pairing completely unrelated markets just because they share
# a category (e.g., two different hockey games under "Sports").
_MIN_KEYWORD_OVERLAP = 2

# Stop words for keyword-based pair matching
_STOP_WORDS = frozenset(
    {
        "will",
        "the",
        "a",
        "an",
        "in",
        "on",
        "by",
        "to",
        "be",
        "is",
        "are",
        "of",
        "for",
        "with",
        "this",
        "that",
        "it",
        "at",
        "from",
        "or",
        "and",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "was",
        "were",
        "been",
        "not",
        "no",
        "yes",
        "what",
        "when",
        "how",
        "who",
        "which",
        "if",
        "but",
        "can",
        "could",
        "would",
        "should",
        "may",
        "might",
        "than",
        "then",
        "so",
        "as",
        "its",
        "their",
        "there",
        "they",
        "them",
        "market",
        "price",
        "2024",
        "2025",
        "2026",
        "2027",
    }
)


def _pearson_correlation(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """
    Calculate Pearson correlation coefficient between two lists.

    Returns (correlation, var_x, var_y):
    - correlation: value in [-1, 1], 0.0 if calculation impossible
    - var_x: variance of xs (unnormalized)
    - var_y: variance of ys (unnormalized)

    Callers should check var_x and var_y against _MIN_PRICE_VARIANCE
    to reject correlations computed from static (non-moving) series.
    """
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0, 0.0, 0.0

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    # Covariance and standard deviations
    cov = 0.0
    var_x = 0.0
    var_y = 0.0

    for i in range(n):
        dx = xs[i] - mean_x
        dy = ys[i] - mean_y
        cov += dx * dy
        var_x += dx * dx
        var_y += dy * dy

    if var_x == 0 or var_y == 0:
        return 0.0, var_x / n, var_y / n

    corr = cov / (math.sqrt(var_x) * math.sqrt(var_y))
    return corr, var_x / n, var_y / n


def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from a market question.

    Strips stop words and short tokens to produce a set of
    content-bearing keywords for semantic similarity comparison.
    """
    words = set()
    for word in text.lower().split():
        # Strip punctuation
        cleaned = word.strip("?.,!:;\"'()[]{}/-")
        if len(cleaned) > 2 and cleaned not in _STOP_WORDS:
            words.add(cleaned)
    return words


class CorrelationArbStrategy(BaseStrategy):
    """
    Correlation Arbitrage: Mean-reversion on correlated market pair spreads.

    Maintains rolling price history per market, calculates Pearson correlation
    for market pairs, and when a historically correlated pair diverges
    (z-score > 2.0), flags a convergence trade opportunity.

    This is a statistical edge strategy, not risk-free arbitrage.
    """

    strategy_type = StrategyType.CORRELATION_ARB
    name = "Correlation Arbitrage"
    description = "Mean-reversion on correlated market pair spreads"

    def __init__(self):
        super().__init__()
        # market_id -> deque of (timestamp, yes_price)
        self._price_history: dict[str, deque[tuple[float, float]]] = {}
        # market_id -> event_id
        self._market_to_event: dict[str, str] = {}
        # market_id -> category
        self._market_to_category: dict[str, str] = {}
        # market_id -> Market (latest snapshot)
        self._market_cache: dict[str, Market] = {}
        # Cache of already-checked pairs (reset periodically)
        self._pair_cache: dict[tuple[str, str], float] = {}  # (id_a, id_b) -> correlation
        self._pair_cache_time: float = 0.0

    def detect(self, events: list[Event], markets: list[Market], prices: dict[str, dict]) -> list[ArbitrageOpportunity]:
        if not settings.CORRELATION_ARB_ENABLED:
            return []

        now = time.time()
        min_corr = settings.CORRELATION_ARB_MIN_CORRELATION

        # Build event/category mappings
        for event in events:
            for m in event.markets:
                self._market_to_event[m.id] = event.id
                self._market_to_category[m.id] = event.category or ""

        # Record current prices
        active_ids: list[str] = []
        for market in markets:
            if market.closed or not market.active:
                continue
            if len(market.outcome_prices) < 2:
                continue

            yes_price = self._get_live_price(market, prices)
            self._market_cache[market.id] = market
            active_ids.append(market.id)

            if market.id not in self._price_history:
                self._price_history[market.id] = deque(maxlen=_MAX_HISTORY)
            self._price_history[market.id].append((now, yes_price))

        # Invalidate pair cache every 5 minutes
        if now - self._pair_cache_time > 300:
            self._pair_cache.clear()
            self._pair_cache_time = now

        # Find candidate pairs (same event or same category)
        candidate_pairs = self._find_candidate_pairs(active_ids)

        opportunities: list[ArbitrageOpportunity] = []

        for id_a, id_b in candidate_pairs:
            hist_a = self._price_history.get(id_a)
            hist_b = self._price_history.get(id_b)

            if not hist_a or not hist_b:
                continue
            if len(hist_a) < _MIN_DATA_POINTS or len(hist_b) < _MIN_DATA_POINTS:
                continue

            # Align price series by matching timestamps
            prices_a, prices_b = self._align_series(hist_a, hist_b)
            if len(prices_a) < _MIN_DATA_POINTS:
                continue

            pair_key = (min(id_a, id_b), max(id_a, id_b))
            corr, var_a, var_b = _pearson_correlation(prices_a, prices_b)
            self._pair_cache[pair_key] = corr

            # Reject pairs where either series has near-zero variance.
            # Static prices produce spurious r≈1.0 correlations.
            if var_a < _MIN_PRICE_VARIANCE or var_b < _MIN_PRICE_VARIANCE:
                continue

            if corr < min_corr:
                continue  # Not sufficiently correlated

            # Reject suspiciously perfect correlations (r > 0.98).
            # With sparse data, two unrelated markets that happen to be
            # stable produce r≈1.0 — this is a statistical artifact, not
            # a genuine economic relationship.
            if corr > _MAX_PLAUSIBLE_CORRELATION:
                continue

            # Calculate spread: spread = price_A - corr * price_B
            spreads = [prices_a[i] - corr * prices_b[i] for i in range(len(prices_a))]

            # Z-score of the current spread
            current_spread = spreads[-1]
            mean_spread = sum(spreads) / len(spreads)
            variance = sum((s - mean_spread) ** 2 for s in spreads) / len(spreads)
            std_spread = math.sqrt(variance) if variance > 0 else 0.0

            if std_spread == 0:
                continue

            zscore = (current_spread - mean_spread) / std_spread

            if abs(zscore) < _ZSCORE_THRESHOLD:
                continue  # Not diverged enough

            # Create convergence opportunity
            opp = self._create_convergence_opportunity(
                id_a,
                id_b,
                corr,
                zscore,
                current_spread,
                mean_spread,
                prices,
            )
            if opp:
                opportunities.append(opp)

        if opportunities:
            logger.info(f"Correlation Arb: found {len(opportunities)} divergence opportunity/ies")

        return opportunities

    def _get_live_price(self, market: Market, prices: dict[str, dict]) -> float:
        """Get the best available YES price for a market."""
        yes_price = market.yes_price
        if market.clob_token_ids and len(market.clob_token_ids) > 0:
            token = market.clob_token_ids[0]
            if token in prices:
                yes_price = prices[token].get("mid", yes_price)
        return yes_price

    def _find_candidate_pairs(self, active_ids: list[str]) -> list[tuple[str, str]]:
        """
        Find candidate market pairs that could be correlated.

        Pairs markets that share the same event (always eligible) or
        the same category (only if their questions share keyword overlap,
        to prevent pairing completely unrelated markets like different
        sports games that happen to be in the same category).
        """
        pairs: set[tuple[str, str]] = set()

        # Group by event — same-event markets are always valid candidates
        event_groups: dict[str, list[str]] = {}
        for mid in active_ids:
            eid = self._market_to_event.get(mid)
            if eid:
                if eid not in event_groups:
                    event_groups[eid] = []
                event_groups[eid].append(mid)

        for eid, group in event_groups.items():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    pair_key = (min(group[i], group[j]), max(group[i], group[j]))
                    pairs.add(pair_key)

        # Group by category — require keyword overlap to avoid pairing
        # unrelated markets (e.g., different hockey games under "Sports")
        cat_groups: dict[str, list[str]] = {}
        for mid in active_ids:
            cat = self._market_to_category.get(mid, "")
            if cat:
                if cat not in cat_groups:
                    cat_groups[cat] = []
                cat_groups[cat].append(mid)

        for cat, group in cat_groups.items():
            # Limit category groups to avoid O(n^2) explosion
            if len(group) > 50:
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    # Require semantic similarity: market questions must
                    # share at least _MIN_KEYWORD_OVERLAP content words.
                    # This prevents pairing "Will Hurricanes win?" with
                    # "Will Sabres win?" — different games, not correlated.
                    mkt_i = self._market_cache.get(group[i])
                    mkt_j = self._market_cache.get(group[j])
                    if mkt_i and mkt_j:
                        kw_i = _extract_keywords(mkt_i.question)
                        kw_j = _extract_keywords(mkt_j.question)
                        overlap = len(kw_i & kw_j)
                        if overlap < _MIN_KEYWORD_OVERLAP:
                            continue
                    pair_key = (min(group[i], group[j]), max(group[i], group[j]))
                    pairs.add(pair_key)

        return list(pairs)

    def _align_series(
        self,
        hist_a: deque[tuple[float, float]],
        hist_b: deque[tuple[float, float]],
    ) -> tuple[list[float], list[float]]:
        """
        Align two price histories by matching timestamps.

        Since both deques are populated on the same scan cycle, timestamps
        should align closely. We match entries with timestamps within
        5 seconds of each other.
        """
        prices_a: list[float] = []
        prices_b: list[float] = []

        list_a = list(hist_a)
        list_b = list(hist_b)

        j = 0
        for ts_a, price_a in list_a:
            while j < len(list_b) and list_b[j][0] < ts_a - 5.0:
                j += 1
            if j < len(list_b) and abs(list_b[j][0] - ts_a) <= 5.0:
                prices_a.append(price_a)
                prices_b.append(list_b[j][1])
                j += 1

        return prices_a, prices_b

    def _create_convergence_opportunity(
        self,
        id_a: str,
        id_b: str,
        correlation: float,
        zscore: float,
        current_spread: float,
        mean_spread: float,
        prices: dict[str, dict],
    ) -> Optional[ArbitrageOpportunity]:
        """
        Create a convergence trade opportunity.

        When z-score > 0: A is overpriced relative to B -> buy B (YES), fade A (buy NO)
        When z-score < 0: A is underpriced relative to B -> buy A (YES), fade B (buy NO)
        """
        market_a = self._market_cache.get(id_a)
        market_b = self._market_cache.get(id_b)

        if not market_a or not market_b:
            return None

        yes_a = self._get_live_price(market_a, prices)
        yes_b = self._get_live_price(market_b, prices)

        no_a = market_a.no_price
        if market_a.clob_token_ids and len(market_a.clob_token_ids) > 1:
            token = market_a.clob_token_ids[1]
            if token in prices:
                no_a = prices[token].get("mid", no_a)

        no_b = market_b.no_price
        if market_b.clob_token_ids and len(market_b.clob_token_ids) > 1:
            token = market_b.clob_token_ids[1]
            if token in prices:
                no_b = prices[token].get("mid", no_b)

        q_a = market_a.question[:50]
        q_b = market_b.question[:50]

        positions = []

        if zscore > 0:
            # A is overpriced relative to B
            # Buy YES on B (underperformer), buy NO on A (fade outperformer)
            buy_price = yes_b
            fade_price = no_a

            buy_token = market_b.clob_token_ids[0] if market_b.clob_token_ids else None
            fade_token = (
                market_a.clob_token_ids[1] if market_a.clob_token_ids and len(market_a.clob_token_ids) > 1 else None
            )

            positions = [
                {
                    "action": "BUY",
                    "outcome": "YES",
                    "market": q_b,
                    "price": buy_price,
                    "token_id": buy_token,
                    "rationale": f"Underperformer in correlated pair (r={correlation:.2f})",
                },
                {
                    "action": "BUY",
                    "outcome": "NO",
                    "market": q_a,
                    "price": fade_price,
                    "token_id": fade_token,
                    "rationale": f"Fade outperformer (z-score={zscore:.2f})",
                },
            ]
        else:
            # A is underpriced relative to B
            # Buy YES on A (underperformer), buy NO on B (fade outperformer)
            buy_price = yes_a
            fade_price = no_b

            buy_token = market_a.clob_token_ids[0] if market_a.clob_token_ids else None
            fade_token = (
                market_b.clob_token_ids[1] if market_b.clob_token_ids and len(market_b.clob_token_ids) > 1 else None
            )

            positions = [
                {
                    "action": "BUY",
                    "outcome": "YES",
                    "market": q_a,
                    "price": buy_price,
                    "token_id": buy_token,
                    "rationale": f"Underperformer in correlated pair (r={correlation:.2f})",
                },
                {
                    "action": "BUY",
                    "outcome": "NO",
                    "market": q_b,
                    "price": fade_price,
                    "token_id": fade_token,
                    "rationale": f"Fade outperformer (z-score={zscore:.2f})",
                },
            ]

        total_cost = buy_price + fade_price

        # Skip if total cost is unreasonable
        if total_cost < 0.10 or total_cost > 1.90:
            return None

        # Risk score: 0.40 - 0.55
        # Higher correlation = more reliable signal = lower risk
        base_risk = 0.55
        corr_adjustment = (correlation - 0.7) * 0.5  # Up to 0.15 reduction
        zscore_adjustment = min((abs(zscore) - 2.0) * 0.05, 0.10)  # Stronger signal
        risk_score = max(base_risk - corr_adjustment - zscore_adjustment, 0.40)

        opp = self.create_opportunity(
            title=f"Correlation Arb: {q_a[:25]} vs {q_b[:25]}",
            description=(
                f"Correlated pair diverged (r={correlation:.2f}, z={zscore:.2f}). "
                f"Spread: {current_spread:.3f} vs mean {mean_spread:.3f}. "
                f"A: ${yes_a:.3f}, B: ${yes_b:.3f}. "
                f"Bet on convergence."
            ),
            total_cost=total_cost,
            markets=[market_a, market_b],
            positions=positions,
            is_guaranteed=False,
        )

        if opp:
            opp.risk_score = risk_score
            opp.risk_factors.insert(
                0,
                "STATISTICAL EDGE — not arbitrage. Convergence is not guaranteed; losses possible.",
            )
            opp.risk_factors.append(
                f"Statistical edge (not risk-free): correlation r={correlation:.2f}, z-score={zscore:.2f}"
            )
            opp.risk_factors.append("Pair correlation may break down (regime change risk)")
            if abs(zscore) > 3.0:
                opp.risk_factors.append(
                    f"Extreme divergence (z={zscore:.2f}): may indicate structural break, not mean-reversion"
                )
            if correlation > 0.95:
                opp.risk_factors.append(
                    f"Very high correlation (r={correlation:.2f}): verify pair has genuine economic relationship"
                )

        return opp
