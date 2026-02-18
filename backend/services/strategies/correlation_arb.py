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

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Optional

from models import Market, Event, Opportunity
from config import settings
from .base import BaseStrategy, DecisionCheck, ExitDecision, ScoringWeights, SizingConfig
from services.quality_filter import QualityFilterOverrides
from utils.kelly import kelly_fraction
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

    strategy_type = "correlation_arb"
    name = "Correlation Arbitrage"
    description = "Mean-reversion on correlated market pair spreads"
    mispricing_type = "cross_market"
    subscriptions = ["market_data_refresh"]
    realtime_processing_mode = "full_snapshot"

    quality_filter_overrides = QualityFilterOverrides(
        min_roi=2.0,
        max_resolution_months=3.0,
    )

    default_config = {
        "min_edge_percent": 4.0,
        "min_confidence": 0.40,
        "max_risk_score": 0.75,
        "min_correlation": 0.6,
        "z_score_threshold": 2.0,
        "base_size_usd": 15.0,
        "max_size_usd": 120.0,
    }

    pipeline_defaults = {
        "min_edge_percent": 3.0,
        "min_confidence": 0.42,
        "max_risk_score": 0.75,
        "base_size_usd": 18.0,
        "max_size_usd": 150.0,
    }

    # Composable evaluate pipeline: score = edge*0.60 + conf*30 - risk*9
    scoring_weights = ScoringWeights(
        edge_weight=0.60,
        confidence_weight=30.0,
        risk_penalty=9.0,
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

    def __init__(self):
        super().__init__()
        # Transient cache rebuilt each scan (Market objects are not serializable)
        self._market_cache: dict[str, Market] = {}

    def detect(self, events: list[Event], markets: list[Market], prices: dict[str, dict]) -> list[Opportunity]:
        if not settings.CORRELATION_ARB_ENABLED:
            return []

        now = time.time()
        min_corr = settings.CORRELATION_ARB_MIN_CORRELATION
        min_divergence = max(0.0, float(settings.CORRELATION_ARB_MIN_DIVERGENCE))

        # Cross-cycle persistence via self.state
        price_history = self.state.setdefault("price_history", {})
        market_to_event = self.state.setdefault("market_to_event", {})
        market_to_category = self.state.setdefault("market_to_category", {})
        spread_history = self.state.setdefault("spread_history", {})

        # Build event/category mappings
        for event in events:
            for m in event.markets:
                market_to_event[m.id] = event.id
                market_to_category[m.id] = event.category or ""

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

            if market.id not in price_history:
                price_history[market.id] = deque(maxlen=_MAX_HISTORY)
            price_history[market.id].append((now, yes_price))

        # Invalidate spread_history (pair cache) every 5 minutes
        pair_cache_time = self.state.get("pair_cache_time", 0.0)
        if now - pair_cache_time > 300:
            spread_history.clear()
            self.state["pair_cache_time"] = now

        # Find candidate pairs (same event or same category)
        candidate_pairs = self._find_candidate_pairs(active_ids)

        opportunities: list[Opportunity] = []

        for id_a, id_b in candidate_pairs:
            hist_a = price_history.get(id_a)
            hist_b = price_history.get(id_b)

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
            spread_history[pair_key] = corr

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

            # ----------------------------------------------------------
            # Ornstein-Uhlenbeck calibration on the spread series.
            #
            # Instead of a simple rolling z-score we calibrate an OU
            # process to the spread history.  This gives us:
            #   theta - mean reversion speed
            #   mu    - calibrated long-run equilibrium
            #   sigma - OU volatility
            #
            # The half-life (ln2/theta) filters out pairs where mean
            # reversion is too slow to be tradeable, and the OU z-score
            # provides a more principled entry signal than the naive
            # (spread - mean) / std formula.
            # ----------------------------------------------------------
            theta, ou_mu, ou_sigma = self._calibrate_ou(spreads)
            hl = self._half_life(theta)

            current_spread = spreads[-1]

            # If OU calibration succeeded (theta > 0), use OU-based
            # z-score and half-life filter.  Otherwise fall back to the
            # simple rolling z-score so we never silently drop pairs
            # just because the series is too short for calibration.
            if theta > 0 and ou_sigma > 0:
                # Filter: skip pairs where mean reversion is too slow.
                # A half-life beyond 50 observations means the spread
                # takes too long to converge for a practical trade.
                _MAX_HALF_LIFE = 50.0
                if hl > _MAX_HALF_LIFE:
                    continue

                # OU z-score: deviation from calibrated equilibrium
                zscore = self._ou_z_score(current_spread, ou_mu, ou_sigma)
                mean_spread = ou_mu
            else:
                # Fallback: simple rolling z-score
                mean_spread = sum(spreads) / len(spreads)
                variance = sum((s - mean_spread) ** 2 for s in spreads) / len(spreads)
                std_spread = math.sqrt(variance) if variance > 0 else 0.0

                if std_spread == 0:
                    continue

                zscore = (current_spread - mean_spread) / std_spread

            if abs(zscore) < _ZSCORE_THRESHOLD:
                continue  # Not diverged enough
            divergence = abs(current_spread - mean_spread)
            if divergence < min_divergence:
                continue  # Absolute spread move is too small

            # Create convergence opportunity
            opp = self._create_convergence_opportunity(
                id_a,
                id_b,
                corr,
                zscore,
                divergence,
                current_spread,
                mean_spread,
                prices,
                ou_theta=theta,
                ou_half_life=hl,
                ou_sigma=ou_sigma,
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

        market_to_event = self.state.setdefault("market_to_event", {})
        market_to_category = self.state.setdefault("market_to_category", {})

        # Group by event — same-event markets are always valid candidates
        event_groups: dict[str, list[str]] = {}
        for mid in active_ids:
            eid = market_to_event.get(mid)
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
            cat = market_to_category.get(mid, "")
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

    # ------------------------------------------------------------------
    # Ornstein-Uhlenbeck mean-reversion model
    #
    # The spread between correlated pairs is modelled as an OU process:
    #
    #   dx_t = theta * (mu - x_t) * dt + sigma * dW_t
    #
    # where theta = mean-reversion speed, mu = long-run equilibrium,
    # sigma = volatility.  Parameters are calibrated from the spread
    # history via OLS regression on consecutive observations.
    #
    # Half-life = ln(2) / theta gives the expected time for a
    # deviation to halve -- pairs with very long half-lives are poor
    # candidates for convergence trades.
    #
    # The OU z-score replaces the simple z-score for entry signals,
    # giving a more principled measure of how far the spread has
    # deviated from its calibrated equilibrium.
    # ------------------------------------------------------------------

    @staticmethod
    def _calibrate_ou(spread_history: list[float], dt: float = 1.0) -> tuple[float, float, float]:
        """Calibrate Ornstein-Uhlenbeck parameters from spread history.

        dx_t = theta * (mu - x_t) * dt + sigma * dW_t

        Returns:
            (theta, mu, sigma) - mean reversion speed, long-run mean, volatility
        """
        if len(spread_history) < 10:
            return (0.0, 0.0, 0.0)

        x = spread_history[:-1]
        y = spread_history[1:]

        # OLS regression: y = a + b*x + epsilon
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi ** 2 for xi in x)
        n_pts = len(x)

        denom = n_pts * sum_x2 - sum_x ** 2
        if abs(denom) < 1e-12:
            return (0.0, sum_y / n_pts if n_pts > 0 else 0.0, 0.0)

        b = (n_pts * sum_xy - sum_x * sum_y) / denom
        a = (sum_y - b * sum_x) / n_pts

        # OU parameters from regression
        # b = exp(-theta * dt), so theta = -ln(b) / dt
        if b <= 0 or b >= 1:
            return (0.0, sum_y / n_pts, 0.0)

        theta = -math.log(b) / dt
        mu = a / (1.0 - b)

        # Residual variance for sigma
        residuals = [yi - (a + b * xi) for xi, yi in zip(x, y)]
        var_resid = sum(r ** 2 for r in residuals) / max(1, len(residuals))
        sigma = math.sqrt(var_resid * 2.0 * theta / (1.0 - b ** 2)) if (1.0 - b ** 2) > 0 else 0.0

        return (theta, mu, sigma)

    @staticmethod
    def _half_life(theta: float) -> float:
        """Half-life of mean reversion in same units as dt."""
        if theta <= 0:
            return float('inf')
        return math.log(2) / theta

    @staticmethod
    def _ou_z_score(current_spread: float, mu: float, sigma: float) -> float:
        """Z-score of current spread relative to OU equilibrium."""
        if sigma <= 0:
            return 0.0
        return (current_spread - mu) / sigma

    def _create_convergence_opportunity(
        self,
        id_a: str,
        id_b: str,
        correlation: float,
        zscore: float,
        divergence: float,
        current_spread: float,
        mean_spread: float,
        prices: dict[str, dict],
        ou_theta: float = 0.0,
        ou_half_life: float = float('inf'),
        ou_sigma: float = 0.0,
    ) -> Optional[Opportunity]:
        """
        Create a convergence trade opportunity.

        When z-score > 0: A is overpriced relative to B -> buy B (YES), fade A (buy NO)
        When z-score < 0: A is underpriced relative to B -> buy A (YES), fade B (buy NO)

        OU parameters (when calibrated) enrich the opportunity description
        and risk assessment:
            ou_theta     - mean reversion speed (higher = faster reversion)
            ou_half_life - expected periods for deviation to halve
            ou_sigma     - OU process volatility
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

        # OU half-life adjustment: faster mean reversion = lower risk.
        # A half-life under 10 observations is quite fast and reduces
        # risk; above 30 is sluggish and adds risk.
        if ou_theta > 0:
            if ou_half_life < 10:
                hl_adjustment = 0.05
            elif ou_half_life < 20:
                hl_adjustment = 0.02
            elif ou_half_life > 30:
                hl_adjustment = -0.03  # Slow reversion increases risk
            else:
                hl_adjustment = 0.0
        else:
            hl_adjustment = 0.0  # No OU data, no adjustment

        risk_score = max(base_risk - corr_adjustment - zscore_adjustment - hl_adjustment, 0.40)

        # Build description with OU model metrics when available
        ou_label = ""
        if ou_theta > 0:
            ou_label = (
                f" OU: theta={ou_theta:.3f}, half-life={ou_half_life:.1f}, "
                f"sigma={ou_sigma:.4f}."
            )

        opp = self.create_opportunity(
            title=f"Correlation Arb: {q_a[:25]} vs {q_b[:25]}",
            description=(
                f"Correlated pair diverged (r={correlation:.2f}, z={zscore:.2f}). "
                f"Spread: {current_spread:.3f} vs mean {mean_spread:.3f} "
                f"(|delta|={divergence:.3f}). "
                f"A: ${yes_a:.3f}, B: ${yes_b:.3f}. "
                f"Bet on convergence.{ou_label}"
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
                "STATISTICAL EDGE -- not arbitrage. Convergence is not guaranteed; losses possible.",
            )
            opp.risk_factors.append(
                f"Statistical edge (not risk-free): correlation r={correlation:.2f}, z-score={zscore:.2f}"
            )
            opp.risk_factors.append("Pair correlation may break down (regime change risk)")
            if ou_theta > 0:
                opp.risk_factors.append(
                    f"OU model: theta={ou_theta:.3f}, half-life={ou_half_life:.1f} obs, sigma={ou_sigma:.4f}"
                )
                if ou_half_life > 30:
                    opp.risk_factors.append(
                        f"Slow mean reversion (half-life={ou_half_life:.1f}): convergence may take many observations"
                    )
            if abs(zscore) > 3.0:
                opp.risk_factors.append(
                    f"Extreme divergence (z={zscore:.2f}): may indicate structural break, not mean-reversion"
                )
            if correlation > 0.95:
                opp.risk_factors.append(
                    f"Very high correlation (r={correlation:.2f}): verify pair has genuine economic relationship"
                )

        return opp

    def compute_size(
        self, base_size: float, max_size: float, edge: float, confidence: float, risk_score: float, market_count: int
    ) -> float:
        """Kelly-informed sizing for correlation arbitrage."""
        p_estimated = 0.5 + (edge / 200.0)
        p_market = 0.5
        kelly_f = kelly_fraction(p_estimated, p_market, fraction=0.25)
        kelly_sz = base_size * (1.0 + kelly_f * 10.0)
        size = kelly_sz * (0.7 + confidence * 0.6) * max(0.4, 1.0 - risk_score)
        return max(1.0, min(max_size, size))

    def custom_checks(self, signal, context, params, payload):
        source = str(getattr(signal, "source", "") or "").strip().lower()
        return [
            DecisionCheck("source", "Signal source", source == "scanner", detail=f"got={source}"),
        ]

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        """Correlation arb: exit when spread mean-reverts or standard TP/SL."""
        if market_state.get("is_resolved"):
            return self.default_exit_check(position, market_state)
        return self.default_exit_check(position, market_state)

    # ------------------------------------------------------------------
    # Platform gate hooks
    # ------------------------------------------------------------------

    def on_blocked(self, signal, reason: str, context: dict) -> None:
        logger.info("%s: signal blocked — %s (market=%s)", self.name, reason, getattr(signal, "market_id", "?"))

    def on_size_capped(self, original_size: float, capped_size: float, reason: str) -> None:
        logger.info("%s: size capped $%.0f → $%.0f — %s", self.name, original_size, capped_size, reason)
