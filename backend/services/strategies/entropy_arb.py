"""
Strategy: Entropy Arbitrage - NegRisk Rebalancing with Entropy Quality

Finds multiway/NegRisk events where the sum of YES prices is less than
$1.00 (guaranteed profit by buying all outcomes), ranked by information-
theoretic quality signals.

This is what real prediction market arbitrage bots do. The documented
$40M+ in Polymarket arbitrage profits came from:
  1. NegRisk rebalancing (sum of YES < $1.00, buy all): $29M (73%)
  2. Binary parity (YES + NO < $1.00): $10.6M (27%)

Entropy analysis is used as a QUALITY SIGNAL to rank opportunities,
not as a primary detection trigger. Shannon entropy and KL divergence
measure how anomalous a distribution is, helping distinguish genuine
mispricings from stale order books.

Previous approach (REMOVED):
  - Used entropy deviation to recommend buying SINGLE longshot outcomes
  - This produced directional bets with 100% loss risk, not arbitrage
  - Every opportunity was correctly rated "STRONG SKIP" by AI analysis

Current approach:
  - Detect multiway events where sum(YES) < $1.00 (structural arbitrage)
  - Buy ALL outcomes = guaranteed $1.00 payout
  - Use entropy deviation as ranking/quality signal
  - Less strict exhaustiveness checks than NegRisk strategy (wider net)
  - Clear risk labeling for non-exhaustive outcome sets
"""

from __future__ import annotations

import math
from typing import Optional

from models import Market, Event, ArbitrageOpportunity, StrategyType
from config import settings
from .base import BaseStrategy, make_aware, utcnow


# ---------------------------------------------------------------------------
# Pure entropy math (retained for quality scoring)
# ---------------------------------------------------------------------------


def binary_entropy(p: float) -> float:
    """Shannon entropy of a binary distribution in [0, 1] bits."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def multi_outcome_entropy(probs: list[float]) -> float:
    """Shannon entropy of a discrete distribution with N outcomes."""
    h = 0.0
    for p in probs:
        if p > 0.0:
            h -= p * math.log2(p)
    return h


def max_entropy(n: int) -> float:
    """Maximum entropy for a uniform distribution over n outcomes."""
    if n <= 1:
        return 0.0
    return math.log2(n)


def kl_divergence(p: list[float], q: list[float]) -> float:
    """KL divergence D_KL(P || Q) between two discrete distributions."""
    eps = 1e-12
    d = 0.0
    for pi, qi in zip(p, q):
        pi_safe = max(pi, eps)
        qi_safe = max(qi, eps)
        d += pi_safe * math.log2(pi_safe / qi_safe)
    return max(d, 0.0)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

# Minimum total YES to consider (below this, non-exhaustive risk is too high)
MIN_TOTAL_YES = 0.88

# Election markets need higher threshold (unlisted candidates are common)
ELECTION_MIN_TOTAL_YES = 0.95

# Maximum resolution date spread across outcomes in a bundle (days)
MAX_RESOLUTION_SPREAD_DAYS = 14

# Keywords that suggest election/primary markets
_ELECTION_KEYWORDS = frozenset({
    "election", "primary", "winner", "nominee", "nomination",
    "governor", "senate", "house", "president", "congressional",
    "mayoral", "caucus", "runoff", "special election",
})

# Keywords that suggest open-ended events (outcome universe unbounded)
_OPEN_ENDED_KEYWORDS = frozenset({
    "nobel", "oscar", "grammy", "emmy", "pulitzer", "ballon d'or",
    "mvp", "best picture", "album of the year", "song of the year",
    "time person", "word of the year", "best", "award",
})

DEFAULT_TOTAL_DAYS = 90
ENTROPY_DECAY_ALPHA = 0.5


class EntropyArbStrategy(BaseStrategy):
    """NegRisk rebalancing with entropy-based quality ranking.

    Scans multiway events for structural arbitrage: buy all outcomes
    when sum(YES) < $1.00, guaranteeing profit since exactly one
    outcome must win.

    Uses Shannon entropy as a quality signal to rank opportunities.
    Markets with anomalous entropy (too uncertain or too concentrated
    given time to resolution) are more likely to be genuinely mispriced
    rather than reflecting stale order books.
    """

    strategy_type = StrategyType.ENTROPY_ARB
    name = "Entropy Signal"
    description = "NegRisk rebalancing ranked by entropy quality"

    def __init__(self) -> None:
        super().__init__()
        self._prev_entropies: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Expected entropy model (for quality scoring)
    # ------------------------------------------------------------------

    @staticmethod
    def _expected_entropy_multi(
        n_outcomes: int, days_remaining: float, total_days: float
    ) -> float:
        """Expected entropy for a multi-outcome market."""
        h_max = max_entropy(n_outcomes)
        if total_days <= 0:
            return 0.0
        ratio = min(max(days_remaining, 0.0) / total_days, 1.0)
        return h_max * (ratio ** ENTROPY_DECAY_ALPHA)

    # ------------------------------------------------------------------
    # Price helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_live_yes_price(market: Market, prices: dict[str, dict]) -> float:
        yes_price = market.yes_price
        if market.clob_token_ids and len(market.clob_token_ids) > 0:
            token = market.clob_token_ids[0]
            if token in prices:
                yes_price = prices[token].get("mid", yes_price)
        return yes_price

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_election_market(title: str) -> bool:
        title_lower = title.lower()
        return any(kw in title_lower for kw in _ELECTION_KEYWORDS)

    @staticmethod
    def _is_open_ended_event(title: str) -> bool:
        title_lower = title.lower()
        return any(kw in title_lower for kw in _OPEN_ENDED_KEYWORDS)

    @staticmethod
    def _is_date_sweep(event: Event) -> bool:
        """Detect cumulative date-based events like 'by March/June/Dec'."""
        questions = [m.question.lower() for m in event.markets if m.question]
        date_keywords = {"by ", "before ", "prior to "}
        date_count = sum(
            1 for q in questions if any(kw in q for kw in date_keywords)
        )
        return date_count >= len(questions) * 0.5

    # ------------------------------------------------------------------
    # Entropy quality score
    # ------------------------------------------------------------------

    def _compute_entropy_quality(
        self, event_id: str, probs: list[float], n_outcomes: int,
        days_remaining: float, total_days: float,
    ) -> dict:
        """Compute entropy-based quality metrics for an event.

        Returns a dict with entropy values, deviation, spike detection,
        and an overall quality score (0-1, higher = more anomalous).
        """
        h_actual = multi_outcome_entropy(probs)
        h_expected = self._expected_entropy_multi(
            n_outcomes, days_remaining, total_days
        )
        h_max = max_entropy(n_outcomes)
        deviation = h_actual - h_expected

        # Normalized entropy (how spread vs concentrated)
        normalized = h_actual / h_max if h_max > 0 else 0.0

        # KL divergence from uniform distribution
        uniform = [1.0 / n_outcomes] * n_outcomes
        kl_from_uniform = kl_divergence(probs, uniform)

        # Cross-scan spike detection
        event_key = f"event_{event_id}"
        spike = False
        prev_h = self._prev_entropies.get(event_key)
        if prev_h is not None:
            delta_h = h_actual - prev_h
            if delta_h >= 0.20:
                spike = True
        self._prev_entropies[event_key] = h_actual

        # Quality score: how anomalous is this distribution?
        # Higher = more likely to be a genuine mispricing
        quality = 0.0
        # Large entropy deviation = stronger signal
        quality += min(abs(deviation) / 0.5, 1.0) * 0.4
        # Spike = recent change, more likely exploitable
        if spike:
            quality += 0.2
        # High KL divergence from uniform = more concentrated (clearer signal)
        quality += min(kl_from_uniform / 2.0, 1.0) * 0.2
        # Markets near resolution with high entropy = strongest signal
        if days_remaining < 30 and deviation > 0:
            quality += 0.2
        quality = min(quality, 1.0)

        return {
            "h_actual": round(h_actual, 4),
            "h_expected": round(h_expected, 4),
            "h_max": round(h_max, 4),
            "deviation": round(deviation, 4),
            "normalized": round(normalized, 4),
            "kl_from_uniform": round(kl_from_uniform, 4),
            "spike": spike,
            "quality": round(quality, 4),
        }

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[ArbitrageOpportunity]:
        if not settings.ENTROPY_ARB_ENABLED:
            return []

        opportunities: list[ArbitrageOpportunity] = []

        for event in events:
            opp = self._detect_rebalancing(event, prices)
            if opp is not None:
                opportunities.append(opp)

        return opportunities

    def _detect_rebalancing(
        self,
        event: Event,
        prices: dict[str, dict],
    ) -> Optional[ArbitrageOpportunity]:
        """Detect NegRisk rebalancing opportunity in a multiway event.

        Only creates an opportunity when sum(YES) < $1.00, meaning
        buying all outcomes guarantees profit. This is real arbitrage.
        """
        if event.closed:
            return None

        active_markets = [m for m in event.markets if m.active and not m.closed]
        if len(active_markets) < 2:
            return None

        # Skip date-sweep events (cumulative, not mutually exclusive)
        if self._is_date_sweep(event):
            return None

        # Get YES prices for all outcomes
        total_yes = 0.0
        positions = []
        yes_prices: list[float] = []

        for market in active_markets:
            yes_price = self._get_live_yes_price(market, prices)
            total_yes += yes_price
            yes_prices.append(yes_price)
            positions.append({
                "action": "BUY",
                "outcome": "YES",
                "market": market.question[:50],
                "price": yes_price,
                "token_id": (
                    market.clob_token_ids[0]
                    if market.clob_token_ids
                    else None
                ),
            })

        # Core arbitrage condition: sum < $1.00
        if total_yes >= 1.0:
            return None

        # Non-exhaustive outcome checks (same logic as NegRisk but less strict)
        if total_yes < MIN_TOTAL_YES:
            return None

        n_outcomes = len(active_markets)

        # Very few outcomes = likely non-exhaustive
        if n_outcomes <= 3 and total_yes < 0.95:
            return None
        if n_outcomes <= 5 and total_yes < 0.93:
            return None

        is_election = self._is_election_market(event.title)
        is_open_ended = self._is_open_ended_event(event.title)

        # Election with only 2 candidates = never exhaustive
        if is_election and n_outcomes <= 2:
            return None

        # Election markets need higher threshold
        if is_election and total_yes < ELECTION_MIN_TOTAL_YES:
            return None

        # Open-ended events: accept only if very close to 1.0
        if is_open_ended and total_yes < 0.97:
            return None

        # Resolution date spread check
        end_dates = [
            make_aware(m.end_date) for m in active_markets if m.end_date
        ]
        if len(end_dates) >= 2:
            earliest = min(end_dates)
            latest = max(end_dates)
            spread_days = (latest - earliest).days
            if spread_days > MAX_RESOLUTION_SPREAD_DAYS:
                return None

        # Compute entropy quality score for ranking
        now = utcnow()
        days_remaining = 0.0
        total_days = DEFAULT_TOTAL_DAYS
        if end_dates:
            resolution = min(end_dates)
            days_remaining = max(
                (resolution - now).total_seconds() / 86400.0, 0.0
            )
            total_days = max(days_remaining, DEFAULT_TOTAL_DAYS)

        # Normalize prices to probability distribution for entropy calc
        probs = [p / total_yes for p in yes_prices] if total_yes > 0 else []

        entropy_info = self._compute_entropy_quality(
            event.id, probs, n_outcomes, days_remaining, total_days,
        )

        # Build description
        spread_pct = (1.0 - total_yes) * 100
        description = (
            f"Buy YES on all {n_outcomes} outcomes for ${total_yes:.3f}, "
            f"one must win = $1.00. Guaranteed ${1.0 - total_yes:.3f} profit "
            f"({spread_pct:.1f}% spread). "
            f"Entropy quality: {entropy_info['quality']:.0%} "
            f"(H={entropy_info['h_actual']:.2f}/{entropy_info['h_max']:.2f}, "
            f"dev={entropy_info['deviation']:+.3f})."
        )

        # Add entropy info to positions metadata
        for pos in positions:
            pos["entropy_quality"] = entropy_info["quality"]

        neg_risk_label = "NegRisk " if event.neg_risk else ""
        opp = self.create_opportunity(
            title=(
                f"Entropy Arb: {neg_risk_label}"
                f"{event.title[:50]}..."
            ),
            description=description,
            total_cost=total_yes,
            markets=active_markets,
            positions=positions,
            event=event,
            is_guaranteed=True,
        )

        if opp is not None:
            # Add entropy-specific risk factors
            if not event.neg_risk:
                opp.risk_factors.insert(
                    0,
                    "Not a NegRisk-flagged event: mutual exclusivity "
                    "not platform-guaranteed. Verify outcomes are exhaustive.",
                )
            if total_yes < 0.93:
                opp.risk_factors.insert(
                    0,
                    f"Total YES ({total_yes:.1%}) well below 100% -- "
                    f"significant non-exhaustive risk (unlisted outcomes).",
                )
            elif total_yes < 0.97:
                opp.risk_factors.append(
                    f"Total YES ({total_yes:.1%}) below 97% -- "
                    f"possible unlisted outcomes.",
                )
            if is_election:
                opp.risk_factors.append(
                    "Election/primary: unlisted candidates may win."
                )
            if is_open_ended:
                opp.risk_factors.append(
                    "Open-ended event: outcome universe unbounded."
                )
            if entropy_info["spike"]:
                opp.risk_factors.append(
                    "Entropy spike detected (distribution changed "
                    "significantly since last scan)."
                )

        return opp
