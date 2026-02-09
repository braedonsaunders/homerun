"""
Strategy: Entropy Arbitrage - Information-Theoretic Mispricing

Uses Shannon entropy and KL divergence to detect markets whose
probability distributions deviate from expected patterns.

Key insight: As markets approach resolution, entropy should decrease
(uncertainty reduces). Markets with anomalously HIGH entropy near
resolution are mispriced - the market hasn't priced in available
information yet.

Conversely, markets with anomalously LOW entropy far from resolution
may be overconfident and vulnerable to mean reversion.

This has never been implemented in a prediction market bot.

Mathematical foundation:
- H(X) = -sum(p * log2(p)) for probability distribution
- For binary market: H = -p*log2(p) - (1-p)*log2(1-p)
- Maximum entropy at p=0.5 (H=1.0 bit), minimum at p=0/1 (H=0)
- Expected entropy decay: H(t) ~ H_max * (days_remaining / total_days)^alpha
"""

from __future__ import annotations

import math
from typing import Optional

from models import Market, Event, ArbitrageOpportunity, StrategyType
from config import settings
from .base import BaseStrategy, utcnow, make_aware


# ---------------------------------------------------------------------------
# Pure entropy math
# ---------------------------------------------------------------------------


def binary_entropy(p: float) -> float:
    """Shannon entropy of a binary distribution.

    H(X) = -p * log2(p) - (1-p) * log2(1-p)

    Returns a value in [0, 1] bits.
    Maximum at p=0.5 (H=1.0), minimum at p in {0, 1} (H=0).
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def multi_outcome_entropy(probs: list[float]) -> float:
    """Shannon entropy of a discrete distribution with N outcomes.

    H(X) = -sum(p_i * log2(p_i))

    Returns value in [0, log2(N)] bits.
    """
    h = 0.0
    for p in probs:
        if p > 0.0:
            h -= p * math.log2(p)
    return h


def max_entropy(n: int) -> float:
    """Maximum entropy for a uniform distribution over n outcomes.

    H_max = log2(n)
    """
    if n <= 1:
        return 0.0
    return math.log2(n)


def kl_divergence(p: list[float], q: list[float]) -> float:
    """KL divergence D_KL(P || Q) between two discrete distributions.

    D_KL(P || Q) = sum(p_i * log2(p_i / q_i))

    Returns value >= 0. Larger means more divergence.
    Uses a small epsilon to avoid log(0).
    """
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

DEFAULT_TOTAL_DAYS = 90  # Assumed market lifespan when creation date unknown
ENTROPY_DECAY_ALPHA = 0.5  # Square-root decay: entropy drops faster near end
ENTROPY_SPIKE_THRESHOLD = 0.20  # Cross-scan spike considered significant


class EntropyArbStrategy(BaseStrategy):
    """Information-theoretic mispricing via entropy analysis.

    IMPORTANT: This is a PROBABILISTIC SIGNAL strategy, NOT risk-free arbitrage.
    All trades are directional bets informed by entropy deviation. If the
    predicted outcome does not occur, the entire position is lost.

    Scans binary markets and multi-outcome NegRisk events for deviations
    between actual entropy (derived from current prices) and the expected
    entropy implied by a square-root decay model tied to resolution timing.

    Anomaly types detected:
    1. **High-entropy anomaly** (deviation > 0): market is too uncertain
       given how close resolution is. The market has not priced in available
       information. Signal: buy the side closer to expected resolution.
    2. **Low-entropy anomaly** (deviation < 0): market is too certain
       given how far resolution is. Overconfidence -> contrarian fade.
    3. **Entropy spike**: cross-scan jump in entropy suggests a resolution
       event is imminent. Pair with the high-entropy signal.
    4. **Multi-outcome entropy anomaly**: NegRisk event distribution
       deviates from the expected entropy for that number of outcomes.
    """

    strategy_type = StrategyType.ENTROPY_ARB
    name = "Entropy Signal"
    description = (
        "Directional edge via information-theoretic entropy analysis (NOT arbitrage)"
    )

    def __init__(self) -> None:
        super().__init__()
        # Track previous-scan entropies for spike detection.
        # Keyed by market id (binary) or event id (multi-outcome).
        self._prev_entropies: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Expected entropy model
    # ------------------------------------------------------------------

    @staticmethod
    def _expected_entropy_binary(days_remaining: float, total_days: float) -> float:
        """Expected entropy for a binary market given time to resolution.

        Model: H_expected = 1.0 * (days_remaining / total_days)^alpha

        Alpha=0.5 gives square-root decay: entropy is still moderate
        halfway through, then drops sharply near resolution — matching
        empirical behaviour of prediction markets.
        """
        if total_days <= 0:
            return 0.0
        ratio = max(days_remaining, 0.0) / total_days
        # Clamp to [0, 1] in case days_remaining > total_days somehow
        ratio = min(ratio, 1.0)
        return 1.0 * (ratio**ENTROPY_DECAY_ALPHA)

    @staticmethod
    def _expected_entropy_multi(
        n_outcomes: int, days_remaining: float, total_days: float
    ) -> float:
        """Expected entropy for a multi-outcome market.

        At creation the maximum entropy is log2(n). As resolution
        approaches, entropy should decay toward 0 (one outcome certain).
        """
        h_max = max_entropy(n_outcomes)
        if total_days <= 0:
            return 0.0
        ratio = max(days_remaining, 0.0) / total_days
        ratio = min(ratio, 1.0)
        return h_max * (ratio**ENTROPY_DECAY_ALPHA)

    # ------------------------------------------------------------------
    # Price helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_live_yes_price(market: Market, prices: dict[str, dict]) -> float:
        """Get the best available YES price for a market."""
        yes_price = market.yes_price
        if market.clob_token_ids and len(market.clob_token_ids) > 0:
            token = market.clob_token_ids[0]
            if token in prices:
                yes_price = prices[token].get("mid", yes_price)
        return yes_price

    @staticmethod
    def _get_live_no_price(market: Market, prices: dict[str, dict]) -> float:
        """Get the best available NO price for a market."""
        no_price = market.no_price
        if market.clob_token_ids and len(market.clob_token_ids) > 1:
            token = market.clob_token_ids[1]
            if token in prices:
                no_price = prices[token].get("mid", no_price)
        return no_price

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[ArbitrageOpportunity]:
        """Detect entropy-based mispricing opportunities.

        Scans:
        1. Every binary market with a known end_date for individual
           entropy anomalies.
        2. Every multi-market NegRisk event for distributional entropy
           anomalies.
        """
        if not settings.ENTROPY_ARB_ENABLED:
            return []

        min_dev = settings.ENTROPY_ARB_MIN_DEVIATION
        opportunities: list[ArbitrageOpportunity] = []

        # --- 1. Binary market entropy anomalies ---
        for market in markets:
            opp = self._detect_binary(market, prices, min_dev)
            if opp is not None:
                opportunities.append(opp)

        # --- 2. Multi-outcome (NegRisk event) entropy anomalies ---
        for event in events:
            opp = self._detect_multi_outcome(event, prices, min_dev)
            if opp is not None:
                opportunities.append(opp)

        return opportunities

    # ------------------------------------------------------------------
    # Binary market detection
    # ------------------------------------------------------------------

    def _detect_binary(
        self,
        market: Market,
        prices: dict[str, dict],
        min_dev: float,
    ) -> Optional[ArbitrageOpportunity]:
        """Detect entropy anomaly on a single binary market."""

        # Only binary, active, open markets with a known resolution date
        if market.closed or not market.active:
            return None
        if len(market.outcome_prices) != 2:
            return None
        if not market.end_date:
            return None

        yes_price = self._get_live_yes_price(market, prices)
        no_price = self._get_live_no_price(market, prices)

        # Skip markets with degenerate prices
        if yes_price <= 0.0 or yes_price >= 1.0:
            return None

        # --- Filter: market already has strong directional consensus ---
        # If the favorite is already priced > 0.75, the market has priced in
        # the expected direction. Entropy deviation at this point is not a
        # mispricing signal — buying an 80-cent favorite is a directional
        # bet with poor risk/reward, not an information edge.
        favorite_price = max(yes_price, 1.0 - yes_price)
        if favorite_price > 0.75:
            return None

        # --- Entropy calculation ---
        h_actual = binary_entropy(yes_price)

        end_aware = make_aware(market.end_date)
        now = utcnow()
        days_remaining = max((end_aware - now).total_seconds() / 86400.0, 0.0)

        # --- Filter: too close to resolution ---
        # Within the last day, prices react to live information (game scores,
        # vote counts, data releases). High entropy here is a feature of
        # genuine uncertainty, not a market inefficiency.
        if days_remaining < 1.0:
            return None

        # Estimate total market lifespan. Without a creation date we
        # default to DEFAULT_TOTAL_DAYS.
        total_days = max(days_remaining, DEFAULT_TOTAL_DAYS)

        h_expected = self._expected_entropy_binary(days_remaining, total_days)

        deviation = h_actual - h_expected

        # --- Cross-scan entropy spike detection ---
        spike = False
        prev_h = self._prev_entropies.get(market.id)
        if prev_h is not None:
            delta_h = h_actual - prev_h
            if delta_h >= ENTROPY_SPIKE_THRESHOLD:
                spike = True
        # Always store current entropy for next scan
        self._prev_entropies[market.id] = h_actual

        # --- Check threshold ---
        if abs(deviation) < min_dev and not spike:
            return None

        # --- Determine trade direction ---
        if deviation > 0:
            # Market is TOO UNCERTAIN (high entropy) near resolution.
            # Expected: price should be further from 0.5.
            # Buy the side closer to the expected resolution direction.
            # Heuristic: if yes_price > 0.5, the market leans YES but not
            # enough — buy YES. If yes_price < 0.5, buy NO.
            if yes_price >= 0.5:
                action, outcome, price = "BUY", "YES", yes_price
                side_label = "YES"
            else:
                action, outcome, price = "BUY", "NO", no_price
                side_label = "NO"
            anomaly_type = "HIGH entropy"
            signal_desc = (
                f"Market too uncertain near resolution. "
                f"H_actual={h_actual:.3f} vs H_expected={h_expected:.3f} "
                f"(+{deviation:.3f} bits). Buy {side_label} to capture info gap."
            )
        else:
            # Market is TOO CERTAIN (low entropy) far from resolution.
            # Overconfidence -> fade the dominant side (contrarian).
            if yes_price >= 0.5:
                # Market is overly confident in YES -> buy NO
                action, outcome, price = "BUY", "NO", no_price
                side_label = "NO (contrarian)"
            else:
                # Market is overly confident in NO -> buy YES
                action, outcome, price = "BUY", "YES", yes_price
                side_label = "YES (contrarian)"
            anomaly_type = "LOW entropy"
            signal_desc = (
                f"Market overconfident far from resolution. "
                f"H_actual={h_actual:.3f} vs H_expected={h_expected:.3f} "
                f"({deviation:.3f} bits). Fade dominant side: buy {side_label}."
            )

        if spike:
            signal_desc += (
                f" Entropy SPIKE detected (prev={prev_h:.3f}, now={h_actual:.3f})."
            )

        # --- Build position ---
        token_id = None
        if market.clob_token_ids:
            if outcome == "YES" and len(market.clob_token_ids) > 0:
                token_id = market.clob_token_ids[0]
            elif outcome == "NO" and len(market.clob_token_ids) > 1:
                token_id = market.clob_token_ids[1]

        positions = [
            {
                "action": action,
                "outcome": outcome,
                "market": market.question[:50],
                "price": price,
                "token_id": token_id,
                "entropy_actual": round(h_actual, 4),
                "entropy_expected": round(h_expected, 4),
                "entropy_deviation": round(deviation, 4),
                "entropy_spike": spike,
            }
        ]

        # total_cost = buy price (the side we're taking)
        total_cost = price

        # --- Create opportunity ---
        # Use create_opportunity which applies all hard filters and fee model.
        # IMPORTANT: This is a DIRECTIONAL BET, not guaranteed arbitrage.
        # The price IS our estimated win probability. ROI is only realized
        # if the bet wins; otherwise the entire position is lost.
        opp = self.create_opportunity(
            title=f"Entropy Signal ({anomaly_type}): {market.question[:45]}...",
            description=signal_desc,
            total_cost=total_cost,
            markets=[market],
            positions=positions,
        )

        if opp is not None:
            # Override risk score: entropy signals are probabilistic
            # directional bets, not guaranteed arbitrage. If the chosen
            # side doesn't win, the entire position is lost.
            base_risk = 0.70
            # Increase risk for smaller deviations (less confident signal)
            deviation_factor = max(0.0, 1.0 - abs(deviation) / 0.5) * 0.10
            # Increase risk for low-entropy (contrarian) trades
            contrarian_penalty = 0.10 if deviation < 0 else 0.0
            # Decrease risk slightly for spike confirmation
            spike_bonus = -0.05 if spike else 0.0

            entropy_risk = (
                base_risk + deviation_factor + contrarian_penalty + spike_bonus
            )
            # Merge with base risk (liquidity, time, multi-leg)
            opp.risk_score = min(max(opp.risk_score, entropy_risk), 1.0)
            opp.risk_factors.append(
                f"Entropy deviation: {deviation:+.3f} bits "
                f"(actual={h_actual:.3f}, expected={h_expected:.3f})"
            )
            if spike:
                opp.risk_factors.append(
                    f"Entropy spike: {prev_h:.3f} -> {h_actual:.3f}"
                )
            opp.risk_factors.insert(
                0,
                "DIRECTIONAL BET — not arbitrage. 100% loss if chosen side loses.",
            )
            opp.risk_factors.append("Probabilistic signal (not guaranteed arbitrage)")

        return opp

    # ------------------------------------------------------------------
    # Multi-outcome (NegRisk) detection
    # ------------------------------------------------------------------

    def _detect_multi_outcome(
        self,
        event: Event,
        prices: dict[str, dict],
        min_dev: float,
    ) -> Optional[ArbitrageOpportunity]:
        """Detect entropy anomaly across a NegRisk event's outcome set.

        For events with N outcomes, the maximum entropy is log2(N).
        We compare actual distribution entropy against an expected
        decay model. Large positive deviation = the market hasn't
        resolved uncertainty that should be resolved by now.
        """
        if not event.neg_risk or event.closed:
            return None

        active_markets = [m for m in event.markets if m.active and not m.closed]
        if len(active_markets) < 2:
            return None

        # We need at least one market with an end_date to compute timing
        end_dates = [
            make_aware(m.end_date) for m in active_markets if m.end_date is not None
        ]
        if not end_dates:
            return None

        # Use the earliest end_date as the event resolution date
        resolution_date = min(end_dates)
        now = utcnow()
        days_remaining = max((resolution_date - now).total_seconds() / 86400.0, 0.0)

        # --- Filter: too close to resolution ---
        # Within the last day, prices react to live information.
        if days_remaining < 1.0:
            return None

        total_days = max(days_remaining, DEFAULT_TOTAL_DAYS)

        # --- Build probability distribution from YES prices ---
        yes_prices: list[float] = []
        for m in active_markets:
            yes_prices.append(self._get_live_yes_price(m, prices))

        total_yes = sum(yes_prices)
        if total_yes <= 0.0:
            return None

        # --- Filter: leading outcome already dominant ---
        # If the favorite is > 0.75, the market has strong consensus.
        # Entropy deviation is not a useful signal at this point.
        # Check BOTH the raw price and the normalized probability,
        # because NegRisk event prices often don't sum to 1.0 —
        # a market at $0.94 can slip past a normalized-only check
        # when the price sum exceeds 1.0 (e.g., 0.94/1.26 = 0.746).
        max_yes = max(yes_prices)
        if max_yes > 0.75 or max_yes / total_yes > 0.75:
            return None

        # Normalize to a proper probability distribution
        probs = [p / total_yes for p in yes_prices]

        n_outcomes = len(active_markets)
        h_actual = multi_outcome_entropy(probs)
        h_expected = self._expected_entropy_multi(
            n_outcomes, days_remaining, total_days
        )

        deviation = h_actual - h_expected

        # --- Cross-scan spike for multi-outcome ---
        event_key = f"event_{event.id}"
        spike = False
        prev_h = self._prev_entropies.get(event_key)
        if prev_h is not None:
            delta_h = h_actual - prev_h
            if delta_h >= ENTROPY_SPIKE_THRESHOLD:
                spike = True
        self._prev_entropies[event_key] = h_actual

        if abs(deviation) < min_dev and not spike:
            return None

        # --- Determine trade ---
        if deviation > 0:
            # Too uncertain: buy the most-likely outcome (highest price),
            # because the market should be more concentrated.
            best_idx = max(range(n_outcomes), key=lambda i: yes_prices[i])
            target_market = active_markets[best_idx]
            target_price = yes_prices[best_idx]
            action, outcome = "BUY", "YES"
            anomaly_type = "HIGH entropy"
            signal_desc = (
                f"NegRisk event too uncertain near resolution. "
                f"{n_outcomes} outcomes, H_actual={h_actual:.3f} vs "
                f"H_expected={h_expected:.3f} (+{deviation:.3f} bits). "
                f"Buy YES on leading outcome: {target_market.question[:40]}."
            )
        else:
            # Too certain: the leading outcome may be overpriced.
            # Fade by buying YES on the second-most-likely outcome.
            sorted_indices = sorted(
                range(n_outcomes), key=lambda i: yes_prices[i], reverse=True
            )
            # Pick the second-best outcome for contrarian fade.
            # If only 2 outcomes, pick the underdog.
            fade_idx = (
                sorted_indices[1] if len(sorted_indices) > 1 else sorted_indices[0]
            )
            target_market = active_markets[fade_idx]
            target_price = yes_prices[fade_idx]
            action, outcome = "BUY", "YES"
            anomaly_type = "LOW entropy"
            signal_desc = (
                f"NegRisk event overconfident far from resolution. "
                f"{n_outcomes} outcomes, H_actual={h_actual:.3f} vs "
                f"H_expected={h_expected:.3f} ({deviation:.3f} bits). "
                f"Contrarian: buy YES on {target_market.question[:40]}."
            )

        if spike:
            signal_desc += (
                f" Entropy SPIKE detected (prev={prev_h:.3f}, now={h_actual:.3f})."
            )

        # --- Build position ---
        token_id = None
        if target_market.clob_token_ids and len(target_market.clob_token_ids) > 0:
            token_id = target_market.clob_token_ids[0]

        positions = [
            {
                "action": action,
                "outcome": outcome,
                "market": target_market.question[:50],
                "price": target_price,
                "token_id": token_id,
                "entropy_actual": round(h_actual, 4),
                "entropy_expected": round(h_expected, 4),
                "entropy_deviation": round(deviation, 4),
                "entropy_spike": spike,
                "n_outcomes": n_outcomes,
                "total_yes": round(total_yes, 4),
            }
        ]

        total_cost = target_price

        opp = self.create_opportunity(
            title=f"Entropy Signal ({anomaly_type}): {event.title[:45]}...",
            description=signal_desc,
            total_cost=total_cost,
            markets=[target_market],
            positions=positions,
            event=event,
        )

        if opp is not None:
            # Risk is HIGH for multi-outcome entropy signals. This is a
            # directional bet on the favorite (or a contrarian underdog),
            # NOT a structural arbitrage. If the chosen outcome doesn't
            # win, the entire position is lost.
            base_risk = 0.75
            deviation_factor = max(0.0, 1.0 - abs(deviation) / 0.5) * 0.10
            contrarian_penalty = 0.15 if deviation < 0 else 0.0
            spike_bonus = -0.05 if spike else 0.0
            entropy_risk = (
                base_risk + deviation_factor + contrarian_penalty + spike_bonus
            )
            opp.risk_score = min(max(opp.risk_score, entropy_risk), 1.0)
            opp.risk_factors.insert(
                0,
                "DIRECTIONAL BET — not arbitrage. 100% loss if chosen outcome loses.",
            )
            opp.risk_factors.append(
                f"Multi-outcome entropy deviation: {deviation:+.3f} bits "
                f"(actual={h_actual:.3f}, expected={h_expected:.3f}, "
                f"max={max_entropy(n_outcomes):.3f})"
            )
            if spike:
                opp.risk_factors.append(
                    f"Multi-outcome entropy spike: {prev_h:.3f} -> {h_actual:.3f}"
                )
            opp.risk_factors.append("Probabilistic signal (not guaranteed arbitrage)")

        return opp
