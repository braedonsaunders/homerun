"""
Strategy: Statistical Arbitrage / Information Edge

Compares Polymarket prices against external probability signals
to find markets where the crowd is wrong.

External signals used:
1. Implied probability from price (YES price = market's probability)
2. Multi-market consensus: average probability across related markets
3. Category base rates: historical resolution rates by category
4. Price momentum: trending markets tend to continue trending
5. Anchoring detection: markets stuck near round numbers (50%, 25%, 75%)

The key insight from the $2.2M bot: ensemble multiple weak signals
into a composite "fair probability" and trade the deviation.

NOT risk-free. This is informed speculation with statistical edge.
"""

import statistics
from typing import Optional

from models import Market, Event, ArbitrageOpportunity, StrategyType
from config import settings
from .base import BaseStrategy


# Round numbers where human anchoring bias is common
ANCHOR_PRICES = [0.10, 0.25, 0.33, 0.50, 0.67, 0.75, 0.90]

# Anchoring detection tolerance: if price is within this of a round number,
# it is considered anchored
ANCHOR_TOLERANCE = 0.02

# Category base rates: historical resolution rates by category.
# These are hard-coded initial estimates of how often YES resolves in
# markets of each type.
CATEGORY_BASE_RATES: dict[str, float] = {
    # Crypto price targets: most ambitious targets don't hit
    "crypto": 0.35,
    "cryptocurrency": 0.35,
    "bitcoin": 0.35,
    "ethereum": 0.35,
    # Political events: roughly balanced but slightly below 50%
    "politics": 0.45,
    "political": 0.45,
    "elections": 0.45,
    "government": 0.45,
    # Sports: balanced by design (bookmakers are efficient)
    "sports": 0.50,
    "nfl": 0.50,
    "nba": 0.50,
    "mlb": 0.50,
    "soccer": 0.50,
    "football": 0.50,
    # Science/tech: most ambitious predictions fail
    "science": 0.25,
    "technology": 0.25,
    "tech": 0.25,
    "ai": 0.30,
    # Entertainment: moderate resolution rate
    "entertainment": 0.40,
    "culture": 0.40,
    "pop culture": 0.40,
    # Finance / economics
    "finance": 0.40,
    "economics": 0.40,
    # Default for unknown categories
    "_default": 0.45,
}

# Signal weights for composite fair probability
SIGNAL_WEIGHTS = {
    "anchoring": 0.15,
    "category_base_rate": 0.25,
    "consensus": 0.25,
    "momentum": 0.15,
    "volume_price": 0.20,
}


class StatArbStrategy(BaseStrategy):
    """
    Strategy: Statistical Arbitrage / Information Edge

    Trade deviations between estimated fair probability and market price.
    Ensembles multiple weak signals into a composite probability estimate,
    then trades when the deviation exceeds a configurable threshold.

    This is NOT risk-free arbitrage. It is informed directional speculation
    with a statistical edge derived from combining independent signals.
    """

    strategy_type = StrategyType.STAT_ARB
    name = "Statistical Arbitrage"
    description = "Trade deviations from estimated fair probability"

    def __init__(self):
        super().__init__()
        self.min_edge: float = settings.STAT_ARB_MIN_EDGE
        # Track previous prices for momentum signal across scans
        self._prev_prices: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Signal 1: Anchoring Detection
    # ------------------------------------------------------------------
    def _signal_anchoring(self, yes_price: float) -> float:
        """Detect whether a market price is anchored to a round number.

        Markets at exactly 50%, 25%, 75% etc. are often anchored because
        humans round to convenient numbers.  When a market sits at one of
        these levels for a long time with volume it is probably anchored
        rather than reflecting true probability.

        Returns 1.0 if anchored (price within ANCHOR_TOLERANCE of a round
        number), 0.0 otherwise.
        """
        for anchor in ANCHOR_PRICES:
            if abs(yes_price - anchor) < ANCHOR_TOLERANCE:
                return 1.0
        return 0.0

    # ------------------------------------------------------------------
    # Signal 2: Category Base Rate
    # ------------------------------------------------------------------
    def _signal_category_base_rate(
        self, yes_price: float, category: Optional[str]
    ) -> float:
        """Compare market price to category base rate.

        If the market price is significantly above the category's
        historical resolution rate the YES side may be overpriced (and
        vice versa).

        Returns a value in [-1, 1]:
          positive => market is under-priced (YES should be higher)
          negative => market is over-priced  (YES should be lower)
        """
        base_rate = CATEGORY_BASE_RATES.get("_default")
        if category:
            cat_lower = category.lower().strip()
            base_rate = CATEGORY_BASE_RATES.get(cat_lower, base_rate)
        deviation = base_rate - yes_price  # type: ignore[operator]
        # Clamp to [-1, 1]
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
        Deviation (MAD).

        For markets in the same event, check consistency.  If event has 5
        markets and 4 imply X should be ~60% but market 5 prices X at 40%,
        market 5 is the outlier.

        Returns a value in [-1, 1] representing deviation from consensus,
        or None if consensus cannot be determined (fewer than 3 sibling
        markets).
        """
        if len(event_markets) < 3:
            return None

        # Gather YES prices for all sibling binary markets
        sibling_prices: list[float] = []
        target_price: Optional[float] = None
        for m in event_markets:
            if len(m.outcome_prices) != 2:
                continue
            if m.closed or not m.active:
                continue
            p = self._live_yes_price(m, prices)
            sibling_prices.append(p)
            if m.id == market.id:
                target_price = p

        if target_price is None or len(sibling_prices) < 3:
            return None

        median = statistics.median(sibling_prices)
        mad = statistics.median([abs(p - median) for p in sibling_prices])
        if mad < 0.01:
            # All prices are essentially identical; no outlier signal
            return 0.0

        # Modified Z-score: how many MADs away is the target?
        z = (target_price - median) / mad
        # Positive z means target is above consensus (possibly overpriced)
        # Negative z means target is below consensus (possibly underpriced)
        # Normalize to [-1, 1] by capping at 3 MADs
        signal = max(-1.0, min(1.0, -z / 3.0))
        return signal

    # ------------------------------------------------------------------
    # Signal 4: Momentum
    # ------------------------------------------------------------------
    def _signal_momentum(self, market_id: str, current_price: float) -> float:
        """Track price changes across scans to detect momentum.

        Markets trending up tend to continue (momentum factor).

        Returns a value in roughly [-1, 1].  Positive means upward momentum
        (price has been rising), negative means downward.
        """
        prev = self._prev_prices.get(market_id)
        # Always store the latest price for next scan
        self._prev_prices[market_id] = current_price

        if prev is None or prev == 0:
            return 0.0

        momentum = (current_price - prev) / prev
        # Cap momentum signal
        return max(-1.0, min(1.0, momentum * 10))  # scale up small moves

    # ------------------------------------------------------------------
    # Signal 5: Volume-Price Divergence
    # ------------------------------------------------------------------
    def _signal_volume_price(
        self, yes_price: float, volume: float, liquidity: float
    ) -> float:
        """Detect volume-price divergence.

        High volume with small price change away from 0.50 = informed
        disagreement, which could signal an upcoming move.
        Low volume with price at an extreme = stale/illiquid (risky, but
        may present an opportunity if the market hasn't adjusted).

        Returns a value in [-1, 1].
          Positive => suggests upward pressure (volume high, price low)
          Negative => suggests downward pressure (volume high, price high)
          Near zero => no clear signal
        """
        if volume <= 0 or liquidity <= 0:
            return 0.0

        # Volume-to-liquidity ratio as a proxy for trading activity
        vol_liq = volume / liquidity if liquidity > 0 else 0.0

        # Price distance from 0.50 (how extreme the price is)
        price_extremity = abs(yes_price - 0.50)

        if vol_liq > 5.0 and price_extremity < 0.15:
            # High volume but price near 50%: strong disagreement
            # Slight positive bias -- active markets tend to be efficient
            return 0.0

        if vol_liq > 5.0 and price_extremity > 0.25:
            # High volume and extreme price: strong conviction
            # The crowd is probably right when volume is high
            return 0.0

        if vol_liq < 1.0 and price_extremity > 0.30:
            # Low volume, extreme price: might be stale / illiquid
            # Signal that price should revert toward 0.50
            direction = -1.0 if yes_price > 0.50 else 1.0
            return direction * 0.5

        if vol_liq < 0.5 and price_extremity > 0.40:
            # Very low volume, very extreme price: likely stale
            direction = -1.0 if yes_price > 0.50 else 1.0
            return direction * 0.8

        return 0.0

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

    def _get_category(self, market: Market, events: list[Event]) -> Optional[str]:
        """Look up the category for a market via its parent event."""
        for event in events:
            for em in event.markets:
                if em.id == market.id:
                    return event.category
        return None

    def _get_event_for_market(
        self, market: Market, events: list[Event]
    ) -> Optional[Event]:
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

        # Signal 1: Anchoring
        anchor = self._signal_anchoring(yes_price)
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
        vol_price = self._signal_volume_price(
            yes_price, market.volume, market.liquidity
        )
        signals["volume_price"] = vol_price

        # Weighted composite adjustment
        # Each signal contributes a directional shift from the market price.
        # Positive signal => fair prob higher than market (buy YES),
        # Negative signal => fair prob lower than market (buy NO).
        adjustment = 0.0
        total_weight = 0.0
        for sig_name, sig_value in signals.items():
            w = SIGNAL_WEIGHTS.get(sig_name, 0.0)
            adjustment += w * sig_value
            total_weight += w

        if total_weight > 0:
            adjustment /= total_weight
            # Scale the adjustment: full signal moves price by up to 0.15
            adjustment *= 0.15

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
    ) -> list[ArbitrageOpportunity]:
        """Detect statistical arbitrage opportunities.

        For each binary market, calculate all signals, compute composite
        fair probability, and if the edge exceeds STAT_ARB_MIN_EDGE
        create an opportunity.
        """
        if not settings.STAT_ARB_ENABLED:
            return []

        opportunities: list[ArbitrageOpportunity] = []

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

            # Get live prices
            yes_price = self._live_yes_price(market, prices)
            no_price = self._live_no_price(market, prices)

            # Skip extreme prices (nearly resolved, nothing to trade)
            if yes_price < 0.03 or yes_price > 0.97:
                continue

            # Look up parent event and category
            event = market_to_event.get(market.id)
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
            if abs(edge) < self.min_edge:
                continue

            # Determine direction
            if edge > 0:
                # Fair prob > market price => YES is underpriced, buy YES
                outcome = "YES"
                buy_price = yes_price
                token_id = (
                    market.clob_token_ids[0]
                    if market.clob_token_ids and len(market.clob_token_ids) > 0
                    else None
                )
            else:
                # Fair prob < market price => YES is overpriced, buy NO
                outcome = "NO"
                buy_price = no_price
                token_id = (
                    market.clob_token_ids[1]
                    if market.clob_token_ids and len(market.clob_token_ids) > 1
                    else None
                )

            # Profit metrics
            # expected_payout is 1.0 if our prediction is correct
            total_cost = buy_price
            expected_payout = 1.0
            gross_profit = expected_payout - total_cost
            fee = expected_payout * self.fee
            net_profit = gross_profit - fee
            roi = (net_profit / total_cost) * 100 if total_cost > 0 else 0

            # Skip if negative expected profit after fees
            if net_profit <= 0:
                continue

            # Risk assessment: stat arb is uncertain by nature
            # Risk score between 0.40 and 0.60 depending on edge strength
            # Stronger edge = slightly lower risk
            edge_strength = min(abs(edge) / 0.20, 1.0)  # Normalize to [0, 1]
            risk_score = 0.60 - (edge_strength * 0.20)  # Range: 0.40 - 0.60
            risk_score = max(0.40, min(0.60, risk_score))

            risk_factors = [
                "Statistical edge, not guaranteed profit",
                f"Composite edge: {abs(edge):.1%} ({outcome} side)",
                f"Fair probability estimate: {fair_prob:.1%} vs market {yes_price:.1%}",
            ]

            # Add signal-specific risk factors
            if signal_breakdown.get("anchoring", 0) > 0:
                risk_factors.append("Anchoring bias detected at round number")
            if abs(signal_breakdown.get("category_base_rate", 0)) > 0.10:
                risk_factors.append(
                    f"Category base rate deviation: {signal_breakdown['category_base_rate']:+.2f}"
                )
            if abs(signal_breakdown.get("consensus", 0)) > 0.3:
                risk_factors.append("Price is an outlier within its event")
            if abs(signal_breakdown.get("momentum", 0)) > 0.3:
                direction_word = (
                    "upward" if signal_breakdown["momentum"] > 0 else "downward"
                )
                risk_factors.append(f"Strong {direction_word} momentum detected")
            if abs(signal_breakdown.get("volume_price", 0)) > 0.3:
                risk_factors.append("Volume-price divergence detected")

            # Liquidity
            min_liquidity = market.liquidity
            max_position = min_liquidity * 0.05  # Conservative: 5% of liquidity

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

            opp = ArbitrageOpportunity(
                strategy=self.strategy_type,
                title=f"Stat Arb: {market.question[:60]}",
                description=(
                    f"Buy {outcome} @ ${buy_price:.3f} | "
                    f"Fair prob {fair_prob:.1%} vs market {yes_price:.1%} | "
                    f"Edge {abs(edge):.1%}"
                ),
                total_cost=total_cost,
                expected_payout=expected_payout,
                gross_profit=gross_profit,
                fee=fee,
                net_profit=net_profit,
                roi_percent=roi,
                risk_score=risk_score,
                risk_factors=risk_factors,
                markets=[
                    {
                        "id": market.id,
                        "slug": market.slug,
                        "question": market.question,
                        "yes_price": yes_price,
                        "no_price": no_price,
                        "liquidity": market.liquidity,
                    }
                ],
                event_id=event.id if event else None,
                event_slug=event.slug if event else None,
                event_title=event.title if event else None,
                category=category,
                min_liquidity=min_liquidity,
                max_position_size=max_position,
                resolution_date=market.end_date,
                positions_to_take=positions,
            )

            opportunities.append(opp)

        # Sort by absolute edge (strongest signals first)
        opportunities.sort(
            key=lambda o: abs(
                o.positions_to_take[0].get("edge", 0) if o.positions_to_take else 0
            ),
            reverse=True,
        )

        return opportunities
