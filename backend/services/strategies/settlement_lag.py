"""
Strategy 8: Settlement Lag Arbitrage

Exploits delayed price updates after an outcome has been determined but
prices haven't fully adjusted yet.

From Kroer et al. Part 2 (Section IV, Type 3 mispricing):
- When outcomes settle, Polymarket prices don't instantly lock
- They drift toward 0 or 1 as traders bet against resolved outcomes
- This creates multi-hour arbitrage windows

Example (from paper): Assad remaining President of Syria through 2024:
- Assad flees country (outcome determined)
- Prices: YES = $0.30, NO = $0.30 (sum = 0.60)
- Should be: YES = $0, NO = $1
- Arbitrage window lasted hours

Detection approach:
1. Find markets where resolution conditions are likely met
2. Check if prices haven't fully adjusted (far from 0/1)
3. Look for very low YES prices on events that appear resolved
4. Look for events where external signals suggest settlement
"""

from typing import Optional
from models import Market, Event, ArbitrageOpportunity, StrategyType, MispricingType
from .base import BaseStrategy, utcnow, make_aware
from utils.logger import get_logger

logger = get_logger(__name__)


class SettlementLagStrategy(BaseStrategy):
    """
    Settlement Lag Arbitrage: Exploit delayed price adjustments.

    Detects markets where:
    1. Resolution date has passed but market isn't closed yet
    2. One outcome is trading at very low prices (< $0.05)
       suggesting it's effectively resolved
    3. Sum of prices deviates significantly from $1 near resolution
    4. Recent dramatic price movements suggest outcome is known

    The key insight: Polymarket chose speed over accuracy (Section IV).
    Settlement creates Type 3 mispricing where prices drift rather than
    snap to 0/1 when outcomes resolve.
    """

    strategy_type = StrategyType.SETTLEMENT_LAG
    name = "Settlement Lag"
    description = "Exploit delayed price updates after outcome determination"

    # Thresholds calibrated from research
    NEAR_ZERO_THRESHOLD = 0.05  # Price below this suggests resolved to NO
    NEAR_ONE_THRESHOLD = 0.95  # Price above this suggests resolved to YES
    MIN_SUM_DEVIATION = 0.03  # Minimum deviation from 1.0 to be interesting
    OVERDUE_RESOLUTION_DAYS = 0  # Market past resolution date

    def detect(
        self, events: list[Event], markets: list[Market], prices: dict[str, dict]
    ) -> list[ArbitrageOpportunity]:
        opportunities = []

        for market in markets:
            if market.closed or not market.active:
                continue

            if len(market.outcome_prices) != 2:
                continue

            # Get live prices
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

            opp = self._check_settlement_lag(market, yes_price, no_price)
            if opp:
                opportunities.append(opp)

        # Also check NegRisk events for settlement lag
        for event in events:
            if not event.neg_risk or event.closed:
                continue

            event_opps = self._check_negrisk_settlement(event, prices)
            opportunities.extend(event_opps)

        return opportunities

    def _check_settlement_lag(
        self,
        market: Market,
        yes_price: float,
        no_price: float,
    ) -> Optional[ArbitrageOpportunity]:
        """Check a binary market for settlement lag opportunities."""

        total = yes_price + no_price
        now = utcnow()

        # --- Signal 1: Market is past resolution date ---
        is_overdue = False
        if market.end_date:
            end_aware = make_aware(market.end_date)
            if end_aware <= now:
                is_overdue = True

        # --- Signal 2: One side is near-zero (effectively resolved) ---
        appears_resolved = False
        resolved_side = None
        if yes_price < self.NEAR_ZERO_THRESHOLD:
            appears_resolved = True
            resolved_side = "NO"  # YES is near 0, NO should be 1
        elif no_price < self.NEAR_ZERO_THRESHOLD:
            appears_resolved = True
            resolved_side = "YES"  # NO is near 0, YES should be 1

        # --- Signal 3: Sum significantly below 1.0 ---
        # When settlement lag occurs, both sides can be cheap because
        # traders are still processing the outcome
        sum_deviation = abs(total - 1.0)
        has_sum_deviation = sum_deviation > self.MIN_SUM_DEVIATION

        # Need at least one signal plus profitable deviation
        if not (is_overdue or appears_resolved) or not has_sum_deviation:
            return None

        if total >= 1.0:
            return None  # No arbitrage if sum >= 1

        # Build the opportunity
        signals = []
        if is_overdue:
            signals.append("past resolution date")
        if appears_resolved:
            signals.append(f"appears resolved to {resolved_side}")
        if has_sum_deviation:
            signals.append(f"sum deviation: {sum_deviation:.3f}")

        positions = []
        if market.clob_token_ids and len(market.clob_token_ids) >= 2:
            positions = [
                {
                    "action": "BUY",
                    "outcome": "YES",
                    "market": market.question[:50],
                    "price": yes_price,
                    "token_id": market.clob_token_ids[0],
                },
                {
                    "action": "BUY",
                    "outcome": "NO",
                    "market": market.question[:50],
                    "price": no_price,
                    "token_id": market.clob_token_ids[1],
                },
            ]

        opp = self.create_opportunity(
            title=f"Settlement Lag: {market.question[:60]}",
            description=(
                f"Delayed price adjustment detected. "
                f"YES: ${yes_price:.3f}, NO: ${no_price:.3f}, Sum: ${total:.3f}. "
                f"Signals: {', '.join(signals)}"
            ),
            total_cost=total,
            markets=[market],
            positions=positions,
        )

        if opp:
            opp.mispricing_type = MispricingType.SETTLEMENT_LAG
        return opp

    def _check_negrisk_settlement(
        self, event: Event, prices: dict[str, dict]
    ) -> list[ArbitrageOpportunity]:
        """Check NegRisk events for settlement lag in multi-outcome markets.

        NegRisk events are especially vulnerable to settlement lag because
        they have multiple outcomes that must be coordinated. When one
        outcome is determined, the others don't instantly adjust.
        """
        opportunities = []
        active_markets = [m for m in event.markets if not m.closed and m.active]

        if len(active_markets) < 2:
            return []

        # Get live prices for all outcomes
        total_yes = 0.0
        near_zero_count = 0
        near_one_count = 0

        market_prices = []
        for m in active_markets:
            yes_price = m.yes_price
            if m.clob_token_ids and len(m.clob_token_ids) > 0:
                token = m.clob_token_ids[0]
                if token in prices:
                    yes_price = prices[token].get("mid", yes_price)

            market_prices.append((m, yes_price))
            total_yes += yes_price

            if yes_price < self.NEAR_ZERO_THRESHOLD:
                near_zero_count += 1
            elif yes_price > self.NEAR_ONE_THRESHOLD:
                near_one_count += 1

        # Settlement lag signals for NegRisk:
        # 1. Multiple outcomes near zero (some have resolved)
        # 2. Sum of YES prices far from 1.0
        # 3. One outcome near 1.0 but others haven't adjusted

        sum_deviation = abs(total_yes - 1.0)

        if sum_deviation < self.MIN_SUM_DEVIATION:
            return []  # Prices look correct

        # If total YES < 1.0, buying YES on all outcomes is arbitrage
        if total_yes < 1.0:
            positions = []
            for m, price in market_prices:
                if m.clob_token_ids and len(m.clob_token_ids) > 0:
                    positions.append(
                        {
                            "action": "BUY",
                            "outcome": "YES",
                            "market": m.question[:50],
                            "price": price,
                            "token_id": m.clob_token_ids[0],
                        }
                    )

            signals = [f"sum YES = ${total_yes:.3f}"]
            if near_zero_count > 0:
                signals.append(f"{near_zero_count} outcomes near zero")
            if near_one_count > 0:
                signals.append(f"{near_one_count} outcomes near one")

            opp = self.create_opportunity(
                title=f"Settlement Lag (NegRisk): {event.title[:50]}",
                description=(
                    f"NegRisk settlement lag. "
                    f"{len(active_markets)} outcomes, sum YES: ${total_yes:.3f}. "
                    f"Signals: {', '.join(signals)}"
                ),
                total_cost=total_yes,
                markets=active_markets,
                positions=positions,
            )

            if opp:
                opp.mispricing_type = MispricingType.SETTLEMENT_LAG
                opportunities.append(opp)

        return opportunities
