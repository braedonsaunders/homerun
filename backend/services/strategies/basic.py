from models import Market, Event, ArbitrageOpportunity, StrategyType
from .base import BaseStrategy


class BasicArbStrategy(BaseStrategy):
    """
    Strategy 1: Basic Arbitrage

    Buy YES + NO on the same binary market when total cost < $1.00
    Guaranteed profit since one of them must win.

    Example:
    - YES price: $0.48
    - NO price: $0.48
    - Total cost: $0.96
    - Payout: $1.00
    - Profit: $0.04 (before fees)
    """

    strategy_type = StrategyType.BASIC
    name = "Basic Arbitrage"
    description = "Buy YES and NO on same market when total < $1"

    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict]
    ) -> list[ArbitrageOpportunity]:
        opportunities = []

        for market in markets:
            # Skip if not a binary market
            if len(market.outcome_prices) != 2:
                continue

            # Skip inactive or closed markets
            if market.closed or not market.active:
                continue

            # Get prices (use live prices if available)
            yes_price = market.yes_price
            no_price = market.no_price

            # Update with live prices if we have them
            if market.clob_token_ids:
                yes_token = market.clob_token_ids[0] if len(market.clob_token_ids) > 0 else None
                no_token = market.clob_token_ids[1] if len(market.clob_token_ids) > 1 else None

                if yes_token and yes_token in prices:
                    yes_price = prices[yes_token].get("mid", yes_price)
                if no_token and no_token in prices:
                    no_price = prices[no_token].get("mid", no_price)

            # Calculate total cost
            total_cost = yes_price + no_price

            # Check for arbitrage (need room for fees)
            # Total must be less than (1 - fee) to profit
            if total_cost >= 1.0:
                continue

            # Create opportunity
            positions = [
                {
                    "action": "BUY",
                    "outcome": "YES",
                    "price": yes_price,
                    "token_id": market.clob_token_ids[0] if market.clob_token_ids else None
                },
                {
                    "action": "BUY",
                    "outcome": "NO",
                    "price": no_price,
                    "token_id": market.clob_token_ids[1] if len(market.clob_token_ids) > 1 else None
                }
            ]

            opp = self.create_opportunity(
                title=f"Basic Arb: {market.question[:50]}...",
                description=f"Buy YES (${yes_price:.3f}) + NO (${no_price:.3f}) = ${total_cost:.3f} for guaranteed $1 payout",
                total_cost=total_cost,
                markets=[market],
                positions=positions
            )

            if opp:
                opportunities.append(opp)

        return opportunities
