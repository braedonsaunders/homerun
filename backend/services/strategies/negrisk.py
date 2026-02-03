from models import Market, Event, ArbitrageOpportunity, StrategyType
from .base import BaseStrategy


class NegRiskStrategy(BaseStrategy):
    """
    Strategy 4: NegRisk / One-of-Many Arbitrage

    For events with multiple related markets (like date-based outcomes),
    buy NO on all outcomes when total cost < $1.00

    This is the "$1M in 7 days" strategy used by anoin123.

    Example (US Strikes Iran):
    - "by March" NO: $0.89
    - "by April" NO: $0.03
    - "by June" NO: $0.02
    - Total cost: $0.94
    - If no strike by June: All NO positions pay = $1.00
    - Profit: $0.06

    Also works for multi-outcome markets where all YES prices sum < $1:
    - Candidate A YES: $0.30
    - Candidate B YES: $0.35
    - Candidate C YES: $0.32
    - Total: $0.97, one must win = $1.00
    """

    strategy_type = StrategyType.NEGRISK
    name = "NegRisk / One-of-Many"
    description = "Buy all NO positions on related date markets, or all YES on multi-outcome"

    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict]
    ) -> list[ArbitrageOpportunity]:
        opportunities = []

        for event in events:
            # Need multiple markets in the event
            if len(event.markets) < 2:
                continue

            # Skip closed events
            if event.closed:
                continue

            # Strategy A: NegRisk events (flagged by Polymarket)
            if event.neg_risk:
                opp = self._detect_negrisk_event(event, prices)
                if opp:
                    opportunities.append(opp)

            # Strategy B: Date-based markets (buy NO on all dates)
            opp = self._detect_date_sweep(event, prices)
            if opp:
                opportunities.append(opp)

            # Strategy C: Multi-outcome (buy YES on all outcomes)
            opp = self._detect_multi_outcome(event, prices)
            if opp:
                opportunities.append(opp)

        return opportunities

    def _detect_negrisk_event(
        self,
        event: Event,
        prices: dict[str, dict]
    ) -> ArbitrageOpportunity | None:
        """Detect arbitrage in official NegRisk events"""
        active_markets = [m for m in event.markets if m.active and not m.closed]
        if len(active_markets) < 2:
            return None

        # Get YES prices for all outcomes
        total_yes = 0.0
        positions = []

        for market in active_markets:
            yes_price = market.yes_price

            # Use live price if available
            if market.clob_token_ids:
                yes_token = market.clob_token_ids[0]
                if yes_token in prices:
                    yes_price = prices[yes_token].get("mid", yes_price)

            total_yes += yes_price
            positions.append({
                "action": "BUY",
                "outcome": "YES",
                "market": market.question[:50],
                "price": yes_price,
                "token_id": market.clob_token_ids[0] if market.clob_token_ids else None
            })

        # In NegRisk, exactly one outcome wins, so total YES should = $1
        if total_yes >= 1.0:
            return None

        return self.create_opportunity(
            title=f"NegRisk: {event.title[:50]}...",
            description=f"Buy YES on all {len(active_markets)} outcomes for ${total_yes:.3f}, one wins = $1",
            total_cost=total_yes,
            markets=active_markets,
            positions=positions,
            event=event
        )

    def _detect_date_sweep(
        self,
        event: Event,
        prices: dict[str, dict]
    ) -> ArbitrageOpportunity | None:
        """
        Detect date-based arbitrage: same event with different date cutoffs
        Buy NO on all dates - if nothing happens, all pay out
        """
        # Look for markets with date patterns in questions
        date_keywords = ["by", "before", "in", "during", "jan", "feb", "mar", "apr",
                         "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
                         "january", "february", "march", "april", "june", "july",
                         "august", "september", "october", "november", "december",
                         "q1", "q2", "q3", "q4", "2025", "2026", "2027"]

        active_markets = [m for m in event.markets if m.active and not m.closed]

        # Check if this looks like a date-based event
        date_markets = []
        for market in active_markets:
            question_lower = market.question.lower()
            if any(kw in question_lower for kw in date_keywords):
                date_markets.append(market)

        if len(date_markets) < 2:
            return None

        # Calculate total NO cost
        total_no = 0.0
        positions = []

        for market in date_markets:
            no_price = market.no_price

            # Use live price if available
            if len(market.clob_token_ids) > 1:
                no_token = market.clob_token_ids[1]
                if no_token in prices:
                    no_price = prices[no_token].get("mid", no_price)

            total_no += no_price
            positions.append({
                "action": "BUY",
                "outcome": "NO",
                "market": market.question[:50],
                "price": no_price,
                "token_id": market.clob_token_ids[1] if len(market.clob_token_ids) > 1 else None
            })

        # If event never happens, all NO positions pay $1 each
        # But we only get $1 total (the latest date NO pays)
        # Actually for date sweeps, if nothing happens ALL NOs pay
        # So expected payout = number of markets (but risk if event happens early)

        # Conservative approach: assume only 1 NO pays (the "nothing happens" scenario)
        # This is the safest interpretation
        if total_no >= 1.0:
            return None

        return self.create_opportunity(
            title=f"Date Sweep: {event.title[:40]}...",
            description=f"Buy NO on {len(date_markets)} date markets for ${total_no:.3f}. If nothing happens = $1 payout",
            total_cost=total_no,
            markets=date_markets,
            positions=positions,
            event=event
        )

    def _detect_multi_outcome(
        self,
        event: Event,
        prices: dict[str, dict]
    ) -> ArbitrageOpportunity | None:
        """
        Detect multi-outcome arbitrage: exhaustive outcomes where one must win
        Buy YES on all outcomes when total < $1
        """
        # Skip if already handled as NegRisk
        if event.neg_risk:
            return None

        active_markets = [m for m in event.markets if m.active and not m.closed]
        if len(active_markets) < 3:  # Need at least 3 outcomes
            return None

        # Calculate total YES cost
        total_yes = 0.0
        positions = []

        for market in active_markets:
            yes_price = market.yes_price

            if market.clob_token_ids:
                yes_token = market.clob_token_ids[0]
                if yes_token in prices:
                    yes_price = prices[yes_token].get("mid", yes_price)

            total_yes += yes_price
            positions.append({
                "action": "BUY",
                "outcome": "YES",
                "market": market.question[:50],
                "price": yes_price,
                "token_id": market.clob_token_ids[0] if market.clob_token_ids else None
            })

        if total_yes >= 1.0:
            return None

        return self.create_opportunity(
            title=f"Multi-Outcome: {event.title[:40]}...",
            description=f"Buy YES on all {len(active_markets)} outcomes for ${total_yes:.3f}, one wins = $1",
            total_cost=total_yes,
            markets=active_markets,
            positions=positions,
            event=event
        )
