from models import Market, Event, ArbitrageOpportunity, StrategyType
from .base import BaseStrategy


class MutuallyExclusiveStrategy(BaseStrategy):
    """
    Strategy 2: Mutually Exclusive Arbitrage

    Find two events where only one can be true, buy YES on both for < $1

    Example:
    - "Democrats win 2028" YES: $0.45
    - "Republicans win 2028" YES: $0.52
    - Total: $0.97
    - One MUST win (mutually exclusive) = $1.00 payout
    - Profit: $0.03
    """

    strategy_type = StrategyType.MUTUALLY_EXCLUSIVE
    name = "Mutually Exclusive"
    description = "Two events where only one can be true, buy YES on both"

    # Pairs of mutually exclusive patterns to look for
    EXCLUSIVE_PATTERNS = [
        # Political
        (["democrat", "biden", "harris", "democratic"], ["republican", "trump", "gop"]),
        (["yes", "will"], ["no", "won't", "will not"]),
        # Sports
        (["home", "team a"], ["away", "team b"]),
        # Binary outcomes with different framing
        (["above", "over", "more than", "higher"], ["below", "under", "less than", "lower"]),
        (["before", "by"], ["after", "not by"]),
        (["win", "victory"], ["lose", "defeat"]),
    ]

    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict]
    ) -> list[ArbitrageOpportunity]:
        opportunities = []

        # Group markets by potential mutual exclusivity
        # This is a simplified approach - real implementation would need NLP

        # Look within events first (related markets)
        for event in events:
            if len(event.markets) < 2:
                continue

            opps = self._find_exclusive_pairs_in_event(event, prices)
            opportunities.extend(opps)

        # Also check across all markets for obvious pairs
        opps = self._find_exclusive_pairs_across_markets(markets, prices)
        opportunities.extend(opps)

        return opportunities

    def _find_exclusive_pairs_in_event(
        self,
        event: Event,
        prices: dict[str, dict]
    ) -> list[ArbitrageOpportunity]:
        """Find mutually exclusive pairs within an event"""
        opportunities = []
        active_markets = [m for m in event.markets if m.active and not m.closed]

        # IMPORTANT: Only apply to events with EXACTLY 2 markets
        # If there are more than 2, it's a multi-outcome scenario (use must_happen/negrisk instead)
        # Two random candidates from a multi-candidate race are NOT mutually exclusive
        # because a third candidate could win
        if len(active_markets) != 2:
            return opportunities

        market_a, market_b = active_markets
        if self._are_mutually_exclusive(market_a, market_b):
            opp = self._check_pair(market_a, market_b, prices, event)
            if opp:
                opportunities.append(opp)

        return opportunities

    def _find_exclusive_pairs_across_markets(
        self,
        markets: list[Market],
        prices: dict[str, dict]
    ) -> list[ArbitrageOpportunity]:
        """Find mutually exclusive pairs across all markets"""
        opportunities = []

        # This is expensive, so we limit to high-volume markets
        high_volume = sorted(markets, key=lambda m: m.volume, reverse=True)[:100]

        for i, market_a in enumerate(high_volume):
            for market_b in high_volume[i+1:]:
                if self._are_mutually_exclusive(market_a, market_b):
                    opp = self._check_pair(market_a, market_b, prices)
                    if opp:
                        opportunities.append(opp)

        return opportunities

    def _are_mutually_exclusive(self, market_a: Market, market_b: Market) -> bool:
        """Check if two markets are mutually exclusive based on patterns"""
        q_a = market_a.question.lower()
        q_b = market_b.question.lower()

        for pattern_a, pattern_b in self.EXCLUSIVE_PATTERNS:
            a_matches_first = any(p in q_a for p in pattern_a)
            b_matches_second = any(p in q_b for p in pattern_b)
            a_matches_second = any(p in q_a for p in pattern_b)
            b_matches_first = any(p in q_b for p in pattern_a)

            # Check if they're opposite patterns
            if (a_matches_first and b_matches_second) or (a_matches_second and b_matches_first):
                # Additional check: questions should be about the same topic
                # Simple heuristic: share significant words
                words_a = set(q_a.split())
                words_b = set(q_b.split())
                common = words_a & words_b
                # Remove common stop words
                stop_words = {"will", "the", "a", "an", "in", "on", "by", "to", "be", "is", "are"}
                common = common - stop_words

                if len(common) >= 2:
                    return True

        return False

    def _check_pair(
        self,
        market_a: Market,
        market_b: Market,
        prices: dict[str, dict],
        event: Event = None
    ) -> ArbitrageOpportunity | None:
        """Check if a pair offers arbitrage opportunity"""
        # Get YES prices
        yes_a = market_a.yes_price
        yes_b = market_b.yes_price

        # Use live prices if available
        if market_a.clob_token_ids:
            token = market_a.clob_token_ids[0]
            if token in prices:
                yes_a = prices[token].get("mid", yes_a)

        if market_b.clob_token_ids:
            token = market_b.clob_token_ids[0]
            if token in prices:
                yes_b = prices[token].get("mid", yes_b)

        total_cost = yes_a + yes_b

        # If total is way below 1.0, these probably aren't exhaustive options
        # (there might be other candidates/outcomes)
        # Only consider if total is at least 0.85 (suggesting these are the main options)
        if total_cost < 0.85:
            return None

        if total_cost >= 1.0:
            return None

        positions = [
            {
                "action": "BUY",
                "outcome": "YES",
                "market": market_a.question[:50],
                "price": yes_a,
                "token_id": market_a.clob_token_ids[0] if market_a.clob_token_ids else None
            },
            {
                "action": "BUY",
                "outcome": "YES",
                "market": market_b.question[:50],
                "price": yes_b,
                "token_id": market_b.clob_token_ids[0] if market_b.clob_token_ids else None
            }
        ]

        return self.create_opportunity(
            title=f"Exclusive: {market_a.question[:25]}... vs {market_b.question[:25]}...",
            description=f"Mutually exclusive markets. YES on both: ${yes_a:.3f} + ${yes_b:.3f} = ${total_cost:.3f}",
            total_cost=total_cost,
            markets=[market_a, market_b],
            positions=positions,
            event=event
        )
