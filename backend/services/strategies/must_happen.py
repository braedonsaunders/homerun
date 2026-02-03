from models import Market, Event, ArbitrageOpportunity, StrategyType
from .base import BaseStrategy


class MustHappenStrategy(BaseStrategy):
    """
    Strategy 5: Must-Happen Arbitrage

    Buy YES on ALL possible outcomes when total < $1.00
    One outcome MUST happen, guaranteeing a $1 payout.

    Example (Multi-candidate election):
    - Candidate A YES: $0.30
    - Candidate B YES: $0.35
    - Candidate C YES: $0.32
    - Total: $0.97
    - One MUST win = $1.00
    - Profit: $0.03

    This is similar to NegRisk but focuses on events where
    the outcomes are explicitly exhaustive (one must happen).
    """

    strategy_type = StrategyType.MUST_HAPPEN
    name = "Must-Happen"
    description = "Buy YES on all outcomes when total < $1, one must win"

    # Keywords indicating exhaustive outcome sets
    EXHAUSTIVE_KEYWORDS = [
        "winner", "who will", "which", "what will",
        "champion", "elected", "nominee", "president",
        "first", "next", "wins"
    ]

    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict]
    ) -> list[ArbitrageOpportunity]:
        opportunities = []

        for event in events:
            # Need multiple outcomes
            if len(event.markets) < 2:
                continue

            # Skip already handled NegRisk events (handled by NegRisk strategy)
            if event.neg_risk:
                continue

            # Skip closed events
            if event.closed:
                continue

            # Check if this looks like an exhaustive outcome event
            if not self._is_exhaustive_event(event):
                continue

            opp = self._detect_must_happen(event, prices)
            if opp:
                opportunities.append(opp)

        return opportunities

    def _is_exhaustive_event(self, event: Event) -> bool:
        """
        Check if an event has exhaustive outcomes (one must happen).

        Heuristics:
        1. Event title contains keywords suggesting exhaustive options
        2. Markets represent different choices for the same question
        """
        title_lower = event.title.lower()

        # Check for exhaustive keywords
        if any(kw in title_lower for kw in self.EXHAUSTIVE_KEYWORDS):
            return True

        # Check if markets look like choices (A, B, C pattern)
        questions = [m.question.lower() for m in event.markets]

        # Look for patterns like "Candidate X wins" across markets
        base_pattern = None
        for q in questions:
            # Simple heuristic: if questions differ by just one word/name
            words = set(q.split())
            if base_pattern is None:
                base_pattern = words
            else:
                # Check similarity
                overlap = len(base_pattern & words) / max(len(base_pattern), len(words))
                if overlap < 0.5:
                    return False

        # If we got here with 3+ markets, likely exhaustive
        return len(event.markets) >= 3

    def _detect_must_happen(
        self,
        event: Event,
        prices: dict[str, dict]
    ) -> ArbitrageOpportunity | None:
        """Detect must-happen arbitrage opportunity"""
        active_markets = [m for m in event.markets if m.active and not m.closed]

        if len(active_markets) < 2:
            return None

        # Calculate total YES cost
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

        # Need to be under $1 for profit
        if total_yes >= 1.0:
            return None

        return self.create_opportunity(
            title=f"Must-Happen: {event.title[:45]}...",
            description=f"Exhaustive outcomes. Buy all {len(active_markets)} YES for ${total_yes:.3f}, winner pays $1",
            total_cost=total_yes,
            markets=active_markets,
            positions=positions,
            event=event
        )
