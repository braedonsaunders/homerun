from __future__ import annotations

from models import Market, Event, ArbitrageOpportunity, StrategyType
from config import settings
from .base import BaseStrategy


class MustHappenStrategy(BaseStrategy):
    """
    Strategy 5: Must-Happen Arbitrage

    Buy YES on ALL possible outcomes when total < $1.00
    One outcome MUST happen, guaranteeing a $1 payout.

    WARNING: This strategy has SIGNIFICANT RISKS!

    The strategy uses keyword heuristics ("winner", "who will", etc.) to guess
    that outcomes are exhaustive, but CANNOT VERIFY this is actually true.

    KNOWN ISSUES:
    1. Hidden candidates: "Who wins the election?" may show 3 candidates but
       there could be others not displayed in the markets
    2. "None of the above": Some events allow outcomes not listed
    3. Cancellation/postponement: Events can be cancelled, voiding all bets
    4. Rule changes: Resolution criteria may change

    ONLY trust this strategy when:
    - The event is explicitly flagged as NegRisk by Polymarket (use NegRisk strategy instead)
    - You have MANUALLY verified the outcomes cover ALL possibilities
    - The event rules explicitly state one outcome must win

    Example of FAILURE:
    - "Who wins the 2024 primary?" with markets for A, B, C
    - Candidate D enters the race and wins
    - All your YES positions resolve to $0

    Example (Multi-candidate election):
    - Candidate A YES: $0.30
    - Candidate B YES: $0.35
    - Candidate C YES: $0.32
    - Total: $0.97
    - ASSUMED one must win = $1.00
    - Profit: $0.03 (IF assumptions hold)
    """

    strategy_type = StrategyType.MUST_HAPPEN
    name = "Must-Happen"
    description = (
        "Buy YES on all outcomes - REQUIRES MANUAL VERIFICATION of exhaustiveness"
    )

    # Keywords indicating POTENTIALLY exhaustive outcome sets
    # WARNING: These are HEURISTICS, not guarantees!
    EXHAUSTIVE_KEYWORDS = [
        "winner",
        "who will",
        "which",
        "what will",
        "champion",
        "elected",
        "nominee",
        "president",
        "first",
        "next",
        "wins",
    ]

    def detect(
        self, events: list[Event], markets: list[Market], prices: dict[str, dict]
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
        self, event: Event, prices: dict[str, dict]
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

            positions.append(
                {
                    "action": "BUY",
                    "outcome": "YES",
                    "market": market.question[:50],
                    "price": yes_price,
                    "token_id": market.clob_token_ids[0]
                    if market.clob_token_ids
                    else None,
                }
            )

        # Need to be under $1 for profit
        if total_yes >= 1.0:
            return None

        # Reject if total YES is too low — non-exhaustive outcome list
        if total_yes < settings.NEGRISK_MIN_TOTAL_YES:
            return None

        opp = self.create_opportunity(
            title=f"⚠️ Must-Happen: {event.title[:40]}...",
            description=f"VERIFY MANUALLY: Ensure all outcomes are listed. Buy all {len(active_markets)} YES for ${total_yes:.3f}",
            total_cost=total_yes,
            markets=active_markets,
            positions=positions,
            event=event,
        )

        # Add extra risk factor for must-happen strategy
        if opp:
            opp.risk_factors.insert(
                0, "⚠️ REQUIRES MANUAL VERIFICATION - may have hidden outcomes"
            )
            if total_yes < 0.90:
                opp.risk_factors.insert(
                    1, f"Low total ({total_yes:.0%}) suggests possible missing outcomes"
                )

        return opp
