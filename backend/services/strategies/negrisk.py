from __future__ import annotations

from models import Market, Event, ArbitrageOpportunity, StrategyType
from .base import BaseStrategy


class NegRiskStrategy(BaseStrategy):
    """
    Strategy 4: NegRisk / One-of-Many Arbitrage

    For Polymarket-flagged NegRisk events where EXACTLY ONE outcome must win,
    buy YES on all outcomes when total cost < $1.00

    TRUE ARBITRAGE requires mutually exclusive, exhaustive outcomes:
    - Polymarket NegRisk flag: Platform guarantees exactly one outcome wins
    - Multi-outcome elections: "Who wins?" where one candidate must win

    WARNING: Date-based "by X" markets are NOT valid for this strategy!
    - "Event by March", "Event by June", "Event by December" are CUMULATIVE
    - If event happens in March, ALL "by later date" markets also resolve YES
    - Buying NO on all dates = 100% correlated loss if event happens early
    - This is SPECULATIVE, not arbitrage

    Valid Example (Multi-candidate election):
    - Candidate A YES: $0.30
    - Candidate B YES: $0.35
    - Candidate C YES: $0.32
    - Total: $0.97, one must win = $1.00
    - Profit: $0.03 (guaranteed)
    """

    strategy_type = StrategyType.NEGRISK
    name = "NegRisk / One-of-Many"
    description = "Buy YES on all outcomes in verified mutually-exclusive events"

    def detect(
        self, events: list[Event], markets: list[Market], prices: dict[str, dict]
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
            # This is TRUE arbitrage - Polymarket guarantees exactly one outcome wins
            if event.neg_risk:
                opp = self._detect_negrisk_event(event, prices)
                if opp:
                    opportunities.append(opp)

            # NOTE: Date sweep strategy REMOVED - it was incorrectly classified as arbitrage
            # "By X date" markets are CUMULATIVE, not mutually exclusive:
            # - If event happens by March, it ALSO happened "by June" and "by December"
            # - So ALL NO positions lose together = 100% correlated loss
            # - This is a SPECULATIVE BET, not arbitrage

            # Strategy B: Multi-outcome (buy YES on all outcomes)
            # Only for non-date-based events with verified exhaustive outcomes
            opp = self._detect_multi_outcome(event, prices)
            if opp:
                opportunities.append(opp)

        return opportunities

    def _detect_negrisk_event(
        self, event: Event, prices: dict[str, dict]
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

        # In NegRisk, exactly one outcome wins, so total YES should = $1
        if total_yes >= 1.0:
            return None

        return self.create_opportunity(
            title=f"NegRisk: {event.title[:50]}...",
            description=f"Buy YES on all {len(active_markets)} outcomes for ${total_yes:.3f}, one wins = $1",
            total_cost=total_yes,
            markets=active_markets,
            positions=positions,
            event=event,
        )

    def _detect_date_sweep(
        self, event: Event, prices: dict[str, dict]
    ) -> ArbitrageOpportunity | None:
        """
        DEPRECATED - DO NOT USE

        This method was REMOVED because it incorrectly classified speculative bets as arbitrage.

        WHY IT'S WRONG:
        "By X date" markets are CUMULATIVE, not mutually exclusive:
        - "Event by March" YES → "Event by June" YES → "Event by December" YES
        - If you buy NO on all dates and the event happens early, ALL positions lose
        - This is a SPECULATIVE BET with 100% correlated downside, NOT arbitrage

        Example of the bug:
        - Buy NO on "Cabinet member out by March" @ $0.002
        - Buy NO on "Cabinet member out by June" @ $0.002
        - Buy NO on "Cabinet member out by December" @ $0.002
        - If cabinet member leaves in February: ALL THREE NO positions = $0
        - Total loss: 100% of investment

        True arbitrage requires MUTUALLY EXCLUSIVE outcomes where exactly one wins.
        Date-based "by X" markets fail this requirement.
        """
        # This method is intentionally disabled
        # Returning None ensures it never produces false arbitrage signals
        return None

    def _is_independent_betting_market(self, question: str) -> bool:
        """
        Check if a market is an independent betting type (spread, over/under, etc.)
        These are NOT mutually exclusive with other markets in the same event.
        """
        question_lower = question.lower()

        # Independent bet type keywords - these can all be true simultaneously
        independent_keywords = [
            "spread",
            "handicap",
            "-1.5",
            "+1.5",
            "-0.5",
            "+0.5",
            "-2.5",
            "+2.5",
            "over/under",
            "o/u",
            "over ",
            "under ",
            "total goals",
            "total points",
            "both teams",
            "btts",
            "both to score",
            "first half",
            "second half",
            "1st half",
            "2nd half",
            "corners",
            "cards",
            "yellow",
            "red card",
            "clean sheet",
            "to nil",
            "anytime scorer",
            "first scorer",
            "last scorer",
            "odd/even",
            "odd goals",
            "even goals",
        ]

        return any(kw in question_lower for kw in independent_keywords)

    def _is_date_based_market(self, question: str) -> bool:
        """
        Check if a market is date-based (cumulative "by X date" style).
        These are NOT mutually exclusive and should be excluded from arbitrage.
        """
        question_lower = question.lower()

        # Date-based keywords that indicate cumulative markets
        date_keywords = [
            "by january",
            "by february",
            "by march",
            "by april",
            "by may",
            "by june",
            "by july",
            "by august",
            "by september",
            "by october",
            "by november",
            "by december",
            "by jan",
            "by feb",
            "by mar",
            "by apr",
            "by jun",
            "by jul",
            "by aug",
            "by sep",
            "by oct",
            "by nov",
            "by dec",
            "before january",
            "before february",
            "before march",
            "before april",
            "by q1",
            "by q2",
            "by q3",
            "by q4",
            "by end of",
            "by the end of",
            "by 2025",
            "by 2026",
            "by 2027",
        ]

        return any(kw in question_lower for kw in date_keywords)

    def _detect_multi_outcome(
        self, event: Event, prices: dict[str, dict]
    ) -> ArbitrageOpportunity | None:
        """
        Detect multi-outcome arbitrage: exhaustive outcomes where one must win
        Buy YES on all outcomes when total < $1

        IMPORTANT: Only works for mutually exclusive outcomes like:
        - "Who wins the election?" with multiple candidates
        - "Which team wins?" with Team A / Team B / Draw

        Does NOT work for:
        - Independent betting markets (spread, over/under, BTTS - can all be true!)
        - Date-based markets ("by March", "by June" - cumulative, not exclusive!)
        """
        # Skip if already handled as NegRisk
        if event.neg_risk:
            return None

        active_markets = [m for m in event.markets if m.active and not m.closed]
        if len(active_markets) < 3:  # Need at least 3 outcomes
            return None

        # CRITICAL: Filter out independent betting markets
        # These are NOT mutually exclusive - spread, over/under, BTTS can all be true!
        exclusive_markets = [
            m
            for m in active_markets
            if not self._is_independent_betting_market(m.question)
        ]

        # CRITICAL: Filter out date-based markets
        # "By X date" markets are CUMULATIVE, not mutually exclusive!
        exclusive_markets = [
            m for m in exclusive_markets if not self._is_date_based_market(m.question)
        ]

        # If most markets are independent bet types or date-based, skip this event
        if len(exclusive_markets) < 3:
            return None

        # If there's a mix, the event likely has multiple bet types - skip
        if len(exclusive_markets) != len(active_markets):
            return None

        # Calculate total YES cost
        total_yes = 0.0
        positions = []

        for market in exclusive_markets:
            yes_price = market.yes_price

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

        if total_yes >= 1.0:
            return None

        # Additional sanity check: if total is way below 1.0,
        # these probably aren't exhaustive options
        if total_yes < 0.7:
            return None

        # Stricter threshold for multi-outcome - require higher confidence
        # that these are truly exhaustive options
        if total_yes < 0.85:
            # Lower totals suggest missing outcomes - add extra warning
            opp = self.create_opportunity(
                title=f"⚠️ Multi-Outcome: {event.title[:35]}...",
                description=f"LOW CONFIDENCE: Total {total_yes:.0%} suggests missing outcomes. Buy all {len(exclusive_markets)} YES for ${total_yes:.3f}",
                total_cost=total_yes,
                markets=exclusive_markets,
                positions=positions,
                event=event,
            )
            if opp:
                opp.risk_factors.insert(
                    0,
                    f"⚠️ LOW TOTAL ({total_yes:.0%}) - likely missing outcomes, HIGH RISK",
                )
            return opp

        opp = self.create_opportunity(
            title=f"Multi-Outcome: {event.title[:40]}...",
            description=f"Buy YES on all {len(exclusive_markets)} outcomes for ${total_yes:.3f}, one wins = $1",
            total_cost=total_yes,
            markets=exclusive_markets,
            positions=positions,
            event=event,
        )

        # Still add a warning since we can't verify exhaustiveness
        if opp:
            opp.risk_factors.insert(
                0, "Verify manually: ensure all possible outcomes are listed"
            )

        return opp
