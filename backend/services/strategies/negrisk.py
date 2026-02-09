from __future__ import annotations

from models import Market, Event, ArbitrageOpportunity, StrategyType
from config import settings
from .base import BaseStrategy, make_aware


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
        """Detect arbitrage in official NegRisk events.

        IMPORTANT: The NegRisk flag guarantees mutual exclusivity among LISTED
        outcomes, but does NOT guarantee the list is EXHAUSTIVE. For example,
        an election NegRisk bundle may list 6 candidates, but if a 7th (unlisted)
        candidate wins, ALL listed outcomes resolve to $0 = 100% loss.

        A low total YES price (e.g., $0.60-0.80) is a strong signal that the
        market prices in a significant probability of an unlisted outcome winning.
        This is NOT a mispricing — it's rational pricing of non-exhaustive risk.
        """
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

        # --- Non-exhaustive outcome detection ---
        # A total YES well below 1.0 indicates the market is pricing in a
        # significant chance of an UNLISTED outcome winning. This is especially
        # common in election primaries, special elections, and multi-candidate races.
        # The "spread" is NOT a mispricing — it's rational non-exhaustive risk pricing.
        if total_yes < settings.NEGRISK_MIN_TOTAL_YES:
            return None

        # --- Outcome-count-aware threshold ---
        # Markets with very few listed outcomes are far more likely to be
        # non-exhaustive (e.g., 4 named companies + "other" for an acquisition
        # with dozens of potential acquirers).  Require higher total_yes for
        # small outcome sets: <=5 outcomes need 0.98+, 6-8 need 0.97+.
        num_outcomes = len(active_markets)
        if num_outcomes <= 5 and total_yes < 0.98:
            return None
        elif num_outcomes <= 8 and total_yes < settings.NEGRISK_WARN_TOTAL_YES:
            return None

        # --- Structural non-exhaustiveness checks ---
        is_election = self._is_election_market(event.title)
        is_open_ended = self._is_open_ended_event(event.title)

        # Election markets with exactly 2 candidates (Dem vs Rep) are NEVER
        # exhaustive — independent/third-party candidates can always win.
        # Only accept election markets if they have enough candidates that
        # the total YES sum is very close to 1.0.
        if is_election and len(active_markets) <= 2:
            return None

        # Election/primary markets almost ALWAYS have unlisted candidates.
        # Require a much higher total YES (closer to 1.0) to accept these.
        # A 3-7% gap in a primary market is the market pricing "Other/Field",
        # not a mispricing we can exploit.
        if is_election and total_yes < settings.NEGRISK_ELECTION_MIN_TOTAL_YES:
            return None

        # Open-ended events (Nobel Prize, awards, "best X") where the outcome
        # universe is inherently unbounded — anyone/anything could win.
        # The listed outcomes can never be exhaustive.
        if is_open_ended:
            return None

        # --- Resolution date mismatch detection ---
        # All outcomes in a NegRisk bundle must resolve at the same time.
        # If outcome A resolves June 30 but outcome B resolves Dec 31,
        # there's a window where ALL outcomes can resolve NO (e.g., event
        # happens between July-December). This makes the "arbitrage" a trap.
        end_dates = [make_aware(m.end_date) for m in active_markets if m.end_date]
        if len(end_dates) >= 2:
            earliest = min(end_dates)
            latest = max(end_dates)
            spread_days = (latest - earliest).days
            if spread_days > settings.NEGRISK_MAX_RESOLUTION_SPREAD_DAYS:
                return None

        opp = self.create_opportunity(
            title=f"NegRisk: {event.title[:50]}...",
            description=f"Buy YES on all {len(active_markets)} outcomes for ${total_yes:.3f}, one wins = $1",
            total_cost=total_yes,
            markets=active_markets,
            positions=positions,
            event=event,
        )

        if opp:
            if total_yes < settings.NEGRISK_WARN_TOTAL_YES:
                opp.risk_factors.insert(
                    0,
                    f"Total YES ({total_yes:.1%}) below 97% — possible missing outcomes",
                )
            if is_election:
                opp.risk_factors.insert(
                    0,
                    "Election/primary market: unlisted candidates or 'Other' outcome may not be covered",
                )

        return opp

    def _is_election_market(self, title: str) -> bool:
        """Check if an event title suggests an election or primary."""
        title_lower = title.lower()
        election_keywords = [
            "election",
            "primary",
            "governor",
            "house",
            "senate",
            "congress",
            "mayor",
            "special election",
            "presidential",
            "nominee",
            "caucus",
            "runoff",
        ]
        return any(kw in title_lower for kw in election_keywords)

    def _is_open_ended_event(self, title: str) -> bool:
        """Check if an event has an inherently unbounded outcome universe.

        These events can NEVER have an exhaustive listed outcome set because
        the set of possible winners/results is effectively infinite.

        Examples:
        - "Nobel Peace Prize Winner 2026" (anyone in the world could win)
        - "Which company has the best AI model" (subjective + any company)
        - "Oscar Best Picture Winner" (nominees change)
        - "Who will acquire Warner Bros" (any company could acquire)
        """
        title_lower = title.lower()
        open_ended_keywords = [
            "nobel",
            "prize winner",
            "award winner",
            "best ai",
            "best model",
            "oscar",
            "grammy",
            "emmy",
            "ballon d'or",
            "mvp",
            "player of the year",
            "time person of the year",
            "pulitzer",
            # M&A / acquisition markets — the universe of potential acquirers
            # is inherently unbounded (any company could acquire the target).
            "acquisition",
            "acquire",
            "merger",
            "takeover",
            "buyout",
            "who will buy",
            "who will purchase",
            "who will close",
            # "Next CEO/coach/leader" type markets
            "next ceo",
            "next coach",
            "next head coach",
            "next manager",
            "next leader",
            "who will replace",
        ]
        return any(kw in title_lower for kw in open_ended_keywords)

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

        # Reject if total YES is too low — almost certainly non-exhaustive outcomes
        if total_yes < settings.NEGRISK_MIN_TOTAL_YES:
            return None

        # Outcome-count-aware threshold (same as NegRisk)
        num_outcomes = len(exclusive_markets)
        if num_outcomes <= 5 and total_yes < 0.98:
            return None
        elif num_outcomes <= 8 and total_yes < settings.NEGRISK_WARN_TOTAL_YES:
            return None

        # Structural non-exhaustiveness checks (same as NegRisk)
        is_election = self._is_election_market(event.title)
        if is_election and len(exclusive_markets) <= 2:
            return None
        if is_election and total_yes < settings.NEGRISK_ELECTION_MIN_TOTAL_YES:
            return None
        if self._is_open_ended_event(event.title):
            return None

        # Resolution date mismatch check (same as NegRisk)
        end_dates = [make_aware(m.end_date) for m in exclusive_markets if m.end_date]
        if len(end_dates) >= 2:
            earliest = min(end_dates)
            latest = max(end_dates)
            spread_days = (latest - earliest).days
            if spread_days > settings.NEGRISK_MAX_RESOLUTION_SPREAD_DAYS:
                return None

        opp = self.create_opportunity(
            title=f"Multi-Outcome: {event.title[:40]}...",
            description=f"Buy YES on all {len(exclusive_markets)} outcomes for ${total_yes:.3f}, one wins = $1",
            total_cost=total_yes,
            markets=exclusive_markets,
            positions=positions,
            event=event,
        )

        if opp:
            opp.risk_factors.insert(
                0, "Verify manually: ensure all possible outcomes are listed"
            )
            if total_yes < settings.NEGRISK_WARN_TOTAL_YES:
                opp.risk_factors.insert(
                    0,
                    f"Total YES ({total_yes:.1%}) below 97% — possible missing outcomes",
                )
            if is_election:
                opp.risk_factors.insert(
                    0,
                    "Election/primary market: unlisted candidates may not be covered",
                )

        return opp
