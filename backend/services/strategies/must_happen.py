from __future__ import annotations

from typing import Any

from models import Market, Event, ArbitrageOpportunity
from config import settings
from .base import BaseStrategy, DecisionCheck, StrategyDecision, ExitDecision
from services.strategies._evaluate_helpers import to_float, to_confidence, signal_payload


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

    strategy_type = "must_happen"
    name = "Must-Happen"
    description = "Buy YES on all outcomes - REQUIRES MANUAL VERIFICATION of exhaustiveness"
    mispricing_type = "within_market"

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
        # Additional keywords
        "who wins",
        "which team",
        "which country",
        "which company",
        "which candidate",
        "who becomes",
        "who is the next",
        "what color",
        "what is the",
        "who gets",
        "mvp",
        "finals winner",
    ]

    def detect(self, events: list[Event], markets: list[Market], prices: dict[str, dict]) -> list[ArbitrageOpportunity]:
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

        # Check if markets include an explicit "Other" / "None of the above" outcome
        # which strongly suggests the event is designed to be exhaustive
        other_keywords = [
            "other",
            "none of the above",
            "someone else",
            "no one",
            "field",
            "another",
            "different",
        ]
        for m in event.markets:
            q_lower = m.question.lower()
            if any(kw in q_lower for kw in other_keywords):
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

    def _is_date_based_event(self, event: Event) -> bool:
        """Check if an event's markets are cumulative date-based ("by X" style).

        "Delisted by March", "Ceasefire broken by June", "Bitcoin above $X by Q4"
        are CUMULATIVE: if the event happens in March, the June market also resolves YES.
        Buying YES on all dates is a directional bet, NOT arbitrage.
        """
        questions = [m.question.lower() for m in event.markets]
        # If most markets contain date/time keywords, it's a date-based event
        date_patterns = [
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
            "by q1",
            "by q2",
            "by q3",
            "by q4",
            "by end of",
            "by the end of",
            "by 2025",
            "by 2026",
            "by 2027",
            "before ",
        ]
        date_count = sum(1 for q in questions if any(pattern in q for pattern in date_patterns))
        # If more than half the markets are date-based, reject the whole event
        return date_count > len(questions) * 0.5

    def _detect_must_happen(self, event: Event, prices: dict[str, dict]) -> ArbitrageOpportunity | None:
        """Detect must-happen arbitrage opportunity"""
        active_markets = [m for m in event.markets if m.active and not m.closed]

        if len(active_markets) < 2:
            return None

        # CRITICAL: Reject date-based cumulative markets
        # "X by March", "X by June" are cumulative, not mutually exclusive
        if self._is_date_based_event(event):
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
                    "token_id": market.clob_token_ids[0] if market.clob_token_ids else None,
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
            opp.risk_factors.insert(0, "⚠️ REQUIRES MANUAL VERIFICATION - may have hidden outcomes")
            if total_yes < 0.90:
                opp.risk_factors.insert(1, f"Low total ({total_yes:.0%}) suggests possible missing outcomes")

        return opp

    def evaluate(self, signal: Any, context: dict) -> StrategyDecision:
        params = context.get("params") or {}
        payload = signal_payload(signal)

        min_edge = to_float(params.get("min_edge_percent", 3.0), 3.0)
        min_conf = to_confidence(params.get("min_confidence", 0.42), 0.42)
        max_risk = to_confidence(params.get("max_risk_score", 0.68), 0.68)
        min_markets = max(1, int(to_float(params.get("min_markets", 2), 2)))
        base_size = max(1.0, to_float(params.get("base_size_usd", 20.0), 20.0))
        max_size = max(base_size, to_float(params.get("max_size_usd", 180.0), 180.0))

        edge = max(0.0, to_float(getattr(signal, "edge_percent", 0.0), 0.0))
        confidence = to_confidence(getattr(signal, "confidence", 0.0), 0.0)
        risk_score = to_confidence(payload.get("risk_score", 0.5), 0.5)
        market_count = len(payload.get("markets") or [])
        is_guaranteed = bool(payload.get("is_guaranteed", True))

        checks = [
            DecisionCheck("edge", "Edge threshold", edge >= min_edge, score=edge, detail=f"min={min_edge:.2f}"),
            DecisionCheck("confidence", "Confidence threshold", confidence >= min_conf, score=confidence, detail=f"min={min_conf:.2f}"),
            DecisionCheck("risk_score", "Risk score ceiling", risk_score <= max_risk, score=risk_score, detail=f"max={max_risk:.2f}"),
            DecisionCheck("markets", "Multi-leg structure", market_count >= min_markets, score=float(market_count), detail=f"min={min_markets}"),
        ]

        score = (edge * 0.65) + (confidence * 35.0) - (risk_score * 10.0) + (min(6, market_count) * 1.2)
        if is_guaranteed:
            score += 4.0

        if not all(c.passed for c in checks):
            return StrategyDecision("skipped", "Structural filters not met", score=score, checks=checks)

        size = base_size * (1.0 + (edge / 120.0)) * (0.8 + confidence) * (1.0 + min(0.45, market_count * 0.06))
        size = max(1.0, min(max_size, size))

        return StrategyDecision("selected", "Structural signal selected", score=score, size_usd=size, checks=checks)

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        """Guaranteed-spread: hold to resolution for maximum value."""
        if market_state.get("is_resolved"):
            return self.default_exit_check(position, market_state)
        config = getattr(position, "config", None) or {}
        if not config.get("resolve_only", True):
            return self.default_exit_check(position, market_state)
        return ExitDecision("hold", "Guaranteed spread — holding to resolution")
