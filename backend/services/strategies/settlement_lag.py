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

from __future__ import annotations

from typing import Any, Optional
from models import Market, Event, ArbitrageOpportunity, MispricingType
from config import settings
from .base import BaseStrategy, DecisionCheck, ExitDecision, ScoringWeights, SizingConfig, utcnow, make_aware
from utils.converters import to_float
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

    strategy_type = "settlement_lag"
    name = "Settlement Lag"
    description = "Exploit delayed price updates after outcome determination"
    mispricing_type = "settlement_lag"
    subscriptions = ["market_data_refresh"]

    scoring_weights = ScoringWeights(
        edge_weight=0.55,
        confidence_weight=30.0,
        risk_penalty=8.0,
        liquidity_weight=8.0,
        liquidity_divisor=5000.0,
        structural_bonus=2.0,
    )
    sizing_config = SizingConfig(
        base_divisor=100.0,
        confidence_offset=0.75,
        risk_scale_factor=0.35,
        risk_floor=0.55,
        market_scale_factor=0.0,
    )
    default_config = {
        "min_edge_percent": 4.0,
        "min_confidence": 0.45,
        "max_risk_score": 0.78,
        "min_liquidity": 25.0,
        "base_size_usd": 18.0,
        "max_size_usd": 150.0,
    }

    # Thresholds — read from config (persisted in DB via Settings UI)
    # Default NEAR_ZERO lowered from 0.05 to 0.02: many active markets trade at
    # $0.04 on the longshot side; resolved markets are typically $0.001-$0.01.
    @property
    def NEAR_ZERO_THRESHOLD(self):
        return settings.SETTLEMENT_LAG_NEAR_ZERO if hasattr(settings, "SETTLEMENT_LAG_NEAR_ZERO") else 0.02

    @property
    def NEAR_ONE_THRESHOLD(self):
        return settings.SETTLEMENT_LAG_NEAR_ONE if hasattr(settings, "SETTLEMENT_LAG_NEAR_ONE") else 0.95

    @property
    def MIN_SUM_DEVIATION(self):
        return (
            settings.SETTLEMENT_LAG_MIN_SUM_DEVIATION if hasattr(settings, "SETTLEMENT_LAG_MIN_SUM_DEVIATION") else 0.03
        )

    OVERDUE_RESOLUTION_DAYS = 0  # Market past resolution date

    def detect(self, events: list[Event], markets: list[Market], prices: dict[str, dict]) -> list[ArbitrageOpportunity]:
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
        """Check a binary market for settlement lag opportunities.

        Settlement lag requires the market to be AT or NEAR its resolution date.
        A market months away from resolution with sum < 1.0 is NOT settlement lag —
        it's normal market pricing reflecting uncertainty and time value of money.
        """

        total = yes_price + no_price
        now = utcnow()
        max_days = settings.SETTLEMENT_LAG_MAX_DAYS_TO_RESOLUTION

        # --- Gate: Resolution date must be near or past ---
        # If the market has a known resolution date far in the future,
        # any deviation from $1.00 is NOT settlement lag.
        if market.end_date:
            end_aware = make_aware(market.end_date)
            days_until = (end_aware - now).days
            if days_until > max_days:
                return None  # Too far from resolution to be settlement lag

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

            # Urgency scoring: fresher settlements are more valuable
            urgency = "LOW"
            if is_overdue:
                end_aware = make_aware(market.end_date)
                hours_overdue = (now - end_aware).total_seconds() / 3600
                if hours_overdue < 1:
                    urgency = "CRITICAL"  # Just resolved
                elif hours_overdue < 6:
                    urgency = "HIGH"
                elif hours_overdue < 24:
                    urgency = "MEDIUM"
                else:
                    urgency = "LOW"  # Likely already captured
            elif appears_resolved:
                urgency = "HIGH"  # Price signal but not overdue yet

            opp.risk_factors.append(f"Settlement urgency: {urgency}")
            if urgency == "LOW":
                opp.risk_factors.append("WARNING: Opportunity may already be captured by faster bots")
        return opp

    def _check_negrisk_settlement(self, event: Event, prices: dict[str, dict]) -> list[ArbitrageOpportunity]:
        """Check NegRisk events for settlement lag in multi-outcome markets.

        NegRisk events are especially vulnerable to settlement lag because
        they have multiple outcomes that must be coordinated. When one
        outcome is determined, the others don't instantly adjust.

        CRITICAL: A NegRisk event with sum YES < 1.0 is NOT settlement lag
        if the event hasn't occurred yet. Common causes of low sums:
        - Non-exhaustive outcome lists (missing "Other" candidate)
        - Time value of money / long-duration capital lockup
        - Low liquidity / stale order books

        True settlement lag requires evidence that the event HAS resolved
        (near-zero/near-one prices) AND is near/past its resolution date.
        """
        opportunities = []
        active_markets = [m for m in event.markets if not m.closed and m.active]

        if len(active_markets) < 2:
            return []

        # --- Gate: Event must be near resolution ---
        # Check if ANY market in the event has a resolution date within the window.
        # If all markets resolve far in the future, this is NOT settlement lag.
        now = utcnow()
        max_days = settings.SETTLEMENT_LAG_MAX_DAYS_TO_RESOLUTION
        any_near_resolution = False
        any_overdue = False
        has_resolution_dates = False

        for m in active_markets:
            if m.end_date:
                has_resolution_dates = True
                end_aware = make_aware(m.end_date)
                days_until = (end_aware - now).days
                if days_until <= max_days:
                    any_near_resolution = True
                if days_until <= 0:
                    any_overdue = True

        if has_resolution_dates and not any_near_resolution:
            return []  # Event is too far from resolution to be settlement lag

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

            if yes_price <= self.NEAR_ZERO_THRESHOLD:
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

        # Require STRONG settlement signals, not just near-zero prices.
        # In a 22-candidate race, most candidates SHOULD have near-zero prices —
        # that's normal market pricing, not settlement lag.
        #
        # True settlement lag requires:
        # - At least one outcome near 1.0 (winner is known but not yet settled), OR
        # - Event is overdue (past resolution date)
        #
        # Near-zero prices alone are NOT evidence of settlement — they're just
        # unpopular outcomes in a multi-candidate race.
        has_settlement_signal = near_one_count > 0 or any_overdue or (near_zero_count >= 2 and total_yes >= 0.80)
        if not has_settlement_signal:
            return []

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

            # Identify the likely winner for sizing/execution priority
            likely_winner = None
            likely_winner_price = 0.0
            for m, price in market_prices:
                if price > likely_winner_price:
                    likely_winner = m.question[:50]
                    likely_winner_price = price

            signals = [f"sum YES = ${total_yes:.3f}"]
            if near_zero_count > 0:
                signals.append(f"{near_zero_count} outcomes near zero")
            if near_one_count > 0:
                signals.append(f"{near_one_count} outcomes near one")
            if any_overdue:
                signals.append("past resolution date")
            if likely_winner and likely_winner_price > self.NEAR_ONE_THRESHOLD:
                signals.append(f"likely winner: {likely_winner} (${likely_winner_price:.3f})")

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

                # Urgency scoring for NegRisk settlement
                urgency = "MEDIUM"
                if any_overdue:
                    urgency = "HIGH"
                if near_one_count > 0 and any_overdue:
                    urgency = "CRITICAL"
                opp.risk_factors.append(f"Settlement urgency: {urgency}")
                if likely_winner:
                    opp.risk_factors.append(f"Likely winner: {likely_winner}")

                opportunities.append(opp)

        return opportunities

    def custom_checks(self, signal: Any, context: dict, params: dict, payload: dict) -> list[DecisionCheck]:
        min_liquidity = max(0.0, to_float(params.get("min_liquidity", 25.0), 25.0))
        liquidity = max(0.0, to_float(getattr(signal, "liquidity", 0.0), 0.0))
        payload["_signal_liquidity"] = liquidity
        return [
            DecisionCheck(
                "liquidity",
                "Liquidity floor",
                liquidity >= min_liquidity,
                score=liquidity,
                detail=f"min={min_liquidity:.0f}",
            ),
        ]

    def compute_score(
        self, edge: float, confidence: float, risk_score: float, market_count: int, payload: dict
    ) -> float:
        liquidity = float(payload.get("_signal_liquidity", 0) or 0)
        is_guaranteed = bool(payload.get("is_guaranteed", True))
        return (
            (edge * 0.55)
            + (confidence * 30.0)
            + (min(1.0, liquidity / 5000.0) * 8.0)
            - (risk_score * 8.0)
            + (2.0 if is_guaranteed else 0.0)
        )

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        """Guaranteed-spread: hold to resolution for maximum value."""
        if market_state.get("is_resolved"):
            return self.default_exit_check(position, market_state)
        config = getattr(position, "config", None) or {}
        if not config.get("resolve_only", True):
            return self.default_exit_check(position, market_state)
        return ExitDecision("hold", "Guaranteed spread — holding to resolution")
