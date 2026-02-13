"""
Comprehensive tests for Tier 1 (simple arithmetic arbitrage) strategies.

Covers:
- BasicArbStrategy: Buy YES + NO on same market when total < $1
- NegRiskStrategy: Buy YES on all outcomes in NegRisk / multi-outcome events
- BaseStrategy: Risk scoring and opportunity creation (via concrete implementations)
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import pytest
from datetime import datetime, timezone, timedelta
from utils.utcnow import utcnow

from models.market import Market, Event, Token
from models.opportunity import ArbitrageOpportunity, StrategyType
from services.strategies.basic import BasicArbStrategy
from services.strategies.negrisk import NegRiskStrategy


# ---------------------------------------------------------------------------
# Fixtures: helper factories for building Market / Event test data
# ---------------------------------------------------------------------------


def make_market(
    market_id: str = "m1",
    question: str = "Will it rain tomorrow?",
    yes_price: float = 0.50,
    no_price: float = 0.50,
    liquidity: float = 10000.0,
    active: bool = True,
    closed: bool = False,
    neg_risk: bool = False,
    end_date: datetime | None = None,
    clob_token_ids: list[str] | None = None,
) -> Market:
    """Create a binary Market with sensible defaults."""
    outcome_prices = [yes_price, no_price]
    if clob_token_ids is None:
        clob_token_ids = [f"yes_{market_id}", f"no_{market_id}"]
    tokens = [
        Token(token_id=clob_token_ids[0], outcome="Yes", price=yes_price),
        Token(token_id=clob_token_ids[1], outcome="No", price=no_price),
    ]
    return Market(
        id=market_id,
        condition_id=f"cond_{market_id}",
        question=question,
        slug=f"slug-{market_id}",
        tokens=tokens,
        clob_token_ids=clob_token_ids,
        outcome_prices=outcome_prices,
        active=active,
        closed=closed,
        neg_risk=neg_risk,
        volume=5000.0,
        liquidity=liquidity,
        end_date=end_date,
    )


def make_single_outcome_market(
    market_id: str = "m1",
    question: str = "Who wins?",
    yes_price: float = 0.30,
    liquidity: float = 10000.0,
    active: bool = True,
    closed: bool = False,
    end_date: datetime | None = None,
    clob_token_ids: list[str] | None = None,
) -> Market:
    """Create a market with only a YES outcome price (used in NegRisk events)."""
    if clob_token_ids is None:
        clob_token_ids = [f"yes_{market_id}", f"no_{market_id}"]
    outcome_prices = [yes_price, 1.0 - yes_price]
    tokens = [
        Token(token_id=clob_token_ids[0], outcome="Yes", price=yes_price),
        Token(token_id=clob_token_ids[1], outcome="No", price=1.0 - yes_price),
    ]
    return Market(
        id=market_id,
        condition_id=f"cond_{market_id}",
        question=question,
        slug=f"slug-{market_id}",
        tokens=tokens,
        clob_token_ids=clob_token_ids,
        outcome_prices=outcome_prices,
        active=active,
        closed=closed,
        liquidity=liquidity,
        end_date=end_date,
    )


def make_event(
    event_id: str = "e1",
    title: str = "Test Event",
    markets: list[Market] | None = None,
    neg_risk: bool = False,
    closed: bool = False,
    category: str | None = "politics",
) -> Event:
    """Create an Event with sensible defaults."""
    return Event(
        id=event_id,
        slug=f"slug-{event_id}",
        title=title,
        description="A test event",
        category=category,
        markets=markets or [],
        neg_risk=neg_risk,
        active=True,
        closed=closed,
    )


# ===========================================================================
# BasicArbStrategy tests
# ===========================================================================


class TestBasicArbStrategy:
    """Tests for BasicArbStrategy.detect()"""

    def setup_method(self):
        self.strategy = BasicArbStrategy()

    # --- Core detection ---

    def test_detect_opportunity_when_sum_below_one(self):
        """YES + NO < 1.0 with profit above threshold should find opportunity."""
        # total = 0.45 + 0.45 = 0.90
        # gross = 0.10, fee = 0.02, net = 0.08, roi = 8.89%
        market = make_market(yes_price=0.45, no_price=0.45)
        opps = self.strategy.detect(events=[], markets=[market], prices={})
        assert len(opps) == 1
        opp = opps[0]
        assert opp.strategy == StrategyType.BASIC
        assert opp.total_cost == pytest.approx(0.90)
        assert opp.expected_payout == 1.0
        assert opp.net_profit == pytest.approx(0.08)
        assert opp.roi_percent == pytest.approx(8.888888888888889)

    def test_no_opportunity_when_sum_equals_one(self):
        """YES + NO == 1.0 should NOT produce an opportunity."""
        market = make_market(yes_price=0.50, no_price=0.50)
        opps = self.strategy.detect(events=[], markets=[market], prices={})
        assert len(opps) == 0

    def test_no_opportunity_when_sum_above_one(self):
        """YES + NO > 1.0 should NOT produce an opportunity."""
        market = make_market(yes_price=0.55, no_price=0.50)
        opps = self.strategy.detect(events=[], markets=[market], prices={})
        assert len(opps) == 0

    def test_no_opportunity_when_profit_below_threshold_after_fees(self):
        """
        YES + NO < 1.0 but profit after 2% fee is below 2.5% threshold.

        total = 0.96, gross = 0.04, fee = 0.02, net = 0.02
        roi = 0.02 / 0.96 = 2.08% < 2.5%
        """
        market = make_market(yes_price=0.48, no_price=0.48)
        opps = self.strategy.detect(events=[], markets=[market], prices={})
        assert len(opps) == 0

    def test_borderline_above_threshold(self):
        """
        Barely above the 2.5% threshold should create opportunity.

        total = 0.95, gross = 0.05, fee = 0.02, net = 0.03
        roi = 0.03 / 0.95 = 3.16% > 2.5%
        """
        market = make_market(yes_price=0.47, no_price=0.48)
        opps = self.strategy.detect(events=[], markets=[market], prices={})
        assert len(opps) == 1
        opp = opps[0]
        assert opp.roi_percent == pytest.approx(3.157894736842105)

    # --- Edge cases ---

    def test_zero_prices(self):
        """Both prices at zero: total=0, huge ROI, should detect."""
        market = make_market(yes_price=0.0, no_price=0.0)
        # total_cost=0 => roi calculation: 0/0 => 0 => won't pass threshold
        opps = self.strategy.detect(events=[], markets=[market], prices={})
        # roi = 0 when total_cost = 0 (guarded by the conditional)
        assert len(opps) == 0

    def test_missing_price_data_empty_outcome_prices(self):
        """Market with empty outcome_prices should use 0.0 defaults."""
        market = Market(
            id="m_empty",
            condition_id="cond_empty",
            question="Missing data market?",
            slug="missing",
            tokens=[],
            clob_token_ids=[],
            outcome_prices=[],
            active=True,
            closed=False,
            liquidity=10000.0,
        )
        # outcome_prices is empty => len != 2 => skip (not binary)
        opps = self.strategy.detect(events=[], markets=[market], prices={})
        assert len(opps) == 0

    def test_skips_non_binary_market(self):
        """Markets with != 2 outcomes should be skipped."""
        market = Market(
            id="m3out",
            condition_id="cond3",
            question="Three outcome?",
            slug="three",
            tokens=[],
            clob_token_ids=["t1", "t2", "t3"],
            outcome_prices=[0.30, 0.30, 0.30],
            active=True,
            closed=False,
            liquidity=10000.0,
        )
        opps = self.strategy.detect(events=[], markets=[market], prices={})
        assert len(opps) == 0

    def test_skips_closed_market(self):
        """Closed markets should be skipped."""
        market = make_market(yes_price=0.40, no_price=0.40, closed=True)
        opps = self.strategy.detect(events=[], markets=[market], prices={})
        assert len(opps) == 0

    def test_skips_inactive_market(self):
        """Inactive markets should be skipped."""
        market = make_market(yes_price=0.40, no_price=0.40, active=False)
        opps = self.strategy.detect(events=[], markets=[market], prices={})
        assert len(opps) == 0

    # --- Live prices override ---

    def test_live_prices_override_market_prices(self):
        """Live prices from prices dict should override market.outcome_prices."""
        # Market prices show no opportunity (total = 1.0)
        market = make_market(yes_price=0.50, no_price=0.50)
        # But live prices show opportunity
        prices = {
            "yes_m1": {"mid": 0.40},
            "no_m1": {"mid": 0.40},
        }
        opps = self.strategy.detect(events=[], markets=[market], prices=prices)
        assert len(opps) == 1
        opp = opps[0]
        assert opp.total_cost == pytest.approx(0.80)

    def test_partial_live_prices(self):
        """Only one token has live price; the other uses market price."""
        market = make_market(yes_price=0.50, no_price=0.50)
        prices = {
            "yes_m1": {"mid": 0.40},
            # no_m1 not in prices => falls back to market price 0.50
        }
        opps = self.strategy.detect(events=[], markets=[market], prices=prices)
        # total = 0.40 + 0.50 = 0.90 => opportunity
        assert len(opps) == 1
        assert opps[0].total_cost == pytest.approx(0.90)

    # --- ROI / fee calculation ---

    def test_roi_calculation_correctness(self):
        """Verify ROI formula: roi = ((1 - total - fee) / total) * 100."""
        market = make_market(yes_price=0.40, no_price=0.40)
        opps = self.strategy.detect(events=[], markets=[market], prices={})
        assert len(opps) == 1
        opp = opps[0]
        total = 0.80
        fee = 1.0 * 0.02
        net = 1.0 - total - fee
        expected_roi = (net / total) * 100
        assert opp.roi_percent == pytest.approx(expected_roi)
        assert opp.fee == pytest.approx(fee)
        assert opp.net_profit == pytest.approx(net)
        assert opp.gross_profit == pytest.approx(1.0 - total)

    def test_fee_is_two_percent_of_payout(self):
        """Fee should be 2% of expected payout ($1), i.e., always $0.02."""
        market = make_market(yes_price=0.45, no_price=0.45)
        opps = self.strategy.detect(events=[], markets=[market], prices={})
        assert len(opps) == 1
        assert opps[0].fee == pytest.approx(0.02)

    # --- Max position size ---

    def test_max_position_size_ten_percent_of_liquidity(self):
        """max_position_size = 10% of min liquidity across involved markets."""
        market = make_market(yes_price=0.40, no_price=0.40, liquidity=8000.0)
        opps = self.strategy.detect(events=[], markets=[market], prices={})
        assert len(opps) == 1
        assert opps[0].max_position_size == pytest.approx(800.0)

    # --- Multiple markets ---

    def test_multiple_markets_some_with_some_without_opportunity(self):
        """Should find opportunities only on qualifying markets."""
        m1 = make_market(market_id="m1", yes_price=0.40, no_price=0.40)  # opp
        m2 = make_market(market_id="m2", yes_price=0.50, no_price=0.50)  # no opp
        m3 = make_market(market_id="m3", yes_price=0.44, no_price=0.44)  # opp
        opps = self.strategy.detect(events=[], markets=[m1, m2, m3], prices={})
        assert len(opps) == 2

    # --- Positions ---

    def test_positions_to_take_structure(self):
        """Positions should have BUY YES and BUY NO entries."""
        market = make_market(yes_price=0.40, no_price=0.40)
        opps = self.strategy.detect(events=[], markets=[market], prices={})
        assert len(opps) == 1
        positions = opps[0].positions_to_take
        assert len(positions) == 2
        assert positions[0]["action"] == "BUY"
        assert positions[0]["outcome"] == "YES"
        assert positions[0]["price"] == pytest.approx(0.40)
        assert positions[1]["action"] == "BUY"
        assert positions[1]["outcome"] == "NO"
        assert positions[1]["price"] == pytest.approx(0.40)


# ===========================================================================
# NegRiskStrategy tests
# ===========================================================================


class TestNegRiskStrategy:
    """Tests for NegRiskStrategy.detect()"""

    def setup_method(self):
        self.strategy = NegRiskStrategy()

    # --- NegRisk event detection ---

    def test_negrisk_event_with_sum_below_one(self):
        """NegRisk event: sum of YES prices < 1.0 should find opportunity."""
        m1 = make_single_outcome_market(market_id="nr1", question="Candidate A?", yes_price=0.30)
        m2 = make_single_outcome_market(market_id="nr2", question="Candidate B?", yes_price=0.35)
        m3 = make_single_outcome_market(market_id="nr3", question="Candidate C?", yes_price=0.25)
        event = make_event(
            event_id="e_nr",
            title="Who wins the election?",
            markets=[m1, m2, m3],
            neg_risk=True,
        )
        opps = self.strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 1
        opp = opps[0]
        assert opp.strategy == StrategyType.NEGRISK
        assert opp.total_cost == pytest.approx(0.90)
        assert opp.event_id == "e_nr"
        assert opp.event_title == "Who wins the election?"

    def test_negrisk_event_with_sum_equals_one(self):
        """NegRisk event: sum == 1.0 should NOT find opportunity."""
        m1 = make_single_outcome_market(market_id="nr1", question="A?", yes_price=0.50)
        m2 = make_single_outcome_market(market_id="nr2", question="B?", yes_price=0.50)
        event = make_event(markets=[m1, m2], neg_risk=True)
        opps = self.strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 0

    def test_negrisk_event_with_sum_above_one(self):
        """NegRisk event: sum > 1.0 should find SHORT arbitrage opportunity."""
        m1 = make_single_outcome_market(market_id="nr1", question="A?", yes_price=0.55)
        m2 = make_single_outcome_market(market_id="nr2", question="B?", yes_price=0.55)
        event = make_event(markets=[m1, m2], neg_risk=True)
        opps = self.strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 1
        opp = opps[0]
        assert "Short" in opp.title
        assert all(p["outcome"] == "NO" for p in opp.positions_to_take)

    def test_non_negrisk_event_skipped_for_negrisk_detection(self):
        """Non-neg_risk event should not be processed by _detect_negrisk_event."""
        # 2-market event without neg_risk: _detect_negrisk_event won't fire,
        # and _detect_multi_outcome requires >=3 markets, so no opps.
        m1 = make_single_outcome_market(market_id="nr1", question="A?", yes_price=0.30)
        m2 = make_single_outcome_market(market_id="nr2", question="B?", yes_price=0.30)
        event = make_event(markets=[m1, m2], neg_risk=False)
        opps = self.strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 0

    def test_negrisk_with_multiple_outcomes(self):
        """NegRisk event with 5 outcomes, all summing below 1.0."""
        markets = [
            make_single_outcome_market(
                market_id=f"nr{i}",
                question=f"Candidate {chr(65 + i)}?",
                yes_price=0.15,
            )
            for i in range(5)
        ]
        # total = 5 * 0.15 = 0.75
        event = make_event(markets=markets, neg_risk=True)
        opps = self.strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 1
        opp = opps[0]
        assert opp.total_cost == pytest.approx(0.75)
        assert len(opp.positions_to_take) == 5

    def test_negrisk_with_some_closed_markets(self):
        """Closed markets within event should be excluded from consideration."""
        m1 = make_single_outcome_market(market_id="nr1", question="A?", yes_price=0.30)
        m2 = make_single_outcome_market(
            market_id="nr2",
            question="B?",
            yes_price=0.40,
            closed=True,
        )
        m3 = make_single_outcome_market(market_id="nr3", question="C?", yes_price=0.25)
        event = make_event(markets=[m1, m2, m3], neg_risk=True)
        opps = self.strategy.detect(events=[event], markets=[], prices={})
        # Active markets: m1 (0.30) + m3 (0.25) = 0.55
        assert len(opps) == 1
        assert opps[0].total_cost == pytest.approx(0.55)

    def test_negrisk_with_some_inactive_markets(self):
        """Inactive markets should be excluded."""
        m1 = make_single_outcome_market(market_id="nr1", question="A?", yes_price=0.30)
        m2 = make_single_outcome_market(
            market_id="nr2",
            question="B?",
            yes_price=0.40,
            active=False,
        )
        m3 = make_single_outcome_market(market_id="nr3", question="C?", yes_price=0.20)
        event = make_event(markets=[m1, m2, m3], neg_risk=True)
        opps = self.strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 1
        assert opps[0].total_cost == pytest.approx(0.50)

    def test_negrisk_single_market_event_skipped(self):
        """Events with only one market should be skipped."""
        m1 = make_single_outcome_market(market_id="nr1", question="A?", yes_price=0.30)
        event = make_event(markets=[m1], neg_risk=True)
        opps = self.strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 0

    def test_negrisk_closed_event_skipped(self):
        """Closed events should be skipped entirely."""
        m1 = make_single_outcome_market(market_id="nr1", question="A?", yes_price=0.30)
        m2 = make_single_outcome_market(market_id="nr2", question="B?", yes_price=0.30)
        event = make_event(markets=[m1, m2], neg_risk=True, closed=True)
        opps = self.strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 0

    def test_negrisk_live_prices_override(self):
        """Live prices should override market YES prices in NegRisk events."""
        m1 = make_single_outcome_market(market_id="nr1", question="A?", yes_price=0.50)
        m2 = make_single_outcome_market(market_id="nr2", question="B?", yes_price=0.50)
        event = make_event(markets=[m1, m2], neg_risk=True)
        # Market prices sum to 1.0 - no opp. But live prices are lower.
        prices = {
            "yes_nr1": {"mid": 0.40},
            "yes_nr2": {"mid": 0.40},
        }
        opps = self.strategy.detect(events=[event], markets=[], prices=prices)
        assert len(opps) == 1
        assert opps[0].total_cost == pytest.approx(0.80)

    def test_negrisk_all_active_markets_closed_returns_none(self):
        """If all markets in NegRisk event are closed, no opportunity."""
        m1 = make_single_outcome_market(market_id="nr1", question="A?", yes_price=0.30, closed=True)
        m2 = make_single_outcome_market(market_id="nr2", question="B?", yes_price=0.30, closed=True)
        event = make_event(markets=[m1, m2], neg_risk=True)
        opps = self.strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 0

    # --- Multi-outcome (non-negrisk) detection ---

    def test_multi_outcome_with_sum_below_one(self):
        """Non-negrisk event with >=3 exclusive outcomes, total YES < 1.0."""
        m1 = make_single_outcome_market(market_id="mo1", question="Team Alpha wins?", yes_price=0.30)
        m2 = make_single_outcome_market(market_id="mo2", question="Team Beta wins?", yes_price=0.30)
        m3 = make_single_outcome_market(market_id="mo3", question="Team Gamma wins?", yes_price=0.30)
        event = make_event(markets=[m1, m2, m3], neg_risk=False)
        # total = 0.90, above 0.85 threshold
        opps = self.strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 1
        opp = opps[0]
        assert opp.total_cost == pytest.approx(0.90)

    def test_multi_outcome_sum_too_low_rejected(self):
        """Multi-outcome with total < 0.7 suggests missing outcomes -> skip."""
        m1 = make_single_outcome_market(market_id="mo1", question="Team Alpha wins?", yes_price=0.20)
        m2 = make_single_outcome_market(market_id="mo2", question="Team Beta wins?", yes_price=0.20)
        m3 = make_single_outcome_market(market_id="mo3", question="Team Gamma wins?", yes_price=0.20)
        event = make_event(markets=[m1, m2, m3], neg_risk=False)
        # total = 0.60 < 0.7 threshold
        opps = self.strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 0

    def test_multi_outcome_low_confidence_warning(self):
        """Multi-outcome with 0.7 <= total < 0.85 should get low-confidence warning."""
        m1 = make_single_outcome_market(market_id="mo1", question="Team Alpha wins?", yes_price=0.25)
        m2 = make_single_outcome_market(market_id="mo2", question="Team Beta wins?", yes_price=0.25)
        m3 = make_single_outcome_market(market_id="mo3", question="Team Gamma wins?", yes_price=0.25)
        event = make_event(markets=[m1, m2, m3], neg_risk=False)
        # total = 0.75, between 0.7 and 0.85
        opps = self.strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 1
        opp = opps[0]
        assert any("LOW TOTAL" in f for f in opp.risk_factors)

    # --- Filtering date-sweep / independent markets ---

    def test_date_sweep_markets_filtered(self):
        """Markets with 'by <month>' keywords should be filtered out."""
        m1 = make_single_outcome_market(market_id="ds1", question="Event by March?", yes_price=0.30)
        m2 = make_single_outcome_market(market_id="ds2", question="Event by June?", yes_price=0.30)
        m3 = make_single_outcome_market(market_id="ds3", question="Event by December?", yes_price=0.30)
        event = make_event(markets=[m1, m2, m3], neg_risk=False)
        opps = self.strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 0

    def test_independent_betting_markets_filtered(self):
        """Markets with spread/over-under keywords should be filtered out."""
        m1 = make_single_outcome_market(market_id="ib1", question="Team A -1.5 spread?", yes_price=0.30)
        m2 = make_single_outcome_market(market_id="ib2", question="Over/under 2.5 goals?", yes_price=0.30)
        m3 = make_single_outcome_market(market_id="ib3", question="Both teams to score?", yes_price=0.30)
        event = make_event(markets=[m1, m2, m3], neg_risk=False)
        opps = self.strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 0

    def test_mixed_exclusive_and_independent_skipped(self):
        """Event mixing exclusive and independent markets should be skipped."""
        m1 = make_single_outcome_market(market_id="mx1", question="Team A wins?", yes_price=0.30)
        m2 = make_single_outcome_market(market_id="mx2", question="Team B wins?", yes_price=0.30)
        m3 = make_single_outcome_market(market_id="mx3", question="Over/under 2.5?", yes_price=0.30)
        m4 = make_single_outcome_market(market_id="mx4", question="Draw?", yes_price=0.05)
        event = make_event(markets=[m1, m2, m3, m4], neg_risk=False)
        # exclusive_markets != active_markets => skip
        opps = self.strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 0

    def test_multi_outcome_requires_at_least_three_markets(self):
        """Non-negrisk events with < 3 active markets should not trigger multi-outcome."""
        m1 = make_single_outcome_market(market_id="mo1", question="Team A wins?", yes_price=0.40)
        m2 = make_single_outcome_market(market_id="mo2", question="Team B wins?", yes_price=0.40)
        event = make_event(markets=[m1, m2], neg_risk=False)
        opps = self.strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 0

    # --- Liquidity in NegRisk ---

    def test_negrisk_varying_liquidity(self):
        """Max position should reflect the lowest liquidity among active markets."""
        m1 = make_single_outcome_market(market_id="nr1", question="A?", yes_price=0.30, liquidity=2000.0)
        m2 = make_single_outcome_market(market_id="nr2", question="B?", yes_price=0.30, liquidity=500.0)
        m3 = make_single_outcome_market(market_id="nr3", question="C?", yes_price=0.30, liquidity=8000.0)
        event = make_event(markets=[m1, m2, m3], neg_risk=True)
        opps = self.strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 1
        # min liquidity = 500 => max_position = 50
        assert opps[0].max_position_size == pytest.approx(50.0)
        assert opps[0].min_liquidity == pytest.approx(500.0)

    # --- NegRisk + multi-outcome do NOT double-count ---

    def test_negrisk_event_does_not_trigger_multi_outcome(self):
        """
        A neg_risk event should only trigger _detect_negrisk_event,
        NOT _detect_multi_outcome (which guards with `if event.neg_risk: return None`).
        """
        markets = [
            make_single_outcome_market(market_id=f"nr{i}", question=f"Candidate {chr(65 + i)}?", yes_price=0.20)
            for i in range(4)
        ]
        event = make_event(markets=markets, neg_risk=True)
        # total = 0.80, should get exactly 1 opportunity from negrisk path
        opps = self.strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 1
        assert opps[0].strategy == StrategyType.NEGRISK

    def test_negrisk_below_profit_threshold_no_opportunity(self):
        """NegRisk where total is close to 1.0 and profit below threshold after fees."""
        # total = 0.96, roi = (0.98 - 0.96)/0.96 = 2.08% < 2.5%
        m1 = make_single_outcome_market(market_id="nr1", question="A?", yes_price=0.48)
        m2 = make_single_outcome_market(market_id="nr2", question="B?", yes_price=0.48)
        event = make_event(markets=[m1, m2], neg_risk=True)
        opps = self.strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 0


# ===========================================================================
# BaseStrategy tests (via concrete implementations)
# ===========================================================================


class TestBaseStrategyRiskScore:
    """Test calculate_risk_score via BasicArbStrategy (concrete implementation)."""

    def setup_method(self):
        self.strategy = BasicArbStrategy()

    # --- Liquidity risk ---

    def test_high_liquidity_low_risk(self):
        """Liquidity > $5000 should add 0 to risk score for liquidity."""
        markets = [make_market(liquidity=10000.0)]
        score, factors = self.strategy.calculate_risk_score(markets)
        # No liquidity factor should be present
        assert not any("liquidity" in f.lower() for f in factors)
        # score should be 0 (no other risk factors with no resolution date)
        assert score == pytest.approx(0.0)

    def test_moderate_liquidity_medium_risk(self):
        """Liquidity $1000-$5000 should add 0.15 risk."""
        markets = [make_market(liquidity=3000.0)]
        score, factors = self.strategy.calculate_risk_score(markets)
        assert score == pytest.approx(0.15)
        assert any("Moderate liquidity" in f for f in factors)

    def test_low_liquidity_high_risk(self):
        """Liquidity < $1000 should add 0.3 risk."""
        markets = [make_market(liquidity=500.0)]
        score, factors = self.strategy.calculate_risk_score(markets)
        assert score == pytest.approx(0.3)
        assert any("Low liquidity" in f for f in factors)

    def test_multiple_markets_uses_min_liquidity(self):
        """Risk is based on the minimum liquidity across all markets."""
        markets = [
            make_market(market_id="m1", liquidity=10000.0),
            make_market(market_id="m2", liquidity=800.0),
        ]
        score, factors = self.strategy.calculate_risk_score(markets)
        assert score == pytest.approx(0.3)
        assert any("Low liquidity" in f for f in factors)

    # --- Number of markets (complexity risk) ---

    def test_many_markets_high_complexity_risk(self):
        """More than 5 markets should add 0.2 complexity risk."""
        markets = [make_market(market_id=f"m{i}") for i in range(6)]
        score, factors = self.strategy.calculate_risk_score(markets)
        assert score == pytest.approx(0.2)
        assert any("Complex trade" in f for f in factors)

    def test_moderate_markets_moderate_complexity_risk(self):
        """4-5 markets should add 0.1 complexity risk."""
        markets = [make_market(market_id=f"m{i}") for i in range(4)]
        score, factors = self.strategy.calculate_risk_score(markets)
        assert score == pytest.approx(0.1)
        assert any("Multiple positions" in f for f in factors)

    def test_few_markets_no_complexity_risk(self):
        """3 or fewer markets should NOT add complexity risk."""
        markets = [make_market(market_id=f"m{i}") for i in range(3)]
        score, factors = self.strategy.calculate_risk_score(markets)
        assert score == pytest.approx(0.0)
        assert not any("Complex" in f or "Multiple positions" in f for f in factors)

    # --- Time to resolution risk ---

    def test_very_short_resolution_high_risk(self):
        """Resolution < 2 days should add 0.4 risk."""
        markets = [make_market()]
        resolution = datetime.now(timezone.utc) + timedelta(days=1)
        score, factors = self.strategy.calculate_risk_score(markets, resolution)
        assert score == pytest.approx(0.4)
        assert any("<2 days" in f for f in factors)

    def test_short_resolution_medium_risk(self):
        """Resolution 2-7 days should add 0.2 risk."""
        markets = [make_market()]
        resolution = datetime.now(timezone.utc) + timedelta(days=5)
        score, factors = self.strategy.calculate_risk_score(markets, resolution)
        assert score == pytest.approx(0.2)
        assert any("<7 days" in f for f in factors)

    def test_long_resolution_no_time_risk(self):
        """Resolution > 7 days should add no time-based risk."""
        markets = [make_market()]
        resolution = datetime.now(timezone.utc) + timedelta(days=30)
        score, factors = self.strategy.calculate_risk_score(markets, resolution)
        assert score == pytest.approx(0.0)
        assert not any("time to resolution" in f.lower() for f in factors)

    def test_no_resolution_date_no_time_risk(self):
        """No resolution date provided should add no time-based risk."""
        markets = [make_market()]
        score, factors = self.strategy.calculate_risk_score(markets, None)
        assert score == pytest.approx(0.0)

    # --- Combined risks ---

    def test_combined_risks_additive(self):
        """Multiple risk factors should add up."""
        # Low liquidity (0.3) + many markets (0.2) + short resolution (0.4)
        markets = [make_market(market_id=f"m{i}", liquidity=500.0) for i in range(6)]
        resolution = datetime.now(timezone.utc) + timedelta(days=1)
        score, factors = self.strategy.calculate_risk_score(markets, resolution)
        assert score == pytest.approx(0.9)
        assert len(factors) == 3

    def test_risk_score_capped_at_one(self):
        """Risk score should never exceed 1.0."""
        # Low liquidity (0.3) + many markets (0.2) + very short resolution (0.4)
        # = 0.9. Could imagine an extreme scenario with rounding.
        # Ensure the min(score, 1.0) clamp works.
        markets = [make_market(market_id=f"m{i}", liquidity=100.0) for i in range(8)]
        resolution = datetime.now(timezone.utc) + timedelta(hours=6)
        score, factors = self.strategy.calculate_risk_score(markets, resolution)
        assert score <= 1.0

    # --- Naive resolution date (timezone handling) ---

    def test_naive_resolution_date_treated_as_utc(self):
        """A naive (no tzinfo) resolution date should be treated as UTC."""
        markets = [make_market()]
        # Create a naive datetime 3 days from now
        naive_future = utcnow() + timedelta(days=3)
        score, factors = self.strategy.calculate_risk_score(markets, naive_future)
        # 3 days is in the <7 range => 0.2
        assert score == pytest.approx(0.2)
        assert any("<7 days" in f for f in factors)


class TestBaseStrategyCreateOpportunity:
    """Test create_opportunity via BasicArbStrategy."""

    def setup_method(self):
        self.strategy = BasicArbStrategy()

    def test_returns_none_when_roi_below_threshold(self):
        """create_opportunity should return None if ROI < 2.5%."""
        market = make_market(yes_price=0.48, no_price=0.48, liquidity=10000.0)
        # total = 0.96, roi = 2.08%
        result = self.strategy.create_opportunity(
            title="Test",
            description="Test opp",
            total_cost=0.96,
            markets=[market],
            positions=[],
        )
        assert result is None

    def test_returns_opportunity_when_roi_above_threshold(self):
        """create_opportunity should return ArbitrageOpportunity when profitable."""
        market = make_market(yes_price=0.40, no_price=0.40, liquidity=10000.0)
        result = self.strategy.create_opportunity(
            title="Test Opp",
            description="Good opportunity",
            total_cost=0.80,
            markets=[market],
            positions=[{"action": "BUY", "outcome": "YES", "price": 0.40}],
        )
        assert result is not None
        assert isinstance(result, ArbitrageOpportunity)
        assert result.total_cost == pytest.approx(0.80)
        assert result.expected_payout == pytest.approx(1.0)
        assert result.gross_profit == pytest.approx(0.20)
        assert result.fee == pytest.approx(0.02)
        assert result.net_profit == pytest.approx(0.18)
        assert result.roi_percent == pytest.approx(22.5)
        assert result.max_position_size == pytest.approx(1000.0)

    def test_directional_roi_cap_blocks_lottery_style_false_positive(self):
        """Directional opportunities with implausibly huge ROI should be rejected."""
        market = make_market(yes_price=0.05, no_price=0.95, liquidity=10000.0)
        result = self.strategy.create_opportunity(
            title="Directional Tail",
            description="Unrealistic payout modeling case",
            total_cost=0.05,
            markets=[market],
            positions=[{"action": "BUY", "outcome": "YES", "price": 0.05}],
            is_guaranteed=False,
        )
        assert result is None

    def test_directional_with_realistic_target_price_is_allowed(self):
        """Directional opportunity should pass when expected payout is realistic."""
        market = make_market(yes_price=0.45, no_price=0.55, liquidity=10000.0)
        result = self.strategy.create_opportunity(
            title="Directional Repricing",
            description="Expected short-term repricing edge",
            total_cost=0.45,
            expected_payout=0.55,
            markets=[market],
            positions=[{"action": "BUY", "outcome": "YES", "price": 0.45}],
            is_guaranteed=False,
        )
        assert result is not None
        assert result.expected_payout == pytest.approx(0.55)
        assert result.roi_percent > 0
        assert result.roi_percent < 120

    def test_returns_none_for_zero_total_cost(self):
        """Zero total cost => roi = 0 => below threshold => None."""
        market = make_market(liquidity=10000.0)
        result = self.strategy.create_opportunity(
            title="Zero",
            description="",
            total_cost=0.0,
            markets=[market],
            positions=[],
        )
        assert result is None

    def test_opportunity_includes_event_info(self):
        """When event is provided, opportunity should have event_id and event_title."""
        market = make_market(yes_price=0.40, no_price=0.40, liquidity=10000.0)
        event = make_event(event_id="ev42", title="Big Election")
        result = self.strategy.create_opportunity(
            title="Test",
            description="Test",
            total_cost=0.80,
            markets=[market],
            positions=[],
            event=event,
        )
        assert result is not None
        assert result.event_id == "ev42"
        assert result.event_title == "Big Election"
        assert result.category == "politics"

    def test_opportunity_resolution_date_from_market(self):
        """Resolution date should come from the first market's end_date."""
        end = datetime(2026, 6, 1, tzinfo=timezone.utc)
        market = make_market(yes_price=0.40, no_price=0.40, liquidity=10000.0, end_date=end)
        result = self.strategy.create_opportunity(
            title="Test",
            description="",
            total_cost=0.80,
            markets=[market],
            positions=[],
        )
        assert result is not None
        assert result.resolution_date == end

    def test_opportunity_markets_list_structure(self):
        """The markets list in the opportunity should contain expected keys."""
        market = make_market(
            market_id="mkt_1",
            question="Will it rain?",
            yes_price=0.40,
            no_price=0.40,
            liquidity=10000.0,
        )
        result = self.strategy.create_opportunity(
            title="Test",
            description="",
            total_cost=0.80,
            markets=[market],
            positions=[],
        )
        assert result is not None
        assert len(result.markets) == 1
        m = result.markets[0]
        assert m["id"] == "mkt_1"
        assert m["question"] == "Will it rain?"
        assert m["yes_price"] == pytest.approx(0.40)
        assert m["no_price"] == pytest.approx(0.40)
        assert m["liquidity"] == pytest.approx(10000.0)

    def test_opportunity_risk_score_in_valid_range(self):
        """Risk score on created opportunity should be between 0 and 1."""
        market = make_market(yes_price=0.40, no_price=0.40, liquidity=10000.0)
        result = self.strategy.create_opportunity(
            title="Test",
            description="",
            total_cost=0.80,
            markets=[market],
            positions=[],
        )
        assert result is not None
        assert 0.0 <= result.risk_score <= 1.0
