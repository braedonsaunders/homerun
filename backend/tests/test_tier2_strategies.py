"""Tests for settlement-lag opportunity detection strategy."""

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import pytest
from datetime import datetime, timedelta, timezone

from models.market import Market, Event, Token
from models.opportunity import MispricingType
from services.strategies.settlement_lag import SettlementLagStrategy


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_market(
    id: str = "m1",
    question: str = "Will something happen?",
    yes_price: float = 0.50,
    no_price: float = 0.50,
    active: bool = True,
    closed: bool = False,
    liquidity: float = 10000.0,
    volume: float = 50000.0,
    end_date: datetime = None,
    neg_risk: bool = False,
    clob_token_ids: list[str] = None,
) -> Market:
    """Create a Market with sensible defaults for testing."""
    if clob_token_ids is None:
        clob_token_ids = [f"{id}_yes", f"{id}_no"]
    outcome_prices = [yes_price, no_price]
    tokens = [
        Token(token_id=clob_token_ids[0], outcome="Yes", price=yes_price),
        Token(token_id=clob_token_ids[1], outcome="No", price=no_price),
    ]
    return Market(
        id=id,
        condition_id=f"cond_{id}",
        question=question,
        slug=f"slug-{id}",
        tokens=tokens,
        clob_token_ids=clob_token_ids,
        outcome_prices=outcome_prices,
        active=active,
        closed=closed,
        neg_risk=neg_risk,
        volume=volume,
        liquidity=liquidity,
        end_date=end_date,
    )


def _make_event(
    id: str = "e1",
    title: str = "Test Event",
    markets: list[Market] = None,
    neg_risk: bool = False,
    active: bool = True,
    closed: bool = False,
    category: str = None,
) -> Event:
    """Create an Event with sensible defaults for testing."""
    return Event(
        id=id,
        slug=f"slug-{id}",
        title=title,
        description="Test event description",
        category=category,
        markets=markets or [],
        neg_risk=neg_risk,
        active=active,
        closed=closed,
    )


# ===========================================================================
# SettlementLagStrategy Tests
# ===========================================================================


class TestSettlementLagStrategy:
    """Tests for SettlementLagStrategy (Strategy 8)."""

    @pytest.fixture
    def strategy(self):
        return SettlementLagStrategy()

    # --- Overdue market with deviation ---

    def test_detects_overdue_market_with_deviation(self, strategy):
        """Market past resolution date with price sum deviation should be detected."""
        past_date = datetime.now(timezone.utc) - timedelta(hours=6)
        m = _make_market(
            id="sl1",
            question="Will Assad remain president through 2024?",
            yes_price=0.30,
            no_price=0.30,
            end_date=past_date,
            liquidity=10000.0,
        )
        opps = strategy.detect(events=[], markets=[m], prices={})
        assert len(opps) == 1
        opp = opps[0]
        assert opp.strategy == "settlement_lag"
        assert opp.mispricing_type == MispricingType.SETTLEMENT_LAG
        assert "past resolution date" in opp.description.lower() or "past resolution date" in str(opp.description)
        assert opp.total_cost == pytest.approx(0.60, abs=0.01)

    # --- Appears resolved (YES near zero) ---

    def test_detects_near_zero_yes_as_resolved(self, strategy):
        """If YES price < 0.02 (NEAR_ZERO_THRESHOLD), market appears resolved to NO."""
        m = _make_market(
            id="sl2",
            question="Will candidate Z win the election?",
            yes_price=0.01,
            no_price=0.92,
            liquidity=10000.0,
        )
        opps = strategy.detect(events=[], markets=[m], prices={})
        assert len(opps) == 1
        opp = opps[0]
        assert opp.total_cost == pytest.approx(0.93, abs=0.01)

    # --- Appears resolved (NO near zero) ---

    def test_detects_near_zero_no_as_resolved(self, strategy):
        """If NO price < 0.02 (NEAR_ZERO_THRESHOLD), market appears resolved to YES."""
        m = _make_market(
            id="sl3",
            question="Will the bill pass the final vote?",
            yes_price=0.92,
            no_price=0.01,
            liquidity=10000.0,
        )
        opps = strategy.detect(events=[], markets=[m], prices={})
        assert len(opps) == 1

    # --- No deviation (prices correct) ---

    def test_no_deviation_no_opportunity(self, strategy):
        """Markets with prices summing close to 1.0 should not be flagged."""
        m = _make_market(
            id="sl4",
            question="Will it rain tomorrow?",
            yes_price=0.50,
            no_price=0.50,
            liquidity=10000.0,
        )
        opps = strategy.detect(events=[], markets=[m], prices={})
        assert len(opps) == 0  # sum = 1.0, no deviation

    # --- Sum >= 1.0 (no arbitrage even if overdue) ---

    def test_sum_gte_one_no_opportunity(self, strategy):
        """Even if overdue, sum >= 1.0 means no arbitrage."""
        past_date = datetime.now(timezone.utc) - timedelta(hours=6)
        m = _make_market(
            id="sl5",
            question="Will something happen?",
            yes_price=0.55,
            no_price=0.50,
            end_date=past_date,
            liquidity=10000.0,
        )
        opps = strategy.detect(events=[], markets=[m], prices={})
        assert len(opps) == 0

    # --- Closed markets skipped ---

    def test_closed_markets_skipped(self, strategy):
        """Closed markets should not be analysed."""
        past_date = datetime.now(timezone.utc) - timedelta(hours=6)
        m = _make_market(
            id="sl6",
            question="Will something happen?",
            yes_price=0.30,
            no_price=0.30,
            end_date=past_date,
            closed=True,
        )
        opps = strategy.detect(events=[], markets=[m], prices={})
        assert len(opps) == 0

    # --- NegRisk settlement lag ---

    def test_negrisk_settlement_lag(self, strategy):
        """NegRisk events with YES sum deviation should be detected."""
        m1 = _make_market(id="nrs1", question="Candidate A wins?", yes_price=0.02, no_price=0.98)
        m2 = _make_market(id="nrs2", question="Candidate B wins?", yes_price=0.85, no_price=0.15)
        m3 = _make_market(id="nrs3", question="Candidate C wins?", yes_price=0.02, no_price=0.98)
        event = _make_event(id="e_nrs", title="Who wins?", markets=[m1, m2, m3], neg_risk=True)
        opps = strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 1
        opp = opps[0]
        assert opp.mispricing_type == MispricingType.SETTLEMENT_LAG
        assert opp.total_cost == pytest.approx(0.89, abs=0.01)

    # --- NegRisk no deviation ---

    def test_negrisk_no_deviation_no_opportunity(self, strategy):
        """NegRisk event with YES prices summing to ~1.0 should not flag."""
        m1 = _make_market(id="nrn1", question="Option A?", yes_price=0.33, no_price=0.67)
        m2 = _make_market(id="nrn2", question="Option B?", yes_price=0.34, no_price=0.66)
        m3 = _make_market(id="nrn3", question="Option C?", yes_price=0.33, no_price=0.67)
        event = _make_event(id="e_nrn", title="Pick one", markets=[m1, m2, m3], neg_risk=True)
        opps = strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 0  # sum = 1.0

    # --- Non-binary markets skipped ---

    def test_non_binary_market_skipped(self, strategy):
        """Markets without exactly 2 outcome prices are skipped."""
        past_date = datetime.now(timezone.utc) - timedelta(hours=6)
        m = _make_market(
            id="slnb1",
            question="Will something happen?",
            yes_price=0.30,
            no_price=0.30,
            end_date=past_date,
        )
        m.outcome_prices = [0.30]  # Only 1 price
        opps = strategy.detect(events=[], markets=[m], prices={})
        assert len(opps) == 0

    # --- Live prices used ---

    def test_live_prices_used(self, strategy):
        """Live prices should override static prices."""
        past_date = datetime.now(timezone.utc) - timedelta(hours=6)
        m = _make_market(
            id="sllp1",
            question="Will something happen soon?",
            yes_price=0.50,
            no_price=0.50,  # Static sum = 1.0
            end_date=past_date,
            liquidity=10000.0,
        )
        live_prices = {
            "sllp1_yes": {"mid": 0.30},
            "sllp1_no": {"mid": 0.30},
        }
        opps = strategy.detect(events=[], markets=[m], prices=live_prices)
        assert len(opps) == 1

    # --- Timing window: recent resolution ---

    def test_recently_resolved_market(self, strategy):
        """Market that just passed its end date should be detected."""
        just_passed = datetime.now(timezone.utc) - timedelta(minutes=30)
        m = _make_market(
            id="slr1",
            question="Will the announcement happen before midnight?",
            yes_price=0.25,
            no_price=0.35,
            end_date=just_passed,
            liquidity=10000.0,
        )
        opps = strategy.detect(events=[], markets=[m], prices={})
        assert len(opps) == 1

    # --- Stale price data (near-zero on one side) ---

    def test_stale_price_both_sides_low(self, strategy):
        """Both YES and NO being low (sum << 1) suggests stale/lagging prices."""
        past_date = datetime.now(timezone.utc) - timedelta(days=1)
        m = _make_market(
            id="stl1",
            question="Will the event occur on schedule?",
            yes_price=0.15,
            no_price=0.20,
            end_date=past_date,
            liquidity=10000.0,
        )
        opps = strategy.detect(events=[], markets=[m], prices={})
        assert len(opps) == 1
        opp = opps[0]
        assert opp.total_cost == pytest.approx(0.35, abs=0.01)

    # --- NegRisk with closed event skipped ---

    def test_negrisk_closed_event_skipped(self, strategy):
        """Closed NegRisk events should be skipped."""
        m1 = _make_market(id="nrcl1", question="A wins?", yes_price=0.02, no_price=0.98)
        m2 = _make_market(id="nrcl2", question="B wins?", yes_price=0.85, no_price=0.15)
        event = _make_event(id="e_nrcl", title="Who wins?", markets=[m1, m2], neg_risk=True, closed=True)
        opps = strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 0

    # --- NegRisk sum > 1.0 (no opportunity) ---

    def test_negrisk_sum_gt_one_no_opportunity(self, strategy):
        """NegRisk event with YES sum > 1.0 should not flag."""
        m1 = _make_market(id="nrgt1", question="X wins?", yes_price=0.50, no_price=0.50)
        m2 = _make_market(id="nrgt2", question="Y wins?", yes_price=0.60, no_price=0.40)
        event = _make_event(id="e_nrgt", title="Who wins?", markets=[m1, m2], neg_risk=True)
        opps = strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 0
