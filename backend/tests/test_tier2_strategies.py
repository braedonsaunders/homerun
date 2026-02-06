"""
Comprehensive tests for Tier 2 (heuristic pattern matching) strategies.

Covers:
- MutuallyExclusiveStrategy
- ContradictionStrategy
- MustHappenStrategy
- MiracleStrategy
- SettlementLagStrategy
"""

import sys

sys.path.insert(0, "/home/user/homerun/backend")

import pytest
from datetime import datetime, timedelta, timezone

from models.market import Market, Event, Token
from models.opportunity import StrategyType, MispricingType
from services.strategies.mutually_exclusive import MutuallyExclusiveStrategy
from services.strategies.contradiction import ContradictionStrategy
from services.strategies.must_happen import MustHappenStrategy
from services.strategies.miracle import (
    MiracleStrategy,
    MIRACLE_KEYWORDS,
    IMPOSSIBILITY_PHRASES,
)
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
# MutuallyExclusiveStrategy Tests
# ===========================================================================


class TestMutuallyExclusiveStrategy:
    """Tests for MutuallyExclusiveStrategy (Strategy 2)."""

    @pytest.fixture
    def strategy(self):
        return MutuallyExclusiveStrategy()

    # --- Profitable exclusive pair (political) ---

    def test_detects_democrat_republican_exclusive_pair(self, strategy):
        """Two mutually exclusive political markets with sum < 1.0 should produce an opportunity."""
        m1 = _make_market(
            id="dem1",
            question="Will the Democrat candidate win the 2028 presidential election?",
            yes_price=0.45,
            no_price=0.55,
        )
        m2 = _make_market(
            id="rep1",
            question="Will the Republican candidate win the 2028 presidential election?",
            yes_price=0.48,
            no_price=0.52,
        )
        event = _make_event(
            id="e_pol", title="2028 Presidential Election", markets=[m1, m2]
        )
        opps = strategy.detect(events=[event], markets=[m1, m2], prices={})

        assert len(opps) >= 1
        opp = opps[0]
        assert opp.strategy == StrategyType.MUTUALLY_EXCLUSIVE
        assert opp.total_cost == pytest.approx(0.93, abs=0.01)
        assert opp.net_profit > 0
        assert opp.roi_percent > 0
        # Should carry manual verification risk factor
        assert any("MANUAL VERIFICATION" in rf for rf in opp.risk_factors)
        # Should carry political risk factor
        assert any("Independent" in rf or "Political" in rf for rf in opp.risk_factors)

    # --- Sum >= 1.0 (no opportunity) ---

    def test_no_opportunity_when_sum_gte_one(self, strategy):
        """When YES prices sum to >= 1.0 no opportunity should be returned."""
        m1 = _make_market(
            id="dem2",
            question="Will the Democrat candidate win the 2028 presidential election?",
            yes_price=0.52,
            no_price=0.48,
        )
        m2 = _make_market(
            id="rep2",
            question="Will the Republican candidate win the 2028 presidential election?",
            yes_price=0.50,
            no_price=0.50,
        )
        event = _make_event(
            id="e_pol2", title="2028 Presidential Election", markets=[m1, m2]
        )
        opps = strategy.detect(events=[event], markets=[m1, m2], prices={})
        assert len(opps) == 0

    # --- Win/Lose exclusive keywords ---

    def test_detects_win_lose_exclusive_pair(self, strategy):
        """Markets with win/lose keywords on the same topic should match."""
        m1 = _make_market(
            id="wl1",
            question="Will Team Alpha win the championship 2028 finals?",
            yes_price=0.44,
            no_price=0.56,
        )
        m2 = _make_market(
            id="wl2",
            question="Will Team Alpha lose the championship 2028 finals?",
            yes_price=0.46,
            no_price=0.54,
        )
        event = _make_event(
            id="e_wl", title="Championship 2028 Finals", markets=[m1, m2]
        )
        opps = strategy.detect(events=[event], markets=[m1, m2], prices={})

        assert len(opps) >= 1
        opp = opps[0]
        assert opp.total_cost == pytest.approx(0.90, abs=0.01)
        # Win/lose risk factor should be present
        assert any("Draw" in rf or "tie" in rf for rf in opp.risk_factors)

    # --- Unrelated markets (no match) ---

    def test_unrelated_markets_no_match(self, strategy):
        """Markets on completely different topics should not be paired."""
        m1 = _make_market(
            id="unrel1",
            question="Will Bitcoin reach 200k by December?",
            yes_price=0.45,
            no_price=0.55,
        )
        m2 = _make_market(
            id="unrel2",
            question="Will the Dodgers win the World Series?",
            yes_price=0.40,
            no_price=0.60,
        )
        event = _make_event(id="e_unrel", title="Unrelated", markets=[m1, m2])
        opps = strategy.detect(events=[event], markets=[m1, m2], prices={})
        assert len(opps) == 0

    # --- Closed markets should be skipped (in-event path) ---

    def test_closed_markets_skipped_in_event(self, strategy):
        """Closed markets must not produce in-event opportunities."""
        m1 = _make_market(
            id="cl1",
            question="Will the Democrat candidate win the 2028 presidential election?",
            yes_price=0.45,
            no_price=0.55,
            closed=True,
        )
        m2 = _make_market(
            id="cl2",
            question="Will the Republican candidate win the 2028 presidential election?",
            yes_price=0.48,
            no_price=0.52,
            closed=True,
        )
        event = _make_event(
            id="e_cl", title="2028 Presidential Election", markets=[m1, m2]
        )
        # Only test the in-event path (empty markets list avoids cross-market detection)
        opps = strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 0

    def test_inactive_markets_skipped_in_event(self, strategy):
        """Inactive markets must not produce in-event opportunities."""
        m1 = _make_market(
            id="inact1",
            question="Will the Democrat candidate win the 2028 presidential election?",
            yes_price=0.45,
            no_price=0.55,
            active=False,
        )
        m2 = _make_market(
            id="inact2",
            question="Will the Republican candidate win the 2028 presidential election?",
            yes_price=0.48,
            no_price=0.52,
            active=False,
        )
        event = _make_event(
            id="e_inact", title="2028 Presidential Election", markets=[m1, m2]
        )
        opps = strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 0

    # --- Only two-market events ---

    def test_more_than_two_markets_skipped_in_event(self, strategy):
        """Events with more than 2 active markets should not produce in-event pairs."""
        m1 = _make_market(
            id="3a", question="Will the Democrat win the 2028 race?", yes_price=0.30
        )
        m2 = _make_market(
            id="3b", question="Will the Republican win the 2028 race?", yes_price=0.35
        )
        m3 = _make_market(
            id="3c", question="Will the Independent win the 2028 race?", yes_price=0.20
        )
        event = _make_event(id="e3", title="2028 Race", markets=[m1, m2, m3])
        # pass empty market list so cross-market check doesn't fire
        opps = strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 0

    # --- Live price override ---

    def test_live_prices_used_when_available(self, strategy):
        """Prices from the prices dict should override static market prices."""
        m1 = _make_market(
            id="lp1",
            question="Will the Democrat candidate win the 2028 presidential election?",
            yes_price=0.52,
            no_price=0.48,
        )
        m2 = _make_market(
            id="lp2",
            question="Will the Republican candidate win the 2028 presidential election?",
            yes_price=0.52,
            no_price=0.48,
        )
        # Static prices sum to 1.04, no opp.  But live prices bring them down.
        live_prices = {
            "lp1_yes": {"mid": 0.44},
            "lp2_yes": {"mid": 0.46},
        }
        event = _make_event(
            id="e_lp", title="2028 Presidential Election", markets=[m1, m2]
        )
        opps = strategy.detect(events=[event], markets=[m1, m2], prices=live_prices)
        assert len(opps) >= 1

    # --- Cross-market detection ---

    def test_cross_market_detection(self, strategy):
        """Exclusive pairs should be detected across markets (not just within events)."""
        m1 = _make_market(
            id="xm1",
            question="Will the Democrat candidate win the 2028 general election?",
            yes_price=0.44,
            no_price=0.56,
            volume=100000,
        )
        m2 = _make_market(
            id="xm2",
            question="Will the Republican candidate win the 2028 general election?",
            yes_price=0.46,
            no_price=0.54,
            volume=100000,
        )
        opps = strategy.detect(events=[], markets=[m1, m2], prices={})
        assert len(opps) >= 1

    # --- Total cost below 0.85 => skipped (not exhaustive) ---

    def test_very_low_total_skipped(self, strategy):
        """If total cost is very low (<0.85) the pair is probably not exhaustive."""
        m1 = _make_market(
            id="low1",
            question="Will the Democrat candidate win the 2028 presidential election?",
            yes_price=0.30,
            no_price=0.70,
        )
        m2 = _make_market(
            id="low2",
            question="Will the Republican candidate win the 2028 presidential election?",
            yes_price=0.30,
            no_price=0.70,
        )
        event = _make_event(
            id="e_low", title="2028 Presidential Election", markets=[m1, m2]
        )
        opps = strategy.detect(events=[event], markets=[m1, m2], prices={})
        assert len(opps) == 0


# ===========================================================================
# ContradictionStrategy Tests
# ===========================================================================


class TestContradictionStrategy:
    """Tests for ContradictionStrategy (Strategy 3)."""

    @pytest.fixture
    def strategy(self):
        return ContradictionStrategy()

    # --- Profitable contradiction (before/after) ---

    def test_detects_before_after_contradiction(self, strategy):
        """Markets with before/after keywords sharing a topic should be detected."""
        m1 = _make_market(
            id="ba1",
            question="Will Congress pass the healthcare reform bill before summer recess?",
            yes_price=0.40,
            no_price=0.60,
        )
        m2 = _make_market(
            id="ba2",
            question="Will Congress pass the healthcare reform bill after summer recess?",
            yes_price=0.45,
            no_price=0.55,
        )
        opps = strategy.detect(events=[], markets=[m1, m2], prices={})

        assert len(opps) >= 1
        opp = opps[0]
        assert opp.strategy == StrategyType.CONTRADICTION
        assert opp.total_cost < 1.0
        assert opp.net_profit > 0
        assert any("MANUAL VERIFICATION" in rf for rf in opp.risk_factors)

    # --- Win/Lose contradiction ---

    def test_detects_win_lose_contradiction(self, strategy):
        """Markets with win vs lose on same topic should be detected."""
        m1 = _make_market(
            id="wl_c1",
            question="Will Team Bravo win the championship finals 2028?",
            yes_price=0.40,
            no_price=0.60,
        )
        m2 = _make_market(
            id="wl_c2",
            question="Will Team Bravo lose the championship finals 2028?",
            yes_price=0.45,
            no_price=0.55,
        )
        opps = strategy.detect(events=[], markets=[m1, m2], prices={})
        assert len(opps) >= 1

    # --- Pass/Fail contradiction ---

    def test_detects_pass_fail_contradiction(self, strategy):
        """Markets with pass vs fail on same topic should be detected."""
        m1 = _make_market(
            id="pf1",
            question="Will the Senate committee pass the environmental regulation?",
            yes_price=0.42,
            no_price=0.58,
        )
        m2 = _make_market(
            id="pf2",
            question="Will the Senate committee fail the environmental regulation?",
            yes_price=0.43,
            no_price=0.57,
        )
        opps = strategy.detect(events=[], markets=[m1, m2], prices={})
        assert len(opps) >= 1

    # --- Sum >= 1.0 (no opportunity) ---

    def test_no_opportunity_when_sum_gte_one(self, strategy):
        """If YES+YES >= 1.0 and YES+NO >= 1.0, no opportunity exists."""
        m1 = _make_market(
            id="nopr1",
            question="Will Congress pass the healthcare reform bill before summer recess?",
            yes_price=0.55,
            no_price=0.45,
        )
        m2 = _make_market(
            id="nopr2",
            question="Will Congress pass the healthcare reform bill after summer recess?",
            yes_price=0.55,
            no_price=0.55,
        )
        opps = strategy.detect(events=[], markets=[m1, m2], prices={})
        assert len(opps) == 0

    # --- Non-contradictory markets ---

    def test_non_contradictory_markets_no_match(self, strategy):
        """Markets that don't contain contradiction keywords should not match."""
        m1 = _make_market(
            id="nc1",
            question="Will Bitcoin reach 100k by December?",
            yes_price=0.40,
            no_price=0.60,
        )
        m2 = _make_market(
            id="nc2",
            question="Will Ethereum reach 10k by December?",
            yes_price=0.35,
            no_price=0.65,
        )
        opps = strategy.detect(events=[], markets=[m1, m2], prices={})
        assert len(opps) == 0

    # --- Closed / inactive markets should be skipped ---

    def test_closed_markets_skipped(self, strategy):
        """Closed markets should be excluded from analysis."""
        m1 = _make_market(
            id="ccl1",
            question="Will Congress pass the healthcare reform bill before summer recess?",
            yes_price=0.40,
            no_price=0.60,
            closed=True,
        )
        m2 = _make_market(
            id="ccl2",
            question="Will Congress pass the healthcare reform bill after summer recess?",
            yes_price=0.45,
            no_price=0.55,
        )
        opps = strategy.detect(events=[], markets=[m1, m2], prices={})
        assert len(opps) == 0

    # --- Partial keyword match (word boundary) ---

    def test_partial_keyword_in_longer_word(self, strategy):
        """Keywords embedded in longer words (e.g. 'winner' contains 'win') should still match if pattern is substring."""
        m1 = _make_market(
            id="pk1",
            question="Will the overall winner of the championship tournament 2028 prevail?",
            yes_price=0.43,
            no_price=0.57,
        )
        m2 = _make_market(
            id="pk2",
            question="Will the overall loser of the championship tournament 2028 prevail?",
            yes_price=0.43,
            no_price=0.57,
        )
        # "win" is in "winner", "lose" is in "loser" -- substring matching should pick them up
        opps = strategy.detect(events=[], markets=[m1, m2], prices={})
        assert len(opps) >= 1

    # --- Approach 2: YES+NO arbitrage ---

    def test_yes_no_approach_when_yes_yes_not_profitable(self, strategy):
        """If YES+YES >= 1.0 but YES+NO < 1.0, approach 2 should detect it."""
        m1 = _make_market(
            id="yn1",
            question="Will Congress pass the healthcare reform bill before summer recess?",
            yes_price=0.55,
            no_price=0.45,
        )
        m2 = _make_market(
            id="yn2",
            question="Will Congress pass the healthcare reform bill after summer recess?",
            yes_price=0.55,
            no_price=0.35,
        )
        # YES+YES = 1.10 (no opp), YES_a + NO_b = 0.55 + 0.35 = 0.90 (opp!)
        opps = strategy.detect(events=[], markets=[m1, m2], prices={})
        assert len(opps) >= 1
        opp = opps[0]
        # The profitable approach should be YES+NO
        assert any(p["outcome"] == "NO" for p in opp.positions_to_take)

    # --- Approve/Reject contradiction ---

    def test_detects_approve_reject_contradiction(self, strategy):
        """Markets with approve vs reject on same topic should be detected."""
        m1 = _make_market(
            id="ar1",
            question="Will the board approve the merger proposal during quarterly review?",
            yes_price=0.40,
            no_price=0.60,
        )
        m2 = _make_market(
            id="ar2",
            question="Will the board reject the merger proposal during quarterly review?",
            yes_price=0.45,
            no_price=0.55,
        )
        opps = strategy.detect(events=[], markets=[m1, m2], prices={})
        assert len(opps) >= 1

    # --- Insufficient topic overlap ---

    def test_no_shared_topic_no_match(self, strategy):
        """Even with contradiction keywords, markets need >= 2 shared topic words."""
        m1 = _make_market(
            id="nst1",
            question="Will it pass tomorrow?",
            yes_price=0.40,
            no_price=0.60,
        )
        m2 = _make_market(
            id="nst2",
            question="Did they fail yesterday?",
            yes_price=0.45,
            no_price=0.55,
        )
        opps = strategy.detect(events=[], markets=[m1, m2], prices={})
        assert len(opps) == 0


# ===========================================================================
# MustHappenStrategy Tests
# ===========================================================================


class TestMustHappenStrategy:
    """Tests for MustHappenStrategy (Strategy 5)."""

    @pytest.fixture
    def strategy(self):
        return MustHappenStrategy()

    # --- Profitable must-happen with "winner" keyword ---

    def test_detects_must_happen_with_winner_keyword(self, strategy):
        """Event whose title contains 'winner' and YES prices sum < 1.0 should produce opp."""
        m1 = _make_market(
            id="mh1",
            question="Candidate Alpha wins the 2028 primary?",
            yes_price=0.30,
            no_price=0.70,
        )
        m2 = _make_market(
            id="mh2",
            question="Candidate Beta wins the 2028 primary?",
            yes_price=0.32,
            no_price=0.68,
        )
        m3 = _make_market(
            id="mh3",
            question="Candidate Gamma wins the 2028 primary?",
            yes_price=0.30,
            no_price=0.70,
        )
        event = _make_event(
            id="e_mh",
            title="Who will be the winner of the 2028 primary?",
            markets=[m1, m2, m3],
        )
        opps = strategy.detect(events=[event], markets=[m1, m2, m3], prices={})

        assert len(opps) == 1
        opp = opps[0]
        assert opp.strategy == StrategyType.MUST_HAPPEN
        assert opp.total_cost == pytest.approx(0.92, abs=0.01)
        assert opp.net_profit > 0
        assert any("MANUAL VERIFICATION" in rf for rf in opp.risk_factors)

    # --- "who will" keyword ---

    def test_detects_with_who_will_keyword(self, strategy):
        """Event with 'who will' in title should be recognised as potentially exhaustive."""
        m1 = _make_market(
            id="ww1", question="Player A wins MVP 2028?", yes_price=0.28, no_price=0.72
        )
        m2 = _make_market(
            id="ww2", question="Player B wins MVP 2028?", yes_price=0.30, no_price=0.70
        )
        m3 = _make_market(
            id="ww3", question="Player C wins MVP 2028?", yes_price=0.32, no_price=0.68
        )
        event = _make_event(
            id="e_ww", title="Who will win MVP 2028?", markets=[m1, m2, m3]
        )
        opps = strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 1

    # --- "elected" keyword ---

    def test_detects_with_elected_keyword(self, strategy):
        """Event with 'elected' in title should match."""
        m1 = _make_market(
            id="el1",
            question="Person A elected governor 2028?",
            yes_price=0.29,
            no_price=0.71,
        )
        m2 = _make_market(
            id="el2",
            question="Person B elected governor 2028?",
            yes_price=0.30,
            no_price=0.70,
        )
        m3 = _make_market(
            id="el3",
            question="Person C elected governor 2028?",
            yes_price=0.31,
            no_price=0.69,
        )
        event = _make_event(
            id="e_el", title="Who gets elected governor 2028?", markets=[m1, m2, m3]
        )
        opps = strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 1

    # --- Sum >= 1.0 (no opportunity) ---

    def test_no_opportunity_when_sum_gte_one(self, strategy):
        """When total YES prices >= 1.0 no must-happen opportunity should exist."""
        m1 = _make_market(
            id="mhn1",
            question="Runner A wins the 2028 marathon?",
            yes_price=0.40,
            no_price=0.60,
        )
        m2 = _make_market(
            id="mhn2",
            question="Runner B wins the 2028 marathon?",
            yes_price=0.35,
            no_price=0.65,
        )
        m3 = _make_market(
            id="mhn3",
            question="Runner C wins the 2028 marathon?",
            yes_price=0.30,
            no_price=0.70,
        )
        event = _make_event(
            id="e_mhn",
            title="Who will be the winner of the 2028 marathon?",
            markets=[m1, m2, m3],
        )
        opps = strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 0  # sum = 1.05

    # --- Total below 0.80 is filtered out (missing outcomes) ---

    def test_very_low_total_filtered_out(self, strategy):
        """If total YES is below 0.80 the strategy suspects hidden outcomes."""
        m1 = _make_market(
            id="lf1",
            question="Candidate A wins the 2028 race?",
            yes_price=0.20,
            no_price=0.80,
        )
        m2 = _make_market(
            id="lf2",
            question="Candidate B wins the 2028 race?",
            yes_price=0.20,
            no_price=0.80,
        )
        m3 = _make_market(
            id="lf3",
            question="Candidate C wins the 2028 race?",
            yes_price=0.20,
            no_price=0.80,
        )
        event = _make_event(
            id="e_lf",
            title="Who will be the winner of the 2028 race?",
            markets=[m1, m2, m3],
        )
        opps = strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 0  # sum = 0.60

    # --- NegRisk events are skipped (handled by NegRisk strategy) ---

    def test_negrisk_events_skipped(self, strategy):
        """NegRisk events should be deferred to the NegRiskStrategy."""
        m1 = _make_market(
            id="nr1", question="Team X wins?", yes_price=0.30, no_price=0.70
        )
        m2 = _make_market(
            id="nr2", question="Team Y wins?", yes_price=0.32, no_price=0.68
        )
        m3 = _make_market(
            id="nr3", question="Team Z wins?", yes_price=0.30, no_price=0.70
        )
        event = _make_event(
            id="e_nr",
            title="Who will be the winner?",
            markets=[m1, m2, m3],
            neg_risk=True,
        )
        opps = strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 0

    # --- Closed events are skipped ---

    def test_closed_events_skipped(self, strategy):
        """Closed events should not be analysed."""
        m1 = _make_market(
            id="ce1", question="Team X wins?", yes_price=0.30, no_price=0.70
        )
        m2 = _make_market(
            id="ce2", question="Team Y wins?", yes_price=0.32, no_price=0.68
        )
        m3 = _make_market(
            id="ce3", question="Team Z wins?", yes_price=0.30, no_price=0.70
        )
        event = _make_event(
            id="e_ce",
            title="Who will be the winner?",
            markets=[m1, m2, m3],
            closed=True,
        )
        opps = strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 0

    # --- Single market events skipped ---

    def test_single_market_event_skipped(self, strategy):
        """Events with fewer than 2 markets should be skipped."""
        m1 = _make_market(
            id="sm1", question="Will X happen?", yes_price=0.30, no_price=0.70
        )
        event = _make_event(id="e_sm", title="Who will be the winner?", markets=[m1])
        opps = strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 0

    # --- Non-exhaustive event title skipped ---

    def test_non_exhaustive_title_skipped(self, strategy):
        """Events without exhaustive keywords and < 3 markets should be skipped."""
        m1 = _make_market(
            id="ne1",
            question="Will the weather be sunny?",
            yes_price=0.30,
            no_price=0.70,
        )
        m2 = _make_market(
            id="ne2",
            question="Will it rain heavily today?",
            yes_price=0.32,
            no_price=0.68,
        )
        event = _make_event(
            id="e_ne", title="Weather forecast for Tuesday", markets=[m1, m2]
        )
        opps = strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 0

    # --- Live prices used ---

    def test_live_prices_override_static(self, strategy):
        """Live prices from the prices dict should be used when available."""
        m1 = _make_market(
            id="mlp1",
            question="Runner A wins the 2028 marathon?",
            yes_price=0.40,
            no_price=0.60,
        )
        m2 = _make_market(
            id="mlp2",
            question="Runner B wins the 2028 marathon?",
            yes_price=0.35,
            no_price=0.65,
        )
        m3 = _make_market(
            id="mlp3",
            question="Runner C wins the 2028 marathon?",
            yes_price=0.30,
            no_price=0.70,
        )
        # Static sum = 1.05 (no opp). Live prices bring it down.
        live_prices = {
            "mlp1_yes": {"mid": 0.30},
            "mlp2_yes": {"mid": 0.30},
            "mlp3_yes": {"mid": 0.30},
        }
        event = _make_event(
            id="e_mlp",
            title="Who will be the winner of the 2028 marathon?",
            markets=[m1, m2, m3],
        )
        opps = strategy.detect(events=[event], markets=[], prices=live_prices)
        assert len(opps) == 1

    # --- Low total risk factor ---

    def test_low_total_adds_risk_factor(self, strategy):
        """Totals between 0.80 and 0.90 should add a 'missing outcomes' risk factor."""
        m1 = _make_market(
            id="lr1",
            question="Candidate A wins the 2028 race?",
            yes_price=0.27,
            no_price=0.73,
        )
        m2 = _make_market(
            id="lr2",
            question="Candidate B wins the 2028 race?",
            yes_price=0.28,
            no_price=0.72,
        )
        m3 = _make_market(
            id="lr3",
            question="Candidate C wins the 2028 race?",
            yes_price=0.30,
            no_price=0.70,
        )
        event = _make_event(
            id="e_lr",
            title="Who will be the winner of the 2028 race?",
            markets=[m1, m2, m3],
        )
        opps = strategy.detect(events=[event], markets=[], prices={})
        # total = 0.85, above 0.80 but below 0.90
        assert len(opps) == 1
        opp = opps[0]
        assert any("missing outcomes" in rf.lower() for rf in opp.risk_factors)


# ===========================================================================
# MiracleStrategy Tests
# ===========================================================================


class TestMiracleStrategy:
    """Tests for MiracleStrategy (Strategy 6)."""

    @pytest.fixture
    def strategy(self):
        return MiracleStrategy()

    # --- Alien invasion detection ---

    def test_detects_alien_invasion_market(self, strategy):
        """Market about alien invasion should be flagged as miracle."""
        m = _make_market(
            id="alien1",
            question="Will aliens land on Earth by the end of 2028?",
            yes_price=0.03,
            no_price=0.95,
            liquidity=5000.0,
        )
        opps = strategy.detect(events=[], markets=[m], prices={})
        assert len(opps) == 1
        opp = opps[0]
        assert opp.strategy == StrategyType.MIRACLE
        assert (
            "alien" in opp.description.lower()
            or "supernatural" in opp.description.lower()
        )
        assert opp.positions_to_take[0]["outcome"] == "NO"

    # --- Supernatural event detection ---

    def test_detects_supernatural_event(self, strategy):
        """Market about supernatural events (ghosts, rapture) should be flagged."""
        m = _make_market(
            id="ghost1",
            question="Will a ghost be scientifically confirmed as paranormal evidence?",
            yes_price=0.05,
            no_price=0.93,
            liquidity=3000.0,
        )
        opps = strategy.detect(events=[], markets=[m], prices={})
        assert len(opps) == 1

    # --- Impossible physics ---

    def test_detects_time_travel(self, strategy):
        """Market about time travel should be flagged."""
        m = _make_market(
            id="tt1",
            question="Will time travel be demonstrated publicly by 2028?",
            yes_price=0.02,
            no_price=0.96,
            liquidity=5000.0,
        )
        opps = strategy.detect(events=[], markets=[m], prices={})
        assert len(opps) == 1

    # --- Apocalypse ---

    def test_detects_apocalypse_market(self, strategy):
        """Market about apocalypse should be flagged."""
        m = _make_market(
            id="apoc1",
            question="Will the apocalypse happen before 2030?",
            yes_price=0.02,
            no_price=0.96,
            liquidity=5000.0,
        )
        opps = strategy.detect(events=[], markets=[m], prices={})
        assert len(opps) == 1

    # --- Plausible events should NOT be flagged ---

    def test_plausible_event_not_flagged(self, strategy):
        """Normal markets about plausible outcomes should not be flagged."""
        m = _make_market(
            id="norm1",
            question="Will the Federal Reserve raise interest rates in March 2028?",
            yes_price=0.40,
            no_price=0.60,
            liquidity=50000.0,
        )
        opps = strategy.detect(events=[], markets=[m], prices={})
        assert len(opps) == 0

    # --- NO price too low (below threshold) ---

    def test_no_price_below_threshold_skipped(self, strategy):
        """Markets where NO price is below min_no_price should be skipped."""
        m = _make_market(
            id="lnp1",
            question="Will aliens land on Earth by 2028?",
            yes_price=0.20,
            no_price=0.80,
            liquidity=5000.0,
        )
        opps = strategy.detect(events=[], markets=[m], prices={})
        assert len(opps) == 0  # NO at 0.80 < 0.90 threshold

    # --- NO price too high (above max threshold) ---

    def test_no_price_above_max_threshold_skipped(self, strategy):
        """Markets where NO is already >= 0.995 should be skipped (not enough profit)."""
        m = _make_market(
            id="hnp1",
            question="Will aliens land on Earth by 2028?",
            yes_price=0.002,
            no_price=0.998,
            liquidity=5000.0,
        )
        opps = strategy.detect(events=[], markets=[m], prices={})
        assert len(opps) == 0

    # --- Closed markets skipped ---

    def test_closed_markets_skipped(self, strategy):
        """Closed markets should not be analysed."""
        m = _make_market(
            id="mc1",
            question="Will aliens land on Earth by 2028?",
            yes_price=0.03,
            no_price=0.95,
            closed=True,
        )
        opps = strategy.detect(events=[], markets=[m], prices={})
        assert len(opps) == 0

    # --- Impossibility score calculation ---

    def test_impossibility_score_alien(self, strategy):
        """Alien keyword should yield high impossibility score."""
        score, category, reasons = strategy.calculate_impossibility_score(
            "Will aliens invade Earth?"
        )
        assert score >= 0.90
        assert category == "supernatural"
        assert len(reasons) > 0

    def test_impossibility_score_time_travel(self, strategy):
        """Time travel keyword should yield high score."""
        score, category, reasons = strategy.calculate_impossibility_score(
            "Will time travel be proven real?"
        )
        assert score >= 0.90
        assert category == "impossible_physics"

    def test_impossibility_score_plausible(self, strategy):
        """Normal question should yield low score."""
        score, category, reasons = strategy.calculate_impossibility_score(
            "Will the Federal Reserve raise interest rates?"
        )
        assert score < 0.70

    def test_impossibility_score_boosted_by_phrase(self, strategy):
        """Phrases like 'by tomorrow' should boost the score."""
        base_score, _, _ = strategy.calculate_impossibility_score("Will aliens land?")
        boosted_score, _, _ = strategy.calculate_impossibility_score(
            "Will aliens land on earth by tomorrow?"
        )
        assert boosted_score > base_score

    def test_impossibility_score_past_year_boost(self, strategy):
        """Reference to a past year should add impossibility."""
        score, _, reasons = strategy.calculate_impossibility_score(
            "Will aliens have visited in 2023?"
        )
        assert any("past year" in r for r in reasons)

    def test_impossibility_score_capped_at_one(self, strategy):
        """Score should never exceed 1.0 even with many triggers."""
        score, _, _ = strategy.calculate_impossibility_score(
            "Will aliens perform time travel and rapture by tomorrow within 1 hour in 2023?"
        )
        assert score <= 1.0

    # --- Keyword coverage: verify 40+ patterns exist ---

    def test_miracle_keywords_count(self):
        """There should be at least 40 keyword patterns."""
        total = len(MIRACLE_KEYWORDS) + len(IMPOSSIBILITY_PHRASES)
        assert total >= 40, f"Only {total} keyword patterns found, expected 40+"

    # --- Sorted by ROI ---

    def test_opportunities_sorted_by_roi(self, strategy):
        """Opportunities should be returned sorted by ROI descending."""
        m1 = _make_market(
            id="sr1",
            question="Will aliens invade Earth by 2028?",
            yes_price=0.05,
            no_price=0.93,
            liquidity=5000.0,
        )
        m2 = _make_market(
            id="sr2",
            question="Will time travel be demonstrated in public?",
            yes_price=0.03,
            no_price=0.95,
            liquidity=5000.0,
        )
        opps = strategy.detect(events=[], markets=[m1, m2], prices={})
        assert len(opps) == 2
        assert opps[0].roi_percent >= opps[1].roi_percent

    # --- Non-binary markets skipped ---

    def test_non_binary_market_skipped(self, strategy):
        """Markets without exactly 2 outcome prices should be skipped."""
        m = _make_market(
            id="nb1",
            question="Will aliens invade Earth?",
            yes_price=0.05,
            no_price=0.93,
        )
        m.outcome_prices = [0.05, 0.45, 0.50]  # 3 outcomes
        opps = strategy.detect(events=[], markets=[m], prices={})
        assert len(opps) == 0

    # --- Live prices override ---

    def test_live_prices_used(self, strategy):
        """Live prices should override static market prices."""
        m = _make_market(
            id="mlpr1",
            question="Will the rapture occur by 2028?",
            yes_price=0.05,
            no_price=0.80,  # Below threshold statically
            liquidity=5000.0,
        )
        live_prices = {
            "mlpr1_no": {"mid": 0.93},
            "mlpr1_yes": {"mid": 0.03},
        }
        opps = strategy.detect(events=[], markets=[m], prices=live_prices)
        assert len(opps) == 1

    # --- Stale markets method ---

    def test_find_stale_markets_returns_empty(self, strategy):
        """find_stale_markets is a placeholder that should return empty list."""
        m = _make_market(id="stale1", question="Test", yes_price=0.50, no_price=0.50)
        result = strategy.find_stale_markets([m], ["some_event"])
        assert result == []

    # --- Various miracle categories ---

    def test_nuclear_war_category(self, strategy):
        score, category, reasons = strategy.calculate_impossibility_score(
            "Will nuclear war start in 2028?"
        )
        assert score >= 0.70
        assert category == "apocalypse"

    def test_perpetual_motion_category(self, strategy):
        score, category, reasons = strategy.calculate_impossibility_score(
            "Will perpetual motion machine be built?"
        )
        assert score >= 0.90

    def test_rapture_category(self, strategy):
        score, category, reasons = strategy.calculate_impossibility_score(
            "Will the rapture happen this year?"
        )
        assert score >= 0.90
        assert category == "supernatural"

    def test_ww3_category(self, strategy):
        score, category, reasons = strategy.calculate_impossibility_score(
            "Will ww3 break out by Friday?"
        )
        assert score >= 0.80

    def test_asteroid_impact_category(self, strategy):
        score, category, reasons = strategy.calculate_impossibility_score(
            "Will an asteroid impact destroy a major city?"
        )
        assert score >= 0.85

    def test_debunked_keyword(self, strategy):
        score, _, reasons = strategy.calculate_impossibility_score(
            "Will the moon landing hoax theory be debunked again?"
        )
        # Both "hoax" and "debunked" should fire
        assert score >= 0.80


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
        assert opp.strategy == StrategyType.SETTLEMENT_LAG
        assert opp.mispricing_type == MispricingType.SETTLEMENT_LAG
        assert (
            "past resolution date" in opp.description.lower()
            or "past resolution date" in str(opp.description)
        )
        assert opp.total_cost == pytest.approx(0.60, abs=0.01)

    # --- Appears resolved (YES near zero) ---

    def test_detects_near_zero_yes_as_resolved(self, strategy):
        """If YES price < 0.05, market appears resolved to NO."""
        m = _make_market(
            id="sl2",
            question="Will candidate Z win the election?",
            yes_price=0.02,
            no_price=0.90,
            liquidity=10000.0,
        )
        opps = strategy.detect(events=[], markets=[m], prices={})
        assert len(opps) == 1
        opp = opps[0]
        assert opp.total_cost == pytest.approx(0.92, abs=0.01)

    # --- Appears resolved (NO near zero) ---

    def test_detects_near_zero_no_as_resolved(self, strategy):
        """If NO price < 0.05, market appears resolved to YES."""
        m = _make_market(
            id="sl3",
            question="Will the bill pass the final vote?",
            yes_price=0.88,
            no_price=0.03,
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
        m1 = _make_market(
            id="nrs1", question="Candidate A wins?", yes_price=0.02, no_price=0.98
        )
        m2 = _make_market(
            id="nrs2", question="Candidate B wins?", yes_price=0.85, no_price=0.15
        )
        m3 = _make_market(
            id="nrs3", question="Candidate C wins?", yes_price=0.02, no_price=0.98
        )
        event = _make_event(
            id="e_nrs", title="Who wins?", markets=[m1, m2, m3], neg_risk=True
        )
        opps = strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 1
        opp = opps[0]
        assert opp.mispricing_type == MispricingType.SETTLEMENT_LAG
        assert opp.total_cost == pytest.approx(0.89, abs=0.01)

    # --- NegRisk no deviation ---

    def test_negrisk_no_deviation_no_opportunity(self, strategy):
        """NegRisk event with YES prices summing to ~1.0 should not flag."""
        m1 = _make_market(
            id="nrn1", question="Option A?", yes_price=0.33, no_price=0.67
        )
        m2 = _make_market(
            id="nrn2", question="Option B?", yes_price=0.34, no_price=0.66
        )
        m3 = _make_market(
            id="nrn3", question="Option C?", yes_price=0.33, no_price=0.67
        )
        event = _make_event(
            id="e_nrn", title="Pick one", markets=[m1, m2, m3], neg_risk=True
        )
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
        event = _make_event(
            id="e_nrcl", title="Who wins?", markets=[m1, m2], neg_risk=True, closed=True
        )
        opps = strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 0

    # --- NegRisk sum > 1.0 (no opportunity) ---

    def test_negrisk_sum_gt_one_no_opportunity(self, strategy):
        """NegRisk event with YES sum > 1.0 should not flag."""
        m1 = _make_market(id="nrgt1", question="X wins?", yes_price=0.50, no_price=0.50)
        m2 = _make_market(id="nrgt2", question="Y wins?", yes_price=0.60, no_price=0.40)
        event = _make_event(
            id="e_nrgt", title="Who wins?", markets=[m1, m2], neg_risk=True
        )
        opps = strategy.detect(events=[event], markets=[], prices={})
        assert len(opps) == 0


# ===========================================================================
# BaseStrategy integration tests
# ===========================================================================


class TestBaseStrategyIntegration:
    """Cross-cutting concerns tested through concrete strategies."""

    def test_risk_score_low_liquidity(self):
        """Low liquidity markets should raise the risk score."""
        strategy = MutuallyExclusiveStrategy()
        m1 = _make_market(
            id="rl1",
            question="Will the Democrat candidate win the 2028 general election?",
            yes_price=0.44,
            no_price=0.56,
            liquidity=500.0,
        )
        m2 = _make_market(
            id="rl2",
            question="Will the Republican candidate win the 2028 general election?",
            yes_price=0.46,
            no_price=0.54,
            liquidity=500.0,
        )
        event = _make_event(id="e_rl", title="2028 General Election", markets=[m1, m2])
        opps = strategy.detect(events=[event], markets=[m1, m2], prices={})
        assert len(opps) >= 1
        opp = opps[0]
        assert opp.risk_score > 0
        assert any("liquidity" in rf.lower() for rf in opp.risk_factors)

    def test_risk_score_short_time_to_resolution(self):
        """Markets resolving soon should have higher risk score."""
        strategy = MutuallyExclusiveStrategy()
        soon = datetime.now(timezone.utc) + timedelta(days=1)
        m1 = _make_market(
            id="rt1",
            question="Will the Democrat candidate win the 2028 general election?",
            yes_price=0.44,
            no_price=0.56,
            end_date=soon,
        )
        m2 = _make_market(
            id="rt2",
            question="Will the Republican candidate win the 2028 general election?",
            yes_price=0.46,
            no_price=0.54,
            end_date=soon,
        )
        event = _make_event(id="e_rt", title="2028 General Election", markets=[m1, m2])
        opps = strategy.detect(events=[event], markets=[m1, m2], prices={})
        assert len(opps) >= 1
        opp = opps[0]
        assert any(
            "short time" in rf.lower() or "very short" in rf.lower()
            for rf in opp.risk_factors
        )

    def test_max_position_size_based_on_liquidity(self):
        """max_position_size should be 10% of min liquidity."""
        strategy = MutuallyExclusiveStrategy()
        m1 = _make_market(
            id="mps1",
            question="Will the Democrat candidate win the 2028 general election?",
            yes_price=0.44,
            no_price=0.56,
            liquidity=20000.0,
        )
        m2 = _make_market(
            id="mps2",
            question="Will the Republican candidate win the 2028 general election?",
            yes_price=0.46,
            no_price=0.54,
            liquidity=10000.0,
        )
        event = _make_event(id="e_mps", title="2028 General Election", markets=[m1, m2])
        opps = strategy.detect(events=[event], markets=[m1, m2], prices={})
        assert len(opps) >= 1
        opp = opps[0]
        assert opp.max_position_size == pytest.approx(1000.0, abs=1.0)  # 10% of 10000

    def test_min_profit_threshold_filtering(self):
        """Opportunities below the min profit threshold should be filtered out."""
        strategy = MutuallyExclusiveStrategy()
        # With fee=0.02 and min_profit=0.025, ROI must exceed 2.5%
        # total_cost=0.97 => gross=0.03, fee=0.02, net=0.01, roi=1.03% => below threshold
        m1 = _make_market(
            id="mpf1",
            question="Will the Democrat candidate win the 2028 general election?",
            yes_price=0.485,
            no_price=0.515,
        )
        m2 = _make_market(
            id="mpf2",
            question="Will the Republican candidate win the 2028 general election?",
            yes_price=0.485,
            no_price=0.515,
        )
        event = _make_event(id="e_mpf", title="2028 General Election", markets=[m1, m2])
        opps = strategy.detect(events=[event], markets=[m1, m2], prices={})
        assert len(opps) == 0
