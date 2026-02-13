"""Quality guard tests for directional/statistical strategies."""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import timedelta

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.market import Market, Event, Token
from services.strategies.temporal_decay import TemporalDecayStrategy
from services.strategies.event_driven import EventDrivenStrategy
from services.strategies.market_making import MarketMakingStrategy
from services.strategies.liquidity_vacuum import LiquidityVacuumStrategy
from utils.utcnow import utcnow


def _make_market(
    *,
    market_id: str,
    question: str,
    yes_price: float,
    no_price: float,
    liquidity: float = 10000.0,
    volume: float = 10000.0,
    end_in_days: float = 14.0,
) -> Market:
    clob_ids = [f"{market_id}_yes", f"{market_id}_no"]
    return Market(
        id=market_id,
        condition_id=f"cond_{market_id}",
        question=question,
        slug=f"slug-{market_id}",
        tokens=[
            Token(token_id=clob_ids[0], outcome="Yes", price=yes_price),
            Token(token_id=clob_ids[1], outcome="No", price=no_price),
        ],
        clob_token_ids=clob_ids,
        outcome_prices=[yes_price, no_price],
        active=True,
        closed=False,
        liquidity=liquidity,
        volume=volume,
        end_date=utcnow() + timedelta(days=end_in_days),
    )


def test_temporal_decay_skips_multileg_sports_prefix():
    strategy = TemporalDecayStrategy()
    market = _make_market(
        market_id="KXMVESPORTSMULTIGAMEEXTENDED-S2026ABC-123",
        question="yes Team A,yes Team B,yes Team C",
        yes_price=0.06,
        no_price=0.94,
        liquidity=6000.0,
    )
    opps = strategy.detect(events=[], markets=[market], prices={})
    assert opps == []


def test_temporal_decay_uses_repricing_target_not_binary_settlement():
    strategy = TemporalDecayStrategy()
    market = _make_market(
        market_id="td_simple_1",
        question="Will Bitcoin be above $120,000 by March 2026?",
        yes_price=0.70,
        no_price=0.30,
        liquidity=20000.0,
    )
    yes_token, no_token = market.clob_token_ids

    # Build sufficient price history with a meaningful deviation.
    for yes in [0.70, 0.69, 0.67, 0.63]:
        strategy.detect(
            events=[],
            markets=[market],
            prices={yes_token: {"mid": yes}, no_token: {"mid": 1.0 - yes}},
        )

    opps = strategy.detect(
        events=[],
        markets=[market],
        prices={yes_token: {"mid": 0.58}, no_token: {"mid": 0.42}},
    )

    assert len(opps) >= 1
    opp = opps[0]
    assert opp.is_guaranteed is False
    assert opp.expected_payout < 1.0
    assert opp.roi_percent < 120.0


def test_temporal_decay_certainty_shock_detects_yes_surge():
    strategy = TemporalDecayStrategy()
    market = _make_market(
        market_id="td_shock_yes_up",
        question="Will bill X pass by Friday?",
        yes_price=0.12,
        no_price=0.88,
        liquidity=25000.0,
        end_in_days=2.0,
    )
    yes_token, no_token = market.clob_token_ids

    for yes in [0.12, 0.18, 0.30, 0.55]:
        strategy.detect(
            events=[],
            markets=[market],
            prices={yes_token: {"mid": yes}, no_token: {"mid": 1.0 - yes}},
        )

    opps = strategy.detect(
        events=[],
        markets=[market],
        prices={yes_token: {"mid": 0.78}, no_token: {"mid": 0.22}},
    )

    assert len(opps) >= 1
    opp = opps[0]
    assert "Certainty Shock" in opp.title
    assert opp.is_guaranteed is False
    assert opp.positions_to_take[0]["outcome"] == "YES"
    assert opp.expected_payout < 1.0


def test_temporal_decay_certainty_shock_detects_no_surge_after_yes_crash():
    strategy = TemporalDecayStrategy()
    market = _make_market(
        market_id="td_shock_yes_down",
        question="Will government shutdown happen by Saturday?",
        yes_price=0.90,
        no_price=0.10,
        liquidity=25000.0,
        end_in_days=1.5,
    )
    yes_token, no_token = market.clob_token_ids

    for yes in [0.90, 0.88, 0.78, 0.62]:
        strategy.detect(
            events=[],
            markets=[market],
            prices={yes_token: {"mid": yes}, no_token: {"mid": 1.0 - yes}},
        )

    opps = strategy.detect(
        events=[],
        markets=[market],
        prices={yes_token: {"mid": 0.35}, no_token: {"mid": 0.65}},
    )

    assert len(opps) >= 1
    opp = opps[0]
    assert "Certainty Shock" in opp.title
    assert opp.is_guaranteed is False
    assert opp.positions_to_take[0]["outcome"] == "NO"
    assert opp.expected_payout < 1.0


def test_event_driven_deduplicates_equivalent_market_questions():
    strategy = EventDrivenStrategy()

    catalyst = _make_market(
        market_id="cat1",
        question="Yale at Dartmouth: Total Points",
        yes_price=0.40,
        no_price=0.60,
        liquidity=80000.0,
    )
    lag1 = _make_market(
        market_id="lag1",
        question="Manhattan at Niagara: Total Points",
        yes_price=0.50,
        no_price=0.50,
        liquidity=79000.0,
    )
    lag2 = _make_market(
        market_id="lag2",
        question="Manhattan at Niagara: Total Points",
        yes_price=0.50,
        no_price=0.50,
        liquidity=81000.0,
    )
    event = Event(
        id="e1",
        slug="e1",
        title="NCAA Totals",
        description="",
        category="sports",
        markets=[catalyst, lag1, lag2],
        active=True,
        closed=False,
    )

    # Scan 1: baseline
    strategy.detect(
        events=[event],
        markets=[catalyst, lag1, lag2],
        prices={
            catalyst.clob_token_ids[0]: {"mid": 0.40},
            lag1.clob_token_ids[0]: {"mid": 0.50},
            lag2.clob_token_ids[0]: {"mid": 0.50},
        },
    )

    # Scan 2: catalyst jumps, lagging markets stay flat.
    opps = strategy.detect(
        events=[event],
        markets=[catalyst, lag1, lag2],
        prices={
            catalyst.clob_token_ids[0]: {"mid": 0.55},
            lag1.clob_token_ids[0]: {"mid": 0.50},
            lag2.clob_token_ids[0]: {"mid": 0.50},
        },
    )

    assert len(opps) == 1
    opp = opps[0]
    assert opp.is_guaranteed is False
    assert opp.expected_payout < 1.0
    assert "Manhattan at Niagara: Total Points" in opp.title


def test_market_making_skips_multileg_contracts():
    strategy = MarketMakingStrategy()
    market = _make_market(
        market_id="KXMVESPORTSMULTIGAMEEXTENDED-S2026XYZ-321",
        question="yes Austin Reaves: 2+,yes LeBron James: 4+,yes Team X",
        yes_price=0.47,
        no_price=0.47,
        liquidity=25000.0,
        volume=50000.0,
    )
    opps = strategy.detect(events=[], markets=[market], prices={})
    assert opps == []


def test_liquidity_vacuum_marks_directional_and_uses_target_payout():
    strategy = LiquidityVacuumStrategy()
    market = _make_market(
        market_id="vac1",
        question="Will BTC close above 100k today?",
        yes_price=0.45,
        no_price=0.55,
        liquidity=30000.0,
        volume=60000.0,
    )
    yes_token, no_token = market.clob_token_ids
    event = Event(
        id="ev1",
        slug="ev1",
        title="BTC Daily",
        description="",
        category="crypto",
        markets=[market],
        active=True,
        closed=False,
    )
    opps = strategy.detect(
        events=[event],
        markets=[market],
        prices={
            yes_token: {"mid": 0.45, "bid_depth": 5000.0, "ask_depth": 500.0},
            no_token: {"mid": 0.55},
        },
    )

    assert len(opps) >= 1
    opp = opps[0]
    assert opp.is_guaranteed is False
    assert opp.expected_payout < 1.0
