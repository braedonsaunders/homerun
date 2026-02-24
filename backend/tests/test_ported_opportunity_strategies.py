"""Regression tests for newly ported opportunity filters."""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.market import Event, Market, Token
from services.strategies.flash_crash_reversion import FlashCrashReversionStrategy
from services.strategies.tail_end_carry import TailEndCarryStrategy
from utils.utcnow import utcnow


def _market(
    *,
    market_id: str,
    question: str,
    yes_price: float,
    no_price: float,
    liquidity: float = 20000.0,
    volume: float = 30000.0,
    end_in_days: float = 7.0,
) -> Market:
    tokens = [f"{market_id}_yes", f"{market_id}_no"]
    return Market(
        id=market_id,
        condition_id=f"cond_{market_id}",
        question=question,
        slug=f"slug-{market_id}",
        tokens=[
            Token(token_id=tokens[0], outcome="Yes", price=yes_price),
            Token(token_id=tokens[1], outcome="No", price=no_price),
        ],
        clob_token_ids=tokens,
        outcome_prices=[yes_price, no_price],
        active=True,
        closed=False,
        liquidity=liquidity,
        volume=volume,
        end_date=utcnow() + timedelta(days=end_in_days),
    )


def _event_for(market: Market) -> Event:
    return Event(
        id=f"event-{market.id}",
        slug=f"event-{market.slug}",
        title=market.question,
        description="",
        category="politics",
        markets=[market],
        active=True,
        closed=False,
    )


def test_flash_crash_reversion_detects_short_window_yes_crash() -> None:
    strategy = FlashCrashReversionStrategy()
    market = _market(
        market_id="flash_yes_1",
        question="Will Candidate A win the race?",
        yes_price=0.62,
        no_price=0.38,
        end_in_days=4.0,
    )
    event = _event_for(market)
    yes_token, no_token = market.clob_token_ids

    baseline_prices = {
        yes_token: {"mid": 0.62, "bid": 0.615, "ask": 0.625},
        no_token: {"mid": 0.38, "bid": 0.375, "ask": 0.385},
    }
    crash_prices = {
        yes_token: {"mid": 0.50, "bid": 0.495, "ask": 0.505},
        no_token: {"mid": 0.50, "bid": 0.495, "ask": 0.505},
    }

    first = strategy.detect(events=[event], markets=[market], prices=baseline_prices)
    assert first == []

    second = strategy.detect(events=[event], markets=[market], prices=crash_prices)
    assert len(second) >= 1
    opp = second[0]
    assert opp.strategy == "flash_crash_reversion"
    assert opp.is_guaranteed is False
    assert (opp.positions_to_take or [{}])[0].get("outcome") == "YES"
    assert opp.expected_payout > opp.total_cost


def test_tail_end_carry_emits_near_expiry_high_probability_entry() -> None:
    strategy = TailEndCarryStrategy()
    market = _market(
        market_id="tail_1",
        question="Will Team X make playoffs?",
        yes_price=0.93,
        no_price=0.07,
        end_in_days=0.7,
        liquidity=30000.0,
    )
    event = _event_for(market)
    yes_token, no_token = market.clob_token_ids

    prices = {
        yes_token: {"mid": 0.93, "bid": 0.925, "ask": 0.935},
        no_token: {"mid": 0.07, "bid": 0.065, "ask": 0.075},
    }

    opps = strategy.detect(events=[event], markets=[market], prices=prices)
    assert len(opps) >= 1
    opp = opps[0]
    assert opp.strategy == "tail_end_carry"
    assert opp.is_guaranteed is False
    assert opp.expected_payout < 1.0
    assert opp.expected_payout > opp.total_cost
    raw_roi = ((opp.expected_payout - opp.total_cost) / opp.total_cost) * 100.0
    expected_fee = opp.fee
    expected_roi = ((opp.expected_payout - opp.total_cost - expected_fee) / opp.total_cost) * 100.0
    assert opp.roi_percent < raw_roi
    assert opp.roi_percent == pytest.approx(expected_roi, rel=1e-9)
