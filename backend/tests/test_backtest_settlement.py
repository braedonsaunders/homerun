"""$1/$0 binary settlement for held positions.

A position held into a market's resolution settles at the outcome
(winning token -> $1.00, losing -> $0.00) at resolution time, NOT at the
last observed order-book mid. Markets that don't resolve in-window keep
the honest mark-to-mid behavior; markets resolved-but-winner-unknown
surface ``is_resolved`` to the strategy instead of auto-redeeming.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import pytest

from services.backtest import (
    BacktestConfig,
    BacktestEngine,
    BookSnapshot,
    InMemoryBookReplay,
    LatencyModel,
    PortfolioConfig,
    PriceLevel,
    TradeIntent,
)
from services.backtest.matching_engine import FeeModel
from services.backtest.settlement import TokenSettlement
from services.backtest.venue_model import TIF_IOC
from services.strategies.base import BaseStrategy, ExitDecision


T0 = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
RESOLVE_AT = T0 + timedelta(seconds=20)


def _snap(t, *, bid=0.49, ask=0.50, depth=5, qty=50):
    return BookSnapshot(
        token_id="tok",
        observed_at=t,
        bids=tuple(PriceLevel(round(bid - 0.01 * k, 4), qty) for k in range(depth)),
        asks=tuple(PriceLevel(round(ask + 0.01 * k, 4), qty) for k in range(depth)),
    )


def _flat_book(n=30):
    return InMemoryBookReplay([_snap(T0 + timedelta(seconds=i)) for i in range(n)])


class _HoldForever(BaseStrategy):
    strategy_type = "hold_test"
    name = "hold"
    description = "never exits — rides to resolution"

    def should_exit(self, position, market_state):
        return ExitDecision("hold", "wait")


class _ExitWhenResolved(BaseStrategy):
    strategy_type = "resolved_test"
    name = "resolved"
    description = "closes once the market is flagged resolved"

    def should_exit(self, position, market_state):
        if market_state.get("is_resolved"):
            # Marketable SELL (<= best bid 0.49) above the $1 min-notional.
            return ExitDecision("close", "resolved", close_price=0.40)
        return ExitDecision("hold", "wait")


def _buy_intent():
    return TradeIntent(
        intent_id="i1", emitted_at=T0, token_id="tok", side="BUY",
        size=20, limit_price=0.51, tif=TIF_IOC, post_only=False,
        strategy_slug="hold_test",
    )


def _engine(settlements, *, strategy=None):
    return BacktestEngine(
        config=BacktestConfig(
            portfolio=PortfolioConfig(initial_capital_usd=1000.0),
            latency=LatencyModel.deterministic(submit_ms=100, cancel_ms=50),
            fees=FeeModel(per_fill_gas_usd=0.0, resolution_fee_rate=0.0, use_taker_fee_curve=False),
            settlements=settlements,
            seed=42,
        ),
        strategy=strategy or _HoldForever(),
    )


@pytest.mark.asyncio
async def test_winning_position_settles_at_one_dollar():
    settlements = {
        "tok": TokenSettlement(
            token_id="tok", settle_price=1.0, resolution_time=RESOLVE_AT,
            winning_outcome="Up", source="test",
        )
    }
    result = await _engine(settlements).run(
        book_source=_flat_book(), trade_intents=[_buy_intent()]
    )
    # Entry 20 @ $0.50 = $10 cost; redeem 20 @ $1.00 = $20 -> +$10.
    assert result.final_equity_usd == pytest.approx(1010.0, abs=0.6)
    assert result.notes["settlement"]["settled_positions"] == 1
    assert result.notes["settlement"]["marked_to_mid_positions"] == 0


@pytest.mark.asyncio
async def test_losing_position_settles_at_zero():
    settlements = {
        "tok": TokenSettlement(
            token_id="tok", settle_price=0.0, resolution_time=RESOLVE_AT,
            winning_outcome="Down", source="test",
        )
    }
    result = await _engine(settlements).run(
        book_source=_flat_book(), trade_intents=[_buy_intent()]
    )
    # Entry $10 cost; redeem 20 @ $0.00 = $0 -> -$10.
    assert result.final_equity_usd == pytest.approx(990.0, abs=0.6)
    assert result.notes["settlement"]["settled_positions"] == 1


@pytest.mark.asyncio
async def test_unresolved_market_marks_to_mid_not_settled():
    # No settlement data -> legacy honest mark-to-mid (mid = 0.495), which
    # is clearly distinct from a $1/$0 settlement.
    result = await _engine({}).run(
        book_source=_flat_book(), trade_intents=[_buy_intent()]
    )
    assert 999.0 < result.final_equity_usd < 1000.5  # ~999.9, NOT 1010 / 990
    assert result.notes["settlement"]["settled_positions"] == 0
    assert result.notes["settlement"]["marked_to_mid_positions"] == 1


@pytest.mark.asyncio
async def test_resolved_winner_unknown_surfaces_is_resolved_to_strategy():
    # resolution_time known, winner unknown (settle_price=None): the engine
    # must NOT auto-redeem; it surfaces is_resolved so the strategy can act.
    settlements = {
        "tok": TokenSettlement(
            token_id="tok", settle_price=None, resolution_time=RESOLVE_AT, source="test",
        )
    }
    result = await _engine(settlements, strategy=_ExitWhenResolved()).run(
        book_source=_flat_book(), trade_intents=[_buy_intent()]
    )
    assert result.notes["settlement"]["settled_positions"] == 0
    assert result.notes["settlement"]["marked_to_mid_positions"] == 0
    assert result.closed_position_count == 1
