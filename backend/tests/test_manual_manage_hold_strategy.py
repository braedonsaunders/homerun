"""Tests for ManualManageHoldStrategy exit management behavior."""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.strategies.manual_manage_hold import ManualManageHoldStrategy


class _Position:
    pass


def _make_position(
    *,
    entry_price: float = 0.50,
    current_price: float = 0.50,
    highest_price: float | None = None,
    lowest_price: float | None = None,
    age_minutes: float = 10.0,
    strategy_context: dict | None = None,
    config: dict | None = None,
) -> _Position:
    pos = _Position()
    pos.entry_price = entry_price
    pos.current_price = current_price
    pos.highest_price = highest_price if highest_price is not None else current_price
    pos.lowest_price = lowest_price if lowest_price is not None else entry_price
    pos.age_minutes = age_minutes
    pos.pnl_percent = ((current_price - entry_price) / entry_price * 100.0) if entry_price > 0 else 0.0
    pos.filled_size = 100.0
    pos.notional_usd = 50.0
    pos.strategy_context = dict(strategy_context or {})
    pos.config = dict(config or {})
    pos.outcome_idx = 0
    return pos


def _market_state(current_price: float, **overrides) -> dict:
    state = {
        "current_price": current_price,
        "market_tradable": True,
        "is_resolved": False,
        "winning_outcome": None,
        "seconds_left": None,
        "end_time": None,
    }
    state.update(overrides)
    return state


def test_evaluate_blocks_new_entries():
    strategy = ManualManageHoldStrategy()
    decision = strategy.evaluate(None, {})
    assert decision.decision == "blocked"
    assert "blocks new entries" in decision.reason.lower()


def test_backside_peak_exit_after_confirmed_drawdown():
    strategy = ManualManageHoldStrategy()
    position = _make_position(entry_price=0.50, current_price=0.56, highest_price=0.56, age_minutes=12.0)

    first = strategy.should_exit(position, _market_state(0.56))
    second = strategy.should_exit(position, _market_state(0.55))
    third = strategy.should_exit(position, _market_state(0.548))

    assert first.action == "hold"
    assert second.action == "hold"
    assert third.action == "close"
    assert "backside peak exit" in third.reason.lower()


def test_breakeven_protection_after_profit_arm():
    strategy = ManualManageHoldStrategy()
    position = _make_position(entry_price=0.50, current_price=0.53, highest_price=0.53, age_minutes=20.0)

    armed = strategy.should_exit(position, _market_state(0.53))
    protected = strategy.should_exit(position, _market_state(0.5004))

    assert armed.action == "hold"
    assert bool(position.strategy_context.get("_manual_breakeven_armed", False))
    assert protected.action == "close"
    assert "breakeven protect" in protected.reason.lower()


def test_hard_stop_loss_closes_loser():
    strategy = ManualManageHoldStrategy()
    position = _make_position(entry_price=0.50, current_price=0.39, age_minutes=8.0)

    decision = strategy.should_exit(position, _market_state(0.39))

    assert decision.action == "close"
    assert "hard stop loss" in decision.reason.lower()


def test_near_resolution_holds_profitable_position():
    strategy = ManualManageHoldStrategy()
    position = _make_position(entry_price=0.50, current_price=0.54, age_minutes=14.0)

    decision = strategy.should_exit(position, _market_state(0.54, seconds_left=120.0))

    assert decision.action == "hold"
    assert "near-resolution hold" in decision.reason.lower()


def test_neutral_recycle_closes_old_flat_position():
    strategy = ManualManageHoldStrategy()
    position = _make_position(entry_price=0.50, current_price=0.501, age_minutes=180.0)

    decision = strategy.should_exit(position, _market_state(0.501))

    assert decision.action == "close"
    assert "neutral recycle" in decision.reason.lower()
