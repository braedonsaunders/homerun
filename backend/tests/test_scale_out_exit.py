"""Tests for BaseStrategy scale-out / partial exit logic."""

from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.strategies.base import (
    BaseStrategy,
    ExitDecision,
    ScaleOutConfig,
    ScaleOutTarget,
)


class _ScaleOutStrategy(BaseStrategy):
    strategy_type = "test_scale_out"
    name = "Test Scale Out"
    description = "Strategy with scale-out config for testing"

    scale_out_config = ScaleOutConfig(
        targets=[
            ScaleOutTarget(trigger_bps=5, exit_fraction=0.33),
            ScaleOutTarget(trigger_bps=10, exit_fraction=0.5),
        ],
        trailing_stop_bps=80,
        near_resolution_exit=True,
        near_resolution_hours=24.0,
        near_resolution_spread_widen_bps=50.0,
    )


class _Position:
    pass


def _make_position(
    entry_price: float = 0.50,
    current_price: float = 0.50,
    highest_price: float | None = None,
    age_minutes: float = 30.0,
    config: dict | None = None,
    strategy_context: dict | None = None,
) -> _Position:
    pos = _Position()
    pos.entry_price = entry_price
    pos.current_price = current_price
    pos.highest_price = highest_price if highest_price is not None else current_price
    pos.lowest_price = entry_price
    pos.age_minutes = age_minutes
    pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
    pos.pnl_percent = pnl_pct
    pos.strategy_context = dict(strategy_context or {})
    pos.config = dict(config or {})
    pos.outcome_idx = 0
    return pos


MARKET_STATE = {
    "current_price": 0.50,
    "market_tradable": True,
    "is_resolved": False,
    "winning_outcome": None,
}


def _market_state(current_price: float, **kwargs) -> dict:
    ms = dict(MARKET_STATE)
    ms["current_price"] = current_price
    ms.update(kwargs)
    return ms


class TestScaleOutTargets:
    def test_first_target_hit(self):
        strategy = _ScaleOutStrategy()
        pos = _make_position(entry_price=0.50, current_price=0.525)  # +5%

        decision = strategy.default_exit_check(pos, _market_state(0.525))

        assert decision.action == "reduce"
        assert decision.reduce_fraction == 0.33
        assert "Scale-out target 1" in decision.reason
        assert pos.strategy_context["_scale_out_targets_hit"] == [0]

    def test_second_target_hit_after_first(self):
        strategy = _ScaleOutStrategy()
        pos = _make_position(
            entry_price=0.50,
            current_price=0.55,  # +10%
            strategy_context={"_scale_out_targets_hit": [0]},
        )

        decision = strategy.default_exit_check(pos, _market_state(0.55))

        assert decision.action == "reduce"
        assert decision.reduce_fraction == 0.5
        assert "Scale-out target 2" in decision.reason
        assert pos.strategy_context["_scale_out_targets_hit"] == [0, 1]

    def test_no_target_hit_below_threshold(self):
        strategy = _ScaleOutStrategy()
        pos = _make_position(entry_price=0.50, current_price=0.52)  # +4% < 5% target

        decision = strategy.default_exit_check(pos, _market_state(0.52))

        assert decision.action == "hold"
        assert "_scale_out_targets_hit" not in pos.strategy_context or pos.strategy_context["_scale_out_targets_hit"] == []

    def test_already_hit_target_not_retriggered(self):
        strategy = _ScaleOutStrategy()
        pos = _make_position(
            entry_price=0.50,
            current_price=0.525,  # +5% (target 1 threshold)
            strategy_context={"_scale_out_targets_hit": [0]},
        )

        decision = strategy.default_exit_check(pos, _market_state(0.525))

        assert decision.action != "reduce" or "target 1" not in decision.reason

    def test_both_targets_hit_simultaneously(self):
        strategy = _ScaleOutStrategy()
        pos = _make_position(entry_price=0.50, current_price=0.55)  # +10%

        decision = strategy.default_exit_check(pos, _market_state(0.55))

        assert decision.action == "reduce"
        assert decision.reduce_fraction == 0.33
        assert "Scale-out target 1" in decision.reason
        assert pos.strategy_context["_scale_out_targets_hit"] == [0]

    def test_strategy_context_initialized_when_missing(self):
        strategy = _ScaleOutStrategy()
        pos = _make_position(entry_price=0.50, current_price=0.525)
        pos.strategy_context = None

        decision = strategy.default_exit_check(pos, _market_state(0.525))

        assert decision.action == "reduce"
        assert isinstance(pos.strategy_context, dict)
        assert pos.strategy_context["_scale_out_targets_hit"] == [0]


class TestScaleOutTrailingStop:
    def test_trailing_stop_after_all_targets_hit(self):
        strategy = _ScaleOutStrategy()
        pos = _make_position(
            entry_price=0.50,
            current_price=0.548,
            highest_price=0.55,
            strategy_context={
                "_scale_out_targets_hit": [0, 1],
                "_scale_out_peak_price": 0.55,
            },
        )

        decision = strategy.default_exit_check(pos, _market_state(0.548))

        # trailing_stop_bps=80 means 0.8% pullback from peak
        # trigger = 0.55 * (1 - 80/10000) = 0.55 * 0.992 = 0.5456
        # 0.548 > 0.5456, so no trigger yet
        assert decision.action == "hold"

    def test_trailing_stop_triggers_on_pullback(self):
        strategy = _ScaleOutStrategy()
        pos = _make_position(
            entry_price=0.50,
            current_price=0.545,
            highest_price=0.55,
            strategy_context={
                "_scale_out_targets_hit": [0, 1],
                "_scale_out_peak_price": 0.55,
            },
        )

        decision = strategy.default_exit_check(pos, _market_state(0.545))

        # trigger = 0.55 * (1 - 80/10000) = 0.55 * 0.992 = 0.5456
        # 0.545 <= 0.5456, so triggers
        assert decision.action == "close"
        assert "Scale-out trailing stop" in decision.reason

    def test_peak_price_tracked(self):
        strategy = _ScaleOutStrategy()
        pos = _make_position(
            entry_price=0.50,
            current_price=0.56,
            strategy_context={"_scale_out_targets_hit": [0, 1]},
        )

        strategy.default_exit_check(pos, _market_state(0.56))

        assert pos.strategy_context["_scale_out_peak_price"] == 0.56

    def test_peak_price_updated_when_higher(self):
        strategy = _ScaleOutStrategy()
        pos = _make_position(
            entry_price=0.50,
            current_price=0.57,
            strategy_context={
                "_scale_out_targets_hit": [0, 1],
                "_scale_out_peak_price": 0.55,
            },
        )

        strategy.default_exit_check(pos, _market_state(0.57))

        assert pos.strategy_context["_scale_out_peak_price"] == 0.57

    def test_peak_price_not_lowered(self):
        strategy = _ScaleOutStrategy()
        pos = _make_position(
            entry_price=0.50,
            current_price=0.54,
            strategy_context={
                "_scale_out_targets_hit": [0, 1],
                "_scale_out_peak_price": 0.55,
            },
        )

        strategy.default_exit_check(pos, _market_state(0.54))

        assert pos.strategy_context["_scale_out_peak_price"] == 0.55


class TestScaleOutNearResolution:
    def test_near_resolution_exit_triggers(self):
        strategy = _ScaleOutStrategy()
        pos = _make_position(
            entry_price=0.50,
            current_price=0.54,
            strategy_context={
                "_scale_out_targets_hit": [0, 1],
                "_scale_out_peak_price": 0.54,
            },
        )
        # Spread change = |0.54 - 0.50| / 0.50 * 10000 = 800 bps >= 50 bps threshold

        decision = strategy.default_exit_check(
            pos, _market_state(0.54, seconds_left=3600.0)
        )

        assert decision.action == "close"
        assert "Near-resolution exit" in decision.reason

    def test_near_resolution_no_trigger_too_far(self):
        strategy = _ScaleOutStrategy()
        pos = _make_position(
            entry_price=0.50,
            current_price=0.54,
            strategy_context={
                "_scale_out_targets_hit": [0, 1],
                "_scale_out_peak_price": 0.54,
            },
        )

        decision = strategy.default_exit_check(
            pos, _market_state(0.54, seconds_left=100000.0)
        )

        # 100000s = ~27.8h > 24h threshold
        assert decision.action == "hold"

    def test_near_resolution_no_trigger_spread_narrow(self):
        strategy = _ScaleOutStrategy()
        pos = _make_position(
            entry_price=0.50,
            current_price=0.5001,
            strategy_context={
                "_scale_out_targets_hit": [0, 1],
                "_scale_out_peak_price": 0.5001,
            },
        )
        # Spread change = |0.5001 - 0.50| / 0.50 * 10000 = 2 bps < 50 bps threshold

        decision = strategy.default_exit_check(
            pos, _market_state(0.5001, seconds_left=3600.0)
        )

        assert decision.action == "hold"

    def test_near_resolution_disabled(self):
        class _NoNearResStrategy(BaseStrategy):
            strategy_type = "test_no_near_res"
            name = "Test"
            description = "Test"
            scale_out_config = ScaleOutConfig(
                targets=[
                    ScaleOutTarget(trigger_bps=5, exit_fraction=0.33),
                ],
                near_resolution_exit=False,
            )

        strategy = _NoNearResStrategy()
        pos = _make_position(
            entry_price=0.50,
            current_price=0.54,
            strategy_context={
                "_scale_out_targets_hit": [0],
                "_scale_out_peak_price": 0.54,
            },
        )

        decision = strategy.default_exit_check(
            pos, _market_state(0.54, seconds_left=3600.0)
        )

        assert decision.action == "hold"


class TestScaleOutFallbackToDefault:
    def test_no_scale_out_config_uses_standard_tp(self):
        class _PlainStrategy(BaseStrategy):
            strategy_type = "test_plain"
            name = "Plain"
            description = "No scale out"

        strategy = _PlainStrategy()
        pos = _make_position(
            entry_price=0.50,
            current_price=0.56,
            config={"take_profit_pct": 10.0},
        )

        decision = strategy.default_exit_check(pos, _market_state(0.56))

        assert decision.action == "close"
        assert "Take profit hit" in decision.reason

    def test_scale_out_does_not_block_stop_loss(self):
        strategy = _ScaleOutStrategy()
        pos = _make_position(
            entry_price=0.50,
            current_price=0.45,
            config={"stop_loss_pct": 8.0},
        )

        decision = strategy.default_exit_check(pos, _market_state(0.45))

        assert decision.action == "close"
        assert "Stop loss hit" in decision.reason

    def test_scale_out_fires_before_take_profit(self):
        strategy = _ScaleOutStrategy()
        pos = _make_position(
            entry_price=0.50,
            current_price=0.525,  # +5%
            config={"take_profit_pct": 5.0},
        )

        decision = strategy.default_exit_check(pos, _market_state(0.525))

        assert decision.action == "reduce"
        assert "Scale-out target 1" in decision.reason

    def test_resolution_always_wins_over_scale_out(self):
        strategy = _ScaleOutStrategy()
        pos = _make_position(entry_price=0.50, current_price=0.525)

        decision = strategy.default_exit_check(
            pos, _market_state(0.525, is_resolved=True, winning_outcome=0)
        )

        assert decision.action == "close"
        assert "Market resolved" in decision.reason

    def test_min_hold_blocks_scale_out(self):
        strategy = _ScaleOutStrategy()
        pos = _make_position(
            entry_price=0.50,
            current_price=0.525,  # +5%
            age_minutes=2.0,
            config={"min_hold_minutes": 5.0},
        )

        decision = strategy.default_exit_check(pos, _market_state(0.525))

        assert decision.action == "hold"
        assert "Min hold" in decision.reason


class TestScaleOutEdgeCases:
    def test_single_target_config(self):
        class _SingleTargetStrategy(BaseStrategy):
            strategy_type = "test_single_target"
            name = "Single Target"
            description = "One target"
            scale_out_config = ScaleOutConfig(
                targets=[ScaleOutTarget(trigger_bps=3, exit_fraction=0.5)],
            )

        strategy = _SingleTargetStrategy()
        pos = _make_position(entry_price=0.50, current_price=0.515)  # +3%

        decision = strategy.default_exit_check(pos, _market_state(0.515))

        assert decision.action == "reduce"
        assert decision.reduce_fraction == 0.5

    def test_zero_trailing_stop_bps_no_trailing(self):
        class _NoTrailingStrategy(BaseStrategy):
            strategy_type = "test_no_trailing"
            name = "No Trailing"
            description = "Zero trailing"
            scale_out_config = ScaleOutConfig(
                targets=[ScaleOutTarget(trigger_bps=3, exit_fraction=0.5)],
                trailing_stop_bps=0.0,
                near_resolution_exit=False,
            )

        strategy = _NoTrailingStrategy()
        pos = _make_position(
            entry_price=0.50,
            current_price=0.40,  # Price dropped significantly
            strategy_context={
                "_scale_out_targets_hit": [0],
                "_scale_out_peak_price": 0.55,
            },
        )

        decision = strategy.default_exit_check(pos, _market_state(0.40))

        # trailing_stop_bps=0 means no trailing stop, should fall through to hold
        assert decision.action == "hold"

    def test_reduce_fraction_on_decision(self):
        strategy = _ScaleOutStrategy()
        pos = _make_position(entry_price=0.50, current_price=0.55)  # +10%

        decision = strategy.default_exit_check(pos, _market_state(0.55))

        assert decision.action == "reduce"
        assert decision.reduce_fraction == 0.33
        assert decision.close_price == 0.55

    def test_close_price_set_on_trailing_stop(self):
        strategy = _ScaleOutStrategy()
        pos = _make_position(
            entry_price=0.50,
            current_price=0.545,
            highest_price=0.55,
            strategy_context={
                "_scale_out_targets_hit": [0, 1],
                "_scale_out_peak_price": 0.55,
            },
        )

        decision = strategy.default_exit_check(pos, _market_state(0.545))

        assert decision.action == "close"
        assert decision.close_price == 0.545
