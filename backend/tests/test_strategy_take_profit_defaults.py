"""Exit take-profit default coverage for configured directional strategies."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.strategies.bayesian_cascade import BayesianCascadeStrategy
from services.strategies.btc_eth_highfreq import BtcEthHighFreqStrategy
from services.strategies.correlation_arb import CorrelationArbStrategy
from services.strategies.crypto_spike_reversion import CryptoSpikeReversionStrategy
from services.strategies.flash_crash_reversion import FlashCrashReversionStrategy
from services.strategies.market_making import MarketMakingStrategy
from services.strategies.prob_surface_arb import ProbSurfaceArbStrategy
from services.strategies.stat_arb import StatArbStrategy
from services.strategies.tail_end_carry import TailEndCarryStrategy
from services.strategies.temporal_decay import TemporalDecayStrategy
from services.strategies.traders_confluence import TradersConfluenceStrategy
from services.strategies.vpin_toxicity import VPINToxicityStrategy


class _Position:
    pass


def _make_position(config: dict | None = None) -> _Position:
    position = _Position()
    position.entry_price = 0.50
    position.current_price = 0.56
    position.highest_price = 0.56
    position.lowest_price = 0.50
    position.age_minutes = 30.0
    position.pnl_percent = 12.0
    position.strategy_context = {}
    position.config = dict(config or {})
    position.outcome_idx = 0
    return position


def _make_loss_position(config: dict | None = None) -> _Position:
    position = _Position()
    position.entry_price = 0.50
    position.current_price = 0.45
    position.highest_price = 0.50
    position.lowest_price = 0.45
    position.age_minutes = 3.0
    position.pnl_percent = -10.0
    position.strategy_context = {}
    position.config = dict(config or {})
    position.outcome_idx = 0
    return position


MARKET_STATE = {
    "current_price": 0.56,
    "market_tradable": True,
    "is_resolved": False,
    "winning_outcome": None,
}


@pytest.mark.parametrize(
    ("strategy_cls", "expected_tp"),
    [
        (BayesianCascadeStrategy, 12.0),
        (CorrelationArbStrategy, 12.0),
        (ProbSurfaceArbStrategy, 12.0),
        (StatArbStrategy, 12.0),
        (TradersConfluenceStrategy, 12.0),
        (VPINToxicityStrategy, 12.0),
        (TemporalDecayStrategy, 10.0),
        (CryptoSpikeReversionStrategy, 8.0),
        (FlashCrashReversionStrategy, 8.0),
        (MarketMakingStrategy, 8.0),
        (TailEndCarryStrategy, 8.0),
    ],
)
def test_should_exit_applies_default_take_profit(strategy_cls, expected_tp):
    strategy = strategy_cls()
    position = _make_position()

    decision = strategy.should_exit(position, dict(MARKET_STATE))

    assert decision.action == "close"
    assert position.config["take_profit_pct"] == expected_tp


@pytest.mark.parametrize(
    "strategy_cls",
    [
        BayesianCascadeStrategy,
        CorrelationArbStrategy,
        ProbSurfaceArbStrategy,
        StatArbStrategy,
        TradersConfluenceStrategy,
        VPINToxicityStrategy,
        TemporalDecayStrategy,
        CryptoSpikeReversionStrategy,
        FlashCrashReversionStrategy,
        MarketMakingStrategy,
        TailEndCarryStrategy,
    ],
)
def test_should_exit_preserves_configured_take_profit(strategy_cls):
    strategy = strategy_cls()
    position = _make_position({"take_profit_pct": 15.0})

    decision = strategy.should_exit(position, dict(MARKET_STATE))

    assert decision.action == "hold"
    assert position.config["take_profit_pct"] == 15.0


def test_btc_highfreq_stop_loss_waits_for_near_close_window():
    strategy = BtcEthHighFreqStrategy()
    position = _make_loss_position()

    decision = strategy.should_exit(
        position,
        {
            "current_price": 0.45,
            "market_tradable": True,
            "is_resolved": False,
            "winning_outcome": None,
            "seconds_left": 240,
        },
    )

    assert decision.action == "close"
    assert "Stop loss hit" in decision.reason
    assert position.config["stop_loss_policy"] == "always"
    assert position.config["stop_loss_activation_seconds"] == 90


def test_btc_highfreq_stop_loss_arms_inside_near_close_window():
    strategy = BtcEthHighFreqStrategy()
    position = _make_loss_position()

    decision = strategy.should_exit(
        position,
        {
            "current_price": 0.45,
            "market_tradable": True,
            "is_resolved": False,
            "winning_outcome": None,
            "seconds_left": 30,
        },
    )

    assert decision.action == "close"
    assert "Stop loss hit" in decision.reason


def test_btc_highfreq_close_can_arm_reverse_intent_when_enabled():
    strategy = BtcEthHighFreqStrategy()
    position = _make_loss_position(
        {
            "reverse_on_adverse_velocity_enabled": True,
            "reverse_min_loss_pct": 1.0,
            "reverse_min_adverse_velocity_score": 0.1,
            "reverse_flow_imbalance_threshold": -0.1,
            "reverse_momentum_short_pct_threshold": -0.1,
            "reverse_min_seconds_left": 30.0,
            "reverse_min_price_headroom": 0.10,
            "reverse_min_edge_percent": 0.5,
            "reverse_confidence": 0.6,
            "reverse_size_multiplier": 1.25,
            "reverse_signal_ttl_seconds": 45.0,
            "reverse_cooldown_seconds": 12.0,
            "reverse_max_reentries_per_position": 2,
        }
    )
    position.direction = "buy_yes"
    position.strategy_context = {
        "timeframe": "5m",
        "_crypto_hf_rapid_exit_state": {
            "flow_imbalance": -0.4,
            "momentum_short_pct": -1.2,
            "executable_hazard_score": 0.75,
            "token_id": "token-reverse-1",
        },
    }

    decision = strategy.should_exit(
        position,
        {
            "current_price": 0.45,
            "market_tradable": True,
            "is_resolved": False,
            "winning_outcome": None,
            "seconds_left": 180,
        },
    )

    assert decision.action == "close"
    reverse_intent = decision.payload.get("reverse_intent")
    assert isinstance(reverse_intent, dict)
    assert reverse_intent["direction"] == "buy_no"
    assert reverse_intent["strategy_type"] == "btc_eth_highfreq"
    assert reverse_intent["token_id"] == "token-reverse-1"
    assert reverse_intent["max_reentries_per_position"] == 2


def test_btc_highfreq_stop_loss_timeframe_override_is_applied():
    strategy = BtcEthHighFreqStrategy()
    position = _make_loss_position(
        {
            "stop_loss_policy_5m": "near_close_only",
            "stop_loss_activation_seconds_5m": 20,
        }
    )
    position.strategy_context = {"timeframe": "5m"}

    decision = strategy.should_exit(
        position,
        {
            "current_price": 0.45,
            "market_tradable": True,
            "is_resolved": False,
            "winning_outcome": None,
            "seconds_left": 30,
        },
    )

    assert decision.action == "hold"
    assert "Stop loss deferred" in decision.reason
    assert position.config["stop_loss_policy"] == "near_close_only"
    assert position.config["stop_loss_activation_seconds"] == 20


def test_btc_highfreq_rapid_window_stall_exits_at_breakeven():
    strategy = BtcEthHighFreqStrategy()
    position = _make_position(
        {
            "rapid_exit_window_minutes_5m": 1.0,
            "min_hold_minutes": 10.0,
            "take_profit_pct": 50.0,
        }
    )
    position.strategy_context = {"timeframe": "5m"}
    position.entry_price = 0.50
    position.current_price = 0.50
    position.highest_price = 0.50
    position.age_minutes = 2.0

    decision = strategy.should_exit(
        position,
        {
            "current_price": 0.50,
            "market_tradable": True,
            "is_resolved": False,
            "winning_outcome": None,
            "seconds_left": 300,
        },
    )

    assert decision.action == "close"
    assert "Rapid window stalled without upside" in decision.reason
    assert position.config["rapid_exit_window_minutes"] == 1.0


def test_btc_highfreq_rapid_window_stall_not_flagged_when_price_increased():
    strategy = BtcEthHighFreqStrategy()
    position = _make_position(
        {
            "rapid_exit_window_minutes_5m": 1.0,
            "take_profit_pct": 50.0,
            "min_hold_minutes": 0.0,
        }
    )
    position.strategy_context = {"timeframe": "5m"}
    position.entry_price = 0.50
    position.current_price = 0.50
    position.highest_price = 0.53
    position.age_minutes = 2.0

    decision = strategy.should_exit(
        position,
        {
            "current_price": 0.50,
            "market_tradable": True,
            "is_resolved": False,
            "winning_outcome": None,
            "seconds_left": 300,
        },
    )

    rapid_state = position.strategy_context.get("_crypto_hf_rapid_exit_state")
    assert decision.action == "close"
    assert "Trailing stop" in decision.reason
    assert isinstance(rapid_state, dict)
    assert rapid_state.get("evaluated") is True
    assert rapid_state.get("stalled") is False


def test_btc_highfreq_trailing_stop_waits_for_activation_profit():
    strategy = BtcEthHighFreqStrategy()
    position = _make_position()
    position.strategy_context = {"timeframe": "5m"}
    position.entry_price = 0.50
    position.highest_price = 0.515  # +3.0% gain; below 5m default arm threshold (4%)
    position.current_price = 0.495  # Below trailing trigger if armed
    position.age_minutes = 1.0

    decision = strategy.should_exit(
        position,
        {
            "current_price": 0.495,
            "market_tradable": True,
            "is_resolved": False,
            "winning_outcome": None,
            "seconds_left": 300,
        },
    )

    assert decision.action == "hold"
    assert position.config["trailing_stop_activation_profit_pct"] == 4.0


def test_btc_highfreq_trailing_stop_arms_after_activation_profit():
    strategy = BtcEthHighFreqStrategy()
    position = _make_position()
    position.strategy_context = {"timeframe": "5m"}
    position.entry_price = 0.50
    position.highest_price = 0.53  # +6.0% gain; above 5m default arm threshold (4%)
    position.current_price = 0.513  # <= trailing trigger (0.5141)
    position.age_minutes = 1.0

    decision = strategy.should_exit(
        position,
        {
            "current_price": 0.513,
            "market_tradable": True,
            "is_resolved": False,
            "winning_outcome": None,
            "seconds_left": 300,
        },
    )

    assert decision.action == "close"
    assert "Trailing stop" in decision.reason


def test_btc_highfreq_rapid_peak_mode_arms_without_immediate_exit():
    strategy = BtcEthHighFreqStrategy()
    position = _make_position()
    position.strategy_context = {"timeframe": "15m"}
    position.entry_price = 0.50
    position.current_price = 0.56
    position.highest_price = 0.56
    position.age_minutes = 0.5

    decision = strategy.should_exit(
        position,
        {
            "current_price": 0.56,
            "market_tradable": True,
            "is_resolved": False,
            "winning_outcome": None,
            "seconds_left": 600,
            "min_order_size_usd": 1.0,
        },
    )

    rapid_state = position.strategy_context.get("_crypto_hf_rapid_exit_state")
    assert decision.action == "hold"
    assert isinstance(rapid_state, dict)
    assert rapid_state.get("armed") is True
    assert position.config["take_profit_pct"] > 15.0


def test_btc_highfreq_rapid_peak_mode_exits_on_backside_reversal():
    strategy = BtcEthHighFreqStrategy()
    position = _make_position()
    position.strategy_context = {"timeframe": "15m"}
    position.entry_price = 0.50
    position.current_price = 0.55
    position.highest_price = 0.58
    position.age_minutes = 1.0

    decision = strategy.should_exit(
        position,
        {
            "current_price": 0.55,
            "market_tradable": True,
            "is_resolved": False,
            "winning_outcome": None,
            "seconds_left": 600,
            "min_order_size_usd": 1.0,
        },
    )

    assert decision.action == "close"
    assert "Adaptive backside peak exit" in decision.reason


def test_btc_highfreq_executable_notional_guard_forces_early_exit():
    strategy = BtcEthHighFreqStrategy()
    position = _make_loss_position()
    position.strategy_context = {"timeframe": "15m"}
    position.entry_price = 0.50
    position.current_price = 0.20
    position.highest_price = 0.53
    position.age_minutes = 1.0
    position.filled_size = 6.0
    position.notional_usd = 3.0

    decision = strategy.should_exit(
        position,
        {
            "current_price": 0.20,
            "market_tradable": True,
            "is_resolved": False,
            "winning_outcome": None,
            "seconds_left": 600,
            "min_order_size_usd": 1.0,
        },
    )

    assert decision.action == "close"
    assert "Executable-notional guard" in decision.reason


def test_btc_highfreq_underwater_rebound_salvage_exit_after_dwell_and_fade():
    strategy = BtcEthHighFreqStrategy()
    position = _make_loss_position(
        {
            "underwater_rebound_exit_enabled": True,
            "underwater_dwell_minutes_5m": 1.0,
            "underwater_recovery_ratio_min_5m": 0.25,
            "underwater_rebound_pct_min_5m": 3.0,
            "underwater_exit_fade_pct_5m": 0.30,
            "underwater_timeout_minutes_5m": 20.0,
            "underwater_timeout_loss_pct": 20.0,
            "stop_loss_pct": 50.0,
            "take_profit_pct": 100.0,
            "trailing_stop_pct": 0.0,
            "min_hold_minutes": 0.0,
        }
    )
    position.strategy_context = {"timeframe": "5m"}
    position.entry_price = 0.50
    position.lowest_price = 0.45
    position.highest_price = 0.50
    position.current_price = 0.45
    position.age_minutes = 1.0

    first = strategy.should_exit(
        position,
        {
            "current_price": 0.45,
            "market_tradable": True,
            "is_resolved": False,
            "winning_outcome": None,
            "seconds_left": 180,
        },
    )

    assert first.action == "hold"

    position.current_price = 0.48
    position.highest_price = 0.50
    position.age_minutes = 3.0
    second = strategy.should_exit(
        position,
        {
            "current_price": 0.48,
            "market_tradable": True,
            "is_resolved": False,
            "winning_outcome": None,
            "seconds_left": 180,
        },
    )

    assert second.action == "hold"

    position.current_price = 0.477
    position.highest_price = 0.50
    position.age_minutes = 3.2
    third = strategy.should_exit(
        position,
        {
            "current_price": 0.477,
            "market_tradable": True,
            "is_resolved": False,
            "winning_outcome": None,
            "seconds_left": 180,
        },
    )

    assert third.action == "close"
    assert "Underwater rebound salvage exit" in third.reason
    assert position.config["underwater_dwell_minutes"] == 1.0


def test_btc_highfreq_underwater_timeout_stop_exits_after_dwell():
    strategy = BtcEthHighFreqStrategy()
    position = _make_loss_position(
        {
            "underwater_rebound_exit_enabled": True,
            "underwater_dwell_minutes_5m": 3.0,
            "underwater_recovery_ratio_min_5m": 0.9,
            "underwater_rebound_pct_min_5m": 8.0,
            "underwater_exit_fade_pct_5m": 2.0,
            "underwater_timeout_minutes_5m": 1.5,
            "underwater_timeout_loss_pct": 6.0,
            "stop_loss_pct": 50.0,
            "take_profit_pct": 100.0,
            "trailing_stop_pct": 0.0,
            "min_hold_minutes": 0.0,
        }
    )
    position.strategy_context = {"timeframe": "5m"}
    position.entry_price = 0.50
    position.lowest_price = 0.45
    position.highest_price = 0.50
    position.current_price = 0.46
    position.age_minutes = 0.2

    first = strategy.should_exit(
        position,
        {
            "current_price": 0.46,
            "market_tradable": True,
            "is_resolved": False,
            "winning_outcome": None,
            "seconds_left": 180,
        },
    )
    assert first.action == "hold"

    position.current_price = 0.46
    position.age_minutes = 2.0
    second = strategy.should_exit(
        position,
        {
            "current_price": 0.46,
            "market_tradable": True,
            "is_resolved": False,
            "winning_outcome": None,
            "seconds_left": 180,
        },
    )

    assert second.action == "close"
    assert "Underwater timeout stop" in second.reason


def test_btc_highfreq_near_expiry_flatten_guard_closes_small_pnl_position():
    strategy = BtcEthHighFreqStrategy()
    position = _make_position(
        {
            "take_profit_pct": 100.0,
            "stop_loss_pct": 50.0,
            "trailing_stop_pct": 0.0,
            "min_hold_minutes": 0.0,
        }
    )
    position.strategy_context = {"timeframe": "15m"}
    position.entry_price = 0.50
    position.current_price = 0.503
    position.highest_price = 0.503
    position.age_minutes = 1.0
    position.filled_size = 20.0

    decision = strategy.should_exit(
        position,
        {
            "current_price": 0.503,
            "market_tradable": True,
            "is_resolved": False,
            "winning_outcome": None,
            "seconds_left": 60,
            "min_order_size_usd": 1.0,
        },
    )

    assert decision.action == "close"
    assert "Resolution-risk flatten" in decision.reason
