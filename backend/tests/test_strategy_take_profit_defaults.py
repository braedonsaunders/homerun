"""Exit take-profit default coverage for configured directional strategies."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.strategies.bayesian_cascade import BayesianCascadeStrategy
from services.strategies.bias_fader import BiasFaderStrategy
from services.strategies.btc_eth_highfreq import BtcEthHighFreqStrategy
from services.strategies.correlation_arb import CorrelationArbStrategy
from services.strategies.crypto_spike_reversion import CryptoSpikeReversionStrategy
from services.strategies.flash_crash_reversion import FlashCrashReversionStrategy
from services.strategies.flb_exploiter import FLBExploiterStrategy
from services.strategies.liquidity_vacuum import LiquidityVacuumStrategy
from services.strategies.market_making import MarketMakingStrategy
from services.strategies.prob_surface_arb import ProbSurfaceArbStrategy
from services.strategies.stat_arb import StatArbStrategy
from services.strategies.tail_end_carry import TailEndCarryStrategy
from services.strategies.temporal_decay import TemporalDecayStrategy
from services.strategies.term_premium import TermPremiumStrategy
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
        (BiasFaderStrategy, 12.0),
        (CorrelationArbStrategy, 12.0),
        (LiquidityVacuumStrategy, 12.0),
        (ProbSurfaceArbStrategy, 12.0),
        (StatArbStrategy, 12.0),
        (TradersConfluenceStrategy, 12.0),
        (VPINToxicityStrategy, 12.0),
        (TermPremiumStrategy, 10.0),
        (TemporalDecayStrategy, 10.0),
        (BtcEthHighFreqStrategy, 8.0),
        (CryptoSpikeReversionStrategy, 8.0),
        (FlashCrashReversionStrategy, 8.0),
        (MarketMakingStrategy, 8.0),
        (TailEndCarryStrategy, 8.0),
        (FLBExploiterStrategy, 8.0),
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
        BiasFaderStrategy,
        CorrelationArbStrategy,
        LiquidityVacuumStrategy,
        ProbSurfaceArbStrategy,
        StatArbStrategy,
        TradersConfluenceStrategy,
        VPINToxicityStrategy,
        TermPremiumStrategy,
        TemporalDecayStrategy,
        BtcEthHighFreqStrategy,
        CryptoSpikeReversionStrategy,
        FlashCrashReversionStrategy,
        MarketMakingStrategy,
        TailEndCarryStrategy,
        FLBExploiterStrategy,
    ],
)
def test_should_exit_preserves_configured_take_profit(strategy_cls):
    strategy = strategy_cls()
    position = _make_position({"take_profit_pct": 15.0})

    decision = strategy.should_exit(position, dict(MARKET_STATE))

    assert decision.action == "hold"
    assert position.config["take_profit_pct"] == 15.0
