"""StrategySDK.optimization facade — Frank-Wolfe + Bregman, exposed as opt-in
strategy math (not orchestrator-universal).

These assert the facade routes to the research modules correctly and
preserves their own invariants (Proposition 4.1: guaranteed ≤ arbitrage
profit), so the live combinatorial strategy's guaranteed_profit / capture
ratio are backed by the real math.
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import pytest


def test_optimization_namespace_resolves():
    from services.strategy_sdk import StrategySDK

    opt = StrategySDK.optimization
    assert hasattr(opt, "cross_market_profit_guarantee")
    assert hasattr(opt, "optimal_sizing")
    # Lazy descriptor returns the same module each access.
    assert StrategySDK.optimization is opt


def test_cross_market_profit_guarantee_respects_proposition_4_1():
    from services.strategy_sdk import StrategySDK

    # Underpriced dependent pair (each market sums < 1) => positive arbitrage.
    out = StrategySDK.optimization.cross_market_profit_guarantee(
        prices_a=[0.2, 0.3],
        prices_b=[0.2, 0.3],
        dependencies=[(0, 0, "implies")],
    )
    assert out["arbitrage_profit"] >= 0.0
    # Proposition 4.1: guaranteed profit never exceeds max arbitrage profit.
    assert out["guaranteed_profit"] <= out["arbitrage_profit"] + 1e-6
    assert 0.0 <= out["capture_ratio"] <= 1.0 + 1e-9
    assert isinstance(out["should_trade"], bool)


def test_cross_market_matches_direct_frank_wolfe():
    """Facade output equals calling the solver directly (no drift)."""
    import numpy as np

    from services.optimization.frank_wolfe import (
        create_cross_market_oracle,
        frank_wolfe_solver,
    )
    from services.strategy_sdk import StrategySDK

    prices_a = [0.25, 0.3]
    prices_b = [0.2, 0.35]
    deps = [(0, 0, "implies")]

    direct = frank_wolfe_solver.solve(
        np.array(prices_a + prices_b, dtype=float),
        create_cross_market_oracle(2, 2, deps),
    )
    via_sdk = StrategySDK.optimization.cross_market_profit_guarantee(
        prices_a=prices_a, prices_b=prices_b, dependencies=deps
    )
    assert via_sdk["guaranteed_profit"] == pytest.approx(direct.guaranteed_profit, abs=1e-6)
    assert via_sdk["arbitrage_profit"] == pytest.approx(direct.arbitrage_profit, abs=1e-6)


def test_optimal_sizing_returns_positions():
    from services.strategy_sdk import StrategySDK

    out = StrategySDK.optimization.optimal_sizing(prices=[0.3, 0.4], token_ids=["a", "b"])
    assert "projected_prices" in out
    assert "positions" in out
    assert "arbitrage_profit" in out
