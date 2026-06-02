"""Strategy-facing facade for the arbitrage-optimization research modules.

Exposed via ``StrategySDK.optimization`` so any user-defined strategy can
compute combinatorial / cross-market arbitrage profit guarantees
(Frank-Wolfe, Proposition 4.1) and Bregman-projected optimal sizing WITHOUT
importing the solver internals.

Architectural intent: this is OPT-IN strategy math a strategy author calls —
only strategies that trade dependent / cross-market arbitrage need it. The
trader orchestrator stays strategy-agnostic and never reaches into these
modules; it only carries the resulting ``guaranteed_profit`` /
``capture_ratio`` fields on the generic Opportunity contract.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence


def cross_market_profit_guarantee(
    *,
    prices_a: Sequence[float],
    prices_b: Sequence[float],
    dependencies: Sequence[tuple],
    execution_cost: float = 0.02,
) -> dict[str, Any]:
    """Frank-Wolfe (Proposition 4.1) arbitrage profit guarantee for a pair of
    dependent markets.

    Args:
        prices_a: outcome prices for market A (e.g. ``[yes, no]``).
        prices_b: outcome prices for market B.
        dependencies: list of ``(idx_a, idx_b, "implies"|"excludes")`` tuples
            relating an outcome of A to an outcome of B.
        execution_cost: gas + slippage estimate used by the execute gate.

    Returns a dict with ``guaranteed_profit`` (D(μ̂||θ) − g(μ̂), ≥ 0),
    ``capture_ratio`` (guaranteed / max arbitrage profit, in [0, 1]),
    ``arbitrage_profit``, ``should_trade``, ``reason``, ``converged``.
    """
    import numpy as np

    from services.optimization.frank_wolfe import (
        create_cross_market_oracle,
        frank_wolfe_solver,
    )

    pa = [float(p) for p in prices_a]
    pb = [float(p) for p in prices_b]
    prices = np.array(pa + pb, dtype=float)
    oracle = create_cross_market_oracle(len(pa), len(pb), list(dependencies))
    result = frank_wolfe_solver.solve(prices, oracle)
    decision = frank_wolfe_solver.should_execute_trade(
        result, execution_cost=float(execution_cost)
    )
    return {
        "guaranteed_profit": float(
            decision.get("guaranteed_profit", result.guaranteed_profit) or 0.0
        ),
        "capture_ratio": float(decision.get("capture_ratio", 0.0) or 0.0),
        "arbitrage_profit": float(result.arbitrage_profit),
        "should_trade": bool(decision.get("should_trade", False)),
        "reason": decision.get("reason"),
        "converged": bool(result.converged),
    }


def optimal_sizing(
    *,
    prices: Sequence[float],
    token_ids: Optional[Sequence[str]] = None,
    constraint_type: str = "sum_to_one",
) -> dict[str, Any]:
    """Bregman-projected optimal trade vector (KL projection onto the
    marginal polytope) for a single market's outcome prices.

    Returns the projected prices + executable positions (filtered of dust)
    via ``BregmanProjector.compute_optimal_trade``.
    """
    from services.optimization.bregman import bregman_projector

    p = [float(x) for x in prices]
    toks = (
        [str(t) for t in token_ids]
        if token_ids is not None
        else [str(i) for i in range(len(p))]
    )
    return bregman_projector.compute_optimal_trade(p, toks, constraint_type=constraint_type)
