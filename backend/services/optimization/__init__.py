"""
Optimization module for advanced arbitrage detection and execution.

This module implements the mathematical infrastructure described in the
research paper "Unravelling the Probabilistic Forest: Arbitrage in
Prediction Markets" (arXiv:2508.03474v1).

Components:
- vwap: Volume-Weighted Average Price calculations for realistic profit estimation
- bregman: Bregman projection for optimal trade sizing using KL divergence
- parallel_executor: Non-atomic parallel order execution via asyncio.gather
- constraint_solver: Integer programming for dependency detection
- frank_wolfe: Frank-Wolfe algorithm for marginal polytope projection
- dependency_detector: LLM-based market dependency detection

Key insights from $40M extraction research:
1. Marginal polytope problem: Arbitrage-free prices must lie within M = conv(Z)
2. Bregman projection: Optimal profit equals D(μ*||θ) using KL divergence
3. Frank-Wolfe: Makes projection tractable for 2^63 outcome spaces
4. Parallel execution: All legs in same block eliminates sequential risk
5. VWAP analysis: Mid-price assumptions overestimate profits by ~40%
"""

from .vwap import VWAPCalculator, OrderBook, OrderBookLevel, VWAPResult
from .parallel_executor import ParallelExecutor, ExecutionLeg, ParallelExecutionResult
from .bregman import BregmanProjector, ProjectionResult, bregman_projector
from .constraint_solver import (
    ConstraintSolver, ArbitrageResult, Dependency, DependencyType, constraint_solver
)
from .frank_wolfe import (
    FrankWolfeSolver, FrankWolfeResult, InitFWResult, IPOracle,
    create_binary_market_oracle, create_cross_market_oracle, frank_wolfe_solver
)
from .dependency_detector import (
    DependencyDetector, DependencyAnalysis, MarketInfo, dependency_detector
)

__all__ = [
    # VWAP
    "VWAPCalculator",
    "OrderBook",
    "OrderBookLevel",
    "VWAPResult",
    # Parallel Execution
    "ParallelExecutor",
    "ExecutionLeg",
    "ParallelExecutionResult",
    # Bregman Projection
    "BregmanProjector",
    "ProjectionResult",
    "bregman_projector",
    # Constraint Solver
    "ConstraintSolver",
    "ArbitrageResult",
    "Dependency",
    "DependencyType",
    "constraint_solver",
    # Frank-Wolfe
    "FrankWolfeSolver",
    "FrankWolfeResult",
    "InitFWResult",
    "IPOracle",
    "create_binary_market_oracle",
    "create_cross_market_oracle",
    "frank_wolfe_solver",
    # Dependency Detection
    "DependencyDetector",
    "DependencyAnalysis",
    "MarketInfo",
    "dependency_detector",
]
