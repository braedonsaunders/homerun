"""
Optimization module for advanced arbitrage detection and execution.

This module implements the mathematical infrastructure described in the
research paper "Unravelling the Probabilistic Forest: Arbitrage in
Prediction Markets" (arXiv:2508.03474v1).

Components:
- vwap: Volume-Weighted Average Price calculations
- bregman: Bregman projection for optimal trade sizing
- parallel_executor: Non-atomic parallel order execution

Future:
- constraint_solver: Integer programming for dependency detection
- frank_wolfe: Frank-Wolfe algorithm for marginal polytope projection
- dependency_detector: LLM-based market dependency detection
"""

from .vwap import VWAPCalculator, OrderBook, OrderBookLevel, VWAPResult
from .parallel_executor import ParallelExecutor

__all__ = [
    "VWAPCalculator",
    "OrderBook",
    "OrderBookLevel",
    "VWAPResult",
    "ParallelExecutor",
]
