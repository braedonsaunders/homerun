"""Production-grade backtester for Polymarket prediction markets.

This package replaces the legacy single-fill simulator (``services/simulation``)
for execution-realistic backtesting. It models:

* **L2 order-book replay** from ``MarketMicrostructureSnapshot`` (real venue
  depth at 0.5s sampling, up to 25 levels per side).
* **Per-venue order semantics**: GTC, IOC, FAK, FOK; post-only cross-book
  rejection; tick rounding; Polymarket $1 min notional.
* **Latency model**: parameterized log-normal latency with calibratable
  quantiles. Submit→ack and ack→fill are modeled separately.
* **Matching engine**: orders walk the visible book; partial fills emit
  multiple fills with per-fill price; cancel/replace lifecycle is tracked.
* **Multi-child exits**: integrates with the live ``exit_executor`` so a
  strategy's ``ExitPolicy`` (ladder, chunk, escalate, reprice) is simulated
  child-by-child against the same book replay path that live trading uses.
* **Portfolio**: capital constraint, per-strategy exposure caps,
  realized-correlation sampling across concurrent positions.
* **Walk-forward**: rolling and anchored splits, with bootstrap confidence
  intervals on Sharpe / Sortino / max-drawdown / hit-rate / profit-factor.

The legacy ``execution_simulator.py`` remains for shadow trading replay; this
package is for *strategy-fitness* backtesting where execution realism matters.
"""

from services.backtest.book_replay import (
    BookSnapshot,
    BookReplay,
    InMemoryBookReplay,
    PriceLevel,
)
from services.backtest.venue_model import (
    PolymarketVenue,
    Venue,
    OrderRule,
    SubmitDecision,
)
from services.backtest.latency_model import (
    LatencyModel,
    LatencyProfile,
)
from services.backtest.matching_engine import (
    BacktestOrder,
    Fill,
    MatchingEngine,
    OrderState,
)
from services.backtest.portfolio import (
    Portfolio,
    PortfolioConfig,
    Position,
)
from services.backtest.metrics import (
    BacktestMetrics,
    bootstrap_ci,
    compute_metrics,
)
from services.backtest.walk_forward import (
    WalkForwardConfig,
    WalkForwardResult,
    walk_forward_split,
)
from services.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    TradeIntent,
)


__all__ = [
    "BookSnapshot",
    "BookReplay",
    "InMemoryBookReplay",
    "PriceLevel",
    "PolymarketVenue",
    "Venue",
    "OrderRule",
    "SubmitDecision",
    "LatencyModel",
    "LatencyProfile",
    "BacktestOrder",
    "Fill",
    "MatchingEngine",
    "OrderState",
    "Portfolio",
    "PortfolioConfig",
    "Position",
    "BacktestMetrics",
    "bootstrap_ci",
    "compute_metrics",
    "WalkForwardConfig",
    "WalkForwardResult",
    "walk_forward_split",
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "TradeIntent",
]
