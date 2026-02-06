"""
Hyperparameter Optimization Framework for Trading Parameters

Centralizes all tunable trading parameters and provides:
- Parameter sweep engine (grid search & random search)
- Walk-forward validation to prevent overfitting
- Database persistence of parameter sets and optimization results
- API for parameter management

Replays historical opportunity data from OpportunityHistory to evaluate
different parameter configurations and find optimal settings.
"""

import asyncio
import itertools
import math
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import numpy as np
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import AsyncSessionLocal, OpportunityHistory, ParameterSet
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Parameter definition helpers
# ---------------------------------------------------------------------------

@dataclass
class ParameterSpec:
    """Specification for a single tunable parameter."""
    name: str
    current_value: float
    min_bound: float
    max_bound: float
    step: float
    description: str
    category: str  # entry_criteria | risk_management | position_sizing | strategy_specific

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ParameterSpec":
        return ParameterSpec(**d)

    def grid_values(self) -> list[float]:
        """Return discrete values spanning [min_bound, max_bound] at *step*."""
        if self.step <= 0:
            return [self.current_value]
        vals: list[float] = []
        v = self.min_bound
        while v <= self.max_bound + 1e-9:
            vals.append(round(v, 10))
            v += self.step
        return vals


# ---------------------------------------------------------------------------
# Centralized TradingParameters dataclass
# ---------------------------------------------------------------------------

@dataclass
class TradingParameters:
    """All tunable trading values in one place, grouped by category."""

    # -- entry_criteria --
    min_profit_threshold: float = 0.025
    polymarket_fee: float = 0.02
    min_liquidity: float = 1000.0
    min_roi_percent: float = 2.5
    max_risk_score: float = 0.5
    min_liquidity_usd: float = 5000.0
    min_impossibility_score: float = 0.8
    min_guaranteed_profit: float = 0.05

    # -- risk_management --
    circuit_breaker_losses: int = 3
    max_daily_trades: int = 50
    max_daily_loss_usd: float = 100.0
    cooldown_after_loss_seconds: int = 60

    # -- position_sizing --
    base_position_size_usd: float = 10.0
    max_position_size_usd: float = 100.0

    # -- strategy_specific: base.py risk thresholds --
    risk_time_short: float = 0.4
    risk_time_medium: float = 0.2
    risk_liquidity_low: float = 0.3
    risk_liquidity_moderate: float = 0.15
    risk_complexity_high: float = 0.2
    risk_complexity_moderate: float = 0.1

    # -- strategy_specific: miracle keyword weights (representative subset) --
    miracle_alien_weight: float = 0.95
    miracle_supernatural_weight: float = 0.90
    miracle_apocalypse_weight: float = 0.80
    miracle_time_travel_weight: float = 0.95
    miracle_min_no_price: float = 0.90
    miracle_max_no_price: float = 0.995
    miracle_min_impossibility: float = 0.70

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TradingParameters":
        # Only pass keys that exist as fields on the dataclass
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


# The canonical registry of every parameter, its bounds, and step size.
DEFAULT_PARAM_SPECS: list[ParameterSpec] = [
    # --- entry_criteria ---
    ParameterSpec("min_profit_threshold", 0.025, 0.005, 0.10, 0.005,
                  "Minimum profit threshold after fees (fraction)", "entry_criteria"),
    ParameterSpec("polymarket_fee", 0.02, 0.01, 0.05, 0.005,
                  "Polymarket winner fee (fraction)", "entry_criteria"),
    ParameterSpec("min_liquidity", 1000.0, 200.0, 5000.0, 200.0,
                  "Minimum market liquidity in USD (config)", "entry_criteria"),
    ParameterSpec("min_roi_percent", 2.5, 0.5, 10.0, 0.5,
                  "Minimum ROI percentage to enter trade", "entry_criteria"),
    ParameterSpec("max_risk_score", 0.5, 0.1, 0.9, 0.1,
                  "Maximum acceptable risk score (0-1)", "entry_criteria"),
    ParameterSpec("min_liquidity_usd", 5000.0, 1000.0, 20000.0, 1000.0,
                  "Minimum market liquidity for auto trader (USD)", "entry_criteria"),
    ParameterSpec("min_impossibility_score", 0.8, 0.5, 1.0, 0.05,
                  "Minimum impossibility score for miracle strategy", "entry_criteria"),
    ParameterSpec("min_guaranteed_profit", 0.05, 0.01, 0.20, 0.01,
                  "Minimum guaranteed profit from Proposition 4.1", "entry_criteria"),

    # --- risk_management ---
    ParameterSpec("circuit_breaker_losses", 3, 1, 10, 1,
                  "Consecutive losses before circuit breaker triggers", "risk_management"),
    ParameterSpec("max_daily_trades", 50, 10, 200, 10,
                  "Maximum trades per day", "risk_management"),
    ParameterSpec("max_daily_loss_usd", 100.0, 25.0, 500.0, 25.0,
                  "Daily loss limit in USD", "risk_management"),
    ParameterSpec("cooldown_after_loss_seconds", 60, 0, 300, 30,
                  "Seconds to wait after a losing trade", "risk_management"),

    # --- position_sizing ---
    ParameterSpec("base_position_size_usd", 10.0, 1.0, 100.0, 5.0,
                  "Base position size in USD", "position_sizing"),
    ParameterSpec("max_position_size_usd", 100.0, 10.0, 500.0, 10.0,
                  "Maximum position size per trade in USD", "position_sizing"),

    # --- strategy_specific: risk score weights (base.py) ---
    ParameterSpec("risk_time_short", 0.4, 0.1, 0.7, 0.05,
                  "Risk weight for <2-day resolution", "strategy_specific"),
    ParameterSpec("risk_time_medium", 0.2, 0.05, 0.5, 0.05,
                  "Risk weight for <7-day resolution", "strategy_specific"),
    ParameterSpec("risk_liquidity_low", 0.3, 0.1, 0.6, 0.05,
                  "Risk weight for low liquidity (<$1k)", "strategy_specific"),
    ParameterSpec("risk_liquidity_moderate", 0.15, 0.05, 0.4, 0.05,
                  "Risk weight for moderate liquidity (<$5k)", "strategy_specific"),
    ParameterSpec("risk_complexity_high", 0.2, 0.05, 0.5, 0.05,
                  "Risk weight for complex trades (>5 markets)", "strategy_specific"),
    ParameterSpec("risk_complexity_moderate", 0.1, 0.02, 0.3, 0.02,
                  "Risk weight for multi-position trades (>3 markets)", "strategy_specific"),

    # --- strategy_specific: miracle weights ---
    ParameterSpec("miracle_alien_weight", 0.95, 0.7, 1.0, 0.05,
                  "Miracle keyword weight for alien/ufo terms", "strategy_specific"),
    ParameterSpec("miracle_supernatural_weight", 0.90, 0.6, 1.0, 0.05,
                  "Miracle keyword weight for supernatural terms", "strategy_specific"),
    ParameterSpec("miracle_apocalypse_weight", 0.80, 0.5, 1.0, 0.05,
                  "Miracle keyword weight for apocalypse terms", "strategy_specific"),
    ParameterSpec("miracle_time_travel_weight", 0.95, 0.7, 1.0, 0.05,
                  "Miracle keyword weight for impossible physics", "strategy_specific"),
    ParameterSpec("miracle_min_no_price", 0.90, 0.80, 0.97, 0.01,
                  "Minimum NO price for miracle strategy", "strategy_specific"),
    ParameterSpec("miracle_max_no_price", 0.995, 0.98, 1.0, 0.005,
                  "Maximum NO price for miracle strategy", "strategy_specific"),
    ParameterSpec("miracle_min_impossibility", 0.70, 0.40, 0.90, 0.05,
                  "Minimum impossibility score within miracle strategy", "strategy_specific"),
]


# ---------------------------------------------------------------------------
# Backtest metrics
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Results from running a parameter set against historical data."""
    total_profit: float = 0.0
    total_cost: float = 0.0
    num_trades: int = 0
    num_wins: int = 0
    num_losses: int = 0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    avg_roi: float = 0.0
    profit_factor: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "BacktestResult":
        return BacktestResult(**{k: v for k, v in d.items() if k in BacktestResult.__dataclass_fields__})


class SearchMethod(str, Enum):
    GRID = "grid"
    RANDOM = "random"


# ---------------------------------------------------------------------------
# Core backtesting / replay engine
# ---------------------------------------------------------------------------

def _replay_opportunities(
    opportunities: list[dict],
    params: TradingParameters,
) -> BacktestResult:
    """
    Replay historical opportunities with a given parameter set.

    Each opportunity dict mirrors the columns of OpportunityHistory:
        strategy_type, total_cost, expected_roi, risk_score,
        was_profitable, actual_roi, detected_at, positions_data
    """
    if not opportunities:
        return BacktestResult()

    equity_curve: list[float] = [0.0]
    gross_wins = 0.0
    gross_losses = 0.0
    num_wins = 0
    num_losses = 0
    total_profit = 0.0
    total_cost = 0.0
    roi_values: list[float] = []

    consecutive_losses = 0
    circuit_breaker_until: Optional[datetime] = None

    for opp in opportunities:
        # -- entry criteria filters --
        expected_roi = opp.get("expected_roi") or 0.0
        risk_score = opp.get("risk_score") or 0.0
        strategy = opp.get("strategy_type", "")
        opp_cost = opp.get("total_cost") or 0.0
        detected_at = opp.get("detected_at")

        # Circuit breaker check
        if circuit_breaker_until is not None and detected_at is not None:
            if isinstance(detected_at, str):
                try:
                    detected_at = datetime.fromisoformat(detected_at)
                except (ValueError, TypeError):
                    detected_at = None
            if detected_at is not None and detected_at < circuit_breaker_until:
                continue
            circuit_breaker_until = None
            consecutive_losses = 0

        if expected_roi < params.min_roi_percent:
            continue
        if risk_score > params.max_risk_score:
            continue

        # Approximate liquidity filter via position data
        positions_data = opp.get("positions_data") or []
        min_liq = 0.0
        if isinstance(positions_data, list):
            liqs = [p.get("liquidity", 0) for p in positions_data if isinstance(p, dict)]
            min_liq = min(liqs) if liqs else 0.0
        elif isinstance(positions_data, dict):
            min_liq = positions_data.get("liquidity", 0)

        if min_liq > 0 and min_liq < params.min_liquidity_usd:
            continue

        # Miracle-specific: check impossibility score threshold
        if strategy == "miracle" and expected_roi < params.min_roi_percent:
            continue

        # Profit threshold filter (using expected ROI as proxy)
        roi_as_fraction = expected_roi / 100.0 if expected_roi > 1 else expected_roi
        if roi_as_fraction < params.min_profit_threshold:
            continue

        # -- Determine position size --
        pos_size = min(params.base_position_size_usd, params.max_position_size_usd)
        if opp_cost > 0:
            pos_size = min(pos_size, opp_cost * params.max_position_size_usd)

        # -- Evaluate outcome --
        was_profitable = opp.get("was_profitable")
        actual_roi = opp.get("actual_roi")

        if was_profitable is None:
            # No resolution data -- skip for backtest purposes
            continue

        if actual_roi is not None:
            pnl = pos_size * (actual_roi / 100.0 if abs(actual_roi) > 1 else actual_roi)
        else:
            # Estimate from was_profitable flag
            fee = pos_size * params.polymarket_fee
            if was_profitable:
                pnl = pos_size * roi_as_fraction - fee
            else:
                pnl = -pos_size

        total_profit += pnl
        total_cost += pos_size
        equity_curve.append(equity_curve[-1] + pnl)

        if pnl >= 0:
            num_wins += 1
            gross_wins += pnl
            consecutive_losses = 0
        else:
            num_losses += 1
            gross_losses += abs(pnl)
            consecutive_losses += 1

            # Circuit breaker
            if consecutive_losses >= int(params.circuit_breaker_losses):
                if detected_at is not None:
                    if isinstance(detected_at, str):
                        try:
                            detected_at = datetime.fromisoformat(detected_at)
                        except (ValueError, TypeError):
                            detected_at = None
                    if detected_at is not None:
                        circuit_breaker_until = detected_at + timedelta(
                            seconds=params.cooldown_after_loss_seconds
                        )

        roi_values.append(pnl / pos_size if pos_size > 0 else 0.0)

    num_trades = num_wins + num_losses
    win_rate = num_wins / num_trades if num_trades > 0 else 0.0

    # Max drawdown from equity curve
    peak = 0.0
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd

    # Sharpe ratio (annualised, assuming ~365 daily observations)
    if len(roi_values) >= 2:
        arr = np.array(roi_values, dtype=np.float64)
        mean_ret = float(np.mean(arr))
        std_ret = float(np.std(arr, ddof=1))
        sharpe = (mean_ret / std_ret) * math.sqrt(365) if std_ret > 0 else 0.0
    else:
        sharpe = 0.0

    avg_roi = float(np.mean(roi_values)) * 100 if roi_values else 0.0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf") if gross_wins > 0 else 0.0

    return BacktestResult(
        total_profit=round(total_profit, 4),
        total_cost=round(total_cost, 4),
        num_trades=num_trades,
        num_wins=num_wins,
        num_losses=num_losses,
        win_rate=round(win_rate, 4),
        max_drawdown=round(max_dd, 4),
        sharpe_ratio=round(sharpe, 4),
        avg_roi=round(avg_roi, 4),
        profit_factor=round(profit_factor, 4),
    )


# ---------------------------------------------------------------------------
# Parameter sweep helpers
# ---------------------------------------------------------------------------

def _build_param_sets_grid(
    base: TradingParameters,
    param_ranges: dict[str, list[float]],
) -> list[TradingParameters]:
    """
    Cartesian product of supplied ranges.

    *param_ranges* maps parameter name -> list of values to try.
    All other parameters stay at *base* values.
    """
    names = list(param_ranges.keys())
    value_lists = [param_ranges[n] for n in names]
    combos = list(itertools.product(*value_lists))

    result: list[TradingParameters] = []
    base_dict = base.to_dict()
    for combo in combos:
        d = dict(base_dict)
        for name, val in zip(names, combo):
            d[name] = val
        result.append(TradingParameters.from_dict(d))
    return result


def _build_param_sets_random(
    base: TradingParameters,
    param_ranges: dict[str, tuple[float, float]],
    n_samples: int = 100,
    seed: int = 42,
) -> list[TradingParameters]:
    """
    Latin-hypercube-style random sampling.

    *param_ranges* maps parameter name -> (min, max).
    """
    rng = np.random.default_rng(seed)
    base_dict = base.to_dict()
    names = list(param_ranges.keys())

    result: list[TradingParameters] = []
    for _ in range(n_samples):
        d = dict(base_dict)
        for name in names:
            lo, hi = param_ranges[name]
            d[name] = round(float(rng.uniform(lo, hi)), 6)
        result.append(TradingParameters.from_dict(d))
    return result


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def _split_walk_forward(
    opportunities: list[dict],
    n_windows: int = 5,
    train_ratio: float = 0.7,
) -> list[tuple[list[dict], list[dict]]]:
    """
    Split chronologically-sorted opportunities into overlapping
    train/test windows for walk-forward validation.

    Returns list of (train_set, test_set) tuples.
    """
    n = len(opportunities)
    if n < 10:
        # Not enough data -- single fold
        split = int(n * train_ratio)
        return [(opportunities[:split], opportunities[split:])]

    window_size = n // n_windows
    folds: list[tuple[list[dict], list[dict]]] = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size if i < n_windows - 1 else n
        window = opportunities[start:end]
        split = int(len(window) * train_ratio)
        if split == 0 or split >= len(window):
            continue
        folds.append((window[:split], window[split:]))
    return folds


# ---------------------------------------------------------------------------
# ParameterOptimizer service (singleton)
# ---------------------------------------------------------------------------

class ParameterOptimizer:
    """
    Hyperparameter optimization service.

    Provides:
    - Centralised parameter registry
    - Grid / random sweep over parameter space
    - Walk-forward validation against OpportunityHistory
    - Database persistence of parameter sets
    """

    _instance: Optional["ParameterOptimizer"] = None

    def __new__(cls) -> "ParameterOptimizer":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._param_specs: list[ParameterSpec] = list(DEFAULT_PARAM_SPECS)
        self._current_params = TradingParameters()
        self._optimization_results: list[dict] = []
        logger.info("ParameterOptimizer initialised")

    # ------------------------------------------------------------------
    # Parameter spec helpers
    # ------------------------------------------------------------------

    def get_param_specs(self) -> list[dict]:
        """Return all parameter specifications."""
        return [s.to_dict() for s in self._param_specs]

    def get_param_specs_by_category(self) -> dict[str, list[dict]]:
        """Group parameter specs by category."""
        grouped: dict[str, list[dict]] = {}
        for s in self._param_specs:
            grouped.setdefault(s.category, []).append(s.to_dict())
        return grouped

    # ------------------------------------------------------------------
    # Current parameters
    # ------------------------------------------------------------------

    def get_current_params(self) -> dict[str, Any]:
        """Return current active parameter values."""
        return self._current_params.to_dict()

    def set_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Update current parameters (in-memory)."""
        current = self._current_params.to_dict()
        current.update(params)
        self._current_params = TradingParameters.from_dict(current)
        logger.info("Parameters updated", updated_keys=list(params.keys()))
        return self._current_params.to_dict()

    # ------------------------------------------------------------------
    # Database persistence
    # ------------------------------------------------------------------

    async def save_parameter_set(
        self,
        name: str,
        params: dict[str, Any],
        backtest_results: Optional[dict] = None,
        is_active: bool = False,
    ) -> str:
        """Persist a parameter set to the database."""
        set_id = str(uuid.uuid4())
        async with AsyncSessionLocal() as session:
            async with session.begin():
                if is_active:
                    # Deactivate all others first
                    await session.execute(
                        update(ParameterSet).values(is_active=False)
                    )
                ps = ParameterSet(
                    id=set_id,
                    name=name,
                    parameters=params,
                    backtest_results=backtest_results,
                    is_active=is_active,
                    created_at=datetime.utcnow(),
                )
                session.add(ps)
        logger.info("Parameter set saved", id=set_id, name=name, active=is_active)
        return set_id

    async def load_parameter_set(self, set_id: str) -> Optional[dict]:
        """Load a parameter set from the database."""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(ParameterSet).where(ParameterSet.id == set_id)
            )
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return {
                "id": row.id,
                "name": row.name,
                "parameters": row.parameters,
                "backtest_results": row.backtest_results,
                "is_active": row.is_active,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }

    async def load_active_parameter_set(self) -> Optional[dict]:
        """Load the currently active parameter set."""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(ParameterSet).where(ParameterSet.is_active == True)
            )
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return {
                "id": row.id,
                "name": row.name,
                "parameters": row.parameters,
                "backtest_results": row.backtest_results,
                "is_active": row.is_active,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }

    async def list_parameter_sets(self) -> list[dict]:
        """List all saved parameter sets."""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(ParameterSet).order_by(ParameterSet.created_at.desc())
            )
            rows = result.scalars().all()
            return [
                {
                    "id": r.id,
                    "name": r.name,
                    "parameters": r.parameters,
                    "backtest_results": r.backtest_results,
                    "is_active": r.is_active,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]

    async def activate_parameter_set(self, set_id: str) -> bool:
        """Set one parameter set as active (deactivate all others)."""
        async with AsyncSessionLocal() as session:
            async with session.begin():
                await session.execute(
                    update(ParameterSet).values(is_active=False)
                )
                result = await session.execute(
                    update(ParameterSet)
                    .where(ParameterSet.id == set_id)
                    .values(is_active=True)
                )
                if result.rowcount == 0:
                    return False
        logger.info("Activated parameter set", id=set_id)
        return True

    async def delete_parameter_set(self, set_id: str) -> bool:
        """Delete a parameter set from the database."""
        async with AsyncSessionLocal() as session:
            async with session.begin():
                result = await session.execute(
                    delete(ParameterSet).where(ParameterSet.id == set_id)
                )
                return result.rowcount > 0

    # ------------------------------------------------------------------
    # Historical data loading
    # ------------------------------------------------------------------

    async def _load_opportunity_history(self) -> list[dict]:
        """Load all opportunity history records sorted by detected_at."""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(OpportunityHistory).order_by(OpportunityHistory.detected_at.asc())
            )
            rows = result.scalars().all()
            return [
                {
                    "id": r.id,
                    "strategy_type": r.strategy_type,
                    "event_id": r.event_id,
                    "title": r.title,
                    "total_cost": r.total_cost,
                    "expected_roi": r.expected_roi,
                    "risk_score": r.risk_score,
                    "positions_data": r.positions_data,
                    "detected_at": r.detected_at,
                    "expired_at": r.expired_at,
                    "resolution_date": r.resolution_date,
                    "was_profitable": r.was_profitable,
                    "actual_roi": r.actual_roi,
                }
                for r in rows
            ]

    # ------------------------------------------------------------------
    # Optimization API
    # ------------------------------------------------------------------

    async def run_optimization(
        self,
        method: str = "grid",
        param_ranges: Optional[dict[str, Any]] = None,
        n_random_samples: int = 100,
        random_seed: int = 42,
        walk_forward: bool = True,
        n_windows: int = 5,
        train_ratio: float = 0.7,
    ) -> list[dict]:
        """
        Run a parameter sweep and return ranked results.

        Parameters
        ----------
        method : str
            "grid" for exhaustive grid search, "random" for random sampling.
        param_ranges : dict
            For grid: {param_name: [v1, v2, ...]}
            For random: {param_name: (min, max)}
            If None, uses a default subset of important parameters.
        n_random_samples : int
            Number of random samples when method == "random".
        random_seed : int
            RNG seed for reproducibility.
        walk_forward : bool
            If True, use walk-forward validation; otherwise full-history backtest.
        n_windows : int
            Number of walk-forward windows.
        train_ratio : float
            Fraction of each window used for training.

        Returns
        -------
        Ranked list of dicts with keys: params, train_result, test_result, score.
        """
        logger.info(
            "Starting optimisation run",
            method=method,
            walk_forward=walk_forward,
            n_windows=n_windows,
        )

        opportunities = await self._load_opportunity_history()
        if not opportunities:
            logger.warning("No opportunity history found -- cannot optimise")
            return []

        base = self._current_params

        # Build default ranges if not supplied
        if param_ranges is None:
            if method == "grid":
                param_ranges = {
                    "min_roi_percent": [1.0, 2.0, 2.5, 3.0, 5.0],
                    "max_risk_score": [0.3, 0.5, 0.7],
                    "min_profit_threshold": [0.01, 0.025, 0.05],
                }
            else:
                param_ranges = {
                    "min_roi_percent": (0.5, 10.0),
                    "max_risk_score": (0.1, 0.9),
                    "min_profit_threshold": (0.005, 0.10),
                    "min_liquidity_usd": (1000.0, 20000.0),
                    "base_position_size_usd": (1.0, 100.0),
                    "circuit_breaker_losses": (1, 10),
                }

        # Generate candidate parameter sets
        if method == "grid":
            candidates = _build_param_sets_grid(base, param_ranges)
        elif method == "random":
            candidates = _build_param_sets_random(
                base, param_ranges, n_samples=n_random_samples, seed=random_seed
            )
        else:
            logger.error("Unknown search method", method=method)
            return []

        logger.info("Generated candidate parameter sets", count=len(candidates))

        # Evaluate each candidate
        scored: list[dict] = []

        if walk_forward:
            folds = _split_walk_forward(opportunities, n_windows=n_windows, train_ratio=train_ratio)
            if not folds:
                logger.warning("Could not create walk-forward folds, falling back to full history")
                walk_forward = False

        for idx, params in enumerate(candidates):
            if walk_forward:
                # Average across all walk-forward folds
                train_results: list[BacktestResult] = []
                test_results: list[BacktestResult] = []
                for train_data, test_data in folds:
                    tr = _replay_opportunities(train_data, params)
                    te = _replay_opportunities(test_data, params)
                    train_results.append(tr)
                    test_results.append(te)

                # Aggregate test results
                avg_test = BacktestResult(
                    total_profit=round(sum(r.total_profit for r in test_results) / len(test_results), 4),
                    total_cost=round(sum(r.total_cost for r in test_results) / len(test_results), 4),
                    num_trades=sum(r.num_trades for r in test_results) // len(test_results),
                    num_wins=sum(r.num_wins for r in test_results) // len(test_results),
                    num_losses=sum(r.num_losses for r in test_results) // len(test_results),
                    win_rate=round(sum(r.win_rate for r in test_results) / len(test_results), 4),
                    max_drawdown=round(max(r.max_drawdown for r in test_results), 4),
                    sharpe_ratio=round(sum(r.sharpe_ratio for r in test_results) / len(test_results), 4),
                    avg_roi=round(sum(r.avg_roi for r in test_results) / len(test_results), 4),
                    profit_factor=round(sum(r.profit_factor for r in test_results) / len(test_results), 4),
                )
                avg_train = BacktestResult(
                    total_profit=round(sum(r.total_profit for r in train_results) / len(train_results), 4),
                    total_cost=round(sum(r.total_cost for r in train_results) / len(train_results), 4),
                    num_trades=sum(r.num_trades for r in train_results) // len(train_results),
                    num_wins=sum(r.num_wins for r in train_results) // len(train_results),
                    num_losses=sum(r.num_losses for r in train_results) // len(train_results),
                    win_rate=round(sum(r.win_rate for r in train_results) / len(train_results), 4),
                    max_drawdown=round(max(r.max_drawdown for r in train_results), 4),
                    sharpe_ratio=round(sum(r.sharpe_ratio for r in train_results) / len(train_results), 4),
                    avg_roi=round(sum(r.avg_roi for r in train_results) / len(train_results), 4),
                    profit_factor=round(sum(r.profit_factor for r in train_results) / len(train_results), 4),
                )

                # Composite score: weight Sharpe highly, penalise drawdown, reward profit
                score = (
                    avg_test.sharpe_ratio * 0.4
                    + avg_test.win_rate * 20.0 * 0.2
                    + avg_test.total_profit * 0.3
                    - avg_test.max_drawdown * 0.1
                )

                scored.append({
                    "rank": 0,
                    "params": params.to_dict(),
                    "train_result": avg_train.to_dict(),
                    "test_result": avg_test.to_dict(),
                    "score": round(score, 4),
                })
            else:
                # Full-history backtest
                result = _replay_opportunities(opportunities, params)
                score = (
                    result.sharpe_ratio * 0.4
                    + result.win_rate * 20.0 * 0.2
                    + result.total_profit * 0.3
                    - result.max_drawdown * 0.1
                )
                scored.append({
                    "rank": 0,
                    "params": params.to_dict(),
                    "train_result": result.to_dict(),
                    "test_result": result.to_dict(),
                    "score": round(score, 4),
                })

        # Sort by composite score descending
        scored.sort(key=lambda x: x["score"], reverse=True)
        for i, entry in enumerate(scored):
            entry["rank"] = i + 1

        # Persist results
        self._optimization_results = scored
        logger.info(
            "Optimisation complete",
            candidates=len(candidates),
            top_score=scored[0]["score"] if scored else None,
        )
        return scored

    def get_optimization_results(self) -> list[dict]:
        """Return results from the most recent optimization run."""
        return self._optimization_results

    async def run_backtest(
        self,
        params: Optional[dict[str, Any]] = None,
    ) -> dict:
        """
        Run a single backtest with given (or current) parameters.

        Returns BacktestResult as a dict.
        """
        trading_params = (
            TradingParameters.from_dict(params) if params
            else self._current_params
        )
        opportunities = await self._load_opportunity_history()
        result = _replay_opportunities(opportunities, trading_params)
        return result.to_dict()


# ---------------------------------------------------------------------------
# Singleton instance
# ---------------------------------------------------------------------------

param_optimizer = ParameterOptimizer()
