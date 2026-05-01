"""Walk-forward validation harness.

Two split modes:

* **Rolling** — fixed-size in-sample window slides forward, OOS window
  follows. Useful when the regime is locally stationary.
* **Anchored** — in-sample starts at t0 and grows; OOS window follows
  the latest in-sample edge. Useful when older data is still informative.

Each fold reports its own ``BacktestMetrics`` (point estimate + bootstrap
CI). The harness returns the OOS metric distribution so callers can
report cross-fold variance — a stronger guard against overfit than a
single train/test split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from services.backtest.metrics import BacktestMetrics, MetricCI


@dataclass
class WalkForwardConfig:
    mode: str = "rolling"  # "rolling" | "anchored"
    n_folds: int = 5
    train_ratio: float = 0.7  # only used for anchored mode
    embargo_seconds: float = 0.0  # gap between train and test windows


@dataclass
class WalkForwardWindow:
    """One walk-forward fold's in-sample / out-of-sample window."""

    fold_index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


@dataclass
class WalkForwardResult:
    config: WalkForwardConfig
    windows: list[WalkForwardWindow] = field(default_factory=list)
    train_metrics: list[BacktestMetrics] = field(default_factory=list)
    test_metrics: list[BacktestMetrics] = field(default_factory=list)

    def aggregate_test_metric(self, attr: str) -> MetricCI:
        """Aggregate a test-fold metric across folds.

        Returns mean of point estimates with min/max as the band — a
        simpler-than-bootstrap summary of cross-fold variance. Use this
        for ``sharpe``, ``sortino``, ``calmar``, ``hit_rate``, etc. that
        are themselves ``MetricCI`` on each fold.
        """
        values: list[float] = []
        for m in self.test_metrics:
            v = getattr(m, attr, None)
            if isinstance(v, MetricCI):
                values.append(v.value)
            elif isinstance(v, (int, float)):
                values.append(float(v))
        if not values:
            return MetricCI(0.0)
        mean = sum(values) / len(values)
        return MetricCI(mean, min(values), max(values))


def walk_forward_split(
    *,
    start: datetime,
    end: datetime,
    config: Optional[WalkForwardConfig] = None,
) -> list[WalkForwardWindow]:
    """Compute the in-sample/out-of-sample windows for a backtest period.

    Both modes guarantee:
      * No look-ahead: train_end < test_start
      * Optional embargo: test_start = train_end + embargo
      * Test windows are roughly equal-sized
      * Folds are returned in chronological order
    """
    cfg = config or WalkForwardConfig()
    if end <= start:
        raise ValueError(f"invalid range: {start} >= {end}")
    n = max(1, int(cfg.n_folds))
    embargo = timedelta(seconds=max(0.0, float(cfg.embargo_seconds)))

    if cfg.mode == "anchored":
        return _anchored_windows(start, end, n, cfg.train_ratio, embargo)
    return _rolling_windows(start, end, n, embargo)


def _rolling_windows(
    start: datetime,
    end: datetime,
    n_folds: int,
    embargo: timedelta,
) -> list[WalkForwardWindow]:
    """Equal-width rolling windows. Each fold is (train, test) of equal
    size, sliding forward by the test window each fold.
    """
    total = end - start
    fold_w = total / (n_folds + 1)  # +1 so first fold has equal train+test
    windows: list[WalkForwardWindow] = []
    for i in range(n_folds):
        train_start = start + fold_w * i
        train_end = train_start + fold_w
        test_start = train_end + embargo
        test_end = test_start + fold_w
        if test_end > end:
            test_end = end
        if test_start >= test_end:
            continue
        windows.append(
            WalkForwardWindow(
                fold_index=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
    return windows


def _anchored_windows(
    start: datetime,
    end: datetime,
    n_folds: int,
    train_ratio: float,
    embargo: timedelta,
) -> list[WalkForwardWindow]:
    """Anchored windows: train always starts at ``start`` and grows; the
    test window follows the most recent train edge.
    """
    total = end - start
    train_ratio = max(0.1, min(0.95, float(train_ratio)))
    # First fold: train = train_ratio * total, test starts after embargo
    # Each subsequent fold extends train_end by (test_w_increment) and
    # tests on the next slice.
    initial_train = total * train_ratio
    remaining = total - initial_train - embargo
    if remaining.total_seconds() <= 0:
        return []
    test_w = remaining / n_folds
    windows: list[WalkForwardWindow] = []
    for i in range(n_folds):
        train_start = start
        train_end = start + initial_train + test_w * i
        test_start = train_end + embargo
        test_end = test_start + test_w
        if test_end > end:
            test_end = end
        if test_start >= test_end:
            continue
        windows.append(
            WalkForwardWindow(
                fold_index=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
    return windows


__all__ = [
    "WalkForwardConfig",
    "WalkForwardWindow",
    "WalkForwardResult",
    "walk_forward_split",
    "run_walk_forward",
    "WalkForwardRunWindow",
    "WalkForwardRunResult",
]


# ---------------------------------------------------------------------
# High-level runner — turns a (start, end, mode, n_folds) request into
# N actual BacktestEngine runs and aggregates their metrics.  This is
# what the new BacktestStudio walk-forward panel calls.
# ---------------------------------------------------------------------

import asyncio
import logging
from datetime import timezone
from typing import Any, Literal

logger = logging.getLogger("backtest.walk_forward.runner")

WalkForwardMode = Literal["anchored", "rolling"]


@dataclass
class WalkForwardRunWindow:
    """One executed walk-forward fold with its run-time metrics."""

    index: int
    train_start_iso: str
    train_end_iso: str
    test_start_iso: str
    test_end_iso: str
    success: bool
    runtime_error: str | None
    initial_capital_usd: float
    final_equity_usd: float
    total_return_pct: float
    sharpe: float | None
    sortino: float | None
    hit_rate: float | None
    trade_count: int
    total_fills: int
    rejected_orders: int
    cancelled_orders: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "train_start_iso": self.train_start_iso,
            "train_end_iso": self.train_end_iso,
            "test_start_iso": self.test_start_iso,
            "test_end_iso": self.test_end_iso,
            "success": self.success,
            "runtime_error": self.runtime_error,
            "initial_capital_usd": self.initial_capital_usd,
            "final_equity_usd": self.final_equity_usd,
            "total_return_pct": self.total_return_pct,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "hit_rate": self.hit_rate,
            "trade_count": self.trade_count,
            "total_fills": self.total_fills,
            "rejected_orders": self.rejected_orders,
            "cancelled_orders": self.cancelled_orders,
        }


@dataclass
class WalkForwardRunResult:
    mode: WalkForwardMode
    n_windows_run: int
    overall_start_iso: str
    overall_end_iso: str
    windows: list[WalkForwardRunWindow] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "n_windows_run": self.n_windows_run,
            "overall_start_iso": self.overall_start_iso,
            "overall_end_iso": self.overall_end_iso,
            "windows": [w.to_dict() for w in self.windows],
            "summary": dict(self.summary),
        }


def _ci_value(metric: dict[str, Any] | None) -> float | None:
    if not isinstance(metric, dict):
        return None
    v = metric.get("value")
    return float(v) if isinstance(v, (int, float)) else None


async def run_walk_forward(
    *,
    source_code: str,
    slug: str = "_backtest_walk_forward",
    config: dict[str, Any] | None = None,
    token_ids: list[str] | None = None,
    start: datetime,
    end: datetime,
    initial_capital_usd: float = 1000.0,
    mode: WalkForwardMode = "anchored",
    n_folds: int = 6,
    train_ratio: float = 0.5,
    embargo_seconds: float = 0.0,
    submit_p50_ms: float | None = None,
    submit_p95_ms: float | None = None,
    cancel_p50_ms: float | None = None,
    cancel_p95_ms: float | None = None,
    seed: int | None = None,
    concurrency: int = 2,
) -> WalkForwardRunResult:
    """Run walk-forward analysis: split [start, end] into N folds, run
    the strategy on each test window, return per-fold metrics.

    Reuses ``walk_forward_split`` for the chronological window
    geometry, then dispatches each fold to ``run_execution_backtest``.
    Folds run concurrently up to ``concurrency``.

    The "training" of the strategy is implicit — strategies are
    deterministic given config, so the train/test labels in the
    output describe what data the strategy could have seen via state
    before the test window started, not separate fitting passes.  The
    actual evaluation is just running the engine on the test window.
    """
    # Lazy import to avoid a circular dep with services.strategy_backtester
    # (which imports from this package).
    from services.strategy_backtester import (
        ExecutionBacktestResult,
        run_execution_backtest,
    )

    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    split_cfg = WalkForwardConfig(
        mode=mode,
        n_folds=n_folds,
        train_ratio=train_ratio,
        embargo_seconds=embargo_seconds,
    )
    fold_windows = walk_forward_split(start=start, end=end, config=split_cfg)

    semaphore = asyncio.Semaphore(max(1, int(concurrency)))

    async def _run_one(fold: WalkForwardWindow) -> WalkForwardRunWindow:
        async with semaphore:
            try:
                exec_result: ExecutionBacktestResult = await run_execution_backtest(
                    source_code=source_code,
                    slug=f"{slug}_w{fold.fold_index}",
                    config=config,
                    token_ids=token_ids,
                    start=fold.test_start,
                    end=fold.test_end,
                    initial_capital_usd=initial_capital_usd,
                    submit_p50_ms=submit_p50_ms,
                    submit_p95_ms=submit_p95_ms,
                    cancel_p50_ms=cancel_p50_ms,
                    cancel_p95_ms=cancel_p95_ms,
                    seed=seed,
                )
                d = exec_result.to_dict()
                return WalkForwardRunWindow(
                    index=fold.fold_index,
                    train_start_iso=fold.train_start.isoformat(),
                    train_end_iso=fold.train_end.isoformat(),
                    test_start_iso=fold.test_start.isoformat(),
                    test_end_iso=fold.test_end.isoformat(),
                    success=bool(d.get("success")),
                    runtime_error=d.get("runtime_error"),
                    initial_capital_usd=float(d.get("initial_capital_usd") or 0.0),
                    final_equity_usd=float(d.get("final_equity_usd") or 0.0),
                    total_return_pct=float(d.get("total_return_pct") or 0.0),
                    sharpe=_ci_value(d.get("sharpe")),
                    sortino=_ci_value(d.get("sortino")),
                    hit_rate=_ci_value(d.get("hit_rate")),
                    trade_count=int(d.get("trade_count") or 0),
                    total_fills=int(d.get("total_fills") or 0),
                    rejected_orders=int(d.get("rejected_orders") or 0),
                    cancelled_orders=int(d.get("cancelled_orders") or 0),
                )
            except Exception as exc:
                logger.exception("Walk-forward fold %d failed", fold.fold_index)
                return WalkForwardRunWindow(
                    index=fold.fold_index,
                    train_start_iso=fold.train_start.isoformat(),
                    train_end_iso=fold.train_end.isoformat(),
                    test_start_iso=fold.test_start.isoformat(),
                    test_end_iso=fold.test_end.isoformat(),
                    success=False,
                    runtime_error=str(exc),
                    initial_capital_usd=float(initial_capital_usd),
                    final_equity_usd=0.0,
                    total_return_pct=0.0,
                    sharpe=None,
                    sortino=None,
                    hit_rate=None,
                    trade_count=0,
                    total_fills=0,
                    rejected_orders=0,
                    cancelled_orders=0,
                )

    completed: list[WalkForwardRunWindow] = await asyncio.gather(
        *[_run_one(fold) for fold in fold_windows]
    )

    returns = [w.total_return_pct for w in completed if w.success]
    sharpes = [w.sharpe for w in completed if w.sharpe is not None]
    summary = {
        "n_windows_run": len(completed),
        "n_windows_succeeded": sum(1 for w in completed if w.success),
        "mean_return_pct": float(sum(returns) / len(returns)) if returns else 0.0,
        "min_return_pct": float(min(returns)) if returns else 0.0,
        "max_return_pct": float(max(returns)) if returns else 0.0,
        "stable_window_pct": (sum(1 for r in returns if r >= 0) / len(returns) * 100.0) if returns else 0.0,
        "mean_sharpe": float(sum(sharpes) / len(sharpes)) if sharpes else None,
        "min_sharpe": float(min(sharpes)) if sharpes else None,
        "max_sharpe": float(max(sharpes)) if sharpes else None,
    }

    return WalkForwardRunResult(
        mode=mode,
        n_windows_run=len(completed),
        overall_start_iso=start.isoformat(),
        overall_end_iso=end.isoformat(),
        windows=completed,
        summary=summary,
    )
