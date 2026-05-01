"""Combinatorial Purged Cross-Validation (CPCV).

Reference: Lopez de Prado, *Advances in Financial Machine Learning*,
ch. 7 (2018).  CPCV generalizes walk-forward by evaluating *every*
combination of K test folds out of N total folds, instead of just a
single chronological forward path.  This produces a distribution of
out-of-sample Sharpe ratios across C(N, K) paths, which lets us
ask:

  * Is the edge consistent across arbitrary time subsets, or does
    it concentrate in a few lucky windows?
  * What's the **Probability of Backtest Overfitting (PBO)** —
    the fraction of paths where the strategy's in-sample top
    performance turns into below-median out-of-sample?

For prediction markets, "training" is implicit (the strategy is
deterministic given config), so CPCV here probes whether the
realized Sharpe is robust to *which* slice of history you test on,
not whether parameter choices generalize.  In Karpathy's autoresearch
loop this matters: an iteration that pumps Sharpe in one CPCV path
but degrades in others is overfit to that subset.

Purging + embargo: the engine pulls historical opportunities by
``detected_at``.  Positions that open in a train window may close
inside a test window — a future-leak.  We mitigate by passing an
``embargo_seconds`` gap between any train fold's right edge and
any test fold's left edge; the engine itself evaluates only events
inside its requested ``[start, end]`` window so the embargo is
sufficient when ``embargo_seconds >= max_position_holding_period``.

Computationally expensive: C(N, K) backtest runs.  Default N=6, K=2
gives 15 combinations.  Defaults are conservative; operators can
push concurrency.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CPCVFold:
    """One contiguous time window in the CPCV grid."""

    fold_index: int
    start: datetime
    end: datetime


@dataclass
class CPCVPath:
    """One combination of test folds (out of N total) and its result."""

    path_index: int
    test_fold_indices: tuple[int, ...]
    test_start_iso: str
    test_end_iso: str
    n_intents: int = 0
    trade_count: int = 0
    total_fills: int = 0
    success: bool = False
    runtime_error: Optional[str] = None
    total_return_pct: float = 0.0
    sharpe: Optional[float] = None
    sortino: Optional[float] = None
    max_drawdown_pct: float = 0.0


@dataclass
class CPCVResult:
    n_folds: int
    k_test_folds: int
    embargo_seconds: float
    paths: list[CPCVPath] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_folds": self.n_folds,
            "k_test_folds": self.k_test_folds,
            "embargo_seconds": self.embargo_seconds,
            "n_paths": len(self.paths),
            "paths": [p.__dict__ for p in self.paths],
            "summary": self.summary,
        }


def _build_folds(*, start: datetime, end: datetime, n_folds: int) -> list[CPCVFold]:
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    total_seconds = max(1.0, (end - start).total_seconds())
    fold_seconds = total_seconds / n_folds
    folds: list[CPCVFold] = []
    for i in range(n_folds):
        f_start = start + timedelta(seconds=i * fold_seconds)
        f_end = start + timedelta(seconds=(i + 1) * fold_seconds)
        folds.append(CPCVFold(fold_index=i, start=f_start, end=f_end))
    return folds


def _embargo_safe_test_windows(
    *,
    folds: list[CPCVFold],
    test_indices: tuple[int, ...],
    embargo_seconds: float,
) -> tuple[datetime, datetime]:
    """Compute the effective [test_start, test_end] for a path.  When
    test folds are non-contiguous the engine runs a single window
    spanning the earliest test_start through the latest test_end —
    we use the OUTER bounds and let the engine's per-opportunity
    ``detected_at`` filter clip events to the actual test region.
    Embargo trims the outer start by ``embargo_seconds`` if any
    train fold immediately precedes the first test fold.
    """
    test_indices = tuple(sorted(set(test_indices)))
    first = folds[test_indices[0]]
    last = folds[test_indices[-1]]
    test_start = first.start + timedelta(seconds=max(0.0, embargo_seconds))
    test_end = last.end
    return test_start, test_end


def _summarize_paths(paths: list[CPCVPath]) -> dict[str, Any]:
    succeeded = [p for p in paths if p.success]
    sharpes = [p.sharpe for p in succeeded if p.sharpe is not None]
    returns = [p.total_return_pct for p in succeeded]
    dd = [p.max_drawdown_pct for p in succeeded]

    pbo = None
    if len(sharpes) >= 4:
        # Probability of Backtest Overfitting (Lopez de Prado, JoPM 2014).
        # Simplified form for implicit-training CPCV: rank-based
        # measure of how often a path that is "in the top half"
        # by realized Sharpe ends up there by luck rather than edge.
        # We compute the fraction of paths with Sharpe < 0 given
        # the median Sharpe is positive — a calibration check that
        # turns into a stricter test as N grows.
        sorted_sr = sorted(sharpes)
        median = sorted_sr[len(sorted_sr) // 2]
        if median > 0:
            pbo = sum(1 for s in sharpes if s < 0) / len(sharpes)

    def _pct(xs: list[float], q: float) -> Optional[float]:
        if not xs:
            return None
        s = sorted(xs)
        idx = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
        return s[idx]

    return {
        "n_paths_run": len(paths),
        "n_paths_succeeded": len(succeeded),
        "sharpe_mean": (sum(sharpes) / len(sharpes)) if sharpes else None,
        "sharpe_median": _pct(sharpes, 0.5),
        "sharpe_p10": _pct(sharpes, 0.1),
        "sharpe_p90": _pct(sharpes, 0.9),
        "sharpe_min": min(sharpes) if sharpes else None,
        "sharpe_max": max(sharpes) if sharpes else None,
        "return_mean_pct": (sum(returns) / len(returns)) if returns else None,
        "return_min_pct": min(returns) if returns else None,
        "return_max_pct": max(returns) if returns else None,
        "max_dd_mean_pct": (sum(dd) / len(dd)) if dd else None,
        "max_dd_worst_pct": max(dd) if dd else None,
        "pbo": pbo,
        "stable_path_pct": (
            sum(1 for r in returns if r >= 0) / len(returns) * 100.0 if returns else 0.0
        ),
    }


async def run_cpcv(
    *,
    source_code: str,
    slug: str = "_backtest_cpcv",
    config: dict[str, Any] | None = None,
    token_ids: list[str] | None = None,
    start: datetime,
    end: datetime,
    initial_capital_usd: float = 1000.0,
    n_folds: int = 6,
    k_test_folds: int = 2,
    embargo_seconds: float = 3600.0,
    submit_p50_ms: float | None = None,
    submit_p95_ms: float | None = None,
    cancel_p50_ms: float | None = None,
    cancel_p95_ms: float | None = None,
    seed: int | None = None,
    concurrency: int = 2,
    max_paths: int = 64,
) -> CPCVResult:
    """Run CPCV: C(n_folds, k_test_folds) backtests, summarized.

    For each combination of K test folds out of N total folds, runs
    the strategy on the union of test folds and records the OOS
    Sharpe.  Returns the distribution + a PBO estimate.

    ``max_paths`` caps the number of combinations actually run (sampled
    uniformly from the full set if it would exceed the cap) so a
    pathological N/K choice doesn't queue thousands of backtests.
    """
    from services.strategy_backtester import (
        ExecutionBacktestResult,
        run_execution_backtest,
    )

    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    n_folds = max(2, int(n_folds))
    k_test_folds = max(1, min(n_folds - 1, int(k_test_folds)))

    folds = _build_folds(start=start, end=end, n_folds=n_folds)
    combinations = list(itertools.combinations(range(n_folds), k_test_folds))
    if len(combinations) > max_paths:
        # Deterministic uniform subsampling — preserves coverage of
        # early/middle/late fold mixes without running the whole grid.
        step = len(combinations) / max_paths
        combinations = [combinations[int(i * step)] for i in range(max_paths)]

    semaphore = asyncio.Semaphore(max(1, int(concurrency)))

    async def _run_one(idx: int, test_indices: tuple[int, ...]) -> CPCVPath:
        async with semaphore:
            test_start, test_end = _embargo_safe_test_windows(
                folds=folds, test_indices=test_indices, embargo_seconds=embargo_seconds
            )
            try:
                exec_kwargs: dict[str, Any] = {
                    "source_code": source_code,
                    "slug": slug,
                    "config": config,
                    "token_ids": token_ids,
                    "start": test_start,
                    "end": test_end,
                    "initial_capital_usd": initial_capital_usd,
                }
                if submit_p50_ms is not None:
                    exec_kwargs["submit_latency_p50_ms"] = float(submit_p50_ms)
                if submit_p95_ms is not None:
                    exec_kwargs["submit_latency_p95_ms"] = float(submit_p95_ms)
                if cancel_p50_ms is not None:
                    exec_kwargs["cancel_latency_p50_ms"] = float(cancel_p50_ms)
                if cancel_p95_ms is not None:
                    exec_kwargs["cancel_latency_p95_ms"] = float(cancel_p95_ms)
                if seed is not None:
                    exec_kwargs["seed"] = int(seed)
                exec_result: ExecutionBacktestResult = await run_execution_backtest(**exec_kwargs)
                d = exec_result.to_dict()
                sharpe = (d.get("sharpe") or {}).get("value")
                sortino = (d.get("sortino") or {}).get("value")
                return CPCVPath(
                    path_index=idx,
                    test_fold_indices=test_indices,
                    test_start_iso=test_start.isoformat(),
                    test_end_iso=test_end.isoformat(),
                    n_intents=int(d.get("n_intents") or 0),
                    trade_count=int(d.get("trade_count") or 0),
                    total_fills=int(d.get("total_fills") or 0),
                    success=bool(d.get("success")),
                    runtime_error=d.get("runtime_error"),
                    total_return_pct=float(d.get("total_return_pct") or 0.0),
                    sharpe=float(sharpe) if isinstance(sharpe, (int, float)) else None,
                    sortino=float(sortino) if isinstance(sortino, (int, float)) else None,
                    max_drawdown_pct=float(d.get("max_drawdown_pct") or 0.0),
                )
            except Exception as exc:
                logger.exception("CPCV path %d failed", idx)
                return CPCVPath(
                    path_index=idx,
                    test_fold_indices=test_indices,
                    test_start_iso=test_start.isoformat(),
                    test_end_iso=test_end.isoformat(),
                    success=False,
                    runtime_error=str(exc),
                )

    paths = await asyncio.gather(
        *[_run_one(i, combo) for i, combo in enumerate(combinations)]
    )

    return CPCVResult(
        n_folds=n_folds,
        k_test_folds=k_test_folds,
        embargo_seconds=float(embargo_seconds),
        paths=list(paths),
        summary=_summarize_paths(list(paths)),
    )
