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

Purging + embargo: each selected test fold is run as its OWN disjoint
backtest window and the per-fold returns are pooled — NOT as a single
outer span from the first test fold to the last.  The outer-span approach
silently fed interior non-test folds into the evaluation (a leak); running
folds separately confines each path strictly to its test data.  Each test
fold's left edge is additionally trimmed by ``embargo_seconds`` to
decorrelate it from the prior fold's tail (positions are force-settled at
each fold's right edge, so none carries across a fold boundary).

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

from services.backtest.metrics import (
    ANN_CALENDAR_SECONDS,
    infer_periods_per_year,
    resample_equity_returns,
    sharpe_of_returns,
    sortino_of_returns,
)

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


def _test_fold_windows(
    *,
    folds: list[CPCVFold],
    test_indices: tuple[int, ...],
    embargo_seconds: float,
) -> list[tuple[datetime, datetime]]:
    """Return the DISJOINT ``[start, end]`` window for each selected test
    fold (not the outer span), each trimmed at the left by
    ``embargo_seconds`` to decorrelate from the prior fold's tail.

    Running each test fold separately is what prevents interior non-test
    folds from leaking into the evaluation.
    """
    emb = timedelta(seconds=max(0.0, embargo_seconds))
    windows: list[tuple[datetime, datetime]] = []
    for i in sorted(set(test_indices)):
        f = folds[i]
        start = (f.start + emb) if i > 0 else f.start
        if start < f.end:
            windows.append((start, f.end))
    return windows


def _curve_from_result(d: dict[str, Any]) -> list[tuple[datetime, float]]:
    """Extract a ``(timestamp, equity_usd)`` series from an execution
    backtest result's downsampled equity curve."""
    out: list[tuple[datetime, float]] = []
    for p in d.get("equity_curve_sample") or []:
        if not (isinstance(p, dict) and isinstance(p.get("equity_usd"), (int, float))):
            continue
        at = p.get("at")
        ts: Optional[datetime] = None
        if isinstance(at, str):
            try:
                ts = datetime.fromisoformat(at)
            except ValueError:
                ts = None
        if ts is not None:
            out.append((ts, float(p["equity_usd"])))
    return out


def _summarize_paths(paths: list[CPCVPath]) -> dict[str, Any]:
    succeeded = [p for p in paths if p.success]
    sharpes = [p.sharpe for p in succeeded if p.sharpe is not None]
    returns = [p.total_return_pct for p in succeeded]
    dd = [p.max_drawdown_pct for p in succeeded]

    # Classic López de Prado PBO requires an IS-best-config selection across
    # a parameter search; single-config CPCV has no config dimension, so
    # reporting a "PBO" here would be a mislabel.  We report path robustness
    # (fraction of paths with negative OOS Sharpe) instead; config-level
    # overfitting is gated separately by the deflated Sharpe in autoresearch.
    fraction_negative_sharpe = (
        sum(1 for s in sharpes if s < 0) / len(sharpes) if sharpes else None
    )

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
        "fraction_negative_sharpe": fraction_negative_sharpe,
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

    def _exec_kwargs_for(w_start: datetime, w_end: datetime) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "source_code": source_code,
            "slug": slug,
            "config": config,
            "token_ids": token_ids,
            "start": w_start,
            "end": w_end,
            "initial_capital_usd": initial_capital_usd,
        }
        if submit_p50_ms is not None:
            kw["submit_latency_p50_ms"] = float(submit_p50_ms)
        if submit_p95_ms is not None:
            kw["submit_latency_p95_ms"] = float(submit_p95_ms)
        if cancel_p50_ms is not None:
            kw["cancel_latency_p50_ms"] = float(cancel_p50_ms)
        if cancel_p95_ms is not None:
            kw["cancel_latency_p95_ms"] = float(cancel_p95_ms)
        if seed is not None:
            kw["seed"] = int(seed)
        return kw

    async def _run_one(idx: int, test_indices: tuple[int, ...]) -> CPCVPath:
        async with semaphore:
            windows = _test_fold_windows(
                folds=folds, test_indices=test_indices, embargo_seconds=embargo_seconds
            )
            tindices = tuple(sorted(set(test_indices)))
            if not windows:
                return CPCVPath(
                    path_index=idx, test_fold_indices=tindices,
                    test_start_iso="", test_end_iso="",
                    success=False, runtime_error="no test window after embargo",
                )
            pooled: list[float] = []
            ppy: Optional[int] = None
            n_intents = trade_count = total_fills = 0
            ok = True
            last_err: Optional[str] = None
            # Run EACH test fold as its own disjoint backtest (no outer-span
            # leak) and pool the per-fold returns.
            for (w_start, w_end) in windows:
                try:
                    res: ExecutionBacktestResult = await run_execution_backtest(
                        **_exec_kwargs_for(w_start, w_end)
                    )
                    d = res.to_dict()
                    n_intents += int(d.get("n_intents") or 0)
                    trade_count += int(d.get("trade_count") or 0)
                    total_fills += int(d.get("total_fills") or 0)
                    if not d.get("success"):
                        ok = False
                        last_err = last_err or d.get("runtime_error")
                    hist = _curve_from_result(d)
                    if len(hist) >= 3:
                        fold_ppy = infer_periods_per_year(hist)
                        ppy = fold_ppy if ppy is None else max(ppy, fold_ppy)
                        pooled.extend(
                            resample_equity_returns(
                                hist, bar_seconds=ANN_CALENDAR_SECONDS / max(1, fold_ppy)
                            )
                        )
                except Exception as exc:
                    ok = False
                    last_err = str(exc)
                    logger.exception(
                        "CPCV path %d window [%s, %s] failed", idx, w_start, w_end
                    )
            # Path metrics from the pooled OOS returns (frequency-correct).
            sharpe: Optional[float] = None
            sortino: Optional[float] = None
            total_ret = dd_pct = 0.0
            if pooled and ppy:
                sharpe = sharpe_of_returns(pooled, periods_per_year=ppy)
                sortino = sortino_of_returns(pooled, periods_per_year=ppy)
                comp = peak = 1.0
                worst = 0.0
                for r in pooled:
                    comp *= (1.0 + r)
                    peak = max(peak, comp)
                    if peak > 0:
                        worst = max(worst, (peak - comp) / peak)
                total_ret = (comp - 1.0) * 100.0
                dd_pct = worst * 100.0
            return CPCVPath(
                path_index=idx,
                test_fold_indices=tindices,
                test_start_iso=windows[0][0].isoformat(),
                test_end_iso=windows[-1][1].isoformat(),
                n_intents=n_intents,
                trade_count=trade_count,
                total_fills=total_fills,
                success=ok,
                runtime_error=last_err,
                total_return_pct=total_ret,
                sharpe=sharpe,
                sortino=sortino,
                max_drawdown_pct=dd_pct,
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
