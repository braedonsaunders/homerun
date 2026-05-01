"""Portfolio-level correlation across strategies.

For each strategy_key with terminal trades in a window, build a daily
PnL series and compute the pairwise Pearson correlation matrix.
Diversified portfolios have low average pairwise correlation; runs
that all rely on the same regime show up as high red blocks.

This complements the per-strategy backtest by surfacing the
PORTFOLIO-level question: "even if each strategy looks fine alone, do
they drown together?"
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import AsyncSessionLocal, TraderOrder


logger = logging.getLogger("backtest.portfolio_correlation")


@dataclass
class PortfolioCorrelationResult:
    window_days: int
    strategies: list[str]
    pnl_series_by_strategy: dict[str, dict[str, float]]  # strategy → {YYYY-MM-DD: pnl}
    correlation_matrix: list[list[float]]
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_days": self.window_days,
            "strategies": list(self.strategies),
            "pnl_series_by_strategy": {k: dict(v) for k, v in self.pnl_series_by_strategy.items()},
            "correlation_matrix": [list(row) for row in self.correlation_matrix],
            "summary": dict(self.summary),
        }


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation; returns 0.0 on insufficient data instead of
    NaN so the heatmap doesn't have holes."""
    n = len(xs)
    if n < 3 or len(ys) != n:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx <= 0 or sy <= 0:
        return 0.0
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    return float(cov / (sx * sy))


async def compute_portfolio_correlation(
    *,
    window_days: int = 30,
    min_strategy_trades: int = 5,
    session: AsyncSession | None = None,
) -> PortfolioCorrelationResult:
    """Build the cross-strategy daily-PnL correlation matrix over the
    last ``window_days``.  Uses TraderOrder.realized_pnl_usd captured on
    terminal status transitions (closed_*, resolved_*).

    Strategies with fewer than ``min_strategy_trades`` terminal events
    in the window are excluded — correlations on tiny series are
    statistical noise.
    """
    own_session = session is None
    if own_session:
        session = AsyncSessionLocal()
        await session.__aenter__()
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, window_days))
        FILL_STATUSES = {"closed_win", "closed_loss", "resolved", "resolved_win", "resolved_loss"}

        result = await session.execute(
            select(
                TraderOrder.strategy_key,
                TraderOrder.payload_json,
                TraderOrder.executed_at,
                TraderOrder.updated_at,
                TraderOrder.created_at,
                TraderOrder.status,
            ).where(TraderOrder.created_at >= cutoff)
        )

        # strategy → date_iso → cumulative pnl_usd
        by_strategy_date: dict[str, dict[str, float]] = {}
        trade_counts: dict[str, int] = {}
        for row in result.all():
            strat = str(row.strategy_key or "").strip()
            if not strat:
                continue
            status = str(row.status or "").strip().lower()
            if status not in FILL_STATUSES:
                continue
            payload = row.payload_json or {}
            # PnL key locations, in preference order:
            # 1. top-level realized_pnl_usd / pnl_usd (newer code paths)
            # 2. position_close.realized_pnl (the existing reconciliation
            #    worker's canonical write site — see worker logs).
            pnl_raw = (
                payload.get("realized_pnl_usd")
                or payload.get("pnl_usd")
            )
            if pnl_raw is None:
                close_block = payload.get("position_close")
                if isinstance(close_block, dict):
                    pnl_raw = close_block.get("realized_pnl")
            if pnl_raw is None:
                continue
            try:
                pnl = float(pnl_raw)
            except Exception:
                continue
            ts = row.executed_at or row.updated_at or row.created_at
            if ts is None:
                continue
            ts_aware = ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)
            day_iso = ts_aware.date().isoformat()
            by_strategy_date.setdefault(strat, {})
            by_strategy_date[strat][day_iso] = by_strategy_date[strat].get(day_iso, 0.0) + pnl
            trade_counts[strat] = trade_counts.get(strat, 0) + 1

        # Filter strategies with enough volume.
        usable = sorted(
            [s for s, n in trade_counts.items() if n >= max(1, min_strategy_trades)]
        )

        # Build the union of dates so all series align on the same x-axis.
        all_dates: set[str] = set()
        for s in usable:
            all_dates.update(by_strategy_date.get(s, {}).keys())
        sorted_dates = sorted(all_dates)

        # Materialize each series with 0.0 for missing days (consistent
        # with "no fills that day → no PnL change").
        aligned: dict[str, list[float]] = {}
        for s in usable:
            series = by_strategy_date.get(s, {})
            aligned[s] = [float(series.get(d, 0.0)) for d in sorted_dates]

        # Correlation matrix.
        n = len(usable)
        matrix: list[list[float]] = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                r = _pearson(aligned[usable[i]], aligned[usable[j]])
                matrix[i][j] = r
                matrix[j][i] = r

        # Aggregate metrics.  We compute BOTH signed and absolute mean
        # because raw mean(ρ) hides concentration risk when high
        # positive and high negative correlations cancel.
        off_diag = [matrix[i][j] for i in range(n) for j in range(n) if i != j]
        mean_corr = sum(off_diag) / len(off_diag) if off_diag else 0.0
        mean_abs_corr = (
            sum(abs(r) for r in off_diag) / len(off_diag) if off_diag else 0.0
        )
        max_corr = max(off_diag) if off_diag else 0.0
        min_corr = min(off_diag) if off_diag else 0.0
        # Diversification ratio: 1 - mean(|ρ|).  Captures BOTH highly
        # positive (drawdown together) and highly negative (artificial
        # hedge that's just two halves of the same trade) as
        # concentration.  Bounded [0, 1]; higher = better diversified.
        div_ratio = max(0.0, min(1.0, 1.0 - mean_abs_corr))

        summary = {
            "n_strategies": n,
            "n_days": len(sorted_dates),
            "mean_pairwise_correlation": mean_corr,
            "mean_abs_pairwise_correlation": mean_abs_corr,
            "max_pairwise_correlation": max_corr,
            "min_pairwise_correlation": min_corr,
            "diversification_ratio": div_ratio,
        }

        # Build the per-strategy series payload (subsample to last 30
        # entries to keep the response small but still chartable).
        pnl_series_by_strategy: dict[str, dict[str, float]] = {}
        for s in usable:
            series = by_strategy_date.get(s, {})
            recent_keys = sorted(series.keys())[-30:]
            pnl_series_by_strategy[s] = {k: float(series[k]) for k in recent_keys}

        return PortfolioCorrelationResult(
            window_days=window_days,
            strategies=usable,
            pnl_series_by_strategy=pnl_series_by_strategy,
            correlation_matrix=matrix,
            summary=summary,
        )
    finally:
        if own_session:
            await session.__aexit__(None, None, None)
