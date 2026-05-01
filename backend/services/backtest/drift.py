"""Live-vs-backtest drift monitor.

For each strategy with closed live/shadow trades in the last N days,
compare the realized stats (Sharpe, hit rate, trade rate, return)
against the most recent ``BacktestRun`` for the same strategy.
Flags strategies whose live performance has materially diverged
from their backtest — the cleanest single signal of model decay,
regime shift, or a bug introduced since the last backtest.

This complements walk-forward (which checks consistency *within*
historical windows) by closing the loop with reality: even a
strategy that walk-forward-tests cleanly can quietly drift once
deployed.

Severity levels:
  stable    — |live - backtest| within tolerance for all metrics
  degraded  — live Sharpe < backtest Sharpe - 0.5 OR live trade
              rate < 50% of backtest trade rate
  improved  — live Sharpe > backtest Sharpe + 0.5
  stale     — no recent backtest in the lookback window (operator
              should rerun the backtest)

Output is a list of strategy reports plus a portfolio summary.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import AsyncSessionLocal, BacktestRun, TraderOrder

logger = logging.getLogger("backtest.drift")


_DEGRADED_SHARPE_DELTA = 0.5
_IMPROVED_SHARPE_DELTA = 0.5
_TRADE_RATE_DEGRADED_RATIO = 0.5  # live < 0.5 * backtest_per_unit_time


@dataclass
class StrategyDriftReport:
    strategy_slug: str
    strategy_name: Optional[str]
    severity: str  # stable | degraded | improved | stale
    reason: str
    backtest_run_id: Optional[str]
    backtest_completed_at: Optional[str]
    backtest_window_days: Optional[float]
    backtest_trade_count: int = 0
    backtest_sharpe: Optional[float] = None
    backtest_total_return_pct: Optional[float] = None
    backtest_trades_per_day: Optional[float] = None
    live_window_days: float = 0.0
    live_trade_count: int = 0
    live_sharpe: Optional[float] = None
    live_total_pnl_usd: float = 0.0
    live_hit_rate: Optional[float] = None
    live_trades_per_day: Optional[float] = None
    sharpe_delta: Optional[float] = None
    trade_rate_ratio: Optional[float] = None


@dataclass
class DriftMonitorResult:
    window_days: int
    generated_at: str
    strategies: list[StrategyDriftReport] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_days": self.window_days,
            "generated_at": self.generated_at,
            "strategies": [s.__dict__ for s in self.strategies],
            "summary": self.summary,
        }


def _sharpe_of(pnls: list[float], periods_per_year: int = 252) -> Optional[float]:
    if len(pnls) < 4:
        return None
    mu = sum(pnls) / len(pnls)
    var = sum((p - mu) ** 2 for p in pnls) / max(1, len(pnls) - 1)
    if var <= 0:
        return None
    sigma = math.sqrt(var)
    return (mu / sigma) * math.sqrt(periods_per_year)


async def _fetch_live_pnl_by_strategy(
    *,
    session: AsyncSession,
    cutoff: datetime,
) -> dict[str, list[tuple[datetime, float]]]:
    """Return strategy_slug -> list of (timestamp, pnl_usd) for closed
    trades since cutoff.  Mirrors portfolio_correlation's PnL extraction
    (same payload schema; same status filter)."""
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
    out: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    for row in result.all():
        strat = str(row.strategy_key or "").strip()
        if not strat:
            continue
        status = str(row.status or "").strip().lower()
        if status not in FILL_STATUSES:
            continue
        payload = row.payload_json or {}
        pnl_raw = payload.get("realized_pnl_usd") or payload.get("pnl_usd")
        if pnl_raw is None:
            close_block = payload.get("position_close")
            if isinstance(close_block, dict):
                pnl_raw = close_block.get("realized_pnl")
        if pnl_raw is None:
            continue
        try:
            pnl = float(pnl_raw)
        except (TypeError, ValueError):
            continue
        ts = row.executed_at or row.updated_at or row.created_at
        if ts is None:
            continue
        ts_aware = (
            ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)
        )
        out[strat].append((ts_aware, pnl))
    return out


async def _fetch_latest_backtest_per_slug(
    *,
    session: AsyncSession,
    cutoff: datetime,
) -> dict[str, BacktestRun]:
    """Return strategy_slug -> most recent BacktestRun completed since
    cutoff.  Slugs without a recent backtest will be flagged 'stale'.
    """
    result = await session.execute(
        select(BacktestRun)
        .where(BacktestRun.completed_at >= cutoff, BacktestRun.status == "ok")
        .order_by(desc(BacktestRun.completed_at))
    )
    out: dict[str, BacktestRun] = {}
    for row in result.scalars().all():
        slug = str(row.strategy_slug or "").strip()
        if slug and slug not in out:
            out[slug] = row
    return out


def _classify(
    *,
    backtest: Optional[BacktestRun],
    live_sharpe: Optional[float],
    live_trades_per_day: Optional[float],
    backtest_trades_per_day: Optional[float],
    live_trade_count: int,
) -> tuple[str, str, Optional[float], Optional[float]]:
    """Return (severity, reason, sharpe_delta, trade_rate_ratio)."""
    if backtest is None:
        return "stale", "no recent backtest in lookback window", None, None
    if live_trade_count < 4:
        return "stale", f"only {live_trade_count} live closed trades — too few to assess", None, None

    bt_payload = backtest.result_json or {}
    bt_exec = bt_payload.get("execution") or {}
    bt_sharpe_obj = bt_exec.get("sharpe") or {}
    bt_sharpe = bt_sharpe_obj.get("value") if isinstance(bt_sharpe_obj, dict) else bt_sharpe_obj
    if not isinstance(bt_sharpe, (int, float)):
        bt_sharpe = None

    delta = (
        float(live_sharpe) - float(bt_sharpe)
        if (live_sharpe is not None and bt_sharpe is not None)
        else None
    )
    rate_ratio = (
        float(live_trades_per_day) / float(backtest_trades_per_day)
        if (live_trades_per_day is not None and backtest_trades_per_day and backtest_trades_per_day > 0)
        else None
    )

    if delta is not None and delta < -_DEGRADED_SHARPE_DELTA:
        return (
            "degraded",
            f"live Sharpe {live_sharpe:.2f} below backtest {bt_sharpe:.2f} by {abs(delta):.2f}",
            delta,
            rate_ratio,
        )
    if rate_ratio is not None and rate_ratio < _TRADE_RATE_DEGRADED_RATIO:
        return (
            "degraded",
            f"live trade rate is {rate_ratio*100:.0f}% of backtest rate — strategy under-firing",
            delta,
            rate_ratio,
        )
    if delta is not None and delta > _IMPROVED_SHARPE_DELTA:
        return (
            "improved",
            f"live Sharpe {live_sharpe:.2f} exceeds backtest {bt_sharpe:.2f} by {delta:.2f}",
            delta,
            rate_ratio,
        )
    return ("stable", "live performance tracks backtest within tolerance", delta, rate_ratio)


async def compute_drift(
    *,
    window_days: int = 30,
    session: AsyncSession | None = None,
) -> DriftMonitorResult:
    """Compute the drift report for every strategy with closed trades
    or recent backtests in the last ``window_days``.
    """
    own_session = session is None
    if own_session:
        session = AsyncSessionLocal()
        await session.__aenter__()
    try:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=max(1, window_days))

        live_by_strat = await _fetch_live_pnl_by_strategy(session=session, cutoff=cutoff)
        backtests_by_slug = await _fetch_latest_backtest_per_slug(
            session=session, cutoff=cutoff
        )

        all_slugs = set(live_by_strat.keys()) | set(backtests_by_slug.keys())
        reports: list[StrategyDriftReport] = []

        for slug in sorted(all_slugs):
            live_trades = live_by_strat.get(slug, [])
            live_count = len(live_trades)
            live_pnls = [p for _, p in live_trades]
            live_sharpe = _sharpe_of(live_pnls)
            live_total_pnl = sum(live_pnls)
            live_wins = sum(1 for p in live_pnls if p > 0)
            live_hit_rate = (live_wins / live_count) if live_count > 0 else None
            live_trades_per_day = (
                live_count / float(window_days) if window_days > 0 else None
            )

            backtest = backtests_by_slug.get(slug)
            bt_run_id: Optional[str] = None
            bt_completed: Optional[str] = None
            bt_window_days: Optional[float] = None
            bt_trade_count = 0
            bt_sharpe: Optional[float] = None
            bt_return_pct: Optional[float] = None
            bt_trades_per_day: Optional[float] = None
            bt_name = None

            if backtest is not None:
                bt_run_id = backtest.id
                bt_completed = backtest.completed_at.isoformat() if backtest.completed_at else None
                bt_name = backtest.strategy_name
                bt_payload = backtest.result_json or {}
                bt_exec = bt_payload.get("execution") or {}
                start_iso = bt_exec.get("start_iso")
                end_iso = bt_exec.get("end_iso")
                if start_iso and end_iso:
                    try:
                        s = datetime.fromisoformat(str(start_iso).replace("Z", "+00:00"))
                        e = datetime.fromisoformat(str(end_iso).replace("Z", "+00:00"))
                        bt_window_days = (e - s).total_seconds() / 86400.0
                    except (TypeError, ValueError):
                        pass
                bt_trade_count = int(bt_exec.get("trade_count") or backtest.trade_count or 0)
                bt_sharpe_obj = bt_exec.get("sharpe") or {}
                if isinstance(bt_sharpe_obj, dict):
                    v = bt_sharpe_obj.get("value")
                    if isinstance(v, (int, float)):
                        bt_sharpe = float(v)
                elif isinstance(bt_sharpe_obj, (int, float)):
                    bt_sharpe = float(bt_sharpe_obj)
                bt_return_pct = float(backtest.total_return_pct or 0.0)
                if bt_window_days and bt_window_days > 0 and bt_trade_count > 0:
                    bt_trades_per_day = bt_trade_count / bt_window_days

            severity, reason, delta, rate_ratio = _classify(
                backtest=backtest,
                live_sharpe=live_sharpe,
                live_trades_per_day=live_trades_per_day,
                backtest_trades_per_day=bt_trades_per_day,
                live_trade_count=live_count,
            )

            reports.append(
                StrategyDriftReport(
                    strategy_slug=slug,
                    strategy_name=bt_name,
                    severity=severity,
                    reason=reason,
                    backtest_run_id=bt_run_id,
                    backtest_completed_at=bt_completed,
                    backtest_window_days=bt_window_days,
                    backtest_trade_count=bt_trade_count,
                    backtest_sharpe=bt_sharpe,
                    backtest_total_return_pct=bt_return_pct,
                    backtest_trades_per_day=bt_trades_per_day,
                    live_window_days=float(window_days),
                    live_trade_count=live_count,
                    live_sharpe=live_sharpe,
                    live_total_pnl_usd=round(live_total_pnl, 2),
                    live_hit_rate=live_hit_rate,
                    live_trades_per_day=live_trades_per_day,
                    sharpe_delta=delta,
                    trade_rate_ratio=rate_ratio,
                )
            )

        # Portfolio summary: counts by severity + worst offender.
        by_severity: dict[str, int] = {"stable": 0, "degraded": 0, "improved": 0, "stale": 0}
        for r in reports:
            by_severity[r.severity] = by_severity.get(r.severity, 0) + 1
        degraded = [r for r in reports if r.severity == "degraded"]
        worst = (
            min(degraded, key=lambda r: (r.sharpe_delta if r.sharpe_delta is not None else 0.0))
            if degraded
            else None
        )

        summary = {
            "n_strategies": len(reports),
            "by_severity": by_severity,
            "worst_offender": (
                {
                    "strategy_slug": worst.strategy_slug,
                    "sharpe_delta": worst.sharpe_delta,
                    "reason": worst.reason,
                }
                if worst is not None
                else None
            ),
        }

        return DriftMonitorResult(
            window_days=int(window_days),
            generated_at=now.isoformat(),
            strategies=reports,
            summary=summary,
        )
    finally:
        if own_session:
            await session.__aexit__(None, None, None)
