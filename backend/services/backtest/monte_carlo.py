"""Monte Carlo sensitivity analysis for backtests.

Two complementary tests:

1. **Trade-order shuffle (cheap, in-process)** — take the realized
   trade ledger from a single backtest run and shuffle the pnl
   sequence many times.  Each shuffle yields a synthetic equity
   curve from which we recompute Sharpe.  The distribution answers:
   "is the observed Sharpe an artifact of the specific sequence
   (lucky timing), or does it survive when we rearrange the same
   trades?".  A small spread relative to the realized value means
   sequence didn't matter — the result is robust.  A large spread
   means a few well-placed wins are doing the heavy lifting.

2. **Latency-perturbation (expensive, multi-run)** — re-run the
   backtest K times with different latency profiles (the same
   strategy, different network-state assumptions).  Default grid:
   p95 multipliers of [0.5, 1.0, 1.5, 2.0] — bracketing "fiber day"
   through "RPC-saturated peak hour".  Reports Sharpe at each
   multiplier so the operator can see the latency-edge curve and
   estimate how a network regression would erode realized PnL.

The trade-order shuffle is computed automatically on every unified
run and surfaced in BacktestStudio's Robustness subtab.  The
latency perturbation is opt-in via POST /backtest/monte-carlo —
running the engine N times is multi-minute work.
"""
from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Trade-order shuffle (cheap, in-process) ───────────────────────────────


def trade_order_monte_carlo(
    *,
    trade_pnls_usd: list[float],
    n_resamples: int = 2000,
    seed: int | None = 42,
) -> dict[str, Any]:
    """Shuffle the trade pnl sequence ``n_resamples`` times and report
    the distribution of resulting Sharpe ratios.

    The realized backtest produces a specific equity path; this asks
    what would have happened if the same set of trades had occurred
    in a different order.  Robust strategies have tight distributions
    (sequence didn't matter); fragile strategies have wide ones (a
    few well-placed wins did all the work).

    Returns ``{n_resamples, sharpe_distribution: {mean, p5, p25, p50,
    p75, p95}, observed_vs_distribution: {position_pct, z_score}}``.
    """
    if not trade_pnls_usd:
        return {
            "n_resamples": 0,
            "sharpe_distribution": {},
            "observed_vs_distribution": None,
        }

    pnls = list(trade_pnls_usd)
    rng = random.Random(seed)

    def _sharpe_from_pnl_sequence(pnl_seq: list[float]) -> float:
        # Synthesize period returns from cumulative pnl over starting
        # capital of 1.0 (so returns are pnl-fraction of unit equity).
        # The shape of the distribution matters, not the absolute
        # value, because we compare against the realized Sharpe.
        if not pnl_seq:
            return 0.0
        equity = 1.0
        equities = [equity]
        for p in pnl_seq:
            equity += p
            if equity <= 0:
                equities.append(0.001)  # floor — bankrupt sequence
                continue
            equities.append(equity)
        rets = []
        for i in range(1, len(equities)):
            prev = equities[i - 1]
            if prev > 0:
                rets.append((equities[i] - prev) / prev)
        if len(rets) < 2:
            return 0.0
        mu = sum(rets) / len(rets)
        var = sum((r - mu) ** 2 for r in rets) / max(1, len(rets) - 1)
        if var <= 0:
            return 0.0
        sigma = var ** 0.5
        return (mu / sigma) * (252 ** 0.5)

    realized = _sharpe_from_pnl_sequence(pnls)
    samples: list[float] = []
    for _ in range(int(n_resamples)):
        shuffled = list(pnls)
        rng.shuffle(shuffled)
        samples.append(_sharpe_from_pnl_sequence(shuffled))
    if not samples:
        return {
            "n_resamples": 0,
            "sharpe_distribution": {},
            "observed_vs_distribution": None,
        }
    samples.sort()

    def _q(p: float) -> float:
        idx = max(0, min(len(samples) - 1, int(round(p * (len(samples) - 1)))))
        return samples[idx]

    mean = sum(samples) / len(samples)
    var = sum((x - mean) ** 2 for x in samples) / max(1, len(samples) - 1)
    sigma = var ** 0.5
    z = (realized - mean) / sigma if sigma > 0 else 0.0
    # Where does the realized Sharpe sit in the shuffle distribution?
    # 1.0 = better than every shuffle, 0.0 = worse than every shuffle.
    rank = sum(1 for x in samples if x <= realized) / len(samples)

    return {
        "n_resamples": len(samples),
        "realized_sharpe": realized,
        "sharpe_distribution": {
            "mean": mean,
            "stdev": sigma,
            "p5": _q(0.05),
            "p25": _q(0.25),
            "p50": _q(0.5),
            "p75": _q(0.75),
            "p95": _q(0.95),
            "min": samples[0],
            "max": samples[-1],
        },
        "observed_vs_distribution": {
            "position_pct": rank * 100.0,
            "z_score": z,
            "interpretation": (
                "sequence-driven"
                if rank > 0.95 or rank < 0.05
                else "robust-to-sequence"
            ),
        },
    }


# ── Latency-perturbation Monte Carlo (expensive, multi-run) ────────────────


@dataclass
class LatencyPerturbationRun:
    p95_multiplier: float
    submit_p50_ms: float
    submit_p95_ms: float
    cancel_p50_ms: float
    cancel_p95_ms: float
    success: bool = False
    runtime_error: Optional[str] = None
    trade_count: int = 0
    total_return_pct: float = 0.0
    sharpe: Optional[float] = None
    max_drawdown_pct: float = 0.0
    fees_paid_usd: float = 0.0


@dataclass
class MonteCarloLatencyResult:
    base_submit_p50_ms: float
    base_submit_p95_ms: float
    base_cancel_p50_ms: float
    base_cancel_p95_ms: float
    runs: list[LatencyPerturbationRun] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_submit_p50_ms": self.base_submit_p50_ms,
            "base_submit_p95_ms": self.base_submit_p95_ms,
            "base_cancel_p50_ms": self.base_cancel_p50_ms,
            "base_cancel_p95_ms": self.base_cancel_p95_ms,
            "runs": [r.__dict__ for r in self.runs],
            "summary": self.summary,
        }


async def run_monte_carlo_latency(
    *,
    source_code: str,
    slug: str = "_backtest_mc_latency",
    config: dict[str, Any] | None = None,
    token_ids: list[str] | None = None,
    start: datetime,
    end: datetime,
    initial_capital_usd: float = 1000.0,
    base_submit_p50_ms: float = 350.0,
    base_submit_p95_ms: float = 900.0,
    base_cancel_p50_ms: float = 200.0,
    base_cancel_p95_ms: float = 600.0,
    multipliers: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0),
    seed: int | None = 42,
    concurrency: int = 2,
) -> MonteCarloLatencyResult:
    """Run the same backtest under multiple latency regimes.

    Each multiplier scales BOTH p50 and p95 of submit and cancel
    profiles uniformly.  Reports a Sharpe-vs-latency curve so the
    operator can see how much their edge depends on the latency
    assumption — and how a network regression in production would
    erode it.
    """
    from services.strategy_backtester import run_execution_backtest

    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    semaphore = asyncio.Semaphore(max(1, int(concurrency)))

    async def _run_one(mult: float) -> LatencyPerturbationRun:
        async with semaphore:
            sub_p50 = base_submit_p50_ms * mult
            sub_p95 = base_submit_p95_ms * mult
            can_p50 = base_cancel_p50_ms * mult
            can_p95 = base_cancel_p95_ms * mult
            try:
                exec_kwargs: dict[str, Any] = {
                    "source_code": source_code,
                    "slug": slug,
                    "config": config,
                    "token_ids": token_ids,
                    "start": start,
                    "end": end,
                    "initial_capital_usd": initial_capital_usd,
                    "submit_latency_p50_ms": sub_p50,
                    "submit_latency_p95_ms": sub_p95,
                    "cancel_latency_p50_ms": can_p50,
                    "cancel_latency_p95_ms": can_p95,
                }
                if seed is not None:
                    exec_kwargs["seed"] = int(seed)
                exec_result = await run_execution_backtest(**exec_kwargs)
                d = exec_result.to_dict()
                sharpe = (d.get("sharpe") or {}).get("value")
                return LatencyPerturbationRun(
                    p95_multiplier=float(mult),
                    submit_p50_ms=sub_p50,
                    submit_p95_ms=sub_p95,
                    cancel_p50_ms=can_p50,
                    cancel_p95_ms=can_p95,
                    success=bool(d.get("success")),
                    runtime_error=d.get("runtime_error"),
                    trade_count=int(d.get("trade_count") or 0),
                    total_return_pct=float(d.get("total_return_pct") or 0.0),
                    sharpe=float(sharpe) if isinstance(sharpe, (int, float)) else None,
                    max_drawdown_pct=float(d.get("max_drawdown_pct") or 0.0),
                    fees_paid_usd=float(d.get("fees_paid_usd") or 0.0),
                )
            except Exception as exc:
                logger.exception("Latency MC run @%.2fx failed", mult)
                return LatencyPerturbationRun(
                    p95_multiplier=float(mult),
                    submit_p50_ms=sub_p50,
                    submit_p95_ms=sub_p95,
                    cancel_p50_ms=can_p50,
                    cancel_p95_ms=can_p95,
                    success=False,
                    runtime_error=str(exc),
                )

    runs = await asyncio.gather(*[_run_one(m) for m in multipliers])
    succeeded = [r for r in runs if r.success and r.sharpe is not None]

    summary: dict[str, Any] = {
        "n_runs": len(runs),
        "n_runs_succeeded": len(succeeded),
    }
    if len(succeeded) >= 2:
        sharpes = [(r.p95_multiplier, r.sharpe) for r in succeeded]  # type: ignore[misc]
        # Slope: (sharpe at 2x - sharpe at 0.5x) / (2 - 0.5) approximated
        # via the available range.  Negative slope = strategy degrades
        # with worse latency (typical maker behavior).
        sharpes.sort(key=lambda x: x[0])
        first_mult, first_sr = sharpes[0]
        last_mult, last_sr = sharpes[-1]
        slope = (last_sr - first_sr) / (last_mult - first_mult) if last_mult != first_mult else 0.0
        summary["sharpe_slope_per_x_latency"] = slope
        summary["sharpe_at_baseline"] = next(
            (sr for mult, sr in sharpes if abs(mult - 1.0) < 1e-6), None
        )
        summary["sharpe_at_worst_latency"] = sharpes[-1][1]
        summary["sharpe_at_best_latency"] = sharpes[0][1]
        # latency_sensitivity = max range of Sharpe across multipliers
        all_sr = [sr for _, sr in sharpes]
        summary["sharpe_range"] = max(all_sr) - min(all_sr)

    return MonteCarloLatencyResult(
        base_submit_p50_ms=base_submit_p50_ms,
        base_submit_p95_ms=base_submit_p95_ms,
        base_cancel_p50_ms=base_cancel_p50_ms,
        base_cancel_p95_ms=base_cancel_p95_ms,
        runs=runs,
        summary=summary,
    )
