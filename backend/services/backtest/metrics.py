"""Backtest metrics with bootstrap confidence intervals.

Standard quant metrics computed from the equity curve and trade ledger:

* **Returns**: total return, annualized return.
* **Risk-adjusted**: Sharpe (annualized), Sortino (annualized; downside
  semideviation), Calmar (annualized return / max drawdown).
* **Drawdown**: max drawdown (USD and %), drawdown duration.
* **Trade-level**: hit rate, win/loss ratio, profit factor, average
  win/loss, expectancy.

Each metric is reported with a bootstrap 95% confidence interval. The
bootstrap resamples *trade outcomes* (not equity points) to preserve the
discrete event structure.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Sequence


# Annualization constants. Polymarket markets resolve discretely; we use
# trading-day-style annualization (252) for risk-adjusted ratios and
# calendar-day (365) for drawdown duration.
ANN_TRADING_DAYS = 252
SECONDS_PER_TRADING_DAY = 6.5 * 3600  # ignored — we use periods returned
ANN_CALENDAR_SECONDS = 365 * 24 * 3600


@dataclass
class TradeOutcome:
    """One closed trade, suitable for trade-level statistics."""

    pnl_usd: float
    return_pct: float  # pnl / cost_basis
    holding_seconds: float
    won: bool


@dataclass
class MetricCI:
    """A point estimate with a bootstrap 95% CI."""

    value: float
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None


@dataclass
class BacktestMetrics:
    total_return_usd: float = 0.0
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    sharpe: MetricCI = field(default_factory=lambda: MetricCI(0.0))
    sortino: MetricCI = field(default_factory=lambda: MetricCI(0.0))
    calmar: MetricCI = field(default_factory=lambda: MetricCI(0.0))
    max_drawdown_usd: float = 0.0
    max_drawdown_pct: float = 0.0
    drawdown_duration_seconds: float = 0.0
    hit_rate: MetricCI = field(default_factory=lambda: MetricCI(0.0))
    profit_factor: MetricCI = field(default_factory=lambda: MetricCI(0.0))
    avg_win_usd: float = 0.0
    avg_loss_usd: float = 0.0
    expectancy_usd: MetricCI = field(default_factory=lambda: MetricCI(0.0))
    trade_count: int = 0
    fees_paid_usd: float = 0.0
    final_equity_usd: float = 0.0
    initial_capital_usd: float = 0.0


def bootstrap_ci(
    samples: Sequence[float],
    statistic,
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> tuple[Optional[float], Optional[float]]:
    """Percentile bootstrap CI for ``statistic(samples)``.

    Returns ``(low, high)`` or ``(None, None)`` if the sample is too small
    to yield a meaningful interval.
    """
    n = len(samples)
    if n < 8:
        return None, None
    rng = random.Random(seed)
    stats: list[float] = []
    sample_list = list(samples)
    for _ in range(int(n_resamples)):
        resample = [sample_list[rng.randrange(n)] for _ in range(n)]
        try:
            stats.append(float(statistic(resample)))
        except Exception:
            continue
    if not stats:
        return None, None
    stats.sort()
    alpha = (1.0 - confidence) / 2.0
    lo_idx = max(0, int(math.floor(alpha * len(stats))))
    hi_idx = min(len(stats) - 1, int(math.ceil((1.0 - alpha) * len(stats))) - 1)
    return stats[lo_idx], stats[hi_idx]


# ── Statistic helpers ────────────────────────────────────────────────────


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _downside_std(xs: Sequence[float], target: float = 0.0) -> float:
    losses = [x for x in xs if x < target]
    if len(losses) < 2:
        return 0.0
    m = target
    return math.sqrt(sum((x - m) ** 2 for x in losses) / (len(losses) - 1))


def sharpe_of_returns(returns: Sequence[float], *, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    mu = _mean(returns)
    sigma = _std(returns)
    if sigma <= 0:
        return 0.0
    return mu / sigma * math.sqrt(periods_per_year)


def _skewness(xs: Sequence[float]) -> float:
    if len(xs) < 3:
        return 0.0
    m = _mean(xs)
    s = _std(xs)
    if s <= 0:
        return 0.0
    n = len(xs)
    return (n / ((n - 1) * (n - 2))) * sum(((x - m) / s) ** 3 for x in xs)


def _kurtosis(xs: Sequence[float]) -> float:
    """Excess kurtosis (kurtosis - 3 ⇒ 0 for normal)."""
    if len(xs) < 4:
        return 0.0
    m = _mean(xs)
    s = _std(xs)
    if s <= 0:
        return 0.0
    n = len(xs)
    g2 = sum(((x - m) / s) ** 4 for x in xs) / n - 3.0
    return g2


def deflated_sharpe_ratio(
    returns: Sequence[float],
    *,
    n_trials: int,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """López de Prado's Deflated Sharpe Ratio.

    Adjusts the observed Sharpe for (a) the number of independent
    parameter trials that were searched and (b) the non-normality of
    the return distribution (skewness + kurtosis).  Returns the
    probability that the TRUE Sharpe exceeds the SR0 floor of 0
    after the multiple-comparisons correction.

    Reference: ``The Sharpe Ratio Efficient Frontier'', JoR 2012.

    Returns a dict with:
      observed_sharpe: raw annualized Sharpe.
      sr_zero: theoretical max Sharpe across n_trials of pure noise.
      probabilistic_sharpe: P(true SR > 0) ignoring trial count.
      deflated_sharpe: P(true SR > sr_zero) — the over-fit-aware metric.

    A deflated_sharpe < 0.95 with n_trials > 1 means you can't
    confidently distinguish the strategy from luck given how many
    knobs you tuned.
    """
    n = len(returns)
    if n < 4:
        return {
            "observed_sharpe": 0.0,
            "sr_zero": 0.0,
            "probabilistic_sharpe": 0.0,
            "deflated_sharpe": 0.0,
            "n_observations": n,
            "n_trials": int(max(1, n_trials)),
        }
    n_trials = max(1, int(n_trials))
    sr = sharpe_of_returns(returns, periods_per_year=periods_per_year)
    skew = _skewness(returns)
    excess_kurt = _kurtosis(returns)

    # SR0: expected max of n_trials draws from N(0, 1) annualised SR.
    # López de Prado eq. 9: SR0 = sqrt(V) * ((1 - γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e)))
    # where V is the variance of trial Sharpes (assumed = 1/T if not
    # estimated) and γ ≈ 0.5772 (Euler-Mascheroni).  We approximate
    # V = 1.0 and use a simpler form widely used in practice:
    # SR0 ≈ Φ⁻¹(1 - 1/N) / sqrt(T)  scaled to the annualisation.
    from math import sqrt as _sqrt
    from statistics import NormalDist as _Normal

    normal = _Normal()
    # Quantile of the max of N normal draws — Bonferroni-style.
    if n_trials == 1:
        sr_zero = 0.0
    else:
        emc = 0.5772156649015329
        z_max = (1 - emc) * normal.inv_cdf(1.0 - 1.0 / n_trials) + emc * normal.inv_cdf(
            1.0 - 1.0 / (n_trials * math.e)
        )
        # Convert per-period SR0 to annualised.
        sr_zero = z_max / _sqrt(max(1, n)) * _sqrt(periods_per_year)

    # Standard error of the observed Sharpe (Mertens 2002):
    sr_var = (1.0 - skew * sr + (excess_kurt / 4.0) * sr * sr) / max(1, n - 1)
    if sr_var <= 0:
        return {
            "observed_sharpe": float(sr),
            "sr_zero": float(sr_zero),
            "probabilistic_sharpe": 1.0 if sr > 0 else 0.0,
            "deflated_sharpe": 1.0 if sr > sr_zero else 0.0,
            "n_observations": n,
            "n_trials": n_trials,
        }
    sr_se = _sqrt(sr_var)
    # PSR: probability that the true SR exceeds 0.
    psr = float(normal.cdf((sr - 0.0) / sr_se))
    # DSR: probability that the true SR exceeds SR0 (the max-of-N floor).
    dsr = float(normal.cdf((sr - sr_zero) / sr_se))
    return {
        "observed_sharpe": float(sr),
        "sr_zero": float(sr_zero),
        "probabilistic_sharpe": psr,
        "deflated_sharpe": dsr,
        "n_observations": n,
        "n_trials": n_trials,
    }


def sortino_of_returns(returns: Sequence[float], *, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    mu = _mean(returns)
    dsd = _downside_std(returns, target=0.0)
    if dsd <= 0:
        return 0.0
    return mu / dsd * math.sqrt(periods_per_year)


def hit_rate_of(trades: Sequence[TradeOutcome]) -> float:
    if not trades:
        return 0.0
    return sum(1 for t in trades if t.won) / len(trades)


def profit_factor_of(trades: Sequence[TradeOutcome]) -> float:
    gross_win = sum(t.pnl_usd for t in trades if t.pnl_usd > 0)
    gross_loss = -sum(t.pnl_usd for t in trades if t.pnl_usd < 0)
    if gross_loss <= 0:
        return float("inf") if gross_win > 0 else 0.0
    return gross_win / gross_loss


def expectancy_of(trades: Sequence[TradeOutcome]) -> float:
    if not trades:
        return 0.0
    return _mean([t.pnl_usd for t in trades])


# ── Equity curve helpers ────────────────────────────────────────────────


def equity_returns(equity_history: Sequence[tuple[datetime, float]]) -> list[float]:
    """Per-period returns from equity history. Skips zero/negative equity."""
    rets: list[float] = []
    for i in range(1, len(equity_history)):
        prev = equity_history[i - 1][1]
        curr = equity_history[i][1]
        if prev <= 0:
            continue
        rets.append((curr - prev) / prev)
    return rets


def max_drawdown(equity_history: Sequence[tuple[datetime, float]]) -> tuple[float, float, float]:
    """Return (max_dd_usd, max_dd_pct, duration_seconds)."""
    if not equity_history:
        return 0.0, 0.0, 0.0
    peak_value = -math.inf
    peak_at: Optional[datetime] = None
    worst_dd_usd = 0.0
    worst_dd_pct = 0.0
    worst_duration = 0.0
    for at, eq in equity_history:
        if eq > peak_value:
            peak_value = eq
            peak_at = at
            continue
        dd_usd = peak_value - eq
        dd_pct = dd_usd / peak_value if peak_value > 0 else 0.0
        if dd_usd > worst_dd_usd:
            worst_dd_usd = dd_usd
            worst_dd_pct = dd_pct
        if peak_at is not None:
            duration = (at - peak_at).total_seconds()
            if duration > worst_duration:
                worst_duration = duration
    return worst_dd_usd, worst_dd_pct, worst_duration


# ── Top-level metrics function ──────────────────────────────────────────


def compute_metrics(
    *,
    initial_capital_usd: float,
    final_equity_usd: float,
    equity_history: Sequence[tuple[datetime, float]],
    trades: Sequence[TradeOutcome],
    fees_paid_usd: float,
    periods_per_year: int = 252,
    bootstrap_resamples: int = 2000,
    seed: Optional[int] = 42,
) -> BacktestMetrics:
    """Compute the standard backtest metric set with bootstrap CIs.

    Bootstrap is over trade outcomes (so CI for hit_rate / profit_factor
    is well-defined). For Sharpe/Sortino we resample equity returns.
    """
    total_usd = final_equity_usd - initial_capital_usd
    total_pct = (total_usd / initial_capital_usd) * 100.0 if initial_capital_usd > 0 else 0.0

    duration_s = 0.0
    if len(equity_history) >= 2:
        duration_s = (equity_history[-1][0] - equity_history[0][0]).total_seconds()
    annualized_pct = 0.0
    if duration_s > 0 and initial_capital_usd > 0:
        years = duration_s / ANN_CALENDAR_SECONDS
        # Reject pathological tiny windows where 1/years explodes the
        # exponent — use the raw period return instead. Threshold of 1
        # hour avoids both overflow and meaningless annualizations.
        if years > 1.0 / (24 * 365):
            growth = max(0.01, final_equity_usd / initial_capital_usd)
            try:
                annualized_pct = (math.pow(growth, 1.0 / years) - 1.0) * 100.0
            except (OverflowError, ValueError):
                annualized_pct = total_pct
        else:
            annualized_pct = total_pct

    rets = equity_returns(equity_history)
    sharpe_pt = sharpe_of_returns(rets, periods_per_year=periods_per_year)
    sortino_pt = sortino_of_returns(rets, periods_per_year=periods_per_year)
    sharpe_lo, sharpe_hi = bootstrap_ci(
        rets,
        lambda r: sharpe_of_returns(r, periods_per_year=periods_per_year),
        n_resamples=bootstrap_resamples,
        seed=seed,
    )
    sortino_lo, sortino_hi = bootstrap_ci(
        rets,
        lambda r: sortino_of_returns(r, periods_per_year=periods_per_year),
        n_resamples=bootstrap_resamples,
        seed=seed,
    )

    dd_usd, dd_pct, dd_duration = max_drawdown(equity_history)
    calmar_pt = (annualized_pct / 100.0) / dd_pct if dd_pct > 0 else 0.0
    calmar_lo: Optional[float] = None
    calmar_hi: Optional[float] = None
    # Bootstrap on equity returns — re-derive DD on each resample
    if rets and dd_pct > 0:
        years_local = duration_s / ANN_CALENDAR_SECONDS if duration_s > 0 else 0.0
        def _calmar_resample(r):
            eq = [initial_capital_usd]
            for x in r:
                eq.append(eq[-1] * (1 + x))
            peak = max(eq)
            min_eq = min(eq)
            local_dd = (peak - min_eq) / peak if peak > 0 else 0.0
            if local_dd <= 0:
                return calmar_pt
            growth_local = max(0.01, eq[-1] / initial_capital_usd)
            if years_local > 1.0 / (24 * 365):
                try:
                    local_ann = (math.pow(growth_local, 1.0 / years_local) - 1.0) * 100.0
                except (OverflowError, ValueError):
                    local_ann = (eq[-1] / initial_capital_usd - 1.0) * 100.0
            else:
                local_ann = (eq[-1] / initial_capital_usd - 1.0) * 100.0
            return (local_ann / 100.0) / local_dd
        try:
            calmar_lo, calmar_hi = bootstrap_ci(
                rets, _calmar_resample, n_resamples=bootstrap_resamples, seed=seed,
            )
        except Exception:
            pass

    # Trade-level
    hit = hit_rate_of(trades)
    pf = profit_factor_of(trades)
    expect = expectancy_of(trades)
    pnls = [t.pnl_usd for t in trades]
    wins = [t.pnl_usd for t in trades if t.pnl_usd > 0]
    losses = [t.pnl_usd for t in trades if t.pnl_usd < 0]
    avg_win = _mean(wins) if wins else 0.0
    avg_loss = _mean(losses) if losses else 0.0

    hit_lo, hit_hi = bootstrap_ci(
        pnls, lambda xs: sum(1 for x in xs if x > 0) / len(xs) if xs else 0.0,
        n_resamples=bootstrap_resamples, seed=seed,
    )
    pf_lo, pf_hi = bootstrap_ci(
        pnls,
        lambda xs: (
            sum(x for x in xs if x > 0)
            / max(1e-9, -sum(x for x in xs if x < 0))
        ) if xs else 0.0,
        n_resamples=bootstrap_resamples, seed=seed,
    )
    expect_lo, expect_hi = bootstrap_ci(
        pnls, _mean, n_resamples=bootstrap_resamples, seed=seed,
    )

    return BacktestMetrics(
        total_return_usd=total_usd,
        total_return_pct=total_pct,
        annualized_return_pct=annualized_pct,
        sharpe=MetricCI(sharpe_pt, sharpe_lo, sharpe_hi),
        sortino=MetricCI(sortino_pt, sortino_lo, sortino_hi),
        calmar=MetricCI(calmar_pt, calmar_lo, calmar_hi),
        max_drawdown_usd=dd_usd,
        max_drawdown_pct=dd_pct * 100.0,
        drawdown_duration_seconds=dd_duration,
        hit_rate=MetricCI(hit, hit_lo, hit_hi),
        profit_factor=MetricCI(pf, pf_lo, pf_hi),
        avg_win_usd=avg_win,
        avg_loss_usd=avg_loss,
        expectancy_usd=MetricCI(expect, expect_lo, expect_hi),
        trade_count=len(trades),
        fees_paid_usd=fees_paid_usd,
        final_equity_usd=final_equity_usd,
        initial_capital_usd=initial_capital_usd,
    )


__all__ = [
    "TradeOutcome",
    "MetricCI",
    "BacktestMetrics",
    "bootstrap_ci",
    "compute_metrics",
    "deflated_sharpe_ratio",
    "sharpe_of_returns",
    "sortino_of_returns",
    "hit_rate_of",
    "profit_factor_of",
    "expectancy_of",
    "max_drawdown",
    "equity_returns",
]
