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


# Annualization. The equity curve is marked in EVENT TIME — one mark per
# book snapshot while a position is open — NOT on fixed daily bars. A
# hardcoded sqrt(252) is therefore wrong by orders of magnitude for the
# sub-second sampling of crypto markets. Risk-adjusted ratios derive their
# periods-per-year from the equity curve's own sampling frequency via
# ``infer_periods_per_year`` / ``resample_equity_returns``. ANN_TRADING_DAYS
# survives only as the degenerate fallback when the curve is too short to
# infer a frequency.
ANN_TRADING_DAYS = 252
ANN_CALENDAR_SECONDS = 365 * 24 * 3600

# Sentinel for ratios whose denominator is zero (winning streak with
# no losses, etc.).  Returned by tail_ratio / gain_to_pain /
# profit_factor instead of math.inf so the result stays JSON-
# compliant — FastAPI's default encoder rejects Infinity / NaN and
# fails the entire response with a 500.  Values >= this sentinel
# should be rendered as "∞ (no denominator)" in the UI.
_NO_DENOM_SENTINEL = 1_000_000.0


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
    # Tail-risk metrics — Lopez de Prado / Cornish-Fisher style.  CVaR
    # answers "in the worst 5% of periods, what's my average return?";
    # tail_ratio answers "is the upside fatter than the downside?";
    # gain_to_pain measures aggregate gains vs aggregate pains and is
    # robust to outliers compared to profit_factor.
    expected_shortfall_5pct: MetricCI = field(default_factory=lambda: MetricCI(0.0))
    expected_shortfall_1pct: MetricCI = field(default_factory=lambda: MetricCI(0.0))
    tail_ratio: MetricCI = field(default_factory=lambda: MetricCI(0.0))
    gain_to_pain: MetricCI = field(default_factory=lambda: MetricCI(0.0))


def bootstrap_ci(
    samples: Sequence[float],
    statistic,
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
    max_sample_size: int = 2000,
    max_resample_draws: int = 250_000,
) -> tuple[Optional[float], Optional[float]]:
    """Percentile bootstrap CI for ``statistic(samples)``.

    Returns ``(low, high)`` or ``(None, None)`` if the sample is too small
    to yield a meaningful interval.
    """
    n = len(samples)
    if n < 8:
        return None, None
    sample_list = list(samples)
    if max_sample_size > 0 and n > max_sample_size:
        step = (n - 1) / float(max_sample_size - 1)
        sample_list = [sample_list[int(round(i * step))] for i in range(max_sample_size)]
        n = len(sample_list)
    if max_resample_draws > 0:
        n_resamples = max(100, min(int(n_resamples), int(max_resample_draws // max(1, n))))
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(int(n_resamples)):
        resample = [sample_list[rng.randrange(n)] for _ in range(n)]
        try:
            v = float(statistic(resample))
        except Exception:
            continue
        # Drop non-finite results so the CI bounds stay JSON-safe.  A
        # statistic that legitimately returns infinity (zero-denominator
        # ratio) should be returning the _NO_DENOM_SENTINEL instead.
        if math.isnan(v) or math.isinf(v):
            continue
        stats.append(v)
    return _ci_from_stats(stats, confidence)


def _ci_from_stats(
    stats: list[float], confidence: float
) -> tuple[Optional[float], Optional[float]]:
    """Percentile CI bounds from a list of bootstrap replicate statistics."""
    if not stats:
        return None, None
    stats.sort()
    alpha = (1.0 - confidence) / 2.0
    lo_idx = max(0, int(math.floor(alpha * len(stats))))
    hi_idx = min(len(stats) - 1, int(math.ceil((1.0 - alpha) * len(stats))) - 1)
    return stats[lo_idx], stats[hi_idx]


def _autocorrelation(xs: Sequence[float], lag: int) -> float:
    n = len(xs)
    if lag < 1 or lag >= n:
        return 0.0
    m = _mean(xs)
    denom = sum((x - m) ** 2 for x in xs)
    if denom <= 0:
        return 0.0
    num = sum((xs[i] - m) * (xs[i - lag] - m) for i in range(lag, n))
    return num / denom


def _mean_block_length(xs: Sequence[float]) -> float:
    """Decorrelation-informed mean block length for the stationary bootstrap.

    Uses the first lag at which the sample autocorrelation falls below the
    ~95% white-noise band (|rho| < 2/sqrt(n)); falls back to the n**(1/3)
    rule of thumb when the series shows no significant autocorrelation.
    """
    n = len(xs)
    if n < 8:
        return 1.0
    threshold = 2.0 / math.sqrt(n)
    decor = 0
    for lag in range(1, min(n // 2, 64) + 1):
        if abs(_autocorrelation(xs, lag)) < threshold:
            decor = lag
            break
    if decor <= 0:
        decor = max(1, round(n ** (1.0 / 3.0)))
    return float(min(max(1, decor), max(1, n // 2)))


def block_bootstrap_ci(
    series: Sequence[float],
    statistic,
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
    max_resample_draws: int = 8_000_000,
) -> tuple[Optional[float], Optional[float]]:
    """Stationary (Politis-Romano 1994) block-bootstrap CI for a statistic of
    an AUTOCORRELATED series.

    Equity returns are serially dependent (overlapping holding periods,
    trending equity), so an IID percentile bootstrap understates CI width.
    The stationary bootstrap resamples geometrically-distributed, wrapped
    blocks — preserving the local dependence structure — with a mean block
    length inferred from the series' own autocorrelation.  Use this for
    return-series statistics (Sharpe, Sortino, tail metrics); the IID
    ``bootstrap_ci`` remains correct for exchangeable trade outcomes.
    """
    n = len(series)
    if n < 8:
        return None, None
    s = list(series)
    if max_resample_draws > 0:
        n_resamples = max(100, min(int(n_resamples), int(max_resample_draws // max(1, n))))
    mean_block = _mean_block_length(s)
    p_new_block = 1.0 / mean_block  # geometric block-length parameter
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(int(n_resamples)):
        resample: list[float] = []
        idx = rng.randrange(n)
        for _ in range(n):
            resample.append(s[idx])
            if rng.random() < p_new_block:
                idx = rng.randrange(n)
            else:
                idx = (idx + 1) % n
        try:
            v = float(statistic(resample))
        except Exception:
            continue
        if math.isnan(v) or math.isinf(v):
            continue
        stats.append(v)
    return _ci_from_stats(stats, confidence)


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


def deflated_sharpe_from_moments(
    *,
    annualized_sharpe: float,
    skew: float,
    excess_kurtosis: float,
    n_observations: int,
    n_trials: int,
    periods_per_year: int = ANN_TRADING_DAYS,
) -> dict[str, float]:
    """Deflated Sharpe Ratio from precomputed return moments.

    DSR is a closed-form function of the (annualized) Sharpe, the return
    distribution's skew / excess-kurtosis, the observation count, the
    number of trials searched, and the annualization factor.  Exposing the
    moment form lets a caller that knows ``n_trials`` (the search size)
    deflate a run's Sharpe without re-shipping or re-sampling its full
    return series — so the deflation uses the SAME frequency-correct basis
    as the headline Sharpe rather than a downsampled curve.
    """
    from math import sqrt as _sqrt
    from statistics import NormalDist as _Normal

    n = int(n_observations)
    if n < 4:
        return {
            "observed_sharpe": float(annualized_sharpe),
            "sr_zero": 0.0,
            "probabilistic_sharpe": 0.0,
            "deflated_sharpe": 0.0,
            "n_observations": n,
            "n_trials": int(max(1, n_trials)),
        }
    n_trials = max(1, int(n_trials))
    sr = float(annualized_sharpe)
    normal = _Normal()
    # SR0: expected max of n_trials draws of pure-noise Sharpe (López de
    # Prado eq. 9, Euler-Mascheroni form), converted to the annualisation.
    if n_trials == 1:
        sr_zero = 0.0
    else:
        emc = 0.5772156649015329
        z_max = (1 - emc) * normal.inv_cdf(1.0 - 1.0 / n_trials) + emc * normal.inv_cdf(
            1.0 - 1.0 / (n_trials * math.e)
        )
        sr_zero = z_max / _sqrt(max(1, n)) * _sqrt(periods_per_year)
    # Standard error of the observed Sharpe (Mertens 2002).
    sr_var = (1.0 - skew * sr + (excess_kurtosis / 4.0) * sr * sr) / max(1, n - 1)
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
    psr = float(normal.cdf((sr - 0.0) / sr_se))
    dsr = float(normal.cdf((sr - sr_zero) / sr_se))
    return {
        "observed_sharpe": float(sr),
        "sr_zero": float(sr_zero),
        "probabilistic_sharpe": psr,
        "deflated_sharpe": dsr,
        "n_observations": n,
        "n_trials": n_trials,
    }


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
    sr = sharpe_of_returns(returns, periods_per_year=periods_per_year) if n >= 2 else 0.0
    return deflated_sharpe_from_moments(
        annualized_sharpe=sr,
        skew=_skewness(returns),
        excess_kurtosis=_kurtosis(returns),
        n_observations=n,
        n_trials=n_trials,
        periods_per_year=periods_per_year,
    )


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
        # No losing trades.  Return a large finite sentinel (not
        # math.inf — JSON has no representation for it and FastAPI
        # rejects the response).  The UI treats >= _NO_DENOM as
        # "no denominator, exceptional run".
        return _NO_DENOM_SENTINEL if gross_win > 0 else 0.0
    return gross_win / gross_loss


def expectancy_of(trades: Sequence[TradeOutcome]) -> float:
    if not trades:
        return 0.0
    return _mean([t.pnl_usd for t in trades])


def _percentile(xs: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile.  ``p`` is in [0, 1]."""
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = max(0.0, min(1.0, p)) * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def expected_shortfall(returns: Sequence[float], *, alpha: float = 0.05) -> float:
    """Conditional VaR / Expected Shortfall.

    Mean of the worst ``alpha`` fraction of period returns — answers
    "in a bad period, how bad is bad on average?".  Reported as a
    negative number when the strategy actually has losing tail
    returns.  Returns 0.0 when the sample is too thin to support a
    meaningful tail (under 1/alpha observations).
    """
    n = len(returns)
    if n == 0:
        return 0.0
    cutoff_count = max(1, int(math.floor(n * max(1e-6, alpha))))
    if n < int(math.ceil(1.0 / alpha)):
        # Not enough samples for a meaningful tail estimate.
        return 0.0
    worst = sorted(returns)[:cutoff_count]
    return _mean(worst)


def tail_ratio(returns: Sequence[float], *, alpha: float = 0.05) -> float:
    """Right-tail / left-tail magnitude ratio.

    ``|p_{1-alpha}| / |p_alpha|`` of period returns.  >1 means the
    upside tail is fatter than the downside (positively skewed
    payouts); <1 means downside dominates.  A common Sharpe-style
    sanity check: a strategy with Sharpe 2 but tail_ratio 0.4 is
    quietly carrying lottery-ticket-style left-tail risk.
    """
    if len(returns) < int(math.ceil(2.0 / alpha)):
        return 0.0
    upside = abs(_percentile(returns, 1.0 - alpha))
    downside = abs(_percentile(returns, alpha))
    if downside <= 0:
        return _NO_DENOM_SENTINEL if upside > 0 else 0.0
    return upside / downside


def gain_to_pain(returns: Sequence[float]) -> float:
    """Schwager's gain-to-pain ratio: sum of positive returns over
    absolute sum of negative returns.  Outlier-robust alternative to
    the trade-level profit_factor; defined on equity returns so it
    captures sustained drawdowns even when individual trades win.
    """
    gains = sum(r for r in returns if r > 0)
    pains = -sum(r for r in returns if r < 0)
    if pains <= 0:
        return _NO_DENOM_SENTINEL if gains > 0 else 0.0
    return gains / pains


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


def _infer_bar_seconds(
    equity_history: Sequence[tuple[datetime, float]]
) -> Optional[float]:
    """Median spacing (seconds) between equity marks, clamped to [1s, 1d].

    Returns None when there are too few marks to estimate a frequency.
    """
    if len(equity_history) < 3:
        return None
    deltas = sorted(
        d
        for d in (
            (equity_history[i][0] - equity_history[i - 1][0]).total_seconds()
            for i in range(1, len(equity_history))
        )
        if d > 0
    )
    if not deltas:
        return None
    median = deltas[len(deltas) // 2]
    return min(max(median, 1.0), 86400.0)


def infer_periods_per_year(
    equity_history: Sequence[tuple[datetime, float]]
) -> int:
    """Annualization factor implied by the equity curve's own sampling
    frequency.  Falls back to ``ANN_TRADING_DAYS`` for degenerate curves.
    """
    bar_seconds = _infer_bar_seconds(equity_history)
    if not bar_seconds:
        return ANN_TRADING_DAYS
    return max(1, int(round(ANN_CALENDAR_SECONDS / bar_seconds)))


def resample_equity_returns(
    equity_history: Sequence[tuple[datetime, float]],
    *,
    bar_seconds: float,
    max_bars: int = 200_000,
) -> list[float]:
    """Returns on a UNIFORM time grid built from the (event-time) equity
    curve via last-observation-carried-forward.

    Idle stretches with no marks become flat (zero-return) bars, so
    volatility isn't over-estimated by sampling only while a position is
    open, and the uniform spacing makes sqrt(periods_per_year)
    annualization valid.  The grid is capped at ``max_bars`` (the bar is
    widened if a run is long enough to exceed it).
    """
    n = len(equity_history)
    if n < 2 or bar_seconds <= 0:
        return []
    t0 = equity_history[0][0]
    span = (equity_history[-1][0] - t0).total_seconds()
    if span <= 0:
        return []
    n_bars = max(1, math.ceil(span / bar_seconds))
    if n_bars > max_bars:
        bar_seconds = span / max_bars
        n_bars = max_bars
    eq_times = [(t - t0).total_seconds() for (t, _) in equity_history]
    eq_vals = [float(v) for (_, v) in equity_history]
    rets: list[float] = []
    j = 0
    prev_val = eq_vals[0]
    for k in range(1, n_bars + 1):
        grid_t = min(k * bar_seconds, span)
        while j + 1 < len(eq_times) and eq_times[j + 1] <= grid_t:
            j += 1
        curr_val = eq_vals[j]
        if prev_val > 0:
            rets.append((curr_val - prev_val) / prev_val)
        prev_val = curr_val
    return rets


def _derive_returns_and_ppy(
    equity_history: Sequence[tuple[datetime, float]],
    periods_per_year: Optional[int] = None,
) -> tuple[list[float], int]:
    """Frequency-correct (returns, periods_per_year) for an equity curve.

    Single source of truth for annualization: resample the event-time
    equity curve onto a uniform grid and derive periods-per-year from that
    grid, unless an explicit ``periods_per_year`` override is supplied.
    """
    bar_seconds = _infer_bar_seconds(equity_history)
    if periods_per_year is None:
        periods_per_year = (
            max(1, int(round(ANN_CALENDAR_SECONDS / bar_seconds)))
            if bar_seconds
            else ANN_TRADING_DAYS
        )
    rets = (
        resample_equity_returns(equity_history, bar_seconds=bar_seconds)
        if bar_seconds
        else equity_returns(equity_history)
    )
    return rets, int(periods_per_year)


def annualization_moments(
    equity_history: Sequence[tuple[datetime, float]]
) -> dict[str, float]:
    """Return-distribution moments + annualization factor for an equity
    curve, computed on the frequency-correct resampled returns.

    Lets a downstream caller that knows the search size (``n_trials``)
    compute the Deflated Sharpe via ``deflated_sharpe_from_moments``
    without the full return series (which is large / downsampled in
    transit).
    """
    rets, ppy = _derive_returns_and_ppy(equity_history)
    return {
        "periods_per_year": int(ppy),
        "n_obs": len(rets),
        "skew": _skewness(rets),
        "kurtosis": _kurtosis(rets),
        "annualized_sharpe": sharpe_of_returns(rets, periods_per_year=ppy),
    }


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
    periods_per_year: Optional[int] = None,
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

    # Annualize from the equity curve's own sampling frequency, not a
    # hardcoded daily clock (single source of truth: _derive_returns_and_ppy).
    rets, periods_per_year = _derive_returns_and_ppy(equity_history, periods_per_year)
    ppy = periods_per_year
    sharpe_pt = sharpe_of_returns(rets, periods_per_year=ppy)
    sortino_pt = sortino_of_returns(rets, periods_per_year=ppy)
    # Return-series statistics use the stationary block bootstrap (returns
    # are autocorrelated); trade-level statistics below keep the IID
    # bootstrap (trade outcomes are exchangeable).
    sharpe_lo, sharpe_hi = block_bootstrap_ci(
        rets,
        lambda r: sharpe_of_returns(r, periods_per_year=ppy),
        n_resamples=bootstrap_resamples,
        seed=seed,
    )
    sortino_lo, sortino_hi = block_bootstrap_ci(
        rets,
        lambda r: sortino_of_returns(r, periods_per_year=ppy),
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
            calmar_lo, calmar_hi = block_bootstrap_ci(
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
    def _pf_stat(xs):
        if not xs:
            return 0.0
        gw = sum(x for x in xs if x > 0)
        gl = -sum(x for x in xs if x < 0)
        if gl <= 0:
            # Same sentinel convention as profit_factor_of() so the CI
            # doesn't silently expand to 1e11+ when the resample has
            # zero losses (which then breaks JSON encoding via inf).
            return _NO_DENOM_SENTINEL if gw > 0 else 0.0
        return gw / gl

    pf_lo, pf_hi = bootstrap_ci(
        pnls, _pf_stat, n_resamples=bootstrap_resamples, seed=seed,
    )
    expect_lo, expect_hi = bootstrap_ci(
        pnls, _mean, n_resamples=bootstrap_resamples, seed=seed,
    )

    # Tail-risk metrics on equity returns (CVaR / tail ratio / gain-to-
    # pain).  All bootstrap CIs use the same trade-outcome resampler
    # mechanism, with the statistic functions defined in this module.
    es_5_pt = expected_shortfall(rets, alpha=0.05)
    es_1_pt = expected_shortfall(rets, alpha=0.01)
    tail_pt = tail_ratio(rets, alpha=0.05)
    g2p_pt = gain_to_pain(rets)
    es_5_lo, es_5_hi = block_bootstrap_ci(
        rets, lambda r: expected_shortfall(r, alpha=0.05),
        n_resamples=bootstrap_resamples, seed=seed,
    )
    es_1_lo, es_1_hi = block_bootstrap_ci(
        rets, lambda r: expected_shortfall(r, alpha=0.01),
        n_resamples=bootstrap_resamples, seed=seed,
    )
    tail_lo, tail_hi = block_bootstrap_ci(
        rets, lambda r: tail_ratio(r, alpha=0.05),
        n_resamples=bootstrap_resamples, seed=seed,
    )
    g2p_lo, g2p_hi = block_bootstrap_ci(
        rets, gain_to_pain, n_resamples=bootstrap_resamples, seed=seed,
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
        expected_shortfall_5pct=MetricCI(es_5_pt, es_5_lo, es_5_hi),
        expected_shortfall_1pct=MetricCI(es_1_pt, es_1_lo, es_1_hi),
        tail_ratio=MetricCI(tail_pt, tail_lo, tail_hi),
        gain_to_pain=MetricCI(g2p_pt, g2p_lo, g2p_hi),
    )


__all__ = [
    "TradeOutcome",
    "MetricCI",
    "BacktestMetrics",
    "bootstrap_ci",
    "block_bootstrap_ci",
    "infer_periods_per_year",
    "resample_equity_returns",
    "compute_metrics",
    "deflated_sharpe_ratio",
    "deflated_sharpe_from_moments",
    "annualization_moments",
    "sharpe_of_returns",
    "sortino_of_returns",
    "hit_rate_of",
    "profit_factor_of",
    "expectancy_of",
    "max_drawdown",
    "equity_returns",
    "expected_shortfall",
    "tail_ratio",
    "gain_to_pain",
]
