"""Annualization + block-bootstrap correctness for backtest metrics.

Covers two institutional-grade fixes:

* Risk-adjusted ratios annualize from the equity curve's OWN sampling
  frequency (derived), not a hardcoded sqrt(252) on event-time data.
* CIs for autocorrelated return-series statistics use the stationary
  block bootstrap, which is wider (correctly) than the IID bootstrap.
"""
import math
import random
from datetime import datetime, timedelta, timezone

from services.backtest.metrics import (
    ANN_CALENDAR_SECONDS,
    ANN_TRADING_DAYS,
    block_bootstrap_ci,
    bootstrap_ci,
    compute_metrics,
    infer_periods_per_year,
    resample_equity_returns,
)


def _curve(start, spacing_s, values):
    t = start
    out = []
    for v in values:
        out.append((t, float(v)))
        t = t + timedelta(seconds=spacing_s)
    return out


def test_infer_periods_per_year_tracks_spacing():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    c60 = _curve(start, 60, [1000.0 + i for i in range(50)])
    assert abs(infer_periods_per_year(c60) - ANN_CALENDAR_SECONDS // 60) <= 2
    c1d = _curve(start, 86400, [1000.0 + i for i in range(50)])
    assert infer_periods_per_year(c1d) == 365


def test_infer_periods_degenerate_falls_back():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert infer_periods_per_year([(start, 1000.0)]) == ANN_TRADING_DAYS
    assert infer_periods_per_year([]) == ANN_TRADING_DAYS


def test_resample_injects_flat_bars():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    hist = [
        (start, 1000.0),
        (start + timedelta(seconds=1), 1001.0),
        (start + timedelta(seconds=2), 1002.0),
        (start + timedelta(seconds=10), 1002.0),
    ]
    rets = resample_equity_returns(hist, bar_seconds=1.0)
    assert len(rets) == 10
    assert rets[0] > 0 and rets[1] > 0
    assert all(abs(r) < 1e-12 for r in rets[2:])


def test_sharpe_annualization_scales_with_frequency():
    """Identical per-bar return pattern at 1s vs 60s spacing must yield
    annualized Sharpe differing by ~sqrt(60) — proving the factor tracks
    the data's true frequency rather than a hardcoded constant."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rng = random.Random(0)
    seq = [1000.0]
    for _ in range(400):
        seq.append(seq[-1] * (1 + rng.gauss(0.0002, 0.01)))
    c1 = _curve(start, 1, seq)
    c60 = _curve(start, 60, seq)
    m1 = compute_metrics(
        initial_capital_usd=1000.0, final_equity_usd=seq[-1],
        equity_history=c1, trades=[], fees_paid_usd=0.0, seed=1,
    )
    m60 = compute_metrics(
        initial_capital_usd=1000.0, final_equity_usd=seq[-1],
        equity_history=c60, trades=[], fees_paid_usd=0.0, seed=1,
    )
    assert math.isfinite(m1.sharpe.value) and math.isfinite(m60.sharpe.value)
    assert m60.sharpe.value != 0
    ratio = m1.sharpe.value / m60.sharpe.value
    assert 6.0 < ratio < 9.5  # sqrt(60) ~= 7.75


def test_explicit_periods_override_is_respected():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rng = random.Random(3)
    seq = [1000.0]
    for _ in range(200):
        seq.append(seq[-1] * (1 + rng.gauss(0.0, 0.01)))
    hist = _curve(start, 1, seq)
    m = compute_metrics(
        initial_capital_usd=1000.0, final_equity_usd=seq[-1],
        equity_history=hist, trades=[], fees_paid_usd=0.0,
        periods_per_year=252, seed=1,
    )
    assert math.isfinite(m.sharpe.value)


def test_block_bootstrap_wider_on_autocorrelated_series():
    rng = random.Random(42)
    xs = [0.0]
    for _ in range(600):
        xs.append(0.9 * xs[-1] + rng.gauss(0, 1))  # AR(1), phi=0.9
    xs = xs[1:]
    mean_stat = lambda s: sum(s) / len(s)
    iid_lo, iid_hi = bootstrap_ci(xs, mean_stat, n_resamples=600, seed=7)
    blk_lo, blk_hi = block_bootstrap_ci(xs, mean_stat, n_resamples=600, seed=7)
    assert None not in (iid_lo, iid_hi, blk_lo, blk_hi)
    assert (blk_hi - blk_lo) > (iid_hi - iid_lo)


def test_dsr_moment_form_matches_series_form():
    """The moment-based DSR (used by the unified runner) must equal the
    series-based DSR for the same inputs — proving the refactor is exact."""
    from services.backtest import metrics as M

    rng = random.Random(9)
    rets = [rng.gauss(0.001, 0.02) for _ in range(300)]
    ppy = 525_600
    series = M.deflated_sharpe_ratio(rets, n_trials=10, periods_per_year=ppy)
    moments = M.deflated_sharpe_from_moments(
        annualized_sharpe=M.sharpe_of_returns(rets, periods_per_year=ppy),
        skew=M._skewness(rets),
        excess_kurtosis=M._kurtosis(rets),
        n_observations=len(rets),
        n_trials=10,
        periods_per_year=ppy,
    )
    assert abs(series["deflated_sharpe"] - moments["deflated_sharpe"]) < 1e-9
    assert abs(series["sr_zero"] - moments["sr_zero"]) < 1e-9
    assert abs(series["observed_sharpe"] - moments["observed_sharpe"]) < 1e-9


def test_block_bootstrap_matches_iid_on_independent_series():
    """On an IID series the block bootstrap should not blow up the CI —
    it stays in the same ballpark as the IID bootstrap (mean block ~1)."""
    rng = random.Random(11)
    xs = [rng.gauss(0, 1) for _ in range(600)]
    mean_stat = lambda s: sum(s) / len(s)
    iid_lo, iid_hi = bootstrap_ci(xs, mean_stat, n_resamples=600, seed=5)
    blk_lo, blk_hi = block_bootstrap_ci(xs, mean_stat, n_resamples=600, seed=5)
    iid_w = iid_hi - iid_lo
    blk_w = blk_hi - blk_lo
    assert 0.6 < (blk_w / iid_w) < 1.8
