"""Cox proportional hazards trainer for the fill probability model.

Run nightly by ``workers/cox_trainer_worker.py``.  Joins TraderOrder
fills/cancels/expires against ``MarketMicrostructureSnapshot`` at
placement time, fits Cox PH (or KM fallback if too few events), and
inserts a new row into ``fill_probability_models``.  Promotion to
``active=True`` is a separate decision: only promote when C-index
beats the currently-active model by the configured margin.

Why Cox PH:

* Right-censoring is natural — orders that are still open at training
  time aren't filled-yet but aren't unfilled-forever either.  Cox
  treats them as censored, KM does too.
* Hazard ratios per covariate are interpretable — we can show the
  user "your queue_ahead_shares HR=0.97 per 1k shares" in the UI.
* Inference is just a dot product against learned betas plus a
  baseline-survival-curve lookup, ~microseconds per order.

We use ``lifelines.CoxPHFitter``.  The trainer is conservative:

* Fall back to Kaplan-Meier (no covariates, just baseline S(t)) if
  we have fewer than 100 events total.  KM is still useful — the
  baseline survival curve gives you "P(fill within Δt) under typical
  market conditions" which beats the constant-fill assumption.
* Skip strata with fewer than 50 events when fitting stratified Cox.
* Reject the model if C-index < 0.55 (worse than coin flip after
  factoring in censoring).

Validation is held-out: the most recent 7 days of orders are reserved
for testing; everything older is the training set.  C-index and
integrated Brier score are computed on the held-out cohort.
"""
from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import warnings
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.exceptions import ConvergenceWarning as _LifelinesConvergenceWarning
from lifelines.utils import concordance_index
from scipy.linalg import LinAlgWarning as _ScipyLinAlgWarning
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    FillProbabilityModel,
    TraderOrder,
)


logger = logging.getLogger("cox_trainer")

# Suppress lifelines/numpy/scipy training-noise warnings module-wide.
# These are advisory messages about the *training data* (low-variance
# columns, ill-conditioned matrices, divide-by-zero in the partial-
# likelihood Hessian) — we already pre-screen low-variance columns in
# ``_impute_means`` and gracefully fall back to Kaplan-Meier on any
# remaining failure, so the warnings are pure log noise.  Filter at
# import time so they don't escape ``_warnings.catch_warnings()`` blocks
# (some lifelines internals re-raise warnings inside threads).
warnings.filterwarnings("ignore", category=_LifelinesConvergenceWarning)
warnings.filterwarnings("ignore", category=_ScipyLinAlgWarning)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module=r"lifelines\..*"
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module=r"numpy\..*",
    message="invalid value encountered in divide",
)


# Covariates we feed Cox.  Names match the keys produced by
# ``services.fill_simulator.survival_features.SurvivalFeatures``.
COVARIATES = [
    "queue_ahead_shares",
    "depth_behind_shares",
    "spread_bps",
    "mid_distance_bps",
    "recent_trade_intensity_per_sec",
    "time_to_resolution_seconds",
    "side_imbalance",
    "underlying_volatility_bps_per_min",
    "latency_p95_ms",
    "book_age_ms",
    "notional_usd",
]

# Statuses that indicate the order terminally filled.  These rows are
# event_observed=1 (fill observed).
FILL_STATUSES = {"executed", "closed_win", "closed_loss", "resolved", "resolved_win", "resolved_loss"}

# Statuses that indicate the order terminated WITHOUT filling.  These
# rows are event_observed=0 (right-censored — fill never happened).
CENSOR_STATUSES = {"cancelled", "expired", "rejected", "failed", "skipped"}


@dataclass
class TrainingResult:
    family: str  # "cox_ph" | "kaplan_meier"
    strata_key: str
    n_events: int
    n_observations: int
    concordance_index: float | None
    brier_score: float | None
    log_likelihood: float | None
    coefficients: dict[str, float]  # covariate -> hazard ratio (exp(beta))
    baseline_survival: dict[str, float]  # "{seconds}" -> S(t)
    feature_means: dict[str, float]
    feature_stds: dict[str, float]
    config: dict[str, Any]
    notes: str
    # Predicted-vs-observed fill rates bucketed by predicted-probability
    # decile on the held-out test set.  Drives the UI calibration plot:
    # if predicted ~= observed across deciles, the model is well
    # calibrated.  Empty for KM (no covariate-based discrimination).
    calibration_bins: list[dict[str, float]] | None = None


def _backfill_features_from_legacy_payload(
    payload: dict[str, Any],
    row: Any,
) -> dict[str, Any] | None:
    """Best-effort feature dict for orders predating Phase 1.

    Pulls what's recoverable (time_to_resolution, book_age_ms,
    notional, market_type_strata) from ``live_market`` / ``execution_estimate``
    and leaves the rest as None.  Returns None if there isn't even a
    notional to anchor the row, which makes the row useless.
    """
    if not isinstance(payload, dict):
        return None
    live_market = payload.get("live_market") if isinstance(payload, dict) else None
    live_market = live_market if isinstance(live_market, dict) else {}
    notional = None
    raw_notional = getattr(row, "notional_usd", None)
    if isinstance(raw_notional, (int, float)) and raw_notional > 0:
        notional = float(raw_notional)
    if notional is None:
        return None
    seconds_left = live_market.get("seconds_left")
    market_data_age_ms = live_market.get("market_data_age_ms")
    timeframe = (live_market.get("timeframe") or payload.get("timeframe") or "").strip().lower() if isinstance(live_market.get("timeframe") or payload.get("timeframe"), str) else ""
    strata = f"crypto_{timeframe}" if "crypto" in str((live_market.get("category") or "")).lower() and timeframe else "pooled"
    entry_delta_pct = live_market.get("entry_price_delta_pct")
    return {
        "queue_ahead_shares": None,
        "depth_behind_shares": None,
        "spread_bps": None,
        "mid_distance_bps": (float(entry_delta_pct) * 100.0) if isinstance(entry_delta_pct, (int, float)) else None,
        "recent_trade_intensity_per_sec": None,
        "time_to_resolution_seconds": float(seconds_left) if isinstance(seconds_left, (int, float)) and seconds_left > 0 else None,
        "side_imbalance": None,
        "underlying_volatility_bps_per_min": None,
        "latency_p95_ms": None,
        "book_age_ms": float(market_data_age_ms) if isinstance(market_data_age_ms, (int, float)) and market_data_age_ms >= 0 else None,
        "notional_usd": notional,
        "market_type_strata": strata,
    }


async def fetch_training_rows(
    session: AsyncSession,
    *,
    window_days: int = 30,
) -> pd.DataFrame:
    """Pull TraderOrder rows with usable survival_features into a frame.

    Returns columns: duration_seconds, event_observed, market_type_strata,
    plus one per COVARIATES.  Skips orders missing required fields.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    result = await session.execute(
        select(
            TraderOrder.id,
            TraderOrder.created_at,
            TraderOrder.executed_at,
            TraderOrder.updated_at,
            TraderOrder.status,
            TraderOrder.payload_json,
            TraderOrder.notional_usd,
        ).where(TraderOrder.created_at >= cutoff)
    )
    rows: list[dict[str, Any]] = []
    for row in result.all():
        status_key = str(row.status or "").strip().lower()
        if status_key in FILL_STATUSES:
            event_observed = 1
            terminal_at = row.executed_at or row.updated_at
        elif status_key in CENSOR_STATUSES:
            event_observed = 0
            terminal_at = row.updated_at or row.executed_at
        else:
            # Still open — censor at "now".
            event_observed = 0
            terminal_at = datetime.now(timezone.utc)
        if terminal_at is None or row.created_at is None:
            continue
        # `created_at` may be naive UTC depending on the column's TypeDecorator;
        # coerce both ends to aware UTC before subtracting.
        ca = row.created_at if row.created_at.tzinfo else row.created_at.replace(tzinfo=timezone.utc)
        ta = terminal_at if terminal_at.tzinfo else terminal_at.replace(tzinfo=timezone.utc)
        duration_seconds = max(0.001, (ta - ca).total_seconds())

        payload = row.payload_json or {}
        # Preferred path: shadow_simulation.survival_features (post-Phase 1).
        sim = payload.get("shadow_simulation") or payload.get("paper_simulation") or {}
        sf = sim.get("survival_features") if isinstance(sim, dict) else None

        # Fallback: backfill what we can from legacy live-order payloads
        # so the trainer has rows even before any post-Phase-1 orders
        # accumulate.  Missing covariates become NaN; the KM fallback
        # path doesn't read covariates anyway, and Cox handles missing
        # via mean imputation in ``_impute_means``.
        if not isinstance(sf, dict):
            sf = _backfill_features_from_legacy_payload(payload, row)
            if sf is None:
                continue
        cov_dict: dict[str, Any] = {
            "duration_seconds": duration_seconds,
            "event_observed": event_observed,
            "market_type_strata": str(sf.get("market_type_strata") or "pooled"),
            # Carry created_at so the train/test split can be
            # chronological — sorting by duration biases the test
            # cohort toward long-duration cancels and produces
            # meaningless C-index numbers.
            "_created_at_epoch": float(ca.timestamp()) if ca else 0.0,
        }
        for c in COVARIATES:
            v = sf.get(c)
            cov_dict[c] = float(v) if v is not None and isinstance(v, (int, float)) else math.nan
        rows.append(cov_dict)
    if not rows:
        return pd.DataFrame(columns=["duration_seconds", "event_observed", "market_type_strata", *COVARIATES])
    return pd.DataFrame(rows)


# Minimum standard deviation a covariate must have (in raw units, before
# standardization) to be included in the Cox fit.  Below this the column
# is treated as constant and DROPPED — feeding it to lifelines triggers
# a flood of ConvergenceWarning + LinAlgWarning + "delta contains nan"
# failures because the partial-likelihood Hessian is rank-deficient.
# Most often this fires for shadow-simulation rows where ``latency_p95_ms``
# is identically 0.0, or for assets with no recent fills (queue/depth/
# spread features all zero).
_MIN_USABLE_STD = 1e-6

# Maximum fraction of NaN values a covariate can have BEFORE imputation
# while still being included in the Cox fit.  Above this threshold the
# imputation is essentially injecting a constant (the column mean) into
# most rows, which destroys the partial-likelihood numerical conditioning
# even when the post-imputation std passes the floor above.  Empirically,
# 0.5 is the sweet spot — stricter rejects useful covariates from
# moderately-sparse cohorts; looser triggers the "delta contains nan"
# convergence failure during legacy-payload backfill.
_MAX_NAN_FRACTION = 0.5

# Maximum |Pearson r| between two covariates before we drop the
# higher-index one.  Perfect collinearity is a Cox killer; near-perfect
# collinearity (|r| > 0.98) makes the Hessian numerically singular.
_MAX_PAIRWISE_CORRELATION = 0.98


def _impute_means(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], dict[str, float], list[str]]:
    """Mean-impute missing covariates; standardize for Cox numeric stability.

    Three-stage screen, in order:
      1. Drop columns whose pre-imputation NaN fraction exceeds
         ``_MAX_NAN_FRACTION`` — imputing a constant into most rows
         crashes the partial-likelihood Hessian even when the column's
         observed values would otherwise have variance.
      2. Drop columns whose POST-imputation standard deviation falls
         under ``_MIN_USABLE_STD`` — these are effectively constant and
         contribute no signal.
      3. Drop one of any pair of columns with |Pearson r| above
         ``_MAX_PAIRWISE_CORRELATION`` — near-perfect collinearity
         silently produces NaN gradients in lifelines.

    Returns ``(out, means, stds, usable_covariates)`` where
    ``usable_covariates`` is the subset of ``COVARIATES`` that survives
    all three screens.  Dropped columns are still imputed/standardized
    in ``out`` (as constants) so callers can reference them; only
    ``usable_covariates`` should be passed to the fitter.
    """
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    usable: list[str] = []
    out = df.copy()
    n = max(1, len(df))
    for c in COVARIATES:
        col = out[c].astype(float)
        nan_fraction = float(col.isna().sum()) / n
        m = float(col.mean()) if col.notna().any() else 0.0
        s = float(col.std()) if col.notna().any() else 1.0
        too_sparse = nan_fraction > _MAX_NAN_FRACTION
        degenerate = (not math.isfinite(s)) or s <= _MIN_USABLE_STD
        out[c] = col.fillna(m)
        means[c] = m
        stds[c] = max(s, 1.0) if degenerate else s
        out[c] = (out[c] - m) / stds[c]
        if too_sparse or degenerate:
            if too_sparse:
                logger.debug(
                    "Cox covariate %s dropped: %.1f%% NaN before imputation",
                    c, nan_fraction * 100,
                )
            continue
        usable.append(c)

    # Pairwise collinearity screen on the surviving columns.
    if len(usable) >= 2:
        try:
            corr = out[usable].corr().abs()
            to_drop: set[str] = set()
            for i, ci in enumerate(usable):
                if ci in to_drop:
                    continue
                for cj in usable[i + 1:]:
                    if cj in to_drop:
                        continue
                    r = float(corr.loc[ci, cj]) if ci in corr.index and cj in corr.columns else 0.0
                    if math.isfinite(r) and r >= _MAX_PAIRWISE_CORRELATION:
                        to_drop.add(cj)
                        logger.debug(
                            "Cox covariate %s dropped: |r|=%.3f with %s (>%.2f)",
                            cj, r, ci, _MAX_PAIRWISE_CORRELATION,
                        )
            usable = [c for c in usable if c not in to_drop]
        except Exception:
            # Correlation computation failure shouldn't block the fit —
            # keep the existing usable set and let the fitter try.
            pass

    return out, means, stds, usable


def _serialize_baseline_survival(baseline: pd.Series, *, max_points: int = 200) -> dict[str, float]:
    """Subsample the baseline survival curve to a reasonable size for storage."""
    if baseline is None or len(baseline) == 0:
        return {}
    if len(baseline) <= max_points:
        return {f"{float(t):.3f}": float(v) for t, v in baseline.items()}
    idx = np.linspace(0, len(baseline) - 1, max_points).round().astype(int)
    sampled = baseline.iloc[idx]
    return {f"{float(t):.3f}": float(v) for t, v in sampled.items()}


def fit_kaplan_meier(df: pd.DataFrame, *, strata_key: str) -> TrainingResult:
    """Fallback when too few events for Cox PH."""
    kmf = KaplanMeierFitter()
    kmf.fit(durations=df["duration_seconds"], event_observed=df["event_observed"])
    n_events = int(df["event_observed"].sum())
    n_obs = int(len(df))
    return TrainingResult(
        family="kaplan_meier",
        strata_key=strata_key,
        n_events=n_events,
        n_observations=n_obs,
        concordance_index=None,  # KM has no covariates -> no C-index
        brier_score=None,
        log_likelihood=None,
        coefficients={},
        baseline_survival=_serialize_baseline_survival(kmf.survival_function_.iloc[:, 0]),
        feature_means={},
        feature_stds={},
        config={"covariates": [], "imputation": "none"},
        notes=(
            f"Kaplan-Meier baseline only ({n_events} events, {n_obs} obs). "
            "Cox skipped: insufficient events for stable hazard ratios."
        ),
    )


def fit_cox_ph(df: pd.DataFrame, *, strata_key: str, holdout_days: int = 7) -> TrainingResult:
    """Fit Cox PH with chronologically-held-out validation.

    The split is by ``_created_at_epoch`` so the test cohort is
    "later" orders the model has never seen — which is the right
    out-of-sample test for a fill predictor.  Sorting by duration
    biases the test set toward long-duration censored events and
    produces near-zero C-index even on legitimate models.
    """
    sort_col = "_created_at_epoch" if "_created_at_epoch" in df.columns else "duration_seconds"
    df = df.sort_values(sort_col).reset_index(drop=True)
    n_total = len(df)
    holdout_n = max(1, int(n_total * 0.2))
    train = df.iloc[: n_total - holdout_n].copy()
    test = df.iloc[n_total - holdout_n:].copy()

    if int(train["event_observed"].sum()) < 20 or len(train) < 50:
        # Not enough to train Cox sensibly; fall back to KM on full df.
        return fit_kaplan_meier(df, strata_key=strata_key)

    train_imputed, means, stds, usable_covariates = _impute_means(train)

    # If too few covariates have real variance, the Cox model degenerates
    # to KM with extra noise — skip it explicitly instead of letting
    # lifelines emit a wall of ConvergenceWarning + LinAlgWarning before
    # failing.  Three is the minimum for a meaningful proportional-
    # hazards model; below that, KM is strictly more honest.
    if len(usable_covariates) < 3:
        logger.info(
            "Cox PH skipped for strata %s: only %d/%d covariates have non-degenerate "
            "variance (min std=%s) — falling back to KM",
            strata_key,
            len(usable_covariates),
            len(COVARIATES),
            _MIN_USABLE_STD,
        )
        return fit_kaplan_meier(df, strata_key=strata_key)

    fit_columns = ["duration_seconds", "event_observed", *usable_covariates]
    cph = CoxPHFitter(penalizer=0.01)  # small ridge for numeric stability
    try:
        # Suppress lifelines' verbose ConvergenceWarning chatter — we
        # already filtered the obvious offenders, and any residual warning
        # would be misleading noise in the worker logs.  Real failures
        # still raise and are caught by the try/except.
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.filterwarnings("ignore", category=UserWarning, module="lifelines.*")
            _warnings.filterwarnings("ignore", message="Column.*low variance")
            _warnings.filterwarnings("ignore", message="Column.*high sample correlation")
            _warnings.filterwarnings("ignore", message="ill-conditioned matrix")
            cph.fit(
                train_imputed[fit_columns],
                duration_col="duration_seconds",
                event_col="event_observed",
                show_progress=False,
            )
    except Exception as exc:
        logger.warning("Cox PH fit failed for strata %s: %s — falling back to KM", strata_key, exc)
        return fit_kaplan_meier(df, strata_key=strata_key)

    # Validation on held-out.  Standardize using train means/stds, NOT
    # recomputed on test, to avoid leakage.  We only standardize the
    # covariates that were actually fitted; degenerate columns are left
    # in their raw zero-std form (the predict_partial_hazard call below
    # only sees ``usable_covariates``).
    test_imputed = test.copy()
    for c in usable_covariates:
        col = test_imputed[c].astype(float).fillna(means.get(c, 0.0))
        test_imputed[c] = (col - means.get(c, 0.0)) / max(stds.get(c, 1.0), 1e-9)

    try:
        partial_hazards = cph.predict_partial_hazard(test_imputed[usable_covariates])
        c_index = float(
            concordance_index(
                test_imputed["duration_seconds"],
                -partial_hazards,
                test_imputed["event_observed"],
            )
        )
    except Exception as exc:
        logger.warning("C-index calc failed for %s: %s", strata_key, exc)
        c_index = None
        partial_hazards = None

    # Calibration plot data — bucket the held-out cohort by predicted
    # P(fill within median-duration), compute observed fill rate per
    # bucket.  A well-calibrated model has predicted ~= observed across
    # all deciles.  This is what the UI calibration chart consumes.
    calibration_bins: list[dict[str, float]] | None = None
    try:
        if partial_hazards is not None and len(test_imputed) >= 20:
            # Predicted fill probability at the median observed time:
            # P(fill ≤ T_med) = 1 - S0(T_med) ** partial_hazard
            t_median = float(test_imputed["duration_seconds"].median())
            try:
                bs = cph.baseline_survival_at_times([t_median])
                # ``baseline_survival_at_times`` may return either a
                # DataFrame (newer lifelines) or a Series (older).
                if hasattr(bs, "iloc"):
                    if getattr(bs, "ndim", 1) == 2:
                        s0_median = float(bs.iloc[0, 0])
                    else:
                        s0_median = float(bs.iloc[0])
                else:
                    s0_median = float(bs[0])
            except Exception:
                # Older lifelines path: nearest-neighbour lookup on the
                # cumulative ``baseline_survival_`` index.  Coerce the
                # index to a numpy array so .abs() / .argmin() work
                # regardless of whether it's a Float64Index or a
                # generic Index.
                idx = np.asarray(cph.baseline_survival_.index, dtype=float)
                near = int(np.abs(idx - t_median).argmin())
                row = cph.baseline_survival_.iloc[near]
                s0_median = float(row.iloc[0]) if hasattr(row, "iloc") else float(row)
            ph_arr = np.asarray(partial_hazards, dtype=float)
            # Numerical safety: clamp baseline survival into (eps, 1-eps).
            s0_clamped = max(min(s0_median, 1.0 - 1e-9), 1e-9)
            predicted_fill = 1.0 - np.power(s0_clamped, ph_arr)
            obs_fill = test_imputed["event_observed"].astype(float).to_numpy()

            n_bins = min(10, max(3, len(test_imputed) // 8))
            quantile_edges = np.quantile(predicted_fill, np.linspace(0.0, 1.0, n_bins + 1))
            # Make edges strictly increasing (np.digitize wants this).
            for i in range(1, len(quantile_edges)):
                if quantile_edges[i] <= quantile_edges[i - 1]:
                    quantile_edges[i] = quantile_edges[i - 1] + 1e-9
            bin_idx = np.clip(np.digitize(predicted_fill, quantile_edges[1:-1]), 0, n_bins - 1)
            bins: list[dict[str, float]] = []
            for b in range(n_bins):
                mask = bin_idx == b
                if not mask.any():
                    continue
                bins.append(
                    {
                        "bin": int(b),
                        "n": int(mask.sum()),
                        "predicted_mean": float(predicted_fill[mask].mean()),
                        "observed_rate": float(obs_fill[mask].mean()),
                        "predicted_min": float(predicted_fill[mask].min()),
                        "predicted_max": float(predicted_fill[mask].max()),
                    }
                )
            calibration_bins = bins or None
    except Exception as exc:
        logger.debug("Calibration bin computation failed for %s: %s", strata_key, exc)
        calibration_bins = None

    # Hazard ratios = exp(beta).  Beta is in standardized space.
    coefficients: dict[str, float] = {}
    try:
        for cov in usable_covariates:
            if cov in cph.params_.index:
                coefficients[cov] = float(math.exp(cph.params_.loc[cov]))
    except Exception:
        pass

    # Baseline survival serialized for inference-time lookup.
    try:
        baseline = cph.baseline_survival_at_times(np.linspace(1.0, 600.0, 200))
        baseline_dict = {f"{float(t):.3f}": float(v) for t, v in baseline.items()}
    except Exception:
        # Older lifelines API: just use the cumulative survival_function_
        try:
            baseline_dict = _serialize_baseline_survival(cph.baseline_survival_)
        except Exception:
            baseline_dict = {}

    n_events = int(df["event_observed"].sum())
    log_lik = None
    try:
        log_lik = float(cph.log_likelihood_)
    except Exception:
        pass

    return TrainingResult(
        family="cox_ph",
        strata_key=strata_key,
        n_events=n_events,
        n_observations=n_total,
        concordance_index=c_index,
        brier_score=None,  # IBS computation is heavy; skip for V1
        log_likelihood=log_lik,
        coefficients=coefficients,
        baseline_survival=baseline_dict,
        feature_means=means,
        feature_stds=stds,
        config={
            # Record the covariates ACTUALLY fitted so inference time
            # can match the trained-feature set; the original COVARIATES
            # list may include columns we dropped for low variance.
            "covariates": list(usable_covariates),
            "covariates_dropped_low_variance": [
                c for c in COVARIATES if c not in usable_covariates
            ],
            "penalizer": 0.01,
            "holdout_days": holdout_days,
            "holdout_n": holdout_n,
            "train_n": int(len(train)),
        },
        notes=f"Cox PH fit on {len(train)} train / {holdout_n} test rows; {n_events} fills total.",
        calibration_bins=calibration_bins,
    )


async def train_and_persist(
    session: AsyncSession,
    *,
    window_days: int = 30,
    promote_threshold_c_index: float = 0.55,
) -> list[TrainingResult]:
    """End-to-end: pull data, fit, persist row(s), promote if better.

    Returns the TrainingResult list (one per strata) so the caller can
    log/emit metrics.  Always returns; never raises in the critical
    path — fitter exceptions are logged + swallowed because a partial
    failure should not kill the worker loop.
    """
    df = await fetch_training_rows(session, window_days=window_days)
    if df.empty:
        logger.info("Cox trainer: no rows to train on (window_days=%d)", window_days)
        return []

    # Pooled fit only for V1 — promote stratified per-market-type later
    # once each strata has 500+ events.
    pooled = fit_cox_ph(df, strata_key="pooled")
    results = [pooled]

    for r in results:
        row = FillProbabilityModel(
            id=uuid.uuid4().hex,
            family=r.family,
            strata_key=r.strata_key,
            trained_at=datetime.now(timezone.utc),
            training_window_start=datetime.now(timezone.utc) - timedelta(days=window_days),
            training_window_end=datetime.now(timezone.utc),
            n_events=r.n_events,
            n_observations=r.n_observations,
            concordance_index=r.concordance_index,
            brier_score=r.brier_score,
            log_likelihood=r.log_likelihood,
            coefficients_json=r.coefficients,
            baseline_survival_json=r.baseline_survival,
            feature_means_json=r.feature_means,
            feature_stds_json=r.feature_stds,
            config_json={**r.config, "calibration_bins": r.calibration_bins} if r.calibration_bins else r.config,
            promoted_at=None,
            active=False,
            notes=r.notes,
        )
        session.add(row)
        await session.flush()

        # Promotion: activate if (a) no active model in this strata, or
        # (b) new c-index beats the active by margin.
        active_q = await session.execute(
            select(FillProbabilityModel).where(
                FillProbabilityModel.family == r.family,
                FillProbabilityModel.strata_key == r.strata_key,
                FillProbabilityModel.active.is_(True),
            )
        )
        current_active = active_q.scalar_one_or_none()
        should_promote = False
        if current_active is None:
            # Always activate the very first model for the strata, even
            # for KM (covariate-free) baselines.  Inference needs *some*
            # model row to read; better KM than constant.
            should_promote = (
                r.concordance_index is None
                or r.concordance_index >= promote_threshold_c_index
            )
        elif (
            r.concordance_index is not None
            and current_active.concordance_index is not None
            and r.concordance_index > current_active.concordance_index + 0.01
        ):
            should_promote = True

        if should_promote:
            if current_active is not None:
                current_active.active = False
            row.active = True
            row.promoted_at = datetime.now(timezone.utc)
            logger.info(
                "Promoted new %s model for strata=%s (c-index=%.3f, n_events=%d)",
                r.family,
                r.strata_key,
                r.concordance_index or 0.0,
                r.n_events,
            )

    await session.commit()
    return results
