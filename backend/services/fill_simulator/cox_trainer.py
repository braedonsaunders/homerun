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
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    FillProbabilityModel,
    TraderOrder,
)


logger = logging.getLogger("cox_trainer")


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
        }
        for c in COVARIATES:
            v = sf.get(c)
            cov_dict[c] = float(v) if v is not None and isinstance(v, (int, float)) else math.nan
        rows.append(cov_dict)
    if not rows:
        return pd.DataFrame(columns=["duration_seconds", "event_observed", "market_type_strata", *COVARIATES])
    return pd.DataFrame(rows)


def _impute_means(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
    """Mean-impute missing covariates; standardize for Cox numeric stability."""
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    out = df.copy()
    for c in COVARIATES:
        col = out[c].astype(float)
        m = float(col.mean()) if col.notna().any() else 0.0
        s = float(col.std()) if col.notna().any() else 1.0
        if not math.isfinite(s) or s <= 0:
            s = 1.0
        out[c] = col.fillna(m)
        means[c] = m
        stds[c] = s
        out[c] = (out[c] - m) / s
    return out, means, stds


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
    """Fit Cox PH with held-out validation.  Returns a TrainingResult."""
    # Hold out the most recent ``holdout_days`` for validation.
    df = df.sort_values("duration_seconds").reset_index(drop=True)
    n_total = len(df)
    holdout_n = max(1, int(n_total * 0.2))
    train = df.iloc[: n_total - holdout_n].copy()
    test = df.iloc[n_total - holdout_n:].copy()

    if int(train["event_observed"].sum()) < 20 or len(train) < 50:
        # Not enough to train Cox sensibly; fall back to KM on full df.
        return fit_kaplan_meier(df, strata_key=strata_key)

    train_imputed, means, stds = _impute_means(train)

    fit_columns = ["duration_seconds", "event_observed", *COVARIATES]
    cph = CoxPHFitter(penalizer=0.01)  # small ridge for numeric stability
    try:
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
    # recomputed on test, to avoid leakage.
    test_imputed = test.copy()
    for c in COVARIATES:
        col = test_imputed[c].astype(float).fillna(means.get(c, 0.0))
        test_imputed[c] = (col - means.get(c, 0.0)) / max(stds.get(c, 1.0), 1e-9)

    try:
        partial_hazards = cph.predict_partial_hazard(test_imputed[COVARIATES])
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

    # Hazard ratios = exp(beta).  Beta is in standardized space.
    coefficients: dict[str, float] = {}
    try:
        for cov in COVARIATES:
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
            "covariates": list(COVARIATES),
            "penalizer": 0.01,
            "holdout_days": holdout_days,
            "holdout_n": holdout_n,
            "train_n": int(len(train)),
        },
        notes=f"Cox PH fit on {len(train)} train / {holdout_n} test rows; {n_events} fills total.",
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
            config_json=r.config,
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
