"""Read path for the Cox PH fill model.

Loads the active ``fill_probability_models`` row, caches it in process
memory, and evaluates ``P(fill within Δt | covariates)`` for live
shadow orders.

The trader hot path imports this module — and through it, ``numpy`` —
but explicitly NOT ``lifelines`` or ``pandas`` or ``scipy``.  Inference
is a dot product against learned betas plus a baseline-survival
table lookup; we don't need lifelines to evaluate.

A read-through cache with a 60-second TTL keeps the model lookup
off the DB hot path for the typical scan cadence (2 s loop on the
crypto binaries).  The trainer worker bumps a generation counter
on promotion; if a future feature wants instant invalidation,
swap the TTL for that.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import AsyncSessionLocal, FillProbabilityModel


logger = logging.getLogger("cox_inference")


# Same covariate ordering as cox_trainer — must stay in sync.
COVARIATES = [
    "queue_ahead_shares",
    "depth_behind_shares",
    "spread_bps",
    "mid_distance_bps",
    "recent_trade_intensity_per_sec",
    "time_to_resolution_seconds",
    "ttr_bucket_lt_30s",
    "ttr_bucket_30_60s",
    "ttr_bucket_60_300s",
    "ttr_bucket_300_1800s",
    "side_imbalance",
    "underlying_volatility_bps_per_min",
    "latency_p95_ms",
    "book_age_ms",
    "notional_usd",
]

_CACHE_TTL_SECONDS = 60.0


@dataclass
class FillModelSnapshot:
    family: str
    strata_key: str
    coefficients: dict[str, float]  # standardized-space hazard ratios (exp beta)
    feature_means: dict[str, float]
    feature_stds: dict[str, float]
    baseline_survival: list[tuple[float, float]]  # sorted (t_seconds, S(t))
    n_events: int
    concordance_index: float | None
    trained_at_epoch: float
    promoted_at_epoch: float | None
    notes: str

    def evaluate(self, *, covariates: dict[str, Any], horizon_seconds: float) -> float:
        """P(fill within ``horizon_seconds`` given the covariate snapshot).

        Returns a probability in [0, 1].  Missing covariates are
        imputed with the trained mean (i.e. zero in standardized
        space).  KM models ignore covariates and return 1 - S(t).
        """
        if self.family == "kaplan_meier":
            return _eval_baseline(self.baseline_survival, horizon_seconds)
        # Cox: partial hazard = exp(beta . z), where z is the
        # standardized covariate vector.  S(t | x) = S0(t) ** exp(beta . z).
        log_hazard = 0.0
        for cov in COVARIATES:
            beta = math.log(self.coefficients[cov]) if cov in self.coefficients and self.coefficients[cov] > 0 else 0.0
            mean = self.feature_means.get(cov, 0.0)
            std = max(self.feature_stds.get(cov, 1.0), 1e-9)
            raw = covariates.get(cov)
            if raw is None or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
                z = 0.0  # impute mean -> standardized = 0
            else:
                z = (float(raw) - mean) / std
            log_hazard += beta * z
        partial_hazard = math.exp(log_hazard)
        baseline_s = _eval_baseline_survival(self.baseline_survival, horizon_seconds)
        # Numerically safe S(t)^pH:
        if baseline_s <= 0:
            return 1.0
        if baseline_s >= 1:
            return 0.0
        survival = baseline_s ** partial_hazard
        return float(max(0.0, min(1.0, 1.0 - survival)))


def _eval_baseline_survival(curve: list[tuple[float, float]], t: float) -> float:
    """Step-function lookup of S(t).  Curve sorted by t ascending."""
    if not curve:
        return 1.0
    if t <= curve[0][0]:
        return curve[0][1]
    last_s = curve[0][1]
    for ts, s in curve:
        if ts > t:
            return last_s
        last_s = s
    return last_s


def _eval_baseline(curve: list[tuple[float, float]], t: float) -> float:
    """1 - S(t) for the KM family."""
    s = _eval_baseline_survival(curve, t)
    return float(max(0.0, min(1.0, 1.0 - s)))


def _baseline_from_json(serialized: Any) -> list[tuple[float, float]]:
    if not isinstance(serialized, dict):
        return []
    out: list[tuple[float, float]] = []
    for k, v in serialized.items():
        try:
            t = float(k)
            sv = float(v)
        except Exception:
            continue
        if math.isfinite(t) and math.isfinite(sv):
            out.append((t, sv))
    out.sort(key=lambda x: x[0])
    return out


_cache: dict[str, tuple[float, FillModelSnapshot | None]] = {}


def _cache_key(strata_key: str) -> str:
    return strata_key or "pooled"


async def load_active_fill_model(
    *,
    strata_key: str = "pooled",
    session: AsyncSession | None = None,
) -> FillModelSnapshot | None:
    """Return the active model snapshot for the strata, or None.

    Caches the result for ``_CACHE_TTL_SECONDS`` seconds; the trainer
    worker re-fits at most every few hours so a 60s TTL is
    immaterial to model freshness but huge for hot-path latency.
    """
    cached = _cache.get(_cache_key(strata_key))
    now = time.monotonic()
    if cached is not None:
        ts, snap = cached
        if now - ts < _CACHE_TTL_SECONDS:
            return snap
    snap = await _fetch_active(strata_key=strata_key, session=session)
    _cache[_cache_key(strata_key)] = (now, snap)
    return snap


async def _fetch_active(
    *,
    strata_key: str,
    session: AsyncSession | None,
) -> FillModelSnapshot | None:
    own = session is None
    if own:
        session = AsyncSessionLocal()
        await session.__aenter__()
    try:
        # Prefer Cox over KM if both are active for the strata.
        result = await session.execute(
            select(FillProbabilityModel)
            .where(
                FillProbabilityModel.strata_key == strata_key,
                FillProbabilityModel.active.is_(True),
            )
            .order_by(FillProbabilityModel.family.desc())  # cox_ph > kaplan_meier alphabetically? No, fix:
        )
        rows = list(result.scalars().all())
        if not rows:
            return None
        # Manual ranking: prefer cox_ph if multiple actives.
        rows.sort(key=lambda r: (0 if r.family == "cox_ph" else 1, -(r.n_events or 0)))
        row = rows[0]
        return FillModelSnapshot(
            family=row.family,
            strata_key=row.strata_key,
            coefficients=dict(row.coefficients_json or {}),
            feature_means=dict(row.feature_means_json or {}),
            feature_stds=dict(row.feature_stds_json or {}),
            baseline_survival=_baseline_from_json(row.baseline_survival_json),
            n_events=int(row.n_events or 0),
            concordance_index=row.concordance_index,
            trained_at_epoch=row.trained_at.timestamp() if row.trained_at else 0.0,
            promoted_at_epoch=row.promoted_at.timestamp() if row.promoted_at else None,
            notes=str(row.notes or ""),
        )
    finally:
        if own:
            await session.__aexit__(None, None, None)


def evaluate_fill_probability(
    *,
    snapshot: FillModelSnapshot | None,
    covariates: dict[str, Any],
    horizon_seconds: float,
    fallback_probability: float = 0.5,
) -> float:
    """Convenience: ``P(fill within Δt)`` falling back to a constant
    when no model is loaded yet."""
    if snapshot is None:
        return float(max(0.0, min(1.0, fallback_probability)))
    return snapshot.evaluate(covariates=covariates, horizon_seconds=horizon_seconds)


def invalidate_cache() -> None:
    """Force the next ``load_active_fill_model`` to re-read the DB.
    Called by the UI on manual retrain / promote."""
    _cache.clear()
