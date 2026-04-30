"""Pessimistic / realistic / optimistic fill estimate ensemble.

Replaces the single-point fill estimate with a three-way bracket so
strategies and the UI can show PnL bands rather than fragile point
forecasts.

The three estimates differ only in:

1. **Latency assumption**: pessimistic uses measured p95, realistic
   p50, optimistic 0.5 * p50.
2. **Fill probability multiplier**: pessimistic shaves 25% off the
   Cox prediction (or the heuristic estimate), realistic uses the
   raw model output, optimistic adds 15%.
3. **Effective queue ahead**: pessimistic adds 20% to queue_ahead
   (assume your priority is worse than measured), realistic uses
   estimator output, optimistic subtracts 20%.

The spread between pessimistic and optimistic is the calibration
signal — wide spread means uncertain regime, tight spread means
confident.  Backtests display all three; live shadow stores all
three on the order payload.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from services.fill_simulator.cox_inference import (
    FillModelSnapshot,
    evaluate_fill_probability,
)
from services.fill_simulator.empirical_constants import (
    EmpiricalConstants,
    get_empirical_constants,
)
from services.fill_simulator.latency import (
    LatencyDistribution,
    measured_latency_cached,
)
from services.optimization.execution_estimator import (
    ExecutionEstimate,
    ExecutionEstimator,
    ExecutionEstimatorConfig,
)


@dataclass
class EnsembleScenario:
    label: str  # "pessimistic" | "realistic" | "optimistic"
    estimate: ExecutionEstimate
    fill_probability: float  # P(fill within time_in_force) — model-derived if available
    latency_ms: float
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "fill_probability": self.fill_probability,
            "latency_ms": self.latency_ms,
            "notes": self.notes,
            "estimate": self.estimate.to_dict(),
        }


@dataclass
class EnsembleResult:
    pessimistic: EnsembleScenario
    realistic: EnsembleScenario
    optimistic: EnsembleScenario
    cox_loaded: bool
    cox_strata: str
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pessimistic": self.pessimistic.to_dict(),
            "realistic": self.realistic.to_dict(),
            "optimistic": self.optimistic.to_dict(),
            "cox_loaded": self.cox_loaded,
            "cox_strata": self.cox_strata,
            "notes": self.notes,
            "extra": dict(self.extra),
        }


_ESTIMATOR = ExecutionEstimator()


def _config_for_scenario(
    *,
    constants: EmpiricalConstants,
    latency_ms: float,
    time_in_force_seconds: float,
    fee_bps: float,
    queue_multiplier: float,
) -> ExecutionEstimatorConfig:
    return ExecutionEstimatorConfig(
        fee_bps=fee_bps,
        latency_ms=max(20.0, float(latency_ms)),
        time_in_force_seconds=time_in_force_seconds,
        displayed_depth_factor=constants.displayed_depth_factor,
        min_depth_factor=constants.min_depth_factor,
        max_book_age_ms=10_000.0,
        stale_depth_decay=constants.stale_depth_decay,
        maker_queue_ahead_fraction=min(
            0.99,
            max(0.05, constants.maker_queue_ahead_fraction * queue_multiplier),
        ),
        maker_trade_flow_multiplier=constants.maker_trade_flow_multiplier,
        adverse_selection_multiplier=constants.adverse_selection_multiplier,
        recent_trade_lookback_seconds=30.0,
    )


def _model_fill_probability(
    *,
    cox_snapshot: FillModelSnapshot | None,
    survival_covariates: dict[str, Any] | None,
    horizon_seconds: float,
    fallback: float,
    haircut: float,
) -> float:
    if not survival_covariates:
        return float(max(0.0, min(1.0, fallback * haircut)))
    p = evaluate_fill_probability(
        snapshot=cox_snapshot,
        covariates=survival_covariates,
        horizon_seconds=horizon_seconds,
        fallback_probability=fallback,
    )
    return float(max(0.0, min(1.0, p * haircut)))


def ensemble_estimate(
    *,
    order_book: Any,
    side: str,
    size_shares: float,
    limit_price: float,
    order_type: str,
    recent_trades: list[Any] | None,
    book_age_ms: float | None,
    time_in_force_seconds: float = 6.0,
    fee_bps: float = 0.0,
    cox_snapshot: FillModelSnapshot | None = None,
    survival_covariates: dict[str, Any] | None = None,
    latency: LatencyDistribution | None = None,
    constants: EmpiricalConstants | None = None,
) -> EnsembleResult:
    """Compute three concurrent fill estimates and return the bracket."""
    constants = constants or get_empirical_constants()
    latency = latency or measured_latency_cached()

    # Pessimistic: high latency, expand queue ahead, haircut Cox p.
    pess_cfg = _config_for_scenario(
        constants=constants,
        latency_ms=latency.pessimistic_ms,
        time_in_force_seconds=time_in_force_seconds,
        fee_bps=fee_bps,
        queue_multiplier=1.20,
    )
    pess_est = _ESTIMATOR.estimate_order(
        order_book=order_book,
        side=side,
        size_shares=size_shares,
        limit_price=limit_price,
        order_type=order_type,
        recent_trades=recent_trades or [],
        book_age_ms=book_age_ms,
        config=pess_cfg,
    )
    pess_p = _model_fill_probability(
        cox_snapshot=cox_snapshot,
        survival_covariates=survival_covariates,
        horizon_seconds=time_in_force_seconds,
        fallback=pess_est.fill_probability,
        haircut=0.75,
    )

    # Realistic: median latency, default queue, raw Cox p.
    real_cfg = _config_for_scenario(
        constants=constants,
        latency_ms=latency.realistic_ms,
        time_in_force_seconds=time_in_force_seconds,
        fee_bps=fee_bps,
        queue_multiplier=1.0,
    )
    real_est = _ESTIMATOR.estimate_order(
        order_book=order_book,
        side=side,
        size_shares=size_shares,
        limit_price=limit_price,
        order_type=order_type,
        recent_trades=recent_trades or [],
        book_age_ms=book_age_ms,
        config=real_cfg,
    )
    real_p = _model_fill_probability(
        cox_snapshot=cox_snapshot,
        survival_covariates=survival_covariates,
        horizon_seconds=time_in_force_seconds,
        fallback=real_est.fill_probability,
        haircut=1.0,
    )

    # Optimistic: low latency, tighter queue, slight model bump.
    opt_cfg = _config_for_scenario(
        constants=constants,
        latency_ms=latency.optimistic_ms,
        time_in_force_seconds=time_in_force_seconds,
        fee_bps=fee_bps,
        queue_multiplier=0.80,
    )
    opt_est = _ESTIMATOR.estimate_order(
        order_book=order_book,
        side=side,
        size_shares=size_shares,
        limit_price=limit_price,
        order_type=order_type,
        recent_trades=recent_trades or [],
        book_age_ms=book_age_ms,
        config=opt_cfg,
    )
    opt_p = _model_fill_probability(
        cox_snapshot=cox_snapshot,
        survival_covariates=survival_covariates,
        horizon_seconds=time_in_force_seconds,
        fallback=opt_est.fill_probability,
        haircut=1.15,
    )

    constants_note = constants.notes
    return EnsembleResult(
        pessimistic=EnsembleScenario(
            label="pessimistic",
            estimate=pess_est,
            fill_probability=pess_p,
            latency_ms=latency.pessimistic_ms,
            notes="p95 latency, +20% queue priors, -25% Cox p",
        ),
        realistic=EnsembleScenario(
            label="realistic",
            estimate=real_est,
            fill_probability=real_p,
            latency_ms=latency.realistic_ms,
            notes="p50 latency, raw Cox p",
        ),
        optimistic=EnsembleScenario(
            label="optimistic",
            estimate=opt_est,
            fill_probability=opt_p,
            latency_ms=latency.optimistic_ms,
            notes="p50/2 latency, -20% queue priors, +15% Cox p",
        ),
        cox_loaded=cox_snapshot is not None,
        cox_strata=cox_snapshot.strata_key if cox_snapshot else "",
        notes=f"empirical constants: {constants_note}",
        extra={
            "latency_p50_ms": latency.realistic_ms,
            "latency_p95_ms": latency.pessimistic_ms,
            "latency_sample_count": latency.sample_count,
            "constants_measured": constants.measured,
            "constants_sample_count": constants.sample_count,
        },
    )
