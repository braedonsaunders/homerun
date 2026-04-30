"""World-class shadow-trading fill simulator.

This package owns the parts of the simulator that go past the
"depth-ahead-must-clear" heuristic everyone else uses:

* ``survival_features`` — extract the covariate snapshot every shadow
  order needs labeled with at placement time.  These get persisted into
  ``TraderOrder.payload_json["survival_features"]`` and become the
  training rows for ``cox_trainer``.
* ``cox_trainer`` — nightly batch job that joins TraderOrder fill /
  cancel / expire events against the recorded book history, fits a
  Cox proportional hazards (or Kaplan-Meier fallback) model, validates
  with C-index + Brier score, and promotes a new model row in
  ``fill_probability_models`` when it beats the active one.
* ``cox_inference`` — at-decision-time read path: load the active
  Cox model, evaluate ``P(fill within Δt | covariates)``.
* ``ensemble`` — pessimistic / realistic / optimistic three-way fill
  estimates per shadow order; the diff is your calibration signal.

Public surface:
    from services.fill_simulator import (
        build_survival_features,
        load_active_fill_model,
        evaluate_fill_probability,
        ensemble_fill_estimates,
    )
"""
from services.fill_simulator.survival_features import (
    SurvivalFeatures,
    build_survival_features,
)

__all__ = [
    "SurvivalFeatures",
    "build_survival_features",
]
