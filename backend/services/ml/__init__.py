"""Strategy-driven ML capabilities registry.

The principle: ML tasks aren't a separate registry of hardcoded
Python classes — they're a *capability* a strategy declares. Each
``BaseStrategy`` subclass that wants to feed an ML model attaches an
``ml_capability = MLCapability(...)`` class attribute, and the
registry iterates loaded strategies to find them.

This replaces the old ``services/machine_learning_tasks/`` registry
that pre-registered exactly one task and rejected any other key.

Backwards-compat shim: ``services.machine_learning_tasks`` now
delegates to ``ml_strategy_registry`` so the existing 18+ callers
in ``machine_learning_sdk.py`` keep working unchanged.
"""
from services.ml.capabilities import MLCapability
from services.ml.strategy_ml_registry import (
    ml_strategy_registry,
    get_ml_capability,
    list_ml_capabilities,
    register_ml_capability,
)

__all__ = [
    "MLCapability",
    "ml_strategy_registry",
    "get_ml_capability",
    "list_ml_capabilities",
    "register_ml_capability",
]
