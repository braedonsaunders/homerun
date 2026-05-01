"""Backwards-compatibility shim — delegates to the new strategy-driven
``services.ml`` registry.

Historically this module was the source of truth, registering exactly
one ``CryptoDirectionalTask`` instance and rejecting any other key.
Today the source of truth is ``services.ml.strategy_ml_registry``,
which discovers MLCapabilities from loaded strategies AND from
built-in registrations.  This module preserves the old function
signatures so the 18+ callers in ``machine_learning_sdk.py`` keep
working unchanged.

The legacy ``CryptoDirectionalTask`` class still lives at
``services.machine_learning_tasks.crypto_directional`` because the
new ``MLCapability`` delegates to it for the heavy feature-engineering
methods.  When a strategy migrates and provides its own implementations
on the ``MLCapability``, the delegate isn't consulted.
"""
from __future__ import annotations

from typing import Any

from services.machine_learning_tasks.crypto_directional import CryptoDirectionalTask
from services.ml.strategy_ml_registry import (
    get_ml_capability as _get_ml_capability,
    list_ml_capabilities as _list_ml_capabilities,
)


def get_machine_learning_task(task_key: str) -> Any:
    """Resolve a task_key to its capability/task object.

    Returns either a strategy-attached or built-in ``MLCapability``.
    Existing callers in ``machine_learning_sdk.py`` use it like the
    old ``CryptoDirectionalTask`` instance, calling methods like
    ``normalize_assets``, ``build_snapshot_record``, etc.  ``MLCapability``
    exposes the same surface (delegating to ``CryptoDirectionalTask``
    under the hood for the heavy methods on the legacy task_key).
    """
    return _get_ml_capability(task_key)


def list_machine_learning_tasks() -> list[Any]:
    return _list_ml_capabilities()


__all__ = [
    "CryptoDirectionalTask",
    "get_machine_learning_task",
    "list_machine_learning_tasks",
]
