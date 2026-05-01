"""Registry that resolves ``task_key`` to an ``MLCapability``.

Two registration paths feed this registry:

1. **Built-in capabilities** — registered via ``register_ml_capability``
   at module-import time.  Used for capabilities that don't have a
   strategy yet (e.g. the migrated ``crypto_directional`` task that
   the platform records passively for any matching market).

2. **Strategy-attached capabilities** — discovered by iterating
   ``strategy_loader.get_all_instances()`` on each lookup.  A
   strategy class can declare ``ml_capability = MLCapability(...)``
   as a class attribute and the registry picks it up automatically.

Resolution precedence on lookup:
- Strategy-attached wins over built-in (lets a strategy override the
  default ``crypto_directional`` capability if it wants stricter
  scope filters or different features).
- Within strategies, first-loaded wins; warn on duplicate task_key.
"""
from __future__ import annotations

import logging
from typing import Any

from services.ml.capabilities import MLCapability


logger = logging.getLogger("services.ml.registry")


class _MLStrategyRegistry:
    def __init__(self) -> None:
        self._builtins: dict[str, MLCapability] = {}

    def register(self, capability: MLCapability) -> None:
        key = str(capability.task_key or "").strip().lower()
        if not key:
            raise ValueError("MLCapability.task_key must be non-empty")
        if key in self._builtins:
            logger.warning("Re-registering built-in MLCapability '%s' (overwriting)", key)
        self._builtins[key] = capability

    def get(self, task_key: str) -> MLCapability:
        key = str(task_key or "").strip().lower()
        # First check strategy-attached.
        from_strategy = self._collect_from_loaded_strategies().get(key)
        if from_strategy is not None:
            return from_strategy
        # Then built-ins.
        builtin = self._builtins.get(key)
        if builtin is not None:
            return builtin
        raise ValueError(f"Unsupported machine learning task '{task_key}'")

    def list(self) -> list[MLCapability]:
        # Merge both sources, strategy-attached wins on key collision.
        out: dict[str, MLCapability] = dict(self._builtins)
        for key, cap in self._collect_from_loaded_strategies().items():
            out[key] = cap
        return list(out.values())

    def _collect_from_loaded_strategies(self) -> dict[str, MLCapability]:
        """Iterate currently-loaded strategy instances and pull
        their ``ml_capability`` attributes."""
        out: dict[str, MLCapability] = {}
        try:
            from services.strategy_loader import strategy_loader

            instances = strategy_loader.get_all_instances() or []
        except Exception:
            return out
        for instance in instances:
            cap = getattr(instance, "ml_capability", None)
            if cap is None:
                continue
            if not isinstance(cap, MLCapability):
                continue
            key = str(cap.task_key or "").strip().lower()
            if not key:
                continue
            if key in out:
                logger.warning(
                    "Duplicate ml_capability task_key '%s' on multiple strategies; keeping first",
                    key,
                )
                continue
            # Stamp the owning strategy slug (overrides any prior value).
            slug = getattr(instance, "slug", None) or type(instance).__name__
            cap.owner_strategy_slug = str(slug)
            out[key] = cap
        return out


ml_strategy_registry = _MLStrategyRegistry()


def register_ml_capability(capability: MLCapability) -> None:
    ml_strategy_registry.register(capability)


def get_ml_capability(task_key: str) -> MLCapability:
    return ml_strategy_registry.get(task_key)


def list_ml_capabilities() -> list[MLCapability]:
    return ml_strategy_registry.list()


# ----------------------------------------------------------------------
# Boot-time built-in registrations.
# ----------------------------------------------------------------------

def _bootstrap_builtins() -> None:
    """Register fallback built-in capabilities for keys not owned by
    any strategy.

    ``crypto_directional`` was a built-in here until it migrated onto
    ``BtcEthDirectionalEdgeStrategy.ml_capability`` (see Item 2 of the
    Phase-12 cleanup).  The fallback registration below covers the
    ~1 second window during cold worker boot when strategies haven't
    been loaded yet — once the StrategyLoader populates instances,
    the strategy-attached version takes precedence on every lookup.

    Removing this fallback entirely is safe ONLY if every consumer
    can tolerate "task not registered yet" errors during the boot
    grace period.  Today the recorder + scanner can tolerate it
    (they retry next tick), so we keep this minimal-info fallback
    rather than risking startup races.
    """
    ml_strategy_registry.register(
        MLCapability(
            task_key="crypto_directional",
            label="Crypto Directional",
            description=(
                "Directional probability for live crypto markets. "
                "Strategy-attached on BtcEthDirectionalEdgeStrategy; "
                "this built-in is a cold-boot fallback only."
            ),
            allowed_assets=("btc", "eth", "sol", "xrp"),
            allowed_timeframes=("5m", "15m", "1h", "4h"),
            default_lookback=5,
            owner_strategy_slug=None,
        )
    )


_bootstrap_builtins()
