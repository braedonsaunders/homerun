"""MLCapability — the contract a strategy declares to opt into ML.

Replaces ``services.machine_learning_tasks.CryptoDirectionalTask`` as
the data shape passed around the system.  Strategies that want their
markets recorded + a model trained + adapters fitted attach an
``ml_capability = MLCapability(...)`` class attribute.

Two ways to use it:

1. **Built-in capability** — a module-level instance registered via
   ``register_ml_capability(...)`` at import time.  This is the
   migration path for ``crypto_directional`` so existing data /
   adapters / models remain valid against an unchanged ``task_key``.

2. **Strategy-attached** — set as a class attribute on a strategy
   subclass.  The registry iterates loaded strategies on access and
   discovers their declarations dynamically.

The interface intentionally mirrors the old ``CryptoDirectionalTask``
methods so ``machine_learning_sdk.py`` callers don't need to change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Protocol


class _SnapshotBuilder(Protocol):
    def __call__(self, market: dict[str, Any], *, recorded_at: datetime) -> dict[str, Any] | None: ...


class _ScopeFilter(Protocol):
    def __call__(
        self,
        market: dict[str, Any],
        *,
        assets: list[str],
        timeframes: list[str],
    ) -> bool: ...


@dataclass
class MLCapability:
    """Plain-data contract describing one ML task.

    ``task_key`` is the canonical identifier persisted on every
    snapshot / model / adapter / deployment row.  Strategies that
    don't declare ``task_key`` get their slug used (so adding ML to
    a strategy is a one-attribute edit).
    """

    task_key: str
    label: str
    description: str = ""
    allowed_assets: tuple[str, ...] = ()
    allowed_timeframes: tuple[str, ...] = ()
    default_lookback: int = 5
    feature_names: tuple[str, ...] = ()

    # Behavior — wrapped in optional callables so strategies can
    # override or extend without subclassing.  When set to None, the
    # registry falls back to the implementation provided by the
    # built-in delegate (see services/ml/builtin/).
    _scope_matches: _ScopeFilter | None = field(default=None, repr=False)
    _build_snapshot_record: _SnapshotBuilder | None = field(default=None, repr=False)
    _build_training_dataset: Callable[..., Any] | None = field(default=None, repr=False)
    _feature_vector_from_market: Callable[..., Any] | None = field(default=None, repr=False)

    # Strategy slug that owns this capability (None for built-ins).
    owner_strategy_slug: str | None = None

    # ------------------------------------------------------------------
    # Normalization helpers — ported verbatim from CryptoDirectionalTask
    # so existing callers see no behavior change.
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_asset(value: Any) -> str:
        return str(value or "").strip().lower()

    @staticmethod
    def normalize_timeframe(value: Any) -> str:
        text = str(value or "").strip().lower()
        if text in {"5m", "5min", "5-minute", "5minutes"}:
            return "5m"
        if text in {"15m", "15min", "15-minute", "15minutes"}:
            return "15m"
        if text in {"1h", "1hr", "1hour", "60m"}:
            return "1h"
        if text in {"4h", "4hr", "4hour", "240m"}:
            return "4h"
        return text

    def normalize_assets(self, values: list[str] | None) -> list[str]:
        normalized: list[str] = []
        source = list(values or self.allowed_assets)
        for value in source:
            asset = self.normalize_asset(value)
            if asset in self.allowed_assets and asset not in normalized:
                normalized.append(asset)
        return normalized or list(self.allowed_assets)

    def normalize_timeframes(self, values: list[str] | None) -> list[str]:
        normalized: list[str] = []
        source = list(values or self.allowed_timeframes)
        for value in source:
            timeframe = self.normalize_timeframe(value)
            if timeframe in self.allowed_timeframes and timeframe not in normalized:
                normalized.append(timeframe)
        return normalized or list(self.allowed_timeframes)

    def default_feature_names(self, *, lookback: int | None = None) -> list[str]:
        # Strategy-extended feature lists override the built-in default.
        if self.feature_names:
            return list(self.feature_names)
        # Fallback: ask the bound delegate (e.g. CryptoDirectionalTask).
        delegate = self._delegate()
        if delegate and hasattr(delegate, "default_feature_names"):
            return delegate.default_feature_names(lookback=lookback)
        return []

    def scope_matches(
        self,
        market: dict[str, Any],
        *,
        assets: list[str],
        timeframes: list[str],
    ) -> bool:
        if self._scope_matches is not None:
            return self._scope_matches(market, assets=assets, timeframes=timeframes)
        delegate = self._delegate()
        if delegate and hasattr(delegate, "scope_matches"):
            return delegate.scope_matches(market, assets=assets, timeframes=timeframes)
        # Default: simple asset+timeframe check.
        asset = self.normalize_asset(market.get("asset"))
        timeframe = self.normalize_timeframe(market.get("timeframe"))
        return asset in set(assets) and timeframe in set(timeframes)

    def build_snapshot_record(self, market: dict[str, Any], *, recorded_at: datetime) -> dict[str, Any] | None:
        if self._build_snapshot_record is not None:
            return self._build_snapshot_record(market, recorded_at=recorded_at)
        delegate = self._delegate()
        if delegate and hasattr(delegate, "build_snapshot_record"):
            return delegate.build_snapshot_record(market, recorded_at=recorded_at)
        return None

    def build_training_dataset(self, *args: Any, **kwargs: Any) -> Any:
        if self._build_training_dataset is not None:
            return self._build_training_dataset(*args, **kwargs)
        delegate = self._delegate()
        if delegate and hasattr(delegate, "build_training_dataset"):
            return delegate.build_training_dataset(*args, **kwargs)
        raise NotImplementedError(f"build_training_dataset not implemented for {self.task_key}")

    def feature_vector_from_market(self, *args: Any, **kwargs: Any) -> Any:
        if self._feature_vector_from_market is not None:
            return self._feature_vector_from_market(*args, **kwargs)
        delegate = self._delegate()
        if delegate and hasattr(delegate, "feature_vector_from_market"):
            return delegate.feature_vector_from_market(*args, **kwargs)
        return None

    # ------------------------------------------------------------------
    # Delegate — the heavy implementation hides here.  For migration
    # we re-use the existing CryptoDirectionalTask methods rather than
    # duplicating their feature math.
    # ------------------------------------------------------------------

    _delegate_cache: Any = field(default=None, repr=False)

    def _delegate(self) -> Any | None:
        if self._delegate_cache is not None:
            return self._delegate_cache
        # Lazy-import to avoid cycles.
        if self.task_key == "crypto_directional":
            try:
                from services.machine_learning_tasks.crypto_directional import (
                    CryptoDirectionalTask,
                )

                self._delegate_cache = CryptoDirectionalTask()
                return self._delegate_cache
            except Exception:
                return None
        return None

    def to_summary(self) -> dict[str, Any]:
        """JSON-safe summary used by the API/UI listing endpoints."""
        return {
            "task_key": self.task_key,
            "label": self.label,
            "description": self.description,
            "allowed_assets": list(self.allowed_assets),
            "allowed_timeframes": list(self.allowed_timeframes),
            "default_lookback": self.default_lookback,
            "owner_strategy_slug": self.owner_strategy_slug,
            "feature_names": self.default_feature_names(),
        }
