"""Pure timeframe-normalization utilities for crypto strategies.

Each crypto strategy owns its own ``default_config`` inline — there is
no shared defaults dict here. This module is intentionally just two
small utility functions used during evaluate() to canonicalize
timeframe strings and look up timeframe-suffixed config overrides like
``take_profit_pct_5m``.

Compat note (2026-05): the BTC/ETH strategies in the DB still
``from services.strategy_helpers.crypto_scope import (CRYPTO_HF_SCOPE_DEFAULTS,
_crypto_hf_default_param_value, merge_crypto_defaults, ...)`` even
though those symbols were stripped from this module long ago.  Without
the stubs below those strategies fail to load at all (every backtest
crashes with ``ImportError: cannot import name 'CRYPTO_HF_SCOPE_DEFAULTS'``).
The stubs are intentionally empty / pass-through — each strategy's
own ``default_config`` and DB config layer already supply real
parameter values through ``BaseStrategy.configure``.
"""
from __future__ import annotations

from typing import Any


CRYPTO_HF_SCOPE_DEFAULTS: dict[str, Any] = {}
"""Compat shim: every strategy's real defaults live in their own
``default_config`` dict.  Returning an empty dict here means strategies
that build ``default_config = dict(CRYPTO_HF_SCOPE_DEFAULTS)`` start
empty and get their values from later assignments + DB config — same
end state as before the symbol was deleted from this module."""


def _crypto_hf_default_param_value(key: str, timeframe: Any) -> Any:
    """Compat shim — historically returned a per-(key, timeframe) default
    from a shared registry that no longer exists.  Strategies wrap every
    call in ``_coerce_float(_crypto_hf_default_param_value(...), <fallback>, ...)``,
    so returning None lets the fallback kick in unchanged."""
    return None


def merge_crypto_defaults(config: Any) -> Any:
    """Compat shim — historically merged a shared defaults dict into the
    strategy config.  With the shared dict gone we just pass the config
    through; strategies that override ``configure()`` to call this still
    end up with their own ``default_config`` applied by ``BaseStrategy.configure``."""
    return config or {}


def _normalize_timeframe(timeframe: Any) -> str:
    """Canonicalize a timeframe string to ``5m`` / ``15m`` / ``1h`` / ``4h``."""
    tf = str(timeframe or "").strip().lower()
    if tf in {"5m", "5min", "5"}:
        return "5m"
    if tf in {"15m", "15min", "15"}:
        return "15m"
    if tf in {"1h", "1hr", "60m", "60min"}:
        return "1h"
    if tf in {"4h", "4hr", "240m", "240min"}:
        return "4h"
    return tf


def _timeframe_override(config: Any, base_key: str, timeframe: str | None) -> Any:
    """Return ``config[f"{base_key}_{timeframe}"]`` when present, else None.

    Pure lookup — no defaults, no merging. Each strategy looks up its own
    config + its own default_config separately.
    """
    if timeframe and isinstance(config, dict):
        tf_key = f"{base_key}_{timeframe}"
        if tf_key in config:
            return config[tf_key]
    return None
