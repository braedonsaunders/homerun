"""Strategy registry — thin wrapper around the unified strategy_loader.

All strategies live in the DB and are loaded by the unified loader.
This module provides convenience accessors for callers that need a
strategy instance by slug.
"""

from __future__ import annotations

from services.strategies.base import BaseStrategy
from services.strategy_loader import strategy_loader

_LEGACY_STRATEGY_ALIASES = {
    "strategy.default": "btc_eth_highfreq",
}


def _resolve_strategy_key(strategy_key: str) -> str:
    key = str(strategy_key or "").strip().lower()
    return _LEGACY_STRATEGY_ALIASES.get(key, key)


def list_strategy_keys() -> list[str]:
    return sorted(slug for slug in strategy_loader._loaded)


def get_strategy(strategy_key: str) -> BaseStrategy:
    """Return the BaseStrategy instance for *strategy_key*, or a default."""
    raw_key = str(strategy_key or "").strip().lower()
    if raw_key == "strategy.default":
        from services.strategies.btc_eth_highfreq import BtcEthHighFreqStrategy

        return BtcEthHighFreqStrategy()

    key = _resolve_strategy_key(raw_key)
    instance = strategy_loader.get_instance(key)
    if instance is not None:
        return instance
    if key == "btc_eth_highfreq":
        from services.strategies.btc_eth_highfreq import BtcEthHighFreqStrategy

        return BtcEthHighFreqStrategy()
    return BaseStrategy()
