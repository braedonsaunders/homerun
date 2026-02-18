"""Strategy registry — thin wrapper around the unified strategy_loader.

All strategies live in the DB and are loaded by the unified loader.
This module provides convenience accessors for callers that need a
strategy instance by slug.
"""

from __future__ import annotations

from services.strategies.base import BaseStrategy
from services.strategies.btc_eth_highfreq import BtcEthHighFreqStrategy
from services.strategy_loader import strategy_loader


def list_strategy_keys() -> list[str]:
    return sorted(slug for slug in strategy_loader._loaded)


def get_strategy(strategy_key: str) -> BaseStrategy:
    """Return the BaseStrategy instance for *strategy_key*, or a default."""
    key = str(strategy_key or "").strip().lower()
    if key in {"strategy.default", "default"}:
        legacy_default = strategy_loader.get_instance("btc_eth_highfreq")
        if legacy_default is not None:
            return legacy_default
        return BtcEthHighFreqStrategy()

    instance = strategy_loader.get_instance(key)
    if instance is not None:
        return instance
    return BaseStrategy()
