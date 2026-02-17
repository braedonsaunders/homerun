from __future__ import annotations

from services.strategies.base import BaseStrategy
from services.trader_orchestrator.strategy_db_loader import strategy_db_loader
from services.strategies.btc_eth_highfreq import BtcEthHighFreqStrategy
from services.strategies.news_edge import NewsEdgeStrategy
from services.strategies.basic import BasicArbStrategy
from services.strategies.negrisk import NegRiskStrategy
from services.strategies.flash_crash_reversion import FlashCrashReversionStrategy
from services.strategies.tail_end_carry import TailEndCarryStrategy
from services.strategies.traders_confluence import TradersConfluenceStrategy
from services.strategies.weather_edge import WeatherEdgeStrategy
from services.strategies.weather_distribution import WeatherDistributionStrategy
from services.strategies.crypto_spike_reversion import CryptoSpikeReversionStrategy


_REFERENCE_STRATEGIES: dict[str, BaseStrategy] = {
    "crypto_5m": BtcEthHighFreqStrategy(),
    "crypto_15m": BtcEthHighFreqStrategy(),
    "crypto_1h": BtcEthHighFreqStrategy(),
    "crypto_4h": BtcEthHighFreqStrategy(),
    "news_reaction": NewsEdgeStrategy(),
    "opportunity_general": BasicArbStrategy(),
    "opportunity_structural": NegRiskStrategy(),
    "opportunity_flash_reversion": FlashCrashReversionStrategy(),
    "opportunity_tail_carry": TailEndCarryStrategy(),
    "weather_consensus": WeatherEdgeStrategy(),
    "weather_alerts": WeatherDistributionStrategy(),
    "traders_flow": TradersConfluenceStrategy(),
    "crypto_spike_reversion": CryptoSpikeReversionStrategy(),
}

_STRATEGY_ALIASES: dict[str, str] = {
    "strategy.default": "crypto_15m",
    "default": "crypto_15m",
    "opportunity_weather": "opportunity_general",
}


def _resolve_strategy_key(strategy_key: str) -> str:
    key = str(strategy_key or "").strip().lower()
    key = _STRATEGY_ALIASES.get(key, key)
    key = strategy_db_loader.resolve_key(key)
    return key


def list_strategy_keys(*, include_reference: bool = True) -> list[str]:
    loaded = set(strategy_db_loader.list_strategy_keys())
    if include_reference:
        loaded.update(_REFERENCE_STRATEGIES.keys())
    return sorted(loaded)


def get_strategy(strategy_key: str, *, use_static_fallback: bool = True) -> BaseStrategy:
    key = _resolve_strategy_key(strategy_key)
    loaded = strategy_db_loader.get_strategy(key)
    if loaded is not None:
        return loaded
    if use_static_fallback:
        return _REFERENCE_STRATEGIES.get(key, BaseStrategy())
    return BaseStrategy()
