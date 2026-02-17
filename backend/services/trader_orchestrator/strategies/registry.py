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


# Unified strategy instances keyed by canonical slug.
_REFERENCE_STRATEGIES: dict[str, BaseStrategy] = {
    "btc_eth_highfreq": BtcEthHighFreqStrategy(),
    "crypto_spike_reversion": CryptoSpikeReversionStrategy(),
    "basic": BasicArbStrategy(),
    "negrisk": NegRiskStrategy(),
    "flash_crash_reversion": FlashCrashReversionStrategy(),
    "tail_end_carry": TailEndCarryStrategy(),
    "news_edge": NewsEdgeStrategy(),
    "traders_confluence": TradersConfluenceStrategy(),
    "weather_edge": WeatherEdgeStrategy(),
    "weather_distribution": WeatherDistributionStrategy(),
}

# Old execution-only slugs → unified canonical slug.
_STRATEGY_ALIASES: dict[str, str] = {
    "strategy.default": "btc_eth_highfreq",
    "default": "btc_eth_highfreq",
    "crypto_5m": "btc_eth_highfreq",
    "crypto_15m": "btc_eth_highfreq",
    "crypto_1h": "btc_eth_highfreq",
    "crypto_4h": "btc_eth_highfreq",
    "opportunity_general": "basic",
    "opportunity_weather": "basic",
    "opportunity_structural": "negrisk",
    "opportunity_flash_reversion": "flash_crash_reversion",
    "opportunity_tail_carry": "tail_end_carry",
    "news_reaction": "news_edge",
    "traders_flow": "traders_confluence",
    "weather_consensus": "weather_edge",
    "weather_alerts": "weather_distribution",
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
