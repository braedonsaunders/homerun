from __future__ import annotations

from .base import BaseTraderStrategy, TraderStrategy
from .crypto_15m import Crypto15mStrategy
from .news_reaction import NewsReactionStrategy
from .omni_aggressive import OmniAggressiveStrategy
from .opportunity_weather import OpportunityWeatherStrategy


_STRATEGIES: dict[str, TraderStrategy] = {
    Crypto15mStrategy.key: Crypto15mStrategy(),
    NewsReactionStrategy.key: NewsReactionStrategy(),
    OpportunityWeatherStrategy.key: OpportunityWeatherStrategy(),
    OmniAggressiveStrategy.key: OmniAggressiveStrategy(),
}

_STRATEGY_ALIASES: dict[str, str] = {
    # Backward compatibility for older UI defaults.
    "strategy.default": Crypto15mStrategy.key,
    "default": Crypto15mStrategy.key,
}


def list_strategy_keys() -> list[str]:
    return sorted(_STRATEGIES.keys())


def get_strategy(strategy_key: str) -> TraderStrategy:
    key = str(strategy_key or "").strip().lower()
    key = _STRATEGY_ALIASES.get(key, key)
    return _STRATEGIES.get(key, BaseTraderStrategy())
