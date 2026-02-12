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


def list_strategy_keys() -> list[str]:
    return sorted(_STRATEGIES.keys())


def get_strategy(strategy_key: str) -> TraderStrategy:
    key = str(strategy_key or "").strip().lower()
    return _STRATEGIES.get(key, BaseTraderStrategy())
