from __future__ import annotations

from .base import BaseTraderStrategy, TraderStrategy
from services.trader_orchestrator.strategy_db_loader import strategy_db_loader
from .crypto_15m import (
    Crypto15mStrategy,
    Crypto1hStrategy,
    Crypto4hStrategy,
    Crypto5mStrategy,
)
from .news_reaction import NewsReactionStrategy
from .opportunity_weather import (
    OpportunityGeneralStrategy,
    OpportunityStructuralStrategy,
)
from .traders_flow import TradersFlowStrategy
from .weather_models import (
    WeatherAlertsStrategy,
    WeatherConsensusStrategy,
)


# Static strategy map is kept as a seed/reference fallback only.
_REFERENCE_STRATEGIES: dict[str, TraderStrategy] = {
    Crypto5mStrategy.key: Crypto5mStrategy(),
    Crypto15mStrategy.key: Crypto15mStrategy(),
    Crypto1hStrategy.key: Crypto1hStrategy(),
    Crypto4hStrategy.key: Crypto4hStrategy(),
    NewsReactionStrategy.key: NewsReactionStrategy(),
    OpportunityGeneralStrategy.key: OpportunityGeneralStrategy(),
    OpportunityStructuralStrategy.key: OpportunityStructuralStrategy(),
    WeatherConsensusStrategy.key: WeatherConsensusStrategy(),
    WeatherAlertsStrategy.key: WeatherAlertsStrategy(),
    TradersFlowStrategy.key: TradersFlowStrategy(),
}

_STRATEGY_ALIASES: dict[str, str] = {
    # Backward compatibility for older UI defaults.
    "strategy.default": Crypto15mStrategy.key,
    "default": Crypto15mStrategy.key,
    "opportunity_weather": OpportunityGeneralStrategy.key,
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


def get_strategy(strategy_key: str, *, use_static_fallback: bool = True) -> TraderStrategy:
    key = _resolve_strategy_key(strategy_key)
    loaded = strategy_db_loader.get_strategy(key)
    if loaded is not None:
        return loaded
    if use_static_fallback:
        return _REFERENCE_STRATEGIES.get(key, BaseTraderStrategy())
    return BaseTraderStrategy()
