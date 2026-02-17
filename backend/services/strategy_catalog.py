"""Unified Strategy Catalog

Single entry point for all strategy seeds. Each strategy is a complete unit
with DETECT + EVALUATE + EXIT lifecycle methods. Seeds are organized into
two sub-catalogs by origin:

  - opportunity_strategy_catalog.py   — scanner-discovered detection strategies
  - trader_orchestrator/strategy_catalog.py — execution strategies for the autotrader

When adding a new strategy:
1. Write your strategy class (extending BaseStrategy) in services/strategies/
2. Add a seed entry to the appropriate sub-catalog
3. The seed routine will create/update the DB row on startup
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from services.opportunity_strategy_catalog import (
    SYSTEM_OPPORTUNITY_STRATEGY_SEEDS,
    SystemOpportunityStrategySeed,
    ensure_system_opportunity_strategies_seeded,
)
from services.trader_orchestrator.strategy_catalog import (
    SYSTEM_STRATEGY_SEEDS,
    SystemStrategySeed,
    ensure_system_trader_strategies_seeded,
)

__all__ = [
    "SYSTEM_OPPORTUNITY_STRATEGY_SEEDS",
    "SystemOpportunityStrategySeed",
    "ensure_system_opportunity_strategies_seeded",
    "SYSTEM_STRATEGY_SEEDS",
    "SystemStrategySeed",
    "ensure_system_trader_strategies_seeded",
    "ensure_all_strategies_seeded",
    "ALL_STRATEGY_SEEDS",
]


async def ensure_all_strategies_seeded(session: AsyncSession) -> dict:
    """Seed both detection and execution strategies in one call."""
    detection_result = await ensure_system_opportunity_strategies_seeded(session)
    execution_result = await ensure_system_trader_strategies_seeded(session)
    return {
        "detection": detection_result,
        "execution": execution_result,
    }


ALL_STRATEGY_SEEDS = {
    "detection": SYSTEM_OPPORTUNITY_STRATEGY_SEEDS,
    "execution": SYSTEM_STRATEGY_SEEDS,
}
