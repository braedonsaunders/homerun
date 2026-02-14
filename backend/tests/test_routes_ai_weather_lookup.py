import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api import routes_ai
from models.opportunity import ArbitrageOpportunity, StrategyType


def _weather_opp(opportunity_id: str) -> ArbitrageOpportunity:
    opp = ArbitrageOpportunity(
        strategy=StrategyType.WEATHER_EDGE,
        title="Weather Edge Test",
        description="test",
        total_cost=0.2,
        expected_payout=0.4,
        gross_profit=0.2,
        fee=0.008,
        net_profit=0.192,
        roi_percent=96.0,
        risk_score=0.25,
        risk_factors=[],
        markets=[{"id": "mkt-weather-1", "question": "Weather Q"}],
        min_liquidity=1000.0,
        max_position_size=10.0,
        positions_to_take=[{"action": "BUY", "outcome": "YES", "market_id": "mkt-weather-1", "price": 0.2}],
    )
    opp.id = opportunity_id
    return opp


@pytest.mark.asyncio
async def test_find_opportunity_by_id_falls_back_to_weather_snapshot(monkeypatch):
    target = _weather_opp("weather_test_123")

    async def _scanner_none(_session, _filter):
        return []
    weather_rows = AsyncMock(return_value=[target])

    monkeypatch.setattr(routes_ai.shared_state, "get_opportunities_from_db", _scanner_none)
    monkeypatch.setattr(
        routes_ai.weather_shared_state,
        "get_weather_opportunities_from_db",
        weather_rows,
    )

    found, source = await routes_ai._find_opportunity_by_id(session=None, opportunity_id=target.id)
    assert found is not None
    assert found.id == target.id
    assert source == "weather"
    assert weather_rows.await_args is not None
    assert weather_rows.await_args.kwargs.get("include_report_only") is True


@pytest.mark.asyncio
async def test_find_opportunity_by_id_matches_report_only_weather_finding(monkeypatch):
    target = _weather_opp("weather_report_only_123")
    target.description = "REPORT ONLY | BUY YES @ $0.20"
    target.max_position_size = 0.0

    async def _scanner_none(_session, _filter):
        return []

    weather_rows = AsyncMock(return_value=[target])

    monkeypatch.setattr(routes_ai.shared_state, "get_opportunities_from_db", _scanner_none)
    monkeypatch.setattr(
        routes_ai.weather_shared_state,
        "get_weather_opportunities_from_db",
        weather_rows,
    )

    found, source = await routes_ai._find_opportunity_by_id(session=None, opportunity_id=target.id)
    assert found is not None
    assert found.id == target.id
    assert source == "weather"
