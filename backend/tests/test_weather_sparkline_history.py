import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.opportunity import ArbitrageOpportunity
from services.weather.workflow_orchestrator import WeatherWorkflowOrchestrator


def _build_opportunity(
    *,
    market_id: str = "0xmarket",
    report_only: bool = False,
) -> ArbitrageOpportunity:
    return ArbitrageOpportunity(
        strategy="weather_edge",
        title="Weather Edge",
        description="REPORT ONLY | Test" if report_only else "Test",
        total_cost=0.2,
        expected_payout=0.6,
        gross_profit=0.4,
        fee=0.01,
        net_profit=0.39,
        roi_percent=195.0,
        risk_score=0.2,
        markets=[
            {
                "id": market_id,
                "platform": "polymarket",
                "clob_token_ids": ["123456789012345678901", "123456789012345678902"],
                "yes_price": 0.2,
                "no_price": 0.8,
            }
        ],
        min_liquidity=1000.0,
        max_position_size=0.0 if report_only else 10.0,
        positions_to_take=[],
    )


@pytest.mark.asyncio
async def test_weather_history_uses_scanner_shared_pipeline(monkeypatch):
    orchestrator = WeatherWorkflowOrchestrator()
    opp = _build_opportunity()

    attach_mock = AsyncMock(return_value=1)
    monkeypatch.setattr(
        "services.scanner.scanner.attach_price_history_to_opportunities",
        attach_mock,
    )

    await orchestrator._attach_market_price_history([opp])

    assert attach_mock.await_count == 1
    args = attach_mock.await_args
    assert args is not None
    assert len(args.args) >= 1
    assert args.args[0] == [opp]
    assert "now" in args.kwargs
    assert args.kwargs.get("timeout_seconds") == 12.0


@pytest.mark.asyncio
async def test_weather_history_tolerates_scanner_attach_failure(monkeypatch):
    orchestrator = WeatherWorkflowOrchestrator()
    opp = _build_opportunity()

    attach_mock = AsyncMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(
        "services.scanner.scanner.attach_price_history_to_opportunities",
        attach_mock,
    )

    # Should not raise: weather scan should continue even if sparkline attach fails.
    await orchestrator._attach_market_price_history([opp])
    assert attach_mock.await_count == 1


@pytest.mark.asyncio
async def test_run_cycle_attaches_history_for_visible_findings_and_opportunities(monkeypatch):
    orchestrator = WeatherWorkflowOrchestrator()
    executable = _build_opportunity(market_id="0xexec")
    finding = _build_opportunity(market_id="0xfinding", report_only=True)

    monkeypatch.setattr(
        "services.weather.workflow_orchestrator.shared_state.get_weather_settings",
        AsyncMock(
            return_value={
                "enabled": True,
                "max_markets_per_scan": 25,
                "scan_interval_seconds": 600,
                "min_liquidity": 10.0,
                "evaluation_concurrency": 1,
            }
        ),
    )
    monkeypatch.setattr(
        "services.weather.workflow_orchestrator.shared_state.read_weather_snapshot",
        AsyncMock(return_value=([], {})),
    )
    monkeypatch.setattr(
        "services.weather.workflow_orchestrator.shared_state.write_weather_snapshot",
        AsyncMock(),
    )
    monkeypatch.setattr(
        "services.weather.workflow_orchestrator.shared_state.upsert_weather_intent",
        AsyncMock(),
    )

    monkeypatch.setattr(
        orchestrator,
        "_fetch_weather_markets",
        AsyncMock(return_value=["market-exec", "market-finding"]),
    )
    monkeypatch.setattr(
        orchestrator,
        "_evaluate_market",
        AsyncMock(
            side_effect=[
                {
                    "contracts_parsed": 1,
                    "signals_generated": 1,
                    "opportunity": executable,
                    "intent": None,
                },
                {
                    "contracts_parsed": 1,
                    "signals_generated": 1,
                    "opportunity": finding,
                    "intent": None,
                },
            ]
        ),
    )
    attach_mock = AsyncMock(return_value=2)
    monkeypatch.setattr(orchestrator, "_attach_market_price_history", attach_mock)

    class _Session:
        def __init__(self):
            self.commit = AsyncMock()

    result = await orchestrator.run_cycle(_Session())

    assert result["opportunities"] == 1
    assert result["findings"] == 1
    assert attach_mock.await_count == 1
    assert attach_mock.await_args is not None
    assert attach_mock.await_args.args[0] == [executable, finding]
