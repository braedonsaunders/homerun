import sys
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone
from typing import Optional

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.opportunity import ArbitrageOpportunity
from services.weather.workflow_orchestrator import WeatherWorkflowOrchestrator


def test_weather_orchestrator_rejects_non_tradable_market_metadata():
    orchestrator = WeatherWorkflowOrchestrator()
    market = SimpleNamespace(
        active=True,
        closed=False,
        archived=False,
        accepting_orders=False,
        enable_order_book=False,
        resolved=False,
        end_date=None,
        winner=None,
        winning_outcome=None,
        status="in review",
    )

    assert orchestrator._is_market_candidate_tradable(market) is False


def test_weather_orchestrator_accepts_open_market_metadata():
    orchestrator = WeatherWorkflowOrchestrator()
    market = SimpleNamespace(
        active=True,
        closed=False,
        archived=False,
        accepting_orders=True,
        enable_order_book=True,
        resolved=False,
        end_date=None,
        winner=None,
        winning_outcome=None,
        status="active",
    )

    assert orchestrator._is_market_candidate_tradable(market) is True


def _weather_opp(
    *,
    description: str,
    max_position_size: float,
    risk_factors: Optional[list[str]] = None,
) -> ArbitrageOpportunity:
    return ArbitrageOpportunity(
        strategy="weather_edge",
        title="Weather edge",
        description=description,
        total_cost=0.22,
        expected_payout=0.55,
        gross_profit=0.33,
        fee=0.01,
        net_profit=0.32,
        roi_percent=145.45,
        risk_score=0.5,
        risk_factors=risk_factors or [],
        markets=[{"id": "m1", "question": "Weather market"}],
        min_liquidity=1000.0,
        max_position_size=max_position_size,
        detected_at=datetime.now(timezone.utc),
        positions_to_take=[{"action": "BUY", "outcome": "YES", "price": 0.22}],
        category="weather",
    )


def test_weather_orchestrator_flags_report_only_as_non_executable():
    orchestrator = WeatherWorkflowOrchestrator()
    finding = _weather_opp(
        description="REPORT ONLY | BUY YES @ $0.22",
        max_position_size=0.0,
    )
    assert orchestrator._is_executable_opportunity(finding) is False


def test_weather_orchestrator_keeps_live_opportunity_executable():
    orchestrator = WeatherWorkflowOrchestrator()
    live = _weather_opp(
        description="BUY YES @ $0.22 | Edge 8.10%",
        max_position_size=25.0,
    )
    assert orchestrator._is_executable_opportunity(live) is True


def test_weather_orchestrator_summarizes_report_only_reasons():
    orchestrator = WeatherWorkflowOrchestrator()
    finding_a = _weather_opp(
        description="REPORT ONLY | A",
        max_position_size=0.0,
        risk_factors=[
            "Report only: does not meet trade thresholds",
            "confidence 0.40 < min 0.60",
            "agreement 0.61 < min 0.75",
        ],
    )
    finding_b = _weather_opp(
        description="REPORT ONLY | B",
        max_position_size=0.0,
        risk_factors=[
            "Report only: does not meet trade thresholds",
            "confidence 0.40 < min 0.60",
        ],
    )

    summary = orchestrator._summarize_report_only_findings([finding_a, finding_b], top_n=5)
    assert summary[0] == {"reason": "confidence 0.40 < min 0.60", "count": 2}
