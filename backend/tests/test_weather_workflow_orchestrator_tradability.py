import sys
from pathlib import Path
from types import SimpleNamespace

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

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
