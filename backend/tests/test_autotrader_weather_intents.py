import sys
from pathlib import Path
from types import SimpleNamespace

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.auto_trader import AutoTrader
from models.opportunity import StrategyType


def _weather_intent(**overrides):
    base = dict(
        id="intent_1",
        market_id="cond_123",
        market_question="Will it rain in NYC tomorrow?",
        direction="buy_yes",
        entry_price=0.22,
        model_probability=0.61,
        edge_percent=39.0,
        confidence=0.78,
        model_agreement=0.86,
        suggested_size_usd=15.0,
        metadata_json={
            "market": {
                "id": "mkt_1",
                "slug": "will-it-rain-nyc",
                "event_slug": "weather-nyc",
                "liquidity": 12000.0,
                "clob_token_ids": ["tok_yes", "tok_no"],
            },
            "weather": {
                "location": "New York, NY",
                "target_time": "2026-01-01T12:00:00Z",
                "agreement": 0.86,
            },
        },
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_weather_intent_to_opportunity_maps_strategy_and_token():
    trader = AutoTrader()
    intent = _weather_intent(direction="buy_no")
    opp = trader._weather_intent_to_opportunity(intent)
    assert opp is not None
    assert opp.strategy == StrategyType.WEATHER_EDGE
    assert opp.category == "weather"
    assert opp.positions_to_take[0]["outcome"] == "NO"
    assert opp.positions_to_take[0]["token_id"] == "tok_no"


def test_weather_intent_to_opportunity_rejects_invalid_price():
    trader = AutoTrader()
    bad_intent = _weather_intent(entry_price=1.0)
    assert trader._weather_intent_to_opportunity(bad_intent) is None


def test_get_config_exposes_weather_workflow_fields():
    trader = AutoTrader()
    cfg = trader.get_config()
    assert "weather_workflow_enabled" in cfg
    assert "weather_workflow_min_edge" in cfg
    assert "weather_workflow_max_age_minutes" in cfg
