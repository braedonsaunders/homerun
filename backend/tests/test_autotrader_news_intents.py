import sys
from pathlib import Path
from types import SimpleNamespace

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.opportunity import StrategyType
from services.auto_trader import AutoTrader


def _news_intent(**overrides):
    base = dict(
        id="intent_news_1",
        market_id="cond_123",
        market_question="Will candidate X win the election?",
        direction="buy_yes",
        entry_price=0.42,
        model_probability=0.64,
        edge_percent=22.0,
        confidence=0.76,
        suggested_size_usd=25.0,
        metadata_json={
            "market": {
                "id": "mkt_1",
                "slug": "candidate-x-election",
                "event_slug": "election-2028",
                "liquidity": 15000.0,
                "token_ids": ["tok_yes", "tok_no"],
            }
        },
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_news_intent_to_opportunity_maps_strategy_and_token():
    trader = AutoTrader()
    intent = _news_intent(direction="buy_no")
    opp = trader._news_intent_to_opportunity(intent)
    assert opp is not None
    assert opp.strategy == StrategyType.NEWS_EDGE
    assert opp.category == "news"
    assert opp.positions_to_take[0]["outcome"] == "NO"
    assert opp.positions_to_take[0]["token_id"] == "tok_no"
    assert opp.event_slug == "election-2028"


def test_news_intent_to_opportunity_rejects_invalid_price():
    trader = AutoTrader()
    bad_intent = _news_intent(entry_price=1.0)
    assert trader._news_intent_to_opportunity(bad_intent) is None


def test_get_config_exposes_news_workflow_fields():
    trader = AutoTrader()
    cfg = trader.get_config()
    assert "news_workflow_enabled" in cfg
    assert "news_workflow_min_edge" in cfg
    assert "news_workflow_max_age_minutes" in cfg
