import sys
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.opportunity import ArbitrageOpportunity
from services.ai.opportunity_judge import OpportunityJudge


def _build_weather_opportunity(*, max_position_size: float = 0.0) -> ArbitrageOpportunity:
    return ArbitrageOpportunity(
        strategy="weather_edge",
        title="Will the highest temperature in Miami be 75°F or below on February 14?",
        description=(
            "REPORT ONLY | BUY YES @ $0.26 | Edge 30.90% | "
            "entry_price 0.260 > max 0.250; insufficient source diversity (<2 forecast sources)"
        ),
        total_cost=0.26,
        expected_payout=0.569,
        gross_profit=0.309,
        fee=0.0114,
        net_profit=0.2976,
        roi_percent=114.47,
        risk_score=0.6,
        risk_factors=[
            "Report only: does not meet trade thresholds",
            "entry_price 0.260 > max 0.250",
            "insufficient source diversity (<2 forecast sources)",
        ],
        markets=[
            {
                "id": "0xabc",
                "question": "Will the highest temperature in Miami be 75°F or below on February 14?",
                "yes_price": 0.26,
                "no_price": 0.74,
                "liquidity": 809.0,
                "weather": {
                    "location": "Miami",
                    "target_time": "2026-02-14T00:00:00Z",
                    "source_count": 1,
                    "consensus_probability": 0.569,
                    "market_probability": 0.26,
                    "consensus_temp_f": 74.0,
                    "market_implied_temp_f": 78.8,
                    "agreement": 1.0,
                    "forecast_sources": [
                        {
                            "source_id": "nws:hourly",
                            "provider": "nws",
                            "model": "hourly",
                            "value_f": 74.0,
                            "probability": 0.569,
                            "weight": 1.0,
                        }
                    ],
                },
            }
        ],
        min_liquidity=809.0,
        max_position_size=max_position_size,
        detected_at=datetime.now(timezone.utc),
        resolution_date=datetime(2026, 2, 14, tzinfo=timezone.utc),
        positions_to_take=[{"action": "BUY", "outcome": "YES", "price": 0.26}],
        category="weather",
    )


@pytest.mark.asyncio
async def test_report_only_opportunity_uses_heuristic_and_skips_llm(monkeypatch):
    opp = _build_weather_opportunity(max_position_size=0.0)
    judge = OpportunityJudge()

    def _fail_llm_manager():
        raise AssertionError("LLM manager should not be called for report-only opportunities")

    monkeypatch.setattr("services.ai.opportunity_judge.get_llm_manager", _fail_llm_manager)
    store_mock = AsyncMock()
    monkeypatch.setattr(judge, "_store_judgment", store_mock)

    result = await judge.judge_opportunity(opp)

    assert result["recommendation"] == "strong_skip"
    assert result["execution_feasibility"] == 0.0
    assert result["model_used"] == "heuristic_report_only"
    assert "report-only" in result["reasoning"].lower()
    store_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_force_llm_overrides_report_only_heuristic(monkeypatch):
    opp = _build_weather_opportunity(max_position_size=0.0)
    judge = OpportunityJudge()

    class _Manager:
        _default_model = "test-model"

        async def structured_output(self, **kwargs):
            return {
                "overall_score": 0.51,
                "profit_viability": 0.62,
                "resolution_safety": 0.58,
                "execution_feasibility": 0.40,
                "market_efficiency": 0.45,
                "recommendation": "review",
                "reasoning": "LLM reviewed weather signal with mixed confidence.",
                "risk_factors": ["single source forecast"],
            }

    monkeypatch.setattr("services.ai.opportunity_judge.get_llm_manager", lambda: _Manager())
    store_mock = AsyncMock()
    monkeypatch.setattr(judge, "_store_judgment", store_mock)

    result = await judge.judge_opportunity(opp, force_llm=True)

    assert result["model_used"] == "test-model"
    assert result["recommendation"] == "review"
    assert result["execution_feasibility"] == 0.4
    store_mock.assert_awaited_once()


def test_weather_prompt_includes_compact_weather_context():
    opp = _build_weather_opportunity(max_position_size=0.0)
    judge = OpportunityJudge()

    prompt = judge._build_judgment_prompt(opp)

    assert "### Weather Context" in prompt
    assert "Report-only mode: yes" in prompt
    assert "Forecast source snapshots" in prompt
    assert "nws:hourly" in prompt
