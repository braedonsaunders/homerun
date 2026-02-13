"""Shared fixtures for Polymarket arbitrage tests."""

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import pytest
from datetime import timedelta
from utils.utcnow import utcnow
from unittest.mock import AsyncMock, MagicMock
import json

from models.market import Market, Event, Token
from models.opportunity import (
    ArbitrageOpportunity,
    StrategyType,
    MispricingType,
)


# ---------------------------------------------------------------------------
# Raw API response fixtures (mimicking Gamma API payloads)
# ---------------------------------------------------------------------------


@pytest.fixture
def raw_market_response():
    """A realistic Gamma-API /markets response dict."""
    return {
        "id": "123456",
        "condition_id": "0xabc123",
        "conditionId": "0xabc123",
        "question": "Will BTC exceed $100k by end of 2025?",
        "slug": "will-btc-exceed-100k-2025",
        "clobTokenIds": json.dumps(["token_yes_1", "token_no_1"]),
        "outcomePrices": json.dumps(["0.65", "0.35"]),
        "active": True,
        "closed": False,
        "negRisk": False,
        "volume": "12345.67",
        "liquidity": "5000.00",
        "endDate": None,
    }


@pytest.fixture
def raw_market_response_neg_risk():
    """Market with neg-risk flag set."""
    return {
        "id": "789",
        "condition_id": "0xdef789",
        "question": "US Election outcome?",
        "slug": "us-election-outcome",
        "clobTokenIds": json.dumps(["tok_y2", "tok_n2"]),
        "outcomePrices": json.dumps(["0.40", "0.55"]),
        "active": True,
        "closed": False,
        "negRisk": True,
        "volume": "99999",
        "liquidity": "20000",
        "endDate": None,
    }


@pytest.fixture
def raw_event_response(raw_market_response, raw_market_response_neg_risk):
    """A realistic Gamma-API /events response dict containing nested markets."""
    return {
        "id": "evt_001",
        "slug": "btc-price-events",
        "title": "Bitcoin Price Predictions",
        "description": "All markets about BTC price targets.",
        "category": "Crypto",
        "negRisk": False,
        "active": True,
        "closed": False,
        "markets": [raw_market_response, raw_market_response_neg_risk],
    }


# ---------------------------------------------------------------------------
# Parsed model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_market(raw_market_response):
    return Market.from_gamma_response(raw_market_response)


@pytest.fixture
def sample_market_neg_risk(raw_market_response_neg_risk):
    return Market.from_gamma_response(raw_market_response_neg_risk)


@pytest.fixture
def sample_event(raw_event_response):
    return Event.from_gamma_response(raw_event_response)


@pytest.fixture
def sample_token():
    return Token(token_id="tok_123", outcome="Yes", price=0.65)


# ---------------------------------------------------------------------------
# Opportunity fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_opportunity():
    """A fully-populated ArbitrageOpportunity."""
    return ArbitrageOpportunity(
        strategy=StrategyType.BASIC,
        title="Basic Arb: Will BTC exceed $100k...",
        description="Buy YES ($0.48) + NO ($0.48) = $0.96 for guaranteed $1 payout",
        total_cost=0.96,
        expected_payout=1.0,
        gross_profit=0.04,
        fee=0.02,
        net_profit=0.02,
        roi_percent=2.08,
        risk_score=0.3,
        risk_factors=["Moderate liquidity ($5000)"],
        markets=[
            {
                "id": "123456",
                "question": "Will BTC exceed $100k?",
                "yes_price": 0.48,
                "no_price": 0.48,
                "liquidity": 5000,
            }
        ],
        event_id="evt_001",
        event_title="Bitcoin Price Predictions",
        category="Crypto",
        min_liquidity=5000.0,
        max_position_size=500.0,
        mispricing_type=MispricingType.WITHIN_MARKET,
    )


@pytest.fixture
def sample_opportunity_high_roi():
    """An opportunity with high ROI for sorting tests."""
    return ArbitrageOpportunity(
        strategy=StrategyType.NEGRISK,
        title="NegRisk Arb: Election market",
        description="NegRisk opportunity",
        total_cost=0.85,
        expected_payout=1.0,
        gross_profit=0.15,
        fee=0.02,
        net_profit=0.13,
        roi_percent=15.29,
        risk_score=0.2,
        markets=[
            {
                "id": "789",
                "question": "Election?",
                "yes_price": 0.40,
                "no_price": 0.45,
                "liquidity": 20000,
            }
        ],
        event_id="evt_002",
        event_title="US Election",
        category="Politics",
        min_liquidity=20000.0,
        max_position_size=2000.0,
        mispricing_type=MispricingType.WITHIN_MARKET,
    )


@pytest.fixture
def sample_opportunity_low_roi():
    """An opportunity with low ROI for sorting tests."""
    return ArbitrageOpportunity(
        strategy=StrategyType.MIRACLE,
        title="Miracle: Aliens land",
        description="Bet against impossible event",
        total_cost=0.97,
        expected_payout=1.0,
        gross_profit=0.03,
        fee=0.02,
        net_profit=0.01,
        roi_percent=1.03,
        risk_score=0.8,
        markets=[
            {
                "id": "999",
                "question": "Will aliens land?",
                "yes_price": 0.03,
                "no_price": 0.94,
                "liquidity": 500,
            }
        ],
        event_id="evt_003",
        event_title="Aliens",
        category="Science",
        min_liquidity=500.0,
        max_position_size=50.0,
        mispricing_type=MispricingType.WITHIN_MARKET,
    )


@pytest.fixture
def expired_opportunity():
    """An opportunity whose resolution_date is in the past."""
    return ArbitrageOpportunity(
        strategy=StrategyType.BASIC,
        title="Expired opp",
        description="Already resolved",
        total_cost=0.95,
        expected_payout=1.0,
        gross_profit=0.05,
        fee=0.02,
        net_profit=0.03,
        roi_percent=3.16,
        risk_score=0.5,
        markets=[{"id": "old1"}],
        resolution_date=utcnow() - timedelta(days=1),
    )


@pytest.fixture
def old_opportunity():
    """An opportunity detected over 2 hours ago."""
    return ArbitrageOpportunity(
        strategy=StrategyType.BASIC,
        title="Old opp",
        description="Old detection",
        total_cost=0.95,
        expected_payout=1.0,
        gross_profit=0.05,
        fee=0.02,
        net_profit=0.03,
        roi_percent=3.16,
        risk_score=0.5,
        markets=[{"id": "old2"}],
        detected_at=utcnow() - timedelta(hours=2),
    )


# ---------------------------------------------------------------------------
# Scanner helper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_polymarket_client():
    """A fully-mocked PolymarketClient."""
    client = AsyncMock()
    client.get_all_events = AsyncMock(return_value=[])
    client.get_all_markets = AsyncMock(return_value=[])
    client.get_recent_markets = AsyncMock(return_value=[])
    client.get_cross_platform_events = AsyncMock(return_value=[])
    client.get_cross_platform_markets = AsyncMock(return_value=[])
    client.get_prices_batch = AsyncMock(return_value={})
    return client


@pytest.fixture
def mock_strategy():
    """A mock strategy that returns configurable opportunities."""
    strategy = MagicMock()
    strategy.name = "MockStrategy"
    strategy.strategy_type = StrategyType.BASIC
    strategy.detect = MagicMock(return_value=[])
    return strategy
