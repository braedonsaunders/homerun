"""Shared fixtures for Polymarket arbitrage tests."""

import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import pytest


_TEST_WALLET_PREFIXES = ("0x1234567890abcdef", "0xdeadbeef", "0x0000000000000000")


@pytest.fixture(autouse=True)
def _seed_wallet_state_cache_for_tests(monkeypatch):
    """Pre-seed the WalletStateCache so the orchestrator's hot-path
    freshness gate doesn't refuse to trade in tests.

    Production behavior: the orchestrator's ``_run_trader_once`` checks
    ``WalletStateCache.is_fresh()`` and skips the cycle when the cache
    has no REST seed yet OR the user-channel WS isn't connected.  In
    tests we don't run the reconciliation worker or the WS feed, so
    without this fixture every orchestrator test would skip cleanly
    with ``Trader cycle skipped: wallet state stale``.

    The fixture marks the cache as freshly seeded with no positions
    (idle wallet → ``is_fresh() == True``).  Tests that exercise
    position-related logic can override by adding rows to the cache.
    """
    try:
        from services.wallet_state_cache import (
            get_wallet_state_cache,
            reset_wallet_state_cache,
        )
    except Exception:
        return
    reset_wallet_state_cache()
    cache = get_wallet_state_cache()
    cache.seed_from_rest(
        wallet_address="0x0000000000000000000000000000000000000000",
        positions=[],
        closed_positions=[],
        succeeded=True,
    )
    yield
    reset_wallet_state_cache()


@pytest.fixture(autouse=True)
def _block_test_wallet_db_writes(monkeypatch):
    # Defense-in-depth: tests that exercise LiveExecutionService with a
    # placeholder wallet (e.g. ``0x1234...5678``) used to leak failed-order
    # rows into the real ``live_trading_orders`` table because the suite has
    # no separate test database.  This autouse fixture intercepts any
    # ``_persist_orders`` / ``_persist_runtime_state`` / ``_persist_positions``
    # call coming from a known test-wallet address and turns it into a no-op,
    # without affecting tests that exercise the persistence path explicitly
    # (those should set a non-test wallet or override this fixture themselves).
    if os.environ.get("HOMERUN_ALLOW_TEST_WALLET_DB_WRITES") == "1":
        return

    try:
        from services import live_execution_service as _live
    except Exception:
        return

    original_persist_orders = _live.LiveExecutionService._persist_orders
    original_persist_runtime = _live.LiveExecutionService._persist_runtime_state
    original_persist_positions = _live.LiveExecutionService._persist_positions

    async def _maybe_skip_orders(self, orders):
        wallet = (str(getattr(self, "_wallet_address", "") or "")).lower()
        if any(wallet.startswith(prefix) for prefix in _TEST_WALLET_PREFIXES):
            return
        return await original_persist_orders(self, orders)

    async def _maybe_skip_runtime(self):
        wallet = (str(getattr(self, "_wallet_address", "") or "")).lower()
        if any(wallet.startswith(prefix) for prefix in _TEST_WALLET_PREFIXES):
            return
        return await original_persist_runtime(self)

    async def _maybe_skip_positions(self):
        wallet = (str(getattr(self, "_wallet_address", "") or "")).lower()
        if any(wallet.startswith(prefix) for prefix in _TEST_WALLET_PREFIXES):
            return
        return await original_persist_positions(self)

    monkeypatch.setattr(
        _live.LiveExecutionService, "_persist_orders", _maybe_skip_orders
    )
    monkeypatch.setattr(
        _live.LiveExecutionService, "_persist_runtime_state", _maybe_skip_runtime
    )
    monkeypatch.setattr(
        _live.LiveExecutionService, "_persist_positions", _maybe_skip_positions
    )

from datetime import timedelta
from utils.utcnow import utcnow
from unittest.mock import AsyncMock, MagicMock
import json

from models.market import Market, Event, Token
from models.opportunity import (
    Opportunity,
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
    """A fully-populated Opportunity."""
    return Opportunity(
        strategy="basic",
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
    return Opportunity(
        strategy="negrisk",
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
    return Opportunity(
        strategy="miracle",
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
    return Opportunity(
        strategy="basic",
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
    return Opportunity(
        strategy="basic",
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
    strategy.strategy_type = "basic"
    strategy.detect = MagicMock(return_value=[])
    return strategy
