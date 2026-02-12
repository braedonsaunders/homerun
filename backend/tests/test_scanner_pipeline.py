"""Tests for ArbitrageScanner: initialisation, scan pipeline, filtering, lifecycle."""

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import pytest
import asyncio
from datetime import datetime, timedelta
from utils.utcnow import utcnow
from unittest.mock import AsyncMock, MagicMock, patch

from models.market import Market
from models.opportunity import (
    ArbitrageOpportunity,
    StrategyType,
    MispricingType,
    OpportunityFilter,
)


# ---------------------------------------------------------------------------
# Helpers to build a scanner with mocked externals
# ---------------------------------------------------------------------------


def _build_scanner(
    mock_client=None,
    strategies=None,
):
    """
    Import ArbitrageScanner inside a mock context so the singleton
    polymarket_client and database are never touched.
    """
    data_provider = mock_client or AsyncMock()
    with patch("services.scanner.AsyncSessionLocal", MagicMock()):
        from services.scanner import ArbitrageScanner

        scanner = ArbitrageScanner(data_provider=data_provider)
        if strategies is not None:
            scanner.strategies = strategies
        return scanner


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestScannerInit:
    """Tests for ArbitrageScanner.__init__."""

    def test_loads_all_enabled_sync_strategies(self):
        scanner = _build_scanner()
        # NewsEdge runs as a separate async/manual path and is not in
        # scanner.strategies, so we expect all StrategyType members except NEWS_EDGE.
        expected_count = len([st for st in StrategyType if st != StrategyType.NEWS_EDGE])
        assert len(scanner.strategies) == expected_count

    def test_strategy_names(self):
        scanner = _build_scanner()
        names = [s.name for s in scanner.strategies]
        assert "Basic Arbitrage" in names

    def test_strategy_types_cover_all_enums(self):
        scanner = _build_scanner()
        types = {s.strategy_type for s in scanner.strategies}
        # NewsEdge is handled by a separate async/manual path, not in scanner.strategies.
        expected = {st for st in StrategyType if st != StrategyType.NEWS_EDGE}
        assert types == expected

    def test_initial_state(self):
        scanner = _build_scanner()
        assert scanner._running is False
        assert scanner._enabled is True
        assert scanner._opportunities == []
        assert scanner._last_scan is None
        assert scanner._scan_callbacks == []
        assert scanner._status_callbacks == []


# ---------------------------------------------------------------------------
# scan_once
# ---------------------------------------------------------------------------


class TestScanOnce:
    """Tests for ArbitrageScanner.scan_once."""

    @pytest.mark.asyncio
    async def test_scan_once_calls_client_methods(self, mock_polymarket_client):
        """scan_once fetches events, markets, and prices."""
        scanner = _build_scanner(mock_client=mock_polymarket_client)

        await scanner.scan_once()

        mock_polymarket_client.get_all_events.assert_awaited_once_with(closed=False)
        mock_polymarket_client.get_all_markets.assert_awaited_once_with(active=True)

    @pytest.mark.asyncio
    async def test_scan_once_returns_opportunities_sorted_by_roi(
        self,
        mock_polymarket_client,
        sample_opportunity,
        sample_opportunity_high_roi,
        sample_opportunity_low_roi,
    ):
        """Opportunities are returned sorted by ROI descending."""
        # Strategy 1 returns low + medium ROI opps
        strategy_a = MagicMock()
        strategy_a.name = "StratA"
        strategy_a.strategy_type = StrategyType.BASIC
        strategy_a.detect = MagicMock(
            return_value=[sample_opportunity_low_roi, sample_opportunity]
        )

        # Strategy 2 returns high ROI opp
        strategy_b = MagicMock()
        strategy_b.name = "StratB"
        strategy_b.strategy_type = StrategyType.NEGRISK
        strategy_b.detect = MagicMock(return_value=[sample_opportunity_high_roi])

        scanner = _build_scanner(
            mock_client=mock_polymarket_client,
            strategies=[strategy_a, strategy_b],
        )

        results = await scanner.scan_once()

        assert len(results) == 3
        # Sorted descending by roi_percent
        assert results[0].roi_percent >= results[1].roi_percent
        assert results[1].roi_percent >= results[2].roi_percent

    @pytest.mark.asyncio
    async def test_scan_once_handles_strategy_exception_gracefully(
        self,
        mock_polymarket_client,
        sample_opportunity,
    ):
        """A failing strategy does not prevent others from running."""
        bad_strategy = MagicMock()
        bad_strategy.name = "BadStrategy"
        bad_strategy.strategy_type = StrategyType.CONTRADICTION
        bad_strategy.detect = MagicMock(side_effect=RuntimeError("boom"))

        good_strategy = MagicMock()
        good_strategy.name = "GoodStrategy"
        good_strategy.strategy_type = StrategyType.BASIC
        good_strategy.detect = MagicMock(return_value=[sample_opportunity])

        scanner = _build_scanner(
            mock_client=mock_polymarket_client,
            strategies=[bad_strategy, good_strategy],
        )

        results = await scanner.scan_once()

        # Only the good strategy's opportunity is returned
        assert len(results) == 1
        assert results[0].strategy == StrategyType.BASIC

    @pytest.mark.asyncio
    async def test_scan_once_sets_last_scan(self, mock_polymarket_client):
        scanner = _build_scanner(mock_client=mock_polymarket_client)

        assert scanner._last_scan is None
        await scanner.scan_once()
        assert scanner._last_scan is not None
        assert isinstance(scanner._last_scan, datetime)

    @pytest.mark.asyncio
    async def test_scan_once_fetches_prices_for_tokens(self, mock_polymarket_client):
        """When markets have tokens, get_prices_batch is called."""
        market = Market(
            id="m1",
            condition_id="c1",
            question="Q?",
            slug="q",
            clob_token_ids=["tok_a", "tok_b"],
            outcome_prices=[0.6, 0.4],
        )
        mock_polymarket_client.get_all_markets.return_value = [market]
        mock_polymarket_client.get_all_events.return_value = []
        mock_polymarket_client.get_prices_batch.return_value = {
            "tok_a": {"mid": 0.61},
            "tok_b": {"mid": 0.39},
        }

        scanner = _build_scanner(
            mock_client=mock_polymarket_client,
            strategies=[],  # no strategies, just testing data flow
        )

        await scanner.scan_once()

        mock_polymarket_client.get_prices_batch.assert_awaited_once()
        call_args = mock_polymarket_client.get_prices_batch.call_args[0][0]
        assert "tok_a" in call_args
        assert "tok_b" in call_args


# ---------------------------------------------------------------------------
# Mispricing type classification
# ---------------------------------------------------------------------------


class TestMispricingClassification:
    """Tests for mispricing type assignment during scan_once."""

    @pytest.mark.asyncio
    async def test_mispricing_type_set_for_basic_strategy(self, mock_polymarket_client):
        """Opportunities from basic strategy get WITHIN_MARKET classification."""
        opp = ArbitrageOpportunity(
            strategy=StrategyType.BASIC,
            title="Test",
            description="D",
            total_cost=0.95,
            gross_profit=0.05,
            fee=0.02,
            net_profit=0.03,
            roi_percent=3.16,
            markets=[{"id": "m1"}],
            mispricing_type=None,  # Not set by strategy
        )

        strategy = MagicMock()
        strategy.name = "Basic"
        strategy.strategy_type = StrategyType.BASIC
        strategy.detect = MagicMock(return_value=[opp])

        scanner = _build_scanner(
            mock_client=mock_polymarket_client,
            strategies=[strategy],
        )

        results = await scanner.scan_once()

        assert results[0].mispricing_type == MispricingType.WITHIN_MARKET

    @pytest.mark.asyncio
    async def test_mispricing_type_set_for_combinatorial(self, mock_polymarket_client):
        """Combinatorial strategy maps to CROSS_MARKET."""
        opp = ArbitrageOpportunity(
            strategy=StrategyType.COMBINATORIAL,
            title="Test",
            description="D",
            total_cost=0.9,
            gross_profit=0.1,
            fee=0.02,
            net_profit=0.08,
            roi_percent=8.89,
            markets=[{"id": "m1"}],
            mispricing_type=None,
        )

        strategy = MagicMock()
        strategy.name = "Combinatorial"
        strategy.strategy_type = StrategyType.COMBINATORIAL
        strategy.detect = MagicMock(return_value=[opp])

        scanner = _build_scanner(
            mock_client=mock_polymarket_client,
            strategies=[strategy],
        )

        results = await scanner.scan_once()

        assert results[0].mispricing_type == MispricingType.CROSS_MARKET

    @pytest.mark.asyncio
    async def test_mispricing_type_set_for_settlement_lag(self, mock_polymarket_client):
        """Settlement-lag strategy maps to SETTLEMENT_LAG."""
        opp = ArbitrageOpportunity(
            strategy=StrategyType.SETTLEMENT_LAG,
            title="Test",
            description="D",
            total_cost=0.9,
            gross_profit=0.1,
            fee=0.02,
            net_profit=0.08,
            roi_percent=8.89,
            markets=[{"id": "m1"}],
            mispricing_type=None,
        )

        strategy = MagicMock()
        strategy.name = "SettlementLag"
        strategy.strategy_type = StrategyType.SETTLEMENT_LAG
        strategy.detect = MagicMock(return_value=[opp])

        scanner = _build_scanner(
            mock_client=mock_polymarket_client,
            strategies=[strategy],
        )

        results = await scanner.scan_once()

        assert results[0].mispricing_type == MispricingType.SETTLEMENT_LAG

    @pytest.mark.asyncio
    async def test_mispricing_type_not_overwritten_if_already_set(
        self, mock_polymarket_client
    ):
        """If a strategy already set mispricing_type, the scanner does not overwrite it."""
        opp = ArbitrageOpportunity(
            strategy=StrategyType.BASIC,
            title="Test",
            description="D",
            total_cost=0.95,
            gross_profit=0.05,
            fee=0.02,
            net_profit=0.03,
            roi_percent=3.16,
            markets=[{"id": "m1"}],
            mispricing_type=MispricingType.CROSS_MARKET,  # Pre-set by strategy
        )

        strategy = MagicMock()
        strategy.name = "Basic"
        strategy.strategy_type = StrategyType.BASIC
        strategy.detect = MagicMock(return_value=[opp])

        scanner = _build_scanner(
            mock_client=mock_polymarket_client,
            strategies=[strategy],
        )

        results = await scanner.scan_once()

        # Should preserve the pre-set value, not overwrite to WITHIN_MARKET
        assert results[0].mispricing_type == MispricingType.CROSS_MARKET


# ---------------------------------------------------------------------------
# Shared sparkline history attach
# ---------------------------------------------------------------------------


class TestSharedPriceHistoryAttach:
    @pytest.mark.asyncio
    async def test_remember_tokens_from_opportunities_parses_json_string(self):
        scanner = _build_scanner(strategies=[])
        yes_token = "123456789012345678901"
        no_token = "123456789012345678902"
        opp = ArbitrageOpportunity(
            strategy=StrategyType.WEATHER_EDGE,
            title="Weather",
            description="D",
            total_cost=0.2,
            expected_payout=0.5,
            gross_profit=0.3,
            fee=0.01,
            net_profit=0.29,
            roi_percent=145.0,
            markets=[
                {
                    "id": "m_weather_1",
                    "platform": "polymarket",
                    "clob_token_ids": f"[\"{yes_token}\", \"{no_token}\"]",
                    "yes_price": 0.2,
                    "no_price": 0.8,
                }
            ],
            min_liquidity=1000.0,
            max_position_size=10.0,
            positions_to_take=[],
        )

        scanner._remember_market_tokens_from_opportunities([opp])

        assert scanner._market_token_ids.get("m_weather_1") == (yes_token, no_token)

    @pytest.mark.asyncio
    async def test_attach_price_history_to_opportunities_uses_shared_backfill(self):
        scanner = _build_scanner(strategies=[])
        yes_token = "123456789012345678901"
        no_token = "123456789012345678902"
        opp = ArbitrageOpportunity(
            strategy=StrategyType.WEATHER_EDGE,
            title="Weather",
            description="D",
            total_cost=0.2,
            expected_payout=0.5,
            gross_profit=0.3,
            fee=0.01,
            net_profit=0.29,
            roi_percent=145.0,
            markets=[
                {
                    "id": "m_weather_2",
                    "platform": "polymarket",
                    "clob_token_ids": [yes_token, no_token],
                    "yes_price": 0.2,
                    "no_price": 0.8,
                }
            ],
            min_liquidity=1000.0,
            max_position_size=10.0,
            positions_to_take=[],
        )

        scanner._backfill_market_history_for_opportunities = AsyncMock(return_value=None)
        scanner.get_market_history_for_opportunities = MagicMock(
            return_value={
                "m_weather_2": [
                    {"t": 1.0, "yes": 0.41, "no": 0.59},
                    {"t": 2.0, "yes": 0.43, "no": 0.57},
                ]
            }
        )

        attached = await scanner.attach_price_history_to_opportunities(
            [opp], timeout_seconds=None
        )

        assert attached == 1
        assert "price_history" in opp.markets[0]
        assert len(opp.markets[0]["price_history"]) == 2


# ---------------------------------------------------------------------------
# get_opportunities with OpportunityFilter
# ---------------------------------------------------------------------------


class TestGetOpportunities:
    """Tests for ArbitrageScanner.get_opportunities with filtering."""

    def _scanner_with_opportunities(self, opps):
        scanner = _build_scanner()
        scanner._opportunities = list(opps)
        return scanner

    def test_no_filter_returns_all(
        self,
        sample_opportunity,
        sample_opportunity_high_roi,
        sample_opportunity_low_roi,
    ):
        scanner = self._scanner_with_opportunities(
            [
                sample_opportunity,
                sample_opportunity_high_roi,
                sample_opportunity_low_roi,
            ]
        )
        result = scanner.get_opportunities()
        assert len(result) == 3

    def test_min_profit_filter(
        self,
        sample_opportunity,
        sample_opportunity_high_roi,
        sample_opportunity_low_roi,
    ):
        """min_profit is multiplied by 100 in the filter logic.
        - sample_opportunity: roi_percent=2.08 -> min_profit=0.02 (2%) pass if min_profit<=0.0208
        - sample_opportunity_high_roi: roi_percent=15.29
        - sample_opportunity_low_roi: roi_percent=1.03
        """
        scanner = self._scanner_with_opportunities(
            [
                sample_opportunity,
                sample_opportunity_high_roi,
                sample_opportunity_low_roi,
            ]
        )
        f = OpportunityFilter(min_profit=0.05)  # 0.05 * 100 = 5% threshold
        result = scanner.get_opportunities(filter=f)

        # Only high-roi opp (15.29%) passes 5% threshold
        assert len(result) == 1
        assert result[0].roi_percent == 15.29

    def test_max_risk_filter(
        self,
        sample_opportunity,
        sample_opportunity_high_roi,
        sample_opportunity_low_roi,
    ):
        """
        Risk scores:
        - sample_opportunity: 0.3
        - sample_opportunity_high_roi: 0.2
        - sample_opportunity_low_roi: 0.8
        """
        scanner = self._scanner_with_opportunities(
            [
                sample_opportunity,
                sample_opportunity_high_roi,
                sample_opportunity_low_roi,
            ]
        )
        f = OpportunityFilter(max_risk=0.5)
        result = scanner.get_opportunities(filter=f)

        assert len(result) == 2
        for opp in result:
            assert opp.risk_score <= 0.5

    def test_strategy_filter(
        self,
        sample_opportunity,
        sample_opportunity_high_roi,
        sample_opportunity_low_roi,
    ):
        scanner = self._scanner_with_opportunities(
            [
                sample_opportunity,
                sample_opportunity_high_roi,
                sample_opportunity_low_roi,
            ]
        )
        f = OpportunityFilter(strategies=[StrategyType.NEGRISK])
        result = scanner.get_opportunities(filter=f)

        assert len(result) == 1
        assert result[0].strategy == StrategyType.NEGRISK

    def test_category_filter(
        self,
        sample_opportunity,
        sample_opportunity_high_roi,
        sample_opportunity_low_roi,
    ):
        scanner = self._scanner_with_opportunities(
            [
                sample_opportunity,
                sample_opportunity_high_roi,
                sample_opportunity_low_roi,
            ]
        )
        f = OpportunityFilter(category="Crypto")
        result = scanner.get_opportunities(filter=f)

        assert len(result) == 1
        assert result[0].category == "Crypto"

    def test_category_filter_case_insensitive(self, sample_opportunity):
        scanner = self._scanner_with_opportunities([sample_opportunity])
        f = OpportunityFilter(category="crypto")  # lowercase
        result = scanner.get_opportunities(filter=f)

        assert len(result) == 1

    def test_combined_filters(
        self,
        sample_opportunity,
        sample_opportunity_high_roi,
        sample_opportunity_low_roi,
    ):
        """Multiple filter criteria combine (AND logic)."""
        scanner = self._scanner_with_opportunities(
            [
                sample_opportunity,
                sample_opportunity_high_roi,
                sample_opportunity_low_roi,
            ]
        )
        f = OpportunityFilter(max_risk=0.5, strategies=[StrategyType.BASIC])
        result = scanner.get_opportunities(filter=f)

        # sample_opportunity: risk=0.3, strategy=BASIC -> matches
        # sample_opportunity_high_roi: risk=0.2, strategy=NEGRISK -> strategy mismatch
        # sample_opportunity_low_roi: risk=0.8 -> risk too high
        assert len(result) == 1
        assert result[0].strategy == StrategyType.BASIC

    def test_min_liquidity_filter(self, sample_opportunity, sample_opportunity_low_roi):
        scanner = self._scanner_with_opportunities(
            [sample_opportunity, sample_opportunity_low_roi]
        )
        f = OpportunityFilter(min_liquidity=1000.0)
        result = scanner.get_opportunities(filter=f)

        # sample_opportunity: min_liquidity=5000 (pass)
        # sample_opportunity_low_roi: min_liquidity=500 (fail)
        assert len(result) == 1
        assert result[0].min_liquidity >= 1000.0

    def test_empty_result(self, sample_opportunity):
        scanner = self._scanner_with_opportunities([sample_opportunity])
        f = OpportunityFilter(min_profit=1.0)  # 100% ROI threshold
        result = scanner.get_opportunities(filter=f)
        assert result == []


# ---------------------------------------------------------------------------
# clear / remove operations
# ---------------------------------------------------------------------------


class TestClearAndRemove:
    """Tests for clear_opportunities, remove_expired, remove_old."""

    def test_clear_opportunities(self, sample_opportunity, sample_opportunity_high_roi):
        scanner = _build_scanner()
        scanner._opportunities = [sample_opportunity, sample_opportunity_high_roi]

        count = scanner.clear_opportunities()

        assert count == 2
        assert scanner._opportunities == []

    def test_clear_empty_returns_zero(self):
        scanner = _build_scanner()
        count = scanner.clear_opportunities()
        assert count == 0

    def test_remove_expired_opportunities(
        self, sample_opportunity, expired_opportunity
    ):
        scanner = _build_scanner()
        scanner._opportunities = [sample_opportunity, expired_opportunity]

        removed = scanner.remove_expired_opportunities()

        assert removed == 1
        assert len(scanner._opportunities) == 1
        assert scanner._opportunities[0].title == sample_opportunity.title

    def test_remove_expired_keeps_none_resolution(self, sample_opportunity):
        """Opportunities with resolution_date=None are never expired."""
        scanner = _build_scanner()
        scanner._opportunities = [sample_opportunity]  # resolution_date=None

        removed = scanner.remove_expired_opportunities()

        assert removed == 0
        assert len(scanner._opportunities) == 1

    def test_remove_old_opportunities(self, sample_opportunity, old_opportunity):
        scanner = _build_scanner()
        scanner._opportunities = [sample_opportunity, old_opportunity]

        removed = scanner.remove_old_opportunities(max_age_minutes=60)

        assert removed == 1
        assert len(scanner._opportunities) == 1
        assert scanner._opportunities[0].title == sample_opportunity.title

    def test_remove_old_custom_age(self, sample_opportunity):
        """With a very small max_age, even recent opportunities are removed."""
        scanner = _build_scanner()
        # Manually set detected_at to 5 minutes ago
        opp = sample_opportunity.model_copy()
        opp.detected_at = utcnow() - timedelta(minutes=5)
        scanner._opportunities = [opp]

        removed = scanner.remove_old_opportunities(max_age_minutes=3)

        assert removed == 1
        assert scanner._opportunities == []


# ---------------------------------------------------------------------------
# Scanner status
# ---------------------------------------------------------------------------


class TestScannerStatus:
    """Tests for get_status."""

    def test_status_initial(self):
        scanner = _build_scanner()
        status = scanner.get_status()

        assert status["running"] is False
        assert status["enabled"] is True
        assert status["last_scan"] is None
        assert status["opportunities_count"] == 0
        assert isinstance(status["strategies"], list)
        assert len(status["strategies"]) == 8

    def test_status_after_scan(self, mock_polymarket_client):
        scanner = _build_scanner(mock_client=mock_polymarket_client)

        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.run_until_complete(scanner.scan_once())

        status = scanner.get_status()
        assert status["last_scan"] is not None
        assert status["last_scan"].endswith("Z")

    def test_status_strategies_structure(self):
        scanner = _build_scanner()
        status = scanner.get_status()

        for s in status["strategies"]:
            assert "name" in s
            assert "type" in s

    def test_status_handles_string_strategy_type(self):
        scanner = _build_scanner()
        plugin_like = MagicMock()
        plugin_like.name = "Plugin Strategy"
        plugin_like.strategy_type = "plugin_test"
        scanner.strategies = [plugin_like]

        status = scanner.get_status()
        assert status["strategies"][0]["name"] == "Plugin Strategy"
        assert status["strategies"][0]["type"] == "plugin_test"

    def test_status_reflects_opportunity_count(self, sample_opportunity):
        scanner = _build_scanner()
        scanner._opportunities = [sample_opportunity]
        status = scanner.get_status()
        assert status["opportunities_count"] == 1


# ---------------------------------------------------------------------------
# set_interval bounds
# ---------------------------------------------------------------------------


class TestSetInterval:
    """Tests for set_interval clamping."""

    @pytest.mark.asyncio
    async def test_set_interval_normal(self):
        scanner = _build_scanner()
        with patch.object(scanner, "save_settings", new_callable=AsyncMock):
            with patch.object(scanner, "_notify_status_change", new_callable=AsyncMock):
                await scanner.set_interval(120)
        assert scanner._interval_seconds == 120

    @pytest.mark.asyncio
    async def test_set_interval_below_minimum_clamped_to_10(self):
        scanner = _build_scanner()
        with patch.object(scanner, "save_settings", new_callable=AsyncMock):
            with patch.object(scanner, "_notify_status_change", new_callable=AsyncMock):
                await scanner.set_interval(3)
        assert scanner._interval_seconds == 10

    @pytest.mark.asyncio
    async def test_set_interval_above_maximum_clamped_to_3600(self):
        scanner = _build_scanner()
        with patch.object(scanner, "save_settings", new_callable=AsyncMock):
            with patch.object(scanner, "_notify_status_change", new_callable=AsyncMock):
                await scanner.set_interval(9999)
        assert scanner._interval_seconds == 3600

    @pytest.mark.asyncio
    async def test_set_interval_at_boundary_10(self):
        scanner = _build_scanner()
        with patch.object(scanner, "save_settings", new_callable=AsyncMock):
            with patch.object(scanner, "_notify_status_change", new_callable=AsyncMock):
                await scanner.set_interval(10)
        assert scanner._interval_seconds == 10

    @pytest.mark.asyncio
    async def test_set_interval_at_boundary_3600(self):
        scanner = _build_scanner()
        with patch.object(scanner, "save_settings", new_callable=AsyncMock):
            with patch.object(scanner, "_notify_status_change", new_callable=AsyncMock):
                await scanner.set_interval(3600)
        assert scanner._interval_seconds == 3600


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class TestScannerCallbacks:
    """Tests for scan and status callbacks."""

    @pytest.mark.asyncio
    async def test_scan_callback_invoked(
        self, mock_polymarket_client, sample_opportunity
    ):
        strategy = MagicMock()
        strategy.name = "S"
        strategy.strategy_type = StrategyType.BASIC
        strategy.detect = MagicMock(return_value=[sample_opportunity])

        scanner = _build_scanner(
            mock_client=mock_polymarket_client,
            strategies=[strategy],
        )

        callback = AsyncMock()
        scanner.add_callback(callback)

        await scanner.scan_once()

        callback.assert_awaited_once()
        # Callback receives the list of opportunities
        call_args = callback.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0].strategy == StrategyType.BASIC

    @pytest.mark.asyncio
    async def test_multiple_scan_callbacks(self, mock_polymarket_client):
        scanner = _build_scanner(
            mock_client=mock_polymarket_client,
            strategies=[],
        )

        cb1 = AsyncMock()
        cb2 = AsyncMock()
        scanner.add_callback(cb1)
        scanner.add_callback(cb2)

        await scanner.scan_once()

        cb1.assert_awaited_once()
        cb2.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_break_scan(
        self, mock_polymarket_client, sample_opportunity
    ):
        strategy = MagicMock()
        strategy.name = "S"
        strategy.strategy_type = StrategyType.BASIC
        strategy.detect = MagicMock(return_value=[sample_opportunity])

        scanner = _build_scanner(
            mock_client=mock_polymarket_client,
            strategies=[strategy],
        )

        bad_cb = AsyncMock(side_effect=RuntimeError("callback failed"))
        good_cb = AsyncMock()
        scanner.add_callback(bad_cb)
        scanner.add_callback(good_cb)

        results = await scanner.scan_once()

        # Scan should still return results
        assert len(results) == 1
        # Good callback still called despite bad one failing
        good_cb.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_status_callback_invoked_on_set_interval(self):
        scanner = _build_scanner()
        status_cb = AsyncMock()
        scanner.add_status_callback(status_cb)

        with patch.object(scanner, "save_settings", new_callable=AsyncMock):
            await scanner.set_interval(30)

        status_cb.assert_awaited_once()
        # Callback receives the status dict
        call_args = status_cb.call_args[0][0]
        assert "running" in call_args
        assert "enabled" in call_args
        assert call_args["interval_seconds"] == 30

    @pytest.mark.asyncio
    async def test_status_callback_exception_handled(self):
        scanner = _build_scanner()
        bad_status_cb = AsyncMock(side_effect=RuntimeError("status cb failed"))
        scanner.add_status_callback(bad_status_cb)

        with patch.object(scanner, "save_settings", new_callable=AsyncMock):
            # Should not raise
            await scanner.set_interval(60)

        bad_status_cb.assert_awaited_once()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestScannerProperties:
    """Tests for scanner property accessors."""

    def test_last_scan_initially_none(self):
        scanner = _build_scanner()
        assert scanner.last_scan is None

    def test_is_running_initially_false(self):
        scanner = _build_scanner()
        assert scanner.is_running is False

    def test_is_enabled_initially_true(self):
        scanner = _build_scanner()
        assert scanner.is_enabled is True

    def test_interval_seconds(self):
        scanner = _build_scanner()
        assert scanner.interval_seconds == 60  # default from settings

    @pytest.mark.asyncio
    async def test_stop_sets_flags(self):
        scanner = _build_scanner()
        scanner._running = True
        scanner._enabled = True
        await scanner.stop()
        assert scanner.is_running is False
        assert scanner.is_enabled is False
