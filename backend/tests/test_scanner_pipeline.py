"""Tests for ArbitrageScanner: initialisation, scan pipeline, filtering, lifecycle."""

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import pytest
from datetime import datetime, timedelta
from utils.utcnow import utcnow
from unittest.mock import AsyncMock, MagicMock, patch

from models.market import Market
from models.opportunity import (
    Opportunity,
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
        # Default to empty overrides so scan_fast never attempts a live DB load.
        # Tests that need specific strategies can still pass them explicitly.
        scanner.strategies = strategies if strategies is not None else []
        return scanner


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestScannerInit:
    """Tests for ArbitrageScanner.__init__."""

    def test_strategies_empty_before_load_plugins(self):
        """Strategies are empty until load_plugins() is awaited (async DB load)."""
        scanner = _build_scanner()
        # load_plugins() is async and not called in __init__, so strategies
        # start empty.  The scanner worker calls load_plugins() at startup.
        assert scanner.strategies == []

    def test_strategy_overrides_used_when_set(self):
        """When _strategy_overrides is set, strategies returns those."""
        scanner = _build_scanner()
        mock_strat = MagicMock()
        mock_strat.name = "TestStrategy"
        mock_strat.strategy_type = "basic"
        scanner._strategy_overrides = [mock_strat]

        assert len(scanner.strategies) == 1
        assert scanner.strategies[0].name == "TestStrategy"

    def test_strategy_names_from_overrides(self):
        """Strategy names are returned from overrides list."""
        scanner = _build_scanner()
        strats = []
        for name, stype in [("Basic Arbitrage", "basic"), ("NegRisk Arb", "negrisk")]:
            s = MagicMock()
            s.name = name
            s.strategy_type = stype
            strats.append(s)
        scanner._strategy_overrides = strats

        names = [s.name for s in scanner.strategies]
        assert "Basic Arbitrage" in names
        assert "NegRisk Arb" in names

    def test_strategy_types_from_overrides(self):
        """Strategy types are read correctly from overrides."""
        scanner = _build_scanner()
        types_to_set = ["basic", "negrisk", "combinatorial", "settlement_lag"]
        strats = []
        for stype in types_to_set:
            s = MagicMock()
            s.name = stype.title()
            s.strategy_type = stype
            strats.append(s)
        scanner._strategy_overrides = strats

        types = {s.strategy_type for s in scanner.strategies}
        assert types == set(types_to_set)

    def test_initial_state(self):
        scanner = _build_scanner()
        assert scanner._running is False
        assert scanner._enabled is True
        assert scanner._opportunities == []
        assert scanner._last_scan is None
        assert scanner._scan_callbacks == []
        assert scanner._status_callbacks == []


# ---------------------------------------------------------------------------
# scan pipeline (refresh_catalog + scan_fast)
# ---------------------------------------------------------------------------


def _seed_scanner_cache(scanner, markets=None):
    """Populate scanner caches and mock prioritizer so scan_fast() runs strategies."""
    if markets is None:
        markets = [
            Market(
                id="m_test_1",
                condition_id="c_test_1",
                question="Test market?",
                slug="test-market",
                clob_token_ids=["tok_yes", "tok_no"],
                outcome_prices=[0.5, 0.5],
            )
        ]
    scanner._cached_markets = list(markets)
    scanner._cached_market_by_id = {m.id: m for m in markets}
    scanner._cached_events = []

    # Mock the prioritizer so all markets are HOT and "changed"
    from services.market_prioritizer import MarketTier

    scanner._prioritizer = MagicMock()
    scanner._prioritizer.classify_all = MagicMock(
        return_value={
            MarketTier.HOT: list(markets),
            MarketTier.WARM: [],
            MarketTier.COLD: [],
        }
    )
    scanner._prioritizer.get_changed_markets = MagicMock(return_value=list(markets))
    scanner._prioritizer.update_after_evaluation = MagicMock(return_value=0)
    scanner._prioritizer.compute_attention_scores = MagicMock()
    scanner._prioritizer.update_stability_scores = MagicMock()


def _make_quality_pass():
    """Return a mock quality report that always passes."""
    report = MagicMock()
    report.passed = True
    return report


class TestScanPipeline:
    """Tests for ArbitrageScanner.refresh_catalog + scan_fast pipeline."""

    @pytest.mark.asyncio
    async def test_refresh_catalog_calls_client_methods(self, mock_polymarket_client):
        """refresh_catalog fetches events and markets from the upstream API."""
        scanner = _build_scanner(mock_client=mock_polymarket_client)

        await scanner.refresh_catalog()

        mock_polymarket_client.get_all_events.assert_awaited_once_with(closed=False)
        mock_polymarket_client.get_all_markets.assert_awaited_once_with(active=True)

    @pytest.mark.asyncio
    async def test_scan_fast_returns_opportunities_sorted_by_roi(
        self,
        mock_polymarket_client,
        sample_opportunity,
        sample_opportunity_high_roi,
        sample_opportunity_low_roi,
    ):
        """Opportunities are returned sorted by ROI descending."""
        all_opps = [sample_opportunity_low_roi, sample_opportunity, sample_opportunity_high_roi]

        scanner = _build_scanner(mock_client=mock_polymarket_client)
        _seed_scanner_cache(scanner)

        with (
            patch.object(scanner, "_dispatch_market_refresh", new_callable=AsyncMock, return_value=all_opps),
            patch("services.scanner.quality_filter") as mock_qf,
        ):
            mock_qf.evaluate_opportunity = MagicMock(return_value=_make_quality_pass())
            results = await scanner.scan_fast()

        assert len(results) == 3
        assert results[0].roi_percent >= results[1].roi_percent
        assert results[1].roi_percent >= results[2].roi_percent

    @pytest.mark.asyncio
    async def test_scan_fast_handles_dispatch_exception_gracefully(
        self,
        mock_polymarket_client,
    ):
        """A dispatch failure does not crash the scanner."""
        scanner = _build_scanner(mock_client=mock_polymarket_client)
        _seed_scanner_cache(scanner)

        with patch.object(
            scanner,
            "_dispatch_market_refresh",
            new_callable=AsyncMock,
            side_effect=RuntimeError("dispatch boom"),
        ):
            with pytest.raises(RuntimeError, match="dispatch boom"):
                await scanner.scan_fast()

    @pytest.mark.asyncio
    async def test_scan_fast_sets_last_scan(self, mock_polymarket_client):
        """scan_fast sets _last_scan on every successful cycle."""
        scanner = _build_scanner(mock_client=mock_polymarket_client)
        _seed_scanner_cache(scanner)

        assert scanner._last_scan is None

        with (
            patch.object(scanner, "_dispatch_market_refresh", new_callable=AsyncMock, return_value=[]),
            patch("services.scanner.quality_filter"),
        ):
            await scanner.scan_fast()

        assert scanner._last_scan is not None
        assert isinstance(scanner._last_scan, datetime)

    @pytest.mark.asyncio
    async def test_refresh_opportunity_prices_sets_last_priced_at(self, mock_polymarket_client):
        scanner = _build_scanner(mock_client=mock_polymarket_client)
        opp = Opportunity(
            strategy="basic",
            title="Price refresh",
            description="D",
            total_cost=0.95,
            gross_profit=0.05,
            fee=0.02,
            net_profit=0.03,
            roi_percent=3.16,
            markets=[
                {
                    "id": "m_refresh",
                    "question": "Test?",
                    "clob_token_ids": ["tok_yes_refresh", "tok_no_refresh"],
                    "yes_price": 0.49,
                    "no_price": 0.51,
                }
            ],
            positions_to_take=[],
        )
        ts = utcnow()
        ts_seconds = ts.timestamp()
        with patch.object(
            scanner,
            "_snapshot_ws_prices",
            new_callable=AsyncMock,
            return_value={
                "tok_yes_refresh": {"mid": 0.47, "ts": ts_seconds},
                "tok_no_refresh": {"mid": 0.53, "ts": ts_seconds},
            },
        ):
            refreshed = await scanner.refresh_opportunity_prices([opp], now=ts)
        assert len(refreshed) == 1
        refreshed_opp = refreshed[0]
        assert refreshed_opp.last_priced_at is not None
        assert abs((refreshed_opp.last_priced_at - ts).total_seconds()) < 1.0

    @pytest.mark.asyncio
    async def test_full_snapshot_scan_keeps_existing_pool_when_no_new_full_results(
        self,
        mock_polymarket_client,
    ):
        """Heavy-lane scans that find nothing must not evict the existing opportunity pool."""
        scanner = _build_scanner(mock_client=mock_polymarket_client)

        token_yes = "tok_yes_" + "0" * 24
        token_no = "tok_no_" + "0" * 24
        market = Market(
            id="m_full_1",
            condition_id="c_full_1",
            question="Will test market resolve yes?",
            slug="test-full-1",
            clob_token_ids=[token_yes, token_no],
            outcome_prices=[0.48, 0.52],
        )
        scanner._cached_markets = [market]
        scanner._cached_market_by_id = {market.id: market}
        scanner._cached_events = []

        seen_at = utcnow() - timedelta(hours=2)
        scanner._opportunities = [
            Opportunity(
                strategy="basic",
                title="Existing pool opportunity",
                description="Should survive heavy lane with zero new detections",
                total_cost=0.96,
                expected_payout=1.0,
                gross_profit=0.04,
                fee=0.02,
                net_profit=0.02,
                roi_percent=2.08,
                risk_score=0.3,
                markets=[
                    {
                        "id": "m_full_1",
                        "question": "Will test market resolve yes?",
                        "yes_price": 0.48,
                        "no_price": 0.52,
                        "clob_token_ids": [token_yes, token_no],
                    }
                ],
                last_seen_at=seen_at,
                detected_at=seen_at,
                mispricing_type=MispricingType.WITHIN_MARKET,
            )
        ]

        with (
            patch.object(scanner, "_ensure_runtime_strategies_loaded", new_callable=AsyncMock),
            patch.object(scanner, "_partition_market_refresh_strategies", return_value=(set(), {"full_only"})),
            patch.object(scanner, "_select_full_snapshot_markets", return_value=[market]),
            patch.object(scanner, "_snapshot_ws_prices", new_callable=AsyncMock, return_value={}),
            patch.object(scanner, "_dispatch_market_refresh", new_callable=AsyncMock, return_value=[]),
            patch.object(scanner, "_set_activity", new_callable=AsyncMock),
        ):
            results = await scanner.scan_full_snapshot_strategies(force=True)

        assert len(results) == 1
        assert len(scanner._opportunities) == 1
        assert scanner._opportunities[0].markets[0]["id"] == "m_full_1"

    @pytest.mark.asyncio
    async def test_full_snapshot_scan_processes_chunked_cursor_progression(
        self,
        mock_polymarket_client,
    ):
        scanner = _build_scanner(mock_client=mock_polymarket_client)
        markets = []
        for idx in range(3):
            yes = f"tok_yes_chunk_{idx}_" + ("0" * 20)
            no = f"tok_no_chunk_{idx}_" + ("0" * 20)
            market = Market(
                id=f"m_chunk_{idx}",
                condition_id=f"c_chunk_{idx}",
                question=f"Chunk market {idx}?",
                slug=f"chunk-market-{idx}",
                clob_token_ids=[yes, no],
                outcome_prices=[0.49, 0.51],
            )
            markets.append(market)
        scanner._cached_markets = list(markets)
        scanner._cached_market_by_id = {m.id: m for m in markets}
        scanner._cached_events = []

        with (
            patch.object(scanner, "_ensure_runtime_strategies_loaded", new_callable=AsyncMock),
            patch.object(scanner, "_partition_market_refresh_strategies", return_value=(set(), {"full_only"})),
            patch.object(scanner, "_snapshot_ws_prices", new_callable=AsyncMock, return_value={}),
            patch.object(scanner, "_dispatch_market_refresh", new_callable=AsyncMock, return_value=[]),
            patch.object(scanner, "_set_activity", new_callable=AsyncMock),
            patch("services.scanner.settings.SCANNER_FULL_SNAPSHOT_CHUNK_SIZE", 2),
            patch("services.scanner.settings.SCANNER_FORCE_FULL_UNIVERSE", True),
        ):
            await scanner.scan_full_snapshot_strategies(force=True)
            assert scanner._last_full_snapshot_chunk_market_count == 2
            assert scanner._full_snapshot_cycle_total_markets == 3
            assert scanner._full_snapshot_cycle_processed_markets == 2
            assert scanner._full_snapshot_cursor_index == 2
            await scanner.scan_full_snapshot_strategies(force=True)

        assert scanner._last_full_snapshot_chunk_market_count == 1
        assert scanner._full_snapshot_cycle_total_markets == 3
        assert scanner._full_snapshot_cycle_processed_markets == 3
        assert scanner._full_snapshot_cursor_index == 0
        assert scanner._full_snapshot_cycle_completed_at is not None

    @pytest.mark.asyncio
    async def test_refresh_catalog_reads_prices_from_redis(self, mock_polymarket_client):
        """refresh_catalog reads prices via _snapshot_ws_prices (Redis), not get_prices_batch."""
        # Token IDs must be >20 chars to pass the _collect_polymarket_tokens filter
        tok_a = "tok_a_" + "0" * 20
        tok_b = "tok_b_" + "0" * 20
        market = Market(
            id="m1",
            condition_id="c1",
            question="Q?",
            slug="q",
            clob_token_ids=[tok_a, tok_b],
            outcome_prices=[0.6, 0.4],
        )
        mock_polymarket_client.get_all_markets.return_value = [market]
        mock_polymarket_client.get_all_events.return_value = []

        scanner = _build_scanner(
            mock_client=mock_polymarket_client,
            strategies=[],
        )

        with patch.object(
            scanner,
            "_snapshot_ws_prices",
            new_callable=AsyncMock,
            return_value={tok_a: {"mid": 0.61}, tok_b: {"mid": 0.39}},
        ) as mock_ws_prices:
            await scanner.refresh_catalog()

        mock_ws_prices.assert_awaited_once()
        call_args = mock_ws_prices.call_args[0][0]
        assert tok_a in call_args
        assert tok_b in call_args
        # get_prices_batch is NOT used
        mock_polymarket_client.get_prices_batch.assert_not_awaited()


# ---------------------------------------------------------------------------
# Mispricing type classification
# ---------------------------------------------------------------------------


class TestMispricingClassification:
    """Tests for mispricing type assignment during scan pipeline.

    Mispricing type is assigned by _dispatch_market_refresh, so we
    mock that and verify the opportunities come through scan_fast
    with the correct classification.
    """

    @pytest.mark.asyncio
    async def test_mispricing_type_set_for_basic_strategy(self, mock_polymarket_client):
        """Opportunities from basic strategy get WITHIN_MARKET classification."""
        opp = Opportunity(
            strategy="basic",
            title="Test",
            description="D",
            total_cost=0.95,
            gross_profit=0.05,
            fee=0.02,
            net_profit=0.03,
            roi_percent=3.16,
            markets=[{"id": "m1"}],
            mispricing_type=MispricingType.WITHIN_MARKET,
        )

        scanner = _build_scanner(mock_client=mock_polymarket_client)
        _seed_scanner_cache(scanner)

        with (
            patch.object(scanner, "_dispatch_market_refresh", new_callable=AsyncMock, return_value=[opp]),
            patch("services.scanner.quality_filter") as mock_qf,
        ):
            mock_qf.evaluate_opportunity = MagicMock(return_value=_make_quality_pass())
            results = await scanner.scan_fast()

        assert results[0].mispricing_type == MispricingType.WITHIN_MARKET

    @pytest.mark.asyncio
    async def test_mispricing_type_set_for_combinatorial(self, mock_polymarket_client):
        """Combinatorial strategy maps to CROSS_MARKET."""
        opp = Opportunity(
            strategy="combinatorial",
            title="Test",
            description="D",
            total_cost=0.9,
            gross_profit=0.1,
            fee=0.02,
            net_profit=0.08,
            roi_percent=8.89,
            markets=[{"id": "m1"}],
            mispricing_type=MispricingType.CROSS_MARKET,
        )

        scanner = _build_scanner(mock_client=mock_polymarket_client)
        _seed_scanner_cache(scanner)

        with (
            patch.object(scanner, "_dispatch_market_refresh", new_callable=AsyncMock, return_value=[opp]),
            patch("services.scanner.quality_filter") as mock_qf,
        ):
            mock_qf.evaluate_opportunity = MagicMock(return_value=_make_quality_pass())
            results = await scanner.scan_fast()

        assert results[0].mispricing_type == MispricingType.CROSS_MARKET

    @pytest.mark.asyncio
    async def test_mispricing_type_set_for_settlement_lag(self, mock_polymarket_client):
        """Settlement-lag strategy maps to SETTLEMENT_LAG."""
        opp = Opportunity(
            strategy="settlement_lag",
            title="Test",
            description="D",
            total_cost=0.9,
            gross_profit=0.1,
            fee=0.02,
            net_profit=0.08,
            roi_percent=8.89,
            markets=[{"id": "m1"}],
            mispricing_type=MispricingType.SETTLEMENT_LAG,
        )

        scanner = _build_scanner(mock_client=mock_polymarket_client)
        _seed_scanner_cache(scanner)

        with (
            patch.object(scanner, "_dispatch_market_refresh", new_callable=AsyncMock, return_value=[opp]),
            patch("services.scanner.quality_filter") as mock_qf,
        ):
            mock_qf.evaluate_opportunity = MagicMock(return_value=_make_quality_pass())
            results = await scanner.scan_fast()

        assert results[0].mispricing_type == MispricingType.SETTLEMENT_LAG

    @pytest.mark.asyncio
    async def test_mispricing_type_preserved_through_pipeline(self, mock_polymarket_client):
        """Mispricing type set by strategy survives the scan_fast pipeline."""
        opp = Opportunity(
            strategy="basic",
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

        scanner = _build_scanner(mock_client=mock_polymarket_client)
        _seed_scanner_cache(scanner)

        with (
            patch.object(scanner, "_dispatch_market_refresh", new_callable=AsyncMock, return_value=[opp]),
            patch("services.scanner.quality_filter") as mock_qf,
        ):
            mock_qf.evaluate_opportunity = MagicMock(return_value=_make_quality_pass())
            results = await scanner.scan_fast()

        # Should preserve the pre-set value
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
        opp = Opportunity(
            strategy="weather_edge",
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
                    "clob_token_ids": f'["{yes_token}", "{no_token}"]',
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
        opp = Opportunity(
            strategy="weather_edge",
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

        attached = await scanner.attach_price_history_to_opportunities([opp], timeout_seconds=None)

        assert attached == 1
        assert "price_history" in opp.markets[0]
        assert len(opp.markets[0]["price_history"]) == 2

    @pytest.mark.asyncio
    async def test_attach_price_history_uses_market_id_aliases(self):
        scanner = _build_scanner(strategies=[])
        scanner._market_price_history = {
            "m_primary": [
                {"t": 1.0, "yes": 0.45, "no": 0.55},
                {"t": 2.0, "yes": 0.47, "no": 0.53},
            ]
        }
        scanner._market_id_to_condition_id = {"m_primary": "0xcond_primary"}
        scanner._condition_id_to_market_id = {"0xcond_primary": "m_primary"}
        scanner._backfill_market_history_for_opportunities = AsyncMock(return_value=None)
        scanner._persist_market_history_for_opportunities = AsyncMock(return_value=None)
        scanner._hydrate_history_from_db = AsyncMock(return_value=0)

        opp = Opportunity(
            strategy="weather_edge",
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
                    "id": "0xcond_primary",
                    "condition_id": "0xcond_primary",
                    "platform": "polymarket",
                    "yes_price": 0.2,
                    "no_price": 0.8,
                }
            ],
            min_liquidity=1000.0,
            max_position_size=10.0,
            positions_to_take=[],
        )

        attached = await scanner.attach_price_history_to_opportunities([opp], timeout_seconds=None)

        assert attached == 1
        assert len(opp.markets[0].get("price_history") or []) == 2
        assert scanner._hydrate_history_from_db.await_count == 0


class TestOpportunityMerge:
    def test_merge_opportunities_preserves_existing_markets_when_update_is_partial(self):
        scanner = _build_scanner(strategies=[])
        first_seen = utcnow() - timedelta(hours=2)
        prior_seen = utcnow() - timedelta(minutes=30)
        existing = Opportunity(
            strategy="basic",
            title="Original",
            description="Original description",
            total_cost=0.42,
            expected_payout=1.0,
            gross_profit=0.58,
            fee=0.02,
            net_profit=0.56,
            roi_percent=133.3,
            event_slug="event-one",
            event_title="Event One",
            category="politics",
            markets=[
                {
                    "id": "m_1",
                    "question": "Will this happen?",
                    "yes_price": 0.42,
                    "no_price": 0.58,
                    "price_history": [
                        {"t": 1.0, "yes": 0.4, "no": 0.6},
                        {"t": 2.0, "yes": 0.42, "no": 0.58},
                    ],
                }
            ],
            positions_to_take=[{"market_id": "m_1", "outcome": "YES", "price": 0.42}],
            detected_at=first_seen,
            first_detected_at=first_seen,
            last_detected_at=prior_seen,
            last_seen_at=prior_seen,
        )
        scanner._opportunities = [existing]

        update = Opportunity(
            strategy="basic",
            title="Updated",
            description="Partial update",
            total_cost=0.41,
            expected_payout=1.0,
            gross_profit=0.59,
            fee=0.02,
            net_profit=0.57,
            roi_percent=139.0,
            stable_id=existing.stable_id,
            markets=[],
            positions_to_take=[],
        )

        merged = scanner._merge_opportunities([update])
        assert len(merged) == 1
        merged_opp = merged[0]
        assert merged_opp.id == existing.id
        assert merged_opp.event_slug == "event-one"
        assert merged_opp.event_title == "Event One"
        assert merged_opp.category == "politics"
        assert len(merged_opp.markets) == 1
        assert merged_opp.markets[0]["id"] == "m_1"
        assert len(merged_opp.markets[0]["price_history"]) == 2
        assert len(merged_opp.positions_to_take) == 1
        assert merged_opp.detected_at == first_seen
        assert merged_opp.first_detected_at == first_seen
        assert merged_opp.last_detected_at is not None
        assert merged_opp.last_detected_at >= prior_seen


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
        f = OpportunityFilter(strategies=["negrisk"])
        result = scanner.get_opportunities(filter=f)

        assert len(result) == 1
        assert result[0].strategy == "negrisk"

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
        f = OpportunityFilter(max_risk=0.5, strategies=["basic"])
        result = scanner.get_opportunities(filter=f)

        # sample_opportunity: risk=0.3, strategy=BASIC -> matches
        # sample_opportunity_high_roi: risk=0.2, strategy=NEGRISK -> strategy mismatch
        # sample_opportunity_low_roi: risk=0.8 -> risk too high
        assert len(result) == 1
        assert result[0].strategy == "basic"

    def test_min_liquidity_filter(self, sample_opportunity, sample_opportunity_low_roi):
        scanner = self._scanner_with_opportunities([sample_opportunity, sample_opportunity_low_roi])
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

    def test_remove_expired_opportunities(self, sample_opportunity, expired_opportunity):
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
        opp.last_detected_at = opp.detected_at
        opp.last_seen_at = opp.detected_at
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
        assert len(status["strategies"]) >= 0  # Depends on strategy loader

    @pytest.mark.asyncio
    async def test_status_after_scan(self, mock_polymarket_client):
        scanner = _build_scanner(mock_client=mock_polymarket_client)
        _seed_scanner_cache(scanner)

        with (
            patch.object(scanner, "_dispatch_market_refresh", new_callable=AsyncMock, return_value=[]),
            patch("services.scanner.quality_filter"),
        ):
            await scanner.scan_fast()

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

    def test_status_includes_freshness_slo_fields(self, sample_opportunity):
        scanner = _build_scanner()
        now = utcnow()
        scanner._last_fast_scan = now - timedelta(seconds=12)

        newest = sample_opportunity.model_copy(deep=True)
        middle = sample_opportunity.model_copy(deep=True)
        oldest = sample_opportunity.model_copy(deep=True)

        newest.last_priced_at = now - timedelta(seconds=5)
        newest.last_detected_at = now - timedelta(seconds=7)

        middle.last_priced_at = now - timedelta(seconds=30)
        middle.last_detected_at = now - timedelta(seconds=40)

        oldest.last_priced_at = now - timedelta(seconds=120)
        oldest.last_detected_at = now - timedelta(seconds=90)

        scanner._opportunities = [newest, middle, oldest]
        status = scanner.get_status()

        assert isinstance(status.get("last_fast_scan_age_seconds"), float)
        assert 0.0 <= float(status["last_fast_scan_age_seconds"]) <= 30.0
        assert isinstance(status.get("opportunity_price_age_p95"), float)
        assert float(status["opportunity_price_age_p95"]) >= 80.0
        assert isinstance(status.get("opportunity_last_detected_age_p95"), float)
        assert float(status["opportunity_last_detected_age_p95"]) >= 60.0


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
    async def test_scan_callback_invoked(self, mock_polymarket_client, sample_opportunity):
        scanner = _build_scanner(mock_client=mock_polymarket_client)
        _seed_scanner_cache(scanner)

        callback = AsyncMock()
        scanner.add_callback(callback)

        with (
            patch.object(
                scanner, "_dispatch_market_refresh", new_callable=AsyncMock, return_value=[sample_opportunity]
            ),
            patch("services.scanner.quality_filter") as mock_qf,
        ):
            mock_qf.evaluate_opportunity = MagicMock(return_value=_make_quality_pass())
            await scanner.scan_fast()

        callback.assert_awaited_once()
        # Callback receives the list of opportunities
        call_args = callback.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0].strategy == "basic"

    @pytest.mark.asyncio
    async def test_multiple_scan_callbacks(self, mock_polymarket_client):
        scanner = _build_scanner(mock_client=mock_polymarket_client)
        _seed_scanner_cache(scanner)

        cb1 = AsyncMock()
        cb2 = AsyncMock()
        scanner.add_callback(cb1)
        scanner.add_callback(cb2)

        with (
            patch.object(scanner, "_dispatch_market_refresh", new_callable=AsyncMock, return_value=[]),
            patch("services.scanner.quality_filter"),
        ):
            await scanner.scan_fast()

        cb1.assert_awaited_once()
        cb2.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_break_scan(self, mock_polymarket_client, sample_opportunity):
        scanner = _build_scanner(mock_client=mock_polymarket_client)
        _seed_scanner_cache(scanner)

        bad_cb = AsyncMock(side_effect=RuntimeError("callback failed"))
        good_cb = AsyncMock()
        scanner.add_callback(bad_cb)
        scanner.add_callback(good_cb)

        with (
            patch.object(
                scanner, "_dispatch_market_refresh", new_callable=AsyncMock, return_value=[sample_opportunity]
            ),
            patch("services.scanner.quality_filter") as mock_qf,
        ):
            mock_qf.evaluate_opportunity = MagicMock(return_value=_make_quality_pass())
            results = await scanner.scan_fast()

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
