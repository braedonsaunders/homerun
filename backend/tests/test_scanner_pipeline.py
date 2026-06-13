"""Tests for ArbitrageScanner: initialisation, scan pipeline, filtering, lifecycle."""

import asyncio
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import pytest
from datetime import datetime, timedelta
from types import SimpleNamespace
from utils.utcnow import utcnow
from unittest.mock import AsyncMock, MagicMock, patch

from models.market import Event, Market
from models.opportunity import (
    Opportunity,
    MispricingType,
    OpportunityFilter,
)
from services.data_events import EventType


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


class _FakeAsyncSessionFactory:
    def __init__(self):
        self.sessions: list[SimpleNamespace] = []

    def __call__(self):
        factory = self

        class _Context:
            async def __aenter__(self_inner):
                session = SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock())
                factory.sessions.append(session)
                return session

            async def __aexit__(self_inner, exc_type, exc, tb):
                return False

        return _Context()


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

    def test_tail_end_carry_routes_to_full_snapshot_lane(self):
        scanner = _build_scanner()
        tail_end_carry = MagicMock()
        tail_end_carry.slug = "tail_end_carry"
        tail_end_carry.name = "Tail-End Carry"
        tail_end_carry.source_key = "scanner"
        tail_end_carry.strategy_type = "tail_end_carry"
        tail_end_carry.subscriptions = [EventType.MARKET_DATA_REFRESH]
        tail_end_carry.mispricing_type = MispricingType.WITHIN_MARKET.value
        scanner._strategy_overrides = [tail_end_carry]

        incremental, full_snapshot = scanner._partition_market_refresh_strategies()

        assert "tail_end_carry" not in incremental
        assert "tail_end_carry" in full_snapshot

    def test_polymarket_active_filter_requires_condition_and_clob_token(self):
        from services.scanner import ArbitrageScanner

        now = utcnow()
        tradable = Market(
            id="market-tradable",
            condition_id="condition-tradable",
            question="Will this trade?",
            slug="tradable",
            clob_token_ids=["yes-token", "no-token"],
            outcome_prices=[0.52, 0.48],
        )
        missing_tokens = Market(
            id="market-missing-tokens",
            condition_id="condition-missing-tokens",
            question="Missing tokens?",
            slug="missing-tokens",
            clob_token_ids=[],
            outcome_prices=[0.52, 0.48],
        )
        orderbook_disabled = Market(
            id="market-disabled",
            condition_id="condition-disabled",
            question="Disabled order book?",
            slug="disabled",
            clob_token_ids=["yes-token", "no-token"],
            outcome_prices=[0.52, 0.48],
            enable_order_book=False,
        )
        json_token_ids = SimpleNamespace(
            platform="polymarket",
            condition_id="condition-json",
            clob_token_ids='["yes-token","no-token"]',
        )
        single_token_market = SimpleNamespace(
            platform="polymarket",
            condition_id="condition-single",
            clob_token_ids=["outcome-token"],
        )
        empty_json_token_ids = SimpleNamespace(
            platform="polymarket",
            condition_id="condition-empty-json",
            clob_token_ids="[]",
        )

        assert ArbitrageScanner._is_market_active(tradable, now) is True
        assert ArbitrageScanner._is_market_active(json_token_ids, now) is True
        assert ArbitrageScanner._is_market_active(single_token_market, now) is True
        assert ArbitrageScanner._is_market_active(missing_tokens, now) is False
        assert ArbitrageScanner._is_market_active(empty_json_token_ids, now) is False
        assert ArbitrageScanner._is_market_active(orderbook_disabled, now) is False


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
    scanner._prioritizer.get_markets_needing_eval = MagicMock(return_value=list(markets))
    scanner._prioritizer.update_after_evaluation = MagicMock(return_value=0)
    scanner._prioritizer.compute_attention_scores = MagicMock()
    scanner._prioritizer.update_stability_scores = MagicMock()


def _make_quality_pass():
    """Return a mock quality report that always passes."""
    report = MagicMock()
    report.passed = True
    return report


def _make_quality_fail():
    """Return a mock quality report that always fails."""
    report = MagicMock()
    report.passed = False
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
    async def test_refresh_catalog_uses_partial_upstream_results_when_one_core_fetch_times_out(
        self,
        mock_polymarket_client,
    ):
        market = Market(
            id="market-timeout-fallback",
            condition_id="condition-timeout-fallback",
            question="Will fallback market stay available?",
            slug="timeout-fallback",
            clob_token_ids=["fallback-yes-token", "fallback-no-token"],
            outcome_prices=[0.52, 0.48],
            # Required by MARKET_UNIVERSE_TRADABLE_ONLY tradable filter
            # applied at the catalog-fetch boundary.
            accepting_orders=True,
            volume=100.0,
        )
        mock_polymarket_client.get_all_events.side_effect = asyncio.TimeoutError()
        mock_polymarket_client.get_all_markets.return_value = [market]

        scanner = _build_scanner(mock_client=mock_polymarket_client, strategies=[])

        with (
            patch.object(scanner, "_ensure_runtime_strategies_loaded", new_callable=AsyncMock),
            patch.object(scanner, "_set_activity", new_callable=AsyncMock),
            patch("services.scanner.settings.WS_FEED_ENABLED", False),
            patch("services.scanner.settings.NEWS_EDGE_ENABLED", False),
        ):
            market_count = await scanner.refresh_catalog()

        assert market_count == 1
        assert len(scanner._cached_markets) == 1
        assert scanner._cached_markets[0].id == "market-timeout-fallback"


@pytest.mark.asyncio
async def test_enqueue_detection_batch_publishes_runtime_signals_when_snapshot_persist_fails(monkeypatch):
    from workers import scanner_worker

    class _Session:
        pass

    class _SessionContext:
        async def __aenter__(self):
            return _Session()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    opportunity = SimpleNamespace(
        stable_id="scanner-opp-1",
        id="scanner-opp-1",
        strategy="basic",
        markets=[],
    )
    bridge_mock = AsyncMock(return_value=1)
    snapshot_mock = AsyncMock(side_effect=TimeoutError("snapshot timeout"))
    clear_mock = AsyncMock(return_value=None)

    monkeypatch.setattr(scanner_worker, "AsyncSessionLocal", lambda: _SessionContext())
    monkeypatch.setattr(scanner_worker, "bridge_opportunities_to_signals", bridge_mock)
    monkeypatch.setattr(scanner_worker, "write_scanner_snapshot", snapshot_mock)
    monkeypatch.setattr(scanner_worker, "clear_scan_request", clear_mock)
    monkeypatch.setattr(
        scanner_worker.quality_filter,
        "evaluate_opportunity",
        lambda opportunity, overrides=None: {"passed": True, "overrides": overrides, "id": opportunity.stable_id},
    )
    monkeypatch.setattr(scanner_worker.strategy_loader, "get_instance", lambda _strategy: None)

    batch_id, pending, dropped = await scanner_worker._enqueue_detection_batch(
        [opportunity],
        {"running": True},
        batch_kind="startup",
        quality_reports={},
    )

    assert batch_id
    assert pending == 0
    assert dropped == 0
    bridge_mock.assert_awaited_once()
    snapshot_mock.assert_awaited_once()
    clear_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_refresh_catalog_backfills_flat_market_event_context_from_event_payload(self, mock_polymarket_client):
        flat_market = Market(
            id="market-antigua",
            condition_id="condition-antigua",
            question="Will Antigua GFC win on 2026-04-01?",
            slug="gtm-ant-mic-2026-04-01-ant",
            event_slug="",
            neg_risk=False,
        )
        event_market = Market(
            id="market-antigua",
            condition_id="condition-antigua",
            question="Will Antigua GFC win on 2026-04-01?",
            slug="gtm-ant-mic-2026-04-01-ant",
            event_slug="gtm-ant-mic-2026-04-01",
            group_item_title="Antigua GFC",
            sports_market_type="moneyline",
            neg_risk=True,
        )
        event = Event(
            id="event-antigua",
            slug="gtm-ant-mic-2026-04-01",
            title="Antigua GFC vs. CSD Mictlán",
            category="Soccer",
            markets=[event_market],
            neg_risk=True,
        )
        mock_polymarket_client.get_all_events.return_value = [event]
        mock_polymarket_client.get_all_markets.return_value = [flat_market]

        scanner = _build_scanner(mock_client=mock_polymarket_client)
        with (
            patch.object(scanner, "_ensure_runtime_strategies_loaded", new_callable=AsyncMock),
            patch.object(scanner, "_set_activity", new_callable=AsyncMock),
            patch("services.scanner.settings.WS_FEED_ENABLED", False),
            patch("services.scanner.settings.NEWS_EDGE_ENABLED", False),
        ):
            await scanner.refresh_catalog()

        assert len(scanner._cached_markets) == 1
        refreshed_market = scanner._cached_markets[0]
        assert refreshed_market.event_slug == "gtm-ant-mic-2026-04-01"
        assert refreshed_market.group_item_title == "Antigua GFC"
        assert refreshed_market.sports_market_type == "moneyline"
        assert refreshed_market.neg_risk is True

    @pytest.mark.asyncio
    async def test_refresh_catalog_incremental_refetches_touched_cached_event_and_restores_sibling_markets(
        self,
        mock_polymarket_client,
    ):
        cached_antigua = Market(
            id="market-antigua",
            condition_id="condition-antigua",
            question="Will Antigua GFC win on 2026-04-01?",
            slug="gtm-ant-mic-2026-04-01-ant",
            event_slug="gtm-ant-mic-2026-04-01",
            group_item_title="Antigua GFC",
            sports_market_type="moneyline",
            neg_risk=True,
        )
        cached_mictlan = Market(
            id="market-mictlan",
            condition_id="condition-mictlan",
            question="Will CSD Mictlán win on 2026-04-01?",
            slug="gtm-ant-mic-2026-04-01-mic",
            event_slug="gtm-ant-mic-2026-04-01",
            group_item_title="CSD Mictlán",
            sports_market_type="moneyline",
            neg_risk=True,
        )
        cached_event = Event(
            id="event-antigua",
            slug="gtm-ant-mic-2026-04-01",
            title="Antigua GFC vs. CSD Mictlán",
            category="Soccer",
            markets=[cached_antigua, cached_mictlan],
            neg_risk=True,
        )
        draw_market = Market(
            id="market-draw",
            condition_id="condition-draw",
            question="Will Antigua GFC vs. CSD Mictlán end in a draw?",
            slug="gtm-ant-mic-2026-04-01-draw",
            event_slug="gtm-ant-mic-2026-04-01",
            group_item_title="Draw (Antigua GFC vs. CSD Mictlán)",
            sports_market_type="moneyline",
            neg_risk=True,
        )
        fetched_event = Event(
            id="event-antigua",
            slug="gtm-ant-mic-2026-04-01",
            title="Antigua GFC vs. CSD Mictlán",
            category="Soccer",
            markets=[cached_antigua.model_copy(deep=True), draw_market, cached_mictlan.model_copy(deep=True)],
            neg_risk=True,
        )
        mock_polymarket_client.get_recent_markets.return_value = [cached_antigua.model_copy(deep=True)]
        mock_polymarket_client.get_events_by_slugs = AsyncMock(return_value=[fetched_event])

        scanner = _build_scanner(mock_client=mock_polymarket_client)
        scanner._cached_markets = [cached_antigua, cached_mictlan]
        scanner._cached_events = [cached_event]

        with (
            patch.object(scanner, "_ensure_runtime_strategies_loaded", new_callable=AsyncMock),
            patch.object(scanner, "_set_activity", new_callable=AsyncMock),
            patch("services.scanner.settings.WS_FEED_ENABLED", False),
            patch("services.scanner.settings.NEWS_EDGE_ENABLED", False),
        ):
            await scanner.refresh_catalog_incremental()

        mock_polymarket_client.get_events_by_slugs.assert_awaited_once_with(
            ["gtm-ant-mic-2026-04-01"],
            closed=False,
        )
        assert {market.slug for market in scanner._cached_markets} == {
            "gtm-ant-mic-2026-04-01-ant",
            "gtm-ant-mic-2026-04-01-draw",
            "gtm-ant-mic-2026-04-01-mic",
        }
        refreshed_event = next(event for event in scanner._cached_events if event.slug == "gtm-ant-mic-2026-04-01")
        assert {market.slug for market in refreshed_event.markets} == {
            "gtm-ant-mic-2026-04-01-ant",
            "gtm-ant-mic-2026-04-01-draw",
            "gtm-ant-mic-2026-04-01-mic",
        }

    @pytest.mark.asyncio
    async def test_refresh_catalog_incremental_skips_single_market_cached_event_refetch(
        self,
        mock_polymarket_client,
    ):
        cached_market = Market(
            id="market-single",
            condition_id="condition-single",
            question="Will Example happen?",
            slug="example-single",
            event_slug="example-single-event",
            neg_risk=False,
        )
        cached_event = Event(
            id="event-single",
            slug="example-single-event",
            title="Example single event",
            category="Politics",
            markets=[cached_market],
            neg_risk=False,
        )
        mock_polymarket_client.get_recent_markets.return_value = [cached_market.model_copy(deep=True)]
        mock_polymarket_client.get_events_by_slugs = AsyncMock(return_value=[])

        scanner = _build_scanner(mock_client=mock_polymarket_client)
        scanner._cached_markets = [cached_market]
        scanner._cached_events = [cached_event]
        scanner._rebuild_realtime_graph(scanner._cached_events, scanner._cached_markets)

        with (
            patch.object(scanner, "_ensure_runtime_strategies_loaded", new_callable=AsyncMock),
            patch.object(scanner, "_set_activity", new_callable=AsyncMock),
            patch("services.scanner.settings.WS_FEED_ENABLED", False),
            patch("services.scanner.settings.NEWS_EDGE_ENABLED", False),
        ):
            await scanner.refresh_catalog_incremental()

        mock_polymarket_client.get_events_by_slugs.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_refresh_catalog_incremental_keeps_partial_event_progress_when_late_batch_times_out(
        self,
        mock_polymarket_client,
    ):
        delta_markets = [
            Market(
                id=f"market-{idx}",
                condition_id=f"condition-{idx}",
                question=f"Will example {idx} happen?",
                slug=f"example-{idx}",
                event_slug=f"event-{idx}",
                neg_risk=False,
            )
            for idx in range(30)
        ]
        first_batch_events = [
            Event(
                id=f"event-{idx}",
                slug=f"event-{idx}",
                title=f"Event {idx}",
                category="Politics",
                markets=[delta_markets[idx].model_copy(deep=True)],
                neg_risk=False,
            )
            for idx in range(25)
        ]

        mock_polymarket_client.get_recent_markets.return_value = [market.model_copy(deep=True) for market in delta_markets]
        mock_polymarket_client.get_events_by_slugs = AsyncMock(
            side_effect=[first_batch_events, asyncio.TimeoutError()]
        )

        scanner = _build_scanner(mock_client=mock_polymarket_client)
        scanner._cached_markets = [delta_markets[0].model_copy(deep=True)]
        scanner._cached_events = []

        with (
            patch.object(scanner, "_ensure_runtime_strategies_loaded", new_callable=AsyncMock),
            patch.object(scanner, "_set_activity", new_callable=AsyncMock),
            patch("services.scanner.settings.WS_FEED_ENABLED", False),
            patch("services.scanner.settings.NEWS_EDGE_ENABLED", False),
        ):
            await scanner.refresh_catalog_incremental()

        assert mock_polymarket_client.get_events_by_slugs.await_count == 2
        assert len(scanner._cached_markets) == 30
        assert {event.slug for event in scanner._cached_events} == {f"event-{idx}" for idx in range(25)}

    def test_enforce_catalog_caps_keeps_event_rosters_atomic(self, mock_polymarket_client):
        scanner = _build_scanner(mock_client=mock_polymarket_client)

        home = Market(
            id="market-home",
            condition_id="condition-home",
            question="Will home team win?",
            slug="event-home",
            event_slug="event-atomic",
            group_item_title="Home",
            sports_market_type="moneyline",
            neg_risk=True,
        )
        draw = Market(
            id="market-draw",
            condition_id="condition-draw",
            question="Will the match end in a draw?",
            slug="event-draw",
            event_slug="event-atomic",
            group_item_title="Draw",
            sports_market_type="moneyline",
            neg_risk=True,
        )
        away = Market(
            id="market-away",
            condition_id="condition-away",
            question="Will away team win?",
            slug="event-away",
            event_slug="event-atomic",
            group_item_title="Away",
            sports_market_type="moneyline",
            neg_risk=True,
        )
        event = Event(
            id="event-atomic",
            slug="event-atomic",
            title="Atomic roster event",
            category="Sports",
            markets=[home, draw, away],
            neg_risk=True,
        )

        with (
            patch("services.scanner.settings.SCANNER_FORCE_FULL_UNIVERSE", False),
            patch("services.scanner.settings.MAX_MARKETS_TO_SCAN", 2),
            patch("services.scanner.settings.MAX_EVENTS_TO_SCAN", 0),
        ):
            capped_events, capped_markets = scanner._enforce_catalog_caps([event], [home, draw, away])

        assert len(capped_events) == 1
        assert len(capped_markets) == 3
        assert {market.id for market in capped_events[0].markets} == {"market-home", "market-draw", "market-away"}

    @pytest.mark.asyncio
    async def test_scan_fast_dispatches_verified_event_peers_for_partial_hot_batch(self, mock_polymarket_client):
        scanner = _build_scanner(mock_client=mock_polymarket_client)

        home = Market(
            id="market-home",
            condition_id="condition-home",
            question="Will home team win?",
            slug="event-home",
            event_slug="event-atomic",
            group_item_title="Home",
            sports_market_type="moneyline",
            neg_risk=True,
            clob_token_ids=["tok-home-yes", "tok-home-no"],
            outcome_prices=[0.4, 0.6],
        )
        draw = Market(
            id="market-draw",
            condition_id="condition-draw",
            question="Will the match end in a draw?",
            slug="event-draw",
            event_slug="event-atomic",
            group_item_title="Draw",
            sports_market_type="moneyline",
            neg_risk=True,
            clob_token_ids=["tok-draw-yes", "tok-draw-no"],
            outcome_prices=[0.2, 0.8],
        )
        away = Market(
            id="market-away",
            condition_id="condition-away",
            question="Will away team win?",
            slug="event-away",
            event_slug="event-atomic",
            group_item_title="Away",
            sports_market_type="moneyline",
            neg_risk=True,
            clob_token_ids=["tok-away-yes", "tok-away-no"],
            outcome_prices=[0.4, 0.6],
        )
        event = Event(
            id="event-atomic",
            slug="event-atomic",
            title="Atomic roster event",
            category="Sports",
            markets=[home, draw, away],
            neg_risk=True,
        )

        scanner._cached_markets = [home, draw, away]
        scanner._cached_events = [event]
        scanner._cached_market_by_id = {market.id: market for market in scanner._cached_markets}
        scanner._rebuild_realtime_graph(scanner._cached_events, scanner._cached_markets)

        from services.market_prioritizer import MarketTier

        scanner._prioritizer = MagicMock()
        scanner._prioritizer.classify_all = MagicMock(
            return_value={
                MarketTier.HOT: [home],
                MarketTier.WARM: [],
                MarketTier.COLD: [],
            }
        )
        scanner._prioritizer.get_markets_needing_eval = MagicMock(return_value=[home])
        scanner._prioritizer.update_after_evaluation = MagicMock(return_value=0)
        scanner._prioritizer.compute_attention_scores = MagicMock()
        scanner._prioritizer.update_stability_scores = MagicMock()

        dispatch_mock = AsyncMock(return_value=[])
        with (
            patch.object(scanner, "_dispatch_market_refresh", dispatch_mock),
            patch.object(scanner, "_partition_market_refresh_strategies", return_value=({"settlement_lag"}, set())),
            patch.object(scanner, "_ensure_runtime_strategies_loaded", new_callable=AsyncMock),
            patch.object(scanner, "_set_activity", new_callable=AsyncMock),
            patch.object(scanner, "refresh_opportunity_prices", new_callable=AsyncMock, return_value=[]),
            patch.object(scanner, "_snapshot_ws_prices", new_callable=AsyncMock, return_value={}),
            patch("services.scanner.settings.INCREMENTAL_FETCH_ENABLED", False),
        ):
            await scanner.scan_fast()

        dispatched_event = dispatch_mock.await_args.args[0]
        assert {market.id for market in dispatched_event.markets} == {"market-home", "market-draw", "market-away"}

    def test_build_dispatch_market_groups_skips_unverified_event_markets(self, mock_polymarket_client):
        scanner = _build_scanner(mock_client=mock_polymarket_client)

        unverified_market = Market(
            id="market-unverified",
            condition_id="condition-unverified",
            question="Will team win?",
            slug="event-unverified-home",
            event_slug="event-unverified",
            group_item_title="Home",
            sports_market_type="moneyline",
            neg_risk=True,
        )

        scanner._cached_markets = [unverified_market]
        scanner._cached_events = []
        scanner._cached_market_by_id = {unverified_market.id: unverified_market}
        scanner._rebuild_realtime_graph([], [unverified_market])

        assert scanner._build_dispatch_market_groups([unverified_market]) == []

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
    async def test_scan_fast_publishes_only_actionable_opportunities(
        self,
        mock_polymarket_client,
        sample_opportunity,
    ):
        scanner = _build_scanner(mock_client=mock_polymarket_client)
        _seed_scanner_cache(scanner)
        rejected = sample_opportunity.model_copy(deep=True)
        rejected.stable_id = f"{sample_opportunity.stable_id}_rejected"
        rejected.id = f"{sample_opportunity.id}_rejected"
        rejected.title = "Rejected"
        callback = AsyncMock()
        scanner.add_callback(callback)
        reports = {
            sample_opportunity.stable_id: _make_quality_pass(),
            rejected.stable_id: _make_quality_fail(),
        }

        with (
            patch.object(
                scanner,
                "_dispatch_market_refresh",
                new_callable=AsyncMock,
                return_value=[rejected, sample_opportunity],
            ),
            patch("services.scanner.quality_filter") as mock_qf,
        ):
            mock_qf.evaluate_opportunity = MagicMock(
                side_effect=lambda opp, overrides=None: reports[opp.stable_id]
            )
            results = await scanner.scan_fast()

        assert [opp.stable_id for opp in results] == [sample_opportunity.stable_id]
        callback.assert_awaited_once()
        assert [opp.stable_id for opp in callback.call_args[0][0]] == [sample_opportunity.stable_id]
        assert set(scanner.quality_reports.keys()) == {sample_opportunity.stable_id}

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
    async def test_scan_fast_falls_back_to_warm_tier_when_hot_tier_empty(self, mock_polymarket_client):
        scanner = _build_scanner(mock_client=mock_polymarket_client)
        warm_markets = [
            Market(
                id="m_warm_1",
                condition_id="c_warm_1",
                question="Warm market one?",
                slug="warm-market-one",
                clob_token_ids=["warm_yes_1", "warm_no_1"],
                outcome_prices=[0.91, 0.09],
                volume=150000.0,
                liquidity=12000.0,
            ),
            Market(
                id="m_warm_2",
                condition_id="c_warm_2",
                question="Warm market two?",
                slug="warm-market-two",
                clob_token_ids=["warm_yes_2", "warm_no_2"],
                outcome_prices=[0.89, 0.11],
                volume=125000.0,
                liquidity=9000.0,
            ),
        ]
        _seed_scanner_cache(scanner, markets=warm_markets)

        from services.market_prioritizer import MarketTier

        scanner._prioritizer.classify_all = MagicMock(
            return_value={
                MarketTier.HOT: [],
                MarketTier.WARM: list(warm_markets),
                MarketTier.COLD: [],
            }
        )
        scanner._prioritizer.get_markets_needing_eval = MagicMock(return_value=list(warm_markets))

        with (
            patch.object(scanner, "_ensure_runtime_strategies_loaded", new_callable=AsyncMock),
            patch.object(scanner, "_dispatch_market_refresh", new_callable=AsyncMock, return_value=[]) as dispatch_mock,
            patch("services.scanner.quality_filter"),
        ):
            await scanner.scan_fast()

        dispatched_event = dispatch_mock.await_args.args[0]
        assert dispatched_event.scan_mode == "fast_timer"
        assert [market.id for market in dispatched_event.markets] == ["m_warm_1", "m_warm_2"]
        assert dispatched_event.payload["affected_market_count"] == 2

    @pytest.mark.asyncio
    async def test_scan_fast_keeps_hot_tier_markets_eligible_even_when_fingerprints_are_unchanged(
        self,
        mock_polymarket_client,
    ):
        scanner = _build_scanner(mock_client=mock_polymarket_client)
        market = Market(
            id="m_hot_eval_1",
            condition_id="c_hot_eval_1",
            question="Hot tier market?",
            slug="hot-tier-market",
            clob_token_ids=["hot_yes_eval_1", "hot_no_eval_1"],
            outcome_prices=[0.9, 0.1],
            volume=250000.0,
            liquidity=14000.0,
        )
        _seed_scanner_cache(scanner, markets=[market])
        scanner._prioritizer.get_markets_needing_eval = MagicMock(return_value=[market])

        with (
            patch.object(scanner, "_ensure_runtime_strategies_loaded", new_callable=AsyncMock),
            patch.object(scanner, "_dispatch_market_refresh", new_callable=AsyncMock, return_value=[]) as dispatch_mock,
            patch("services.scanner.quality_filter"),
        ):
            await scanner.scan_fast()

        scanner._prioritizer.get_markets_needing_eval.assert_called_once()
        dispatched_event = dispatch_mock.await_args.args[0]
        assert [candidate.id for candidate in dispatched_event.markets] == ["m_hot_eval_1"]

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
    async def test_scan_fast_drops_stale_opportunities_on_runtime_refresh(self, mock_polymarket_client):
        scanner = _build_scanner(mock_client=mock_polymarket_client)
        market = Market(
            id="m_runtime_refresh",
            condition_id="c_runtime_refresh",
            question="Will runtime refresh stay fresh?",
            slug="runtime-refresh",
            clob_token_ids=["tok_runtime_yes", "tok_runtime_no"],
            outcome_prices=[0.48, 0.52],
        )
        _seed_scanner_cache(scanner, markets=[market])
        scanner._prioritizer.get_markets_needing_eval = MagicMock(return_value=[])
        refresh_mock = AsyncMock(return_value=[])

        with (
            patch.object(scanner, "_ensure_runtime_strategies_loaded", new_callable=AsyncMock),
            patch.object(scanner, "_set_activity", new_callable=AsyncMock),
            patch.object(scanner, "refresh_opportunity_prices", refresh_mock),
            patch.object(scanner, "_snapshot_ws_prices", new_callable=AsyncMock, return_value={}),
        ):
            await scanner.scan_fast()

        assert refresh_mock.await_count == 1
        assert refresh_mock.await_args.kwargs["drop_stale"] is True

    @pytest.mark.asyncio
    async def test_full_snapshot_drops_stale_opportunities_on_runtime_refresh(self, mock_polymarket_client):
        scanner = _build_scanner(mock_client=mock_polymarket_client)
        market = Market(
            id="m_full_refresh",
            condition_id="c_full_refresh",
            question="Will full snapshot refresh stay fresh?",
            slug="full-refresh",
            clob_token_ids=["tok_full_yes", "tok_full_no"],
            outcome_prices=[0.49, 0.51],
        )
        scanner._cached_markets = [market]
        scanner._cached_market_by_id = {market.id: market}
        scanner._cached_events = []
        refresh_mock = AsyncMock(return_value=[])

        with (
            patch.object(scanner, "_ensure_runtime_strategies_loaded", new_callable=AsyncMock),
            patch.object(scanner, "_partition_market_refresh_strategies", return_value=(set(), {"tail_end_carry"})),
            patch.object(scanner, "_dispatch_market_refresh", new_callable=AsyncMock, return_value=[]),
            patch.object(scanner, "_set_activity", new_callable=AsyncMock),
            patch.object(scanner, "refresh_opportunity_prices", refresh_mock),
            patch.object(scanner, "_snapshot_ws_prices", new_callable=AsyncMock, return_value={}),
            patch("services.scanner.settings.SCANNER_FORCE_FULL_UNIVERSE", True),
            patch("services.scanner.settings.SCANNER_FULL_SNAPSHOT_CHUNK_SIZE", 300),
        ):
            await scanner.scan_full_snapshot_strategies(force=True)

        assert refresh_mock.await_count == 1
        assert refresh_mock.await_args.kwargs["drop_stale"] is True

    @pytest.mark.asyncio
    async def test_refresh_opportunity_prices_preserves_multi_outcome_prices(self, mock_polymarket_client):
        scanner = _build_scanner(mock_client=mock_polymarket_client)
        opp = Opportunity(
            strategy="basic",
            title="Multi outcome refresh",
            description="D",
            total_cost=0.95,
            gross_profit=0.05,
            fee=0.02,
            net_profit=0.03,
            roi_percent=3.16,
            markets=[
                {
                    "id": "m_multi_refresh",
                    "question": "Test?",
                    "clob_token_ids": ["tok_a", "tok_b", "tok_c", "tok_d"],
                    "outcome_prices": [0.22, 0.28, 0.31, 0.19],
                    "yes_price": 0.22,
                    "no_price": 0.28,
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
                "tok_a": {"mid": 0.25, "ts": ts_seconds},
                "tok_b": {"mid": 0.30, "ts": ts_seconds},
                "tok_c": {"mid": 0.27, "ts": ts_seconds},
                "tok_d": {"mid": 0.18, "ts": ts_seconds},
            },
        ):
            refreshed = await scanner.refresh_opportunity_prices([opp], now=ts)

        assert len(refreshed) == 1
        market = refreshed[0].markets[0]
        assert market["yes_price"] == pytest.approx(0.25)
        assert market["no_price"] == pytest.approx(0.30)
        assert market["outcome_prices"] == pytest.approx([0.25, 0.30, 0.27, 0.18])

    @pytest.mark.asyncio
    async def test_refresh_opportunity_prices_keeps_full_snapshot_opportunity_until_cycle_can_revisit(
        self, mock_polymarket_client
    ):
        scanner = _build_scanner(mock_client=mock_polymarket_client)
        seen_at = utcnow() - timedelta(minutes=4)
        stale_price_at = utcnow() - timedelta(minutes=3)
        opp = Opportunity(
            strategy="tail_end_carry",
            title="Heavy lane stale retention",
            description="D",
            total_cost=0.91,
            gross_profit=0.09,
            fee=0.01,
            net_profit=0.08,
            roi_percent=8.79,
            markets=[
                {
                    "id": "m_heavy_retention",
                    "question": "Test?",
                    "clob_token_ids": ["tok_yes", "tok_no"],
                    "yes_price": 0.09,
                    "no_price": 0.91,
                    "price_updated_at": stale_price_at.replace(tzinfo=None).isoformat() + "Z",
                }
            ],
            positions_to_take=[],
            detected_at=seen_at,
            last_detected_at=seen_at,
            last_seen_at=seen_at,
        )
        scanner._full_snapshot_cycle_total_markets = 47000
        scanner._full_snapshot_cycle_processed_markets = 4700
        scanner._full_snapshot_cycle_started_at = utcnow() - timedelta(minutes=2)
        scanner._full_snapshot_cycle_completed_at = None

        with (
            patch.object(scanner, "_snapshot_ws_prices", new_callable=AsyncMock, return_value={}),
            patch("services.scanner.settings.SCANNER_MARKET_PRICE_MAX_AGE_SECONDS", 60),
        ):
            refreshed = await scanner.refresh_opportunity_prices([opp], now=utcnow(), drop_stale=True)

        assert refreshed == [opp]

    def test_opportunity_last_detected_retention_expands_for_large_full_snapshot_cycle(self):
        scanner = _build_scanner()
        scanner._full_snapshot_cycle_total_markets = 47000
        scanner._full_snapshot_cycle_processed_markets = 4700
        scanner._full_snapshot_cycle_started_at = utcnow() - timedelta(minutes=2)
        scanner._full_snapshot_cycle_completed_at = None

        retention_seconds = scanner._opportunity_last_detected_retention_seconds(utcnow())

        assert retention_seconds >= 300.0
        assert retention_seconds > 180.0

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
    async def test_full_snapshot_scan_dispatches_incremental_strategies_across_full_universe(
        self,
        mock_polymarket_client,
    ):
        scanner = _build_scanner(mock_client=mock_polymarket_client)
        market = Market(
            id="m_full_dispatch_1",
            condition_id="c_full_dispatch_1",
            question="Will the heavy lane include incremental strategies?",
            slug="full-dispatch-1",
            clob_token_ids=["tok_full_dispatch_yes", "tok_full_dispatch_no"],
            outcome_prices=[0.47, 0.53],
        )
        scanner._cached_markets = [market]
        scanner._cached_market_by_id = {market.id: market}
        scanner._cached_events = []

        with (
            patch.object(scanner, "_ensure_runtime_strategies_loaded", new_callable=AsyncMock),
            patch.object(
                scanner,
                "_partition_market_refresh_strategies",
                return_value=({"basic"}, {"tail_end_carry"}),
            ),
            patch.object(scanner, "_snapshot_ws_prices", new_callable=AsyncMock, return_value={}),
            patch.object(scanner, "_dispatch_market_refresh", new_callable=AsyncMock, return_value=[]) as dispatch_mock,
            patch.object(scanner, "_set_activity", new_callable=AsyncMock),
            patch("services.scanner.settings.SCANNER_FULL_SNAPSHOT_CHUNK_SIZE", 100),
            patch("services.scanner.settings.SCANNER_FORCE_FULL_UNIVERSE", True),
        ):
            await scanner.scan_full_snapshot_strategies(force=True)

        dispatched_kwargs = dispatch_mock.await_args.kwargs
        assert dispatched_kwargs["incremental_slugs"] == set()
        assert dispatched_kwargs["full_slugs"] == {"basic", "tail_end_carry"}

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
            patch.object(scanner, "_dispatch_market_refresh", new_callable=AsyncMock, return_value=[]) as dispatch_mock,
            patch.object(scanner, "_set_activity", new_callable=AsyncMock),
            patch("services.scanner.settings.SCANNER_FULL_SNAPSHOT_CHUNK_SIZE", 2),
            patch("services.scanner.settings.SCANNER_FORCE_FULL_UNIVERSE", True),
        ):
            await scanner.scan_full_snapshot_strategies(force=True)
            assert dispatch_mock.await_count == 2
            assert scanner._last_full_snapshot_chunk_market_count == 1
            assert scanner._full_snapshot_cycle_total_markets == 3
            assert scanner._full_snapshot_cycle_processed_markets == 3
            assert scanner._full_snapshot_cursor_index == 0
            assert scanner._full_snapshot_cycle_completed_at is not None

        assert scanner._full_snapshot_cycle_total_markets == 3
        assert scanner._full_snapshot_cycle_processed_markets == 3
        assert scanner._full_snapshot_cursor_index == 0
        assert scanner._full_snapshot_cycle_completed_at is not None

    @pytest.mark.asyncio
    async def test_full_snapshot_scan_honors_force_full_universe_without_hidden_cap(
        self,
        mock_polymarket_client,
    ):
        scanner = _build_scanner(mock_client=mock_polymarket_client)
        markets = []
        for idx in range(15005):
            yes = f"tok_yes_force_{idx}_" + ("0" * 20)
            no = f"tok_no_force_{idx}_" + ("0" * 20)
            markets.append(
                Market(
                    id=f"m_force_{idx}",
                    condition_id=f"c_force_{idx}",
                    question=f"Force market {idx}?",
                    slug=f"force-market-{idx}",
                    clob_token_ids=[yes, no],
                    outcome_prices=[0.49, 0.51],
                )
            )
        scanner._cached_markets = list(markets)
        scanner._cached_market_by_id = {m.id: m for m in markets}
        scanner._cached_events = []

        with (
            patch.object(scanner, "_ensure_runtime_strategies_loaded", new_callable=AsyncMock),
            patch.object(scanner, "_partition_market_refresh_strategies", return_value=(set(), {"full_only"})),
            patch.object(scanner, "_snapshot_ws_prices", new_callable=AsyncMock, return_value={}),
            patch.object(scanner, "_dispatch_market_refresh", new_callable=AsyncMock, return_value=[]) as dispatch_mock,
            patch.object(scanner, "_set_activity", new_callable=AsyncMock),
            patch("services.scanner.settings.SCANNER_FULL_SNAPSHOT_CHUNK_SIZE", 20000),
            patch("services.scanner.settings.SCANNER_FORCE_FULL_UNIVERSE", True),
            patch("services.scanner.settings.SCANNER_FULL_SNAPSHOT_MAX_MARKETS", 0),
        ):
            await scanner.scan_full_snapshot_strategies(force=True)

        assert dispatch_mock.await_count == 1
        assert scanner._last_full_snapshot_strategy_market_count == 15005
        assert scanner._last_full_snapshot_chunk_market_count == 15005
        assert scanner._full_snapshot_cycle_total_markets == 15005
        assert scanner._full_snapshot_cycle_processed_markets == 15005
        assert scanner._full_snapshot_cursor_index == 0
        assert scanner._full_snapshot_cycle_completed_at is not None

    @pytest.mark.asyncio
    async def test_refresh_catalog_reads_prices_from_ws_cache(self, mock_polymarket_client):
        """refresh_catalog reads prices via _snapshot_ws_prices, not get_prices_batch."""
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
        third_token = "123456789012345678903"
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
                    "clob_token_ids": f'["{yes_token}", "{no_token}", "{third_token}"]',
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
        assert scanner._market_outcome_token_ids.get("m_weather_1") == (yes_token, no_token, third_token)

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
    async def test_attach_price_history_to_markets_uses_shared_backfill(self):
        """The flat-markets entry point reuses the same backfill plumbing."""
        scanner = _build_scanner(strategies=[])
        yes_token = "123456789012345678901"
        no_token = "123456789012345678902"
        markets = [
            {
                "id": "m_crypto_1",
                "platform": "polymarket",
                "clob_token_ids": [yes_token, no_token],
                "yes_price": 0.45,
                "no_price": 0.55,
            }
        ]

        scanner._backfill_market_history_for_opportunities = AsyncMock(return_value=None)
        scanner.get_market_history_for_opportunities = MagicMock(
            return_value={
                "m_crypto_1": [
                    {"t": 1.0, "yes": 0.41, "no": 0.59},
                    {"t": 2.0, "yes": 0.43, "no": 0.57},
                ]
            }
        )

        attached = await scanner.attach_price_history_to_markets(markets, timeout_seconds=None)

        assert attached == 1
        assert markets[0]["price_history"][0]["yes"] == 0.41
        assert len(markets[0]["price_history"]) == 2

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

    @pytest.mark.asyncio
    async def test_attach_price_history_nonblocking_skips_db_hydration(self):
        scanner = _build_scanner(strategies=[])
        scanner._hydrate_history_from_db = AsyncMock(return_value=0)
        scanner._backfill_market_history_for_opportunities = AsyncMock(return_value=None)
        scanner._persist_market_history_for_opportunities = AsyncMock(return_value=None)

        opp = Opportunity(
            strategy="crypto_edge",
            title="Crypto",
            description="D",
            total_cost=0.2,
            expected_payout=0.5,
            gross_profit=0.3,
            fee=0.01,
            net_profit=0.29,
            roi_percent=145.0,
            markets=[
                {
                    "id": "m_crypto_hot",
                    "condition_id": "m_crypto_hot",
                    "platform": "polymarket",
                    "yes_price": 0.2,
                    "no_price": 0.8,
                }
            ],
            min_liquidity=1000.0,
            max_position_size=10.0,
            positions_to_take=[],
        )

        attached = await scanner.attach_price_history_to_opportunities([opp], timeout_seconds=0.0)

        assert attached == 0
        scanner._hydrate_history_from_db.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_persist_market_history_for_opportunities_writes_only_changed_rows(self, monkeypatch):
        import services.scanner as scanner_module
        import services.shared_state as shared_state_module

        scanner = _build_scanner(strategies=[])
        scanner._market_price_history = {
            "m_same": [
                {"t": 1.0, "yes": 0.41, "no": 0.59},
                {"t": 2.0, "yes": 0.43, "no": 0.57},
            ],
            "m_new": [
                {"t": 1.0, "yes": 0.21, "no": 0.79},
                {"t": 2.0, "yes": 0.24, "no": 0.76},
            ],
        }
        scanner._persisted_market_history_signatures = {
            "m_same": scanner._market_history_signature(scanner._market_price_history["m_same"]),
        }
        upsert_mock = AsyncMock(return_value=1)
        session_factory = _FakeAsyncSessionFactory()
        monkeypatch.setattr(scanner_module, "is_db_pressure_active", lambda: False)
        monkeypatch.setattr(shared_state_module, "upsert_scanner_market_history", upsert_mock)
        monkeypatch.setattr(scanner_module, "AsyncSessionLocal", session_factory)

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
                {"id": "m_same", "platform": "polymarket", "yes_price": 0.2, "no_price": 0.8},
                {"id": "m_new", "platform": "polymarket", "yes_price": 0.2, "no_price": 0.8},
            ],
            min_liquidity=1000.0,
            max_position_size=10.0,
            positions_to_take=[],
        )

        await scanner._persist_market_history_for_opportunities([opp])

        upsert_mock.assert_awaited_once()
        written_history = upsert_mock.await_args.args[1]
        assert set(written_history.keys()) == {"m_new"}
        assert len(session_factory.sessions) == 1
        assert session_factory.sessions[0].commit.await_count == 1
        assert scanner._persisted_market_history_signatures["m_new"] == scanner._market_history_signature(
            scanner._market_price_history["m_new"]
        )

    def test_merge_market_history_points_keeps_multi_outcome_vectors(self):
        scanner = _build_scanner(strategies=[])
        now_ms = int(utcnow().timestamp() * 1000)
        incoming = [
            {
                "t": float(now_ms - 120_000),
                "idx_0": 0.22,
                "idx_1": 0.31,
                "idx_2": 0.47,
            },
            {
                "t": float(now_ms - 60_000),
                "outcome_prices": [0.24, 0.30, 0.46],
            },
        ]

        merged = scanner._merge_market_history_points("m_multi_hist", incoming, now_ms)

        assert merged == 2
        history = scanner._market_price_history.get("m_multi_hist") or []
        assert len(history) == 2
        assert history[-1].get("yes") == pytest.approx(0.24)
        assert history[-1].get("no") == pytest.approx(0.30)
        assert history[-1].get("idx_2") == pytest.approx(0.46)
        assert history[-1].get("outcome_prices") == pytest.approx([0.24, 0.30, 0.46])


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
    async def test_full_snapshot_scan_callback_invoked(self, mock_polymarket_client):
        scanner = _build_scanner(mock_client=mock_polymarket_client)
        market = Market(
            id="m_full_callback_1",
            condition_id="c_full_callback_1",
            question="Will the full snapshot callback fire?",
            slug="full-callback-1",
            clob_token_ids=["tok_full_cb_yes_" + ("0" * 20), "tok_full_cb_no_" + ("0" * 20)],
            outcome_prices=[0.48, 0.52],
        )
        scanner._cached_markets = [market]
        scanner._cached_market_by_id = {market.id: market}
        scanner._cached_events = []
        opportunity = Opportunity(
            strategy="basic",
            title="Full snapshot callback opportunity",
            description="Heavy lane callback regression",
            total_cost=0.96,
            expected_payout=1.0,
            gross_profit=0.04,
            fee=0.02,
            net_profit=0.02,
            roi_percent=2.08,
            risk_score=0.3,
            markets=[
                {
                    "id": market.id,
                    "question": market.question,
                    "yes_price": 0.48,
                    "no_price": 0.52,
                    "clob_token_ids": list(market.clob_token_ids),
                }
            ],
            mispricing_type=MispricingType.WITHIN_MARKET,
            last_seen_at=utcnow(),
            detected_at=utcnow(),
        )
        callback = AsyncMock()
        scanner.add_callback(callback)

        with (
            patch.object(scanner, "_ensure_runtime_strategies_loaded", new_callable=AsyncMock),
            patch.object(scanner, "_partition_market_refresh_strategies", return_value=(set(), {"full_only"})),
            patch.object(scanner, "_snapshot_ws_prices", new_callable=AsyncMock, return_value={}),
            patch.object(scanner, "_dispatch_market_refresh", new_callable=AsyncMock, return_value=[opportunity]),
            patch.object(scanner, "_set_activity", new_callable=AsyncMock),
            patch("services.scanner.quality_filter") as mock_qf,
        ):
            mock_qf.evaluate_opportunity = MagicMock(return_value=_make_quality_pass())
            results = await scanner.scan_full_snapshot_strategies(force=True)

        assert len(results) == 1
        callback.assert_awaited_once()
        call_args = callback.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0].title == "Full snapshot callback opportunity"

    @pytest.mark.asyncio
    async def test_full_snapshot_scan_publishes_only_actionable_opportunities(self, mock_polymarket_client):
        scanner = _build_scanner(mock_client=mock_polymarket_client)
        market = Market(
            id="m_full_callback_2",
            condition_id="c_full_callback_2",
            question="Will only actionable heavy-lane results survive?",
            slug="full-callback-2",
            clob_token_ids=["tok_full_cb2_yes_" + ("0" * 20), "tok_full_cb2_no_" + ("0" * 20)],
            outcome_prices=[0.47, 0.53],
        )
        scanner._cached_markets = [market]
        scanner._cached_market_by_id = {market.id: market}
        scanner._cached_events = []
        passed = Opportunity(
            strategy="basic",
            title="Heavy lane actionable opportunity",
            description="Should remain in pool",
            total_cost=0.96,
            expected_payout=1.0,
            gross_profit=0.04,
            fee=0.02,
            net_profit=0.02,
            roi_percent=2.08,
            risk_score=0.3,
            markets=[
                {
                    "id": market.id,
                    "question": market.question,
                    "yes_price": 0.47,
                    "no_price": 0.53,
                    "clob_token_ids": list(market.clob_token_ids),
                }
            ],
            mispricing_type=MispricingType.WITHIN_MARKET,
            last_seen_at=utcnow(),
            detected_at=utcnow(),
        )
        rejected = passed.model_copy(deep=True)
        rejected.stable_id = f"{passed.stable_id}_rejected"
        rejected.id = f"{passed.id}_rejected"
        rejected.title = "Heavy lane rejected opportunity"
        callback = AsyncMock()
        scanner.add_callback(callback)
        reports = {
            passed.stable_id: _make_quality_pass(),
            rejected.stable_id: _make_quality_fail(),
        }

        with (
            patch.object(scanner, "_ensure_runtime_strategies_loaded", new_callable=AsyncMock),
            patch.object(scanner, "_partition_market_refresh_strategies", return_value=(set(), {"full_only"})),
            patch.object(scanner, "_snapshot_ws_prices", new_callable=AsyncMock, return_value={}),
            patch.object(scanner, "_dispatch_market_refresh", new_callable=AsyncMock, return_value=[rejected, passed]),
            patch.object(scanner, "_set_activity", new_callable=AsyncMock),
            patch("services.scanner.quality_filter") as mock_qf,
        ):
            mock_qf.evaluate_opportunity = MagicMock(
                side_effect=lambda opp, overrides=None: reports[opp.stable_id]
            )
            results = await scanner.scan_full_snapshot_strategies(force=True)

        assert [opp.stable_id for opp in results] == [passed.stable_id]
        callback.assert_awaited_once()
        assert [opp.stable_id for opp in callback.call_args[0][0]] == [passed.stable_id]
        assert set(scanner.quality_reports.keys()) == {passed.stable_id}

    @pytest.mark.asyncio
    async def test_full_snapshot_dispatch_clones_only_current_chunk_events(self, mock_polymarket_client):
        scanner = _build_scanner(mock_client=mock_polymarket_client)
        event_1_markets = []
        event_2_markets = []
        for event_idx, bucket in ((1, event_1_markets), (2, event_2_markets)):
            for market_idx in range(2):
                bucket.append(
                    Market(
                        id=f"m_chunk_event_{event_idx}_{market_idx}",
                        condition_id=f"c_chunk_event_{event_idx}_{market_idx}",
                        question=f"Chunk event {event_idx} market {market_idx}?",
                        slug=f"chunk-event-{event_idx}-{market_idx}",
                        event_slug=f"event-{event_idx}",
                        clob_token_ids=[
                            f"tok_chunk_event_{event_idx}_{market_idx}_yes",
                            f"tok_chunk_event_{event_idx}_{market_idx}_no",
                        ],
                        outcome_prices=[0.49, 0.51],
                    )
                )
        event_1 = Event(id="event-1", slug="event-1", title="Chunk event one", markets=event_1_markets)
        event_2 = Event(id="event-2", slug="event-2", title="Chunk event two", markets=event_2_markets)
        scanner._cached_events = [event_1, event_2]
        scanner._cached_markets = [*event_1_markets, *event_2_markets]
        scanner._cached_market_by_id = {market.id: market for market in scanner._cached_markets}
        dispatches: list[tuple[list, list]] = []

        async def _capture_dispatch(event, **kwargs):
            dispatches.append((list(event.events or []), list(kwargs["full_market_snapshot"])))
            return []

        with (
            patch.object(scanner, "_ensure_runtime_strategies_loaded", new_callable=AsyncMock),
            patch.object(scanner, "_partition_market_refresh_strategies", return_value=(set(), {"full_only"})),
            patch.object(scanner, "_snapshot_ws_prices", new_callable=AsyncMock, return_value={}),
            patch.object(scanner, "_dispatch_market_refresh", side_effect=_capture_dispatch),
            patch.object(scanner, "_set_activity", new_callable=AsyncMock),
            patch("services.scanner.settings.SCANNER_FULL_SNAPSHOT_CHUNK_SIZE", 2),
            patch("services.scanner.settings.SCANNER_FORCE_FULL_UNIVERSE", True),
        ):
            await scanner.scan_full_snapshot_strategies(force=True)

        assert len(dispatches) == 2
        for chunk_events, chunk_markets in dispatches:
            assert len(chunk_events) == 1
            assert len(chunk_markets) == 2
            assert {market.id for market in chunk_events[0].markets} == {market.id for market in chunk_markets}
            assert all(
                all(market is not cached_market for cached_market in scanner._cached_markets)
                for market in chunk_markets
            )
        assert scanner._cached_events[0].markets[0] is event_1_markets[0]
        assert scanner._cached_events[1].markets[0] is event_2_markets[0]

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
