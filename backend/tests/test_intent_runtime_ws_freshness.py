from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from services.intent_runtime import (
    IntentRuntime,
    _SIGNAL_PUBLICATION_BATCH_SIZE,
    _extract_required_token_ids,
    _projection_pressure_ttl_seconds,
    _snapshot_has_strict_scanner_live_market,
    _strict_ws_ttl_seconds_for_source,
)
from services.market_roster import build_market_roster
from services.signal_bus import build_signal_contract_from_opportunity
from services.strategies.base import BaseStrategy
from models.market import Event, Market
from models.opportunity import Opportunity
from utils.utcnow import utcnow


def test_projection_pressure_ttl_ignores_first_projection_failure():
    assert _projection_pressure_ttl_seconds(0) is None
    assert _projection_pressure_ttl_seconds(1) == pytest.approx(15.0)
    assert _projection_pressure_ttl_seconds(5) == pytest.approx(45.0)


def test_tokens_have_fresh_ws_quotes_uses_scanner_age_budget(monkeypatch):
    seen_max_age: list[float] = []

    class _Cache:
        def is_fresh(self, token_id: str, *, max_age_seconds: float | None = None) -> bool:
            assert token_id == "scanner-token"
            seen_max_age.append(float(max_age_seconds or 0.0))
            return True

        def get_mid_price(self, token_id: str):
            assert token_id == "scanner-token"
            return 0.42

    monkeypatch.setattr(
        "services.intent_runtime.get_feed_manager",
        lambda: SimpleNamespace(_started=True, cache=_Cache()),
    )

    runtime = IntentRuntime()
    assert runtime._tokens_have_fresh_ws_quotes(["scanner-token"], source="scanner") is True
    assert seen_max_age == [pytest.approx(30.0)]


def test_tokens_have_fresh_ws_quotes_allows_current_scanner_subscription_without_recent_tick(monkeypatch):
    seen_max_age: list[float] = []

    class _Cache:
        def is_fresh(self, token_id: str, *, max_age_seconds: float | None = None) -> bool:
            assert token_id == "scanner-token"
            seen_max_age.append(float(max_age_seconds or 0.0))
            return False

        def get_mid_price(self, token_id: str):
            assert token_id == "scanner-token"
            return 0.42

        def get_observed_at_epoch(self, token_id: str):
            return 1_700_000_000.0

    feed_manager = SimpleNamespace(
        _started=True,
        cache=_Cache(),
        has_current_subscription_price=lambda token_id, **kwargs: token_id == "scanner-token",
    )
    monkeypatch.setattr("services.intent_runtime.get_feed_manager", lambda: feed_manager)

    runtime = IntentRuntime()
    assert runtime._tokens_have_fresh_ws_quotes(["scanner-token"], source="scanner") is True
    assert runtime._tokens_have_fresh_ws_quotes(["scanner-token"], source="crypto") is False
    assert seen_max_age == [
        pytest.approx(_strict_ws_ttl_seconds_for_source("scanner")),
        pytest.approx(_strict_ws_ttl_seconds_for_source("crypto")),
    ]


def test_extract_required_token_ids_prefers_execution_tokens_for_single_leg_signal():
    payload = {
        "selected_token_id": "selected-token",
        "yes_token_id": "yes-token",
        "no_token_id": "no-token",
        "execution_plan": {
            "legs": [
                {
                    "side": "buy",
                    "token_id": "selected-token",
                }
            ]
        },
        "markets": [
            {
                "clob_token_ids": ["yes-token", "no-token"],
            }
        ],
    }

    assert _extract_required_token_ids(payload, direction="buy_no") == ["selected-token"]


@pytest.mark.asyncio
async def test_ensure_hot_subscriptions_resubscribes_when_live_feed_lost_token(monkeypatch):
    subscribe_mock = AsyncMock(return_value=None)
    feed_manager = SimpleNamespace(
        _started=True,
        start=AsyncMock(return_value=None),
        polymarket_feed=SimpleNamespace(
            subscribe=subscribe_mock,
            _subscribed_assets=set(),
        ),
    )
    monkeypatch.setattr("services.intent_runtime.get_feed_manager", lambda: feed_manager)

    runtime = IntentRuntime()
    runtime._hot_subscription_tokens.add("scanner-token")

    await runtime._ensure_hot_subscriptions(["scanner-token"])

    subscribe_mock.assert_awaited_once_with(["scanner-token"])


@pytest.mark.asyncio
async def test_ensure_hot_subscriptions_seeds_missing_cache_entries(monkeypatch):
    subscribe_mock = AsyncMock(return_value=None)
    seed_mock = AsyncMock(return_value=None)

    class _Cache:
        def get_mid_price(self, token_id: str):
            return None

    feed_manager = SimpleNamespace(
        _started=True,
        start=AsyncMock(return_value=None),
        cache=_Cache(),
        get_order_book=seed_mock,
        polymarket_feed=SimpleNamespace(
            subscribe=subscribe_mock,
            _subscribed_assets=set(),
        ),
    )
    monkeypatch.setattr("services.intent_runtime.get_feed_manager", lambda: feed_manager)

    runtime = IntentRuntime()

    await runtime._ensure_hot_subscriptions(["scanner-token", "scanner-token-2"])

    subscribe_mock.assert_awaited_once_with(["scanner-token", "scanner-token-2"])
    assert seed_mock.await_count == 2
    assert {call.args[0] for call in seed_mock.await_args_list} == {"scanner-token", "scanner-token-2"}


@pytest.mark.asyncio
async def test_ensure_hot_subscriptions_seeds_every_missing_cache_entry(monkeypatch):
    subscribe_mock = AsyncMock(return_value=None)
    seed_mock = AsyncMock(return_value=None)

    class _Cache:
        def get_mid_price(self, token_id: str):
            return None

    token_ids = [f"scanner-token-{idx}" for idx in range(30)]
    feed_manager = SimpleNamespace(
        _started=True,
        start=AsyncMock(return_value=None),
        cache=_Cache(),
        get_order_book=seed_mock,
        polymarket_feed=SimpleNamespace(
            subscribe=subscribe_mock,
            _subscribed_assets=set(),
        ),
    )
    monkeypatch.setattr("services.intent_runtime.get_feed_manager", lambda: feed_manager)

    runtime = IntentRuntime()

    await runtime._ensure_hot_subscriptions(token_ids)

    subscribe_mock.assert_awaited_once_with(token_ids)
    assert seed_mock.await_count == len(token_ids)
    assert [call.args[0] for call in seed_mock.await_args_list] == token_ids


def test_snapshot_has_strict_scanner_live_market_accepts_current_subscription_without_recent_tick():
    snapshot = {
        "source": "scanner",
        "required_token_ids": ["scanner-token"],
        "payload_json": {
            "strategy_runtime": {"execution_activation": "ws_current"},
            "live_market": {
                "available": True,
                "market_data_source": "ws_strict",
                "live_selected_price": 0.42,
                "market_data_age_ms": 120000.0,
                "ws_subscription_current": True,
            },
        },
    }

    assert _snapshot_has_strict_scanner_live_market(snapshot) is True


@pytest.mark.asyncio
async def test_defer_signal_rewarms_required_tokens(monkeypatch):
    monkeypatch.setattr("services.intent_runtime.event_bus.publish", AsyncMock(return_value=None))

    runtime = IntentRuntime()
    runtime._ensure_hot_subscriptions = AsyncMock(return_value=None)
    runtime._signals_by_id["signal-1"] = {
        "id": "signal-1",
        "source": "scanner",
        "status": "pending",
        "direction": "buy_no",
        "payload_json": {
            "positions_to_take": [
                {
                    "token_id": "scanner-token",
                }
            ]
        },
        "required_token_ids": ["scanner-token"],
        "updated_at": utcnow().isoformat().replace("+00:00", "Z"),
    }

    deferred = await runtime.defer_signal(signal_id="signal-1", reason="strict_ws_pricing_live_context_unavailable")
    await asyncio.sleep(0)

    assert deferred is True
    runtime._ensure_hot_subscriptions.assert_awaited_once_with(["scanner-token"])
    assert runtime._signals_by_id["signal-1"]["deferred_until_ws"] is True


@pytest.mark.asyncio
async def test_publish_opportunities_restamps_signal_emitted_at_to_actionable_publish_time(monkeypatch):
    published_batches: list[dict[str, object]] = []

    async def _publish_signal_batch(**kwargs):
        published_batches.append(kwargs)
        return "batch-1"

    monkeypatch.setattr(
        "services.intent_runtime.build_signal_contract_from_opportunity",
        lambda _opportunity: (
            "market-1",
            "buy_yes",
            0.42,
            "Will event happen?",
            {"markets": [{"id": "market-1"}]},
            {},
        ),
    )
    monkeypatch.setattr("services.intent_runtime.publish_signal_batch", _publish_signal_batch)
    monkeypatch.setattr("services.intent_runtime.event_bus.publish", AsyncMock(return_value=None))

    runtime = IntentRuntime()
    opportunity = SimpleNamespace(
        id="opp-1",
        stable_id="stable-1",
        strategy="tail_end_carry",
        roi_percent=7.5,
        confidence=0.8,
        min_liquidity=1000.0,
        resolution_date=None,
    )

    published = await runtime.publish_opportunities([opportunity], source="scanner")

    assert published == 1
    assert len(published_batches) == 1
    emitted_at = str(published_batches[0]["emitted_at"])
    snapshots = published_batches[0]["signal_snapshots"]
    snapshot = next(iter(snapshots.values()))
    assert snapshot["payload_json"]["signal_emitted_at"] == emitted_at


@pytest.mark.asyncio
async def test_publish_opportunities_reactivates_unchanged_scanner_terminal_signal_after_cooldown(monkeypatch):
    published_batches: list[dict[str, object]] = []

    async def _publish_signal_batch(**kwargs):
        published_batches.append(kwargs)
        return "batch-1"

    monkeypatch.setattr(
        "services.intent_runtime.build_signal_contract_from_opportunity",
        lambda _opportunity: (
            "market-1",
            "buy_yes",
            0.42,
            "Will event happen?",
            {"markets": [{"id": "market-1"}]},
            {},
        ),
    )
    monkeypatch.setattr("services.intent_runtime.publish_signal_batch", _publish_signal_batch)
    monkeypatch.setattr("services.intent_runtime.event_bus.publish", AsyncMock(return_value=None))

    runtime = IntentRuntime()
    opportunity = SimpleNamespace(
        id="opp-reactivate-1",
        stable_id="stable-reactivate-1",
        strategy="generic_strategy",
        roi_percent=7.5,
        confidence=0.8,
        min_liquidity=1000.0,
        resolution_date=None,
    )

    assert await runtime.publish_opportunities([opportunity], source="scanner") == 1
    signal_id = next(iter(runtime._signals_by_id))
    runtime._signals_by_id[signal_id]["status"] = "skipped"
    runtime._signals_by_id[signal_id]["runtime_sequence"] = None
    runtime._signals_by_id[signal_id]["updated_at"] = (
        utcnow() - timedelta(seconds=181)
    ).isoformat().replace("+00:00", "Z")
    published_batches.clear()

    published = await runtime.publish_opportunities([opportunity], source="scanner")

    assert published == 1
    assert len(published_batches) == 1
    assert published_batches[0]["event_type"] == "upsert_reactivated"
    refreshed = runtime._signals_by_id[signal_id]
    assert refreshed["status"] == "pending"
    assert refreshed["runtime_sequence"] is not None


@pytest.mark.asyncio
async def test_publish_opportunities_does_not_reactivate_unchanged_scanner_terminal_signal_inside_cooldown(monkeypatch):
    published_batches: list[dict[str, object]] = []

    async def _publish_signal_batch(**kwargs):
        published_batches.append(kwargs)
        return "batch-1"

    monkeypatch.setattr(
        "services.intent_runtime.build_signal_contract_from_opportunity",
        lambda _opportunity: (
            "market-1",
            "buy_yes",
            0.42,
            "Will event happen?",
            {"markets": [{"id": "market-1"}]},
            {},
        ),
    )
    monkeypatch.setattr("services.intent_runtime.publish_signal_batch", _publish_signal_batch)
    monkeypatch.setattr("services.intent_runtime.event_bus.publish", AsyncMock(return_value=None))

    runtime = IntentRuntime()
    opportunity = SimpleNamespace(
        id="opp-reactivate-2",
        stable_id="stable-reactivate-2",
        strategy="generic_strategy",
        roi_percent=7.5,
        confidence=0.8,
        min_liquidity=1000.0,
        resolution_date=None,
    )

    assert await runtime.publish_opportunities([opportunity], source="scanner") == 1
    signal_id = next(iter(runtime._signals_by_id))
    runtime._signals_by_id[signal_id]["status"] = "skipped"
    runtime._signals_by_id[signal_id]["runtime_sequence"] = None
    runtime._signals_by_id[signal_id]["updated_at"] = utcnow().isoformat().replace("+00:00", "Z")
    published_batches.clear()

    published = await runtime.publish_opportunities([opportunity], source="scanner")

    assert published == 0
    assert published_batches == []
    refreshed = runtime._signals_by_id[signal_id]
    assert refreshed["status"] == "skipped"
    assert refreshed["runtime_sequence"] is None


@pytest.mark.asyncio
async def test_publish_opportunities_batches_actionable_and_projection_work(monkeypatch):
    published_batches: list[dict[str, object]] = []
    projection_payloads: list[dict[str, object]] = []

    async def _publish_signal_batch(**kwargs):
        published_batches.append(kwargs)
        return f"batch-{len(published_batches)}"

    def _build_signal_contract(opportunity):
        index = str(getattr(opportunity, "id"))
        return (
            f"market-{index}",
            "buy_yes",
            0.42,
            f"Question {index}?",
            {"markets": [{"id": f"market-{index}"}]},
            {},
        )

    async def _enqueue_projection(payload: dict[str, object]) -> None:
        projection_payloads.append(payload)

    monkeypatch.setattr(
        "services.intent_runtime.build_signal_contract_from_opportunity",
        _build_signal_contract,
    )
    monkeypatch.setattr("services.intent_runtime.publish_signal_batch", _publish_signal_batch)
    monkeypatch.setattr("services.intent_runtime.event_bus.publish", AsyncMock(return_value=None))

    runtime = IntentRuntime()
    runtime._publish_signal_stats = AsyncMock(return_value=None)
    runtime._attach_live_market_contexts = AsyncMock(return_value=None)
    runtime._defer_scanner_snapshots_without_strict_live_market = AsyncMock(return_value=None)
    runtime._enqueue_projection = _enqueue_projection

    opportunities = [
        SimpleNamespace(
            id=f"opp-{index}",
            stable_id=f"stable-{index}",
            strategy="generic_strategy",
            roi_percent=7.5,
            confidence=0.8,
            min_liquidity=1000.0,
            resolution_date=None,
        )
        for index in range((_SIGNAL_PUBLICATION_BATCH_SIZE * 2) + 50)
    ]

    published = await runtime.publish_opportunities(
        opportunities,
        source="scanner",
        sweep_missing=True,
    )

    assert published == len(opportunities)
    assert [len(batch["signal_ids"]) for batch in published_batches] == [
        _SIGNAL_PUBLICATION_BATCH_SIZE,
        _SIGNAL_PUBLICATION_BATCH_SIZE,
        50,
    ]
    assert runtime._attach_live_market_contexts.await_count == 3
    assert runtime._defer_scanner_snapshots_without_strict_live_market.await_count == 3
    assert len(projection_payloads) == 4
    assert [len(payload["snapshots"]) for payload in projection_payloads[:-1]] == [
        _SIGNAL_PUBLICATION_BATCH_SIZE,
        _SIGNAL_PUBLICATION_BATCH_SIZE,
        50,
    ]
    assert all(payload["sweep_missing"] is False for payload in projection_payloads[:-1])
    assert projection_payloads[-1]["snapshots"] == {}
    assert projection_payloads[-1]["sweep_missing"] is True
    assert len(projection_payloads[-1]["keep_dedupe_keys"]) == len(opportunities)


@pytest.mark.asyncio
async def test_project_upserts_scopes_source_sweep_by_strategy_type(monkeypatch):
    class _Result:
        def __init__(self, rows):
            self._rows = rows

        def all(self):
            return list(self._rows)

    class _Session:
        async def execute(self, statement):
            del statement
            return _Result([])

        async def commit(self) -> None:
            return None

    class _SessionContext:
        async def __aenter__(self) -> _Session:
            return _Session()

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            del exc_type, exc, tb
            return False

    async def _upsert_trade_signal(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(status="pending", runtime_sequence=None, effective_price=None)

    expire_mock = AsyncMock(return_value=0)
    monkeypatch.setattr("services.intent_runtime.AsyncSessionLocal", lambda: _SessionContext())
    monkeypatch.setattr("services.intent_runtime.upsert_trade_signal", _upsert_trade_signal)
    monkeypatch.setattr("services.intent_runtime.expire_source_signals_except", expire_mock)

    runtime = IntentRuntime()
    await runtime._project_upsert_batch(
        {
            "source": "scanner",
            "snapshots": {
                "sig-corr-1": {
                    "id": "sig-corr-1",
                    "source": "scanner",
                    "source_item_id": "stable-corr-1",
                    "signal_type": "scanner_opportunity",
                    "strategy_type": "generic_pair_strategy",
                    "market_id": "market-corr-1",
                    "market_question": "Question?",
                    "direction": "buy_yes",
                    "entry_price": 0.42,
                    "edge_percent": 6.0,
                    "confidence": 0.7,
                    "liquidity": 1500.0,
                    "status": "pending",
                    "dedupe_key": "dedupe-corr-1",
                    "payload_json": {},
                    "strategy_context_json": {},
                    "runtime_sequence": 101,
                    "updated_at": utcnow(),
                }
            },
            "sweep_missing": True,
            "keep_dedupe_keys": ["dedupe-corr-1"],
        }
    )

    assert expire_mock.await_count == 1
    kwargs = expire_mock.await_args.kwargs
    assert kwargs["source"] == "scanner"
    assert kwargs["keep_dedupe_keys"] == {"dedupe-corr-1"}
    assert kwargs["signal_types"] == ["scanner_opportunity"]
    assert kwargs["strategy_types"] == ["generic_pair_strategy"]


@pytest.mark.asyncio
async def test_project_upserts_skip_nonreactivable_existing_rows(monkeypatch):
    class _Result:
        def __init__(self, rows):
            self._rows = rows

        def all(self):
            return list(self._rows)

    class _Session:
        async def execute(self, statement):
            del statement
            return _Result([("dedupe-frozen-1", "filtered", None, None)])

        async def commit(self) -> None:
            return None

    class _SessionContext:
        async def __aenter__(self) -> _Session:
            return _Session()

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            del exc_type, exc, tb
            return False

    upsert_mock = AsyncMock()
    monkeypatch.setattr("services.intent_runtime.AsyncSessionLocal", lambda: _SessionContext())
    monkeypatch.setattr("services.intent_runtime.upsert_trade_signal", upsert_mock)

    runtime = IntentRuntime()
    await runtime._project_upsert_batch(
        {
            "source": "scanner",
            "snapshots": {
                "sig-frozen-1": {
                    "id": "sig-frozen-1",
                    "source": "scanner",
                    "source_item_id": "stable-frozen-1",
                    "signal_type": "scanner_opportunity",
                    "strategy_type": "generic_pair_strategy",
                    "market_id": "market-frozen-1",
                    "market_question": "Question?",
                    "direction": "buy_yes",
                    "entry_price": 0.42,
                    "edge_percent": 6.0,
                    "confidence": 0.7,
                    "liquidity": 1500.0,
                    "status": "pending",
                    "dedupe_key": "dedupe-frozen-1",
                    "payload_json": {},
                    "strategy_context_json": {},
                    "runtime_sequence": 101,
                    "updated_at": utcnow(),
                }
            },
            "sweep_missing": False,
            "keep_dedupe_keys": [],
        }
    )

    upsert_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_publish_opportunities_uses_strategy_ttl_instead_of_resolution_date(monkeypatch):
    monkeypatch.setattr(
        "services.intent_runtime.build_signal_contract_from_opportunity",
        lambda _opportunity: (
            "market-1",
            "buy_yes",
            0.42,
            "Will event happen?",
            {"markets": [{"id": "market-1"}]},
            {},
        ),
    )
    monkeypatch.setattr("services.intent_runtime.publish_signal_batch", AsyncMock(return_value="batch-1"))
    monkeypatch.setattr("services.intent_runtime.event_bus.publish", AsyncMock(return_value=None))
    monkeypatch.setattr(
        "services.intent_runtime.strategy_loader.get_instance",
        lambda slug: SimpleNamespace(config={"retention_window": "15m"}) if slug == "ttl_strategy" else None,
    )

    runtime = IntentRuntime()
    resolution_date = utcnow() + timedelta(days=2)
    opportunity = SimpleNamespace(
        id="opp-ttl-1",
        stable_id="stable-ttl-1",
        strategy="ttl_strategy",
        roi_percent=7.5,
        confidence=0.8,
        min_liquidity=1000.0,
        resolution_date=resolution_date,
    )

    published = await runtime.publish_opportunities([opportunity], source="scanner")

    assert published == 1
    snapshot = next(iter(runtime._signals_by_id.values()))
    expires_at = datetime.fromisoformat(snapshot["expires_at"].replace("Z", "+00:00"))
    ttl_seconds = (expires_at - utcnow()).total_seconds()
    assert ttl_seconds < 20 * 60
    assert expires_at < resolution_date


def test_build_signal_contract_infers_runtime_metadata_from_loaded_strategy(monkeypatch):
    monkeypatch.setattr(
        "services.strategy_loader.strategy_loader.get_strategy",
        lambda slug: SimpleNamespace(
            instance=SimpleNamespace(
                source_key="scanner",
                subscriptions=["market_data_refresh"],
            )
        )
        if slug == "custom_runtime_strategy"
        else None,
    )

    opportunity = Opportunity(
        strategy="custom_runtime_strategy",
        title="Custom strategy signal",
        description="Test custom strategy runtime metadata",
        total_cost=0.42,
        expected_payout=1.0,
        gross_profit=0.1,
        fee=0.0,
        net_profit=0.1,
        roi_percent=10.0,
        markets=[{"id": "market-1", "question": "Will it happen?"}],
        positions_to_take=[
            {
                "market_id": "market-1",
                "token_id": "scanner-token",
                "outcome": "YES",
                "side": "buy",
                "price": 0.42,
            }
        ],
    )

    _market_id, _direction, _entry_price, _market_question, payload, strategy_context = (
        build_signal_contract_from_opportunity(opportunity)
    )

    assert payload["strategy_runtime"]["source_key"] == "scanner"
    assert payload["strategy_runtime"]["subscriptions"] == ["market_data_refresh"]
    assert payload["strategy_runtime"]["execution_activation"] == "ws_current"
    assert strategy_context["source_key"] == "scanner"
    assert strategy_context["execution_activation"] == "ws_current"


def test_build_signal_contract_limits_non_guaranteed_event_roster_to_selected_markets():
    class _TestStrategy(BaseStrategy):
        strategy_type = "test_roster"
        name = "Test Roster"
        description = "Test roster persistence"

    strategy = _TestStrategy()
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

    opportunity = strategy.create_opportunity(
        title="Atomic roster opportunity",
        description="Test event roster persistence",
        total_cost=0.8,
        expected_payout=1.0,
        is_guaranteed=False,
        markets=[home, away],
        positions=[
            {"market_id": "market-home", "token_id": "token-home", "outcome": "YES", "side": "buy", "price": 0.4},
            {"market_id": "market-away", "token_id": "token-away", "outcome": "YES", "side": "buy", "price": 0.4},
        ],
        event=event,
    )

    _market_id, _direction, _entry_price, _market_question, payload, _strategy_context = build_signal_contract_from_opportunity(opportunity)

    assert payload["market_roster"]["scope"] == "signal"
    assert payload["market_roster"]["market_count"] == 2
    assert {market["id"] for market in payload["markets"]} == {
        "market-home",
        "market-away",
    }
    assert {market["id"] for market in payload["market_roster"]["markets"]} == {
        "market-home",
        "market-away",
    }


def test_build_signal_contract_keeps_full_event_market_roster_for_guaranteed_multileg_bundle():
    class _TestStrategy(BaseStrategy):
        strategy_type = "test_roster"
        name = "Test Roster"
        description = "Test roster persistence"

    strategy = _TestStrategy()
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

    opportunity = Opportunity(
        strategy=strategy.strategy_type,
        title="Atomic roster opportunity",
        description="Test event roster persistence",
        total_cost=0.8,
        expected_payout=1.0,
        gross_profit=0.2,
        fee=0.0,
        net_profit=0.2,
        roi_percent=25.0,
        is_guaranteed=True,
        markets=[home.model_dump(mode="json"), away.model_dump(mode="json")],
        market_roster=build_market_roster(
            [home, draw, away],
            scope="event",
            event_id=event.id,
            event_slug=event.slug,
            event_title=event.title,
            category=event.category,
        ),
        positions_to_take=[
            {"market_id": "market-home", "token_id": "token-home", "outcome": "YES", "side": "buy", "price": 0.4},
            {"market_id": "market-away", "token_id": "token-away", "outcome": "YES", "side": "buy", "price": 0.4},
        ],
        event_id=event.id,
        event_slug=event.slug,
        event_title=event.title,
        category=event.category,
    )

    _market_id, _direction, _entry_price, _market_question, payload, _strategy_context = build_signal_contract_from_opportunity(opportunity)

    assert payload["market_roster"]["scope"] == "event"
    assert payload["market_roster"]["market_count"] == 3
    assert {market["id"] for market in payload["market_roster"]["markets"]} == {
        "market-home",
        "market-draw",
        "market-away",
    }


def test_build_signal_contract_strips_market_history_from_durable_payload():
    opportunity = Opportunity(
        strategy="generic_contract",
        title="Contract trimming",
        description="Ensure signal payload stays execution-focused",
        total_cost=0.42,
        expected_payout=1.0,
        gross_profit=0.58,
        fee=0.0,
        net_profit=0.58,
        roi_percent=138.0,
        markets=[
            {
                "id": "market-1",
                "question": "Will it happen?",
                "slug": "will-it-happen",
                "event_slug": "event-1",
                "clob_token_ids": ["yes-token", "no-token"],
                "outcome_prices": [0.42, 0.58],
                "price_updated_at": utcnow().isoformat(),
                "price_history": [{"t": "2026-04-09T00:00:00Z", "yes": 0.4, "no": 0.6}],
                "history_tail": [{"t": "2026-04-09T00:00:00Z", "yes": 0.4, "no": 0.6}],
                "oracle_history": [{"t": "2026-04-09T00:00:00Z", "price": 123.4}],
            }
        ],
        positions_to_take=[
            {
                "market_id": "market-1",
                "token_id": "yes-token",
                "outcome": "YES",
                "side": "buy",
                "price": 0.42,
            }
        ],
    )

    _market_id, _direction, _entry_price, _market_question, payload, _strategy_context = (
        build_signal_contract_from_opportunity(opportunity)
    )

    payload_market = payload["markets"][0]
    roster_market = payload["market_roster"]["markets"][0]
    assert "price_history" not in payload_market
    assert "history_tail" not in payload_market
    assert "oracle_history" not in payload_market
    assert "price_history" not in roster_market
    assert "history_tail" not in roster_market
    assert "oracle_history" not in roster_market


@pytest.mark.asyncio
async def test_publish_opportunities_uses_fresh_scanner_ws_quotes_without_post_arm_deferral(monkeypatch):
    published_batches: list[dict[str, object]] = []

    async def _publish_signal_batch(**kwargs):
        published_batches.append(kwargs)
        return "batch-1"

    class _Cache:
        def __init__(self) -> None:
            self.observed_at_epoch = datetime.now(timezone.utc).timestamp()

        def is_fresh(self, token_id: str, *, max_age_seconds: float | None = None) -> bool:
            assert token_id == "scanner-token"
            assert max_age_seconds == pytest.approx(30.0)
            return True

        def get_mid_price(self, token_id: str):
            assert token_id == "scanner-token"
            return 0.42

        def get_observed_at_epoch(self, token_id: str) -> float:
            assert token_id == "scanner-token"
            return self.observed_at_epoch

    cache = _Cache()

    monkeypatch.setattr(
        "services.intent_runtime.get_feed_manager",
        lambda: SimpleNamespace(
            _started=True,
            cache=cache,
            polymarket_feed=SimpleNamespace(subscribe=AsyncMock(return_value=None)),
        ),
    )
    monkeypatch.setattr("services.intent_runtime.publish_signal_batch", _publish_signal_batch)
    monkeypatch.setattr("services.intent_runtime.event_bus.publish", AsyncMock(return_value=None))
    monkeypatch.setattr(
        "services.intent_runtime.build_signal_contract_from_opportunity",
        lambda _opportunity: (
            "market-1",
            "buy_yes",
            0.42,
            "Will event happen?",
            {
                "markets": [{"id": "market-1"}],
                "strategy_runtime": {
                    "source_key": "scanner",
                    "subscriptions": ["market_data_refresh"],
                    "execution_activation": "ws_current",
                },
                "positions_to_take": [{"token_id": "scanner-token"}],
            },
            {},
        ),
    )

    runtime = IntentRuntime()
    runtime._ensure_hot_subscriptions = AsyncMock(return_value=None)
    opportunity = SimpleNamespace(
        id="opp-1",
        stable_id="stable-1",
        strategy="custom_runtime_strategy",
        roi_percent=7.5,
        confidence=0.8,
        min_liquidity=1000.0,
        resolution_date=None,
    )

    published = await runtime.publish_opportunities([opportunity], source="scanner")

    assert published == 1
    assert len(published_batches) == 1
    snapshot = next(iter(runtime._signals_by_id.values()))
    assert snapshot["deferred_until_ws"] is False
    assert snapshot["deferred_reason"] is None
    assert snapshot["runtime_sequence"] is not None
    assert snapshot["payload_json"]["execution_armed_at"]
    published_snapshot = next(iter(published_batches[0]["signal_snapshots"].values()))
    assert published_snapshot["payload_json"]["signal_emitted_at"] == published_batches[0]["emitted_at"]


@pytest.mark.asyncio
async def test_deferred_timeout_uses_stable_deferred_start_under_repeated_refresh(monkeypatch):
    published_batches: list[dict[str, object]] = []

    async def _publish_signal_batch(**kwargs):
        published_batches.append(kwargs)
        return "batch-1"

    class _Cache:
        def is_fresh(self, token_id: str, *, max_age_seconds: float | None = None) -> bool:
            assert token_id == "scanner-token"
            return False

        def get_mid_price(self, token_id: str):
            assert token_id == "scanner-token"
            return 0.42

    monkeypatch.setattr(
        "services.intent_runtime.get_feed_manager",
        lambda: SimpleNamespace(
            _started=True,
            cache=_Cache(),
            polymarket_feed=SimpleNamespace(subscribe=AsyncMock(return_value=None)),
        ),
    )
    monkeypatch.setattr("services.intent_runtime.publish_signal_batch", _publish_signal_batch)
    monkeypatch.setattr("services.intent_runtime.event_bus.publish", AsyncMock(return_value=None))
    monkeypatch.setattr("services.intent_runtime.settings.INTENT_RUNTIME_DEFERRED_MAX_AGE_SECONDS", 5.0)
    monkeypatch.setattr(
        "services.intent_runtime.build_signal_contract_from_opportunity",
        lambda _opportunity: (
            "market-1",
            "buy_yes",
            0.42,
            "Will event happen?",
            {
                "markets": [{"id": "market-1"}],
                "positions_to_take": [{"token_id": "scanner-token"}],
            },
            {},
        ),
    )

    runtime = IntentRuntime()
    runtime._ensure_hot_subscriptions = AsyncMock(return_value=None)
    opportunity = SimpleNamespace(
        id="opp-1",
        stable_id="stable-1",
        strategy="generic_scanner_strategy",
        roi_percent=7.5,
        confidence=0.8,
        min_liquidity=1000.0,
        resolution_date=None,
    )

    published = await runtime.publish_opportunities([opportunity], source="scanner")

    assert published == 0
    assert published_batches == []
    snapshot = next(iter(runtime._signals_by_id.values()))
    original_deferred_started_at = snapshot["deferred_started_at"]
    assert snapshot["deferred_until_ws"] is True
    assert original_deferred_started_at
    assert snapshot["deferred_reason"] == "prewarm_waiting_for_strict_ws_quote"
    assert "execution_armed_at" not in snapshot["payload_json"]
    assert "deferred_quote_min_observed_at" not in snapshot["payload_json"]

    await runtime.publish_opportunities([opportunity], source="scanner")
    refreshed_snapshot = next(iter(runtime._signals_by_id.values()))
    assert refreshed_snapshot["deferred_started_at"] == original_deferred_started_at

    stale_start = (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat().replace("+00:00", "Z")
    refreshed_snapshot["deferred_started_at"] = stale_start
    refreshed_snapshot["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    await runtime._release_stale_deferred_signals()

    assert published_batches == []
    assert refreshed_snapshot["deferred_until_ws"] is True
    assert refreshed_snapshot["deferred_reason"] == "prewarm_waiting_for_strict_ws_quote"
    assert refreshed_snapshot["runtime_sequence"] is None


@pytest.mark.asyncio
async def test_hydrate_from_db_keeps_cold_scanner_signal_deferred_until_ws_ready(monkeypatch):
    now = utcnow()

    class _Result:
        def __init__(self, rows: list[dict[str, object]]) -> None:
            self._rows = rows

        def mappings(self) -> "_Result":
            return self

        def all(self) -> list[dict[str, object]]:
            return self._rows

    # ``hydrate_from_db`` now selects the heavy JSON columns as text
    # (``payload_json``, ``strategy_context_json``,
    # ``quality_rejection_reasons``) so asyncpg does not parse them on
    # the event loop.  The fake row therefore returns the JSON-encoded
    # text under the ``*_text`` aliases the SELECT uses.
    import json as _json

    payload_text = _json.dumps(
        {
            "signal_emitted_at": now.isoformat().replace("+00:00", "Z"),
            "strategy_runtime": {
                "source_key": "scanner",
                "execution_activation": "ws_current",
                "subscriptions": ["market_data_refresh"],
            },
            "positions_to_take": [{"token_id": "scanner-token"}],
        }
    )
    strategy_context_text = _json.dumps(
        {
            "source_key": "scanner",
            "execution_activation": "ws_current",
        }
    )

    class _Session:
        async def execute(self, *args, **kwargs) -> _Result:
            del args, kwargs
            return _Result(
                [
                    {
                        "id": "sig-bootstrap-cold-1",
                        "source": "scanner",
                        "source_item_id": "stable-cold-1",
                        "signal_type": "scanner_opportunity",
                        "strategy_type": "tail",
                        "market_id": "market-cold-1",
                        "market_question": "Question?",
                        "direction": "buy_no",
                        "entry_price": 0.91,
                        "effective_price": None,
                        "edge_percent": 1.7,
                        "confidence": 0.75,
                        "liquidity": 120.0,
                        "expires_at": now,
                        "status": "pending",
                        "payload_json_text": payload_text,
                        "strategy_context_json_text": strategy_context_text,
                        "quality_passed": True,
                        "quality_rejection_reasons_text": None,
                        "dedupe_key": "dedupe-cold-1",
                        "runtime_sequence": None,
                        "created_at": now,
                        "updated_at": now,
                    }
                ]
            )

    class _SessionContext:
        async def __aenter__(self) -> _Session:
            return _Session()

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            del exc_type, exc, tb
            return False

    class _Cache:
        def is_fresh(self, token_id: str, *, max_age_seconds: float | None = None) -> bool:
            assert token_id == "scanner-token"
            return False

        def get_mid_price(self, token_id: str):
            assert token_id == "scanner-token"
            return None

    monkeypatch.setattr("services.intent_runtime.AsyncSessionLocal", lambda: _SessionContext())
    monkeypatch.setattr(
        "services.intent_runtime.get_feed_manager",
        lambda: SimpleNamespace(
            _started=True,
            cache=_Cache(),
            has_current_subscription_price=lambda token_id, **kwargs: False,
            polymarket_feed=SimpleNamespace(subscribe=AsyncMock(return_value=None)),
        ),
    )
    publish_mock = AsyncMock(return_value="batch-1")
    monkeypatch.setattr("services.intent_runtime.publish_signal_batch", publish_mock)
    monkeypatch.setattr("services.intent_runtime.event_bus.publish", AsyncMock(return_value=None))

    runtime = IntentRuntime()
    runtime._enqueue_projection = AsyncMock(return_value=None)
    runtime._publish_signal_stats = AsyncMock(return_value=None)

    await runtime.hydrate_from_db()

    snapshot = runtime._signals_by_id["sig-bootstrap-cold-1"]
    assert snapshot["runtime_sequence"] is None
    assert snapshot["deferred_until_ws"] is True
    assert snapshot["deferred_reason"] == "prewarm_waiting_for_strict_ws_quote"
    assert "execution_armed_at" not in snapshot["payload_json"]
    assert runtime._deferred_tokens_by_signal_id["sig-bootstrap-cold-1"] == {"scanner-token"}
    assert publish_mock.await_count == 0


@pytest.mark.asyncio
async def test_release_stale_deferred_signals_releases_ready_scanner_signal_on_next_sweep(monkeypatch):
    published_batches: list[dict[str, object]] = []

    async def _publish_signal_batch(**kwargs):
        published_batches.append(kwargs)
        return "batch-1"

    class _Cache:
        def is_fresh(self, token_id: str, *, max_age_seconds: float | None = None) -> bool:
            assert token_id == "scanner-token"
            return False

        def get_mid_price(self, token_id: str):
            assert token_id == "scanner-token"
            return 0.42

        def get_observed_at_epoch(self, token_id: str) -> float:
            return datetime.now(timezone.utc).timestamp() - 60.0

    monkeypatch.setattr(
        "services.intent_runtime.get_feed_manager",
        lambda: SimpleNamespace(
            _started=True,
            cache=_Cache(),
            has_current_subscription_price=lambda token_id, **kwargs: token_id == "scanner-token",
            polymarket_feed=SimpleNamespace(subscribe=AsyncMock(return_value=None)),
        ),
    )
    monkeypatch.setattr("services.intent_runtime.build_cached_live_signal_contexts", AsyncMock(return_value={}))
    monkeypatch.setattr("services.intent_runtime.publish_signal_batch", _publish_signal_batch)
    monkeypatch.setattr("services.intent_runtime.event_bus.publish", AsyncMock(return_value=None))

    runtime = IntentRuntime()
    signal_id = "scanner-deferred-1"
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    runtime._signals_by_id[signal_id] = {
        "id": signal_id,
        "source": "scanner",
        "status": "pending",
        "market_id": "market-1",
        "market_question": "Will event happen?",
        "direction": "buy_yes",
        "entry_price": 0.42,
        "payload_json": {},
        "required_token_ids": ["scanner-token"],
        "runtime_sequence": None,
        "deferred_until_ws": True,
        "deferred_reason": "strict_ws_pricing_live_context_unavailable",
        "deferred_started_at": now_iso,
        "created_at": now_iso,
        "updated_at": now_iso,
    }
    runtime._deferred_tokens_by_signal_id[signal_id] = {"scanner-token"}
    runtime._deferred_signal_ids_by_token["scanner-token"] = {signal_id}

    await runtime._release_stale_deferred_signals()

    snapshot = runtime._signals_by_id[signal_id]
    assert snapshot["runtime_sequence"] is not None
    assert snapshot["deferred_until_ws"] is False
    assert snapshot["deferred_reason"] is None
    assert snapshot["payload_json"]["execution_armed_at"]
    assert len(published_batches) == 1
    assert published_batches[0]["event_type"] == "upsert_reactivated"


def test_build_signal_contract_publishes_trader_strategy_with_immediate_activation(monkeypatch):
    monkeypatch.setattr(
        "services.strategy_loader.strategy_loader.get_strategy",
        lambda slug: SimpleNamespace(
            instance=SimpleNamespace(
                source_key="traders",
                subscriptions=["trader_activity"],
            )
        )
        if slug == "custom_copy_trade"
        else None,
    )

    opportunity = Opportunity(
        strategy="custom_copy_trade",
        title="Copy trade signal",
        description="Trader-source strategy publishes with immediate activation",
        total_cost=0.41,
        expected_payout=1.0,
        gross_profit=0.09,
        fee=0.0,
        net_profit=0.09,
        roi_percent=9.0,
        markets=[{"id": "market-1", "question": "Will it happen?"}],
        positions_to_take=[
            {
                "market_id": "market-1",
                "token_id": "trader-token",
                "outcome": "YES",
                "side": "buy",
                "price": 0.41,
            }
        ],
    )

    _market_id, _direction, _entry_price, _market_question, payload, strategy_context = (
        build_signal_contract_from_opportunity(opportunity)
    )

    assert payload["strategy_runtime"]["source_key"] == "traders"
    assert payload["strategy_runtime"]["execution_activation"] == "immediate"
    assert strategy_context["execution_activation"] == "immediate"


@pytest.mark.asyncio
async def test_publish_opportunities_defers_signal_when_activation_is_ws_post_arm_tick(monkeypatch):
    published_batches: list[dict[str, object]] = []

    async def _publish_signal_batch(**kwargs):
        published_batches.append(kwargs)
        return "batch-1"

    class _Cache:
        def __init__(self) -> None:
            self.observed_at_epoch = datetime.now(timezone.utc).timestamp()

        def is_fresh(self, token_id: str, *, max_age_seconds: float | None = None) -> bool:
            assert token_id == "deferred-token"
            return True

        def get_mid_price(self, token_id: str):
            assert token_id == "deferred-token"
            return 0.41

        def get_observed_at_epoch(self, token_id: str) -> float:
            assert token_id == "deferred-token"
            return self.observed_at_epoch

    cache = _Cache()

    monkeypatch.setattr(
        "services.intent_runtime.get_feed_manager",
        lambda: SimpleNamespace(
            _started=True,
            cache=cache,
            polymarket_feed=SimpleNamespace(subscribe=AsyncMock(return_value=None)),
        ),
    )
    monkeypatch.setattr("services.intent_runtime.publish_signal_batch", _publish_signal_batch)
    monkeypatch.setattr("services.intent_runtime.event_bus.publish", AsyncMock(return_value=None))
    monkeypatch.setattr(
        "services.intent_runtime.build_signal_contract_from_opportunity",
        lambda _opportunity: (
            "market-1",
            "buy_yes",
            0.41,
            "Will event happen?",
            {
                "markets": [{"id": "market-1"}],
                "strategy_runtime": {
                    "source_key": "future_unknown_source",
                    "subscriptions": ["future_unknown_topic"],
                    "execution_activation": "ws_post_arm_tick",
                },
                "positions_to_take": [{"token_id": "deferred-token"}],
            },
            {},
        ),
    )

    runtime = IntentRuntime()
    runtime._ensure_hot_subscriptions = AsyncMock(return_value=None)
    opportunity = SimpleNamespace(
        id="opp-2",
        stable_id="stable-2",
        strategy="custom_future_strategy",
        roi_percent=5.0,
        confidence=0.7,
        min_liquidity=900.0,
        resolution_date=None,
    )

    published = await runtime.publish_opportunities([opportunity], source="future_unknown_source")

    assert published == 0
    assert published_batches == []
    snapshot = next(iter(runtime._signals_by_id.values()))
    assert snapshot["deferred_until_ws"] is True
    assert snapshot["deferred_reason"] == "awaiting_post_arm_ws_tick"


def test_snapshot_ready_for_runtime_accepts_current_fresh_scanner_quote_after_strict_ws_defer(monkeypatch):
    state = {
        "now_epoch": datetime.now(timezone.utc).timestamp(),
        "observed_at_epoch": datetime.now(timezone.utc).timestamp() - 20.0,
    }
    seen_max_age_seconds: list[float] = []

    class _Cache:
        def is_fresh(self, token_id: str, *, max_age_seconds: float | None = None) -> bool:
            assert token_id == "scanner-token"
            seen_max_age_seconds.append(float(max_age_seconds or 0.0))
            return (state["now_epoch"] - state["observed_at_epoch"]) <= float(max_age_seconds or 0.0)

        def get_mid_price(self, token_id: str):
            assert token_id == "scanner-token"
            return 0.42

        def get_observed_at_epoch(self, token_id: str) -> float:
            assert token_id == "scanner-token"
            return state["observed_at_epoch"]

    monkeypatch.setattr(
        "services.intent_runtime.get_feed_manager",
        lambda: SimpleNamespace(_started=True, cache=_Cache()),
    )

    runtime = IntentRuntime()
    deferred_after = datetime.fromtimestamp(state["now_epoch"] - 5.0, tz=timezone.utc)
    snapshot = {
        "source": "scanner",
        "deferred_reason": "strict_ws_pricing_live_context_unavailable",
        "required_token_ids": ["scanner-token"],
        "payload_json": {
            "deferred_quote_min_observed_at": deferred_after.isoformat().replace("+00:00", "Z"),
            "strict_ws_required_max_age_ms": 15000,
        },
    }

    assert runtime._snapshot_ready_for_runtime(snapshot) is False

    state["observed_at_epoch"] = deferred_after.timestamp() - 60.0
    state["now_epoch"] = state["observed_at_epoch"] + 10.0

    assert runtime._snapshot_ready_for_runtime(snapshot) is True
    assert seen_max_age_seconds[-1] == pytest.approx(15.0)
