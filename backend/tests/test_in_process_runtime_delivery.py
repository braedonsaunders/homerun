from __future__ import annotations

import asyncio
from datetime import timezone
from unittest.mock import AsyncMock

import pytest

from services import intent_runtime as intent_runtime_module
from services.data_events import DataEvent, EventType
from services.event_bus import EventBus
from services.event_dispatcher import EventDispatcher
from services.runtime_signal_queue import get_queue_depth, publish_signal_batch, wait_for_signal_batch
from utils.utcnow import utcnow


@pytest.mark.asyncio
async def test_event_bus_delivers_to_local_subscriber() -> None:
    bus = EventBus()
    received: list[tuple[str, dict]] = []
    delivered = asyncio.Event()

    async def _handler(event_type: str, data: dict) -> None:
        received.append((event_type, data))
        delivered.set()

    bus.subscribe("trade.executed", _handler)
    await bus.start()
    try:
        await bus.publish("trade.executed", {"trade_id": "t-1", "pnl": 4.25})
        await asyncio.wait_for(delivered.wait(), timeout=1.0)
    finally:
        await bus.stop()

    assert received == [("trade.executed", {"trade_id": "t-1", "pnl": 4.25})]


@pytest.mark.asyncio
async def test_event_dispatcher_routes_data_events_in_process() -> None:
    dispatcher = EventDispatcher()
    seen: list[DataEvent] = []

    async def _handler(event: DataEvent) -> list:
        seen.append(event)
        return [{"signal_id": "sig-1"}]

    dispatcher.subscribe("strategy.local", EventType.PRICE_CHANGE, _handler)
    await dispatcher.start()
    try:
        event = DataEvent(
            event_type=EventType.PRICE_CHANGE,
            source="scanner_worker",
            timestamp=utcnow().astimezone(timezone.utc),
            market_id="mkt-1",
            token_id="tok-1",
            old_price=0.41,
            new_price=0.43,
            payload={"field": "value"},
        )
        results = await dispatcher.dispatch(event)
    finally:
        await dispatcher.stop()

    assert len(seen) == 1
    assert seen[0].event_type == EventType.PRICE_CHANGE
    assert results == [{"signal_id": "sig-1"}]


@pytest.mark.asyncio
async def test_runtime_signal_queue_coalesces_runtime_batches_for_all_lanes() -> None:
    starting_depth = get_queue_depth()
    assert isinstance(starting_depth, dict)
    assert all(int(value) == 0 for value in starting_depth.values())

    crypto_batch_id = await publish_signal_batch(
        event_type="upsert_insert",
        source="crypto",
        signal_ids=["sig-1", "sig-2", "sig-1"],
        trigger="strategy_signal_bridge",
        signal_snapshots={
            "sig-1": {"id": "sig-1", "source": "crypto"},
            "sig-2": {"id": "sig-2", "source": "crypto"},
        },
    )
    general_batch_id = await publish_signal_batch(
        event_type="upsert_insert",
        source="scanner",
        signal_ids=["sig-3", "sig-4", "sig-3"],
        trigger="strategy_signal_bridge",
        signal_snapshots={
            "sig-3": {"id": "sig-3", "source": "scanner"},
            "sig-4": {"id": "sig-4", "source": "scanner"},
        },
    )

    assert isinstance(crypto_batch_id, str)
    assert isinstance(general_batch_id, str)

    general_payload = await wait_for_signal_batch(lane="general", timeout_seconds=0.5)
    crypto_payload = await wait_for_signal_batch(lane="crypto", timeout_seconds=0.5)

    assert general_payload is not None
    assert crypto_payload is not None
    assert general_payload["event_type"] == "runtime_signal_batch"
    assert crypto_payload["event_type"] == "runtime_signal_batch"
    assert general_payload["source"] == "scanner"
    assert crypto_payload["source"] == "crypto"
    assert general_payload["source_signal_ids"]["scanner"] == ["sig-3", "sig-4"]
    assert crypto_payload["source_signal_ids"]["crypto"] == ["sig-1", "sig-2"]
    assert get_queue_depth() == {"general": 0, "crypto": 0}


@pytest.fixture
def _fresh_runtime_queues():
    """Give each test its own loop-bound runtime_signal_queue lanes.

    The module-level ``asyncio.Queue`` objects bind to the first event loop
    that touches them; pytest-asyncio runs each test in a fresh loop, so a
    queue bound by an earlier test raises "bound to a different event loop"
    when a later test touches it.  Production runs one loop per process, so
    this only matters for the test harness.
    """
    from services import runtime_signal_queue as rsq

    original = rsq._queues
    rsq._queues = {lane: asyncio.Queue() for lane in rsq._LANES}
    try:
        yield
    finally:
        rsq._queues = original


@pytest.mark.asyncio
async def test_redis_bridge_emission_wakes_runtime_queue(_fresh_runtime_queues) -> None:
    """A cross-plane signal emission (scanner running on the detection plane)
    must wake the trading orchestrator's in-process runtime_signal_queue — not
    only the event_bus.  Regression: after scanner moved to its own plane the
    orchestrator stopped cycling because nothing woke its lane queue.
    """
    from services import signal_bus_redis_bridge

    bridge = signal_bus_redis_bridge._Bridge()
    await bridge._bridge_emission(
        {
            "signal_id": "xplane-scanner-1",
            "source": "scanner",
            "status": "pending",
            "event_type": "upsert_insert",
            "reason": "scanner_detection",
            "updated_at": "2026-06-04T02:00:00Z",
        }
    )

    payload = await wait_for_signal_batch(lane="general", timeout_seconds=0.5)
    assert payload is not None
    assert payload["event_type"] == "runtime_signal_batch"
    assert payload["source"] == "scanner"
    assert payload["source_signal_ids"]["scanner"] == ["xplane-scanner-1"]
    assert get_queue_depth(lane="general") == 0


@pytest.mark.asyncio
async def test_redis_bridge_emission_routes_crypto_source_to_crypto_lane(_fresh_runtime_queues) -> None:
    """A crypto-source cross-plane emission must wake the crypto lane."""
    from services import signal_bus_redis_bridge

    bridge = signal_bus_redis_bridge._Bridge()
    await bridge._bridge_emission(
        {
            "signal_id": "xplane-crypto-1",
            "source": "crypto",
            "status": "pending",
            "event_type": "upsert_update",
            "updated_at": "2026-06-04T02:00:00Z",
        }
    )

    payload = await wait_for_signal_batch(lane="crypto", timeout_seconds=0.5)
    assert payload is not None
    assert payload["source"] == "crypto"
    assert payload["source_signal_ids"]["crypto"] == ["xplane-crypto-1"]


@pytest.mark.asyncio
async def test_redis_bridge_emission_ignores_nonactionable_status(_fresh_runtime_queues) -> None:
    """status_update / status_expired emissions must NOT wake a trade cycle —
    only upsert_* / reverse_entry events are actionable.
    """
    from services import signal_bus_redis_bridge

    bridge = signal_bus_redis_bridge._Bridge()
    await bridge._bridge_emission(
        {
            "signal_id": "xplane-status-1",
            "source": "scanner",
            "status": "filled",
            "event_type": "status_update",
            "updated_at": "2026-06-04T02:00:00Z",
        }
    )

    payload = await wait_for_signal_batch(lane="general", timeout_seconds=0.2)
    assert payload is None


@pytest.mark.asyncio
async def test_redis_bridge_emission_dedups_locally_emitted_signal(_fresh_runtime_queues) -> None:
    """A signal the trading plane already emitted in-process (and already woke
    the queue via intent_runtime) must be deduped when the Redis broadcast
    loops back, so the bridge does not double-fire a cycle.
    """
    from services import signal_bus_redis_bridge

    await signal_bus_redis_bridge.mark_locally_emitted(["xplane-local-1"])
    bridge = signal_bus_redis_bridge._Bridge()
    await bridge._bridge_emission(
        {
            "signal_id": "xplane-local-1",
            "source": "scanner",
            "status": "pending",
            "event_type": "upsert_insert",
            "updated_at": "2026-06-04T02:00:00Z",
        }
    )

    payload = await wait_for_signal_batch(lane="general", timeout_seconds=0.2)
    assert payload is None


@pytest.mark.asyncio
async def test_intent_runtime_hydrate_republishes_bootstrap_pending_signals(monkeypatch) -> None:
    now = utcnow()

    class _Result:
        def __init__(self, rows: list[dict[str, object]]) -> None:
            self._rows = rows

        def mappings(self) -> "_Result":
            return self

        def all(self) -> list[dict[str, object]]:
            return self._rows

    class _Session:
        async def execute(self, *args, **kwargs) -> _Result:
            del args, kwargs
            return _Result(
                [
                    {
                        "id": "sig-bootstrap-1",
                        "source": "crypto",
                        "source_item_id": "stable-1",
                        "signal_type": "crypto_worker_15m",
                        "strategy_type": "btc_eth_maker_quote",
                        "market_id": "market-1",
                        "market_question": "Question?",
                        "direction": "no",
                        "entry_price": 0.91,
                        "effective_price": None,
                        "edge_percent": 1.7,
                        "confidence": 0.75,
                        "liquidity": 120.0,
                        "expires_at": now,
                        "status": "pending",
                        "payload_json": {"signal_emitted_at": now.isoformat().replace("+00:00", "Z")},
                        "strategy_context_json": {"source_key": "crypto", "execution_activation": "immediate"},
                        "quality_passed": True,
                        "dedupe_key": "dedupe-1",
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

    publish_mock = AsyncMock(return_value="batch-1")
    monkeypatch.setattr(intent_runtime_module, "AsyncSessionLocal", lambda: _SessionContext())
    monkeypatch.setattr(intent_runtime_module, "publish_signal_batch", publish_mock)

    runtime = intent_runtime_module.IntentRuntime()
    runtime._enqueue_projection = AsyncMock(return_value=None)
    runtime._publish_signal_stats = AsyncMock(return_value=None)

    await runtime.hydrate_from_db()

    assert publish_mock.await_count == 1
    assert publish_mock.await_args.kwargs["event_type"] == "upsert_reactivated"
    assert publish_mock.await_args.kwargs["source"] == "crypto"
    assert publish_mock.await_args.kwargs["signal_ids"] == ["sig-bootstrap-1"]
    assert publish_mock.await_args.kwargs["reason"] == "bootstrap_pending_signals"
    assert runtime._signals_by_id["sig-bootstrap-1"]["runtime_sequence"] == 1


@pytest.mark.asyncio
async def test_intent_runtime_hydrate_skips_stale_terminal_history(monkeypatch) -> None:
    now = utcnow()
    stale_terminal_updated_at = now - intent_runtime_module.timedelta(
        hours=intent_runtime_module._BOOTSTRAP_REACTIVATABLE_LOOKBACK_HOURS + 1
    )

    class _Result:
        def __init__(self, rows: list[dict[str, object]]) -> None:
            self._rows = rows

        def mappings(self) -> "_Result":
            return self

        def all(self) -> list[dict[str, object]]:
            return self._rows

    class _Session:
        async def execute(self, *args, **kwargs) -> _Result:
            del args, kwargs
            return _Result(
                [
                    {
                        "id": "sig-active-old",
                        "source": "crypto",
                        "source_item_id": "stable-active-old",
                        "signal_type": "crypto_worker_15m",
                        "strategy_type": "btc_eth_maker_quote",
                        "market_id": "market-active-old",
                        "market_question": "Active question?",
                        "direction": "no",
                        "entry_price": 0.91,
                        "effective_price": None,
                        "edge_percent": 1.7,
                        "confidence": 0.75,
                        "liquidity": 120.0,
                        "expires_at": now,
                        "status": "pending",
                        "payload_json": {"signal_emitted_at": now.isoformat().replace("+00:00", "Z")},
                        "strategy_context_json": {"source_key": "crypto", "execution_activation": "immediate"},
                        "quality_passed": True,
                        "quality_rejection_reasons": None,
                        "dedupe_key": "dedupe-active-old",
                        "runtime_sequence": None,
                        "created_at": now - intent_runtime_module.timedelta(days=7),
                        "updated_at": now - intent_runtime_module.timedelta(days=7),
                    },
                    {
                        "id": "sig-terminal-recent",
                        "source": "scanner",
                        "source_item_id": "stable-terminal-recent",
                        "signal_type": "scanner_opportunity",
                        "strategy_type": "tail_end_carry",
                        "market_id": "market-terminal-recent",
                        "market_question": "Recent terminal question?",
                        "direction": "buy_yes",
                        "entry_price": 0.93,
                        "effective_price": 0.93,
                        "edge_percent": 2.0,
                        "confidence": 0.8,
                        "liquidity": 180.0,
                        "expires_at": now,
                        "status": "expired",
                        "payload_json": {"signal_emitted_at": now.isoformat().replace("+00:00", "Z")},
                        "strategy_context_json": {"source_key": "scanner", "execution_activation": "ws_current"},
                        "quality_passed": True,
                        "quality_rejection_reasons": None,
                        "dedupe_key": "dedupe-terminal-recent",
                        "runtime_sequence": None,
                        "created_at": now - intent_runtime_module.timedelta(hours=2),
                        "updated_at": now - intent_runtime_module.timedelta(hours=2),
                    },
                    {
                        "id": "sig-terminal-stale",
                        "source": "scanner",
                        "source_item_id": "stable-terminal-stale",
                        "signal_type": "scanner_opportunity",
                        "strategy_type": "tail_end_carry",
                        "market_id": "market-terminal-stale",
                        "market_question": "Stale terminal question?",
                        "direction": "buy_yes",
                        "entry_price": 0.94,
                        "effective_price": 0.94,
                        "edge_percent": 1.5,
                        "confidence": 0.7,
                        "liquidity": 90.0,
                        "expires_at": now,
                        "status": "expired",
                        "payload_json": {"signal_emitted_at": now.isoformat().replace("+00:00", "Z")},
                        "strategy_context_json": {"source_key": "scanner", "execution_activation": "ws_current"},
                        "quality_passed": True,
                        "quality_rejection_reasons": None,
                        "dedupe_key": "dedupe-terminal-stale",
                        "runtime_sequence": None,
                        "created_at": stale_terminal_updated_at,
                        "updated_at": stale_terminal_updated_at,
                    },
                ]
            )

    class _SessionContext:
        async def __aenter__(self) -> _Session:
            return _Session()

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            del exc_type, exc, tb
            return False

    publish_mock = AsyncMock(return_value="batch-1")
    monkeypatch.setattr(intent_runtime_module, "AsyncSessionLocal", lambda: _SessionContext())
    monkeypatch.setattr(intent_runtime_module, "publish_signal_batch", publish_mock)

    runtime = intent_runtime_module.IntentRuntime()
    runtime._enqueue_projection = AsyncMock(return_value=None)
    runtime._publish_signal_stats = AsyncMock(return_value=None)

    await runtime.hydrate_from_db()

    assert set(runtime._signals_by_id.keys()) == {"sig-active-old", "sig-terminal-recent"}
    assert runtime._signals_by_id["sig-active-old"]["runtime_sequence"] == 1
    assert runtime._signals_by_id["sig-terminal-recent"]["status"] == "expired"
    assert "sig-terminal-stale" not in runtime._signals_by_id
    assert publish_mock.await_count == 1
    assert publish_mock.await_args.kwargs["signal_ids"] == ["sig-active-old"]
