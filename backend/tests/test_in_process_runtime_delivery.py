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

    batch_id = await publish_signal_batch(
        event_type="upsert_insert",
        source="crypto",
        signal_ids=["sig-1", "sig-2", "sig-1"],
        trigger="strategy_signal_bridge",
        signal_snapshots={
            "sig-1": {"id": "sig-1", "source": "crypto"},
            "sig-2": {"id": "sig-2", "source": "crypto"},
        },
    )

    assert isinstance(batch_id, str)

    general_payload = await wait_for_signal_batch(lane="general", timeout_seconds=0.5)
    crypto_payload = await wait_for_signal_batch(lane="crypto", timeout_seconds=0.5)

    assert general_payload is not None
    assert crypto_payload is not None
    assert general_payload["event_type"] == "runtime_signal_batch"
    assert crypto_payload["event_type"] == "runtime_signal_batch"
    assert general_payload["source"] == "crypto"
    assert crypto_payload["source"] == "crypto"
    assert general_payload["source_signal_ids"]["crypto"] == ["sig-1", "sig-2"]
    assert crypto_payload["source_signal_ids"]["crypto"] == ["sig-1", "sig-2"]
    assert get_queue_depth() == {"general": 0, "crypto": 0}


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
                        "source": "scanner",
                        "source_item_id": "stable-1",
                        "signal_type": "scanner_opportunity",
                        "strategy_type": "tail",
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
                        "strategy_context_json": {},
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
    assert publish_mock.await_args.kwargs["source"] == "scanner"
    assert publish_mock.await_args.kwargs["signal_ids"] == ["sig-bootstrap-1"]
    assert publish_mock.await_args.kwargs["reason"] == "bootstrap_pending_signals"
    assert runtime._signals_by_id["sig-bootstrap-1"]["runtime_sequence"] == 1
