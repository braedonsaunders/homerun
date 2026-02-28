import asyncio
import json
import sys
from datetime import timezone
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services import event_bus as event_bus_module
from services import event_dispatcher as event_dispatcher_module
from services.data_events import DataEvent, EventType
from utils.utcnow import utcnow


class _InMemoryRedisStreams:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._counter = 0
        self._streams: dict[str, list[tuple[str, str]]] = {}

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        return None

    async def append_json(self, stream: str, payload: dict, *, maxlen: int) -> str:
        async with self._lock:
            self._counter += 1
            entry_id = f"{self._counter}-0"
            entries = self._streams.setdefault(stream, [])
            entries.append((entry_id, json.dumps(payload, separators=(",", ":"), default=str)))
            if len(entries) > maxlen:
                del entries[: len(entries) - maxlen]
            return entry_id

    async def read_raw(self, stream: str, *, last_id: str, block_ms: int, count: int) -> list[tuple[str, str]]:
        deadline = asyncio.get_running_loop().time() + (max(1, block_ms) / 1000.0)
        while True:
            async with self._lock:
                entries = list(self._streams.get(stream, []))
            out = self._entries_after(entries, last_id=last_id, count=count)
            if out:
                return out
            if asyncio.get_running_loop().time() >= deadline:
                return []
            await asyncio.sleep(0.01)

    @staticmethod
    def _entries_after(entries: list[tuple[str, str]], *, last_id: str, count: int) -> list[tuple[str, str]]:
        if not entries:
            return []
        limit = max(1, int(count))
        if last_id == "$":
            return entries[:limit]
        try:
            threshold = int(str(last_id).split("-")[0])
        except Exception:
            threshold = 0
        selected = [entry for entry in entries if int(entry[0].split("-")[0]) > threshold]
        return selected[:limit]


class _NoopSessionContext:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_event_bus_cross_instance_publish_reaches_subscriber(monkeypatch):
    fake_streams = _InMemoryRedisStreams()
    monkeypatch.setattr(event_bus_module, "redis_streams", fake_streams)

    sender = event_bus_module.EventBus()
    receiver = event_bus_module.EventBus()
    received: list[tuple[str, dict]] = []
    received_event = asyncio.Event()

    async def _handler(event_type: str, data: dict) -> None:
        received.append((event_type, data))
        received_event.set()

    receiver.subscribe("trade.executed", _handler)
    await sender.start()
    await receiver.start()
    try:
        await sender.publish("trade.executed", {"trade_id": "t-1", "pnl": 4.25})
        await asyncio.wait_for(received_event.wait(), timeout=1.5)
    finally:
        await sender.stop()
        await receiver.stop()

    assert received == [("trade.executed", {"trade_id": "t-1", "pnl": 4.25})]


@pytest.mark.asyncio
async def test_event_dispatcher_cross_instance_dispatch_reaches_subscribed_strategy(monkeypatch):
    fake_streams = _InMemoryRedisStreams()
    monkeypatch.setattr(event_dispatcher_module, "redis_streams", fake_streams)

    sender = event_dispatcher_module.EventDispatcher()
    receiver = event_dispatcher_module.EventDispatcher()
    received: list[DataEvent] = []
    received_event = asyncio.Event()

    async def _handler(event: DataEvent) -> list:
        received.append(event)
        received_event.set()
        return []

    receiver.subscribe("strategy.news", EventType.PRICE_CHANGE, _handler)
    await sender.start()
    await receiver.start()

    outbound = DataEvent(
        event_type=EventType.PRICE_CHANGE,
        source="scanner_worker",
        timestamp=utcnow().astimezone(timezone.utc),
        market_id="mkt-1",
        token_id="tok-1",
        old_price=0.41,
        new_price=0.43,
        payload={"field": "value"},
    )
    try:
        await sender.dispatch(outbound)
        await asyncio.wait_for(received_event.wait(), timeout=1.5)
    finally:
        await sender.stop()
        await receiver.stop()

    assert len(received) == 1
    inbound = received[0]
    assert inbound.event_type == EventType.PRICE_CHANGE
    assert inbound.source == "scanner_worker"
    assert inbound.market_id == "mkt-1"
    assert inbound.token_id == "tok-1"
    assert inbound.new_price == pytest.approx(0.43)


@pytest.mark.asyncio
@pytest.mark.parametrize("event_type", [EventType.MARKET_RESOLVED, EventType.TRADE_EXECUTION])
async def test_event_dispatcher_cross_instance_fanout_includes_resolution_and_trade_events(monkeypatch, event_type):
    fake_streams = _InMemoryRedisStreams()
    monkeypatch.setattr(event_dispatcher_module, "redis_streams", fake_streams)

    sender = event_dispatcher_module.EventDispatcher()
    receiver = event_dispatcher_module.EventDispatcher()
    received: list[DataEvent] = []
    received_event = asyncio.Event()

    async def _handler(event: DataEvent) -> list:
        received.append(event)
        received_event.set()
        return []

    receiver.subscribe("strategy.lifecycle", event_type, _handler)
    await sender.start()
    await receiver.start()

    outbound = DataEvent(
        event_type=event_type,
        source="ws_feed",
        timestamp=utcnow().astimezone(timezone.utc),
        market_id="mkt-2",
        token_id="tok-2",
        payload={"winner": "YES"} if event_type == EventType.MARKET_RESOLVED else {"price": 0.72, "size": 55.0},
    )
    try:
        await sender.dispatch(outbound)
        await asyncio.wait_for(received_event.wait(), timeout=1.5)
    finally:
        await sender.stop()
        await receiver.stop()

    assert len(received) == 1
    inbound = received[0]
    assert inbound.event_type == event_type
    assert inbound.source == "ws_feed"
    assert inbound.market_id == "mkt-2"
    assert inbound.token_id == "tok-2"


@pytest.mark.asyncio
async def test_event_dispatcher_cross_instance_data_source_update_bridges_signals(monkeypatch):
    fake_streams = _InMemoryRedisStreams()
    monkeypatch.setattr(event_dispatcher_module, "redis_streams", fake_streams)
    monkeypatch.setattr(event_dispatcher_module, "AsyncSessionLocal", lambda: _NoopSessionContext())

    bridged: list[dict] = []
    bridged_event = asyncio.Event()

    async def _fake_bridge(session, opportunities, *, source, sweep_missing=False, **kwargs):
        items = list(opportunities)
        bridged.append(
            {
                "source": source,
                "sweep_missing": sweep_missing,
                "opportunities": len(items),
            }
        )
        bridged_event.set()
        return len(items)

    monkeypatch.setattr(event_dispatcher_module, "bridge_opportunities_to_signals", _fake_bridge)

    sender = event_dispatcher_module.EventDispatcher()
    receiver = event_dispatcher_module.EventDispatcher()

    async def _handler(event: DataEvent) -> list:
        return [object()]

    receiver.subscribe("strategy.data", EventType.DATA_SOURCE_UPDATE, _handler)
    await sender.start()
    await receiver.start()

    outbound = DataEvent(
        event_type=EventType.DATA_SOURCE_UPDATE,
        source="data_source:unit_source",
        timestamp=utcnow().astimezone(timezone.utc),
        payload={"source_slug": "unit_source", "records_count": 4, "run_id": "run_1"},
    )
    try:
        await sender.dispatch(outbound)
        await asyncio.wait_for(bridged_event.wait(), timeout=1.5)
    finally:
        await sender.stop()
        await receiver.stop()

    assert bridged == [{"source": "data_source", "sweep_missing": False, "opportunities": 1}]


@pytest.mark.asyncio
async def test_event_dispatcher_cross_instance_events_update_skips_listener_bridge(monkeypatch):
    fake_streams = _InMemoryRedisStreams()
    monkeypatch.setattr(event_dispatcher_module, "redis_streams", fake_streams)
    monkeypatch.setattr(event_dispatcher_module, "AsyncSessionLocal", lambda: _NoopSessionContext())

    bridge_calls = 0

    async def _fake_bridge(session, opportunities, *, source, sweep_missing=False, **kwargs):
        nonlocal bridge_calls
        bridge_calls += 1
        return len(list(opportunities))

    monkeypatch.setattr(event_dispatcher_module, "bridge_opportunities_to_signals", _fake_bridge)

    sender = event_dispatcher_module.EventDispatcher()
    receiver = event_dispatcher_module.EventDispatcher()

    async def _handler(event: DataEvent) -> list:
        return [object()]

    receiver.subscribe("strategy.events", EventType.EVENTS_UPDATE, _handler)
    await sender.start()
    await receiver.start()

    outbound = DataEvent(
        event_type=EventType.EVENTS_UPDATE,
        source="events_worker",
        timestamp=utcnow().astimezone(timezone.utc),
        payload={"signals": [], "summary": {"total": 0, "critical": 0, "by_type": {}}},
    )
    try:
        await sender.dispatch(outbound)
        await asyncio.sleep(0.2)
    finally:
        await sender.stop()
        await receiver.stop()

    assert bridge_calls == 0


@pytest.mark.asyncio
async def test_event_dispatcher_unowned_remote_opportunities_fail_fast_in_production(monkeypatch):
    fake_streams = _InMemoryRedisStreams()
    monkeypatch.setattr(event_dispatcher_module, "redis_streams", fake_streams)
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("EVENT_DISPATCHER_FAIL_ON_UNOWNED_REMOTE_OPPS", raising=False)

    sender = event_dispatcher_module.EventDispatcher()
    receiver = event_dispatcher_module.EventDispatcher()

    async def _handler(event: DataEvent) -> list:
        return [object()]

    receiver.subscribe("strategy.price", EventType.PRICE_CHANGE, _handler)
    await sender.start()
    await receiver.start()

    outbound = DataEvent(
        event_type=EventType.PRICE_CHANGE,
        source="ws_feed",
        timestamp=utcnow().astimezone(timezone.utc),
        token_id="tok-fast-fail",
        old_price=0.40,
        new_price=0.45,
    )
    try:
        await sender.dispatch(outbound)
        for _ in range(20):
            task = receiver._listener_task
            if task is not None and task.done():
                break
            await asyncio.sleep(0.05)
        task = receiver._listener_task
        assert task is not None
        assert task.done()
        assert isinstance(task.exception(), RuntimeError)
    finally:
        await sender.stop()
        await receiver.stop()


@pytest.mark.asyncio
async def test_event_dispatcher_timeout_cancels_handler(monkeypatch):
    fake_streams = _InMemoryRedisStreams()
    monkeypatch.setattr(event_dispatcher_module, "redis_streams", fake_streams)

    dispatcher = event_dispatcher_module.EventDispatcher()
    handler_started = asyncio.Event()
    handler_release = asyncio.Event()
    handler_cancelled = False

    async def _slow_handler(event: DataEvent) -> list:
        nonlocal handler_cancelled
        handler_started.set()
        try:
            await handler_release.wait()
        except asyncio.CancelledError:
            handler_cancelled = True
            raise
        return []

    dispatcher.subscribe("strategy.timeout", EventType.PRICE_CHANGE, _slow_handler)
    outbound = DataEvent(
        event_type=EventType.PRICE_CHANGE,
        source="scanner_worker",
        timestamp=utcnow().astimezone(timezone.utc),
        token_id="tok-timeout",
        old_price=0.5,
        new_price=0.55,
    )
    try:
        opportunities = await dispatcher.dispatch(outbound, handler_timeout_seconds=1.0)
        assert opportunities == []
        await asyncio.wait_for(handler_started.wait(), timeout=1.0)
        await asyncio.sleep(0)
        assert handler_cancelled
        assert len(dispatcher._timed_out_handler_tasks) == 0
    finally:
        handler_release.set()
        await dispatcher.stop()


@pytest.mark.asyncio
async def test_event_dispatcher_stop_keeps_pending_timed_out_handlers_tracked(monkeypatch):
    fake_streams = _InMemoryRedisStreams()
    monkeypatch.setattr(event_dispatcher_module, "redis_streams", fake_streams)

    dispatcher = event_dispatcher_module.EventDispatcher()
    dispatcher._handler_cancel_grace_seconds = 0.05
    handler_started = asyncio.Event()
    handler_release = asyncio.Event()
    cancel_count = 0

    async def _stubborn_handler(event: DataEvent) -> list:
        nonlocal cancel_count
        handler_started.set()
        while True:
            try:
                await handler_release.wait()
                return []
            except asyncio.CancelledError:
                cancel_count += 1
                continue

    dispatcher.subscribe("strategy.stubborn-timeout", EventType.PRICE_CHANGE, _stubborn_handler)
    outbound = DataEvent(
        event_type=EventType.PRICE_CHANGE,
        source="scanner_worker",
        timestamp=utcnow().astimezone(timezone.utc),
        token_id="tok-stubborn-timeout",
        old_price=0.5,
        new_price=0.55,
    )
    try:
        opportunities = await dispatcher.dispatch(outbound, handler_timeout_seconds=1.0)
        assert opportunities == []
        await asyncio.wait_for(handler_started.wait(), timeout=1.0)
        for _ in range(40):
            if len(dispatcher._timed_out_handler_tasks) > 0:
                break
            await asyncio.sleep(0.01)
        assert len(dispatcher._timed_out_handler_tasks) > 0

        await dispatcher.stop()
        assert len(dispatcher._timed_out_handler_tasks) > 0
    finally:
        handler_release.set()
        for _ in range(50):
            if len(dispatcher._timed_out_handler_tasks) == 0:
                break
            await asyncio.sleep(0.01)
        await dispatcher.stop()

    assert cancel_count >= 1
    assert len(dispatcher._timed_out_handler_tasks) == 0
