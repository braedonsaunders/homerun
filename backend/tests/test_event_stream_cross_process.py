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
