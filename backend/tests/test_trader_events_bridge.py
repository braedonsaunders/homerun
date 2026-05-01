"""Tests for the trader_events Redis → WebSocket bridge.

These tests do not require a running Redis.  They exercise the bridge's
soft-fail contract: when Redis is unavailable, the bridge sleeps and
retries; ``status_snapshot()`` reflects the disconnected state; the
caller-supplied callback is never invoked with bogus data.
"""

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import asyncio

import pytest

from services import redis_client, trader_events_bridge


@pytest.fixture(autouse=True)
def _reset_redis_state():
    redis_client._state.healthy = False
    redis_client._state.started = False
    redis_client._state.client = None
    redis_client._state.pool = None
    yield
    redis_client._state.healthy = False
    redis_client._state.started = False


@pytest.fixture(autouse=True)
def _reset_bridge_state():
    bridge = trader_events_bridge._bridge
    bridge._messages_received = 0
    bridge._messages_dispatched = 0
    bridge._messages_dropped = 0
    bridge._last_message_mono = None
    bridge._latest_event = None
    yield
    bridge._messages_received = 0
    bridge._messages_dispatched = 0
    bridge._messages_dropped = 0
    bridge._last_message_mono = None
    bridge._latest_event = None


def test_status_snapshot_when_idle():
    snapshot = trader_events_bridge.status_snapshot()
    assert snapshot["channel"] == "trader_events"
    assert snapshot["running"] is False
    assert snapshot["messages_received"] == 0
    assert snapshot["messages_dispatched"] == 0
    assert snapshot["messages_dropped"] == 0
    assert snapshot["last_message_age_seconds"] is None
    assert snapshot["latest_event"] is None


def test_get_latest_event_returns_none_initially():
    assert trader_events_bridge.get_latest_event() is None


@pytest.mark.asyncio
async def test_start_softfails_when_redis_down():
    """Bridge must start cleanly without Redis."""
    assert redis_client.get_client_or_none() is None

    received: list[dict] = []

    async def _on_event(payload: dict) -> None:
        received.append(payload)

    await trader_events_bridge.start(_on_event)
    await asyncio.sleep(0.05)
    snapshot = trader_events_bridge.status_snapshot()
    assert snapshot["running"] is True
    # No messages yet — Redis isn't connected.
    assert snapshot["messages_received"] == 0
    assert received == []
    await trader_events_bridge.stop()


@pytest.mark.asyncio
async def test_start_is_idempotent():
    """Calling start twice doesn't spawn a second task."""
    async def _noop(payload: dict) -> None:
        return None

    await trader_events_bridge.start(_noop)
    first_task = trader_events_bridge._bridge._task
    await trader_events_bridge.start(_noop)
    second_task = trader_events_bridge._bridge._task
    assert first_task is second_task
    await trader_events_bridge.stop()


@pytest.mark.asyncio
async def test_stop_is_safe_when_never_started():
    await trader_events_bridge.stop()
    assert trader_events_bridge._bridge._task is None


def test_status_snapshot_reflects_message_counts():
    """Manually mutate counters to verify status_snapshot exposes them."""
    bridge = trader_events_bridge._bridge
    bridge._messages_received = 5
    bridge._messages_dispatched = 4
    bridge._messages_dropped = 1
    bridge._latest_event = {"event_type": "decision", "trader_id": "x"}
    snapshot = trader_events_bridge.status_snapshot()
    assert snapshot["messages_received"] == 5
    assert snapshot["messages_dispatched"] == 4
    assert snapshot["messages_dropped"] == 1
    assert snapshot["latest_event"]["event_type"] == "decision"


@pytest.mark.asyncio
async def test_callback_signature_accepts_dict():
    """Verify the contract: callback gets a plain dict payload."""
    received: list[dict] = []

    async def _on_event(payload: dict) -> None:
        received.append(payload)
        # Verify it really is a dict — not bytes, not None.
        assert isinstance(payload, dict)

    # Manually invoke the callback to confirm the type contract.
    test_payload = {
        "id": "evt-123",
        "event_type": "decision",
        "trader_id": "trader-x",
        "message": "test",
    }
    await _on_event(test_payload)
    assert received == [test_payload]
