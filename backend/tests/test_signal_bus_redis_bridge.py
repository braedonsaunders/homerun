"""Tests for the cross-plane trade-signal Redis bridge.

The bridge subscribes to Redis pub/sub channels written by the news
plane's ``signal_bus._publish_redis`` and re-publishes onto the local
``event_bus`` for the trading plane's orchestrator + fast trader.

These tests cover the dedup ring (which prevents the trading plane from
double-firing its own locally-emitted signals) and the bridge handlers'
soft-fail semantics.
"""

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import asyncio

import pytest

from services import redis_client, signal_bus_redis_bridge
from services.event_bus import event_bus


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
    """Reset the dedup ring and bridge bookkeeping between tests."""
    signal_bus_redis_bridge._dedup._seen_at.clear()
    bridge = signal_bus_redis_bridge._bridge
    bridge._messages_received = 0
    bridge._messages_bridged = 0
    bridge._messages_skipped_dup = 0
    bridge._last_message_mono = None
    yield
    signal_bus_redis_bridge._dedup._seen_at.clear()


@pytest.fixture(autouse=True)
def _fresh_runtime_queues():
    """Give each bridge test its own loop-bound runtime_signal_queue lanes.

    These tests call ``bridge._bridge_emission`` / ``_bridge_batch`` ->
    ``publish_signal_batch``, which enqueues onto the module-level
    ``runtime_signal_queue._queues`` (per-lane ``asyncio.Queue`` objects). Left
    unswapped, those leftover items — bound to this test's event loop — leak into
    later tests (e.g. ``test_trader_orchestrator_worker`` reads
    ``get_queue_depth`` in ``run_worker_loop``). Swap in fresh lanes and restore
    the originals afterward (mirrors the fixture in
    ``test_in_process_runtime_delivery``).
    """
    from services import runtime_signal_queue as _rsq

    original = _rsq._queues
    _rsq._queues = {lane: asyncio.Queue() for lane in _rsq._LANES}
    try:
        yield
    finally:
        _rsq._queues = original


@pytest.mark.asyncio
async def test_dedup_remembers_added_ids():
    dedup = signal_bus_redis_bridge._dedup
    await dedup.add("sig-1")
    await dedup.add("sig-2")
    assert await dedup.seen("sig-1") is True
    assert await dedup.seen("sig-2") is True
    assert await dedup.seen("sig-unseen") is False


@pytest.mark.asyncio
async def test_dedup_handles_empty_id():
    dedup = signal_bus_redis_bridge._dedup
    await dedup.add("")
    assert await dedup.seen("") is False  # empty IDs are ignored


def test_dedup_ttl_stays_below_execution_wake_budget():
    assert 0 < signal_bus_redis_bridge._DEDUP_TTL_SECONDS <= 0.1


@pytest.mark.asyncio
async def test_dedup_evicts_oldest_when_full(monkeypatch):
    """The dedup ring is memory-bounded; stale IDs are swept when oversized."""
    monkeypatch.setattr(signal_bus_redis_bridge, "_DEDUP_WINDOW_SIZE", 4)
    fresh = signal_bus_redis_bridge._SignalDedup()
    monkeypatch.setattr(signal_bus_redis_bridge, "_dedup", fresh)
    # Add 4 items, then backdate their timestamps so they look expired.
    for i in range(4):
        await fresh.add(f"sig-{i}")
        fresh._seen_at[f"sig-{i}"] = 0.0  # far in the past, past TTL
    # Adding a 5th item (len > WINDOW_SIZE) triggers the sweep that removes
    # items past _DEDUP_TTL_SECONDS — the backdated entries are evicted.
    await fresh.add("sig-4")
    assert await fresh.seen("sig-0") is False
    assert await fresh.seen("sig-1") is False
    assert await fresh.seen("sig-4") is True


@pytest.mark.asyncio
async def test_mark_locally_emitted_string():
    await signal_bus_redis_bridge.mark_locally_emitted("sig-abc")
    assert await signal_bus_redis_bridge._dedup.seen("sig-abc") is True


@pytest.mark.asyncio
async def test_mark_locally_emitted_list():
    await signal_bus_redis_bridge.mark_locally_emitted(["sig-1", "sig-2", "sig-3"])
    for sid in ("sig-1", "sig-2", "sig-3"):
        assert await signal_bus_redis_bridge._dedup.seen(sid) is True


@pytest.mark.asyncio
async def test_bridge_emission_skips_duplicate(monkeypatch):
    """If a signal_id is already in the dedup ring, the bridge skips it."""
    bridge = signal_bus_redis_bridge._bridge
    await signal_bus_redis_bridge._dedup.add("sig-dup")

    fired: list[tuple[str, dict]] = []

    async def fake_publish(event_type, payload):
        fired.append((event_type, payload))

    monkeypatch.setattr(event_bus, "publish", fake_publish)
    await bridge._bridge_emission({"signal_id": "sig-dup"})
    assert fired == []
    assert bridge._messages_skipped_dup == 1


@pytest.mark.asyncio
async def test_bridge_emission_fires_for_new_signal(monkeypatch):
    bridge = signal_bus_redis_bridge._bridge

    fired: list[tuple[str, dict]] = []

    async def fake_publish(event_type, payload):
        fired.append((event_type, payload))

    monkeypatch.setattr(event_bus, "publish", fake_publish)
    await bridge._bridge_emission({"signal_id": "sig-new", "source": "news"})
    assert len(fired) == 1
    assert fired[0][0] == "trade_signal_emission"
    assert fired[0][1]["signal_id"] == "sig-new"
    # Should now be in the dedup ring.
    assert await signal_bus_redis_bridge._dedup.seen("sig-new") is True
    assert bridge._messages_bridged == 1


@pytest.mark.asyncio
async def test_bridge_batch_filters_already_seen_ids(monkeypatch):
    """In a batch, only the IDs we haven't seen are forwarded."""
    bridge = signal_bus_redis_bridge._bridge
    # Mark 2 of 4 as already seen locally.
    await signal_bus_redis_bridge._dedup.add("sig-a")
    await signal_bus_redis_bridge._dedup.add("sig-b")

    fired: list[tuple[str, dict]] = []

    async def fake_publish(event_type, payload):
        fired.append((event_type, payload))

    monkeypatch.setattr(event_bus, "publish", fake_publish)
    await bridge._bridge_batch({
        "signal_ids": ["sig-a", "sig-b", "sig-c", "sig-d"],
        "signal_count": 4,
        "event_type": "upsert_insert",
        "signal_snapshots": {
            "sig-a": {"id": "sig-a"},
            "sig-b": {"id": "sig-b"},
            "sig-c": {"id": "sig-c"},
            "sig-d": {"id": "sig-d"},
        },
    })
    assert len(fired) == 1
    bridged_ids = fired[0][1]["signal_ids"]
    assert set(bridged_ids) == {"sig-c", "sig-d"}
    assert fired[0][1]["signal_count"] == 2
    assert set(fired[0][1]["signal_snapshots"].keys()) == {"sig-c", "sig-d"}


@pytest.mark.asyncio
async def test_bridge_batch_all_dup_skips_publish(monkeypatch):
    bridge = signal_bus_redis_bridge._bridge
    await signal_bus_redis_bridge._dedup.add("sig-a")
    await signal_bus_redis_bridge._dedup.add("sig-b")

    fired: list[tuple[str, dict]] = []

    async def fake_publish(event_type, payload):
        fired.append((event_type, payload))

    monkeypatch.setattr(event_bus, "publish", fake_publish)
    await bridge._bridge_batch({"signal_ids": ["sig-a", "sig-b"]})
    assert fired == []
    assert bridge._messages_skipped_dup == 1


@pytest.mark.asyncio
async def test_bridge_batch_handles_empty_signal_ids(monkeypatch):
    bridge = signal_bus_redis_bridge._bridge

    fired: list[tuple[str, dict]] = []

    async def fake_publish(event_type, payload):
        fired.append((event_type, payload))

    monkeypatch.setattr(event_bus, "publish", fake_publish)
    await bridge._bridge_batch({"signal_ids": [], "event_type": "noop"})
    # Empty payload still publishes (preserves existing event_bus
    # contract — consumers can decide whether to act).
    assert len(fired) == 1


@pytest.mark.asyncio
async def test_signal_batch_snapshots_are_restamped_to_publish_time(monkeypatch):
    from services import signal_bus

    published: list[tuple[str, dict]] = []

    async def fake_publish_redis(channel, payload):
        published.append((channel, payload))

    async def fake_event_publish(_event_type, _payload):
        return None

    async def fake_mark_locally_emitted(_signal_ids):
        return None

    monkeypatch.setattr(signal_bus, "_publish_redis", fake_publish_redis)
    monkeypatch.setattr(signal_bus.event_bus, "publish", fake_event_publish)
    monkeypatch.setattr(signal_bus_redis_bridge, "mark_locally_emitted", fake_mark_locally_emitted)

    await signal_bus._publish_trade_signal_batch(
        event_type="upsert_update",
        source="scanner",
        signal_ids=["sig-restamp"],
        signal_snapshots={
            "sig-restamp": {
                "id": "sig-restamp",
                "source": "scanner",
                "market_id": "market-1",
                "payload_json": {"signal_emitted_at": "2000-01-01T00:00:00Z"},
            }
        },
    )

    batch_payload = next(payload for channel, payload in published if channel == signal_bus.SIGNAL_BATCH_CHANNEL)
    snapshot = batch_payload["signal_snapshots"]["sig-restamp"]
    assert snapshot["payload_json"]["signal_emitted_at"] == batch_payload["emitted_at"]


def test_status_snapshot_initial():
    snapshot = signal_bus_redis_bridge.status_snapshot()
    assert snapshot["running"] is False
    assert snapshot["messages_received"] == 0
    assert snapshot["messages_bridged"] == 0
    assert snapshot["messages_skipped_dup"] == 0
    assert snapshot["last_message_age_seconds"] is None


@pytest.mark.asyncio
async def test_start_softfails_when_redis_down():
    """The bridge must start cleanly even when Redis is unavailable.

    The internal task should sit in the reconnect loop and not raise.
    """
    assert redis_client.get_client_or_none() is None
    await signal_bus_redis_bridge.start()
    # Let the task spin once.
    await asyncio.sleep(0.05)
    snapshot = signal_bus_redis_bridge.status_snapshot()
    assert snapshot["running"] is True
    await signal_bus_redis_bridge.stop()
