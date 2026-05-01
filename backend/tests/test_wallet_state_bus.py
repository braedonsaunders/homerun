"""Tests for the wallet_state_bus pub/sub helpers.

These tests do not require a real Redis.  They cover the two soft-fail
contracts:
  1. The publisher periodically samples the wallet cache; when Redis is
     unavailable, the publish step is silently dropped (no raise).
  2. The subscriber tolerates Redis being down at startup and exposes
     ``is_stale() == True`` until a real heartbeat arrives.
"""

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import asyncio
import json
import time

import pytest

from services import redis_client, wallet_state_bus


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
def _reset_subscriber():
    sub = wallet_state_bus._subscriber
    sub._latest = None
    sub._latest_received_mono = None
    sub._latest_delta = None
    sub._latest_delta_received_mono = None
    sub._delta_count = 0
    sub._delta_event = None
    wallet_state_bus._delta_counter.value = 0
    yield
    sub._latest = None
    sub._latest_received_mono = None
    sub._latest_delta = None
    sub._latest_delta_received_mono = None
    sub._delta_count = 0
    sub._delta_event = None
    wallet_state_bus._delta_counter.value = 0


def test_status_snapshot_when_idle():
    snapshot = wallet_state_bus.status_snapshot()
    assert snapshot["heartbeat_channel"] == "wallet_state:heartbeat"
    assert snapshot["delta_channel"] == "wallet_state:delta"
    assert snapshot["last_heartbeat_age_seconds"] is None
    assert snapshot["last_delta_age_seconds"] is None
    assert snapshot["deltas_received"] == 0
    assert snapshot["stale"] is True
    assert snapshot["latest_heartbeat"] is None
    assert snapshot["latest_delta"] is None


def test_get_latest_heartbeat_returns_none_initially():
    assert wallet_state_bus.get_latest_heartbeat() is None


def test_is_stale_when_no_heartbeat_received():
    assert wallet_state_bus.is_stale() is True


def test_is_stale_threshold_with_recent_heartbeat():
    """A freshly-received heartbeat should not be stale, regardless of payload."""
    wallet_state_bus._subscriber._latest = {"type": "wallet_state_heartbeat"}
    wallet_state_bus._subscriber._latest_received_mono = time.monotonic()
    assert wallet_state_bus.is_stale() is False


def test_is_stale_threshold_with_old_heartbeat():
    """Heartbeats older than the threshold should be marked stale."""
    wallet_state_bus._subscriber._latest = {"type": "wallet_state_heartbeat"}
    # 100 seconds ago — well past the 15s threshold.
    wallet_state_bus._subscriber._latest_received_mono = time.monotonic() - 100.0
    assert wallet_state_bus.is_stale() is True


def test_consume_message_updates_latest():
    sub = wallet_state_bus._subscriber
    payload = {
        "type": "wallet_state_heartbeat",
        "ts": "2026-04-29T00:00:00+00:00",
        "stats": {"open_positions": 3},
    }
    sub._consume_message(json.dumps(payload))
    assert sub.latest() == payload
    assert sub.latest_age_seconds() is not None and sub.latest_age_seconds() < 1.0


def test_consume_message_handles_bytes():
    sub = wallet_state_bus._subscriber
    payload = {"type": "wallet_state_heartbeat", "stats": {}}
    sub._consume_message(json.dumps(payload).encode("utf-8"))
    assert sub.latest() == payload


def test_consume_message_ignores_non_json():
    sub = wallet_state_bus._subscriber
    sub._consume_message("not json at all")
    assert sub.latest() is None


def test_consume_message_ignores_empty():
    sub = wallet_state_bus._subscriber
    sub._consume_message("")
    sub._consume_message(None)
    assert sub.latest() is None


def test_consume_message_ignores_non_dict_payload():
    sub = wallet_state_bus._subscriber
    sub._consume_message(json.dumps([1, 2, 3]))
    assert sub.latest() is None


@pytest.mark.asyncio
async def test_publisher_softfails_when_redis_down():
    """Without Redis, publish() must not raise — the loop should idle silently."""
    # Redis state: explicitly unhealthy.
    assert redis_client.get_client_or_none() is None

    pub = wallet_state_bus._Publisher()
    await pub.start()
    # Let the publisher run for a brief moment to confirm no exception.
    await asyncio.sleep(0.05)
    await pub.stop()
    # No assertion on side-effects: pure soft-fail.


@pytest.mark.asyncio
async def test_publisher_publish_method_is_noop_without_client():
    pub = wallet_state_bus._Publisher()
    payload = pub._build_payload({"open_positions": 0})
    # Must not raise and must not require a client.
    await pub._publish(payload)


def test_publisher_payload_structure():
    pub = wallet_state_bus._Publisher()
    payload = pub._build_payload({"open_positions": 5})
    assert payload["type"] == "wallet_state_heartbeat"
    assert "ts" in payload
    assert "monotonic" in payload
    assert payload["stats"] == {"open_positions": 5}
    assert "deltas_emitted" in payload  # surfaces emit count


def test_consume_delta_updates_state_and_event():
    sub = wallet_state_bus._subscriber
    # Bind a fresh asyncio event so `_consume_delta` can set it.
    sub._delta_event = asyncio.Event()
    payload = {
        "type": "wallet_state_delta",
        "ts": "2026-04-29T00:00:00+00:00",
        "seq": 1,
        "payload": {"kind": "trade", "side": "BUY"},
    }
    sub._consume_delta(json.dumps(payload))
    assert sub.latest_delta() == payload
    assert sub.delta_count() == 1
    assert sub._delta_event.is_set()


def test_consume_delta_handles_bytes():
    sub = wallet_state_bus._subscriber
    sub._delta_event = asyncio.Event()
    payload = {"type": "wallet_state_delta", "payload": {"kind": "order"}}
    sub._consume_delta(json.dumps(payload).encode("utf-8"))
    assert sub.latest_delta() == payload
    assert sub.delta_count() == 1


def test_consume_delta_ignores_invalid_json():
    sub = wallet_state_bus._subscriber
    sub._delta_event = asyncio.Event()
    sub._consume_delta("not json")
    sub._consume_delta("")
    sub._consume_delta(None)
    assert sub.latest_delta() is None
    assert sub.delta_count() == 0


@pytest.mark.asyncio
async def test_publish_delta_softfails_without_redis():
    """publish_delta must be a no-op when Redis is unavailable."""
    assert redis_client.get_client_or_none() is None
    # Must not raise.
    await wallet_state_bus.publish_delta({"kind": "trade", "side": "BUY"})
    # Counter should still increment regardless of Redis availability —
    # it's a process-local emit count, not a successful-publish count.
    initial = wallet_state_bus._delta_counter.value
    await wallet_state_bus.publish_delta({"kind": "order"})
    assert wallet_state_bus._delta_counter.value == initial + 1


def test_get_latest_delta_returns_none_initially():
    assert wallet_state_bus.get_latest_delta() is None
