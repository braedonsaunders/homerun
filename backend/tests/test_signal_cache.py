"""Tests for the Redis-pushed signal cache.

These tests do NOT require a running Redis.  They exercise the
in-memory ``SignalCache`` directly: LRU eviction, per-trader consumed
sets, filter/sort semantics, soft-fail behavior of the subscriber when
Redis is unavailable.
"""

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from services import redis_client, signal_cache


def _make_snapshot(
    *,
    signal_id: str,
    source: str = "scanner",
    market_id: str = "market-1",
    runtime_sequence: int | None = 1,
    status: str = "pending",
    expires_at: datetime | None = None,
    created_at: datetime | None = None,
) -> signal_cache.SignalSnapshot:
    return signal_cache.SignalSnapshot(
        id=signal_id,
        source=source,
        source_item_id=None,
        signal_type="entry",
        strategy_type=None,
        market_id=market_id,
        market_question=None,
        direction="buy_yes",
        entry_price=0.5,
        effective_price=None,
        edge_percent=2.5,
        confidence=0.7,
        liquidity=1000.0,
        expires_at=expires_at,
        status=status,
        quality_passed=True,
        dedupe_key=f"dedupe-{signal_id}",
        runtime_sequence=runtime_sequence,
        created_at=created_at or datetime(2026, 4, 30, tzinfo=timezone.utc),
        updated_at=None,
    )


@pytest.fixture(autouse=True)
def _fresh_cache():
    # Replace the singleton with a small-capacity instance so eviction
    # tests are tractable.
    signal_cache._cache = signal_cache.SignalCache(max_entries=100)
    yield
    signal_cache._cache = None


@pytest.fixture(autouse=True)
def _reset_redis_state():
    redis_client._state.healthy = False
    redis_client._state.started = False
    redis_client._state.client = None
    redis_client._state.pool = None
    yield


def test_upsert_and_lookup_by_id():
    cache = signal_cache.get_signal_cache()
    snap = _make_snapshot(signal_id="s1")
    cache.upsert(snap)
    assert cache.get_signal("s1") is snap
    assert cache.get_signal("missing") is None


def test_upsert_replaces_existing():
    cache = signal_cache.get_signal_cache()
    cache.upsert(_make_snapshot(signal_id="s1", runtime_sequence=1))
    cache.upsert(_make_snapshot(signal_id="s1", runtime_sequence=99))
    assert cache.get_signal("s1").runtime_sequence == 99


def test_lru_eviction_when_full():
    cache = signal_cache.SignalCache(max_entries=3)
    cache.upsert(_make_snapshot(signal_id="a"))
    cache.upsert(_make_snapshot(signal_id="b"))
    cache.upsert(_make_snapshot(signal_id="c"))
    cache.upsert(_make_snapshot(signal_id="d"))
    # 'a' is the oldest — evicted.
    assert cache.get_signal("a") is None
    assert cache.get_signal("b") is not None
    assert cache.get_signal("c") is not None
    assert cache.get_signal("d") is not None
    snapshot = cache.status_snapshot()
    assert snapshot["signals_evicted_total"] == 1


def test_get_unconsumed_filters_by_source():
    cache = signal_cache.get_signal_cache()
    cache.upsert(_make_snapshot(signal_id="s1", source="scanner"))
    cache.upsert(_make_snapshot(signal_id="s2", source="news"))
    cache.upsert(_make_snapshot(signal_id="s3", source="crypto"))
    cache.hydrate_trader_consumed_ids("trader-x", [])
    results = cache.get_unconsumed_signals(
        trader_id="trader-x",
        sources={"scanner", "crypto"},
    )
    ids = {s.id for s in results}
    assert ids == {"s1", "s3"}


def test_get_unconsumed_filters_by_status():
    cache = signal_cache.get_signal_cache()
    cache.upsert(_make_snapshot(signal_id="s1", status="pending"))
    cache.upsert(_make_snapshot(signal_id="s2", status="executed"))
    cache.hydrate_trader_consumed_ids("trader-x", [])
    results = cache.get_unconsumed_signals(trader_id="trader-x")
    assert {s.id for s in results} == {"s1"}


def test_get_unconsumed_filters_by_cursor():
    cache = signal_cache.get_signal_cache()
    cache.upsert(_make_snapshot(signal_id="s1", runtime_sequence=10))
    cache.upsert(_make_snapshot(signal_id="s2", runtime_sequence=20))
    cache.upsert(_make_snapshot(signal_id="s3", runtime_sequence=30))
    cache.hydrate_trader_consumed_ids("trader-x", [])
    results = cache.get_unconsumed_signals(
        trader_id="trader-x",
        cursor_runtime_sequence=15,
    )
    assert {s.id for s in results} == {"s2", "s3"}


def test_get_unconsumed_skips_consumed():
    cache = signal_cache.get_signal_cache()
    cache.upsert(_make_snapshot(signal_id="s1"))
    cache.upsert(_make_snapshot(signal_id="s2"))
    cache.hydrate_trader_consumed_ids("trader-x", ["s1"])
    results = cache.get_unconsumed_signals(trader_id="trader-x")
    assert {s.id for s in results} == {"s2"}


def test_get_unconsumed_filters_expired():
    cache = signal_cache.get_signal_cache()
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    cache.upsert(_make_snapshot(signal_id="s_expired", expires_at=past))
    cache.upsert(_make_snapshot(signal_id="s_live", expires_at=future))
    cache.upsert(_make_snapshot(signal_id="s_no_expiry", expires_at=None))
    cache.hydrate_trader_consumed_ids("trader-x", [])
    results = cache.get_unconsumed_signals(trader_id="trader-x")
    assert {s.id for s in results} == {"s_live", "s_no_expiry"}


def test_get_unconsumed_sorted_by_runtime_sequence():
    cache = signal_cache.get_signal_cache()
    # Insert out of order.
    cache.upsert(_make_snapshot(signal_id="s_third", runtime_sequence=30))
    cache.upsert(_make_snapshot(signal_id="s_first", runtime_sequence=10))
    cache.upsert(_make_snapshot(signal_id="s_second", runtime_sequence=20))
    cache.hydrate_trader_consumed_ids("trader-x", [])
    results = cache.get_unconsumed_signals(trader_id="trader-x")
    assert [s.id for s in results] == ["s_first", "s_second", "s_third"]


def test_get_unconsumed_respects_limit():
    cache = signal_cache.get_signal_cache()
    for i in range(20):
        cache.upsert(_make_snapshot(signal_id=f"s{i}", runtime_sequence=i))
    cache.hydrate_trader_consumed_ids("trader-x", [])
    results = cache.get_unconsumed_signals(trader_id="trader-x", limit=5)
    assert len(results) == 5


def test_mark_consumed_persists_for_subsequent_lookups():
    cache = signal_cache.get_signal_cache()
    cache.upsert(_make_snapshot(signal_id="s1"))
    cache.upsert(_make_snapshot(signal_id="s2"))
    cache.hydrate_trader_consumed_ids("trader-x", [])
    cache.mark_consumed("trader-x", "s1")
    results = cache.get_unconsumed_signals(trader_id="trader-x")
    assert {s.id for s in results} == {"s2"}


def test_mark_consumed_is_idempotent():
    cache = signal_cache.get_signal_cache()
    cache.upsert(_make_snapshot(signal_id="s1"))
    cache.mark_consumed("trader-x", "s1")
    cache.mark_consumed("trader-x", "s1")
    cache.mark_consumed("trader-x", "s1")
    snapshot = cache.status_snapshot()
    # Three calls but only one consumption recorded.
    assert snapshot["consumptions_recorded_total"] == 1


def test_per_trader_consumption_isolation():
    cache = signal_cache.get_signal_cache()
    cache.upsert(_make_snapshot(signal_id="s1"))
    cache.hydrate_trader_consumed_ids("trader-A", [])
    cache.hydrate_trader_consumed_ids("trader-B", [])
    cache.mark_consumed("trader-A", "s1")
    a_results = cache.get_unconsumed_signals(trader_id="trader-A")
    b_results = cache.get_unconsumed_signals(trader_id="trader-B")
    assert a_results == []
    assert {s.id for s in b_results} == {"s1"}


def test_consumed_ring_evicts_oldest():
    cache = signal_cache.SignalCache(max_entries=10000)
    # Replace the per-trader ring cap with a tiny limit for the test.
    # We can't easily monkeypatch _MAX_CONSUMED_RING_PER_TRADER so we
    # rely on the public behavior: the ring is bounded.  Drive enough
    # consumptions to overflow the default 1000 cap.
    for i in range(1100):
        cache.upsert(_make_snapshot(signal_id=f"s{i}"))
        cache.mark_consumed("trader-x", f"s{i}")
    # The first 100 should have rolled off the consumed ring.
    assert len(cache._consumed_set["trader-x"]) <= 1000


def test_hydrate_trader_consumed_ids_marks_hydrated():
    cache = signal_cache.get_signal_cache()
    assert cache.is_trader_hydrated("trader-x") is False
    cache.hydrate_trader_consumed_ids("trader-x", ["a", "b", "c"])
    assert cache.is_trader_hydrated("trader-x") is True


def test_from_redis_payload_parses_iso_dates():
    payload = {
        "id": "s1",
        "source": "scanner",
        "signal_type": "entry",
        "market_id": "m1",
        "status": "pending",
        "dedupe_key": "k1",
        "created_at": "2026-04-30T12:34:56+00:00",
        "expires_at": "2026-05-01T00:00:00Z",
    }
    snap = signal_cache.SignalSnapshot.from_redis_payload(payload)
    assert snap is not None
    assert snap.id == "s1"
    assert snap.created_at.year == 2026
    assert snap.expires_at.year == 2026


def test_from_redis_payload_returns_none_for_invalid():
    assert signal_cache.SignalSnapshot.from_redis_payload({}) is None
    assert signal_cache.SignalSnapshot.from_redis_payload({"id": ""}) is None
    assert signal_cache.SignalSnapshot.from_redis_payload({"id": "x"}) is None  # missing source


def test_status_snapshot_initial_values():
    cache = signal_cache.get_signal_cache()
    s = cache.status_snapshot()
    assert s["size"] == 0
    assert s["signals_added_total"] == 0
    assert s["lookups_total"] == 0
    assert s["hit_rate"] is None
    assert s["last_received_age_seconds"] is None


def test_status_snapshot_tracks_hit_rate():
    cache = signal_cache.get_signal_cache()
    cache.upsert(_make_snapshot(signal_id="s1"))
    cache.hydrate_trader_consumed_ids("trader-x", [])
    cache.get_unconsumed_signals(trader_id="trader-x")  # hit
    # Lookup with no signals matching (different source) → miss.
    cache.get_unconsumed_signals(
        trader_id="trader-x",
        sources={"nonexistent_source"},
    )
    s = cache.status_snapshot()
    assert s["lookups_total"] == 2
    assert s["lookups_hit"] == 1
    assert s["hit_rate"] == 0.5


@pytest.mark.asyncio
async def test_subscriber_softfails_when_redis_down():
    assert redis_client.get_client_or_none() is None
    await signal_cache.start_subscriber()
    await asyncio.sleep(0.05)
    # Subscriber must be running but unable to receive.
    assert signal_cache._subscriber._task is not None
    await signal_cache.stop_subscriber()
