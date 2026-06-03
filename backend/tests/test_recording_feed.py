"""RecordingFeedManager: the isolated broad-recording WS pool.

Guards the institutional split — recording rides its OWN cache + a sharded
connection pool, never the trading feed — so broad recording subscriptions can
never load the orchestrator's low-latency trading socket.
"""
from __future__ import annotations

import pytest

from services.recording_feed import RecordingFeedManager


@pytest.mark.asyncio
async def test_recording_pool_shards_disjointly_and_unions():
    RecordingFeedManager.reset_instance()
    rfm = RecordingFeedManager(pool_size=4)
    toks = [f"tok{i}" for i in range(200)]
    n = await rfm.subscribe(toks)
    assert n == 200
    assert rfm.get_subscribed_assets() == set(toks)
    # Every token lands on exactly one shard — no connection carries duplicates.
    seen: set[str] = set()
    for feed in rfm._feeds:
        shard = {str(x) for x in feed._subscribed_assets}
        assert not (shard & seen)  # shards are disjoint
        seen |= shard
    assert seen == set(toks)


@pytest.mark.asyncio
async def test_recording_pool_idempotent_consistent_shard():
    RecordingFeedManager.reset_instance()
    rfm = RecordingFeedManager(pool_size=4)
    await rfm.subscribe(["a", "b", "c", "d", "e"])
    snap = [set(f._subscribed_assets) for f in rfm._feeds]
    await rfm.subscribe(["a", "b", "c", "d", "e"])  # re-subscribe is a no-op
    assert [set(f._subscribed_assets) for f in rfm._feeds] == snap


def test_recording_pool_uses_isolated_cache():
    RecordingFeedManager.reset_instance()
    rfm = RecordingFeedManager(pool_size=3)
    # All pool connections share ONE recording cache — and it is NOT the trading
    # feed's cache (constructed fresh here), so recording never touches trading.
    for feed in rfm._feeds:
        assert feed._cache is rfm.cache


def test_recording_pool_status_shape():
    RecordingFeedManager.reset_instance()
    rfm = RecordingFeedManager(pool_size=2)
    st = rfm.status()
    assert st["started"] is False
    assert st["pool_size"] == 2
    assert st["subscribed_tokens"] == 0
    assert len(st["per_connection"]) == 2


def test_record_rest_baseline_lands_book_in_cache():
    """The REST-baseline path (record_rest_baseline) must route a fetched book
    through the SAME parse -> cache -> record_book pipeline the WS uses, so quiet
    markets get a recorded baseline. Verify a POST /books-shaped book lands in
    the recording cache."""
    RecordingFeedManager.reset_instance()
    rfm = RecordingFeedManager(pool_size=2)
    book = {
        "asset_id": "tokX",
        "bids": [{"price": "0.40", "size": "100"}],
        "asks": [{"price": "0.42", "size": "80"}],
        "timestamp": "1700000000000",
    }
    n = rfm.record_rest_baseline({"tokX": book})
    assert n == 1
    assert rfm.cache.get_order_book("tokX") is not None  # landed in recording cache
