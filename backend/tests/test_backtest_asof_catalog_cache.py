"""As-of catalog cache: a backtest replays a frozen historical window, so the
recorded catalog snapshots for a fully-past window are immutable and memoizable.
Guards the institutional perf fix (avoid re-reading ~300 catalog.snapshot
envelopes / ~225s on every backtest of the same window) AND its safety rail
(never cache a window that is still being recorded)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import services.strategy_backtester as sb


def _patch_bus(monkeypatch, markets, counter):
    import services.recorded_event_bus as reb

    def _replay(window):
        counter["n"] += 1

        async def _gen():
            yield SimpleNamespace(payload={"markets": markets, "events": []})

        return _gen()

    monkeypatch.setattr(reb.bus, "replay", _replay)


@pytest.mark.asyncio
async def test_asof_catalog_caches_fully_past_window(monkeypatch):
    sb._ASOF_CATALOG_CACHE.clear()
    counter = {"n": 0}
    _patch_bus(monkeypatch, [{"id": "m1", "clob_token_ids": ["t1"]}], counter)
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)  # long past → cacheable

    r1 = await sb._load_asof_catalog_markets(start, end)
    r2 = await sb._load_asof_catalog_markets(start, end)

    assert r1 is not None
    assert r2 is r1            # 2nd call returns the SAME cached tuple
    assert counter["n"] == 1   # bus.replay ran once; 2nd was a cache hit


@pytest.mark.asyncio
async def test_asof_catalog_does_not_cache_live_window(monkeypatch):
    sb._ASOF_CATALOG_CACHE.clear()
    counter = {"n": 0}
    _patch_bus(monkeypatch, [{"id": "m1", "clob_token_ids": ["t1"]}], counter)
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=1)
    end = now  # ends "now" → still being recorded → must NOT cache

    await sb._load_asof_catalog_markets(start, end)
    await sb._load_asof_catalog_markets(start, end)

    assert counter["n"] == 2                     # re-read each time
    assert len(sb._ASOF_CATALOG_CACHE) == 0      # nothing cached for a live window


@pytest.mark.asyncio
async def test_asof_catalog_cache_is_bounded(monkeypatch):
    sb._ASOF_CATALOG_CACHE.clear()
    counter = {"n": 0}
    _patch_bus(monkeypatch, [{"id": "m1", "clob_token_ids": ["t1"]}], counter)
    base = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    # Insert more distinct past windows than the cap; oldest must evict.
    for i in range(sb._ASOF_CATALOG_CACHE_MAX + 3):
        s = base + timedelta(days=i)
        await sb._load_asof_catalog_markets(s, s + timedelta(hours=1))

    assert len(sb._ASOF_CATALOG_CACHE) == sb._ASOF_CATALOG_CACHE_MAX
