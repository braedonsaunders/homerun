from __future__ import annotations

import pytest

from services.redis_price_cache import RedisPriceCache


@pytest.mark.asyncio
async def test_write_prices_persists_metadata_fields(monkeypatch):
    cache = RedisPriceCache()
    captured: dict[str, dict[str, str]] = {}
    captured_ttl: int | None = None

    async def _hset_many(payload, expire_seconds):
        nonlocal captured_ttl
        captured.update(payload)
        captured_ttl = int(expire_seconds)

    monkeypatch.setattr("services.redis_price_cache.redis_streams.hset_many", _hset_many)

    await cache.write_prices(
        {
            "tok_1": (0.61, 0.60, 0.62, 1_706_000_000.0, 1_706_000_001.0, 17),
        }
    )

    key = "homerun:live_price:tok_1"
    assert key in captured
    row = captured[key]
    assert row["mid"] == "0.61"
    assert row["bid"] == "0.6"
    assert row["ask"] == "0.62"
    assert row["exchange_ts"] == "1706000000.0"
    assert row["ingest_ts"] == "1706000001.0"
    assert row["seq"] == "17"
    assert row["fresh"] == "1"
    assert captured_ttl is not None
    assert captured_ttl >= 1


@pytest.mark.asyncio
async def test_read_prices_returns_metadata_and_filters_stale(monkeypatch):
    cache = RedisPriceCache()

    async def _hgetall_many(_keys):
        return [
            {
                "mid": "0.44",
                "bid": "0.43",
                "ask": "0.45",
                "ingest_ts": "99.0",
                "exchange_ts": "98.5",
                "seq": "23",
            },
            {
                "mid": "0.10",
                "ingest_ts": "1.0",
                "exchange_ts": "1.0",
                "seq": "1",
            },
        ]

    monkeypatch.setattr("services.redis_price_cache.redis_streams.hgetall_many", _hgetall_many)
    monkeypatch.setattr("services.redis_price_cache.time.time", lambda: 100.0)

    rows = await cache.read_prices(["tok_fresh", "tok_stale"], stale_seconds=5.0)
    assert "tok_fresh" in rows
    assert "tok_stale" not in rows

    row = rows["tok_fresh"]
    assert row["mid"] == 0.44
    assert row["bid"] == 0.43
    assert row["ask"] == 0.45
    assert row["ts"] == 99.0
    assert row["ingest_ts"] == 99.0
    assert row["exchange_ts"] == 98.5
    assert row["sequence"] == 23
    assert row["is_fresh"] is True
