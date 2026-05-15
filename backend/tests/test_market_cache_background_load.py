import asyncio
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.market_cache import MarketCacheService


def test_market_cache_background_load_skips_discovery_worker(monkeypatch):
    monkeypatch.setenv("HOMERUN_PROCESS_ROLE", "worker")
    monkeypatch.setenv("HOMERUN_WORKER_PLANE", "discovery")

    service = MarketCacheService()

    assert service.start_background_load() is None


@pytest.mark.asyncio
async def test_market_cache_background_load_is_single_flight():
    service = MarketCacheService()
    gate = asyncio.Event()
    calls: list[str] = []

    async def fake_load() -> None:
        calls.append("load")
        await gate.wait()
        service._loaded = True

    service.load_from_db = fake_load  # type: ignore[method-assign]

    first = service.start_background_load()
    second = service.start_background_load()

    assert first is not None
    assert first is second
    assert calls == []

    await asyncio.sleep(0)
    assert calls == ["load"]

    gate.set()
    await first


@pytest.mark.asyncio
async def test_market_cache_stats_report_loading_state():
    service = MarketCacheService()
    gate = asyncio.Event()

    async def fake_load() -> None:
        await gate.wait()
        service._loaded = True

    service.load_from_db = fake_load  # type: ignore[method-assign]

    task = service.start_background_load()
    assert task is not None

    await asyncio.sleep(0)
    stats = await service.get_cache_stats()
    assert stats["loading"] is True
    assert stats["loaded"] is False

    gate.set()
    await task


@pytest.mark.asyncio
async def test_market_cache_hygiene_skips_discovery_worker(monkeypatch):
    monkeypatch.setenv("HOMERUN_PROCESS_ROLE", "worker")
    monkeypatch.setenv("HOMERUN_WORKER_PLANE", "discovery")

    service = MarketCacheService()
    result = await service.run_hygiene_if_due(force=True)

    assert result == {"status": "skipped", "reason": "worker_plane", "plane": "discovery"}
