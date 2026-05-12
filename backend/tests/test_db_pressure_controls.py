import asyncio
import time
from types import SimpleNamespace

import pytest

from services import data_source_runner
from services import worker_state
from services.news import feed_service
from services.news.feed_service import NewsFeedService
from services.recorded_event_bus import catalog


@pytest.mark.asyncio
async def test_scheduled_catalog_touches_coalesce_by_topic(monkeypatch):
    calls = []

    async def fake_touch_published(slug, *, n_events=1, bytes_added=0, published_at=None):
        calls.append(
            {
                "slug": slug,
                "n_events": n_events,
                "bytes_added": bytes_added,
                "published_at": published_at,
            }
        )

    catalog._touch_published_pending.clear()
    catalog._touch_published_task = None
    monkeypatch.setattr(catalog, "_TOUCH_PUBLISHED_FLUSH_DELAY_SECONDS", 0.0)
    monkeypatch.setattr(catalog, "touch_published", fake_touch_published)

    catalog.schedule_touch_published("prices.book", n_events=1)
    catalog.schedule_touch_published("prices.book", n_events=4, bytes_added=128)

    task = catalog._touch_published_task
    assert task is not None
    await asyncio.wait_for(task, timeout=1.0)

    assert calls == [
        {
            "slug": "prices.book",
            "n_events": 5,
            "bytes_added": 128,
            "published_at": calls[0]["published_at"],
        }
    ]


def test_data_source_retention_throttle_is_per_source(monkeypatch):
    data_source_runner._retention_last_applied_mono.clear()
    monkeypatch.setattr(data_source_runner, "_RETENTION_MIN_INTERVAL_SECONDS", 10.0)

    assert data_source_runner._retention_due("source-a") is True

    data_source_runner._retention_last_applied_mono["source-a"] = time.monotonic()
    assert data_source_runner._retention_due("source-a") is False
    assert data_source_runner._retention_due("source-b") is True

    data_source_runner._retention_last_applied_mono["source-a"] -= 11.0
    assert data_source_runner._retention_due("source-a") is True


@pytest.mark.asyncio
async def test_news_source_fetch_concurrency_collapses_under_db_pressure(monkeypatch):
    service = NewsFeedService()
    active = 0
    max_active = 0

    async def fake_fetch_source_rows(_source):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return []

    monkeypatch.setattr(feed_service, "is_db_pressure_active", lambda: True)
    monkeypatch.setattr(service, "_fetch_source_rows", fake_fetch_source_rows)

    sources = [SimpleNamespace(slug=f"source-{i}", config={}) for i in range(5)]
    await service._fetch_articles_from_sources(sources)

    assert max_active == 1


@pytest.mark.asyncio
async def test_worker_snapshot_heartbeat_defers_during_recent_db_pressure(monkeypatch):
    class FailingSession:
        async def execute(self, *_args, **_kwargs):
            raise AssertionError("snapshot write should have been deferred")

    worker_state._worker_snapshot_last_write_mono.clear()
    worker_state._worker_snapshot_last_write_mono["news"] = time.monotonic()
    monkeypatch.setattr(worker_state, "is_db_pressure_active", lambda: True)

    await worker_state.write_worker_snapshot(
        FailingSession(),
        "news",
        running=True,
        enabled=True,
        current_activity="idle",
        interval_seconds=30,
        last_run_at=None,
        last_error=None,
        stats={"loop": "idle"},
    )
