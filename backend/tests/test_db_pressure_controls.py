import asyncio
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

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


def test_data_source_runner_installs_aware_source_datetime_parser():
    instance = SimpleNamespace()

    data_source_runner._install_source_datetime_parser(instance)

    parsed = instance._parse_datetime("2026-05-12T04:45:00Z")

    assert parsed is not None
    assert parsed == datetime(2026, 5, 12, 4, 45, tzinfo=timezone.utc)


def test_data_source_runner_aware_parser_compares_against_naive_datetimes():
    instance = SimpleNamespace()

    data_source_runner._install_source_datetime_parser(instance)

    parsed = instance._parse_datetime("2026-05-12T04:45:00Z")
    naive_before = datetime(2026, 5, 12, 4, 44)
    naive_after = datetime(2026, 5, 12, 4, 46)

    assert parsed is not None
    assert parsed > naive_before
    assert naive_before < parsed
    assert parsed < naive_after
    assert naive_after > parsed


def test_data_source_runner_can_install_naive_datetime_parser_for_legacy_sources():
    instance = SimpleNamespace()

    data_source_runner._install_source_datetime_parser(instance, aware=False)

    parsed = instance._parse_datetime("2026-05-12T04:45:00Z")

    assert parsed is not None
    assert parsed == datetime(2026, 5, 12, 4, 45)
    assert parsed.tzinfo is None


@pytest.mark.asyncio
async def test_trader_signal_db_fallback_releases_clean_read_transaction(monkeypatch):
    from workers import trader_orchestrator_worker as worker

    rows = [SimpleNamespace(id="signal-1")]

    class EmptyIntentRuntime:
        async def list_unconsumed_signals(self, **_kwargs):
            return []

    async def fake_authoritative(_session, **_kwargs):
        return rows

    monkeypatch.setattr(worker, "get_intent_runtime", lambda: EmptyIntentRuntime())
    monkeypatch.setattr(worker, "is_db_pressure_active", lambda: False)
    monkeypatch.setattr(worker, "current_backpressure_level", lambda: 0.0)
    monkeypatch.setattr(worker, "_list_unconsumed_trade_signals_authoritative", fake_authoritative)

    session = SimpleNamespace(
        new=[],
        dirty=[],
        deleted=[],
        in_transaction=lambda: True,
        expunge=Mock(),
        rollback=AsyncMock(),
    )

    result = await worker.list_unconsumed_trade_signals(
        session,
        trader_id="trader-1",
        sources=["scanner"],
        statuses=["pending"],
        limit=1,
    )

    assert result == rows
    session.expunge.assert_called_once_with(rows[0])
    session.rollback.assert_awaited_once()


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
    worker_state._worker_snapshot_last_signature.clear()
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


@pytest.mark.asyncio
async def test_worker_snapshot_unchanged_heartbeat_defers_without_db_pressure(monkeypatch):
    class FailingSession:
        async def execute(self, *_args, **_kwargs):
            raise AssertionError("unchanged snapshot write should have been deferred")

    worker_state._worker_snapshot_last_write_mono.clear()
    worker_state._worker_snapshot_last_signature.clear()
    last_run_at = datetime(2026, 5, 12, 4, 45, tzinfo=timezone.utc)
    signature = worker_state._snapshot_signature(
        running=True,
        enabled=True,
        current_activity="idle",
        interval_seconds=30,
        last_run_at=last_run_at,
        lag_seconds=None,
        last_error=None,
        stats={"loop": "idle"},
    )
    worker_state._worker_snapshot_last_write_mono["news"] = time.monotonic()
    worker_state._worker_snapshot_last_signature["news"] = signature
    monkeypatch.setattr(worker_state, "is_db_pressure_active", lambda: False)

    await worker_state.write_worker_snapshot(
        FailingSession(),
        "news",
        running=True,
        enabled=True,
        current_activity="idle",
        interval_seconds=30,
        last_run_at=last_run_at,
        last_error=None,
        stats={"loop": "idle"},
    )


@pytest.mark.asyncio
async def test_orchestrator_snapshot_persist_defers_low_priority_under_backpressure(monkeypatch):
    from workers import trader_orchestrator_worker as worker

    async def unexpected_write(*_args, **_kwargs):
        raise AssertionError("snapshot write should have been deferred")

    monkeypatch.setattr(worker, "current_backpressure_level", lambda: 0.7)
    monkeypatch.setattr(worker, "is_db_pressure_active", lambda: False)
    monkeypatch.setattr(worker, "write_orchestrator_snapshot", unexpected_write)

    session = SimpleNamespace(new=[], dirty=[], deleted=[])

    await worker._write_orchestrator_snapshot_best_effort(
        session,
        lane="crypto",
        running=False,
        enabled=False,
        current_activity="under pressure",
        interval_seconds=5,
    )


@pytest.mark.asyncio
async def test_orchestrator_snapshot_persist_keeps_general_heartbeat_under_backpressure(monkeypatch):
    from workers import trader_orchestrator_worker as worker

    calls = []

    async def fake_write(session, **kwargs):
        calls.append((session, kwargs))

    worker._orchestrator_snapshot_last_persist_mono.clear()
    monkeypatch.setattr(worker, "current_backpressure_level", lambda: 0.7)
    monkeypatch.setattr(worker, "is_db_pressure_active", lambda: False)
    monkeypatch.setattr(worker, "write_orchestrator_snapshot", fake_write)

    session = SimpleNamespace(new=[], dirty=[], deleted=[])

    await worker._write_orchestrator_snapshot_best_effort(
        session,
        lane="general",
        running=True,
        enabled=True,
        current_activity="under pressure",
        interval_seconds=5,
    )

    assert calls == [
        (
            session,
            {
                "running": True,
                "enabled": True,
                "current_activity": "under pressure",
                "interval_seconds": 5,
            },
        )
    ]


@pytest.mark.asyncio
async def test_orchestrator_snapshot_persist_keeps_pending_writes_under_backpressure(monkeypatch):
    from workers import trader_orchestrator_worker as worker

    calls = []

    async def fake_write(session, **kwargs):
        calls.append((session, kwargs))

    pending = object()
    monkeypatch.setattr(worker, "current_backpressure_level", lambda: 0.7)
    monkeypatch.setattr(worker, "is_db_pressure_active", lambda: False)
    monkeypatch.setattr(worker, "write_orchestrator_snapshot", fake_write)

    session = SimpleNamespace(new=[pending], dirty=[], deleted=[])

    await worker._write_orchestrator_snapshot_best_effort(
        session,
        running=True,
        enabled=True,
        current_activity="pending writes",
        interval_seconds=5,
    )

    assert calls == [
        (
            session,
            {
                "running": True,
                "enabled": True,
                "current_activity": "pending writes",
                "interval_seconds": 5,
            },
        )
    ]


@pytest.mark.asyncio
async def test_orchestrator_snapshot_persist_throttles_empty_sessions(monkeypatch):
    from workers import trader_orchestrator_worker as worker

    calls = []

    async def fake_write(session, **kwargs):
        calls.append((session, kwargs))

    worker._orchestrator_snapshot_last_persist_mono.clear()
    monkeypatch.setattr(worker, "current_backpressure_level", lambda: 0.0)
    monkeypatch.setattr(worker, "is_db_pressure_active", lambda: False)
    monkeypatch.setattr(worker, "write_orchestrator_snapshot", fake_write)

    session = SimpleNamespace(new=[], dirty=[], deleted=[])

    await worker._write_orchestrator_snapshot_best_effort(
        session,
        lane="general",
        running=True,
        enabled=True,
        current_activity="first",
        interval_seconds=5,
    )
    await worker._write_orchestrator_snapshot_best_effort(
        session,
        lane="general",
        running=True,
        enabled=True,
        current_activity="second",
        interval_seconds=5,
    )

    assert len(calls) == 1
    assert calls[0][1]["current_activity"] == "first"
