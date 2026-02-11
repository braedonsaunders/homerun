import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from workers import news_worker


class _DummySession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_worker_respects_pause_without_manual_request(monkeypatch):
    fake_ai = SimpleNamespace(
        initialize_ai=AsyncMock(
            return_value=SimpleNamespace(is_available=lambda: False)
        )
    )
    fake_feed_service = SimpleNamespace(load_from_db=AsyncMock())
    monkeypatch.setitem(sys.modules, "services.ai", fake_ai)
    monkeypatch.setitem(
        sys.modules,
        "services.news.feed_service",
        SimpleNamespace(news_feed_service=fake_feed_service),
    )

    monkeypatch.setattr(news_worker, "AsyncSessionLocal", lambda: _DummySession())
    monkeypatch.setattr(news_worker.shared_state, "write_news_snapshot", AsyncMock())
    monkeypatch.setattr(
        news_worker.shared_state,
        "read_news_control",
        AsyncMock(
            return_value={
                "is_enabled": True,
                "is_paused": True,
                "requested_scan_at": None,
                "scan_interval_seconds": 120,
            }
        ),
    )
    monkeypatch.setattr(
        news_worker.shared_state,
        "get_news_settings",
        AsyncMock(
            return_value={
                "enabled": True,
                "auto_run": True,
                "scan_interval_seconds": 120,
                "auto_trader_max_age_minutes": 120,
            }
        ),
    )
    monkeypatch.setattr(
        news_worker.shared_state, "expire_stale_news_intents", AsyncMock(return_value=0)
    )
    monkeypatch.setattr(
        news_worker.shared_state, "count_pending_news_intents", AsyncMock(return_value=0)
    )
    run_cycle_mock = AsyncMock(return_value={"status": "completed", "findings": 0, "intents": 0, "stats": {}})
    monkeypatch.setattr(news_worker.workflow_orchestrator, "run_cycle", run_cycle_mock)
    monkeypatch.setattr(
        news_worker.shared_state, "try_acquire_news_lease", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        news_worker.shared_state, "release_news_lease", AsyncMock()
    )
    monkeypatch.setattr(news_worker.shared_state, "clear_news_scan_request", AsyncMock())
    monkeypatch.setattr(
        news_worker.asyncio,
        "sleep",
        AsyncMock(side_effect=asyncio.CancelledError()),
    )

    with pytest.raises(asyncio.CancelledError):
        await news_worker._run_loop()

    run_cycle_mock.assert_not_called()


@pytest.mark.asyncio
async def test_worker_runs_once_when_manual_request_is_set(monkeypatch):
    fake_ai = SimpleNamespace(
        initialize_ai=AsyncMock(
            return_value=SimpleNamespace(is_available=lambda: False)
        )
    )
    fake_feed_service = SimpleNamespace(load_from_db=AsyncMock())
    monkeypatch.setitem(sys.modules, "services.ai", fake_ai)
    monkeypatch.setitem(
        sys.modules,
        "services.news.feed_service",
        SimpleNamespace(news_feed_service=fake_feed_service),
    )

    monkeypatch.setattr(news_worker, "AsyncSessionLocal", lambda: _DummySession())
    monkeypatch.setattr(news_worker.shared_state, "write_news_snapshot", AsyncMock())
    monkeypatch.setattr(
        news_worker.shared_state,
        "read_news_control",
        AsyncMock(
            return_value={
                "is_enabled": True,
                "is_paused": True,
                "requested_scan_at": datetime.now(timezone.utc),
                "scan_interval_seconds": 120,
            }
        ),
    )
    monkeypatch.setattr(
        news_worker.shared_state,
        "get_news_settings",
        AsyncMock(
            return_value={
                "enabled": True,
                "auto_run": True,
                "scan_interval_seconds": 120,
                "auto_trader_max_age_minutes": 120,
            }
        ),
    )
    monkeypatch.setattr(
        news_worker.shared_state, "expire_stale_news_intents", AsyncMock(return_value=0)
    )
    monkeypatch.setattr(
        news_worker.shared_state, "count_pending_news_intents", AsyncMock(return_value=0)
    )
    run_cycle_mock = AsyncMock(
        return_value={"status": "completed", "findings": 1, "intents": 1, "stats": {}}
    )
    monkeypatch.setattr(news_worker.workflow_orchestrator, "run_cycle", run_cycle_mock)
    monkeypatch.setattr(
        news_worker.shared_state, "try_acquire_news_lease", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        news_worker.shared_state, "release_news_lease", AsyncMock()
    )
    clear_mock = AsyncMock()
    monkeypatch.setattr(news_worker.shared_state, "clear_news_scan_request", clear_mock)
    monkeypatch.setattr(
        news_worker.asyncio,
        "sleep",
        AsyncMock(side_effect=asyncio.CancelledError()),
    )

    with pytest.raises(asyncio.CancelledError):
        await news_worker._run_loop()

    run_cycle_mock.assert_awaited_once()
    clear_mock.assert_awaited_once()
