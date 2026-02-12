import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api import routes_workers


@pytest.mark.asyncio
async def test_set_all_workers_paused_true_clears_requests(monkeypatch):
    fake_session = object()

    set_scanner_paused = AsyncMock()
    clear_scan_request = AsyncMock()
    set_news_paused = AsyncMock()
    clear_news_scan_request = AsyncMock()
    set_weather_paused = AsyncMock()
    clear_weather_scan_request = AsyncMock()
    set_discovery_paused = AsyncMock()
    clear_discovery_run_request = AsyncMock()
    update_autotrader_control = AsyncMock()
    set_worker_paused = AsyncMock()
    clear_worker_run_request = AsyncMock()

    pause_called = {"value": False}
    resume_called = {"value": False}

    monkeypatch.setattr(routes_workers.shared_state, "set_scanner_paused", set_scanner_paused)
    monkeypatch.setattr(routes_workers.shared_state, "clear_scan_request", clear_scan_request)
    monkeypatch.setattr(routes_workers.news_shared_state, "set_news_paused", set_news_paused)
    monkeypatch.setattr(
        routes_workers.news_shared_state,
        "clear_news_scan_request",
        clear_news_scan_request,
    )
    monkeypatch.setattr(
        routes_workers.weather_shared_state, "set_weather_paused", set_weather_paused
    )
    monkeypatch.setattr(
        routes_workers.weather_shared_state,
        "clear_weather_scan_request",
        clear_weather_scan_request,
    )
    monkeypatch.setattr(
        routes_workers.discovery_shared_state,
        "set_discovery_paused",
        set_discovery_paused,
    )
    monkeypatch.setattr(
        routes_workers.discovery_shared_state,
        "clear_discovery_run_request",
        clear_discovery_run_request,
    )
    monkeypatch.setattr(
        routes_workers,
        "update_autotrader_control",
        update_autotrader_control,
    )
    monkeypatch.setattr(routes_workers, "set_worker_paused", set_worker_paused)
    monkeypatch.setattr(
        routes_workers, "clear_worker_run_request", clear_worker_run_request
    )
    monkeypatch.setattr(
        routes_workers.global_pause_state,
        "pause",
        lambda: pause_called.__setitem__("value", True),
    )
    monkeypatch.setattr(
        routes_workers.global_pause_state,
        "resume",
        lambda: resume_called.__setitem__("value", True),
    )

    await routes_workers._set_all_workers_paused(fake_session, True)

    set_scanner_paused.assert_awaited_once_with(fake_session, True)
    clear_scan_request.assert_awaited_once_with(fake_session)
    set_news_paused.assert_awaited_once_with(fake_session, True)
    clear_news_scan_request.assert_awaited_once_with(fake_session)
    set_weather_paused.assert_awaited_once_with(fake_session, True)
    clear_weather_scan_request.assert_awaited_once_with(fake_session)
    set_discovery_paused.assert_awaited_once_with(fake_session, True)
    clear_discovery_run_request.assert_awaited_once_with(fake_session)
    update_autotrader_control.assert_awaited_once_with(
        fake_session,
        is_paused=True,
        requested_run=False,
    )

    assert set_worker_paused.await_count == len(routes_workers.GENERIC_WORKERS)
    assert clear_worker_run_request.await_count == len(routes_workers.GENERIC_WORKERS)
    assert pause_called["value"] is True
    assert resume_called["value"] is False


@pytest.mark.asyncio
async def test_set_all_workers_paused_false_resumes_without_clears(monkeypatch):
    fake_session = object()

    clear_scan_request = AsyncMock()
    clear_news_scan_request = AsyncMock()
    clear_weather_scan_request = AsyncMock()
    clear_discovery_run_request = AsyncMock()
    clear_worker_run_request = AsyncMock()
    update_autotrader_control = AsyncMock()

    pause_called = {"value": False}
    resume_called = {"value": False}

    monkeypatch.setattr(routes_workers.shared_state, "set_scanner_paused", AsyncMock())
    monkeypatch.setattr(routes_workers.shared_state, "clear_scan_request", clear_scan_request)
    monkeypatch.setattr(routes_workers.news_shared_state, "set_news_paused", AsyncMock())
    monkeypatch.setattr(
        routes_workers.news_shared_state,
        "clear_news_scan_request",
        clear_news_scan_request,
    )
    monkeypatch.setattr(
        routes_workers.weather_shared_state, "set_weather_paused", AsyncMock()
    )
    monkeypatch.setattr(
        routes_workers.weather_shared_state,
        "clear_weather_scan_request",
        clear_weather_scan_request,
    )
    monkeypatch.setattr(
        routes_workers.discovery_shared_state,
        "set_discovery_paused",
        AsyncMock(),
    )
    monkeypatch.setattr(
        routes_workers.discovery_shared_state,
        "clear_discovery_run_request",
        clear_discovery_run_request,
    )
    monkeypatch.setattr(
        routes_workers,
        "update_autotrader_control",
        update_autotrader_control,
    )
    monkeypatch.setattr(routes_workers, "set_worker_paused", AsyncMock())
    monkeypatch.setattr(
        routes_workers, "clear_worker_run_request", clear_worker_run_request
    )
    monkeypatch.setattr(
        routes_workers.global_pause_state,
        "pause",
        lambda: pause_called.__setitem__("value", True),
    )
    monkeypatch.setattr(
        routes_workers.global_pause_state,
        "resume",
        lambda: resume_called.__setitem__("value", True),
    )

    await routes_workers._set_all_workers_paused(fake_session, False)

    update_autotrader_control.assert_awaited_once_with(
        fake_session,
        is_paused=False,
        requested_run=None,
    )
    clear_scan_request.assert_not_awaited()
    clear_news_scan_request.assert_not_awaited()
    clear_weather_scan_request.assert_not_awaited()
    clear_discovery_run_request.assert_not_awaited()
    clear_worker_run_request.assert_not_awaited()
    assert pause_called["value"] is False
    assert resume_called["value"] is True


@pytest.mark.asyncio
async def test_start_worker_blocked_when_global_pause_active():
    fake_session = object()
    routes_workers.global_pause_state.pause()
    try:
        with pytest.raises(HTTPException) as excinfo:
            await routes_workers.start_worker("scanner", fake_session)
        assert excinfo.value.status_code == 409
    finally:
        routes_workers.global_pause_state.resume()


@pytest.mark.asyncio
async def test_run_worker_once_blocked_when_global_pause_active():
    fake_session = object()
    routes_workers.global_pause_state.pause()
    try:
        with pytest.raises(HTTPException) as excinfo:
            await routes_workers.run_worker_once("scanner", fake_session)
        assert excinfo.value.status_code == 409
    finally:
        routes_workers.global_pause_state.resume()
