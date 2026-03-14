import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.notifier import TelegramNotifier
from services import notifier as notifier_module


@pytest.mark.asyncio
async def test_handle_telegram_command_autotrader_on_dispatches_start(monkeypatch):
    notifier = TelegramNotifier()
    start_mock = AsyncMock(return_value="start-ok")
    stop_mock = AsyncMock(return_value="stop-ok")
    monkeypatch.setattr(notifier, "_telegram_start_autotrader", start_mock)
    monkeypatch.setattr(notifier, "_telegram_stop_autotrader", stop_mock)

    response = await notifier._handle_telegram_command(
        text="/autotrader on shadow",
        operator="telegram:test",
        chat_id="12345",
    )

    assert response == "start-ok"
    start_mock.assert_awaited_once_with(
        mode_arg="shadow",
        operator="telegram:test",
    )
    stop_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_telegram_command_autotrader_off_dispatches_stop(monkeypatch):
    notifier = TelegramNotifier()
    start_mock = AsyncMock(return_value="start-ok")
    stop_mock = AsyncMock(return_value="stop-ok")
    monkeypatch.setattr(notifier, "_telegram_start_autotrader", start_mock)
    monkeypatch.setattr(notifier, "_telegram_stop_autotrader", stop_mock)

    response = await notifier._handle_telegram_command(
        text="/autotrader off",
        operator="telegram:test",
        chat_id="12345",
    )

    assert response == "stop-ok"
    stop_mock.assert_awaited_once_with(operator="telegram:test")
    start_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_telegram_command_on_alias_dispatches_start(monkeypatch):
    notifier = TelegramNotifier()
    start_mock = AsyncMock(return_value="start-ok")
    stop_mock = AsyncMock(return_value="stop-ok")
    monkeypatch.setattr(notifier, "_telegram_start_autotrader", start_mock)
    monkeypatch.setattr(notifier, "_telegram_stop_autotrader", stop_mock)

    response = await notifier._handle_telegram_command(
        text="/on shadow",
        operator="telegram:test",
        chat_id="12345",
    )

    assert response == "start-ok"
    start_mock.assert_awaited_once_with(
        mode_arg="shadow",
        operator="telegram:test",
    )
    stop_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_telegram_command_autotrader_requires_valid_action():
    notifier = TelegramNotifier()

    response = await notifier._handle_telegram_command(
        text="/autotrader maybe",
        operator="telegram:test",
        chat_id="12345",
    )

    assert response == "Usage: <code>/autotrader on [shadow|live]</code> or <code>/autotrader off</code>"


@pytest.mark.asyncio
async def test_telegram_start_autotrader_live_mode_uses_live_flow(monkeypatch):
    notifier = TelegramNotifier()

    class _Session:
        pass

    class _SessionContext:
        async def __aenter__(self):
            return _Session()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    preflight_mock = AsyncMock(
        return_value={
            "status": "passed",
            "preflight_id": "preflight-1",
        }
    )
    arm_mock = AsyncMock(return_value={"arm_token": "arm-token"})
    consume_mock = AsyncMock(return_value=True)
    update_control_mock = AsyncMock(return_value={"run_interval_seconds": 2})
    snapshot_mock = AsyncMock(return_value=None)
    event_mock = AsyncMock(return_value=None)

    monkeypatch.setattr(notifier_module, "AsyncSessionLocal", lambda: _SessionContext())
    monkeypatch.setattr(notifier_module, "global_pause_state", SimpleNamespace(is_paused=False))
    monkeypatch.setattr(
        notifier_module,
        "read_orchestrator_control",
        AsyncMock(return_value={"mode": "shadow", "settings": {}}),
    )
    monkeypatch.setattr(notifier_module, "create_live_preflight", preflight_mock)
    monkeypatch.setattr(notifier_module, "arm_live_start", arm_mock)
    monkeypatch.setattr(notifier_module, "consume_live_arm_token", consume_mock)
    monkeypatch.setattr(notifier_module, "update_orchestrator_control", update_control_mock)
    monkeypatch.setattr(notifier_module, "write_orchestrator_snapshot", snapshot_mock)
    monkeypatch.setattr(notifier_module, "create_trader_event", event_mock)

    response = await notifier._telegram_start_autotrader(
        mode_arg="live",
        operator="telegram:test",
    )

    assert response == "✅ Autotrader start queued.\nMode: LIVE"
    preflight_mock.assert_awaited_once_with(
        ANY,
        requested_mode="live",
        requested_by="telegram:test",
    )
    arm_mock.assert_awaited_once_with(
        ANY,
        preflight_id="preflight-1",
        ttl_seconds=300,
        requested_by="telegram:test",
    )
    consume_mock.assert_awaited_once_with(ANY, "arm-token")
    update_control_mock.assert_awaited_once_with(
        ANY,
        is_enabled=True,
        is_paused=False,
        mode="live",
        requested_run_at=ANY,
    )
    snapshot_mock.assert_awaited_once()
    event_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_telegram_start_autotrader_without_mode_uses_live_when_control_mode_live(monkeypatch):
    notifier = TelegramNotifier()

    class _Session:
        pass

    class _SessionContext:
        async def __aenter__(self):
            return _Session()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    preflight_mock = AsyncMock(return_value={"status": "passed", "preflight_id": "preflight-2"})
    arm_mock = AsyncMock(return_value={"arm_token": "arm-token-2"})
    consume_mock = AsyncMock(return_value=True)
    update_control_mock = AsyncMock(return_value={"run_interval_seconds": 2})

    monkeypatch.setattr(notifier_module, "AsyncSessionLocal", lambda: _SessionContext())
    monkeypatch.setattr(notifier_module, "global_pause_state", SimpleNamespace(is_paused=False))
    monkeypatch.setattr(
        notifier_module,
        "read_orchestrator_control",
        AsyncMock(return_value={"mode": "live", "settings": {}}),
    )
    monkeypatch.setattr(notifier_module, "create_live_preflight", preflight_mock)
    monkeypatch.setattr(notifier_module, "arm_live_start", arm_mock)
    monkeypatch.setattr(notifier_module, "consume_live_arm_token", consume_mock)
    monkeypatch.setattr(notifier_module, "update_orchestrator_control", update_control_mock)
    monkeypatch.setattr(notifier_module, "write_orchestrator_snapshot", AsyncMock(return_value=None))
    monkeypatch.setattr(notifier_module, "create_trader_event", AsyncMock(return_value=None))

    response = await notifier._telegram_start_autotrader(
        mode_arg=None,
        operator="telegram:test",
    )

    assert response == "✅ Autotrader start queued.\nMode: LIVE"
    preflight_mock.assert_awaited_once()
    arm_mock.assert_awaited_once()
    consume_mock.assert_awaited_once()
    update_control_mock.assert_awaited_once_with(
        ANY,
        is_enabled=True,
        is_paused=False,
        mode="live",
        requested_run_at=ANY,
    )
