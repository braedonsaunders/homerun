import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api import routes_settings


class _FakeSession:
    async def commit(self):
        return None


class _FakeSessionContext:
    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_export_settings_bundle_returns_status_counts_and_bundle(monkeypatch):
    expected_bundle = {
        "schema_version": 1,
        "categories": [routes_settings.SettingsTransferCategory.BOT_TRADERS.value],
        "bot_traders": [],
    }
    expected_counts = {routes_settings.SettingsTransferCategory.BOT_TRADERS.value: 0}
    export_mock = AsyncMock(return_value=(expected_bundle, expected_counts))
    monkeypatch.setattr(routes_settings, "_export_transfer_bundle", export_mock)

    response = await routes_settings.export_settings_bundle(
        routes_settings.SettingsExportRequest(
            include_categories=[routes_settings.SettingsTransferCategory.BOT_TRADERS]
        )
    )

    assert response["status"] == "success"
    assert response["bundle"] == expected_bundle
    assert response["counts"] == expected_counts
    export_mock.assert_awaited_once_with([routes_settings.SettingsTransferCategory.BOT_TRADERS.value])


@pytest.mark.asyncio
async def test_import_settings_bundle_uses_bundle_categories_when_not_overridden(monkeypatch):
    fake_session = _FakeSession()
    monkeypatch.setattr(routes_settings, "AsyncSessionLocal", lambda: _FakeSessionContext(fake_session))
    import_traders_mock = AsyncMock(return_value={"created": 2, "updated": 1, "orchestrator_updated": 0})
    monkeypatch.setattr(routes_settings, "_import_traders", import_traders_mock)

    response = await routes_settings.import_settings_bundle(
        routes_settings.SettingsImportRequest(
            bundle={
                "schema_version": 1,
                "categories": [routes_settings.SettingsTransferCategory.BOT_TRADERS.value],
                "bot_traders": {
                    "traders": [{"name": "Aggro", "source_configs": []}],
                    "orchestrator": {"is_enabled": True},
                    "trade_state": {"orders": []},
                },
            }
        )
    )

    assert response["status"] == "success"
    assert response["imported_categories"] == [routes_settings.SettingsTransferCategory.BOT_TRADERS.value]
    assert response["results"][routes_settings.SettingsTransferCategory.BOT_TRADERS.value]["created"] == 2
    assert response["results"][routes_settings.SettingsTransferCategory.BOT_TRADERS.value]["updated"] == 1
    assert response["results"][routes_settings.SettingsTransferCategory.BOT_TRADERS.value]["orchestrator_updated"] == 0
    assert response["results"][routes_settings.SettingsTransferCategory.BOT_TRADERS.value]["orders_imported"] == 0
    assert response["results"][routes_settings.SettingsTransferCategory.BOT_TRADERS.value]["positions_synced"] == 0
    assert response["results"][routes_settings.SettingsTransferCategory.BOT_TRADERS.value]["open_positions"] == 0
    assert (
        response["results"][routes_settings.SettingsTransferCategory.BOT_TRADERS.value][
            "live_wallet_runtime_states_imported"
        ]
        == 0
    )
    assert response["results"][routes_settings.SettingsTransferCategory.BOT_TRADERS.value]["live_wallet_orders_imported"] == 0
    assert (
        response["results"][routes_settings.SettingsTransferCategory.BOT_TRADERS.value][
            "live_wallet_positions_imported"
        ]
        == 0
    )
    assert response["results"][routes_settings.SettingsTransferCategory.BOT_TRADERS.value]["live_wallets_imported"] == 0
    import_traders_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_import_traders_applies_trade_state(monkeypatch):
    import services.trader_orchestrator_state as orchestrator_state

    session_obj = object()
    list_traders_mock = AsyncMock(
        return_value=[
            {
                "id": "trader_alpha",
                "name": "Alpha",
                "source_configs": [],
                "risk_limits": {},
                "metadata": {},
                "mode": "shadow",
                "is_enabled": True,
                "is_paused": False,
                "interval_seconds": 60,
            }
        ]
    )
    update_trader_mock = AsyncMock(
        return_value={
            "id": "trader_alpha",
            "name": "Alpha",
            "source_configs": [],
            "risk_limits": {},
            "metadata": {},
            "mode": "shadow",
            "is_enabled": True,
            "is_paused": False,
            "interval_seconds": 60,
        }
    )
    monkeypatch.setattr(orchestrator_state, "list_traders", list_traders_mock)
    monkeypatch.setattr(orchestrator_state, "update_trader", update_trader_mock)
    monkeypatch.setattr(orchestrator_state, "create_trader", AsyncMock())

    import_trade_state_mock = AsyncMock(
        return_value={
            "orders_imported": 4,
            "positions_synced": 1,
            "open_positions": 2,
            "live_wallet_runtime_states_imported": 1,
            "live_wallet_orders_imported": 3,
            "live_wallet_positions_imported": 2,
            "live_wallets_imported": 1,
        }
    )
    monkeypatch.setattr(routes_settings, "_import_trader_trade_state", import_trade_state_mock)
    monkeypatch.setattr(routes_settings, "_apply_orchestrator_control_import", AsyncMock(return_value=True))

    payload = {
        "traders": [
            {
                "name": "Alpha",
                "source_configs": [],
                "risk_limits": {},
                "metadata": {},
            }
        ],
        "orchestrator": {"is_enabled": True, "is_paused": False},
        "trade_state": {"orders": [{"trader_name": "Alpha", "market_id": "mkt_1", "source": "manual"}]},
    }

    result = await routes_settings._import_traders(session_obj, payload)

    assert result["created"] == 0
    assert result["updated"] == 1
    assert result["orders_imported"] == 4
    assert result["positions_synced"] == 1
    assert result["open_positions"] == 2
    assert result["live_wallet_runtime_states_imported"] == 1
    assert result["live_wallet_orders_imported"] == 3
    assert result["live_wallet_positions_imported"] == 2
    assert result["live_wallets_imported"] == 1
    import_trade_state_mock.assert_awaited_once_with(
        session_obj,
        {"alpha": "trader_alpha"},
        {"orders": [{"trader_name": "Alpha", "market_id": "mkt_1", "source": "manual"}]},
    )


@pytest.mark.asyncio
async def test_import_traders_skips_trade_state_when_absent(monkeypatch):
    import services.trader_orchestrator_state as orchestrator_state

    session_obj = object()
    list_traders_mock = AsyncMock(
        return_value=[
            {
                "id": "trader_alpha",
                "name": "Alpha",
                "source_configs": [],
                "risk_limits": {},
                "metadata": {},
                "mode": "shadow",
                "is_enabled": True,
                "is_paused": False,
                "interval_seconds": 60,
            }
        ]
    )
    update_trader_mock = AsyncMock(
        return_value={
            "id": "trader_alpha",
            "name": "Alpha",
            "source_configs": [],
            "risk_limits": {},
            "metadata": {},
            "mode": "shadow",
            "is_enabled": True,
            "is_paused": False,
            "interval_seconds": 60,
        }
    )
    monkeypatch.setattr(orchestrator_state, "list_traders", list_traders_mock)
    monkeypatch.setattr(orchestrator_state, "update_trader", update_trader_mock)
    monkeypatch.setattr(orchestrator_state, "create_trader", AsyncMock())

    import_trade_state_mock = AsyncMock()
    monkeypatch.setattr(routes_settings, "_import_trader_trade_state", import_trade_state_mock)
    monkeypatch.setattr(routes_settings, "_apply_orchestrator_control_import", AsyncMock(return_value=True))

    result = await routes_settings._import_traders(
        session_obj,
        {
            "traders": [{"name": "Alpha", "source_configs": []}],
            "orchestrator": {"is_enabled": True},
        },
    )

    assert result["orders_imported"] == 0
    assert result["positions_synced"] == 0
    assert result["open_positions"] == 0
    assert result["live_wallet_runtime_states_imported"] == 0
    assert result["live_wallet_orders_imported"] == 0
    assert result["live_wallet_positions_imported"] == 0
    assert result["live_wallets_imported"] == 0
    import_trade_state_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_import_trader_trade_state_imports_live_wallet_state_without_traders(monkeypatch):
    import_live_wallet_state_mock = AsyncMock(
        return_value={
            "live_wallet_runtime_states_imported": 2,
            "live_wallet_orders_imported": 5,
            "live_wallet_positions_imported": 3,
            "live_wallets_imported": 1,
        }
    )
    monkeypatch.setattr(routes_settings, "_import_live_wallet_trade_state", import_live_wallet_state_mock)
    session_obj = object()

    result = await routes_settings._import_trader_trade_state(
        session_obj,
        {},
        {"live_wallet_state": {"runtime_states": [{"wallet_address": "0xabc"}]}},
    )

    assert result["orders_imported"] == 0
    assert result["positions_synced"] == 0
    assert result["open_positions"] == 0
    assert result["live_wallet_runtime_states_imported"] == 2
    assert result["live_wallet_orders_imported"] == 5
    assert result["live_wallet_positions_imported"] == 3
    assert result["live_wallets_imported"] == 1
    import_live_wallet_state_mock.assert_awaited_once_with(
        session_obj,
        {"runtime_states": [{"wallet_address": "0xabc"}]},
    )


@pytest.mark.asyncio
async def test_import_traders_rejects_non_object_payload():
    with pytest.raises(ValueError, match="bot_traders payload must be an object"):
        await routes_settings._import_traders(object(), [{"name": "legacy"}])


@pytest.mark.asyncio
async def test_import_settings_bundle_respects_include_override(monkeypatch):
    fake_session = _FakeSession()
    monkeypatch.setattr(routes_settings, "AsyncSessionLocal", lambda: _FakeSessionContext(fake_session))
    market_import_mock = AsyncMock(return_value=SimpleNamespace(updated_at=None))
    monkeypatch.setattr(routes_settings, "_get_or_create_settings_row", market_import_mock)
    apply_market_mock = MagicMock()

    def _apply_market_credentials(settings_row, payload):
        apply_market_mock(settings_row, payload)

    monkeypatch.setattr(routes_settings, "_apply_market_credentials_import", _apply_market_credentials)

    response = await routes_settings.import_settings_bundle(
        routes_settings.SettingsImportRequest(
            include_categories=[routes_settings.SettingsTransferCategory.MARKET_CREDENTIALS],
            bundle={
                "schema_version": 1,
                "categories": [routes_settings.SettingsTransferCategory.BOT_TRADERS.value],
                "market_credentials": {
                    "polymarket": {"api_key": "pm-key"},
                    "kalshi": {"email": "ops@example.com"},
                },
            },
        )
    )

    assert response["status"] == "success"
    assert response["imported_categories"] == [routes_settings.SettingsTransferCategory.MARKET_CREDENTIALS.value]
    assert response["results"][routes_settings.SettingsTransferCategory.MARKET_CREDENTIALS.value]["updated"] == 1
    market_import_mock.assert_awaited_once()
    assert apply_market_mock.call_count == 1


@pytest.mark.asyncio
async def test_import_settings_bundle_applies_telegram_configuration(monkeypatch):
    fake_session = _FakeSession()
    monkeypatch.setattr(routes_settings, "AsyncSessionLocal", lambda: _FakeSessionContext(fake_session))
    settings_row = SimpleNamespace(updated_at=None)
    settings_row_mock = AsyncMock(return_value=settings_row)
    monkeypatch.setattr(routes_settings, "_get_or_create_settings_row", settings_row_mock)
    apply_telegram_mock = MagicMock()

    def _apply_telegram_configuration(settings_row, payload):
        apply_telegram_mock(settings_row, payload)

    monkeypatch.setattr(routes_settings, "_apply_telegram_configuration_import", _apply_telegram_configuration)

    response = await routes_settings.import_settings_bundle(
        routes_settings.SettingsImportRequest(
            include_categories=[routes_settings.SettingsTransferCategory.TELEGRAM_CONFIGURATION],
            bundle={
                "schema_version": 1,
                "categories": [routes_settings.SettingsTransferCategory.BOT_TRADERS.value],
                "telegram_configuration": {
                    "enabled": True,
                    "telegram_bot_token": "abc123",
                    "telegram_chat_id": "998877",
                    "notify_on_trade": True,
                },
            },
        )
    )

    assert response["status"] == "success"
    assert response["imported_categories"] == [routes_settings.SettingsTransferCategory.TELEGRAM_CONFIGURATION.value]
    assert response["results"][routes_settings.SettingsTransferCategory.TELEGRAM_CONFIGURATION.value]["updated"] == 1
    settings_row_mock.assert_awaited_once()
    assert apply_telegram_mock.call_count == 1


@pytest.mark.asyncio
async def test_import_settings_bundle_skips_missing_telegram_payload(monkeypatch):
    fake_session = _FakeSession()
    monkeypatch.setattr(routes_settings, "AsyncSessionLocal", lambda: _FakeSessionContext(fake_session))
    settings_row_mock = AsyncMock(return_value=SimpleNamespace(updated_at=None))
    monkeypatch.setattr(routes_settings, "_get_or_create_settings_row", settings_row_mock)
    apply_telegram_mock = MagicMock()

    def _apply_telegram_configuration(settings_row, payload):
        apply_telegram_mock(settings_row, payload)

    monkeypatch.setattr(routes_settings, "_apply_telegram_configuration_import", _apply_telegram_configuration)

    response = await routes_settings.import_settings_bundle(
        routes_settings.SettingsImportRequest(
            include_categories=[routes_settings.SettingsTransferCategory.TELEGRAM_CONFIGURATION],
            bundle={
                "schema_version": 1,
                "categories": [routes_settings.SettingsTransferCategory.BOT_TRADERS.value],
            },
        )
    )

    assert response["status"] == "success"
    assert response["imported_categories"] == [routes_settings.SettingsTransferCategory.TELEGRAM_CONFIGURATION.value]
    assert response["results"][routes_settings.SettingsTransferCategory.TELEGRAM_CONFIGURATION.value]["updated"] == 0
    settings_row_mock.assert_not_awaited()
    assert apply_telegram_mock.call_count == 0
