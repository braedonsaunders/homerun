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
    import_traders_mock = AsyncMock(return_value={"created": 2, "updated": 1})
    monkeypatch.setattr(routes_settings, "_import_traders", import_traders_mock)

    response = await routes_settings.import_settings_bundle(
        routes_settings.SettingsImportRequest(
            bundle={
                "schema_version": 1,
                "categories": [routes_settings.SettingsTransferCategory.BOT_TRADERS.value],
                "bot_traders": [{"name": "Aggro", "source_configs": []}],
            }
        )
    )

    assert response["status"] == "success"
    assert response["imported_categories"] == [routes_settings.SettingsTransferCategory.BOT_TRADERS.value]
    assert response["results"][routes_settings.SettingsTransferCategory.BOT_TRADERS.value]["created"] == 2
    assert response["results"][routes_settings.SettingsTransferCategory.BOT_TRADERS.value]["updated"] == 1
    import_traders_mock.assert_awaited_once()


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
