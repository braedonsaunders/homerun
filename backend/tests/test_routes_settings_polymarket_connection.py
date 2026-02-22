import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api import routes_settings
from services import kalshi_client as kalshi_client_module


def _install_fake_polymarket_modules(
    monkeypatch,
    *,
    response=None,
    error: Exception | None = None,
    install_eth_account: bool = True,
):
    py_clob_client = types.ModuleType("py_clob_client")
    client_module = types.ModuleType("py_clob_client.client")
    clob_types = types.ModuleType("py_clob_client.clob_types")

    class ClobClient:
        def __init__(self, host, key, chain_id, creds):
            self.host = host
            self.key = key
            self.chain_id = chain_id
            self.creds = creds

        def get_orders(self):
            if error is not None:
                raise error
            return [] if response is None else response

    class ApiCreds:
        def __init__(self, api_key, api_secret, api_passphrase):
            self.api_key = api_key
            self.api_secret = api_secret
            self.api_passphrase = api_passphrase

    client_module.ClobClient = ClobClient
    clob_types.ApiCreds = ApiCreds

    monkeypatch.setitem(sys.modules, "py_clob_client", py_clob_client)
    monkeypatch.setitem(sys.modules, "py_clob_client.client", client_module)
    monkeypatch.setitem(sys.modules, "py_clob_client.clob_types", clob_types)
    if install_eth_account:
        eth_account = types.ModuleType("eth_account")

        class _Account:
            @staticmethod
            def from_key(_private_key):
                return SimpleNamespace(address="0xabc123")

        eth_account.Account = _Account
        monkeypatch.setitem(sys.modules, "eth_account", eth_account)
    else:
        monkeypatch.delitem(sys.modules, "eth_account", raising=False)


def _install_fake_kalshi_client(monkeypatch, *, initialize_ok: bool = True, balance: dict | None = None):
    state = {"init_calls": [], "closed_count": 0}

    class FakeKalshiClient:
        tracker = state

        async def initialize_auth(self, email=None, password=None, api_key=None):
            self.tracker["init_calls"].append(
                {
                    "email": email,
                    "password": password,
                    "api_key": api_key,
                }
            )
            return initialize_ok

        async def get_balance(self):
            return balance

        async def close(self):
            self.tracker["closed_count"] += 1

    monkeypatch.setattr(kalshi_client_module, "KalshiClient", FakeKalshiClient)
    return FakeKalshiClient


@pytest.mark.asyncio
async def test_polymarket_connection_reports_missing_credentials(monkeypatch):
    settings_row = SimpleNamespace(
        polymarket_private_key=None,
        polymarket_api_key=None,
        polymarket_api_secret=None,
        polymarket_api_passphrase=None,
    )
    monkeypatch.setattr(routes_settings, "get_or_create_settings", AsyncMock(return_value=settings_row))
    monkeypatch.setattr(routes_settings, "decrypt_secret", lambda value: value)

    response = await routes_settings.test_polymarket_connection()

    assert response["status"] == "error"
    assert "Missing Polymarket credentials" in response["message"]
    assert "private_key" in response["message"]
    assert "api_key" in response["message"]
    assert "api_secret" in response["message"]
    assert "api_passphrase" in response["message"]


@pytest.mark.asyncio
async def test_polymarket_connection_authenticates_with_live_call(monkeypatch):
    settings_row = SimpleNamespace(
        polymarket_private_key="pk",
        polymarket_api_key="key",
        polymarket_api_secret="secret",
        polymarket_api_passphrase="passphrase",
    )
    monkeypatch.setattr(routes_settings, "get_or_create_settings", AsyncMock(return_value=settings_row))
    monkeypatch.setattr(routes_settings, "decrypt_secret", lambda value: value)
    _install_fake_polymarket_modules(monkeypatch, response=[{"id": "one"}, {"id": "two"}])

    response = await routes_settings.test_polymarket_connection()

    assert response["status"] == "success"
    assert response["message"] == "Polymarket API authentication succeeded"
    assert response["wallet_address"] == "0xabc123"
    assert response["open_orders_count"] == 2


@pytest.mark.asyncio
async def test_polymarket_connection_reports_auth_failure(monkeypatch):
    settings_row = SimpleNamespace(
        polymarket_private_key="pk",
        polymarket_api_key="key",
        polymarket_api_secret="secret",
        polymarket_api_passphrase="passphrase",
    )
    monkeypatch.setattr(routes_settings, "get_or_create_settings", AsyncMock(return_value=settings_row))
    monkeypatch.setattr(routes_settings, "decrypt_secret", lambda value: value)
    _install_fake_polymarket_modules(monkeypatch, error=RuntimeError("401 unauthorized"))

    response = await routes_settings.test_polymarket_connection()

    assert response["status"] == "error"
    assert "Polymarket API authentication failed" in response["message"]
    assert "401 unauthorized" in response["message"]


@pytest.mark.asyncio
async def test_polymarket_connection_succeeds_without_eth_account(monkeypatch):
    settings_row = SimpleNamespace(
        polymarket_private_key="pk",
        polymarket_api_key="key",
        polymarket_api_secret="secret",
        polymarket_api_passphrase="passphrase",
    )
    monkeypatch.setattr(routes_settings, "get_or_create_settings", AsyncMock(return_value=settings_row))
    monkeypatch.setattr(routes_settings, "decrypt_secret", lambda value: value)
    _install_fake_polymarket_modules(monkeypatch, response=[{"id": "one"}], install_eth_account=False)

    response = await routes_settings.test_polymarket_connection()

    assert response["status"] == "success"
    assert response["message"] == "Polymarket API authentication succeeded"
    assert response["wallet_address"] is None
    assert response["open_orders_count"] == 1


@pytest.mark.asyncio
async def test_kalshi_connection_reports_missing_credentials(monkeypatch):
    settings_row = SimpleNamespace(
        kalshi_email=None,
        kalshi_password=None,
        kalshi_api_key=None,
    )
    monkeypatch.setattr(routes_settings, "get_or_create_settings", AsyncMock(return_value=settings_row))
    monkeypatch.setattr(routes_settings, "decrypt_secret", lambda value: value)

    response = await routes_settings.test_kalshi_connection()

    assert response["status"] == "error"
    assert response["message"] == "Missing Kalshi credentials: provide api_key or email/password"


@pytest.mark.asyncio
async def test_kalshi_connection_authenticates_with_api_key(monkeypatch):
    settings_row = SimpleNamespace(
        kalshi_email=None,
        kalshi_password=None,
        kalshi_api_key="k-api",
    )
    monkeypatch.setattr(routes_settings, "get_or_create_settings", AsyncMock(return_value=settings_row))
    monkeypatch.setattr(routes_settings, "decrypt_secret", lambda value: value)
    fake_client = _install_fake_kalshi_client(
        monkeypatch,
        initialize_ok=True,
        balance={"balance": 125.0, "available": 125.0, "currency": "USD"},
    )

    response = await routes_settings.test_kalshi_connection()

    assert response["status"] == "success"
    assert response["message"] == "Kalshi API authentication succeeded"
    assert response["auth_method"] == "api_key"
    assert response["balance"] == 125.0
    assert response["available"] == 125.0
    assert response["currency"] == "USD"
    assert fake_client.tracker["init_calls"][0]["api_key"] == "k-api"
    assert fake_client.tracker["init_calls"][0]["email"] is None
    assert fake_client.tracker["closed_count"] == 1


@pytest.mark.asyncio
async def test_kalshi_connection_authenticates_with_email_password(monkeypatch):
    settings_row = SimpleNamespace(
        kalshi_email="trader@example.com",
        kalshi_password="pw",
        kalshi_api_key=None,
    )
    monkeypatch.setattr(routes_settings, "get_or_create_settings", AsyncMock(return_value=settings_row))
    monkeypatch.setattr(routes_settings, "decrypt_secret", lambda value: value)
    fake_client = _install_fake_kalshi_client(
        monkeypatch,
        initialize_ok=True,
        balance={"balance": 41.0, "available": 20.0, "currency": "USD"},
    )

    response = await routes_settings.test_kalshi_connection()

    assert response["status"] == "success"
    assert response["auth_method"] == "email_password"
    assert fake_client.tracker["init_calls"][0]["email"] == "trader@example.com"
    assert fake_client.tracker["init_calls"][0]["password"] == "pw"
    assert fake_client.tracker["init_calls"][0]["api_key"] is None
    assert fake_client.tracker["closed_count"] == 1


@pytest.mark.asyncio
async def test_kalshi_connection_reports_auth_failure(monkeypatch):
    settings_row = SimpleNamespace(
        kalshi_email=None,
        kalshi_password=None,
        kalshi_api_key="bad-key",
    )
    monkeypatch.setattr(routes_settings, "get_or_create_settings", AsyncMock(return_value=settings_row))
    monkeypatch.setattr(routes_settings, "decrypt_secret", lambda value: value)
    _install_fake_kalshi_client(
        monkeypatch,
        initialize_ok=False,
        balance=None,
    )

    response = await routes_settings.test_kalshi_connection()

    assert response["status"] == "error"
    assert response["message"] == "Kalshi API authentication failed: invalid credentials"


@pytest.mark.asyncio
async def test_kalshi_connection_reports_balance_fetch_failure(monkeypatch):
    settings_row = SimpleNamespace(
        kalshi_email=None,
        kalshi_password=None,
        kalshi_api_key="good-key",
    )
    monkeypatch.setattr(routes_settings, "get_or_create_settings", AsyncMock(return_value=settings_row))
    monkeypatch.setattr(routes_settings, "decrypt_secret", lambda value: value)
    _install_fake_kalshi_client(
        monkeypatch,
        initialize_ok=True,
        balance=None,
    )

    response = await routes_settings.test_kalshi_connection()

    assert response["status"] == "error"
    assert response["message"] == "Kalshi API authentication failed: unable to fetch account balance"
