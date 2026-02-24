import sys
import types
from datetime import datetime
from pathlib import Path

import pytest
from unittest.mock import AsyncMock

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import settings
from services.trading import Position, TradingService


class _BalanceClient:
    def __init__(self, payload, *, fail_on_update: bool = False, builder_sig_type: int | None = 0):
        self.payload = payload
        self.fail_on_update = fail_on_update
        self.update_calls = 0
        self.get_calls = 0
        if builder_sig_type is not None:
            self.builder = type("Builder", (), {})()
            self.builder.sig_type = builder_sig_type

    def update_balance_allowance(self, params=None):
        self.update_calls += 1
        if self.fail_on_update:
            raise RuntimeError("refresh failed")
        return {"ok": True}

    def get_balance_allowance(self, params=None):
        self.get_calls += 1
        if isinstance(self.payload, dict) and self.payload and all(isinstance(key, int) for key in self.payload):
            signature_type = int(getattr(params, "signature_type", -1))
            if signature_type in self.payload:
                return self.payload[signature_type]
            return next(iter(self.payload.values()))
        return self.payload


def _install_balance_allowance_modules(monkeypatch) -> None:
    py_clob_client = types.ModuleType("py_clob_client")
    clob_types = types.ModuleType("py_clob_client.clob_types")

    class AssetType:
        COLLATERAL = "COLLATERAL"
        CONDITIONAL = "CONDITIONAL"

    class BalanceAllowanceParams:
        def __init__(self, *, asset_type, signature_type, token_id=None):
            self.asset_type = asset_type
            self.signature_type = signature_type
            self.token_id = token_id

    clob_types.AssetType = AssetType
    clob_types.BalanceAllowanceParams = BalanceAllowanceParams

    monkeypatch.setitem(sys.modules, "py_clob_client", py_clob_client)
    monkeypatch.setitem(sys.modules, "py_clob_client.clob_types", clob_types)


@pytest.mark.asyncio
async def test_get_balance_returns_error_when_service_not_initialized():
    service = TradingService()
    service.ensure_initialized = AsyncMock(return_value=False)
    result = await service.get_balance()
    assert result == {"error": "Polymarket credentials not configured"}


@pytest.mark.asyncio
async def test_get_balance_returns_error_when_wallet_address_missing(monkeypatch):
    service = TradingService()
    service._initialized = True
    service._client = _BalanceClient({"balance": "0", "allowance": "0"})
    service._wallet_address = None
    monkeypatch.setattr(settings, "POLYMARKET_PRIVATE_KEY", None)

    result = await service.get_balance()
    assert result == {"error": "Could not derive wallet address"}


@pytest.mark.asyncio
async def test_get_balance_returns_collateral_balance_allowance_and_positions_value():
    service = TradingService()
    service._initialized = True
    service._wallet_address = "0x1234567890abcdef1234567890abcdef12345678"
    service._client = _BalanceClient({"balance": "150.25", "allowance": "120.00"})
    service._positions = {
        "token-1": Position(
            token_id="token-1",
            market_id="market-1",
            market_question="Question 1",
            outcome="YES",
            size=10.0,
            average_cost=0.45,
            current_price=0.50,
        ),
        "token-2": Position(
            token_id="token-2",
            market_id="market-2",
            market_question="Question 2",
            outcome="NO",
            size=20.0,
            average_cost=0.65,
            current_price=0.40,
        ),
    }

    result = await service.get_balance()

    assert result["address"] == service._wallet_address
    assert result["balance"] == pytest.approx(150.25)
    assert result["available"] == pytest.approx(120.0)
    assert result["reserved"] == pytest.approx(30.25)
    assert result["currency"] == "USDC"
    assert result["positions_value"] == pytest.approx(13.0)
    assert isinstance(datetime.fromisoformat(result["timestamp"]), datetime)
    assert service._client.update_calls == 1
    assert service._client.get_calls == 1


@pytest.mark.asyncio
async def test_get_balance_returns_error_on_unexpected_payload_shape():
    service = TradingService()
    service._initialized = True
    service._wallet_address = "0x1234567890abcdef1234567890abcdef12345678"
    service._client = _BalanceClient(["unexpected"])

    result = await service.get_balance()
    assert result == {"error": "Could not fetch balance from CLOB API"}


@pytest.mark.asyncio
async def test_get_balance_uses_cached_values_when_refresh_fails():
    service = TradingService()
    service._initialized = True
    service._wallet_address = "0x1234567890abcdef1234567890abcdef12345678"
    service._client = _BalanceClient({"balance": "90.5"}, fail_on_update=True)

    result = await service.get_balance()
    assert result["balance"] == pytest.approx(90.5)
    assert result["available"] == pytest.approx(90.5)
    assert result["reserved"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_get_balance_converts_collateral_base_units_to_usdc():
    service = TradingService()
    service._initialized = True
    service._wallet_address = "0x1234567890abcdef1234567890abcdef12345678"
    service._client = _BalanceClient(
        {
            "balance": "150250000",
            "allowances": {
                "exchange": "120000000",
            },
        }
    )

    result = await service.get_balance()
    assert result["balance"] == pytest.approx(150.25)
    assert result["available"] == pytest.approx(120.0)
    assert result["reserved"] == pytest.approx(30.25)


@pytest.mark.asyncio
async def test_get_balance_probes_signature_types_and_picks_non_zero_bucket():
    service = TradingService()
    service._initialized = True
    service._wallet_address = "0x1234567890abcdef1234567890abcdef12345678"
    service._proxy_funder_address = service._wallet_address
    service._client = _BalanceClient(
        {
            0: {"balance": "0", "allowances": {"exchange": "0"}},
            1: {
                "balance": "72675329",
                "allowances": {
                    "exchange": "115792089237316195423570985008687907853269984665640564039457584007913129639935"
                },
            },
            2: {"balance": "0", "allowances": {"exchange": "0"}},
        },
        builder_sig_type=0,
    )

    result = await service.get_balance()
    assert result["balance"] == pytest.approx(72.675329)
    assert result["available"] == pytest.approx(72.675329)
    assert result["reserved"] == pytest.approx(0.0)
    assert service._balance_signature_type == 1
    assert service._client.builder.sig_type == 1


@pytest.mark.asyncio
async def test_prepare_sell_balance_allowance_selects_signature_type_with_conditional_balance(monkeypatch):
    _install_balance_allowance_modules(monkeypatch)

    service = TradingService()
    service._initialized = True
    service._wallet_address = "0x1234567890abcdef1234567890abcdef12345678"
    service._proxy_funder_address = service._wallet_address
    service._balance_signature_type = 0
    service._client = _BalanceClient(
        {
            0: {"balance": "0", "allowance": "0"},
            1: {"balance": "0", "allowance": "0"},
            2: {"balance": "12.5", "allowance": "100.0"},
        },
        builder_sig_type=0,
    )

    refreshed = await service.prepare_sell_balance_allowance("token-123")

    assert refreshed is True
    assert service._balance_signature_type == 2
    assert service._client.builder.sig_type == 2
