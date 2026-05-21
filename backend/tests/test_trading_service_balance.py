import sys
import types
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest
from unittest.mock import AsyncMock

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import settings
import services.live_execution_service as live_execution_module
from services.live_execution_service import (
    OrderSide,
    OrderStatus,
    Position,
    LiveExecutionService,
)


class _BalanceClient:
    def __init__(self, payload, *, fail_on_update: bool = False, builder_signature_type: int | None = 0):
        self.payload = payload
        self.fail_on_update = fail_on_update
        self.update_calls = 0
        self.get_calls = 0
        if builder_signature_type is not None:
            self.builder = type("Builder", (), {})()
            self.builder.signature_type = builder_signature_type

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


class _FunderAwareBalanceClient(_BalanceClient):
    def __init__(self, payload_by_funder, *, builder_signature_type: int = 0, builder_funder: str):
        super().__init__({}, builder_signature_type=builder_signature_type)
        self.payload_by_funder = payload_by_funder
        self.builder.funder = builder_funder

    def get_balance_allowance(self, params=None):
        self.get_calls += 1
        return self.payload_by_funder.get(str(self.builder.funder), {"balance": "0", "allowance": "0"})


def _install_balance_allowance_modules(monkeypatch) -> None:
    py_clob_client = types.ModuleType("py_clob_client_v2")
    clob_types = types.ModuleType("py_clob_client_v2.clob_types")
    order_builder = types.ModuleType("py_clob_client_v2.order_builder")
    constants = types.ModuleType("py_clob_client_v2.order_builder.constants")

    class AssetType:
        COLLATERAL = "COLLATERAL"
        CONDITIONAL = "CONDITIONAL"

    class BalanceAllowanceParams:
        def __init__(self, *, asset_type, signature_type, token_id=None):
            self.asset_type = asset_type
            self.signature_type = signature_type
            self.token_id = token_id

    class OrderArgs:
        def __init__(self, price, size, side, token_id):
            self.price = price
            self.size = size
            self.side = side
            self.token_id = token_id

    clob_types.AssetType = AssetType
    clob_types.BalanceAllowanceParams = BalanceAllowanceParams
    clob_types.OrderArgs = OrderArgs
    constants.BUY = "BUY"
    constants.SELL = "SELL"

    monkeypatch.setitem(sys.modules, "py_clob_client_v2", py_clob_client)
    monkeypatch.setitem(sys.modules, "py_clob_client_v2.clob_types", clob_types)
    monkeypatch.setitem(sys.modules, "py_clob_client_v2.order_builder", order_builder)
    monkeypatch.setitem(sys.modules, "py_clob_client_v2.order_builder.constants", constants)


@pytest.mark.asyncio
async def test_get_balance_returns_error_when_service_not_initialized():
    service = LiveExecutionService()
    service.ensure_initialized = AsyncMock(return_value=False)
    result = await service.get_balance()
    assert result == {"error": "Polymarket credentials not configured"}


@pytest.mark.asyncio
async def test_get_balance_returns_error_when_wallet_address_missing(monkeypatch):
    service = LiveExecutionService()
    service._initialized = True
    service._client = _BalanceClient({"balance": "0", "allowance": "0"})
    service._wallet_address = None
    monkeypatch.setattr(settings, "POLYMARKET_PRIVATE_KEY", None)

    result = await service.get_balance()
    assert result == {"error": "Could not derive wallet address"}


@pytest.mark.asyncio
async def test_get_balance_returns_collateral_balance_allowance_and_positions_value():
    service = LiveExecutionService()
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
    service = LiveExecutionService()
    service._initialized = True
    service._wallet_address = "0x1234567890abcdef1234567890abcdef12345678"
    service._client = _BalanceClient(["unexpected"])

    result = await service.get_balance()
    assert result == {"error": "Could not fetch balance from CLOB API"}


@pytest.mark.asyncio
async def test_get_balance_uses_cached_values_when_refresh_fails():
    service = LiveExecutionService()
    service._initialized = True
    service._wallet_address = "0x1234567890abcdef1234567890abcdef12345678"
    service._client = _BalanceClient({"balance": "90.5"}, fail_on_update=True)

    result = await service.get_balance()
    assert result["balance"] == pytest.approx(90.5)
    assert result["available"] == pytest.approx(90.5)
    assert result["reserved"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_get_balance_converts_collateral_base_units_to_usdc():
    service = LiveExecutionService()
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
    service = LiveExecutionService()
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
        builder_signature_type=0,
    )

    result = await service.get_balance()
    assert result["balance"] == pytest.approx(72.675329)
    assert result["available"] == pytest.approx(72.675329)
    assert result["reserved"] == pytest.approx(0.0)
    assert service._balance_signature_type == 1
    assert service._client.builder.signature_type == 1


@pytest.mark.asyncio
async def test_get_balance_repoints_builder_funder_for_each_signature_probe():
    eoa = "0x0000000000000000000000000000000000000001"
    proxy = "0x0000000000000000000000000000000000000002"
    service = LiveExecutionService()
    service._initialized = True
    service._wallet_address = eoa
    service._eoa_address = eoa
    service._proxy_funder_address = proxy
    service._balance_signature_type = 0
    service._client = _FunderAwareBalanceClient(
        {
            eoa: {"balance": "0", "allowance": "0"},
            proxy: {"balance": "165.0", "allowance": "165.0"},
        },
        builder_signature_type=0,
        builder_funder=eoa,
    )

    result = await service.get_balance(force_probe_all=True)

    assert result["address"] == proxy
    assert result["balance"] == pytest.approx(165.0)
    assert result["available"] == pytest.approx(165.0)
    assert result["signature_type"] == 1
    assert service._balance_signature_type == 1
    assert service._client.builder.signature_type == 1
    assert service._client.builder.funder == proxy


@pytest.mark.asyncio
async def test_buy_pre_submit_gate_force_probes_signature_types_when_current_bucket_is_short(monkeypatch):
    # Verifies the legacy chain-probe fallback path (active when the
    # collateral keeper is disabled or has not yet bootstrapped) still
    # walks all signature types on a cache-insufficient read.  The
    # default keeper-backed path is exercised in test_collateral_keeper.
    monkeypatch.setenv("HOMERUN_COLLATERAL_KEEPER_ENABLED", "0")
    monkeypatch.setattr(settings, "MIN_ACCOUNT_BALANCE_USD", 50.0)

    service = LiveExecutionService()
    service._initialized = True
    service._wallet_address = "0x1234567890abcdef1234567890abcdef12345678"
    service._proxy_funder_address = service._wallet_address
    service._balance_signature_type = 0
    service._client = _BalanceClient(
        {
            0: {"balance": "12.0", "allowance": "12.0"},
            1: {"balance": "165.0", "allowance": "165.0"},
            2: {"balance": "12.0", "allowance": "12.0"},
        },
        builder_signature_type=0,
    )
    service.refresh_collateral_balance_allowance = AsyncMock(return_value=True)

    gate_ok, gate_error = await service._enforce_buy_pre_submit_gate(
        token_id="token-123",
        required_notional_usd=Decimal("4.45"),
    )

    assert gate_ok is True
    assert gate_error is None
    assert service._balance_signature_type == 1
    assert service._client.builder.signature_type == 1


@pytest.mark.asyncio
async def test_prepare_sell_balance_allowance_selects_signature_type_with_conditional_balance(monkeypatch):
    _install_balance_allowance_modules(monkeypatch)

    service = LiveExecutionService()
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
        builder_signature_type=0,
    )

    refreshed = await service.prepare_sell_balance_allowance("token-123")

    assert refreshed is True
    assert service._balance_signature_type == 2
    assert service._client.builder.signature_type == 2


@pytest.mark.asyncio
async def test_sell_pre_submit_gate_blocks_when_conditional_shares_insufficient(monkeypatch):
    _install_balance_allowance_modules(monkeypatch)

    service = LiveExecutionService()
    service._initialized = True
    service._wallet_address = "0x1234567890abcdef1234567890abcdef12345678"
    service._proxy_funder_address = service._wallet_address
    service._balance_signature_type = 1
    service._client = _BalanceClient(
        {
            1: {"balance": "1.2", "allowance": "1.0"},
        },
        builder_signature_type=1,
    )

    gate_ok, gate_error = await service._enforce_sell_pre_submit_gate(token_id="token-123", size=2.0)

    assert gate_ok is False
    assert gate_error is not None
    assert "required_shares=2.0" in gate_error
    assert "available_shares=1.0" in gate_error
    assert "signature_type=1" in gate_error


@pytest.mark.asyncio
async def test_sell_pre_submit_gate_rechecks_fresh_snapshot_before_blocking(monkeypatch):
    _install_balance_allowance_modules(monkeypatch)

    class _StaleThenFreshClient(_BalanceClient):
        def __init__(self):
            super().__init__({}, builder_signature_type=1)
            self.snapshot_calls = 0

        def get_balance_allowance(self, params=None):
            self.snapshot_calls += 1
            if self.snapshot_calls == 1:
                return {"balance": "5.0", "allowance": "1.0"}
            return {"balance": "5.0", "allowance": "5.0"}

    service = LiveExecutionService()
    service._initialized = True
    service._wallet_address = "0x1234567890abcdef1234567890abcdef12345678"
    service._proxy_funder_address = service._wallet_address
    service._balance_signature_type = 1
    service._client = _StaleThenFreshClient()
    service._select_signature_type_for_conditional_token = AsyncMock(return_value=1)
    service.refresh_conditional_balance_allowance = AsyncMock(return_value=False)

    gate_ok, gate_error = await service._enforce_sell_pre_submit_gate(token_id="token-123", size=2.0)

    assert gate_ok is True
    assert gate_error is None
    assert service._client.snapshot_calls >= 2


@pytest.mark.asyncio
async def test_buy_pre_submit_gate_blocks_when_collateral_insufficient(monkeypatch):
    # Legacy chain-probe rejection format with signature_type / per-
    # type funder diagnostics.  The keeper-backed gate emits a
    # different (keeper-flavored) rejection message — exercised in
    # test_collateral_keeper.
    monkeypatch.setenv("HOMERUN_COLLATERAL_KEEPER_ENABLED", "0")
    _install_balance_allowance_modules(monkeypatch)

    service = LiveExecutionService()
    service._initialized = True
    service._wallet_address = "0x1234567890abcdef1234567890abcdef12345678"
    service._proxy_funder_address = service._wallet_address
    service._balance_signature_type = 1
    service._client = _BalanceClient(
        {
            1: {"balance": "4.0", "allowance": "3.0"},
        },
        builder_signature_type=1,
    )

    gate_ok, gate_error = await service._enforce_buy_pre_submit_gate(
        token_id="token-123",
        required_notional_usd=Decimal("5.0"),
    )

    assert gate_ok is False
    assert gate_error is not None
    assert "required_usdc=5.0" in gate_error
    assert "available_usdc=3.0" in gate_error
    assert "signature_type=1" in gate_error


@pytest.mark.asyncio
async def test_place_sell_order_fails_before_submit_when_pre_submit_gate_fails(monkeypatch):
    _install_balance_allowance_modules(monkeypatch)
    monkeypatch.setattr(settings, "MIN_ORDER_SIZE_USD", 1.0)
    monkeypatch.setattr(settings, "MAX_TRADE_SIZE_USD", 10_000.0)
    monkeypatch.setattr(settings, "MAX_DAILY_TRADE_VOLUME", 10_000.0)
    monkeypatch.setattr(settings, "MAX_PER_MARKET_USD", 10_000.0)
    monkeypatch.setattr(
        live_execution_module.global_pause_state,
        "refresh_from_db",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(live_execution_module, "pre_trade_vpn_check", AsyncMock(return_value=(True, "")))

    class _SellGateClient(_BalanceClient):
        def __init__(self):
            super().__init__(
                {
                    1: {"balance": "1.2", "allowance": "1.0"},
                },
                builder_signature_type=1,
            )
            self.post_calls = 0

        def create_order(self, order_args):
            return {"order_args": order_args}

        def post_order(self, signed_order, order_type, post_only=False):
            self.post_calls += 1
            return {"success": True, "orderID": "oid-1"}

    service = LiveExecutionService()
    service._initialized = True
    service._wallet_address = "0x1234567890abcdef1234567890abcdef12345678"
    service._proxy_funder_address = service._wallet_address
    service._balance_signature_type = 1
    client = _SellGateClient()
    service._client = client
    # ``place_order`` always persists; without these stubs the test wallet
    # would leak rows into the real ``live_trading_orders`` table.
    monkeypatch.setattr(service, "_persist_orders", AsyncMock())
    monkeypatch.setattr(service, "_persist_runtime_state", AsyncMock())

    order = await service.place_order(
        token_id="token-123",
        side=OrderSide.SELL,
        price=0.5,
        size=2.0,
    )

    assert order.status == OrderStatus.FAILED
    assert order.error_message is not None
    assert "SELL pre-submit gate failed" in order.error_message
    assert client.post_calls == 0


@pytest.mark.asyncio
async def test_place_buy_order_fails_before_submit_when_pre_submit_gate_fails(monkeypatch):
    _install_balance_allowance_modules(monkeypatch)
    monkeypatch.setattr(settings, "MIN_ORDER_SIZE_USD", 1.0)
    monkeypatch.setattr(settings, "MAX_TRADE_SIZE_USD", 10_000.0)
    monkeypatch.setattr(settings, "MAX_DAILY_TRADE_VOLUME", 10_000.0)
    monkeypatch.setattr(settings, "MAX_PER_MARKET_USD", 10_000.0)
    monkeypatch.setattr(
        live_execution_module.global_pause_state,
        "refresh_from_db",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(live_execution_module, "pre_trade_vpn_check", AsyncMock(return_value=(True, "")))

    class _BuyGateClient(_BalanceClient):
        def __init__(self):
            super().__init__(
                {
                    1: {"balance": "2.0", "allowance": "2.0"},
                },
                builder_signature_type=1,
            )
            self.post_calls = 0

        def create_order(self, order_args):
            return {"order_args": order_args}

        def post_order(self, signed_order, order_type, post_only=False):
            self.post_calls += 1
            return {"success": True, "orderID": "oid-1"}

    service = LiveExecutionService()
    service._initialized = True
    service._wallet_address = "0x1234567890abcdef1234567890abcdef12345678"
    service._proxy_funder_address = service._wallet_address
    service._balance_signature_type = 1
    client = _BuyGateClient()
    service._client = client
    # ``place_order`` always persists; without these stubs the test wallet
    # would leak rows into the real ``live_trading_orders`` table.
    monkeypatch.setattr(service, "_persist_orders", AsyncMock())
    monkeypatch.setattr(service, "_persist_runtime_state", AsyncMock())

    order = await service.place_order(
        token_id="token-123",
        side=OrderSide.BUY,
        price=0.6,
        size=5.0,
    )

    assert order.status == OrderStatus.FAILED
    assert order.error_message is not None
    assert "BUY pre-submit gate failed" in order.error_message
    assert client.post_calls == 0


@pytest.mark.asyncio
async def test_buy_pre_submit_gate_auto_recovers_collateral_allowance(monkeypatch):
    # Legacy chain-probe recovery path: on cache-insufficient the
    # legacy gate forces a refresh + re-probe.  The keeper-backed gate
    # explicitly does NOT do this on the hot path — it rejects fast
    # and lets the background reconciler converge.  This test verifies
    # the legacy fallback still recovers correctly when invoked.
    monkeypatch.setenv("HOMERUN_COLLATERAL_KEEPER_ENABLED", "0")
    _install_balance_allowance_modules(monkeypatch)

    class _StaleThenFreshClient(_BalanceClient):
        def __init__(self):
            super().__init__({}, builder_signature_type=1)
            self.snapshot_calls = 0

        def get_balance_allowance(self, params=None):
            self.snapshot_calls += 1
            if self.snapshot_calls == 1:
                return {"balance": "5.0", "allowance": "1.0"}
            return {"balance": "5.0", "allowance": "5.0"}

    service = LiveExecutionService()
    service._initialized = True
    service._wallet_address = "0x1234567890abcdef1234567890abcdef12345678"
    service._proxy_funder_address = service._wallet_address
    service._balance_signature_type = 1
    service._client = _StaleThenFreshClient()
    service.refresh_collateral_balance_allowance = AsyncMock(return_value=True)

    gate_ok, gate_error = await service._enforce_buy_pre_submit_gate(
        token_id="token-123",
        required_notional_usd=Decimal("4.0"),
    )

    assert gate_ok is True
    assert gate_error is None
    service.refresh_collateral_balance_allowance.assert_awaited_once()
    assert service._client.snapshot_calls >= 2


@pytest.mark.asyncio
async def test_prepare_sell_balance_allowance_refreshes_clob_cache_only(monkeypatch):
    _install_balance_allowance_modules(monkeypatch)
    service = LiveExecutionService()
    service._initialized = True
    service._wallet_address = "0x1234567890abcdef1234567890abcdef12345678"
    service._proxy_funder_address = service._wallet_address
    service._balance_signature_type = 1
    service._client = _BalanceClient({1: {"balance": "1.0", "allowance": "1.0"}}, builder_signature_type=1)
    service.refresh_conditional_balance_allowance = AsyncMock(return_value=True)
    service._select_signature_type_for_conditional_token = AsyncMock(return_value=1)

    refreshed = await service.prepare_sell_balance_allowance("token-123")

    assert refreshed is True
    service._select_signature_type_for_conditional_token.assert_awaited_once_with("token-123")
    service.refresh_conditional_balance_allowance.assert_awaited_once_with("token-123")
