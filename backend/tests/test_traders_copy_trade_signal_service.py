import asyncio
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import services.traders_copy_trade_signal_service as copy_trade_signal_service_module
from services.traders_copy_trade_signal_service import (
    TradersCopyTradeSignalService,
    _TokenMarketSnapshot,
)
from services.wallet_ws_monitor import WalletTradeEvent
from utils.utcnow import utcnow


def _event(
    *,
    wallet: str = "0xabc",
    tx_hash: str = "0xhash",
    order_hash: str = "0xorder",
    log_index: int = 1,
    size: float = 25.0,
    price: float = 0.62,
) -> WalletTradeEvent:
    now = utcnow()
    return WalletTradeEvent(
        wallet_address=wallet,
        token_id="token-1",
        side="BUY",
        size=size,
        price=price,
        tx_hash=tx_hash,
        order_hash=order_hash,
        log_index=log_index,
        block_number=123,
        timestamp=now,
        detected_at=now,
        latency_ms=12.5,
    )


def test_copy_trade_event_dedupe_distinguishes_per_log_fill_identity():
    service = TradersCopyTradeSignalService()

    first = _event(log_index=10, size=10.0, price=0.50)
    second_different_log = _event(log_index=11, size=10.0, price=0.50)
    second_different_size = _event(log_index=10, size=12.0, price=0.50)
    exact_duplicate = _event(log_index=10, size=10.0, price=0.50)

    assert service._is_duplicate_event(first) is False
    assert service._is_duplicate_event(second_different_log) is False
    assert service._is_duplicate_event(second_different_size) is False
    assert service._is_duplicate_event(exact_duplicate) is True


@pytest.mark.asyncio
async def test_queue_overflow_falls_back_to_direct_processing(monkeypatch):
    service = TradersCopyTradeSignalService()
    service._running = True
    service._tracked_wallets = {"0xabc"}
    service._execution_wallet = ""
    service._queue = asyncio.Queue(maxsize=1)
    await service._queue.put(_event(tx_hash="0xprimed", log_index=1))

    called = asyncio.Event()
    observed: dict[str, str] = {}

    async def fake_process(event: WalletTradeEvent) -> None:
        observed["tx_hash"] = str(event.tx_hash)
        called.set()

    monkeypatch.setattr(service, "_process_wallet_trade_event", fake_process)

    overflow_event = _event(tx_hash="0xfallback", log_index=2)
    await service._on_wallet_trade(overflow_event)
    await asyncio.wait_for(called.wait(), timeout=1.0)
    assert observed["tx_hash"] == "0xfallback"


@pytest.mark.asyncio
async def test_on_wallet_trade_ignores_execution_wallet():
    service = TradersCopyTradeSignalService()
    service._running = True
    service._tracked_wallets = {"0xabc"}
    service._execution_wallet = "0xabc"
    service._queue = asyncio.Queue(maxsize=10)

    await service._on_wallet_trade(_event(wallet="0xabc", tx_hash="0xself"))
    assert service._queue.empty()


@pytest.mark.asyncio
async def test_process_wallet_trade_skips_unresolved_token_outcome(monkeypatch):
    service = TradersCopyTradeSignalService()

    async def _unresolved(_: str) -> _TokenMarketSnapshot:
        return _TokenMarketSnapshot(
            market_id="market-1",
            market_question="Question",
            market_slug="slug",
            outcome="",
            liquidity=1000.0,
        )

    async def _unexpected_bridge(*args, **kwargs):
        raise AssertionError("bridge_opportunities_to_signals should not run when token outcome is unresolved")

    monkeypatch.setattr(service, "_resolve_market_snapshot", _unresolved)
    monkeypatch.setattr(copy_trade_signal_service_module, "bridge_opportunities_to_signals", _unexpected_bridge)

    await service._process_wallet_trade_event(_event())


@pytest.mark.asyncio
async def test_process_wallet_trade_uses_strategy_detect_and_bridge(monkeypatch):
    service = TradersCopyTradeSignalService()

    async def _resolved(_: str) -> _TokenMarketSnapshot:
        return _TokenMarketSnapshot(
            market_id="market-1",
            market_question="Question",
            market_slug="slug",
            outcome="YES",
            liquidity=1000.0,
        )

    class _FakeStrategy:
        def __init__(self) -> None:
            self.calls: list[list[dict]] = []

        async def detect_async(self, events, markets, prices):
            self.calls.append(events)
            return [{"ok": True}]

    fake_strategy = _FakeStrategy()
    bridge_calls: list[dict] = []

    async def _bridge(session, opportunities, source, **kwargs):
        bridge_calls.append(
            {
                "opportunities": opportunities,
                "source": source,
                "kwargs": kwargs,
            }
        )
        return 1

    monkeypatch.setattr(service, "_resolve_market_snapshot", _resolved)
    monkeypatch.setattr(service, "_resolve_copy_trade_strategy", lambda: fake_strategy)
    monkeypatch.setattr(copy_trade_signal_service_module, "bridge_opportunities_to_signals", _bridge)

    await service._process_wallet_trade_event(_event())

    assert len(fake_strategy.calls) == 1
    assert len(bridge_calls) == 1
    assert bridge_calls[0]["source"] == "traders"
    assert bridge_calls[0]["kwargs"]["signal_type_override"] == "copy_trade"
    assert bridge_calls[0]["kwargs"]["default_ttl_minutes"] == 15
    assert bridge_calls[0]["kwargs"]["refresh_prices"] is False


@pytest.mark.asyncio
async def test_resolve_market_snapshot_reuses_stale_cache_when_lookup_fails(monkeypatch):
    service = TradersCopyTradeSignalService()
    stale_snapshot = _TokenMarketSnapshot(
        market_id="market-1",
        market_question="Question",
        market_slug="slug",
        outcome="YES",
        liquidity=1234.0,
    )
    service._token_cache["token-1"] = (
        utcnow() - timedelta(seconds=service._token_cache_ttl_seconds + 30),
        stale_snapshot,
    )

    async def _raise_lookup(*args, **kwargs):
        raise RuntimeError("lookup failed")

    monkeypatch.setattr(copy_trade_signal_service_module.polymarket_client, "get_market_by_token_id", _raise_lookup)

    resolved = await service._resolve_market_snapshot("token-1")
    assert resolved == stale_snapshot


@pytest.mark.asyncio
async def test_copy_existing_open_positions_skips_when_operator_disables():
    service = TradersCopyTradeSignalService()

    summary = await service.copy_existing_open_positions_for_trader(
        trader_id="trader-1",
        copy_existing_positions=False,
    )

    assert summary["status"] == "skipped"
    assert summary["reason"] == "operator_disabled_copy_existing_positions"
    assert summary["signals_created"] == 0


@pytest.mark.asyncio
async def test_copy_existing_open_positions_bootstraps_wallet_positions(monkeypatch):
    service = TradersCopyTradeSignalService()
    trader_id = "trader-bootstrap"
    trader_row = SimpleNamespace(
        source_configs_json=[
            {
                "source_key": "traders",
                "strategy_key": "traders_copy_trade",
                "strategy_params": {
                    "copy_existing_positions_on_start": True,
                    "traders_scope": {
                        "modes": ["individual"],
                        "individual_wallets": ["0xabc"],
                        "group_ids": [],
                    },
                },
            }
        ]
    )

    class _FakeSession:
        async def get(self, model, key):
            return trader_row if key == trader_id else None

    class _FakeSessionContext:
        async def __aenter__(self):
            return _FakeSession()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _FakeStrategy:
        async def detect_async(self, events, markets, prices):
            return [{"ok": True}]

    bridge_calls: list[dict] = []

    async def _bridge(session, opportunities, source, **kwargs):
        bridge_calls.append({"source": source, "kwargs": kwargs, "count": len(opportunities)})
        return len(opportunities)

    async def _resolved(_: str) -> _TokenMarketSnapshot:
        return _TokenMarketSnapshot(
            market_id="market-1",
            market_question="Question",
            market_slug="slug",
            outcome="YES",
            liquidity=1000.0,
        )

    monkeypatch.setattr(copy_trade_signal_service_module, "AsyncSessionLocal", lambda: _FakeSessionContext())
    monkeypatch.setattr(service, "_resolve_wallets_for_scopes", AsyncMock(return_value={"0xabc"}))
    monkeypatch.setattr(copy_trade_signal_service_module.live_execution_service, "get_execution_wallet_address", lambda: "")
    monkeypatch.setattr(copy_trade_signal_service_module.polymarket_client, "get_wallet_positions_with_prices", AsyncMock(return_value=[{
        "asset": "token-1",
        "size": 15,
        "avgPrice": 0.61,
    }]))
    monkeypatch.setattr(service, "_resolve_copy_trade_strategy", lambda: _FakeStrategy())
    monkeypatch.setattr(service, "_resolve_market_snapshot", _resolved)
    monkeypatch.setattr(copy_trade_signal_service_module, "bridge_opportunities_to_signals", _bridge)

    summary = await service.copy_existing_open_positions_for_trader(trader_id=trader_id, copy_existing_positions=None)

    assert summary["status"] == "completed"
    assert summary["wallets_targeted"] == 1
    assert summary["wallets_scanned"] == 1
    assert summary["positions_seen"] == 1
    assert summary["positions_queued"] == 1
    assert summary["signals_created"] == 1
    assert len(bridge_calls) == 1
    assert bridge_calls[0]["source"] == "traders"
    assert bridge_calls[0]["kwargs"]["signal_type_override"] == "copy_trade"
