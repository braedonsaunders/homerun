import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import Base, TradeSignal, Trader, TraderOrder
from services.trader_orchestrator import position_lifecycle
from services.strategies.base import ExitDecision
from tests.postgres_test_db import build_postgres_session_factory


async def _build_session_factory(_tmp_path: Path):
    return await build_postgres_session_factory(Base, "trader_position_lifecycle_resolution")


async def _seed_order(
    session: AsyncSession,
    *,
    direction: str = "buy_yes",
    order_id: str = "order-1",
    signal_id: str = "signal-1",
    mode: str = "paper",
    status: str = "executed",
    payload_json: dict | None = None,
) -> None:
    now = datetime.utcnow()
    session.add(
        Trader(
            id="trader-1",
            name="Crypto Trader",
            strategy_key="crypto_15m",
            strategy_version="v1",
            sources_json=["crypto"],
            params_json={},
            risk_limits_json={},
            metadata_json={},
            is_enabled=True,
            is_paused=False,
            interval_seconds=60,
            created_at=now,
            updated_at=now,
        )
    )
    session.add(
        TradeSignal(
            id=signal_id,
            source="crypto",
            signal_type="entry",
            strategy_type="crypto_15m",
            market_id="market-1",
            direction=direction,
            entry_price=0.4,
            dedupe_key=f"dedupe-{signal_id}",
            payload_json={"yes_price": 0.4, "no_price": 0.6},
            created_at=now,
            updated_at=now,
        )
    )
    session.add(
        TraderOrder(
            id=order_id,
            trader_id="trader-1",
            signal_id=signal_id,
            source="crypto",
            market_id="market-1",
            direction=direction,
            mode=mode,
            status=status,
            notional_usd=40.0,
            entry_price=0.4,
            effective_price=0.4,
            payload_json=payload_json or {},
            created_at=now,
            executed_at=now,
            updated_at=now,
        )
    )
    await session.commit()


@pytest.mark.asyncio
async def test_reconcile_infers_resolution_from_settled_prices_when_terminal(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(session, direction="buy_yes")
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": True,
                            "accepting_orders": False,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [1.0, 0.0],
                        }
                    }
                ),
            )

            result = await position_lifecycle.reconcile_paper_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 1
            assert result["by_status"]["resolved_win"] == 1
            assert order is not None
            assert order.status == "resolved_win"
            assert (order.payload_json or {}).get("position_close", {}).get("close_trigger") == "resolution_inferred"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_reconcile_prefers_explicit_winner_over_price_inference(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(session, direction="buy_yes")
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": True,
                            "accepting_orders": False,
                            "winner": 1,
                            "winning_outcome": None,
                            "outcome_prices": [1.0, 0.0],
                        }
                    }
                ),
            )

            result = await position_lifecycle.reconcile_paper_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 1
            assert result["by_status"]["resolved_loss"] == 1
            assert order is not None
            assert order.status == "resolved_loss"
            assert (order.payload_json or {}).get("position_close", {}).get("close_trigger") == "resolution"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_reconcile_does_not_infer_resolution_on_tradable_market(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(session, direction="buy_yes")
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [1.0, 0.0],
                        }
                    }
                ),
            )

            result = await position_lifecycle.reconcile_paper_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 0
            assert result["held"] == 1
            assert order is not None
            assert order.status == "executed"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_load_market_info_prefers_force_refreshed_condition_lookup(monkeypatch):
    market_id = "0x67ac92b23bf20382bff35cf4519403d3aab4b2015dd2e3943b6a511e32f1687e"
    stale_info = {
        "condition_id": market_id,
        "closed": False,
        "active": True,
        "outcomes": ["Up", "Down"],
        "outcome_prices": [0.52, 0.48],
    }
    fresh_info = {
        "condition_id": market_id,
        "closed": True,
        "active": True,
        "accepting_orders": False,
        "outcomes": ["Up", "Down"],
        "outcome_prices": [1.0, 0.0],
    }

    async def _condition_lookup(_market_id: str, **kwargs):
        if kwargs.get("force_refresh"):
            return fresh_info
        return stale_info

    condition_mock = AsyncMock(side_effect=_condition_lookup)
    token_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(position_lifecycle.polymarket_client, "get_market_by_condition_id", condition_mock)
    monkeypatch.setattr(position_lifecycle.polymarket_client, "get_market_by_token_id", token_mock)

    result = await position_lifecycle.load_market_info_for_orders([SimpleNamespace(market_id=market_id)])

    assert result[market_id] == fresh_info
    condition_mock.assert_awaited_once_with(market_id, force_refresh=True)
    assert token_mock.await_count == 0


@pytest.mark.asyncio
async def test_load_market_info_falls_back_to_cached_condition_lookup_when_refresh_misses(monkeypatch):
    market_id = "0x23e435a43d5304be68ddeffc53a1ad57163407672e37456c3a29df4c6c8dbe5b"
    cached_info = {
        "condition_id": market_id,
        "closed": False,
        "active": True,
        "outcomes": ["Up", "Down"],
        "outcome_prices": [0.61, 0.39],
    }

    async def _condition_lookup(_market_id: str, **kwargs):
        if kwargs.get("force_refresh"):
            return None
        return cached_info

    condition_mock = AsyncMock(side_effect=_condition_lookup)
    token_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(position_lifecycle.polymarket_client, "get_market_by_condition_id", condition_mock)
    monkeypatch.setattr(position_lifecycle.polymarket_client, "get_market_by_token_id", token_mock)

    result = await position_lifecycle.load_market_info_for_orders([SimpleNamespace(market_id=market_id)])

    assert result[market_id] == cached_info
    assert condition_mock.await_count == 2
    first_call, second_call = condition_mock.await_args_list
    assert first_call.args == (market_id,)
    assert first_call.kwargs == {"force_refresh": True}
    assert second_call.args == (market_id,)
    assert second_call.kwargs == {}
    assert token_mock.await_count == 0


@pytest.mark.asyncio
async def test_live_mark_to_market_keeps_position_open_and_records_pending_exit(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                payload_json={
                    "provider_reconciliation": {
                        "filled_size": 100.0,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 40.0,
                    }
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.55, 0.45],
                        }
                    }
                ),
            )

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
                force_mark_to_market=True,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 0
            assert result["would_close"] == 1
            assert result["held"] == 1
            assert order is not None
            assert order.status == "executed"
            assert order.actual_profit is None
            pending_exit = (order.payload_json or {}).get("pending_live_exit")
            assert isinstance(pending_exit, dict)
            assert pending_exit.get("close_trigger") == "manual_mark_to_market"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_external_wallet_flatten_closes_position_from_wallet_trade(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="open",
                payload_json={
                    "token_id": "token-1",
                    "provider_reconciliation": {
                        "filled_size": 5.0,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 2.0,
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.45, 0.55],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={"token-1": {"asset": "token-1", "size": 0.0}}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(
                    return_value={
                        "token-1": {
                            "trade_id": "trade-1",
                            "token_id": "token-1",
                            "size": 5.0,
                            "price": 0.21,
                            "timestamp": datetime.utcnow(),
                        }
                    }
                ),
            )

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 1
            assert order is not None
            assert order.status == "closed_loss"
            position_close = (order.payload_json or {}).get("position_close", {})
            assert position_close.get("close_trigger") == "external_wallet_flatten"
            assert position_close.get("price_source") == "wallet_trade"
            assert position_close.get("wallet_trade_id") == "trade-1"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_exit_submission_uses_gtc_for_lifecycle_close(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="open",
                payload_json={
                    "token_id": "token-1",
                    "provider_reconciliation": {
                        "filled_size": 5.0,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 2.0,
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.5, 0.5],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={"token-1": {"asset": "token-1", "size": 5.0, "curPrice": 0.5}}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            execute_mock = AsyncMock(
                return_value=SimpleNamespace(
                    status="failed",
                    error_message="mock_exit_failure",
                    order_id=None,
                    payload={},
                )
            )
            monkeypatch.setattr("services.live_execution_adapter.execute_live_order", execute_mock)

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={"live_take_profit_pct": 1.0},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 0
            assert order is not None
            pending_exit = (order.payload_json or {}).get("pending_live_exit")
            assert isinstance(pending_exit, dict)
            assert pending_exit.get("status") == "failed"
            assert execute_mock.await_count == 1
            assert execute_mock.await_args.kwargs.get("time_in_force") == "GTC"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_pending_exit_partial_fill_does_not_close_position(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="open",
                payload_json={
                    "provider_reconciliation": {
                        "filled_size": 7.4,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 2.96,
                    },
                    "pending_live_exit": {
                        "status": "submitted",
                        "close_trigger": "strategy:Stop loss hit",
                        "exit_size": 7.4,
                        "exit_order_id": "exit-order-1",
                        "provider_clob_order_id": "0xexit-order",
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.45, 0.55],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle.trading_service,
                "get_order_snapshots_by_clob_ids",
                AsyncMock(
                    return_value={
                        "0xexit-order": {
                            "normalized_status": "filled",
                            "filled_size": 6.22,
                            "average_fill_price": 0.18,
                        }
                    }
                ),
            )

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 0
            assert result["held"] >= 1
            assert order is not None
            assert order.status == "open"
            pending_exit = (order.payload_json or {}).get("pending_live_exit", {})
            assert pending_exit.get("status") == "submitted"
            assert pending_exit.get("fill_ratio") == pytest.approx(6.22 / 7.4, rel=1e-9)
            assert (order.payload_json or {}).get("position_close") is None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_pending_exit_closes_when_fill_threshold_met_with_stale_open_status(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="open",
                payload_json={
                    "provider_reconciliation": {
                        "filled_size": 7.4,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 2.96,
                    },
                    "pending_live_exit": {
                        "status": "submitted",
                        "close_trigger": "strategy:Stop loss hit",
                        "exit_size": 7.4,
                        "exit_order_id": "exit-order-1",
                        "provider_clob_order_id": "0xexit-order",
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.45, 0.55],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle.trading_service,
                "get_order_snapshots_by_clob_ids",
                AsyncMock(
                    return_value={
                        "0xexit-order": {
                            "normalized_status": "open",
                            "filled_size": 7.4,
                            "average_fill_price": 0.18,
                        }
                    }
                ),
            )

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 1
            assert order is not None
            assert order.status == "closed_loss"
            pending_exit = (order.payload_json or {}).get("pending_live_exit", {})
            assert pending_exit.get("status") == "filled"
            assert (order.payload_json or {}).get("position_close", {}).get("price_source") == "provider_exit_fill"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_pending_exit_closes_when_provider_filled_and_wallet_flat_with_lower_fill_ratio(
    tmp_path, monkeypatch
):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="open",
                payload_json={
                    "token_id": "token-1",
                    "provider_reconciliation": {
                        "filled_size": 7.4,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 2.96,
                    },
                    "pending_live_exit": {
                        "status": "submitted",
                        "close_trigger": "strategy:Stop loss hit",
                        "exit_size": 7.4,
                        "exit_order_id": "exit-order-1",
                        "provider_clob_order_id": "0xexit-order",
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.45, 0.55],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(
                    return_value={
                        "other-token": {"asset": "other-token", "size": 1.0, "curPrice": 0.5},
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle.trading_service,
                "get_order_snapshots_by_clob_ids",
                AsyncMock(
                    return_value={
                        "0xexit-order": {
                            "normalized_status": "filled",
                            "filled_size": 6.22,
                            "average_fill_price": 0.18,
                        }
                    }
                ),
            )

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 1
            assert order is not None
            assert order.status == "closed_loss"
            pending_exit = (order.payload_json or {}).get("pending_live_exit", {})
            assert pending_exit.get("status") == "filled"
            assert pending_exit.get("allow_partial_fill_terminal") is True
            assert (order.payload_json or {}).get("position_close", {}).get("price_source") == "provider_exit_fill"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_reopens_terminal_order_when_exit_fill_is_partial(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="closed_loss",
                payload_json={
                    "provider_reconciliation": {
                        "filled_size": 7.4,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 2.96,
                    },
                    "pending_live_exit": {
                        "status": "filled",
                        "close_trigger": "strategy:Stop loss hit",
                        "exit_size": 7.4,
                        "filled_size": 6.22,
                        "fill_ratio": 6.22 / 7.4,
                        "exit_order_id": "exit-order-1",
                        "provider_clob_order_id": "0xexit-order",
                    },
                    "position_close": {
                        "close_trigger": "strategy:Stop loss hit",
                        "filled_size": 6.22,
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.45, 0.55],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle.trading_service,
                "get_order_snapshots_by_clob_ids",
                AsyncMock(
                    return_value={
                        "0xexit-order": {
                            "normalized_status": "filled",
                            "filled_size": 6.22,
                            "average_fill_price": 0.18,
                        }
                    }
                ),
            )

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["state_updates"] >= 1
            assert order is not None
            assert order.status == "open"
            pending_exit = (order.payload_json or {}).get("pending_live_exit", {})
            assert pending_exit.get("reopen_reason") == "partial_exit_fill_below_threshold"
            assert pending_exit.get("status") == "submitted"
            assert (order.payload_json or {}).get("position_close") is None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_price_uses_wallet_mark_when_market_price_unavailable(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="open",
                payload_json={
                    "token_id": "token-1",
                    "provider_reconciliation": {
                        "filled_size": 5.0,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 2.0,
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={"token-1": {"asset": "token-1", "size": 5.0, "curPrice": 0.47}}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            signal = await session.get(TradeSignal, "signal-1")
            assert signal is not None
            signal.payload_json = {}
            await session.commit()

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 0
            assert result["held"] == 1
            assert order is not None
            position_state = (order.payload_json or {}).get("position_state", {})
            assert position_state.get("last_mark_source") == "wallet_mark"
            assert position_state.get("last_mark_price") == pytest.approx(0.47, abs=1e-9)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_paper_strategy_reverse_intent_emits_reverse_signal(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                payload_json={
                    "strategy_type": "reverse_test",
                    "strategy_context": {},
                },
            )

            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.28, 0.72],
                            "token_ids": ["yes-token", "no-token"],
                            "outcomes": ["Yes", "No"],
                        }
                    }
                ),
            )

            class _ReverseExitStrategy:
                def should_exit(self, position, market_state):
                    return ExitDecision(
                        "close",
                        "Velocity guard",
                        close_price=0.28,
                        payload={
                            "reverse_intent": {
                                "enabled": True,
                                "direction": "buy_no",
                                "entry_price": 0.72,
                                "signal_type": "crypto_worker_reverse",
                                "confidence": 0.68,
                                "edge_percent": 2.8,
                                "size_multiplier": 1.2,
                                "strategy_type": "reverse_test",
                                "expires_in_seconds": 120,
                            }
                        },
                    )

            monkeypatch.setattr(
                "services.strategy_loader.strategy_loader.get_strategy",
                lambda strategy_key: (
                    SimpleNamespace(instance=_ReverseExitStrategy())
                    if str(strategy_key or "").strip().lower() == "reverse_test"
                    else None
                ),
            )

            upsert_mock = AsyncMock(return_value=SimpleNamespace(id="reverse-signal-paper-1"))
            refresh_mock = AsyncMock(return_value=[])
            publish_mock = AsyncMock(return_value=None)
            monkeypatch.setattr(position_lifecycle, "upsert_trade_signal", upsert_mock)
            monkeypatch.setattr(position_lifecycle, "refresh_trade_signal_snapshots", refresh_mock)
            monkeypatch.setattr("services.event_bus.event_bus.publish", publish_mock)

            result = await position_lifecycle.reconcile_paper_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 1
            assert result.get("reverse_signals_emitted") == 1
            assert upsert_mock.await_count == 1
            kwargs = upsert_mock.await_args.kwargs
            assert kwargs["source"] == "crypto"
            assert kwargs["direction"] == "buy_no"
            assert kwargs["strategy_type"] == "reverse_test"

            assert order is not None
            pending_reverse = (order.payload_json or {}).get("pending_reverse_entry", {})
            assert pending_reverse.get("status") == "emitted"
            assert pending_reverse.get("signal_id") == "reverse-signal-paper-1"
            assert any(call.args and call.args[0] == "trade_signal_batch" for call in publish_mock.await_args_list)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_pending_exit_fill_emits_reverse_signal_when_armed(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="open",
                payload_json={
                    "strategy_type": "reverse_test",
                    "strategy_context": {},
                    "token_id": "yes-token",
                    "provider_reconciliation": {
                        "filled_size": 5.0,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 2.0,
                    },
                    "pending_live_exit": {
                        "status": "submitted",
                        "close_trigger": "strategy:Velocity guard",
                        "close_price": 0.20,
                        "exit_size": 5.0,
                        "provider_clob_order_id": "0xexit-order-1",
                    },
                    "pending_reverse_entry": {
                        "status": "armed",
                        "direction": "buy_no",
                        "entry_price": 0.72,
                        "signal_type": "crypto_worker_reverse",
                        "confidence": 0.66,
                        "edge_percent": 2.4,
                        "size_multiplier": 1.1,
                        "strategy_type": "reverse_test",
                        "armed_at": "2099-01-01T00:00:00Z",
                        "attempt_index": 1,
                        "expires_at": "2099-01-01T00:05:00Z",
                    },
                },
            )

            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.20, 0.80],
                            "token_ids": ["yes-token", "no-token"],
                            "outcomes": ["Yes", "No"],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={"yes-token": {"asset": "yes-token", "size": 0.0}}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )
            monkeypatch.setattr(
                position_lifecycle.trading_service,
                "get_order_snapshots_by_clob_ids",
                AsyncMock(
                    return_value={
                        "0xexit-order-1": {
                            "normalized_status": "filled",
                            "filled_size": 5.0,
                            "average_fill_price": 0.20,
                        }
                    }
                ),
            )

            upsert_mock = AsyncMock(return_value=SimpleNamespace(id="reverse-signal-live-1"))
            refresh_mock = AsyncMock(return_value=[])
            publish_mock = AsyncMock(return_value=None)
            monkeypatch.setattr(position_lifecycle, "upsert_trade_signal", upsert_mock)
            monkeypatch.setattr(position_lifecycle, "refresh_trade_signal_snapshots", refresh_mock)
            monkeypatch.setattr("services.event_bus.event_bus.publish", publish_mock)

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
            )
            order = await session.get(TraderOrder, "order-1")

            assert result["closed"] == 1
            assert result.get("reverse_signals_emitted") == 1
            assert upsert_mock.await_count == 1
            kwargs = upsert_mock.await_args.kwargs
            assert kwargs["source"] == "crypto"
            assert kwargs["direction"] == "buy_no"
            assert kwargs["strategy_type"] == "reverse_test"

            assert order is not None
            assert order.status == "closed_loss"
            pending_reverse = (order.payload_json or {}).get("pending_reverse_entry", {})
            assert pending_reverse.get("status") == "emitted"
            assert pending_reverse.get("signal_id") == "reverse-signal-live-1"
            assert any(call.args and call.args[0] == "trade_signal_batch" for call in publish_mock.await_args_list)
    finally:
        await engine.dispose()
