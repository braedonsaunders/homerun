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
