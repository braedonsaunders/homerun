import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api import routes_traders
from models.database import Base, TradeSignal, Trader, TraderOrder
from services.trader_orchestrator import position_lifecycle
from services.trader_orchestrator_state import (
    get_open_order_count_for_trader,
    get_open_position_count_for_trader,
    sync_trader_position_inventory,
)


async def _build_session_factory(tmp_path: Path):
    db_path = tmp_path / "routes_trader_positions_cleanup.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine, session_factory


async def _seed_trader_with_order(session: AsyncSession) -> None:
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
            id="signal-1",
            source="crypto",
            signal_type="entry",
            market_id="market-1",
            direction="buy_yes",
            entry_price=0.4,
            dedupe_key="dedupe-1",
            payload_json={"yes_price": 0.7, "no_price": 0.3},
            created_at=now,
            updated_at=now,
        )
    )
    session.add(
        TraderOrder(
            id="order-1",
            trader_id="trader-1",
            signal_id="signal-1",
            source="crypto",
            market_id="market-1",
            direction="buy_yes",
            mode="paper",
            status="executed",
            notional_usd=40.0,
            entry_price=0.4,
            effective_price=0.4,
            created_at=now,
            executed_at=now,
            updated_at=now,
        )
    )
    await session.commit()


@pytest.mark.asyncio
async def test_mark_to_market_dry_run_keeps_status(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_trader_with_order(session)
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(return_value={"market-1": {"yes_price": 0.7, "no_price": 0.3}}),
            )

            result = await routes_traders.cleanup_trader_positions(
                "trader-1",
                routes_traders.TraderPositionCleanupRequest(
                    scope=routes_traders.TraderPositionCleanupScope.paper,
                    method=routes_traders.TraderPositionCleanupMethod.mark_to_market,
                    dry_run=True,
                ),
                session=session,
            )

            order = await session.get(TraderOrder, "order-1")
            assert result["matched"] == 1
            assert result["updated"] == 0
            assert len(result["details"]) == 1
            assert order is not None
            assert order.status == "executed"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_mark_to_market_updates_realized_pnl_and_status(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_trader_with_order(session)
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(return_value={"market-1": {"yes_price": 0.7, "no_price": 0.3}}),
            )

            result = await routes_traders.cleanup_trader_positions(
                "trader-1",
                routes_traders.TraderPositionCleanupRequest(
                    scope=routes_traders.TraderPositionCleanupScope.paper,
                    method=routes_traders.TraderPositionCleanupMethod.mark_to_market,
                    dry_run=False,
                ),
                session=session,
            )

            order = await session.get(TraderOrder, "order-1")
            assert result["updated"] == 1
            assert order is not None
            assert order.status == "resolved_win"
            assert pytest.approx(float(order.actual_profit or 0.0), rel=1e-9) == 30.0
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_cleanup_requires_confirm(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_trader_with_order(session)

            with pytest.raises(HTTPException) as excinfo:
                await routes_traders.cleanup_trader_positions(
                    "trader-1",
                    routes_traders.TraderPositionCleanupRequest(
                        scope=routes_traders.TraderPositionCleanupScope.live,
                        method=routes_traders.TraderPositionCleanupMethod.cancel,
                        confirm_live=False,
                    ),
                    session=session,
                )
            assert excinfo.value.status_code == 409
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_open_order_count_includes_executed_paper_positions(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_trader_with_order(session)
            count = await get_open_order_count_for_trader(session, "trader-1", mode="paper")
            assert count == 1
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_open_position_count_aggregates_same_market_direction(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            await _seed_trader_with_order(session)
            now = datetime.utcnow()
            session.add(
                TraderOrder(
                    id="order-2",
                    trader_id="trader-1",
                    signal_id="signal-1",
                    source="crypto",
                    market_id="market-1",
                    direction="buy_yes",
                    mode="paper",
                    status="submitted",
                    notional_usd=20.0,
                    entry_price=0.5,
                    effective_price=0.5,
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()

            await sync_trader_position_inventory(session, trader_id="trader-1", mode="paper")
            open_orders = await get_open_order_count_for_trader(session, "trader-1", mode="paper")
            open_positions = await get_open_position_count_for_trader(session, "trader-1", mode="paper")

            assert open_orders == 2
            assert open_positions == 1
    finally:
        await engine.dispose()
