import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import Base, Trader, TraderOrder  # noqa: E402
from services.trader_orchestrator_state import (  # noqa: E402
    OPEN_ORDER_STATUSES,
    delete_trader,
    sync_trader_position_inventory,
)
import services.trader_orchestrator_state as trader_state  # noqa: E402
from tests.postgres_test_db import build_postgres_session_factory  # noqa: E402
from utils.utcnow import utcnow  # noqa: E402


async def _build_session_factory(_tmp_path: Path):
    return await build_postgres_session_factory(Base, "trader_force_delete_cleanup")


@pytest.mark.asyncio
async def test_force_delete_cleans_up_active_orders_before_delete(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "trader-force-delete"
    try:
        async with session_factory() as session:
            now = utcnow()
            session.add(
                Trader(
                    id=trader_id,
                    name="Force Delete Trader",
                    source_configs_json=[{"source_key": "crypto", "strategy_key": "btc_eth_highfreq", "strategy_params": {}}],
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
                TraderOrder(
                    id="order-force-delete",
                    trader_id=trader_id,
                    source="crypto",
                    market_id="market-force-delete",
                    direction="buy_yes",
                    mode="paper",
                    status="executed",
                    notional_usd=25.0,
                    entry_price=0.5,
                    effective_price=0.5,
                    payload_json={},
                    created_at=now,
                    executed_at=now,
                    updated_at=now,
                )
            )
            await session.commit()
            await sync_trader_position_inventory(session, trader_id=trader_id)

            deleted = await delete_trader(session, trader_id, force=True)
            assert deleted is True
            assert await session.get(Trader, trader_id) is None

            order_rows = list(
                (await session.execute(select(TraderOrder).where(TraderOrder.trader_id == trader_id))).scalars().all()
            )
            assert len(order_rows) == 1
            assert all(str(row.status or "").strip().lower() not in OPEN_ORDER_STATUSES for row in order_rows)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_delete_allows_paper_exposure_without_force(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "trader-paper-delete"
    try:
        async with session_factory() as session:
            now = utcnow()
            session.add(
                Trader(
                    id=trader_id,
                    name="Paper Exposure Trader",
                    source_configs_json=[{"source_key": "crypto", "strategy_key": "btc_eth_highfreq", "strategy_params": {}}],
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
                TraderOrder(
                    id="order-paper-delete",
                    trader_id=trader_id,
                    source="crypto",
                    market_id="market-paper-delete",
                    direction="buy_yes",
                    mode="paper",
                    status="executed",
                    notional_usd=20.0,
                    entry_price=0.44,
                    effective_price=0.44,
                    payload_json={},
                    created_at=now,
                    executed_at=now,
                    updated_at=now,
                )
            )
            await session.commit()
            await sync_trader_position_inventory(session, trader_id=trader_id)

            deleted = await delete_trader(session, trader_id, force=False)
            assert deleted is True
            assert await session.get(Trader, trader_id) is None
            order_row = await session.get(TraderOrder, "order-paper-delete")
            assert order_row is not None
            assert order_row.trader_id == trader_id
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_delete_blocks_live_exposure_without_force(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "trader-live-blocked-delete"
    try:
        async with session_factory() as session:
            now = utcnow()
            session.add(
                Trader(
                    id=trader_id,
                    name="Live Exposure Trader",
                    source_configs_json=[{"source_key": "crypto", "strategy_key": "btc_eth_highfreq", "strategy_params": {}}],
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
                TraderOrder(
                    id="order-live-blocked-delete",
                    trader_id=trader_id,
                    source="crypto",
                    market_id="market-live-blocked-delete",
                    direction="buy_yes",
                    mode="live",
                    status="executed",
                    notional_usd=30.0,
                    entry_price=0.51,
                    effective_price=0.51,
                    payload_json={},
                    created_at=now,
                    executed_at=now,
                    updated_at=now,
                )
            )
            await session.commit()
            await sync_trader_position_inventory(session, trader_id=trader_id)

            with pytest.raises(ValueError, match="non-paper exposure"):
                await delete_trader(session, trader_id, force=False)
            assert await session.get(Trader, trader_id) is not None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_force_delete_override_skips_provider_cancel_path(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "trader-force-delete-override"
    monkeypatch.setattr(
        trader_state,
        "cleanup_trader_open_orders",
        AsyncMock(side_effect=AssertionError("cleanup_trader_open_orders should not be called for force delete")),
    )
    try:
        async with session_factory() as session:
            now = utcnow()
            session.add(
                Trader(
                    id=trader_id,
                    name="Force Delete Override Trader",
                    source_configs_json=[{"source_key": "crypto", "strategy_key": "btc_eth_highfreq", "strategy_params": {}}],
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
                TraderOrder(
                    id="order-force-delete-override",
                    trader_id=trader_id,
                    source="crypto",
                    market_id="market-force-delete-override",
                    direction="buy_yes",
                    mode="live",
                    status="executed",
                    notional_usd=31.0,
                    entry_price=0.53,
                    effective_price=0.53,
                    payload_json={},
                    created_at=now,
                    executed_at=now,
                    updated_at=now,
                )
            )
            await session.commit()
            await sync_trader_position_inventory(session, trader_id=trader_id)

            deleted = await delete_trader(session, trader_id, force=True)
            assert deleted is True
            assert await session.get(Trader, trader_id) is None
            order_row = await session.get(TraderOrder, "order-force-delete-override")
            assert order_row is not None
            assert order_row.trader_id == trader_id
    finally:
        await engine.dispose()
