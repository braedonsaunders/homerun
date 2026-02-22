import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import Base, Trader, TraderOrder
from services import trader_orchestrator_state
from services.trader_orchestrator_state import (
    cleanup_trader_open_orders,
    get_open_order_count_for_trader,
    get_open_position_count_for_trader,
    reconcile_live_provider_orders,
    sync_trader_position_inventory,
)
from tests.postgres_test_db import build_postgres_session_factory
from utils.utcnow import utcnow


async def _build_session_factory(_tmp_path: Path):
    return await build_postgres_session_factory(Base, "trader_live_provider_reconciliation")


async def _seed_trader(session, trader_id: str) -> None:
    now = utcnow()
    session.add(
        Trader(
            id=trader_id,
            name="Live Trader",
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
    await session.commit()


@pytest.mark.asyncio
async def test_sync_inventory_ignores_unfilled_live_open_orders(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "live-trader-unfilled"
    try:
        async with session_factory() as session:
            await _seed_trader(session, trader_id)
            now = utcnow()
            session.add(
                TraderOrder(
                    id="live-order-unfilled",
                    trader_id=trader_id,
                    source="crypto",
                    market_id="market-live-1",
                    direction="buy_yes",
                    mode="live",
                    status="open",
                    notional_usd=40.0,
                    entry_price=0.4,
                    effective_price=0.4,
                    payload_json={"provider_clob_order_id": "clob-unfilled"},
                    created_at=now,
                    executed_at=None,
                    updated_at=now,
                )
            )
            await session.commit()

            await sync_trader_position_inventory(session, trader_id=trader_id, mode="live")

            open_orders = await get_open_order_count_for_trader(session, trader_id, mode="live")
            open_positions = await get_open_position_count_for_trader(session, trader_id, mode="live")
            assert open_orders == 1
            assert open_positions == 0
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_reconcile_live_provider_orders_updates_fill_state(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "live-trader-reconcile"
    try:
        async with session_factory() as session:
            await _seed_trader(session, trader_id)
            now = utcnow()
            session.add(
                TraderOrder(
                    id="live-order-1",
                    trader_id=trader_id,
                    source="crypto",
                    market_id="market-live-2",
                    direction="buy_yes",
                    mode="live",
                    status="open",
                    notional_usd=50.0,
                    entry_price=0.5,
                    effective_price=0.5,
                    payload_json={"provider_clob_order_id": "clob-live-2"},
                    created_at=now,
                    executed_at=None,
                    updated_at=now,
                )
            )
            await session.commit()

            monkeypatch.setattr(
                trader_orchestrator_state.trading_service,
                "get_order_snapshots_by_clob_ids",
                AsyncMock(
                    return_value={
                        "clob-live-2": {
                            "clob_order_id": "clob-live-2",
                            "normalized_status": "filled",
                            "filled_size": 30.0,
                            "average_fill_price": 0.41,
                            "filled_notional_usd": 12.3,
                            "limit_price": 0.42,
                        }
                    }
                ),
            )

            result = await reconcile_live_provider_orders(session, trader_id=trader_id, commit=True)
            assert result["updated_orders"] == 1
            assert result["status_changes"] == 1

            refreshed = await session.get(TraderOrder, "live-order-1")
            assert refreshed is not None
            assert refreshed.status == "executed"
            assert refreshed.notional_usd == pytest.approx(12.3)
            assert refreshed.effective_price == pytest.approx(0.41)
            assert refreshed.executed_at is not None
            assert isinstance((refreshed.payload_json or {}).get("provider_reconciliation"), dict)

            await sync_trader_position_inventory(session, trader_id=trader_id, mode="live")
            open_positions = await get_open_position_count_for_trader(session, trader_id, mode="live")
            assert open_positions == 1
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_live_cleanup_requires_provider_cancel(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "live-trader-cleanup"
    try:
        async with session_factory() as session:
            await _seed_trader(session, trader_id)
            now = utcnow()
            session.add(
                TraderOrder(
                    id="live-order-no-provider",
                    trader_id=trader_id,
                    source="crypto",
                    market_id="market-live-3",
                    direction="buy_yes",
                    mode="live",
                    status="open",
                    notional_usd=25.0,
                    entry_price=0.5,
                    effective_price=0.5,
                    payload_json={},
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()

            with pytest.raises(ValueError, match="missing provider order identifiers"):
                await cleanup_trader_open_orders(
                    session,
                    trader_id=trader_id,
                    scope="live",
                    dry_run=False,
                    target_status="cancelled",
                )
    finally:
        await engine.dispose()
