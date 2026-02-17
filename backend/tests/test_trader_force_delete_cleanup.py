import sys
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import Base, Trader, TraderOrder  # noqa: E402
from services.trader_orchestrator_state import (  # noqa: E402
    OPEN_ORDER_STATUSES,
    delete_trader,
    sync_trader_position_inventory,
)


async def _build_session_factory(tmp_path: Path):
    db_path = tmp_path / "trader_force_delete_cleanup.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine, session_factory


@pytest.mark.asyncio
async def test_force_delete_cleans_up_active_orders_before_delete(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    trader_id = "trader-force-delete"
    try:
        async with session_factory() as session:
            now = datetime.utcnow()
            session.add(
                Trader(
                    id=trader_id,
                    name="Force Delete Trader",
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

            order_rows = list(
                (
                    await session.execute(
                        select(TraderOrder).where(TraderOrder.trader_id == trader_id)
                    )
                )
                .scalars()
                .all()
            )
            assert all(
                str(row.status or "").strip().lower() not in OPEN_ORDER_STATUSES
                for row in order_rows
            )
    finally:
        await engine.dispose()
