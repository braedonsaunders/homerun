import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import Base, Trader
from services.trader_orchestrator_state import (
    create_trader,
    delete_trader,
    get_orchestrator_overview,
)
from workers import trader_orchestrator_worker


async def _build_session_factory(tmp_path: Path):
    db_path = tmp_path / "trader_orchestrator_no_auto_seed.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine, session_factory


@pytest.mark.asyncio
async def test_overview_does_not_seed_default_traders(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            overview = await get_orchestrator_overview(session)
            count = int((await session.execute(select(func.count(Trader.id)))).scalar() or 0)

        assert overview["traders"] == []
        assert int(overview["metrics"]["traders_total"]) == 0
        assert count == 0
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_overview_stays_empty_after_deleting_last_trader(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            trader = await create_trader(
                session,
                {
                    "name": "One-Off Trader",
                    "strategy_key": "crypto_15m",
                    "sources": ["crypto"],
                },
            )
            deleted = await delete_trader(session, trader["id"])
            overview = await get_orchestrator_overview(session)
            count = int((await session.execute(select(func.count(Trader.id)))).scalar() or 0)

        assert deleted is True
        assert overview["traders"] == []
        assert int(overview["metrics"]["traders_total"]) == 0
        assert count == 0
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_worker_loop_does_not_seed_default_traders(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", session_factory)
        monkeypatch.setattr(trader_orchestrator_worker, "expire_stale_signals", AsyncMock())
        monkeypatch.setattr(
            trader_orchestrator_worker,
            "read_orchestrator_control",
            AsyncMock(
                return_value={
                    "is_enabled": False,
                    "is_paused": True,
                    "run_interval_seconds": 1,
                }
            ),
        )
        monkeypatch.setattr(trader_orchestrator_worker, "write_orchestrator_snapshot", AsyncMock())

        async def _cancel_sleep(_interval: float):
            raise asyncio.CancelledError()

        monkeypatch.setattr(trader_orchestrator_worker.asyncio, "sleep", _cancel_sleep)

        with pytest.raises(asyncio.CancelledError):
            await trader_orchestrator_worker.run_worker_loop()

        async with session_factory() as session:
            count = int((await session.execute(select(func.count(Trader.id)))).scalar() or 0)

        assert count == 0
    finally:
        await engine.dispose()
