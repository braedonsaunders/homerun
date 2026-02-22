import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import func, select

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import Base, Trader
from services.trader_orchestrator_state import (
    create_trader,
    delete_trader,
    enforce_manual_start_on_startup,
    get_orchestrator_overview,
    read_orchestrator_snapshot,
    update_orchestrator_control,
    update_trader,
    write_orchestrator_snapshot,
)
from tests.postgres_test_db import build_postgres_session_factory
from utils.utcnow import utcnow
from workers import trader_orchestrator_worker


async def _build_session_factory(_tmp_path: Path):
    return await build_postgres_session_factory(Base, "trader_orchestrator_no_auto_seed")


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
                    "source_configs": [
                        {
                            "source_key": "crypto",
                            "strategy_key": "crypto_15m",
                            "strategy_params": {},
                        }
                    ],
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

        monkeypatch.setattr(trader_orchestrator_worker, "_worker_sleep", _cancel_sleep)

        with pytest.raises(asyncio.CancelledError):
            await trader_orchestrator_worker.run_worker_loop()

        async with session_factory() as session:
            count = int((await session.execute(select(func.count(Trader.id)))).scalar() or 0)

        assert count == 0
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_create_trader_normalizes_legacy_default_strategy_key(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            trader = await create_trader(
                session,
                {
                    "name": "Legacy Default Trader",
                    "source_configs": [
                        {
                            "source_key": "crypto",
                            "strategy_key": "strategy.default",
                            "strategy_params": {},
                        }
                    ],
                },
            )

        assert trader["source_configs"][0]["strategy_key"] == "btc_eth_highfreq"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_update_trader_rejects_unknown_strategy_key(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            trader = await create_trader(
                session,
                {
                    "name": "Strict Strategy Trader",
                    "source_configs": [
                        {
                            "source_key": "crypto",
                            "strategy_key": "crypto_15m",
                            "strategy_params": {},
                        }
                    ],
                },
            )

            with pytest.raises(ValueError, match="Unknown strategy_key"):
                await update_trader(
                    session,
                    trader["id"],
                    {
                        "source_configs": [
                            {
                                "source_key": "crypto",
                                "strategy_key": "not_a_real_strategy",
                                "strategy_params": {},
                            }
                        ],
                    },
                )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_startup_enforces_manual_orchestrator_start(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            started = await update_orchestrator_control(
                session,
                is_enabled=True,
                is_paused=False,
                mode="live",
                requested_run_at=utcnow(),
            )
            await write_orchestrator_snapshot(
                session,
                running=True,
                enabled=True,
                current_activity="Cycle decisions=1 orders=1",
                interval_seconds=int(started.get("run_interval_seconds") or 5),
                last_run_at=utcnow(),
            )

            control = await enforce_manual_start_on_startup(session)
            snapshot = await read_orchestrator_snapshot(session)

        assert control["is_enabled"] is False
        assert control["is_paused"] is True
        assert control["mode"] == "paper"
        assert control["requested_run_at"] is None
        assert snapshot["running"] is False
        assert snapshot["enabled"] is False
        assert snapshot["interval_seconds"] == int(control.get("run_interval_seconds") or 5)
        assert snapshot["current_activity"] == "Stopped on startup; manual start required"
    finally:
        await engine.dispose()
