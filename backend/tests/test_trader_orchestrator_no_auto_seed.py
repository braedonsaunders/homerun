import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import func, select, text

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import Base, Trader
from services.trader_orchestrator_state import (
    create_trader,
    delete_trader,
    enforce_manual_start_on_startup,
    get_orchestrator_overview,
    list_traders,
    read_orchestrator_snapshot,
    update_orchestrator_control,
    update_trader,
    write_orchestrator_snapshot,
)
from tests.postgres_test_db import build_postgres_session_factory
from utils.utcnow import utcnow
from workers import trader_orchestrator_worker


def _quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


@pytest_asyncio.fixture(scope="function")
async def postgres_session_factory():
    engine, session_factory = await build_postgres_session_factory(Base, "trader_orchestrator_no_auto_seed")
    try:
        yield session_factory
    finally:
        await engine.dispose()


@pytest_asyncio.fixture(autouse=True)
async def _reset_database_state(postgres_session_factory):
    table_names = [_quote_ident(table.name) for table in Base.metadata.sorted_tables]
    if not table_names:
        return
    async with postgres_session_factory() as session:
        await session.execute(text(f"TRUNCATE TABLE {', '.join(table_names)} RESTART IDENTITY CASCADE"))
        await session.commit()


@pytest.mark.asyncio
async def test_overview_does_not_seed_default_traders(postgres_session_factory):
    async with postgres_session_factory() as session:
        overview = await get_orchestrator_overview(session)
        count = int((await session.execute(select(func.count(Trader.id)))).scalar() or 0)

    assert overview["traders"] == []
    assert int(overview["metrics"]["traders_total"]) == 0
    assert count == 0


@pytest.mark.asyncio
async def test_overview_stays_empty_after_deleting_last_trader(postgres_session_factory):
    async with postgres_session_factory() as session:
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


@pytest.mark.asyncio
async def test_worker_loop_does_not_seed_default_traders(postgres_session_factory, monkeypatch):
    monkeypatch.setattr(trader_orchestrator_worker, "AsyncSessionLocal", postgres_session_factory)
    monkeypatch.setattr(trader_orchestrator_worker, "expire_stale_signals", AsyncMock())
    monkeypatch.setattr(trader_orchestrator_worker, "ensure_all_strategies_seeded", AsyncMock())
    monkeypatch.setattr(trader_orchestrator_worker, "ensure_trade_signal_group", AsyncMock(return_value=True))
    monkeypatch.setattr(trader_orchestrator_worker, "refresh_strategy_runtime_if_needed", AsyncMock())
    monkeypatch.setattr(trader_orchestrator_worker, "compute_orchestrator_metrics", AsyncMock(return_value={}))
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
    monkeypatch.setattr(
        trader_orchestrator_worker,
        "_ensure_orchestrator_cycle_lock_owner",
        AsyncMock(return_value=True),
    )

    from services.event_bus import event_bus

    monkeypatch.setattr(event_bus, "start", AsyncMock())
    monkeypatch.setattr(event_bus, "subscribe", lambda *a, **kw: None)

    async def _cancel_wait(*_args, **_kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(trader_orchestrator_worker, "_wait_for_runtime_trigger", _cancel_wait)

    with pytest.raises(asyncio.CancelledError):
        await trader_orchestrator_worker.run_worker_loop()

    async with postgres_session_factory() as session:
        count = int((await session.execute(select(func.count(Trader.id)))).scalar() or 0)

    assert count == 0


@pytest.mark.asyncio
async def test_create_trader_copies_settings_from_existing_trader(postgres_session_factory):
    async with postgres_session_factory() as session:
        source = await create_trader(
            session,
            {
                "name": "Source Trader",
                "description": "Copy source",
                "mode": "live",
                "source_configs": [
                    {
                        "source_key": "crypto",
                        "strategy_key": "btc_eth_highfreq",
                        "strategy_params": {"strategy_mode": "maker_quote"},
                    }
                ],
                "interval_seconds": 9,
                "risk_limits": {"max_orders_per_cycle": 3},
                "metadata": {"notes": "copied metadata"},
                "is_enabled": False,
                "is_paused": True,
            },
        )

        copied = await create_trader(
            session,
            {
                "name": "Copied Trader",
                "copy_from_trader_id": source["id"],
            },
        )

    assert copied["id"] != source["id"]
    assert copied["name"] == "Copied Trader"
    assert copied["description"] == source["description"]
    assert copied["mode"] == source["mode"]
    assert copied["source_configs"] == source["source_configs"]
    assert copied["interval_seconds"] == source["interval_seconds"]
    assert copied["risk_limits"] == source["risk_limits"]
    assert copied["metadata"] == source["metadata"]
    assert copied["is_enabled"] == source["is_enabled"]
    assert copied["is_paused"] == source["is_paused"]
    assert copied["requested_run_at"] is None
    assert copied["last_run_at"] is None
    assert copied["next_run_at"] is None


@pytest.mark.asyncio
async def test_create_trader_copy_rejects_unknown_source(postgres_session_factory):
    async with postgres_session_factory() as session:
        with pytest.raises(ValueError, match="Source trader not found"):
            await create_trader(
                session,
                {
                    "name": "Copied Trader",
                    "copy_from_trader_id": "missing-source",
                },
            )


@pytest.mark.asyncio
async def test_create_trader_scopes_by_mode_and_list_filter(postgres_session_factory):
    async with postgres_session_factory() as session:
        paper = await create_trader(
            session,
            {
                "name": "Paper Scoped Trader",
                "mode": "paper",
                "source_configs": [
                    {
                        "source_key": "crypto",
                        "strategy_key": "btc_eth_highfreq",
                        "strategy_params": {},
                    }
                ],
            },
        )
        live = await create_trader(
            session,
            {
                "name": "Live Scoped Trader",
                "mode": "live",
                "source_configs": [
                    {
                        "source_key": "crypto",
                        "strategy_key": "btc_eth_highfreq",
                        "strategy_params": {},
                    }
                ],
            },
        )
        shadow = await create_trader(
            session,
            {
                "name": "Shadow Scoped Trader",
                "mode": "shadow",
                "source_configs": [
                    {
                        "source_key": "crypto",
                        "strategy_key": "btc_eth_highfreq",
                        "strategy_params": {},
                    }
                ],
            },
        )
        paper_rows = await list_traders(session, mode="paper")
        shadow_rows = await list_traders(session, mode="shadow")
        live_rows = await list_traders(session, mode="live")

    assert paper["mode"] == "paper"
    assert shadow["mode"] == "shadow"
    assert live["mode"] == "live"
    assert {row["id"] for row in paper_rows} == {paper["id"]}
    assert {row["id"] for row in shadow_rows} == {shadow["id"]}
    assert {row["id"] for row in live_rows} == {live["id"]}


@pytest.mark.asyncio
async def test_create_trader_rejects_invalid_mode(postgres_session_factory):
    async with postgres_session_factory() as session:
        with pytest.raises(ValueError, match="mode must be 'paper', 'shadow', or 'live'"):
            await create_trader(
                session,
                {
                    "name": "Invalid Mode Trader",
                    "mode": "both",
                    "source_configs": [
                        {
                            "source_key": "crypto",
                            "strategy_key": "btc_eth_highfreq",
                            "strategy_params": {},
                        }
                    ],
                },
            )


@pytest.mark.asyncio
async def test_update_trader_rejects_unknown_strategy_key(postgres_session_factory):
    async with postgres_session_factory() as session:
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


@pytest.mark.asyncio
async def test_startup_enforces_manual_orchestrator_start(postgres_session_factory):
    async with postgres_session_factory() as session:
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
