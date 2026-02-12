import sys
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import models.database as database


@pytest.mark.asyncio
async def test_init_database_drops_legacy_auto_trader_tables(tmp_path, monkeypatch):
    db_path = tmp_path / "cutover.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    monkeypatch.setattr(database, "async_engine", engine)
    monkeypatch.setattr(database, "AsyncSessionLocal", session_factory)

    async with engine.begin() as conn:
        await conn.execute(text("CREATE TABLE auto_trader_control (id TEXT PRIMARY KEY)"))
        await conn.execute(text("CREATE TABLE auto_trader_snapshot (id TEXT PRIMARY KEY)"))

    await database.init_database()

    async with engine.begin() as conn:
        legacy_rows = await conn.execute(
            text(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name LIKE 'auto_trader_%'"
            )
        )
        assert legacy_rows.fetchall() == []

        orchestrator_rows = await conn.execute(
            text(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='trader_orchestrator_control'"
            )
        )
        assert orchestrator_rows.first() is not None

    await engine.dispose()
