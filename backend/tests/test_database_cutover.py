import sys
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import models.database as database


def _repo_head_revision() -> str:
    cfg = Config(str(BACKEND_ROOT / "alembic.ini"))
    script = ScriptDirectory.from_config(cfg)
    return script.get_current_head()


@pytest.mark.asyncio
async def test_init_database_creates_schema_and_revision(tmp_path, monkeypatch):
    db_path = tmp_path / "schema.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    monkeypatch.setattr(database, "async_engine", engine)
    monkeypatch.setattr(database, "AsyncSessionLocal", session_factory)

    await database.init_database()

    async with engine.begin() as conn:
        orchestrator_rows = await conn.execute(
            text(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='trader_orchestrator_control'"
            )
        )
        assert orchestrator_rows.first() is not None

        revision_row = await conn.execute(text("SELECT version_num FROM alembic_version"))
        assert revision_row.scalar_one() == _repo_head_revision()

        app_columns = await conn.execute(text("PRAGMA table_info(app_settings)"))
        app_column_names = {row[1] for row in app_columns.fetchall()}
        assert "world_intel_settings_json" in app_column_names

    await engine.dispose()


@pytest.mark.asyncio
async def test_init_database_is_idempotent(tmp_path, monkeypatch):
    db_path = tmp_path / "idempotent.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    monkeypatch.setattr(database, "async_engine", engine)
    monkeypatch.setattr(database, "AsyncSessionLocal", session_factory)

    await database.init_database()
    await database.init_database()

    async with engine.begin() as conn:
        revision_rows = await conn.execute(text("SELECT version_num FROM alembic_version"))
        rows = revision_rows.fetchall()
        assert len(rows) == 1
        assert rows[0][0] == _repo_head_revision()

    await engine.dispose()
