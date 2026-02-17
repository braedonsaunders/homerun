import sys
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api import routes_plugins
from models.database import Base, StrategyPlugin, StrategyPluginTombstone
from services.opportunity_strategy_catalog import (
    SYSTEM_OPPORTUNITY_STRATEGY_SEEDS,
    ensure_system_opportunity_strategies_seeded,
)


async def _build_session_factory(tmp_path: Path):
    db_path = tmp_path / "opportunity_strategy_tombstones.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine, session_factory


@pytest.mark.asyncio
async def test_seed_skips_tombstoned_system_slug(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            session.add(
                StrategyPluginTombstone(
                    slug="basic",
                    reason="test_tombstone",
                )
            )
            await session.commit()

            changed = await ensure_system_opportunity_strategies_seeded(session)
            expected = len(SYSTEM_OPPORTUNITY_STRATEGY_SEEDS) - 1
            assert changed == expected

            basic_row = (
                (
                    await session.execute(
                        select(StrategyPlugin).where(StrategyPlugin.slug == "basic")
                    )
                )
                .scalars()
                .first()
            )
            assert basic_row is None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_delete_system_strategy_creates_tombstone_and_blocks_reseed(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    monkeypatch.setattr(routes_plugins, "AsyncSessionLocal", session_factory)

    try:
        async with session_factory() as session:
            await ensure_system_opportunity_strategies_seeded(session)
            basic_row = (
                (
                    await session.execute(
                        select(StrategyPlugin).where(StrategyPlugin.slug == "basic")
                    )
                )
                .scalars()
                .one()
            )
            basic_id = basic_row.id

        result = await routes_plugins.delete_plugin(basic_id)
        assert result["status"] == "success"

        async with session_factory() as session:
            tombstone = await session.get(StrategyPluginTombstone, "basic")
            assert tombstone is not None

            deleted_row = (
                (
                    await session.execute(
                        select(StrategyPlugin).where(StrategyPlugin.slug == "basic")
                    )
                )
                .scalars()
                .first()
            )
            assert deleted_row is None

            await ensure_system_opportunity_strategies_seeded(session)
            reseeded = (
                (
                    await session.execute(
                        select(StrategyPlugin).where(StrategyPlugin.slug == "basic")
                    )
                )
                .scalars()
                .first()
            )
            assert reseeded is None
    finally:
        await engine.dispose()
