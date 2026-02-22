import sys
from pathlib import Path

import pytest
from sqlalchemy import select

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api import routes_strategies
from models.database import Base, Strategy, StrategyTombstone
from services.opportunity_strategy_catalog import (
    SYSTEM_OPPORTUNITY_STRATEGY_SEEDS,
    ensure_system_opportunity_strategies_seeded,
)
from tests.postgres_test_db import build_postgres_session_factory


async def _build_session_factory(_tmp_path: Path):
    return await build_postgres_session_factory(Base, "opportunity_strategy_tombstones")


@pytest.mark.asyncio
async def test_seed_skips_tombstoned_system_slug(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            session.add(
                StrategyTombstone(
                    slug="basic",
                    reason="test_tombstone",
                )
            )
            await session.commit()

            changed = await ensure_system_opportunity_strategies_seeded(session)
            expected = len(SYSTEM_OPPORTUNITY_STRATEGY_SEEDS) - 1
            assert changed == expected

            basic_row = (await session.execute(select(Strategy).where(Strategy.slug == "basic"))).scalars().first()
            assert basic_row is None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_delete_system_strategy_creates_tombstone_and_blocks_reseed(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    monkeypatch.setattr(routes_strategies, "AsyncSessionLocal", session_factory)

    try:
        async with session_factory() as session:
            await ensure_system_opportunity_strategies_seeded(session)
            basic_row = (await session.execute(select(Strategy).where(Strategy.slug == "basic"))).scalars().one()
            basic_id = basic_row.id

        result = await routes_strategies.delete_strategy(basic_id)
        assert result["status"] == "success"

        async with session_factory() as session:
            tombstone = await session.get(StrategyTombstone, "basic")
            assert tombstone is not None

            deleted_row = (await session.execute(select(Strategy).where(Strategy.slug == "basic"))).scalars().first()
            assert deleted_row is None

            await ensure_system_opportunity_strategies_seeded(session)
            reseeded = (await session.execute(select(Strategy).where(Strategy.slug == "basic"))).scalars().first()
            assert reseeded is None
    finally:
        await engine.dispose()
