import sys
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import Base, DataSource
from services.data_source_catalog import ensure_system_data_sources_seeded
from services.strategy_sdk import StrategySDK


async def _build_session_factory(tmp_path: Path):
    db_path = tmp_path / "data_source_catalog_sdk.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine, session_factory


@pytest.mark.asyncio
async def test_catalog_seeds_events_and_stories_and_removes_legacy_rows(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    legacy_seed_code = (
        "# System data source seed\n"
        "from services.data_source_sdk import BaseDataSource\n\n"
        "class LegacySource(BaseDataSource):\n"
        "    name = 'Legacy Source'\n"
        "    async def fetch_async(self):\n"
        "        return []\n"
    )

    async with session_factory() as session:
        session.add_all(
            [
                DataSource(
                    id="legacy_seed",
                    slug="news_custom_legacy",
                    source_key="news",
                    source_kind="rss",
                    name="Legacy Seed",
                    source_code=legacy_seed_code,
                    is_system=True,
                    enabled=True,
                ),
                DataSource(
                    id="legacy_world_seed",
                    slug="world_legacy_seed",
                    source_key="world_intelligence",
                    source_kind="bridge",
                    name="Legacy World Seed",
                    source_code="class SomethingElse:\n    pass\n",
                    is_system=True,
                    enabled=True,
                ),
            ]
        )
        await session.commit()

    async with session_factory() as session:
        seeded_count = await ensure_system_data_sources_seeded(session)
        rows = (await session.execute(select(DataSource).order_by(DataSource.slug.asc()))).scalars().all()

    slugs = {str(row.slug) for row in rows}
    source_keys = {str(row.source_key) for row in rows if bool(row.is_system)}

    assert seeded_count > 0
    assert "news_custom_legacy" not in slugs
    assert "world_legacy_seed" not in slugs
    assert {"events", "stories"}.issubset(source_keys)
    assert "news" not in source_keys
    assert "world_intelligence" not in source_keys
    assert "events_all" in slugs
    assert "stories_all" in slugs
    assert "stories_google_news" in slugs
    assert "events_acled" in slugs

    await engine.dispose()


@pytest.mark.asyncio
async def test_strategy_sdk_exposes_full_source_workflow(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    import services.data_source_sdk as data_source_sdk_module

    monkeypatch.setattr(data_source_sdk_module, "AsyncSessionLocal", session_factory)

    source_code = (
        "from services.data_source_sdk import BaseDataSource\n\n"
        "class StrategySdkSource(BaseDataSource):\n"
        "    name = 'Strategy SDK Source'\n"
        "    description = 'Source managed via StrategySDK'\n"
        "    async def fetch_async(self):\n"
        "        return []\n"
    )

    created = await StrategySDK.create_data_source(
        slug="strategy_sdk_source",
        source_key="stories",
        source_kind="python",
        source_code=source_code,
        enabled=True,
    )
    assert created.get("slug") == "strategy_sdk_source"

    listed = await StrategySDK.list_data_sources(enabled_only=False)
    assert any(row.get("slug") == "strategy_sdk_source" for row in listed)

    fetched = await StrategySDK.get_data_source("strategy_sdk_source")
    assert fetched.get("name") == "Strategy SDK Source"

    validated = StrategySDK.validate_data_source(source_code)
    assert bool(validated.get("valid")) is True

    updated = await StrategySDK.update_data_source(
        "strategy_sdk_source",
        name="Strategy SDK Source Updated",
    )
    assert updated.get("name") == "Strategy SDK Source Updated"

    reloaded = await StrategySDK.reload_data_source("strategy_sdk_source")
    assert reloaded.get("status") in {"loaded", "unloaded"}

    run_result = await StrategySDK.run_data_source("strategy_sdk_source", max_records=10)
    assert run_result.get("status") == "success"

    deleted = await StrategySDK.delete_data_source("strategy_sdk_source")
    assert deleted.get("status") == "deleted"

    await engine.dispose()
