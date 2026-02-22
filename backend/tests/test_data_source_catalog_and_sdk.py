import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import select

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import Base, DataSource
from services.data_source_catalog import BASE_SYSTEM_DATA_SOURCE_SEEDS, ensure_system_data_sources_seeded
from services.data_source_loader import validate_data_source_source
from services.data_source_sdk import DataSourceSDK
from services.strategy_sdk import StrategySDK
from tests.postgres_test_db import build_postgres_session_factory


async def _build_session_factory(_tmp_path: Path):
    return await build_postgres_session_factory(Base, "data_source_catalog_sdk")


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
                    source_key="events",
                    source_kind="bridge",
                    name="Legacy World Seed",
                    source_code="class SomethingElse:\n    pass\n",
                    is_system=True,
                    enabled=True,
                ),
                DataSource(
                    id="legacy_story_import",
                    slug="stories_custom_custom_https_example_com_feed_a1b2c3",
                    source_key="stories",
                    source_kind="rss",
                    name="Stories: Custom - Legacy Feed",
                    description="Imported custom RSS source (Legacy Feed) from legacy app settings",
                    source_code="",
                    is_system=False,
                    enabled=True,
                    config={"url": "https://example.com/feed.xml", "feed_source": "custom_rss"},
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
    assert "stories_custom_custom_https_example_com_feed_a1b2c3" not in slugs
    assert {"events", "stories"}.issubset(source_keys)
    assert "news" not in source_keys
    assert "events_all" not in slugs
    assert any(slug.startswith("stories_google_") for slug in slugs)
    assert any(slug.startswith("stories_gdelt_") for slug in slugs)
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


@pytest.mark.asyncio
async def test_data_source_runner_serializes_datetime_payloads(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    import services.data_source_sdk as data_source_sdk_module

    monkeypatch.setattr(data_source_sdk_module, "AsyncSessionLocal", session_factory)

    source_code = (
        "from datetime import datetime, timezone\n"
        "from services.data_source_sdk import BaseDataSource\n\n"
        "class DatePayloadSource(BaseDataSource):\n"
        "    name = 'Date Payload Source'\n"
        "    description = 'Returns datetime-rich payloads'\n\n"
        "    async def fetch_async(self):\n"
        "        return [\n"
        "            {\n"
        "                'external_id': 'dt-1',\n"
        "                'title': 'Datetime payload row',\n"
        "                'summary': 'payload includes nested datetimes',\n"
        "                'category': 'events',\n"
        "                'source': 'unit_test',\n"
        "                'observed_at': datetime(2026, 2, 18, 12, 0, tzinfo=timezone.utc),\n"
        "                'payload': {'nested_at': datetime(2026, 2, 18, 10, 30, tzinfo=timezone.utc)},\n"
        "                'list_with_datetime': [datetime(2026, 2, 18, 11, 0, tzinfo=timezone.utc)],\n"
        "                'tags': ['unit'],\n"
        "            }\n"
        "        ]\n\n"
        "    def transform(self, item):\n"
        "        item['transformed_at'] = datetime(2026, 2, 18, 12, 30, tzinfo=timezone.utc)\n"
        "        return item\n"
    )

    created = await DataSourceSDK.create_source(
        slug="datetime_payload_source",
        source_key="events",
        source_kind="python",
        source_code=source_code,
        enabled=True,
    )
    assert created.get("slug") == "datetime_payload_source"

    run_result = await DataSourceSDK.run_source("datetime_payload_source", max_records=5)
    assert run_result.get("status") == "success"
    assert int(run_result.get("upserted_count") or 0) == 1

    records = await DataSourceSDK.get_records(source_slug="datetime_payload_source", limit=5)
    assert len(records) == 1

    payload = records[0].get("payload") or {}
    transformed = records[0].get("transformed") or {}

    assert payload.get("observed_at") == datetime(2026, 2, 18, 12, 0, tzinfo=timezone.utc).isoformat()
    assert payload.get("payload", {}).get("nested_at") == datetime(2026, 2, 18, 10, 30, tzinfo=timezone.utc).isoformat()
    assert payload.get("list_with_datetime", [None])[0] == datetime(2026, 2, 18, 11, 0, tzinfo=timezone.utc).isoformat()
    assert transformed.get("transformed_at") == datetime(2026, 2, 18, 12, 30, tzinfo=timezone.utc).isoformat()

    await engine.dispose()


@pytest.mark.asyncio
async def test_data_source_sdk_rejects_unknown_source_kind(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    import services.data_source_sdk as data_source_sdk_module

    monkeypatch.setattr(data_source_sdk_module, "AsyncSessionLocal", session_factory)

    with pytest.raises(ValueError, match="Unsupported source_kind"):
        await DataSourceSDK.create_source(
            slug="invalid_kind_source",
            source_key="stories",
            source_kind="bridge",
            source_code="",
            name="Invalid Kind Source",
            enabled=False,
        )

    await engine.dispose()


@pytest.mark.asyncio
async def test_data_source_sdk_rejects_invalid_source_key(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    import services.data_source_sdk as data_source_sdk_module

    monkeypatch.setattr(data_source_sdk_module, "AsyncSessionLocal", session_factory)

    with pytest.raises(ValueError, match="Invalid source_key"):
        await DataSourceSDK.create_source(
            slug="invalid_key_source",
            source_key="bad-key",
            source_kind="rest_api",
            source_code="",
            name="Invalid Key Source",
            config={"url": "https://example.com/feed"},
            enabled=False,
        )

    await engine.dispose()


@pytest.mark.asyncio
async def test_data_source_sdk_applies_default_retention_by_source_family(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    import services.data_source_sdk as data_source_sdk_module

    monkeypatch.setattr(data_source_sdk_module, "AsyncSessionLocal", session_factory)

    events_source_code = (
        "from services.data_source_sdk import BaseDataSource\n\n"
        "class EventsRetentionDefaultSource(BaseDataSource):\n"
        "    name = 'Events Retention Default Source'\n"
        "    async def fetch_async(self):\n"
        "        return []\n"
    )
    events_source = await DataSourceSDK.create_source(
        slug="events_retention_default_source",
        source_key="events",
        source_kind="python",
        source_code=events_source_code,
        enabled=False,
    )
    assert events_source.get("retention") == {"max_records": 50000, "max_age_days": 365}

    stories_rss_source = await DataSourceSDK.create_source(
        slug="stories_rss_retention_default_source",
        source_key="stories",
        source_kind="rss",
        source_code="",
        name="Stories RSS Retention Default Source",
        config={"url": "https://example.com/feed.xml"},
        enabled=False,
    )
    assert stories_rss_source.get("retention") == {"max_records": 7500, "max_age_days": 30}

    stories_rest_source = await DataSourceSDK.create_source(
        slug="stories_rest_retention_default_source",
        source_key="stories",
        source_kind="rest_api",
        source_code="",
        name="Stories REST Retention Default Source",
        config={"url": "https://example.com/api/articles", "json_path": "$.articles[*]"},
        enabled=False,
    )
    assert stories_rest_source.get("retention") == {"max_records": 15000, "max_age_days": 45}

    await engine.dispose()


@pytest.mark.asyncio
async def test_data_source_sdk_applies_max_records_retention(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    import services.data_source_sdk as data_source_sdk_module

    monkeypatch.setattr(data_source_sdk_module, "AsyncSessionLocal", session_factory)

    source_code = (
        "from datetime import datetime, timezone\\n"
        "from services.data_source_sdk import BaseDataSource\\n\\n"
        "class MaxRecordsRetentionSource(BaseDataSource):\\n"
        "    name = 'Max Records Retention Source'\\n"
        "    description = 'Emits six deterministic rows'\\n\\n"
        "    async def fetch_async(self):\\n"
        "        rows = []\\n"
        "        for idx in range(1, 7):\\n"
        "            rows.append({\\n"
        "                'external_id': f'r{idx}',\\n"
        "                'title': f'Row {idx}',\\n"
        "                'summary': 'retention test',\\n"
        "                'category': 'events',\\n"
        "                'source': 'unit_test',\\n"
        "                'observed_at': datetime(2026, 2, 18, 12, idx, tzinfo=timezone.utc),\\n"
        "                'tags': ['retention'],\\n"
        "            })\\n"
        "        return rows\\n"
    )

    created = await DataSourceSDK.create_source(
        slug="retention_max_records_source",
        source_key="events",
        source_kind="python",
        source_code=source_code,
        retention={"max_records": 3},
        enabled=True,
    )
    created_retention = created.get("retention") or {}
    assert int(created_retention.get("max_records") or 0) == 3

    run_result = await DataSourceSDK.run_source("retention_max_records_source", max_records=20)
    assert run_result.get("status") == "success"
    assert int(run_result.get("retention_pruned_count") or 0) >= 3

    records = await DataSourceSDK.get_records(source_slug="retention_max_records_source", limit=10)
    assert [record.get("external_id") for record in records] == ["r6", "r5", "r4"]

    await engine.dispose()


@pytest.mark.asyncio
async def test_data_source_sdk_applies_max_age_days_retention(tmp_path, monkeypatch):
    engine, session_factory = await _build_session_factory(tmp_path)
    import services.data_source_sdk as data_source_sdk_module

    monkeypatch.setattr(data_source_sdk_module, "AsyncSessionLocal", session_factory)

    source_code = (
        "from datetime import datetime, timedelta, timezone\\n"
        "from services.data_source_sdk import BaseDataSource\\n\\n"
        "class MaxAgeRetentionSource(BaseDataSource):\\n"
        "    name = 'Max Age Retention Source'\\n"
        "    description = 'Emits one old row and one fresh row'\\n\\n"
        "    async def fetch_async(self):\\n"
        "        now = datetime.now(timezone.utc)\\n"
        "        return [\\n"
        "            {\\n"
        "                'external_id': 'old',\\n"
        "                'title': 'Old row',\\n"
        "                'summary': 'old',\\n"
        "                'category': 'events',\\n"
        "                'source': 'unit_test',\\n"
        "                'observed_at': now - timedelta(days=30),\\n"
        "                'tags': ['retention'],\\n"
        "            },\\n"
        "            {\\n"
        "                'external_id': 'recent',\\n"
        "                'title': 'Recent row',\\n"
        "                'summary': 'recent',\\n"
        "                'category': 'events',\\n"
        "                'source': 'unit_test',\\n"
        "                'observed_at': now - timedelta(days=1),\\n"
        "                'tags': ['retention'],\\n"
        "            },\\n"
        "        ]\\n"
    )

    created = await DataSourceSDK.create_source(
        slug="retention_max_age_source",
        source_key="events",
        source_kind="python",
        source_code=source_code,
        retention={"max_age_days": 7},
        enabled=True,
    )
    created_retention = created.get("retention") or {}
    assert int(created_retention.get("max_age_days") or 0) == 7

    run_result = await DataSourceSDK.run_source("retention_max_age_source", max_records=20)
    assert run_result.get("status") == "success"
    assert int(run_result.get("retention_pruned_count") or 0) >= 1

    records = await DataSourceSDK.get_records(source_slug="retention_max_age_source", limit=10)
    assert [record.get("external_id") for record in records] == ["recent"]

    await engine.dispose()


def test_events_seed_sources_are_self_contained():
    events_seeds = [seed for seed in BASE_SYSTEM_DATA_SOURCE_SEEDS if seed.source_key == "events"]
    assert len(events_seeds) == 13
    assert {seed.slug for seed in events_seeds} == {
        "events_acled",
        "events_gdelt_tensions",
        "events_military",
        "events_infrastructure",
        "events_gdelt_news",
        "events_usgs",
        "events_ucdp_conflicts",
        "events_trade_dependencies",
        "events_chokepoint_reference",
        "events_country_instability",
        "events_airplanes_live",
        "events_ais_ships",
        "events_nasa_firms",
    }

    for seed in events_seeds:
        assert "services.events" not in seed.source_code
        assert "'provider':" not in seed.source_code
        assert '"provider":' not in seed.source_code
        validation = validate_data_source_source(seed.source_code)
        assert bool(validation.get("valid")) is True


def test_seed_sources_define_retention_policies():
    assert BASE_SYSTEM_DATA_SOURCE_SEEDS
    for seed in BASE_SYSTEM_DATA_SOURCE_SEEDS:
        assert isinstance(seed.retention, dict)
        assert int(seed.retention.get("max_records") or 0) > 0
        assert int(seed.retention.get("max_age_days") or 0) > 0

    events_seed = next(seed for seed in BASE_SYSTEM_DATA_SOURCE_SEEDS if seed.slug == "events_acled")
    assert events_seed.retention == {"max_records": 50000, "max_age_days": 365}

    stories_google_seed = next(
        seed for seed in BASE_SYSTEM_DATA_SOURCE_SEEDS if seed.slug.startswith("stories_google_")
    )
    assert stories_google_seed.retention == {"max_records": 7500, "max_age_days": 30}

    stories_gdelt_seed = next(seed for seed in BASE_SYSTEM_DATA_SOURCE_SEEDS if seed.slug.startswith("stories_gdelt_"))
    assert stories_gdelt_seed.retention == {"max_records": 15000, "max_age_days": 45}
