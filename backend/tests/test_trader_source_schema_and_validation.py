import sys
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import Base
from services.trader_orchestrator.config_schema import build_trader_config_schema
from services.trader_orchestrator_state import _normalize_trader_payload


async def _build_session_factory(tmp_path: Path):
    db_path = tmp_path / "trader_source_schema.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine, session_factory


@pytest.mark.asyncio
async def test_source_schema_excludes_world_intelligence_and_omni(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    async with session_factory() as session:
        schema = await build_trader_config_schema(session)
    source_keys = {str(source.get("key")) for source in schema.get("sources", [])}
    assert "world_intelligence" not in source_keys

    strategy_keys = {
        str(option.get("key"))
        for source in schema.get("sources", [])
        for option in (source.get("strategy_options") or [])
    }
    assert "omni_aggressive" not in strategy_keys
    await engine.dispose()


@pytest.mark.asyncio
async def test_crypto_schema_exposes_exact_timeframe_strategies(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    async with session_factory() as session:
        schema = await build_trader_config_schema(session)
    crypto_source = next(source for source in schema["sources"] if source["key"] == "crypto")
    crypto_strategies = [str(option["key"]) for option in crypto_source.get("strategy_options", [])]
    assert crypto_strategies == ["crypto_5m", "crypto_15m", "crypto_1h", "crypto_4h"]
    await engine.dispose()


@pytest.mark.asyncio
async def test_scanner_and_weather_have_separate_strategy_sets(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    async with session_factory() as session:
        schema = await build_trader_config_schema(session)
    scanner_source = next(source for source in schema["sources"] if source["key"] == "scanner")
    weather_source = next(source for source in schema["sources"] if source["key"] == "weather")

    scanner_strategies = [str(option["key"]) for option in scanner_source.get("strategy_options", [])]
    weather_strategies = [str(option["key"]) for option in weather_source.get("strategy_options", [])]

    assert scanner_strategies == ["opportunity_general", "opportunity_structural"]
    assert weather_strategies == ["weather_consensus", "weather_alerts"]
    await engine.dispose()


@pytest.mark.asyncio
async def test_normalize_trader_payload_rejects_invalid_source_strategy_pair(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    async with session_factory() as session:
        with pytest.raises(ValueError, match="Invalid strategy_key"):
            await _normalize_trader_payload(
                session,
                {
                    "name": "Invalid Pair",
                    "source_configs": [
                        {
                            "source_key": "news",
                            "strategy_key": "crypto_15m",
                            "strategy_params": {},
                        }
                    ],
                },
            )
    await engine.dispose()


@pytest.mark.asyncio
async def test_normalize_trader_payload_rejects_scanner_strategy_on_weather_source(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    async with session_factory() as session:
        with pytest.raises(ValueError, match="Invalid strategy_key"):
            await _normalize_trader_payload(
                session,
                {
                    "name": "Legacy Weather Pair",
                    "source_configs": [
                        {
                            "source_key": "weather",
                            "strategy_key": "opportunity_general",
                            "strategy_params": {},
                        }
                    ],
                },
            )
    await engine.dispose()


@pytest.mark.asyncio
async def test_normalize_trader_payload_rejects_invalid_traders_scope(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    async with session_factory() as session:
        with pytest.raises(ValueError, match="individual_wallets"):
            await _normalize_trader_payload(
                session,
                {
                    "name": "Invalid Scope",
                    "source_configs": [
                        {
                            "source_key": "traders",
                            "strategy_key": "traders_flow",
                            "strategy_params": {},
                            "traders_scope": {
                                "modes": ["individual"],
                                "individual_wallets": [],
                                "group_ids": [],
                            },
                        }
                    ],
                },
            )
    await engine.dispose()
