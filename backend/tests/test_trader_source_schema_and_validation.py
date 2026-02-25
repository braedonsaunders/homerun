import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import Base
from services.trader_orchestrator.config_schema import build_trader_config_schema
from services.trader_orchestrator_state import _normalize_trader_payload
from tests.postgres_test_db import build_postgres_session_factory


async def _build_session_factory(_tmp_path: Path):
    return await build_postgres_session_factory(Base, "trader_source_schema")


@pytest.mark.asyncio
async def test_source_schema_excludes_events_and_omni(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    async with session_factory() as session:
        schema = await build_trader_config_schema(session)
    source_keys = {str(source.get("key")) for source in schema.get("sources", [])}
    assert "events" not in source_keys

    strategy_keys = {
        str(option.get("key"))
        for source in schema.get("sources", [])
        for option in (source.get("strategy_options") or [])
    }
    assert "omni_aggressive" not in strategy_keys
    await engine.dispose()


@pytest.mark.asyncio
async def test_crypto_schema_exposes_dynamic_strategy_options(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    async with session_factory() as session:
        schema = await build_trader_config_schema(session)
    crypto_source = next(source for source in schema["sources"] if source["key"] == "crypto")
    crypto_strategies = {str(option["key"]) for option in (crypto_source.get("strategy_options") or [])}
    assert crypto_strategies
    assert str(crypto_source.get("default_strategy_key") or "") in crypto_strategies
    default_config = crypto_source.get("default_config") or {}
    assert str(default_config.get("strategy_key") or "") in crypto_strategies
    assert "__passthrough__" not in crypto_strategies
    await engine.dispose()


@pytest.mark.asyncio
async def test_crypto_schema_populates_defaults_and_dedupes_param_fields(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    async with session_factory() as session:
        schema = await build_trader_config_schema(session)

    crypto_source = next(source for source in schema["sources"] if source["key"] == "crypto")
    btc_option = next(option for option in (crypto_source.get("strategy_options") or []) if option.get("key") == "btc_eth_highfreq")

    default_params = dict(btc_option.get("default_params") or {})
    assert len(default_params) >= 120
    assert default_params.get("max_open_order_seconds") == 14.0
    assert default_params.get("reverse_on_adverse_velocity_enabled") is False

    param_keys = [str(field.get("key") or "").strip() for field in (btc_option.get("param_fields") or []) if isinstance(field, dict)]
    assert param_keys
    assert len(param_keys) == len(set(param_keys))
    assert "max_opportunities" not in param_keys
    assert "retention_window" not in param_keys
    assert "allowed_timeframes" not in param_keys
    assert "enforce_hard_timeframe_allowlist" not in param_keys
    assert "hard_allowed_timeframes" not in param_keys
    await engine.dispose()


@pytest.mark.asyncio
async def test_scanner_and_weather_have_separate_strategy_sets(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    async with session_factory() as session:
        schema = await build_trader_config_schema(session)
    scanner_source = next(source for source in schema["sources"] if source["key"] == "scanner")
    weather_source = next(source for source in schema["sources"] if source["key"] == "weather")

    scanner_strategies = {str(option["key"]) for option in scanner_source.get("strategy_options", [])}
    weather_strategies = {str(option["key"]) for option in weather_source.get("strategy_options", [])}

    assert scanner_strategies
    assert weather_strategies
    assert str(scanner_source.get("default_strategy_key") or "") in scanner_strategies
    assert str(weather_source.get("default_strategy_key") or "") in weather_strategies
    assert bool(scanner_strategies - weather_strategies)
    assert bool(weather_strategies - scanner_strategies)
    assert "__passthrough__" not in scanner_strategies
    assert "__passthrough__" not in weather_strategies
    await engine.dispose()


@pytest.mark.asyncio
async def test_normalize_trader_payload_rejects_invalid_source_strategy_pair(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    async with session_factory() as session:
        schema = await build_trader_config_schema(session)
        source_to_strategies = {
            str(source.get("key")): {str(option.get("key")) for option in (source.get("strategy_options") or [])}
            for source in schema.get("sources", [])
        }
        invalid_source_key = ""
        invalid_strategy_key = ""
        for source_key, strategy_keys in source_to_strategies.items():
            if not strategy_keys:
                continue
            for target_key, target_strategy_keys in source_to_strategies.items():
                if target_key == source_key:
                    continue
                candidate = next((key for key in strategy_keys if key not in target_strategy_keys), "")
                if candidate:
                    invalid_source_key = target_key
                    invalid_strategy_key = candidate
                    break
            if invalid_strategy_key:
                break

        if not invalid_source_key or not invalid_strategy_key:
            pytest.skip("No cross-source invalid strategy pair available in current dynamic schema")

        with pytest.raises(ValueError, match="Invalid strategy_key"):
            await _normalize_trader_payload(
                session,
                {
                    "name": "Invalid Pair",
                    "source_configs": [
                        {
                            "source_key": invalid_source_key,
                            "strategy_key": invalid_strategy_key,
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
        schema = await build_trader_config_schema(session)
        scanner_source = next(source for source in schema.get("sources", []) if source.get("key") == "scanner")
        weather_source = next(source for source in schema.get("sources", []) if source.get("key") == "weather")
        scanner_strategies = {str(option.get("key")) for option in (scanner_source.get("strategy_options") or [])}
        weather_strategies = {str(option.get("key")) for option in (weather_source.get("strategy_options") or [])}
        scanner_only_key = next((key for key in scanner_strategies if key not in weather_strategies), "")
        if not scanner_only_key:
            pytest.skip("No scanner-only strategy key available in current dynamic schema")

        with pytest.raises(ValueError, match="Invalid strategy_key"):
            await _normalize_trader_payload(
                session,
                {
                    "name": "Legacy Weather Pair",
                    "source_configs": [
                        {
                            "source_key": "weather",
                            "strategy_key": scanner_only_key,
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
        schema = await build_trader_config_schema(session)
        traders_source = next(source for source in schema.get("sources", []) if source.get("key") == "traders")
        strategy_key = str(
            next(
                (option.get("key") for option in (traders_source.get("strategy_options") or []) if option.get("key")),
                "",
            )
        )
        if not strategy_key:
            pytest.skip("No traders strategy options available in current dynamic schema")

        with pytest.raises(ValueError, match="individual_wallets"):
            await _normalize_trader_payload(
                session,
                {
                    "name": "Invalid Scope",
                    "source_configs": [
                        {
                            "source_key": "traders",
                            "strategy_key": strategy_key,
                            "strategy_params": {
                                "traders_scope": {
                                    "modes": ["individual"],
                                    "individual_wallets": [],
                                    "group_ids": [],
                                }
                            },
                        }
                    ],
                },
            )
    await engine.dispose()
