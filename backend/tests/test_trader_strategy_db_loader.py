import sys
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import Base, TraderStrategyDefinition
from services.trader_orchestrator.strategy_catalog import (
    build_system_strategy_rows,
    ensure_system_trader_strategies_seeded,
)
from services.trader_orchestrator.strategy_db_loader import (
    StrategyDBLoader,
    validate_trader_strategy_source,
)


REQUIRED_STRATEGY_KEYS = {
    "crypto_5m",
    "crypto_15m",
    "crypto_1h",
    "crypto_4h",
    "opportunity_general",
    "opportunity_structural",
    "news_reaction",
    "traders_flow",
    "weather_consensus",
    "weather_alerts",
}


async def _build_session_factory(tmp_path: Path):
    db_path = tmp_path / "trader_strategy_loader.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine, session_factory


def test_system_strategy_catalog_contains_required_keys():
    rows = build_system_strategy_rows()
    keys = {str(row.get("strategy_key") or "").strip().lower() for row in rows}
    assert keys == REQUIRED_STRATEGY_KEYS


def test_system_strategy_catalog_uses_executable_source_files():
    rows = build_system_strategy_rows()
    for row in rows:
        source_code = str(row.get("source_code") or "")
        class_name = str(row.get("class_name") or "")
        assert "System strategy seed wrapper loaded from DB" not in source_code
        assert class_name in source_code
        validation = validate_trader_strategy_source(source_code, class_name)
        assert validation["valid"] is True
        assert validation.get("class_name") == class_name


def test_validate_strategy_source_rejects_blocked_import():
    source_code = "\n".join(
        [
            "import os",
            "from services.trader_orchestrator.strategies.base import BaseTraderStrategy, StrategyDecision",
            "",
            "class BlockedImportStrategy(BaseTraderStrategy):",
            "    key = 'blocked_import_strategy'",
            "    def evaluate(self, signal, context):",
            "        return StrategyDecision(decision='skipped', reason='blocked', score=0.0, checks=[], payload={})",
        ]
    )
    validation = validate_trader_strategy_source(source_code, "BlockedImportStrategy")
    assert validation["valid"] is False
    assert any("Blocked import" in err for err in validation["errors"])


@pytest.mark.asyncio
async def test_loader_isolates_error_rows_and_loads_valid_rows(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            valid_source = "\n".join(
                [
                    "from services.trader_orchestrator.strategies.base import BaseTraderStrategy, StrategyDecision",
                    "",
                    "class UnitGoodStrategy(BaseTraderStrategy):",
                    "    key = 'unit_good_strategy'",
                    "    def evaluate(self, signal, context):",
                    "        return StrategyDecision(decision='skipped', reason='ok', score=0.0, checks=[], payload={})",
                ]
            )
            invalid_source = "\n".join(
                [
                    "import os",
                    "from services.trader_orchestrator.strategies.base import BaseTraderStrategy, StrategyDecision",
                    "",
                    "class UnitBadStrategy(BaseTraderStrategy):",
                    "    key = 'unit_bad_strategy'",
                    "    def evaluate(self, signal, context):",
                    "        return StrategyDecision(decision='skipped', reason='bad', score=0.0, checks=[], payload={})",
                ]
            )

            session.add(
                TraderStrategyDefinition(
                    id="unit-good-row",
                    strategy_key="unit_good_strategy",
                    source_key="crypto",
                    label="Unit Good",
                    description="Unit good row",
                    class_name="UnitGoodStrategy",
                    source_code=valid_source,
                    enabled=True,
                    is_system=False,
                    status="unloaded",
                    version=1,
                )
            )
            session.add(
                TraderStrategyDefinition(
                    id="unit-bad-row",
                    strategy_key="unit_bad_strategy",
                    source_key="crypto",
                    label="Unit Bad",
                    description="Unit bad row",
                    class_name="UnitBadStrategy",
                    source_code=invalid_source,
                    enabled=True,
                    is_system=False,
                    status="unloaded",
                    version=1,
                )
            )
            await session.commit()

            loader = StrategyDBLoader()
            result = await loader.refresh_from_db(session=session)

            assert "unit_good_strategy" in result["loaded"]
            assert "unit_bad_strategy" in result["errors"]
            assert loader.get_strategy("unit_good_strategy") is not None
            assert loader.get_strategy("unit_bad_strategy") is None

            good_row = await session.get(TraderStrategyDefinition, "unit-good-row")
            bad_row = await session.get(TraderStrategyDefinition, "unit-bad-row")
            assert good_row is not None and good_row.status == "loaded"
            assert bad_row is not None and bad_row.status == "error"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_ensure_system_seed_rewrites_legacy_wrapper_rows(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            legacy_source = "\n".join(
                [
                    '"""System strategy seed wrapper loaded from DB."""',
                    "from services.trader_orchestrator.strategies.base import BaseTraderStrategy",
                    "from services.trader_orchestrator.strategies.crypto_15m import Crypto15mStrategy as _SeedStrategy",
                    "",
                    "class Crypto15mStrategy(_SeedStrategy):",
                    "    pass",
                ]
            )
            session.add(
                TraderStrategyDefinition(
                    id="legacy-crypto-15m",
                    strategy_key="crypto_15m",
                    source_key="crypto",
                    label="Crypto 15m",
                    description="Legacy wrapper",
                    class_name="Crypto15mStrategy",
                    source_code=legacy_source,
                    default_params_json={},
                    param_schema_json={},
                    aliases_json=[],
                    is_system=True,
                    enabled=True,
                    status="loaded",
                    version=1,
                )
            )
            await session.commit()

            changed = await ensure_system_trader_strategies_seeded(session)
            assert changed == len(REQUIRED_STRATEGY_KEYS)

            row = (
                (
                    await session.execute(
                        select(TraderStrategyDefinition).where(
                            TraderStrategyDefinition.strategy_key == "crypto_15m"
                        )
                    )
                )
                .scalars()
                .one()
            )
            assert "System strategy seed wrapper loaded from DB" not in (row.source_code or "")
            assert "from services.trader_orchestrator.strategies.base import" in (row.source_code or "")
            assert int(row.version or 0) == 2
    finally:
        await engine.dispose()
