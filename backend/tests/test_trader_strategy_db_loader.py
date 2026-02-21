import sys
from pathlib import Path

import pytest
from sqlalchemy import select

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import Base, Strategy
from services.opportunity_strategy_catalog import (
    SYSTEM_OPPORTUNITY_STRATEGY_SEEDS,
    build_system_opportunity_strategy_rows,
    ensure_system_opportunity_strategies_seeded,
)
from services.strategy_loader import (
    StrategyLoader as StrategyDBLoader,
    validate_strategy_source,
)
from tests.postgres_test_db import build_postgres_session_factory

# Every seed slug that the unified catalog produces.
REQUIRED_STRATEGY_SLUGS = {seed.slug for seed in SYSTEM_OPPORTUNITY_STRATEGY_SEEDS}


async def _build_session_factory(_tmp_path: Path):
    return await build_postgres_session_factory(Base, "strategy_loader")


def test_system_strategy_catalog_contains_required_keys():
    rows = build_system_opportunity_strategy_rows()
    keys = {str(row.get("slug") or "").strip().lower() for row in rows}
    assert keys == REQUIRED_STRATEGY_SLUGS


def test_system_strategy_catalog_uses_executable_source_files():
    rows = build_system_opportunity_strategy_rows()
    for row in rows:
        source_code = str(row.get("source_code") or "")
        class_name = str(row.get("class_name") or "")
        assert "System strategy seed wrapper loaded from DB" not in source_code
        assert class_name in source_code
        validation = validate_strategy_source(source_code, class_name)
        assert validation["valid"] is True, f"{row['slug']}: {validation['errors']}"
        assert validation.get("class_name") == class_name


def test_validate_strategy_source_rejects_blocked_import():
    source_code = "\n".join(
        [
            "import os",
            "from services.strategies.base import BaseStrategy, StrategyDecision",
            "",
            "class BlockedImportStrategy(BaseStrategy):",
            "    key = 'blocked_import_strategy'",
            "    def evaluate(self, signal, context):",
            "        return StrategyDecision(decision='skipped', reason='blocked', score=0.0, checks=[], payload={})",
        ]
    )
    validation = validate_strategy_source(source_code, "BlockedImportStrategy")
    assert validation["valid"] is False
    assert any("Blocked import" in err for err in validation["errors"])


def test_validate_strategy_source_accepts_detect_async_only_strategy():
    source_code = "\n".join(
        [
            "from services.strategies.base import BaseStrategy",
            "",
            "class AsyncOnlyStrategy(BaseStrategy):",
            "    name = 'Async Only'",
            "    description = 'Detects in async mode only'",
            "    async def detect_async(self, events, markets, prices):",
            "        return []",
        ]
    )
    validation = validate_strategy_source(source_code, "AsyncOnlyStrategy")
    assert validation["valid"] is True
    assert validation["capabilities"]["has_detect_async"] is True
    assert validation["capabilities"]["has_detect"] is False
    assert validation["capabilities"]["has_evaluate"] is False


def test_validate_strategy_source_accepts_on_event_only_strategy():
    source_code = "\n".join(
        [
            "from services.strategies.base import BaseStrategy",
            "from services.data_events import EventType",
            "",
            "class EventOnlyStrategy(BaseStrategy):",
            "    name = 'Event Only'",
            "    description = 'Reacts to event bus updates'",
            "    subscriptions = [EventType.CRYPTO_UPDATE]",
            "    async def on_event(self, event):",
            "        return []",
        ]
    )
    validation = validate_strategy_source(source_code, "EventOnlyStrategy")
    assert validation["valid"] is True
    assert validation["capabilities"]["has_on_event"] is True
    assert validation["capabilities"]["has_detect"] is False
    assert validation["capabilities"]["has_detect_async"] is False
    assert validation["capabilities"]["has_evaluate"] is False


def test_validate_strategy_source_rejects_should_exit_only_strategy():
    source_code = "\n".join(
        [
            "from services.strategies.base import BaseStrategy, ExitDecision",
            "",
            "class ExitOnlyStrategy(BaseStrategy):",
            "    name = 'Exit Only'",
            "    description = 'Exit-only strategy'",
            "    def should_exit(self, position, market_state):",
            "        return ExitDecision(action='hold', reason='not used')",
        ]
    )
    validation = validate_strategy_source(source_code, "ExitOnlyStrategy")
    assert validation["valid"] is False
    assert any("must implement at least one of" in err for err in validation["errors"])


def test_loader_sets_slug_identity_and_merges_config_for_dynamic_strategy():
    source_code = "\n".join(
        [
            "from services.strategies.base import BaseStrategy",
            "",
            "class ConfigAwareStrategy(BaseStrategy):",
            "    name = 'Config Aware'",
            "    description = 'Tests runtime contract for dynamic strategies'",
            "    default_config = {'threshold': 0.15, 'window': 3}",
            "    def detect(self, events, markets, prices):",
            "        return [",
            "            {",
            "                'strategy_type': self.strategy_type,",
            "                'key': self.key,",
            "                'threshold': self.config.get('threshold'),",
            "                'window': self.config.get('window'),",
            "                'market_count': len(markets),",
            "            }",
            "        ]",
        ]
    )
    loader = StrategyDBLoader()
    loaded = loader.load("config_aware_strategy", source_code, {"threshold": 0.4})
    strategy = loaded.instance

    assert strategy.strategy_type == "config_aware_strategy"
    assert strategy.key == "config_aware_strategy"
    assert strategy.config["threshold"] == 0.4
    assert strategy.config["window"] == 3
    assert strategy.default_config["threshold"] == 0.15

    opportunities = strategy.detect(events=[], markets=[object(), object()], prices={})
    assert len(opportunities) == 1
    assert opportunities[0]["strategy_type"] == "config_aware_strategy"
    assert opportunities[0]["key"] == "config_aware_strategy"
    assert opportunities[0]["threshold"] == 0.4
    assert opportunities[0]["window"] == 3
    assert opportunities[0]["market_count"] == 2

    loader.unload("config_aware_strategy")
    assert loader.get_strategy("config_aware_strategy") is None


@pytest.mark.asyncio
async def test_loader_isolates_error_rows_and_loads_valid_rows(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            valid_source = "\n".join(
                [
                    "from services.strategies.base import BaseStrategy, StrategyDecision",
                    "",
                    "class UnitGoodStrategy(BaseStrategy):",
                    "    key = 'unit_good_strategy'",
                    "    def evaluate(self, signal, context):",
                    "        return StrategyDecision(decision='skipped', reason='ok', score=0.0, checks=[], payload={})",
                ]
            )
            invalid_source = "\n".join(
                [
                    "import os",
                    "from services.strategies.base import BaseStrategy, StrategyDecision",
                    "",
                    "class UnitBadStrategy(BaseStrategy):",
                    "    key = 'unit_bad_strategy'",
                    "    def evaluate(self, signal, context):",
                    "        return StrategyDecision(decision='skipped', reason='bad', score=0.0, checks=[], payload={})",
                ]
            )

            session.add(
                Strategy(
                    id="unit-good-row",
                    slug="unit_good_strategy",
                    source_key="crypto",
                    name="Unit Good",
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
                Strategy(
                    id="unit-bad-row",
                    slug="unit_bad_strategy",
                    source_key="crypto",
                    name="Unit Bad",
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

            good_row = await session.get(Strategy, "unit-good-row")
            bad_row = await session.get(Strategy, "unit-bad-row")
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
                    '"""System opportunity strategy wrapper loaded from DB."""',
                    "from services.strategies.base import BaseStrategy",
                    "from services.strategies.basic import BasicArbStrategy as _SeedStrategy",
                    "",
                    "class BasicArbStrategy(_SeedStrategy):",
                    "    pass",
                ]
            )
            session.add(
                Strategy(
                    id="legacy-basic",
                    slug="basic",
                    source_key="scanner",
                    name="Basic Arbitrage",
                    description="Legacy wrapper",
                    class_name="BasicArbStrategy",
                    source_code=legacy_source,
                    config={},
                    config_schema={},
                    aliases=[],
                    is_system=True,
                    enabled=True,
                    status="loaded",
                    version=1,
                )
            )
            await session.commit()

            changed = await ensure_system_opportunity_strategies_seeded(session)
            assert changed >= 1

            row = (await session.execute(select(Strategy).where(Strategy.slug == "basic"))).scalars().one()
            assert "System opportunity strategy wrapper loaded from DB" not in (row.source_code or "")
            assert "from services.strategies" in (row.source_code or "")
            assert int(row.version or 0) == 2
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_ensure_system_seed_rewrites_arbitrage_opportunity_imports(tmp_path):
    engine, session_factory = await _build_session_factory(tmp_path)
    try:
        async with session_factory() as session:
            legacy_source = "\n".join(
                [
                    "from models import Market, Event, ArbitrageOpportunity",
                    "from services.strategies.base import BaseStrategy",
                    "",
                    "class ContradictionStrategy(BaseStrategy):",
                    "    name = 'Contradiction'",
                    "    description = 'Legacy import test'",
                    "    def detect(self, events, markets, prices):",
                    "        return []",
                ]
            )
            session.add(
                Strategy(
                    id="legacy-contradiction",
                    slug="contradiction",
                    source_key="scanner",
                    name="Contradiction",
                    description="Legacy import row",
                    class_name="ContradictionStrategy",
                    source_code=legacy_source,
                    config={},
                    config_schema={},
                    aliases=[],
                    is_system=True,
                    enabled=True,
                    status="loaded",
                    version=1,
                )
            )
            await session.commit()

            changed = await ensure_system_opportunity_strategies_seeded(session)
            assert changed >= 1

            row = (await session.execute(select(Strategy).where(Strategy.slug == "contradiction"))).scalars().one()
            assert "ArbitrageOpportunity" not in (row.source_code or "")
            assert "from models import Market, Event, Opportunity" in (row.source_code or "")
            assert int(row.version or 0) == 2
    finally:
        await engine.dispose()
