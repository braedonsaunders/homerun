from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import uuid

from sqlalchemy import select
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import StrategyPlugin, StrategyPluginTombstone


_RELATIVE_IMPORT_RE = re.compile(r"(?m)^(\s*)from\s+\.([A-Za-z_][A-Za-z0-9_]*)\s+import\s+")
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_LEGACY_WRAPPER_MARKERS = (
    "System opportunity strategy wrapper loaded from DB",
    "delegates runtime behavior to the shipped strategy class",
    "as _SeedStrategy",
)


@dataclass(frozen=True)
class SystemOpportunityStrategySeed:
    slug: str
    source_key: str
    name: str
    description: str
    import_module: str
    class_name: str
    sort_order: int
    config_schema: dict | None = None  # param_fields for dynamic config UI


def _seed_source_code(seed: SystemOpportunityStrategySeed) -> str:
    module_rel_path = seed.import_module.replace(".", "/") + ".py"
    source_path = _BACKEND_ROOT / module_rel_path
    source = source_path.read_text(encoding="utf-8")
    # DB-loaded modules run outside the services.strategies package, so
    # relative imports must be rewritten to absolute imports.
    source = _RELATIVE_IMPORT_RE.sub(r"\1from services.strategies.\2 import ", source)
    return source


SYSTEM_OPPORTUNITY_STRATEGY_SEEDS: list[SystemOpportunityStrategySeed] = [
    SystemOpportunityStrategySeed(
        slug="basic",
        source_key="scanner",
        name="Basic Arbitrage",
        description="Simple within-market YES+NO arbitrage detection.",
        import_module="services.strategies.basic",
        class_name="BasicArbStrategy",
        sort_order=10,
    ),
    SystemOpportunityStrategySeed(
        slug="negrisk",
        source_key="scanner",
        name="NegRisk",
        description="Negative-risk bundle consistency arbitrage.",
        import_module="services.strategies.negrisk",
        class_name="NegRiskStrategy",
        sort_order=20,
    ),
    SystemOpportunityStrategySeed(
        slug="mutually_exclusive",
        source_key="scanner",
        name="Mutually Exclusive",
        description="Mutually exclusive outcome basket checks.",
        import_module="services.strategies.mutually_exclusive",
        class_name="MutuallyExclusiveStrategy",
        sort_order=30,
    ),
    SystemOpportunityStrategySeed(
        slug="contradiction",
        source_key="scanner",
        name="Contradiction",
        description="Cross-market contradiction opportunities.",
        import_module="services.strategies.contradiction",
        class_name="ContradictionStrategy",
        sort_order=40,
    ),
    SystemOpportunityStrategySeed(
        slug="must_happen",
        source_key="scanner",
        name="Must Happen",
        description="Mandatory-outcome structural opportunities.",
        import_module="services.strategies.must_happen",
        class_name="MustHappenStrategy",
        sort_order=50,
    ),
    SystemOpportunityStrategySeed(
        slug="miracle",
        source_key="scanner",
        name="Miracle",
        description="Garbage collection / stale pricing opportunities.",
        import_module="services.strategies.miracle",
        class_name="MiracleStrategy",
        sort_order=60,
    ),
    SystemOpportunityStrategySeed(
        slug="combinatorial",
        source_key="scanner",
        name="Combinatorial",
        description="Cross-market integer-programming opportunities.",
        import_module="services.strategies.combinatorial",
        class_name="CombinatorialStrategy",
        sort_order=70,
    ),
    SystemOpportunityStrategySeed(
        slug="settlement_lag",
        source_key="scanner",
        name="Settlement Lag",
        description="Delayed market adjustment opportunities.",
        import_module="services.strategies.settlement_lag",
        class_name="SettlementLagStrategy",
        sort_order=80,
    ),
    SystemOpportunityStrategySeed(
        slug="cross_platform",
        source_key="scanner",
        name="Cross Platform",
        description="Polymarket/Kalshi cross-platform arbitrage.",
        import_module="services.strategies.cross_platform",
        class_name="CrossPlatformStrategy",
        sort_order=90,
    ),
    SystemOpportunityStrategySeed(
        slug="bayesian_cascade",
        source_key="scanner",
        name="Bayesian Cascade",
        description="Graph-propagation probability edge strategy.",
        import_module="services.strategies.bayesian_cascade",
        class_name="BayesianCascadeStrategy",
        sort_order=100,
    ),
    SystemOpportunityStrategySeed(
        slug="liquidity_vacuum",
        source_key="scanner",
        name="Liquidity Vacuum",
        description="Order-book imbalance exploitation strategy.",
        import_module="services.strategies.liquidity_vacuum",
        class_name="LiquidityVacuumStrategy",
        sort_order=110,
    ),
    SystemOpportunityStrategySeed(
        slug="entropy_arb",
        source_key="scanner",
        name="Entropy Arb",
        description="Information-theoretic mispricing strategy.",
        import_module="services.strategies.entropy_arb",
        class_name="EntropyArbStrategy",
        sort_order=120,
    ),
    SystemOpportunityStrategySeed(
        slug="event_driven",
        source_key="scanner",
        name="Event Driven",
        description="Catalyst-driven lag/edge opportunities.",
        import_module="services.strategies.event_driven",
        class_name="EventDrivenStrategy",
        sort_order=130,
    ),
    SystemOpportunityStrategySeed(
        slug="temporal_decay",
        source_key="scanner",
        name="Temporal Decay",
        description="Deadline-proximity temporal mispricing strategy.",
        import_module="services.strategies.temporal_decay",
        class_name="TemporalDecayStrategy",
        sort_order=140,
    ),
    SystemOpportunityStrategySeed(
        slug="correlation_arb",
        source_key="scanner",
        name="Correlation Arb",
        description="Correlated spread mean-reversion opportunities.",
        import_module="services.strategies.correlation_arb",
        class_name="CorrelationArbStrategy",
        sort_order=150,
    ),
    SystemOpportunityStrategySeed(
        slug="market_making",
        source_key="scanner",
        name="Market Making",
        description="Bid/ask spread capture strategy.",
        import_module="services.strategies.market_making",
        class_name="MarketMakingStrategy",
        sort_order=160,
    ),
    SystemOpportunityStrategySeed(
        slug="stat_arb",
        source_key="scanner",
        name="Stat Arb",
        description="Statistical ensemble edge strategy.",
        import_module="services.strategies.stat_arb",
        class_name="StatArbStrategy",
        sort_order=170,
    ),
    SystemOpportunityStrategySeed(
        slug="news_edge",
        source_key="news",
        name="News Edge",
        description="News-driven semantic and LLM edge strategy.",
        import_module="services.strategies.news_edge",
        class_name="NewsEdgeStrategy",
        sort_order=180,
    ),
    SystemOpportunityStrategySeed(
        slug="btc_eth_highfreq",
        source_key="crypto",
        name="BTC/ETH High-Frequency",
        description="Dedicated high-frequency crypto strategy family.",
        import_module="services.strategies.btc_eth_highfreq",
        class_name="BtcEthHighFreqStrategy",
        sort_order=190,
    ),
    SystemOpportunityStrategySeed(
        slug="weather_edge",
        source_key="weather",
        name="Weather Edge",
        description="Weather-driven mispricings via multi-source forecast consensus.",
        import_module="services.strategies.weather_edge",
        class_name="WeatherEdgeStrategy",
        sort_order=200,
        config_schema={
            "param_fields": [
                {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0},
                {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
                {"key": "min_model_agreement", "label": "Min Model Agreement", "type": "number", "min": 0, "max": 1},
                {"key": "min_source_count", "label": "Min Forecast Sources", "type": "integer", "min": 1},
                {"key": "max_source_spread_c", "label": "Max Source Spread (C)", "type": "number", "min": 0},
                {"key": "max_entry_price", "label": "Max Entry Price", "type": "number", "min": 0, "max": 1},
                {"key": "risk_base_score", "label": "Base Risk Score", "type": "number", "min": 0, "max": 1},
            ]
        },
    ),
    SystemOpportunityStrategySeed(
        slug="traders_confluence",
        source_key="traders",
        name="Traders Confluence",
        description="Smart money convergence via tracked wallet confluence analysis.",
        import_module="services.strategies.traders_confluence",
        class_name="TradersConfluenceStrategy",
        sort_order=210,
        config_schema={
            "param_fields": [
                {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0},
                {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
                {"key": "min_confluence_strength", "label": "Min Confluence Strength", "type": "number", "min": 0, "max": 1},
                {"key": "min_tier", "label": "Min Tier", "type": "enum", "options": ["low", "medium", "high", "extreme"]},
                {"key": "min_wallet_count", "label": "Min Wallet Count", "type": "integer", "min": 1},
                {"key": "max_entry_price", "label": "Max Entry Price", "type": "number", "min": 0, "max": 1},
                {"key": "risk_base_score", "label": "Base Risk Score", "type": "number", "min": 0, "max": 1},
            ]
        },
    ),
]


def build_system_opportunity_strategy_rows(*, now: datetime | None = None) -> list[dict]:
    ts = now or datetime.utcnow()
    rows: list[dict] = []
    for seed in SYSTEM_OPPORTUNITY_STRATEGY_SEEDS:
        rows.append(
            {
                "id": uuid.uuid4().hex,
                "slug": seed.slug,
                "source_key": seed.source_key,
                "name": seed.name,
                "description": seed.description,
                "source_code": _seed_source_code(seed),
                "class_name": seed.class_name,
                "is_system": True,
                "enabled": True,
                "status": "unloaded",
                "error_message": None,
                "config": {"_schema": seed.config_schema} if seed.config_schema else {},
                "version": 1,
                "sort_order": seed.sort_order,
                "created_at": ts,
                "updated_at": ts,
            }
        )
    return rows


async def ensure_system_opportunity_strategies_seeded(session: AsyncSession) -> int:
    rows = build_system_opportunity_strategy_rows()
    seed_by_slug = {row["slug"]: row for row in rows}
    try:
        tombstoned_slugs = set(
            (
                await session.execute(
                    select(StrategyPluginTombstone.slug).where(
                        StrategyPluginTombstone.slug.in_(list(seed_by_slug.keys()))
                    )
                )
            )
            .scalars()
            .all()
        )
    except (ProgrammingError, OperationalError):
        # Backward compatibility for environments mid-upgrade where the tombstone table
        # might not exist yet. Seeder behavior falls back to legacy non-tombstoned mode.
        tombstoned_slugs = set()
    existing = {
        plugin.slug: plugin
        for plugin in (
            (
                await session.execute(
                    select(StrategyPlugin).where(StrategyPlugin.slug.in_(list(seed_by_slug.keys())))
                )
            )
            .scalars()
            .all()
        )
    }

    inserted = 0
    rewritten = 0
    for slug, row in seed_by_slug.items():
        if slug in tombstoned_slugs:
            continue

        current = existing.get(slug)
        if current is None:
            session.add(StrategyPlugin(**row))
            inserted += 1
            continue

        source_code = current.source_code or ""
        class_name = (current.class_name or "").strip()
        has_wrapper_marker = any(marker in source_code for marker in _LEGACY_WRAPPER_MARKERS)
        if not has_wrapper_marker and not class_name.endswith("System"):
            continue

        current.source_key = row["source_key"]
        current.name = row["name"]
        current.description = row["description"]
        current.source_code = row["source_code"]
        current.class_name = row["class_name"]
        current.is_system = True
        current.sort_order = row["sort_order"]
        current.status = "unloaded"
        current.error_message = None
        current.version = int(current.version or 0) + 1
        current.updated_at = datetime.utcnow()
        rewritten += 1

    if inserted == 0 and rewritten == 0:
        return 0
    await session.commit()
    return inserted + rewritten
