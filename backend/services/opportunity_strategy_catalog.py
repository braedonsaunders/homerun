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
        slug="flash_crash_reversion",
        source_key="scanner",
        name="Flash Crash Reversion",
        description="Short-window crash-reversion filter with execution gates.",
        import_module="services.strategies.flash_crash_reversion",
        class_name="FlashCrashReversionStrategy",
        sort_order=175,
        config_schema={
            "param_fields": [
                {"key": "lookback_seconds", "label": "Lookback (seconds)", "type": "number", "min": 30},
                {"key": "drop_threshold", "label": "Drop Threshold", "type": "number", "min": 0.01, "max": 0.5},
                {"key": "min_rebound_fraction", "label": "Min Rebound Fraction", "type": "number", "min": 0.1, "max": 0.95},
                {"key": "min_target_move", "label": "Min Target Move", "type": "number", "min": 0.005, "max": 0.15},
                {"key": "max_entry_price", "label": "Max Entry Price", "type": "number", "min": 0.1, "max": 1},
                {"key": "max_spread", "label": "Max Spread", "type": "number", "min": 0.005, "max": 0.25},
                {"key": "min_liquidity", "label": "Min Liquidity", "type": "number", "min": 0},
            ]
        },
    ),
    SystemOpportunityStrategySeed(
        slug="tail_end_carry",
        source_key="scanner",
        name="Tail-End Carry",
        description="Near-expiry high-probability carry opportunities.",
        import_module="services.strategies.tail_end_carry",
        class_name="TailEndCarryStrategy",
        sort_order=176,
        config_schema={
            "param_fields": [
                {"key": "min_probability", "label": "Min Probability", "type": "number", "min": 0.5, "max": 1},
                {"key": "max_probability", "label": "Max Probability", "type": "number", "min": 0.5, "max": 1},
                {"key": "min_days_to_resolution", "label": "Min Days To Resolution", "type": "number", "min": 0},
                {"key": "max_days_to_resolution", "label": "Max Days To Resolution", "type": "number", "min": 0},
                {"key": "min_liquidity", "label": "Min Liquidity", "type": "number", "min": 0},
                {"key": "max_spread", "label": "Max Spread", "type": "number", "min": 0.005, "max": 0.2},
                {"key": "min_repricing_buffer", "label": "Min Repricing Buffer", "type": "number", "min": 0.005, "max": 0.1},
                {"key": "repricing_weight", "label": "Repricing Weight", "type": "number", "min": 0.1, "max": 0.9},
            ]
        },
    ),
    SystemOpportunityStrategySeed(
        slug="spread_dislocation",
        source_key="scanner",
        name="Spread Dislocation",
        description="Wide bid/ask dislocation filter for spread capture entries.",
        import_module="services.strategies.spread_dislocation",
        class_name="SpreadDislocationStrategy",
        sort_order=177,
        config_schema={
            "param_fields": [
                {"key": "min_spread", "label": "Min Spread", "type": "number", "min": 0.005, "max": 0.5},
                {"key": "max_spread", "label": "Max Spread", "type": "number", "min": 0.01, "max": 0.6},
                {"key": "min_mid_price", "label": "Min Mid Price", "type": "number", "min": 0.01, "max": 0.99},
                {"key": "max_mid_price", "label": "Max Mid Price", "type": "number", "min": 0.01, "max": 0.99},
                {"key": "capture_fraction", "label": "Capture Fraction", "type": "number", "min": 0.1, "max": 0.95},
                {"key": "min_target_move", "label": "Min Target Move", "type": "number", "min": 0.002, "max": 0.15},
                {"key": "min_liquidity", "label": "Min Liquidity", "type": "number", "min": 0},
                {"key": "min_days_to_resolution", "label": "Min Days To Resolution", "type": "number", "min": 0},
                {"key": "max_days_to_resolution", "label": "Max Days To Resolution", "type": "number", "min": 0},
            ]
        },
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
                {"key": "probability_scale_c", "label": "Sigmoid Scale (C)", "type": "number", "min": 0.5, "max": 5.0},
                {"key": "risk_base_score", "label": "Base Risk Score", "type": "number", "min": 0, "max": 1},
            ]
        },
    ),
    SystemOpportunityStrategySeed(
        slug="weather_ensemble_edge",
        source_key="weather",
        name="Weather Ensemble Edge",
        description="Ensemble Monte Carlo: count fraction of 31 GFS ensemble members in each temperature bucket.",
        import_module="services.strategies.weather_ensemble_edge",
        class_name="WeatherEnsembleEdgeStrategy",
        sort_order=201,
        config_schema={
            "param_fields": [
                {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0},
                {"key": "min_ensemble_members", "label": "Min Ensemble Members", "type": "integer", "min": 1},
                {"key": "min_ensemble_agreement", "label": "Min Ensemble Agreement", "type": "number", "min": 0, "max": 1},
                {"key": "max_entry_price", "label": "Max Entry Price", "type": "number", "min": 0, "max": 1},
                {"key": "deterministic_fallback", "label": "Deterministic Fallback", "type": "enum", "options": ["true", "false"]},
                {"key": "probability_scale_c", "label": "Fallback Sigmoid Scale (C)", "type": "number", "min": 0.5, "max": 5.0},
                {"key": "risk_base_score", "label": "Base Risk Score", "type": "number", "min": 0, "max": 1},
            ]
        },
    ),
    SystemOpportunityStrategySeed(
        slug="weather_distribution",
        source_key="weather",
        name="Weather Distribution",
        description="Full distribution comparison: build probability across all buckets, buy the most underpriced.",
        import_module="services.strategies.weather_distribution",
        class_name="WeatherDistributionStrategy",
        sort_order=202,
        config_schema={
            "param_fields": [
                {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0},
                {"key": "sigma_c", "label": "Sigma (C)", "type": "number", "min": 0.5, "max": 5.0},
                {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
                {"key": "max_entry_price", "label": "Max Entry Price", "type": "number", "min": 0, "max": 1},
                {"key": "max_buckets_per_event", "label": "Max Buckets per Event", "type": "integer", "min": 1, "max": 10},
                {"key": "risk_base_score", "label": "Base Risk Score", "type": "number", "min": 0, "max": 1},
            ]
        },
    ),
    SystemOpportunityStrategySeed(
        slug="weather_conservative_no",
        source_key="weather",
        name="Weather Conservative NO",
        description="Conservative NO-betting: bet NO on buckets far from forecast consensus for high win rate.",
        import_module="services.strategies.weather_conservative_no",
        class_name="WeatherConservativeNoStrategy",
        sort_order=203,
        config_schema={
            "param_fields": [
                {"key": "min_safe_distance_c", "label": "Min Distance from Forecast (C)", "type": "number", "min": 0},
                {"key": "max_no_price", "label": "Max NO Price", "type": "number", "min": 0, "max": 1},
                {"key": "min_model_agreement", "label": "Min Model Agreement", "type": "number", "min": 0, "max": 1},
                {"key": "max_source_spread_c", "label": "Max Source Spread (C)", "type": "number", "min": 0},
                {"key": "min_source_count", "label": "Min Forecast Sources", "type": "integer", "min": 1},
                {"key": "max_positions_per_event", "label": "Max Positions per Event", "type": "integer", "min": 1, "max": 10},
                {"key": "risk_base_score", "label": "Base Risk Score", "type": "number", "min": 0, "max": 1},
            ]
        },
    ),
    SystemOpportunityStrategySeed(
        slug="weather_bucket_edge",
        source_key="weather",
        name="Weather Bucket Edge",
        description="Per-bucket edge: compare model probability directly to market price for each temperature bucket.",
        import_module="services.strategies.weather_bucket_edge",
        class_name="WeatherBucketEdgeStrategy",
        sort_order=204,
        config_schema={
            "param_fields": [
                {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0},
                {"key": "probability_scale_c", "label": "Sigmoid Scale (C)", "type": "number", "min": 0.5, "max": 5.0},
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
                {"key": "firehose_require_tradable_market", "label": "Require Tradable Market", "type": "boolean"},
                {"key": "firehose_exclude_crypto_markets", "label": "Exclude Crypto Markets", "type": "boolean"},
                {"key": "firehose_require_qualified_source", "label": "Require Qualified Source", "type": "boolean"},
                {"key": "firehose_max_age_minutes", "label": "Firehose Max Age (min)", "type": "integer", "min": 1, "max": 1440},
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
