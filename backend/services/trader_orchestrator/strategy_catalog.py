from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import re
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

_RELATIVE_IMPORT_RE = re.compile(r"(?m)^(\s*)from\s+\.([A-Za-z_][A-Za-z0-9_]*)\s+import\s+")
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_LEGACY_WRAPPER_MARKERS = (
    "System strategy seed wrapper loaded from DB",
    "delegates runtime behavior to the built-in strategy implementation",
    "as _SeedStrategy",
)


@dataclass(frozen=True)
class SystemTraderStrategySeed:
    strategy_key: str
    source_key: str
    label: str
    description: str
    class_name: str
    import_module: str
    import_class: str
    default_params: dict[str, Any]
    param_schema: dict[str, Any]
    aliases: list[str]


@lru_cache(maxsize=None)
def _seed_source_code(import_module: str) -> str:
    module_rel_path = import_module.replace(".", "/") + ".py"
    source_path = _BACKEND_ROOT / module_rel_path
    source = source_path.read_text(encoding="utf-8")
    # DB-loaded modules are not package-scoped, so rewrite local relative imports.
    source = _RELATIVE_IMPORT_RE.sub(r"\1from services.trader_orchestrator.strategies.\2 import ", source)
    return source


def _is_legacy_wrapper_source(source_code: str) -> bool:
    return any(marker in source_code for marker in _LEGACY_WRAPPER_MARKERS)


SYSTEM_TRADER_STRATEGY_SEEDS: list[SystemTraderStrategySeed] = [
    SystemTraderStrategySeed(
        strategy_key="crypto_5m",
        source_key="crypto",
        label="Crypto 5m",
        description="Dedicated crypto execution for 5m markets.",
        class_name="Crypto5mStrategy",
        import_module="services.trader_orchestrator.strategies.crypto_15m",
        import_class="Crypto5mStrategy",
        default_params={
            "strategy_mode": "auto",
            "target_assets": ["BTC", "ETH", "SOL", "XRP"],
            "min_edge_percent": 3.0,
            "min_confidence": 0.45,
            "base_size_usd": 25.0,
            "max_size_usd": 100.0,
            "direction_guardrail_enabled": True,
            "direction_guardrail_prob_floor": 0.55,
            "direction_guardrail_price_floor": 0.8,
            "direction_guardrail_regimes": ["mid", "closing"],
        },
        param_schema={
            "param_fields": [
                {"key": "strategy_mode", "label": "Strategy Mode", "type": "enum", "options": ["auto", "directional", "pure_arb", "rebalance"]},
                {"key": "target_assets", "label": "Target Assets", "type": "array[string]"},
                {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0},
                {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
                {"key": "direction_guardrail_enabled", "label": "Direction Guardrail Enabled", "type": "boolean"},
                {"key": "direction_guardrail_prob_floor", "label": "Direction Guardrail Prob Floor", "type": "number", "min": 0, "max": 1},
                {"key": "direction_guardrail_price_floor", "label": "Direction Guardrail Price Floor", "type": "number", "min": 0, "max": 1},
                {"key": "direction_guardrail_regimes", "label": "Direction Guardrail Regimes", "type": "array[string]", "options": ["opening", "mid", "closing"]},
                {"key": "base_size_usd", "label": "Base Size (USD)", "type": "number", "min": 1},
                {"key": "max_size_usd", "label": "Max Size (USD)", "type": "number", "min": 1},
            ]
        },
        aliases=[],
    ),
    SystemTraderStrategySeed(
        strategy_key="crypto_15m",
        source_key="crypto",
        label="Crypto 15m",
        description="Dedicated crypto execution for 15m markets.",
        class_name="Crypto15mStrategy",
        import_module="services.trader_orchestrator.strategies.crypto_15m",
        import_class="Crypto15mStrategy",
        default_params={
            "strategy_mode": "auto",
            "target_assets": ["BTC", "ETH", "SOL", "XRP"],
            "min_edge_percent": 3.0,
            "min_confidence": 0.45,
            "base_size_usd": 25.0,
            "max_size_usd": 100.0,
            "direction_guardrail_enabled": True,
            "direction_guardrail_prob_floor": 0.55,
            "direction_guardrail_price_floor": 0.8,
            "direction_guardrail_regimes": ["mid", "closing"],
        },
        param_schema={
            "param_fields": [
                {"key": "strategy_mode", "label": "Strategy Mode", "type": "enum", "options": ["auto", "directional", "pure_arb", "rebalance"]},
                {"key": "target_assets", "label": "Target Assets", "type": "array[string]"},
                {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0},
                {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
                {"key": "direction_guardrail_enabled", "label": "Direction Guardrail Enabled", "type": "boolean"},
                {"key": "direction_guardrail_prob_floor", "label": "Direction Guardrail Prob Floor", "type": "number", "min": 0, "max": 1},
                {"key": "direction_guardrail_price_floor", "label": "Direction Guardrail Price Floor", "type": "number", "min": 0, "max": 1},
                {"key": "direction_guardrail_regimes", "label": "Direction Guardrail Regimes", "type": "array[string]", "options": ["opening", "mid", "closing"]},
                {"key": "base_size_usd", "label": "Base Size (USD)", "type": "number", "min": 1},
                {"key": "max_size_usd", "label": "Max Size (USD)", "type": "number", "min": 1},
            ]
        },
        aliases=["strategy.default", "default"],
    ),
    SystemTraderStrategySeed(
        strategy_key="crypto_1h",
        source_key="crypto",
        label="Crypto 1h",
        description="Dedicated crypto execution for 1h markets.",
        class_name="Crypto1hStrategy",
        import_module="services.trader_orchestrator.strategies.crypto_15m",
        import_class="Crypto1hStrategy",
        default_params={
            "strategy_mode": "auto",
            "target_assets": ["BTC", "ETH", "SOL", "XRP"],
            "min_edge_percent": 3.0,
            "min_confidence": 0.45,
            "base_size_usd": 25.0,
            "max_size_usd": 100.0,
            "direction_guardrail_enabled": True,
            "direction_guardrail_prob_floor": 0.55,
            "direction_guardrail_price_floor": 0.8,
            "direction_guardrail_regimes": ["mid", "closing"],
        },
        param_schema={
            "param_fields": [
                {"key": "strategy_mode", "label": "Strategy Mode", "type": "enum", "options": ["auto", "directional", "pure_arb", "rebalance"]},
                {"key": "target_assets", "label": "Target Assets", "type": "array[string]"},
                {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0},
                {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
                {"key": "direction_guardrail_enabled", "label": "Direction Guardrail Enabled", "type": "boolean"},
                {"key": "direction_guardrail_prob_floor", "label": "Direction Guardrail Prob Floor", "type": "number", "min": 0, "max": 1},
                {"key": "direction_guardrail_price_floor", "label": "Direction Guardrail Price Floor", "type": "number", "min": 0, "max": 1},
                {"key": "direction_guardrail_regimes", "label": "Direction Guardrail Regimes", "type": "array[string]", "options": ["opening", "mid", "closing"]},
                {"key": "base_size_usd", "label": "Base Size (USD)", "type": "number", "min": 1},
                {"key": "max_size_usd", "label": "Max Size (USD)", "type": "number", "min": 1},
            ]
        },
        aliases=[],
    ),
    SystemTraderStrategySeed(
        strategy_key="crypto_4h",
        source_key="crypto",
        label="Crypto 4h",
        description="Dedicated crypto execution for 4h markets.",
        class_name="Crypto4hStrategy",
        import_module="services.trader_orchestrator.strategies.crypto_15m",
        import_class="Crypto4hStrategy",
        default_params={
            "strategy_mode": "auto",
            "target_assets": ["BTC", "ETH", "SOL", "XRP"],
            "min_edge_percent": 3.0,
            "min_confidence": 0.45,
            "base_size_usd": 25.0,
            "max_size_usd": 100.0,
            "direction_guardrail_enabled": True,
            "direction_guardrail_prob_floor": 0.55,
            "direction_guardrail_price_floor": 0.8,
            "direction_guardrail_regimes": ["mid", "closing"],
        },
        param_schema={
            "param_fields": [
                {"key": "strategy_mode", "label": "Strategy Mode", "type": "enum", "options": ["auto", "directional", "pure_arb", "rebalance"]},
                {"key": "target_assets", "label": "Target Assets", "type": "array[string]"},
                {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0},
                {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
                {"key": "direction_guardrail_enabled", "label": "Direction Guardrail Enabled", "type": "boolean"},
                {"key": "direction_guardrail_prob_floor", "label": "Direction Guardrail Prob Floor", "type": "number", "min": 0, "max": 1},
                {"key": "direction_guardrail_price_floor", "label": "Direction Guardrail Price Floor", "type": "number", "min": 0, "max": 1},
                {"key": "direction_guardrail_regimes", "label": "Direction Guardrail Regimes", "type": "array[string]", "options": ["opening", "mid", "closing"]},
                {"key": "base_size_usd", "label": "Base Size (USD)", "type": "number", "min": 1},
                {"key": "max_size_usd", "label": "Max Size (USD)", "type": "number", "min": 1},
            ]
        },
        aliases=[],
    ),
    SystemTraderStrategySeed(
        strategy_key="opportunity_general",
        source_key="scanner",
        label="Opportunity General",
        description="General-purpose scanner strategy across all opportunity shapes.",
        class_name="OpportunityGeneralStrategy",
        import_module="services.trader_orchestrator.strategies.opportunity_weather",
        import_class="OpportunityGeneralStrategy",
        default_params={
            "min_edge_percent": 4.0,
            "min_confidence": 0.45,
            "max_risk_score": 0.78,
            "min_liquidity": 25.0,
            "min_markets": 1,
            "base_size_usd": 18.0,
            "max_size_usd": 150.0,
        },
        param_schema={
            "param_fields": [
                {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0},
                {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
                {"key": "max_risk_score", "label": "Max Risk Score", "type": "number", "min": 0, "max": 1},
                {"key": "min_liquidity", "label": "Min Liquidity", "type": "number", "min": 0},
                {"key": "min_markets", "label": "Min Markets", "type": "integer", "min": 1},
                {"key": "base_size_usd", "label": "Base Size (USD)", "type": "number", "min": 1},
                {"key": "max_size_usd", "label": "Max Size (USD)", "type": "number", "min": 1},
            ]
        },
        aliases=["opportunity_weather"],
    ),
    SystemTraderStrategySeed(
        strategy_key="opportunity_structural",
        source_key="scanner",
        label="Opportunity Structural",
        description="Risk-first structural/multi-leg opportunity execution.",
        class_name="OpportunityStructuralStrategy",
        import_module="services.trader_orchestrator.strategies.opportunity_weather",
        import_class="OpportunityStructuralStrategy",
        default_params={
            "min_edge_percent": 3.0,
            "min_confidence": 0.42,
            "max_risk_score": 0.68,
            "min_markets": 2,
            "base_size_usd": 20.0,
            "max_size_usd": 180.0,
        },
        param_schema={
            "param_fields": [
                {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0},
                {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
                {"key": "max_risk_score", "label": "Max Risk Score", "type": "number", "min": 0, "max": 1},
                {"key": "min_markets", "label": "Min Markets", "type": "integer", "min": 1},
                {"key": "base_size_usd", "label": "Base Size (USD)", "type": "number", "min": 1},
                {"key": "max_size_usd", "label": "Max Size (USD)", "type": "number", "min": 1},
            ]
        },
        aliases=[],
    ),
    SystemTraderStrategySeed(
        strategy_key="news_reaction",
        source_key="news",
        label="News Reaction",
        description="Executes high-conviction news signals.",
        class_name="NewsReactionStrategy",
        import_module="services.trader_orchestrator.strategies.news_reaction",
        import_class="NewsReactionStrategy",
        default_params={
            "min_edge_percent": 8.0,
            "min_confidence": 0.55,
            "base_size_usd": 20.0,
        },
        param_schema={
            "param_fields": [
                {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0},
                {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
                {"key": "base_size_usd", "label": "Base Size (USD)", "type": "number", "min": 1},
            ]
        },
        aliases=[],
    ),
    SystemTraderStrategySeed(
        strategy_key="traders_flow",
        source_key="traders",
        label="Traders Flow",
        description="Confluence-driven trader-flow strategy.",
        class_name="TradersFlowStrategy",
        import_module="services.trader_orchestrator.strategies.traders_flow",
        import_class="TradersFlowStrategy",
        default_params={
            "min_edge_percent": 3.0,
            "min_confidence": 0.48,
            "min_confluence_strength": 0.55,
            "base_size_usd": 18.0,
            "max_size_usd": 120.0,
        },
        param_schema={
            "param_fields": [
                {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0},
                {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
                {"key": "min_confluence_strength", "label": "Min Confluence Strength", "type": "number", "min": 0, "max": 1},
                {"key": "base_size_usd", "label": "Base Size (USD)", "type": "number", "min": 1},
                {"key": "max_size_usd", "label": "Max Size (USD)", "type": "number", "min": 1},
            ]
        },
        aliases=[],
    ),
    SystemTraderStrategySeed(
        strategy_key="weather_consensus",
        source_key="weather",
        label="Weather Consensus",
        description="Consensus-driven weather strategy with source agreement gating.",
        class_name="WeatherConsensusStrategy",
        import_module="services.trader_orchestrator.strategies.weather_models",
        import_class="WeatherConsensusStrategy",
        default_params={
            "min_edge_percent": 6.0,
            "min_confidence": 0.58,
            "min_model_agreement": 0.62,
            "min_source_count": 2,
            "max_source_spread_c": 4.0,
            "max_entry_price": 0.8,
            "base_size_usd": 14.0,
            "max_size_usd": 90.0,
        },
        param_schema={
            "param_fields": [
                {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0},
                {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
                {"key": "min_model_agreement", "label": "Min Model Agreement", "type": "number", "min": 0, "max": 1},
                {"key": "min_source_count", "label": "Min Forecast Sources", "type": "integer", "min": 1},
                {"key": "max_source_spread_c", "label": "Max Source Spread (C)", "type": "number", "min": 0},
                {"key": "max_entry_price", "label": "Max Entry Price", "type": "number", "min": 0, "max": 1},
                {"key": "base_size_usd", "label": "Base Size (USD)", "type": "number", "min": 1},
                {"key": "max_size_usd", "label": "Max Size (USD)", "type": "number", "min": 1},
            ]
        },
        aliases=[],
    ),
    SystemTraderStrategySeed(
        strategy_key="weather_alerts",
        source_key="weather",
        label="Weather Alerts",
        description="Alert-style weather dislocation strategy for sharp model/market divergence.",
        class_name="WeatherAlertsStrategy",
        import_module="services.trader_orchestrator.strategies.weather_models",
        import_class="WeatherAlertsStrategy",
        default_params={
            "min_edge_percent": 8.0,
            "min_confidence": 0.46,
            "min_temp_dislocation_c": 1.5,
            "min_source_count": 1,
            "max_source_spread_c": 6.0,
            "max_target_hours": 96.0,
            "base_size_usd": 12.0,
            "max_size_usd": 80.0,
        },
        param_schema={
            "param_fields": [
                {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0},
                {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
                {"key": "min_temp_dislocation_c", "label": "Min Temp Dislocation (C)", "type": "number", "min": 0},
                {"key": "min_source_count", "label": "Min Forecast Sources", "type": "integer", "min": 1},
                {"key": "max_source_spread_c", "label": "Max Source Spread (C)", "type": "number", "min": 0},
                {"key": "max_target_hours", "label": "Max Target Horizon (hours)", "type": "number", "min": 1},
                {"key": "base_size_usd", "label": "Base Size (USD)", "type": "number", "min": 1},
                {"key": "max_size_usd", "label": "Max Size (USD)", "type": "number", "min": 1},
            ]
        },
        aliases=[],
    ),
]


def list_system_strategy_keys() -> list[str]:
    return [seed.strategy_key for seed in SYSTEM_TRADER_STRATEGY_SEEDS]


def source_to_strategy_keys() -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for seed in SYSTEM_TRADER_STRATEGY_SEEDS:
        grouped.setdefault(seed.source_key, []).append(seed.strategy_key)
    for key in grouped:
        grouped[key] = sorted(grouped[key])
    return grouped


def default_strategy_by_source() -> dict[str, str]:
    defaults: dict[str, str] = {}
    for seed in SYSTEM_TRADER_STRATEGY_SEEDS:
        defaults.setdefault(seed.source_key, seed.strategy_key)
    return defaults


def build_system_strategy_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SYSTEM_TRADER_STRATEGY_SEEDS:
        strategy_id = uuid.uuid5(uuid.NAMESPACE_URL, f"homerun.trader_strategy.{seed.strategy_key}").hex
        rows.append(
            {
                "id": strategy_id,
                "strategy_key": seed.strategy_key,
                "source_key": seed.source_key,
                "label": seed.label,
                "description": seed.description,
                "class_name": seed.class_name,
                "source_code": _seed_source_code(seed.import_module),
                "default_params_json": dict(seed.default_params),
                "param_schema_json": dict(seed.param_schema),
                "aliases_json": list(seed.aliases),
                "is_system": True,
                "enabled": True,
                "status": "unloaded",
                "error_message": None,
                "version": 1,
            }
        )
    return rows


async def ensure_system_trader_strategies_seeded(session: AsyncSession) -> int:
    from models.database import TraderStrategyDefinition

    rows = build_system_strategy_rows()
    seed_by_key = {row["strategy_key"]: row for row in rows}
    existing = {
        row.strategy_key: row
        for row in (
            (
                await session.execute(
                    select(TraderStrategyDefinition).where(
                        TraderStrategyDefinition.strategy_key.in_(list(seed_by_key.keys()))
                    )
                )
            )
            .scalars()
            .all()
        )
    }

    inserted = 0
    rewritten = 0
    for strategy_key, seed_row in seed_by_key.items():
        current = existing.get(strategy_key)
        if current is None:
            session.add(TraderStrategyDefinition(**seed_row))
            inserted += 1
            continue

        source_code = current.source_code or ""
        if not bool(current.is_system):
            continue
        if not _is_legacy_wrapper_source(source_code):
            continue

        current.source_key = seed_row["source_key"]
        current.label = seed_row["label"]
        current.description = seed_row["description"]
        current.class_name = seed_row["class_name"]
        current.source_code = seed_row["source_code"]
        current.default_params_json = seed_row["default_params_json"]
        current.param_schema_json = seed_row["param_schema_json"]
        current.aliases_json = seed_row["aliases_json"]
        current.status = "unloaded"
        current.error_message = None
        current.version = int(current.version or 0) + 1
        current.updated_at = datetime.utcnow()
        rewritten += 1

    if inserted == 0 and rewritten == 0:
        return 0
    await session.commit()
    return inserted + rewritten
