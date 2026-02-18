from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import importlib
import re
import uuid

from sqlalchemy import select
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import Strategy, StrategyTombstone
from utils.logger import get_logger

logger = get_logger(__name__)

_RELATIVE_IMPORT_RE = re.compile(r"(?m)^(\s*)from\s+\.([A-Za-z_][A-Za-z0-9_]*)\s+import\s+")
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_LEGACY_WRAPPER_MARKERS = (
    "System opportunity strategy wrapper loaded from DB",
    "delegates runtime behavior to the shipped strategy class",
    "as _SeedStrategy",
)
_LEGACY_SOURCE_MARKERS = (
    "StrategyType",
    "BaseTraderStrategy",
)


@dataclass(frozen=True)
class SystemOpportunityStrategySeed:
    """Lightweight seed entry — only fields the class can't provide."""

    slug: str
    source_key: str
    import_module: str
    sort_order: int
    config_schema: dict | None = None  # param_fields for dynamic config UI
    # Optional overrides (auto-derived from class if omitted)
    name: str | None = None
    description: str | None = None
    class_name: str | None = None
    default_config: dict | None = None


def _derive_class_metadata(seed: SystemOpportunityStrategySeed) -> dict:
    """Import the strategy module and extract name, description, class_name,
    default_config from the actual Python class.  Falls back to seed values."""
    result = {
        "name": seed.name or seed.slug.replace("_", " ").title(),
        "description": seed.description or "",
        "class_name": seed.class_name or "",
        "default_config": dict(seed.default_config) if seed.default_config else {},
    }
    try:
        mod = importlib.import_module(seed.import_module)
        # Find the BaseStrategy subclass
        from services.strategies.base import BaseStrategy

        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if isinstance(obj, type) and issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
                result["class_name"] = attr_name
                if hasattr(obj, "name") and obj.name:
                    result["name"] = obj.name
                if hasattr(obj, "description") and obj.description:
                    result["description"] = obj.description
                if hasattr(obj, "default_config") and obj.default_config:
                    result["default_config"] = dict(obj.default_config)
                break
    except Exception as exc:
        logger.warning("Could not import %s to derive metadata: %s", seed.import_module, exc)
    return result


def _seed_source_code(seed: SystemOpportunityStrategySeed) -> str:
    module_rel_path = seed.import_module.replace(".", "/") + ".py"
    source_path = _BACKEND_ROOT / module_rel_path
    source = source_path.read_text(encoding="utf-8")
    # DB-loaded modules run outside the services.strategies package, so
    # relative imports must be rewritten to absolute imports.
    source = _RELATIVE_IMPORT_RE.sub(r"\1from services.strategies.\2 import ", source)
    return source


def _is_legacy_system_source(source_code: str) -> bool:
    if not source_code.strip():
        return True
    if any(marker in source_code for marker in _LEGACY_WRAPPER_MARKERS):
        return True
    return any(marker in source_code for marker in _LEGACY_SOURCE_MARKERS)


# ---------------------------------------------------------------------------
# System strategy seeds — lightweight registry
#
# name, description, class_name, and default_config are AUTO-DERIVED from
# the strategy's Python class at seed time via _derive_class_metadata().
# Only slug, source_key, import_module, sort_order, and config_schema
# (UI form metadata) are required here.
# ---------------------------------------------------------------------------

_COMMON_SCANNER_SCHEMA = {
    "param_fields": [
        {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0, "max": 100},
        {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
        {"key": "max_risk_score", "label": "Max Risk Score", "type": "number", "min": 0, "max": 1},
        {"key": "base_size_usd", "label": "Base Size (USD)", "type": "number", "min": 1, "max": 10000},
        {"key": "max_size_usd", "label": "Max Size (USD)", "type": "number", "min": 1, "max": 50000},
    ]
}

_SCANNER_SCHEMA_WITH_MARKETS = {
    "param_fields": [
        *_COMMON_SCANNER_SCHEMA["param_fields"][:3],
        {"key": "min_markets", "label": "Min Markets", "type": "integer", "min": 1},
        *_COMMON_SCANNER_SCHEMA["param_fields"][3:],
    ]
}

_SCANNER_SCHEMA_WITH_LIQUIDITY = {
    "param_fields": [
        *_COMMON_SCANNER_SCHEMA["param_fields"][:3],
        {"key": "min_liquidity", "label": "Min Liquidity", "type": "number", "min": 0},
        *_COMMON_SCANNER_SCHEMA["param_fields"][3:],
    ]
}

_SCANNER_SCHEMA_WITH_HOLD = {
    "param_fields": [
        *_COMMON_SCANNER_SCHEMA["param_fields"][:3],
        {"key": "max_hold_minutes", "label": "Max Hold (min)", "type": "number", "min": 1},
        *_COMMON_SCANNER_SCHEMA["param_fields"][3:],
    ]
}

_SCANNER_SCHEMA_WITH_LIQ_HOLD = {
    "param_fields": [
        *_COMMON_SCANNER_SCHEMA["param_fields"][:3],
        {"key": "min_liquidity", "label": "Min Liquidity", "type": "number", "min": 0},
        {"key": "max_hold_minutes", "label": "Max Hold (min)", "type": "number", "min": 1},
        *_COMMON_SCANNER_SCHEMA["param_fields"][3:],
    ]
}


SYSTEM_OPPORTUNITY_STRATEGY_SEEDS: list[SystemOpportunityStrategySeed] = [
    # ── Scanner strategies ──────────────────────────────────────
    SystemOpportunityStrategySeed(
        slug="basic",
        source_key="scanner",
        import_module="services.strategies.basic",
        sort_order=10,
        config_schema={
            "param_fields": [
                *_COMMON_SCANNER_SCHEMA["param_fields"][:3],
                {"key": "min_liquidity", "label": "Min Liquidity", "type": "number", "min": 0},
                {"key": "min_markets", "label": "Min Markets", "type": "integer", "min": 1},
                *_COMMON_SCANNER_SCHEMA["param_fields"][3:],
            ]
        },
    ),
    SystemOpportunityStrategySeed(
        slug="negrisk",
        source_key="scanner",
        import_module="services.strategies.negrisk",
        sort_order=20,
        config_schema=_SCANNER_SCHEMA_WITH_MARKETS,
    ),
    SystemOpportunityStrategySeed(
        slug="mutually_exclusive",
        source_key="scanner",
        import_module="services.strategies.mutually_exclusive",
        sort_order=30,
        config_schema=_SCANNER_SCHEMA_WITH_MARKETS,
    ),
    SystemOpportunityStrategySeed(
        slug="contradiction",
        source_key="scanner",
        import_module="services.strategies.contradiction",
        sort_order=40,
        config_schema=_SCANNER_SCHEMA_WITH_MARKETS,
    ),
    SystemOpportunityStrategySeed(
        slug="must_happen",
        source_key="scanner",
        import_module="services.strategies.must_happen",
        sort_order=50,
        config_schema=_SCANNER_SCHEMA_WITH_MARKETS,
    ),
    SystemOpportunityStrategySeed(
        slug="miracle",
        source_key="scanner",
        import_module="services.strategies.miracle",
        sort_order=60,
        config_schema=_COMMON_SCANNER_SCHEMA,
    ),
    SystemOpportunityStrategySeed(
        slug="combinatorial",
        source_key="scanner",
        import_module="services.strategies.combinatorial",
        sort_order=70,
        config_schema=_SCANNER_SCHEMA_WITH_MARKETS,
    ),
    SystemOpportunityStrategySeed(
        slug="settlement_lag",
        source_key="scanner",
        import_module="services.strategies.settlement_lag",
        sort_order=80,
        config_schema=_COMMON_SCANNER_SCHEMA,
    ),
    SystemOpportunityStrategySeed(
        slug="cross_platform",
        source_key="scanner",
        import_module="services.strategies.cross_platform",
        sort_order=90,
        config_schema=_COMMON_SCANNER_SCHEMA,
    ),
    SystemOpportunityStrategySeed(
        slug="bayesian_cascade",
        source_key="scanner",
        import_module="services.strategies.bayesian_cascade",
        sort_order=100,
        config_schema=_COMMON_SCANNER_SCHEMA,
    ),
    SystemOpportunityStrategySeed(
        slug="liquidity_vacuum",
        source_key="scanner",
        import_module="services.strategies.liquidity_vacuum",
        sort_order=110,
        config_schema=_COMMON_SCANNER_SCHEMA,
    ),
    SystemOpportunityStrategySeed(
        slug="entropy_arb",
        source_key="scanner",
        import_module="services.strategies.entropy_arb",
        sort_order=120,
        config_schema=_COMMON_SCANNER_SCHEMA,
    ),
    SystemOpportunityStrategySeed(
        slug="event_driven",
        source_key="scanner",
        import_module="services.strategies.event_driven",
        sort_order=130,
        config_schema=_SCANNER_SCHEMA_WITH_HOLD,
    ),
    SystemOpportunityStrategySeed(
        slug="temporal_decay",
        source_key="scanner",
        import_module="services.strategies.temporal_decay",
        sort_order=140,
        config_schema=_COMMON_SCANNER_SCHEMA,
    ),
    SystemOpportunityStrategySeed(
        slug="correlation_arb",
        source_key="scanner",
        import_module="services.strategies.correlation_arb",
        sort_order=150,
        config_schema=_COMMON_SCANNER_SCHEMA,
    ),
    SystemOpportunityStrategySeed(
        slug="market_making",
        source_key="scanner",
        import_module="services.strategies.market_making",
        sort_order=160,
        config_schema=_SCANNER_SCHEMA_WITH_LIQ_HOLD,
    ),
    SystemOpportunityStrategySeed(
        slug="stat_arb",
        source_key="scanner",
        import_module="services.strategies.stat_arb",
        sort_order=170,
        config_schema=_COMMON_SCANNER_SCHEMA,
    ),
    SystemOpportunityStrategySeed(
        slug="flash_crash_reversion",
        source_key="scanner",
        import_module="services.strategies.flash_crash_reversion",
        sort_order=175,
        config_schema={
            "param_fields": [
                {"key": "lookback_seconds", "label": "Lookback (seconds)", "type": "number", "min": 30},
                {"key": "drop_threshold", "label": "Drop Threshold", "type": "number", "min": 0.01, "max": 0.5},
                {
                    "key": "min_rebound_fraction",
                    "label": "Min Rebound Fraction",
                    "type": "number",
                    "min": 0.1,
                    "max": 0.95,
                },
                {"key": "min_target_move", "label": "Min Target Move", "type": "number", "min": 0.005, "max": 0.15},
                {"key": "max_entry_price", "label": "Max Entry Price", "type": "number", "min": 0.1, "max": 1},
                {"key": "max_spread", "label": "Max Spread", "type": "number", "min": 0.005, "max": 0.25},
                {"key": "min_liquidity", "label": "Min Liquidity", "type": "number", "min": 0},
                {"key": "min_abs_move_5m", "label": "Min |5m Move| (%)", "type": "number", "min": 0, "max": 100},
                {"key": "require_crash_alignment", "label": "Require Crash Alignment", "type": "boolean"},
                {
                    "key": "sizing_policy",
                    "label": "Sizing Policy",
                    "type": "enum",
                    "options": ["fixed", "linear", "adaptive", "kelly"],
                },
                {
                    "key": "kelly_fractional_scale",
                    "label": "Kelly Fractional Scale",
                    "type": "number",
                    "min": 0.05,
                    "max": 1,
                },
                {"key": "base_size_usd", "label": "Base Size (USD)", "type": "number", "min": 1, "max": 10000},
                {"key": "max_size_usd", "label": "Max Size (USD)", "type": "number", "min": 1, "max": 50000},
            ]
        },
    ),
    SystemOpportunityStrategySeed(
        slug="tail_end_carry",
        source_key="scanner",
        import_module="services.strategies.tail_end_carry",
        sort_order=176,
        config_schema={
            "param_fields": [
                {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0, "max": 100},
                {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
                {"key": "max_risk_score", "label": "Max Risk Score", "type": "number", "min": 0, "max": 1},
                {"key": "min_entry_price", "label": "Min Entry Price", "type": "number", "min": 0, "max": 1},
                {"key": "max_entry_price", "label": "Max Entry Price", "type": "number", "min": 0, "max": 1},
                {"key": "min_probability", "label": "Min Probability", "type": "number", "min": 0.5, "max": 1},
                {"key": "max_probability", "label": "Max Probability", "type": "number", "min": 0.5, "max": 1},
                {"key": "min_days_to_resolution", "label": "Min Days To Resolution", "type": "number", "min": 0},
                {"key": "max_days_to_resolution", "label": "Max Days To Resolution", "type": "number", "min": 0},
                {"key": "min_liquidity", "label": "Min Liquidity", "type": "number", "min": 0},
                {"key": "max_spread", "label": "Max Spread", "type": "number", "min": 0.005, "max": 0.2},
                {
                    "key": "min_repricing_buffer",
                    "label": "Min Repricing Buffer",
                    "type": "number",
                    "min": 0.005,
                    "max": 0.1,
                },
                {"key": "repricing_weight", "label": "Repricing Weight", "type": "number", "min": 0.1, "max": 0.9},
                {
                    "key": "sizing_policy",
                    "label": "Sizing Policy",
                    "type": "enum",
                    "options": ["fixed", "linear", "adaptive", "kelly"],
                },
                {"key": "base_size_usd", "label": "Base Size (USD)", "type": "number", "min": 1, "max": 10000},
                {"key": "max_size_usd", "label": "Max Size (USD)", "type": "number", "min": 1, "max": 50000},
            ]
        },
    ),
    SystemOpportunityStrategySeed(
        slug="spread_dislocation",
        source_key="scanner",
        import_module="services.strategies.spread_dislocation",
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
    # ── News strategies ──────────────────────────────────────
    SystemOpportunityStrategySeed(
        slug="news_edge",
        source_key="news",
        import_module="services.strategies.news_edge",
        sort_order=180,
        config_schema={
            "param_fields": [
                {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0, "max": 100},
                {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
                {"key": "base_size_usd", "label": "Base Size (USD)", "type": "number", "min": 1, "max": 10000},
                {"key": "max_size_usd", "label": "Max Size (USD)", "type": "number", "min": 1, "max": 50000},
            ]
        },
    ),
    # ── Crypto strategies ────────────────────────────────────
    SystemOpportunityStrategySeed(
        slug="btc_eth_highfreq",
        source_key="crypto",
        import_module="services.strategies.btc_eth_highfreq",
        sort_order=190,
        config_schema={
            "param_fields": [
                {
                    "key": "strategy_mode",
                    "label": "Strategy Mode",
                    "type": "enum",
                    "options": ["auto", "5m", "15m", "1h", "4h"],
                },
                {"key": "target_assets", "label": "Target Assets", "type": "list"},
                {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0, "max": 100},
                {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
                {"key": "direction_guardrail_enabled", "label": "Direction Guardrail Enabled", "type": "boolean"},
                {
                    "key": "direction_guardrail_prob_floor",
                    "label": "Direction Guardrail Prob Floor",
                    "type": "number",
                    "min": 0,
                    "max": 1,
                },
                {
                    "key": "direction_guardrail_price_floor",
                    "label": "Direction Guardrail Price Floor",
                    "type": "number",
                    "min": 0,
                    "max": 1,
                },
                {"key": "direction_guardrail_regimes", "label": "Direction Guardrail Regimes", "type": "list"},
                {"key": "base_size_usd", "label": "Base Size (USD)", "type": "number", "min": 1, "max": 10000},
                {"key": "max_size_usd", "label": "Max Size (USD)", "type": "number", "min": 1, "max": 50000},
            ]
        },
    ),
    SystemOpportunityStrategySeed(
        slug="crypto_spike_reversion",
        source_key="crypto",
        import_module="services.strategies.crypto_spike_reversion",
        sort_order=191,
        config_schema={
            "param_fields": [
                {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0, "max": 100},
                {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
                {"key": "min_abs_move_5m", "label": "Min |5m Move| (%)", "type": "number", "min": 0, "max": 100},
                {"key": "max_abs_move_2h", "label": "Max |2h Move| (%)", "type": "number", "min": 0, "max": 100},
                {"key": "require_reversion_shape", "label": "Require Reversion Shape", "type": "boolean"},
                {
                    "key": "sizing_policy",
                    "label": "Sizing Policy",
                    "type": "enum",
                    "options": ["fixed", "linear", "adaptive", "kelly"],
                },
                {
                    "key": "kelly_fractional_scale",
                    "label": "Kelly Fractional Scale",
                    "type": "number",
                    "min": 0.05,
                    "max": 1,
                },
                {"key": "base_size_usd", "label": "Base Size (USD)", "type": "number", "min": 1, "max": 10000},
                {"key": "max_size_usd", "label": "Max Size (USD)", "type": "number", "min": 1, "max": 50000},
            ]
        },
    ),
    # ── Weather strategies ───────────────────────────────────
    SystemOpportunityStrategySeed(
        slug="weather_edge",
        source_key="weather",
        import_module="services.strategies.weather_edge",
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
                {"key": "base_size_usd", "label": "Base Size (USD)", "type": "number", "min": 1, "max": 10000},
                {"key": "max_size_usd", "label": "Max Size (USD)", "type": "number", "min": 1, "max": 50000},
            ]
        },
    ),
    SystemOpportunityStrategySeed(
        slug="weather_ensemble_edge",
        source_key="weather",
        import_module="services.strategies.weather_ensemble_edge",
        sort_order=201,
        config_schema={
            "param_fields": [
                {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0},
                {"key": "min_ensemble_members", "label": "Min Ensemble Members", "type": "integer", "min": 1},
                {
                    "key": "min_ensemble_agreement",
                    "label": "Min Ensemble Agreement",
                    "type": "number",
                    "min": 0,
                    "max": 1,
                },
                {"key": "max_entry_price", "label": "Max Entry Price", "type": "number", "min": 0, "max": 1},
                {
                    "key": "deterministic_fallback",
                    "label": "Deterministic Fallback",
                    "type": "enum",
                    "options": ["true", "false"],
                },
                {
                    "key": "probability_scale_c",
                    "label": "Fallback Sigmoid Scale (C)",
                    "type": "number",
                    "min": 0.5,
                    "max": 5.0,
                },
                {"key": "risk_base_score", "label": "Base Risk Score", "type": "number", "min": 0, "max": 1},
            ]
        },
    ),
    SystemOpportunityStrategySeed(
        slug="weather_distribution",
        source_key="weather",
        import_module="services.strategies.weather_distribution",
        sort_order=202,
        config_schema={
            "param_fields": [
                {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0},
                {"key": "sigma_c", "label": "Sigma (C)", "type": "number", "min": 0.5, "max": 5.0},
                {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
                {"key": "max_entry_price", "label": "Max Entry Price", "type": "number", "min": 0, "max": 1},
                {
                    "key": "max_buckets_per_event",
                    "label": "Max Buckets per Event",
                    "type": "integer",
                    "min": 1,
                    "max": 10,
                },
                {"key": "risk_base_score", "label": "Base Risk Score", "type": "number", "min": 0, "max": 1},
                {"key": "base_size_usd", "label": "Base Size (USD)", "type": "number", "min": 1, "max": 10000},
                {"key": "max_size_usd", "label": "Max Size (USD)", "type": "number", "min": 1, "max": 50000},
            ]
        },
    ),
    SystemOpportunityStrategySeed(
        slug="weather_conservative_no",
        source_key="weather",
        import_module="services.strategies.weather_conservative_no",
        sort_order=203,
        config_schema={
            "param_fields": [
                {"key": "min_safe_distance_c", "label": "Min Distance from Forecast (C)", "type": "number", "min": 0},
                {"key": "max_no_price", "label": "Max NO Price", "type": "number", "min": 0, "max": 1},
                {"key": "min_model_agreement", "label": "Min Model Agreement", "type": "number", "min": 0, "max": 1},
                {"key": "max_source_spread_c", "label": "Max Source Spread (C)", "type": "number", "min": 0},
                {"key": "min_source_count", "label": "Min Forecast Sources", "type": "integer", "min": 1},
                {
                    "key": "max_positions_per_event",
                    "label": "Max Positions per Event",
                    "type": "integer",
                    "min": 1,
                    "max": 10,
                },
                {"key": "risk_base_score", "label": "Base Risk Score", "type": "number", "min": 0, "max": 1},
            ]
        },
    ),
    # ── Traders strategies ───────────────────────────────────
    SystemOpportunityStrategySeed(
        slug="traders_confluence",
        source_key="traders",
        import_module="services.strategies.traders_confluence",
        sort_order=210,
        config_schema={
            "param_fields": [
                {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0},
                {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
                {
                    "key": "min_confluence_strength",
                    "label": "Min Confluence Strength",
                    "type": "number",
                    "min": 0,
                    "max": 1,
                },
                {
                    "key": "min_tier",
                    "label": "Min Tier",
                    "type": "enum",
                    "options": ["low", "medium", "high", "extreme"],
                },
                {"key": "min_wallet_count", "label": "Min Wallet Count", "type": "integer", "min": 1},
                {"key": "max_entry_price", "label": "Max Entry Price", "type": "number", "min": 0, "max": 1},
                {"key": "risk_base_score", "label": "Base Risk Score", "type": "number", "min": 0, "max": 1},
                {"key": "firehose_require_tradable_market", "label": "Require Tradable Market", "type": "boolean"},
                {"key": "firehose_exclude_crypto_markets", "label": "Exclude Crypto Markets", "type": "boolean"},
                {"key": "firehose_require_qualified_source", "label": "Require Qualified Source", "type": "boolean"},
                {
                    "key": "firehose_max_age_minutes",
                    "label": "Firehose Max Age (min)",
                    "type": "integer",
                    "min": 1,
                    "max": 1440,
                },
                {"key": "base_size_usd", "label": "Base Size (USD)", "type": "number", "min": 1, "max": 10000},
                {"key": "max_size_usd", "label": "Max Size (USD)", "type": "number", "min": 1, "max": 50000},
            ]
        },
    ),
]


def build_system_opportunity_strategy_rows(*, now: datetime | None = None) -> list[dict]:
    ts = now or datetime.utcnow()
    rows: list[dict] = []
    for seed in SYSTEM_OPPORTUNITY_STRATEGY_SEEDS:
        meta = _derive_class_metadata(seed)
        rows.append(
            {
                "id": uuid.uuid4().hex,
                "slug": seed.slug,
                "source_key": seed.source_key,
                "name": meta["name"],
                "description": meta["description"],
                "source_code": _seed_source_code(seed),
                "class_name": meta["class_name"],
                "is_system": True,
                "enabled": True,
                "status": "unloaded",
                "error_message": None,
                "config": meta["default_config"],
                "config_schema": seed.config_schema or {},
                "aliases": [],
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
                    select(StrategyTombstone.slug).where(StrategyTombstone.slug.in_(list(seed_by_slug.keys())))
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
            (await session.execute(select(Strategy).where(Strategy.slug.in_(list(seed_by_slug.keys())))))
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
            session.add(Strategy(**row))
            inserted += 1
            continue

        if bool(current.is_system) and _is_legacy_system_source(str(current.source_code or "")):
            current.source_key = row["source_key"]
            current.name = row["name"]
            current.description = row["description"]
            current.source_code = row["source_code"]
            current.class_name = row["class_name"]
            current.config_schema = row["config_schema"]
            current.is_system = True
            current.status = "unloaded"
            current.error_message = None
            current.sort_order = row["sort_order"]
            current.updated_at = row["updated_at"]
            rewritten += 1
            continue

        # Seed-once: if the strategy already exists in the DB, DO NOT overwrite.
        # User may have customized source_code, config, or description.
        # A separate "reset to factory" API endpoint can restore the original.
        continue

    # Disable old execution-only duplicate slugs that are now aliases on
    # unified entries, plus any previously removed strategies.
    _REMOVED_SLUGS = [
        "weather_bucket_edge",
        # Old execution-only duplicates (now aliases on unified entries)
        "crypto_5m",
        "crypto_15m",
        "crypto_1h",
        "crypto_4h",
        "opportunity_general",
        "opportunity_structural",
        "opportunity_flash_reversion",
        "opportunity_tail_carry",
        "news_reaction",
        "traders_flow",
        "weather_consensus",
        "weather_alerts",
    ]
    orphan_rows = {
        row.slug: row
        for row in ((await session.execute(select(Strategy).where(Strategy.slug.in_(_REMOVED_SLUGS)))).scalars().all())
    }
    for slug in _REMOVED_SLUGS:
        removed = orphan_rows.get(slug)
        if removed:
            await session.delete(removed)
            rewritten += 1

    if inserted == 0 and rewritten == 0:
        return 0
    await session.commit()
    return inserted + rewritten


# ---------------------------------------------------------------------------
# Helper functions — single catalog, single source of truth
# ---------------------------------------------------------------------------


def list_system_strategy_keys() -> list[str]:
    """Return every system strategy slug."""
    return sorted(seed.slug for seed in SYSTEM_OPPORTUNITY_STRATEGY_SEEDS)


def source_to_strategy_keys() -> dict[str, list[str]]:
    """Map source_key → list of strategy slugs."""
    grouped: dict[str, list[str]] = {}
    for seed in SYSTEM_OPPORTUNITY_STRATEGY_SEEDS:
        items = grouped.setdefault(seed.source_key, [])
        items.append(seed.slug)
    for key in grouped:
        grouped[key] = sorted(set(grouped[key]))
    return grouped


def default_strategy_by_source() -> dict[str, str]:
    """Return the first (default) strategy slug for each source."""
    defaults: dict[str, str] = {}
    for seed in SYSTEM_OPPORTUNITY_STRATEGY_SEEDS:
        defaults.setdefault(seed.source_key, seed.slug)
    return defaults


async def ensure_all_strategies_seeded(session: AsyncSession) -> dict:
    """Seed all strategies from the single unified catalog."""
    result = await ensure_system_opportunity_strategies_seeded(session)
    return {"seeded": result}


async def reset_strategy_to_factory(session: AsyncSession, slug: str) -> dict:
    """Reset a system strategy to its factory seed definition.

    Only works for system strategies with a corresponding seed.
    Returns dict with status and details.
    """
    rows = build_system_opportunity_strategy_rows()
    seed_row = next((r for r in rows if r["slug"] == slug), None)
    if seed_row is None:
        return {"status": "not_found", "detail": f"No system seed for slug '{slug}'"}

    current = (await session.execute(select(Strategy).where(Strategy.slug == slug))).scalar_one_or_none()

    if current is None:
        session.add(Strategy(**seed_row))
        await session.commit()
        return {"status": "created", "detail": f"Strategy '{slug}' recreated from seed"}

    current.source_key = seed_row["source_key"]
    current.name = seed_row["name"]
    current.description = seed_row["description"]
    current.source_code = seed_row["source_code"]
    current.class_name = seed_row["class_name"]
    current.config = seed_row["config"]
    current.config_schema = seed_row["config_schema"]
    current.is_system = True
    current.sort_order = seed_row["sort_order"]
    current.status = "unloaded"
    current.error_message = None
    current.version = int(current.version or 0) + 1
    current.updated_at = datetime.utcnow()
    await session.commit()
    return {"status": "reset", "detail": f"Strategy '{slug}' reset to factory defaults"}
