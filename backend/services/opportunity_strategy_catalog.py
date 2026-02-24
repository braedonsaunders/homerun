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
from services.strategy_sdk import StrategySDK
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
_STALE_SYSTEM_IMPORT_MARKERS = (
    "from ._evaluate_helpers import",
    "from services.strategies._evaluate_helpers import",
    "ArbitrageOpportunity",
)
_SYSTEM_SOURCE_REQUIRED_MARKERS: dict[str, tuple[str, ...]] = {
    "traders_confluence": (
        'source_key = "traders"',
        'subscriptions = ["trader_activity"]',
        "def apply_firehose_filters",
        "def build_opportunities_from_firehose",
        "async def on_event",
    ),
    "weather_distribution": (
        'source_key = "weather"',
        'subscriptions = ["weather_update"]',
        "async def on_event",
        'weather_payload = normalized.get("weather")',
        "StrategySDK.resolve_position_sizing(",
    ),
    "weather_ensemble_edge": (
        'source_key = "weather"',
        'subscriptions = ["weather_update"]',
        "async def on_event",
        'weather_payload = normalized.get("weather")',
        "StrategySDK.resolve_position_sizing(",
    ),
    "news_edge": (
        'source_key = "news"',
        'subscriptions = ["news_update"]',
        "async def on_event",
        "StrategySDK.resolve_position_sizing(",
    ),
}


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
    if any(marker in source_code for marker in _STALE_SYSTEM_IMPORT_MARKERS):
        return True
    return any(marker in source_code for marker in _LEGACY_SOURCE_MARKERS)


def _is_stale_system_source(slug: str, source_code: str) -> bool:
    if _is_legacy_system_source(source_code):
        return True

    required = _SYSTEM_SOURCE_REQUIRED_MARKERS.get(slug, ())
    if not required:
        return False

    return any(marker not in source_code for marker in required)


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

_SCANNER_SCHEMA_NEGRISK = {
    "param_fields": [
        *_COMMON_SCANNER_SCHEMA["param_fields"][:3],
        {"key": "min_markets", "label": "Min Markets", "type": "integer", "min": 1},
        {"key": "min_total_yes", "label": "Min Total YES", "type": "number", "min": 0.5, "max": 1},
        {"key": "warn_total_yes", "label": "Warn Total YES", "type": "number", "min": 0.5, "max": 1},
        {
            "key": "election_min_total_yes",
            "label": "Election Min Total YES",
            "type": "number",
            "min": 0.5,
            "max": 1,
        },
        {
            "key": "max_resolution_spread_days",
            "label": "Max Resolution Spread (days)",
            "type": "integer",
            "min": 0,
            "max": 365,
        },
        *_COMMON_SCANNER_SCHEMA["param_fields"][3:],
    ]
}

_SCANNER_SCHEMA_COMBINATORIAL = {
    "param_fields": [
        *_COMMON_SCANNER_SCHEMA["param_fields"][:3],
        {"key": "min_markets", "label": "Min Markets", "type": "integer", "min": 1},
        {
            "key": "medium_confidence_threshold",
            "label": "Medium Confidence Threshold",
            "type": "number",
            "min": 0,
            "max": 1,
        },
        {
            "key": "high_confidence_threshold",
            "label": "High Confidence Threshold",
            "type": "number",
            "min": 0,
            "max": 1,
        },
        *_COMMON_SCANNER_SCHEMA["param_fields"][3:],
    ]
}

_SCANNER_SCHEMA_SETTLEMENT_LAG = {
    "param_fields": [
        *_COMMON_SCANNER_SCHEMA["param_fields"][:3],
        {"key": "min_liquidity", "label": "Min Liquidity", "type": "number", "min": 0},
        {
            "key": "max_days_to_resolution",
            "label": "Max Days To Resolution",
            "type": "integer",
            "min": 0,
            "max": 365,
        },
        {"key": "near_zero_threshold", "label": "Near-Zero Threshold", "type": "number", "min": 0.001, "max": 0.5},
        {"key": "near_one_threshold", "label": "Near-One Threshold", "type": "number", "min": 0.5, "max": 0.999},
        {"key": "min_sum_deviation", "label": "Min Sum Deviation", "type": "number", "min": 0.001, "max": 0.5},
        *_COMMON_SCANNER_SCHEMA["param_fields"][3:],
    ]
}

_SCANNER_SCHEMA_CROSS_PLATFORM = {
    "param_fields": [
        *_COMMON_SCANNER_SCHEMA["param_fields"][:3],
        {
            "key": "min_spread_after_fees",
            "label": "Min Spread After Fees",
            "type": "number",
            "min": 0,
            "max": 1,
        },
        *_COMMON_SCANNER_SCHEMA["param_fields"][3:],
    ]
}

_SCANNER_SCHEMA_BAYESIAN = {
    "param_fields": [
        *_COMMON_SCANNER_SCHEMA["param_fields"][:3],
        {"key": "min_propagation_edge", "label": "Min Propagation Edge", "type": "number", "min": 0, "max": 1},
        {
            "key": "max_propagation_depth",
            "label": "Max Propagation Depth",
            "type": "integer",
            "min": 1,
            "max": 10,
        },
        *_COMMON_SCANNER_SCHEMA["param_fields"][3:],
    ]
}

_SCANNER_SCHEMA_CORRELATION = {
    "param_fields": [
        *_COMMON_SCANNER_SCHEMA["param_fields"][:3],
        {"key": "min_correlation", "label": "Min Correlation", "type": "number", "min": 0, "max": 1},
        {"key": "min_divergence", "label": "Min Divergence", "type": "number", "min": 0, "max": 1},
        {"key": "z_score_threshold", "label": "Z-Score Threshold", "type": "number", "min": 0},
        *_COMMON_SCANNER_SCHEMA["param_fields"][3:],
    ]
}

_SCANNER_SCHEMA_MARKET_MAKING = {
    "param_fields": [
        *_COMMON_SCANNER_SCHEMA["param_fields"][:3],
        {"key": "min_liquidity", "label": "Min Liquidity", "type": "number", "min": 0},
        {"key": "min_volume_24h", "label": "Min Volume 24h", "type": "number", "min": 0},
        {"key": "min_spread", "label": "Min Spread", "type": "number", "min": 0, "max": 1},
        {"key": "max_spread", "label": "Max Spread", "type": "number", "min": 0, "max": 1},
        {"key": "gamma", "label": "Risk Aversion (Gamma)", "type": "number", "min": 0.01, "max": 10},
        {"key": "max_inventory_usd", "label": "Max Inventory (USD)", "type": "number", "min": 0},
        {"key": "max_hold_minutes", "label": "Max Hold (min)", "type": "number", "min": 1},
        *_COMMON_SCANNER_SCHEMA["param_fields"][3:],
    ]
}


_LABEL_ACRONYMS = {"api", "arb", "btc", "eth", "hf", "llm", "no", "roi", "sdk", "sol", "usd", "vpin", "xrp", "yes"}


def _label_from_key(key: str) -> str:
    parts = [part for part in str(key or "").split("_") if part]
    labels: list[str] = []
    for part in parts:
        lowered = part.lower()
        if lowered in _LABEL_ACRONYMS:
            labels.append(lowered.upper())
        else:
            labels.append(part.capitalize())
    return " ".join(labels) or str(key)


def _infer_param_field(key: str, default_value: object) -> dict | None:
    clean_key = str(key or "").strip()
    if not clean_key or clean_key == "_schema":
        return None

    if isinstance(default_value, bool):
        field_type = "boolean"
    elif isinstance(default_value, int):
        field_type = "integer"
    elif isinstance(default_value, float):
        field_type = "number"
    elif isinstance(default_value, list):
        field_type = "list"
    elif isinstance(default_value, dict):
        field_type = "json"
    else:
        field_type = "string"

    return {"key": clean_key, "label": _label_from_key(clean_key), "type": field_type}


def _with_retention_schema(schema: dict | None, default_config: dict | None = None) -> dict:
    merged = dict(schema or {})
    raw_param_fields = list(merged.get("param_fields") or [])
    param_fields = [dict(field) for field in raw_param_fields if isinstance(field, dict)]
    existing_keys = {str(field.get("key") or "").strip() for field in param_fields if isinstance(field, dict)}

    if isinstance(default_config, dict):
        for key, default_value in default_config.items():
            inferred = _infer_param_field(str(key), default_value)
            if not isinstance(inferred, dict):
                continue
            inferred_key = str(inferred.get("key") or "").strip()
            if not inferred_key or inferred_key in existing_keys:
                continue
            param_fields.append(inferred)
            existing_keys.add(inferred_key)

    for field in list(StrategySDK.strategy_retention_config_schema().get("param_fields", [])):
        if not isinstance(field, dict):
            continue
        key = str(field.get("key") or "").strip()
        if not key or key in existing_keys:
            continue
        param_fields.append(dict(field))
        existing_keys.add(key)

    merged["param_fields"] = param_fields
    return merged


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
        config_schema=_SCANNER_SCHEMA_NEGRISK,
    ),
    SystemOpportunityStrategySeed(
        slug="combinatorial",
        source_key="scanner",
        import_module="services.strategies.combinatorial",
        sort_order=70,
        config_schema=_SCANNER_SCHEMA_COMBINATORIAL,
    ),
    SystemOpportunityStrategySeed(
        slug="settlement_lag",
        source_key="scanner",
        import_module="services.strategies.settlement_lag",
        sort_order=80,
        config_schema=_SCANNER_SCHEMA_SETTLEMENT_LAG,
    ),
    SystemOpportunityStrategySeed(
        slug="cross_platform",
        source_key="scanner",
        import_module="services.strategies.cross_platform",
        sort_order=90,
        config_schema=_SCANNER_SCHEMA_CROSS_PLATFORM,
    ),
    SystemOpportunityStrategySeed(
        slug="bayesian_cascade",
        source_key="scanner",
        import_module="services.strategies.bayesian_cascade",
        sort_order=100,
        config_schema=_SCANNER_SCHEMA_BAYESIAN,
    ),
    SystemOpportunityStrategySeed(
        slug="vpin_toxicity",
        source_key="scanner",
        import_module="services.strategies.vpin_toxicity",
        sort_order=115,
        config_schema=_COMMON_SCANNER_SCHEMA,
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
        config_schema=_SCANNER_SCHEMA_CORRELATION,
    ),
    SystemOpportunityStrategySeed(
        slug="prob_surface_arb",
        source_key="scanner",
        import_module="services.strategies.prob_surface_arb",
        sort_order=155,
        config_schema=_COMMON_SCANNER_SCHEMA,
    ),
    SystemOpportunityStrategySeed(
        slug="market_making",
        source_key="scanner",
        import_module="services.strategies.market_making",
        sort_order=160,
        config_schema=_SCANNER_SCHEMA_MARKET_MAKING,
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
    # ── News strategies ──────────────────────────────────────
    SystemOpportunityStrategySeed(
        slug="news_edge",
        source_key="news",
        import_module="services.strategies.news_edge",
        sort_order=180,
        config_schema={
            "param_fields": [
                *StrategySDK.news_filter_config_schema().get("param_fields", []),
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
                *StrategySDK.crypto_highfreq_scope_config_schema().get("param_fields", []),
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
    # ── Traders strategies ───────────────────────────────────
    SystemOpportunityStrategySeed(
        slug="traders_confluence",
        source_key="traders",
        import_module="services.strategies.traders_confluence",
        sort_order=210,
        config_schema={
            "param_fields": [
                {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0},
                {
                    "key": "min_confluence_strength",
                    "label": "Min Confluence Strength",
                    "type": "number",
                    "min": 0,
                    "max": 1,
                },
                {"key": "risk_base_score", "label": "Base Risk Score", "type": "number", "min": 0, "max": 1},
                {"key": "base_size_usd", "label": "Base Size (USD)", "type": "number", "min": 1, "max": 10000},
                {"key": "max_size_usd", "label": "Max Size (USD)", "type": "number", "min": 1, "max": 50000},
                *StrategySDK.trader_filter_config_schema().get("param_fields", []),
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
                "config_schema": _with_retention_schema(seed.config_schema, meta["default_config"]),
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

        if bool(current.is_system):
            source_changed = (
                str(current.source_key or "") != str(row["source_key"])
                or str(current.name or "") != str(row["name"])
                or str(current.description or "") != str(row["description"])
                or str(current.source_code or "") != str(row["source_code"])
                or str(current.class_name or "") != str(row["class_name"])
                or dict(current.config_schema or {}) != dict(row["config_schema"] or {})
                or int(current.sort_order or 0) != int(row["sort_order"] or 0)
            )
            if source_changed:
                current.source_key = row["source_key"]
                current.name = row["name"]
                current.description = row["description"]
                current.source_code = row["source_code"]
                current.class_name = row["class_name"]
                current.config_schema = row["config_schema"]
                current.is_system = True
                current.status = "unloaded"
                current.error_message = None
                current.version = int(current.version or 1) + 1
                current.sort_order = row["sort_order"]
                current.updated_at = row["updated_at"]
                rewritten += 1
            continue

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
        # C/D tier strategies removed in consolidation
        "entropy_arb",
        "must_happen",
        "mutually_exclusive",
        "contradiction",
        "event_driven",
        "spread_dislocation",
        "miracle",
        "weather_edge",
        # 2026 strategy consolidation removals
        "liquidity_vacuum",
        "flb_exploiter",
        "term_premium",
        "bias_fader",
        "crypto_oracle_dislocation_reversion",
        "crypto_trend_confirmation",
        "crypto_inventory_maker",
        "crypto_regime_router",
        "crypto_velocity_reversal",
        "weather_base",
        "weather_conservative_no",
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
