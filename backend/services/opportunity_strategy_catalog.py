from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from datetime import datetime, timezone
from pathlib import Path
import importlib
import re
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import Strategy
from services.strategy_sdk import StrategySDK
from services.strategies.news_edge import news_edge_config_schema
from services.strategies.traders_copy_trade import traders_copy_trade_config_schema
from utils.logger import get_logger
from utils.utcnow import utcnow

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
    "traders_copy_trade": (
        'source_key = "traders"',
        "strategy_type = \"traders_copy_trade\"",
        "def evaluate",
    ),
    "weather_distribution": (
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
        {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0, "max": 100, "phase": "signal"},
        {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1, "phase": "signal"},
        {"key": "max_risk_score", "label": "Max Risk Score", "type": "number", "min": 0, "max": 1, "phase": "signal"},
    ]
}

_SCANNER_SCHEMA_NEGRISK = {
    "param_fields": [
        *_COMMON_SCANNER_SCHEMA["param_fields"][:3],
        {"key": "min_markets", "label": "Min Markets", "type": "integer", "min": 1, "phase": "signal"},
        {"key": "min_total_yes", "label": "Min Total YES", "type": "number", "min": 0.5, "max": 1, "phase": "signal"},
        {"key": "warn_total_yes", "label": "Warn Total YES", "type": "number", "min": 0.5, "max": 1, "phase": "signal"},
        {
            "key": "election_min_total_yes",
            "label": "Election Min Total YES",
            "type": "number",
            "min": 0.5,
            "max": 1,
            "phase": "signal",
        },
        {
            "key": "max_resolution_spread_days",
            "label": "Max Resolution Spread (days)",
            "type": "integer",
            "min": 0,
            "max": 365,
            "phase": "signal",
        },
        *_COMMON_SCANNER_SCHEMA["param_fields"][3:],
    ]
}

_SCANNER_SCHEMA_COMBINATORIAL = {
    "param_fields": [
        *_COMMON_SCANNER_SCHEMA["param_fields"][:3],
        {"key": "min_markets", "label": "Min Markets", "type": "integer", "min": 1, "phase": "signal"},
        {
            "key": "medium_confidence_threshold",
            "label": "Medium Confidence Threshold",
            "type": "number",
            "min": 0,
            "max": 1,
            "phase": "signal",
        },
        {
            "key": "high_confidence_threshold",
            "label": "High Confidence Threshold",
            "type": "number",
            "min": 0,
            "max": 1,
            "phase": "signal",
        },
        *_COMMON_SCANNER_SCHEMA["param_fields"][3:],
    ]
}

_SCANNER_SCHEMA_SETTLEMENT_LAG = {
    "param_fields": [
        *_COMMON_SCANNER_SCHEMA["param_fields"][:3],
        {"key": "min_liquidity", "label": "Min Liquidity", "type": "number", "min": 0, "phase": "signal"},
        {
            "key": "max_days_to_resolution",
            "label": "Max Days To Resolution",
            "type": "integer",
            "min": 0,
            "max": 365,
            "phase": "signal",
        },
        {"key": "near_zero_threshold", "label": "Near-Zero Threshold", "type": "number", "min": 0.001, "max": 0.5, "phase": "signal"},
        {"key": "near_one_threshold", "label": "Near-One Threshold", "type": "number", "min": 0.5, "max": 0.999, "phase": "signal"},
        {"key": "min_sum_deviation", "label": "Min Sum Deviation", "type": "number", "min": 0.001, "max": 0.5, "phase": "signal"},
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
            "phase": "signal",
        },
        *_COMMON_SCANNER_SCHEMA["param_fields"][3:],
    ]
}

_SCANNER_SCHEMA_STAT_ARB = {
    "param_fields": [
        *_COMMON_SCANNER_SCHEMA["param_fields"][:3],
        {"key": "enable_stat_signals", "label": "Enable Statistical Signals", "type": "boolean", "phase": "signal"},
        {"key": "exclude_market_keywords", "label": "Exclude Market Keywords", "type": "list", "phase": "signal"},
        {"key": "take_profit_pct", "label": "Take Profit (%)", "type": "number", "min": 1, "max": 100, "phase": "exit"},
        {"key": "stop_loss_pct", "label": "Stop Loss (%)", "type": "number", "min": 1, "max": 100, "phase": "exit"},
        {"key": "trailing_stop_pct", "label": "Trailing Stop (%)", "type": "number", "min": 1, "max": 100, "phase": "exit"},
        *_COMMON_SCANNER_SCHEMA["param_fields"][3:],
    ]
}

_SCANNER_SCHEMA_CERTAINTY_SHOCK = {
    "param_fields": [
        *_COMMON_SCANNER_SCHEMA["param_fields"][:3],
        {"key": "shock_lookback_seconds", "label": "Lookback (sec)", "type": "number", "min": 60, "phase": "signal"},
        {"key": "shock_min_abs_move", "label": "Min Absolute Move", "type": "number", "min": 0.05, "max": 1, "phase": "signal"},
        {"key": "shock_max_retrace", "label": "Max Retrace", "type": "number", "min": 0, "max": 1, "phase": "signal"},
        {"key": "shock_min_favored_price", "label": "Min Favored Price", "type": "number", "min": 0.01, "max": 0.99, "phase": "signal"},
        {"key": "shock_target_certainty", "label": "Target Certainty", "type": "number", "min": 0.5, "max": 0.995, "phase": "signal"},
        {"key": "min_days_to_deadline", "label": "Min Days To Deadline", "type": "number", "min": 0, "max": 365, "phase": "signal"},
        {"key": "max_days_to_deadline", "label": "Max Days To Deadline", "type": "number", "min": 0, "max": 365, "phase": "signal"},
        {"key": "exclude_market_keywords", "label": "Exclude Market Keywords", "type": "list", "phase": "signal"},
        {"key": "take_profit_pct", "label": "Take Profit (%)", "type": "number", "min": 1, "max": 100, "phase": "exit"},
        {"key": "stop_loss_pct", "label": "Stop Loss (%)", "type": "number", "min": 1, "max": 100, "phase": "exit"},
        {"key": "trailing_stop_pct", "label": "Trailing Stop (%)", "type": "number", "min": 1, "max": 100, "phase": "exit"},
        *_COMMON_SCANNER_SCHEMA["param_fields"][3:],
    ]
}

_SCANNER_SCHEMA_TEMPORAL_DECAY = {
    "param_fields": [
        *_COMMON_SCANNER_SCHEMA["param_fields"][:3],
        {"key": "min_days_to_deadline", "label": "Min Days To Deadline", "type": "number", "min": 0, "max": 365, "phase": "signal"},
        {"key": "max_days_to_deadline", "label": "Max Days To Deadline", "type": "number", "min": 0, "max": 365, "phase": "signal"},
        {"key": "min_deviation", "label": "Min Deviation", "type": "number", "min": 0.01, "max": 1, "phase": "signal"},
        {"key": "min_history_points", "label": "Min History Points", "type": "integer", "min": 3, "max": 50, "phase": "signal"},
        {"key": "decay_rate", "label": "Decay Rate (sqrt-time exponent)", "type": "number", "min": 0.1, "max": 2, "phase": "signal"},
        {"key": "exclude_market_keywords", "label": "Exclude Market Keywords", "type": "list", "phase": "signal"},
        {"key": "take_profit_pct", "label": "Take Profit (%)", "type": "number", "min": 1, "max": 100, "phase": "exit"},
        {"key": "stop_loss_pct", "label": "Stop Loss (%)", "type": "number", "min": 1, "max": 100, "phase": "exit"},
        {"key": "trailing_stop_pct", "label": "Trailing Stop (%)", "type": "number", "min": 1, "max": 100, "phase": "exit"},
        *_COMMON_SCANNER_SCHEMA["param_fields"][3:],
    ]
}

_SCANNER_SCHEMA_MARKET_MAKING = {
    "param_fields": [
        *_COMMON_SCANNER_SCHEMA["param_fields"][:3],
        {"key": "min_liquidity", "label": "Min Liquidity", "type": "number", "min": 0, "phase": "signal"},
        {"key": "min_volume_24h", "label": "Min Volume 24h", "type": "number", "min": 0, "phase": "signal"},
        {"key": "min_spread", "label": "Min Spread", "type": "number", "min": 0, "max": 1, "phase": "signal"},
        {"key": "max_spread", "label": "Max Spread", "type": "number", "min": 0, "max": 1, "phase": "signal"},
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


def _dedupe_param_fields(raw_fields: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[str] = set()
    for field in raw_fields:
        if not isinstance(field, dict):
            continue
        key = str(field.get("key") or "").strip()
        if not key or key in seen:
            continue
        deduped.append(dict(field))
        seen.add(key)
    return deduped


def _with_retention_schema(schema: dict | None, default_config: dict | None = None) -> dict:
    merged = dict(schema or {})
    raw_param_fields = list(merged.get("param_fields") or [])
    param_fields = _dedupe_param_fields([dict(field) for field in raw_param_fields if isinstance(field, dict)])
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

    for field in list(StrategySDK.trader_strategy_execution_config_schema().get("param_fields", [])):
        if not isinstance(field, dict):
            continue
        key = str(field.get("key") or "").strip()
        if not key or key in existing_keys:
            continue
        param_fields.append(dict(field))
        existing_keys.add(key)

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
                {"key": "min_liquidity", "label": "Min Liquidity", "type": "number", "min": 0, "phase": "signal"},
                {"key": "min_book_depth", "label": "Min Ask Depth", "type": "number", "min": 0, "phase": "signal"},
                {"key": "max_leg_spread", "label": "Max Leg Spread", "type": "number", "min": 0, "max": 1, "phase": "signal"},
                {
                    "key": "require_accepting_orders",
                    "label": "Require Accepting Orders",
                    "type": "boolean",
                    "phase": "signal",
                },
                {"key": "require_order_book", "label": "Require Order Book", "type": "boolean", "phase": "signal"},
                {"key": "require_polymarket", "label": "Require Polymarket", "type": "boolean", "phase": "signal"},
                *_COMMON_SCANNER_SCHEMA["param_fields"][3:],
            ]
        },
    ),
    SystemOpportunityStrategySeed(
        slug="ctf_basic_arb",
        source_key="scanner",
        import_module="services.strategies.ctf_basic_arb",
        sort_order=15,
        config_schema={
            "param_fields": [
                {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0, "max": 100, "phase": "signal"},
                {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1, "phase": "signal"},
                {"key": "max_risk_score", "label": "Max Risk Score", "type": "number", "min": 0, "max": 1, "phase": "signal"},
                {"key": "min_liquidity", "label": "Min Liquidity", "type": "number", "min": 0, "phase": "signal"},
                {"key": "gas_buffer_usd", "label": "Gas Buffer (USD)", "type": "number", "min": 0, "phase": "signal"},
                {"key": "assumed_trade_size_usd", "label": "Assumed Trade Size (USD)", "type": "number", "min": 1, "phase": "signal"},
                {"key": "require_accepting_orders", "label": "Require Accepting Orders", "type": "boolean", "phase": "signal"},
                {"key": "require_order_book", "label": "Require Order Book", "type": "boolean", "phase": "signal"},
                {"key": "allow_split_sell", "label": "Allow Split/Sell", "type": "boolean", "phase": "signal"},
                {"key": "allow_buy_merge", "label": "Allow Buy/Merge", "type": "boolean", "phase": "signal"},
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
        slug="vpin_toxicity",
        source_key="scanner",
        import_module="services.strategies.vpin_toxicity",
        sort_order=115,
        config_schema=_COMMON_SCANNER_SCHEMA,
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
        slug="holding_reward_yield",
        source_key="scanner",
        import_module="services.strategies.holding_reward_yield",
        sort_order=165,
        config_schema={
            "param_fields": [
                {"key": "min_apy", "label": "Min APY (%)", "type": "number", "min": 0.1, "max": 50, "phase": "signal"},
                {"key": "min_liquidity", "label": "Min Liquidity (USD)", "type": "number", "min": 0, "phase": "signal"},
                {"key": "min_days_to_resolution", "label": "Min Days To Resolution", "type": "number", "min": 1, "phase": "signal"},
                {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0, "max": 100, "phase": "signal"},
                {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
                {"key": "max_risk_score", "label": "Max Risk Score", "type": "number", "min": 0, "max": 1},
            ]
        },
    ),
    SystemOpportunityStrategySeed(
        slug="stat_arb",
        source_key="scanner",
        import_module="services.strategies.stat_arb",
        sort_order=170,
        config_schema=_SCANNER_SCHEMA_STAT_ARB,
    ),
    SystemOpportunityStrategySeed(
        slug="certainty_shock",
        source_key="scanner",
        import_module="services.strategies.certainty_shock",
        sort_order=171,
        config_schema=_SCANNER_SCHEMA_CERTAINTY_SHOCK,
    ),
    SystemOpportunityStrategySeed(
        slug="temporal_decay",
        source_key="scanner",
        import_module="services.strategies.temporal_decay",
        sort_order=172,
        config_schema=_SCANNER_SCHEMA_TEMPORAL_DECAY,
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
                {"key": "exclude_crypto_markets", "label": "Exclude Crypto Markets", "type": "boolean"},
                {"key": "exclude_market_keywords", "label": "Exclude Market Keywords", "type": "list"},
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
            ]
        },
    ),
    SystemOpportunityStrategySeed(
        slug="news_momentum_breakout",
        source_key="scanner",
        import_module="services.strategies.news_momentum_breakout",
        sort_order=177,
        config_schema={
            "param_fields": [
                {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0, "max": 100, "phase": "signal"},
                {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1, "phase": "signal"},
                {"key": "max_risk_score", "label": "Max Risk Score", "type": "number", "min": 0, "max": 1, "phase": "signal"},
                {"key": "lookback_seconds", "label": "Lookback (seconds)", "type": "number", "min": 60, "phase": "signal"},
                {"key": "breakout_threshold", "label": "Breakout Threshold (price)", "type": "number", "min": 0.02, "max": 0.5, "phase": "signal"},
                {"key": "min_target_move", "label": "Min Target Move (price)", "type": "number", "min": 0.005, "max": 0.30, "phase": "signal"},
                {"key": "target_distance_to_one_fraction", "label": "Target Fraction of Distance to 1.0", "type": "number", "min": 0.10, "max": 0.95, "phase": "signal"},
                {"key": "min_entry_price", "label": "Min Entry Price", "type": "number", "min": 0.05, "max": 0.5, "phase": "signal"},
                {"key": "max_entry_price", "label": "Max Entry Price", "type": "number", "min": 0.5, "max": 0.95, "phase": "signal"},
                {"key": "max_spread", "label": "Max Spread", "type": "number", "min": 0.005, "max": 0.25, "phase": "signal"},
                {"key": "min_liquidity", "label": "Min Liquidity (USD)", "type": "number", "min": 0, "phase": "signal"},
                {"key": "max_retrace_from_peak_fraction", "label": "Max Retrace From Peak", "type": "number", "min": 0.05, "max": 0.90, "phase": "signal"},
                {"key": "min_5m_share_of_30m", "label": "Min 5m Share of 30m Move", "type": "number", "min": 0.0, "max": 1.0, "phase": "signal"},
                {"key": "max_abs_move_2h_pct", "label": "Max |2h Move| (%)", "type": "number", "min": 5, "max": 500, "phase": "signal"},
                {"key": "require_breakout_shape", "label": "Require Breakout Shape", "type": "boolean", "phase": "signal"},
                {"key": "require_breakout_alignment", "label": "Require Breakout Alignment", "type": "boolean", "phase": "signal"},
                {"key": "min_abs_move_5m", "label": "Min |5m Move| (%)", "type": "number", "min": 0, "max": 100, "phase": "signal"},
                {"key": "emit_cooldown_seconds", "label": "Emit Cooldown (seconds)", "type": "number", "min": 0, "phase": "signal"},
                {"key": "exclude_crypto_markets", "label": "Exclude Crypto Markets", "type": "boolean", "phase": "signal"},
                {"key": "exclude_sports_markets", "label": "Exclude Sports Markets", "type": "boolean", "phase": "signal"},
                {"key": "exclude_market_keywords", "label": "Exclude Market Keywords", "type": "list", "phase": "signal"},
                {
                    "key": "sizing_policy",
                    "label": "Sizing Policy",
                    "type": "enum",
                    "options": ["fixed", "linear", "adaptive", "kelly"],
                },
                {"key": "kelly_fractional_scale", "label": "Kelly Fractional Scale", "type": "number", "min": 0.05, "max": 1},
                {"key": "take_profit_pct", "label": "Take Profit (%)", "type": "number", "min": 0, "max": 500},
                {"key": "stop_loss_pct", "label": "Stop Loss (%)", "type": "number", "min": 0, "max": 100},
                {"key": "trailing_stop_pct", "label": "Trailing Stop (%)", "type": "number", "min": 0, "max": 80},
                {
                    "key": "trailing_stop_activation_profit_pct",
                    "label": "Trailing Activation Profit (%)",
                    "type": "number",
                    "min": 0,
                    "max": 200,
                },
                {"key": "max_hold_minutes", "label": "Max Hold (minutes)", "type": "number", "min": 0, "max": 10080},
                {"key": "momentum_stall_minutes", "label": "Momentum Stall (minutes)", "type": "number", "min": 0, "max": 1440},
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
                {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0, "max": 100, "phase": "signal"},
                {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1, "phase": "signal"},
                {"key": "max_risk_score", "label": "Max Risk Score", "type": "number", "min": 0, "max": 1, "phase": "signal"},
                {"key": "min_entry_price", "label": "Min Entry Price", "type": "number", "min": 0, "max": 1, "phase": "signal"},
                {"key": "max_entry_price", "label": "Max Entry Price", "type": "number", "min": 0, "max": 1, "phase": "signal"},
                {"key": "min_probability", "label": "Min Probability", "type": "number", "min": 0.5, "max": 1, "phase": "signal"},
                {"key": "max_probability", "label": "Max Probability", "type": "number", "min": 0.5, "max": 1, "phase": "signal"},
                {
                    "key": "min_upside_percent",
                    "label": "Min Settlement Upside (%)",
                    "type": "number",
                    "min": 5,
                    "max": 100,
                    "phase": "signal",
                },
                {"key": "min_days_to_resolution", "label": "Min Days To Resolution", "type": "number", "min": 0, "phase": "signal"},
                {"key": "max_days_to_resolution", "label": "Max Days To Resolution", "type": "number", "min": 0, "phase": "signal"},
                {
                    "key": "exclude_market_keywords",
                    "label": "Exclude Market Name Contains",
                    "type": "list",
                    "phase": "signal",
                },
                {"key": "min_liquidity", "label": "Min Liquidity", "type": "number", "min": 0, "phase": "signal"},
                {"key": "max_spread", "label": "Max Spread", "type": "number", "min": 0.005, "max": 0.2, "phase": "signal"},
                {
                    "key": "min_repricing_buffer",
                    "label": "Min Repricing Buffer",
                    "type": "number",
                    "min": 0.005,
                    "max": 0.1,
                    "phase": "signal",
                },
                {"key": "repricing_weight", "label": "Repricing Weight", "type": "number", "min": 0.1, "max": 0.9, "phase": "signal"},
                {
                    "key": "sizing_policy",
                    "label": "Sizing Policy",
                    "type": "enum",
                    "options": ["fixed", "linear", "adaptive", "kelly"],
                },
                {"key": "block_spread_markets", "label": "Block Spread Markets", "type": "boolean", "phase": "signal"},
                {"key": "skip_live_games", "label": "Skip Live Games", "type": "boolean", "phase": "signal"},
                {
                    "key": "live_game_buffer_minutes",
                    "label": "Live Game Buffer (min)",
                    "type": "number",
                    "min": 0,
                    "max": 120,
                    "phase": "signal",
                },
                {"key": "take_profit_pct", "label": "Take Profit (%)", "type": "number", "min": 0, "max": 100},
                {"key": "trailing_stop_pct", "label": "Trailing Stop (%)", "type": "number", "min": 0, "max": 100},
                {"key": "smart_take_profit_enabled", "label": "Smart Take Profit Enabled", "type": "boolean"},
                {
                    "key": "smart_take_profit_min_pnl_pct",
                    "label": "Smart TP Min PnL (%)",
                    "type": "number",
                    "min": 0,
                    "max": 100,
                },
                {
                    "key": "smart_take_profit_max_price_headroom",
                    "label": "Smart TP Max Price Headroom",
                    "type": "number",
                    "min": 0.001,
                    "max": 0.2,
                },
                {
                    "key": "inversion_stop_enabled",
                    "label": "Inversion Stop Enabled",
                    "type": "boolean",
                },
                {
                    "key": "inversion_price_threshold",
                    "label": "Inversion Price Threshold",
                    "type": "number",
                    "min": 0.05,
                    "max": 0.95,
                },
                {"key": "trailing_stop_enabled", "label": "Trailing Stop Enabled", "type": "boolean"},
                {
                    "key": "trailing_stop_pct",
                    "label": "Trailing Stop (%)",
                    "type": "number",
                    "min": 5,
                    "max": 80,
                },
                {
                    "key": "sports_inversion_stop_enabled",
                    "label": "Sports: Inversion Stop",
                    "type": "boolean",
                },
                {
                    "key": "sports_trailing_stop_pct",
                    "label": "Sports: Trailing Stop (%)",
                    "type": "number",
                    "min": 5,
                    "max": 80,
                },
                {
                    "key": "sports_sizing_multiplier",
                    "label": "Sports: Sizing Multiplier",
                    "type": "number",
                    "min": 0.1,
                    "max": 1.0,
                },
                {
                    "key": "sports_min_probability",
                    "label": "Sports: Min Probability",
                    "type": "number",
                    "min": 0.5,
                    "max": 1,
                    "phase": "signal",
                },
                {
                    "key": "sports_max_days_to_resolution",
                    "label": "Sports: Max Days To Resolution",
                    "type": "number",
                    "min": 0,
                    "max": 2,
                    "phase": "signal",
                },
                {
                    "key": "sports_resolution_hold_minutes",
                    "label": "Sports: Resolution Hold (min)",
                    "type": "number",
                    "min": 0,
                    "max": 720,
                },
                {
                    "key": "resolution_hold_max_loss_pct",
                    "label": "Resolution Hold Max Loss (%)",
                    "type": "number",
                    "min": 5,
                    "max": 80,
                },
                {"key": "resolution_hold_enabled", "label": "Resolution Hold Enabled", "type": "boolean"},
                {
                    "key": "resolution_hold_minutes",
                    "label": "Resolution Hold (min)",
                    "type": "number",
                    "min": 0,
                    "max": 1440,
                },
                {
                    "key": "max_hold_minutes",
                    "label": "Max Hold (minutes)",
                    "type": "number",
                    "min": 0,
                    "max": 10080,
                },
                {
                    "key": "session_timeout_seconds",
                    "label": "Session Timeout (sec)",
                    "type": "number",
                    "min": 30,
                    "max": 600,
                },
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
                *news_edge_config_schema().get("param_fields", []),
            ]
        },
    ),
    # ── Crypto strategies ────────────────────────────────────
    # Three independent BTC/ETH strategies split out from the deprecated
    # ``btc_eth_highfreq`` (which was a multi-mode dispatcher). Each is a
    # standalone Python module, individually editable in the strategies UI,
    # with its own focused config blob (no shared scope dict).
    SystemOpportunityStrategySeed(
        slug="btc_eth_maker_quote",
        source_key="crypto",
        import_module="services.strategies.btc_eth_maker_quote",
        sort_order=190,
        config_schema={
            "param_fields": [
                {"key": "take_profit_pct", "label": "Take profit %", "type": "float", "default": 8.0},
                {"key": "stop_loss_pct", "label": "Stop loss %", "type": "float", "default": 4.0},
                {"key": "trailing_stop_pct", "label": "Trailing stop %", "type": "float", "default": 3.0},
                {
                    "key": "force_flatten_seconds_left",
                    "label": "Force-flatten seconds left",
                    "type": "float",
                    "default": 120.0,
                },
                {
                    "key": "opening_directional_buy_yes_enabled",
                    "label": "Allow directional BUY YES",
                    "type": "bool",
                    "default": True,
                },
                {
                    "key": "opening_directional_buy_no_enabled",
                    "label": "Allow directional BUY NO",
                    "type": "bool",
                    "default": True,
                },
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
            ]
        },
    ),
    SystemOpportunityStrategySeed(
        slug="btc_eth_directional_edge",
        source_key="crypto",
        import_module="services.strategies.btc_eth_directional_edge",
        sort_order=192,
        config_schema={
            "param_fields": [
                {"key": "take_profit_pct", "label": "Take profit %", "type": "float", "default": 8.0},
                {"key": "stop_loss_pct", "label": "Stop loss %", "type": "float", "default": 4.0},
                {"key": "trailing_stop_pct", "label": "Trailing stop %", "type": "float", "default": 3.0},
                {
                    "key": "force_flatten_seconds_left",
                    "label": "Force-flatten seconds left",
                    "type": "float",
                    "default": 120.0,
                },
                {
                    "key": "opening_directional_buy_yes_enabled",
                    "label": "Allow directional BUY YES",
                    "type": "bool",
                    "default": True,
                },
                {
                    "key": "opening_directional_buy_no_enabled",
                    "label": "Allow directional BUY NO",
                    "type": "bool",
                    "default": True,
                },
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
            ]
        },
    ),
    SystemOpportunityStrategySeed(
        slug="crypto_5m_midcycle",
        source_key="crypto",
        import_module="services.strategies.crypto_5m_midcycle",
        sort_order=193,
        config_schema={
            "param_fields": [
                {"key": "enabled", "label": "Enabled", "type": "boolean", "default": True, "phase": "signal"},
                {
                    "key": "assets",
                    "label": "Assets",
                    "type": "list",
                    "options": ["BTC", "ETH", "SOL", "XRP"],
                    "default": ["SOL", "XRP"],
                    "phase": "signal",
                },
                {
                    "key": "min_distance_bps",
                    "label": "Min |Distance from Reference| (bps)",
                    "type": "number",
                    "min": 0.0,
                    "max": 1000.0,
                    "default": 15.0,
                    "phase": "signal",
                },
                {
                    "key": "max_entry_price",
                    "label": "Max VWAP Entry Price",
                    "type": "number",
                    "min": 0.0,
                    "max": 1.0,
                    "default": 0.70,
                    "phase": "execution",
                },
                {
                    "key": "min_entry_price",
                    "label": "Min VWAP Entry Price",
                    "type": "number",
                    "min": 0.0,
                    "max": 1.0,
                    "default": 0.05,
                    "phase": "execution",
                },
                {
                    "key": "bet_size_usd",
                    "label": "Bet Size (USD)",
                    "type": "number",
                    "min": 1.0,
                    "max": 10000.0,
                    "default": 15.0,
                    "phase": "execution",
                },
                {
                    "key": "midcycle_seconds",
                    "label": "Midcycle Milestone (sec since cycle start)",
                    "type": "number",
                    "min": 1.0,
                    "max": 299.0,
                    "default": 150.0,
                    "phase": "signal",
                },
                {
                    "key": "min_seconds_to_resolution",
                    "label": "Min Seconds to Resolution",
                    "type": "number",
                    "min": 0.0,
                    "max": 300.0,
                    "default": 90.0,
                    "phase": "signal",
                },
                {
                    "key": "max_oracle_age_ms",
                    "label": "Max Oracle Age (ms)",
                    "type": "integer",
                    "min": 0,
                    "max": 60000,
                    "default": 5000,
                    "phase": "signal",
                },
            ]
        },
    ),
    SystemOpportunityStrategySeed(
        slug="crypto_digital_sigma_edge",
        source_key="crypto",
        import_module="services.strategies.crypto_digital_sigma_edge",
        sort_order=194,
        config_schema={
            "param_fields": [
                {"key": "enabled", "label": "Enabled", "type": "boolean", "default": True, "phase": "signal"},
                {
                    "key": "assets",
                    "label": "Assets",
                    "type": "list",
                    "options": ["BTC", "ETH", "SOL", "XRP"],
                    "default": ["BTC", "ETH", "SOL", "XRP"],
                    "phase": "signal",
                },
                {
                    "key": "timeframes",
                    "label": "Cycle timeframes",
                    "type": "list",
                    "options": ["5m", "15m", "1h", "4h"],
                    "default": ["5m", "15m"],
                    "phase": "signal",
                },
                {
                    "key": "vol_lookback_seconds",
                    "label": "Realized-vol lookback (sec)",
                    "type": "number",
                    "min": 120.0,
                    "max": 7200.0,
                    "default": 900.0,
                    "phase": "signal",
                },
                {
                    "key": "min_vol_intervals",
                    "label": "Min vol sample intervals",
                    "type": "integer",
                    "min": 4,
                    "max": 200,
                    "default": 12,
                    "phase": "signal",
                },
                {
                    "key": "min_history_span_seconds",
                    "label": "Min oracle-history span (sec)",
                    "type": "number",
                    "min": 60.0,
                    "max": 3600.0,
                    "default": 240.0,
                    "phase": "signal",
                },
                {
                    "key": "vol_floor",
                    "label": "Sigma floor (return/sec)",
                    "type": "number",
                    "min": 0.0,
                    "max": 0.001,
                    "default": 0.000002,
                    "phase": "signal",
                },
                {
                    "key": "vol_cap",
                    "label": "Sigma cap (return/sec)",
                    "type": "number",
                    "min": 0.0001,
                    "max": 0.05,
                    "default": 0.005,
                    "phase": "signal",
                },
                {
                    "key": "oracle_source_preference",
                    "label": "Oracle source (resolution truth)",
                    "type": "select",
                    "options": ["chainlink", "binance_direct", "binance"],
                    "default": "chainlink",
                    "phase": "signal",
                },
                {
                    "key": "max_oracle_age_ms",
                    "label": "Max oracle age (ms)",
                    "type": "integer",
                    "min": 250,
                    "max": 60000,
                    "default": 4000,
                    "phase": "signal",
                },
                {
                    "key": "min_cycle_fraction",
                    "label": "Min cycle fraction elapsed",
                    "type": "number",
                    "min": 0.0,
                    "max": 0.9,
                    "default": 0.15,
                    "phase": "signal",
                },
                {
                    "key": "max_cycle_fraction",
                    "label": "Max cycle fraction elapsed",
                    "type": "number",
                    "min": 0.1,
                    "max": 1.0,
                    "default": 0.85,
                    "phase": "signal",
                },
                {
                    "key": "min_seconds_left",
                    "label": "Min seconds to resolution",
                    "type": "number",
                    "min": 0.0,
                    "max": 600.0,
                    "default": 45.0,
                    "phase": "signal",
                },
                {
                    "key": "fee_buffer",
                    "label": "Fee buffer ($/share)",
                    "type": "number",
                    "min": 0.0,
                    "max": 0.10,
                    "default": 0.015,
                    "phase": "execution",
                },
                {
                    "key": "min_edge",
                    "label": "Min net edge ($/share)",
                    "type": "number",
                    "min": 0.0,
                    "max": 0.50,
                    "default": 0.045,
                    "phase": "signal",
                },
                {
                    "key": "down_side_slippage_buffer",
                    "label": "DOWN-side slippage buffer ($/share)",
                    "type": "number",
                    "min": 0.0,
                    "max": 0.10,
                    "default": 0.01,
                    "phase": "execution",
                },
                {
                    "key": "min_entry_price",
                    "label": "Min entry price",
                    "type": "number",
                    "min": 0.0,
                    "max": 1.0,
                    "default": 0.05,
                    "phase": "execution",
                },
                {
                    "key": "max_entry_price",
                    "label": "Max entry price",
                    "type": "number",
                    "min": 0.0,
                    "max": 1.0,
                    "default": 0.92,
                    "phase": "execution",
                },
                {
                    "key": "max_spread",
                    "label": "Max UP-book spread",
                    "type": "number",
                    "min": 0.0,
                    "max": 0.5,
                    "default": 0.08,
                    "phase": "signal",
                },
                {
                    "key": "min_liquidity",
                    "label": "Min market liquidity (USD)",
                    "type": "number",
                    "min": 0.0,
                    "max": 1000000.0,
                    "default": 500.0,
                    "phase": "signal",
                },
                {
                    "key": "max_recent_move_zscore",
                    "label": "Max |recent move z-score|",
                    "type": "number",
                    "min": 0.5,
                    "max": 20.0,
                    "default": 3.5,
                    "phase": "signal",
                },
                {
                    "key": "bet_size_usd",
                    "label": "Bet size (USD)",
                    "type": "number",
                    "min": 1.0,
                    "max": 10000.0,
                    "default": 15.0,
                    "phase": "execution",
                },
            ]
        },
    ),
    SystemOpportunityStrategySeed(
        slug="crypto_distance_edge",
        source_key="crypto",
        import_module="services.strategies.crypto_distance_edge",
        sort_order=195,
        config_schema={
            "param_fields": [
                {"key": "enabled", "label": "Enabled", "type": "boolean", "default": True, "phase": "signal"},
                {
                    "key": "assets",
                    "label": "Assets (distance tiers are USD — BTC-calibrated)",
                    "type": "list",
                    "options": ["BTC", "ETH", "SOL", "XRP"],
                    "default": ["BTC"],
                    "phase": "signal",
                },
                {
                    "key": "distance_cost_tiers",
                    "label": "Distance($)→Min-Cost(¢) Tiers",
                    "type": "json",
                    "default": [
                        {"distance_usd": 150.0, "min_cost_cents": 90.0},
                        {"distance_usd": 200.0, "min_cost_cents": 85.0},
                        {"distance_usd": 300.0, "min_cost_cents": 80.0},
                    ],
                    "phase": "signal",
                },
                {
                    "key": "max_cost_cents",
                    "label": "Max Cost (¢)",
                    "type": "number",
                    "min": 0.0,
                    "max": 100.0,
                    "default": 98.0,
                    "phase": "execution",
                },
                {
                    "key": "bet_size_usd",
                    "label": "Bet Size (USD)",
                    "type": "number",
                    "min": 1.0,
                    "max": 10000.0,
                    "default": 15.0,
                    "phase": "execution",
                },
                {
                    "key": "min_seconds_to_resolution",
                    "label": "Min Seconds to Resolution",
                    "type": "number",
                    "min": 0.0,
                    "max": 900.0,
                    "default": 10.0,
                    "phase": "signal",
                },
                {
                    "key": "max_oracle_age_ms",
                    "label": "Max Oracle Age (ms)",
                    "type": "integer",
                    "min": 0,
                    "max": 60000,
                    "default": 5000,
                    "phase": "signal",
                },
                {
                    "key": "edge_uplift_pct",
                    "label": "Assumed Edge Uplift (pct points over implied)",
                    "type": "number",
                    "min": 0.0,
                    "max": 50.0,
                    "default": 0.0,
                    "phase": "signal",
                },
            ]
        },
    ),
    SystemOpportunityStrategySeed(
        slug="btc_eth_convergence",
        source_key="crypto",
        import_module="services.strategies.btc_eth_convergence",
        sort_order=194,
        config_schema={
            "param_fields": [
                {"key": "take_profit_pct", "label": "Take profit %", "type": "float", "default": 8.0},
                {"key": "stop_loss_pct", "label": "Stop loss %", "type": "float", "default": 4.0},
                {"key": "trailing_stop_pct", "label": "Trailing stop %", "type": "float", "default": 3.0},
                {
                    "key": "force_flatten_seconds_left",
                    "label": "Force-flatten seconds left",
                    "type": "float",
                    "default": 120.0,
                },
                {
                    "key": "opening_directional_buy_yes_enabled",
                    "label": "Allow directional BUY YES",
                    "type": "bool",
                    "default": True,
                },
                {
                    "key": "opening_directional_buy_no_enabled",
                    "label": "Allow directional BUY NO",
                    "type": "bool",
                    "default": True,
                },
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
                {"key": "min_order_size_usd", "label": "Min Order Size (USD)", "type": "number", "min": 0},
                {"key": "base_size_usd", "label": "Base Size (USD)", "type": "number", "min": 0},
                {"key": "max_size_usd", "label": "Max Size (USD)", "type": "number", "min": 0},
                {"key": "take_profit_pct", "label": "Take Profit (%)", "type": "number", "min": 0},
                {"key": "stop_loss_pct", "label": "Stop Loss (%)", "type": "number", "min": 0},
                {"key": "max_hold_minutes", "label": "Max Hold (min)", "type": "number", "min": 0},
                {"key": "liquidity_cap_fraction", "label": "Liquidity Cap Fraction", "type": "number", "min": 0, "max": 1},
                {"key": "min_liquidity_usd", "label": "Min Liquidity (USD)", "type": "number", "min": 0},
                {"key": "max_entry_price", "label": "Max Entry Price", "type": "number", "min": 0, "max": 1},
                {"key": "max_markets_per_event", "label": "Max Markets per Event", "type": "integer", "min": 1},
            ]
        },
    ),
    SystemOpportunityStrategySeed(
        slug="crypto_entropy_maker",
        source_key="crypto",
        import_module="services.strategies.crypto_entropy_maker",
        sort_order=196,
        config_schema={
            "param_fields": [
                {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0, "max": 100},
                {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
                {"key": "min_entropy", "label": "Min Entropy", "type": "number", "min": 0, "max": 1},
                {"key": "min_spread_pct", "label": "Min Spread", "type": "number", "min": 0, "max": 1},
                {"key": "max_spread_pct", "label": "Max Spread", "type": "number", "min": 0, "max": 1},
                {
                    "key": "max_spread_widening_bps",
                    "label": "Max Spread Widening (bps)",
                    "type": "number",
                    "min": 0,
                },
                {"key": "max_cancel_rate_30s", "label": "Max Cancel Rate 30s", "type": "number", "min": 0, "max": 1},
                {
                    "key": "min_prior_peak_cancel_rate",
                    "label": "Min Prior Peak Cancel Rate",
                    "type": "number",
                    "min": 0,
                    "max": 1,
                },
                {"key": "min_cancel_drop", "label": "Min Cancel Drop", "type": "number", "min": 0, "max": 1},
                {
                    "key": "min_orderflow_alignment",
                    "label": "Min Orderflow Alignment",
                    "type": "number",
                    "min": 0,
                    "max": 1,
                },
                {
                    "key": "min_recent_move_zscore",
                    "label": "Min Recent Move Z-Score",
                    "type": "number",
                    "min": 0,
                    "max": 10,
                },
                {"key": "min_liquidity_usd", "label": "Min Liquidity (USD)", "type": "number", "min": 0},
                {
                    "key": "sizing_policy",
                    "label": "Sizing Policy",
                    "type": "enum",
                    "options": ["fixed", "linear", "adaptive", "kelly"],
                },
                {"key": "min_order_size_usd", "label": "Min Order Size (USD)", "type": "number", "min": 0},
                {"key": "base_size_usd", "label": "Base Size (USD)", "type": "number", "min": 0},
                {"key": "max_size_usd", "label": "Max Size (USD)", "type": "number", "min": 0},
                {"key": "take_profit_pct", "label": "Take Profit (%)", "type": "number", "min": 0},
                {"key": "stop_loss_pct", "label": "Stop Loss (%)", "type": "number", "min": 0},
                {"key": "max_hold_minutes", "label": "Max Hold (min)", "type": "number", "min": 0},
                {"key": "min_entry_price", "label": "Min Entry Price", "type": "number", "min": 0, "max": 1},
                {"key": "max_entry_price", "label": "Max Entry Price", "type": "number", "min": 0, "max": 1},
                {"key": "max_markets_per_event", "label": "Max Markets per Event", "type": "integer", "min": 1},
            ]
        },
    ),
    # ── Weather strategies ───────────────────────────────────
    SystemOpportunityStrategySeed(
        slug="weather_distribution",
        source_key="weather",
        import_module="services.strategies.weather_distribution",
        sort_order=202,
        config_schema={
            "param_fields": [
                {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0},
                {"key": "sigma_c", "label": "Sigma (C)", "type": "number", "min": 0.5, "max": 5.0},
                {"key": "min_ensemble_members", "label": "Min Ensemble Members", "type": "integer", "min": 1},
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
            ]
        },
    ),
    # ── Manual management strategies ───────────────────────────
    SystemOpportunityStrategySeed(
        slug="manual_wallet_position",
        source_key="manual",
        import_module="services.strategies.manual_manage_hold",
        sort_order=205,
        config_schema={
            "param_fields": [
                {"key": "min_hold_minutes", "label": "Min Hold (min)", "type": "number", "min": 0},
                {"key": "hard_stop_loss_pct", "label": "Hard Stop Loss (%)", "type": "number", "min": 0},
                {"key": "breakeven_arm_profit_pct", "label": "Breakeven Arm Profit (%)", "type": "number", "min": 0},
                {"key": "breakeven_buffer_pct", "label": "Breakeven Buffer (%)", "type": "number"},
                {
                    "key": "near_resolution_window_seconds",
                    "label": "Near Resolution Window (sec)",
                    "type": "number",
                    "min": 0,
                },
                {
                    "key": "near_resolution_stop_loss_pct",
                    "label": "Near Resolution Stop Loss (%)",
                    "type": "number",
                    "min": 0,
                },
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
                *StrategySDK.trader_filter_config_schema().get("param_fields", []),
            ]
        },
    ),
    SystemOpportunityStrategySeed(
        slug="traders_copy_trade",
        source_key="traders",
        import_module="services.strategies.traders_copy_trade",
        sort_order=211,
        config_schema={
            "param_fields": [
                *traders_copy_trade_config_schema().get("param_fields", []),
            ]
        },
    ),
    # ── Sports strategies ────────────────────────────────────
    SystemOpportunityStrategySeed(
        slug="sports_overreaction_fader",
        source_key="sports",
        import_module="services.strategies.sports_overreaction_fader",
        sort_order=300,
        config_schema={
            "param_fields": [
                {"key": "min_move_pct", "label": "Min Move (%)", "type": "number", "min": 1.0, "max": 50.0},
                {"key": "max_move_pct", "label": "Max Move (%)", "type": "number", "min": 2.0, "max": 80.0},
                {"key": "move_window_seconds", "label": "Move Window (sec)", "type": "number", "min": 30, "max": 3600},
                {"key": "min_favorite_prob", "label": "Min Favorite Prob", "type": "number", "min": 0.5, "max": 0.99},
                {"key": "max_favorite_prob", "label": "Max Favorite Prob", "type": "number", "min": 0.5, "max": 0.999},
                {"key": "min_liquidity", "label": "Min Liquidity (USD)", "type": "number", "min": 0},
                {"key": "max_spread_bps", "label": "Max Spread (bps)", "type": "number", "min": 0, "max": 5000},
                {"key": "reversion_fraction", "label": "Reversion Fraction", "type": "number", "min": 0.1, "max": 1.0},
                {"key": "min_reversion_edge", "label": "Min Reversion Edge", "type": "number", "min": 0.001, "max": 0.50},
                {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0, "max": 100},
                {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
                {"key": "max_risk_score", "label": "Max Risk Score", "type": "number", "min": 0, "max": 1},
                {"key": "take_profit_pct", "label": "Take Profit (%)", "type": "number", "min": 0, "max": 100},
                {"key": "stop_loss_pct", "label": "Stop Loss (%)", "type": "number", "min": 0, "max": 100},
                {"key": "max_hold_minutes", "label": "Max Hold (min)", "type": "number", "min": 1, "max": 10080},
            ]
        },
    ),
]


def build_system_opportunity_strategy_rows(*, now: datetime | None = None) -> list[dict]:
    ts = now or utcnow()
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
    """Insert any missing shipped strategies; never overwrite existing rows.

    Once a strategy row exists for a registered seed slug, this function
    leaves it alone — the user's edits to ``source_code``, ``config``,
    ``config_schema``, etc. are preserved across reboots. To get a
    fresh copy of the shipped template, call ``reset_strategy_to_factory``
    explicitly (wired to the ``POST /{id}/reset-to-factory`` endpoint).

    Tombstones are no longer consulted — deleted strategies stay deleted.
    Users who want a shipped template back can hit reset-to-factory or
    re-create manually.
    """
    rows = build_system_opportunity_strategy_rows()
    seed_by_slug = {row["slug"]: row for row in rows}
    existing = {
        plugin.slug: plugin
        for plugin in (
            (await session.execute(select(Strategy).where(Strategy.slug.in_(list(seed_by_slug.keys())))))
            .scalars()
            .all()
        )
    }

    inserted = 0
    for slug, row in seed_by_slug.items():
        if slug in existing:
            # Row exists — leave it alone, regardless of is_system.
            continue
        session.add(Strategy(**row))
        inserted += 1

    # Track removals separately so the return-value semantics survive.
    rewritten = 0

    # Disable old execution-only duplicate slugs that are now aliases on
    # unified entries, plus any previously removed strategies.
    _REMOVED_SLUGS = [
        "weather_bucket_edge",
        "correlation_arb",
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
        "weather_ensemble_edge",
        # C/D tier strategies removed in consolidation
        "entropy_arb",
        "must_happen",
        "mutually_exclusive",
        "contradiction",
        "event_driven",
        "crypto_micro_sniper",
        "crypto_oracle_lag_capture",
        "crypto_queue_hazard_flip",
        "crypto_cancel_cluster_reentry",
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
        "bayesian_cascade",
        "late_favorite_alpha",
        # NOTE: temporal_decay was previously folded into stat_arb but is now
        # back as a standalone strategy with its own performance tracking.
        "btc_5m_reversal_sniper",
        "btc_5m_threshold_flip",
        "crypto_twin_parallel",
        # Replaced by 3 standalone strategies: btc_eth_maker_quote,
        # btc_eth_directional_edge, btc_eth_convergence (each independently
        # user-editable). Existing trader rows pointing at "btc_eth_highfreq"
        # will need to be retargeted to one of the new slugs.
        "btc_eth_highfreq",
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


_LEGACY_HIGHFREQ_MODE_TO_SUCCESSOR: dict[str, str] = {
    "directional": "btc_eth_directional_edge",
    "maker_quote": "btc_eth_maker_quote",
    "convergence": "btc_eth_convergence",
    # "auto" was the legacy multi-mode dispatcher.  Without an explicit
    # signal mode we pick the directional successor — it's the most
    # general (oracle-driven, no orderflow / cluster gates), so the
    # closest behavioural match to "auto".
    "auto": "btc_eth_directional_edge",
    "": "btc_eth_directional_edge",
}

# All 3 successor slugs — used by the heal path to detect duplicate
# crypto source_configs left behind by an earlier broken migration
# revision that emitted a 1→3 fan-out.
_CRYPTO_HIGHFREQ_SUCCESSORS: frozenset[str] = frozenset(
    {
        "btc_eth_directional_edge",
        "btc_eth_maker_quote",
        "btc_eth_convergence",
    }
)


async def _migrate_legacy_trader_source_configs(session: AsyncSession) -> int:
    """Rewrite Trader.source_configs_json entries that still target the
    now-deleted ``btc_eth_highfreq`` slug to a single successor strategy.

    The old slug was a multi-mode dispatcher selected via
    ``strategy_params.mode`` (auto / directional / maker_quote /
    convergence).  We resolve the same mode to its standalone successor
    strategy so the user keeps the behaviour they actually configured.
    "auto" / missing mode maps to ``btc_eth_directional_edge`` — the
    most general successor.

    The rewrite is **1→1**, not 1→N.  ``_serialize_trader`` enforces a
    no-duplicate-source_key invariant, so emitting 3 configs all under
    ``source_key="crypto"`` would make ``list_traders`` raise and the
    trader list goes empty in the UI.

    Without this migration, a trader still pointing at
    ``btc_eth_highfreq`` keeps running but matches zero published
    signals (the ``intent_runtime`` filter rejects strategy_types not
    in the trader's source_configs), surfacing as "Fast cycle idle:
    no pending signals" in the trader UI even when the new strategies
    are firing.

    Idempotent — runs every startup and exits silently when there's
    nothing to migrate.  Returns the count of mutated trader rows.
    Users can switch to a different successor afterwards via the
    trader-edit UI.
    """
    # Local import to avoid widening this module's import surface.
    from models.database import Trader

    legacy_slug = "btc_eth_highfreq"

    rows = list((await session.execute(select(Trader))).scalars().all())
    mutated_count = 0
    for trader in rows:
        configs = trader.source_configs_json
        if not isinstance(configs, list):
            continue
        new_configs: list[Any] = []
        changed = False
        for cfg in configs:
            if not isinstance(cfg, dict):
                new_configs.append(cfg)
                continue
            slug = str(cfg.get("strategy_key") or "").strip().lower()
            if slug == legacy_slug:
                base_params = cfg.get("strategy_params") or {}
                if not isinstance(base_params, dict):
                    base_params = {}
                mode_key = str(base_params.get("mode") or "").strip().lower()
                successor_slug = _LEGACY_HIGHFREQ_MODE_TO_SUCCESSOR.get(
                    mode_key,
                    "btc_eth_directional_edge",
                )
                source_key = str(cfg.get("source_key") or "crypto").strip().lower()
                new_cfg = dict(cfg)
                new_cfg["strategy_key"] = successor_slug
                new_cfg["source_key"] = source_key
                # Preserve user-tuned params except the now-meaningless
                # ``mode`` selector — the successor is the mode.
                migrated_params = {k: v for k, v in base_params.items() if k != "mode"}
                new_cfg["strategy_params"] = migrated_params
                new_configs.append(new_cfg)
                changed = True
            else:
                new_configs.append(cfg)

        # Heal traders mangled by the previous broken revision of this
        # migration — that revision fanned each legacy config out to 3
        # successor configs all under source_key="crypto", which fails
        # the no-duplicate-source_key invariant in
        # ``_normalize_source_configs`` and makes ``list_traders`` raise
        # (the trader list goes empty in the UI).  Collapse any group
        # of >=2 crypto configs whose strategy_keys are all successor
        # slugs into the FIRST one — keeps the user's earliest choice.
        seen_crypto_successor_idx: int | None = None
        healed_configs: list[Any] = []
        for cfg in new_configs:
            if (
                isinstance(cfg, dict)
                and str(cfg.get("source_key") or "").strip().lower() == "crypto"
                and str(cfg.get("strategy_key") or "").strip().lower() in _CRYPTO_HIGHFREQ_SUCCESSORS
            ):
                if seen_crypto_successor_idx is None:
                    seen_crypto_successor_idx = len(healed_configs)
                    healed_configs.append(cfg)
                else:
                    # Drop the duplicate.
                    changed = True
                    continue
            else:
                healed_configs.append(cfg)
        new_configs = healed_configs

        if changed:
            trader.source_configs_json = new_configs
            try:
                trader.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            except Exception:
                pass
            mutated_count += 1

    if mutated_count > 0:
        await session.commit()
    return mutated_count


async def ensure_all_strategies_seeded(session: AsyncSession) -> dict:
    """Seed all strategies from the single unified catalog."""
    result = await ensure_system_opportunity_strategies_seeded(session)
    trader_configs_migrated = await _migrate_legacy_trader_source_configs(session)
    return {"seeded": result, "trader_configs_migrated": trader_configs_migrated}


async def reset_strategy_to_factory(session: AsyncSession, slug: str) -> dict:
    """Reset a strategy to its shipped seed definition.

    Works for any slug that has a registered seed in
    ``SYSTEM_OPPORTUNITY_STRATEGY_SEEDS`` — ``is_system`` is no longer
    a precondition. Returns ``{"status": "not_found", ...}`` when the
    slug has no shipped seed (i.e. user-authored strategies have nothing
    to revert to).
    """
    rows = build_system_opportunity_strategy_rows()
    seed_row = next((r for r in rows if r["slug"] == slug), None)
    if seed_row is None:
        return {"status": "not_found", "detail": f"No shipped seed for slug '{slug}'"}

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
    current.updated_at = utcnow()
    await session.commit()
    return {"status": "reset", "detail": f"Strategy '{slug}' reset to factory defaults"}
