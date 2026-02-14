from __future__ import annotations

from typing import Any

from services.trader_orchestrator.sources.registry import (
    list_source_adapters,
    list_source_aliases,
)
from services.trader_orchestrator.strategies import list_strategy_keys
from services.trader_orchestrator.templates import TRADER_TEMPLATES


_DEFAULT_METADATA = {
    "cadence_profile": "custom",
    "trading_window_utc": {"start": "00:00", "end": "23:59"},
    "tags": [],
    "notes": "",
    "resume_policy": "resume_full",
}

_SHARED_RISK_FIELDS: list[dict[str, Any]] = [
    {"key": "max_open_positions", "label": "Max Open Positions", "type": "integer", "min": 1, "max": 1000},
    {"key": "max_open_orders", "label": "Max Open Orders", "type": "integer", "min": 1, "max": 2000},
    {"key": "max_orders_per_cycle", "label": "Max Orders / Cycle", "type": "integer", "min": 1, "max": 1000},
    {"key": "max_trade_notional_usd", "label": "Max Trade Notional (USD)", "type": "number", "min": 1},
    {"key": "max_per_market_exposure_usd", "label": "Max Per-Market Exposure (USD)", "type": "number", "min": 1},
    {"key": "max_daily_loss_usd", "label": "Max Daily Loss (USD)", "type": "number", "min": 1},
    {"key": "cooldown_seconds", "label": "Cooldown (seconds)", "type": "integer", "min": 0},
]

_RUNTIME_FIELDS: list[dict[str, Any]] = [
    {
        "key": "resume_policy",
        "label": "Resume Policy",
        "type": "enum",
        "options": [
            {
                "value": "resume_full",
                "label": "Resume Full",
                "description": "Manage existing positions and continue processing new signals.",
            },
            {
                "value": "manage_only",
                "label": "Manage Existing Only",
                "description": "Manage/exit existing positions without opening new positions.",
            },
            {
                "value": "flatten_then_start",
                "label": "Flatten Then Start",
                "description": "Close existing positions first, then enable new entries.",
            },
        ],
    },
    {
        "key": "trading_window_utc",
        "label": "Trading Window (UTC)",
        "type": "object",
        "properties": [
            {"key": "start", "type": "time_hhmm"},
            {"key": "end", "type": "time_hhmm"},
        ],
    },
]

_STRATEGY_METADATA: dict[str, dict[str, Any]] = {
    "crypto_15m": {
        "label": "Crypto HF Trader",
        "description": "Dedicated crypto execution for short-horizon up/down markets.",
        "supported_sources": ["crypto"],
        "param_fields": [
            {"key": "strategy_mode", "label": "Strategy Mode", "type": "enum", "options": ["auto", "directional", "pure_arb", "rebalance"]},
            {"key": "target_assets", "label": "Target Assets", "type": "array[string]"},
            {"key": "target_timeframes", "label": "Target Timeframes", "type": "array[string]"},
            {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0},
            {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
            {"key": "base_size_usd", "label": "Base Size (USD)", "type": "number", "min": 1},
            {"key": "max_size_usd", "label": "Max Size (USD)", "type": "number", "min": 1},
        ],
    },
    "news_reaction": {
        "label": "News Trader",
        "description": "Executes high-conviction news/intelligence signals.",
        "supported_sources": ["news", "insider", "world_intelligence"],
        "param_fields": [
            {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0},
            {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
            {"key": "base_size_usd", "label": "Base Size (USD)", "type": "number", "min": 1},
        ],
    },
    "opportunity_weather": {
        "label": "General + Weather Trader",
        "description": "Executes scanner/weather opportunity signals with conservative thresholds.",
        "supported_sources": ["scanner", "weather", "news", "world_intelligence"],
        "param_fields": [
            {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0},
            {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
            {"key": "base_size_usd", "label": "Base Size (USD)", "type": "number", "min": 1},
        ],
    },
    "omni_aggressive": {
        "label": "Omni Aggressive Trader",
        "description": "Cross-source high-frequency strategy with configurable aggressiveness.",
        "supported_sources": ["scanner", "crypto", "news", "weather", "world_intelligence", "insider", "tracked_traders"],
        "param_fields": [
            {"key": "aggressiveness", "label": "Aggressiveness", "type": "number", "min": 0.1},
            {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0},
            {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
            {"key": "base_size_usd", "label": "Base Size (USD)", "type": "number", "min": 1},
            {"key": "max_size_usd", "label": "Max Size (USD)", "type": "number", "min": 1},
        ],
    },
}


def _template_by_strategy_key() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for template in TRADER_TEMPLATES:
        key = str(template.get("strategy_key") or "").strip().lower()
        if key:
            out[key] = template
    return out


def build_trader_config_schema() -> dict[str, Any]:
    adapters = list_source_adapters()
    source_aliases = list_source_aliases()
    aliases_by_canonical: dict[str, list[str]] = {}
    for alias, canonical in source_aliases.items():
        aliases_by_canonical.setdefault(canonical, []).append(alias)

    sources: list[dict[str, Any]] = [
        {
            "key": adapter.key,
            "label": adapter.label,
            "description": adapter.description,
            "domains": adapter.domains,
            "signal_types": adapter.signal_types,
            "aliases": sorted(aliases_by_canonical.get(adapter.key, [])),
        }
        for adapter in adapters
    ]

    templates_by_strategy = _template_by_strategy_key()
    strategy_items: list[dict[str, Any]] = []
    for strategy_key in list_strategy_keys():
        metadata = _STRATEGY_METADATA.get(strategy_key, {})
        template = templates_by_strategy.get(strategy_key, {})
        defaults = {
            "interval_seconds": int(template.get("interval_seconds") or 60),
            "sources": list(template.get("sources") or []),
            "params": dict(template.get("params") or {}),
            "risk_limits": dict(template.get("risk_limits") or {}),
            "metadata": dict(_DEFAULT_METADATA),
        }
        strategy_items.append(
            {
                "key": strategy_key,
                "label": metadata.get("label") or strategy_key,
                "description": metadata.get("description") or "",
                "supported_sources": list(metadata.get("supported_sources") or template.get("sources") or []),
                "defaults": defaults,
                "param_fields": list(metadata.get("param_fields") or []),
                "risk_fields": list(_SHARED_RISK_FIELDS),
                "metadata_fields": list(_RUNTIME_FIELDS),
            }
        )

    return {
        "version": "2026-02-14",
        "default_strategy_key": "crypto_15m",
        "sources": sources,
        "source_aliases": source_aliases,
        "strategies": strategy_items,
        "shared_risk_fields": list(_SHARED_RISK_FIELDS),
        "runtime_fields": list(_RUNTIME_FIELDS),
    }
