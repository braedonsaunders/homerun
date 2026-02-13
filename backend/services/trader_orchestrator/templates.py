from __future__ import annotations

from typing import Any


DEFAULT_GLOBAL_RISK = {
    "max_gross_exposure_usd": 5000.0,
    "max_daily_loss_usd": 500.0,
    "max_orders_per_cycle": 50,
}


TRADER_TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "btc_15m",
        "name": "Crypto HF Trader",
        "description": "Crypto directional high-frequency specialist (5m/15m/1h/4h).",
        "strategy_key": "crypto_15m",
        "sources": ["crypto"],
        "interval_seconds": 60,
        "params": {
            "min_edge_percent": 3.0,
            "min_confidence": 0.45,
            "base_size_usd": 25.0,
        },
        "risk_limits": {
            "max_open_orders": 8,
            "max_per_market_exposure_usd": 400.0,
        },
    },
    {
        "id": "news_reaction",
        "name": "News Trader",
        "description": "News and intelligence reaction trader.",
        "strategy_key": "news_reaction",
        "sources": ["news", "insider", "world_intelligence"],
        "interval_seconds": 120,
        "params": {
            "min_edge_percent": 8.0,
            "min_confidence": 0.55,
            "base_size_usd": 20.0,
        },
        "risk_limits": {
            "max_open_orders": 6,
            "max_per_market_exposure_usd": 300.0,
        },
    },
    {
        "id": "opportunity_weather",
        "name": "General + Weather Trader",
        "description": "General opportunity and weather signal executor.",
        "strategy_key": "opportunity_weather",
        "sources": ["scanner", "weather", "news"],
        "interval_seconds": 120,
        "params": {
            "min_edge_percent": 6.0,
            "min_confidence": 0.5,
            "base_size_usd": 18.0,
        },
        "risk_limits": {
            "max_open_orders": 10,
            "max_per_market_exposure_usd": 350.0,
        },
    },
    {
        "id": "omni_aggressive",
        "name": "Omni Aggressive Trader",
        "description": "Cross-source aggressive trader with tunable aggression.",
        "strategy_key": "omni_aggressive",
        "sources": [
            "scanner",
            "crypto",
            "news",
            "weather",
            "world_intelligence",
            "insider",
            "tracked_traders",
        ],
        "interval_seconds": 90,
        "params": {
            "min_edge_percent": 2.5,
            "min_confidence": 0.35,
            "base_size_usd": 20.0,
            "max_size_usd": 120.0,
            "aggressiveness": 1.7,
        },
        "risk_limits": {
            "max_open_orders": 14,
            "max_per_market_exposure_usd": 450.0,
        },
    },
]


def get_template(template_id: str) -> dict[str, Any] | None:
    key = str(template_id or "").strip().lower()
    for template in TRADER_TEMPLATES:
        if template["id"] == key:
            return template
    return None
