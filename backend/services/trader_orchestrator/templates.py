from __future__ import annotations

from typing import Any


DEFAULT_GLOBAL_RISK = {
    "max_gross_exposure_usd": 5000.0,
    "max_daily_loss_usd": 500.0,
    "max_orders_per_cycle": 50,
}


TRADER_TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "crypto_15m",
        "name": "Crypto 15m Trader",
        "description": "Dedicated crypto execution on 15m cadence signals.",
        "source_configs": [
            {
                "source_key": "crypto",
                "strategy_key": "crypto_15m",
                "strategy_params": {
                    "min_edge_percent": 3.0,
                    "min_confidence": 0.45,
                    "base_size_usd": 25.0,
                },
            }
        ],
        "interval_seconds": 60,
        "risk_limits": {
            "max_open_orders": 8,
            "max_per_market_exposure_usd": 400.0,
        },
    },
    {
        "id": "news_reaction",
        "name": "News Trader",
        "description": "News-event reaction strategy on validated news intents.",
        "source_configs": [
            {
                "source_key": "news",
                "strategy_key": "news_reaction",
                "strategy_params": {
                    "min_edge_percent": 8.0,
                    "min_confidence": 0.55,
                    "base_size_usd": 20.0,
                },
            }
        ],
        "interval_seconds": 120,
        "risk_limits": {
            "max_open_orders": 6,
            "max_per_market_exposure_usd": 300.0,
        },
    },
    {
        "id": "opportunity_weather",
        "name": "General + Weather Trader",
        "description": "Scanner + weather executor with source-specific strategies.",
        "source_configs": [
            {
                "source_key": "scanner",
                "strategy_key": "opportunity_general",
                "strategy_params": {
                    "min_edge_percent": 4.0,
                    "min_confidence": 0.45,
                    "max_risk_score": 0.78,
                    "min_liquidity": 25.0,
                    "base_size_usd": 18.0,
                },
            },
            {
                "source_key": "weather",
                "strategy_key": "weather_consensus",
                "strategy_params": {
                    "min_edge_percent": 6.0,
                    "min_confidence": 0.58,
                    "min_model_agreement": 0.62,
                    "min_source_count": 2,
                    "max_source_spread_c": 4.0,
                    "base_size_usd": 14.0,
                },
            },
        ],
        "interval_seconds": 120,
        "risk_limits": {
            "max_open_orders": 10,
            "max_per_market_exposure_usd": 350.0,
        },
    },
    {
        "id": "traders_flow",
        "name": "Traders Flow",
        "description": "Confluence trader-flow strategy across tracked and pool scopes.",
        "source_configs": [
            {
                "source_key": "traders",
                "strategy_key": "traders_flow",
                "strategy_params": {
                    "min_edge_percent": 3.0,
                    "min_confidence": 0.5,
                    "min_confluence_strength": 0.55,
                    "base_size_usd": 18.0,
                },
                "traders_scope": {
                    "modes": ["tracked", "pool"],
                    "individual_wallets": [],
                    "group_ids": [],
                },
            }
        ],
        "interval_seconds": 90,
        "risk_limits": {
            "max_open_orders": 10,
            "max_per_market_exposure_usd": 350.0,
        },
    },
]


def get_template(template_id: str) -> dict[str, Any] | None:
    key = str(template_id or "").strip().lower()
    for template in TRADER_TEMPLATES:
        if template["id"] == key:
            return template
    return None
