from __future__ import annotations

from typing import Any


DEFAULT_GLOBAL_RISK = {
    "max_gross_exposure_usd": 5000.0,
    "max_daily_loss_usd": 500.0,
    "max_orders_per_cycle": 50,
}


def _normalize_template(template: dict[str, Any]) -> dict[str, Any]:
    configs = []
    for source_config in template.get("source_configs", []) or []:
        if not isinstance(source_config, dict):
            continue
        normalized_source_config = dict(source_config)
        normalized_source_config["strategy_key"] = str(source_config.get("strategy_key") or "").strip().lower()
        configs.append(normalized_source_config)

    out = dict(template)
    out["source_configs"] = configs
    return out


TRADER_TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "btc_eth_highfreq",
        "name": "Crypto High-Frequency Trader",
        "description": "Dedicated BTC/ETH high-frequency crypto execution.",
        "source_configs": [
            {
                "source_key": "crypto",
                "strategy_key": "btc_eth_highfreq",
                "strategy_params": {
                    "min_edge_percent": 2.6,
                    "min_confidence": 0.48,
                    "base_size_usd": 20.0,
                    "max_size_usd": 150.0,
                    "max_open_order_seconds": 8.0,
                    "include_assets": ["BTC", "ETH"],
                    "exclude_assets": ["SOL", "XRP"],
                    "include_timeframes": ["15m"],
                    "exclude_timeframes": ["5m", "1h", "4h"],
                    "enabled_sub_strategies": [
                        "maker_quote",
                        "directional_edge",
                        "convergence",
                    ],
                    "directional_min_entry_price_floor": 0.25,
                    "stop_loss_pct": 3.0,
                    "stop_loss_policy": "always",
                    "immediate_stop_loss_pct": 2.0,
                    "preplace_take_profit_exit": False,
                },
            }
        ],
        "interval_seconds": 1,
        "risk_limits": {
            "max_open_orders": 8,
            "max_per_market_exposure_usd": 400.0,
        },
    },
    {
        "id": "news_edge",
        "name": "News Trader",
        "description": "News-event reaction strategy on validated news intents.",
        "source_configs": [
            {
                "source_key": "news",
                "strategy_key": "news_edge",
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
        "id": "scanner_weather",
        "name": "Scanner + Weather Trader",
        "description": "Scanner + weather executor with source-specific strategies.",
        "source_configs": [
            {
                "source_key": "scanner",
                "strategy_key": "basic",
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
                "strategy_key": "weather_distribution",
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
        "id": "flash_tape_ported",
        "name": "Flash Reversion Stack",
        "description": "Scanner execution tuned for flash-reversion and tail-carry opportunities.",
        "source_configs": [
            {
                "source_key": "scanner",
                "strategy_key": "flash_crash_reversion",
                "strategy_params": {
                    "min_edge_percent": 3.0,
                    "min_confidence": 0.4,
                    "max_risk_score": 0.8,
                    "min_liquidity": 1500.0,
                    "min_abs_move_5m": 1.5,
                    "sizing_policy": "kelly",
                    "kelly_fractional_scale": 0.5,
                    "base_size_usd": 16.0,
                },
            }
        ],
        "interval_seconds": 60,
        "risk_limits": {
            "max_open_orders": 8,
            "max_per_market_exposure_usd": 300.0,
        },
    },
    {
        "id": "traders_copy_trade",
        "name": "Traders Copy Trade",
        "description": "Real-time explicit copy trading strategy with wallet-scope controls.",
        "source_configs": [
            {
                "source_key": "traders",
                "strategy_key": "traders_copy_trade",
                "strategy_params": {
                    "min_confidence": 0.45,
                    "min_source_notional_usd": 15.0,
                    "max_signal_age_seconds": 600,
                    "copy_delay_seconds": 0,
                    "copy_buys": True,
                    "copy_sells": True,
                    "max_position_size": 500.0,
                    "proportional_sizing": True,
                    "proportional_multiplier": 1.0,
                    "base_size_usd": 20.0,
                    "max_size_usd": 1000.0,
                    "traders_scope": {
                        "modes": ["tracked", "pool"],
                        "individual_wallets": [],
                        "group_ids": [],
                    },
                },
            }
        ],
        "interval_seconds": 5,
        "risk_limits": {
            "max_open_orders": 30,
            "max_per_market_exposure_usd": 600.0,
        },
    },
]


def get_template(template_id: str) -> dict[str, Any] | None:
    key = str(template_id or "").strip().lower()
    for template in TRADER_TEMPLATES:
        if template["id"] == key:
            return _normalize_template(template)
    return None
