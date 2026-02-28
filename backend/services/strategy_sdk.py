"""
Strategy SDK — high-level helpers for custom strategy development.

This module provides a clean, stable API surface for strategy authors.
Import it in your strategy code with:

    from services.strategy_sdk import StrategySDK

All methods are designed to be safe and handle missing data gracefully.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import json
import logging
from pathlib import Path
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


class StrategySDK:
    """High-level helpers for strategy authors.

    All methods are static or class methods — no instantiation needed.

    Usage in opportunity strategies (detect method):
        from services.strategy_sdk import StrategySDK

        price = StrategySDK.get_live_price(market, prices)
        spread = StrategySDK.get_spread_bps(market, prices)

    Usage for LLM calls (async):
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            StrategySDK.ask_llm("Analyze this market situation...")
        )
    """

    TRADER_TIER_CANONICAL = ("low", "medium", "high", "extreme")
    TRADER_SIDE_CANONICAL = ("all", "buy", "sell")
    TRADER_SOURCE_SCOPE_CANONICAL = ("all", "tracked", "pool")
    TRADER_SCOPE_MODE_CANONICAL = ("tracked", "pool", "individual", "group")
    TRADER_SCHEDULE_DAY_CANONICAL = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
    TRADER_FILTER_DEFAULTS: dict[str, Any] = {
        "min_confidence": 0.45,
        "min_tier": "low",
        "min_wallet_count": 2,
        "max_entry_price": 0.85,
        "min_order_size_usd": 1.0,
        "paper_min_order_size_usd": 1.0,
        "live_min_order_size_usd": 1.0,
        "firehose_require_active_signal": True,
        "firehose_require_tradable_market": True,
        "firehose_exclude_crypto_markets": False,
        "firehose_require_qualified_source": True,
        "firehose_max_age_minutes": 720,
        "firehose_source_scope": "all",
        "firehose_side_filter": "all",
    }
    TRADER_FILTER_CONFIG_SCHEMA: dict[str, Any] = {
        "param_fields": [
            {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
            {
                "key": "min_tier",
                "label": "Min Tier",
                "type": "enum",
                "options": ["low", "medium", "high", "extreme"],
            },
            {"key": "min_wallet_count", "label": "Min Wallet Count", "type": "integer", "min": 1},
            {"key": "max_entry_price", "label": "Max Entry Price", "type": "number", "min": 0, "max": 1},
            {"key": "min_order_size_usd", "label": "Min Order Size (USD)", "type": "number", "min": 0.01},
            {"key": "paper_min_order_size_usd", "label": "Paper Min Order Size (USD)", "type": "number", "min": 0.01},
            {"key": "live_min_order_size_usd", "label": "Live Min Order Size (USD)", "type": "number", "min": 0.01},
            {"key": "firehose_require_active_signal", "label": "Require Active Signal", "type": "boolean"},
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
            {
                "key": "firehose_source_scope",
                "label": "Source Scope",
                "type": "enum",
                "options": ["all", "tracked", "pool"],
            },
            {
                "key": "firehose_side_filter",
                "label": "Side Filter",
                "type": "enum",
                "options": ["all", "buy", "sell"],
            },
        ]
    }
    TRADER_SCOPE_DEFAULTS: dict[str, Any] = {
        "modes": ["tracked", "pool"],
        "individual_wallets": [],
        "group_ids": [],
    }
    TRADER_SCOPE_FIELDS_SCHEMA: list[dict[str, Any]] = [
        {
            "key": "traders_scope",
            "label": "Wallet Scope",
            "type": "object",
            "properties": [
                {
                    "key": "modes",
                    "label": "Modes",
                    "type": "array[string]",
                    "options": ["tracked", "pool", "individual", "group"],
                    "required": True,
                },
                {
                    "key": "individual_wallets",
                    "label": "Individual Wallets",
                    "type": "array[string]",
                },
                {
                    "key": "group_ids",
                    "label": "Group IDs",
                    "type": "array[string]",
                },
            ],
        }
    ]
    TRADER_RUNTIME_DEFAULTS: dict[str, Any] = {
        "cadence_profile": "custom",
        "trading_schedule_utc": {
            "enabled": False,
            "days": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
            "start_time": "00:00",
            "end_time": "23:59",
            "start_date": None,
            "end_date": None,
            "end_at": None,
        },
        "tags": [],
        "notes": "",
        "resume_policy": "resume_full",
    }
    TRADER_RUNTIME_FIELDS_SCHEMA: list[dict[str, Any]] = [
        {
            "key": "resume_policy",
            "label": "Resume Policy",
            "type": "enum",
            "options": [
                {
                    "value": "resume_full",
                    "label": "Resume Full",
                },
                {
                    "value": "manage_only",
                    "label": "Manage Existing Only",
                },
                {
                    "value": "flatten_then_start",
                    "label": "Flatten Then Start",
                },
            ],
        },
        {
            "key": "trading_schedule_utc",
            "label": "Trading Schedule (UTC)",
            "type": "object",
            "properties": [
                {"key": "enabled", "label": "Enable Schedule", "type": "boolean"},
                {
                    "key": "days",
                    "label": "Days",
                    "type": "array[string]",
                    "options": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
                },
                {
                    "key": "start_time",
                    "label": "Start Time (UTC)",
                    "type": "string",
                    "format": "time",
                },
                {
                    "key": "end_time",
                    "label": "End Time (UTC)",
                    "type": "string",
                    "format": "time",
                },
                {
                    "key": "start_date",
                    "label": "Start Date (UTC)",
                    "type": "string",
                    "format": "date",
                },
                {
                    "key": "end_date",
                    "label": "End Date (UTC)",
                    "type": "string",
                    "format": "date",
                },
                {
                    "key": "end_at",
                    "label": "End At (UTC ISO)",
                    "type": "string",
                    "format": "date-time",
                    "placeholder": "2026-02-28T23:59:59Z",
                },
            ],
        },
        {"key": "cadence_profile", "label": "Cadence Profile", "type": "string"},
        {"key": "tags", "label": "Tags", "type": "array[string]"},
        {"key": "notes", "label": "Operator Notes", "type": "string"},
    ]
    TRADER_RISK_DEFAULTS: dict[str, Any] = {
        "max_orders_per_cycle": 6,
        "max_open_orders": 20,
        "max_open_positions": 12,
        "position_cap_scope": "market_direction",
        "max_position_notional_usd": 350.0,
        "max_gross_exposure_usd": 2000.0,
        "max_trade_notional_usd": 350.0,
        "max_daily_loss_usd": 300.0,
        "max_daily_spend_usd": 2000.0,
        "cooldown_seconds": 0,
        "order_ttl_seconds": 1200,
        "slippage_bps": 35.0,
        "max_spread_bps": 75.0,
        "retry_limit": 2,
        "retry_backoff_ms": 250,
        "allow_averaging": False,
        "use_dynamic_sizing": True,
        "halt_on_consecutive_losses": True,
        "max_consecutive_losses": 4,
        "circuit_breaker_drawdown_pct": 12.0,
        "portfolio": {
            "enabled": False,
            "target_utilization_pct": 100.0,
            "max_source_exposure_pct": 100.0,
            "min_order_notional_usd": 10.0,
        },
    }
    TRADER_RISK_FIELDS_SCHEMA: list[dict[str, Any]] = [
        {"key": "max_open_positions", "label": "Max Open Positions", "type": "integer", "min": 1, "max": 1000},
        {
            "key": "position_cap_scope",
            "label": "Position Cap Scope",
            "type": "enum",
            "options": ["market_direction", "market", "asset_timeframe"],
        },
        {"key": "max_open_orders", "label": "Max Open Orders", "type": "integer", "min": 1, "max": 2000},
        {"key": "max_orders_per_cycle", "label": "Max Orders / Cycle", "type": "integer", "min": 1, "max": 1000},
        {"key": "max_trade_notional_usd", "label": "Max Trade Notional (USD)", "type": "number", "min": 1},
        {"key": "max_position_notional_usd", "label": "Max Position Notional (USD)", "type": "number", "min": 1},
        {"key": "max_gross_exposure_usd", "label": "Max Gross Exposure (USD)", "type": "number", "min": 1},
        {"key": "max_daily_loss_usd", "label": "Max Daily Loss (USD)", "type": "number", "min": 1},
        {"key": "max_daily_spend_usd", "label": "Max Daily Spend (USD)", "type": "number", "min": 1},
        {"key": "cooldown_seconds", "label": "Cooldown (seconds)", "type": "integer", "min": 0},
        {"key": "order_ttl_seconds", "label": "Order TTL (seconds)", "type": "integer", "min": 1, "max": 86400},
        {"key": "slippage_bps", "label": "Slippage Guard (bps)", "type": "number", "min": 0, "max": 10000},
        {"key": "max_spread_bps", "label": "Max Spread (bps)", "type": "number", "min": 0, "max": 10000},
        {"key": "retry_limit", "label": "Retry Limit", "type": "integer", "min": 0, "max": 50},
        {"key": "retry_backoff_ms", "label": "Retry Backoff (ms)", "type": "integer", "min": 0, "max": 60000},
        {"key": "allow_averaging", "label": "Allow Averaging", "type": "boolean"},
        {"key": "use_dynamic_sizing", "label": "Dynamic Position Sizing", "type": "boolean"},
        {"key": "halt_on_consecutive_losses", "label": "Halt on Loss Streak", "type": "boolean"},
        {"key": "max_consecutive_losses", "label": "Max Consecutive Losses", "type": "integer", "min": 0, "max": 1000},
        {
            "key": "circuit_breaker_drawdown_pct",
            "label": "Circuit Breaker Drawdown (%)",
            "type": "number",
            "min": 0,
            "max": 100,
        },
        {
            "key": "portfolio",
            "label": "Portfolio Allocator",
            "type": "object",
            "properties": [
                {"key": "enabled", "label": "Enabled", "type": "boolean"},
                {
                    "key": "target_utilization_pct",
                    "label": "Target Gross Utilization (%)",
                    "type": "number",
                    "min": 1,
                    "max": 100,
                },
                {
                    "key": "max_source_exposure_pct",
                    "label": "Max Source Exposure (%)",
                    "type": "number",
                    "min": 1,
                    "max": 100,
                },
                {
                    "key": "min_order_notional_usd",
                    "label": "Min Allocated Order (USD)",
                    "type": "number",
                    "min": 1,
                },
            ],
        },
    ]
    TRADER_OPPORTUNITY_FILTER_TIER_CANONICAL = ("WATCH", "HIGH", "EXTREME")
    TRADER_OPPORTUNITY_FILTER_DEFAULTS: dict[str, Any] = {
        "source_filter": "all",
        "min_tier": "WATCH",
        "side_filter": "all",
        "confluence_limit": 50,
        "individual_trade_limit": 40,
        "individual_trade_min_confidence": 0.62,
        "individual_trade_max_age_minutes": 180,
    }
    TRADER_OPPORTUNITY_FILTER_CONFIG_SCHEMA: dict[str, Any] = {
        "param_fields": [
            {
                "key": "source_filter",
                "label": "Source Filter",
                "type": "enum",
                "options": ["all", "tracked", "pool"],
            },
            {
                "key": "min_tier",
                "label": "Min Confluence Tier",
                "type": "enum",
                "options": ["WATCH", "HIGH", "EXTREME"],
            },
            {
                "key": "side_filter",
                "label": "Side Filter",
                "type": "enum",
                "options": ["all", "buy", "sell"],
            },
            {
                "key": "confluence_limit",
                "label": "Confluence Limit",
                "type": "integer",
                "min": 1,
                "max": 200,
            },
            {
                "key": "individual_trade_limit",
                "label": "Individual Trade Limit",
                "type": "integer",
                "min": 1,
                "max": 500,
            },
            {
                "key": "individual_trade_min_confidence",
                "label": "Min Confidence",
                "type": "number",
                "min": 0,
                "max": 1,
            },
            {
                "key": "individual_trade_max_age_minutes",
                "label": "Max Age (minutes)",
                "type": "integer",
                "min": 1,
                "max": 1440,
            },
        ]
    }
    TRADERS_COPY_TRADE_DEFAULTS: dict[str, Any] = {
        "min_confidence": 0.45,
        "min_source_notional_usd": 10.0,
        "max_entry_price": 0.98,
        "max_signal_age_seconds": 900,
        "min_live_liquidity_usd": 150.0,
        "max_adverse_entry_drift_pct": 2.0,
        "copy_delay_seconds": 0,
        "copy_buys": True,
        "copy_sells": True,
        "max_position_size": 1000.0,
        "proportional_sizing": True,
        "proportional_multiplier": 1.0,
        "base_size_usd": 25.0,
        "max_size_usd": 1500.0,
        "traders_scope": {
            "modes": ["tracked", "pool"],
            "individual_wallets": [],
            "group_ids": [],
        },
    }
    TRADERS_COPY_TRADE_CONFIG_SCHEMA: dict[str, Any] = {
        "param_fields": [
            {
                "key": "min_confidence",
                "label": "Min Confidence",
                "type": "number",
                "min": 0,
                "max": 1,
            },
            {
                "key": "min_source_notional_usd",
                "label": "Min Source Notional (USD)",
                "type": "number",
                "min": 0,
            },
            {
                "key": "max_entry_price",
                "label": "Max Entry Price",
                "type": "number",
                "min": 0,
                "max": 1,
            },
            {
                "key": "max_signal_age_seconds",
                "label": "Max Signal Age (sec)",
                "type": "integer",
                "min": 1,
                "max": 3600,
            },
            {
                "key": "min_live_liquidity_usd",
                "label": "Min Live Liquidity (USD)",
                "type": "number",
                "min": 0,
            },
            {
                "key": "max_adverse_entry_drift_pct",
                "label": "Max Adverse Entry Drift (%)",
                "type": "number",
                "min": 0,
                "max": 100,
            },
            {
                "key": "copy_delay_seconds",
                "label": "Copy Delay (sec)",
                "type": "integer",
                "min": 0,
                "max": 300,
            },
            {"key": "copy_buys", "label": "Copy Buys", "type": "boolean"},
            {"key": "copy_sells", "label": "Copy Sells", "type": "boolean"},
            {
                "key": "max_position_size",
                "label": "Max Position Size (USD)",
                "type": "number",
                "min": 1,
                "max": 1000000,
            },
            {
                "key": "proportional_sizing",
                "label": "Proportional Sizing",
                "type": "boolean",
            },
            {
                "key": "proportional_multiplier",
                "label": "Proportional Multiplier",
                "type": "number",
                "min": 0.01,
                "max": 100,
            },
            {
                "key": "base_size_usd",
                "label": "Base Size (USD)",
                "type": "number",
                "min": 1,
                "max": 10000,
            },
            {
                "key": "max_size_usd",
                "label": "Max Size (USD)",
                "type": "number",
                "min": 1,
                "max": 50000,
            },
            {
                "key": "traders_scope",
                "label": "Wallet Scope",
                "type": "object",
                "properties": [
                    {
                        "key": "modes",
                        "label": "Modes",
                        "type": "array[string]",
                        "options": ["tracked", "pool", "individual", "group"],
                        "required": True,
                    },
                    {
                        "key": "individual_wallets",
                        "label": "Individual Wallets",
                        "type": "array[string]",
                    },
                    {
                        "key": "group_ids",
                        "label": "Group IDs",
                        "type": "array[string]",
                    },
                ],
            },
        ]
    }
    POOL_ELIGIBILITY_DEFAULTS: dict[str, Any] = {
        "target_pool_size": 500,
        "min_pool_size": 400,
        "max_pool_size": 600,
        "active_window_hours": 168,
        "inactive_rising_retention_hours": 720,
        "selection_score_quality_target_floor": 0.35,
        "max_hourly_replacement_rate": 0.15,
        "replacement_score_cutoff": 0.05,
        "max_cluster_share": 0.08,
        "high_conviction_threshold": 0.72,
        "insider_priority_threshold": 0.62,
        "min_eligible_trades": 25,
        "max_eligible_anomaly": 0.5,
        "core_min_win_rate": 0.60,
        "core_min_sharpe": 0.5,
        "core_min_profit_factor": 1.2,
        "rising_min_win_rate": 0.50,
        "slo_min_analyzed_pct": 95.0,
        "slo_min_profitable_pct": 80.0,
        "leaderboard_wallet_trade_sample": 160,
        "incremental_wallet_trade_sample": 80,
        "full_sweep_interval_seconds": 1800,
        "incremental_refresh_interval_seconds": 120,
        "activity_reconciliation_interval_seconds": 120,
        "pool_recompute_interval_seconds": 60,
    }
    POOL_ELIGIBILITY_CONFIG_SCHEMA: dict[str, Any] = {
        "param_fields": [
            {"key": "target_pool_size", "label": "Target Pool Size", "type": "integer", "min": 10, "max": 5000},
            {"key": "min_pool_size", "label": "Min Pool Size", "type": "integer", "min": 0, "max": 5000},
            {"key": "max_pool_size", "label": "Max Pool Size", "type": "integer", "min": 1, "max": 10000},
            {"key": "active_window_hours", "label": "Active Window (hrs)", "type": "integer", "min": 1, "max": 720},
            {
                "key": "inactive_rising_retention_hours",
                "label": "Rising Retention (hrs)",
                "type": "integer",
                "min": 0,
                "max": 8760,
            },
            {
                "key": "selection_score_quality_target_floor",
                "label": "Selection Score Floor",
                "type": "number",
                "min": 0,
                "max": 1,
            },
            {"key": "min_eligible_trades", "label": "Min Eligible Trades", "type": "integer", "min": 1, "max": 100000},
            {"key": "max_eligible_anomaly", "label": "Max Anomaly Score", "type": "number", "min": 0, "max": 5},
            {"key": "core_min_win_rate", "label": "Core Min Win Rate", "type": "number", "min": 0, "max": 1},
            {"key": "core_min_sharpe", "label": "Core Min Sharpe", "type": "number", "min": -10, "max": 20},
            {"key": "core_min_profit_factor", "label": "Core Min Profit Factor", "type": "number", "min": 0, "max": 20},
            {"key": "rising_min_win_rate", "label": "Rising Min Win Rate", "type": "number", "min": 0, "max": 1},
            {
                "key": "max_hourly_replacement_rate",
                "label": "Max Hourly Replacement Rate",
                "type": "number",
                "min": 0,
                "max": 1,
            },
            {"key": "max_cluster_share", "label": "Max Cluster Share", "type": "number", "min": 0.01, "max": 1},
        ]
    }
    NEWS_FILTER_DEFAULTS: dict[str, Any] = {
        "min_edge_percent": 5.0,
        "min_confidence": 0.45,
        "orchestrator_min_edge": 10.0,
        "require_verifier": True,
        "require_second_source": False,
        "min_supporting_articles": 2,
        "min_supporting_sources": 2,
        "max_signal_age_minutes": 60,
    }
    NEWS_FILTER_CONFIG_SCHEMA: dict[str, Any] = {
        "param_fields": [
            {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0, "max": 100},
            {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
            {
                "key": "orchestrator_min_edge",
                "label": "Min Intent Edge (%)",
                "type": "number",
                "min": 0,
                "max": 100,
            },
            {"key": "require_verifier", "label": "Require Verifier", "type": "boolean"},
            {"key": "require_second_source", "label": "Require Second Source", "type": "boolean"},
            {
                "key": "min_supporting_articles",
                "label": "Min Supporting Articles",
                "type": "integer",
                "min": 1,
                "max": 10,
            },
            {
                "key": "min_supporting_sources",
                "label": "Min Supporting Sources",
                "type": "integer",
                "min": 1,
                "max": 10,
            },
            {
                "key": "max_signal_age_minutes",
                "label": "Max Signal Age (Minutes)",
                "type": "integer",
                "min": 1,
                "max": 1440,
            },
        ]
    }
    CRYPTO_HF_SCOPE_CONFIG_SCHEMA: dict[str, Any] = {
        "param_fields": [
            {
                "key": "include_assets",
                "label": "Include Assets",
                "type": "list",
                "options": ["BTC", "ETH", "SOL", "XRP"],
            },
            {
                "key": "exclude_assets",
                "label": "Exclude Assets",
                "type": "list",
                "options": ["BTC", "ETH", "SOL", "XRP"],
            },
            {
                "key": "include_timeframes",
                "label": "Include Timeframes",
                "type": "list",
                "options": ["5m", "15m", "1h", "4h"],
            },
            {
                "key": "exclude_timeframes",
                "label": "Exclude Timeframes",
                "type": "list",
                "options": ["5m", "15m", "1h", "4h"],
            },
            {
                "key": "enabled_sub_strategies",
                "label": "Enabled Sub-Strategies",
                "type": "array[string]",
                "options": ["maker_quote", "directional_edge", "convergence"],
            },
            {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0, "max": 100},
            {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
            {"key": "max_risk_score", "label": "Max Risk Score", "type": "number", "min": 0, "max": 1},
            {"key": "base_size_usd", "label": "Base Size (USD)", "type": "number", "min": 1, "max": 1000000},
            {"key": "max_size_usd", "label": "Max Size (USD)", "type": "number", "min": 1, "max": 1000000},
            {"key": "live_window_required", "label": "Live Window Required", "type": "boolean"},
            {
                "key": "min_liquidity_usd",
                "label": "Min Liquidity (USD)",
                "type": "number",
                "min": 0,
                "max": 1000000,
            },
            {
                "key": "min_liquidity_usd_opening",
                "label": "Min Liquidity Opening (USD)",
                "type": "number",
                "min": 0,
                "max": 1000000,
            },
            {
                "key": "max_spread_pct",
                "label": "Max Spread (0-1)",
                "type": "number",
                "min": 0,
                "max": 1,
            },
            {
                "key": "max_signal_age_seconds",
                "label": "Max Signal Age (sec)",
                "type": "number",
                "min": 1,
                "max": 3600,
            },
            {
                "key": "max_open_order_seconds",
                "label": "Max Open Order Time (sec)",
                "type": "number",
                "min": 1,
                "max": 86400,
            },
            {
                "key": "max_signal_age_seconds_5m",
                "label": "Max Signal Age (5m sec)",
                "type": "number",
                "min": 0.1,
                "max": 300,
            },
            {
                "key": "max_signal_age_seconds_15m",
                "label": "Max Signal Age (15m sec)",
                "type": "number",
                "min": 0.1,
                "max": 900,
            },
            {
                "key": "max_signal_age_seconds_1h",
                "label": "Max Signal Age (1h sec)",
                "type": "number",
                "min": 0.1,
                "max": 3600,
            },
            {
                "key": "max_signal_age_seconds_4h",
                "label": "Max Signal Age (4h sec)",
                "type": "number",
                "min": 0.1,
                "max": 14400,
            },
            {
                "key": "max_market_data_age_ms",
                "label": "Max Market Data Age (ms)",
                "type": "integer",
                "min": 50,
                "max": 300000,
            },
            {
                "key": "max_market_data_age_ms_5m",
                "label": "Max Data Age 5m (ms)",
                "type": "integer",
                "min": 50,
                "max": 300000,
            },
            {
                "key": "max_market_data_age_ms_15m",
                "label": "Max Data Age 15m (ms)",
                "type": "integer",
                "min": 50,
                "max": 300000,
            },
            {
                "key": "max_market_data_age_ms_1h",
                "label": "Max Data Age 1h (ms)",
                "type": "integer",
                "min": 50,
                "max": 300000,
            },
            {
                "key": "max_market_data_age_ms_4h",
                "label": "Max Data Age 4h (ms)",
                "type": "integer",
                "min": 50,
                "max": 300000,
            },
            {"key": "enforce_market_data_freshness", "label": "Enforce Market Data Freshness", "type": "boolean"},
            {"key": "require_market_data_age_for_sources", "label": "Require Data Age For Sources", "type": "list"},
            {
                "key": "max_live_context_age_seconds",
                "label": "Max Live Context Age (sec)",
                "type": "number",
                "min": 0.1,
                "max": 60,
            },
            {
                "key": "max_oracle_age_seconds",
                "label": "Max Oracle Age (sec)",
                "type": "number",
                "min": 1,
                "max": 3600,
            },
            {
                "key": "max_oracle_age_ms",
                "label": "Max Oracle Age (ms)",
                "type": "number",
                "min": 100,
                "max": 3_600_000,
            },
            {"key": "require_oracle_for_directional", "label": "Require Oracle For Directional", "type": "boolean"},
            {
                "key": "oracle_source_policy",
                "label": "Oracle Source Policy",
                "type": "enum",
                "options": ["degrade", "hard_skip", "allow_fallback"],
            },
            {
                "key": "oracle_fallback_degrade_edge_multiplier",
                "label": "Oracle Fallback Edge Multiplier",
                "type": "number",
                "min": 1.0,
                "max": 3.0,
            },
            {
                "key": "oracle_fallback_degrade_confidence_multiplier",
                "label": "Oracle Fallback Confidence Multiplier",
                "type": "number",
                "min": 1.0,
                "max": 2.0,
            },
            {
                "key": "oracle_fallback_degrade_size_multiplier",
                "label": "Oracle Fallback Size Multiplier",
                "type": "number",
                "min": 0.05,
                "max": 1.0,
            },
            {
                "key": "min_edge_persistence_ms",
                "label": "Min Edge Persistence (ms)",
                "type": "integer",
                "min": 0,
                "max": 60000,
            },
            {
                "key": "max_recent_move_zscore_for_entry",
                "label": "Max Recent Move Z-Score",
                "type": "number",
                "min": 0.2,
                "max": 12.0,
            },
            {
                "key": "max_spread_widening_bps",
                "label": "Max Spread Widening (bps)",
                "type": "number",
                "min": 0,
                "max": 5000,
            },
            {
                "key": "max_orderbook_imbalance",
                "label": "Max Orderbook Imbalance",
                "type": "number",
                "min": 0.5,
                "max": 1.0,
            },
            {
                "key": "reentry_cooldown_seconds_per_market",
                "label": "Re-entry Cooldown (sec)",
                "type": "integer",
                "min": 0,
                "max": 86400,
            },
            {
                "key": "min_seconds_left_for_entry_5m",
                "label": "Min Seconds Left (5m)",
                "type": "number",
                "min": 0,
                "max": 300,
            },
            {
                "key": "min_seconds_left_for_entry_15m",
                "label": "Min Seconds Left (15m)",
                "type": "number",
                "min": 0,
                "max": 900,
            },
            {
                "key": "min_seconds_left_for_entry_1h",
                "label": "Min Seconds Left (1h)",
                "type": "number",
                "min": 0,
                "max": 3600,
            },
            {
                "key": "min_seconds_left_for_entry_4h",
                "label": "Min Seconds Left (4h)",
                "type": "number",
                "min": 0,
                "max": 14400,
            },
            {
                "key": "opening_directional_buy_yes_enabled",
                "label": "Opening Directional Buy-Yes Enabled",
                "type": "boolean",
            },
            {
                "key": "opening_directional_buy_yes_block_elapsed_pct",
                "label": "Opening Buy-Yes Block Window Default (Elapsed %)",
                "type": "number",
                "min": 0.0,
                "max": 1.0,
            },
            {
                "key": "opening_directional_buy_yes_block_elapsed_pct_5m",
                "label": "Opening Buy-Yes Block Window 5m (Elapsed %)",
                "type": "number",
                "min": 0.0,
                "max": 1.0,
            },
            {
                "key": "opening_directional_buy_yes_block_elapsed_pct_15m",
                "label": "Opening Buy-Yes Block Window 15m (Elapsed %)",
                "type": "number",
                "min": 0.0,
                "max": 1.0,
            },
            {
                "key": "opening_directional_buy_yes_block_elapsed_pct_1h",
                "label": "Opening Buy-Yes Block Window 1h (Elapsed %)",
                "type": "number",
                "min": 0.0,
                "max": 1.0,
            },
            {
                "key": "opening_directional_buy_yes_block_elapsed_pct_4h",
                "label": "Opening Buy-Yes Block Window 4h (Elapsed %)",
                "type": "number",
                "min": 0.0,
                "max": 1.0,
            },
            {
                "key": "opening_directional_buy_no_enabled",
                "label": "Opening Directional Buy-No Enabled",
                "type": "boolean",
            },
            {
                "key": "entry_executable_exit_ratio_floor",
                "label": "Entry Exitability Ratio Floor",
                "type": "number",
                "min": 0.01,
                "max": 1.0,
            },
            {
                "key": "entry_executable_exit_ratio_floor_closing",
                "label": "Entry Exitability Floor Closing",
                "type": "number",
                "min": 0.01,
                "max": 1.0,
            },
            {
                "key": "directional_min_entry_price_floor",
                "label": "Directional Min Entry Price",
                "type": "number",
                "min": 0.0,
                "max": 0.99,
            },
            {
                "key": "maker_min_entry_price_floor",
                "label": "Maker Min Entry Price",
                "type": "number",
                "min": 0.0,
                "max": 0.99,
            },
            {
                "key": "directional_max_entry_price_ceiling",
                "label": "Directional Max Entry Price",
                "type": "number",
                "min": 0.01,
                "max": 1.0,
            },
            {
                "key": "maker_max_entry_price_ceiling",
                "label": "Maker Max Entry Price",
                "type": "number",
                "min": 0.01,
                "max": 1.0,
            },
            {"key": "rapid_take_profit_pct", "label": "Rapid Take Profit (%)", "type": "number", "min": 0, "max": 100},
            {
                "key": "rapid_take_profit_pct_5m",
                "label": "Rapid Take Profit 5m (%)",
                "type": "number",
                "min": 0,
                "max": 100,
            },
            {
                "key": "rapid_take_profit_pct_15m",
                "label": "Rapid Take Profit 15m (%)",
                "type": "number",
                "min": 0,
                "max": 100,
            },
            {
                "key": "rapid_take_profit_pct_1h",
                "label": "Rapid Take Profit 1h (%)",
                "type": "number",
                "min": 0,
                "max": 100,
            },
            {
                "key": "rapid_take_profit_pct_4h",
                "label": "Rapid Take Profit 4h (%)",
                "type": "number",
                "min": 0,
                "max": 100,
            },
            {
                "key": "rapid_exit_window_minutes",
                "label": "Rapid Exit Window (min)",
                "type": "number",
                "min": 0,
                "max": 240,
            },
            {
                "key": "rapid_exit_window_minutes_5m",
                "label": "Rapid Exit Window 5m (min)",
                "type": "number",
                "min": 0,
                "max": 240,
            },
            {
                "key": "rapid_exit_window_minutes_15m",
                "label": "Rapid Exit Window 15m (min)",
                "type": "number",
                "min": 0,
                "max": 240,
            },
            {
                "key": "rapid_exit_window_minutes_1h",
                "label": "Rapid Exit Window 1h (min)",
                "type": "number",
                "min": 0,
                "max": 240,
            },
            {
                "key": "rapid_exit_window_minutes_4h",
                "label": "Rapid Exit Window 4h (min)",
                "type": "number",
                "min": 0,
                "max": 720,
            },
            {
                "key": "rapid_exit_min_increase_pct",
                "label": "Rapid Exit Min Increase (%)",
                "type": "number",
                "min": 0,
                "max": 100,
            },
            {
                "key": "rapid_exit_breakeven_buffer_pct",
                "label": "Rapid Exit Breakeven Buffer (%)",
                "type": "number",
                "min": 0,
                "max": 100,
            },
            {"key": "reverse_on_adverse_velocity_enabled", "label": "Enable Stop-And-Reverse", "type": "boolean"},
            {"key": "reverse_min_loss_pct", "label": "Reverse Min Loss (%)", "type": "number", "min": 0, "max": 100},
            {
                "key": "reverse_min_adverse_velocity_score",
                "label": "Reverse Min Adverse Velocity Score",
                "type": "number",
                "min": 0,
                "max": 1,
            },
            {
                "key": "reverse_flow_imbalance_threshold",
                "label": "Reverse Flow Imbalance Threshold",
                "type": "number",
                "min": -1,
                "max": 1,
            },
            {
                "key": "reverse_momentum_short_pct_threshold",
                "label": "Reverse Momentum Threshold (%)",
                "type": "number",
                "min": -100,
                "max": 100,
            },
            {
                "key": "reverse_min_seconds_left",
                "label": "Reverse Min Seconds Left",
                "type": "number",
                "min": 0,
                "max": 14400,
            },
            {
                "key": "reverse_min_price_headroom",
                "label": "Reverse Min Price Headroom",
                "type": "number",
                "min": 0,
                "max": 1,
            },
            {
                "key": "reverse_min_edge_percent",
                "label": "Reverse Min Edge (%)",
                "type": "number",
                "min": 0,
                "max": 100,
            },
            {"key": "reverse_confidence", "label": "Reverse Confidence", "type": "number", "min": 0, "max": 1},
            {
                "key": "reverse_size_multiplier",
                "label": "Reverse Size Multiplier",
                "type": "number",
                "min": 0.1,
                "max": 10,
            },
            {
                "key": "reverse_signal_ttl_seconds",
                "label": "Reverse Signal TTL (sec)",
                "type": "number",
                "min": 5,
                "max": 3600,
            },
            {
                "key": "reverse_cooldown_seconds",
                "label": "Reverse Cooldown (sec)",
                "type": "number",
                "min": 0,
                "max": 3600,
            },
            {
                "key": "reverse_max_reentries_per_position",
                "label": "Reverse Max Reentries / Position",
                "type": "integer",
                "min": 0,
                "max": 10,
            },
            {"key": "underwater_rebound_exit_enabled", "label": "Underwater Rebound Exit", "type": "boolean"},
            {
                "key": "underwater_dwell_minutes",
                "label": "Underwater Dwell (min)",
                "type": "number",
                "min": 0,
                "max": 720,
            },
            {
                "key": "underwater_dwell_minutes_5m",
                "label": "Underwater Dwell 5m (min)",
                "type": "number",
                "min": 0,
                "max": 720,
            },
            {
                "key": "underwater_dwell_minutes_15m",
                "label": "Underwater Dwell 15m (min)",
                "type": "number",
                "min": 0,
                "max": 720,
            },
            {
                "key": "underwater_dwell_minutes_1h",
                "label": "Underwater Dwell 1h (min)",
                "type": "number",
                "min": 0,
                "max": 720,
            },
            {
                "key": "underwater_dwell_minutes_4h",
                "label": "Underwater Dwell 4h (min)",
                "type": "number",
                "min": 0,
                "max": 1440,
            },
            {
                "key": "underwater_recovery_ratio_min",
                "label": "Min Recovery Ratio",
                "type": "number",
                "min": 0,
                "max": 1,
            },
            {
                "key": "underwater_recovery_ratio_min_5m",
                "label": "Min Recovery Ratio 5m",
                "type": "number",
                "min": 0,
                "max": 1,
            },
            {
                "key": "underwater_recovery_ratio_min_15m",
                "label": "Min Recovery Ratio 15m",
                "type": "number",
                "min": 0,
                "max": 1,
            },
            {
                "key": "underwater_recovery_ratio_min_1h",
                "label": "Min Recovery Ratio 1h",
                "type": "number",
                "min": 0,
                "max": 1,
            },
            {
                "key": "underwater_recovery_ratio_min_4h",
                "label": "Min Recovery Ratio 4h",
                "type": "number",
                "min": 0,
                "max": 1,
            },
            {
                "key": "underwater_rebound_pct_min",
                "label": "Min Rebound (%)",
                "type": "number",
                "min": 0,
                "max": 100,
            },
            {
                "key": "underwater_rebound_pct_min_5m",
                "label": "Min Rebound 5m (%)",
                "type": "number",
                "min": 0,
                "max": 100,
            },
            {
                "key": "underwater_rebound_pct_min_15m",
                "label": "Min Rebound 15m (%)",
                "type": "number",
                "min": 0,
                "max": 100,
            },
            {
                "key": "underwater_rebound_pct_min_1h",
                "label": "Min Rebound 1h (%)",
                "type": "number",
                "min": 0,
                "max": 100,
            },
            {
                "key": "underwater_rebound_pct_min_4h",
                "label": "Min Rebound 4h (%)",
                "type": "number",
                "min": 0,
                "max": 100,
            },
            {
                "key": "underwater_exit_fade_pct",
                "label": "Rebound Fade Trigger (%)",
                "type": "number",
                "min": 0,
                "max": 100,
            },
            {
                "key": "underwater_exit_fade_pct_5m",
                "label": "Rebound Fade 5m (%)",
                "type": "number",
                "min": 0,
                "max": 100,
            },
            {
                "key": "underwater_exit_fade_pct_15m",
                "label": "Rebound Fade 15m (%)",
                "type": "number",
                "min": 0,
                "max": 100,
            },
            {
                "key": "underwater_exit_fade_pct_1h",
                "label": "Rebound Fade 1h (%)",
                "type": "number",
                "min": 0,
                "max": 100,
            },
            {
                "key": "underwater_exit_fade_pct_4h",
                "label": "Rebound Fade 4h (%)",
                "type": "number",
                "min": 0,
                "max": 100,
            },
            {
                "key": "underwater_timeout_minutes",
                "label": "Underwater Timeout (min)",
                "type": "number",
                "min": 0,
                "max": 1440,
            },
            {
                "key": "underwater_timeout_minutes_5m",
                "label": "Underwater Timeout 5m (min)",
                "type": "number",
                "min": 0,
                "max": 1440,
            },
            {
                "key": "underwater_timeout_minutes_15m",
                "label": "Underwater Timeout 15m (min)",
                "type": "number",
                "min": 0,
                "max": 1440,
            },
            {
                "key": "underwater_timeout_minutes_1h",
                "label": "Underwater Timeout 1h (min)",
                "type": "number",
                "min": 0,
                "max": 1440,
            },
            {
                "key": "underwater_timeout_minutes_4h",
                "label": "Underwater Timeout 4h (min)",
                "type": "number",
                "min": 0,
                "max": 1440,
            },
            {
                "key": "underwater_timeout_loss_pct",
                "label": "Underwater Timeout Loss (%)",
                "type": "number",
                "min": 0,
                "max": 100,
            },
            {"key": "take_profit_pct", "label": "Take Profit (%)", "type": "number", "min": 0, "max": 100},
            {"key": "stop_loss_pct", "label": "Stop Loss (%)", "type": "number", "min": 0, "max": 100},
            {
                "key": "stop_loss_policy",
                "label": "Stop Loss Policy",
                "type": "enum",
                "options": ["always", "near_close_only", "volatility_adaptive"],
            },
            {
                "key": "stop_loss_policy_5m",
                "label": "Stop Loss Policy (5m)",
                "type": "enum",
                "options": ["always", "near_close_only", "volatility_adaptive"],
            },
            {
                "key": "stop_loss_policy_15m",
                "label": "Stop Loss Policy (15m)",
                "type": "enum",
                "options": ["always", "near_close_only", "volatility_adaptive"],
            },
            {
                "key": "stop_loss_policy_1h",
                "label": "Stop Loss Policy (1h)",
                "type": "enum",
                "options": ["always", "near_close_only", "volatility_adaptive"],
            },
            {
                "key": "stop_loss_policy_4h",
                "label": "Stop Loss Policy (4h)",
                "type": "enum",
                "options": ["always", "near_close_only", "volatility_adaptive"],
            },
            {
                "key": "stop_loss_activation_seconds",
                "label": "Stop Loss Arm Window (sec)",
                "type": "integer",
                "min": 0,
                "max": 86400,
            },
            {
                "key": "stop_loss_activation_seconds_5m",
                "label": "Stop Loss Arm (5m sec)",
                "type": "number",
                "min": 0,
                "max": 300,
            },
            {
                "key": "stop_loss_activation_seconds_15m",
                "label": "Stop Loss Arm (15m sec)",
                "type": "number",
                "min": 0,
                "max": 900,
            },
            {
                "key": "stop_loss_activation_seconds_1h",
                "label": "Stop Loss Arm (1h sec)",
                "type": "number",
                "min": 0,
                "max": 3600,
            },
            {
                "key": "stop_loss_activation_seconds_4h",
                "label": "Stop Loss Arm (4h sec)",
                "type": "number",
                "min": 0,
                "max": 14400,
            },
            {
                "key": "trailing_stop_pct",
                "label": "Trailing Stop (%)",
                "type": "number",
                "min": 0,
                "max": 100,
            },
            {
                "key": "trailing_stop_activation_profit_pct",
                "label": "Trailing Arm Profit (%)",
                "type": "number",
                "min": 0,
                "max": 100,
            },
            {
                "key": "trailing_stop_activation_profit_pct_5m",
                "label": "Trailing Arm Profit 5m (%)",
                "type": "number",
                "min": 0,
                "max": 100,
            },
            {
                "key": "trailing_stop_activation_profit_pct_15m",
                "label": "Trailing Arm Profit 15m (%)",
                "type": "number",
                "min": 0,
                "max": 100,
            },
            {
                "key": "trailing_stop_activation_profit_pct_1h",
                "label": "Trailing Arm Profit 1h (%)",
                "type": "number",
                "min": 0,
                "max": 100,
            },
            {
                "key": "trailing_stop_activation_profit_pct_4h",
                "label": "Trailing Arm Profit 4h (%)",
                "type": "number",
                "min": 0,
                "max": 100,
            },
            {"key": "min_hold_minutes", "label": "Min Hold (Minutes)", "type": "number", "min": 0, "max": 1440},
            {"key": "max_hold_minutes", "label": "Max Hold (Minutes)", "type": "number", "min": 1, "max": 1440},
            {
                "key": "force_flatten_seconds_left",
                "label": "Force Flatten Seconds Left",
                "type": "number",
                "min": 0,
                "max": 14400,
            },
            {
                "key": "force_flatten_seconds_left_5m",
                "label": "Force Flatten 5m Seconds Left",
                "type": "number",
                "min": 0,
                "max": 300,
            },
            {
                "key": "force_flatten_seconds_left_15m",
                "label": "Force Flatten 15m Seconds Left",
                "type": "number",
                "min": 0,
                "max": 900,
            },
            {
                "key": "force_flatten_seconds_left_1h",
                "label": "Force Flatten 1h Seconds Left",
                "type": "number",
                "min": 0,
                "max": 3600,
            },
            {
                "key": "force_flatten_seconds_left_4h",
                "label": "Force Flatten 4h Seconds Left",
                "type": "number",
                "min": 0,
                "max": 14400,
            },
            {
                "key": "force_flatten_max_profit_pct",
                "label": "Force Flatten Max Profit (%)",
                "type": "number",
                "min": 0,
                "max": 100,
            },
            {
                "key": "force_flatten_headroom_floor",
                "label": "Force Flatten Min Headroom Ratio",
                "type": "number",
                "min": 0,
                "max": 10,
            },
            {
                "key": "force_flatten_min_loss_pct",
                "label": "Force Flatten Max Loss (%)",
                "type": "number",
                "min": 0,
                "max": 100,
            },
            {
                "key": "resolution_risk_flatten_enabled",
                "label": "Resolution Risk Flatten Enabled",
                "type": "boolean",
            },
            {
                "key": "resolution_risk_seconds_left",
                "label": "Resolution Risk Seconds Left",
                "type": "number",
                "min": 0,
                "max": 14400,
            },
            {
                "key": "resolution_risk_seconds_left_5m",
                "label": "Resolution Risk 5m Seconds Left",
                "type": "number",
                "min": 0,
                "max": 300,
            },
            {
                "key": "resolution_risk_seconds_left_15m",
                "label": "Resolution Risk 15m Seconds Left",
                "type": "number",
                "min": 0,
                "max": 900,
            },
            {
                "key": "resolution_risk_seconds_left_1h",
                "label": "Resolution Risk 1h Seconds Left",
                "type": "number",
                "min": 0,
                "max": 3600,
            },
            {
                "key": "resolution_risk_seconds_left_4h",
                "label": "Resolution Risk 4h Seconds Left",
                "type": "number",
                "min": 0,
                "max": 14400,
            },
            {
                "key": "resolution_risk_max_profit_pct",
                "label": "Resolution Risk Max Profit (%)",
                "type": "number",
                "min": 0,
                "max": 100,
            },
            {
                "key": "resolution_risk_max_profit_pct_5m",
                "label": "Resolution Risk 5m Max Profit (%)",
                "type": "number",
                "min": 0,
                "max": 100,
            },
            {
                "key": "resolution_risk_min_loss_pct",
                "label": "Resolution Risk Max Loss (%)",
                "type": "number",
                "min": 0,
                "max": 100,
            },
            {
                "key": "resolution_risk_min_headroom_ratio",
                "label": "Resolution Risk Min Headroom Ratio",
                "type": "number",
                "min": 0,
                "max": 10,
            },
            {
                "key": "resolution_risk_disable_when_take_profit_armed",
                "label": "Resolution Risk Ignore Armed TP",
                "type": "boolean",
            },
            {"key": "resolve_only", "label": "Resolve Only Exit", "type": "boolean"},
            {"key": "close_on_inactive_market", "label": "Close On Inactive Market", "type": "boolean"},
            {"key": "preplace_take_profit_exit", "label": "Pre-Place Take Profit Exit", "type": "boolean"},
            {"key": "enforce_min_exit_notional", "label": "Enforce Min Exit Notional", "type": "boolean"},
        ]
    }
    STRATEGY_RETENTION_CONFIG_SCHEMA: dict[str, Any] = {
        "param_fields": [
            {"key": "max_opportunities", "label": "Max Opportunities", "type": "integer", "min": 0, "max": 5000},
            {
                "key": "retention_window",
                "label": "Retention Window (15m, 2d)",
                "type": "string",
            },
        ]
    }
    _DURATION_VALUE_RE = re.compile(r"^\s*(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>[a-z]+)?\s*$", re.IGNORECASE)
    _DURATION_UNITS_TO_MINUTES: dict[str, int] = {
        "m": 1,
        "min": 1,
        "mins": 1,
        "minute": 1,
        "minutes": 1,
        "h": 60,
        "hr": 60,
        "hrs": 60,
        "hour": 60,
        "hours": 60,
        "d": 1440,
        "day": 1440,
        "days": 1440,
        "w": 10080,
        "week": 10080,
        "weeks": 10080,
    }

    @staticmethod
    def load_json_from_disk(path: str, default: Any = None) -> Any:
        target = Path(path)
        if not target.exists():
            return default
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("StrategySDK.load_json_from_disk failed for %s", path, exc_info=exc)
            return default

    @staticmethod
    def save_json_to_disk(path: str, payload: Any) -> bool:
        target = Path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload), encoding="utf-8")
            return True
        except Exception as exc:
            logger.warning("StrategySDK.save_json_to_disk failed for %s", path, exc_info=exc)
            return False

    @staticmethod
    def _coerce_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return default

    @staticmethod
    def _coerce_float(value: Any, default: float, lo: float, hi: float) -> float:
        try:
            parsed = float(value)
        except Exception:
            parsed = default
        if parsed != parsed or parsed in (float("inf"), float("-inf")):
            parsed = default
        return max(lo, min(hi, parsed))

    @staticmethod
    def _coerce_int(value: Any, default: int, lo: int, hi: int) -> int:
        try:
            parsed = int(float(value))
        except Exception:
            parsed = default
        return max(lo, min(hi, parsed))

    @staticmethod
    def _coerce_str(value: Any, default: str) -> str:
        text = str(value or "").strip()
        return text if text else str(default)

    @staticmethod
    def _coerce_hhmm(value: Any, default: str) -> str:
        text = str(value or "").strip()
        if not text:
            return default
        parts = text.split(":")
        if len(parts) < 2:
            return default
        try:
            hour = int(parts[0])
            minute = int(parts[1])
        except Exception:
            return default
        if hour < 0 or hour > 23 or minute < 0 or minute > 59:
            return default
        return f"{hour:02d}:{minute:02d}"

    @staticmethod
    def _coerce_iso_date(value: Any) -> date | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            if "T" in text:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
                if parsed.tzinfo is not None:
                    parsed = parsed.astimezone(timezone.utc)
                return parsed.date()
            return date.fromisoformat(text)
        except Exception:
            return None

    @staticmethod
    def _coerce_iso_datetime_utc(value: Any) -> datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            if "T" in text:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            else:
                parsed = datetime.fromisoformat(f"{text}T00:00:00")
        except Exception:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _normalize_schedule_day(value: Any) -> str | None:
        token = str(value or "").strip().lower()
        if not token:
            return None
        aliases = {
            "monday": "mon",
            "mon": "mon",
            "tuesday": "tue",
            "tue": "tue",
            "wednesday": "wed",
            "wed": "wed",
            "thursday": "thu",
            "thu": "thu",
            "friday": "fri",
            "fri": "fri",
            "saturday": "sat",
            "sat": "sat",
            "sunday": "sun",
            "sun": "sun",
        }
        if token in aliases:
            return aliases[token]
        if len(token) >= 3:
            short = token[:3]
            if short in StrategySDK.TRADER_SCHEDULE_DAY_CANONICAL:
                return short
        return None

    @staticmethod
    def _normalize_schedule_days(value: Any) -> list[str]:
        if not isinstance(value, list):
            return list(StrategySDK.TRADER_SCHEDULE_DAY_CANONICAL)
        out: list[str] = []
        seen: set[str] = set()
        for raw in value:
            day = StrategySDK._normalize_schedule_day(raw)
            if day is None or day in seen:
                continue
            seen.add(day)
            out.append(day)
        return out or list(StrategySDK.TRADER_SCHEDULE_DAY_CANONICAL)

    @staticmethod
    def _coerce_string_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for raw in value:
            item = str(raw or "").strip()
            if not item:
                continue
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    @staticmethod
    def normalize_trader_tier(value: Any, default: str = "low") -> str:
        normalized = str(value or "").strip().lower()
        if normalized in StrategySDK.TRADER_TIER_CANONICAL:
            return normalized
        fallback = str(default or "low").strip().lower()
        if fallback in StrategySDK.TRADER_TIER_CANONICAL:
            return fallback
        return "low"

    @staticmethod
    def normalize_trader_side(value: Any, default: str = "all") -> str:
        normalized = str(value or "").strip().lower()
        if normalized in StrategySDK.TRADER_SIDE_CANONICAL:
            return normalized
        fallback = str(default or "all").strip().lower()
        if fallback in StrategySDK.TRADER_SIDE_CANONICAL:
            return fallback
        return "all"

    @staticmethod
    def opposite_direction(direction: Any, default: str = "") -> str:
        normalized = str(direction or "").strip().lower()
        if normalized == "buy_yes":
            return "buy_no"
        if normalized == "buy_no":
            return "buy_yes"
        fallback = str(default or "").strip().lower()
        if fallback in {"buy_yes", "buy_no"}:
            return fallback
        return ""

    @staticmethod
    def normalize_reverse_intent(
        value: Any,
        *,
        fallback_direction: str = "",
        default_signal_type: str = "strategy_reverse",
    ) -> dict[str, Any] | None:
        cfg = dict(value or {}) if isinstance(value, dict) else {}
        if not cfg:
            return None
        if not StrategySDK._coerce_bool(cfg.get("enabled"), True):
            return None

        direction = str(cfg.get("direction") or "").strip().lower()
        if direction not in {"buy_yes", "buy_no"}:
            direction = StrategySDK.opposite_direction(fallback_direction, default="")
        if direction not in {"buy_yes", "buy_no"}:
            return None

        signal_type = str(cfg.get("signal_type") or default_signal_type or "strategy_reverse").strip().lower()
        if not signal_type:
            signal_type = "strategy_reverse"

        out: dict[str, Any] = {
            "enabled": True,
            "direction": direction,
            "signal_type": signal_type,
            "size_multiplier": StrategySDK._coerce_float(cfg.get("size_multiplier"), 1.0, 0.05, 10.0),
            "min_seconds_left": StrategySDK._coerce_float(cfg.get("min_seconds_left"), 0.0, 0.0, 86_400.0),
            "min_price_headroom": StrategySDK._coerce_float(cfg.get("min_price_headroom"), 0.0, 0.0, 1.0),
            "expires_in_seconds": StrategySDK._coerce_float(cfg.get("expires_in_seconds"), 60.0, 5.0, 3_600.0),
            "max_reentries_per_position": StrategySDK._coerce_int(cfg.get("max_reentries_per_position"), 1, 0, 10),
            "cooldown_seconds": StrategySDK._coerce_float(cfg.get("cooldown_seconds"), 0.0, 0.0, 3_600.0),
        }

        raw_entry_price = cfg.get("entry_price")
        try:
            parsed_entry = float(raw_entry_price)
        except Exception:
            parsed_entry = None
        if parsed_entry is not None and 0.0 < parsed_entry < 1.0:
            out["entry_price"] = parsed_entry

        raw_size_usd = cfg.get("size_usd")
        try:
            parsed_size = float(raw_size_usd)
        except Exception:
            parsed_size = None
        if parsed_size is not None and parsed_size > 0.0:
            out["size_usd"] = parsed_size

        raw_edge = cfg.get("edge_percent")
        try:
            parsed_edge = float(raw_edge)
        except Exception:
            parsed_edge = None
        if parsed_edge is not None and parsed_edge >= 0.0:
            out["edge_percent"] = max(0.0, min(parsed_edge, 1_000.0))

        raw_conf = cfg.get("confidence")
        try:
            parsed_conf = float(raw_conf)
        except Exception:
            parsed_conf = None
        if parsed_conf is not None:
            out["confidence"] = max(0.0, min(parsed_conf, 1.0))

        raw_slippage = cfg.get("max_slippage_bps")
        try:
            parsed_slippage = float(raw_slippage)
        except Exception:
            parsed_slippage = None
        if parsed_slippage is not None and parsed_slippage >= 0.0:
            out["max_slippage_bps"] = min(parsed_slippage, 10_000.0)

        strategy_type = str(cfg.get("strategy_type") or "").strip().lower()
        if strategy_type:
            out["strategy_type"] = strategy_type

        token_id = str(cfg.get("token_id") or "").strip()
        if token_id:
            out["token_id"] = token_id

        reason = str(cfg.get("reason") or "").strip()
        if reason:
            out["reason"] = reason

        metadata = cfg.get("metadata")
        if isinstance(metadata, dict):
            out["metadata"] = dict(metadata)

        return out

    @staticmethod
    def build_reverse_intent(
        *,
        direction: str,
        signal_type: str = "strategy_reverse",
        enabled: bool = True,
        entry_price: float | None = None,
        edge_percent: float | None = None,
        confidence: float | None = None,
        size_multiplier: float = 1.0,
        size_usd: float | None = None,
        min_seconds_left: float = 0.0,
        min_price_headroom: float = 0.0,
        expires_in_seconds: float = 60.0,
        max_slippage_bps: float | None = None,
        strategy_type: str | None = None,
        token_id: str | None = None,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
        cooldown_seconds: float = 0.0,
        max_reentries_per_position: int = 1,
    ) -> dict[str, Any]:
        raw: dict[str, Any] = {
            "enabled": bool(enabled),
            "direction": direction,
            "signal_type": signal_type,
            "size_multiplier": size_multiplier,
            "size_usd": size_usd,
            "entry_price": entry_price,
            "edge_percent": edge_percent,
            "confidence": confidence,
            "min_seconds_left": min_seconds_left,
            "min_price_headroom": min_price_headroom,
            "expires_in_seconds": expires_in_seconds,
            "max_slippage_bps": max_slippage_bps,
            "strategy_type": strategy_type,
            "token_id": token_id,
            "reason": reason,
            "metadata": dict(metadata or {}),
            "cooldown_seconds": cooldown_seconds,
            "max_reentries_per_position": max_reentries_per_position,
        }
        normalized = StrategySDK.normalize_reverse_intent(
            raw, fallback_direction=direction, default_signal_type=signal_type
        )
        return dict(normalized or {})

    @staticmethod
    def normalize_trader_source_scope(value: Any, default: str = "all") -> str:
        normalized = str(value or "").strip().lower()
        if normalized in StrategySDK.TRADER_SOURCE_SCOPE_CANONICAL:
            return normalized
        fallback = str(default or "all").strip().lower()
        if fallback in StrategySDK.TRADER_SOURCE_SCOPE_CANONICAL:
            return fallback
        return "all"

    @staticmethod
    def derive_trader_source_flags(
        *,
        from_pool: Any = None,
        from_tracked_traders: Any = None,
        from_trader_groups: Any = None,
        qualified: Any = None,
        pool_wallets: Any = None,
        tracked_wallets: Any = None,
        group_wallets: Any = None,
    ) -> dict[str, bool]:
        def _count(value: Any) -> int:
            try:
                parsed = int(float(value))
            except Exception:
                return 0
            return max(0, parsed)

        pool_count = _count(pool_wallets)
        tracked_count = _count(tracked_wallets)
        group_count = _count(group_wallets)

        pool_flag = bool(from_pool) if from_pool is not None else pool_count > 0
        tracked_flag = bool(from_tracked_traders) if from_tracked_traders is not None else tracked_count > 0
        group_flag = bool(from_trader_groups) if from_trader_groups is not None else group_count > 0
        qualified_flag = bool(qualified) if qualified is not None else (pool_flag or tracked_flag or group_flag)

        return {
            "from_pool": pool_flag,
            "from_tracked_traders": tracked_flag,
            "from_trader_groups": group_flag,
            "qualified": qualified_flag,
        }

    @staticmethod
    def normalize_trader_source_flags(value: Any) -> dict[str, bool]:
        flags = value if isinstance(value, dict) else {}
        return StrategySDK.derive_trader_source_flags(
            from_pool=flags.get("from_pool"),
            from_tracked_traders=flags.get("from_tracked_traders"),
            from_trader_groups=flags.get("from_trader_groups"),
            qualified=flags.get("qualified"),
        )

    @staticmethod
    def infer_trader_side(signal: dict[str, Any]) -> str:
        raw_direction = StrategySDK.normalize_trader_side(signal.get("direction"), default="")
        if raw_direction:
            return raw_direction
        raw_outcome = StrategySDK.normalize_trader_side(signal.get("outcome"), default="")
        if raw_outcome:
            return raw_outcome
        raw_signal_type = StrategySDK.normalize_trader_side(signal.get("signal_type"), default="")
        if raw_signal_type:
            return raw_signal_type
        return "all"

    @staticmethod
    def normalize_trader_signal(signal: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(signal or {})
        raw_tier = normalized.get("tier")
        normalized["tier_raw"] = raw_tier
        normalized["tier"] = StrategySDK.normalize_trader_tier(raw_tier)
        normalized["side"] = StrategySDK.infer_trader_side(normalized)
        normalized["source_flags"] = StrategySDK.normalize_trader_source_flags(normalized.get("source_flags"))
        return normalized

    @staticmethod
    def normalize_trader_wallet(value: Any) -> str:
        return str(value or "").strip().lower()

    @staticmethod
    def extract_trader_signal_wallets(signal: Any) -> set[str]:
        payload = signal if isinstance(signal, dict) else getattr(signal, "payload_json", None)
        if not isinstance(payload, dict):
            payload = {}

        wallets: set[str] = set()
        for raw in payload.get("wallets") or []:
            normalized = StrategySDK.normalize_trader_wallet(raw)
            if normalized:
                wallets.add(normalized)
        for raw in payload.get("wallet_addresses") or []:
            normalized = StrategySDK.normalize_trader_wallet(raw)
            if normalized:
                wallets.add(normalized)
        for item in payload.get("top_wallets") or []:
            if not isinstance(item, dict):
                continue
            normalized = StrategySDK.normalize_trader_wallet(item.get("address"))
            if normalized:
                wallets.add(normalized)

        strategy_context = payload.get("strategy_context")
        if isinstance(strategy_context, dict):
            for raw in strategy_context.get("wallets") or []:
                normalized = StrategySDK.normalize_trader_wallet(raw)
                if normalized:
                    wallets.add(normalized)
            for raw in strategy_context.get("wallet_addresses") or []:
                normalized = StrategySDK.normalize_trader_wallet(raw)
                if normalized:
                    wallets.add(normalized)
            for item in strategy_context.get("top_wallets") or []:
                if not isinstance(item, dict):
                    continue
                normalized = StrategySDK.normalize_trader_wallet(item.get("address"))
                if normalized:
                    wallets.add(normalized)

            firehose = strategy_context.get("firehose")
            if isinstance(firehose, dict):
                for raw in firehose.get("wallets") or []:
                    normalized = StrategySDK.normalize_trader_wallet(raw)
                    if normalized:
                        wallets.add(normalized)
                for raw in firehose.get("wallet_addresses") or []:
                    normalized = StrategySDK.normalize_trader_wallet(raw)
                    if normalized:
                        wallets.add(normalized)
                for item in firehose.get("top_wallets") or []:
                    if not isinstance(item, dict):
                        continue
                    normalized = StrategySDK.normalize_trader_wallet(item.get("address"))
                    if normalized:
                        wallets.add(normalized)

        return wallets

    @staticmethod
    def build_trader_scope_runtime_context(
        traders_scope: Any,
        *,
        tracked_wallets: Any = None,
        pool_wallets: Any = None,
        group_wallets: Any = None,
    ) -> dict[str, Any]:
        cfg = StrategySDK.validate_trader_scope_config(traders_scope)

        def _normalize_wallet_set(values: Any) -> set[str]:
            if not isinstance(values, (list, tuple, set)):
                return set()
            out: set[str] = set()
            for raw in values:
                wallet = StrategySDK.normalize_trader_wallet(raw)
                if wallet:
                    out.add(wallet)
            return out

        return {
            "modes": {str(mode or "").strip().lower() for mode in (cfg.get("modes") or []) if str(mode or "").strip()},
            "individual_wallets": _normalize_wallet_set(cfg.get("individual_wallets") or []),
            "group_ids": [
                str(group_id or "").strip() for group_id in (cfg.get("group_ids") or []) if str(group_id or "").strip()
            ],
            "tracked_wallets": _normalize_wallet_set(tracked_wallets),
            "pool_wallets": _normalize_wallet_set(pool_wallets),
            "group_wallets": _normalize_wallet_set(group_wallets),
        }

    @staticmethod
    def match_trader_signal_scope(signal: Any, scope_context: Any) -> tuple[bool, dict[str, Any]]:
        context = scope_context if isinstance(scope_context, dict) else {}
        wallets = StrategySDK.extract_trader_signal_wallets(signal)
        modes = {str(mode or "").strip().lower() for mode in (context.get("modes") or set()) if str(mode or "").strip()}
        matched_modes: list[str] = []

        if "tracked" in modes and wallets.intersection(context.get("tracked_wallets") or set()):
            matched_modes.append("tracked")
        if "pool" in modes and wallets.intersection(context.get("pool_wallets") or set()):
            matched_modes.append("pool")
        if "individual" in modes and wallets.intersection(context.get("individual_wallets") or set()):
            matched_modes.append("individual")
        if "group" in modes and wallets.intersection(context.get("group_wallets") or set()):
            matched_modes.append("group")

        payload = {
            "signal_wallets": sorted(wallets),
            "selected_modes": sorted(modes),
            "matched_modes": sorted(matched_modes),
        }
        return bool(matched_modes), payload

    @staticmethod
    def trader_filter_defaults() -> dict[str, Any]:
        return dict(StrategySDK.TRADER_FILTER_DEFAULTS)

    @staticmethod
    def trader_filter_config_schema() -> dict[str, Any]:
        base_schema = dict(StrategySDK.TRADER_FILTER_CONFIG_SCHEMA)
        fields = [dict(field) for field in list(base_schema.get("param_fields") or []) if isinstance(field, dict)]
        existing = {str(field.get("key") or "").strip() for field in fields}
        for field in StrategySDK.trader_scope_fields_schema():
            key = str(field.get("key") or "").strip()
            if not key or key in existing:
                continue
            fields.append(dict(field))
            existing.add(key)
        base_schema["param_fields"] = fields
        return base_schema

    @staticmethod
    def trader_scope_defaults() -> dict[str, Any]:
        return dict(StrategySDK.TRADER_SCOPE_DEFAULTS)

    @staticmethod
    def trader_scope_fields_schema() -> list[dict[str, Any]]:
        return [dict(field) for field in StrategySDK.TRADER_SCOPE_FIELDS_SCHEMA]

    @staticmethod
    def trader_runtime_defaults() -> dict[str, Any]:
        return dict(StrategySDK.TRADER_RUNTIME_DEFAULTS)

    @staticmethod
    def trader_runtime_fields_schema() -> list[dict[str, Any]]:
        return [dict(field) for field in StrategySDK.TRADER_RUNTIME_FIELDS_SCHEMA]

    @staticmethod
    def trader_risk_defaults() -> dict[str, Any]:
        return dict(StrategySDK.TRADER_RISK_DEFAULTS)

    @staticmethod
    def trader_risk_fields_schema() -> list[dict[str, Any]]:
        return [dict(field) for field in StrategySDK.TRADER_RISK_FIELDS_SCHEMA]

    @staticmethod
    def trader_opportunity_filter_defaults() -> dict[str, Any]:
        return dict(StrategySDK.TRADER_OPPORTUNITY_FILTER_DEFAULTS)

    @staticmethod
    def trader_opportunity_filter_config_schema() -> dict[str, Any]:
        return dict(StrategySDK.TRADER_OPPORTUNITY_FILTER_CONFIG_SCHEMA)

    @staticmethod
    def traders_copy_trade_defaults() -> dict[str, Any]:
        return dict(StrategySDK.TRADERS_COPY_TRADE_DEFAULTS)

    @staticmethod
    def traders_copy_trade_config_schema() -> dict[str, Any]:
        return dict(StrategySDK.TRADERS_COPY_TRADE_CONFIG_SCHEMA)

    @staticmethod
    def pool_eligibility_defaults() -> dict[str, Any]:
        return dict(StrategySDK.POOL_ELIGIBILITY_DEFAULTS)

    @staticmethod
    def pool_eligibility_config_schema() -> dict[str, Any]:
        return dict(StrategySDK.POOL_ELIGIBILITY_CONFIG_SCHEMA)

    @staticmethod
    def validate_pool_eligibility_config(config: Any) -> dict[str, Any]:
        cfg = dict(StrategySDK.POOL_ELIGIBILITY_DEFAULTS)
        if isinstance(config, dict):
            cfg.update({str(k): v for k, v in config.items() if str(k) != "_schema"})

        cfg["target_pool_size"] = StrategySDK._coerce_int(cfg.get("target_pool_size"), 500, 10, 5000)
        cfg["min_pool_size"] = StrategySDK._coerce_int(cfg.get("min_pool_size"), 400, 0, 5000)
        cfg["max_pool_size"] = StrategySDK._coerce_int(cfg.get("max_pool_size"), 600, 1, 10000)
        cfg["active_window_hours"] = StrategySDK._coerce_int(cfg.get("active_window_hours"), 168, 1, 720)
        cfg["inactive_rising_retention_hours"] = StrategySDK._coerce_int(
            cfg.get("inactive_rising_retention_hours"), 720, 0, 8760
        )
        cfg["selection_score_quality_target_floor"] = StrategySDK._coerce_float(
            cfg.get("selection_score_quality_target_floor"), 0.35, 0.0, 1.0
        )
        cfg["max_hourly_replacement_rate"] = StrategySDK._coerce_float(
            cfg.get("max_hourly_replacement_rate"), 0.15, 0.0, 1.0
        )
        cfg["replacement_score_cutoff"] = StrategySDK._coerce_float(cfg.get("replacement_score_cutoff"), 0.05, 0.0, 1.0)
        cfg["max_cluster_share"] = StrategySDK._coerce_float(cfg.get("max_cluster_share"), 0.08, 0.01, 1.0)
        cfg["high_conviction_threshold"] = StrategySDK._coerce_float(
            cfg.get("high_conviction_threshold"), 0.72, 0.0, 1.0
        )
        cfg["insider_priority_threshold"] = StrategySDK._coerce_float(
            cfg.get("insider_priority_threshold"), 0.62, 0.0, 1.0
        )
        cfg["min_eligible_trades"] = StrategySDK._coerce_int(cfg.get("min_eligible_trades"), 25, 1, 100000)
        cfg["max_eligible_anomaly"] = StrategySDK._coerce_float(cfg.get("max_eligible_anomaly"), 0.5, 0.0, 5.0)
        cfg["core_min_win_rate"] = StrategySDK._coerce_float(cfg.get("core_min_win_rate"), 0.60, 0.0, 1.0)
        cfg["core_min_sharpe"] = StrategySDK._coerce_float(cfg.get("core_min_sharpe"), 0.5, -10.0, 20.0)
        cfg["core_min_profit_factor"] = StrategySDK._coerce_float(cfg.get("core_min_profit_factor"), 1.2, 0.0, 20.0)
        cfg["rising_min_win_rate"] = StrategySDK._coerce_float(cfg.get("rising_min_win_rate"), 0.50, 0.0, 1.0)
        cfg["slo_min_analyzed_pct"] = StrategySDK._coerce_float(cfg.get("slo_min_analyzed_pct"), 95.0, 0.0, 100.0)
        cfg["slo_min_profitable_pct"] = StrategySDK._coerce_float(cfg.get("slo_min_profitable_pct"), 80.0, 0.0, 100.0)
        cfg["leaderboard_wallet_trade_sample"] = StrategySDK._coerce_int(
            cfg.get("leaderboard_wallet_trade_sample"), 160, 1, 5000
        )
        cfg["incremental_wallet_trade_sample"] = StrategySDK._coerce_int(
            cfg.get("incremental_wallet_trade_sample"), 80, 1, 5000
        )
        cfg["full_sweep_interval_seconds"] = StrategySDK._coerce_int(
            cfg.get("full_sweep_interval_seconds"), 1800, 10, 86400
        )
        cfg["incremental_refresh_interval_seconds"] = StrategySDK._coerce_int(
            cfg.get("incremental_refresh_interval_seconds"), 120, 10, 86400
        )
        cfg["activity_reconciliation_interval_seconds"] = StrategySDK._coerce_int(
            cfg.get("activity_reconciliation_interval_seconds"), 120, 10, 86400
        )
        cfg["pool_recompute_interval_seconds"] = StrategySDK._coerce_int(
            cfg.get("pool_recompute_interval_seconds"), 60, 10, 86400
        )
        return cfg

    @staticmethod
    def news_filter_defaults() -> dict[str, Any]:
        return dict(StrategySDK.NEWS_FILTER_DEFAULTS)

    @staticmethod
    def news_filter_config_schema() -> dict[str, Any]:
        return dict(StrategySDK.NEWS_FILTER_CONFIG_SCHEMA)

    @staticmethod
    def crypto_highfreq_scope_defaults() -> dict[str, Any]:
        try:
            from services.strategies.btc_eth_highfreq import crypto_highfreq_scope_defaults as _strategy_defaults

            defaults = _strategy_defaults()
            if isinstance(defaults, dict):
                return dict(defaults)
        except Exception:
            pass
        return {}

    @staticmethod
    def crypto_highfreq_scope_config_schema() -> dict[str, Any]:
        return dict(StrategySDK.CRYPTO_HF_SCOPE_CONFIG_SCHEMA)

    @staticmethod
    def should_preplace_take_profit_exit(
        params: Any,
        *,
        default_enabled: bool = False,
    ) -> bool:
        cfg = params if isinstance(params, dict) else {}
        raw = (
            cfg.get("live_preplace_take_profit_exit")
            if "live_preplace_take_profit_exit" in cfg
            else cfg.get("preplace_take_profit_exit")
        )
        return StrategySDK._coerce_bool(raw, default_enabled)

    @staticmethod
    def resolve_open_order_timeout_seconds(
        params: Any,
        *,
        default_seconds: Optional[float] = None,
        min_seconds: float = 1.0,
        max_seconds: float = 86_400.0,
    ) -> Optional[float]:
        cfg = params if isinstance(params, dict) else {}
        selected: Any = None
        selected_set = False
        for key in (
            "max_open_order_seconds",
            "open_order_timeout_seconds",
            "max_order_open_seconds",
            "order_timeout_seconds",
            "order_ttl_seconds",
        ):
            if key in cfg:
                selected = cfg.get(key)
                selected_set = True
                break

        def _parse_positive(value: Any) -> Optional[float]:
            if value is None or isinstance(value, bool):
                return None
            try:
                parsed = float(value)
            except Exception:
                return None
            if parsed <= 0.0:
                return None
            return parsed

        parsed = _parse_positive(selected if selected_set else default_seconds)
        if parsed is None and selected_set and default_seconds is not None:
            parsed = _parse_positive(default_seconds)
        if parsed is None:
            return None
        lower = max(0.1, float(min_seconds))
        upper = max(lower, float(max_seconds))
        return max(lower, min(float(parsed), upper))

    @staticmethod
    def resolve_min_order_size_usd(
        params: Any,
        *,
        mode: Any = None,
        fallback: float = 1.0,
        min_value: float = 0.01,
        max_value: float = 10_000_000.0,
    ) -> float:
        cfg = params if isinstance(params, dict) else {}
        mode_key = str(mode or "").strip().lower()

        candidate_keys: list[str] = []
        if mode_key:
            candidate_keys.extend(
                (
                    f"{mode_key}_min_order_size_usd",
                    f"{mode_key}_min_order_notional_usd",
                )
            )
        candidate_keys.extend(
            (
                "min_order_size_usd",
                "min_order_notional_usd",
                "entry_exitability_min_order_size_usd",
                "global_min_order_size_usd",
            )
        )

        for key in candidate_keys:
            if key not in cfg:
                continue
            return float(StrategySDK._coerce_float(cfg.get(key), fallback, min_value, max_value))

        portfolio_cfg = cfg.get("portfolio")
        if isinstance(portfolio_cfg, dict):
            for key in ("min_order_size_usd", "min_order_notional_usd"):
                if key not in portfolio_cfg:
                    continue
                return float(StrategySDK._coerce_float(portfolio_cfg.get(key), fallback, min_value, max_value))

        return float(StrategySDK._coerce_float(fallback, fallback, min_value, max_value))

    @staticmethod
    def strategy_retention_config_schema() -> dict[str, Any]:
        return dict(StrategySDK.STRATEGY_RETENTION_CONFIG_SCHEMA)

    @staticmethod
    def parse_duration_minutes(value: Any) -> Optional[int]:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            parsed = int(value)
            if parsed < 0:
                return None
            return parsed

        text = str(value or "").strip().lower()
        if not text:
            return None
        match = StrategySDK._DURATION_VALUE_RE.match(text)
        if not match:
            return None

        try:
            raw_value = float(match.group("value"))
        except Exception:
            return None
        if raw_value < 0:
            return None

        unit = str(match.group("unit") or "").strip().lower()
        if not unit:
            return int(raw_value)
        factor = StrategySDK._DURATION_UNITS_TO_MINUTES.get(unit)
        if factor is None:
            return None
        return int(raw_value * factor)

    @staticmethod
    def normalize_strategy_retention_config(config: Any) -> dict[str, Any]:
        if not isinstance(config, dict):
            return {}
        cfg = dict(config)

        for key in ("max_opportunities", "retention_max_opportunities"):
            if key not in cfg:
                continue
            try:
                parsed = int(float(cfg.get(key)))
            except Exception:
                continue
            cfg[key] = max(0, min(parsed, 5000))

        retention_minutes: Optional[int] = None
        for key in (
            "retention_max_age_minutes",
            "retention_window",
            "retention_period",
            "retention_duration",
            "opportunity_ttl_minutes",
            "opportunity_ttl",
        ):
            if key not in cfg:
                continue
            parsed = StrategySDK.parse_duration_minutes(cfg.get(key))
            if parsed is None:
                continue
            retention_minutes = max(0, min(parsed, 60 * 24 * 90))
            break

        if retention_minutes is not None:
            cfg["retention_max_age_minutes"] = retention_minutes
            if "retention_window" not in cfg:
                cfg["retention_window"] = f"{retention_minutes}m"

        return cfg

    @staticmethod
    def validate_trader_filter_config(config: Any) -> dict[str, Any]:
        cfg = dict(StrategySDK.TRADER_FILTER_DEFAULTS)
        if isinstance(config, dict):
            cfg.update({str(k): v for k, v in config.items() if str(k) != "_schema"})

        cfg["min_confidence"] = StrategySDK._coerce_float(cfg.get("min_confidence"), 0.45, 0.0, 1.0)
        cfg["min_tier"] = StrategySDK.normalize_trader_tier(cfg.get("min_tier"), default="low")
        cfg["min_wallet_count"] = StrategySDK._coerce_int(cfg.get("min_wallet_count"), 2, 1, 1000)
        cfg["max_entry_price"] = StrategySDK._coerce_float(cfg.get("max_entry_price"), 0.85, 0.0, 1.0)
        cfg["min_order_size_usd"] = StrategySDK._coerce_float(cfg.get("min_order_size_usd"), 1.0, 0.01, 10_000_000.0)
        cfg["paper_min_order_size_usd"] = StrategySDK._coerce_float(
            cfg.get("paper_min_order_size_usd"),
            cfg["min_order_size_usd"],
            0.01,
            10_000_000.0,
        )
        cfg["live_min_order_size_usd"] = StrategySDK._coerce_float(
            cfg.get("live_min_order_size_usd"),
            cfg["min_order_size_usd"],
            0.01,
            10_000_000.0,
        )
        cfg["firehose_require_active_signal"] = StrategySDK._coerce_bool(
            cfg.get("firehose_require_active_signal"),
            True,
        )
        cfg["firehose_require_tradable_market"] = StrategySDK._coerce_bool(
            cfg.get("firehose_require_tradable_market"),
            False,
        )
        cfg["firehose_exclude_crypto_markets"] = StrategySDK._coerce_bool(
            cfg.get("firehose_exclude_crypto_markets"),
            False,
        )
        cfg["firehose_require_qualified_source"] = StrategySDK._coerce_bool(
            cfg.get("firehose_require_qualified_source"),
            True,
        )
        cfg["firehose_max_age_minutes"] = StrategySDK._coerce_int(
            cfg.get("firehose_max_age_minutes"),
            720,
            0,
            1440,
        )
        cfg["firehose_source_scope"] = StrategySDK.normalize_trader_source_scope(
            cfg.get("firehose_source_scope"),
            default="all",
        )
        cfg["firehose_side_filter"] = StrategySDK.normalize_trader_side(
            cfg.get("firehose_side_filter"),
            default="all",
        )
        cfg["traders_scope"] = StrategySDK.validate_trader_scope_config(cfg.get("traders_scope"))
        return StrategySDK.normalize_strategy_retention_config(cfg)

    @staticmethod
    def validate_trader_scope_config(config: Any) -> dict[str, Any]:
        cfg = dict(StrategySDK.TRADER_SCOPE_DEFAULTS)
        raw = config if isinstance(config, dict) else {}

        modes: list[str] = []
        seen_modes: set[str] = set()
        for raw_mode in raw.get("modes") or []:
            mode = str(raw_mode or "").strip().lower()
            if not mode or mode in seen_modes or mode not in StrategySDK.TRADER_SCOPE_MODE_CANONICAL:
                continue
            seen_modes.add(mode)
            modes.append(mode)
        cfg["modes"] = modes or list(StrategySDK.TRADER_SCOPE_DEFAULTS["modes"])

        wallets: list[str] = []
        seen_wallets: set[str] = set()
        for raw_wallet in raw.get("individual_wallets") or []:
            wallet = str(raw_wallet or "").strip().lower()
            if not wallet or wallet in seen_wallets:
                continue
            seen_wallets.add(wallet)
            wallets.append(wallet)
        cfg["individual_wallets"] = wallets

        groups: list[str] = []
        seen_groups: set[str] = set()
        for raw_group in raw.get("group_ids") or []:
            group_id = str(raw_group or "").strip()
            if not group_id or group_id in seen_groups:
                continue
            seen_groups.add(group_id)
            groups.append(group_id)
        cfg["group_ids"] = groups
        return cfg

    @staticmethod
    def validate_trader_runtime_metadata(config: Any) -> dict[str, Any]:
        cfg = dict(StrategySDK.TRADER_RUNTIME_DEFAULTS)
        raw = config if isinstance(config, dict) else {}
        cfg.update({str(k): v for k, v in raw.items() if str(k)})

        resume_policy = str(cfg.get("resume_policy") or "resume_full").strip().lower()
        if resume_policy not in {"resume_full", "manage_only", "flatten_then_start"}:
            resume_policy = "resume_full"
        cfg["resume_policy"] = resume_policy

        trading_schedule = cfg.get("trading_schedule_utc")
        schedule = trading_schedule if isinstance(trading_schedule, dict) else {}
        schedule_enabled = StrategySDK._coerce_bool(schedule.get("enabled"), False)
        schedule_days = StrategySDK._normalize_schedule_days(schedule.get("days"))
        schedule_start_time = StrategySDK._coerce_hhmm(schedule.get("start_time"), "00:00")
        schedule_end_time = StrategySDK._coerce_hhmm(schedule.get("end_time"), "23:59")
        schedule_start_date = StrategySDK._coerce_iso_date(schedule.get("start_date"))
        schedule_end_date = StrategySDK._coerce_iso_date(schedule.get("end_date"))
        if (
            schedule_start_date is not None
            and schedule_end_date is not None
            and schedule_start_date > schedule_end_date
        ):
            schedule_start_date, schedule_end_date = schedule_end_date, schedule_start_date
        schedule_end_at = StrategySDK._coerce_iso_datetime_utc(schedule.get("end_at"))
        cfg["trading_schedule_utc"] = {
            "enabled": schedule_enabled,
            "days": schedule_days,
            "start_time": schedule_start_time,
            "end_time": schedule_end_time,
            "start_date": schedule_start_date.isoformat() if schedule_start_date is not None else None,
            "end_date": schedule_end_date.isoformat() if schedule_end_date is not None else None,
            "end_at": (schedule_end_at.isoformat().replace("+00:00", "Z") if schedule_end_at is not None else None),
        }

        cfg["cadence_profile"] = StrategySDK._coerce_str(cfg.get("cadence_profile"), "custom")
        cfg["tags"] = StrategySDK._coerce_string_list(cfg.get("tags"))
        cfg["notes"] = StrategySDK._coerce_str(cfg.get("notes"), "")
        return cfg

    @staticmethod
    def validate_trader_risk_config(config: Any) -> dict[str, Any]:
        cfg = dict(StrategySDK.TRADER_RISK_DEFAULTS)
        raw = config if isinstance(config, dict) else {}
        cfg.update({str(k): v for k, v in raw.items() if str(k) != "_schema"})

        cfg["max_orders_per_cycle"] = StrategySDK._coerce_int(cfg.get("max_orders_per_cycle"), 6, 1, 1000)
        cfg["max_open_orders"] = StrategySDK._coerce_int(cfg.get("max_open_orders"), 20, 1, 2000)
        cfg["max_open_positions"] = StrategySDK._coerce_int(cfg.get("max_open_positions"), 12, 1, 1000)
        position_cap_scope = str(cfg.get("position_cap_scope") or "market_direction").strip().lower()
        if position_cap_scope not in {"market_direction", "market", "asset_timeframe"}:
            position_cap_scope = "market_direction"
        cfg["position_cap_scope"] = position_cap_scope
        cfg["max_position_notional_usd"] = StrategySDK._coerce_float(
            cfg.get("max_position_notional_usd"), 350.0, 1.0, 10_000_000.0
        )
        cfg["max_gross_exposure_usd"] = StrategySDK._coerce_float(
            cfg.get("max_gross_exposure_usd"), 2000.0, 1.0, 100_000_000.0
        )
        cfg["max_trade_notional_usd"] = StrategySDK._coerce_float(
            cfg.get("max_trade_notional_usd"), 350.0, 1.0, 10_000_000.0
        )
        cfg["max_daily_loss_usd"] = StrategySDK._coerce_float(cfg.get("max_daily_loss_usd"), 300.0, 1.0, 100_000_000.0)
        cfg["max_daily_spend_usd"] = StrategySDK._coerce_float(
            cfg.get("max_daily_spend_usd"), 2000.0, 1.0, 100_000_000.0
        )
        cfg["cooldown_seconds"] = StrategySDK._coerce_int(cfg.get("cooldown_seconds"), 0, 0, 86_400)
        cfg["order_ttl_seconds"] = StrategySDK._coerce_int(cfg.get("order_ttl_seconds"), 1200, 1, 86_400)
        cfg["slippage_bps"] = StrategySDK._coerce_float(cfg.get("slippage_bps"), 35.0, 0.0, 10_000.0)
        cfg["max_spread_bps"] = StrategySDK._coerce_float(cfg.get("max_spread_bps"), 75.0, 0.0, 10_000.0)
        cfg["retry_limit"] = StrategySDK._coerce_int(cfg.get("retry_limit"), 2, 0, 50)
        cfg["retry_backoff_ms"] = StrategySDK._coerce_int(cfg.get("retry_backoff_ms"), 250, 0, 60_000)
        cfg["allow_averaging"] = StrategySDK._coerce_bool(cfg.get("allow_averaging"), False)
        cfg["use_dynamic_sizing"] = StrategySDK._coerce_bool(cfg.get("use_dynamic_sizing"), True)
        cfg["halt_on_consecutive_losses"] = StrategySDK._coerce_bool(cfg.get("halt_on_consecutive_losses"), True)
        cfg["max_consecutive_losses"] = StrategySDK._coerce_int(cfg.get("max_consecutive_losses"), 4, 0, 1000)
        cfg["circuit_breaker_drawdown_pct"] = StrategySDK._coerce_float(
            cfg.get("circuit_breaker_drawdown_pct"), 12.0, 0.0, 100.0
        )
        default_portfolio = StrategySDK.TRADER_RISK_DEFAULTS.get("portfolio")
        portfolio_cfg = dict(default_portfolio) if isinstance(default_portfolio, dict) else {}
        raw_portfolio = cfg.get("portfolio")
        if isinstance(raw_portfolio, dict):
            portfolio_cfg.update({str(k): v for k, v in raw_portfolio.items() if str(k)})
        cfg["portfolio"] = {
            "enabled": StrategySDK._coerce_bool(portfolio_cfg.get("enabled"), False),
            "target_utilization_pct": StrategySDK._coerce_float(
                portfolio_cfg.get("target_utilization_pct"),
                100.0,
                1.0,
                100.0,
            ),
            "max_source_exposure_pct": StrategySDK._coerce_float(
                portfolio_cfg.get("max_source_exposure_pct"),
                100.0,
                1.0,
                100.0,
            ),
            "min_order_notional_usd": StrategySDK._coerce_float(
                portfolio_cfg.get("min_order_notional_usd"),
                10.0,
                1.0,
                10_000_000.0,
            ),
        }
        return cfg

    @staticmethod
    def validate_trader_opportunity_filter_config(config: Any) -> dict[str, Any]:
        cfg = dict(StrategySDK.TRADER_OPPORTUNITY_FILTER_DEFAULTS)
        raw = config if isinstance(config, dict) else {}
        cfg.update({str(k): v for k, v in raw.items() if str(k) != "_schema"})

        cfg["source_filter"] = StrategySDK.normalize_trader_source_scope(cfg.get("source_filter"), default="all")
        min_tier = str(cfg.get("min_tier") or "WATCH").strip().upper()
        if min_tier not in StrategySDK.TRADER_OPPORTUNITY_FILTER_TIER_CANONICAL:
            min_tier = "WATCH"
        cfg["min_tier"] = min_tier
        cfg["side_filter"] = StrategySDK.normalize_trader_side(cfg.get("side_filter"), default="all")
        cfg["confluence_limit"] = StrategySDK._coerce_int(cfg.get("confluence_limit"), 50, 1, 200)
        cfg["individual_trade_limit"] = StrategySDK._coerce_int(cfg.get("individual_trade_limit"), 40, 1, 500)
        cfg["individual_trade_min_confidence"] = StrategySDK._coerce_float(
            cfg.get("individual_trade_min_confidence"), 0.62, 0.0, 1.0
        )
        cfg["individual_trade_max_age_minutes"] = StrategySDK._coerce_int(
            cfg.get("individual_trade_max_age_minutes"), 180, 1, 1440
        )
        return cfg

    @staticmethod
    def validate_traders_copy_trade_config(config: Any) -> dict[str, Any]:
        cfg = dict(StrategySDK.TRADERS_COPY_TRADE_DEFAULTS)
        raw = config if isinstance(config, dict) else {}
        for key in (
            "min_confidence",
            "min_source_notional_usd",
            "max_entry_price",
            "max_signal_age_seconds",
            "min_live_liquidity_usd",
            "max_adverse_entry_drift_pct",
            "copy_delay_seconds",
            "copy_buys",
            "copy_sells",
            "max_position_size",
            "proportional_sizing",
            "proportional_multiplier",
            "base_size_usd",
            "max_size_usd",
            "traders_scope",
            "max_opportunities",
            "retention_max_opportunities",
            "retention_max_age_minutes",
            "retention_window",
            "retention_period",
            "retention_duration",
            "opportunity_ttl_minutes",
            "opportunity_ttl",
        ):
            if key in raw:
                cfg[key] = raw[key]

        cfg["min_confidence"] = StrategySDK._coerce_float(cfg.get("min_confidence"), 0.45, 0.0, 1.0)
        cfg["min_source_notional_usd"] = StrategySDK._coerce_float(
            cfg.get("min_source_notional_usd"),
            10.0,
            0.0,
            1_000_000.0,
        )
        cfg["max_entry_price"] = StrategySDK._coerce_float(cfg.get("max_entry_price"), 0.98, 0.0, 1.0)
        cfg["max_signal_age_seconds"] = StrategySDK._coerce_int(
            cfg.get("max_signal_age_seconds"),
            900,
            1,
            3600,
        )
        cfg["min_live_liquidity_usd"] = StrategySDK._coerce_float(
            cfg.get("min_live_liquidity_usd"),
            150.0,
            0.0,
            1_000_000_000.0,
        )
        cfg["max_adverse_entry_drift_pct"] = StrategySDK._coerce_float(
            cfg.get("max_adverse_entry_drift_pct"),
            2.0,
            0.0,
            100.0,
        )
        cfg["copy_delay_seconds"] = StrategySDK._coerce_int(cfg.get("copy_delay_seconds"), 0, 0, 300)
        cfg["copy_buys"] = StrategySDK._coerce_bool(cfg.get("copy_buys"), True)
        cfg["copy_sells"] = StrategySDK._coerce_bool(cfg.get("copy_sells"), True)
        cfg["max_position_size"] = StrategySDK._coerce_float(cfg.get("max_position_size"), 1000.0, 1.0, 1_000_000.0)
        cfg["proportional_sizing"] = StrategySDK._coerce_bool(cfg.get("proportional_sizing"), True)
        cfg["proportional_multiplier"] = StrategySDK._coerce_float(
            cfg.get("proportional_multiplier"),
            1.0,
            0.01,
            100.0,
        )
        cfg["base_size_usd"] = StrategySDK._coerce_float(cfg.get("base_size_usd"), 25.0, 1.0, 10_000.0)
        cfg["max_size_usd"] = StrategySDK._coerce_float(
            cfg.get("max_size_usd"),
            max(1.0, float(cfg["base_size_usd"])),
            1.0,
            50_000.0,
        )
        base_size_usd = StrategySDK._coerce_float(cfg.get("base_size_usd"), 25.0, 1.0, 10_000.0)
        max_size_usd = StrategySDK._coerce_float(cfg.get("max_size_usd"), base_size_usd, 1.0, 50_000.0)
        if max_size_usd < base_size_usd:
            max_size_usd = base_size_usd
        cfg["base_size_usd"] = base_size_usd
        cfg["max_size_usd"] = max_size_usd
        cfg["traders_scope"] = StrategySDK.validate_trader_scope_config(cfg.get("traders_scope"))
        return StrategySDK.normalize_strategy_retention_config(cfg)

    @staticmethod
    def validate_news_filter_config(config: Any) -> dict[str, Any]:
        cfg = dict(StrategySDK.NEWS_FILTER_DEFAULTS)
        if isinstance(config, dict):
            cfg.update({str(k): v for k, v in config.items() if str(k) != "_schema"})

        cfg["min_edge_percent"] = StrategySDK._coerce_float(cfg.get("min_edge_percent"), 5.0, 0.0, 100.0)
        cfg["min_confidence"] = StrategySDK._coerce_float(cfg.get("min_confidence"), 0.45, 0.0, 1.0)
        cfg["orchestrator_min_edge"] = StrategySDK._coerce_float(cfg.get("orchestrator_min_edge"), 10.0, 0.0, 100.0)
        cfg["require_verifier"] = StrategySDK._coerce_bool(cfg.get("require_verifier"), True)
        cfg["require_second_source"] = StrategySDK._coerce_bool(cfg.get("require_second_source"), False)
        cfg["min_supporting_articles"] = StrategySDK._coerce_int(cfg.get("min_supporting_articles"), 2, 1, 10)
        cfg["min_supporting_sources"] = StrategySDK._coerce_int(cfg.get("min_supporting_sources"), 2, 1, 10)
        cfg["max_signal_age_minutes"] = StrategySDK._coerce_int(cfg.get("max_signal_age_minutes"), 60, 1, 1440)
        return StrategySDK.normalize_strategy_retention_config(cfg)

    # ── Price helpers ──────────────────────────────────────────────

    @staticmethod
    def get_live_price(market: Any, prices: dict[str, dict], side: str = "YES") -> float:
        """Get the best available mid price for a market.

        Uses live CLOB prices when available, falls back to API prices.

        Args:
            market: A Market object with clob_token_ids and outcome_prices.
            prices: The prices dict passed to detect().
            side: 'YES' or 'NO'.

        Returns:
            The mid price as a float (0.0 if unavailable).
        """
        idx = 0 if side.upper() == "YES" else 1
        fallback = 0.0
        if hasattr(market, "outcome_prices") and len(market.outcome_prices) > idx:
            fallback = market.outcome_prices[idx]

        token_ids = getattr(market, "clob_token_ids", None) or []
        if len(token_ids) > idx:
            token_id = token_ids[idx]
            if token_id in prices:
                return prices[token_id].get("mid", fallback)
        return fallback

    @staticmethod
    def get_spread_bps(market: Any, prices: dict[str, dict], side: str = "YES") -> Optional[float]:
        """Get the bid-ask spread in basis points for a market side.

        Args:
            market: A Market object.
            prices: The prices dict passed to detect().
            side: 'YES' or 'NO'.

        Returns:
            Spread in basis points, or None if data unavailable.
        """
        idx = 0 if side.upper() == "YES" else 1
        token_ids = getattr(market, "clob_token_ids", None) or []
        if len(token_ids) <= idx:
            return None
        token_id = token_ids[idx]
        data = prices.get(token_id)
        if not data:
            return None
        bid = data.get("best_bid", 0)
        ask = data.get("best_ask", 0)
        mid = data.get("mid", 0)
        if mid <= 0 or bid <= 0 or ask <= 0:
            return None
        return ((ask - bid) / mid) * 10000

    @staticmethod
    def get_ws_mid_price(token_id: str) -> Optional[float]:
        """Get the real-time WebSocket mid price for a token.

        Args:
            token_id: The CLOB token ID.

        Returns:
            Mid price or None if not fresh.
        """
        try:
            from services.ws_feeds import get_feed_manager

            fm = get_feed_manager()
            if fm and fm.cache.is_fresh(token_id):
                return fm.cache.get_mid_price(token_id)
        except Exception:
            pass
        return None

    @staticmethod
    def get_ws_spread_bps(token_id: str) -> Optional[float]:
        """Get the real-time WebSocket spread in basis points.

        Args:
            token_id: The CLOB token ID.

        Returns:
            Spread in bps or None.
        """
        try:
            from services.ws_feeds import get_feed_manager

            fm = get_feed_manager()
            if fm:
                return fm.cache.get_spread_bps(token_id)
        except Exception:
            pass
        return None

    # ── Chainlink oracle ──────────────────────────────────────────

    @staticmethod
    def get_chainlink_price(asset: str) -> Optional[float]:
        """Get the latest Chainlink oracle price for a crypto asset.

        Args:
            asset: Asset symbol — 'BTC', 'ETH', 'SOL', or 'XRP'.

        Returns:
            The oracle price in USD, or None if unavailable.
        """
        try:
            from services.chainlink_feed import get_chainlink_feed

            feed = get_chainlink_feed()
            if feed:
                p = feed.get_price(asset.upper())
                return p.price if p else None
        except Exception:
            pass
        return None

    # ── LLM helpers ───────────────────────────────────────────────

    @staticmethod
    async def ask_llm(
        prompt: str,
        model: str = "gpt-4o-mini",
        system: str = "",
        purpose: str = "custom_strategy",
    ) -> str:
        """Send a prompt to an LLM and get the text response.

        Args:
            prompt: The user prompt.
            model: Model identifier (e.g. 'gpt-4o-mini', 'claude-haiku-4-5-20251001').
            system: Optional system prompt.
            purpose: Usage attribution tag (default: 'custom_strategy').

        Returns:
            The LLM response text, or empty string on failure.
        """
        try:
            from services.ai import get_llm_manager
            from services.ai.llm_provider import LLMMessage

            manager = get_llm_manager()
            if not manager.is_available():
                return ""

            messages = []
            if system:
                messages.append(LLMMessage(role="system", content=system))
            messages.append(LLMMessage(role="user", content=prompt))

            response = await manager.chat(
                messages=messages,
                model=model,
                purpose=purpose,
            )
            return response.content
        except Exception as e:
            logger.warning("StrategySDK.ask_llm failed: %s", e)
            return ""

    @staticmethod
    async def ask_llm_json(
        prompt: str,
        schema: dict,
        model: str = "gpt-4o-mini",
        system: str = "",
        purpose: str = "custom_strategy",
    ) -> dict:
        """Get structured JSON output from an LLM.

        Args:
            prompt: The user prompt.
            schema: JSON Schema the response must conform to.
            model: Model identifier.
            system: Optional system prompt.
            purpose: Usage attribution tag.

        Returns:
            Parsed dict conforming to the schema, or empty dict on failure.
        """
        try:
            from services.ai import get_llm_manager
            from services.ai.llm_provider import LLMMessage

            manager = get_llm_manager()
            if not manager.is_available():
                return {}

            messages = []
            if system:
                messages.append(LLMMessage(role="system", content=system))
            messages.append(LLMMessage(role="user", content=prompt))

            return await manager.structured_output(
                messages=messages,
                schema=schema,
                model=model,
                purpose=purpose,
            )
        except Exception as e:
            logger.warning("StrategySDK.ask_llm_json failed: %s", e)
            return {}

    # ── Fee calculation ───────────────────────────────────────────

    @staticmethod
    def calculate_fees(
        total_cost: float,
        expected_payout: float = 1.0,
        n_legs: int = 1,
    ) -> Optional[dict]:
        """Calculate comprehensive fees for a potential trade.

        Args:
            total_cost: Total cost to enter the position.
            expected_payout: Expected payout if the trade succeeds.
            n_legs: Number of legs in the trade.

        Returns:
            Dict with fee breakdown, or None on failure.
        """
        try:
            from services.fee_model import fee_model

            breakdown = fee_model.full_breakdown(
                total_cost=total_cost,
                expected_payout=expected_payout,
                n_legs=n_legs,
            )
            return {
                "winner_fee": breakdown.winner_fee,
                "gas_cost_usd": breakdown.gas_cost_usd,
                "spread_cost": breakdown.spread_cost,
                "multi_leg_slippage": breakdown.multi_leg_slippage,
                "total_fees": breakdown.total_fees,
                "fee_as_pct_of_payout": breakdown.fee_as_pct_of_payout,
            }
        except Exception as e:
            logger.warning("StrategySDK.calculate_fees failed: %s", e)
            return None

    @staticmethod
    def resolve_min_position_size(
        signal: dict[str, Any] | None = None,
        *,
        default_min_size: float = 0.0,
    ) -> float:
        """Resolve minimum executable position size from strategy signal payload.

        Priority:
        1) Explicit signal override: ``min_position_size_usd`` / ``min_position_size``
        2) Suggested size hint: ``suggested_size_usd`` (caps default floor downward)
        3) Fallback default floor supplied by caller.
        """
        min_size = max(0.0, float(default_min_size or 0.0))
        payload = signal if isinstance(signal, dict) else {}

        explicit = payload.get("min_position_size_usd")
        if explicit is None:
            explicit = payload.get("min_position_size")
        try:
            explicit_value = float(explicit) if explicit is not None else None
        except Exception:
            explicit_value = None
        if explicit_value is not None and explicit_value > 0:
            return explicit_value

        suggested = payload.get("suggested_size_usd")
        try:
            suggested_value = float(suggested) if suggested is not None else None
        except Exception:
            suggested_value = None
        if suggested_value is not None and suggested_value > 0:
            min_size = min(min_size, suggested_value) if min_size > 0 else suggested_value

        return max(0.0, min_size)

    @staticmethod
    def resolve_position_sizing(
        *,
        liquidity_usd: float,
        liquidity_fraction: float,
        hard_cap_usd: float,
        signal: dict[str, Any] | None = None,
        default_min_size: float = 0.0,
    ) -> dict[str, Any]:
        """Resolve max/min position size and tradeability from one SDK call."""
        try:
            liquidity = max(0.0, float(liquidity_usd or 0.0))
        except Exception:
            liquidity = 0.0
        try:
            fraction = max(0.0, float(liquidity_fraction or 0.0))
        except Exception:
            fraction = 0.0
        try:
            hard_cap = max(0.0, float(hard_cap_usd or 0.0))
        except Exception:
            hard_cap = 0.0

        max_position = liquidity * fraction
        if hard_cap > 0:
            max_position = min(max_position, hard_cap)

        min_position = StrategySDK.resolve_min_position_size(
            signal,
            default_min_size=default_min_size,
        )
        tradeable = max_position >= min_position if min_position > 0 else max_position > 0

        return {
            "liquidity_usd": liquidity,
            "max_position_size": max_position,
            "min_position_size": min_position,
            "is_tradeable": bool(tradeable),
        }

    # ── Market filtering helpers ──────────────────────────────────

    @staticmethod
    def active_binary_markets(markets: list) -> list:
        """Filter to only active, non-closed, binary (2-outcome) markets.

        Args:
            markets: List of Market objects.

        Returns:
            Filtered list of markets.
        """
        return [
            m
            for m in markets
            if getattr(m, "active", False)
            and not getattr(m, "closed", False)
            and len(getattr(m, "outcome_prices", []) or []) == 2
        ]

    @staticmethod
    def markets_by_category(events: list, category: str) -> list:
        """Get all active markets belonging to events of a given category.

        Args:
            events: List of Event objects.
            category: Category to filter by (e.g. 'crypto', 'politics').

        Returns:
            List of Market objects in matching events.
        """
        result = []
        for event in events:
            if getattr(event, "category", None) == category:
                for market in getattr(event, "markets", []) or []:
                    if getattr(market, "active", False) and not getattr(market, "closed", False):
                        result.append(market)
        return result

    @staticmethod
    def find_event_for_market(market: Any, events: list) -> Optional[Any]:
        """Find the parent Event for a given Market.

        Args:
            market: A Market object.
            events: List of Event objects.

        Returns:
            The parent Event, or None.
        """
        market_id = getattr(market, "id", None)
        if not market_id:
            return None
        for event in events:
            for m in getattr(event, "markets", []) or []:
                if getattr(m, "id", None) == market_id:
                    return event
        return None

    # ── Order book depth ───────────────────────────────────────

    @staticmethod
    def get_order_book_depth(
        market: Any,
        side: str = "YES",
        size_usd: float = 100.0,
    ) -> Optional[dict]:
        """Get order book depth analysis for a market side.

        Uses the internal VWAP calculator to estimate execution cost,
        slippage, and fill probability for a given position size.

        Args:
            market: A Market object.
            side: 'YES' or 'NO'.
            size_usd: Position size in USD to estimate execution for.

        Returns:
            Dict with depth analysis, or None if unavailable:
            {
                "vwap_price": float,       # Volume-weighted avg price
                "slippage_bps": float,     # Slippage in basis points
                "fill_probability": float, # Estimated fill probability (0-1)
                "available_liquidity": float,  # Liquidity at this level
            }
        """
        try:
            from services.optimization.vwap import vwap_calculator

            if not vwap_calculator:
                return None

            token_idx = 0 if side.upper() == "YES" else 1
            token_ids = getattr(market, "clob_token_ids", None) or []
            if len(token_ids) <= token_idx:
                return None

            result = vwap_calculator.estimate_execution(
                token_id=token_ids[token_idx],
                size_usd=size_usd,
            )
            if result is None:
                return None

            return {
                "vwap_price": result.get("vwap_price", 0),
                "slippage_bps": result.get("slippage_bps", 0),
                "fill_probability": result.get("fill_probability", 1.0),
                "available_liquidity": result.get("available_liquidity", 0),
            }
        except Exception:
            return None

    @staticmethod
    def get_book_levels(
        market: Any,
        side: str = "YES",
        max_levels: int = 10,
    ) -> Optional[list[dict]]:
        """Get raw order book levels (bids or asks) for a market side.

        Args:
            market: A Market object.
            side: 'YES' or 'NO'.
            max_levels: Maximum number of price levels to return.

        Returns:
            List of {price, size} dicts sorted by best price, or None.
        """
        try:
            from services.optimization.vwap import vwap_calculator

            if not vwap_calculator:
                return None

            token_idx = 0 if side.upper() == "YES" else 1
            token_ids = getattr(market, "clob_token_ids", None) or []
            if len(token_ids) <= token_idx:
                return None

            return vwap_calculator.get_book_levels(
                token_id=token_ids[token_idx],
                max_levels=max_levels,
            )
        except Exception:
            return None

    # ── Historical price access ────────────────────────────────

    @staticmethod
    def get_price_history(
        token_id: str,
        max_snapshots: int = 60,
    ) -> list[dict]:
        """Get recent price snapshots for a token.

        Returns the most recent snapshots from the WebSocket feed cache.
        Each snapshot includes timestamp, mid, bid, ask.

        Args:
            token_id: The CLOB token ID.
            max_snapshots: Maximum number of snapshots to return.

        Returns:
            List of price snapshots (newest first), each:
            {"timestamp": str, "mid": float, "bid": float, "ask": float}
            Empty list if no history available.
        """
        try:
            from services.ws_feeds import get_feed_manager

            fm = get_feed_manager()
            if fm and hasattr(fm.cache, "get_price_history"):
                return fm.cache.get_price_history(token_id, max_snapshots=max_snapshots)
        except Exception:
            pass
        return []

    @staticmethod
    def get_price_change(
        token_id: str,
        lookback_seconds: int = 300,
    ) -> Optional[dict]:
        """Get price change over a lookback period.

        Args:
            token_id: The CLOB token ID.
            lookback_seconds: How far back to look (default 5 minutes).

        Returns:
            Dict with price change info, or None:
            {
                "current_mid": float,
                "prior_mid": float,
                "change_abs": float,
                "change_pct": float,
                "snapshots_in_window": int,
            }
        """
        try:
            from services.ws_feeds import get_feed_manager

            fm = get_feed_manager()
            if fm and hasattr(fm.cache, "get_price_change"):
                return fm.cache.get_price_change(token_id, lookback_seconds=lookback_seconds)
        except Exception:
            pass
        return None

    # ── Trade tape access ──────────────────────────────────────

    @staticmethod
    def get_recent_trades(token_id: str, max_trades: int = 100) -> list:
        """Return recent trades for a token from the WebSocket trade tape.

        Returns list of TradeRecord objects with price, size, side, timestamp.
        Returns empty list if trade data is unavailable.
        """
        try:
            from services.ws_feeds import FeedManager

            mgr = FeedManager.get_instance()
            return mgr.cache.get_recent_trades(token_id, max_trades)
        except Exception:
            return []

    @staticmethod
    def get_trade_volume(token_id: str, lookback_seconds: float = 300.0) -> dict:
        """Return buy/sell volume over a lookback window.

        Returns dict with keys: buy_volume, sell_volume, total, trade_count.
        Returns zero-volume dict if trade data is unavailable.
        """
        try:
            from services.ws_feeds import FeedManager

            mgr = FeedManager.get_instance()
            return mgr.cache.get_trade_volume(token_id, lookback_seconds)
        except Exception:
            return {"buy_volume": 0.0, "sell_volume": 0.0, "total": 0.0, "trade_count": 0}

    @staticmethod
    def get_buy_sell_imbalance(token_id: str, lookback_seconds: float = 300.0) -> float:
        """Return buy/sell order flow imbalance in [-1, 1].

        +1 = all buys, -1 = all sells, 0 = balanced or no data.
        """
        try:
            from services.ws_feeds import FeedManager

            mgr = FeedManager.get_instance()
            return mgr.cache.get_buy_sell_imbalance(token_id, lookback_seconds)
        except Exception:
            return 0.0

    # ── News data access ───────────────────────────────────────

    @staticmethod
    def get_recent_news(
        query: str = "",
        max_articles: int = 20,
    ) -> list[dict]:
        """Get recent news articles, optionally filtered by query.

        Args:
            query: Optional search query to filter articles.
            max_articles: Maximum number of articles to return.

        Returns:
            List of article dicts with title, source, published_at, summary.
        """
        try:
            from services.news.feed_service import news_feed_service

            articles = news_feed_service.search_articles(query=query, limit=max_articles)
            return [
                {
                    "title": getattr(a, "title", ""),
                    "source": getattr(a, "source", ""),
                    "published_at": str(getattr(a, "published_at", "")),
                    "summary": getattr(a, "summary", getattr(a, "description", "")),
                    "url": getattr(a, "url", getattr(a, "link", "")),
                }
                for a in (articles or [])
            ]
        except Exception:
            return []

    @staticmethod
    def get_news_for_market(
        market: Any,
        max_articles: int = 10,
    ) -> list[dict]:
        """Get news articles semantically matched to a specific market.

        Uses the semantic matcher to find cached news articles whose content
        is most similar to the market's question.  Articles are embedded by
        the scanner prefetch loop; this call is lightweight (numpy dot
        product or FAISS search).

        Args:
            market: A Market object (must have a ``question`` attribute).
            max_articles: Maximum number of articles to return.

        Returns:
            List of article dicts with keys: title, source, relevance_score,
            published_at, summary.
        """
        try:
            from services.news.semantic_matcher import semantic_matcher

            question = getattr(market, "question", "") or ""
            if not question:
                return []

            matches = semantic_matcher.find_matches(question, top_k=max_articles)
            return [
                {
                    "title": m.get("title", ""),
                    "source": m.get("source", ""),
                    "relevance_score": m.get("score", 0),
                    "published_at": str(m.get("published_at", "")),
                    "summary": m.get("summary", ""),
                }
                for m in (matches or [])
            ]
        except Exception as e:
            import logging

            logging.getLogger(__name__).debug("get_news_for_market failed: %s", e)
            return []

    # ── Data source access ───────────────────────────────

    @staticmethod
    async def get_data_records(
        source_slug: str | None = None,
        source_slugs: list[str] | None = None,
        limit: int = 200,
        geotagged: bool | None = None,
        category: str | None = None,
        since: str | None = None,
    ) -> list[dict]:
        """Read normalized records from the data source store."""
        try:
            from services.data_source_sdk import DataSourceSDK

            return await DataSourceSDK.get_records(
                source_slug=source_slug,
                source_slugs=source_slugs,
                limit=limit,
                geotagged=geotagged,
                category=category,
                since=since,
            )
        except Exception as e:
            logger.warning("StrategySDK.get_data_records failed: %s", e)
            return []

    @staticmethod
    async def get_latest_data_record(
        source_slug: str,
        external_id: str | None = None,
    ) -> dict | None:
        """Read the newest normalized record for a source (optionally by external_id)."""
        try:
            from services.data_source_sdk import DataSourceSDK

            return await DataSourceSDK.get_latest_record(
                source_slug=source_slug,
                external_id=external_id,
            )
        except Exception as e:
            logger.warning("StrategySDK.get_latest_data_record failed: %s", e)
            return None

    @staticmethod
    async def run_data_source(source_slug: str, max_records: int = 500) -> dict:
        """Trigger a source run and return ingestion summary."""
        try:
            from services.data_source_sdk import DataSourceSDK

            return await DataSourceSDK.run_source(source_slug=source_slug, max_records=max_records)
        except Exception as e:
            logger.warning("StrategySDK.run_data_source failed: %s", e)
            return {}

    @staticmethod
    async def list_data_sources(
        enabled_only: bool = True,
        source_key: str | None = None,
        include_code: bool = False,
    ) -> list[dict]:
        """List DB-backed data sources."""
        try:
            from services.data_source_sdk import DataSourceSDK

            return await DataSourceSDK.list_sources(
                enabled_only=enabled_only,
                source_key=source_key,
                include_code=include_code,
            )
        except Exception as e:
            logger.warning("StrategySDK.list_data_sources failed: %s", e)
            return []

    @staticmethod
    async def get_data_source(source_slug: str, include_code: bool = True) -> dict:
        """Fetch one DB-backed data source by slug."""
        try:
            from services.data_source_sdk import DataSourceSDK

            return await DataSourceSDK.get_source(source_slug=source_slug, include_code=include_code)
        except Exception as e:
            logger.warning("StrategySDK.get_data_source failed: %s", e)
            return {}

    @staticmethod
    def validate_data_source(source_code: str, class_name: str | None = None) -> dict:
        """Validate source code before create/update."""
        try:
            from services.data_source_sdk import DataSourceSDK

            return DataSourceSDK.validate_source(source_code=source_code, class_name=class_name)
        except Exception as e:
            logger.warning("StrategySDK.validate_data_source failed: %s", e)
            return {
                "valid": False,
                "errors": [str(e)],
                "warnings": [],
                "class_name": None,
                "source_name": None,
                "source_description": None,
                "capabilities": {"has_fetch": False, "has_fetch_async": False, "has_transform": False},
            }

    @staticmethod
    async def create_data_source(
        *,
        slug: str,
        source_code: str,
        source_key: str = "custom",
        source_kind: str = "python",
        name: str | None = None,
        description: str | None = None,
        class_name: str | None = None,
        retention: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        config_schema: dict[str, Any] | None = None,
        enabled: bool = True,
        is_system: bool = False,
        sort_order: int = 0,
    ) -> dict:
        """Create a new DB-backed data source."""
        try:
            from services.data_source_sdk import DataSourceSDK

            return await DataSourceSDK.create_source(
                slug=slug,
                source_code=source_code,
                source_key=source_key,
                source_kind=source_kind,
                name=name,
                description=description,
                class_name=class_name,
                retention=retention,
                config=config,
                config_schema=config_schema,
                enabled=enabled,
                is_system=is_system,
                sort_order=sort_order,
            )
        except Exception as e:
            logger.warning("StrategySDK.create_data_source failed: %s", e)
            return {}

    @staticmethod
    async def update_data_source(
        source_slug: str,
        *,
        slug: str | None = None,
        source_key: str | None = None,
        source_kind: str | None = None,
        name: str | None = None,
        description: str | None = None,
        source_code: str | None = None,
        class_name: str | None = None,
        retention: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        config_schema: dict[str, Any] | None = None,
        enabled: bool | None = None,
        unlock_system: bool = False,
    ) -> dict:
        """Update an existing DB-backed data source."""
        try:
            from services.data_source_sdk import DataSourceSDK

            return await DataSourceSDK.update_source(
                source_slug=source_slug,
                slug=slug,
                source_key=source_key,
                source_kind=source_kind,
                name=name,
                description=description,
                source_code=source_code,
                class_name=class_name,
                retention=retention,
                config=config,
                config_schema=config_schema,
                enabled=enabled,
                unlock_system=unlock_system,
            )
        except Exception as e:
            logger.warning("StrategySDK.update_data_source failed: %s", e)
            return {}

    @staticmethod
    async def delete_data_source(
        source_slug: str,
        *,
        tombstone_system_source: bool = True,
        unlock_system: bool = False,
        reason: str = "deleted_via_strategy_sdk",
    ) -> dict:
        """Delete a data source by slug."""
        try:
            from services.data_source_sdk import DataSourceSDK

            return await DataSourceSDK.delete_source(
                source_slug=source_slug,
                tombstone_system_source=tombstone_system_source,
                unlock_system=unlock_system,
                reason=reason,
            )
        except Exception as e:
            logger.warning("StrategySDK.delete_data_source failed: %s", e)
            return {}

    @staticmethod
    async def reload_data_source(source_slug: str) -> dict:
        """Reload a data source runtime by slug."""
        try:
            from services.data_source_sdk import DataSourceSDK

            return await DataSourceSDK.reload_source(source_slug=source_slug)
        except Exception as e:
            logger.warning("StrategySDK.reload_data_source failed: %s", e)
            return {}

    @staticmethod
    async def get_data_source_runs(source_slug: str, limit: int = 20) -> list[dict]:
        """Read recent run history for a source."""
        try:
            from services.data_source_sdk import DataSourceSDK

            return await DataSourceSDK.get_recent_runs(source_slug=source_slug, limit=limit)
        except Exception as e:
            logger.warning("StrategySDK.get_data_source_runs failed: %s", e)
            return []

    # ── Trader data access ───────────────────────────────

    @staticmethod
    async def get_trader_firehose_signals(
        *,
        limit: int = 250,
        include_filtered: bool = False,
        include_source_context: bool = True,
    ) -> list[dict[str, Any]]:
        """Read trader firehose rows (pool/tracked/group provenance included)."""
        try:
            from services.traders_sdk import TradersSDK

            return await TradersSDK.get_firehose_signals(
                limit=limit,
                include_filtered=include_filtered,
                include_source_context=include_source_context,
            )
        except Exception as e:
            logger.warning("StrategySDK.get_trader_firehose_signals failed: %s", e)
            return []

    @staticmethod
    async def get_trader_strategy_signals(
        *,
        limit: int = 50,
        include_filtered: bool = False,
    ) -> list[dict[str, Any]]:
        """Read strategy-filtered trader signals used by traders_confluence."""
        try:
            from services.traders_sdk import TradersSDK

            return await TradersSDK.get_strategy_filtered_signals(
                limit=limit,
                include_filtered=include_filtered,
            )
        except Exception as e:
            logger.warning("StrategySDK.get_trader_strategy_signals failed: %s", e)
            return []

    @staticmethod
    async def get_trader_confluence_signals(
        *,
        min_strength: float = 0.0,
        min_tier: str = "WATCH",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Read active confluence signals from trader intelligence."""
        try:
            from services.traders_sdk import TradersSDK

            return await TradersSDK.get_confluence_signals(
                min_strength=min_strength,
                min_tier=min_tier,
                limit=limit,
            )
        except Exception as e:
            logger.warning("StrategySDK.get_trader_confluence_signals failed: %s", e)
            return []

    @staticmethod
    async def get_pooled_traders(
        *,
        limit: int = 200,
        tier: str | None = None,
        include_blacklisted: bool = True,
        tracked_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Read wallets currently selected into the smart pool."""
        try:
            from services.traders_sdk import TradersSDK

            return await TradersSDK.get_pooled_traders(
                limit=limit,
                tier=tier,
                include_blacklisted=include_blacklisted,
                tracked_only=tracked_only,
            )
        except Exception as e:
            logger.warning("StrategySDK.get_pooled_traders failed: %s", e)
            return []

    @staticmethod
    async def get_tracked_traders(
        *,
        limit: int = 200,
        include_recent_activity: bool = False,
        activity_hours: int = 24,
    ) -> list[dict[str, Any]]:
        """Read tracked trader wallets with optional activity enrichment."""
        try:
            from services.traders_sdk import TradersSDK

            return await TradersSDK.get_tracked_traders(
                limit=limit,
                include_recent_activity=include_recent_activity,
                activity_hours=activity_hours,
            )
        except Exception as e:
            logger.warning("StrategySDK.get_tracked_traders failed: %s", e)
            return []

    @staticmethod
    async def get_trader_groups(
        *,
        include_members: bool = False,
        member_limit: int = 25,
    ) -> list[dict[str, Any]]:
        """Read trader groups with optional member payloads."""
        try:
            from services.traders_sdk import TradersSDK

            return await TradersSDK.get_groups(
                include_members=include_members,
                member_limit=member_limit,
            )
        except Exception as e:
            logger.warning("StrategySDK.get_trader_groups failed: %s", e)
            return []

    @staticmethod
    async def get_trader_tags() -> list[dict[str, Any]]:
        """Read trader tag definitions with wallet counts."""
        try:
            from services.traders_sdk import TradersSDK

            return await TradersSDK.get_tags()
        except Exception as e:
            logger.warning("StrategySDK.get_trader_tags failed: %s", e)
            return []

    @staticmethod
    async def get_traders_by_tag(tag_name: str, *, limit: int = 100) -> list[dict[str, Any]]:
        """Read wallets carrying a specific trader tag."""
        try:
            from services.traders_sdk import TradersSDK

            return await TradersSDK.get_traders_by_tag(tag_name=tag_name, limit=limit)
        except Exception as e:
            logger.warning("StrategySDK.get_traders_by_tag failed: %s", e)
            return []

    @staticmethod
    def get_whale_cohorts() -> list[dict]:
        from services.wallet_intelligence import wallet_intelligence

        return list(wallet_intelligence.cohort_analyzer._cached_cohorts)

    @staticmethod
    def get_market_regime(market_id: str) -> dict:
        from services.market_regime import market_regime_classifier

        return {
            "regime": market_regime_classifier.get_regime(market_id),
            "params": market_regime_classifier.get_regime_params(market_id),
        }
