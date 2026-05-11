"""
Strategy SDK — high-level helpers for custom strategy development.

This module provides a clean, stable API surface for strategy authors.
Import it in your strategy code with:

    from services.strategy_sdk import StrategySDK

All methods are designed to be safe and handle missing data gracefully.

Live orderbook access (sub-200ms p50 via Polymarket CLOB WebSocket):

    StrategySDK.get_best_bid_ask(market, side)
    StrategySDK.get_book_levels(market, side, max_levels)
    StrategySDK.get_order_book_depth(market, side, size_usd)  # VWAP + slippage
    StrategySDK.get_book_imbalance(market, side, levels)      # signed [-1, +1]

These read from the WS-fed PriceCache in services.ws_feeds.FeedManager.
The token must be subscribed; production strategies should call
``feed_manager.polymarket_feed.subscribe(token_ids=...)`` once on market
discovery to keep the cache warm.

Rolling price windows:

    StrategySDK.PriceWindow         # generic single-stream rolling window
                                    # (use one per token / outcome / feed)

For any market (binary, multi-outcome, cycle), keep a
``dict[token_id, StrategySDK.PriceWindow]`` — one window per outcome's
token id. For external feeds (oracle, spot, etc.) instantiate a single
``PriceWindow`` keyed by whatever stream identifier you choose.

Cycle markets (5-minute / 15-minute / 1-hour crypto over-under, etc.):

    StrategySDK.CycleTracker        # idempotent milestone tracking
                                    # (only meaningful for recurring markets)
    StrategySDK.time_to_resolution  # universal: works on any market

Binary fair-value math (cash-or-nothing digital under zero-drift GBM):

    StrategySDK.prob_above(value, threshold, vol_per_sec, seconds_remaining)
    StrategySDK.prob_below(value, threshold, vol_per_sec, seconds_remaining)
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import json
import logging
import math
from pathlib import Path
import re
import time
from typing import Any, Optional
from utils.converters import coerce_bool as _coerce_bool
from services.strategy_helpers.persistent_state import PersistentState
from services.strategy_helpers.price_window import (
    MultiWindow,
    PriceWindow,
    timeframes_agree,
    weighted_signal,
)
from services.strategy_helpers.cycle_tracker import CycleTracker
from services.strategy_helpers.price_history import MarketPriceHistory, PriceSnapshot
# Note: crypto_strategy_utils is loaded lazily via a descriptor below to
# avoid a circular import — that module re-exports from
# services.strategies.crypto_strategy_utils, and importing it at
# module-load time triggers services.strategies.__init__.py, which in
# turn imports strategies (e.g. news_edge) that import StrategySDK.

logger = logging.getLogger(__name__)

_REST_PRIME_COOLDOWN_S: float = 3.0
_rest_prime_last_ts: dict[str, float] = {}


def _prime_book_via_rest(token_id: str) -> None:
    try:
        import asyncio
        from services.ws_feeds import get_feed_manager
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        feed_mgr = get_feed_manager()
        loop.create_task(feed_mgr.get_order_book(token_id))
    except Exception as exc:
        logger.debug(
            "StrategySDK REST prime failed for token %s: %s",
            token_id[:18] if token_id else "?", exc,
        )


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

    # PriceWindow is the canonical rolling-window primitive — it makes
    # no outcome assumptions and works for any single price stream
    # (one outcome of a multi-outcome market, a crypto spot price, an
    # oracle feed, etc.). Multi-stream callers maintain their own
    # ``dict[stream_id, PriceWindow]``.
    PriceWindow = PriceWindow

    # MultiWindow fans one price stream into N rolling lookbacks at
    # different sizes — the canonical primitive for compound-movement
    # / multi-timeframe-confirmation strategies (e.g. 5m+15m+1h+4h on
    # the same crypto spot price). One ``record(price)`` updates every
    # child window; ``all_agree()`` / ``aligned_count()`` provide the
    # standard confirmation checks. Module helpers ``timeframes_agree``
    # and ``weighted_signal`` accept the dict shapes MultiWindow emits.
    MultiWindow = MultiWindow
    timeframes_agree = staticmethod(timeframes_agree)
    weighted_signal = staticmethod(weighted_signal)

    # PersistentState is the durable counterpart to ``BaseStrategy.state``
    # — a per-strategy key/value cache backed by ``strategy_persistent_state``.
    # Unlike ``self.state`` (in-memory, lost on worker restart), this
    # survives across restarts and across worker processes. Use it for
    # rolling stats, last-seen timestamps, multi-window snapshots, or any
    # other state your strategy needs to outlive a single process.
    #
    # Usage:
    #     ps = StrategySDK.PersistentState(strategy_slug=self.strategy_type)
    #     await ps.load()           # hydrate from DB once
    #     ps.set("last_ts", 12345)  # cached + marked dirty
    #     await ps.flush()          # write dirty entries
    PersistentState = PersistentState

    # CycleTracker is opt-in: only meaningful for fixed-cycle recurring
    # markets (e.g. Polymarket's 5m / 15m / 1h crypto over-under
    # markets). One-shot markets like elections or sports should use
    # ``time_to_resolution`` and reason about their lifecycle directly.
    CycleTracker = CycleTracker

    # MarketPriceHistory / PriceSnapshot — binary (yes/no) rolling window.
    # Re-exported here so strategies can use ``StrategySDK.MarketPriceHistory``
    # without a separate import from strategy_helpers.price_history.
    MarketPriceHistory = MarketPriceHistory
    PriceSnapshot = PriceSnapshot

    # Crypto-strategy utility namespace.
    #
    # All public functions in services.strategies.crypto_strategy_utils are
    # accessible via ``StrategySDK.crypto.<name>(...)`` so a strategy never
    # needs to import that file directly. This is the canonical entry point
    # for crypto helpers — anything that lives there is fair game from any
    # strategy.
    #
    # Examples:
    #     pick = StrategySDK.crypto.pick_oracle_source(row, prefer="binance_direct")
    #     status = StrategySDK.crypto.extract_oracle_status(
    #         live_market=lm, payload=p, now_ms=int(time.time() * 1000),
    #     )
    #     tf = StrategySDK.crypto.normalize_timeframe(row["timeframe"])
    #     fee_pct = StrategySDK.crypto.taker_fee_pct(entry_price)
    #
    # Surface includes: oracle (pick_oracle_source, extract_oracle_status,
    # parse_oracle_point, normalize_oracle_source, resolve_oracle_availability,
    # to_epoch_ms, compute_age_ms), timeframe (normalize_timeframe,
    # timeframe_seconds, default_min_seconds_left_for_entry,
    # default_max_market_data_age_ms, default_max_oracle_age_ms), market
    # construction (build_binary_crypto_market, build_binary_market_from_row),
    # row helpers (seconds_left_from_row, spread_pct_from_row,
    # market_ml_probability_yes, history_cancel_peak), fee (taker_fee_pct,
    # fee_aware_min_edge_pct), and generic (first_present, normalize_ratio,
    # normalize_signed_ratio, bounded_sigmoid, parse_datetime_utc).
    class _CryptoDescriptor:
        """Lazy proxy for ``services.strategy_helpers.crypto_strategy_utils``.

        Resolved on first access to avoid pulling the strategies-package
        init chain at SDK module-load time. Returns the actual module so
        ``StrategySDK.crypto.foo(...)`` works exactly like a direct import.
        """

        _resolved = None

        def __get__(self, obj, objtype=None):
            if StrategySDK._CryptoDescriptor._resolved is None:
                from services.strategy_helpers import crypto_strategy_utils as _module
                StrategySDK._CryptoDescriptor._resolved = _module
            return StrategySDK._CryptoDescriptor._resolved

    crypto = _CryptoDescriptor()

    _ai_instance = None

    class _AIDescriptor:
        def __get__(self, obj, objtype=None):
            if StrategySDK._ai_instance is None:
                from services.ai import get_llm_manager
                from services.ai_sdk import AISDK

                StrategySDK._ai_instance = AISDK(get_llm_manager(), purpose="custom_strategy")
            return StrategySDK._ai_instance

    ai = _AIDescriptor()

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
        "shadow_min_order_size_usd": 1.0,
        "live_min_order_size_usd": 1.0,
        "firehose_require_active_signal": True,
        "firehose_require_tradable_market": True,
        "firehose_exclude_crypto_markets": False,
        "firehose_require_qualified_source": True,
        "firehose_max_age_minutes": 720,
        "firehose_source_scope": "all",
        "firehose_side_filter": "all",
    }
    TRADER_EXECUTION_POLICY_DEFAULTS: dict[str, Any] = {
        "allow_taker_limit_buy_above_signal": False,
        "aggressive_limit_buy_submit_as_gtc": False,
    }
    TRADER_EXECUTION_POLICY_CONFIG_SCHEMA: dict[str, Any] = {
        "param_fields": [
            {
                "key": "allow_taker_limit_buy_above_signal",
                "label": "Allow Taker Limit Buy Above Signal Price",
                "type": "boolean",
            },
            {
                "key": "aggressive_limit_buy_submit_as_gtc",
                "label": "Submit Aggressive Buy Limits As GTC",
                "type": "boolean",
            },
        ]
    }
    TRADER_FILTER_CONFIG_SCHEMA: dict[str, Any] = {
        "param_fields": [
            {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
            {
                "key": "min_tier",
                "label": "Min Tier",
                "type": "enum",
                "options": ["low", "medium", "high", "extreme"],
                "phase": "signal",
            },
            {"key": "min_wallet_count", "label": "Min Wallet Count", "type": "integer", "min": 1, "phase": "signal"},
            {"key": "max_entry_price", "label": "Max Entry Price", "type": "number", "min": 0, "max": 1},
            {"key": "min_order_size_usd", "label": "Min Order Size (USD)", "type": "number", "min": 0.01},
            {"key": "shadow_min_order_size_usd", "label": "Shadow Min Order Size (USD)", "type": "number", "min": 0.01},
            {"key": "live_min_order_size_usd", "label": "Live Min Order Size (USD)", "type": "number", "min": 0.01},
            {"key": "firehose_require_active_signal", "label": "Require Active Signal", "type": "boolean", "phase": "signal"},
            {"key": "firehose_require_tradable_market", "label": "Require Tradable Market", "type": "boolean", "phase": "signal"},
            {"key": "firehose_exclude_crypto_markets", "label": "Exclude Crypto Markets", "type": "boolean", "phase": "signal"},
            {"key": "firehose_require_qualified_source", "label": "Require Qualified Source", "type": "boolean", "phase": "signal"},
            {
                "key": "firehose_max_age_minutes",
                "label": "Firehose Max Age (min)",
                "type": "integer",
                "min": 1,
                "max": 1440,
                "phase": "signal",
            },
            {
                "key": "firehose_source_scope",
                "label": "Source Scope",
                "type": "enum",
                "options": ["all", "tracked", "pool"],
                "phase": "signal",
            },
            {
                "key": "firehose_side_filter",
                "label": "Side Filter",
                "type": "enum",
                "options": ["all", "buy", "sell"],
                "phase": "signal",
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
        "max_position_notional_usd": 5.0,
        "max_gross_exposure_usd": 2000.0,
        "max_trade_notional_usd": 5.0,
        "max_daily_loss_usd": 300.0,
        "max_daily_spend_usd": 2000.0,
        "cooldown_seconds": 0,
        "order_ttl_seconds": 1200,
        "slippage_bps": 35.0,
        "max_spread_bps": 75.0,
        "retry_limit": 2,
        "retry_backoff_ms": 250,
        "allow_averaging": False,
        "halt_on_consecutive_losses": True,
        "max_consecutive_losses": 4,
        "max_entry_drift_pct": 10.0,
        "max_market_data_age_ms": None,
        "allow_taker_limit_buy_above_signal": False,
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
        {"key": "halt_on_consecutive_losses", "label": "Halt on Loss Streak", "type": "boolean"},
        {"key": "max_consecutive_losses", "label": "Max Consecutive Losses", "type": "integer", "min": 0, "max": 1000},
        {
            "key": "max_entry_drift_pct",
            "label": "Max Entry Drift From Signal (%)",
            "type": "number",
            "min": 0,
            "max": 100,
            "description": (
                "Symmetric tolerance: |live_price - signal_entry_price| / signal_entry_price * 100. "
                "Default 10%. Lower → strategies skip when market moves; higher → more fills at worse prices."
            ),
        },
        {
            "key": "max_market_data_age_ms",
            "label": "Max Market Data Age (ms)",
            "type": "integer",
            "min": 50,
            "max": 300000,
            "description": (
                "Per-bot ceiling for live quote staleness at gate-time. Empty → fall back to "
                "strategy_params.max_market_data_age_ms or env EXECUTION_MARKET_DATA_MAX_AGE_MS."
            ),
        },
        {
            "key": "allow_taker_limit_buy_above_signal",
            "label": "Allow Taker Limit Buy Above Signal Price",
            "type": "boolean",
            "description": (
                "When ON, shadow simulator may fill BUY legs at prices above signal entry_price (chase up). "
                "Default OFF means simulator rejects whenever the book has moved up since signal — the dominant "
                "cause of `Execution submission: limit_price_not_executable` blocks."
            ),
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

    # ── Advanced exit-execution helpers ───────────────────────────────────
    #
    # These build ``ExitPolicy`` instances that strategies attach to
    # ``BaseStrategy.exit_policies`` (a ``{trigger: ExitPolicy}`` map) or
    # ``ExitDecision.exit_policy`` (per-decision override). When attached,
    # the orchestrator's ``exit_executor`` splits the exit into a ladder of
    # child orders instead of placing one large sell. See
    # ``services.trader_orchestrator.exit_executor`` for the runtime details.
    #
    # Example use in a strategy:
    #
    #     from services.strategy_sdk import StrategySDK
    #     from services.strategies.base import BaseStrategy
    #
    #     class MyStrategy(BaseStrategy):
    #         strategy_type = "my_strategy"
    #         name = "My Strategy"
    #         description = "..."
    #         exit_policies = {
    #             "stop_loss": StrategySDK.build_ladder_exit_policy(
    #                 levels=10,
    #                 step_ticks=1,
    #                 chunk_size=2,           # contracts per child order
    #                 distribution="back_loaded",
    #                 escalation_after_seconds=5.0,
    #                 escalation_action="marketable_ioc",
    #                 reprice_on_mid_drift_bps=80,
    #             ),
    #             "take_profit": None,        # legacy single-order path
    #         }

    @staticmethod
    def build_ladder_exit_policy(
        *,
        levels: int = 5,
        step_ticks: int = 1,
        offset_ticks: int = 0,
        chunk_size: float | None = None,
        max_chunks: int = 50,
        distribution: str = "uniform",
        escalation_after_seconds: float | None = 5.0,
        escalation_action: str = "marketable_ioc",
        escalation_widen_bps: float | None = None,
        max_escalations: int = 1,
        reprice_on_mid_drift_bps: float | None = None,
        order_type_mix: list | None = None,
        min_chunk_notional_usd: float = 1.0,
        min_reprice_interval_seconds: float = 1.0,
    ) -> Any:
        """Construct an ``ExitPolicy`` for a laddered, chunked exit.

        Polymarket-tuned defaults: 1¢ ticks, $1 min notional. Tune
        ``levels`` × ``step_ticks`` to your expected drawdown speed.

        Pass ``escalation_after_seconds=None`` to disable time-based
        escalation. Pass ``reprice_on_mid_drift_bps=None`` to disable
        mid-drift repricing (the ladder will rest at its initial prices).

        Returns:
            An ``ExitPolicy`` dataclass instance ready to attach to
            ``BaseStrategy.exit_policies`` or ``ExitDecision.exit_policy``.
        """
        from services.strategies.base import (
            EscalationSpec,
            ExitPolicy,
            LadderSpec,
        )

        ladder = LadderSpec(
            levels=int(max(1, levels)),
            step_ticks=int(max(0, step_ticks)),
            offset_ticks=int(max(0, offset_ticks)),
            distribution=str(distribution or "uniform"),
        )
        escalation = None
        if escalation_after_seconds is not None and float(escalation_after_seconds) > 0:
            escalation = EscalationSpec(
                after_seconds=float(escalation_after_seconds),
                action=str(escalation_action or "marketable_ioc"),
                widen_bps=(
                    float(escalation_widen_bps)
                    if escalation_widen_bps is not None
                    else None
                ),
                max_escalations=int(max(0, max_escalations)),
            )
        return ExitPolicy(
            ladder=ladder,
            chunk_size=(float(chunk_size) if chunk_size is not None and chunk_size > 0 else None),
            max_chunks=int(max(1, max_chunks)),
            order_type_mix=order_type_mix,
            escalation=escalation,
            reprice_on_mid_drift_bps=(
                float(reprice_on_mid_drift_bps)
                if reprice_on_mid_drift_bps is not None
                else None
            ),
            min_chunk_notional_usd=float(max(0.01, min_chunk_notional_usd)),
            min_reprice_interval_seconds=float(max(0.0, min_reprice_interval_seconds)),
        )

    @staticmethod
    def build_chunked_exit_policy(
        *,
        chunk_size: float,
        max_chunks: int = 50,
        escalation_after_seconds: float | None = None,
        escalation_action: str = "marketable_ioc",
        min_chunk_notional_usd: float = 1.0,
    ) -> Any:
        """Convenience: chunk-only policy (no price laddering).

        Splits the position into ``chunk_size`` contract slices at the
        trigger price. Useful when book depth is the constraint but the
        price level is correct.
        """
        return StrategySDK.build_ladder_exit_policy(
            levels=1,
            step_ticks=0,
            chunk_size=chunk_size,
            max_chunks=max_chunks,
            distribution="uniform",
            escalation_after_seconds=escalation_after_seconds,
            escalation_action=escalation_action,
            min_chunk_notional_usd=min_chunk_notional_usd,
        )

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
        if not _coerce_bool(cfg.get("enabled"), True):
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
    def normalize_trader_wallet_weights(
        value: Any,
        *,
        min_weight: float = 0.0,
        max_weight: float = 100.0,
    ) -> dict[str, float]:
        raw = value if isinstance(value, dict) else {}
        weights: dict[str, float] = {}
        floor = float(min_weight)
        ceiling = max(float(max_weight), floor)
        for raw_wallet, raw_weight in raw.items():
            wallet = StrategySDK.normalize_trader_wallet(raw_wallet)
            if not wallet:
                continue
            try:
                parsed_weight = float(raw_weight)
            except Exception:
                continue
            if parsed_weight != parsed_weight:
                continue
            weights[wallet] = max(floor, min(ceiling, parsed_weight))
        return weights

    @staticmethod
    def extract_trader_signal_wallets(signal: Any) -> set[str]:
        payload = signal if isinstance(signal, dict) else getattr(signal, "payload_json", None)
        if not isinstance(payload, dict):
            payload = {}

        wallets: set[str] = set()
        direct_wallet = StrategySDK.normalize_trader_wallet(payload.get("wallet_address") or payload.get("source_wallet"))
        if direct_wallet:
            wallets.add(direct_wallet)
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
        source_trade_payload = payload.get("source_trade")
        source_trade_payload = source_trade_payload if isinstance(source_trade_payload, dict) else {}
        source_wallet_payload = StrategySDK.normalize_trader_wallet(
            source_trade_payload.get("wallet_address") or source_trade_payload.get("source_wallet")
        )
        if source_wallet_payload:
            wallets.add(source_wallet_payload)

        strategy_context = payload.get("strategy_context")
        if isinstance(strategy_context, dict):
            copy_event = strategy_context.get("copy_event")
            copy_event = copy_event if isinstance(copy_event, dict) else {}
            copy_wallet = StrategySDK.normalize_trader_wallet(
                copy_event.get("wallet_address") or copy_event.get("source_wallet")
            )
            if copy_wallet:
                wallets.add(copy_wallet)
            source_trade = strategy_context.get("source_trade")
            source_trade = source_trade if isinstance(source_trade, dict) else {}
            source_wallet = StrategySDK.normalize_trader_wallet(
                source_trade.get("wallet_address") or source_trade.get("source_wallet")
            )
            if source_wallet:
                wallets.add(source_wallet)
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
    def extract_primary_trader_signal_wallet(signal: Any) -> str:
        payload = signal if isinstance(signal, dict) else getattr(signal, "payload_json", None)
        if not isinstance(payload, dict):
            payload = {}

        strategy_context = payload.get("strategy_context")
        strategy_context = strategy_context if isinstance(strategy_context, dict) else {}
        copy_event = strategy_context.get("copy_event")
        copy_event = copy_event if isinstance(copy_event, dict) else {}
        source_trade = payload.get("source_trade")
        source_trade = source_trade if isinstance(source_trade, dict) else {}

        candidates = (
            copy_event.get("wallet_address"),
            copy_event.get("source_wallet"),
            source_trade.get("wallet_address"),
            source_trade.get("source_wallet"),
            payload.get("wallet_address"),
            payload.get("source_wallet"),
        )
        for raw in candidates:
            wallet = StrategySDK.normalize_trader_wallet(raw)
            if wallet:
                return wallet

        wallets = sorted(StrategySDK.extract_trader_signal_wallets(signal))
        return wallets[0] if wallets else ""

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
    def trader_execution_policy_defaults() -> dict[str, Any]:
        return dict(StrategySDK.TRADER_EXECUTION_POLICY_DEFAULTS)

    @staticmethod
    def trader_execution_policy_config_schema() -> dict[str, Any]:
        return dict(StrategySDK.TRADER_EXECUTION_POLICY_CONFIG_SCHEMA)

    @staticmethod
    def validate_trader_execution_policy_config(config: Any) -> dict[str, Any]:
        cfg = dict(StrategySDK.TRADER_EXECUTION_POLICY_DEFAULTS)
        raw = config if isinstance(config, dict) else {}
        cfg.update({str(k): v for k, v in raw.items() if str(k) != "_schema"})

        selected = False
        for alias in (
            "allow_taker_limit_buy_above_signal",
            "allow_taker_limit_pay_up",
            "allow_taker_limit_to_exceed_signal_price",
            "allow_buy_above_signal_price",
        ):
            if alias in raw:
                cfg["allow_taker_limit_buy_above_signal"] = _coerce_bool(
                    raw.get(alias),
                    StrategySDK.TRADER_EXECUTION_POLICY_DEFAULTS["allow_taker_limit_buy_above_signal"],
                )
                selected = True
                break
        for alias in (
            "aggressive_limit_buy_submit_as_gtc",
            "submit_aggressive_buy_limits_as_gtc",
            "submit_taker_limit_buy_as_gtc",
        ):
            if alias in raw:
                cfg["aggressive_limit_buy_submit_as_gtc"] = _coerce_bool(
                    raw.get(alias),
                    StrategySDK.TRADER_EXECUTION_POLICY_DEFAULTS["aggressive_limit_buy_submit_as_gtc"],
                )
                break

        execution_policy = cfg.get("execution_policy")
        if not selected and isinstance(execution_policy, dict):
            for alias in (
                "allow_taker_limit_buy_above_signal",
                "allow_taker_limit_pay_up",
                "allow_taker_limit_to_exceed_signal_price",
                "allow_buy_above_signal_price",
            ):
                if alias in execution_policy:
                    cfg["allow_taker_limit_buy_above_signal"] = _coerce_bool(
                        execution_policy.get(alias),
                        StrategySDK.TRADER_EXECUTION_POLICY_DEFAULTS["allow_taker_limit_buy_above_signal"],
                    )
                    break
            for alias in (
                "aggressive_limit_buy_submit_as_gtc",
                "submit_aggressive_buy_limits_as_gtc",
                "submit_taker_limit_buy_as_gtc",
            ):
                if alias in execution_policy:
                    cfg["aggressive_limit_buy_submit_as_gtc"] = _coerce_bool(
                        execution_policy.get(alias),
                        StrategySDK.TRADER_EXECUTION_POLICY_DEFAULTS["aggressive_limit_buy_submit_as_gtc"],
                    )
                    break

        cfg["allow_taker_limit_buy_above_signal"] = _coerce_bool(
            cfg.get("allow_taker_limit_buy_above_signal"),
            StrategySDK.TRADER_EXECUTION_POLICY_DEFAULTS["allow_taker_limit_buy_above_signal"],
        )
        cfg["aggressive_limit_buy_submit_as_gtc"] = _coerce_bool(
            cfg.get("aggressive_limit_buy_submit_as_gtc"),
            StrategySDK.TRADER_EXECUTION_POLICY_DEFAULTS["aggressive_limit_buy_submit_as_gtc"],
        )
        return cfg

    @staticmethod
    def allow_taker_limit_buy_above_signal_price(params: Any, *, default: bool = False) -> bool:
        cfg = params if isinstance(params, dict) else {}
        normalized = StrategySDK.validate_trader_execution_policy_config(cfg)
        return _coerce_bool(normalized.get("allow_taker_limit_buy_above_signal"), default)

    @staticmethod
    def aggressive_limit_buy_submit_as_gtc(params: Any, *, default: bool = False) -> bool:
        cfg = params if isinstance(params, dict) else {}
        normalized = StrategySDK.validate_trader_execution_policy_config(cfg)
        return _coerce_bool(normalized.get("aggressive_limit_buy_submit_as_gtc"), default)

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
        return _coerce_bool(raw, default_enabled)

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
        cfg["shadow_min_order_size_usd"] = StrategySDK._coerce_float(
            cfg.get("shadow_min_order_size_usd", cfg.get("paper_min_order_size_usd")),
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
        cfg["firehose_require_active_signal"] = _coerce_bool(
            cfg.get("firehose_require_active_signal"),
            True,
        )
        cfg["firehose_require_tradable_market"] = _coerce_bool(
            cfg.get("firehose_require_tradable_market"),
            False,
        )
        cfg["firehose_exclude_crypto_markets"] = _coerce_bool(
            cfg.get("firehose_exclude_crypto_markets"),
            False,
        )
        cfg["firehose_require_qualified_source"] = _coerce_bool(
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
        schedule_enabled = _coerce_bool(schedule.get("enabled"), False)
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
            cfg.get("max_position_notional_usd"), 5.0, 1.0, 10_000_000.0
        )
        cfg["max_gross_exposure_usd"] = StrategySDK._coerce_float(
            cfg.get("max_gross_exposure_usd"), 2000.0, 1.0, 100_000_000.0
        )
        cfg["max_trade_notional_usd"] = StrategySDK._coerce_float(
            cfg.get("max_trade_notional_usd"), 5.0, 1.0, 10_000_000.0
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
        cfg["allow_averaging"] = _coerce_bool(cfg.get("allow_averaging"), False)
        cfg["halt_on_consecutive_losses"] = _coerce_bool(cfg.get("halt_on_consecutive_losses"), True)
        cfg["max_consecutive_losses"] = StrategySDK._coerce_int(cfg.get("max_consecutive_losses"), 4, 0, 1000)
        cfg["max_entry_drift_pct"] = StrategySDK._coerce_float(
            cfg.get("max_entry_drift_pct"), 10.0, 0.0, 100.0
        )
        raw_max_md_age = cfg.get("max_market_data_age_ms")
        if raw_max_md_age is None or (isinstance(raw_max_md_age, str) and not raw_max_md_age.strip()):
            cfg["max_market_data_age_ms"] = None
        else:
            cfg["max_market_data_age_ms"] = StrategySDK._coerce_int(raw_max_md_age, 10000, 50, 300_000)
        cfg["allow_taker_limit_buy_above_signal"] = _coerce_bool(
            cfg.get("allow_taker_limit_buy_above_signal"), False
        )
        default_portfolio = StrategySDK.TRADER_RISK_DEFAULTS.get("portfolio")
        portfolio_cfg = dict(default_portfolio) if isinstance(default_portfolio, dict) else {}
        raw_portfolio = cfg.get("portfolio")
        if isinstance(raw_portfolio, dict):
            portfolio_cfg.update({str(k): v for k, v in raw_portfolio.items() if str(k)})
        cfg["portfolio"] = {
            "enabled": _coerce_bool(portfolio_cfg.get("enabled"), False),
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

        def _as_float(value: Any, default: float = 0.0) -> float:
            try:
                return float(value)
            except Exception:
                return default

        bid = _as_float(data.get("best_bid", data.get("bid", 0.0)), 0.0)
        ask = _as_float(data.get("best_ask", data.get("ask", 0.0)), 0.0)
        mid = _as_float(data.get("mid", 0.0), 0.0)
        if mid <= 0.0 and bid > 0.0 and ask > 0.0:
            mid = (bid + ask) / 2.0
        if mid <= 0.0 or bid <= 0.0 or ask <= 0.0:
            return None
        return ((ask - bid) / mid) * 10000

    # ------------------------------------------------------------------
    # Realistic-fill pricing
    # ------------------------------------------------------------------
    @staticmethod
    def get_ask_priced_fill(
        market: Any,
        prices: dict[str, dict],
        side: str = "YES",
        *,
        spread_cushion: float = 0.005,
    ) -> tuple[float, float, bool]:
        """Return ``(fill_price, mid_price, ask_known)`` for buying ``side``.

        ``fill_price`` is the price a market-taker BUY would actually pay:

        - If a live ask is available, returns the ask.
        - If only a bid is available, infers ``ask = max(mid, 2*mid - bid)``.
        - If neither is available, falls back to ``mid + spread_cushion``
          (default 0.005 = half a cent) so a strategy never assumes a
          mid-priced fill it can't realistically achieve.

        Use this for any strategy that:
          - Builds a multi-leg bundle whose profit margin can be eaten by
            spread costs (negrisk arb, multi-outcome bundles).
          - Needs realistic capital efficiency on a directional bet rather
            than an idealized mid-priced ROI.

        The mid-price trap killed a number of negrisk-short opportunities
        in production where a 1-2¢ ask premium per leg silently consumed
        a $0.30 gross profit. Strategies should call this helper rather
        than reading ``prices[token]["mid"]`` directly.

        Args:
            market: Market object with ``clob_token_ids`` and ``yes_price`` /
                ``no_price`` attributes.
            prices: The prices dict passed to detect().
            side: 'YES' or 'NO' — which side this strategy is buying.
            spread_cushion: Per-leg cushion added to mid when neither bid
                nor ask is available. Half a cent is conservative; raise
                for thinner / less-liquid markets.

        Returns:
            ``(fill_price, mid_price, ask_known)``. ``ask_known`` is True
            when the live ask was used (highest confidence in the fill cost).
        """
        side_upper = side.upper()
        idx = 0 if side_upper == "YES" else 1
        token_ids = getattr(market, "clob_token_ids", None) or []
        # Fallback price comes from the market's static yes_price/no_price.
        if side_upper == "YES":
            mid = float(getattr(market, "yes_price", 0.0) or 0.0)
        else:
            mid = float(getattr(market, "no_price", 0.0) or 0.0)
        ask: float | None = None
        bid: float | None = None
        if len(token_ids) > idx:
            payload = prices.get(token_ids[idx]) if isinstance(prices, dict) else None
            if isinstance(payload, dict):
                mid_raw = payload.get("mid")
                if isinstance(mid_raw, (int, float)) and mid_raw > 0:
                    mid = float(mid_raw)
                ask_raw = payload.get("best_ask", payload.get("ask"))
                if isinstance(ask_raw, (int, float)) and ask_raw > 0:
                    ask = float(ask_raw)
                bid_raw = payload.get("best_bid", payload.get("bid"))
                if isinstance(bid_raw, (int, float)) and bid_raw > 0:
                    bid = float(bid_raw)
        if ask is not None:
            return ask, mid, True
        if bid is not None and bid > 0:
            inferred = max(mid, 2.0 * mid - bid)
            return inferred, mid, False
        return mid + max(0.0, spread_cushion), mid, False

    @staticmethod
    def get_bid_priced_exit(
        market: Any,
        prices: dict[str, dict],
        side: str = "YES",
        *,
        spread_cushion: float = 0.005,
    ) -> tuple[float, float, bool]:
        """Return ``(exit_price, mid_price, bid_known)`` for selling ``side``.

        Mirror of :meth:`get_ask_priced_fill` for the exit side: a
        market-taker SELL receives the bid. When no bid is observable
        infers ``bid = min(mid, 2*mid - ask)``; when neither side is
        available falls back to ``mid - spread_cushion``.

        Use this when projecting the realized P&L of an exit, not the
        theoretical mid-priced one.
        """
        side_upper = side.upper()
        idx = 0 if side_upper == "YES" else 1
        token_ids = getattr(market, "clob_token_ids", None) or []
        if side_upper == "YES":
            mid = float(getattr(market, "yes_price", 0.0) or 0.0)
        else:
            mid = float(getattr(market, "no_price", 0.0) or 0.0)
        ask: float | None = None
        bid: float | None = None
        if len(token_ids) > idx:
            payload = prices.get(token_ids[idx]) if isinstance(prices, dict) else None
            if isinstance(payload, dict):
                mid_raw = payload.get("mid")
                if isinstance(mid_raw, (int, float)) and mid_raw > 0:
                    mid = float(mid_raw)
                ask_raw = payload.get("best_ask", payload.get("ask"))
                if isinstance(ask_raw, (int, float)) and ask_raw > 0:
                    ask = float(ask_raw)
                bid_raw = payload.get("best_bid", payload.get("bid"))
                if isinstance(bid_raw, (int, float)) and bid_raw > 0:
                    bid = float(bid_raw)
        if bid is not None:
            return bid, mid, True
        if ask is not None and ask > 0:
            inferred = min(mid, 2.0 * mid - ask)
            return max(0.01, inferred), mid, False
        return max(0.01, mid - max(0.0, spread_cushion)), mid, False

    # ------------------------------------------------------------------
    # Conviction / confidence math (broadly applicable to ensemble strategies)
    # ------------------------------------------------------------------
    @staticmethod
    def confidence_from_signal_agreement(
        signals: dict[str, float],
        *,
        edge_sign: float,
        edge_magnitude: float,
        magnitude_full_scale: float = 0.15,
        signal_zero_threshold: float = 0.05,
        floor: float = 0.30,
        ceiling: float = 0.85,
        agreement_weight: float = 0.6,
    ) -> tuple[float, int, int]:
        """Derive a [floor, ceiling] confidence from ensemble signal agreement.

        For any strategy that combines multiple weak signals into a
        composite edge, this collapses two questions into a single
        confidence number:

        1. **Signal agreement** — what fraction of non-zero signals point
           in the same direction as the composite edge?
        2. **Edge magnitude** — how big is the composite edge relative to
           ``magnitude_full_scale`` (the largest plausible adjustment for
           the strategy)?

        Final score = ``floor + (ceiling - floor) * (agreement_weight ×
        agreement_ratio + (1 - agreement_weight) × magnitude_factor)``.

        Returns ``(confidence, agreeing_count, voting_count)``. Returns
        ``(floor, 0, 0)`` when the signals dict is empty or no signal
        crosses ``signal_zero_threshold``.

        Usage::

            conf, agree, voting = StrategySDK.confidence_from_signal_agreement(
                signal_breakdown,
                edge_sign=1.0 if edge > 0 else -1.0,
                edge_magnitude=abs(edge),
            )

        Args:
            signals: Mapping ``{signal_name: signed_value}`` where the sign
                indicates direction (positive = buy YES, negative = buy NO)
                and magnitude indicates strength.
            edge_sign: +1.0 / -1.0 / 0.0 — direction of the composite edge.
            edge_magnitude: Absolute composite edge in the same units as
                ``magnitude_full_scale``.
            magnitude_full_scale: Edge magnitude that maps to a full
                magnitude score of 1.0 (clipped above).
            signal_zero_threshold: Signals with ``|value| <`` this are
                ignored when counting agreement.
            floor: Minimum confidence to claim, even when nothing agrees.
            ceiling: Maximum confidence to claim, even on a unanimous vote.
            agreement_weight: Weight given to the agreement ratio versus
                edge magnitude in the blended score (0..1).
        """
        if not isinstance(signals, dict) or not signals:
            return floor, 0, 0
        if edge_sign == 0.0:
            return floor, 0, 0

        agreeing = 0
        disagreeing = 0
        for value in signals.values():
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if abs(v) < signal_zero_threshold:
                continue
            if (v > 0) == (edge_sign > 0):
                agreeing += 1
            else:
                disagreeing += 1

        voting = agreeing + disagreeing
        if voting == 0:
            return floor, 0, 0

        agreement_ratio = agreeing / voting
        magnitude_factor = max(0.0, min(1.0, edge_magnitude / max(1e-6, magnitude_full_scale)))
        w = max(0.0, min(1.0, agreement_weight))
        blended = w * agreement_ratio + (1.0 - w) * magnitude_factor
        confidence = floor + (ceiling - floor) * blended
        return max(floor, min(ceiling, confidence)), agreeing, voting

    # ------------------------------------------------------------------
    # Sizing (Kelly-fraction-from-edge, broadly applicable to directional bets)
    # ------------------------------------------------------------------
    @staticmethod
    def fractional_kelly_size(
        base_size: float,
        max_size: float,
        *,
        edge_price: float,
        confidence: float,
        risk_score: float,
        kelly_fraction: float = 0.25,
        kelly_size_multiplier: float = 4.0,
        confidence_floor: float = 0.5,
        risk_floor: float = 0.4,
    ) -> float:
        """Quarter-Kelly position sizing from a price-space edge.

        For any directional binary-market strategy. Approximates the Kelly
        fraction as ``f* = edge_price / (1 - edge_price)`` (the optimal
        fraction for a 1:1 binary contract with edge ``edge_price``), then:

          1. Caps to ``kelly_fraction`` (default 0.25 = quarter-Kelly) so a
             model error doesn't overstake.
          2. Scales the base size by ``1 + kelly_size_multiplier × f``.
          3. Multiplies by a confidence factor in
             ``[confidence_floor, 1.0]`` so high-conviction signals size
             larger.
          4. Multiplies by a risk-damping factor in
             ``[risk_floor, 1.0 - risk_score]`` so high-risk signals size
             smaller.
          5. Clips to ``[1.0, max_size]``.

        This replaces the older p_estimated = 0.5 + edge/200 hack which
        structurally assumed a $0.50 market price regardless of the
        actual contract being traded.

        Args:
            base_size: Trader's per-signal base notional (USD).
            max_size: Trader's per-signal cap (USD).
            edge_price: Price-space edge in [0, 1] — e.g. 0.09 for a 9¢
                disagreement between fair value and market.
            confidence: [0, 1] strategy-emitted conviction.
            risk_score: [0, 1] strategy-emitted risk score (lower = safer).
            kelly_fraction: Maximum Kelly fraction to commit (quarter-Kelly
                = 0.25 is a standard cap).
            kelly_size_multiplier: Multiplier applied to the capped Kelly
                fraction when scaling base_size. Higher values let
                higher-edge signals scale up more aggressively.
            confidence_floor: Minimum confidence damping factor.
            risk_floor: Minimum risk damping factor.

        Returns:
            Position size in USD, clipped to ``[1.0, max_size]``.
        """
        edge_price = max(0.0, min(0.99, abs(float(edge_price))))
        f_star = edge_price / max(1e-3, 1.0 - edge_price)
        f_capped = max(0.0, min(float(kelly_fraction), f_star * float(kelly_fraction)))
        conf_scale = max(float(confidence_floor),
                         min(1.0, float(confidence) + 0.15))
        risk_scale = max(float(risk_floor), 1.0 - float(risk_score))
        size = float(base_size) * (1.0 + float(kelly_size_multiplier) * f_capped) * conf_scale * risk_scale
        return max(1.0, min(float(max_size), size))

    # ------------------------------------------------------------------
    # Category-level resolution rates (broadly applicable to any
    # base-rate-aware strategy)
    # ------------------------------------------------------------------

    # Path to the rates JSON. Refreshed periodically by a background job
    # (scripts/refresh_category_yes_rates.py) which queries the DB and
    # writes the file. The SDK reads it lazily — no DB dependency from
    # synchronous detect() paths.
    _CATEGORY_RATES_PATH = (
        Path(__file__).resolve().parents[1] / "data" / "category_yes_rates.json"
    )
    _category_yes_rates_cache: dict[str, float] | None = None
    _category_yes_rates_cache_at: float = 0.0
    _CATEGORY_RATES_TTL_SECONDS = 600.0

    # Hardcoded baseline. Only the universal ``_default`` is set — every
    # category-specific number must come from the JSON file produced by
    # ``scripts/refresh_category_yes_rates.py`` (which queries the actual
    # DB for terminal outcome_prices on closed binary markets).
    #
    # The 0.45 default is a slight YES-skeptical prior: a randomly-chosen
    # binary prediction-market question is more often a "Will X happen?"
    # framing where most ambitious propositions don't resolve YES. It's
    # the same default the legacy implementation used. Override per-call
    # via the ``defaults`` parameter when needed.
    _CATEGORY_YES_RATES_DEFAULTS: dict[str, float] = {
        "_default": 0.45,
    }

    @classmethod
    def get_category_yes_resolve_rates(
        cls,
        *,
        defaults: Optional[dict[str, float]] = None,
    ) -> dict[str, float]:
        """Per-category YES-resolution rate for binary markets.

        Returns a mapping of ``{category_lower: rate}`` where ``rate`` is
        the empirical fraction of resolved binary markets in that category
        whose YES side won.

        **Binary markets only**: a category that contains multi-outcome
        bundles (NegRisk events with 5 candidates, etc.) will see each
        individual binary market counted separately — exactly one of the
        bundle's markets resolves YES, so the per-market YES rate is
        approximately ``1/N`` in such categories. This is the rate any
        single-market base-rate signal needs anyway.

        Lookup order:
          1. JSON cache file (refreshed offline by
             ``scripts/refresh_category_yes_rates.py``).
          2. Hardcoded empirical defaults.
          3. Caller-supplied ``defaults`` dict.

        ``defaults`` is overlaid LAST so caller intent wins.

        Always returns a ``_default`` key for categories not otherwise
        covered. Strategies that depend on category base rates should
        call this rather than hardcoding constants:

            rates = StrategySDK.get_category_yes_resolve_rates()
            rate = rates.get((category or "").lower(), rates["_default"])
        """
        now = time.time()
        cached = cls._category_yes_rates_cache
        if (
            cached is not None
            and (now - cls._category_yes_rates_cache_at) < cls._CATEGORY_RATES_TTL_SECONDS
        ):
            base = dict(cached)
            if defaults:
                base.update(defaults)
            return base

        # Start from the hardcoded baseline.
        rates = dict(cls._CATEGORY_YES_RATES_DEFAULTS)

        # Overlay any JSON-cached rates written by the refresh script.
        try:
            blob = cls.load_json_from_disk(str(cls._CATEGORY_RATES_PATH))
            if isinstance(blob, dict):
                payload = blob.get("rates") if "rates" in blob else blob
                if isinstance(payload, dict):
                    for cat, rate in payload.items():
                        try:
                            r = float(rate)
                        except (TypeError, ValueError):
                            continue
                        if 0.0 < r < 1.0:
                            rates[str(cat).strip().lower()] = r
        except Exception:
            pass

        cls._category_yes_rates_cache = dict(rates)
        cls._category_yes_rates_cache_at = now

        if defaults:
            rates.update(defaults)
        return rates

    @staticmethod
    def is_ws_feed_started() -> bool:
        """Return True when this process has an active FeedManager."""
        try:
            from services.market_runtime import get_market_runtime

            return bool(get_market_runtime().started)
        except Exception:
            return False

    @staticmethod
    def get_ws_mid_price(token_id: str) -> Optional[float]:
        """Get the real-time WebSocket mid price for a token.

        Args:
            token_id: The CLOB token ID.

        Returns:
            Mid price or None if not fresh.
        """
        try:
            from services.market_runtime import get_market_runtime

            return get_market_runtime().get_token_mid_price(str(token_id or "").strip())
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
            from services.market_runtime import get_market_runtime

            return get_market_runtime().get_token_spread_bps(str(token_id or "").strip())
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
            from services.reference_runtime import get_reference_runtime

            payload = get_reference_runtime().get_oracle_price(asset.upper())
            if isinstance(payload, dict) and payload.get("price") is not None:
                return float(payload["price"])
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
    #
    # Both methods read the live orderbook from the WebSocket-fed PriceCache
    # in services.ws_feeds.FeedManager. The CLOB WS handler in ws_feeds.py
    # ingests `book` snapshots and `price_change` deltas in real time
    # (measured ~176ms p50 send-to-receive); calls here are O(1) reads
    # against that in-process cache.
    #
    # Token must already be subscribed (FeedManager.polymarket_feed.subscribe).
    # Strategies that operate on transient markets should subscribe once on
    # discovery; the global runtime keeps the subscription alive until
    # eviction.

    @staticmethod
    def _resolve_token_id(market: Any, side: str) -> Optional[str]:
        """Return the CLOB token id for the requested side, or None.

        Side semantics: YES → index 0, NO → index 1 (Polymarket convention
        for binary markets). Accepts either a Market-shaped object (uses
        ``getattr``) or a raw dict (uses ``.get``) — strategies that work
        from worker payloads commonly pass dicts; opportunity strategies
        pass typed Market objects.

        Returns None when the market lacks token ids for the requested
        side; logs at debug since this is usually a not-yet-loaded market
        rather than a bug.
        """
        if isinstance(market, dict):
            token_ids = market.get("clob_token_ids") or []
            market_label = market.get("id") or market.get("condition_id") or "?"
        else:
            token_ids = getattr(market, "clob_token_ids", None) or []
            market_label = getattr(market, "id", "?")
        token_idx = 0 if str(side).upper() == "YES" else 1
        if len(token_ids) <= token_idx:
            logger.debug(
                "StrategySDK orderbook lookup: market %s has no token id for side=%s",
                market_label, side,
            )
            return None
        return str(token_ids[token_idx])

    @staticmethod
    def get_order_book_depth(
        market: Any,
        side: str = "YES",
        size_usd: float = 100.0,
    ) -> Optional[dict]:
        """Estimate execution cost for a target buy size on one side.

        Walks the live ask book from the PriceCache and runs it through
        VWAPCalculator (services.optimization.vwap) to produce VWAP,
        slippage, and fill probability for ``size_usd`` of buying pressure.

        Args:
            market: A Market object with ``clob_token_ids``.
            side: 'YES' or 'NO' — which side's ask book to walk.
            size_usd: Notional buy size in USD.

        Returns:
            Dict with execution estimate, or None when the orderbook is
            unavailable (token not subscribed, cache empty, or feed down):
            {
                "vwap_price": float,        # weighted avg fill price
                "slippage_bps": float,      # slippage from best ask in bps
                "fill_probability": float,  # 0..1
                "available_liquidity": float,  # USD of book reachable
                "is_fresh": bool,           # PriceCache staleness check
                "staleness_ms": float,      # age of the cached book
            }
        """
        token_id = StrategySDK._resolve_token_id(market, side)
        if token_id is None:
            return None
        try:
            from services.ws_feeds import get_feed_manager
            from services.optimization.vwap import VWAPCalculator
        except ImportError as e:
            logger.error("StrategySDK orderbook deps unavailable: %s", e)
            return None

        cache = get_feed_manager().cache
        order_book = cache.get_order_book(token_id)
        if order_book is None:
            logger.debug(
                "StrategySDK.get_order_book_depth: no cached book for token %s "
                "(market %s, side %s) — token may not be subscribed",
                token_id[:18], getattr(market, "id", "?"), side,
            )
            now_s = time.monotonic()
            last = _rest_prime_last_ts.get(token_id, 0.0)
            if now_s - last >= _REST_PRIME_COOLDOWN_S:
                _rest_prime_last_ts[token_id] = now_s
                _prime_book_via_rest(token_id)
            return None

        try:
            result = VWAPCalculator().calculate_buy_vwap(order_book, size_usd)
        except Exception as e:
            logger.exception(
                "StrategySDK.get_order_book_depth: VWAP failed for token %s: %s",
                token_id[:18], e,
            )
            return None

        staleness_s = cache.staleness(token_id) or 0.0
        return {
            "vwap_price": float(result.vwap),
            "slippage_bps": float(result.slippage_bps),
            "fill_probability": float(result.fill_probability),
            "available_liquidity": float(result.total_available),
            "is_fresh": cache.is_fresh(token_id),
            "staleness_ms": round(staleness_s * 1000.0, 1),
        }

    @staticmethod
    def get_book_levels(
        market: Any,
        side: str = "YES",
        max_levels: int = 10,
    ) -> Optional[list[dict]]:
        """Read raw orderbook levels for a market side from the live cache.

        Returns the *bids* side when ``side="YES"`` would be reading the
        YES token's book — but here we follow the historical SDK contract
        which returns *asks* (i.e. the side a buyer would lift). This
        matches what callers reading "what's available to buy at this
        side" expect.

        Args:
            market: A Market object with ``clob_token_ids``.
            side: 'YES' or 'NO'.
            max_levels: Cap on returned levels.

        Returns:
            List of ``{"price": float, "size": float}`` sorted from best
            (lowest ask) outward, or None when the book is unavailable.
        """
        token_id = StrategySDK._resolve_token_id(market, side)
        if token_id is None:
            return None
        try:
            from services.ws_feeds import get_feed_manager
        except ImportError as e:
            logger.error("StrategySDK orderbook deps unavailable: %s", e)
            return None

        cache = get_feed_manager().cache
        order_book = cache.get_order_book(token_id)
        if order_book is None:
            logger.debug(
                "StrategySDK.get_book_levels: no cached book for token %s "
                "(market %s, side %s) — token may not be subscribed",
                token_id[:18], getattr(market, "id", "?"), side,
            )
            now_s = time.monotonic()
            last = _rest_prime_last_ts.get(token_id, 0.0)
            if now_s - last >= _REST_PRIME_COOLDOWN_S:
                _rest_prime_last_ts[token_id] = now_s
                _prime_book_via_rest(token_id)
            return None

        levels = order_book.asks[: max(0, int(max_levels))]
        return [{"price": float(lvl.price), "size": float(lvl.size)} for lvl in levels]

    @staticmethod
    def get_best_bid_ask(market: Any, side: str = "YES") -> Optional[tuple[float, float]]:
        """Return ``(best_bid, best_ask)`` for one side from the live cache.

        Returns None when the token isn't subscribed or the cache is empty.
        """
        token_id = StrategySDK._resolve_token_id(market, side)
        if token_id is None:
            return None
        try:
            from services.ws_feeds import get_feed_manager
        except ImportError as e:
            logger.error("StrategySDK orderbook deps unavailable: %s", e)
            return None
        return get_feed_manager().cache.get_best_bid_ask(token_id)

    @staticmethod
    def get_book_imbalance(
        market: Any,
        side: str = "YES",
        levels: int = 5,
    ) -> Optional[float]:
        """Compute live book imbalance from the WS-fed cache.

        Returns ``(bid_size_top_N - ask_size_top_N) / (bid_size_top_N + ask_size_top_N)``
        in [-1, +1] using the top ``levels`` price levels of each side.
        Positive values mean a heavier bid side (buy-pressure leaning),
        negative values mean a heavier ask side. None when the book is
        unavailable or both sides are empty.

        Computed against live data — preferred over scanner-emitted
        ``orderbook_imbalance`` payload fields, which are stale by one
        scanner cycle (typically 1-5s). Sub-200ms freshness via the WS
        cache.
        """
        token_id = StrategySDK._resolve_token_id(market, side)
        if token_id is None:
            return None
        try:
            from services.ws_feeds import get_feed_manager
        except ImportError as e:
            logger.error("StrategySDK orderbook deps unavailable: %s", e)
            return None

        order_book = get_feed_manager().cache.get_order_book(token_id)
        if order_book is None:
            return None
        n = max(1, int(levels))
        bid_size = sum(float(lvl.size) for lvl in order_book.bids[:n])
        ask_size = sum(float(lvl.size) for lvl in order_book.asks[:n])
        denom = bid_size + ask_size
        if denom <= 0.0:
            return None
        return (bid_size - ask_size) / denom

    # ── Market lifecycle ───────────────────────────────────────

    @staticmethod
    def _coerce_end_ts_ms(value: Any) -> Optional[int]:
        """Best-effort extract a resolution-time epoch-millis int from
        a Market ORM object, a market dict, an ISO datetime string, a
        ``datetime``, or a raw int/float.
        """
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
        if isinstance(value, datetime):
            try:
                return int(value.timestamp() * 1000.0)
            except (OSError, OverflowError, ValueError):
                return None
        # Pydantic / ORM Market — try common attribute names
        for attr in ("end_date", "endDate", "end_time", "end_ts_ms", "end_ms"):
            if hasattr(value, attr):
                inner = getattr(value, attr)
                if inner is not None and inner is not value:
                    return StrategySDK._coerce_end_ts_ms(inner)
        if isinstance(value, dict):
            for key in ("end_date", "endDate", "end_time", "end_ts_ms", "end_ms"):
                v = value.get(key)
                if v is not None:
                    return StrategySDK._coerce_end_ts_ms(v)
            return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                from services.strategy_helpers.crypto_strategy_utils import (
                    parse_datetime_utc,
                )

                dt = parse_datetime_utc(text)
                if dt is not None:
                    return int(dt.timestamp() * 1000.0)
            except Exception:
                pass
            try:
                return int(float(text))
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def time_to_resolution(
        market_or_end_ts_ms: Any,
        now_ms: Optional[int] = None,
    ) -> Optional[float]:
        """Seconds until the market resolves, or None if unknown.

        Universal across market types — crypto, election, sports,
        weather, anything with a parseable resolution timestamp. For
        open-ended markets (no ``end_date``) or markets whose end time
        can't be parsed, returns ``None``.

        Args:
            market_or_end_ts_ms: A Market ORM/Pydantic object (with
                ``.end_date``), a market dict (``end_date`` /
                ``endDate`` / ``end_time``), an ISO datetime string,
                a ``datetime``, or a raw epoch-millis int/float.
            now_ms: Wall-clock millis. Defaults to ``time.time() * 1000``.

        Returns:
            Float seconds until resolution. Negative when the market
            is past its resolution time. ``None`` when end_date is
            absent or unparseable.
        """
        end_ms = StrategySDK._coerce_end_ts_ms(market_or_end_ts_ms)
        if end_ms is None:
            return None
        now = int(now_ms) if now_ms is not None else int(time.time() * 1000)
        return (end_ms - now) / 1000.0

    # ── Binary fair-value math ─────────────────────────────────

    @staticmethod
    def prob_above(
        value: float,
        threshold: float,
        vol_per_sec: float,
        seconds_remaining: float,
    ) -> Optional[float]:
        """P(underlying ends > threshold) under zero-drift GBM.

        Cash-or-nothing digital fair value for a binary "above" market
        (e.g. "will BTC be above $100k at 4pm?"). Useful for *any*
        binary market resolving on a continuous underlying — crypto
        over $K, sports total over N, temperature over X°.

        Math: under GBM with zero drift,
            P(S_T > K) = N(d2),  d2 = ln(S/K) / (sigma * sqrt(T))
        where N is the standard normal CDF (computed via
        ``math.erf`` — no scipy dependency).

        Args:
            value: Current underlying value (S). Must be > 0 finite.
            threshold: Resolution threshold (K). Must be > 0 finite.
            vol_per_sec: Per-second volatility (sigma). Must be >= 0
                finite. Express as a unitless return per second
                (e.g. 0.0001 for 1 bps/sec).
            seconds_remaining: Time to resolution (T). Must be >= 0
                finite.

        Returns:
            Probability in [0.0, 1.0]. ``None`` if any input is
            invalid (NaN, infinite, or out-of-range).

            When ``T == 0`` or ``sigma == 0`` the math is degenerate;
            returns 1.0 if ``value > threshold``, 0.0 if
            ``value < threshold``, 0.5 if equal.
        """
        try:
            s = float(value)
            k = float(threshold)
            sigma = float(vol_per_sec)
            t = float(seconds_remaining)
        except (TypeError, ValueError):
            return None
        if not (
            math.isfinite(s)
            and math.isfinite(k)
            and math.isfinite(sigma)
            and math.isfinite(t)
        ):
            return None
        if s <= 0.0 or k <= 0.0 or sigma < 0.0 or t < 0.0:
            return None
        if t == 0.0 or sigma == 0.0:
            if s > k:
                return 1.0
            if s < k:
                return 0.0
            return 0.5
        d2 = math.log(s / k) / (sigma * math.sqrt(t))
        return 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))

    @staticmethod
    def prob_below(
        value: float,
        threshold: float,
        vol_per_sec: float,
        seconds_remaining: float,
    ) -> Optional[float]:
        """P(underlying ends < threshold). Equals ``1 - prob_above``.

        See :meth:`prob_above` for argument semantics. Returns ``None``
        on invalid input. At ``T == 0`` and ``S == K`` returns 0.5
        (mirrors prob_above's tie-breaking convention).
        """
        p_up = StrategySDK.prob_above(value, threshold, vol_per_sec, seconds_remaining)
        if p_up is None:
            return None
        return 1.0 - p_up

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
            from services.market_runtime import get_market_runtime

            return get_market_runtime().get_price_history(
                str(token_id or "").strip(),
                max_snapshots=max_snapshots,
            )
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
            snapshots = StrategySDK.get_price_history(token_id, max_snapshots=max(20, int(lookback_seconds)))
            if snapshots:
                cutoff_ms = (time.time() - max(1, int(lookback_seconds))) * 1000.0
                normalized: list[tuple[float, float]] = []
                for snapshot in snapshots:
                    if not isinstance(snapshot, dict):
                        continue
                    ts_raw = snapshot.get("timestamp") or snapshot.get("t")
                    mid_raw = snapshot.get("mid") or snapshot.get("p")
                    try:
                        ts = float(ts_raw)
                    except (TypeError, ValueError):
                        ts = None
                    try:
                        mid = float(mid_raw)
                    except (TypeError, ValueError):
                        mid = None
                    if ts is None or mid is None:
                        continue
                    if ts < 10_000_000_000:
                        ts *= 1000.0
                    normalized.append((ts, mid))
                if normalized:
                    normalized.sort(key=lambda item: item[0])
                    current_ts, current_mid = normalized[-1]
                    prior_mid = normalized[0][1]
                    for ts, mid in normalized:
                        if ts >= cutoff_ms:
                            prior_mid = mid
                            break
                    change_abs = current_mid - prior_mid
                    return {
                        "current_mid": current_mid,
                        "prior_mid": prior_mid,
                        "change_abs": change_abs,
                        "change_pct": ((change_abs / prior_mid) * 100.0) if prior_mid > 0 else None,
                        "snapshots_in_window": len(normalized),
                    }
            from services.market_runtime import get_market_runtime

            return get_market_runtime().get_price_change(
                str(token_id or "").strip(),
                lookback_seconds=lookback_seconds,
            )
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
            from services.market_runtime import get_market_runtime

            return get_market_runtime().get_recent_trades(
                str(token_id or "").strip(),
                max_trades=max_trades,
            )
        except Exception:
            return []

    @staticmethod
    def get_trade_volume(token_id: str, lookback_seconds: float = 300.0) -> dict:
        """Return buy/sell volume over a lookback window.

        Returns dict with keys: buy_volume, sell_volume, total, trade_count.
        Returns zero-volume dict if trade data is unavailable.
        """
        try:
            from services.market_runtime import get_market_runtime

            return get_market_runtime().get_trade_volume(
                str(token_id or "").strip(),
                lookback_seconds=lookback_seconds,
            )
        except Exception:
            return {"buy_volume": 0.0, "sell_volume": 0.0, "total": 0.0, "trade_count": 0}

    @staticmethod
    def get_buy_sell_imbalance(token_id: str, lookback_seconds: float = 300.0) -> float:
        """Return buy/sell order flow imbalance in [-1, 1].

        +1 = all buys, -1 = all sells, 0 = balanced or no data.
        """
        try:
            from services.market_runtime import get_market_runtime

            return get_market_runtime().get_buy_sell_imbalance(
                str(token_id or "").strip(),
                lookback_seconds=lookback_seconds,
            )
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
