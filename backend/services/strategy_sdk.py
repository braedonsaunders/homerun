"""
Strategy SDK — high-level helpers for custom strategy development.

This module provides a clean, stable API surface for strategy authors.
Import it in your strategy code with:

    from services.strategy_sdk import StrategySDK

All methods are designed to be safe and handle missing data gracefully.
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from pathlib import Path
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

    _CONFIG_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]{1,48}[a-z0-9]$")
    TRADER_TIER_CANONICAL = ("low", "medium", "high", "extreme")
    TRADER_SIDE_CANONICAL = ("all", "buy", "sell")
    TRADER_SOURCE_SCOPE_CANONICAL = ("all", "tracked", "pool")
    TRADER_FILTER_DEFAULTS: dict[str, Any] = {
        "min_confidence": 0.45,
        "min_tier": "low",
        "min_wallet_count": 2,
        "max_entry_price": 0.85,
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
        ]
    }
    CRYPTO_HF_SCOPE_DEFAULTS: dict[str, Any] = {
        "include_assets": ["BTC", "ETH", "SOL", "XRP"],
        "exclude_assets": [],
        "include_timeframes": ["5m", "15m", "1h", "4h"],
        "exclude_timeframes": [],
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
    def _normalize_strategy_slug(slug: str) -> str:
        value = str(slug or "").strip().lower()
        if not value:
            return ""
        if StrategySDK._CONFIG_SLUG_RE.match(value):
            return value
        value = re.sub(r"[^a-z0-9_]", "_", value)
        value = value.strip("_")
        if not value:
            return ""
        if not value[0].isalpha():
            value = f"s_{value}"
        if len(value) > 50:
            value = value[:50].rstrip("_")
        if value and not value[-1].isalnum():
            value = f"{value}0"
        if StrategySDK._CONFIG_SLUG_RE.match(value):
            return value
        return ""

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
    def trader_filter_defaults() -> dict[str, Any]:
        return dict(StrategySDK.TRADER_FILTER_DEFAULTS)

    @staticmethod
    def trader_filter_config_schema() -> dict[str, Any]:
        return dict(StrategySDK.TRADER_FILTER_CONFIG_SCHEMA)

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
        cfg["replacement_score_cutoff"] = StrategySDK._coerce_float(
            cfg.get("replacement_score_cutoff"), 0.05, 0.0, 1.0
        )
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
        cfg["slo_min_profitable_pct"] = StrategySDK._coerce_float(
            cfg.get("slo_min_profitable_pct"), 80.0, 0.0, 100.0
        )
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
        return dict(StrategySDK.CRYPTO_HF_SCOPE_DEFAULTS)

    @staticmethod
    def crypto_highfreq_scope_config_schema() -> dict[str, Any]:
        return dict(StrategySDK.CRYPTO_HF_SCOPE_CONFIG_SCHEMA)

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
        return StrategySDK.normalize_strategy_retention_config(cfg)

    @staticmethod
    def _strategy_config_dir() -> Path:
        return Path(__file__).resolve().parents[1] / "strategy_configs"

    @staticmethod
    def _strategy_config_path(slug: str) -> Optional[Path]:
        normalized = StrategySDK._normalize_strategy_slug(slug)
        if not normalized:
            return None
        return StrategySDK._strategy_config_dir() / f"{normalized}.json"

    @staticmethod
    def get_strategy_config_path(slug: str) -> Optional[str]:
        """Return the absolute JSON config path for a strategy slug."""
        path = StrategySDK._strategy_config_path(slug)
        if path is None:
            return None
        return str(path)

    @staticmethod
    def get_strategy_config_mtime_ns(slug: str) -> int:
        """Return config file mtime (ns), or 0 when no file exists."""
        path = StrategySDK._strategy_config_path(slug)
        if path is None or not path.exists():
            return 0
        try:
            return int(path.stat().st_mtime_ns)
        except Exception:
            return 0

    @staticmethod
    def read_strategy_config_file(slug: str) -> dict[str, Any]:
        """Read JSON config overrides from backend/strategy_configs/<slug>.json.

        .. deprecated::
            File-based config overrides are no longer applied by the strategy
            loader. Config is now managed exclusively via the DB (UI). This
            function is retained for compatibility but has no effect on strategy
            runtime config.
        """
        warnings.warn(
            "read_strategy_config_file() is deprecated. Config is now managed exclusively via the DB. "
            "File-based overrides are no longer applied by the strategy loader.",
            DeprecationWarning,
            stacklevel=2,
        )
        path = StrategySDK._strategy_config_path(slug)
        if path is None or not path.exists():
            return {}
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                return parsed
            logger.warning("Strategy config file is not an object: %s", path)
            return {}
        except Exception as e:
            logger.warning("Failed to read strategy config file %s: %s", path, e)
            return {}

    @staticmethod
    def write_strategy_config_file(slug: str, config: dict[str, Any]) -> bool:
        """Write config overrides to backend/strategy_configs/<slug>.json."""
        path = StrategySDK._strategy_config_path(slug)
        if path is None:
            return False
        payload = config if isinstance(config, dict) else {}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            encoded = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)
            path.write_text(f"{encoded}\n", encoding="utf-8")
            return True
        except Exception as e:
            logger.warning("Failed to write strategy config file %s: %s", path, e)
            return False

    @staticmethod
    def rename_strategy_config_file(old_slug: str, new_slug: str) -> bool:
        """Rename a strategy config file when strategy slug changes."""
        old_path = StrategySDK._strategy_config_path(old_slug)
        new_path = StrategySDK._strategy_config_path(new_slug)
        if old_path is None or new_path is None or old_path == new_path:
            return True
        if not old_path.exists():
            return True
        try:
            new_path.parent.mkdir(parents=True, exist_ok=True)
            old_path.replace(new_path)
            return True
        except Exception as e:
            logger.warning(
                "Failed to rename strategy config file %s -> %s: %s",
                old_path,
                new_path,
                e,
            )
            return False

    @staticmethod
    def delete_strategy_config_file(slug: str) -> bool:
        """Delete backend/strategy_configs/<slug>.json if it exists."""
        path = StrategySDK._strategy_config_path(slug)
        if path is None or not path.exists():
            return True
        try:
            path.unlink()
            return True
        except Exception as e:
            logger.warning("Failed to delete strategy config file %s: %s", path, e)
            return False

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
