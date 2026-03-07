"""Helpers for settings payload serialization and update application."""

from __future__ import annotations

from utils.utcnow import utcnow
from typing import Any, Optional

from models.database import AppSettings
from services.strategy_sdk import StrategySDK
from utils.passwords import hash_password
from utils.secrets import decrypt_secret, encrypt_secret

SEARCH_FILTER_DEFAULTS: dict[str, Any] = {
    "min_liquidity_hard": 1000.0,
    "min_position_size": 50.0,
    "min_absolute_profit": 10.0,
    "min_annualized_roi": 10.0,
    "max_resolution_months": 18,
    "max_plausible_roi": 30.0,
    "max_trade_legs": 6,
    "min_liquidity_per_leg": 500.0,
    "risk_very_short_days": 2,
    "risk_short_days": 7,
    "risk_long_lockup_days": 180,
    "risk_extended_lockup_days": 90,
    "risk_low_liquidity": 1000.0,
    "risk_moderate_liquidity": 5000.0,
    "risk_complex_legs": 5,
    "risk_multiple_legs": 3,
    "btc_eth_hf_series_btc_15m": "10192",
    "btc_eth_hf_series_eth_15m": "10191",
    "btc_eth_hf_series_sol_15m": "10423",
    "btc_eth_hf_series_xrp_15m": "10422",
    "btc_eth_hf_series_btc_5m": "10684",
    "btc_eth_hf_series_eth_5m": "",
    "btc_eth_hf_series_sol_5m": "",
    "btc_eth_hf_series_xrp_5m": "",
    "btc_eth_hf_series_btc_1h": "10114",
    "btc_eth_hf_series_eth_1h": "10117",
    "btc_eth_hf_series_sol_1h": "10122",
    "btc_eth_hf_series_xrp_1h": "10123",
    "btc_eth_hf_series_btc_4h": "10331",
    "btc_eth_hf_series_eth_4h": "10332",
    "btc_eth_hf_series_sol_4h": "10326",
    "btc_eth_hf_series_xrp_4h": "10327",
    "btc_eth_pure_arb_max_combined": 0.98,
    "btc_eth_dump_hedge_drop_pct": 0.05,
    "btc_eth_thin_liquidity_usd": 500.0,
    "btc_eth_hf_enabled": True,
    "btc_eth_hf_maker_mode": True,
}

_TRADER_OPPS_DEFAULTS = StrategySDK.trader_opportunity_filter_defaults()
_POOL_DEFAULTS = StrategySDK.pool_eligibility_defaults()
DISCOVERY_SETTINGS_DEFAULTS: dict[str, Any] = {
    "max_discovered_wallets": 20_000,
    "maintenance_enabled": True,
    "keep_recent_trade_days": 7,
    "keep_new_discoveries_days": 30,
    "maintenance_batch": 900,
    "stale_analysis_hours": 12,
    "analysis_priority_batch_limit": 2500,
    "delay_between_markets": 0.25,
    "delay_between_wallets": 0.15,
    "max_markets_per_run": 100,
    "max_wallets_per_market": 50,
    "trader_opps_source_filter": _TRADER_OPPS_DEFAULTS["source_filter"],
    "trader_opps_min_tier": _TRADER_OPPS_DEFAULTS["min_tier"],
    "trader_opps_side_filter": _TRADER_OPPS_DEFAULTS["side_filter"],
    "trader_opps_confluence_limit": _TRADER_OPPS_DEFAULTS["confluence_limit"],
    "trader_opps_insider_limit": _TRADER_OPPS_DEFAULTS["individual_trade_limit"],
    "trader_opps_insider_min_confidence": _TRADER_OPPS_DEFAULTS["individual_trade_min_confidence"],
    "trader_opps_insider_max_age_minutes": _TRADER_OPPS_DEFAULTS["individual_trade_max_age_minutes"],
    "pool_recompute_mode": "quality_only",
    "pool_target_size": _POOL_DEFAULTS["target_pool_size"],
    "pool_min_size": _POOL_DEFAULTS["min_pool_size"],
    "pool_max_size": _POOL_DEFAULTS["max_pool_size"],
    "pool_active_window_hours": _POOL_DEFAULTS["active_window_hours"],
    "pool_inactive_rising_retention_hours": _POOL_DEFAULTS["inactive_rising_retention_hours"],
    "pool_selection_score_floor": _POOL_DEFAULTS["selection_score_quality_target_floor"],
    "pool_max_hourly_replacement_rate": _POOL_DEFAULTS["max_hourly_replacement_rate"],
    "pool_replacement_score_cutoff": _POOL_DEFAULTS["replacement_score_cutoff"],
    "pool_max_cluster_share": _POOL_DEFAULTS["max_cluster_share"],
    "pool_high_conviction_threshold": _POOL_DEFAULTS["high_conviction_threshold"],
    "pool_insider_priority_threshold": _POOL_DEFAULTS["insider_priority_threshold"],
    "pool_min_eligible_trades": _POOL_DEFAULTS["min_eligible_trades"],
    "pool_max_eligible_anomaly": _POOL_DEFAULTS["max_eligible_anomaly"],
    "pool_core_min_win_rate": _POOL_DEFAULTS["core_min_win_rate"],
    "pool_core_min_sharpe": _POOL_DEFAULTS["core_min_sharpe"],
    "pool_core_min_profit_factor": _POOL_DEFAULTS["core_min_profit_factor"],
    "pool_rising_min_win_rate": _POOL_DEFAULTS["rising_min_win_rate"],
    "pool_slo_min_analyzed_pct": _POOL_DEFAULTS["slo_min_analyzed_pct"],
    "pool_slo_min_profitable_pct": _POOL_DEFAULTS["slo_min_profitable_pct"],
    "pool_leaderboard_wallet_trade_sample": _POOL_DEFAULTS["leaderboard_wallet_trade_sample"],
    "pool_incremental_wallet_trade_sample": _POOL_DEFAULTS["incremental_wallet_trade_sample"],
    "pool_full_sweep_interval_seconds": _POOL_DEFAULTS["full_sweep_interval_seconds"],
    "pool_incremental_refresh_interval_seconds": _POOL_DEFAULTS["incremental_refresh_interval_seconds"],
    "pool_activity_reconciliation_interval_seconds": _POOL_DEFAULTS["activity_reconciliation_interval_seconds"],
    "pool_recompute_interval_seconds": _POOL_DEFAULTS["pool_recompute_interval_seconds"],
}

EVENTS_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "interval_seconds": 300,
    "emit_trade_signals": False,
    "ais_enabled": True,
    "ais_sample_seconds": 10,
    "ais_max_messages": 250,
    "airplanes_live_enabled": True,
    "airplanes_live_timeout_seconds": 20.0,
    "airplanes_live_max_records": 1500,
    "convergence_min_types": 2,
    "anomaly_threshold": 1.8,
    "anomaly_min_baseline_points": 3,
    "instability_signal_min": 15.0,
    "instability_critical": 60.0,
    "tension_critical": 70.0,
    "gdelt_query_delay_seconds": 5.0,
    "gdelt_max_concurrency": 1,
    "gdelt_news_enabled": True,
    "gdelt_news_timespan_hours": 6,
    "gdelt_news_max_records": 40,
    "gdelt_news_request_timeout_seconds": 20.0,
    "gdelt_news_cache_seconds": 300,
    "gdelt_news_query_delay_seconds": 5.0,
    "gdelt_news_sync_enabled": True,
    "gdelt_news_sync_hours": 1,
    "acled_rate_limit_per_min": 5,
    "acled_auth_rate_limit_per_min": 12,
    "acled_cb_max_failures": 8,
    "acled_cb_cooldown_seconds": 180.0,
    "usgs_min_magnitude": 4.5,
    "usgs_enabled": True,
}


def _with_default(value: Any, default: Any) -> Any:
    if isinstance(value, str) and value == "":
        return default
    return default if value is None else value


def _normalize_discovery_pool_config(values: dict[str, Any]) -> dict[str, Any]:
    sdk_input = {
        "target_pool_size": values.get("pool_target_size"),
        "min_pool_size": values.get("pool_min_size"),
        "max_pool_size": values.get("pool_max_size"),
        "active_window_hours": values.get("pool_active_window_hours"),
        "inactive_rising_retention_hours": values.get("pool_inactive_rising_retention_hours"),
        "selection_score_quality_target_floor": values.get("pool_selection_score_floor"),
        "max_hourly_replacement_rate": values.get("pool_max_hourly_replacement_rate"),
        "replacement_score_cutoff": values.get("pool_replacement_score_cutoff"),
        "max_cluster_share": values.get("pool_max_cluster_share"),
        "high_conviction_threshold": values.get("pool_high_conviction_threshold"),
        "insider_priority_threshold": values.get("pool_insider_priority_threshold"),
        "min_eligible_trades": values.get("pool_min_eligible_trades"),
        "max_eligible_anomaly": values.get("pool_max_eligible_anomaly"),
        "core_min_win_rate": values.get("pool_core_min_win_rate"),
        "core_min_sharpe": values.get("pool_core_min_sharpe"),
        "core_min_profit_factor": values.get("pool_core_min_profit_factor"),
        "rising_min_win_rate": values.get("pool_rising_min_win_rate"),
        "slo_min_analyzed_pct": values.get("pool_slo_min_analyzed_pct"),
        "slo_min_profitable_pct": values.get("pool_slo_min_profitable_pct"),
        "leaderboard_wallet_trade_sample": values.get("pool_leaderboard_wallet_trade_sample"),
        "incremental_wallet_trade_sample": values.get("pool_incremental_wallet_trade_sample"),
        "full_sweep_interval_seconds": values.get("pool_full_sweep_interval_seconds"),
        "incremental_refresh_interval_seconds": values.get("pool_incremental_refresh_interval_seconds"),
        "activity_reconciliation_interval_seconds": values.get("pool_activity_reconciliation_interval_seconds"),
        "pool_recompute_interval_seconds": values.get("pool_recompute_interval_seconds"),
    }
    normalized = StrategySDK.validate_pool_eligibility_config(sdk_input)
    return {
        "pool_target_size": normalized["target_pool_size"],
        "pool_min_size": normalized["min_pool_size"],
        "pool_max_size": normalized["max_pool_size"],
        "pool_active_window_hours": normalized["active_window_hours"],
        "pool_inactive_rising_retention_hours": normalized["inactive_rising_retention_hours"],
        "pool_selection_score_floor": normalized["selection_score_quality_target_floor"],
        "pool_max_hourly_replacement_rate": normalized["max_hourly_replacement_rate"],
        "pool_replacement_score_cutoff": normalized["replacement_score_cutoff"],
        "pool_max_cluster_share": normalized["max_cluster_share"],
        "pool_high_conviction_threshold": normalized["high_conviction_threshold"],
        "pool_insider_priority_threshold": normalized["insider_priority_threshold"],
        "pool_min_eligible_trades": normalized["min_eligible_trades"],
        "pool_max_eligible_anomaly": normalized["max_eligible_anomaly"],
        "pool_core_min_win_rate": normalized["core_min_win_rate"],
        "pool_core_min_sharpe": normalized["core_min_sharpe"],
        "pool_core_min_profit_factor": normalized["core_min_profit_factor"],
        "pool_rising_min_win_rate": normalized["rising_min_win_rate"],
        "pool_slo_min_analyzed_pct": normalized["slo_min_analyzed_pct"],
        "pool_slo_min_profitable_pct": normalized["slo_min_profitable_pct"],
        "pool_leaderboard_wallet_trade_sample": normalized["leaderboard_wallet_trade_sample"],
        "pool_incremental_wallet_trade_sample": normalized["incremental_wallet_trade_sample"],
        "pool_full_sweep_interval_seconds": normalized["full_sweep_interval_seconds"],
        "pool_incremental_refresh_interval_seconds": normalized["incremental_refresh_interval_seconds"],
        "pool_activity_reconciliation_interval_seconds": normalized["activity_reconciliation_interval_seconds"],
        "pool_recompute_interval_seconds": normalized["pool_recompute_interval_seconds"],
    }


def _coerce_typed(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(default, bool):
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(float(value))
        except Exception:
            return default
    if isinstance(default, float):
        try:
            return float(value)
        except Exception:
            return default
    if isinstance(default, list):
        return value if isinstance(value, list) else list(default)
    if isinstance(default, dict):
        return value if isinstance(value, dict) else dict(default)
    if isinstance(default, str):
        return str(value)
    return value


def mask_secret(value: Optional[str], show_chars: int = 4) -> Optional[str]:
    """Mask a secret value, showing only first few characters."""
    if not value:
        return None
    if len(value) <= show_chars:
        return "*" * len(value)
    return value[:show_chars] + "*" * (len(value) - show_chars)


def mask_stored_secret(value: Optional[str], show_chars: int = 4) -> Optional[str]:
    """Mask an encrypted-or-plaintext value from DB storage."""
    return mask_secret(decrypt_secret(value), show_chars=show_chars)


def set_encrypted_secret(obj: AppSettings, field_name: str, value: Optional[str]) -> None:
    """Encrypt and set a DB-backed secret field."""
    setattr(obj, field_name, encrypt_secret(value or None))


def polymarket_payload(settings: AppSettings) -> dict[str, Any]:
    return {
        "api_key": mask_stored_secret(settings.polymarket_api_key),
        "api_secret": mask_stored_secret(settings.polymarket_api_secret),
        "api_passphrase": mask_stored_secret(settings.polymarket_api_passphrase),
        "private_key": mask_stored_secret(settings.polymarket_private_key),
    }


def kalshi_payload(settings: AppSettings) -> dict[str, Any]:
    return {
        "email": settings.kalshi_email,
        "password": mask_stored_secret(settings.kalshi_password),
        "api_key": mask_stored_secret(settings.kalshi_api_key),
    }


def llm_payload(settings: AppSettings) -> dict[str, Any]:
    return {
        "provider": settings.llm_provider or "none",
        "openai_api_key": mask_stored_secret(settings.openai_api_key),
        "anthropic_api_key": mask_stored_secret(settings.anthropic_api_key),
        "google_api_key": mask_stored_secret(settings.google_api_key),
        "xai_api_key": mask_stored_secret(settings.xai_api_key),
        "deepseek_api_key": mask_stored_secret(settings.deepseek_api_key),
        "ollama_api_key": mask_stored_secret(settings.ollama_api_key),
        "ollama_base_url": settings.ollama_base_url,
        "lmstudio_api_key": mask_stored_secret(settings.lmstudio_api_key),
        "lmstudio_base_url": settings.lmstudio_base_url,
        "model": settings.llm_model,
        "max_monthly_spend": settings.ai_max_monthly_spend,
    }


def notifications_payload(settings: AppSettings) -> dict[str, Any]:
    return {
        "enabled": settings.notifications_enabled,
        "telegram_bot_token": mask_stored_secret(settings.telegram_bot_token),
        "telegram_chat_id": settings.telegram_chat_id,
        "notify_on_opportunity": settings.notify_on_opportunity,
        "notify_on_trade": settings.notify_on_trade,
        "notify_min_roi": settings.notify_min_roi,
        "notify_autotrader_orders": bool(getattr(settings, "notify_autotrader_orders", False)),
        "notify_autotrader_closes": bool(getattr(settings, "notify_autotrader_closes", True)),
        "notify_autotrader_issues": bool(getattr(settings, "notify_autotrader_issues", True)),
        "notify_autotrader_timeline": bool(getattr(settings, "notify_autotrader_timeline", True)),
        "notify_autotrader_summary_interval_minutes": int(
            getattr(settings, "notify_autotrader_summary_interval_minutes", 60) or 60
        ),
        "notify_autotrader_summary_per_trader": bool(getattr(settings, "notify_autotrader_summary_per_trader", False)),
    }


def scanner_payload(settings: AppSettings) -> dict[str, Any]:
    return {
        "scan_interval_seconds": settings.scan_interval_seconds,
        "min_profit_threshold": settings.min_profit_threshold,
        "max_markets_to_scan": settings.max_markets_to_scan,
        "max_events_to_scan": getattr(settings, "max_events_to_scan", None) or 3000,
        "market_fetch_page_size": getattr(settings, "market_fetch_page_size", None) or 200,
        "market_fetch_order": getattr(settings, "market_fetch_order", None) or "volume",
        "min_liquidity": settings.min_liquidity,
        "max_opportunities_total": _with_default(getattr(settings, "scanner_max_opportunities_total", None), 500),
        "max_opportunities_per_strategy": _with_default(
            getattr(settings, "scanner_max_opportunities_per_strategy", None),
            120,
        ),
    }


def live_execution_payload(settings: AppSettings) -> dict[str, Any]:
    return {
        "max_trade_size_usd": settings.max_trade_size_usd,
        "max_daily_trade_volume": settings.max_daily_trade_volume,
        "max_slippage_percent": settings.max_slippage_percent,
        "min_account_balance_usd": settings.min_account_balance_usd,
        "btc_eth_hf_series_btc_15m": settings.btc_eth_hf_series_btc_15m,
        "btc_eth_hf_series_eth_15m": settings.btc_eth_hf_series_eth_15m,
        "btc_eth_hf_series_sol_15m": settings.btc_eth_hf_series_sol_15m,
        "btc_eth_hf_series_xrp_15m": settings.btc_eth_hf_series_xrp_15m,
        "btc_eth_hf_series_btc_5m": settings.btc_eth_hf_series_btc_5m,
        "btc_eth_hf_series_eth_5m": settings.btc_eth_hf_series_eth_5m,
        "btc_eth_hf_series_sol_5m": settings.btc_eth_hf_series_sol_5m,
        "btc_eth_hf_series_xrp_5m": settings.btc_eth_hf_series_xrp_5m,
        "btc_eth_hf_series_btc_1h": settings.btc_eth_hf_series_btc_1h,
        "btc_eth_hf_series_eth_1h": settings.btc_eth_hf_series_eth_1h,
        "btc_eth_hf_series_sol_1h": settings.btc_eth_hf_series_sol_1h,
        "btc_eth_hf_series_xrp_1h": settings.btc_eth_hf_series_xrp_1h,
        "btc_eth_hf_series_btc_4h": settings.btc_eth_hf_series_btc_4h,
        "btc_eth_hf_series_eth_4h": settings.btc_eth_hf_series_eth_4h,
        "btc_eth_hf_series_sol_4h": settings.btc_eth_hf_series_sol_4h,
        "btc_eth_hf_series_xrp_4h": settings.btc_eth_hf_series_xrp_4h,
    }


def maintenance_payload(settings: AppSettings) -> dict[str, Any]:
    return {
        "auto_cleanup_enabled": settings.auto_cleanup_enabled,
        "cleanup_interval_hours": settings.cleanup_interval_hours,
        "cleanup_resolved_trade_days": settings.cleanup_resolved_trade_days,
        "cleanup_trade_signal_emission_days": _with_default(settings.cleanup_trade_signal_emission_days, 21),
        "cleanup_trade_signal_update_days": _with_default(settings.cleanup_trade_signal_update_days, 3),
        "cleanup_wallet_activity_rollup_days": _with_default(settings.cleanup_wallet_activity_rollup_days, 60),
        "cleanup_wallet_activity_dedupe_enabled": _with_default(settings.cleanup_wallet_activity_dedupe_enabled, True),
        "llm_usage_retention_days": _with_default(settings.llm_usage_retention_days, 30),
        "market_cache_hygiene_enabled": _with_default(settings.market_cache_hygiene_enabled, True),
        "market_cache_hygiene_interval_hours": _with_default(settings.market_cache_hygiene_interval_hours, 6),
        "market_cache_retention_days": _with_default(settings.market_cache_retention_days, 120),
        "market_cache_reference_lookback_days": _with_default(settings.market_cache_reference_lookback_days, 45),
        "market_cache_weak_entry_grace_days": _with_default(settings.market_cache_weak_entry_grace_days, 7),
        "market_cache_max_entries_per_slug": _with_default(settings.market_cache_max_entries_per_slug, 3),
    }


def trading_proxy_payload(settings: AppSettings) -> dict[str, Any]:
    return {
        "enabled": settings.trading_proxy_enabled or False,
        "proxy_url": mask_stored_secret(settings.trading_proxy_url, show_chars=12),
        "verify_ssl": _with_default(settings.trading_proxy_verify_ssl, True),
        "timeout": settings.trading_proxy_timeout or 30.0,
        "require_vpn": _with_default(settings.trading_proxy_require_vpn, True),
    }


def ui_lock_payload(settings: AppSettings) -> dict[str, Any]:
    return {
        "enabled": bool(getattr(settings, "ui_lock_enabled", False)),
        "idle_timeout_minutes": int(_with_default(getattr(settings, "ui_lock_idle_timeout_minutes", None), 15)),
        "has_password": bool(getattr(settings, "ui_lock_password_hash", None)),
    }


def discovery_payload(settings: AppSettings) -> dict[str, Any]:
    trader_filters = StrategySDK.validate_trader_opportunity_filter_config(
        {
            "source_filter": _with_default(
                settings.discovery_trader_opps_source_filter,
                DISCOVERY_SETTINGS_DEFAULTS["trader_opps_source_filter"],
            ),
            "min_tier": _with_default(
                settings.discovery_trader_opps_min_tier,
                DISCOVERY_SETTINGS_DEFAULTS["trader_opps_min_tier"],
            ),
            "side_filter": _with_default(
                settings.discovery_trader_opps_side_filter,
                DISCOVERY_SETTINGS_DEFAULTS["trader_opps_side_filter"],
            ),
            "confluence_limit": _with_default(
                settings.discovery_trader_opps_confluence_limit,
                DISCOVERY_SETTINGS_DEFAULTS["trader_opps_confluence_limit"],
            ),
            "individual_trade_limit": _with_default(
                settings.discovery_trader_opps_insider_limit,
                DISCOVERY_SETTINGS_DEFAULTS["trader_opps_insider_limit"],
            ),
            "individual_trade_min_confidence": _with_default(
                settings.discovery_trader_opps_insider_min_confidence,
                DISCOVERY_SETTINGS_DEFAULTS["trader_opps_insider_min_confidence"],
            ),
            "individual_trade_max_age_minutes": _with_default(
                settings.discovery_trader_opps_insider_max_age_minutes,
                DISCOVERY_SETTINGS_DEFAULTS["trader_opps_insider_max_age_minutes"],
            ),
        }
    )
    pool_settings = _normalize_discovery_pool_config(
        {
            "pool_target_size": _with_default(
                settings.discovery_pool_target_size,
                DISCOVERY_SETTINGS_DEFAULTS["pool_target_size"],
            ),
            "pool_min_size": _with_default(
                settings.discovery_pool_min_size,
                DISCOVERY_SETTINGS_DEFAULTS["pool_min_size"],
            ),
            "pool_max_size": _with_default(
                settings.discovery_pool_max_size,
                DISCOVERY_SETTINGS_DEFAULTS["pool_max_size"],
            ),
            "pool_active_window_hours": _with_default(
                settings.discovery_pool_active_window_hours,
                DISCOVERY_SETTINGS_DEFAULTS["pool_active_window_hours"],
            ),
            "pool_inactive_rising_retention_hours": _with_default(
                settings.discovery_pool_inactive_rising_retention_hours,
                DISCOVERY_SETTINGS_DEFAULTS["pool_inactive_rising_retention_hours"],
            ),
            "pool_selection_score_floor": _with_default(
                settings.discovery_pool_selection_score_floor,
                DISCOVERY_SETTINGS_DEFAULTS["pool_selection_score_floor"],
            ),
            "pool_max_hourly_replacement_rate": _with_default(
                settings.discovery_pool_max_hourly_replacement_rate,
                DISCOVERY_SETTINGS_DEFAULTS["pool_max_hourly_replacement_rate"],
            ),
            "pool_replacement_score_cutoff": _with_default(
                settings.discovery_pool_replacement_score_cutoff,
                DISCOVERY_SETTINGS_DEFAULTS["pool_replacement_score_cutoff"],
            ),
            "pool_max_cluster_share": _with_default(
                settings.discovery_pool_max_cluster_share,
                DISCOVERY_SETTINGS_DEFAULTS["pool_max_cluster_share"],
            ),
            "pool_high_conviction_threshold": _with_default(
                settings.discovery_pool_high_conviction_threshold,
                DISCOVERY_SETTINGS_DEFAULTS["pool_high_conviction_threshold"],
            ),
            "pool_insider_priority_threshold": _with_default(
                settings.discovery_pool_insider_priority_threshold,
                DISCOVERY_SETTINGS_DEFAULTS["pool_insider_priority_threshold"],
            ),
            "pool_min_eligible_trades": _with_default(
                settings.discovery_pool_min_eligible_trades,
                DISCOVERY_SETTINGS_DEFAULTS["pool_min_eligible_trades"],
            ),
            "pool_max_eligible_anomaly": _with_default(
                settings.discovery_pool_max_eligible_anomaly,
                DISCOVERY_SETTINGS_DEFAULTS["pool_max_eligible_anomaly"],
            ),
            "pool_core_min_win_rate": _with_default(
                settings.discovery_pool_core_min_win_rate,
                DISCOVERY_SETTINGS_DEFAULTS["pool_core_min_win_rate"],
            ),
            "pool_core_min_sharpe": _with_default(
                settings.discovery_pool_core_min_sharpe,
                DISCOVERY_SETTINGS_DEFAULTS["pool_core_min_sharpe"],
            ),
            "pool_core_min_profit_factor": _with_default(
                settings.discovery_pool_core_min_profit_factor,
                DISCOVERY_SETTINGS_DEFAULTS["pool_core_min_profit_factor"],
            ),
            "pool_rising_min_win_rate": _with_default(
                settings.discovery_pool_rising_min_win_rate,
                DISCOVERY_SETTINGS_DEFAULTS["pool_rising_min_win_rate"],
            ),
            "pool_slo_min_analyzed_pct": _with_default(
                settings.discovery_pool_slo_min_analyzed_pct,
                DISCOVERY_SETTINGS_DEFAULTS["pool_slo_min_analyzed_pct"],
            ),
            "pool_slo_min_profitable_pct": _with_default(
                settings.discovery_pool_slo_min_profitable_pct,
                DISCOVERY_SETTINGS_DEFAULTS["pool_slo_min_profitable_pct"],
            ),
            "pool_leaderboard_wallet_trade_sample": _with_default(
                settings.discovery_pool_leaderboard_wallet_trade_sample,
                DISCOVERY_SETTINGS_DEFAULTS["pool_leaderboard_wallet_trade_sample"],
            ),
            "pool_incremental_wallet_trade_sample": _with_default(
                settings.discovery_pool_incremental_wallet_trade_sample,
                DISCOVERY_SETTINGS_DEFAULTS["pool_incremental_wallet_trade_sample"],
            ),
            "pool_full_sweep_interval_seconds": _with_default(
                settings.discovery_pool_full_sweep_interval_seconds,
                DISCOVERY_SETTINGS_DEFAULTS["pool_full_sweep_interval_seconds"],
            ),
            "pool_incremental_refresh_interval_seconds": _with_default(
                settings.discovery_pool_incremental_refresh_interval_seconds,
                DISCOVERY_SETTINGS_DEFAULTS["pool_incremental_refresh_interval_seconds"],
            ),
            "pool_activity_reconciliation_interval_seconds": _with_default(
                settings.discovery_pool_activity_reconciliation_interval_seconds,
                DISCOVERY_SETTINGS_DEFAULTS["pool_activity_reconciliation_interval_seconds"],
            ),
            "pool_recompute_interval_seconds": _with_default(
                settings.discovery_pool_recompute_interval_seconds,
                DISCOVERY_SETTINGS_DEFAULTS["pool_recompute_interval_seconds"],
            ),
        }
    )
    return {
        "max_discovered_wallets": _with_default(
            settings.discovery_max_discovered_wallets,
            DISCOVERY_SETTINGS_DEFAULTS["max_discovered_wallets"],
        ),
        "maintenance_enabled": _with_default(
            settings.discovery_maintenance_enabled,
            DISCOVERY_SETTINGS_DEFAULTS["maintenance_enabled"],
        ),
        "keep_recent_trade_days": _with_default(
            settings.discovery_keep_recent_trade_days,
            DISCOVERY_SETTINGS_DEFAULTS["keep_recent_trade_days"],
        ),
        "keep_new_discoveries_days": _with_default(
            settings.discovery_keep_new_discoveries_days,
            DISCOVERY_SETTINGS_DEFAULTS["keep_new_discoveries_days"],
        ),
        "maintenance_batch": _with_default(
            settings.discovery_maintenance_batch,
            DISCOVERY_SETTINGS_DEFAULTS["maintenance_batch"],
        ),
        "stale_analysis_hours": _with_default(
            settings.discovery_stale_analysis_hours,
            DISCOVERY_SETTINGS_DEFAULTS["stale_analysis_hours"],
        ),
        "analysis_priority_batch_limit": _with_default(
            settings.discovery_analysis_priority_batch_limit,
            DISCOVERY_SETTINGS_DEFAULTS["analysis_priority_batch_limit"],
        ),
        "delay_between_markets": _with_default(
            settings.discovery_delay_between_markets,
            DISCOVERY_SETTINGS_DEFAULTS["delay_between_markets"],
        ),
        "delay_between_wallets": _with_default(
            settings.discovery_delay_between_wallets,
            DISCOVERY_SETTINGS_DEFAULTS["delay_between_wallets"],
        ),
        "max_markets_per_run": _with_default(
            settings.discovery_max_markets_per_run,
            DISCOVERY_SETTINGS_DEFAULTS["max_markets_per_run"],
        ),
        "max_wallets_per_market": _with_default(
            settings.discovery_max_wallets_per_market,
            DISCOVERY_SETTINGS_DEFAULTS["max_wallets_per_market"],
        ),
        "trader_opps_source_filter": trader_filters["source_filter"],
        "trader_opps_min_tier": trader_filters["min_tier"],
        "trader_opps_side_filter": trader_filters["side_filter"],
        "trader_opps_confluence_limit": trader_filters["confluence_limit"],
        "trader_opps_insider_limit": trader_filters["individual_trade_limit"],
        "trader_opps_insider_min_confidence": trader_filters["individual_trade_min_confidence"],
        "trader_opps_insider_max_age_minutes": trader_filters["individual_trade_max_age_minutes"],
        "pool_recompute_mode": _with_default(
            settings.discovery_pool_recompute_mode,
            DISCOVERY_SETTINGS_DEFAULTS["pool_recompute_mode"],
        ),
        **pool_settings,
    }


def search_filters_payload(settings: AppSettings) -> dict[str, Any]:
    return {
        field_name: _with_default(getattr(settings, field_name), default)
        for field_name, default in SEARCH_FILTER_DEFAULTS.items()
    }


def network_payload(settings: AppSettings) -> dict[str, Any]:
    return {
        "allow_network_access": bool(getattr(settings, "allow_network_access", False)),
    }


def events_payload(settings: AppSettings) -> dict[str, Any]:
    raw = getattr(settings, "events_settings_json", None)
    stored = raw if isinstance(raw, dict) else {}

    if "gdelt_news_enabled" not in stored:
        stored = {
            **stored,
            "gdelt_news_enabled": getattr(settings, "events_gdelt_news_enabled", None),
            "gdelt_news_timespan_hours": getattr(settings, "events_gdelt_news_timespan_hours", None),
            "gdelt_news_max_records": getattr(settings, "events_gdelt_news_max_records", None),
        }

    payload = {
        field_name: _coerce_typed(stored.get(field_name), default) for field_name, default in EVENTS_DEFAULTS.items()
    }
    payload.update(
        {
            "acled_api_key": mask_stored_secret(settings.events_acled_api_key),
            "acled_email": getattr(settings, "events_acled_email", None),
            "opensky_username": getattr(settings, "events_opensky_username", None),
            "opensky_password": mask_stored_secret(settings.events_opensky_password),
            "aisstream_api_key": mask_stored_secret(settings.events_aisstream_api_key),
            "cloudflare_radar_token": mask_stored_secret(settings.events_cloudflare_radar_token),
        }
    )
    return payload


def apply_update_request(settings: AppSettings, request: Any) -> dict[str, bool]:
    """Apply a partial UpdateSettingsRequest onto an AppSettings row."""
    polymarket = getattr(request, "polymarket", None)
    kalshi = getattr(request, "kalshi", None)
    llm = getattr(request, "llm", None)
    notifications = getattr(request, "notifications", None)
    scanner = getattr(request, "scanner", None)
    live_execution = getattr(request, "live_execution", None)
    maintenance = getattr(request, "maintenance", None)
    discovery = getattr(request, "discovery", None)
    search_filters = getattr(request, "search_filters", None)
    trading_proxy = getattr(request, "trading_proxy", None)
    events = getattr(request, "events", None)
    ui_lock = getattr(request, "ui_lock", None)

    previous_ui_lock_enabled = bool(getattr(settings, "ui_lock_enabled", False))
    previous_ui_lock_password_hash = getattr(settings, "ui_lock_password_hash", None)

    if polymarket:
        pm = polymarket
        if pm.api_key is not None:
            set_encrypted_secret(settings, "polymarket_api_key", pm.api_key)
        if pm.api_secret is not None:
            set_encrypted_secret(settings, "polymarket_api_secret", pm.api_secret)
        if pm.api_passphrase is not None:
            set_encrypted_secret(settings, "polymarket_api_passphrase", pm.api_passphrase)
        if pm.private_key is not None:
            set_encrypted_secret(settings, "polymarket_private_key", pm.private_key)

    if kalshi:
        kal = kalshi
        if kal.email is not None:
            settings.kalshi_email = kal.email or None
        if kal.password is not None:
            set_encrypted_secret(settings, "kalshi_password", kal.password)
        if kal.api_key is not None:
            set_encrypted_secret(settings, "kalshi_api_key", kal.api_key)

    if llm:
        if llm.provider is not None:
            settings.llm_provider = llm.provider
        if llm.openai_api_key is not None:
            set_encrypted_secret(settings, "openai_api_key", llm.openai_api_key)
        if llm.anthropic_api_key is not None:
            set_encrypted_secret(settings, "anthropic_api_key", llm.anthropic_api_key)
        if llm.google_api_key is not None:
            set_encrypted_secret(settings, "google_api_key", llm.google_api_key)
        if llm.xai_api_key is not None:
            set_encrypted_secret(settings, "xai_api_key", llm.xai_api_key)
        if llm.deepseek_api_key is not None:
            set_encrypted_secret(settings, "deepseek_api_key", llm.deepseek_api_key)
        if llm.ollama_api_key is not None:
            set_encrypted_secret(settings, "ollama_api_key", llm.ollama_api_key)
        if llm.ollama_base_url is not None:
            settings.ollama_base_url = (llm.ollama_base_url or "").strip() or None
        if llm.lmstudio_api_key is not None:
            set_encrypted_secret(settings, "lmstudio_api_key", llm.lmstudio_api_key)
        if llm.lmstudio_base_url is not None:
            settings.lmstudio_base_url = (llm.lmstudio_base_url or "").strip() or None
        if llm.model is not None:
            settings.llm_model = llm.model or None
            settings.ai_default_model = llm.model or None
        if llm.max_monthly_spend is not None:
            settings.ai_max_monthly_spend = llm.max_monthly_spend

    if notifications:
        notif = notifications
        settings.notifications_enabled = notif.enabled
        if notif.telegram_bot_token is not None:
            set_encrypted_secret(settings, "telegram_bot_token", notif.telegram_bot_token)
        if notif.telegram_chat_id is not None:
            settings.telegram_chat_id = notif.telegram_chat_id or None
        settings.notify_on_opportunity = notif.notify_on_opportunity
        settings.notify_on_trade = notif.notify_on_trade
        settings.notify_min_roi = notif.notify_min_roi
        settings.notify_autotrader_orders = bool(getattr(notif, "notify_autotrader_orders", False))
        settings.notify_autotrader_closes = bool(getattr(notif, "notify_autotrader_closes", True))
        settings.notify_autotrader_issues = bool(getattr(notif, "notify_autotrader_issues", True))
        settings.notify_autotrader_timeline = bool(getattr(notif, "notify_autotrader_timeline", True))
        interval_minutes = int(
            max(
                5,
                min(
                    1440,
                    int(getattr(notif, "notify_autotrader_summary_interval_minutes", 60) or 60),
                ),
            )
        )
        settings.notify_autotrader_summary_interval_minutes = interval_minutes
        settings.notify_autotrader_summary_per_trader = bool(
            getattr(notif, "notify_autotrader_summary_per_trader", False)
        )

    if scanner:
        scan = scanner
        settings.scan_interval_seconds = scan.scan_interval_seconds
        settings.min_profit_threshold = scan.min_profit_threshold
        settings.max_markets_to_scan = scan.max_markets_to_scan
        settings.max_events_to_scan = getattr(scan, "max_events_to_scan", 3000)
        settings.market_fetch_page_size = getattr(scan, "market_fetch_page_size", 200)
        settings.market_fetch_order = getattr(scan, "market_fetch_order", "volume")
        settings.min_liquidity = scan.min_liquidity
        settings.scanner_max_opportunities_total = int(getattr(scan, "max_opportunities_total", 500))
        settings.scanner_max_opportunities_per_strategy = int(getattr(scan, "max_opportunities_per_strategy", 120))

    if live_execution:
        trade = live_execution
        settings.max_trade_size_usd = trade.max_trade_size_usd
        settings.max_daily_trade_volume = trade.max_daily_trade_volume
        if getattr(trade, "max_open_positions", None) is not None:
            settings.max_open_positions = trade.max_open_positions
        settings.max_slippage_percent = trade.max_slippage_percent
        settings.min_account_balance_usd = trade.min_account_balance_usd

    if maintenance:
        maint = maintenance
        settings.auto_cleanup_enabled = maint.auto_cleanup_enabled
        settings.cleanup_interval_hours = maint.cleanup_interval_hours
        settings.cleanup_resolved_trade_days = maint.cleanup_resolved_trade_days
        settings.cleanup_trade_signal_emission_days = maint.cleanup_trade_signal_emission_days
        settings.cleanup_trade_signal_update_days = maint.cleanup_trade_signal_update_days
        settings.cleanup_wallet_activity_rollup_days = maint.cleanup_wallet_activity_rollup_days
        settings.cleanup_wallet_activity_dedupe_enabled = maint.cleanup_wallet_activity_dedupe_enabled
        settings.llm_usage_retention_days = maint.llm_usage_retention_days
        settings.market_cache_hygiene_enabled = maint.market_cache_hygiene_enabled
        settings.market_cache_hygiene_interval_hours = maint.market_cache_hygiene_interval_hours
        settings.market_cache_retention_days = maint.market_cache_retention_days
        settings.market_cache_reference_lookback_days = maint.market_cache_reference_lookback_days
        settings.market_cache_weak_entry_grace_days = maint.market_cache_weak_entry_grace_days
        settings.market_cache_max_entries_per_slug = maint.market_cache_max_entries_per_slug

    if discovery:
        settings.discovery_max_discovered_wallets = discovery.max_discovered_wallets
        settings.discovery_maintenance_enabled = discovery.maintenance_enabled
        settings.discovery_keep_recent_trade_days = discovery.keep_recent_trade_days
        settings.discovery_keep_new_discoveries_days = discovery.keep_new_discoveries_days
        settings.discovery_maintenance_batch = discovery.maintenance_batch
        settings.discovery_stale_analysis_hours = discovery.stale_analysis_hours
        settings.discovery_analysis_priority_batch_limit = discovery.analysis_priority_batch_limit
        settings.discovery_delay_between_markets = discovery.delay_between_markets
        settings.discovery_delay_between_wallets = discovery.delay_between_wallets
        settings.discovery_max_markets_per_run = discovery.max_markets_per_run
        settings.discovery_max_wallets_per_market = discovery.max_wallets_per_market
        trader_filters = StrategySDK.validate_trader_opportunity_filter_config(
            {
                "source_filter": discovery.trader_opps_source_filter,
                "min_tier": discovery.trader_opps_min_tier,
                "side_filter": discovery.trader_opps_side_filter,
                "confluence_limit": discovery.trader_opps_confluence_limit,
                "individual_trade_limit": discovery.trader_opps_insider_limit,
                "individual_trade_min_confidence": discovery.trader_opps_insider_min_confidence,
                "individual_trade_max_age_minutes": discovery.trader_opps_insider_max_age_minutes,
            }
        )
        settings.discovery_trader_opps_source_filter = trader_filters["source_filter"]
        settings.discovery_trader_opps_min_tier = trader_filters["min_tier"]
        settings.discovery_trader_opps_side_filter = trader_filters["side_filter"]
        settings.discovery_trader_opps_confluence_limit = trader_filters["confluence_limit"]
        settings.discovery_trader_opps_insider_limit = trader_filters["individual_trade_limit"]
        settings.discovery_trader_opps_insider_min_confidence = trader_filters["individual_trade_min_confidence"]
        settings.discovery_trader_opps_insider_max_age_minutes = trader_filters["individual_trade_max_age_minutes"]
        settings.discovery_pool_recompute_mode = (
            str(discovery.pool_recompute_mode or "quality_only").strip().lower() or "quality_only"
        )
        pool_settings = _normalize_discovery_pool_config(
            {
                "pool_target_size": discovery.pool_target_size,
                "pool_min_size": discovery.pool_min_size,
                "pool_max_size": discovery.pool_max_size,
                "pool_active_window_hours": discovery.pool_active_window_hours,
                "pool_inactive_rising_retention_hours": discovery.pool_inactive_rising_retention_hours,
                "pool_selection_score_floor": discovery.pool_selection_score_floor,
                "pool_max_hourly_replacement_rate": discovery.pool_max_hourly_replacement_rate,
                "pool_replacement_score_cutoff": discovery.pool_replacement_score_cutoff,
                "pool_max_cluster_share": discovery.pool_max_cluster_share,
                "pool_high_conviction_threshold": discovery.pool_high_conviction_threshold,
                "pool_insider_priority_threshold": discovery.pool_insider_priority_threshold,
                "pool_min_eligible_trades": discovery.pool_min_eligible_trades,
                "pool_max_eligible_anomaly": discovery.pool_max_eligible_anomaly,
                "pool_core_min_win_rate": discovery.pool_core_min_win_rate,
                "pool_core_min_sharpe": discovery.pool_core_min_sharpe,
                "pool_core_min_profit_factor": discovery.pool_core_min_profit_factor,
                "pool_rising_min_win_rate": discovery.pool_rising_min_win_rate,
                "pool_slo_min_analyzed_pct": discovery.pool_slo_min_analyzed_pct,
                "pool_slo_min_profitable_pct": discovery.pool_slo_min_profitable_pct,
                "pool_leaderboard_wallet_trade_sample": discovery.pool_leaderboard_wallet_trade_sample,
                "pool_incremental_wallet_trade_sample": discovery.pool_incremental_wallet_trade_sample,
                "pool_full_sweep_interval_seconds": discovery.pool_full_sweep_interval_seconds,
                "pool_incremental_refresh_interval_seconds": discovery.pool_incremental_refresh_interval_seconds,
                "pool_activity_reconciliation_interval_seconds": discovery.pool_activity_reconciliation_interval_seconds,
                "pool_recompute_interval_seconds": discovery.pool_recompute_interval_seconds,
            }
        )
        settings.discovery_pool_target_size = pool_settings["pool_target_size"]
        settings.discovery_pool_min_size = pool_settings["pool_min_size"]
        settings.discovery_pool_max_size = pool_settings["pool_max_size"]
        settings.discovery_pool_active_window_hours = pool_settings["pool_active_window_hours"]
        settings.discovery_pool_inactive_rising_retention_hours = pool_settings["pool_inactive_rising_retention_hours"]
        settings.discovery_pool_selection_score_floor = pool_settings["pool_selection_score_floor"]
        settings.discovery_pool_max_hourly_replacement_rate = pool_settings["pool_max_hourly_replacement_rate"]
        settings.discovery_pool_replacement_score_cutoff = pool_settings["pool_replacement_score_cutoff"]
        settings.discovery_pool_max_cluster_share = pool_settings["pool_max_cluster_share"]
        settings.discovery_pool_high_conviction_threshold = pool_settings["pool_high_conviction_threshold"]
        settings.discovery_pool_insider_priority_threshold = pool_settings["pool_insider_priority_threshold"]
        settings.discovery_pool_min_eligible_trades = pool_settings["pool_min_eligible_trades"]
        settings.discovery_pool_max_eligible_anomaly = pool_settings["pool_max_eligible_anomaly"]
        settings.discovery_pool_core_min_win_rate = pool_settings["pool_core_min_win_rate"]
        settings.discovery_pool_core_min_sharpe = pool_settings["pool_core_min_sharpe"]
        settings.discovery_pool_core_min_profit_factor = pool_settings["pool_core_min_profit_factor"]
        settings.discovery_pool_rising_min_win_rate = pool_settings["pool_rising_min_win_rate"]
        settings.discovery_pool_slo_min_analyzed_pct = pool_settings["pool_slo_min_analyzed_pct"]
        settings.discovery_pool_slo_min_profitable_pct = pool_settings["pool_slo_min_profitable_pct"]
        settings.discovery_pool_leaderboard_wallet_trade_sample = pool_settings["pool_leaderboard_wallet_trade_sample"]
        settings.discovery_pool_incremental_wallet_trade_sample = pool_settings["pool_incremental_wallet_trade_sample"]
        settings.discovery_pool_full_sweep_interval_seconds = pool_settings["pool_full_sweep_interval_seconds"]
        settings.discovery_pool_incremental_refresh_interval_seconds = pool_settings[
            "pool_incremental_refresh_interval_seconds"
        ]
        settings.discovery_pool_activity_reconciliation_interval_seconds = pool_settings[
            "pool_activity_reconciliation_interval_seconds"
        ]
        settings.discovery_pool_recompute_interval_seconds = pool_settings["pool_recompute_interval_seconds"]

    if search_filters:
        sf = search_filters
        provided_fields = getattr(sf, "model_fields_set", set())
        for field_name in SEARCH_FILTER_DEFAULTS:
            if field_name in provided_fields:
                setattr(settings, field_name, getattr(sf, field_name))

    if trading_proxy:
        proxy = trading_proxy
        settings.trading_proxy_enabled = proxy.enabled
        if proxy.proxy_url is not None:
            set_encrypted_secret(settings, "trading_proxy_url", proxy.proxy_url)
        settings.trading_proxy_verify_ssl = proxy.verify_ssl
        settings.trading_proxy_timeout = proxy.timeout
        settings.trading_proxy_require_vpn = proxy.require_vpn

    if ui_lock is not None:
        enabled = bool(getattr(ui_lock, "enabled", False))
        timeout_minutes = int(getattr(ui_lock, "idle_timeout_minutes", 15) or 15)
        timeout_minutes = max(1, min(timeout_minutes, 24 * 60))
        clear_password = bool(getattr(ui_lock, "clear_password", False))
        incoming_password = getattr(ui_lock, "password", None)

        next_password_hash = getattr(settings, "ui_lock_password_hash", None)
        if clear_password:
            next_password_hash = None

        if incoming_password is not None:
            normalized = str(incoming_password).strip()
            if not normalized:
                raise ValueError("UI lock password cannot be empty.")
            next_password_hash = hash_password(normalized)

        if enabled and not next_password_hash:
            raise ValueError("UI lock password is required when enabling UI lock.")

        settings.ui_lock_enabled = enabled
        settings.ui_lock_idle_timeout_minutes = timeout_minutes
        settings.ui_lock_password_hash = next_password_hash

    network = getattr(request, "network", None)
    if network is not None:
        settings.allow_network_access = bool(getattr(network, "allow_network_access", False))

    if events is not None:
        current_raw = getattr(settings, "events_settings_json", None)
        current_config = current_raw if isinstance(current_raw, dict) else {}
        next_config = dict(current_config)

        for field_name, default in EVENTS_DEFAULTS.items():
            if hasattr(events, field_name):
                incoming = getattr(events, field_name)
                if incoming is not None:
                    next_config[field_name] = _coerce_typed(incoming, default)

        settings.events_settings_json = next_config

        # Keep existing GDELT DB-backed config columns aligned.
        settings.events_gdelt_news_enabled = bool(
            _coerce_typed(
                next_config.get("gdelt_news_enabled"),
                EVENTS_DEFAULTS["gdelt_news_enabled"],
            )
        )
        settings.events_gdelt_news_timespan_hours = int(
            _coerce_typed(
                next_config.get("gdelt_news_timespan_hours"),
                EVENTS_DEFAULTS["gdelt_news_timespan_hours"],
            )
        )
        settings.events_gdelt_news_max_records = int(
            _coerce_typed(
                next_config.get("gdelt_news_max_records"),
                EVENTS_DEFAULTS["gdelt_news_max_records"],
            )
        )

        if getattr(events, "acled_api_key", None) is not None:
            set_encrypted_secret(settings, "events_acled_api_key", events.acled_api_key)
        if getattr(events, "acled_email", None) is not None:
            settings.events_acled_email = str(events.acled_email or "").strip() or None
        if getattr(events, "opensky_username", None) is not None:
            settings.events_opensky_username = str(events.opensky_username or "").strip() or None
        if getattr(events, "opensky_password", None) is not None:
            set_encrypted_secret(
                settings,
                "events_opensky_password",
                events.opensky_password,
            )
        if getattr(events, "aisstream_api_key", None) is not None:
            set_encrypted_secret(
                settings,
                "events_aisstream_api_key",
                events.aisstream_api_key,
            )
        if getattr(events, "cloudflare_radar_token", None) is not None:
            set_encrypted_secret(
                settings,
                "events_cloudflare_radar_token",
                events.cloudflare_radar_token,
            )

    settings.updated_at = utcnow()
    ui_lock_password_changed = previous_ui_lock_password_hash != getattr(settings, "ui_lock_password_hash", None)
    ui_lock_enabled_changed = previous_ui_lock_enabled != bool(getattr(settings, "ui_lock_enabled", False))
    return {
        "needs_llm_reinit": bool(llm),
        "needs_proxy_reinit": bool(trading_proxy),
        "needs_filter_reload": bool(search_filters) or bool(scanner) or bool(live_execution),
        "needs_events_reload": events is not None,
        "needs_ui_lock_reload": ui_lock is not None,
        "reset_ui_lock_sessions": ui_lock_password_changed or ui_lock_enabled_changed,
    }
