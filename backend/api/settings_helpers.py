"""Helpers for settings payload serialization and update application."""

from __future__ import annotations

from utils.utcnow import utcnow
from typing import Any, Optional

from models.database import AppSettings
from utils.secrets import decrypt_secret, encrypt_secret

SEARCH_FILTER_DEFAULTS: dict[str, Any] = {
    "min_liquidity_hard": 200.0,
    "min_position_size": 25.0,
    "min_absolute_profit": 5.0,
    "min_annualized_roi": 10.0,
    "max_resolution_months": 18,
    "max_plausible_roi": 30.0,
    "max_trade_legs": 8,
    "negrisk_min_total_yes": 0.95,
    "negrisk_warn_total_yes": 0.97,
    "negrisk_election_min_total_yes": 0.97,
    "negrisk_max_resolution_spread_days": 7,
    "settlement_lag_max_days_to_resolution": 14,
    "settlement_lag_near_zero": 0.05,
    "settlement_lag_near_one": 0.95,
    "settlement_lag_min_sum_deviation": 0.03,
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
    "miracle_min_no_price": 0.90,
    "miracle_max_no_price": 0.995,
    "miracle_min_impossibility_score": 0.70,
    "btc_eth_hf_enabled": True,
    "cross_platform_enabled": True,
    "combinatorial_min_confidence": 0.75,
    "combinatorial_high_confidence": 0.90,
    "bayesian_cascade_enabled": True,
    "bayesian_min_edge_percent": 5.0,
    "bayesian_propagation_depth": 3,
    "liquidity_vacuum_enabled": True,
    "liquidity_vacuum_min_imbalance_ratio": 5.0,
    "liquidity_vacuum_min_depth_usd": 100.0,
    "entropy_arb_enabled": True,
    "entropy_arb_min_deviation": 0.25,
    "event_driven_enabled": True,
    "temporal_decay_enabled": True,
    "correlation_arb_enabled": True,
    "correlation_arb_min_correlation": 0.7,
    "correlation_arb_min_divergence": 0.05,
    "market_making_enabled": True,
    "market_making_spread_bps": 100.0,
    "market_making_max_inventory_usd": 500.0,
    "stat_arb_enabled": True,
    "stat_arb_min_edge": 0.05,
}

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
}


def _with_default(value: Any, default: Any) -> Any:
    if isinstance(value, str) and value == "":
        return default
    return default if value is None else value


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
        "notify_autotrader_orders": bool(
            getattr(settings, "notify_autotrader_orders", False)
        ),
        "notify_autotrader_issues": bool(
            getattr(settings, "notify_autotrader_issues", True)
        ),
        "notify_autotrader_timeline": bool(
            getattr(settings, "notify_autotrader_timeline", True)
        ),
        "notify_autotrader_summary_interval_minutes": int(
            getattr(settings, "notify_autotrader_summary_interval_minutes", 60) or 60
        ),
        "notify_autotrader_summary_per_trader": bool(
            getattr(settings, "notify_autotrader_summary_per_trader", False)
        ),
    }


def scanner_payload(settings: AppSettings) -> dict[str, Any]:
    return {
        "scan_interval_seconds": settings.scan_interval_seconds,
        "min_profit_threshold": settings.min_profit_threshold,
        "max_markets_to_scan": settings.max_markets_to_scan,
        "min_liquidity": settings.min_liquidity,
    }


def trading_payload(settings: AppSettings) -> dict[str, Any]:
    return {
        "trading_enabled": settings.trading_enabled,
        "max_trade_size_usd": settings.max_trade_size_usd,
        "max_daily_trade_volume": settings.max_daily_trade_volume,
        "max_open_positions": settings.max_open_positions,
        "max_slippage_percent": settings.max_slippage_percent,
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
        "market_cache_hygiene_enabled": _with_default(
            settings.market_cache_hygiene_enabled, True
        ),
        "market_cache_hygiene_interval_hours": _with_default(
            settings.market_cache_hygiene_interval_hours, 6
        ),
        "market_cache_retention_days": _with_default(
            settings.market_cache_retention_days, 120
        ),
        "market_cache_reference_lookback_days": _with_default(
            settings.market_cache_reference_lookback_days, 45
        ),
        "market_cache_weak_entry_grace_days": _with_default(
            settings.market_cache_weak_entry_grace_days, 7
        ),
        "market_cache_max_entries_per_slug": _with_default(
            settings.market_cache_max_entries_per_slug, 3
        ),
    }


def trading_proxy_payload(settings: AppSettings) -> dict[str, Any]:
    return {
        "enabled": settings.trading_proxy_enabled or False,
        "proxy_url": mask_stored_secret(settings.trading_proxy_url, show_chars=12),
        "verify_ssl": _with_default(settings.trading_proxy_verify_ssl, True),
        "timeout": settings.trading_proxy_timeout or 30.0,
        "require_vpn": _with_default(settings.trading_proxy_require_vpn, True),
    }


def discovery_payload(settings: AppSettings) -> dict[str, Any]:
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
    }


def search_filters_payload(settings: AppSettings) -> dict[str, Any]:
    return {
        field_name: _with_default(getattr(settings, field_name), default)
        for field_name, default in SEARCH_FILTER_DEFAULTS.items()
    }


def apply_update_request(settings: AppSettings, request: Any) -> dict[str, bool]:
    """Apply a partial UpdateSettingsRequest onto an AppSettings row."""
    if request.polymarket:
        pm = request.polymarket
        if pm.api_key is not None:
            set_encrypted_secret(settings, "polymarket_api_key", pm.api_key)
        if pm.api_secret is not None:
            set_encrypted_secret(settings, "polymarket_api_secret", pm.api_secret)
        if pm.api_passphrase is not None:
            set_encrypted_secret(
                settings, "polymarket_api_passphrase", pm.api_passphrase
            )
        if pm.private_key is not None:
            set_encrypted_secret(settings, "polymarket_private_key", pm.private_key)

    if request.kalshi:
        kal = request.kalshi
        if kal.email is not None:
            settings.kalshi_email = kal.email or None
        if kal.password is not None:
            set_encrypted_secret(settings, "kalshi_password", kal.password)
        if kal.api_key is not None:
            set_encrypted_secret(settings, "kalshi_api_key", kal.api_key)

    if request.llm:
        llm = request.llm
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

    if request.notifications:
        notif = request.notifications
        settings.notifications_enabled = notif.enabled
        if notif.telegram_bot_token is not None:
            set_encrypted_secret(settings, "telegram_bot_token", notif.telegram_bot_token)
        if notif.telegram_chat_id is not None:
            settings.telegram_chat_id = notif.telegram_chat_id or None
        settings.notify_on_opportunity = notif.notify_on_opportunity
        settings.notify_on_trade = notif.notify_on_trade
        settings.notify_min_roi = notif.notify_min_roi
        settings.notify_autotrader_orders = bool(
            getattr(notif, "notify_autotrader_orders", False)
        )
        settings.notify_autotrader_issues = bool(
            getattr(notif, "notify_autotrader_issues", True)
        )
        settings.notify_autotrader_timeline = bool(
            getattr(notif, "notify_autotrader_timeline", True)
        )
        interval_minutes = int(
            max(
                5,
                min(
                    1440,
                    int(
                        getattr(
                            notif, "notify_autotrader_summary_interval_minutes", 60
                        )
                        or 60
                    ),
                ),
            )
        )
        settings.notify_autotrader_summary_interval_minutes = interval_minutes
        settings.notify_autotrader_summary_per_trader = bool(
            getattr(notif, "notify_autotrader_summary_per_trader", False)
        )

    if request.scanner:
        scan = request.scanner
        settings.scan_interval_seconds = scan.scan_interval_seconds
        settings.min_profit_threshold = scan.min_profit_threshold
        settings.max_markets_to_scan = scan.max_markets_to_scan
        settings.min_liquidity = scan.min_liquidity

    if request.trading:
        trade = request.trading
        settings.trading_enabled = trade.trading_enabled
        settings.max_trade_size_usd = trade.max_trade_size_usd
        settings.max_daily_trade_volume = trade.max_daily_trade_volume
        settings.max_open_positions = trade.max_open_positions
        settings.max_slippage_percent = trade.max_slippage_percent
        settings.btc_eth_hf_series_btc_15m = trade.btc_eth_hf_series_btc_15m
        settings.btc_eth_hf_series_eth_15m = trade.btc_eth_hf_series_eth_15m
        settings.btc_eth_hf_series_sol_15m = trade.btc_eth_hf_series_sol_15m
        settings.btc_eth_hf_series_xrp_15m = trade.btc_eth_hf_series_xrp_15m
        settings.btc_eth_hf_series_btc_5m = trade.btc_eth_hf_series_btc_5m
        settings.btc_eth_hf_series_eth_5m = trade.btc_eth_hf_series_eth_5m
        settings.btc_eth_hf_series_sol_5m = trade.btc_eth_hf_series_sol_5m
        settings.btc_eth_hf_series_xrp_5m = trade.btc_eth_hf_series_xrp_5m
        settings.btc_eth_hf_series_btc_1h = trade.btc_eth_hf_series_btc_1h
        settings.btc_eth_hf_series_eth_1h = trade.btc_eth_hf_series_eth_1h
        settings.btc_eth_hf_series_sol_1h = trade.btc_eth_hf_series_sol_1h
        settings.btc_eth_hf_series_xrp_1h = trade.btc_eth_hf_series_xrp_1h
        settings.btc_eth_hf_series_btc_4h = trade.btc_eth_hf_series_btc_4h
        settings.btc_eth_hf_series_eth_4h = trade.btc_eth_hf_series_eth_4h
        settings.btc_eth_hf_series_sol_4h = trade.btc_eth_hf_series_sol_4h
        settings.btc_eth_hf_series_xrp_4h = trade.btc_eth_hf_series_xrp_4h

    if request.maintenance:
        maint = request.maintenance
        settings.auto_cleanup_enabled = maint.auto_cleanup_enabled
        settings.cleanup_interval_hours = maint.cleanup_interval_hours
        settings.cleanup_resolved_trade_days = maint.cleanup_resolved_trade_days
        settings.market_cache_hygiene_enabled = maint.market_cache_hygiene_enabled
        settings.market_cache_hygiene_interval_hours = (
            maint.market_cache_hygiene_interval_hours
        )
        settings.market_cache_retention_days = maint.market_cache_retention_days
        settings.market_cache_reference_lookback_days = (
            maint.market_cache_reference_lookback_days
        )
        settings.market_cache_weak_entry_grace_days = (
            maint.market_cache_weak_entry_grace_days
        )
        settings.market_cache_max_entries_per_slug = (
            maint.market_cache_max_entries_per_slug
        )

    if request.discovery:
        discovery = request.discovery
        settings.discovery_max_discovered_wallets = discovery.max_discovered_wallets
        settings.discovery_maintenance_enabled = discovery.maintenance_enabled
        settings.discovery_keep_recent_trade_days = discovery.keep_recent_trade_days
        settings.discovery_keep_new_discoveries_days = (
            discovery.keep_new_discoveries_days
        )
        settings.discovery_maintenance_batch = discovery.maintenance_batch
        settings.discovery_stale_analysis_hours = discovery.stale_analysis_hours
        settings.discovery_analysis_priority_batch_limit = (
            discovery.analysis_priority_batch_limit
        )
        settings.discovery_delay_between_markets = discovery.delay_between_markets
        settings.discovery_delay_between_wallets = discovery.delay_between_wallets
        settings.discovery_max_markets_per_run = discovery.max_markets_per_run
        settings.discovery_max_wallets_per_market = discovery.max_wallets_per_market

    if request.search_filters:
        sf = request.search_filters
        for field_name in SEARCH_FILTER_DEFAULTS:
            setattr(settings, field_name, getattr(sf, field_name))

    if request.trading_proxy:
        proxy = request.trading_proxy
        settings.trading_proxy_enabled = proxy.enabled
        if proxy.proxy_url is not None:
            set_encrypted_secret(settings, "trading_proxy_url", proxy.proxy_url)
        settings.trading_proxy_verify_ssl = proxy.verify_ssl
        settings.trading_proxy_timeout = proxy.timeout
        settings.trading_proxy_require_vpn = proxy.require_vpn

    settings.updated_at = utcnow()
    return {
        "needs_llm_reinit": bool(request.llm),
        "needs_proxy_reinit": bool(request.trading_proxy),
        "needs_filter_reload": bool(request.search_filters),
    }
