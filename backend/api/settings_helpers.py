"""Helpers for settings payload serialization and update application."""

from __future__ import annotations

from datetime import datetime
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


def _with_default(value: Any, default: Any) -> Any:
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
    }


def maintenance_payload(settings: AppSettings) -> dict[str, Any]:
    return {
        "auto_cleanup_enabled": settings.auto_cleanup_enabled,
        "cleanup_interval_hours": settings.cleanup_interval_hours,
        "cleanup_resolved_trade_days": settings.cleanup_resolved_trade_days,
    }


def trading_proxy_payload(settings: AppSettings) -> dict[str, Any]:
    return {
        "enabled": settings.trading_proxy_enabled or False,
        "proxy_url": mask_stored_secret(settings.trading_proxy_url, show_chars=12),
        "verify_ssl": _with_default(settings.trading_proxy_verify_ssl, True),
        "timeout": settings.trading_proxy_timeout or 30.0,
        "require_vpn": _with_default(settings.trading_proxy_require_vpn, True),
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

    if request.maintenance:
        maint = request.maintenance
        settings.auto_cleanup_enabled = maint.auto_cleanup_enabled
        settings.cleanup_interval_hours = maint.cleanup_interval_hours
        settings.cleanup_resolved_trade_days = maint.cleanup_resolved_trade_days

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

    settings.updated_at = datetime.utcnow()
    return {
        "needs_llm_reinit": bool(request.llm),
        "needs_proxy_reinit": bool(request.trading_proxy),
        "needs_filter_reload": bool(request.search_filters),
    }
