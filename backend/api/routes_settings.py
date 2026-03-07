"""
Settings API Routes

Endpoints for managing application settings.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from enum import Enum

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Literal, Optional

from sqlalchemy import delete, func, select
from config import settings as runtime_settings
from models.database import (
    AsyncSessionLocal,
    AppSettings,
    DataSource,
    LiveTradingOrder,
    LiveTradingPosition,
    LiveTradingRuntimeState,
    Strategy,
    StrategyVersion,
    TraderOrder,
    TraderOrchestratorControl,
    TraderPosition,
)
from services.strategy_versioning import normalize_strategy_version
from utils.logger import get_logger
from utils.utcnow import utcnow
from utils.secrets import decrypt_secret
from api.settings_helpers import (
    apply_update_request,
    discovery_payload,
    kalshi_payload,
    llm_payload,
    maintenance_payload,
    network_payload,
    notifications_payload,
    polymarket_payload,
    scanner_payload,
    search_filters_payload,
    live_execution_payload,
    trading_proxy_payload,
    ui_lock_payload,
    events_payload,
    set_encrypted_secret,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/settings", tags=["Settings"])


# ==================== REQUEST/RESPONSE MODELS ====================


class PolymarketSettings(BaseModel):
    """Polymarket API credentials"""

    api_key: Optional[str] = Field(default=None, description="Polymarket API key")
    api_secret: Optional[str] = Field(default=None, description="Polymarket API secret")
    api_passphrase: Optional[str] = Field(default=None, description="Polymarket API passphrase")
    private_key: Optional[str] = Field(default=None, description="Wallet private key for signing")


class KalshiSettings(BaseModel):
    """Kalshi API credentials"""

    email: Optional[str] = Field(default=None, description="Kalshi account email")
    password: Optional[str] = Field(default=None, description="Kalshi account password")
    api_key: Optional[str] = Field(default=None, description="Kalshi API key")


class LLMSettings(BaseModel):
    """LLM service configuration"""

    provider: str = Field(
        default="none",
        description="LLM provider: none, openai, anthropic, google, xai, deepseek, ollama, lmstudio",
    )
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    google_api_key: Optional[str] = Field(default=None, description="Google (Gemini) API key")
    xai_api_key: Optional[str] = Field(default=None, description="xAI (Grok) API key")
    deepseek_api_key: Optional[str] = Field(default=None, description="DeepSeek API key")
    ollama_api_key: Optional[str] = Field(default=None, description="Ollama API key (optional)")
    ollama_base_url: Optional[str] = Field(
        default=None, description="Ollama base URL (default: http://localhost:11434)"
    )
    lmstudio_api_key: Optional[str] = Field(default=None, description="LM Studio API key (optional)")
    lmstudio_base_url: Optional[str] = Field(
        default=None, description="LM Studio base URL (default: http://localhost:1234/v1)"
    )
    model: Optional[str] = Field(default=None, description="Model to use (e.g., gpt-4o, gemini-2.0-flash)")
    max_monthly_spend: Optional[float] = Field(default=None, ge=0, description="Monthly LLM cost cap in USD")


class NotificationSettings(BaseModel):
    """Notification configuration"""

    enabled: bool = Field(default=False, description="Enable notifications")
    telegram_bot_token: Optional[str] = Field(default=None, description="Telegram bot token")
    telegram_chat_id: Optional[str] = Field(default=None, description="Telegram chat ID")
    notify_on_opportunity: bool = Field(default=True, description="Notify on new opportunities")
    notify_on_trade: bool = Field(default=True, description="Notify on trade execution")
    notify_min_roi: float = Field(default=5.0, ge=0, description="Minimum ROI % to notify")
    notify_autotrader_orders: bool = Field(
        default=False, description="Notify immediately when orchestrator creates orders"
    )
    notify_autotrader_closes: bool = Field(
        default=True, description="Notify when positions transition into closed/resolved states"
    )
    notify_autotrader_issues: bool = Field(
        default=True,
        description="Notify on orchestrator warnings/errors and failed orders",
    )
    notify_autotrader_timeline: bool = Field(
        default=True,
        description="Send timeline summaries while orchestrator is actively running",
    )
    notify_autotrader_summary_interval_minutes: int = Field(
        default=60,
        ge=5,
        le=1440,
        description="Minutes between autotrader timeline summaries",
    )
    notify_autotrader_summary_per_trader: bool = Field(
        default=False,
        description="Break timeline summaries out per trader instead of only overall totals",
    )


class ScannerSettingsModel(BaseModel):
    """Scanner configuration"""

    scan_interval_seconds: int = Field(default=60, ge=10, le=3600, description="Scan interval in seconds")
    min_profit_threshold: float = Field(default=2.5, ge=0, description="Minimum profit threshold %")
    max_markets_to_scan: int = Field(
        default=0,
        ge=0,
        le=200000,
        description="Maximum markets to scan per cycle (0 disables cap)",
    )
    max_events_to_scan: int = Field(
        default=0,
        ge=0,
        le=200000,
        description="Maximum events to scan per cycle (0 disables cap)",
    )
    market_fetch_page_size: int = Field(
        default=200, ge=50, le=500, description="API page size for market/event fetches"
    )
    market_fetch_order: str = Field(
        default="volume", description="Sort order for market fetches (volume, updatedAt, createdAt, or empty)"
    )
    min_liquidity: float = Field(default=1000.0, ge=0, description="Minimum liquidity in USD")
    max_opportunities_total: int = Field(
        default=500,
        ge=0,
        le=50000,
        description="Global cap for opportunities retained in scanner pool (0 disables)",
    )
    max_opportunities_per_strategy: int = Field(
        default=120,
        ge=0,
        le=10000,
        description="Per-strategy cap for opportunities retained in scanner pool (0 disables)",
    )


class LiveExecutionSettings(BaseModel):
    """Live execution safety configuration"""

    max_trade_size_usd: float = Field(default=100.0, ge=1, description="Maximum single trade size")
    max_daily_trade_volume: float = Field(default=1000.0, ge=10, description="Maximum daily trading volume")
    max_slippage_percent: float = Field(default=2.0, ge=0.1, le=10, description="Maximum acceptable slippage %")
    min_account_balance_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum USDC balance to keep in Polymarket account after buy orders",
    )


class DiscoverySettings(BaseModel):
    """Wallet discovery settings"""

    max_discovered_wallets: int = Field(
        default=20_000, ge=10, le=1_000_000, description="Cap for discovered wallet catalog"
    )
    maintenance_enabled: bool = Field(
        default=True,
        description="Enable periodic cleanup of weak/disused discovered wallets",
    )
    keep_recent_trade_days: int = Field(
        default=7,
        ge=1,
        le=365,
        description="Keep wallets with recent trades within this many days",
    )
    keep_new_discoveries_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Keep newly discovered wallets within this window",
    )
    maintenance_batch: int = Field(default=900, ge=10, le=5000, description="Cleanup insert/chunk batch size")
    stale_analysis_hours: int = Field(
        default=12,
        ge=1,
        le=720,
        description="Re-analyze wallets when last analysis is older than this many hours",
    )
    analysis_priority_batch_limit: int = Field(
        default=2500,
        ge=100,
        le=10_000,
        description="Maximum size of priority analysis queue",
    )
    delay_between_markets: float = Field(
        default=0.25,
        ge=0.0,
        le=10.0,
        description="Delay in seconds between market trade scans",
    )
    delay_between_wallets: float = Field(
        default=0.15,
        ge=0.0,
        le=10.0,
        description="Delay in seconds between wallet analysis calls",
    )
    max_markets_per_run: int = Field(
        default=100,
        ge=1,
        le=1_000,
        description="Maximum active markets sampled per discovery run",
    )
    max_wallets_per_market: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Max wallets to extract from each market",
    )
    trader_opps_source_filter: Literal[
        "all",
        "tracked",
        "pool",
    ] = Field(
        default="all",
        description="Opportunities -> Traders source view filter",
    )
    trader_opps_min_tier: Literal["WATCH", "HIGH", "EXTREME"] = Field(
        default="WATCH",
        description="Minimum confluence tier for tracked-trader opportunities",
    )
    trader_opps_side_filter: Literal["all", "buy", "sell"] = Field(
        default="all",
        description="Opportunities -> Traders directional side filter",
    )
    trader_opps_confluence_limit: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Tracked-trader confluence fetch limit",
    )
    trader_opps_insider_limit: int = Field(
        default=40,
        ge=1,
        le=500,
        description="Tracked/group individual-trade feed limit",
    )
    trader_opps_insider_min_confidence: float = Field(
        default=0.62,
        ge=0.0,
        le=1.0,
        description="Legacy field retained for backward compatibility",
    )
    trader_opps_insider_max_age_minutes: int = Field(
        default=180,
        ge=1,
        le=1440,
        description="Legacy field retained for backward compatibility",
    )
    pool_recompute_mode: Literal["quality_only", "balanced"] = Field(
        default="quality_only",
        description="Smart-wallet pool recompute mode",
    )
    pool_target_size: int = Field(
        default=500,
        ge=10,
        le=5000,
        description="Target smart-wallet pool size",
    )
    pool_min_size: int = Field(
        default=400,
        ge=0,
        le=5000,
        description="Minimum desired pool size in balanced mode",
    )
    pool_max_size: int = Field(
        default=600,
        ge=1,
        le=10000,
        description="Hard upper bound for pool size",
    )
    pool_active_window_hours: int = Field(
        default=72,
        ge=1,
        le=720,
        description="Recency window (hours) used by activity-aware gating",
    )
    pool_inactive_rising_retention_hours: int = Field(
        default=336,
        ge=0,
        le=8760,
        description="How long to retain rising-tier traders after inactivity (hours)",
    )
    pool_selection_score_floor: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Selection score floor for quality-only pool sizing",
    )
    pool_max_hourly_replacement_rate: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Maximum fraction of pool replaced per churn cycle",
    )
    pool_replacement_score_cutoff: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum score delta required to replace an existing wallet",
    )
    pool_max_cluster_share: float = Field(
        default=0.08,
        ge=0.01,
        le=1.0,
        description="Max share of pool allocated to a single cluster",
    )
    pool_high_conviction_threshold: float = Field(
        default=0.72,
        ge=0.0,
        le=1.0,
        description="Selection score threshold for high-conviction reason tagging",
    )
    pool_insider_priority_threshold: float = Field(
        default=0.62,
        ge=0.0,
        le=1.0,
        description="Insider-score threshold for priority reason tagging",
    )
    pool_min_eligible_trades: int = Field(
        default=50,
        ge=1,
        le=100000,
        description="Minimum total trades required for pool eligibility",
    )
    pool_max_eligible_anomaly: float = Field(
        default=0.5,
        ge=0.0,
        le=5.0,
        description="Maximum anomaly score allowed for pool entry",
    )
    pool_core_min_win_rate: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Core tier minimum win rate (ratio)",
    )
    pool_core_min_sharpe: float = Field(
        default=1.0,
        ge=-10.0,
        le=20.0,
        description="Core tier minimum Sharpe ratio",
    )
    pool_core_min_profit_factor: float = Field(
        default=1.5,
        ge=0.0,
        le=20.0,
        description="Core tier minimum profit factor",
    )
    pool_rising_min_win_rate: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Rising tier minimum win rate (ratio)",
    )
    pool_slo_min_analyzed_pct: float = Field(
        default=95.0,
        ge=0.0,
        le=100.0,
        description="SLO minimum analyzed percent for active pool",
    )
    pool_slo_min_profitable_pct: float = Field(
        default=80.0,
        ge=0.0,
        le=100.0,
        description="SLO minimum profitable percent for active pool",
    )
    pool_leaderboard_wallet_trade_sample: int = Field(
        default=160,
        ge=1,
        le=5000,
        description="Wallet-trade sample size used during full sweeps",
    )
    pool_incremental_wallet_trade_sample: int = Field(
        default=80,
        ge=1,
        le=5000,
        description="Wallet-trade sample size used during incremental refresh",
    )
    pool_full_sweep_interval_seconds: int = Field(
        default=1800,
        ge=10,
        le=86400,
        description="Smart-pool full sweep cadence in seconds",
    )
    pool_incremental_refresh_interval_seconds: int = Field(
        default=120,
        ge=10,
        le=86400,
        description="Smart-pool incremental refresh cadence in seconds",
    )
    pool_activity_reconciliation_interval_seconds: int = Field(
        default=120,
        ge=10,
        le=86400,
        description="Smart-pool activity reconciliation cadence in seconds",
    )
    pool_recompute_interval_seconds: int = Field(
        default=60,
        ge=10,
        le=86400,
        description="Smart-pool recompute cadence in seconds",
    )


class MaintenanceSettings(BaseModel):
    """Database maintenance configuration"""

    auto_cleanup_enabled: bool = Field(default=False, description="Enable automatic database cleanup")
    cleanup_interval_hours: int = Field(default=24, ge=1, le=168, description="Cleanup interval in hours")
    cleanup_resolved_trade_days: int = Field(
        default=30, ge=1, le=365, description="Delete resolved trades older than X days"
    )
    cleanup_trade_signal_emission_days: int = Field(
        default=21,
        ge=1,
        le=3650,
        description="Delete trade signal emission rows older than this many days",
    )
    cleanup_trade_signal_update_days: int = Field(
        default=3,
        ge=0,
        le=3650,
        description="Delete upsert-update emission rows older than this many days (0 disables)",
    )
    cleanup_wallet_activity_rollup_days: int = Field(
        default=60,
        ge=45,
        le=3650,
        description="Delete wallet activity rollup rows older than this many days",
    )
    cleanup_wallet_activity_dedupe_enabled: bool = Field(
        default=True,
        description="Run duplicate cleanup pass for wallet activity rollups during maintenance",
    )
    llm_usage_retention_days: int = Field(
        default=30,
        ge=0,
        le=3650,
        description=(
            "Delete LLM usage log rows older than this many days (0 disables). Current-month rows are always retained."
        ),
    )
    market_cache_hygiene_enabled: bool = Field(
        default=True,
        description="Enable automatic market metadata hygiene cleanup",
    )
    market_cache_hygiene_interval_hours: int = Field(
        default=6,
        ge=1,
        le=168,
        description="Market metadata hygiene run interval in hours",
    )
    market_cache_retention_days: int = Field(
        default=120,
        ge=7,
        le=3650,
        description="Delete unreferenced market metadata older than this many days",
    )
    market_cache_reference_lookback_days: int = Field(
        default=45,
        ge=1,
        le=365,
        description="Keep metadata referenced by activity/signals within this lookback",
    )
    market_cache_weak_entry_grace_days: int = Field(
        default=7,
        ge=1,
        le=180,
        description="Grace period before pruning weak cache entries missing strong identifiers",
    )
    market_cache_max_entries_per_slug: int = Field(
        default=3,
        ge=1,
        le=50,
        description="Maximum cache entries allowed per slug before collision pruning",
    )


class TradingProxySettings(BaseModel):
    """Trading VPN/Proxy configuration - routes ONLY trading requests through proxy"""

    enabled: bool = Field(default=False, description="Enable VPN proxy for trading")
    proxy_url: Optional[str] = Field(
        default=None,
        description="Proxy URL: socks5://user:pass@host:port, http://host:port",
    )
    verify_ssl: bool = Field(default=True, description="Verify SSL certs through proxy")
    timeout: float = Field(default=30.0, ge=5, le=120, description="Timeout for proxied requests (seconds)")
    require_vpn: bool = Field(default=True, description="Block trades if VPN proxy is unreachable")


class UILockSettings(BaseModel):
    """Local UI lock settings."""

    enabled: bool = Field(default=False, description="Require password unlock for the UI")
    idle_timeout_minutes: int = Field(
        default=15,
        ge=1,
        le=1440,
        description="Minutes of inactivity before the UI locks",
    )
    has_password: bool = Field(default=False, description="Whether a lock password is configured")
    password: Optional[str] = Field(default=None, description="Set a new lock password")
    clear_password: bool = Field(default=False, description="Clear existing lock password")


class NetworkSettings(BaseModel):
    """Network access configuration"""

    allow_network_access: bool = Field(
        default=False,
        description="Allow other devices on the local network to access the dashboard (requires restart)",
    )


class EventsSettings(BaseModel):
    """Events source and threshold configuration."""

    enabled: Optional[bool] = None
    interval_seconds: Optional[int] = None
    emit_trade_signals: Optional[bool] = None

    acled_api_key: Optional[str] = None
    acled_email: Optional[str] = None
    opensky_username: Optional[str] = None
    opensky_password: Optional[str] = None
    aisstream_api_key: Optional[str] = None
    cloudflare_radar_token: Optional[str] = None

    ais_enabled: Optional[bool] = None
    ais_sample_seconds: Optional[int] = None
    ais_max_messages: Optional[int] = None
    airplanes_live_enabled: Optional[bool] = None
    airplanes_live_timeout_seconds: Optional[float] = None
    airplanes_live_max_records: Optional[int] = None
    convergence_min_types: Optional[int] = None
    anomaly_threshold: Optional[float] = None
    anomaly_min_baseline_points: Optional[int] = None
    instability_signal_min: Optional[float] = None
    instability_critical: Optional[float] = None
    tension_critical: Optional[float] = None

    gdelt_query_delay_seconds: Optional[float] = None
    gdelt_max_concurrency: Optional[int] = None
    gdelt_news_enabled: Optional[bool] = None
    gdelt_news_timespan_hours: Optional[int] = None
    gdelt_news_max_records: Optional[int] = None
    gdelt_news_request_timeout_seconds: Optional[float] = None
    gdelt_news_cache_seconds: Optional[int] = None
    gdelt_news_query_delay_seconds: Optional[float] = None
    gdelt_news_sync_enabled: Optional[bool] = None
    gdelt_news_sync_hours: Optional[int] = None

    acled_rate_limit_per_min: Optional[int] = None
    acled_auth_rate_limit_per_min: Optional[int] = None
    acled_cb_max_failures: Optional[int] = None
    acled_cb_cooldown_seconds: Optional[float] = None
    usgs_enabled: Optional[bool] = None
    usgs_min_magnitude: Optional[float] = None


class SearchFilterSettings(BaseModel):
    """Opportunity search filter thresholds — controls which opportunities are shown"""

    # Hard rejection filters
    min_liquidity_hard: float = Field(default=1000.0, ge=0, description="Hard reject below this liquidity ($)")
    min_position_size: float = Field(default=50.0, ge=0, description="Reject if max position < this ($)")
    min_absolute_profit: float = Field(
        default=10.0, ge=0, description="Reject if net profit on max position < this ($)"
    )
    min_annualized_roi: float = Field(default=10.0, ge=0, description="Reject if annualized ROI < this %")
    max_resolution_months: int = Field(
        default=18,
        ge=1,
        le=120,
        description="Reject if resolution > this many months away",
    )
    max_plausible_roi: float = Field(default=30.0, ge=1, description="ROI above this % rejected as false positive")
    max_trade_legs: int = Field(default=6, ge=2, le=20, description="Maximum legs in a multi-leg trade")
    min_liquidity_per_leg: float = Field(
        default=500.0,
        ge=0,
        description="Minimum liquidity required per leg in multi-leg trades ($)",
    )

    # Risk scoring thresholds
    risk_very_short_days: int = Field(
        default=2,
        ge=0,
        le=30,
        description="Days threshold for 'very short time to resolution' risk",
    )
    risk_short_days: int = Field(
        default=7,
        ge=1,
        le=60,
        description="Days threshold for 'short time to resolution' risk",
    )
    risk_long_lockup_days: int = Field(
        default=180,
        ge=30,
        le=3650,
        description="Days threshold for 'long capital lockup' risk",
    )
    risk_extended_lockup_days: int = Field(
        default=90,
        ge=14,
        le=1825,
        description="Days threshold for 'extended capital lockup' risk",
    )
    risk_low_liquidity: float = Field(default=1000.0, ge=0, description="Liquidity below this adds high risk ($)")
    risk_moderate_liquidity: float = Field(
        default=5000.0, ge=0, description="Liquidity below this adds moderate risk ($)"
    )
    risk_complex_legs: int = Field(default=5, ge=2, le=20, description="Legs above this = complex trade risk")
    risk_multiple_legs: int = Field(default=3, ge=2, le=20, description="Legs above this = multiple positions risk")

    # BTC/ETH high-frequency
    btc_eth_hf_series_btc_15m: str = Field(default="10192", description="Polymarket series ID for BTC 15-min markets")
    btc_eth_hf_series_eth_15m: str = Field(default="10191", description="Polymarket series ID for ETH 15-min markets")
    btc_eth_hf_series_sol_15m: str = Field(default="10423", description="Polymarket series ID for SOL 15-min markets")
    btc_eth_hf_series_xrp_15m: str = Field(default="10422", description="Polymarket series ID for XRP 15-min markets")
    btc_eth_hf_series_btc_5m: str = Field(default="10684", description="Polymarket series ID for BTC 5-min markets")
    btc_eth_hf_series_eth_5m: str = Field(default="", description="Polymarket series ID for ETH 5-min markets")
    btc_eth_hf_series_sol_5m: str = Field(default="", description="Polymarket series ID for SOL 5-min markets")
    btc_eth_hf_series_xrp_5m: str = Field(default="", description="Polymarket series ID for XRP 5-min markets")
    btc_eth_hf_series_btc_1h: str = Field(default="10114", description="Polymarket series ID for BTC hourly markets")
    btc_eth_hf_series_eth_1h: str = Field(default="10117", description="Polymarket series ID for ETH hourly markets")
    btc_eth_hf_series_sol_1h: str = Field(default="10122", description="Polymarket series ID for SOL hourly markets")
    btc_eth_hf_series_xrp_1h: str = Field(default="10123", description="Polymarket series ID for XRP hourly markets")
    btc_eth_hf_series_btc_4h: str = Field(default="10331", description="Polymarket series ID for BTC 4-hour markets")
    btc_eth_hf_series_eth_4h: str = Field(default="10332", description="Polymarket series ID for ETH 4-hour markets")
    btc_eth_hf_series_sol_4h: str = Field(default="10326", description="Polymarket series ID for SOL 4-hour markets")
    btc_eth_hf_series_xrp_4h: str = Field(default="10327", description="Polymarket series ID for XRP 4-hour markets")
    btc_eth_pure_arb_max_combined: float = Field(
        default=0.98, ge=0.5, le=1.0, description="Use pure arb when YES+NO < this"
    )
    btc_eth_dump_hedge_drop_pct: float = Field(
        default=0.05,
        ge=0.01,
        le=0.5,
        description="Min price drop to trigger dump-hedge",
    )
    btc_eth_thin_liquidity_usd: float = Field(default=500.0, ge=0, description="Below this = thin order book ($)")

    # BTC/ETH high-frequency enable
    btc_eth_hf_enabled: bool = Field(default=True, description="Enable BTC/ETH high-frequency strategy")
    btc_eth_hf_maker_mode: bool = Field(
        default=True,
        description="Use maker/limit execution for BTC/ETH high-frequency strategy",
    )


class AllSettings(BaseModel):
    """Complete settings response"""

    polymarket: PolymarketSettings
    kalshi: KalshiSettings
    llm: LLMSettings
    notifications: NotificationSettings
    scanner: ScannerSettingsModel
    live_execution: LiveExecutionSettings
    maintenance: MaintenanceSettings
    discovery: DiscoverySettings
    trading_proxy: TradingProxySettings
    ui_lock: UILockSettings
    network: NetworkSettings
    events: EventsSettings
    search_filters: SearchFilterSettings
    updated_at: Optional[str] = None


class UpdateSettingsRequest(BaseModel):
    """Request to update settings (partial updates supported)"""

    polymarket: Optional[PolymarketSettings] = None
    kalshi: Optional[KalshiSettings] = None
    llm: Optional[LLMSettings] = None
    notifications: Optional[NotificationSettings] = None
    scanner: Optional[ScannerSettingsModel] = None
    live_execution: Optional[LiveExecutionSettings] = None
    maintenance: Optional[MaintenanceSettings] = None
    discovery: Optional[DiscoverySettings] = None
    trading_proxy: Optional[TradingProxySettings] = None
    ui_lock: Optional[UILockSettings] = None
    network: Optional[NetworkSettings] = None
    events: Optional[EventsSettings] = None
    search_filters: Optional[SearchFilterSettings] = None


class SettingsTransferCategory(str, Enum):
    BOT_TRADERS = "bot_traders"
    STRATEGIES = "strategies"
    DATA_SOURCES = "data_sources"
    MARKET_CREDENTIALS = "market_credentials"
    VPN_CONFIGURATION = "vpn_configuration"
    LLM_CONFIGURATION = "llm_configuration"
    TELEGRAM_CONFIGURATION = "telegram_configuration"


SETTINGS_TRANSFER_CATEGORY_ORDER: tuple[str, ...] = (
    SettingsTransferCategory.BOT_TRADERS.value,
    SettingsTransferCategory.STRATEGIES.value,
    SettingsTransferCategory.DATA_SOURCES.value,
    SettingsTransferCategory.MARKET_CREDENTIALS.value,
    SettingsTransferCategory.VPN_CONFIGURATION.value,
    SettingsTransferCategory.LLM_CONFIGURATION.value,
    SettingsTransferCategory.TELEGRAM_CONFIGURATION.value,
)

_ALLOWED_DATA_SOURCE_KINDS = {"python", "rss", "rest_api", "twitter"}
_ALLOWED_LLM_PROVIDERS = {"none", "openai", "anthropic", "google", "xai", "deepseek", "ollama", "lmstudio"}


class SettingsExportRequest(BaseModel):
    include_categories: Optional[list[SettingsTransferCategory]] = None


class SettingsImportRequest(BaseModel):
    bundle: dict[str, Any] = Field(default_factory=dict)
    include_categories: Optional[list[SettingsTransferCategory]] = None


def _coerce_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _coerce_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _coerce_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _coerce_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _coerce_categories(raw_values: Any) -> list[str]:
    if not isinstance(raw_values, list):
        return []
    categories: list[str] = []
    valid = set(SETTINGS_TRANSFER_CATEGORY_ORDER)
    for value in raw_values:
        normalized = str(value or "").strip().lower()
        if not normalized or normalized not in valid or normalized in categories:
            continue
        categories.append(normalized)
    return categories


def _resolve_export_categories(include_categories: Optional[list[SettingsTransferCategory]]) -> list[str]:
    if not include_categories:
        return list(SETTINGS_TRANSFER_CATEGORY_ORDER)
    categories: list[str] = []
    for category in include_categories:
        value = str(category.value if isinstance(category, SettingsTransferCategory) else category).strip().lower()
        if value and value not in categories and value in SETTINGS_TRANSFER_CATEGORY_ORDER:
            categories.append(value)
    if not categories:
        raise HTTPException(status_code=400, detail="At least one export category is required.")
    return categories


def _resolve_import_categories(
    include_categories: Optional[list[SettingsTransferCategory]],
    bundle: dict[str, Any],
) -> list[str]:
    if include_categories:
        categories = _resolve_export_categories(include_categories)
        if categories:
            return categories

    bundle_categories = _coerce_categories(bundle.get("categories"))
    if bundle_categories:
        return bundle_categories
    inferred = [category for category in SETTINGS_TRANSFER_CATEGORY_ORDER if category in bundle]
    if inferred:
        return inferred
    return list(SETTINGS_TRANSFER_CATEGORY_ORDER)


async def _get_or_create_settings_row(session) -> AppSettings:
    result = await session.execute(select(AppSettings).where(AppSettings.id == "default"))
    row = result.scalar_one_or_none()
    if row is None:
        row = AppSettings(id="default")
        session.add(row)
        await session.flush()
    return row


def _serialize_trader_for_transfer(trader: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(trader.get("name") or "").strip(),
        "description": trader.get("description"),
        "mode": str(trader.get("mode") or "shadow").strip().lower() or "shadow",
        "source_configs": [dict(item) for item in trader.get("source_configs") or [] if isinstance(item, dict)],
        "risk_limits": _coerce_dict(trader.get("risk_limits")),
        "metadata": _coerce_dict(trader.get("metadata")),
        "is_enabled": bool(trader.get("is_enabled", True)),
        "is_paused": bool(trader.get("is_paused", False)),
        "interval_seconds": int(trader.get("interval_seconds") or 60),
    }


def _serialize_trader_order_for_transfer(row: TraderOrder, trader_name: str) -> dict[str, Any]:
    return {
        "id": str(row.id or ""),
        "trader_name": trader_name,
        "signal_id": _coerce_string(row.signal_id),
        "decision_id": _coerce_string(row.decision_id),
        "source": _coerce_string(row.source) or "manual",
        "strategy_key": _coerce_string(row.strategy_key),
        "strategy_version": int(row.strategy_version) if row.strategy_version is not None else None,
        "market_id": _coerce_string(row.market_id) or "",
        "market_question": row.market_question,
        "direction": row.direction,
        "event_id": _coerce_string(row.event_id),
        "trace_id": _coerce_string(row.trace_id),
        "mode": _coerce_string(row.mode) or "shadow",
        "status": _coerce_string(row.status) or "submitted",
        "notional_usd": float(row.notional_usd) if row.notional_usd is not None else None,
        "entry_price": float(row.entry_price) if row.entry_price is not None else None,
        "effective_price": float(row.effective_price) if row.effective_price is not None else None,
        "edge_percent": float(row.edge_percent) if row.edge_percent is not None else None,
        "confidence": float(row.confidence) if row.confidence is not None else None,
        "actual_profit": float(row.actual_profit) if row.actual_profit is not None else None,
        "reason": row.reason,
        "payload": _coerce_dict(row.payload_json),
        "error_message": row.error_message,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "executed_at": row.executed_at.isoformat() if row.executed_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def _serialize_live_runtime_state_for_transfer(row: LiveTradingRuntimeState) -> dict[str, Any]:
    return {
        "id": str(row.id or ""),
        "wallet_address": _coerce_string(row.wallet_address) or "",
        "total_trades": int(row.total_trades or 0),
        "winning_trades": int(row.winning_trades or 0),
        "losing_trades": int(row.losing_trades or 0),
        "total_volume": float(row.total_volume or 0.0),
        "total_pnl": float(row.total_pnl or 0.0),
        "daily_volume": float(row.daily_volume or 0.0),
        "daily_pnl": float(row.daily_pnl or 0.0),
        "open_positions": int(row.open_positions or 0),
        "last_trade_at": row.last_trade_at.isoformat() if row.last_trade_at else None,
        "daily_volume_reset_at": row.daily_volume_reset_at.isoformat() if row.daily_volume_reset_at else None,
        "market_positions_json": _coerce_dict(row.market_positions_json),
        "balance_signature_type": int(row.balance_signature_type) if row.balance_signature_type is not None else None,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def _serialize_live_order_for_transfer(row: LiveTradingOrder) -> dict[str, Any]:
    return {
        "id": str(row.id or ""),
        "wallet_address": _coerce_string(row.wallet_address) or "",
        "clob_order_id": _coerce_string(row.clob_order_id),
        "token_id": _coerce_string(row.token_id) or "",
        "side": _coerce_string(row.side) or "BUY",
        "price": float(row.price or 0.0),
        "size": float(row.size or 0.0),
        "order_type": _coerce_string(row.order_type) or "GTC",
        "status": _coerce_string(row.status) or "pending",
        "filled_size": float(row.filled_size or 0.0),
        "average_fill_price": float(row.average_fill_price or 0.0),
        "market_question": row.market_question,
        "opportunity_id": _coerce_string(row.opportunity_id),
        "error_message": row.error_message,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def _serialize_live_position_for_transfer(row: LiveTradingPosition) -> dict[str, Any]:
    return {
        "id": str(row.id or ""),
        "wallet_address": _coerce_string(row.wallet_address) or "",
        "token_id": _coerce_string(row.token_id) or "",
        "market_id": _coerce_string(row.market_id) or "",
        "market_question": row.market_question,
        "outcome": _coerce_string(row.outcome),
        "size": float(row.size or 0.0),
        "average_cost": float(row.average_cost or 0.0),
        "current_price": float(row.current_price or 0.0),
        "unrealized_pnl": float(row.unrealized_pnl or 0.0),
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def _serialize_orchestrator_control_for_transfer(row: TraderOrchestratorControl) -> dict[str, Any]:
    return {
        "is_enabled": bool(row.is_enabled),
        "is_paused": bool(row.is_paused),
        "mode": _coerce_string(row.mode) or "shadow",
        "run_interval_seconds": int(row.run_interval_seconds or 5),
        "kill_switch": bool(row.kill_switch),
        "settings_json": _coerce_dict(row.settings_json),
    }


def _strategy_version_to_transfer_payload(row: StrategyVersion) -> dict[str, Any]:
    return {
        "version": int(row.version or 1),
        "is_latest": bool(row.is_latest),
        "name": row.name,
        "description": row.description,
        "source_code": str(row.source_code or ""),
        "class_name": row.class_name,
        "config": _coerce_dict(row.config),
        "config_schema": _coerce_dict(row.config_schema),
        "aliases": list(row.aliases or []),
        "enabled": bool(row.enabled),
        "is_system": bool(row.is_system),
        "sort_order": int(row.sort_order or 0),
        "parent_version": int(row.parent_version) if row.parent_version is not None else None,
        "created_by": row.created_by,
        "reason": row.reason,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def _strategy_row_to_transfer_payload(row: Strategy, versions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "slug": str(row.slug or "").strip().lower(),
        "source_key": str(row.source_key or "scanner").strip().lower() or "scanner",
        "name": str(row.name or row.slug or "").strip(),
        "description": row.description,
        "source_code": str(row.source_code or ""),
        "class_name": _coerce_string(row.class_name),
        "is_system": bool(row.is_system),
        "enabled": bool(row.enabled),
        "config": _coerce_dict(row.config),
        "config_schema": _coerce_dict(row.config_schema),
        "aliases": list(row.aliases or []),
        "version": int(row.version or 1),
        "sort_order": int(row.sort_order or 0),
        "versions": versions,
    }


def _data_source_row_to_transfer_payload(row: DataSource) -> dict[str, Any]:
    return {
        "slug": str(row.slug or "").strip().lower(),
        "source_key": str(row.source_key or "custom").strip().lower() or "custom",
        "source_kind": str(row.source_kind or "python").strip().lower() or "python",
        "name": str(row.name or row.slug or "").strip(),
        "description": row.description,
        "source_code": str(row.source_code or ""),
        "class_name": _coerce_string(row.class_name),
        "is_system": bool(row.is_system),
        "enabled": bool(row.enabled),
        "retention": _coerce_dict(row.retention),
        "config": _coerce_dict(row.config),
        "config_schema": _coerce_dict(row.config_schema),
        "version": int(row.version or 1),
        "sort_order": int(row.sort_order or 0),
    }


async def _export_transfer_bundle(categories: list[str]) -> tuple[dict[str, Any], dict[str, int]]:
    from services.trader_orchestrator_state import list_traders

    bundle: dict[str, Any] = {
        "schema_version": 1,
        "exported_at": utcnow().isoformat(),
        "categories": list(categories),
    }
    counts = {category: 0 for category in SETTINGS_TRANSFER_CATEGORY_ORDER}

    async with AsyncSessionLocal() as session:
        settings_row: AppSettings | None = None
        needs_settings = any(
            category in categories
            for category in (
                SettingsTransferCategory.MARKET_CREDENTIALS.value,
                SettingsTransferCategory.VPN_CONFIGURATION.value,
                SettingsTransferCategory.LLM_CONFIGURATION.value,
                SettingsTransferCategory.TELEGRAM_CONFIGURATION.value,
            )
        )
        if needs_settings:
            settings_row = await _get_or_create_settings_row(session)

        if SettingsTransferCategory.BOT_TRADERS.value in categories:
            traders = await list_traders(session)
            exported_traders = [_serialize_trader_for_transfer(trader) for trader in traders]
            trader_name_by_id = {
                str(trader.get("id") or "").strip(): str(trader.get("name") or "").strip()
                for trader in traders
                if str(trader.get("id") or "").strip() and str(trader.get("name") or "").strip()
            }
            trader_ids = list(trader_name_by_id.keys())
            exported_orders: list[dict[str, Any]] = []
            if trader_ids:
                order_rows = (
                    (
                        await session.execute(
                            select(TraderOrder)
                            .where(TraderOrder.trader_id.in_(trader_ids))
                            .order_by(TraderOrder.created_at.asc(), TraderOrder.id.asc())
                        )
                    )
                    .scalars()
                    .all()
                )
                for order_row in order_rows:
                    trader_name = trader_name_by_id.get(str(order_row.trader_id or "").strip())
                    if not trader_name:
                        continue
                    exported_orders.append(_serialize_trader_order_for_transfer(order_row, trader_name))

            live_runtime_rows = (
                (
                    await session.execute(
                        select(LiveTradingRuntimeState).order_by(
                            LiveTradingRuntimeState.wallet_address.asc(),
                            LiveTradingRuntimeState.updated_at.asc(),
                        )
                    )
                )
                .scalars()
                .all()
            )
            exported_live_runtime_states = [
                _serialize_live_runtime_state_for_transfer(row) for row in live_runtime_rows
            ]

            live_order_rows = (
                (
                    await session.execute(
                        select(LiveTradingOrder).order_by(
                            LiveTradingOrder.wallet_address.asc(),
                            LiveTradingOrder.created_at.asc(),
                            LiveTradingOrder.id.asc(),
                        )
                    )
                )
                .scalars()
                .all()
            )
            exported_live_orders = [_serialize_live_order_for_transfer(row) for row in live_order_rows]

            live_position_rows = (
                (
                    await session.execute(
                        select(LiveTradingPosition).order_by(
                            LiveTradingPosition.wallet_address.asc(),
                            LiveTradingPosition.market_id.asc(),
                            LiveTradingPosition.token_id.asc(),
                        )
                    )
                )
                .scalars()
                .all()
            )
            exported_live_positions = [_serialize_live_position_for_transfer(row) for row in live_position_rows]

            orchestrator_row = await session.get(TraderOrchestratorControl, "default")
            bundle[SettingsTransferCategory.BOT_TRADERS.value] = {
                "traders": exported_traders,
                "orchestrator": (
                    _serialize_orchestrator_control_for_transfer(orchestrator_row) if orchestrator_row is not None else None
                ),
                "trade_state": {
                    "orders": exported_orders,
                    "live_wallet_state": {
                        "runtime_states": exported_live_runtime_states,
                        "orders": exported_live_orders,
                        "positions": exported_live_positions,
                    },
                },
            }
            counts[SettingsTransferCategory.BOT_TRADERS.value] = len(exported_traders)

        if SettingsTransferCategory.STRATEGIES.value in categories:
            strategy_rows = (
                (
                    await session.execute(
                        select(Strategy).order_by(Strategy.sort_order.asc(), Strategy.slug.asc())
                    )
                )
                .scalars()
                .all()
            )
            versions_by_strategy_id: dict[str, list[dict[str, Any]]] = {}
            strategy_ids = [str(row.id) for row in strategy_rows if str(row.id or "").strip()]
            if strategy_ids:
                version_rows = (
                    (
                        await session.execute(
                            select(StrategyVersion)
                            .where(StrategyVersion.strategy_id.in_(strategy_ids))
                            .order_by(StrategyVersion.strategy_slug.asc(), StrategyVersion.version.asc())
                        )
                    )
                    .scalars()
                    .all()
                )
                for version_row in version_rows:
                    strategy_id = str(version_row.strategy_id or "").strip()
                    if not strategy_id:
                        continue
                    versions_by_strategy_id.setdefault(strategy_id, []).append(
                        _strategy_version_to_transfer_payload(version_row)
                    )

            exported_strategies = [
                _strategy_row_to_transfer_payload(
                    row,
                    versions_by_strategy_id.get(str(row.id), []),
                )
                for row in strategy_rows
            ]
            bundle[SettingsTransferCategory.STRATEGIES.value] = exported_strategies
            counts[SettingsTransferCategory.STRATEGIES.value] = len(exported_strategies)

        if SettingsTransferCategory.DATA_SOURCES.value in categories:
            source_rows = (
                (
                    await session.execute(
                        select(DataSource).order_by(DataSource.sort_order.asc(), DataSource.slug.asc())
                    )
                )
                .scalars()
                .all()
            )
            exported_sources = [_data_source_row_to_transfer_payload(row) for row in source_rows]
            bundle[SettingsTransferCategory.DATA_SOURCES.value] = exported_sources
            counts[SettingsTransferCategory.DATA_SOURCES.value] = len(exported_sources)

        if SettingsTransferCategory.MARKET_CREDENTIALS.value in categories:
            if settings_row is None:
                settings_row = await _get_or_create_settings_row(session)
            market_credentials = {
                "polymarket": {
                    "api_key": decrypt_secret(settings_row.polymarket_api_key),
                    "api_secret": decrypt_secret(settings_row.polymarket_api_secret),
                    "api_passphrase": decrypt_secret(settings_row.polymarket_api_passphrase),
                    "private_key": decrypt_secret(settings_row.polymarket_private_key),
                },
                "kalshi": {
                    "email": _coerce_string(settings_row.kalshi_email),
                    "password": decrypt_secret(settings_row.kalshi_password),
                    "api_key": decrypt_secret(settings_row.kalshi_api_key),
                },
            }
            bundle[SettingsTransferCategory.MARKET_CREDENTIALS.value] = market_credentials
            counts[SettingsTransferCategory.MARKET_CREDENTIALS.value] = 1

        if SettingsTransferCategory.VPN_CONFIGURATION.value in categories:
            if settings_row is None:
                settings_row = await _get_or_create_settings_row(session)
            vpn_configuration = {
                "enabled": bool(settings_row.trading_proxy_enabled),
                "proxy_url": decrypt_secret(settings_row.trading_proxy_url),
                "verify_ssl": bool(getattr(settings_row, "trading_proxy_verify_ssl", True)),
                "timeout": float(getattr(settings_row, "trading_proxy_timeout", 30.0) or 30.0),
                "require_vpn": bool(getattr(settings_row, "trading_proxy_require_vpn", True)),
            }
            bundle[SettingsTransferCategory.VPN_CONFIGURATION.value] = vpn_configuration
            counts[SettingsTransferCategory.VPN_CONFIGURATION.value] = 1

        if SettingsTransferCategory.LLM_CONFIGURATION.value in categories:
            if settings_row is None:
                settings_row = await _get_or_create_settings_row(session)
            llm_configuration = {
                "provider": str(settings_row.llm_provider or "none").strip().lower() or "none",
                "model": _coerce_string(settings_row.llm_model),
                "max_monthly_spend": float(getattr(settings_row, "ai_max_monthly_spend", 50.0) or 0.0),
                "openai_api_key": decrypt_secret(settings_row.openai_api_key),
                "anthropic_api_key": decrypt_secret(settings_row.anthropic_api_key),
                "google_api_key": decrypt_secret(settings_row.google_api_key),
                "xai_api_key": decrypt_secret(settings_row.xai_api_key),
                "deepseek_api_key": decrypt_secret(settings_row.deepseek_api_key),
                "ollama_api_key": decrypt_secret(settings_row.ollama_api_key),
                "ollama_base_url": _coerce_string(settings_row.ollama_base_url),
                "lmstudio_api_key": decrypt_secret(settings_row.lmstudio_api_key),
                "lmstudio_base_url": _coerce_string(settings_row.lmstudio_base_url),
            }
            bundle[SettingsTransferCategory.LLM_CONFIGURATION.value] = llm_configuration
            counts[SettingsTransferCategory.LLM_CONFIGURATION.value] = 1

        if SettingsTransferCategory.TELEGRAM_CONFIGURATION.value in categories:
            if settings_row is None:
                settings_row = await _get_or_create_settings_row(session)
            notify_min_roi_raw = getattr(settings_row, "notify_min_roi", None)
            interval_minutes_raw = getattr(settings_row, "notify_autotrader_summary_interval_minutes", None)
            telegram_configuration = {
                "enabled": bool(getattr(settings_row, "notifications_enabled", False)),
                "telegram_bot_token": decrypt_secret(getattr(settings_row, "telegram_bot_token", None)),
                "telegram_chat_id": _coerce_string(getattr(settings_row, "telegram_chat_id", None)),
                "notify_on_opportunity": bool(getattr(settings_row, "notify_on_opportunity", True)),
                "notify_on_trade": bool(getattr(settings_row, "notify_on_trade", True)),
                "notify_min_roi": float(notify_min_roi_raw if notify_min_roi_raw is not None else 5.0),
                "notify_autotrader_orders": bool(getattr(settings_row, "notify_autotrader_orders", False)),
                "notify_autotrader_closes": bool(getattr(settings_row, "notify_autotrader_closes", True)),
                "notify_autotrader_issues": bool(getattr(settings_row, "notify_autotrader_issues", True)),
                "notify_autotrader_timeline": bool(getattr(settings_row, "notify_autotrader_timeline", True)),
                "notify_autotrader_summary_interval_minutes": int(
                    interval_minutes_raw if interval_minutes_raw is not None else 60
                ),
                "notify_autotrader_summary_per_trader": bool(
                    getattr(settings_row, "notify_autotrader_summary_per_trader", False)
                ),
            }
            bundle[SettingsTransferCategory.TELEGRAM_CONFIGURATION.value] = telegram_configuration
            counts[SettingsTransferCategory.TELEGRAM_CONFIGURATION.value] = 1

    return bundle, counts


def _apply_market_credentials_import(settings_row: AppSettings, payload: dict[str, Any]) -> None:
    polymarket = _coerce_dict(payload.get("polymarket"))
    kalshi = _coerce_dict(payload.get("kalshi"))

    set_encrypted_secret(settings_row, "polymarket_api_key", _coerce_string(polymarket.get("api_key")))
    set_encrypted_secret(settings_row, "polymarket_api_secret", _coerce_string(polymarket.get("api_secret")))
    set_encrypted_secret(settings_row, "polymarket_api_passphrase", _coerce_string(polymarket.get("api_passphrase")))
    set_encrypted_secret(settings_row, "polymarket_private_key", _coerce_string(polymarket.get("private_key")))

    settings_row.kalshi_email = _coerce_string(kalshi.get("email"))
    set_encrypted_secret(settings_row, "kalshi_password", _coerce_string(kalshi.get("password")))
    set_encrypted_secret(settings_row, "kalshi_api_key", _coerce_string(kalshi.get("api_key")))


def _apply_vpn_configuration_import(settings_row: AppSettings, payload: dict[str, Any]) -> None:
    timeout_raw = payload.get("timeout")
    timeout_value = 30.0
    if timeout_raw is not None:
        try:
            timeout_value = float(timeout_raw)
        except (TypeError, ValueError):
            timeout_value = 30.0
    timeout_value = max(5.0, min(120.0, timeout_value))

    settings_row.trading_proxy_enabled = bool(payload.get("enabled", False))
    settings_row.trading_proxy_verify_ssl = bool(payload.get("verify_ssl", True))
    settings_row.trading_proxy_timeout = timeout_value
    settings_row.trading_proxy_require_vpn = bool(payload.get("require_vpn", True))
    set_encrypted_secret(settings_row, "trading_proxy_url", _coerce_string(payload.get("proxy_url")))


def _apply_llm_configuration_import(settings_row: AppSettings, payload: dict[str, Any]) -> None:
    provider = str(payload.get("provider") or "none").strip().lower() or "none"
    if provider not in _ALLOWED_LLM_PROVIDERS:
        raise ValueError(f"Unsupported LLM provider '{provider}'.")

    monthly_spend_raw = payload.get("max_monthly_spend")
    monthly_spend = 0.0
    if monthly_spend_raw is not None:
        try:
            monthly_spend = float(monthly_spend_raw)
        except (TypeError, ValueError):
            monthly_spend = 0.0
    monthly_spend = max(0.0, monthly_spend)

    settings_row.llm_provider = provider
    settings_row.llm_model = _coerce_string(payload.get("model"))
    settings_row.ai_default_model = _coerce_string(payload.get("model"))
    settings_row.ai_max_monthly_spend = monthly_spend
    settings_row.ollama_base_url = _coerce_string(payload.get("ollama_base_url"))
    settings_row.lmstudio_base_url = _coerce_string(payload.get("lmstudio_base_url"))

    set_encrypted_secret(settings_row, "openai_api_key", _coerce_string(payload.get("openai_api_key")))
    set_encrypted_secret(settings_row, "anthropic_api_key", _coerce_string(payload.get("anthropic_api_key")))
    set_encrypted_secret(settings_row, "google_api_key", _coerce_string(payload.get("google_api_key")))
    set_encrypted_secret(settings_row, "xai_api_key", _coerce_string(payload.get("xai_api_key")))
    set_encrypted_secret(settings_row, "deepseek_api_key", _coerce_string(payload.get("deepseek_api_key")))
    set_encrypted_secret(settings_row, "ollama_api_key", _coerce_string(payload.get("ollama_api_key")))
    set_encrypted_secret(settings_row, "lmstudio_api_key", _coerce_string(payload.get("lmstudio_api_key")))


def _apply_telegram_configuration_import(settings_row: AppSettings, payload: dict[str, Any]) -> None:
    notify_min_roi_raw = payload.get("notify_min_roi")
    notify_min_roi = 5.0
    if notify_min_roi_raw is not None:
        try:
            notify_min_roi = float(notify_min_roi_raw)
        except (TypeError, ValueError):
            notify_min_roi = 5.0
    notify_min_roi = max(0.0, notify_min_roi)

    summary_interval_raw = payload.get("notify_autotrader_summary_interval_minutes")
    summary_interval_minutes = 60
    if summary_interval_raw is not None:
        try:
            summary_interval_minutes = int(summary_interval_raw)
        except (TypeError, ValueError):
            summary_interval_minutes = 60
    summary_interval_minutes = max(5, min(1440, summary_interval_minutes))

    settings_row.notifications_enabled = bool(payload.get("enabled", False))
    set_encrypted_secret(settings_row, "telegram_bot_token", _coerce_string(payload.get("telegram_bot_token")))
    settings_row.telegram_chat_id = _coerce_string(payload.get("telegram_chat_id"))
    settings_row.notify_on_opportunity = bool(payload.get("notify_on_opportunity", True))
    settings_row.notify_on_trade = bool(payload.get("notify_on_trade", True))
    settings_row.notify_min_roi = notify_min_roi
    settings_row.notify_autotrader_orders = bool(payload.get("notify_autotrader_orders", False))
    settings_row.notify_autotrader_closes = bool(payload.get("notify_autotrader_closes", True))
    settings_row.notify_autotrader_issues = bool(payload.get("notify_autotrader_issues", True))
    settings_row.notify_autotrader_timeline = bool(payload.get("notify_autotrader_timeline", True))
    settings_row.notify_autotrader_summary_interval_minutes = summary_interval_minutes
    settings_row.notify_autotrader_summary_per_trader = bool(payload.get("notify_autotrader_summary_per_trader", False))


def _normalize_strategy_payload(raw: dict[str, Any]) -> dict[str, Any]:
    from services.strategy_loader import validate_strategy_source

    slug = str(raw.get("slug") or "").strip().lower()
    if not slug:
        raise ValueError("Each strategy requires a non-empty slug.")

    source_code = str(raw.get("source_code") or "")
    if len(source_code.strip()) < 10:
        raise ValueError(f"Strategy '{slug}' has invalid source_code.")

    requested_class_name = _coerce_string(raw.get("class_name"))
    validation = validate_strategy_source(source_code, class_name=requested_class_name)
    if not bool(validation.get("valid")):
        errors = "; ".join(str(error) for error in list(validation.get("errors") or [])) or "unknown validation error"
        raise ValueError(f"Strategy '{slug}' failed source validation: {errors}")

    parsed_version = normalize_strategy_version(raw.get("version"))
    version = int(parsed_version or 1)
    if version <= 0:
        version = 1

    source_key = str(raw.get("source_key") or "scanner").strip().lower() or "scanner"
    sort_order_raw = raw.get("sort_order")
    try:
        sort_order = int(sort_order_raw if sort_order_raw is not None else 0)
    except (TypeError, ValueError):
        sort_order = 0

    aliases: list[str] = []
    for alias in raw.get("aliases") or []:
        normalized_alias = _coerce_string(alias)
        if normalized_alias and normalized_alias not in aliases:
            aliases.append(normalized_alias)

    return {
        "slug": slug,
        "source_key": source_key,
        "name": _coerce_string(raw.get("name")) or slug,
        "description": _coerce_string(raw.get("description")),
        "source_code": source_code,
        "class_name": requested_class_name or _coerce_string(validation.get("class_name")),
        "is_system": bool(raw.get("is_system", False)),
        "enabled": bool(raw.get("enabled", True)),
        "config": _coerce_dict(raw.get("config")),
        "config_schema": _coerce_dict(raw.get("config_schema")),
        "aliases": aliases,
        "version": version,
        "sort_order": sort_order,
        "versions": _coerce_dict_list(raw.get("versions")),
    }


def _normalize_data_source_payload(raw: dict[str, Any]) -> dict[str, Any]:
    from services.data_source_loader import validate_data_source_source

    slug = str(raw.get("slug") or "").strip().lower()
    if not slug:
        raise ValueError("Each data source requires a non-empty slug.")

    source_code = str(raw.get("source_code") or "")
    if len(source_code.strip()) < 10:
        raise ValueError(f"Data source '{slug}' has invalid source_code.")

    source_kind = str(raw.get("source_kind") or "python").strip().lower() or "python"
    if source_kind not in _ALLOWED_DATA_SOURCE_KINDS:
        raise ValueError(f"Data source '{slug}' has unsupported source_kind '{source_kind}'.")

    requested_class_name = _coerce_string(raw.get("class_name"))
    validation = validate_data_source_source(source_code, class_name=requested_class_name)
    if not bool(validation.get("valid")):
        errors = "; ".join(str(error) for error in list(validation.get("errors") or [])) or "unknown validation error"
        raise ValueError(f"Data source '{slug}' failed source validation: {errors}")

    version_raw = raw.get("version")
    try:
        version = int(version_raw if version_raw is not None else 1)
    except (TypeError, ValueError):
        version = 1
    version = max(1, version)

    sort_order_raw = raw.get("sort_order")
    try:
        sort_order = int(sort_order_raw if sort_order_raw is not None else 0)
    except (TypeError, ValueError):
        sort_order = 0

    return {
        "slug": slug,
        "source_key": str(raw.get("source_key") or "custom").strip().lower() or "custom",
        "source_kind": source_kind,
        "name": _coerce_string(raw.get("name")) or slug,
        "description": _coerce_string(raw.get("description")),
        "source_code": source_code,
        "class_name": requested_class_name or _coerce_string(validation.get("class_name")),
        "is_system": bool(raw.get("is_system", False)),
        "enabled": bool(raw.get("enabled", True)),
        "retention": _coerce_dict(raw.get("retention")),
        "config": _coerce_dict(raw.get("config")),
        "config_schema": _coerce_dict(raw.get("config_schema")),
        "version": version,
        "sort_order": sort_order,
    }


def _normalize_trader_payload(raw: dict[str, Any]) -> dict[str, Any]:
    name = _coerce_string(raw.get("name"))
    if not name:
        raise ValueError("Each trader requires a non-empty name.")

    interval_raw = raw.get("interval_seconds")
    try:
        interval_seconds = int(interval_raw if interval_raw is not None else 60)
    except (TypeError, ValueError):
        interval_seconds = 60
    interval_seconds = max(1, min(86400, interval_seconds))

    return {
        "name": name,
        "description": _coerce_string(raw.get("description")),
        "mode": str(raw.get("mode") or "shadow").strip().lower() or "shadow",
        "source_configs": _coerce_dict_list(raw.get("source_configs")),
        "risk_limits": _coerce_dict(raw.get("risk_limits")),
        "metadata": _coerce_dict(raw.get("metadata")),
        "is_enabled": bool(raw.get("is_enabled", True)),
        "is_paused": bool(raw.get("is_paused", False)),
        "interval_seconds": interval_seconds,
    }


def _normalize_trader_order_payload(raw: dict[str, Any]) -> dict[str, Any]:
    trader_name = _coerce_string(raw.get("trader_name"))
    if not trader_name:
        raise ValueError("Each trader trade requires a non-empty trader_name.")

    market_id = _coerce_string(raw.get("market_id"))
    if not market_id:
        raise ValueError(f"Trader '{trader_name}' trade is missing market_id.")

    created_at = _coerce_datetime(raw.get("created_at")) or utcnow()
    updated_at = _coerce_datetime(raw.get("updated_at")) or created_at
    executed_at = _coerce_datetime(raw.get("executed_at"))

    strategy_version = None
    if raw.get("strategy_version") is not None:
        strategy_version = _coerce_int(raw.get("strategy_version"), 0) or None

    notional_usd = _coerce_optional_float(raw.get("notional_usd"))
    entry_price = _coerce_optional_float(raw.get("entry_price"))
    effective_price = _coerce_optional_float(raw.get("effective_price"))
    edge_percent = _coerce_optional_float(raw.get("edge_percent"))
    confidence = _coerce_optional_float(raw.get("confidence"))
    actual_profit = _coerce_optional_float(raw.get("actual_profit"))

    payload = _coerce_dict(raw.get("payload"))
    import_origin = _coerce_dict(payload.get("import_origin"))
    source_order_id = _coerce_string(raw.get("id"))
    if source_order_id:
        import_origin["source_order_id"] = source_order_id
    source_signal_id = _coerce_string(raw.get("signal_id"))
    if source_signal_id:
        import_origin["source_signal_id"] = source_signal_id
    source_decision_id = _coerce_string(raw.get("decision_id"))
    if source_decision_id:
        import_origin["source_decision_id"] = source_decision_id
    if import_origin:
        payload["import_origin"] = import_origin

    return {
        "trader_name": trader_name,
        "source": (_coerce_string(raw.get("source")) or "manual").lower(),
        "strategy_key": _coerce_string(raw.get("strategy_key")),
        "strategy_version": strategy_version,
        "market_id": market_id,
        "market_question": _coerce_string(raw.get("market_question")),
        "direction": _coerce_string(raw.get("direction")),
        "event_id": _coerce_string(raw.get("event_id")),
        "trace_id": _coerce_string(raw.get("trace_id")),
        "mode": (_coerce_string(raw.get("mode")) or "shadow").lower(),
        "status": (_coerce_string(raw.get("status")) or "submitted").lower(),
        "notional_usd": notional_usd,
        "entry_price": entry_price,
        "effective_price": effective_price,
        "edge_percent": edge_percent,
        "confidence": confidence,
        "actual_profit": actual_profit,
        "reason": _coerce_string(raw.get("reason")),
        "payload": payload,
        "error_message": _coerce_string(raw.get("error_message")),
        "created_at": created_at,
        "executed_at": executed_at,
        "updated_at": updated_at,
    }


def _normalize_live_runtime_state_payload(raw: dict[str, Any]) -> dict[str, Any]:
    wallet_address = str(raw.get("wallet_address") or "").strip().lower()
    if not wallet_address:
        raise ValueError("Live runtime state requires wallet_address.")

    created_at = _coerce_datetime(raw.get("created_at")) or utcnow()
    updated_at = _coerce_datetime(raw.get("updated_at")) or created_at
    runtime_id = _coerce_string(raw.get("id")) or f"wallet:{wallet_address}"

    balance_signature_type = None
    if raw.get("balance_signature_type") is not None:
        balance_signature_type = _coerce_int(raw.get("balance_signature_type"), 0)

    return {
        "id": runtime_id,
        "wallet_address": wallet_address,
        "total_trades": max(0, _coerce_int(raw.get("total_trades"), 0)),
        "winning_trades": max(0, _coerce_int(raw.get("winning_trades"), 0)),
        "losing_trades": max(0, _coerce_int(raw.get("losing_trades"), 0)),
        "total_volume": float(_coerce_optional_float(raw.get("total_volume")) or 0.0),
        "total_pnl": float(_coerce_optional_float(raw.get("total_pnl")) or 0.0),
        "daily_volume": float(_coerce_optional_float(raw.get("daily_volume")) or 0.0),
        "daily_pnl": float(_coerce_optional_float(raw.get("daily_pnl")) or 0.0),
        "open_positions": max(0, _coerce_int(raw.get("open_positions"), 0)),
        "last_trade_at": _coerce_datetime(raw.get("last_trade_at")),
        "daily_volume_reset_at": _coerce_datetime(raw.get("daily_volume_reset_at")),
        "market_positions_json": _coerce_dict(raw.get("market_positions_json")),
        "balance_signature_type": balance_signature_type,
        "created_at": created_at,
        "updated_at": updated_at,
    }


def _normalize_live_order_payload(raw: dict[str, Any]) -> dict[str, Any]:
    wallet_address = str(raw.get("wallet_address") or "").strip().lower()
    if not wallet_address:
        raise ValueError("Live order requires wallet_address.")

    token_id = _coerce_string(raw.get("token_id"))
    if not token_id:
        raise ValueError("Live order requires token_id.")

    created_at = _coerce_datetime(raw.get("created_at")) or utcnow()
    updated_at = _coerce_datetime(raw.get("updated_at")) or created_at
    order_id = _coerce_string(raw.get("id")) or uuid.uuid4().hex

    return {
        "id": order_id,
        "wallet_address": wallet_address,
        "clob_order_id": _coerce_string(raw.get("clob_order_id")),
        "token_id": token_id,
        "side": (_coerce_string(raw.get("side")) or "BUY").upper(),
        "price": float(_coerce_optional_float(raw.get("price")) or 0.0),
        "size": float(_coerce_optional_float(raw.get("size")) or 0.0),
        "order_type": (_coerce_string(raw.get("order_type")) or "GTC").upper(),
        "status": (_coerce_string(raw.get("status")) or "pending").lower(),
        "filled_size": float(_coerce_optional_float(raw.get("filled_size")) or 0.0),
        "average_fill_price": float(_coerce_optional_float(raw.get("average_fill_price")) or 0.0),
        "market_question": _coerce_string(raw.get("market_question")),
        "opportunity_id": _coerce_string(raw.get("opportunity_id")),
        "error_message": _coerce_string(raw.get("error_message")),
        "created_at": created_at,
        "updated_at": updated_at,
    }


def _normalize_live_position_payload(raw: dict[str, Any]) -> dict[str, Any]:
    wallet_address = str(raw.get("wallet_address") or "").strip().lower()
    if not wallet_address:
        raise ValueError("Live position requires wallet_address.")

    token_id = _coerce_string(raw.get("token_id"))
    if not token_id:
        raise ValueError("Live position requires token_id.")

    market_id = _coerce_string(raw.get("market_id")) or token_id
    created_at = _coerce_datetime(raw.get("created_at")) or utcnow()
    updated_at = _coerce_datetime(raw.get("updated_at")) or created_at
    position_id = _coerce_string(raw.get("id")) or f"{wallet_address}:{token_id}"

    return {
        "id": position_id,
        "wallet_address": wallet_address,
        "token_id": token_id,
        "market_id": market_id,
        "market_question": _coerce_string(raw.get("market_question")),
        "outcome": _coerce_string(raw.get("outcome")),
        "size": float(_coerce_optional_float(raw.get("size")) or 0.0),
        "average_cost": float(_coerce_optional_float(raw.get("average_cost")) or 0.0),
        "current_price": float(_coerce_optional_float(raw.get("current_price")) or 0.0),
        "unrealized_pnl": float(_coerce_optional_float(raw.get("unrealized_pnl")) or 0.0),
        "created_at": created_at,
        "updated_at": updated_at,
    }


async def _import_live_wallet_trade_state(session, payload: dict[str, Any]) -> dict[str, int]:
    normalized_runtime_rows = [
        _normalize_live_runtime_state_payload(item)
        for item in _coerce_dict_list(payload.get("runtime_states"))
    ]
    normalized_order_rows = [
        _normalize_live_order_payload(item)
        for item in _coerce_dict_list(payload.get("orders"))
    ]
    normalized_position_rows = [
        _normalize_live_position_payload(item)
        for item in _coerce_dict_list(payload.get("positions"))
    ]

    runtime_by_id: dict[str, dict[str, Any]] = {}
    for row in normalized_runtime_rows:
        runtime_by_id[str(row["id"])] = row
    order_by_id: dict[str, dict[str, Any]] = {}
    for row in normalized_order_rows:
        order_by_id[str(row["id"])] = row
    positions_by_wallet_token: dict[tuple[str, str], dict[str, Any]] = {}
    for row in normalized_position_rows:
        key = (str(row["wallet_address"]), str(row["token_id"]))
        positions_by_wallet_token[key] = row

    runtime_rows = list(runtime_by_id.values())
    order_rows = list(order_by_id.values())
    position_rows = list(positions_by_wallet_token.values())

    wallet_addresses = {
        str(wallet).strip().lower()
        for wallet in (
            [row["wallet_address"] for row in runtime_rows]
            + [row["wallet_address"] for row in order_rows]
            + [row["wallet_address"] for row in position_rows]
        )
        if str(wallet).strip()
    }
    if wallet_addresses:
        wallets = tuple(sorted(wallet_addresses))
        await session.execute(
            delete(LiveTradingRuntimeState).where(
                func.lower(func.coalesce(LiveTradingRuntimeState.wallet_address, "")).in_(wallets)
            )
        )
        await session.execute(
            delete(LiveTradingOrder).where(
                func.lower(func.coalesce(LiveTradingOrder.wallet_address, "")).in_(wallets)
            )
        )
        await session.execute(
            delete(LiveTradingPosition).where(
                func.lower(func.coalesce(LiveTradingPosition.wallet_address, "")).in_(wallets)
            )
        )

    for row in runtime_rows:
        session.add(
            LiveTradingRuntimeState(
                id=str(row["id"]),
                wallet_address=str(row["wallet_address"]),
                total_trades=int(row["total_trades"]),
                winning_trades=int(row["winning_trades"]),
                losing_trades=int(row["losing_trades"]),
                total_volume=float(row["total_volume"]),
                total_pnl=float(row["total_pnl"]),
                daily_volume=float(row["daily_volume"]),
                daily_pnl=float(row["daily_pnl"]),
                open_positions=int(row["open_positions"]),
                last_trade_at=row["last_trade_at"],
                daily_volume_reset_at=row["daily_volume_reset_at"],
                market_positions_json=_coerce_dict(row["market_positions_json"]),
                balance_signature_type=row["balance_signature_type"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
        )

    for row in order_rows:
        session.add(
            LiveTradingOrder(
                id=str(row["id"]),
                wallet_address=str(row["wallet_address"]),
                clob_order_id=row["clob_order_id"],
                token_id=str(row["token_id"]),
                side=str(row["side"]),
                price=float(row["price"]),
                size=float(row["size"]),
                order_type=str(row["order_type"]),
                status=str(row["status"]),
                filled_size=float(row["filled_size"]),
                average_fill_price=float(row["average_fill_price"]),
                market_question=row["market_question"],
                opportunity_id=row["opportunity_id"],
                error_message=row["error_message"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
        )

    for row in position_rows:
        session.add(
            LiveTradingPosition(
                id=str(row["id"]),
                wallet_address=str(row["wallet_address"]),
                token_id=str(row["token_id"]),
                market_id=str(row["market_id"]),
                market_question=row["market_question"],
                outcome=row["outcome"],
                size=float(row["size"]),
                average_cost=float(row["average_cost"]),
                current_price=float(row["current_price"]),
                unrealized_pnl=float(row["unrealized_pnl"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
        )

    return {
        "live_wallet_runtime_states_imported": len(runtime_rows),
        "live_wallet_orders_imported": len(order_rows),
        "live_wallet_positions_imported": len(position_rows),
        "live_wallets_imported": len(wallet_addresses),
    }


async def _import_trader_trade_state(
    session,
    trader_ids_by_name: dict[str, str],
    trade_state_payload: dict[str, Any],
) -> dict[str, Any]:
    from services.trader_orchestrator_state import sync_trader_position_inventory

    live_wallet_result = await _import_live_wallet_trade_state(
        session,
        _coerce_dict(trade_state_payload.get("live_wallet_state")),
    )

    if not trader_ids_by_name:
        return {
            "orders_imported": 0,
            "positions_synced": 0,
            "open_positions": 0,
            "live_wallet_runtime_states_imported": int(
                live_wallet_result.get("live_wallet_runtime_states_imported") or 0
            ),
            "live_wallet_orders_imported": int(live_wallet_result.get("live_wallet_orders_imported") or 0),
            "live_wallet_positions_imported": int(live_wallet_result.get("live_wallet_positions_imported") or 0),
            "live_wallets_imported": int(live_wallet_result.get("live_wallets_imported") or 0),
        }

    trader_ids = [trader_id for trader_id in trader_ids_by_name.values() if str(trader_id or "").strip()]
    if not trader_ids:
        return {
            "orders_imported": 0,
            "positions_synced": 0,
            "open_positions": 0,
            "live_wallet_runtime_states_imported": int(
                live_wallet_result.get("live_wallet_runtime_states_imported") or 0
            ),
            "live_wallet_orders_imported": int(live_wallet_result.get("live_wallet_orders_imported") or 0),
            "live_wallet_positions_imported": int(live_wallet_result.get("live_wallet_positions_imported") or 0),
            "live_wallets_imported": int(live_wallet_result.get("live_wallets_imported") or 0),
        }

    await session.execute(delete(TraderPosition).where(TraderPosition.trader_id.in_(trader_ids)))
    await session.execute(delete(TraderOrder).where(TraderOrder.trader_id.in_(trader_ids)))

    normalized_orders = [
        _normalize_trader_order_payload(item)
        for item in _coerce_dict_list(trade_state_payload.get("orders"))
    ]

    orders_imported = 0
    for normalized in normalized_orders:
        trader_name_key = str(normalized["trader_name"]).strip().lower()
        trader_id = trader_ids_by_name.get(trader_name_key)
        if not trader_id:
            raise ValueError(
                f"Trader trade state references unknown trader '{normalized['trader_name']}'."
            )

        session.add(
            TraderOrder(
                id=uuid.uuid4().hex,
                trader_id=trader_id,
                signal_id=None,
                decision_id=None,
                source=normalized["source"],
                strategy_key=normalized["strategy_key"],
                strategy_version=normalized["strategy_version"],
                market_id=normalized["market_id"],
                market_question=normalized["market_question"],
                direction=normalized["direction"],
                event_id=normalized["event_id"],
                trace_id=normalized["trace_id"],
                mode=normalized["mode"],
                status=normalized["status"],
                notional_usd=normalized["notional_usd"],
                entry_price=normalized["entry_price"],
                effective_price=normalized["effective_price"],
                edge_percent=normalized["edge_percent"],
                confidence=normalized["confidence"],
                actual_profit=normalized["actual_profit"],
                reason=normalized["reason"],
                payload_json=normalized["payload"],
                error_message=normalized["error_message"],
                created_at=normalized["created_at"],
                executed_at=normalized["executed_at"],
                updated_at=normalized["updated_at"],
            )
        )
        orders_imported += 1

    await session.flush()

    positions_synced = 0
    open_positions = 0
    for trader_id in trader_ids:
        sync_result = await sync_trader_position_inventory(
            session,
            trader_id=trader_id,
            commit=False,
        )
        positions_synced += 1
        open_positions += int(sync_result.get("open_positions") or 0)

    return {
        "orders_imported": orders_imported,
        "positions_synced": positions_synced,
        "open_positions": open_positions,
        "live_wallet_runtime_states_imported": int(live_wallet_result.get("live_wallet_runtime_states_imported") or 0),
        "live_wallet_orders_imported": int(live_wallet_result.get("live_wallet_orders_imported") or 0),
        "live_wallet_positions_imported": int(live_wallet_result.get("live_wallet_positions_imported") or 0),
        "live_wallets_imported": int(live_wallet_result.get("live_wallets_imported") or 0),
    }


async def _import_strategies(session, payload: Any) -> dict[str, Any]:
    normalized_payloads = [_normalize_strategy_payload(item) for item in _coerce_dict_list(payload)]
    seen: set[str] = set()
    for row in normalized_payloads:
        slug = row["slug"]
        if slug in seen:
            raise ValueError(f"Duplicate strategy slug '{slug}' in import bundle.")
        seen.add(slug)

    now = utcnow()
    existing_rows = (
        (await session.execute(select(Strategy).order_by(Strategy.slug.asc()))).scalars().all()
    )
    existing_by_slug = {str(row.slug or "").strip().lower(): row for row in existing_rows}

    created = 0
    updated = 0
    versions_imported = 0
    imported_slugs: list[str] = []

    strategy_rows_by_slug: dict[str, Strategy] = {}
    version_payloads_by_slug: dict[str, list[dict[str, Any]]] = {}

    for normalized in normalized_payloads:
        slug = normalized["slug"]
        row = existing_by_slug.get(slug)
        if row is None:
            row = Strategy(
                id=uuid.uuid4().hex,
                slug=slug,
                created_at=now,
            )
            session.add(row)
            existing_by_slug[slug] = row
            created += 1
        else:
            updated += 1

        row.source_key = normalized["source_key"]
        row.name = normalized["name"]
        row.description = normalized["description"]
        row.source_code = normalized["source_code"]
        row.class_name = normalized["class_name"]
        row.is_system = bool(normalized["is_system"])
        row.enabled = bool(normalized["enabled"])
        row.status = "unloaded"
        row.error_message = None
        row.config = normalized["config"]
        row.config_schema = normalized["config_schema"]
        row.aliases = normalized["aliases"]
        row.version = int(normalized["version"])
        row.sort_order = int(normalized["sort_order"])
        row.updated_at = now

        strategy_rows_by_slug[slug] = row
        version_payloads_by_slug[slug] = normalized["versions"]
        imported_slugs.append(slug)

    await session.flush()

    for slug, row in strategy_rows_by_slug.items():
        await session.execute(
            delete(StrategyVersion).where(StrategyVersion.strategy_id == str(row.id))
        )
        provided_versions = version_payloads_by_slug.get(slug, [])
        parsed_versions: dict[int, dict[str, Any]] = {}
        for provided_version in provided_versions:
            try:
                parsed_number = normalize_strategy_version(provided_version.get("version"))
            except ValueError:
                continue
            if parsed_number is None or parsed_number <= 0:
                continue
            parsed_versions[int(parsed_number)] = dict(provided_version)

        if not parsed_versions:
            parsed_versions[int(row.version or 1)] = {
                "version": int(row.version or 1),
                "name": row.name,
                "description": row.description,
                "source_code": row.source_code,
                "class_name": row.class_name,
                "config": _coerce_dict(row.config),
                "config_schema": _coerce_dict(row.config_schema),
                "aliases": list(row.aliases or []),
                "enabled": bool(row.enabled),
                "is_system": bool(row.is_system),
                "sort_order": int(row.sort_order or 0),
            }

        if int(row.version or 1) not in parsed_versions:
            parsed_versions[int(row.version or 1)] = {
                "version": int(row.version or 1),
                "name": row.name,
                "description": row.description,
                "source_code": row.source_code,
                "class_name": row.class_name,
                "config": _coerce_dict(row.config),
                "config_schema": _coerce_dict(row.config_schema),
                "aliases": list(row.aliases or []),
                "enabled": bool(row.enabled),
                "is_system": bool(row.is_system),
                "sort_order": int(row.sort_order or 0),
            }

        latest_version = int(row.version or 1)
        if latest_version not in parsed_versions:
            latest_version = max(parsed_versions.keys())
            row.version = latest_version

        for version_number in sorted(parsed_versions.keys()):
            version_payload = parsed_versions[version_number]
            snapshot = StrategyVersion(
                id=uuid.uuid4().hex,
                strategy_id=str(row.id),
                strategy_slug=slug,
                source_key=str(row.source_key or "scanner").strip().lower() or "scanner",
                version=int(version_number),
                is_latest=bool(version_number == latest_version),
                name=_coerce_string(version_payload.get("name")) or row.name,
                description=_coerce_string(version_payload.get("description")),
                source_code=str(version_payload.get("source_code") or row.source_code or ""),
                class_name=_coerce_string(version_payload.get("class_name")) or row.class_name,
                config=_coerce_dict(version_payload.get("config")) or _coerce_dict(row.config),
                config_schema=_coerce_dict(version_payload.get("config_schema")) or _coerce_dict(row.config_schema),
                aliases=version_payload.get("aliases")
                if isinstance(version_payload.get("aliases"), list)
                else list(row.aliases or []),
                enabled=bool(version_payload.get("enabled", row.enabled)),
                is_system=bool(version_payload.get("is_system", row.is_system)),
                sort_order=int(version_payload.get("sort_order", row.sort_order or 0) or 0),
                parent_version=normalize_strategy_version(version_payload.get("parent_version"))
                if version_payload.get("parent_version") is not None
                else None,
                created_by=_coerce_string(version_payload.get("created_by")),
                reason=_coerce_string(version_payload.get("reason")) or "settings_import",
                created_at=now,
            )
            session.add(snapshot)
            versions_imported += 1

    return {
        "created": created,
        "updated": updated,
        "versions_imported": versions_imported,
        "imported_slugs": imported_slugs,
    }


async def _import_data_sources(session, payload: Any) -> dict[str, Any]:
    normalized_payloads = [_normalize_data_source_payload(item) for item in _coerce_dict_list(payload)]
    seen: set[str] = set()
    for row in normalized_payloads:
        slug = row["slug"]
        if slug in seen:
            raise ValueError(f"Duplicate data source slug '{slug}' in import bundle.")
        seen.add(slug)

    now = utcnow()
    existing_rows = (
        (await session.execute(select(DataSource).order_by(DataSource.slug.asc()))).scalars().all()
    )
    existing_by_slug = {str(row.slug or "").strip().lower(): row for row in existing_rows}

    created = 0
    updated = 0
    imported_slugs: list[str] = []

    for normalized in normalized_payloads:
        slug = normalized["slug"]
        row = existing_by_slug.get(slug)
        if row is None:
            row = DataSource(
                id=uuid.uuid4().hex,
                slug=slug,
                created_at=now,
            )
            session.add(row)
            existing_by_slug[slug] = row
            created += 1
        else:
            updated += 1

        row.source_key = normalized["source_key"]
        row.source_kind = normalized["source_kind"]
        row.name = normalized["name"]
        row.description = normalized["description"]
        row.source_code = normalized["source_code"]
        row.class_name = normalized["class_name"]
        row.is_system = bool(normalized["is_system"])
        row.enabled = bool(normalized["enabled"])
        row.status = "unloaded"
        row.error_message = None
        row.retention = normalized["retention"]
        row.config = normalized["config"]
        row.config_schema = normalized["config_schema"]
        row.version = int(normalized["version"])
        row.sort_order = int(normalized["sort_order"])
        row.updated_at = now
        imported_slugs.append(slug)

    return {
        "created": created,
        "updated": updated,
        "imported_slugs": imported_slugs,
    }


async def _apply_orchestrator_control_import(session, payload: dict[str, Any]) -> bool:
    control_payload = _coerce_dict(payload)
    if not control_payload:
        return False

    row = await session.get(TraderOrchestratorControl, "default")
    if row is None:
        row = TraderOrchestratorControl(id="default")
        session.add(row)

    interval_raw = control_payload.get("run_interval_seconds")
    try:
        interval_seconds = int(interval_raw if interval_raw is not None else (row.run_interval_seconds or 5))
    except (TypeError, ValueError):
        interval_seconds = int(row.run_interval_seconds or 5)
    interval_seconds = max(1, min(3600, interval_seconds))

    row.is_enabled = bool(control_payload.get("is_enabled", row.is_enabled))
    row.is_paused = bool(control_payload.get("is_paused", row.is_paused))
    row.mode = str(control_payload.get("mode") or row.mode or "shadow").strip().lower() or "shadow"
    row.run_interval_seconds = interval_seconds
    row.kill_switch = bool(control_payload.get("kill_switch", row.kill_switch))
    row.settings_json = _coerce_dict(control_payload.get("settings_json"))
    row.updated_at = utcnow()
    return True


async def _import_traders(session, payload: Any) -> dict[str, Any]:
    from services.trader_orchestrator_state import create_trader, list_traders, update_trader

    payload_dict = _coerce_dict(payload)
    if not payload_dict:
        raise ValueError("bot_traders payload must be an object.")

    traders_payload = payload_dict.get("traders")
    orchestrator_payload = _coerce_dict(payload_dict.get("orchestrator"))
    include_trade_state = "trade_state" in payload_dict
    trade_state_payload = _coerce_dict(payload_dict.get("trade_state"))

    normalized_payloads = [_normalize_trader_payload(item) for item in _coerce_dict_list(traders_payload)]
    seen: set[str] = set()
    for row in normalized_payloads:
        key = str(row["name"]).strip().lower()
        if key in seen:
            raise ValueError(f"Duplicate trader name '{row['name']}' in import bundle.")
        seen.add(key)

    existing = await list_traders(session)
    existing_by_name = {
        str(trader.get("name") or "").strip().lower(): trader
        for trader in existing
        if str(trader.get("name") or "").strip()
    }

    created = 0
    updated = 0
    imported_names: list[str] = []

    for normalized in normalized_payloads:
        name_key = str(normalized["name"]).strip().lower()
        existing_trader = existing_by_name.get(name_key)
        if existing_trader is None:
            created_trader = await create_trader(session, normalized)
            existing_by_name[str(created_trader.get("name") or "").strip().lower()] = created_trader
            created += 1
        else:
            updated_trader = await update_trader(session, str(existing_trader["id"]), normalized)
            if updated_trader is None:
                raise ValueError(f"Unable to update trader '{normalized['name']}'.")
            existing_by_name[str(updated_trader.get("name") or "").strip().lower()] = updated_trader
            updated += 1
        imported_names.append(normalized["name"])

    imported_trader_ids_by_name: dict[str, str] = {}
    for imported_name in imported_names:
        trader = existing_by_name.get(str(imported_name).strip().lower())
        trader_id = _coerce_string((trader or {}).get("id") if isinstance(trader, dict) else None)
        if not trader_id:
            raise ValueError(f"Trader '{imported_name}' is missing an id after import.")
        imported_trader_ids_by_name[str(imported_name).strip().lower()] = trader_id

    trade_state_result = {
        "orders_imported": 0,
        "positions_synced": 0,
        "open_positions": 0,
        "live_wallet_runtime_states_imported": 0,
        "live_wallet_orders_imported": 0,
        "live_wallet_positions_imported": 0,
        "live_wallets_imported": 0,
    }
    if include_trade_state:
        trade_state_result = await _import_trader_trade_state(
            session,
            imported_trader_ids_by_name,
            trade_state_payload,
        )

    orchestrator_updated = await _apply_orchestrator_control_import(session, orchestrator_payload)

    return {
        "created": created,
        "updated": updated,
        "orchestrator_updated": 1 if orchestrator_updated else 0,
        "orders_imported": int(trade_state_result.get("orders_imported") or 0),
        "positions_synced": int(trade_state_result.get("positions_synced") or 0),
        "open_positions": int(trade_state_result.get("open_positions") or 0),
        "live_wallet_runtime_states_imported": int(
            trade_state_result.get("live_wallet_runtime_states_imported") or 0
        ),
        "live_wallet_orders_imported": int(trade_state_result.get("live_wallet_orders_imported") or 0),
        "live_wallet_positions_imported": int(trade_state_result.get("live_wallet_positions_imported") or 0),
        "live_wallets_imported": int(trade_state_result.get("live_wallets_imported") or 0),
        "imported_names": imported_names,
    }


async def get_or_create_settings() -> AppSettings:
    """Get existing settings or create default"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(AppSettings).where(AppSettings.id == "default"))
        settings = result.scalar_one_or_none()

        if not settings:
            settings = AppSettings(id="default")
            session.add(settings)
            await session.commit()
            await session.refresh(settings)

        return settings


# ==================== ENDPOINTS ====================


@router.get("", response_model=AllSettings)
async def get_settings():
    """
    Get all application settings.

    Sensitive fields (API keys, secrets) are masked in the response.
    """
    try:
        settings = await get_or_create_settings()

        return AllSettings(
            polymarket=PolymarketSettings(**polymarket_payload(settings)),
            kalshi=KalshiSettings(**kalshi_payload(settings)),
            llm=LLMSettings(**llm_payload(settings)),
            notifications=NotificationSettings(**notifications_payload(settings)),
            scanner=ScannerSettingsModel(**scanner_payload(settings)),
            live_execution=LiveExecutionSettings(**live_execution_payload(settings)),
            maintenance=MaintenanceSettings(**maintenance_payload(settings)),
            discovery=DiscoverySettings(**discovery_payload(settings)),
            trading_proxy=TradingProxySettings(**trading_proxy_payload(settings)),
            ui_lock=UILockSettings(**ui_lock_payload(settings)),
            network=NetworkSettings(**network_payload(settings)),
            events=EventsSettings(**events_payload(settings)),
            search_filters=SearchFilterSettings(**search_filters_payload(settings)),
            updated_at=settings.updated_at.isoformat() if settings.updated_at else None,
        )
    except Exception as e:
        logger.error("Failed to get settings", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.put("")
async def update_settings(request: UpdateSettingsRequest):
    """
    Update application settings.

    Only provided fields will be updated. Omitted fields remain unchanged.
    Pass null/empty string to clear a field.
    """
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(AppSettings).where(AppSettings.id == "default"))
            settings = result.scalar_one_or_none()

            if not settings:
                settings = AppSettings(id="default")
                session.add(settings)

            try:
                update_flags = apply_update_request(settings, request)
            except ValueError as exc:
                await session.rollback()
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            await session.commit()
            updated_at = settings.updated_at.isoformat()
            needs_llm_reinit = update_flags["needs_llm_reinit"]
            needs_proxy_reinit = update_flags["needs_proxy_reinit"]
            needs_filter_reload = update_flags["needs_filter_reload"]
            needs_events_reload = update_flags["needs_events_reload"]
            needs_ui_lock_reload = bool(update_flags.get("needs_ui_lock_reload"))
            reset_ui_lock_sessions = bool(update_flags.get("reset_ui_lock_sessions"))

        # Re-initialize LLM manager OUTSIDE the DB session so initialize()
        # reads newly committed settings from a fresh transaction.
        if needs_llm_reinit:
            try:
                from services.ai import get_llm_manager

                manager = get_llm_manager()
                await manager.initialize()
                logger.info(
                    f"LLM manager re-initialized, active model: {manager._default_model}",
                )
            except RuntimeError:
                pass  # AI module not loaded yet
            except Exception as reinit_err:
                logger.error(
                    f"Failed to re-initialize LLM manager after settings update: {reinit_err}",
                )

        # Re-initialize trading proxy if settings changed
        if needs_proxy_reinit:
            try:
                from services.trading_proxy import reload_proxy_settings

                await reload_proxy_settings()
                logger.info("Trading proxy re-initialized after settings update")
            except Exception as reinit_err:
                logger.error(
                    f"Failed to re-initialize trading proxy: {reinit_err}",
                )

        # Reload DB-backed runtime settings into the singleton when any
        # settings-backed runtime filters changed.
        if needs_filter_reload or needs_events_reload:
            try:
                from config import apply_runtime_settings_overrides

                await apply_runtime_settings_overrides()
                logger.info("Runtime settings overrides reloaded after settings update")
            except Exception as reinit_err:
                logger.error(
                    f"Failed to reload runtime settings overrides: {reinit_err}",
                )

        if needs_ui_lock_reload:
            try:
                from services.ui_lock import ui_lock_service

                ui_lock_service.mark_settings_dirty()
                if reset_ui_lock_sessions:
                    await ui_lock_service.invalidate_all_sessions()
            except Exception as reinit_err:
                logger.error(
                    f"Failed to refresh UI lock settings after update: {reinit_err}",
                )

        logger.info("Settings updated successfully")

        return {
            "status": "success",
            "message": "Settings updated successfully",
            "updated_at": updated_at,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update settings", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ==================== SECTION-SPECIFIC ENDPOINTS ====================


@router.get("/polymarket")
async def get_polymarket_settings():
    """Get Polymarket settings only"""
    settings = await get_or_create_settings()
    return PolymarketSettings(**polymarket_payload(settings))


@router.put("/polymarket")
async def update_polymarket_settings(request: PolymarketSettings):
    """Update Polymarket settings only"""
    return await update_settings(UpdateSettingsRequest(polymarket=request))


@router.get("/kalshi")
async def get_kalshi_settings():
    """Get Kalshi settings only"""
    settings = await get_or_create_settings()
    return KalshiSettings(**kalshi_payload(settings))


@router.put("/kalshi")
async def update_kalshi_settings(request: KalshiSettings):
    """Update Kalshi settings only"""
    return await update_settings(UpdateSettingsRequest(kalshi=request))


@router.get("/llm")
async def get_llm_settings():
    """Get LLM settings only"""
    settings = await get_or_create_settings()
    return LLMSettings(**llm_payload(settings))


@router.put("/llm")
async def update_llm_settings(request: LLMSettings):
    """Update LLM settings only"""
    return await update_settings(UpdateSettingsRequest(llm=request))


@router.get("/notifications")
async def get_notification_settings():
    """Get notification settings only"""
    settings = await get_or_create_settings()
    return NotificationSettings(**notifications_payload(settings))


@router.put("/notifications")
async def update_notification_settings(request: NotificationSettings):
    """Update notification settings only"""
    return await update_settings(UpdateSettingsRequest(notifications=request))


@router.get("/scanner")
async def get_scanner_settings():
    """Get scanner settings only"""
    settings = await get_or_create_settings()
    return ScannerSettingsModel(**scanner_payload(settings))


@router.put("/scanner")
async def update_scanner_settings(request: ScannerSettingsModel):
    """Update scanner settings only"""
    return await update_settings(UpdateSettingsRequest(scanner=request))


@router.get("/live-execution")
async def get_live_execution_settings():
    """Get live execution settings only"""
    settings = await get_or_create_settings()
    return LiveExecutionSettings(**live_execution_payload(settings))


@router.put("/live-execution")
async def update_live_execution_settings(request: LiveExecutionSettings):
    """Update live execution settings only"""
    return await update_settings(UpdateSettingsRequest(live_execution=request))


@router.get("/maintenance")
async def get_maintenance_settings():
    """Get maintenance settings only"""
    settings = await get_or_create_settings()
    return MaintenanceSettings(**maintenance_payload(settings))


@router.put("/maintenance")
async def update_maintenance_settings(request: MaintenanceSettings):
    """Update maintenance settings only"""
    return await update_settings(UpdateSettingsRequest(maintenance=request))


@router.get("/discovery")
async def get_discovery_settings():
    """Get wallet discovery settings"""
    settings = await get_or_create_settings()
    return DiscoverySettings(**discovery_payload(settings))


@router.put("/discovery")
async def update_discovery_settings(request: DiscoverySettings):
    """Update wallet discovery settings"""
    return await update_settings(UpdateSettingsRequest(discovery=request))


@router.get("/trading-proxy")
async def get_trading_proxy_settings():
    """Get trading VPN/proxy settings only"""
    settings = await get_or_create_settings()
    return TradingProxySettings(**trading_proxy_payload(settings))


@router.put("/trading-proxy")
async def update_trading_proxy_settings(request: TradingProxySettings):
    """Update trading VPN/proxy settings only"""
    return await update_settings(UpdateSettingsRequest(trading_proxy=request))


@router.get("/ui-lock")
async def get_ui_lock_settings():
    """Get local UI lock settings only"""
    settings = await get_or_create_settings()
    return UILockSettings(**ui_lock_payload(settings))


@router.put("/ui-lock")
async def update_ui_lock_settings(request: UILockSettings):
    """Update local UI lock settings only"""
    return await update_settings(UpdateSettingsRequest(ui_lock=request))


@router.get("/search-filters")
async def get_search_filter_settings():
    """Get search filter settings only"""
    settings = await get_or_create_settings()
    return SearchFilterSettings(**search_filters_payload(settings))


@router.put("/search-filters")
async def update_search_filter_settings(request: SearchFilterSettings):
    """Update search filter settings only"""
    return await update_settings(UpdateSettingsRequest(search_filters=request))


@router.post("/export")
async def export_settings_bundle(request: SettingsExportRequest):
    """Export selected settings/trader/strategy/source configuration as a JSON bundle."""
    try:
        categories = _resolve_export_categories(request.include_categories)
        bundle, counts = await _export_transfer_bundle(categories)
        return {
            "status": "success",
            "bundle": bundle,
            "counts": counts,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to export settings bundle", exc_info=exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/import")
async def import_settings_bundle(request: SettingsImportRequest):
    """Import selected settings/trader/strategy/source configuration from a JSON bundle."""
    bundle = request.bundle if isinstance(request.bundle, dict) else {}
    if not bundle:
        raise HTTPException(status_code=400, detail="Import bundle is required.")

    schema_version = bundle.get("schema_version")
    if schema_version not in (None, 1):
        raise HTTPException(status_code=400, detail=f"Unsupported settings bundle schema_version '{schema_version}'.")

    categories = _resolve_import_categories(request.include_categories, bundle)
    if not categories:
        raise HTTPException(status_code=400, detail="At least one import category is required.")

    results: dict[str, Any] = {}
    needs_llm_reinit = False
    needs_proxy_reinit = False

    try:
        async with AsyncSessionLocal() as session:
            telegram_payload = bundle.get(SettingsTransferCategory.TELEGRAM_CONFIGURATION.value)
            settings_row: AppSettings | None = None
            if any(
                category in categories
                for category in (
                    SettingsTransferCategory.MARKET_CREDENTIALS.value,
                    SettingsTransferCategory.VPN_CONFIGURATION.value,
                    SettingsTransferCategory.LLM_CONFIGURATION.value,
                )
            ) or (
                SettingsTransferCategory.TELEGRAM_CONFIGURATION.value in categories
                and telegram_payload is not None
            ):
                settings_row = await _get_or_create_settings_row(session)

            if SettingsTransferCategory.STRATEGIES.value in categories:
                strategy_result = await _import_strategies(
                    session,
                    bundle.get(SettingsTransferCategory.STRATEGIES.value),
                )
                results[SettingsTransferCategory.STRATEGIES.value] = {
                    "created": strategy_result["created"],
                    "updated": strategy_result["updated"],
                    "versions_imported": strategy_result["versions_imported"],
                }

            if SettingsTransferCategory.DATA_SOURCES.value in categories:
                source_result = await _import_data_sources(
                    session,
                    bundle.get(SettingsTransferCategory.DATA_SOURCES.value),
                )
                results[SettingsTransferCategory.DATA_SOURCES.value] = {
                    "created": source_result["created"],
                    "updated": source_result["updated"],
                }

            if SettingsTransferCategory.MARKET_CREDENTIALS.value in categories:
                if settings_row is None:
                    settings_row = await _get_or_create_settings_row(session)
                _apply_market_credentials_import(
                    settings_row,
                    _coerce_dict(bundle.get(SettingsTransferCategory.MARKET_CREDENTIALS.value)),
                )
                results[SettingsTransferCategory.MARKET_CREDENTIALS.value] = {"updated": 1}

            if SettingsTransferCategory.VPN_CONFIGURATION.value in categories:
                if settings_row is None:
                    settings_row = await _get_or_create_settings_row(session)
                _apply_vpn_configuration_import(
                    settings_row,
                    _coerce_dict(bundle.get(SettingsTransferCategory.VPN_CONFIGURATION.value)),
                )
                needs_proxy_reinit = True
                results[SettingsTransferCategory.VPN_CONFIGURATION.value] = {"updated": 1}

            if SettingsTransferCategory.LLM_CONFIGURATION.value in categories:
                if settings_row is None:
                    settings_row = await _get_or_create_settings_row(session)
                _apply_llm_configuration_import(
                    settings_row,
                    _coerce_dict(bundle.get(SettingsTransferCategory.LLM_CONFIGURATION.value)),
                )
                needs_llm_reinit = True
                results[SettingsTransferCategory.LLM_CONFIGURATION.value] = {"updated": 1}

            if SettingsTransferCategory.TELEGRAM_CONFIGURATION.value in categories:
                if telegram_payload is None:
                    results[SettingsTransferCategory.TELEGRAM_CONFIGURATION.value] = {"updated": 0}
                else:
                    if settings_row is None:
                        settings_row = await _get_or_create_settings_row(session)
                    _apply_telegram_configuration_import(
                        settings_row,
                        _coerce_dict(telegram_payload),
                    )
                    results[SettingsTransferCategory.TELEGRAM_CONFIGURATION.value] = {"updated": 1}

            if SettingsTransferCategory.BOT_TRADERS.value in categories:
                trader_result = await _import_traders(
                    session,
                    bundle.get(SettingsTransferCategory.BOT_TRADERS.value),
                )
                results[SettingsTransferCategory.BOT_TRADERS.value] = {
                    "created": trader_result["created"],
                    "updated": trader_result["updated"],
                    "orchestrator_updated": trader_result["orchestrator_updated"],
                    "orders_imported": int(trader_result.get("orders_imported") or 0),
                    "positions_synced": int(trader_result.get("positions_synced") or 0),
                    "open_positions": int(trader_result.get("open_positions") or 0),
                    "live_wallet_runtime_states_imported": int(
                        trader_result.get("live_wallet_runtime_states_imported") or 0
                    ),
                    "live_wallet_orders_imported": int(trader_result.get("live_wallet_orders_imported") or 0),
                    "live_wallet_positions_imported": int(
                        trader_result.get("live_wallet_positions_imported") or 0
                    ),
                    "live_wallets_imported": int(trader_result.get("live_wallets_imported") or 0),
                }

            if settings_row is not None:
                settings_row.updated_at = utcnow()

            await session.commit()

            if SettingsTransferCategory.STRATEGIES.value in categories:
                from services.strategy_loader import strategy_loader

                refresh_result = await strategy_loader.refresh_all_from_db(session=session)
                results["strategies_runtime"] = {
                    "loaded": len(refresh_result.get("loaded", [])),
                    "errors": len((refresh_result.get("errors") or {}).keys()),
                }

            if SettingsTransferCategory.DATA_SOURCES.value in categories:
                from services.data_source_loader import data_source_loader

                refresh_result = await data_source_loader.refresh_all_from_db(session=session)
                results["data_sources_runtime"] = {
                    "loaded": len(refresh_result.get("loaded", [])),
                    "errors": len((refresh_result.get("errors") or {}).keys()),
                }

        if needs_llm_reinit:
            try:
                from services.ai import get_llm_manager

                manager = get_llm_manager()
                await manager.initialize()
            except RuntimeError:
                pass
            except Exception as reinit_exc:
                logger.error("Failed to re-initialize LLM manager after settings import", exc_info=reinit_exc)

        if needs_proxy_reinit:
            try:
                from services.trading_proxy import reload_proxy_settings

                await reload_proxy_settings()
            except Exception as reinit_exc:
                logger.error("Failed to re-initialize trading proxy after settings import", exc_info=reinit_exc)

        return {
            "status": "success",
            "imported_categories": categories,
            "results": results,
            "imported_at": utcnow().isoformat(),
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Failed to import settings bundle", exc_info=exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ==================== VALIDATION ENDPOINTS ====================


@router.post("/test/polymarket")
async def test_polymarket_connection():
    """Test Polymarket API connection with stored credentials"""
    try:
        app_settings = await get_or_create_settings()
        private_key = decrypt_secret(app_settings.polymarket_private_key)
        api_key = decrypt_secret(app_settings.polymarket_api_key)
        api_secret = decrypt_secret(app_settings.polymarket_api_secret)
        api_passphrase = decrypt_secret(app_settings.polymarket_api_passphrase)

        missing: list[str] = []
        if not private_key:
            missing.append("private_key")
        if not api_key:
            missing.append("api_key")
        if not api_secret:
            missing.append("api_secret")
        if not api_passphrase:
            missing.append("api_passphrase")
        if missing:
            return {
                "status": "error",
                "message": f"Missing Polymarket credentials: {', '.join(missing)}",
            }

        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds

        client = ClobClient(
            host=runtime_settings.CLOB_API_URL,
            key=private_key,
            chain_id=runtime_settings.CHAIN_ID,
            creds=ApiCreds(
                api_key=api_key,
                api_secret=api_secret,
                api_passphrase=api_passphrase,
            ),
        )
        response = await asyncio.to_thread(client.get_orders)

        open_orders_count: Optional[int] = None
        if isinstance(response, list):
            open_orders_count = len(response)
        elif isinstance(response, dict):
            orders = response.get("orders")
            if not isinstance(orders, list):
                orders = response.get("data")
            if isinstance(orders, list):
                open_orders_count = len(orders)

        wallet_address: Optional[str] = None
        try:
            from eth_account import Account

            wallet_address = Account.from_key(private_key).address
        except Exception:
            wallet_address = None
        return {
            "status": "success",
            "message": "Polymarket API authentication succeeded",
            "wallet_address": wallet_address,
            "open_orders_count": open_orders_count,
        }
    except Exception as e:
        return {"status": "error", "message": f"Polymarket API authentication failed: {e}"}


@router.post("/test/kalshi")
async def test_kalshi_connection():
    """Test Kalshi API connection with stored credentials"""
    client = None
    try:
        app_settings = await get_or_create_settings()
        email = str(app_settings.kalshi_email or "").strip() or None
        password = decrypt_secret(app_settings.kalshi_password)
        api_key = decrypt_secret(app_settings.kalshi_api_key)

        if not api_key and not (email and password):
            return {
                "status": "error",
                "message": "Missing Kalshi credentials: provide api_key or email/password",
            }

        from services.kalshi_client import KalshiClient

        client = KalshiClient()
        auth_method = "api_key" if api_key else "email_password"
        authenticated = await client.initialize_auth(
            email=email,
            password=password,
            api_key=api_key,
        )
        if not authenticated:
            return {
                "status": "error",
                "message": "Kalshi API authentication failed: invalid credentials",
            }

        balance = await client.get_balance()
        if balance is None:
            return {
                "status": "error",
                "message": "Kalshi API authentication failed: unable to fetch account balance",
            }

        return {
            "status": "success",
            "message": "Kalshi API authentication succeeded",
            "auth_method": auth_method,
            "balance": balance.get("balance"),
            "available": balance.get("available"),
            "currency": balance.get("currency"),
        }
    except Exception as e:
        return {"status": "error", "message": f"Kalshi API authentication failed: {e}"}
    finally:
        if client is not None:
            try:
                await client.close()
            except Exception:
                pass


@router.post("/test/telegram")
async def test_telegram_connection():
    """Test Telegram bot connection with stored credentials"""
    try:
        settings = await get_or_create_settings()

        if not decrypt_secret(settings.telegram_bot_token) or not settings.telegram_chat_id:
            return {
                "status": "error",
                "message": "Telegram bot token or chat ID not configured",
            }

        from services.notifier import notifier

        sent = await notifier.send_test_message()
        if sent:
            return {
                "status": "success",
                "message": "Test message sent to Telegram successfully",
            }
        return {
            "status": "error",
            "message": "Telegram API request failed; check bot token/chat ID and bot permissions",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/test/trading-proxy")
async def test_trading_proxy():
    """Test trading VPN proxy connectivity and verify IP differs from direct connection"""
    try:
        from services.trading_proxy import verify_vpn_active

        status = await verify_vpn_active()

        if status.get("error"):
            return {"status": "error", "message": status["error"], **status}

        if not status.get("proxy_reachable"):
            return {
                "status": "error",
                "message": f"Proxy unreachable: {status.get('proxy_ip_error', 'connection failed')}",
                **status,
            }

        proxy_ip = status.get("proxy_ip")
        enabled_note = "" if status.get("proxy_enabled") else " (proxy toggle is off — enable it to route trades)"

        if status.get("vpn_active"):
            return {
                "status": "success",
                "message": f"VPN active — trading through {proxy_ip}{enabled_note}",
                **status,
            }
        else:
            return {
                "status": "warning",
                "message": "Proxy reachable but IP matches direct connection — VPN may not be active",
                **status,
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/test/llm")
async def test_llm_connection(provider: Optional[str] = None):
    """Test connectivity to an LLM backend by refreshing models for a provider."""
    try:
        provider_name = (provider or "").strip().lower()

        manager_provider = provider_name or None
        if provider_name in {"", "none"}:
            manager_provider = None
        elif provider_name not in {"openai", "anthropic", "google", "xai", "deepseek", "ollama", "lmstudio"}:
            return {
                "status": "error",
                "message": f"Unsupported LLM provider '{provider_name}'.",
            }

        from services.ai import get_llm_manager

        manager = get_llm_manager()
        await manager.initialize()
        if manager_provider is None:
            settings = await get_or_create_settings()
            provider_name = (settings.llm_provider or "none").strip().lower()
            if provider_name in {"", "none"}:
                if (settings.ollama_base_url or "").strip():
                    provider_name = "ollama"
                elif (settings.lmstudio_base_url or "").strip():
                    provider_name = "lmstudio"

            if provider_name in {"", "none"}:
                return {
                    "status": "warning",
                    "message": "No LLM provider configured. Set a provider and save settings first.",
                }
            models = await manager.fetch_and_cache_models(provider_name=provider_name)
            models_for_provider = models.get(provider_name, [])
        else:
            models = await manager.fetch_and_cache_models(provider_name=manager_provider)
            models_for_provider = models.get(provider_name, [])

        if not models_for_provider:
            target = provider_name or "the configured provider"
            return {
                "status": "error",
                "message": f"No models returned from {target}. Check API URL and credentials.",
            }

        return {
            "status": "success",
            "message": f"{provider_name or 'Configured provider'} connectivity OK ({len(models_for_provider)} models).",
            "provider": provider_name or "configured",
            "model_count": len(models_for_provider),
        }
    except RuntimeError as exc:
        return {"status": "error", "message": str(exc)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ==================== MODEL LISTING ENDPOINTS ====================


@router.get("/llm/models")
async def get_available_models(provider: Optional[str] = None):
    """Get cached list of available models for configured providers.

    Returns models from the database cache. Use POST /settings/llm/models/refresh
    to fetch fresh models from provider APIs.
    """
    try:
        from services.ai import get_llm_manager

        manager = get_llm_manager()
        models = await manager.get_cached_models(provider_name=provider)
        return {"models": models}
    except RuntimeError:
        # AI not initialized - return empty
        return {"models": {}}
    except Exception as e:
        logger.error("Failed to get models", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/llm/models/refresh")
async def refresh_models(provider: Optional[str] = None):
    """Fetch fresh models from provider APIs and update the cache.

    Queries each configured provider's API for available models and
    stores the results in the database for quick dropdown population.
    """
    try:
        from services.ai import get_llm_manager

        manager = get_llm_manager()

        # Re-initialize to pick up any newly saved API keys
        await manager.initialize()

        models = await manager.fetch_and_cache_models(provider_name=provider)
        return {
            "status": "success",
            "message": f"Refreshed models for {len(models)} provider(s)",
            "models": models,
        }
    except RuntimeError as e:
        return {"status": "error", "message": str(e), "models": {}}
    except Exception as e:
        logger.error("Failed to refresh models", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
