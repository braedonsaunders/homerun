"""
Settings API Routes

Endpoints for managing application settings.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from sqlalchemy import select
from models.database import AsyncSessionLocal, AppSettings
from utils.logger import get_logger
from utils.secrets import decrypt_secret
from api.settings_helpers import (
    apply_update_request,
    discovery_payload,
    kalshi_payload,
    llm_payload,
    maintenance_payload,
    notifications_payload,
    polymarket_payload,
    scanner_payload,
    search_filters_payload,
    trading_payload,
    trading_proxy_payload,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/settings", tags=["Settings"])


# ==================== REQUEST/RESPONSE MODELS ====================


class PolymarketSettings(BaseModel):
    """Polymarket API credentials"""

    api_key: Optional[str] = Field(default=None, description="Polymarket API key")
    api_secret: Optional[str] = Field(default=None, description="Polymarket API secret")
    api_passphrase: Optional[str] = Field(
        default=None, description="Polymarket API passphrase"
    )
    private_key: Optional[str] = Field(
        default=None, description="Wallet private key for signing"
    )


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
    anthropic_api_key: Optional[str] = Field(
        default=None, description="Anthropic API key"
    )
    google_api_key: Optional[str] = Field(
        default=None, description="Google (Gemini) API key"
    )
    xai_api_key: Optional[str] = Field(default=None, description="xAI (Grok) API key")
    deepseek_api_key: Optional[str] = Field(
        default=None, description="DeepSeek API key"
    )
    ollama_api_key: Optional[str] = Field(
        default=None, description="Ollama API key (optional)"
    )
    ollama_base_url: Optional[str] = Field(
        default=None, description="Ollama base URL (default: http://localhost:11434)"
    )
    lmstudio_api_key: Optional[str] = Field(
        default=None, description="LM Studio API key (optional)"
    )
    lmstudio_base_url: Optional[str] = Field(
        default=None, description="LM Studio base URL (default: http://localhost:1234/v1)"
    )
    model: Optional[str] = Field(
        default=None, description="Model to use (e.g., gpt-4o, gemini-2.0-flash)"
    )
    max_monthly_spend: Optional[float] = Field(
        default=None, ge=0, description="Monthly LLM cost cap in USD"
    )


class NotificationSettings(BaseModel):
    """Notification configuration"""

    enabled: bool = Field(default=False, description="Enable notifications")
    telegram_bot_token: Optional[str] = Field(
        default=None, description="Telegram bot token"
    )
    telegram_chat_id: Optional[str] = Field(
        default=None, description="Telegram chat ID"
    )
    notify_on_opportunity: bool = Field(
        default=True, description="Notify on new opportunities"
    )
    notify_on_trade: bool = Field(default=True, description="Notify on trade execution")
    notify_min_roi: float = Field(
        default=5.0, ge=0, description="Minimum ROI % to notify"
    )
    notify_autotrader_orders: bool = Field(
        default=False, description="Notify immediately when orchestrator creates orders"
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

    scan_interval_seconds: int = Field(
        default=60, ge=10, le=3600, description="Scan interval in seconds"
    )
    min_profit_threshold: float = Field(
        default=2.5, ge=0, description="Minimum profit threshold %"
    )
    max_markets_to_scan: int = Field(
        default=500, ge=10, le=5000, description="Maximum markets to scan"
    )
    min_liquidity: float = Field(
        default=1000.0, ge=0, description="Minimum liquidity in USD"
    )


class TradingSettings(BaseModel):
    """Trading safety configuration"""

    trading_enabled: bool = Field(default=False, description="Enable live trading")
    max_trade_size_usd: float = Field(
        default=100.0, ge=1, description="Maximum single trade size"
    )
    max_daily_trade_volume: float = Field(
        default=1000.0, ge=10, description="Maximum daily trading volume"
    )
    max_open_positions: int = Field(
        default=10, ge=1, le=100, description="Maximum concurrent open positions"
    )
    max_slippage_percent: float = Field(
        default=2.0, ge=0.1, le=10, description="Maximum acceptable slippage %"
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
    maintenance_batch: int = Field(
        default=900, ge=10, le=5000, description="Cleanup insert/chunk batch size"
    )
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


class MaintenanceSettings(BaseModel):
    """Database maintenance configuration"""

    auto_cleanup_enabled: bool = Field(
        default=False, description="Enable automatic database cleanup"
    )
    cleanup_interval_hours: int = Field(
        default=24, ge=1, le=168, description="Cleanup interval in hours"
    )
    cleanup_resolved_trade_days: int = Field(
        default=30, ge=1, le=365, description="Delete resolved trades older than X days"
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
    timeout: float = Field(
        default=30.0, ge=5, le=120, description="Timeout for proxied requests (seconds)"
    )
    require_vpn: bool = Field(
        default=True, description="Block trades if VPN proxy is unreachable"
    )


class SearchFilterSettings(BaseModel):
    """Opportunity search filter thresholds — controls which opportunities are shown"""

    # Hard rejection filters
    min_liquidity_hard: float = Field(
        default=200.0, ge=0, description="Hard reject below this liquidity ($)"
    )
    min_position_size: float = Field(
        default=25.0, ge=0, description="Reject if max position < this ($)"
    )
    min_absolute_profit: float = Field(
        default=5.0, ge=0, description="Reject if net profit on max position < this ($)"
    )
    min_annualized_roi: float = Field(
        default=10.0, ge=0, description="Reject if annualized ROI < this %"
    )
    max_resolution_months: int = Field(
        default=18,
        ge=1,
        le=120,
        description="Reject if resolution > this many months away",
    )
    max_plausible_roi: float = Field(
        default=30.0, ge=1, description="ROI above this % rejected as false positive"
    )
    max_trade_legs: int = Field(
        default=8, ge=2, le=20, description="Maximum legs in a multi-leg trade"
    )

    # NegRisk exhaustivity thresholds
    negrisk_min_total_yes: float = Field(
        default=0.95,
        ge=0.5,
        le=1.0,
        description="Hard reject NegRisk if total YES < this",
    )
    negrisk_warn_total_yes: float = Field(
        default=0.97, ge=0.5, le=1.0, description="Warn if total YES below this"
    )
    negrisk_election_min_total_yes: float = Field(
        default=0.97, ge=0.5, le=1.0, description="Stricter reject for election markets"
    )
    negrisk_max_resolution_spread_days: int = Field(
        default=7,
        ge=0,
        le=365,
        description="Max resolution date spread in NegRisk bundle (days)",
    )

    # Settlement lag
    settlement_lag_max_days_to_resolution: int = Field(
        default=14,
        ge=0,
        le=365,
        description="Only detect settlement lag within this window (days)",
    )
    settlement_lag_near_zero: float = Field(
        default=0.05,
        ge=0.001,
        le=0.5,
        description="Price below this suggests resolved to NO",
    )
    settlement_lag_near_one: float = Field(
        default=0.95,
        ge=0.5,
        le=0.999,
        description="Price above this suggests resolved to YES",
    )
    settlement_lag_min_sum_deviation: float = Field(
        default=0.03,
        ge=0.001,
        le=0.5,
        description="Min deviation from 1.0 to be interesting",
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
    risk_low_liquidity: float = Field(
        default=1000.0, ge=0, description="Liquidity below this adds high risk ($)"
    )
    risk_moderate_liquidity: float = Field(
        default=5000.0, ge=0, description="Liquidity below this adds moderate risk ($)"
    )
    risk_complex_legs: int = Field(
        default=5, ge=2, le=20, description="Legs above this = complex trade risk"
    )
    risk_multiple_legs: int = Field(
        default=3, ge=2, le=20, description="Legs above this = multiple positions risk"
    )

    # BTC/ETH high-frequency
    btc_eth_hf_series_btc_15m: str = Field(
        default="10192", description="Polymarket series ID for BTC 15-min markets"
    )
    btc_eth_hf_series_eth_15m: str = Field(
        default="10191", description="Polymarket series ID for ETH 15-min markets"
    )
    btc_eth_hf_series_sol_15m: str = Field(
        default="10423", description="Polymarket series ID for SOL 15-min markets"
    )
    btc_eth_hf_series_xrp_15m: str = Field(
        default="10422", description="Polymarket series ID for XRP 15-min markets"
    )
    btc_eth_hf_series_btc_5m: str = Field(
        default="10684", description="Polymarket series ID for BTC 5-min markets"
    )
    btc_eth_hf_series_eth_5m: str = Field(
        default="", description="Polymarket series ID for ETH 5-min markets"
    )
    btc_eth_hf_series_sol_5m: str = Field(
        default="", description="Polymarket series ID for SOL 5-min markets"
    )
    btc_eth_hf_series_xrp_5m: str = Field(
        default="", description="Polymarket series ID for XRP 5-min markets"
    )
    btc_eth_hf_series_btc_1h: str = Field(
        default="10114", description="Polymarket series ID for BTC hourly markets"
    )
    btc_eth_hf_series_eth_1h: str = Field(
        default="10117", description="Polymarket series ID for ETH hourly markets"
    )
    btc_eth_hf_series_sol_1h: str = Field(
        default="10122", description="Polymarket series ID for SOL hourly markets"
    )
    btc_eth_hf_series_xrp_1h: str = Field(
        default="10123", description="Polymarket series ID for XRP hourly markets"
    )
    btc_eth_hf_series_btc_4h: str = Field(
        default="10331", description="Polymarket series ID for BTC 4-hour markets"
    )
    btc_eth_hf_series_eth_4h: str = Field(
        default="10332", description="Polymarket series ID for ETH 4-hour markets"
    )
    btc_eth_hf_series_sol_4h: str = Field(
        default="10326", description="Polymarket series ID for SOL 4-hour markets"
    )
    btc_eth_hf_series_xrp_4h: str = Field(
        default="10327", description="Polymarket series ID for XRP 4-hour markets"
    )
    btc_eth_pure_arb_max_combined: float = Field(
        default=0.98, ge=0.5, le=1.0, description="Use pure arb when YES+NO < this"
    )
    btc_eth_dump_hedge_drop_pct: float = Field(
        default=0.05,
        ge=0.01,
        le=0.5,
        description="Min price drop to trigger dump-hedge",
    )
    btc_eth_thin_liquidity_usd: float = Field(
        default=500.0, ge=0, description="Below this = thin order book ($)"
    )

    # Miracle strategy
    miracle_min_no_price: float = Field(
        default=0.90, ge=0.5, le=0.999, description="Only consider NO prices >= this"
    )
    miracle_max_no_price: float = Field(
        default=0.995, ge=0.9, le=1.0, description="Skip if NO already at this+"
    )
    miracle_min_impossibility_score: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Min confidence event is impossible (0-1)",
    )

    # BTC/ETH high-frequency enable
    btc_eth_hf_enabled: bool = Field(
        default=True, description="Enable BTC/ETH high-frequency strategy"
    )

    # Cross-platform arbitrage
    cross_platform_enabled: bool = Field(
        default=True,
        description="Enable cross-platform (Polymarket vs Kalshi) arbitrage",
    )

    # Combinatorial arbitrage
    combinatorial_min_confidence: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Min LLM confidence for combinatorial trades",
    )
    combinatorial_high_confidence: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="High confidence threshold for combinatorial",
    )

    # Bayesian cascade
    bayesian_cascade_enabled: bool = Field(
        default=True, description="Enable Bayesian cascade strategy"
    )
    bayesian_min_edge_percent: float = Field(
        default=5.0,
        ge=0.0,
        le=100.0,
        description="Min expected-vs-actual price diff to flag (%)",
    )
    bayesian_propagation_depth: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max hops through the dependency graph",
    )

    # Liquidity vacuum
    liquidity_vacuum_enabled: bool = Field(
        default=True, description="Enable liquidity vacuum strategy"
    )
    liquidity_vacuum_min_imbalance_ratio: float = Field(
        default=5.0,
        ge=1.0,
        le=100.0,
        description="Min bid/ask imbalance ratio to trigger",
    )
    liquidity_vacuum_min_depth_usd: float = Field(
        default=100.0, ge=0, description="Min order book depth ($)"
    )

    # Entropy arbitrage
    entropy_arb_enabled: bool = Field(
        default=True, description="Enable entropy arbitrage strategy"
    )
    entropy_arb_min_deviation: float = Field(
        default=0.25,
        ge=0.0,
        le=2.0,
        description="Min entropy deviation from expected decay",
    )

    # Event-driven arbitrage
    event_driven_enabled: bool = Field(
        default=True, description="Enable event-driven arbitrage strategy"
    )

    # Temporal decay
    temporal_decay_enabled: bool = Field(
        default=True, description="Enable temporal decay strategy"
    )

    # Correlation arbitrage
    correlation_arb_enabled: bool = Field(
        default=True, description="Enable correlation arbitrage strategy"
    )
    correlation_arb_min_correlation: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Min correlation coefficient for pair detection",
    )
    correlation_arb_min_divergence: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Min price divergence to trigger trade",
    )

    # Market making
    market_making_enabled: bool = Field(
        default=True, description="Enable market making strategy"
    )
    market_making_spread_bps: float = Field(
        default=100.0,
        ge=10.0,
        le=1000.0,
        description="Min bid-ask spread in basis points",
    )
    market_making_max_inventory_usd: float = Field(
        default=500.0, ge=0, description="Max inventory per market ($)"
    )

    # Statistical arbitrage
    stat_arb_enabled: bool = Field(
        default=True, description="Enable statistical arbitrage strategy"
    )
    stat_arb_min_edge: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Min composite fair-value edge to trade",
    )


class AllSettings(BaseModel):
    """Complete settings response"""

    polymarket: PolymarketSettings
    kalshi: KalshiSettings
    llm: LLMSettings
    notifications: NotificationSettings
    scanner: ScannerSettingsModel
    trading: TradingSettings
    maintenance: MaintenanceSettings
    discovery: DiscoverySettings
    trading_proxy: TradingProxySettings
    search_filters: SearchFilterSettings
    updated_at: Optional[str] = None


class UpdateSettingsRequest(BaseModel):
    """Request to update settings (partial updates supported)"""

    polymarket: Optional[PolymarketSettings] = None
    kalshi: Optional[KalshiSettings] = None
    llm: Optional[LLMSettings] = None
    notifications: Optional[NotificationSettings] = None
    scanner: Optional[ScannerSettingsModel] = None
    trading: Optional[TradingSettings] = None
    maintenance: Optional[MaintenanceSettings] = None
    discovery: Optional[DiscoverySettings] = None
    trading_proxy: Optional[TradingProxySettings] = None
    search_filters: Optional[SearchFilterSettings] = None


async def get_or_create_settings() -> AppSettings:
    """Get existing settings or create default"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(AppSettings).where(AppSettings.id == "default")
        )
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
            trading=TradingSettings(**trading_payload(settings)),
            maintenance=MaintenanceSettings(**maintenance_payload(settings)),
            discovery=DiscoverySettings(**discovery_payload(settings)),
            trading_proxy=TradingProxySettings(**trading_proxy_payload(settings)),
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
            result = await session.execute(
                select(AppSettings).where(AppSettings.id == "default")
            )
            settings = result.scalar_one_or_none()

            if not settings:
                settings = AppSettings(id="default")
                session.add(settings)

            update_flags = apply_update_request(settings, request)
            await session.commit()
            updated_at = settings.updated_at.isoformat()
            needs_llm_reinit = update_flags["needs_llm_reinit"]
            needs_proxy_reinit = update_flags["needs_proxy_reinit"]
            needs_filter_reload = update_flags["needs_filter_reload"]

        # Re-initialize LLM manager OUTSIDE the DB session so the new
        # session inside initialize() can see the just-committed data
        # (SQLite + aiosqlite share a single connection in the pool).
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

        # Reload search filter config into the running settings singleton
        if needs_filter_reload:
            try:
                from config import apply_search_filters

                await apply_search_filters()
                logger.info("Search filter config reloaded after settings update")
            except Exception as reinit_err:
                logger.error(
                    f"Failed to reload search filters: {reinit_err}",
                )

        logger.info("Settings updated successfully")

        return {
            "status": "success",
            "message": "Settings updated successfully",
            "updated_at": updated_at,
        }
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


@router.get("/trading")
async def get_trading_settings():
    """Get trading settings only"""
    settings = await get_or_create_settings()
    return TradingSettings(**trading_payload(settings))


@router.put("/trading")
async def update_trading_settings(request: TradingSettings):
    """Update trading settings only"""
    return await update_settings(UpdateSettingsRequest(trading=request))


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


@router.get("/search-filters")
async def get_search_filter_settings():
    """Get search filter settings only"""
    settings = await get_or_create_settings()
    return SearchFilterSettings(**search_filters_payload(settings))


@router.put("/search-filters")
async def update_search_filter_settings(request: SearchFilterSettings):
    """Update search filter settings only"""
    return await update_settings(UpdateSettingsRequest(search_filters=request))


# ==================== VALIDATION ENDPOINTS ====================


@router.post("/test/polymarket")
async def test_polymarket_connection():
    """Test Polymarket API connection with stored credentials"""
    try:
        settings = await get_or_create_settings()

        if not decrypt_secret(settings.polymarket_api_key):
            return {"status": "error", "message": "Polymarket API key not configured"}

        # TODO: Implement actual API test
        return {
            "status": "success",
            "message": "Polymarket credentials are configured (connection test not implemented)",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


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

        if not status.get("proxy_reachable"):
            return {
                "status": "error",
                "message": f"Proxy unreachable: {status.get('proxy_ip_error', 'unknown error')}",
                **status,
            }

        if status.get("vpn_active"):
            return {
                "status": "success",
                "message": f"VPN active — trading through {status.get('proxy_ip')}",
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
