"""
Settings API Routes

Endpoints for managing application settings.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

from sqlalchemy import select
from models.database import AsyncSessionLocal, AppSettings
from utils.logger import get_logger

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
        description="LLM provider: none, openai, anthropic, google, xai, deepseek",
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


class AllSettings(BaseModel):
    """Complete settings response"""

    polymarket: PolymarketSettings
    kalshi: KalshiSettings
    llm: LLMSettings
    notifications: NotificationSettings
    scanner: ScannerSettingsModel
    trading: TradingSettings
    maintenance: MaintenanceSettings
    trading_proxy: TradingProxySettings
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
    trading_proxy: Optional[TradingProxySettings] = None


# ==================== HELPER FUNCTIONS ====================


def mask_secret(value: Optional[str], show_chars: int = 4) -> Optional[str]:
    """Mask a secret value, showing only first few characters"""
    if not value:
        return None
    if len(value) <= show_chars:
        return "*" * len(value)
    return value[:show_chars] + "*" * (len(value) - show_chars)


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
            polymarket=PolymarketSettings(
                api_key=mask_secret(settings.polymarket_api_key),
                api_secret=mask_secret(settings.polymarket_api_secret),
                api_passphrase=mask_secret(settings.polymarket_api_passphrase),
                private_key=mask_secret(settings.polymarket_private_key),
            ),
            kalshi=KalshiSettings(
                email=settings.kalshi_email,
                password=mask_secret(settings.kalshi_password),
                api_key=mask_secret(settings.kalshi_api_key),
            ),
            llm=LLMSettings(
                provider=settings.llm_provider or "none",
                openai_api_key=mask_secret(settings.openai_api_key),
                anthropic_api_key=mask_secret(settings.anthropic_api_key),
                google_api_key=mask_secret(settings.google_api_key),
                xai_api_key=mask_secret(settings.xai_api_key),
                deepseek_api_key=mask_secret(settings.deepseek_api_key),
                model=settings.llm_model,
                max_monthly_spend=settings.ai_max_monthly_spend,
            ),
            notifications=NotificationSettings(
                enabled=settings.notifications_enabled,
                telegram_bot_token=mask_secret(settings.telegram_bot_token),
                telegram_chat_id=settings.telegram_chat_id,
                notify_on_opportunity=settings.notify_on_opportunity,
                notify_on_trade=settings.notify_on_trade,
                notify_min_roi=settings.notify_min_roi,
            ),
            scanner=ScannerSettingsModel(
                scan_interval_seconds=settings.scan_interval_seconds,
                min_profit_threshold=settings.min_profit_threshold,
                max_markets_to_scan=settings.max_markets_to_scan,
                min_liquidity=settings.min_liquidity,
            ),
            trading=TradingSettings(
                trading_enabled=settings.trading_enabled,
                max_trade_size_usd=settings.max_trade_size_usd,
                max_daily_trade_volume=settings.max_daily_trade_volume,
                max_open_positions=settings.max_open_positions,
                max_slippage_percent=settings.max_slippage_percent,
            ),
            maintenance=MaintenanceSettings(
                auto_cleanup_enabled=settings.auto_cleanup_enabled,
                cleanup_interval_hours=settings.cleanup_interval_hours,
                cleanup_resolved_trade_days=settings.cleanup_resolved_trade_days,
            ),
            trading_proxy=TradingProxySettings(
                enabled=settings.trading_proxy_enabled or False,
                proxy_url=mask_secret(settings.trading_proxy_url, show_chars=12),
                verify_ssl=settings.trading_proxy_verify_ssl
                if settings.trading_proxy_verify_ssl is not None
                else True,
                timeout=settings.trading_proxy_timeout or 30.0,
                require_vpn=settings.trading_proxy_require_vpn
                if settings.trading_proxy_require_vpn is not None
                else True,
            ),
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

            # Update Polymarket settings
            if request.polymarket:
                pm = request.polymarket
                if pm.api_key is not None:
                    settings.polymarket_api_key = pm.api_key or None
                if pm.api_secret is not None:
                    settings.polymarket_api_secret = pm.api_secret or None
                if pm.api_passphrase is not None:
                    settings.polymarket_api_passphrase = pm.api_passphrase or None
                if pm.private_key is not None:
                    settings.polymarket_private_key = pm.private_key or None

            # Update Kalshi settings
            if request.kalshi:
                kal = request.kalshi
                if kal.email is not None:
                    settings.kalshi_email = kal.email or None
                if kal.password is not None:
                    settings.kalshi_password = kal.password or None
                if kal.api_key is not None:
                    settings.kalshi_api_key = kal.api_key or None

            # Update LLM settings
            if request.llm:
                llm = request.llm
                if llm.provider is not None:
                    settings.llm_provider = llm.provider
                if llm.openai_api_key is not None:
                    settings.openai_api_key = llm.openai_api_key or None
                if llm.anthropic_api_key is not None:
                    settings.anthropic_api_key = llm.anthropic_api_key or None
                if llm.google_api_key is not None:
                    settings.google_api_key = llm.google_api_key or None
                if llm.xai_api_key is not None:
                    settings.xai_api_key = llm.xai_api_key or None
                if llm.deepseek_api_key is not None:
                    settings.deepseek_api_key = llm.deepseek_api_key or None
                if llm.model is not None:
                    settings.llm_model = llm.model or None
                    settings.ai_default_model = llm.model or None
                if llm.max_monthly_spend is not None:
                    settings.ai_max_monthly_spend = llm.max_monthly_spend

            # Update Notification settings
            if request.notifications:
                notif = request.notifications
                settings.notifications_enabled = notif.enabled
                if notif.telegram_bot_token is not None:
                    settings.telegram_bot_token = notif.telegram_bot_token or None
                if notif.telegram_chat_id is not None:
                    settings.telegram_chat_id = notif.telegram_chat_id or None
                settings.notify_on_opportunity = notif.notify_on_opportunity
                settings.notify_on_trade = notif.notify_on_trade
                settings.notify_min_roi = notif.notify_min_roi

            # Update Scanner settings
            if request.scanner:
                scan = request.scanner
                settings.scan_interval_seconds = scan.scan_interval_seconds
                settings.min_profit_threshold = scan.min_profit_threshold
                settings.max_markets_to_scan = scan.max_markets_to_scan
                settings.min_liquidity = scan.min_liquidity

            # Update Trading settings
            if request.trading:
                trade = request.trading
                settings.trading_enabled = trade.trading_enabled
                settings.max_trade_size_usd = trade.max_trade_size_usd
                settings.max_daily_trade_volume = trade.max_daily_trade_volume
                settings.max_open_positions = trade.max_open_positions
                settings.max_slippage_percent = trade.max_slippage_percent

            # Update Maintenance settings
            if request.maintenance:
                maint = request.maintenance
                settings.auto_cleanup_enabled = maint.auto_cleanup_enabled
                settings.cleanup_interval_hours = maint.cleanup_interval_hours
                settings.cleanup_resolved_trade_days = maint.cleanup_resolved_trade_days

            # Update Trading Proxy settings
            if request.trading_proxy:
                proxy = request.trading_proxy
                settings.trading_proxy_enabled = proxy.enabled
                if proxy.proxy_url is not None:
                    settings.trading_proxy_url = proxy.proxy_url or None
                settings.trading_proxy_verify_ssl = proxy.verify_ssl
                settings.trading_proxy_timeout = proxy.timeout
                settings.trading_proxy_require_vpn = proxy.require_vpn

            settings.updated_at = datetime.utcnow()
            await session.commit()
            updated_at = settings.updated_at.isoformat()
            needs_llm_reinit = bool(request.llm)
            needs_proxy_reinit = bool(request.trading_proxy)

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
    return PolymarketSettings(
        api_key=mask_secret(settings.polymarket_api_key),
        api_secret=mask_secret(settings.polymarket_api_secret),
        api_passphrase=mask_secret(settings.polymarket_api_passphrase),
        private_key=mask_secret(settings.polymarket_private_key),
    )


@router.put("/polymarket")
async def update_polymarket_settings(request: PolymarketSettings):
    """Update Polymarket settings only"""
    return await update_settings(UpdateSettingsRequest(polymarket=request))


@router.get("/kalshi")
async def get_kalshi_settings():
    """Get Kalshi settings only"""
    settings = await get_or_create_settings()
    return KalshiSettings(
        email=settings.kalshi_email,
        password=mask_secret(settings.kalshi_password),
        api_key=mask_secret(settings.kalshi_api_key),
    )


@router.put("/kalshi")
async def update_kalshi_settings(request: KalshiSettings):
    """Update Kalshi settings only"""
    return await update_settings(UpdateSettingsRequest(kalshi=request))


@router.get("/llm")
async def get_llm_settings():
    """Get LLM settings only"""
    settings = await get_or_create_settings()
    return LLMSettings(
        provider=settings.llm_provider or "none",
        openai_api_key=mask_secret(settings.openai_api_key),
        anthropic_api_key=mask_secret(settings.anthropic_api_key),
        google_api_key=mask_secret(settings.google_api_key),
        xai_api_key=mask_secret(settings.xai_api_key),
        deepseek_api_key=mask_secret(settings.deepseek_api_key),
        model=settings.llm_model,
    )


@router.put("/llm")
async def update_llm_settings(request: LLMSettings):
    """Update LLM settings only"""
    return await update_settings(UpdateSettingsRequest(llm=request))


@router.get("/notifications")
async def get_notification_settings():
    """Get notification settings only"""
    settings = await get_or_create_settings()
    return NotificationSettings(
        enabled=settings.notifications_enabled,
        telegram_bot_token=mask_secret(settings.telegram_bot_token),
        telegram_chat_id=settings.telegram_chat_id,
        notify_on_opportunity=settings.notify_on_opportunity,
        notify_on_trade=settings.notify_on_trade,
        notify_min_roi=settings.notify_min_roi,
    )


@router.put("/notifications")
async def update_notification_settings(request: NotificationSettings):
    """Update notification settings only"""
    return await update_settings(UpdateSettingsRequest(notifications=request))


@router.get("/scanner")
async def get_scanner_settings():
    """Get scanner settings only"""
    settings = await get_or_create_settings()
    return ScannerSettingsModel(
        scan_interval_seconds=settings.scan_interval_seconds,
        min_profit_threshold=settings.min_profit_threshold,
        max_markets_to_scan=settings.max_markets_to_scan,
        min_liquidity=settings.min_liquidity,
    )


@router.put("/scanner")
async def update_scanner_settings(request: ScannerSettingsModel):
    """Update scanner settings only"""
    return await update_settings(UpdateSettingsRequest(scanner=request))


@router.get("/trading")
async def get_trading_settings():
    """Get trading settings only"""
    settings = await get_or_create_settings()
    return TradingSettings(
        trading_enabled=settings.trading_enabled,
        max_trade_size_usd=settings.max_trade_size_usd,
        max_daily_trade_volume=settings.max_daily_trade_volume,
        max_open_positions=settings.max_open_positions,
        max_slippage_percent=settings.max_slippage_percent,
    )


@router.put("/trading")
async def update_trading_settings(request: TradingSettings):
    """Update trading settings only"""
    return await update_settings(UpdateSettingsRequest(trading=request))


@router.get("/maintenance")
async def get_maintenance_settings():
    """Get maintenance settings only"""
    settings = await get_or_create_settings()
    return MaintenanceSettings(
        auto_cleanup_enabled=settings.auto_cleanup_enabled,
        cleanup_interval_hours=settings.cleanup_interval_hours,
        cleanup_resolved_trade_days=settings.cleanup_resolved_trade_days,
    )


@router.put("/maintenance")
async def update_maintenance_settings(request: MaintenanceSettings):
    """Update maintenance settings only"""
    return await update_settings(UpdateSettingsRequest(maintenance=request))


@router.get("/trading-proxy")
async def get_trading_proxy_settings():
    """Get trading VPN/proxy settings only"""
    settings = await get_or_create_settings()
    return TradingProxySettings(
        enabled=settings.trading_proxy_enabled or False,
        proxy_url=mask_secret(settings.trading_proxy_url, show_chars=12),
        verify_ssl=settings.trading_proxy_verify_ssl
        if settings.trading_proxy_verify_ssl is not None
        else True,
        timeout=settings.trading_proxy_timeout or 30.0,
        require_vpn=settings.trading_proxy_require_vpn
        if settings.trading_proxy_require_vpn is not None
        else True,
    )


@router.put("/trading-proxy")
async def update_trading_proxy_settings(request: TradingProxySettings):
    """Update trading VPN/proxy settings only"""
    return await update_settings(UpdateSettingsRequest(trading_proxy=request))


# ==================== VALIDATION ENDPOINTS ====================


@router.post("/test/polymarket")
async def test_polymarket_connection():
    """Test Polymarket API connection with stored credentials"""
    try:
        settings = await get_or_create_settings()

        if not settings.polymarket_api_key:
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

        if not settings.telegram_bot_token or not settings.telegram_chat_id:
            return {
                "status": "error",
                "message": "Telegram bot token or chat ID not configured",
            }

        # TODO: Implement actual bot test (send test message)
        return {
            "status": "success",
            "message": "Telegram credentials are configured (connection test not implemented)",
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
