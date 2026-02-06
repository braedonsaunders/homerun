from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from pydantic import BaseModel, Field, field_validator

from services.copy_trader import copy_trader
from utils.validation import validate_eth_address

copy_trading_router = APIRouter()


class CreateCopyConfigRequest(BaseModel):
    source_wallet: str
    account_id: str
    copy_mode: str = Field(
        default="all_trades",
        description="all_trades = mirror every trade; arb_only = only copy arb-matching trades",
    )
    min_roi_threshold: float = Field(default=2.5, ge=0.0, le=100.0)
    max_position_size: float = Field(default=1000.0, ge=10.0, le=1000000.0)
    copy_delay_seconds: int = Field(default=5, ge=0, le=300)
    slippage_tolerance: float = Field(default=1.0, ge=0.0, le=10.0)
    proportional_sizing: bool = Field(
        default=False, description="Scale positions relative to source wallet size"
    )
    proportional_multiplier: float = Field(
        default=1.0,
        ge=0.01,
        le=100.0,
        description="Multiplier for proportional sizing (0.1 = 10% of source size)",
    )
    copy_buys: bool = Field(default=True, description="Copy buy trades")
    copy_sells: bool = Field(default=True, description="Copy sell/close trades")
    market_categories: list[str] = Field(
        default=[],
        description="Only copy trades in these market categories (empty = all)",
    )

    @field_validator("source_wallet")
    @classmethod
    def validate_wallet(cls, v: str) -> str:
        return validate_eth_address(v)

    @field_validator("copy_mode")
    @classmethod
    def validate_copy_mode(cls, v: str) -> str:
        if v not in ("all_trades", "arb_only"):
            raise ValueError("copy_mode must be 'all_trades' or 'arb_only'")
        return v


class UpdateCopyConfigRequest(BaseModel):
    enabled: Optional[bool] = None
    copy_mode: Optional[str] = None
    min_roi_threshold: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    max_position_size: Optional[float] = Field(default=None, ge=10.0, le=1000000.0)
    copy_delay_seconds: Optional[int] = Field(default=None, ge=0, le=300)
    slippage_tolerance: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    proportional_sizing: Optional[bool] = None
    proportional_multiplier: Optional[float] = Field(default=None, ge=0.01, le=100.0)
    copy_buys: Optional[bool] = None
    copy_sells: Optional[bool] = None
    market_categories: Optional[list[str]] = None

    @field_validator("copy_mode")
    @classmethod
    def validate_copy_mode(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("all_trades", "arb_only"):
            raise ValueError("copy_mode must be 'all_trades' or 'arb_only'")
        return v


# ==================== COPY CONFIGURATIONS ====================


@copy_trading_router.post("/configs")
async def create_copy_config(request: CreateCopyConfigRequest):
    """Create a new copy trading configuration"""
    try:
        config = await copy_trader.add_copy_config(
            source_wallet=request.source_wallet,
            account_id=request.account_id,
            copy_mode=request.copy_mode,
            min_roi_threshold=request.min_roi_threshold,
            max_position_size=request.max_position_size,
            copy_delay_seconds=request.copy_delay_seconds,
            slippage_tolerance=request.slippage_tolerance,
            proportional_sizing=request.proportional_sizing,
            proportional_multiplier=request.proportional_multiplier,
            copy_buys=request.copy_buys,
            copy_sells=request.copy_sells,
            market_categories=request.market_categories,
        )

        return {
            "config_id": config.id,
            "source_wallet": config.source_wallet,
            "account_id": config.account_id,
            "enabled": config.enabled,
            "copy_mode": config.copy_mode.value,
            "message": "Copy trading configuration created",
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@copy_trading_router.get("/configs")
async def list_copy_configs(account_id: Optional[str] = Query(default=None)):
    """List all copy trading configurations"""
    configs = await copy_trader.get_configs(account_id)
    return [
        {
            "id": cfg.id,
            "source_wallet": cfg.source_wallet,
            "account_id": cfg.account_id,
            "enabled": cfg.enabled,
            "copy_mode": cfg.copy_mode.value if cfg.copy_mode else "all_trades",
            "settings": {
                "min_roi_threshold": cfg.min_roi_threshold,
                "max_position_size": cfg.max_position_size,
                "copy_delay_seconds": cfg.copy_delay_seconds,
                "slippage_tolerance": cfg.slippage_tolerance,
                "proportional_sizing": cfg.proportional_sizing,
                "proportional_multiplier": cfg.proportional_multiplier,
                "copy_buys": cfg.copy_buys,
                "copy_sells": cfg.copy_sells,
                "market_categories": cfg.market_categories,
            },
            "stats": {
                "total_copied": cfg.total_copied,
                "successful_copies": cfg.successful_copies,
                "failed_copies": cfg.failed_copies,
                "total_pnl": cfg.total_pnl,
                "total_buys_copied": cfg.total_buys_copied,
                "total_sells_copied": cfg.total_sells_copied,
            },
        }
        for cfg in configs
    ]


@copy_trading_router.get("/configs/{config_id}")
async def get_copy_config(config_id: str):
    """Get detailed copy trading configuration"""
    stats = await copy_trader.get_copy_stats(config_id)
    if not stats:
        raise HTTPException(status_code=404, detail="Configuration not found")
    return stats


@copy_trading_router.patch("/configs/{config_id}")
async def update_copy_config(config_id: str, request: UpdateCopyConfigRequest):
    """Update a copy trading configuration"""
    update_fields = request.model_dump(exclude_none=True)
    if not update_fields:
        return {"message": "No fields to update", "config_id": config_id}

    try:
        await copy_trader.update_config(config_id, **update_fields)
        return {"message": "Configuration updated", "config_id": config_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@copy_trading_router.delete("/configs/{config_id}")
async def delete_copy_config(config_id: str):
    """Delete a copy trading configuration"""
    await copy_trader.remove_copy_config(config_id)
    return {"message": "Configuration deleted", "config_id": config_id}


@copy_trading_router.post("/configs/{config_id}/enable")
async def enable_copy_config(config_id: str):
    """Enable a copy trading configuration"""
    await copy_trader.enable_config(config_id, True)
    return {"message": "Copy trading enabled", "config_id": config_id}


@copy_trading_router.post("/configs/{config_id}/disable")
async def disable_copy_config(config_id: str):
    """Disable a copy trading configuration"""
    await copy_trader.enable_config(config_id, False)
    return {"message": "Copy trading disabled", "config_id": config_id}


@copy_trading_router.post("/configs/{config_id}/sync")
async def force_sync_config(config_id: str):
    """Force an immediate sync for a copy trading configuration.
    Processes all uncopied trades from the source wallet immediately."""
    try:
        result = await copy_trader.force_sync(config_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ==================== COPIED TRADE HISTORY ====================


@copy_trading_router.get("/trades")
async def list_copied_trades(
    config_id: Optional[str] = Query(default=None),
    status: Optional[str] = Query(
        default=None, description="Filter by status: pending, executed, failed, skipped"
    ),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    """Get history of all copied trades with optional filters"""
    trades = await copy_trader.get_copied_trades(
        config_id=config_id,
        status=status,
        limit=limit,
        offset=offset,
    )
    return trades


# ==================== SOURCE WALLET ====================


@copy_trading_router.get("/wallet/{wallet_address}/positions")
async def get_source_wallet_positions(wallet_address: str):
    """Get current open positions for a source wallet being copied"""
    wallet_address = validate_eth_address(wallet_address)
    positions = await copy_trader.get_source_wallet_positions(wallet_address)
    return {
        "wallet": wallet_address,
        "positions_count": len(positions),
        "positions": positions,
    }


# ==================== SERVICE STATUS ====================


@copy_trading_router.get("/status")
async def get_copy_trading_status():
    """Get copy trading service status"""
    configs = await copy_trader.get_configs()
    enabled_configs = [c for c in configs if c.enabled]

    return {
        "service_running": copy_trader._running,
        "poll_interval_seconds": copy_trader._poll_interval,
        "total_configs": len(configs),
        "enabled_configs": len(enabled_configs),
        "tracked_wallets": list(set(c.source_wallet for c in configs)),
        "configs_summary": [
            {
                "id": c.id,
                "source_wallet": c.source_wallet,
                "copy_mode": c.copy_mode.value if c.copy_mode else "all_trades",
                "enabled": c.enabled,
                "total_copied": c.total_copied,
                "successful_copies": c.successful_copies,
            }
            for c in configs
        ],
    }
