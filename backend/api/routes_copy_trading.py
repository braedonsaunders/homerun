from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from pydantic import BaseModel, Field, field_validator

from services.copy_trader import copy_trader
from utils.validation import validate_eth_address

copy_trading_router = APIRouter()


class CreateCopyConfigRequest(BaseModel):
    source_wallet: str
    account_id: str
    min_roi_threshold: float = Field(default=2.5, ge=0.0, le=100.0)
    max_position_size: float = Field(default=1000.0, ge=10.0, le=1000000.0)
    copy_delay_seconds: int = Field(default=5, ge=0, le=300)
    slippage_tolerance: float = Field(default=1.0, ge=0.0, le=10.0)

    @field_validator("source_wallet")
    @classmethod
    def validate_wallet(cls, v: str) -> str:
        return validate_eth_address(v)


class UpdateCopyConfigRequest(BaseModel):
    enabled: Optional[bool] = None
    min_roi_threshold: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    max_position_size: Optional[float] = Field(default=None, ge=10.0, le=1000000.0)
    copy_delay_seconds: Optional[int] = Field(default=None, ge=0, le=300)
    slippage_tolerance: Optional[float] = Field(default=None, ge=0.0, le=10.0)


# ==================== COPY CONFIGURATIONS ====================

@copy_trading_router.post("/configs")
async def create_copy_config(request: CreateCopyConfigRequest):
    """Create a new copy trading configuration"""
    try:
        config = await copy_trader.add_copy_config(
            source_wallet=request.source_wallet,
            account_id=request.account_id,
            min_roi_threshold=request.min_roi_threshold,
            max_position_size=request.max_position_size,
            copy_delay_seconds=request.copy_delay_seconds,
            slippage_tolerance=request.slippage_tolerance
        )

        return {
            "config_id": config.id,
            "source_wallet": config.source_wallet,
            "account_id": config.account_id,
            "enabled": config.enabled,
            "message": "Copy trading configuration created"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@copy_trading_router.get("/configs")
async def list_copy_configs(
    account_id: Optional[str] = Query(default=None)
):
    """List all copy trading configurations"""
    configs = await copy_trader.get_configs(account_id)
    return [{
        "id": cfg.id,
        "source_wallet": cfg.source_wallet,
        "account_id": cfg.account_id,
        "enabled": cfg.enabled,
        "settings": {
            "min_roi_threshold": cfg.min_roi_threshold,
            "max_position_size": cfg.max_position_size,
            "copy_delay_seconds": cfg.copy_delay_seconds,
            "slippage_tolerance": cfg.slippage_tolerance
        },
        "stats": {
            "total_copied": cfg.total_copied,
            "successful_copies": cfg.successful_copies,
            "failed_copies": cfg.failed_copies,
            "total_pnl": cfg.total_pnl
        }
    } for cfg in configs]


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
    if request.enabled is not None:
        await copy_trader.enable_config(config_id, request.enabled)

    # TODO: Update other fields if provided

    return {"message": "Configuration updated", "config_id": config_id}


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
        "tracked_wallets": list(set(c.source_wallet for c in configs))
    }
