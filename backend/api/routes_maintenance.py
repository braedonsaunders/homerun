"""
Database Maintenance API Routes

Endpoints for cleaning up old trades and managing database health.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

from services.maintenance import maintenance_service
from models.database import TradeStatus
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/maintenance", tags=["Maintenance"])


# ==================== REQUEST MODELS ====================

class CleanupRequest(BaseModel):
    """Request for cleanup operations"""
    resolved_trade_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Delete resolved trades older than this many days"
    )
    open_trade_expiry_days: int = Field(
        default=90,
        ge=1,
        le=365,
        description="Expire open trades older than this many days"
    )
    wallet_trade_days: int = Field(
        default=60,
        ge=1,
        le=365,
        description="Delete wallet trades older than this many days"
    )
    anomaly_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Delete resolved anomalies older than this many days"
    )


class DeleteTradesRequest(BaseModel):
    """Request for deleting trades"""
    older_than_days: Optional[int] = Field(
        default=None,
        ge=1,
        le=365,
        description="Delete trades older than this many days"
    )
    statuses: Optional[list[str]] = Field(
        default=None,
        description="Delete trades with these statuses (e.g., ['resolved_win', 'resolved_loss'])"
    )
    account_id: Optional[str] = Field(
        default=None,
        description="Only delete trades for this account"
    )
    delete_all: bool = Field(
        default=False,
        description="Delete ALL trades (dangerous!)"
    )
    confirm: bool = Field(
        default=False,
        description="Must be True to proceed with delete_all"
    )


# ==================== ENDPOINTS ====================

@router.get("/stats")
async def get_database_stats():
    """
    Get database statistics.

    Returns counts of trades, positions, and other records.
    Useful for understanding database size before cleanup.
    """
    try:
        stats = await maintenance_service.get_database_stats()
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "stats": stats
        }
    except Exception as e:
        logger.error("Failed to get database stats", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup")
async def run_cleanup(request: CleanupRequest = CleanupRequest()):
    """
    Run full database cleanup.

    This will:
    1. Expire old open trades (mark as cancelled)
    2. Delete resolved trades older than specified days
    3. Delete old wallet trades
    4. Delete resolved anomalies

    Defaults:
    - resolved_trade_days: 30
    - open_trade_expiry_days: 90
    - wallet_trade_days: 60
    - anomaly_days: 30
    """
    try:
        results = await maintenance_service.full_cleanup(
            resolved_trade_days=request.resolved_trade_days,
            open_trade_expiry_days=request.open_trade_expiry_days,
            wallet_trade_days=request.wallet_trade_days,
            anomaly_days=request.anomaly_days
        )
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "results": results
        }
    except Exception as e:
        logger.error("Cleanup failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/expire-old-trades")
async def expire_old_trades(
    older_than_days: int = Query(
        default=90,
        ge=1,
        le=365,
        description="Expire open trades older than this many days"
    )
):
    """
    Expire old open trades.

    Marks trades that have been open for too long as cancelled.
    This handles markets that were cancelled or never resolved.
    """
    try:
        result = await maintenance_service.expire_old_open_trades(
            older_than_days=older_than_days
        )
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            **result
        }
    except Exception as e:
        logger.error("Failed to expire old trades", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/trades")
async def delete_trades(request: DeleteTradesRequest):
    """
    Delete trades based on criteria.

    Options:
    - older_than_days: Delete resolved trades older than X days
    - statuses: Delete trades with specific statuses
    - account_id: Only delete for specific account
    - delete_all: Delete ALL trades (requires confirm=True)

    At least one filter (older_than_days, statuses, or delete_all) must be specified.
    """
    try:
        # Validate request
        if not request.older_than_days and not request.statuses and not request.delete_all:
            raise HTTPException(
                status_code=400,
                detail="Must specify older_than_days, statuses, or delete_all"
            )

        results = {}

        if request.delete_all:
            if not request.confirm:
                raise HTTPException(
                    status_code=400,
                    detail="Must set confirm=True to delete all trades"
                )
            results = await maintenance_service.delete_all_trades(
                account_id=request.account_id,
                confirm=True
            )
        elif request.statuses:
            # Convert status strings to enums
            try:
                status_enums = [TradeStatus(s) for s in request.statuses]
            except ValueError as e:
                valid_statuses = [s.value for s in TradeStatus]
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status. Valid statuses: {valid_statuses}"
                )

            results = await maintenance_service.delete_trades_by_status(
                statuses=status_enums,
                account_id=request.account_id
            )
        elif request.older_than_days:
            results = await maintenance_service.cleanup_resolved_trades(
                older_than_days=request.older_than_days,
                account_id=request.account_id
            )

        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            **results
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete trades", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/wallet-trades")
async def delete_wallet_trades(
    older_than_days: int = Query(
        default=60,
        ge=1,
        le=365,
        description="Delete wallet trades older than this many days"
    ),
    wallet_address: Optional[str] = Query(
        default=None,
        description="Only delete for specific wallet"
    )
):
    """
    Delete old wallet trades.

    These are trades tracked from monitored wallets.
    """
    try:
        result = await maintenance_service.cleanup_wallet_trades(
            older_than_days=older_than_days,
            wallet_address=wallet_address
        )
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            **result
        }
    except Exception as e:
        logger.error("Failed to delete wallet trades", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/anomalies")
async def delete_anomalies(
    older_than_days: int = Query(
        default=30,
        ge=1,
        le=365,
        description="Delete anomalies older than this many days"
    ),
    resolved_only: bool = Query(
        default=True,
        description="Only delete resolved anomalies"
    )
):
    """
    Delete old anomaly records.
    """
    try:
        result = await maintenance_service.cleanup_anomalies(
            older_than_days=older_than_days,
            resolved_only=resolved_only
        )
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            **result
        }
    except Exception as e:
        logger.error("Failed to delete anomalies", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ==================== CONVENIENCE ENDPOINTS ====================

@router.post("/cleanup/resolved")
async def cleanup_resolved_only(
    older_than_days: int = Query(
        default=30,
        ge=1,
        le=365,
        description="Delete resolved trades older than this many days"
    )
):
    """
    Quick cleanup of resolved trades only.

    Deletes trades that are resolved (win/loss), cancelled, or failed.
    Does NOT touch open trades.
    """
    try:
        result = await maintenance_service.cleanup_resolved_trades(
            older_than_days=older_than_days
        )
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            **result
        }
    except Exception as e:
        logger.error("Failed to cleanup resolved trades", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset")
async def reset_all_trades(
    confirm: bool = Query(
        default=False,
        description="Must be True to proceed"
    ),
    account_id: Optional[str] = Query(
        default=None,
        description="Only reset specific account"
    )
):
    """
    Reset/delete ALL trades.

    WARNING: This is destructive! Use with caution.
    Requires confirm=True to proceed.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="This will delete ALL trades! Set confirm=True to proceed."
        )

    try:
        result = await maintenance_service.delete_all_trades(
            account_id=account_id,
            confirm=True
        )
        return {
            "status": "success",
            "message": "All trades deleted" if not account_id else f"All trades for account {account_id} deleted",
            "timestamp": datetime.utcnow().isoformat(),
            **result
        }
    except Exception as e:
        logger.error("Failed to reset trades", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
