"""
Database Maintenance API Routes

Endpoints for cleaning up old trades and managing database health.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Literal, Optional
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from utils.utcnow import utcnow

from services.maintenance import maintenance_service
from models.database import (
    AutoTraderControl,
    AutoTraderPreflightRun,
    AutoTraderSnapshot,
    AutoTraderTrade,
    NewsArticleCache,
    NewsMarketWatcher,
    NewsTradeIntent,
    NewsWorkflowFinding,
    NewsWorkflowSnapshot,
    OpportunityEvent,
    OpportunityHistory,
    OpportunityLifetime,
    OpportunityState,
    ScannerRun,
    ScannerSnapshot,
    TradeSignal,
    TradeSignalSnapshot,
    TradeStatus,
    WeatherSnapshot,
    WeatherTradeIntent,
    get_db_session,
)
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
        description="Delete resolved trades older than this many days",
    )
    open_trade_expiry_days: int = Field(
        default=90,
        ge=1,
        le=365,
        description="Expire open trades older than this many days",
    )
    wallet_trade_days: int = Field(
        default=60,
        ge=1,
        le=365,
        description="Delete wallet trades older than this many days",
    )
    anomaly_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Delete resolved anomalies older than this many days",
    )


class DeleteTradesRequest(BaseModel):
    """Request for deleting trades"""

    older_than_days: Optional[int] = Field(
        default=None,
        ge=1,
        le=365,
        description="Delete trades older than this many days",
    )
    statuses: Optional[list[str]] = Field(
        default=None,
        description="Delete trades with these statuses (e.g., ['resolved_win', 'resolved_loss'])",
    )
    account_id: Optional[str] = Field(
        default=None, description="Only delete trades for this account"
    )
    delete_all: bool = Field(
        default=False, description="Delete ALL trades (dangerous!)"
    )
    confirm: bool = Field(
        default=False, description="Must be True to proceed with delete_all"
    )


FlushTarget = Literal["scanner", "weather", "news", "autotrader", "all"]

PROTECTED_DATASETS = (
    "auto_trader_trades (live/executed trade history)",
    "simulation_positions (position ledger)",
    "simulation_trades (trade history ledger)",
)


class FlushDataRequest(BaseModel):
    """Request for manual data flush operations in the database settings UI."""

    target: FlushTarget = Field(
        ...,
        description="Dataset to flush: scanner, weather, news, autotrader, or all",
    )
    confirm: bool = Field(
        default=False,
        description="Must be true to acknowledge destructive flush action",
    )


async def _delete_rows(session: AsyncSession, model) -> int:
    result = await session.execute(delete(model))
    return max(0, int(result.rowcount or 0))


async def _flush_scanner_data(session: AsyncSession) -> dict[str, int]:
    snapshot = (
        (
            await session.execute(
                select(ScannerSnapshot).where(ScannerSnapshot.id == "latest")
            )
        )
        .scalars()
        .first()
    )

    snapshot_opportunities = len(snapshot.opportunities_json or []) if snapshot else 0
    if snapshot is not None:
        snapshot.opportunities_json = []
        snapshot.market_history_json = {}
        snapshot.last_scan_at = None
        snapshot.current_activity = "Scanner snapshot cleared by manual maintenance flush."

    return {
        "scanner_snapshot_opportunities": snapshot_opportunities,
        "opportunity_events": await _delete_rows(session, OpportunityEvent),
        "opportunity_state": await _delete_rows(session, OpportunityState),
        "scanner_runs": await _delete_rows(session, ScannerRun),
        "opportunity_history": await _delete_rows(session, OpportunityHistory),
        "opportunity_lifetimes": await _delete_rows(session, OpportunityLifetime),
    }


async def _flush_news_data(session: AsyncSession) -> dict[str, int]:
    from services.news.feed_service import news_feed_service

    memory_cache_cleared = int(news_feed_service.clear() or 0)
    snapshot = (
        (
            await session.execute(
                select(NewsWorkflowSnapshot).where(NewsWorkflowSnapshot.id == "latest")
            )
        )
        .scalars()
        .first()
    )
    if snapshot is not None:
        snapshot.last_scan_at = None
        snapshot.next_scan_at = None
        snapshot.last_error = None
        snapshot.degraded_mode = False
        snapshot.budget_remaining_usd = None
        snapshot.stats_json = {}
        snapshot.current_activity = "News workflow snapshot cleared by manual maintenance flush."

    return {
        "news_memory_cache": memory_cache_cleared,
        "news_article_cache": await _delete_rows(session, NewsArticleCache),
        "news_market_watchers": await _delete_rows(session, NewsMarketWatcher),
        "news_workflow_findings": await _delete_rows(session, NewsWorkflowFinding),
        "news_trade_intents": await _delete_rows(session, NewsTradeIntent),
    }


async def _flush_weather_data(session: AsyncSession) -> dict[str, int]:
    snapshot = (
        (
            await session.execute(
                select(WeatherSnapshot).where(WeatherSnapshot.id == "latest")
            )
        )
        .scalars()
        .first()
    )

    snapshot_opportunities = len(snapshot.opportunities_json or []) if snapshot else 0
    if snapshot is not None:
        snapshot.last_scan_at = None
        snapshot.opportunities_json = []
        snapshot.stats_json = {}
        snapshot.current_activity = (
            "Weather workflow snapshot cleared by manual maintenance flush."
        )

    return {
        "weather_snapshot_opportunities": snapshot_opportunities,
        "weather_trade_intents": await _delete_rows(session, WeatherTradeIntent),
    }


async def _flush_autotrader_runtime_data(session: AsyncSession) -> dict[str, int]:
    signal_id_subquery = (
        select(AutoTraderTrade.signal_id)
        .where(AutoTraderTrade.signal_id.is_not(None))
        .distinct()
    )
    orphan_signal_delete = await session.execute(
        delete(TradeSignal).where(~TradeSignal.id.in_(signal_id_subquery))
    )
    orphan_signals_cleared = max(0, int(orphan_signal_delete.rowcount or 0))

    snapshot = (
        (
            await session.execute(
                select(AutoTraderSnapshot).where(AutoTraderSnapshot.id == "latest")
            )
        )
        .scalars()
        .first()
    )
    if snapshot is not None:
        snapshot.last_error = None
        snapshot.stats_json = {}
        snapshot.current_activity = (
            "Autotrader runtime caches cleared by manual maintenance flush."
        )

    control = (
        (
            await session.execute(
                select(AutoTraderControl).where(AutoTraderControl.id == "default")
            )
        )
        .scalars()
        .first()
    )
    if control is not None:
        control.requested_run_at = None

    return {
        "trade_signal_snapshots": await _delete_rows(session, TradeSignalSnapshot),
        "trade_signals_orphaned": orphan_signals_cleared,
        "autotrader_preflight_runs": await _delete_rows(session, AutoTraderPreflightRun),
    }


async def _run_flush(session: AsyncSession, target: FlushTarget) -> dict[str, dict[str, int]]:
    selected_targets: tuple[FlushTarget, ...]
    if target == "all":
        selected_targets = ("scanner", "weather", "news", "autotrader")
    else:
        selected_targets = (target,)

    results: dict[str, dict[str, int]] = {}
    for selected in selected_targets:
        if selected == "scanner":
            results[selected] = await _flush_scanner_data(session)
        elif selected == "weather":
            results[selected] = await _flush_weather_data(session)
        elif selected == "news":
            results[selected] = await _flush_news_data(session)
        elif selected == "autotrader":
            results[selected] = await _flush_autotrader_runtime_data(session)
    return results


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
            "timestamp": utcnow().isoformat(),
            "stats": stats,
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
            anomaly_days=request.anomaly_days,
        )
        return {
            "status": "success",
            "timestamp": utcnow().isoformat(),
            "results": results,
        }
    except Exception as e:
        logger.error("Cleanup failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/flush")
async def flush_data(
    request: FlushDataRequest,
    session: AsyncSession = Depends(get_db_session),
):
    """
    Flush runtime/cache datasets from Database Settings UI.

    Safety constraints:
    - Requires confirm=true.
    - Never deletes live/executed trade history or position ledgers.
    """
    if not request.confirm:
        raise HTTPException(
            status_code=400,
            detail="This operation is destructive. Set confirm=true to proceed.",
        )

    try:
        flushed = await _run_flush(session, request.target)
        await session.commit()
        logger.warning(
            "Manual maintenance flush executed",
            target=request.target,
            datasets=list(flushed.keys()),
        )
        return {
            "status": "success",
            "target": request.target,
            "timestamp": utcnow().isoformat(),
            "flushed": flushed,
            "protected_datasets": list(PROTECTED_DATASETS),
            "message": "Flush complete. Live positions and trade history were preserved.",
        }
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        logger.error("Manual maintenance flush failed", error=str(e), target=request.target)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/expire-old-trades")
async def expire_old_trades(
    older_than_days: int = Query(
        default=90,
        ge=1,
        le=365,
        description="Expire open trades older than this many days",
    ),
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
            "timestamp": utcnow().isoformat(),
            **result,
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
        if (
            not request.older_than_days
            and not request.statuses
            and not request.delete_all
        ):
            raise HTTPException(
                status_code=400,
                detail="Must specify older_than_days, statuses, or delete_all",
            )

        results = {}

        if request.delete_all:
            if not request.confirm:
                raise HTTPException(
                    status_code=400, detail="Must set confirm=True to delete all trades"
                )
            results = await maintenance_service.delete_all_trades(
                account_id=request.account_id, confirm=True
            )
        elif request.statuses:
            # Convert status strings to enums
            try:
                status_enums = [TradeStatus(s) for s in request.statuses]
            except ValueError:
                valid_statuses = [s.value for s in TradeStatus]
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status. Valid statuses: {valid_statuses}",
                )

            results = await maintenance_service.delete_trades_by_status(
                statuses=status_enums, account_id=request.account_id
            )
        elif request.older_than_days:
            results = await maintenance_service.cleanup_resolved_trades(
                older_than_days=request.older_than_days, account_id=request.account_id
            )

        return {
            "status": "success",
            "timestamp": utcnow().isoformat(),
            **results,
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
        description="Delete wallet trades older than this many days",
    ),
    wallet_address: Optional[str] = Query(
        default=None, description="Only delete for specific wallet"
    ),
):
    """
    Delete old wallet trades.

    These are trades tracked from monitored wallets.
    """
    try:
        result = await maintenance_service.cleanup_wallet_trades(
            older_than_days=older_than_days, wallet_address=wallet_address
        )
        return {
            "status": "success",
            "timestamp": utcnow().isoformat(),
            **result,
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
        description="Delete anomalies older than this many days",
    ),
    resolved_only: bool = Query(
        default=True, description="Only delete resolved anomalies"
    ),
):
    """
    Delete old anomaly records.
    """
    try:
        result = await maintenance_service.cleanup_anomalies(
            older_than_days=older_than_days, resolved_only=resolved_only
        )
        return {
            "status": "success",
            "timestamp": utcnow().isoformat(),
            **result,
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
        description="Delete resolved trades older than this many days",
    ),
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
            "timestamp": utcnow().isoformat(),
            **result,
        }
    except Exception as e:
        logger.error("Failed to cleanup resolved trades", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset")
async def reset_all_trades(
    confirm: bool = Query(default=False, description="Must be True to proceed"),
    account_id: Optional[str] = Query(
        default=None, description="Only reset specific account"
    ),
):
    """
    Reset/delete ALL trades.

    WARNING: This is destructive! Use with caution.
    Requires confirm=True to proceed.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="This will delete ALL trades! Set confirm=True to proceed.",
        )

    try:
        result = await maintenance_service.delete_all_trades(
            account_id=account_id, confirm=True
        )
        return {
            "status": "success",
            "message": "All trades deleted"
            if not account_id
            else f"All trades for account {account_id} deleted",
            "timestamp": utcnow().isoformat(),
            **result,
        }
    except Exception as e:
        logger.error("Failed to reset trades", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
