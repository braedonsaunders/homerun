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
from models.database import async_engine
from models.database import (
    ExecutionSession,
    ExecutionSessionEvent,
    ExecutionSessionLeg,
    ExecutionSessionOrder,
    NewsArticleCache,
    NewsMarketWatcher,
    NewsTradeIntent,
    NewsWorkflowFinding,
    NewsWorkflowSnapshot,
    OpportunityHistory,
    OpportunityLifetime,
    ScannerMarketHistory,
    OpportunityState,
    ScannerRun,
    ScannerSnapshot,
    TradeSignal,
    TradeStatus,
    TraderDecision,
    TraderDecisionCheck,
    TraderEvent,
    TraderOrder,
    TraderOrchestratorControl,
    TraderOrchestratorSnapshot,
    TraderSignalConsumption,
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
        description="Delete terminal trades (resolved/closed/cancelled/failed) older than this many days",
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
    trade_signal_emission_days: int = Field(
        default=21,
        ge=1,
        le=3650,
        description="Delete trade signal emission rows older than this many days",
    )
    trade_signal_update_days: int = Field(
        default=3,
        ge=0,
        le=3650,
        description="Delete upsert_update emission rows older than this many days (0 disables)",
    )
    wallet_activity_rollup_days: int = Field(
        default=60,
        ge=45,
        le=3650,
        description="Delete wallet activity rollup rows older than this many days",
    )
    wallet_activity_dedupe_enabled: bool = Field(
        default=True,
        description="Run duplicate cleanup pass for wallet activity rollups",
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
        description="Delete trades with these statuses (e.g., ['closed_win', 'resolved_loss'])",
    )
    account_id: Optional[str] = Field(default=None, description="Only delete trades for this account")
    delete_all: bool = Field(default=False, description="Delete ALL trades (dangerous!)")
    confirm: bool = Field(default=False, description="Must be True to proceed with delete_all")


FlushTarget = Literal["scanner", "weather", "news", "trader_orchestrator", "all"]

PROTECTED_DATASETS = (
    "trader_orders (live/executed order history)",
    "simulation_positions (position ledger)",
    "simulation_trades (trade history ledger)",
)


class FlushDataRequest(BaseModel):
    """Request for manual data flush operations in the database settings UI."""

    target: FlushTarget = Field(
        ...,
        description="Dataset to flush: scanner, weather, news, trader_orchestrator, or all",
    )
    confirm: bool = Field(
        default=False,
        description="Must be true to acknowledge destructive flush action",
    )


async def _delete_rows(session: AsyncSession, model) -> int:
    result = await session.execute(delete(model))
    return max(0, int(result.rowcount or 0))


async def _flush_scanner_data(session: AsyncSession) -> dict[str, int]:
    snapshot = (await session.execute(select(ScannerSnapshot).where(ScannerSnapshot.id == "latest"))).scalars().first()

    snapshot_opportunities = len(snapshot.opportunities_json or []) if snapshot else 0
    if snapshot is not None:
        snapshot.opportunities_json = []
        snapshot.last_scan_at = None
        snapshot.current_activity = "Scanner snapshot cleared by manual maintenance flush."

    return {
        "scanner_snapshot_opportunities": snapshot_opportunities,
        "scanner_market_history_rows": await _delete_rows(session, ScannerMarketHistory),
        "opportunity_state": await _delete_rows(session, OpportunityState),
        "scanner_runs": await _delete_rows(session, ScannerRun),
        "opportunity_history": await _delete_rows(session, OpportunityHistory),
        "opportunity_lifetimes": await _delete_rows(session, OpportunityLifetime),
    }


async def _flush_news_data(session: AsyncSession) -> dict[str, int]:
    from services.news.feed_service import news_feed_service

    memory_cache_cleared = int(news_feed_service.clear() or 0)
    snapshot = (
        (await session.execute(select(NewsWorkflowSnapshot).where(NewsWorkflowSnapshot.id == "latest")))
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
    snapshot = (await session.execute(select(WeatherSnapshot).where(WeatherSnapshot.id == "latest"))).scalars().first()

    snapshot_opportunities = len(snapshot.opportunities_json or []) if snapshot else 0
    if snapshot is not None:
        snapshot.last_scan_at = None
        snapshot.opportunities_json = []
        snapshot.stats_json = {}
        snapshot.current_activity = "Weather workflow snapshot cleared by manual maintenance flush."

    return {
        "weather_snapshot_opportunities": snapshot_opportunities,
        "weather_trade_intents": await _delete_rows(session, WeatherTradeIntent),
    }


async def _flush_trader_orchestrator_runtime_data(session: AsyncSession) -> dict[str, int]:
    signal_id_subquery = select(TraderOrder.signal_id).where(TraderOrder.signal_id.is_not(None)).distinct()
    orphan_signal_delete = await session.execute(delete(TradeSignal).where(~TradeSignal.id.in_(signal_id_subquery)))
    orphan_signals_cleared = max(0, int(orphan_signal_delete.rowcount or 0))

    snapshot = (
        (await session.execute(select(TraderOrchestratorSnapshot).where(TraderOrchestratorSnapshot.id == "latest")))
        .scalars()
        .first()
    )
    if snapshot is not None:
        snapshot.last_error = None
        snapshot.stats_json = {}
        snapshot.current_activity = "Trader orchestrator runtime caches cleared by manual maintenance flush."

    control = (
        (await session.execute(select(TraderOrchestratorControl).where(TraderOrchestratorControl.id == "default")))
        .scalars()
        .first()
    )
    if control is not None:
        control.requested_run_at = None

    return {
        "execution_session_events": await _delete_rows(session, ExecutionSessionEvent),
        "execution_session_orders": await _delete_rows(session, ExecutionSessionOrder),
        "execution_session_legs": await _delete_rows(session, ExecutionSessionLeg),
        "execution_sessions": await _delete_rows(session, ExecutionSession),
        "trade_signals_orphaned": orphan_signals_cleared,
        "trader_signal_consumption": await _delete_rows(session, TraderSignalConsumption),
        "trader_decision_checks": await _delete_rows(session, TraderDecisionCheck),
        "trader_decisions": await _delete_rows(session, TraderDecision),
        "trader_events": await _delete_rows(session, TraderEvent),
    }


async def _run_flush(session: AsyncSession, target: FlushTarget) -> dict[str, dict[str, int]]:
    selected_targets: tuple[FlushTarget, ...]
    if target == "all":
        selected_targets = ("scanner", "weather", "news", "trader_orchestrator")
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
        elif selected == "trader_orchestrator":
            results[selected] = await _flush_trader_orchestrator_runtime_data(session)
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
    5. Delete stale LLM usage logs (from settings retention policy)
    6. Prune noisy + aged trade signal emissions
    7. Dedupe and age-prune wallet activity rollups
    8. Run market metadata hygiene when enabled

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
            trade_signal_emission_days=request.trade_signal_emission_days,
            trade_signal_update_days=request.trade_signal_update_days,
            wallet_activity_rollup_days=request.wallet_activity_rollup_days,
            wallet_activity_dedupe_enabled=request.wallet_activity_dedupe_enabled,
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
        result = await maintenance_service.expire_old_open_trades(older_than_days=older_than_days)
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
    - older_than_days: Delete terminal trades older than X days
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
                detail="Must specify older_than_days, statuses, or delete_all",
            )

        results = {}

        if request.delete_all:
            if not request.confirm:
                raise HTTPException(status_code=400, detail="Must set confirm=True to delete all trades")
            results = await maintenance_service.delete_all_trades(account_id=request.account_id, confirm=True)
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
    wallet_address: Optional[str] = Query(default=None, description="Only delete for specific wallet"),
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


@router.delete("/trade-signal-emissions")
async def delete_trade_signal_emissions(
    older_than_days: int = Query(
        default=21,
        ge=1,
        le=3650,
        description="Delete trade signal emission rows older than this many days",
    ),
    source: Optional[str] = Query(
        default=None,
        description="Optional source filter (scanner/news/weather/crypto/traders/events)",
    ),
    event_type: Optional[str] = Query(
        default=None,
        description="Optional event type filter (upsert_insert/upsert_update/status_transition)",
    ),
):
    """
    Delete old trade signal emission rows.

    Supports optional source/event-type scoped cleanup for selective retention.
    """
    try:
        result = await maintenance_service.cleanup_trade_signal_emissions(
            older_than_days=older_than_days,
            source=source,
            event_type=event_type,
        )
        return {
            "status": "success",
            "timestamp": utcnow().isoformat(),
            **result,
        }
    except Exception as e:
        logger.error("Failed to delete trade signal emissions", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/trade-signal-emissions/updates")
async def delete_trade_signal_update_emissions(
    older_than_days: int = Query(
        default=3,
        ge=0,
        le=3650,
        description="Delete upsert_update trade signal emissions older than this many days (0 disables)",
    ),
    source: Optional[str] = Query(
        default=None,
        description="Optional source filter (scanner/news/weather/crypto/traders/events)",
    ),
):
    """
    Delete noisy upsert_update emission rows while retaining inserts/transitions.
    """
    try:
        result = await maintenance_service.cleanup_trade_signal_update_emissions(
            older_than_days=older_than_days,
            source=source,
        )
        return {
            "status": "success",
            "timestamp": utcnow().isoformat(),
            **result,
        }
    except Exception as e:
        logger.error("Failed to delete trade signal update emissions", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/wallet-activity-rollups")
async def delete_wallet_activity_rollups(
    older_than_days: int = Query(
        default=60,
        ge=45,
        le=3650,
        description="Delete wallet activity rollup rows older than this many days",
    ),
    source: Optional[str] = Query(default=None, description="Optional source filter"),
    wallet_address: Optional[str] = Query(default=None, description="Optional wallet filter"),
):
    """
    Delete old wallet activity rollup rows.
    """
    try:
        result = await maintenance_service.cleanup_wallet_activity_rollups(
            older_than_days=older_than_days,
            source=source,
            wallet_address=wallet_address,
        )
        return {
            "status": "success",
            "timestamp": utcnow().isoformat(),
            **result,
        }
    except Exception as e:
        logger.error("Failed to delete wallet activity rollups", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/wallet-activity-rollups/dedupe")
async def dedupe_wallet_activity_rollups(
    source: Optional[str] = Query(default=None, description="Optional source filter"),
    older_than_minutes: int = Query(
        default=10,
        ge=0,
        le=1440,
        description="Skip rows newer than this many minutes to avoid racing ingestion",
    ),
    batch_limit: int = Query(
        default=100000,
        ge=1,
        le=1000000,
        description="Maximum duplicate rows to delete per batch",
    ),
    max_batches: int = Query(
        default=50,
        ge=1,
        le=1000,
        description="Maximum batches per dedupe operation",
    ),
):
    """
    Delete duplicate wallet activity rollups while retaining one canonical row.
    """
    try:
        result = await maintenance_service.cleanup_wallet_activity_rollup_duplicates(
            source=source,
            older_than_minutes=older_than_minutes,
            batch_limit=batch_limit,
            max_batches=max_batches,
        )
        return {
            "status": "success",
            "timestamp": utcnow().isoformat(),
            **result,
        }
    except Exception as e:
        logger.error("Failed to dedupe wallet activity rollups", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/anomalies")
async def delete_anomalies(
    older_than_days: int = Query(
        default=30,
        ge=1,
        le=365,
        description="Delete anomalies older than this many days",
    ),
    resolved_only: bool = Query(default=True, description="Only delete resolved anomalies"),
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
        description="Delete terminal trades older than this many days",
    ),
):
    """
    Quick cleanup of terminal trades only.

    Deletes trades that are resolved (win/loss), closed (win/loss), cancelled, or failed.
    Does NOT touch open trades.
    """
    try:
        result = await maintenance_service.cleanup_resolved_trades(older_than_days=older_than_days)
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
    account_id: Optional[str] = Query(default=None, description="Only reset specific account"),
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
        result = await maintenance_service.delete_all_trades(account_id=account_id, confirm=True)
        return {
            "status": "success",
            "message": "All trades deleted" if not account_id else f"All trades for account {account_id} deleted",
            "timestamp": utcnow().isoformat(),
            **result,
        }
    except Exception as e:
        logger.error("Failed to reset trades", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vacuum")
async def vacuum_analyze(
    full: bool = Query(
        default=False,
        description="Run VACUUM FULL (rewrites tables, reclaims disk, takes exclusive lock). Default is regular VACUUM.",
    ),
    tables: Optional[list[str]] = Query(
        default=None,
        description=(
            "Optional list of specific table names to vacuum. When omitted, "
            "the built-in high-churn list is used. Useful for targeted ops "
            "on a single large table (e.g. ?tables=trader_events) without "
            "locking the whole sweep on one slow operation."
        ),
    ),
):
    """
    Run VACUUM ANALYZE on high-churn tables.

    Regular VACUUM (default) reclaims dead tuples without locking tables.
    VACUUM FULL rewrites the entire table to reclaim maximum disk space
    but takes an exclusive lock — avoid during active trading.
    """
    try:
        result = await maintenance_service.vacuum_analyze(full=full, tables=tables)
        return {
            "status": "success",
            "timestamp": utcnow().isoformat(),
            **result,
        }
    except Exception as e:
        logger.error("VACUUM ANALYZE failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/table-stats")
async def table_stats(
    tables: Optional[list[str]] = Query(
        default=None,
        description=(
            "Optional list of table names. When omitted, returns stats for "
            "the orchestrator commit-path hot tables (trade_signals + family)."
        ),
    ),
):
    """Return pg_stat_user_tables snapshot for the requested tables.

    Surfaces the same data as the SLOW COMMIT DIAGNOSTIC log line
    (live/dead tuples, dead %, vacuum/analyze ages, WAL counters,
    HOT update %) for ad-hoc verification — e.g. "did the VACUUM
    FULL I just ran actually take effect?" without having to grep
    worker logs for the next slow-commit event.
    """
    default_tables = [
        "trade_signals",
        "trade_signal_emissions",
        "trader_decisions",
        "trader_decision_checks",
        "trader_events",
    ]
    targets = [str(t).strip() for t in (tables or default_tables) if str(t).strip()]
    if not targets:
        return {"status": "skipped", "reason": "no_tables_requested"}

    try:
        import asyncpg
        from config import settings as _settings

        dsn = str(_settings.DATABASE_URL).replace("+asyncpg", "")
        conn = await asyncpg.connect(dsn=dsn, timeout=5)
    except Exception as exc:
        logger.error("table-stats: probe connect failed", error=str(exc))
        raise HTTPException(status_code=503, detail=f"db connect failed: {exc}")

    try:
        table_rows = await conn.fetch(
            """
            SELECT relname,
                   n_live_tup,
                   n_dead_tup,
                   n_mod_since_analyze,
                   EXTRACT(EPOCH FROM now() - last_vacuum)::int AS last_vacuum_age_s,
                   EXTRACT(EPOCH FROM now() - last_autovacuum)::int AS last_autovacuum_age_s,
                   EXTRACT(EPOCH FROM now() - last_analyze)::int AS last_analyze_age_s,
                   EXTRACT(EPOCH FROM now() - last_autoanalyze)::int AS last_autoanalyze_age_s,
                   n_tup_ins,
                   n_tup_upd,
                   n_tup_del,
                   n_tup_hot_upd
            FROM pg_stat_user_tables
            WHERE relname = ANY($1::text[])
            """,
            targets,
        )
    finally:
        try:
            await conn.close()
        except Exception:
            pass

    payload: dict[str, dict] = {}
    for r in table_rows:
        live = int(r["n_live_tup"] or 0)
        dead = int(r["n_dead_tup"] or 0)
        upd = int(r["n_tup_upd"] or 0)
        hot_upd = int(r["n_tup_hot_upd"] or 0)
        payload[r["relname"]] = {
            "live": live,
            "dead": dead,
            "dead_pct": round(100.0 * dead / max(1, live + dead), 2),
            "mod_since_analyze": int(r["n_mod_since_analyze"] or 0),
            "vacuum_age_s": int(r["last_vacuum_age_s"] or 0),
            "autovacuum_age_s": int(r["last_autovacuum_age_s"] or 0),
            "analyze_age_s": int(r["last_analyze_age_s"] or 0),
            "autoanalyze_age_s": int(r["last_autoanalyze_age_s"] or 0),
            "ins": int(r["n_tup_ins"] or 0),
            "upd": upd,
            "del": int(r["n_tup_del"] or 0),
            "hot_upd": hot_upd,
            "hot_pct": round(100.0 * hot_upd / max(1, upd), 2),
        }
    return {
        "status": "ok",
        "timestamp": utcnow().isoformat(),
        "tables": payload,
    }


@router.post("/reindex")
async def reindex_tables():
    """
    Rebuild indexes on high-churn tables to reclaim bloated index space.

    Regular VACUUM does not fix index bloat. REINDEX rebuilds each index
    from scratch, reclaiming disk space. Takes a short lock on each index
    while rebuilding but is much faster than VACUUM FULL.
    """
    try:
        result = await maintenance_service.reindex_tables()
        return {
            "status": "success",
            "timestamp": utcnow().isoformat(),
            **result,
        }
    except Exception as e:
        logger.error("REINDEX failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pool-status")
async def pool_status():
    """Return connection pool diagnostics."""
    pool = async_engine.pool
    return {
        "pool_size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "invalid": pool.status(),
        "timeout": pool.timeout(),
    }


class RepairPnlRequest(BaseModel):
    """Request to repair physically-impossible realized PnL values."""

    dry_run: bool = Field(default=True)
    window_days: int = Field(default=120, ge=1, le=3650)


@router.post("/repair-implausible-pnl")
async def repair_implausible_pnl(
    req: RepairPnlRequest,
    session: AsyncSession = Depends(get_db_session),
):
    """Recompute ``actual_profit`` for terminal live orders whose stored value
    is physically impossible given the entry cost basis.

    A binary position bought for ``cost`` over ``size`` shares can net at most
    ``size*$1 - cost`` (every share wins) and at least ``-cost`` (total loss).
    Legacy rows from older resolution-verifier code recorded
    ``actual_profit == cost`` (the notional) instead of net ``payout - cost``,
    inflating every surface that sums realized PnL (incl. ``daily_pnl``, which
    the risk guardian reads). This deterministically corrects only rows that
    violate those bounds:

    * ``resolved_win``  -> ``size - cost``  (held to a $1 settlement; exact)
    * ``resolved_loss`` -> ``-cost``        (settled to $0; exact)
    * ``closed_*``      -> clamp the stored value into ``[-cost, size-cost]``

    Correct rows are left untouched. Defaults to a dry run.
    """
    from datetime import timedelta

    from services.polymarket_trade_verifier import (
        _entry_cost_basis,
        _entry_fill_size,
    )
    from utils.pnl import (
        LOSS_STATES,
        WIN_STATES,
        canonical_terminal_net_pnl,
        is_implausible_pnl,
    )

    cutoff = (utcnow() - timedelta(days=req.window_days)).replace(tzinfo=None)
    rows = (
        await session.execute(
            select(TraderOrder).where(
                TraderOrder.mode == "live",
                TraderOrder.status.in_(WIN_STATES | LOSS_STATES),
                TraderOrder.actual_profit.is_not(None),
                TraderOrder.updated_at >= cutoff,
            )
        )
    ).scalars().all()

    examined = corrected = 0
    before_sum = after_sum = 0.0
    samples: list[dict] = []
    for row in rows:
        examined += 1
        cost = _entry_cost_basis(row)
        size = _entry_fill_size(row)
        if cost <= 0.0 or size <= 0.0:
            continue
        ap = float(row.actual_profit)
        if not is_implausible_pnl(ap, cost, size):
            continue  # within physically-possible bounds -> trust it
        status = str(row.status or "")
        fixed = canonical_terminal_net_pnl(status, cost, size, stored_pnl=ap)
        if fixed is None:
            continue
        before_sum += ap
        after_sum += fixed
        corrected += 1
        if len(samples) < 12:
            samples.append({
                "id": row.id,
                "status": status,
                "cost": round(cost, 4),
                "size": round(size, 4),
                "old_actual_profit": round(ap, 4),
                "new_actual_profit": round(fixed, 4),
            })
        if not req.dry_run:
            row.actual_profit = float(fixed)
            row.updated_at = utcnow()

    if not req.dry_run and corrected:
        await session.commit()

    return {
        "dry_run": req.dry_run,
        "examined": examined,
        "corrected": corrected,
        "pnl_before_sum": round(before_sum, 2),
        "pnl_after_sum": round(after_sum, 2),
        "delta": round(after_sum - before_sum, 2),
        "samples": samples,
    }
