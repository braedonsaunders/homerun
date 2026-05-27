"""
Database Maintenance Service

Handles cleanup of old trades, expiration of stale data, and database maintenance.
"""

import asyncio
import time
from datetime import timedelta
from typing import Optional
from utils.utcnow import utcnow
from sqlalchemy import select, delete, func, and_, text

from models.database import (
    SimulationTrade,
    SimulationPosition,
    WalletTrade,
    WalletActivityRollup,
    OpportunityHistory,
    DetectedAnomaly,
    LLMUsageLog,
    TradeSignal,
    TradeSignalEmission,
    TradeStatus,
    AsyncSessionLocal,
    AppSettings,
    async_engine,
)
from services.market_cache import market_cache_service
from utils.logger import get_logger

logger = get_logger("maintenance")


class MaintenanceService:
    """Database maintenance and cleanup service"""

    # Default age thresholds (in days)
    DEFAULT_DATABASE_BACKUP_RETENTION_DAYS = 14
    DEFAULT_RESOLVED_TRADE_AGE = 30  # Delete resolved trades older than 30 days
    DEFAULT_OPEN_TRADE_EXPIRY = 90  # Mark open trades as expired after 90 days
    DEFAULT_WALLET_TRADE_AGE = 60  # Delete wallet trades older than 60 days
    DEFAULT_ANOMALY_AGE = 30  # Delete resolved anomalies older than 30 days
    DEFAULT_LLM_USAGE_RETENTION_DAYS = 30  # Delete raw LLM usage logs older than 30 days
    DEFAULT_TRADE_SIGNAL_EMISSION_AGE = 3  # Delete trade signal emission rows older than 3 days
    DEFAULT_TRADE_SIGNAL_UPDATE_AGE = 3  # Delete upsert_update emissions older than 3 days
    DEFAULT_TRADE_SIGNAL_AGE = 30  # Delete trade signal rows older than 30 days
    DEFAULT_WALLET_ACTIVITY_ROLLUP_AGE = 60  # Delete wallet activity rollups older than 60 days
    # 2026-05-26: retention for previously-unbounded high-volume tables that
    # filled the disk (see migration 202605260001).  0 disables a sweep.
    DEFAULT_MARKET_MICROSTRUCTURE_AGE = 7
    DEFAULT_BOOK_DELTA_EVENTS_AGE = 7
    DEFAULT_WALLET_MONITOR_EVENTS_AGE = 14
    DEFAULT_TRADER_DECISION_CHECKS_AGE = 14
    DEFAULT_TRADER_DECISIONS_AGE = 30
    DEFAULT_OPPORTUNITY_HISTORY_AGE = 30

    async def _market_cache_hygiene_settings(self) -> dict:
        config = {
            "enabled": True,
            "interval_hours": 6,
            "retention_days": 120,
            "reference_lookback_days": 45,
            "weak_entry_grace_days": 7,
            "max_entries_per_slug": 3,
        }
        try:
            async with AsyncSessionLocal() as session:
                row = (
                    await session.execute(select(AppSettings).where(AppSettings.id == "default"))
                ).scalar_one_or_none()
                if not row:
                    return config
                config["enabled"] = bool(
                    row.market_cache_hygiene_enabled if row.market_cache_hygiene_enabled is not None else True
                )
                config["interval_hours"] = int(row.market_cache_hygiene_interval_hours or 6)
                config["retention_days"] = int(row.market_cache_retention_days or 120)
                config["reference_lookback_days"] = int(row.market_cache_reference_lookback_days or 45)
                config["weak_entry_grace_days"] = int(row.market_cache_weak_entry_grace_days or 7)
                config["max_entries_per_slug"] = int(row.market_cache_max_entries_per_slug or 3)
        except Exception as e:
            logger.warning("Failed to read market cache hygiene settings", error=str(e))
        return config

    async def _llm_usage_retention_days_setting(self) -> int:
        retention_days = self.DEFAULT_LLM_USAGE_RETENTION_DAYS
        try:
            async with AsyncSessionLocal() as session:
                row = (
                    await session.execute(select(AppSettings).where(AppSettings.id == "default"))
                ).scalar_one_or_none()
                if row and row.llm_usage_retention_days is not None:
                    retention_days = max(0, int(row.llm_usage_retention_days))
        except Exception as e:
            logger.warning("Failed to read LLM usage retention setting", error=str(e))
        return retention_days

    async def _high_volume_cleanup_settings(self) -> dict:
        config = {
            "trade_signal_emission_days": self.DEFAULT_TRADE_SIGNAL_EMISSION_AGE,
            "trade_signal_update_days": self.DEFAULT_TRADE_SIGNAL_UPDATE_AGE,
            "trade_signal_days": self.DEFAULT_TRADE_SIGNAL_AGE,
            "wallet_activity_rollup_days": self.DEFAULT_WALLET_ACTIVITY_ROLLUP_AGE,
            "wallet_activity_dedupe_enabled": True,
        }
        try:
            async with AsyncSessionLocal() as session:
                row = (
                    await session.execute(select(AppSettings).where(AppSettings.id == "default"))
                ).scalar_one_or_none()
                if row is None:
                    return config
                if row.cleanup_trade_signal_emission_days is not None:
                    config["trade_signal_emission_days"] = max(1, int(row.cleanup_trade_signal_emission_days))
                if row.cleanup_trade_signal_update_days is not None:
                    config["trade_signal_update_days"] = max(0, int(row.cleanup_trade_signal_update_days))
                if row.cleanup_trade_signal_days is not None:
                    config["trade_signal_days"] = max(0, int(row.cleanup_trade_signal_days))
                if row.cleanup_wallet_activity_rollup_days is not None:
                    config["wallet_activity_rollup_days"] = max(45, int(row.cleanup_wallet_activity_rollup_days))
                if row.cleanup_wallet_activity_dedupe_enabled is not None:
                    config["wallet_activity_dedupe_enabled"] = bool(row.cleanup_wallet_activity_dedupe_enabled)
        except Exception as e:
            logger.warning("Failed to read high-volume cleanup settings", error=str(e))
        return config

    async def cleanup_database_backups(self, older_than_days: int = DEFAULT_DATABASE_BACKUP_RETENTION_DAYS) -> dict:
        """
        Delete backup files older than the configured retention window.

        Args:
            older_than_days: Delete files older than this many days (0 disables).

        Returns:
            Dict with deletion count and retention metadata.
        """
        return {
            "status": "skipped",
            "reason": "database_backups_managed_externally",
            "older_than_days": int(older_than_days),
            "retention_days": int(older_than_days),
        }

    async def get_database_stats(self) -> dict:
        """Get statistics about database contents"""
        async with AsyncSessionLocal() as session:
            db_size_bytes: int | None = None
            total_rows: int | None = None
            estimated_total_rows: int | None = None
            table_bloat: list[dict] | None = None
            bind = session.get_bind()
            if bind is not None and bind.dialect.name == "postgresql":
                db_size_result = await session.execute(select(func.pg_database_size(func.current_database())))
                db_size_value = db_size_result.scalar()
                db_size_bytes = int(db_size_value) if db_size_value is not None else 0

                estimated_total_rows_result = await session.execute(
                    text("SELECT COALESCE(SUM(n_live_tup)::bigint, 0) FROM pg_stat_user_tables")
                )
                total_rows_value = estimated_total_rows_result.scalar()
                estimated_total_rows = int(total_rows_value) if total_rows_value is not None else 0

                exact_total_rows_result = await session.execute(
                    text(
                        """
SELECT COALESCE(
    SUM(
        (
            xpath(
                '/row/c/text()',
                query_to_xml(
                    format('SELECT count(*) AS c FROM %I.%I', schemaname, tablename),
                    true,
                    true,
                    ''
                )
            )
        )[1]::text::bigint
    ),
    0
) AS total_rows
FROM pg_tables
WHERE schemaname = 'public'
"""
                    )
                )
                total_rows_value = exact_total_rows_result.scalar()
                total_rows = int(total_rows_value) if total_rows_value is not None else 0

                # Dead tuple / bloat stats per table
                bloat_result = await session.execute(
                    text(
                        """
SELECT
    relname AS table_name,
    n_live_tup::bigint AS live_tuples,
    n_dead_tup::bigint AS dead_tuples,
    CASE WHEN n_live_tup > 0
         THEN round(100.0 * n_dead_tup / n_live_tup, 1)
         ELSE 0 END AS dead_pct,
    pg_total_relation_size(relid) AS total_bytes,
    pg_table_size(relid) AS table_bytes,
    pg_indexes_size(relid) AS index_bytes,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY n_dead_tup DESC
LIMIT 20
"""
                    )
                )
                table_bloat = [
                    {
                        "table": row.table_name,
                        "live_tuples": row.live_tuples,
                        "dead_tuples": row.dead_tuples,
                        "dead_pct": float(row.dead_pct),
                        "total_bytes": row.total_bytes,
                        "table_bytes": row.table_bytes,
                        "index_bytes": row.index_bytes,
                        "last_vacuum": row.last_vacuum.isoformat() if row.last_vacuum else None,
                        "last_autovacuum": row.last_autovacuum.isoformat() if row.last_autovacuum else None,
                        "last_analyze": row.last_analyze.isoformat() if row.last_analyze else None,
                        "last_autoanalyze": row.last_autoanalyze.isoformat() if row.last_autoanalyze else None,
                    }
                    for row in bloat_result
                ]

            # Count simulation trades by status
            trade_counts = {}
            for status in TradeStatus:
                result = await session.execute(
                    select(func.count(SimulationTrade.id)).where(SimulationTrade.status == status)
                )
                trade_counts[status.value] = result.scalar() or 0

            # Count total simulation trades
            total_trades_result = await session.execute(select(func.count(SimulationTrade.id)))
            total_trades = int(total_trades_result.scalar() or 0)

            # Count simulation positions
            total_positions_result = await session.execute(select(func.count(SimulationPosition.id)))
            total_positions = int(total_positions_result.scalar() or 0)
            open_positions_result = await session.execute(
                select(func.count(SimulationPosition.id)).where(SimulationPosition.status == TradeStatus.OPEN)
            )
            open_positions = int(open_positions_result.scalar() or 0)

            # Count wallet trades
            wallet_trades_result = await session.execute(select(func.count(WalletTrade.id)))
            wallet_trades = int(wallet_trades_result.scalar() or 0)
            wallet_activity_rollups_result = await session.execute(select(func.count(WalletActivityRollup.id)))
            wallet_activity_rollups = int(wallet_activity_rollups_result.scalar() or 0)
            trade_signal_emissions_result = await session.execute(select(func.count(TradeSignalEmission.id)))
            trade_signal_emissions = int(trade_signal_emissions_result.scalar() or 0)

            # Count opportunity history
            opportunities_result = await session.execute(select(func.count(OpportunityHistory.id)))
            opportunities = int(opportunities_result.scalar() or 0)

            # Count anomalies
            anomalies_result = await session.execute(select(func.count(DetectedAnomaly.id)))
            anomalies = int(anomalies_result.scalar() or 0)
            resolved_anomalies_result = await session.execute(
                select(func.count(DetectedAnomaly.id)).where(DetectedAnomaly.is_resolved)
            )
            resolved_anomalies = int(resolved_anomalies_result.scalar() or 0)

            # Get oldest and newest trade dates
            oldest_trade = await session.execute(select(func.min(SimulationTrade.executed_at)))
            newest_trade = await session.execute(select(func.max(SimulationTrade.executed_at)))
            oldest_trade_at = oldest_trade.scalar()
            newest_trade_at = newest_trade.scalar()

            return {
                "db_size_bytes": db_size_bytes,
                "total_rows": total_rows,
                "estimated_total_rows": estimated_total_rows,
                "simulation_trades": {
                    "total": total_trades,
                    "by_status": trade_counts,
                },
                "simulation_positions": {
                    "total": total_positions,
                    "open": open_positions,
                },
                "wallet_trades": wallet_trades,
                "wallet_activity_rollups": wallet_activity_rollups,
                "trade_signal_emissions": trade_signal_emissions,
                "opportunity_history": opportunities,
                "anomalies": {
                    "total": anomalies,
                    "resolved": resolved_anomalies,
                },
                "date_range": {
                    "oldest_trade": oldest_trade_at.isoformat() if oldest_trade_at else None,
                    "newest_trade": newest_trade_at.isoformat() if newest_trade_at else None,
                },
                "table_bloat": table_bloat,
            }

    async def cleanup_resolved_trades(
        self,
        older_than_days: int = DEFAULT_RESOLVED_TRADE_AGE,
        account_id: Optional[str] = None,
    ) -> dict:
        """
        Delete resolved trades older than specified days.

        Args:
            older_than_days: Delete trades resolved more than this many days ago
            account_id: Optional - only delete for specific account

        Returns:
            Dict with deletion counts
        """
        cutoff_date = utcnow() - timedelta(days=older_than_days)

        async with AsyncSessionLocal() as session:
            # Build conditions
            conditions = [
                SimulationTrade.resolved_at < cutoff_date,
                SimulationTrade.status.in_(
                    [
                        TradeStatus.CLOSED_WIN,
                        TradeStatus.CLOSED_LOSS,
                        TradeStatus.RESOLVED_WIN,
                        TradeStatus.RESOLVED_LOSS,
                        TradeStatus.CANCELLED,
                        TradeStatus.FAILED,
                    ]
                ),
            ]

            if account_id:
                conditions.append(SimulationTrade.account_id == account_id)

            # Get trade IDs to delete (for position cleanup)
            trades_to_delete = await session.execute(
                select(SimulationTrade.id, SimulationTrade.opportunity_id).where(and_(*conditions))
            )
            trade_data = trades_to_delete.all()
            trade_ids = [t[0] for t in trade_data]
            opportunity_ids = [t[1] for t in trade_data if t[1]]

            if not trade_ids:
                return {"trades_deleted": 0, "positions_deleted": 0}

            # Delete associated positions
            positions_deleted = await session.execute(
                delete(SimulationPosition).where(SimulationPosition.opportunity_id.in_(opportunity_ids))
            )

            # Delete trades
            trades_deleted = await session.execute(delete(SimulationTrade).where(SimulationTrade.id.in_(trade_ids)))

            await session.commit()

            logger.info(
                "Cleaned up resolved trades",
                trades_deleted=trades_deleted.rowcount,
                positions_deleted=positions_deleted.rowcount,
                older_than_days=older_than_days,
            )

            return {
                "trades_deleted": trades_deleted.rowcount,
                "positions_deleted": positions_deleted.rowcount,
                "cutoff_date": cutoff_date.isoformat(),
            }

    async def expire_old_open_trades(self, older_than_days: int = DEFAULT_OPEN_TRADE_EXPIRY) -> dict:
        """
        Mark old open trades as expired/cancelled.

        This handles trades that were never resolved (market might have been cancelled).

        Args:
            older_than_days: Expire trades open for more than this many days

        Returns:
            Dict with count of expired trades
        """
        cutoff_date = utcnow() - timedelta(days=older_than_days)

        async with AsyncSessionLocal() as session:
            # Update old open trades to cancelled
            result = await session.execute(
                select(SimulationTrade).where(
                    and_(
                        SimulationTrade.executed_at < cutoff_date,
                        SimulationTrade.status == TradeStatus.OPEN,
                    )
                )
            )
            trades = result.scalars().all()

            expired_count = 0
            for trade in trades:
                trade.status = TradeStatus.CANCELLED
                trade.resolved_at = utcnow()
                trade.actual_pnl = -trade.total_cost  # Consider as total loss
                expired_count += 1

            # Also update associated positions
            if trades:
                opportunity_ids = [t.opportunity_id for t in trades if t.opportunity_id]
                if opportunity_ids:
                    positions = await session.execute(
                        select(SimulationPosition).where(SimulationPosition.opportunity_id.in_(opportunity_ids))
                    )
                    for pos in positions.scalars():
                        pos.status = TradeStatus.CANCELLED

            await session.commit()

            logger.info(
                "Expired old open trades",
                expired_count=expired_count,
                older_than_days=older_than_days,
            )

            return {
                "trades_expired": expired_count,
                "cutoff_date": cutoff_date.isoformat(),
            }

    async def cleanup_wallet_trades(
        self,
        older_than_days: int = DEFAULT_WALLET_TRADE_AGE,
        wallet_address: Optional[str] = None,
    ) -> dict:
        """
        Delete old wallet trades.

        Args:
            older_than_days: Delete trades older than this many days
            wallet_address: Optional - only delete for specific wallet

        Returns:
            Dict with deletion count
        """
        cutoff_date = utcnow() - timedelta(days=older_than_days)

        async with AsyncSessionLocal() as session:
            conditions = [WalletTrade.timestamp < cutoff_date]

            if wallet_address:
                conditions.append(WalletTrade.wallet_address == wallet_address)

            result = await session.execute(delete(WalletTrade).where(and_(*conditions)))
            await session.commit()

            logger.info(
                "Cleaned up wallet trades",
                deleted_count=result.rowcount,
                older_than_days=older_than_days,
            )

            return {
                "wallet_trades_deleted": result.rowcount,
                "cutoff_date": cutoff_date.isoformat(),
            }

    async def cleanup_anomalies(self, older_than_days: int = DEFAULT_ANOMALY_AGE, resolved_only: bool = True) -> dict:
        """
        Delete old anomalies.

        Args:
            older_than_days: Delete anomalies older than this many days
            resolved_only: Only delete resolved anomalies if True

        Returns:
            Dict with deletion count
        """
        cutoff_date = utcnow() - timedelta(days=older_than_days)

        async with AsyncSessionLocal() as session:
            conditions = [DetectedAnomaly.detected_at < cutoff_date]

            if resolved_only:
                conditions.append(DetectedAnomaly.is_resolved)

            result = await session.execute(delete(DetectedAnomaly).where(and_(*conditions)))
            await session.commit()

            logger.info(
                "Cleaned up anomalies",
                deleted_count=result.rowcount,
                older_than_days=older_than_days,
                resolved_only=resolved_only,
            )

            return {
                "anomalies_deleted": result.rowcount,
                "cutoff_date": cutoff_date.isoformat(),
            }

    async def cleanup_llm_usage_logs(
        self,
        older_than_days: int = DEFAULT_LLM_USAGE_RETENTION_DAYS,
        preserve_current_month: bool = True,
    ) -> dict:
        """
        Delete old LLM usage logs.

        Args:
            older_than_days: Delete LLM usage rows older than this many days.
                `0` disables cleanup.
            preserve_current_month: Keep current-month rows even if older than cutoff
                so monthly spend tracking stays accurate.

        Returns:
            Dict with deletion count and retention metadata.
        """
        if older_than_days <= 0:
            return {
                "status": "disabled",
                "llm_usage_logs_deleted": 0,
                "older_than_days": int(older_than_days),
                "preserve_current_month": bool(preserve_current_month),
            }

        now = utcnow()
        cutoff_date = now - timedelta(days=older_than_days)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        async with AsyncSessionLocal() as session:
            conditions = [LLMUsageLog.requested_at < cutoff_date]
            if preserve_current_month:
                conditions.append(LLMUsageLog.requested_at < month_start)

            result = await session.execute(delete(LLMUsageLog).where(and_(*conditions)))
            await session.commit()

            logger.info(
                "Cleaned up LLM usage logs",
                deleted_count=result.rowcount,
                older_than_days=older_than_days,
                preserve_current_month=preserve_current_month,
            )

            return {
                "status": "success",
                "llm_usage_logs_deleted": int(result.rowcount or 0),
                "cutoff_date": cutoff_date.isoformat(),
                "month_start": month_start.isoformat(),
                "older_than_days": int(older_than_days),
                "preserve_current_month": bool(preserve_current_month),
            }

    # High-volume "firehose" event types that get the short retention
    # tier.  These are per-evaluation / per-emit debug-firehose rows the
    # scanner+orchestrator write at ~13 rows/sec combined — the dominant
    # source of trader_events growth (2026-05-21: 19.8M of 20.4M rows,
    # 24 GB table, driving 5-6s WAL-fsync commit latency).  BOTH
    # ``firehose_evaluation`` AND ``firehose_emit`` belong here; a prior
    # version only swept ``firehose_evaluation`` so ``firehose_emit``
    # (9.1M rows) silently fell into the 90-day "other" tier and never
    # got pruned.
    FIREHOSE_EVENT_TYPES = ("firehose_evaluation", "firehose_emit")

    async def cleanup_trader_events(
        self,
        firehose_older_than_days: int = 7,
        other_older_than_days: int = 90,
    ) -> dict:
        """Delete old trader_events rows with two-tier retention.

        Firehose events (see ``FIREHOSE_EVENT_TYPES``) are the bulk of
        volume and use a shorter retention.  All other event types
        (decision, order, provider_health, circuit_breaker, etc.) use a
        longer audit-trail retention.

        Deletes in batched chunks of 50 000 with a short pause between
        batches to avoid holding locks on the high-churn table.
        """
        from models.database import TraderEvent

        if firehose_older_than_days <= 0 and other_older_than_days <= 0:
            return {
                "status": "disabled",
                "firehose_deleted": 0,
                "other_deleted": 0,
            }

        batch_size = 50_000
        firehose_deleted = 0
        other_deleted = 0
        firehose_types = list(self.FIREHOSE_EVENT_TYPES)

        if firehose_older_than_days > 0:
            cutoff = utcnow() - timedelta(days=firehose_older_than_days)
            while True:
                async with AsyncSessionLocal() as session:
                    await session.execute(text("SET LOCAL statement_timeout = 0"))
                    batch_ids = (
                        await session.execute(
                            select(TraderEvent.id)
                            .where(
                                TraderEvent.event_type.in_(firehose_types),
                                TraderEvent.created_at < cutoff,
                            )
                            .limit(batch_size)
                        )
                    ).scalars().all()
                    if not batch_ids:
                        break
                    result = await session.execute(
                        delete(TraderEvent).where(TraderEvent.id.in_(batch_ids))
                    )
                    await session.commit()
                    n = int(result.rowcount or 0)
                    firehose_deleted += n
                    if n < batch_size:
                        break
                await asyncio.sleep(0.1)

        if other_older_than_days > 0:
            cutoff = utcnow() - timedelta(days=other_older_than_days)
            while True:
                async with AsyncSessionLocal() as session:
                    await session.execute(text("SET LOCAL statement_timeout = 0"))
                    batch_ids = (
                        await session.execute(
                            select(TraderEvent.id)
                            .where(
                                TraderEvent.event_type.not_in(firehose_types),
                                TraderEvent.created_at < cutoff,
                            )
                            .limit(batch_size)
                        )
                    ).scalars().all()
                    if not batch_ids:
                        break
                    result = await session.execute(
                        delete(TraderEvent).where(TraderEvent.id.in_(batch_ids))
                    )
                    await session.commit()
                    n = int(result.rowcount or 0)
                    other_deleted += n
                    if n < batch_size:
                        break
                await asyncio.sleep(0.1)

        logger.info(
            "Cleaned up trader_events",
            firehose_deleted=firehose_deleted,
            other_deleted=other_deleted,
            firehose_retention_days=firehose_older_than_days,
            other_retention_days=other_older_than_days,
        )
        return {
            "status": "success",
            "firehose_deleted": firehose_deleted,
            "other_deleted": other_deleted,
            "firehose_retention_days": firehose_older_than_days,
            "other_retention_days": other_older_than_days,
        }

    async def _trader_events_retention_settings(self) -> dict:
        config = {
            "firehose_days": 7,
            "other_days": 90,
        }
        try:
            async with AsyncSessionLocal() as session:
                row = (
                    await session.execute(select(AppSettings).where(AppSettings.id == "default"))
                ).scalar_one_or_none()
                if row is None:
                    return config
                if row.trader_events_firehose_retention_days is not None:
                    config["firehose_days"] = max(1, int(row.trader_events_firehose_retention_days))
                if row.trader_events_other_retention_days is not None:
                    config["other_days"] = max(1, int(row.trader_events_other_retention_days))
        except Exception as e:
            logger.warning("Failed to read trader_events retention settings", error=str(e))
        return config

    async def cleanup_trade_signal_emissions(
        self,
        older_than_days: int = DEFAULT_TRADE_SIGNAL_EMISSION_AGE,
        source: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> dict:
        """
        Delete old trade signal emission rows.

        Args:
            older_than_days: Delete emission rows older than this many days.
                `0` disables cleanup.
            source: Optional source filter (scanner/news/weather/crypto/traders/events).
            event_type: Optional event type filter (upsert_insert/upsert_update/status_transition).

        Returns:
            Dict with deletion count and retention metadata.
        """
        if older_than_days <= 0:
            return {
                "status": "disabled",
                "trade_signal_emissions_deleted": 0,
                "older_than_days": int(older_than_days),
                "source": source,
                "event_type": event_type,
            }

        cutoff_date = utcnow() - timedelta(days=older_than_days)
        conditions = [TradeSignalEmission.created_at < cutoff_date]
        if source:
            conditions.append(TradeSignalEmission.source == source)
        if event_type:
            conditions.append(TradeSignalEmission.event_type == event_type)

        async with AsyncSessionLocal() as session:
            result = await session.execute(delete(TradeSignalEmission).where(and_(*conditions)))
            await session.commit()

        deleted_count = int(result.rowcount or 0)
        logger.info(
            "Cleaned up trade signal emissions",
            deleted_count=deleted_count,
            older_than_days=older_than_days,
            source=source,
            event_type=event_type,
        )
        return {
            "status": "success",
            "trade_signal_emissions_deleted": deleted_count,
            "cutoff_date": cutoff_date.isoformat(),
            "older_than_days": int(older_than_days),
            "source": source,
            "event_type": event_type,
        }

    async def cleanup_trade_signal_update_emissions(
        self,
        older_than_days: int = DEFAULT_TRADE_SIGNAL_UPDATE_AGE,
        source: Optional[str] = None,
    ) -> dict:
        """
        Delete noisy upsert_update emissions older than the configured window.

        Args:
            older_than_days: Delete upsert_update rows older than this many days.
                `0` disables cleanup.
            source: Optional source filter.

        Returns:
            Dict with deletion count and retention metadata.
        """
        if older_than_days <= 0:
            return {
                "status": "disabled",
                "trade_signal_updates_deleted": 0,
                "older_than_days": int(older_than_days),
                "source": source,
            }

        cutoff_date = utcnow() - timedelta(days=older_than_days)
        conditions = [
            TradeSignalEmission.event_type == "upsert_update",
            TradeSignalEmission.created_at < cutoff_date,
        ]
        if source:
            conditions.append(TradeSignalEmission.source == source)

        async with AsyncSessionLocal() as session:
            result = await session.execute(delete(TradeSignalEmission).where(and_(*conditions)))
            await session.commit()

        deleted_count = int(result.rowcount or 0)
        logger.info(
            "Cleaned up trade signal update emissions",
            deleted_count=deleted_count,
            older_than_days=older_than_days,
            source=source,
        )
        return {
            "status": "success",
            "trade_signal_updates_deleted": deleted_count,
            "cutoff_date": cutoff_date.isoformat(),
            "older_than_days": int(older_than_days),
            "source": source,
        }

    # Terminal signals (executed/skipped/expired/failed/filtered) past
    # the reactivation lookback are dead weight in the hot table.  The
    # 30-day ``cleanup_trade_signals`` runs only via the API/manual
    # path; without a hot pruner the table grows to 100K+ rows / 6+ GB
    # and ``list_unconsumed_trade_signals`` slows to 3+ seconds per
    # cycle, blowing the fast-trader 2.5s ``statement_timeout`` and
    # corrupting asyncpg's protocol state with cancelled-mid-flight
    # queries.
    _TERMINAL_SIGNAL_STATUSES: tuple[str, ...] = (
        "expired",
        "filtered",
        "skipped",
        "failed",
    )

    async def cleanup_terminal_trade_signals(
        self,
        older_than_hours: int = 24,
    ) -> dict:
        """Aggressively prune terminal ``trade_signals`` past the
        reactivation lookback (default 24h).  Keeps ``executed``,
        ``pending``, and recent terminal rows; deletes the rest.
        Designed to run on a fast cadence (every 30-60 minutes).
        """
        if older_than_hours <= 0:
            return {
                "status": "disabled",
                "trade_signals_deleted": 0,
                "older_than_hours": int(older_than_hours),
            }

        cutoff = utcnow() - timedelta(hours=older_than_hours)
        batch_size = 500
        deleted_count = 0
        terminal_statuses = list(self._TERMINAL_SIGNAL_STATUSES)
        while True:
            async with AsyncSessionLocal() as session:
                # Pruning the hot table can take several seconds when
                # the backlog is large; allow it.  ``SET LOCAL`` is
                # transaction-scoped.
                await session.execute(text("SET LOCAL statement_timeout = '120000'"))
                batch_ids = (
                    await session.execute(
                        select(TradeSignal.id)
                        .where(TradeSignal.status.in_(terminal_statuses))
                        .where(TradeSignal.created_at < cutoff)
                        .limit(batch_size)
                    )
                ).scalars().all()
                if not batch_ids:
                    break
                await session.execute(
                    delete(TradeSignalEmission).where(
                        TradeSignalEmission.signal_id.in_(batch_ids)
                    )
                )
                result = await session.execute(
                    delete(TradeSignal).where(TradeSignal.id.in_(batch_ids))
                )
                await session.commit()
                deleted_count += int(result.rowcount or 0)
                if int(result.rowcount or 0) < batch_size:
                    break
        if deleted_count > 0:
            logger.info(
                "Pruned terminal trade signals",
                deleted_count=deleted_count,
                older_than_hours=older_than_hours,
            )
        return {
            "status": "success",
            "trade_signals_deleted": deleted_count,
            "cutoff": cutoff.isoformat(),
            "older_than_hours": int(older_than_hours),
        }

    async def cleanup_trade_signals(
        self,
        older_than_days: int = DEFAULT_TRADE_SIGNAL_AGE,
    ) -> dict:
        """Delete trade_signals rows older than the configured window.

        Emissions carry an ``ON DELETE NO ACTION`` FK to trade_signals. A signal
        beyond the signal-retention window may still have a fresh emission
        within the emission-retention window — delete those emissions first so
        the parent delete never trips the FK.
        """
        if older_than_days <= 0:
            return {
                "status": "disabled",
                "trade_signals_deleted": 0,
                "older_than_days": int(older_than_days),
            }

        cutoff_date = utcnow() - timedelta(days=older_than_days)
        batch_size = 100
        deleted_count = 0
        while True:
            async with AsyncSessionLocal() as session:
                await session.execute(text("SET LOCAL statement_timeout = 0"))
                batch_ids = (
                    await session.execute(
                        select(TradeSignal.id)
                        .where(TradeSignal.created_at < cutoff_date)
                        .limit(batch_size)
                    )
                ).scalars().all()
                if not batch_ids:
                    break
                await session.execute(
                    delete(TradeSignalEmission).where(
                        TradeSignalEmission.signal_id.in_(batch_ids)
                    )
                )
                result = await session.execute(
                    delete(TradeSignal).where(TradeSignal.id.in_(batch_ids))
                )
                await session.commit()
                deleted_count += int(result.rowcount or 0)
        logger.info(
            "Cleaned up trade signals",
            deleted_count=deleted_count,
            older_than_days=older_than_days,
        )
        return {
            "status": "success",
            "trade_signals_deleted": deleted_count,
            "cutoff_date": cutoff_date.isoformat(),
            "older_than_days": int(older_than_days),
        }

    async def cleanup_wallet_activity_rollups(
        self,
        older_than_days: int = DEFAULT_WALLET_ACTIVITY_ROLLUP_AGE,
        source: Optional[str] = None,
        wallet_address: Optional[str] = None,
    ) -> dict:
        """
        Delete old wallet activity rollup rows.

        Args:
            older_than_days: Delete rollups older than this many days.
            source: Optional source filter.
            wallet_address: Optional wallet filter.

        Returns:
            Dict with deletion count and retention metadata.
        """
        if older_than_days <= 0:
            return {
                "status": "disabled",
                "wallet_activity_rollups_deleted": 0,
                "older_than_days": int(older_than_days),
                "source": source,
                "wallet_address": wallet_address,
            }

        cutoff_date = utcnow() - timedelta(days=older_than_days)
        conditions = [WalletActivityRollup.traded_at < cutoff_date]
        if source:
            conditions.append(WalletActivityRollup.source == source)
        if wallet_address:
            conditions.append(WalletActivityRollup.wallet_address == wallet_address.lower())

        async with AsyncSessionLocal() as session:
            result = await session.execute(delete(WalletActivityRollup).where(and_(*conditions)))
            await session.commit()

        deleted_count = int(result.rowcount or 0)
        logger.info(
            "Cleaned up wallet activity rollups",
            deleted_count=deleted_count,
            older_than_days=older_than_days,
            source=source,
            wallet_address=wallet_address,
        )
        return {
            "status": "success",
            "wallet_activity_rollups_deleted": deleted_count,
            "cutoff_date": cutoff_date.isoformat(),
            "older_than_days": int(older_than_days),
            "source": source,
            "wallet_address": wallet_address,
        }

    async def cleanup_wallet_activity_rollup_duplicates(
        self,
        source: Optional[str] = None,
        older_than_minutes: int = 10,
        batch_limit: int = 100000,
        max_batches: int = 50,
    ) -> dict:
        """
        Delete duplicate wallet activity rollups while retaining one canonical row.

        Duplicate identity:
        - wallet_address (case-insensitive)
        - market_id (case-insensitive)
        - side (normalized uppercase)
        - tx_hash (case-insensitive)
        - traded_at second bucket
        - rounded price/size/notional (6 decimals)

        Args:
            source: Optional source filter.
            older_than_minutes: Skip very recent rows to avoid racing active ingestion.
            batch_limit: Maximum duplicates to delete per SQL statement.
            max_batches: Cap number of deletion batches in one maintenance run.

        Returns:
            Dict with duplicate deletion totals.
        """
        safe_batch_limit = max(1, min(1_000_000, int(batch_limit or 100000)))
        safe_max_batches = max(1, min(1000, int(max_batches or 50)))
        safe_minutes = max(0, int(older_than_minutes or 0))
        cutoff = utcnow() - timedelta(minutes=safe_minutes)

        source_clause = ""
        params: dict[str, object] = {
            "cutoff": cutoff,
            "batch_limit": safe_batch_limit,
        }
        if source:
            source_clause = "AND source = :source"
            params["source"] = source

        statement = text(
            f"""
WITH ranked AS (
    SELECT
        id,
        ROW_NUMBER() OVER (
            PARTITION BY
                lower(wallet_address),
                lower(market_id),
                upper(COALESCE(side, '')),
                COALESCE(NULLIF(lower(tx_hash), ''), ''),
                EXTRACT(EPOCH FROM date_trunc('second', traded_at))::bigint,
                COALESCE(round(price::numeric, 6), -1::numeric),
                COALESCE(round(size::numeric, 6), -1::numeric),
                COALESCE(round(notional::numeric, 6), -1::numeric)
            ORDER BY created_at DESC, id DESC
        ) AS row_num
    FROM wallet_activity_rollups
    WHERE traded_at <= :cutoff
    {source_clause}
),
to_delete AS (
    SELECT id
    FROM ranked
    WHERE row_num > 1
    LIMIT :batch_limit
)
DELETE FROM wallet_activity_rollups target
USING to_delete d
WHERE target.id = d.id
RETURNING target.id
"""
        )

        deleted_total = 0
        batches = 0
        async with AsyncSessionLocal() as session:
            while batches < safe_max_batches:
                result = await session.execute(statement, params)
                await session.commit()
                deleted = len(result.scalars().all())
                if deleted <= 0:
                    break
                deleted_total += deleted
                batches += 1
                if deleted < safe_batch_limit:
                    break

        logger.info(
            "Cleaned duplicate wallet activity rollups",
            deleted_total=deleted_total,
            batches=batches,
            source=source,
            older_than_minutes=safe_minutes,
            batch_limit=safe_batch_limit,
        )
        return {
            "status": "success",
            "wallet_activity_rollup_duplicates_deleted": int(deleted_total),
            "batches": int(batches),
            "source": source,
            "older_than_minutes": safe_minutes,
            "batch_limit": safe_batch_limit,
            "max_batches": safe_max_batches,
            "cutoff_date": cutoff.isoformat(),
        }

    # ------------------------------------------------------------------
    # Retention for previously-unbounded high-volume tables (2026-05-26).
    # These had NO DELETE retention and grew until the host disk filled,
    # which stalled WAL fsync (5-6s commits).  All are ephemeral
    # microstructure / per-decision audit / wallet-event tables with no
    # inbound FKs (verified) EXCEPT trader_decisions, which is pruned only
    # when not referenced by orders/sessions so the trade->decision audit
    # link is never severed.
    # ------------------------------------------------------------------
    async def _prune_table_by_age(
        self,
        table: str,
        time_column: str,
        older_than_days: int,
        *,
        batch_size: int = 5000,
        extra_where: str = "",
        max_batches: int = 200,
    ) -> dict:
        """Batched, lock-light age prune for an id-keyed table.

        ``table``/``time_column``/``extra_where`` come exclusively from the
        fixed internal call sites below (never user input), so the f-string
        composition carries no injection risk.  Deletes in keyset batches,
        committing each batch, so it never holds a long lock on the
        high-churn table.

        ``max_batches`` bounds a single run so the daily sweep stays gentle
        on the live DB even when a large historical backlog exists (these
        tables can hold millions of rows on first prune).  Hitting the cap
        leaves ``capped=True``; the next scheduled run continues where this
        left off.  steady-state daily volume is far below the cap, so once
        the backlog is worked down each run clears everything and stops
        early on ``rowcount < batch_size``.  Returns a per-table result dict.
        """
        result_key = f"{table}_deleted"
        if older_than_days <= 0:
            return {"status": "disabled", result_key: 0, "older_than_days": int(older_than_days)}

        cutoff = utcnow() - timedelta(days=older_than_days)
        where = f"{time_column} < :cutoff"
        if extra_where:
            where = f"{where} AND {extra_where}"
        stmt = text(
            f"DELETE FROM {table} WHERE id IN "
            f"(SELECT id FROM {table} WHERE {where} LIMIT :batch)"
        )
        deleted = 0
        batches = 0
        capped = False
        while True:
            if max_batches and batches >= max_batches:
                capped = True
                break
            async with AsyncSessionLocal() as session:
                await session.execute(text("SET LOCAL statement_timeout = 0"))
                res = await session.execute(stmt, {"cutoff": cutoff, "batch": int(batch_size)})
                await session.commit()
            n = int(res.rowcount or 0)
            deleted += n
            batches += 1
            if n < batch_size:
                break
            await asyncio.sleep(0.05)
        if deleted:
            logger.info(
                "Pruned table by age",
                table=table,
                deleted=deleted,
                batches=batches,
                capped=capped,
                older_than_days=older_than_days,
            )
        return {
            "status": "success",
            result_key: deleted,
            "batches": batches,
            "capped": capped,
            "cutoff_date": cutoff.isoformat(),
            "older_than_days": int(older_than_days),
        }

    async def cleanup_market_microstructure_snapshots(
        self, older_than_days: int = DEFAULT_MARKET_MICROSTRUCTURE_AGE
    ) -> dict:
        """Delete L2 microstructure snapshots older than the window."""
        return await self._prune_table_by_age(
            "market_microstructure_snapshots", "observed_at", older_than_days
        )

    async def cleanup_book_delta_events(
        self, older_than_days: int = DEFAULT_BOOK_DELTA_EVENTS_AGE
    ) -> dict:
        """Delete tick-by-tick book delta events older than the window."""
        return await self._prune_table_by_age(
            "book_delta_events", "observed_at", older_than_days
        )

    async def cleanup_wallet_monitor_events(
        self, older_than_days: int = DEFAULT_WALLET_MONITOR_EVENTS_AGE
    ) -> dict:
        """Delete smart-money wallet monitor events older than the window."""
        return await self._prune_table_by_age(
            "wallet_monitor_events", "detected_at", older_than_days
        )

    async def cleanup_trader_decision_checks(
        self, older_than_days: int = DEFAULT_TRADER_DECISION_CHECKS_AGE
    ) -> dict:
        """Delete per-decision gate-audit rows older than the window.

        Independent of ``cleanup_trader_decisions`` — checks carry their own
        ``created_at`` and a CASCADE FK to trader_decisions, so deleting a
        check never affects its parent decision.
        """
        return await self._prune_table_by_age(
            "trader_decision_checks", "created_at", older_than_days, batch_size=10000
        )

    async def cleanup_trader_decisions(
        self, older_than_days: int = DEFAULT_TRADER_DECISIONS_AGE
    ) -> dict:
        """Delete decision records older than the window — but ONLY those not
        referenced by a real trade artifact (trader_orders / execution_sessions
        / trader_signal_consumption / strategy_experiment_assignments), all of
        which FK to trader_decisions with ON DELETE SET NULL.  Pruning a
        referenced decision would NULL the order->decision audit link, so we
        skip those rows entirely and keep the trade audit trail intact.
        """
        guard = (
            "NOT EXISTS (SELECT 1 FROM trader_orders o WHERE o.decision_id = trader_decisions.id) "
            "AND NOT EXISTS (SELECT 1 FROM execution_sessions e WHERE e.decision_id = trader_decisions.id) "
            "AND NOT EXISTS (SELECT 1 FROM trader_signal_consumption c WHERE c.decision_id = trader_decisions.id) "
            "AND NOT EXISTS (SELECT 1 FROM strategy_experiment_assignments a WHERE a.decision_id = trader_decisions.id)"
        )
        return await self._prune_table_by_age(
            "trader_decisions", "created_at", older_than_days, batch_size=2000, extra_where=guard
        )

    async def cleanup_opportunity_history(
        self, older_than_days: int = DEFAULT_OPPORTUNITY_HISTORY_AGE
    ) -> dict:
        """Delete opportunity history snapshots older than the window."""
        return await self._prune_table_by_age(
            "opportunity_history", "detected_at", older_than_days
        )

    async def _extended_retention_settings(self) -> dict:
        """Read the unbounded-table retention windows from AppSettings,
        falling back to the conservative defaults.  Mirrors the existing
        ``_high_volume_cleanup_settings`` / ``_trader_events_retention_settings``
        pattern so every window is operator-overridable from the Settings UI.
        """
        config = {
            "market_microstructure_days": self.DEFAULT_MARKET_MICROSTRUCTURE_AGE,
            "book_delta_events_days": self.DEFAULT_BOOK_DELTA_EVENTS_AGE,
            "wallet_monitor_events_days": self.DEFAULT_WALLET_MONITOR_EVENTS_AGE,
            "trader_decision_checks_days": self.DEFAULT_TRADER_DECISION_CHECKS_AGE,
            "trader_decisions_days": self.DEFAULT_TRADER_DECISIONS_AGE,
            "opportunity_history_days": self.DEFAULT_OPPORTUNITY_HISTORY_AGE,
        }
        try:
            async with AsyncSessionLocal() as session:
                row = (
                    await session.execute(select(AppSettings).where(AppSettings.id == "default"))
                ).scalar_one_or_none()
                if row is None:
                    return config
                if getattr(row, "cleanup_market_microstructure_days", None) is not None:
                    config["market_microstructure_days"] = max(0, int(row.cleanup_market_microstructure_days))
                if getattr(row, "cleanup_book_delta_events_days", None) is not None:
                    config["book_delta_events_days"] = max(0, int(row.cleanup_book_delta_events_days))
                if getattr(row, "cleanup_wallet_monitor_events_days", None) is not None:
                    config["wallet_monitor_events_days"] = max(0, int(row.cleanup_wallet_monitor_events_days))
                if getattr(row, "cleanup_trader_decision_checks_days", None) is not None:
                    config["trader_decision_checks_days"] = max(0, int(row.cleanup_trader_decision_checks_days))
                if getattr(row, "cleanup_trader_decisions_days", None) is not None:
                    config["trader_decisions_days"] = max(0, int(row.cleanup_trader_decisions_days))
                if getattr(row, "cleanup_opportunity_history_days", None) is not None:
                    config["opportunity_history_days"] = max(0, int(row.cleanup_opportunity_history_days))
        except Exception as e:
            logger.warning("Failed to read extended retention settings", error=str(e))
        return config

    async def delete_all_trades(self, account_id: Optional[str] = None, confirm: bool = False) -> dict:
        """
        Delete ALL trades (nuclear option).

        Args:
            account_id: Optional - only delete for specific account
            confirm: Must be True to proceed (safety check)

        Returns:
            Dict with deletion counts
        """
        if not confirm:
            raise ValueError("Must set confirm=True to delete all trades")

        async with AsyncSessionLocal() as session:
            if account_id:
                # Delete for specific account
                # Get opportunity IDs first
                trades = await session.execute(
                    select(SimulationTrade.opportunity_id).where(SimulationTrade.account_id == account_id)
                )
                [t[0] for t in trades.all() if t[0]]

                # Delete positions
                positions_result = await session.execute(
                    delete(SimulationPosition).where(SimulationPosition.account_id == account_id)
                )

                # Delete trades
                trades_result = await session.execute(
                    delete(SimulationTrade).where(SimulationTrade.account_id == account_id)
                )
            else:
                # Delete everything
                positions_result = await session.execute(delete(SimulationPosition))
                trades_result = await session.execute(delete(SimulationTrade))

            await session.commit()

            logger.warning(
                "Deleted all trades",
                trades_deleted=trades_result.rowcount,
                positions_deleted=positions_result.rowcount,
                account_id=account_id,
            )

            return {
                "trades_deleted": trades_result.rowcount,
                "positions_deleted": positions_result.rowcount,
                "account_id": account_id,
            }

    async def delete_trades_by_status(self, statuses: list[TradeStatus], account_id: Optional[str] = None) -> dict:
        """
        Delete trades by status.

        Args:
            statuses: List of TradeStatus values to delete
            account_id: Optional - only delete for specific account

        Returns:
            Dict with deletion counts
        """
        async with AsyncSessionLocal() as session:
            conditions = [SimulationTrade.status.in_(statuses)]

            if account_id:
                conditions.append(SimulationTrade.account_id == account_id)

            # Get opportunity IDs for position cleanup
            trades = await session.execute(select(SimulationTrade.opportunity_id).where(and_(*conditions)))
            opportunity_ids = [t[0] for t in trades.all() if t[0]]

            # Delete positions
            positions_deleted = 0
            if opportunity_ids:
                positions_result = await session.execute(
                    delete(SimulationPosition).where(SimulationPosition.opportunity_id.in_(opportunity_ids))
                )
                positions_deleted = positions_result.rowcount

            # Delete trades
            trades_result = await session.execute(delete(SimulationTrade).where(and_(*conditions)))

            await session.commit()

            logger.info(
                "Deleted trades by status",
                trades_deleted=trades_result.rowcount,
                positions_deleted=positions_deleted,
                statuses=[s.value for s in statuses],
            )

            return {
                "trades_deleted": trades_result.rowcount,
                "positions_deleted": positions_deleted,
                "statuses": [s.value for s in statuses],
            }

    # ------------------------------------------------------------------
    # PostgreSQL-level maintenance (VACUUM, ANALYZE, REINDEX)
    # ------------------------------------------------------------------

    # High-churn tables that accumulate dead tuples rapidly.
    _HIGH_CHURN_TABLES: list[str] = [
        "market_catalog",
        "discovered_wallets",
        "simulation_trades",
        "simulation_positions",
        "opportunity_history",
        "opportunity_states",
        "trade_signals",
        "trade_signal_emissions",
        # 2026-05-09: added trader_events. Post-fillfactor SLOW COMMIT
        # DIAGNOSTIC showed trader_events.autovacuum_age_s = 22000
        # (6+ hours) on a 4.9M-row table. The aggressive autovacuum
        # tuning in migration 202605090003 will catch up over time,
        # but a one-shot VACUUM FULL via this API clears the backlog.
        "trader_events",
        # 2026-05-09: added trader_decisions + trader_decision_checks
        # so the orchestrator's audit table family is included in the
        # periodic sweep. trader_decisions had analyze_age_s = 18,500s
        # (5+ hours) -- planner stats go stale.
        "trader_decisions",
        "trader_decision_checks",
        "wallet_activity_rollups",
        "wallet_trades",
        "cached_markets",
        "scanner_snapshots",
        "data_source_records",
        # 2026-05-26: newly-retained high-volume tables — vacuum them so the
        # space freed by the new DELETE sweeps is reclaimed/reused.
        "market_microstructure_snapshots",
        "book_delta_events",
        "wallet_monitor_events",
    ]

    async def vacuum_analyze(
        self,
        full: bool = False,
        tables: Optional[list[str]] = None,
    ) -> dict:
        """Run VACUUM (ANALYZE) on high-churn tables to reclaim dead tuples.

        VACUUM cannot run inside a transaction, so we acquire a raw
        connection and set isolation_level to AUTOCOMMIT.

        Args:
            full: If True, run VACUUM FULL (rewrites table, reclaims disk,
                  but takes an exclusive lock). Default False uses regular
                  VACUUM which is non-blocking.
            tables: Optional list of specific table names. If provided,
                  only those tables (intersected with public-schema
                  existence) are vacuumed. Useful for targeted ops on
                  large tables (e.g. trader_events) where the default
                  full-list run would block on smaller-table failures.

        Returns:
            Dict with per-table timing and overall summary.
        """
        mode = "VACUUM FULL ANALYZE" if full else "VACUUM ANALYZE"
        results: dict[str, dict] = {}
        total_start = time.monotonic()
        tables_processed = 0
        tables_skipped = 0

        # Discover which of our target tables actually exist
        existing_tables: set[str] = set()
        async with AsyncSessionLocal() as session:
            bind = session.get_bind()
            if bind is None or bind.dialect.name != "postgresql":
                return {"status": "skipped", "reason": "not_postgresql"}

            rows = await session.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            )
            existing_tables = {r[0] for r in rows}

        if tables:
            requested = [str(t).strip() for t in tables if str(t).strip()]
            tables_to_vacuum = [t for t in requested if t in existing_tables]
        else:
            tables_to_vacuum = [t for t in self._HIGH_CHURN_TABLES if t in existing_tables]
        if not tables_to_vacuum:
            return {"status": "skipped", "reason": "no_matching_tables"}

        logger.info("Starting PostgreSQL maintenance", mode=mode, tables=len(tables_to_vacuum))

        # VACUUM cannot run inside a transaction. Use AUTOCOMMIT isolation.
        async with async_engine.connect() as conn:
            autocommit_conn = await conn.execution_options(isolation_level="AUTOCOMMIT")
            # 2026-05-09: lift the per-statement timeout for the
            # duration of VACUUM operations. The pool default (30s)
            # caused VACUUM FULL on wallet_activity_rollups and
            # data_source_records to error with QueryCanceledError.
            # 30 minutes is a generous cap that still bounds runaway
            # vacuum runs without truncating legitimate work.
            try:
                await autocommit_conn.execute(text("SET statement_timeout = '1800000'"))
            except Exception as exc:
                logger.warning("VACUUM: failed to raise statement_timeout: %s", exc)
            # 2026-05-09: lift the per-statement lock_timeout for the
            # duration of VACUUM operations. The pool default (5s)
            # caused VACUUM FULL on trade_signals to error with
            # ``LockNotAvailableError: canceling statement due to lock
            # timeout`` when AccessExclusiveLock contended with
            # background producer UPSERTs. ``0`` = wait indefinitely.
            # Bounded by the statement_timeout above (30 min cap), so
            # there's no infinite-hang risk; just enough patience to
            # actually acquire the lock between writer transactions.
            try:
                await autocommit_conn.execute(text("SET lock_timeout = '0'"))
            except Exception as exc:
                logger.warning("VACUUM: failed to clear lock_timeout: %s", exc)
            for table_name in tables_to_vacuum:
                t0 = time.monotonic()
                try:
                    await autocommit_conn.execute(text(f"{mode} {table_name}"))
                    elapsed = round(time.monotonic() - t0, 2)
                    results[table_name] = {"status": "ok", "seconds": elapsed}
                    tables_processed += 1
                    logger.info("Vacuumed table", table=table_name, mode=mode, seconds=elapsed)
                except Exception as exc:
                    elapsed = round(time.monotonic() - t0, 2)
                    results[table_name] = {"status": "error", "error": str(exc), "seconds": elapsed}
                    tables_skipped += 1
                    logger.warning("VACUUM failed for table", table=table_name, error=str(exc))

        total_elapsed = round(time.monotonic() - total_start, 2)
        logger.info(
            "PostgreSQL maintenance completed",
            mode=mode,
            tables_processed=tables_processed,
            tables_skipped=tables_skipped,
            total_seconds=total_elapsed,
        )

        return {
            "status": "completed",
            "mode": mode,
            "tables_processed": tables_processed,
            "tables_skipped": tables_skipped,
            "total_seconds": total_elapsed,
            "tables": results,
        }

    async def reindex_tables(self) -> dict:
        """Run REINDEX on high-churn tables to rebuild bloated indexes.

        Index bloat is not fixed by regular VACUUM — only VACUUM FULL
        or REINDEX can reclaim index space. REINDEX takes a lock on
        each index while rebuilding but is faster than VACUUM FULL
        because it only rebuilds indexes, not the whole table.

        Returns:
            Dict with per-table timing and overall summary.
        """
        results: dict[str, dict] = {}
        total_start = time.monotonic()
        tables_processed = 0
        tables_skipped = 0

        existing_tables: set[str] = set()
        async with AsyncSessionLocal() as session:
            bind = session.get_bind()
            if bind is None or bind.dialect.name != "postgresql":
                return {"status": "skipped", "reason": "not_postgresql"}

            rows = await session.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            )
            existing_tables = {r[0] for r in rows}

        tables_to_reindex = [t for t in self._HIGH_CHURN_TABLES if t in existing_tables]
        if not tables_to_reindex:
            return {"status": "skipped", "reason": "no_matching_tables"}

        logger.info("Starting REINDEX", tables=len(tables_to_reindex))

        async with async_engine.connect() as conn:
            autocommit_conn = await conn.execution_options(isolation_level="AUTOCOMMIT")
            # Same rationale as vacuum_analyze: REINDEX takes locks
            # that may queue behind active producer transactions on
            # high-churn tables. Lift the pool's 5s default lock_timeout
            # and 30s statement_timeout so REINDEX can complete.
            try:
                await autocommit_conn.execute(text("SET statement_timeout = '1800000'"))
            except Exception as exc:
                logger.warning("REINDEX: failed to raise statement_timeout: %s", exc)
            try:
                await autocommit_conn.execute(text("SET lock_timeout = '0'"))
            except Exception as exc:
                logger.warning("REINDEX: failed to clear lock_timeout: %s", exc)
            for table_name in tables_to_reindex:
                t0 = time.monotonic()
                try:
                    await autocommit_conn.execute(text(f"REINDEX TABLE {table_name}"))
                    elapsed = round(time.monotonic() - t0, 2)
                    results[table_name] = {"status": "ok", "seconds": elapsed}
                    tables_processed += 1
                    logger.info("Reindexed table", table=table_name, seconds=elapsed)
                except Exception as exc:
                    elapsed = round(time.monotonic() - t0, 2)
                    results[table_name] = {"status": "error", "error": str(exc), "seconds": elapsed}
                    tables_skipped += 1
                    logger.warning("REINDEX failed for table", table=table_name, error=str(exc))

        total_elapsed = round(time.monotonic() - total_start, 2)
        logger.info(
            "REINDEX completed",
            tables_processed=tables_processed,
            tables_skipped=tables_skipped,
            total_seconds=total_elapsed,
        )

        return {
            "status": "completed",
            "mode": "REINDEX",
            "tables_processed": tables_processed,
            "tables_skipped": tables_skipped,
            "total_seconds": total_elapsed,
            "tables": results,
        }

    async def full_cleanup(
        self,
        resolved_trade_days: int = DEFAULT_RESOLVED_TRADE_AGE,
        open_trade_expiry_days: int = DEFAULT_OPEN_TRADE_EXPIRY,
        wallet_trade_days: int = DEFAULT_WALLET_TRADE_AGE,
        anomaly_days: int = DEFAULT_ANOMALY_AGE,
        llm_usage_retention_days: Optional[int] = None,
        trade_signal_emission_days: Optional[int] = None,
        trade_signal_update_days: Optional[int] = None,
        trade_signal_days: Optional[int] = None,
        wallet_activity_rollup_days: Optional[int] = None,
        wallet_activity_dedupe_enabled: Optional[bool] = None,
        trader_events_firehose_days: Optional[int] = None,
        trader_events_other_days: Optional[int] = None,
    ) -> dict:
        """
        Run full database cleanup with all maintenance tasks.

        Args:
            resolved_trade_days: Delete resolved trades older than this
            open_trade_expiry_days: Expire open trades older than this
            wallet_trade_days: Delete wallet trades older than this
            anomaly_days: Delete resolved anomalies older than this
            llm_usage_retention_days: Delete LLM usage logs older than this.
                `None` reads from AppSettings.
            trade_signal_emission_days: Delete trade signal emissions older than this.
                `None` reads from AppSettings.
            trade_signal_update_days: Delete upsert_update emissions older than this.
                `None` reads from AppSettings.
            wallet_activity_rollup_days: Delete wallet activity rollups older than this.
                `None` reads from AppSettings.
            wallet_activity_dedupe_enabled: Run rollup duplicate cleanup pass.
                `None` reads from AppSettings.

        Returns:
            Dict with all cleanup results
        """
        logger.info("Starting full database cleanup")

        results = {}
        high_volume_cfg = await self._high_volume_cleanup_settings()

        # 1. Expire old open trades first
        results["expired_trades"] = await self.expire_old_open_trades(older_than_days=open_trade_expiry_days)

        # 2. Clean up resolved trades
        results["resolved_trades"] = await self.cleanup_resolved_trades(older_than_days=resolved_trade_days)

        # 3. Clean up wallet trades
        results["wallet_trades"] = await self.cleanup_wallet_trades(older_than_days=wallet_trade_days)

        # 4. Clean up anomalies
        results["anomalies"] = await self.cleanup_anomalies(older_than_days=anomaly_days)

        # 5. Prune old LLM usage logs
        try:
            retention_days = llm_usage_retention_days
            if retention_days is None:
                retention_days = await self._llm_usage_retention_days_setting()
            results["llm_usage_logs"] = await self.cleanup_llm_usage_logs(
                older_than_days=int(retention_days),
                preserve_current_month=True,
            )
        except Exception as e:
            logger.warning("LLM usage log cleanup failed during full maintenance run", error=str(e))
            results["llm_usage_logs"] = {"status": "error", "error": str(e)}

        # 5b. Prune trader_events with two-tier retention.
        try:
            te_cfg = await self._trader_events_retention_settings()
            firehose_days = trader_events_firehose_days
            if firehose_days is None:
                firehose_days = te_cfg["firehose_days"]
            other_days = trader_events_other_days
            if other_days is None:
                other_days = te_cfg["other_days"]
            results["trader_events"] = await self.cleanup_trader_events(
                firehose_older_than_days=int(firehose_days),
                other_older_than_days=int(other_days),
            )
        except Exception as e:
            logger.warning("Trader events cleanup failed during full maintenance run", error=str(e))
            results["trader_events"] = {"status": "error", "error": str(e)}

        # 5c. Prune the previously-unbounded high-volume tables (2026-05-26).
        # These had no DELETE retention and were the actual disk-fill driver
        # behind the WAL-fsync stall incident.  Each runs independently so one
        # failure doesn't abort the rest.
        try:
            ext_cfg = await self._extended_retention_settings()
            ext_results: dict = {}
            for label, fn, days in (
                ("market_microstructure_snapshots", self.cleanup_market_microstructure_snapshots, ext_cfg["market_microstructure_days"]),
                ("book_delta_events", self.cleanup_book_delta_events, ext_cfg["book_delta_events_days"]),
                ("wallet_monitor_events", self.cleanup_wallet_monitor_events, ext_cfg["wallet_monitor_events_days"]),
                ("trader_decision_checks", self.cleanup_trader_decision_checks, ext_cfg["trader_decision_checks_days"]),
                ("trader_decisions", self.cleanup_trader_decisions, ext_cfg["trader_decisions_days"]),
                ("opportunity_history", self.cleanup_opportunity_history, ext_cfg["opportunity_history_days"]),
            ):
                try:
                    ext_results[label] = await fn(older_than_days=int(days))
                except Exception as inner:
                    logger.warning("Extended retention sweep failed", table=label, error=str(inner))
                    ext_results[label] = {"status": "error", "error": str(inner)}
            results["extended_retention"] = ext_results
        except Exception as e:
            logger.warning("Extended retention cleanup failed during full maintenance run", error=str(e))
            results["extended_retention"] = {"status": "error", "error": str(e)}

        # 6. Prune noisy upsert updates while preserving meaningful transitions.
        try:
            update_retention_days = trade_signal_update_days
            if update_retention_days is None:
                update_retention_days = int(high_volume_cfg["trade_signal_update_days"])
            results["trade_signal_updates"] = await self.cleanup_trade_signal_update_emissions(
                older_than_days=int(update_retention_days),
            )
        except Exception as e:
            logger.warning("Trade signal update cleanup failed during full maintenance run", error=str(e))
            results["trade_signal_updates"] = {"status": "error", "error": str(e)}

        # 7. Prune old trade signal emission history.
        try:
            emission_retention_days = trade_signal_emission_days
            if emission_retention_days is None:
                emission_retention_days = int(high_volume_cfg["trade_signal_emission_days"])
            results["trade_signal_emissions"] = await self.cleanup_trade_signal_emissions(
                older_than_days=int(emission_retention_days),
            )
        except Exception as e:
            logger.warning("Trade signal emission cleanup failed during full maintenance run", error=str(e))
            results["trade_signal_emissions"] = {"status": "error", "error": str(e)}

        # 7b. Prune aged trade signals (emissions FK-reference them; must run after
        # the emissions cleanup so newer emissions don't block deletion).
        try:
            signal_retention_days = trade_signal_days
            if signal_retention_days is None:
                signal_retention_days = int(high_volume_cfg["trade_signal_days"])
            results["trade_signals"] = await self.cleanup_trade_signals(
                older_than_days=int(signal_retention_days),
            )
        except Exception as e:
            logger.warning("Trade signal cleanup failed during full maintenance run", error=str(e))
            results["trade_signals"] = {"status": "error", "error": str(e)}

        # 8. Remove duplicate wallet activity rollups.
        try:
            dedupe_enabled = wallet_activity_dedupe_enabled
            if dedupe_enabled is None:
                dedupe_enabled = bool(high_volume_cfg["wallet_activity_dedupe_enabled"])
            if dedupe_enabled:
                results["wallet_activity_rollup_duplicates"] = await self.cleanup_wallet_activity_rollup_duplicates()
            else:
                results["wallet_activity_rollup_duplicates"] = {"status": "disabled"}
        except Exception as e:
            logger.warning("Wallet activity duplicate cleanup failed during full maintenance run", error=str(e))
            results["wallet_activity_rollup_duplicates"] = {"status": "error", "error": str(e)}

        # 9. Prune aged wallet activity rollups.
        try:
            rollup_retention_days = wallet_activity_rollup_days
            if rollup_retention_days is None:
                rollup_retention_days = int(high_volume_cfg["wallet_activity_rollup_days"])
            results["wallet_activity_rollups"] = await self.cleanup_wallet_activity_rollups(
                older_than_days=max(45, int(rollup_retention_days)),
            )
        except Exception as e:
            logger.warning("Wallet activity rollup cleanup failed during full maintenance run", error=str(e))
            results["wallet_activity_rollups"] = {"status": "error", "error": str(e)}

        # 10. Prune old database backups
        try:
            results["db_backups"] = await self.cleanup_database_backups(
                older_than_days=self.DEFAULT_DATABASE_BACKUP_RETENTION_DAYS
            )
        except Exception as e:
            logger.warning("Database backup cleanup failed during full maintenance run", error=str(e))
            results["db_backups"] = {"status": "error", "error": str(e)}

        # 11. Prune stale/mismatched market metadata cache entries
        try:
            market_cache_cfg = await self._market_cache_hygiene_settings()
            if market_cache_cfg["enabled"]:
                results["market_cache"] = await market_cache_service.run_hygiene_if_due(
                    force=True,
                    interval_hours=market_cache_cfg["interval_hours"],
                    retention_days=market_cache_cfg["retention_days"],
                    reference_lookback_days=market_cache_cfg["reference_lookback_days"],
                    weak_entry_grace_days=market_cache_cfg["weak_entry_grace_days"],
                    max_entries_per_slug=market_cache_cfg["max_entries_per_slug"],
                )
            else:
                results["market_cache"] = {"status": "disabled"}
        except Exception as e:
            logger.warning("Market cache cleanup failed during full maintenance run", error=str(e))
            results["market_cache"] = {"status": "error", "error": str(e)}

        # 12. VACUUM ANALYZE high-churn tables to reclaim dead tuples
        try:
            results["vacuum_analyze"] = await self.vacuum_analyze(full=False)
        except Exception as e:
            logger.warning("VACUUM ANALYZE failed during full maintenance run", error=str(e))
            results["vacuum_analyze"] = {"status": "error", "error": str(e)}

        logger.info("Full database cleanup completed", results=results)

        return results

    async def start_background_cleanup(self, interval_hours: int = 24, cleanup_config: Optional[dict] = None):
        """
        Start background cleanup task that runs periodically.

        Args:
            interval_hours: Run cleanup every X hours (default: 24)
            cleanup_config: Optional config for cleanup thresholds
        """
        self._running = True
        config = cleanup_config or {}

        logger.info("Starting background cleanup task", interval_hours=interval_hours)

        # Short grace period after startup so the app can finish booting before
        # the first cleanup pass runs.
        await asyncio.sleep(60)

        while self._running:
            try:
                logger.info("Running scheduled database cleanup")

                # Run full cleanup with configured thresholds
                await self.full_cleanup(
                    resolved_trade_days=config.get("resolved_trade_days", self.DEFAULT_RESOLVED_TRADE_AGE),
                    open_trade_expiry_days=config.get("open_trade_expiry_days", self.DEFAULT_OPEN_TRADE_EXPIRY),
                    wallet_trade_days=config.get("wallet_trade_days", self.DEFAULT_WALLET_TRADE_AGE),
                    anomaly_days=config.get("anomaly_days", self.DEFAULT_ANOMALY_AGE),
                    llm_usage_retention_days=config.get("llm_usage_retention_days"),
                    trade_signal_emission_days=config.get(
                        "trade_signal_emission_days", self.DEFAULT_TRADE_SIGNAL_EMISSION_AGE
                    ),
                    trade_signal_update_days=config.get(
                        "trade_signal_update_days", self.DEFAULT_TRADE_SIGNAL_UPDATE_AGE
                    ),
                    trade_signal_days=config.get(
                        "trade_signal_days", self.DEFAULT_TRADE_SIGNAL_AGE
                    ),
                    wallet_activity_rollup_days=config.get(
                        "wallet_activity_rollup_days", self.DEFAULT_WALLET_ACTIVITY_ROLLUP_AGE
                    ),
                    wallet_activity_dedupe_enabled=config.get("wallet_activity_dedupe_enabled", True),
                )

            except asyncio.CancelledError:
                logger.info("Background cleanup task cancelled")
                break
            except Exception as e:
                logger.error("Background cleanup failed", error=str(e))

            await asyncio.sleep(interval_hours * 3600)
            if not self._running:
                break

        logger.info("Background cleanup task stopped")

    def stop(self):
        """Stop the background cleanup task"""
        self._running = False


# Singleton instance
maintenance_service = MaintenanceService()
