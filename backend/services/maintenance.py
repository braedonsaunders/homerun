"""
Database Maintenance Service

Handles cleanup of old trades, expiration of stale data, and database maintenance.
"""

import asyncio
from datetime import timedelta
from utils.utcnow import utcnow
from typing import Optional
from sqlalchemy import select, delete, func, and_

from models.database import (
    SimulationTrade,
    SimulationPosition,
    WalletTrade,
    OpportunityHistory,
    DetectedAnomaly,
    LLMUsageLog,
    TradeStatus,
    AsyncSessionLocal,
    AppSettings,
)
from services.market_cache import market_cache_service
from utils.logger import get_logger

logger = get_logger("maintenance")


class MaintenanceService:
    """Database maintenance and cleanup service"""

    # Default age thresholds (in days)
    DEFAULT_RESOLVED_TRADE_AGE = 30  # Delete resolved trades older than 30 days
    DEFAULT_OPEN_TRADE_EXPIRY = 90  # Mark open trades as expired after 90 days
    DEFAULT_WALLET_TRADE_AGE = 60  # Delete wallet trades older than 60 days
    DEFAULT_ANOMALY_AGE = 30  # Delete resolved anomalies older than 30 days
    DEFAULT_LLM_USAGE_RETENTION_DAYS = 30  # Delete raw LLM usage logs older than 30 days

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

    async def get_database_stats(self) -> dict:
        """Get statistics about database contents"""
        async with AsyncSessionLocal() as session:
            # Count simulation trades by status
            trade_counts = {}
            for status in TradeStatus:
                result = await session.execute(
                    select(func.count(SimulationTrade.id)).where(SimulationTrade.status == status)
                )
                trade_counts[status.value] = result.scalar() or 0

            # Count total simulation trades
            total_trades = await session.execute(select(func.count(SimulationTrade.id)))

            # Count simulation positions
            total_positions = await session.execute(select(func.count(SimulationPosition.id)))
            open_positions = await session.execute(
                select(func.count(SimulationPosition.id)).where(SimulationPosition.status == TradeStatus.OPEN)
            )

            # Count wallet trades
            wallet_trades = await session.execute(select(func.count(WalletTrade.id)))

            # Count opportunity history
            opportunities = await session.execute(select(func.count(OpportunityHistory.id)))

            # Count anomalies
            anomalies = await session.execute(select(func.count(DetectedAnomaly.id)))
            resolved_anomalies = await session.execute(
                select(func.count(DetectedAnomaly.id)).where(DetectedAnomaly.is_resolved)
            )

            # Get oldest and newest trade dates
            oldest_trade = await session.execute(select(func.min(SimulationTrade.executed_at)))
            newest_trade = await session.execute(select(func.max(SimulationTrade.executed_at)))

            return {
                "simulation_trades": {
                    "total": total_trades.scalar() or 0,
                    "by_status": trade_counts,
                },
                "simulation_positions": {
                    "total": total_positions.scalar() or 0,
                    "open": open_positions.scalar() or 0,
                },
                "wallet_trades": wallet_trades.scalar() or 0,
                "opportunity_history": opportunities.scalar() or 0,
                "anomalies": {
                    "total": anomalies.scalar() or 0,
                    "resolved": resolved_anomalies.scalar() or 0,
                },
                "date_range": {
                    "oldest_trade": oldest_trade.scalar().isoformat() if oldest_trade.scalar() else None,
                    "newest_trade": newest_trade.scalar().isoformat() if newest_trade.scalar() else None,
                },
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

    async def full_cleanup(
        self,
        resolved_trade_days: int = DEFAULT_RESOLVED_TRADE_AGE,
        open_trade_expiry_days: int = DEFAULT_OPEN_TRADE_EXPIRY,
        wallet_trade_days: int = DEFAULT_WALLET_TRADE_AGE,
        anomaly_days: int = DEFAULT_ANOMALY_AGE,
        llm_usage_retention_days: Optional[int] = None,
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

        Returns:
            Dict with all cleanup results
        """
        logger.info("Starting full database cleanup")

        results = {}

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

        # 6. Prune stale/mismatched market metadata cache entries
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

        while self._running:
            try:
                # Wait for the interval
                await asyncio.sleep(interval_hours * 3600)

                if not self._running:
                    break

                logger.info("Running scheduled database cleanup")

                # Run full cleanup with configured thresholds
                await self.full_cleanup(
                    resolved_trade_days=config.get("resolved_trade_days", self.DEFAULT_RESOLVED_TRADE_AGE),
                    open_trade_expiry_days=config.get("open_trade_expiry_days", self.DEFAULT_OPEN_TRADE_EXPIRY),
                    wallet_trade_days=config.get("wallet_trade_days", self.DEFAULT_WALLET_TRADE_AGE),
                    anomaly_days=config.get("anomaly_days", self.DEFAULT_ANOMALY_AGE),
                    llm_usage_retention_days=config.get("llm_usage_retention_days"),
                )

            except asyncio.CancelledError:
                logger.info("Background cleanup task cancelled")
                break
            except Exception as e:
                logger.error("Background cleanup failed", error=str(e))
                # Continue running, try again next interval

        logger.info("Background cleanup task stopped")

    def stop(self):
        """Stop the background cleanup task"""
        self._running = False


# Singleton instance
maintenance_service = MaintenanceService()
