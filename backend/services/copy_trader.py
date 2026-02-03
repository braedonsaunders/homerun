import asyncio
import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    CopyTradingConfig, TrackedWallet, WalletTrade,
    SimulationAccount, AsyncSessionLocal
)
from models.opportunity import ArbitrageOpportunity
from services.polymarket import polymarket_client
from services.scanner import scanner
from services.simulation import simulation_service
from utils.logger import get_logger

logger = get_logger("copy_trader")


class CopyTradingService:
    """Service for copy trading from profitable wallets"""

    def __init__(self):
        self._running = False
        self._poll_interval = 30  # seconds
        self._active_configs: dict[str, CopyTradingConfig] = {}

    async def add_copy_config(
        self,
        source_wallet: str,
        account_id: str,
        min_roi_threshold: float = 2.5,
        max_position_size: float = 1000.0,
        copy_delay_seconds: int = 5,
        slippage_tolerance: float = 1.0
    ) -> CopyTradingConfig:
        """Add a wallet to copy trade"""
        async with AsyncSessionLocal() as session:
            # Verify account exists
            account = await session.get(SimulationAccount, account_id)
            if not account:
                raise ValueError(f"Account not found: {account_id}")

            config = CopyTradingConfig(
                id=str(uuid.uuid4()),
                source_wallet=source_wallet.lower(),
                account_id=account_id,
                enabled=True,
                min_roi_threshold=min_roi_threshold,
                max_position_size=max_position_size,
                copy_delay_seconds=copy_delay_seconds,
                slippage_tolerance=slippage_tolerance
            )
            session.add(config)

            # Also ensure wallet is tracked
            wallet = await session.get(TrackedWallet, source_wallet.lower())
            if not wallet:
                wallet = TrackedWallet(
                    address=source_wallet.lower(),
                    label=f"Copy Target"
                )
                session.add(wallet)

            await session.commit()
            await session.refresh(config)

            self._active_configs[config.id] = config

            logger.info(
                "Added copy trading config",
                config_id=config.id,
                source_wallet=source_wallet,
                account_id=account_id
            )

            return config

    async def remove_copy_config(self, config_id: str):
        """Remove a copy trading configuration"""
        async with AsyncSessionLocal() as session:
            config = await session.get(CopyTradingConfig, config_id)
            if config:
                await session.delete(config)
                await session.commit()
                self._active_configs.pop(config_id, None)

                logger.info("Removed copy trading config", config_id=config_id)

    async def get_configs(self, account_id: Optional[str] = None) -> list[CopyTradingConfig]:
        """Get all copy trading configurations"""
        async with AsyncSessionLocal() as session:
            query = select(CopyTradingConfig)
            if account_id:
                query = query.where(CopyTradingConfig.account_id == account_id)

            result = await session.execute(query)
            return list(result.scalars().all())

    async def enable_config(self, config_id: str, enabled: bool):
        """Enable or disable a copy trading configuration"""
        async with AsyncSessionLocal() as session:
            config = await session.get(CopyTradingConfig, config_id)
            if config:
                config.enabled = enabled
                await session.commit()

                logger.info(
                    "Updated copy trading config",
                    config_id=config_id,
                    enabled=enabled
                )

    async def _check_wallet_for_opportunities(
        self,
        wallet_address: str,
        config: CopyTradingConfig
    ) -> list[dict]:
        """Check if a wallet has entered any arbitrage opportunities"""
        try:
            # Get recent trades from wallet
            trades = await polymarket_client.get_wallet_trades(wallet_address, limit=20)

            # Get current opportunities
            opportunities = scanner.get_opportunities()

            matches = []
            for trade in trades:
                # Check if trade matches any known opportunity
                for opp in opportunities:
                    if self._trade_matches_opportunity(trade, opp):
                        # Check ROI threshold
                        if opp.roi_percent >= config.min_roi_threshold:
                            matches.append({
                                "trade": trade,
                                "opportunity": opp,
                                "config": config
                            })

            return matches

        except Exception as e:
            logger.error(
                "Error checking wallet for opportunities",
                wallet=wallet_address,
                error=str(e)
            )
            return []

    def _trade_matches_opportunity(
        self,
        trade: dict,
        opportunity: ArbitrageOpportunity
    ) -> bool:
        """Check if a wallet trade matches a detected opportunity"""
        trade_market = trade.get("market", trade.get("condition_id", ""))

        for market in opportunity.markets:
            if market.get("id") == trade_market:
                return True

        return False

    async def _execute_copy(
        self,
        match: dict
    ) -> bool:
        """Execute a copy trade"""
        config: CopyTradingConfig = match["config"]
        opportunity: ArbitrageOpportunity = match["opportunity"]
        source_trade = match["trade"]

        try:
            # Wait for configured delay
            if config.copy_delay_seconds > 0:
                logger.info(
                    "Waiting before copy trade",
                    delay=config.copy_delay_seconds,
                    opportunity_id=opportunity.id
                )
                await asyncio.sleep(config.copy_delay_seconds)

            # Re-check opportunity is still valid
            current_opps = scanner.get_opportunities()
            still_valid = any(o.id == opportunity.id for o in current_opps)

            if not still_valid:
                logger.warning(
                    "Opportunity no longer valid for copy",
                    opportunity_id=opportunity.id
                )
                async with AsyncSessionLocal() as session:
                    db_config = await session.get(CopyTradingConfig, config.id)
                    if db_config:
                        db_config.failed_copies += 1
                        await session.commit()
                return False

            # Execute in simulation
            position_size = min(
                config.max_position_size,
                opportunity.max_position_size
            )

            trade = await simulation_service.execute_opportunity(
                account_id=config.account_id,
                opportunity=opportunity,
                position_size=position_size,
                copied_from=config.source_wallet
            )

            # Update config stats
            async with AsyncSessionLocal() as session:
                db_config = await session.get(CopyTradingConfig, config.id)
                if db_config:
                    db_config.total_copied += 1
                    db_config.successful_copies += 1
                    await session.commit()

            logger.info(
                "Successfully copied trade",
                config_id=config.id,
                source_wallet=config.source_wallet,
                trade_id=trade.id,
                opportunity_id=opportunity.id,
                position_size=position_size
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to copy trade",
                config_id=config.id,
                opportunity_id=opportunity.id,
                error=str(e)
            )

            async with AsyncSessionLocal() as session:
                db_config = await session.get(CopyTradingConfig, config.id)
                if db_config:
                    db_config.failed_copies += 1
                    await session.commit()

            return False

    async def _poll_loop(self):
        """Main polling loop for copy trading"""
        while self._running:
            try:
                # Get all enabled configs
                async with AsyncSessionLocal() as session:
                    result = await session.execute(
                        select(CopyTradingConfig).where(CopyTradingConfig.enabled == True)
                    )
                    configs = list(result.scalars().all())

                # Check each wallet
                for config in configs:
                    matches = await self._check_wallet_for_opportunities(
                        config.source_wallet,
                        config
                    )

                    for match in matches:
                        await self._execute_copy(match)

            except Exception as e:
                logger.error("Copy trading poll error", error=str(e))

            await asyncio.sleep(self._poll_interval)

    async def start(self):
        """Start copy trading service"""
        if self._running:
            return

        self._running = True
        logger.info("Starting copy trading service")

        # Load existing configs
        configs = await self.get_configs()
        for config in configs:
            self._active_configs[config.id] = config

        asyncio.create_task(self._poll_loop())

    def stop(self):
        """Stop copy trading service"""
        self._running = False
        logger.info("Stopped copy trading service")

    async def get_copy_stats(self, config_id: str) -> dict:
        """Get statistics for a copy trading configuration"""
        async with AsyncSessionLocal() as session:
            config = await session.get(CopyTradingConfig, config_id)
            if not config:
                return None

            success_rate = (
                config.successful_copies / config.total_copied * 100
                if config.total_copied > 0 else 0
            )

            return {
                "config_id": config.id,
                "source_wallet": config.source_wallet,
                "account_id": config.account_id,
                "enabled": config.enabled,
                "total_copied": config.total_copied,
                "successful_copies": config.successful_copies,
                "failed_copies": config.failed_copies,
                "success_rate": success_rate,
                "total_pnl": config.total_pnl,
                "settings": {
                    "min_roi_threshold": config.min_roi_threshold,
                    "max_position_size": config.max_position_size,
                    "copy_delay_seconds": config.copy_delay_seconds,
                    "slippage_tolerance": config.slippage_tolerance
                }
            }


# Singleton instance
copy_trader = CopyTradingService()
