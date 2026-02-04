import asyncio
from datetime import datetime
from typing import Optional, Callable, List

from config import settings
from models import Market, Event, ArbitrageOpportunity, OpportunityFilter
from models.database import AsyncSessionLocal, ScannerSettings
from services.polymarket import polymarket_client
from services.strategies import (
    BasicArbStrategy,
    NegRiskStrategy,
    MutuallyExclusiveStrategy,
    ContradictionStrategy,
    MustHappenStrategy,
    MiracleStrategy,
    CombinatorialStrategy
)
from sqlalchemy import select


class ArbitrageScanner:
    """Main scanner that orchestrates arbitrage detection"""

    def __init__(self):
        self.client = polymarket_client
        self.strategies = [
            BasicArbStrategy(),
            NegRiskStrategy(),          # Most profitable historically
            MutuallyExclusiveStrategy(),
            ContradictionStrategy(),
            MustHappenStrategy(),
            MiracleStrategy(),          # Swisstony's garbage collection strategy
            CombinatorialStrategy()     # Cross-market arbitrage via integer programming
        ]
        self._running = False
        self._enabled = True
        self._interval_seconds = settings.SCAN_INTERVAL_SECONDS
        self._last_scan: Optional[datetime] = None
        self._opportunities: list[ArbitrageOpportunity] = []
        self._scan_callbacks: List[Callable] = []
        self._status_callbacks: List[Callable] = []
        self._scan_task: Optional[asyncio.Task] = None

    def add_callback(self, callback: Callable):
        """Add callback to be notified of new opportunities"""
        self._scan_callbacks.append(callback)

    def add_status_callback(self, callback: Callable):
        """Add callback to be notified of scanner status changes"""
        self._status_callbacks.append(callback)

    async def _notify_status_change(self):
        """Notify all status callbacks of a change"""
        status = self.get_status()
        for callback in self._status_callbacks:
            try:
                await callback(status)
            except Exception as e:
                print(f"  Status callback error: {e}")

    async def load_settings(self):
        """Load scanner settings from database"""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(ScannerSettings).where(ScannerSettings.id == "default")
                )
                settings_row = result.scalar_one_or_none()

                if settings_row:
                    self._enabled = settings_row.is_enabled
                    self._interval_seconds = settings_row.scan_interval_seconds
                    print(f"Loaded scanner settings: enabled={self._enabled}, interval={self._interval_seconds}s")
                else:
                    # Create default settings
                    new_settings = ScannerSettings(
                        id="default",
                        is_enabled=True,
                        scan_interval_seconds=settings.SCAN_INTERVAL_SECONDS
                    )
                    session.add(new_settings)
                    await session.commit()
                    print("Created default scanner settings")
        except Exception as e:
            print(f"Error loading scanner settings: {e}")

    async def save_settings(self):
        """Save scanner settings to database"""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(ScannerSettings).where(ScannerSettings.id == "default")
                )
                settings_row = result.scalar_one_or_none()

                if settings_row:
                    settings_row.is_enabled = self._enabled
                    settings_row.scan_interval_seconds = self._interval_seconds
                    settings_row.updated_at = datetime.utcnow()
                else:
                    settings_row = ScannerSettings(
                        id="default",
                        is_enabled=self._enabled,
                        scan_interval_seconds=self._interval_seconds
                    )
                    session.add(settings_row)

                await session.commit()
                print(f"Saved scanner settings: enabled={self._enabled}, interval={self._interval_seconds}s")
        except Exception as e:
            print(f"Error saving scanner settings: {e}")

    async def scan_once(self) -> list[ArbitrageOpportunity]:
        """Perform a single scan for arbitrage opportunities"""
        print(f"[{datetime.utcnow().isoformat()}] Starting arbitrage scan...")

        try:
            # Fetch all active events and markets
            events = await self.client.get_all_events(closed=False)
            markets = await self.client.get_all_markets(active=True)

            print(f"  Fetched {len(events)} events and {len(markets)} markets")

            # Get live prices for all tokens
            all_token_ids = []
            for market in markets:
                all_token_ids.extend(market.clob_token_ids)

            # Batch price fetching (limit to avoid rate limits)
            prices = {}
            if all_token_ids:
                # Sample tokens if too many
                token_sample = all_token_ids[:500]
                prices = await self.client.get_prices_batch(token_sample)
                print(f"  Fetched prices for {len(prices)} tokens")

            # Run all strategies
            all_opportunities = []
            for strategy in self.strategies:
                try:
                    opps = strategy.detect(events, markets, prices)
                    all_opportunities.extend(opps)
                    print(f"  {strategy.name}: found {len(opps)} opportunities")
                except Exception as e:
                    print(f"  {strategy.name}: error - {e}")

            # Sort by ROI
            all_opportunities.sort(key=lambda x: x.roi_percent, reverse=True)

            self._opportunities = all_opportunities
            self._last_scan = datetime.utcnow()

            # Notify callbacks
            for callback in self._scan_callbacks:
                try:
                    await callback(all_opportunities)
                except Exception as e:
                    print(f"  Callback error: {e}")

            print(f"[{datetime.utcnow().isoformat()}] Scan complete. {len(all_opportunities)} total opportunities")
            return all_opportunities

        except Exception as e:
            print(f"[{datetime.utcnow().isoformat()}] Scan error: {e}")
            raise

    async def _scan_loop(self):
        """Internal scan loop"""
        while self._running:
            if self._enabled:
                try:
                    await self.scan_once()
                except Exception as e:
                    print(f"Scan error: {e}")

            await asyncio.sleep(self._interval_seconds)

    async def start_continuous_scan(self, interval_seconds: int = None):
        """Start continuous scanning loop"""
        # Load persisted settings first
        await self.load_settings()

        if interval_seconds is not None:
            self._interval_seconds = interval_seconds

        self._running = True
        print(f"Starting continuous scan (interval: {self._interval_seconds}s, enabled: {self._enabled})")

        # Run the scan loop
        await self._scan_loop()

    async def start(self):
        """Enable scanning"""
        self._enabled = True
        await self.save_settings()
        await self._notify_status_change()

        # If not running, do an immediate scan
        if self._running:
            await self.scan_once()

    async def pause(self):
        """Pause scanning (keeps loop running but doesn't scan)"""
        self._enabled = False
        await self.save_settings()
        await self._notify_status_change()

    def stop(self):
        """Stop continuous scanning loop completely"""
        self._running = False
        self._enabled = False

    async def set_interval(self, seconds: int):
        """Update scan interval"""
        if seconds < 10:
            seconds = 10  # Minimum 10 seconds
        if seconds > 3600:
            seconds = 3600  # Maximum 1 hour

        self._interval_seconds = seconds
        await self.save_settings()
        await self._notify_status_change()

    def get_status(self) -> dict:
        """Get current scanner status"""
        return {
            "running": self._running,
            "enabled": self._enabled,
            "interval_seconds": self._interval_seconds,
            "last_scan": (self._last_scan.isoformat() + "Z") if self._last_scan else None,
            "opportunities_count": len(self._opportunities),
            "strategies": [
                {"name": s.name, "type": s.strategy_type.value}
                for s in self.strategies
            ]
        }

    def get_opportunities(
        self,
        filter: Optional[OpportunityFilter] = None
    ) -> list[ArbitrageOpportunity]:
        """Get current opportunities with optional filtering"""
        opps = self._opportunities

        if filter:
            if filter.min_profit > 0:
                opps = [o for o in opps if o.roi_percent >= filter.min_profit * 100]
            if filter.max_risk < 1.0:
                opps = [o for o in opps if o.risk_score <= filter.max_risk]
            if filter.strategies:
                opps = [o for o in opps if o.strategy in filter.strategies]
            if filter.min_liquidity > 0:
                opps = [o for o in opps if o.min_liquidity >= filter.min_liquidity]

        return opps

    @property
    def last_scan(self) -> Optional[datetime]:
        return self._last_scan

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def interval_seconds(self) -> int:
        return self._interval_seconds

    def clear_opportunities(self) -> int:
        """Clear all opportunities from memory. Returns count of cleared opportunities."""
        count = len(self._opportunities)
        self._opportunities = []
        print(f"Cleared {count} opportunities from memory")
        return count

    def remove_expired_opportunities(self) -> int:
        """Remove opportunities whose resolution date has passed. Returns count removed."""
        now = datetime.utcnow()
        before_count = len(self._opportunities)

        self._opportunities = [
            opp for opp in self._opportunities
            if opp.resolution_date is None or opp.resolution_date > now
        ]

        removed = before_count - len(self._opportunities)
        if removed > 0:
            print(f"Removed {removed} expired opportunities")
        return removed

    def remove_old_opportunities(self, max_age_minutes: int = 60) -> int:
        """Remove opportunities older than max_age_minutes. Returns count removed."""
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(minutes=max_age_minutes)
        before_count = len(self._opportunities)

        self._opportunities = [
            opp for opp in self._opportunities
            if opp.detected_at >= cutoff
        ]

        removed = before_count - len(self._opportunities)
        if removed > 0:
            print(f"Removed {removed} opportunities older than {max_age_minutes} minutes")
        return removed


# Singleton instance
scanner = ArbitrageScanner()
