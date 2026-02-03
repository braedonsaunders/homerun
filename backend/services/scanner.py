import asyncio
from datetime import datetime
from typing import Optional

from config import settings
from models import Market, Event, ArbitrageOpportunity, OpportunityFilter
from services.polymarket import polymarket_client
from services.strategies import (
    BasicArbStrategy,
    NegRiskStrategy,
    MutuallyExclusiveStrategy,
    ContradictionStrategy,
    MustHappenStrategy,
    MiracleStrategy
)


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
            MiracleStrategy()           # Swisstony's garbage collection strategy
        ]
        self._running = False
        self._last_scan: Optional[datetime] = None
        self._opportunities: list[ArbitrageOpportunity] = []
        self._scan_callbacks: list[callable] = []

    def add_callback(self, callback: callable):
        """Add callback to be notified of new opportunities"""
        self._scan_callbacks.append(callback)

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

    async def start_continuous_scan(self, interval_seconds: int = None):
        """Start continuous scanning loop"""
        if interval_seconds is None:
            interval_seconds = settings.SCAN_INTERVAL_SECONDS

        self._running = True
        print(f"Starting continuous scan (interval: {interval_seconds}s)")

        while self._running:
            try:
                await self.scan_once()
            except Exception as e:
                print(f"Scan error: {e}")

            await asyncio.sleep(interval_seconds)

    def stop(self):
        """Stop continuous scanning"""
        self._running = False

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


# Singleton instance
scanner = ArbitrageScanner()
