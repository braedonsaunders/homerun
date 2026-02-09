import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, Callable, List

from config import settings
from models import ArbitrageOpportunity, OpportunityFilter
from models.opportunity import AIAnalysis, MispricingType
from models.database import AsyncSessionLocal, ScannerSettings, OpportunityJudgment
from services.polymarket import polymarket_client
from services.kalshi_client import kalshi_client
from services.strategies import (
    BasicArbStrategy,
    NegRiskStrategy,
    MutuallyExclusiveStrategy,
    ContradictionStrategy,
    MustHappenStrategy,
    MiracleStrategy,
    CombinatorialStrategy,
    SettlementLagStrategy,
    BtcEthHighFreqStrategy,
    NewsEdgeStrategy,
    CrossPlatformStrategy,
    BayesianCascadeStrategy,
    LiquidityVacuumStrategy,
    EntropyArbStrategy,
    EventDrivenStrategy,
    TemporalDecayStrategy,
    CorrelationArbStrategy,
    MarketMakingStrategy,
    StatArbStrategy,
)
from services.pause_state import global_pause_state
from services.market_prioritizer import market_prioritizer, MarketTier
from sqlalchemy import select


def _make_aware(dt: Optional[datetime]) -> Optional[datetime]:
    """Ensure a datetime is timezone-aware (UTC). Returns None for None input."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class ArbitrageScanner:
    """Main scanner that orchestrates arbitrage detection"""

    def __init__(self):
        self.client = polymarket_client
        self.strategies = [
            BasicArbStrategy(),
            NegRiskStrategy(),  # Most profitable historically
            MutuallyExclusiveStrategy(),
            ContradictionStrategy(),
            MustHappenStrategy(),
            MiracleStrategy(),  # Swisstony's garbage collection strategy
            CombinatorialStrategy(),  # Cross-market arbitrage via integer programming
            SettlementLagStrategy(),  # Exploit delayed price adjustments (article Part IV)
            BtcEthHighFreqStrategy(),  # BTC/ETH 15min/1hr high-frequency arb
            CrossPlatformStrategy(),  # Cross-platform arb (Polymarket vs Kalshi)
            BayesianCascadeStrategy(),  # Probability graph belief propagation
            LiquidityVacuumStrategy(),  # Order book imbalance exploitation
            EntropyArbStrategy(),  # Information-theoretic mispricing detection
            EventDrivenStrategy(),  # Price lag after catalyst moves
            TemporalDecayStrategy(),  # Time-decay mispricing in deadline markets
            CorrelationArbStrategy(),  # Mean-reversion on correlated pair spreads
            MarketMakingStrategy(),  # Earn bid-ask spread as liquidity provider
            StatArbStrategy(),  # Statistical edge from ensemble probability signals
        ]

        # Async strategies (require network I/O or LLM calls, run separately)
        self._news_edge_strategy = NewsEdgeStrategy()

        # Mispricing type mapping for strategies that don't set it themselves
        self._strategy_mispricing_map = {
            "basic": MispricingType.WITHIN_MARKET,
            "negrisk": MispricingType.WITHIN_MARKET,
            "mutually_exclusive": MispricingType.WITHIN_MARKET,
            "contradiction": MispricingType.WITHIN_MARKET,
            "must_happen": MispricingType.WITHIN_MARKET,
            "miracle": MispricingType.WITHIN_MARKET,
            "combinatorial": MispricingType.CROSS_MARKET,
            "settlement_lag": MispricingType.SETTLEMENT_LAG,
            "btc_eth_highfreq": MispricingType.WITHIN_MARKET,
            "news_edge": MispricingType.NEWS_INFORMATION,
            "cross_platform": MispricingType.CROSS_MARKET,
            "bayesian_cascade": MispricingType.CROSS_MARKET,
            "liquidity_vacuum": MispricingType.WITHIN_MARKET,
            "entropy_arb": MispricingType.WITHIN_MARKET,
            "event_driven": MispricingType.CROSS_MARKET,
            "temporal_decay": MispricingType.WITHIN_MARKET,
            "correlation_arb": MispricingType.CROSS_MARKET,
            "market_making": MispricingType.WITHIN_MARKET,
            "stat_arb": MispricingType.WITHIN_MARKET,
        }
        self._running = False
        self._enabled = True
        self._interval_seconds = settings.SCAN_INTERVAL_SECONDS
        self._last_scan: Optional[datetime] = None
        self._last_full_scan: Optional[datetime] = None
        self._last_fast_scan: Optional[datetime] = None
        self._opportunities: list[ArbitrageOpportunity] = []
        self._scan_callbacks: List[Callable] = []
        self._status_callbacks: List[Callable] = []
        self._activity_callbacks: List[Callable] = []
        self._scan_task: Optional[asyncio.Task] = None

        # Live scanning activity line (streamed to frontend via WebSocket)
        self._current_activity: str = "Idle"

        # Track the running AI scoring task so we can cancel it on pause
        self._ai_scoring_task: Optional[asyncio.Task] = None

        # Auto AI scoring: when False (default), the scanner does NOT
        # automatically score all opportunities with LLM after each scan.
        # Manual per-opportunity analysis (via the Analyze button) still works.
        self._auto_ai_scoring: bool = False

        # Tiered scanning: track scan cycles for fast/full alternation
        self._fast_scan_cycle: int = 0
        self._prioritizer = market_prioritizer

        # Cached full market/event data for use between full scans
        self._cached_events: list = []
        self._cached_markets: list = []
        self._cached_prices: dict = {}

    def set_auto_ai_scoring(self, enabled: bool):
        """Enable or disable automatic AI scoring of all opportunities after each scan."""
        self._auto_ai_scoring = enabled
        print(f"Auto AI scoring {'enabled' if enabled else 'disabled'}")

    @property
    def auto_ai_scoring(self) -> bool:
        return self._auto_ai_scoring

    def add_callback(self, callback: Callable):
        """Add callback to be notified of new opportunities"""
        self._scan_callbacks.append(callback)

    def add_status_callback(self, callback: Callable):
        """Add callback to be notified of scanner status changes"""
        self._status_callbacks.append(callback)

    def add_activity_callback(self, callback: Callable):
        """Add callback to be notified of scanning activity updates"""
        self._activity_callbacks.append(callback)

    async def _set_activity(self, activity: str):
        """Update the current scanning activity and broadcast to clients."""
        self._current_activity = activity
        for cb in self._activity_callbacks:
            try:
                await cb(activity)
            except Exception:
                pass

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
                    # Sync global pause state with persisted setting
                    if self._enabled:
                        global_pause_state.resume()
                    else:
                        global_pause_state.pause()
                    print(
                        f"Loaded scanner settings: enabled={self._enabled}, interval={self._interval_seconds}s"
                    )
                else:
                    # Create default settings
                    new_settings = ScannerSettings(
                        id="default",
                        is_enabled=True,
                        scan_interval_seconds=settings.SCAN_INTERVAL_SECONDS,
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
                        scan_interval_seconds=self._interval_seconds,
                    )
                    session.add(settings_row)

                await session.commit()
                print(
                    f"Saved scanner settings: enabled={self._enabled}, interval={self._interval_seconds}s"
                )
        except Exception as e:
            print(f"Error saving scanner settings: {e}")

    async def scan_once(self) -> list[ArbitrageOpportunity]:
        """Perform a single scan for arbitrage opportunities"""
        print(f"[{datetime.utcnow().isoformat()}] Starting arbitrage scan...")
        await self._set_activity("Fetching Polymarket events and markets...")

        try:
            # Fetch events and markets concurrently (they are independent)
            events, markets = await asyncio.gather(
                self.client.get_all_events(closed=False),
                self.client.get_all_markets(active=True),
            )

            # Filter out markets whose end_date has already passed —
            # these are resolved events awaiting settlement and can't
            # be traded profitably.
            now = datetime.now(timezone.utc)
            markets = [
                m
                for m in markets
                if m.end_date is None or _make_aware(m.end_date) > now
            ]

            # Also prune expired markets inside events so strategies
            # like NegRisk that iterate event.markets don't pick them up.
            for event in events:
                event.markets = [
                    m
                    for m in event.markets
                    if m.end_date is None or _make_aware(m.end_date) > now
                ]

            print(
                f"  Fetched {len(events)} Polymarket events and {len(markets)} markets"
            )
            await self._set_activity(
                f"Fetched {len(events)} events, {len(markets)} markets"
            )

            # Fetch Kalshi markets and merge them so ALL strategies
            # (not just cross-platform) can detect opportunities on Kalshi.
            kalshi_market_count = 0
            kalshi_event_count = 0
            if settings.CROSS_PLATFORM_ENABLED:
                await self._set_activity("Fetching Kalshi markets...")
                try:
                    kalshi_markets = await kalshi_client.get_all_markets(active=True)
                    kalshi_markets = [
                        m
                        for m in kalshi_markets
                        if m.end_date is None or _make_aware(m.end_date) > now
                    ]
                    kalshi_market_count = len(kalshi_markets)
                    markets.extend(kalshi_markets)

                    kalshi_events = await kalshi_client.get_all_events(closed=False)
                    for ke in kalshi_events:
                        ke.markets = [
                            m
                            for m in ke.markets
                            if m.end_date is None or _make_aware(m.end_date) > now
                        ]
                    kalshi_event_count = len(kalshi_events)
                    events.extend(kalshi_events)

                    if kalshi_market_count > 0:
                        print(
                            f"  Fetched {kalshi_event_count} Kalshi events "
                            f"and {kalshi_market_count} Kalshi markets"
                        )
                        await self._set_activity(
                            f"Fetched {kalshi_event_count} Kalshi events, {kalshi_market_count} markets"
                        )
                except Exception as e:
                    print(f"  Kalshi fetch failed (non-fatal): {e}")

            # Get live prices for Polymarket tokens only.
            # Kalshi markets already have prices baked in from the API;
            # their synthetic token IDs (e.g. "TICKER_yes") should not
            # be sent to Polymarket's CLOB.
            all_token_ids = []
            for market in markets:
                for tid in market.clob_token_ids:
                    # Polymarket CLOB token IDs are long hex strings (>20 chars).
                    # Kalshi synthetic IDs are short like "TICKER_yes".
                    if len(tid) > 20:
                        all_token_ids.append(tid)

            # Batch price fetching (limit to avoid rate limits)
            prices = {}
            if all_token_ids:
                await self._set_activity(
                    f"Fetching prices for {min(len(all_token_ids), 500)} tokens..."
                )
                # Sample tokens if too many
                token_sample = all_token_ids[:500]
                prices = await self.client.get_prices_batch(token_sample)
                print(f"  Fetched prices for {len(prices)} tokens")

            # Cache full data for fast scans between full scans
            self._cached_events = list(events)
            self._cached_markets = list(markets)
            self._cached_prices = dict(prices)

            # Tiered scanning: classify markets and apply change detection
            markets_to_evaluate = markets
            unchanged_count = 0
            if settings.TIERED_SCANNING_ENABLED:
                try:
                    # Update stability scores from MarketMonitor
                    self._prioritizer.update_stability_scores()

                    # Classify all markets into tiers
                    tier_map = self._prioritizer.classify_all(markets, now)

                    # Compute volume/liquidity attention scores
                    self._prioritizer.compute_attention_scores(markets)

                    # Change detection: filter out markets with identical prices
                    changed = self._prioritizer.get_changed_markets(markets)
                    unchanged_count = len(markets) - len(changed)

                    # For full scans, still run all markets through strategies
                    # but log the change detection stats. The real savings come
                    # from fast scans where we only evaluate changed markets.
                    # However, for COLD-tier markets, we DO skip them if unchanged.
                    cold_ids = {m.id for m in tier_map[MarketTier.COLD]}
                    markets_to_evaluate = [
                        m
                        for m in markets
                        if m.id not in cold_ids
                        or self._prioritizer.has_market_changed(m)
                    ]
                    cold_skipped = len(markets) - len(markets_to_evaluate)

                    print(
                        f"  Tiers: {len(tier_map[MarketTier.HOT])} hot, "
                        f"{len(tier_map[MarketTier.WARM])} warm, "
                        f"{len(tier_map[MarketTier.COLD])} cold "
                        f"({cold_skipped} unchanged cold skipped)"
                    )
                except Exception as e:
                    print(f"  Prioritizer error (non-fatal, using all markets): {e}")
                    markets_to_evaluate = markets

            # Run all strategies concurrently in a thread pool.
            # strategy.detect() is synchronous CPU-bound work; running it
            # on the event loop blocks all async handlers (including
            # GET /api/opportunities) causing the UI to hang during scans.
            all_opportunities = []
            loop = asyncio.get_running_loop()
            await self._set_activity(f"Running {len(self.strategies)} strategies...")

            async def _run_strategy(strategy):
                """Run a single strategy in the default thread-pool executor."""
                return strategy, await loop.run_in_executor(
                    None, strategy.detect, events, markets_to_evaluate, prices
                )

            results = await asyncio.gather(
                *[_run_strategy(s) for s in self.strategies],
                return_exceptions=True,
            )

            for result in results:
                if isinstance(result, Exception):
                    print(f"  Strategy error: {result}")
                    continue
                strategy, opps = result
                # Classify mispricing type if not already set by strategy
                for opp in opps:
                    if opp.mispricing_type is None:
                        opp.mispricing_type = self._strategy_mispricing_map.get(
                            opp.strategy.value, MispricingType.WITHIN_MARKET
                        )
                all_opportunities.extend(opps)
                print(f"  {strategy.name}: found {len(opps)} opportunities")

            # Update prioritizer state after evaluation
            if settings.TIERED_SCANNING_ENABLED:
                try:
                    self._prioritizer.update_after_evaluation(markets, now)
                except Exception:
                    pass

            # Run async strategies (News Edge — requires LLM calls)
            if settings.NEWS_EDGE_ENABLED:
                try:
                    news_opps = await self._news_edge_strategy.detect_async(
                        events, markets, prices
                    )
                    for opp in news_opps:
                        if opp.mispricing_type is None:
                            opp.mispricing_type = MispricingType.NEWS_INFORMATION
                    all_opportunities.extend(news_opps)
                    print(f"  {self._news_edge_strategy.name}: found {len(news_opps)} opportunities")
                except Exception as e:
                    print(f"  {self._news_edge_strategy.name}: error - {e}")

            # Deduplicate across strategies: when the same underlying markets
            # are detected by multiple strategies, keep only the highest-ROI one.
            all_opportunities = self._deduplicate_cross_strategy(all_opportunities)

            # Sort by ROI
            all_opportunities.sort(key=lambda x: x.roi_percent, reverse=True)

            # Attach existing AI judgments from the database
            await self._attach_ai_judgments(all_opportunities)

            self._opportunities = self._merge_opportunities(all_opportunities)
            self._last_scan = datetime.now(timezone.utc)

            # AI Intelligence: Score unscored opportunities (non-blocking)
            # Only run if auto_ai_scoring is enabled (opt-in, default OFF).
            # Manual per-opportunity analysis is always available via the UI.
            if self._auto_ai_scoring:
                try:
                    from services.ai import get_llm_manager

                    manager = get_llm_manager()
                    if manager.is_available():
                        # Cancel any still-running scoring task from a previous scan
                        if self._ai_scoring_task and not self._ai_scoring_task.done():
                            self._ai_scoring_task.cancel()
                            try:
                                await self._ai_scoring_task
                            except (asyncio.CancelledError, Exception):
                                pass
                        # Score the full merged pool so retained opps get scored too
                        self._ai_scoring_task = asyncio.create_task(
                            self._ai_score_opportunities(self._opportunities)
                        )
                except Exception:
                    pass  # AI scoring is non-critical

            # Notify callbacks (pass only newly detected opportunities)
            for callback in self._scan_callbacks:
                try:
                    await callback(all_opportunities)
                except Exception as e:
                    print(f"  Callback error: {e}")

            self._last_full_scan = now

            scan_suffix = ""
            if settings.TIERED_SCANNING_ENABLED and unchanged_count > 0:
                scan_suffix = f" ({unchanged_count} unchanged markets detected)"

            print(
                f"[{datetime.utcnow().isoformat()}] Scan complete. "
                f"{len(all_opportunities)} detected this scan, "
                f"{len(self._opportunities)} total in pool{scan_suffix}"
            )
            await self._set_activity(
                f"Scan complete — {len(all_opportunities)} found, "
                f"{len(self._opportunities)} total in pool"
            )
            return self._opportunities

        except Exception as e:
            print(f"[{datetime.utcnow().isoformat()}] Scan error: {e}")
            await self._set_activity(f"Scan error: {e}")
            raise

    async def scan_fast(self) -> list[ArbitrageOpportunity]:
        """Fast-path scan targeting only HOT-tier and changed markets.

        This runs every FAST_SCAN_INTERVAL_SECONDS (default 15s) between
        full scans. It:
          1. Incrementally fetches only recently created markets (delta)
          2. Uses cached data from the last full scan for existing markets
          3. Re-fetches prices only for HOT-tier markets
          4. Runs strategies only on markets whose prices have changed
          5. Merges results into the main opportunity pool

        Much cheaper than a full scan: fewer API calls, fewer strategy runs.
        """
        now = datetime.now(timezone.utc)
        print(f"[{now.isoformat()}] Starting fast scan (hot-tier + incremental)...")
        await self._set_activity("Fast scan: fetching recent markets...")

        try:
            # 1. Incremental fetch: get only recently created markets
            new_markets: list = []
            if settings.INCREMENTAL_FETCH_ENABLED:
                try:
                    new_markets = await self.client.get_recent_markets(since_minutes=5)
                    if new_markets:
                        print(
                            f"  Incremental: {len(new_markets)} recently created markets"
                        )
                except Exception as e:
                    print(f"  Incremental fetch failed (non-fatal): {e}")

            # 2. Merge incremental markets into cached data
            cached_market_ids = {m.id for m in self._cached_markets}
            truly_new = [m for m in new_markets if m.id not in cached_market_ids]
            if truly_new:
                self._cached_markets.extend(truly_new)
                print(f"  Added {len(truly_new)} brand-new markets to cache")

            # 3. Update MarketMonitor with the new markets to generate alerts
            try:
                from services.market_monitor import market_monitor

                await market_monitor.get_fresh_opportunities()
            except Exception:
                pass

            # 4. Classify all cached markets into tiers
            self._prioritizer.update_stability_scores()
            tier_map = self._prioritizer.classify_all(self._cached_markets, now)
            hot_markets = tier_map[MarketTier.HOT]

            if not hot_markets:
                print("  No HOT-tier markets, skipping fast scan")
                self._last_fast_scan = now
                return self._opportunities

            # 5. Re-fetch prices for HOT-tier markets only
            hot_token_ids = []
            for market in hot_markets:
                for tid in market.clob_token_ids:
                    if len(tid) > 20:  # Polymarket tokens only
                        hot_token_ids.append(tid)

            hot_prices = {}
            if hot_token_ids:
                await self._set_activity(
                    f"Fast scan: fetching prices for {min(len(hot_token_ids), 200)} hot-tier tokens..."
                )
                hot_prices = await self.client.get_prices_batch(hot_token_ids[:200])
                print(f"  Fetched prices for {len(hot_prices)} hot-tier tokens")

            # Merge with cached prices
            merged_prices = {**self._cached_prices, **hot_prices}

            # 6. Change detection: only evaluate markets whose prices moved
            changed_markets = self._prioritizer.get_changed_markets(hot_markets)
            if not changed_markets:
                print(
                    f"  All {len(hot_markets)} hot-tier markets unchanged, skipping strategies"
                )
                await self._set_activity(
                    f"Fast scan: {len(hot_markets)} markets unchanged, skipping"
                )
                self._prioritizer.update_after_evaluation(hot_markets, now)
                self._last_scan = now
                self._last_fast_scan = now
                return self._opportunities

            print(
                f"  {len(changed_markets)}/{len(hot_markets)} hot-tier markets have price changes"
            )
            await self._set_activity(
                f"Fast scan: running strategies on {len(changed_markets)} changed markets..."
            )

            # 7. Run strategies on changed markets only
            # Use the full cached events/markets for context, but strategies
            # will find opportunities primarily in the changed subset
            all_markets_for_strategies = self._cached_markets
            events_for_strategies = self._cached_events

            # Filter expired
            all_markets_for_strategies = [
                m
                for m in all_markets_for_strategies
                if m.end_date is None or _make_aware(m.end_date) > now
            ]

            loop = asyncio.get_running_loop()
            fast_opportunities = []

            async def _run_strategy(strategy):
                return strategy, await loop.run_in_executor(
                    None,
                    strategy.detect,
                    events_for_strategies,
                    all_markets_for_strategies,
                    merged_prices,
                )

            results = await asyncio.gather(
                *[_run_strategy(s) for s in self.strategies],
                return_exceptions=True,
            )

            for result in results:
                if isinstance(result, Exception):
                    continue
                strategy, opps = result
                for opp in opps:
                    if opp.mispricing_type is None:
                        opp.mispricing_type = self._strategy_mispricing_map.get(
                            opp.strategy.value, MispricingType.WITHIN_MARKET
                        )
                fast_opportunities.extend(opps)

            fast_opportunities = self._deduplicate_cross_strategy(fast_opportunities)
            fast_opportunities.sort(key=lambda x: x.roi_percent, reverse=True)

            # 8. Update prioritizer state
            unchanged = self._prioritizer.update_after_evaluation(hot_markets, now)
            self._prioritizer.compute_attention_scores(hot_markets)

            # 9. Merge into main pool
            if fast_opportunities:
                await self._attach_ai_judgments(fast_opportunities)
                self._opportunities = self._merge_opportunities(fast_opportunities)

            self._last_scan = now
            self._last_fast_scan = now
            self._fast_scan_cycle += 1

            # Notify callbacks
            for callback in self._scan_callbacks:
                try:
                    await callback(fast_opportunities)
                except Exception as e:
                    print(f"  Callback error: {e}")

            print(
                f"[{now.isoformat()}] Fast scan complete. "
                f"{len(fast_opportunities)} detected, "
                f"{len(self._opportunities)} total in pool "
                f"({unchanged} unchanged markets skipped)"
            )
            await self._set_activity(
                f"Fast scan complete — {len(fast_opportunities)} found, "
                f"{len(self._opportunities)} total"
            )
            return self._opportunities

        except Exception as e:
            print(f"[{datetime.utcnow().isoformat()}] Fast scan error: {e}")
            await self._set_activity(f"Fast scan error: {e}")
            raise

    def _deduplicate_cross_strategy(
        self, opportunities: list[ArbitrageOpportunity]
    ) -> list[ArbitrageOpportunity]:
        """Remove duplicate opportunities that cover the same underlying markets.

        When multiple strategies detect the same set of markets (e.g., NegRisk
        and settlement_lag both find MI-07 House Election), keep only the one
        with the highest ROI. This prevents the same trade from appearing
        multiple times in the output.

        Uses a market-ID fingerprint: sorted set of market IDs involved.
        """
        seen: dict[str, ArbitrageOpportunity] = {}  # fingerprint -> best opp

        for opp in opportunities:
            # Build a fingerprint from the sorted market IDs
            market_ids = sorted(m.get("id", "") for m in opp.markets)
            fingerprint = "|".join(market_ids)

            if fingerprint in seen:
                # Keep the opportunity with higher ROI
                if opp.roi_percent > seen[fingerprint].roi_percent:
                    seen[fingerprint] = opp
            else:
                seen[fingerprint] = opp

        deduped = list(seen.values())
        removed = len(opportunities) - len(deduped)
        if removed > 0:
            print(f"  Deduplicated: removed {removed} cross-strategy duplicates")
        return deduped

    def _merge_opportunities(
        self, new_opportunities: list[ArbitrageOpportunity]
    ) -> list[ArbitrageOpportunity]:
        """Merge newly detected opportunities into the existing pool.

        Instead of replacing all opportunities on each scan, this method:
        - Adds newly discovered opportunities to the pool
        - Updates existing opportunities (matched by stable_id) with fresh
          market data while preserving original detection time and AI analysis
        - Removes expired opportunities whose resolution date has passed
        """
        now = datetime.now(timezone.utc)

        # Index existing opportunities by stable_id
        existing_map: dict[str, ArbitrageOpportunity] = {
            opp.stable_id: opp for opp in self._opportunities
        }

        new_count = 0
        updated_count = 0

        for new_opp in new_opportunities:
            new_opp.last_seen_at = now
            existing = existing_map.get(new_opp.stable_id)
            if existing:
                # Preserve original detection time and ID
                new_opp.detected_at = existing.detected_at
                new_opp.id = existing.id
                # Preserve AI analysis if not freshly attached from DB
                if existing.ai_analysis and not new_opp.ai_analysis:
                    new_opp.ai_analysis = existing.ai_analysis
                updated_count += 1
            else:
                new_count += 1
            existing_map[new_opp.stable_id] = new_opp

        # Remove expired opportunities (resolution date has passed)
        merged = [
            opp
            for opp in existing_map.values()
            if opp.resolution_date is None or _make_aware(opp.resolution_date) > now
        ]

        expired_count = len(existing_map) - len(merged)

        # Sort by ROI
        merged.sort(key=lambda x: x.roi_percent, reverse=True)

        retained = len(merged) - new_count - updated_count
        if retained < 0:
            retained = 0
        parts = []
        if new_count:
            parts.append(f"{new_count} new")
        if updated_count:
            parts.append(f"{updated_count} updated")
        if retained:
            parts.append(f"{retained} retained from prior scans")
        if expired_count:
            parts.append(f"{expired_count} expired removed")
        if parts:
            print(f"  Merge: {', '.join(parts)} -> {len(merged)} total")

        return merged

    # Maximum number of opportunities to score per scan cycle
    AI_SCORE_MAX_PER_SCAN = 50
    # How many LLM calls can run concurrently
    AI_SCORE_CONCURRENCY = 3
    # Don't re-score an opportunity within this many seconds
    AI_SCORE_CACHE_TTL_SECONDS = 300  # 5 minutes

    async def _ai_score_opportunities(self, opportunities: list):
        """Score unscored opportunities using AI (runs in background).

        Judgments are persisted in the OpportunityJudgment DB table (by
        the judge itself) and looked up from there on subsequent scans.

        Cost controls:
        - Limits to AI_SCORE_MAX_PER_SCAN per scan cycle
        - Caps concurrency via AI_SCORE_CONCURRENCY semaphore
        - Skips opportunities already judged within AI_SCORE_CACHE_TTL_SECONDS (DB lookup)
        - Respects cancellation (e.g. on pause) between each scoring call
        """
        try:
            from services.ai.opportunity_judge import opportunity_judge

            # Filter: only unscored (DB dedup already attached scored ones)
            candidates = [o for o in opportunities if o.ai_analysis is None]

            if not candidates:
                return

            # Prioritise by ROI descending — score the best opportunities first
            candidates.sort(key=lambda x: x.roi_percent, reverse=True)
            # Cap the number of LLM calls per scan cycle
            candidates = candidates[: self.AI_SCORE_MAX_PER_SCAN]

            print(f"  AI Judge: scoring {len(candidates)} unscored opportunities...")

            sem = asyncio.Semaphore(self.AI_SCORE_CONCURRENCY)

            async def _score_one(opp):
                async with sem:
                    result = await opportunity_judge.judge_opportunity(opp)
                    opp.ai_analysis = AIAnalysis(
                        overall_score=result.get("overall_score", 0.0),
                        profit_viability=result.get("profit_viability", 0.0),
                        resolution_safety=result.get("resolution_safety", 0.0),
                        execution_feasibility=result.get("execution_feasibility", 0.0),
                        market_efficiency=result.get("market_efficiency", 0.0),
                        recommendation=result.get("recommendation", "review"),
                        reasoning=result.get("reasoning"),
                        risk_factors=result.get("risk_factors", []),
                        judged_at=datetime.now(timezone.utc),
                    )
                    print(
                        f"  AI Judge: {opp.title[:50]}... "
                        f"-> {result.get('recommendation', 'unknown')} "
                        f"(score: {result.get('overall_score', 0):.2f})"
                    )

            tasks = [asyncio.create_task(_score_one(opp)) for opp in candidates]

            for task in tasks:
                try:
                    await task
                except asyncio.CancelledError:
                    for t in tasks:
                        t.cancel()
                    raise
                except Exception as e:
                    print(f"  AI Judge error: {e}")

        except asyncio.CancelledError:
            print("  AI scoring cancelled")
            raise
        except Exception as e:
            print(f"  AI scoring error: {e}")

    async def _scan_loop(self):
        """Internal scan loop with tiered polling.

        When TIERED_SCANNING_ENABLED:
          - Full scans run every FULL_SCAN_INTERVAL_SECONDS (default 120s)
          - Fast scans run every FAST_SCAN_INTERVAL_SECONDS (default 15s) between full scans
          - Fast scans only fetch/evaluate HOT-tier markets + incremental new markets
          - Crypto schedule predictions can trigger immediate fast scans

        When disabled, falls back to the original fixed-interval full scan.
        """
        while self._running:
            if not self._enabled:
                await asyncio.sleep(self._interval_seconds)
                continue

            if not settings.TIERED_SCANNING_ENABLED:
                # Legacy mode: simple fixed-interval full scan
                try:
                    await self.scan_once()
                except Exception as e:
                    print(f"Scan error: {e}")
                await asyncio.sleep(self._interval_seconds)
                continue

            # Tiered scanning mode
            now = datetime.now(timezone.utc)

            # Determine if it's time for a full scan
            full_interval = settings.FULL_SCAN_INTERVAL_SECONDS
            needs_full = (
                self._last_full_scan is None
                or (now - self._last_full_scan).total_seconds() >= full_interval
            )

            if needs_full:
                try:
                    await self.scan_once()
                except Exception as e:
                    print(f"Full scan error: {e}")
            else:
                # Fast scan (only if we have cached data from a previous full scan)
                if self._cached_markets:
                    # Check if crypto prediction triggers an immediate scan
                    triggered = self._prioritizer.should_trigger_fast_scan(now)
                    if triggered:
                        print("  [TRIGGER] Crypto market creation imminent — fast scan")

                    try:
                        await self.scan_fast()
                    except Exception as e:
                        print(f"Fast scan error: {e}")
                else:
                    # No cached data yet — force a full scan
                    try:
                        await self.scan_once()
                    except Exception as e:
                        print(f"Full scan error (first run): {e}")

            # Sleep for fast interval (the loop will decide full vs fast next iteration)
            await self._set_activity(
                f"Idle — next scan in {settings.FAST_SCAN_INTERVAL_SECONDS}s"
            )
            await asyncio.sleep(settings.FAST_SCAN_INTERVAL_SECONDS)

    async def _attach_ai_judgments(self, opportunities: list):
        """Attach existing AI judgments from the DB to opportunity objects.

        Performs a single batch query for recent judgments and matches them
        to opportunities by stable_id. This is the source of truth —
        no in-memory cache needed.
        """
        if not opportunities:
            return

        try:
            from sqlalchemy import func

            cutoff = datetime.now(timezone.utc) - timedelta(
                seconds=self.AI_SCORE_CACHE_TTL_SECONDS
            )

            async with AsyncSessionLocal() as session:
                # Get the most recent judgment per opportunity_id,
                # but only those within the TTL window
                subq = (
                    select(
                        OpportunityJudgment.opportunity_id,
                        func.max(OpportunityJudgment.judged_at).label("latest"),
                    )
                    .where(OpportunityJudgment.judged_at >= cutoff)
                    .group_by(OpportunityJudgment.opportunity_id)
                    .subquery()
                )
                rows = (
                    (
                        await session.execute(
                            select(OpportunityJudgment).join(
                                subq,
                                (
                                    OpportunityJudgment.opportunity_id
                                    == subq.c.opportunity_id
                                )
                                & (OpportunityJudgment.judged_at == subq.c.latest),
                            )
                        )
                    )
                    .scalars()
                    .all()
                )

            # Build stable_id -> AIAnalysis lookup
            judgment_map: dict[str, AIAnalysis] = {}
            for row in rows:
                opp_id = row.opportunity_id or ""
                # Convert opportunity_id to stable_id by stripping trailing _<timestamp>
                parts = opp_id.rsplit("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    stable_id = parts[0]
                else:
                    stable_id = opp_id

                judgment_map[stable_id] = AIAnalysis(
                    overall_score=row.overall_score or 0.0,
                    profit_viability=row.profit_viability or 0.0,
                    resolution_safety=row.resolution_safety or 0.0,
                    execution_feasibility=row.execution_feasibility or 0.0,
                    market_efficiency=row.market_efficiency or 0.0,
                    recommendation=row.recommendation or "review",
                    reasoning=row.reasoning,
                    risk_factors=row.risk_factors or [],
                    judged_at=row.judged_at,
                )

            # Attach to matching opportunities
            attached = 0
            for opp in opportunities:
                analysis = judgment_map.get(opp.stable_id)
                if analysis:
                    opp.ai_analysis = analysis
                    attached += 1

            if attached:
                print(f"  Attached {attached} existing AI judgments from DB")

        except Exception as e:
            print(f"  Error loading AI judgments from DB: {e}")

    async def start_continuous_scan(self, interval_seconds: int = None):
        """Start continuous scanning loop"""
        # Load persisted settings first
        await self.load_settings()

        if interval_seconds is not None:
            self._interval_seconds = interval_seconds

        self._running = True
        print(
            f"Starting continuous scan (interval: {self._interval_seconds}s, enabled: {self._enabled})"
        )

        # Run the scan loop
        await self._scan_loop()

    async def start(self):
        """Enable scanning and resume all background services"""
        self._enabled = True
        global_pause_state.resume()
        await self.save_settings()
        await self._notify_status_change()

        # Kick off an immediate scan in the background so this
        # method returns quickly and doesn't block the API response.
        if self._running:
            asyncio.create_task(self.scan_once())

    async def pause(self):
        """Pause all background services (scanner, auto trader, copy trader, wallet tracker, discovery, etc.)"""
        self._enabled = False
        global_pause_state.pause()
        # Cancel any in-flight AI scoring task to stop incurring API costs
        await self._cancel_ai_scoring()
        await self.save_settings()
        await self._notify_status_change()

    async def stop(self):
        """Stop continuous scanning loop completely"""
        self._running = False
        self._enabled = False
        await self._cancel_ai_scoring()

    async def _cancel_ai_scoring(self):
        """Cancel any running AI scoring background task."""
        if self._ai_scoring_task and not self._ai_scoring_task.done():
            self._ai_scoring_task.cancel()
            try:
                await self._ai_scoring_task
            except (asyncio.CancelledError, Exception):
                pass
            print("  AI scoring task cancelled")

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
        status = {
            "running": self._running,
            "enabled": self._enabled,
            "interval_seconds": self._interval_seconds,
            "auto_ai_scoring": self._auto_ai_scoring,
            "last_scan": (self._last_scan.isoformat() + "Z")
            if self._last_scan
            else None,
            "opportunities_count": len(self._opportunities),
            "current_activity": self._current_activity,
            "strategies": [
                {"name": s.name, "type": s.strategy_type.value} for s in self.strategies
            ] + [
                {"name": self._news_edge_strategy.name, "type": self._news_edge_strategy.strategy_type.value}
            ],
        }

        # Add tiered scanning status
        if settings.TIERED_SCANNING_ENABLED:
            prioritizer_stats = self._prioritizer.get_stats()
            status["tiered_scanning"] = {
                "enabled": True,
                "fast_scan_interval": settings.FAST_SCAN_INTERVAL_SECONDS,
                "full_scan_interval": settings.FULL_SCAN_INTERVAL_SECONDS,
                "fast_scan_cycle": self._fast_scan_cycle,
                "last_full_scan": (self._last_full_scan.isoformat() + "Z")
                if self._last_full_scan
                else None,
                "last_fast_scan": (self._last_fast_scan.isoformat() + "Z")
                if self._last_fast_scan
                else None,
                "cached_markets": len(self._cached_markets),
                "cached_events": len(self._cached_events),
                **prioritizer_stats,
            }

        return status

    def get_opportunities(
        self, filter: Optional[OpportunityFilter] = None
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
            if filter.category:
                # Case-insensitive category matching
                category_lower = filter.category.lower()
                opps = [
                    o
                    for o in opps
                    if o.category and o.category.lower() == category_lower
                ]

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
        now = datetime.now(timezone.utc)
        before_count = len(self._opportunities)

        self._opportunities = [
            opp
            for opp in self._opportunities
            if opp.resolution_date is None or _make_aware(opp.resolution_date) > now
        ]

        removed = before_count - len(self._opportunities)
        if removed > 0:
            print(f"Removed {removed} expired opportunities")
        return removed

    def remove_old_opportunities(self, max_age_minutes: int = 60) -> int:
        """Remove opportunities older than max_age_minutes. Returns count removed."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)
        before_count = len(self._opportunities)

        self._opportunities = [
            opp for opp in self._opportunities if _make_aware(opp.detected_at) >= cutoff
        ]

        removed = before_count - len(self._opportunities)
        if removed > 0:
            print(
                f"Removed {removed} opportunities older than {max_age_minutes} minutes"
            )
        return removed


# Singleton instance
scanner = ArbitrageScanner()
