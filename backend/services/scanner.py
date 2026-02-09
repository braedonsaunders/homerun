import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, Callable, List

from config import settings


def _make_aware(dt: Optional[datetime]) -> Optional[datetime]:
    """Ensure a datetime is timezone-aware (UTC). Returns None for None input."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


from models import ArbitrageOpportunity, OpportunityFilter
from models.opportunity import AIAnalysis
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
from models.opportunity import MispricingType
from services.pause_state import global_pause_state
from sqlalchemy import select


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
        self._opportunities: list[ArbitrageOpportunity] = []
        self._scan_callbacks: List[Callable] = []
        self._status_callbacks: List[Callable] = []
        self._scan_task: Optional[asyncio.Task] = None

        # Track the running AI scoring task so we can cancel it on pause
        self._ai_scoring_task: Optional[asyncio.Task] = None

        # Auto AI scoring: when False (default), the scanner does NOT
        # automatically score all opportunities with LLM after each scan.
        # Manual per-opportunity analysis (via the Analyze button) still works.
        self._auto_ai_scoring: bool = False

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

        try:
            # Fetch all active events and markets from Polymarket
            events = await self.client.get_all_events(closed=False)
            markets = await self.client.get_all_markets(active=True)

            # Filter out markets whose end_date has already passed —
            # these are resolved events awaiting settlement and can't
            # be traded profitably.
            now = datetime.now(timezone.utc)
            markets = [
                m for m in markets
                if m.end_date is None or _make_aware(m.end_date) > now
            ]

            # Also prune expired markets inside events so strategies
            # like NegRisk that iterate event.markets don't pick them up.
            for event in events:
                event.markets = [
                    m for m in event.markets
                    if m.end_date is None or _make_aware(m.end_date) > now
                ]

            print(
                f"  Fetched {len(events)} Polymarket events and {len(markets)} markets"
            )

            # Fetch Kalshi markets and merge them so ALL strategies
            # (not just cross-platform) can detect opportunities on Kalshi.
            kalshi_market_count = 0
            kalshi_event_count = 0
            if settings.CROSS_PLATFORM_ENABLED:
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
                # Sample tokens if too many
                token_sample = all_token_ids[:500]
                prices = await self.client.get_prices_batch(token_sample)
                print(f"  Fetched prices for {len(prices)} tokens")

            # Run all strategies and classify mispricing types
            all_opportunities = []
            for strategy in self.strategies:
                try:
                    opps = strategy.detect(events, markets, prices)
                    # Classify mispricing type if not already set by strategy
                    for opp in opps:
                        if opp.mispricing_type is None:
                            opp.mispricing_type = self._strategy_mispricing_map.get(
                                opp.strategy.value, MispricingType.WITHIN_MARKET
                            )
                    all_opportunities.extend(opps)
                    print(f"  {strategy.name}: found {len(opps)} opportunities")
                except Exception as e:
                    print(f"  {strategy.name}: error - {e}")

            # Deduplicate across strategies: when the same underlying markets
            # are detected by multiple strategies, keep only the highest-ROI one.
            all_opportunities = self._deduplicate_cross_strategy(all_opportunities)

            # Sort by ROI
            all_opportunities.sort(key=lambda x: x.roi_percent, reverse=True)

            # Attach existing AI judgments from the database
            await self._attach_ai_judgments(all_opportunities)

            self._opportunities = all_opportunities
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
                        self._ai_scoring_task = asyncio.create_task(
                            self._ai_score_opportunities(all_opportunities)
                        )
                except Exception:
                    pass  # AI scoring is non-critical

            # Notify callbacks
            for callback in self._scan_callbacks:
                try:
                    await callback(all_opportunities)
                except Exception as e:
                    print(f"  Callback error: {e}")

            print(
                f"[{datetime.utcnow().isoformat()}] Scan complete. {len(all_opportunities)} total opportunities"
            )
            return all_opportunities

        except Exception as e:
            print(f"[{datetime.utcnow().isoformat()}] Scan error: {e}")
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
                        f"\u2192 {result.get('recommendation', 'unknown')} "
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
        """Internal scan loop"""
        while self._running:
            if self._enabled:
                try:
                    await self.scan_once()
                except Exception as e:
                    print(f"Scan error: {e}")

            await asyncio.sleep(self._interval_seconds)

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

        # If not running, do an immediate scan
        if self._running:
            await self.scan_once()

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
        return {
            "running": self._running,
            "enabled": self._enabled,
            "interval_seconds": self._interval_seconds,
            "auto_ai_scoring": self._auto_ai_scoring,
            "last_scan": (self._last_scan.isoformat() + "Z")
            if self._last_scan
            else None,
            "opportunities_count": len(self._opportunities),
            "strategies": [
                {"name": s.name, "type": s.strategy_type.value} for s in self.strategies
            ],
        }

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
            if opp.resolution_date is None
            or _make_aware(opp.resolution_date) > now
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
            opp
            for opp in self._opportunities
            if _make_aware(opp.detected_at) >= cutoff
        ]

        removed = before_count - len(self._opportunities)
        if removed > 0:
            print(
                f"Removed {removed} opportunities older than {max_age_minutes} minutes"
            )
        return removed


# Singleton instance
scanner = ArbitrageScanner()
