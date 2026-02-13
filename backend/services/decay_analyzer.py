"""
Opportunity Decay Analyzer

Tracks how long arbitrage opportunities survive before closing and uses
historical decay data to prioritize fast-closing opportunities.

Polymarket arb windows close as other traders discover them. This service
measures actual decay rates by strategy type, ROI magnitude, liquidity,
and time of day -- then produces an urgency score so the execution engine
can act on the fastest-closing opportunities first.
"""

import uuid
import asyncio
from datetime import datetime, timedelta
from utils.utcnow import utcnow
from dataclasses import dataclass
from typing import Optional, Dict, List, Set

import numpy as np
from sqlalchemy import select

from models.database import AsyncSessionLocal, OpportunityLifetime
from models.opportunity import ArbitrageOpportunity
from utils.logger import get_logger

logger = get_logger("decay_analyzer")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class DecayStats:
    """Aggregate decay statistics for a group of opportunities."""

    median_lifetime_seconds: float
    mean_lifetime_seconds: float
    p25_lifetime: float  # 25th percentile -- fast closers
    p75_lifetime: float  # 75th percentile -- slow closers
    total_tracked: int


@dataclass
class _TrackedOpportunity:
    """In-memory record for an opportunity we are actively watching."""

    opportunity_id: str
    strategy_type: str
    roi_at_detection: float
    liquidity_at_detection: float
    first_seen: datetime
    last_seen: datetime
    db_record_id: str


# ---------------------------------------------------------------------------
# Default stats returned when we have zero historical data
# ---------------------------------------------------------------------------

_DEFAULT_STATS = DecayStats(
    median_lifetime_seconds=300.0,  # 5 minutes
    mean_lifetime_seconds=360.0,  # 6 minutes
    p25_lifetime=120.0,  # 2 minutes
    p75_lifetime=600.0,  # 10 minutes
    total_tracked=0,
)


# ---------------------------------------------------------------------------
# Singleton service
# ---------------------------------------------------------------------------


class DecayAnalyzer:
    """Tracks opportunity lifetimes and scores urgency for new detections.

    Usage
    -----
    1.  Register this as a scan callback on the ArbitrageScanner:

            scanner.add_callback(decay_analyzer.on_scan_complete)

    2.  After some data accumulates, query stats or urgency:

            stats = await decay_analyzer.get_decay_stats("negrisk")
            score = await decay_analyzer.get_urgency_score(opportunity)
    """

    _instance: Optional["DecayAnalyzer"] = None

    def __new__(cls) -> "DecayAnalyzer":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        # In-memory tracking of currently-active opportunities.
        # Key = opportunity_id, Value = _TrackedOpportunity
        self._active: Dict[str, _TrackedOpportunity] = {}

        # In-memory cache for computed stats, keyed by strategy (None = all).
        # Invalidated every time we close opportunities.
        self._stats_cache: Dict[Optional[str], DecayStats] = {}
        self._stats_cache_valid = False

        # Lock to prevent concurrent scan processing
        self._lock = asyncio.Lock()

        logger.info("DecayAnalyzer initialized")

    # ------------------------------------------------------------------
    # Public API: scan callback
    # ------------------------------------------------------------------

    async def on_scan_complete(self, opportunities: List[ArbitrageOpportunity]) -> None:
        """Scanner callback -- compare incoming opportunities with the
        previously-active set to detect appearances and disappearances.

        Parameters
        ----------
        opportunities : list[ArbitrageOpportunity]
            The full set of opportunities returned by the latest scan.
        """
        async with self._lock:
            now = utcnow()
            incoming_ids: Set[str] = {opp.id for opp in opportunities}
            incoming_map: Dict[str, ArbitrageOpportunity] = {opp.id: opp for opp in opportunities}

            # --- Detect newly-appeared opportunities ---
            new_ids = incoming_ids - set(self._active.keys())
            for oid in new_ids:
                opp = incoming_map[oid]
                db_id = str(uuid.uuid4())
                tracked = _TrackedOpportunity(
                    opportunity_id=oid,
                    strategy_type=opp.strategy,
                    roi_at_detection=opp.roi_percent,
                    liquidity_at_detection=opp.min_liquidity,
                    first_seen=now,
                    last_seen=now,
                    db_record_id=db_id,
                )
                self._active[oid] = tracked
                await self._persist_first_seen(tracked)

            # --- Detect disappeared (closed) opportunities ---
            disappeared_ids = set(self._active.keys()) - incoming_ids
            for oid in disappeared_ids:
                tracked = self._active.pop(oid)
                lifetime = (now - tracked.first_seen).total_seconds()
                await self._persist_closed(tracked, now, lifetime, "price_moved")

            # --- Update last_seen for still-present opportunities ---
            still_present_ids = incoming_ids & set(self._active.keys())
            for oid in still_present_ids:
                self._active[oid].last_seen = now

            # Bulk-update last_seen in DB for still-present entries
            if still_present_ids:
                await self._bulk_update_last_seen([self._active[oid] for oid in still_present_ids], now)

            # Invalidate cache whenever we record closures
            if disappeared_ids:
                self._stats_cache_valid = False

            logger.info(
                "Scan diff processed",
                new=len(new_ids),
                closed=len(disappeared_ids),
                still_active=len(still_present_ids),
            )

    # ------------------------------------------------------------------
    # Public API: decay statistics
    # ------------------------------------------------------------------

    async def get_decay_stats(self, strategy: Optional[str] = None) -> DecayStats:
        """Return aggregate decay statistics.

        Parameters
        ----------
        strategy : str or None
            Filter to a single strategy type (e.g. ``"negrisk"``).
            ``None`` returns stats across all strategies.

        Returns
        -------
        DecayStats
            Aggregated lifetime percentiles and counts.  When no data has
            been collected yet the method returns sensible defaults.
        """
        # Check cache first
        if self._stats_cache_valid and strategy in self._stats_cache:
            return self._stats_cache[strategy]

        lifetimes = await self._load_lifetimes(strategy)

        if len(lifetimes) == 0:
            stats = DecayStats(
                median_lifetime_seconds=_DEFAULT_STATS.median_lifetime_seconds,
                mean_lifetime_seconds=_DEFAULT_STATS.mean_lifetime_seconds,
                p25_lifetime=_DEFAULT_STATS.p25_lifetime,
                p75_lifetime=_DEFAULT_STATS.p75_lifetime,
                total_tracked=0,
            )
        else:
            arr = np.array(lifetimes, dtype=np.float64)
            stats = DecayStats(
                median_lifetime_seconds=float(np.median(arr)),
                mean_lifetime_seconds=float(np.mean(arr)),
                p25_lifetime=float(np.percentile(arr, 25)),
                p75_lifetime=float(np.percentile(arr, 75)),
                total_tracked=len(lifetimes),
            )

        self._stats_cache[strategy] = stats
        # Mark cache valid only when we have cached the "all" key too
        if strategy is None:
            self._stats_cache_valid = True

        return stats

    # ------------------------------------------------------------------
    # Public API: urgency scoring
    # ------------------------------------------------------------------

    async def get_urgency_score(self, opportunity: ArbitrageOpportunity) -> float:
        """Score how urgently an opportunity must be acted upon (0-1).

        Higher values mean the window is predicted to close faster.

        Factors
        -------
        * **Strategy type** -- some strategies produce shorter-lived windows.
        * **ROI size** -- larger arb spread is more visible, attracting
          faster competition.
        * **Liquidity** -- higher liquidity means more bots watching,
          shrinking the window.
        * **Time of day** -- peak US/EU trading hours see faster closures.

        Each factor produces a sub-score in ``[0, 1]``; the final score is
        a weighted combination clamped to ``[0, 1]``.
        """
        strategy_key = opportunity.strategy
        stats = await self.get_decay_stats(strategy_key)

        # --- Factor 1: strategy-based decay speed ---
        # Faster median decay -> higher urgency
        strategy_score = self._score_from_median(stats.median_lifetime_seconds)

        # --- Factor 2: ROI visibility ---
        # Bigger ROI attracts faster competition
        roi = opportunity.roi_percent
        roi_score = self._score_roi(roi)

        # --- Factor 3: liquidity ---
        # Higher liquidity -> more watchers -> faster close
        liquidity = opportunity.min_liquidity
        liquidity_score = self._score_liquidity(liquidity)

        # --- Factor 4: time of day (UTC) ---
        hour = utcnow().hour
        time_score = self._score_time_of_day(hour)

        # Weighted combination
        weights = {
            "strategy": 0.40,
            "roi": 0.25,
            "liquidity": 0.20,
            "time": 0.15,
        }
        raw = (
            weights["strategy"] * strategy_score
            + weights["roi"] * roi_score
            + weights["liquidity"] * liquidity_score
            + weights["time"] * time_score
        )
        score = max(0.0, min(1.0, raw))

        logger.debug(
            "Urgency score computed",
            opportunity_id=opportunity.id,
            strategy=strategy_key,
            score=round(score, 4),
            strategy_score=round(strategy_score, 4),
            roi_score=round(roi_score, 4),
            liquidity_score=round(liquidity_score, 4),
            time_score=round(time_score, 4),
        )

        return score

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def active_count(self) -> int:
        """Number of opportunities currently being tracked."""
        return len(self._active)

    async def mark_resolved(self, opportunity_id: str) -> None:
        """Manually mark an opportunity as resolved (e.g. market settled)."""
        async with self._lock:
            if opportunity_id in self._active:
                tracked = self._active.pop(opportunity_id)
                now = utcnow()
                lifetime = (now - tracked.first_seen).total_seconds()
                await self._persist_closed(tracked, now, lifetime, "resolved")
                self._stats_cache_valid = False

    async def expire_stale(self, max_age_seconds: float = 7200.0) -> int:
        """Close any tracked opportunity older than *max_age_seconds* that
        is still marked active.  Returns the number of entries expired."""
        async with self._lock:
            now = utcnow()
            cutoff = now - timedelta(seconds=max_age_seconds)
            expired_ids = [oid for oid, t in self._active.items() if t.first_seen < cutoff]
            for oid in expired_ids:
                tracked = self._active.pop(oid)
                lifetime = (now - tracked.first_seen).total_seconds()
                await self._persist_closed(tracked, now, lifetime, "unknown")

            if expired_ids:
                self._stats_cache_valid = False
                logger.info("Expired stale opportunities", count=len(expired_ids))

            return len(expired_ids)

    # ------------------------------------------------------------------
    # Internal: scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_from_median(median_seconds: float) -> float:
        """Map median lifetime to a 0-1 urgency score.

        Very short medians (< 60 s) -> ~1.0 urgency.
        Very long medians (> 1800 s) -> ~0.0 urgency.
        Uses a shifted logistic so the curve is smooth.
        """
        # Logistic: 1 / (1 + exp(k*(x - x0)))
        # x0 = 600 s (10 min), k = 0.005
        k = 0.005
        x0 = 600.0
        return 1.0 / (1.0 + np.exp(k * (median_seconds - x0)))

    @staticmethod
    def _score_roi(roi_percent: float) -> float:
        """Bigger arb spread is more visible -> higher urgency.

        roi < 2%  -> low urgency (small, hard to spot)
        roi > 15% -> high urgency (very visible)
        """
        if roi_percent <= 0:
            return 0.0
        # Logistic centred at 8%
        k = 0.3
        x0 = 8.0
        return 1.0 / (1.0 + np.exp(-k * (roi_percent - x0)))

    @staticmethod
    def _score_liquidity(liquidity_usd: float) -> float:
        """Higher liquidity -> more watchers -> faster close.

        < $500   -> low urgency (few bots bother)
        > $50k   -> high urgency
        """
        if liquidity_usd <= 0:
            return 0.0
        # Log scale with sigmoid
        log_liq = np.log10(max(liquidity_usd, 1.0))
        # Map log10(500)=2.7 -> 0.2,  log10(50000)=4.7 -> 0.9
        k = 1.5
        x0 = 3.7  # log10(~5000)
        return 1.0 / (1.0 + np.exp(-k * (log_liq - x0)))

    @staticmethod
    def _score_time_of_day(utc_hour: int) -> float:
        """Peak trading hours see faster closures.

        Peak hours (13-21 UTC, US afternoon + EU evening) -> higher score.
        Off-peak (02-09 UTC) -> lower score.
        """
        # Simple sinusoidal model centred on 17 UTC
        # Score ranges from ~0.3 (off-peak) to ~1.0 (peak)
        phase = 2.0 * np.pi * (utc_hour - 17.0) / 24.0
        return float(0.65 + 0.35 * np.cos(phase))

    # ------------------------------------------------------------------
    # Internal: persistence
    # ------------------------------------------------------------------

    async def _persist_first_seen(self, tracked: _TrackedOpportunity) -> None:
        """Insert a new lifetime record when an opportunity first appears."""
        try:
            async with AsyncSessionLocal() as session:
                record = OpportunityLifetime(
                    id=tracked.db_record_id,
                    opportunity_id=tracked.opportunity_id,
                    strategy_type=tracked.strategy_type,
                    roi_at_detection=tracked.roi_at_detection,
                    liquidity_at_detection=tracked.liquidity_at_detection,
                    first_seen=tracked.first_seen,
                    last_seen=tracked.last_seen,
                    closed_at=None,
                    lifetime_seconds=None,
                    close_reason=None,
                )
                session.add(record)
                await session.commit()
        except Exception:
            logger.error(
                "Failed to persist first_seen",
                opportunity_id=tracked.opportunity_id,
            )

    async def _persist_closed(
        self,
        tracked: _TrackedOpportunity,
        closed_at: datetime,
        lifetime_seconds: float,
        close_reason: str,
    ) -> None:
        """Update the lifetime record when an opportunity closes."""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(OpportunityLifetime).where(OpportunityLifetime.id == tracked.db_record_id)
                )
                record = result.scalar_one_or_none()
                if record:
                    record.last_seen = tracked.last_seen
                    record.closed_at = closed_at
                    record.lifetime_seconds = lifetime_seconds
                    record.close_reason = close_reason
                    await session.commit()
                    logger.debug(
                        "Opportunity closed",
                        opportunity_id=tracked.opportunity_id,
                        lifetime_seconds=round(lifetime_seconds, 1),
                        reason=close_reason,
                    )
        except Exception:
            logger.error(
                "Failed to persist closure",
                opportunity_id=tracked.opportunity_id,
            )

    async def _bulk_update_last_seen(
        self,
        tracked_list: List[_TrackedOpportunity],
        now: datetime,
    ) -> None:
        """Batch update last_seen for still-active opportunities."""
        try:
            async with AsyncSessionLocal() as session:
                ids = [t.db_record_id for t in tracked_list]
                result = await session.execute(select(OpportunityLifetime).where(OpportunityLifetime.id.in_(ids)))
                records = result.scalars().all()
                for record in records:
                    record.last_seen = now
                await session.commit()
        except Exception:
            logger.error("Failed to bulk update last_seen")

    async def _load_lifetimes(self, strategy: Optional[str] = None) -> List[float]:
        """Load closed lifetime_seconds from the database."""
        try:
            async with AsyncSessionLocal() as session:
                query = select(OpportunityLifetime.lifetime_seconds).where(
                    OpportunityLifetime.lifetime_seconds.isnot(None)
                )
                if strategy is not None:
                    query = query.where(OpportunityLifetime.strategy_type == strategy)
                result = await session.execute(query)
                rows = result.scalars().all()
                return [float(v) for v in rows if v is not None]
        except Exception:
            logger.error("Failed to load lifetimes", strategy=strategy)
            return []


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

decay_analyzer = DecayAnalyzer()
