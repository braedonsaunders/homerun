"""
Opportunity Recorder Service

Records every arbitrage opportunity detected by the scanner and tracks
their outcomes over time. Provides replay/analysis queries for strategy
evaluation: accuracy rates, ROI distributions, false positive rates,
and opportunity decay analysis.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select, func, and_, case

from models.database import AsyncSessionLocal, OpportunityHistory
from models.opportunity import ArbitrageOpportunity
from services.polymarket import polymarket_client
from utils.logger import get_logger

logger = get_logger("opportunity_recorder")

# How often to check for resolved markets (seconds)
_RESOLUTION_CHECK_INTERVAL = 600  # 10 minutes

# Only check opportunities detected within this window for resolution
_MAX_RESOLUTION_AGE_DAYS = 90


class OpportunityRecorder:
    """Records scanner opportunities and tracks their real-world outcomes."""

    def __init__(self):
        self._running = False
        self._resolution_task: Optional[asyncio.Task] = None
        # In-memory set of IDs already persisted in this process lifetime,
        # used to skip unnecessary DB round-trips on hot path.
        self._known_ids: set[str] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        """Register the scanner callback and kick off the resolution tracker."""
        from services.scanner import scanner

        scanner.add_callback(self._on_opportunities)
        logger.info("Registered scanner callback for opportunity recording")

        # Pre-load known IDs so the first scan after restart deduplicates
        await self._load_known_ids()

        # Start background resolution checker
        self._running = True
        self._resolution_task = asyncio.create_task(self._resolution_loop())
        logger.info("Opportunity recorder started")

    def stop(self):
        """Stop background tasks."""
        self._running = False
        if self._resolution_task and not self._resolution_task.done():
            self._resolution_task.cancel()
        logger.info("Opportunity recorder stopped")

    # ------------------------------------------------------------------
    # Scanner callback  (records every detected opportunity)
    # ------------------------------------------------------------------

    async def _on_opportunities(self, opportunities: list[ArbitrageOpportunity]):
        """Callback invoked by the scanner after each scan cycle."""
        if not opportunities:
            return

        new_count = 0
        try:
            async with AsyncSessionLocal() as session:
                for opp in opportunities:
                    if opp.id in self._known_ids:
                        continue

                    # Check DB as well (handles restart dedup)
                    exists = await session.execute(
                        select(OpportunityHistory.id).where(
                            OpportunityHistory.id == opp.id
                        )
                    )
                    if exists.scalar_one_or_none() is not None:
                        self._known_ids.add(opp.id)
                        continue

                    record = OpportunityHistory(
                        id=opp.id,
                        strategy_type=opp.strategy.value,
                        event_id=opp.event_id,
                        title=opp.title,
                        total_cost=opp.total_cost,
                        expected_roi=opp.roi_percent,
                        risk_score=opp.risk_score,
                        positions_data=self._serialise_positions(opp),
                        detected_at=opp.detected_at,
                        resolution_date=opp.resolution_date,
                    )
                    session.add(record)
                    self._known_ids.add(opp.id)
                    new_count += 1

                if new_count:
                    await session.commit()

        except Exception:
            logger.error("Failed to record opportunities", exc_info=True)
            return

        if new_count:
            logger.info(
                "Recorded new opportunities",
                new=new_count,
                total_batch=len(opportunities),
            )

    # ------------------------------------------------------------------
    # Resolution tracking
    # ------------------------------------------------------------------

    async def _resolution_loop(self):
        """Periodically resolve open opportunities."""
        while self._running:
            try:
                await asyncio.sleep(_RESOLUTION_CHECK_INTERVAL)
                if not self._running:
                    break
                await self._check_resolutions()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.error("Resolution check failed", exc_info=True)

    async def _check_resolutions(self):
        """Check unresolved opportunities against the Polymarket API."""
        cutoff = datetime.utcnow() - timedelta(days=_MAX_RESOLUTION_AGE_DAYS)

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(OpportunityHistory).where(
                    and_(
                        OpportunityHistory.was_profitable.is_(None),
                        OpportunityHistory.detected_at >= cutoff,
                    )
                )
            )
            unresolved = result.scalars().all()

        if not unresolved:
            return

        logger.info("Checking resolution status", count=len(unresolved))

        resolved_count = 0
        for record in unresolved:
            try:
                outcome = await self._resolve_opportunity(record)
                if outcome is not None:
                    was_profitable, actual_roi = outcome
                    async with AsyncSessionLocal() as session:
                        row = await session.get(OpportunityHistory, record.id)
                        if row:
                            row.was_profitable = was_profitable
                            row.actual_roi = actual_roi
                            row.expired_at = datetime.utcnow()
                            await session.commit()
                    resolved_count += 1
            except Exception:
                logger.error(
                    "Failed to resolve opportunity",
                    opportunity_id=record.id,
                    exc_info=True,
                )

        if resolved_count:
            logger.info("Resolved opportunities", resolved=resolved_count)

    async def _resolve_opportunity(
        self, record: OpportunityHistory
    ) -> Optional[tuple[bool, float]]:
        """
        Determine if a recorded opportunity was actually profitable.

        Returns (was_profitable, actual_roi) or None if not yet resolvable.
        """
        positions = record.positions_data or {}
        markets_data = positions.get("markets", [])
        if not markets_data:
            return None

        # Gather the condition_ids we need to look up
        condition_ids = [
            m.get("condition_id") or m.get("id", "")
            for m in markets_data
            if m.get("condition_id") or m.get("id")
        ]
        if not condition_ids:
            return None

        # Fetch current market state for each involved market
        all_closed = True
        final_prices: dict[str, list[float]] = {}

        for cid in condition_ids:
            try:
                info = await polymarket_client.get_market_by_condition_id(cid)
                if not info:
                    all_closed = False
                    continue

                # get_market_by_condition_id caches a trimmed dict; we need the
                # full market response to check closed/outcome prices.  Fetch
                # directly from Gamma for resolution data.
                client = await polymarket_client._get_client()
                resp = await client.get(
                    f"{polymarket_client.gamma_url}/markets",
                    params={"condition_id": cid, "limit": 1},
                )
                resp.raise_for_status()
                data = resp.json()

                if not data:
                    all_closed = False
                    continue

                market_raw = data[0]
                if not market_raw.get("closed", False):
                    all_closed = False
                    continue

                # Parse final outcome prices
                try:
                    outcome_prices = [
                        float(p)
                        for p in json.loads(market_raw.get("outcomePrices", "[]"))
                    ]
                except (json.JSONDecodeError, TypeError):
                    outcome_prices = []

                final_prices[cid] = outcome_prices

            except Exception:
                all_closed = False

        if not all_closed:
            return None

        # All markets closed -- compute actual outcome
        total_cost = record.total_cost or 0.0
        if total_cost <= 0:
            return (False, 0.0)

        # Calculate actual payout from final prices and the positions we
        # would have taken.
        actual_payout = 0.0
        positions_to_take = positions.get("positions_to_take", [])

        for pos in positions_to_take:
            cid = pos.get("condition_id") or pos.get("market_id", "")
            side_index = 0 if pos.get("side", "").upper() in ("YES", "0") else 1
            size = float(pos.get("size", 0) or pos.get("quantity", 0) or 0)

            prices = final_prices.get(cid, [])
            if side_index < len(prices):
                # Final price near 1.0 means the outcome happened
                final_p = prices[side_index]
                actual_payout += size * final_p

        # If we had no specific positions data, fall back to a simple
        # heuristic: check whether every outcome price matches what we bet on.
        if not positions_to_take and markets_data:
            # Assume the expected payout was $1 per share set and check if
            # final prices would have delivered that.
            actual_payout = self._heuristic_payout(markets_data, final_prices)

        actual_roi = (
            ((actual_payout - total_cost) / total_cost * 100) if total_cost > 0 else 0.0
        )
        was_profitable = actual_payout > total_cost

        return (was_profitable, round(actual_roi, 4))

    # ------------------------------------------------------------------
    # Analysis queries
    # ------------------------------------------------------------------

    async def get_strategy_accuracy(self) -> dict:
        """
        Per-strategy true positive rate and false positive rate.

        Returns::

            {
                "basic": {"total": 10, "resolved": 8, "profitable": 6,
                          "true_positive_rate": 0.75, "false_positive_rate": 0.25},
                ...
            }
        """
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(
                    OpportunityHistory.strategy_type,
                    func.count(OpportunityHistory.id).label("total"),
                    func.count(OpportunityHistory.was_profitable).label("resolved"),
                    func.sum(
                        case(
                            (OpportunityHistory.was_profitable == True, 1),  # noqa: E712
                            else_=0,
                        )
                    ).label("profitable"),
                    func.sum(
                        case(
                            (OpportunityHistory.was_profitable == False, 1),  # noqa: E712
                            else_=0,
                        )
                    ).label("unprofitable"),
                ).group_by(OpportunityHistory.strategy_type)
            )
            rows = result.all()

        out: dict = {}
        for row in rows:
            strategy = row.strategy_type
            total = row.total or 0
            resolved = row.resolved or 0
            profitable = row.profitable or 0
            unprofitable = row.unprofitable or 0

            tp_rate = (profitable / resolved) if resolved > 0 else None
            fp_rate = (unprofitable / resolved) if resolved > 0 else None

            out[strategy] = {
                "total": total,
                "resolved": resolved,
                "profitable": profitable,
                "unprofitable": unprofitable,
                "true_positive_rate": round(tp_rate, 4)
                if tp_rate is not None
                else None,
                "false_positive_rate": round(fp_rate, 4)
                if fp_rate is not None
                else None,
            }

        return out

    async def get_historical_roi(
        self,
        days: int = 30,
        strategy: Optional[str] = None,
    ) -> dict:
        """
        Actual ROI distribution over time for resolved opportunities.

        Returns::

            {
                "period_days": 30,
                "strategy": "all",
                "count": 42,
                "mean_roi": 3.21,
                "median_roi": 2.5,
                "min_roi": -10.0,
                "max_roi": 25.3,
                "buckets": [
                    {"date": "2026-01-01", "avg_roi": 2.1, "count": 5},
                    ...
                ]
            }
        """
        cutoff = datetime.utcnow() - timedelta(days=days)

        filters = [
            OpportunityHistory.actual_roi.isnot(None),
            OpportunityHistory.detected_at >= cutoff,
        ]
        if strategy:
            filters.append(OpportunityHistory.strategy_type == strategy)

        async with AsyncSessionLocal() as session:
            # Aggregates
            agg = await session.execute(
                select(
                    func.count(OpportunityHistory.id).label("cnt"),
                    func.avg(OpportunityHistory.actual_roi).label("mean"),
                    func.min(OpportunityHistory.actual_roi).label("min_roi"),
                    func.max(OpportunityHistory.actual_roi).label("max_roi"),
                ).where(and_(*filters))
            )
            agg_row = agg.one()

            # All ROIs for median calculation
            roi_result = await session.execute(
                select(OpportunityHistory.actual_roi)
                .where(and_(*filters))
                .order_by(OpportunityHistory.actual_roi)
            )
            roi_values = [r[0] for r in roi_result.all() if r[0] is not None]

            # Daily buckets
            bucket_result = await session.execute(
                select(
                    func.date(OpportunityHistory.detected_at).label("day"),
                    func.avg(OpportunityHistory.actual_roi).label("avg_roi"),
                    func.count(OpportunityHistory.id).label("cnt"),
                )
                .where(and_(*filters))
                .group_by(func.date(OpportunityHistory.detected_at))
                .order_by(func.date(OpportunityHistory.detected_at))
            )
            buckets = [
                {
                    "date": str(r.day),
                    "avg_roi": round(r.avg_roi, 4) if r.avg_roi else 0,
                    "count": r.cnt,
                }
                for r in bucket_result.all()
            ]

        median_roi = None
        if roi_values:
            mid = len(roi_values) // 2
            if len(roi_values) % 2 == 0 and len(roi_values) >= 2:
                median_roi = round((roi_values[mid - 1] + roi_values[mid]) / 2, 4)
            else:
                median_roi = round(roi_values[mid], 4)

        return {
            "period_days": days,
            "strategy": strategy or "all",
            "count": agg_row.cnt or 0,
            "mean_roi": round(agg_row.mean, 4) if agg_row.mean is not None else None,
            "median_roi": median_roi,
            "min_roi": round(agg_row.min_roi, 4)
            if agg_row.min_roi is not None
            else None,
            "max_roi": round(agg_row.max_roi, 4)
            if agg_row.max_roi is not None
            else None,
            "buckets": buckets,
        }

    async def get_opportunity_stats(self) -> dict:
        """
        High-level stats: total detected, total profitable, breakdown by strategy.

        Returns::

            {
                "total_detected": 200,
                "total_resolved": 80,
                "total_profitable": 55,
                "total_unprofitable": 25,
                "overall_profitable_rate": 0.6875,
                "by_strategy": { ... }
            }
        """
        async with AsyncSessionLocal() as session:
            total = await session.execute(select(func.count(OpportunityHistory.id)))
            total_detected = total.scalar() or 0

            resolved = await session.execute(
                select(func.count(OpportunityHistory.id)).where(
                    OpportunityHistory.was_profitable.isnot(None)
                )
            )
            total_resolved = resolved.scalar() or 0

            profitable = await session.execute(
                select(func.count(OpportunityHistory.id)).where(
                    OpportunityHistory.was_profitable == True  # noqa: E712
                )
            )
            total_profitable = profitable.scalar() or 0

            unprofitable = await session.execute(
                select(func.count(OpportunityHistory.id)).where(
                    OpportunityHistory.was_profitable == False  # noqa: E712
                )
            )
            total_unprofitable = unprofitable.scalar() or 0

        by_strategy = await self.get_strategy_accuracy()
        profitable_rate = (
            round(total_profitable / total_resolved, 4) if total_resolved > 0 else None
        )

        return {
            "total_detected": total_detected,
            "total_resolved": total_resolved,
            "total_profitable": total_profitable,
            "total_unprofitable": total_unprofitable,
            "overall_profitable_rate": profitable_rate,
            "by_strategy": by_strategy,
        }

    async def get_false_positive_rate(self, strategy: str) -> dict:
        """
        What percentage of detected opportunities for *strategy* were
        actually unprofitable once resolved.

        Returns::

            {
                "strategy": "negrisk",
                "total_detected": 50,
                "resolved": 30,
                "false_positives": 8,
                "false_positive_rate": 0.2667
            }
        """
        async with AsyncSessionLocal() as session:
            total = await session.execute(
                select(func.count(OpportunityHistory.id)).where(
                    OpportunityHistory.strategy_type == strategy
                )
            )
            total_detected = total.scalar() or 0

            resolved = await session.execute(
                select(func.count(OpportunityHistory.id)).where(
                    and_(
                        OpportunityHistory.strategy_type == strategy,
                        OpportunityHistory.was_profitable.isnot(None),
                    )
                )
            )
            total_resolved = resolved.scalar() or 0

            fp = await session.execute(
                select(func.count(OpportunityHistory.id)).where(
                    and_(
                        OpportunityHistory.strategy_type == strategy,
                        OpportunityHistory.was_profitable == False,  # noqa: E712
                    )
                )
            )
            false_positives = fp.scalar() or 0

        fp_rate = (
            round(false_positives / total_resolved, 4) if total_resolved > 0 else None
        )

        return {
            "strategy": strategy,
            "total_detected": total_detected,
            "resolved": total_resolved,
            "false_positives": false_positives,
            "false_positive_rate": fp_rate,
        }

    async def get_decay_analysis(
        self,
        days: int = 30,
        strategy: Optional[str] = None,
    ) -> dict:
        """
        How quickly do opportunities close after detection?

        Measures the time between ``detected_at`` and ``expired_at`` (set
        when the resolution check first resolves the opportunity).

        Returns::

            {
                "period_days": 30,
                "strategy": "all",
                "sample_size": 40,
                "mean_hours": 18.5,
                "median_hours": 12.0,
                "min_hours": 0.5,
                "max_hours": 168.0,
                "buckets": [
                    {"bucket": "<1h", "count": 5},
                    {"bucket": "1-6h", "count": 12},
                    ...
                ]
            }
        """
        cutoff = datetime.utcnow() - timedelta(days=days)

        filters = [
            OpportunityHistory.expired_at.isnot(None),
            OpportunityHistory.detected_at >= cutoff,
        ]
        if strategy:
            filters.append(OpportunityHistory.strategy_type == strategy)

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(
                    OpportunityHistory.detected_at,
                    OpportunityHistory.expired_at,
                ).where(and_(*filters))
            )
            rows = result.all()

        if not rows:
            return {
                "period_days": days,
                "strategy": strategy or "all",
                "sample_size": 0,
                "mean_hours": None,
                "median_hours": None,
                "min_hours": None,
                "max_hours": None,
                "buckets": [],
            }

        durations_hours: list[float] = []
        for detected, expired in rows:
            if detected and expired:
                delta = (expired - detected).total_seconds() / 3600.0
                durations_hours.append(max(delta, 0.0))

        durations_hours.sort()
        n = len(durations_hours)

        mean_h = sum(durations_hours) / n if n else 0
        min_h = durations_hours[0] if n else 0
        max_h = durations_hours[-1] if n else 0

        if n % 2 == 0 and n >= 2:
            median_h = (durations_hours[n // 2 - 1] + durations_hours[n // 2]) / 2
        elif n:
            median_h = durations_hours[n // 2]
        else:
            median_h = 0

        # Build histogram buckets
        bucket_defs = [
            ("<1h", 0, 1),
            ("1-6h", 1, 6),
            ("6-24h", 6, 24),
            ("1-3d", 24, 72),
            ("3-7d", 72, 168),
            (">7d", 168, float("inf")),
        ]
        buckets = []
        for label, lo, hi in bucket_defs:
            count = sum(1 for d in durations_hours if lo <= d < hi)
            buckets.append({"bucket": label, "count": count})

        return {
            "period_days": days,
            "strategy": strategy or "all",
            "sample_size": n,
            "mean_hours": round(mean_h, 2),
            "median_hours": round(median_h, 2),
            "min_hours": round(min_h, 2),
            "max_hours": round(max_h, 2),
            "buckets": buckets,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _load_known_ids(self):
        """Populate the in-memory dedup set from the database on startup."""
        try:
            async with AsyncSessionLocal() as session:
                cutoff = datetime.utcnow() - timedelta(days=7)
                result = await session.execute(
                    select(OpportunityHistory.id).where(
                        OpportunityHistory.detected_at >= cutoff
                    )
                )
                self._known_ids = {row[0] for row in result.all()}
            logger.info("Loaded known opportunity IDs", count=len(self._known_ids))
        except Exception:
            logger.error("Failed to load known IDs", exc_info=True)

    @staticmethod
    def _serialise_positions(opp: ArbitrageOpportunity) -> dict:
        """Convert opportunity data to a JSON-serialisable dict for storage."""
        return {
            "markets": [
                {
                    k: (v.isoformat() if isinstance(v, datetime) else v)
                    for k, v in m.items()
                }
                for m in opp.markets
            ],
            "positions_to_take": [
                {
                    k: (v.isoformat() if isinstance(v, datetime) else v)
                    for k, v in p.items()
                }
                for p in opp.positions_to_take
            ],
            "expected_payout": opp.expected_payout,
            "gross_profit": opp.gross_profit,
            "fee": opp.fee,
            "net_profit": opp.net_profit,
            "risk_factors": opp.risk_factors,
            "mispricing_type": opp.mispricing_type.value
            if opp.mispricing_type
            else None,
            "category": opp.category,
            "min_liquidity": opp.min_liquidity,
            "max_position_size": opp.max_position_size,
        }

    @staticmethod
    def _heuristic_payout(
        markets_data: list[dict],
        final_prices: dict[str, list[float]],
    ) -> float:
        """
        Fallback payout estimate when no explicit positions_to_take exist.

        For basic/negrisk arb the expected payout is $1 per share set if
        all markets in the set resolve.  We check whether the sum of
        winning-outcome prices across markets reaches ~$1.
        """
        total_payout = 0.0
        for mkt in markets_data:
            cid = mkt.get("condition_id") or mkt.get("id", "")
            prices = final_prices.get(cid, [])
            if prices:
                # Take the maximum final price (the winning outcome)
                total_payout += max(prices)
        return total_payout


# Singleton
opportunity_recorder = OpportunityRecorder()
