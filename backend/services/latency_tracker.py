"""
End-to-End Pipeline Latency Tracker

Measures full pipeline latency: detection -> risk check -> sizing ->
order placement -> fill confirmation. Tracks fill quality metrics
and identifies bottlenecks.

Inspired by terauss/Polymarket-Copy-Trading-Bot latency tracking.
"""

import uuid
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean, JSON, Index, select, func
from models.database import Base, AsyncSessionLocal
from utils.logger import get_logger

logger = get_logger("latency_tracker")


class PipelineLatencyLog(Base):
    """Record of end-to-end pipeline execution timing."""
    __tablename__ = "pipeline_latency_logs"

    id = Column(String, primary_key=True)
    opportunity_id = Column(String, nullable=True)
    trade_context = Column(String, nullable=True)  # "auto_trader", "copy_trader"

    # Stage timings (milliseconds)
    detection_ms = Column(Float, nullable=True)
    risk_check_ms = Column(Float, nullable=True)
    depth_check_ms = Column(Float, nullable=True)
    sizing_ms = Column(Float, nullable=True)
    order_placement_ms = Column(Float, nullable=True)
    fill_confirmation_ms = Column(Float, nullable=True)
    total_ms = Column(Float, nullable=True)

    # Fill quality
    fill_percent = Column(Float, nullable=True)  # 0-100
    fill_quality = Column(String, nullable=True)  # "excellent", "good", "poor"
    slippage_bps = Column(Float, nullable=True)  # Basis points of slippage

    # Result
    success = Column(Boolean, default=False)
    error = Column(String, nullable=True)

    recorded_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_pll_context", "trade_context"),
        Index("idx_pll_recorded", "recorded_at"),
    )


@dataclass
class StageTimer:
    """Tracks timing for individual pipeline stages."""
    stages: dict = field(default_factory=dict)  # stage_name -> duration_ms
    _current_stage: Optional[str] = None
    _stage_start: Optional[float] = None
    _pipeline_start: Optional[float] = None

    def start_pipeline(self):
        self._pipeline_start = time.monotonic()

    def start_stage(self, name: str):
        self._current_stage = name
        self._stage_start = time.monotonic()

    def end_stage(self):
        if self._current_stage and self._stage_start:
            elapsed_ms = (time.monotonic() - self._stage_start) * 1000
            self.stages[self._current_stage] = round(elapsed_ms, 2)
            self._current_stage = None
            self._stage_start = None

    def end_pipeline(self) -> float:
        if self._pipeline_start:
            total = (time.monotonic() - self._pipeline_start) * 1000
            self.stages["total"] = round(total, 2)
            return total
        return 0.0


@dataclass
class FillQuality:
    """Fill quality assessment."""
    fill_percent: float
    quality: str  # "excellent" >= 80%, "good" 40-79%, "poor" < 40%
    slippage_bps: float

    @staticmethod
    def assess(requested_size: float, filled_size: float,
               expected_price: float, actual_price: float) -> "FillQuality":
        fill_pct = (filled_size / requested_size * 100) if requested_size > 0 else 0
        slippage = abs(actual_price - expected_price) / expected_price * 10000 if expected_price > 0 else 0

        if fill_pct >= 80:
            quality = "excellent"
        elif fill_pct >= 40:
            quality = "good"
        else:
            quality = "poor"

        return FillQuality(
            fill_percent=round(fill_pct, 2),
            quality=quality,
            slippage_bps=round(slippage, 2),
        )


class LatencyTracker:
    """Tracks end-to-end pipeline latency and fill quality."""

    def __init__(self):
        self._active_timers: dict[str, StageTimer] = {}  # tracking_id -> timer

    def create_timer(self, tracking_id: str = None) -> tuple[str, StageTimer]:
        """Create a new pipeline timer. Returns (tracking_id, timer)."""
        tid = tracking_id or str(uuid.uuid4())
        timer = StageTimer()
        timer.start_pipeline()
        self._active_timers[tid] = timer
        return tid, timer

    def get_timer(self, tracking_id: str) -> Optional[StageTimer]:
        return self._active_timers.get(tracking_id)

    async def record(
        self,
        tracking_id: str,
        opportunity_id: str = None,
        trade_context: str = None,
        fill_quality: FillQuality = None,
        success: bool = True,
        error: str = None,
    ):
        """Finalize and persist a pipeline execution record."""
        timer = self._active_timers.pop(tracking_id, None)
        if not timer:
            return

        total_ms = timer.end_pipeline()
        stages = timer.stages

        log_data = {
            "total_ms": round(total_ms, 2),
            "stages": stages,
            "success": success,
            "context": trade_context,
        }

        if fill_quality:
            log_data["fill_percent"] = fill_quality.fill_percent
            log_data["fill_quality"] = fill_quality.quality
            log_data["slippage_bps"] = fill_quality.slippage_bps

        # Log with quality-based level
        if fill_quality and fill_quality.quality == "poor":
            logger.warning("Pipeline execution - poor fill", **log_data)
        else:
            logger.info("Pipeline execution complete", **log_data)

        # Persist to DB
        try:
            async with AsyncSessionLocal() as session:
                session.add(PipelineLatencyLog(
                    id=tracking_id,
                    opportunity_id=opportunity_id,
                    trade_context=trade_context,
                    detection_ms=stages.get("detection"),
                    risk_check_ms=stages.get("risk_check"),
                    depth_check_ms=stages.get("depth_check"),
                    sizing_ms=stages.get("sizing"),
                    order_placement_ms=stages.get("order_placement"),
                    fill_confirmation_ms=stages.get("fill_confirmation"),
                    total_ms=stages.get("total"),
                    fill_percent=fill_quality.fill_percent if fill_quality else None,
                    fill_quality=fill_quality.quality if fill_quality else None,
                    slippage_bps=fill_quality.slippage_bps if fill_quality else None,
                    success=success,
                    error=error,
                ))
                await session.commit()
        except Exception as e:
            logger.error("Failed to persist latency log", error=str(e))

    async def get_stats(self, trade_context: str = None, hours: int = 24) -> dict:
        """Get latency statistics for recent pipeline executions."""
        try:
            async with AsyncSessionLocal() as session:
                query = select(
                    func.count(PipelineLatencyLog.id).label("total"),
                    func.avg(PipelineLatencyLog.total_ms).label("avg_total_ms"),
                    func.min(PipelineLatencyLog.total_ms).label("min_total_ms"),
                    func.max(PipelineLatencyLog.total_ms).label("max_total_ms"),
                    func.avg(PipelineLatencyLog.fill_percent).label("avg_fill_pct"),
                    func.avg(PipelineLatencyLog.slippage_bps).label("avg_slippage_bps"),
                ).where(
                    PipelineLatencyLog.recorded_at >= datetime.utcnow() - __import__("datetime").timedelta(hours=hours)
                )
                if trade_context:
                    query = query.where(PipelineLatencyLog.trade_context == trade_context)

                result = await session.execute(query)
                row = result.one()

                # Quality breakdown
                quality_result = await session.execute(
                    select(
                        PipelineLatencyLog.fill_quality,
                        func.count(PipelineLatencyLog.id),
                    ).where(
                        PipelineLatencyLog.fill_quality.isnot(None)
                    ).group_by(PipelineLatencyLog.fill_quality)
                )
                quality_counts = {r[0]: r[1] for r in quality_result.all()}

                return {
                    "total_executions": row.total or 0,
                    "avg_total_ms": round(row.avg_total_ms, 2) if row.avg_total_ms else 0,
                    "min_total_ms": round(row.min_total_ms, 2) if row.min_total_ms else 0,
                    "max_total_ms": round(row.max_total_ms, 2) if row.max_total_ms else 0,
                    "avg_fill_percent": round(row.avg_fill_pct, 2) if row.avg_fill_pct else 0,
                    "avg_slippage_bps": round(row.avg_slippage_bps, 2) if row.avg_slippage_bps else 0,
                    "fill_quality_breakdown": quality_counts,
                }
        except Exception as e:
            logger.error("Failed to get latency stats", error=str(e))
            return {}


latency_tracker = LatencyTracker()
