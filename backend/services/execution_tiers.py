"""
Tiered Execution with Price Buffers

Implements a 4-tier execution system inspired by terauss/Polymarket-Copy-Trading-Bot.
Each tier is determined by signal confidence (ROI, liquidity, strategy type) and
controls price buffers, position sizing multipliers, retry behavior, and order types.

Higher confidence trades receive more aggressive pricing and additional retries.
Category-aware adjustments add extra buffers for volatile market types (sports, crypto).

All tier assignments are persisted to the database for analytics.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, String, Float, Integer, DateTime, Index, select, func

from models.database import Base, AsyncSessionLocal
from utils.logger import get_logger

logger = get_logger("execution_tiers")


# ==================== SQLALCHEMY MODEL ====================


class TierAssignment(Base):
    """Record of a tier assignment for an opportunity, stored for analytics."""

    __tablename__ = "tier_assignments"

    id = Column(String, primary_key=True)
    opportunity_id = Column(String, nullable=True)
    tier = Column(Integer, nullable=False)
    tier_name = Column(String, nullable=False)
    roi_percent = Column(Float)
    liquidity = Column(Float)
    strategy = Column(String)
    category = Column(String, nullable=True)
    price_buffer_applied = Column(Float)
    category_buffer_applied = Column(Float, default=0.0)
    total_buffer = Column(Float)
    assigned_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_tier_tier", "tier"),
        Index("idx_tier_strategy", "strategy"),
    )


# ==================== EXECUTION TIER DATACLASS ====================


@dataclass
class ExecutionTier:
    """Defines execution parameters for a given confidence tier.

    Attributes:
        tier: Tier number from 1 (highest confidence) to 4 (lowest).
        name: Human-readable tier name.
        price_buffer: Additional price buffer in dollars (e.g. 0.01 = 1 cent).
        size_multiplier: Multiplier applied to base position size (1.0 = no change).
        max_retries: Maximum number of order placement retries.
        order_type: Order type string ("GTC", "FOK", "GTD").
        description: Short description of the tier criteria.
    """

    tier: int
    name: str
    price_buffer: float
    size_multiplier: float
    max_retries: int
    order_type: str
    description: str


# ==================== EXECUTION TIER SERVICE ====================


class ExecutionTierService:
    """Service for classifying opportunities into execution tiers and applying
    tier-specific price buffers.

    The 4 tiers are ranked by signal confidence derived from ROI and liquidity:
        - Tier 1 (high_conviction): ROI >= 5% and liquidity >= $20,000
        - Tier 2 (standard): ROI >= 3% and liquidity >= $5,000
        - Tier 3 (cautious): ROI >= 2% and liquidity >= $1,000
        - Tier 4 (minimal): Everything else that passes minimum filters

    Category-specific buffers are added on top of the tier buffer to account
    for volatility differences across market types.
    """

    # Default tier definitions
    TIERS = {
        1: ExecutionTier(
            tier=1,
            name="high_conviction",
            price_buffer=0.01,
            size_multiplier=1.25,
            max_retries=5,
            order_type="GTC",
            description="High ROI + high liquidity",
        ),
        2: ExecutionTier(
            tier=2,
            name="standard",
            price_buffer=0.005,
            size_multiplier=1.0,
            max_retries=4,
            order_type="GTC",
            description="Standard execution",
        ),
        3: ExecutionTier(
            tier=3,
            name="cautious",
            price_buffer=0.003,
            size_multiplier=0.8,
            max_retries=3,
            order_type="GTC",
            description="Lower confidence",
        ),
        4: ExecutionTier(
            tier=4,
            name="minimal",
            price_buffer=0.001,
            size_multiplier=0.5,
            max_retries=2,
            order_type="FOK",
            description="Minimum confidence",
        ),
    }

    # Category-specific buffer adjustments (added on top of tier buffer)
    CATEGORY_BUFFERS = {
        "SPORTS": 0.01,
        "CRYPTO": 0.008,
        "CULTURE": 0.005,
        "POLITICS": 0.002,
        "WEATHER": 0.001,
    }

    def classify_opportunity(
        self,
        roi_percent: float,
        liquidity: float,
        strategy: str,
        category: Optional[str] = None,
    ) -> ExecutionTier:
        """Classify an opportunity into an execution tier based on signal strength.

        Args:
            roi_percent: Expected return on investment as a percentage.
            liquidity: Available market liquidity in USD.
            strategy: Strategy type identifier (e.g. "basic", "negrisk").
            category: Optional market category (e.g. "SPORTS", "CRYPTO").

        Returns:
            The ExecutionTier matching the opportunity's confidence level.
        """
        if roi_percent >= 5.0 and liquidity >= 20_000:
            tier = self.TIERS[1]
        elif roi_percent >= 3.0 and liquidity >= 5_000:
            tier = self.TIERS[2]
        elif roi_percent >= 2.0 and liquidity >= 1_000:
            tier = self.TIERS[3]
        else:
            tier = self.TIERS[4]

        logger.info(
            "Classified opportunity into execution tier",
            tier=tier.tier,
            tier_name=tier.name,
            roi_percent=roi_percent,
            liquidity=liquidity,
            strategy=strategy,
            category=category,
        )

        return tier

    def apply_price_buffer(
        self,
        base_price: float,
        side: str,
        tier: ExecutionTier,
        category: Optional[str] = None,
    ) -> float:
        """Apply tier-specific price buffer to a base price.

        For BUY orders the price is increased (willing to pay more to ensure fill).
        For SELL orders the price is decreased (willing to accept less for fill).

        The category buffer is added on top of the tier buffer when the category
        is recognized.

        Args:
            base_price: The raw market price before buffer adjustment.
            side: Trade side, either "BUY" or "SELL".
            tier: The ExecutionTier to apply.
            category: Optional market category for additional volatility buffer.

        Returns:
            The adjusted price with buffers applied.
        """
        tier_buffer = tier.price_buffer
        category_buffer = 0.0

        if category:
            category_upper = category.upper()
            category_buffer = self.CATEGORY_BUFFERS.get(category_upper, 0.0)

        total_buffer = tier_buffer + category_buffer

        side_upper = side.upper()
        if side_upper == "BUY":
            adjusted_price = base_price + total_buffer
        elif side_upper == "SELL":
            adjusted_price = base_price - total_buffer
        else:
            logger.warning(
                "Unknown trade side, returning base price without buffer",
                side=side,
            )
            return base_price

        logger.info(
            "Applied price buffer",
            base_price=base_price,
            adjusted_price=adjusted_price,
            side=side_upper,
            tier=tier.tier,
            tier_buffer=tier_buffer,
            category_buffer=category_buffer,
            total_buffer=total_buffer,
            category=category,
        )

        return adjusted_price

    async def record_tier_assignment(
        self,
        tier: ExecutionTier,
        roi_percent: float,
        liquidity: float,
        strategy: str,
        category: Optional[str] = None,
        opportunity_id: Optional[str] = None,
    ) -> TierAssignment:
        """Persist a tier assignment to the database for analytics.

        Args:
            tier: The assigned ExecutionTier.
            roi_percent: ROI that drove the classification.
            liquidity: Liquidity that drove the classification.
            strategy: Strategy type identifier.
            category: Optional market category.
            opportunity_id: Optional link to the opportunity being classified.

        Returns:
            The created TierAssignment database record.
        """
        tier_buffer = tier.price_buffer
        category_buffer = 0.0
        if category:
            category_upper = category.upper()
            category_buffer = self.CATEGORY_BUFFERS.get(category_upper, 0.0)
        total_buffer = tier_buffer + category_buffer

        assignment = TierAssignment(
            id=str(uuid.uuid4()),
            opportunity_id=opportunity_id,
            tier=tier.tier,
            tier_name=tier.name,
            roi_percent=roi_percent,
            liquidity=liquidity,
            strategy=strategy,
            category=category,
            price_buffer_applied=tier_buffer,
            category_buffer_applied=category_buffer,
            total_buffer=total_buffer,
        )

        async with AsyncSessionLocal() as session:
            session.add(assignment)
            await session.commit()
            await session.refresh(assignment)

        logger.info(
            "Recorded tier assignment",
            assignment_id=assignment.id,
            tier=tier.tier,
            tier_name=tier.name,
            opportunity_id=opportunity_id,
            total_buffer=total_buffer,
        )

        return assignment

    async def get_tier_stats(self) -> dict:
        """Get statistics on tier usage across all recorded assignments.

        Returns:
            A dictionary containing per-tier counts, average ROI, average
            liquidity, average buffer, and the overall total assignments.
        """
        async with AsyncSessionLocal() as session:
            # Aggregate counts and averages per tier
            result = await session.execute(
                select(
                    TierAssignment.tier,
                    TierAssignment.tier_name,
                    func.count(TierAssignment.id).label("count"),
                    func.avg(TierAssignment.roi_percent).label("avg_roi"),
                    func.avg(TierAssignment.liquidity).label("avg_liquidity"),
                    func.avg(TierAssignment.total_buffer).label("avg_total_buffer"),
                ).group_by(
                    TierAssignment.tier,
                    TierAssignment.tier_name,
                )
            )
            rows = result.all()

            # Per-category breakdown
            category_result = await session.execute(
                select(
                    TierAssignment.category,
                    func.count(TierAssignment.id).label("count"),
                    func.avg(TierAssignment.category_buffer_applied).label("avg_category_buffer"),
                )
                .where(TierAssignment.category.isnot(None))
                .group_by(TierAssignment.category)
            )
            category_rows = category_result.all()

            # Total count
            total_result = await session.execute(select(func.count(TierAssignment.id)))
            total_count = total_result.scalar() or 0

        tier_breakdown = {}
        for row in rows:
            tier_breakdown[row.tier] = {
                "tier": row.tier,
                "name": row.tier_name,
                "count": row.count,
                "avg_roi_percent": round(row.avg_roi, 4) if row.avg_roi else 0.0,
                "avg_liquidity": round(row.avg_liquidity, 2) if row.avg_liquidity else 0.0,
                "avg_total_buffer": round(row.avg_total_buffer, 6) if row.avg_total_buffer else 0.0,
            }

        category_breakdown = {}
        for row in category_rows:
            category_breakdown[row.category or "unknown"] = {
                "count": row.count,
                "avg_category_buffer": round(row.avg_category_buffer, 6) if row.avg_category_buffer else 0.0,
            }

        return {
            "total_assignments": total_count,
            "tier_breakdown": tier_breakdown,
            "category_breakdown": category_breakdown,
        }


# Singleton instance
execution_tier_service = ExecutionTierService()
