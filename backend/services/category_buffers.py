"""
Category-Specific Risk Buffers

Implements category-specific risk buffers for prediction market trading.
Inspired by terauss/Polymarket-Copy-Trading-Bot which applies extra price
buffers for ATP tennis and Ligue 1 soccer markets due to live event
volatility.

Different market categories exhibit different volatility profiles.  Sports
markets, for example, can swing dramatically during live events, while
political markets tend to be more stable and slow-moving.  This service
defines per-category risk profiles that adjust slippage tolerance, price
buffers, minimum liquidity requirements, and position sizing to account for
these differences.

All buffer applications are persisted to the database for analytics and
post-trade review.
"""

import uuid
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import Column, String, Float, DateTime, Index

from models.database import Base, AsyncSessionLocal
from utils.logger import get_logger

logger = get_logger("category_buffers")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class CategoryRiskProfile:
    """Risk profile for a specific market category.

    Attributes
    ----------
    category : str
        Upper-case category identifier (e.g. ``"SPORTS"``).
    display_name : str
        Human-readable label for UI display.
    extra_slippage_tolerance : float
        Additional slippage percentage added on top of the base tolerance.
    price_buffer : float
        Extra price buffer in dollars applied to trade prices.
    min_liquidity_multiplier : float
        Multiplier applied to the base minimum liquidity requirement.
    position_size_multiplier : float
        Scaling factor for position size.  Values below 1.0 result in
        more conservative (smaller) positions.
    volatility_rating : str
        Qualitative volatility label: ``"low"``, ``"medium"``, ``"high"``,
        or ``"very_high"``.
    description : str
        Short explanation of why the category warrants its risk profile.
    """

    category: str
    display_name: str
    extra_slippage_tolerance: float  # Additional slippage % on top of base
    price_buffer: float  # Extra price buffer in dollars
    min_liquidity_multiplier: float  # Multiply base min_liquidity by this
    position_size_multiplier: float  # Scale position size (< 1.0 = more conservative)
    volatility_rating: str  # "low", "medium", "high", "very_high"
    description: str


# ---------------------------------------------------------------------------
# SQLAlchemy audit model
# ---------------------------------------------------------------------------


class CategoryBufferLog(Base):
    """Persisted record of a category buffer application for analytics."""

    __tablename__ = "category_buffer_logs"

    id = Column(String, primary_key=True)
    opportunity_id = Column(String, nullable=True)
    category = Column(String, nullable=False)
    base_slippage = Column(Float, nullable=True)
    adjusted_slippage = Column(Float, nullable=True)
    base_liquidity = Column(Float, nullable=True)
    adjusted_liquidity = Column(Float, nullable=True)
    base_position_size = Column(Float, nullable=True)
    adjusted_position_size = Column(Float, nullable=True)
    price_buffer_applied = Column(Float, nullable=True)
    applied_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_catbuf_category", "category"),
        Index("idx_catbuf_applied", "applied_at"),
    )


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class CategoryBufferService:
    """Provides category-aware risk adjustments for trade execution.

    Each market category has a pre-defined risk profile that adjusts
    slippage tolerance, minimum liquidity, position sizing, and price
    buffers.  Unknown categories fall back to a neutral default profile
    with no extra adjustments.

    Usage
    -----
    ::

        adjusted_slippage = category_buffer_service.adjust_slippage_tolerance(
            base_tolerance=2.0, category="SPORTS"
        )
        # => 3.5  (2.0 + 1.5 extra for SPORTS)

        price_buf = category_buffer_service.get_price_buffer("CRYPTO")
        # => 0.008
    """

    # Pre-defined risk profiles keyed by upper-case category name.
    DEFAULT_PROFILES: dict[str, CategoryRiskProfile] = {
        "SPORTS": CategoryRiskProfile(
            category="SPORTS",
            display_name="Sports",
            extra_slippage_tolerance=1.5,
            price_buffer=0.01,
            min_liquidity_multiplier=1.5,
            position_size_multiplier=0.7,
            volatility_rating="very_high",
            description="Live sporting events cause rapid price swings",
        ),
        "CRYPTO": CategoryRiskProfile(
            category="CRYPTO",
            display_name="Crypto",
            extra_slippage_tolerance=1.0,
            price_buffer=0.008,
            min_liquidity_multiplier=1.3,
            position_size_multiplier=0.8,
            volatility_rating="high",
            description="Crypto markets are volatile and correlated",
        ),
        "CULTURE": CategoryRiskProfile(
            category="CULTURE",
            display_name="Pop Culture",
            extra_slippage_tolerance=0.5,
            price_buffer=0.005,
            min_liquidity_multiplier=1.1,
            position_size_multiplier=0.9,
            volatility_rating="medium",
            description="Pop culture events have moderate volatility",
        ),
        "POLITICS": CategoryRiskProfile(
            category="POLITICS",
            display_name="Politics",
            extra_slippage_tolerance=0.2,
            price_buffer=0.002,
            min_liquidity_multiplier=1.0,
            position_size_multiplier=1.0,
            volatility_rating="low",
            description="Political markets are relatively stable",
        ),
        "WEATHER": CategoryRiskProfile(
            category="WEATHER",
            display_name="Weather",
            extra_slippage_tolerance=0.1,
            price_buffer=0.001,
            min_liquidity_multiplier=1.0,
            position_size_multiplier=1.0,
            volatility_rating="low",
            description="Weather markets have predictable volatility",
        ),
        "ECONOMICS": CategoryRiskProfile(
            category="ECONOMICS",
            display_name="Economics",
            extra_slippage_tolerance=0.3,
            price_buffer=0.003,
            min_liquidity_multiplier=1.1,
            position_size_multiplier=0.95,
            volatility_rating="medium",
            description="Economic indicators have moderate volatility",
        ),
        "TECH": CategoryRiskProfile(
            category="TECH",
            display_name="Technology",
            extra_slippage_tolerance=0.3,
            price_buffer=0.003,
            min_liquidity_multiplier=1.0,
            position_size_multiplier=0.95,
            volatility_rating="medium",
            description="Tech markets have moderate volatility",
        ),
        "FINANCE": CategoryRiskProfile(
            category="FINANCE",
            display_name="Finance",
            extra_slippage_tolerance=0.5,
            price_buffer=0.005,
            min_liquidity_multiplier=1.2,
            position_size_multiplier=0.85,
            volatility_rating="medium",
            description="Financial markets can be volatile",
        ),
    }

    # Neutral fallback profile applied when the category is unknown.
    _DEFAULT_PROFILE = CategoryRiskProfile(
        category="UNKNOWN",
        display_name="Unknown",
        extra_slippage_tolerance=0.0,
        price_buffer=0.0,
        min_liquidity_multiplier=1.0,
        position_size_multiplier=1.0,
        volatility_rating="medium",
        description="Unknown category - neutral risk profile applied",
    )

    def __init__(self) -> None:
        self._profiles: dict[str, CategoryRiskProfile] = dict(self.DEFAULT_PROFILES)
        logger.info(
            "CategoryBufferService initialized",
            categories=list(self._profiles.keys()),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_profile(self, category: str) -> CategoryRiskProfile:
        """Get risk profile for a category.

        Parameters
        ----------
        category : str
            Market category identifier (case-insensitive).

        Returns
        -------
        CategoryRiskProfile
            The matching profile, or a neutral default if the category
            is not recognised.
        """
        return self._profiles.get(category.upper(), self._DEFAULT_PROFILE)

    def adjust_slippage_tolerance(self, base_tolerance: float, category: str) -> float:
        """Adjust slippage tolerance based on category.

        The category's ``extra_slippage_tolerance`` is added to the
        *base_tolerance* so that volatile categories receive wider
        slippage allowances.

        Parameters
        ----------
        base_tolerance : float
            The base slippage tolerance percentage.
        category : str
            Market category identifier (case-insensitive).

        Returns
        -------
        float
            The adjusted slippage tolerance.
        """
        profile = self.get_profile(category)
        adjusted = base_tolerance + profile.extra_slippage_tolerance
        logger.debug(
            "Adjusted slippage tolerance",
            category=category.upper(),
            base=base_tolerance,
            extra=profile.extra_slippage_tolerance,
            adjusted=adjusted,
        )
        return adjusted

    def adjust_min_liquidity(self, base_liquidity: float, category: str) -> float:
        """Adjust minimum liquidity requirement based on category.

        The base liquidity is multiplied by the category's
        ``min_liquidity_multiplier`` so that volatile categories demand
        deeper order books before a trade is allowed.

        Parameters
        ----------
        base_liquidity : float
            The base minimum liquidity in USD.
        category : str
            Market category identifier (case-insensitive).

        Returns
        -------
        float
            The adjusted minimum liquidity requirement.
        """
        profile = self.get_profile(category)
        adjusted = base_liquidity * profile.min_liquidity_multiplier
        logger.debug(
            "Adjusted minimum liquidity",
            category=category.upper(),
            base=base_liquidity,
            multiplier=profile.min_liquidity_multiplier,
            adjusted=adjusted,
        )
        return adjusted

    def adjust_position_size(self, base_size: float, category: str) -> float:
        """Adjust position size based on category risk profile.

        The base position size is scaled by the category's
        ``position_size_multiplier``.  Volatile categories use
        multipliers below 1.0 to reduce exposure.

        Parameters
        ----------
        base_size : float
            The base position size in USD.
        category : str
            Market category identifier (case-insensitive).

        Returns
        -------
        float
            The adjusted position size.
        """
        profile = self.get_profile(category)
        adjusted = base_size * profile.position_size_multiplier
        logger.debug(
            "Adjusted position size",
            category=category.upper(),
            base=base_size,
            multiplier=profile.position_size_multiplier,
            adjusted=adjusted,
        )
        return adjusted

    def get_price_buffer(self, category: str) -> float:
        """Get the additional price buffer for a category.

        Parameters
        ----------
        category : str
            Market category identifier (case-insensitive).

        Returns
        -------
        float
            Extra price buffer in dollars that should be added to
            the trade price to account for category-specific volatility.
        """
        profile = self.get_profile(category)
        return profile.price_buffer

    def get_all_profiles(self) -> dict[str, CategoryRiskProfile]:
        """Get all risk profiles.

        Returns
        -------
        dict[str, CategoryRiskProfile]
            A copy of the internal profiles dictionary keyed by
            upper-case category name.
        """
        return dict(self._profiles)

    async def log_buffer_application(
        self,
        opportunity_id: str,
        category: str,
        adjustments: dict,
    ) -> None:
        """Log a buffer application to the database for analytics.

        Parameters
        ----------
        opportunity_id : str
            Identifier of the trading opportunity that triggered the
            buffer application.
        category : str
            The market category (case-insensitive).
        adjustments : dict
            Dictionary containing any of the following optional keys:

            - ``base_slippage`` / ``adjusted_slippage``
            - ``base_liquidity`` / ``adjusted_liquidity``
            - ``base_position_size`` / ``adjusted_position_size``
            - ``price_buffer_applied``
        """
        try:
            async with AsyncSessionLocal() as session:
                record = CategoryBufferLog(
                    id=str(uuid.uuid4()),
                    opportunity_id=opportunity_id,
                    category=category.upper(),
                    base_slippage=adjustments.get("base_slippage"),
                    adjusted_slippage=adjustments.get("adjusted_slippage"),
                    base_liquidity=adjustments.get("base_liquidity"),
                    adjusted_liquidity=adjustments.get("adjusted_liquidity"),
                    base_position_size=adjustments.get("base_position_size"),
                    adjusted_position_size=adjustments.get("adjusted_position_size"),
                    price_buffer_applied=adjustments.get("price_buffer_applied"),
                    applied_at=datetime.utcnow(),
                )
                session.add(record)
                await session.commit()
            logger.info(
                "Logged category buffer application",
                opportunity_id=opportunity_id,
                category=category.upper(),
            )
        except Exception as exc:
            logger.error(
                "Failed to log category buffer application",
                opportunity_id=opportunity_id,
                category=category.upper(),
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

category_buffer_service = CategoryBufferService()
