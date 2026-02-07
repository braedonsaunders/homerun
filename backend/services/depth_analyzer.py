"""
Order Book Depth Analyzer

Analyzes order book depth before trade execution to ensure sufficient
liquidity exists at acceptable price levels. Inspired by the
terauss/Polymarket-Copy-Trading-Bot approach to pre-trade validation.

The service fetches the CLOB order book, calculates executable depth at a
target price, estimates VWAP and slippage, and blocks trades that fall
below a configurable minimum depth threshold (default $200).

All depth check results are persisted to the database for auditing.
"""

import uuid
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List

from sqlalchemy import Column, String, Float, Boolean, DateTime

from models.database import Base, AsyncSessionLocal
from services.polymarket import polymarket_client
from utils.logger import get_logger

logger = get_logger("depth_analyzer")

# Minimum depth in USD required to allow a trade through.
# Can be overridden at construction time or per-call.
MIN_DEPTH_USD = 200.0


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class DepthCheckResult:
    """Result of an order book depth check for a proposed trade."""

    token_id: str
    side: str
    has_sufficient_depth: bool
    available_depth_usd: float
    required_depth_usd: float
    best_price: float
    vwap_price: float  # volume-weighted average price for requested size
    slippage_percent: float  # estimated slippage from mid to VWAP
    checked_at: datetime


# ---------------------------------------------------------------------------
# SQLAlchemy audit model
# ---------------------------------------------------------------------------


class DepthCheck(Base):
    """Persisted record of a depth check for auditing purposes."""

    __tablename__ = "depth_checks"

    id = Column(String, primary_key=True)
    token_id = Column(String, nullable=False)
    side = Column(String, nullable=False)
    available_depth_usd = Column(Float)
    required_depth_usd = Column(Float)
    passed = Column(Boolean)
    best_price = Column(Float)
    vwap_price = Column(Float)
    slippage_percent = Column(Float)
    checked_at = Column(DateTime, default=datetime.utcnow)
    trade_context = Column(String, nullable=True)  # "auto_trader" or "copy_trader"


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class DepthAnalyzer:
    """Analyses order book depth before trade execution.

    Usage
    -----
    ::

        result = await depth_analyzer.check_depth(
            token_id="0xabc...",
            side="BUY",
            target_price=0.55,
            required_size_usd=500.0,
        )
        if not result.has_sufficient_depth:
            logger.warning("Insufficient depth, blocking trade")

    Parameters
    ----------
    min_depth_usd : float
        Global minimum depth threshold.  Trades below this level are
        flagged as having insufficient depth.  Defaults to ``MIN_DEPTH_USD``.
    """

    def __init__(self, min_depth_usd: float = MIN_DEPTH_USD) -> None:
        self.min_depth_usd = min_depth_usd
        logger.info(
            "DepthAnalyzer initialized",
            min_depth_usd=self.min_depth_usd,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_order_book_levels(
        order_book: dict, side: str
    ) -> List[tuple]:
        """Return a list of ``(price, size)`` tuples from the raw order book.

        For a **BUY** trade we consume from the *asks* (sorted ascending by
        price so we eat the cheapest asks first).

        For a **SELL** trade we consume from the *bids* (sorted descending by
        price so we hit the best bids first).

        Each level is returned as ``(float_price, float_size)``.
        """
        side_upper = side.upper()
        if side_upper == "BUY":
            raw_levels = order_book.get("asks", [])
            levels = [
                (float(lvl["price"]), float(lvl["size"]))
                for lvl in raw_levels
            ]
            # Ascending: cheapest asks first
            levels.sort(key=lambda x: x[0])
        else:
            raw_levels = order_book.get("bids", [])
            levels = [
                (float(lvl["price"]), float(lvl["size"]))
                for lvl in raw_levels
            ]
            # Descending: best (highest) bids first
            levels.sort(key=lambda x: x[0], reverse=True)
        return levels

    @staticmethod
    def _calculate_mid_price(order_book: dict) -> float:
        """Derive mid-price from the best bid and best ask.

        Returns 0.0 if either side of the book is empty.
        """
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])
        if not bids or not asks:
            return 0.0
        best_bid = max(float(b["price"]) for b in bids)
        best_ask = min(float(a["price"]) for a in asks)
        return (best_bid + best_ask) / 2.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def check_depth(
        self,
        token_id: str,
        side: str,
        target_price: float,
        required_size_usd: float,
        trade_context: Optional[str] = None,
    ) -> DepthCheckResult:
        """Check whether sufficient order book depth exists for a trade.

        Parameters
        ----------
        token_id : str
            The CLOB token identifier.
        side : str
            ``"BUY"`` or ``"SELL"``.
        target_price : float
            The worst acceptable price for the trade.
        required_size_usd : float
            How much USD-equivalent liquidity the trade needs.
        trade_context : str, optional
            Free-form label persisted for auditing (e.g. ``"auto_trader"``).

        Returns
        -------
        DepthCheckResult
            Detailed depth analysis including VWAP, slippage, and a
            boolean ``has_sufficient_depth`` flag.
        """
        now = datetime.utcnow()
        side_upper = side.upper()

        try:
            order_book = await polymarket_client.get_order_book(token_id)
        except Exception as exc:
            logger.error(
                "Failed to fetch order book",
                token_id=token_id,
                error=str(exc),
            )
            # Return a conservative "no depth" result on API failure
            result = DepthCheckResult(
                token_id=token_id,
                side=side_upper,
                has_sufficient_depth=False,
                available_depth_usd=0.0,
                required_depth_usd=required_size_usd,
                best_price=0.0,
                vwap_price=0.0,
                slippage_percent=0.0,
                checked_at=now,
            )
            await self._persist_result(result, trade_context)
            return result

        levels = self._parse_order_book_levels(order_book, side_upper)
        mid_price = self._calculate_mid_price(order_book)

        # --- Calculate available depth at or better than target_price ---
        available_depth_usd = 0.0
        for price, size in levels:
            if side_upper == "BUY" and price > target_price:
                break
            if side_upper == "SELL" and price < target_price:
                break
            available_depth_usd += price * size

        # --- Best price ---
        best_price = levels[0][0] if levels else 0.0

        # --- VWAP for the requested size ---
        vwap_price = self._compute_vwap(levels, required_size_usd)

        # --- Slippage from mid to VWAP ---
        if mid_price > 0 and vwap_price > 0:
            slippage_percent = abs(vwap_price - mid_price) / mid_price * 100.0
        else:
            slippage_percent = 0.0

        # --- Determine if depth is sufficient ---
        min_required = max(self.min_depth_usd, required_size_usd)
        has_sufficient_depth = available_depth_usd >= min_required

        result = DepthCheckResult(
            token_id=token_id,
            side=side_upper,
            has_sufficient_depth=has_sufficient_depth,
            available_depth_usd=available_depth_usd,
            required_depth_usd=required_size_usd,
            best_price=best_price,
            vwap_price=vwap_price,
            slippage_percent=round(slippage_percent, 4),
            checked_at=now,
        )

        await self._persist_result(result, trade_context)

        logger.info(
            "Depth check completed",
            token_id=token_id,
            side=side_upper,
            available_depth_usd=round(available_depth_usd, 2),
            required_size_usd=required_size_usd,
            has_sufficient_depth=has_sufficient_depth,
            best_price=best_price,
            vwap_price=round(vwap_price, 6),
            slippage_percent=round(slippage_percent, 4),
        )

        return result

    async def calculate_depth_at_price(
        self, token_id: str, side: str, price: float
    ) -> float:
        """Calculate total executable depth (USD) at or better than *price*.

        Parameters
        ----------
        token_id : str
            The CLOB token identifier.
        side : str
            ``"BUY"`` or ``"SELL"``.
        price : float
            The worst acceptable price level.

        Returns
        -------
        float
            Total USD-equivalent liquidity available at or better than
            the given price.
        """
        side_upper = side.upper()

        try:
            order_book = await polymarket_client.get_order_book(token_id)
        except Exception as exc:
            logger.error(
                "Failed to fetch order book for depth calculation",
                token_id=token_id,
                error=str(exc),
            )
            return 0.0

        levels = self._parse_order_book_levels(order_book, side_upper)

        depth_usd = 0.0
        for level_price, size in levels:
            if side_upper == "BUY" and level_price > price:
                break
            if side_upper == "SELL" and level_price < price:
                break
            depth_usd += level_price * size

        return depth_usd

    async def get_executable_price(
        self, token_id: str, side: str, size: float
    ) -> float:
        """Compute the volume-weighted average price (VWAP) to fill *size* USD.

        Walks through the order book consuming levels until the requested
        USD amount has been filled.

        Parameters
        ----------
        token_id : str
            The CLOB token identifier.
        side : str
            ``"BUY"`` or ``"SELL"``.
        size : float
            The USD amount to fill.

        Returns
        -------
        float
            The VWAP across all levels consumed.  Returns ``0.0`` when the
            order book cannot fill the requested size or the book is empty.
        """
        side_upper = side.upper()

        try:
            order_book = await polymarket_client.get_order_book(token_id)
        except Exception as exc:
            logger.error(
                "Failed to fetch order book for VWAP calculation",
                token_id=token_id,
                error=str(exc),
            )
            return 0.0

        levels = self._parse_order_book_levels(order_book, side_upper)
        return self._compute_vwap(levels, size)

    # ------------------------------------------------------------------
    # Internal: VWAP computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_vwap(
        levels: List[tuple], target_usd: float
    ) -> float:
        """Walk order book levels to compute VWAP for *target_usd*.

        Parameters
        ----------
        levels : list[tuple[float, float]]
            Sorted ``(price, size)`` tuples (cheapest asks first for BUY,
            best bids first for SELL).
        target_usd : float
            The USD amount to fill.

        Returns
        -------
        float
            Volume-weighted average price.  Returns ``0.0`` if the book is
            empty or has no liquidity.
        """
        if not levels or target_usd <= 0:
            return 0.0

        filled_usd = 0.0
        weighted_sum = 0.0  # sum of (price * usd_consumed_at_this_level)

        for price, size in levels:
            level_usd = price * size
            remaining = target_usd - filled_usd

            if level_usd >= remaining:
                # This level can fill the rest
                weighted_sum += price * remaining
                filled_usd += remaining
                break
            else:
                weighted_sum += price * level_usd
                filled_usd += level_usd

        if filled_usd <= 0:
            return 0.0

        return weighted_sum / filled_usd

    # ------------------------------------------------------------------
    # Internal: persistence
    # ------------------------------------------------------------------

    async def _persist_result(
        self,
        result: DepthCheckResult,
        trade_context: Optional[str] = None,
    ) -> None:
        """Store the depth check result in the database for auditing."""
        try:
            async with AsyncSessionLocal() as session:
                record = DepthCheck(
                    id=str(uuid.uuid4()),
                    token_id=result.token_id,
                    side=result.side,
                    available_depth_usd=result.available_depth_usd,
                    required_depth_usd=result.required_depth_usd,
                    passed=result.has_sufficient_depth,
                    best_price=result.best_price,
                    vwap_price=result.vwap_price,
                    slippage_percent=result.slippage_percent,
                    checked_at=result.checked_at,
                    trade_context=trade_context,
                )
                session.add(record)
                await session.commit()
        except Exception as exc:
            logger.error(
                "Failed to persist depth check result",
                token_id=result.token_id,
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

depth_analyzer = DepthAnalyzer()
