"""
Price Chaser Service - Adaptive price chasing on order retries.

Inspired by terauss/Polymarket-Copy-Trading-Bot.

When an order fails or isn't filled, instead of retrying at the same price,
this service adjusts the limit price to chase the market:
- BUY orders: increase price on each retry (willing to pay more)
- SELL orders: decrease price on each retry (willing to accept less)
- Final retry switches to GTD order type as a last resort
- Never chases beyond MAX_SLIPPAGE_PERCENT from the original price

All retry attempts are logged to the database for post-hoc analysis.
"""

import asyncio
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import Callable, Optional

from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, JSON, Index

from models.database import Base, AsyncSessionLocal
from utils.logger import get_logger

logger = get_logger("price_chaser")


# ==================== SQLAlchemy Model ====================


class OrderRetryLog(Base):
    """Persistent log of every order retry sequence for analysis."""

    __tablename__ = "order_retry_logs"

    id = Column(String, primary_key=True)
    order_id = Column(String, nullable=True)
    opportunity_id = Column(String, nullable=True)
    token_id = Column(String, nullable=False)
    side = Column(String, nullable=False)
    original_price = Column(Float, nullable=False)
    final_price = Column(Float, nullable=True)
    total_attempts = Column(Integer, default=1)
    total_filled = Column(Float, default=0.0)
    total_price_adjustment = Column(Float, default=0.0)
    success = Column(Boolean, default=False)
    attempts_data = Column(JSON)  # List of retry attempt details
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_retry_token", "token_id"),
        Index("idx_retry_success", "success"),
    )


# ==================== Data Classes ====================


@dataclass
class PriceChaseConfig:
    """Configuration knobs for the price chasing algorithm."""

    max_retries: int = 5
    price_increment_per_retry: float = 0.005  # $0.005 per retry
    max_total_chase: float = 0.02  # Maximum total price adjustment
    final_retry_order_type: str = "GTD"  # Switch to GTD on last retry
    final_retry_gtd_seconds: int = 61  # GTD expiration
    max_slippage_percent: float = 2.0  # Never exceed this slippage from original
    chase_on_first_retry: bool = True  # Start chasing immediately


@dataclass
class RetryAttempt:
    """Record of a single retry attempt within a chase sequence."""

    attempt_number: int
    original_price: float
    adjusted_price: float
    price_adjustment: float
    order_type: str
    result: str  # "filled", "partial", "failed", "timeout"
    fill_size: float
    attempted_at: datetime


# ==================== Service ====================


class PriceChaserService:
    """Orchestrates price-chasing retries for order execution.

    On each retry the limit price is nudged toward the market so the order
    is more likely to fill.  The final attempt switches to a GTD order as
    a last-ditch effort.  Slippage is hard-capped at
    ``max_slippage_percent`` of the original price.

    Every retry sequence is persisted to the ``order_retry_logs`` table.
    """

    def __init__(self, config: PriceChaseConfig = None):
        self.config = config or PriceChaseConfig()

    # ------------------------------------------------------------------
    # Price calculation
    # ------------------------------------------------------------------

    def calculate_chase_price(
        self,
        original_price: float,
        side: str,  # "BUY" or "SELL"
        attempt: int,
        current_market_price: float = None,
    ) -> tuple[float, str]:
        """Calculate the chased price for a retry attempt.

        Returns ``(adjusted_price, order_type)``.

        Logic:
        - For BUY: price goes UP on each retry (willing to pay more)
        - For SELL: price goes DOWN on each retry (willing to accept less)
        - On the final attempt, switches to GTD order type
        - Clamps to ``[0.01, 0.99]`` valid price range
        - Never exceeds ``max_slippage_percent`` from original
        """

        # Determine order type -- final attempt uses GTD
        is_final = attempt >= self.config.max_retries
        order_type = self.config.final_retry_order_type if is_final else "GTC"

        # First attempt (attempt == 0) uses the original price unless
        # chase_on_first_retry is True *and* this is already a retry (>0).
        if attempt == 0:
            return (max(0.01, min(0.99, original_price)), order_type)

        # How many chase increments to apply
        chase_steps = attempt if self.config.chase_on_first_retry else max(0, attempt - 1)
        raw_adjustment = chase_steps * self.config.price_increment_per_retry

        # Cap at max_total_chase
        raw_adjustment = min(raw_adjustment, self.config.max_total_chase)

        # Cap at max_slippage_percent of the original price
        max_slippage = original_price * (self.config.max_slippage_percent / 100.0)
        raw_adjustment = min(raw_adjustment, max_slippage)

        # Apply adjustment in the correct direction
        side_upper = side.upper()
        if side_upper == "BUY":
            adjusted = original_price + raw_adjustment
        else:  # SELL
            adjusted = original_price - raw_adjustment

        # Clamp to valid Polymarket price range [0.01, 0.99]
        adjusted = max(0.01, min(0.99, adjusted))

        return (round(adjusted, 4), order_type)

    # ------------------------------------------------------------------
    # Execution with chase
    # ------------------------------------------------------------------

    async def execute_with_chase(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
        place_order_fn: Callable,  # async fn that places the order
        get_market_price_fn: Callable = None,  # async fn to get current price
        opportunity_id: str = None,
        tier: int = None,  # From ExecutionTierService
    ) -> dict:
        """Execute an order with price chasing retries.

        Parameters
        ----------
        token_id : str
            The CLOB token to trade.
        side : str
            ``"BUY"`` or ``"SELL"``.
        price : float
            The initial limit price.
        size : float
            Number of shares to trade.
        place_order_fn : Callable
            An ``async`` callable with signature
            ``(token_id, side, price, size, order_type, **kwargs) -> dict``.
            Must return a dict with at least ``"status"`` (str) and
            ``"filled_size"`` (float).  Optionally ``"order_id"`` (str).
        get_market_price_fn : Callable, optional
            An ``async`` callable ``(token_id, side) -> float`` that
            returns the current best market price.
        opportunity_id : str, optional
            Opportunity ID for cross-referencing.
        tier : int, optional
            Execution tier (from ExecutionTierService) for logging.

        Returns
        -------
        dict
            ``success``            - bool
            ``final_price``        - float
            ``total_filled``       - float
            ``attempts``           - list[RetryAttempt]
            ``total_price_adjustment`` - float
        """

        original_price = price
        remaining_size = size
        total_filled = 0.0
        attempts: list[RetryAttempt] = []
        last_order_id: Optional[str] = None

        for attempt in range(self.config.max_retries + 1):
            # Optionally refresh market price
            current_market_price = None
            if get_market_price_fn is not None:
                try:
                    current_market_price = await get_market_price_fn(token_id, side)
                except Exception as exc:
                    logger.warning(
                        "Failed to fetch market price for chase",
                        token_id=token_id,
                        error=str(exc),
                    )

            # Calculate chased price
            adjusted_price, order_type = self.calculate_chase_price(
                original_price=original_price,
                side=side,
                attempt=attempt,
                current_market_price=current_market_price,
            )

            price_adjustment = round(adjusted_price - original_price, 6)

            logger.info(
                "Price chase attempt",
                attempt=attempt,
                original_price=original_price,
                adjusted_price=adjusted_price,
                order_type=order_type,
                remaining_size=remaining_size,
                tier=tier,
            )

            # Build extra kwargs for the place_order_fn
            extra_kwargs: dict = {}
            if order_type == "GTD":
                extra_kwargs["gtd_seconds"] = self.config.final_retry_gtd_seconds

            # Place the order
            try:
                result = await place_order_fn(
                    token_id,
                    side,
                    adjusted_price,
                    remaining_size,
                    order_type,
                    **extra_kwargs,
                )
            except Exception as exc:
                logger.error(
                    "Order placement raised exception during chase",
                    attempt=attempt,
                    error=str(exc),
                )
                result = {"status": "failed", "filled_size": 0.0}

            status = result.get("status", "failed")
            fill_size = float(result.get("filled_size", 0.0))
            last_order_id = result.get("order_id", last_order_id)

            # Classify the result
            if status == "filled" or (fill_size > 0 and fill_size >= remaining_size):
                outcome = "filled"
            elif fill_size > 0:
                outcome = "partial"
            elif status in ("timeout", "expired"):
                outcome = "timeout"
            else:
                outcome = "failed"

            retry_record = RetryAttempt(
                attempt_number=attempt,
                original_price=original_price,
                adjusted_price=adjusted_price,
                price_adjustment=price_adjustment,
                order_type=order_type,
                result=outcome,
                fill_size=fill_size,
                attempted_at=datetime.utcnow(),
            )
            attempts.append(retry_record)

            total_filled += fill_size
            remaining_size -= fill_size
            remaining_size = max(remaining_size, 0.0)

            # If fully filled, we are done
            if remaining_size <= 0 or outcome == "filled":
                logger.info(
                    "Order fully filled via price chase",
                    total_attempts=attempt + 1,
                    total_filled=total_filled,
                    final_price=adjusted_price,
                )
                break

            # If this was the last attempt, stop
            if attempt >= self.config.max_retries:
                logger.warning(
                    "Price chase exhausted all retries",
                    total_attempts=attempt + 1,
                    total_filled=total_filled,
                )
                break

            # Brief pause before next retry to let the book update
            await asyncio.sleep(1)

        # Determine success
        success = total_filled > 0 and remaining_size <= 0
        final_price = attempts[-1].adjusted_price if attempts else original_price
        total_price_adjustment = round(final_price - original_price, 6)

        # Persist to database
        await self._save_retry_log(
            token_id=token_id,
            side=side,
            original_price=original_price,
            final_price=final_price,
            total_attempts=len(attempts),
            total_filled=total_filled,
            total_price_adjustment=total_price_adjustment,
            success=success,
            attempts=attempts,
            order_id=last_order_id,
            opportunity_id=opportunity_id,
        )

        return {
            "success": success,
            "final_price": final_price,
            "total_filled": total_filled,
            "attempts": attempts,
            "total_price_adjustment": total_price_adjustment,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def _save_retry_log(
        self,
        token_id: str,
        side: str,
        original_price: float,
        final_price: float,
        total_attempts: int,
        total_filled: float,
        total_price_adjustment: float,
        success: bool,
        attempts: list[RetryAttempt],
        order_id: Optional[str] = None,
        opportunity_id: Optional[str] = None,
    ) -> None:
        """Persist the full retry sequence to the database."""
        attempts_data = [
            {
                "attempt_number": a.attempt_number,
                "original_price": a.original_price,
                "adjusted_price": a.adjusted_price,
                "price_adjustment": a.price_adjustment,
                "order_type": a.order_type,
                "result": a.result,
                "fill_size": a.fill_size,
                "attempted_at": a.attempted_at.isoformat(),
            }
            for a in attempts
        ]

        try:
            async with AsyncSessionLocal() as session:
                log_entry = OrderRetryLog(
                    id=str(uuid.uuid4()),
                    order_id=order_id,
                    opportunity_id=opportunity_id,
                    token_id=token_id,
                    side=side,
                    original_price=original_price,
                    final_price=final_price,
                    total_attempts=total_attempts,
                    total_filled=total_filled,
                    total_price_adjustment=total_price_adjustment,
                    success=success,
                    attempts_data=attempts_data,
                )
                session.add(log_entry)
                await session.commit()

                logger.info(
                    "Saved retry log",
                    log_id=log_entry.id,
                    success=success,
                    total_attempts=total_attempts,
                    total_filled=total_filled,
                )
        except Exception as exc:
            logger.error("Failed to save retry log", error=str(exc))


# ==================== Singleton ====================

price_chaser = PriceChaserService()
