"""
Parallel execution for non-atomic arbitrage trades.

Critical insight from research: CLOB execution is sequential, not atomic.
If you execute orders one-by-one, prices move between legs, eating profits.

Solution: Submit ALL orders in parallel so they're included in the same
block (~2 seconds on Polygon), eliminating sequential execution risk.

Research findings:
- Fast wallets submit all legs within 30ms
- By the time retail sees confirmation, arbitrage is already captured
- Partial execution creates exposure risk
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable, Any
from enum import Enum

from utils.logger import get_logger

logger = get_logger(__name__)


class ExecutionStatus(str, Enum):
    """Status of parallel execution."""

    SUCCESS = "success"  # All legs filled
    PARTIAL = "partial"  # Some legs filled, exposure risk
    FAILED = "failed"  # No legs filled
    TIMEOUT = "timeout"  # Execution timed out


@dataclass
class ExecutionLeg:
    """Single leg of a multi-leg arbitrage trade."""

    token_id: str
    side: str  # BUY or SELL
    price: float
    size_usd: float
    market_question: Optional[str] = None


@dataclass
class LegResult:
    """Result of executing a single leg."""

    leg: ExecutionLeg
    success: bool
    order_id: Optional[str] = None
    fill_price: Optional[float] = None
    fill_size: Optional[float] = None
    error: Optional[str] = None
    latency_ms: float = 0


@dataclass
class ParallelExecutionResult:
    """Result of parallel execution attempt."""

    status: ExecutionStatus
    legs: list[LegResult]
    total_latency_ms: float
    successful_legs: int
    failed_legs: int
    total_cost: float
    exposure_risk: bool  # True if partial fill
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ParallelExecutor:
    """
    Execute multiple order legs in parallel for atomic-like behavior.

    Key optimizations:
    1. Pre-validate all legs before any execution
    2. Submit all orders via asyncio.gather (not sequential await)
    3. Track partial execution for risk management
    4. Configurable timeout and retry logic
    """

    def __init__(
        self,
        order_placer: Callable[[ExecutionLeg], Any],
        max_latency_ms: float = 2000.0,  # 1 Polygon block
        retry_failed_legs: bool = False,
        max_retries: int = 2,
    ):
        """
        Args:
            order_placer: Async function that places a single order.
                         Should accept ExecutionLeg and return order result.
            max_latency_ms: Maximum time to wait for all legs (default 1 block)
            retry_failed_legs: Whether to retry failed legs
            max_retries: Maximum retry attempts per leg
        """
        self.order_placer = order_placer
        self.max_latency_ms = max_latency_ms
        self.retry_failed_legs = retry_failed_legs
        self.max_retries = max_retries

    async def execute(self, legs: list[ExecutionLeg], opportunity_id: Optional[str] = None) -> ParallelExecutionResult:
        """
        Execute all legs in parallel.

        This is the critical function that transforms sequential execution
        (each order waits for previous) into parallel execution (all orders
        submitted simultaneously).

        Args:
            legs: List of execution legs
            opportunity_id: Optional ID for tracking

        Returns:
            ParallelExecutionResult with all leg outcomes
        """
        if not legs:
            return ParallelExecutionResult(
                status=ExecutionStatus.FAILED,
                legs=[],
                total_latency_ms=0,
                successful_legs=0,
                failed_legs=0,
                total_cost=0,
                exposure_risk=False,
            )

        start_time = time.perf_counter()

        # Pre-validation: Check all legs before executing any
        validation_errors = self._validate_legs(legs)
        if any(e is not None for e in validation_errors):
            logger.error(f"Leg validation failed: {validation_errors}")
            return ParallelExecutionResult(
                status=ExecutionStatus.FAILED,
                legs=[LegResult(leg=leg, success=False, error=err) for leg, err in zip(legs, validation_errors)],
                total_latency_ms=(time.perf_counter() - start_time) * 1000,
                successful_legs=0,
                failed_legs=len(legs),
                total_cost=0,
                exposure_risk=False,
            )

        # Execute ALL legs in parallel using asyncio.gather
        # This is the key: gather submits all coroutines before awaiting any
        tasks = [self._execute_single_leg(leg, opportunity_id) for leg in legs]

        try:
            # Set timeout to max_latency_ms
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.max_latency_ms / 1000,
            )
        except asyncio.TimeoutError:
            logger.error(f"Parallel execution timed out after {self.max_latency_ms}ms")
            return ParallelExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                legs=[LegResult(leg=leg, success=False, error="Timeout") for leg in legs],
                total_latency_ms=self.max_latency_ms,
                successful_legs=0,
                failed_legs=len(legs),
                total_cost=0,
                exposure_risk=True,  # Unknown state after timeout
            )

        # Process results
        leg_results = []
        successful = 0
        failed = 0
        total_cost = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                leg_results.append(LegResult(leg=legs[i], success=False, error=str(result)))
                failed += 1
            elif isinstance(result, LegResult):
                leg_results.append(result)
                if result.success:
                    successful += 1
                    total_cost += result.fill_price * result.fill_size if result.fill_price else legs[i].size_usd
                else:
                    failed += 1
            else:
                leg_results.append(LegResult(leg=legs[i], success=False, error="Unknown result type"))
                failed += 1

        total_latency = (time.perf_counter() - start_time) * 1000

        # Determine execution status
        if successful == len(legs):
            status = ExecutionStatus.SUCCESS
        elif successful == 0:
            status = ExecutionStatus.FAILED
        else:
            status = ExecutionStatus.PARTIAL

        # Exposure risk if partial execution
        exposure_risk = 0 < successful < len(legs)
        if exposure_risk:
            logger.warning(f"PARTIAL EXECUTION: {successful}/{len(legs)} legs filled. Position has exposure risk!")

        return ParallelExecutionResult(
            status=status,
            legs=leg_results,
            total_latency_ms=total_latency,
            successful_legs=successful,
            failed_legs=failed,
            total_cost=total_cost,
            exposure_risk=exposure_risk,
        )

    def _validate_legs(self, legs: list[ExecutionLeg]) -> list[Optional[str]]:
        """
        Pre-validate all legs before execution.

        Returns list of error messages (None if valid).
        """
        errors = []
        for leg in legs:
            if not leg.token_id:
                errors.append("Missing token_id")
            elif leg.price <= 0:
                errors.append(f"Invalid price: {leg.price}")
            elif leg.price > 1:
                errors.append(f"Price > 1: {leg.price}")
            elif leg.size_usd <= 0:
                errors.append(f"Invalid size: {leg.size_usd}")
            else:
                errors.append(None)
        return errors

    async def _execute_single_leg(self, leg: ExecutionLeg, opportunity_id: Optional[str]) -> LegResult:
        """Execute a single leg with timing."""
        start = time.perf_counter()

        try:
            # Call the provided order placer
            result = await self.order_placer(leg)

            latency = (time.perf_counter() - start) * 1000

            # Parse result based on what order_placer returns
            if hasattr(result, "status"):
                success = result.status in ["open", "filled", "OPEN", "FILLED"]
                return LegResult(
                    leg=leg,
                    success=success,
                    order_id=getattr(result, "id", None) or getattr(result, "clob_order_id", None),
                    fill_price=getattr(result, "average_fill_price", None) or leg.price,
                    fill_size=getattr(result, "filled_size", None) or (leg.size_usd / leg.price),
                    error=getattr(result, "error_message", None) if not success else None,
                    latency_ms=latency,
                )
            elif isinstance(result, dict):
                success = result.get("success", result.get("status") in ["open", "filled"])
                return LegResult(
                    leg=leg,
                    success=success,
                    order_id=result.get("order_id") or result.get("orderID"),
                    fill_price=result.get("fill_price", leg.price),
                    fill_size=result.get("fill_size", leg.size_usd / leg.price),
                    error=result.get("error") if not success else None,
                    latency_ms=latency,
                )
            else:
                return LegResult(
                    leg=leg,
                    success=True,
                    fill_price=leg.price,
                    fill_size=leg.size_usd / leg.price,
                    latency_ms=latency,
                )

        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return LegResult(leg=leg, success=False, error=str(e), latency_ms=latency)

    async def execute_with_hedge(
        self,
        legs: list[ExecutionLeg],
        hedge_on_partial: bool = True,
        opportunity_id: Optional[str] = None,
    ) -> ParallelExecutionResult:
        """
        Execute with automatic hedging on partial fills.

        If only some legs fill, we have directional exposure.
        This method can automatically close positions to eliminate risk.

        Args:
            legs: Execution legs
            hedge_on_partial: Whether to hedge partial fills
            opportunity_id: Tracking ID

        Returns:
            ParallelExecutionResult (may include hedge trades)
        """
        result = await self.execute(legs, opportunity_id)

        if result.exposure_risk and hedge_on_partial:
            logger.info("Attempting to hedge partial execution...")

            # Find filled legs that need hedging
            filled_legs = [r for r in result.legs if r.success]

            # Create opposite trades to close positions
            hedge_legs = []
            for filled in filled_legs:
                hedge_legs.append(
                    ExecutionLeg(
                        token_id=filled.leg.token_id,
                        side="SELL" if filled.leg.side == "BUY" else "BUY",
                        price=filled.fill_price or filled.leg.price,  # Market price
                        size_usd=filled.fill_size * (filled.fill_price or filled.leg.price),
                        market_question=filled.leg.market_question,
                    )
                )

            if hedge_legs:
                hedge_result = await self.execute(hedge_legs, f"{opportunity_id}_hedge")
                logger.info(f"Hedge result: {hedge_result.successful_legs}/{len(hedge_legs)} positions closed")

        return result


def create_legs_from_opportunity(positions: list[dict], total_size_usd: float) -> list[ExecutionLeg]:
    """
    Helper to create ExecutionLegs from opportunity positions.

    Args:
        positions: List of position dicts from ArbitrageOpportunity.positions_to_take
        total_size_usd: Total USD to invest (split equally among legs)

    Returns:
        List of ExecutionLeg objects ready for parallel execution
    """
    size_per_leg = total_size_usd / len(positions) if positions else 0

    legs = []
    for pos in positions:
        legs.append(
            ExecutionLeg(
                token_id=pos.get("token_id", ""),
                side=pos.get("action", "BUY"),
                price=pos.get("price", 0),
                size_usd=size_per_leg,
                market_question=pos.get("market", ""),
            )
        )

    return legs
