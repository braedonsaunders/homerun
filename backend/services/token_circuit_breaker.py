import uuid
from datetime import datetime, timedelta
from utils.utcnow import utcnow
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import Column, String, Integer, Boolean, DateTime, JSON, Index

from models.database import Base, AsyncSessionLocal
from utils.logger import get_logger

logger = get_logger("token_circuit_breaker")


# ==================== SQLAlchemy Model ====================


class TokenTrip(Base):
    """Persisted record of a token trip event for historical analysis."""

    __tablename__ = "token_trips"

    id = Column(String, primary_key=True)
    token_id = Column(String, nullable=False, index=True)
    reason = Column(String, nullable=False)
    trade_count = Column(Integer, default=0)
    triggered_at = Column(DateTime, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    cleared_at = Column(DateTime, nullable=True)
    was_auto_expired = Column(Boolean, default=False)
    details = Column(JSON, nullable=True)

    __table_args__ = (
        Index("idx_trip_token", "token_id"),
        Index("idx_trip_triggered", "triggered_at"),
    )


# ==================== Data Classes ====================


@dataclass
class TokenTripConfig:
    """Configuration for per-token circuit breaker thresholds."""

    large_trade_threshold_shares: float = 1500.0  # What counts as a "large" trade
    consecutive_trigger: int = 2  # Number of large trades to trigger trip
    detection_window_seconds: int = 30  # Window to detect rapid trading
    trip_duration_seconds: int = 120  # How long to block the token
    trip_on_api_error: bool = True  # Trip on API errors (conservative)


@dataclass
class TokenTripEvent:
    """Represents an active or historical trip event for a specific token."""

    token_id: str
    reason: str  # "rapid_large_trades", "api_error", "manual"
    triggered_at: datetime
    expires_at: datetime
    trade_count: int  # Number of trades that triggered the trip
    details: dict  # Additional context


# ==================== Token Circuit Breaker Service ====================


class TokenCircuitBreaker:
    """
    Per-token circuit breaker that tracks activity per token_id and blocks
    individual tokens showing suspicious rapid large trading.

    Inspired by terauss/Polymarket-Copy-Trading-Bot. Instead of pausing ALL
    trading globally, this isolates problematic tokens so other tokens can
    continue trading normally.

    Trip triggers:
    - 2+ large trades on the same token within 30 seconds
    - API errors during depth analysis for a specific token
    - Manual trip via trip_token()
    """

    def __init__(self, config: TokenTripConfig = None):
        self.config = config or TokenTripConfig()
        self._trips: dict[str, TokenTripEvent] = {}  # token_id -> active trip
        self._recent_trades: dict[str, list[dict]] = {}  # token_id -> recent trades
        self._trip_history: list[TokenTripEvent] = []  # All trips for stats
        self._api_errors: dict[str, list[datetime]] = {}  # token_id -> error timestamps
        self._max_consecutive_api_errors = 5

    def record_trade(self, token_id: str, size: float, price: float, side: str) -> Optional[TokenTripEvent]:
        """
        Record a trade and check if it triggers a trip.
        Returns TokenTripEvent if tripped, None otherwise.

        Logic:
        1. Add trade to recent trades for this token
        2. Clean up trades older than detection_window_seconds
        3. Count trades >= large_trade_threshold_shares in window
        4. If count >= consecutive_trigger, trip the token
        """
        now = utcnow()

        trade_record = {
            "size": size,
            "price": price,
            "side": side,
            "timestamp": now,
        }

        # Initialize token's trade list if needed
        if token_id not in self._recent_trades:
            self._recent_trades[token_id] = []

        self._recent_trades[token_id].append(trade_record)

        # Clean up trades older than the detection window
        cutoff = now - timedelta(seconds=self.config.detection_window_seconds)
        self._recent_trades[token_id] = [t for t in self._recent_trades[token_id] if t["timestamp"] >= cutoff]

        # Count large trades within the window
        large_trades = [
            t for t in self._recent_trades[token_id] if t["size"] >= self.config.large_trade_threshold_shares
        ]

        if len(large_trades) >= self.config.consecutive_trigger:
            # Already tripped? Don't re-trip, just log
            is_already_tripped, _ = self.is_tripped(token_id)
            if is_already_tripped:
                logger.warning(
                    "Token already tripped, ignoring additional trigger",
                    token_id=token_id,
                    large_trade_count=len(large_trades),
                )
                return self._trips[token_id]

            details = {
                "large_trades_in_window": [
                    {
                        "size": t["size"],
                        "price": t["price"],
                        "side": t["side"],
                        "timestamp": t["timestamp"].isoformat(),
                    }
                    for t in large_trades
                ],
                "window_seconds": self.config.detection_window_seconds,
                "threshold_shares": self.config.large_trade_threshold_shares,
            }

            event = self.trip_token(
                token_id=token_id,
                reason="rapid_large_trades",
                details=details,
                trade_count=len(large_trades),
            )

            logger.warning(
                "Token tripped due to rapid large trades",
                token_id=token_id,
                large_trade_count=len(large_trades),
                window_seconds=self.config.detection_window_seconds,
            )

            # Clear the recent trades for this token after tripping
            self._recent_trades[token_id] = []

            return event

        return None

    def record_api_error(self, token_id: str) -> Optional[TokenTripEvent]:
        """Record an API error for a token. Trips after consecutive errors exceed threshold.

        Tracks API errors (e.g. from depth analysis, price fetching) per token
        within the detection window. After ``_max_consecutive_api_errors``
        errors in the window, the token is automatically tripped with reason
        ``"consecutive_api_errors"``.

        Args:
            token_id: The token that experienced the API error.

        Returns:
            A TokenTripEvent if the error threshold was reached, None otherwise.
        """
        now = utcnow()
        if token_id not in self._api_errors:
            self._api_errors[token_id] = []

        self._api_errors[token_id].append(now)

        # Keep only recent errors (within detection window)
        cutoff = now - timedelta(seconds=self.config.detection_window_seconds)
        self._api_errors[token_id] = [t for t in self._api_errors[token_id] if t > cutoff]

        if len(self._api_errors[token_id]) >= self._max_consecutive_api_errors:
            logger.warning(
                "Token tripped due to consecutive API errors",
                token_id=token_id,
                error_count=len(self._api_errors[token_id]),
                window_seconds=self.config.detection_window_seconds,
            )
            # Clear the error list after tripping
            error_count = len(self._api_errors[token_id])
            self._api_errors[token_id] = []
            return self.trip_token(
                token_id,
                "consecutive_api_errors",
                {"error_count": error_count},
            )
        return None

    def clear_api_errors(self, token_id: str):
        """Clear tracked API errors for a token (e.g. after a successful call)."""
        self._api_errors.pop(token_id, None)

    def is_tripped(self, token_id: str) -> tuple[bool, Optional[str]]:
        """
        Check if a token is currently tripped.
        Returns (is_tripped, reason).
        Auto-expires trips past their duration.
        """
        if token_id not in self._trips:
            return False, None

        event = self._trips[token_id]
        now = utcnow()

        if now >= event.expires_at:
            # Trip has expired, auto-clear it
            logger.info(
                "Token trip auto-expired",
                token_id=token_id,
                reason=event.reason,
                duration_seconds=self.config.trip_duration_seconds,
            )
            del self._trips[token_id]
            return False, None

        return True, event.reason

    def trip_token(
        self,
        token_id: str,
        reason: str,
        details: dict = None,
        trade_count: int = 0,
    ) -> TokenTripEvent:
        """Manually trip a token (e.g., on API error during depth check)."""
        now = utcnow()
        expires_at = now + timedelta(seconds=self.config.trip_duration_seconds)

        event = TokenTripEvent(
            token_id=token_id,
            reason=reason,
            triggered_at=now,
            expires_at=expires_at,
            trade_count=trade_count,
            details=details or {},
        )

        self._trips[token_id] = event
        self._trip_history.append(event)

        logger.info(
            "Token tripped",
            token_id=token_id,
            reason=reason,
            expires_at=expires_at.isoformat(),
            trade_count=trade_count,
        )

        return event

    def clear_trip(self, token_id: str):
        """Manually clear a trip."""
        if token_id in self._trips:
            event = self._trips[token_id]
            logger.info(
                "Token trip manually cleared",
                token_id=token_id,
                reason=event.reason,
            )
            del self._trips[token_id]
        else:
            logger.debug(
                "No active trip to clear for token",
                token_id=token_id,
            )

    def get_active_trips(self) -> list[TokenTripEvent]:
        """Get all currently active (non-expired) trips."""
        now = utcnow()
        active = []
        expired_tokens = []

        for token_id, event in self._trips.items():
            if now >= event.expires_at:
                expired_tokens.append(token_id)
            else:
                active.append(event)

        # Clean up expired trips
        for token_id in expired_tokens:
            logger.info(
                "Token trip auto-expired during active trips check",
                token_id=token_id,
            )
            del self._trips[token_id]

        return active

    def get_trip_stats(self) -> dict:
        """Get statistics on trips."""
        now = utcnow()
        active_trips = self.get_active_trips()

        # Count trips by reason across all history
        reason_counts: dict[str, int] = {}
        for event in self._trip_history:
            reason_counts[event.reason] = reason_counts.get(event.reason, 0) + 1

        # Count unique tokens that have been tripped
        unique_tokens_tripped = len(set(event.token_id for event in self._trip_history))

        return {
            "active_trips": len(active_trips),
            "total_trips_recorded": len(self._trip_history),
            "unique_tokens_tripped": unique_tokens_tripped,
            "trips_by_reason": reason_counts,
            "active_trip_details": [
                {
                    "token_id": t.token_id,
                    "reason": t.reason,
                    "triggered_at": t.triggered_at.isoformat(),
                    "expires_at": t.expires_at.isoformat(),
                    "remaining_seconds": max(0, (t.expires_at - now).total_seconds()),
                    "trade_count": t.trade_count,
                }
                for t in active_trips
            ],
            "config": {
                "large_trade_threshold_shares": self.config.large_trade_threshold_shares,
                "consecutive_trigger": self.config.consecutive_trigger,
                "detection_window_seconds": self.config.detection_window_seconds,
                "trip_duration_seconds": self.config.trip_duration_seconds,
                "trip_on_api_error": self.config.trip_on_api_error,
            },
        }

    async def save_trip_event(self, event: TokenTripEvent):
        """Persist trip event to database for historical analysis."""
        trip_record = TokenTrip(
            id=str(uuid.uuid4()),
            token_id=event.token_id,
            reason=event.reason,
            trade_count=event.trade_count,
            triggered_at=event.triggered_at,
            expires_at=event.expires_at,
            cleared_at=None,
            was_auto_expired=False,
            details=event.details,
        )

        try:
            async with AsyncSessionLocal() as session:
                session.add(trip_record)
                await session.commit()
                logger.info(
                    "Trip event saved to database",
                    token_id=event.token_id,
                    reason=event.reason,
                    trip_id=trip_record.id,
                )
        except Exception as e:
            logger.error(
                "Failed to save trip event to database",
                token_id=event.token_id,
                error=str(e),
            )


# ==================== Singleton ====================

token_circuit_breaker = TokenCircuitBreaker()
