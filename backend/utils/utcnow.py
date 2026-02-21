"""UTC datetime helpers used across the runtime."""

from datetime import datetime, timezone
from typing import Optional


def utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


def utcfromtimestamp(ts: float) -> datetime:
    """Convert a POSIX timestamp to a timezone-aware UTC datetime."""
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def as_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def as_utc_naive(dt: Optional[datetime]) -> Optional[datetime]:
    value = as_utc(dt)
    if value is None:
        return None
    return value.replace(tzinfo=None)
