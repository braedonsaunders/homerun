"""UTC datetime helpers used across the runtime."""

from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Optional


# ── Replay clock ─────────────────────────────────────────────────────
# During backtest replay the strategy/orchestrator must read SIMULATED
# event time, not the wall clock — otherwise time-derived decisions
# (time-to-resolution, staleness gates, schedule windows, cooldowns)
# diverge between live and replay and the backtest stops being a pure
# function of recorded inputs.  The backtester sets this around each
# detect call to the tick's as-of time; live trading leaves it unset and
# utcnow() returns the real wall clock.
#
# A ContextVar (not a module global) so concurrent replays on the same
# loop don't clobber each other's simulated time, and so the value
# propagates into child tasks of the detect coroutine automatically.
_replay_clock_us: ContextVar[Optional[int]] = ContextVar("_replay_clock_us", default=None)


def set_replay_clock_us(now_us: Optional[int]):
    """Set the simulated replay clock (UTC microseconds).  Returns a
    token for ``restore_replay_clock``.  Pass None to clear."""
    return _replay_clock_us.set(now_us)


def restore_replay_clock(token) -> None:
    try:
        _replay_clock_us.reset(token)
    except (ValueError, LookupError):
        pass


def replay_clock_us() -> Optional[int]:
    """Current simulated replay clock in UTC microseconds, or None when
    running live (wall clock)."""
    return _replay_clock_us.get()


def utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime.

    Under a backtest replay clock this returns the SIMULATED event time
    so decisions are reproducible; otherwise the real wall clock.
    """
    us = _replay_clock_us.get()
    if us is not None:
        return datetime.fromtimestamp(us / 1_000_000, tz=timezone.utc)
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
