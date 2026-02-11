"""Cross-version UTC helpers (Python 3.9 – 3.12+).

``datetime.utcnow()`` and ``datetime.utcfromtimestamp()`` are deprecated
since Python 3.12 and scheduled for removal.  These thin wrappers produce
the same **naive** UTC datetimes the rest of the codebase expects, without
triggering DeprecationWarning on 3.12+.
"""

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Return the current UTC time as a naive (tzinfo=None) datetime."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def utcfromtimestamp(ts: float) -> datetime:
    """Convert a POSIX timestamp to a naive UTC datetime."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
