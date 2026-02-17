"""Shared value-conversion utilities used across the backend.

Consolidates duplicated _safe_float, _safe_int, _clamp, _to_confidence,
_to_iso, and _normalize_market_id helpers that previously appeared in
dozens of individual modules.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any


def safe_float(
    value: Any,
    default: float | None = None,
    *,
    reject_nan_inf: bool = False,
) -> float | None:
    """Parse *value* to float, returning *default* on failure.

    When *reject_nan_inf* is ``True``, ``NaN`` / ``±Inf`` results are
    treated the same as parse failures and *default* is returned.
    """
    try:
        parsed = float(value)
    except Exception:
        return default
    if reject_nan_inf and not math.isfinite(parsed):
        return default
    return parsed


def safe_int(value: Any, default: int = 0) -> int:
    """Parse *value* to int, returning *default* on failure."""
    try:
        return int(value)
    except Exception:
        return default


def clamp(value: float, low: float, high: float) -> float:
    """Clamp *value* to the closed interval [*low*, *high*]."""
    return max(low, min(high, value))


def to_confidence(value: Any, default: float = 0.0) -> float:
    """Normalize a confidence value to [0, 1].

    Values > 1.0 are treated as percentages and divided by 100.
    """
    parsed = safe_float(value, default=default) or default
    if parsed > 1.0:
        parsed = parsed / 100.0
    return clamp(parsed, 0.0, 1.0)


def to_iso(value: datetime | None) -> str | None:
    """Serialize a datetime to an ISO 8601 UTC string ending in ``Z``.

    Naive datetimes are assumed to be UTC.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.replace(tzinfo=None).isoformat() + "Z"


def normalize_market_id(value: object) -> str:
    """Strip and lowercase a market identifier."""
    return str(value or "").strip().lower()
