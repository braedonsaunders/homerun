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
    result = safe_float(value, default=default)
    parsed = default if result is None else result
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


def to_float(value: Any, default: float = 0.0) -> float:
    """Parse any value to float, returning *default* on failure."""
    result = safe_float(value, default=default)
    return default if result is None else result


def to_bool(value: Any, default: bool = False) -> bool:
    """Parse any value to bool."""
    parsed = coerce_bool(value, default)
    return default if parsed is None else parsed


def coerce_bool(value: Any, default: bool | None = False) -> bool | None:
    """Parse any value to bool, optionally preserving ``None``."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return default
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    if text == "" and default is None:
        return None
    return default


def normalize_market_id(value: object) -> str:
    """Strip and lowercase a market identifier."""
    return str(value or "").strip().lower()


def normalize_identifier(value: object) -> str:
    """Strip and lowercase an identifier."""
    return normalize_market_id(value)


def parse_iso_datetime(value: Any, *, naive: bool = True) -> datetime | None:
    """Parse ISO datetime input, normalizing to UTC when an offset is present."""
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        if text.endswith("+00:00+00:00"):
            text = text[:-6]
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        return parsed if naive else parsed.replace(tzinfo=timezone.utc)
    normalized = parsed.astimezone(timezone.utc)
    return normalized.replace(tzinfo=None) if naive else normalized


def format_iso_utc_z(value: datetime | None) -> str | None:
    """Serialize a datetime to a UTC ISO string ending in ``Z``."""
    return to_iso(value)
