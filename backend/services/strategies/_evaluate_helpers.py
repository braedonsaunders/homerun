"""Shared helpers for strategy evaluate() and should_exit() methods.

These utilities are used across all unified strategies to parse signal
data, normalize values, and compute position sizes.

Generic converters (to_float, clamp, to_confidence) are delegated to
:mod:`utils.converters` and re-exported here for backward compatibility.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from utils.converters import (
    clamp,  # noqa: F401 – re-exported
    safe_float,
    to_confidence,  # noqa: F401 – re-exported
)


def to_float(value: Any, default: float = 0.0) -> float:
    """Parse any value to float, returning *default* on failure."""
    result = safe_float(value, default=default)
    return default if result is None else result


def to_bool(value: Any, default: bool = False) -> bool:
    """Parse any value to bool."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def signal_payload(signal: Any) -> dict[str, Any]:
    """Extract payload_json dict from a signal object."""
    payload = getattr(signal, "payload_json", None)
    return payload if isinstance(payload, dict) else {}


def parse_iso(value: Any) -> datetime | None:
    """Parse an ISO datetime string to timezone-aware datetime."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return None


def days_to_resolution(payload: dict[str, Any]) -> float | None:
    """Compute days until resolution from payload resolution_date."""
    resolution = parse_iso(payload.get("resolution_date"))
    if resolution is None:
        return None
    now = datetime.now(timezone.utc)
    return (resolution - now).total_seconds() / 86400.0


def selected_probability(
    signal: Any, payload: dict[str, Any], direction: str,
) -> float | None:
    """Extract the best probability estimate for the selected side."""
    entry = to_float(getattr(signal, "entry_price", None), -1.0)
    if 0.0 < entry < 1.0:
        return entry

    positions = payload.get("positions_to_take") if isinstance(
        payload.get("positions_to_take"), list,
    ) else []
    if positions:
        candidate = to_float((positions[0] or {}).get("price"), -1.0)
        if 0.0 < candidate < 1.0:
            return candidate

    model_prob = to_float(payload.get("model_probability"), -1.0)
    if 0.0 <= model_prob <= 1.0:
        return model_prob

    edge = to_float(getattr(signal, "edge_percent", 0.0), 0.0)
    if 0.0 < entry < 1.0:
        implied = entry + (edge / 100.0)
        if 0.0 <= implied <= 1.0:
            return implied

    return None


def live_move(context: dict[str, Any], key: str) -> float | None:
    """Extract a price move percentage from live market context."""
    summary = context.get("history_summary")
    if not isinstance(summary, dict):
        return None
    move = summary.get(key)
    if not isinstance(move, dict):
        return None
    raw = move.get("percent")
    if raw is None:
        return None
    return to_float(raw, 0.0)


def weather_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract nested weather metadata from signal payload."""
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    market = metadata.get("market")
    if not isinstance(market, dict):
        return {}
    weather = market.get("weather")
    if not isinstance(weather, dict):
        return {}
    return weather


def hours_to_target(target_time: Any) -> float | None:
    """Compute hours until a target time."""
    text = str(target_time or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        delta = parsed.astimezone(timezone.utc) - datetime.now(timezone.utc)
        return delta.total_seconds() / 3600.0
    except Exception:
        return None
