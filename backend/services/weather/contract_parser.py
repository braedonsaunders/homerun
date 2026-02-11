from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass
class ParsedWeatherContract:
    location: str
    target_time: datetime
    metric: str
    operator: str
    threshold_c: Optional[float]
    raw_threshold: Optional[float]
    raw_unit: Optional[str]


def _to_celsius(value: float, unit: str) -> float:
    unit_up = (unit or "").upper()
    if unit_up == "F":
        return (value - 32.0) * (5.0 / 9.0)
    return value


def _normalize_operator(text: str) -> str:
    t = text.lower().strip()
    if t in {"above", "over", ">", ">=", "at least"}:
        return "gt"
    if t in {"below", "under", "<", "<=", "at most"}:
        return "lt"
    return "gt"


def _parse_date_from_question(question: str) -> Optional[datetime]:
    # Handles "on Jan 29" and "on January 29, 2026".
    m = re.search(
        r"\bon\s+([A-Za-z]{3,9}\s+\d{1,2}(?:,\s*\d{4})?)",
        question,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    raw = m.group(1).strip()
    now = datetime.now(timezone.utc)

    fmts = ["%B %d, %Y", "%b %d, %Y", "%B %d", "%b %d"]
    for fmt in fmts:
        try:
            parsed = datetime.strptime(raw, fmt)
            if "%Y" not in fmt:
                parsed = parsed.replace(year=now.year)
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _parse_location(question: str) -> Optional[str]:
    # Capture text after "in" and before "on|by|before|after|?".
    m = re.search(
        r"\bin\s+([A-Za-z0-9\s,.'\-/]+?)(?:\s+on\b|\s+by\b|\s+before\b|\s+after\b|\?|$)",
        question,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    loc = " ".join(m.group(1).split()).strip(" ,.-")
    return loc or None


def parse_weather_contract(
    question: str, resolution_date: Optional[datetime] = None
) -> Optional[ParsedWeatherContract]:
    """Parse a weather market question into forecastable contract details.

    Supported forms:
    - temperature thresholds (above/below X F/C)
    - rain/snow/precipitation occurrence
    """
    if not question:
        return None

    q = question.strip()
    q_lower = q.lower()

    location = _parse_location(q) or "New York, NY"
    target_time = _parse_date_from_question(q)
    if target_time is None:
        if resolution_date is not None:
            target_time = (
                resolution_date
                if resolution_date.tzinfo is not None
                else resolution_date.replace(tzinfo=timezone.utc)
            )
        else:
            target_time = datetime.now(timezone.utc)

    # Temperature contract.
    temp_match = re.search(
        r"(?:temperature|high|low)[^\d]*(above|over|below|under|at least|at most)\s*(-?\d+(?:\.\d+)?)\s*°?\s*([FC])",
        q,
        flags=re.IGNORECASE,
    )
    if temp_match:
        op_text, raw_val, unit = temp_match.groups()
        raw_threshold = float(raw_val)
        return ParsedWeatherContract(
            location=location,
            target_time=target_time,
            metric="temp_threshold",
            operator=_normalize_operator(op_text),
            threshold_c=_to_celsius(raw_threshold, unit),
            raw_threshold=raw_threshold,
            raw_unit=unit.upper(),
        )

    # Rain/snow/precip occurrence contract.
    if any(k in q_lower for k in ["rain", "snow", "precip", "precipitation"]):
        operator = "gt"
        if any(k in q_lower for k in ["no rain", "without rain", "won't rain", "will not rain"]):
            operator = "lt"
        return ParsedWeatherContract(
            location=location,
            target_time=target_time,
            metric="precip_occurrence",
            operator=operator,
            threshold_c=None,
            raw_threshold=None,
            raw_unit=None,
        )

    return None
