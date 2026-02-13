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
    threshold_c_low: Optional[float]
    threshold_c_high: Optional[float]
    raw_threshold: Optional[float]
    raw_threshold_low: Optional[float]
    raw_threshold_high: Optional[float]
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


def _detect_unit(text: str) -> Optional[str]:
    m = re.search(r"\b([FC])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    if "°f" in text.lower():
        return "F"
    if "°c" in text.lower():
        return "C"
    return None


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
    # Prefer "in <location> on <date>" form, then handle variants like
    # "in London be 13°C on Feb 11?" where location is followed by a verb.
    patterns = [
        r"\bin\s+([A-Za-z][A-Za-z0-9\s,.'\-/]+?)\s+on\b",
        r"\bin\s+([A-Za-z][A-Za-z0-9\s,.'\-/]+?)\s+(?:be|is|was|will(?:\s+be)?)\b",
        r"\bin\s+([A-Za-z][A-Za-z0-9\s,.'\-/]+?)(?:\?|$)",
    ]
    for pattern in patterns:
        m = re.search(pattern, question, flags=re.IGNORECASE)
        if not m:
            continue
        loc = " ".join(m.group(1).split()).strip(" ,.-")
        # Drop accidental trailing numeric fragments from malformed captures.
        loc = re.sub(r"\s+-?\d.*$", "", loc).strip(" ,.-")
        if loc:
            return loc
    return None


def _build_temp_threshold_contract(
    location: str,
    target_time: datetime,
    op_text: str,
    raw_value: float,
    unit: str,
    metric: str = "temp_threshold",
) -> ParsedWeatherContract:
    unit_up = unit.upper()
    return ParsedWeatherContract(
        location=location,
        target_time=target_time,
        metric=metric,
        operator=_normalize_operator(op_text),
        threshold_c=_to_celsius(raw_value, unit_up),
        threshold_c_low=None,
        threshold_c_high=None,
        raw_threshold=raw_value,
        raw_threshold_low=None,
        raw_threshold_high=None,
        raw_unit=unit_up,
    )


def _build_temp_range_contract(
    location: str,
    target_time: datetime,
    raw_low: float,
    raw_high: float,
    unit: str,
    metric: str = "temp_range",
) -> ParsedWeatherContract:
    unit_up = unit.upper()
    low = min(raw_low, raw_high)
    high = max(raw_low, raw_high)
    return ParsedWeatherContract(
        location=location,
        target_time=target_time,
        metric=metric,
        operator="between",
        threshold_c=None,
        threshold_c_low=_to_celsius(low, unit_up),
        threshold_c_high=_to_celsius(high, unit_up),
        raw_threshold=None,
        raw_threshold_low=low,
        raw_threshold_high=high,
        raw_unit=unit_up,
    )


def _parse_temperature_band(
    text: str, default_unit: Optional[str]
) -> tuple[str, float, Optional[float], str] | None:
    """Parse threshold/range bands from group labels or market text.

    Returns:
      ("threshold", value, None, unit) for one-sided thresholds
      ("range", low, high, unit) for bounded ranges
    """
    if not text:
        return None

    t = text.strip()
    t_norm = (
        t.replace("º", "°")
        .replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
        .replace(" to ", "-")
    )
    unit = _detect_unit(t_norm) or default_unit
    if unit is None:
        return None

    # e.g. "40-41°F", "40 - 41 F", "between 40 & 41 F"
    range_match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*(?:°?\s*[FC])?\s*(?:-|&)\s*(-?\d+(?:\.\d+)?)\s*(?:°?\s*([FC]))?",
        t_norm,
        flags=re.IGNORECASE,
    )
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        explicit = range_match.group(3).upper() if range_match.group(3) else None
        return ("range", low, high, explicit or unit)

    # e.g. "36°F or higher", "36F+", ">= 36F", "above 36F"
    high_match = re.search(
        r"(?:>=?|above|over|at least)?\s*(-?\d+(?:\.\d+)?)\s*(?:°?\s*([FC]))?\s*(?:\+|or higher|or above|and above)?",
        t_norm,
        flags=re.IGNORECASE,
    )
    if high_match and any(
        k in t_norm.lower() for k in [">", "above", "over", "at least", "higher", "+"]
    ):
        val = float(high_match.group(1))
        explicit = high_match.group(2).upper() if high_match.group(2) else None
        return ("threshold_gt", val, None, explicit or unit)

    # e.g. "32°C or below", "< 32C", "under 32C"
    low_match = re.search(
        r"(?:<=?|below|under|at most)?\s*(-?\d+(?:\.\d+)?)\s*(?:°?\s*([FC]))?\s*(?:or lower|or below|and below)?",
        t_norm,
        flags=re.IGNORECASE,
    )
    if low_match and any(
        k in t_norm.lower() for k in ["<", "below", "under", "at most", "lower"]
    ):
        val = float(low_match.group(1))
        explicit = low_match.group(2).upper() if low_match.group(2) else None
        return ("threshold_lt", val, None, explicit or unit)

    # e.g. "12°C" exact bucket => treat as a narrow band.
    exact_match = re.fullmatch(
        r"\s*(-?\d+(?:\.\d+)?)\s*(?:°?\s*([FC]))\s*",
        t_norm,
        flags=re.IGNORECASE,
    )
    if exact_match:
        center = float(exact_match.group(1))
        explicit = exact_match.group(2).upper()
        return ("range", center - 0.5, center + 0.5, explicit)

    return None


def _temperature_context(question: str) -> Optional[str]:
    q = question.lower()
    if re.search(
        r"\b(?:highest|max(?:imum)?|high(?:est)?\s+temperature|the\s+high\b|high\s+in)\b",
        q,
    ):
        return "max"
    if re.search(
        r"\b(?:lowest|min(?:imum)?|low(?:est)?\s+temperature|the\s+low\b|low\s+in)\b",
        q,
    ):
        return "min"
    return None


def _metric_for_context(base: str, temp_context: Optional[str]) -> str:
    if temp_context == "max":
        return f"temp_max_{base}"
    if temp_context == "min":
        return f"temp_min_{base}"
    return f"temp_{base}"


def parse_weather_contract(
    question: str,
    resolution_date: Optional[datetime] = None,
    group_item_title: Optional[str] = None,
) -> Optional[ParsedWeatherContract]:
    """Parse a weather market question into forecastable contract details.

    Supported forms:
    - temperature thresholds (above/below X F/C)
    - grouped temperature ranges (e.g. "Highest temperature ...", bucket in group item)
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

    default_unit = _detect_unit(q) or _detect_unit(group_item_title or "") or "F"
    temp_context = _temperature_context(q)

    # Temperature contract with explicit threshold in question.
    temp_match = re.search(
        r"(?:temperature|high|low).*?(above|over|below|under|at least|at most)\s*(-?\d+(?:\.\d+)?)\s*°?\s*([FC])",
        q,
        flags=re.IGNORECASE,
    )
    if temp_match:
        op_text, raw_val, unit = temp_match.groups()
        return _build_temp_threshold_contract(
            location=location,
            target_time=target_time,
            op_text=op_text,
            raw_value=float(raw_val),
            unit=unit,
            metric=_metric_for_context("threshold", temp_context),
        )

    # Postfix threshold contracts:
    # "Will the highest temperature in Dallas be 68°F or higher on February 13?"
    postfix_threshold_match = re.search(
        r"\b(?:be|is|was|will(?:\s+be)?|reach(?:es)?|hit(?:s)?)\s*(-?\d+(?:\.\d+)?)\s*°?\s*([FC])\s*(?:\+|(?:or|and)\s+(?:higher|above|lower|below))\b",
        q,
        flags=re.IGNORECASE,
    )
    if postfix_threshold_match and temp_context is not None:
        raw_val = float(postfix_threshold_match.group(1))
        unit = postfix_threshold_match.group(2).upper()
        suffix = postfix_threshold_match.group(0).lower()
        op_text = (
            "above"
            if ("+" in suffix or "higher" in suffix or "above" in suffix)
            else "below"
        )
        return _build_temp_threshold_contract(
            location=location,
            target_time=target_time,
            op_text=op_text,
            raw_value=raw_val,
            unit=unit,
            metric=_metric_for_context("threshold", temp_context),
        )

    # Exact value contracts (common in daily high/low markets):
    # "Will the highest temperature in London be 13°C on February 11?"
    exact_match = re.search(
        r"\b(?:be|is|equals?|reach(?:es)?|hit(?:s)?)\s*(-?\d+(?:\.\d+)?)\s*°?\s*([FC])\b",
        q,
        flags=re.IGNORECASE,
    )
    if exact_match and temp_context is not None:
        center = float(exact_match.group(1))
        unit = exact_match.group(2).upper()
        return _build_temp_range_contract(
            location=location,
            target_time=target_time,
            raw_low=center - 0.5,
            raw_high=center + 0.5,
            unit=unit,
            metric=_metric_for_context("range", temp_context),
        )

    # Generic weather page style contracts use the main question for context
    # and group outcome label for the actual temperature bucket.
    # Example: "Highest temperature in New York City on February 11?" + "40-41°F"
    if re.search(r"\b(?:highest|lowest|max(?:imum)?|min(?:imum)?)\s+temperature\b", q_lower):
        band = _parse_temperature_band(group_item_title or "", default_unit=default_unit)
        if band is not None:
            kind, left, right, unit = band
            if kind == "range" and right is not None:
                return _build_temp_range_contract(
                    location=location,
                    target_time=target_time,
                    raw_low=left,
                    raw_high=right,
                    unit=unit,
                    metric=_metric_for_context("range", temp_context),
                )
            if kind == "threshold_gt":
                return _build_temp_threshold_contract(
                    location=location,
                    target_time=target_time,
                    op_text="above",
                    raw_value=left,
                    unit=unit,
                    metric=_metric_for_context("threshold", temp_context),
                )
            if kind == "threshold_lt":
                return _build_temp_threshold_contract(
                    location=location,
                    target_time=target_time,
                    op_text="below",
                    raw_value=left,
                    unit=unit,
                    metric=_metric_for_context("threshold", temp_context),
                )

    # Rain/snow/precip occurrence contract.
    rain_or_snow_occurrence = re.search(
        r"\b(?:will it rain|will it snow|rain in .* on|snow in .* on|rain on|snow on)\b",
        q_lower,
        flags=re.IGNORECASE,
    )
    precip_occurrence = re.search(
        r"\b(?:will there be precipitation|precipitation expected|precip expected)\b",
        q_lower,
        flags=re.IGNORECASE,
    )
    if rain_or_snow_occurrence or precip_occurrence:
        # Quantitative precipitation amount contracts are not supported by the
        # current adapter (which models occurrence probability only).
        if re.search(
            r"\b(?:inch|inches|mm|cm|between|more than|less than|at least|at most|total precipitation)\b",
            q_lower,
            flags=re.IGNORECASE,
        ):
            return None

        operator = "gt"
        if re.search(
            r"\b(?:no rain|without rain|won['’]t rain|will not rain|no snow|without snow)\b",
            q_lower,
            flags=re.IGNORECASE,
        ):
            operator = "lt"
        return ParsedWeatherContract(
            location=location,
            target_time=target_time,
            metric="precip_occurrence",
            operator=operator,
            threshold_c=None,
            threshold_c_low=None,
            threshold_c_high=None,
            raw_threshold=None,
            raw_threshold_low=None,
            raw_threshold_high=None,
            raw_unit=None,
        )

    return None
