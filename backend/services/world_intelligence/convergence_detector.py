"""Multi-signal geographic convergence detection.

Detects when 2+ different event types cluster in the same geographic
area within a 24-hour rolling window.  High-urgency convergence zones
(e.g., military flights + conflict + internet outage in the same 1-degree
grid cell) are strong predictors of imminent geopolitical events that
move prediction markets.
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from config import settings
from .taxonomy_catalog import taxonomy_catalog

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Grid resolution in degrees.  1 degree ~ 111 km at equator.
DEFAULT_GRID_RESOLUTION = 1.0

# Signal types tracked by the convergence detector
SIGNAL_TYPES = taxonomy_catalog.convergence_signal_types()

# Rolling window for signal relevance
_SIGNAL_TTL_HOURS = 24

# Haversine proximity thresholds (km) by zone type
_PROXIMITY_THRESHOLDS = {
    "conflict": 300,
    "waterway": 200,
    "city": 150,
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ConvergenceZone:
    """A geographic area where multiple signal types converge."""

    grid_key: str  # e.g., "48.0_35.0"
    latitude: float
    longitude: float
    signal_types: set[str]
    signal_count: int
    urgency_score: float  # 0-100
    country: str
    nearby_markets: list[str] = field(default_factory=list)
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _Signal:
    """Internal: a single ingested signal with location and timestamp."""

    signal_type: str
    latitude: float
    longitude: float
    timestamp: float  # monotonic seconds
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometres."""
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _grid_key(lat: float, lon: float, resolution: float) -> str:
    """Quantize lat/lon to a grid cell key."""
    glat = math.floor(lat / resolution) * resolution
    glon = math.floor(lon / resolution) * resolution
    return f"{glat:.1f}_{glon:.1f}"


def _grid_center(key: str, resolution: float) -> tuple[float, float]:
    """Return the center lat/lon of a grid cell."""
    parts = key.split("_")
    lat = float(parts[0]) + resolution / 2
    lon = float(parts[1]) + resolution / 2
    return lat, lon


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class ConvergenceDetector:
    """Detects geographic convergence of heterogeneous signals.

    Signals are bucketed into 1-degree lat/lon grid cells.  When a cell
    accumulates multiple distinct signal types within the 24-hour rolling
    window, it is flagged as a convergence zone.
    """

    def __init__(self, grid_resolution: float = DEFAULT_GRID_RESOLUTION) -> None:
        self._resolution = grid_resolution
        # grid_key -> list of _Signal
        self._grid: dict[str, list[_Signal]] = defaultdict(list)
        # Cache of detected convergences
        self._active_convergences: list[ConvergenceZone] = []

    # -- Ingestion -----------------------------------------------------------

    def ingest_signal(
        self,
        signal_type: str,
        latitude: float,
        longitude: float,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Add a signal observation to the grid.

        Args:
            signal_type: One of the SIGNAL_TYPES constants.
            latitude: WGS84 latitude.
            longitude: WGS84 longitude.
            metadata: Optional dict with extra event data (severity, country, etc.).
        """
        if signal_type not in SIGNAL_TYPES:
            logger.debug("Unknown signal type '%s', accepting anyway", signal_type)

        key = _grid_key(latitude, longitude, self._resolution)
        signal = _Signal(
            signal_type=signal_type,
            latitude=latitude,
            longitude=longitude,
            timestamp=time.monotonic(),
            metadata=metadata or {},
        )
        self._grid[key].append(signal)

    # -- Expiration ----------------------------------------------------------

    def clear_expired(self) -> int:
        """Remove signals older than 24 hours.  Returns count of pruned signals."""
        cutoff = time.monotonic() - (_SIGNAL_TTL_HOURS * 3600)
        pruned = 0
        empty_keys: list[str] = []

        for key, signals in self._grid.items():
            before = len(signals)
            self._grid[key] = [s for s in signals if s.timestamp >= cutoff]
            pruned += before - len(self._grid[key])
            if not self._grid[key]:
                empty_keys.append(key)

        for key in empty_keys:
            del self._grid[key]

        if pruned > 0:
            logger.debug("Convergence detector pruned %d expired signals", pruned)
        return pruned

    # -- Detection -----------------------------------------------------------

    def _compute_urgency(self, signals: list[_Signal]) -> float:
        """Compute urgency score (0-100) for a set of co-located signals.

        Components:
        - type_bonus:     15 points per unique signal type
        - count_bonus:    5 points per signal, capped at 30
        - severity_bonus: max severity from metadata, scaled 0-20
        """
        unique_types = {s.signal_type for s in signals}
        type_bonus = len(unique_types) * 15.0
        count_bonus = min(len(signals) * 5.0, 30.0)

        max_severity = 0.0
        for s in signals:
            sev = s.metadata.get("severity", 0.0)
            if isinstance(sev, (int, float)):
                max_severity = max(max_severity, float(sev))
        severity_bonus = max_severity * 20.0

        return min(100.0, type_bonus + count_bonus + severity_bonus)

    def _infer_country(self, signals: list[_Signal]) -> str:
        """Best-effort country extraction from signal metadata."""
        for s in signals:
            country = s.metadata.get("country", "")
            if country:
                return str(country)
        return "unknown"

    def detect_convergences(self, min_types: int = 2) -> list[ConvergenceZone]:
        """Scan all grid cells and return those with >= min_types distinct signals.

        Also prunes expired signals before scanning.
        """
        self.clear_expired()

        convergences: list[ConvergenceZone] = []

        for key, signals in self._grid.items():
            unique_types = {s.signal_type for s in signals}
            if len(unique_types) < min_types:
                continue

            center_lat, center_lon = _grid_center(key, self._resolution)
            urgency = self._compute_urgency(signals)
            country = self._infer_country(signals)

            # Build contributing event summaries
            events: list[dict[str, Any]] = []
            for s in signals:
                events.append({
                    "type": s.signal_type,
                    "lat": s.latitude,
                    "lon": s.longitude,
                    "metadata": s.metadata,
                })

            zone = ConvergenceZone(
                grid_key=key,
                latitude=center_lat,
                longitude=center_lon,
                signal_types=unique_types,
                signal_count=len(signals),
                urgency_score=round(urgency, 1),
                country=country,
                events=events,
            )
            convergences.append(zone)

        # Sort by urgency descending
        convergences.sort(key=lambda z: z.urgency_score, reverse=True)
        self._active_convergences = convergences

        if convergences:
            logger.info(
                "Convergence detector found %d zones (top urgency=%.1f)",
                len(convergences),
                convergences[0].urgency_score,
            )
        return convergences

    # -- Market matching -----------------------------------------------------

    async def match_to_markets(
        self,
        convergence: ConvergenceZone,
        markets: list[Any],
    ) -> list[str]:
        """Find prediction markets related to a convergence zone.

        Matches by:
        1. Country name / ISO code in market question or description
        2. Region keywords
        3. Geographic proximity (haversine) for markets with coordinates

        Args:
            convergence: The convergence zone to match.
            markets: List of market objects (must have .question or .title attribute).

        Returns:
            List of matched market IDs.
        """
        matched_ids: list[str] = []
        country = convergence.country.lower()

        for market in markets:
            market_id = getattr(market, "market_id", getattr(market, "id", ""))
            question = str(
                getattr(market, "question", getattr(market, "title", ""))
            ).lower()

            # Country keyword match
            if country and country != "unknown" and country in question:
                matched_ids.append(str(market_id))
                continue

            # Check signal-type keywords against market text
            keyword_map = taxonomy_catalog.convergence_market_keyword_map()
            for sig_type in convergence.signal_types:
                for kw in keyword_map.get(sig_type, []):
                    if kw in question:
                        matched_ids.append(str(market_id))
                        break

            # Proximity check if market has coordinates
            market_lat = getattr(market, "latitude", None)
            market_lon = getattr(market, "longitude", None)
            if market_lat is not None and market_lon is not None:
                dist = _haversine_km(
                    convergence.latitude, convergence.longitude,
                    float(market_lat), float(market_lon),
                )
                threshold = _PROXIMITY_THRESHOLDS.get("conflict", 300)
                if dist <= threshold:
                    matched_ids.append(str(market_id))

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for mid in matched_ids:
            if mid not in seen:
                seen.add(mid)
                unique.append(mid)
        return unique

    def export_state(self) -> dict[str, Any]:
        now = time.monotonic()
        rows: list[dict[str, Any]] = []
        for grid_key, signals in self._grid.items():
            for sig in signals[-500:]:
                age_seconds = max(0.0, now - sig.timestamp)
                rows.append(
                    {
                        "grid_key": grid_key,
                        "signal_type": sig.signal_type,
                        "latitude": sig.latitude,
                        "longitude": sig.longitude,
                        "age_seconds": round(age_seconds, 3),
                        "metadata": sig.metadata,
                    }
                )
        return {"signals": rows[-5000:]}

    def import_state(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        rows = payload.get("signals") or []
        if not isinstance(rows, list):
            return
        now = time.monotonic()
        cutoff_age = _SIGNAL_TTL_HOURS * 3600
        grid: dict[str, list[_Signal]] = defaultdict(list)
        for item in rows:
            if not isinstance(item, dict):
                continue
            grid_key = str(item.get("grid_key") or "").strip()
            signal_type = str(item.get("signal_type") or "").strip()
            if not grid_key or not signal_type:
                continue
            try:
                lat = float(item.get("latitude"))
                lon = float(item.get("longitude"))
                age = float(item.get("age_seconds") or 0.0)
            except Exception:
                continue
            if age > cutoff_age:
                continue
            metadata = item.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            grid[grid_key].append(
                _Signal(
                    signal_type=signal_type,
                    latitude=lat,
                    longitude=lon,
                    timestamp=now - max(0.0, age),
                    metadata=metadata,
                )
            )
        if grid:
            self._grid = grid
            self.detect_convergences(
                min_types=int(max(2, getattr(settings, "WORLD_INTEL_CONVERGENCE_MIN_TYPES", 2) or 2))
            )

    # -- Accessors -----------------------------------------------------------

    def get_active_convergences(self) -> list[ConvergenceZone]:
        """Return the most recently detected convergence zones."""
        return list(self._active_convergences)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

convergence_detector = ConvergenceDetector()
