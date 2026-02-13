"""USGS Earthquake Hazards Program client.

Fetches real-time earthquake data from the USGS GeoJSON API.
Relevant for natural disaster markets on Kalshi and Polymarket.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import httpx

from config import settings

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 15
_USER_AGENT = "Mozilla/5.0 (compatible; Homerun/2.0)"

# USGS GeoJSON feed endpoints (real-time, updated every minute)
USGS_FEEDS = {
    "significant_month": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_month.geojson",
    "m4.5_day": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_day.geojson",
    "m2.5_day": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson",
    "all_hour": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson",
}


@dataclass
class Earthquake:
    """A single earthquake event from USGS."""

    event_id: str
    magnitude: float
    place: str  # Human-readable location
    latitude: float
    longitude: float
    depth_km: float
    timestamp: datetime
    url: str  # USGS event detail page
    felt: Optional[int] = None  # Number of felt reports
    tsunami: bool = False  # Tsunami warning issued
    alert: Optional[str] = None  # green, yellow, orange, red
    significance: int = 0  # 0-1000 composite significance
    mmi: Optional[float] = None  # Modified Mercalli Intensity
    country: str = ""  # Extracted country from place string
    severity_score: float = 0.0  # 0-1 computed severity


@dataclass
class EarthquakeSummary:
    """Summary of recent seismic activity."""

    total_events: int = 0
    significant_events: int = 0
    max_magnitude: float = 0.0
    tsunami_warnings: int = 0
    by_region: dict = field(default_factory=dict)
    last_updated: Optional[datetime] = None


class USGSClient:
    """Client for USGS Earthquake GeoJSON feeds."""

    def __init__(self) -> None:
        self._earthquakes: dict[str, Earthquake] = {}
        self._last_fetch_at: Optional[datetime] = None
        self._consecutive_failures: int = 0
        self._cooldown_until: Optional[datetime] = None
        self._last_error: Optional[str] = None

    async def fetch_earthquakes(
        self,
        feed: str = "m4.5_day",
        min_magnitude: float = 4.0,
    ) -> list[Earthquake]:
        """Fetch earthquakes from USGS. Returns NEW events only."""
        if not settings.WORLD_INTEL_USGS_ENABLED:
            self._last_error = "disabled"
            return []

        if self._cooldown_until and datetime.now(timezone.utc) < self._cooldown_until:
            self._last_error = "cooldown_active"
            return []

        url = USGS_FEEDS.get(feed)
        if not url:
            logger.warning("Unknown USGS feed: %s", feed)
            self._last_error = f"unknown_feed:{feed}"
            return []

        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.get(url, headers={"User-Agent": _USER_AGENT})
                if resp.status_code != 200:
                    self._record_failure()
                    self._last_error = f"http_{resp.status_code}"
                    return []

            data = resp.json()
            features = data.get("features", [])
            new_quakes: list[Earthquake] = []

            for feature in features:
                props = feature.get("properties", {})
                geom = feature.get("geometry", {})
                coords = geom.get("coordinates", [0, 0, 0])

                mag = float(props.get("mag", 0) or 0)
                if mag < min_magnitude:
                    continue

                event_id = feature.get("id", "")
                if event_id in self._earthquakes:
                    continue

                quake = Earthquake(
                    event_id=event_id,
                    magnitude=mag,
                    place=props.get("place", "Unknown"),
                    latitude=float(coords[1]) if len(coords) > 1 else 0.0,
                    longitude=float(coords[0]) if len(coords) > 0 else 0.0,
                    depth_km=float(coords[2]) if len(coords) > 2 else 0.0,
                    timestamp=datetime.fromtimestamp(
                        (props.get("time", 0) or 0) / 1000, tz=timezone.utc
                    ),
                    url=props.get("url", ""),
                    felt=props.get("felt"),
                    tsunami=bool(props.get("tsunami", 0)),
                    alert=props.get("alert"),
                    significance=int(props.get("sig", 0) or 0),
                    mmi=props.get("mmi"),
                    country=_extract_country(props.get("place", "")),
                    severity_score=_compute_severity(mag, props),
                )

                self._earthquakes[event_id] = quake
                new_quakes.append(quake)

            self._consecutive_failures = 0
            self._last_error = None
            self._last_fetch_at = datetime.now(timezone.utc)
            self._prune_old()

            if new_quakes:
                logger.info(
                    "USGS: %d new earthquakes (max M%.1f)",
                    len(new_quakes),
                    max(q.magnitude for q in new_quakes),
                )
            return new_quakes

        except Exception as e:
            self._record_failure()
            self._last_error = str(e)
            logger.debug("USGS fetch failed: %s", e)
            return []

    def get_recent(self, hours: int = 24) -> list[Earthquake]:
        """Get recent earthquakes from cache."""
        cutoff = datetime.now(timezone.utc).timestamp() - (hours * 3600)
        quakes = [
            q for q in self._earthquakes.values()
            if q.timestamp.timestamp() > cutoff
        ]
        quakes.sort(key=lambda q: q.magnitude, reverse=True)
        return quakes

    def get_significant(self, min_magnitude: float = 5.5) -> list[Earthquake]:
        """Get significant recent earthquakes."""
        return [q for q in self.get_recent(48) if q.magnitude >= min_magnitude]

    def get_summary(self) -> EarthquakeSummary:
        """Get a summary of recent activity."""
        quakes = self.get_recent(24)
        by_region: dict[str, int] = {}
        for q in quakes:
            region = q.country or "Unknown"
            by_region[region] = by_region.get(region, 0) + 1

        return EarthquakeSummary(
            total_events=len(quakes),
            significant_events=len([q for q in quakes if q.magnitude >= 5.5]),
            max_magnitude=max((q.magnitude for q in quakes), default=0.0),
            tsunami_warnings=len([q for q in quakes if q.tsunami]),
            by_region=by_region,
            last_updated=self._last_fetch_at,
        )

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= 3:
            from datetime import timedelta
            self._cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=5)
            logger.warning("USGS client in cooldown after %d failures", self._consecutive_failures)

    def get_health(self) -> dict[str, object]:
        return {
            "enabled": bool(settings.WORLD_INTEL_USGS_ENABLED),
            "consecutive_failures": self._consecutive_failures,
            "cooldown_until": self._cooldown_until.isoformat() if self._cooldown_until else None,
            "cached_quakes": len(self._earthquakes),
            "last_fetch_at": self._last_fetch_at.isoformat() if self._last_fetch_at else None,
            "last_error": self._last_error,
        }

    def _prune_old(self) -> None:
        cutoff = datetime.now(timezone.utc).timestamp() - (7 * 24 * 3600)  # 7 days
        to_remove = [
            eid for eid, q in self._earthquakes.items()
            if q.timestamp.timestamp() < cutoff
        ]
        for eid in to_remove:
            del self._earthquakes[eid]


def _extract_country(place: str) -> str:
    """Extract country from USGS place string like '10km SE of Tokyo, Japan'."""
    if not place:
        return ""
    parts = place.rsplit(",", 1)
    if len(parts) == 2:
        return parts[1].strip()
    return ""


def _compute_severity(magnitude: float, props: dict) -> float:
    """Compute severity score 0-1 from earthquake properties."""
    # Base severity from magnitude (exponential scale)
    if magnitude >= 8.0:
        severity = 1.0
    elif magnitude >= 7.0:
        severity = 0.85
    elif magnitude >= 6.0:
        severity = 0.65
    elif magnitude >= 5.5:
        severity = 0.45
    elif magnitude >= 5.0:
        severity = 0.3
    elif magnitude >= 4.5:
        severity = 0.15
    else:
        severity = 0.05

    # Boost for tsunami warning
    if props.get("tsunami"):
        severity = min(1.0, severity + 0.2)

    # Boost for PAGER alert level
    alert = props.get("alert")
    if alert == "red":
        severity = min(1.0, severity + 0.3)
    elif alert == "orange":
        severity = min(1.0, severity + 0.2)
    elif alert == "yellow":
        severity = min(1.0, severity + 0.1)

    return min(1.0, severity)


# Singleton
usgs_client = USGSClient()
