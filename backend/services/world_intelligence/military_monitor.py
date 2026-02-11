"""Military activity monitor via OpenSky Network and vessel tracking.

Monitors global airspace for military flights using the OpenSky Network
REST API (free tier, no auth required for basic state vectors).  Filters
results to likely military aircraft via callsign prefix matching and
ICAO24 address ranges.

Vessel monitoring (AIS-based) is stubbed as a placeholder; the AISStream
API requires a paid subscription.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import httpx

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENSKY_API_URL = "https://opensky-network.org/api/states/all"

# Rate limiting: OpenSky free tier is aggressive (max ~10 req/min for anon)
_RATE_LIMIT_MAX_REQUESTS = 10
_RATE_LIMIT_WINDOW_SECONDS = 60.0

# Circuit breaker
_CB_MAX_FAILURES = 3
_CB_COOLDOWN_SECONDS = 300.0  # 5 minutes

# Known military callsign prefixes (partial list; covers major NATO & others)
MILITARY_CALLSIGN_PREFIXES: list[str] = [
    "FORTE",   # US Global Hawk RQ-4 (ISR over Black Sea)
    "RCH",     # US Air Mobility Command (C-17, C-5)
    "JAKE",    # US KC-135 tankers
    "HOMER",   # US KC-135 / KC-46 tankers
    "DUKE",    # US special ops
    "VIPER",   # US F-16 flights
    "REACH",   # US AMC transports
    "VALOR",   # US Army rotary
    "NOBLE",   # NATO missions
    "NATO",    # NATO AWACS / JSTARS
    "LAGR",    # US SIGINT
    "EVIL",    # US fighter exercises
    "TOPCAT",  # UK RAF
    "ASCOT",   # UK RAF transport
    "RRR",     # UK RAF
    "GAF",     # German Air Force
    "FAF",     # French Air Force
    "IAM",     # Italian Air Force
    "CNV",     # US Navy carrier aviation
    "NAVY",    # Various navy flights
    "DOOM",    # US nuclear command (E-6B)
    "IRON",    # US B-52 / bomber callsigns
    "TROJAN",  # US Army fixed wing
    "CASA",    # Spanish Air Force
    "PLF",     # Polish Air Force
    "BAF",     # Belgian Air Force
    "HAF",     # Hellenic Air Force
    "TKF",     # Turkish Air Force
]

# Hotspot bounding boxes: (lat_min, lat_max, lon_min, lon_max)
HOTSPOT_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "black_sea":       (41.0, 47.0, 27.5, 42.0),
    "eastern_med":     (31.0, 37.0, 24.0, 36.0),
    "taiwan_strait":   (22.0, 26.0, 117.0, 122.0),
    "korean_dmz":      (37.0, 39.0, 125.0, 130.0),
    "persian_gulf":    (24.0, 30.0, 48.0, 57.0),
    "south_china_sea": (5.0, 22.0, 105.0, 121.0),
    "baltics":         (54.0, 60.0, 20.0, 28.0),
}

# ICAO24 address blocks commonly allocated to military (hex ranges by country)
# This is a simplified heuristic; production would use a full database.
_MILITARY_ICAO_RANGES: list[tuple[int, int]] = [
    (0xADF7C0, 0xAFFFFF),  # US military
    (0x3F0000, 0x3FFFFF),  # France military
    (0x3CC000, 0x3CFFFF),  # Germany military
    (0x43C000, 0x43CFFF),  # UK military
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class MilitaryActivity:
    """A single detected military aircraft or vessel."""

    activity_type: str  # "flight" | "vessel"
    callsign: str
    country: str
    latitude: float
    longitude: float
    altitude: float  # meters, for flights; 0 for vessels
    heading: float  # degrees
    speed: float  # m/s for flights, knots for vessels
    aircraft_type: str  # inferred type or "unknown"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_unusual: bool = False
    region: str = ""


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


class MilitaryMonitor:
    """Monitors military airspace activity via OpenSky Network.

    Identifies military flights by callsign prefix matching and ICAO24
    address ranges.  Tracks activity counts per region to detect surges.
    """

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=30.0)

        # Rate limiter
        self._request_timestamps: list[float] = []

        # Circuit breaker
        self._consecutive_failures: int = 0
        self._last_failure_at: float = 0.0

        # Historical activity counts per region for surge detection
        self._region_history: dict[str, list[int]] = defaultdict(list)
        _REGION_HISTORY_MAX = 100  # Keep last 100 observations per region
        self._region_history_max = _REGION_HISTORY_MAX

    # -- Rate limiting -------------------------------------------------------

    async def _wait_for_rate_limit(self) -> None:
        now = time.monotonic()
        self._request_timestamps = [
            ts for ts in self._request_timestamps
            if now - ts < _RATE_LIMIT_WINDOW_SECONDS
        ]
        if len(self._request_timestamps) >= _RATE_LIMIT_MAX_REQUESTS:
            oldest = self._request_timestamps[0]
            wait = _RATE_LIMIT_WINDOW_SECONDS - (now - oldest) + 0.5
            if wait > 0:
                logger.debug("OpenSky rate limit: sleeping %.1fs", wait)
                await asyncio.sleep(wait)
        self._request_timestamps.append(time.monotonic())

    # -- Circuit breaker -----------------------------------------------------

    def _circuit_open(self) -> bool:
        if self._consecutive_failures < _CB_MAX_FAILURES:
            return False
        elapsed = time.monotonic() - self._last_failure_at
        if elapsed >= _CB_COOLDOWN_SECONDS:
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self) -> None:
        self._consecutive_failures = 0

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        self._last_failure_at = time.monotonic()

    # -- Military identification ---------------------------------------------

    @staticmethod
    def _is_military_callsign(callsign: str) -> bool:
        """Check if a callsign matches known military patterns."""
        cs = callsign.strip().upper()
        if not cs:
            return False
        for prefix in MILITARY_CALLSIGN_PREFIXES:
            if cs.startswith(prefix):
                return True
        return False

    @staticmethod
    def _is_military_icao(icao24_hex: str) -> bool:
        """Check if an ICAO24 address falls in a military allocation block."""
        try:
            addr = int(icao24_hex.strip(), 16)
        except (ValueError, AttributeError):
            return False
        for low, high in _MILITARY_ICAO_RANGES:
            if low <= addr <= high:
                return True
        return False

    @staticmethod
    def _infer_aircraft_type(callsign: str) -> str:
        """Best-effort aircraft type inference from callsign."""
        cs = callsign.strip().upper()
        type_map = {
            "FORTE": "RQ-4 Global Hawk",
            "RCH": "C-17 Globemaster",
            "REACH": "C-17/C-5 Transport",
            "JAKE": "KC-135 Tanker",
            "HOMER": "KC-135/KC-46 Tanker",
            "DOOM": "E-6B Mercury",
            "IRON": "B-52 Stratofortress",
            "NATO": "E-3 AWACS",
            "VIPER": "F-16",
            "NOBLE": "NATO Mission",
        }
        for prefix, atype in type_map.items():
            if cs.startswith(prefix):
                return atype
        return "unknown"

    def _classify_region(self, lat: float, lon: float) -> str:
        """Determine which hotspot region a coordinate falls in."""
        for region, (lat_min, lat_max, lon_min, lon_max) in HOTSPOT_BBOXES.items():
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                return region
        return "other"

    # -- Fetching ------------------------------------------------------------

    async def _fetch_opensky_states(
        self,
        bbox: Optional[tuple[float, float, float, float]] = None,
    ) -> list[list]:
        """Fetch state vectors from OpenSky Network.

        Args:
            bbox: Optional (lat_min, lat_max, lon_min, lon_max) bounding box.

        Returns:
            List of state vector arrays from the API.
        """
        if self._circuit_open():
            logger.warning("OpenSky circuit breaker open, skipping")
            return []

        await self._wait_for_rate_limit()

        params: dict[str, str] = {}
        if bbox:
            lat_min, lat_max, lon_min, lon_max = bbox
            params["lamin"] = str(lat_min)
            params["lamax"] = str(lat_max)
            params["lomin"] = str(lon_min)
            params["lomax"] = str(lon_max)

        try:
            resp = await self._client.get(OPENSKY_API_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            self._record_failure()
            logger.error("OpenSky API error (failure %d): %s", self._consecutive_failures, exc)
            return []

        self._record_success()
        states = data.get("states", [])
        return states if isinstance(states, list) else []

    def _parse_state_to_activity(
        self,
        state: list,
        region: str = "",
    ) -> Optional[MilitaryActivity]:
        """Parse an OpenSky state vector into a MilitaryActivity.

        OpenSky state vector indices:
            0: icao24, 1: callsign, 2: origin_country, 3: time_position,
            4: last_contact, 5: longitude, 6: latitude, 7: baro_altitude,
            8: on_ground, 9: velocity, 10: true_track, ...
        """
        if len(state) < 11:
            return None

        icao24 = str(state[0] or "")
        callsign = str(state[1] or "").strip()
        origin_country = str(state[2] or "")
        longitude = state[5]
        latitude = state[6]
        altitude = state[7]
        on_ground = state[8]
        velocity = state[9]
        heading = state[10]

        # Skip aircraft on the ground
        if on_ground:
            return None

        # Skip entries with missing position
        if latitude is None or longitude is None:
            return None

        # Check for military identification
        is_mil_callsign = self._is_military_callsign(callsign)
        is_mil_icao = self._is_military_icao(icao24)

        if not is_mil_callsign and not is_mil_icao:
            return None

        return MilitaryActivity(
            activity_type="flight",
            callsign=callsign,
            country=origin_country,
            latitude=float(latitude),
            longitude=float(longitude),
            altitude=float(altitude or 0),
            heading=float(heading or 0),
            speed=float(velocity or 0),
            aircraft_type=self._infer_aircraft_type(callsign),
            is_unusual=False,  # Updated later by surge detection
            region=region or self._classify_region(float(latitude), float(longitude)),
        )

    async def fetch_military_flights(
        self,
        region: Optional[str] = None,
    ) -> list[MilitaryActivity]:
        """Fetch and filter military flights from OpenSky.

        Args:
            region: If specified, only scan that hotspot region.
                    Otherwise scans all HOTSPOT_BBOXES.

        Returns:
            List of identified military flights.
        """
        activities: list[MilitaryActivity] = []

        if region and region in HOTSPOT_BBOXES:
            regions = {region: HOTSPOT_BBOXES[region]}
        elif region:
            logger.warning("Unknown region '%s', scanning all", region)
            regions = HOTSPOT_BBOXES
        else:
            regions = HOTSPOT_BBOXES

        for rgn_name, bbox in regions.items():
            states = await self._fetch_opensky_states(bbox=bbox)
            for state in states:
                activity = self._parse_state_to_activity(state, region=rgn_name)
                if activity:
                    activities.append(activity)

            # Small delay between region queries to be respectful
            if len(regions) > 1:
                await asyncio.sleep(1.0)

        # Record counts for surge detection
        region_counts: dict[str, int] = defaultdict(int)
        for act in activities:
            region_counts[act.region] += 1

        for rgn, count in region_counts.items():
            self._region_history[rgn].append(count)
            if len(self._region_history[rgn]) > self._region_history_max:
                self._region_history[rgn] = self._region_history[rgn][-self._region_history_max:]

        # Mark flights in surge regions as unusual
        surge_regions = set(self.get_surge_regions())
        for act in activities:
            if act.region in surge_regions:
                act.is_unusual = True

        logger.info(
            "Military monitor: %d flights across %d regions",
            len(activities),
            len(region_counts),
        )
        return activities

    async def fetch_vessel_activity(
        self,
        region: Optional[str] = None,
    ) -> list[MilitaryActivity]:
        """Fetch military vessel activity via AIS data.

        This is a placeholder: AISStream requires a paid API key.
        Returns an empty list and logs a warning if unconfigured.
        """
        ais_key = getattr(settings, "AISSTREAM_API_KEY", None)
        if not ais_key:
            logger.debug(
                "AISStream API key not configured; vessel monitoring disabled"
            )
            return []

        # Placeholder for future AIS integration
        logger.info("Vessel monitoring not yet implemented (key present but client stub)")
        return []

    # -- Summaries -----------------------------------------------------------

    async def get_activity_summary(self) -> dict:
        """Return aggregate counts of recent military activity by region and type."""
        flights = await self.fetch_military_flights()
        vessels = await self.fetch_vessel_activity()
        all_activity = flights + vessels

        by_region: dict[str, int] = defaultdict(int)
        by_type: dict[str, int] = defaultdict(int)

        for act in all_activity:
            by_region[act.region] += 1
            by_type[act.activity_type] += 1

        return {
            "total": len(all_activity),
            "by_region": dict(by_region),
            "by_type": dict(by_type),
            "surge_regions": self.get_surge_regions(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_surge_regions(self) -> list[str]:
        """Return regions with above-average military activity.

        A region is in "surge" if the latest count exceeds the historical
        mean + 1 standard deviation for that region.
        """
        import math

        surges: list[str] = []
        for region, counts in self._region_history.items():
            if len(counts) < 5:
                continue

            mean = sum(counts) / len(counts)
            variance = sum((c - mean) ** 2 for c in counts) / len(counts)
            std = math.sqrt(variance) if variance > 0 else 0.0

            latest = counts[-1]
            if latest > mean + std and std > 0:
                surges.append(region)

        return surges


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

military_monitor = MilitaryMonitor()
