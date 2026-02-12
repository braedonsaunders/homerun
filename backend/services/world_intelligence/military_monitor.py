"""Military activity monitor via OpenSky and AIS stream providers."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from config import settings
from .military_catalog import military_catalog
from .region_catalog import region_catalog

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENSKY_API_URL = "https://opensky-network.org/api/states/all"

# Rate limiting: OpenSky free tier is aggressive (max ~10 req/min for anon)
_RATE_LIMIT_MAX_REQUESTS = 10
_RATE_LIMIT_WINDOW_SECONDS = 60.0

# Circuit breaker
_CB_MAX_FAILURES = int(
    max(1, getattr(settings, "WORLD_INTEL_OPENSKY_CB_MAX_FAILURES", 6) or 6)
)
_CB_COOLDOWN_SECONDS = float(
    max(30.0, getattr(settings, "WORLD_INTEL_OPENSKY_CB_COOLDOWN_SECONDS", 120.0) or 120.0)
)

_AIS_WS_URL = str(
    getattr(settings, "WORLD_INTEL_AIS_WS_URL", "wss://stream.aisstream.io/v0/stream")
    or "wss://stream.aisstream.io/v0/stream"
)
_AIS_MAX_MESSAGES = int(max(10, getattr(settings, "WORLD_INTEL_AIS_MAX_MESSAGES", 250) or 250))
_AIS_SAMPLE_SECONDS = int(max(3, getattr(settings, "WORLD_INTEL_AIS_SAMPLE_SECONDS", 10) or 10))
_AIS_ENABLED = bool(getattr(settings, "WORLD_INTEL_AIS_ENABLED", True))
_AIRPLANES_LIVE_ENABLED = bool(getattr(settings, "WORLD_INTEL_AIRPLANES_LIVE_ENABLED", True))
_AIRPLANES_LIVE_URL = str(
    getattr(settings, "WORLD_INTEL_AIRPLANES_LIVE_URL", "https://api.airplanes.live/v2/mil")
    or "https://api.airplanes.live/v2/mil"
)
_AIRPLANES_LIVE_TIMEOUT_SECONDS = float(
    max(5.0, getattr(settings, "WORLD_INTEL_AIRPLANES_LIVE_TIMEOUT_SECONDS", 20.0) or 20.0)
)
_AIRPLANES_LIVE_MAX_RECORDS = int(
    max(50, getattr(settings, "WORLD_INTEL_AIRPLANES_LIVE_MAX_RECORDS", 1500) or 1500)
)
_FLIGHT_DEDUPE_RADIUS_KM = float(
    max(5.0, getattr(settings, "WORLD_INTEL_MILITARY_DEDUPE_RADIUS_KM", 45.0) or 45.0)
)


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
    provider: str = ""
    transponder: str = ""
    providers: list[str] = field(default_factory=list)


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
        self._opensky_username: Optional[str] = getattr(settings, "OPENSKY_USERNAME", None)
        self._opensky_password: Optional[str] = getattr(settings, "OPENSKY_PASSWORD", None)
        self._ais_key: Optional[str] = getattr(settings, "AISSTREAM_API_KEY", None)

        # Rate limiter
        self._request_timestamps: list[float] = []

        # Circuit breaker
        self._consecutive_failures: int = 0
        self._last_failure_at: float = 0.0
        self._last_error: Optional[str] = None
        self._last_vessel_error: Optional[str] = None
        self._last_airplanes_live_error: Optional[str] = None
        self._last_total_states: int = 0
        self._last_identified_flights: int = 0
        self._last_identified_flights_by_provider: dict[str, int] = {
            "opensky": 0,
            "airplanes_live": 0,
        }
        self._last_deduped_flights: int = 0

        # Historical activity counts per region for surge detection
        self._region_history: dict[str, list[int]] = defaultdict(list)
        _REGION_HISTORY_MAX = 100  # Keep last 100 observations per region
        self._region_history_max = _REGION_HISTORY_MAX

    @staticmethod
    def _normalize_iso3(country: str) -> str:
        text = str(country or "").strip().upper()
        if not text:
            return ""
        if len(text) == 3 and text.isalpha():
            return text
        aliases = military_catalog.country_aliases()
        return aliases.get(text, "")

    @staticmethod
    def _hotspot_bboxes() -> dict[str, tuple[float, float, float, float]]:
        return region_catalog.hotspot_bboxes()

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
        self._last_error = None

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
        for prefix in military_catalog.callsign_prefixes():
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
        for item in military_catalog.icao_ranges():
            if item.start <= addr <= item.end:
                return True
        return False

    @staticmethod
    def _infer_aircraft_type(callsign: str) -> str:
        """Best-effort aircraft type inference from callsign."""
        cs = callsign.strip().upper()
        for prefix, atype in military_catalog.aircraft_type_map().items():
            if cs.startswith(prefix):
                return atype
        return "unknown"

    def _classify_region(self, lat: float, lon: float) -> str:
        """Determine which hotspot region a coordinate falls in."""
        for region, (lat_min, lat_max, lon_min, lon_max) in self._hotspot_bboxes().items():
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                return region
        return "other"

    @staticmethod
    def _normalize_callsign(callsign: str) -> str:
        return "".join(str(callsign or "").strip().upper().split())

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        import math

        lat1_r = math.radians(float(lat1))
        lon1_r = math.radians(float(lon1))
        lat2_r = math.radians(float(lat2))
        lon2_r = math.radians(float(lon2))
        d_lat = lat2_r - lat1_r
        d_lon = lon2_r - lon1_r
        a = (
            math.sin(d_lat / 2.0) ** 2
            + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(d_lon / 2.0) ** 2
        )
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
        return 6371.0 * c

    @staticmethod
    def _flight_quality_score(activity: MilitaryActivity) -> float:
        score = 0.0
        if activity.transponder:
            score += 2.0
        if activity.callsign:
            score += 1.5
        if activity.altitude > 0:
            score += 0.75
        if activity.speed > 0:
            score += 0.5
        if activity.country:
            score += 0.3
        provider = str(activity.provider or "").strip().lower()
        if provider == "opensky":
            score += 0.25
        if provider == "airplanes_live":
            score += 0.15
        return score

    def _flights_match(self, a: MilitaryActivity, b: MilitaryActivity) -> bool:
        if a.activity_type != "flight" or b.activity_type != "flight":
            return False

        a_hex = str(a.transponder or "").strip().lower()
        b_hex = str(b.transponder or "").strip().lower()
        if a_hex and b_hex and a_hex == b_hex:
            return True

        a_callsign = self._normalize_callsign(a.callsign)
        b_callsign = self._normalize_callsign(b.callsign)
        if not a_callsign or not b_callsign:
            return False
        if a_callsign != b_callsign:
            return False

        try:
            distance = self._haversine_km(
                float(a.latitude),
                float(a.longitude),
                float(b.latitude),
                float(b.longitude),
            )
        except Exception:
            return False
        return distance <= _FLIGHT_DEDUPE_RADIUS_KM

    def _merge_flights(self, a: MilitaryActivity, b: MilitaryActivity) -> MilitaryActivity:
        first = a if self._flight_quality_score(a) >= self._flight_quality_score(b) else b
        second = b if first is a else a

        merged_providers = sorted(
            {
                str(provider).strip().lower()
                for provider in [*first.providers, *second.providers, first.provider, second.provider]
                if str(provider or "").strip()
            }
        )
        if not merged_providers:
            merged_providers = [str(first.provider or "unknown").strip().lower() or "unknown"]

        return MilitaryActivity(
            activity_type="flight",
            callsign=first.callsign or second.callsign,
            country=first.country or second.country,
            latitude=float(first.latitude),
            longitude=float(first.longitude),
            altitude=float(first.altitude if first.altitude else second.altitude),
            heading=float(first.heading if first.heading else second.heading),
            speed=float(first.speed if first.speed else second.speed),
            aircraft_type=first.aircraft_type or second.aircraft_type or "unknown",
            timestamp=max(first.timestamp, second.timestamp),
            is_unusual=bool(first.is_unusual or second.is_unusual),
            region=first.region or second.region or self._classify_region(first.latitude, first.longitude),
            provider=first.provider or second.provider,
            transponder=first.transponder or second.transponder,
            providers=merged_providers,
        )

    def _dedupe_flights(self, flights: list[MilitaryActivity]) -> list[MilitaryActivity]:
        if not flights:
            return []
        deduped: list[MilitaryActivity] = []
        for flight in sorted(flights, key=self._flight_quality_score, reverse=True):
            merged = False
            for idx, existing in enumerate(deduped):
                if not self._flights_match(existing, flight):
                    continue
                deduped[idx] = self._merge_flights(existing, flight)
                merged = True
                break
            if not merged:
                if not flight.providers:
                    flight.providers = [str(flight.provider or "unknown").strip().lower() or "unknown"]
                deduped.append(flight)
        return deduped

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
            resp = await self._client.get(
                OPENSKY_API_URL,
                params=params,
                auth=(
                    (self._opensky_username, self._opensky_password)
                    if self._opensky_username and self._opensky_password
                    else None
                ),
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            self._record_failure()
            self._last_error = str(exc)
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
            country=self._normalize_iso3(origin_country),
            latitude=float(latitude),
            longitude=float(longitude),
            altitude=float(altitude or 0),
            heading=float(heading or 0),
            speed=float(velocity or 0),
            aircraft_type=self._infer_aircraft_type(callsign),
            is_unusual=False,  # Updated later by surge detection
            region=region or self._classify_region(float(latitude), float(longitude)),
            provider="opensky",
            transponder=icao24.strip().lower(),
            providers=["opensky"],
        )

    async def _collect_opensky_flights(
        self,
        region: Optional[str],
    ) -> tuple[list[MilitaryActivity], int]:
        activities: list[MilitaryActivity] = []
        total_states_seen = 0

        if region:
            hotspots = self._hotspot_bboxes()
            if region in hotspots:
                states = await self._fetch_opensky_states(bbox=hotspots[region])
                total_states_seen += len(states)
                for state in states:
                    activity = self._parse_state_to_activity(state, region=region)
                    if activity:
                        activities.append(activity)
                return activities, total_states_seen
            logger.warning("Unknown region '%s', falling back to global scan", region)

        states = await self._fetch_opensky_states(bbox=None)
        total_states_seen += len(states)
        for state in states:
            activity = self._parse_state_to_activity(state)
            if activity:
                activities.append(activity)
        return activities, total_states_seen

    def _parse_airplanes_live_record(
        self,
        row: dict[str, Any],
    ) -> Optional[MilitaryActivity]:
        if not isinstance(row, dict):
            return None

        lat_raw = row.get("lat")
        lon_raw = row.get("lon")
        if lat_raw is None or lon_raw is None:
            return None

        try:
            latitude = float(lat_raw)
            longitude = float(lon_raw)
        except Exception:
            return None

        callsign = str(
            row.get("flight")
            or row.get("r")
            or row.get("callsign")
            or row.get("hex")
            or ""
        ).strip()
        transponder = str(row.get("hex") or "").strip().lower()

        altitude_m = 0.0
        alt_baro = row.get("alt_baro")
        if alt_baro not in (None, "", "ground"):
            try:
                altitude_m = float(alt_baro) * 0.3048
            except Exception:
                altitude_m = 0.0
        elif str(alt_baro).strip().lower() == "ground":
            return None

        try:
            heading = float(row.get("track") or row.get("true_heading") or 0.0)
        except Exception:
            heading = 0.0
        try:
            # Provider reports ground speed in knots.
            speed = float(row.get("gs") or 0.0) * 0.514444
        except Exception:
            speed = 0.0

        type_code = str(row.get("t") or "").strip().upper()
        desc = str(row.get("desc") or "").strip()
        aircraft_type = type_code or desc or self._infer_aircraft_type(callsign)

        owner = str(row.get("ownOp") or row.get("ownOpCode") or "").strip()
        country = self._normalize_iso3(owner)

        return MilitaryActivity(
            activity_type="flight",
            callsign=callsign,
            country=country,
            latitude=latitude,
            longitude=longitude,
            altitude=altitude_m,
            heading=heading,
            speed=speed,
            aircraft_type=aircraft_type,
            is_unusual=False,
            region=self._classify_region(latitude, longitude),
            provider="airplanes_live",
            transponder=transponder,
            providers=["airplanes_live"],
        )

    async def _fetch_airplanes_live_flights(
        self,
        region: Optional[str] = None,
    ) -> list[MilitaryActivity]:
        if not _AIRPLANES_LIVE_ENABLED:
            self._last_airplanes_live_error = "disabled"
            return []

        try:
            resp = await self._client.get(
                _AIRPLANES_LIVE_URL,
                timeout=_AIRPLANES_LIVE_TIMEOUT_SECONDS,
            )
            if resp.status_code == 429:
                self._last_airplanes_live_error = "rate_limited_429"
                return []
            resp.raise_for_status()
            payload = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            self._last_airplanes_live_error = str(exc)
            logger.warning("airplanes.live query failed: %s", exc)
            return []

        rows = payload.get("ac", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            self._last_airplanes_live_error = "unexpected_payload"
            return []

        activities: list[MilitaryActivity] = []
        for row in rows[:_AIRPLANES_LIVE_MAX_RECORDS]:
            parsed = self._parse_airplanes_live_record(row)
            if not parsed:
                continue
            if region and parsed.region != region:
                continue
            activities.append(parsed)

        self._last_airplanes_live_error = None
        return activities

    async def fetch_military_flights(
        self,
        region: Optional[str] = None,
    ) -> list[MilitaryActivity]:
        """Fetch and filter military flights from OpenSky.

        Args:
            region: If specified, only scan that hotspot region.
                    Otherwise scans globally.

        Returns:
            List of identified military flights.
        """
        opensky_flights: list[MilitaryActivity] = []
        airplanes_live_flights: list[MilitaryActivity] = []
        total_states_seen = 0

        target_region = None
        if region:
            if region in self._hotspot_bboxes():
                target_region = region
            else:
                logger.warning("Unknown region '%s', falling back to global scan", region)

        opensky_result, airplanes_result = await asyncio.gather(
            self._collect_opensky_flights(target_region),
            self._fetch_airplanes_live_flights(target_region),
            return_exceptions=True,
        )

        if isinstance(opensky_result, Exception):
            self._last_error = str(opensky_result)
            logger.warning("OpenSky collection failed: %s", opensky_result)
        else:
            opensky_flights, total_states_seen = opensky_result

        if isinstance(airplanes_result, Exception):
            self._last_airplanes_live_error = str(airplanes_result)
            logger.warning("airplanes.live collection failed: %s", airplanes_result)
        else:
            airplanes_live_flights = airplanes_result

        activities = self._dedupe_flights(opensky_flights + airplanes_live_flights)
        self._last_total_states = int(total_states_seen)
        self._last_identified_flights = int(len(activities))
        self._last_identified_flights_by_provider = {
            "opensky": len(opensky_flights),
            "airplanes_live": len(airplanes_live_flights),
        }
        self._last_deduped_flights = max(
            0,
            (len(opensky_flights) + len(airplanes_live_flights)) - len(activities),
        )

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
            "Military monitor: %d flights (%d opensky + %d airplanes.live, %d deduped) across %d regions",
            len(activities),
            len(opensky_flights),
            len(airplanes_live_flights),
            self._last_deduped_flights,
            len(region_counts),
        )
        return activities

    async def fetch_vessel_activity(
        self,
        region: Optional[str] = None,
    ) -> list[MilitaryActivity]:
        """Fetch military vessel activity via AIS stream messages."""
        if not _AIS_ENABLED:
            self._last_vessel_error = "disabled"
            return []
        if not self._ais_key:
            self._last_vessel_error = "missing_api_key"
            logger.debug("AISStream key not configured; vessel monitoring disabled")
            return []

        try:
            import websockets
        except Exception as exc:
            self._last_vessel_error = f"websocket_dependency_missing:{exc}"
            logger.warning("AIS websocket dependency unavailable: %s", exc)
            return []

        hotspots = self._hotspot_bboxes()
        if region and region in hotspots:
            regions = {region: hotspots[region]}
        elif region:
            logger.warning("Unknown region '%s', scanning all", region)
            regions = hotspots
        else:
            regions = hotspots

        bounding_boxes = [
            [[lon_min, lat_min], [lon_max, lat_max]]
            for lat_min, lat_max, lon_min, lon_max in regions.values()
        ]
        if not bounding_boxes:
            self._last_vessel_error = "no_bounding_boxes"
            return []

        subscription = {
            "APIKey": self._ais_key,
            "BoundingBoxes": bounding_boxes,
            "FilterMessageTypes": ["PositionReport", "ShipStaticData"],
        }

        packets: list[dict] = []
        try:
            async with websockets.connect(_AIS_WS_URL, ping_interval=20, close_timeout=3) as ws:
                await ws.send(json.dumps(subscription))
                started = time.monotonic()
                while (
                    (time.monotonic() - started) < _AIS_SAMPLE_SECONDS
                    and len(packets) < _AIS_MAX_MESSAGES
                ):
                    remaining = _AIS_SAMPLE_SECONDS - (time.monotonic() - started)
                    timeout = max(0.1, min(1.0, remaining))
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    except asyncio.TimeoutError:
                        continue
                    try:
                        parsed = json.loads(raw)
                    except Exception:
                        continue
                    if isinstance(parsed, dict):
                        packets.append(parsed)
        except Exception as exc:
            self._last_vessel_error = str(exc)
            logger.warning("AIS stream error: %s", exc)
            return []

        ship_meta: dict[str, dict[str, Any]] = {}
        activities: list[MilitaryActivity] = []
        vessel_types = military_catalog.vessel_ship_type_codes()
        vessel_keywords = military_catalog.vessel_name_keywords()
        for packet in packets:
            message = packet.get("Message")
            if not isinstance(message, dict):
                continue

            static = message.get("ShipStaticData")
            if isinstance(static, dict):
                mmsi = str(static.get("UserID") or static.get("MMSI") or "").strip()
                if not mmsi:
                    continue
                ship_meta[mmsi] = {
                    "name": str(static.get("Name") or "").strip(),
                    "callsign": str(static.get("CallSign") or "").strip(),
                    "ship_type": static.get("Type"),
                }
                continue

            position = message.get("PositionReport")
            if not isinstance(position, dict):
                continue

            mmsi = str(position.get("UserID") or position.get("MMSI") or "").strip()
            if not mmsi:
                continue

            try:
                lat = float(position.get("Latitude"))
                lon = float(position.get("Longitude"))
            except Exception:
                continue

            meta = ship_meta.get(mmsi, {})
            callsign = str(position.get("CallSign") or meta.get("callsign") or mmsi).strip()
            name = str(position.get("Name") or meta.get("name") or "").strip()

            try:
                ship_type = int(position.get("ShipType") or meta.get("ship_type") or -1)
            except Exception:
                ship_type = -1

            text_blob = f"{callsign} {name}".lower()
            is_military = ship_type in vessel_types or any(kw in text_blob for kw in vessel_keywords)
            if not is_military:
                continue

            vessel_region = self._classify_region(lat, lon)
            if region and vessel_region != region:
                continue

            country = ""
            mid = mmsi[:3]
            if len(mid) == 3 and mid.isdigit():
                country = military_catalog.vessel_mid_iso3().get(mid, "")

            vessel_type = f"AIS type {ship_type}" if ship_type >= 0 else "military vessel"
            try:
                heading = float(position.get("Cog") or position.get("COG") or 0.0)
            except Exception:
                heading = 0.0
            try:
                speed = float(position.get("Sog") or position.get("SOG") or 0.0)
            except Exception:
                speed = 0.0

            activities.append(
                MilitaryActivity(
                    activity_type="vessel",
                    callsign=callsign or mmsi,
                    country=country,
                    latitude=lat,
                    longitude=lon,
                    altitude=0.0,
                    heading=heading,
                    speed=speed,
                    aircraft_type=vessel_type,
                    is_unusual=False,
                    region=vessel_region,
                    provider="aisstream",
                    transponder=mmsi,
                    providers=["aisstream"],
                )
            )

        self._last_vessel_error = None
        logger.info("Military monitor: %d vessels from AIS stream", len(activities))
        return activities

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
            "source_health": self.get_health(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def export_state(self) -> dict[str, Any]:
        return {
            "region_history": {k: list(v[-200:]) for k, v in self._region_history.items()},
            "last_error": self._last_error,
            "last_vessel_error": self._last_vessel_error,
            "last_airplanes_live_error": self._last_airplanes_live_error,
        }

    def import_state(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        raw_history = payload.get("region_history") or {}
        if isinstance(raw_history, dict):
            next_history: dict[str, list[int]] = defaultdict(list)
            for region, counts in raw_history.items():
                if not isinstance(counts, list):
                    continue
                cleaned: list[int] = []
                for count in counts[-self._region_history_max:]:
                    try:
                        cleaned.append(int(count))
                    except Exception:
                        continue
                if cleaned:
                    next_history[str(region)] = cleaned
            if next_history:
                self._region_history = next_history
        self._last_error = str(payload.get("last_error") or "") or None
        self._last_vessel_error = str(payload.get("last_vessel_error") or "") or None
        self._last_airplanes_live_error = str(payload.get("last_airplanes_live_error") or "") or None

    def get_health(self) -> dict[str, object]:
        provider_status = {
            "opensky": {
                "enabled": True,
                "ok": not bool(self._last_error) and not self._circuit_open(),
                "error": self._last_error,
                "identified_flights": int(self._last_identified_flights_by_provider.get("opensky") or 0),
            },
            "airplanes_live": {
                "enabled": _AIRPLANES_LIVE_ENABLED,
                "ok": (not bool(self._last_airplanes_live_error)) or (not _AIRPLANES_LIVE_ENABLED),
                "error": self._last_airplanes_live_error,
                "identified_flights": int(
                    self._last_identified_flights_by_provider.get("airplanes_live") or 0
                ),
                "url": _AIRPLANES_LIVE_URL,
            },
            "aisstream": {
                "enabled": _AIS_ENABLED,
                "ok": (not bool(self._last_vessel_error)) or (not _AIS_ENABLED),
                "error": self._last_vessel_error,
                "configured": bool(self._ais_key),
                "url": _AIS_WS_URL,
            },
        }
        return {
            "enabled": bool(getattr(settings, "WORLD_INTEL_MILITARY_ENABLED", True)),
            "circuit_open": self._circuit_open(),
            "consecutive_failures": self._consecutive_failures,
            "cooldown_seconds": _CB_COOLDOWN_SECONDS,
            "rate_limit_per_minute": _RATE_LIMIT_MAX_REQUESTS,
            "authenticated": bool(self._opensky_username and self._opensky_password),
            "last_error": self._last_error,
            "ais_enabled": _AIS_ENABLED,
            "ais_configured": bool(self._ais_key),
            "ais_ws_url": _AIS_WS_URL,
            "last_vessel_error": self._last_vessel_error,
            "airplanes_live_enabled": _AIRPLANES_LIVE_ENABLED,
            "airplanes_live_url": _AIRPLANES_LIVE_URL,
            "last_airplanes_live_error": self._last_airplanes_live_error,
            "last_total_states_seen": self._last_total_states,
            "last_identified_flights": self._last_identified_flights,
            "last_identified_flights_by_provider": dict(self._last_identified_flights_by_provider),
            "last_deduped_flights": int(self._last_deduped_flights),
            "callsign_prefix_count": len(military_catalog.callsign_prefixes()),
            "icao_range_count": len(military_catalog.icao_ranges()),
            "coverage_level": "partial",
            "coverage_scope": (
                "Tracks military-identifiable aircraft across OpenSky + airplanes.live "
                "and military AIS vessels; still partial vs full global military coverage."
            ),
            "providers": ["opensky", "airplanes_live", "aisstream"],
            "provider_status": provider_status,
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
