"""Infrastructure monitoring: internet outages and cascade modeling.

Tracks internet outages via the Cloudflare Radar API and models
cascade impacts through a simplified global infrastructure dependency
graph (undersea cables, chokepoints, major ports).

Critical for prediction markets involving trade disruption, regional
stability, and supply-chain exposure.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import httpx

from config import settings
from .infrastructure_catalog import infrastructure_catalog
from .military_catalog import military_catalog

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Cloudflare Radar free endpoint (no auth required for basic annotations)
CLOUDFLARE_RADAR_URL = "https://radar.cloudflare.com/api/v1/annotations"

# Authenticated Cloudflare API (optional, higher limits)
CLOUDFLARE_API_URL = "https://api.cloudflare.com/client/v4/radar/annotations/outages"

# Cache TTL for outage results (don't hammer the API)
_CACHE_TTL_SECONDS = 600  # 10 minutes

# BFS cascade propagation depth limit
_MAX_CASCADE_HOPS = 3


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class InfrastructureEvent:
    """A detected infrastructure disruption."""

    event_type: str  # "internet_outage" | "cable_fault" | "port_disruption"
    country: str
    severity: float  # 0-1
    started_at: datetime
    description: str
    source: str
    affected_services: list[str] = field(default_factory=list)
    cascade_risk_score: float = 0.0  # 0-1, computed by cascade model
    latitude: Optional[float] = None
    longitude: Optional[float] = None


# ---------------------------------------------------------------------------
# Infrastructure dependency graph
# ---------------------------------------------------------------------------

def _graph_state() -> tuple[
    set[str],
    dict[str, list[tuple[str, float]]],
    dict[str, float],
    dict[str, dict[str, float]],
    dict[str, list[str]],
]:
    nodes = infrastructure_catalog.nodes()
    edges = infrastructure_catalog.edges()
    redundancy = infrastructure_catalog.redundancy()
    trade_dependencies = infrastructure_catalog.trade_dependencies()
    country_to_nodes = infrastructure_catalog.country_to_nodes()

    adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for src, dst, weight in edges:
        adjacency[src].append((dst, weight))

    return nodes, adjacency, redundancy, trade_dependencies, country_to_nodes


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


class InfrastructureMonitor:
    """Monitors internet outages and models infrastructure cascade risks.

    Uses Cloudflare Radar for real-time outage data and a simplified
    dependency graph for cascade impact modelling.
    """

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=30.0)
        self._cf_api_token: Optional[str] = getattr(settings, "CLOUDFLARE_API_TOKEN", None)

        # Outage cache
        self._cached_outages: list[InfrastructureEvent] = []
        self._cache_timestamp: float = 0.0
        self._last_error: Optional[str] = None

    @staticmethod
    def _normalize_iso3(value: str) -> str:
        text = str(value or "").strip().upper()
        if not text:
            return ""
        if len(text) == 3 and text.isalpha():
            return text
        aliases = military_catalog.country_aliases()
        return aliases.get(text, "")

    # -- Cloudflare Radar ----------------------------------------------------

    async def fetch_outages(self) -> list[InfrastructureEvent]:
        """Fetch recent internet outages from Cloudflare Radar.

        Results are cached for 10 minutes to avoid excessive API calls.
        """
        now = time.monotonic()
        if (
            self._cached_outages
            and now - self._cache_timestamp < _CACHE_TTL_SECONDS
        ):
            return list(self._cached_outages)

        events: list[InfrastructureEvent] = []

        # Try authenticated endpoint first, fall back to free
        if self._cf_api_token:
            events = await self._fetch_cf_authenticated()
        if not events:
            events = await self._fetch_cf_free()

        self._cached_outages = events
        self._cache_timestamp = time.monotonic()
        if events:
            self._last_error = None

        if events:
            logger.info("Infrastructure monitor: %d outages fetched", len(events))
        return events

    async def _fetch_cf_authenticated(self) -> list[InfrastructureEvent]:
        """Fetch from the authenticated Cloudflare API v4 endpoint."""
        headers = {
            "Authorization": f"Bearer {self._cf_api_token}",
            "Content-Type": "application/json",
        }
        params = {"limit": "25", "dateRange": "1d"}

        try:
            resp = await self._client.get(
                CLOUDFLARE_API_URL, headers=headers, params=params,
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Cloudflare API v4 error: %s", exc)
            self._last_error = str(exc)
            return []

        self._last_error = None
        return self._parse_cf_response(data)

    async def _fetch_cf_free(self) -> list[InfrastructureEvent]:
        """Fetch from the free Cloudflare Radar endpoint."""
        params = {"dateRange": "1d"}

        try:
            resp = await self._client.get(CLOUDFLARE_RADAR_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Cloudflare Radar free endpoint error: %s", exc)
            self._last_error = str(exc)
            return []

        self._last_error = None
        return self._parse_cf_response(data)

    def _parse_cf_response(self, data: dict) -> list[InfrastructureEvent]:
        """Parse Cloudflare Radar response into InfrastructureEvent instances."""
        events: list[InfrastructureEvent] = []

        # The response structure varies between endpoints; handle both
        annotations = (
            data.get("result", {}).get("annotations", [])
            or data.get("annotations", [])
            or data.get("data", [])
        )

        for ann in annotations:
            if not isinstance(ann, dict):
                continue

            country = str(ann.get("locations", ann.get("country", "unknown")))
            # Cloudflare sometimes returns a list of location codes
            if isinstance(ann.get("locations"), list):
                country = ann["locations"][0] if ann["locations"] else "unknown"
            country_iso3 = self._normalize_iso3(country) or str(country).upper()

            description = str(
                ann.get("description", ann.get("text", "Internet outage"))
            )

            # Parse start time
            start_str = ann.get("startDate", ann.get("start", ""))
            try:
                started_at = datetime.fromisoformat(
                    str(start_str).replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                started_at = datetime.now(timezone.utc)

            # Estimate severity from scope/type hints
            scope = str(ann.get("scope", "")).lower()
            if "nationwide" in scope or "country" in scope:
                severity = 0.9
            elif "regional" in scope or "province" in scope:
                severity = 0.6
            elif "city" in scope or "local" in scope:
                severity = 0.3
            else:
                severity = 0.5

            lat = ann.get("latitude", ann.get("lat"))
            lon = ann.get("longitude", ann.get("lon"))
            if isinstance(ann.get("location"), dict):
                lat = ann["location"].get("lat", lat)
                lon = ann["location"].get("lon", lon)
            try:
                lat_f = float(lat) if lat is not None else None
            except Exception:
                lat_f = None
            try:
                lon_f = float(lon) if lon is not None else None
            except Exception:
                lon_f = None

            event = InfrastructureEvent(
                event_type="internet_outage",
                country=country_iso3,
                severity=severity,
                started_at=started_at,
                description=description,
                source="cloudflare_radar",
                affected_services=["internet"],
                cascade_risk_score=0.0,  # Computed separately
                latitude=lat_f,
                longitude=lon_f,
            )
            events.append(event)

        return events

    # -- Cascade modelling ---------------------------------------------------

    @staticmethod
    def compute_cascade_impact(disrupted_node: str) -> dict[str, float]:
        """Compute downstream impact of disrupting an infrastructure node.

        Uses BFS propagation up to 3 hops through the dependency graph.
        Impact at each hop is attenuated by edge weight and node redundancy.

        Args:
            disrupted_node: A node key from the infrastructure graph.

        Returns:
            Dict of node -> impact_score (0-1) for all affected nodes.
        """
        nodes, adjacency, redundancy, _, _ = _graph_state()
        if disrupted_node not in nodes:
            return {}

        impacts: dict[str, float] = {disrupted_node: 1.0}
        frontier: deque[tuple[str, float, int]] = deque()
        frontier.append((disrupted_node, 1.0, 0))

        while frontier:
            node, current_impact, depth = frontier.popleft()
            if depth >= _MAX_CASCADE_HOPS:
                continue

            for neighbor, weight in adjacency.get(node, []):
                neighbor_redundancy = redundancy.get(neighbor, 0.0)
                # Impact = upstream_impact * edge_weight * (1 - redundancy)
                propagated = current_impact * weight * (1.0 - neighbor_redundancy)
                if propagated < 0.01:
                    continue  # Below significance threshold

                existing = impacts.get(neighbor, 0.0)
                if propagated > existing:
                    impacts[neighbor] = round(propagated, 3)
                    frontier.append((neighbor, propagated, depth + 1))

        # Remove the source node from results (it's fully disrupted by definition)
        impacts.pop(disrupted_node, None)
        return impacts

    # -- Public API ----------------------------------------------------------

    async def get_current_disruptions(self) -> list[InfrastructureEvent]:
        """Fetch outages and compute cascade risk scores for each."""
        events = await self.fetch_outages()

        _, _, _, _, country_to_nodes = _graph_state()
        for event in events:
            # Find infrastructure nodes related to this outage's country
            country_code = self._normalize_iso3(event.country) or str(event.country).upper()
            related_nodes = country_to_nodes.get(country_code, [])

            max_cascade = 0.0
            for node in related_nodes:
                cascade = self.compute_cascade_impact(node)
                if cascade:
                    total = sum(cascade.values()) / len(cascade)
                    max_cascade = max(max_cascade, total)

            event.cascade_risk_score = round(max_cascade, 2)

        return events

    def get_cascade_risks(self) -> list[dict]:
        """Compute cascade impacts for all currently disrupted nodes.

        Returns a list of dicts with node, impact map, and affected countries.
        """
        _, _, _, trade_dependencies, country_to_nodes = _graph_state()
        risks: list[dict] = []

        for event in self._cached_outages:
            country_code = self._normalize_iso3(event.country) or str(event.country).upper()
            related_nodes = country_to_nodes.get(country_code, [])

            for node in related_nodes:
                cascade = self.compute_cascade_impact(node)
                if not cascade:
                    continue

                # Map cascade impacts to affected countries
                affected: list[str] = []
                for country_iso3, deps in trade_dependencies.items():
                    for affected_node, impact in cascade.items():
                        if affected_node in deps:
                            weighted = impact * deps[affected_node]
                            if weighted > 0.1:
                                affected.append(country_iso3)
                                break

                risks.append({
                    "disrupted_node": node,
                    "source_event": event.description,
                    "cascade_impacts": cascade,
                    "affected_countries": sorted(set(affected)),
                    "max_impact": max(cascade.values()) if cascade else 0.0,
                })

        return sorted(risks, key=lambda r: r["max_impact"], reverse=True)

    @staticmethod
    def get_affected_countries(event: InfrastructureEvent) -> list[str]:
        """Return countries likely affected by an infrastructure event.

        Uses the trade dependency map to find countries with exposure
        to infrastructure nodes associated with the event's location.
        """
        _, _, _, trade_dependencies, country_to_nodes = _graph_state()
        country_code = InfrastructureMonitor._normalize_iso3(event.country) or str(event.country).upper()
        related_nodes = country_to_nodes.get(country_code, [])

        affected: set[str] = set()
        for country_iso3, deps in trade_dependencies.items():
            for node in related_nodes:
                if node in deps and deps[node] >= 0.3:
                    affected.add(country_iso3)

        return sorted(affected)

    def get_health(self) -> dict[str, object]:
        return {
            "enabled": True,
            "authenticated": bool(self._cf_api_token),
            "cached_outages": len(self._cached_outages),
            "cache_age_seconds": (
                round(max(0.0, time.monotonic() - self._cache_timestamp), 1)
                if self._cache_timestamp
                else None
            ),
            "last_error": self._last_error,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

infrastructure_monitor = InfrastructureMonitor()
