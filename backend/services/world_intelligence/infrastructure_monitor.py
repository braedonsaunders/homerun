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
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx

from config import settings

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


# ---------------------------------------------------------------------------
# Infrastructure dependency graph
# ---------------------------------------------------------------------------

# Nodes represent key infrastructure assets.  Edges are directed and
# weighted: (from_node, to_node, weight).  Weight represents how much
# the downstream node depends on the upstream one (0-1).

_INFRA_NODES: set[str] = {
    # Chokepoints
    "suez_canal", "strait_of_hormuz", "malacca_strait",
    "panama_canal", "bosphorus_strait",
    # Undersea cables (simplified clusters)
    "transatlantic_cables", "transpacific_cables",
    "asia_africa_cables", "med_cables",
    # Major port clusters
    "shanghai_port", "singapore_port", "rotterdam_port",
    "houston_port", "dubai_port",
}

_INFRA_EDGES: list[tuple[str, str, float]] = [
    # Chokepoint -> cable / port dependencies
    ("suez_canal", "med_cables", 0.7),
    ("suez_canal", "asia_africa_cables", 0.5),
    ("suez_canal", "rotterdam_port", 0.4),
    ("suez_canal", "dubai_port", 0.3),
    ("strait_of_hormuz", "dubai_port", 0.8),
    ("strait_of_hormuz", "asia_africa_cables", 0.3),
    ("malacca_strait", "singapore_port", 0.9),
    ("malacca_strait", "shanghai_port", 0.6),
    ("malacca_strait", "transpacific_cables", 0.4),
    ("panama_canal", "houston_port", 0.5),
    ("bosphorus_strait", "med_cables", 0.4),
    # Cable -> port dependencies
    ("transatlantic_cables", "rotterdam_port", 0.3),
    ("transpacific_cables", "shanghai_port", 0.3),
    ("transpacific_cables", "singapore_port", 0.2),
    ("med_cables", "rotterdam_port", 0.2),
    ("asia_africa_cables", "dubai_port", 0.3),
    ("asia_africa_cables", "singapore_port", 0.3),
]

# Build adjacency list for BFS
_ADJACENCY: dict[str, list[tuple[str, float]]] = defaultdict(list)
for _src, _dst, _w in _INFRA_EDGES:
    _ADJACENCY[_src].append((_dst, _w))

# Redundancy factors: nodes with backup routes have lower cascade impact
_REDUNDANCY: dict[str, float] = {
    "transatlantic_cables": 0.7,   # Many redundant cables
    "transpacific_cables": 0.6,
    "med_cables": 0.5,
    "asia_africa_cables": 0.4,
    "singapore_port": 0.3,        # Major hub with alternatives
    "rotterdam_port": 0.3,
    "shanghai_port": 0.2,
    "houston_port": 0.3,
    "dubai_port": 0.2,
    "suez_canal": 0.1,            # Cape route exists but very slow
    "strait_of_hormuz": 0.05,     # Almost no alternative for oil
    "malacca_strait": 0.15,       # Lombok Strait alternative
    "panama_canal": 0.1,
    "bosphorus_strait": 0.1,
}

# Trade dependencies: country -> chokepoint/node -> dependency weight
# How much does this country's economy depend on this infrastructure?
TRADE_DEPENDENCIES: dict[str, dict[str, float]] = {
    "JPN": {"strait_of_hormuz": 0.8, "malacca_strait": 0.7, "transpacific_cables": 0.6},
    "KOR": {"strait_of_hormuz": 0.7, "malacca_strait": 0.6, "transpacific_cables": 0.5},
    "CHN": {"malacca_strait": 0.8, "strait_of_hormuz": 0.5, "transpacific_cables": 0.6, "shanghai_port": 0.9},
    "DEU": {"suez_canal": 0.5, "bosphorus_strait": 0.3, "rotterdam_port": 0.7, "transatlantic_cables": 0.5},
    "GBR": {"suez_canal": 0.4, "transatlantic_cables": 0.7, "rotterdam_port": 0.3},
    "FRA": {"suez_canal": 0.4, "med_cables": 0.5, "transatlantic_cables": 0.4},
    "USA": {"panama_canal": 0.4, "transatlantic_cables": 0.5, "transpacific_cables": 0.5, "houston_port": 0.3},
    "IND": {"strait_of_hormuz": 0.6, "malacca_strait": 0.3, "asia_africa_cables": 0.5},
    "SGP": {"malacca_strait": 0.9, "singapore_port": 0.95, "transpacific_cables": 0.4},
    "ARE": {"strait_of_hormuz": 0.9, "dubai_port": 0.8, "asia_africa_cables": 0.4},
    "SAU": {"strait_of_hormuz": 0.7, "suez_canal": 0.4, "asia_africa_cables": 0.3},
    "TUR": {"bosphorus_strait": 0.8, "med_cables": 0.4, "suez_canal": 0.3},
    "EGY": {"suez_canal": 0.9, "med_cables": 0.5, "asia_africa_cables": 0.4},
    "NLD": {"rotterdam_port": 0.9, "suez_canal": 0.4, "transatlantic_cables": 0.5},
    "AUS": {"transpacific_cables": 0.6, "malacca_strait": 0.3},
    "BRA": {"panama_canal": 0.3, "transatlantic_cables": 0.4},
}

# Map outage country names to relevant infrastructure nodes for cascade seeding
_COUNTRY_TO_INFRA_NODES: dict[str, list[str]] = {
    "EG": ["suez_canal", "med_cables"],
    "TR": ["bosphorus_strait", "med_cables"],
    "SG": ["malacca_strait", "singapore_port"],
    "PA": ["panama_canal"],
    "AE": ["strait_of_hormuz", "dubai_port"],
    "OM": ["strait_of_hormuz"],
    "IR": ["strait_of_hormuz"],
    "NL": ["rotterdam_port"],
    "CN": ["shanghai_port", "transpacific_cables"],
}


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
            return []

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
            return []

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

            event = InfrastructureEvent(
                event_type="internet_outage",
                country=country,
                severity=severity,
                started_at=started_at,
                description=description,
                source="cloudflare_radar",
                affected_services=["internet"],
                cascade_risk_score=0.0,  # Computed separately
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
        if disrupted_node not in _INFRA_NODES:
            return {}

        impacts: dict[str, float] = {disrupted_node: 1.0}
        frontier: deque[tuple[str, float, int]] = deque()
        frontier.append((disrupted_node, 1.0, 0))

        while frontier:
            node, current_impact, depth = frontier.popleft()
            if depth >= _MAX_CASCADE_HOPS:
                continue

            for neighbor, weight in _ADJACENCY.get(node, []):
                redundancy = _REDUNDANCY.get(neighbor, 0.0)
                # Impact = upstream_impact * edge_weight * (1 - redundancy)
                propagated = current_impact * weight * (1.0 - redundancy)
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

        for event in events:
            # Find infrastructure nodes related to this outage's country
            country_code = event.country.upper()[:2]
            related_nodes = _COUNTRY_TO_INFRA_NODES.get(country_code, [])

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
        risks: list[dict] = []

        for event in self._cached_outages:
            country_code = event.country.upper()[:2]
            related_nodes = _COUNTRY_TO_INFRA_NODES.get(country_code, [])

            for node in related_nodes:
                cascade = self.compute_cascade_impact(node)
                if not cascade:
                    continue

                # Map cascade impacts to affected countries
                affected: list[str] = []
                for country_iso3, deps in TRADE_DEPENDENCIES.items():
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
        country_code = event.country.upper()[:2]
        related_nodes = _COUNTRY_TO_INFRA_NODES.get(country_code, [])

        affected: set[str] = set()
        for country_iso3, deps in TRADE_DEPENDENCIES.items():
            for node in related_nodes:
                if node in deps and deps[node] >= 0.3:
                    affected.add(country_iso3)

        return sorted(affected)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

infrastructure_monitor = InfrastructureMonitor()
