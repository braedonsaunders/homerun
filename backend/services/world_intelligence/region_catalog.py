"""Shared world-intelligence region catalog.

Single source of truth for map overlays and regional monitors.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_REGION_FILE = Path(__file__).resolve().parents[2] / "data" / "world_intelligence" / "regions.json"


@dataclass(frozen=True)
class Hotspot:
    id: str
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def to_bbox(self) -> tuple[float, float, float, float]:
        return (self.lat_min, self.lat_max, self.lon_min, self.lon_max)


@dataclass(frozen=True)
class Chokepoint:
    id: str
    name: str
    latitude: float
    longitude: float


class RegionCatalog:
    def __init__(self) -> None:
        self._loaded = False
        self._version = 0
        self._updated_at: str | None = None
        self._hotspots: list[Hotspot] = []
        self._chokepoints: list[Chokepoint] = []

    def _load(self) -> None:
        if self._loaded:
            return

        try:
            payload = json.loads(_REGION_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Failed to load region catalog %s: %s", _REGION_FILE, exc)
            self._loaded = True
            return

        self._version = int(payload.get("version", 0) or 0)
        self._updated_at = payload.get("updated_at")

        hotspots: list[Hotspot] = []
        for row in payload.get("hotspots", []) or []:
            try:
                hotspots.append(
                    Hotspot(
                        id=str(row["id"]),
                        name=str(row.get("name") or row["id"]),
                        lat_min=float(row["lat_min"]),
                        lat_max=float(row["lat_max"]),
                        lon_min=float(row["lon_min"]),
                        lon_max=float(row["lon_max"]),
                    )
                )
            except Exception as exc:
                logger.warning("Skipping invalid hotspot row %s: %s", row, exc)

        chokepoints: list[Chokepoint] = []
        for row in payload.get("chokepoints", []) or []:
            try:
                chokepoints.append(
                    Chokepoint(
                        id=str(row["id"]),
                        name=str(row.get("name") or row["id"]),
                        latitude=float(row["latitude"]),
                        longitude=float(row["longitude"]),
                    )
                )
            except Exception as exc:
                logger.warning("Skipping invalid chokepoint row %s: %s", row, exc)

        self._hotspots = hotspots
        self._chokepoints = chokepoints
        self._loaded = True

    def hotspots(self) -> list[Hotspot]:
        self._load()
        return list(self._hotspots)

    def chokepoints(self) -> list[Chokepoint]:
        self._load()
        return list(self._chokepoints)

    def hotspot_bboxes(self) -> dict[str, tuple[float, float, float, float]]:
        self._load()
        return {h.id: h.to_bbox() for h in self._hotspots}

    def payload(self) -> dict[str, Any]:
        self._load()
        return {
            "version": self._version,
            "updated_at": self._updated_at,
            "hotspots": [
                {
                    "id": h.id,
                    "name": h.name,
                    "lat_min": h.lat_min,
                    "lat_max": h.lat_max,
                    "lon_min": h.lon_min,
                    "lon_max": h.lon_max,
                }
                for h in self._hotspots
            ],
            "chokepoints": [
                {
                    "id": c.id,
                    "name": c.name,
                    "latitude": c.latitude,
                    "longitude": c.longitude,
                }
                for c in self._chokepoints
            ],
        }


region_catalog = RegionCatalog()
