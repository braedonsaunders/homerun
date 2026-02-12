"""Maintained maritime chokepoint feed via IMF PortWatch ArcGIS."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx

from config import settings

logger = logging.getLogger(__name__)

_PORTWATCH_POINTS_QUERY_URL = str(
    getattr(
        settings,
        "WORLD_INTEL_CHOKEPOINTS_PORTWATCH_POINTS_URL",
        (
            "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/"
            "PortWatch_chokepoints_database/FeatureServer/0/query"
        ),
    )
    or (
        "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/"
        "PortWatch_chokepoints_database/FeatureServer/0/query"
    )
)
_PORTWATCH_DAILY_QUERY_URL = str(
    getattr(
        settings,
        "WORLD_INTEL_CHOKEPOINTS_PORTWATCH_DAILY_URL",
        (
            "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/"
            "Daily_Chokepoints_Data/FeatureServer/0/query"
        ),
    )
    or (
        "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/"
        "Daily_Chokepoints_Data/FeatureServer/0/query"
    )
)
_CHOKEPOINTS_ENABLED = bool(getattr(settings, "WORLD_INTEL_CHOKEPOINTS_ENABLED", True))
_CHOKEPOINTS_REFRESH_SECONDS = int(
    max(60, getattr(settings, "WORLD_INTEL_CHOKEPOINTS_REFRESH_SECONDS", 1800) or 1800)
)
_CHOKEPOINTS_REQUEST_TIMEOUT_SECONDS = float(
    max(
        5.0,
        getattr(settings, "WORLD_INTEL_CHOKEPOINTS_REQUEST_TIMEOUT_SECONDS", 20.0)
        or 20.0,
    )
)
_CHOKEPOINTS_MAX_DAILY_ROWS = int(
    max(100, getattr(settings, "WORLD_INTEL_CHOKEPOINTS_MAX_DAILY_ROWS", 500) or 500)
)

_STATIC_REGIONS_FILE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "world_intelligence"
    / "regions.json"
)


def _to_iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.replace(tzinfo=None).isoformat() + "Z"


def _to_utc_datetime_from_epoch_ms(value: Any) -> Optional[datetime]:
    try:
        raw = float(value)
    except Exception:
        return None
    if raw <= 0:
        return None
    try:
        return datetime.fromtimestamp(raw / 1000.0, tz=timezone.utc)
    except Exception:
        return None


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(float(value))
    except Exception:
        return None


class ChokepointFeed:
    """Fetches and caches maritime chokepoints from maintained external feeds."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=_CHOKEPOINTS_REQUEST_TIMEOUT_SECONDS)
        self._refresh_lock: Optional[asyncio.Lock] = None
        self._cache: list[dict[str, Any]] = []
        self._cache_refreshed_at: float = 0.0
        self._last_updated_at: Optional[datetime] = None
        self._last_error: Optional[str] = None
        self._last_source: str = "none"
        self._last_duration_seconds: float = 0.0
        self._static_fallback_cache: list[dict[str, Any]] = []

    def _cache_fresh(self) -> bool:
        if not self._cache:
            return False
        return (time.monotonic() - self._cache_refreshed_at) < _CHOKEPOINTS_REFRESH_SECONDS

    def _load_static_fallback(self) -> list[dict[str, Any]]:
        if self._static_fallback_cache:
            return list(self._static_fallback_cache)
        try:
            payload = json.loads(_STATIC_REGIONS_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Failed to load static chokepoint fallback %s: %s", _STATIC_REGIONS_FILE, exc)
            self._static_fallback_cache = []
            return []

        fallback: list[dict[str, Any]] = []
        for row in payload.get("chokepoints", []) or []:
            lat = _coerce_float(row.get("latitude"))
            lon = _coerce_float(row.get("longitude"))
            if lat is None or lon is None:
                continue
            fallback.append(
                {
                    "id": str(row.get("id") or "").strip() or str(row.get("name") or "unknown"),
                    "name": str(row.get("name") or row.get("id") or "Chokepoint").strip(),
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "source": "static_catalog",
                    "last_updated": payload.get("updated_at"),
                }
            )
        self._static_fallback_cache = fallback
        return list(self._static_fallback_cache)

    async def _fetch_portwatch_points(self) -> list[dict[str, Any]]:
        params = {
            "where": "1=1",
            "outFields": (
                "portid,portname,fullname,lat,lon,"
                "vessel_count_total,vessel_count_container,vessel_count_dry_bulk,"
                "vessel_count_general_cargo,vessel_count_RoRo,vessel_count_tanker,"
                "industry_top1,industry_top2,industry_top3,pageid"
            ),
            "f": "json",
        }
        resp = await self._client.get(_PORTWATCH_POINTS_QUERY_URL, params=params)
        resp.raise_for_status()
        payload = resp.json()
        features = payload.get("features", []) if isinstance(payload, dict) else []
        if not isinstance(features, list):
            return []

        rows: list[dict[str, Any]] = []
        for feature in features:
            attrs = feature.get("attributes") if isinstance(feature, dict) else {}
            if not isinstance(attrs, dict):
                continue
            portid = str(attrs.get("portid") or "").strip()
            if not portid:
                continue
            lat = _coerce_float(attrs.get("lat"))
            lon = _coerce_float(attrs.get("lon"))
            if lat is None or lon is None:
                continue
            rows.append(
                {
                    "id": portid,
                    "portid": portid,
                    "name": str(attrs.get("portname") or attrs.get("fullname") or portid).strip(),
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "source": "imf_portwatch",
                    "baseline_vessel_count_total": _coerce_int(attrs.get("vessel_count_total")),
                    "baseline_vessel_count_container": _coerce_int(attrs.get("vessel_count_container")),
                    "baseline_vessel_count_dry_bulk": _coerce_int(attrs.get("vessel_count_dry_bulk")),
                    "baseline_vessel_count_general_cargo": _coerce_int(
                        attrs.get("vessel_count_general_cargo")
                    ),
                    "baseline_vessel_count_roro": _coerce_int(attrs.get("vessel_count_RoRo")),
                    "baseline_vessel_count_tanker": _coerce_int(attrs.get("vessel_count_tanker")),
                    "industry_top1": str(attrs.get("industry_top1") or "").strip() or None,
                    "industry_top2": str(attrs.get("industry_top2") or "").strip() or None,
                    "industry_top3": str(attrs.get("industry_top3") or "").strip() or None,
                    "reference_page_id": str(attrs.get("pageid") or "").strip() or None,
                }
            )
        return rows

    async def _fetch_portwatch_daily_metrics(
        self,
    ) -> tuple[dict[str, dict[str, Any]], Optional[datetime]]:
        params = {
            "where": "1=1",
            "outFields": (
                "portid,portname,date,year,month,day,"
                "n_total,n_cargo,n_container,n_tanker,n_dry_bulk,n_general_cargo,n_roro,capacity"
            ),
            "orderByFields": "date DESC",
            "resultRecordCount": str(_CHOKEPOINTS_MAX_DAILY_ROWS),
            "f": "json",
        }
        resp = await self._client.get(_PORTWATCH_DAILY_QUERY_URL, params=params)
        resp.raise_for_status()
        payload = resp.json()
        features = payload.get("features", []) if isinstance(payload, dict) else []
        if not isinstance(features, list):
            return {}, None

        latest_ms: Optional[int] = None
        for feature in features:
            attrs = feature.get("attributes") if isinstance(feature, dict) else {}
            if not isinstance(attrs, dict):
                continue
            value = _coerce_int(attrs.get("date"))
            if value is None:
                continue
            if latest_ms is None or value > latest_ms:
                latest_ms = value
        latest_dt = _to_utc_datetime_from_epoch_ms(latest_ms)

        by_portid: dict[str, dict[str, Any]] = {}
        for feature in features:
            attrs = feature.get("attributes") if isinstance(feature, dict) else {}
            if not isinstance(attrs, dict):
                continue
            portid = str(attrs.get("portid") or "").strip()
            if not portid or portid in by_portid:
                continue
            row_date = _coerce_int(attrs.get("date"))
            if latest_ms is not None and row_date is not None and row_date != latest_ms:
                continue
            by_portid[portid] = {
                "daily_transit_total": _coerce_int(attrs.get("n_total")),
                "daily_transit_cargo": _coerce_int(attrs.get("n_cargo")),
                "daily_transit_container": _coerce_int(attrs.get("n_container")),
                "daily_transit_tanker": _coerce_int(attrs.get("n_tanker")),
                "daily_transit_dry_bulk": _coerce_int(attrs.get("n_dry_bulk")),
                "daily_transit_general_cargo": _coerce_int(attrs.get("n_general_cargo")),
                "daily_transit_roro": _coerce_int(attrs.get("n_roro")),
                "daily_capacity_estimate": _coerce_int(attrs.get("capacity")),
                "daily_metrics_date": _to_iso(_to_utc_datetime_from_epoch_ms(attrs.get("date"))),
            }
        return by_portid, latest_dt

    async def refresh(self, force: bool = False) -> list[dict[str, Any]]:
        if not force and self._cache_fresh():
            self._last_source = "cache"
            return list(self._cache)

        if self._refresh_lock is None:
            self._refresh_lock = asyncio.Lock()
        async with self._refresh_lock:
            if not force and self._cache_fresh():
                self._last_source = "cache"
                return list(self._cache)

            started = time.monotonic()
            if not _CHOKEPOINTS_ENABLED:
                fallback = self._load_static_fallback()
                self._cache = list(fallback)
                self._cache_refreshed_at = time.monotonic()
                self._last_updated_at = datetime.now(timezone.utc)
                self._last_error = "disabled"
                self._last_source = "static_disabled"
                self._last_duration_seconds = round(
                    time.monotonic() - started,
                    3,
                )
                return list(self._cache)

            try:
                points_task = self._fetch_portwatch_points()
                metrics_task = self._fetch_portwatch_daily_metrics()
                points, (daily_metrics, daily_latest_dt) = await asyncio.gather(
                    points_task,
                    metrics_task,
                )
                if not points:
                    raise RuntimeError("no chokepoint rows from portwatch")

                merged: list[dict[str, Any]] = []
                for point in points:
                    metrics = daily_metrics.get(str(point.get("portid") or ""), {})
                    item = dict(point)
                    item.update(metrics)
                    item["daily_dataset_updated_at"] = _to_iso(daily_latest_dt)
                    item["last_updated"] = _to_iso(datetime.now(timezone.utc))
                    merged.append(item)
                merged.sort(
                    key=lambda item: (
                        int(item.get("daily_transit_total") or 0),
                        int(item.get("baseline_vessel_count_total") or 0),
                    ),
                    reverse=True,
                )

                self._cache = merged
                self._cache_refreshed_at = time.monotonic()
                self._last_updated_at = datetime.now(timezone.utc)
                self._last_error = None
                self._last_source = "imf_portwatch"
                self._last_duration_seconds = round(
                    time.monotonic() - started,
                    3,
                )
                return list(self._cache)
            except Exception as exc:
                self._last_error = str(exc)
                self._last_duration_seconds = round(
                    time.monotonic() - started,
                    3,
                )
                logger.warning("PortWatch chokepoint refresh failed: %s", exc)
                if self._cache:
                    self._last_source = "cache_stale"
                    self._cache_refreshed_at = time.monotonic()
                    self._last_updated_at = datetime.now(timezone.utc)
                    return list(self._cache)
                fallback = self._load_static_fallback()
                self._cache = list(fallback)
                self._cache_refreshed_at = time.monotonic()
                self._last_updated_at = datetime.now(timezone.utc)
                self._last_source = "static_fallback"
                return list(self._cache)

    async def get_chokepoints(
        self,
        fallback: Optional[list[dict[str, Any]]] = None,
    ) -> list[dict[str, Any]]:
        data = await self.refresh(force=False)
        if data:
            return data
        if isinstance(fallback, list) and fallback:
            return list(fallback)
        return self._load_static_fallback()

    def get_health(self) -> dict[str, Any]:
        has_data = bool(self._cache)
        return {
            "ok": has_data and (self._last_source != "none"),
            "count": len(self._cache),
            "enabled": _CHOKEPOINTS_ENABLED,
            "source": self._last_source,
            "last_error": self._last_error,
            "last_updated": _to_iso(self._last_updated_at),
            "cache_ttl_seconds": _CHOKEPOINTS_REFRESH_SECONDS,
            "request_timeout_seconds": _CHOKEPOINTS_REQUEST_TIMEOUT_SECONDS,
            "max_daily_rows": _CHOKEPOINTS_MAX_DAILY_ROWS,
            "duration_seconds": self._last_duration_seconds,
        }


chokepoint_feed = ChokepointFeed()
