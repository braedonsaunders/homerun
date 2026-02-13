"""World Intelligence API routes (DB-backed)."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from models.database import (
    CountryInstabilityRecord,
    TensionPairRecord,
    WorldIntelligenceSignal,
    WorldIntelligenceSnapshot,
    get_db_session,
)
from services.worker_state import read_worker_snapshot
from services.world_intelligence.country_catalog import country_catalog
from services.world_intelligence.chokepoint_feed import chokepoint_feed
from services.world_intelligence.region_catalog import region_catalog
from services.world_intelligence.resolver import resolve_world_signal_opportunities
from services.world_intelligence.country_reference_source import (
    get_country_reference_source_status,
    sync_country_reference_from_world_bank,
)
from services.world_intelligence.ucdp_conflict_source import (
    get_ucdp_conflict_source_status,
    sync_ucdp_conflict_lists,
)
from services.world_intelligence.mid_reference_source import (
    get_mid_reference_source_status,
    sync_mid_reference_from_itu,
)
from services.world_intelligence.trade_dependency_source import (
    get_trade_dependency_source_status,
    sync_trade_dependencies_from_world_bank,
)
from services.world_intelligence.gdelt_news_source import (
    get_gdelt_news_source_status,
    sync_gdelt_news_from_source,
)
from services.world_intelligence.chokepoint_reference_source import (
    get_chokepoint_reference_source_status,
    sync_chokepoint_reference_from_portwatch,
)

router = APIRouter(tags=["world-intelligence"])
logger = logging.getLogger(__name__)

_BENIGN_SOURCE_ERROR_MARKERS = (
    "credentials_missing",
    "missing_api_key",
    "missing_api_token",
    "disabled",
    "rate-limited",
    "rate limited",
    "http 429",
    "client error '429",
    "status code 429",
    "client error '403",
    "status code 403",
    "403 forbidden",
    "soft rate-limit",
    "soft rate-limited",
    "please limit requests to one every 5 seconds",
    "nodename nor servname provided",
    "name or service not known",
    "temporary failure in name resolution",
)


def _to_iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.replace(tzinfo=None).isoformat() + "Z"


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    try:
        if text.endswith("Z"):
            text = text[:-1]
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        return None


def _is_benign_source_error(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    return any(marker in text for marker in _BENIGN_SOURCE_ERROR_MARKERS)


def _normalize_source_health_entry(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    payload = dict(value)
    if bool(payload.get("ok", True)):
        return payload
    raw_error = str(payload.get("error") or payload.get("last_error") or "").strip()
    if raw_error and _is_benign_source_error(raw_error):
        payload["degraded"] = True
    return payload


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers."""
    try:
        lat1_r = math.radians(float(lat1))
        lon1_r = math.radians(float(lon1))
        lat2_r = math.radians(float(lat2))
        lon2_r = math.radians(float(lon2))
    except Exception:
        return float("inf")

    d_lat = lat2_r - lat1_r
    d_lon = lon2_r - lon1_r
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(d_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1 - a)))
    earth_radius_km = 6371.0
    return earth_radius_km * c


def _severity_label(severity: float) -> str:
    if severity >= 0.8:
        return "critical"
    if severity >= 0.6:
        return "high"
    if severity >= 0.3:
        return "medium"
    return "normal"


def _normalize_country_value(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized = country_catalog.normalize_iso3(text)
    if normalized:
        return normalized
    return text.upper()


def _instability_country_key(row: Any) -> str:
    iso3 = _normalize_country_value(getattr(row, "iso3", ""))
    if iso3:
        return iso3
    return _normalize_country_value(getattr(row, "country", ""))


def _normalized_tension_pair_key(country_a: Any, country_b: Any) -> str:
    a = _normalize_country_value(country_a)
    b = _normalize_country_value(country_b)
    if not a or not b or a == b:
        return ""
    return "-".join(sorted((a, b)))


def _snapshot_timestamp(row: Any) -> datetime:
    value = getattr(row, "computed_at", None)
    if not isinstance(value, datetime):
        return datetime.min
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _is_distinct_previous_snapshot(current: Any, candidate: Any, score_attr: str) -> bool:
    current_ts = _snapshot_timestamp(current)
    candidate_ts = _snapshot_timestamp(candidate)
    if candidate_ts < current_ts:
        return True
    if candidate_ts > current_ts:
        return False
    try:
        return float(getattr(candidate, score_attr) or 0.0) != float(
            getattr(current, score_attr) or 0.0
        )
    except Exception:
        return True


def _military_entity_key(signal_row: dict[str, Any]) -> str:
    if str(signal_row.get("signal_type") or "") != "military":
        return ""
    metadata = signal_row.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    activity_type = str(metadata.get("activity_type") or "").strip().lower() or "flight"
    transponder = str(metadata.get("transponder") or "").strip().lower()
    if transponder:
        return f"{activity_type}:{transponder}"
    callsign = "".join(str(metadata.get("callsign") or "").strip().upper().split())
    country = (
        str(signal_row.get("country_iso3") or signal_row.get("country") or "")
        .strip()
        .upper()
    )
    if callsign:
        return f"{activity_type}:{callsign}:{country}"
    signal_id = str(signal_row.get("signal_id") or "").strip()
    return f"{activity_type}:{signal_id}" if signal_id else ""


def _dedupe_military_signals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen_entities: set[str] = set()
    for row in rows:
        key = _military_entity_key(row)
        if key:
            if key in seen_entities:
                continue
            seen_entities.add(key)
        out.append(row)
    return out


def _signal_row_to_dict(row: WorldIntelligenceSignal) -> dict[str, Any]:
    country_value = str(row.country or "").strip() or None
    country_iso3 = country_catalog.normalize_iso3(str(row.iso3 or row.country or ""))
    country_name = country_catalog.country_name(country_iso3 or country_value or "")
    return {
        "signal_id": row.id,
        "signal_type": row.signal_type,
        "severity": round(float(row.severity or 0.0), 3),
        "country": country_value,
        "country_iso3": country_iso3 or None,
        "country_name": country_name or None,
        "latitude": row.latitude,
        "longitude": row.longitude,
        "title": row.title,
        "description": row.description or "",
        "source": row.source or "unknown",
        "detected_at": _to_iso(row.detected_at),
        "metadata": row.metadata_json or {},
        "related_market_ids": list(row.related_market_ids or []),
        "market_relevance_score": (
            round(float(row.market_relevance_score), 3)
            if row.market_relevance_score is not None
            else None
        ),
    }


async def _get_world_snapshot(session: AsyncSession) -> Optional[WorldIntelligenceSnapshot]:
    result = await session.execute(
        select(WorldIntelligenceSnapshot).where(WorldIntelligenceSnapshot.id == "latest")
    )
    return result.scalar_one_or_none()


async def _latest_instability_by_country(
    session: AsyncSession,
) -> dict[str, tuple[Any, Optional[Any]]]:
    rows = (
        (
            await session.execute(
                select(CountryInstabilityRecord)
                .order_by(CountryInstabilityRecord.computed_at.desc())
                .limit(5000)
            )
        )
        .scalars()
        .all()
    )
    latest: dict[str, tuple[Any, Optional[Any]]] = {}
    for row in rows:
        key = _instability_country_key(row)
        if not key:
            continue
        if key not in latest:
            latest[key] = (row, None)
            continue
        current, prev = latest[key]
        if prev is None and _is_distinct_previous_snapshot(current, row, "score"):
            latest[key] = (current, row)
    if latest:
        return latest

    # Fallback: derive scores from recent instability signals when score history
    # hasn't been persisted yet.
    cutoff = datetime.now(timezone.utc) - timedelta(days=3)
    signal_rows = (
        (
            await session.execute(
                select(WorldIntelligenceSignal)
                .where(WorldIntelligenceSignal.signal_type == "instability")
                .where(WorldIntelligenceSignal.detected_at >= cutoff)
                .order_by(WorldIntelligenceSignal.detected_at.desc())
                .limit(2000)
            )
        )
        .scalars()
        .all()
    )
    for row in signal_rows:
        meta = row.metadata_json if isinstance(row.metadata_json, dict) else {}
        iso3 = _normalize_country_value(str(row.iso3 or row.country or ""))
        if len(iso3) != 3:
            continue
        components = meta.get("components")
        if not isinstance(components, dict):
            components = {}
        contributing = meta.get("contributing_signals")
        if "contributing_signals" not in components and isinstance(contributing, list):
            components["contributing_signals"] = contributing
        fallback_row = SimpleNamespace(
            country=str(row.country or iso3),
            iso3=iso3,
            score=round(float(row.severity or 0.0) * 100.0, 1),
            trend=str(meta.get("trend") or "stable"),
            components=components,
            computed_at=row.detected_at,
        )
        if iso3 not in latest:
            latest[iso3] = (fallback_row, None)
            continue
        current, prev = latest[iso3]
        if prev is None and _is_distinct_previous_snapshot(current, fallback_row, "score"):
            latest[iso3] = (current, fallback_row)
    return latest


async def _latest_tension_pairs(
    session: AsyncSession,
) -> dict[str, tuple[Any, Optional[Any]]]:
    rows = (
        (
            await session.execute(
                select(TensionPairRecord)
                .order_by(TensionPairRecord.computed_at.desc())
                .limit(5000)
            )
        )
        .scalars()
        .all()
    )
    latest: dict[str, tuple[Any, Optional[Any]]] = {}
    for row in rows:
        key = _normalized_tension_pair_key(row.country_a, row.country_b)
        if not key:
            continue
        if key not in latest:
            latest[key] = (row, None)
            continue
        current, prev = latest[key]
        if prev is None and _is_distinct_previous_snapshot(current, row, "tension_score"):
            latest[key] = (current, row)
    if latest:
        return latest

    # Fallback: derive pair scores from recent tension signals if pair history is empty.
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    signal_rows = (
        (
            await session.execute(
                select(WorldIntelligenceSignal)
                .where(WorldIntelligenceSignal.signal_type == "tension")
                .where(WorldIntelligenceSignal.detected_at >= cutoff)
                .order_by(WorldIntelligenceSignal.detected_at.desc())
                .limit(3000)
            )
        )
        .scalars()
        .all()
    )
    for row in signal_rows:
        meta = row.metadata_json if isinstance(row.metadata_json, dict) else {}
        a = str(meta.get("country_a") or "").strip().upper()
        b = str(meta.get("country_b") or "").strip().upper()
        if not a or not b:
            raw_pair = str(row.country or "").strip().upper()
            if "-" in raw_pair:
                parts = [part.strip() for part in raw_pair.split("-", 1)]
                if len(parts) == 2:
                    a, b = parts[0], parts[1]
        if not a or not b:
            continue
        score = meta.get("tension_score")
        try:
            tension_score = (
                float(score)
                if score is not None
                else float(row.severity or 0.0) * 100.0
            )
        except Exception:
            tension_score = float(row.severity or 0.0) * 100.0
        avg_goldstein = meta.get("avg_goldstein_scale")
        try:
            avg_goldstein_scale = float(avg_goldstein) if avg_goldstein is not None else None
        except Exception:
            avg_goldstein_scale = None
        event_count_raw = meta.get("event_count")
        try:
            event_count = int(event_count_raw or 0)
        except Exception:
            event_count = 0
        fallback_row = SimpleNamespace(
            country_a=a,
            country_b=b,
            tension_score=round(tension_score, 1),
            event_count=event_count,
            avg_goldstein_scale=avg_goldstein_scale,
            trend=str(meta.get("trend") or "stable"),
            computed_at=row.detected_at,
        )
        key = _normalized_tension_pair_key(a, b)
        if not key:
            continue
        if key not in latest:
            latest[key] = (fallback_row, None)
            continue
        current, prev = latest[key]
        if prev is None and _is_distinct_previous_snapshot(current, fallback_row, "tension_score"):
            latest[key] = (current, fallback_row)
    return latest


async def _latest_tension_signal_meta(
    session: AsyncSession,
) -> dict[str, dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    rows = (
        (
            await session.execute(
                select(WorldIntelligenceSignal)
                .where(WorldIntelligenceSignal.signal_type == "tension")
                .where(WorldIntelligenceSignal.detected_at >= cutoff)
                .order_by(WorldIntelligenceSignal.detected_at.desc())
                .limit(2000)
            )
        )
        .scalars()
        .all()
    )
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        meta = row.metadata_json or {}
        if not isinstance(meta, dict):
            continue
        key = _normalized_tension_pair_key(meta.get("country_a"), meta.get("country_b"))
        if not key:
            continue
        if key not in latest:
            latest[key] = meta
    return latest


async def _dynamic_military_hotspots(
    session: AsyncSession,
    max_hotspots: int = 16,
    min_events: int = 2,
) -> list[dict[str, Any]]:
    """Build map hotspots from recent real military signal coordinates."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    rows = (
        (
            await session.execute(
                select(WorldIntelligenceSignal)
                .where(WorldIntelligenceSignal.signal_type == "military")
                .where(WorldIntelligenceSignal.detected_at >= cutoff)
                .where(WorldIntelligenceSignal.latitude.is_not(None))
                .where(WorldIntelligenceSignal.longitude.is_not(None))
                .order_by(WorldIntelligenceSignal.detected_at.desc())
                .limit(5000)
            )
        )
        .scalars()
        .all()
    )
    if not rows:
        return []

    # Spatial clustering on a fixed-size geographic grid.
    # Keeps hotspot derivation stable while remaining purely data-driven.
    grid_size_deg = 4.0
    clusters: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        try:
            lat = float(row.latitude)
            lon = float(row.longitude)
        except Exception:
            continue
        lat_bucket = int(math.floor(lat / grid_size_deg))
        lon_bucket = int(math.floor(lon / grid_size_deg))
        key = (lat_bucket, lon_bucket)
        cluster = clusters.setdefault(
            key,
            {
                "count": 0,
                "min_lat": lat,
                "max_lat": lat,
                "min_lon": lon,
                "max_lon": lon,
                "latest_at": row.detected_at,
                "countries": {},
                "activity_types": {},
            },
        )
        cluster["count"] += 1
        cluster["min_lat"] = min(cluster["min_lat"], lat)
        cluster["max_lat"] = max(cluster["max_lat"], lat)
        cluster["min_lon"] = min(cluster["min_lon"], lon)
        cluster["max_lon"] = max(cluster["max_lon"], lon)
        if isinstance(row.detected_at, datetime) and (
            cluster["latest_at"] is None or row.detected_at > cluster["latest_at"]
        ):
            cluster["latest_at"] = row.detected_at

        country = str(row.country or row.iso3 or "").strip().upper()
        if country:
            cluster["countries"][country] = cluster["countries"].get(country, 0) + 1

        meta = row.metadata_json if isinstance(row.metadata_json, dict) else {}
        activity_type = str(meta.get("activity_type") or "unknown").strip().lower()
        cluster["activity_types"][activity_type] = cluster["activity_types"].get(activity_type, 0) + 1

    ranked = sorted(
        (
            item for item in clusters.items()
            if int(item[1].get("count", 0)) >= int(min_events)
        ),
        key=lambda item: (
            int(item[1].get("count", 0)),
            (
                float(item[1]["latest_at"].timestamp())
                if isinstance(item[1].get("latest_at"), datetime)
                else 0.0
            ),
        ),
        reverse=True,
    )[: max(1, int(max_hotspots))]

    hotspots: list[dict[str, Any]] = []
    for (lat_bucket, lon_bucket), data in ranked:
        event_count = int(data.get("count", 0))
        min_lat = float(data.get("min_lat", 0.0))
        max_lat = float(data.get("max_lat", 0.0))
        min_lon = float(data.get("min_lon", 0.0))
        max_lon = float(data.get("max_lon", 0.0))
        pad = 0.8
        lat_min = max(-90.0, min(min_lat, max_lat) - pad)
        lat_max = min(90.0, max(min_lat, max_lat) + pad)
        lon_min = max(-180.0, min(min_lon, max_lon) - pad)
        lon_max = min(180.0, max(min_lon, max_lon) + pad)

        countries = data.get("countries") or {}
        top_country = max(countries, key=countries.get) if countries else ""
        top_activity = ""
        activity_types = data.get("activity_types") or {}
        if activity_types:
            top_activity = max(activity_types, key=activity_types.get)
        name_parts = [part for part in [top_country, top_activity.replace("_", " ").title()] if part]
        name = " / ".join(name_parts) if name_parts else "Military Activity Cluster"

        hotspots.append(
            {
                "id": f"mil_{lat_bucket}_{lon_bucket}",
                "name": name,
                "lat_min": round(lat_min, 3),
                "lat_max": round(lat_max, 3),
                "lon_min": round(lon_min, 3),
                "lon_max": round(lon_max, 3),
                "event_count": event_count,
                "last_detected_at": _to_iso(data.get("latest_at")),
                "activity_types": sorted(activity_types, key=activity_types.get, reverse=True),
            }
        )
    return hotspots


async def _dynamic_chokepoint_scores(
    session: AsyncSession,
    chokepoints: list[dict[str, Any]],
    radius_km: float = 420.0,
) -> list[dict[str, Any]]:
    """Attach live risk scores to catalog chokepoints using nearby recent signals."""
    if not chokepoints:
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=72)
    rows = (
        (
            await session.execute(
                select(WorldIntelligenceSignal)
                .where(WorldIntelligenceSignal.detected_at >= cutoff)
                .where(WorldIntelligenceSignal.latitude.is_not(None))
                .where(WorldIntelligenceSignal.longitude.is_not(None))
                .order_by(WorldIntelligenceSignal.detected_at.desc())
                .limit(6000)
            )
        )
        .scalars()
        .all()
    )
    if not rows:
        out = []
        for cp in chokepoints:
            base_source = str(cp.get("source") or "catalog")
            enriched = dict(cp)
            enriched.update(
                {
                    "risk_score": 0.0,
                    "nearby_signal_count": 0,
                    "signal_breakdown": {},
                    "risk_method": "proximity_weighted_v1",
                    "source": f"{base_source}+live_signals",
                    "chokepoint_source": base_source,
                    "last_updated": str(cp.get("last_updated") or _to_iso(datetime.now(timezone.utc))),
                }
            )
            out.append(enriched)
        return out

    type_weight = {
        "military": 1.0,
        "infrastructure": 0.95,
        "conflict": 0.85,
        "tension": 0.75,
        "convergence": 0.65,
        "instability": 0.55,
        "anomaly": 0.45,
    }
    out: list[dict[str, Any]] = []
    for cp in chokepoints:
        try:
            cp_lat = float(cp.get("latitude"))
            cp_lon = float(cp.get("longitude"))
        except Exception:
            continue

        weighted_sum = 0.0
        nearby = 0
        breakdown: dict[str, int] = {}
        newest_ts: Optional[datetime] = None

        for row in rows:
            if row.latitude is None or row.longitude is None:
                continue
            distance = _haversine_km(cp_lat, cp_lon, float(row.latitude), float(row.longitude))
            if not math.isfinite(distance) or distance > radius_km:
                continue

            nearby += 1
            signal_type = str(row.signal_type or "unknown")
            breakdown[signal_type] = breakdown.get(signal_type, 0) + 1
            if isinstance(row.detected_at, datetime) and (
                newest_ts is None or row.detected_at > newest_ts
            ):
                newest_ts = row.detected_at

            proximity = max(0.0, 1.0 - (distance / radius_km))
            weight = float(type_weight.get(signal_type, 0.35))
            severity = max(0.0, min(1.0, float(row.severity or 0.0)))
            weighted_sum += severity * proximity * weight

        risk_score = round(min(100.0, weighted_sum * 22.0), 1)
        base_source = str(cp.get("source") or "catalog")
        enriched = dict(cp)
        enriched.update(
            {
                "risk_score": risk_score,
                "nearby_signal_count": nearby,
                "signal_breakdown": breakdown,
                "risk_method": "proximity_weighted_v1",
                "source": f"{base_source}+live_signals",
                "chokepoint_source": base_source,
                "last_updated": _to_iso(newest_ts) if newest_ts else _to_iso(datetime.now(timezone.utc)),
            }
        )
        out.append(enriched)

    out.sort(
        key=lambda item: (
            float(item.get("risk_score") or 0.0),
            int(item.get("nearby_signal_count") or 0),
        ),
        reverse=True,
    )
    return out


async def _instability_change_7d(
    session: AsyncSession,
    row: CountryInstabilityRecord,
) -> Optional[float]:
    if row.score is None:
        return None
    iso3 = (row.iso3 or "").upper()
    country = (row.country or "")
    if not iso3 and not country:
        return None
    current_at = row.computed_at or datetime.now(timezone.utc)
    lookback = current_at - timedelta(days=6, hours=12)
    query = (
        select(CountryInstabilityRecord)
        .where(CountryInstabilityRecord.computed_at <= lookback)
        .order_by(CountryInstabilityRecord.computed_at.desc())
        .limit(1)
    )
    if iso3:
        query = query.where(func.upper(func.coalesce(CountryInstabilityRecord.iso3, "")) == iso3)
    else:
        query = query.where(func.upper(func.coalesce(CountryInstabilityRecord.country, "")) == country.upper())
    prev = (await session.execute(query)).scalar_one_or_none()
    if prev is None or prev.score is None:
        return None
    try:
        return round(float(row.score) - float(prev.score), 1)
    except Exception:
        return None


@router.get("/world-intelligence/regions")
async def get_world_regions(session: AsyncSession = Depends(get_db_session)):
    """Get map overlays (dynamic hotspots + maintained chokepoints + live risk)."""
    static_regions = region_catalog.payload()
    hotspots = await _dynamic_military_hotspots(session)
    raw_chokepoints = await chokepoint_feed.get_chokepoints(
        fallback=list(static_regions.get("chokepoints") or []),
    )
    chokepoints = await _dynamic_chokepoint_scores(
        session,
        raw_chokepoints,
    )
    return {
        "version": int(static_regions.get("version", 0) or 0),
        "updated_at": _to_iso(datetime.now(timezone.utc)),
        "hotspots": hotspots,
        "chokepoints": chokepoints,
        "chokepoint_source_health": chokepoint_feed.get_health(),
    }


@router.get("/world-intelligence/signals")
async def get_world_signals(
    signal_type: Optional[str] = Query(None, description="Filter by signal type"),
    country: Optional[str] = Query(None, description="Filter by country/ISO3"),
    min_severity: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(250, ge=1, le=5000),
    offset: int = Query(0, ge=0, le=500000),
    session: AsyncSession = Depends(get_db_session),
):
    """Get current world intelligence signals from persisted DB state."""
    query = select(WorldIntelligenceSignal)
    signal_type_value = signal_type if isinstance(signal_type, str) else None
    country_value = country if isinstance(country, str) else None
    try:
        min_severity_value = float(min_severity)
    except Exception:
        min_severity_value = 0.0
    try:
        limit_value = max(1, min(5000, int(limit)))
    except Exception:
        limit_value = 250
    try:
        offset_value = max(0, min(500000, int(offset)))
    except Exception:
        offset_value = 0

    if signal_type_value:
        query = query.where(WorldIntelligenceSignal.signal_type == signal_type_value)
    if country_value:
        country_lower = country_value.strip().lower()
        query = query.where(
            or_(
                func.lower(func.coalesce(WorldIntelligenceSignal.country, "")) == country_lower,
                func.lower(func.coalesce(WorldIntelligenceSignal.iso3, "")) == country_lower,
            )
        )
    if min_severity_value > 0:
        query = query.where(WorldIntelligenceSignal.severity >= min_severity_value)

    fetch_limit = limit_value

    total = int(
        (
            await session.execute(
                select(func.count()).select_from(query.subquery())
            )
        ).scalar()
        or 0
    )

    rows = (
        (
            await session.execute(
                query.order_by(
                    WorldIntelligenceSignal.detected_at.desc(),
                    WorldIntelligenceSignal.severity.desc(),
                ).limit(fetch_limit).offset(offset_value)
            )
        )
        .scalars()
        .all()
    )

    snapshot = await _get_world_snapshot(session)

    scanned_count = len(rows)
    signal_payload = [_signal_row_to_dict(r) for r in rows]
    signal_payload = _dedupe_military_signals(signal_payload)
    if not signal_payload and snapshot and isinstance(snapshot.signals_json, list):
        fallback_rows: list[dict[str, Any]] = []
        country_filter = country_value.strip().lower() if country_value else None
        for raw in snapshot.signals_json:
            if not isinstance(raw, dict):
                continue
            row_signal_type = str(raw.get("signal_type") or "").strip()
            if signal_type_value and row_signal_type != signal_type_value:
                continue
            row_country = str(raw.get("country") or "").strip().lower()
            row_iso3 = str(raw.get("iso3") or "").strip().lower()
            if country_filter and country_filter not in {row_country, row_iso3}:
                continue
            try:
                row_severity = float(raw.get("severity") or 0.0)
            except Exception:
                row_severity = 0.0
            if row_severity < min_severity_value:
                continue
            fallback_rows.append(
                {
                    "signal_id": raw.get("signal_id") or raw.get("id"),
                    "signal_type": row_signal_type or "unknown",
                    "severity": round(row_severity, 3),
                    "country": raw.get("country"),
                    "country_iso3": country_catalog.normalize_iso3(str(raw.get("iso3") or raw.get("country") or "")) or None,
                    "country_name": country_catalog.country_name(str(raw.get("iso3") or raw.get("country") or "")) or None,
                    "latitude": raw.get("latitude"),
                    "longitude": raw.get("longitude"),
                    "title": raw.get("title") or "Signal",
                    "description": raw.get("description") or "",
                    "source": raw.get("source") or "unknown",
                    "detected_at": raw.get("detected_at"),
                    "metadata": raw.get("metadata") or {},
                    "related_market_ids": list(raw.get("related_market_ids") or []),
                    "market_relevance_score": raw.get("market_relevance_score"),
                }
            )
        fallback_rows.sort(
            key=lambda item: (
                str(item.get("detected_at") or ""),
                float(item.get("severity") or 0.0),
            ),
            reverse=True,
        )
        fallback_rows = _dedupe_military_signals(fallback_rows)
        total = len(fallback_rows)
        signal_payload = fallback_rows[offset_value: offset_value + limit_value]
        scanned_count = len(signal_payload)
    else:
        signal_payload = signal_payload[:limit_value]

    last_collection = None
    if snapshot and isinstance(snapshot.status, dict):
        last_collection = snapshot.status.get("last_scan")
    if not last_collection and snapshot:
        last_collection = _to_iso(snapshot.updated_at)

    next_offset_raw = offset_value + scanned_count
    has_more = next_offset_raw < int(total)
    next_offset = next_offset_raw if has_more else None

    return {
        "signals": signal_payload,
        "total": total,
        "offset": offset_value,
        "limit": limit_value,
        "has_more": has_more,
        "next_offset": next_offset,
        "last_collection": last_collection,
    }


@router.get("/world-intelligence/opportunities")
async def get_world_opportunities(
    signal_type: Optional[str] = Query(None, description="Filter by signal type"),
    country: Optional[str] = Query(None, description="Filter by country/ISO3"),
    min_severity: float = Query(0.5, ge=0.0, le=1.0),
    min_relevance: float = Query(0.3, ge=0.0, le=1.0),
    tradable_only: bool = Query(False),
    hours: int = Query(72, ge=1, le=720),
    max_markets_per_signal: int = Query(5, ge=1, le=20),
    limit: int = Query(200, ge=1, le=2000),
    session: AsyncSession = Depends(get_db_session),
):
    """Resolve world signals into opportunities with explicit tradability diagnostics."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=int(hours))
    signal_limit = max(
        50,
        min(5000, int(limit) * max(1, int(max_markets_per_signal))),
    )

    query = (
        select(WorldIntelligenceSignal)
        .where(WorldIntelligenceSignal.detected_at >= cutoff)
        .where(WorldIntelligenceSignal.severity >= float(min_severity))
    )
    if signal_type:
        query = query.where(WorldIntelligenceSignal.signal_type == str(signal_type))
    if country:
        country_key = str(country).strip().lower()
        query = query.where(
            or_(
                func.lower(func.coalesce(WorldIntelligenceSignal.country, "")) == country_key,
                func.lower(func.coalesce(WorldIntelligenceSignal.iso3, "")) == country_key,
            )
        )

    rows = (
        (
            await session.execute(
                query.order_by(
                    WorldIntelligenceSignal.detected_at.desc(),
                    WorldIntelligenceSignal.severity.desc(),
                ).limit(signal_limit)
            )
        )
        .scalars()
        .all()
    )

    resolved = await resolve_world_signal_opportunities(
        session,
        rows,
        max_markets_per_signal=max_markets_per_signal,
    )
    resolved = [
        row
        for row in resolved
        if float(row.get("market_relevance_score") or 0.0) >= float(min_relevance)
    ]
    if tradable_only:
        resolved = [row for row in resolved if bool(row.get("tradable"))]

    resolved.sort(
        key=lambda row: (
            bool(row.get("tradable")),
            float(row.get("market_relevance_score") or 0.0),
            float(row.get("severity") or 0.0),
        ),
        reverse=True,
    )

    total = len(resolved)
    rows_out = resolved[: int(limit)]
    tradable_count = sum(1 for row in rows_out if bool(row.get("tradable")))

    return {
        "opportunities": rows_out,
        "total": total,
        "summary": {
            "returned": len(rows_out),
            "tradable": tradable_count,
            "unresolved": max(0, len(rows_out) - tradable_count),
            "window_hours": int(hours),
            "min_severity": float(min_severity),
            "min_relevance": float(min_relevance),
        },
    }


@router.get("/world-intelligence/instability")
async def get_instability_scores(
    country: Optional[str] = Query(None),
    min_score: float = Query(0.0),
    limit: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_db_session),
):
    """Get latest country instability index scores from DB."""
    country_value = country if isinstance(country, str) else None
    try:
        min_score_value = float(min_score)
    except Exception:
        min_score_value = 0.0
    try:
        limit_value = max(1, min(200, int(limit)))
    except Exception:
        limit_value = 50

    latest = await _latest_instability_by_country(session)

    rows: list[dict[str, Any]] = []
    for _, (row, prev) in latest.items():
        iso3 = (row.iso3 or "").upper()
        row_country = (row.country or "")
        if country_value:
            c = country_value.strip().lower()
            if c not in {iso3.lower(), row_country.lower()}:
                continue
        if float(row.score or 0.0) < min_score_value:
            continue

        change_24h = None
        if prev is not None and prev.score is not None and row.score is not None:
            change_24h = round(float(row.score) - float(prev.score), 1)
        change_7d = await _instability_change_7d(session, row)

        rows.append(
            {
                "country": country_catalog.country_name(iso3 or row_country) or row_country,
                "country_name": country_catalog.country_name(iso3 or row_country) or row_country,
                "iso3": country_catalog.normalize_iso3(iso3 or row_country) or iso3,
                "score": round(float(row.score or 0.0), 1),
                "trend": row.trend or "stable",
                "change_24h": change_24h,
                "change_7d": change_7d,
                "components": row.components or {},
                "contributing_signals": (row.components or {}).get("contributing_signals", []),
                "last_updated": _to_iso(row.computed_at),
            }
        )

    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    rows = rows[:limit_value]
    return {"scores": rows, "total": len(rows)}


@router.get("/world-intelligence/tensions")
async def get_tension_pairs(
    min_tension: float = Query(0.0),
    limit: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_db_session),
):
    """Get latest country-pair tension scores from DB."""
    try:
        min_tension_value = float(min_tension)
    except Exception:
        min_tension_value = 0.0
    try:
        limit_value = max(1, min(100, int(limit)))
    except Exception:
        limit_value = 20

    latest = await _latest_tension_pairs(session)
    tension_meta = await _latest_tension_signal_meta(session)

    tensions: list[dict[str, Any]] = []
    for _, (row, _prev) in latest.items():
        score = float(row.tension_score or 0.0)
        if score < min_tension_value:
            continue
        tensions.append(
            {
                "country_a": row.country_a,
                "country_b": row.country_b,
                "country_a_iso3": country_catalog.normalize_iso3(row.country_a),
                "country_b_iso3": country_catalog.normalize_iso3(row.country_b),
                "country_a_name": country_catalog.country_name(row.country_a),
                "country_b_name": country_catalog.country_name(row.country_b),
                "tension_score": round(score, 1),
                "event_count": int(row.event_count or 0),
                "avg_goldstein_scale": (
                    round(float(row.avg_goldstein_scale), 2)
                    if row.avg_goldstein_scale is not None
                    else None
                ),
                "trend": row.trend or "stable",
                "top_event_types": list(
                    (
                        tension_meta.get(
                            _normalized_tension_pair_key(row.country_a, row.country_b),
                            {},
                        )
                        or {}
                    )
                    .get("top_event_types")
                    or []
                ),
                "last_updated": _to_iso(row.computed_at),
            }
        )

    tensions.sort(key=lambda t: float(t["tension_score"]), reverse=True)
    tensions = tensions[:limit_value]
    return {"tensions": tensions, "total": len(tensions)}


@router.get("/world-intelligence/convergences")
async def get_convergence_zones(
    session: AsyncSession = Depends(get_db_session),
):
    """Get active geo-convergence zones from persisted signals."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
    rows = (
        (
            await session.execute(
                select(WorldIntelligenceSignal)
                .where(WorldIntelligenceSignal.signal_type == "convergence")
                .where(WorldIntelligenceSignal.detected_at >= cutoff)
                .order_by(WorldIntelligenceSignal.severity.desc(), WorldIntelligenceSignal.detected_at.desc())
                .limit(250)
            )
        )
        .scalars()
        .all()
    )

    zones = []
    for row in rows:
        meta = row.metadata_json or {}
        signal_types = meta.get("signal_types") or []
        zones.append(
            {
                "grid_key": meta.get("grid_key") or row.id,
                "latitude": row.latitude or 0.0,
                "longitude": row.longitude or 0.0,
                "signal_types": list(signal_types),
                "signal_count": int(meta.get("signal_count") or len(signal_types) or 0),
                "urgency_score": round(float(meta.get("urgency_score") or (float(row.severity or 0.0) * 100.0)), 1),
                "country": country_catalog.country_name(row.country) or "",
                "nearby_markets": list(row.related_market_ids or []),
                "detected_at": _to_iso(row.detected_at),
            }
        )

    zones.sort(key=lambda z: float(z["urgency_score"]), reverse=True)
    return {"zones": zones, "total": len(zones)}


@router.get("/world-intelligence/anomalies")
async def get_temporal_anomalies(
    min_severity: str = Query("medium", description="Minimum severity: normal, medium, high, critical"),
    session: AsyncSession = Depends(get_db_session),
):
    """Get temporal anomalies from persisted anomaly signals."""
    severity_order = {"normal": 0, "medium": 1, "high": 2, "critical": 3}
    min_severity_value = min_severity if isinstance(min_severity, str) else "medium"
    min_level = severity_order.get((min_severity_value or "medium").lower(), 1)

    cutoff = datetime.now(timezone.utc) - timedelta(hours=72)
    rows = (
        (
            await session.execute(
                select(WorldIntelligenceSignal)
                .where(WorldIntelligenceSignal.signal_type == "anomaly")
                .where(WorldIntelligenceSignal.detected_at >= cutoff)
                .order_by(WorldIntelligenceSignal.detected_at.desc())
                .limit(500)
            )
        )
        .scalars()
        .all()
    )

    anomalies = []
    for row in rows:
        meta = row.metadata_json or {}
        severity_name = _severity_label(float(row.severity or 0.0))
        if severity_order.get(severity_name, 0) < min_level:
            continue
        anomalies.append(
            {
                "signal_type": meta.get("signal_type") or row.title.split(":", 1)[0],
                "country": country_catalog.country_name(row.country) or "unknown",
                "z_score": round(float(meta.get("z_score") or 0.0), 2),
                "severity": severity_name,
                "current_value": float(meta.get("current_value") or 0.0),
                "baseline_mean": round(float(meta.get("baseline_mean") or 0.0), 2),
                "baseline_std": round(float(meta.get("baseline_std") or 0.0), 2),
                "description": row.description or "",
                "detected_at": _to_iso(row.detected_at),
            }
        )

    anomalies.sort(key=lambda a: abs(float(a["z_score"])), reverse=True)
    return {"anomalies": anomalies, "total": len(anomalies)}


@router.get("/world-intelligence/military")
async def get_military_activity(session: AsyncSession = Depends(get_db_session)):
    """Get persisted military activity summary from DB."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    rows = (
        (
            await session.execute(
                select(WorldIntelligenceSignal)
                .where(WorldIntelligenceSignal.signal_type == "military")
                .where(WorldIntelligenceSignal.detected_at >= cutoff)
                .order_by(WorldIntelligenceSignal.detected_at.desc())
                .limit(2000)
            )
        )
        .scalars()
        .all()
    )
    by_region: dict[str, int] = {}
    by_type: dict[str, int] = {}
    surge_regions: set[str] = set()
    for row in rows:
        meta = row.metadata_json or {}
        region = str(meta.get("region") or "other")
        by_region[region] = by_region.get(region, 0) + 1
        activity_type = str(meta.get("activity_type") or ("flight" if row.source == "opensky" else "vessel"))
        by_type[activity_type] = by_type.get(activity_type, 0) + 1
        if bool(meta.get("is_unusual")):
            surge_regions.add(region)

    snapshot = await _get_world_snapshot(session)
    source_health = {}
    if snapshot and isinstance(snapshot.stats, dict):
        source_health = ((snapshot.stats or {}).get("source_status") or {}).get("military", {}) or {}
    from services.world_intelligence.military_monitor import military_monitor
    live_health = military_monitor.get_health()
    if isinstance(source_health, dict):
        source_health = {**source_health, **live_health}
    else:
        source_health = live_health

    return {
        "total": len(rows),
        "by_region": by_region,
        "by_type": by_type,
        "surge_regions": sorted(surge_regions),
        "hotspots": await _dynamic_military_hotspots(session, max_hotspots=20, min_events=1),
        "source_health": source_health,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/world-intelligence/infrastructure")
async def get_infrastructure_events(session: AsyncSession = Depends(get_db_session)):
    """Get persisted infrastructure disruptions and cascade summaries."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=72)
    rows = (
        (
            await session.execute(
                select(WorldIntelligenceSignal)
                .where(WorldIntelligenceSignal.signal_type == "infrastructure")
                .where(WorldIntelligenceSignal.detected_at >= cutoff)
                .order_by(WorldIntelligenceSignal.detected_at.desc())
                .limit(2000)
            )
        )
        .scalars()
        .all()
    )

    disruptions: list[dict[str, Any]] = []
    for row in rows:
        meta = row.metadata_json or {}
        disruptions.append(
            {
                "event_type": meta.get("event_type") or "internet_outage",
                "country": country_catalog.country_name(row.country) or row.country,
                "severity": round(float(row.severity or 0.0), 2),
                "started_at": _to_iso(row.detected_at),
                "description": row.description or "",
                "source": row.source,
                "cascade_risk_score": round(float(meta.get("cascade_risk_score") or 0.0), 2),
                "latitude": row.latitude,
                "longitude": row.longitude,
            }
        )

    snapshot = await _get_world_snapshot(session)
    source_health = {}
    if snapshot and isinstance(snapshot.stats, dict):
        source_health = ((snapshot.stats or {}).get("source_status") or {}).get("infrastructure", {}) or {}

    cascade_risks = sorted(
        [
            {
                "country": item.get("country"),
                "event_type": item.get("event_type"),
                "description": item.get("description"),
                "cascade_risk_score": item.get("cascade_risk_score"),
                "started_at": item.get("started_at"),
            }
            for item in disruptions
            if float(item.get("cascade_risk_score") or 0.0) > 0.0
        ],
        key=lambda item: float(item.get("cascade_risk_score") or 0.0),
        reverse=True,
    )

    return {
        "disruptions": disruptions,
        "cascade_risks": cascade_risks[:100],
        "total_disruptions": len(disruptions),
        "source_health": source_health,
    }


@router.get("/world-intelligence/sources")
async def get_world_source_status(session: AsyncSession = Depends(get_db_session)):
    snapshot = await _get_world_snapshot(session)
    stats = (snapshot.stats if snapshot and isinstance(snapshot.stats, dict) else {}) or {}
    worker = await read_worker_snapshot(session, "world_intelligence")
    worker_stats = worker.get("stats") if isinstance(worker.get("stats"), dict) else {}
    sources = stats.get("source_status") or worker_stats.get("source_status") or {}
    errors = stats.get("source_errors") or worker_stats.get("source_errors") or []
    merged_sources = dict(sources) if isinstance(sources, dict) else {}
    merged_sources["chokepoints"] = await get_chokepoint_reference_source_status(session)
    merged_sources["country_reference"] = await get_country_reference_source_status(session)
    merged_sources["ucdp_conflicts"] = await get_ucdp_conflict_source_status(session)
    merged_sources["mid_reference"] = await get_mid_reference_source_status(session)
    merged_sources["trade_dependencies"] = await get_trade_dependency_source_status(session)
    merged_sources["gdelt_news"] = await get_gdelt_news_source_status(session)
    normalized_sources = {
        str(name): _normalize_source_health_entry(details)
        for name, details in merged_sources.items()
    }
    normalized_errors = [
        str(error)
        for error in (errors or [])
        if str(error).strip() and not _is_benign_source_error(error)
    ]
    return {
        "sources": normalized_sources,
        "errors": normalized_errors,
        "updated_at": _to_iso(snapshot.updated_at) if snapshot else worker.get("updated_at"),
    }


@router.get("/world-intelligence/reference/countries/status")
async def get_world_country_reference_status(
    session: AsyncSession = Depends(get_db_session),
):
    status = await get_country_reference_source_status(session)
    status["runtime_source"] = country_catalog.runtime_source() or status.get("source")
    return status


@router.post("/world-intelligence/reference/countries/sync")
async def sync_world_country_reference(
    force: bool = Query(False),
    session: AsyncSession = Depends(get_db_session),
):
    result = await sync_country_reference_from_world_bank(session, force=bool(force))
    result["runtime_source"] = country_catalog.runtime_source() or result.get("source")
    return result


@router.get("/world-intelligence/reference/ucdp/status")
async def get_world_ucdp_status(
    session: AsyncSession = Depends(get_db_session),
):
    return await get_ucdp_conflict_source_status(session)


@router.post("/world-intelligence/reference/ucdp/sync")
async def sync_world_ucdp_conflicts(
    force: bool = Query(False),
    session: AsyncSession = Depends(get_db_session),
):
    return await sync_ucdp_conflict_lists(session, force=bool(force))


@router.get("/world-intelligence/reference/mid/status")
async def get_world_mid_reference_status(
    session: AsyncSession = Depends(get_db_session),
):
    return await get_mid_reference_source_status(session)


@router.post("/world-intelligence/reference/mid/sync")
async def sync_world_mid_reference(
    force: bool = Query(False),
    session: AsyncSession = Depends(get_db_session),
):
    return await sync_mid_reference_from_itu(session, force=bool(force))


@router.get("/world-intelligence/reference/trade-dependencies/status")
async def get_world_trade_dependency_status(
    session: AsyncSession = Depends(get_db_session),
):
    return await get_trade_dependency_source_status(session)


@router.post("/world-intelligence/reference/trade-dependencies/sync")
async def sync_world_trade_dependencies(
    force: bool = Query(False),
    session: AsyncSession = Depends(get_db_session),
):
    return await sync_trade_dependencies_from_world_bank(session, force=bool(force))


@router.get("/world-intelligence/reference/chokepoints/status")
async def get_world_chokepoint_reference_status(
    session: AsyncSession = Depends(get_db_session),
):
    return await get_chokepoint_reference_source_status(session)


@router.post("/world-intelligence/reference/chokepoints/sync")
async def sync_world_chokepoint_reference(
    force: bool = Query(False),
    session: AsyncSession = Depends(get_db_session),
):
    return await sync_chokepoint_reference_from_portwatch(session, force=bool(force))


@router.get("/world-intelligence/reference/gdelt-news/status")
async def get_world_gdelt_news_status(
    session: AsyncSession = Depends(get_db_session),
):
    return await get_gdelt_news_source_status(session)


@router.post("/world-intelligence/reference/gdelt-news/sync")
async def sync_world_gdelt_news(
    force: bool = Query(False),
    session: AsyncSession = Depends(get_db_session),
):
    return await sync_gdelt_news_from_source(session, force=bool(force))


@router.get("/world-intelligence/summary")
async def get_world_intelligence_summary(
    session: AsyncSession = Depends(get_db_session),
):
    """Get high-level world intelligence summary from persisted data."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=2)
    signal_rows = (
        (
            await session.execute(
                select(WorldIntelligenceSignal)
                .where(WorldIntelligenceSignal.detected_at >= cutoff)
                .order_by(WorldIntelligenceSignal.detected_at.desc())
                .limit(5000)
            )
        )
        .scalars()
        .all()
    )

    by_type: dict[str, int] = {}
    by_severity = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    critical_anomalies = 0
    for row in signal_rows:
        typ = row.signal_type or "unknown"
        by_type[typ] = by_type.get(typ, 0) + 1
        sev = float(row.severity or 0.0)
        if sev >= 0.8:
            by_severity["critical"] += 1
        elif sev >= 0.6:
            by_severity["high"] += 1
        elif sev >= 0.3:
            by_severity["medium"] += 1
        else:
            by_severity["low"] += 1
        if typ == "anomaly" and sev >= 0.8:
            critical_anomalies += 1

    instability_latest = await _latest_instability_by_country(session)
    instability_threshold = float(
        max(0.0, getattr(settings, "WORLD_INTEL_INSTABILITY_CRITICAL", 60.0) or 60.0)
    )
    critical_countries = []
    for _, (row, _prev) in instability_latest.items():
        score = float(row.score or 0.0)
        if score >= instability_threshold:
            critical_countries.append(
                {
                    "country": country_catalog.country_name(row.iso3 or row.country),
                    "iso3": country_catalog.normalize_iso3(row.iso3 or row.country) or row.iso3,
                    "score": round(score, 1),
                    "trend": row.trend or "stable",
                }
            )
    critical_countries.sort(key=lambda c: float(c["score"]), reverse=True)

    tension_latest = await _latest_tension_pairs(session)
    tension_threshold = float(
        max(0.0, getattr(settings, "WORLD_INTEL_TENSION_CRITICAL", 70.0) or 70.0)
    )
    high_tensions = []
    for _, (row, _prev) in tension_latest.items():
        score = float(row.tension_score or 0.0)
        if score >= tension_threshold:
            high_tensions.append(
                {
                    "pair": f"{row.country_a}-{row.country_b}",
                    "country_a_iso3": country_catalog.normalize_iso3(row.country_a),
                    "country_b_iso3": country_catalog.normalize_iso3(row.country_b),
                    "country_a_name": country_catalog.country_name(row.country_a),
                    "country_b_name": country_catalog.country_name(row.country_b),
                    "score": round(score, 1),
                    "trend": row.trend or "stable",
                }
            )
    high_tensions.sort(key=lambda t: float(t["score"]), reverse=True)

    active_convergences = sum(1 for s in signal_rows if s.signal_type == "convergence")
    snapshot = await _get_world_snapshot(session)
    worker = await read_worker_snapshot(session, "world_intelligence")
    last_collection = None
    if snapshot and isinstance(snapshot.status, dict):
        last_collection = snapshot.status.get("last_scan")
    if not last_collection:
        last_collection = worker.get("last_run_at")
    if not last_collection and snapshot:
        last_collection = _to_iso(snapshot.updated_at)

    return {
        "signal_summary": {
            "total": len(signal_rows),
            "by_type": by_type,
            "by_severity": by_severity,
            "critical": by_severity["critical"],
            "high": by_severity["high"],
            "medium": by_severity["medium"],
            "low": by_severity["low"],
        },
        "critical_countries": critical_countries[:10],
        "high_tensions": high_tensions[:10],
        "critical_anomalies": critical_anomalies,
        "active_convergences": active_convergences,
        "last_collection": last_collection,
    }


@router.get("/world-intelligence/status")
async def get_world_intelligence_status(
    session: AsyncSession = Depends(get_db_session),
):
    """Get world intelligence runtime status with stale-data detection."""
    snapshot = await _get_world_snapshot(session)
    worker = await read_worker_snapshot(session, "world_intelligence")

    status = dict(snapshot.status or {}) if snapshot and isinstance(snapshot.status, dict) else {}
    stats = dict(snapshot.stats or {}) if snapshot and isinstance(snapshot.stats, dict) else {}
    worker_stats = worker.get("stats") if isinstance(worker.get("stats"), dict) else {}
    if not stats and worker_stats:
        stats = dict(worker_stats)
    public_stats = dict(stats)
    public_stats.pop("runtime_state", None)

    interval_seconds = int(
        status.get("interval_seconds")
        or worker.get("interval_seconds")
        or settings.WORLD_INTELLIGENCE_INTERVAL_SECONDS
    )
    status.setdefault("running", bool(worker.get("running", False)))
    status.setdefault("enabled", bool(worker.get("enabled", False)))
    status.setdefault("current_activity", worker.get("current_activity"))
    status.setdefault("last_scan", worker.get("last_run_at"))

    last_scan = _parse_iso(status.get("last_scan"))
    if last_scan is None and worker.get("last_run_at"):
        last_scan = _parse_iso(worker.get("last_run_at"))

    lag_seconds = None
    stale = False
    if last_scan is not None:
        lag_seconds = max(
            0.0,
            (datetime.now(timezone.utc).replace(tzinfo=None) - last_scan).total_seconds(),
        )
        stale_after = max(interval_seconds * 2, 900)
        stale = lag_seconds > stale_after

    status["interval_seconds"] = interval_seconds
    status["lag_seconds"] = round(lag_seconds, 1) if lag_seconds is not None else None
    status["stale"] = stale
    status["last_error"] = status.get("last_error") or worker.get("last_error")
    status["source_status"] = public_stats.get("source_status") or worker_stats.get("source_status") or {}
    status["source_errors"] = public_stats.get("source_errors") or worker_stats.get("source_errors") or []

    return {
        "status": status,
        "stats": public_stats,
        "updated_at": _to_iso(snapshot.updated_at) if snapshot else worker.get("updated_at"),
    }
