"""DB persistence and sync for PortWatch chokepoint reference rows."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from models.database import AppSettings
from .chokepoint_feed import chokepoint_feed
from .region_catalog import region_catalog


def _clean_chokepoint_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        row_id = str(item.get("id") or item.get("portid") or "").strip()
        if not row_id or row_id in seen:
            continue
        try:
            lat = float(item.get("latitude"))
            lon = float(item.get("longitude"))
        except Exception:
            continue
        normalized: dict[str, Any] = {}
        for key, raw in item.items():
            if raw is None or isinstance(raw, (str, int, float, bool)):
                normalized[str(key)] = raw
            elif isinstance(raw, datetime):
                normalized[str(key)] = raw.isoformat()
            else:
                normalized[str(key)] = str(raw)
        normalized["id"] = row_id
        normalized["latitude"] = lat
        normalized["longitude"] = lon
        normalized["name"] = str(normalized.get("name") or row_id).strip() or row_id
        normalized["source"] = str(normalized.get("source") or "db").strip() or "db"
        out.append(normalized)
        seen.add(row_id)
    return out


def _default_chokepoint_rows() -> list[dict[str, Any]]:
    payload = region_catalog.payload()
    updated_at = payload.get("updated_at")
    rows: list[dict[str, Any]] = []
    for row in payload.get("chokepoints", []) or []:
        if not isinstance(row, dict):
            continue
        row_id = str(row.get("id") or row.get("name") or "").strip()
        if not row_id:
            continue
        try:
            lat = float(row.get("latitude"))
            lon = float(row.get("longitude"))
        except Exception:
            continue
        rows.append(
            {
                "id": row_id,
                "name": str(row.get("name") or row_id).strip() or row_id,
                "latitude": lat,
                "longitude": lon,
                "source": "static_seed",
                "last_updated": updated_at,
            }
        )
    return _clean_chokepoint_rows(rows)


async def _get_or_create_app_settings(session: AsyncSession) -> AppSettings:
    result = await session.execute(select(AppSettings).where(AppSettings.id == "default"))
    row = result.scalar_one_or_none()
    if row is None:
        row = AppSettings(id="default")
        row.world_intel_chokepoints_json = _default_chokepoint_rows()
        row.world_intel_chokepoints_source = "static_seed"
        session.add(row)
        await session.commit()
        await session.refresh(row)
    return row


async def load_chokepoint_reference_from_db(session: AsyncSession) -> dict[str, Any]:
    row = await _get_or_create_app_settings(session)
    rows = _clean_chokepoint_rows(getattr(row, "world_intel_chokepoints_json", None))
    if not rows:
        rows = _default_chokepoint_rows()
        row.world_intel_chokepoints_json = rows
        row.world_intel_chokepoints_source = (
            str(getattr(row, "world_intel_chokepoints_source", "") or "").strip()
            or "static_seed"
        )
        await session.commit()

    source = str(getattr(row, "world_intel_chokepoints_source", "") or "").strip() or "db"
    synced_at = getattr(row, "world_intel_chokepoints_synced_at", None)
    count = chokepoint_feed.seed_cache(rows, source=source, synced_at=synced_at)
    return {
        "source": source,
        "count": count,
        "last_synced_at": synced_at.isoformat() if isinstance(synced_at, datetime) else None,
    }


async def sync_chokepoint_reference_from_portwatch(
    session: AsyncSession,
    *,
    force: bool = False,
) -> dict[str, Any]:
    row = await _get_or_create_app_settings(session)
    existing = _clean_chokepoint_rows(getattr(row, "world_intel_chokepoints_json", None))

    enabled = bool(getattr(settings, "WORLD_INTEL_CHOKEPOINTS_DB_SYNC_ENABLED", True))
    interval_hours = int(
        max(1, getattr(settings, "WORLD_INTEL_CHOKEPOINTS_DB_SYNC_HOURS", 6) or 6)
    )
    last_sync = getattr(row, "world_intel_chokepoints_synced_at", None)

    due = force or not existing
    if not due and isinstance(last_sync, datetime):
        last_sync_utc = (
            last_sync.astimezone(timezone.utc).replace(tzinfo=None)
            if last_sync.tzinfo is not None
            else last_sync
        )
        due = (datetime.now(timezone.utc).replace(tzinfo=None) - last_sync_utc) >= timedelta(
            hours=interval_hours
        )

    if not due:
        if existing and int(chokepoint_feed.get_health().get("count") or 0) <= 0:
            chokepoint_feed.seed_cache(
                existing,
                source=str(getattr(row, "world_intel_chokepoints_source", "") or "db"),
                synced_at=last_sync,
            )
        return {
            "updated": False,
            "reason": "fresh",
            "source": str(getattr(row, "world_intel_chokepoints_source", "") or "").strip() or "db",
            "count": len(existing),
            "last_synced_at": last_sync.isoformat() if isinstance(last_sync, datetime) else None,
        }

    if not enabled and not force:
        return {
            "updated": False,
            "reason": "disabled",
            "source": str(getattr(row, "world_intel_chokepoints_source", "") or "").strip() or "db",
            "count": len(existing),
            "last_synced_at": last_sync.isoformat() if isinstance(last_sync, datetime) else None,
        }

    rows = await chokepoint_feed.refresh(force=True)
    cleaned = _clean_chokepoint_rows(rows)
    if not cleaned:
        if existing:
            chokepoint_feed.seed_cache(
                existing,
                source=str(getattr(row, "world_intel_chokepoints_source", "") or "db"),
                synced_at=last_sync,
            )
            return {
                "updated": False,
                "reason": "empty_fetch",
                "source": str(getattr(row, "world_intel_chokepoints_source", "") or "").strip() or "db",
                "count": len(existing),
                "last_synced_at": last_sync.isoformat() if isinstance(last_sync, datetime) else None,
            }
        fallback = _default_chokepoint_rows()
        chokepoint_feed.seed_cache(fallback, source="static_fallback")
        return {
            "updated": False,
            "reason": "empty_fetch",
            "source": "static_fallback",
            "count": len(fallback),
            "last_synced_at": None,
        }

    health = chokepoint_feed.get_health()
    source = str(health.get("source") or "imf_portwatch").strip() or "imf_portwatch"
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    row.world_intel_chokepoints_json = cleaned
    row.world_intel_chokepoints_source = source
    row.world_intel_chokepoints_synced_at = now
    await session.commit()
    chokepoint_feed.seed_cache(cleaned, source=source, synced_at=now)
    return {
        "updated": True,
        "reason": "synced",
        "source": source,
        "count": len(cleaned),
        "last_synced_at": now.isoformat(),
    }


async def get_chokepoint_reference_source_status(session: AsyncSession) -> dict[str, Any]:
    row = await _get_or_create_app_settings(session)
    rows = _clean_chokepoint_rows(getattr(row, "world_intel_chokepoints_json", None))
    source = str(getattr(row, "world_intel_chokepoints_source", "") or "").strip() or "db"
    last_sync = getattr(row, "world_intel_chokepoints_synced_at", None)
    return {
        "source": source,
        "count": len(rows),
        "last_synced_at": last_sync.isoformat() if isinstance(last_sync, datetime) else None,
        "sync_enabled": bool(getattr(settings, "WORLD_INTEL_CHOKEPOINTS_DB_SYNC_ENABLED", True)),
        "sync_interval_hours": int(
            max(1, getattr(settings, "WORLD_INTEL_CHOKEPOINTS_DB_SYNC_HOURS", 6) or 6)
        ),
        "runtime_health": chokepoint_feed.get_health(),
    }

