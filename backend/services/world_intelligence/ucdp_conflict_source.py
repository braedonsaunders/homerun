"""UCDP conflict-list sync for instability scoring floors."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from models.database import AppSettings
from .country_catalog import country_catalog
from .instability_catalog import instability_catalog

_UCDP_CONFLICT_URL = "https://ucdpapi.pcr.uu.se/api/ucdpprioconflict/25.1"


def _clean_iso3_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        iso3 = country_catalog.normalize_iso3(str(value or ""))
        if len(iso3) != 3:
            continue
        if iso3 in seen:
            continue
        seen.add(iso3)
        out.append(iso3)
    return out


def _default_conflict_lists() -> tuple[list[str], list[str]]:
    payload = instability_catalog.payload()
    active = _clean_iso3_list(payload.get("ucdp_active_wars") or [])
    minor = _clean_iso3_list(payload.get("ucdp_minor_conflicts") or [])
    return active, minor


async def _get_or_create_app_settings(session: AsyncSession) -> AppSettings:
    result = await session.execute(select(AppSettings).where(AppSettings.id == "default"))
    row = result.scalar_one_or_none()
    if row is None:
        active_default, minor_default = _default_conflict_lists()
        row = AppSettings(id="default")
        row.world_intel_ucdp_active_wars_json = active_default
        row.world_intel_ucdp_minor_conflicts_json = minor_default
        row.world_intel_ucdp_source = "static_seed"
        session.add(row)
        await session.commit()
        await session.refresh(row)
    return row


async def _fetch_ucdp_rows_for_year(year: int) -> list[dict[str, Any]]:
    timeout = float(
        max(
            5.0,
            getattr(settings, "WORLD_INTEL_UCDP_REQUEST_TIMEOUT_SECONDS", 25.0) or 25.0,
        )
    )
    max_pages = int(
        max(1, getattr(settings, "WORLD_INTEL_UCDP_MAX_PAGES", 100) or 100)
    )
    rows: list[dict[str, Any]] = []
    page = 0
    next_url: Optional[str] = None

    async with httpx.AsyncClient(timeout=timeout) as client:
        while page < max_pages:
            if page == 0:
                resp = await client.get(
                    _UCDP_CONFLICT_URL,
                    params={"year": str(year), "pagesize": "200"},
                )
            else:
                if not next_url:
                    break
                resp = await client.get(next_url)
            resp.raise_for_status()
            payload = resp.json()
            result_rows = payload.get("Result", []) if isinstance(payload, dict) else []
            if isinstance(result_rows, list):
                rows.extend([row for row in result_rows if isinstance(row, dict)])
            next_url = (
                str(payload.get("NextPageUrl") or "").strip()
                if isinstance(payload, dict)
                else ""
            )
            page += 1
            if not next_url:
                break
    return rows


async def _fetch_latest_ucdp_conflict_rows() -> tuple[int, list[dict[str, Any]]]:
    lookback_years = int(
        max(1, getattr(settings, "WORLD_INTEL_UCDP_LOOKBACK_YEARS", 8) or 8)
    )
    current_year = datetime.now(timezone.utc).year
    for year in range(current_year, current_year - lookback_years, -1):
        rows = await _fetch_ucdp_rows_for_year(year)
        if rows:
            return year, rows
    return current_year, []


def _derive_conflict_lists(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    active_wars: set[str] = set()
    minor_conflicts: set[str] = set()
    for row in rows:
        try:
            intensity = int(float(row.get("intensity_level") or 0))
        except Exception:
            continue
        location = str(row.get("location") or "").strip()
        iso3 = country_catalog.normalize_iso3(location)
        if not iso3:
            continue
        if intensity >= 2:
            active_wars.add(iso3)
        elif intensity == 1:
            minor_conflicts.add(iso3)
    active_sorted = sorted(active_wars)
    minor_sorted = sorted(minor_conflicts - active_wars)
    return active_sorted, minor_sorted


async def load_ucdp_conflict_lists_from_db(session: AsyncSession) -> dict[str, Any]:
    row = await _get_or_create_app_settings(session)
    active = _clean_iso3_list(getattr(row, "world_intel_ucdp_active_wars_json", None))
    minor = _clean_iso3_list(getattr(row, "world_intel_ucdp_minor_conflicts_json", None))
    if not active and not minor:
        active, minor = _default_conflict_lists()
        row.world_intel_ucdp_active_wars_json = active
        row.world_intel_ucdp_minor_conflicts_json = minor
        row.world_intel_ucdp_source = (
            str(getattr(row, "world_intel_ucdp_source", "") or "").strip()
            or "static_seed"
        )
        await session.commit()
    source = str(getattr(row, "world_intel_ucdp_source", "") or "").strip() or "db"
    year = getattr(row, "world_intel_ucdp_year", None)
    instability_catalog.set_runtime_conflict_lists(
        active_wars=active,
        minor_conflicts=minor,
        source=source,
        year=year,
    )
    return {
        "source": source,
        "year": int(year) if year is not None else None,
        "active_wars": len(active),
        "minor_conflicts": len(minor),
        "last_synced_at": (
            row.world_intel_ucdp_synced_at.isoformat()
            if isinstance(row.world_intel_ucdp_synced_at, datetime)
            else None
        ),
    }


async def sync_ucdp_conflict_lists(
    session: AsyncSession,
    *,
    force: bool = False,
) -> dict[str, Any]:
    row = await _get_or_create_app_settings(session)
    existing_active = _clean_iso3_list(getattr(row, "world_intel_ucdp_active_wars_json", None))
    existing_minor = _clean_iso3_list(getattr(row, "world_intel_ucdp_minor_conflicts_json", None))

    enabled = bool(getattr(settings, "WORLD_INTEL_UCDP_SYNC_ENABLED", True))
    interval_hours = int(max(1, getattr(settings, "WORLD_INTEL_UCDP_SYNC_HOURS", 24) or 24))
    last_sync = getattr(row, "world_intel_ucdp_synced_at", None)

    due = force or not (existing_active or existing_minor)
    if not due and enabled and isinstance(last_sync, datetime):
        last_sync_utc = (
            last_sync.astimezone(timezone.utc).replace(tzinfo=None)
            if last_sync.tzinfo is not None
            else last_sync
        )
        due = (datetime.now(timezone.utc).replace(tzinfo=None) - last_sync_utc) >= timedelta(
            hours=interval_hours
        )

    if not due:
        source = str(getattr(row, "world_intel_ucdp_source", "") or "").strip() or "db"
        year = getattr(row, "world_intel_ucdp_year", None)
        instability_catalog.set_runtime_conflict_lists(
            active_wars=existing_active,
            minor_conflicts=existing_minor,
            source=source,
            year=year,
        )
        return {
            "updated": False,
            "reason": "fresh",
            "source": source,
            "year": int(year) if year is not None else None,
            "active_wars": len(existing_active),
            "minor_conflicts": len(existing_minor),
            "last_synced_at": last_sync.isoformat() if isinstance(last_sync, datetime) else None,
        }

    if not enabled and not force:
        source = str(getattr(row, "world_intel_ucdp_source", "") or "").strip() or "db"
        year = getattr(row, "world_intel_ucdp_year", None)
        instability_catalog.set_runtime_conflict_lists(
            active_wars=existing_active,
            minor_conflicts=existing_minor,
            source=source,
            year=year,
        )
        return {
            "updated": False,
            "reason": "disabled",
            "source": source,
            "year": int(year) if year is not None else None,
            "active_wars": len(existing_active),
            "minor_conflicts": len(existing_minor),
            "last_synced_at": last_sync.isoformat() if isinstance(last_sync, datetime) else None,
        }

    year, fetched_rows = await _fetch_latest_ucdp_conflict_rows()
    active, minor = _derive_conflict_lists(fetched_rows)
    if not active and not minor:
        source = str(getattr(row, "world_intel_ucdp_source", "") or "").strip() or "db"
        instability_catalog.set_runtime_conflict_lists(
            active_wars=existing_active,
            minor_conflicts=existing_minor,
            source=source,
            year=getattr(row, "world_intel_ucdp_year", None),
        )
        return {
            "updated": False,
            "reason": "empty_fetch",
            "source": source,
            "year": int(getattr(row, "world_intel_ucdp_year", 0) or 0) or None,
            "active_wars": len(existing_active),
            "minor_conflicts": len(existing_minor),
            "last_synced_at": last_sync.isoformat() if isinstance(last_sync, datetime) else None,
        }

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    row.world_intel_ucdp_active_wars_json = active
    row.world_intel_ucdp_minor_conflicts_json = minor
    row.world_intel_ucdp_source = "ucdp_api"
    row.world_intel_ucdp_year = int(year)
    row.world_intel_ucdp_synced_at = now
    await session.commit()
    instability_catalog.set_runtime_conflict_lists(
        active_wars=active,
        minor_conflicts=minor,
        source="ucdp_api",
        year=year,
    )
    return {
        "updated": True,
        "reason": "synced",
        "source": "ucdp_api",
        "year": int(year),
        "active_wars": len(active),
        "minor_conflicts": len(minor),
        "last_synced_at": now.isoformat(),
    }


async def get_ucdp_conflict_source_status(session: AsyncSession) -> dict[str, Any]:
    row = await _get_or_create_app_settings(session)
    active = _clean_iso3_list(getattr(row, "world_intel_ucdp_active_wars_json", None))
    minor = _clean_iso3_list(getattr(row, "world_intel_ucdp_minor_conflicts_json", None))
    source = str(getattr(row, "world_intel_ucdp_source", "") or "").strip() or "db"
    year = getattr(row, "world_intel_ucdp_year", None)
    last_sync = getattr(row, "world_intel_ucdp_synced_at", None)
    return {
        "source": instability_catalog.runtime_source() or source,
        "year": instability_catalog.runtime_year() or (int(year) if year is not None else None),
        "active_wars": len(active),
        "minor_conflicts": len(minor),
        "last_synced_at": last_sync.isoformat() if isinstance(last_sync, datetime) else None,
        "sync_enabled": bool(getattr(settings, "WORLD_INTEL_UCDP_SYNC_ENABLED", True)),
        "sync_interval_hours": int(
            max(1, getattr(settings, "WORLD_INTEL_UCDP_SYNC_HOURS", 24) or 24)
        ),
    }
