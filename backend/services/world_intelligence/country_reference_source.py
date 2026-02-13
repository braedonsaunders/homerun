"""Authoritative country-reference sync and DB runtime loading."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from models.database import AppSettings
from .country_catalog import country_catalog

_WORLD_BANK_COUNTRIES_URL = "https://api.worldbank.org/v2/country"


def _clean_country_rows(rows: Any) -> list[dict[str, str]]:
    if not isinstance(rows, list):
        return []
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        alpha2 = str(row.get("alpha2") or "").strip().upper()
        alpha3 = str(row.get("alpha3") or "").strip().upper()
        if not name or len(alpha2) != 2 or len(alpha3) != 3:
            continue
        if alpha3 in seen:
            continue
        seen.add(alpha3)
        out.append({"name": name, "alpha2": alpha2, "alpha3": alpha3})
    out.sort(key=lambda item: (item["alpha3"], item["name"]))
    return out


def _default_country_rows() -> list[dict[str, str]]:
    payload = country_catalog.payload()
    rows = payload if isinstance(payload, list) else payload.get("countries", [])
    return _clean_country_rows(rows)


async def _get_or_create_app_settings(session: AsyncSession) -> AppSettings:
    result = await session.execute(select(AppSettings).where(AppSettings.id == "default"))
    row = result.scalar_one_or_none()
    if row is None:
        row = AppSettings(id="default")
        row.world_intel_country_reference_json = _default_country_rows()
        row.world_intel_country_reference_source = "static_seed"
        session.add(row)
        await session.commit()
        await session.refresh(row)
    return row


async def fetch_world_bank_country_rows() -> list[dict[str, str]]:
    timeout = float(
        max(
            5.0,
            getattr(settings, "WORLD_INTEL_COUNTRY_REFERENCE_REQUEST_TIMEOUT_SECONDS", 20.0) or 20.0,
        )
    )
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(
            _WORLD_BANK_COUNTRIES_URL,
            params={"format": "json", "per_page": "400"},
        )
        resp.raise_for_status()
        payload = resp.json()

    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
        return []

    rows: list[dict[str, str]] = []
    for item in payload[1]:
        if not isinstance(item, dict):
            continue
        region = item.get("region")
        region_name = str(region.get("value") or "").strip().lower() if isinstance(region, dict) else ""
        iso3 = str(item.get("id") or "").strip().upper()
        iso2 = str(item.get("iso2Code") or "").strip().upper()
        name = str(item.get("name") or "").strip()
        if region_name == "aggregates":
            continue
        if len(iso3) != 3 or not iso3.isalpha():
            continue
        if len(iso2) != 2 or not iso2.isalpha():
            continue
        if not name:
            continue
        rows.append({"name": name, "alpha2": iso2, "alpha3": iso3})
    return _clean_country_rows(rows)


async def load_country_reference_from_db(session: AsyncSession) -> int:
    row = await _get_or_create_app_settings(session)
    rows = _clean_country_rows(getattr(row, "world_intel_country_reference_json", None))
    if not rows:
        rows = _default_country_rows()
        row.world_intel_country_reference_json = rows
        row.world_intel_country_reference_source = (
            str(getattr(row, "world_intel_country_reference_source", "") or "").strip() or "static_seed"
        )
        await session.commit()
    source = str(getattr(row, "world_intel_country_reference_source", "") or "").strip() or "db"
    country_catalog.set_runtime_rows(rows, source=source)
    return len(rows)


async def sync_country_reference_from_world_bank(
    session: AsyncSession,
    *,
    force: bool = False,
) -> dict[str, Any]:
    row = await _get_or_create_app_settings(session)
    existing_rows = _clean_country_rows(getattr(row, "world_intel_country_reference_json", None))

    enabled = bool(getattr(settings, "WORLD_INTEL_COUNTRY_REFERENCE_SYNC_ENABLED", True))
    interval_hours = int(
        max(
            1,
            getattr(settings, "WORLD_INTEL_COUNTRY_REFERENCE_SYNC_HOURS", 24) or 24,
        )
    )
    last_sync = getattr(row, "world_intel_country_reference_synced_at", None)
    due = force or not existing_rows
    if not due and enabled and isinstance(last_sync, datetime):
        age = datetime.now(timezone.utc).replace(tzinfo=None) - (
            last_sync.astimezone(timezone.utc).replace(tzinfo=None) if last_sync.tzinfo is not None else last_sync
        )
        due = age >= timedelta(hours=interval_hours)
    if not due:
        source = str(getattr(row, "world_intel_country_reference_source", "") or "").strip() or "db"
        country_catalog.set_runtime_rows(existing_rows, source=source)
        return {
            "updated": False,
            "source": source,
            "count": len(existing_rows),
            "last_synced_at": last_sync.isoformat() if isinstance(last_sync, datetime) else None,
            "reason": "fresh",
        }

    if not enabled and not force:
        source = str(getattr(row, "world_intel_country_reference_source", "") or "").strip() or "db"
        country_catalog.set_runtime_rows(existing_rows, source=source)
        return {
            "updated": False,
            "source": source,
            "count": len(existing_rows),
            "last_synced_at": last_sync.isoformat() if isinstance(last_sync, datetime) else None,
            "reason": "disabled",
        }

    rows = await fetch_world_bank_country_rows()
    if not rows:
        source = str(getattr(row, "world_intel_country_reference_source", "") or "").strip() or "db"
        if existing_rows:
            country_catalog.set_runtime_rows(existing_rows, source=source)
            return {
                "updated": False,
                "source": source,
                "count": len(existing_rows),
                "last_synced_at": last_sync.isoformat() if isinstance(last_sync, datetime) else None,
                "reason": "empty_fetch",
            }
        fallback = _default_country_rows()
        country_catalog.set_runtime_rows(fallback, source="static_fallback")
        return {
            "updated": False,
            "source": "static_fallback",
            "count": len(fallback),
            "last_synced_at": None,
            "reason": "empty_fetch",
        }

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    row.world_intel_country_reference_json = rows
    row.world_intel_country_reference_source = "world_bank_api"
    row.world_intel_country_reference_synced_at = now
    await session.commit()
    country_catalog.set_runtime_rows(rows, source="world_bank_api")
    return {
        "updated": True,
        "source": "world_bank_api",
        "count": len(rows),
        "last_synced_at": now.isoformat(),
        "reason": "synced",
    }


async def get_country_reference_source_status(session: AsyncSession) -> dict[str, Any]:
    row = await _get_or_create_app_settings(session)
    db_rows = _clean_country_rows(getattr(row, "world_intel_country_reference_json", None))
    source = str(getattr(row, "world_intel_country_reference_source", "") or "").strip() or None
    last_sync = getattr(row, "world_intel_country_reference_synced_at", None)
    return {
        "source": source or country_catalog.runtime_source() or "static",
        "count": len(db_rows),
        "last_synced_at": last_sync.isoformat() if isinstance(last_sync, datetime) else None,
        "sync_enabled": bool(getattr(settings, "WORLD_INTEL_COUNTRY_REFERENCE_SYNC_ENABLED", True)),
        "sync_interval_hours": int(
            max(
                1,
                getattr(settings, "WORLD_INTEL_COUNTRY_REFERENCE_SYNC_HOURS", 24) or 24,
            )
        ),
    }
