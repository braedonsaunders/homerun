"""ITU MID sync for AIS vessel MID -> ISO3 mapping."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import html
import re
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from models.database import AppSettings
from .country_catalog import country_catalog
from .military_catalog import military_catalog

_ITU_MID_URL = "https://www.itu.int/gladapp/Allocation/MIDs"

# Explicit crosswalk for administration labels that do not normalize directly.
_ADMIN_OVERRIDES: dict[str, str] = {
    "Vatican City State": "VAT",
    "United Kingdom of Great Britain and Northern Ireland": "GBR",
    "United States of America": "USA",
}


def _clean_mid_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for key, val in value.items():
        mid = str(key).strip()
        iso3 = country_catalog.normalize_iso3(str(val or ""))
        if mid and len(iso3) == 3:
            out[mid] = iso3
    return out


def _default_mid_map() -> dict[str, str]:
    return _clean_mid_map(military_catalog.vessel_mid_iso3())


def _admin_candidates(name: str) -> list[str]:
    text = str(name or "").strip()
    if not text:
        return []
    out: list[str] = [text]
    # Remove parenthetical qualifiers
    out.append(re.sub(r"\s*\([^)]*\)", "", text).strip())
    # Split compound labels (e.g. "Portugal - Azores")
    if " - " in text:
        left, right = text.split(" - ", 1)
        out.extend([left.strip(), right.strip()])
    # Normalize common prefixes
    prefixes = (
        "Republic of ",
        "Kingdom of ",
        "State of ",
        "Principality of ",
        "Federation of ",
    )
    for item in list(out):
        for prefix in prefixes:
            if item.startswith(prefix):
                out.append(item[len(prefix):].strip())
    # Deduplicate preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for item in out:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def _extract_mid_rows_from_html(payload: str) -> list[tuple[str, str]]:
    table_match = re.search(
        r'<table[^>]*class="table table-striped table-condensed table-hover"[^>]*>(.*?)</table>',
        payload,
        re.S,
    )
    block = table_match.group(1) if table_match else payload
    rows: list[tuple[str, str]] = []
    for tr in re.findall(r"<tr>(.*?)</tr>", block, re.S):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
        if len(cells) < 2:
            continue
        mid_text = html.unescape(re.sub(r"<[^>]+>", "", cells[0])).strip()
        admin = html.unescape(re.sub(r"<[^>]+>", "", cells[1])).strip()
        if len(mid_text) == 3 and mid_text.isdigit() and admin:
            rows.append((mid_text, admin))
    return rows


def _mid_rows_to_iso3(rows: list[tuple[str, str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for mid, admin in rows:
        override = _ADMIN_OVERRIDES.get(admin)
        if override:
            out[mid] = override
            continue
        iso3 = ""
        for candidate in _admin_candidates(admin):
            iso3 = country_catalog.normalize_iso3(candidate)
            if iso3:
                break
        if len(iso3) == 3:
            out[mid] = iso3
    return out


async def _get_or_create_app_settings(session: AsyncSession) -> AppSettings:
    result = await session.execute(select(AppSettings).where(AppSettings.id == "default"))
    row = result.scalar_one_or_none()
    if row is None:
        row = AppSettings(id="default")
        row.world_intel_mid_iso3_json = _default_mid_map()
        row.world_intel_mid_source = "static_seed"
        session.add(row)
        await session.commit()
        await session.refresh(row)
    return row


async def fetch_itu_mid_map() -> dict[str, str]:
    timeout = float(
        max(
            5.0,
            getattr(settings, "WORLD_INTEL_MID_REQUEST_TIMEOUT_SECONDS", 20.0) or 20.0,
        )
    )
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(_ITU_MID_URL)
        resp.raise_for_status()
        text = resp.text
    rows = _extract_mid_rows_from_html(text)
    return _mid_rows_to_iso3(rows)


async def load_mid_reference_from_db(session: AsyncSession) -> dict[str, Any]:
    row = await _get_or_create_app_settings(session)
    mapping = _clean_mid_map(getattr(row, "world_intel_mid_iso3_json", None))
    if not mapping:
        mapping = _default_mid_map()
        row.world_intel_mid_iso3_json = mapping
        row.world_intel_mid_source = str(getattr(row, "world_intel_mid_source", "") or "").strip() or "static_seed"
        await session.commit()
    source = str(getattr(row, "world_intel_mid_source", "") or "").strip() or "db"
    military_catalog.set_runtime_vessel_mid_iso3(mapping, source=source)
    return {
        "source": source,
        "count": len(mapping),
        "last_synced_at": (
            row.world_intel_mid_synced_at.isoformat()
            if isinstance(row.world_intel_mid_synced_at, datetime)
            else None
        ),
    }


async def sync_mid_reference_from_itu(
    session: AsyncSession,
    *,
    force: bool = False,
) -> dict[str, Any]:
    row = await _get_or_create_app_settings(session)
    existing = _clean_mid_map(getattr(row, "world_intel_mid_iso3_json", None))

    enabled = bool(getattr(settings, "WORLD_INTEL_MID_SYNC_ENABLED", True))
    interval_hours = int(max(1, getattr(settings, "WORLD_INTEL_MID_SYNC_HOURS", 168) or 168))
    last_sync = getattr(row, "world_intel_mid_synced_at", None)

    due = force or not existing
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
        source = str(getattr(row, "world_intel_mid_source", "") or "").strip() or "db"
        military_catalog.set_runtime_vessel_mid_iso3(existing, source=source)
        return {
            "updated": False,
            "reason": "fresh",
            "source": source,
            "count": len(existing),
            "last_synced_at": last_sync.isoformat() if isinstance(last_sync, datetime) else None,
        }

    if not enabled and not force:
        source = str(getattr(row, "world_intel_mid_source", "") or "").strip() or "db"
        military_catalog.set_runtime_vessel_mid_iso3(existing, source=source)
        return {
            "updated": False,
            "reason": "disabled",
            "source": source,
            "count": len(existing),
            "last_synced_at": last_sync.isoformat() if isinstance(last_sync, datetime) else None,
        }

    fetched = await fetch_itu_mid_map()
    if not fetched:
        source = str(getattr(row, "world_intel_mid_source", "") or "").strip() or "db"
        military_catalog.set_runtime_vessel_mid_iso3(existing, source=source)
        return {
            "updated": False,
            "reason": "empty_fetch",
            "source": source,
            "count": len(existing),
            "last_synced_at": last_sync.isoformat() if isinstance(last_sync, datetime) else None,
        }

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    row.world_intel_mid_iso3_json = fetched
    row.world_intel_mid_source = "itu_mid_table"
    row.world_intel_mid_synced_at = now
    await session.commit()
    military_catalog.set_runtime_vessel_mid_iso3(fetched, source="itu_mid_table")
    return {
        "updated": True,
        "reason": "synced",
        "source": "itu_mid_table",
        "count": len(fetched),
        "last_synced_at": now.isoformat(),
    }


async def get_mid_reference_source_status(session: AsyncSession) -> dict[str, Any]:
    row = await _get_or_create_app_settings(session)
    mapping = _clean_mid_map(getattr(row, "world_intel_mid_iso3_json", None))
    source = str(getattr(row, "world_intel_mid_source", "") or "").strip() or "db"
    last_sync = getattr(row, "world_intel_mid_synced_at", None)
    return {
        "source": military_catalog.runtime_mid_source() or source,
        "count": len(mapping),
        "last_synced_at": last_sync.isoformat() if isinstance(last_sync, datetime) else None,
        "sync_enabled": bool(getattr(settings, "WORLD_INTEL_MID_SYNC_ENABLED", True)),
        "sync_interval_hours": int(
            max(1, getattr(settings, "WORLD_INTEL_MID_SYNC_HOURS", 168) or 168)
        ),
    }
