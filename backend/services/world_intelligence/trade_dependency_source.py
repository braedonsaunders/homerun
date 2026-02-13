"""World Bank trade-dependency sync for infrastructure graph weights."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from models.database import AppSettings
from .infrastructure_catalog import infrastructure_catalog

_WORLD_BANK_INDICATOR_URL = "https://api.worldbank.org/v2/country/all/indicator/NE.TRD.GNFS.ZS"


def _clean_trade_dependencies(value: Any) -> dict[str, dict[str, float]]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, dict[str, float]] = {}
    for iso3, deps in value.items():
        code = str(iso3).upper().strip()
        if len(code) != 3 or not isinstance(deps, dict):
            continue
        dep_map: dict[str, float] = {}
        for node, val in deps.items():
            try:
                dep_map[str(node)] = max(0.0, min(1.0, float(val)))
            except Exception:
                continue
        if dep_map:
            out[code] = dep_map
    return out


def _default_trade_dependencies() -> dict[str, dict[str, float]]:
    payload = infrastructure_catalog.payload()
    return _clean_trade_dependencies(payload.get("trade_dependencies") or {})


async def _get_or_create_app_settings(session: AsyncSession) -> AppSettings:
    result = await session.execute(select(AppSettings).where(AppSettings.id == "default"))
    row = result.scalar_one_or_none()
    if row is None:
        row = AppSettings(id="default")
        row.world_intel_trade_dependencies_json = _default_trade_dependencies()
        row.world_intel_trade_dependency_source = "static_seed"
        session.add(row)
        await session.commit()
        await session.refresh(row)
    return row


async def _fetch_world_bank_trade_indicator() -> tuple[int | None, dict[str, float]]:
    timeout = float(
        max(
            5.0,
            getattr(settings, "WORLD_INTEL_TRADE_DEPENDENCY_REQUEST_TIMEOUT_SECONDS", 20.0)
            or 20.0,
        )
    )
    per_page = int(
        max(500, getattr(settings, "WORLD_INTEL_TRADE_DEPENDENCY_WB_PER_PAGE", 5000) or 5000)
    )
    max_pages = int(
        max(1, getattr(settings, "WORLD_INTEL_TRADE_DEPENDENCY_WB_MAX_PAGES", 50) or 50)
    )
    rows: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        page = 1
        total_pages: int | None = None
        while page <= max_pages:
            resp = await client.get(
                _WORLD_BANK_INDICATOR_URL,
                params={
                    "format": "json",
                    "per_page": str(per_page),
                    "page": str(page),
                },
            )
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, list) or len(payload) < 2:
                break
            meta = payload[0] if isinstance(payload[0], dict) else {}
            if total_pages is None:
                try:
                    total_pages = int(meta.get("pages") or 0)
                except Exception:
                    total_pages = 0
            page_rows = payload[1] if isinstance(payload[1], list) else []
            if not page_rows:
                break
            rows.extend([row for row in page_rows if isinstance(row, dict)])
            if not total_pages or page >= total_pages:
                break
            page += 1

    # Keep latest available year per ISO3.
    latest_by_country: dict[str, tuple[int, float]] = {}
    for row in rows:
        country = row.get("countryiso3code")
        iso3 = str(country or "").upper().strip()
        if len(iso3) != 3:
            continue
        year_raw = row.get("date")
        value_raw = row.get("value")
        try:
            year = int(str(year_raw).strip())
            value = float(value_raw)
        except Exception:
            continue
        prev = latest_by_country.get(iso3)
        if prev is None or year > prev[0]:
            latest_by_country[iso3] = (year, value)

    if not latest_by_country:
        return None, {}

    latest_year = max(item[0] for item in latest_by_country.values())
    values = {
        iso3: val
        for iso3, (year, val) in latest_by_country.items()
        if year == latest_year
    }
    return latest_year, values


def _scaled_trade_dependencies_from_indicator(
    base: dict[str, dict[str, float]],
    trade_pct_by_iso3: dict[str, float],
) -> dict[str, dict[str, float]]:
    # NE.TRD.GNFS.ZS is Trade (% of GDP). Convert to scale factor.
    base_divisor = float(
        max(
            1.0,
            getattr(settings, "WORLD_INTEL_TRADE_DEPENDENCY_BASE_DIVISOR", 120.0) or 120.0,
        )
    )
    min_factor = float(
        max(
            0.1,
            getattr(settings, "WORLD_INTEL_TRADE_DEPENDENCY_MIN_FACTOR", 0.5) or 0.5,
        )
    )
    max_factor = float(
        max(
            min_factor,
            getattr(settings, "WORLD_INTEL_TRADE_DEPENDENCY_MAX_FACTOR", 1.5) or 1.5,
        )
    )

    scaled: dict[str, dict[str, float]] = {}
    for iso3, deps in base.items():
        trade_pct = trade_pct_by_iso3.get(iso3)
        if trade_pct is None:
            factor = 1.0
        else:
            factor = max(min_factor, min(max_factor, float(trade_pct) / base_divisor))
        out: dict[str, float] = {}
        for node, value in deps.items():
            out[str(node)] = max(0.0, min(1.0, round(float(value) * factor, 4)))
        if out:
            scaled[iso3] = out
    return scaled


async def load_trade_dependencies_from_db(session: AsyncSession) -> dict[str, Any]:
    row = await _get_or_create_app_settings(session)
    deps = _clean_trade_dependencies(getattr(row, "world_intel_trade_dependencies_json", None))
    if not deps:
        deps = _default_trade_dependencies()
        row.world_intel_trade_dependencies_json = deps
        row.world_intel_trade_dependency_source = (
            str(getattr(row, "world_intel_trade_dependency_source", "") or "").strip()
            or "static_seed"
        )
        await session.commit()
    source = str(getattr(row, "world_intel_trade_dependency_source", "") or "").strip() or "db"
    infrastructure_catalog.set_runtime_trade_dependencies(deps, source=source)
    return {
        "source": source,
        "countries": len(deps),
        "indicator_year": (
            int(row.world_intel_trade_dependency_year)
            if getattr(row, "world_intel_trade_dependency_year", None) is not None
            else None
        ),
        "last_synced_at": (
            row.world_intel_trade_dependency_synced_at.isoformat()
            if isinstance(row.world_intel_trade_dependency_synced_at, datetime)
            else None
        ),
    }


async def sync_trade_dependencies_from_world_bank(
    session: AsyncSession,
    *,
    force: bool = False,
) -> dict[str, Any]:
    row = await _get_or_create_app_settings(session)
    existing = _clean_trade_dependencies(getattr(row, "world_intel_trade_dependencies_json", None))

    enabled = bool(getattr(settings, "WORLD_INTEL_TRADE_DEPENDENCY_SYNC_ENABLED", True))
    interval_hours = int(
        max(1, getattr(settings, "WORLD_INTEL_TRADE_DEPENDENCY_SYNC_HOURS", 24) or 24)
    )
    last_sync = getattr(row, "world_intel_trade_dependency_synced_at", None)

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
        source = str(getattr(row, "world_intel_trade_dependency_source", "") or "").strip() or "db"
        infrastructure_catalog.set_runtime_trade_dependencies(existing, source=source)
        return {
            "updated": False,
            "reason": "fresh",
            "source": source,
            "countries": len(existing),
            "indicator_year": (
                int(row.world_intel_trade_dependency_year)
                if getattr(row, "world_intel_trade_dependency_year", None) is not None
                else None
            ),
            "last_synced_at": last_sync.isoformat() if isinstance(last_sync, datetime) else None,
        }

    if not enabled and not force:
        source = str(getattr(row, "world_intel_trade_dependency_source", "") or "").strip() or "db"
        infrastructure_catalog.set_runtime_trade_dependencies(existing, source=source)
        return {
            "updated": False,
            "reason": "disabled",
            "source": source,
            "countries": len(existing),
            "indicator_year": (
                int(row.world_intel_trade_dependency_year)
                if getattr(row, "world_intel_trade_dependency_year", None) is not None
                else None
            ),
            "last_synced_at": last_sync.isoformat() if isinstance(last_sync, datetime) else None,
        }

    try:
        indicator_year, values = await _fetch_world_bank_trade_indicator()
    except Exception:
        source = str(getattr(row, "world_intel_trade_dependency_source", "") or "").strip() or "db"
        infrastructure_catalog.set_runtime_trade_dependencies(existing, source=source)
        return {
            "updated": False,
            "reason": "fetch_error",
            "source": source,
            "countries": len(existing),
            "indicator_year": (
                int(row.world_intel_trade_dependency_year)
                if getattr(row, "world_intel_trade_dependency_year", None) is not None
                else None
            ),
            "last_synced_at": last_sync.isoformat() if isinstance(last_sync, datetime) else None,
        }
    if not values:
        source = str(getattr(row, "world_intel_trade_dependency_source", "") or "").strip() or "db"
        infrastructure_catalog.set_runtime_trade_dependencies(existing, source=source)
        return {
            "updated": False,
            "reason": "empty_fetch",
            "source": source,
            "countries": len(existing),
            "indicator_year": (
                int(row.world_intel_trade_dependency_year)
                if getattr(row, "world_intel_trade_dependency_year", None) is not None
                else None
            ),
            "last_synced_at": last_sync.isoformat() if isinstance(last_sync, datetime) else None,
        }

    base = _default_trade_dependencies()
    scaled = _scaled_trade_dependencies_from_indicator(base, values)
    if not scaled:
        scaled = base

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    row.world_intel_trade_dependencies_json = scaled
    row.world_intel_trade_dependency_source = "world_bank_ne_trd_gnfs_zs"
    row.world_intel_trade_dependency_year = int(indicator_year) if indicator_year is not None else None
    row.world_intel_trade_dependency_synced_at = now
    await session.commit()

    infrastructure_catalog.set_runtime_trade_dependencies(
        scaled,
        source="world_bank_ne_trd_gnfs_zs",
    )
    return {
        "updated": True,
        "reason": "synced",
        "source": "world_bank_ne_trd_gnfs_zs",
        "countries": len(scaled),
        "indicator_year": int(indicator_year) if indicator_year is not None else None,
        "last_synced_at": now.isoformat(),
    }


async def get_trade_dependency_source_status(session: AsyncSession) -> dict[str, Any]:
    row = await _get_or_create_app_settings(session)
    deps = _clean_trade_dependencies(getattr(row, "world_intel_trade_dependencies_json", None))
    source = str(getattr(row, "world_intel_trade_dependency_source", "") or "").strip() or "db"
    year = getattr(row, "world_intel_trade_dependency_year", None)
    last_sync = getattr(row, "world_intel_trade_dependency_synced_at", None)
    return {
        "source": infrastructure_catalog.runtime_trade_source() or source,
        "countries": len(deps),
        "indicator_year": int(year) if year is not None else None,
        "last_synced_at": last_sync.isoformat() if isinstance(last_sync, datetime) else None,
        "sync_enabled": bool(getattr(settings, "WORLD_INTEL_TRADE_DEPENDENCY_SYNC_ENABLED", True)),
        "sync_interval_hours": int(
            max(1, getattr(settings, "WORLD_INTEL_TRADE_DEPENDENCY_SYNC_HOURS", 24) or 24)
        ),
    }
