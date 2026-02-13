"""Shared DB state for weather workflow worker/API/orchestrator."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from utils.utcnow import utcnow
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings as app_settings
from models.database import (
    AppSettings,
    WeatherControl,
    WeatherSnapshot,
    WeatherTradeIntent,
)
from models.opportunity import ArbitrageOpportunity
from services.market_tradability import get_market_tradability_map

WEATHER_SNAPSHOT_ID = "latest"
WEATHER_CONTROL_ID = "default"
MIN_TIME_TO_RESOLUTION = timedelta(minutes=30)


def _parse_iso_datetime(value: str) -> datetime:
    text = value.strip()
    if text.endswith("+00:00+00:00"):
        text = text[:-6]
    if text.endswith("Z"):
        text = text[:-1]
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _format_iso_utc_z(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(tzinfo=None).isoformat() + "Z"


def _normalize_weather_edge_title(title: str) -> str:
    prefix = "weather edge:"
    return title[len(prefix) :].lstrip() if title.lower().startswith(prefix) else title


def _default_status() -> dict[str, Any]:
    return {
        "running": False,
        "enabled": True,
        "interval_seconds": app_settings.WEATHER_WORKFLOW_SCAN_INTERVAL_SECONDS,
        "last_scan": None,
        "opportunities_count": 0,
        "current_activity": "Waiting for weather worker.",
        "stats": {},
    }


async def write_weather_snapshot(
    session: AsyncSession,
    opportunities: list[ArbitrageOpportunity],
    status: dict[str, Any],
    stats: Optional[dict[str, Any]] = None,
) -> None:
    last_scan = status.get("last_scan")
    if isinstance(last_scan, str):
        try:
            last_scan = _parse_iso_datetime(last_scan)
        except Exception:
            last_scan = utcnow()
    elif last_scan is None:
        last_scan = utcnow()

    payload: list[dict[str, Any]] = []
    for o in opportunities:
        if hasattr(o, "model_dump"):
            payload.append(o.model_dump(mode="json"))
        else:
            payload.append(ArbitrageOpportunity.model_validate(o).model_dump(mode="json"))

    result = await session.execute(
        select(WeatherSnapshot).where(WeatherSnapshot.id == WEATHER_SNAPSHOT_ID)
    )
    row = result.scalar_one_or_none()
    if row is None:
        row = WeatherSnapshot(id=WEATHER_SNAPSHOT_ID)
        session.add(row)

    row.updated_at = utcnow()
    row.last_scan_at = last_scan
    row.opportunities_json = payload
    row.running = status.get("running", True)
    row.enabled = status.get("enabled", True)
    row.current_activity = status.get("current_activity")
    row.interval_seconds = status.get(
        "interval_seconds", app_settings.WEATHER_WORKFLOW_SCAN_INTERVAL_SECONDS
    )
    row.stats_json = stats if stats is not None else status.get("stats", {})
    await session.commit()


async def read_weather_snapshot(
    session: AsyncSession,
) -> tuple[list[ArbitrageOpportunity], dict[str, Any]]:
    result = await session.execute(
        select(WeatherSnapshot).where(WeatherSnapshot.id == WEATHER_SNAPSHOT_ID)
    )
    row = result.scalar_one_or_none()
    if row is None:
        return [], _default_status()

    opportunities: list[ArbitrageOpportunity] = []
    for d in row.opportunities_json or []:
        try:
            opportunities.append(ArbitrageOpportunity.model_validate(d))
        except Exception:
            continue

    status = {
        "running": row.running,
        "enabled": row.enabled,
        "interval_seconds": row.interval_seconds,
        "last_scan": _format_iso_utc_z(row.last_scan_at),
        "opportunities_count": len(opportunities),
        "current_activity": row.current_activity,
        "stats": row.stats_json or {},
    }
    return opportunities, status


async def get_weather_opportunities_from_db(
    session: AsyncSession,
    min_edge_percent: Optional[float] = None,
    direction: Optional[str] = None,
    max_entry_price: Optional[float] = None,
    location_query: Optional[str] = None,
    require_tradable_markets: bool = False,
    exclude_near_resolution: bool = False,
) -> list[ArbitrageOpportunity]:
    opportunities, _ = await read_weather_snapshot(session)
    for opp in opportunities:
        opp.title = _normalize_weather_edge_title(opp.title)

    if opportunities and exclude_near_resolution:
        now = datetime.now(timezone.utc)
        filtered: list[ArbitrageOpportunity] = []
        for opp in opportunities:
            if opp.resolution_date is None:
                filtered.append(opp)
                continue
            rd = (
                opp.resolution_date
                if opp.resolution_date.tzinfo is not None
                else opp.resolution_date.replace(tzinfo=timezone.utc)
            )
            if rd <= (now + MIN_TIME_TO_RESOLUTION):
                continue
            filtered.append(opp)
        opportunities = filtered

    if min_edge_percent is not None:
        opportunities = [o for o in opportunities if o.roi_percent >= min_edge_percent]

    if direction:
        dir_lower = direction.lower()
        filtered = []
        for o in opportunities:
            pos = (o.positions_to_take or [{}])[0]
            outcome = str(pos.get("outcome", "")).lower()
            if dir_lower in ("yes", "buy_yes") and outcome == "yes":
                filtered.append(o)
            elif dir_lower in ("no", "buy_no") and outcome == "no":
                filtered.append(o)
        opportunities = filtered

    if max_entry_price is not None:
        filtered: list[ArbitrageOpportunity] = []
        for opp in opportunities:
            positions = opp.positions_to_take or []
            # Keep report-only rows (no executable leg) visible in weather UI.
            if not positions:
                filtered.append(opp)
                continue
            try:
                price = float(positions[0].get("price", 1.0))
            except Exception:
                continue
            if price <= max_entry_price:
                filtered.append(opp)
        opportunities = filtered

    if location_query:
        q = location_query.lower()
        opportunities = [
            o
            for o in opportunities
            if any(q in str(m.get("weather", {}).get("location", "")).lower() for m in o.markets)
            or q in o.title.lower()
            or q in o.description.lower()
        ]

    if opportunities and require_tradable_markets:
        market_ids: set[str] = set()
        by_index: dict[int, list[str]] = {}
        for idx, opp in enumerate(opportunities):
            mids: list[str] = []
            seen: set[str] = set()
            for market in opp.markets or []:
                if not isinstance(market, dict):
                    continue
                mid = str(market.get("id") or market.get("condition_id") or "").strip().lower()
                if not mid or mid in seen:
                    continue
                seen.add(mid)
                mids.append(mid)
                market_ids.add(mid)
            by_index[idx] = mids

        if market_ids:
            tradability = await get_market_tradability_map(market_ids)
            opportunities = [
                opp
                for idx, opp in enumerate(opportunities)
                if all(tradability.get(mid, True) for mid in by_index.get(idx, []))
            ]

    return opportunities


async def get_weather_status_from_db(session: AsyncSession) -> dict[str, Any]:
    _, status = await read_weather_snapshot(session)
    return status


async def ensure_weather_control(session: AsyncSession) -> WeatherControl:
    result = await session.execute(
        select(WeatherControl).where(WeatherControl.id == WEATHER_CONTROL_ID)
    )
    row = result.scalar_one_or_none()
    if row is None:
        row = WeatherControl(id=WEATHER_CONTROL_ID)
        session.add(row)
        await session.commit()
        await session.refresh(row)
    return row


async def read_weather_control(session: AsyncSession) -> dict[str, Any]:
    result = await session.execute(
        select(WeatherControl).where(WeatherControl.id == WEATHER_CONTROL_ID)
    )
    row = result.scalar_one_or_none()
    if row is None:
        return {
            "is_enabled": True,
            "is_paused": False,
            "scan_interval_seconds": app_settings.WEATHER_WORKFLOW_SCAN_INTERVAL_SECONDS,
            "requested_scan_at": None,
        }
    return {
        "is_enabled": row.is_enabled,
        "is_paused": row.is_paused,
        "scan_interval_seconds": row.scan_interval_seconds,
        "requested_scan_at": row.requested_scan_at,
    }


async def set_weather_paused(session: AsyncSession, paused: bool) -> None:
    row = await ensure_weather_control(session)
    row.is_paused = paused
    row.updated_at = utcnow()
    await session.commit()


async def set_weather_interval(session: AsyncSession, interval_seconds: int) -> None:
    row = await ensure_weather_control(session)
    row.scan_interval_seconds = max(300, min(86400, interval_seconds))
    row.updated_at = utcnow()
    await session.commit()


async def request_one_weather_scan(session: AsyncSession) -> None:
    row = await ensure_weather_control(session)
    row.requested_scan_at = utcnow()
    row.updated_at = utcnow()
    await session.commit()


async def clear_weather_scan_request(session: AsyncSession) -> None:
    row = await ensure_weather_control(session)
    row.requested_scan_at = None
    row.updated_at = utcnow()
    await session.commit()


async def upsert_weather_intent(session: AsyncSession, intent: dict[str, Any]) -> None:
    result = await session.execute(
        select(WeatherTradeIntent).where(WeatherTradeIntent.id == intent["id"])
    )
    row = result.scalar_one_or_none()
    if row is None:
        row = WeatherTradeIntent(**intent)
        session.add(row)
    else:
        # Do not overwrite consumed intents; keep only fresh updates for pending/submitted.
        if row.status in {"pending", "submitted"}:
            for key, value in intent.items():
                setattr(row, key, value)


async def list_weather_intents(
    session: AsyncSession,
    status_filter: Optional[str] = None,
    limit: int = 100,
) -> list[WeatherTradeIntent]:
    query = select(WeatherTradeIntent).order_by(WeatherTradeIntent.created_at.desc())
    if status_filter:
        query = query.where(WeatherTradeIntent.status == status_filter)
    query = query.limit(limit)
    result = await session.execute(query)
    rows = list(result.scalars().all())

    actionable = [r for r in rows if r.status in {"pending", "submitted"} and r.market_id]
    if actionable:
        tradability = await get_market_tradability_map([str(r.market_id) for r in actionable])
        now = utcnow()
        changed = 0
        for row in actionable:
            if tradability.get(str(row.market_id).strip().lower(), True):
                continue
            row.status = "expired"
            row.consumed_at = now
            changed += 1
        if changed:
            await session.commit()
            if status_filter in {"pending", "submitted"}:
                rows = [r for r in rows if r.status == status_filter]

    return rows


async def mark_weather_intent(
    session: AsyncSession,
    intent_id: str,
    status: str,
) -> bool:
    result = await session.execute(
        select(WeatherTradeIntent).where(WeatherTradeIntent.id == intent_id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        return False
    row.status = status
    row.consumed_at = datetime.now(timezone.utc)
    await session.commit()
    return True


def _stable_id_from_opportunity_id(opportunity_id: Optional[str]) -> Optional[str]:
    if not opportunity_id:
        return None
    text = str(opportunity_id).strip()
    if not text:
        return None
    parts = text.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return text


async def update_weather_opportunity_ai_analysis_in_snapshot(
    session: AsyncSession,
    opportunity_id: str,
    stable_id: Optional[str],
    ai_analysis: dict[str, Any],
) -> bool:
    """Persist ai_analysis on a weather opportunity inside weather_snapshot."""
    sid = (stable_id or "").strip() or _stable_id_from_opportunity_id(opportunity_id)
    oid = (opportunity_id or "").strip()
    if not oid and not sid:
        return False

    result = await session.execute(
        select(WeatherSnapshot).where(WeatherSnapshot.id == WEATHER_SNAPSHOT_ID)
    )
    row = result.scalar_one_or_none()
    if row is None or not isinstance(row.opportunities_json, list):
        return False

    updated = False
    patched_payload: list[dict[str, Any]] = []
    for item in row.opportunities_json:
        if not isinstance(item, dict):
            patched_payload.append(item)
            continue
        item_id = str(item.get("id") or "").strip()
        item_sid = str(item.get("stable_id") or "").strip()
        if (oid and item_id == oid) or (sid and item_sid == sid):
            patched = dict(item)
            patched["ai_analysis"] = ai_analysis
            patched_payload.append(patched)
            updated = True
        else:
            patched_payload.append(item)

    if not updated:
        return False

    row.opportunities_json = patched_payload
    row.updated_at = utcnow()
    await session.commit()
    return True


async def _get_or_create_app_settings(session: AsyncSession) -> AppSettings:
    result = await session.execute(select(AppSettings).where(AppSettings.id == "default"))
    db = result.scalar_one_or_none()
    if db is None:
        db = AppSettings(id="default")
        session.add(db)
        await session.commit()
        await session.refresh(db)
    return db


async def get_weather_settings(session: AsyncSession) -> dict[str, Any]:
    db = await _get_or_create_app_settings(session)
    return {
        "enabled": bool(getattr(db, "weather_workflow_enabled", True)),
        "auto_run": bool(getattr(db, "weather_workflow_auto_run", True)),
        "scan_interval_seconds": int(
            getattr(
                db,
                "weather_workflow_scan_interval_seconds",
                app_settings.WEATHER_WORKFLOW_SCAN_INTERVAL_SECONDS,
            )
        ),
        "entry_max_price": float(
            getattr(db, "weather_workflow_entry_max_price", app_settings.WEATHER_WORKFLOW_ENTRY_MAX_PRICE)
        ),
        "take_profit_price": float(
            getattr(
                db,
                "weather_workflow_take_profit_price",
                app_settings.WEATHER_WORKFLOW_TAKE_PROFIT_PRICE,
            )
        ),
        "stop_loss_pct": float(
            getattr(db, "weather_workflow_stop_loss_pct", app_settings.WEATHER_WORKFLOW_STOP_LOSS_PCT)
        ),
        "min_edge_percent": float(
            getattr(
                db,
                "weather_workflow_min_edge_percent",
                app_settings.WEATHER_WORKFLOW_MIN_EDGE_PERCENT,
            )
        ),
        "min_confidence": float(
            getattr(
                db,
                "weather_workflow_min_confidence",
                app_settings.WEATHER_WORKFLOW_MIN_CONFIDENCE,
            )
        ),
        "min_model_agreement": float(
            getattr(
                db,
                "weather_workflow_min_model_agreement",
                app_settings.WEATHER_WORKFLOW_MIN_MODEL_AGREEMENT,
            )
        ),
        "min_liquidity": float(
            getattr(db, "weather_workflow_min_liquidity", app_settings.WEATHER_WORKFLOW_MIN_LIQUIDITY)
        ),
        "max_markets_per_scan": int(
            getattr(
                db,
                "weather_workflow_max_markets_per_scan",
                app_settings.WEATHER_WORKFLOW_MAX_MARKETS_PER_SCAN,
            )
        ),
        "orchestrator_enabled": bool(
            getattr(db, "weather_workflow_orchestrator_enabled", True)
        ),
        "orchestrator_min_edge": float(
            getattr(db, "weather_workflow_orchestrator_min_edge", 10.0)
        ),
        "orchestrator_max_age_minutes": int(
            getattr(db, "weather_workflow_orchestrator_max_age_minutes", 240)
        ),
        "default_size_usd": float(
            getattr(db, "weather_workflow_default_size_usd", app_settings.WEATHER_WORKFLOW_DEFAULT_SIZE_USD)
        ),
        "max_size_usd": float(
            getattr(db, "weather_workflow_max_size_usd", app_settings.WEATHER_WORKFLOW_MAX_SIZE_USD)
        ),
        "model": getattr(db, "weather_workflow_model", None),
    }


async def update_weather_settings(
    session: AsyncSession,
    updates: dict[str, Any],
) -> dict[str, Any]:
    db = await _get_or_create_app_settings(session)

    mapping = {
        "enabled": "weather_workflow_enabled",
        "auto_run": "weather_workflow_auto_run",
        "scan_interval_seconds": "weather_workflow_scan_interval_seconds",
        "entry_max_price": "weather_workflow_entry_max_price",
        "take_profit_price": "weather_workflow_take_profit_price",
        "stop_loss_pct": "weather_workflow_stop_loss_pct",
        "min_edge_percent": "weather_workflow_min_edge_percent",
        "min_confidence": "weather_workflow_min_confidence",
        "min_model_agreement": "weather_workflow_min_model_agreement",
        "min_liquidity": "weather_workflow_min_liquidity",
        "max_markets_per_scan": "weather_workflow_max_markets_per_scan",
        "orchestrator_enabled": "weather_workflow_orchestrator_enabled",
        "orchestrator_min_edge": "weather_workflow_orchestrator_min_edge",
        "orchestrator_max_age_minutes": "weather_workflow_orchestrator_max_age_minutes",
        "default_size_usd": "weather_workflow_default_size_usd",
        "max_size_usd": "weather_workflow_max_size_usd",
        "model": "weather_workflow_model",
    }

    for key, value in updates.items():
        if key not in mapping:
            continue
        setattr(db, mapping[key], value)

    db.updated_at = utcnow()
    await session.commit()
    return await get_weather_settings(session)
