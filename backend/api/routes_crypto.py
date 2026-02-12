"""Crypto routes backed by worker snapshots with live-source fallback."""

from __future__ import annotations

import asyncio
import math
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import get_db_session
from services.chainlink_feed import get_chainlink_feed
from services.crypto_service import get_crypto_service
from services.worker_state import read_worker_snapshot
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

_SNAPSHOT_FRESH_MAX_AGE_SECONDS = 10.0


def _to_float(value: object) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _to_dict_list(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _snapshot_age_seconds(snapshot: dict) -> Optional[float]:
    raw = snapshot.get("updated_at")
    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    age = (now - dt.astimezone(timezone.utc)).total_seconds()
    return max(0.0, age)


def _snapshot_markets(snapshot: dict) -> list[dict]:
    stats = snapshot.get("stats")
    if not isinstance(stats, dict):
        return []
    return _to_dict_list(stats.get("markets"))


def _is_snapshot_fresh(snapshot: dict) -> bool:
    age = _snapshot_age_seconds(snapshot)
    return age is not None and age <= _SNAPSHOT_FRESH_MAX_AGE_SECONDS


def _hydrate_price_to_beat_cache_from_snapshot(
    snapshot_markets: list[dict],
) -> dict[str, float]:
    svc = get_crypto_service()
    hydrated: dict[str, float] = {}
    for row in snapshot_markets:
        slug = str(row.get("slug") or "").strip()
        if not slug:
            continue
        ptb = _to_float(row.get("price_to_beat"))
        if ptb is None:
            continue
        svc._price_to_beat.setdefault(slug, ptb)
        hydrated[slug] = ptb
    return hydrated


def _oracle_history_payload(feed, asset: str) -> list[dict]:
    history = feed._history.get(asset) if hasattr(feed, "_history") else None
    if not history:
        return []
    points = list(history)
    if len(points) > 80:
        step = max(1, len(points) // 80)
        points = points[::step]
    last = history[-1]
    if points and points[-1] != last:
        points.append(last)
    return [{"t": int(t), "p": round(float(p), 2)} for t, p in points]


async def _build_live_markets_from_source(snapshot_markets: list[dict]) -> list[dict]:
    svc = get_crypto_service()
    snapshot_ptb = _hydrate_price_to_beat_cache_from_snapshot(snapshot_markets)

    try:
        markets = await asyncio.to_thread(svc.get_live_markets, True)
    except Exception as exc:
        logger.warning("Crypto source fallback fetch failed", error=str(exc))
        return []

    if not markets:
        return []

    try:
        feed = get_chainlink_feed()
        if not feed.started:
            await feed.start()
    except Exception as exc:
        feed = None
        logger.warning("Crypto source fallback feed start failed", error=str(exc))

    try:
        svc._update_price_to_beat(markets)
    except Exception as exc:
        logger.debug("Crypto price-to-beat refresh failed", error=str(exc))

    now_ms = int(time.time() * 1000)
    result: list[dict] = []
    for market in markets:
        row = market.to_dict()

        oracle = feed.get_price(market.asset) if feed else None
        if oracle:
            row["oracle_price"] = oracle.price
            row["oracle_updated_at_ms"] = oracle.updated_at_ms
            row["oracle_age_seconds"] = (
                round((now_ms - oracle.updated_at_ms) / 1000, 1)
                if oracle.updated_at_ms
                else None
            )
        else:
            row["oracle_price"] = None
            row["oracle_updated_at_ms"] = None
            row["oracle_age_seconds"] = None

        slug = market.slug
        row["price_to_beat"] = svc._price_to_beat.get(slug) or snapshot_ptb.get(slug)
        row["oracle_history"] = _oracle_history_payload(feed, market.asset) if feed else []
        result.append(row)

    return result


@router.get("/crypto/markets")
async def get_crypto_markets(session: AsyncSession = Depends(get_db_session)):
    """Return live crypto markets with stale-snapshot source fallback."""
    snapshot = await read_worker_snapshot(session, "crypto")
    markets = _snapshot_markets(snapshot)

    if markets and _is_snapshot_fresh(snapshot):
        return markets

    live = await _build_live_markets_from_source(markets)
    if live:
        return live

    return markets


@router.get("/crypto/oracle-prices")
async def get_oracle_prices(session: AsyncSession = Depends(get_db_session)):
    """Return latest oracle prices derived from crypto market payload."""
    snapshot = await read_worker_snapshot(session, "crypto")
    markets = _snapshot_markets(snapshot)

    if not markets or not _is_snapshot_fresh(snapshot):
        live = await _build_live_markets_from_source(markets)
        if live:
            markets = live

    out: dict[str, dict] = {}
    for market in markets:
        asset = (market or {}).get("asset")
        if not asset:
            continue
        out[str(asset)] = {
            "price": (market or {}).get("oracle_price"),
            "updated_at_ms": (market or {}).get("oracle_updated_at_ms"),
            "age_seconds": (market or {}).get("oracle_age_seconds"),
        }
    return out
