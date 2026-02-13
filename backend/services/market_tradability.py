"""Shared market tradability guard used across opportunity/intent surfaces."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta
from typing import Iterable, Optional

from services.polymarket import polymarket_client
from utils.utcnow import utcnow

_CACHE_TTL = timedelta(minutes=3)
_CACHE_MAX_SIZE = 5000
_cache: dict[str, tuple[datetime, bool]] = {}
_POLYMARKET_CONDITION_ID_RE = re.compile(r"^0x[0-9a-f]{64}$")
_POLYMARKET_NUMERIC_TOKEN_ID_RE = re.compile(r"^\d{18,}$")
_POLYMARKET_HEX_TOKEN_ID_RE = re.compile(r"^(?:0x)?[0-9a-f]{40,}$")


def _normalize_market_id(value: object) -> str:
    return str(value or "").strip().lower()


def _is_polymarket_condition_id(value: str) -> bool:
    return bool(_POLYMARKET_CONDITION_ID_RE.fullmatch(value))


def _is_polymarket_token_id(value: str) -> bool:
    if _is_polymarket_condition_id(value):
        return False
    return bool(
        _POLYMARKET_NUMERIC_TOKEN_ID_RE.fullmatch(value)
        or _POLYMARKET_HEX_TOKEN_ID_RE.fullmatch(value)
    )


def _trim_cache(now: datetime) -> None:
    stale = [
        key
        for key, (cached_at, _) in _cache.items()
        if (now - cached_at) > _CACHE_TTL
    ]
    for key in stale:
        _cache.pop(key, None)
    if len(_cache) <= _CACHE_MAX_SIZE:
        return
    oldest = sorted(_cache.items(), key=lambda item: item[1][0])[
        : max(0, len(_cache) - _CACHE_MAX_SIZE)
    ]
    for key, _ in oldest:
        _cache.pop(key, None)


async def is_market_tradable(
    market_id: str,
    *,
    now: Optional[datetime] = None,
) -> bool:
    """Return whether market_id currently appears tradable.

    Unknown/lookup-failed markets are treated as tradable to avoid false drops.
    """
    key = _normalize_market_id(market_id)
    if not key:
        return False

    ref_now = now or utcnow()
    _trim_cache(ref_now)

    cached = _cache.get(key)
    if cached and (ref_now - cached[0]) <= _CACHE_TTL:
        return bool(cached[1])

    is_condition_id = _is_polymarket_condition_id(key)
    is_token_id = _is_polymarket_token_id(key)
    if not is_condition_id and not is_token_id:
        _cache[key] = (ref_now, True)
        return True

    info = None
    try:
        if is_condition_id:
            info = await polymarket_client.get_market_by_condition_id(key)
        else:
            info = await polymarket_client.get_market_by_token_id(key)
    except Exception:
        info = None

    tradable = (
        polymarket_client.is_market_tradable(info, now=ref_now)
        if isinstance(info, dict) and info
        else True
    )
    _cache[key] = (ref_now, bool(tradable))
    return bool(tradable)


async def get_market_tradability_map(
    market_ids: Iterable[str],
    *,
    now: Optional[datetime] = None,
    max_concurrency: int = 12,
) -> dict[str, bool]:
    """Resolve a batch of market ids -> tradability boolean."""
    ref_now = now or utcnow()
    keys = sorted(
        {
            _normalize_market_id(mid)
            for mid in market_ids
            if _normalize_market_id(mid)
        }
    )
    if not keys:
        return {}

    semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
    result: dict[str, bool] = {}

    async def _resolve(mid: str) -> None:
        async with semaphore:
            result[mid] = await is_market_tradable(mid, now=ref_now)

    await asyncio.gather(*[_resolve(mid) for mid in keys])
    return result
