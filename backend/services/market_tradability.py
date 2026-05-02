"""Shared market tradability guard used across opportunity/intent surfaces."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta
from typing import Iterable, Optional

from services.polymarket import polymarket_client
from utils.utcnow import utcnow
from utils.converters import normalize_market_id

_CACHE_TTL = timedelta(minutes=3)
_CACHE_MAX_SIZE = 5000
_cache: dict[str, tuple[datetime, bool]] = {}

# Global concurrency cap across ALL get_market_tradability_map callers.
# The function is invoked from 4 unrelated paths (scanner shared_state,
# news shared_state, signal_bus, tracked_traders_worker) each defaulting
# to per-call concurrency=12.  Without this cap, 2-3 concurrent calls
# spawn 24-36 simultaneous Polymarket HTTP fetches, observed during a
# crypto-backtest run as ``polymarket._fetch:722 x12 + _fetch:720 x10``
# parked on the event loop.  The semaphore is created lazily on the
# first call to bind it to the active asyncio loop (modules can be
# imported before any loop exists).
_GLOBAL_CONCURRENCY_LIMIT = 12
_global_semaphore: Optional[asyncio.Semaphore] = None


def _get_global_semaphore() -> asyncio.Semaphore:
    global _global_semaphore
    if _global_semaphore is None:
        _global_semaphore = asyncio.Semaphore(_GLOBAL_CONCURRENCY_LIMIT)
    return _global_semaphore
_POLYMARKET_CONDITION_ID_RE = re.compile(r"^0x[0-9a-f]{64}$")
_POLYMARKET_NUMERIC_TOKEN_ID_RE = re.compile(r"^\d{18,}$")
_POLYMARKET_HEX_TOKEN_ID_RE = re.compile(r"^(?:0x)?[0-9a-f]{40,}$")


def _is_polymarket_condition_id(value: str) -> bool:
    return bool(_POLYMARKET_CONDITION_ID_RE.fullmatch(value))


def _is_polymarket_token_id(value: str) -> bool:
    if _is_polymarket_condition_id(value):
        return False
    return bool(_POLYMARKET_NUMERIC_TOKEN_ID_RE.fullmatch(value) or _POLYMARKET_HEX_TOKEN_ID_RE.fullmatch(value))


def _trim_cache(now: datetime) -> None:
    stale = [key for key, (cached_at, _) in _cache.items() if (now - cached_at) > _CACHE_TTL]
    for key in stale:
        _cache.pop(key, None)
    if len(_cache) <= _CACHE_MAX_SIZE:
        return
    oldest = sorted(_cache.items(), key=lambda item: item[1][0])[: max(0, len(_cache) - _CACHE_MAX_SIZE)]
    for key, _ in oldest:
        _cache.pop(key, None)


async def _lookup_market_info(*, market_id: str, is_condition_id: bool) -> Optional[dict]:
    if is_condition_id:
        lookup = polymarket_client.get_market_by_condition_id
    else:
        lookup = polymarket_client.get_market_by_token_id

    # Prefer fresh Gamma metadata so resolved/disputed flags cannot be
    # masked by stale in-memory/persistent cache rows.
    try:
        return await lookup(market_id, force_refresh=True)
    except TypeError:
        # Backward-compatible fallback for tests/mocks that do not accept kwargs.
        return await lookup(market_id)


async def is_market_tradable(
    market_id: str,
    *,
    now: Optional[datetime] = None,
) -> bool:
    """Return whether market_id currently appears tradable.

    Unknown/lookup-failed markets are treated as tradable to avoid false drops.
    """
    key = normalize_market_id(market_id)
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
        info = await _lookup_market_info(
            market_id=key,
            is_condition_id=is_condition_id,
        )
    except Exception:
        info = None

    tradable = polymarket_client.is_market_tradable(info, now=ref_now) if isinstance(info, dict) and info else True
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
    keys = sorted({normalize_market_id(mid) for mid in market_ids if normalize_market_id(mid)})
    if not keys:
        return {}

    # Bounded worker pool: was ``[task for x in keys] + asyncio.gather(*tasks)``
    # with a Semaphore.  Even at concurrency=12, ALL N tasks lived in
    # the asyncio task registry — production saw 143 parked tasks for
    # one batch.  Worker-pool keeps live count at exactly N regardless
    # of input size.  See ``services/wallet_discovery.py``'s
    # ``_run_with_bounded_workers`` for the same pattern.
    queue: asyncio.Queue = asyncio.Queue()
    for mid in keys:
        queue.put_nowait(mid)
    result: dict[str, bool] = {}

    sem = _get_global_semaphore()

    async def _worker() -> None:
        while True:
            try:
                mid = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                # Hold the global semaphore only across the actual
                # Polymarket I/O.  Cache hits inside is_market_tradable
                # short-circuit before any HTTP call, so the semaphore
                # window is just the network leg — no point queuing
                # cache hits behind in-flight HTTP requests.
                async with sem:
                    result[mid] = await is_market_tradable(mid, now=ref_now)
            except Exception:
                pass
            finally:
                queue.task_done()
                await asyncio.sleep(0)

    workers = [
        asyncio.create_task(_worker(), name=f"market-tradability-worker-{i}")
        for i in range(max(1, int(max_concurrency)))
    ]
    try:
        await asyncio.gather(*workers, return_exceptions=True)
    finally:
        for w in workers:
            if not w.done():
                w.cancel()
    return result
