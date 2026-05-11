"""Cross-mode trader→strategy binding cache.

Plan 0041 introduces per-trader strategy parameter overrides at the
signal-generation layer. The dispatcher needs to enumerate every
trader bound to a given source key (e.g. ``"crypto"``) — across BOTH
``mode='live'`` AND ``mode='shadow'`` traders — along with the
per-trader ``strategy_params`` dict the trader saved under the Tune
UI. This module is that lookup.

It is intentionally a sibling of ``services.strategies._firehose``'s
existing binding cache (which filters to ``mode='live'`` for
firehose-event suppression) rather than a generalisation of it:
firehose cares only about whether the orchestrator will eventually
consume an event, while the dispatcher must also fan out to shadow
traders. Mixing the two filters into one cache would have introduced
a state matrix that surface-bug-prone.

Refresh policy mirrors firehose's: 3 s soft-TTL (serve stale, schedule
background refresh), 30 s hard-stale ceiling (block to refresh). A
trader-config edit in the UI therefore takes effect on the dispatcher
within at most 3 s without an explicit invalidation hook.
"""

from __future__ import annotations

import asyncio
import json as _json
import time as _time
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)


_BINDING_TTL_SECONDS = 3.0
_BINDING_STALE_HARD_SECONDS = 30.0


# (source_key, slug) -> [(trader_id, strategy_params_dict), ...]
_bindings: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = {}
_bindings_cache_at: float = 0.0
_refresh_lock: asyncio.Lock | None = None
_refresh_inflight: bool = False


def _binding_cache_fresh() -> bool:
    return (_time.monotonic() - _bindings_cache_at) < _BINDING_TTL_SECONDS


def _binding_cache_hard_stale() -> bool:
    return (_time.monotonic() - _bindings_cache_at) >= _BINDING_STALE_HARD_SECONDS


async def _refresh() -> None:
    """Pull the trader→strategy binding map from the ``traders`` table.

    Only ``is_enabled = True`` traders are included. Within each trader,
    only ``source_configs_json`` entries with ``enabled != False`` are
    counted (the per-strategy enable flag, not the trader-wide one).
    """
    global _bindings, _bindings_cache_at
    try:
        from sqlalchemy import select
        from models.database import AsyncSessionLocal, Trader

        async with AsyncSessionLocal() as session:
            traders = (
                (
                    await session.execute(
                        select(Trader).where(Trader.is_enabled.is_(True))
                    )
                )
                .scalars()
                .all()
            )

        new_bindings: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = {}
        for trader in traders:
            trader_id = str(getattr(trader, "id", "") or "").strip()
            if not trader_id:
                continue
            cfgs = getattr(trader, "source_configs_json", None) or []
            if isinstance(cfgs, str):
                try:
                    cfgs = _json.loads(cfgs)
                except Exception:
                    cfgs = []
            if not isinstance(cfgs, list):
                continue
            for cfg in cfgs:
                if not isinstance(cfg, dict):
                    continue
                if not cfg.get("enabled", True):
                    continue
                source_key = str(cfg.get("source_key") or "").strip().lower()
                slug = str(cfg.get("strategy_key") or "").strip().lower()
                if not source_key or not slug:
                    continue
                params = cfg.get("strategy_params") or {}
                if not isinstance(params, dict):
                    params = {}
                new_bindings.setdefault((source_key, slug), []).append(
                    (trader_id, dict(params))
                )

        _bindings = new_bindings
        _bindings_cache_at = _time.monotonic()
    except Exception as exc:
        logger.debug("trader_binding_cache refresh failed", exc_info=exc)


async def _refresh_guarded() -> None:
    """Refresh under the inflight latch so concurrent callers don't pile on."""
    global _refresh_inflight
    if _refresh_inflight:
        return
    _refresh_inflight = True
    try:
        await _refresh()
    finally:
        _refresh_inflight = False


async def _ensure_fresh() -> None:
    global _refresh_lock
    if _binding_cache_fresh():
        return
    if _refresh_lock is None:
        _refresh_lock = asyncio.Lock()
    # Hard-stale -> block to refresh so we never serve a wildly outdated
    # binding map (e.g. first call after process start with no prior
    # refresh, or DB outage exceeded soft TTL window).
    if _binding_cache_hard_stale():
        async with _refresh_lock:
            if _binding_cache_fresh():
                return
            await _refresh()
        return
    # Soft-stale -> serve cached, schedule background refresh.
    if _refresh_inflight:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(_refresh_guarded(), name="trader-binding-cache-refresh")


async def get_bindings_for_source(
    source_key: str,
) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    """Return ``{slug → [(trader_id, strategy_params), ...]}`` for ``source_key``.

    Empty mapping when no enabled traders are bound to that source. The
    returned dict is a defensive shallow copy — callers may mutate it
    freely. Inner ``strategy_params`` dicts are also copies.
    """
    await _ensure_fresh()
    normalized = str(source_key or "").strip().lower()
    out: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for (src, slug), bindings in _bindings.items():
        if src != normalized:
            continue
        out[slug] = [(tid, dict(params)) for tid, params in bindings]
    return out


def invalidate() -> None:
    """Force the next ``_ensure_fresh`` to do a full DB read.

    Call from the trader-update route once a ``source_configs_json``
    mutation lands, so the per-trader gate change is observable
    immediately rather than after the 3 s soft-TTL.
    """
    global _bindings_cache_at
    _bindings_cache_at = 0.0


def snapshot_for_diagnostics() -> dict[str, Any]:
    """Read-only snapshot of the cache state for status endpoints / tests."""
    return {
        "age_seconds": max(0.0, _time.monotonic() - _bindings_cache_at)
        if _bindings_cache_at > 0
        else None,
        "binding_count": sum(len(v) for v in _bindings.values()),
        "unique_keys": len(_bindings),
        "fresh": _binding_cache_fresh(),
        "hard_stale": _binding_cache_hard_stale(),
    }
