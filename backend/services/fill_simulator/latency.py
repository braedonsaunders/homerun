"""Bridge from ExecutionLatencyMetrics into the fill estimator.

Replaces the hardcoded ``latency_ms=350.0`` in
``services/trader_orchestrator/order_manager.py`` and the equivalent
constant in the ExecutionEstimator with the rolling p50/p95/p99
measured across the last 15 minutes of orders.

The metrics singleton is fully sync-safe to read from snapshot() —
which itself is async because it acquires a lock — so this helper
exposes both an async (real read) and a sync (cached) variant.

The cache uses a small TTL (5 s) so the estimator can be called
inline from the order-manager hot path without paying for an async
hop on every fill simulation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from services.execution_latency_metrics import execution_latency_metrics


_CACHE_TTL_SECONDS = 5.0
_FALLBACK_P50 = 200.0
_FALLBACK_P95 = 600.0
_FALLBACK_P99 = 1500.0

# Operator-overridable fallback values, refreshed from AppSettings on a
# 30 s TTL.  Read both inline (sync, from the cached values) and async
# (refresh hook).  Falls back to the module constants when the settings
# row is absent or the columns are NULL — matches existing behavior on
# fresh deployments.
_OVERRIDE_TTL_SECONDS = 30.0
_overrides: tuple[float, float, float, float] = (
    0.0, _FALLBACK_P50, _FALLBACK_P95, _FALLBACK_P99,
)


def _current_fallbacks() -> tuple[float, float, float]:
    """Return the active (p50, p95, p99) fallback values.

    Reads from the cached AppSettings overrides if fresh, else from
    the module constants.  Refresh happens out-of-band via
    ``refresh_fallback_overrides()``.
    """
    ts, p50, p95, p99 = _overrides
    if ts <= 0:
        return _FALLBACK_P50, _FALLBACK_P95, _FALLBACK_P99
    if time.monotonic() - ts > _OVERRIDE_TTL_SECONDS:
        # Stale — keep using the values but signal a refresh on next
        # async hop.  Not a hard error; the values are sane defaults.
        pass
    return p50, p95, p99


async def refresh_fallback_overrides() -> None:
    """Pull the latest AppSettings row and cache its latency fallbacks.

    Called from ``measured_latency_async()`` opportunistically; safe
    to call directly from a background task.  Silently no-ops if the
    settings row doesn't exist or the columns are NULL.
    """
    global _overrides
    try:
        from sqlalchemy import select
        from models.database import AsyncSessionLocal, AppSettings

        async with AsyncSessionLocal() as session:
            row = (
                await session.execute(select(AppSettings).limit(1))
            ).scalar_one_or_none()
            if row is None:
                return
            p50 = getattr(row, "latency_fallback_p50_ms", None)
            p95 = getattr(row, "latency_fallback_p95_ms", None)
            p99 = getattr(row, "latency_fallback_p99_ms", None)
            _overrides = (
                time.monotonic(),
                float(p50) if isinstance(p50, (int, float)) and p50 > 0 else _FALLBACK_P50,
                float(p95) if isinstance(p95, (int, float)) and p95 > 0 else _FALLBACK_P95,
                float(p99) if isinstance(p99, (int, float)) and p99 > 0 else _FALLBACK_P99,
            )
    except Exception:
        # Settings table may not be migrated yet on a fresh DB; fall
        # through to module constants.
        pass


@dataclass
class LatencyDistribution:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    sample_count: int

    @property
    def realistic_ms(self) -> float:
        return self.p50_ms

    @property
    def pessimistic_ms(self) -> float:
        return self.p95_ms

    @property
    def optimistic_ms(self) -> float:
        # No measured p10 in the buffer; conservatively half the p50
        # rather than min, since the absolute floor of any sample is
        # a noise spike (e.g. a clock-skew zero), not a representative
        # best case.
        return max(20.0, self.p50_ms * 0.5)


_cached: tuple[float, LatencyDistribution | None] = (0.0, None)


def _from_snapshot(snapshot: dict[str, Any]) -> LatencyDistribution:
    overall = snapshot.get("overall") if isinstance(snapshot, dict) else None
    overall = overall if isinstance(overall, dict) else {}
    rt = overall.get("submit_round_trip_ms") if isinstance(overall, dict) else None
    rt = rt if isinstance(rt, dict) else {}
    p50 = rt.get("p50")
    p95 = rt.get("p95")
    p99 = rt.get("p99")
    fp50, fp95, fp99 = _current_fallbacks()
    return LatencyDistribution(
        p50_ms=float(p50) if isinstance(p50, (int, float)) else fp50,
        p95_ms=float(p95) if isinstance(p95, (int, float)) else fp95,
        p99_ms=float(p99) if isinstance(p99, (int, float)) else fp99,
        sample_count=int(snapshot.get("sample_count") or 0),
    )


async def measured_latency_async() -> LatencyDistribution:
    # Refresh the operator overrides opportunistically — once per call
    # is fine because the cache itself has a 30s TTL.
    await refresh_fallback_overrides()
    snapshot = await execution_latency_metrics.snapshot()
    dist = _from_snapshot(snapshot)
    global _cached
    _cached = (time.monotonic(), dist)
    return dist


def measured_latency_cached() -> LatencyDistribution:
    """Sync read of the most-recent cached latency distribution.

    Returns the fallback (configurable via AppSettings, defaults
    200/600/1500 ms) if no async refresh has populated the cache yet.
    Callers on the order hot path call this in-line; a separate
    background task can call ``measured_latency_async()`` periodically
    to keep the cache warm.
    """
    now = time.monotonic()
    ts, dist = _cached
    if dist is not None and now - ts < _CACHE_TTL_SECONDS:
        return dist
    if dist is None:
        fp50, fp95, fp99 = _current_fallbacks()
        return LatencyDistribution(
            p50_ms=fp50,
            p95_ms=fp95,
            p99_ms=fp99,
            sample_count=0,
        )
    return dist
