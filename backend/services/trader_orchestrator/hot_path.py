"""Contextvar-based gate that forbids Polymarket REST calls on the
hot orchestrator/trader loop.

**Architecture mandate**: opportunities are detected by ``strategy.detect()``,
subscribed to the WebSocket feed, and handed to the orchestrator with
their live state already resident in:

  * ``feed_manager.cache``               — WS price snapshots
  * ``market_cache_service``             — market metadata (cached_markets)
  * ``position_lifecycle._wallet_*_cache`` — wallet state, refreshed by
                                            ``trader_reconciliation_worker``
                                            on its own schedule

The orchestrator and fast trader read these caches.  They MUST NOT call
``polymarket_client.get_*`` directly — every such call is an HTTP round
trip that blocks the event loop, blows the per-cycle timeout, and (when
the API is slow) cascades into pool exhaustion + WS pong starvation.

Usage::

    async with hot_path_no_rest():
        await reconcile_live_positions(...)   # any inner polymarket_client
                                              # call sees the gate raised
                                              # and returns cached/empty
                                              # instead of hitting REST.

The helper ``allow_polymarket_rest_call(label)`` is the single chokepoint
every Polymarket call site checks before issuing the request.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
from typing import Iterator

logger = logging.getLogger(__name__)

# Default False: existing reconciliation worker / batch jobs that
# legitimately fetch from Polymarket are unaffected.  The orchestrator
# explicitly enters the no-REST scope per cycle.
_ORCHESTRATOR_NO_REST: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "trader_orchestrator_no_rest",
    default=False,
)

# Counter exposed for diagnostics — every call to
# ``allow_polymarket_rest_call`` while the gate is up increments this so
# we can tell whether the orchestrator is silently degrading by skipping
# expected REST calls (good — that's the architectural goal) or whether
# the cache layer is missing (bad — needs reconciliation worker fix).
_BLOCKED_CALL_COUNTERS: dict[str, int] = {}


@contextlib.contextmanager
def hot_path_no_rest() -> Iterator[None]:
    """Mark the current async scope as the orchestrator hot path.

    Any ``polymarket_client.get_*`` call site that consults
    ``allow_polymarket_rest_call`` will short-circuit to a cached/empty
    result instead of issuing the HTTP request.
    """
    token = _ORCHESTRATOR_NO_REST.set(True)
    try:
        yield
    finally:
        _ORCHESTRATOR_NO_REST.reset(token)


def is_hot_path_no_rest() -> bool:
    """True when the current task is inside ``hot_path_no_rest()``."""
    return _ORCHESTRATOR_NO_REST.get()


def allow_polymarket_rest_call(label: str) -> bool:
    """Permission check at every ``polymarket_client.X`` call site.

    Returns ``True`` when the call is allowed (default).  Returns
    ``False`` when the orchestrator hot path scope is active — in that
    case the caller MUST return cached/empty data instead of issuing
    the REST request.

    ``label`` identifies the call site (e.g. ``"closed_positions"``)
    for the blocked-call counter so we can spot which sites the
    orchestrator hits frequently.
    """
    if not _ORCHESTRATOR_NO_REST.get():
        return True
    _BLOCKED_CALL_COUNTERS[label] = _BLOCKED_CALL_COUNTERS.get(label, 0) + 1
    return False


def get_blocked_call_counters() -> dict[str, int]:
    """Snapshot of how many REST calls were blocked, by label."""
    return dict(_BLOCKED_CALL_COUNTERS)


def reset_blocked_call_counters() -> None:
    """Clear the counters (used by diagnostics rotation)."""
    _BLOCKED_CALL_COUNTERS.clear()
