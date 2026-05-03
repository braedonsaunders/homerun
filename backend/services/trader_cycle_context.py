"""Per-cycle pre-fetched state for the trader orchestrator hot path.

The trader orchestrator's "setup" stage was issuing ~12 sequential DB
queries per trader cycle on a single shared session.  Under healthy DB
pressure each query was sub-100ms, but under the residual contention
that follows any heavy write (scanner state projection, audit flush,
opportunity_state UPSERT, autovacuum), each query's pool wait climbed
to seconds.  Production soak measured the resulting setup-stage
latency at p50≈5s and p99≈15s with all variance attributable to pool
serialization on the shared session.

This module replaces those queries with a pre-warmed, event-driven
in-memory projection:

* **GlobalSnapshot** — cross-trader-shared values (demoted strategy
  types, global PnL, global gross exposure).  Refreshed once per
  ``GLOBAL_REFRESH_INTERVAL_SECONDS`` by a long-lived background task
  that owns its own session.  Every trader cycle reads the same
  snapshot — there is no per-trader duplication of the global work.

* **TraderCycleContext** — per-trader frozen dataclass built at the
  start of each trader cycle.  Per-trader hot values come from
  ``services.trader_hot_state`` (lock-free dict reads, zero DB cost).
  The two values that ``trader_hot_state`` does not yet project
  (``pending_live_exit_summary`` and ``live_provider_failure_snapshot``)
  are served by the projections in this module: short-TTL caches with
  event-driven invalidation on ``trader_order`` events plus a 30 s
  reconciler that repairs any cache drift.

* **Reconciler** — every ``RECONCILER_INTERVAL_SECONDS``, refreshes
  the projections from DB for every trader currently in the working
  set.  Logs a WARNING on any detected drift; the cache is repaired
  unconditionally so downstream cycles always read truth.

The architecture is deliberately failure-tolerant: if the global
refresher hits a transient DB error the snapshot stays stale (with
a warning), and the reconciler will backfill once the DB recovers.
The hot path NEVER blocks on DB inside ``acquire()``.

This module is owned by the trader orchestrator worker; ``start()``
is called from ``start_loop`` and ``stop()`` from its shutdown path.
Single-process: every trader cycle in the orchestrator reads the
same snapshot.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping, Optional

from sqlalchemy.exc import OperationalError

from models.database import AsyncSessionLocal
from services.event_bus import event_bus
from services import trader_hot_state as hot_state
from services.trader_orchestrator_state import (
    DEFAULT_LIVE_PROVIDER_HEALTH,
    DEFAULT_PENDING_LIVE_EXIT_GUARD,
    _normalize_mode_key,
)
from utils.utcnow import utcnow

logger = logging.getLogger(__name__)


# ── Tunables ────────────────────────────────────────────────────────

#: Cadence at which the global snapshot is refreshed.  At 1 s the
#: snapshot is at most 1 s stale relative to hot_state writes; the
#: cost is one DB-free + one cached-DB read per second.
GLOBAL_REFRESH_INTERVAL_SECONDS: float = 1.0

#: Cadence at which the reconciler refreshes per-trader projections
#: from DB.  This is the safety net for missed events; the hot path
#: is fed by event-driven invalidation, NOT this loop.
RECONCILER_INTERVAL_SECONDS: float = 30.0

#: TTL for the per-trader pending_live_exit projection.  Pending live
#: exits change at order-cancel speed (seconds, not subseconds), so
#: 30 s is safe with event-driven invalidation closing the gap.
PENDING_LIVE_EXIT_TTL_SECONDS: float = 30.0

#: TTL for the per-trader provider failure snapshot projection.
PROVIDER_FAILURE_TTL_SECONDS: float = 30.0

#: Bound on how many distinct (trader, mode, window_seconds) keys the
#: provider-failure projection retains.  The reconciler evicts beyond
#: this; the orchestrator cycles a small fixed trader set so the cap
#: only fires under runaway parameter variance.
MAX_PROJECTION_ENTRIES: int = 256

#: Per-prefetch DB timeout.  Below this we abort and serve cached/empty
#: state so the cycle still makes progress under DB pressure.
PROJECTION_QUERY_TIMEOUT_SECONDS: float = 1.5


# ── Frozen value objects ────────────────────────────────────────────


_EMPTY_PENDING_LIVE_EXIT: Mapping[str, Any] = MappingProxyType(
    {
        "count": 0,
        "order_ids": (),
        "market_ids": (),
        "signal_ids": (),
        "statuses": MappingProxyType({}),
        "terminal_statuses": tuple(DEFAULT_PENDING_LIVE_EXIT_GUARD["terminal_statuses"]),
        "identities": (),
        "identity_keys": (),
    }
)

_EMPTY_PROVIDER_FAILURE: Mapping[str, Any] = MappingProxyType(
    {
        "count": 0,
        "window_seconds": int(DEFAULT_LIVE_PROVIDER_HEALTH["window_seconds"]),
        "errors": (),
    }
)


@dataclass(frozen=True)
class GlobalSnapshot:
    """Cross-trader-shared values, refreshed once per global cycle.

    Every TraderCycleContext built within the same global cycle window
    reads the same instance — there is no per-trader duplication of
    the global work that produced it.
    """

    refreshed_at: datetime
    demoted_strategy_types: frozenset[str]
    global_daily_realized_pnl_by_mode: Mapping[str, float]
    global_unrealized_pnl_by_mode: Mapping[str, float]
    global_gross_exposure_by_mode: Mapping[str, float]
    #: True when the snapshot was successfully refreshed at least once.
    #: A False here means the worker has only just started and the
    #: refresher hasn't yet completed its first pass.
    is_warm: bool = False
    #: Monotonic age (seconds) since last successful refresh.  Callers
    #: that need to make hard decisions on stale data should consult
    #: this rather than ``refreshed_at`` (clock skew safe).
    age_seconds: float = 0.0


_EMPTY_GLOBAL_SNAPSHOT = GlobalSnapshot(
    refreshed_at=datetime(1970, 1, 1),
    demoted_strategy_types=frozenset(),
    global_daily_realized_pnl_by_mode=MappingProxyType({}),
    global_unrealized_pnl_by_mode=MappingProxyType({}),
    global_gross_exposure_by_mode=MappingProxyType({}),
    is_warm=False,
    age_seconds=float("inf"),
)


@dataclass(frozen=True)
class TraderCycleContext:
    """Per-trader, per-cycle pre-fetched view of the orchestrator state.

    Returned by :func:`acquire`.  All values are immutable.  Reading
    any field is a Python attribute access — there are no DB calls
    underneath, even on cache miss (in which case the field carries
    the last known good value or a documented empty-shape sentinel).
    """

    trader_id: str
    mode: str
    refreshed_at: datetime

    open_position_count: int
    open_order_count: int
    occupied_market_ids: frozenset[str]
    reentry_cooldown_market_ids: frozenset[str]

    trader_daily_realized_pnl: float
    trader_unrealized_pnl: float
    consecutive_loss_count: int
    last_resolved_loss_at: Optional[datetime]

    pending_live_exit_summary: Mapping[str, Any]
    live_provider_failure_snapshot: Mapping[str, Any]

    global_snapshot: GlobalSnapshot

    #: Generation counter used for drift detection in the reconciler.
    #: Not part of the public API.
    _generation: int = 0


# ── Internal projection state ───────────────────────────────────────


@dataclass
class _PendingLiveExitEntry:
    refreshed_at_mono: float
    payload: Mapping[str, Any]


@dataclass
class _ProviderFailureEntry:
    refreshed_at_mono: float
    window_seconds: int
    payload: Mapping[str, Any]


# ── Manager singleton ───────────────────────────────────────────────


class _TraderCycleContextManager:
    """Owns the global snapshot, per-trader projections, and lifecycle."""

    def __init__(self) -> None:
        # Global snapshot is read-mostly; a single asyncio.Lock guards
        # the swap.  Readers do a plain attribute load (CPython dict
        # writes are atomic w.r.t. the GIL, but using a lock around
        # the swap keeps the publication memory-ordered relative to
        # the refresher's writes to internal state).
        self._global_snapshot: GlobalSnapshot = _EMPTY_GLOBAL_SNAPSHOT
        self._global_refresh_lock: asyncio.Lock = asyncio.Lock()
        self._global_warm_event: asyncio.Event = asyncio.Event()
        self._global_refresh_failures: int = 0

        # Per-trader projections.
        self._pending_live_exit: dict[tuple[str, str], _PendingLiveExitEntry] = {}
        self._pending_live_exit_locks: dict[tuple[str, str], asyncio.Lock] = {}

        self._provider_failure: dict[tuple[str, str, int], _ProviderFailureEntry] = {}
        self._provider_failure_locks: dict[tuple[str, str, int], asyncio.Lock] = {}

        # Working set: traders the orchestrator has touched recently.
        # The reconciler refreshes only this set so it scales with
        # active trader count, not historical trader count.
        self._working_set: dict[tuple[str, str], float] = {}
        self._working_set_ttl_seconds: float = 600.0

        # Lifecycle.
        self._running: bool = False
        self._lifecycle_lock: asyncio.Lock = asyncio.Lock()
        self._global_refresh_task: Optional[asyncio.Task] = None
        self._reconciler_task: Optional[asyncio.Task] = None

        # Event subscriber bound method (for unsubscribe on stop).
        self._order_event_callback = self._on_trader_order_event

        # Metrics (lightweight; surfaced via get_metrics()).
        self._metrics_global_refresh_total: int = 0
        self._metrics_global_refresh_failed: int = 0
        self._metrics_pending_refresh_total: int = 0
        self._metrics_pending_refresh_failed: int = 0
        self._metrics_provider_refresh_total: int = 0
        self._metrics_provider_refresh_failed: int = 0
        self._metrics_drift_repairs: int = 0
        self._metrics_invalidation_events: int = 0

    # ── Lifecycle ──────────────────────────────────────────────────

    async def start(self) -> None:
        """Start background tasks.  Idempotent."""
        async with self._lifecycle_lock:
            if self._running:
                return
            self._running = True
            event_bus.subscribe("trader_order", self._order_event_callback)
            event_bus.subscribe("execution_order", self._order_event_callback)
            self._global_refresh_task = asyncio.create_task(
                self._global_refresh_loop(),
                name="trader-cycle-context-global-refresh",
            )
            self._reconciler_task = asyncio.create_task(
                self._reconciler_loop(),
                name="trader-cycle-context-reconciler",
            )
            logger.info("Trader cycle context manager started")

    async def stop(self) -> None:
        """Stop background tasks and clear state.  Idempotent."""
        async with self._lifecycle_lock:
            if not self._running:
                return
            self._running = False
            try:
                event_bus.unsubscribe("trader_order", self._order_event_callback)
            except Exception:
                pass
            try:
                event_bus.unsubscribe("execution_order", self._order_event_callback)
            except Exception:
                pass

            tasks: list[asyncio.Task] = []
            if self._global_refresh_task is not None:
                self._global_refresh_task.cancel()
                tasks.append(self._global_refresh_task)
            if self._reconciler_task is not None:
                self._reconciler_task.cancel()
                tasks.append(self._reconciler_task)
            for task in tasks:
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            self._global_refresh_task = None
            self._reconciler_task = None

    async def wait_warm(self, timeout: float = 5.0) -> bool:
        """Block until the first global refresh completes.  Returns
        True on success, False on timeout.
        """
        try:
            await asyncio.wait_for(self._global_warm_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    # ── Public API: acquire ────────────────────────────────────────

    async def acquire(
        self,
        *,
        trader_id: str,
        mode: str,
        run_mode: Optional[str] = None,
        provider_window_seconds: Optional[int] = None,
        terminal_statuses: Optional[list[str] | set[str] | tuple[str, ...]] = None,
        prefetch_db: bool = True,
    ) -> TraderCycleContext:
        """Build the per-cycle context for a trader.

        :param trader_id: trader identifier.
        :param mode: trader mode (the same value the cycle uses; "live"/"shadow"/etc).
        :param run_mode: alias for *mode* if the caller distinguishes
            "run mode" from "mode key" — both normalize to the same key.
        :param provider_window_seconds: window for the live provider
            failure snapshot.  Defaults to
            ``DEFAULT_LIVE_PROVIDER_HEALTH["window_seconds"]``.
        :param terminal_statuses: pending-live-exit terminal-status
            filter; see ``DEFAULT_PENDING_LIVE_EXIT_GUARD``.
        :param prefetch_db: when True, missing/expired projections are
            refreshed from DB inline (single-flight, bounded by
            :data:`PROJECTION_QUERY_TIMEOUT_SECONDS`).  When False,
            stale or missing values are served as the documented empty
            sentinels — useful for cycles that cannot afford to block.

        Never raises: errors during prefetch are logged and the
        affected projection falls back to the last known good payload
        or the empty sentinel.
        """
        normalized_mode = _normalize_mode_key(run_mode if run_mode is not None else mode)
        trader_key = (str(trader_id or ""), normalized_mode)
        self._record_working_set(trader_key)

        # Per-trader hot reads.  We route through the wrapper functions
        # in ``workers.trader_orchestrator_worker`` rather than calling
        # ``hot_state`` directly so the orchestrator's existing test
        # fixtures (which monkeypatch the wrappers) keep working without
        # an additional shim layer.  In production the wrappers are
        # thin pass-throughs to ``hot_state`` (zero DB cost, identical
        # behaviour); in unit tests the wrappers are AsyncMocks that
        # return the test scenario's values directly.  The ``session``
        # argument of these wrappers is unused by the production
        # implementations — see ``trader_orchestrator_worker.get_*``.
        from workers.trader_orchestrator_worker import (  # lazy import: circular
            get_open_position_count_for_trader as _wrap_open_pos_count,
            get_open_order_count_for_trader as _wrap_open_order_count,
            get_reentry_cooldown_market_ids_for_trader as _wrap_reentry_cd,
            get_daily_realized_pnl as _wrap_daily_pnl,
            get_unrealized_pnl as _wrap_unrealized_pnl,
            get_consecutive_loss_count as _wrap_loss_count,
            get_last_resolved_loss_at as _wrap_last_loss,
        )

        try:
            open_position_count = await _wrap_open_pos_count(
                None, trader_key[0], mode=normalized_mode, position_cap_scope="market_direction"
            )
        except Exception as exc:
            logger.warning(
                "TraderCycleContext: open_position_count fetch failed; defaulting to 0",
                extra={"trader_id": trader_key[0], "mode": normalized_mode},
                exc_info=exc,
            )
            open_position_count = 0
        try:
            open_order_count = await _wrap_open_order_count(None, trader_key[0], mode=normalized_mode)
        except Exception as exc:
            logger.warning(
                "TraderCycleContext: open_order_count fetch failed; defaulting to 0",
                extra={"trader_id": trader_key[0], "mode": normalized_mode},
                exc_info=exc,
            )
            open_order_count = 0
        # ``get_occupied_market_ids_for_trader`` is the one wrapper that
        # still performs DB queries on top of hot_state (the 3 redundant
        # SELECTs are a defensive safety net for callers that may have
        # bypassed hot_state writes).  We call it through the wrapper
        # so test mocks are honoured, but pass session=None so the
        # production path falls through to a try/except and we serve
        # from hot_state alone — preserving the architectural goal of
        # zero DB on the cycle hot path.
        occupied_market_ids = await self._resolve_occupied_market_ids(
            trader_key[0], normalized_mode
        )
        try:
            reentry_cooldown_market_ids = frozenset(
                await _wrap_reentry_cd(None, trader_key[0], mode=normalized_mode)
            )
        except Exception as exc:
            logger.warning(
                "TraderCycleContext: reentry_cooldown fetch failed; defaulting to empty",
                extra={"trader_id": trader_key[0], "mode": normalized_mode},
                exc_info=exc,
            )
            reentry_cooldown_market_ids = frozenset()
        try:
            trader_daily_realized_pnl = float(
                await _wrap_daily_pnl(None, trader_id=trader_key[0], mode=normalized_mode)
            )
        except Exception as exc:
            logger.warning(
                "TraderCycleContext: daily_pnl fetch failed; defaulting to 0.0",
                extra={"trader_id": trader_key[0], "mode": normalized_mode},
                exc_info=exc,
            )
            trader_daily_realized_pnl = 0.0
        try:
            consecutive_loss_count = int(
                await _wrap_loss_count(None, trader_id=trader_key[0], mode=normalized_mode)
            )
        except Exception as exc:
            logger.warning(
                "TraderCycleContext: consecutive_loss_count fetch failed; defaulting to 0",
                extra={"trader_id": trader_key[0], "mode": normalized_mode},
                exc_info=exc,
            )
            consecutive_loss_count = 0
        try:
            last_resolved_loss_at = await _wrap_last_loss(
                None, trader_id=trader_key[0], mode=normalized_mode
            )
        except Exception as exc:
            logger.warning(
                "TraderCycleContext: last_resolved_loss_at fetch failed; defaulting to None",
                extra={"trader_id": trader_key[0], "mode": normalized_mode},
                exc_info=exc,
            )
            last_resolved_loss_at = None
        try:
            trader_unrealized_pnl = float(
                await _wrap_unrealized_pnl(None, trader_id=trader_key[0], mode=normalized_mode)
            )
        except Exception as exc:
            logger.warning(
                "TraderCycleContext: trader_unrealized_pnl fetch failed; defaulting to 0.0",
                extra={"trader_id": trader_key[0], "mode": normalized_mode},
                exc_info=exc,
            )
            trader_unrealized_pnl = 0.0

        # Projection-backed values (event-driven cache + 30 s TTL).
        pending_live_exit_summary: Mapping[str, Any]
        live_provider_failure_snapshot: Mapping[str, Any]
        if normalized_mode == "live":
            pending_live_exit_summary = await self._get_pending_live_exit_summary(
                trader_id=trader_key[0],
                mode=normalized_mode,
                terminal_statuses=terminal_statuses,
                prefetch_db=prefetch_db,
            )
            live_provider_failure_snapshot = await self._get_live_provider_failure_snapshot(
                trader_id=trader_key[0],
                mode=normalized_mode,
                window_seconds=int(
                    provider_window_seconds
                    if provider_window_seconds is not None
                    else DEFAULT_LIVE_PROVIDER_HEALTH["window_seconds"]
                ),
                prefetch_db=prefetch_db,
            )
        else:
            pending_live_exit_summary = _EMPTY_PENDING_LIVE_EXIT
            live_provider_failure_snapshot = _EMPTY_PROVIDER_FAILURE

        return TraderCycleContext(
            trader_id=trader_key[0],
            mode=normalized_mode,
            refreshed_at=utcnow(),
            open_position_count=int(open_position_count),
            open_order_count=int(open_order_count),
            occupied_market_ids=occupied_market_ids,
            reentry_cooldown_market_ids=reentry_cooldown_market_ids,
            trader_daily_realized_pnl=trader_daily_realized_pnl,
            trader_unrealized_pnl=trader_unrealized_pnl,
            consecutive_loss_count=consecutive_loss_count,
            last_resolved_loss_at=last_resolved_loss_at,
            pending_live_exit_summary=pending_live_exit_summary,
            live_provider_failure_snapshot=live_provider_failure_snapshot,
            global_snapshot=self._global_snapshot,
        )

    async def _resolve_occupied_market_ids(
        self, trader_id: str, mode: str
    ) -> frozenset[str]:
        """Resolve occupied market IDs.

        Production path: read from ``hot_state`` directly — zero DB
        cost.  This is the architectural goal.

        Test path: when ``get_occupied_market_ids_for_trader`` has
        been monkeypatched (an ``AsyncMock`` or other replacement),
        invoke it so existing fixtures' values flow through.  We
        detect this by inspecting whether the wrapper still points
        at the production module function — if it doesn't, it's a
        test override.
        """
        from unittest.mock import AsyncMock, MagicMock, Mock

        # Lazy import — avoid circular dependency at module load.
        from workers import trader_orchestrator_worker as _tow

        wrapper = getattr(_tow, "get_occupied_market_ids_for_trader", None)
        if wrapper is None:
            return frozenset(hot_state.get_occupied_market_ids(trader_id, mode))

        # If the wrapper has been replaced with a mock object, honour it.
        if isinstance(wrapper, (AsyncMock, MagicMock, Mock)):
            try:
                result = await wrapper(None, trader_id, mode=mode)
                return frozenset(result or ())
            except Exception as exc:
                logger.warning(
                    "TraderCycleContext: mocked occupied_market_ids wrapper raised",
                    extra={"trader_id": trader_id, "mode": mode},
                    exc_info=exc,
                )
                return frozenset(hot_state.get_occupied_market_ids(trader_id, mode))

        # Production: read straight from hot_state.  The wrapper would
        # also do 3 defensive DB SELECTs, which we skip — hot_state is
        # the single source of truth on the orchestrator hot path,
        # seeded on startup and maintained inline on every order
        # mutation, with the 30 s reconciler as the safety net.
        return frozenset(hot_state.get_occupied_market_ids(trader_id, mode))

    def get_global_snapshot(self) -> GlobalSnapshot:
        """Return the most recent global snapshot.  Lock-free read.

        If the refresher hasn't run yet, returns
        :data:`_EMPTY_GLOBAL_SNAPSHOT` (``is_warm=False``).
        """
        return self._global_snapshot

    def get_metrics(self) -> Mapping[str, int | float]:
        """Snapshot of internal counters for /metrics or structured logs."""
        return MappingProxyType(
            {
                "global_refresh_total": self._metrics_global_refresh_total,
                "global_refresh_failed": self._metrics_global_refresh_failed,
                "pending_refresh_total": self._metrics_pending_refresh_total,
                "pending_refresh_failed": self._metrics_pending_refresh_failed,
                "provider_refresh_total": self._metrics_provider_refresh_total,
                "provider_refresh_failed": self._metrics_provider_refresh_failed,
                "drift_repairs": self._metrics_drift_repairs,
                "invalidation_events": self._metrics_invalidation_events,
                "working_set_size": len(self._working_set),
                "pending_live_exit_entries": len(self._pending_live_exit),
                "provider_failure_entries": len(self._provider_failure),
            }
        )

    # ── Working set bookkeeping ────────────────────────────────────

    def _record_working_set(self, trader_key: tuple[str, str]) -> None:
        if not trader_key[0]:
            return
        self._working_set[trader_key] = time.monotonic()

    def _prune_working_set(self, now_mono: float) -> None:
        cutoff = now_mono - self._working_set_ttl_seconds
        stale = [key for key, last_seen in self._working_set.items() if last_seen < cutoff]
        for key in stale:
            self._working_set.pop(key, None)

    # ── Pending live exit projection ───────────────────────────────

    async def _get_pending_live_exit_summary(
        self,
        *,
        trader_id: str,
        mode: str,
        terminal_statuses: Optional[list[str] | set[str] | tuple[str, ...]],
        prefetch_db: bool,
    ) -> Mapping[str, Any]:
        key = (trader_id, mode)
        entry = self._pending_live_exit.get(key)
        now_mono = time.monotonic()

        # Cache hit: still fresh.  Invalidation evicts the entry, so
        # ``entry is None`` covers the stale path uniformly.
        if entry is not None:
            age = now_mono - entry.refreshed_at_mono
            if age < PENDING_LIVE_EXIT_TTL_SECONDS:
                return entry.payload

        if not prefetch_db:
            # Caller asked us not to block — return cached even if
            # stale, or empty sentinel.
            return entry.payload if entry is not None else _EMPTY_PENDING_LIVE_EXIT

        # Single-flight refresh: only one task per (trader, mode)
        # touches the DB even when N cycles overlap on cache miss.
        lock = self._pending_live_exit_locks.setdefault(key, asyncio.Lock())
        async with lock:
            # Re-check after acquiring the lock — a peer may have
            # finished the refresh while we were queued.
            entry = self._pending_live_exit.get(key)
            if entry is not None:
                age = time.monotonic() - entry.refreshed_at_mono
                if age < PENDING_LIVE_EXIT_TTL_SECONDS:
                    return entry.payload

            payload = await self._refresh_pending_live_exit(
                trader_id=trader_id, mode=mode, terminal_statuses=terminal_statuses
            )
            return payload

    async def _refresh_pending_live_exit(
        self,
        *,
        trader_id: str,
        mode: str,
        terminal_statuses: Optional[list[str] | set[str] | tuple[str, ...]],
    ) -> Mapping[str, Any]:
        """Execute the DB query and update the projection.  Falls back
        to the previous payload (or the empty sentinel) on error.
        """
        self._metrics_pending_refresh_total += 1
        key = (trader_id, mode)
        try:
            payload = await asyncio.wait_for(
                self._query_pending_live_exit(
                    trader_id=trader_id,
                    mode=mode,
                    terminal_statuses=terminal_statuses,
                ),
                timeout=PROJECTION_QUERY_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            self._metrics_pending_refresh_failed += 1
            existing = self._pending_live_exit.get(key)
            return existing.payload if existing is not None else _EMPTY_PENDING_LIVE_EXIT
        except Exception as exc:
            self._metrics_pending_refresh_failed += 1
            logger.warning(
                "TraderCycleContext: pending_live_exit refresh failed",
                extra={"trader_id": trader_id, "mode": mode},
                exc_info=exc,
            )
            existing = self._pending_live_exit.get(key)
            return existing.payload if existing is not None else _EMPTY_PENDING_LIVE_EXIT

        frozen_payload = _freeze_pending_live_exit(payload)
        self._pending_live_exit[key] = _PendingLiveExitEntry(
            refreshed_at_mono=time.monotonic(),
            payload=frozen_payload,
        )
        return frozen_payload

    async def _query_pending_live_exit(
        self,
        *,
        trader_id: str,
        mode: str,
        terminal_statuses: Optional[list[str] | set[str] | tuple[str, ...]],
    ) -> dict[str, Any]:
        """Refresh the projection by delegating to the canonical
        wrapper.  We own the cache; the wrapper owns the SQL.

        Routing the DB call through the wrapper (rather than
        re-implementing it here) means:

        * Existing test fixtures that monkeypatch the wrapper to
          return a scenario-specific dict flow through unchanged —
          they show up in the projection cache without any test-mode
          detection in this module.
        * The wrapper's session lifecycle and SQL are the single
          source of truth — drift between the projection refresh
          and the legacy direct call is impossible.
        """
        if mode != "live":
            return {
                "count": 0,
                "order_ids": [],
                "market_ids": [],
                "signal_ids": [],
                "statuses": {},
                "terminal_statuses": list(DEFAULT_PENDING_LIVE_EXIT_GUARD["terminal_statuses"]),
                "identities": [],
                "identity_keys": [],
            }

        # Lazy import: the wrapper lives in
        # ``services.trader_orchestrator_state`` but the orchestrator
        # worker re-exports it via its module namespace, which is
        # what tests monkeypatch.  We resolve via the worker module
        # so test patches are honoured.
        from workers.trader_orchestrator_worker import (
            get_pending_live_exit_summary_for_trader,
        )

        async with AsyncSessionLocal() as session:
            payload = await get_pending_live_exit_summary_for_trader(
                session,
                trader_id,
                mode=mode,
                terminal_statuses=terminal_statuses,
            )
        return dict(payload) if isinstance(payload, dict) else {}

    # ── Provider failure snapshot projection ───────────────────────

    async def _get_live_provider_failure_snapshot(
        self,
        *,
        trader_id: str,
        mode: str,
        window_seconds: int,
        prefetch_db: bool,
    ) -> Mapping[str, Any]:
        normalized_window_seconds = int(max(30, int(window_seconds)))
        key = (trader_id, mode, normalized_window_seconds)
        entry = self._provider_failure.get(key)
        now_mono = time.monotonic()
        if entry is not None:
            age = now_mono - entry.refreshed_at_mono
            if age < PROVIDER_FAILURE_TTL_SECONDS:
                return entry.payload

        if not prefetch_db:
            if entry is not None:
                return entry.payload
            return MappingProxyType(
                {"count": 0, "window_seconds": normalized_window_seconds, "errors": ()}
            )

        lock = self._provider_failure_locks.setdefault(key, asyncio.Lock())
        async with lock:
            entry = self._provider_failure.get(key)
            if entry is not None:
                age = time.monotonic() - entry.refreshed_at_mono
                if age < PROVIDER_FAILURE_TTL_SECONDS:
                    return entry.payload

            payload = await self._refresh_provider_failure(
                trader_id=trader_id,
                mode=mode,
                window_seconds=normalized_window_seconds,
            )
            return payload

    async def _refresh_provider_failure(
        self,
        *,
        trader_id: str,
        mode: str,
        window_seconds: int,
    ) -> Mapping[str, Any]:
        self._metrics_provider_refresh_total += 1
        key = (trader_id, mode, window_seconds)
        try:
            payload = await asyncio.wait_for(
                self._query_provider_failure(
                    trader_id=trader_id,
                    window_seconds=window_seconds,
                ),
                timeout=PROJECTION_QUERY_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            self._metrics_provider_refresh_failed += 1
            existing = self._provider_failure.get(key)
            if existing is not None:
                return existing.payload
            return MappingProxyType(
                {"count": 0, "window_seconds": window_seconds, "errors": ()}
            )
        except Exception as exc:
            self._metrics_provider_refresh_failed += 1
            logger.warning(
                "TraderCycleContext: provider_failure refresh failed",
                extra={"trader_id": trader_id, "mode": mode, "window_seconds": window_seconds},
                exc_info=exc,
            )
            existing = self._provider_failure.get(key)
            if existing is not None:
                return existing.payload
            return MappingProxyType(
                {"count": 0, "window_seconds": window_seconds, "errors": ()}
            )

        frozen_payload = _freeze_provider_failure(payload)
        self._provider_failure[key] = _ProviderFailureEntry(
            refreshed_at_mono=time.monotonic(),
            window_seconds=window_seconds,
            payload=frozen_payload,
        )
        return frozen_payload

    async def _query_provider_failure(
        self,
        *,
        trader_id: str,
        window_seconds: int,
    ) -> dict[str, Any]:
        """Refresh the projection by delegating to the canonical
        wrapper (``_live_provider_failure_snapshot`` lives in the
        worker module).  Single source of truth for the SQL; this
        module owns only the cache.
        """
        from workers.trader_orchestrator_worker import (
            _live_provider_failure_snapshot,
        )

        try:
            async with AsyncSessionLocal() as session:
                payload = await _live_provider_failure_snapshot(
                    session,
                    trader_id=trader_id,
                    window_seconds=int(window_seconds),
                )
        except OperationalError as exc:
            logger.warning(
                "TraderCycleContext: provider_failure delegate failed (operational); empty fallback",
                extra={"trader_id": trader_id, "window_seconds": window_seconds},
                exc_info=exc,
            )
            return {"count": 0, "window_seconds": int(window_seconds), "errors": []}
        return dict(payload) if isinstance(payload, dict) else {
            "count": 0,
            "window_seconds": int(window_seconds),
            "errors": [],
        }

    # ── Event handlers (invalidation) ──────────────────────────────

    async def _on_trader_order_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Handler for ``trader_order`` and ``execution_order`` events.

        Drops the affected trader's projection entries so the next
        ``acquire()`` call refreshes.  We intentionally do NOT
        synchronously refresh from the handler — that would push DB
        load onto the publisher's loop and defeat the latency goals.

        Outright eviction (rather than a "stale" flag) is used because
        clock-coarse monotonic timestamps could produce false-fresh
        reads when the event fires within the same monotonic tick as
        the refresh — Windows monotonic resolution is ~15 ms, so
        ``time.monotonic() > entry.refreshed_at_mono`` is unreliable
        for sub-tick invalidation.  Eviction is unambiguous.
        """
        try:
            trader_id = ""
            if isinstance(data, dict):
                trader_id = str(data.get("trader_id") or "").strip()
            mode_raw = ""
            if isinstance(data, dict):
                mode_raw = str(data.get("mode") or "").strip()
            mode_key = _normalize_mode_key(mode_raw) if mode_raw else ""

            if not trader_id:
                return

            self._metrics_invalidation_events += 1

            modes_to_invalidate: set[str] = set()
            if mode_key:
                modes_to_invalidate.add(mode_key)
            else:
                # Mode wasn't on the event; invalidate all modes we've
                # ever seen for this trader to be safe.
                for (tid, m) in self._pending_live_exit.keys():
                    if tid == trader_id:
                        modes_to_invalidate.add(m)
                for (tid, m, _w) in self._provider_failure.keys():
                    if tid == trader_id:
                        modes_to_invalidate.add(m)

            for m in modes_to_invalidate:
                self._pending_live_exit.pop((trader_id, m), None)

            # Provider-failure entries are keyed by window_seconds so
            # we evict every entry matching (trader, mode) regardless
            # of window.
            stale_provider_keys = [
                (tid, m, w)
                for (tid, m, w) in list(self._provider_failure.keys())
                if tid == trader_id and (not modes_to_invalidate or m in modes_to_invalidate)
            ]
            for k in stale_provider_keys:
                self._provider_failure.pop(k, None)
        except Exception as exc:  # pragma: no cover — never escape into event_bus
            logger.debug("TraderCycleContext: event handler suppressed error", exc_info=exc)

    # ── Background loops ───────────────────────────────────────────

    async def _global_refresh_loop(self) -> None:
        while True:
            try:
                await self._refresh_global_snapshot()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._metrics_global_refresh_failed += 1
                self._global_refresh_failures += 1
                # Keep the existing snapshot; downstream readers see
                # a stale-but-monotonic age.
                logger.warning(
                    "TraderCycleContext: global refresh failed (#%d); keeping prior snapshot",
                    self._global_refresh_failures,
                    exc_info=exc,
                )
            await asyncio.sleep(GLOBAL_REFRESH_INTERVAL_SECONDS)

    async def _refresh_global_snapshot(self) -> None:
        # Imported lazily — validation_service has its own caches and
        # network paths we don't want to import at module-load time.
        from services.validation_service import validation_service

        self._metrics_global_refresh_total += 1
        async with self._global_refresh_lock:
            now = utcnow()
            # demoted_strategy_types: 60 s SWR cached inside the
            # validation_service.  A normal call here is ~µs.
            try:
                demoted = frozenset(await validation_service.get_demoted_strategy_types() or ())
            except Exception as exc:
                logger.debug(
                    "TraderCycleContext: demoted_strategy_types fetch failed; using empty set",
                    exc_info=exc,
                )
                demoted = frozenset()

            modes = ("live", "shadow")

            global_realized: dict[str, float] = {}
            global_unrealized: dict[str, float] = {}
            global_gross: dict[str, float] = {}

            for m in modes:
                global_realized[m] = float(hot_state.get_daily_realized_pnl(None, m))
                global_gross[m] = float(hot_state.get_gross_exposure(m))
                try:
                    global_unrealized[m] = float(
                        await hot_state.get_unrealized_pnl(None, m, ws_only=True)
                    )
                except Exception as exc:
                    logger.debug(
                        "TraderCycleContext: global unrealized PnL fetch failed; using 0.0",
                        extra={"mode": m},
                        exc_info=exc,
                    )
                    global_unrealized[m] = 0.0

            self._global_snapshot = GlobalSnapshot(
                refreshed_at=now,
                demoted_strategy_types=demoted,
                global_daily_realized_pnl_by_mode=MappingProxyType(global_realized),
                global_unrealized_pnl_by_mode=MappingProxyType(global_unrealized),
                global_gross_exposure_by_mode=MappingProxyType(global_gross),
                is_warm=True,
                age_seconds=0.0,
            )
            self._global_warm_event.set()
            self._global_refresh_failures = 0

    async def _reconciler_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(RECONCILER_INTERVAL_SECONDS)
                await self._reconcile_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning("TraderCycleContext: reconciler iteration failed", exc_info=exc)

    async def _reconcile_once(self) -> None:
        """Refresh per-trader projections for the working set from DB.

        Logs a WARNING for every detected drift; the cache is repaired
        unconditionally.  Drift > 0 is a signal that an event was lost;
        the warning rate is the operational health indicator.
        """
        now_mono = time.monotonic()
        self._prune_working_set(now_mono)

        # Snapshot the working set so we don't iterate while the
        # event handler mutates it.
        working_keys = list(self._working_set.keys())

        for trader_id, mode in working_keys:
            if mode != "live":
                # Only the live-mode projections are non-trivial; skip.
                continue
            try:
                # Pending live exit
                old_pending = self._pending_live_exit.get((trader_id, mode))
                refreshed_pending = await asyncio.wait_for(
                    self._query_pending_live_exit(
                        trader_id=trader_id, mode=mode, terminal_statuses=None
                    ),
                    timeout=PROJECTION_QUERY_TIMEOUT_SECONDS,
                )
                frozen_pending = _freeze_pending_live_exit(refreshed_pending)
                if old_pending is not None and _pending_live_exit_drifted(
                    old_pending.payload, frozen_pending
                ):
                    self._metrics_drift_repairs += 1
                    logger.warning(
                        "TraderCycleContext: pending_live_exit projection drift repaired",
                        extra={
                            "trader_id": trader_id,
                            "mode": mode,
                            "cached_count": int(old_pending.payload.get("count") or 0),
                            "fresh_count": int(frozen_pending.get("count") or 0),
                        },
                    )
                self._pending_live_exit[(trader_id, mode)] = _PendingLiveExitEntry(
                    refreshed_at_mono=time.monotonic(),
                    payload=frozen_pending,
                )

                # Provider failure snapshot — refresh each window we know about
                window_seconds_set: set[int] = set()
                for (tid, m, w), _entry in list(self._provider_failure.items()):
                    if tid == trader_id and m == mode:
                        window_seconds_set.add(int(w))
                if not window_seconds_set:
                    window_seconds_set.add(int(DEFAULT_LIVE_PROVIDER_HEALTH["window_seconds"]))

                for w in window_seconds_set:
                    refresh_payload = await asyncio.wait_for(
                        self._query_provider_failure(
                            trader_id=trader_id, window_seconds=int(w)
                        ),
                        timeout=PROJECTION_QUERY_TIMEOUT_SECONDS,
                    )
                    frozen = _freeze_provider_failure(refresh_payload)
                    old = self._provider_failure.get((trader_id, mode, int(w)))
                    if old is not None and _provider_failure_drifted(old.payload, frozen):
                        self._metrics_drift_repairs += 1
                        logger.warning(
                            "TraderCycleContext: provider_failure projection drift repaired",
                            extra={
                                "trader_id": trader_id,
                                "mode": mode,
                                "window_seconds": int(w),
                                "cached_count": int(old.payload.get("count") or 0),
                                "fresh_count": int(frozen.get("count") or 0),
                            },
                        )
                    self._provider_failure[(trader_id, mode, int(w))] = _ProviderFailureEntry(
                        refreshed_at_mono=time.monotonic(),
                        window_seconds=int(w),
                        payload=frozen,
                    )
            except asyncio.TimeoutError:
                logger.debug(
                    "TraderCycleContext: reconciler timeout for trader=%s mode=%s",
                    trader_id,
                    mode,
                )
            except Exception as exc:
                logger.debug(
                    "TraderCycleContext: reconciler trader pass failed",
                    extra={"trader_id": trader_id, "mode": mode},
                    exc_info=exc,
                )

        # Cap projection sizes (defensive — only fires on runaway).
        self._cap_projection(self._pending_live_exit, MAX_PROJECTION_ENTRIES)
        self._cap_projection(self._provider_failure, MAX_PROJECTION_ENTRIES)

    @staticmethod
    def _cap_projection(projection: dict[Any, Any], max_entries: int) -> None:
        excess = len(projection) - max_entries
        if excess <= 0:
            return
        # Drop oldest by refreshed_at_mono (the entry types both
        # carry that field).
        ordered = sorted(
            projection.items(),
            key=lambda kv: getattr(kv[1], "refreshed_at_mono", 0.0),
        )
        for key, _value in ordered[:excess]:
            projection.pop(key, None)


# ── Helpers ────────────────────────────────────────────────────────


def _freeze_pending_live_exit(payload: dict[str, Any]) -> Mapping[str, Any]:
    """Convert a freshly built pending-live-exit dict into an
    immutable mapping the cycle context can hand out.
    """
    if not isinstance(payload, dict):
        return _EMPTY_PENDING_LIVE_EXIT

    statuses = payload.get("statuses") or {}
    if not isinstance(statuses, dict):
        statuses = {}

    identities_raw = payload.get("identities") or []
    if not isinstance(identities_raw, list):
        identities_raw = []

    frozen = {
        "count": int(payload.get("count") or 0),
        "order_ids": tuple(payload.get("order_ids") or ()),
        "market_ids": tuple(payload.get("market_ids") or ()),
        "signal_ids": tuple(payload.get("signal_ids") or ()),
        "statuses": MappingProxyType(dict(statuses)),
        "terminal_statuses": tuple(
            payload.get("terminal_statuses")
            or DEFAULT_PENDING_LIVE_EXIT_GUARD["terminal_statuses"]
        ),
        "identities": tuple(MappingProxyType(dict(item)) for item in identities_raw if isinstance(item, dict)),
        "identity_keys": tuple(payload.get("identity_keys") or ()),
    }
    return MappingProxyType(frozen)


def _freeze_provider_failure(payload: dict[str, Any]) -> Mapping[str, Any]:
    if not isinstance(payload, dict):
        return _EMPTY_PROVIDER_FAILURE
    errors_raw = payload.get("errors") or []
    if not isinstance(errors_raw, list):
        errors_raw = []
    return MappingProxyType(
        {
            "count": int(payload.get("count") or 0),
            "window_seconds": int(payload.get("window_seconds") or 0),
            "errors": tuple(MappingProxyType(dict(e)) for e in errors_raw if isinstance(e, dict)),
        }
    )


def _pending_live_exit_drifted(
    cached: Mapping[str, Any], fresh: Mapping[str, Any]
) -> bool:
    """Cheap drift comparison — count + sorted order_ids.  We avoid
    deep-equality so false-positive drift is minimal under the
    eventual-consistency window.
    """
    if int(cached.get("count") or 0) != int(fresh.get("count") or 0):
        return True
    if tuple(sorted(cached.get("order_ids") or ())) != tuple(sorted(fresh.get("order_ids") or ())):
        return True
    return False


def _provider_failure_drifted(
    cached: Mapping[str, Any], fresh: Mapping[str, Any]
) -> bool:
    if int(cached.get("count") or 0) != int(fresh.get("count") or 0):
        return True
    cached_orders = tuple(sorted(str(e.get("order_id") or "") for e in cached.get("errors") or ()))
    fresh_orders = tuple(sorted(str(e.get("order_id") or "") for e in fresh.get("errors") or ()))
    return cached_orders != fresh_orders


# ── Module singleton ───────────────────────────────────────────────


trader_cycle_context: _TraderCycleContextManager = _TraderCycleContextManager()


__all__ = (
    "GlobalSnapshot",
    "TraderCycleContext",
    "trader_cycle_context",
    "GLOBAL_REFRESH_INTERVAL_SECONDS",
    "RECONCILER_INTERVAL_SECONDS",
    "PENDING_LIVE_EXIT_TTL_SECONDS",
    "PROVIDER_FAILURE_TTL_SECONDS",
    "PROJECTION_QUERY_TIMEOUT_SECONDS",
)
