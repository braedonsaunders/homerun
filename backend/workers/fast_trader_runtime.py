"""Fast-tier trader runtime.

A dedicated worker that owns every trader with ``latency_class='fast'``.
Each trader gets its own long-lived asyncio Task with:

* An isolated DB engine (``FastAsyncSessionLocal``) — aggressive 1500ms
  statement_timeout, small dedicated pool, cannot be starved by the
  slow-tier reconciler or session_engine.
* Event-driven wakeup via ``event_bus`` (``trade_signal_batch`` /
  ``trade_signal_emission``) with a 250ms polling fallback so a missed
  wakeup never stalls a trader longer than a quarter second.
* Direct single-leg submission via ``fast_submit.execute_fast_signal``
  — no ``execution_sessions``, no pre-submit placeholder, no 45s ack
  wait, no leg-wave orchestration.

A fast-tier cycle looks like: event arrives → session opens → one quick
SELECT for pending signals → strategy.evaluate → one quick INSERT of
the trader_order row → commit → done.  Target end-to-end: <150ms under
normal CLOB latency.

Fast traders are *opt-in*: the shared orchestrator loop skips any trader
with ``latency_class='fast'`` (see trader_orchestrator_worker), so the
two runtimes never fight over the same trader.  Pre-existing traders
default to ``latency_class='normal'`` and keep the existing path.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Optional

from sqlalchemy.exc import DBAPIError

from models.database import AsyncSessionLocal, FastAsyncSessionLocal, Trader, TraderSignalCursor
from services.event_bus import event_bus
from services.intent_runtime import get_intent_runtime
from services.live_pressure import (
    backpressure_extra_sleep_seconds,
    is_db_pressure_active,
)
from services.strategy_loader import strategy_loader
import services.trader_hot_state as hot_state
from services.trader_hot_state import get_signal_sequence_cursor as _hot_get_signal_sequence_cursor
from services.trader_orchestrator.decision_gates import enrich_final_reason
from services.trader_orchestrator.hot_path import hot_path_no_rest
from services.wallet_state_cache import get_wallet_state_cache
from services.trader_orchestrator.fast_submit import (
    execute_fast_signal,
)
from services.trader_orchestrator_state import (
    fetch_recent_consumed_signal_ids,
    get_trader_signal_cursor,
    list_fast_traders,
    list_unconsumed_trade_signals,
    read_orchestrator_control,
    reconcile_orphaned_fast_submissions,
)
from services.worker_state import _is_retryable_db_error, write_worker_snapshot
from utils.converters import safe_int
from utils.logger import get_logger
from utils.utcnow import utcnow

logger = get_logger(__name__)

WORKER_NAME = "fast_trader_runtime"

# Budgets for the fast tier.  The cycle budget covers up to
# _MAX_SIGNALS_PER_CYCLE (4) signal submissions; each submission is a
# CLOB roundtrip of ~300-500ms even on a clean network, so a 4-signal
# cycle realistically lands around 1.5-2s today.  Hold the budget
# warning at a value that still catches genuine outliers (loop
# starvation, reconciler stalls, slow CLOB) without crying wolf on
# every healthy cycle.
_CYCLE_HARD_BUDGET_SECONDS = 3.0
_EVALUATE_BUDGET_SECONDS = 0.2
_SUBMIT_BUDGET_SECONDS = 1.0
_POLL_FALLBACK_SECONDS = 0.25
_HEARTBEAT_INTERVAL_SECONDS = 5.0
_TRADER_REFRESH_INTERVAL_SECONDS = 15.0
_FAST_TASK_STALE_SECONDS = 30.0
# Periodic orphan-reconcile sweep cadence.  The startup sweep runs once
# right after bootstrap; this re-runs it on a fixed cadence so any
# in_flight / clob_exception / post_update_failed rows produced by a
# cycle teardown (CancelledError, process kill, network blip) get
# resolved within minutes instead of waiting for the next worker
# restart.  60s strikes a balance: well below the operator-visible
# threshold for "the cap is stuck at N/N" and well above the venue's
# per-call rate limits (one wallet-snapshot fetch per fast trader per
# 60s is trivial).
_ORPHAN_RECONCILE_INTERVAL_SECONDS = 60.0
_MAX_SIGNALS_PER_CYCLE = 4
_TRADER_LAST_RUN_TOUCH_SECONDS = 30.0
_IDLE_EVENT_INTERVAL_SECONDS = 300.0

# Events that should wake up a fast trader.  Anything that adds a new
# signal row or changes signal state belongs here.
_WAKE_EVENTS = (
    "trade_signal_emission",
    "trade_signal_batch",
    "signals_update",
)

_FAST_ORDER_SUCCESS_STATUSES = {"executed", "submitted", "open", "working", "completed", "partial", "hedging"}


def _snapshot_from_db_row(row: Any):
    """Adapter so the safety-net sweep can seed the cache from DB rows.

    Returns a ``SignalSnapshot`` or None.  Soft-fails if signal_cache
    isn't importable for any reason (test environments, partial
    bootstrap) so the fast trader keeps running.
    """
    try:
        from services import signal_cache as _signal_cache

        return _signal_cache.SignalSnapshot.from_db_row(row)
    except Exception:
        return None


def _get_sequence_cursor(trader_id: str) -> Optional[int]:
    """Read the live signal-sequence cursor from hot state (non-blocking)."""
    try:
        value = _hot_get_signal_sequence_cursor(trader_id, "live")
    except Exception:
        return None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


class _FastTraderTask:
    """One fast-tier trader's event-driven loop."""

    def __init__(self, trader: dict[str, Any], wake: asyncio.Event) -> None:
        self._trader = dict(trader)
        self._wake = wake
        self._stopped = False
        self._last_trader_touch_at = 0.0
        self._last_idle_event_at = 0.0
        self._started_at_mono = time.monotonic()
        self._last_cycle_started_at = 0.0
        self._last_cycle_finished_at = 0.0
        self._last_cycle_duration_seconds = 0.0
        self._last_cycle_error: str | None = None
        self._cycles_completed = 0
        # Per-stage timings (ms) populated by _run_once and surfaced via the
        # cycle-exceeded-budget warning to pinpoint which stage stalled.
        self._last_stage_timings_ms: dict[str, float] = {}
        # Per-task cached signal cursor.  The DB-resident cursor only
        # changes when this trader consumes a signal; the trader task
        # owns those writes so it knows when its cache is stale.  This
        # eliminates the two per-cycle cursor SELECTs that the previous
        # design ran every time hot_state hadn't been seeded yet.
        self._cached_cursor_runtime_sequence: Optional[int] = None
        self._cached_cursor_created_at: Optional[Any] = None
        self._cached_cursor_signal_id: Optional[str] = None
        self._cached_cursor_loaded: bool = False
        # Plan 0032 Task 7: hydrate the per-trader consumed-set once at
        # task lifetime, BEFORE the first intent_runtime fetch, so the
        # post-fetch filter below sees an authoritative consumption
        # history. The deploy-cycle restart of ``worker-trading``
        # otherwise leaves the consumed-set empty until ``mark_consumed``
        # populates it incrementally — and during that warm-up window
        # ``intent_runtime`` happily re-presents every reactivated
        # ``skipped`` scanner signal whose ``TraderOrder`` already
        # exists, producing 200-400 "trader_order already exists"
        # decisions/h on busy traders.
        self._consumed_set_hydrated: bool = False
        # Throttle the "wallet state stale" warning — under degraded
        # conditions the fast trader's per-trader 250ms wake loop
        # would otherwise emit this hundreds of times per minute and
        # bury the actual diagnostic signal.  Log once when the gate
        # first refuses, then at most once per 30s per trader while
        # it stays refusing.  The freshness gate itself still runs
        # every cycle — only the LOG is throttled.
        self._last_wallet_stale_log_mono: float = 0.0
        self._last_wallet_state_was_fresh: bool = True

    @property
    def trader_id(self) -> str:
        return str(self._trader.get("id") or "")

    def refresh_trader(self, trader: dict[str, Any]) -> None:
        self._trader = dict(trader)

    def stop(self) -> None:
        self._stopped = True
        self._wake.set()

    def is_stale(self, now_mono: float, threshold_seconds: float) -> bool:
        if self._stopped:
            return False
        if not self._trader.get("is_enabled", True) or self._trader.get("is_paused", False):
            return False
        last_progress = max(self._last_cycle_finished_at, self._last_cycle_started_at, self._started_at_mono)
        return now_mono - last_progress > threshold_seconds

    def snapshot(self, now_mono: float, threshold_seconds: float) -> dict[str, Any]:
        last_progress = max(self._last_cycle_finished_at, self._last_cycle_started_at, self._started_at_mono)
        inflight_seconds = 0.0
        if self._last_cycle_started_at > self._last_cycle_finished_at:
            inflight_seconds = max(0.0, now_mono - self._last_cycle_started_at)
        return {
            "trader_id": self.trader_id,
            "enabled": bool(self._trader.get("is_enabled", True)),
            "paused": bool(self._trader.get("is_paused", False)),
            "stopped": self._stopped,
            "stale": self.is_stale(now_mono, threshold_seconds),
            "last_progress_age_seconds": round(max(0.0, now_mono - last_progress), 3),
            "inflight_seconds": round(inflight_seconds, 3),
            "last_cycle_duration_seconds": round(max(0.0, self._last_cycle_duration_seconds), 3),
            "cycles_completed": self._cycles_completed,
            "last_cycle_error": self._last_cycle_error,
        }

    async def run(self) -> None:
        """Infinite per-trader loop.  Exits only when ``stop()`` is called."""
        trader_id = self.trader_id
        if not trader_id:
            return
        logger.info("Fast trader runtime started", trader_id=trader_id, name=self._trader.get("name"))
        while not self._stopped:
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=_POLL_FALLBACK_SECONDS)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()
            if self._stopped:
                break
            if not self._trader.get("is_enabled", True) or self._trader.get("is_paused", False):
                continue
            # Voluntary backpressure: when the system is approaching
            # saturation (intent_runtime queue >50% full, future
            # observers add their signals too) extend the cycle interval
            # so we don't pile additional load onto an overloaded DB.
            # Curve in backpressure_extra_sleep_seconds: 0 below 40%,
            # ramps to 3× the base interval at 100%. For a fast trader
            # on the 0.25s wake-poll, this caps the extra sleep at
            # 0.75s — still snappy under contention.
            extra_sleep = backpressure_extra_sleep_seconds(_POLL_FALLBACK_SECONDS)
            if extra_sleep > 0:
                await asyncio.sleep(extra_sleep)
                if self._stopped:
                    break
            cycle_started_at = time.monotonic()
            self._last_cycle_started_at = cycle_started_at
            try:
                await self._run_once()
                self._last_cycle_error = None
            except asyncio.CancelledError:
                self._last_cycle_error = "cancelled"
                raise
            except asyncio.TimeoutError as exc:
                self._last_cycle_error = type(exc).__name__
                logger.warning(
                    "Fast trader cycle timed out",
                    trader_id=trader_id,
                    exc_info=exc,
                )
            except Exception as exc:
                self._last_cycle_error = type(exc).__name__
                if _is_retryable_db_error(exc):
                    logger.warning(
                        "Fast trader cycle hit retryable DB error",
                        trader_id=trader_id,
                        exc_info=exc,
                    )
                else:
                    logger.error("Fast trader cycle failed", trader_id=trader_id, exc_info=exc)
            finally:
                cycle_finished_at = time.monotonic()
                self._last_cycle_finished_at = cycle_finished_at
                self._last_cycle_duration_seconds = cycle_finished_at - cycle_started_at
                self._cycles_completed += 1
                if self._last_cycle_duration_seconds > _CYCLE_HARD_BUDGET_SECONDS:
                    logger.warning(
                        "Fast trader cycle exceeded hard budget",
                        trader_id=trader_id,
                        duration_s=round(self._last_cycle_duration_seconds, 3),
                        budget_s=_CYCLE_HARD_BUDGET_SECONDS,
                        # Per-stage breakdown captured in _run_once. Keys
                        # vary by code path (runtime-signals vs db-fallback
                        # vs idle), so log whichever fired this cycle so we
                        # can pinpoint which stage stalled.
                        stage_timings_ms=getattr(self, "_last_stage_timings_ms", None) or {},
                    )
        logger.info("Fast trader runtime stopped", trader_id=trader_id)

    def _accepted_sources(self) -> list[str]:
        configs = self._trader.get("source_configs") or []
        sources: list[str] = []
        for cfg in configs:
            if not isinstance(cfg, dict):
                continue
            if not cfg.get("enabled", True):
                continue
            src = str(cfg.get("source_key") or "").strip().lower()
            if src:
                sources.append(src)
        return sources

    def _accepted_strategy_types_by_source(self) -> dict[str, list[str]]:
        configs = self._trader.get("source_configs") or []
        accepted_by_source: dict[str, list[str]] = {}
        for cfg in configs:
            if not isinstance(cfg, dict) or not cfg.get("enabled", True):
                continue
            source_key = str(cfg.get("source_key") or "").strip().lower()
            if not source_key:
                continue
            raw_values: list[Any] = [
                cfg.get("strategy_key"),
                cfg.get("requested_strategy_key"),
            ]
            params = cfg.get("strategy_params")
            if isinstance(params, dict):
                accepted = params.get("accepted_signal_strategy_types")
                if isinstance(accepted, str):
                    raw_values.extend(part.strip() for part in accepted.split(","))
                elif isinstance(accepted, (list, tuple, set)):
                    raw_values.extend(accepted)
            normalized: list[str] = []
            seen: set[str] = set()
            for raw_value in raw_values:
                value = str(raw_value or "").strip().lower()
                if not value or value in seen:
                    continue
                seen.add(value)
                normalized.append(value)
            if normalized:
                accepted_by_source[source_key] = normalized
        return accepted_by_source

    def _strategy_params_for_source(self, source_key: str) -> dict[str, Any]:
        configs = self._trader.get("source_configs") or []
        src_key = str(source_key or "").strip().lower()
        for cfg in configs:
            if not isinstance(cfg, dict):
                continue
            if str(cfg.get("source_key") or "").strip().lower() != src_key:
                continue
            params = cfg.get("strategy_params")
            return dict(params) if isinstance(params, dict) else {}
        return {}

    def _strategy_key_for_source(self, source_key: str) -> Optional[str]:
        configs = self._trader.get("source_configs") or []
        src_key = str(source_key or "").strip().lower()
        for cfg in configs:
            if not isinstance(cfg, dict):
                continue
            if str(cfg.get("source_key") or "").strip().lower() != src_key:
                continue
            sk = str(cfg.get("strategy_key") or "").strip().lower()
            return sk or None
        return None

    def _decision_payload(
        self,
        *,
        signal: Any,
        source_key: str,
        strategy_key: str,
        mode: str,
        evaluated_size: float | None = None,
        result_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "fast_tier": True,
            "mode": mode,
            "source_config": {"source_key": source_key, "strategy_key": strategy_key},
            "signal": {
                "id": str(getattr(signal, "id", "") or ""),
                "market_id": str(getattr(signal, "market_id", "") or ""),
                "runtime_sequence": getattr(signal, "runtime_sequence", None),
                "created_at": getattr(getattr(signal, "created_at", None), "isoformat", lambda: None)(),
                "expires_at": getattr(getattr(signal, "expires_at", None), "isoformat", lambda: None)(),
            },
        }
        if evaluated_size is not None:
            payload["evaluated_size_usd"] = evaluated_size
        if result_payload:
            payload["submit_result"] = result_payload
        return payload

    def _checks_summary(self, decision: Any) -> dict[str, Any]:
        checks = getattr(decision, "checks", None) or []
        checks_payload: list[dict[str, Any]] = []
        for check in checks:
            if isinstance(check, dict):
                checks_payload.append(dict(check))
                continue
            checks_payload.append(
                {
                    "key": str(getattr(check, "key", "") or "check"),
                    "label": str(getattr(check, "label", "") or "Check"),
                    "passed": bool(getattr(check, "passed", False)),
                    "score": getattr(check, "score", None),
                    "detail": getattr(check, "detail", None),
                    "payload": getattr(check, "payload", None) or {},
                }
            )
        return {"fast_tier": True, "checks": checks_payload}

    def _enriched_strategy_reason(
        self, decision: Any, base_reason: str, final_decision: str
    ) -> str:
        # Mirror the slow-tier worker so generic skip strings like
        # "Crypto worker filters not met" get the specific failed checks
        # appended.  enrich_final_reason expects "check_key"/"check_label"
        # keys, so translate from the strategy's DecisionCheck shape.
        checks = getattr(decision, "checks", None) or []
        checks_payload: list[dict[str, Any]] = []
        for check in checks:
            if isinstance(check, dict):
                checks_payload.append(
                    {
                        "check_key": str(check.get("check_key") or check.get("key") or "check"),
                        "check_label": str(check.get("check_label") or check.get("label") or "Check"),
                        "passed": bool(check.get("passed", False)),
                        "score": check.get("score"),
                        "detail": check.get("detail"),
                        "payload": check.get("payload") or {},
                    }
                )
                continue
            checks_payload.append(
                {
                    "check_key": str(getattr(check, "key", "") or "check"),
                    "check_label": str(getattr(check, "label", "") or "Check"),
                    "passed": bool(getattr(check, "passed", False)),
                    "score": getattr(check, "score", None),
                    "detail": getattr(check, "detail", None),
                    "payload": getattr(check, "payload", None) or {},
                }
            )
        return enrich_final_reason(
            final_decision=final_decision,
            final_reason=base_reason,
            checks_payload=checks_payload,
        )

    def _accumulate_stage_ms(self, stage: str, started_mono: float) -> None:
        # Per-cycle timing accumulator.  ``_process_one`` may run more
        # than once per cycle when several markets fire at the same
        # instant — summing into one bucket keeps the budget-exceeded
        # log readable while still surfacing where the wall-clock went.
        elapsed_ms = (time.monotonic() - started_mono) * 1000.0
        existing = float(self._last_stage_timings_ms.get(stage) or 0.0)
        self._last_stage_timings_ms[stage] = round(existing + elapsed_ms, 1)

    async def _touch_trader_run(self, session, *, force: bool = False) -> bool:
        now_mono = time.monotonic()
        if not force and now_mono - self._last_trader_touch_at < _TRADER_LAST_RUN_TOUCH_SECONDS:
            return False
        trader_id = self.trader_id
        if not trader_id:
            return False
        row = await session.get(Trader, trader_id)
        if row is None:
            return False
        now = utcnow().replace(tzinfo=None)
        row.last_run_at = now
        row.updated_at = now
        self._last_trader_touch_at = now_mono
        return True

    async def _maybe_emit_idle_event(
        self,
        session,
        *,
        accepted_sources: list[str],
        cursor_runtime_sequence: Optional[int],
        cursor_created_at: Any,
        cursor_signal_id: Any,
    ) -> bool:
        del session
        now_mono = time.monotonic()
        if now_mono - self._last_idle_event_at < _IDLE_EVENT_INTERVAL_SECONDS:
            return False
        await hot_state.buffer_trader_event(
            trader_id=self.trader_id,
            event_type="fast_cycle_heartbeat",
            severity="info",
            source=WORKER_NAME,
            message="Fast cycle idle: no pending signals.",
            payload={
                "fast_tier": True,
                "accepted_sources": accepted_sources,
                "cursor_runtime_sequence": cursor_runtime_sequence,
                "cursor_created_at": getattr(cursor_created_at, "isoformat", lambda: None)(),
                "cursor_signal_id": str(cursor_signal_id or "") or None,
            },
        )
        self._last_idle_event_at = now_mono
        return True

    async def _record_signal_event(
        self,
        session,
        *,
        signal: Any,
        event_type: str,
        severity: str,
        source_key: str,
        strategy_key: str | None,
        message: str,
        reason: str | None = None,
    ) -> None:
        del session
        await hot_state.buffer_trader_event(
            trader_id=self.trader_id,
            event_type=event_type,
            severity=severity,
            source=WORKER_NAME,
            message=message,
            payload={
                "fast_tier": True,
                "reason": reason,
                "source_key": source_key,
                "strategy_key": strategy_key,
                "signal_id": str(getattr(signal, "id", "") or ""),
                "market_id": str(getattr(signal, "market_id", "") or ""),
                "runtime_sequence": getattr(signal, "runtime_sequence", None),
            },
        )

    async def _buffer_decision(
        self,
        *,
        signal: Any,
        source_key: str,
        strategy_key: str,
        mode: str,
        decision: str,
        reason: str | None,
        score: Any = None,
        checks_summary: dict[str, Any] | None = None,
        risk_snapshot: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str:
        return await hot_state.buffer_decision(
            trader_id=self.trader_id,
            signal_id=str(getattr(signal, "id", "") or ""),
            signal_source=str(getattr(signal, "source", "") or source_key),
            strategy_key=strategy_key,
            strategy_version=None,
            decision=decision,
            reason=reason,
            score=score,
            trace_id=None,
            checks_summary=checks_summary or {"fast_tier": True, "checks": []},
            risk_snapshot=risk_snapshot or {"fast_tier": True},
            payload=payload
            or self._decision_payload(
                signal=signal,
                source_key=source_key,
                strategy_key=strategy_key,
                mode=mode,
            ),
        )

    async def _advance_cursor_buffered(
        self,
        *,
        signal: Any,
        mode: str,
        decision_id: str | None,
        outcome: str,
        reason: str | None,
    ) -> None:
        signal_id = str(getattr(signal, "id", "") or "").strip()
        if not signal_id:
            return
        # Mark consumed in the in-memory signal_cache BEFORE writing to
        # DB so subsequent cache reads (which may race) correctly skip
        # this signal.  The cache's consumed-set is per-trader and
        # idempotent, so re-marking is harmless.
        try:
            from services import signal_cache as _signal_cache

            _signal_cache.get_signal_cache().mark_consumed(self.trader_id, signal_id)
        except Exception:
            pass
        signal_status = str(outcome or "").strip().lower()
        if signal_status == "blocked":
            signal_status = "skipped"
        if signal_status not in {"executed", "skipped", "expired", "failed", "submitted", "selected"}:
            signal_status = "failed" if signal_status == "error" else "skipped"
        effective_price = getattr(signal, "effective_price", None)
        try:
            await get_intent_runtime().update_signal_status(
                signal_id=signal_id,
                status=signal_status,
                effective_price=effective_price,
            )
        except Exception:
            await hot_state.buffer_signal_status(
                signal_id=signal_id,
                status=signal_status,
                effective_price=effective_price,
            )
        created_at = getattr(signal, "created_at", None) or utcnow()
        runtime_sequence = safe_int(getattr(signal, "runtime_sequence", None), None)
        hot_state.update_signal_cursor(
            self.trader_id,
            mode,
            created_at,
            signal_id,
            runtime_sequence,
        )
        await hot_state.buffer_signal_consumption(
            trader_id=self.trader_id,
            signal_id=signal_id,
            outcome=outcome,
            reason=reason,
            decision_id=decision_id,
        )
        await hot_state.buffer_signal_cursor(
            trader_id=self.trader_id,
            last_signal_created_at=created_at,
            last_signal_id=signal_id,
            last_runtime_sequence=runtime_sequence,
        )

    async def _process_signals_parallel_by_market(
        self,
        signals: list[Any],
        *,
        mode: str,
        default_size_usd: float,
    ) -> None:
        """Process signals in parallel, but serialize within each market.

        Each ``_process_one`` is dominated by the CLOB submission network
        roundtrip (~300-500ms).  Running the four signals of a cycle
        sequentially produces ~1.6s cycles; running them in parallel
        collapses that to roughly the slowest single network call.

        Signals that target the *same* market_id, however, must stay
        serial — the strategy's ``evaluate`` step consults
        ``trader_hot_state`` to decide whether the trader is already
        positioned on that market, and racing two concurrent submits on
        the same market can defeat the per-market position cap.  Grouping
        by ``market_id`` and running each group serially while the groups
        execute concurrently preserves the cap while still capturing the
        bulk of the speedup (most cycles see signals on different
        markets).
        """
        if not signals:
            return
        trader_id = self.trader_id
        groups: dict[str, list[Any]] = {}
        for signal in signals:
            market_key = str(getattr(signal, "market_id", "") or "").strip().lower()
            # Empty market_key all share the same bucket; this is degenerate
            # and means the signal is malformed, but the surrounding
            # _process_one will reject it without submitting.
            groups.setdefault(market_key, []).append(signal)

        async def _process_group(group_signals: list[Any]) -> None:
            for signal in group_signals:
                try:
                    await self._process_one(
                        None,
                        signal=signal,
                        mode=mode,
                        default_size_usd=default_size_usd,
                    )
                except Exception as exc:
                    if _is_retryable_db_error(exc):
                        logger.debug(
                            "Fast trader signal processing hit retryable DB error",
                            exc_info=exc,
                        )
                    logger.warning(
                        "Fast trader signal processing failed",
                        trader_id=trader_id,
                        signal_id=getattr(signal, "id", None),
                        exc_info=exc,
                    )

        if len(groups) == 1:
            # Single market — no benefit from gather, save the overhead.
            await _process_group(next(iter(groups.values())))
            return
        await asyncio.gather(
            *(_process_group(group) for group in groups.values()),
            return_exceptions=True,
        )

    async def _run_once(self) -> None:
        trader_id = self.trader_id
        accepted_sources = self._accepted_sources()
        if not accepted_sources:
            return

        # ARCHITECTURAL CONTRACT: fast trader hot path MUST NOT call
        # Polymarket REST APIs.  Wallet state is push-driven via the
        # CLOB user-channel WS into ``WalletStateCache``.  See
        # ``services.trader_orchestrator.hot_path`` for the contextvar
        # gate that enforces this.
        with hot_path_no_rest():
            await self._run_once_inner(trader_id, accepted_sources)

    async def _run_once_inner(self, trader_id: str, accepted_sources: list[str]) -> None:
        # Freshness gate: refuse to trade on stale wallet state in live
        # mode.  This should NEVER fire after bootstrap completes —
        # ``live_execution_service.initialize()`` performs a synchronous
        # REST seed before returning success, and the user-channel WS
        # is connected by then.  If we hit this gate after bootstrap it
        # indicates a real problem (WS dropped, reconciliation worker
        # dead, etc.) and the warning is signal, not noise.
        mode = str(self._trader.get("mode", "shadow")).strip().lower() or "shadow"
        if mode == "live":
            ws_cache = get_wallet_state_cache()
            fresh, fresh_reason = ws_cache.is_fresh()
            if not fresh:
                now_mono = time.monotonic()
                # Log on the transition from fresh→stale, and at most
                # once per 30s thereafter.  Fast trader cycles hit
                # this every 250ms when stale; unthrottled it's a
                # log flood that buries every other diagnostic.
                should_log = (
                    self._last_wallet_state_was_fresh
                    or now_mono - self._last_wallet_stale_log_mono >= 30.0
                )
                if should_log:
                    logger.warning(
                        "Fast trader cycle skipped: wallet state stale",
                        trader_id=trader_id,
                        reason=fresh_reason,
                        cache_stats=ws_cache.stats_snapshot(),
                    )
                    self._last_wallet_stale_log_mono = now_mono
                self._last_wallet_state_was_fresh = False
                return
            if not self._last_wallet_state_was_fresh:
                # Recovery transition — log once so operators can see
                # exactly when the gate cleared.
                logger.info(
                    "Fast trader wallet state recovered",
                    trader_id=trader_id,
                    cache_stats=ws_cache.stats_snapshot(),
                )
                self._last_wallet_state_was_fresh = True

        # Per-stage timing so when a cycle blows the 3s hard budget we can
        # see *which* stage stalled (intent_runtime list, db cursor read,
        # parallel processing, idle-touch commit). Captured into
        # self._last_stage_timings_ms; the budget-exceeded warning above
        # logs them, and they're also reported via worker stats.
        self._last_stage_timings_ms = {}
        stage_start = time.monotonic()

        # Plan 0032 Task 7: hydrate the per-trader consumed-set BEFORE the
        # first ``intent_runtime.list_unconsumed_signals`` call so the
        # post-fetch filter below has authoritative consumption history.
        # ``intent_runtime`` itself does not gate on consumption (see
        # ``intent_runtime.list_unconsumed_signals`` which drops
        # ``trader_id``); without this pre-hydrate every restart leaks
        # the entire backlog of "trader_order already exists" decisions
        # for signals the scanner keeps reactivating via the 180s
        # ``_UNCHANGED_SCANNER_TERMINAL_REACTIVATION_COOLDOWN_SECONDS``
        # cooldown. Soft-fail: a DB hiccup falls back to an empty set;
        # ``fast_submit``'s ``(trader_id, signal_id)`` idempotency-guard
        # still prevents double submission so the worst case is the
        # pre-Plan-0032 behaviour for that one trader.
        if not self._consumed_set_hydrated:
            hydrate_t0 = time.monotonic()
            consumed_ids: list[str] = []
            try:
                async with FastAsyncSessionLocal() as hydrate_session:
                    consumed_ids = await fetch_recent_consumed_signal_ids(
                        hydrate_session,
                        trader_id=trader_id,
                        hours=48,
                        limit=50_000,
                    )
            except Exception as exc:
                logger.warning(
                    "Fast trader pre-cycle consumed-set hydrate failed",
                    trader_id=trader_id,
                    exc_info=exc,
                )
            try:
                from services import signal_cache as _signal_cache

                _signal_cache.get_signal_cache().hydrate_trader_consumed_ids(
                    trader_id, consumed_ids
                )
            except Exception:
                pass
            self._consumed_set_hydrated = True
            self._last_stage_timings_ms["precycle_consumed_hydrate"] = round(
                (time.monotonic() - hydrate_t0) * 1000.0, 1
            )

        strategy_types_by_source = self._accepted_strategy_types_by_source()
        cursor_created_at = None
        cursor_signal_id = None
        cursor_runtime_sequence = _get_sequence_cursor(trader_id)
        runtime_signals = await get_intent_runtime().list_unconsumed_signals(
            trader_id=trader_id,
            sources=accepted_sources,
            statuses=["pending", "selected"],
            strategy_types_by_source=strategy_types_by_source,
            cursor_runtime_sequence=cursor_runtime_sequence,
            cursor_created_at=None,
            cursor_signal_id=None,
            limit=_MAX_SIGNALS_PER_CYCLE,
        )
        self._last_stage_timings_ms["runtime_list_signals"] = round(
            (time.monotonic() - stage_start) * 1000.0, 1
        )
        # Plan 0032 Task 7: post-filter ``intent_runtime``'s output by the
        # signal_cache's per-trader consumed-set. ``intent_runtime`` does
        # not perform this filter itself — see the ``del trader_id`` at
        # the top of ``IntentRuntime.list_unconsumed_signals``. Without
        # this filter the scanner's 180s reactivation cooldown for
        # ``skipped`` signals re-presents every ``TraderOrder``-already-
        # exists case to the trader, producing the duplicate-decision
        # spam Plan 0032 was opened against.
        if runtime_signals:
            try:
                from services import signal_cache as _signal_cache

                consumed_set = _signal_cache.get_signal_cache().consumed_ids_for(
                    trader_id
                )
            except Exception:
                consumed_set = frozenset()
            if consumed_set:
                before_count = len(runtime_signals)
                runtime_signals = [
                    sig
                    for sig in runtime_signals
                    if str(getattr(sig, "id", "") or "") not in consumed_set
                ]
                dropped = before_count - len(runtime_signals)
                if dropped:
                    self._last_stage_timings_ms["consumed_set_filtered"] = dropped
        if runtime_signals:
            mode = str(self._trader.get("mode", "shadow")).strip().lower() or "shadow"
            risk_limits = dict(self._trader.get("risk_limits") or {})
            default_size_usd = float(max(0.01, float(risk_limits.get("max_trade_notional_usd") or 1.0)))
            stage_start = time.monotonic()
            await self._process_signals_parallel_by_market(
                runtime_signals,
                mode=mode,
                default_size_usd=default_size_usd,
            )
            self._last_stage_timings_ms["process_runtime_signals"] = round(
                (time.monotonic() - stage_start) * 1000.0, 1
            )
            return
        # Fall back to task-local cached cursor before paying for a DB
        # read.  ``self._cached_cursor_*`` is populated the FIRST time
        # we cold-start through the DB path; from there on we re-use
        # it across cycles instead of re-reading TraderSignalCursor.
        if cursor_runtime_sequence is None and self._cached_cursor_loaded:
            cursor_runtime_sequence = self._cached_cursor_runtime_sequence
            cursor_created_at = self._cached_cursor_created_at
            cursor_signal_id = self._cached_cursor_signal_id

        if cursor_runtime_sequence is not None:
            if is_db_pressure_active():
                return
            await self._maybe_emit_idle_event(
                None,
                accepted_sources=accepted_sources,
                cursor_runtime_sequence=cursor_runtime_sequence,
                cursor_created_at=cursor_created_at,
                cursor_signal_id=cursor_signal_id,
            )
            return

        # In-memory cache is the ONLY hot-path source.  Populated by:
        #   * Bootstrap-from-DB at trading-plane startup (one shot)
        #   * Redis ``signal_payloads`` channel for every subsequent
        #     signal insert / status flip (sub-millisecond push)
        #   * Bootstrap re-run on every Redis (re)subscribe to
        #     reconcile publishes dropped during a connection gap
        #
        # The cold-start branch below ONLY runs while the cache hasn't
        # yet been bootstrapped (the brief window before the eager
        # bootstrap completes) AND this trader has not been seen yet.
        # In steady state the DB is NEVER touched on this path.
        from services import signal_cache as _signal_cache

        cache = _signal_cache.get_signal_cache()
        cache_ready = cache.is_ready()
        trader_hydrated = cache.is_trader_hydrated(trader_id)

        if cache_ready and trader_hydrated:
            # FAST PATH — pure in-memory dict lookup, no session
            # checkout, no DB queries, no event-loop yield points
            # beyond the one inside the lock.
            cache_t0 = time.monotonic()
            signals = cache.get_unconsumed_signals(
                trader_id=trader_id,
                sources=accepted_sources,
                cursor_runtime_sequence=cursor_runtime_sequence,
                limit=_MAX_SIGNALS_PER_CYCLE,
            )
            self._last_stage_timings_ms["signal_cache_hit"] = round(
                (time.monotonic() - cache_t0) * 1000.0, 1
            )
            self._last_stage_timings_ms["signal_source"] = "cache"

            if not signals:
                # Idle: only open a DB session when actual writes are
                # due (touch every 30s, idle event every 5min).  Most
                # idle cycles exit here without touching the pool.
                if is_db_pressure_active():
                    return
                touch_due = (
                    time.monotonic() - self._last_trader_touch_at
                    >= _TRADER_LAST_RUN_TOUCH_SECONDS
                )
                idle_event_due = (
                    time.monotonic() - self._last_idle_event_at
                    >= _IDLE_EVENT_INTERVAL_SECONDS
                )
                if not touch_due and not idle_event_due:
                    return
                idle_t0 = time.monotonic()
                async with FastAsyncSessionLocal() as session:
                    touched = await self._touch_trader_run(session, force=False)
                    emitted = await self._maybe_emit_idle_event(
                        session,
                        accepted_sources=accepted_sources,
                        cursor_runtime_sequence=cursor_runtime_sequence,
                        cursor_created_at=cursor_created_at,
                        cursor_signal_id=cursor_signal_id,
                    )
                    if touched or emitted:
                        try:
                            await asyncio.shield(session.commit())
                        except Exception as exc:
                            logger.warning(
                                "Fast trader idle-cycle commit failed",
                                trader_id=trader_id,
                                exc_info=exc,
                            )
                self._last_stage_timings_ms["idle_touch_commit"] = round(
                    (time.monotonic() - idle_t0) * 1000.0, 1
                )
                return
        else:
            # COLD-START PATH — open a session, read the cursor once,
            # fetch from DB, seed the cache, hydrate this trader.
            # Subsequent cycles short-circuit on the FAST PATH above.
            stage_start = time.monotonic()
            async with FastAsyncSessionLocal() as session:
                self._last_stage_timings_ms["session_checkout"] = round(
                    (time.monotonic() - stage_start) * 1000.0, 1
                )
                cursor_t0 = time.monotonic()
                cursor_created_at, cursor_signal_id = await get_trader_signal_cursor(
                    session, trader_id=trader_id
                )
                cursor_row = await session.get(TraderSignalCursor, trader_id)
                cursor_runtime_sequence = safe_int(
                    getattr(cursor_row, "last_runtime_sequence", None), None
                )
                self._last_stage_timings_ms["coldstart_cursor_read"] = round(
                    (time.monotonic() - cursor_t0) * 1000.0, 1
                )
                # Cache the cursor on the task so future cold-start
                # cycles (until cache_ready flips on) skip the DB read.
                self._cached_cursor_runtime_sequence = cursor_runtime_sequence
                self._cached_cursor_created_at = cursor_created_at
                self._cached_cursor_signal_id = cursor_signal_id
                self._cached_cursor_loaded = True

                db_t0 = time.monotonic()
                signals = await list_unconsumed_trade_signals(
                    session,
                    trader_id=trader_id,
                    sources=accepted_sources,
                    strategy_types_by_source=strategy_types_by_source,
                    cursor_runtime_sequence=cursor_runtime_sequence,
                    cursor_created_at=cursor_created_at,
                    cursor_signal_id=cursor_signal_id,
                    limit=_MAX_SIGNALS_PER_CYCLE,
                    # Fast trader never reads payload_json /
                    # strategy_context_json; deferring them turns this
                    # SELECT from a 200-row × 3KB JSON-decode-on-loop
                    # blocker into a header-only fetch.
                    defer_heavy_columns=True,
                )
                self._last_stage_timings_ms["coldstart_db_list"] = round(
                    (time.monotonic() - db_t0) * 1000.0, 1
                )
                # Plan 0032 Task 7: per-trader consumed-set is now
                # hydrated unconditionally at the top of ``_run_once_inner``
                # before the first ``intent_runtime.list_unconsumed_signals``
                # call. The cold-start fall-through no longer needs its
                # own hydrate — re-running it here would just re-issue the
                # same DB read for no benefit.
                # Seed the cache so the NEXT cycle takes the fast path
                # even if the eager bootstrap hasn't completed yet.
                for db_sig in signals:
                    snap = _snapshot_from_db_row(db_sig)
                    if snap is not None:
                        cache.upsert(snap)
                self._last_stage_timings_ms["signal_source"] = "db_coldstart"

                if not signals:
                    if is_db_pressure_active():
                        return
                    idle_t0 = time.monotonic()
                    touched = await self._touch_trader_run(session, force=False)
                    emitted = await self._maybe_emit_idle_event(
                        session,
                        accepted_sources=accepted_sources,
                        cursor_runtime_sequence=cursor_runtime_sequence,
                        cursor_created_at=cursor_created_at,
                        cursor_signal_id=cursor_signal_id,
                    )
                    if touched or emitted:
                        try:
                            await asyncio.shield(session.commit())
                        except Exception as exc:
                            logger.warning(
                                "Fast trader idle-cycle commit failed",
                                trader_id=trader_id,
                                exc_info=exc,
                            )
                    self._last_stage_timings_ms["idle_touch_commit"] = round(
                        (time.monotonic() - idle_t0) * 1000.0, 1
                    )
                    return

        mode = str(self._trader.get("mode", "shadow")).strip().lower() or "shadow"
        risk_limits = dict(self._trader.get("risk_limits") or {})
        default_size_usd = float(max(0.01, float(risk_limits.get("max_trade_notional_usd") or 1.0)))

        stage_start = time.monotonic()
        await self._process_signals_parallel_by_market(
            signals,
            mode=mode,
            default_size_usd=default_size_usd,
        )
        self._last_stage_timings_ms["process_signals_parallel"] = round(
            (time.monotonic() - stage_start) * 1000.0, 1
        )

    async def _process_one(
        self,
        session,
        *,
        signal: Any,
        mode: str,
        default_size_usd: float,
    ) -> None:
        trader_id = self.trader_id
        # Per-bot block_new_orders is the fast-tier analogue of the global
        # kill switch. Advance the cursor so this signal won't be re-fed
        # next cycle; the trader's exit/reconciliation paths are unaffected.
        if bool(self._trader.get("block_new_orders")):
            await self._advance_cursor_buffered(
                signal=signal,
                mode=mode,
                decision_id=None,
                outcome="skipped",
                reason="trader_block_new_orders",
            )
            return
        source_key = str(getattr(signal, "source", "") or "").strip().lower()
        allowed_strategy_types = set(self._accepted_strategy_types_by_source().get(source_key) or [])
        signal_strategy_type = str(getattr(signal, "strategy_type", "") or "").strip().lower()
        if allowed_strategy_types and signal_strategy_type not in allowed_strategy_types:
            await self._advance_cursor_buffered(
                signal=signal,
                mode=mode,
                decision_id=None,
                outcome="skipped",
                reason=(
                    "source_strategy_filter:"
                    f"signal={signal_strategy_type or 'unknown'};"
                    f"allowed={','.join(sorted(allowed_strategy_types))}"
                ),
            )
            return

        strategy_key = self._strategy_key_for_source(source_key) or str(
            getattr(signal, "strategy_type", "") or ""
        ).strip().lower()
        if not strategy_key:
            await self._record_signal_event(
                session,
                signal=signal,
                event_type="fast_signal_skipped",
                severity="warning",
                source_key=source_key,
                strategy_key=None,
                message="Fast signal skipped: missing strategy key.",
                reason="missing_strategy_key",
            )
            await self._advance_cursor_buffered(
                signal=signal,
                mode=mode,
                decision_id=None,
                outcome="failed",
                reason="missing_strategy_key",
            )
            logger.debug("Fast trader skipped signal: no strategy_key", signal_id=getattr(signal, "id", None))
            return

        strategy = strategy_loader.get_instance(strategy_key)
        if strategy is None:
            decision_id = await self._buffer_decision(
                signal=signal,
                source_key=source_key,
                strategy_key=strategy_key,
                mode=mode,
                decision="failed",
                reason=f"Fast strategy not loaded: {strategy_key}",
                score=None,
                checks_summary={"fast_tier": True, "checks": []},
                risk_snapshot={"fast_tier": True},
                payload=self._decision_payload(
                    signal=signal,
                    source_key=source_key,
                    strategy_key=strategy_key,
                    mode=mode,
                ),
            )
            await self._record_signal_event(
                session,
                signal=signal,
                event_type="fast_signal_skipped",
                severity="warning",
                source_key=source_key,
                strategy_key=strategy_key,
                message="Fast signal skipped: strategy is not loaded.",
                reason="strategy_not_loaded",
            )
            await self._advance_cursor_buffered(
                signal=signal,
                mode=mode,
                decision_id=decision_id,
                outcome="failed",
                reason=f"Fast strategy not loaded: {strategy_key}",
            )
            logger.debug(
                "Fast trader skipped signal: strategy not loaded",
                strategy_key=strategy_key,
                signal_id=getattr(signal, "id", None),
            )
            return

        strategy_params = self._strategy_params_for_source(source_key)

        # Layer the strategy's loaded DB config UNDER the per-trader
        # params so evaluate() sees the same gate values as on_event().
        # Historically ``context["params"]`` was just the trader's
        # ``source_config.strategy_params`` — meaning a strategy gate
        # tuned via the strategy-manager UI applied to opportunity
        # generation but the trader silently re-imposed default
        # thresholds at decision time, requiring users to set the same
        # gate in two places.  Trader params still win (they're the
        # last layer) so any per-trader override behaves as before.
        merged_params: dict[str, Any] = {}
        strategy_db_config = getattr(strategy, "config", None)
        if isinstance(strategy_db_config, dict):
            merged_params.update(strategy_db_config)
        if isinstance(strategy_params, dict):
            merged_params.update(strategy_params)

        # Build a minimal evaluation context.  The fast path intentionally
        # skips live_market_context enrichment and heavy risk snapshots —
        # the strategy's own evaluate() is the gate, nothing else.
        context: dict[str, Any] = {
            "trader": self._trader,
            "params": merged_params,
            "mode": mode,
            "source_config": {"source_key": source_key, "strategy_key": strategy_key},
            "fast_tier": True,
        }

        try:
            _evaluate_started = time.monotonic()
            decision = await asyncio.wait_for(
                asyncio.to_thread(strategy.evaluate, signal, context),
                timeout=_EVALUATE_BUDGET_SECONDS,
            )
            self._accumulate_stage_ms("evaluate", _evaluate_started)
        except asyncio.TimeoutError:
            reason = f"Fast evaluate exceeded budget ({_EVALUATE_BUDGET_SECONDS}s)."
            decision_id = await self._buffer_decision(
                signal=signal,
                source_key=source_key,
                strategy_key=strategy_key,
                mode=mode,
                decision="failed",
                reason=reason,
                score=None,
                checks_summary={"fast_tier": True, "checks": []},
                risk_snapshot={"fast_tier": True},
                payload=self._decision_payload(
                    signal=signal,
                    source_key=source_key,
                    strategy_key=strategy_key,
                    mode=mode,
                ),
            )
            await self._record_signal_event(
                session,
                signal=signal,
                event_type="fast_evaluate_timeout",
                severity="warning",
                source_key=source_key,
                strategy_key=strategy_key,
                message="Fast strategy evaluate exceeded budget.",
                reason=f"evaluate_timeout:{_EVALUATE_BUDGET_SECONDS}",
            )
            await self._advance_cursor_buffered(
                signal=signal,
                mode=mode,
                decision_id=decision_id,
                outcome="failed",
                reason=reason,
            )
            logger.warning(
                "Fast trader evaluate() exceeded budget",
                trader_id=trader_id,
                signal_id=getattr(signal, "id", None),
                budget_s=_EVALUATE_BUDGET_SECONDS,
            )
            return
        except Exception as exc:
            reason = f"Fast evaluate raised: {type(exc).__name__}: {exc}"
            decision_id = await self._buffer_decision(
                signal=signal,
                source_key=source_key,
                strategy_key=strategy_key,
                mode=mode,
                decision="failed",
                reason=reason,
                score=None,
                checks_summary={"fast_tier": True, "checks": []},
                risk_snapshot={"fast_tier": True},
                payload=self._decision_payload(
                    signal=signal,
                    source_key=source_key,
                    strategy_key=strategy_key,
                    mode=mode,
                ),
            )
            await self._record_signal_event(
                session,
                signal=signal,
                event_type="fast_evaluate_error",
                severity="warning",
                source_key=source_key,
                strategy_key=strategy_key,
                message="Fast strategy evaluate raised.",
                reason=f"{type(exc).__name__}: {exc}",
            )
            await self._advance_cursor_buffered(
                signal=signal,
                mode=mode,
                decision_id=decision_id,
                outcome="failed",
                reason=reason,
            )
            logger.warning(
                "Fast trader evaluate() raised",
                trader_id=trader_id,
                signal_id=getattr(signal, "id", None),
                exc_info=exc,
            )
            return

        final_decision = str(getattr(decision, "decision", "failed") or "failed").strip().lower()
        reason = str(getattr(decision, "reason", "") or "")
        evaluated_size = float(getattr(decision, "size_usd", None) or default_size_usd)

        if final_decision != "selected":
            outcome = final_decision if final_decision in {"blocked", "failed", "skipped"} else "blocked"
            enriched_reason = self._enriched_strategy_reason(decision, reason, final_decision) or f"fast:{final_decision}"
            decision_id = await self._buffer_decision(
                signal=signal,
                source_key=source_key,
                strategy_key=strategy_key,
                mode=mode,
                decision=final_decision,
                reason=enriched_reason,
                score=getattr(decision, "score", None),
                checks_summary=self._checks_summary(decision),
                risk_snapshot={"fast_tier": True},
                payload=self._decision_payload(
                    signal=signal,
                    source_key=source_key,
                    strategy_key=strategy_key,
                    mode=mode,
                    evaluated_size=evaluated_size,
                ),
            )
            await self._advance_cursor_buffered(
                signal=signal,
                mode=mode,
                decision_id=decision_id,
                outcome=outcome,
                reason=enriched_reason,
            )
            return

        decision_id = uuid.uuid4().hex
        selected_decision_audit = {
            "decision": "selected",
            "reason": reason or "fast:selected",
            "score": getattr(decision, "score", None),
            "checks_summary": self._checks_summary(decision),
            "risk_snapshot": {"fast_tier": True},
            "payload": self._decision_payload(
                signal=signal,
                source_key=source_key,
                strategy_key=strategy_key,
                mode=mode,
                evaluated_size=evaluated_size,
            ),
        }

        async def _submit_and_persist(active_session) -> None:
            try:
                submit_started_at = time.monotonic()
                result = await execute_fast_signal(
                    active_session,
                    trader_id=trader_id,
                    signal=signal,
                    decision_id=decision_id,
                    decision_audit=selected_decision_audit,
                    strategy_key=strategy_key,
                    strategy_version=None,
                    strategy_params=strategy_params,
                    mode=mode,
                    size_usd=evaluated_size,
                    reason=reason or None,
                    risk_limits=dict(self._trader.get("risk_limits") or {}),
                )
                submit_duration_seconds = time.monotonic() - submit_started_at
                self._accumulate_stage_ms("submit", submit_started_at)
                if submit_duration_seconds > _SUBMIT_BUDGET_SECONDS:
                    logger.warning(
                        "Fast trader submit exceeded budget",
                        trader_id=trader_id,
                        signal_id=getattr(signal, "id", None),
                        duration_s=round(submit_duration_seconds, 3),
                        budget_s=_SUBMIT_BUDGET_SECONDS,
                    )
            except Exception as exc:
                failed_decision_id = await self._buffer_decision(
                    signal=signal,
                    source_key=source_key,
                    strategy_key=strategy_key,
                    mode=mode,
                    decision="failed",
                    reason=f"Fast submit raised: {type(exc).__name__}: {exc}",
                    score=getattr(decision, "score", None),
                    checks_summary=self._checks_summary(decision),
                    risk_snapshot={"fast_tier": True},
                    payload=self._decision_payload(
                        signal=signal,
                        source_key=source_key,
                        strategy_key=strategy_key,
                        mode=mode,
                        evaluated_size=evaluated_size,
                        result_payload={"submit_exception": type(exc).__name__},
                    ),
                )
                await self._record_signal_event(
                    active_session,
                    signal=signal,
                    event_type="fast_submit_error",
                    severity="warning",
                    source_key=source_key,
                    strategy_key=strategy_key,
                    message="Fast submit raised.",
                    reason=f"{type(exc).__name__}: {exc}",
                )
                await self._advance_cursor_buffered(
                    signal=signal,
                    mode=mode,
                    decision_id=failed_decision_id,
                    outcome="failed",
                    reason=f"Fast submit raised: {type(exc).__name__}: {exc}",
                )
                logger.warning(
                    "Fast trader submit raised",
                    trader_id=trader_id,
                    signal_id=getattr(signal, "id", None),
                    exc_info=exc,
                )
                return

            result_status = str(result.status or "").strip().lower()
            outcome_for_cursor = (
                "executed"
                if result.orders_written > 0 and result_status in _FAST_ORDER_SUCCESS_STATUSES
                else result_status
            )
            if result.orders_written <= 0:
                no_order_decision_id = await self._buffer_decision(
                    signal=signal,
                    source_key=source_key,
                    strategy_key=strategy_key,
                    mode=mode,
                    decision="failed" if result.status == "failed" else "skipped",
                    reason=result.error_message or reason or f"fast_submit:{result.status}",
                    score=getattr(decision, "score", None),
                    checks_summary=self._checks_summary(decision),
                    risk_snapshot={"fast_tier": True},
                    payload=self._decision_payload(
                        signal=signal,
                        source_key=source_key,
                        strategy_key=strategy_key,
                        mode=mode,
                        evaluated_size=evaluated_size,
                        result_payload=result.payload,
                    ),
                )
                await self._record_signal_event(
                    active_session,
                    signal=signal,
                    event_type="fast_submit_no_order",
                    severity="warning",
                    source_key=source_key,
                    strategy_key=strategy_key,
                    message="Fast submit did not write an order.",
                    reason=result.error_message or result.status,
                )
                await self._advance_cursor_buffered(
                    signal=signal,
                    mode=mode,
                    decision_id=no_order_decision_id,
                    outcome=outcome_for_cursor,
                    reason=result.error_message or reason,
                )
                return

            await self._advance_cursor_buffered(
                signal=signal,
                mode=mode,
                decision_id=decision_id,
                outcome=outcome_for_cursor,
                reason=result.error_message or reason,
            )

        if session is None:
            async with FastAsyncSessionLocal() as active_session:
                try:
                    await _submit_and_persist(active_session)
                    # 2026-05-05 hardening: explicit pending-rollback guard.
                    # Production observed cycles where ``_submit_and_persist``
                    # returned without raising but the session was left in
                    # SQLAlchemy's ``pending-rollback`` state by an inner
                    # flush whose own rollback failed silently. Calling
                    # ``commit()`` directly on that session raises
                    # PendingRollbackError, the outer rollback fails for
                    # the same reason, and the cycle's signal counts as
                    # "Fast trader signal processing failed" with a stack
                    # that obscures the original CLOB-side cause. Always
                    # rollback first if the session shows pending state,
                    # then only commit if there's still pending work to
                    # commit — this never destroys durably-flushed pre-
                    # submit rows (those committed in fast_submit.py:505
                    # before the network call) and prevents the cascade.
                    rolled_back_pending = False
                    try:
                        if _session_has_pending_rollback(active_session):
                            await active_session.rollback()
                            rolled_back_pending = True
                    except Exception as guard_exc:
                        logger.debug(
                            "Fast trader pre-commit pending-rollback guard failed",
                            trader_id=trader_id,
                            exc_info=guard_exc,
                        )
                    if not rolled_back_pending and active_session.in_transaction():
                        _persist_started = time.monotonic()
                        try:
                            await asyncio.shield(active_session.commit())
                            self._accumulate_stage_ms("persist", _persist_started)
                        except Exception as commit_exc:
                            # Last-resort: if commit raised PendingRollback
                            # (race window where ``in_transaction`` reported
                            # True but a parallel inner-task drain marked
                            # the session dirty between the check and the
                            # commit), absorb the error after rolling back
                            # so the worker keeps processing the next
                            # signal cleanly.
                            try:
                                await active_session.rollback()
                            except Exception:
                                pass
                            raise commit_exc
                except Exception:
                    try:
                        await active_session.rollback()
                    except Exception as rollback_exc:
                        logger.debug("Fast trader rollback failed", trader_id=trader_id, exc_info=rollback_exc)
                    raise
        else:
            await _submit_and_persist(session)


def _session_has_pending_rollback(session) -> bool:
    """Return True if SQLAlchemy considers the session to be in a state
    where the next operation will raise ``PendingRollbackError``.

    The relevant signal lives on the underlying ``SessionTransaction``
    (``_rollback_exception``); when it's set, every subsequent execute /
    commit / flush short-circuits with the cached error until rollback
    runs. We probe defensively because the attribute is private — fall
    back to False on any failure so we never block a healthy commit.
    """
    try:
        sync_session = getattr(session, "sync_session", None)
        if sync_session is None:
            return False
        transaction = getattr(sync_session, "_transaction", None)
        if transaction is None:
            return False
        rb_exc = getattr(transaction, "_rollback_exception", None)
        return rb_exc is not None
    except Exception:
        return False


# Module-level reference to the currently-active _FastRuntime instance.
# The event-bus subscriber is a module-level function (_dispatch_wake) that
# forwards to whichever runtime is current.  This indirection lets a
# restarted worker replace the previous runtime without leaving a
# closure-capturing callback alive in the EventBus — which was the primary
# path by which old _FastRuntime instances (with all their per-trader
# task dicts and trader config copies) stayed resident in memory.
_current_runtime: Optional["_FastRuntime"] = None


async def _dispatch_wake(event_type: str, data: dict[str, Any]) -> None:
    runtime = _current_runtime
    if runtime is None or runtime._stopping:
        return
    for wake in runtime._wake_events.values():
        wake.set()


class _FastRuntime:
    """Supervises the per-trader tasks + event bus subscription."""

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task] = {}
        self._wake_events: dict[str, asyncio.Event] = {}
        self._task_objs: dict[str, _FastTraderTask] = {}
        self._refresh_lock = asyncio.Lock()
        self._stopping = False
        self._last_heartbeat: float = 0.0

    async def _on_signal_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Kept for test compatibility — production path goes through _dispatch_wake."""
        if self._stopping:
            return
        for wake in self._wake_events.values():
            wake.set()

    async def _emit_heartbeat(self) -> None:
        now_mono = time.monotonic()
        if now_mono - self._last_heartbeat < _HEARTBEAT_INTERVAL_SECONDS:
            return
        self._last_heartbeat = now_mono
        task_snapshots = {
            trader_id: runner.snapshot(now_mono, _FAST_TASK_STALE_SECONDS)
            for trader_id, runner in self._task_objs.items()
        }
        stale_count = sum(1 for snapshot in task_snapshots.values() if snapshot.get("stale"))
        dead_count = sum(1 for task in self._tasks.values() if task.done())
        try:
            # Heartbeat uses the MAIN pool — the fast-tier pool is reserved
            # for the trading hot path, not worker-snapshot writes.  Keeping
            # it here caused _fast_on_checkin warnings ("held for 7.8s").
            async with AsyncSessionLocal() as session:
                await write_worker_snapshot(
                    session,
                    WORKER_NAME,
                    running=True,
                    enabled=True,
                    current_activity=(
                        f"Supervising {len(self._task_objs)} fast trader(s); "
                        f"stale={stale_count} dead={dead_count}"
                    ),
                    interval_seconds=int(_TRADER_REFRESH_INTERVAL_SECONDS),
                    stats={
                        "fast_trader_count": len(self._task_objs),
                        "stale_trader_tasks": stale_count,
                        "dead_trader_tasks": dead_count,
                        "task_snapshots": task_snapshots,
                    },
                )
        except DBAPIError as exc:
            if not _is_retryable_db_error(exc):
                logger.debug("Fast runtime heartbeat failed", exc_info=exc)
        except Exception as exc:
            logger.debug("Fast runtime heartbeat failed", exc_info=exc)

    async def _run_orphan_reconcile_sweep(self, *, label: str) -> None:
        """Run ``reconcile_orphaned_fast_submissions`` for every fast trader.

        Invoked at supervisor startup and periodically thereafter (see
        ``_ORPHAN_RECONCILE_INTERVAL_SECONDS``).  Each per-trader call
        opens its own short-lived session so a stuck trader can't
        starve the rest.  All exceptions are caught and logged — this
        sweep is best-effort cleanup, never load-bearing on the live
        trading path.
        """
        if self._stopping:
            return
        try:
            async with AsyncSessionLocal() as session:
                fast_trader_rows = list(await list_fast_traders(session))
        except Exception as exc:
            logger.warning(
                "Fast-tier orphan reconcile bootstrap failed",
                label=label,
                exc_info=exc,
            )
            return
        for trader_row in fast_trader_rows:
            if self._stopping:
                return
            trader_id_for_reconcile = str(trader_row.get("id") or "").strip()
            if not trader_id_for_reconcile:
                continue
            try:
                async with AsyncSessionLocal() as session:
                    result = await reconcile_orphaned_fast_submissions(
                        session,
                        trader_id=trader_id_for_reconcile,
                        commit=True,
                    )
                if result.get("eligible"):
                    # Operator-visible signal: emitted only when the
                    # sweep actually had work, so a healthy steady
                    # state stays quiet.  ``matched`` / ``marked_orphan``
                    # are the headline counts; ``venue_unreachable``
                    # flags partial passes (kept untouched + retried
                    # next sweep). ``matched_via_wallet_position`` and
                    # ``marked_orphan_with_wallet_evidence`` surface
                    # the 2026-05-05 fallback that links orphans to
                    # wallet positions when the venue's open-orders
                    # list missed the fill.
                    logger.info(
                        "Fast-tier orphan reconcile",
                        label=label,
                        trader_id=trader_id_for_reconcile,
                        eligible=result.get("eligible"),
                        matched=result.get("matched"),
                        matched_via_wallet_position=result.get(
                            "matched_via_wallet_position"
                        ),
                        marked_orphan=result.get("marked_orphan"),
                        marked_orphan_with_wallet_evidence=result.get(
                            "marked_orphan_with_wallet_evidence"
                        ),
                        venue_unreachable=result.get("venue_unreachable"),
                    )
            except asyncio.CancelledError:
                # Supervisor is shutting down — propagate so the outer
                # loop can finalize cleanup.
                raise
            except Exception as exc:
                logger.warning(
                    "Fast-tier orphan reconcile failed",
                    label=label,
                    trader_id=trader_id_for_reconcile,
                    exc_info=exc,
                )

    async def _stop_task(self, trader_id: str, runner: _FastTraderTask | None, task: asyncio.Task | None) -> None:
        if runner is not None:
            runner.stop()
        if task is None:
            return
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.CancelledError:
            pass
        except asyncio.TimeoutError:
            task.cancel()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Fast trader stopped with error", trader_id=trader_id, exc_info=exc)

    async def _refresh_roster(self) -> None:
        """Reconcile the set of running per-trader tasks with the DB roster."""
        async with self._refresh_lock:
            if self._stopping:
                return
            # Roster refresh is a slow-cadence maintenance read; use the
            # main pool so we do not touch the tight fast-tier pool for
            # anything other than the per-trader trading loop.
            async with AsyncSessionLocal() as session:
                control = await read_orchestrator_control(session)
                control_enabled = bool(control.get("is_enabled", False))
                control_paused = bool(control.get("is_paused", True))
                control_kill_switch = bool(control.get("kill_switch", False))
                control_mode = str(control.get("mode") or "shadow").strip().lower()
                if not control_enabled or control_paused or control_kill_switch:
                    traders = []
                else:
                    traders = [
                        trader
                        for trader in await list_fast_traders(session)
                        if str(trader.get("mode") or "shadow").strip().lower() == control_mode
                    ]
            seen_ids: set[str] = set()
            now_mono = time.monotonic()
            for trader in traders:
                trader_id = str(trader.get("id") or "").strip()
                if not trader_id:
                    continue
                seen_ids.add(trader_id)
                existing = self._task_objs.get(trader_id)
                existing_task = self._tasks.get(trader_id)
                if existing is not None and existing_task is not None and not existing_task.done():
                    existing.refresh_trader(trader)
                    if not existing.is_stale(now_mono, _FAST_TASK_STALE_SECONDS):
                        continue
                    logger.warning(
                        "Restarting stale fast trader task",
                        trader_id=trader_id,
                        snapshot=existing.snapshot(now_mono, _FAST_TASK_STALE_SECONDS),
                    )
                    await self._stop_task(trader_id, existing, existing_task)
                if existing is not None:
                    existing.stop()
                if existing_task is not None and existing_task.done():
                    try:
                        stopped_exc = existing_task.exception()
                    except asyncio.CancelledError:
                        stopped_exc = None
                    if stopped_exc is not None:
                        logger.warning("Restarting failed fast trader task", trader_id=trader_id, exc_info=stopped_exc)
                    else:
                        logger.warning("Restarting stopped fast trader task", trader_id=trader_id)
                self._task_objs.pop(trader_id, None)
                self._tasks.pop(trader_id, None)
                self._wake_events.pop(trader_id, None)
                wake = asyncio.Event()
                wake.set()  # Process pending work on first tick
                runner = _FastTraderTask(trader, wake)
                task = asyncio.create_task(runner.run(), name=f"fast-trader-{trader_id[:12]}")
                self._wake_events[trader_id] = wake
                self._task_objs[trader_id] = runner
                self._tasks[trader_id] = task
            stale_ids = [tid for tid in list(self._tasks.keys()) if tid not in seen_ids]
            for tid in stale_ids:
                runner = self._task_objs.pop(tid, None)
                task = self._tasks.pop(tid, None)
                self._wake_events.pop(tid, None)
                await self._stop_task(tid, runner, task)

    async def run(self) -> None:
        global _current_runtime
        logger.info("Fast-tier runtime supervisor started")
        try:
            await hot_state.seed()
        except Exception as exc:
            logger.warning("Fast-tier runtime hot-state seed failed", exc_info=exc)
        # Startup orphan-reconcile sweep.  Catches any ``in_flight`` /
        # ``clob_exception`` / ``post_update_failed`` skeleton rows from
        # a prior crash and matches them against the venue by their
        # deterministic fast_idempotency_key.  The same sweep runs
        # periodically (see ``_ORPHAN_RECONCILE_INTERVAL_SECONDS`` and
        # the call inside the supervisor loop) so orphans produced
        # mid-session — by a CancelledError teardown or network blip —
        # get cleared without waiting for a worker restart.
        await self._run_orphan_reconcile_sweep(label="startup")
        # Install ourselves as the active runtime BEFORE subscribing so the
        # dispatcher has a target to forward to.  The dedup in EventBus.subscribe
        # plus the module-level _dispatch_wake indirection guarantees the
        # callback list stays at length 1 per event across arbitrarily many
        # restarts (the old runtime instance becomes unreachable as soon as
        # _current_runtime is overwritten).
        _current_runtime = self
        for evt in _WAKE_EVENTS:
            event_bus.subscribe(evt, _dispatch_wake)

        try:
            await self._refresh_roster()
            await self._emit_heartbeat()
            last_orphan_sweep_mono = time.monotonic()
            while not self._stopping:
                try:
                    await asyncio.sleep(_TRADER_REFRESH_INTERVAL_SECONDS)
                except asyncio.CancelledError:
                    break
                try:
                    await self._refresh_roster()
                except Exception as exc:
                    logger.warning("Fast-tier runtime roster refresh failed", exc_info=exc)
                await self._emit_heartbeat()
                # Periodic orphan reconcile.  Runs in the supervisor
                # thread (not on a separate task) so it serializes
                # against roster refresh — both touch the fast-trader
                # roster, and serialising them is cheaper than holding
                # an extra lock.  The sweep is short (one venue HTTP
                # call + per-trader DB pass) and tolerates supervisor
                # cancellation cleanly.
                now_mono = time.monotonic()
                if (now_mono - last_orphan_sweep_mono) >= _ORPHAN_RECONCILE_INTERVAL_SECONDS:
                    last_orphan_sweep_mono = now_mono
                    try:
                        await self._run_orphan_reconcile_sweep(label="periodic")
                    except asyncio.CancelledError:
                        break
                    except Exception as exc:
                        logger.warning(
                            "Fast-tier periodic orphan reconcile sweep failed",
                            exc_info=exc,
                        )
        finally:
            self._stopping = True
            for trader_id, runner in list(self._task_objs.items()):
                await self._stop_task(trader_id, runner, self._tasks.get(trader_id))
            # Only clear the module pointer if we are still the current
            # runtime — a rapid restart may have already installed the next
            # instance by the time we reach finally.
            if _current_runtime is self:
                _current_runtime = None
                for evt in _WAKE_EVENTS:
                    try:
                        event_bus.unsubscribe(evt, _dispatch_wake)
                    except Exception:
                        pass
            # Proactively drop references so garbage collection can reclaim
            # per-trader state even if something else briefly holds a pointer
            # to this _FastRuntime.
            self._tasks.clear()
            self._wake_events.clear()
            self._task_objs.clear()
            logger.info("Fast-tier runtime supervisor stopped")


async def start_loop() -> None:
    """Worker entrypoint.  Exported at module level for the host registry."""
    runtime = _FastRuntime()
    try:
        await runtime.run()
    except asyncio.CancelledError:
        logger.info("Fast-tier runtime cancelled")
        raise
