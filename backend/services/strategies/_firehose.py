"""Crypto strategy firehose — terminal-visible per-gate evaluation events.

The trader Terminal tab only renders events emitted via
``buffer_trader_event``.  Crypto strategies historically silently
``return None`` at each rejection point; this helper makes those
decisions visible to the user, tagged with a verbosity tier so the
volume can be tuned in the UI.

Tiers (lowest → loudest, matched to the UI volume dial):

* ``WHISPER`` — every market evaluated every cycle, even ones that
  fail the cheapest gates (timeframe match, asset list, milestone
  not yet crossed).  Hundreds of events per minute under load.
* ``MURMUR``  — only markets that passed the cheap gates and died
  on a meaningful one (oracle freshness, distance, VWAP, book depth).
* ``VOICE``   — an Opportunity was emitted; passed every gate.
* ``SHOUT``   — orders / executions (used by the execution layer,
  not strategies).
* ``ALARM``   — errors / exceptions; emitted as ``severity="error"``
  and always shown regardless of the user's volume setting.

Firehose events use ``event_type="firehose_gate"`` (single-gate
rejections), ``event_type="firehose_evaluation"`` (full gate-by-gate
summary), and ``event_type="firehose_emit"`` (opportunity emitted).
All carry ``source="crypto"`` so the UI can route them to the Crypto
bot's Terminal tab.

Trader-id is intentionally ``None`` — these events describe global
strategy state, not a specific trader's decision flow.  The UI
matches them to a trader by ``source_key`` + the trader's enabled
strategies.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Iterable

import time as _time

from services.trader_hot_state import buffer_trader_event
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Strategy-emit eligibility cache
# ---------------------------------------------------------------------------
#
# Pre-fix, every strategy that ran on_event for a tick fired firehose
# evaluation events to the trader-events stream.  That meant:
#   * Strategies loaded but bound to no active trader (e.g. spike_
#     reversion when no trader has it in source_configs) still spammed
#     the Live Pulse feed with rejections.
#   * Events fired even when the orchestrator was disabled.
#   * Per-bot terminals leaked events from strategies the bot doesn't
#     run (because the events carried no trader_id).
#
# This cache resolves all three:
#   * ``_orchestrator_enabled`` → if False, suppress emit entirely.
#   * ``_strategy_to_trader_ids`` → the set of LIVE-mode trader_ids
#     that have ``strategy_slug`` in their ``source_configs``.  If
#     empty, suppress emit (no one consumes this strategy's signals).
#     Otherwise, tag the event payload with the list so per-bot
#     terminals can filter by ``trader_id IN bound_trader_ids``.
#
# TTL is short (3 s) — orchestrator enable / trader-config edits show
# up to firehose within a single refresh.

_BINDING_TTL_SECONDS = 3.0
# Hard ceiling on the cache age before we *must* block to refresh.
# Between TTL_SECONDS and STALE_HARD_SECONDS we serve the stale value
# and kick off a background refresh — that prevents a thundering herd
# of evaluation tasks from queueing behind one DB roundtrip when the
# DB momentarily slows down.  Above the hard ceiling we resume blocking
# behaviour so we never return a wildly outdated binding map.
_BINDING_STALE_HARD_SECONDS = 30.0
_orchestrator_enabled: bool = False
_strategy_to_trader_ids: dict[str, list[str]] = {}
_binding_cache_at: float = 0.0
# Round 5: replace {_binding_refresh_lock, _binding_refresh_inflight}
# with a single shared Future that every waiter can await in parallel.
#
# Pre-fix stall dumps showed x22-x27 concurrent emission tasks parked
# at _refresh_binding_cache_guarded:169 and x27-x54 at _tracked_emission:337.
# Two latent bugs drove the herd:
#
#   1. Race in the soft-stale path: ``if not _binding_refresh_inflight``
#      was a sync read, but the flag was set INSIDE the refresh task
#      (after the await boundary).  N concurrent callers all passed the
#      check and each scheduled its own refresh task — hence the x27
#      copies of _refresh_binding_cache_guarded in stall dumps.
#
#   2. Hard-stale path used ``async with _binding_refresh_lock:``, which
#      serialized N waiters through one DB roundtrip.  27 waiters × one
#      roundtrip queued the whole orchestrator behind the slowest path.
#
# The Future replaces both paths:
#   - Sync assignment (``_binding_refresh_future = loop.create_future()``)
#     claims the refresh slot atomically — concurrent readers either
#     observe the existing future or create one, never both.
#   - All waiters await the SAME future in parallel.  One refresh, N
#     wakeups simultaneously.
_binding_refresh_future: asyncio.Future[None] | None = None

# ---------------------------------------------------------------------------
# Fix OO — Firehose emission backpressure.
#
# Pre-fix observation: stall dumps showed 1000+ tasks parked at
# ``_firehose.py:emit_evaluation:327`` and ``emit_emit:391``.  Every
# crypto_update tick, six strategies each emit several gate/eval/emit
# events per market they consider — easily 300-600 fire-and-forget
# tasks per second.  When the binding cache refresh blocked on a slow
# DB query (or audit-write Redis publish hiccupped), tasks accumulated
# faster than they drained, saturating the event loop and pushing
# orchestrator cycles to 30-90 s.
#
# Firehose events are observability, not load-bearing.  Drop them
# under pressure rather than letting them gum up the event loop.  The
# budget below is generous enough to never bite during normal
# operation but hard-caps the leak surface area.
_INFLIGHT_TASK_BUDGET = 256
_inflight_emission_tasks: int = 0
_dropped_emission_tasks: int = 0


async def _refresh_binding_cache() -> None:
    """Pull orchestrator state + strategy→trader binding map from DB."""
    global _orchestrator_enabled, _strategy_to_trader_ids, _binding_cache_at
    try:
        from sqlalchemy import select
        from models.database import (
            AsyncSessionLocal,
            Trader,
            TraderOrchestratorControl,
        )

        async with AsyncSessionLocal() as session:
            control = await session.get(TraderOrchestratorControl, "default")
            orchestrator_enabled = bool(
                control and control.is_enabled and not control.is_paused and not control.kill_switch
            )
            traders = (
                (
                    await session.execute(
                        select(Trader).where(Trader.is_enabled.is_(True))
                    )
                )
                .scalars()
                .all()
            )
        new_map: dict[str, list[str]] = {}
        for trader in traders:
            mode_lower = str(getattr(trader, "mode", "") or "").strip().lower()
            if mode_lower != "live":
                continue
            cfgs = getattr(trader, "source_configs_json", None) or []
            if isinstance(cfgs, str):
                try:
                    import json as _json
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
                slug = str(cfg.get("strategy_key") or "").strip().lower()
                if slug:
                    new_map.setdefault(slug, []).append(str(trader.id))
        _orchestrator_enabled = orchestrator_enabled
        _strategy_to_trader_ids = new_map
        _binding_cache_at = _time.monotonic()
    except Exception as exc:
        logger.debug("firehose binding cache refresh failed", exc_info=exc)


def _binding_cache_fresh() -> bool:
    return (_time.monotonic() - _binding_cache_at) < _BINDING_TTL_SECONDS


def _binding_cache_hard_stale() -> bool:
    return (_time.monotonic() - _binding_cache_at) >= _BINDING_STALE_HARD_SECONDS


def _claim_or_join_refresh(
    loop: asyncio.AbstractEventLoop,
) -> tuple[asyncio.Future[None], bool]:
    """Atomically either claim the refresh slot or join an in-flight one.

    Returns ``(future, is_owner)``.  The owner must drive the refresh
    and set the future's result/exception; joiners just await the
    future.  Because this runs entirely synchronously between any two
    awaits, concurrent callers on the same event loop cannot both
    become owners — the asyncio single-threaded model guarantees
    atomicity of the check-and-set.
    """
    global _binding_refresh_future
    existing = _binding_refresh_future
    if existing is not None and not existing.done():
        return existing, False
    fut: asyncio.Future[None] = loop.create_future()
    _binding_refresh_future = fut
    return fut, True


async def _drive_refresh(fut: asyncio.Future[None]) -> None:
    """Run the refresh and publish the result on the shared future."""
    global _binding_refresh_future
    try:
        await _refresh_binding_cache()
        if not fut.done():
            fut.set_result(None)
    except BaseException as exc:  # noqa: BLE001 — must propagate to waiters
        if not fut.done():
            fut.set_exception(exc)
        raise
    finally:
        # Release the slot so the next cache expiry can start a new
        # refresh.  Only clear if we're still the published future —
        # don't stomp on a subsequent refresh that raced in.
        if _binding_refresh_future is fut:
            _binding_refresh_future = None


async def _ensure_binding_cache() -> None:
    if _binding_cache_fresh():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    fut, is_owner = _claim_or_join_refresh(loop)
    if is_owner:
        # Owner schedules the actual refresh as a background task so
        # the owner call itself doesn't block any longer than the
        # joiners.  If we're hard-stale the owner ALSO awaits the
        # future below — N waiters wake up simultaneously when the
        # single refresh task completes.
        loop.create_task(_drive_refresh(fut))
    # Soft-stale: serve the stale value and let the refresh run in the
    # background.  This preserves the behaviour that made soft-stale
    # cheap — strategy evaluations never pay a DB roundtrip on a
    # warm-but-expired cache.
    if not _binding_cache_hard_stale():
        return
    # Hard-stale: cache is too old to trust.  All N waiters now park
    # on the same Future (not a Lock), so when the single refresh
    # task completes they resume in parallel instead of serialising
    # through a lock.
    try:
        await fut
    except Exception:
        # Refresh failed — fall through and return; _emit_should_fire
        # will observe whatever is in the globals (typically the last
        # successful refresh).  Errors are logged inside
        # _refresh_binding_cache.
        return


async def _emit_should_fire(strategy_slug: str) -> tuple[bool, list[str]]:
    """Return (should_emit, bound_trader_ids).

    ``False, []`` means: drop the event entirely — either the
    orchestrator is off or the strategy has no live consumer.
    ``True, [trader_ids…]`` means: emit and tag the payload so per-
    bot terminals can filter.
    """
    await _ensure_binding_cache()
    if not _orchestrator_enabled:
        return False, []
    bound = _strategy_to_trader_ids.get(str(strategy_slug or "").strip().lower(), [])
    if not bound:
        return False, []
    return True, list(bound)


# Verbosity tiers — frontend's volume dial selects a minimum tier and
# everything at-or-louder is shown.  Order matters: WHISPER < MURMUR <
# VOICE < SHOUT.  ALARM is severity, not verbosity.
WHISPER = "whisper"
MURMUR = "murmur"
VOICE = "voice"
SHOUT = "shout"

_TIER_RANK = {WHISPER: 1, MURMUR: 2, VOICE: 3, SHOUT: 4}


def tier_rank(verbosity: str | None) -> int:
    if not verbosity:
        return 0
    return _TIER_RANK.get(str(verbosity).strip().lower(), 0)


@dataclass(slots=True)
class GateResult:
    """One gate evaluated for one market.

    ``passed`` → True/False/None (None = "not evaluated; earlier gate
    short-circuited and skipped this one").  ``score`` is whatever
    numeric the gate measures (distance bps, oracle age ms, VWAP
    price, etc.) — frontend renders it raw.
    """

    name: str          # short slug, e.g. "timeframe", "asset_enabled", "min_distance"
    label: str         # human-readable label for the UI
    passed: bool | None
    score: float | None = None
    detail: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "passed": self.passed,
            "score": self.score,
            "detail": self.detail,
        }


def _market_summary(market: dict[str, Any] | Any) -> dict[str, Any]:
    """Best-effort identifying fields for a market.

    Accepts the dict form crypto strategies see in ``crypto_update``
    payloads, or any object with the same attributes.
    """
    if isinstance(market, dict):
        get = market.get
    else:
        get = lambda k, default=None: getattr(market, k, default)  # noqa: E731
    return {
        "market_id": str(get("condition_id") or get("id") or ""),
        "slug": str(get("slug") or ""),
        "question": str(get("question") or ""),
        "asset": str(get("asset") or get("symbol") or get("coin") or "").upper() or None,
        "timeframe": str(get("timeframe") or "") or None,
    }


def _fire_and_forget(coro) -> None:
    """Schedule an emission without blocking the caller.

    Strategies run inside the market_runtime dispatch loop; we don't
    want gate emissions to add latency to the hot path.  If no event
    loop is available (sync test path), drop the event silently.

    Fix OO — drop emissions when the in-flight budget is saturated.
    Firehose events are debug observability and must never queue
    enough tasks to saturate the event loop or stall the orchestrator.
    """
    global _inflight_emission_tasks, _dropped_emission_tasks
    if _inflight_emission_tasks >= _INFLIGHT_TASK_BUDGET:
        try:
            coro.close()
        except Exception:
            pass
        _dropped_emission_tasks += 1
        # Log every 1000th drop so the situation is visible without
        # spamming the log itself when the firehose is over-budget.
        if _dropped_emission_tasks % 1000 == 1:
            logger.warning(
                "firehose dropping emissions (in-flight budget exhausted)",
                extra={
                    "inflight": _inflight_emission_tasks,
                    "budget": _INFLIGHT_TASK_BUDGET,
                    "total_dropped": _dropped_emission_tasks,
                },
            )
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        try:
            coro.close()
        except Exception:
            pass
        return
    _inflight_emission_tasks += 1
    task = loop.create_task(_tracked_emission(coro))
    # Avoid 'Task was destroyed but it is pending' warnings if the
    # loop tears down before the task runs.
    task.add_done_callback(lambda _t: None)


async def _tracked_emission(coro) -> None:
    global _inflight_emission_tasks
    try:
        await coro
    finally:
        _inflight_emission_tasks -= 1


def get_firehose_stats() -> dict[str, int]:
    """Expose in-flight / dropped counters for observability."""
    return {
        "inflight_emission_tasks": _inflight_emission_tasks,
        "dropped_emission_tasks_total": _dropped_emission_tasks,
        "inflight_budget": _INFLIGHT_TASK_BUDGET,
    }


async def emit_gate(
    *,
    strategy_slug: str,
    market: dict[str, Any] | Any,
    gate: GateResult,
    verbosity: str = MURMUR,
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit a single-gate decision (typically a rejection)."""
    should_emit, bound_trader_ids = await _emit_should_fire(strategy_slug)
    if not should_emit:
        return
    market_info = _market_summary(market)
    pass_word = "passed" if gate.passed else ("skipped" if gate.passed is None else "rejected")
    msg = (
        f"{strategy_slug} • {market_info.get('slug') or market_info.get('market_id') or '?'} • "
        f"{gate.label}: {pass_word}"
    )
    if gate.detail:
        msg += f" — {gate.detail}"
    payload: dict[str, Any] = {
        "strategy_slug": strategy_slug,
        "source_key": "crypto",
        "market": market_info,
        "gate": gate.to_payload(),
        "bound_trader_ids": bound_trader_ids,
    }
    if extra:
        payload.update(extra)
    try:
        await buffer_trader_event(
            event_type="firehose_gate",
            severity="info",
            verbosity=verbosity,
            source="crypto",
            message=msg,
            payload=payload,
        )
    except Exception as exc:  # never let firehose break a strategy
        logger.debug("firehose emit_gate failed: %s", exc)


def emit_gate_nowait(
    *,
    strategy_slug: str,
    market: dict[str, Any] | Any,
    gate: GateResult,
    verbosity: str = MURMUR,
    extra: dict[str, Any] | None = None,
) -> None:
    """Sync convenience — schedule emission and return immediately.

    Use from sync code paths (most strategy gates).  Hot path is
    unaffected.
    """
    _fire_and_forget(
        emit_gate(
            strategy_slug=strategy_slug,
            market=market,
            gate=gate,
            verbosity=verbosity,
            extra=extra,
        )
    )


async def emit_evaluation(
    *,
    strategy_slug: str,
    market: dict[str, Any] | Any,
    gates: Iterable[GateResult],
    outcome: str,                 # "emitted" | "rejected" | "skipped"
    verbosity: str = WHISPER,
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit a full gate-by-gate evaluation summary for one market.

    Use this for WHISPER mode when you want to record the entire
    decision tree, including gates that didn't run because an
    earlier one short-circuited.
    """
    should_emit, bound_trader_ids = await _emit_should_fire(strategy_slug)
    if not should_emit:
        return
    market_info = _market_summary(market)
    gate_list = [g.to_payload() for g in gates]
    failed = [g for g in gate_list if g.get("passed") is False]
    summary = (
        f"{strategy_slug} • {market_info.get('slug') or market_info.get('market_id') or '?'} • "
        f"{outcome.upper()}"
    )
    if outcome == "rejected" and failed:
        summary += f" at {failed[0].get('label') or failed[0].get('name')}"
    payload: dict[str, Any] = {
        "strategy_slug": strategy_slug,
        "source_key": "crypto",
        "market": market_info,
        "outcome": outcome,
        "gates": gate_list,
        "bound_trader_ids": bound_trader_ids,
    }
    if extra:
        payload.update(extra)
    try:
        await buffer_trader_event(
            event_type="firehose_evaluation",
            severity="info",
            verbosity=verbosity,
            source="crypto",
            message=summary,
            payload=payload,
        )
    except Exception as exc:
        logger.debug("firehose emit_evaluation failed: %s", exc)


def emit_evaluation_nowait(
    *,
    strategy_slug: str,
    market: dict[str, Any] | Any,
    gates: Iterable[GateResult],
    outcome: str,
    verbosity: str = WHISPER,
    extra: dict[str, Any] | None = None,
) -> None:
    _fire_and_forget(
        emit_evaluation(
            strategy_slug=strategy_slug,
            market=market,
            gates=gates,
            outcome=outcome,
            verbosity=verbosity,
            extra=extra,
        )
    )


async def emit_emit(
    *,
    strategy_slug: str,
    market: dict[str, Any] | Any,
    detail: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    """An Opportunity was produced (passed every gate).  VOICE tier."""
    should_emit, bound_trader_ids = await _emit_should_fire(strategy_slug)
    if not should_emit:
        return
    market_info = _market_summary(market)
    msg = (
        f"{strategy_slug} • {market_info.get('slug') or market_info.get('market_id') or '?'} • "
        f"OPPORTUNITY EMITTED"
    )
    if detail:
        msg += f" — {detail}"
    payload: dict[str, Any] = {
        "strategy_slug": strategy_slug,
        "source_key": "crypto",
        "market": market_info,
        "detail": detail,
        "bound_trader_ids": bound_trader_ids,
    }
    if extra:
        payload.update(extra)
    try:
        await buffer_trader_event(
            event_type="firehose_emit",
            severity="info",
            verbosity=VOICE,
            source="crypto",
            message=msg,
            payload=payload,
        )
    except Exception as exc:
        logger.debug("firehose emit_emit failed: %s", exc)


def emit_emit_nowait(
    *,
    strategy_slug: str,
    market: dict[str, Any] | Any,
    detail: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    _fire_and_forget(
        emit_emit(
            strategy_slug=strategy_slug,
            market=market,
            detail=detail,
            extra=extra,
        )
    )
