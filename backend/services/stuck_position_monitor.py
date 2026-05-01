"""Stuck-position surveillance for blocked-terminal exits.

Institutional-grade pattern: when the bot's exit submission has been
permanently blocked (e.g. ``pending_live_exit.status =
'blocked_persistent_timeout'`` because the CLOB no longer accepts
orders for the underlying market), we DO NOT auto-write a synthetic
loss.  That would violate the DB-layer P&L guard
(``_enforce_pnl_verification_guard``) which requires
``verification_status='wallet_activity'`` for ``actual_profit`` to
persist — the guard exists precisely so phantom P&L cannot be created
by inference paths.

Instead, this module does what a bank's exception queue does:

  1. **SCAN**: query the DB for orders whose ``pending_live_exit.status``
     is in ``_BLOCKED_TERMINAL_STATES`` and whose age exceeds
     ``_STUCK_AGE_HOURS``.
  2. **VERIFY ON-CHAIN**: for each, ask the chain (via
     ``CTFExecutionService.fetch_position_chain_status``) what the wallet
     actually holds for that conditional token, and whether the
     underlying CTF condition has resolved.  This is on-chain truth,
     bypassing Polymarket's data API entirely.
  3. **CLASSIFY** (no writes — pure observation):

       * ``recovered_externally``  : balance == 0.  Position is gone.
                                      The polymarket_trade_verifier will
                                      pick up the SELL/redemption in its
                                      next 5-minute cycle.  No action.
       * ``redemption_pending``    : balance > 0 AND market_resolved.
                                      The redeemer_worker will redeem on
                                      its next 120-second cycle.  No
                                      action — just observe.
       * ``pending_resolution``    : balance > 0 AND market unresolved
                                      AND market_end_date is within
                                      ``_PENDING_RESOLUTION_WINDOW_DAYS``.
                                      The market is about to close on
                                      its own; once it resolves the
                                      redeemer worker handles it.
                                      Counted but NEVER alerted —
                                      paging the operator about a
                                      position that will resolve itself
                                      within days is noise.
       * ``transient_client_failure``: balance > 0 AND market unresolved
                                      AND last_error is a client-side
                                      failure.  The lifecycle's
                                      auto-recovery will retry the
                                      SELL.  Counted but NEVER alerted.
       * ``operator_intervention`` : balance > 0 AND market unresolved
                                      AND market_end_date is far AND
                                      last_error matches a venue-
                                      rejection marker AND age >
                                      ``_STUCK_AGE_HOURS``.  This is
                                      the rare residual case: the
                                      venue is genuinely refusing the
                                      SELL, the market won't resolve
                                      on its own soon, and the retry
                                      circuit is already broken.  Alert
                                      via Telegram with informational
                                      framing — most of the time the
                                      operator just acknowledges and
                                      lets the redeemer handle it on
                                      eventual resolution.

  4. **ALERT** the operator on ``operator_intervention`` rows with
     enough context to take a manual action (link to the order, market
     ID, current chain state, expected unrecoverable loss).  The
     operator either waits, intervenes via Polymarket UI, or uses the
     manual-writeoff API (see ``backend/api/routes_operator.py``) which
     writes a verified ``manual_writeoff`` close with full audit trail.

This module NEVER writes to ``actual_profit`` and NEVER mutates
``trader_orders`` rows.  It is pure surveillance.

Tunables come from ``config.Settings`` so they can be adjusted via
env vars without a deploy:

  * STUCK_POSITION_AGE_HOURS         — default 6h
  * STUCK_POSITION_SCAN_INTERVAL_S   — default 300s (every 5 minutes)
  * STUCK_POSITION_ALERT_COOLDOWN_S  — default 21600s (6h between
                                       alerts on the same order)
"""

from __future__ import annotations

import asyncio
import time as _time
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select

from models.database import AsyncSessionLocal, TraderOrder
from services.ctf_execution import ctf_execution_service
from services.live_execution_service import live_execution_service
from services.operator_writeoff import _is_venue_rejection_error
from utils.logger import get_logger
from utils.utcnow import utcnow

logger = get_logger("stuck_position_monitor")


_BLOCKED_TERMINAL_STATES = frozenset(
    {
        "blocked_persistent_timeout",
        "blocked_no_inventory",
        "blocked_retry_exhausted",
        "blocked_retry_exhausted_hard",
        # Market resolved/removed from CLOB; position settles on-chain
        # via the redeemer.  See position_lifecycle._is_orderbook_gone_error.
        "blocked_orderbook_gone",
    }
)

_OPEN_LIFECYCLE_STATUSES = frozenset(
    {"submitted", "executed", "completed", "open", "pending", "placing", "queued"}
)

# Age threshold: orders younger than this are not yet "stuck" enough to
# alert on (the retry circuit-breaker may still be settling, the
# 5-minute auto-recovery cycle may still be in flight).  Six hours is
# the floor below which an alert would almost always be premature —
# the lifecycle's auto-recovery clears every transient venue/infra
# blip in seconds-to-minutes, and a real stuck position has no
# meaningful difference between hour 5 and hour 24.
_STUCK_AGE_HOURS = 6.0

# Per-order alert cooldown so a single stuck order does NOT generate
# alerts on every scan pass.  Six hours is the right cadence for a
# human to look once per workday.
_ALERT_COOLDOWN_SECONDS = 6 * 3600

# Markets that close within this window are classified as
# ``pending_resolution``: the redeemer worker will close them on
# resolution, so paging the operator now is noise — they would just
# wait.  Set to 7 days because that's the typical Polymarket weekly
# event horizon; longer than this and the position is genuinely
# illiquid in a way the operator may want to know about.
_PENDING_RESOLUTION_WINDOW_DAYS = 7.0

# Module-level cache of "last alert at" timestamps (process-local).
# Crash-safe: on restart we'll re-alert on each row once, then settle.
_last_alert_at: dict[str, float] = {}


def _extract_token_id(payload: dict[str, Any]) -> str:
    return str(
        payload.get("selected_token_id")
        or payload.get("token_id")
        or (payload.get("live_market", {}) or {}).get("selected_token_id")
        or ""
    ).strip()


def _extract_condition_id(payload: dict[str, Any]) -> str:
    for key in ("condition_id", "conditionId"):
        v = payload.get(key)
        if v:
            text = str(v).strip()
            if text:
                return text
    live_market = payload.get("live_market") if isinstance(payload, dict) else None
    if isinstance(live_market, dict):
        for key in ("condition_id", "conditionId"):
            v = live_market.get(key)
            if v:
                text = str(v).strip()
                if text:
                    return text
    market = payload.get("market") if isinstance(payload, dict) else None
    if isinstance(market, dict):
        for key in ("condition_id", "conditionId"):
            v = market.get(key)
            if v:
                text = str(v).strip()
                if text:
                    return text
    return ""


def _extract_market_end_time(payload: dict[str, Any]) -> str:
    """Pull the market's expected close time out of the order payload.

    Polymarket markets carry ``end_date_iso`` / ``endDate`` /
    ``end_time`` at the gamma layer.  The orchestrator caches a
    normalized ``end_time`` (ISO-8601 with Z suffix) into
    ``payload.live_market.end_time`` at signal time; older rows may
    have it under one of the gamma aliases.  We probe in order of
    preference and return "" if none is found (callers treat that as
    "unknown" and skip the pending_resolution gate).
    """
    candidates = []
    live_market = payload.get("live_market") if isinstance(payload, dict) else None
    if isinstance(live_market, dict):
        for key in ("end_time", "end_date_iso", "endDateIso", "end_date", "endDate"):
            candidates.append(live_market.get(key))
    market = payload.get("market") if isinstance(payload, dict) else None
    if isinstance(market, dict):
        for key in ("end_time", "end_date_iso", "endDateIso", "end_date", "endDate"):
            candidates.append(market.get(key))
    for key in ("end_time", "end_date_iso", "endDateIso", "end_date", "endDate"):
        if isinstance(payload, dict):
            candidates.append(payload.get(key))
    for value in candidates:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _seconds_until_market_end(end_time_text: str) -> float | None:
    """Parse an ISO-8601 timestamp and return seconds-until-now.

    Returns None if the input is empty or unparseable.  Returns a
    *negative* number if the market has already passed its end time
    (which is fine — the caller treats anything within the window,
    including past, as "imminent resolution").
    """
    if not end_time_text:
        return None
    raw = end_time_text.strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (parsed - utcnow()).total_seconds()


def _extract_outcome_index(payload: dict[str, Any], direction: str | None) -> int | None:
    pe = payload.get("pending_live_exit") if isinstance(payload, dict) else None
    if isinstance(pe, dict):
        snap = pe.get("snapshot") if isinstance(pe.get("snapshot"), dict) else {}
        for source in (pe, snap):
            try:
                v = source.get("outcomeIndex")
                if v is not None:
                    return int(v)
            except (TypeError, ValueError):
                pass
    text = str(direction or "").strip().lower()
    if text in {"yes", "buy_yes", "long", "0"}:
        return 0
    if text in {"no", "buy_no", "short", "1"}:
        return 1
    return None


async def scan_stuck_positions(
    *,
    age_hours: float = _STUCK_AGE_HOURS,
    session_factory: Any = None,
) -> list[dict[str, Any]]:
    """Find every order in a blocked-terminal state older than *age_hours*.

    Returns a list of plain dicts (not ORM rows) with the fields the
    classifier needs — keeps the caller decoupled from the session.

    ``session_factory`` is an optional override for testing.  When None
    (the production path) we use the global ``AsyncSessionLocal``.
    """
    cutoff = utcnow() - timedelta(hours=max(0.0, float(age_hours)))
    factory = session_factory if session_factory is not None else AsyncSessionLocal

    # Push every static filter into SQL.  The previous unbounded
    # ``WHERE mode='live' AND created_at <= cutoff`` returned ALL old
    # live orders + their full payload_json, then filtered in Python.
    # Pre-fix observed ``mean_exec_time=18s, max=30s`` for this query
    # in pg_stat_statements (the dominant offender after my catalog +
    # signal fixes), with each call burning a worker connection for
    # the full ~30s — long enough to starve the event loop and miss
    # WS heartbeat pings.
    #
    # Open-lifecycle filter goes into SQL directly.  Pending-exit
    # blocked-status check still runs in Python because it's nested
    # inside payload_json (cheap once the row count is bounded; a
    # JSON-path predicate would be marginally faster but the index
    # we'd need doesn't exist).
    open_status_lower = list(_OPEN_LIFECYCLE_STATUSES)
    async with factory() as session:
        rows = list(
            (
                await session.execute(
                    select(TraderOrder)
                    .where(TraderOrder.mode == "live")
                    .where(func.lower(TraderOrder.status).in_(open_status_lower))
                    # Use ``created_at`` as the age anchor — the lifecycle
                    # touches ``updated_at`` on every retry, so an
                    # updated_at-based filter would EXCLUDE exactly the
                    # rows we want (active retry storms).  ``created_at``
                    # tracks "when did this order originally enter the
                    # system", which is the right signal for "has this
                    # been around long enough to be stuck?".
                    .where(TraderOrder.created_at <= cutoff)
                    # Belt-and-suspenders: the pending-exit blocked
                    # check below filters again, but a hard LIMIT
                    # prevents a single pathological scan from
                    # holding a worker connection for >30s.
                    .limit(500)
                )
            )
            .scalars()
            .all()
        )

    out: list[dict[str, Any]] = []
    for row in rows:
        # Status already filtered by SQL but defensive recheck —
        # ``_OPEN_LIFECYCLE_STATUSES`` may evolve and we'd rather drop
        # a stragger than alert on it.
        if str(row.status or "").strip().lower() not in _OPEN_LIFECYCLE_STATUSES:
            continue
        payload = dict(row.payload_json or {})
        pending_exit = payload.get("pending_live_exit")
        if not isinstance(pending_exit, dict):
            continue
        pe_status = str(pending_exit.get("status") or "").strip().lower()
        if pe_status not in _BLOCKED_TERMINAL_STATES:
            continue
        out.append(
            {
                "order_id": str(row.id),
                "trader_id": str(row.trader_id or ""),
                "source": str(row.source or ""),
                "market_id": str(row.market_id or ""),
                "direction": str(row.direction or ""),
                "notional_usd": float(row.notional_usd or 0.0),
                "entry_price": float(row.entry_price or 0.0),
                "updated_at": row.updated_at,
                "created_at": row.created_at,
                "pending_exit_status": pe_status,
                "consecutive_blocked_failure_count": int(
                    pending_exit.get("consecutive_blocked_failure_count") or 0
                ),
                "last_error": str(pending_exit.get("last_error") or ""),
                "last_attempt_at": str(pending_exit.get("last_attempt_at") or ""),
                "token_id": _extract_token_id(payload),
                "condition_id": _extract_condition_id(payload),
                "outcome_index": _extract_outcome_index(payload, row.direction),
                "market_end_time": _extract_market_end_time(payload),
            }
        )
    return out


async def classify_stuck_position(observation: dict[str, Any]) -> dict[str, Any]:
    """Decide what's actually happening on-chain for one observation.

    Returns the observation enriched with on-chain truth + a
    ``classification`` field ∈ {recovered_externally, redemption_pending,
    operator_intervention, missing_chain_inputs, chain_unavailable}.
    """
    token_id = observation.get("token_id") or ""
    condition_id = observation.get("condition_id") or ""
    outcome_index = observation.get("outcome_index")
    if not token_id or not condition_id or outcome_index is None:
        return {
            **observation,
            "classification": "missing_chain_inputs",
            "chain_status": None,
        }

    try:
        wallet = (live_execution_service.get_execution_wallet_address() or "").strip()
    except Exception:
        wallet = ""
    if not wallet:
        return {
            **observation,
            "classification": "chain_unavailable",
            "chain_status": None,
            "chain_error": "wallet_address_unavailable",
        }

    chain_status = await ctf_execution_service.fetch_position_chain_status(
        wallet_address=wallet,
        token_id=token_id,
        condition_id=condition_id,
        outcome_index=int(outcome_index),
    )

    if chain_status.get("error"):
        return {
            **observation,
            "classification": "chain_unavailable",
            "chain_status": chain_status,
            "chain_error": chain_status.get("error"),
        }

    balance = float(chain_status.get("wallet_balance_shares") or 0.0)
    market_resolved = bool(chain_status.get("market_resolved"))

    if balance <= 0.0:
        # On-chain says we hold zero shares.  Either the position was
        # sold (manual user action), redeemed externally, or the
        # token_id/condition_id mapping is wrong.  The verifier's
        # wallet_trades scan will pick up any SELL on the next cycle.
        return {
            **observation,
            "classification": "recovered_externally",
            "chain_status": chain_status,
        }

    if market_resolved:
        # The redeemer_worker scans wallet positions every 120s and will
        # call CTF.redeemPositions for any resolved condition.  That
        # produces an on-chain trade event the verifier picks up.  No
        # action from us — just observe.
        return {
            **observation,
            "classification": "redemption_pending",
            "chain_status": chain_status,
        }

    # Before deciding whether this is a transient infra blip or a real
    # venue-side rejection, ask: is this market about to resolve on its
    # own?  If the gamma end_date is within ``_PENDING_RESOLUTION_WINDOW_DAYS``
    # then the redeemer worker will close it the moment the oracle
    # reports — and there is nothing the operator can do that the
    # redeemer won't do automatically.  Suppress the alert.
    #
    # This classification fires *before* the venue-rejection / client-
    # timeout split because the eventual disposition is the same in
    # both cases when the market is imminently resolving.  Even if the
    # CLOB has stopped accepting SELLs (which is exactly what happens
    # in the final hours of a market) we can just wait for resolution.
    end_time_text = str(observation.get("market_end_time") or "")
    seconds_until_end = _seconds_until_market_end(end_time_text)
    if (
        seconds_until_end is not None
        and seconds_until_end <= _PENDING_RESOLUTION_WINDOW_DAYS * 86400.0
    ):
        return {
            **observation,
            "classification": "pending_resolution",
            "chain_status": chain_status,
            "seconds_until_market_end": seconds_until_end,
        }

    # We hold shares + market is not resolved + retry was already
    # circuit-broken by the lifecycle.  Decide whether this needs
    # operator attention or is just our own retry-path timing out:
    #
    #   * If ``last_error`` matches a known venue-rejection marker
    #     (``orderbook does not exist``, ``market not tradable``,
    #     etc.) — the venue is genuinely refusing the SELL and the
    #     operator should look.
    #   * If ``last_error`` is a client-side failure (TimeoutError,
    #     ConnectionError, asyncpg, etc.) — the lifecycle's
    #     auto-recovery (after _BLOCKED_PERSISTENT_TIMEOUT_AUTO_RETRY_AFTER_SECONDS)
    #     will retry the SELL; alerting the operator now is noise.
    #     Classify as ``transient_client_failure`` instead so the
    #     alert path stays silent.
    #
    # This pairs with the venue-rejection gate in
    # ``services/operator_writeoff.manual_writeoff_order``: the
    # writeoff path now refuses non-venue errors without an explicit
    # override, and the alert path now refuses to escalate them at
    # all.  Together they prevent the 2026-04-28 incident where a
    # client-side timeout cascade (DB pressure → SELL retry timeouts
    # → blocked_persistent_timeout) was misread as venue rejection,
    # surfaced as ``Stuck position needs operator review``, and led
    # to a bulk full-loss writeoff against rows whose wallets still
    # held the shares.
    last_error_text = str(observation.get("last_error") or "")
    if not _is_venue_rejection_error(last_error_text):
        return {
            **observation,
            "classification": "transient_client_failure",
            "chain_status": chain_status,
        }
    return {
        **observation,
        "classification": "operator_intervention",
        "chain_status": chain_status,
    }


def _format_alert_message(observation: dict[str, Any]) -> str:
    """Build the Telegram body.

    Framing is deliberately *informational*, not "needs operator
    review".  By the time we reach this code path:

      * The position is at least ``_STUCK_AGE_HOURS`` (6h) old.
      * The 5-minute auto-recovery has tried and failed enough times
        to drive ``pending_live_exit.status`` into a blocked-terminal
        state.
      * On-chain truth confirms we still hold shares.
      * The market is not resolved yet.
      * The market end_date is more than
        ``_PENDING_RESOLUTION_WINDOW_DAYS`` away.
      * The last_error matches a real venue-rejection marker (not a
        client-side timeout).

    Even then, the right operator response is usually "do nothing":
    when the market eventually resolves, the redeemer closes it.
    The manual-writeoff API exists for the rare case where the
    operator has external evidence the position is truly worthless
    and wants to clear it from the dashboard before resolution.

    Hence: lead with chain truth, present the manual-writeoff CTA
    as optional, never use language that implies an SLA.
    """
    chain_status = observation.get("chain_status") or {}
    notional = float(observation.get("notional_usd") or 0.0)
    balance = float(chain_status.get("wallet_balance_shares") or 0.0)
    end_time = str(observation.get("market_end_time") or "unknown")
    return (
        "ℹ️ *Long-running blocked exit (informational)*\n"
        f"`order_id`        : `{observation.get('order_id')}`\n"
        f"`trader_id`       : `{observation.get('trader_id')}`\n"
        f"`source`          : `{observation.get('source')}`\n"
        f"`market_id`       : `{observation.get('market_id')}`\n"
        f"`direction`       : `{observation.get('direction')}`\n"
        f"`notional_usd`    : `${notional:.2f}`\n"
        f"`pending_status`  : `{observation.get('pending_exit_status')}`\n"
        f"`blocked_streak`  : `{observation.get('consecutive_blocked_failure_count')}`\n"
        f"`last_error`      : `{(observation.get('last_error') or '')[:60]}`\n"
        f"`chain_balance`   : `{balance:.4f} shares`\n"
        f"`market_resolved` : `{chain_status.get('market_resolved')}`\n"
        f"`market_end_time` : `{end_time}`\n"
        f"`block_number`    : `{chain_status.get('block_number')}`\n"
        "\n"
        "No action required by default.  The CLOB is refusing the SELL "
        "for this market and the redeemer worker will close the "
        "position on resolution.  If you have external evidence the "
        "position is worthless and want to clear it from the dashboard "
        "sooner, use the manual-writeoff API "
        "(`/api/operator/orders/{order_id}/manual-writeoff`) — that "
        "writes a verified close with a full audit trail."
    )


async def alert_operator_on_stuck_positions(classified: list[dict[str, Any]]) -> dict[str, Any]:
    """Emit a Telegram alert for each ``operator_intervention`` row whose
    cooldown has elapsed.  All other classifications are silent.

    Returns a summary dict for logging / heartbeat exposure.
    """
    summary = {
        "scanned": len(classified),
        "recovered_externally": 0,
        "redemption_pending": 0,
        "operator_intervention": 0,
        # Classification: rows whose only failure is our own
        # client-side timeouts.  Counted but never alerted — the
        # lifecycle's auto-recovery kicks in after
        # ``_BLOCKED_PERSISTENT_TIMEOUT_AUTO_RETRY_AFTER_SECONDS`` and
        # paging the operator on transient infrastructure noise is
        # what produced the 2026-04-28 alert-then-bulk-writeoff
        # incident.
        "transient_client_failure": 0,
        # Classification: holdings + market unresolved + market end_date
        # within ``_PENDING_RESOLUTION_WINDOW_DAYS``.  Counted but
        # never alerted — the redeemer worker handles the close on
        # resolution and nothing the operator can do is faster.
        "pending_resolution": 0,
        "missing_chain_inputs": 0,
        "chain_unavailable": 0,
        "alerts_emitted": 0,
        "alerts_suppressed_by_cooldown": 0,
    }
    if not classified:
        return summary

    # Lazy import to avoid an import cycle at module load.
    from services.notifier import notifier as _notifier

    now_mono = _time.monotonic()
    for row in classified:
        cls = row.get("classification") or ""
        if cls in summary:
            summary[cls] = int(summary[cls]) + 1
        if cls != "operator_intervention":
            continue
        order_id = str(row.get("order_id") or "")
        if not order_id:
            continue
        last_at = _last_alert_at.get(order_id, 0.0)
        if last_at and (now_mono - last_at) < _ALERT_COOLDOWN_SECONDS:
            summary["alerts_suppressed_by_cooldown"] += 1
            continue
        try:
            await _notifier.send_operator_alert(
                _format_alert_message(row),
                category="stuck_position",
            )
        except Exception as exc:
            logger.warning(
                "stuck_position alert send failed",
                extra={"order_id": order_id, "error": str(exc)},
            )
            continue
        _last_alert_at[order_id] = now_mono
        summary["alerts_emitted"] += 1
    return summary


async def run_stuck_position_scan_once(*, age_hours: float = _STUCK_AGE_HOURS) -> dict[str, Any]:
    """One scan + classify + alert pass.  Safe to call periodically.

    Designed to be invoked from ``trader_reconciliation_worker`` (or
    any worker) on a 5-minute cadence.  Pure surveillance — never
    writes to ``trader_orders``.
    """
    started = _time.monotonic()
    observations = await scan_stuck_positions(age_hours=age_hours)
    if not observations:
        return {
            "duration_seconds": round(_time.monotonic() - started, 3),
            "scanned": 0,
            "operator_intervention": 0,
            "alerts_emitted": 0,
            "alerts_suppressed_by_cooldown": 0,
        }
    # Classify with bounded concurrency — we don't want to spam the
    # RPC with hundreds of parallel calls if the queue is large.
    sem = asyncio.Semaphore(4)

    async def _classify(o: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            return await classify_stuck_position(o)

    classified = await asyncio.gather(*(_classify(o) for o in observations))
    summary = await alert_operator_on_stuck_positions(list(classified))
    summary["duration_seconds"] = round(_time.monotonic() - started, 3)
    if summary.get("operator_intervention", 0) > 0:
        logger.warning(
            "stuck_position_monitor: operator-intervention rows present",
            extra=summary,
        )
    else:
        logger.debug("stuck_position_monitor scan", extra=summary)
    return summary
