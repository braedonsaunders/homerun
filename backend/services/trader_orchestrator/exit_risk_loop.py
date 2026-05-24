"""Isolated fast exit-risk loop — institutional stop-loss isolation.

The three full-notional losses across the 2026-05 soaks all shared one
mechanism: stop EVALUATION was buried inside the heavyweight
``reconcile_live_positions`` (wallet/provider/projection reconciliation,
20-26s, timing out at 30s — worst for the very trader taking the losses).
So during multi-hour gradual price declines the positions were never
re-marked and stops never fired; they rode to resolution at ~0.

This loop fixes that by making stop evaluation a first-class, isolated
control loop that CANNOT be starved by bulk reconciliation:

  * Runs in the trading worker process (shares the in-process WS price
    cache — fresh marks, no extra subscription) on a DEDICATED fast DB
    pool (``FastAsyncSessionLocal``, 2.5s stmt / 500ms lock timeout), so
    it never queues behind the saturated worker pool.
  * Periodic sweep (~2s) over EVERY open live position, off the no-REST
    hot path, so the CLOB-REST mark fallback fires for stale-WS tokens
    (it never could inside the orchestrator's ``hot_path_no_rest`` scope).
  * WS ``on_change`` tick: when a HELD token's mark moves materially, that
    position is evaluated immediately (sub-second) — exit latency parity
    with the fast entry lane.

Decision + submit reuse the SHARED ``evaluate_position_exit`` /
``execute_position_exit`` / ``_submit_live_partial_exit`` so the logic is
identical to reconcile (single source of truth, no divergence).

Concurrency: a position with an in-flight ``pending_live_exit`` is skipped
(mutex shared with reconcile); the fast pool's 500ms lock timeout makes
row contention fail-fast (skip this cycle, retry in ~2s).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select

from models.database import FastAsyncSessionLocal, Trader, TraderOrder
from services.polymarket import polymarket_client
from services.trader_orchestrator import position_lifecycle as pl
from utils.converters import safe_float
from utils.utcnow import utcnow

logger = logging.getLogger("exit_risk_loop")

_SWEEP_INTERVAL_SECONDS = 2.0
# pending_live_exit states that mean "an exit is already being handled" —
# skip so we never double-submit (shared mutex with reconcile).
_EXIT_IN_FLIGHT_STATES = {
    "pending", "submitted", "filled",
    "blocked_min_notional", "blocked_retry_exhausted",
    "blocked_retry_exhausted_hard", "superseded_resolution",
}


class ExitRiskLoop:
    """Singleton fast exit-risk evaluator/executor for the trading worker."""

    def __init__(self) -> None:
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._wake = asyncio.Event()
        self._tick_tokens: set[str] = set()      # held tokens that moved, pending eval
        self._held_tokens: set[str] = set()       # current held-token index (tick filter)
        self._callback_registered = False

    # ── WS price-change callback (sub-second tick path) ────────────────
    def on_price_change(self, token_id: str, old_mid: float, new_mid: float) -> None:
        """Fires from the WS feed on a material mid move. Sync + cheap."""
        if token_id in self._held_tokens:
            self._tick_tokens.add(token_id)
            if not self._wake.is_set():
                self._wake.set()

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._register_callback()
        self._task = asyncio.create_task(self._run(), name="exit-risk-loop")
        self._task.add_done_callback(lambda _t: None)
        logger.info("Exit-risk loop started (sweep=%.1fs)", _SWEEP_INTERVAL_SECONDS)

    async def stop(self) -> None:
        self._running = False
        self._wake.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    def _register_callback(self) -> None:
        if self._callback_registered:
            return
        try:
            from services.ws_feeds import get_feed_manager

            fm = get_feed_manager()
            cache = getattr(fm, "cache", None)
            if cache is not None and hasattr(cache, "add_on_change_callback"):
                cache.add_on_change_callback(self.on_price_change)
                self._callback_registered = True
        except Exception as exc:
            logger.debug("Exit-risk loop: WS callback registration deferred: %s", exc)

    async def _run(self) -> None:
        while self._running:
            # Re-attempt callback registration until the feed manager exists.
            if not self._callback_registered:
                self._register_callback()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=_SWEEP_INTERVAL_SECONDS)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                raise
            self._wake.clear()
            self._tick_tokens.clear()  # the sweep covers everything; ticks just wake us
            try:
                await self._sweep()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Exit-risk sweep failed: %s", exc, exc_info=exc)

    async def _sweep(self) -> None:
        """Evaluate (and exit if triggered) every open live position."""
        now = utcnow()
        now_naive = now.astimezone(timezone.utc).replace(tzinfo=None) if now.tzinfo else now

        async with FastAsyncSessionLocal() as session:
            rows = list(
                (
                    await session.execute(
                        select(TraderOrder).where(
                            TraderOrder.mode == "live",
                            TraderOrder.status.in_(tuple(pl.LIVE_ACTIVE_STATUSES)),
                        )
                    )
                )
                .scalars()
                .all()
            )
            if not rows:
                self._held_tokens = set()
                return

            # Refresh held-token index for the tick path.
            token_ids: list[str] = []
            held: set[str] = set()
            for row in rows:
                tid = pl._extract_live_token_id(dict(row.payload_json or {}))
                if tid:
                    token_ids.append(tid)
                    held.add(tid)
            self._held_tokens = held

            # Fresh marks: WS book (strict→relaxed) + CLOB-REST fallback for
            # stale tokens (allow_rest=True — this loop is NOT hot_path_no_rest).
            ws_mid, clob_mid, books = await pl._collect_live_exit_marks(
                token_ids=sorted(set(token_ids)), session=session, allow_rest=True
            )
            market_info_by_id = await pl.load_market_info_for_orders(rows)
            wallet_by_token = dict(pl._wallet_positions_cache[1] or {})

            # Per-trader params (risk_limits) — small set; one query.
            trader_ids = {str(r.trader_id) for r in rows if r.trader_id}
            params_by_trader: dict[str, dict[str, Any]] = {}
            if trader_ids:
                traders = (
                    (await session.execute(select(Trader).where(Trader.id.in_(tuple(trader_ids)))))
                    .scalars()
                    .all()
                )
                for t in traders:
                    rl = getattr(t, "risk_limits_json", None)
                    params_by_trader[str(t.id)] = dict(rl) if isinstance(rl, dict) else {}

            submissions = [0]  # per-sweep submission budget (shared with execute_position_exit)
            for row in rows:
                try:
                    await self._process_position(
                        session=session,
                        row=row,
                        now=now,
                        now_naive=now_naive,
                        ws_mid=ws_mid,
                        clob_mid=clob_mid,
                        books=books,
                        market_info_by_id=market_info_by_id,
                        wallet_by_token=wallet_by_token,
                        params=params_by_trader.get(str(row.trader_id), {}),
                        submissions=submissions,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("Exit-risk: position %s failed: %s", getattr(row, "id", "?"), exc, exc_info=exc)

    async def _process_position(
        self,
        *,
        session,
        row: TraderOrder,
        now: datetime,
        now_naive: datetime,
        ws_mid: dict[str, float],
        clob_mid: dict[str, float],
        books: dict[str, Any],
        market_info_by_id: dict[str, Any],
        wallet_by_token: dict[str, Any],
        params: dict[str, Any],
        submissions: list[int],
    ) -> None:
        payload = dict(row.payload_json or {})
        pe = payload.get("pending_live_exit")
        if isinstance(pe, dict) and str(pe.get("status") or "").strip().lower() in _EXIT_IN_FLIGHT_STATES:
            return  # exit already in flight — mutex with reconcile / prior cycle

        token_id = pl._extract_live_token_id(payload)
        filled_notional, filled_size, avg_px = pl._extract_live_fill_metrics(payload)
        entry_price = safe_float(avg_px) or safe_float(row.effective_price) or safe_float(row.entry_price) or 0.0
        if entry_price <= 0.0:
            return
        notional = filled_notional if filled_notional > 0.0 else (safe_float(row.notional_usd) or 0.0)
        if notional <= 0.0:
            return

        market_info = market_info_by_id.get(str(row.market_id or ""))
        outcome_idx = pl._direction_outcome_index(row.direction, market_info=market_info, token_id=token_id)
        if outcome_idx is None:
            return
        market_tradable = polymarket_client.is_market_tradable(market_info, now=now)
        wallet_pos = wallet_by_token.get(token_id) if token_id else None
        wallet_size = pl._extract_wallet_position_size(wallet_pos)
        wallet_mark = pl._extract_wallet_mark_price(wallet_pos)
        market_side_price = pl._extract_market_side_price(market_info, outcome_idx)

        min_hold_minutes = max(
            0.0, safe_float(pl._payload_exit_param(payload, prefix_key="live", name="min_hold_minutes")) or 0.0
        )
        decision = await pl.evaluate_position_exit(
            session=session,
            row=row,
            payload=payload,
            now=now,
            now_naive=now_naive,
            ws_side_price=ws_mid.get(token_id) if token_id else None,
            clob_side_price=clob_mid.get(token_id) if token_id else None,
            market_side_price=market_side_price,
            wallet_mark_price=wallet_mark,
            book=books.get(token_id) if token_id else None,
            entry_price=entry_price,
            notional=notional,
            filled_size=filled_size,
            wallet_position_size=wallet_size,
            outcome_idx=outcome_idx,
            market_info=market_info,
            market_tradable=market_tradable,
            market_seconds_left=pl._market_seconds_left(market_info, now),
            market_end_time=pl._market_end_time_iso(market_info),
            take_profit_pct=safe_float(pl._payload_exit_param(payload, prefix_key="live", name="take_profit_pct")),
            stop_loss_pct=safe_float(pl._payload_exit_param(payload, prefix_key="live", name="stop_loss_pct")),
            trailing_stop_pct=safe_float(pl._payload_exit_param(payload, prefix_key="live", name="trailing_stop_pct")),
            max_hold_minutes=safe_float(pl._payload_exit_param(payload, prefix_key="live", name="max_hold_minutes")),
            min_hold_minutes=min_hold_minutes,
            resolve_only=pl._safe_bool(pl._payload_exit_param(payload, prefix_key="live", name="resolve_only"), False),
            close_on_inactive_market=pl._safe_bool(
                pl._payload_exit_param(payload, prefix_key="live", name="close_on_inactive_market"), True
            ),
            pending_exit=pe,
            params=params,
            mark_touch_interval_seconds=10.0,
        )

        payload["position_state"] = decision.next_state
        if decision.action == "close" and decision.close_price is not None:
            exit_instance = (
                await pl._strategy_exit_instance(session, (payload.get("strategy_type") or "").strip().lower())
                if payload.get("strategy_type")
                else None
            )
            quantity = filled_size if filled_size > 0.0 else (notional / entry_price if entry_price > 0 else 0.0)
            await pl.execute_position_exit(
                session=session,
                row=row,
                payload=payload,
                now=now,
                close_price=decision.close_price,
                close_trigger=decision.close_trigger,
                price_source=decision.price_source,
                pnl=None,
                market_tradable=market_tradable,
                age_minutes=decision.age_minutes,
                filled_size=filled_size,
                quantity=quantity,
                wallet_position_size=wallet_size,
                pending_exit=pe,
                exit_instance=exit_instance,
                strategy_exit=decision.strategy_exit,
                params=params,
                reason="exit_risk_loop",
                submissions_this_pass=submissions,
            )
            logger.warning(
                "exit_risk_loop FIRED %s order=%s @%.4f (mark_src=%s pnl_pct=%s)",
                decision.close_trigger, row.id, decision.close_price, decision.price_source,
                None if decision.pnl_pct is None else round(decision.pnl_pct, 1),
            )
        elif decision.action == "reduce" and decision.strategy_exit is not None:
            await pl._submit_live_partial_exit(
                session=session,
                row=row,
                payload=payload,
                decision=decision.strategy_exit,
                close_price=decision.close_price,
                price_source=decision.price_source,
                filled_size=filled_size,
                notional_usd=notional,
                entry_price=entry_price,
                params=params,
                now=now,
            )
            logger.info("exit_risk_loop reduce order=%s @%s", row.id, decision.close_price)

        # Persist anchors (and any pending_live_exit the submit set).
        row.payload_json = payload
        row.updated_at = now
        await session.commit()
        return None


_exit_risk_loop: Optional[ExitRiskLoop] = None


def get_exit_risk_loop() -> ExitRiskLoop:
    global _exit_risk_loop
    if _exit_risk_loop is None:
        _exit_risk_loop = ExitRiskLoop()
    return _exit_risk_loop
