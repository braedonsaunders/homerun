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
import time
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
# Per-order fire cooldown. The pending_live_exit dict is the primary mutex,
# but reconcile can legitimately pop/supersede it (wallet-flat / fallback-
# manual / terminalization paths) while a position is still selectable for a
# few cycles — e.g. an already-settled phantom on an inactive market. Without
# a cooldown the loop re-submits a (doomed) exit every sweep during that
# window. This in-memory cooldown caps re-submission to once per window PER
# ORDER, with no dependency on the (boot-cold-unsafe) wallet cache, so it can
# never disable stops. Long enough to let reconcile's retry/terminalization
# act; short enough to re-attempt a genuinely-stuck real exit.
_FIRE_COOLDOWN_SECONDS = 30.0


class ExitRiskLoop:
    """Singleton fast exit-risk evaluator/executor for the trading worker."""

    def __init__(self) -> None:
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._wake = asyncio.Event()
        self._tick_tokens: set[str] = set()      # held tokens that moved, pending eval
        self._held_tokens: set[str] = set()       # current held-token index (tick filter)
        self._callback_registered = False
        self._last_fire_at: dict[str, float] = {}  # order_id -> monotonic ts of last exit submit

    # ── WS price-change callback (sub-second tick path) ────────────────
    def on_price_change(self, token_id: str, old_mid: float, new_mid: float) -> None:
        """Fires from the WS feed on a material mid move. Sync + cheap."""
        if token_id in self._held_tokens:
            self._tick_tokens.add(token_id)
            if not self._wake.is_set():
                self._wake.set()

    async def run_forever(self) -> None:
        """Long-running entry point. The worker host owns the task lifecycle
        (``workers/exit_risk_worker.py`` calls this from ``start_loop``)."""
        if self._running:
            return
        self._running = True
        self._register_callback()
        logger.info("Exit-risk loop started (sweep=%.1fs)", _SWEEP_INTERVAL_SECONDS)
        while self._running:
            # Re-attempt callback registration until the feed manager exists.
            if not self._callback_registered:
                self._register_callback()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=_SWEEP_INTERVAL_SECONDS)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                self._running = False
                raise
            self._wake.clear()
            self._tick_tokens.clear()  # the sweep covers everything; ticks just wake us
            try:
                await self._sweep()
            except asyncio.CancelledError:
                self._running = False
                raise
            except Exception as exc:
                logger.warning("Exit-risk sweep failed: %s", exc, exc_info=exc)

    def stop(self) -> None:
        self._running = False
        self._wake.set()

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
                self._last_fire_at.clear()
                return

            # Prune the fire-cooldown map to currently-open orders so it can't
            # grow unbounded across a long soak (closed orders drop out).
            open_ids = {str(r.id) for r in rows}
            if self._last_fire_at:
                self._last_fire_at = {
                    oid: ts for oid, ts in self._last_fire_at.items() if oid in open_ids
                }

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

        token_id = pl._extract_live_token_id(payload)
        filled_notional, filled_size, avg_px = pl._extract_live_fill_metrics(payload)
        entry_price = safe_float(avg_px) or safe_float(row.effective_price) or safe_float(row.entry_price) or 0.0
        if entry_price <= 0.0:
            return
        notional = filled_notional if filled_notional > 0.0 else (safe_float(row.notional_usd) or 0.0)
        if notional <= 0.0:
            return
        entry_size = filled_size if filled_size > 0.0 else (notional / entry_price)

        market_info = market_info_by_id.get(str(row.market_id or ""))
        wallet_pos = wallet_by_token.get(token_id) if token_id else None
        wallet_size = pl._extract_wallet_position_size(wallet_pos)

        # ── Ghost-position terminalization (fast-path) ────────────────────
        # The fast loop sells in seconds, but the orchestrator row is closed
        # by the SLOW reconcile loop (starved: 9-16s cycles in the soak). So a
        # fully-liquidated position can stay ``executed`` for 20+ min, and the
        # loop keeps firing doomed sells against a now-empty wallet. When the
        # wallet shows flat AND our own exit fills (live_trading_orders, the
        # on-chain fill source of truth) account for the whole entry size, we
        # terminalize HERE via the shared ``terminalize_filled_exit`` so the
        # close record is identical to reconcile's (single source of truth).
        # Gated on wallet-flat so it's not a per-sweep query for healthy
        # positions, and never closes a position still holding shares.
        if token_id and entry_size > 0.0 and wallet_size <= pl._WALLET_SIZE_EPSILON:
            sold_size, sold_proceeds = await pl.summarize_live_exit_fills(
                session, token_id=token_id, since=(row.executed_at or row.created_at),
            )
            if sold_proceeds > 0.0 and sold_size >= entry_size - max(0.5, entry_size * 0.02):
                vwap = sold_proceeds / sold_size if sold_size > 0 else 0.0
                realized = sold_proceeds - notional
                pe0 = payload.get("pending_live_exit")
                trig = (pe0.get("close_trigger") if isinstance(pe0, dict) else None) or "exit_fill_confirmed"
                exit_instance = (
                    await pl._strategy_exit_instance(session, (payload.get("strategy_type") or "").strip().lower())
                    if payload.get("strategy_type") else None
                )
                ns = await pl.terminalize_filled_exit(
                    session=session, row=row, payload=payload, now=now,
                    close_price=vwap, realized_pnl=realized, filled_size=sold_size,
                    close_trigger=str(trig), price_source="live_exit_fill_vwap",
                    reason="exit_risk_loop", exit_instance=exit_instance, market_info=market_info,
                )
                await session.commit()
                logger.warning(
                    "exit_risk_loop TERMINALIZED order=%s -> %s sold=%.2f/%.2f vwap=%.4f realized_pnl=%.2f",
                    row.id, ns, sold_size, entry_size, vwap, realized,
                )
                return

        pe = payload.get("pending_live_exit")
        # Mutex / single-owner handoff: the loop fires only the FIRST exit for
        # a position. Once ANY ``pending_live_exit`` exists (in-flight OR a
        # failed/blocked retry state), the full retry lifecycle — backoff via
        # ``next_retry_at``, escalation to ``blocked_retry_exhausted[_hard]``,
        # the soft-bypass — is owned by ``reconcile_live_positions`` (a single
        # stateful path, no divergence). Re-evaluating here would ignore that
        # backoff and hammer a resubmit every ~2s (observed on a phantom
        # zero-share position: market_inactive re-fired every sweep). So skip
        # whenever a pending_live_exit dict is present, not just the narrow
        # in-flight subset.
        if isinstance(pe, dict) and str(pe.get("status") or "").strip():
            return  # an exit attempt exists — its lifecycle belongs to reconcile

        outcome_idx = pl._direction_outcome_index(row.direction, market_info=market_info, token_id=token_id)
        if outcome_idx is None:
            return
        market_tradable = polymarket_client.is_market_tradable(market_info, now=now)
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
            # Per-order cooldown: don't re-submit an exit for the same order
            # within the window even if the pending_live_exit mutex was cleared
            # by reconcile (wallet-flat / terminalization). Persist anchors and
            # bail. Resolution-class closes still flow through reconcile.
            order_id = str(row.id)
            last_fire = self._last_fire_at.get(order_id)
            now_mono = time.monotonic()
            if last_fire is not None and (now_mono - last_fire) < _FIRE_COOLDOWN_SECONDS:
                row.payload_json = payload
                row.updated_at = now
                await session.commit()
                return None
            self._last_fire_at[order_id] = now_mono
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
