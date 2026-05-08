"""Polymarket-truth verification for TraderOrder realized P&L.

The ONLY legitimate sources of realized P&L are:

  1. A confirmed on-chain trade record from polymarket.get_wallet_trades —
     this includes both bot-initiated SELLs and any manual user sells
     made via the Polymarket UI. The trade record carries the actual
     fill price and a transactionHash that pins the close to a specific
     on-chain event.

  2. A market-resolution payout: when a market settles, winning shares
     pay $1 and losing shares pay $0. This is deterministic and can be
     computed from the market's resolution metadata + our recorded entry
     fill size.

NO other source — wallet aggregate `realizedPnl`, `currentPrice`, etc.
— is acceptable. Inferred P&L conflates manual user trades with bot
fills and produces phantom numbers.

This module:

  * fetches the wallet's recent trades from Polymarket
  * matches sells to TraderOrder rows by (asset_id == token_id) +
    (trade.timestamp > order.entry_anchor) + side == SELL, FIFO by
    timestamp so the earliest sell consumes the earliest open order
  * computes realized_pnl = sum(sell_price * matched_size) -
    matched_cost_basis
  * writes the verified P&L back to the row with
    verification_status = wallet_activity (matched-by-trade) and the
    transactionHash recorded as verification_tx_hash

Orders that cannot be matched to any sell trade AND whose market has
not resolved are left at verification_status = summary_only with
actual_profit = NULL — the aggregation queries already exclude these.

For each row updated, an immutable TraderOrderVerificationEvent is
appended so the lineage is auditable.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import TraderOrder, release_conn
from services.polymarket import polymarket_client
from services.trader_order_verification import (
    TRADER_ORDER_VERIFICATION_WALLET_ACTIVITY,
    apply_trader_order_verification,
    append_trader_order_verification_event,
)
from utils.converters import safe_float
from utils.utcnow import utcnow

from utils.logger import get_logger

logger = get_logger("polymarket_trade_verifier")

# A trade record from /data/trades:
#   {
#     "proxyWallet": "0x...",      # wallet address (lowercase)
#     "side": "BUY" | "SELL",
#     "asset": "<erc1155 token id>",
#     "conditionId": "0x...",
#     "size": <float>,
#     "price": <float>,             # actual fill price 0-1
#     "timestamp": <unix seconds>,  # int
#     "transactionHash": "0x...",   # on-chain tx
#     "outcomeIndex": 0 | 1,
#     ...
#   }


_RESOLVED_STATUSES = (
    "resolved",
    "resolved_win",
    "resolved_loss",
    "closed_win",
    "closed_loss",
    "win",
    "loss",
)


def _reconcile_status_with_pnl(row: TraderOrder, verified_pnl: float) -> None:
    # The verifier may overwrite ``actual_profit`` with on-chain truth long
    # after the orchestrator decided the win/loss label from a different
    # (often coarser) close-price estimate.  When the verifier's number
    # disagrees in sign with the existing label, flip the label so the UI
    # and aggregation queries don't show "resolved_win" with negative pnl.
    current_status = str(row.status or "").strip().lower()
    if current_status not in {"closed_win", "closed_loss", "resolved_win", "resolved_loss"}:
        return
    is_resolution = current_status.startswith("resolved")
    if verified_pnl > 0:
        target = "resolved_win" if is_resolution else "closed_win"
    elif verified_pnl < 0:
        target = "resolved_loss" if is_resolution else "closed_loss"
    else:
        return
    if target != current_status:
        row.status = target

# Cap HTTP-fetch portion of the verifier sweeps separately from the
# overall session timeout (30s).  Without this, a slow Polymarket
# pagination consumes the entire budget and the verifier holds its
# session checked out for ~30s while doing zero DB work — leaving the
# pool starved for live-path traffic.
#
# Bumped from 12s to 25s in Fix KK: the prior budget was failing under
# normal Polymarket latency (worker logs showed
# ``closed_positions fetch exceeded 12s HTTP budget; skipping cycle``
# repeatedly), so resolved positions never got their on-chain
# verification and the lifecycle baseline was the only available
# value.  25s leaves headroom inside the session's 30s statement
# timeout while letting the typical 8-15s pagination complete.  The
# Phase 2 release_conn pattern means we're not holding a DB
# connection during the HTTP wait, so the longer budget does not
# starve the live-path pool.
_HTTP_FETCH_TIMEOUT_SECONDS = 25.0


def _direction_to_outcome_index(direction: str | None) -> int | None:
    """Map our `direction` field to a binary outcome index (0/1).

    Polymarket binary markets index outcomes as 0=Yes (or first listed)
    and 1=No (or second listed). We canonicalize "yes"/"buy_yes"/etc.
    """
    if not direction:
        return None
    text = str(direction).strip().lower()
    if text in {"yes", "buy_yes", "long", "0"}:
        return 0
    if text in {"no", "buy_no", "short", "1"}:
        return 1
    return None


def _market_winning_outcome_index(market_info: dict[str, Any]) -> int | None:
    """Extract the winning outcome index from market metadata."""
    if not isinstance(market_info, dict):
        return None
    raw = (
        market_info.get("winning_outcome")
        if market_info.get("winning_outcome") is not None
        else market_info.get("winningOutcome")
    )
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    if text in {"yes", "0", "true"}:
        return 0
    if text in {"no", "1", "false"}:
        return 1
    # Some payloads encode the winning index directly
    try:
        return int(text)
    except ValueError:
        return None


def _entry_condition_id(row: TraderOrder) -> str:
    """Best-effort condition_id (Polymarket 0x... hash) from the order payload."""
    payload = row.payload_json if isinstance(row.payload_json, dict) else {}
    for key in ("condition_id", "conditionId"):
        value = payload.get(key)
        if value:
            text = str(value).strip()
            if text:
                return text
    live_market = payload.get("live_market") if isinstance(payload, dict) else None
    if isinstance(live_market, dict):
        for key in ("condition_id", "conditionId"):
            value = live_market.get(key)
            if value:
                text = str(value).strip()
                if text:
                    return text
    market = payload.get("market") if isinstance(payload, dict) else None
    if isinstance(market, dict):
        for key in ("condition_id", "conditionId"):
            value = market.get(key)
            if value:
                text = str(value).strip()
                if text:
                    return text
    return ""


async def _fetch_market_info(row: TraderOrder) -> dict[str, Any] | None:
    """Resolve the row to Polymarket market metadata.

    TraderOrder.market_id is Polymarket's internal numeric id, NOT the
    condition_id. The polymarket client only exposes
    get_market_by_condition_id and get_market_by_token_id; both need
    different identifiers. Pull condition_id from payload if present;
    otherwise fall back to looking up by entry token_id.
    """
    cond_id = _entry_condition_id(row)
    if cond_id:
        try:
            info = await polymarket_client.get_market_by_condition_id(cond_id)
            if isinstance(info, dict):
                return info
        except Exception:
            pass
    token_id = _entry_token_id(row)
    if token_id:
        try:
            info = await polymarket_client.get_market_by_token_id(token_id)
            if isinstance(info, dict):
                return info
        except Exception:
            pass
    return None


_MARKET_INFO_PREFETCH_CONCURRENCY = 4
_CLOSED_POSITIONS_VERIFY_MAX_FETCH = 300
# Cap the number of rows fed into ``_prefetch_market_info`` per cycle.
# Each row costs 1-2 HTTP round-trips to Polymarket's gamma API; a
# backlog of 400 rows × 500 ms = 50 s of wall-clock work, blowing the
# 30 s verifier budget observed in the 2026-05-08 soak. With a 100-row
# cap, at most one in-flight prefetch takes ~15 s even under the
# slowest observed API latency, leaving margin for phase-3 writes.
# The remaining rows are picked up by the next verifier tick
# (interval = 300 s) so we still make progress without timing out.
_BOT_LINEAGE_PREFETCH_MAX_ROWS = 100
# Per-market HTTP lookup budget. A single pathological
# ``get_market_by_condition_id`` call that hangs for 10 s would
# otherwise stall the whole prefetch fan-out behind its worker. The
# worker pool has only 4 slots, so one slow call dominates 25% of
# total throughput.
_MARKET_INFO_FETCH_TIMEOUT_SECONDS = 8.0


async def _prefetch_market_info(
    rows: Iterable[TraderOrder],
) -> dict[str, dict[str, Any] | None]:
    """Concurrently look up market_info for *rows*.

    Returns a dict keyed by ``str(row.id)``.  Failures are recorded as
    ``None``.  This MUST be called WITHOUT an open DB transaction (or
    inside ``async with release_conn(session):``) — the whole point of
    pre-fetching is to keep PG sessions out of the HTTP critical path.
    """
    rows_list = [r for r in rows if r is not None]
    if not rows_list:
        return {}

    queue: asyncio.Queue[TraderOrder] = asyncio.Queue()
    for row in rows_list:
        queue.put_nowait(row)
    results: dict[str, dict[str, Any] | None] = {}

    async def _worker() -> None:
        while True:
            try:
                row = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                # Per-fetch wall-clock bound. Without this, a single
                # pathological ``get_market_by_*`` call that hangs for
                # 10+ s would stall the fan-out behind the worker.
                results[str(row.id)] = await asyncio.wait_for(
                    _fetch_market_info(row),
                    timeout=_MARKET_INFO_FETCH_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                results[str(row.id)] = None
            except Exception:
                results[str(row.id)] = None
            finally:
                queue.task_done()
                await asyncio.sleep(0)

    workers = [
        asyncio.create_task(_worker(), name=f"trade-verifier-market-info-{idx}")
        for idx in range(min(_MARKET_INFO_PREFETCH_CONCURRENCY, len(rows_list)))
    ]
    try:
        await asyncio.gather(*workers, return_exceptions=True)
    finally:
        for worker in workers:
            if not worker.done():
                worker.cancel()
    return results


def _market_is_resolved(market_info: dict[str, Any]) -> bool:
    if not isinstance(market_info, dict):
        return False
    if market_info.get("resolved") is True or market_info.get("is_resolved") is True:
        return True
    if _market_winning_outcome_index(market_info) is not None:
        return True
    status = str(
        market_info.get("status")
        or market_info.get("market_status")
        or market_info.get("marketStatus")
        or ""
    ).strip().lower()
    return status in {"resolved", "settled", "final"}


def _trade_token_id(trade: dict[str, Any]) -> str:
    return str(trade.get("asset") or trade.get("token_id") or "").strip()


def _trade_side(trade: dict[str, Any]) -> str:
    return str(trade.get("side") or "").strip().upper()


def _trade_timestamp(trade: dict[str, Any]) -> datetime | None:
    ts = trade.get("timestamp")
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).replace(tzinfo=None)
    except (TypeError, ValueError, OSError):
        return None


def _entry_anchor(row: TraderOrder) -> datetime | None:
    """Earliest time we believe this order was on the book.

    We use ``executed_at`` if set, otherwise ``created_at``. A trade
    must have happened at-or-after this anchor to be eligible to close
    this order.
    """
    anchor = row.executed_at or row.created_at
    if anchor is None:
        return None
    if anchor.tzinfo is not None:
        anchor = anchor.astimezone(timezone.utc).replace(tzinfo=None)
    return anchor


def _entry_token_id(row: TraderOrder) -> str:
    """Best-effort token_id from the order's payload."""
    payload = row.payload_json if isinstance(row.payload_json, dict) else {}
    candidates: list[Any] = []
    for key in (
        "token_id",
        "live_token_id",
        "outcome_token_id",
        "asset_id",
    ):
        value = payload.get(key)
        if value:
            candidates.append(value)
    market_payload = payload.get("market") if isinstance(payload, dict) else None
    if isinstance(market_payload, dict):
        for key in ("token_id", "outcome_token_id"):
            value = market_payload.get(key)
            if value:
                candidates.append(value)
    fill = payload.get("live_fill") if isinstance(payload, dict) else None
    if isinstance(fill, dict):
        value = fill.get("token_id")
        if value:
            candidates.append(value)
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text:
            return text
    return ""


def _entry_fill_size(row: TraderOrder) -> float:
    payload = row.payload_json if isinstance(row.payload_json, dict) else {}
    fill = payload.get("live_fill") if isinstance(payload, dict) else None
    if isinstance(fill, dict):
        size = safe_float(fill.get("filled_size"), None)
        if size is not None and size > 0.0:
            return float(size)
    # Fallback: derive from notional / entry_price.
    notional = safe_float(row.notional_usd, None)
    price = safe_float(row.entry_price, None) or safe_float(row.effective_price, None)
    if notional is not None and notional > 0.0 and price is not None and price > 0.0:
        return float(notional / price)
    return 0.0


def _entry_cost_basis(row: TraderOrder) -> float:
    """USD cost basis for the entry — what we paid to open."""
    payload = row.payload_json if isinstance(row.payload_json, dict) else {}
    fill = payload.get("live_fill") if isinstance(payload, dict) else None
    if isinstance(fill, dict):
        notional = safe_float(fill.get("filled_notional"), None)
        if notional is not None and notional > 0.0:
            return float(notional)
    notional = safe_float(row.notional_usd, None)
    if notional is not None and notional > 0.0:
        return float(notional)
    return 0.0


class _MatchResult:
    __slots__ = (
        "matched_size",
        "matched_proceeds",
        "matched_trades",
        "tx_hashes",
    )

    def __init__(self) -> None:
        self.matched_size: float = 0.0
        self.matched_proceeds: float = 0.0
        self.matched_trades: list[dict[str, Any]] = []
        self.tx_hashes: list[str] = []


def _match_sells_to_order(
    row: TraderOrder,
    candidate_sells: list[dict[str, Any]],
    *,
    consumed_sizes_by_tx: dict[str, float],
) -> _MatchResult:
    """FIFO match SELL trades to this order.

    Each sell trade has a transactionHash and a size. A single sell can
    fill multiple orders (rare — usually 1-1) so we track how much of
    each sell has already been allocated via consumed_sizes_by_tx and
    only consume the remaining portion.
    """
    needed = _entry_fill_size(row)
    result = _MatchResult()
    if needed <= 0.0:
        return result
    anchor = _entry_anchor(row) or datetime.min
    for trade in candidate_sells:
        ts = _trade_timestamp(trade)
        if ts is None or ts < anchor:
            continue
        tx_hash = str(trade.get("transactionHash") or "").strip()
        trade_size = safe_float(trade.get("size"), 0.0) or 0.0
        if trade_size <= 0.0:
            continue
        already_consumed = consumed_sizes_by_tx.get(tx_hash, 0.0)
        available = trade_size - already_consumed
        if available <= 0.0:
            continue
        take = min(available, needed - result.matched_size)
        if take <= 0.0:
            continue
        price = safe_float(trade.get("price"), None)
        if price is None or price < 0.0:
            continue
        result.matched_size += take
        result.matched_proceeds += take * price
        if tx_hash:
            consumed_sizes_by_tx[tx_hash] = already_consumed + take
            result.tx_hashes.append(tx_hash)
        result.matched_trades.append({
            "tx_hash": tx_hash,
            "price": price,
            "size": take,
            "timestamp": ts.isoformat() if isinstance(ts, datetime) else None,
        })
        if result.matched_size >= needed - 1e-9:
            break
    return result


async def verify_orders_against_wallet_trades(
    session: AsyncSession,
    *,
    wallet_address: str,
    order_window_start: datetime | None = None,
    order_ids: Iterable[str] | None = None,
    max_trades: int = 10000,
    commit: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Match TraderOrder closes to actual on-chain SELL trades.

    Args:
        session: open AsyncSession.
        wallet_address: proxy wallet address.
        order_window_start: only re-verify orders updated since this time
            (default: last 7 days). Pass datetime.min to verify all rows.
        order_ids: optional explicit set of order ids to verify.
        max_trades: cap on wallet_trades fetched (default 2000 — covers
            the last ~few weeks of activity for a typical bot).
        commit: whether to commit the session after updates.
        dry_run: if True, compute matches but do not write any changes.

    Returns:
        Summary dict with counts of orders examined / verified /
        unmatched, and totals.
    """
    wallet_lower = str(wallet_address or "").strip().lower()
    if not wallet_lower:
        return {"error": "missing_wallet_address", "examined": 0, "verified": 0, "unmatched": 0}

    try:
        trades_raw = await asyncio.wait_for(
            polymarket_client.get_wallet_trades_paginated(
                wallet_lower, max_trades=max_trades, page_size=500
            ),
            timeout=_HTTP_FETCH_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "polymarket_trade_verifier: get_wallet_trades exceeded %.0fs HTTP budget; skipping cycle",
            _HTTP_FETCH_TIMEOUT_SECONDS,
            extra={"wallet": wallet_lower},
        )
        return {
            "error": "wallet_trades_fetch_timeout",
            "examined": 0,
            "verified": 0,
            "unmatched": 0,
        }
    except Exception as exc:
        logger.warning(
            "polymarket_trade_verifier: get_wallet_trades failed",
            wallet=wallet_lower,
            exc_info=exc,
        )
        return {
            "error": "wallet_trades_fetch_failed",
            "error_type": type(exc).__name__,
            "examined": 0,
            "verified": 0,
            "unmatched": 0,
        }

    sells_by_token: dict[str, list[dict[str, Any]]] = {}
    for trade in trades_raw or []:
        if not isinstance(trade, dict):
            continue
        if _trade_side(trade) != "SELL":
            continue
        token_id = _trade_token_id(trade)
        if not token_id:
            continue
        sells_by_token.setdefault(token_id, []).append(trade)
    # Sort each bucket FIFO so earliest sells consume earliest orders.
    for trades in sells_by_token.values():
        trades.sort(key=lambda t: (_trade_timestamp(t) or datetime.min))

    consumed_sizes_by_tx: dict[str, float] = {}

    query = (
        select(TraderOrder)
        .where(TraderOrder.mode == "live")
        .where(TraderOrder.status.in_(_RESOLVED_STATUSES))
    )
    if order_ids is not None:
        ids = [str(oid).strip() for oid in order_ids if str(oid).strip()]
        if not ids:
            return {"examined": 0, "verified": 0, "unmatched": 0}
        query = query.where(TraderOrder.id.in_(ids))
    else:
        if order_window_start is None:
            order_window_start = utcnow().replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        query = query.where(TraderOrder.updated_at >= order_window_start)
        # Skip already-verified rows at SQL — same rationale as the
        # other verifier paths: avoids re-scanning verified history and
        # keeps the loop's per-row work bounded.
        query = query.where(
            (TraderOrder.verification_status.is_(None))
            | (TraderOrder.verification_status != TRADER_ORDER_VERIFICATION_WALLET_ACTIVITY)
        )
    query = query.order_by(TraderOrder.executed_at.asc(), TraderOrder.created_at.asc())

    rows = list((await session.execute(query)).scalars().all())

    examined = 0
    verified = 0
    unmatched = 0
    pnl_total_before = 0.0
    pnl_total_after = 0.0

    now = utcnow()

    # Commit per-batch to keep row-lock duration short.  Without this
    # the loop holds TransactionID locks on every updated trader_orders
    # row until the final commit, blocking other writers (orchestrator,
    # fast_trader_runtime) for seconds.  Production saw 3s+ lock holds
    # surface as ``LOCK CONTENTION`` watchdog warnings on UPDATE
    # trader_orders.  See _on_observe_lock_contention pool listener.
    _COMMIT_BATCH_SIZE = 50
    _commit_batch = 0

    for row in rows:
        examined += 1
        # Skip rows already verified by closed_positions or a prior
        # trade-matcher run — closed_positions takes priority because
        # its settlement price is deterministic and immune to
        # manual-user-sell conflation. Trade matcher only fills in
        # gaps where closed_positions didn't have data.
        existing_status = str(row.verification_status or "").strip().lower()
        if existing_status == "wallet_activity" and row.actual_profit is not None:
            continue
        prior_profit = float(row.actual_profit) if row.actual_profit is not None else 0.0
        token_id = _entry_token_id(row)
        if not token_id:
            unmatched += 1
            continue
        candidates = sells_by_token.get(token_id) or []
        if not candidates:
            unmatched += 1
            continue
        match = _match_sells_to_order(
            row, candidates, consumed_sizes_by_tx=consumed_sizes_by_tx
        )
        if match.matched_size <= 0.0:
            unmatched += 1
            continue
        cost_basis = _entry_cost_basis(row)
        # Allocate cost proportionally to matched size if we couldn't
        # match the full entry (rare — usually we match all of it).
        entry_size = _entry_fill_size(row)
        if entry_size > 0.0:
            allocated_cost = cost_basis * min(1.0, match.matched_size / entry_size)
        else:
            allocated_cost = cost_basis
        verified_pnl = match.matched_proceeds - allocated_cost
        verified += 1
        pnl_total_before += prior_profit
        pnl_total_after += verified_pnl

        if dry_run:
            continue

        row.actual_profit = float(verified_pnl)
        _reconcile_status_with_pnl(row, float(verified_pnl))
        row.updated_at = now
        first_tx = match.tx_hashes[0] if match.tx_hashes else None
        apply_trader_order_verification(
            row,
            verification_status=TRADER_ORDER_VERIFICATION_WALLET_ACTIVITY,
            verification_source="polymarket_wallet_trades",
            verification_reason=f"matched {len(match.matched_trades)} sell trade(s)",
            execution_wallet_address=wallet_lower,
            verification_tx_hash=first_tx,
            verified_at=now,
            force=True,
        )
        # Stamp the verified close on the payload so downstream code
        # has the trade lineage.
        payload = dict(row.payload_json or {})
        payload["verified_close"] = {
            "verified_at": now.isoformat(),
            "matched_size": match.matched_size,
            "matched_proceeds": match.matched_proceeds,
            "allocated_cost": allocated_cost,
            "realized_pnl": verified_pnl,
            "trades": match.matched_trades,
            "source": "polymarket_wallet_trades",
        }
        row.payload_json = payload
        append_trader_order_verification_event(
            session,
            trader_order_id=str(row.id),
            verification_status=TRADER_ORDER_VERIFICATION_WALLET_ACTIVITY,
            source="polymarket_wallet_trades",
            event_type="verified_close_from_wallet_trades",
            reason=f"verified_pnl={verified_pnl:.4f} matched_size={match.matched_size:.4f}",
            execution_wallet_address=wallet_lower,
            tx_hash=first_tx,
            payload_json={
                "matched_trades": match.matched_trades,
                "prior_actual_profit": prior_profit,
                "verified_actual_profit": verified_pnl,
            },
            created_at=now,
        )
        _commit_batch += 1
        if commit and _commit_batch >= _COMMIT_BATCH_SIZE:
            await session.commit()
            _commit_batch = 0

    if commit and not dry_run and _commit_batch > 0:
        await session.commit()

    return {
        "examined": examined,
        "verified": verified,
        "unmatched": unmatched,
        "wallet_trades_fetched": len(trades_raw or []),
        "pnl_total_before": pnl_total_before,
        "pnl_total_after": pnl_total_after,
        "pnl_delta": pnl_total_after - pnl_total_before,
        "dry_run": dry_run,
    }


def _bot_recorded_sell_fill(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Pull the bot's own confirmed SELL fill data off the order payload.

    The bot stamps its SELL submission's clob_order_id and (when filled)
    the actual fill price + size on payload['pending_live_exit']. This
    is the bot's OWN observation of its trade — Polymarket confirmed
    these specific fills against our specific clob_order_id, so they
    carry NO conflation with manual user trades on the same wallet.

    Returns dict with {filled_size, filled_notional_usd, average_fill_price,
    clob_order_id, status} or None if the SELL didn't confirm a fill.
    """
    if not isinstance(payload, dict):
        return None
    pending_exit = payload.get("pending_live_exit") or payload.get("exit_state")
    if not isinstance(pending_exit, dict):
        return None
    snapshot = pending_exit.get("snapshot")
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    candidates = (snapshot, pending_exit)

    def _first(keys: tuple[str, ...]) -> float | None:
        for src in candidates:
            if not isinstance(src, dict):
                continue
            for k in keys:
                v = safe_float(src.get(k), None, reject_nan_inf=True)
                if v is not None:
                    return v
        return None

    filled_size = _first(("filled_size", "size_matched", "matched_size", "filled_shares", "executed_size")) or 0.0
    filled_notional = _first(("filled_notional_usd", "filled_notional", "executed_notional_usd", "matched_notional_usd")) or 0.0
    avg_price = _first(("average_fill_price", "avg_fill_price", "executed_price", "matched_price", "price"))
    if filled_size <= 0.0 and filled_notional <= 0.0:
        return None
    if filled_notional <= 0.0 and avg_price is not None and filled_size > 0.0:
        filled_notional = filled_size * avg_price
    if avg_price is None and filled_size > 0.0:
        avg_price = filled_notional / filled_size
    clob_id = ""
    for src in candidates:
        if not isinstance(src, dict):
            continue
        for k in ("provider_clob_order_id", "exit_order_clob_id", "clob_order_id"):
            text = str(src.get(k) or "").strip()
            if text:
                clob_id = text
                break
        if clob_id:
            break
    status = ""
    for src in candidates:
        if not isinstance(src, dict):
            continue
        text = str(src.get("normalized_status") or src.get("status") or src.get("provider_status") or "").strip().lower()
        if text:
            status = text
            break
    return {
        "filled_size": float(filled_size),
        "filled_notional_usd": float(filled_notional),
        "average_fill_price": float(avg_price) if avg_price is not None else None,
        "clob_order_id": clob_id,
        "status": status,
    }


async def verify_orders_from_bot_lineage(
    session: AsyncSession,
    *,
    order_window_start: datetime | None = None,
    order_ids: Iterable[str] | None = None,
    commit: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Verify P&L from data the BOT itself recorded — not wallet aggregates.

    This is the authoritative path. It uses ONLY:
      * The bot's BUY entry fill (size, cost) — captured when our
        BUY clob_order_id confirmed a fill on Polymarket
      * The bot's SELL fill on its OWN clob_order_id (size, proceeds)
        — captured when polling Polymarket for the status of the
        SELL we submitted
      * The market's resolution (winning outcome) — deterministic
        from market metadata

    Manual user trades on the same wallet have DIFFERENT clob_order_ids,
    so they cannot affect the bot's recorded SELL fills. This eliminates
    the wallet conflation problem that closed_positions / trade-matcher
    suffer from.

    Verification rules per row:
      A) If payload['pending_live_exit'] has the bot's confirmed SELL
         fill data → realized_pnl = sell_proceeds - cost_basis_pro_rated
      B) Else, if market resolved → realized_pnl =
         (won?size:0) - cost_basis (deterministic)
      C) Else → unverified, no write, defer

    Writes verification_status='wallet_activity' (the only status the
    DB-layer guard accepts) with verification_source='bot_lineage'.
    """
    query = (
        select(TraderOrder)
        .where(TraderOrder.mode == "live")
        .where(TraderOrder.status.in_(_RESOLVED_STATUSES))
    )
    if order_ids is not None:
        ids = [str(oid).strip() for oid in order_ids if str(oid).strip()]
        if not ids:
            return {"examined": 0, "verified_sell_fill": 0, "verified_resolution": 0, "unmatched": 0}
        query = query.where(TraderOrder.id.in_(ids))
    else:
        if order_window_start is None:
            from datetime import timedelta
            order_window_start = utcnow() - timedelta(days=14)
        query = query.where(TraderOrder.updated_at >= order_window_start)
        # Skip rows already wallet_activity-verified by a prior cycle.
        # Without this, the loop re-fetches the entire backlog every
        # reconciliation tick and re-issues a Polymarket HTTP lookup
        # per unmatched row (Path B below), holding the session well
        # past the 30s statement timeout.
        query = query.where(
            (TraderOrder.verification_status.is_(None))
            | (TraderOrder.verification_status != TRADER_ORDER_VERIFICATION_WALLET_ACTIVITY)
        )
    rows = list((await session.execute(query)).scalars().all())

    examined = 0
    verified_sell_fill = 0
    verified_resolution = 0
    unmatched = 0
    pnl_total_before = 0.0
    pnl_total_after = 0.0
    now = utcnow()

    # ── Phase 1 (Python only, NO DB transaction held): figure out which
    # rows fall into Path B (bot didn't record its own SELL fill, so we
    # need market resolution metadata).  Path A rows just use payload
    # data and don't need any HTTP.
    rows_needing_market_info: list[TraderOrder] = []
    for row in rows:
        payload = row.payload_json if isinstance(row.payload_json, dict) else {}
        bot_size = _entry_fill_size(row)
        bot_cost = _entry_cost_basis(row)
        if bot_size <= 0.0 or bot_cost <= 0.0:
            continue
        sell = _bot_recorded_sell_fill(payload)
        if sell is not None and sell["filled_size"] > 0.0:
            continue  # Path A — no HTTP needed
        rows_needing_market_info.append(row)  # Path B — needs market_info

    # ── Phase 2: prefetch market_info OUTSIDE the transaction so PG
    # doesn't see this connection as ``idle in transaction`` for the
    # 200-500ms × N rows the HTTP fan-out takes.  The previous in-loop
    # pattern was the dominant source of "polymarket_trade_verifier
    # (bot_lineage) timed out after 30.0s" — N rows × per-row HTTP
    # held the session well past the 30s budget.
    #
    # Cap the per-cycle prefetch batch so a 400-row backlog can't still
    # blow the 30 s budget.  Un-prefetched rows fall through Path B
    # below with ``market_info=None`` → they stay unmatched and are
    # reattempted on the next reconciliation tick (300 s later).
    market_info_by_id: dict[str, dict[str, Any] | None] = {}
    deferred_prefetch_rows = 0
    if rows_needing_market_info:
        prefetch_batch = rows_needing_market_info[:_BOT_LINEAGE_PREFETCH_MAX_ROWS]
        deferred_prefetch_rows = len(rows_needing_market_info) - len(prefetch_batch)
        async with release_conn(session):
            market_info_by_id = await _prefetch_market_info(prefetch_batch)
        if deferred_prefetch_rows > 0:
            logger.info(
                "polymarket_trade_verifier (bot_lineage) deferring rows past prefetch cap",
                total_needing_info=len(rows_needing_market_info),
                prefetched=len(prefetch_batch),
                deferred=deferred_prefetch_rows,
                cap=_BOT_LINEAGE_PREFETCH_MAX_ROWS,
            )

    # ── Phase 3: writes only.  All HTTP is done; the session/transaction
    # is held strictly for SQL.
    # Commit per-batch (every 50 rows) to keep row-lock duration short.
    _commit_batch = 0
    for row in rows:
        examined += 1
        payload = row.payload_json if isinstance(row.payload_json, dict) else {}
        prior = float(row.actual_profit) if row.actual_profit is not None else 0.0
        bot_size = _entry_fill_size(row)
        bot_cost = _entry_cost_basis(row)
        if bot_size <= 0.0 or bot_cost <= 0.0:
            unmatched += 1
            continue
        # Path A: bot recorded a confirmed SELL fill on its own
        # clob_order_id. This is the strongest evidence — manual user
        # trades have different clob_order_ids and cannot pollute it.
        sell = _bot_recorded_sell_fill(payload)
        if sell is not None and sell["filled_size"] > 0.0:
            raw_sell_size = sell["filled_size"]
            raw_sell_proceeds = sell["filled_notional_usd"]
            # CRITICAL: pending_live_exit.snapshot can carry the WALLET's
            # aggregate filled_size for the token (when reconciliation
            # populated it from the wallet positions API rather than
            # from the bot's specific clob_order_id status). The bot
            # could only have sold what it bought — cap sell_size to
            # bot_size, and pro-rate proceeds at the same avg price.
            # Without this cap, the bot's small entry gets credited the
            # wallet's entire close (real example: bot's 11.45-share SPY
            # entry got credited 267.85 shares of wallet sells at $229
            # proceeds, attributed +$219 phantom win).
            if raw_sell_size > bot_size + 1e-9:
                avg_sell_price = raw_sell_proceeds / raw_sell_size if raw_sell_size > 0 else 0.0
                sell_size = bot_size
                sell_proceeds = bot_size * avg_sell_price
            else:
                sell_size = raw_sell_size
                sell_proceeds = raw_sell_proceeds
            allocated_cost = bot_cost * min(1.0, sell_size / bot_size)
            verified_pnl = sell_proceeds - allocated_cost
            verified_sell_fill += 1
            pnl_total_before += prior
            pnl_total_after += verified_pnl
            if not dry_run:
                row.actual_profit = float(verified_pnl)
                _reconcile_status_with_pnl(row, float(verified_pnl))
                row.updated_at = now
                apply_trader_order_verification(
                    row,
                    verification_status=TRADER_ORDER_VERIFICATION_WALLET_ACTIVITY,
                    verification_source="bot_lineage_sell_fill",
                    verification_reason=f"bot_sell_clob={sell['clob_order_id'][:14]}... fill={sell_size:.4f}@{(sell['average_fill_price'] or 0.0):.4f}",
                    verification_tx_hash=sell["clob_order_id"] or None,
                    verified_at=now,
                    force=True,
                )
                payload["verified_close"] = {
                    "verified_at": now.isoformat(),
                    "source": "bot_lineage_sell_fill",
                    "matched_size": sell_size,
                    "matched_proceeds": sell_proceeds,
                    "allocated_cost": allocated_cost,
                    "realized_pnl": verified_pnl,
                    "bot_sell_clob_order_id": sell["clob_order_id"],
                    "bot_sell_avg_price": sell["average_fill_price"],
                }
                row.payload_json = payload
                append_trader_order_verification_event(
                    session,
                    trader_order_id=str(row.id),
                    verification_status=TRADER_ORDER_VERIFICATION_WALLET_ACTIVITY,
                    source="bot_lineage_sell_fill",
                    event_type="verified_close_from_bot_sell_fill",
                    reason=f"verified_pnl={verified_pnl:.4f} bot_sell_size={sell_size:.4f}",
                    tx_hash=sell["clob_order_id"] or None,
                    payload_json={
                        "bot_sell_clob_order_id": sell["clob_order_id"],
                        "bot_sell_avg_price": sell["average_fill_price"],
                        "prior_actual_profit": prior,
                        "verified_actual_profit": verified_pnl,
                    },
                    created_at=now,
                )
                _commit_batch += 1
                if commit and _commit_batch >= 50:
                    await session.commit()
                    _commit_batch = 0
            continue

        # Path B: market resolution. Deterministic payout from the
        # bot's recorded outcome + market's winning outcome. Does NOT
        # rely on wallet aggregates.  market_info was pre-fetched in
        # Phase 2 above with no transaction held; this is now a pure
        # dict lookup.
        market_info = market_info_by_id.get(str(row.id))
        if isinstance(market_info, dict) and _market_is_resolved(market_info):
            winning_idx = _market_winning_outcome_index(market_info)
            our_idx = _direction_to_outcome_index(row.direction)
            if winning_idx is not None and our_idx is not None:
                won = our_idx == winning_idx
                payout = bot_size * (1.0 if won else 0.0)
                verified_pnl = payout - bot_cost
                verified_resolution += 1
                pnl_total_before += prior
                pnl_total_after += verified_pnl
                if not dry_run:
                    row.actual_profit = float(verified_pnl)
                    _reconcile_status_with_pnl(row, float(verified_pnl))
                    row.updated_at = now
                    apply_trader_order_verification(
                        row,
                        verification_status=TRADER_ORDER_VERIFICATION_WALLET_ACTIVITY,
                        verification_source="bot_lineage_resolution",
                        verification_reason=("won" if won else "lost") + " on resolution",
                        verified_at=now,
                        force=True,
                    )
                    payload["verified_close"] = {
                        "verified_at": now.isoformat(),
                        "source": "bot_lineage_resolution",
                        "matched_size": bot_size,
                        "matched_proceeds": payout,
                        "allocated_cost": bot_cost,
                        "realized_pnl": verified_pnl,
                        "winning_outcome_index": winning_idx,
                        "our_outcome_index": our_idx,
                        "won": won,
                    }
                    row.payload_json = payload
                    append_trader_order_verification_event(
                        session,
                        trader_order_id=str(row.id),
                        verification_status=TRADER_ORDER_VERIFICATION_WALLET_ACTIVITY,
                        source="bot_lineage_resolution",
                        event_type="verified_close_from_market_resolution",
                        reason=f"verified_pnl={verified_pnl:.4f} won={won}",
                        payload_json={
                            "winning_outcome_index": winning_idx,
                            "our_outcome_index": our_idx,
                            "prior_actual_profit": prior,
                            "verified_actual_profit": verified_pnl,
                        },
                        created_at=now,
                    )
                    _commit_batch += 1
                    if commit and _commit_batch >= 50:
                        await session.commit()
                        _commit_batch = 0
                continue

        unmatched += 1

    if commit and _commit_batch > 0 and not dry_run:
        await session.commit()

    return {
        "examined": examined,
        "verified_sell_fill": verified_sell_fill,
        "verified_resolution": verified_resolution,
        "unmatched": unmatched,
        "pnl_total_before": pnl_total_before,
        "pnl_total_after": pnl_total_after,
        "pnl_delta": pnl_total_after - pnl_total_before,
        "dry_run": dry_run,
    }


async def verify_orders_against_closed_positions(
    session: AsyncSession,
    *,
    wallet_address: str,
    order_window_start: datetime | None = None,
    order_ids: Iterable[str] | None = None,
    max_positions: int = _CLOSED_POSITIONS_VERIFY_MAX_FETCH,
    commit: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Resolve held-to-resolution P&L from Polymarket closed_positions.

    For positions the bot held until market resolution, there is no
    SELL trade in /data/trades — the wallet simply gets paid out
    deterministically ($1 per winning share, $0 per losing share). The
    /data/closed-positions endpoint reflects exactly this state with
    curPrice = the resolved settlement price.

    For each TraderOrder not yet wallet_activity-verified by the trade
    matcher above:
      1. Look up the matching closed_position by asset (token_id)
      2. Compute realized_pnl using the bot's own cost basis and the
         resolution settlement price:
            bot_pnl = (settled_curPrice - bot_avg_price) * bot_size
      3. This is robust to the wallet/manual-trade conflation problem
         because settled curPrice is the same for everyone who held
         the token (resolution payouts are deterministic per outcome).

    Writes verification_status = wallet_activity with
    verification_source = polymarket_closed_positions, since the
    on-chain settlement is unambiguous truth.
    """
    wallet_lower = str(wallet_address or "").strip().lower()
    if not wallet_lower:
        return {"error": "missing_wallet_address", "examined": 0, "verified": 0, "unmatched": 0}

    fetch_limit = min(
        max(1, int(max_positions or _CLOSED_POSITIONS_VERIFY_MAX_FETCH)),
        _CLOSED_POSITIONS_VERIFY_MAX_FETCH,
    )
    try:
        positions_raw = await asyncio.wait_for(
            polymarket_client.get_closed_positions_paginated(
                wallet_lower, max_positions=fetch_limit
            ),
            timeout=_HTTP_FETCH_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "polymarket_trade_verifier closed_positions fetch exceeded %.0fs HTTP budget; skipping cycle",
            _HTTP_FETCH_TIMEOUT_SECONDS,
            extra={"wallet": wallet_lower},
        )
        return {
            "error": "closed_positions_fetch_timeout",
            "examined": 0,
            "verified": 0,
            "unmatched": 0,
        }
    except Exception as exc:
        logger.warning(
            "polymarket_trade_verifier closed_positions fetch failed",
            wallet=wallet_lower,
            exc_info=exc,
        )
        return {
            "error": "closed_positions_fetch_failed",
            "error_type": type(exc).__name__,
            "examined": 0,
            "verified": 0,
            "unmatched": 0,
        }

    closed_by_asset: dict[str, dict[str, Any]] = {}
    for cp in positions_raw or []:
        if not isinstance(cp, dict):
            continue
        asset = str(cp.get("asset") or "").strip()
        if not asset:
            continue
        # If multiple closed_positions for same asset (rare — wallet
        # opens, fully closes, re-opens, fully closes again), keep the
        # most recent by timestamp.
        existing = closed_by_asset.get(asset)
        if existing is not None:
            new_ts = safe_float(cp.get("timestamp"), 0.0) or 0.0
            old_ts = safe_float(existing.get("timestamp"), 0.0) or 0.0
            if new_ts <= old_ts:
                continue
        closed_by_asset[asset] = cp

    query = (
        select(TraderOrder)
        .where(TraderOrder.mode == "live")
        .where(TraderOrder.status.in_(_RESOLVED_STATUSES))
    )
    if order_ids is not None:
        ids = [str(oid).strip() for oid in order_ids if str(oid).strip()]
        if not ids:
            return {"examined": 0, "verified": 0, "unmatched": 0}
        query = query.where(TraderOrder.id.in_(ids))
    else:
        if order_window_start is None:
            from datetime import timedelta
            order_window_start = utcnow() - timedelta(days=14)
        query = query.where(TraderOrder.updated_at >= order_window_start)
        # Skip already-verified rows at SQL — same rationale as the
        # bot_lineage path: avoids re-scanning the full backlog every
        # reconciliation cycle and busting the 30s statement timeout.
        query = query.where(
            (TraderOrder.verification_status.is_(None))
            | (TraderOrder.verification_status != TRADER_ORDER_VERIFICATION_WALLET_ACTIVITY)
        )

    rows = list((await session.execute(query)).scalars().all())

    examined = 0
    verified = 0
    unmatched = 0
    skipped_already_verified = 0
    pnl_total_before = 0.0
    pnl_total_after = 0.0
    now = utcnow()

    for row in rows:
        examined += 1
        # The trade-matcher above is HIGHER fidelity (transaction-hash
        # level) than per-position aggregate, so don't override its
        # values. This path only fills in held-to-resolution positions
        # the trade matcher couldn't see.
        existing_status = str(row.verification_status or "").strip().lower()
        if existing_status == "wallet_activity" and row.actual_profit is not None:
            skipped_already_verified += 1
            continue

        token_id = _entry_token_id(row)
        if not token_id:
            unmatched += 1
            continue
        cp = closed_by_asset.get(token_id)
        if not cp:
            unmatched += 1
            continue

        # The bot's own cost basis and size from our recorded fill.
        bot_size = _entry_fill_size(row)
        bot_cost = _entry_cost_basis(row)
        if bot_size <= 0.0 or bot_cost <= 0.0:
            unmatched += 1
            continue
        bot_avg_price = bot_cost / bot_size
        # Resolution settlement price (curPrice on a closed position
        # is the on-chain settlement value: $1 per winning share, $0
        # per losing share, or fractional in rare cases).
        settled_price = safe_float(cp.get("curPrice"), None)
        if settled_price is None or settled_price < 0.0:
            unmatched += 1
            continue
        verified_pnl = (settled_price - bot_avg_price) * bot_size
        prior_profit = float(row.actual_profit) if row.actual_profit is not None else 0.0
        verified += 1
        pnl_total_before += prior_profit
        pnl_total_after += verified_pnl

        if dry_run:
            continue

        row.actual_profit = float(verified_pnl)
        _reconcile_status_with_pnl(row, float(verified_pnl))
        row.updated_at = now
        apply_trader_order_verification(
            row,
            verification_status=TRADER_ORDER_VERIFICATION_WALLET_ACTIVITY,
            verification_source="polymarket_closed_positions",
            verification_reason=f"settled@{settled_price:.4f}",
            execution_wallet_address=wallet_lower,
            verified_at=now,
            force=True,
        )
        payload = dict(row.payload_json or {})
        payload["verified_close"] = {
            "verified_at": now.isoformat(),
            "matched_size": bot_size,
            "settled_price": settled_price,
            "bot_avg_price": bot_avg_price,
            "matched_proceeds": bot_size * settled_price,
            "allocated_cost": bot_cost,
            "realized_pnl": verified_pnl,
            "source": "polymarket_closed_positions",
            "wallet_aggregate_realized_pnl": cp.get("realizedPnl"),
            "wallet_aggregate_total_bought": cp.get("totalBought"),
        }
        row.payload_json = payload
        append_trader_order_verification_event(
            session,
            trader_order_id=str(row.id),
            verification_status=TRADER_ORDER_VERIFICATION_WALLET_ACTIVITY,
            source="polymarket_closed_positions",
            event_type="verified_close_from_closed_positions",
            reason=f"verified_pnl={verified_pnl:.4f} settled={settled_price:.4f}",
            payload_json={
                "settled_price": settled_price,
                "bot_avg_price": bot_avg_price,
                "bot_size": bot_size,
                "prior_actual_profit": prior_profit,
                "verified_actual_profit": verified_pnl,
            },
            created_at=now,
        )

    if commit and not dry_run and verified > 0:
        await session.commit()

    return {
        "examined": examined,
        "verified": verified,
        "unmatched": unmatched,
        "skipped_already_verified": skipped_already_verified,
        "closed_positions_fetched": len(positions_raw or []),
        "pnl_total_before": pnl_total_before,
        "pnl_total_after": pnl_total_after,
        "pnl_delta": pnl_total_after - pnl_total_before,
        "dry_run": dry_run,
    }


async def verify_orders_against_market_resolutions(
    session: AsyncSession,
    *,
    order_window_start: datetime | None = None,
    order_ids: Iterable[str] | None = None,
    commit: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Verify P&L for resolved markets via deterministic payout.

    For markets that have settled, winning outcome shares pay $1 each
    and losing shares pay $0. This is on-chain deterministic — when the
    market resolves, the proxy wallet's winning shares become claimable
    USDC at par. We DO NOT need a SELL trade record to know the P&L for
    resolution closures.

    This function targets orders that:
      * are in a closed status (resolved/closed_win/closed_loss)
      * have verification_status = summary_only OR actual_profit IS NULL
        (i.e. weren't matched by the trade-based verifier above)
      * the underlying market has resolved with a known winning outcome

    For each match: realized_pnl = (winning ? size * 1.0 : 0.0) - cost_basis
    Writes verification_status = wallet_activity (this IS truth — the
    payout is deterministic from chain state) with verification_source
    = polymarket_market_resolution.
    """
    from services.polymarket import polymarket_client

    query = (
        select(TraderOrder)
        .where(TraderOrder.mode == "live")
        .where(TraderOrder.status.in_(_RESOLVED_STATUSES))
    )
    if order_ids is not None:
        ids = [str(oid).strip() for oid in order_ids if str(oid).strip()]
        if not ids:
            return {"examined": 0, "verified": 0, "skipped": 0}
        query = query.where(TraderOrder.id.in_(ids))
    else:
        if order_window_start is None:
            # 7-day default window — resolutions usually happen within
            # days/weeks of entry.
            from datetime import timedelta
            order_window_start = utcnow() - timedelta(days=90)
        query = query.where(TraderOrder.updated_at >= order_window_start)

    rows = list((await session.execute(query)).scalars().all())

    examined = 0
    verified = 0
    skipped_unresolved = 0
    skipped_already_verified = 0
    pnl_total_before = 0.0
    pnl_total_after = 0.0
    now = utcnow()

    # Pre-fetch wallet trades — the BUY trade is the authoritative
    # record of which outcome we hold (the order's "direction" string
    # has been observed to mismap on multi-outcome markets, e.g. the
    # Athletics vs Texas Rangers row recorded a $49.70 win when we
    # actually bought Texas Rangers — the LOSING side — for a $9 loss).
    # The on-chain BUY trade carries an `outcomeIndex` which is
    # unambiguous truth.
    wallet_lower_for_buys = ""
    try:
        from services.live_execution_service import live_execution_service as _les
        wallet_lower_for_buys = (_les.get_execution_wallet_address() or "").strip().lower()
    except Exception:
        wallet_lower_for_buys = ""
    buys_by_token: dict[str, list[dict[str, Any]]] = {}
    if wallet_lower_for_buys:
        try:
            all_trades = await polymarket_client.get_wallet_trades_paginated(
                wallet_lower_for_buys, max_trades=3000, page_size=500
            )
            for trade in all_trades or []:
                if not isinstance(trade, dict) or _trade_side(trade) != "BUY":
                    continue
                tok = _trade_token_id(trade)
                if not tok:
                    continue
                buys_by_token.setdefault(tok, []).append(trade)
        except Exception as exc:
            logger.warning(
                "polymarket_trade_verifier resolution: get_wallet_trades failed",
                exc_info=exc,
            )

    # Pre-fetch market_info for all rows OUTSIDE the transaction.  The
    # old per-row in-loop pattern held the session "idle in transaction"
    # for the full HTTP fan-out duration — same root cause as the
    # bot_lineage timeouts.
    market_info_by_id: dict[str, dict[str, Any] | None] = {}
    if rows:
        async with release_conn(session):
            market_info_by_id = await _prefetch_market_info(rows)

    for row in rows:
        examined += 1
        # AUTHORITATIVE: Polymarket resolution data is single source of
        # truth. We OVERRIDE any prior actual_profit (even venue_fill /
        # wallet_activity values) when the market has resolved and we
        # can determine win/loss from the on-chain BUY trade. The
        # original Athletics/Rangers row was venue_fill = $49.70 but
        # the truth was a $9 loss — the only way to catch that is to
        # not skip "already verified" rows here.
        market_info = market_info_by_id.get(str(row.id))
        if not isinstance(market_info, dict) or not _market_is_resolved(market_info):
            skipped_unresolved += 1
            continue

        winning_idx = _market_winning_outcome_index(market_info)
        if winning_idx is None:
            skipped_unresolved += 1
            continue

        # Determine which outcome WE actually held. Prefer the BUY
        # trade's outcomeIndex (on-chain truth); fall back to the
        # order's direction mapping. The fallback is unreliable on
        # multi-outcome markets — the BUY trade is what we want.
        token_id = _entry_token_id(row)
        anchor = _entry_anchor(row) or datetime.min
        buy_trade: dict[str, Any] | None = None
        for candidate in buys_by_token.get(token_id, []):
            ts = _trade_timestamp(candidate)
            if ts is not None and ts < anchor:
                continue
            buy_trade = candidate
            break
        our_idx: int | None = None
        if buy_trade is not None:
            try:
                our_idx = int(buy_trade.get("outcomeIndex"))
            except (TypeError, ValueError):
                our_idx = None
        if our_idx is None:
            our_idx = _direction_to_outcome_index(row.direction)
        if our_idx is None:
            skipped_unresolved += 1
            continue

        # Cost basis: prefer the actual BUY trade (size * price) which
        # captures real slippage; fall back to recorded notional.
        if buy_trade is not None:
            buy_size = safe_float(buy_trade.get("size"), 0.0) or 0.0
            buy_price = safe_float(buy_trade.get("price"), None)
            if buy_size > 0.0 and buy_price is not None and buy_price >= 0.0:
                size = buy_size
                cost_basis = buy_size * buy_price
            else:
                size = _entry_fill_size(row)
                cost_basis = _entry_cost_basis(row)
        else:
            size = _entry_fill_size(row)
            cost_basis = _entry_cost_basis(row)
        if size <= 0.0:
            skipped_unresolved += 1
            continue

        won = our_idx == winning_idx
        payout = size * (1.0 if won else 0.0)
        verified_pnl = payout - cost_basis
        prior_profit = float(row.actual_profit) if row.actual_profit is not None else 0.0
        verified += 1
        pnl_total_before += prior_profit
        pnl_total_after += verified_pnl

        if dry_run:
            continue

        row.actual_profit = float(verified_pnl)
        _reconcile_status_with_pnl(row, float(verified_pnl))
        row.updated_at = now
        apply_trader_order_verification(
            row,
            verification_status=TRADER_ORDER_VERIFICATION_WALLET_ACTIVITY,
            verification_source="polymarket_market_resolution",
            verification_reason=("won" if won else "lost") + " on resolution",
            verified_at=now,
            force=True,
        )
        payload = dict(row.payload_json or {})
        payload["verified_close"] = {
            "verified_at": now.isoformat(),
            "matched_size": size,
            "matched_proceeds": payout,
            "allocated_cost": cost_basis,
            "realized_pnl": verified_pnl,
            "winning_outcome_index": winning_idx,
            "our_outcome_index": our_idx,
            "won": won,
            "source": "polymarket_market_resolution",
        }
        row.payload_json = payload
        append_trader_order_verification_event(
            session,
            trader_order_id=str(row.id),
            verification_status=TRADER_ORDER_VERIFICATION_WALLET_ACTIVITY,
            source="polymarket_market_resolution",
            event_type="verified_close_from_market_resolution",
            reason=f"verified_pnl={verified_pnl:.4f} won={won} size={size:.4f}",
            payload_json={
                "winning_outcome_index": winning_idx,
                "our_outcome_index": our_idx,
                "prior_actual_profit": prior_profit,
                "verified_actual_profit": verified_pnl,
            },
            created_at=now,
        )

    if commit and not dry_run and verified > 0:
        await session.commit()

    return {
        "examined": examined,
        "verified": verified,
        "skipped_unresolved": skipped_unresolved,
        "skipped_already_verified": skipped_already_verified,
        "pnl_total_before": pnl_total_before,
        "pnl_total_after": pnl_total_after,
        "pnl_delta": pnl_total_after - pnl_total_before,
        "dry_run": dry_run,
    }
