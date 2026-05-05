"""In-memory wallet state, derived from Polymarket CLOB user-channel WS.

**Architecture role**: this is the single source of truth for wallet
positions/orders/fills on the orchestrator's hot path.  Two writers
populate it:

1. ``trader_reconciliation_worker`` calls ``seed_from_rest()`` on
   bootstrap and periodically (~30s) — REST snapshot for baseline.
2. ``PolymarketUserFeed`` calls ``apply_trade()`` / ``apply_order()``
   for real-time deltas as the wallet's own orders fill.

The orchestrator + fast trader **only read** from this cache via the
``get_*`` methods.  Reads are sub-microsecond dict lookups; the
orchestrator never blocks on REST.

Freshness contract: hot-path callers MUST call ``is_fresh()`` before
trusting the cache.  When stale, the orchestrator skips the cycle and
emits an audit alert — institutional-grade defensive degradation, no
silent miss.

Position derivation: WS doesn't push position snapshots, only fills.
Positions are derived from cumulative fills since the last REST seed.
The reconciliation worker periodically reseeds from REST to absorb any
drift (typically zero, but defends against missed events on reconnect).

Event-driven mutation contract
------------------------------
Every state-changing call (``apply_trade``, ``apply_order``,
``seed_from_rest``, ``mark_ws_state``) emits a single
``wallet_state.changed`` event on the in-process ``event_bus`` AFTER the
lock is released and the mutation has been committed.  Subscribers (the
trader orchestrator's freshness check, the wallet_state_bus
cross-process publisher) wake on this event instead of polling.  The
emit is fire-and-forget: requires an active asyncio loop (all production
callsites have one), silently no-ops otherwise, and dispatches via
``loop.create_task(...)`` so the lock is released before subscribers run.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Literal, Optional

from utils.logger import get_logger
from utils.utcnow import utcnow

logger = get_logger("wallet_state_cache")

# Single event type emitted on every wallet-state mutation.  Subscribers
# switch on the ``kind`` field in the payload to react selectively.
WALLET_STATE_CHANGED_EVENT = "wallet_state.changed"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result else default  # filter NaN


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_event_timestamp(value: Any) -> datetime:
    """Polymarket WS uses unix timestamps as strings."""
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return utcnow()
    # Polymarket timestamps are seconds (sometimes ms — auto-detect).
    if ts > 1e12:
        ts = ts / 1000.0
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except (OverflowError, ValueError, OSError):
        return utcnow()


@dataclass(slots=True)
class WalletPosition:
    """Net open position derived from cumulative trade fills.

    Cost basis is FIFO-style: ``cost_basis_usd`` accumulates buy
    notional minus sell proceeds.  ``avg_entry_price`` is
    ``cost_basis_usd / size`` when size > 0.  When size hits zero
    (position fully exited) the position is removed from the cache.
    """

    token_id: str
    condition_id: str
    outcome_index: int = 0
    size: float = 0.0
    cost_basis_usd: float = 0.0
    last_fill_at: Optional[datetime] = None
    # Settlement fields are populated by the reconciliation worker via
    # ``seed_from_rest``; the user-channel WS does not push resolution
    # events (those live on-chain, not on CLOB).
    is_resolved: bool = False
    settlement_price: Optional[float] = None
    redeemable: bool = False
    # Free-form mark price from the most recent REST snapshot.  Used
    # by mark-to-market checks; keep in sync with ``feed_manager.cache``
    # when WS prices are stale.
    last_rest_mark_price: Optional[float] = None

    @property
    def avg_entry_price(self) -> float:
        if self.size <= 0.0:
            return 0.0
        return self.cost_basis_usd / self.size

    def to_legacy_dict(self) -> dict[str, Any]:
        """Render in the shape the orchestrator's old REST-driven code
        expects.  This lets ``_load_execution_*`` callers be
        re-pointed at the cache without touching every consumer.
        """
        return {
            "token_id": self.token_id,
            "asset": self.token_id,
            "asset_id": self.token_id,
            "conditionId": self.condition_id,
            "condition_id": self.condition_id,
            "outcomeIndex": self.outcome_index,
            "size": self.size,
            "positionSize": self.size,
            "shares": self.size,
            "avgPrice": self.avg_entry_price,
            "average_price": self.avg_entry_price,
            "initialValue": self.cost_basis_usd,
            "curPrice": self.last_rest_mark_price,
            "currentPrice": self.last_rest_mark_price,
            "redeemable": self.redeemable,
            "counts_as_open": self.size > 0.0,
        }


@dataclass(slots=True)
class WalletOrder:
    """An order placed by this wallet, tracked through its lifecycle."""

    order_id: str
    token_id: str
    condition_id: str
    side: Literal["BUY", "SELL"]
    price: float
    original_size: float
    size_matched: float = 0.0
    status: str = "PLACEMENT"  # PLACEMENT|UPDATE|MATCHED|MINED|CONFIRMED|CANCELLATION|FAILED
    placed_at: Optional[datetime] = None
    last_update: Optional[datetime] = None

    @property
    def is_terminal(self) -> bool:
        return self.status in {"CONFIRMED", "CANCELLATION", "FAILED"}

    @property
    def remaining_size(self) -> float:
        return max(0.0, self.original_size - self.size_matched)


@dataclass(slots=True)
class WalletFill:
    """A single trade fill on the wallet."""

    trade_id: str
    order_id: str
    token_id: str
    condition_id: str
    side: Literal["BUY", "SELL"]
    price: float
    size: float
    status: str  # MATCHED|MINED|CONFIRMED|FAILED
    matched_at: datetime
    confirmed_at: Optional[datetime] = None


# Caps to keep memory bounded.  Open-order count is ~hundreds in
# steady state; recent fills are a sliding window for matching against
# bot-submitted orders by the verification layer.
_MAX_TRACKED_ORDERS = 5_000
_MAX_RECENT_FILLS = 1_000
# Hot-path freshness threshold.  Composite check: WS must be connected
# AND at least one signal seen recently (or REST seed within window).
# Tuned for Polymarket's typical fill cadence.
_DEFAULT_FRESH_MAX_AGE_SECONDS = 30.0
# REST seed must have run at least once for the cache to be trusted;
# this threshold guards against "WS connected but reconciliation
# worker dead" — the cache could be missing rows that exist on
# Polymarket's side.
_REST_SEED_MAX_AGE_SECONDS = 120.0


class WalletStateCache:
    """Thread-safe in-memory wallet state.

    All state mutation goes through methods that acquire ``self._lock``.
    Hot-path readers acquire the same lock; expected contention is
    sub-microsecond.

    Lifecycle:
      1. ``feed_manager.start()`` constructs the cache as a singleton
      2. ``trader_reconciliation_worker`` boots, calls
         ``seed_from_rest(...)`` once with the REST baseline
      3. ``PolymarketUserFeed`` connects, applies live deltas via
         ``apply_trade()`` / ``apply_order()``
      4. Reconciliation worker re-seeds every 30s (drift audit)
      5. Hot path reads via ``get_position()``, ``get_open_orders()``,
         etc. — never writes.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._positions: dict[str, WalletPosition] = {}
        # Secondary index for condition_id-based lookups (the user channel
        # subscribes by condition_id, so we need the reverse map).
        self._condition_to_token_ids: dict[str, set[str]] = {}
        self._orders: dict[str, WalletOrder] = {}
        self._recent_fills: deque[WalletFill] = deque(maxlen=_MAX_RECENT_FILLS)
        # Trade IDs we've already applied — guards against duplicate
        # events on reconnect/replay.
        self._applied_trade_ids: deque[str] = deque(maxlen=_MAX_RECENT_FILLS * 2)
        self._applied_trade_id_set: set[str] = set()

        self._wallet_address: Optional[str] = None
        self._ws_connected: bool = False
        self._ws_connected_since_mono: Optional[float] = None
        # Time of the last WS event applied (any event) — used to
        # detect "WS connected but feed silent" liveness gap.
        self._last_ws_event_mono: Optional[float] = None
        # REST seeding state — set by the reconciliation worker.
        self._last_rest_seed_mono: Optional[float] = None
        self._last_rest_seed_succeeded: bool = False
        self._rest_seed_count: int = 0
        # Counters for diagnostics.
        self._trades_applied: int = 0
        self._orders_applied: int = 0
        self._duplicate_trades_skipped: int = 0

    # -------------------- Event emit --------------------

    def _emit_change(self, kind: str, payload: Optional[dict[str, Any]] = None) -> None:
        """Fire a ``wallet_state.changed`` event for downstream subscribers.

        Two delivery channels, both soft-fail:
          1. In-process ``event_bus`` — wakes orchestrator + fast trader.
          2. Cross-process Redis pub/sub — wakes API plane subscribers.

        Both run as ``loop.create_task`` so the caller (which holds no
        lock at this point) returns immediately.  If no asyncio loop is
        running (tests, sync scripts), the emit is silently dropped.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no async loop — fine for tests, sync entry points

        kind_str = str(kind or "").strip() or "change"
        body: dict[str, Any] = {"kind": kind_str, "wallet": self._wallet_address}
        if payload:
            body.update(payload)

        async def _dispatch() -> None:
            # In-process event_bus.  Imported lazily to avoid an import
            # cycle (services/event_bus → services/wallet_state_cache via
            # any future helper).
            try:
                from services.event_bus import event_bus

                await event_bus.publish(WALLET_STATE_CHANGED_EVENT, body)
            except Exception as exc:
                logger.debug("event_bus publish failed: %s", exc)

            # Cross-process bus (lazy import — wallet_state_bus imports
            # this module, so we must defer the import to break the cycle).
            try:
                from services import wallet_state_bus

                await wallet_state_bus.publish_delta(body)
            except Exception as exc:
                logger.debug("wallet_state_bus publish_delta failed: %s", exc)

        try:
            loop.create_task(_dispatch())
        except RuntimeError:
            return  # loop closed mid-call

    # -------------------- Configuration --------------------

    def configure_wallet(self, wallet_address: Optional[str]) -> None:
        """Pin the wallet this cache tracks.  Called once at startup.

        2026-05-05 hardening: refuse silent rebinds to a different wallet.
        On a single-wallet install (the only configuration we currently
        support) the cache is bound to either the proxy funder or the EOA.
        Live execution service has been observed to call this twice within
        seconds with different addresses when ``initialize()`` re-runs and
        ``_proxy_funder_address`` transiently flips between funder and
        ``None`` (e.g. signature-type probe race) — each rebind dropped the
        whole cache (positions / orders / fills / dedup set), forcing a
        full WS reseed and producing a freshness-stale window during which
        the orchestrator refused to trade.

        New behavior: keep the FIRST wallet seen and ignore subsequent
        attempts to bind to a different one. Operators can still force a
        rebind by calling :meth:`reset` first — which is the only path that
        clears the wallet pin AND the cached state in lockstep, with
        explicit intent. The previous behavior of "clear everything if the
        bind address looked different" is dangerous because the second
        bind is almost always a transient init-time fallback, not a real
        wallet swap. To detect a genuine wallet change (e.g. operator
        rotated keys), the orchestrator should be restarted, which calls
        ``reset_wallet_state_cache()`` and then re-binds cleanly.
        """
        with self._lock:
            normalized = str(wallet_address or "").strip().lower() or None
            if self._wallet_address is None:
                self._wallet_address = normalized
                return
            if normalized and normalized != self._wallet_address:
                # 2026-05-05: REFUSE the rebind. Log loudly so operators
                # can investigate why two wallets are racing for the cache,
                # but DO NOT clear the state — that destroys the freshness
                # signal the orchestrator depends on to trade.
                logger.warning(
                    "WalletStateCache rebind refused (already pinned); "
                    "ignoring new wallet to preserve cache state. "
                    "Restart the worker to switch wallets cleanly.",
                    pinned=self._wallet_address,
                    refused=normalized,
                )
                return

    # -------------------- WS push side --------------------

    def mark_ws_state(self, connected: bool) -> None:
        """Called by ``PolymarketUserFeed`` on connect/disconnect."""
        with self._lock:
            changed = bool(connected) != self._ws_connected
            self._ws_connected = bool(connected)
            self._ws_connected_since_mono = time.monotonic() if connected else None
        if changed:
            self._emit_change("ws_state", {"ws_connected": bool(connected)})

    def apply_trade(self, event: dict[str, Any]) -> None:
        """Apply a ``trade`` event from the user channel.

        Trades have a status lifecycle (MATCHED → MINED → CONFIRMED).
        Position math is applied once at ``MATCHED`` (deduplicated by
        ``id``); subsequent updates only refresh the fill record so the
        verification layer can see ``CONFIRMED`` status.
        """
        try:
            trade_id = str(event.get("id") or "").strip()
            if not trade_id:
                return
            owner = str(event.get("owner") or event.get("trade_owner") or "").strip().lower()
            wallet = self._wallet_address
            if wallet and owner and owner != wallet:
                # Defensive: should never happen because we authed for
                # this wallet, but the user channel can deliver events
                # for related accounts (proxy wallets) — drop strictly.
                return

            token_id = str(event.get("asset_id") or "").strip()
            condition_id = str(event.get("market") or "").strip()
            order_id = str(event.get("taker_order_id") or "").strip()
            side = str(event.get("side") or "").strip().upper()
            price = _safe_float(event.get("price"))
            size = _safe_float(event.get("size"))
            status = str(event.get("status") or "MATCHED").strip().upper()
            matched_at = _parse_event_timestamp(event.get("matchtime") or event.get("timestamp"))

            if side not in {"BUY", "SELL"}:
                return
            if not token_id or price <= 0 or size <= 0:
                return

            position_changed = False
            with self._lock:
                self._last_ws_event_mono = time.monotonic()
                # Dedup: only apply position math the first time we see
                # a trade.  Subsequent status updates (MINED/CONFIRMED)
                # update the existing fill record but don't re-bump
                # position size.
                already_applied = trade_id in self._applied_trade_id_set
                if status == "FAILED":
                    # Reverse a previously-applied fill.  This is rare
                    # but the protocol allows it (e.g. the trade was
                    # MATCHED but later RETRYING → FAILED on-chain).
                    self._reverse_fill_locked(trade_id)
                    position_changed = True
                elif not already_applied and status in {"MATCHED", "MINED", "CONFIRMED"}:
                    self._apply_fill_locked(
                        trade_id=trade_id,
                        order_id=order_id,
                        token_id=token_id,
                        condition_id=condition_id,
                        side=side,
                        price=price,
                        size=size,
                        status=status,
                        matched_at=matched_at,
                    )
                    position_changed = True
                # Update or insert the fill record for verification.
                fill = self._find_recent_fill_by_trade_id(trade_id)
                if fill is not None:
                    fill.status = status
                    if status == "CONFIRMED":
                        fill.confirmed_at = utcnow()
                self._trades_applied += 1
            # Emit AFTER releasing the lock so subscribers can read fresh
            # state without contention.  Only emit when position math
            # actually changed — pure status updates (MATCHED→MINED with
            # no size delta) are tracked by the verification layer
            # directly and don't need to wake the orchestrator.
            if position_changed:
                self._emit_change(
                    "trade",
                    {
                        "trade_id": trade_id,
                        "token_id": token_id,
                        "condition_id": condition_id,
                        "side": side,
                        "size": size,
                        "price": price,
                        "status": status,
                    },
                )
        except Exception as exc:
            logger.warning("WalletStateCache.apply_trade failed", exc_info=exc)

    def apply_order(self, event: dict[str, Any]) -> None:
        """Apply an ``order`` event from the user channel."""
        try:
            order_id = str(event.get("id") or "").strip()
            if not order_id:
                return
            owner = str(event.get("owner") or event.get("order_owner") or "").strip().lower()
            wallet = self._wallet_address
            if wallet and owner and owner != wallet:
                return

            token_id = str(event.get("asset_id") or "").strip()
            condition_id = str(event.get("market") or "").strip()
            side = str(event.get("side") or "").strip().upper()
            price = _safe_float(event.get("price"))
            original_size = _safe_float(event.get("original_size"))
            size_matched = _safe_float(event.get("size_matched"))
            order_type = str(event.get("type") or "PLACEMENT").strip().upper()
            timestamp = _parse_event_timestamp(event.get("timestamp"))

            if side not in {"BUY", "SELL"}:
                return
            if not token_id:
                return

            with self._lock:
                self._last_ws_event_mono = time.monotonic()
                existing = self._orders.get(order_id)
                if existing is None:
                    if len(self._orders) >= _MAX_TRACKED_ORDERS:
                        # Evict oldest terminal order to stay bounded.
                        self._evict_terminal_order_locked()
                    self._orders[order_id] = WalletOrder(
                        order_id=order_id,
                        token_id=token_id,
                        condition_id=condition_id,
                        side=side,  # type: ignore[arg-type]
                        price=price,
                        original_size=original_size,
                        size_matched=size_matched,
                        status=order_type,
                        placed_at=timestamp if order_type == "PLACEMENT" else None,
                        last_update=timestamp,
                    )
                else:
                    existing.size_matched = max(existing.size_matched, size_matched)
                    existing.status = order_type
                    existing.last_update = timestamp
                    if order_type == "PLACEMENT" and existing.placed_at is None:
                        existing.placed_at = timestamp
                self._orders_applied += 1
            self._emit_change(
                "order",
                {
                    "order_id": order_id,
                    "token_id": token_id,
                    "condition_id": condition_id,
                    "side": side,
                    "status": order_type,
                    "size_matched": size_matched,
                },
            )
        except Exception as exc:
            logger.warning("WalletStateCache.apply_order failed", exc_info=exc)

    # -------------------- REST seed side --------------------

    def seed_from_rest(
        self,
        *,
        wallet_address: str,
        positions: Iterable[dict[str, Any]],
        closed_positions: Iterable[dict[str, Any]],
        succeeded: bool = True,
    ) -> dict[str, int]:
        """Reseed positions from a REST snapshot.

        Called by ``trader_reconciliation_worker`` on bootstrap and
        every ~30s.  ``positions`` is the live open-positions list from
        ``polymarket_client.get_user_positions``; ``closed_positions``
        is the held-to-resolution list.

        Open positions overwrite the in-memory derivation (drift
        correction).  Closed positions update the ``is_resolved`` /
        ``settlement_price`` / ``redeemable`` fields on existing
        position rows.

        Returns counters for diagnostics.
        """
        self.configure_wallet(wallet_address)
        result = {"open_seeded": 0, "closed_seeded": 0, "removed_stale": 0}
        with self._lock:
            self._last_rest_seed_mono = time.monotonic()
            self._last_rest_seed_succeeded = bool(succeeded)
            self._rest_seed_count += 1

            if not succeeded:
                return result

            seen_token_ids: set[str] = set()
            for raw in positions or []:
                if not isinstance(raw, dict):
                    continue
                token_id = str(
                    raw.get("asset")
                    or raw.get("asset_id")
                    or raw.get("token_id")
                    or ""
                ).strip()
                if not token_id:
                    continue
                seen_token_ids.add(token_id)
                condition_id = str(raw.get("conditionId") or raw.get("condition_id") or "").strip()
                outcome_idx = _safe_int(raw.get("outcomeIndex"), 0)
                size = _safe_float(raw.get("size") or raw.get("positionSize") or raw.get("shares"))
                cost_basis = _safe_float(
                    raw.get("initialValue")
                    or (
                        _safe_float(raw.get("avgPrice") or raw.get("average_price"))
                        * size
                    )
                )
                mark_price = _safe_float(raw.get("curPrice") or raw.get("currentPrice"), default=0.0) or None
                redeemable = bool(raw.get("redeemable"))

                pos = self._positions.get(token_id)
                if pos is None:
                    pos = WalletPosition(
                        token_id=token_id,
                        condition_id=condition_id,
                        outcome_index=outcome_idx,
                    )
                    self._positions[token_id] = pos
                    if condition_id:
                        self._condition_to_token_ids.setdefault(condition_id, set()).add(token_id)
                # REST is authoritative for size/cost — overwrites WS-derived.
                pos.condition_id = condition_id or pos.condition_id
                pos.outcome_index = outcome_idx
                pos.size = size
                pos.cost_basis_usd = cost_basis
                pos.last_rest_mark_price = mark_price
                pos.redeemable = redeemable
                if condition_id:
                    self._condition_to_token_ids.setdefault(condition_id, set()).add(token_id)
                result["open_seeded"] += 1

            # Drop positions we used to track but Polymarket's REST no
            # longer reports — they were closed off-app or settled.
            stale = [
                tid for tid, pos in self._positions.items()
                if tid not in seen_token_ids and not pos.is_resolved and pos.size > 0.0
            ]
            for tid in stale:
                self._remove_position_locked(tid)
                result["removed_stale"] += 1

            for raw in closed_positions or []:
                if not isinstance(raw, dict):
                    continue
                token_id = str(
                    raw.get("asset")
                    or raw.get("asset_id")
                    or raw.get("token_id")
                    or ""
                ).strip()
                if not token_id:
                    continue
                pos = self._positions.get(token_id)
                if pos is None:
                    # Resolved-only position we never derived from WS.
                    condition_id = str(raw.get("conditionId") or raw.get("condition_id") or "").strip()
                    pos = WalletPosition(
                        token_id=token_id,
                        condition_id=condition_id,
                        outcome_index=_safe_int(raw.get("outcomeIndex"), 0),
                    )
                    self._positions[token_id] = pos
                    if condition_id:
                        self._condition_to_token_ids.setdefault(condition_id, set()).add(token_id)
                pos.is_resolved = True
                pos.redeemable = bool(raw.get("redeemable", True))
                cur = _safe_float(raw.get("curPrice") or raw.get("currentPrice"), default=0.0)
                pos.settlement_price = cur if cur >= 0 else None
                pos.last_rest_mark_price = cur or pos.last_rest_mark_price
                result["closed_seeded"] += 1

        # Emit after the lock releases.  Seed events are critical for
        # cross-process visibility — this is the moment the trader
        # orchestrator's freshness gate becomes valid.
        self._emit_change(
            "seed",
            {
                "succeeded": bool(succeeded),
                "open_seeded": result["open_seeded"],
                "closed_seeded": result["closed_seeded"],
                "removed_stale": result["removed_stale"],
            },
        )
        return result

    # -------------------- Hot-path read API --------------------

    def get_position(self, token_id: str) -> Optional[WalletPosition]:
        with self._lock:
            return self._positions.get(token_id)

    def get_position_size(self, token_id: str) -> float:
        with self._lock:
            pos = self._positions.get(token_id)
            return pos.size if pos is not None else 0.0

    def positions_by_token(self) -> dict[str, dict[str, Any]]:
        """Snapshot in the legacy dict shape, for the migration phase
        where ``_load_execution_wallet_positions_by_token`` callers
        haven't been refactored yet.
        """
        with self._lock:
            return {tid: pos.to_legacy_dict() for tid, pos in self._positions.items() if pos.size > 0.0}

    def closed_positions_by_token(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {
                tid: pos.to_legacy_dict()
                for tid, pos in self._positions.items()
                if pos.is_resolved
            }

    def to_token_inventory(self) -> dict[str, dict[str, Any]]:
        """Snapshot in the orchestrator's copy-trade inventory shape.

        Replaces the per-cycle ``list_live_wallet_positions_for_trader``
        DB hit. The mapping convention (outcome_index 0 = YES / buy_yes,
        1 = NO / buy_no) matches ``polymarket_trade_verifier
        ._direction_to_outcome_index``.
        """
        inventory: dict[str, dict[str, Any]] = {}
        with self._lock:
            for token_id, pos in self._positions.items():
                if pos.size <= 0.0:
                    continue
                key = str(token_id or "").strip().lower()
                if not key:
                    continue
                if pos.outcome_index == 1:
                    outcome = "NO"
                    direction = "buy_no"
                else:
                    outcome = "YES"
                    direction = "buy_yes"
                inventory[key] = {
                    "size": float(pos.size),
                    "market_id": pos.condition_id or "",
                    "outcome": outcome,
                    "direction": direction,
                    "current_price": pos.last_rest_mark_price,
                }
        return inventory

    def wallet_address(self) -> Optional[str]:
        with self._lock:
            return self._wallet_address

    def get_open_orders(self) -> list[WalletOrder]:
        with self._lock:
            return [o for o in self._orders.values() if not o.is_terminal]

    def get_orders_for_token(self, token_id: str) -> list[WalletOrder]:
        with self._lock:
            return [o for o in self._orders.values() if o.token_id == token_id]

    def get_order(self, order_id: str) -> Optional[WalletOrder]:
        with self._lock:
            return self._orders.get(order_id)

    def get_recent_fills_for_order(self, order_id: str) -> list[WalletFill]:
        with self._lock:
            return [f for f in self._recent_fills if f.order_id == order_id]

    def get_recent_fills_for_token(self, token_id: str, *, since_seconds: float = 600.0) -> list[WalletFill]:
        cutoff = utcnow().timestamp() - max(0.0, since_seconds)
        with self._lock:
            return [
                f for f in self._recent_fills
                if f.token_id == token_id and f.matched_at.timestamp() >= cutoff
            ]

    def iter_tracked_condition_ids(self) -> list[str]:
        """All condition_ids the WS feed should subscribe to.

        Per the user mandate (option a): subscribe to ALL condition_ids
        the bot has ever held a position in.  Bandwidth is cheap, churn
        is risky.
        """
        with self._lock:
            return sorted(self._condition_to_token_ids.keys())

    # -------------------- Freshness gate --------------------

    def is_fresh(
        self,
        *,
        max_event_age_seconds: float = _DEFAULT_FRESH_MAX_AGE_SECONDS,
        max_seed_age_seconds: float = _REST_SEED_MAX_AGE_SECONDS,
    ) -> tuple[bool, str]:
        """Hot-path freshness check.  Returns ``(fresh, reason)``.

        Composite check:
          * REST seed must have run at least once and be within
            ``max_seed_age_seconds`` (proves reconciliation worker is alive)
          * EITHER WS connected and an event arrived within
            ``max_event_age_seconds`` OR the wallet has no positions
            (idle wallets won't generate WS events — that's fine)
        """
        with self._lock:
            now_mono = time.monotonic()

            if self._last_rest_seed_mono is None:
                return False, "no_rest_seed_yet"
            if not self._last_rest_seed_succeeded:
                return False, "last_rest_seed_failed"
            seed_age = now_mono - self._last_rest_seed_mono
            if seed_age > max_seed_age_seconds:
                return False, f"rest_seed_stale:{seed_age:.0f}s"

            # Empty-position wallet: WS won't generate events; trust the
            # REST seed alone.
            has_positions = any(p.size > 0.0 for p in self._positions.values())
            if not has_positions:
                return True, "ok_idle"

            if not self._ws_connected:
                return False, "ws_disconnected"
            # If WS just connected, give it a brief grace period before
            # demanding events (no events on freshly-connected feed for
            # a quiet wallet is normal).
            if self._ws_connected_since_mono is not None:
                connected_for = now_mono - self._ws_connected_since_mono
                if connected_for < 5.0:
                    return True, "ok_ws_warming"

            # Otherwise we need a recent WS event OR a fresh seed
            # within max_event_age_seconds.  Either covers the wallet.
            last_event = self._last_ws_event_mono
            if last_event is not None and (now_mono - last_event) <= max_event_age_seconds:
                return True, "ok_ws_active"
            if seed_age <= max_event_age_seconds:
                return True, "ok_recent_seed"
            return True, "ok_idle_ws"

    # -------------------- Diagnostics --------------------

    def stats_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "wallet_address": self._wallet_address,
                "positions": len(self._positions),
                "open_positions": sum(1 for p in self._positions.values() if p.size > 0.0),
                "resolved_positions": sum(1 for p in self._positions.values() if p.is_resolved),
                "tracked_condition_ids": len(self._condition_to_token_ids),
                "open_orders": sum(1 for o in self._orders.values() if not o.is_terminal),
                "tracked_orders": len(self._orders),
                "recent_fills": len(self._recent_fills),
                "ws_connected": self._ws_connected,
                "ws_event_age_seconds": (
                    None if self._last_ws_event_mono is None
                    else round(time.monotonic() - self._last_ws_event_mono, 1)
                ),
                "rest_seed_age_seconds": (
                    None if self._last_rest_seed_mono is None
                    else round(time.monotonic() - self._last_rest_seed_mono, 1)
                ),
                "rest_seed_count": self._rest_seed_count,
                "trades_applied": self._trades_applied,
                "orders_applied": self._orders_applied,
                "duplicate_trades_skipped": self._duplicate_trades_skipped,
            }

    # -------------------- Internal --------------------

    def _apply_fill_locked(
        self,
        *,
        trade_id: str,
        order_id: str,
        token_id: str,
        condition_id: str,
        side: str,
        price: float,
        size: float,
        status: str,
        matched_at: datetime,
    ) -> None:
        # Position math.
        pos = self._positions.get(token_id)
        if pos is None:
            pos = WalletPosition(
                token_id=token_id,
                condition_id=condition_id,
                outcome_index=0,
            )
            self._positions[token_id] = pos
            if condition_id:
                self._condition_to_token_ids.setdefault(condition_id, set()).add(token_id)
        # Track condition mapping for any trade we've ever seen on this
        # token, even if REST hasn't seeded the condition_id yet.
        if condition_id and pos.condition_id != condition_id:
            pos.condition_id = condition_id
            self._condition_to_token_ids.setdefault(condition_id, set()).add(token_id)

        notional = price * size
        if side == "BUY":
            pos.size += size
            pos.cost_basis_usd += notional
        else:  # SELL
            # Reduce size; reduce cost basis proportionally (FIFO-like
            # average-cost reduction).  If the SELL exceeds current
            # size (shouldn't happen but defend), zero out.
            if pos.size > 0.0:
                avg = pos.cost_basis_usd / pos.size
                actual_reduction = min(size, pos.size)
                pos.size -= actual_reduction
                pos.cost_basis_usd -= avg * actual_reduction
                if pos.size <= 0.0:
                    pos.size = 0.0
                    pos.cost_basis_usd = 0.0
        pos.last_fill_at = matched_at

        # Track in fills deque.
        fill = WalletFill(
            trade_id=trade_id,
            order_id=order_id,
            token_id=token_id,
            condition_id=condition_id,
            side=side,  # type: ignore[arg-type]
            price=price,
            size=size,
            status=status,
            matched_at=matched_at,
        )
        self._recent_fills.append(fill)

        # Mark trade as applied for dedup.
        self._applied_trade_ids.append(trade_id)
        self._applied_trade_id_set.add(trade_id)
        # Bound the dedup set: pop the oldest applied trade IDs that
        # have rolled out of the deque.
        while len(self._applied_trade_id_set) > _MAX_RECENT_FILLS * 2:
            old = self._applied_trade_ids.popleft() if self._applied_trade_ids else None
            if old is not None:
                self._applied_trade_id_set.discard(old)

        # Remove fully-exited positions.
        if pos.size <= 0.0 and not pos.is_resolved:
            self._remove_position_locked(token_id)

    def _reverse_fill_locked(self, trade_id: str) -> None:
        if trade_id not in self._applied_trade_id_set:
            return
        fill = self._find_recent_fill_by_trade_id(trade_id)
        if fill is None:
            return
        pos = self._positions.get(fill.token_id)
        if pos is None:
            return
        notional = fill.price * fill.size
        if fill.side == "BUY":
            actual_reduction = min(fill.size, pos.size)
            avg = pos.cost_basis_usd / pos.size if pos.size > 0 else fill.price
            pos.size -= actual_reduction
            pos.cost_basis_usd -= avg * actual_reduction
        else:
            pos.size += fill.size
            pos.cost_basis_usd += notional
        if pos.size <= 0.0:
            pos.size = 0.0
            pos.cost_basis_usd = 0.0
        self._applied_trade_id_set.discard(trade_id)
        # Mark fill as failed but leave it in the deque for audit.
        fill.status = "FAILED"
        # Drop fully-reversed positions so the snapshot doesn't carry
        # zero-sized rows.  Resolved positions stay (their settlement
        # state is meaningful even at zero size).
        if pos.size <= 0.0 and not pos.is_resolved:
            self._remove_position_locked(fill.token_id)

    def _find_recent_fill_by_trade_id(self, trade_id: str) -> Optional[WalletFill]:
        for fill in self._recent_fills:
            if fill.trade_id == trade_id:
                return fill
        return None

    def _remove_position_locked(self, token_id: str) -> None:
        pos = self._positions.pop(token_id, None)
        if pos is None:
            return
        if pos.condition_id:
            condition_set = self._condition_to_token_ids.get(pos.condition_id)
            if condition_set is not None:
                condition_set.discard(token_id)
                if not condition_set:
                    self._condition_to_token_ids.pop(pos.condition_id, None)

    def _evict_terminal_order_locked(self) -> None:
        # Walk in insertion order; remove the first terminal order.
        for order_id, order in self._orders.items():
            if order.is_terminal:
                self._orders.pop(order_id, None)
                return
        # No terminal orders — drop the oldest by insertion to stay bounded.
        if self._orders:
            oldest_id = next(iter(self._orders))
            self._orders.pop(oldest_id, None)


# Singleton — every consumer uses the same instance.
_wallet_state_cache: WalletStateCache | None = None


def get_wallet_state_cache() -> WalletStateCache:
    """Return the process-wide singleton."""
    global _wallet_state_cache
    if _wallet_state_cache is None:
        _wallet_state_cache = WalletStateCache()
    return _wallet_state_cache


def reset_wallet_state_cache() -> None:
    """Test helper — drop the singleton."""
    global _wallet_state_cache
    _wallet_state_cache = None


