"""In-memory state cache for the trader orchestrator hot path.

Seeds from Postgres on startup, then maintained in-memory via inline
mutations on every order create / status-change / resolution.  All
reads on the critical path hit this module instead of the database.

A background write-behind task flushes audit records (decisions, checks,
signal consumption, cursor updates) to Postgres asynchronously so the
hot path never blocks on audit writes.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import and_, func, or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    AsyncSessionLocal,
    TradeSignal,
    TraderDecision,
    TraderDecisionCheck,
    TraderEvent,
    TraderOrder,
    TraderPosition,
    TraderSignalConsumption,
    TraderSignalCursor,
)
from services.event_bus import event_bus
from services.trader_orchestrator_state import (
    ACTIVE_POSITION_STATUS,
    REALIZED_LOSS_ORDER_STATUSES,
    REALIZED_ORDER_STATUSES,
    REALIZED_WIN_ORDER_STATUSES,
    _extract_live_fill_metrics,
    _is_active_order_status,
    _live_active_notional,
    _normalize_mode_key,
    _normalize_status_key,
    _position_cap_scope_key,
)
from utils.converters import safe_float
from utils.logger import get_logger
from utils.utcnow import utcnow

logger = get_logger("trader_hot_state")

_new_id = lambda: __import__("uuid").uuid4().hex  # noqa: E731
_HOT_UNFILLED_ORDER_STATUSES = {"pending", "placing", "submitted", "open", "working", "partial", "hedging", "partially_filled"}


# ── Lightweight signal proxy (hydrated from Redis stream snapshot) ─


class SignalSnapshot:
    """Drop-in replacement for a TradeSignal ORM row on the hot path.

    Constructed from the dict embedded in the Redis stream message so the
    orchestrator never needs to query Postgres for signal payloads.
    """

    __slots__ = (
        "id", "source", "source_item_id", "signal_type", "strategy_type",
        "market_id", "market_question", "direction", "entry_price",
        "effective_price", "edge_percent", "confidence", "liquidity",
        "expires_at", "status", "payload_json", "strategy_context_json",
        "quality_passed", "dedupe_key", "created_at", "updated_at",
    )

    def __init__(self, d: dict[str, Any]) -> None:
        self.id = str(d.get("id") or "")
        self.source = str(d.get("source") or "")
        self.source_item_id = str(d.get("source_item_id") or "")
        self.signal_type = str(d.get("signal_type") or "")
        self.strategy_type = str(d.get("strategy_type") or "")
        self.market_id = str(d.get("market_id") or "")
        self.market_question = str(d.get("market_question") or "")
        self.direction = str(d.get("direction") or "")
        self.entry_price = safe_float(d.get("entry_price"), 0.0)
        self.effective_price = safe_float(d.get("effective_price"), None)
        self.edge_percent = safe_float(d.get("edge_percent"), 0.0)
        self.confidence = safe_float(d.get("confidence"), 0.0)
        self.liquidity = safe_float(d.get("liquidity"), 0.0)
        raw_expires = d.get("expires_at")
        self.expires_at = _parse_iso_dt(raw_expires) if raw_expires else None
        self.status = str(d.get("status") or "pending")
        self.payload_json = d.get("payload_json") or {}
        self.strategy_context_json = d.get("strategy_context_json") or {}
        self.quality_passed = d.get("quality_passed")
        self.dedupe_key = str(d.get("dedupe_key") or "")
        raw_created = d.get("created_at")
        self.created_at = _parse_iso_dt(raw_created) if raw_created else None
        raw_updated = d.get("updated_at")
        self.updated_at = _parse_iso_dt(raw_updated) if raw_updated else None


def _parse_iso_dt(raw: Any) -> Optional[datetime]:
    if isinstance(raw, datetime):
        return raw
    s = str(raw or "").strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def hydrate_signals_from_stream(
    trigger_event: dict[str, Any],
    signal_ids: list[str],
) -> list[SignalSnapshot] | None:
    """Try to hydrate SignalSnapshot objects from embedded stream data.

    Returns None if the stream payload doesn't contain signal_snapshots
    (caller should fall back to the DB query).
    """
    snapshots = trigger_event.get("signal_snapshots")
    if not isinstance(snapshots, dict) or not snapshots:
        return None
    results: list[SignalSnapshot] = []
    for sid in signal_ids:
        snap = snapshots.get(sid)
        if not isinstance(snap, dict):
            return None  # incomplete — fall back to DB
        results.append(SignalSnapshot(snap))
    return results


# ── Per-trader snapshot ────────────────────────────────────────────


@dataclass
class _TraderSnapshot:
    """Mutable in-memory risk/position state for a single trader+mode."""

    open_position_keys: set[tuple[str, str, str]] = field(default_factory=set)
    open_order_keys: set[tuple[str, str, str]] = field(default_factory=set)
    open_order_ids: set[str] = field(default_factory=set)
    open_order_count: int = 0
    open_market_ids: set[str] = field(default_factory=set)
    gross_notional: float = 0.0
    market_notional: dict[str, float] = field(default_factory=dict)
    source_notional: dict[str, float] = field(default_factory=dict)
    copy_leader_notional: dict[str, float] = field(default_factory=dict)
    daily_realized_pnl: float = 0.0
    daily_pnl_date: str = ""
    consecutive_losses: int = 0
    last_loss_at: Optional[datetime] = None
    cursor_created_at: Optional[datetime] = None
    cursor_signal_id: Optional[str] = None
    cursor_runtime_sequence: Optional[int] = None
    # Active orders for unrealized PnL — keyed by order_id
    active_order_legs: dict[str, _ActiveLeg] = field(default_factory=dict)


@dataclass
class _ActiveLeg:
    """Minimal state needed for mark-to-market unrealized PnL."""

    token_id: str
    entry_price: float
    quantity: float
    direction: str
    notional: float


# ── Global state ───────────────────────────────────────────────────

# Keyed by (trader_id, mode)
_snapshots: dict[tuple[str, str], _TraderSnapshot] = {}
# Global gross exposure across all traders for a mode
_global_gross: dict[str, float] = {}
_global_daily_pnl: dict[str, float] = {}
_global_daily_pnl_date: dict[str, str] = {}
_seeded = False
_seed_lock = asyncio.Lock()


# ── Write-behind audit buffer ──────────────────────────────────────


@dataclass
class _AuditEntry:
    kind: str  # "decision" | "decision_checks" | "consumption" | "cursor" | "signal_status" | "trader_event" | "experiment_assignment"
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)


_audit_buffer: list[_AuditEntry] = []
_audit_lock = asyncio.Lock()
_audit_task: asyncio.Task | None = None
_AUDIT_FLUSH_INTERVAL = 0.5
_AUDIT_FLUSH_BATCH_SIZE = 1000
_AUDIT_BUFFER_MAX_SIZE = 50_000
_AUDIT_KIND_PRIORITY = {
    "decision": 0,
    "decision_checks": 1,
    "consumption": 2,
    "cursor": 3,
    "signal_status": 4,
    "trader_event": 5,
    "experiment_assignment": 6,
}

_AUDIT_OVERFLOW_LOGGED_AT: float = 0.0


def _audit_buffer_append(entry: _AuditEntry) -> None:
    """Append *entry* to audit buffer.  Caller must hold ``_audit_lock``.

    Drops the oldest entries when the buffer exceeds the cap to prevent
    unbounded memory growth during sustained DB failures.
    """
    global _AUDIT_OVERFLOW_LOGGED_AT
    _audit_buffer.append(entry)
    overflow = len(_audit_buffer) - _AUDIT_BUFFER_MAX_SIZE
    if overflow > 0:
        del _audit_buffer[:overflow]
        now = time.monotonic()
        if now - _AUDIT_OVERFLOW_LOGGED_AT > 30.0:
            _AUDIT_OVERFLOW_LOGGED_AT = now
            logger.warning(
                "Audit buffer at cap, dropped oldest entries",
                dropped=overflow,
                cap=_AUDIT_BUFFER_MAX_SIZE,
            )


_RESEED_INTERVAL_SECONDS = 120.0
_last_seed_at: float = 0.0


# ── Seeding ────────────────────────────────────────────────────────


async def seed(*, force: bool = False) -> None:
    """Load all trader state from Postgres into memory.  Safe to call
    multiple times — subsequent calls are no-ops unless *force* is set.
    Also starts the background audit flusher.
    """
    global _seeded, _last_seed_at
    async with _seed_lock:
        if _seeded and not force:
            return
        async with AsyncSessionLocal() as session:
            await _seed_from_db(session)
        _seeded = True
        _last_seed_at = time.monotonic()
        _start_audit_flusher()
    logger.info(
        "Hot state seeded",
        traders=len(_snapshots),
        global_modes=list(_global_gross.keys()),
    )


async def reseed_if_stale() -> bool:
    global _last_seed_at
    elapsed = time.monotonic() - _last_seed_at
    if elapsed < _RESEED_INTERVAL_SECONDS:
        return False
    async with _seed_lock:
        elapsed = time.monotonic() - _last_seed_at
        if elapsed < _RESEED_INTERVAL_SECONDS:
            return False
        async with AsyncSessionLocal() as session:
            await _seed_from_db(session)
        _last_seed_at = time.monotonic()
    logger.info(
        "Hot state reseeded",
        traders=len(_snapshots),
        elapsed_since_last=f"{elapsed:.1f}s",
    )
    return True


async def _seed_from_db(session: AsyncSession) -> None:
    _snapshots.clear()
    _global_gross.clear()
    _global_daily_pnl.clear()
    _global_daily_pnl_date.clear()

    today_str = utcnow().strftime("%Y-%m-%d")
    today_start = utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    # ── Orders (single pass) ───────────────────────────────────────
    order_status_key = func.lower(func.trim(func.coalesce(TraderOrder.status, "")))
    seed_active_statuses = tuple(_normalize_status_key(status) for status in ("submitted", "executed", "open"))
    seed_unfilled_statuses = tuple(_normalize_status_key(status) for status in _HOT_UNFILLED_ORDER_STATUSES)
    seed_realized_statuses = tuple(_normalize_status_key(status) for status in REALIZED_ORDER_STATUSES)
    order_rows = await session.stream(
        select(
            TraderOrder.id,
            TraderOrder.trader_id,
            TraderOrder.mode,
            TraderOrder.payload_json,
            TraderOrder.status,
            TraderOrder.notional_usd,
            TraderOrder.market_id,
            TraderOrder.direction,
            TraderOrder.source,
            TraderOrder.effective_price,
            TraderOrder.entry_price,
            TraderOrder.updated_at,
            TraderOrder.actual_profit,
        ).where(
            or_(
                order_status_key.in_(seed_active_statuses),
                and_(
                    order_status_key.in_(seed_realized_statuses),
                    TraderOrder.updated_at >= today_start,
                ),
            )
        )
    )
    async for order in order_rows:
        trader_id = str(order.trader_id or "")
        mode = _normalize_mode_key(order.mode)
        snap = _ensure_snapshot(trader_id, mode)
        payload = dict(order.payload_json or {})
        status_key = _normalize_status_key(order.status)
        row_notional = safe_float(order.notional_usd, 0.0) or 0.0
        active_notional = _live_active_notional(mode, order.status, row_notional, payload)

        if _is_active_order_status(mode, order.status):
            order_id = str(order.id or "").strip()
            if status_key in seed_unfilled_statuses and order_id:
                snap.open_order_ids.add(order_id)
                snap.open_order_count = len(snap.open_order_ids)
            market_id = str(order.market_id or "").strip()
            if market_id:
                snap.open_market_ids.add(market_id)
            for scope in ("market_direction",):
                key = _position_cap_scope_key(
                    position_cap_scope=scope,
                    mode=mode,
                    market_id=order.market_id,
                    direction=order.direction,
                    payload=payload,
                )
                if key is not None:
                    snap.open_order_keys.add(key)

            if active_notional > 0:
                snap.gross_notional += active_notional
                _global_gross[mode] = _global_gross.get(mode, 0.0) + active_notional
                if market_id:
                    snap.market_notional[market_id] = snap.market_notional.get(market_id, 0.0) + active_notional
                source_key = str(order.source or "").strip().lower()
                if source_key:
                    snap.source_notional[source_key] = snap.source_notional.get(source_key, 0.0) + active_notional
                # Copy leader wallet
                copy_wallet = _extract_copy_wallet(payload)
                if copy_wallet:
                    snap.copy_leader_notional[copy_wallet] = (
                        snap.copy_leader_notional.get(copy_wallet, 0.0) + active_notional
                    )

            # Build active leg for unrealized PnL
            _, filled_shares, fill_price = _extract_live_fill_metrics(payload)
            entry_price = (
                fill_price
                if fill_price is not None and fill_price > 0
                else safe_float(order.effective_price, 0.0) or safe_float(order.entry_price, 0.0) or 0.0
            )
            token_id = str(payload.get("token_id") or payload.get("selected_token_id") or "").strip()
            quantity = (
                float(filled_shares)
                if mode in {"live", "shadow"} and filled_shares > 0.0
                else float(row_notional / entry_price if entry_price > 0 else 0.0)
            )
            if token_id and entry_price > 0 and quantity > 0:
                snap.active_order_legs[str(order.id)] = _ActiveLeg(
                    token_id=token_id,
                    entry_price=entry_price,
                    quantity=quantity,
                    direction=str(order.direction or "").strip().lower(),
                    notional=active_notional,
                )

        # Realized PnL for today
        if status_key in REALIZED_ORDER_STATUSES:
            updated = order.updated_at
            if updated is not None:
                if updated.tzinfo is None:
                    updated = updated.replace(tzinfo=timezone.utc)
                if updated >= today_start:
                    profit = safe_float(order.actual_profit, 0.0) or 0.0
                    snap.daily_realized_pnl += profit
                    snap.daily_pnl_date = today_str
                    _global_daily_pnl[mode] = _global_daily_pnl.get(mode, 0.0) + profit
                    _global_daily_pnl_date[mode] = today_str

    # ── Positions (open only, for position cap scope) ──────────────
    pos_rows = (
        await session.execute(
            select(
                TraderPosition.trader_id,
                TraderPosition.mode,
                TraderPosition.market_id,
                TraderPosition.direction,
                TraderPosition.payload_json,
            ).where(func.lower(func.coalesce(TraderPosition.status, "")) == ACTIVE_POSITION_STATUS)
        )
    ).all()
    for row in pos_rows:
        trader_id = str(row.trader_id or "")
        mode = _normalize_mode_key(row.mode)
        snap = _ensure_snapshot(trader_id, mode)
        key = _position_cap_scope_key(
            position_cap_scope="market_direction",
            mode=mode,
            market_id=row.market_id,
            direction=row.direction,
            payload=row.payload_json,
        )
        if key is not None:
            snap.open_position_keys.add(key)
        market_id = str(row.market_id or "").strip()
        if market_id:
            snap.open_market_ids.add(market_id)
    # Snapshot index by trader
    snapshots_by_trader: dict[str, list[_TraderSnapshot]] = {}
    for (tid, _), snap in _snapshots.items():
        snapshots_by_trader.setdefault(tid, []).append(snap)
    trader_ids = set(snapshots_by_trader.keys())

    # Compute per-trader consecutive loss count from latest resolved outcomes.
    if trader_ids:
        ranked_realized = (
            select(
                TraderOrder.trader_id.label("trader_id"),
                TraderOrder.status.label("status"),
                TraderOrder.updated_at.label("updated_at"),
                func.row_number()
                .over(
                    partition_by=TraderOrder.trader_id,
                    order_by=(TraderOrder.updated_at.desc(), TraderOrder.id.desc()),
                )
                .label("rn"),
            )
            .where(TraderOrder.trader_id.in_(tuple(trader_ids)))
            .where(TraderOrder.status.in_(tuple(REALIZED_ORDER_STATUSES)))
            .subquery()
        )
        loss_rows = (
            await session.execute(
                select(
                    ranked_realized.c.trader_id,
                    ranked_realized.c.status,
                    ranked_realized.c.updated_at,
                )
                .where(ranked_realized.c.rn <= 100)
                .order_by(ranked_realized.c.trader_id.asc(), ranked_realized.c.rn.asc())
            )
        ).all()

        losses_by_trader: dict[str, tuple[int, Optional[datetime]]] = {}
        current_trader_id: str | None = None
        current_losses = 0
        current_last_loss: Optional[datetime] = None
        current_locked = False

        for row in loss_rows:
            trader_id = str(row.trader_id or "")
            if not trader_id:
                continue
            if trader_id != current_trader_id:
                if current_trader_id is not None:
                    losses_by_trader[current_trader_id] = (current_losses, current_last_loss)
                current_trader_id = trader_id
                current_losses = 0
                current_last_loss = None
                current_locked = False
            if current_locked:
                continue
            status_key = _normalize_status_key(row.status)
            if status_key in REALIZED_LOSS_ORDER_STATUSES:
                current_losses += 1
                if current_last_loss is None:
                    current_last_loss = row.updated_at
            elif status_key in REALIZED_WIN_ORDER_STATUSES:
                current_locked = True

        if current_trader_id is not None:
            losses_by_trader[current_trader_id] = (current_losses, current_last_loss)

        for trader_id, trader_snapshots in snapshots_by_trader.items():
            losses, last_loss = losses_by_trader.get(trader_id, (0, None))
            for snap in trader_snapshots:
                snap.consecutive_losses = losses
                snap.last_loss_at = last_loss


    cursor_rows = (
        await session.execute(
            select(
                TraderSignalCursor.trader_id,
                TraderSignalCursor.last_signal_created_at,
                TraderSignalCursor.last_signal_id,
                TraderSignalCursor.last_runtime_sequence,
            )
        )
    ).all()
    for cursor in cursor_rows:
        trader_id = str(cursor.trader_id or "")
        trader_snapshots = snapshots_by_trader.get(trader_id)
        if not trader_snapshots:
            continue
        cursor_signal_id = str(cursor.last_signal_id or "") or None
        for snap in trader_snapshots:
            snap.cursor_created_at = cursor.last_signal_created_at
            snap.cursor_signal_id = cursor_signal_id
            snap.cursor_runtime_sequence = int(cursor.last_runtime_sequence) if cursor.last_runtime_sequence is not None else None


def _ensure_snapshot(trader_id: str, mode: str) -> _TraderSnapshot:
    key = (trader_id, mode)
    snap = _snapshots.get(key)
    if snap is None:
        snap = _TraderSnapshot()
        _snapshots[key] = snap
    return snap


def _extract_copy_wallet(payload: dict[str, Any]) -> str:
    raw = payload.get("source_wallet") or payload.get("copy_source_wallet") or ""
    return str(raw).strip().lower()


# ── Hot reads (zero-cost) ─────────────────────────────────────────


def get_open_position_count(trader_id: str, mode: str, position_cap_scope: str = "market_direction") -> int:
    snap = _snapshots.get((trader_id, _normalize_mode_key(mode)))
    if snap is None:
        return 0
    return max(len(snap.open_position_keys), len(snap.open_order_keys))


def get_open_order_count(trader_id: str, mode: str) -> int:
    snap = _snapshots.get((trader_id, _normalize_mode_key(mode)))
    if snap is None:
        return 0
    snap.open_order_count = len(snap.open_order_ids)
    return snap.open_order_count


def get_open_market_ids(trader_id: str, mode: str) -> set[str]:
    snap = _snapshots.get((trader_id, _normalize_mode_key(mode)))
    if snap is None:
        return set()
    return set(snap.open_market_ids)


def get_gross_exposure(mode: str) -> float:
    return _global_gross.get(_normalize_mode_key(mode), 0.0)


def get_trader_gross_exposure(trader_id: str, mode: str) -> float:
    snap = _snapshots.get((trader_id, _normalize_mode_key(mode)))
    if snap is None:
        return 0.0
    return float(snap.gross_notional or 0.0)


def get_market_exposure(market_id: str, mode: str) -> float:
    mode_key = _normalize_mode_key(mode)
    total = 0.0
    for (_, m), snap in _snapshots.items():
        if m == mode_key:
            total += snap.market_notional.get(market_id, 0.0)
    return total


def get_trader_source_exposure(trader_id: str, source: str, mode: str) -> float:
    snap = _snapshots.get((trader_id, _normalize_mode_key(mode)))
    if snap is None:
        return 0.0
    return snap.source_notional.get(str(source or "").strip().lower(), 0.0)


def get_copy_leader_exposure(trader_id: str, source_wallet: str, mode: str) -> float:
    snap = _snapshots.get((trader_id, _normalize_mode_key(mode)))
    if snap is None:
        return 0.0
    normalized = str(source_wallet or "").strip().lower()
    return snap.copy_leader_notional.get(normalized, 0.0)


def get_daily_realized_pnl(trader_id: Optional[str], mode: str) -> float:
    mode_key = _normalize_mode_key(mode)
    today_str = utcnow().strftime("%Y-%m-%d")
    if trader_id is None:
        if _global_daily_pnl_date.get(mode_key) != today_str:
            return 0.0
        return _global_daily_pnl.get(mode_key, 0.0)
    snap = _snapshots.get((trader_id, mode_key))
    if snap is None or snap.daily_pnl_date != today_str:
        return 0.0
    return snap.daily_realized_pnl


def get_consecutive_loss_count(trader_id: str, mode: str) -> int:
    snap = _snapshots.get((trader_id, _normalize_mode_key(mode)))
    if snap is None:
        return 0
    return snap.consecutive_losses


def get_last_resolved_loss_at(trader_id: str, mode: str) -> Optional[datetime]:
    snap = _snapshots.get((trader_id, _normalize_mode_key(mode)))
    if snap is None:
        return None
    return snap.last_loss_at


def get_signal_cursor(trader_id: str, mode: str) -> tuple[Optional[datetime], Optional[str]]:
    snap = _snapshots.get((trader_id, _normalize_mode_key(mode)))
    if snap is None:
        return None, None
    return snap.cursor_created_at, snap.cursor_signal_id


def get_signal_sequence_cursor(trader_id: str, mode: str) -> int | None:
    snap = _snapshots.get((trader_id, _normalize_mode_key(mode)))
    if snap is None:
        return None
    return snap.cursor_runtime_sequence


async def get_unrealized_pnl(trader_id: Optional[str], mode: str) -> float:
    """Compute mark-to-market unrealized PnL from cached active legs + live prices."""
    from services.live_price_snapshot import get_live_mid_prices

    mode_key = _normalize_mode_key(mode)
    legs: list[tuple[str, _ActiveLeg]] = []
    token_ids: list[str] = []
    seen_tokens: set[str] = set()
    for (tid, m), snap in _snapshots.items():
        if m != mode_key:
            continue
        if trader_id is not None and tid != trader_id:
            continue
        for order_id, leg in snap.active_order_legs.items():
            legs.append((order_id, leg))
            if leg.token_id not in seen_tokens:
                seen_tokens.add(leg.token_id)
                token_ids.append(leg.token_id)

    if not legs:
        return 0.0

    try:
        prices = await get_live_mid_prices(token_ids)
    except Exception:
        prices = {}

    total = 0.0
    for _, leg in legs:
        current_price = prices.get(leg.token_id)
        if current_price is None or current_price <= 0:
            continue
        if leg.direction in ("sell", "no", "short"):
            total += leg.quantity * (leg.entry_price - current_price)
        else:
            total += leg.quantity * (current_price - leg.entry_price)
    return total


# ── Hot mutations (called inline after order operations) ──────────


def record_order_created(
    *,
    trader_id: str,
    mode: str,
    order_id: str,
    status: str,
    market_id: str,
    direction: str,
    source: str,
    notional_usd: float,
    entry_price: float,
    token_id: str,
    filled_shares: float,
    payload: dict[str, Any],
    copy_source_wallet: str = "",
) -> None:
    mode_key = _normalize_mode_key(mode)
    snap = _ensure_snapshot(trader_id, mode_key)
    active_notional = _live_active_notional(mode_key, "executed", notional_usd, payload)
    if active_notional <= 0:
        active_notional = abs(notional_usd)

    order_id_clean = str(order_id or "").strip()
    status_key = _normalize_status_key(status)
    if order_id_clean and status_key in _HOT_UNFILLED_ORDER_STATUSES:
        snap.open_order_ids.add(order_id_clean)
        snap.open_order_count = len(snap.open_order_ids)
    market_id_clean = str(market_id or "").strip()
    if market_id_clean:
        snap.open_market_ids.add(market_id_clean)

    key = _position_cap_scope_key(
        position_cap_scope="market_direction",
        mode=mode_key,
        market_id=market_id,
        direction=direction,
        payload=payload,
    )
    if key is not None:
        snap.open_order_keys.add(key)

    snap.gross_notional += active_notional
    _global_gross[mode_key] = _global_gross.get(mode_key, 0.0) + active_notional
    if market_id_clean:
        snap.market_notional[market_id_clean] = snap.market_notional.get(market_id_clean, 0.0) + active_notional
    source_key = str(source or "").strip().lower()
    if source_key:
        snap.source_notional[source_key] = snap.source_notional.get(source_key, 0.0) + active_notional
    wallet = (str(copy_source_wallet).strip().lower() if copy_source_wallet else _extract_copy_wallet(payload))
    if wallet:
        snap.copy_leader_notional[wallet] = snap.copy_leader_notional.get(wallet, 0.0) + active_notional

    quantity = (
        float(filled_shares) if filled_shares > 0 else (notional_usd / entry_price if entry_price > 0 else 0.0)
    )
    token_id_clean = str(token_id or "").strip()
    if token_id_clean and entry_price > 0 and quantity > 0:
        snap.active_order_legs[order_id_clean] = _ActiveLeg(
            token_id=token_id_clean,
            entry_price=entry_price,
            quantity=quantity,
            direction=str(direction or "").strip().lower(),
            notional=active_notional,
        )


def record_order_resolved(
    *,
    trader_id: str,
    mode: str,
    order_id: str,
    market_id: str,
    direction: str,
    source: str,
    status: str,
    actual_profit: float,
    payload: dict[str, Any],
    copy_source_wallet: str = "",
) -> None:
    mode_key = _normalize_mode_key(mode)
    snap = _snapshots.get((trader_id, mode_key))
    if snap is None:
        return

    # Remove active notional
    leg = snap.active_order_legs.pop(order_id, None)
    notional_to_remove = leg.notional if leg else 0.0

    status_key = _normalize_status_key(status)
    was_active_before = True  # assume it was active
    if str(order_id or "").strip():
        snap.open_order_ids.discard(str(order_id or "").strip())
        snap.open_order_count = len(snap.open_order_ids)

    if notional_to_remove > 0:
        snap.gross_notional = max(0.0, snap.gross_notional - notional_to_remove)
        _global_gross[mode_key] = max(0.0, _global_gross.get(mode_key, 0.0) - notional_to_remove)
        market_id_clean = str(market_id or "").strip()
        if market_id_clean:
            snap.market_notional[market_id_clean] = max(
                0.0, snap.market_notional.get(market_id_clean, 0.0) - notional_to_remove
            )
        source_key = str(source or "").strip().lower()
        if source_key:
            snap.source_notional[source_key] = max(
                0.0, snap.source_notional.get(source_key, 0.0) - notional_to_remove
            )
        if copy_source_wallet:
            wallet = str(copy_source_wallet).strip().lower()
            snap.copy_leader_notional[wallet] = max(
                0.0, snap.copy_leader_notional.get(wallet, 0.0) - notional_to_remove
            )

    # Update PnL / loss streak
    if status_key in REALIZED_ORDER_STATUSES:
        today_str = utcnow().strftime("%Y-%m-%d")
        profit = safe_float(actual_profit, 0.0) or 0.0
        if snap.daily_pnl_date != today_str:
            snap.daily_realized_pnl = 0.0
            snap.daily_pnl_date = today_str
        snap.daily_realized_pnl += profit
        if _global_daily_pnl_date.get(mode_key) != today_str:
            _global_daily_pnl[mode_key] = 0.0
            _global_daily_pnl_date[mode_key] = today_str
        _global_daily_pnl[mode_key] = _global_daily_pnl.get(mode_key, 0.0) + profit

        if status_key in REALIZED_LOSS_ORDER_STATUSES:
            snap.consecutive_losses += 1
            snap.last_loss_at = utcnow()
        elif status_key in REALIZED_WIN_ORDER_STATUSES:
            snap.consecutive_losses = 0


def record_order_cancelled(
    *,
    trader_id: str,
    mode: str,
    order_id: str,
    market_id: str,
    source: str,
    copy_source_wallet: str = "",
) -> None:
    mode_key = _normalize_mode_key(mode)
    snap = _snapshots.get((trader_id, mode_key))
    if snap is None:
        return

    leg = snap.active_order_legs.pop(order_id, None)
    notional_to_remove = leg.notional if leg else 0.0

    if str(order_id or "").strip():
        snap.open_order_ids.discard(str(order_id or "").strip())
        snap.open_order_count = len(snap.open_order_ids)

    if notional_to_remove > 0:
        snap.gross_notional = max(0.0, snap.gross_notional - notional_to_remove)
        _global_gross[mode_key] = max(0.0, _global_gross.get(mode_key, 0.0) - notional_to_remove)
        market_id_clean = str(market_id or "").strip()
        if market_id_clean:
            snap.market_notional[market_id_clean] = max(
                0.0, snap.market_notional.get(market_id_clean, 0.0) - notional_to_remove
            )
        source_key = str(source or "").strip().lower()
        if source_key:
            snap.source_notional[source_key] = max(
                0.0, snap.source_notional.get(source_key, 0.0) - notional_to_remove
            )
        if copy_source_wallet:
            wallet = str(copy_source_wallet).strip().lower()
            snap.copy_leader_notional[wallet] = max(
                0.0, snap.copy_leader_notional.get(wallet, 0.0) - notional_to_remove
            )


def update_signal_cursor(
    trader_id: str,
    mode: str,
    created_at: Optional[datetime],
    signal_id: Optional[str],
    runtime_sequence: Optional[int] = None,
) -> None:
    mode_key = _normalize_mode_key(mode)
    snap = _ensure_snapshot(trader_id, mode_key)
    snap.cursor_created_at = created_at
    snap.cursor_signal_id = str(signal_id or "") or None
    snap.cursor_runtime_sequence = int(runtime_sequence) if runtime_sequence is not None else snap.cursor_runtime_sequence


# ── Buffered audit writes ─────────────────────────────────────────


async def buffer_decision(
    *,
    trader_id: str,
    signal_id: str,
    signal_source: str,
    strategy_key: str,
    strategy_version: int | None,
    decision: str,
    reason: Optional[str],
    score: Optional[float],
    trace_id: Optional[str],
    checks_summary: Optional[dict[str, Any]],
    risk_snapshot: Optional[dict[str, Any]],
    payload: Optional[dict[str, Any]],
    decision_id: Optional[str] = None,
    publish: bool = True,
) -> str:
    row_id = decision_id or _new_id()
    entry = _AuditEntry(
        kind="decision",
        payload={
            "id": row_id,
            "trader_id": trader_id,
            "signal_id": signal_id,
            "source": signal_source,
            "strategy_key": strategy_key,
            "strategy_version": strategy_version,
            "decision": decision,
            "reason": reason,
            "score": score,
            "trace_id": trace_id,
            "checks_summary_json": checks_summary or {},
            "risk_snapshot_json": risk_snapshot or {},
            "payload_json": payload or {},
            "created_at": utcnow(),
        },
    )
    async with _audit_lock:
        _audit_buffer_append(entry)
    if publish:
        try:
            await event_bus.publish(
                "trader_decision",
                {
                    "id": row_id,
                    "trader_id": trader_id,
                    "signal_id": signal_id,
                    "source": signal_source,
                    "strategy_key": strategy_key,
                    "decision": decision,
                    "reason": reason,
                    "score": score,
                },
            )
        except Exception:
            pass
    return row_id


async def update_buffered_decision(
    *,
    decision_id: str,
    decision: str | None = None,
    reason: str | None = None,
    payload_patch: dict[str, Any] | None = None,
    checks_summary_patch: dict[str, Any] | None = None,
) -> bool:
    decision_key = str(decision_id or "").strip()
    if not decision_key:
        return False

    published_payload: dict[str, Any] | None = None
    async with _audit_lock:
        for entry in reversed(_audit_buffer):
            if entry.kind != "decision":
                continue
            payload = entry.payload if isinstance(entry.payload, dict) else {}
            if str(payload.get("id") or "").strip() != decision_key:
                continue
            if decision is not None:
                payload["decision"] = str(decision)
            if reason is not None:
                payload["reason"] = reason
            if payload_patch:
                merged_payload = dict(payload.get("payload_json") or {})
                merged_payload.update(payload_patch)
                payload["payload_json"] = merged_payload
            if checks_summary_patch:
                merged_summary = dict(payload.get("checks_summary_json") or {})
                merged_summary.update(checks_summary_patch)
                payload["checks_summary_json"] = merged_summary
            published_payload = dict(payload)
            break

    if published_payload is None:
        return False

    try:
        await event_bus.publish(
            "trader_decision",
            {
                "id": decision_key,
                "trader_id": published_payload.get("trader_id"),
                "signal_id": published_payload.get("signal_id"),
                "source": published_payload.get("source"),
                "strategy_key": published_payload.get("strategy_key"),
                "decision": published_payload.get("decision"),
                "reason": published_payload.get("reason"),
                "score": published_payload.get("score"),
            },
        )
    except Exception:
        pass

    return True


async def buffer_decision_checks(*, decision_id: str, checks: list[dict[str, Any]]) -> None:
    if not checks:
        return
    entry = _AuditEntry(
        kind="decision_checks",
        payload={"decision_id": decision_id, "checks": checks},
    )
    async with _audit_lock:
        _audit_buffer_append(entry)


async def buffer_signal_consumption(
    *,
    trader_id: str,
    signal_id: str,
    outcome: str,
    reason: Optional[str] = None,
    decision_id: Optional[str] = None,
    payload: Optional[dict[str, Any]] = None,
) -> None:
    entry = _AuditEntry(
        kind="consumption",
        payload={
            "trader_id": trader_id,
            "signal_id": signal_id,
            "outcome": outcome,
            "reason": reason,
            "decision_id": decision_id,
            "payload_json": payload or {},
            "consumed_at": utcnow(),
        },
    )
    async with _audit_lock:
        _audit_buffer_append(entry)


async def buffer_signal_cursor(
    *,
    trader_id: str,
    last_signal_created_at: Optional[datetime],
    last_signal_id: Optional[str],
    last_runtime_sequence: Optional[int],
) -> None:
    entry = _AuditEntry(
        kind="cursor",
        payload={
            "trader_id": trader_id,
            "last_signal_created_at": last_signal_created_at,
            "last_signal_id": last_signal_id,
            "last_runtime_sequence": last_runtime_sequence,
        },
    )
    async with _audit_lock:
        _audit_buffer_append(entry)


async def buffer_signal_status(*, signal_id: str, status: str, effective_price: float | None = None) -> None:
    entry = _AuditEntry(
        kind="signal_status",
        payload={"signal_id": signal_id, "status": status, "effective_price": effective_price},
    )
    async with _audit_lock:
        _audit_buffer_append(entry)


async def buffer_trader_event(
    *,
    event_type: str,
    severity: str = "info",
    trader_id: str | None = None,
    source: str | None = None,
    operator: str | None = None,
    message: str | None = None,
    trace_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> str:
    row_id = _new_id()
    entry = _AuditEntry(
        kind="trader_event",
        payload={
            "id": row_id,
            "trader_id": trader_id,
            "event_type": str(event_type or ""),
            "severity": str(severity or "info"),
            "source": source,
            "operator": operator,
            "message": message,
            "trace_id": trace_id,
            "payload_json": payload or {},
            "created_at": utcnow(),
        },
    )
    async with _audit_lock:
        _audit_buffer_append(entry)
    return row_id


async def buffer_experiment_assignment(
    *,
    experiment_id: str,
    trader_id: str | None,
    signal_id: str | None,
    source_key: str,
    strategy_key: str,
    strategy_version: int,
    assignment_group: str,
    decision_id: str | None = None,
    order_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    entry = _AuditEntry(
        kind="experiment_assignment",
        payload={
            "experiment_id": experiment_id,
            "trader_id": trader_id,
            "signal_id": signal_id,
            "source_key": source_key,
            "strategy_key": strategy_key,
            "strategy_version": int(strategy_version),
            "assignment_group": assignment_group,
            "decision_id": decision_id,
            "order_id": order_id,
            "payload_json": payload or {},
        },
    )
    async with _audit_lock:
        _audit_buffer_append(entry)


# ── Flush logic ────────────────────────────────────────────────────


def _start_audit_flusher() -> None:
    global _audit_task
    if _audit_task is not None and not _audit_task.done():
        return
    _audit_task = asyncio.create_task(_audit_flush_loop())


async def _audit_flush_loop() -> None:
    while True:
        await asyncio.sleep(_AUDIT_FLUSH_INTERVAL)
        try:
            await flush_audit_buffer()
        except Exception as exc:
            logger.warning("Audit flush failed", exc_info=exc)


async def flush_audit_buffer() -> int:
    """Drain buffered audit entries to Postgres.  Returns count flushed."""
    async with _audit_lock:
        if not _audit_buffer:
            return 0
        batch = _audit_buffer[:_AUDIT_FLUSH_BATCH_SIZE]
        del _audit_buffer[:len(batch)]
    batch.sort(key=lambda entry: (_AUDIT_KIND_PRIORITY.get(entry.kind, 99), entry.created_at))

    try:
        async with AsyncSessionLocal() as session:
            try:
                with session.no_autoflush:
                    for entry in batch:
                        if entry.kind == "decision":
                            await _flush_decision(session, entry.payload)
                        elif entry.kind == "decision_checks":
                            _flush_decision_checks(session, entry.payload)
                        elif entry.kind == "consumption":
                            await _flush_consumption(session, entry.payload)
                        elif entry.kind == "cursor":
                            await _flush_cursor(session, entry.payload)
                        elif entry.kind == "signal_status":
                            await _flush_signal_status(session, entry.payload)
                        elif entry.kind == "trader_event":
                            _flush_trader_event(session, entry.payload)
                        elif entry.kind == "experiment_assignment":
                            await _flush_experiment_assignment(session, entry.payload)
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    except Exception as exc:
        async with _audit_lock:
            headroom = _AUDIT_BUFFER_MAX_SIZE - len(_audit_buffer)
            if headroom > 0:
                preserved_batch = batch[:headroom]
                _audit_buffer[:0] = preserved_batch
                dropped = len(batch) - len(preserved_batch)
            else:
                dropped = len(batch)
        if dropped > 0:
            logger.warning(
                "Audit batch commit failed; buffer at cap, dropped entries",
                count=len(batch),
                dropped=dropped,
                buffer_size=len(_audit_buffer),
                exc_info=exc,
            )
        else:
            logger.warning("Audit batch commit failed, re-queuing", count=len(batch), exc_info=exc)
        return 0
    return len(batch)


async def _flush_decision(session: AsyncSession, p: dict[str, Any]) -> None:
    await session.execute(
        pg_insert(TraderDecision)
        .values(
            id=p["id"],
            trader_id=p["trader_id"],
            signal_id=p["signal_id"],
            source=p["source"],
            strategy_key=p["strategy_key"],
            strategy_version=p.get("strategy_version"),
            decision=p["decision"],
            reason=p.get("reason"),
            score=p.get("score"),
            trace_id=p.get("trace_id"),
            checks_summary_json=p.get("checks_summary_json") or {},
            risk_snapshot_json=p.get("risk_snapshot_json") or {},
            payload_json=p.get("payload_json") or {},
            created_at=p.get("created_at") or utcnow(),
        )
        .on_conflict_do_nothing(index_elements=[TraderDecision.id])
    )


def _flush_decision_checks(session: AsyncSession, p: dict[str, Any]) -> None:
    decision_id = p["decision_id"]
    for check in p.get("checks") or []:
        session.add(
            TraderDecisionCheck(
                id=_new_id(),
                decision_id=decision_id,
                check_key=str(check.get("check_key") or check.get("key") or "check"),
                check_label=str(check.get("check_label") or check.get("label") or "Check"),
                passed=bool(check.get("passed", False)),
                score=check.get("score"),
                detail=check.get("detail"),
                payload_json=check.get("payload") or {},
                created_at=utcnow(),
            )
        )


async def _flush_consumption(session: AsyncSession, p: dict[str, Any]) -> None:
    trader_id = p["trader_id"]
    signal_id = p["signal_id"]
    existing = (
        await session.execute(
            select(TraderSignalConsumption).where(
                TraderSignalConsumption.trader_id == trader_id,
                TraderSignalConsumption.signal_id == signal_id,
            )
        )
    ).scalars().first()
    if existing is not None:
        existing.decision_id = p.get("decision_id")
        existing.outcome = p["outcome"]
        existing.reason = p.get("reason")
        existing.payload_json = p.get("payload_json") or {}
        existing.consumed_at = p.get("consumed_at") or utcnow()
    else:
        session.add(
            TraderSignalConsumption(
                id=_new_id(),
                trader_id=trader_id,
                signal_id=signal_id,
                decision_id=p.get("decision_id"),
                outcome=p["outcome"],
                reason=p.get("reason"),
                payload_json=p.get("payload_json") or {},
                consumed_at=p.get("consumed_at") or utcnow(),
            )
        )


async def _flush_cursor(session: AsyncSession, p: dict[str, Any]) -> None:
    trader_id = p["trader_id"]
    row = await session.get(TraderSignalCursor, trader_id)
    if row is None:
        session.add(
            TraderSignalCursor(
                trader_id=trader_id,
                last_signal_created_at=p.get("last_signal_created_at"),
                last_signal_id=p.get("last_signal_id"),
                last_runtime_sequence=p.get("last_runtime_sequence"),
                updated_at=utcnow(),
            )
        )
    else:
        row.last_signal_created_at = p.get("last_signal_created_at")
        row.last_signal_id = str(p.get("last_signal_id") or "") or None
        row.last_runtime_sequence = p.get("last_runtime_sequence")
        row.updated_at = utcnow()


async def _flush_signal_status(session: AsyncSession, p: dict[str, Any]) -> None:
    signal = await session.get(TradeSignal, p["signal_id"])
    if signal is not None:
        signal.status = p["status"]
        if p.get("effective_price") is not None:
            signal.effective_price = p["effective_price"]
        signal.updated_at = utcnow()


def _flush_trader_event(session: AsyncSession, p: dict[str, Any]) -> None:
    session.add(
        TraderEvent(
            id=p["id"],
            trader_id=p.get("trader_id"),
            event_type=str(p.get("event_type") or ""),
            severity=str(p.get("severity") or "info"),
            source=p.get("source"),
            operator=p.get("operator"),
            message=p.get("message"),
            trace_id=p.get("trace_id"),
            payload_json=p.get("payload_json") or {},
            created_at=p.get("created_at") or utcnow(),
        )
    )


async def _flush_experiment_assignment(session: AsyncSession, p: dict[str, Any]) -> None:
    from services.strategy_experiments import upsert_strategy_experiment_assignment

    await upsert_strategy_experiment_assignment(
        session,
        experiment_id=str(p.get("experiment_id") or ""),
        trader_id=str(p.get("trader_id") or "") or None,
        signal_id=str(p.get("signal_id") or "") or None,
        source_key=str(p.get("source_key") or ""),
        strategy_key=str(p.get("strategy_key") or ""),
        strategy_version=int(p.get("strategy_version") or 1),
        assignment_group=str(p.get("assignment_group") or "control"),
        decision_id=str(p.get("decision_id") or "") or None,
        order_id=str(p.get("order_id") or "") or None,
        payload=dict(p.get("payload_json") or {}),
        commit=False,
    )
