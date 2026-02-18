"""Normalized trade signal bus helpers.

Worker pipelines emit into ``trade_signals`` and consumers (trader orchestrator/UI)
read from one normalized contract.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from utils.utcnow import utcnow
from typing import Any, Optional

from sqlalchemy import case, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    TradeSignal,
    TradeSignalEmission,
    TradeSignalSnapshot,
)
from models.opportunity import ArbitrageOpportunity
from services.market_tradability import get_market_tradability_map


SIGNAL_TERMINAL_STATUSES = {"executed", "skipped", "expired", "failed"}
SIGNAL_ACTIVE_STATUSES = {"pending", "selected", "submitted"}


def _utc_now() -> datetime:
    return utcnow()


def _to_utc_naive(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _safe_json(value: Any) -> Any:
    """Ensure payload is JSON-serializable."""
    if value is None:
        return None
    try:
        json.dumps(value, default=str)
        return value
    except Exception:
        return {"raw": str(value)}


async def _record_signal_emission(
    session: AsyncSession,
    row: TradeSignal,
    *,
    event_type: str,
    reason: str | None = None,
) -> None:
    snapshot = {
        "signal_id": row.id,
        "source": row.source,
        "source_item_id": row.source_item_id,
        "signal_type": row.signal_type,
        "strategy_type": row.strategy_type,
        "market_id": row.market_id,
        "direction": row.direction,
        "entry_price": row.entry_price,
        "effective_price": row.effective_price,
        "edge_percent": row.edge_percent,
        "confidence": row.confidence,
        "liquidity": row.liquidity,
        "status": row.status,
        "dedupe_key": row.dedupe_key,
        "expires_at": row.expires_at.isoformat() if row.expires_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }
    session.add(
        TradeSignalEmission(
            id=uuid.uuid4().hex,
            signal_id=row.id,
            source=str(row.source or ""),
            source_item_id=row.source_item_id,
            signal_type=str(row.signal_type or ""),
            strategy_type=row.strategy_type,
            market_id=str(row.market_id or ""),
            direction=row.direction,
            entry_price=row.entry_price,
            effective_price=row.effective_price,
            edge_percent=row.edge_percent,
            confidence=row.confidence,
            liquidity=row.liquidity,
            status=str(row.status or ""),
            dedupe_key=str(row.dedupe_key or ""),
            event_type=str(event_type),
            reason=reason,
            payload_json=_safe_json(row.payload_json),
            snapshot_json=snapshot,
            created_at=_utc_now(),
        )
    )


def make_dedupe_key(*parts: Any) -> str:
    packed = "|".join(str(p or "") for p in parts)
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()[:32]


def traders_signal_dedupe_key(channel: str, market_id: str) -> str:
    normalized_channel = str(channel or "").strip().lower() or "confluence"
    return make_dedupe_key(
        "traders",
        normalized_channel,
        str(market_id or "").strip().lower(),
    )


def tracked_trader_signal_dedupe_key(market_id: str) -> str:
    return traders_signal_dedupe_key("confluence", market_id)


async def upsert_trade_signal(
    session: AsyncSession,
    *,
    source: str,
    source_item_id: Optional[str],
    signal_type: str,
    strategy_type: Optional[str],
    market_id: str,
    market_question: Optional[str],
    direction: Optional[str],
    entry_price: Optional[float],
    edge_percent: Optional[float],
    confidence: Optional[float],
    liquidity: Optional[float],
    expires_at: Optional[datetime],
    payload_json: Optional[dict[str, Any]],
    strategy_context_json: Optional[dict[str, Any]] = None,
    dedupe_key: str,
    commit: bool = True,
) -> TradeSignal:
    """Idempotently upsert a normalized trade signal by ``(source, dedupe_key)``."""
    row: Optional[TradeSignal] = None

    # Prefer pending in-session rows first so we can avoid query-invoked
    # autoflush while still preserving same-transaction dedupe behavior.
    for pending in session.new:
        if not isinstance(pending, TradeSignal):
            continue
        if pending.source == source and pending.dedupe_key == dedupe_key:
            row = pending
            break

    if row is None:
        with session.no_autoflush:
            result = await session.execute(
                select(TradeSignal).where(
                    TradeSignal.source == source,
                    TradeSignal.dedupe_key == dedupe_key,
                )
            )
            row = result.scalar_one_or_none()

    if row is None:
        row = TradeSignal(
            id=uuid.uuid4().hex,
            source=source,
            source_item_id=source_item_id,
            signal_type=signal_type,
            strategy_type=strategy_type,
            market_id=market_id,
            market_question=market_question,
            direction=direction,
            entry_price=entry_price,
            edge_percent=edge_percent,
            confidence=confidence,
            liquidity=liquidity,
            expires_at=_to_utc_naive(expires_at),
            payload_json=_safe_json(payload_json),
            strategy_context_json=_safe_json(strategy_context_json),
            dedupe_key=dedupe_key,
            status="pending",
            created_at=_utc_now(),
            updated_at=_utc_now(),
        )
        session.add(row)
        await _record_signal_emission(
            session,
            row,
            event_type="upsert_insert",
        )
    else:
        # Terminal rows remain immutable to preserve auditability.
        if row.status in SIGNAL_ACTIVE_STATUSES:
            row.source_item_id = source_item_id
            row.signal_type = signal_type
            row.strategy_type = strategy_type
            row.market_id = market_id
            row.market_question = market_question
            row.direction = direction
            row.entry_price = entry_price
            row.edge_percent = edge_percent
            row.confidence = confidence
            row.liquidity = liquidity
            row.expires_at = _to_utc_naive(expires_at)
            row.payload_json = _safe_json(payload_json)
            row.strategy_context_json = _safe_json(strategy_context_json)
            row.updated_at = _utc_now()
            await _record_signal_emission(
                session,
                row,
                event_type="upsert_update",
            )
        else:
            await _record_signal_emission(
                session,
                row,
                event_type="upsert_ignored_terminal",
                reason=f"terminal_status:{row.status}",
            )

    if commit:
        await session.commit()
        await refresh_trade_signal_snapshots(session)
    return row


async def set_trade_signal_status(
    session: AsyncSession,
    signal_id: str,
    status: str,
    *,
    effective_price: Optional[float] = None,
    commit: bool = True,
) -> bool:
    result = await session.execute(select(TradeSignal).where(TradeSignal.id == signal_id))
    row = result.scalar_one_or_none()
    if row is None:
        return False
    row.status = status
    row.updated_at = _utc_now()
    if effective_price is not None:
        row.effective_price = effective_price
    await _record_signal_emission(
        session,
        row,
        event_type="status_update",
        reason=f"status:{status}",
    )
    if commit:
        await session.commit()
        await refresh_trade_signal_snapshots(session)
    return True


async def expire_stale_signals(session: AsyncSession, *, commit: bool = True) -> int:
    now = _utc_now()
    result = await session.execute(
        select(TradeSignal).where(
            TradeSignal.status.in_(tuple(SIGNAL_ACTIVE_STATUSES)),
            TradeSignal.expires_at.is_not(None),
            TradeSignal.expires_at < now,
        )
    )
    rows = list(result.scalars().all())
    for row in rows:
        row.status = "expired"
        row.updated_at = now
        await _record_signal_emission(
            session,
            row,
            event_type="status_expired",
            reason="expires_at_passed",
        )
    if rows and commit:
        await session.commit()
        await refresh_trade_signal_snapshots(session)
    elif commit:
        await session.commit()
    return len(rows)


async def expire_source_signals_except(
    session: AsyncSession,
    *,
    source: str,
    keep_dedupe_keys: set[str],
    signal_types: Optional[list[str]] = None,
    commit: bool = True,
) -> int:
    """Expire active source signals not present in the current emission set."""
    now = _utc_now()
    keep = {str(key) for key in keep_dedupe_keys if str(key).strip()}

    query = select(TradeSignal).where(
        TradeSignal.source == str(source),
        TradeSignal.status.in_(tuple(SIGNAL_ACTIVE_STATUSES)),
    )
    if signal_types:
        normalized_signal_types = [
            str(value or "").strip().lower() for value in signal_types if str(value or "").strip()
        ]
        if normalized_signal_types:
            query = query.where(func.lower(func.coalesce(TradeSignal.signal_type, "")).in_(normalized_signal_types))
    if keep:
        query = query.where(~TradeSignal.dedupe_key.in_(list(keep)))

    rows = list((await session.execute(query)).scalars().all())
    for row in rows:
        row.status = "expired"
        row.expires_at = now
        row.updated_at = now
        await _record_signal_emission(
            session,
            row,
            event_type="status_expired",
            reason="source_sweep",
        )

    if rows and commit:
        await session.commit()
        await refresh_trade_signal_snapshots(session)
    elif commit:
        await session.commit()

    return len(rows)


async def list_trade_signals(
    session: AsyncSession,
    *,
    source: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
) -> list[TradeSignal]:
    query = select(TradeSignal).order_by(TradeSignal.created_at.desc())
    if source:
        query = query.where(TradeSignal.source == source)
    if status:
        query = query.where(TradeSignal.status == status)
    query = query.offset(max(0, offset)).limit(max(1, min(limit, 1000)))
    result = await session.execute(query)
    rows = list(result.scalars().all())
    if status in {None, "pending", "selected", "submitted"}:
        rows = await _expire_non_tradable_pending_signals(session, rows)
    return rows


async def list_pending_trade_signals(
    session: AsyncSession,
    *,
    sources: Optional[list[str]] = None,
    limit: int = 500,
) -> list[TradeSignal]:
    now = _utc_now()
    query = (
        select(TradeSignal)
        .where(TradeSignal.status == "pending")
        .where(or_(TradeSignal.expires_at.is_(None), TradeSignal.expires_at >= now))
        .order_by(TradeSignal.created_at.asc())
        .limit(max(1, min(limit, 5000)))
    )
    if sources:
        query = query.where(TradeSignal.source.in_(sources))
    result = await session.execute(query)
    rows = list(result.scalars().all())
    rows = await _expire_non_tradable_pending_signals(session, rows)
    return rows


async def _expire_non_tradable_pending_signals(
    session: AsyncSession,
    rows: list[TradeSignal],
) -> list[TradeSignal]:
    if not rows:
        return rows

    pending_rows = [row for row in rows if row.status == "pending" and row.market_id]
    if not pending_rows:
        return rows

    tradability = await get_market_tradability_map([str(row.market_id) for row in pending_rows])
    now = _utc_now()
    changed = False
    output: list[TradeSignal] = []
    for row in rows:
        if row.status != "pending":
            output.append(row)
            continue
        market_key = str(row.market_id or "").strip().lower()
        if tradability.get(market_key, True):
            output.append(row)
            continue
        row.status = "expired"
        row.updated_at = now
        await _record_signal_emission(
            session,
            row,
            event_type="status_expired",
            reason="market_not_tradable",
        )
        changed = True

    if changed:
        await session.commit()
        await refresh_trade_signal_snapshots(session)

    return output


async def refresh_trade_signal_snapshots(session: AsyncSession) -> list[dict[str, Any]]:
    """Recompute per-source snapshot rows from ``trade_signals``."""

    rows = (
        await session.execute(
            select(
                TradeSignal.source,
                TradeSignal.status,
                func.count(TradeSignal.id),
                func.max(TradeSignal.created_at),
                func.min(
                    case(
                        (TradeSignal.status == "pending", TradeSignal.created_at),
                        else_=None,
                    )
                ),
            ).group_by(TradeSignal.source, TradeSignal.status)
        )
    ).all()

    source_stats: dict[str, dict[str, Any]] = {}
    for source, status, count, latest_created, oldest_pending in rows:
        stats = source_stats.setdefault(
            source,
            {
                "source": source,
                "pending_count": 0,
                "selected_count": 0,
                "submitted_count": 0,
                "executed_count": 0,
                "skipped_count": 0,
                "expired_count": 0,
                "failed_count": 0,
                "latest_signal_at": None,
                "oldest_pending_at": None,
            },
        )
        key = f"{status}_count"
        if key in stats:
            stats[key] = int(count or 0)
        if latest_created and (stats["latest_signal_at"] is None or latest_created > stats["latest_signal_at"]):
            stats["latest_signal_at"] = latest_created
        if oldest_pending and (stats["oldest_pending_at"] is None or oldest_pending < stats["oldest_pending_at"]):
            stats["oldest_pending_at"] = oldest_pending

    now = _utc_now()
    existing_rows = (await session.execute(select(TradeSignalSnapshot))).scalars().all()
    existing_by_source = {r.source: r for r in existing_rows}

    for source, stats in source_stats.items():
        row = existing_by_source.get(source)
        if row is None:
            row = TradeSignalSnapshot(source=source)
            session.add(row)

        for key in (
            "pending_count",
            "selected_count",
            "submitted_count",
            "executed_count",
            "skipped_count",
            "expired_count",
            "failed_count",
        ):
            setattr(row, key, int(stats.get(key, 0) or 0))

        row.latest_signal_at = stats.get("latest_signal_at")
        row.oldest_pending_at = stats.get("oldest_pending_at")
        row.freshness_seconds = (
            (now - stats["latest_signal_at"]).total_seconds() if stats.get("latest_signal_at") else None
        )
        row.updated_at = now
        row.stats_json = {
            "total": sum(
                int(stats.get(k, 0) or 0)
                for k in (
                    "pending_count",
                    "selected_count",
                    "submitted_count",
                    "executed_count",
                    "skipped_count",
                    "expired_count",
                    "failed_count",
                )
            )
        }

    await session.commit()

    out: list[dict[str, Any]] = []
    snapshot_rows = (
        (await session.execute(select(TradeSignalSnapshot).order_by(TradeSignalSnapshot.source.asc()))).scalars().all()
    )
    for row in snapshot_rows:
        out.append(
            {
                "source": row.source,
                "pending_count": int(row.pending_count or 0),
                "selected_count": int(row.selected_count or 0),
                "submitted_count": int(row.submitted_count or 0),
                "executed_count": int(row.executed_count or 0),
                "skipped_count": int(row.skipped_count or 0),
                "expired_count": int(row.expired_count or 0),
                "failed_count": int(row.failed_count or 0),
                "latest_signal_at": row.latest_signal_at,
                "oldest_pending_at": row.oldest_pending_at,
                "freshness_seconds": row.freshness_seconds,
                "updated_at": row.updated_at,
            }
        )
    return out


def _direction_from_outcome(outcome: Optional[str]) -> Optional[str]:
    val = (outcome or "").lower().strip()
    if val in {"yes", "buy_yes"}:
        return "buy_yes"
    if val in {"no", "buy_no"}:
        return "buy_no"
    return None


async def emit_scanner_signals(
    session: AsyncSession,
    opportunities: list[ArbitrageOpportunity],
    *,
    default_ttl_minutes: int = 120,
) -> int:
    emitted = 0
    for opp in opportunities:
        market = (opp.markets or [{}])[0]
        position = (opp.positions_to_take or [{}])[0]
        market_id = str(market.get("id") or opp.event_id or opp.id)
        if not market_id:
            continue

        dedupe_key = make_dedupe_key(
            opp.stable_id,
            opp.strategy,
            market_id,
            round(float(opp.roi_percent or 0.0), 4),
        )
        expires = opp.resolution_date or (_utc_now() + timedelta(minutes=default_ttl_minutes))

        await upsert_trade_signal(
            session,
            source="scanner",
            source_item_id=opp.stable_id,
            signal_type="scanner_opportunity",
            strategy_type=opp.strategy,
            market_id=market_id,
            market_question=market.get("question") or opp.title,
            direction=_direction_from_outcome(position.get("outcome")),
            entry_price=position.get("price"),
            edge_percent=float(opp.roi_percent or 0.0),
            confidence=float(opp.confidence),
            liquidity=float(opp.min_liquidity or 0.0),
            expires_at=expires,
            payload_json=opp.model_dump(mode="json"),
            strategy_context_json=opp.strategy_context or None,
            dedupe_key=dedupe_key,
            commit=False,
        )
        emitted += 1

    await session.commit()
    await refresh_trade_signal_snapshots(session)
    return emitted


