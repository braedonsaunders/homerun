"""Normalized trade signal bus helpers.

Worker pipelines emit into ``trade_signals`` and consumers (trader orchestrator/UI)
read from one normalized contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from datetime import datetime, timedelta, timezone
from utils.utcnow import utcnow
from typing import Any, Optional

from sqlalchemy import case, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import TradeSignal, TradeSignalSnapshot
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


def make_dedupe_key(*parts: Any) -> str:
    packed = "|".join(str(p or "") for p in parts)
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()[:32]


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
            dedupe_key=dedupe_key,
            status="pending",
            created_at=_utc_now(),
            updated_at=_utc_now(),
        )
        session.add(row)
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
            row.updated_at = _utc_now()

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
        changed = True

    if changed:
        await session.commit()
        await refresh_trade_signal_snapshots(session)

    return output


async def refresh_trade_signal_snapshots(session: AsyncSession) -> list[dict[str, Any]]:
    """Recompute per-source snapshot rows from ``trade_signals``."""

    rows = (
        (
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
                )
                .group_by(TradeSignal.source, TradeSignal.status)
            )
        )
        .all()
    )

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
        if latest_created and (
            stats["latest_signal_at"] is None or latest_created > stats["latest_signal_at"]
        ):
            stats["latest_signal_at"] = latest_created
        if oldest_pending and (
            stats["oldest_pending_at"] is None
            or oldest_pending < stats["oldest_pending_at"]
        ):
            stats["oldest_pending_at"] = oldest_pending

    now = _utc_now()
    existing_rows = (
        (await session.execute(select(TradeSignalSnapshot))).scalars().all()
    )
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
            (now - stats["latest_signal_at"]).total_seconds()
            if stats.get("latest_signal_at")
            else None
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
        (await session.execute(select(TradeSignalSnapshot).order_by(TradeSignalSnapshot.source.asc())))
        .scalars()
        .all()
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
            confidence=max(0.0, min(1.0, 1.0 - float(opp.risk_score or 0.5))),
            liquidity=float(opp.min_liquidity or 0.0),
            expires_at=expires,
            payload_json=opp.model_dump(mode="json"),
            dedupe_key=dedupe_key,
            commit=False,
        )
        emitted += 1

    await session.commit()
    await refresh_trade_signal_snapshots(session)
    return emitted


async def emit_news_intent_signals(
    session: AsyncSession,
    intents: list[Any],
    *,
    max_age_minutes: int = 120,
) -> int:
    emitted = 0
    ttl = timedelta(minutes=max(1, max_age_minutes))
    for intent in intents:
        created_at = _to_utc_naive(getattr(intent, "created_at", None)) or _utc_now()
        expires = created_at + ttl
        dedupe_key = getattr(intent, "signal_key", None) or make_dedupe_key(intent.id)
        metadata = getattr(intent, "metadata_json", None) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        market_metadata = metadata.get("market") if isinstance(metadata.get("market"), dict) else {}
        finding_metadata = metadata.get("finding") if isinstance(metadata.get("finding"), dict) else {}

        await upsert_trade_signal(
            session,
            source="news",
            source_item_id=str(intent.id),
            signal_type="news_intent",
            strategy_type="news_edge",
            market_id=str(intent.market_id),
            market_question=intent.market_question,
            direction=getattr(intent, "direction", None),
            entry_price=getattr(intent, "entry_price", None),
            edge_percent=getattr(intent, "edge_percent", None),
            confidence=getattr(intent, "confidence", None),
            liquidity=market_metadata.get("liquidity"),
            expires_at=expires,
            payload_json={
                "intent_id": intent.id,
                "finding_id": getattr(intent, "finding_id", None),
                "metadata": metadata,
                "reasoning": finding_metadata.get("reasoning"),
                "evidence": finding_metadata.get("evidence"),
            },
            dedupe_key=dedupe_key,
            commit=False,
        )
        emitted += 1

    await session.commit()
    await refresh_trade_signal_snapshots(session)
    return emitted


async def emit_weather_intent_signals(
    session: AsyncSession,
    intents: list[Any],
    *,
    max_age_minutes: int = 240,
) -> int:
    emitted = 0
    ttl = timedelta(minutes=max(1, max_age_minutes))
    for intent in intents:
        created_at = _to_utc_naive(getattr(intent, "created_at", None)) or _utc_now()
        expires = created_at + ttl
        dedupe_key = make_dedupe_key(
            intent.market_id,
            intent.direction,
            round(float(getattr(intent, "edge_percent", 0.0) or 0.0), 3),
            created_at.isoformat(),
        )

        await upsert_trade_signal(
            session,
            source="weather",
            source_item_id=str(intent.id),
            signal_type="weather_intent",
            strategy_type="weather_edge",
            market_id=str(intent.market_id),
            market_question=intent.market_question,
            direction=getattr(intent, "direction", None),
            entry_price=getattr(intent, "entry_price", None),
            edge_percent=getattr(intent, "edge_percent", None),
            confidence=getattr(intent, "confidence", None),
            liquidity=(getattr(intent, "metadata_json", {}) or {}).get("liquidity"),
            expires_at=expires,
            payload_json={
                "intent_id": intent.id,
                "metadata": getattr(intent, "metadata_json", None),
                "take_profit_price": getattr(intent, "take_profit_price", None),
                "stop_loss_pct": getattr(intent, "stop_loss_pct", None),
            },
            dedupe_key=dedupe_key,
            commit=False,
        )
        emitted += 1

    await session.commit()
    await refresh_trade_signal_snapshots(session)
    return emitted


async def emit_insider_intent_signals(
    session: AsyncSession,
    intents: list[Any],
    *,
    max_age_minutes: int = 180,
) -> int:
    emitted = 0
    ttl = timedelta(minutes=max(1, max_age_minutes))
    for intent in intents:
        created_at = _to_utc_naive(getattr(intent, "created_at", None)) or _utc_now()
        expires = created_at + ttl
        dedupe_key = getattr(intent, "signal_key", None) or make_dedupe_key(intent.id)
        metadata = getattr(intent, "metadata_json", {}) or {}
        wallet_addresses = getattr(intent, "wallet_addresses_json", None) or []

        await upsert_trade_signal(
            session,
            source="insider",
            source_item_id=str(intent.id),
            signal_type="insider_intent",
            strategy_type="insider_edge",
            market_id=str(intent.market_id),
            market_question=intent.market_question,
            direction=getattr(intent, "direction", None),
            entry_price=getattr(intent, "entry_price", None),
            edge_percent=getattr(intent, "edge_percent", None),
            confidence=getattr(intent, "confidence", None),
            liquidity=metadata.get("market_liquidity"),
            expires_at=expires,
            payload_json={
                "intent_id": intent.id,
                "insider_score": getattr(intent, "insider_score", None),
                "wallet_addresses": wallet_addresses,
                "metadata": metadata,
            },
            dedupe_key=dedupe_key,
            commit=False,
        )
        emitted += 1

    await session.commit()
    await refresh_trade_signal_snapshots(session)
    return emitted


async def emit_tracked_trader_signals(
    session: AsyncSession,
    confluence_signals: list[dict[str, Any]],
) -> int:
    emitted = 0
    for sig in confluence_signals:
        tier = str(sig.get("tier") or "WATCH").upper()
        if tier not in {"HIGH", "EXTREME"}:
            continue

        market_id = str(sig.get("market_id") or "")
        if not market_id:
            continue

        detected_at = sig.get("detected_at")
        if isinstance(detected_at, str):
            try:
                detected_dt = datetime.fromisoformat(detected_at.replace("Z", "+00:00"))
                if detected_dt.tzinfo is not None:
                    detected_dt = detected_dt.astimezone(timezone.utc).replace(tzinfo=None)
            except Exception:
                detected_dt = _utc_now()
        elif isinstance(detected_at, datetime):
            detected_dt = _to_utc_naive(detected_at) or _utc_now()
        else:
            detected_dt = _utc_now()

        expires_at = detected_dt + timedelta(minutes=max(30, int(sig.get("window_minutes") or 60)))

        outcome = str(sig.get("outcome") or "").upper()
        direction = "buy_yes" if outcome == "YES" else "buy_no" if outcome == "NO" else None

        dedupe_key = make_dedupe_key(
            sig.get("id") or "",
            market_id,
            outcome,
            round(float(sig.get("conviction_score") or 0.0), 2),
        )

        await upsert_trade_signal(
            session,
            source="tracked_traders",
            source_item_id=str(sig.get("id") or ""),
            signal_type="confluence",
            strategy_type="tracked_trader_confluence",
            market_id=market_id,
            market_question=sig.get("market_question"),
            direction=direction,
            entry_price=sig.get("avg_entry_price"),
            edge_percent=float(sig.get("conviction_score") or 0.0),
            confidence=max(0.0, min(1.0, float(sig.get("strength") or 0.0))),
            liquidity=sig.get("market_liquidity"),
            expires_at=expires_at,
            payload_json=sig,
            dedupe_key=dedupe_key,
            commit=False,
        )
        emitted += 1

    await session.commit()
    await refresh_trade_signal_snapshots(session)
    return emitted


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return default


def _timeframe_seconds(value: Any) -> int:
    tf = str(value or "").strip().lower()
    if tf in {"5m", "5min"}:
        return 300
    if tf in {"15m", "15min"}:
        return 900
    if tf in {"1h", "1hr", "60m"}:
        return 3600
    if tf in {"4h", "4hr", "240m"}:
        return 14400
    return 900


def _parse_end_time(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    return _to_utc_naive(parsed)


def _crypto_regime(seconds_left: float, timeframe_seconds: int) -> str:
    denom = float(max(1, timeframe_seconds))
    ratio = _clamp(seconds_left / denom, 0.0, 1.0)
    if ratio > 0.67:
        return "opening"
    if ratio < 0.33:
        return "closing"
    return "mid"


def _regime_weights(regime: str) -> dict[str, float]:
    if regime == "opening":
        return {"directional": 0.65, "pure_arb": 0.25, "rebalance": 0.10}
    if regime == "closing":
        return {"directional": 0.35, "pure_arb": 0.20, "rebalance": 0.45}
    return {"directional": 0.50, "pure_arb": 0.25, "rebalance": 0.25}


def _regime_weights_without_oracle(regime: str) -> dict[str, float]:
    if regime == "opening":
        return {"directional": 0.0, "pure_arb": 0.60, "rebalance": 0.40}
    if regime == "closing":
        return {"directional": 0.0, "pure_arb": 0.45, "rebalance": 0.55}
    return {"directional": 0.0, "pure_arb": 0.55, "rebalance": 0.45}


async def emit_crypto_market_signals(
    session: AsyncSession,
    markets: list[dict[str, Any]],
) -> int:
    """Emit dedicated crypto-worker 15m-class multi-strategy signals."""
    emitted = 0
    now = _utc_now()

    for market in markets:
        market_id = str(market.get("condition_id") or market.get("id") or "")
        if not market_id:
            continue

        price_to_beat = market.get("price_to_beat")
        oracle_price = market.get("oracle_price")
        up_price = market.get("up_price")
        down_price = market.get("down_price")

        if up_price is None or down_price is None:
            continue

        ptb = _to_float(price_to_beat)
        oracle = _to_float(oracle_price)
        up = _to_float(up_price)
        down = _to_float(down_price)
        if up is None or down is None:
            continue
        if not (0.0 <= up <= 1.0 and 0.0 <= down <= 1.0):
            continue
        has_oracle = ptb is not None and ptb > 0 and oracle is not None

        timeframe_seconds = _timeframe_seconds(market.get("timeframe"))
        parsed_end_time = _parse_end_time(market.get("end_time"))
        seconds_left = _to_float(market.get("seconds_left"))
        if seconds_left is None:
            if parsed_end_time is not None:
                seconds_left = max(0.0, (parsed_end_time - now).total_seconds())
            else:
                seconds_left = float(timeframe_seconds)
        regime = _crypto_regime(seconds_left, timeframe_seconds)

        if has_oracle and ptb is not None and oracle is not None:
            diff_pct = ((oracle - ptb) / ptb) * 100.0
            time_ratio = _clamp(seconds_left / float(max(1, timeframe_seconds)), 0.08, 1.0)
            directional_scale = max(0.08, 0.50 * time_ratio)
            directional_z = _clamp(diff_pct / directional_scale, -60.0, 60.0)
            model_prob_yes = _clamp(
                1.0 / (1.0 + math.exp(-directional_z)),
                0.03,
                0.97,
            )
            model_prob_no = 1.0 - model_prob_yes
            directional_yes = max(0.0, (model_prob_yes - up) * 100.0)
            directional_no = max(0.0, (model_prob_no - down) * 100.0)
        else:
            diff_pct = 0.0
            model_prob_yes = 0.5
            model_prob_no = 0.5
            directional_yes = 0.0
            directional_no = 0.0

        combined = up + down
        underround = max(0.0, 1.0 - combined)
        pure_arb_yes = underround * 100.0
        pure_arb_no = underround * 100.0

        neutrality = _clamp(1.0 - (abs(diff_pct) / 0.45), 0.0, 1.0)
        rebalance_yes = max(0.0, (0.5 - up) * 100.0) * neutrality
        rebalance_no = max(0.0, (0.5 - down) * 100.0) * neutrality

        weights = _regime_weights(regime) if has_oracle else _regime_weights_without_oracle(regime)
        gross_yes = (
            (directional_yes * weights["directional"])
            + (pure_arb_yes * weights["pure_arb"])
            + (rebalance_yes * weights["rebalance"])
        )
        gross_no = (
            (directional_no * weights["directional"])
            + (pure_arb_no * weights["pure_arb"])
            + (rebalance_no * weights["rebalance"])
        )

        spread = _to_float(market.get("spread"), 0.0) or 0.0
        spread = _clamp(spread, 0.0, 0.10)
        liquidity = max(0.0, _to_float(market.get("liquidity"), 0.0) or 0.0)
        fees_enabled = bool(market.get("fees_enabled", False))

        fee_penalty = 0.45 if fees_enabled else 0.25
        spread_penalty = spread * 100.0 * 0.35
        liquidity_scale = _clamp(liquidity / 250000.0, 0.0, 1.0)
        regime_slippage_factor = 1.1 if regime == "closing" else 1.0
        slippage_penalty = (1.35 - (0.95 * liquidity_scale)) * regime_slippage_factor
        execution_penalty = fee_penalty + spread_penalty + slippage_penalty

        net_yes = gross_yes - execution_penalty
        net_no = gross_no - execution_penalty
        direction = "buy_yes" if net_yes >= net_no else "buy_no"
        entry_price = up if direction == "buy_yes" else down
        edge_percent = net_yes if direction == "buy_yes" else net_no

        if edge_percent < 1.0:
            continue

        selected_components = (
            {
                "directional": directional_yes,
                "pure_arb": pure_arb_yes,
                "rebalance": rebalance_yes,
            }
            if direction == "buy_yes"
            else {
                "directional": directional_no,
                "pure_arb": pure_arb_no,
                "rebalance": rebalance_no,
            }
        )
        weighted_components = {
            key: selected_components[key] * weights[key] for key in selected_components
        }
        dominant_strategy = max(weighted_components, key=lambda key: weighted_components[key])
        dominant_weighted_edge = weighted_components[dominant_strategy]
        edge_gap = abs(net_yes - net_no)
        confidence = _clamp(
            0.32
            + _clamp(edge_percent / 20.0, 0.0, 0.35)
            + _clamp(edge_gap / 18.0, 0.0, 0.12)
            + _clamp(dominant_weighted_edge / max(1.0, edge_percent) * 0.08, 0.0, 0.08),
            0.05,
            0.97,
        )
        if not has_oracle:
            confidence = _clamp(confidence * 0.75, 0.05, 0.85)

        dedupe_key = make_dedupe_key(
            market.get("slug"),
            direction,
            regime,
            dominant_strategy,
            "oracle" if has_oracle else "no_oracle",
            round(entry_price, 4),
            round(ptb, 2) if ptb is not None else "na",
            market.get("end_time"),
        )

        expires_at: Optional[datetime] = parsed_end_time
        if expires_at is None:
            expires_at = now + timedelta(minutes=20)

        payload = dict(market)
        payload.update(
            {
                "signal_version": "crypto_worker_v2",
                "signal_family": "crypto_multistrategy",
                "strategy_origin": "crypto_worker",
                "selected_direction": direction,
                "regime": regime,
                "oracle_available": has_oracle,
                "timeframe_seconds": timeframe_seconds,
                "time_remaining_seconds": round(seconds_left, 2),
                "oracle_delta_pct": round(diff_pct, 6) if has_oracle else None,
                "model_prob_yes": round(model_prob_yes, 6),
                "model_prob_no": round(model_prob_no, 6),
                "strategy_weights": weights,
                "component_edges": {
                    "buy_yes": {
                        "directional": round(directional_yes, 6),
                        "pure_arb": round(pure_arb_yes, 6),
                        "rebalance": round(rebalance_yes, 6),
                    },
                    "buy_no": {
                        "directional": round(directional_no, 6),
                        "pure_arb": round(pure_arb_no, 6),
                        "rebalance": round(rebalance_no, 6),
                    },
                },
                "gross_edges": {
                    "buy_yes": round(gross_yes, 6),
                    "buy_no": round(gross_no, 6),
                },
                "execution_penalty_percent": round(execution_penalty, 6),
                "execution_penalty_breakdown": {
                    "fees": round(fee_penalty, 6),
                    "spread": round(spread_penalty, 6),
                    "slippage": round(slippage_penalty, 6),
                },
                "net_edges": {
                    "buy_yes": round(net_yes, 6),
                    "buy_no": round(net_no, 6),
                },
                "dominant_strategy": dominant_strategy,
                "dominant_component_edge": round(dominant_weighted_edge, 6),
                "edge_gap_percent": round(edge_gap, 6),
            }
        )

        await upsert_trade_signal(
            session,
            source="crypto",
            source_item_id=str(market.get("slug") or market_id),
            signal_type="crypto_worker_multistrat",
            strategy_type="crypto_15m",
            market_id=market_id,
            market_question=market.get("question"),
            direction=direction,
            entry_price=entry_price,
            edge_percent=edge_percent,
            confidence=confidence,
            liquidity=liquidity,
            expires_at=expires_at,
            payload_json=payload,
            dedupe_key=dedupe_key,
            commit=False,
        )
        emitted += 1

    await session.commit()
    await refresh_trade_signal_snapshots(session)
    return emitted
