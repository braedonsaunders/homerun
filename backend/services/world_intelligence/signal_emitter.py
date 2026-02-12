"""World Intelligence -> trade-signal emitter with resolver-based market mapping."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


async def emit_world_intelligence_signals(
    session: AsyncSession,
    signals: list[Any],
    max_age_minutes: int = 120,
) -> int:
    """Convert world intelligence signals into normalized tradable trade signals.

    Resolver behavior:
    - matches world signals to market IDs already attached by the aggregator
    - enriches with direction, token_id, and entry_price from the opportunity snapshot
    - emits only tradable candidates to avoid live/paper execution failures
    """
    from services.signal_bus import (
        make_dedupe_key,
        refresh_trade_signal_snapshots,
        upsert_trade_signal,
    )
    from services.world_intelligence.resolver import resolve_world_signal_opportunities

    if not signals:
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)
    resolved = await resolve_world_signal_opportunities(
        session,
        signals,
        max_markets_per_signal=5,
    )

    emitted = 0
    skipped_untradable = 0

    for candidate in resolved:
        severity = float(candidate.get("severity") or 0.0)
        relevance = float(candidate.get("market_relevance_score") or 0.0)
        detected_at_raw = candidate.get("detected_at")
        detected_at = None
        if isinstance(detected_at_raw, str):
            try:
                detected_at = datetime.fromisoformat(detected_at_raw.replace("Z", "+00:00"))
            except Exception:
                detected_at = None

        if detected_at is not None and detected_at < cutoff:
            continue
        if severity < 0.5:
            continue
        if relevance < 0.4:
            continue
        if not bool(candidate.get("tradable")):
            skipped_untradable += 1
            continue

        signal_id = str(candidate.get("signal_id") or "")
        market_id = str(candidate.get("market_id") or "")
        strategy_type = str(candidate.get("strategy_type") or "event_driven")
        direction = str(candidate.get("direction") or "")
        entry_price = float(candidate.get("entry_price") or 0.0)
        token_id = str(candidate.get("token_id") or "")
        if not signal_id or not market_id or not direction or not token_id or entry_price <= 0:
            skipped_untradable += 1
            continue

        dedupe_key = make_dedupe_key(
            "world_intelligence",
            signal_id,
            market_id,
            token_id,
        )

        expires_at = datetime.now(timezone.utc) + timedelta(hours=4)
        await upsert_trade_signal(
            session,
            source="world_intelligence",
            source_item_id=signal_id,
            signal_type=str(candidate.get("signal_type") or "unknown"),
            strategy_type=strategy_type,
            market_id=market_id,
            market_question=candidate.get("market_question"),
            direction=direction,
            entry_price=entry_price,
            edge_percent=float(candidate.get("edge_percent") or 0.0),
            confidence=float(candidate.get("confidence") or 0.0),
            liquidity=float(candidate.get("liquidity") or 0.0),
            expires_at=expires_at,
            payload_json={
                "signal_type": candidate.get("signal_type"),
                "severity": severity,
                "country": candidate.get("country"),
                "title": candidate.get("title"),
                "source": candidate.get("source"),
                "market_relevance_score": relevance,
                "event_id": candidate.get("event_id"),
                "event_slug": candidate.get("event_slug"),
                "event_title": candidate.get("event_title"),
                "token_id": token_id,
                "positions_to_take": candidate.get("positions_to_take") or [],
                "resolver_status": candidate.get("resolver_status"),
                "resolver_missing_fields": candidate.get("missing_fields") or [],
                "metadata": candidate.get("metadata") or {},
            },
            dedupe_key=dedupe_key,
            commit=False,
        )
        emitted += 1

    if emitted > 0:
        await session.commit()
        await refresh_trade_signal_snapshots(session)
    logger.info(
        "World-intel signal emission: emitted=%d skipped_untradable=%d candidates=%d",
        emitted,
        skipped_untradable,
        len(resolved),
    )
    return emitted

