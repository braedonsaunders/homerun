"""World Intelligence → Trade Signal Bus emitter.

Converts high-confidence world intelligence signals (instability spikes,
convergence zones, tension escalations, anomalies) into normalized trade
signals that the autotrader can consume.  This bridges the world
intelligence engine with the existing trading pipeline.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from config import settings

logger = logging.getLogger(__name__)


async def emit_world_intelligence_signals(
    session: AsyncSession,
    signals: list[Any],
    max_age_minutes: int = 120,
) -> int:
    """Convert world intelligence signals into trade signals.

    Only emits signals that:
    1. Have related_market_ids (matched to active prediction markets)
    2. Have severity >= 0.5 (meaningful signal strength)
    3. Have market_relevance_score >= 0.4

    Args:
        session: Async DB session.
        signals: List of WorldSignal dataclass instances.
        max_age_minutes: Ignore signals older than this.

    Returns:
        Number of trade signals emitted.
    """
    from services.signal_bus import upsert_trade_signal, make_dedupe_key

    if not signals:
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)
    emitted = 0

    for signal in signals:
        # Skip old signals
        if signal.detected_at and signal.detected_at < cutoff:
            continue

        # Skip low-severity signals
        if signal.severity < 0.5:
            continue

        # Skip signals without market matches
        if not signal.related_market_ids:
            continue

        # Skip low-relevance matches
        if signal.market_relevance_score is not None and signal.market_relevance_score < 0.4:
            continue

        # Map signal type to strategy type
        strategy_type = _map_signal_to_strategy(signal.signal_type)

        # Compute confidence from severity + relevance
        relevance = signal.market_relevance_score or 0.5
        confidence = min(1.0, signal.severity * 0.6 + relevance * 0.4)

        # Compute edge estimate based on signal type and severity
        edge_percent = _estimate_edge(signal)

        # Emit a trade signal for each related market
        for market_id in signal.related_market_ids[:5]:  # Cap at 5 markets per signal
            dedupe_key = make_dedupe_key(
                "world_intelligence",
                signal.signal_id,
                market_id,
            )

            try:
                await upsert_trade_signal(
                    session,
                    source="world_intelligence",
                    source_item_id=signal.signal_id,
                    signal_type=signal.signal_type,
                    strategy_type=strategy_type,
                    market_id=market_id,
                    market_question=None,  # Will be resolved by autotrader
                    direction=_infer_direction(signal),
                    entry_price=None,  # Market-dependent
                    edge_percent=edge_percent,
                    confidence=confidence,
                    liquidity=None,
                    expires_at=datetime.now(timezone.utc) + timedelta(hours=4),
                    payload_json={
                        "signal_type": signal.signal_type,
                        "severity": signal.severity,
                        "country": signal.country,
                        "title": signal.title,
                        "source": signal.source,
                        "market_relevance_score": signal.market_relevance_score,
                    },
                    dedupe_key=dedupe_key,
                    commit=False,
                )
                emitted += 1
            except Exception as e:
                logger.debug(
                    "Failed to emit world signal %s for market %s: %s",
                    signal.signal_id,
                    market_id,
                    e,
                )

    if emitted > 0:
        await session.commit()
        logger.info("Emitted %d world intelligence trade signals", emitted)

    return emitted


def _map_signal_to_strategy(signal_type: str) -> str:
    """Map a world intelligence signal type to a trading strategy type."""
    mapping = {
        "conflict": "event_driven",
        "tension": "bayesian_cascade",
        "instability": "event_driven",
        "convergence": "event_driven",
        "anomaly": "stat_arb",
        "military": "event_driven",
        "infrastructure": "event_driven",
    }
    return mapping.get(signal_type, "event_driven")


def _estimate_edge(signal: Any) -> float:
    """Estimate the informational edge percentage from a world signal.

    Higher severity signals imply larger market mispricings because
    the information hasn't been fully incorporated into prices yet.
    """
    base_edge = signal.severity * 15.0  # 0-15% base edge

    # Boost for convergence (multiple signal types in one area)
    if signal.signal_type == "convergence":
        base_edge *= 1.3

    # Boost for critical anomalies (significant deviation from baseline)
    if signal.signal_type == "anomaly":
        base_edge *= 1.2

    # Cap at reasonable maximum
    return min(25.0, max(5.0, base_edge))


def _infer_direction(signal: Any) -> Optional[str]:
    """Infer trade direction from signal characteristics.

    Most world intelligence signals indicate negative events (conflict,
    instability, disruption), which typically push YES prices higher
    on related geopolitical/risk markets.

    Returns None when direction can't be inferred (let autotrader decide).
    """
    # High-severity signals on geopolitical events → buy YES on risk markets
    if signal.severity >= 0.7 and signal.signal_type in (
        "conflict",
        "military",
        "instability",
        "convergence",
    ):
        return "buy_yes"

    # Infrastructure disruptions → situation-dependent
    # Don't guess direction for these
    return None
