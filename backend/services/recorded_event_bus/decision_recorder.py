"""Records the live trader's strategy decisions to the recorded-event bus.

This is the *measurement* half of the determinism work: every signal the
live orchestrator emits is teed to the ``strategy.decision`` topic so that
a backtest replay of the same window can be diffed against what the bot
actually decided live.  We do NOT replay *from* this topic — decisions are
re-derived from the recorded input stream during backtest.  This topic is
the oracle that proves the re-derivation matches live (see
``services.backtest.decision_parity``).

Design mirrors ``workers.news_worker``'s bus tee:
  * topic registered once (idempotent) with a ``parquet_root()``-derived
    storage_uri so nothing is hardcoded;
  * the publish is best-effort and never blocks or breaks the signal
    transaction — a lost decision record is a measurement gap, not a
    trading bug.

Entity = ``market_id`` (the Polymarket condition_id) so the parity harness
can entity-filter to a single market.  ``observed_at_us`` is the decision's
as-of time (``signal_emitted_at``), which is the timestamp a faithful
replay must reproduce.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

STRATEGY_DECISION_TOPIC = "strategy.decision"
ORDER_FILL_TOPIC = "order.fill"

# Payload keys that constitute the *decision* — the fields a replay must
# reproduce to be considered faithful.  Kept explicit (rather than dumping
# the whole signal payload) so the parity diff compares decision identity,
# not volatile bookkeeping (timestamps, roster hashes, sequence numbers).
_DECISION_FIELDS = (
    "signal_type",
    "strategy_type",
    "direction",
    "entry_price",
    "edge_percent",
    "confidence",
    "dedupe_key",
)

_topic_registered = False
_fill_topic_registered = False


def _to_us(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000)


async def ensure_decision_topic_registered() -> None:
    """Idempotently register ``strategy.decision`` in the topic catalog.

    Safe to call on the hot path — short-circuits after the first
    successful registration in this process.
    """
    global _topic_registered
    if _topic_registered:
        return
    from services.recorded_event_bus.catalog import register_topic
    from services.external_data.parquet_schema import parquet_root

    root = parquet_root()
    storage_uri = json.dumps({
        "sources": [{
            "kind": "parquet",
            "uri": str(root / "recorded_event_bus" / STRATEGY_DECISION_TOPIC),
        }],
    })
    await register_topic(
        slug=STRATEGY_DECISION_TOPIC,
        title="Strategy decisions (live signal emissions)",
        description=(
            "One envelope per signal the live orchestrator emits via "
            "signal_bus.upsert_trade_signal.  This is the parity oracle: "
            "a backtest replay of the same window re-derives decisions from "
            "the recorded input stream and is diffed against this topic to "
            "prove the strategy is a deterministic function of its inputs. "
            "Not used as a replay source."
        ),
        storage_kind="parquet",
        storage_uri=storage_uri,
        retention_days=30,
        publishers=["signal_bus"],
    )
    _topic_registered = True


def build_decision_payload(
    *,
    market_id: str,
    signal_type: str,
    strategy_type: Optional[str],
    direction: Optional[str],
    entry_price: Optional[float],
    edge_percent: Optional[float],
    confidence: Optional[float],
    dedupe_key: str,
    signal_id: Optional[str],
) -> dict[str, Any]:
    """The canonical decision-identity payload.  Kept as a free function
    so the parity harness can build the *same* shape from a replayed
    opportunity and compare apples to apples."""
    return {
        "market_id": market_id,
        "signal_type": signal_type,
        "strategy_type": strategy_type,
        "direction": direction,
        "entry_price": entry_price,
        "edge_percent": edge_percent,
        "confidence": confidence,
        "dedupe_key": dedupe_key,
        "signal_id": signal_id,
    }


async def publish_decision(
    *,
    market_id: str,
    observed_at: datetime,
    signal_type: str,
    strategy_type: Optional[str],
    direction: Optional[str],
    entry_price: Optional[float],
    edge_percent: Optional[float],
    confidence: Optional[float],
    dedupe_key: str,
    signal_id: Optional[str],
) -> None:
    """Tee one decision to the bus.  Best-effort: any failure is logged
    and swallowed so the signal path is never affected."""
    try:
        await ensure_decision_topic_registered()
        from services.recorded_event_bus import RecordedEvent
        from services.recorded_event_bus import bus as _bus
        import services.recorded_event_bus.storage  # noqa: F401  attach storage

        envelope = RecordedEvent(
            topic=STRATEGY_DECISION_TOPIC,
            entity_id=str(market_id or "unknown"),
            observed_at_us=_to_us(observed_at),
            payload=build_decision_payload(
                market_id=str(market_id or "unknown"),
                signal_type=signal_type,
                strategy_type=strategy_type,
                direction=direction,
                entry_price=entry_price,
                edge_percent=edge_percent,
                confidence=confidence,
                dedupe_key=dedupe_key,
                signal_id=signal_id,
            ),
            source="signal_bus",
        )
        await _bus.publish(envelope)
    except Exception:  # noqa: BLE001 — measurement gap, never a trading bug
        logger.debug("publish_decision tee failed (non-fatal)", exc_info=True)


async def ensure_fill_topic_registered() -> None:
    """Idempotently register ``order.fill`` — the live fill record the
    fill-model calibration harness compares its predictions against."""
    global _fill_topic_registered
    if _fill_topic_registered:
        return
    from services.recorded_event_bus.catalog import register_topic
    from services.external_data.parquet_schema import parquet_root

    root = parquet_root()
    storage_uri = json.dumps({
        "sources": [{
            "kind": "parquet",
            "uri": str(root / "recorded_event_bus" / ORDER_FILL_TOPIC),
        }],
    })
    await register_topic(
        slug=ORDER_FILL_TOPIC,
        title="Order fills (live execution)",
        description=(
            "One envelope per fill the live fill-monitor records. Fills are "
            "MODELED in backtest (your own order perturbs the book, so they "
            "can't be replayed from a static book); this topic is the ground "
            "truth the venue/fill model is calibrated against — predicted vs "
            "realized slippage and fill-rate. Not a replay source."
        ),
        storage_kind="parquet",
        storage_uri=storage_uri,
        retention_days=30,
        publishers=["fill_monitor"],
    )
    _fill_topic_registered = True


async def publish_fill(
    *,
    order_id: str,
    token_id: str,
    side: Optional[str],
    price: Optional[float],
    size_filled: Optional[float],
    size_requested: Optional[float],
    fill_percent: Optional[float],
    fee: Optional[float],
    observed_at: datetime,
) -> None:
    """Tee one live fill to the bus.  Best-effort; never affects the
    fill-persistence path."""
    try:
        await ensure_fill_topic_registered()
        from services.recorded_event_bus import RecordedEvent
        from services.recorded_event_bus import bus as _bus
        import services.recorded_event_bus.storage  # noqa: F401  attach storage

        envelope = RecordedEvent(
            topic=ORDER_FILL_TOPIC,
            entity_id=str(token_id or "unknown"),
            observed_at_us=_to_us(observed_at),
            payload={
                "order_id": order_id,
                "token_id": token_id,
                "side": side,
                "price": price,
                "size_filled": size_filled,
                "size_requested": size_requested,
                "fill_percent": fill_percent,
                "fee": fee,
            },
            source="fill_monitor",
        )
        await _bus.publish(envelope)
    except Exception:  # noqa: BLE001 — calibration gap, never a trading bug
        logger.debug("publish_fill tee failed (non-fatal)", exc_info=True)


def decision_identity(payload: dict[str, Any]) -> tuple:
    """Stable comparison key over the decision fields.  Used by the parity
    harness so live and replayed decisions compare on identity, not on
    volatile bookkeeping."""
    return tuple(payload.get(k) for k in _DECISION_FIELDS)
