"""SQL-table adapters: read-only wrappers around pre-existing
tables so backtest replay can stream them as bus topics.

Why read-only:
  These tables already have authoritative writers (the
  market_data_ingestor, the ws-monitor recorder, etc.).  Writing
  through the bus AND through the legacy recorder would double-
  persist.  The migration plan is "tap recorders to ALSO publish
  envelopes to live subscribers" — see batch C — not "rewrite the
  recorder."  These adapters exist so the *replay* side of the bus
  works against existing data on day one.

Adapters dispatch on the ``adapter`` field of the topic's
storage_uri JSON.  Each adapter knows:
  * the ORM model
  * how to apply a window filter (which timestamp column)
  * how to apply an entity filter
  * how to project one row → RecordedEvent

Bounded memory: each adapter is an async generator that streams from
the DB cursor.  Polymarket book deltas alone are 7M rows; the
replayer cannot materialise a whole window.

Where the adapter doesn't have a sequence number to populate, it
synthesises one from the row's autoincrement id (or leaves None) so
the bus's heap-merge has a deterministic tiebreaker.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    AsyncSessionLocal,
    BacktestAsyncSessionLocal,
    BookDeltaEvent,
    MarketMicrostructureSnapshot,
    OpportunityHistory,
)
from services.recorded_event_bus.catalog import TopicSpec
from services.recorded_event_bus.envelope import RecordedEvent

logger = logging.getLogger(__name__)


# ── Per-adapter projection functions ─────────────────────────────────


def _dt_to_us(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000)


def _project_microstructure_snapshot(
    row: MarketMicrostructureSnapshot,
    *,
    topic: str,
) -> RecordedEvent:
    """Project one row into a RecordedEvent under the caller's topic.

    The same table feeds multiple topics (one per provider —
    ``polymarket.book.snapshot`` for live, ``polybacktest.book.snapshot``
    for batch import, etc.).  The projector takes the topic slug from
    the caller so subscribers see a topic name that matches what they
    asked for, not whichever provider happens to own the row.
    """
    payload = {
        "snapshot_type": row.snapshot_type,
        "best_bid": row.best_bid,
        "best_ask": row.best_ask,
        "spread_bps": row.spread_bps,
        "bids_json": row.bids_json,
        "asks_json": row.asks_json,
        "trade_price": row.trade_price,
        "trade_size": row.trade_size,
        "trade_side": row.trade_side,
        "exchange_ts_ms": row.exchange_ts_ms,
        "provider": row.provider,
    }
    obs_us = _dt_to_us(row.observed_at)
    return RecordedEvent(
        topic=topic,
        entity_id=row.token_id,
        observed_at_us=obs_us,
        ingested_at_us=_dt_to_us(row.created_at) if row.created_at else obs_us,
        source=row.provider or "polymarket",
        sequence=int(row.sequence) if row.sequence is not None else None,
        payload=payload,
    )


def _project_book_delta(row: BookDeltaEvent, *, topic: str) -> RecordedEvent:
    payload: dict[str, Any] = {
        "event_type": row.event_type,
        "side": row.side,
        "price": row.price,
        "trade_size": row.trade_size,
        "cancel_size": row.cancel_size,
        "exchange_ts_ms": row.exchange_ts_ms,
        "provider": row.provider,
    }
    obs_us = _dt_to_us(row.observed_at)
    return RecordedEvent(
        topic=topic,
        entity_id=row.token_id,
        observed_at_us=obs_us,
        ingested_at_us=obs_us,
        source=row.provider or "polymarket",
        sequence=int(row.sequence) if row.sequence is not None else None,
        payload=payload,
    )


def _project_opportunity_history(row: OpportunityHistory, *, topic: str) -> RecordedEvent:
    payload = {
        "strategy_type": getattr(row, "strategy_type", None),
        "positions_data": getattr(row, "positions_data", None),
        "title": getattr(row, "title", None),
        "expected_roi": getattr(row, "expected_roi", None),
        "total_cost": getattr(row, "total_cost", None),
        "event_id": getattr(row, "event_id", None),
        "first_seen_at": (
            row.first_seen_at.isoformat() if getattr(row, "first_seen_at", None) else None
        ),
    }
    obs_us = _dt_to_us(row.detected_at)
    ing_us = _dt_to_us(row.created_at) if getattr(row, "created_at", None) else obs_us
    return RecordedEvent(
        topic=topic,
        entity_id=row.id,
        observed_at_us=obs_us,
        ingested_at_us=ing_us,
        source="scanner",
        sequence=None,
        payload=payload,
    )


async def _project_wallet_monitor_row(
    *, topic: str,
    id: str, wallet_address: str, token_id: str,
    side: str, size: float, price: float,
    tx_hash: str, order_hash: str, log_index: int,
    block_number: int, detection_latency_ms: float,
    detected_at: datetime,
) -> RecordedEvent:
    obs_us = _dt_to_us(detected_at)
    payload = {
        "side": side,
        "size": size,
        "price": price,
        "tx_hash": tx_hash,
        "order_hash": order_hash,
        "log_index": log_index,
        "block_number": block_number,
        "detection_latency_ms": detection_latency_ms,
        "wallet_address": wallet_address,
        "token_id": token_id,
    }
    return RecordedEvent(
        topic=topic,
        entity_id=wallet_address,
        observed_at_us=obs_us,
        ingested_at_us=obs_us,
        source="ws_monitor",
        sequence=int(block_number) * 1_000_000 + int(log_index or 0)
            if block_number is not None else None,
        payload=payload,
    )


# ── Stream functions ─────────────────────────────────────────────────


async def _stream_microstructure(
    spec: TopicSpec, window
) -> AsyncIterator[RecordedEvent]:
    """Stream rows from ``market_microstructure_snapshots`` for one
    provider's slice of the table.

    The ``market_microstructure_snapshots`` table is a multi-tenant
    home for snapshot data from several providers ('polymarket' for
    live WS, 'polybacktest' for batch-imported history, future
    providers as they're added).  Each provider gets its OWN bus topic
    so subscribers don't mix sources — the topic's ``storage_uri``
    JSON carries the provider filter:

        {"adapter": "MarketMicrostructureSnapshot",
         "table": "market_microstructure_snapshots",
         "provider": "polymarket"}

    Missing ``provider`` key in storage_uri = no filter applied (legacy
    behaviour — keeps the original ``polymarket.book.snapshot`` topic
    working unchanged even before its catalog row is updated).
    """
    start_dt = datetime.fromtimestamp(window.start_us / 1e6, tz=timezone.utc).replace(tzinfo=None)
    end_dt = datetime.fromtimestamp(window.end_us / 1e6, tz=timezone.utc).replace(tzinfo=None)
    entity_set = (
        window.entity_filter.get(spec.slug)
        if window.entity_filter else None
    )
    # Resolve the provider filter from the topic's storage_uri JSON.
    provider_filter: str | None = None
    if spec.storage_uri:
        try:
            cfg = json.loads(spec.storage_uri)
            provider_filter = cfg.get("provider")
        except (TypeError, ValueError, json.JSONDecodeError):
            provider_filter = None

    async with BacktestAsyncSessionLocal() as session:
        q = (
            select(MarketMicrostructureSnapshot)
            .where(MarketMicrostructureSnapshot.observed_at >= start_dt)
            .where(MarketMicrostructureSnapshot.observed_at < end_dt)
            # Restrict to book snapshots (this topic is books, not the
            # composite mms table that also holds trade rows).
            .where(MarketMicrostructureSnapshot.snapshot_type == "book")
        )
        if provider_filter:
            q = q.where(MarketMicrostructureSnapshot.provider == provider_filter)
        if entity_set is not None:
            q = q.where(MarketMicrostructureSnapshot.token_id.in_(entity_set))
        q = q.order_by(
            MarketMicrostructureSnapshot.observed_at,
            MarketMicrostructureSnapshot.sequence,
        )
        # Stream via yield_per to keep memory bounded.
        result = await session.stream(q.execution_options(yield_per=1000))
        async for row in result.scalars():
            yield _project_microstructure_snapshot(row, topic=spec.slug)


async def _stream_book_deltas(
    spec: TopicSpec, window
) -> AsyncIterator[RecordedEvent]:
    """Stream rows from ``book_delta_events`` for one provider slice.
    Same provider-filter convention as ``_stream_microstructure`` —
    see that docstring."""
    start_dt = datetime.fromtimestamp(window.start_us / 1e6, tz=timezone.utc).replace(tzinfo=None)
    end_dt = datetime.fromtimestamp(window.end_us / 1e6, tz=timezone.utc).replace(tzinfo=None)
    entity_set = (
        window.entity_filter.get(spec.slug)
        if window.entity_filter else None
    )
    provider_filter: str | None = None
    if spec.storage_uri:
        try:
            cfg = json.loads(spec.storage_uri)
            provider_filter = cfg.get("provider")
        except (TypeError, ValueError, json.JSONDecodeError):
            provider_filter = None

    async with BacktestAsyncSessionLocal() as session:
        q = (
            select(BookDeltaEvent)
            .where(BookDeltaEvent.observed_at >= start_dt)
            .where(BookDeltaEvent.observed_at < end_dt)
        )
        if provider_filter:
            q = q.where(BookDeltaEvent.provider == provider_filter)
        if entity_set is not None:
            q = q.where(BookDeltaEvent.token_id.in_(entity_set))
        q = q.order_by(BookDeltaEvent.observed_at, BookDeltaEvent.sequence)
        result = await session.stream(q.execution_options(yield_per=2000))
        async for row in result.scalars():
            yield _project_book_delta(row, topic=spec.slug)


async def _stream_opportunities(
    spec: TopicSpec, window
) -> AsyncIterator[RecordedEvent]:
    start_dt = datetime.fromtimestamp(window.start_us / 1e6, tz=timezone.utc).replace(tzinfo=None)
    end_dt = datetime.fromtimestamp(window.end_us / 1e6, tz=timezone.utc).replace(tzinfo=None)
    entity_set = (
        window.entity_filter.get(spec.slug)
        if window.entity_filter else None
    )
    async with AsyncSessionLocal() as session:
        q = (
            select(OpportunityHistory)
            .where(OpportunityHistory.detected_at >= start_dt)
            .where(OpportunityHistory.detected_at < end_dt)
        )
        if entity_set is not None:
            q = q.where(OpportunityHistory.id.in_(entity_set))
        q = q.order_by(OpportunityHistory.detected_at)
        result = await session.stream(q.execution_options(yield_per=500))
        async for row in result.scalars():
            yield _project_opportunity_history(row, topic=spec.slug)


async def _stream_wallet_trades(
    spec: TopicSpec, window
) -> AsyncIterator[RecordedEvent]:
    """wallet_monitor_events is queried via raw SQL since the table
    doesn't have an ORM class in models.database (the existing replay
    paths use SQL text)."""
    from sqlalchemy import text
    start_dt = datetime.fromtimestamp(window.start_us / 1e6, tz=timezone.utc).replace(tzinfo=None)
    end_dt = datetime.fromtimestamp(window.end_us / 1e6, tz=timezone.utc).replace(tzinfo=None)
    entity_set = (
        window.entity_filter.get(spec.slug)
        if window.entity_filter else None
    )
    sql = """
        SELECT id, wallet_address, token_id, side, size, price,
               tx_hash, order_hash, log_index, block_number,
               detection_latency_ms, detected_at
        FROM wallet_monitor_events
        WHERE detected_at >= :start_dt
          AND detected_at < :end_dt
    """
    params: dict[str, Any] = {"start_dt": start_dt, "end_dt": end_dt}
    if entity_set is not None:
        # Postgres ANY(:wallets) lets us bind a list parameter.
        sql += " AND wallet_address = ANY(:wallets)"
        params["wallets"] = list(entity_set)
    sql += " ORDER BY detected_at, block_number, log_index"

    async with AsyncSessionLocal() as session:
        result = await session.stream(
            text(sql).execution_options(yield_per=2000), params,
        )
        async for r in result.mappings():
            yield await _project_wallet_monitor_row(topic=spec.slug, **dict(r))


# ── Registry ─────────────────────────────────────────────────────────


ADAPTERS: dict[
    str,
    Callable[[TopicSpec, Any], AsyncIterator[RecordedEvent]],
] = {
    "MarketMicrostructureSnapshot": _stream_microstructure,
    "BookDeltaEvent": _stream_book_deltas,
    "OpportunityHistory": _stream_opportunities,
    "WalletMonitorEvent": _stream_wallet_trades,
}


def get_sql_adapter(spec: TopicSpec) -> Callable[[TopicSpec, Any], AsyncIterator[RecordedEvent]]:
    """Resolve the SQL adapter from the topic's storage_uri JSON."""
    if not spec.storage_uri:
        raise ValueError(f"sql_table topic {spec.slug!r} missing storage_uri JSON")
    try:
        cfg = json.loads(spec.storage_uri)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"sql_table topic {spec.slug!r} has invalid storage_uri JSON: {exc}"
        )
    adapter_name = cfg.get("adapter")
    fn = ADAPTERS.get(adapter_name)
    if fn is None:
        raise ValueError(
            f"sql_table topic {spec.slug!r} references unknown adapter {adapter_name!r}; "
            f"known: {sorted(ADAPTERS)}"
        )
    return fn
