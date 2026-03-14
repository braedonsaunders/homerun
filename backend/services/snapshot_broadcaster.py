"""
Snapshot broadcaster.

Bridges worker data to connected WebSocket clients in the API process.

**Primary path (event-driven):** Workers publish events on ``event_bus``;
the broadcaster subscribes and immediately relays them to WS
clients, applying signature-based deduplication for high-frequency events.

**Fallback path (DB poll):** A slow 30-second DB-poll loop runs as a safety
net so data is never lost even if a worker forgets (or fails) to publish.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import select

from api.websocket import manager
from models.database import (
    AsyncSessionLocal,
    OpportunityEvent,
    EventsSignal,
    EventsSnapshot,
)
from services import shared_state
from services.event_bus import event_bus
from services.news import shared_state as news_shared_state
from services.signal_bus import read_trade_signal_source_stats
from services.trader_orchestrator_state import read_orchestrator_snapshot
from services.worker_state import list_worker_snapshots, summarize_worker_stats
from services.weather import shared_state as weather_shared_state
from utils.logger import get_logger
from utils.market_urls import serialize_opportunity_with_links

logger = get_logger("snapshot_broadcaster")

# Event types that use signature-based deduplication to avoid flooding WS.
_DEDUP_EVENT_TYPES = frozenset(
    {
        "scanner_status",
        "scanner_activity",
        "worker_status_update",
        "crypto_markets_update",
        "weather_status",
        "news_workflow_status",
        "trader_orchestrator_status",
    }
)


class SnapshotBroadcaster:
    """Event-driven broadcaster with DB-poll fallback."""

    def __init__(self) -> None:
        self._running = False
        self._fallback_task: Optional[asyncio.Task] = None
        self._listener_task: Optional[asyncio.Task] = None
        # Signature caches for deduplication (shared by event path + fallback).
        self._last_activity: Optional[str] = None
        self._last_status_sig: Optional[tuple] = None
        self._last_opp_sig: Optional[tuple] = None
        self._last_event_ts: Optional[datetime] = None
        self._last_weather_status_sig: Optional[tuple] = None
        self._last_weather_opp_sig: Optional[tuple] = None
        self._last_news_status_sig: Optional[tuple] = None
        self._last_news_update_sig: Optional[tuple] = None
        self._last_worker_status_sig: Optional[tuple] = None
        self._last_crypto_markets_sig: Optional[tuple] = None
        self._last_signals_sig: Optional[tuple] = None
        self._last_world_status_sig: Optional[tuple] = None
        self._last_world_update_sig: Optional[tuple] = None
        self._last_orchestrator_status_sig: Optional[tuple] = None
        # Async queue fed by event bus subscription.
        self._event_queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()

    # ------------------------------------------------------------------
    # Signature helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _opportunity_ai_signature(opportunities: list) -> int:
        """Compact signature that changes when inline AI analysis changes."""
        signature_rows: list[tuple] = []
        for opp in opportunities:
            analysis = getattr(opp, "ai_analysis", None)
            if analysis is None:
                signature_rows.append((opp.stable_id or opp.id, None, None, None))
                continue
            judged_at = analysis.judged_at.isoformat() if getattr(analysis, "judged_at", None) is not None else None
            score = round(float(getattr(analysis, "overall_score", 0.0) or 0.0), 6)
            signature_rows.append(
                (
                    opp.stable_id or opp.id,
                    getattr(analysis, "recommendation", None),
                    score,
                    judged_at,
                )
            )
        return hash(tuple(signature_rows))

    def _compute_dedup_sig(self, event_type: str, data: dict[str, Any]) -> Optional[tuple]:
        """Return a signature tuple for dedup-eligible event types, else None."""
        if event_type == "scanner_status":
            return (
                data.get("running"),
                data.get("enabled"),
                data.get("interval_seconds"),
                data.get("last_scan"),
                data.get("opportunities_count"),
            )
        if event_type == "scanner_activity":
            return (data.get("activity"),)
        if event_type == "worker_status_update":
            workers = data.get("workers") or []
            return tuple(
                (
                    row.get("worker_name"),
                    row.get("running"),
                    row.get("enabled"),
                    row.get("updated_at"),
                    row.get("last_run_at"),
                    row.get("last_error"),
                )
                for row in workers
            )
        if event_type == "crypto_markets_update":
            markets = data.get("markets") or []
            rows: list[tuple[str, float | None, float | None, float | None, float | None, Any, Any, tuple]] = []

            def _to_price_sig(value: Any) -> float | None:
                try:
                    parsed = float(value)
                except (TypeError, ValueError):
                    return None
                if parsed != parsed:
                    return None
                return round(parsed, 6)

            def _source_sig(source_map: Any) -> tuple:
                if not isinstance(source_map, dict):
                    return ()
                rows: list[tuple[str, float | None, Any]] = []
                for raw_key, raw_value in source_map.items():
                    if not isinstance(raw_value, dict):
                        continue
                    source_key = str(raw_value.get("source") or raw_key or "")
                    rows.append(
                        (
                            source_key,
                            _to_price_sig(raw_value.get("price")),
                            raw_value.get("updated_at_ms"),
                        )
                    )
                rows.sort(key=lambda row: row[0])
                return tuple(rows)

            for market in markets:
                if not isinstance(market, dict):
                    continue
                market_id = str(
                    market.get("id")
                    or market.get("slug")
                    or market.get("condition_id")
                    or ""
                )
                rows.append(
                    (
                        market_id,
                        _to_price_sig(market.get("up_price")),
                        _to_price_sig(market.get("down_price")),
                        _to_price_sig(market.get("combined")),
                        _to_price_sig(market.get("oracle_price")),
                        market.get("oracle_updated_at_ms"),
                        market.get("updated_at_ms"),
                        _source_sig(market.get("oracle_prices_by_source")),
                    )
                )

            rows.sort(key=lambda row: row[0])
            return (len(rows), tuple(rows))
        if event_type == "weather_status":
            return (
                data.get("running"),
                data.get("enabled"),
                data.get("interval_seconds"),
                data.get("last_scan"),
                data.get("opportunities_count"),
            )
        if event_type == "news_workflow_status":
            return (
                data.get("running"),
                data.get("enabled"),
                data.get("paused"),
                data.get("interval_seconds"),
                data.get("last_scan"),
                data.get("next_scan"),
                data.get("pending_intents"),
                data.get("degraded_mode"),
                data.get("last_error"),
            )
        if event_type == "trader_orchestrator_status":
            return (
                data.get("running"),
                data.get("enabled"),
                data.get("last_run_at"),
                data.get("decisions_count"),
                data.get("orders_count"),
                data.get("open_orders"),
                data.get("gross_exposure_usd"),
                data.get("daily_pnl"),
                data.get("last_error"),
            )
        return None

    def _get_last_sig(self, event_type: str) -> Optional[tuple]:
        sig_map = {
            "scanner_status": "_last_status_sig",
            "scanner_activity": "_last_activity",
            "worker_status_update": "_last_worker_status_sig",
            "crypto_markets_update": "_last_crypto_markets_sig",
            "weather_status": "_last_weather_status_sig",
            "news_workflow_status": "_last_news_status_sig",
            "trader_orchestrator_status": "_last_orchestrator_status_sig",
        }
        attr = sig_map.get(event_type)
        if attr is None:
            return None
        val = getattr(self, attr, None)
        # scanner_activity stores a plain string; wrap it for comparison.
        if event_type == "scanner_activity":
            return (val,) if val is not None else None
        return val

    def _set_last_sig(self, event_type: str, sig: tuple) -> None:
        if event_type == "scanner_status":
            self._last_status_sig = sig
        elif event_type == "scanner_activity":
            self._last_activity = sig[0] if sig else None
        elif event_type == "worker_status_update":
            self._last_worker_status_sig = sig
        elif event_type == "crypto_markets_update":
            self._last_crypto_markets_sig = sig
        elif event_type == "weather_status":
            self._last_weather_status_sig = sig
        elif event_type == "news_workflow_status":
            self._last_news_status_sig = sig
        elif event_type == "trader_orchestrator_status":
            self._last_orchestrator_status_sig = sig

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, interval_seconds: float = 1.0) -> None:
        """Start event listener + fallback DB-poll loop (idempotent)."""
        if self._running:
            return
        self._running = True

        # Subscribe to ALL events on the bus.
        event_bus.subscribe("*", self._on_event)

        # Primary: event listener that drains the queue and broadcasts.
        self._listener_task = asyncio.create_task(
            self._run_event_listener(),
            name="snapshot-broadcaster-events",
        )
        # Fallback: slow DB poll at 30s for data consistency.
        self._fallback_task = asyncio.create_task(
            self._run_loop(interval_seconds=30.0),
            name="snapshot-broadcaster-fallback",
        )
        logger.info(
            "Snapshot broadcaster started (event-driven primary, DB-poll fallback at 30s)",
            interval_seconds=interval_seconds,
        )

    async def stop(self) -> None:
        """Stop all tasks."""
        self._running = False
        event_bus.unsubscribe("*", self._on_event)
        for task in (self._listener_task, self._fallback_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._listener_task = None
        self._fallback_task = None
        self._last_activity = None
        self._last_status_sig = None
        self._last_opp_sig = None
        self._last_event_ts = None
        self._last_weather_status_sig = None
        self._last_weather_opp_sig = None
        self._last_news_status_sig = None
        self._last_news_update_sig = None
        self._last_worker_status_sig = None
        self._last_crypto_markets_sig = None
        self._last_signals_sig = None
        self._last_world_status_sig = None
        self._last_world_update_sig = None
        self._last_orchestrator_status_sig = None
        logger.info("Snapshot broadcaster stopped")

    # ------------------------------------------------------------------
    # Event-driven path (primary)
    # ------------------------------------------------------------------

    async def _on_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Event bus callback -- enqueue for the listener task."""
        try:
            self._event_queue.put_nowait((event_type, data))
        except asyncio.QueueFull:
            logger.debug("Event queue full, dropping event", event_type=event_type)

    async def _run_event_listener(self) -> None:
        """Drain events from the queue and broadcast to WS clients."""
        while self._running:
            try:
                event_type, data = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                await self._broadcast_event(event_type, data)
            except Exception as exc:
                logger.debug("Event broadcast failed", event_type=event_type, error=str(exc))

    async def _broadcast_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Broadcast a single event, applying dedup for high-frequency types."""
        # For dedup-eligible events, check signature first.
        if event_type in _DEDUP_EVENT_TYPES:
            sig = self._compute_dedup_sig(event_type, data)
            if sig is not None:
                last = self._get_last_sig(event_type)
                if sig == last:
                    return  # duplicate -- skip
                self._set_last_sig(event_type, sig)

        # Broadcast the message in the exact {type, data} format the
        # frontend expects.
        await manager.broadcast({"type": event_type, "data": data})

    # ------------------------------------------------------------------
    # Fallback DB-poll path (runs every 30 seconds)
    # ------------------------------------------------------------------

    async def _run_loop(self, interval_seconds: float) -> None:
        """FALLBACK: Slow DB poll loop to ensure data consistency.

        This is the original polling logic, reduced to 30s interval now that
        workers publish directly to the in-process event bus. It catches any data that workers may
        not have published to the bus.

        Trader terminal streams (events/decisions/orders) are intentionally
        excluded from fallback polling so terminal cadence is driven only by
        real-time event bus delivery.
        """
        while self._running:
            try:
                async with AsyncSessionLocal() as session:
                    opportunities, status = await shared_state.read_scanner_snapshot(session)
                    weather_opps = await weather_shared_state.get_weather_opportunities_from_db(session)
                    weather_status = await weather_shared_state.get_weather_status_from_db(session)
                    weather_status["opportunities_count"] = len(weather_opps)
                    news_status = await news_shared_state.get_news_status_from_db(session)
                    worker_statuses = await list_worker_snapshots(
                        session,
                        include_stats=True,
                        stats_mode="summary",
                    )
                    orchestrator_status = await read_orchestrator_snapshot(session)
                    if isinstance(orchestrator_status, dict):
                        orchestrator_status = dict(orchestrator_status)
                        orchestrator_status["stats"] = summarize_worker_stats(orchestrator_status.get("stats"))
                    signal_sources = await read_trade_signal_source_stats(session)
                    world_snapshot = (
                        (await session.execute(select(EventsSnapshot).where(EventsSnapshot.id == "latest")))
                        .scalars()
                        .one_or_none()
                    )
                    world_rows = (
                        (
                            await session.execute(
                                select(EventsSignal)
                                .order_by(
                                    EventsSignal.detected_at.desc(),
                                    EventsSignal.severity.desc(),
                                )
                                .limit(100)
                            )
                        )
                        .scalars()
                        .all()
                    )
                    event_query = select(OpportunityEvent).order_by(OpportunityEvent.created_at.asc())
                    if self._last_event_ts is not None:
                        event_query = event_query.where(OpportunityEvent.created_at > self._last_event_ts)
                    event_query = event_query.limit(200)
                    event_rows = (await session.execute(event_query)).scalars().all()

                if event_rows:
                    self._last_event_ts = event_rows[-1].created_at
                    await manager.broadcast(
                        {
                            "type": "opportunity_events",
                            "topic": "opportunities.summary",
                            "data": {
                                "events": [
                                    {
                                        "id": row.id,
                                        "stable_id": row.stable_id,
                                        "run_id": row.run_id,
                                        "event_type": row.event_type,
                                        "opportunity": row.opportunity_json,
                                        "created_at": row.created_at.isoformat() if row.created_at else None,
                                    }
                                    for row in event_rows
                                ]
                            },
                        }
                    )
                    for row in event_rows:
                        payload = row.opportunity_json if isinstance(row.opportunity_json, dict) else {}
                        revision = int(payload.get("revision") or 0)
                        await manager.broadcast(
                            {
                                "type": "opportunity_update",
                                "topic": f"opportunities.detail:{row.stable_id}",
                                "data": {
                                    "id": row.id,
                                    "stable_id": row.stable_id,
                                    "run_id": row.run_id,
                                    "event_type": row.event_type,
                                    "revision": revision,
                                    "opportunity": payload,
                                    "created_at": row.created_at.isoformat() if row.created_at else None,
                                },
                            }
                        )

                activity = status.get("current_activity") or "Idle"
                status_sig = (
                    status.get("running"),
                    status.get("enabled"),
                    status.get("interval_seconds"),
                    status.get("last_scan"),
                    status.get("opportunities_count"),
                )
                first_id = opportunities[0].id if opportunities else None
                ai_sig = self._opportunity_ai_signature(opportunities)
                opp_sig = (
                    status.get("last_scan"),
                    len(opportunities),
                    first_id,
                    ai_sig,
                )

                if activity != self._last_activity:
                    self._last_activity = activity
                    await manager.broadcast(
                        {
                            "type": "scanner_activity",
                            "topic": "opportunities.summary",
                            "data": {"activity": activity},
                        }
                    )

                if status_sig != self._last_status_sig:
                    self._last_status_sig = status_sig
                    await manager.broadcast(
                        {
                            "type": "scanner_status",
                            "topic": "opportunities.summary",
                            "data": status,
                        }
                    )

                if opp_sig != self._last_opp_sig:
                    self._last_opp_sig = opp_sig
                    await manager.broadcast(
                        {
                            "type": "opportunities_update",
                            "topic": "opportunities.summary",
                            "data": {
                                "count": len(opportunities),
                                "opportunities": [serialize_opportunity_with_links(o) for o in opportunities[:50]],
                            },
                        }
                    )

                weather_status_sig = (
                    weather_status.get("running"),
                    weather_status.get("enabled"),
                    weather_status.get("interval_seconds"),
                    weather_status.get("last_scan"),
                    weather_status.get("opportunities_count"),
                )
                weather_first_id = weather_opps[0].id if weather_opps else None
                weather_opp_sig = (
                    weather_status.get("last_scan"),
                    len(weather_opps),
                    weather_first_id,
                )

                if weather_status_sig != self._last_weather_status_sig:
                    self._last_weather_status_sig = weather_status_sig
                    await manager.broadcast({"type": "weather_status", "data": weather_status})

                if weather_opp_sig != self._last_weather_opp_sig:
                    self._last_weather_opp_sig = weather_opp_sig
                    await manager.broadcast(
                        {
                            "type": "weather_update",
                            "data": {
                                "count": len(weather_opps),
                                "opportunities": [serialize_opportunity_with_links(o) for o in weather_opps[:100]],
                                "status": weather_status,
                            },
                        }
                    )

                news_stats = news_status.get("stats") or {}
                news_status_sig = (
                    news_status.get("running"),
                    news_status.get("enabled"),
                    news_status.get("paused"),
                    news_status.get("interval_seconds"),
                    news_status.get("last_scan"),
                    news_status.get("next_scan"),
                    news_status.get("pending_intents"),
                    news_status.get("degraded_mode"),
                    news_status.get("last_error"),
                )
                news_update_sig = (
                    news_status.get("last_scan"),
                    news_stats.get("findings"),
                    news_stats.get("intents"),
                    news_status.get("pending_intents"),
                )

                if news_status_sig != self._last_news_status_sig:
                    self._last_news_status_sig = news_status_sig
                    await manager.broadcast({"type": "news_workflow_status", "data": news_status})

                if news_update_sig != self._last_news_update_sig:
                    self._last_news_update_sig = news_update_sig
                    await manager.broadcast(
                        {
                            "type": "news_workflow_update",
                            "data": {
                                "status": news_status,
                                "findings": int(news_stats.get("findings", 0) or 0),
                                "intents": int(news_stats.get("intents", 0) or 0),
                                "pending_intents": int(news_status.get("pending_intents", 0) or 0),
                            },
                        }
                    )

                worker_sig = tuple(
                    (
                        row.get("worker_name"),
                        row.get("running"),
                        row.get("enabled"),
                        row.get("updated_at"),
                        row.get("last_run_at"),
                        row.get("last_error"),
                    )
                    for row in worker_statuses
                )
                if worker_sig != self._last_worker_status_sig:
                    self._last_worker_status_sig = worker_sig
                    await manager.broadcast({"type": "worker_status_update", "data": {"workers": worker_statuses}})

                crypto_markets: list[dict[str, Any]] | None = None
                for worker in worker_statuses:
                    if worker.get("worker_name") != "crypto":
                        continue
                    stats = worker.get("stats") if isinstance(worker.get("stats"), dict) else {}
                    if isinstance(stats.get("markets"), list):
                        crypto_markets = list(stats.get("markets") or [])
                    break
                if not isinstance(crypto_markets, list):
                    crypto_markets = []

                crypto_sig = self._compute_dedup_sig("crypto_markets_update", {"markets": crypto_markets})
                if crypto_sig != self._last_crypto_markets_sig:
                    self._last_crypto_markets_sig = crypto_sig
                    await manager.broadcast(
                        {
                            "type": "crypto_markets_update",
                            "data": {"markets": crypto_markets},
                        }
                    )

                signals_sig = tuple(
                    (
                        row["source"],
                        row["pending_count"],
                        row["selected_count"],
                        row["submitted_count"],
                        row["executed_count"],
                        row["skipped_count"],
                        row["expired_count"],
                        row["failed_count"],
                        row["latest_signal_at"],
                    )
                    for row in signal_sources
                )
                if signals_sig != self._last_signals_sig:
                    self._last_signals_sig = signals_sig
                    await manager.broadcast({"type": "signals_update", "data": {"sources": signal_sources}})

                world_status = {}
                world_stats = {}
                world_updated_at = None
                if world_snapshot is not None:
                    if isinstance(world_snapshot.status, dict):
                        world_status = dict(world_snapshot.status)
                    if isinstance(world_snapshot.stats, dict):
                        world_stats = dict(world_snapshot.stats)
                        world_stats.pop("runtime_state", None)
                    world_updated_at = world_snapshot.updated_at.isoformat() if world_snapshot.updated_at else None
                world_status_sig = (
                    world_status.get("running"),
                    world_status.get("enabled"),
                    world_status.get("last_scan"),
                    world_status.get("next_scan"),
                    world_status.get("last_error"),
                    world_updated_at,
                )
                if world_status_sig != self._last_world_status_sig:
                    self._last_world_status_sig = world_status_sig
                    await manager.broadcast(
                        {
                            "type": "events_status",
                            "data": {
                                "status": world_status,
                                "stats": world_stats,
                                "updated_at": world_updated_at,
                            },
                        }
                    )

                world_signals = [
                    {
                        "signal_id": row.id,
                        "signal_type": row.signal_type,
                        "severity": float(row.severity or 0.0),
                        "country": row.country,
                        "latitude": row.latitude,
                        "longitude": row.longitude,
                        "title": row.title,
                        "description": row.description or "",
                        "source": row.source or "unknown",
                        "detected_at": row.detected_at.isoformat() if row.detected_at else None,
                        "metadata": row.metadata_json or {},
                        "related_market_ids": row.related_market_ids or [],
                        "market_relevance_score": row.market_relevance_score,
                    }
                    for row in world_rows
                ]
                world_update_sig = (
                    world_status.get("last_scan"),
                    len(world_signals),
                    world_signals[0]["signal_id"] if world_signals else None,
                    world_signals[0]["detected_at"] if world_signals else None,
                )
                if world_update_sig != self._last_world_update_sig:
                    self._last_world_update_sig = world_update_sig
                    await manager.broadcast(
                        {
                            "type": "events_update",
                            "data": {
                                "count": len(world_signals),
                                "signals": world_signals[:50],
                                "summary": world_stats.get("signal_breakdown", {}),
                            },
                        }
                    )

                orchestrator_sig = (
                    orchestrator_status.get("running"),
                    orchestrator_status.get("enabled"),
                    orchestrator_status.get("last_run_at"),
                    orchestrator_status.get("decisions_count"),
                    orchestrator_status.get("orders_count"),
                    orchestrator_status.get("open_orders"),
                    orchestrator_status.get("gross_exposure_usd"),
                    orchestrator_status.get("daily_pnl"),
                    orchestrator_status.get("last_error"),
                )
                if orchestrator_sig != self._last_orchestrator_status_sig:
                    self._last_orchestrator_status_sig = orchestrator_sig
                    await manager.broadcast(
                        {
                            "type": "trader_orchestrator_status",
                            "data": orchestrator_status,
                        }
                    )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("Snapshot broadcaster fallback poll failed", error=str(exc))

            await asyncio.sleep(interval_seconds)


snapshot_broadcaster = SnapshotBroadcaster()
