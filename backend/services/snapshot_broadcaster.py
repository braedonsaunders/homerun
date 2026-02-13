"""
Snapshot broadcaster.

Bridges scanner-worker DB snapshots to connected WebSocket clients in the API
process. This removes the need for aggressive frontend polling.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional

from sqlalchemy import select

from api.websocket import manager
from models.database import (
    AsyncSessionLocal,
    OpportunityEvent,
    TradeSignalSnapshot,
    TraderDecision,
    TraderEvent,
    TraderOrder,
    WorldIntelligenceSignal,
    WorldIntelligenceSnapshot,
)
from services import shared_state
from services.news import shared_state as news_shared_state
from services.trader_orchestrator_state import read_orchestrator_snapshot
from services.worker_state import list_worker_snapshots
from services.weather import shared_state as weather_shared_state
from utils.logger import get_logger
from utils.market_urls import serialize_opportunity_with_links

logger = get_logger("snapshot_broadcaster")


class SnapshotBroadcaster:
    """Poll scanner snapshot and broadcast deltas over WebSocket."""

    def __init__(self) -> None:
        self._running = False
        self._task: Optional[asyncio.Task] = None
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
        self._last_trader_decision_ts: Optional[datetime] = None
        self._last_trader_order_ts: Optional[datetime] = None
        self._last_trader_event_ts: Optional[datetime] = None

    @staticmethod
    def _opportunity_ai_signature(opportunities: list) -> int:
        """Compact signature that changes when inline AI analysis changes."""
        signature_rows: list[tuple] = []
        for opp in opportunities:
            analysis = getattr(opp, "ai_analysis", None)
            if analysis is None:
                signature_rows.append((opp.stable_id or opp.id, None, None, None))
                continue
            judged_at = (
                analysis.judged_at.isoformat()
                if getattr(analysis, "judged_at", None) is not None
                else None
            )
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

    async def start(self, interval_seconds: float = 1.0) -> None:
        """Start background poll loop (idempotent)."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(
            self._run_loop(interval_seconds=max(0.25, interval_seconds)),
            name="snapshot-broadcaster",
        )
        logger.info("Snapshot broadcaster started", interval_seconds=interval_seconds)

    async def stop(self) -> None:
        """Stop background poll loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
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
        self._last_trader_decision_ts = None
        self._last_trader_order_ts = None
        self._last_trader_event_ts = None
        logger.info("Snapshot broadcaster stopped")

    async def _run_loop(self, interval_seconds: float) -> None:
        while self._running:
            try:
                async with AsyncSessionLocal() as session:
                    opportunities = await shared_state.get_opportunities_from_db(session, None)
                    status = await shared_state.get_scanner_status_from_db(session)
                    status["opportunities_count"] = len(opportunities)
                    weather_opps = await weather_shared_state.get_weather_opportunities_from_db(session)
                    weather_status = await weather_shared_state.get_weather_status_from_db(session)
                    weather_status["opportunities_count"] = len(weather_opps)
                    news_status = await news_shared_state.get_news_status_from_db(session)
                    worker_statuses = await list_worker_snapshots(session)
                    orchestrator_status = await read_orchestrator_snapshot(session)
                    signal_rows = (
                        (
                            await session.execute(
                                select(TradeSignalSnapshot).order_by(TradeSignalSnapshot.source.asc())
                            )
                        )
                        .scalars()
                        .all()
                    )
                    world_snapshot = (
                        (
                            await session.execute(
                                select(WorldIntelligenceSnapshot).where(
                                    WorldIntelligenceSnapshot.id == "latest"
                                )
                            )
                        )
                        .scalars()
                        .one_or_none()
                    )
                    world_rows = (
                        (
                            await session.execute(
                                select(WorldIntelligenceSignal)
                                .order_by(
                                    WorldIntelligenceSignal.detected_at.desc(),
                                    WorldIntelligenceSignal.severity.desc(),
                                )
                                .limit(100)
                            )
                        )
                        .scalars()
                        .all()
                    )
                    event_query = select(OpportunityEvent).order_by(
                        OpportunityEvent.created_at.asc()
                    )
                    if self._last_event_ts is not None:
                        event_query = event_query.where(
                            OpportunityEvent.created_at > self._last_event_ts
                        )
                    event_query = event_query.limit(200)
                    event_rows = (
                        (await session.execute(event_query)).scalars().all()
                    )
                    decision_query = select(TraderDecision).order_by(
                        TraderDecision.created_at.asc()
                    )
                    if self._last_trader_decision_ts is not None:
                        decision_query = decision_query.where(
                            TraderDecision.created_at > self._last_trader_decision_ts
                        )
                    decision_query = decision_query.limit(200)
                    decision_rows = (
                        (await session.execute(decision_query)).scalars().all()
                    )

                    trader_event_query = select(TraderEvent).order_by(
                        TraderEvent.created_at.asc(),
                        TraderEvent.id.asc(),
                    )
                    if self._last_trader_event_ts is not None:
                        trader_event_query = trader_event_query.where(
                            TraderEvent.created_at > self._last_trader_event_ts
                        )
                    trader_event_query = trader_event_query.limit(200)
                    trader_event_rows = (
                        (await session.execute(trader_event_query)).scalars().all()
                    )

                    trade_query = select(TraderOrder).order_by(
                        TraderOrder.updated_at.asc(),
                        TraderOrder.id.asc(),
                    )
                    if self._last_trader_order_ts is not None:
                        trade_query = trade_query.where(
                            TraderOrder.updated_at > self._last_trader_order_ts
                        )
                    trade_query = trade_query.limit(200)
                    order_rows = (
                        (await session.execute(trade_query)).scalars().all()
                    )

                if event_rows:
                    self._last_event_ts = event_rows[-1].created_at
                    await manager.broadcast(
                        {
                            "type": "opportunity_events",
                            "data": {
                                "events": [
                                    {
                                        "id": row.id,
                                        "stable_id": row.stable_id,
                                        "run_id": row.run_id,
                                        "event_type": row.event_type,
                                        "opportunity": row.opportunity_json,
                                        "created_at": row.created_at.isoformat()
                                        if row.created_at
                                        else None,
                                    }
                                    for row in event_rows
                                ]
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
                        {"type": "scanner_activity", "data": {"activity": activity}}
                    )

                if status_sig != self._last_status_sig:
                    self._last_status_sig = status_sig
                    await manager.broadcast({"type": "scanner_status", "data": status})

                if opp_sig != self._last_opp_sig:
                    self._last_opp_sig = opp_sig
                    await manager.broadcast(
                        {
                            "type": "opportunities_update",
                            "data": {
                                "count": len(opportunities),
                                "opportunities": [
                                    serialize_opportunity_with_links(o)
                                    for o in opportunities[:50]
                                ],
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
                    await manager.broadcast(
                        {"type": "weather_status", "data": weather_status}
                    )

                if weather_opp_sig != self._last_weather_opp_sig:
                    self._last_weather_opp_sig = weather_opp_sig
                    await manager.broadcast(
                        {
                            "type": "weather_update",
                            "data": {
                                "count": len(weather_opps),
                                "opportunities": [
                                    serialize_opportunity_with_links(o)
                                    for o in weather_opps[:100]
                                ],
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
                    await manager.broadcast(
                        {"type": "news_workflow_status", "data": news_status}
                    )

                if news_update_sig != self._last_news_update_sig:
                    self._last_news_update_sig = news_update_sig
                    await manager.broadcast(
                        {
                            "type": "news_workflow_update",
                            "data": {
                                "status": news_status,
                                "findings": int(news_stats.get("findings", 0) or 0),
                                "intents": int(news_stats.get("intents", 0) or 0),
                                "pending_intents": int(
                                    news_status.get("pending_intents", 0) or 0
                                ),
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
                    await manager.broadcast(
                        {"type": "worker_status_update", "data": {"workers": worker_statuses}}
                    )

                crypto_row = next(
                    (
                        row
                        for row in worker_statuses
                        if row.get("worker_name") == "crypto"
                    ),
                    None,
                )
                crypto_stats = (
                    (crypto_row or {}).get("stats")
                    if isinstance((crypto_row or {}).get("stats"), dict)
                    else {}
                )
                crypto_markets = crypto_stats.get("markets")
                if not isinstance(crypto_markets, list):
                    crypto_markets = []

                crypto_sig = (
                    (crypto_row or {}).get("updated_at"),
                    (crypto_row or {}).get("last_run_at"),
                    len(crypto_markets),
                    (
                        (crypto_markets[0] or {}).get("id")
                        if crypto_markets
                        else None
                    ),
                    (
                        (crypto_markets[0] or {}).get("oracle_updated_at_ms")
                        if crypto_markets
                        else None
                    ),
                )
                if crypto_sig != self._last_crypto_markets_sig:
                    self._last_crypto_markets_sig = crypto_sig
                    await manager.broadcast(
                        {
                            "type": "crypto_markets_update",
                            "data": {"markets": crypto_markets},
                        }
                    )

                signal_sources = [
                    {
                        "source": row.source,
                        "pending_count": int(row.pending_count or 0),
                        "selected_count": int(row.selected_count or 0),
                        "submitted_count": int(row.submitted_count or 0),
                        "executed_count": int(row.executed_count or 0),
                        "skipped_count": int(row.skipped_count or 0),
                        "expired_count": int(row.expired_count or 0),
                        "failed_count": int(row.failed_count or 0),
                        "latest_signal_at": row.latest_signal_at.isoformat()
                        if row.latest_signal_at
                        else None,
                        "updated_at": row.updated_at.isoformat()
                        if row.updated_at
                        else None,
                    }
                    for row in signal_rows
                ]
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
                    await manager.broadcast(
                        {"type": "signals_update", "data": {"sources": signal_sources}}
                    )

                world_status = {}
                world_stats = {}
                world_updated_at = None
                if world_snapshot is not None:
                    if isinstance(world_snapshot.status, dict):
                        world_status = dict(world_snapshot.status)
                    if isinstance(world_snapshot.stats, dict):
                        world_stats = dict(world_snapshot.stats)
                        world_stats.pop("runtime_state", None)
                    world_updated_at = (
                        world_snapshot.updated_at.isoformat()
                        if world_snapshot.updated_at
                        else None
                    )
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
                            "type": "world_intelligence_status",
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
                        "detected_at": row.detected_at.isoformat()
                        if row.detected_at
                        else None,
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
                            "type": "world_intelligence_update",
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

                if decision_rows:
                    self._last_trader_decision_ts = decision_rows[-1].created_at
                    for row in decision_rows:
                        await manager.broadcast(
                            {
                                "type": "trader_decision",
                                "data": {
                                    "id": row.id,
                                    "trader_id": row.trader_id,
                                    "signal_id": row.signal_id,
                                    "source": row.source,
                                    "strategy_key": row.strategy_key,
                                    "decision": row.decision,
                                    "reason": row.reason,
                                    "score": row.score,
                                    "event_id": row.event_id,
                                    "trace_id": row.trace_id,
                                    "checks_summary": row.checks_summary_json or {},
                                    "risk_snapshot": row.risk_snapshot_json or {},
                                    "payload": row.payload_json or {},
                                    "created_at": row.created_at.isoformat()
                                    if row.created_at
                                    else None,
                                },
                            }
                        )

                if trader_event_rows:
                    self._last_trader_event_ts = trader_event_rows[-1].created_at
                    for row in trader_event_rows:
                        await manager.broadcast(
                            {
                                "type": "trader_event",
                                "data": {
                                    "id": row.id,
                                    "trader_id": row.trader_id,
                                    "event_type": row.event_type,
                                    "severity": row.severity,
                                    "source": row.source,
                                    "operator": row.operator,
                                    "message": row.message,
                                    "trace_id": row.trace_id,
                                    "payload": row.payload_json or {},
                                    "created_at": row.created_at.isoformat()
                                    if row.created_at
                                    else None,
                                },
                            }
                        )

                if order_rows:
                    latest_order = order_rows[-1]
                    self._last_trader_order_ts = (
                        latest_order.updated_at or latest_order.created_at
                    )
                    for row in order_rows:
                        await manager.broadcast(
                            {
                                "type": "trader_order",
                                "data": {
                                    "id": row.id,
                                    "trader_id": row.trader_id,
                                    "signal_id": row.signal_id,
                                    "decision_id": row.decision_id,
                                    "source": row.source,
                                    "market_id": row.market_id,
                                    "market_question": row.market_question,
                                    "direction": row.direction,
                                    "status": row.status,
                                    "mode": row.mode,
                                    "notional_usd": row.notional_usd,
                                    "entry_price": row.entry_price,
                                    "effective_price": row.effective_price,
                                    "edge_percent": row.edge_percent,
                                    "confidence": row.confidence,
                                    "actual_profit": row.actual_profit,
                                    "reason": row.reason,
                                    "error_message": row.error_message,
                                    "event_id": row.event_id,
                                    "trace_id": row.trace_id,
                                    "payload": row.payload_json or {},
                                    "created_at": row.created_at.isoformat()
                                    if row.created_at
                                    else None,
                                    "executed_at": row.executed_at.isoformat()
                                    if row.executed_at
                                    else None,
                                    "updated_at": row.updated_at.isoformat()
                                    if row.updated_at
                                    else None,
                                },
                            }
                        )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("Snapshot broadcaster poll failed", error=str(exc))

            await asyncio.sleep(interval_seconds)


snapshot_broadcaster = SnapshotBroadcaster()
