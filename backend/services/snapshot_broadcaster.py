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
    AutoTraderDecision,
    AutoTraderTrade,
    OpportunityEvent,
    TradeSignalSnapshot,
)
from services.autotrader_state import read_autotrader_snapshot
from services import shared_state
from services.news import shared_state as news_shared_state
from services.worker_state import list_worker_snapshots
from services.weather import shared_state as weather_shared_state
from utils.logger import get_logger

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
        self._last_signals_sig: Optional[tuple] = None
        self._last_autotrader_status_sig: Optional[tuple] = None
        self._last_autotrader_decision_ts: Optional[datetime] = None
        self._last_autotrader_trade_ts: Optional[datetime] = None

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
        self._last_signals_sig = None
        self._last_autotrader_status_sig = None
        self._last_autotrader_decision_ts = None
        self._last_autotrader_trade_ts = None
        logger.info("Snapshot broadcaster stopped")

    async def _run_loop(self, interval_seconds: float) -> None:
        while self._running:
            try:
                async with AsyncSessionLocal() as session:
                    opportunities, status = await shared_state.read_scanner_snapshot(session)
                    weather_opps, weather_status = await weather_shared_state.read_weather_snapshot(session)
                    news_status = await news_shared_state.get_news_status_from_db(session)
                    worker_statuses = await list_worker_snapshots(session)
                    autotrader_status = await read_autotrader_snapshot(session)
                    signal_rows = (
                        (
                            await session.execute(
                                select(TradeSignalSnapshot).order_by(TradeSignalSnapshot.source.asc())
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
                    decision_query = select(AutoTraderDecision).order_by(
                        AutoTraderDecision.created_at.asc()
                    )
                    if self._last_autotrader_decision_ts is not None:
                        decision_query = decision_query.where(
                            AutoTraderDecision.created_at > self._last_autotrader_decision_ts
                        )
                    decision_query = decision_query.limit(200)
                    decision_rows = (
                        (await session.execute(decision_query)).scalars().all()
                    )

                    trade_query = select(AutoTraderTrade).order_by(
                        AutoTraderTrade.created_at.asc()
                    )
                    if self._last_autotrader_trade_ts is not None:
                        trade_query = trade_query.where(
                            AutoTraderTrade.created_at > self._last_autotrader_trade_ts
                        )
                    trade_query = trade_query.limit(200)
                    trade_rows = (
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
                opp_sig = (
                    status.get("last_scan"),
                    len(opportunities),
                    first_id,
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
                                    o.model_dump(mode="json") for o in opportunities[:50]
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
                                    o.model_dump(mode="json") for o in weather_opps[:100]
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

                autotrader_sig = (
                    autotrader_status.get("running"),
                    autotrader_status.get("enabled"),
                    autotrader_status.get("last_run_at"),
                    autotrader_status.get("signals_seen"),
                    autotrader_status.get("signals_selected"),
                    autotrader_status.get("decisions_count"),
                    autotrader_status.get("trades_count"),
                    autotrader_status.get("open_positions"),
                    autotrader_status.get("last_error"),
                )
                if autotrader_sig != self._last_autotrader_status_sig:
                    self._last_autotrader_status_sig = autotrader_sig
                    await manager.broadcast(
                        {"type": "autotrader_status", "data": autotrader_status}
                    )

                if decision_rows:
                    self._last_autotrader_decision_ts = decision_rows[-1].created_at
                    for row in decision_rows:
                        await manager.broadcast(
                            {
                                "type": "autotrader_decision",
                                "data": {
                                    "id": row.id,
                                    "signal_id": row.signal_id,
                                    "source": row.source,
                                    "decision": row.decision,
                                    "reason": row.reason,
                                    "score": row.score,
                                    "payload": row.payload_json or {},
                                    "created_at": row.created_at.isoformat()
                                    if row.created_at
                                    else None,
                                },
                            }
                        )

                if trade_rows:
                    self._last_autotrader_trade_ts = trade_rows[-1].created_at
                    for row in trade_rows:
                        await manager.broadcast(
                            {
                                "type": "autotrader_trade",
                                "data": {
                                    "id": row.id,
                                    "signal_id": row.signal_id,
                                    "source": row.source,
                                    "market_id": row.market_id,
                                    "direction": row.direction,
                                    "status": row.status,
                                    "mode": row.mode,
                                    "notional_usd": row.notional_usd,
                                    "entry_price": row.entry_price,
                                    "effective_price": row.effective_price,
                                    "created_at": row.created_at.isoformat()
                                    if row.created_at
                                    else None,
                                    "executed_at": row.executed_at.isoformat()
                                    if row.executed_at
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
