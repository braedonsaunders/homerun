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
from models.database import AsyncSessionLocal, OpportunityEvent
from services import shared_state
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
        logger.info("Snapshot broadcaster stopped")

    async def _run_loop(self, interval_seconds: float) -> None:
        while self._running:
            try:
                async with AsyncSessionLocal() as session:
                    opportunities, status = await shared_state.read_scanner_snapshot(session)
                    weather_opps, weather_status = await weather_shared_state.read_weather_snapshot(session)
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

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("Snapshot broadcaster poll failed", error=str(exc))

            await asyncio.sleep(interval_seconds)


snapshot_broadcaster = SnapshotBroadcaster()
