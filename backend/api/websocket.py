from fastapi import WebSocket, WebSocketDisconnect
from typing import Set
import json

from models.database import AsyncSessionLocal, EventsSnapshot
from sqlalchemy import select
from services import shared_state
from services.news import shared_state as news_shared_state
from services.trader_orchestrator_state import (
    list_serialized_execution_sessions,
    list_traders,
    read_orchestrator_snapshot,
)
from services.wallet_tracker import wallet_tracker
from services.worker_state import list_worker_snapshots, read_worker_snapshot
from services.ui_lock import UI_LOCK_SESSION_COOKIE, ui_lock_service
from services.weather import shared_state as weather_shared_state
from utils.market_urls import serialize_opportunity_with_links


class ConnectionManager:
    """Manages WebSocket connections"""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: dict):
        """Send message to all connected clients"""
        if not self.active_connections:
            return

        message_json = json.dumps(message, default=str)
        disconnected = set()

        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception:
                disconnected.add(connection)

        # Clean up disconnected clients
        self.active_connections -= disconnected

    async def send_personal(self, websocket: WebSocket, message: dict):
        """Send message to specific client"""
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except Exception:
            self.disconnect(websocket)


# Global connection manager
manager = ConnectionManager()


async def handle_websocket(websocket: WebSocket):
    """Main WebSocket handler"""
    await manager.connect(websocket)

    # Send current state (from DB snapshot)
    async with AsyncSessionLocal() as session:
        opportunities, status = await shared_state.read_scanner_snapshot(session)
        weather_opportunities = await weather_shared_state.get_weather_opportunities_from_db(session)
        weather_status = await weather_shared_state.get_weather_status_from_db(session)
        news_workflow_status = await news_shared_state.get_news_status_from_db(session)
        worker_statuses = await list_worker_snapshots(session, include_stats=False)
        crypto_snapshot = await read_worker_snapshot(session, "crypto")
        orchestrator_status = await read_orchestrator_snapshot(session)
        traders = await list_traders(session)
        execution_sessions = await list_serialized_execution_sessions(
            session,
            trader_id=None,
            status=None,
            limit=100,
        )
        world_snapshot = (
            (await session.execute(select(EventsSnapshot).where(EventsSnapshot.id == "latest"))).scalars().one_or_none()
        )
    crypto_markets = []
    crypto_stats = crypto_snapshot.get("stats")
    if isinstance(crypto_stats, dict):
        raw_markets = crypto_stats.get("markets")
        if isinstance(raw_markets, list):
            crypto_markets = raw_markets
    if crypto_markets:
        for worker in worker_statuses:
            if worker.get("worker_name") == "crypto":
                worker["stats"] = {"markets": crypto_markets}
                break
    await manager.send_personal(
        websocket,
        {
            "type": "init",
            "data": {
                "opportunities": [serialize_opportunity_with_links(o) for o in opportunities[:20]],
                "weather_opportunities": [serialize_opportunity_with_links(o) for o in weather_opportunities[:20]],
                "scanner_status": {
                    "running": status.get("running", False),
                    "last_scan": status.get("last_scan"),
                },
                "weather_status": weather_status,
                "news_workflow_status": news_workflow_status,
                "workers_status": worker_statuses,
                "trader_orchestrator_status": orchestrator_status,
                "traders": traders,
                "execution_sessions": execution_sessions,
                "events_status": {
                    "status": (world_snapshot.status if world_snapshot else {}) or {},
                    "stats": (world_snapshot.stats if world_snapshot else {}) or {},
                    "updated_at": (
                        world_snapshot.updated_at.isoformat() if world_snapshot and world_snapshot.updated_at else None
                    ),
                },
            },
        },
    )

    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            token = websocket.cookies.get(UI_LOCK_SESSION_COOKIE)
            unlocked = await ui_lock_service.is_token_unlocked(token)
            if not unlocked:
                status = await ui_lock_service.status(token)
                if status.get("enabled"):
                    await manager.send_personal(
                        websocket,
                        {
                            "type": "ui_locked",
                            "data": {"message": "UI lock is active.", "ui_lock": status},
                        },
                    )
                    await websocket.close(code=4403)
                    break
            message = json.loads(data)

            # Handle different message types
            if message.get("type") == "subscribe":
                await manager.send_personal(
                    websocket,
                    {"type": "subscribed", "data": message.get("channels", [])},
                )

            elif message.get("type") == "ping":
                await manager.send_personal(websocket, {"type": "pong"})

            elif message.get("type") == "scan":
                # Request one scan; worker will run it
                async with AsyncSessionLocal() as session:
                    await shared_state.request_one_scan(session)
                async with AsyncSessionLocal() as session:
                    opportunities, _ = await shared_state.read_scanner_snapshot(session)
                await manager.send_personal(
                    websocket,
                    {
                        "type": "scan_requested",
                        "data": {
                            "message": "Scan requested; results will update when worker runs.",
                            "current_count": len(opportunities),
                        },
                    },
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


async def broadcast_opportunities(opportunities):
    """Callback to broadcast new opportunities"""
    await manager.broadcast(
        {
            "type": "opportunities_update",
            "data": {
                "count": len(opportunities),
                "opportunities": [serialize_opportunity_with_links(o) for o in opportunities[:20]],
            },
        }
    )


async def broadcast_wallet_trade(trade):
    """Callback to broadcast new wallet trades"""
    await manager.broadcast({"type": "wallet_trade", "data": trade})


async def broadcast_scanner_status(status):
    """Callback to broadcast scanner status changes"""
    await manager.broadcast({"type": "scanner_status", "data": status})


async def broadcast_scanner_activity(activity: str):
    """Callback to broadcast scanning activity updates (live status line)"""
    await manager.broadcast({"type": "scanner_activity", "data": {"activity": activity}})


async def broadcast_news_update(article_count: int):
    """Callback to broadcast new news articles arriving"""
    await manager.broadcast({"type": "news_update", "data": {"new_count": article_count}})


async def broadcast_crypto_markets(markets_data: list[dict]):
    """Broadcast live crypto market data to all connected clients."""
    await manager.broadcast({"type": "crypto_markets_update", "data": {"markets": markets_data}})


async def broadcast_weather_update(opportunities: list[dict], status: dict):
    """Broadcast weather workflow opportunity updates to all clients."""
    await manager.broadcast(
        {
            "type": "weather_update",
            "data": {
                "count": len(opportunities),
                "opportunities": opportunities[:100],
                "status": status,
            },
        }
    )


async def broadcast_weather_status(status: dict):
    """Broadcast weather workflow status changes to all clients."""
    await manager.broadcast({"type": "weather_status", "data": status})


async def broadcast_events_update(signals: list[dict], summary: dict):
    """Broadcast events signal updates to all clients."""
    await manager.broadcast(
        {
            "type": "events_update",
            "data": {
                "count": len(signals),
                "signals": signals[:50],
                "summary": summary,
            },
        }
    )


async def broadcast_events_status(status: dict):
    """Broadcast events worker status changes."""
    await manager.broadcast({"type": "events_status", "data": status})


# Register callbacks (scanner runs in worker process; no scanner callbacks here)
# Frontend gets opportunities/status via polling or init; or add API polling→broadcast later
wallet_tracker.add_callback(broadcast_wallet_trade)
