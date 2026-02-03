from fastapi import WebSocket, WebSocketDisconnect
from typing import Set
import json
import asyncio

from services import scanner, wallet_tracker


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

    # Send current state
    await manager.send_personal(websocket, {
        "type": "init",
        "data": {
            "opportunities": [
                o.model_dump() for o in scanner.get_opportunities()[:20]
            ],
            "scanner_status": {
                "running": scanner.is_running,
                "last_scan": scanner.last_scan
            }
        }
    })

    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle different message types
            if message.get("type") == "subscribe":
                # Client wants to subscribe to specific updates
                await manager.send_personal(websocket, {
                    "type": "subscribed",
                    "data": message.get("channels", [])
                })

            elif message.get("type") == "ping":
                await manager.send_personal(websocket, {"type": "pong"})

            elif message.get("type") == "scan":
                # Trigger manual scan
                opportunities = await scanner.scan_once()
                await manager.send_personal(websocket, {
                    "type": "scan_complete",
                    "data": {
                        "count": len(opportunities),
                        "opportunities": [o.model_dump() for o in opportunities[:20]]
                    }
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


async def broadcast_opportunities(opportunities):
    """Callback to broadcast new opportunities"""
    await manager.broadcast({
        "type": "opportunities_update",
        "data": {
            "count": len(opportunities),
            "opportunities": [o.model_dump() for o in opportunities[:20]]
        }
    })


async def broadcast_wallet_trade(trade):
    """Callback to broadcast new wallet trades"""
    await manager.broadcast({
        "type": "wallet_trade",
        "data": trade
    })


# Register callbacks
scanner.add_callback(broadcast_opportunities)
wallet_tracker.add_callback(broadcast_wallet_trade)
