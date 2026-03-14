import asyncio
import json
import time
from datetime import datetime
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from config import settings
from models.database import AsyncSessionLocal, EventsSnapshot
from sqlalchemy import select
from services import shared_state
from services.news import shared_state as news_shared_state
from services.signal_bus import read_trade_signal_source_stats
from services.trader_orchestrator_state import (
    list_serialized_execution_sessions,
    list_traders,
    read_orchestrator_snapshot,
)
from services.wallet_tracker import wallet_tracker
from services.worker_state import list_worker_snapshots, summarize_worker_stats
from services.ui_lock import UI_LOCK_SESSION_COOKIE, ui_lock_service
from services.weather import shared_state as weather_shared_state
from services.ws_feeds import get_feed_manager
from utils.market_urls import serialize_opportunity_with_links


class ConnectionManager:
    """Manages WebSocket connections"""

    def __init__(self):
        self.active_connections: dict[WebSocket, dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[websocket] = {
            "channels": {"*"},
            "topics": {"*"},
            "visible": True,
            "price_market_ids": set(),
            "all_price_topics": False,
            "price_signatures": {},
        }

    def disconnect(self, websocket: WebSocket):
        self.active_connections.pop(websocket, None)

    def update_presence(self, websocket: WebSocket, *, visible: bool | None = None) -> None:
        state = self.active_connections.get(websocket)
        if state is None:
            return
        if visible is not None:
            state["visible"] = bool(visible)

    def update_channels(self, websocket: WebSocket, channels: list[str] | None) -> list[str]:
        state = self.active_connections.get(websocket)
        if state is None:
            return []
        if not isinstance(channels, list):
            state["channels"] = {"*"}
            return ["*"]
        normalized = {
            str(channel or "").strip().lower()
            for channel in channels
            if str(channel or "").strip()
        }
        state["channels"] = normalized or {"*"}
        return sorted(state["channels"])

    def update_topics(self, websocket: WebSocket, topics: list[str] | None) -> list[str]:
        state = self.active_connections.get(websocket)
        if state is None:
            return []
        if not isinstance(topics, list):
            state["topics"] = {"*"}
            state["price_market_ids"] = set()
            state["all_price_topics"] = False
            return ["*"]
        normalized = {
            str(topic or "").strip().lower()
            for topic in topics
            if str(topic or "").strip()
        }
        state["topics"] = normalized or {"*"}
        all_prices = False
        price_market_ids: set[str] = set()
        for topic in state["topics"]:
            if topic in {"prices", "prices:*"}:
                all_prices = True
                continue
            if topic.startswith("prices:"):
                market_id = str(topic.split(":", 1)[1] or "").strip().lower()
                if market_id:
                    price_market_ids.add(market_id)
        state["all_price_topics"] = all_prices
        state["price_market_ids"] = price_market_ids
        state["price_signatures"] = {}
        return sorted(state["topics"])

    @staticmethod
    def _message_channel(message_type: str) -> str | None:
        mapping = {
            "scanner_status": "core",
            "scanner_activity": "opportunities",
            "opportunities_update": "opportunities",
            "opportunity_events": "opportunities",
            "opportunity_update": "opportunities",
            "prices_update": "opportunities",
            "crypto_markets_update": "crypto",
            "weather_update": "weather",
            "weather_status": "weather",
            "news_update": "news",
            "news_workflow_status": "news",
            "news_workflow_update": "news",
            "events_status": "events",
            "events_update": "events",
            "worker_status_update": "workers",
            "signals_update": "signals",
            "wallet_trade": "wallet",
            "trader_orchestrator_status": "core",
            "trader_decision": "trading",
            "trader_order": "trading",
            "execution_order": "trading",
            "trader_event": "trading",
        }
        return mapping.get(message_type)

    @staticmethod
    def _default_message_topic(message_type: str) -> str | None:
        if message_type in {"scanner_status", "scanner_activity", "opportunities_update", "opportunity_events"}:
            return "opportunities.summary"
        if message_type == "opportunity_update":
            return "opportunities.detail"
        if message_type == "prices_update":
            return "prices"
        return None

    @staticmethod
    def _requires_visible_tab(message_type: str) -> bool:
        return message_type in {
            "scanner_activity",
            "opportunities_update",
            "opportunity_events",
            "opportunity_update",
            "prices_update",
            "crypto_markets_update",
            "weather_update",
            "news_workflow_update",
            "events_update",
        }

    @staticmethod
    def _channel_allowed(channels: set[str], message_channel: str | None) -> bool:
        if not channels or "*" in channels:
            return True
        if message_channel is None:
            return True
        return message_channel in channels

    @staticmethod
    def _topic_allowed(topics: set[str], message_topic: str | None) -> bool:
        if not topics or "*" in topics:
            return True
        if message_topic is None:
            return True
        normalized_topic = str(message_topic or "").strip().lower()
        if not normalized_topic:
            return True
        if normalized_topic in topics:
            return True
        if ":" in normalized_topic:
            prefix = normalized_topic.split(":", 1)[0]
            return f"{prefix}:*" in topics or prefix in topics
        return False

    async def broadcast(self, message: dict):
        """Send message to all connected clients"""
        if not self.active_connections:
            return

        message_json = json.dumps(message, default=str)
        disconnected: list[WebSocket] = []
        message_type = str(message.get("type", "") or "")
        channel = self._message_channel(message_type)
        topic = str(message.get("topic", "") or "").strip().lower() or self._default_message_topic(message_type)
        require_visible = self._requires_visible_tab(message_type)

        for connection, state in list(self.active_connections.items()):
            try:
                channels = state.get("channels")
                channels = channels if isinstance(channels, set) else {"*"}
                topics = state.get("topics")
                topics = topics if isinstance(topics, set) else {"*"}
                if not self._channel_allowed(channels, channel):
                    continue
                if not self._topic_allowed(topics, topic):
                    continue
                if require_visible and not bool(state.get("visible", True)):
                    continue
                await connection.send_text(message_json)
            except Exception:
                disconnected.append(connection)

        # Clean up disconnected clients
        for websocket in disconnected:
            self.disconnect(websocket)

    async def send_personal(self, websocket: WebSocket, message: dict):
        """Send message to specific client"""
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except Exception:
            self.disconnect(websocket)


# Global connection manager
manager = ConnectionManager()
_market_token_cache: dict[str, tuple[str, str]] = {}
_market_token_cache_loaded_at_mono = 0.0
_market_token_cache_updated_at: datetime | None = None
_market_token_cache_lock = asyncio.Lock()


async def _get_market_token_cache() -> dict[str, tuple[str, str]]:
    global _market_token_cache
    global _market_token_cache_loaded_at_mono
    global _market_token_cache_updated_at

    now_mono = time.monotonic()
    if _market_token_cache and (now_mono - _market_token_cache_loaded_at_mono) < 2.0:
        return _market_token_cache

    async with _market_token_cache_lock:
        now_mono = time.monotonic()
        if _market_token_cache and (now_mono - _market_token_cache_loaded_at_mono) < 2.0:
            return _market_token_cache

        async with AsyncSessionLocal() as session:
            _, markets, metadata = await shared_state.read_market_catalog(
                session,
                include_events=False,
                include_markets=True,
                validate=False,
            )
            updated_at = metadata.get("updated_at")
            if (
                _market_token_cache
                and _market_token_cache_updated_at is not None
                and updated_at is not None
                and updated_at <= _market_token_cache_updated_at
            ):
                _market_token_cache_loaded_at_mono = now_mono
                return _market_token_cache

            rebuilt: dict[str, tuple[str, str]] = {}
            for market in markets:
                if not isinstance(market, dict):
                    continue
                market_id = str(market.get("condition_id") or market.get("id") or "").strip().lower()
                raw_tokens = list(market.get("clob_token_ids") or [])
                if not market_id:
                    continue
                clean_tokens = [str(token or "").strip() for token in raw_tokens if str(token or "").strip()]
                if len(clean_tokens) < 2:
                    continue
                rebuilt[market_id] = (clean_tokens[0], clean_tokens[1])

        _market_token_cache = rebuilt
        _market_token_cache_loaded_at_mono = now_mono
        _market_token_cache_updated_at = updated_at if isinstance(updated_at, datetime) else _market_token_cache_updated_at
        return _market_token_cache


async def _price_stream_loop(websocket: WebSocket) -> None:
    interval_seconds = max(0.2, float(getattr(settings, "MARKET_DATA_PRICE_STREAM_INTERVAL_SECONDS", 1.0) or 1.0))
    all_market_cap = 250
    while True:
        state = manager.active_connections.get(websocket)
        if state is None:
            return
        if not bool(state.get("visible", True)):
            await asyncio.sleep(interval_seconds)
            continue

        market_ids: set[str] = set()
        explicit_ids = state.get("price_market_ids")
        if isinstance(explicit_ids, set):
            market_ids.update(str(mid or "").strip().lower() for mid in explicit_ids if str(mid or "").strip())

        token_cache = await _get_market_token_cache()
        if bool(state.get("all_price_topics", False)):
            market_ids.update(list(token_cache.keys())[:all_market_cap])

        if not market_ids:
            await asyncio.sleep(interval_seconds)
            continue

        requested_market_ids = sorted(mid for mid in market_ids if mid in token_cache)
        if not requested_market_ids:
            await asyncio.sleep(interval_seconds)
            continue

        token_ids = sorted(
            {
                token
                for market_id in requested_market_ids
                for token in token_cache.get(market_id, ())
                if token
            }
        )
        prices: dict[str, dict[str, Any]] = {}
        try:
            feed_manager = get_feed_manager()
            if getattr(feed_manager, "_started", False):
                for token_id in token_ids:
                    if not feed_manager.cache.is_fresh(token_id):
                        continue
                    mid = feed_manager.cache.get_mid_price(token_id)
                    if mid is None:
                        continue
                    bid_ask = feed_manager.cache.get_best_bid_ask(token_id)
                    bid = mid
                    ask = mid
                    if isinstance(bid_ask, tuple):
                        bid, ask = bid_ask
                    now_ts = time.time()
                    prices[token_id] = {
                        "mid": float(mid),
                        "bid": float(bid),
                        "ask": float(ask),
                        "ingest_ts": float(now_ts),
                        "exchange_ts": float(now_ts),
                        "sequence": 0,
                        "is_fresh": True,
                    }
        except Exception:
            prices = {}
        signatures = state.get("price_signatures")
        if not isinstance(signatures, dict):
            signatures = {}
            state["price_signatures"] = signatures

        for market_id in requested_market_ids:
            yes_token, no_token = token_cache.get(market_id, ("", ""))
            yes_price = prices.get(yes_token)
            no_price = prices.get(no_token)
            yes_mid = float(yes_price.get("mid")) if isinstance(yes_price, dict) and yes_price.get("mid") is not None else None
            no_mid = float(no_price.get("mid")) if isinstance(no_price, dict) and no_price.get("mid") is not None else None
            yes_ts = float(yes_price.get("ingest_ts")) if isinstance(yes_price, dict) and yes_price.get("ingest_ts") else None
            no_ts = float(no_price.get("ingest_ts")) if isinstance(no_price, dict) and no_price.get("ingest_ts") else None
            yes_seq = int(yes_price.get("sequence") or 0) if isinstance(yes_price, dict) else 0
            no_seq = int(no_price.get("sequence") or 0) if isinstance(no_price, dict) else 0
            signature = (yes_mid, no_mid, yes_ts, no_ts, yes_seq, no_seq)
            if signatures.get(market_id) == signature:
                continue
            signatures[market_id] = signature
            await manager.send_personal(
                websocket,
                {
                    "type": "prices_update",
                    "topic": f"prices:{market_id}",
                    "data": {
                        "market_id": market_id,
                        "yes_token_id": yes_token,
                        "no_token_id": no_token,
                        "yes_price": yes_mid,
                        "no_price": no_mid,
                        "yes_ingest_ts": yes_ts,
                        "no_ingest_ts": no_ts,
                        "yes_sequence": yes_seq,
                        "no_sequence": no_seq,
                        "is_fresh": bool(
                            isinstance(yes_price, dict)
                            and isinstance(no_price, dict)
                            and yes_price.get("is_fresh")
                            and no_price.get("is_fresh")
                        ),
                    },
                },
            )

        await asyncio.sleep(interval_seconds)


async def handle_websocket(websocket: WebSocket):
    """Main WebSocket handler"""
    await manager.connect(websocket)
    price_stream_task = asyncio.create_task(_price_stream_loop(websocket), name="ws-price-stream")

    # Send current state from the in-process runtimes plus durable projections.
    async with AsyncSessionLocal() as session:
        opportunities, status = await shared_state.read_scanner_snapshot(session)
        weather_opportunities = await weather_shared_state.get_weather_opportunities_from_db(session)
        weather_status = await weather_shared_state.get_weather_status_from_db(session)
        news_workflow_status = await news_shared_state.get_news_status_from_db(session)
        worker_statuses = await list_worker_snapshots(session, include_stats=True, stats_mode="summary")
        orchestrator_status = await read_orchestrator_snapshot(session)
        traders = await list_traders(session)
        execution_sessions = await list_serialized_execution_sessions(
            session,
            trader_id=None,
            status=None,
            limit=100,
        )
        signal_sources = await read_trade_signal_source_stats(session)
        world_snapshot = (
            (await session.execute(select(EventsSnapshot).where(EventsSnapshot.id == "latest"))).scalars().one_or_none()
        )
        events_status = {
            "status": (world_snapshot.status if world_snapshot else {}) or {},
            "stats": (world_snapshot.stats if world_snapshot else {}) or {},
            "updated_at": (
                world_snapshot.updated_at.isoformat() if world_snapshot and world_snapshot.updated_at else None
            ),
        }
    crypto_markets: list[dict[str, Any]] = []
    for worker in worker_statuses:
        if worker.get("worker_name") != "crypto":
            continue
        stats = worker.get("stats") if isinstance(worker.get("stats"), dict) else {}
        if isinstance(stats.get("markets"), list):
            crypto_markets = list(stats.get("markets") or [])
        break
    if crypto_markets:
        for worker in worker_statuses:
            if worker.get("worker_name") == "crypto":
                worker["stats"] = {"markets": crypto_markets}
                break
    if isinstance(orchestrator_status, dict):
        orchestrator_status = dict(orchestrator_status)
        orchestrator_status["stats"] = summarize_worker_stats(orchestrator_status.get("stats"))
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
                    "last_fast_scan": status.get("last_fast_scan"),
                    "last_heavy_scan": status.get("last_heavy_scan"),
                    "opportunities_count": status.get("opportunities_count", len(opportunities)),
                    "current_activity": status.get("current_activity"),
                    "lane_watchdogs": status.get("lane_watchdogs"),
                },
                "weather_status": weather_status,
                "news_workflow_status": news_workflow_status,
                "workers_status": worker_statuses,
                "trader_orchestrator_status": orchestrator_status,
                "signals_status": {"sources": signal_sources},
                "traders": traders,
                "execution_sessions": execution_sessions,
                "events_status": events_status,
                "crypto_markets": crypto_markets,
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
                subscribed_channels = manager.update_channels(websocket, message.get("channels"))
                subscribed_topics = manager.update_topics(websocket, message.get("topics"))
                visible = message.get("visible")
                if isinstance(visible, bool):
                    manager.update_presence(websocket, visible=visible)
                await manager.send_personal(
                    websocket,
                    {
                        "type": "subscribed",
                        "data": {
                            "channels": subscribed_channels,
                            "topics": subscribed_topics,
                            "visible": bool(manager.active_connections.get(websocket, {}).get("visible", True)),
                        },
                    },
                )

            elif message.get("type") == "ui_presence":
                visible = message.get("visible")
                if isinstance(visible, bool):
                    manager.update_presence(websocket, visible=visible)
                await manager.send_personal(
                    websocket,
                    {
                        "type": "presence_ack",
                        "data": {"visible": bool(manager.active_connections.get(websocket, {}).get("visible", True))},
                    },
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
    finally:
        price_stream_task.cancel()
        try:
            await price_stream_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass


async def broadcast_opportunities(opportunities):
    """Callback to broadcast new opportunities"""
    await manager.broadcast(
        {
            "type": "opportunities_update",
            "topic": "opportunities.summary",
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
    await manager.broadcast({"type": "scanner_status", "topic": "opportunities.summary", "data": status})


async def broadcast_scanner_activity(activity: str):
    """Callback to broadcast scanning activity updates (live status line)"""
    await manager.broadcast(
        {"type": "scanner_activity", "topic": "opportunities.summary", "data": {"activity": activity}}
    )


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
