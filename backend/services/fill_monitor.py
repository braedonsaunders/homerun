"""
Read-Only Fill Monitor

Monitors the user's own order fills without executing any trades.
Zero risk, minimal credentials needed. Useful for debugging execution
quality and monitoring live orders.

Inspired by terauss trade_monitor binary.
"""

import uuid
import asyncio
from datetime import datetime
from utils.utcnow import utcnow
from dataclasses import dataclass
from typing import Callable
from sqlalchemy import Column, String, Float, DateTime, Index, select, func
from models.database import Base, AsyncSessionLocal
from utils.logger import get_logger

logger = get_logger("fill_monitor")


class FillEvent(Base):
    """Record of an order fill detected by the monitor."""

    __tablename__ = "fill_events"

    id = Column(String, primary_key=True)
    order_id = Column(String, nullable=True)
    token_id = Column(String, nullable=True)
    side = Column(String, nullable=True)
    price = Column(Float, nullable=True)
    size = Column(Float, nullable=True)
    fee = Column(Float, nullable=True)
    status = Column(String, nullable=True)  # "filled", "partially_filled"
    fill_percent = Column(Float, nullable=True)
    detected_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_fe_token", "token_id"),
        Index("idx_fe_detected", "detected_at"),
    )


@dataclass
class FillInfo:
    order_id: str
    token_id: str
    side: str
    price: float
    size_filled: float
    size_requested: float
    fill_percent: float
    fee: float
    detected_at: datetime


class FillMonitor:
    """Monitors own order fills using WS wallet events + HTTP polling fallback.

    Primary detection: Polygon WS monitor detects OrderFilled events for our
    wallet in real-time (<1s). When a fill is detected, ``_on_ws_fill`` triggers
    an immediate fill check.

    Fallback: HTTP polling every 30s catches any fills missed by the WS path
    (e.g. during WS reconnection windows).
    """

    def __init__(self):
        self._running = False
        self._poll_interval = 30  # seconds (fallback; WS provides real-time)
        self._callbacks: list[Callable] = []
        self._known_fills: set[str] = set()  # order IDs already processed
        self._ws_registered = False

    def add_callback(self, callback: Callable):
        self._callbacks.append(callback)

    async def start(self):
        if self._running:
            return
        self._running = True
        logger.info(
            "Fill monitor started (WS-primary, HTTP-fallback at %ds)",
            self._poll_interval,
        )
        # Register with wallet WS monitor for real-time fill detection
        self._register_ws_monitor()
        asyncio.create_task(self._poll_loop())

    def _register_ws_monitor(self):
        """Hook into the wallet WS monitor so fills on our wallet are instant."""
        if self._ws_registered:
            return
        try:
            from services.trading import trading_service
            from services.wallet_ws_monitor import wallet_ws_monitor

            if trading_service.is_ready():
                wallet_addr = getattr(trading_service, "_wallet_address", None)
                if wallet_addr:
                    wallet_ws_monitor.add_wallet(wallet_addr)
                    wallet_ws_monitor.add_callback(self._on_ws_fill)
                    self._ws_registered = True
                    logger.info(
                        "Fill monitor registered with WS wallet monitor",
                        wallet=wallet_addr,
                    )
        except Exception as e:
            logger.debug("WS fill monitor registration skipped: %s", e)

    async def _on_ws_fill(self, event):
        """Callback from WS wallet monitor — triggers immediate fill check."""
        try:
            await self._check_fills()
        except Exception as e:
            logger.error("WS-triggered fill check error", error=str(e))

    def stop(self):
        self._running = False
        logger.info("Fill monitor stopped")

    async def _poll_loop(self):
        while self._running:
            try:
                # Retry WS registration if not done yet
                if not self._ws_registered:
                    self._register_ws_monitor()
                await self._check_fills()
            except Exception as e:
                logger.error("Fill monitor error", error=str(e))
            await asyncio.sleep(self._poll_interval)

    async def _check_fills(self):
        """Check for new fills on our orders."""
        try:
            from services.trading import trading_service

            if not trading_service.is_ready():
                return

            orders = await trading_service.get_open_orders()
            for order in orders:
                if order.filled_size > 0 and order.id not in self._known_fills:
                    fill = FillInfo(
                        order_id=order.id,
                        token_id=order.token_id,
                        side=order.side.value,
                        price=order.average_fill_price or order.price,
                        size_filled=order.filled_size,
                        size_requested=order.size,
                        fill_percent=(order.filled_size / order.size * 100)
                        if order.size > 0
                        else 0,
                        fee=0.0,
                        detected_at=utcnow(),
                    )

                    logger.info(
                        "Fill detected",
                        order_id=order.id,
                        token_id=order.token_id,
                        fill_percent=f"{fill.fill_percent:.1f}%",
                        price=fill.price,
                    )

                    # Persist
                    await self._persist_fill(fill)

                    # Notify callbacks
                    for cb in self._callbacks:
                        try:
                            if asyncio.iscoroutinefunction(cb):
                                await cb(fill)
                            else:
                                cb(fill)
                        except Exception as e:
                            logger.error("Fill callback error", error=str(e))

                    if order.filled_size >= order.size:
                        self._known_fills.add(order.id)
        except Exception as e:
            logger.error("Check fills error", error=str(e))

    async def _persist_fill(self, fill: FillInfo):
        try:
            async with AsyncSessionLocal() as session:
                session.add(
                    FillEvent(
                        id=str(uuid.uuid4()),
                        order_id=fill.order_id,
                        token_id=fill.token_id,
                        side=fill.side,
                        price=fill.price,
                        size=fill.size_filled,
                        fee=fill.fee,
                        status="filled"
                        if fill.fill_percent >= 99.9
                        else "partially_filled",
                        fill_percent=fill.fill_percent,
                    )
                )
                await session.commit()
        except Exception as e:
            logger.error("Failed to persist fill event", error=str(e))

    async def get_recent_fills(self, limit: int = 50) -> list[dict]:
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(FillEvent)
                    .order_by(FillEvent.detected_at.desc())
                    .limit(limit)
                )
                return [
                    {
                        "order_id": f.order_id,
                        "token_id": f.token_id,
                        "side": f.side,
                        "price": f.price,
                        "size": f.size,
                        "fill_percent": f.fill_percent,
                        "detected_at": f.detected_at.isoformat(),
                    }
                    for f in result.scalars().all()
                ]
        except Exception:
            return []

    async def get_stats(self) -> dict:
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(
                        func.count(FillEvent.id),
                        func.avg(FillEvent.fill_percent),
                        func.avg(FillEvent.price),
                    )
                )
                row = result.one()
                return {
                    "total_fills": row[0] or 0,
                    "avg_fill_percent": round(row[1], 2) if row[1] else 0,
                    "avg_price": round(row[2], 4) if row[2] else 0,
                }
        except Exception:
            return {}


fill_monitor = FillMonitor()
