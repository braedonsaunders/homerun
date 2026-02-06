import uuid
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional
from sqlalchemy import Column, String, Float, Boolean, DateTime, Integer, Index
from models.database import Base, AsyncSessionLocal
from utils.logger import get_logger

logger = get_logger("live_market_detector")

LIVE_CACHE_TTL_SECONDS = 60
NON_LIVE_CACHE_TTL_SECONDS = 300
LIVE_GTD_SECONDS = 61       # Polymarket adds 60s security buffer
NON_LIVE_GTD_SECONDS = 1800  # 30 minutes for non-live markets

class MarketLiveStatus(Base):
    __tablename__ = "market_live_status"
    token_id = Column(String, primary_key=True)
    is_live = Column(Boolean, default=False)
    last_checked = Column(DateTime, default=datetime.utcnow)
    gtd_seconds = Column(Integer, default=NON_LIVE_GTD_SECONDS)
    check_count = Column(Integer, default=0)
    __table_args__ = (
        Index("idx_mls_live", "is_live"),
    )

@dataclass
class LiveStatus:
    token_id: str
    is_live: bool
    gtd_seconds: int
    cache_ttl_seconds: int
    checked_at: datetime

@dataclass
class _CacheEntry:
    status: LiveStatus
    expires_at: datetime

class LiveMarketDetector:
    def __init__(self):
        self._cache: dict[str, _CacheEntry] = {}  # token_id -> CacheEntry

    async def is_live(self, token_id: str) -> LiveStatus:
        """Check if a market is currently live. Uses TTL-differentiated cache."""
        # Check cache
        now = datetime.utcnow()
        entry = self._cache.get(token_id)
        if entry and entry.expires_at > now:
            return entry.status

        # Fetch from CLOB API
        is_live_result = False
        try:
            from services.polymarket import polymarket_client
            client = await polymarket_client._get_client()
            resp = await client.get(
                f"{polymarket_client.clob_url}/markets/{token_id}"
            )
            if resp.status_code == 200:
                data = resp.json()
                # A market is "live" if it has active orderbook AND
                # the condition has end_date_iso in the near future
                # or if the event is flagged as happening now
                is_live_result = bool(data.get("active", False) and data.get("accepting_orders", False))
                # Check for live sports/events indicators
                description = str(data.get("description", "")).lower()
                if any(kw in description for kw in ["live", "in-play", "in progress", "happening now"]):
                    is_live_result = True
        except Exception as e:
            logger.error("Failed to check live status", token_id=token_id, error=str(e))

        gtd_secs = LIVE_GTD_SECONDS if is_live_result else NON_LIVE_GTD_SECONDS
        cache_ttl = LIVE_CACHE_TTL_SECONDS if is_live_result else NON_LIVE_CACHE_TTL_SECONDS

        status = LiveStatus(
            token_id=token_id,
            is_live=is_live_result,
            gtd_seconds=gtd_secs,
            cache_ttl_seconds=cache_ttl,
            checked_at=now,
        )

        # Update cache with appropriate TTL
        self._cache[token_id] = _CacheEntry(
            status=status,
            expires_at=now + timedelta(seconds=cache_ttl),
        )

        # Persist to DB
        await self._persist_status(status)

        return status

    def get_gtd_seconds(self, token_id: str) -> int:
        """Get recommended GTD expiration. Returns non-live default if not cached."""
        entry = self._cache.get(token_id)
        if entry and entry.expires_at > datetime.utcnow():
            return entry.status.gtd_seconds
        return NON_LIVE_GTD_SECONDS

    def invalidate(self, token_id: str):
        """Force re-check on next query."""
        self._cache.pop(token_id, None)

    def get_cache_stats(self) -> dict:
        now = datetime.utcnow()
        live_count = sum(1 for e in self._cache.values() if e.status.is_live and e.expires_at > now)
        total = sum(1 for e in self._cache.values() if e.expires_at > now)
        return {"total_cached": total, "live_markets": live_count, "non_live_markets": total - live_count}

    async def _persist_status(self, status: LiveStatus):
        try:
            async with AsyncSessionLocal() as session:
                from sqlalchemy import select
                result = await session.execute(
                    select(MarketLiveStatus).where(MarketLiveStatus.token_id == status.token_id)
                )
                existing = result.scalar_one_or_none()
                if existing:
                    existing.is_live = status.is_live
                    existing.last_checked = status.checked_at
                    existing.gtd_seconds = status.gtd_seconds
                    existing.check_count = (existing.check_count or 0) + 1
                else:
                    session.add(MarketLiveStatus(
                        token_id=status.token_id,
                        is_live=status.is_live,
                        last_checked=status.checked_at,
                        gtd_seconds=status.gtd_seconds,
                        check_count=1,
                    ))
                await session.commit()
        except Exception as e:
            logger.error("Failed to persist live status", error=str(e))

live_market_detector = LiveMarketDetector()
