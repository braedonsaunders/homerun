"""
Persistent market metadata caching service.

Provides SQL-backed persistence for market metadata and username lookups,
inspired by terauss/Polymarket-Copy-Trading-Bot which persists market cache
to disk so restarts don't require re-fetching everything.

Replaces in-memory-only caching (_market_cache, _username_cache in polymarket.py)
with a write-through strategy: updates go to both in-memory dicts (for fast O(1)
lookups on the hot path) and the SQL database (for persistence across restarts).
"""

from datetime import datetime, timedelta
from utils.utcnow import utcnow
from typing import Optional

from sqlalchemy import (
    Column,
    String,
    Boolean,
    DateTime,
    Text,
    JSON,
    Index,
    select,
    delete,
    func,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert

from models.database import (
    Base,
    AsyncSessionLocal,
    WalletActivityRollup,
    MarketConfluenceSignal,
)
from utils.logger import get_logger

logger = get_logger("market_cache")

MARKET_CACHE_HYGIENE_INTERVAL = timedelta(hours=6)
MARKET_CACHE_RETENTION_DAYS = 120
MARKET_CACHE_REFERENCE_LOOKBACK_DAYS = 45
MARKET_CACHE_WEAK_ENTRY_GRACE_DAYS = 7
MARKET_CACHE_MAX_ENTRIES_PER_SLUG = 3


# ==================== SQLAlchemy Models ====================


class CachedMarket(Base):
    """Persisted market metadata keyed by condition_id."""

    __tablename__ = "cached_markets"

    condition_id = Column(String, primary_key=True)
    question = Column(Text, nullable=True)
    slug = Column(String, nullable=True)
    group_item_title = Column(String, nullable=True)
    category = Column(String, nullable=True)
    active = Column(Boolean, nullable=True)
    extra_data = Column(JSON, nullable=True)  # Any additional metadata
    cached_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_cm_category", "category"),
        Index("idx_cm_cached_at", "cached_at"),
    )


class CachedUsername(Base):
    """Persisted username lookup keyed by lowercase wallet address."""

    __tablename__ = "cached_usernames"

    address = Column(String, primary_key=True)  # Lowercase wallet address
    username = Column(String, nullable=False)
    cached_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (Index("idx_cu_cached_at", "cached_at"),)


# ==================== Cache Service ====================


class MarketCacheService:
    """SQL-backed persistent cache for market metadata and username lookups.

    On startup, ``load_from_db`` populates in-memory dicts so every subsequent
    read is an O(1) dict lookup.  Writes go to both memory and the database
    (write-through) so that restarts never lose cached data.
    """

    def __init__(self):
        self._market_cache: dict[str, dict] = {}  # In-memory for fast access
        self._username_cache: dict[str, str] = {}  # In-memory for fast access
        self._loaded = False
        self._last_hygiene_at: Optional[datetime] = None

    @staticmethod
    def _norm(value: object) -> str:
        return str(value or "").strip().lower()

    # -------------------- Startup --------------------

    async def load_from_db(self):
        """Load all cached data from SQL into memory on startup."""
        try:
            async with AsyncSessionLocal() as session:
                # Load markets
                result = await session.execute(select(CachedMarket))
                rows = result.scalars().all()
                for row in rows:
                    self._market_cache[row.condition_id] = {
                        "question": row.question,
                        "slug": row.slug,
                        "groupItemTitle": row.group_item_title,
                        "category": row.category,
                        "active": row.active,
                        **(row.extra_data or {}),
                    }

                # Load usernames
                result = await session.execute(select(CachedUsername))
                rows = result.scalars().all()
                for row in rows:
                    self._username_cache[row.address] = row.username

            self._loaded = True
            logger.info(
                "Cache loaded from database",
                markets=len(self._market_cache),
                usernames=len(self._username_cache),
            )
            try:
                hygiene = await self.run_hygiene_if_due(force=True)
                if int(hygiene.get("markets_deleted", 0)) > 0:
                    logger.info("Market cache hygiene removed stale entries", **hygiene)
            except Exception as e:
                logger.warning("Market cache hygiene failed after load", error=str(e))
        except Exception as e:
            logger.error("Failed to load cache from database", error=str(e))
            # Service remains usable with empty in-memory cache
            self._loaded = True

    # -------------------- Market metadata --------------------

    async def get_market(self, condition_id: str) -> Optional[dict]:
        """Get cached market metadata. Returns None if not cached."""
        return self._market_cache.get(condition_id)

    async def delete_market(self, condition_id: str) -> bool:
        """Delete a market metadata entry from memory and DB."""
        if not condition_id:
            return False
        key = str(condition_id)
        key_norm = self._norm(key)
        self._market_cache.pop(key, None)
        self._market_cache.pop(key_norm, None)

        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    delete(CachedMarket).where(func.lower(CachedMarket.condition_id) == key_norm)
                )
                await session.commit()
                return bool(result.rowcount)
        except Exception as e:
            logger.error(
                "Failed to delete market cache entry",
                condition_id=key,
                error=str(e),
            )
            return False

    async def set_market(self, condition_id: str, data: dict):
        """Cache market metadata (write-through: memory + DB)."""
        # Update in-memory cache immediately
        self._market_cache[condition_id] = data

        # Persist to database
        try:
            async with AsyncSessionLocal() as session:
                now = utcnow()
                # Separate known columns from extra data
                known_keys = {
                    "question",
                    "slug",
                    "groupItemTitle",
                    "category",
                    "active",
                }
                extra = {k: v for k, v in data.items() if k not in known_keys}

                stmt = sqlite_upsert(CachedMarket).values(
                    condition_id=condition_id,
                    question=data.get("question"),
                    slug=data.get("slug"),
                    group_item_title=data.get("groupItemTitle"),
                    category=data.get("category"),
                    active=data.get("active"),
                    extra_data=extra if extra else None,
                    cached_at=now,
                    updated_at=now,
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=["condition_id"],
                    set_={
                        "question": stmt.excluded.question,
                        "slug": stmt.excluded.slug,
                        "group_item_title": stmt.excluded.group_item_title,
                        "category": stmt.excluded.category,
                        "active": stmt.excluded.active,
                        "extra_data": stmt.excluded.extra_data,
                        "updated_at": now,
                    },
                )
                await session.execute(stmt)
                await session.commit()
        except Exception as e:
            logger.error(
                "Failed to persist market cache entry",
                condition_id=condition_id,
                error=str(e),
            )

    async def bulk_set_markets(self, markets: dict[str, dict]):
        """Bulk cache market metadata."""
        if not markets:
            return

        # Update in-memory cache
        self._market_cache.update(markets)

        # Persist to database in a single transaction
        try:
            async with AsyncSessionLocal() as session:
                now = utcnow()
                known_keys = {
                    "question",
                    "slug",
                    "groupItemTitle",
                    "category",
                    "active",
                }

                for condition_id, data in markets.items():
                    extra = {k: v for k, v in data.items() if k not in known_keys}
                    stmt = sqlite_upsert(CachedMarket).values(
                        condition_id=condition_id,
                        question=data.get("question"),
                        slug=data.get("slug"),
                        group_item_title=data.get("groupItemTitle"),
                        category=data.get("category"),
                        active=data.get("active"),
                        extra_data=extra if extra else None,
                        cached_at=now,
                        updated_at=now,
                    )
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["condition_id"],
                        set_={
                            "question": stmt.excluded.question,
                            "slug": stmt.excluded.slug,
                            "group_item_title": stmt.excluded.group_item_title,
                            "category": stmt.excluded.category,
                            "active": stmt.excluded.active,
                            "extra_data": stmt.excluded.extra_data,
                            "updated_at": now,
                        },
                    )
                    await session.execute(stmt)

                await session.commit()
                logger.info("Bulk cached markets", count=len(markets))
        except Exception as e:
            logger.error("Failed to bulk persist market cache", error=str(e))

    def get_market_sync(self, condition_id: str) -> Optional[dict]:
        """Synchronous in-memory lookup (for hot path)."""
        return self._market_cache.get(condition_id)

    # -------------------- Username lookups --------------------

    async def get_username(self, address: str) -> Optional[str]:
        """Get cached username for address."""
        return self._username_cache.get(address.lower())

    async def set_username(self, address: str, username: str):
        """Cache username (write-through: memory + DB)."""
        addr_lower = address.lower()
        self._username_cache[addr_lower] = username

        try:
            async with AsyncSessionLocal() as session:
                now = utcnow()
                stmt = sqlite_upsert(CachedUsername).values(
                    address=addr_lower,
                    username=username,
                    cached_at=now,
                    updated_at=now,
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=["address"],
                    set_={
                        "username": stmt.excluded.username,
                        "updated_at": now,
                    },
                )
                await session.execute(stmt)
                await session.commit()
        except Exception as e:
            logger.error(
                "Failed to persist username cache entry",
                address=addr_lower,
                error=str(e),
            )

    async def bulk_set_usernames(self, usernames: dict[str, str]):
        """Bulk cache usernames."""
        if not usernames:
            return

        # Normalise keys to lowercase and update in-memory cache
        normalised = {addr.lower(): uname for addr, uname in usernames.items()}
        self._username_cache.update(normalised)

        try:
            async with AsyncSessionLocal() as session:
                now = utcnow()
                for addr, uname in normalised.items():
                    stmt = sqlite_upsert(CachedUsername).values(
                        address=addr,
                        username=uname,
                        cached_at=now,
                        updated_at=now,
                    )
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["address"],
                        set_={
                            "username": stmt.excluded.username,
                            "updated_at": now,
                        },
                    )
                    await session.execute(stmt)

                await session.commit()
                logger.info("Bulk cached usernames", count=len(normalised))
        except Exception as e:
            logger.error("Failed to bulk persist username cache", error=str(e))

    def get_username_sync(self, address: str) -> Optional[str]:
        """Synchronous in-memory lookup (for hot path)."""
        return self._username_cache.get(address.lower())

    # -------------------- Statistics & maintenance --------------------

    async def get_cache_stats(self) -> dict:
        """Get cache statistics (counts, oldest entry, etc.)."""
        stats: dict = {
            "markets_cached": len(self._market_cache),
            "usernames_cached": len(self._username_cache),
            "loaded": self._loaded,
        }

        try:
            async with AsyncSessionLocal() as session:
                # Oldest / newest market entry
                result = await session.execute(
                    select(
                        func.min(CachedMarket.cached_at),
                        func.max(CachedMarket.updated_at),
                        func.count(CachedMarket.condition_id),
                    )
                )
                row = result.one()
                stats["markets_db_count"] = row[2]
                stats["markets_oldest"] = row[0].isoformat() if row[0] else None
                stats["markets_newest_update"] = row[1].isoformat() if row[1] else None

                # Oldest / newest username entry
                result = await session.execute(
                    select(
                        func.min(CachedUsername.cached_at),
                        func.max(CachedUsername.updated_at),
                        func.count(CachedUsername.address),
                    )
                )
                row = result.one()
                stats["usernames_db_count"] = row[2]
                stats["usernames_oldest"] = row[0].isoformat() if row[0] else None
                stats["usernames_newest_update"] = row[1].isoformat() if row[1] else None

        except Exception as e:
            logger.error("Failed to gather cache DB stats", error=str(e))

        return stats

    async def cleanup_old_entries(self, max_age_days: int = 30):
        """Remove entries older than max_age_days from both DB and memory."""
        cutoff = utcnow() - timedelta(days=max_age_days)
        removed_markets = 0
        removed_usernames = 0

        try:
            async with AsyncSessionLocal() as session:
                # Find stale market condition_ids so we can remove from memory too
                result = await session.execute(
                    select(CachedMarket.condition_id).where(CachedMarket.updated_at < cutoff)
                )
                stale_market_ids = [row[0] for row in result.all()]

                if stale_market_ids:
                    await session.execute(delete(CachedMarket).where(CachedMarket.condition_id.in_(stale_market_ids)))
                    for cid in stale_market_ids:
                        self._market_cache.pop(cid, None)
                    removed_markets = len(stale_market_ids)

                # Find stale username addresses
                result = await session.execute(select(CachedUsername.address).where(CachedUsername.updated_at < cutoff))
                stale_addrs = [row[0] for row in result.all()]

                if stale_addrs:
                    await session.execute(delete(CachedUsername).where(CachedUsername.address.in_(stale_addrs)))
                    for addr in stale_addrs:
                        self._username_cache.pop(addr, None)
                    removed_usernames = len(stale_addrs)

                await session.commit()

            logger.info(
                "Cache cleanup complete",
                max_age_days=max_age_days,
                removed_markets=removed_markets,
                removed_usernames=removed_usernames,
            )
        except Exception as e:
            logger.error("Cache cleanup failed", error=str(e))

        return {
            "removed_markets": removed_markets,
            "removed_usernames": removed_usernames,
        }

    async def run_hygiene_if_due(
        self,
        force: bool = False,
        *,
        interval_hours: Optional[int] = None,
        retention_days: Optional[int] = None,
        reference_lookback_days: Optional[int] = None,
        weak_entry_grace_days: Optional[int] = None,
        max_entries_per_slug: Optional[int] = None,
    ) -> dict:
        """Run market cache hygiene on an interval to keep metadata clean."""
        now = utcnow()
        interval = timedelta(
            hours=max(
                1,
                int(
                    interval_hours
                    if interval_hours is not None
                    else MARKET_CACHE_HYGIENE_INTERVAL.total_seconds() // 3600
                ),
            )
        )
        if not force and self._last_hygiene_at and (now - self._last_hygiene_at) < interval:
            return {
                "status": "skipped",
                "next_due_after": (self._last_hygiene_at + interval).isoformat(),
            }
        result = await self.prune_market_metadata(
            retention_days=(int(retention_days) if retention_days is not None else MARKET_CACHE_RETENTION_DAYS),
            reference_lookback_days=(
                int(reference_lookback_days)
                if reference_lookback_days is not None
                else MARKET_CACHE_REFERENCE_LOOKBACK_DAYS
            ),
            weak_entry_grace_days=(
                int(weak_entry_grace_days) if weak_entry_grace_days is not None else MARKET_CACHE_WEAK_ENTRY_GRACE_DAYS
            ),
            max_entries_per_slug=(
                int(max_entries_per_slug) if max_entries_per_slug is not None else MARKET_CACHE_MAX_ENTRIES_PER_SLUG
            ),
        )
        self._last_hygiene_at = now
        return result

    async def prune_market_metadata(
        self,
        *,
        retention_days: int = MARKET_CACHE_RETENTION_DAYS,
        reference_lookback_days: int = MARKET_CACHE_REFERENCE_LOOKBACK_DAYS,
        weak_entry_grace_days: int = MARKET_CACHE_WEAK_ENTRY_GRACE_DAYS,
        max_entries_per_slug: int = MARKET_CACHE_MAX_ENTRIES_PER_SLUG,
    ) -> dict:
        """Prune stale or inconsistent market cache rows.

        Strategy:
        - Keep recently referenced markets (activity/confluence lookback window).
        - Remove rows with key mismatch, empty payload, or suspicious slug fanout.
        - Remove stale rows older than retention.
        """
        now = utcnow()
        stale_cutoff = now - timedelta(days=max(1, int(retention_days)))
        ref_cutoff = now - timedelta(days=max(1, int(reference_lookback_days)))
        weak_cutoff = now - timedelta(days=max(1, int(weak_entry_grace_days)))

        deleted_ids: set[str] = set()
        reasons: dict[str, int] = {}

        async with AsyncSessionLocal() as session:
            rows = (await session.execute(select(CachedMarket))).scalars().all()

            referenced_ids: set[str] = set()
            activity_refs = await session.execute(
                select(WalletActivityRollup.market_id).where(WalletActivityRollup.traded_at >= ref_cutoff).distinct()
            )
            referenced_ids.update(self._norm(mid) for mid in activity_refs.scalars().all() if mid)

            confluence_refs = await session.execute(
                select(MarketConfluenceSignal.market_id)
                .where(
                    func.coalesce(
                        MarketConfluenceSignal.last_seen_at,
                        MarketConfluenceSignal.detected_at,
                    )
                    >= ref_cutoff
                )
                .distinct()
            )
            referenced_ids.update(self._norm(mid) for mid in confluence_refs.scalars().all() if mid)

            slug_counts: dict[str, int] = {}
            for row in rows:
                slug_norm = self._norm(row.slug)
                if slug_norm:
                    slug_counts[slug_norm] = slug_counts.get(slug_norm, 0) + 1

            for row in rows:
                key = self._norm(row.condition_id)
                payload = row.extra_data if isinstance(row.extra_data, dict) else {}
                embedded_key = self._norm(payload.get("condition_id") or payload.get("conditionId"))
                question = str(row.question or "").strip()
                slug_norm = self._norm(row.slug)
                referenced = key in referenced_ids
                updated_at = row.updated_at or row.cached_at

                reason: Optional[str] = None
                if not key:
                    reason = "empty_key"
                elif embedded_key and embedded_key != key:
                    reason = "embedded_key_mismatch"
                elif not question and not slug_norm and not referenced:
                    reason = "empty_payload"
                elif slug_norm and slug_counts.get(slug_norm, 0) > max_entries_per_slug and not embedded_key:
                    reason = "suspicious_slug_collision"
                elif updated_at and updated_at < weak_cutoff and not embedded_key and not referenced:
                    reason = "missing_embedded_key"
                elif updated_at and updated_at < stale_cutoff and not referenced:
                    reason = "stale_unreferenced"

                if reason:
                    deleted_ids.add(str(row.condition_id))
                    reasons[reason] = reasons.get(reason, 0) + 1

            deleted_count = 0
            if deleted_ids:
                result = await session.execute(
                    delete(CachedMarket).where(CachedMarket.condition_id.in_(list(deleted_ids)))
                )
                deleted_count = int(result.rowcount or 0)
                await session.commit()
                for cid in deleted_ids:
                    self._market_cache.pop(cid, None)

        summary = {
            "status": "ok",
            "scanned": len(rows),
            "markets_deleted": len(deleted_ids),
            "reasons": reasons,
            "retention_days": retention_days,
            "reference_lookback_days": reference_lookback_days,
            "db_rowcount_deleted": deleted_count,
        }
        if len(deleted_ids) > 0:
            logger.info("Pruned stale market cache metadata", **summary)
        return summary


# ==================== Singleton ====================

market_cache_service = MarketCacheService()
