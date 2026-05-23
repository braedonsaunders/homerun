"""
Persistent market metadata caching service.

Provides SQL-backed persistence for market metadata and username lookups,
inspired by terauss/Polymarket-Copy-Trading-Bot which persists market cache
to disk so restarts don't require re-fetching everything.

Replaces in-memory-only caching (_market_cache, _username_cache in polymarket.py)
with a write-through strategy: updates go to both in-memory dicts (for fast O(1)
lookups on the hot path) and the SQL database (for persistence across restarts).
"""

import asyncio
import os
from datetime import datetime, timedelta, timezone
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
    text,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import DBAPIError

from models.database import (
    Base,
    AsyncSessionLocal,
    WalletActivityRollup,
    MarketConfluenceSignal,
)
from services.live_pressure import is_db_pressure_active
from utils.logger import get_logger
from utils.retry import run_with_db_retries

logger = get_logger("market_cache")

MARKET_CACHE_HYGIENE_INTERVAL = timedelta(hours=6)
MARKET_CACHE_RETENTION_DAYS = 120
MARKET_CACHE_REFERENCE_LOOKBACK_DAYS = 45
MARKET_CACHE_WEAK_ENTRY_GRACE_DAYS = 7
MARKET_CACHE_MAX_ENTRIES_PER_SLUG = 3


def _utcnow_naive() -> datetime:
    return utcnow().replace(tzinfo=None)


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
    cached_at = Column(DateTime, default=_utcnow_naive)
    updated_at = Column(DateTime, default=_utcnow_naive, onupdate=_utcnow_naive)

    __table_args__ = (
        Index("idx_cm_category", "category"),
        Index("idx_cm_cached_at", "cached_at"),
        {"prefixes": ["UNLOGGED"]},  # market cache; see migration 202605230001
    )


class CachedUsername(Base):
    """Persisted username lookup keyed by lowercase wallet address."""

    __tablename__ = "cached_usernames"

    address = Column(String, primary_key=True)  # Lowercase wallet address
    username = Column(String, nullable=False)
    cached_at = Column(DateTime, default=_utcnow_naive)
    updated_at = Column(DateTime, default=_utcnow_naive, onupdate=_utcnow_naive)

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
        self._load_lock = asyncio.Lock()
        self._load_task: Optional[asyncio.Task] = None
        self._load_started_at: Optional[datetime] = None
        self._load_finished_at: Optional[datetime] = None
        self._load_error: Optional[str] = None

    def start_background_load(self) -> Optional[asyncio.Task]:
        if self._loaded:
            return None
        if not self._bulk_load_enabled():
            return None
        task = self._load_task
        if task is not None and not task.done():
            return task

        async def _runner() -> None:
            await self.load_from_db()

        task = asyncio.create_task(_runner(), name="market-cache-load")
        self._load_task = task

        def _clear(ref: asyncio.Task) -> None:
            if self._load_task is ref:
                self._load_task = None

        task.add_done_callback(_clear)
        return task

    @staticmethod
    def _norm(value: object) -> str:
        return str(value or "").strip().lower()

    @staticmethod
    def _bulk_load_enabled() -> bool:
        if os.environ.get("HOMERUN_PROCESS_ROLE") != "worker":
            return True
        return os.environ.get("HOMERUN_WORKER_PLANE", "").strip().lower() in {"trading", "all"}

    # -------------------- Startup --------------------

    async def load_from_db(self):
        """Load all cached data from SQL into memory on startup."""
        if self._loaded:
            return
        existing_task = self._load_task
        current_task = asyncio.current_task()
        if existing_task is not None and existing_task is not current_task and not existing_task.done():
            await existing_task
            return

        async with self._load_lock:
            if self._loaded:
                return
            self._load_started_at = utcnow()
            self._load_finished_at = None
            self._load_error = None
            try:
                next_market_cache: dict[str, dict] = {}
                next_username_cache: dict[str, str] = {}
                market_count = 0
                username_count = 0

                async with AsyncSessionLocal() as session:
                    # ``cached_markets`` is ~65K rows / ~300MB and the
                    # streaming SELECT that hydrates the in-memory cache
                    # routinely exceeds the regular pool's 30s
                    # statement_timeout under live load — observed as
                    # "Connection held for 31.2s ... task=market-cache-
                    # load" + "Intent runtime initialization failed"
                    # right after every restart.  This is a one-time
                    # startup load, not a hot-path query, so a longer
                    # cap is appropriate.  ``SET LOCAL`` only applies
                    # to this transaction's connection — won't bleed
                    # into the rest of the worker.
                    await session.execute(text("SET LOCAL statement_timeout = '300000'"))
                    market_stream = await session.stream_scalars(
                        select(CachedMarket).execution_options(yield_per=1000)
                    )
                    async for row in market_stream:
                        key = self._norm(row.condition_id)
                        if not key:
                            continue
                        next_market_cache[key] = {
                            "question": row.question,
                            "slug": row.slug,
                            "groupItemTitle": row.group_item_title,
                            "category": row.category,
                            "active": row.active,
                            **(row.extra_data or {}),
                        }
                        market_count += 1
                        if market_count % 1000 == 0:
                            await asyncio.sleep(0)

                    username_stream = await session.stream_scalars(
                        select(CachedUsername).execution_options(yield_per=1000)
                    )
                    async for row in username_stream:
                        next_username_cache[row.address] = row.username
                        username_count += 1
                        if username_count % 1000 == 0:
                            await asyncio.sleep(0)

                self._market_cache = next_market_cache
                self._username_cache = next_username_cache
                self._loaded = True
                self._load_finished_at = utcnow()
                logger.info(
                    "Cache loaded from database",
                    markets=len(self._market_cache),
                    usernames=len(self._username_cache),
                )
                # Hygiene used to run synchronously here with ``force=True``,
                # which scheduled a ``SELECT DISTINCT market_id FROM
                # wallet_activity_rollups`` that routinely blew the 30s
                # statement_timeout at startup and blocked the trading
                # plane's intent_runtime hydrate.  Defer it to a
                # background task on the natural ``MARKET_CACHE_HYGIENE_
                # INTERVAL`` cadence — the first periodic run cleans
                # stale entries without paying the latency tax on boot.
                async def _run_initial_hygiene() -> None:
                    try:
                        hygiene = await self.run_hygiene_if_due(force=False)
                        if int(hygiene.get("markets_deleted", 0)) > 0:
                            logger.info("Market cache hygiene removed stale entries", **hygiene)
                    except Exception as exc:
                        logger.warning("Market cache hygiene failed after load", error=str(exc))

                asyncio.create_task(_run_initial_hygiene(), name="market-cache-hygiene-bg")
            except Exception as e:
                self._load_error = str(e)
                self._load_finished_at = utcnow()
                logger.error("Failed to load cache from database", error=str(e))
                # Service remains usable with empty in-memory cache
                self._loaded = True

    # -------------------- Market metadata --------------------

    async def get_market(self, condition_id: str) -> Optional[dict]:
        """Get cached market metadata. Returns None if not cached."""
        key = self._norm(condition_id)
        if not key:
            return None
        cached = self._market_cache.get(key)
        if cached is not None or self._loaded:
            return cached
        async with AsyncSessionLocal() as session:
            row = await session.get(CachedMarket, key)
        if row is None:
            return None
        payload = {
            "question": row.question,
            "slug": row.slug,
            "groupItemTitle": row.group_item_title,
            "category": row.category,
            "active": row.active,
            **(row.extra_data or {}),
        }
        self._market_cache[key] = payload
        return payload

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
            # Classify poisoned-connection / pool-pressure transients —
            # observed during the 2026-05-09 13:31:46 cascade as
            # "cannot switch to state 11; another operation (2) is in
            # progress". The in-memory cache pop has already happened,
            # so the operator-visible behavior (next set/get re-populates)
            # is unchanged. Don't ERROR-log a transient that has nothing
            # to do with this method.
            err_text = str(e).lower()
            err_type = type(e).__name__.lower()
            is_transient_db = (
                "cannot switch to state" in err_text
                or "connection is closed" in err_text
                or "another operation" in err_text
                or "connectiondoesnotexisterror" in err_type
                or "interfaceerror" in err_type
            )
            if is_transient_db:
                logger.info(
                    "Market cache DB delete skipped under DB pressure",
                    condition_id=key,
                    error_type=type(e).__name__,
                )
            else:
                logger.error(
                    "Failed to delete market cache entry",
                    condition_id=key,
                    error=str(e),
                )
            return False

    async def set_market(self, condition_id: str, data: dict):
        """Cache market metadata (write-through: memory + DB)."""
        key = self._norm(condition_id)
        if not key:
            return
        # Update in-memory cache immediately
        self._market_cache[key] = data

        async def _persist(_attempt: int) -> None:
            async with AsyncSessionLocal() as session:
                try:
                    now = _utcnow_naive()
                    known_keys = {
                        "question",
                        "slug",
                        "groupItemTitle",
                        "category",
                        "active",
                    }
                    extra = {k: v for k, v in data.items() if k not in known_keys}

                    stmt = pg_insert(CachedMarket).values(
                        condition_id=key,
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
                except Exception:
                    await session.rollback()
                    raise

        try:
            await run_with_db_retries(_persist)
        except (DBAPIError, asyncio.TimeoutError) as e:
            logger.error(
                "Failed to persist market cache entry",
                condition_id=key,
                error=str(e),
            )
        except Exception as e:
            logger.error(
                "Failed to persist market cache entry",
                condition_id=key,
                error=str(e),
            )

    async def bulk_set_markets(self, markets: dict[str, dict]):
        """Bulk cache market metadata."""
        if not markets:
            return
        normalized_markets = {
            key: data
            for raw_key, data in markets.items()
            if (key := self._norm(raw_key))
        }
        if not normalized_markets:
            return

        # Update in-memory cache
        self._market_cache.update(normalized_markets)

        # Persist to database in a single transaction
        try:
            async with AsyncSessionLocal() as session:
                now = _utcnow_naive()
                known_keys = {
                    "question",
                    "slug",
                    "groupItemTitle",
                    "category",
                    "active",
                }

                for condition_id, data in normalized_markets.items():
                    extra = {k: v for k, v in data.items() if k not in known_keys}
                    stmt = pg_insert(CachedMarket).values(
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
                now = _utcnow_naive()
                stmt = pg_insert(CachedUsername).values(
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
                now = _utcnow_naive()
                for addr, uname in normalised.items():
                    stmt = pg_insert(CachedUsername).values(
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

    # -------------------- Statistics & maintenance --------------------

    async def get_cache_stats(self) -> dict:
        """Get cache statistics (counts, oldest entry, etc.)."""
        stats: dict = {
            "markets_cached": len(self._market_cache),
            "usernames_cached": len(self._username_cache),
            "loaded": self._loaded,
            "loading": bool(self._load_task and not self._load_task.done()),
            "load_started_at": self._load_started_at.isoformat() if self._load_started_at else None,
            "load_finished_at": self._load_finished_at.isoformat() if self._load_finished_at else None,
            "load_error": self._load_error,
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
        cutoff = _utcnow_naive() - timedelta(days=max_age_days)
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
        if os.environ.get("HOMERUN_PROCESS_ROLE") == "worker":
            plane = os.environ.get("HOMERUN_WORKER_PLANE", "").strip().lower()
            if plane not in {"trading", "all"}:
                return {"status": "skipped", "reason": "worker_plane", "plane": plane or "unknown"}
        if not force and is_db_pressure_active():
            return {"status": "skipped", "reason": "db_pressure"}
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
            # Hygiene scans 2M+ row ``wallet_activity_rollups`` with
            # ``DISTINCT market_id`` over a 45-day window — even with
            # the ``idx_war_traded_wallet_market`` index this routinely
            # exceeds the regular 30s ``statement_timeout`` on a busy
            # DB.  ``SET LOCAL`` raises only this transaction's cap.
            await session.execute(text("SET LOCAL statement_timeout = '180000'"))
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
                if updated_at is not None:
                    if updated_at.tzinfo is None:
                        updated_at = updated_at.replace(tzinfo=timezone.utc)
                    else:
                        updated_at = updated_at.astimezone(timezone.utc)

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
