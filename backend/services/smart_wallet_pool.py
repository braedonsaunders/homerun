"""
Near-real-time smart wallet pool management.

Builds and maintains a 400-600 wallet pool (target 500) using
quality + recency + stability scoring. Also persists wallet activity
events for confluence detection windows.
"""

from __future__ import annotations

import asyncio
import math
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from utils.utcnow import utcnow, utcfromtimestamp
from typing import Any, Optional

from sqlalchemy import select, func

from models.database import (
    AsyncSessionLocal,
    DiscoveredWallet,
    MarketConfluenceSignal,
    WalletActivityRollup,
)
from services.pause_state import global_pause_state
from services.polymarket import polymarket_client
from utils.logger import get_logger

logger = get_logger("smart_wallet_pool")


# Pool sizing and activity windows
TARGET_POOL_SIZE = 500
MIN_POOL_SIZE = 400
MAX_POOL_SIZE = 600
ACTIVE_WINDOW_HOURS = 72

# Scheduling targets
FULL_SWEEP_INTERVAL = timedelta(minutes=60)
INCREMENTAL_REFRESH_INTERVAL = timedelta(minutes=5)
ACTIVITY_RECONCILIATION_INTERVAL = timedelta(minutes=2)
POOL_RECOMPUTE_INTERVAL = timedelta(minutes=1)

# Churn guard
MAX_HOURLY_REPLACEMENT_RATE = 0.15
REPLACEMENT_SCORE_CUTOFF = 0.05

# Source categories for leaderboard matrix scan
LEADERBOARD_PERIODS = ("DAY", "WEEK", "MONTH", "ALL")
LEADERBOARD_SORTS = ("PNL", "VOL")
LEADERBOARD_CATEGORIES = (
    "OVERALL",
    "POLITICS",
    "SPORTS",
    "CRYPTO",
    "CULTURE",
    "ECONOMICS",
    "TECH",
    "FINANCE",
)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(value, hi))


class SmartWalletPoolService:
    """Maintains the smart wallet pool and wallet activity rollups."""

    def __init__(self):
        self.client = polymarket_client
        self._running = False
        self._lock = asyncio.Lock()
        self._activity_cache: dict[str, datetime] = {}
        self._callback_registered = False
        self._ws_broadcast_callback = None

        self._stats: dict[str, Any] = {
            "target_pool_size": TARGET_POOL_SIZE,
            "min_pool_size": MIN_POOL_SIZE,
            "max_pool_size": MAX_POOL_SIZE,
            "last_full_sweep_at": None,
            "last_incremental_refresh_at": None,
            "last_activity_reconciliation_at": None,
            "last_pool_recompute_at": None,
            "last_error": None,
            "churn_rate": 0.0,
            "pool_size": 0,
            "candidates_last_sweep": 0,
            "events_last_reconcile": 0,
        }

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    async def start_background(self):
        """Run all smart pool jobs on target cadence."""
        if self._running:
            return

        self._running = True
        logger.info("Starting smart wallet pool background loop")
        await self._ensure_ws_callback_registered()

        # Force an initial run so the pool is available shortly after boot.
        await self.run_full_sweep()
        await self.recompute_pool()

        while self._running:
            try:
                if not global_pause_state.is_paused:
                    now = utcnow()

                    if self._is_due("last_full_sweep_at", FULL_SWEEP_INTERVAL, now):
                        await self.run_full_sweep()

                    if self._is_due(
                        "last_incremental_refresh_at",
                        INCREMENTAL_REFRESH_INTERVAL,
                        now,
                    ):
                        await self.run_incremental_refresh()

                    if self._is_due(
                        "last_activity_reconciliation_at",
                        ACTIVITY_RECONCILIATION_INTERVAL,
                        now,
                    ):
                        await self.reconcile_activity()

                    if self._is_due(
                        "last_pool_recompute_at",
                        POOL_RECOMPUTE_INTERVAL,
                        now,
                    ):
                        await self.recompute_pool()
            except Exception as e:
                self._stats["last_error"] = str(e)
                logger.error("Smart wallet pool loop error", error=str(e))

            await asyncio.sleep(15)

    def stop(self):
        """Stop the background scheduler."""
        self._running = False
        logger.info("Smart wallet pool background loop stopped")

    def set_ws_broadcast(self, callback):
        """Set websocket broadcast callback for pool status events."""
        self._ws_broadcast_callback = callback

    async def _broadcast_event(self, event_type: str, data: dict):
        if not self._ws_broadcast_callback:
            return
        try:
            await self._ws_broadcast_callback({"type": event_type, "data": data})
        except Exception as e:
            logger.debug("Smart pool websocket broadcast failed", error=str(e))

    def _is_due(self, key: str, interval: timedelta, now: datetime) -> bool:
        raw = self._stats.get(key)
        if raw is None:
            return True
        try:
            last = datetime.fromisoformat(raw)
        except Exception:
            return True
        return now - last >= interval

    # ------------------------------------------------------------------
    # Public jobs
    # ------------------------------------------------------------------

    async def run_full_sweep(self):
        """Collect candidates from all configured sources."""
        async with self._lock:
            logger.info("Starting smart wallet full candidate sweep")

            candidate_sources: dict[str, dict[str, bool]] = defaultdict(dict)
            events: list[dict] = []

            await self._collect_leaderboard_candidates(
                candidate_sources,
                periods=LEADERBOARD_PERIODS,
                sorts=LEADERBOARD_SORTS,
                categories=LEADERBOARD_CATEGORIES,
                per_matrix_limit=100,
            )
            markets = await self._collect_market_trade_candidates(
                candidate_sources,
                events,
                max_markets=30,
                max_trades_per_market=120,
            )
            await self._collect_activity_candidates(
                candidate_sources,
                events,
                limit=500,
            )
            await self._collect_holder_candidates(
                candidate_sources,
                events,
                market_ids=markets,
                per_market_limit=100,
            )

            await self._upsert_candidate_wallets(candidate_sources)
            inserted = await self._persist_activity_events(events)

            self._stats["last_full_sweep_at"] = utcnow().isoformat()
            self._stats["candidates_last_sweep"] = len(candidate_sources)
            self._stats["events_last_reconcile"] = inserted

            logger.info(
                "Smart wallet full sweep complete",
                candidates=len(candidate_sources),
                events=inserted,
            )

    async def run_incremental_refresh(self):
        """Lightweight candidate refresh intended for frequent runs."""
        async with self._lock:
            candidate_sources: dict[str, dict[str, bool]] = defaultdict(dict)
            events: list[dict] = []

            await self._collect_leaderboard_candidates(
                candidate_sources,
                periods=("DAY",),
                sorts=LEADERBOARD_SORTS,
                categories=("OVERALL", "CRYPTO", "POLITICS", "SPORTS"),
                per_matrix_limit=80,
            )
            await self._collect_market_trade_candidates(
                candidate_sources,
                events,
                max_markets=12,
                max_trades_per_market=80,
            )
            await self._upsert_candidate_wallets(candidate_sources)
            await self._persist_activity_events(events)

            self._stats["last_incremental_refresh_at"] = utcnow().isoformat()
            logger.info(
                "Smart wallet incremental refresh complete",
                candidates=len(candidate_sources),
                events=len(events),
            )

    async def reconcile_activity(self):
        """Use activity endpoint to backfill missed trades."""
        async with self._lock:
            candidate_sources: dict[str, dict[str, bool]] = defaultdict(dict)
            events: list[dict] = []

            await self._collect_activity_candidates(
                candidate_sources,
                events,
                limit=250,
            )
            await self._upsert_candidate_wallets(candidate_sources)
            inserted = await self._persist_activity_events(events)

            self._stats["last_activity_reconciliation_at"] = utcnow().isoformat()
            self._stats["events_last_reconcile"] = inserted

            logger.info(
                "Smart wallet activity reconciliation complete",
                candidates=len(candidate_sources),
                events=inserted,
            )

    async def recompute_pool(self):
        """Recompute scoring, apply churn guard, and update pool membership."""
        async with self._lock:
            now = utcnow()
            churn = await self._refresh_metrics_and_apply_pool(now)
            self._stats["churn_rate"] = round(churn, 4)
            self._stats["last_pool_recompute_at"] = utcnow().isoformat()

            pool_size = await self._count_pool_wallets()
            self._stats["pool_size"] = pool_size
            await self._broadcast_event(
                "tracked_trader_pool_update",
                {
                    "pool_size": pool_size,
                    "target_pool_size": TARGET_POOL_SIZE,
                    "churn_rate": round(churn, 4),
                    "updated_at": utcnow().isoformat(),
                },
            )
            logger.info(
                "Smart wallet pool recompute complete",
                pool_size=pool_size,
                churn_rate=round(churn, 4),
            )

    async def get_pool_stats(self) -> dict:
        """Return aggregate pool health and freshness stats."""
        async with AsyncSessionLocal() as session:
            pool_size = (
                (
                    await session.execute(
                        select(func.count(DiscoveredWallet.address)).where(
                            DiscoveredWallet.in_top_pool == True  # noqa: E712
                        )
                    )
                )
                .scalar()
                or 0
            )
            active_1h = (
                (
                    await session.execute(
                        select(func.count(DiscoveredWallet.address)).where(
                            DiscoveredWallet.in_top_pool == True,  # noqa: E712
                            DiscoveredWallet.trades_1h > 0,
                        )
                    )
                )
                .scalar()
                or 0
            )
            active_24h = (
                (
                    await session.execute(
                        select(func.count(DiscoveredWallet.address)).where(
                            DiscoveredWallet.in_top_pool == True,  # noqa: E712
                            DiscoveredWallet.trades_24h > 0,
                        )
                    )
                )
                .scalar()
                or 0
            )
            newest = (
                (
                    await session.execute(
                        select(func.max(DiscoveredWallet.last_trade_at)).where(
                            DiscoveredWallet.in_top_pool == True  # noqa: E712
                        )
                    )
                )
                .scalar()
                or None
            )
            oldest = (
                (
                    await session.execute(
                        select(func.min(DiscoveredWallet.last_trade_at)).where(
                            DiscoveredWallet.in_top_pool == True,  # noqa: E712
                            DiscoveredWallet.last_trade_at.is_not(None),
                        )
                    )
                )
                .scalar()
                or None
            )

        return {
            **self._stats,
            "pool_size": pool_size,
            "active_1h": active_1h,
            "active_24h": active_24h,
            "active_1h_pct": round((active_1h / pool_size) * 100, 2) if pool_size else 0.0,
            "active_24h_pct": round((active_24h / pool_size) * 100, 2) if pool_size else 0.0,
            "freshest_trade_at": newest.isoformat() if newest else None,
            "stale_floor_trade_at": oldest.isoformat() if oldest else None,
        }

    async def get_tracked_trader_opportunities(
        self,
        limit: int = 50,
        min_tier: str = "WATCH",
    ) -> list[dict]:
        """Get signal-first opportunities for the Tracked Traders surface."""
        tier_rank = {"WATCH": 1, "HIGH": 2, "EXTREME": 3}
        min_rank = tier_rank.get(min_tier.upper(), 1)

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(MarketConfluenceSignal)
                .where(MarketConfluenceSignal.is_active == True)  # noqa: E712
                .order_by(
                    MarketConfluenceSignal.conviction_score.desc(),
                    MarketConfluenceSignal.last_seen_at.desc(),
                )
                .limit(limit * 2)
            )
            raw = list(result.scalars().all())

            signals = [
                s
                for s in raw
                if tier_rank.get((s.tier or "WATCH").upper(), 1) >= min_rank
            ][:limit]

            addresses = {
                addr.lower()
                for s in signals
                for addr in (s.wallets or [])
                if isinstance(addr, str)
            }

            profile_rows = await session.execute(
                select(DiscoveredWallet).where(DiscoveredWallet.address.in_(list(addresses)))
            )
            profiles = {
                w.address: {
                    "address": w.address,
                    "username": w.username,
                    "rank_score": w.rank_score or 0.0,
                    "composite_score": w.composite_score or 0.0,
                    "quality_score": w.quality_score or 0.0,
                    "activity_score": w.activity_score or 0.0,
                }
                for w in profile_rows.scalars().all()
            }

            output = []
            for s in signals:
                top_wallets = []
                for address in (s.wallets or [])[:8]:
                    profile = profiles.get(address.lower())
                    if profile:
                        top_wallets.append(profile)

                output.append(
                    {
                        "id": s.id,
                        "market_id": s.market_id,
                        "market_question": s.market_question,
                        "market_slug": s.market_slug,
                        "signal_type": s.signal_type,
                        "outcome": s.outcome,
                        "tier": s.tier or "WATCH",
                        "conviction_score": s.conviction_score or 0.0,
                        "strength": s.strength or 0.0,
                        "wallet_count": s.wallet_count or 0,
                        "cluster_adjusted_wallet_count": s.cluster_adjusted_wallet_count
                        or 0,
                        "unique_core_wallets": s.unique_core_wallets or 0,
                        "weighted_wallet_score": s.weighted_wallet_score or 0.0,
                        "window_minutes": s.window_minutes or 60,
                        "avg_entry_price": s.avg_entry_price,
                        "total_size": s.total_size,
                        "net_notional": s.net_notional,
                        "conflicting_notional": s.conflicting_notional,
                        "market_liquidity": s.market_liquidity,
                        "market_volume_24h": s.market_volume_24h,
                        "first_seen_at": s.first_seen_at.isoformat()
                        if s.first_seen_at
                        else None,
                        "last_seen_at": s.last_seen_at.isoformat()
                        if s.last_seen_at
                        else None,
                        "detected_at": (
                            s.detected_at.isoformat()
                            if s.detected_at
                            else (
                                s.last_seen_at.isoformat()
                                if s.last_seen_at
                                else utcnow().isoformat()
                            )
                        ),
                        "is_active": bool(s.is_active),
                        "wallets": s.wallets or [],
                        "top_wallets": top_wallets,
                    }
                )

        return output

    # ------------------------------------------------------------------
    # Candidate collection
    # ------------------------------------------------------------------

    async def _collect_leaderboard_candidates(
        self,
        candidates: dict[str, dict[str, bool]],
        periods: tuple[str, ...],
        sorts: tuple[str, ...],
        categories: tuple[str, ...],
        per_matrix_limit: int,
    ):
        for period in periods:
            for sort in sorts:
                for category in categories:
                    try:
                        rows = await self.client.get_leaderboard_paginated(
                            total_limit=per_matrix_limit,
                            time_period=period,
                            order_by=sort,
                            category=category,
                        )
                    except Exception as e:
                        logger.warning(
                            "Leaderboard matrix scan failed",
                            period=period,
                            sort=sort,
                            category=category,
                            error=str(e),
                        )
                        continue

                    for row in rows:
                        address = (row.get("proxyWallet", "") or "").lower()
                        if not address:
                            continue
                        candidates[address]["leaderboard"] = True
                        if sort == "PNL":
                            candidates[address]["leaderboard_pnl"] = True
                        else:
                            candidates[address]["leaderboard_vol"] = True

    async def _collect_market_trade_candidates(
        self,
        candidates: dict[str, dict[str, bool]],
        events: list[dict],
        max_markets: int,
        max_trades_per_market: int,
    ) -> list[str]:
        market_ids: list[str] = []
        try:
            markets = await self.client.get_markets(active=True, limit=200, offset=0)
        except Exception as e:
            logger.warning("Failed to fetch markets for trade sampling", error=str(e))
            return market_ids

        ranked = sorted(
            markets,
            key=lambda m: (getattr(m, "liquidity", 0.0) or 0.0) + (getattr(m, "volume", 0.0) or 0.0),
            reverse=True,
        )
        sampled = ranked[:max_markets]

        for market in sampled:
            market_id = getattr(market, "condition_id", None) or getattr(market, "id", None)
            if not market_id:
                continue
            market_ids.append(str(market_id))

            try:
                trades = await self.client.get_market_trades(
                    str(market_id),
                    limit=min(max_trades_per_market, 500),
                )
            except Exception:
                continue

            for trade in trades:
                side = (trade.get("side", "") or "").upper()
                price = float(trade.get("price", 0) or 0)
                size = float(trade.get("size", 0) or trade.get("amount", 0) or 0)
                ts = self._parse_timestamp(
                    trade.get("timestamp")
                    or trade.get("created_at")
                    or trade.get("match_time")
                    or trade.get("time")
                )
                if ts is None:
                    continue

                user = (trade.get("user", "") or "").lower()
                maker = (trade.get("maker", "") or "").lower()
                taker = (trade.get("taker", "") or "").lower()
                tx_hash = trade.get("transactionHash") or trade.get("tx_hash")

                if user:
                    candidates[user]["market_trades"] = True
                    events.append(
                        self._event_record(
                            wallet=user,
                            market_id=str(market_id),
                            side=side or "TRADE",
                            size=size,
                            price=price,
                            traded_at=ts,
                            source="trades_api",
                            tx_hash=tx_hash,
                        )
                    )
                else:
                    if maker:
                        candidates[maker]["market_trades"] = True
                        events.append(
                            self._event_record(
                                wallet=maker,
                                market_id=str(market_id),
                                side="SELL",
                                size=size,
                                price=price,
                                traded_at=ts,
                                source="trades_api",
                                tx_hash=tx_hash,
                            )
                        )
                    if taker:
                        candidates[taker]["market_trades"] = True
                        events.append(
                            self._event_record(
                                wallet=taker,
                                market_id=str(market_id),
                                side="BUY",
                                size=size,
                                price=price,
                                traded_at=ts,
                                source="trades_api",
                                tx_hash=tx_hash,
                            )
                        )

        return market_ids

    async def _collect_activity_candidates(
        self,
        candidates: dict[str, dict[str, bool]],
        events: list[dict],
        limit: int,
    ):
        try:
            rows = await self.client.get_activity(
                limit=min(limit, 500),
                offset=0,
                activity_type="TRADE",
            )
        except Exception as e:
            logger.warning("Failed to fetch activity backfill", error=str(e))
            return

        for row in rows:
            address = (
                row.get("proxyWallet")
                or row.get("user")
                or row.get("wallet")
                or row.get("maker")
                or row.get("taker")
                or ""
            )
            address = address.lower()
            if not address:
                continue

            market_id = (
                row.get("market")
                or row.get("condition_id")
                or row.get("asset")
                or row.get("token_id")
                or ""
            )
            if not market_id:
                continue

            side = (row.get("side", "") or row.get("direction", "") or "TRADE").upper()
            size = float(row.get("size", 0) or row.get("amount", 0) or 0)
            price = float(row.get("price", 0) or 0)
            ts = self._parse_timestamp(
                row.get("timestamp")
                or row.get("created_at")
                or row.get("createdAt")
                or row.get("time")
            )
            if ts is None:
                continue

            candidates[address]["activity"] = True
            events.append(
                self._event_record(
                    wallet=address,
                    market_id=str(market_id),
                    side=side,
                    size=size,
                    price=price,
                    traded_at=ts,
                    source="activity_api",
                    tx_hash=row.get("transactionHash") or row.get("tx_hash"),
                )
            )

    async def _collect_holder_candidates(
        self,
        candidates: dict[str, dict[str, bool]],
        events: list[dict],
        market_ids: list[str],
        per_market_limit: int,
    ):
        for market_id in market_ids[:15]:
            try:
                holders = await self.client.get_market_holders(
                    market_id,
                    limit=min(per_market_limit, 500),
                    offset=0,
                )
            except Exception:
                continue

            now = utcnow()
            for holder in holders:
                address = (
                    holder.get("proxyWallet")
                    or holder.get("address")
                    or holder.get("wallet")
                    or holder.get("user")
                    or ""
                )
                address = address.lower()
                if not address:
                    continue

                size = float(holder.get("shares", 0) or holder.get("size", 0) or 0)
                price = float(holder.get("price", 0) or 0)

                candidates[address]["holders"] = True
                events.append(
                    self._event_record(
                        wallet=address,
                        market_id=str(market_id),
                        side="HOLD",
                        size=size,
                        price=price,
                        traded_at=now,
                        source="holders_api",
                        tx_hash=None,
                    )
                )

    # ------------------------------------------------------------------
    # Persistence and scoring
    # ------------------------------------------------------------------

    async def _upsert_candidate_wallets(
        self, candidates: dict[str, dict[str, bool]]
    ):
        if not candidates:
            return

        addresses = list(candidates.keys())
        async with AsyncSessionLocal() as session:
            existing_result = await session.execute(
                select(DiscoveredWallet).where(DiscoveredWallet.address.in_(addresses))
            )
            existing = {w.address: w for w in existing_result.scalars().all()}

            for address, flags in candidates.items():
                wallet = existing.get(address)
                if wallet is None:
                    wallet = DiscoveredWallet(
                        address=address,
                        discovered_at=utcnow(),
                        discovery_source="smart_pool",
                        source_flags=dict(flags),
                    )
                    session.add(wallet)
                    continue

                prior = wallet.source_flags or {}
                if not isinstance(prior, dict):
                    prior = {}
                for key, value in flags.items():
                    prior[key] = bool(value)
                wallet.source_flags = prior

            await session.commit()

    async def _persist_activity_events(self, events: list[dict]) -> int:
        if not events:
            return 0

        now = utcnow()
        self._trim_activity_cache(now)

        inserts: list[dict] = []
        for event in events:
            key = (
                f"{event['wallet_address']}|{event['market_id']}|{event.get('side')}"
                f"|{int(event['traded_at'].timestamp())}|{event.get('tx_hash') or ''}"
            )
            if key in self._activity_cache:
                continue
            self._activity_cache[key] = now
            inserts.append(event)

        if not inserts:
            return 0

        async with AsyncSessionLocal() as session:
            for event in inserts:
                session.add(
                    WalletActivityRollup(
                        id=str(uuid.uuid4()),
                        wallet_address=event["wallet_address"],
                        market_id=event["market_id"],
                        side=event.get("side"),
                        size=event.get("size"),
                        price=event.get("price"),
                        notional=event.get("notional"),
                        tx_hash=event.get("tx_hash"),
                        source=event.get("source", "unknown"),
                        traded_at=event["traded_at"],
                    )
                )
            await session.commit()

        return len(inserts)

    def _trim_activity_cache(self, now: datetime):
        if len(self._activity_cache) < 50_000:
            return
        cutoff = now - timedelta(hours=6)
        stale = [k for k, t in self._activity_cache.items() if t < cutoff]
        for key in stale:
            self._activity_cache.pop(key, None)

    async def _refresh_metrics_and_apply_pool(self, now: datetime) -> float:
        cutoff_1h = now - timedelta(hours=1)
        cutoff_24h = now - timedelta(hours=24)
        cutoff_72h = now - timedelta(hours=ACTIVE_WINDOW_HOURS)

        async with AsyncSessionLocal() as session:
            one_hour = await session.execute(
                select(
                    WalletActivityRollup.wallet_address,
                    func.count(WalletActivityRollup.id).label("trades_1h"),
                )
                .where(WalletActivityRollup.traded_at >= cutoff_1h)
                .group_by(WalletActivityRollup.wallet_address)
            )
            map_1h = {row.wallet_address: int(row.trades_1h or 0) for row in one_hour}

            twenty_four = await session.execute(
                select(
                    WalletActivityRollup.wallet_address,
                    func.count(WalletActivityRollup.id).label("trades_24h"),
                    func.count(func.distinct(WalletActivityRollup.market_id)).label(
                        "unique_markets_24h"
                    ),
                    func.max(WalletActivityRollup.traded_at).label("last_trade_at"),
                )
                .where(WalletActivityRollup.traded_at >= cutoff_24h)
                .group_by(WalletActivityRollup.wallet_address)
            )
            map_24h = {
                row.wallet_address: {
                    "trades_24h": int(row.trades_24h or 0),
                    "unique_markets_24h": int(row.unique_markets_24h or 0),
                    "last_trade_at": row.last_trade_at,
                }
                for row in twenty_four
            }

            wallets_result = await session.execute(select(DiscoveredWallet))
            wallets = list(wallets_result.scalars().all())

            for wallet in wallets:
                row_24 = map_24h.get(wallet.address, {})
                trades_1h = map_1h.get(wallet.address, 0)
                trades_24h = int(row_24.get("trades_24h", 0))
                unique_markets_24h = int(row_24.get("unique_markets_24h", 0))

                last_trade_at = row_24.get("last_trade_at")
                if last_trade_at is None:
                    # Keep existing value when no new events are present.
                    last_trade_at = wallet.last_trade_at

                quality = self._score_quality(wallet)
                activity = self._score_activity(trades_1h, trades_24h, last_trade_at, now)
                stability = self._score_stability(wallet)
                composite = _clamp(0.45 * quality + 0.35 * activity + 0.20 * stability)

                wallet.trades_1h = trades_1h
                wallet.trades_24h = trades_24h
                wallet.unique_markets_24h = unique_markets_24h
                wallet.last_trade_at = last_trade_at
                wallet.quality_score = quality
                wallet.activity_score = activity
                wallet.stability_score = stability
                wallet.composite_score = composite

            # Selection pool: prioritize active wallets, but allow fill from
            # broader ranked set when active coverage is sparse.
            ranked_wallets = sorted(
                wallets,
                key=lambda w: (w.composite_score or 0.0, w.rank_score or 0.0),
                reverse=True,
            )
            active_ranked = [
                w
                for w in ranked_wallets
                if w.last_trade_at is not None and w.last_trade_at >= cutoff_72h
            ]

            desired = [w.address for w in active_ranked[:TARGET_POOL_SIZE]]
            if len(desired) < MIN_POOL_SIZE:
                for wallet in ranked_wallets:
                    if wallet.address in desired:
                        continue
                    desired.append(wallet.address)
                    if len(desired) >= TARGET_POOL_SIZE:
                        break

            current_pool = [w.address for w in wallets if w.in_top_pool]
            final_pool, churn_rate = self._apply_churn_guard(
                desired=desired,
                current=current_pool,
                scores={w.address: (w.composite_score or 0.0) for w in ranked_wallets},
            )

            final_index = {address: i for i, address in enumerate(final_pool)}
            core_cut = int(TARGET_POOL_SIZE * 0.7)

            for wallet in wallets:
                idx = final_index.get(wallet.address)
                wallet.in_top_pool = idx is not None
                if idx is None:
                    wallet.pool_tier = None
                    wallet.pool_membership_reason = None
                else:
                    wallet.pool_tier = "core" if idx < core_cut else "rising"
                    if wallet.last_trade_at and wallet.last_trade_at >= cutoff_72h:
                        wallet.pool_membership_reason = "active_composite"
                    else:
                        wallet.pool_membership_reason = "fill_from_rank"

            await session.commit()

        await self._sync_ws_membership(final_pool)
        return churn_rate

    def _score_quality(self, wallet: DiscoveredWallet) -> float:
        rank = _clamp(float(wallet.rank_score or 0.0))
        win = _clamp(float(wallet.win_rate or 0.0))

        sharpe = wallet.sharpe_ratio
        sharpe_norm = 0.5 if sharpe is None or not math.isfinite(sharpe) else _clamp(sharpe / 3.0)

        pf = wallet.profit_factor
        pf_norm = 0.5 if pf is None or not math.isfinite(pf) else _clamp(pf / 5.0)

        pnl = float(wallet.total_pnl or 0.0)
        pnl_norm = _clamp((math.tanh(pnl / 25000.0) + 1.0) / 2.0)

        return _clamp(
            0.35 * rank
            + 0.25 * win
            + 0.15 * sharpe_norm
            + 0.15 * pf_norm
            + 0.10 * pnl_norm
        )

    def _score_activity(
        self,
        trades_1h: int,
        trades_24h: int,
        last_trade_at: Optional[datetime],
        now: datetime,
    ) -> float:
        flow_1h = _clamp(trades_1h / 6.0)
        flow_24h = _clamp(trades_24h / 40.0)

        if last_trade_at is None:
            recency = 0.0
        else:
            age_hours = max((now - last_trade_at).total_seconds() / 3600.0, 0.0)
            recency = _clamp(1.0 - (age_hours / ACTIVE_WINDOW_HOURS))

        return _clamp(0.50 * flow_1h + 0.30 * flow_24h + 0.20 * recency)

    def _score_stability(self, wallet: DiscoveredWallet) -> float:
        drawdown = wallet.max_drawdown
        consistency = 0.5 if drawdown is None else _clamp(1.0 - min(drawdown, 1.0))

        roi_std = float(wallet.roi_std or 0.0)
        roi_penalty = _clamp(abs(roi_std) / 50.0) * 0.25

        anomaly = _clamp(float(wallet.anomaly_score or 0.0))
        anomaly_penalty = anomaly * 0.35

        cluster_penalty = 0.10 if wallet.cluster_id else 0.0
        profitable_bonus = 0.15 if wallet.is_profitable else 0.0

        return _clamp(consistency - roi_penalty - anomaly_penalty - cluster_penalty + profitable_bonus)

    def _apply_churn_guard(
        self,
        desired: list[str],
        current: list[str],
        scores: dict[str, float],
    ) -> tuple[list[str], float]:
        desired = desired[:TARGET_POOL_SIZE]
        current = current[:TARGET_POOL_SIZE]

        # If no existing pool, initialize directly from desired.
        if not current:
            initialized = desired[:TARGET_POOL_SIZE]
            if len(initialized) < MIN_POOL_SIZE:
                # Keep deterministic ordering from scores.
                ordered = sorted(scores.keys(), key=lambda a: scores.get(a, 0.0), reverse=True)
                for address in ordered:
                    if address in initialized:
                        continue
                    initialized.append(address)
                    if len(initialized) >= MIN_POOL_SIZE:
                        break
            return initialized[:MAX_POOL_SIZE], 0.0

        max_replacements = max(1, int(TARGET_POOL_SIZE * MAX_HOURLY_REPLACEMENT_RATE))
        pool_set = set(current)

        # Trim if current pool is larger than target.
        if len(pool_set) > TARGET_POOL_SIZE:
            keep = sorted(list(pool_set), key=lambda a: scores.get(a, 0.0), reverse=True)[
                :TARGET_POOL_SIZE
            ]
            pool_set = set(keep)

        desired_set = set(desired)
        additions = sorted(
            [a for a in desired if a not in pool_set],
            key=lambda a: scores.get(a, 0.0),
            reverse=True,
        )
        removals = sorted(
            [a for a in pool_set if a not in desired_set],
            key=lambda a: scores.get(a, 0.0),
        )

        replacements = 0
        for address in additions:
            if len(pool_set) < TARGET_POOL_SIZE:
                pool_set.add(address)
                replacements += 1
                continue

            if not removals:
                break

            outgoing = removals[0]
            incoming_score = scores.get(address, 0.0)
            outgoing_score = scores.get(outgoing, 0.0)

            can_replace = replacements < max_replacements or (
                incoming_score >= outgoing_score + REPLACEMENT_SCORE_CUTOFF
            )
            if not can_replace:
                continue

            pool_set.discard(outgoing)
            pool_set.add(address)
            removals.pop(0)
            replacements += 1

        # Ensure minimum pool floor.
        if len(pool_set) < MIN_POOL_SIZE:
            ordered = sorted(scores.keys(), key=lambda a: scores.get(a, 0.0), reverse=True)
            for address in ordered:
                if address in pool_set:
                    continue
                pool_set.add(address)
                if len(pool_set) >= MIN_POOL_SIZE:
                    break

        # Cap hard upper bound.
        if len(pool_set) > MAX_POOL_SIZE:
            ordered = sorted(list(pool_set), key=lambda a: scores.get(a, 0.0), reverse=True)
            pool_set = set(ordered[:MAX_POOL_SIZE])

        final = sorted(list(pool_set), key=lambda a: scores.get(a, 0.0), reverse=True)
        churn_rate = replacements / max(len(current), 1)
        return final, churn_rate

    async def _sync_ws_membership(self, pool_addresses: list[str]):
        try:
            from services.wallet_ws_monitor import wallet_ws_monitor

            wallet_ws_monitor.set_wallets_for_source("discovery_pool", pool_addresses)
            # Ensure the monitor is running even if copy trading is disabled.
            asyncio.create_task(wallet_ws_monitor.start())
        except Exception as e:
            logger.warning("Failed to sync discovery pool WS memberships", error=str(e))

    async def _count_pool_wallets(self) -> int:
        async with AsyncSessionLocal() as session:
            count = await session.execute(
                select(func.count(DiscoveredWallet.address)).where(
                    DiscoveredWallet.in_top_pool == True  # noqa: E712
                )
            )
            return int(count.scalar() or 0)

    # ------------------------------------------------------------------
    # WS callback integration
    # ------------------------------------------------------------------

    async def _ensure_ws_callback_registered(self):
        if self._callback_registered:
            return
        try:
            from services.wallet_ws_monitor import wallet_ws_monitor

            wallet_ws_monitor.add_callback(self._on_ws_trade_event)
            self._callback_registered = True
        except Exception as e:
            logger.warning("Failed to register smart pool WS callback", error=str(e))

    async def _on_ws_trade_event(self, event):
        """Capture WS trades into rollups for minute-level recency updates."""
        market_id = event.token_id
        try:
            info = await self.client.get_market_by_token_id(event.token_id)
            if info:
                market_id = info.get("condition_id") or info.get("slug") or event.token_id
        except Exception:
            pass

        record = self._event_record(
            wallet=event.wallet_address.lower(),
            market_id=str(market_id),
            side=(event.side or "").upper() or "TRADE",
            size=float(event.size or 0),
            price=float(event.price or 0),
            traded_at=event.timestamp or utcnow(),
            source="ws",
            tx_hash=event.tx_hash,
        )
        await self._persist_activity_events([record])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _event_record(
        self,
        wallet: str,
        market_id: str,
        side: str,
        size: float,
        price: float,
        traded_at: datetime,
        source: str,
        tx_hash: Optional[str],
    ) -> dict:
        notional = abs(size) * abs(price)
        return {
            "wallet_address": wallet.lower(),
            "market_id": market_id,
            "side": side,
            "size": size,
            "price": price,
            "notional": notional,
            "traded_at": traded_at,
            "source": source,
            "tx_hash": tx_hash,
        }

    def _parse_timestamp(self, raw: Any) -> Optional[datetime]:
        if raw is None:
            return None
        if isinstance(raw, datetime):
            return raw
        if isinstance(raw, (int, float)):
            try:
                return utcfromtimestamp(float(raw))
            except (OSError, ValueError):
                return None
        if isinstance(raw, str):
            try:
                if raw.replace(".", "", 1).isdigit():
                    return utcfromtimestamp(float(raw))
                return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(
                    tzinfo=None
                )
            except (OSError, ValueError, TypeError):
                return None
        return None


smart_wallet_pool = SmartWalletPoolService()
