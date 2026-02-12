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
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from utils.utcnow import utcnow, utcfromtimestamp
from typing import Any, Optional

from sqlalchemy import select, func
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

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
FULL_SWEEP_INTERVAL = timedelta(minutes=30)
INCREMENTAL_REFRESH_INTERVAL = timedelta(minutes=2)
ACTIVITY_RECONCILIATION_INTERVAL = timedelta(minutes=2)
POOL_RECOMPUTE_INTERVAL = timedelta(minutes=1)

# Churn guard
MAX_HOURLY_REPLACEMENT_RATE = 0.15
REPLACEMENT_SCORE_CUTOFF = 0.05
MAX_CLUSTER_SHARE = 0.08
HIGH_CONVICTION_THRESHOLD = 0.72
INSIDER_PRIORITY_THRESHOLD = 0.62

# Pool override flags stored in DiscoveredWallet.source_flags
POOL_FLAG_MANUAL_INCLUDE = "pool_manual_include"
POOL_FLAG_MANUAL_EXCLUDE = "pool_manual_exclude"
POOL_FLAG_BLACKLISTED = "pool_blacklisted"
POOL_SELECTION_META_KEY = "pool_selection_meta"
POOL_SELECTION_META_VERSION = 3

# Quality gate versioning / modes
QUALITY_GATE_VERSION = "quality_first_v1"
QUALITY_METRICS_SOURCE_VERSION = "accuracy_v2_closed_positions"
POOL_RECOMPUTE_MODE_QUALITY_ONLY = "quality_only"
POOL_RECOMPUTE_MODE_BALANCED = "balanced"
POOL_RECOMPUTE_MODES = {
    POOL_RECOMPUTE_MODE_QUALITY_ONLY,
    POOL_RECOMPUTE_MODE_BALANCED,
}

# Hard quality eligibility thresholds
MIN_ELIGIBLE_TRADES = 50
MAX_ELIGIBLE_ANOMALY = 0.5
CORE_MIN_WIN_RATE = 0.60
CORE_MIN_SHARPE = 1.0
CORE_MIN_PROFIT_FACTOR = 1.5
RISING_MIN_WIN_RATE = 0.55

# Pool health SLOs
SLO_MIN_ANALYZED_PCT = 95.0
SLO_MIN_PROFITABLE_PCT = 80.0

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

# Fallback wallet-trade sampling so activity rollups still populate even if
# market-level trade endpoints become sparse or schema-shift.
LEADERBOARD_WALLET_TRADE_SAMPLE = 160
INCREMENTAL_WALLET_TRADE_SAMPLE = 80

# Signal metadata refresh controls (self-heals stale market titles/slugs)
SIGNAL_METADATA_REFRESH_TTL = timedelta(minutes=20)
SIGNAL_METADATA_REFRESH_LIMIT = 12
SIGNAL_DUPLICATE_QUESTION_THRESHOLD = 3


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
        self._signal_market_refresh_at: dict[str, datetime] = {}
        self._recompute_mode = POOL_RECOMPUTE_MODE_QUALITY_ONLY
        self._last_copy_candidate_pct: Optional[float] = None

        self._stats: dict[str, Any] = {
            "target_pool_size": TARGET_POOL_SIZE,
            "min_pool_size": MIN_POOL_SIZE,
            "max_pool_size": MAX_POOL_SIZE,
            "quality_gate_version": QUALITY_GATE_VERSION,
            "recompute_mode": self._recompute_mode,
            "last_full_sweep_at": None,
            "last_incremental_refresh_at": None,
            "last_activity_reconciliation_at": None,
            "last_pool_recompute_at": None,
            "last_error": None,
            "churn_rate": 0.0,
            "pool_size": 0,
            "candidates_last_sweep": 0,
            "events_last_reconcile": 0,
            "analyzed_pool_pct": 0.0,
            "profitable_pool_pct": 0.0,
            "copy_candidate_pool_pct": 0.0,
            "copy_candidate_pool_pct_delta": 0.0,
            "slo_violations": [],
            "quality_only_auto_enforced": False,
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
            candidate_usernames: dict[str, str] = {}
            events: list[dict] = []

            await self._collect_leaderboard_candidates(
                candidate_sources,
                candidate_usernames=candidate_usernames,
                periods=LEADERBOARD_PERIODS,
                sorts=LEADERBOARD_SORTS,
                categories=LEADERBOARD_CATEGORIES,
                per_matrix_limit=100,
            )
            await self._collect_wallet_trade_candidates(
                candidate_sources,
                events,
                wallet_addresses=list(candidate_sources.keys())[
                    :LEADERBOARD_WALLET_TRADE_SAMPLE
                ],
                per_wallet_limit=80,
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

            await self._upsert_candidate_wallets(
                candidate_sources,
                candidate_usernames=candidate_usernames,
            )
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
            candidate_usernames: dict[str, str] = {}
            events: list[dict] = []

            await self._collect_leaderboard_candidates(
                candidate_sources,
                candidate_usernames=candidate_usernames,
                periods=("DAY",),
                sorts=LEADERBOARD_SORTS,
                categories=("OVERALL", "CRYPTO", "POLITICS", "SPORTS"),
                per_matrix_limit=80,
            )
            await self._collect_wallet_trade_candidates(
                candidate_sources,
                events,
                wallet_addresses=list(candidate_sources.keys())[
                    :INCREMENTAL_WALLET_TRADE_SAMPLE
                ],
                per_wallet_limit=50,
            )
            await self._collect_market_trade_candidates(
                candidate_sources,
                events,
                max_markets=12,
                max_trades_per_market=80,
            )
            await self._upsert_candidate_wallets(
                candidate_sources,
                candidate_usernames=candidate_usernames,
            )
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
            candidate_usernames: dict[str, str] = {}
            events: list[dict] = []

            await self._collect_activity_candidates(
                candidate_sources,
                events,
                limit=250,
            )
            await self._upsert_candidate_wallets(
                candidate_sources,
                candidate_usernames=candidate_usernames,
            )
            inserted = await self._persist_activity_events(events)

            self._stats["last_activity_reconciliation_at"] = utcnow().isoformat()
            self._stats["events_last_reconcile"] = inserted

            logger.info(
                "Smart wallet activity reconciliation complete",
                candidates=len(candidate_sources),
                events=inserted,
            )

    async def recompute_pool(self, mode: Optional[str] = None):
        """Recompute scoring, apply churn guard, and update pool membership."""
        async with self._lock:
            if mode is not None:
                self.set_recompute_mode(mode)
            now = utcnow()
            churn = await self._refresh_metrics_and_apply_pool(now)
            self._stats["churn_rate"] = round(churn, 4)
            self._stats["last_pool_recompute_at"] = utcnow().isoformat()
            self._stats["recompute_mode"] = self._recompute_mode

            pool_size = await self._count_pool_wallets()
            self._stats["pool_size"] = pool_size
            await self._broadcast_event(
                "tracked_trader_pool_update",
                {
                    "pool_size": pool_size,
                    "target_pool_size": TARGET_POOL_SIZE,
                    "churn_rate": round(churn, 4),
                    "recompute_mode": self._recompute_mode,
                    "updated_at": utcnow().isoformat(),
                },
            )
            logger.info(
                "Smart wallet pool recompute complete",
                pool_size=pool_size,
                churn_rate=round(churn, 4),
                recompute_mode=self._recompute_mode,
            )

    @staticmethod
    def _normalize_recompute_mode(mode: str) -> str:
        normalized = str(mode or "").strip().lower()
        if normalized not in POOL_RECOMPUTE_MODES:
            raise ValueError(f"unsupported recompute mode: {mode}")
        return normalized

    def set_recompute_mode(self, mode: str) -> str:
        normalized = self._normalize_recompute_mode(mode)
        self._recompute_mode = normalized
        self._stats["recompute_mode"] = normalized
        return normalized

    def get_recompute_mode(self) -> str:
        return self._recompute_mode

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
            "slo_targets": {
                "analyzed_pool_pct_min": SLO_MIN_ANALYZED_PCT,
                "profitable_pool_pct_min": SLO_MIN_PROFITABLE_PCT,
            },
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
            ]

            signals = await self._prune_non_tradable_signals(
                session=session,
                signals=signals,
            )
            signals = signals[:limit]

            await self._refresh_signal_market_metadata(session=session, signals=signals)

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

            question_market_ids: dict[str, set[str]] = {}
            for s in signals:
                question_norm = str(s.market_question or "").strip().lower()
                market_id_norm = str(s.market_id or "").strip().lower()
                if question_norm and market_id_norm:
                    question_market_ids.setdefault(question_norm, set()).add(market_id_norm)

            output = []
            for s in signals:
                top_wallets = []
                for address in (s.wallets or [])[:8]:
                    profile = profiles.get(address.lower())
                    if profile:
                        top_wallets.append(profile)

                market_question = str(s.market_question or "").strip()
                question_norm = market_question.lower()
                distinct_market_count = len(question_market_ids.get(question_norm, set()))
                if (
                    market_question
                    and distinct_market_count >= SIGNAL_DUPLICATE_QUESTION_THRESHOLD
                ):
                    # Safety fallback: avoid displaying clearly poisoned duplicate metadata.
                    market_question = f"Market {s.market_id}"
                if not market_question:
                    market_question = f"Market {s.market_id}"

                output.append(
                    {
                        "id": s.id,
                        "market_id": s.market_id,
                        "market_question": market_question,
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

    async def _prune_non_tradable_signals(
        self,
        *,
        session,
        signals: list[MarketConfluenceSignal],
    ) -> list[MarketConfluenceSignal]:
        """Deactivate and remove signals whose markets are no longer tradable."""
        if not signals:
            return []

        now = utcnow()
        kept: list[MarketConfluenceSignal] = []
        deactivated = 0
        metadata_updates = 0
        market_info_by_id: dict[str, Optional[dict[str, Any]]] = {}

        for signal in signals:
            market_id = str(signal.market_id or "").strip()
            if not market_id:
                continue

            cache_key = market_id.lower()
            if cache_key in market_info_by_id:
                info = market_info_by_id[cache_key]
            else:
                try:
                    if market_id.startswith("0x"):
                        info = await self.client.get_market_by_condition_id(market_id)
                    else:
                        info = await self.client.get_market_by_token_id(market_id)
                except Exception:
                    info = None
                market_info_by_id[cache_key] = info

            tradable = self.client.is_market_tradable(info, now=now) if info else True
            if tradable:
                if info:
                    changed = False
                    new_question = str(info.get("question") or "").strip()
                    new_slug = str(info.get("event_slug") or info.get("slug") or "").strip()
                    if new_question and new_question != str(signal.market_question or "").strip():
                        signal.market_question = new_question
                        changed = True
                    if new_slug and new_slug != str(signal.market_slug or "").strip():
                        signal.market_slug = new_slug
                        changed = True
                    if changed:
                        metadata_updates += 1
                kept.append(signal)
                continue

            signal.is_active = False
            signal.expired_at = now
            deactivated += 1

        if deactivated or metadata_updates:
            await session.commit()
            logger.info(
                "Pruned non-tradable confluence signals",
                checked=len(signals),
                deactivated=deactivated,
                metadata_updates=metadata_updates,
                remaining=len(kept),
            )

        return kept

    async def _refresh_signal_market_metadata(
        self,
        *,
        session,
        signals: list[MarketConfluenceSignal],
    ) -> int:
        """Refresh suspect signal market metadata to self-heal stale DB rows."""
        if not signals:
            return 0

        now = utcnow()
        question_counts = Counter(
            str(s.market_question or "").strip().lower()
            for s in signals
            if str(s.market_question or "").strip()
        )
        candidates: list[MarketConfluenceSignal] = []
        for signal in signals:
            market_id = str(signal.market_id or "").strip()
            if not market_id:
                continue
            cache_key = market_id.lower()
            last_refresh = self._signal_market_refresh_at.get(cache_key)
            if last_refresh and (now - last_refresh) < SIGNAL_METADATA_REFRESH_TTL:
                continue

            question = str(signal.market_question or "").strip()
            question_norm = question.lower()
            duplicate_question = (
                bool(question_norm)
                and question_counts.get(question_norm, 0)
                >= SIGNAL_DUPLICATE_QUESTION_THRESHOLD
            )
            missing_metadata = not question or not str(signal.market_slug or "").strip()
            if not duplicate_question and not missing_metadata:
                continue

            candidates.append(signal)
            self._signal_market_refresh_at[cache_key] = now
            if len(candidates) >= SIGNAL_METADATA_REFRESH_LIMIT:
                break

        if not candidates:
            return 0

        updated = 0
        for signal in candidates:
            market_id = str(signal.market_id or "")
            question_before = str(signal.market_question or "").strip()
            question_norm = question_before.lower()
            duplicate_question = (
                bool(question_norm)
                and question_counts.get(question_norm, 0)
                >= SIGNAL_DUPLICATE_QUESTION_THRESHOLD
            )
            try:
                if market_id.startswith("0x"):
                    info = await self.client.get_market_by_condition_id(market_id)
                else:
                    info = await self.client.get_market_by_token_id(market_id)
            except Exception:
                info = None

            if not info:
                if duplicate_question and (
                    signal.market_question is not None or signal.market_slug is not None
                ):
                    # Clear obviously poisoned metadata so fallback rendering can kick in.
                    signal.market_question = None
                    signal.market_slug = None
                    updated += 1
                continue

            new_question = str(info.get("question") or "").strip()
            new_slug = str(info.get("event_slug") or info.get("slug") or "").strip()
            changed = False
            if new_question and new_question != str(signal.market_question or "").strip():
                signal.market_question = new_question
                changed = True
            if new_slug and new_slug != str(signal.market_slug or "").strip():
                signal.market_slug = new_slug
                changed = True
            if changed:
                updated += 1

        if updated:
            await session.commit()
            logger.info("Refreshed stale confluence market metadata", signals_updated=updated)
        return updated

    # ------------------------------------------------------------------
    # Candidate collection
    # ------------------------------------------------------------------

    async def _collect_leaderboard_candidates(
        self,
        candidates: dict[str, dict[str, bool]],
        candidate_usernames: Optional[dict[str, str]],
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
                        username = (
                            row.get("userName")
                            or row.get("username")
                            or row.get("name")
                            or ""
                        )
                        username = str(username).strip()
                        if candidate_usernames is not None and username:
                            candidate_usernames[address] = username
                        candidates[address]["leaderboard"] = True
                        if sort == "PNL":
                            candidates[address]["leaderboard_pnl"] = True
                        else:
                            candidates[address]["leaderboard_vol"] = True
                        cat_key = f"leaderboard_category_{category.lower()}"
                        candidates[address][cat_key] = True

    async def _collect_wallet_trade_candidates(
        self,
        candidates: dict[str, dict[str, bool]],
        events: list[dict],
        wallet_addresses: list[str],
        per_wallet_limit: int,
    ):
        """Backfill activity rollups from per-wallet trade streams."""
        sample = [addr.lower() for addr in wallet_addresses if addr]
        if not sample:
            return

        semaphore = asyncio.Semaphore(8)

        async def _scan_wallet(address: str) -> int:
            async with semaphore:
                try:
                    trades = await self.client.get_wallet_trades(
                        address,
                        limit=min(per_wallet_limit, 200),
                    )
                except Exception as e:
                    logger.debug(
                        "Wallet trade candidate fetch failed",
                        wallet=address,
                        error=str(e),
                    )
                    return 0

                inserted = 0
                for trade in trades:
                    if not isinstance(trade, dict):
                        continue

                    market_id = str(
                        trade.get("market")
                        or trade.get("condition_id")
                        or trade.get("conditionId")
                        or trade.get("asset_id")
                        or trade.get("assetId")
                        or trade.get("asset")
                        or trade.get("token_id")
                        or trade.get("tokenId")
                        or ""
                    ).strip()
                    if not market_id:
                        continue

                    side = self._normalize_trade_side(
                        trade.get("side"),
                        trade.get("outcome"),
                    )
                    size = float(trade.get("size", 0) or trade.get("amount", 0) or 0)
                    price = float(trade.get("price", 0) or 0)
                    traded_at = self._parse_timestamp(
                        trade.get("match_time")
                        or trade.get("timestamp_iso")
                        or trade.get("timestamp")
                        or trade.get("created_at")
                        or trade.get("createdAt")
                        or trade.get("time")
                    )
                    if traded_at is None:
                        continue

                    candidates[address]["wallet_trades"] = True
                    events.append(
                        self._event_record(
                            wallet=address,
                            market_id=market_id,
                            side=side,
                            size=size,
                            price=price,
                            traded_at=traded_at,
                            source="wallet_trades_api",
                            tx_hash=trade.get("transactionHash")
                            or trade.get("tx_hash"),
                        )
                    )
                    inserted += 1

                return inserted

        counts = await asyncio.gather(*[_scan_wallet(addr) for addr in sample])
        total_inserted = sum(counts)
        if total_inserted:
            logger.debug(
                "Collected wallet trade candidate events",
                wallets=len(sample),
                events=total_inserted,
            )

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
                if not isinstance(trade, dict):
                    continue
                side = self._normalize_trade_side(
                    trade.get("side"),
                    trade.get("outcome"),
                )
                price = float(trade.get("price", 0) or 0)
                size = float(trade.get("size", 0) or trade.get("amount", 0) or 0)
                ts = self._parse_timestamp(
                    trade.get("timestamp")
                    or trade.get("timestamp_iso")
                    or trade.get("created_at")
                    or trade.get("createdAt")
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
            if not isinstance(row, dict):
                continue
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

            side = self._normalize_trade_side(
                row.get("side") or row.get("direction"),
                row.get("outcome"),
            )
            size = float(row.get("size", 0) or row.get("amount", 0) or 0)
            price = float(row.get("price", 0) or 0)
            ts = self._parse_timestamp(
                row.get("timestamp")
                or row.get("timestamp_iso")
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
        self,
        candidates: dict[str, dict[str, bool]],
        candidate_usernames: Optional[dict[str, str]] = None,
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
                discovered_username = (
                    str((candidate_usernames or {}).get(address) or "").strip() or None
                )
                if wallet is None:
                    wallet = DiscoveredWallet(
                        address=address,
                        discovered_at=utcnow(),
                        discovery_source="smart_pool",
                        username=discovered_username,
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
                if discovered_username and discovered_username != wallet.username:
                    wallet.username = discovered_username

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
            rows = [
                {
                    "id": str(uuid.uuid4()),
                    "wallet_address": event["wallet_address"],
                    "market_id": event["market_id"],
                    "side": event.get("side"),
                    "size": event.get("size"),
                    "price": event.get("price"),
                    "notional": event.get("notional"),
                    "tx_hash": event.get("tx_hash"),
                    "source": event.get("source", "unknown"),
                    "traded_at": event["traded_at"],
                }
                for event in inserts
            ]
            stmt = sqlite_insert(WalletActivityRollup).values(rows).prefix_with(
                "OR IGNORE"
            )
            result = await session.execute(stmt)
            await session.commit()
            if result is None or result.rowcount is None:
                # Conservative fallback if driver does not report rowcount.
                return len(inserts)
            return int(max(result.rowcount, 0))

    def _trim_activity_cache(self, now: datetime):
        if len(self._activity_cache) < 50_000:
            return
        cutoff = now - timedelta(hours=24)
        stale = [k for k, t in self._activity_cache.items() if t < cutoff]
        for key in stale:
            self._activity_cache.pop(key, None)

    async def _refresh_metrics_and_apply_pool(self, now: datetime) -> float:
        cutoff_1h = now - timedelta(hours=1)
        cutoff_24h = now - timedelta(hours=24)
        cutoff_72h = now - timedelta(hours=ACTIVE_WINDOW_HOURS)
        quality_only_mode = self._recompute_mode == POOL_RECOMPUTE_MODE_QUALITY_ONLY

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

            selection_scores: dict[str, float] = {}
            insider_scores: dict[str, float] = {}
            source_confidence_scores: dict[str, float] = {}
            active_recently: dict[str, bool] = {}
            tier_hint: dict[str, str] = {}
            eligibility_blockers: dict[str, list[dict[str, str]]] = {}
            eligible_addresses: set[str] = set()

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
                activity = self._score_activity(
                    trades_1h,
                    trades_24h,
                    last_trade_at,
                    now,
                    wallet=wallet,
                )
                stability = self._score_stability(wallet)
                composite = _clamp(0.45 * quality + 0.35 * activity + 0.20 * stability)
                insider = _clamp(float(wallet.insider_score or 0.0))
                source_confidence = self._score_source_confidence(wallet)
                diversity_score = _clamp(unique_markets_24h / 14.0)
                momentum = _clamp(0.6 * (trades_1h / 4.0) + 0.4 * (trades_24h / 24.0))
                selection_score = self._score_selection(
                    composite=composite,
                    rank_score=float(wallet.rank_score or 0.0),
                    insider_score=insider,
                    source_confidence=source_confidence,
                    diversity_score=diversity_score,
                    momentum_score=momentum,
                )

                wallet.trades_1h = trades_1h
                wallet.trades_24h = trades_24h
                wallet.unique_markets_24h = unique_markets_24h
                wallet.last_trade_at = last_trade_at
                wallet.quality_score = quality
                wallet.activity_score = activity
                wallet.stability_score = stability
                wallet.composite_score = composite
                selection_scores[wallet.address] = selection_score
                insider_scores[wallet.address] = insider
                source_confidence_scores[wallet.address] = source_confidence
                is_recent = bool(
                    last_trade_at is not None and last_trade_at >= cutoff_72h
                )
                active_recently[wallet.address] = is_recent

                blockers = self._eligibility_blockers(wallet)
                core_eligible, rising_eligible = self._tier_eligibility(
                    wallet,
                    is_active_recent=is_recent,
                )
                if core_eligible:
                    tier_hint[wallet.address] = "core"
                elif rising_eligible:
                    tier_hint[wallet.address] = "rising"
                else:
                    tier_hint[wallet.address] = "blocked"
                    if not blockers:
                        blockers = [
                            {
                                "code": "tier_thresholds_not_met",
                                "label": "Tier thresholds not met",
                                "detail": (
                                    "Wallet did not satisfy core or rising quality tier "
                                    "thresholds."
                                ),
                            }
                        ]
                eligibility_blockers[wallet.address] = blockers

                if self._is_pool_blocked(wallet):
                    continue
                if self._is_pool_manually_included(wallet):
                    eligible_addresses.add(wallet.address)
                    if tier_hint.get(wallet.address) == "blocked":
                        tier_hint[wallet.address] = "rising"
                    continue
                if blockers:
                    continue
                if core_eligible or rising_eligible:
                    eligible_addresses.add(wallet.address)

            ranked_wallets = sorted(
                wallets,
                key=lambda w: (
                    selection_scores.get(w.address, 0.0),
                    w.composite_score or 0.0,
                    w.rank_score or 0.0,
                ),
                reverse=True,
            )
            wallet_by_address = {w.address: w for w in wallets}
            eligible_ranked = [
                w for w in ranked_wallets if w.address in eligible_addresses
            ]
            manual_includes = [
                w.address
                for w in ranked_wallets
                if self._is_pool_manually_included(w) and not self._is_pool_blocked(w)
            ]
            core_candidates = [
                w.address
                for w in eligible_ranked
                if tier_hint.get(w.address) == "core"
            ]
            rising_candidates = [
                w.address
                for w in eligible_ranked
                if tier_hint.get(w.address) == "rising"
            ]
            diversified_ranked = self._rank_with_cluster_diversity(
                eligible_ranked,
                target_size=TARGET_POOL_SIZE,
            )

            desired: list[str] = []
            desired_set: set[str] = set()
            self._append_unique_inplace(
                desired,
                desired_set,
                manual_includes,
                TARGET_POOL_SIZE,
            )
            self._append_unique_inplace(
                desired,
                desired_set,
                core_candidates,
                int(TARGET_POOL_SIZE * 0.70),
            )
            self._append_unique_inplace(
                desired,
                desired_set,
                rising_candidates,
                TARGET_POOL_SIZE,
            )
            self._append_unique_inplace(
                desired,
                desired_set,
                [w.address for w in diversified_ranked],
                TARGET_POOL_SIZE,
            )
            if not quality_only_mode:
                self._append_unique_inplace(
                    desired,
                    desired_set,
                    [
                        w.address
                        for w in diversified_ranked
                        if active_recently.get(w.address, False)
                    ],
                    TARGET_POOL_SIZE,
                )

            current_pool = [
                w.address
                for w in wallets
                if w.in_top_pool
                and not self._is_pool_blocked(w)
                and (
                    w.address in eligible_addresses
                    or self._is_pool_manually_included(w)
                )
            ]
            final_pool, churn_rate = self._apply_churn_guard(
                desired=desired,
                current=current_pool,
                scores=selection_scores,
                quality_only_mode=quality_only_mode,
            )
            if manual_includes:
                final_pool = self._enforce_manual_includes(
                    final_pool=final_pool,
                    manual_includes=manual_includes,
                    scores=selection_scores,
                    eligible_addresses=eligible_addresses,
                )

            final_index = {address: i for i, address in enumerate(final_pool)}
            score_ranks = self._score_ranks(selection_scores)
            desired_set = set(desired)
            current_set = set(current_pool)
            cluster_counts = self._count_clusters(final_pool, wallet_by_address)
            cluster_cap = max(3, int(TARGET_POOL_SIZE * MAX_CLUSTER_SHARE))
            final_wallets = [
                wallet_by_address[address]
                for address in final_pool
                if address in wallet_by_address
            ]
            analyzed_pool_count = sum(1 for w in final_wallets if w.last_analyzed_at is not None)
            profitable_pool_count = sum(1 for w in final_wallets if bool(w.is_profitable))
            copy_candidate_pool_count = sum(
                1
                for w in final_wallets
                if str(w.recommendation or "").strip().lower() == "copy_candidate"
            )
            pool_size = len(final_wallets)
            analyzed_pool_pct = (
                round((analyzed_pool_count / pool_size) * 100.0, 2) if pool_size else 0.0
            )
            profitable_pool_pct = (
                round((profitable_pool_count / pool_size) * 100.0, 2) if pool_size else 0.0
            )
            copy_candidate_pool_pct = (
                round((copy_candidate_pool_count / pool_size) * 100.0, 2) if pool_size else 0.0
            )
            previous_copy_pct = self._last_copy_candidate_pct
            copy_pct_delta = (
                round(copy_candidate_pool_pct - previous_copy_pct, 2)
                if previous_copy_pct is not None
                else 0.0
            )
            self._last_copy_candidate_pct = copy_candidate_pool_pct

            slo_violations: list[str] = []
            if pool_size > 0 and analyzed_pool_pct < SLO_MIN_ANALYZED_PCT:
                slo_violations.append("analyzed_pool_pct_below_slo")
            if pool_size > 0 and profitable_pool_pct < SLO_MIN_PROFITABLE_PCT:
                slo_violations.append("profitable_pool_pct_below_slo")

            auto_enforced = False
            if slo_violations and self._recompute_mode == POOL_RECOMPUTE_MODE_BALANCED:
                self._recompute_mode = POOL_RECOMPUTE_MODE_QUALITY_ONLY
                auto_enforced = True
                logger.warning(
                    "Pool SLO violated; switching recompute mode to quality_only",
                    violations=slo_violations,
                    analyzed_pool_pct=analyzed_pool_pct,
                    profitable_pool_pct=profitable_pool_pct,
                    pool_size=pool_size,
                )

            self._stats["recompute_mode"] = self._recompute_mode
            self._stats["analyzed_pool_pct"] = analyzed_pool_pct
            self._stats["profitable_pool_pct"] = profitable_pool_pct
            self._stats["copy_candidate_pool_pct"] = copy_candidate_pool_pct
            self._stats["copy_candidate_pool_pct_delta"] = copy_pct_delta
            self._stats["slo_violations"] = slo_violations
            self._stats["quality_only_auto_enforced"] = auto_enforced

            for wallet in wallets:
                idx = final_index.get(wallet.address)
                reason_rows: list[dict[str, str]]
                wallet.in_top_pool = idx is not None
                if idx is None:
                    wallet.pool_tier = None
                    if self._is_pool_blacklisted(wallet):
                        reason_rows = [
                            {
                                "code": "blacklisted",
                                "label": "Blacklisted from pool",
                                "detail": "Operator blacklist flag is enabled for this wallet.",
                            }
                        ]
                    elif self._is_pool_manually_excluded(wallet):
                        reason_rows = [
                            {
                                "code": "manual_exclude",
                                "label": "Manually excluded",
                                "detail": "Operator manual exclusion flag is enabled for this wallet.",
                            }
                        ]
                    elif eligibility_blockers.get(wallet.address):
                        reason_rows = list(eligibility_blockers.get(wallet.address, []))
                    else:
                        reason_rows = [
                            {
                                "code": "below_selection_cutoff",
                                "label": "Below pool cutoff",
                                "detail": "Wallet did not clear current rank/churn thresholds for the active pool.",
                            }
                        ]
                else:
                    hint = tier_hint.get(wallet.address)
                    wallet.pool_tier = "core" if hint == "core" else "rising"
                    reason_rows = self._derive_selection_reasons(
                        wallet=wallet,
                        address=wallet.address,
                        selection_score=selection_scores.get(wallet.address, 0.0),
                        insider_score=insider_scores.get(wallet.address, 0.0),
                        cutoff_72h=cutoff_72h,
                        desired_addresses=desired_set,
                        current_addresses=current_set,
                        cluster_count=cluster_counts.get(self._cluster_id(wallet), 0),
                        cluster_cap=cluster_cap,
                    )

                primary_reason = reason_rows[0]["code"] if reason_rows else None
                wallet.pool_membership_reason = primary_reason

                rank = score_ranks.get(wallet.address)
                percentile = self._rank_percentile(rank, len(score_ranks))
                source_flags = self._source_flags(wallet)
                if not isinstance(source_flags, dict):
                    source_flags = {}
                analysis_freshness_hours = self._analysis_freshness_hours(wallet, now)
                status = "eligible" if wallet.address in eligible_addresses else "blocked"
                blockers_payload = list(eligibility_blockers.get(wallet.address, []))
                if self._is_pool_manually_included(wallet) and not self._is_pool_blocked(wallet):
                    status = "eligible"
                source_flags[POOL_SELECTION_META_KEY] = {
                    "version": POOL_SELECTION_META_VERSION,
                    "quality_gate_version": QUALITY_GATE_VERSION,
                    "recompute_mode": self._recompute_mode,
                    "eligibility_status": status,
                    "eligibility_blockers": blockers_payload,
                    "analysis_freshness_hours": analysis_freshness_hours,
                    "selection_score": round(selection_scores.get(wallet.address, 0.0), 6),
                    "selection_rank": int(rank) if rank is not None else None,
                    "selection_percentile": percentile,
                    "reasons": reason_rows,
                    "score_breakdown": {
                        "composite_score": round(float(wallet.composite_score or 0.0), 6),
                        "quality_score": round(float(wallet.quality_score or 0.0), 6),
                        "activity_score": round(float(wallet.activity_score or 0.0), 6),
                        "stability_score": round(float(wallet.stability_score or 0.0), 6),
                        "rank_score": round(float(wallet.rank_score or 0.0), 6),
                        "insider_score": round(insider_scores.get(wallet.address, 0.0), 6),
                        "source_confidence": round(
                            source_confidence_scores.get(wallet.address, 0.0),
                            6,
                        ),
                        "trades_1h": int(wallet.trades_1h or 0),
                        "trades_24h": int(wallet.trades_24h or 0),
                        "unique_markets_24h": int(wallet.unique_markets_24h or 0),
                    },
                    "active_within_hours": (
                        round(
                            max((now - wallet.last_trade_at).total_seconds(), 0.0) / 3600.0,
                            2,
                        )
                        if wallet.last_trade_at
                        else None
                    ),
                    "updated_at": now.isoformat(),
                }
                wallet.source_flags = source_flags

            await session.commit()

        await self._sync_ws_membership(final_pool)
        return churn_rate

    def _source_flags(self, wallet: DiscoveredWallet) -> dict[str, Any]:
        raw = wallet.source_flags or {}
        return raw if isinstance(raw, dict) else {}

    def _analysis_freshness_hours(
        self,
        wallet: DiscoveredWallet,
        now: datetime,
    ) -> Optional[float]:
        if wallet.last_analyzed_at is None:
            return None
        return round(
            max((now - wallet.last_analyzed_at).total_seconds(), 0.0) / 3600.0,
            2,
        )

    def _eligibility_blockers(self, wallet: DiscoveredWallet) -> list[dict[str, str]]:
        blockers: list[dict[str, str]] = []
        recommendation = str(getattr(wallet, "recommendation", "") or "").strip().lower()
        total_trades = int(wallet.total_trades or 0)
        total_pnl = float(wallet.total_pnl or 0.0)
        anomaly = float(wallet.anomaly_score or 0.0)

        if wallet.last_analyzed_at is None:
            blockers.append(
                {
                    "code": "not_analyzed",
                    "label": "Analysis missing",
                    "detail": "Wallet has not completed a discovery analysis pass yet.",
                }
            )
        if recommendation not in {"copy_candidate", "monitor"}:
            blockers.append(
                {
                    "code": "recommendation_blocked",
                    "label": "Recommendation blocked",
                    "detail": (
                        "Wallet recommendation must be copy_candidate or monitor "
                        "for pool eligibility."
                    ),
                }
            )
        if total_trades < MIN_ELIGIBLE_TRADES:
            blockers.append(
                {
                    "code": "insufficient_trades",
                    "label": "Insufficient trade sample",
                    "detail": f"Wallet needs at least {MIN_ELIGIBLE_TRADES} total trades.",
                }
            )
        if anomaly > MAX_ELIGIBLE_ANOMALY:
            blockers.append(
                {
                    "code": "anomaly_too_high",
                    "label": "Anomaly score too high",
                    "detail": (
                        f"Wallet anomaly score must be <= {MAX_ELIGIBLE_ANOMALY:.2f} "
                        "to enter the pool."
                    ),
                }
            )
        if total_pnl <= 0:
            blockers.append(
                {
                    "code": "non_positive_pnl",
                    "label": "Non-positive PnL",
                    "detail": "Wallet total PnL must be positive for pool inclusion.",
                }
            )
        return blockers

    def _tier_eligibility(
        self,
        wallet: DiscoveredWallet,
        *,
        is_active_recent: bool,
    ) -> tuple[bool, bool]:
        win_rate = float(wallet.win_rate or 0.0)
        total_trades = int(wallet.total_trades or 0)
        total_pnl = float(wallet.total_pnl or 0.0)
        recommendation = str(getattr(wallet, "recommendation", "") or "").strip().lower()
        sharpe = wallet.sharpe_ratio
        profit_factor = wallet.profit_factor
        sharpe_ok = sharpe is not None and math.isfinite(sharpe) and sharpe >= CORE_MIN_SHARPE
        profit_factor_ok = (
            profit_factor is not None
            and math.isfinite(profit_factor)
            and profit_factor >= CORE_MIN_PROFIT_FACTOR
        )

        core_eligible = recommendation == "copy_candidate" or (
            win_rate >= CORE_MIN_WIN_RATE and sharpe_ok and profit_factor_ok
        )
        rising_eligible = (
            is_active_recent
            and win_rate >= RISING_MIN_WIN_RATE
            and total_pnl > 0
            and total_trades >= MIN_ELIGIBLE_TRADES
        )
        return core_eligible, rising_eligible

    def _is_pool_blacklisted(self, wallet: DiscoveredWallet) -> bool:
        return bool(self._source_flags(wallet).get(POOL_FLAG_BLACKLISTED))

    def _is_pool_manually_excluded(self, wallet: DiscoveredWallet) -> bool:
        return bool(self._source_flags(wallet).get(POOL_FLAG_MANUAL_EXCLUDE))

    def _is_pool_manually_included(self, wallet: DiscoveredWallet) -> bool:
        return bool(self._source_flags(wallet).get(POOL_FLAG_MANUAL_INCLUDE))

    def _is_pool_blocked(self, wallet: DiscoveredWallet) -> bool:
        return self._is_pool_blacklisted(wallet) or self._is_pool_manually_excluded(wallet)

    def _prepend_unique(self, prioritized: list[str], base: list[str]) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for address in prioritized + base:
            if address in seen:
                continue
            ordered.append(address)
            seen.add(address)
        return ordered

    def _enforce_manual_includes(
        self,
        final_pool: list[str],
        manual_includes: list[str],
        scores: dict[str, float],
        eligible_addresses: set[str],
    ) -> list[str]:
        ordered = [a for a in final_pool if a in eligible_addresses]
        for address in manual_includes:
            if address not in eligible_addresses:
                continue
            if address in ordered:
                continue
            if len(ordered) >= MAX_POOL_SIZE:
                ordered.pop()
            ordered.append(address)

        # Re-rank after manual additions and enforce bounds.
        ordered = sorted(ordered, key=lambda a: scores.get(a, 0.0), reverse=True)
        if len(ordered) > MAX_POOL_SIZE:
            ordered = ordered[:MAX_POOL_SIZE]
        return ordered

    def _score_source_confidence(self, wallet: DiscoveredWallet) -> float:
        flags = self._source_flags(wallet)
        if not isinstance(flags, dict):
            return 0.0

        score = 0.0
        if flags.get("leaderboard"):
            score += 0.35
        if flags.get("leaderboard_pnl"):
            score += 0.15
        if flags.get("leaderboard_vol"):
            score += 0.10
        if flags.get("wallet_trades"):
            score += 0.15
        if flags.get("market_trades"):
            score += 0.10
        if flags.get("activity"):
            score += 0.10
        if flags.get("holders"):
            score += 0.05
        return _clamp(score)

    def _score_selection(
        self,
        *,
        composite: float,
        rank_score: float,
        insider_score: float,
        source_confidence: float,
        diversity_score: float,
        momentum_score: float,
    ) -> float:
        return _clamp(
            0.62 * _clamp(composite)
            + 0.16 * _clamp(rank_score)
            + 0.08 * _clamp(insider_score)
            + 0.06 * _clamp(source_confidence)
            + 0.05 * _clamp(diversity_score)
            + 0.03 * _clamp(momentum_score)
        )

    def _append_unique_inplace(
        self,
        base: list[str],
        seen: set[str],
        candidates: list[str],
        limit: int,
    ) -> None:
        if len(base) >= limit:
            return
        for address in candidates:
            if address in seen:
                continue
            base.append(address)
            seen.add(address)
            if len(base) >= limit:
                break

    def _cluster_id(self, wallet: DiscoveredWallet) -> str:
        return str(wallet.cluster_id or "").strip().lower()

    def _rank_with_cluster_diversity(
        self,
        wallets: list[DiscoveredWallet],
        target_size: int,
    ) -> list[DiscoveredWallet]:
        if not wallets:
            return []

        cap = max(3, int(target_size * MAX_CLUSTER_SHARE))
        selected: list[DiscoveredWallet] = []
        deferred: list[DiscoveredWallet] = []
        cluster_counts: dict[str, int] = defaultdict(int)

        for wallet in wallets:
            cluster = self._cluster_id(wallet)
            if not cluster:
                selected.append(wallet)
                continue
            if cluster_counts[cluster] < cap:
                selected.append(wallet)
                cluster_counts[cluster] += 1
            else:
                deferred.append(wallet)

        # Keep pool fill robust: only enforce cap while enough alternatives exist.
        if len(selected) < target_size:
            for wallet in deferred:
                selected.append(wallet)
                if len(selected) >= target_size:
                    break
        return selected

    def _count_clusters(
        self,
        addresses: list[str],
        wallet_by_address: dict[str, DiscoveredWallet],
    ) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for address in addresses:
            wallet = wallet_by_address.get(address)
            if wallet is None:
                continue
            cluster = self._cluster_id(wallet)
            if cluster:
                counts[cluster] += 1
        return counts

    def _score_ranks(self, scores: dict[str, float]) -> dict[str, int]:
        ordered = sorted(scores.keys(), key=lambda address: scores.get(address, 0.0), reverse=True)
        return {address: idx + 1 for idx, address in enumerate(ordered)}

    def _rank_percentile(self, rank: Optional[int], total: int) -> Optional[float]:
        if rank is None or total <= 0:
            return None
        if total == 1:
            return 1.0
        return round(_clamp(1.0 - ((rank - 1) / max(total - 1, 1))), 6)

    def _derive_selection_reasons(
        self,
        *,
        wallet: DiscoveredWallet,
        address: str,
        selection_score: float,
        insider_score: float,
        cutoff_72h: datetime,
        desired_addresses: set[str],
        current_addresses: set[str],
        cluster_count: int,
        cluster_cap: int,
    ) -> list[dict[str, str]]:
        reasons: list[dict[str, str]] = []

        if self._is_pool_manually_included(wallet):
            reasons.append(
                {
                    "code": "manual_include",
                    "label": "Manual include override",
                    "detail": "Operator forced this wallet into the pool.",
                }
            )

        pool_tier = str(getattr(wallet, "pool_tier", "") or "").lower()
        if pool_tier == "core":
            reasons.append(
                {
                    "code": "core_quality_gate",
                    "label": "Core quality tier",
                    "detail": (
                        "Wallet passed hard quality gates and core-tier strategy "
                        "requirements."
                    ),
                }
            )
        elif pool_tier == "rising":
            reasons.append(
                {
                    "code": "rising_quality_gate",
                    "label": "Rising quality tier",
                    "detail": (
                        "Wallet passed hard quality gates and rising-tier activity "
                        "requirements."
                    ),
                }
            )

        if address in current_addresses and address not in desired_addresses:
            reasons.append(
                {
                    "code": "churn_guard_retained",
                    "label": "Churn guard retention",
                    "detail": "Retained to avoid excessive hourly pool turnover.",
                }
            )

        if selection_score >= HIGH_CONVICTION_THRESHOLD and (wallet.quality_score or 0.0) >= 0.55:
            reasons.append(
                {
                    "code": "elite_composite",
                    "label": "Elite composite profile",
                    "detail": "High combined quality/activity/stability score.",
                }
            )

        if (wallet.trades_1h or 0) > 0 or (wallet.trades_24h or 0) >= 6:
            reasons.append(
                {
                    "code": "active_momentum",
                    "label": "Active momentum",
                    "detail": "Recent trade velocity meets pool momentum thresholds.",
                }
            )
        elif wallet.last_trade_at and wallet.last_trade_at >= cutoff_72h:
            reasons.append(
                {
                    "code": "active_recent",
                    "label": "Recent activity",
                    "detail": "Traded recently inside the active-window requirement.",
                }
            )

        if insider_score >= INSIDER_PRIORITY_THRESHOLD and (wallet.trades_24h or 0) >= 2:
            reasons.append(
                {
                    "code": "insider_alignment",
                    "label": "Insider-aligned signal",
                    "detail": "Elevated insider score with confirmed recent activity.",
                }
            )

        cluster = self._cluster_id(wallet)
        if cluster and cluster_count >= cluster_cap:
            reasons.append(
                {
                    "code": "cluster_capped",
                    "label": "Cluster-capped slot",
                    "detail": "Included while respecting per-cluster concentration limits.",
                }
            )

        if not reasons:
            reasons.append(
                {
                    "code": "quality_gate_pass",
                    "label": "Quality gate pass",
                    "detail": "Wallet passed quality-first eligibility and ranking checks.",
                }
            )
        return reasons

    def _score_quality(self, wallet: DiscoveredWallet) -> float:
        rank = _clamp(float(wallet.rank_score or 0.0))
        win = _clamp(float(wallet.win_rate or 0.0))

        sharpe = wallet.sharpe_ratio
        sharpe_norm = (
            0.0
            if sharpe is None or not math.isfinite(sharpe)
            else _clamp(sharpe / 3.0)
        )

        pf = wallet.profit_factor
        pf_norm = (
            0.0
            if pf is None or not math.isfinite(pf)
            else _clamp(pf / 5.0)
        )

        pnl = float(wallet.total_pnl or 0.0)
        pnl_norm = _clamp((math.tanh(pnl / 25000.0) + 1.0) / 2.0)

        recommendation = str(getattr(wallet, "recommendation", "") or "").strip().lower()
        recommendation_boost = 0.0
        if recommendation == "copy_candidate":
            recommendation_boost = 0.08
        elif recommendation == "monitor":
            recommendation_boost = 0.02

        return _clamp(
            0.35 * rank
            + 0.25 * win
            + 0.15 * sharpe_norm
            + 0.15 * pf_norm
            + 0.10 * pnl_norm
            + recommendation_boost
        )

    def _score_activity(
        self,
        trades_1h: int,
        trades_24h: int,
        last_trade_at: Optional[datetime],
        now: datetime,
        wallet: Optional[DiscoveredWallet] = None,
    ) -> float:
        flow_1h = _clamp(trades_1h / 6.0)
        flow_24h = _clamp(trades_24h / 40.0)

        if last_trade_at is None:
            recency = 0.0
        else:
            age_hours = max((now - last_trade_at).total_seconds() / 3600.0, 0.0)
            recency = _clamp(1.0 - (age_hours / ACTIVE_WINDOW_HOURS))

        base_score = _clamp(0.50 * flow_1h + 0.30 * flow_24h + 0.20 * recency)
        if wallet is None:
            return base_score

        # Downweight activity for wallets with unverified/legacy analysis profiles.
        source_version = str(wallet.metrics_source_version or "").strip()
        analysis_verified = (
            wallet.last_analyzed_at is not None
            and source_version == QUALITY_METRICS_SOURCE_VERSION
        )
        if analysis_verified:
            return base_score
        return _clamp(base_score * 0.35)

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
        quality_only_mode: bool = False,
    ) -> tuple[list[str], float]:
        desired = desired[:TARGET_POOL_SIZE]
        current = current[:MAX_POOL_SIZE]

        if quality_only_mode:
            final = desired[:MAX_POOL_SIZE]
            current_set = set(current)
            final_set = set(final)
            if not current_set:
                return final, 0.0
            churn = len(current_set.symmetric_difference(final_set)) / max(
                len(current_set), 1
            )
            return final, churn

        # If no existing pool, initialize directly from desired.
        if not current:
            initialized = desired[:TARGET_POOL_SIZE]
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

        # Gradually trim stale members when desired set is materially smaller.
        if len(pool_set) > len(desired):
            trim_candidates = sorted(
                [a for a in pool_set if a not in desired_set],
                key=lambda a: scores.get(a, 0.0),
            )
            for address in trim_candidates:
                if replacements >= max_replacements:
                    break
                pool_set.discard(address)
                replacements += 1
                if len(pool_set) <= len(desired):
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

    def _normalize_trade_side(self, side_raw: Any, outcome_raw: Any = None) -> str:
        side = str(side_raw or "").strip().upper()
        if side in {"BUY", "YES"}:
            return "BUY"
        if side in {"SELL", "NO"}:
            return "SELL"

        outcome = str(outcome_raw or "").strip().upper()
        if outcome in {"YES", "BUY"}:
            return "BUY"
        if outcome in {"NO", "SELL"}:
            return "SELL"
        return side or "TRADE"

    def _parse_timestamp(self, raw: Any) -> Optional[datetime]:
        if raw is None:
            return None
        if isinstance(raw, datetime):
            return raw
        if isinstance(raw, (int, float)):
            try:
                ts = float(raw)
                if ts > 10_000_000_000:  # likely milliseconds
                    ts /= 1000.0
                return utcfromtimestamp(ts)
            except (OSError, ValueError):
                return None
        if isinstance(raw, str):
            try:
                text = raw.strip()
                if not text:
                    return None

                numeric = text.replace(".", "", 1)
                if numeric.startswith("-"):
                    numeric = numeric[1:]
                if numeric.isdigit():
                    ts = float(text)
                    if ts > 10_000_000_000:  # likely milliseconds
                        ts /= 1000.0
                    return utcfromtimestamp(ts)
                return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(
                    tzinfo=None
                )
            except (OSError, ValueError, TypeError):
                return None
        return None


smart_wallet_pool = SmartWalletPoolService()
