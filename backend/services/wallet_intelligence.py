"""
Wallet Intelligence Layer for Homerun Arbitrage Scanner.

Four subsystems:
  1. ConfluenceDetector  - Multi-wallet convergence on the same market
  2. EntityClusterer     - Groups wallets that likely belong to the same entity
  3. WalletTagger        - Auto-classifies wallets with behavioral tags
  4. CrossPlatformTracker - Tracks traders across Polymarket and Kalshi

Orchestrated by the WalletIntelligence class, which runs all subsystems
on a configurable schedule.
"""

import asyncio
import math
import uuid
from datetime import datetime, timedelta
from utils.utcnow import utcnow
from typing import Optional

from sqlalchemy import select, text, update, func

from models.database import (
    DiscoveredWallet,
    WalletTag,
    WalletCluster,
    MarketConfluenceSignal,
    CrossPlatformEntity,
    WalletActivityRollup,
    AsyncSessionLocal,
)
from services.polymarket import polymarket_client
from services.pause_state import global_pause_state
from utils.logger import get_logger

logger = get_logger("wallet_intelligence")


# ============================================================================
#  SUBSYSTEM 1: Multi-Wallet Confluence Detection (Priority 4)
# ============================================================================


class ConfluenceDetector:
    """Detect markets where many smart wallets enter the same side together."""

    MIN_WALLETS_WATCH = 5
    MIN_WALLETS_HIGH = 10
    MIN_WALLETS_EXTREME = 15
    MIN_WALLET_RANK_SCORE = 0.20
    SIGNAL_DECAY_MINUTES = 90

    def __init__(self):
        self._ws_broadcast_callback = None

    def set_ws_broadcast(self, callback):
        """Set websocket broadcast callback for tracked trader signal events."""
        self._ws_broadcast_callback = callback

    async def _broadcast_signal_event(self, payload: dict):
        if not self._ws_broadcast_callback:
            return
        try:
            await self._ws_broadcast_callback(
                {"type": "tracked_trader_signal", "data": payload}
            )
        except Exception:
            pass

    async def scan_for_confluence(self) -> list[dict]:
        """Scan recent wallet activity rollups and upsert confluence signals."""
        logger.info("Starting confluence scan...")
        now = utcnow()
        cutoff_60m = now - timedelta(minutes=60)
        cutoff_15m = now - timedelta(minutes=15)

        wallets = await self._get_qualifying_wallets()
        if not wallets:
            logger.info("No qualifying wallets found for confluence scan")
            return []

        wallet_map = {w["address"]: w for w in wallets}
        addresses = list(wallet_map.keys())

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(WalletActivityRollup)
                .where(
                    WalletActivityRollup.wallet_address.in_(addresses),
                    WalletActivityRollup.traded_at >= cutoff_60m,
                )
                .order_by(WalletActivityRollup.traded_at.desc())
            )
            events = list(result.scalars().all())

        if not events:
            await self.expire_old_signals()
            return await self.get_active_signals()

        grouped: dict[tuple[str, str], list[WalletActivityRollup]] = {}
        for event in events:
            side = self._normalize_side(event.side)
            if side not in ("BUY", "SELL"):
                continue
            key = (event.market_id, side)
            grouped.setdefault(key, []).append(event)

        signals_created = 0
        for (market_id, side), side_events in grouped.items():
            unique_wallets = {
                e.wallet_address.lower()
                for e in side_events
                if e.wallet_address
            }
            if len(unique_wallets) < self.MIN_WALLETS_WATCH:
                continue

            # Calculate adjusted wallet count by entity cluster to reduce
            # multi-wallet inflation from the same controller.
            cluster_groups: dict[str, set[str]] = {}
            unclustered = 0
            for addr in unique_wallets:
                cluster_id = wallet_map.get(addr, {}).get("cluster_id")
                if cluster_id:
                    cluster_groups.setdefault(cluster_id, set()).add(addr)
                else:
                    unclustered += 1
            cluster_adjusted = len(cluster_groups) + unclustered
            if cluster_adjusted < self.MIN_WALLETS_WATCH:
                continue

            window_events = [e for e in side_events if e.traded_at and e.traded_at >= cutoff_15m]
            window_minutes = 15 if window_events else 60
            active_events = window_events or side_events

            wallets_sorted = sorted(unique_wallets)
            avg_rank = 0.0
            weighted_wallet_score = 0.0
            anomaly_avg = 0.0
            core_wallets = 0
            if wallets_sorted:
                rank_vals = [wallet_map.get(w, {}).get("rank_score", 0.0) for w in wallets_sorted]
                comp_vals = [wallet_map.get(w, {}).get("composite_score", 0.0) for w in wallets_sorted]
                an_vals = [wallet_map.get(w, {}).get("anomaly_score", 0.0) for w in wallets_sorted]
                avg_rank = sum(rank_vals) / len(rank_vals)
                weighted_wallet_score = sum(comp_vals) / len(comp_vals)
                anomaly_avg = sum(an_vals) / len(an_vals)
                core_wallets = sum(1 for w in wallets_sorted if wallet_map.get(w, {}).get("in_top_pool"))

            prices = [float(e.price or 0) for e in active_events if e.price and e.price > 0]
            sizes = [abs(float(e.size or 0)) for e in active_events if e.size]
            avg_entry_price = sum(prices) / len(prices) if prices else None
            total_size = sum(sizes) if sizes else None

            net_notional = sum(abs(float(e.notional or 0.0)) for e in active_events)
            conflicting_notional = await self._conflicting_notional(
                market_id=market_id,
                side=side,
                addresses=addresses,
                cutoff=cutoff_60m,
            )

            timestamps = [e.traded_at for e in active_events if e.traded_at]
            timing_tightness = self._timing_tightness(timestamps)

            market_context = await self._resolve_market_context(market_id)
            outcome = "YES" if side == "BUY" else "NO"
            signal_type = "multi_wallet_buy" if side == "BUY" else "multi_wallet_sell"
            tier = self._tier_for_count(cluster_adjusted)
            conviction = self._conviction_score(
                adjusted_wallet_count=cluster_adjusted,
                weighted_wallet_score=weighted_wallet_score,
                timing_tightness=timing_tightness,
                net_notional=net_notional,
                conflicting_notional=conflicting_notional,
                market_liquidity=market_context.get("liquidity"),
                market_volume_24h=market_context.get("volume_24h"),
                anomaly_avg=anomaly_avg,
                unique_wallet_count=len(unique_wallets),
            )
            strength = max(0.0, min(conviction / 100.0, 1.0))

            await self._upsert_signal(
                market_id=market_id,
                market_question=market_context.get("question") or "",
                signal_type=signal_type,
                strength=strength,
                conviction_score=conviction,
                tier=tier,
                window_minutes=window_minutes,
                wallet_count=len(unique_wallets),
                cluster_adjusted_wallet_count=cluster_adjusted,
                unique_core_wallets=core_wallets,
                weighted_wallet_score=weighted_wallet_score,
                wallets=wallets_sorted,
                outcome=outcome,
                avg_entry_price=avg_entry_price,
                total_size=total_size,
                avg_wallet_rank=avg_rank,
                net_notional=net_notional,
                conflicting_notional=conflicting_notional,
                market_slug=market_context.get("market_slug"),
                market_liquidity=market_context.get("liquidity"),
                market_volume_24h=market_context.get("volume_24h"),
            )
            if tier in ("HIGH", "EXTREME"):
                await self._broadcast_signal_event(
                    {
                        "market_id": market_id,
                        "market_question": market_context.get("question") or "",
                        "market_slug": market_context.get("market_slug"),
                        "outcome": outcome,
                        "tier": tier,
                        "conviction_score": conviction,
                        "cluster_adjusted_wallet_count": cluster_adjusted,
                        "wallet_count": len(unique_wallets),
                        "window_minutes": window_minutes,
                        "detected_at": now.isoformat(),
                    }
                )
            signals_created += 1

        await self.expire_old_signals()
        active = await self.get_active_signals()
        logger.info(
            "Confluence scan complete",
            signals_created=signals_created,
            active_signals=len(active),
        )
        return active

    async def _get_qualifying_wallets(self) -> list[dict]:
        """Get wallets eligible for confluence scoring."""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(DiscoveredWallet).where(
                    DiscoveredWallet.rank_score >= self.MIN_WALLET_RANK_SCORE,
                )
            )
            wallets = list(result.scalars().all())
            return [
                {
                    "address": w.address,
                    "rank_score": w.rank_score or 0.0,
                    "composite_score": w.composite_score or 0.0,
                    "anomaly_score": w.anomaly_score or 0.0,
                    "cluster_id": w.cluster_id,
                    "in_top_pool": bool(w.in_top_pool),
                }
                for w in wallets
            ]

    def _normalize_side(self, raw: Optional[str]) -> str:
        side = (raw or "").upper().strip()
        if side in ("BUY", "YES"):
            return "BUY"
        if side in ("SELL", "NO"):
            return "SELL"
        return side

    def _tier_for_count(self, adjusted_wallet_count: int) -> str:
        if adjusted_wallet_count >= self.MIN_WALLETS_EXTREME:
            return "EXTREME"
        if adjusted_wallet_count >= self.MIN_WALLETS_HIGH:
            return "HIGH"
        return "WATCH"

    def _timing_tightness(self, timestamps: list[datetime]) -> float:
        """Return 0-1 timing tightness where 1 means near-simultaneous entries."""
        if len(timestamps) < 2:
            return 0.5
        ts_sorted = sorted(timestamps)
        span_seconds = (ts_sorted[-1] - ts_sorted[0]).total_seconds()
        # <=2 minutes => 1.0, 60 minutes => 0.0
        return max(0.0, min(1.0, 1.0 - (span_seconds / 3600.0)))

    def _conviction_score(
        self,
        adjusted_wallet_count: int,
        weighted_wallet_score: float,
        timing_tightness: float,
        net_notional: float,
        conflicting_notional: float,
        market_liquidity: Optional[float],
        market_volume_24h: Optional[float],
        anomaly_avg: float,
        unique_wallet_count: int,
    ) -> float:
        # Count strength (30)
        count_factor = min(adjusted_wallet_count / float(self.MIN_WALLETS_EXTREME), 1.0)
        count_component = 30.0 * count_factor

        # Wallet quality mix (25)
        quality_component = 25.0 * max(0.0, min(weighted_wallet_score, 1.0))

        # Entry-time tightness (15)
        timing_component = 15.0 * max(0.0, min(timing_tightness, 1.0))

        # Net notional (10, log scale)
        if net_notional > 0:
            notional_factor = min(math.log10(net_notional + 1.0) / 5.0, 1.0)
        else:
            notional_factor = 0.0
        notional_component = 10.0 * notional_factor

        # Side consensus (10)
        if net_notional > 0:
            conflict_ratio = min(conflicting_notional / max(net_notional, 1.0), 1.0)
            consensus_factor = 1.0 - conflict_ratio
        else:
            consensus_factor = 0.5
        consensus_component = 10.0 * consensus_factor

        # Market context (10)
        context_factor = 0.5
        if market_liquidity and market_liquidity > 0:
            context_factor = max(
                context_factor, min(math.log10(market_liquidity + 1.0) / 6.0, 1.0)
            )
        if market_volume_24h and market_volume_24h > 0:
            context_factor = max(
                context_factor, min(math.log10(market_volume_24h + 1.0) / 6.0, 1.0)
            )
        context_component = 10.0 * context_factor

        base = (
            count_component
            + quality_component
            + timing_component
            + notional_component
            + consensus_component
            + context_component
        )

        # Penalties
        concentration_penalty = 0.0
        if unique_wallet_count > 0:
            concentration = 1.0 - (adjusted_wallet_count / float(unique_wallet_count))
            concentration_penalty = max(0.0, concentration) * 10.0
        anomaly_penalty = max(0.0, min(anomaly_avg, 1.0)) * 8.0

        conviction = base - concentration_penalty - anomaly_penalty
        return round(max(0.0, min(conviction, 100.0)), 2)

    async def _conflicting_notional(
        self,
        market_id: str,
        side: str,
        addresses: list[str],
        cutoff: datetime,
    ) -> float:
        opposite = "SELL" if side == "BUY" else "BUY"
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(func.sum(WalletActivityRollup.notional))
                .where(
                    WalletActivityRollup.market_id == market_id,
                    WalletActivityRollup.wallet_address.in_(addresses),
                    WalletActivityRollup.traded_at >= cutoff,
                    WalletActivityRollup.side.in_(
                        [opposite, "YES" if opposite == "BUY" else "NO"]
                    ),
                )
            )
            value = result.scalar() or 0.0
            return float(value or 0.0)

    async def _resolve_market_context(self, market_id: str) -> dict:
        market_slug = None
        question = ""
        liquidity = None
        volume_24h = None

        try:
            if market_id.startswith("0x"):
                info = await polymarket_client.get_market_by_condition_id(market_id)
            else:
                info = await polymarket_client.get_market_by_token_id(market_id)
            if info:
                market_slug = info.get("event_slug") or info.get("slug")
                question = info.get("question", "")
                if info.get("liquidity") is not None:
                    liquidity = float(info.get("liquidity") or 0)
                if info.get("volume") is not None:
                    volume_24h = float(info.get("volume") or 0)
        except Exception:
            pass

        return {
            "market_slug": market_slug,
            "question": question,
            "liquidity": liquidity,
            "volume_24h": volume_24h,
        }

    async def _upsert_signal(
        self,
        market_id: str,
        market_question: str,
        signal_type: str,
        strength: float,
        conviction_score: float,
        tier: str,
        window_minutes: int,
        wallet_count: int,
        cluster_adjusted_wallet_count: int,
        unique_core_wallets: int,
        weighted_wallet_score: float,
        wallets: list[str],
        outcome: Optional[str],
        avg_entry_price: Optional[float],
        total_size: Optional[float],
        avg_wallet_rank: Optional[float],
        net_notional: Optional[float],
        conflicting_notional: Optional[float],
        market_slug: Optional[str] = None,
        market_liquidity: Optional[float] = None,
        market_volume_24h: Optional[float] = None,
    ):
        """Create or update a confluence signal in the database."""
        now = utcnow()
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(MarketConfluenceSignal).where(
                    MarketConfluenceSignal.market_id == market_id,
                    MarketConfluenceSignal.outcome == outcome,
                    MarketConfluenceSignal.is_active == True,  # noqa: E712
                )
            )
            existing = result.scalars().first()

            if existing:
                existing.signal_type = signal_type
                existing.strength = strength
                existing.conviction_score = conviction_score
                existing.tier = tier
                existing.window_minutes = window_minutes
                existing.wallet_count = wallet_count
                existing.cluster_adjusted_wallet_count = cluster_adjusted_wallet_count
                existing.unique_core_wallets = unique_core_wallets
                existing.weighted_wallet_score = weighted_wallet_score
                existing.wallets = wallets
                existing.avg_entry_price = avg_entry_price
                existing.total_size = total_size
                existing.avg_wallet_rank = avg_wallet_rank
                existing.net_notional = net_notional
                existing.conflicting_notional = conflicting_notional
                existing.market_liquidity = market_liquidity
                existing.market_volume_24h = market_volume_24h
                existing.last_seen_at = now
                existing.detected_at = now
                if market_slug:
                    existing.market_slug = market_slug
                if market_question:
                    existing.market_question = market_question
                # Cooldown is used by downstream alerting so repeated scans
                # do not spam notifications.
                if conviction_score >= (existing.conviction_score or 0) + 5:
                    existing.cooldown_until = now + timedelta(minutes=5)
            else:
                session.add(
                    MarketConfluenceSignal(
                        id=str(uuid.uuid4()),
                        market_id=market_id,
                        market_question=market_question,
                        market_slug=market_slug,
                        signal_type=signal_type,
                        strength=strength,
                        conviction_score=conviction_score,
                        tier=tier,
                        window_minutes=window_minutes,
                        wallet_count=wallet_count,
                        cluster_adjusted_wallet_count=cluster_adjusted_wallet_count,
                        unique_core_wallets=unique_core_wallets,
                        weighted_wallet_score=weighted_wallet_score,
                        wallets=wallets,
                        outcome=outcome,
                        avg_entry_price=avg_entry_price,
                        total_size=total_size,
                        avg_wallet_rank=avg_wallet_rank,
                        net_notional=net_notional,
                        conflicting_notional=conflicting_notional,
                        market_liquidity=market_liquidity,
                        market_volume_24h=market_volume_24h,
                        is_active=True,
                        first_seen_at=now,
                        last_seen_at=now,
                        detected_at=now,
                        cooldown_until=now + timedelta(minutes=5),
                    )
                )

            await session.commit()

    async def get_active_signals(
        self,
        min_strength: float = 0.0,
        limit: int = 50,
        min_tier: str = "WATCH",
    ) -> list[dict]:
        """Get active confluence signals from DB, sorted by conviction."""
        tier_rank = {"WATCH": 1, "HIGH": 2, "EXTREME": 3}
        required_rank = tier_rank.get((min_tier or "WATCH").upper(), 1)

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(MarketConfluenceSignal)
                .where(
                    MarketConfluenceSignal.is_active == True,  # noqa: E712
                    MarketConfluenceSignal.strength >= min_strength,
                )
                .order_by(
                    MarketConfluenceSignal.conviction_score.desc(),
                    MarketConfluenceSignal.last_seen_at.desc(),
                )
                .limit(limit * 2)
            )
            raw_signals = list(result.scalars().all())

            signals = [
                s
                for s in raw_signals
                if tier_rank.get((s.tier or "WATCH").upper(), 1) >= required_rank
            ][:limit]

            return [
                {
                    "id": s.id,
                    "market_id": s.market_id,
                    "market_question": s.market_question or "",
                    "market_slug": s.market_slug or None,
                    "signal_type": s.signal_type,
                    "tier": s.tier or "WATCH",
                    "strength": s.strength,
                    "conviction_score": s.conviction_score or 0.0,
                    "window_minutes": s.window_minutes or 60,
                    "wallet_count": s.wallet_count,
                    "cluster_adjusted_wallet_count": s.cluster_adjusted_wallet_count
                    or 0,
                    "unique_core_wallets": s.unique_core_wallets or 0,
                    "weighted_wallet_score": s.weighted_wallet_score or 0.0,
                    "wallets": s.wallets or [],
                    "outcome": s.outcome,
                    "avg_entry_price": s.avg_entry_price,
                    "total_size": s.total_size,
                    "avg_wallet_rank": s.avg_wallet_rank,
                    "net_notional": s.net_notional,
                    "conflicting_notional": s.conflicting_notional,
                    "market_liquidity": s.market_liquidity,
                    "market_volume_24h": s.market_volume_24h,
                    "is_active": s.is_active,
                    "first_seen_at": s.first_seen_at.isoformat()
                    if s.first_seen_at
                    else None,
                    "last_seen_at": s.last_seen_at.isoformat() if s.last_seen_at else None,
                    "detected_at": s.detected_at.isoformat() if s.detected_at else None,
                }
                for s in signals
            ]

    async def expire_old_signals(self):
        """Mark signals inactive when not reinforced within decay window."""
        cutoff = utcnow() - timedelta(minutes=self.SIGNAL_DECAY_MINUTES)
        async with AsyncSessionLocal() as session:
            await session.execute(
                update(MarketConfluenceSignal)
                .where(
                    MarketConfluenceSignal.is_active == True,  # noqa: E712
                    func.coalesce(
                        MarketConfluenceSignal.last_seen_at,
                        MarketConfluenceSignal.detected_at,
                    )
                    < cutoff,
                )
                .values(is_active=False, expired_at=utcnow())
            )
            await session.commit()


# ============================================================================
#  SUBSYSTEM 2: Entity Clustering (Priority 5)
# ============================================================================


class EntityClusterer:
    """Group wallets that likely belong to the same entity."""

    TIMING_CORRELATION_THRESHOLD = 0.7  # Similarity threshold for timing
    MIN_SHARED_MARKETS = 5  # Minimum shared markets to consider a pair
    SIMILAR_TRADE_TIME_WINDOW = 300  # 5 minutes in seconds
    MAX_WALLETS_TO_COMPARE = 200  # Limit pairwise comparisons

    async def run_clustering(self):
        """
        Main clustering pipeline:
        1. Get all DiscoveredWallets with >= 20 trades
        2. For each pair, compute similarity scores
        3. Group wallets exceeding thresholds into clusters
        4. Store/update WalletCluster records
        5. Update cluster_id on DiscoveredWallet records
        """
        logger.info("Starting entity clustering...")

        # Step 1: Get wallets with enough trades
        wallets = await self._get_candidate_wallets()
        if len(wallets) < 2:
            logger.info("Not enough wallets for clustering", count=len(wallets))
            return

        wallet_count = len(wallets)
        logger.info("Clustering candidates loaded", count=wallet_count)

        # Step 2: Fetch trade data for each wallet
        wallet_data: dict[str, dict] = {}
        semaphore = asyncio.Semaphore(5)

        async def fetch_wallet_data(wallet: dict):
            async with semaphore:
                try:
                    trades = await polymarket_client.get_wallet_trades(
                        wallet["address"], limit=200
                    )
                    markets = set()
                    trade_timestamps = []
                    for t in trades:
                        mid = t.get("market", t.get("condition_id", ""))
                        if mid:
                            markets.add(mid)
                        ts = (
                            t.get("timestamp")
                            or t.get("created_at")
                            or t.get("match_time")
                        )
                        if ts:
                            trade_timestamps.append(self._parse_timestamp(ts))

                    # Filter out None timestamps
                    trade_timestamps = [t for t in trade_timestamps if t is not None]

                    wallet_data[wallet["address"]] = {
                        "address": wallet["address"],
                        "markets": markets,
                        "trade_timestamps": sorted(trade_timestamps),
                        "win_rate": wallet.get("win_rate", 0.0),
                        "avg_roi": wallet.get("avg_roi", 0.0),
                        "strategies": wallet.get("strategies_detected", []),
                        "total_pnl": wallet.get("total_pnl", 0.0),
                        "total_trades": wallet.get("total_trades", 0),
                    }
                except Exception as e:
                    logger.debug(
                        "Failed to fetch data for clustering",
                        wallet=wallet["address"],
                        error=str(e),
                    )

        await asyncio.gather(*[fetch_wallet_data(w) for w in wallets])

        if len(wallet_data) < 2:
            logger.info("Not enough wallet data for clustering")
            return

        # Step 3: Compare all pairs (CPU-bound O(n^2) pairwise comparison).
        # Run in thread pool so the event loop stays free for API requests.
        def _pairwise_cluster(clusterer, w_data):
            addresses = list(w_data.keys())
            parent = {a: a for a in addresses}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

            p_compared = 0
            p_linked = 0

            for i in range(len(addresses)):
                for j in range(i + 1, len(addresses)):
                    a_addr = addresses[i]
                    b_addr = addresses[j]
                    a_d = w_data[a_addr]
                    b_d = w_data[b_addr]

                    shared_count = len(a_d["markets"] & b_d["markets"])
                    if shared_count < clusterer.MIN_SHARED_MARKETS:
                        continue

                    p_compared += 1

                    market_overlap = clusterer._calculate_market_overlap(
                        a_d["markets"], b_d["markets"]
                    )
                    timing_sim = clusterer._calculate_timing_similarity(
                        a_d["trade_timestamps"], b_d["trade_timestamps"]
                    )
                    pattern_sim = clusterer._calculate_pattern_similarity(a_d, b_d)

                    combined = (
                        0.4 * market_overlap + 0.35 * timing_sim + 0.25 * pattern_sim
                    )

                    if combined >= 0.5:
                        union(a_addr, b_addr)
                        p_linked += 1

            clusters = {}
            for addr in addresses:
                root = find(addr)
                if root not in clusters:
                    clusters[root] = []
                clusters[root].append(addr)

            multi = {k: v for k, v in clusters.items() if len(v) >= 2}
            return multi, p_compared, p_linked

        multi_clusters, pairs_compared, pairs_linked = await asyncio.to_thread(
            _pairwise_cluster, self, wallet_data
        )

        logger.info(
            "Clustering complete",
            pairs_compared=pairs_compared,
            pairs_linked=pairs_linked,
            clusters_found=len(multi_clusters),
        )

        # Step 5: Store clusters in DB
        await self._store_clusters(multi_clusters, wallet_data)

    async def _get_candidate_wallets(self) -> list[dict]:
        """Get wallets with sufficient trade history for clustering."""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(DiscoveredWallet)
                .where(DiscoveredWallet.total_trades >= 20)
                .order_by(DiscoveredWallet.rank_score.desc())
                .limit(self.MAX_WALLETS_TO_COMPARE)
            )
            wallets = list(result.scalars().all())
            return [
                {
                    "address": w.address,
                    "win_rate": w.win_rate or 0.0,
                    "avg_roi": w.avg_roi or 0.0,
                    "total_pnl": w.total_pnl or 0.0,
                    "total_trades": w.total_trades or 0,
                    "strategies_detected": w.strategies_detected or [],
                }
                for w in wallets
            ]

    def _parse_timestamp(self, ts) -> Optional[float]:
        """Parse a timestamp value into a Unix epoch float."""
        if ts is None:
            return None
        try:
            if isinstance(ts, (int, float)):
                return float(ts)
            if isinstance(ts, str):
                if ts.replace(".", "", 1).replace("-", "", 1).isdigit():
                    return float(ts)
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                return dt.timestamp()
        except (ValueError, TypeError, OSError):
            pass
        return None

    def _calculate_market_overlap(self, markets_a: set, markets_b: set) -> float:
        """Jaccard similarity: |A intersection B| / |A union B|."""
        if not markets_a and not markets_b:
            return 0.0
        union = markets_a | markets_b
        if not union:
            return 0.0
        intersection = markets_a & markets_b
        return len(intersection) / len(union)

    def _calculate_timing_similarity(
        self, trades_a: list[float], trades_b: list[float]
    ) -> float:
        """
        How often do two wallets trade within the same time window?
        Count trades from A that have a matching trade from B within
        SIMILAR_TRADE_TIME_WINDOW. Return ratio of matched / total.
        """
        if not trades_a or not trades_b:
            return 0.0

        matched = 0
        b_idx = 0
        window = self.SIMILAR_TRADE_TIME_WINDOW

        for a_ts in trades_a:
            # Advance b_idx to the first trade within the window
            while b_idx < len(trades_b) and trades_b[b_idx] < a_ts - window:
                b_idx += 1

            # Check if any trade in B is within the window
            check_idx = b_idx
            while check_idx < len(trades_b) and trades_b[check_idx] <= a_ts + window:
                if abs(trades_b[check_idx] - a_ts) <= window:
                    matched += 1
                    break
                check_idx += 1

        total = len(trades_a)
        return matched / total if total > 0 else 0.0

    def _calculate_pattern_similarity(self, wallet_a: dict, wallet_b: dict) -> float:
        """
        Compare trading patterns:
        - Similar win rates (within 10%)
        - Similar avg ROI (within 20%)
        - Overlapping strategies
        Returns 0-1 similarity score.
        """
        score = 0.0
        components = 0

        # Win rate similarity
        wr_a = wallet_a.get("win_rate", 0.0) or 0.0
        wr_b = wallet_b.get("win_rate", 0.0) or 0.0
        wr_diff = abs(wr_a - wr_b)
        if wr_diff <= 0.1:
            score += 1.0
        elif wr_diff <= 0.2:
            score += 0.5
        components += 1

        # ROI similarity
        roi_a = wallet_a.get("avg_roi", 0.0) or 0.0
        roi_b = wallet_b.get("avg_roi", 0.0) or 0.0
        max_roi = max(abs(roi_a), abs(roi_b), 1.0)
        roi_diff_pct = abs(roi_a - roi_b) / max_roi
        if roi_diff_pct <= 0.2:
            score += 1.0
        elif roi_diff_pct <= 0.4:
            score += 0.5
        components += 1

        # Strategy overlap
        strats_a = set(wallet_a.get("strategies", []) or [])
        strats_b = set(wallet_b.get("strategies", []) or [])
        if strats_a and strats_b:
            strat_union = strats_a | strats_b
            strat_inter = strats_a & strats_b
            if strat_union:
                score += len(strat_inter) / len(strat_union)
        elif not strats_a and not strats_b:
            # Both have no strategies detected -- neutral
            score += 0.5
        components += 1

        return score / components if components > 0 else 0.0

    async def _store_clusters(
        self,
        clusters: dict[str, list[str]],
        wallet_data: dict[str, dict],
    ):
        """Persist clusters to the database and update wallet cluster_id fields."""
        async with AsyncSessionLocal() as session:
            for root_addr, member_addrs in clusters.items():
                # Aggregate stats
                total_pnl = sum(
                    wallet_data.get(a, {}).get("total_pnl", 0.0) for a in member_addrs
                )
                total_trades = sum(
                    wallet_data.get(a, {}).get("total_trades", 0) for a in member_addrs
                )
                win_rates = [
                    wallet_data.get(a, {}).get("win_rate", 0.0)
                    for a in member_addrs
                    if wallet_data.get(a, {}).get("win_rate") is not None
                ]
                avg_wr = sum(win_rates) / len(win_rates) if win_rates else 0.0

                cluster_id = str(uuid.uuid4())

                cluster = WalletCluster(
                    id=cluster_id,
                    label=f"Cluster ({len(member_addrs)} wallets)",
                    confidence=0.6,
                    total_wallets=len(member_addrs),
                    combined_pnl=total_pnl,
                    combined_trades=total_trades,
                    avg_win_rate=avg_wr,
                    detection_method="timing_correlation+pattern_match",
                    evidence={
                        "members": member_addrs,
                        "root": root_addr,
                    },
                    created_at=utcnow(),
                    updated_at=utcnow(),
                )
                session.add(cluster)

                # Update wallet records with cluster_id
                for addr in member_addrs:
                    await session.execute(
                        update(DiscoveredWallet)
                        .where(DiscoveredWallet.address == addr)
                        .values(cluster_id=cluster_id)
                    )

            await session.commit()
        logger.info(
            "Clusters stored in DB",
            cluster_count=len(clusters),
        )

    async def get_clusters(self, min_wallets: int = 2) -> list[dict]:
        """Get all clusters with their member wallets."""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(WalletCluster)
                .where(WalletCluster.total_wallets >= min_wallets)
                .order_by(WalletCluster.combined_pnl.desc())
            )
            clusters = list(result.scalars().all())

            output = []
            for c in clusters:
                # Fetch member wallets
                members_result = await session.execute(
                    select(DiscoveredWallet).where(DiscoveredWallet.cluster_id == c.id)
                )
                members = list(members_result.scalars().all())

                output.append(
                    {
                        "id": c.id,
                        "label": c.label,
                        "confidence": c.confidence,
                        "total_wallets": c.total_wallets,
                        "combined_pnl": c.combined_pnl,
                        "combined_trades": c.combined_trades,
                        "avg_win_rate": c.avg_win_rate,
                        "detection_method": c.detection_method,
                        "evidence": c.evidence,
                        "created_at": c.created_at.isoformat()
                        if c.created_at
                        else None,
                        "wallets": [
                            {
                                "address": m.address,
                                "username": m.username,
                                "total_pnl": m.total_pnl,
                                "win_rate": m.win_rate,
                                "total_trades": m.total_trades,
                                "rank_score": m.rank_score,
                            }
                            for m in members
                        ],
                    }
                )
            return output

    async def get_cluster_detail(self, cluster_id: str) -> dict:
        """Get detailed info about a specific cluster."""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(WalletCluster).where(WalletCluster.id == cluster_id)
            )
            cluster = result.scalars().first()

            if not cluster:
                return {}

            members_result = await session.execute(
                select(DiscoveredWallet).where(
                    DiscoveredWallet.cluster_id == cluster_id
                )
            )
            members = list(members_result.scalars().all())

            return {
                "id": cluster.id,
                "label": cluster.label,
                "confidence": cluster.confidence,
                "total_wallets": cluster.total_wallets,
                "combined_pnl": cluster.combined_pnl,
                "combined_trades": cluster.combined_trades,
                "avg_win_rate": cluster.avg_win_rate,
                "detection_method": cluster.detection_method,
                "evidence": cluster.evidence,
                "created_at": cluster.created_at.isoformat()
                if cluster.created_at
                else None,
                "updated_at": cluster.updated_at.isoformat()
                if cluster.updated_at
                else None,
                "wallets": [
                    {
                        "address": m.address,
                        "username": m.username,
                        "total_pnl": m.total_pnl,
                        "win_rate": m.win_rate,
                        "total_trades": m.total_trades,
                        "rank_score": m.rank_score,
                        "avg_roi": m.avg_roi,
                        "strategies_detected": m.strategies_detected or [],
                        "tags": m.tags or [],
                        "anomaly_score": m.anomaly_score,
                    }
                    for m in members
                ],
            }


# ============================================================================
#  SUBSYSTEM 3: Wallet Tagging (Priority 6)
# ============================================================================


class WalletTagger:
    """Auto-classify wallets with behavioral and performance tags."""

    TAG_DEFINITIONS = [
        {
            "name": "smart_predictor",
            "display_name": "Smart Predictor",
            "description": "Consistently profitable with 100+ trades and >60% win rate",
            "category": "performance",
            "color": "#10B981",
        },
        {
            "name": "arb_specialist",
            "display_name": "Arb Specialist",
            "description": "Primarily uses arbitrage strategies",
            "category": "strategy",
            "color": "#6366F1",
        },
        {
            "name": "whale",
            "display_name": "Whale",
            "description": "Large average position sizes (>$5000)",
            "category": "behavioral",
            "color": "#3B82F6",
        },
        {
            "name": "bot",
            "display_name": "Bot",
            "description": "Automated trading pattern detected",
            "category": "behavioral",
            "color": "#8B5CF6",
        },
        {
            "name": "human",
            "display_name": "Human",
            "description": "Human-like trading patterns",
            "category": "behavioral",
            "color": "#F59E0B",
        },
        {
            "name": "consistent",
            "display_name": "Consistent",
            "description": "Low return variance, steady performance",
            "category": "performance",
            "color": "#14B8A6",
        },
        {
            "name": "high_risk",
            "display_name": "High Risk",
            "description": "High drawdown or anomaly score",
            "category": "risk",
            "color": "#EF4444",
        },
        {
            "name": "fading",
            "display_name": "Fading",
            "description": "Was profitable but declining in recent windows",
            "category": "performance",
            "color": "#F97316",
        },
        {
            "name": "insider_suspect",
            "display_name": "Insider Suspect",
            "description": "Suspicious pre-event accuracy",
            "category": "risk",
            "color": "#DC2626",
        },
        {
            "name": "new_talent",
            "display_name": "New Talent",
            "description": "New wallet (<30 days) showing strong results",
            "category": "performance",
            "color": "#22D3EE",
        },
    ]

    async def initialize_tags(self):
        """Ensure all tag definitions exist in the DB."""
        async with AsyncSessionLocal() as session:
            for tag_def in self.TAG_DEFINITIONS:
                result = await session.execute(
                    select(WalletTag).where(WalletTag.name == tag_def["name"])
                )
                existing = result.scalars().first()
                if not existing:
                    tag = WalletTag(
                        id=str(uuid.uuid4()),
                        name=tag_def["name"],
                        display_name=tag_def["display_name"],
                        description=tag_def["description"],
                        category=tag_def["category"],
                        color=tag_def["color"],
                        criteria=tag_def,
                        created_at=utcnow(),
                    )
                    session.add(tag)
                else:
                    # Update fields in case definitions changed
                    existing.display_name = tag_def["display_name"]
                    existing.description = tag_def["description"]
                    existing.category = tag_def["category"]
                    existing.color = tag_def["color"]
                    existing.criteria = tag_def

            await session.commit()
        logger.info(
            "Tag definitions initialized",
            tag_count=len(self.TAG_DEFINITIONS),
        )

    async def auto_tag_wallet(self, wallet: DiscoveredWallet) -> list[str]:
        """
        Evaluate a wallet against all tag criteria and return matching tag names.

        Rules:
        - smart_predictor: total_trades >= 100, win_rate >= 0.6, total_pnl > 0, anomaly_score < 0.5
        - arb_specialist: "basic_arbitrage" or "negrisk_date_sweep" in strategies_detected
        - whale: avg_position_size > 5000
        - bot: trades_per_day > 20 or is_bot flag
        - human: trades_per_day < 10 and not is_bot
        - consistent: roi_std < 15 and sharpe_ratio > 1.0 (if available)
        - high_risk: max_drawdown > 0.3 or anomaly_score > 0.6
        - fading: rolling_pnl["30d"] < 0 when total_pnl > 0
        - insider_suspect: insider_score >= 0.72 and insider_confidence >= 0.60
        - new_talent: days_active < 30, total_trades >= 20, win_rate > 0.6, total_pnl > 0
        """
        tags = []
        total_trades = wallet.total_trades or 0
        win_rate = wallet.win_rate or 0.0
        total_pnl = wallet.total_pnl or 0.0
        anomaly_score = wallet.anomaly_score or 0.0
        strategies = wallet.strategies_detected or []
        avg_position_size = wallet.avg_position_size or 0.0
        trades_per_day = wallet.trades_per_day or 0.0
        is_bot = wallet.is_bot or False
        roi_std = wallet.roi_std or 0.0
        sharpe_ratio = wallet.sharpe_ratio
        max_drawdown = wallet.max_drawdown
        rolling_pnl = wallet.rolling_pnl or {}
        days_active = wallet.days_active or 0
        insider_score = wallet.insider_score or 0.0
        insider_confidence = wallet.insider_confidence or 0.0
        insider_sample = wallet.insider_sample_size or 0

        # smart_predictor
        if (
            total_trades >= 100
            and win_rate >= 0.6
            and total_pnl > 0
            and anomaly_score < 0.5
        ):
            tags.append("smart_predictor")

        # arb_specialist
        arb_strategies = {"basic_arbitrage", "negrisk_date_sweep"}
        if arb_strategies & set(strategies):
            tags.append("arb_specialist")

        # whale
        if avg_position_size > 5000:
            tags.append("whale")

        # bot
        if trades_per_day > 20 or is_bot:
            tags.append("bot")

        # human (mutually exclusive with bot in practice)
        if trades_per_day < 10 and not is_bot:
            tags.append("human")

        # consistent
        if roi_std < 15 and sharpe_ratio is not None and sharpe_ratio > 1.0:
            tags.append("consistent")

        # high_risk
        if (max_drawdown is not None and max_drawdown > 0.3) or anomaly_score > 0.6:
            tags.append("high_risk")

        # fading
        rolling_30d = rolling_pnl.get("30d")
        if rolling_30d is not None and rolling_30d < 0 and total_pnl > 0:
            tags.append("fading")

        # insider_suspect
        if insider_score >= 0.72 and insider_confidence >= 0.60 and insider_sample >= 25:
            tags.append("insider_suspect")

        # new_talent
        if days_active < 30 and total_trades >= 20 and win_rate > 0.6 and total_pnl > 0:
            tags.append("new_talent")

        return tags

    async def tag_all_wallets(self):
        """Run auto-tagging on all discovered wallets."""
        logger.info("Starting wallet auto-tagging...")

        async with AsyncSessionLocal() as session:
            result = await session.execute(select(DiscoveredWallet))
            wallets = list(result.scalars().all())

        tagged_count = 0
        for wallet in wallets:
            try:
                tags = await self.auto_tag_wallet(wallet)
                if tags != (wallet.tags or []):
                    async with AsyncSessionLocal() as session:
                        await session.execute(
                            update(DiscoveredWallet)
                            .where(DiscoveredWallet.address == wallet.address)
                            .values(tags=tags)
                        )
                        await session.commit()
                    tagged_count += 1
            except Exception as e:
                logger.debug(
                    "Failed to tag wallet",
                    wallet=wallet.address,
                    error=str(e),
                )

        logger.info(
            "Wallet auto-tagging complete",
            total_wallets=len(wallets),
            wallets_updated=tagged_count,
        )

    async def get_wallets_by_tag(self, tag_name: str, limit: int = 100) -> list[dict]:
        """Get wallets with a specific tag.

        Since tags are stored as a JSON list in SQLite, we query all wallets
        and filter in Python. For larger datasets, consider a join table.
        """
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(DiscoveredWallet).order_by(DiscoveredWallet.rank_score.desc())
            )
            wallets = list(result.scalars().all())

        matches = []
        for w in wallets:
            wallet_tags = w.tags or []
            if tag_name in wallet_tags:
                matches.append(
                    {
                        "address": w.address,
                        "username": w.username,
                        "total_pnl": w.total_pnl,
                        "win_rate": w.win_rate,
                        "total_trades": w.total_trades,
                        "rank_score": w.rank_score,
                        "tags": wallet_tags,
                        "anomaly_score": w.anomaly_score,
                        "recommendation": w.recommendation,
                    }
                )
                if len(matches) >= limit:
                    break

        return matches

    async def get_all_tags(self) -> list[dict]:
        """Get all tag definitions with wallet counts. Uses one aggregated query for counts."""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(WalletTag).order_by(WalletTag.category, WalletTag.name)
            )
            tags = list(result.scalars().all())

        # Ensure tag definitions exist (e.g. first request before initialize_tags ran)
        if not tags:
            await self.initialize_tags()
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(WalletTag).order_by(WalletTag.category, WalletTag.name)
                )
                tags = list(result.scalars().all())

        async with AsyncSessionLocal() as session:
            # Count wallets per tag via SQL (json_each) instead of loading all wallets
            tag_counts: dict[str, int] = {}
            try:
                count_result = await session.execute(
                    text(
                        "SELECT value AS tag_name, COUNT(DISTINCT address) AS cnt "
                        "FROM discovered_wallets, json_each(discovered_wallets.tags) "
                        "WHERE discovered_wallets.tags IS NOT NULL "
                        "GROUP BY value"
                    )
                )
                for row in count_result:
                    tag_counts[str(row.tag_name)] = row.cnt
            except Exception:
                # Fallback if json_each not supported or column type differs
                pass

        return [
            {
                "id": t.id,
                "name": t.name,
                "display_name": t.display_name,
                "description": t.description,
                "category": t.category,
                "color": t.color,
                "wallet_count": tag_counts.get(t.name, 0),
            }
            for t in tags
        ]


# ============================================================================
#  SUBSYSTEM 4: Cross-Platform Tracking (Priority 7)
# ============================================================================


class CrossPlatformTracker:
    """Track traders across Polymarket and Kalshi."""

    async def scan_cross_platform(self):
        """
        Compare trading activity across platforms:
        1. Get active markets that exist on both Polymarket and Kalshi
        2. For matched markets, compare position holders
        3. Look for wallets/accounts that trade the same events on both platforms
        4. Identify cross-platform arbitrageurs
        5. Store in CrossPlatformEntity
        """
        logger.info("Starting cross-platform scan...")

        try:
            # Use the existing cross-platform strategy's Kalshi cache
            from services.strategies.cross_platform import (
                _KalshiMarketCache,
                _tokenize,
                _jaccard_similarity,
                _MATCH_THRESHOLD,
            )
            from config import settings

            kalshi_cache = _KalshiMarketCache(
                api_url=settings.KALSHI_API_URL, ttl_seconds=120
            )
            # Kalshi cache uses synchronous HTTP — run in thread pool
            kalshi_markets = await asyncio.to_thread(kalshi_cache.get_markets)

            if not kalshi_markets:
                logger.info("No Kalshi markets available, skipping cross-platform scan")
                return

            # Get Polymarket markets
            try:
                poly_markets = await polymarket_client.get_all_markets(active=True)
            except Exception as e:
                logger.warning("Failed to fetch Polymarket markets", error=str(e))
                return

            if not poly_markets:
                logger.info("No Polymarket markets available")
                return

            # CPU-bound market matching: run in thread pool
            def _match_markets(poly_mkts, kalshi_mkts, threshold):
                kalshi_token_idx = {}
                for km in kalshi_mkts:
                    kalshi_token_idx[km.id] = _tokenize(km.question)

                pairs = []
                for pm in poly_mkts:
                    if pm.closed or not pm.active:
                        continue
                    pm_tokens = _tokenize(pm.question)
                    if not pm_tokens:
                        continue

                    best_score = 0.0
                    best_km = None
                    for km in kalshi_mkts:
                        km_tokens = kalshi_token_idx.get(km.id)
                        if not km_tokens:
                            continue
                        score = _jaccard_similarity(pm_tokens, km_tokens)
                        if score > best_score:
                            best_score = score
                            best_km = km

                    if best_km and best_score >= threshold:
                        pairs.append(
                            {
                                "polymarket_id": pm.id,
                                "polymarket_question": pm.question,
                                "kalshi_id": best_km.id,
                                "kalshi_question": best_km.question,
                                "similarity": best_score,
                                "pm_yes_price": pm.yes_price,
                                "pm_no_price": pm.no_price,
                                "k_yes_price": best_km.yes_price,
                                "k_no_price": best_km.no_price,
                            }
                        )
                return pairs

            matched_pairs = await asyncio.to_thread(
                _match_markets, poly_markets, kalshi_markets, _MATCH_THRESHOLD
            )

            logger.info(
                "Cross-platform market matches found",
                matched_pairs=len(matched_pairs),
            )

            if not matched_pairs:
                return

            # For each matched pair, look for wallets active on both sides
            # Get top wallets from DiscoveredWallet that trade these markets
            active_wallets = await self._get_active_wallets_for_markets(
                [p["polymarket_id"] for p in matched_pairs]
            )

            # Build cross-platform entity records for wallets that trade
            # matched markets
            entities_found = 0
            for wallet_addr, traded_markets in active_wallets.items():
                cross_markets = []
                for pair in matched_pairs:
                    if pair["polymarket_id"] in traded_markets:
                        price_diff = abs(pair["pm_yes_price"] - pair["k_yes_price"])
                        is_arb = price_diff > 0.05  # >5 cents difference
                        cross_markets.append(
                            {
                                "polymarket_id": pair["polymarket_id"],
                                "kalshi_id": pair["kalshi_id"],
                                "question": pair["polymarket_question"],
                                "price_diff": price_diff,
                                "potential_arb": is_arb,
                            }
                        )

                if cross_markets:
                    arb_detected = any(m["potential_arb"] for m in cross_markets)
                    await self._upsert_cross_platform_entity(
                        polymarket_address=wallet_addr,
                        matching_markets=cross_markets,
                        cross_platform_arb=arb_detected,
                    )
                    entities_found += 1

            logger.info(
                "Cross-platform scan complete",
                entities_found=entities_found,
                matched_markets=len(matched_pairs),
            )

        except ImportError:
            logger.warning(
                "Cross-platform strategy module not available, skipping scan"
            )
        except Exception as e:
            logger.error("Cross-platform scan failed", error=str(e))

    async def _get_active_wallets_for_markets(
        self, market_ids: list[str]
    ) -> dict[str, set[str]]:
        """Get wallets that are active in the specified markets.

        Returns a dict of wallet_address -> set of market_ids they trade.
        """
        wallet_markets: dict[str, set[str]] = {}

        # Get top discovered wallets
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(DiscoveredWallet)
                .where(DiscoveredWallet.is_profitable == True)  # noqa: E712
                .order_by(DiscoveredWallet.rank_score.desc())
                .limit(100)
            )
            wallets = list(result.scalars().all())

        market_id_set = set(market_ids)
        semaphore = asyncio.Semaphore(5)

        async def check_wallet(wallet_addr: str):
            async with semaphore:
                try:
                    positions = await polymarket_client.get_wallet_positions(
                        wallet_addr
                    )
                    traded = set()
                    for pos in positions:
                        mid = (
                            pos.get("market", "")
                            or pos.get("condition_id", "")
                            or pos.get("asset", "")
                        )
                        if mid in market_id_set:
                            traded.add(mid)
                    if traded:
                        wallet_markets[wallet_addr] = traded
                except Exception:
                    pass

        await asyncio.gather(*[check_wallet(w.address) for w in wallets])
        return wallet_markets

    async def _upsert_cross_platform_entity(
        self,
        polymarket_address: str,
        matching_markets: list[dict],
        cross_platform_arb: bool = False,
    ):
        """Create or update a cross-platform entity record."""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(CrossPlatformEntity).where(
                    CrossPlatformEntity.polymarket_address == polymarket_address
                )
            )
            existing = result.scalars().first()

            if existing:
                existing.matching_markets = matching_markets
                existing.cross_platform_arb = cross_platform_arb
                existing.updated_at = utcnow()
            else:
                # Try to get PnL from DiscoveredWallet
                wallet_result = await session.execute(
                    select(DiscoveredWallet).where(
                        DiscoveredWallet.address == polymarket_address
                    )
                )
                wallet = wallet_result.scalars().first()
                poly_pnl = wallet.total_pnl if wallet else 0.0

                entity = CrossPlatformEntity(
                    id=str(uuid.uuid4()),
                    label=polymarket_address[:10] + "...",
                    polymarket_address=polymarket_address,
                    polymarket_pnl=poly_pnl,
                    combined_pnl=poly_pnl,
                    cross_platform_arb=cross_platform_arb,
                    matching_markets=matching_markets,
                    confidence=0.5,
                    created_at=utcnow(),
                    updated_at=utcnow(),
                )
                session.add(entity)

            await session.commit()

    async def get_cross_platform_entities(self, limit: int = 50) -> list[dict]:
        """Get entities tracked across platforms."""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(CrossPlatformEntity)
                .order_by(CrossPlatformEntity.combined_pnl.desc())
                .limit(limit)
            )
            entities = list(result.scalars().all())

            return [
                {
                    "id": e.id,
                    "label": e.label,
                    "polymarket_address": e.polymarket_address,
                    "kalshi_username": e.kalshi_username,
                    "polymarket_pnl": e.polymarket_pnl,
                    "kalshi_pnl": e.kalshi_pnl,
                    "combined_pnl": e.combined_pnl,
                    "cross_platform_arb": e.cross_platform_arb,
                    "hedging_detected": e.hedging_detected,
                    "matching_markets": e.matching_markets or [],
                    "confidence": e.confidence,
                    "created_at": e.created_at.isoformat() if e.created_at else None,
                    "updated_at": e.updated_at.isoformat() if e.updated_at else None,
                }
                for e in entities
            ]

    async def get_cross_platform_arb_activity(self) -> list[dict]:
        """Get recent cross-platform arbitrage activity."""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(CrossPlatformEntity)
                .where(CrossPlatformEntity.cross_platform_arb == True)  # noqa: E712
                .order_by(CrossPlatformEntity.updated_at.desc())
                .limit(50)
            )
            entities = list(result.scalars().all())

            activity = []
            for e in entities:
                arb_markets = [
                    m
                    for m in (e.matching_markets or [])
                    if m.get("potential_arb", False)
                ]
                activity.append(
                    {
                        "entity_id": e.id,
                        "polymarket_address": e.polymarket_address,
                        "kalshi_username": e.kalshi_username,
                        "arb_market_count": len(arb_markets),
                        "arb_markets": arb_markets,
                        "combined_pnl": e.combined_pnl,
                        "updated_at": e.updated_at.isoformat()
                        if e.updated_at
                        else None,
                    }
                )
            return activity


# ============================================================================
#  MAIN ORCHESTRATOR
# ============================================================================


class WalletIntelligence:
    """Orchestrates all intelligence subsystems."""

    def __init__(self):
        self.confluence = ConfluenceDetector()
        self.clusterer = EntityClusterer()
        self.tagger = WalletTagger()
        self.cross_platform = CrossPlatformTracker()
        self._running = False

    async def initialize(self):
        """Initialize all subsystems."""
        await self.tagger.initialize_tags()
        logger.info("Wallet intelligence initialized")

    async def run_full_analysis(self):
        """Run all intelligence subsystems."""
        logger.info("Running full intelligence analysis...")

        # Phase 1: Confluence detection (fast, positions-only scan)
        try:
            await self.confluence.scan_for_confluence()
        except Exception as e:
            logger.error("Confluence scan failed", error=str(e))

        # Phase 2: Auto-tag wallets (reads DB, light computation)
        try:
            await self.tagger.tag_all_wallets()
        except Exception as e:
            logger.error("Wallet tagging failed", error=str(e))

        # Phase 3: Entity clustering (heavier, pairwise comparisons)
        try:
            await self.clusterer.run_clustering()
        except Exception as e:
            logger.error("Entity clustering failed", error=str(e))

        # Phase 4: Cross-platform tracking (lowest priority, depends on Kalshi API)
        try:
            await self.cross_platform.scan_cross_platform()
        except Exception as e:
            logger.error("Cross-platform scan failed", error=str(e))

        logger.info("Intelligence analysis complete")

    async def start_background(self, interval_minutes: int = 30):
        """Run intelligence analysis on a schedule."""
        self._running = True
        logger.info(
            "Starting background intelligence loop",
            interval_minutes=interval_minutes,
        )
        while self._running:
            if not global_pause_state.is_paused:
                try:
                    await self.run_full_analysis()
                except Exception as e:
                    logger.error("Intelligence analysis failed", error=str(e))
            await asyncio.sleep(interval_minutes * 60)

    def stop(self):
        """Stop the background intelligence loop."""
        self._running = False
        logger.info("Wallet intelligence background loop stopped")


# Singleton
wallet_intelligence = WalletIntelligence()
