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
from typing import Optional

from sqlalchemy import select, update

from models.database import (
    DiscoveredWallet,
    WalletTag,
    WalletCluster,
    MarketConfluenceSignal,
    CrossPlatformEntity,
    AsyncSessionLocal,
)
from services.polymarket import polymarket_client
from utils.logger import get_logger

logger = get_logger("wallet_intelligence")


# ============================================================================
#  SUBSYSTEM 1: Multi-Wallet Confluence Detection (Priority 4)
# ============================================================================


class ConfluenceDetector:
    """Detect when multiple profitable wallets are buying/selling the same market."""

    MIN_WALLETS_FOR_SIGNAL = 3  # Minimum wallets to trigger a signal
    MIN_WALLET_RANK_SCORE = 0.3  # Only consider wallets above this rank
    SIGNAL_DECAY_HOURS = 48  # Signals expire after this many hours

    async def scan_for_confluence(self) -> list[dict]:
        """
        Main confluence detection pipeline:
        1. Get all profitable DiscoveredWallets with rank_score > MIN_WALLET_RANK_SCORE
        2. For each wallet, fetch their current positions
        3. Build a dict: market_id -> [list of wallet positions]
        4. For markets with >= MIN_WALLETS_FOR_SIGNAL wallets:
           - Calculate signal strength
           - Determine signal type
           - Create/update MarketConfluenceSignal in DB
        5. Expire old signals
        6. Return active signals
        """
        logger.info("Starting confluence scan...")

        # Step 1: Get high-ranking wallets
        wallets = await self._get_qualifying_wallets()
        if not wallets:
            logger.info("No qualifying wallets found for confluence scan")
            return []

        logger.info(
            "Qualifying wallets for confluence scan",
            count=len(wallets),
        )

        # Step 2 & 3: Fetch positions and build market map
        market_positions: dict[str, list[dict]] = {}
        semaphore = asyncio.Semaphore(5)

        async def fetch_positions(wallet: dict):
            async with semaphore:
                try:
                    positions = await polymarket_client.get_wallet_positions(
                        wallet["address"]
                    )
                    for pos in positions:
                        market_id = (
                            pos.get("market", "")
                            or pos.get("condition_id", "")
                            or pos.get("asset", "")
                        )
                        if not market_id:
                            continue

                        size = float(pos.get("size", 0) or 0)
                        if size <= 0:
                            continue

                        entry = {
                            "wallet_address": wallet["address"],
                            "rank_score": wallet["rank_score"],
                            "market_id": market_id,
                            "size": size,
                            "side": (
                                pos.get("side", "") or pos.get("outcome", "")
                            ).upper(),
                            "price": float(
                                pos.get("avgPrice", 0) or pos.get("price", 0) or 0
                            ),
                            "title": pos.get("title", ""),
                        }
                        if market_id not in market_positions:
                            market_positions[market_id] = []
                        market_positions[market_id].append(entry)
                except Exception as e:
                    logger.debug(
                        "Failed to fetch positions for wallet",
                        wallet=wallet["address"],
                        error=str(e),
                    )

        await asyncio.gather(*[fetch_positions(w) for w in wallets])

        # Step 4: Evaluate each market for confluence
        signals_created = 0
        for market_id, positions in market_positions.items():
            # Only unique wallets count
            unique_wallets = {p["wallet_address"] for p in positions}
            if len(unique_wallets) < self.MIN_WALLETS_FOR_SIGNAL:
                continue

            wallet_count = len(unique_wallets)
            avg_rank = sum(p["rank_score"] for p in positions) / len(positions)
            total_size = sum(p["size"] for p in positions)

            strength = self._calculate_signal_strength(
                wallet_count, avg_rank, total_size
            )

            # Determine signal type from dominant side
            buy_count = sum(1 for p in positions if p["side"] in ("BUY", "YES"))
            sell_count = sum(1 for p in positions if p["side"] in ("SELL", "NO"))

            if buy_count > sell_count * 2:
                signal_type = "multi_wallet_buy"
            elif sell_count > buy_count * 2:
                signal_type = "multi_wallet_sell"
            else:
                signal_type = "accumulation"

            # Derive a representative outcome and avg entry price
            outcome = "YES" if buy_count >= sell_count else "NO"
            prices = [p["price"] for p in positions if p["price"] > 0]
            avg_entry_price = sum(prices) / len(prices) if prices else None

            market_title = ""
            for p in positions:
                if p.get("title"):
                    market_title = p["title"]
                    break

            wallet_list = list(unique_wallets)

            await self._upsert_signal(
                market_id=market_id,
                market_question=market_title,
                signal_type=signal_type,
                strength=strength,
                wallet_count=wallet_count,
                wallets=wallet_list,
                outcome=outcome,
                avg_entry_price=avg_entry_price,
                total_size=total_size,
                avg_wallet_rank=avg_rank,
            )
            signals_created += 1

        # Step 5: Expire old signals
        await self.expire_old_signals()

        # Step 6: Return active signals
        active = await self.get_active_signals()
        logger.info(
            "Confluence scan complete",
            signals_created=signals_created,
            active_signals=len(active),
        )
        return active

    async def _get_qualifying_wallets(self) -> list[dict]:
        """Get wallets with rank_score above the minimum threshold."""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(DiscoveredWallet).where(
                    DiscoveredWallet.rank_score >= self.MIN_WALLET_RANK_SCORE,
                    DiscoveredWallet.is_profitable == True,  # noqa: E712
                )
            )
            wallets = list(result.scalars().all())
            return [
                {
                    "address": w.address,
                    "rank_score": w.rank_score or 0.0,
                    "total_pnl": w.total_pnl or 0.0,
                }
                for w in wallets
            ]

    def _calculate_signal_strength(
        self, wallet_count: int, avg_rank_score: float, total_size: float
    ) -> float:
        """
        Signal strength 0-1 based on:
        - Number of wallets (more = stronger, logarithmic scaling)
        - Average rank score of participating wallets (higher = stronger)
        - Total position size (larger = stronger, logarithmic)

        Formula: 0.4 * wallet_factor + 0.4 * rank_factor + 0.2 * size_factor
        """
        # Wallet factor: log scale, 3 wallets ~= 0.5, 10 ~= 0.83, 30 ~= 1.0
        wallet_factor = min(math.log(max(wallet_count, 1)) / math.log(30), 1.0)

        # Rank factor: direct mapping (already 0-1)
        rank_factor = min(max(avg_rank_score, 0.0), 1.0)

        # Size factor: log scale on dollar amount, $1000 ~= 0.5, $100k ~= 1.0
        if total_size > 0:
            size_factor = min(math.log10(max(total_size, 1)) / 5.0, 1.0)
        else:
            size_factor = 0.0

        strength = 0.4 * wallet_factor + 0.4 * rank_factor + 0.2 * size_factor
        return round(min(max(strength, 0.0), 1.0), 4)

    async def _upsert_signal(
        self,
        market_id: str,
        market_question: str,
        signal_type: str,
        strength: float,
        wallet_count: int,
        wallets: list[str],
        outcome: Optional[str],
        avg_entry_price: Optional[float],
        total_size: Optional[float],
        avg_wallet_rank: Optional[float],
    ):
        """Create or update a confluence signal in the database."""
        async with AsyncSessionLocal() as session:
            # Check for existing active signal on this market
            result = await session.execute(
                select(MarketConfluenceSignal).where(
                    MarketConfluenceSignal.market_id == market_id,
                    MarketConfluenceSignal.is_active == True,  # noqa: E712
                )
            )
            existing = result.scalars().first()

            if existing:
                existing.signal_type = signal_type
                existing.strength = strength
                existing.wallet_count = wallet_count
                existing.wallets = wallets
                existing.outcome = outcome
                existing.avg_entry_price = avg_entry_price
                existing.total_size = total_size
                existing.avg_wallet_rank = avg_wallet_rank
                existing.detected_at = datetime.utcnow()
            else:
                signal = MarketConfluenceSignal(
                    id=str(uuid.uuid4()),
                    market_id=market_id,
                    market_question=market_question,
                    signal_type=signal_type,
                    strength=strength,
                    wallet_count=wallet_count,
                    wallets=wallets,
                    outcome=outcome,
                    avg_entry_price=avg_entry_price,
                    total_size=total_size,
                    avg_wallet_rank=avg_wallet_rank,
                    is_active=True,
                    detected_at=datetime.utcnow(),
                )
                session.add(signal)

            await session.commit()

    async def get_active_signals(
        self, min_strength: float = 0.0, limit: int = 50
    ) -> list[dict]:
        """Get currently active confluence signals from DB, sorted by strength."""
        async with AsyncSessionLocal() as session:
            query = (
                select(MarketConfluenceSignal)
                .where(
                    MarketConfluenceSignal.is_active == True,  # noqa: E712
                    MarketConfluenceSignal.strength >= min_strength,
                )
                .order_by(MarketConfluenceSignal.strength.desc())
                .limit(limit)
            )
            result = await session.execute(query)
            signals = list(result.scalars().all())

            return [
                {
                    "id": s.id,
                    "market_id": s.market_id,
                    "market_question": s.market_question or "",
                    "signal_type": s.signal_type,
                    "strength": s.strength,
                    "wallet_count": s.wallet_count,
                    "wallets": s.wallets or [],
                    "outcome": s.outcome,
                    "avg_entry_price": s.avg_entry_price,
                    "total_size": s.total_size,
                    "avg_wallet_rank": s.avg_wallet_rank,
                    "is_active": s.is_active,
                    "detected_at": s.detected_at.isoformat() if s.detected_at else None,
                }
                for s in signals
            ]

    async def expire_old_signals(self):
        """Mark signals as inactive if older than SIGNAL_DECAY_HOURS."""
        cutoff = datetime.utcnow() - timedelta(hours=self.SIGNAL_DECAY_HOURS)
        async with AsyncSessionLocal() as session:
            await session.execute(
                update(MarketConfluenceSignal)
                .where(
                    MarketConfluenceSignal.is_active == True,  # noqa: E712
                    MarketConfluenceSignal.detected_at < cutoff,
                )
                .values(is_active=False, expired_at=datetime.utcnow())
            )
            await session.commit()
        logger.debug("Expired old confluence signals", cutoff=cutoff.isoformat())


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

        # Step 3: Compare all pairs
        addresses = list(wallet_data.keys())
        # Union-find for grouping
        parent: dict[str, str] = {a: a for a in addresses}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        pairs_compared = 0
        pairs_linked = 0

        for i in range(len(addresses)):
            for j in range(i + 1, len(addresses)):
                a_addr = addresses[i]
                b_addr = addresses[j]
                a_data = wallet_data[a_addr]
                b_data = wallet_data[b_addr]

                # Quick filter: must share enough markets
                market_overlap = self._calculate_market_overlap(
                    a_data["markets"], b_data["markets"]
                )
                shared_count = len(a_data["markets"] & b_data["markets"])
                if shared_count < self.MIN_SHARED_MARKETS:
                    continue

                pairs_compared += 1

                timing_sim = self._calculate_timing_similarity(
                    a_data["trade_timestamps"], b_data["trade_timestamps"]
                )
                pattern_sim = self._calculate_pattern_similarity(a_data, b_data)

                # Combined score: market overlap + timing + pattern
                combined = 0.4 * market_overlap + 0.35 * timing_sim + 0.25 * pattern_sim

                if combined >= 0.5:
                    union(a_addr, b_addr)
                    pairs_linked += 1

        # Step 4: Build clusters from union-find
        clusters: dict[str, list[str]] = {}
        for addr in addresses:
            root = find(addr)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(addr)

        # Only keep clusters with 2+ members
        multi_clusters = {k: v for k, v in clusters.items() if len(v) >= 2}

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
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
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
                        created_at=datetime.utcnow(),
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
        - insider_suspect: anomaly_score > 0.7 and "insider_pattern" in strategies
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
        if anomaly_score > 0.7 and "insider_pattern" in strategies:
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
        """Get all tag definitions with wallet counts."""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(WalletTag).order_by(WalletTag.category, WalletTag.name)
            )
            tags = list(result.scalars().all())

        # Count wallets per tag
        all_wallets_result = None
        async with AsyncSessionLocal() as session:
            all_wallets_result = await session.execute(select(DiscoveredWallet))
            all_wallets = list(all_wallets_result.scalars().all())

        tag_counts: dict[str, int] = {}
        for w in all_wallets:
            for t in w.tags or []:
                tag_counts[t] = tag_counts.get(t, 0) + 1

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
            kalshi_markets = kalshi_cache.get_markets()

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

            # Build token index for Kalshi
            kalshi_token_index: dict[str, set[str]] = {}
            kalshi_by_id: dict[str, object] = {}
            for km in kalshi_markets:
                kalshi_token_index[km.id] = _tokenize(km.question)
                kalshi_by_id[km.id] = km

            # Match Polymarket markets to Kalshi markets
            matched_pairs: list[dict] = []
            for pm in poly_markets:
                if pm.closed or not pm.active:
                    continue
                pm_tokens = _tokenize(pm.question)
                if not pm_tokens:
                    continue

                best_score = 0.0
                best_kalshi = None
                for km in kalshi_markets:
                    km_tokens = kalshi_token_index.get(km.id)
                    if not km_tokens:
                        continue
                    score = _jaccard_similarity(pm_tokens, km_tokens)
                    if score > best_score:
                        best_score = score
                        best_kalshi = km

                if best_kalshi and best_score >= _MATCH_THRESHOLD:
                    matched_pairs.append(
                        {
                            "polymarket_id": pm.id,
                            "polymarket_question": pm.question,
                            "kalshi_id": best_kalshi.id,
                            "kalshi_question": best_kalshi.question,
                            "similarity": best_score,
                            "pm_yes_price": pm.yes_price,
                            "pm_no_price": pm.no_price,
                            "k_yes_price": best_kalshi.yes_price,
                            "k_no_price": best_kalshi.no_price,
                        }
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
                existing.updated_at = datetime.utcnow()
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
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
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
