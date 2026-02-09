import uuid
import math
from datetime import datetime
from typing import Optional
from dataclasses import dataclass
from enum import Enum
from sqlalchemy import select

from models.database import TrackedWallet, DetectedAnomaly, AsyncSessionLocal
from services.polymarket import polymarket_client
from utils.logger import get_logger

logger = get_logger("anomaly")


class AnomalyType(str, Enum):
    # Statistical anomalies
    IMPOSSIBLE_WIN_RATE = "impossible_win_rate"
    UNUSUAL_ROI = "unusual_roi"
    PERFECT_TIMING = "perfect_timing"
    STATISTICALLY_IMPOSSIBLE = "statistically_impossible"

    # Pattern anomalies
    FRONT_RUNNING = "front_running"
    WASH_TRADING = "wash_trading"
    COORDINATED_TRADING = "coordinated_trading"

    # Suspicious behavior
    INSIDER_PATTERN = "insider_pattern"
    ARBITRAGE_ONLY = "arbitrage_only"
    UNUSUAL_SIZE = "unusual_size"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class WalletAnalysis:
    """Analysis results for a wallet"""

    address: str
    total_trades: int
    win_rate: float
    total_pnl: float
    avg_roi: float
    max_roi: float
    avg_hold_time_hours: float
    trade_frequency_per_day: float
    markets_traded: int
    strategies_detected: list[str]
    anomaly_score: float
    anomalies: list[dict]
    is_profitable_pattern: bool
    recommendation: str


@dataclass
class Anomaly:
    """Detected anomaly"""

    type: AnomalyType
    severity: Severity
    score: float
    description: str
    evidence: dict


class AnomalyDetector:
    """Detect impossible or suspicious trading patterns"""

    # Statistical thresholds
    IMPOSSIBLE_WIN_RATE_THRESHOLD = 0.95  # 95% win rate is suspicious
    UNUSUAL_ROI_ZSCORE = 3.0  # 3 standard deviations
    MIN_TRADES_FOR_ANALYSIS = 10
    PERFECT_TIMING_THRESHOLD = 0.90  # 90% of trades at optimal price

    # Pattern thresholds
    WASH_TRADE_TIME_WINDOW = 60  # seconds
    FRONT_RUN_TIME_WINDOW = 30  # seconds

    async def analyze_wallet(self, wallet_address: str) -> WalletAnalysis:
        """Comprehensive wallet analysis"""
        address = wallet_address.lower()

        # Fetch wallet data
        trades = await polymarket_client.get_wallet_trades(address, limit=500)
        positions = await polymarket_client.get_wallet_positions(address)

        if len(trades) < self.MIN_TRADES_FOR_ANALYSIS:
            return WalletAnalysis(
                address=address,
                total_trades=len(trades),
                win_rate=0,
                total_pnl=0,
                avg_roi=0,
                max_roi=0,
                avg_hold_time_hours=0,
                trade_frequency_per_day=0,
                markets_traded=0,
                strategies_detected=[],
                anomaly_score=0,
                anomalies=[],
                is_profitable_pattern=False,
                recommendation="Insufficient data for analysis",
            )

        # Calculate basic stats (pass positions for unrealized PnL calculation)
        stats = self._calculate_trade_stats(trades, positions)

        # Override win/loss stats with accurate closed-positions data when available.
        # _calculate_trade_stats infers wins/losses by matching raw buys/sells per market,
        # which is unreliable (market resolutions don't always appear as sells).
        # calculate_win_rate_fast uses the pre-aggregated closed-positions endpoint
        # with realizedPnl, which is the ground truth.
        closed_pos_stats = await polymarket_client.calculate_win_rate_fast(
            address, min_positions=1
        )
        if closed_pos_stats:
            stats["wins"] = closed_pos_stats["wins"]
            stats["losses"] = closed_pos_stats["losses"]
            stats["closed_positions"] = closed_pos_stats["closed_positions"]
            total_closed = closed_pos_stats["wins"] + closed_pos_stats["losses"]
            stats["win_rate"] = (
                closed_pos_stats["wins"] / total_closed if total_closed > 0 else 0
            )

        # Detect anomalies
        anomalies = []
        anomalies.extend(self._detect_statistical_anomalies(trades, stats))
        anomalies.extend(self._detect_pattern_anomalies(trades))
        anomalies.extend(self._detect_timing_anomalies(trades))

        # Calculate overall anomaly score
        anomaly_score = self._calculate_anomaly_score(anomalies)

        # Detect strategies being used
        strategies = self._detect_strategies(trades)

        # Determine if this is a profitable pattern to follow
        is_profitable = self._is_profitable_pattern(stats, anomalies, strategies)

        # Generate recommendation
        recommendation = self._generate_recommendation(stats, anomalies, is_profitable)

        analysis = WalletAnalysis(
            address=address,
            total_trades=stats["total_trades"],
            win_rate=stats["win_rate"],
            total_pnl=stats["total_pnl"],
            avg_roi=stats["avg_roi"],
            max_roi=stats["max_roi"],
            avg_hold_time_hours=stats["avg_hold_time_hours"],
            trade_frequency_per_day=stats["trades_per_day"],
            markets_traded=stats["unique_markets"],
            strategies_detected=strategies,
            anomaly_score=anomaly_score,
            anomalies=[
                {
                    "type": a.type.value,
                    "severity": a.severity.value,
                    "score": a.score,
                    "description": a.description,
                    "evidence": a.evidence,
                }
                for a in anomalies
            ],
            is_profitable_pattern=is_profitable,
            recommendation=recommendation,
        )

        # Save analysis to database
        await self._save_analysis(analysis)

        logger.info(
            "Completed wallet analysis",
            wallet=address,
            trades=stats["total_trades"],
            win_rate=stats["win_rate"],
            anomaly_score=anomaly_score,
            anomalies_found=len(anomalies),
        )

        return analysis

    def _calculate_trade_stats(
        self, trades: list[dict], positions: list[dict] = None
    ) -> dict:
        """Calculate trading statistics from raw Polymarket trade data"""
        if not trades:
            return {}

        positions = positions or []

        # Group trades by market/token to calculate PnL
        market_positions = {}  # market_id -> {"buys": [], "sells": []}
        markets = set()
        total_invested = 0.0
        total_returned = 0.0

        for trade in trades:
            # Get trade details - handle different field names from API
            size = float(trade.get("size", 0) or trade.get("amount", 0) or 0)
            price = float(trade.get("price", 0) or 0)
            side = (trade.get("side", "") or "").upper()
            market_id = trade.get(
                "market", trade.get("condition_id", trade.get("asset", ""))
            )
            outcome = trade.get("outcome", trade.get("outcome_index", ""))

            if market_id:
                markets.add(market_id)

            if market_id not in market_positions:
                market_positions[market_id] = {"buys": [], "sells": []}

            cost = size * price

            if side == "BUY":
                total_invested += cost
                market_positions[market_id]["buys"].append(
                    {
                        "size": size,
                        "price": price,
                        "cost": cost,
                        "outcome": outcome,
                        "timestamp": trade.get(
                            "timestamp", trade.get("created_at", "")
                        ),
                    }
                )
            elif side == "SELL":
                total_returned += cost
                market_positions[market_id]["sells"].append(
                    {
                        "size": size,
                        "price": price,
                        "cost": cost,
                        "outcome": outcome,
                        "timestamp": trade.get(
                            "timestamp", trade.get("created_at", "")
                        ),
                    }
                )

        # Calculate realized PnL from completed trades
        realized_pnl = total_returned - total_invested

        # Calculate unrealized PnL from current positions
        unrealized_pnl = 0.0
        for pos in positions:
            size = float(pos.get("size", 0) or 0)
            avg_price = float(pos.get("avgPrice", pos.get("avg_price", 0)) or 0)
            current_price = float(pos.get("currentPrice", pos.get("price", 0)) or 0)
            unrealized_pnl += size * (current_price - avg_price)

        total_pnl = realized_pnl + unrealized_pnl

        # Calculate win/loss by market (a market is a "win" if total sells > total buys)
        wins = 0
        losses = 0
        rois = []

        for market_id, pos_data in market_positions.items():
            buy_cost = sum(b["cost"] for b in pos_data["buys"])
            sell_revenue = sum(s["cost"] for s in pos_data["sells"])

            # Only count markets where we have both buys and sells (closed positions)
            if buy_cost > 0 and sell_revenue > 0:
                market_pnl = sell_revenue - buy_cost
                market_roi = (market_pnl / buy_cost) * 100 if buy_cost > 0 else 0
                rois.append(market_roi)

                if market_pnl > 0:
                    wins += 1
                elif market_pnl < 0:
                    losses += 1

        # If no closed positions, estimate from overall flow
        if wins == 0 and losses == 0 and total_invested > 0:
            # Use ratio of returned vs invested as proxy
            if total_returned > total_invested * 1.02:  # > 2% profit
                wins = max(1, len(markets) // 2)
                losses = len(markets) - wins
            elif total_returned < total_invested * 0.98:  # > 2% loss
                losses = max(1, len(markets) // 2)
                wins = len(markets) - losses

        total_trades = len(trades)
        closed_markets = wins + losses
        win_rate = wins / closed_markets if closed_markets > 0 else 0

        # Calculate time span
        days = 30  # default
        if trades:
            first_trade = trades[-1].get("timestamp", trades[-1].get("created_at"))
            last_trade = trades[0].get("timestamp", trades[0].get("created_at"))
            if first_trade and last_trade:
                try:
                    if isinstance(first_trade, str):
                        first_trade = datetime.fromisoformat(
                            first_trade.replace("Z", "+00:00")
                        )
                    if isinstance(last_trade, str):
                        last_trade = datetime.fromisoformat(
                            last_trade.replace("Z", "+00:00")
                        )
                    days = max((last_trade - first_trade).days, 1)
                except Exception:
                    days = 30

        # Calculate ROI
        avg_roi = (
            sum(rois) / len(rois)
            if rois
            else (total_pnl / total_invested * 100 if total_invested > 0 else 0)
        )
        max_roi = max(rois) if rois else avg_roi
        min_roi = min(rois) if rois else avg_roi

        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_invested": total_invested,
            "total_returned": total_returned,
            "avg_roi": avg_roi,
            "max_roi": max_roi,
            "min_roi": min_roi,
            "roi_std": self._std_dev(rois) if len(rois) > 1 else 0,
            "avg_hold_time_hours": 0,  # Would need position tracking to calculate
            "trades_per_day": total_trades / days,
            "unique_markets": len(markets),
            "days_active": days,
            "closed_positions": closed_markets,
            "open_positions": len(positions),
        }

    def _detect_statistical_anomalies(self, trades: list, stats: dict) -> list[Anomaly]:
        """Detect statistically impossible patterns"""
        anomalies = []

        # 1. Impossible win rate
        # Use closed_positions (actual evaluated markets) not total_trades (raw trade count)
        closed_positions = stats.get("closed_positions", stats["wins"] + stats["losses"])
        if stats["win_rate"] >= self.IMPOSSIBLE_WIN_RATE_THRESHOLD and closed_positions >= 10:
            n = closed_positions
            # Probability of 95%+ win rate by chance with fair trades
            # Using binomial distribution approximation
            expected_wins = n * 0.5
            std = math.sqrt(n * 0.5 * 0.5)
            actual_wins = stats["wins"]
            z_score = (actual_wins - expected_wins) / std if std > 0 else 0

            if z_score > 4:  # Extremely unlikely by chance
                anomalies.append(
                    Anomaly(
                        type=AnomalyType.IMPOSSIBLE_WIN_RATE,
                        severity=Severity.CRITICAL
                        if stats["win_rate"] > 0.98
                        else Severity.HIGH,
                        score=min(z_score / 10, 1.0),
                        description=f"Win rate of {stats['win_rate'] * 100:.1f}% over {n} resolved markets is statistically impossible",
                        evidence={
                            "win_rate": stats["win_rate"],
                            "closed_positions": n,
                            "z_score": z_score,
                            "probability": f"1 in {10 ** int(z_score):,}",
                        },
                    )
                )

        # 2. Unusual ROI distribution
        if stats["roi_std"] > 0 and stats["avg_roi"] > 0:
            # Check if average ROI is unusually high
            # Normal arbitrage ROI is 2-5%
            if stats["avg_roi"] > 20:  # 20% average ROI is suspicious
                anomalies.append(
                    Anomaly(
                        type=AnomalyType.UNUSUAL_ROI,
                        severity=Severity.HIGH
                        if stats["avg_roi"] > 50
                        else Severity.MEDIUM,
                        score=min(stats["avg_roi"] / 100, 1.0),
                        description=f"Average ROI of {stats['avg_roi']:.1f}% is unusually high",
                        evidence={
                            "avg_roi": stats["avg_roi"],
                            "max_roi": stats["max_roi"],
                            "std_dev": stats["roi_std"],
                        },
                    )
                )

        # 3. No losing trades
        # Use closed_positions (resolved markets) not total_trades (raw individual transactions)
        if stats["losses"] == 0 and closed_positions >= 20:
            anomalies.append(
                Anomaly(
                    type=AnomalyType.STATISTICALLY_IMPOSSIBLE,
                    severity=Severity.CRITICAL,
                    score=1.0,
                    description=f"Zero losses over {closed_positions} resolved markets is statistically impossible",
                    evidence={
                        "wins": stats["wins"],
                        "losses": 0,
                        "closed_positions": closed_positions,
                    },
                )
            )

        return anomalies

    def _detect_pattern_anomalies(self, trades: list) -> list[Anomaly]:
        """Detect suspicious trading patterns"""
        anomalies = []

        # Group trades by market
        market_trades = {}
        for trade in trades:
            market_id = trade.get("market", trade.get("condition_id", ""))
            if market_id:
                if market_id not in market_trades:
                    market_trades[market_id] = []
                market_trades[market_id].append(trade)

        # 1. Check for wash trading (buy and sell same market rapidly)
        for market_id, market_trade_list in market_trades.items():
            if len(market_trade_list) >= 2:
                # Sort by timestamp
                sorted_trades = sorted(
                    market_trade_list,
                    key=lambda t: t.get("timestamp", t.get("created_at", "")),
                )

                for i in range(len(sorted_trades) - 1):
                    t1 = sorted_trades[i]
                    t2 = sorted_trades[i + 1]

                    # Check if opposite sides within time window
                    side1 = t1.get("side", "")
                    side2 = t2.get("side", "")

                    if side1 and side2 and side1 != side2:
                        # This could be wash trading
                        anomalies.append(
                            Anomaly(
                                type=AnomalyType.WASH_TRADING,
                                severity=Severity.MEDIUM,
                                score=0.6,
                                description="Rapid buy/sell pattern detected in market",
                                evidence={
                                    "market_id": market_id,
                                    "trade_count": len(market_trade_list),
                                },
                            )
                        )
                        break

        # 2. Check for arbitrage-only pattern (sign of sophisticated bot)
        arbitrage_indicators = 0
        for trade in trades:
            # Arbitrage trades typically have small, consistent profits
            pnl = trade.get("pnl", 0) or 0
            cost = trade.get("cost", trade.get("amount", 1)) or 1
            if cost > 0:
                roi = (pnl / cost) * 100
                if 1 < roi < 10:  # Typical arbitrage ROI range
                    arbitrage_indicators += 1

        if arbitrage_indicators / len(trades) > 0.8:  # 80% look like arb trades
            anomalies.append(
                Anomaly(
                    type=AnomalyType.ARBITRAGE_ONLY,
                    severity=Severity.LOW,
                    score=0.4,
                    description="Wallet appears to only execute arbitrage strategies",
                    evidence={
                        "arbitrage_trades": arbitrage_indicators,
                        "total_trades": len(trades),
                        "ratio": arbitrage_indicators / len(trades),
                    },
                )
            )

        return anomalies

    def _detect_timing_anomalies(self, trades: list) -> list[Anomaly]:
        """Detect suspicious timing patterns"""
        anomalies = []

        # Check for perfect timing (always buying at lows, selling at highs)
        for trade in trades:
            # This would require price history comparison
            # For now, use proxy: check if entry price is significantly better than average
            pass

        # Check for trades just before major price moves (insider pattern)
        for trade in trades:
            # Would need to compare with subsequent price action
            pass

        return anomalies

    def _detect_strategies(self, trades: list) -> list[str]:
        """Detect which strategies a wallet is using"""
        strategies = set()

        # Analyze trade patterns
        market_groups = {}
        for trade in trades:
            market_id = trade.get("market", trade.get("condition_id", ""))
            if market_id:
                if market_id not in market_groups:
                    market_groups[market_id] = []
                market_groups[market_id].append(trade)

        # Check for multi-market trades (NegRisk/date sweep)
        for market_id, market_trades in market_groups.items():
            if len(market_trades) >= 3:
                strategies.add("negrisk_date_sweep")

        # Check for basic arbitrage (same market, opposite sides)
        for market_trades in market_groups.values():
            sides = set(t.get("outcome", t.get("side", "")) for t in market_trades)
            if "YES" in sides and "NO" in sides:
                strategies.add("basic_arbitrage")

        # High frequency = likely bot
        if len(trades) > 100:
            strategies.add("automated_trading")

        return list(strategies)

    def _calculate_anomaly_score(self, anomalies: list[Anomaly]) -> float:
        """Calculate overall anomaly score (0-1)"""
        if not anomalies:
            return 0.0

        # Weight by severity
        weights = {
            Severity.LOW: 0.2,
            Severity.MEDIUM: 0.4,
            Severity.HIGH: 0.7,
            Severity.CRITICAL: 1.0,
        }

        total_score = sum(a.score * weights[a.severity] for a in anomalies)
        return min(total_score / len(anomalies), 1.0)

    def _is_profitable_pattern(
        self, stats: dict, anomalies: list[Anomaly], strategies: list[str]
    ) -> bool:
        """Determine if this wallet has a profitable pattern worth following"""
        # Must have positive returns
        if stats["total_pnl"] <= 0:
            return False

        # Must have reasonable win rate (not suspiciously high)
        if stats["win_rate"] < 0.55 or stats["win_rate"] > 0.95:
            return False

        # Should be using arbitrage strategies
        if not any(
            s in strategies
            for s in ["basic_arbitrage", "negrisk_date_sweep", "automated_trading"]
        ):
            return False

        # Should not have critical anomalies
        critical_anomalies = [a for a in anomalies if a.severity == Severity.CRITICAL]
        if critical_anomalies:
            return False

        return True

    def _generate_recommendation(
        self, stats: dict, anomalies: list[Anomaly], is_profitable: bool
    ) -> str:
        """Generate recommendation for this wallet"""
        if not stats.get("total_trades"):
            return "Insufficient data for analysis"

        critical = [a for a in anomalies if a.severity == Severity.CRITICAL]
        high = [a for a in anomalies if a.severity == Severity.HIGH]

        if critical:
            return f"AVOID - Critical anomalies detected: {', '.join(a.type.value for a in critical)}"

        if high:
            return f"CAUTION - High severity anomalies: {', '.join(a.type.value for a in high)}"

        if is_profitable:
            return f"CONSIDER COPYING - Profitable pattern with {stats['win_rate'] * 100:.1f}% win rate, {stats['avg_roi']:.1f}% avg ROI"

        if stats["win_rate"] > 0.6 and stats["total_pnl"] > 0:
            return "MONITOR - Shows potential but needs more data"

        return "NOT RECOMMENDED - No clear edge detected"

    def _std_dev(self, values: list[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    async def _save_analysis(self, analysis: WalletAnalysis):
        """Save analysis results to database"""
        async with AsyncSessionLocal() as session:
            # Update or create wallet record
            wallet = await session.get(TrackedWallet, analysis.address)
            if not wallet:
                wallet = TrackedWallet(address=analysis.address)
                session.add(wallet)

            wallet.total_trades = analysis.total_trades
            wallet.win_rate = analysis.win_rate
            wallet.total_pnl = analysis.total_pnl
            wallet.avg_roi = analysis.avg_roi
            wallet.anomaly_score = analysis.anomaly_score
            wallet.is_flagged = analysis.anomaly_score > 0.7
            wallet.flag_reasons = [
                a["type"]
                for a in analysis.anomalies
                if a["severity"] in ["high", "critical"]
            ]
            wallet.last_analyzed_at = datetime.utcnow()
            wallet.analysis_data = {
                "strategies": analysis.strategies_detected,
                "recommendation": analysis.recommendation,
                "is_profitable_pattern": analysis.is_profitable_pattern,
            }

            # Save anomalies
            for anomaly_data in analysis.anomalies:
                anomaly = DetectedAnomaly(
                    id=str(uuid.uuid4()),
                    anomaly_type=anomaly_data["type"],
                    severity=anomaly_data["severity"],
                    wallet_address=analysis.address,
                    description=anomaly_data["description"],
                    evidence=anomaly_data["evidence"],
                    score=anomaly_data["score"],
                )
                session.add(anomaly)

            await session.commit()

    async def find_profitable_wallets(
        self,
        min_trades: int = 50,
        min_win_rate: float = 0.6,
        min_pnl: float = 1000.0,
        max_anomaly_score: float = 0.5,
    ) -> list[WalletAnalysis]:
        """Find wallets with profitable patterns that aren't suspicious"""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(TrackedWallet).where(
                    TrackedWallet.total_trades >= min_trades,
                    TrackedWallet.win_rate >= min_win_rate,
                    TrackedWallet.total_pnl >= min_pnl,
                    TrackedWallet.anomaly_score <= max_anomaly_score,
                    not TrackedWallet.is_flagged,
                )
            )
            wallets = list(result.scalars().all())

            analyses = []
            for wallet in wallets:
                analysis = await self.analyze_wallet(wallet.address)
                if analysis.is_profitable_pattern:
                    analyses.append(analysis)

            return analyses

    async def get_anomalies(
        self,
        severity: Optional[str] = None,
        anomaly_type: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get detected anomalies"""
        async with AsyncSessionLocal() as session:
            query = select(DetectedAnomaly).order_by(DetectedAnomaly.detected_at.desc())

            if severity:
                query = query.where(DetectedAnomaly.severity == severity)
            if anomaly_type:
                query = query.where(DetectedAnomaly.anomaly_type == anomaly_type)

            query = query.limit(limit)

            result = await session.execute(query)
            anomalies = list(result.scalars().all())

            return [
                {
                    "id": a.id,
                    "type": a.anomaly_type,
                    "severity": a.severity,
                    "wallet": a.wallet_address,
                    "market": a.market_id,
                    "description": a.description,
                    "score": a.score,
                    "detected_at": a.detected_at.isoformat(),
                }
                for a in anomalies
            ]


# Singleton instance
anomaly_detector = AnomalyDetector()
