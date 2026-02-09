"""
Wallet Discovery Engine
=======================

Automated discovery and profiling of ALL active Polymarket wallets.
This is the core engine that powers the trader leaderboard and
copy-trading candidate selection.

Pipeline:
    1. Fetch active markets from Polymarket Gamma API
    2. For each market, fetch recent trades to discover wallet addresses
    3. For each discovered wallet, calculate comprehensive metrics
    4. Compute risk-adjusted scores (Sharpe, Sortino, Drawdown, etc.)
    5. Compute rolling window statistics (1d, 7d, 30d, 90d)
    6. Calculate composite rank scores
    7. Store/update in DiscoveredWallet table
    8. Refresh leaderboard positions
"""

import asyncio
import math
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select, func, update, desc, asc

from models.database import DiscoveredWallet, AsyncSessionLocal
from services.polymarket import polymarket_client
from utils.logger import get_logger

logger = get_logger("wallet_discovery")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Rolling window periods (label -> timedelta)
ROLLING_WINDOWS = {
    "1d": timedelta(days=1),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
    "90d": timedelta(days=90),
}

# Minimum data requirements
MIN_TRADES_FOR_ANALYSIS = 5
MIN_TRADES_FOR_RISK_METRICS = 3

# Rate-limiting delays (seconds)
DELAY_BETWEEN_MARKETS = 0.25
DELAY_BETWEEN_WALLETS = 0.15

# Staleness threshold: re-analyze wallets older than this
STALE_ANALYSIS_HOURS = 12


class WalletDiscoveryEngine:
    """
    Discovers, profiles, and ranks every active Polymarket wallet.

    Combines trade-level data, position snapshots, and closed-position
    records to build a comprehensive profile for each wallet, then ranks
    them using a weighted composite score.
    """

    def __init__(self):
        self.client = polymarket_client
        self._running = False
        self._last_run_at: Optional[datetime] = None
        self._wallets_discovered_last_run: int = 0
        self._wallets_analyzed_last_run: int = 0

    # ------------------------------------------------------------------
    # 1. Trade Statistics (mirrors anomaly_detector pattern)
    # ------------------------------------------------------------------

    def _calculate_trade_stats(
        self, trades: list[dict], positions: list[dict] | None = None
    ) -> dict:
        """
        Calculate comprehensive trading statistics from raw Polymarket
        trade data and open positions.

        Returns a dict with all fields needed to populate DiscoveredWallet.
        """
        positions = positions or []

        if not trades and not positions:
            return self._empty_stats()

        # Group trades by market/condition_id to compute per-market PnL
        market_positions: dict[str, dict] = {}
        markets: set[str] = set()
        total_invested = 0.0
        total_returned = 0.0

        for trade in trades:
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

        # Realized PnL from completed trades
        realized_pnl = total_returned - total_invested

        # Unrealized PnL from open positions
        unrealized_pnl = 0.0
        for pos in positions:
            size = float(pos.get("size", 0) or 0)
            avg_price = float(
                pos.get("avgPrice", pos.get("avg_price", 0)) or 0
            )
            current_price = float(
                pos.get("currentPrice", pos.get("curPrice", pos.get("price", 0))) or 0
            )
            unrealized_pnl += size * (current_price - avg_price)

        total_pnl = realized_pnl + unrealized_pnl

        # Per-market win/loss + ROI series
        wins = 0
        losses = 0
        market_rois: list[float] = []
        market_pnls: list[float] = []

        for _market_id, pos_data in market_positions.items():
            buy_cost = sum(b["cost"] for b in pos_data["buys"])
            sell_revenue = sum(s["cost"] for s in pos_data["sells"])

            if buy_cost > 0 and sell_revenue > 0:
                m_pnl = sell_revenue - buy_cost
                m_roi = (m_pnl / buy_cost) * 100 if buy_cost > 0 else 0.0
                market_rois.append(m_roi)
                market_pnls.append(m_pnl)

                if m_pnl > 0:
                    wins += 1
                elif m_pnl < 0:
                    losses += 1

        # Fallback estimation when no closed positions are detectable
        if wins == 0 and losses == 0 and total_invested > 0:
            if total_returned > total_invested * 1.02:
                wins = max(1, len(markets) // 2)
                losses = len(markets) - wins
            elif total_returned < total_invested * 0.98:
                losses = max(1, len(markets) // 2)
                wins = len(markets) - losses

        total_trades = len(trades)
        closed_markets = wins + losses
        win_rate = wins / closed_markets if closed_markets > 0 else 0.0

        # Time span
        days_active = self._compute_days_active(trades)

        # ROI statistics
        avg_roi = (
            sum(market_rois) / len(market_rois)
            if market_rois
            else (total_pnl / total_invested * 100 if total_invested > 0 else 0.0)
        )
        max_roi = max(market_rois) if market_rois else avg_roi
        min_roi = min(market_rois) if market_rois else avg_roi
        roi_std = self._std_dev(market_rois) if len(market_rois) > 1 else 0.0

        # Average position size
        avg_position_size = total_invested / total_trades if total_trades > 0 else 0.0

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
            "roi_std": roi_std,
            "unique_markets": len(markets),
            "open_positions": len(positions),
            "days_active": days_active,
            "avg_hold_time_hours": 0.0,  # Requires position-level timestamps
            "trades_per_day": total_trades / max(days_active, 1),
            "avg_position_size": avg_position_size,
            # Raw series for downstream calculations
            "_market_rois": market_rois,
            "_market_pnls": market_pnls,
        }

    def _empty_stats(self) -> dict:
        """Return a zeroed-out stats dict."""
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "total_invested": 0.0,
            "total_returned": 0.0,
            "avg_roi": 0.0,
            "max_roi": 0.0,
            "min_roi": 0.0,
            "roi_std": 0.0,
            "unique_markets": 0,
            "open_positions": 0,
            "days_active": 0,
            "avg_hold_time_hours": 0.0,
            "trades_per_day": 0.0,
            "avg_position_size": 0.0,
            "_market_rois": [],
            "_market_pnls": [],
        }

    # ------------------------------------------------------------------
    # 2. Risk-Adjusted Metrics
    # ------------------------------------------------------------------

    def _calculate_risk_adjusted_metrics(
        self,
        market_rois: list[float],
        market_pnls: list[float],
        days_active: int,
        total_pnl: float = 0.0,
        total_invested: float = 0.0,
    ) -> dict:
        """
        Calculate risk-adjusted performance metrics.

        Returns dict with:
            sharpe_ratio, sortino_ratio, max_drawdown, profit_factor, calmar_ratio
        """
        result: dict = {
            "sharpe_ratio": None,
            "sortino_ratio": None,
            "max_drawdown": None,
            "profit_factor": None,
            "calmar_ratio": None,
        }

        if len(market_rois) < MIN_TRADES_FOR_RISK_METRICS:
            return result

        # ---- Sharpe Ratio ----
        # (mean_return - risk_free_rate) / std_dev_returns
        # risk_free_rate = 0 (crypto convention)
        mean_return = sum(market_rois) / len(market_rois)
        std_return = self._std_dev(market_rois)

        if std_return > 0:
            result["sharpe_ratio"] = mean_return / std_return
        else:
            # Perfect consistency: infinite Sharpe in theory; cap to a large value
            result["sharpe_ratio"] = float("inf") if mean_return > 0 else 0.0

        # ---- Sortino Ratio ----
        # (mean_return - risk_free_rate) / downside_deviation
        negative_returns = [r for r in market_rois if r < 0]
        if negative_returns:
            downside_variance = sum(r ** 2 for r in negative_returns) / len(
                negative_returns
            )
            downside_deviation = math.sqrt(downside_variance)
            if downside_deviation > 0:
                result["sortino_ratio"] = mean_return / downside_deviation
            else:
                result["sortino_ratio"] = float("inf") if mean_return > 0 else 0.0
        else:
            # No negative returns at all
            result["sortino_ratio"] = float("inf") if mean_return > 0 else 0.0

        # ---- Max Drawdown ----
        # Largest peak-to-trough decline in cumulative PnL series.
        # Walk through PnL in chronological order.
        if market_pnls:
            cumulative = 0.0
            peak = 0.0
            max_dd = 0.0

            for pnl in market_pnls:
                cumulative += pnl
                if cumulative > peak:
                    peak = cumulative
                drawdown = peak - cumulative
                if drawdown > max_dd:
                    max_dd = drawdown

            # Express as positive fraction of peak (0.15 = 15%)
            if peak > 0:
                result["max_drawdown"] = max_dd / peak
            elif max_dd > 0:
                # Never had a peak but had losses: use total invested as base
                base = total_invested if total_invested > 0 else abs(sum(market_pnls))
                result["max_drawdown"] = max_dd / base if base > 0 else 0.0
            else:
                result["max_drawdown"] = 0.0

        # ---- Profit Factor ----
        # gross_profit / abs(gross_loss)
        gross_profit = sum(p for p in market_pnls if p > 0)
        gross_loss = sum(p for p in market_pnls if p < 0)

        if gross_loss < 0:
            result["profit_factor"] = gross_profit / abs(gross_loss)
        elif gross_profit > 0:
            result["profit_factor"] = float("inf")
        else:
            result["profit_factor"] = 0.0

        # ---- Calmar Ratio ----
        # annualized_return / max_drawdown
        days = max(days_active, 1)
        if total_invested > 0 and days > 0:
            annualized_return = (total_pnl / total_invested) * (365.0 / days)
            max_dd = result.get("max_drawdown") or 0.0
            if max_dd > 0:
                result["calmar_ratio"] = annualized_return / max_dd
            else:
                result["calmar_ratio"] = None  # Undefined with zero drawdown
        else:
            result["calmar_ratio"] = None

        return result

    # ------------------------------------------------------------------
    # 3. Rolling Time Windows
    # ------------------------------------------------------------------

    def _calculate_rolling_windows(
        self, trades: list[dict], current_time: datetime
    ) -> dict:
        """
        Calculate metrics over rolling time windows.

        Returns dict with keys:
            rolling_pnl, rolling_roi, rolling_win_rate,
            rolling_trade_count, rolling_sharpe

        Each is a dict like {"1d": value, "7d": value, "30d": value, "90d": value}.
        """
        rolling_pnl: dict[str, float] = {}
        rolling_roi: dict[str, float] = {}
        rolling_win_rate: dict[str, float] = {}
        rolling_trade_count: dict[str, int] = {}
        rolling_sharpe: dict[str, float | None] = {}

        for label, delta in ROLLING_WINDOWS.items():
            cutoff = current_time - delta
            window_trades = self._filter_trades_after(trades, cutoff)

            if not window_trades:
                rolling_pnl[label] = 0.0
                rolling_roi[label] = 0.0
                rolling_win_rate[label] = 0.0
                rolling_trade_count[label] = 0
                rolling_sharpe[label] = None
                continue

            stats = self._calculate_trade_stats(window_trades)
            rois = stats.get("_market_rois", [])

            rolling_pnl[label] = stats["total_pnl"]
            rolling_roi[label] = stats["avg_roi"]
            rolling_win_rate[label] = stats["win_rate"]
            rolling_trade_count[label] = stats["total_trades"]

            # Window Sharpe
            if len(rois) >= 2:
                mean_r = sum(rois) / len(rois)
                std_r = self._std_dev(rois)
                rolling_sharpe[label] = (mean_r / std_r) if std_r > 0 else None
            else:
                rolling_sharpe[label] = None

        return {
            "rolling_pnl": rolling_pnl,
            "rolling_roi": rolling_roi,
            "rolling_win_rate": rolling_win_rate,
            "rolling_trade_count": rolling_trade_count,
            "rolling_sharpe": rolling_sharpe,
        }

    # ------------------------------------------------------------------
    # 4. Composite Rank Score
    # ------------------------------------------------------------------

    def _calculate_rank_score(self, metrics: dict) -> float:
        """
        Calculate a composite score for leaderboard ranking.

        Weighted formula:
            30% normalized Sharpe ratio   (clamped 0-5 -> 0-1)
            25% normalized profit factor  (clamped 0-10 -> 0-1)
            20% win rate                  (already 0-1)
            15% normalized total PnL      (log scale, clamped)
            10% consistency               (1 - normalized drawdown)

        Returns a score in [0, 1].
        """
        # -- Sharpe component (30%) --
        sharpe = metrics.get("sharpe_ratio")
        if sharpe is None or not math.isfinite(sharpe):
            sharpe_norm = 0.5  # Default to middle if unavailable
        else:
            sharpe_norm = max(0.0, min(sharpe / 5.0, 1.0))

        # -- Profit factor component (25%) --
        pf = metrics.get("profit_factor")
        if pf is None or not math.isfinite(pf):
            pf_norm = 0.5
        else:
            pf_norm = max(0.0, min(pf / 10.0, 1.0))

        # -- Win rate component (20%) --
        win_rate = metrics.get("win_rate", 0.0)
        wr_norm = max(0.0, min(win_rate, 1.0))

        # -- PnL component (15%, log scale) --
        total_pnl = metrics.get("total_pnl", 0.0)
        if total_pnl > 0:
            # log10 scale: $10 -> 1, $100 -> 2, $1000 -> 3, etc.
            # Normalize to 0-1 by dividing by 5 (covers up to $100k)
            pnl_norm = max(0.0, min(math.log10(total_pnl + 1) / 5.0, 1.0))
        else:
            pnl_norm = 0.0

        # -- Consistency component (10%) --
        max_dd = metrics.get("max_drawdown")
        if max_dd is not None and math.isfinite(max_dd):
            consistency_norm = max(0.0, 1.0 - min(max_dd, 1.0))
        else:
            consistency_norm = 0.5

        rank_score = (
            0.30 * sharpe_norm
            + 0.25 * pf_norm
            + 0.20 * wr_norm
            + 0.15 * pnl_norm
            + 0.10 * consistency_norm
        )

        return max(0.0, min(rank_score, 1.0))

    # ------------------------------------------------------------------
    # 5. Strategy & Classification Helpers
    # ------------------------------------------------------------------

    def _detect_strategies(self, trades: list[dict]) -> list[str]:
        """Detect which trading strategies a wallet is using."""
        strategies: set[str] = set()

        market_groups: dict[str, list[dict]] = {}
        for trade in trades:
            market_id = trade.get(
                "market", trade.get("condition_id", "")
            )
            if market_id:
                market_groups.setdefault(market_id, []).append(trade)

        # Multi-market date sweep (NegRisk)
        for _mid, mtrades in market_groups.items():
            if len(mtrades) >= 3:
                strategies.add("negrisk_date_sweep")

        # Basic arbitrage (same market, opposite outcomes)
        for mtrades in market_groups.values():
            sides = set(
                t.get("outcome", t.get("side", "")) for t in mtrades
            )
            if "Yes" in sides and "No" in sides:
                strategies.add("basic_arbitrage")
            if "YES" in sides and "NO" in sides:
                strategies.add("basic_arbitrage")

        # Automated / high-frequency
        if len(trades) > 100:
            strategies.add("automated_trading")

        # Scalping (many trades per market)
        for mtrades in market_groups.values():
            if len(mtrades) >= 6:
                strategies.add("scalping")
                break

        return list(strategies)

    def _classify_wallet(self, stats: dict, risk_metrics: dict) -> dict:
        """
        Classify a wallet and generate recommendation/tags.

        Returns dict with: anomaly_score, is_bot, is_profitable,
                           recommendation, tags
        """
        tags: list[str] = []
        anomaly_score = 0.0

        win_rate = stats.get("win_rate", 0.0)
        total_pnl = stats.get("total_pnl", 0.0)
        total_trades = stats.get("total_trades", 0)
        trades_per_day = stats.get("trades_per_day", 0.0)
        sharpe = risk_metrics.get("sharpe_ratio")
        profit_factor = risk_metrics.get("profit_factor")

        # -- Bot detection --
        is_bot = False
        if trades_per_day > 20:
            is_bot = True
            tags.append("bot")
        if total_trades > 500:
            is_bot = True
            if "bot" not in tags:
                tags.append("bot")

        # -- Profitability --
        is_profitable = total_pnl > 0 and win_rate > 0.5

        # -- Performance tags --
        if win_rate >= 0.7 and total_trades >= 20:
            tags.append("high_win_rate")
        if total_pnl >= 10000:
            tags.append("whale")
        elif total_pnl >= 1000:
            tags.append("profitable")
        if sharpe is not None and math.isfinite(sharpe) and sharpe >= 2.0:
            tags.append("risk_adjusted_alpha")
        if profit_factor is not None and math.isfinite(profit_factor) and profit_factor >= 3.0:
            tags.append("strong_edge")
        if win_rate >= 0.55 and total_pnl > 0 and total_trades >= 50:
            tags.append("consistent")

        # -- Anomaly score (light version) --
        if win_rate >= 0.95 and total_trades >= 20:
            anomaly_score = max(anomaly_score, 0.8)
            tags.append("suspicious_win_rate")
        if win_rate >= 0.85 and total_trades >= 50:
            anomaly_score = max(anomaly_score, 0.5)

        # -- Recommendation --
        if anomaly_score >= 0.7:
            recommendation = "avoid"
        elif is_profitable and win_rate >= 0.6 and total_trades >= 20:
            recommendation = "copy_candidate"
        elif is_profitable and total_trades >= 10:
            recommendation = "monitor"
        elif total_trades < MIN_TRADES_FOR_ANALYSIS:
            recommendation = "unanalyzed"
        else:
            recommendation = "monitor"

        return {
            "anomaly_score": anomaly_score,
            "is_bot": is_bot,
            "is_profitable": is_profitable,
            "recommendation": recommendation,
            "tags": tags,
        }

    # ------------------------------------------------------------------
    # 6. Full Wallet Analysis
    # ------------------------------------------------------------------

    async def analyze_wallet(self, address: str) -> dict | None:
        """
        Run full analysis pipeline for a single wallet.

        Returns a dict of all fields ready to be stored in DiscoveredWallet,
        or None if there is insufficient data.
        """
        address = address.lower()

        try:
            # Fetch trade history, open positions, and profile in parallel
            trades, positions, profile = await asyncio.gather(
                self.client.get_wallet_trades(address, limit=500),
                self.client.get_wallet_positions(address),
                self.client.get_user_profile(address),
            )
        except Exception as e:
            logger.error(
                "Failed to fetch data for wallet",
                address=address,
                error=str(e),
            )
            return None

        if len(trades) < MIN_TRADES_FOR_ANALYSIS and not positions:
            return None

        # Basic stats
        stats = self._calculate_trade_stats(trades, positions)

        # Risk-adjusted metrics
        risk_metrics = self._calculate_risk_adjusted_metrics(
            market_rois=stats["_market_rois"],
            market_pnls=stats["_market_pnls"],
            days_active=stats["days_active"],
            total_pnl=stats["total_pnl"],
            total_invested=stats["total_invested"],
        )

        # Rolling windows
        now = datetime.utcnow()
        rolling = self._calculate_rolling_windows(trades, now)

        # Strategy detection
        strategies = self._detect_strategies(trades)

        # Classification
        classification = self._classify_wallet(stats, risk_metrics)

        # Composite rank
        rank_input = {
            "sharpe_ratio": risk_metrics["sharpe_ratio"],
            "profit_factor": risk_metrics["profit_factor"],
            "win_rate": stats["win_rate"],
            "total_pnl": stats["total_pnl"],
            "max_drawdown": risk_metrics["max_drawdown"],
        }
        rank_score = self._calculate_rank_score(rank_input)

        username = profile.get("username") if profile else None

        return {
            "address": address,
            "username": username,
            "last_analyzed_at": now,
            "discovery_source": "scan",
            # Basic stats
            "total_trades": stats["total_trades"],
            "wins": stats["wins"],
            "losses": stats["losses"],
            "win_rate": stats["win_rate"],
            "total_pnl": stats["total_pnl"],
            "realized_pnl": stats["realized_pnl"],
            "unrealized_pnl": stats["unrealized_pnl"],
            "total_invested": stats["total_invested"],
            "total_returned": stats["total_returned"],
            "avg_roi": stats["avg_roi"],
            "max_roi": stats["max_roi"],
            "min_roi": stats["min_roi"],
            "roi_std": stats["roi_std"],
            "unique_markets": stats["unique_markets"],
            "open_positions": stats["open_positions"],
            "days_active": stats["days_active"],
            "avg_hold_time_hours": stats["avg_hold_time_hours"],
            "trades_per_day": stats["trades_per_day"],
            "avg_position_size": stats["avg_position_size"],
            # Risk-adjusted
            "sharpe_ratio": risk_metrics["sharpe_ratio"],
            "sortino_ratio": risk_metrics["sortino_ratio"],
            "max_drawdown": risk_metrics["max_drawdown"],
            "profit_factor": risk_metrics["profit_factor"],
            "calmar_ratio": risk_metrics["calmar_ratio"],
            # Rolling windows
            "rolling_pnl": rolling["rolling_pnl"],
            "rolling_roi": rolling["rolling_roi"],
            "rolling_win_rate": rolling["rolling_win_rate"],
            "rolling_trade_count": rolling["rolling_trade_count"],
            "rolling_sharpe": rolling["rolling_sharpe"],
            # Classification
            "anomaly_score": classification["anomaly_score"],
            "is_bot": classification["is_bot"],
            "is_profitable": classification["is_profitable"],
            "recommendation": classification["recommendation"],
            "strategies_detected": strategies,
            "tags": classification["tags"],
            # Ranking
            "rank_score": rank_score,
        }

    # ------------------------------------------------------------------
    # 7. Wallet Discovery from Market Trades
    # ------------------------------------------------------------------

    async def _discover_wallets_from_market(
        self, market: object, max_wallets: int = 50
    ) -> set[str]:
        """
        Fetch recent trades for a single market and extract unique wallet
        addresses. ``market`` is a Market model instance from the Gamma API.
        """
        discovered: set[str] = set()

        try:
            condition_id = getattr(market, "condition_id", None)
            if not condition_id:
                return discovered

            trades = await self.client.get_market_trades(
                condition_id, limit=min(max_wallets * 2, 200)
            )

            for trade in trades:
                # The data API returns 'user' or 'taker' or 'maker' fields
                for field in ("user", "taker", "maker"):
                    addr = trade.get(field, "")
                    if addr and isinstance(addr, str) and len(addr) >= 10:
                        discovered.add(addr.lower())

                if len(discovered) >= max_wallets:
                    break

        except Exception as e:
            logger.warning(
                "Failed to fetch trades for market",
                market=getattr(market, "question", "?")[:60],
                error=str(e),
            )

        return discovered

    async def _discover_wallets_from_leaderboard(
        self, scan_count: int = 200
    ) -> set[str]:
        """
        Supplement market-based discovery with wallets from the
        Polymarket leaderboard API (PNL and VOL sorted).
        """
        discovered: set[str] = set()

        for order_by in ("PNL", "VOL"):
            try:
                entries = await self.client.get_leaderboard_paginated(
                    total_limit=scan_count,
                    order_by=order_by,
                )
                for entry in entries:
                    addr = (entry.get("proxyWallet", "") or "").lower()
                    if addr and len(addr) >= 10:
                        discovered.add(addr)
            except Exception as e:
                logger.warning(
                    "Leaderboard scan failed",
                    order_by=order_by,
                    error=str(e),
                )

        return discovered

    # ------------------------------------------------------------------
    # 8. Database Persistence
    # ------------------------------------------------------------------

    async def _upsert_wallet(self, data: dict):
        """Insert or update a DiscoveredWallet record."""
        address = data["address"]

        async with AsyncSessionLocal() as session:
            wallet = await session.get(DiscoveredWallet, address)

            if wallet is None:
                wallet = DiscoveredWallet(
                    address=address,
                    discovered_at=datetime.utcnow(),
                )
                session.add(wallet)

            # Update all fields from analysis data
            for key, value in data.items():
                if key == "address":
                    continue
                # Handle float('inf') and float('nan') values that cannot
                # be serialized to JSON or stored in SQLite.
                if isinstance(value, float) and (
                    math.isinf(value) or math.isnan(value)
                ):
                    value = None
                if hasattr(wallet, key):
                    setattr(wallet, key, value)

            await session.commit()

    async def _is_stale(self, address: str) -> bool:
        """Check if a wallet's analysis is stale or missing."""
        async with AsyncSessionLocal() as session:
            wallet = await session.get(DiscoveredWallet, address)
            if wallet is None:
                return True
            if wallet.last_analyzed_at is None:
                return True
            age = datetime.utcnow() - wallet.last_analyzed_at
            return age.total_seconds() > STALE_ANALYSIS_HOURS * 3600

    # ------------------------------------------------------------------
    # 9. Leaderboard Refresh
    # ------------------------------------------------------------------

    async def refresh_leaderboard(self):
        """
        Recalculate rank_position for all discovered wallets based
        on rank_score descending.
        """
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(DiscoveredWallet.address, DiscoveredWallet.rank_score)
                .order_by(desc(DiscoveredWallet.rank_score))
            )
            rows = result.all()

            for position, row in enumerate(rows, start=1):
                await session.execute(
                    update(DiscoveredWallet)
                    .where(DiscoveredWallet.address == row.address)
                    .values(rank_position=position)
                )

            await session.commit()

        logger.info(
            "Leaderboard refreshed",
            total_wallets=len(rows),
        )

    # ------------------------------------------------------------------
    # 10. Full Discovery Run
    # ------------------------------------------------------------------

    async def run_discovery(
        self,
        max_markets: int = 100,
        max_wallets_per_market: int = 50,
    ):
        """
        Full discovery pipeline:
            1. Fetch recent/active markets from Polymarket
            2. For each market, fetch recent trades to discover wallets
            3. Supplement with leaderboard wallets
            4. Deduplicate
            5. For each new/stale wallet, run full analysis
            6. Store/update in DB
            7. Refresh leaderboard
        """
        run_start = datetime.utcnow()
        logger.info(
            "Starting discovery run",
            max_markets=max_markets,
            max_wallets_per_market=max_wallets_per_market,
        )

        # --- Step 1: Fetch active markets ---
        try:
            markets = await self.client.get_markets(
                active=True, limit=min(max_markets, 100), offset=0
            )
        except Exception as e:
            logger.error("Failed to fetch markets", error=str(e))
            markets = []

        # Paginate if we need more
        if len(markets) < max_markets:
            offset = len(markets)
            while offset < max_markets:
                try:
                    page = await self.client.get_markets(
                        active=True,
                        limit=min(100, max_markets - offset),
                        offset=offset,
                    )
                    if not page:
                        break
                    markets.extend(page)
                    offset += len(page)
                    await asyncio.sleep(DELAY_BETWEEN_MARKETS)
                except Exception:
                    break

        logger.info("Fetched markets", count=len(markets))

        # --- Step 2 & 3: Discover wallet addresses ---
        all_addresses: set[str] = set()

        # From market trades (with concurrency limiter)
        semaphore = asyncio.Semaphore(5)

        async def discover_from_market(market):
            async with semaphore:
                addrs = await self._discover_wallets_from_market(
                    market, max_wallets=max_wallets_per_market
                )
                await asyncio.sleep(DELAY_BETWEEN_MARKETS)
                return addrs

        market_tasks = [discover_from_market(m) for m in markets]
        market_results = await asyncio.gather(*market_tasks, return_exceptions=True)

        for result in market_results:
            if isinstance(result, set):
                all_addresses.update(result)

        # From leaderboard
        leaderboard_addrs = await self._discover_wallets_from_leaderboard(
            scan_count=200
        )
        all_addresses.update(leaderboard_addrs)

        self._wallets_discovered_last_run = len(all_addresses)
        logger.info(
            "Wallet addresses discovered",
            total=len(all_addresses),
            from_markets=len(all_addresses) - len(leaderboard_addrs),
            from_leaderboard=len(leaderboard_addrs),
        )

        # --- Step 4: Filter to new/stale wallets ---
        addresses_to_analyze: list[str] = []
        for addr in all_addresses:
            if await self._is_stale(addr):
                addresses_to_analyze.append(addr)

        logger.info(
            "Wallets requiring analysis",
            total=len(addresses_to_analyze),
            skipped_fresh=len(all_addresses) - len(addresses_to_analyze),
        )

        # --- Step 5 & 6: Analyze and store ---
        analyzed_count = 0
        analysis_semaphore = asyncio.Semaphore(5)

        async def analyze_and_store(addr: str):
            nonlocal analyzed_count
            async with analysis_semaphore:
                try:
                    profile = await self.analyze_wallet(addr)
                    if profile is not None:
                        await self._upsert_wallet(profile)
                        analyzed_count += 1
                    await asyncio.sleep(DELAY_BETWEEN_WALLETS)
                except Exception as e:
                    logger.warning(
                        "Wallet analysis failed",
                        address=addr,
                        error=str(e),
                    )

        # Process in batches to avoid overwhelming the API
        batch_size = 50
        for i in range(0, len(addresses_to_analyze), batch_size):
            batch = addresses_to_analyze[i : i + batch_size]
            await asyncio.gather(*[analyze_and_store(a) for a in batch])
            logger.info(
                "Batch complete",
                progress=min(i + batch_size, len(addresses_to_analyze)),
                total=len(addresses_to_analyze),
                analyzed=analyzed_count,
            )

        # --- Step 7: Refresh leaderboard ---
        await self.refresh_leaderboard()

        # --- Record run metadata ---
        self._last_run_at = datetime.utcnow()
        self._wallets_analyzed_last_run = analyzed_count
        duration = (self._last_run_at - run_start).total_seconds()

        logger.info(
            "Discovery run complete",
            wallets_discovered=self._wallets_discovered_last_run,
            wallets_analyzed=analyzed_count,
            duration_seconds=round(duration, 1),
        )

    # ------------------------------------------------------------------
    # 11. Background Scheduler
    # ------------------------------------------------------------------

    async def start_background_discovery(self, interval_minutes: int = 60):
        """Run discovery on a recurring schedule."""
        self._running = True
        logger.info(
            "Background discovery started",
            interval_minutes=interval_minutes,
        )

        while self._running:
            try:
                await self.run_discovery()
            except Exception as e:
                logger.error("Discovery run failed", error=str(e))

            await asyncio.sleep(interval_minutes * 60)

    def stop(self):
        """Stop the background discovery loop."""
        self._running = False
        logger.info("Background discovery stopped")

    # ------------------------------------------------------------------
    # 12. Query Methods
    # ------------------------------------------------------------------

    async def get_leaderboard(
        self,
        limit: int = 100,
        offset: int = 0,
        min_trades: int = 10,
        min_pnl: float = 0.0,
        sort_by: str = "rank_score",
        sort_dir: str = "desc",
        tags: list[str] | None = None,
        recommendation: str | None = None,
    ) -> list[dict]:
        """
        Get the wallet leaderboard with filtering and sorting.

        Args:
            limit: Max results to return.
            offset: Pagination offset.
            min_trades: Minimum trade count filter.
            min_pnl: Minimum total PnL filter.
            sort_by: Column to sort by (rank_score, total_pnl, win_rate, sharpe_ratio, etc.).
            sort_dir: "asc" or "desc".
            tags: If provided, only return wallets that have ALL of these tags.
            recommendation: If provided, filter to this recommendation level.

        Returns:
            List of wallet dicts.
        """
        async with AsyncSessionLocal() as session:
            query = select(DiscoveredWallet).where(
                DiscoveredWallet.total_trades >= min_trades,
                DiscoveredWallet.total_pnl >= min_pnl,
            )

            if recommendation:
                query = query.where(
                    DiscoveredWallet.recommendation == recommendation
                )

            # Determine sort column
            sort_column = getattr(DiscoveredWallet, sort_by, None)
            if sort_column is None:
                sort_column = DiscoveredWallet.rank_score

            if sort_dir.lower() == "asc":
                query = query.order_by(asc(sort_column))
            else:
                query = query.order_by(desc(sort_column))

            query = query.offset(offset).limit(limit)

            result = await session.execute(query)
            wallets = result.scalars().all()

            rows = []
            for w in wallets:
                row = self._wallet_to_dict(w)

                # Client-side tag filtering (JSON column)
                if tags:
                    wallet_tags = row.get("tags") or []
                    if not all(t in wallet_tags for t in tags):
                        continue

                rows.append(row)

            return rows

    async def get_wallet_profile(self, address: str) -> dict | None:
        """
        Get comprehensive wallet profile with all metrics.

        Returns None if wallet has not been discovered yet.
        """
        address = address.lower()

        async with AsyncSessionLocal() as session:
            wallet = await session.get(DiscoveredWallet, address)
            if wallet is None:
                return None
            return self._wallet_to_dict(wallet)

    async def get_discovery_stats(self) -> dict:
        """
        Get aggregate statistics about the discovery engine's state.

        Returns dict with total_wallets, last_run_at, avg_rank_score, etc.
        """
        async with AsyncSessionLocal() as session:
            total_q = await session.execute(
                select(func.count(DiscoveredWallet.address))
            )
            total_wallets = total_q.scalar() or 0

            profitable_q = await session.execute(
                select(func.count(DiscoveredWallet.address)).where(
                    DiscoveredWallet.is_profitable == True  # noqa: E712
                )
            )
            profitable_count = profitable_q.scalar() or 0

            copy_q = await session.execute(
                select(func.count(DiscoveredWallet.address)).where(
                    DiscoveredWallet.recommendation == "copy_candidate"
                )
            )
            copy_candidates = copy_q.scalar() or 0

            avg_score_q = await session.execute(
                select(func.avg(DiscoveredWallet.rank_score))
            )
            avg_rank_score = avg_score_q.scalar() or 0.0

            avg_wr_q = await session.execute(
                select(func.avg(DiscoveredWallet.win_rate)).where(
                    DiscoveredWallet.total_trades >= 10
                )
            )
            avg_win_rate = avg_wr_q.scalar() or 0.0

        return {
            "total_wallets": total_wallets,
            "profitable_wallets": profitable_count,
            "copy_candidates": copy_candidates,
            "avg_rank_score": round(avg_rank_score, 4),
            "avg_win_rate": round(avg_win_rate, 4),
            "last_run_at": self._last_run_at.isoformat() if self._last_run_at else None,
            "wallets_discovered_last_run": self._wallets_discovered_last_run,
            "wallets_analyzed_last_run": self._wallets_analyzed_last_run,
            "is_running": self._running,
        }

    # ------------------------------------------------------------------
    # Internal Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _wallet_to_dict(w: DiscoveredWallet) -> dict:
        """Serialize a DiscoveredWallet ORM object to a plain dict."""
        return {
            "address": w.address,
            "username": w.username,
            "discovered_at": w.discovered_at.isoformat() if w.discovered_at else None,
            "last_analyzed_at": w.last_analyzed_at.isoformat() if w.last_analyzed_at else None,
            "discovery_source": w.discovery_source,
            # Basic stats
            "total_trades": w.total_trades,
            "wins": w.wins,
            "losses": w.losses,
            "win_rate": w.win_rate,
            "total_pnl": w.total_pnl,
            "realized_pnl": w.realized_pnl,
            "unrealized_pnl": w.unrealized_pnl,
            "total_invested": w.total_invested,
            "total_returned": w.total_returned,
            "avg_roi": w.avg_roi,
            "max_roi": w.max_roi,
            "min_roi": w.min_roi,
            "roi_std": w.roi_std,
            "unique_markets": w.unique_markets,
            "open_positions": w.open_positions,
            "days_active": w.days_active,
            "avg_hold_time_hours": w.avg_hold_time_hours,
            "trades_per_day": w.trades_per_day,
            "avg_position_size": w.avg_position_size,
            # Risk-adjusted
            "sharpe_ratio": w.sharpe_ratio,
            "sortino_ratio": w.sortino_ratio,
            "max_drawdown": w.max_drawdown,
            "profit_factor": w.profit_factor,
            "calmar_ratio": w.calmar_ratio,
            # Rolling
            "rolling_pnl": w.rolling_pnl,
            "rolling_roi": w.rolling_roi,
            "rolling_win_rate": w.rolling_win_rate,
            "rolling_trade_count": w.rolling_trade_count,
            "rolling_sharpe": w.rolling_sharpe,
            # Classification
            "anomaly_score": w.anomaly_score,
            "is_bot": w.is_bot,
            "is_profitable": w.is_profitable,
            "recommendation": w.recommendation,
            "strategies_detected": w.strategies_detected,
            "tags": w.tags,
            # Ranking
            "rank_score": w.rank_score,
            "rank_position": w.rank_position,
            # Clustering
            "cluster_id": w.cluster_id,
        }

    @staticmethod
    def _std_dev(values: list[float]) -> float:
        """Calculate sample standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    @staticmethod
    def _parse_timestamp(raw) -> datetime | None:
        """Parse a raw timestamp value into a datetime."""
        if raw is None:
            return None
        if isinstance(raw, datetime):
            return raw
        if isinstance(raw, (int, float)):
            try:
                return datetime.utcfromtimestamp(raw)
            except (ValueError, OSError):
                return None
        if isinstance(raw, str):
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(
                    tzinfo=None
                )
            except (ValueError, TypeError):
                return None
        return None

    def _compute_days_active(self, trades: list[dict]) -> int:
        """Compute the number of days between the first and last trade."""
        if not trades:
            return 0

        first_raw = trades[-1].get("timestamp", trades[-1].get("created_at"))
        last_raw = trades[0].get("timestamp", trades[0].get("created_at"))

        first = self._parse_timestamp(first_raw)
        last = self._parse_timestamp(last_raw)

        if first and last:
            return max((last - first).days, 1)
        return 30  # Default fallback

    def _filter_trades_after(
        self, trades: list[dict], cutoff: datetime
    ) -> list[dict]:
        """Return only trades whose timestamp is after ``cutoff``."""
        filtered: list[dict] = []
        for trade in trades:
            raw = trade.get(
                "timestamp",
                trade.get("created_at", trade.get("createdAt")),
            )
            ts = self._parse_timestamp(raw)
            if ts is None:
                # Keep trades we cannot parse (conservative)
                filtered.append(trade)
                continue
            if ts >= cutoff:
                filtered.append(trade)
        return filtered


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

wallet_discovery = WalletDiscoveryEngine()
