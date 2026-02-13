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

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timedelta
from utils.utcnow import utcnow, utcfromtimestamp
from typing import Any, Optional

from sqlalchemy import delete, select, func, update, desc, asc, cast, String, or_

from models.database import AppSettings, DiscoveredWallet, AsyncSessionLocal
from services.polymarket import polymarket_client
from services.smart_wallet_pool import smart_wallet_pool
from services.pause_state import global_pause_state
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

METRICS_SOURCE_VERSION = "accuracy_v2_closed_positions"

# Discovery behavior defaults (can be overridden at runtime from AppSettings)
DISCOVERY_DEFAULT_MAX_DISCOVERED_WALLETS = 20_000
DISCOVERY_DEFAULT_MAINTENANCE_ENABLED = True
DISCOVERY_DEFAULT_KEEP_RECENT_TRADE_DAYS = 7
DISCOVERY_DEFAULT_KEEP_NEW_DISCOVERIES_DAYS = 30
DISCOVERY_DEFAULT_MAINTENANCE_BATCH = 900
DISCOVERY_DEFAULT_STALE_ANALYSIS_HOURS = 12
DISCOVERY_DEFAULT_ANALYSIS_PRIORITY_BATCH_LIMIT = 2500
DISCOVERY_DEFAULT_DELAY_BETWEEN_MARKETS = 0.25
DISCOVERY_DEFAULT_DELAY_BETWEEN_WALLETS = 0.15
DISCOVERY_DEFAULT_MAX_MARKETS_PER_RUN = 100
DISCOVERY_DEFAULT_MAX_WALLETS_PER_MARKET = 50

POOL_FLAG_MANUAL_INCLUDE = "pool_manual_include"
POOL_FLAG_MANUAL_EXCLUDE = "pool_manual_exclude"
POOL_FLAG_BLACKLISTED = "pool_blacklisted"


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
        self._background_running = False
        self._last_run_at: Optional[datetime] = None
        self._wallets_discovered_last_run: int = 0
        self._wallets_analyzed_last_run: int = 0
        self._wallets_seeded_last_run: int = 0
        self._wallets_pruned_last_run: int = 0
        self._runtime_discovery_settings: dict[str, Any] | None = None

    @staticmethod
    def _coerce_int(value: Any, default: int, minimum: Optional[int] = None) -> int:
        try:
            value_int = int(value)
        except (TypeError, ValueError):
            return default
        if minimum is not None and value_int < minimum:
            return minimum
        return value_int

    @staticmethod
    def _coerce_float(value: Any, default: float, minimum: Optional[float] = None) -> float:
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            return default
        if minimum is not None and value_float < minimum:
            return minimum
        return value_float

    @staticmethod
    def _coerce_bool(value: Any, default: bool) -> bool:
        if value is None:
            return default
        return bool(value)

    @staticmethod
    def _discovery_settings_defaults() -> dict[str, Any]:
        return {
            "max_discovered_wallets": DISCOVERY_DEFAULT_MAX_DISCOVERED_WALLETS,
            "maintenance_enabled": DISCOVERY_DEFAULT_MAINTENANCE_ENABLED,
            "keep_recent_trade_days": DISCOVERY_DEFAULT_KEEP_RECENT_TRADE_DAYS,
            "keep_new_discoveries_days": DISCOVERY_DEFAULT_KEEP_NEW_DISCOVERIES_DAYS,
            "maintenance_batch": DISCOVERY_DEFAULT_MAINTENANCE_BATCH,
            "stale_analysis_hours": DISCOVERY_DEFAULT_STALE_ANALYSIS_HOURS,
            "analysis_priority_batch_limit": DISCOVERY_DEFAULT_ANALYSIS_PRIORITY_BATCH_LIMIT,
            "delay_between_markets": DISCOVERY_DEFAULT_DELAY_BETWEEN_MARKETS,
            "delay_between_wallets": DISCOVERY_DEFAULT_DELAY_BETWEEN_WALLETS,
            "max_markets_per_run": DISCOVERY_DEFAULT_MAX_MARKETS_PER_RUN,
            "max_wallets_per_market": DISCOVERY_DEFAULT_MAX_WALLETS_PER_MARKET,
        }

    def _discovery_setting(self, key: str, default: Any) -> Any:
        return (self._runtime_discovery_settings or {}).get(
            key, self._discovery_settings_defaults().get(key, default)
        )

    async def _load_discovery_settings(self) -> dict[str, Any]:
        settings = self._discovery_settings_defaults()
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(AppSettings).where(AppSettings.id == "default")
                )
                row = result.scalar_one_or_none()
                if row is None:
                    return settings

                settings["max_discovered_wallets"] = self._coerce_int(
                    row.discovery_max_discovered_wallets,
                    DISCOVERY_DEFAULT_MAX_DISCOVERED_WALLETS,
                    minimum=10,
                )
                settings["maintenance_enabled"] = self._coerce_bool(
                    row.discovery_maintenance_enabled,
                    DISCOVERY_DEFAULT_MAINTENANCE_ENABLED,
                )
                settings["keep_recent_trade_days"] = self._coerce_int(
                    row.discovery_keep_recent_trade_days,
                    DISCOVERY_DEFAULT_KEEP_RECENT_TRADE_DAYS,
                    minimum=1,
                )
                settings["keep_new_discoveries_days"] = self._coerce_int(
                    row.discovery_keep_new_discoveries_days,
                    DISCOVERY_DEFAULT_KEEP_NEW_DISCOVERIES_DAYS,
                    minimum=1,
                )
                settings["maintenance_batch"] = self._coerce_int(
                    row.discovery_maintenance_batch,
                    DISCOVERY_DEFAULT_MAINTENANCE_BATCH,
                    minimum=10,
                )
                settings["stale_analysis_hours"] = self._coerce_int(
                    row.discovery_stale_analysis_hours,
                    DISCOVERY_DEFAULT_STALE_ANALYSIS_HOURS,
                    minimum=1,
                )
                settings["analysis_priority_batch_limit"] = self._coerce_int(
                    row.discovery_analysis_priority_batch_limit,
                    DISCOVERY_DEFAULT_ANALYSIS_PRIORITY_BATCH_LIMIT,
                    minimum=100,
                )
                settings["delay_between_markets"] = self._coerce_float(
                    row.discovery_delay_between_markets,
                    DISCOVERY_DEFAULT_DELAY_BETWEEN_MARKETS,
                    minimum=0.0,
                )
                settings["delay_between_wallets"] = self._coerce_float(
                    row.discovery_delay_between_wallets,
                    DISCOVERY_DEFAULT_DELAY_BETWEEN_WALLETS,
                    minimum=0.0,
                )
                settings["max_markets_per_run"] = self._coerce_int(
                    row.discovery_max_markets_per_run,
                    DISCOVERY_DEFAULT_MAX_MARKETS_PER_RUN,
                    minimum=1,
                )
                settings["max_wallets_per_market"] = self._coerce_int(
                    row.discovery_max_wallets_per_market,
                    DISCOVERY_DEFAULT_MAX_WALLETS_PER_MARKET,
                    minimum=1,
                )
        except Exception:
            logger.exception("Failed to load discovery settings from DB; using defaults")

        self._runtime_discovery_settings = settings
        return settings

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
            avg_price = float(pos.get("avgPrice", pos.get("avg_price", 0)) or 0)
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

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _summarize_closed_positions(self, closed_positions: list[dict]) -> dict:
        wins = 0
        losses = 0
        market_ids: set[str] = set()
        market_pnls: list[float] = []
        market_rois: list[float] = []

        for pos in closed_positions:
            if not isinstance(pos, dict):
                continue

            market_id = (
                pos.get("market")
                or pos.get("condition_id")
                or pos.get("conditionId")
                or pos.get("asset")
                or pos.get("token_id")
                or pos.get("tokenId")
                or ""
            )
            market_key = str(market_id).strip()
            if market_key:
                market_ids.add(market_key)

            realized_pnl = self._to_float(pos.get("realizedPnl", 0) or 0)
            initial_value = self._to_float(pos.get("initialValue", 0) or 0)
            market_pnls.append(realized_pnl)
            if initial_value > 0:
                market_rois.append((realized_pnl / initial_value) * 100.0)

            if realized_pnl > 0:
                wins += 1
            elif realized_pnl < 0:
                losses += 1

        return {
            "wins": wins,
            "losses": losses,
            "market_ids": market_ids,
            "market_pnls": market_pnls,
            "market_rois": market_rois,
        }

    def _build_accuracy_first_stats(
        self,
        *,
        base_stats: dict,
        pnl_snapshot: dict | None,
        closed_positions: list[dict],
    ) -> dict:
        """Merge legacy trade-derived stats with closed-position/PnL endpoint truth."""
        stats = dict(base_stats)
        closed_summary = self._summarize_closed_positions(closed_positions)

        if isinstance(pnl_snapshot, dict):
            total_trades = int(self._to_float(pnl_snapshot.get("total_trades"), 0))
            open_positions = int(self._to_float(pnl_snapshot.get("open_positions"), 0))
            total_invested = self._to_float(pnl_snapshot.get("total_invested"), 0.0)
            total_returned = self._to_float(pnl_snapshot.get("total_returned"), 0.0)
            realized_pnl = self._to_float(pnl_snapshot.get("realized_pnl"), 0.0)
            unrealized_pnl = self._to_float(pnl_snapshot.get("unrealized_pnl"), 0.0)
            total_pnl = self._to_float(
                pnl_snapshot.get("total_pnl"),
                realized_pnl + unrealized_pnl,
            )

            if total_trades > 0:
                stats["total_trades"] = total_trades
            stats["open_positions"] = open_positions
            stats["total_invested"] = total_invested
            stats["total_returned"] = total_returned
            stats["realized_pnl"] = realized_pnl
            stats["unrealized_pnl"] = unrealized_pnl
            stats["total_pnl"] = total_pnl

        wins = int(closed_summary["wins"])
        losses = int(closed_summary["losses"])
        closed_total = wins + losses
        if closed_total > 0:
            stats["wins"] = wins
            stats["losses"] = losses
            stats["win_rate"] = wins / closed_total

        market_rois = list(closed_summary["market_rois"])
        market_pnls = list(closed_summary["market_pnls"])
        if market_rois:
            stats["avg_roi"] = sum(market_rois) / len(market_rois)
            stats["max_roi"] = max(market_rois)
            stats["min_roi"] = min(market_rois)
            stats["roi_std"] = self._std_dev(market_rois) if len(market_rois) > 1 else 0.0
            stats["_market_rois"] = market_rois
            stats["_market_pnls"] = market_pnls
        elif market_pnls:
            stats["_market_pnls"] = market_pnls
            total_invested = self._to_float(stats.get("total_invested"), 0.0)
            total_pnl = self._to_float(stats.get("total_pnl"), 0.0)
            stats["avg_roi"] = (total_pnl / total_invested * 100.0) if total_invested > 0 else 0.0
            stats["max_roi"] = stats["avg_roi"]
            stats["min_roi"] = stats["avg_roi"]
            stats["roi_std"] = 0.0

        if closed_summary["market_ids"]:
            stats["unique_markets"] = max(
                int(stats.get("unique_markets", 0) or 0),
                len(closed_summary["market_ids"]),
            )

        days_active = max(int(stats.get("days_active", 0) or 0), 1)
        total_trades = int(stats.get("total_trades", 0) or 0)
        total_invested = self._to_float(stats.get("total_invested"), 0.0)
        stats["trades_per_day"] = total_trades / days_active
        stats["avg_position_size"] = (
            total_invested / total_trades if total_trades > 0 else 0.0
        )

        for key in (
            "win_rate",
            "total_pnl",
            "realized_pnl",
            "unrealized_pnl",
            "total_invested",
            "total_returned",
            "avg_roi",
            "max_roi",
            "min_roi",
            "roi_std",
            "trades_per_day",
            "avg_position_size",
        ):
            value = stats.get(key)
            if isinstance(value, float) and not math.isfinite(value):
                stats[key] = 0.0

        return stats

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
            downside_variance = sum(r**2 for r in negative_returns) / len(
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
            market_id = trade.get("market", trade.get("condition_id", ""))
            if market_id:
                market_groups.setdefault(market_id, []).append(trade)

        # Multi-market date sweep (NegRisk)
        for _mid, mtrades in market_groups.items():
            if len(mtrades) >= 3:
                strategies.add("negrisk_date_sweep")

        # Basic arbitrage (same market, opposite outcomes)
        for mtrades in market_groups.values():
            sides = set(t.get("outcome", t.get("side", "")) for t in mtrades)
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
        if (
            profit_factor is not None
            and math.isfinite(profit_factor)
            and profit_factor >= 3.0
        ):
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
            # Fetch data in parallel. Trade history is widened beyond 500 so
            # strategy/rolling windows use deeper context while PnL truth comes
            # from accuracy-first endpoints.
            trades, positions, profile, pnl_snapshot, closed_positions = await asyncio.gather(
                self.client.get_wallet_trades(address, limit=1500),
                self.client.get_wallet_positions(address),
                self.client.get_user_profile(address),
                self.client.get_wallet_pnl(address, time_period="ALL"),
                self.client.get_closed_positions_paginated(address, max_positions=1500),
            )
        except Exception as e:
            logger.error(
                "Failed to fetch data for wallet",
                address=address,
                error=str(e),
            )
            return None

        if (
            len(trades) < MIN_TRADES_FOR_ANALYSIS
            and not positions
            and len(closed_positions) < MIN_TRADES_FOR_ANALYSIS
        ):
            return None

        # All statistical computations are CPU-bound. Run them in the
        # thread pool so the async event loop stays free for API requests.
        def _compute_profile(
            engine,
            trade_list,
            position_list,
            pnl_snapshot_payload,
            closed_position_payload,
        ):
            stats = engine._calculate_trade_stats(trade_list, position_list)
            stats = engine._build_accuracy_first_stats(
                base_stats=stats,
                pnl_snapshot=pnl_snapshot_payload,
                closed_positions=closed_position_payload,
            )
            risk_metrics = engine._calculate_risk_adjusted_metrics(
                market_rois=stats["_market_rois"],
                market_pnls=stats["_market_pnls"],
                days_active=stats["days_active"],
                total_pnl=stats["total_pnl"],
                total_invested=stats["total_invested"],
            )
            now_inner = utcnow()
            rolling = engine._calculate_rolling_windows(trade_list, now_inner)
            strategies = engine._detect_strategies(trade_list)
            classification = engine._classify_wallet(stats, risk_metrics)
            rank_input = {
                "sharpe_ratio": risk_metrics["sharpe_ratio"],
                "profit_factor": risk_metrics["profit_factor"],
                "win_rate": stats["win_rate"],
                "total_pnl": stats["total_pnl"],
                "max_drawdown": risk_metrics["max_drawdown"],
            }
            rank_score = engine._calculate_rank_score(rank_input)
            return (
                stats,
                risk_metrics,
                rolling,
                strategies,
                classification,
                rank_score,
                now_inner,
            )

        (
            stats,
            risk_metrics,
            rolling,
            strategies,
            classification,
            rank_score,
            now,
        ) = await asyncio.to_thread(
            _compute_profile,
            self,
            trades,
            positions,
            pnl_snapshot,
            closed_positions,
        )

        username = profile.get("username") if profile else None

        return {
            "address": address,
            "username": username,
            "last_analyzed_at": now,
            "discovery_source": "scan",
            "metrics_source_version": METRICS_SOURCE_VERSION,
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

    @staticmethod
    def _coerce_source_flags(raw_flags: Any) -> dict[str, Any]:
        if isinstance(raw_flags, dict):
            return raw_flags
        return {}

    @staticmethod
    def _is_pool_protected(flags: dict[str, Any]) -> bool:
        return bool(
            flags.get(POOL_FLAG_MANUAL_INCLUDE)
            or flags.get(POOL_FLAG_MANUAL_EXCLUDE)
            or flags.get(POOL_FLAG_BLACKLISTED)
        )

    def _is_wallet_discovery_protected(self, wallet: DiscoveredWallet) -> bool:
        if wallet.in_top_pool:
            return True

        source = str(wallet.discovery_source or "").strip().lower()
        if source in {"manual", "manual_pool", "referral"}:
            return True

        flags = self._coerce_source_flags(wallet.source_flags)
        return self._is_pool_protected(flags)

    def _discovery_curation_score(self, wallet: DiscoveredWallet, now: datetime) -> float:
        score = float(wallet.rank_score or 0.0)
        score += min(float(wallet.total_trades or 0), 2500) / 2500.0
        if wallet.last_analyzed_at is not None:
            score += max(
                0.0,
                1.0 - (now - wallet.last_analyzed_at).total_seconds() / 2_592_000.0,
            )
        if wallet.total_pnl and wallet.total_pnl > 0:
            score += 0.3
        if str(wallet.recommendation or "").strip().lower() == "copy_candidate":
            score += 0.5
        if bool(wallet.is_profitable):
            score += 0.2
        return score

    @staticmethod
    def _chunked(addresses: list[str], size: int) -> list[list[str]]:
        if size <= 0:
            return [addresses]
        return [addresses[i : i + size] for i in range(0, len(addresses), size)]

    async def _upsert_discovered_placeholders(
        self,
        addresses: set[str],
        discovery_source: str = "scan",
    ) -> int:
        """Insert discovered addresses as seed rows even before full analysis."""
        normalized = {
            str(address).strip().lower()
            for address in addresses
            if isinstance(address, str) and str(address).strip()
        }
        if not normalized:
            return 0

        now = utcnow()
        normalized_list = sorted(normalized)
        async with AsyncSessionLocal() as session:
            existing: set[str] = set()
            maintenance_batch = self._coerce_int(
                self._discovery_setting("maintenance_batch", 900),
                DISCOVERY_DEFAULT_MAINTENANCE_BATCH,
                minimum=10,
            )
            for chunk in self._chunked(normalized_list, maintenance_batch):
                rows = await session.execute(
                    select(DiscoveredWallet.address).where(
                        DiscoveredWallet.address.in_(chunk)
                    )
                )
                existing.update({str(row.address).lower() for row in rows.all() if row.address})

            created = 0
            for address in normalized_list:
                if address in existing:
                    continue
                session.add(
                    DiscoveredWallet(
                        address=address,
                        discovered_at=now,
                        discovery_source=discovery_source,
                    )
                )
                created += 1

            if created:
                await session.commit()
            return created

    async def _cleanup_discovered_wallet_catalog(self, now: datetime | None = None) -> int:
        """
        Keep the discovered-wallet catalog bounded by removing weak non-critical rows.
        """
        max_discovered_wallets = self._coerce_int(
            self._discovery_setting("max_discovered_wallets", 20000),
            DISCOVERY_DEFAULT_MAX_DISCOVERED_WALLETS,
            minimum=10,
        )
        maintenance_enabled = self._coerce_bool(
            self._discovery_setting("maintenance_enabled", True),
            DISCOVERY_DEFAULT_MAINTENANCE_ENABLED,
        )
        if not maintenance_enabled:
            return 0

        now = now or utcnow()
        async with AsyncSessionLocal() as session:
            total_result = await session.execute(
                select(func.count(DiscoveredWallet.address))
            )
            total_wallets = int(total_result.scalar() or 0)
            if total_wallets <= max_discovered_wallets:
                return 0

            cutoff_trade = now - timedelta(
                days=max(
                    1,
                    self._coerce_int(
                        self._discovery_setting("keep_recent_trade_days", 7),
                        DISCOVERY_DEFAULT_KEEP_RECENT_TRADE_DAYS,
                        minimum=1,
                    ),
                )
            )
            cutoff_discovered = now - timedelta(
                days=max(
                    1,
                    self._coerce_int(
                        self._discovery_setting("keep_new_discoveries_days", 30),
                        DISCOVERY_DEFAULT_KEEP_NEW_DISCOVERIES_DAYS,
                        minimum=1,
                    ),
                )
            )

            rows = await session.execute(select(DiscoveredWallet))
            wallets = list(rows.scalars().all())

            candidates: list[DiscoveredWallet] = []
            non_protected: list[DiscoveredWallet] = []

            for wallet in wallets:
                if self._is_wallet_discovery_protected(wallet):
                    continue
                non_protected.append(wallet)
                if (
                    wallet.last_trade_at is not None
                    and wallet.last_trade_at >= cutoff_trade
                ):
                    continue
                if (
                    wallet.discovered_at is not None
                    and wallet.discovered_at >= cutoff_discovered
                ):
                    continue
                candidates.append(wallet)

            # Prefer removing non-recent/non-essential rows first.
            removable = candidates if candidates else non_protected

            removable.sort(
                key=lambda wallet: self._discovery_curation_score(wallet, now)
            )
            remove_count = total_wallets - max_discovered_wallets
            if remove_count <= 0:
                return 0

            # If recent/preserved rows prevent hitting the target size, fall back
            # to all non-protected rows.
            if len(removable) < remove_count:
                removable = non_protected

            if not removable:
                return 0

            to_remove = [wallet.address for wallet in removable[:remove_count]]
            removed = 0
            maintenance_batch = self._coerce_int(
                self._discovery_setting("maintenance_batch", 900),
                DISCOVERY_DEFAULT_MAINTENANCE_BATCH,
                minimum=10,
            )
            for chunk in self._chunked(to_remove, maintenance_batch):
                result = await session.execute(
                    delete(DiscoveredWallet).where(DiscoveredWallet.address.in_(chunk))
                )
                removed += int(result.rowcount or 0)

            await session.commit()
            return removed

    async def _upsert_wallet(self, data: dict):
        """Insert or update a DiscoveredWallet record."""
        address = data["address"]

        async with AsyncSessionLocal() as session:
            wallet = await session.get(DiscoveredWallet, address)

            if wallet is None:
                wallet = DiscoveredWallet(
                    address=address,
                    discovered_at=utcnow(),
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
            age = utcnow() - wallet.last_analyzed_at
            stale_analysis_hours = self._coerce_int(
                self._discovery_setting("stale_analysis_hours", 12),
                DISCOVERY_DEFAULT_STALE_ANALYSIS_HOURS,
                minimum=1,
            )
            return age.total_seconds() > stale_analysis_hours * 3600

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
                select(DiscoveredWallet.address, DiscoveredWallet.rank_score).order_by(
                    desc(DiscoveredWallet.rank_score)
                )
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

    @staticmethod
    def _merge_unique_addresses(*groups: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for group in groups:
            for raw in group:
                addr = str(raw or "").strip().lower()
                if not addr or addr in seen:
                    continue
                seen.add(addr)
                merged.append(addr)
        return merged

    async def _priority_analysis_queue(
        self,
        limit: int | None = None,
    ) -> tuple[list[str], dict[str, int]]:
        """Build priority addresses: in-pool unanalyzed, smart-pool unanalyzed, metrics backfill."""
        default_limit = self._coerce_int(
            self._discovery_setting("analysis_priority_batch_limit", 2500),
            DISCOVERY_DEFAULT_ANALYSIS_PRIORITY_BATCH_LIMIT,
            minimum=100,
        )
        priority_limit = max(50, min(limit or default_limit, default_limit))
        backfill_limit = max(100, min(priority_limit, default_limit))

        async with AsyncSessionLocal() as session:
            top_pool_rows = await session.execute(
                select(DiscoveredWallet.address)
                .where(
                    DiscoveredWallet.in_top_pool == True,  # noqa: E712
                    DiscoveredWallet.last_analyzed_at.is_(None),
                )
                .order_by(
                    desc(func.coalesce(DiscoveredWallet.trades_24h, 0)),
                    desc(func.coalesce(DiscoveredWallet.trades_1h, 0)),
                    desc(DiscoveredWallet.last_trade_at),
                    desc(DiscoveredWallet.discovered_at),
                )
                .limit(priority_limit)
            )
            top_pool = [str(row.address).lower() for row in top_pool_rows.all() if row.address]

            smart_pool_query = (
                select(DiscoveredWallet.address)
                .where(
                    DiscoveredWallet.discovery_source == "smart_pool",
                    DiscoveredWallet.last_analyzed_at.is_(None),
                )
                .order_by(
                    desc(func.coalesce(DiscoveredWallet.trades_24h, 0)),
                    desc(func.coalesce(DiscoveredWallet.trades_1h, 0)),
                    desc(DiscoveredWallet.last_trade_at),
                    desc(DiscoveredWallet.discovered_at),
                )
                .limit(priority_limit)
            )
            if top_pool:
                smart_pool_query = smart_pool_query.where(
                    DiscoveredWallet.address.notin_(top_pool)
                )
            smart_pool_rows = await session.execute(smart_pool_query)
            smart_pool = [
                str(row.address).lower() for row in smart_pool_rows.all() if row.address
            ]

            backfill_rows = await session.execute(
                select(DiscoveredWallet.address)
                .where(
                    DiscoveredWallet.last_analyzed_at.is_not(None),
                    or_(
                        DiscoveredWallet.metrics_source_version.is_(None),
                        DiscoveredWallet.metrics_source_version != METRICS_SOURCE_VERSION,
                    ),
                )
                .order_by(
                    desc(func.coalesce(DiscoveredWallet.in_top_pool, False)),
                    desc(func.coalesce(DiscoveredWallet.trades_24h, 0)),
                    DiscoveredWallet.last_analyzed_at.asc(),
                )
                .limit(backfill_limit)
            )
            backfill = [str(row.address).lower() for row in backfill_rows.all() if row.address]

        ordered = self._merge_unique_addresses(top_pool, smart_pool, backfill)
        counts = {
            "top_pool_unanalyzed": len(top_pool),
            "smart_pool_unanalyzed": len(smart_pool),
            "metrics_backfill": len(backfill),
            "priority_total": len(ordered),
        }
        return ordered, counts

    async def get_priority_backlog_count(self) -> int:
        """Count high-priority wallets waiting for initial analysis."""
        async with AsyncSessionLocal() as session:
            top_pool_unanalyzed = (
                (
                    await session.execute(
                        select(func.count(DiscoveredWallet.address)).where(
                            DiscoveredWallet.in_top_pool == True,  # noqa: E712
                            DiscoveredWallet.last_analyzed_at.is_(None),
                        )
                    )
                ).scalar()
                or 0
            )
            smart_pool_unanalyzed = (
                (
                    await session.execute(
                        select(func.count(DiscoveredWallet.address)).where(
                            DiscoveredWallet.discovery_source == "smart_pool",
                            DiscoveredWallet.last_analyzed_at.is_(None),
                            or_(
                                DiscoveredWallet.in_top_pool.is_(None),
                                DiscoveredWallet.in_top_pool == False,  # noqa: E712
                            ),
                        )
                    )
                ).scalar()
                or 0
            )
        return int(top_pool_unanalyzed + smart_pool_unanalyzed)

    # ------------------------------------------------------------------
    # 10. Full Discovery Run
    # ------------------------------------------------------------------

    async def run_discovery(
        self,
        max_markets: int | None = None,
        max_wallets_per_market: int | None = None,
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
        if self._running:
            logger.warning("Discovery already running, skipping")
            return

        self._running = True
        self._runtime_discovery_settings = None
        run_start = utcnow()
        try:
            runtime_settings = await self._load_discovery_settings()
            max_markets = self._coerce_int(
                max_markets,
                runtime_settings["max_markets_per_run"],
                minimum=1,
            )
            max_wallets_per_market = self._coerce_int(
                max_wallets_per_market,
                runtime_settings["max_wallets_per_market"],
                minimum=1,
            )
            delay_between_markets = self._coerce_float(
                runtime_settings["delay_between_markets"],
                DISCOVERY_DEFAULT_DELAY_BETWEEN_MARKETS,
                minimum=0.0,
            )
            delay_between_wallets = self._coerce_float(
                runtime_settings["delay_between_wallets"],
                DISCOVERY_DEFAULT_DELAY_BETWEEN_WALLETS,
                minimum=0.0,
            )
        except Exception:
            max_markets = self._coerce_int(max_markets, DISCOVERY_DEFAULT_MAX_MARKETS_PER_RUN, minimum=1)
            max_wallets_per_market = self._coerce_int(
                max_wallets_per_market,
                DISCOVERY_DEFAULT_MAX_WALLETS_PER_MARKET,
                minimum=1,
            )
            delay_between_markets = DISCOVERY_DEFAULT_DELAY_BETWEEN_MARKETS
            delay_between_wallets = DISCOVERY_DEFAULT_DELAY_BETWEEN_WALLETS

        logger.info(
            "Starting discovery run",
            max_markets=max_markets,
            max_wallets_per_market=max_wallets_per_market,
        )

        try:
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
                        await asyncio.sleep(delay_between_markets)
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
                    await asyncio.sleep(delay_between_markets)
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

            seeded = await self._upsert_discovered_placeholders(
                all_addresses,
                discovery_source="scan",
            )
            self._wallets_seeded_last_run = seeded

            self._wallets_discovered_last_run = len(all_addresses)
            logger.info(
                "Wallet addresses discovered",
                total=len(all_addresses),
                from_markets=len(all_addresses) - len(leaderboard_addrs),
                from_leaderboard=len(leaderboard_addrs),
            )

            # --- Step 4: Filter to new/stale wallets ---
            # Batch the staleness check into a single DB query instead of
            # one query per wallet, which was blocking the event loop for
            # hundreds/thousands of sequential DB round-trips.
            stale_addresses: list[str] = []
            stale_cutoff = utcnow() - timedelta(
                hours=self._coerce_int(
                    self._discovery_setting("stale_analysis_hours", 12),
                    DISCOVERY_DEFAULT_STALE_ANALYSIS_HOURS,
                    minimum=1,
                )
            )
            async with AsyncSessionLocal() as session:
                existing = {}
                address_list = sorted(all_addresses)
                maintenance_batch = self._coerce_int(
                    self._discovery_setting("maintenance_batch", 900),
                    DISCOVERY_DEFAULT_MAINTENANCE_BATCH,
                    minimum=10,
                )
                for chunk in self._chunked(address_list, maintenance_batch):
                    result = await session.execute(
                        select(
                            DiscoveredWallet.address,
                            DiscoveredWallet.last_analyzed_at,
                        ).where(DiscoveredWallet.address.in_(chunk))
                    )
                    existing.update(
                        {
                            row.address: row.last_analyzed_at
                            for row in result.all()
                            if row.address
                        }
                    )

            for addr in all_addresses:
                last_analyzed = existing.get(addr)
                if last_analyzed is None or last_analyzed < stale_cutoff:
                    stale_addresses.append(addr)

            priority_addresses, priority_counts = await self._priority_analysis_queue()
            addresses_to_analyze = self._merge_unique_addresses(
                priority_addresses,
                stale_addresses,
            )

            logger.info(
                "Wallets requiring analysis",
                total=len(addresses_to_analyze),
                priority_total=priority_counts.get("priority_total", 0),
                priority_top_pool_unanalyzed=priority_counts.get(
                    "top_pool_unanalyzed", 0
                ),
                priority_smart_pool_unanalyzed=priority_counts.get(
                    "smart_pool_unanalyzed", 0
                ),
                priority_metrics_backfill=priority_counts.get("metrics_backfill", 0),
                stale_discovered=len(stale_addresses),
                skipped_fresh=max(len(all_addresses) - len(stale_addresses), 0),
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
                        await asyncio.sleep(delay_between_wallets)
                    except Exception as e:
                        logger.warning(
                            "Wallet analysis failed",
                            address=addr,
                            error=str(e),
                        )

            # Process in batches to avoid overwhelming the API.
            # Yield to the event loop between batches so API requests
            # can be served while discovery is running.
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
                # Yield to event loop between batches
                await asyncio.sleep(0)

            # --- Step 7: Refresh leaderboard ---
            await self.refresh_leaderboard()

            wallets_pruned = await self._cleanup_discovered_wallet_catalog()
            self._wallets_pruned_last_run = wallets_pruned

            try:
                await smart_wallet_pool.recompute_pool()
            except Exception as e:
                logger.warning(
                    "Smart pool recompute after discovery failed",
                    error=str(e),
                )

            # --- Record run metadata ---
            self._last_run_at = utcnow()
            self._wallets_analyzed_last_run = analyzed_count
            duration = (self._last_run_at - run_start).total_seconds()

            logger.info(
                "Discovery run complete",
                wallets_discovered=self._wallets_discovered_last_run,
                wallets_analyzed=analyzed_count,
                wallets_seeded=seeded,
                wallets_pruned=wallets_pruned,
                duration_seconds=round(duration, 1),
            )
        finally:
            self._running = False
            self._runtime_discovery_settings = None

    # ------------------------------------------------------------------
    # 11. Background Scheduler
    # ------------------------------------------------------------------

    async def start_background_discovery(self, interval_minutes: int = 60):
        """Run discovery on a recurring schedule. First run after 60s so API can serve requests at startup."""
        if self._background_running:
            logger.info("Background discovery already running")
            return
        self._background_running = True
        logger.info(
            "Background discovery started",
            interval_minutes=interval_minutes,
        )

        # Defer first run so /opportunities, /scanner/status, etc. don't timeout at startup
        await asyncio.sleep(60)

        while self._background_running:
            if not global_pause_state.is_paused:
                try:
                    await self.run_discovery()
                except Exception as e:
                    logger.error("Discovery run failed", error=str(e))

            await asyncio.sleep(interval_minutes * 60)

    def stop(self):
        """Stop the background discovery loop."""
        self._background_running = False
        logger.info("Background discovery stopped")

    # ------------------------------------------------------------------
    # 12. Query Methods
    # ------------------------------------------------------------------

    # Maps sort_by field names to their rolling window JSON column equivalents
    SORT_FIELD_TO_ROLLING_COLUMN = {
        "total_pnl": "rolling_pnl",
        "win_rate": "rolling_win_rate",
        "sharpe_ratio": "rolling_sharpe",
        "total_trades": "rolling_trade_count",
        "avg_roi": "rolling_roi",
    }

    async def get_leaderboard(
        self,
        limit: int = 100,
        offset: int = 0,
        min_trades: int = 0,
        min_pnl: float = 0.0,
        insider_only: bool = False,
        min_insider_score: float | None = None,
        sort_by: str = "rank_score",
        sort_dir: str = "desc",
        tags: list[str] | None = None,
        recommendation: str | None = None,
        window_key: str | None = None,
        active_within_hours: int | None = None,
        min_activity_score: float | None = None,
        pool_only: bool = False,
        tier: str | None = None,
        search_text: str | None = None,
        unique_entities_only: bool = False,
        market_category: str | None = None,
    ) -> dict:
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
            window_key: Rolling window key ("1d", "7d", "30d", "90d").
                        When set, sorts/filters use rolling window metrics for that period.
            search_text: Optional free-text filter against address/username/tags/strategies.
            unique_entities_only: If true, collapse clustered wallets to one row per cluster.
            market_category: Optional market-focus category filter.

        Returns:
            Dict with 'wallets' list, 'total' count, and 'window_key' if set.
        """
        async with AsyncSessionLocal() as session:
            normalized_tags = [
                t.strip().lower() for t in (tags or []) if isinstance(t, str) and t.strip()
            ]
            base_filter = [
                DiscoveredWallet.total_trades >= min_trades,
                DiscoveredWallet.total_pnl >= min_pnl,
            ]
            if insider_only:
                base_filter.append(DiscoveredWallet.insider_score >= 0.60)
                base_filter.append(DiscoveredWallet.insider_confidence >= 0.50)
                base_filter.append(DiscoveredWallet.insider_sample_size >= 15)
            if min_insider_score is not None:
                base_filter.append(DiscoveredWallet.insider_score >= min_insider_score)
            if recommendation:
                base_filter.append(DiscoveredWallet.recommendation == recommendation)
            if active_within_hours is not None:
                cutoff = utcnow() - timedelta(hours=active_within_hours)
                base_filter.append(DiscoveredWallet.last_trade_at >= cutoff)
            if min_activity_score is not None:
                base_filter.append(DiscoveredWallet.activity_score >= min_activity_score)
            if pool_only:
                base_filter.append(DiscoveredWallet.in_top_pool == True)  # noqa: E712
            if tier:
                base_filter.append(DiscoveredWallet.pool_tier == tier.lower())
            if normalized_tags:
                for tag in normalized_tags:
                    base_filter.append(
                        func.lower(cast(DiscoveredWallet.tags, String)).like(
                            f'%"{tag}"%'
                        )
                    )
            if search_text:
                q = f"%{search_text.strip().lower()}%"
                if q != "%%":
                    base_filter.append(
                        or_(
                            func.lower(DiscoveredWallet.address).like(q),
                            func.lower(func.coalesce(DiscoveredWallet.username, "")).like(q),
                            func.lower(cast(DiscoveredWallet.tags, String)).like(q),
                            func.lower(cast(DiscoveredWallet.strategies_detected, String)).like(q),
                            func.lower(cast(DiscoveredWallet.source_flags, String)).like(q),
                        )
                    )
            if market_category:
                normalized_category = market_category.strip().lower()
                if normalized_category not in {"", "all", "overall"}:
                    category_flag = f'%\"leaderboard_category_{normalized_category}\": true%'
                    category_tag = f'%\"market_{normalized_category}\"%'
                    base_filter.append(
                        or_(
                            func.lower(cast(DiscoveredWallet.source_flags, String)).like(
                                category_flag
                            ),
                            func.lower(cast(DiscoveredWallet.tags, String)).like(
                                category_tag
                            ),
                        )
                    )

            # When a rolling window is active, filter to wallets with trades in that window
            if window_key:
                trade_count_expr = func.json_extract(
                    DiscoveredWallet.rolling_trade_count, f"$.{window_key}"
                )
                base_filter.append(trade_count_expr > 0)

            # Build main query
            query = select(DiscoveredWallet).where(*base_filter)

            # Determine sort expression
            if window_key and sort_by in self.SORT_FIELD_TO_ROLLING_COLUMN:
                # Sort using the rolling window JSON value for this period
                rolling_col_name = self.SORT_FIELD_TO_ROLLING_COLUMN[sort_by]
                rolling_col = getattr(DiscoveredWallet, rolling_col_name, None)
                if rolling_col is not None:
                    sort_expr = func.json_extract(rolling_col, f"$.{window_key}")
                else:
                    sort_expr = DiscoveredWallet.rank_score
            else:
                # All-time: sort by the regular column
                sort_expr = getattr(DiscoveredWallet, sort_by, None)
                if sort_expr is None:
                    sort_expr = DiscoveredWallet.rank_score

            if sort_dir.lower() == "asc":
                query = query.order_by(asc(sort_expr))
            else:
                query = query.order_by(desc(sort_expr))

            if unique_entities_only:
                result = await session.execute(query)
                ordered_wallets = list(result.scalars().all())
                deduped_wallets: list[DiscoveredWallet] = []
                seen_entities: set[str] = set()
                for wallet in ordered_wallets:
                    entity_key = (
                        f"cluster:{wallet.cluster_id}"
                        if wallet.cluster_id
                        else f"wallet:{wallet.address}"
                    )
                    if entity_key in seen_entities:
                        continue
                    seen_entities.add(entity_key)
                    deduped_wallets.append(wallet)

                total_count = len(deduped_wallets)
                wallets = deduped_wallets[offset : offset + limit]
            else:
                # Get total count for pagination
                count_query = select(func.count(DiscoveredWallet.address)).where(
                    *base_filter
                )
                count_result = await session.execute(count_query)
                total_count = count_result.scalar() or 0

                query = query.offset(offset).limit(limit)
                result = await session.execute(query)
                wallets = result.scalars().all()

            rows = []
            for w in wallets:
                row = self._wallet_to_dict(w)

                # Inject period-specific metrics when a rolling window is active
                if window_key:
                    row["period_pnl"] = (w.rolling_pnl or {}).get(window_key, 0.0)
                    row["period_roi"] = (w.rolling_roi or {}).get(window_key, 0.0)
                    row["period_win_rate"] = (w.rolling_win_rate or {}).get(window_key, 0.0)
                    row["period_trades"] = (w.rolling_trade_count or {}).get(window_key, 0)
                    row["period_sharpe"] = (w.rolling_sharpe or {}).get(window_key)

                rows.append(row)

            resp = {"wallets": rows, "total": total_count}
            if window_key:
                resp["window_key"] = window_key
            return resp

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
        Uses a single query with scalar subqueries to avoid 5 round-trips.
        """
        async with AsyncSessionLocal() as session:
            # Single query with 5 scalar subqueries - one round-trip instead of 5
            stats_query = select(
                select(func.count(DiscoveredWallet.address)).scalar_subquery().label(
                    "total"
                ),
                select(func.count(DiscoveredWallet.address))
                .where(DiscoveredWallet.is_profitable == True)  # noqa: E712
                .scalar_subquery()
                .label("profitable"),
                select(func.count(DiscoveredWallet.address))
                .where(DiscoveredWallet.recommendation == "copy_candidate")
                .scalar_subquery()
                .label("copy_candidates"),
                select(func.avg(DiscoveredWallet.rank_score)).scalar_subquery().label(
                    "avg_score"
                ),
                select(func.avg(DiscoveredWallet.win_rate))
                .where(DiscoveredWallet.total_trades >= 10)
                .scalar_subquery()
                .label("avg_win_rate"),
            )
            row = (await session.execute(stats_query)).one()

        total_wallets = row.total or 0
        profitable_count = row.profitable or 0
        copy_candidates = row.copy_candidates or 0
        avg_rank_score = row.avg_score or 0.0
        avg_win_rate = row.avg_win_rate or 0.0

        return {
            "total_discovered": total_wallets,
            "total_profitable": profitable_count,
            "total_copy_candidates": copy_candidates,
            "avg_rank_score": round(avg_rank_score, 4),
            "avg_win_rate": round(avg_win_rate, 4),
            "last_run_at": self._last_run_at.isoformat() if self._last_run_at else None,
            "wallets_discovered_last_run": self._wallets_discovered_last_run,
            "wallets_analyzed_last_run": self._wallets_analyzed_last_run,
            "wallets_seeded_last_run": self._wallets_seeded_last_run,
            "wallets_pruned_last_run": self._wallets_pruned_last_run,
            "is_running": self._running,
        }

    # ------------------------------------------------------------------
    # Internal Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _wallet_to_dict(w: DiscoveredWallet) -> dict:
        """Serialize a DiscoveredWallet ORM object to a plain dict."""
        source_flags = w.source_flags or {}
        if not isinstance(source_flags, dict):
            source_flags = {}
        market_categories = [
            key.replace("leaderboard_category_", "")
            for key, enabled in source_flags.items()
            if key.startswith("leaderboard_category_") and bool(enabled)
        ]

        return {
            "address": w.address,
            "username": w.username,
            "discovered_at": w.discovered_at.isoformat() if w.discovered_at else None,
            "last_analyzed_at": w.last_analyzed_at.isoformat()
            if w.last_analyzed_at
            else None,
            "discovery_source": w.discovery_source,
            "metrics_source_version": w.metrics_source_version,
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
            "quality_score": w.quality_score,
            "activity_score": w.activity_score,
            "stability_score": w.stability_score,
            "composite_score": w.composite_score,
            "last_trade_at": w.last_trade_at.isoformat() if w.last_trade_at else None,
            "trades_1h": w.trades_1h,
            "trades_24h": w.trades_24h,
            "unique_markets_24h": w.unique_markets_24h,
            "in_top_pool": w.in_top_pool,
            "pool_tier": w.pool_tier,
            "pool_membership_reason": w.pool_membership_reason,
            "source_flags": source_flags,
            "market_categories": sorted(set(market_categories)),
            # Clustering
            "cluster_id": w.cluster_id,
            # Insider detection
            "insider_score": w.insider_score or 0.0,
            "insider_confidence": w.insider_confidence or 0.0,
            "insider_sample_size": w.insider_sample_size or 0,
            "insider_last_scored_at": (
                w.insider_last_scored_at.isoformat()
                if w.insider_last_scored_at
                else None
            ),
            "insider_metrics": w.insider_metrics_json,
            "insider_reasons": w.insider_reasons_json or [],
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
                return utcfromtimestamp(raw)
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

    def _filter_trades_after(self, trades: list[dict], cutoff: datetime) -> list[dict]:
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
