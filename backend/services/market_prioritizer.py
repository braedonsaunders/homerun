"""
Market Prioritizer - Intelligent Tiered Scanning & Change Detection

Replaces the naive "fetch everything every 60s" approach with a tiered
polling system that allocates scan budget based on market characteristics:

  HOT tier  (poll every ~15s): New markets (<5min), unstable prices,
            thin books, crypto markets near predicted creation, markets
            with active MarketMonitor alerts.
  WARM tier (poll every ~60s): Moderate activity, recent price movement,
            medium liquidity.
  COLD tier (poll every ~180s): Stable, well-established markets with no
            recent price changes.

Also provides:
  - Change-detection via price fingerprinting (skip unchanged markets)
  - Volume/liquidity-weighted attention scoring
  - Integration with MarketMonitor alerts and CryptoMarketSchedule predictions
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

from config import settings
from models import Market
from utils.logger import get_logger

logger = get_logger("market_prioritizer")


class MarketTier(str, Enum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


@dataclass
class MarketPriorityState:
    """Per-market tracking state for prioritization decisions."""

    market_id: str
    tier: MarketTier = MarketTier.WARM
    first_seen_at: Optional[datetime] = None
    last_price_fingerprint: str = ""
    last_evaluated_at: Optional[datetime] = None
    last_price_change_at: Optional[datetime] = None
    consecutive_unchanged_cycles: int = 0
    attention_score: float = 0.5  # 0 = ignore, 1 = highest priority
    has_monitor_alert: bool = False
    liquidity: float = 0.0
    volume: float = 0.0
    price_stability: float = 1.0  # From MarketMonitor snapshot


@dataclass
class TierStats:
    """Summary of the current tier distribution."""

    hot_count: int = 0
    warm_count: int = 0
    cold_count: int = 0
    total_tracked: int = 0
    markets_skipped_unchanged: int = 0


class MarketPrioritizer:
    """
    Central intelligence for deciding which markets to scan and when.

    Maintains per-market state and classifies each market into a tier
    on every update cycle. The scanner uses this to decide:
      1. Which markets to fetch prices for (tier-based)
      2. Which markets to run strategies on (change-detection)
      3. When to trigger fast-path scans (crypto prediction pre-positioning)
    """

    def __init__(self):
        self._states: dict[str, MarketPriorityState] = {}
        self._monitor = None  # Lazy-loaded MarketMonitor reference
        self._last_full_classify: Optional[datetime] = None
        self._stats = TierStats()

    def _get_monitor(self):
        """Lazy-load MarketMonitor to avoid circular imports."""
        if self._monitor is None:
            try:
                from services.market_monitor import market_monitor
                self._monitor = market_monitor
            except ImportError:
                pass
        return self._monitor

    # ------------------------------------------------------------------
    # Price fingerprinting for change detection
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_price_fingerprint(market: Market) -> str:
        """Compute a compact hash of a market's price state.

        Prices are rounded to 3 decimal places (0.1 cent resolution) to
        avoid triggering on insignificant floating-point noise.
        """
        rounded = [round(p, 3) for p in market.outcome_prices]
        raw = f"{market.id}:{rounded}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def has_market_changed(self, market: Market) -> bool:
        """Check if a market's price state has changed since last evaluation.

        Returns True if the market is new or its prices have moved.
        """
        state = self._states.get(market.id)
        if state is None:
            return True  # New market = changed

        new_fp = self._compute_price_fingerprint(market)
        return new_fp != state.last_price_fingerprint

    # ------------------------------------------------------------------
    # Tier classification
    # ------------------------------------------------------------------

    def classify_market(self, market: Market, now: Optional[datetime] = None) -> MarketTier:
        """Classify a single market into a priority tier.

        Classification is based on multiple signals:
          - Age: new markets are HOT
          - Price stability: unstable markets are HOT
          - Liquidity: thin books are HOT/WARM
          - Volume: high volume with price movement = WARM
          - Monitor alerts: any active alert = HOT
          - Crypto schedule: near predicted creation = HOT
          - Change detection: many unchanged cycles = COLD
        """
        now = now or datetime.now(timezone.utc)
        state = self._states.get(market.id)

        if state is None:
            # Brand new market — always HOT
            return MarketTier.HOT

        # Score-based classification
        hot_signals = 0
        warm_signals = 0

        # Signal 1: Age — new markets are the highest priority
        age_seconds = (now - state.first_seen_at).total_seconds() if state.first_seen_at else 0
        if age_seconds <= settings.HOT_TIER_MAX_AGE_SECONDS:
            hot_signals += 2  # Strong signal
        elif age_seconds <= settings.WARM_TIER_MAX_AGE_SECONDS:
            warm_signals += 1

        # Signal 2: Price stability (from MarketMonitor snapshot)
        if state.price_stability < 0.3:
            hot_signals += 2
        elif state.price_stability < 0.6:
            warm_signals += 1

        # Signal 3: Thin book / low liquidity
        if market.liquidity < settings.THIN_BOOK_LIQUIDITY_THRESHOLD:
            hot_signals += 1
        elif market.liquidity < settings.MIN_LIQUIDITY * 2:
            warm_signals += 1

        # Signal 4: MarketMonitor alert active
        if state.has_monitor_alert:
            hot_signals += 2

        # Signal 5: Recent price change (market is active)
        if state.last_price_change_at:
            since_change = (now - state.last_price_change_at).total_seconds()
            if since_change < 60:
                hot_signals += 1
            elif since_change < 300:
                warm_signals += 1

        # Signal 6: Volume-weighted attention
        # High volume markets with any price movement deserve attention
        if market.volume > 50000 and state.consecutive_unchanged_cycles < 3:
            warm_signals += 1
        if market.volume > 200000:
            warm_signals += 1  # Always keep high-volume markets at least warm

        # Signal 7: Consecutive unchanged — promotes demotion to COLD
        if state.consecutive_unchanged_cycles >= settings.COLD_TIER_UNCHANGED_CYCLES:
            # Only demote if no other hot/warm signals
            if hot_signals == 0 and warm_signals <= 1:
                return MarketTier.COLD

        # Final classification
        if hot_signals >= 2:
            return MarketTier.HOT
        elif hot_signals >= 1 or warm_signals >= 2:
            return MarketTier.WARM
        else:
            return MarketTier.COLD

    def classify_all(
        self,
        markets: list[Market],
        now: Optional[datetime] = None,
    ) -> dict[MarketTier, list[Market]]:
        """Classify all markets into tiers. Returns tier -> market list mapping."""
        now = now or datetime.now(timezone.utc)

        # Refresh monitor alerts before classification
        self._refresh_monitor_alerts()
        # Check crypto schedule proximity
        self._check_crypto_schedule_proximity(now)

        result: dict[MarketTier, list[Market]] = {
            MarketTier.HOT: [],
            MarketTier.WARM: [],
            MarketTier.COLD: [],
        }

        for market in markets:
            tier = self.classify_market(market, now)
            result[tier].append(market)

            # Ensure state exists and update tier
            state = self._ensure_state(market, now)
            state.tier = tier

        # Update stats
        self._stats = TierStats(
            hot_count=len(result[MarketTier.HOT]),
            warm_count=len(result[MarketTier.WARM]),
            cold_count=len(result[MarketTier.COLD]),
            total_tracked=len(self._states),
        )
        self._last_full_classify = now

        logger.info(
            "Market classification complete",
            hot=self._stats.hot_count,
            warm=self._stats.warm_count,
            cold=self._stats.cold_count,
        )

        return result

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _ensure_state(self, market: Market, now: datetime) -> MarketPriorityState:
        """Get or create the priority state for a market."""
        state = self._states.get(market.id)
        if state is None:
            state = MarketPriorityState(
                market_id=market.id,
                first_seen_at=now,
                liquidity=market.liquidity,
                volume=market.volume,
            )
            self._states[market.id] = state
        return state

    def update_after_evaluation(
        self,
        markets: list[Market],
        now: Optional[datetime] = None,
    ) -> int:
        """Update state after strategy evaluation. Returns count of unchanged markets.

        Called after each scan cycle to:
          1. Record new price fingerprints
          2. Track consecutive unchanged cycles
          3. Update liquidity/volume metrics
        """
        now = now or datetime.now(timezone.utc)
        unchanged_count = 0

        for market in markets:
            state = self._ensure_state(market, now)
            new_fp = self._compute_price_fingerprint(market)

            if new_fp == state.last_price_fingerprint:
                state.consecutive_unchanged_cycles += 1
                unchanged_count += 1
            else:
                state.consecutive_unchanged_cycles = 0
                state.last_price_change_at = now

            state.last_price_fingerprint = new_fp
            state.last_evaluated_at = now
            state.liquidity = market.liquidity
            state.volume = market.volume

        self._stats.markets_skipped_unchanged = unchanged_count
        return unchanged_count

    def get_changed_markets(self, markets: list[Market]) -> list[Market]:
        """Filter to only markets whose prices have changed since last evaluation.

        This is the core change-detection optimization: skip running all 17
        strategies on a market whose prices haven't moved.
        """
        return [m for m in markets if self.has_market_changed(m)]

    # ------------------------------------------------------------------
    # Monitor & crypto schedule integration
    # ------------------------------------------------------------------

    def _refresh_monitor_alerts(self) -> None:
        """Pull active alert market IDs from MarketMonitor and flag them."""
        monitor = self._get_monitor()
        if monitor is None:
            return

        # Get IDs of markets with active alerts (new, dislocation, thin book)
        try:
            alert_ids = monitor.get_high_priority_market_ids()
        except Exception:
            alert_ids = set()

        for state in self._states.values():
            state.has_monitor_alert = state.market_id in alert_ids

    def _check_crypto_schedule_proximity(self, now: datetime) -> None:
        """Check if we're near a predicted crypto market creation time.

        If within CRYPTO_PREDICTION_WINDOW_SECONDS of a predicted creation,
        flag all crypto-related markets as HOT so we poll them aggressively.
        """
        monitor = self._get_monitor()
        if monitor is None:
            return

        window = timedelta(seconds=settings.CRYPTO_PREDICTION_WINDOW_SECONDS)
        near_creation = False

        try:
            btc_preds = monitor.predict_next_btc_market()
            eth_preds = monitor.predict_next_eth_market()

            for pred_time in list(btc_preds.values()) + list(eth_preds.values()):
                if pred_time is not None:
                    time_until = pred_time - now
                    if timedelta(0) <= time_until <= window:
                        near_creation = True
                        logger.info(
                            "Crypto market creation imminent",
                            seconds_until=time_until.total_seconds(),
                        )
                        break
        except Exception:
            return

        if near_creation:
            # Crypto creation imminence is handled via the monitor's
            # high-priority IDs — the classify_market() method picks up
            # the has_monitor_alert flag set by _refresh_monitor_alerts().
            pass

    def update_stability_scores(self) -> None:
        """Pull price stability scores from MarketMonitor snapshots."""
        monitor = self._get_monitor()
        if monitor is None:
            return

        for market_id, state in self._states.items():
            snapshot = monitor.get_snapshot(market_id)
            if snapshot is not None:
                state.price_stability = snapshot.price_stability_score

    # ------------------------------------------------------------------
    # Volume/liquidity-weighted attention
    # ------------------------------------------------------------------

    def compute_attention_scores(self, markets: list[Market]) -> None:
        """Compute attention scores based on volume and liquidity.

        Markets where arbitrage is actually executable (sufficient liquidity,
        meaningful volume) get higher attention. Markets with very low
        liquidity get deprioritized because slippage eats the edge.
        """
        if not markets:
            return

        # Find max volume and liquidity for normalization
        max_vol = max((m.volume for m in markets), default=1.0) or 1.0
        max_liq = max((m.liquidity for m in markets), default=1.0) or 1.0

        for market in markets:
            state = self._states.get(market.id)
            if state is None:
                continue

            # Normalized volume and liquidity (0-1)
            vol_score = min(market.volume / max_vol, 1.0) if max_vol > 0 else 0.0
            liq_score = min(market.liquidity / max_liq, 1.0) if max_liq > 0 else 0.0

            # Attention = weighted combination
            # Liquidity matters more (we need it to execute)
            # but volume indicates market interest/activity
            base_attention = 0.6 * liq_score + 0.4 * vol_score

            # Boost for recent price changes (active market)
            if state.consecutive_unchanged_cycles == 0:
                base_attention = min(1.0, base_attention + 0.15)

            # Boost for monitor alerts
            if state.has_monitor_alert:
                base_attention = min(1.0, base_attention + 0.25)

            # Penalty for very low liquidity (can't execute anyway)
            if market.liquidity < settings.MIN_LIQUIDITY_HARD:
                base_attention *= 0.3

            state.attention_score = round(base_attention, 4)

    # ------------------------------------------------------------------
    # Query methods for scanner integration
    # ------------------------------------------------------------------

    def get_hot_market_ids(self) -> set[str]:
        """Return IDs of all markets currently in the HOT tier."""
        return {
            mid for mid, state in self._states.items()
            if state.tier == MarketTier.HOT
        }

    def get_markets_needing_eval(
        self,
        markets: list[Market],
        current_tier_filter: Optional[MarketTier] = None,
    ) -> list[Market]:
        """Return markets that need strategy evaluation this cycle.

        Combines tier filtering with change detection:
        - If tier_filter is set, only include markets in that tier (or hotter)
        - Then further filter to only markets whose prices have changed
        - Markets in HOT tier always pass (never skipped by change detection)
        """
        result = []

        tier_rank = {MarketTier.HOT: 0, MarketTier.WARM: 1, MarketTier.COLD: 2}
        filter_rank = tier_rank.get(current_tier_filter, 2) if current_tier_filter else 2

        for market in markets:
            state = self._states.get(market.id)
            if state is None:
                # New market — always evaluate
                result.append(market)
                continue

            # Tier filtering
            market_rank = tier_rank.get(state.tier, 2)
            if market_rank > filter_rank:
                continue

            # Change detection (HOT markets skip this — always evaluated)
            if state.tier != MarketTier.HOT and not self.has_market_changed(market):
                continue

            result.append(market)

        return result

    def should_trigger_fast_scan(self, now: Optional[datetime] = None) -> bool:
        """Check if conditions warrant an immediate fast scan.

        Returns True if:
        - Crypto market creation is imminent (within prediction window)
        - Multiple HOT-tier markets exist with active alerts
        """
        now = now or datetime.now(timezone.utc)

        # Check crypto schedule
        monitor = self._get_monitor()
        if monitor:
            try:
                window = timedelta(seconds=settings.CRYPTO_PREDICTION_WINDOW_SECONDS)
                btc_preds = monitor.predict_next_btc_market()
                eth_preds = monitor.predict_next_eth_market()

                for pred_time in list(btc_preds.values()) + list(eth_preds.values()):
                    if pred_time is not None:
                        time_until = pred_time - now
                        if timedelta(0) <= time_until <= window:
                            return True
            except Exception:
                pass

        # Check if many HOT markets with alerts
        hot_alert_count = sum(
            1 for s in self._states.values()
            if s.tier == MarketTier.HOT and s.has_monitor_alert
        )
        if hot_alert_count >= 3:
            return True

        return False

    # ------------------------------------------------------------------
    # Cleanup and stats
    # ------------------------------------------------------------------

    def cleanup_stale(self, max_age_hours: int = 24) -> int:
        """Remove states for markets not seen in max_age_hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        stale = [
            mid for mid, state in self._states.items()
            if state.last_evaluated_at and state.last_evaluated_at < cutoff
        ]
        for mid in stale:
            del self._states[mid]

        if stale:
            logger.debug("Cleaned up stale market states", count=len(stale))
        return len(stale)

    def get_stats(self) -> dict:
        """Return current prioritizer statistics."""
        hot_ids = [s.market_id for s in self._states.values() if s.tier == MarketTier.HOT]
        return {
            "total_tracked": len(self._states),
            "hot_count": self._stats.hot_count,
            "warm_count": self._stats.warm_count,
            "cold_count": self._stats.cold_count,
            "markets_skipped_unchanged": self._stats.markets_skipped_unchanged,
            "hot_market_ids": hot_ids[:20],  # Cap for API response size
            "last_full_classify": (
                self._last_full_classify.isoformat()
                if self._last_full_classify
                else None
            ),
        }


# Singleton
market_prioritizer = MarketPrioritizer()
