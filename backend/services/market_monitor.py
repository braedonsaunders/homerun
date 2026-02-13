"""
Market Monitor - New Market Detection & First-Mover Arbitrage

Monitors Polymarket (and optionally Kalshi) for newly created markets,
price dislocations, and thin order books. The most profitable arbitrage
windows occur in the first minutes after a market is created, before
prices equilibrate across platforms and liquidity deepens.

Key capabilities:
  - Track all known market IDs with first-seen timestamps
  - Detect brand-new markets and events as they appear
  - Flag price dislocations (yes + no far from 1.0) on fresh markets
  - Identify thin-book markets where limit orders can fill cheaply
  - Predict recurring BTC/ETH market creation schedules
  - Score price stability to gauge remaining arb window
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Literal, Optional

from config import settings
from models import Event, Market
from utils.logger import get_logger

logger = get_logger("market_monitor")

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

# How long a market is considered "new" after first detection
NEW_MARKET_WINDOW_SECONDS: int = 300  # 5 minutes

# Markets below this USD liquidity are flagged as thin-book
THIN_BOOK_LIQUIDITY_THRESHOLD: float = 500.0

# Spread thresholds for price dislocation detection
# spread = sum(outcome_prices) - 1.0  (for a binary market, should be ~0)
DISLOCATION_SPREAD_HIGH: float = 0.05  # >= 5 cents off parity -> high urgency
DISLOCATION_SPREAD_MEDIUM: float = 0.03  # >= 3 cents -> medium
DISLOCATION_SPREAD_LOW: float = 0.015  # >= 1.5 cents -> low

# How long to retain entries in the known-market registry before cleanup
REGISTRY_RETENTION_SECONDS: int = 86400  # 24 hours

# Number of price snapshots to retain per market for stability scoring
MAX_PRICE_HISTORY: int = 60

# Polymarket fee applied to winner payouts
POLYMARKET_FEE: float = settings.POLYMARKET_FEE

# BTC/ETH recurring market schedules (Polymarket patterns)
BTC_15MIN_INTERVAL = timedelta(minutes=15)
BTC_1HR_INTERVAL = timedelta(hours=1)
ETH_15MIN_INTERVAL = timedelta(minutes=15)
ETH_1HR_INTERVAL = timedelta(hours=1)

# Keywords used to identify crypto-interval markets
_BTC_KEYWORDS = re.compile(
    r"\bBTC\b|\bBitcoin\b",
    re.IGNORECASE,
)
_ETH_KEYWORDS = re.compile(
    r"\bETH\b|\bEthereum\b",
    re.IGNORECASE,
)
_15MIN_KEYWORDS = re.compile(r"15[\s-]?min", re.IGNORECASE)
_1HR_KEYWORDS = re.compile(r"1[\s-]?h(ou)?r", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


class AlertType(str, Enum):
    NEW_MARKET = "new_market"
    NEW_EVENT = "new_event"
    PRICE_DISLOCATION = "price_dislocation"
    THIN_BOOK = "thin_book"


class Urgency(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class MarketSnapshot:
    """Point-in-time record of a market's state when first observed."""

    market_id: str
    first_seen_at: datetime
    initial_prices: list[float]
    current_prices: list[float]
    price_stability_score: float = 1.0  # 0 = very unstable, 1 = equilibrated
    is_new: bool = True
    has_thin_book: bool = False

    # Internal: rolling price history for stability computation
    _price_history: list[tuple[datetime, list[float]]] = field(default_factory=list, repr=False)

    def age_seconds(self, now: Optional[datetime] = None) -> float:
        """Seconds elapsed since first detection."""
        now = now or datetime.now(timezone.utc)
        first = self.first_seen_at
        if first.tzinfo is None:
            first = first.replace(tzinfo=timezone.utc)
        return (now - first).total_seconds()

    def update_prices(
        self,
        prices: list[float],
        now: Optional[datetime] = None,
        new_window_seconds: int = NEW_MARKET_WINDOW_SECONDS,
    ) -> None:
        """Record a new price observation and recompute derived fields."""
        now = now or datetime.now(timezone.utc)
        self.current_prices = prices
        self._price_history.append((now, list(prices)))

        # Trim history to bounded size
        if len(self._price_history) > MAX_PRICE_HISTORY:
            self._price_history = self._price_history[-MAX_PRICE_HISTORY:]

        self.is_new = self.age_seconds(now) <= new_window_seconds
        self.price_stability_score = self._compute_stability()

    def _compute_stability(self) -> float:
        """
        Price stability score in [0, 1].

        Computed as 1 - normalised average absolute price change across
        consecutive observations. A market whose prices never move gets
        a score of 1.0; one with large swings approaches 0.0.

        When fewer than 2 observations exist we return 0.5 (unknown).
        """
        history = self._price_history
        if len(history) < 2:
            return 0.5  # Not enough data to judge

        total_delta = 0.0
        comparisons = 0
        for i in range(1, len(history)):
            prev_prices = history[i - 1][1]
            curr_prices = history[i][1]
            for j in range(min(len(prev_prices), len(curr_prices))):
                total_delta += abs(curr_prices[j] - prev_prices[j])
                comparisons += 1

        if comparisons == 0:
            return 0.5

        avg_delta = total_delta / comparisons
        # Normalise: a 10-cent average swing per observation maps to score ~0
        # Clamp to [0, 1]
        score = max(0.0, 1.0 - (avg_delta / 0.10))
        return round(score, 4)


@dataclass
class NewMarketAlert:
    """An actionable alert about a market that may contain first-mover arb."""

    market: Market
    event: Optional[Event]
    alert_type: Literal["new_market", "new_event", "price_dislocation", "thin_book"]
    detected_at: datetime
    initial_spread: float  # sum(outcome_prices) - 1.0; 0 = perfectly priced
    estimated_arb_profit: float  # rough profit estimate in USD-equivalent cents
    urgency: Literal["high", "medium", "low"]

    # Optional enrichment
    snapshot: Optional[MarketSnapshot] = None

    def to_dict(self) -> dict:
        return {
            "market_id": self.market.id,
            "question": self.market.question,
            "slug": self.market.slug,
            "event_title": self.event.title if self.event else None,
            "alert_type": self.alert_type,
            "detected_at": self.detected_at.isoformat(),
            "initial_spread": round(self.initial_spread, 6),
            "estimated_arb_profit": round(self.estimated_arb_profit, 6),
            "urgency": self.urgency,
            "outcome_prices": self.market.outcome_prices,
            "liquidity": self.market.liquidity,
            "volume": self.market.volume,
            "stability_score": (self.snapshot.price_stability_score if self.snapshot else None),
        }


@dataclass
class CryptoMarketSchedule:
    """Tracks the cadence of recurring crypto-interval markets."""

    asset: str  # "BTC" or "ETH"
    interval: timedelta
    last_seen_creation: Optional[datetime] = None

    def predict_next(self, now: Optional[datetime] = None) -> Optional[datetime]:
        """
        Estimate when the next market of this type will be created.

        If we have never observed one, return None. Otherwise, step forward
        from the last seen creation time by the interval until we exceed *now*.
        """
        if self.last_seen_creation is None:
            return None
        now = now or datetime.now(timezone.utc)
        last = self.last_seen_creation
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        # Walk forward from last seen
        next_time = last + self.interval
        while next_time <= now:
            next_time += self.interval
        return next_time


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------


class MarketMonitor:
    """
    Watches for newly created Polymarket markets and flags first-mover
    arbitrage opportunities.

    Usage:
        alerts = await market_monitor.get_fresh_opportunities()
        for alert in alerts:
            print(alert.to_dict())
    """

    def __init__(
        self,
        new_market_window_seconds: int = NEW_MARKET_WINDOW_SECONDS,
        thin_book_threshold: float = THIN_BOOK_LIQUIDITY_THRESHOLD,
        registry_retention_seconds: int = REGISTRY_RETENTION_SECONDS,
    ):
        # Config
        self._new_window = new_market_window_seconds
        self._thin_threshold = thin_book_threshold
        self._retention = registry_retention_seconds

        # Known-market registry: market_id -> MarketSnapshot
        self._registry: dict[str, MarketSnapshot] = {}

        # Known-event registry: event_id -> first_seen_at
        self._known_events: dict[str, datetime] = {}

        # Crypto-interval schedule trackers
        self._btc_15m = CryptoMarketSchedule("BTC", BTC_15MIN_INTERVAL)
        self._btc_1h = CryptoMarketSchedule("BTC", BTC_1HR_INTERVAL)
        self._eth_15m = CryptoMarketSchedule("ETH", ETH_15MIN_INTERVAL)
        self._eth_1h = CryptoMarketSchedule("ETH", ETH_1HR_INTERVAL)

        # Lock for concurrent access to the registry
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def get_fresh_opportunities(self) -> list[NewMarketAlert]:
        """
        Main entry point. Fetches current markets and events from
        Polymarket, compares against the known registry, and returns
        alerts sorted by urgency (high first).
        """
        try:
            from services.polymarket import polymarket_client

            events, markets = await asyncio.gather(
                polymarket_client.get_all_events(closed=False),
                polymarket_client.get_all_markets(active=True),
            )
        except Exception as exc:
            logger.error("Failed to fetch markets/events from Polymarket", error=str(exc))
            return []

        # Build a market_id -> Event lookup for enrichment
        event_by_market: dict[str, Event] = {}
        all_event_markets: list[Market] = []
        for event in events:
            for m in event.markets:
                event_by_market[m.id] = event
                all_event_markets.append(m)

        # Merge standalone markets with event-embedded markets
        seen_ids: set[str] = set()
        combined: list[Market] = []
        for m in all_event_markets + markets:
            if m.id not in seen_ids:
                combined.append(m)
                seen_ids.add(m.id)

        alerts: list[NewMarketAlert] = []

        async with self._lock:
            new_market_alerts = self._detect_new_markets(combined, event_by_market)
            new_event_alerts = self._detect_new_events(events, combined)
            dislocation_alerts = self._detect_price_dislocations(combined, event_by_market)
            thin_book_alerts = self._detect_thin_books(combined, event_by_market)

            alerts.extend(new_market_alerts)
            alerts.extend(new_event_alerts)
            alerts.extend(dislocation_alerts)
            alerts.extend(thin_book_alerts)

            # Update existing snapshots with latest prices
            self._update_snapshots(combined)

            # Detect recurring crypto markets
            self._update_crypto_schedules(combined)

            # Housekeeping
            self._cleanup_old_entries()

        # De-duplicate: keep highest-urgency alert per market
        alerts = self._deduplicate(alerts)

        # Sort by urgency (high > medium > low), then by estimated profit desc
        urgency_rank = {"high": 0, "medium": 1, "low": 2}
        alerts.sort(
            key=lambda a: (
                urgency_rank.get(a.urgency, 3),
                -a.estimated_arb_profit,
            )
        )

        if alerts:
            logger.info(
                "Fresh opportunities detected",
                total=len(alerts),
                high=sum(1 for a in alerts if a.urgency == "high"),
                medium=sum(1 for a in alerts if a.urgency == "medium"),
                low=sum(1 for a in alerts if a.urgency == "low"),
            )

        return alerts

    # ------------------------------------------------------------------
    # Detection methods
    # ------------------------------------------------------------------

    def _detect_new_markets(
        self,
        markets: list[Market],
        event_by_market: dict[str, Event],
    ) -> list[NewMarketAlert]:
        """
        Compare incoming markets against the registry. Any market_id not
        previously seen is new. Register it and return alerts.
        """
        now = datetime.now(timezone.utc)
        alerts: list[NewMarketAlert] = []

        for market in markets:
            if market.id in self._registry:
                continue

            # New market -- register it
            snapshot = MarketSnapshot(
                market_id=market.id,
                first_seen_at=now,
                initial_prices=list(market.outcome_prices),
                current_prices=list(market.outcome_prices),
                is_new=True,
                has_thin_book=market.liquidity < self._thin_threshold,
            )
            snapshot._price_history = [(now, list(market.outcome_prices))]
            self._registry[market.id] = snapshot

            spread = self._compute_spread(market)
            profit = self._estimate_arb_profit(spread)
            urgency = self._classify_urgency(spread, market.liquidity)

            alert = NewMarketAlert(
                market=market,
                event=event_by_market.get(market.id),
                alert_type="new_market",
                detected_at=now,
                initial_spread=spread,
                estimated_arb_profit=profit,
                urgency=urgency,
                snapshot=snapshot,
            )
            alerts.append(alert)

            logger.info(
                "New market detected",
                market_id=market.id,
                question=market.question[:80],
                spread=round(spread, 4),
                liquidity=market.liquidity,
                urgency=urgency,
            )

        return alerts

    def _detect_new_events(
        self,
        events: list[Event],
        all_markets: list[Market],
    ) -> list[NewMarketAlert]:
        """
        Detect events that have never been seen before. An event alert is
        distinct from individual market alerts because a new event often
        signals a whole cluster of fresh markets.
        """
        now = datetime.now(timezone.utc)
        alerts: list[NewMarketAlert] = []

        for event in events:
            if event.id in self._known_events:
                continue

            self._known_events[event.id] = now

            # Pick the event's market with the widest spread as representative
            best_market: Optional[Market] = None
            best_spread: float = 0.0
            for m in event.markets:
                s = abs(self._compute_spread(m))
                if s > best_spread or best_market is None:
                    best_spread = s
                    best_market = m

            if best_market is None:
                continue

            spread = self._compute_spread(best_market)
            profit = self._estimate_arb_profit(spread)
            urgency = self._classify_urgency(spread, best_market.liquidity)

            # New events with multiple markets are always at least medium urgency
            if len(event.markets) >= 3 and urgency == "low":
                urgency = "medium"

            alert = NewMarketAlert(
                market=best_market,
                event=event,
                alert_type="new_event",
                detected_at=now,
                initial_spread=spread,
                estimated_arb_profit=profit,
                urgency=urgency,
                snapshot=self._registry.get(best_market.id),
            )
            alerts.append(alert)

            logger.info(
                "New event detected",
                event_id=event.id,
                title=event.title[:80],
                market_count=len(event.markets),
                urgency=urgency,
            )

        return alerts

    def _detect_price_dislocations(
        self,
        markets: list[Market],
        event_by_market: dict[str, Event],
    ) -> list[NewMarketAlert]:
        """
        Find markets whose outcome prices sum to something far from 1.0.
        This indicates a temporary inefficiency where arbitrage profit is
        available.

        Only checks markets that are already in the registry (to avoid
        double-counting brand-new markets, which get their own alerts).
        Focus on markets that are still within the "new" window or have
        low stability scores, as those are most actionable.
        """
        now = datetime.now(timezone.utc)
        alerts: list[NewMarketAlert] = []

        for market in markets:
            snapshot = self._registry.get(market.id)
            if snapshot is None:
                continue  # Will be picked up by _detect_new_markets

            spread = self._compute_spread(market)
            abs_spread = abs(spread)

            if abs_spread < DISLOCATION_SPREAD_LOW:
                continue  # Within normal bounds

            # Only flag if the market is relatively fresh or unstable
            age = snapshot.age_seconds(now)
            if age > self._new_window * 6 and snapshot.price_stability_score > 0.8:
                continue  # Old and stable, not interesting

            profit = self._estimate_arb_profit(spread)
            urgency = self._classify_urgency(spread, market.liquidity)

            alert = NewMarketAlert(
                market=market,
                event=event_by_market.get(market.id),
                alert_type="price_dislocation",
                detected_at=now,
                initial_spread=spread,
                estimated_arb_profit=profit,
                urgency=urgency,
                snapshot=snapshot,
            )
            alerts.append(alert)

            logger.debug(
                "Price dislocation detected",
                market_id=market.id,
                spread=round(spread, 4),
                stability=snapshot.price_stability_score,
                age_seconds=round(age, 1),
            )

        return alerts

    def _detect_thin_books(
        self,
        markets: list[Market],
        event_by_market: dict[str, Event],
    ) -> list[NewMarketAlert]:
        """
        Identify active markets with very low liquidity. These are
        opportunities where a well-placed limit order can fill at
        advantageous prices before the book fills in.
        """
        now = datetime.now(timezone.utc)
        alerts: list[NewMarketAlert] = []

        for market in markets:
            if not market.active or market.closed:
                continue
            if market.liquidity >= self._thin_threshold:
                continue

            # Skip markets with zero prices (not yet initialised)
            if not market.outcome_prices or all(p == 0.0 for p in market.outcome_prices):
                continue

            snapshot = self._registry.get(market.id)
            if snapshot is not None:
                snapshot.has_thin_book = True

            spread = self._compute_spread(market)
            profit = self._estimate_arb_profit(spread)

            # Thin books are medium urgency by default; high if spread is also wide
            urgency: Literal["high", "medium", "low"] = "medium"
            if abs(spread) >= DISLOCATION_SPREAD_HIGH:
                urgency = "high"
            elif abs(spread) < DISLOCATION_SPREAD_LOW:
                urgency = "low"

            alert = NewMarketAlert(
                market=market,
                event=event_by_market.get(market.id),
                alert_type="thin_book",
                detected_at=now,
                initial_spread=spread,
                estimated_arb_profit=profit,
                urgency=urgency,
                snapshot=snapshot,
            )
            alerts.append(alert)

        if alerts:
            logger.debug(
                "Thin-book markets detected",
                count=len(alerts),
            )

        return alerts

    # ------------------------------------------------------------------
    # Snapshot maintenance
    # ------------------------------------------------------------------

    def _update_snapshots(self, markets: list[Market]) -> None:
        """Update price history for all markets already in the registry."""
        now = datetime.now(timezone.utc)
        for market in markets:
            snapshot = self._registry.get(market.id)
            if snapshot is None:
                continue
            snapshot.update_prices(
                market.outcome_prices,
                now=now,
                new_window_seconds=self._new_window,
            )
            snapshot.has_thin_book = market.liquidity < self._thin_threshold

    # ------------------------------------------------------------------
    # Crypto recurring market detection
    # ------------------------------------------------------------------

    def _update_crypto_schedules(self, markets: list[Market]) -> None:
        """
        Scan newly registered markets for BTC/ETH interval patterns and
        update the schedule trackers so we can predict the next creation.
        """
        now = datetime.now(timezone.utc)
        for market in markets:
            snapshot = self._registry.get(market.id)
            if snapshot is None:
                continue
            # Only consider recently registered markets
            if snapshot.age_seconds(now) > 120:
                continue

            question = market.question
            is_btc = bool(_BTC_KEYWORDS.search(question))
            is_eth = bool(_ETH_KEYWORDS.search(question))
            is_15m = bool(_15MIN_KEYWORDS.search(question))
            is_1h = bool(_1HR_KEYWORDS.search(question))

            if is_btc and is_15m:
                self._btc_15m.last_seen_creation = now
            elif is_btc and is_1h:
                self._btc_1h.last_seen_creation = now
            elif is_eth and is_15m:
                self._eth_15m.last_seen_creation = now
            elif is_eth and is_1h:
                self._eth_1h.last_seen_creation = now

    def predict_next_btc_market(self) -> dict[str, Optional[datetime]]:
        """
        Return predicted creation times for the next BTC 15-min and 1-hr
        markets based on observed cadence.

        Returns:
            {"15m": datetime | None, "1h": datetime | None}
        """
        now = datetime.now(timezone.utc)
        return {
            "15m": self._btc_15m.predict_next(now),
            "1h": self._btc_1h.predict_next(now),
        }

    def predict_next_eth_market(self) -> dict[str, Optional[datetime]]:
        """
        Return predicted creation times for the next ETH 15-min and 1-hr
        markets based on observed cadence.

        Returns:
            {"15m": datetime | None, "1h": datetime | None}
        """
        now = datetime.now(timezone.utc)
        return {
            "15m": self._eth_15m.predict_next(now),
            "1h": self._eth_1h.predict_next(now),
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_old_entries(self) -> int:
        """
        Public wrapper for registry cleanup. Returns the number of entries
        removed.
        """
        return self._cleanup_old_entries()

    def _cleanup_old_entries(self) -> int:
        """
        Remove market snapshots older than the retention period and
        event entries older than 2x the retention period.
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self._retention)
        event_cutoff = now - timedelta(seconds=self._retention * 2)

        stale_market_ids = [
            mid for mid, snap in self._registry.items() if snap.first_seen_at.replace(tzinfo=timezone.utc) < cutoff
        ]
        for mid in stale_market_ids:
            del self._registry[mid]

        stale_event_ids = [
            eid
            for eid, first_seen in self._known_events.items()
            if first_seen.replace(tzinfo=timezone.utc) < event_cutoff
        ]
        for eid in stale_event_ids:
            del self._known_events[eid]

        removed = len(stale_market_ids) + len(stale_event_ids)
        if removed > 0:
            logger.debug(
                "Cleaned up old registry entries",
                markets_removed=len(stale_market_ids),
                events_removed=len(stale_event_ids),
                registry_size=len(self._registry),
            )
        return removed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_spread(market: Market) -> float:
        """
        Compute the raw spread for a market's outcome prices.

        For a binary market: spread = yes_price + no_price - 1.0
        For multi-outcome (neg-risk): spread = sum(all_yes_prices) - 1.0

        A positive spread means the market is over-priced (selling
        opportunity); a negative spread means under-priced (buying
        opportunity). Zero means perfectly priced.
        """
        prices = market.outcome_prices
        if not prices:
            return 0.0
        return sum(prices) - 1.0

    @staticmethod
    def _estimate_arb_profit(spread: float) -> float:
        """
        Rough estimate of arbitrage profit per $1 of capital deployed,
        accounting for Polymarket's winner fee.

        If the spread is positive (overpriced), an arber can sell all
        outcomes for >$1 total, guaranteeing profit equal to spread minus
        fees. If negative (underpriced), an arber buys all outcomes for
        <$1 and redeems for $1.

        Returns profit as a fraction (e.g. 0.03 = 3 cents per dollar).
        """
        abs_spread = abs(spread)
        if abs_spread < 0.001:
            return 0.0

        # Gross profit is the absolute spread
        # Fee is charged on the winning side's $1 redemption
        fee = POLYMARKET_FEE  # Applied to the $1 payout
        net_profit = abs_spread - fee

        # Only profitable if spread exceeds the fee
        return max(0.0, net_profit)

    @staticmethod
    def _classify_urgency(
        spread: float,
        liquidity: float,
    ) -> Literal["high", "medium", "low"]:
        """
        Classify alert urgency based on spread magnitude and liquidity.

        High urgency: large spread, especially on low-liquidity markets
        (opportunity may vanish quickly).
        """
        abs_spread = abs(spread)

        if abs_spread >= DISLOCATION_SPREAD_HIGH:
            return "high"

        if abs_spread >= DISLOCATION_SPREAD_MEDIUM:
            # Large spread on a thin book is high urgency
            if liquidity < THIN_BOOK_LIQUIDITY_THRESHOLD:
                return "high"
            return "medium"

        if abs_spread >= DISLOCATION_SPREAD_LOW:
            return "low"

        # Below minimum dislocation, but might still be interesting
        # for new-market or thin-book alerts
        return "low"

    @staticmethod
    def _deduplicate(alerts: list[NewMarketAlert]) -> list[NewMarketAlert]:
        """
        Keep only the highest-priority alert per market_id.
        Priority order: new_event > new_market > price_dislocation > thin_book.
        Within the same type, keep the one with higher urgency.
        """
        type_priority = {
            "new_event": 0,
            "new_market": 1,
            "price_dislocation": 2,
            "thin_book": 3,
        }
        urgency_priority = {"high": 0, "medium": 1, "low": 2}

        best: dict[str, NewMarketAlert] = {}
        for alert in alerts:
            mid = alert.market.id
            existing = best.get(mid)
            if existing is None:
                best[mid] = alert
                continue

            # Compare: lower priority number = higher priority
            existing_rank = (
                type_priority.get(existing.alert_type, 9),
                urgency_priority.get(existing.urgency, 9),
            )
            new_rank = (
                type_priority.get(alert.alert_type, 9),
                urgency_priority.get(alert.urgency, 9),
            )
            if new_rank < existing_rank:
                best[mid] = alert

        return list(best.values())

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_registry_stats(self) -> dict:
        """Return summary statistics about the current registry state."""
        now = datetime.now(timezone.utc)
        new_count = sum(1 for s in self._registry.values() if s.age_seconds(now) <= self._new_window)
        thin_count = sum(1 for s in self._registry.values() if s.has_thin_book)
        unstable_count = sum(1 for s in self._registry.values() if s.price_stability_score < 0.5)
        return {
            "total_tracked": len(self._registry),
            "total_events": len(self._known_events),
            "new_markets": new_count,
            "thin_book_markets": thin_count,
            "unstable_markets": unstable_count,
            "btc_next_15m": (self._btc_15m.predict_next(now).isoformat() if self._btc_15m.predict_next(now) else None),
            "btc_next_1h": (self._btc_1h.predict_next(now).isoformat() if self._btc_1h.predict_next(now) else None),
            "eth_next_15m": (self._eth_15m.predict_next(now).isoformat() if self._eth_15m.predict_next(now) else None),
            "eth_next_1h": (self._eth_1h.predict_next(now).isoformat() if self._eth_1h.predict_next(now) else None),
        }

    def get_snapshot(self, market_id: str) -> Optional[MarketSnapshot]:
        """Retrieve the snapshot for a specific market, if tracked."""
        return self._registry.get(market_id)

    def get_new_markets(self) -> list[MarketSnapshot]:
        """Return all snapshots that are still within the 'new' window."""
        now = datetime.now(timezone.utc)
        return [s for s in self._registry.values() if s.age_seconds(now) <= self._new_window]

    def get_high_priority_market_ids(self) -> set[str]:
        """Return IDs of markets that currently warrant elevated scan priority.

        A market is high-priority if any of:
          - It's still within the "new market" window (< 5 min)
          - It has low price stability (< 0.5)
          - It has a thin order book
        """
        now = datetime.now(timezone.utc)
        priority_ids: set[str] = set()

        for market_id, snapshot in self._registry.items():
            # New market
            if snapshot.age_seconds(now) <= self._new_window:
                priority_ids.add(market_id)
                continue

            # Unstable prices
            if snapshot.price_stability_score < 0.5:
                priority_ids.add(market_id)
                continue

            # Thin order book
            if snapshot.has_thin_book:
                priority_ids.add(market_id)

        return priority_ids

    def get_priority_signals(self) -> dict:
        """Return a summary of priority signals for scanner integration.

        Includes counts, predicted crypto times, and the full set of
        high-priority market IDs.
        """
        now = datetime.now(timezone.utc)
        hp_ids = self.get_high_priority_market_ids()

        new_count = sum(1 for s in self._registry.values() if s.age_seconds(now) <= self._new_window)
        unstable_count = sum(1 for s in self._registry.values() if s.price_stability_score < 0.5)
        thin_count = sum(1 for s in self._registry.values() if s.has_thin_book)

        # Predicted crypto creation times
        btc_predictions = self.predict_next_btc_market()
        eth_predictions = self.predict_next_eth_market()

        # Check if any crypto creation is imminent
        crypto_imminent = False
        for pred_time in list(btc_predictions.values()) + list(eth_predictions.values()):
            if pred_time is not None:
                seconds_until = (pred_time - now).total_seconds()
                if 0 <= seconds_until <= 60:
                    crypto_imminent = True
                    break

        return {
            "high_priority_count": len(hp_ids),
            "high_priority_ids": hp_ids,
            "new_market_count": new_count,
            "unstable_count": unstable_count,
            "thin_book_count": thin_count,
            "crypto_imminent": crypto_imminent,
            "btc_predictions": {k: v.isoformat() if v else None for k, v in btc_predictions.items()},
            "eth_predictions": {k: v.isoformat() if v else None for k, v in eth_predictions.items()},
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

market_monitor = MarketMonitor()
