"""
Strategy: BTC/ETH High-Frequency Arbitrage

Specialized strategy for Bitcoin and Ethereum binary markets on Polymarket,
targeting the highly liquid 15-minute and 1-hour "up or down" markets.

These markets are the most liquid arbitrage venue on Polymarket. The "gabagool"
bot reportedly earns ~$58 every 15 minutes by exploiting inefficiencies in
BTC 15-min markets alone.

This strategy uses dynamic sub-strategy selection (Option C):
  A. Pure Arbitrage   -- Buy YES + NO when combined < $1.00
  B. Dump-Hedge       -- Buy the dumped side after a >5% drop, then hedge
  C. Pre-Placed Limits -- Pre-place limit orders at $0.45-$0.47 on new markets

The selector scores each sub-strategy against current market conditions
(price levels, volatility, time to expiry, liquidity, order book state) and
returns opportunities from the best-fitting sub-strategy.
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from models import Market, Event, ArbitrageOpportunity, StrategyType
from config import settings as _cfg
from .base import BaseStrategy
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Question / slug patterns used to identify BTC/ETH high-frequency markets
_ASSET_PATTERNS: dict[str, list[str]] = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
    "SOL": ["solana", "sol"],
    "XRP": ["ripple", "xrp"],
}

_TIMEFRAME_PATTERNS: dict[str, list[str]] = {
    "5min": [
        "updown-5m",
        "5m-",
        "5m",
        "5 min",
        "5-minute",
    ],
    "15min": [
        "updown-15m",  # actual Polymarket slug pattern
        "updown-15m-",  # with trailing timestamp
        "15m-",  # short form in slugs (e.g. "btc…15m-17707…")
        "15 min",
        "15-min",
        "15min",
        "15m",  # bare short form
        "fifteen min",
        "15 minute",
        "15-minute",
        "15 minutes",
        "15-minutes",
        "quarter hour",
        "quarter-hour",
    ],
    "1hr": [
        "updown-1h",  # actual Polymarket slug pattern
        "updown-1h-",  # with trailing timestamp
        "1 hour",
        "1-hour",
        "1hr",
        "1h-",  # short form in slugs
        "1h",  # bare short form
        "one hour",
        "60 min",
        "60-min",
        "60m",  # short form
        "60 minute",
        "60-minute",
        "60 minutes",
        "60-minutes",
        "hourly",
        "next hour",
    ],
}

_DIRECTION_KEYWORDS: list[str] = [
    "up or down",
    "higher or lower",
    "go up",
    "go down",
    "above or below",
    "increase or decrease",
    "up",
    "down",
    "higher",
    "lower",
    "price",
    "beat",
    "price to beat",
]

# Slug regex: matches slugs where asset and timeframe may be separated by
# other words.  Allows "bitcoin-15-minute-up-or-down", "btc-price-15min",
# "ethereum-1-hour-up-down", etc.
_SLUG_REGEX = re.compile(
    r"(btc|eth|sol|xrp|bitcoin|ethereum|solana|ripple)"
    r".*?"  # allow intervening words (non-greedy)
    r"(5[\s_-]?m(?:in(?:ute)?s?)?"
    r"|15[\s_-]?m(?:in(?:ute)?s?)?"  # "15m", "15min", "15-minute", …
    r"|1[\s_-]?h(?:(?:ou)?r)?"  # "1h", "1hr", "1hour", "1-h", …
    r"|60[\s_-]?m(?:in(?:ute)?s?)?"
    r"|quarter[\s_-]?hour|hourly)",
    re.IGNORECASE,
)

# Polymarket crypto series definitions.
# Each series has a unique ID on the Gamma API that returns all active events
# in the series.  Querying /events?series_id=X&active=true&closed=false
# reliably returns the currently-live and upcoming 15-minute markets.
# Series IDs are configurable via the Settings UI (persisted in DB).
def _get_crypto_series() -> list[tuple[str, str, str]]:
    """Return crypto series configs, reading IDs from the live config singleton."""
    return [
        # (series_id, asset, timeframe)
        (_cfg.BTC_ETH_HF_SERIES_BTC_15M, "BTC", "15min"),
        (_cfg.BTC_ETH_HF_SERIES_ETH_15M, "ETH", "15min"),
        (_cfg.BTC_ETH_HF_SERIES_SOL_15M, "SOL", "15min"),
        (_cfg.BTC_ETH_HF_SERIES_XRP_15M, "XRP", "15min"),
    ]


# ---------------------------------------------------------------------------
# Fee curve — official Polymarket taker fee for 15-minute crypto markets
# ---------------------------------------------------------------------------


def polymarket_fee_curve(price: float) -> float:
    """Compute the taker fee per share at a given price.

    Official formula from Polymarket docs:
        fee = price * 0.25 * (price * (1 - price))^2

    At $0.50: fee = 0.50 * 0.25 * (0.50 * 0.50)^2 = $0.0078 → 1.56%
    At $0.10: fee = 0.10 * 0.25 * (0.10 * 0.90)^2 = $0.0002 → 0.20%
    At $0.90: fee = 0.90 * 0.25 * (0.90 * 0.10)^2 = $0.0018 → 0.20%
    """
    p = max(0.0, min(1.0, price))
    return p * 0.25 * (p * (1.0 - p)) ** 2


def polymarket_fee_pct(price: float) -> float:
    """Fee as a fraction of the share price (0.0 – 0.0156)."""
    if price <= 0:
        return 0.0
    return polymarket_fee_curve(price) / price


# Strategy selector thresholds — read from config (persisted in DB via Settings UI)


def _pure_arb_max_combined():
    return _cfg.BTC_ETH_HF_PURE_ARB_MAX_COMBINED


def _dump_hedge_drop_pct():
    return _cfg.BTC_ETH_HF_DUMP_THRESHOLD


def _thin_liquidity_usd():
    return _cfg.BTC_ETH_HF_THIN_LIQUIDITY_USD


# Dump-hedge combined cost target raised from 0.97 to 0.98: after 2% fee,
# you need combined < 0.98 to profit. 0.97 left only 1% gross before fees.
_DUMP_HEDGE_MAX_COMBINED = 0.98  # Combined cost target after dump-hedge
_LIMIT_ORDER_TARGET_LOW = 0.45  # Lower limit order price
_LIMIT_ORDER_TARGET_HIGH = 0.47  # Upper limit order price
_NEW_MARKET_VOLUME_THRESHOLD = 5000.0  # Markets with volume below this are "new"

# Price history defaults
_DEFAULT_HISTORY_WINDOW_SEC = 300  # 5 minutes for 15-min markets
_1HR_HISTORY_WINDOW_SEC = 600  # 10 minutes for 1-hr markets
_MAX_HISTORY_ENTRIES = 200  # Maximum price snapshots per market


# ---------------------------------------------------------------------------
# Gamma API crypto market fetcher
# ---------------------------------------------------------------------------


class _CryptoMarketFetcher:
    """Sync HTTP fetcher that queries Polymarket's Gamma API for crypto markets
    using series_id-based discovery (the same approach used by
    PolymarketBTC15mAssistant and other production bots).

    Each crypto asset/timeframe has a stable ``series_id`` on the Gamma API.
    Querying ``GET /events?series_id=X&active=true&closed=false`` reliably
    returns the currently-live and upcoming 15-minute (or hourly) markets
    with correct ``endDate`` values, real-time ``bestBid``/``bestAsk``
    pricing, CLOB token IDs, and liquidity data.

    Results are cached for ``ttl_seconds`` to avoid hammering the API.
    """

    def __init__(self, gamma_url: str = "", ttl_seconds: int = 15):
        self._gamma_url = gamma_url or _cfg.GAMMA_API_URL
        self._ttl = ttl_seconds
        self._markets: list[Market] = []
        self._last_fetch: float = 0.0

    @property
    def is_stale(self) -> bool:
        return (time.monotonic() - self._last_fetch) > self._ttl

    def get_markets(self) -> list[Market]:
        """Return cached crypto markets, refreshing if stale."""
        if self.is_stale:
            fetched = self._fetch()
            if fetched is not None:
                self._markets = fetched
                self._last_fetch = time.monotonic()
                # Subscribe new market tokens to the WS feed for real-time prices
                self._subscribe_tokens_to_ws(fetched)
        return self._markets

    @staticmethod
    def _subscribe_tokens_to_ws(markets: list[Market]) -> None:
        """Fire-and-forget: subscribe crypto market CLOB tokens to the
        WebSocket price feed so we get real-time bid/ask updates instead
        of relying on stale HTTP polling."""
        import asyncio

        token_ids = []
        for m in markets:
            token_ids.extend(t for t in m.clob_token_ids if len(t) > 20)
        if not token_ids:
            return

        try:
            from services.ws_feeds import get_feed_manager

            feed_mgr = get_feed_manager()
            if not feed_mgr._started:
                return

            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(
                    feed_mgr.polymarket_feed.subscribe(token_ids=token_ids)
                )
            else:
                loop.run_until_complete(
                    feed_mgr.polymarket_feed.subscribe(token_ids=token_ids)
                )
            logger.debug(
                "BtcEthHighFreq: subscribed %d crypto tokens to WS feed",
                len(token_ids),
            )
        except Exception as e:
            logger.debug("BtcEthHighFreq: WS subscription failed (non-critical): %s", e)

    @staticmethod
    def _is_currently_live(event: dict, now_ms: float) -> bool:
        """Check if an event's market window is currently live.

        A 15-minute market is live when:
          startTime <= now < endDate
        The event-level ``startTime`` is when the 15-min window opens;
        ``endDate`` is when it resolves.
        """
        start_str = event.get("startTime") or event.get("startDate")
        end_str = event.get("endDate")
        if not end_str:
            return False

        try:
            end_ms = datetime.fromisoformat(
                end_str.replace("Z", "+00:00")
            ).timestamp() * 1000
        except (ValueError, AttributeError):
            return False

        if now_ms >= end_ms:
            return False  # Already resolved

        if start_str:
            try:
                start_ms = datetime.fromisoformat(
                    start_str.replace("Z", "+00:00")
                ).timestamp() * 1000
                if now_ms < start_ms:
                    return False  # Not started yet
            except (ValueError, AttributeError):
                pass

        return True

    @staticmethod
    def _pick_live_and_upcoming(events: list[dict], max_upcoming: int = 2) -> list[dict]:
        """From a list of events, return the currently-live one plus next upcoming.

        This mirrors the reference bot's ``pickLatestLiveMarket`` logic but
        returns multiple events so we can show upcoming opportunities too.
        """
        now_ms = time.time() * 1000
        live: list[dict] = []
        upcoming: list[dict] = []

        for evt in events:
            if evt.get("closed"):
                continue
            start_str = evt.get("startTime") or evt.get("startDate")
            end_str = evt.get("endDate")
            if not end_str:
                continue
            try:
                end_ms = datetime.fromisoformat(
                    end_str.replace("Z", "+00:00")
                ).timestamp() * 1000
            except (ValueError, AttributeError):
                continue
            if end_ms <= now_ms:
                continue  # Already resolved

            start_ms = None
            if start_str:
                try:
                    start_ms = datetime.fromisoformat(
                        start_str.replace("Z", "+00:00")
                    ).timestamp() * 1000
                except (ValueError, AttributeError):
                    pass

            if start_ms is not None and start_ms <= now_ms:
                live.append((end_ms, evt))
            else:
                upcoming.append((end_ms, evt))

        # Sort by end time (soonest first)
        live.sort(key=lambda x: x[0])
        upcoming.sort(key=lambda x: x[0])

        result = [e for _, e in live]
        result.extend(e for _, e in upcoming[:max_upcoming])
        return result

    def _fetch(self) -> list[Market]:
        """Fetch live crypto markets from Gamma API using series_id.

        For each crypto series (BTC 15m, ETH 15m, etc.), queries the
        events endpoint to get active events, then picks the currently-live
        and next-upcoming markets.
        """
        import httpx

        all_markets: list[Market] = []
        seen_ids: set[str] = set()

        def _market_id(mkt: dict) -> str:
            return str(
                mkt.get("conditionId")
                or mkt.get("condition_id")
                or mkt.get("id", "")
            )

        try:
            series = _get_crypto_series()
            with httpx.Client(timeout=10.0) as client:
                for series_id, asset, timeframe in series:
                    try:
                        resp = client.get(
                            f"{self._gamma_url}/events",
                            params={
                                "series_id": series_id,
                                "active": "true",
                                "closed": "false",
                                "limit": 10,
                            },
                        )
                        if resp.status_code != 200:
                            logger.debug(
                                "BtcEthHighFreq: Gamma series_id=%s returned %s",
                                series_id, resp.status_code,
                            )
                            continue

                        events = resp.json()
                        if not isinstance(events, list):
                            continue

                        # Pick live + upcoming events
                        selected = self._pick_live_and_upcoming(events)

                        for event_data in selected:
                            for mkt_data in event_data.get("markets", []):
                                mid = _market_id(mkt_data)
                                if mid and mid not in seen_ids:
                                    try:
                                        m = Market.from_gamma_response(mkt_data)
                                        all_markets.append(m)
                                        seen_ids.add(mid)
                                    except Exception as e:
                                        logger.debug(
                                            "BtcEthHighFreq: failed to parse market %s: %s",
                                            mid, e,
                                        )

                        time.sleep(0.05)  # Rate limit between series
                    except Exception as e:
                        logger.debug(
                            "BtcEthHighFreq: series_id=%s fetch failed: %s",
                            series_id, e,
                        )

        except Exception as exc:
            logger.warning(
                "Crypto market fetch failed: %s",
                str(exc),
                exc_info=True,
            )

        if all_markets:
            logger.info(
                "BtcEthHighFreq: fetched %d live crypto markets via Gamma series API "
                "(%s)",
                len(all_markets),
                ", ".join(f"{a} {tf}" for _, a, tf in series),
            )
        else:
            logger.debug(
                "BtcEthHighFreq: no live crypto markets found across %d series",
                len(series),
            )
        return all_markets


# Module-level crypto market fetcher (lazy-initialized)
_crypto_fetcher: Optional[_CryptoMarketFetcher] = None


def _get_crypto_fetcher() -> _CryptoMarketFetcher:
    """Get or create the singleton crypto market fetcher."""
    global _crypto_fetcher
    if _crypto_fetcher is None:
        _crypto_fetcher = _CryptoMarketFetcher()
    return _crypto_fetcher


# ---------------------------------------------------------------------------
# Sub-strategy enum
# ---------------------------------------------------------------------------


class SubStrategy(str, Enum):
    PURE_ARB = "pure_arb"
    DUMP_HEDGE = "dump_hedge"
    PRE_PLACED_LIMITS = "pre_placed_limits"
    DIRECTIONAL_EDGE = "directional_edge"


# ---------------------------------------------------------------------------
# Price history tracker
# ---------------------------------------------------------------------------


@dataclass
class PriceSnapshot:
    """A single price observation at a point in time."""

    timestamp: float  # time.monotonic()
    yes_price: float
    no_price: float


@dataclass
class MarketPriceHistory:
    """Rolling window of price snapshots for a single market."""

    window_seconds: float = _DEFAULT_HISTORY_WINDOW_SEC
    snapshots: deque[PriceSnapshot] = field(default_factory=deque)

    def record(self, yes_price: float, no_price: float) -> None:
        """Append a snapshot and evict stale entries."""
        now = time.monotonic()
        self.snapshots.append(
            PriceSnapshot(
                timestamp=now,
                yes_price=yes_price,
                no_price=no_price,
            )
        )
        self._evict(now)

    def _evict(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self.snapshots and self.snapshots[0].timestamp < cutoff:
            self.snapshots.popleft()

    @property
    def has_data(self) -> bool:
        return len(self.snapshots) >= 2

    def max_drop_yes(self) -> float:
        """Return the largest drop (positive value) in YES price over the window."""
        if not self.has_data:
            return 0.0
        peak = max(s.yes_price for s in self.snapshots)
        current = self.snapshots[-1].yes_price
        return max(peak - current, 0.0)

    def max_drop_no(self) -> float:
        """Return the largest drop (positive value) in NO price over the window."""
        if not self.has_data:
            return 0.0
        peak = max(s.no_price for s in self.snapshots)
        current = self.snapshots[-1].no_price
        return max(peak - current, 0.0)

    def recent_volatility(self) -> float:
        """Simple volatility proxy: max price range over the window (YES side)."""
        if not self.has_data:
            return 0.0
        prices = [s.yes_price for s in self.snapshots]
        return max(prices) - min(prices)


# ---------------------------------------------------------------------------
# Candidate detection helper
# ---------------------------------------------------------------------------


@dataclass
class HighFreqCandidate:
    """A market identified as a BTC/ETH high-frequency binary market."""

    market: Market
    asset: str  # "BTC", "ETH", "SOL", or "XRP"
    timeframe: str  # "5min", "15min", or "1hr"
    yes_price: float
    no_price: float


# ---------------------------------------------------------------------------
# Sub-strategy scoring
# ---------------------------------------------------------------------------


@dataclass
class SubStrategyScore:
    """Score and metadata for a candidate sub-strategy."""

    strategy: SubStrategy
    score: float  # Higher is better (0-100 scale)
    reason: str
    params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main strategy class
# ---------------------------------------------------------------------------


class BtcEthHighFreqStrategy(BaseStrategy):
    """
    High-frequency arbitrage strategy for BTC and ETH binary markets.

    Dynamically selects among three sub-strategies based on current market
    conditions:
      A. Pure Arbitrage   -- guaranteed profit when YES + NO < $1.00
      B. Dump-Hedge       -- buy a dumped side, hedge with opposite
      C. Pre-Placed Limits -- limit orders on new/thin markets

    Designed for Polymarket's 15-min and 1-hr BTC/ETH up-or-down markets.
    """

    strategy_type = StrategyType.BTC_ETH_HIGHFREQ
    name = "BTC/ETH High-Frequency"
    description = (
        "Dynamic high-frequency arbitrage on BTC/ETH 15-min and 1-hr binary markets"
    )

    def __init__(self) -> None:
        super().__init__()
        # 15-minute crypto markets have taker-only fees using a price-curve:
        #   fee_per_share = price * 0.25 * (price * (1 - price))^2
        # At 50% (where up/down markets sit), this is ~1.56%.
        # We set self.fee to the midpoint estimate; the scoring methods
        # use polymarket_fee_curve() for price-specific calculations.
        # See: https://docs.polymarket.com/polymarket-learn/trading/maker-rebates-program
        self.fee = 0.0156  # ~1.56% at 50% probability
        # Per-market price history keyed by market ID
        self._price_histories: dict[str, MarketPriceHistory] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[ArbitrageOpportunity]:
        """Detect arbitrage opportunities across BTC/ETH high-freq markets.

        1. Filter markets to find BTC/ETH high-freq candidates.
        2. Update price history for each candidate.
        3. Run the dynamic strategy selector on each candidate.
        4. Return all detected opportunities.
        """
        opportunities: list[ArbitrageOpportunity] = []

        candidates = self._find_candidates(markets, prices)
        if not candidates:
            logger.debug("BtcEthHighFreq: no BTC/ETH high-freq candidates found")
            return opportunities

        logger.info(
            f"BtcEthHighFreq: found {len(candidates)} candidate market(s) — evaluating sub-strategies"
        )

        for candidate in candidates:
            # Update price history
            self._update_price_history(candidate)

            # Dynamic strategy selection
            selected, all_scores = self._select_sub_strategy(candidate)
            if selected is None:
                reasons = " | ".join(
                    f"{s.strategy.value}: {s.reason}" for s in all_scores
                )
                logger.debug(
                    f"BtcEthHighFreq: no viable sub-strategy for market "
                    f"{candidate.market.id} ({candidate.asset} {candidate.timeframe}, "
                    f"yes={candidate.yes_price:.3f} no={candidate.no_price:.3f} "
                    f"liq=${candidate.market.liquidity:.0f}) — {reasons}"
                )
                continue

            scores_str = ", ".join(
                f"{s.strategy.value}={s.score:.1f}" for s in all_scores
            )
            logger.info(
                f"BtcEthHighFreq: market {candidate.market.id} "
                f"({candidate.asset} {candidate.timeframe}) — "
                f"selected {selected.strategy.value} (score={selected.score:.1f}). "
                f"All scores: {scores_str}"
            )

            # Generate opportunity from the selected sub-strategy
            opp = self._generate_opportunity(candidate, selected)
            if opp is not None:
                opportunities.append(opp)
                logger.info(
                    f"BtcEthHighFreq: opportunity detected — {opp.title} | "
                    f"ROI {opp.roi_percent:.2f}% | sub-strategy={selected.strategy.value} | "
                    f"market={candidate.market.id}"
                )
            else:
                logger.debug(
                    f"BtcEthHighFreq: create_opportunity rejected market "
                    f"{candidate.market.id} ({candidate.asset} {candidate.timeframe}, "
                    f"sub={selected.strategy.value}, score={selected.score:.1f}) — "
                    f"hard filters in base strategy blocked it "
                    f"(yes={candidate.yes_price:.3f} no={candidate.no_price:.3f} "
                    f"liq=${candidate.market.liquidity:.0f})"
                )

        logger.info(
            f"BtcEthHighFreq: scan complete — {len(opportunities)} opportunity(ies) found"
        )
        return opportunities

    # ------------------------------------------------------------------
    # Market identification
    # ------------------------------------------------------------------

    def _find_candidates(
        self,
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[HighFreqCandidate]:
        """Filter the full market list to BTC/ETH high-freq binary markets.

        Also queries the Gamma API directly for crypto markets by tag/slug
        to catch BTC/ETH 15-min markets that may not be in the top 500.
        """
        candidates: list[HighFreqCandidate] = []
        seen_ids: set[str] = set()

        # Combine scanner markets with directly-fetched crypto markets
        all_markets = list(markets)
        try:
            fetcher = _get_crypto_fetcher()
            extra = fetcher.get_markets()
            for m in extra:
                if m.id not in seen_ids:
                    all_markets.append(m)
            # Also mark scanner-provided markets so we don't double-count
            for m in markets:
                seen_ids.add(m.id)
        except Exception:
            pass  # Non-fatal: fall back to scanner markets only

        logger.debug(
            f"BtcEthHighFreq: scanning {len(all_markets)} markets "
            f"({len(markets)} from scanner, {len(all_markets) - len(markets)} from Gamma)"
        )

        asset_hit_no_tf = 0  # track markets that pass asset but fail timeframe
        for market in all_markets:
            if market.closed or not market.active:
                continue
            if len(market.outcome_prices) != 2:
                continue
            if market.id in seen_ids and market not in markets:
                continue  # Already processed from scanner list
            seen_ids.add(market.id)

            asset = self._detect_asset(market)
            if asset is None:
                continue

            timeframe = self._detect_timeframe(market)
            if timeframe is None:
                asset_hit_no_tf += 1
                if asset_hit_no_tf <= 5:
                    logger.debug(
                        f"BtcEthHighFreq: asset={asset} but no timeframe — "
                        f"slug={market.slug} question={market.question[:80]}"
                    )
                continue

            # Resolve live prices
            yes_price, no_price = self._resolve_prices(market, prices)

            candidates.append(
                HighFreqCandidate(
                    market=market,
                    asset=asset,
                    timeframe=timeframe,
                    yes_price=yes_price,
                    no_price=no_price,
                )
            )

        return candidates

    @staticmethod
    def _detect_asset(market: Market) -> Optional[str]:
        """Return 'BTC' or 'ETH' if the market targets one of those assets."""
        text = f"{market.question} {market.slug}".lower()
        for asset, keywords in _ASSET_PATTERNS.items():
            if any(kw in text for kw in keywords):
                return asset
        return None

    @staticmethod
    def _detect_timeframe(market: Market) -> Optional[str]:
        """Return '15min' or '1hr' if a matching timeframe is detected."""
        text = f"{market.question} {market.slug}".lower()

        # Try slug regex first — now allows words between asset and timeframe
        slug_text = f"{market.slug} {market.question}".lower()
        slug_match = _SLUG_REGEX.search(slug_text)
        if slug_match:
            raw_tf = slug_match.group(2).lower().replace("-", "").replace("_", "")
            # Check 15m before 5m (15m contains "5m" substring)
            if "15" in raw_tf or "quarter" in raw_tf:
                return "15min"
            if "5m" in raw_tf or (raw_tf.startswith("5") and "15" not in raw_tf):
                return "5min"
            if "1h" in raw_tf or "60" in raw_tf or "hourly" in raw_tf:
                return "1hr"

        # Fallback: question-text keyword matching (broadened patterns)
        # Check 15m before 5m (15m contains "5m" substring)
        for tf_key in ("15min", "5min", "1hr"):
            patterns = _TIMEFRAME_PATTERNS.get(tf_key, [])
            if any(p in text for p in patterns):
                return tf_key

        return None

    @staticmethod
    def _is_direction_market(market: Market) -> bool:
        """Check if the market is a directional up/down style question."""
        text = market.question.lower()
        return any(kw in text for kw in _DIRECTION_KEYWORDS)

    @staticmethod
    def _resolve_prices(
        market: Market,
        prices: dict[str, dict],
    ) -> tuple[float, float]:
        """Return (yes_price, no_price) using live CLOB prices when available."""
        yes_price = market.yes_price
        no_price = market.no_price

        if market.clob_token_ids:
            if len(market.clob_token_ids) > 0:
                token = market.clob_token_ids[0]
                if token in prices:
                    yes_price = prices[token].get("mid", yes_price)
            if len(market.clob_token_ids) > 1:
                token = market.clob_token_ids[1]
                if token in prices:
                    no_price = prices[token].get("mid", no_price)

        return yes_price, no_price

    # ------------------------------------------------------------------
    # Price history
    # ------------------------------------------------------------------

    def _update_price_history(self, candidate: HighFreqCandidate) -> None:
        """Record the latest prices into the rolling window for this market."""
        mid = candidate.market.id
        if mid not in self._price_histories:
            if candidate.timeframe == "1hr":
                window = _1HR_HISTORY_WINDOW_SEC
            elif candidate.timeframe == "5min":
                window = 120  # 2 min for 5-min markets
            else:
                window = _DEFAULT_HISTORY_WINDOW_SEC
            self._price_histories[mid] = MarketPriceHistory(window_seconds=window)

        self._price_histories[mid].record(
            candidate.yes_price,
            candidate.no_price,
        )

    def _get_history(self, market_id: str) -> Optional[MarketPriceHistory]:
        return self._price_histories.get(market_id)

    # ------------------------------------------------------------------
    # Dynamic strategy selector
    # ------------------------------------------------------------------

    def _select_sub_strategy(
        self,
        candidate: HighFreqCandidate,
    ) -> tuple[Optional[SubStrategyScore], list[SubStrategyScore]]:
        """Score all three sub-strategies and return the best one.

        Returns (best_score_or_None, all_scores).
        A sub-strategy with score <= 0 is considered non-viable.
        """
        scores: list[SubStrategyScore] = [
            self._score_pure_arb(candidate),
            self._score_dump_hedge(candidate),
            self._score_pre_placed_limits(candidate),
            self._score_directional_edge(candidate),
        ]

        # Sort descending by score
        scores.sort(key=lambda s: s.score, reverse=True)

        best = scores[0] if scores[0].score > 0 else None
        return best, scores

    # -- Sub-strategy A: Pure Arbitrage scoring --

    def _score_pure_arb(self, c: HighFreqCandidate) -> SubStrategyScore:
        """Score pure arbitrage opportunity (YES + NO < $1.00).

        Higher score when combined cost is lower (larger guaranteed spread).
        Select when combined < 0.98.
        """
        combined = c.yes_price + c.no_price
        # Use actual Polymarket fee curve: fee depends on the winning price
        # For pure arb we buy both sides, winner is at ~$1 so fee is near 0.
        # Conservative: use fee at the more expensive side.
        fee_cost = polymarket_fee_curve(max(c.yes_price, c.no_price))

        # Net profit per $1 payout after fees
        net_profit = 1.0 - combined - fee_cost
        if net_profit <= 0:
            return SubStrategyScore(
                strategy=SubStrategy.PURE_ARB,
                score=0.0,
                reason=f"No spread after fees (combined={combined:.4f}, fee={fee_cost:.4f})",
            )

        if combined >= _pure_arb_max_combined():
            return SubStrategyScore(
                strategy=SubStrategy.PURE_ARB,
                score=0.0,
                reason=f"Combined cost {combined:.4f} >= {_pure_arb_max_combined()} threshold",
            )

        # Base score proportional to net profit (scale: 1 cent = 10 points)
        base_score = net_profit * 1000.0  # e.g. 0.02 net profit -> 20 pts

        # Bonus for high liquidity (confidence we can fill)
        liquidity = c.market.liquidity
        if liquidity >= 5000:
            base_score += 15.0
        elif liquidity >= 2000:
            base_score += 8.0
        elif liquidity >= 1000:
            base_score += 3.0

        # Bonus for balanced prices (both sides near 0.49-0.50 = most liquid)
        balance = 1.0 - abs(c.yes_price - c.no_price)
        base_score += balance * 5.0

        return SubStrategyScore(
            strategy=SubStrategy.PURE_ARB,
            score=base_score,
            reason=(
                f"Pure arb: combined={combined:.4f}, net_profit={net_profit:.4f}, "
                f"liquidity=${liquidity:.0f}"
            ),
            params={
                "combined_cost": combined,
                "net_profit": net_profit,
                "yes_price": c.yes_price,
                "no_price": c.no_price,
            },
        )

    # -- Sub-strategy B: Dump-Hedge scoring --

    def _score_dump_hedge(self, c: HighFreqCandidate) -> SubStrategyScore:
        """Score dump-hedge opportunity.

        Triggered when one side drops > 5% in the recent window. Buy the dumped
        side (now cheap), wait for partial recovery, then hedge with the other
        side. If the combined cost after hedge < target, there is profit.

        For high-frequency up/down markets that open at 0.50/0.50, any
        deviation from 0.50 on first observation represents a dump from
        the opening price — we don't need historical snapshots to detect it.
        """
        history = self._get_history(c.market.id)

        # For 15-min/1-hr up-or-down markets, the opening price is always 0.50.
        # If we have no history yet, infer the "dump" from deviation off 0.50.
        if history is None or not history.has_data:
            yes_dev = 0.50 - c.yes_price  # positive if YES dropped below 0.50
            no_dev = 0.50 - c.no_price  # positive if NO dropped below 0.50
            max_drop = max(yes_dev, no_dev, 0.0)
            dumped_side = "YES" if yes_dev >= no_dev else "NO"
        else:
            yes_drop = history.max_drop_yes()
            no_drop = history.max_drop_no()
            max_drop = max(yes_drop, no_drop)
            dumped_side = "YES" if yes_drop >= no_drop else "NO"

        if max_drop < _dump_hedge_drop_pct():
            return SubStrategyScore(
                strategy=SubStrategy.DUMP_HEDGE,
                score=0.0,
                reason=(
                    f"Insufficient dump: max drop {max_drop:.4f} "
                    f"< {_dump_hedge_drop_pct()} threshold"
                ),
            )

        # Profit model for up-or-down markets: these are ~50/50 binary
        # outcomes, so the fair value of each side is ~$0.50.  When one
        # side dumps to e.g. $0.40, expected value = 0.50 - 0.40 = $0.10
        # minus fees.  We can also attempt to hedge with the other side
        # if combined < $1.00.
        dumped_price = c.yes_price if dumped_side == "YES" else c.no_price

        # Skip near-resolved markets: if the dumped side is below $0.05,
        # the market has effectively resolved (one outcome is ~certain)
        # and this is NOT a temporary dump worth trading.
        if dumped_price < 0.05:
            return SubStrategyScore(
                strategy=SubStrategy.DUMP_HEDGE,
                score=0.0,
                reason=(
                    f"Market effectively resolved ({dumped_side} at "
                    f"${dumped_price:.4f} < $0.05 threshold)"
                ),
            )

        fair_value = 0.50  # up-or-down markets are coin flips
        dump_fee = polymarket_fee_curve(dumped_price)
        ev_profit = fair_value - dumped_price - dump_fee

        combined = c.yes_price + c.no_price
        # If combined < $1.00 there's also a guaranteed arb component
        arb_fee = polymarket_fee_curve(max(c.yes_price, c.no_price))
        guaranteed_component = max(1.0 - combined - arb_fee, 0.0)

        if ev_profit <= 0 and guaranteed_component <= 0:
            return SubStrategyScore(
                strategy=SubStrategy.DUMP_HEDGE,
                score=0.0,
                reason=(
                    f"No profit after fees: dumped={dumped_side}@{dumped_price:.4f}, "
                    f"EV profit={ev_profit:.4f}, combined={combined:.4f}"
                ),
            )

        # Score: larger drop and larger EV profit = better
        base_score = max_drop * 200.0  # 5% drop -> 10 pts, 10% drop -> 20 pts
        base_score += max(ev_profit, 0) * 500.0  # reward EV-profitable dumps
        base_score += guaranteed_component * 1000.0  # strongly reward guaranteed arb

        # Volatility bonus: higher volatility means more dump-hedge opportunities
        volatility = (
            history.recent_volatility() if (history and history.has_data) else 0.0
        )
        base_score += volatility * 50.0

        # Liquidity matters: need to be able to fill quickly
        if c.market.liquidity >= 3000:
            base_score += 5.0
        elif c.market.liquidity < 1000:
            base_score *= 0.5  # penalize low liquidity heavily

        return SubStrategyScore(
            strategy=SubStrategy.DUMP_HEDGE,
            score=base_score,
            reason=(
                f"Dump-hedge: {dumped_side} dropped {max_drop:.4f} "
                f"(price={dumped_price:.4f}), EV profit={ev_profit:.4f}, "
                f"combined={combined:.4f}, volatility={volatility:.4f}"
            ),
            params={
                "dumped_side": dumped_side,
                "drop_amount": max_drop,
                "dumped_price": dumped_price,
                "ev_profit": ev_profit,
                "combined_cost": combined,
                "guaranteed_component": guaranteed_component,
                "volatility": volatility,
                "yes_price": c.yes_price,
                "no_price": c.no_price,
            },
        )

    # -- Sub-strategy C: Pre-Placed Limits scoring --

    def _calculate_dynamic_limit_prices(
        self, c: HighFreqCandidate
    ) -> tuple[float, float]:
        """Calculate optimal limit order prices based on current market state.

        Instead of fixed $0.45-$0.47, adjusts based on:
        - Current market prices (bid closer to current for fill probability)
        - Liquidity level (thinner book = can be more aggressive)
        - Required minimum profit margin
        """
        # Base targets
        min_target = 0.42  # Absolute minimum (most aggressive)

        # Start from the aggressive end
        base_price = 0.45

        # Adjust based on liquidity: thinner book = can be more aggressive
        if c.market.liquidity < 100:
            base_price = min_target  # Very thin, be aggressive
        elif c.market.liquidity < 250:
            base_price = 0.44
        elif c.market.liquidity < 400:
            base_price = 0.45
        else:
            base_price = 0.46  # Moderate book, bid closer to fill

        # Ensure combined still profits: need combined < 1.0 - fee
        # If we bid base_price on both sides, combined = 2 * base_price
        # Need: 2 * base_price + fee < 1.0
        limit_fee = polymarket_fee_curve(0.48)  # fee at typical limit price
        max_combined = 1.0 - limit_fee - 0.01  # Leave 1% minimum profit
        max_per_side = max_combined / 2.0

        target_price = min(base_price, max_per_side)
        target_price = max(target_price, min_target)  # Never go below absolute min

        return target_price, target_price

    def _score_pre_placed_limits(self, c: HighFreqCandidate) -> SubStrategyScore:
        """Score pre-placed limit order opportunity.

        For markets about to open or with very thin order books, pre-place
        limit orders at $0.45-$0.47 on both sides. If both fill, combined cost
        is $0.90-$0.94 for guaranteed $1.00 payout.

        Select when: new market detected with thin order book.
        """
        liquidity = c.market.liquidity

        # This sub-strategy targets thin/new markets
        if liquidity > _thin_liquidity_usd():
            return SubStrategyScore(
                strategy=SubStrategy.PRE_PLACED_LIMITS,
                score=0.0,
                reason=(
                    f"Market too liquid (${liquidity:.0f}) for pre-placed limits "
                    f"(threshold=${_thin_liquidity_usd():.0f})"
                ),
            )

        # Check if prices are near the sweet spot (0.45-0.55 per side = new market)
        both_near_half = _LIMIT_ORDER_TARGET_LOW <= c.yes_price <= (
            1.0 - _LIMIT_ORDER_TARGET_LOW
        ) and _LIMIT_ORDER_TARGET_LOW <= c.no_price <= (1.0 - _LIMIT_ORDER_TARGET_LOW)

        # Calculate dynamic limit prices based on market conditions
        target_yes, target_no = self._calculate_dynamic_limit_prices(c)
        target_combined = target_yes + target_no
        target_fee = polymarket_fee_curve(max(target_yes, target_no))
        target_profit = 1.0 - target_combined - target_fee

        if target_profit <= 0:
            return SubStrategyScore(
                strategy=SubStrategy.PRE_PLACED_LIMITS,
                score=0.0,
                reason=f"No profit at target prices (combined=${target_combined:.4f})",
            )

        # Base score: thin book is a strong signal
        base_score = 10.0

        # Lower liquidity = more opportunity for limit fills
        if liquidity < 100:
            base_score += 20.0
        elif liquidity < 250:
            base_score += 10.0
        else:
            base_score += 3.0

        # Near-half prices suggest a freshly opened market (ideal)
        if both_near_half:
            base_score += 15.0

        # Market age proxy: very low volume = likely just opened
        if c.market.volume < 100:
            base_score += 20.0  # Almost certainly a new market
        elif c.market.volume < 500:
            base_score += 12.0  # Very new
        elif c.market.volume < _NEW_MARKET_VOLUME_THRESHOLD:
            base_score += 5.0  # Relatively new

        # Bonus for the expected profit
        base_score += target_profit * 300.0

        # Penalty: if the market already has significant volume, limits are
        # less likely to fill at our targets
        if c.market.volume > 10000:
            base_score *= 0.4

        return SubStrategyScore(
            strategy=SubStrategy.PRE_PLACED_LIMITS,
            score=base_score,
            reason=(
                f"Pre-placed limits: liquidity=${liquidity:.0f}, "
                f"target_combined=${target_combined:.4f}, "
                f"target_profit=${target_profit:.4f}, "
                f"prices_near_half={both_near_half}"
            ),
            params={
                "target_yes_price": target_yes,
                "target_no_price": target_no,
                "target_combined": target_combined,
                "target_profit": target_profit,
                "current_yes_price": c.yes_price,
                "current_no_price": c.no_price,
                "liquidity": liquidity,
            },
        )

    # -- Sub-strategy D: Directional Edge scoring --

    def _score_directional_edge(self, c: HighFreqCandidate) -> SubStrategyScore:
        """Score directional edge opportunity using Chainlink oracle prices.

        Compares real-time Chainlink oracle price against the market's
        "price to beat" to estimate probability of Up vs Down.  When the
        model probability diverges from market-implied probability by >5%,
        there's a directional edge.

        This is the primary alpha strategy for 15-minute crypto markets.
        """
        try:
            from services.chainlink_feed import get_chainlink_feed
        except ImportError:
            return SubStrategyScore(
                strategy=SubStrategy.DIRECTIONAL_EDGE,
                score=0.0,
                reason="Chainlink feed not available",
            )

        feed = get_chainlink_feed()
        oracle = feed.get_price(c.asset)
        if not oracle or not oracle.price:
            return SubStrategyScore(
                strategy=SubStrategy.DIRECTIONAL_EDGE,
                score=0.0,
                reason=f"No oracle price for {c.asset}",
            )

        # Check oracle freshness (must be <60 seconds old)
        age_ms = (time.time() * 1000) - (oracle.updated_at_ms or 0)
        if age_ms > 60_000:
            return SubStrategyScore(
                strategy=SubStrategy.DIRECTIONAL_EDGE,
                score=0.0,
                reason=f"Oracle price stale ({age_ms / 1000:.0f}s old)",
            )

        # Extract "price to beat" from the market question
        # E.g. "Bitcoin Up or Down - February 10, 10:15AM-10:30AM ET"
        # The price to beat is the Chainlink price at start_time
        # For now, we use the midpoint: if up_price > 0.5, market thinks Up
        market_up_prob = c.yes_price  # Market-implied probability of Up
        market_down_prob = c.no_price

        # Build a simple directional model:
        # If oracle price is trending in a direction, that direction is more likely
        # The market at fair value has Up/Down both at ~0.50
        # Any deviation > 5% from 0.50 implies the oracle is moving
        price_history = self._price_histories.get(c.market.id)
        if not price_history or len(price_history.snapshots) < 3:
            return SubStrategyScore(
                strategy=SubStrategy.DIRECTIONAL_EDGE,
                score=0.0,
                reason="Insufficient price history for directional signal",
            )

        # Calculate model probability from market movement
        # If yes_price (Up) has been rising, model should agree
        snapshots = list(price_history.snapshots)
        recent_yes = [s.yes_price for s in snapshots[-5:]]  # Last 5 yes prices
        if len(recent_yes) < 3:
            return SubStrategyScore(
                strategy=SubStrategy.DIRECTIONAL_EDGE,
                score=0.0,
                reason="Not enough recent snapshots",
            )

        # Trend: is the market moving consistently in one direction?
        trend = recent_yes[-1] - recent_yes[0]  # Positive = trending Up
        trend_strength = abs(trend)

        # Model probability: base 50/50, adjusted by trend
        model_up = 0.50 + (trend * 2.0)  # Scale trend into probability
        model_up = max(0.30, min(0.70, model_up))  # Clamp
        model_down = 1.0 - model_up

        # Calculate edge: model vs market
        edge_up = model_up - market_up_prob
        edge_down = model_down - market_down_prob

        best_side = "UP" if edge_up > edge_down else "DOWN"
        best_edge = edge_up if best_side == "UP" else edge_down

        # Time-phase awareness: determine where we are in the 15-min window
        # EARLY (10-15 min left): conservative, require large edge
        # MID (5-10 min left): moderate thresholds
        # LATE (<5 min left): aggressive, model is most reliable
        remaining_secs = None
        if c.market.end_date:
            try:
                end_str = str(c.market.end_date)
                if hasattr(c.market.end_date, "timestamp"):
                    end_ts = c.market.end_date.timestamp()
                else:
                    end_ts = datetime.fromisoformat(
                        end_str.replace("Z", "+00:00")
                    ).timestamp()
                remaining_secs = max(0, end_ts - time.time())
            except (ValueError, AttributeError):
                pass

        remaining_min = (remaining_secs / 60.0) if remaining_secs else 15.0

        if remaining_min > 10:
            phase = "EARLY"
            min_edge = 0.08       # Require 8% edge early (less predictable)
            score_multiplier = 1.0
        elif remaining_min > 5:
            phase = "MID"
            min_edge = 0.05       # 5% edge in the middle
            score_multiplier = 1.5
        else:
            phase = "LATE"
            min_edge = 0.03       # 3% edge late (model is most accurate)
            score_multiplier = 2.0

        if best_edge < min_edge:
            return SubStrategyScore(
                strategy=SubStrategy.DIRECTIONAL_EDGE,
                score=0.0,
                reason=(
                    f"Edge too small ({phase}): {best_side} edge={best_edge:.3f} "
                    f"(need >{min_edge:.2f}), model_up={model_up:.2f} "
                    f"vs market_up={market_up_prob:.3f}, "
                    f"{remaining_min:.1f}min left"
                ),
            )

        # Score based on edge size, amplified by time phase
        score = min(best_edge * 500.0 * score_multiplier, 80.0)

        # Bonus for trend strength
        if trend_strength > 0.05:
            score += 15.0
        elif trend_strength > 0.02:
            score += 8.0

        # LATE phase bonus: we're most confident here
        if phase == "LATE":
            score += 20.0

        # The price we'd buy at
        buy_price = market_up_prob if best_side == "UP" else market_down_prob
        buy_fee = polymarket_fee_curve(buy_price)

        return SubStrategyScore(
            strategy=SubStrategy.DIRECTIONAL_EDGE,
            score=score,
            reason=(
                f"Directional {best_side} ({phase}, {remaining_min:.0f}m left): "
                f"edge={best_edge:.3f}, model_up={model_up:.2f}, "
                f"market_up={market_up_prob:.3f}, trend={trend:+.4f}, "
                f"fee={buy_fee:.4f}"
            ),
            params={
                "side": best_side,
                "edge": best_edge,
                "model_up": model_up,
                "model_down": model_down,
                "market_up": market_up_prob,
                "market_down": market_down_prob,
                "buy_price": buy_price,
                "oracle_price": oracle.price,
                "trend": trend,
                "trend_strength": trend_strength,
                "phase": phase,
                "remaining_minutes": remaining_min,
            },
        )

    # ------------------------------------------------------------------
    # Opportunity generation
    # ------------------------------------------------------------------

    def _generate_opportunity(
        self,
        candidate: HighFreqCandidate,
        selected: SubStrategyScore,
    ) -> Optional[ArbitrageOpportunity]:
        """Turn a scored sub-strategy into an ArbitrageOpportunity via the base
        class ``create_opportunity`` (which applies all hard filters)."""

        market = candidate.market
        sub = selected.strategy
        params = selected.params

        if sub == SubStrategy.PURE_ARB:
            return self._generate_pure_arb(candidate, params)
        elif sub == SubStrategy.DUMP_HEDGE:
            return self._generate_dump_hedge(candidate, params)
        elif sub == SubStrategy.PRE_PLACED_LIMITS:
            return self._generate_pre_placed_limits(candidate, params)
        elif sub == SubStrategy.DIRECTIONAL_EDGE:
            return self._generate_directional_edge(candidate, params)

        logger.warning(
            "BtcEthHighFreq: unknown sub-strategy %s for market %s",
            sub,
            market.id,
        )
        return None

    def _generate_pure_arb(
        self,
        c: HighFreqCandidate,
        params: dict,
    ) -> Optional[ArbitrageOpportunity]:
        """Generate opportunity for sub-strategy A: Pure Arbitrage."""
        market = c.market
        yes_price = params["yes_price"]
        no_price = params["no_price"]
        combined = params["combined_cost"]

        positions = self._build_both_sides_positions(market, yes_price, no_price)

        opp = self.create_opportunity(
            title=(
                f"BTC/ETH HF Pure Arb: {c.asset} {c.timeframe} ({market.question[:40]})"
            ),
            description=(
                f"Pure arbitrage on {c.asset} {c.timeframe} market. "
                f"Buy YES (${yes_price:.4f}) + NO (${no_price:.4f}) = "
                f"${combined:.4f} for guaranteed $1.00 payout."
            ),
            total_cost=combined,
            markets=[market],
            positions=positions,
            min_liquidity_hard=200.0,
            min_position_size=10.0,
            min_absolute_profit=2.0,
        )

        if opp is not None:
            self._attach_highfreq_metadata(opp, c, SubStrategy.PURE_ARB, params)
        return opp

    def _generate_dump_hedge(
        self,
        c: HighFreqCandidate,
        params: dict,
    ) -> Optional[ArbitrageOpportunity]:
        """Generate opportunity for sub-strategy B: Dump-Hedge.

        Modeled as a directional bet: buy only the dumped side at a price
        below fair value ($0.50 for 50/50 binary markets).  The hedge
        (buying the opposite side) is an optional follow-up, not part of
        the initial cost.
        """
        market = c.market
        dumped_side = params["dumped_side"]
        drop_amount = params["drop_amount"]
        dumped_price = params.get(
            "dumped_price",
            params["yes_price"] if dumped_side == "YES" else params["no_price"],
        )
        ev_profit = params.get("ev_profit", 0)
        yes_price = params["yes_price"]
        no_price = params["no_price"]
        combined = params["combined_cost"]

        # Build position for the dumped side only (directional bet).
        # The "hedge" is a potential follow-up, not an immediate action.
        positions = []
        if market.clob_token_ids and len(market.clob_token_ids) >= 2:
            token_idx = 0 if dumped_side == "YES" else 1
            positions = [
                {
                    "action": "BUY",
                    "outcome": dumped_side,
                    "price": dumped_price,
                    "token_id": market.clob_token_ids[token_idx],
                    "role": "primary",
                    "note": (
                        f"Buy dumped side ({dumped_side} dropped "
                        f"{drop_amount:.4f} to {dumped_price:.4f})"
                    ),
                },
            ]

        opp = self.create_opportunity(
            title=(
                f"BTC/ETH HF Dump-Hedge: {c.asset} {c.timeframe} "
                f"({dumped_side} dropped to {dumped_price:.2f})"
            ),
            description=(
                f"Dump-hedge on {c.asset} {c.timeframe} market. "
                f"{dumped_side} dropped {drop_amount:.4f} to {dumped_price:.4f} — "
                f"buy dumped side (EV profit ~${ev_profit:.4f}). "
                f"Fair value $0.50 for 50/50 binary. "
                f"Optional hedge: buy {('NO' if dumped_side == 'YES' else 'YES')} "
                f"if combined (${combined:.4f}) drops below $1.00."
            ),
            total_cost=dumped_price,
            expected_payout=0.50,  # EV: 50% probability * $1.00 payout
            is_guaranteed=False,
            markets=[market],
            positions=positions,
            min_liquidity_hard=200.0,
            min_position_size=5.0,
            min_absolute_profit=1.0,
        )

        if opp is not None:
            self._attach_highfreq_metadata(opp, c, SubStrategy.DUMP_HEDGE, params)
            opp.risk_factors.insert(
                0,
                f"Directional bet: profit depends on {dumped_side} recovering "
                f"toward fair value ($0.50)",
            )
        return opp

    def _generate_pre_placed_limits(
        self,
        c: HighFreqCandidate,
        params: dict,
    ) -> Optional[ArbitrageOpportunity]:
        """Generate opportunity for sub-strategy C: Pre-Placed Limits.

        Hard filters are relaxed because these are LIMIT orders on newly
        opened / thin-book markets.  Current liquidity may be very low
        (or zero), but the orders will fill as liquidity arrives.
        """
        market = c.market
        target_yes = params["target_yes_price"]
        target_no = params["target_no_price"]
        target_combined = params["target_combined"]

        positions = []
        if market.clob_token_ids and len(market.clob_token_ids) >= 2:
            positions = [
                {
                    "action": "LIMIT_BUY",
                    "outcome": "YES",
                    "price": target_yes,
                    "token_id": market.clob_token_ids[0],
                    "note": f"Limit order at ${target_yes:.2f}",
                },
                {
                    "action": "LIMIT_BUY",
                    "outcome": "NO",
                    "price": target_no,
                    "token_id": market.clob_token_ids[1],
                    "note": f"Limit order at ${target_no:.2f}",
                },
            ]

        # Relax hard filters: limit orders on new/thin markets don't
        # depend on current liquidity for execution — they fill when
        # liquidity arrives.  Zero-liquidity markets are expected.
        opp = self.create_opportunity(
            title=(f"BTC/ETH HF Pre-Limits: {c.asset} {c.timeframe} (thin book)"),
            description=(
                f"Pre-placed limit orders on {c.asset} {c.timeframe} market "
                f"(liquidity=${params.get('liquidity', 0):.0f}). "
                f"Target: YES@${target_yes:.2f} + NO@${target_no:.2f} = "
                f"${target_combined:.4f} for $1.00 payout."
            ),
            total_cost=target_combined,
            markets=[market],
            positions=positions,
            min_liquidity_hard=0.0,   # New markets may have $0 liquidity
            min_position_size=0.0,    # Limit orders, not market orders
            min_absolute_profit=0.0,  # Profit realized on fill, not now
        )

        if opp is not None:
            # Set a reasonable position size for limit orders (not
            # constrained by current liquidity like market orders).
            opp.max_position_size = max(opp.max_position_size, 50.0)

            self._attach_highfreq_metadata(
                opp,
                c,
                SubStrategy.PRE_PLACED_LIMITS,
                params,
            )
            opp.risk_factors.insert(
                0,
                "Pre-placed limits: profit only if BOTH sides fill at target prices",
            )
        return opp

    def _generate_directional_edge(
        self,
        c: HighFreqCandidate,
        params: dict,
    ) -> Optional[ArbitrageOpportunity]:
        """Generate opportunity for sub-strategy D: Directional Edge.

        Buys only the predicted winning side (directional bet, not arb).
        Uses maker orders to avoid taker fees and earn rebates.
        """
        market = c.market
        side = params["side"]  # "UP" or "DOWN"
        edge = params["edge"]
        buy_price = params["buy_price"]
        model_up = params["model_up"]

        # Build single-side position
        maker_mode = _cfg.BTC_ETH_HF_MAKER_MODE
        positions = []
        if market.clob_token_ids and len(market.clob_token_ids) >= 2:
            if side == "UP":
                token_id = market.clob_token_ids[0]
                outcome = "YES"
            else:
                token_id = market.clob_token_ids[1]
                outcome = "NO"

            positions = [
                {
                    "action": "BUY",
                    "outcome": outcome,
                    "price": buy_price,
                    "token_id": token_id,
                    "_maker_mode": maker_mode,
                    "_maker_price": buy_price,
                }
            ]

        # Fair value for directional bet: model probability * $1.00 payout
        expected_payout = model_up if side == "UP" else (1.0 - model_up)

        opp = self.create_opportunity(
            title=(
                f"BTC/ETH HF Directional: {c.asset} {c.timeframe} "
                f"({side} edge {edge:.1%})"
            ),
            description=(
                f"Directional {side} bet on {c.asset} {c.timeframe} market. "
                f"Model: {model_up:.0%} Up / {1-model_up:.0%} Down. "
                f"Market: {params['market_up']:.1%} Up. "
                f"Edge: {edge:.1%}. "
                f"{'Maker order (0% fee + rebates).' if maker_mode else ''}"
            ),
            total_cost=buy_price,
            expected_payout=expected_payout,
            markets=[market],
            positions=positions,
            is_guaranteed=False,  # Directional, not guaranteed
            min_liquidity_hard=200.0,
            min_position_size=5.0,
            min_absolute_profit=0.5,
        )

        if opp is not None:
            self._attach_highfreq_metadata(
                opp,
                c,
                SubStrategy.DIRECTIONAL_EDGE,
                params,
            )
            opp.risk_factors.insert(
                0,
                f"Directional bet: profit depends on {c.asset} going {side}",
            )
        return opp

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_both_sides_positions(
        market: Market,
        yes_price: float,
        no_price: float,
    ) -> list[dict]:
        """Build standard BUY YES + BUY NO position list."""
        maker_mode = _cfg.BTC_ETH_HF_MAKER_MODE
        positions: list[dict] = []
        if market.clob_token_ids and len(market.clob_token_ids) >= 2:
            positions = [
                {
                    "action": "BUY",
                    "outcome": "YES",
                    "price": yes_price,
                    "token_id": market.clob_token_ids[0],
                    "_maker_mode": maker_mode,
                    "_maker_price": yes_price,
                },
                {
                    "action": "BUY",
                    "outcome": "NO",
                    "price": no_price,
                    "token_id": market.clob_token_ids[1],
                    "_maker_mode": maker_mode,
                    "_maker_price": no_price,
                },
            ]
        return positions

    @staticmethod
    def _attach_highfreq_metadata(
        opp: ArbitrageOpportunity,
        candidate: HighFreqCandidate,
        sub_strategy: SubStrategy,
        params: dict,
    ) -> None:
        """Attach BTC/ETH high-freq metadata to the opportunity for
        downstream consumers (execution engine, dashboard, logging)."""
        # Store in the existing positions_to_take metadata (which is a list
        # of dicts). We append a metadata entry at the end.
        opp.positions_to_take.append(
            {
                "_highfreq_metadata": True,
                "asset": candidate.asset,
                "timeframe": candidate.timeframe,
                "sub_strategy": sub_strategy.value,
                "sub_strategy_params": params,
            }
        )
