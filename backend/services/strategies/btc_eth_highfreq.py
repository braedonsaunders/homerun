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
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

import math

from models import Market, Event, Opportunity
from config import settings as _cfg
from .base import BaseStrategy, DecisionCheck, StrategyDecision, ExitDecision
from services.data_events import DataEvent
from utils.converters import to_float, to_confidence, to_bool, clamp
from utils.signal_helpers import signal_payload
from services.quality_filter import QualityFilterOverrides
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Evaluate-method constants (ported from BaseCryptoTimeframeStrategy)
# ---------------------------------------------------------------------------

_ALLOWED_MODES = {"auto", "directional", "pure_arb", "rebalance"}
_REGIMES = {"opening", "mid", "closing"}

_EDGE_MODE_FACTORS: dict[str, dict[str, float]] = {
    "opening": {"auto": 1.0, "directional": 1.05, "pure_arb": 0.90, "rebalance": 1.05},
    "mid": {"auto": 1.0, "directional": 1.00, "pure_arb": 0.85, "rebalance": 1.00},
    "closing": {"auto": 0.9, "directional": 0.90, "pure_arb": 0.80, "rebalance": 0.85},
}

_CONF_MODE_FACTORS: dict[str, float] = {
    "auto": 1.0,
    "directional": 1.0,
    "pure_arb": 0.9,
    "rebalance": 0.95,
}

_REGIME_CONF_FACTORS: dict[str, float] = {
    "opening": 1.0,
    "mid": 1.0,
    "closing": 0.95,
}

_MODE_SIZE_FACTORS: dict[str, float] = {
    "auto": 1.0,
    "directional": 1.0,
    "pure_arb": 0.85,
    "rebalance": 0.9,
}

_REGIME_SIZE_FACTORS: dict[str, float] = {
    "opening": 0.95,
    "mid": 1.0,
    "closing": 1.1,
}


# ---------------------------------------------------------------------------
# Evaluate-method helpers (ported from BaseCryptoTimeframeStrategy)
# ---------------------------------------------------------------------------


def _normalize_mode(value: Any) -> str:
    mode = str(value or "auto").strip().lower()
    if mode not in _ALLOWED_MODES:
        return "auto"
    return mode


def _normalize_regime(value: Any) -> str:
    regime = str(value or "mid").strip().lower()
    if regime not in _REGIMES:
        return "mid"
    return regime


def _normalize_asset(value: Any) -> str:
    asset = str(value or "").strip().upper()
    if asset == "XBT":
        return "BTC"
    return asset


def _normalize_timeframe(value: Any) -> str:
    tf = str(value or "").strip().lower()
    if tf in {"5m", "5min", "5"}:
        return "5m"
    if tf in {"15m", "15min", "15"}:
        return "15m"
    if tf in {"1h", "1hr", "60m", "60min"}:
        return "1h"
    if tf in {"4h", "4hr", "240m", "240min"}:
        return "4h"
    return tf


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",")]
    return []


def _normalize_scope(value: Any, normalizer: Callable[[Any], str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in _as_list(value):
        normalized = normalizer(raw)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _normalize_regime_scope(value: Any) -> set[str]:
    allowed = set(_REGIMES)
    normalized: set[str] = set()
    for raw in _as_list(value):
        regime = _normalize_regime(raw)
        if regime in allowed:
            normalized.add(regime)
    return normalized


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _get_component_edge(payload: dict[str, Any], direction: str, mode: str) -> float:
    component_edges = payload.get("component_edges")
    if not isinstance(component_edges, dict):
        return 0.0
    side_edges = component_edges.get(direction)
    if not isinstance(side_edges, dict):
        return 0.0
    return max(0.0, to_float(side_edges.get(mode), 0.0))


def _get_net_edge(payload: dict[str, Any], direction: str, fallback: float) -> float:
    net_edges = payload.get("net_edges")
    if not isinstance(net_edges, dict):
        return fallback
    return to_float(net_edges.get(direction), fallback)


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
    "4hr": [
        "updown-4h",  # actual Polymarket slug pattern
        "updown-4h-",  # with trailing timestamp
        "4 hour",
        "4-hour",
        "4hr",
        "4h-",  # short form in slugs
        "4h",  # bare short form
        "four hour",
        "four-hour",
        "240 min",
        "240-min",
        "240m",  # short form
        "240 minute",
        "240-minute",
        "240 minutes",
        "240-minutes",
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
    r"|4[\s_-]?h(?:(?:ou)?r)?"  # "4h", "4hr", "4hour", "4-h", …
    r"|240[\s_-]?m(?:in(?:ute)?s?)?"  # "240m", "240min", "240 minutes", etc.
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
        (_cfg.BTC_ETH_HF_SERIES_BTC_5M, "BTC", "5min"),
        (_cfg.BTC_ETH_HF_SERIES_ETH_5M, "ETH", "5min"),
        (_cfg.BTC_ETH_HF_SERIES_SOL_5M, "SOL", "5min"),
        (_cfg.BTC_ETH_HF_SERIES_XRP_5M, "XRP", "5min"),
        (_cfg.BTC_ETH_HF_SERIES_BTC_1H, "BTC", "1hr"),
        (_cfg.BTC_ETH_HF_SERIES_ETH_1H, "ETH", "1hr"),
        (_cfg.BTC_ETH_HF_SERIES_SOL_1H, "SOL", "1hr"),
        (_cfg.BTC_ETH_HF_SERIES_XRP_1H, "XRP", "1hr"),
        (_cfg.BTC_ETH_HF_SERIES_BTC_4H, "BTC", "4hr"),
        (_cfg.BTC_ETH_HF_SERIES_ETH_4H, "ETH", "4hr"),
        (_cfg.BTC_ETH_HF_SERIES_SOL_4H, "SOL", "4hr"),
        (_cfg.BTC_ETH_HF_SERIES_XRP_4H, "XRP", "4hr"),
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


# ---------------------------------------------------------------------------
# Sub-strategy scoring constants
# ---------------------------------------------------------------------------

# -- Pure Arb scoring --
_PURE_ARB_NET_PROFIT_SCALE = 1000.0  # Converts net profit to score points (0.02 -> 20 pts)
_PURE_ARB_HIGH_LIQUIDITY_USD = 5000.0  # Liquidity threshold for full bonus
_PURE_ARB_HIGH_LIQUIDITY_BONUS = 15.0
_PURE_ARB_MED_LIQUIDITY_USD = 2000.0  # Liquidity threshold for medium bonus
_PURE_ARB_MED_LIQUIDITY_BONUS = 8.0
_PURE_ARB_LOW_LIQUIDITY_USD = 1000.0  # Liquidity threshold for small bonus
_PURE_ARB_LOW_LIQUIDITY_BONUS = 3.0
_PURE_ARB_BALANCE_SCALE = 5.0  # Score weight for balanced yes/no prices

# -- Dump-Hedge scoring --
_DUMP_HEDGE_NEAR_RESOLVED_THRESHOLD = 0.05  # Below this, market is effectively resolved
_DUMP_HEDGE_FAIR_VALUE = 0.50  # Fair value for 50/50 binary up-or-down market
_DUMP_HEDGE_DROP_SCORE_SCALE = 200.0  # Score scale for drop magnitude
_DUMP_HEDGE_EV_PROFIT_SCALE = 500.0  # Score scale for expected-value profit
_DUMP_HEDGE_GUARANTEED_SCALE = 1000.0  # Score scale for guaranteed arb component
_DUMP_HEDGE_VOLATILITY_SCALE = 50.0  # Score scale for recent volatility
_DUMP_HEDGE_HIGH_LIQUIDITY_USD = 3000.0  # Above this, full liquidity bonus
_DUMP_HEDGE_HIGH_LIQUIDITY_BONUS = 5.0
_DUMP_HEDGE_LOW_LIQUIDITY_USD = 1000.0  # Below this, heavy penalty
_DUMP_HEDGE_LOW_LIQUIDITY_PENALTY = 0.5  # Multiplied against score

# -- Pre-Placed Limits scoring --
_LIMIT_ORDER_TARGET_LOW = 0.45  # Lower limit order price
_LIMIT_ORDER_TARGET_HIGH = 0.47  # Upper limit order price
_LIMIT_ORDER_MIN_TARGET = 0.42  # Absolute minimum limit order price
_LIMIT_ORDER_FEE_REFERENCE_PRICE = 0.48  # Typical limit price for fee calculation
_LIMIT_ORDER_MIN_PROFIT_MARGIN = 0.01  # Minimum profit margin after fees
_LIMIT_ORDER_BASE_SCORE = 10.0  # Base score for thin-book opportunity
_LIMIT_VERY_THIN_LIQUIDITY_USD = 100.0  # Very thin book threshold
_LIMIT_VERY_THIN_BONUS = 20.0
_LIMIT_THIN_LIQUIDITY_USD = 250.0  # Thin book threshold
_LIMIT_THIN_BONUS = 10.0
_LIMIT_MODERATE_LIQUIDITY_USD = 400.0  # Moderate book threshold
_LIMIT_MODERATE_BONUS = 3.0
_LIMIT_NEAR_HALF_BONUS = 15.0  # Bonus when prices are near 0.50
_LIMIT_VERY_NEW_VOLUME_USD = 100.0  # Almost certainly new market
_LIMIT_VERY_NEW_BONUS = 20.0
_LIMIT_NEW_VOLUME_USD = 500.0  # Very new market
_LIMIT_NEW_BONUS = 12.0
_LIMIT_PROFIT_SCORE_SCALE = 300.0  # Score scale for expected profit
_LIMIT_SIGNIFICANT_VOLUME_USD = 10000.0  # High volume reduces fill probability
_LIMIT_SIGNIFICANT_VOLUME_PENALTY = 0.4  # Multiplied against score
_NEW_MARKET_VOLUME_THRESHOLD = 5000.0  # Markets with volume below this are "new"

# -- Directional Edge scoring --
_DIRECTIONAL_TREND_SCALE = 2.0  # Scale factor: trend -> probability adjustment
_DIRECTIONAL_MODEL_PROB_MIN = 0.30  # Min model probability clamp
_DIRECTIONAL_MODEL_PROB_MAX = 0.70  # Max model probability clamp
_DIRECTIONAL_EARLY_PHASE_MINUTES = 10.0  # Remaining minutes for early/mid boundary
_DIRECTIONAL_EARLY_MIN_EDGE = 0.08  # Required edge in early phase
_DIRECTIONAL_EARLY_SCORE_MULT = 1.0
_DIRECTIONAL_MID_PHASE_MINUTES = 5.0  # Remaining minutes for mid/late boundary
_DIRECTIONAL_MID_MIN_EDGE = 0.05  # Required edge in mid phase
_DIRECTIONAL_MID_SCORE_MULT = 1.5
_DIRECTIONAL_LATE_MIN_EDGE = 0.03  # Required edge in late phase
_DIRECTIONAL_LATE_SCORE_MULT = 2.0
_DIRECTIONAL_EDGE_SCORE_SCALE = 500.0  # Score scale for edge magnitude
_DIRECTIONAL_MAX_SCORE = 80.0  # Cap on directional score
_DIRECTIONAL_STRONG_TREND_THRESHOLD = 0.05  # High trend strength
_DIRECTIONAL_STRONG_TREND_BONUS = 15.0
_DIRECTIONAL_MODERATE_TREND_THRESHOLD = 0.02  # Moderate trend strength
_DIRECTIONAL_MODERATE_TREND_BONUS = 8.0
_DIRECTIONAL_LATE_PHASE_BONUS = 20.0  # Extra score in late phase

# Price history defaults
_DEFAULT_HISTORY_WINDOW_SEC = 300  # 5 minutes for 15-min markets
_1HR_HISTORY_WINDOW_SEC = 600  # 10 minutes for 1-hr markets
_4HR_HISTORY_WINDOW_SEC = 1800  # 30 minutes for 4-hr markets
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
                asyncio.ensure_future(feed_mgr.polymarket_feed.subscribe(token_ids=token_ids))
            else:
                loop.run_until_complete(feed_mgr.polymarket_feed.subscribe(token_ids=token_ids))
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
            end_ms = datetime.fromisoformat(end_str.replace("Z", "+00:00")).timestamp() * 1000
        except (ValueError, AttributeError):
            return False

        if now_ms >= end_ms:
            return False  # Already resolved

        if start_str:
            try:
                start_ms = datetime.fromisoformat(start_str.replace("Z", "+00:00")).timestamp() * 1000
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
                end_ms = datetime.fromisoformat(end_str.replace("Z", "+00:00")).timestamp() * 1000
            except (ValueError, AttributeError):
                continue
            if end_ms <= now_ms:
                continue  # Already resolved

            start_ms = None
            if start_str:
                try:
                    start_ms = datetime.fromisoformat(start_str.replace("Z", "+00:00")).timestamp() * 1000
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
            return str(mkt.get("conditionId") or mkt.get("condition_id") or mkt.get("id", ""))

        try:
            series = _get_crypto_series()
            now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            with httpx.Client(timeout=10.0) as client:
                for series_id, asset, timeframe in series:
                    try:
                        resp = client.get(
                            f"{self._gamma_url}/events",
                            params={
                                "series_id": series_id,
                                "active": "true",
                                "closed": "false",
                                # Exclude stale unresolved history and walk forward
                                # from now for live/nearest-upcoming selection.
                                "end_date_min": now_iso,
                                "order": "endDate",
                                "ascending": "true",
                                "limit": 10,
                            },
                        )
                        if resp.status_code != 200:
                            logger.debug(
                                "BtcEthHighFreq: Gamma series_id=%s returned %s",
                                series_id,
                                resp.status_code,
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
                                            mid,
                                            e,
                                        )

                        time.sleep(0.05)  # Rate limit between series
                    except Exception as e:
                        logger.debug(
                            "BtcEthHighFreq: series_id=%s fetch failed: %s",
                            series_id,
                            e,
                        )

        except Exception as exc:
            logger.warning(
                "Crypto market fetch failed: %s",
                str(exc),
                exc_info=True,
            )

        if all_markets:
            logger.info(
                "BtcEthHighFreq: fetched %d live crypto markets via Gamma series API (%s)",
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
    timeframe: str  # "5min", "15min", "1hr", or "4hr"
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

    strategy_type = "btc_eth_highfreq"
    name = "BTC/ETH High-Frequency"
    description = "Dynamic high-frequency arbitrage on BTC/ETH 15-min and 1-hr binary markets"
    mispricing_type = "within_market"
    source_key = "crypto"
    worker_affinity = "crypto"
    market_categories = ["crypto"]
    requires_historical_prices = True
    subscriptions = ["crypto_update"]

    quality_filter_overrides = QualityFilterOverrides(
        min_roi=1.0,
        max_resolution_months=0.1,
    )

    default_config = {
        "min_edge_percent": 2.0,
        "min_confidence": 0.40,
        "max_risk_score": 0.80,
        "base_size_usd": 20.0,
        "max_size_usd": 150.0,
        "take_profit_pct": 8.0,
    }

    def __init__(self) -> None:
        super().__init__()
        # 15-minute crypto markets have taker-only fees using a price-curve:
        #   fee_per_share = price * 0.25 * (price * (1 - price))^2
        # At 50% (where up/down markets sit), this is ~1.56%.
        # We set self.fee to the midpoint estimate; the scoring methods
        # use polymarket_fee_curve() for price-specific calculations.
        # See: https://docs.polymarket.com/polymarket-learn/trading/maker-rebates-program
        self.fee = _cfg.BTC_ETH_HF_FEE_ESTIMATE  # default ~1.56% at 50% probability
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
    ) -> list[Opportunity]:
        """Detect arbitrage opportunities across BTC/ETH high-freq markets.

        1. Filter markets to find BTC/ETH high-freq candidates.
        2. Update price history for each candidate.
        3. Run the dynamic strategy selector on each candidate.
        4. Return all detected opportunities.
        """
        if not _cfg.BTC_ETH_HF_ENABLED:
            return []

        opportunities: list[Opportunity] = []

        candidates = self._find_candidates(markets, prices)
        if not candidates:
            logger.debug("BtcEthHighFreq: no BTC/ETH high-freq candidates found")
            return opportunities

        logger.info(f"BtcEthHighFreq: found {len(candidates)} candidate market(s) — evaluating sub-strategies")

        for candidate in candidates:
            # Update price history
            self._update_price_history(candidate)

            # Dynamic strategy selection
            selected, all_scores = self._select_sub_strategy(candidate)
            if selected is None:
                reasons = " | ".join(f"{s.strategy.value}: {s.reason}" for s in all_scores)
                logger.debug(
                    f"BtcEthHighFreq: no viable sub-strategy for market "
                    f"{candidate.market.id} ({candidate.asset} {candidate.timeframe}, "
                    f"yes={candidate.yes_price:.3f} no={candidate.no_price:.3f} "
                    f"liq=${candidate.market.liquidity:.0f}) — {reasons}"
                )
                continue

            scores_str = ", ".join(f"{s.strategy.value}={s.score:.1f}" for s in all_scores)
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

        logger.info(f"BtcEthHighFreq: scan complete — {len(opportunities)} opportunity(ies) found")
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
        """Return a supported timeframe if a match is detected."""
        text = f"{market.question} {market.slug}".lower()

        # Try slug regex first — now allows words between asset and timeframe
        slug_text = f"{market.slug} {market.question}".lower()
        slug_match = _SLUG_REGEX.search(slug_text)
        if slug_match:
            raw_tf = slug_match.group(2).lower().replace("-", "").replace("_", "")
            # Check 15m before 5m (15m contains "5m" substring)
            if "15" in raw_tf or "quarter" in raw_tf:
                return "15min"
            if "4h" in raw_tf or "240" in raw_tf:
                return "4hr"
            if "5m" in raw_tf or (raw_tf.startswith("5") and "15" not in raw_tf):
                return "5min"
            if "1h" in raw_tf or "60" in raw_tf or "hourly" in raw_tf:
                return "1hr"

        # Fallback: question-text keyword matching (broadened patterns)
        # Check 15m before 5m (15m contains "5m" substring)
        for tf_key in ("15min", "5min", "1hr", "4hr"):
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
            if candidate.timeframe == "4hr":
                window = _4HR_HISTORY_WINDOW_SEC
            elif candidate.timeframe == "1hr":
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
        base_score = net_profit * _PURE_ARB_NET_PROFIT_SCALE

        # Bonus for high liquidity (confidence we can fill)
        liquidity = c.market.liquidity
        if liquidity >= _PURE_ARB_HIGH_LIQUIDITY_USD:
            base_score += _PURE_ARB_HIGH_LIQUIDITY_BONUS
        elif liquidity >= _PURE_ARB_MED_LIQUIDITY_USD:
            base_score += _PURE_ARB_MED_LIQUIDITY_BONUS
        elif liquidity >= _PURE_ARB_LOW_LIQUIDITY_USD:
            base_score += _PURE_ARB_LOW_LIQUIDITY_BONUS

        # Bonus for balanced prices (both sides near 0.49-0.50 = most liquid)
        balance = 1.0 - abs(c.yes_price - c.no_price)
        base_score += balance * _PURE_ARB_BALANCE_SCALE

        return SubStrategyScore(
            strategy=SubStrategy.PURE_ARB,
            score=base_score,
            reason=(f"Pure arb: combined={combined:.4f}, net_profit={net_profit:.4f}, liquidity=${liquidity:.0f}"),
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
            yes_dev = _DUMP_HEDGE_FAIR_VALUE - c.yes_price  # positive if YES dropped below 0.50
            no_dev = _DUMP_HEDGE_FAIR_VALUE - c.no_price  # positive if NO dropped below 0.50
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
                reason=(f"Insufficient dump: max drop {max_drop:.4f} < {_dump_hedge_drop_pct()} threshold"),
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
        if dumped_price < _DUMP_HEDGE_NEAR_RESOLVED_THRESHOLD:
            return SubStrategyScore(
                strategy=SubStrategy.DUMP_HEDGE,
                score=0.0,
                reason=(
                    f"Market effectively resolved ({dumped_side} at ${dumped_price:.4f} "
                    f"< ${_DUMP_HEDGE_NEAR_RESOLVED_THRESHOLD} threshold)"
                ),
            )

        fair_value = _DUMP_HEDGE_FAIR_VALUE
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
        base_score = max_drop * _DUMP_HEDGE_DROP_SCORE_SCALE
        base_score += max(ev_profit, 0) * _DUMP_HEDGE_EV_PROFIT_SCALE
        base_score += guaranteed_component * _DUMP_HEDGE_GUARANTEED_SCALE

        # Volatility bonus: higher volatility means more dump-hedge opportunities
        volatility = history.recent_volatility() if (history and history.has_data) else 0.0
        base_score += volatility * _DUMP_HEDGE_VOLATILITY_SCALE

        # Liquidity matters: need to be able to fill quickly
        if c.market.liquidity >= _DUMP_HEDGE_HIGH_LIQUIDITY_USD:
            base_score += _DUMP_HEDGE_HIGH_LIQUIDITY_BONUS
        elif c.market.liquidity < _DUMP_HEDGE_LOW_LIQUIDITY_USD:
            base_score *= _DUMP_HEDGE_LOW_LIQUIDITY_PENALTY

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

    def _calculate_dynamic_limit_prices(self, c: HighFreqCandidate) -> tuple[float, float]:
        """Calculate optimal limit order prices based on current market state.

        Instead of fixed $0.45-$0.47, adjusts based on:
        - Current market prices (bid closer to current for fill probability)
        - Liquidity level (thinner book = can be more aggressive)
        - Required minimum profit margin
        """
        # Base targets
        min_target = _LIMIT_ORDER_MIN_TARGET

        # Start from the aggressive end
        base_price = _LIMIT_ORDER_TARGET_LOW

        # Adjust based on liquidity: thinner book = can be more aggressive
        if c.market.liquidity < _LIMIT_VERY_THIN_LIQUIDITY_USD:
            base_price = min_target  # Very thin, be aggressive
        elif c.market.liquidity < _LIMIT_THIN_LIQUIDITY_USD:
            base_price = 0.44
        elif c.market.liquidity < _LIMIT_MODERATE_LIQUIDITY_USD:
            base_price = _LIMIT_ORDER_TARGET_LOW
        else:
            base_price = 0.46  # Moderate book, bid closer to fill

        # Ensure combined still profits: need combined < 1.0 - fee
        # If we bid base_price on both sides, combined = 2 * base_price
        # Need: 2 * base_price + fee < 1.0
        limit_fee = polymarket_fee_curve(_LIMIT_ORDER_FEE_REFERENCE_PRICE)
        max_combined = 1.0 - limit_fee - _LIMIT_ORDER_MIN_PROFIT_MARGIN
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
        base_score = _LIMIT_ORDER_BASE_SCORE

        # Lower liquidity = more opportunity for limit fills
        if liquidity < _LIMIT_VERY_THIN_LIQUIDITY_USD:
            base_score += _LIMIT_VERY_THIN_BONUS
        elif liquidity < _LIMIT_THIN_LIQUIDITY_USD:
            base_score += _LIMIT_THIN_BONUS
        else:
            base_score += _LIMIT_MODERATE_BONUS

        # Near-half prices suggest a freshly opened market (ideal)
        if both_near_half:
            base_score += _LIMIT_NEAR_HALF_BONUS

        # Market age proxy: very low volume = likely just opened
        if c.market.volume < _LIMIT_VERY_NEW_VOLUME_USD:
            base_score += _LIMIT_VERY_NEW_BONUS
        elif c.market.volume < _LIMIT_NEW_VOLUME_USD:
            base_score += _LIMIT_NEW_BONUS
        elif c.market.volume < _NEW_MARKET_VOLUME_THRESHOLD:
            base_score += 5.0  # Relatively new

        # Bonus for the expected profit
        base_score += target_profit * _LIMIT_PROFIT_SCORE_SCALE

        # Penalty: if the market already has significant volume, limits are
        # less likely to fill at our targets
        if c.market.volume > _LIMIT_SIGNIFICANT_VOLUME_USD:
            base_score *= _LIMIT_SIGNIFICANT_VOLUME_PENALTY

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
        model_up = 0.50 + (trend * _DIRECTIONAL_TREND_SCALE)
        model_up = max(_DIRECTIONAL_MODEL_PROB_MIN, min(_DIRECTIONAL_MODEL_PROB_MAX, model_up))
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
                    end_ts = datetime.fromisoformat(end_str.replace("Z", "+00:00")).timestamp()
                remaining_secs = max(0, end_ts - time.time())
            except (ValueError, AttributeError):
                pass

        remaining_min = (remaining_secs / 60.0) if remaining_secs else 15.0

        if remaining_min > _DIRECTIONAL_EARLY_PHASE_MINUTES:
            phase = "EARLY"
            min_edge = _DIRECTIONAL_EARLY_MIN_EDGE
            score_multiplier = _DIRECTIONAL_EARLY_SCORE_MULT
        elif remaining_min > _DIRECTIONAL_MID_PHASE_MINUTES:
            phase = "MID"
            min_edge = _DIRECTIONAL_MID_MIN_EDGE
            score_multiplier = _DIRECTIONAL_MID_SCORE_MULT
        else:
            phase = "LATE"
            min_edge = _DIRECTIONAL_LATE_MIN_EDGE
            score_multiplier = _DIRECTIONAL_LATE_SCORE_MULT

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
        score = min(best_edge * _DIRECTIONAL_EDGE_SCORE_SCALE * score_multiplier, _DIRECTIONAL_MAX_SCORE)

        # Bonus for trend strength
        if trend_strength > _DIRECTIONAL_STRONG_TREND_THRESHOLD:
            score += _DIRECTIONAL_STRONG_TREND_BONUS
        elif trend_strength > _DIRECTIONAL_MODERATE_TREND_THRESHOLD:
            score += _DIRECTIONAL_MODERATE_TREND_BONUS

        # LATE phase bonus: we're most confident here
        if phase == "LATE":
            score += _DIRECTIONAL_LATE_PHASE_BONUS

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
    ) -> Optional[Opportunity]:
        """Turn a scored sub-strategy into an Opportunity via the base
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
    ) -> Optional[Opportunity]:
        """Generate opportunity for sub-strategy A: Pure Arbitrage."""
        market = c.market
        yes_price = params["yes_price"]
        no_price = params["no_price"]
        combined = params["combined_cost"]

        positions = self._build_both_sides_positions(market, yes_price, no_price)

        opp = self.create_opportunity(
            title=(f"BTC/ETH HF Pure Arb: {c.asset} {c.timeframe} ({market.question[:40]})"),
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
    ) -> Optional[Opportunity]:
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
        params["yes_price"]
        params["no_price"]
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
                    "note": (f"Buy dumped side ({dumped_side} dropped {drop_amount:.4f} to {dumped_price:.4f})"),
                },
            ]

        opp = self.create_opportunity(
            title=(f"BTC/ETH HF Dump-Hedge: {c.asset} {c.timeframe} ({dumped_side} dropped to {dumped_price:.2f})"),
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
                f"Directional bet: profit depends on {dumped_side} recovering toward fair value ($0.50)",
            )
        return opp

    def _generate_pre_placed_limits(
        self,
        c: HighFreqCandidate,
        params: dict,
    ) -> Optional[Opportunity]:
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
            min_liquidity_hard=0.0,  # New markets may have $0 liquidity
            min_position_size=0.0,  # Limit orders, not market orders
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
    ) -> Optional[Opportunity]:
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
            title=(f"BTC/ETH HF Directional: {c.asset} {c.timeframe} ({side} edge {edge:.1%})"),
            description=(
                f"Directional {side} bet on {c.asset} {c.timeframe} market. "
                f"Model: {model_up:.0%} Up / {1 - model_up:.0%} Down. "
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
        opp: Opportunity,
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

    # ------------------------------------------------------------------
    # Evaluate / Should-Exit  (unified strategy interface)
    # ------------------------------------------------------------------

    def evaluate(self, signal: Any, context: dict[str, Any]) -> StrategyDecision:
        """Full crypto high-frequency evaluation with multi-mode regime system,
        direction guardrails, component edges, and asset/timeframe filtering.

        Ported from BaseCryptoTimeframeStrategy.evaluate().
        """
        params = context.get("params") or {}
        payload = signal_payload(signal)

        # --- Core thresholds ---
        min_edge = to_float(params.get("min_edge_percent", 3.0), 3.0)
        min_conf = to_confidence(params.get("min_confidence", 0.45), 0.45)
        base_size = to_float(params.get("base_size_usd", 25.0), 25.0)
        max_size = max(1.0, to_float(params.get("max_size_usd", base_size * 3.0), base_size * 3.0))

        # --- Direction guardrail parameters ---
        guardrail_enabled = to_bool(params.get("direction_guardrail_enabled"), True)
        guardrail_prob_floor = max(
            0.5,
            min(1.0, to_float(params.get("direction_guardrail_prob_floor", 0.55), 0.55)),
        )
        guardrail_price_floor = max(
            0.5,
            min(1.0, to_float(params.get("direction_guardrail_price_floor", 0.80), 0.80)),
        )
        guardrail_regimes = _normalize_regime_scope(params.get("direction_guardrail_regimes", ["mid", "closing"]))
        if not guardrail_regimes:
            guardrail_regimes = {"mid", "closing"}

        # --- Mode selection ---
        requested_mode = _normalize_mode(params.get("strategy_mode") or params.get("mode"))
        direction = str(getattr(signal, "direction", "") or "").strip().lower()
        regime = _normalize_regime(payload.get("regime"))

        # --- Asset / timeframe extraction ---
        signal_asset = _normalize_asset(payload.get("asset") or payload.get("coin") or payload.get("symbol"))
        signal_timeframe = _normalize_timeframe(
            payload.get("timeframe") or payload.get("cadence") or payload.get("interval")
        )

        # --- Asset/timeframe include+exclude filtering ---
        include_assets = _normalize_scope(
            params.get("include_assets"),
            _normalize_asset,
        )
        exclude_assets = _normalize_scope(
            _first_present(
                params.get("exclude_assets"),
            ),
            _normalize_asset,
        )
        include_timeframes = _normalize_scope(
            params.get("include_timeframes"),
            _normalize_timeframe,
        )
        exclude_timeframes = _normalize_scope(
            _first_present(
                params.get("exclude_timeframes"),
            ),
            _normalize_timeframe,
        )
        asset_in_scope = (not include_assets) or (bool(signal_asset) and signal_asset in include_assets)
        asset_not_excluded = not (bool(signal_asset) and signal_asset in exclude_assets)
        asset_scope_ok = asset_in_scope and asset_not_excluded
        # Unified strategy handles all timeframes — no fixed expected_timeframe.
        # The strategy_timeframe check passes when no single timeframe is enforced.
        strategy_timeframe_ok = True
        timeframe_in_scope = (not include_timeframes) or (
            bool(signal_timeframe) and signal_timeframe in include_timeframes
        )
        timeframe_not_excluded = not (bool(signal_timeframe) and signal_timeframe in exclude_timeframes)
        timeframe_scope_ok = timeframe_in_scope and timeframe_not_excluded

        # --- Active mode resolution ---
        dominant_mode = _normalize_mode(payload.get("dominant_strategy"))
        active_mode = dominant_mode if requested_mode == "auto" and dominant_mode != "auto" else requested_mode
        if active_mode == "auto":
            active_mode = "directional"

        # --- Source / origin checks ---
        source_ok = str(getattr(signal, "source", "")) == "crypto"
        signal_type = str(getattr(signal, "signal_type", "") or "").strip().lower()
        origin_ok = str(
            payload.get("strategy_origin") or ""
        ).strip().lower() == "crypto_worker" or signal_type.startswith("crypto_worker")

        # --- Edge / confidence ---
        edge = max(0.0, to_float(getattr(signal, "edge_percent", 0.0), 0.0))
        confidence = to_confidence(getattr(signal, "confidence", 0.0), 0.0)
        mode_edge = _get_component_edge(payload, direction, active_mode)
        net_edge = _get_net_edge(payload, direction, edge)

        # --- Oracle / guardrail data ---
        model_prob_yes = max(0.0, min(1.0, to_float(payload.get("model_prob_yes"), 0.5)))
        model_prob_no = max(0.0, min(1.0, to_float(payload.get("model_prob_no"), 0.5)))
        up_price = max(0.0, min(1.0, to_float(payload.get("up_price"), 0.5)))
        down_price = max(0.0, min(1.0, to_float(payload.get("down_price"), 0.5)))
        oracle_available = bool(payload.get("oracle_available")) or payload.get("oracle_delta_pct") is not None

        # --- Regime-aware required thresholds ---
        required_edge = min_edge * _EDGE_MODE_FACTORS.get(regime, {}).get(active_mode, 1.0)
        required_conf = min_conf * _CONF_MODE_FACTORS.get(active_mode, 1.0) * _REGIME_CONF_FACTORS.get(regime, 1.0)

        # --- Direction guardrail ---
        guardrail_blocked = False
        guardrail_detail = "disabled"
        if guardrail_enabled:
            guardrail_detail = "guardrail conditions not met"
            if oracle_available and regime in guardrail_regimes:
                if (
                    direction == "buy_no"
                    and model_prob_yes >= guardrail_prob_floor
                    and up_price >= guardrail_price_floor
                ):
                    guardrail_blocked = True
                    guardrail_detail = (
                        f"blocked contrarian buy_no: model_prob_yes={model_prob_yes:.3f} up_price={up_price:.3f}"
                    )
                elif (
                    direction == "buy_yes"
                    and model_prob_no >= guardrail_prob_floor
                    and down_price >= guardrail_price_floor
                ):
                    guardrail_blocked = True
                    guardrail_detail = (
                        f"blocked contrarian buy_yes: model_prob_no={model_prob_no:.3f} down_price={down_price:.3f}"
                    )

        # --- Adaptive edge gating ---
        edge_for_gate = min(edge, mode_edge) if mode_edge > 0.0 else edge

        # --- Decision checks ---
        checks = [
            DecisionCheck("source", "Crypto source", source_ok, detail="Requires crypto worker signals."),
            DecisionCheck(
                "signal_origin",
                "Dedicated crypto worker signal",
                origin_ok,
                detail="Legacy scanner crypto opportunities are unsupported.",
            ),
            DecisionCheck(
                "asset_scope",
                "Asset include/exclude scope",
                asset_scope_ok,
                detail=(
                    f"asset={signal_asset or 'unknown'} "
                    f"include={','.join(include_assets) or 'all'} "
                    f"exclude={','.join(exclude_assets) or 'none'}"
                ),
            ),
            DecisionCheck(
                "timeframe_scope",
                "Cadence include/exclude scope",
                timeframe_scope_ok,
                detail=(
                    f"timeframe={signal_timeframe or 'unknown'} "
                    f"include={','.join(include_timeframes) or 'all'} "
                    f"exclude={','.join(exclude_timeframes) or 'none'}"
                ),
            ),
            DecisionCheck(
                "strategy_timeframe",
                "Strategy timeframe",
                strategy_timeframe_ok,
                detail=(f"observed={signal_timeframe or 'unknown'} (unified strategy accepts all timeframes)"),
            ),
            DecisionCheck(
                "direction_guardrail",
                "Direction guardrail",
                not guardrail_blocked,
                detail=guardrail_detail,
            ),
            DecisionCheck(
                "edge",
                "Edge threshold",
                edge_for_gate >= required_edge,
                score=edge_for_gate,
                detail=f"mode={active_mode} regime={regime} min={required_edge:.2f}",
            ),
            DecisionCheck(
                "confidence",
                "Confidence threshold",
                confidence >= required_conf,
                score=confidence,
                detail=f"min={required_conf:.2f}",
            ),
            DecisionCheck(
                "execution_edge",
                "Execution-adjusted edge",
                net_edge > 0.0,
                score=net_edge,
                detail="Requires positive post-penalty edge.",
            ),
        ]

        if requested_mode != "auto":
            checks.append(
                DecisionCheck(
                    "mode_signal",
                    "Requested strategy mode has signal",
                    mode_edge > 0.0,
                    score=mode_edge,
                    detail=f"requested={requested_mode}",
                )
            )

        # --- Build shared payload dict (28+ fields) ---
        decision_payload: dict[str, Any] = {
            "requested_mode": requested_mode,
            "active_mode": active_mode,
            "dominant_mode": dominant_mode,
            "regime": regime,
            "edge": edge,
            "mode_edge": mode_edge,
            "net_edge": net_edge,
            "confidence": confidence,
            "required_edge": required_edge,
            "required_confidence": required_conf,
            "asset": signal_asset,
            "timeframe": signal_timeframe,
            "direction_guardrail": {
                "enabled": guardrail_enabled,
                "blocked": guardrail_blocked,
                "oracle_available": oracle_available,
                "regime": regime,
                "regimes": sorted(guardrail_regimes),
                "prob_floor": guardrail_prob_floor,
                "price_floor": guardrail_price_floor,
                "model_prob_yes": model_prob_yes,
                "model_prob_no": model_prob_no,
                "up_price": up_price,
                "down_price": down_price,
            },
            "include_assets": include_assets,
            "exclude_assets": exclude_assets,
            "include_timeframes": include_timeframes,
            "exclude_timeframes": exclude_timeframes,
        }

        score = (edge_for_gate * 0.7) + (confidence * 30.0)

        if not all(c.passed for c in checks):
            return StrategyDecision(
                decision="skipped",
                reason="Crypto worker filters not met",
                score=score,
                checks=checks,
                payload=decision_payload,
            )

        # --- Position sizing ---
        edge_boost = 1.0 + max(0.0, edge_for_gate - required_edge) / 30.0
        conf_boost = 0.8 + (confidence * 0.8)
        size = (
            base_size
            * _MODE_SIZE_FACTORS.get(active_mode, 1.0)
            * _REGIME_SIZE_FACTORS.get(regime, 1.0)
            * edge_boost
            * conf_boost
        )
        size = max(1.0, min(max_size, size))

        return StrategyDecision(
            decision="selected",
            reason=f"Crypto {active_mode} setup validated ({regime})",
            score=score,
            size_usd=size,
            checks=checks,
            payload=decision_payload,
        )

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        """Crypto: tight TP/SL with short max hold."""
        if market_state.get("is_resolved"):
            return self.default_exit_check(position, market_state)
        config = getattr(position, "config", None) or {}
        config = dict(config)
        config.setdefault("take_profit_pct", 8.0)
        config.setdefault("stop_loss_pct", 5.0)
        config.setdefault("trailing_stop_pct", 3.0)
        config.setdefault("max_hold_minutes", 60)
        position.config = config
        return self.default_exit_check(position, market_state)

    # ------------------------------------------------------------------
    # Event-driven detection (crypto_update from crypto worker)
    # ------------------------------------------------------------------

    async def on_event(self, event: DataEvent) -> list[Opportunity]:
        if event.event_type != "crypto_update":
            return []

        markets = event.payload.get("markets") or []
        if not markets:
            return []

        return self._detect_from_crypto_markets(markets)

    @staticmethod
    def _market_from_crypto_dict(d: dict) -> Market:
        """Deserialize a crypto worker market dict into a typed Market object.

        Crypto worker dicts use non-standard keys (condition_id, up_price,
        down_price, end_time). We map them to Market fields immediately at
        the DataEvent boundary so every downstream code path sees only typed
        objects — never raw dicts.
        """
        market_id = str(d.get("condition_id") or d.get("id") or "")
        up_price = float(d.get("up_price") or 0.0)
        down_price = float(d.get("down_price") or 0.0)
        liquidity = max(0.0, float(d.get("liquidity") or 0.0))
        slug = d.get("slug") or market_id
        question = d.get("question") or slug

        end_date = None
        end_time_raw = d.get("end_time")
        if isinstance(end_time_raw, str) and end_time_raw.strip():
            try:
                from datetime import datetime as _dt

                end_date = _dt.fromisoformat(end_time_raw.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        return Market(
            id=market_id,
            condition_id=market_id,
            question=question,
            slug=slug,
            # yes_price = up_price (market-implied prob of going Up)
            outcome_prices=[up_price, down_price],
            liquidity=liquidity,
            end_date=end_date,
            platform="polymarket",
        )

    def _detect_from_crypto_markets(self, markets: list[dict]) -> list[Opportunity]:
        """Replicate the multi-strategy signal logic from the former emit_crypto_market_signals.

        For each crypto market dict, compute regime-weighted edge across
        directional / pure_arb / rebalance sub-strategies and return
        Opportunity objects for markets with positive net edge.
        """
        opportunities: list[Opportunity] = []

        for market in markets:
            market_id = str(market.get("condition_id") or market.get("id") or "")
            if not market_id:
                continue
            typed_market = self._market_from_crypto_dict(market)
            asset = self._detect_asset(typed_market) or _normalize_asset(
                market.get("asset") or market.get("symbol") or market.get("coin")
            )
            timeframe = _normalize_timeframe(market.get("timeframe"))
            if not timeframe:
                timeframe = _normalize_timeframe(self._detect_timeframe(typed_market))

            up_price = self._float(market.get("up_price"))
            down_price = self._float(market.get("down_price"))
            if up_price is None or down_price is None:
                continue
            if not (0.0 <= up_price <= 1.0 and 0.0 <= down_price <= 1.0):
                continue

            price_to_beat = self._float(market.get("price_to_beat"))
            oracle_price = self._float(market.get("oracle_price"))
            has_oracle = price_to_beat is not None and price_to_beat > 0 and oracle_price is not None

            timeframe_seconds = self._timeframe_seconds(market.get("timeframe"))
            seconds_left = self._float(market.get("seconds_left"))
            if seconds_left is None:
                end_time = market.get("end_time")
                if isinstance(end_time, str) and end_time.strip():
                    try:
                        from datetime import datetime as _dt, timezone as _tz

                        parsed = _dt.fromisoformat(end_time.replace("Z", "+00:00"))
                        seconds_left = max(0.0, (parsed - _dt.now(_tz.utc)).total_seconds())
                    except Exception:
                        seconds_left = float(timeframe_seconds)
                else:
                    seconds_left = float(timeframe_seconds)

            regime = self._crypto_regime(seconds_left, timeframe_seconds)

            # Directional edge (oracle-based)
            if has_oracle and price_to_beat is not None and oracle_price is not None:
                diff_pct = ((oracle_price - price_to_beat) / price_to_beat) * 100.0
                time_ratio = clamp(seconds_left / float(max(1, timeframe_seconds)), 0.08, 1.0)
                directional_scale = max(0.08, 0.50 * time_ratio)
                directional_z = clamp(diff_pct / directional_scale, -60.0, 60.0)
                model_prob_yes = clamp(1.0 / (1.0 + math.exp(-directional_z)), 0.03, 0.97)
                model_prob_no = 1.0 - model_prob_yes
                directional_yes = max(0.0, (model_prob_yes - up_price) * 100.0)
                directional_no = max(0.0, (model_prob_no - down_price) * 100.0)
            else:
                diff_pct = 0.0
                directional_yes = 0.0
                directional_no = 0.0

            # Pure arb edge
            combined = up_price + down_price
            underround = max(0.0, 1.0 - combined)
            pure_arb_yes = underround * 100.0
            pure_arb_no = underround * 100.0

            # Rebalance edge
            neutrality = clamp(1.0 - (abs(diff_pct) / 0.45), 0.0, 1.0)
            rebalance_yes = max(0.0, (0.5 - up_price) * 100.0) * neutrality
            rebalance_no = max(0.0, (0.5 - down_price) * 100.0) * neutrality

            # Regime weights
            if has_oracle:
                weights = self._regime_weights(regime)
            else:
                weights = self._regime_weights_without_oracle(regime)

            gross_yes = (
                (directional_yes * weights["directional"])
                + (pure_arb_yes * weights["pure_arb"])
                + (rebalance_yes * weights["rebalance"])
            )
            gross_no = (
                (directional_no * weights["directional"])
                + (pure_arb_no * weights["pure_arb"])
                + (rebalance_no * weights["rebalance"])
            )

            # Execution penalties
            spread = clamp(self._float(market.get("spread")) or 0.0, 0.0, 0.10)
            liquidity = max(0.0, self._float(market.get("liquidity")) or 0.0)
            fees_enabled = bool(market.get("fees_enabled", False))

            fee_penalty = 0.45 if fees_enabled else 0.25
            spread_penalty = spread * 100.0 * 0.35
            liquidity_scale = clamp(liquidity / 250000.0, 0.0, 1.0)
            regime_slippage_factor = 1.1 if regime == "closing" else 1.0
            slippage_penalty = (1.35 - (0.95 * liquidity_scale)) * regime_slippage_factor
            execution_penalty = fee_penalty + spread_penalty + slippage_penalty

            net_yes = gross_yes - execution_penalty
            net_no = gross_no - execution_penalty
            direction = "buy_yes" if net_yes >= net_no else "buy_no"
            entry_price = up_price if direction == "buy_yes" else down_price
            edge_percent = net_yes if direction == "buy_yes" else net_no

            if edge_percent < 1.0:
                continue

            # Confidence
            edge_gap = abs(net_yes - net_no)
            selected_components = (
                {"directional": directional_yes, "pure_arb": pure_arb_yes, "rebalance": rebalance_yes}
                if direction == "buy_yes"
                else {"directional": directional_no, "pure_arb": pure_arb_no, "rebalance": rebalance_no}
            )
            weighted_components = {k: selected_components[k] * weights[k] for k in selected_components}
            dominant_strategy = max(weighted_components, key=lambda k: weighted_components[k])
            dominant_weighted_edge = weighted_components[dominant_strategy]

            confidence = clamp(
                0.32
                + clamp(edge_percent / 20.0, 0.0, 0.35)
                + clamp(edge_gap / 18.0, 0.0, 0.12)
                + clamp(dominant_weighted_edge / max(1.0, edge_percent) * 0.08, 0.0, 0.08),
                0.05,
                0.97,
            )
            if not has_oracle:
                confidence = clamp(confidence * 0.75, 0.05, 0.85)

            side = "YES" if direction == "buy_yes" else "NO"
            slug = market.get("slug") or market_id

            opp = self.create_opportunity(
                title=f"Crypto HF: {slug} {side}",
                description=(
                    f"{regime} regime, {dominant_strategy} dominant | edge={edge_percent:.1f}%, conf={confidence:.0%}"
                ),
                total_cost=entry_price,
                expected_payout=entry_price + (edge_percent / 100.0),
                markets=[typed_market],
                positions=[
                    {
                        "action": "BUY",
                        "outcome": side,
                        "price": entry_price,
                        # Carry crypto-specific context through positions payload
                        "_crypto_context": {
                            "signal_version": "crypto_worker_v2",
                            "signal_family": "crypto_multistrategy",
                            "strategy_origin": "crypto_worker",
                            "selected_direction": direction,
                            "asset": asset,
                            "timeframe": timeframe,
                            "regime": regime,
                            "oracle_available": has_oracle,
                            "dominant_strategy": dominant_strategy,
                            "execution_penalty_percent": round(execution_penalty, 6),
                        },
                    }
                ],
                is_guaranteed=False,
                skip_fee_model=True,  # Crypto uses its own execution penalty model
                custom_roi_percent=edge_percent,
                custom_risk_score=1.0 - confidence,
                confidence=confidence,
            )
            if opp is not None:
                # Attach crypto-specific risk factors (bypassed by custom_risk_score)
                opp.risk_factors = [
                    f"Crypto {regime} regime",
                    f"Dominant: {dominant_strategy} ({dominant_weighted_edge:.1f}%)",
                    f"Oracle: {'available' if has_oracle else 'unavailable'}",
                ]
                opp.strategy_context = {
                    "source_key": "crypto",
                    "strategy_slug": self.strategy_type,
                    "strategy_origin": "crypto_worker",
                    "asset": asset,
                    "timeframe": timeframe,
                    "regime": regime,
                    "selected_direction": direction,
                    "oracle_available": has_oracle,
                    "dominant_strategy": dominant_strategy,
                    "execution_penalty_percent": round(execution_penalty, 6),
                }
                opportunities.append(opp)

        return opportunities

    @staticmethod
    def _float(value: object) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    @staticmethod
    def _timeframe_seconds(value: object) -> int:
        tf = str(value or "").strip().lower()
        if tf in {"5m", "5min"}:
            return 300
        if tf in {"15m", "15min"}:
            return 900
        if tf in {"1h", "1hr", "60m"}:
            return 3600
        if tf in {"4h", "4hr", "240m"}:
            return 14400
        return 900

    @staticmethod
    def _crypto_regime(seconds_left: float, timeframe_seconds: int) -> str:
        denom = float(max(1, timeframe_seconds))
        ratio = clamp(seconds_left / denom, 0.0, 1.0)
        if ratio > 0.67:
            return "opening"
        if ratio < 0.33:
            return "closing"
        return "mid"

    @staticmethod
    def _regime_weights(regime: str) -> dict[str, float]:
        if regime == "opening":
            return {"directional": 0.65, "pure_arb": 0.25, "rebalance": 0.10}
        if regime == "closing":
            return {"directional": 0.35, "pure_arb": 0.20, "rebalance": 0.45}
        return {"directional": 0.50, "pure_arb": 0.25, "rebalance": 0.25}

    @staticmethod
    def _regime_weights_without_oracle(regime: str) -> dict[str, float]:
        if regime == "opening":
            return {"directional": 0.0, "pure_arb": 0.60, "rebalance": 0.40}
        if regime == "closing":
            return {"directional": 0.0, "pure_arb": 0.45, "rebalance": 0.55}
        return {"directional": 0.0, "pure_arb": 0.55, "rebalance": 0.45}

    # ------------------------------------------------------------------
    # Platform gate hooks
    # ------------------------------------------------------------------

    def on_blocked(self, signal, reason: str, context: dict) -> None:
        logger.info("%s: signal blocked — %s (market=%s)", self.name, reason, getattr(signal, "market_id", "?"))

    def on_size_capped(self, original_size: float, capped_size: float, reason: str) -> None:
        logger.info("%s: size capped $%.0f → $%.0f — %s", self.name, original_size, capped_size, reason)
