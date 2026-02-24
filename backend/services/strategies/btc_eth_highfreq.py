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
from services.strategy_sdk import StrategySDK
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


_TIMEFRAME_PARAM_SUFFIXES: dict[str, tuple[str, ...]] = {
    "5m": ("5m", "5min"),
    "15m": ("15m", "15min"),
    "1h": ("1h", "1hr", "60m"),
    "4h": ("4h", "4hr", "240m"),
}


def _timeframe_override(params: dict[str, Any], base_key: str, timeframe: str) -> Any:
    normalized_tf = _normalize_timeframe(timeframe)
    if not normalized_tf:
        return None
    for suffix in _TIMEFRAME_PARAM_SUFFIXES.get(normalized_tf, (normalized_tf,)):
        key = f"{base_key}_{suffix}"
        if key in params:
            return params.get(key)
    return None


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


def _parse_datetime_utc(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts <= 0:
            return None
        if ts > 1_000_000_000_000:
            ts /= 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            return None
    text = str(value or "").strip()
    if not text:
        return None
    numeric = to_float(text)
    if numeric is not None and numeric > 0:
        return _parse_datetime_utc(numeric)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _to_iso_utc(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


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
    supports_entry_take_profit_exit = True
    default_open_order_timeout_seconds = 20.0

    quality_filter_overrides = QualityFilterOverrides(
        min_roi=1.0,
        max_resolution_months=0.1,
    )

    default_config = dict(StrategySDK.crypto_highfreq_scope_defaults())

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
        # Runtime anti-churn controls used by evaluate().
        self._edge_first_seen_ms: dict[str, int] = {}
        self._last_selected_at_ms_by_market: dict[str, int] = {}

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
        live_market = context.get("live_market")
        if not isinstance(live_market, dict):
            live_market = payload.get("live_market")
        if not isinstance(live_market, dict):
            live_market = {}

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
        signal_asset = _normalize_asset(
            _first_present(
                live_market.get("asset"),
                live_market.get("coin"),
                live_market.get("symbol"),
                payload.get("asset"),
                payload.get("coin"),
                payload.get("symbol"),
            )
        )
        signal_timeframe = _normalize_timeframe(
            _first_present(
                live_market.get("timeframe"),
                live_market.get("cadence"),
                live_market.get("interval"),
                payload.get("timeframe"),
                payload.get("cadence"),
                payload.get("interval"),
            )
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
        enforce_hard_timeframe_allowlist = to_bool(
            params.get("enforce_hard_timeframe_allowlist"),
            True,
        )
        hard_allowed_timeframes = _normalize_scope(
            params.get("hard_allowed_timeframes"),
            _normalize_timeframe,
        )
        if not hard_allowed_timeframes:
            hard_allowed_timeframes = {"5m", "15m", "1h"}
        timeframe_hardening_ok = (not enforce_hard_timeframe_allowlist) or (
            bool(signal_timeframe) and signal_timeframe in hard_allowed_timeframes
        )

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

        live_window_required = to_bool(params.get("live_window_required"), True)
        signal_is_live_raw = _first_present(live_market.get("is_live"), payload.get("is_live"))
        signal_is_live = signal_is_live_raw if isinstance(signal_is_live_raw, bool) else None
        signal_is_current_raw = _first_present(live_market.get("is_current"), payload.get("is_current"))
        signal_is_current = signal_is_current_raw if isinstance(signal_is_current_raw, bool) else None
        signal_seconds_left = to_float(
            _first_present(
                live_market.get("seconds_left"),
                payload.get("seconds_left"),
            ),
            -1.0,
        )
        signal_end_time = str(
            _first_present(
                live_market.get("market_end_time"),
                live_market.get("end_time"),
                payload.get("end_time"),
            )
            or ""
        ).strip()
        if signal_is_live is None and signal_end_time:
            try:
                parsed_end = datetime.fromisoformat(signal_end_time.replace("Z", "+00:00"))
                signal_is_live = parsed_end.timestamp() > time.time()
            except Exception:
                signal_is_live = None
        if signal_is_live is None and signal_seconds_left >= 0:
            signal_is_live = signal_seconds_left > 0
        if signal_is_current is None:
            signal_is_current = signal_is_live
        live_window_ok = (not live_window_required) or bool(signal_is_live and signal_is_current)

        max_live_context_age_seconds = max(
            0.1,
            to_float(params.get("max_live_context_age_seconds", 3.0), 3.0),
        )
        live_context_fetched_at = _parse_datetime_utc(
            _first_present(
                live_market.get("fetched_at"),
                payload.get("live_market_fetched_at"),
            )
        )
        live_context_age_seconds: Optional[float] = None
        if live_context_fetched_at is not None:
            live_context_age_seconds = max(
                0.0,
                (datetime.now(timezone.utc) - live_context_fetched_at.astimezone(timezone.utc)).total_seconds(),
            )
        live_context_fresh_ok = (
            live_context_age_seconds is None or live_context_age_seconds <= max_live_context_age_seconds
        )

        signal_timestamp_used = _parse_datetime_utc(
            _first_present(
                live_market.get("signal_updated_at"),
                live_market.get("updated_at"),
                live_market.get("fetched_at"),
                payload.get("signal_updated_at"),
                payload.get("signal_emitted_at"),
                payload.get("live_market_fetched_at"),
                getattr(signal, "updated_at", None),
                getattr(signal, "created_at", None),
            )
        )
        signal_created_at = _parse_datetime_utc(getattr(signal, "created_at", None))
        signal_updated_at = _parse_datetime_utc(getattr(signal, "updated_at", None))
        signal_age_seconds: Optional[float] = None
        if isinstance(signal_timestamp_used, datetime):
            signal_ts_utc = (
                signal_timestamp_used
                if signal_timestamp_used.tzinfo
                else signal_timestamp_used.replace(tzinfo=timezone.utc)
            )
            signal_age_seconds = max(
                0.0,
                (datetime.now(timezone.utc) - signal_ts_utc.astimezone(timezone.utc)).total_seconds(),
            )
        max_signal_age_seconds_cfg = self._float(
            _timeframe_override(params, "max_signal_age_seconds", signal_timeframe)
        )
        if max_signal_age_seconds_cfg is None:
            max_signal_age_seconds_cfg = to_float(params.get("max_signal_age_seconds", 20.0), 20.0)
        max_signal_age_seconds = max(0.1, float(max_signal_age_seconds_cfg))
        signal_fresh_ok = signal_age_seconds is None or signal_age_seconds <= max_signal_age_seconds

        market_data_age_ms = self._float(
            _first_present(
                live_market.get("market_data_age_ms"),
                payload.get("market_data_age_ms"),
            )
        )
        if market_data_age_ms is None:
            observed_at = _parse_datetime_utc(
                _first_present(
                    live_market.get("source_observed_at"),
                    payload.get("source_observed_at"),
                    live_market.get("fetched_at"),
                    payload.get("live_market_fetched_at"),
                    payload.get("signal_updated_at"),
                    getattr(signal, "updated_at", None),
                    getattr(signal, "created_at", None),
                )
            )
            if observed_at is not None:
                market_data_age_ms = max(
                    0.0,
                    (datetime.now(timezone.utc) - observed_at.astimezone(timezone.utc)).total_seconds() * 1000.0,
                )
        max_market_data_age_ms_cfg = self._float(
            _timeframe_override(params, "max_market_data_age_ms", signal_timeframe)
        )
        if max_market_data_age_ms_cfg is None:
            max_market_data_age_ms_cfg = to_float(params.get("max_market_data_age_ms", 900.0), 900.0)
        max_market_data_age_ms = max(50.0, float(max_market_data_age_ms_cfg))
        require_market_data_age_for_sources = {
            str(item or "").strip().lower()
            for item in _as_list(_first_present(params.get("require_market_data_age_for_sources"), ["crypto"]))
            if str(item or "").strip()
        }
        require_market_data_age = (
            str(getattr(signal, "source", "") or "").strip().lower() in require_market_data_age_for_sources
        )
        market_data_freshness_enforced = to_bool(params.get("enforce_market_data_freshness"), True)
        market_data_fresh_ok = (
            (not market_data_freshness_enforced)
            or (market_data_age_ms is not None and market_data_age_ms <= max_market_data_age_ms)
            or (market_data_age_ms is None and not require_market_data_age)
        )

        default_min_seconds_by_timeframe: dict[str, float] = {
            "5m": 45.0,
            "15m": 180.0,
            "1h": 360.0,
            "4h": 900.0,
        }
        timeframe_specific_floor = self._float(
            _timeframe_override(params, "min_seconds_left_for_entry", signal_timeframe)
        )
        global_min_seconds = self._float(params.get("min_seconds_left_for_entry"))
        min_seconds_left_for_entry = (
            max(0.0, timeframe_specific_floor)
            if timeframe_specific_floor is not None
            else (
                max(0.0, global_min_seconds)
                if global_min_seconds is not None
                else default_min_seconds_by_timeframe.get(signal_timeframe, 0.0)
            )
        )
        entry_window_ok = signal_seconds_left < 0 or signal_seconds_left >= float(min_seconds_left_for_entry)

        if signal_seconds_left >= 0 and signal_timeframe:
            regime = self._crypto_regime(signal_seconds_left, self._timeframe_seconds(signal_timeframe))

        live_window_detail = (
            f"required={live_window_required} "
            f"is_live={signal_is_live if signal_is_live is not None else 'unknown'} "
            f"is_current={signal_is_current if signal_is_current is not None else 'unknown'} "
            f"seconds_left={signal_seconds_left if signal_seconds_left >= 0 else 'unknown'}"
        )
        signal_freshness_detail = (
            f"age={signal_age_seconds:.1f}s max={max_signal_age_seconds:.1f}s "
            f"timestamp={_to_iso_utc(signal_timestamp_used)}"
            if signal_age_seconds is not None
            else "signal timestamp unavailable"
        )
        market_data_freshness_detail = (
            (
                f"age_ms={market_data_age_ms:.0f} max={max_market_data_age_ms:.0f} "
                f"source={str(getattr(signal, 'source', '') or '').strip().lower() or 'unknown'}"
            )
            if market_data_age_ms is not None
            else (
                "market_data_age unavailable but optional"
                if not require_market_data_age
                else "market_data_age unavailable and required"
            )
        )
        entry_window_detail = (
            f"seconds_left={signal_seconds_left:.1f} required>={min_seconds_left_for_entry:.1f}"
            if signal_seconds_left >= 0
            else "seconds_left unavailable"
        )
        live_context_freshness_detail = (
            f"age={live_context_age_seconds:.1f}s max={max_live_context_age_seconds:.1f}s"
            if live_context_age_seconds is not None
            else "live context timestamp unavailable"
        )

        # --- Edge / confidence ---
        edge = max(0.0, to_float(getattr(signal, "edge_percent", 0.0), 0.0))
        confidence = to_confidence(getattr(signal, "confidence", 0.0), 0.0)
        mode_edge = _get_component_edge(payload, direction, active_mode)
        net_edge = _get_net_edge(payload, direction, edge)

        signal_liquidity_usd = self._float(
            _first_present(
                live_market.get("liquidity_usd"),
                live_market.get("liquidity"),
                payload.get("liquidity_usd"),
                payload.get("liquidity"),
                getattr(signal, "liquidity", None),
            )
        )
        min_liquidity_usd = max(0.0, to_float(params.get("min_liquidity_usd", 300.0), 300.0))
        if regime == "opening":
            min_liquidity_opening = self._float(
                _timeframe_override(params, "min_liquidity_usd_opening", signal_timeframe)
            )
            if min_liquidity_opening is None:
                min_liquidity_opening = self._float(params.get("min_liquidity_usd_opening"))
            if min_liquidity_opening is not None:
                min_liquidity_usd = max(min_liquidity_usd, max(0.0, float(min_liquidity_opening)))
        liquidity_ok = signal_liquidity_usd is None or signal_liquidity_usd >= min_liquidity_usd
        liquidity_detail = (
            f"liquidity={signal_liquidity_usd:.0f} min={min_liquidity_usd:.0f}"
            if signal_liquidity_usd is not None
            else "liquidity unavailable"
        )

        signal_volume_usd = self._float(
            _first_present(
                live_market.get("volume_usd"),
                live_market.get("volume"),
                payload.get("volume_usd"),
                payload.get("volume"),
                getattr(signal, "volume", None),
            )
        )
        min_volume_usd = max(
            0.0,
            StrategySDK.crypto_highfreq_min_volume_usd(
                params,
                timeframe=signal_timeframe,
                regime=regime,
                active_mode=active_mode,
            ),
        )
        if min_volume_usd <= 0.0:
            volume_ok = True
        elif signal_volume_usd is None:
            volume_ok = False
        else:
            volume_ok = signal_volume_usd >= min_volume_usd
        volume_detail = (
            f"volume={signal_volume_usd:.0f} min={min_volume_usd:.0f}"
            if signal_volume_usd is not None
            else f"volume unavailable min={min_volume_usd:.0f}"
        )

        direction_policy_ok, direction_policy_detail = StrategySDK.crypto_highfreq_direction_allowed(
            params,
            regime=regime,
            active_mode=active_mode,
            direction=direction,
        )

        signal_spread = self._float(
            _first_present(
                live_market.get("spread"),
                payload.get("spread"),
                payload.get("market_spread"),
            )
        )
        if signal_spread is not None:
            signal_spread = max(0.0, min(1.0, signal_spread))
        max_spread_pct = max(0.0, min(1.0, to_float(params.get("max_spread_pct", 0.06), 0.06)))
        spread_ok = signal_spread is None or signal_spread <= max_spread_pct
        spread_detail = (
            f"spread={signal_spread:.4f} max={max_spread_pct:.4f}"
            if signal_spread is not None
            else "spread unavailable"
        )

        history_summary = live_market.get("history_summary")
        if not isinstance(history_summary, dict):
            history_summary = payload.get("history_summary")
        if not isinstance(history_summary, dict):
            history_summary = {}

        move_5m_pct = self._float(
            _first_present(
                (
                    (history_summary.get("move_5m") or {}).get("percent")
                    if isinstance(history_summary.get("move_5m"), dict)
                    else None
                ),
                payload.get("move_5m_percent"),
                payload.get("move_5m_pct"),
            )
        )
        move_30m_pct = self._float(
            _first_present(
                (
                    (history_summary.get("move_30m") or {}).get("percent")
                    if isinstance(history_summary.get("move_30m"), dict)
                    else None
                ),
                payload.get("move_30m_percent"),
                payload.get("move_30m_pct"),
            )
        )
        move_2h_pct = self._float(
            _first_present(
                (
                    (history_summary.get("move_2h") or {}).get("percent")
                    if isinstance(history_summary.get("move_2h"), dict)
                    else None
                ),
                payload.get("move_2h_percent"),
                payload.get("move_2h_pct"),
            )
        )

        recent_move_zscore: Optional[float] = None
        if move_5m_pct is not None:
            move_scale = max(
                0.15,
                abs(move_30m_pct or 0.0) / 2.0,
                abs(move_2h_pct or 0.0) / 3.0,
            )
            recent_move_zscore = abs(move_5m_pct) / move_scale if move_scale > 0 else None
        max_recent_move_zscore_for_entry = max(
            0.2,
            to_float(params.get("max_recent_move_zscore_for_entry", 2.0), 2.0),
        )
        recent_move_ok = recent_move_zscore is None or recent_move_zscore <= max_recent_move_zscore_for_entry
        recent_move_detail = (
            f"z={recent_move_zscore:.2f} max={max_recent_move_zscore_for_entry:.2f} "
            f"move_5m={move_5m_pct if move_5m_pct is not None else 'n/a'} "
            f"move_30m={move_30m_pct if move_30m_pct is not None else 'n/a'} "
            f"move_2h={move_2h_pct if move_2h_pct is not None else 'n/a'}"
            if recent_move_zscore is not None
            else "recent move z-score unavailable"
        )

        spread_widening_bps = self._float(
            _first_present(
                live_market.get("spread_widening_bps"),
                live_market.get("spread_delta_bps"),
                payload.get("spread_widening_bps"),
                payload.get("spread_delta_bps"),
            )
        )
        max_spread_widening_bps = max(
            0.0,
            to_float(params.get("max_spread_widening_bps", 20.0), 20.0),
        )
        spread_widening_ok = spread_widening_bps is None or spread_widening_bps <= max_spread_widening_bps
        spread_widening_detail = (
            f"widening_bps={spread_widening_bps:.2f} max={max_spread_widening_bps:.2f}"
            if spread_widening_bps is not None
            else "spread widening unavailable"
        )

        raw_orderbook_imbalance = self._float(
            _first_present(
                live_market.get("orderbook_imbalance"),
                live_market.get("book_imbalance"),
                live_market.get("imbalance"),
                payload.get("orderbook_imbalance"),
                payload.get("book_imbalance"),
                payload.get("imbalance"),
            )
        )
        orderbook_imbalance = None
        if raw_orderbook_imbalance is not None:
            normalized = abs(raw_orderbook_imbalance)
            if normalized > 1.0 and normalized <= 100.0:
                normalized /= 100.0
            orderbook_imbalance = min(1.0, max(0.0, normalized))
        max_orderbook_imbalance = min(
            1.0,
            max(0.5, to_float(params.get("max_orderbook_imbalance", 0.88), 0.88)),
        )
        orderbook_imbalance_ok = orderbook_imbalance is None or orderbook_imbalance <= max_orderbook_imbalance
        orderbook_imbalance_detail = (
            f"imbalance={orderbook_imbalance:.3f} max={max_orderbook_imbalance:.3f}"
            if orderbook_imbalance is not None
            else "orderbook imbalance unavailable"
        )

        # --- Oracle / guardrail data ---
        model_prob_yes = max(0.0, min(1.0, to_float(payload.get("model_prob_yes"), 0.5)))
        model_prob_no = max(0.0, min(1.0, to_float(payload.get("model_prob_no"), 0.5)))
        up_price = max(0.0, min(1.0, to_float(payload.get("up_price"), 0.5)))
        down_price = max(0.0, min(1.0, to_float(payload.get("down_price"), 0.5)))
        oracle_available = bool(payload.get("oracle_available")) or payload.get("oracle_delta_pct") is not None
        oracle_age_seconds = self._float(
            _first_present(
                live_market.get("oracle_age_seconds"),
                payload.get("oracle_age_seconds"),
            )
        )
        if oracle_age_seconds is None:
            oracle_updated_at_ms = self._float(
                _first_present(
                    live_market.get("oracle_updated_at_ms"),
                    payload.get("oracle_updated_at_ms"),
                )
            )
            if oracle_updated_at_ms is not None and oracle_updated_at_ms > 0:
                if oracle_updated_at_ms > 1_000_000_000_000:
                    oracle_updated_at_ms /= 1000.0
                oracle_age_seconds = max(0.0, time.time() - oracle_updated_at_ms)
        max_oracle_age_seconds = max(1.0, to_float(params.get("max_oracle_age_seconds", 12.0), 12.0))
        require_oracle_for_directional = to_bool(params.get("require_oracle_for_directional"), True)
        oracle_required = require_oracle_for_directional and active_mode in {"directional", "rebalance"}
        if oracle_required:
            oracle_fresh_ok = (
                oracle_available and oracle_age_seconds is not None and oracle_age_seconds <= max_oracle_age_seconds
            )
        else:
            oracle_fresh_ok = (
                not oracle_available or oracle_age_seconds is None or oracle_age_seconds <= max_oracle_age_seconds
            )
        oracle_freshness_detail = (
            (
                f"available={oracle_available} age={oracle_age_seconds:.1f}s "
                f"max={max_oracle_age_seconds:.1f}s required={oracle_required}"
            )
            if oracle_age_seconds is not None
            else f"available={oracle_available} age=unknown required={oracle_required}"
        )

        # --- Regime-aware required thresholds ---
        required_edge = min_edge * _EDGE_MODE_FACTORS.get(regime, {}).get(active_mode, 1.0)
        required_conf = min_conf * _CONF_MODE_FACTORS.get(active_mode, 1.0) * _REGIME_CONF_FACTORS.get(regime, 1.0)

        entry_price_for_execution = self._float(
            _first_present(
                getattr(signal, "entry_price", None),
                payload.get("entry_price"),
                payload.get("selected_price"),
                live_market.get("live_selected_price"),
                live_market.get("signal_entry_price"),
            )
        )
        if entry_price_for_execution is None or entry_price_for_execution <= 0.0:
            if direction == "buy_yes":
                entry_price_for_execution = self._float(
                    _first_present(
                        live_market.get("yes_price"),
                        payload.get("up_price"),
                        payload.get("yes_price"),
                    )
                )
            elif direction == "buy_no":
                entry_price_for_execution = self._float(
                    _first_present(
                        live_market.get("no_price"),
                        payload.get("down_price"),
                        payload.get("no_price"),
                    )
                )
        if entry_price_for_execution is not None and entry_price_for_execution > 0.0:
            entry_price_for_execution = clamp(entry_price_for_execution, 0.0, 1.0)
        else:
            entry_price_for_execution = None

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
        now_ms = int(time.time() * 1000.0)
        market_id = str(getattr(signal, "market_id", "") or "").strip()
        edge_tracker_key = f"{market_id}|{direction}|{active_mode}" if market_id else ""
        min_edge_persistence_ms = max(
            0,
            int(to_float(params.get("min_edge_persistence_ms", 1400), 1400.0)),
        )
        edge_persistence_elapsed_ms: Optional[int] = None
        edge_persistence_ok = True
        if edge_tracker_key and edge_for_gate >= required_edge and min_edge_persistence_ms > 0:
            first_seen = self._edge_first_seen_ms.get(edge_tracker_key)
            if first_seen is None:
                self._edge_first_seen_ms[edge_tracker_key] = now_ms
                first_seen = now_ms
            edge_persistence_elapsed_ms = max(0, int(now_ms - first_seen))
            edge_persistence_ok = edge_persistence_elapsed_ms >= min_edge_persistence_ms
        elif edge_tracker_key and edge_for_gate < required_edge:
            self._edge_first_seen_ms.pop(edge_tracker_key, None)
        edge_persistence_detail = (
            f"elapsed_ms={edge_persistence_elapsed_ms if edge_persistence_elapsed_ms is not None else 0} "
            f"required_ms={min_edge_persistence_ms}"
            if min_edge_persistence_ms > 0
            else "edge persistence disabled"
        )

        force_disable_reentry_cooldown = to_bool(params.get("force_disable_reentry_cooldown"), False)
        if force_disable_reentry_cooldown:
            reentry_cooldown_seconds = 0.0
        else:
            default_reentry_cooldown_seconds = max(
                0.0,
                to_float(getattr(self, "config", {}).get("reentry_cooldown_seconds_per_market"), 0.0),
            )
            reentry_cooldown_seconds = max(
                0.0,
                to_float(
                    params.get("reentry_cooldown_seconds_per_market", default_reentry_cooldown_seconds),
                    default_reentry_cooldown_seconds,
                ),
            )
        reentry_cooldown_ms = int(reentry_cooldown_seconds * 1000.0)
        reentry_cooldown_elapsed_ms: Optional[int] = None
        reentry_cooldown_ok = True
        if market_id and reentry_cooldown_ms > 0:
            last_selected_at_ms = self._last_selected_at_ms_by_market.get(market_id)
            if last_selected_at_ms is not None:
                reentry_cooldown_elapsed_ms = max(0, int(now_ms - last_selected_at_ms))
                reentry_cooldown_ok = reentry_cooldown_elapsed_ms >= reentry_cooldown_ms
        reentry_cooldown_detail = (
            f"elapsed_ms={reentry_cooldown_elapsed_ms if reentry_cooldown_elapsed_ms is not None else 'none'} "
            f"required_ms={reentry_cooldown_ms}"
            if reentry_cooldown_ms > 0
            else "re-entry cooldown disabled"
        )

        trader_context = context.get("trader")
        risk_limits_context = (
            trader_context.get("risk_limits")
            if isinstance(trader_context, dict) and isinstance(trader_context.get("risk_limits"), dict)
            else {}
        )
        max_trade_notional_hint = self._float(risk_limits_context.get("max_trade_notional_usd"))
        edge_boost_for_checks = 1.0 + max(0.0, edge_for_gate - required_edge) / 30.0
        conf_boost_for_checks = 0.8 + (confidence * 0.8)
        estimated_size_for_checks = (
            base_size
            * _MODE_SIZE_FACTORS.get(active_mode, 1.0)
            * _REGIME_SIZE_FACTORS.get(regime, 1.0)
            * edge_boost_for_checks
            * conf_boost_for_checks
        )
        estimated_size_for_checks = max(1.0, min(max_size, estimated_size_for_checks))
        if max_trade_notional_hint is not None and max_trade_notional_hint > 0.0:
            estimated_size_for_checks = min(estimated_size_for_checks, max_trade_notional_hint)

        min_order_size_usd = StrategySDK.resolve_min_order_size_usd(
            params,
            mode=context.get("mode"),
            fallback=1.0,
        )
        default_exit_ratio_floor_by_timeframe = {
            "5m": 0.38,
            "15m": 0.34,
            "1h": 0.30,
            "4h": 0.26,
        }
        default_entry_exit_ratio_floor = default_exit_ratio_floor_by_timeframe.get(signal_timeframe, 0.38)
        if active_mode == "rebalance":
            default_entry_exit_ratio_floor -= 0.08
        elif active_mode == "directional":
            default_entry_exit_ratio_floor -= 0.03
        if regime == "closing":
            default_entry_exit_ratio_floor -= 0.05
        default_entry_exit_ratio_floor = clamp(default_entry_exit_ratio_floor, 0.16, 0.90)
        configured_entry_exit_ratio_floor = self._float(
            _first_present(
                _timeframe_override(params, "entry_executable_exit_ratio_floor", signal_timeframe),
                params.get("entry_executable_exit_ratio_floor_closing") if regime == "closing" else None,
                params.get("entry_executable_exit_ratio_floor"),
            )
        )
        executable_exit_ratio_floor = (
            clamp(configured_entry_exit_ratio_floor, 0.05, 0.95)
            if configured_entry_exit_ratio_floor is not None
            else default_entry_exit_ratio_floor
        )
        required_size_for_exitability = min_order_size_usd / max(0.01, executable_exit_ratio_floor)
        entry_exitability_ok = estimated_size_for_checks + 1e-9 >= required_size_for_exitability
        entry_exitability_detail = (
            f"est_size={estimated_size_for_checks:.2f} required>={required_size_for_exitability:.2f} "
            f"ratio_floor={executable_exit_ratio_floor:.2f}"
        )

        directional_entry_price_floor = max(
            0.0,
            to_float(params.get("directional_min_entry_price_floor", 0.10), 0.10),
        )
        rebalance_entry_price_floor = max(
            directional_entry_price_floor,
            to_float(params.get("rebalance_min_entry_price_floor", 0.16), 0.16),
        )
        default_entry_price_floor = (
            rebalance_entry_price_floor if active_mode == "rebalance" else directional_entry_price_floor
        )
        if regime == "closing":
            default_entry_price_floor = max(default_entry_price_floor, 0.05)
        entry_price_floor = clamp(
            to_float(params.get("min_entry_price_floor"), default_entry_price_floor),
            0.0,
            0.99,
        )
        entry_price_floor_ok = entry_price_for_execution is None or entry_price_for_execution >= entry_price_floor
        entry_price_floor_detail = (
            f"entry_price={entry_price_for_execution:.4f} floor={entry_price_floor:.4f}"
            if entry_price_for_execution is not None
            else "entry_price unavailable"
        )
        directional_entry_price_ceiling = clamp(
            to_float(params.get("directional_max_entry_price_ceiling", 0.75), 0.75),
            0.01,
            1.0,
        )
        rebalance_entry_price_ceiling = clamp(
            to_float(params.get("rebalance_max_entry_price_ceiling", 0.70), 0.70),
            0.01,
            1.0,
        )
        default_entry_price_ceiling = (
            rebalance_entry_price_ceiling if active_mode == "rebalance" else directional_entry_price_ceiling
        )
        if regime == "opening":
            default_entry_price_ceiling = min(
                default_entry_price_ceiling,
                0.68 if active_mode == "rebalance" else 0.72,
            )
        entry_price_ceiling = clamp(
            to_float(params.get("max_entry_price_ceiling"), default_entry_price_ceiling),
            0.01,
            1.0,
        )
        if entry_price_ceiling < entry_price_floor:
            entry_price_ceiling = entry_price_floor
        entry_price_ceiling_ok = entry_price_for_execution is None or entry_price_for_execution <= entry_price_ceiling
        entry_price_ceiling_detail = (
            f"entry_price={entry_price_for_execution:.4f} ceiling={entry_price_ceiling:.4f}"
            if entry_price_for_execution is not None
            else "entry_price unavailable"
        )

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
                "live_window",
                "Current live window only",
                live_window_ok,
                detail=live_window_detail,
            ),
            DecisionCheck(
                "live_context_freshness",
                "Live context freshness",
                live_context_fresh_ok,
                score=live_context_age_seconds,
                detail=live_context_freshness_detail,
            ),
            DecisionCheck(
                "signal_freshness",
                "Signal freshness",
                signal_fresh_ok,
                score=signal_age_seconds,
                detail=signal_freshness_detail,
            ),
            DecisionCheck(
                "market_data_freshness",
                "Market data freshness",
                market_data_fresh_ok,
                score=market_data_age_ms,
                detail=market_data_freshness_detail,
            ),
            DecisionCheck(
                "entry_window",
                "Minimum seconds-left entry window",
                entry_window_ok,
                score=signal_seconds_left if signal_seconds_left >= 0 else None,
                detail=entry_window_detail,
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
                "timeframe_hardening",
                "Hardened timeframe allowlist",
                timeframe_hardening_ok,
                detail=(
                    f"timeframe={signal_timeframe or 'unknown'} "
                    f"allowed={','.join(hard_allowed_timeframes) or 'none'} "
                    f"enforced={enforce_hard_timeframe_allowlist}"
                ),
            ),
            DecisionCheck(
                "liquidity",
                "Minimum liquidity",
                liquidity_ok,
                score=signal_liquidity_usd,
                detail=liquidity_detail,
            ),
            DecisionCheck(
                "volume",
                "Minimum volume",
                volume_ok,
                score=signal_volume_usd,
                detail=volume_detail,
            ),
            DecisionCheck(
                "spread",
                "Maximum spread",
                spread_ok,
                score=signal_spread,
                detail=spread_detail,
            ),
            DecisionCheck(
                "spread_widening",
                "Spread widening guard",
                spread_widening_ok,
                score=spread_widening_bps,
                detail=spread_widening_detail,
            ),
            DecisionCheck(
                "orderbook_imbalance",
                "Orderbook imbalance guard",
                orderbook_imbalance_ok,
                score=orderbook_imbalance,
                detail=orderbook_imbalance_detail,
            ),
            DecisionCheck(
                "recent_move_zscore",
                "Recent move z-score guard",
                recent_move_ok,
                score=recent_move_zscore,
                detail=recent_move_detail,
            ),
            DecisionCheck(
                "oracle_freshness",
                "Oracle freshness",
                oracle_fresh_ok,
                score=oracle_age_seconds,
                detail=oracle_freshness_detail,
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
                "direction_policy",
                "Direction policy",
                direction_policy_ok,
                detail=direction_policy_detail,
            ),
            DecisionCheck(
                "edge",
                "Edge threshold",
                edge_for_gate >= required_edge,
                score=edge_for_gate,
                detail=f"mode={active_mode} regime={regime} min={required_edge:.2f}",
            ),
            DecisionCheck(
                "edge_persistence",
                "Edge persistence",
                edge_persistence_ok,
                score=edge_persistence_elapsed_ms,
                detail=edge_persistence_detail,
            ),
            DecisionCheck(
                "confidence",
                "Confidence threshold",
                confidence >= required_conf,
                score=confidence,
                detail=f"min={required_conf:.2f}",
            ),
            DecisionCheck(
                "entry_price_floor",
                "Entry price floor",
                entry_price_floor_ok,
                score=entry_price_for_execution,
                detail=entry_price_floor_detail,
            ),
            DecisionCheck(
                "entry_price_ceiling",
                "Entry price ceiling",
                entry_price_ceiling_ok,
                score=entry_price_for_execution,
                detail=entry_price_ceiling_detail,
            ),
            DecisionCheck(
                "entry_exitability",
                "Entry exitability under stress",
                entry_exitability_ok,
                score=estimated_size_for_checks,
                detail=entry_exitability_detail,
            ),
            DecisionCheck(
                "reentry_cooldown",
                "Re-entry cooldown",
                reentry_cooldown_ok,
                score=reentry_cooldown_elapsed_ms,
                detail=reentry_cooldown_detail,
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
            "live_window_required": live_window_required,
            "is_live": signal_is_live,
            "is_current": signal_is_current,
            "seconds_left": signal_seconds_left if signal_seconds_left >= 0 else None,
            "end_time": signal_end_time or None,
            "live_context_age_seconds": live_context_age_seconds,
            "max_live_context_age_seconds": float(max_live_context_age_seconds),
            "min_seconds_left_for_entry": float(min_seconds_left_for_entry),
            "signal_age_seconds": signal_age_seconds,
            "max_signal_age_seconds": float(max_signal_age_seconds),
            "market_data_age_ms": market_data_age_ms,
            "max_market_data_age_ms": float(max_market_data_age_ms),
            "enforce_market_data_freshness": bool(market_data_freshness_enforced),
            "require_market_data_age_for_sources": sorted(require_market_data_age_for_sources),
            "signal_timestamp_used": _to_iso_utc(signal_timestamp_used),
            "signal_created_at": _to_iso_utc(signal_created_at),
            "signal_updated_at": _to_iso_utc(signal_updated_at),
            "liquidity_usd": signal_liquidity_usd,
            "min_liquidity_usd": float(min_liquidity_usd),
            "volume_usd": signal_volume_usd,
            "min_volume_usd": float(min_volume_usd),
            "spread": signal_spread,
            "max_spread_pct": float(max_spread_pct),
            "spread_widening_bps": spread_widening_bps,
            "max_spread_widening_bps": float(max_spread_widening_bps),
            "orderbook_imbalance": orderbook_imbalance,
            "max_orderbook_imbalance": float(max_orderbook_imbalance),
            "move_5m_percent": move_5m_pct,
            "move_30m_percent": move_30m_pct,
            "move_2h_percent": move_2h_pct,
            "recent_move_zscore": recent_move_zscore,
            "max_recent_move_zscore_for_entry": float(max_recent_move_zscore_for_entry),
            "oracle_age_seconds": oracle_age_seconds,
            "max_oracle_age_seconds": float(max_oracle_age_seconds),
            "require_oracle_for_directional": bool(require_oracle_for_directional),
            "edge_persistence_elapsed_ms": edge_persistence_elapsed_ms,
            "min_edge_persistence_ms": int(min_edge_persistence_ms),
            "force_disable_reentry_cooldown": bool(force_disable_reentry_cooldown),
            "reentry_cooldown_elapsed_ms": reentry_cooldown_elapsed_ms,
            "reentry_cooldown_seconds_per_market": float(reentry_cooldown_seconds),
            "entry_price": entry_price_for_execution,
            "entry_price_floor": float(entry_price_floor),
            "entry_price_ceiling": float(entry_price_ceiling),
            "entry_exitability_ratio_floor": float(executable_exit_ratio_floor),
            "entry_exitability_required_size_usd": float(required_size_for_exitability),
            "entry_exitability_estimated_size_usd": float(estimated_size_for_checks),
            "entry_exitability_min_order_size_usd": float(min_order_size_usd),
            "max_trade_notional_hint_usd": max_trade_notional_hint,
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
            "direction_policy": {
                "allowed": bool(direction_policy_ok),
                "detail": direction_policy_detail,
            },
            "include_assets": include_assets,
            "exclude_assets": exclude_assets,
            "include_timeframes": include_timeframes,
            "exclude_timeframes": exclude_timeframes,
            "enforce_hard_timeframe_allowlist": bool(enforce_hard_timeframe_allowlist),
            "hard_allowed_timeframes": sorted(hard_allowed_timeframes),
            "live_market_context": {
                "available": bool(live_market.get("available")),
                "fetched_at": live_market.get("fetched_at"),
                "selected_token_id": live_market.get("selected_token_id"),
                "market_end_time": live_market.get("market_end_time"),
                "seconds_left": live_market.get("seconds_left"),
                "is_live": live_market.get("is_live"),
                "is_current": live_market.get("is_current"),
            },
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
        if market_id:
            self._last_selected_at_ms_by_market[market_id] = now_ms
        if edge_tracker_key:
            self._edge_first_seen_ms.pop(edge_tracker_key, None)

        return StrategyDecision(
            decision="selected",
            reason=f"Crypto {active_mode} setup validated ({regime})",
            score=score,
            size_usd=size,
            checks=checks,
            payload=decision_payload,
        )

    def _evaluate_local_exit(self, position: Any, market_state: dict) -> dict[str, Any]:
        state = market_state if isinstance(market_state, dict) else {}
        base_config = getattr(position, "config", None)
        config = dict(base_config) if isinstance(base_config, dict) else {}
        defaults = StrategySDK.crypto_highfreq_scope_defaults()

        strategy_context = getattr(position, "strategy_context", None)
        if isinstance(strategy_context, dict):
            context_payload = strategy_context
        else:
            context_payload = {}
            try:
                setattr(position, "strategy_context", context_payload)
            except Exception:
                pass

        timeframe = _normalize_timeframe(
            context_payload.get("timeframe") or context_payload.get("cadence") or context_payload.get("interval")
        )

        default_stop_loss_activation_by_timeframe = {
            "5m": 45.0,
            "15m": 120.0,
            "1h": 300.0,
            "4h": 900.0,
        }
        default_trailing_activation_profit_by_timeframe = {
            "5m": 4.0,
            "15m": 6.0,
            "1h": 8.0,
            "4h": 10.0,
        }
        default_min_hold_by_timeframe = {
            "5m": 0.25,
            "15m": 0.5,
            "1h": 1.0,
            "4h": 2.0,
        }
        default_max_hold_by_timeframe = {
            "5m": 15.0,
            "15m": 45.0,
            "1h": 120.0,
            "4h": 360.0,
        }
        default_rapid_exit_window_by_timeframe = {
            "5m": 1.0,
            "15m": 2.0,
            "1h": 6.0,
            "4h": 15.0,
        }
        default_underwater_dwell_minutes_by_timeframe = {
            "5m": 0.75,
            "15m": 2.0,
            "1h": 6.0,
            "4h": 18.0,
        }
        default_underwater_recovery_ratio_by_timeframe = {
            "5m": 0.30,
            "15m": 0.35,
            "1h": 0.40,
            "4h": 0.45,
        }
        default_underwater_rebound_pct_by_timeframe = {
            "5m": 0.8,
            "15m": 1.2,
            "1h": 1.8,
            "4h": 2.4,
        }
        default_underwater_exit_fade_pct_by_timeframe = {
            "5m": 0.35,
            "15m": 0.45,
            "1h": 0.6,
            "4h": 0.8,
        }
        default_force_flatten_seconds_left_by_timeframe = {
            "5m": to_float(defaults.get("force_flatten_seconds_left_5m"), 30.0),
            "15m": to_float(defaults.get("force_flatten_seconds_left_15m"), 75.0),
            "1h": to_float(defaults.get("force_flatten_seconds_left_1h"), 240.0),
            "4h": to_float(defaults.get("force_flatten_seconds_left_4h"), 600.0),
        }
        default_underwater_timeout_minutes_by_timeframe = {
            "5m": 2.0,
            "15m": 6.0,
            "1h": 18.0,
            "4h": 45.0,
        }
        default_executable_time_pressure_seconds_by_timeframe = {
            "5m": 40.0,
            "15m": 120.0,
            "1h": 300.0,
            "4h": 900.0,
        }

        config.setdefault("rapid_take_profit_pct", defaults.get("rapid_take_profit_pct", 10.0))
        config.setdefault("take_profit_pct", defaults.get("take_profit_pct", 8.0))
        config.setdefault("stop_loss_pct", to_float(defaults.get("stop_loss_pct"), 4.0))
        default_stop_loss_policy = str(defaults.get("stop_loss_policy") or "always").strip().lower()
        if default_stop_loss_policy in {"near_close", "near_close_only", "close_window"}:
            default_stop_loss_policy = "always"
        config.setdefault("stop_loss_policy", default_stop_loss_policy)
        config.setdefault(
            "stop_loss_activation_seconds",
            default_stop_loss_activation_by_timeframe.get(
                timeframe,
                to_float(defaults.get("stop_loss_activation_seconds"), 90.0),
            ),
        )
        config.setdefault("trailing_stop_pct", defaults.get("trailing_stop_pct", 3.0))
        config.setdefault(
            "trailing_stop_activation_profit_pct",
            default_trailing_activation_profit_by_timeframe.get(
                timeframe,
                to_float(defaults.get("trailing_stop_activation_profit_pct"), 4.0),
            ),
        )
        config.setdefault(
            "min_hold_minutes",
            default_min_hold_by_timeframe.get(timeframe, to_float(defaults.get("min_hold_minutes"), 1.0)),
        )
        config.setdefault(
            "max_hold_minutes",
            default_max_hold_by_timeframe.get(timeframe, to_float(defaults.get("max_hold_minutes"), 60.0)),
        )
        config.setdefault(
            "rapid_exit_window_minutes",
            default_rapid_exit_window_by_timeframe.get(
                timeframe,
                to_float(defaults.get("rapid_exit_window_minutes"), 2.0),
            ),
        )
        config.setdefault("rapid_exit_min_increase_pct", defaults.get("rapid_exit_min_increase_pct", 0.0))
        config.setdefault("rapid_exit_breakeven_buffer_pct", defaults.get("rapid_exit_breakeven_buffer_pct", 0.0))
        config.setdefault(
            "underwater_rebound_exit_enabled",
            to_bool(defaults.get("underwater_rebound_exit_enabled"), True),
        )
        config.setdefault(
            "underwater_dwell_minutes",
            default_underwater_dwell_minutes_by_timeframe.get(
                timeframe,
                to_float(defaults.get("underwater_dwell_minutes"), 2.5),
            ),
        )
        config.setdefault(
            "underwater_recovery_ratio_min",
            default_underwater_recovery_ratio_by_timeframe.get(
                timeframe,
                to_float(defaults.get("underwater_recovery_ratio_min"), 0.35),
            ),
        )
        config.setdefault(
            "underwater_rebound_pct_min",
            default_underwater_rebound_pct_by_timeframe.get(
                timeframe,
                to_float(defaults.get("underwater_rebound_pct_min"), 1.2),
            ),
        )
        config.setdefault(
            "underwater_exit_fade_pct",
            default_underwater_exit_fade_pct_by_timeframe.get(
                timeframe,
                to_float(defaults.get("underwater_exit_fade_pct"), 0.45),
            ),
        )
        config.setdefault(
            "underwater_timeout_minutes",
            default_underwater_timeout_minutes_by_timeframe.get(
                timeframe,
                to_float(defaults.get("underwater_timeout_minutes"), 10.0),
            ),
        )
        config.setdefault("underwater_timeout_loss_pct", defaults.get("underwater_timeout_loss_pct", 8.0))
        config.setdefault(
            "force_flatten_seconds_left",
            default_force_flatten_seconds_left_by_timeframe.get(
                timeframe,
                to_float(defaults.get("force_flatten_seconds_left"), 120.0),
            ),
        )
        config.setdefault("force_flatten_max_profit_pct", to_float(defaults.get("force_flatten_max_profit_pct"), 1.0))
        config.setdefault("force_flatten_headroom_floor", to_float(defaults.get("force_flatten_headroom_floor"), 1.15))
        config.setdefault("force_flatten_min_loss_pct", to_float(defaults.get("force_flatten_min_loss_pct"), 0.0))
        config.setdefault(
            "resolution_risk_flatten_enabled", to_bool(defaults.get("resolution_risk_flatten_enabled"), True)
        )
        config.setdefault("resolution_risk_seconds_left", to_float(defaults.get("resolution_risk_seconds_left"), 180.0))
        config.setdefault(
            "resolution_risk_max_profit_pct", to_float(defaults.get("resolution_risk_max_profit_pct"), 6.0)
        )
        config.setdefault("resolution_risk_min_loss_pct", to_float(defaults.get("resolution_risk_min_loss_pct"), 2.0))
        config.setdefault(
            "resolution_risk_min_headroom_ratio",
            to_float(defaults.get("resolution_risk_min_headroom_ratio"), 0.9),
        )
        config.setdefault(
            "resolution_risk_disable_when_take_profit_armed",
            to_bool(defaults.get("resolution_risk_disable_when_take_profit_armed"), True),
        )
        config.setdefault("executable_exit_headroom_urgent", 1.35)
        config.setdefault("executable_exit_headroom_warn", 2.0)
        config.setdefault("executable_exit_hazard_threshold", 0.62)
        config.setdefault(
            "executable_exit_time_pressure_seconds",
            default_executable_time_pressure_seconds_by_timeframe.get(timeframe, 120.0),
        )

        if timeframe:
            for key in (
                "rapid_take_profit_pct",
                "rapid_exit_window_minutes",
                "take_profit_pct",
                "stop_loss_pct",
                "stop_loss_policy",
                "stop_loss_activation_seconds",
                "trailing_stop_pct",
                "trailing_stop_activation_profit_pct",
                "min_hold_minutes",
                "max_hold_minutes",
                "underwater_dwell_minutes",
                "underwater_recovery_ratio_min",
                "underwater_rebound_pct_min",
                "underwater_exit_fade_pct",
                "underwater_timeout_minutes",
                "force_flatten_seconds_left",
                "force_flatten_max_profit_pct",
                "force_flatten_min_loss_pct",
                "resolution_risk_seconds_left",
                "resolution_risk_max_profit_pct",
                "resolution_risk_min_loss_pct",
                "resolution_risk_min_headroom_ratio",
                "executable_exit_headroom_urgent",
                "executable_exit_headroom_warn",
                "executable_exit_hazard_threshold",
                "executable_exit_time_pressure_seconds",
            ):
                override = _timeframe_override(config, key, timeframe)
                if override is not None:
                    config[key] = override
            config["timeframe"] = timeframe

        entry_price = self._float(getattr(position, "entry_price", None))
        current_price = self._float(state.get("current_price"))
        if entry_price is not None and entry_price > 0.0 and current_price is not None and current_price > 0.0:
            pnl_pct = ((current_price - entry_price) / entry_price) * 100.0
            take_profit_pct = max(0.1, to_float(config.get("take_profit_pct"), 8.0))
            rapid_take_profit_pct = max(0.1, to_float(config.get("rapid_take_profit_pct"), 10.0))
            rapid_arm_pct = max(
                0.1,
                min(
                    rapid_take_profit_pct,
                    max(take_profit_pct, rapid_take_profit_pct * 0.85),
                ),
            )
            rapid_hard_cap_pct = max(rapid_take_profit_pct + 6.0, rapid_take_profit_pct * 1.8)
            trailing_stop_pct = max(0.5, to_float(config.get("trailing_stop_pct"), 3.0))

            timeframe_backside_factor = {
                "5m": 0.85,
                "15m": 1.0,
                "1h": 1.2,
                "4h": 1.35,
            }.get(timeframe, 1.0)
            base_backside_pct = clamp(
                trailing_stop_pct * 0.58 * timeframe_backside_factor,
                0.45,
                max(0.9, trailing_stop_pct * 0.98),
            )

            rapid_window_minutes = max(0.0, to_float(config.get("rapid_exit_window_minutes"), 0.0))
            min_increase_pct = max(0.0, to_float(config.get("rapid_exit_min_increase_pct"), 0.0))
            breakeven_buffer_pct = max(0.0, to_float(config.get("rapid_exit_breakeven_buffer_pct"), 0.0))

            rapid_state_key = "_crypto_hf_rapid_exit_state"
            raw_rapid_state = context_payload.get(rapid_state_key)
            rapid_state = dict(raw_rapid_state) if isinstance(raw_rapid_state, dict) else {}
            highest_seen = self._float(getattr(position, "highest_price", None))
            tracked_peak = self._float(rapid_state.get("peak_price"))
            peak_price = entry_price
            if highest_seen is not None:
                peak_price = max(peak_price, highest_seen)
            if tracked_peak is not None:
                peak_price = max(peak_price, tracked_peak)
            peak_price = max(peak_price, current_price)
            rapid_state["peak_price"] = peak_price
            rapid_state["peak_pnl_pct"] = ((peak_price - entry_price) / entry_price) * 100.0
            rapid_state["arm_pct"] = rapid_arm_pct
            rapid_state["hard_cap_pct"] = rapid_hard_cap_pct

            age_minutes = self._float(getattr(position, "age_minutes", None))
            evaluated = bool(rapid_state.get("evaluated"))
            if not evaluated and (
                rapid_window_minutes <= 0.0 or (age_minutes is not None and age_minutes >= rapid_window_minutes)
            ):
                required_peak = entry_price * (1.0 + (min_increase_pct / 100.0))
                rapid_state["evaluated"] = True
                rapid_state["stalled"] = peak_price <= (required_peak + 1e-9)
                rapid_state["evaluated_at_minutes"] = age_minutes
                rapid_state["required_peak_price"] = required_peak
                rapid_state["window_minutes"] = rapid_window_minutes

            token_id = str(state.get("token_id") or context_payload.get("token_id") or "").strip()
            if not token_id:
                execution_plan = context_payload.get("execution_plan")
                if isinstance(execution_plan, dict):
                    legs = execution_plan.get("legs")
                    if isinstance(legs, list):
                        for leg in legs:
                            if not isinstance(leg, dict):
                                continue
                            candidate = str(leg.get("token_id") or "").strip()
                            if candidate:
                                token_id = candidate
                                break

            flow_imbalance: float | None = None
            momentum_short_pct: float | None = None
            recent_range_pct: float | None = None
            if token_id:
                price_history = StrategySDK.get_price_history(token_id, max_snapshots=24)
                mids: list[float] = []
                for snapshot in price_history:
                    if not isinstance(snapshot, dict):
                        continue
                    mid = self._float(snapshot.get("mid"))
                    if mid is None or mid <= 0.0:
                        continue
                    mids.append(float(mid))
                    if len(mids) >= 18:
                        break
                if len(mids) >= 3:
                    lookback_idx = min(len(mids) - 1, 6)
                    anchor_price = mids[lookback_idx]
                    if anchor_price > 0.0:
                        momentum_short_pct = ((mids[0] - anchor_price) / anchor_price) * 100.0
                    window = mids[: min(len(mids), 8)]
                    window_max = max(window)
                    window_min = min(window)
                    if window_max > 0.0:
                        recent_range_pct = ((window_max - window_min) / window_max) * 100.0
                raw_imbalance = self._float(StrategySDK.get_buy_sell_imbalance(token_id, lookback_seconds=20.0))
                if raw_imbalance is not None:
                    flow_imbalance = clamp(raw_imbalance, -1.0, 1.0)

            dynamic_backside_pct = base_backside_pct
            if recent_range_pct is not None:
                dynamic_backside_pct *= clamp(1.0 + (recent_range_pct / 16.0), 0.85, 1.35)
            if flow_imbalance is not None:
                dynamic_backside_pct *= clamp(1.0 + (flow_imbalance * 0.30), 0.65, 1.35)
            if momentum_short_pct is not None:
                dynamic_backside_pct *= clamp(1.0 + (momentum_short_pct / 5.0), 0.60, 1.25)
            dynamic_backside_pct = clamp(
                dynamic_backside_pct,
                0.45,
                max(1.2, trailing_stop_pct * 0.95),
            )

            drawdown_from_peak_pct = 0.0
            if peak_price > 0.0:
                drawdown_from_peak_pct = max(0.0, ((peak_price - current_price) / peak_price) * 100.0)

            min_order_size_usd = max(0.01, to_float(state.get("min_order_size_usd"), 1.0))
            filled_size = self._float(getattr(position, "filled_size", None))
            if filled_size is None or filled_size <= 0.0:
                position_notional = self._float(getattr(position, "notional_usd", None))
                if position_notional is None:
                    position_notional = self._float(state.get("notional_usd"))
                if position_notional is not None and position_notional > 0.0:
                    filled_size = position_notional / entry_price

            min_exit_price = None
            panic_exit_price = None
            exit_notional_estimate = None
            exit_headroom_ratio = None
            if filled_size is not None and filled_size > 0.0:
                min_exit_price = min_order_size_usd / filled_size
                panic_exit_price = min_exit_price * 1.35
                exit_notional_estimate = filled_size * current_price
                if min_order_size_usd > 0.0:
                    exit_headroom_ratio = exit_notional_estimate / min_order_size_usd

            urgent_headroom_ratio = max(1.0, to_float(config.get("executable_exit_headroom_urgent"), 1.35))
            warn_headroom_ratio = max(
                urgent_headroom_ratio + 0.05, to_float(config.get("executable_exit_headroom_warn"), 2.0)
            )
            hazard_threshold = clamp(to_float(config.get("executable_exit_hazard_threshold"), 0.62), 0.0, 1.0)
            time_pressure_seconds = max(1.0, to_float(config.get("executable_exit_time_pressure_seconds"), 120.0))

            seconds_left = self._float(
                _first_present(
                    state.get("seconds_left"),
                    context_payload.get("seconds_left"),
                    context_payload.get("live_market_context", {}).get("seconds_left")
                    if isinstance(context_payload.get("live_market_context"), dict)
                    else None,
                )
            )
            time_pressure = 0.0
            if seconds_left is not None and seconds_left >= 0.0:
                time_pressure = clamp((time_pressure_seconds - seconds_left) / time_pressure_seconds, 0.0, 1.0)
            headroom_stress = 0.0
            if exit_headroom_ratio is not None:
                denom = max(0.05, warn_headroom_ratio - 1.0)
                headroom_stress = clamp((warn_headroom_ratio - exit_headroom_ratio) / denom, 0.0, 1.0)
            flow_neg = clamp(-(flow_imbalance or 0.0), 0.0, 1.0)
            momentum_neg = clamp(-((momentum_short_pct or 0.0) / 2.0), 0.0, 1.0)
            hazard_score = clamp(
                (headroom_stress * 0.55) + (flow_neg * 0.25) + (momentum_neg * 0.15) + (time_pressure * 0.05),
                0.0,
                1.0,
            )

            if exit_headroom_ratio is not None:
                dynamic_backside_pct *= clamp(exit_headroom_ratio / 3.0, 0.30, 1.0)
                dynamic_backside_pct = clamp(dynamic_backside_pct, 0.30, max(1.2, trailing_stop_pct * 0.95))

            rapid_state["token_id"] = token_id or None
            rapid_state["flow_imbalance"] = flow_imbalance
            rapid_state["momentum_short_pct"] = momentum_short_pct
            rapid_state["recent_range_pct"] = recent_range_pct
            rapid_state["dynamic_backside_pct"] = dynamic_backside_pct
            rapid_state["drawdown_from_peak_pct"] = drawdown_from_peak_pct
            rapid_state["min_exit_price"] = min_exit_price
            rapid_state["panic_exit_price"] = panic_exit_price
            rapid_state["exit_notional_estimate"] = exit_notional_estimate
            rapid_state["exit_headroom_ratio"] = exit_headroom_ratio
            rapid_state["executable_hazard_score"] = hazard_score
            rapid_state["time_pressure"] = time_pressure

            if exit_headroom_ratio is not None and exit_headroom_ratio <= urgent_headroom_ratio and pnl_pct <= 0.5:
                context_payload[rapid_state_key] = rapid_state
                return {
                    "config": config,
                    "decision": {
                        "action": "close",
                        "reason": (
                            "Executable-notional guard (headroom emergency) "
                            f"(headroom={exit_headroom_ratio:.2f}x <= {urgent_headroom_ratio:.2f}x)"
                        ),
                        "close_price": current_price,
                    },
                }

            if panic_exit_price is not None and current_price <= panic_exit_price and pnl_pct <= 0.0:
                context_payload[rapid_state_key] = rapid_state
                return {
                    "config": config,
                    "decision": {
                        "action": "close",
                        "reason": (
                            "Executable-notional guard "
                            f"({current_price:.4f} <= {panic_exit_price:.4f}, min_exit={min_exit_price:.4f})"
                        ),
                        "close_price": current_price,
                    },
                }

            if (
                exit_headroom_ratio is not None
                and hazard_score >= hazard_threshold
                and pnl_pct <= max(1.0, rapid_take_profit_pct * 0.20)
            ):
                context_payload[rapid_state_key] = rapid_state
                return {
                    "config": config,
                    "decision": {
                        "action": "close",
                        "reason": (
                            "Executable hazard exit "
                            f"(hazard={hazard_score:.2f} >= {hazard_threshold:.2f}, "
                            f"headroom={exit_headroom_ratio:.2f}x)"
                        ),
                        "close_price": current_price,
                    },
                }

            if pnl_pct >= rapid_arm_pct:
                rapid_state["armed"] = True
                if rapid_state.get("armed_at_minutes") is None:
                    rapid_state["armed_at_minutes"] = age_minutes
                if rapid_state.get("armed_price") is None:
                    rapid_state["armed_price"] = current_price
                rapid_state["armed_peak_price"] = peak_price

            should_flatten_resolution_risk, resolution_risk_detail = (
                StrategySDK.crypto_highfreq_should_flatten_resolution_risk(
                    config,
                    timeframe=timeframe,
                    seconds_left=seconds_left,
                    pnl_percent=pnl_pct,
                    exit_headroom_ratio=exit_headroom_ratio,
                    take_profit_armed=bool(rapid_state.get("armed")),
                )
            )
            if should_flatten_resolution_risk:
                context_payload[rapid_state_key] = rapid_state
                return {
                    "config": config,
                    "decision": {
                        "action": "close",
                        "reason": f"Resolution-risk flatten ({resolution_risk_detail})",
                        "close_price": current_price,
                    },
                }

            if bool(rapid_state.get("armed")):
                config["take_profit_pct"] = max(take_profit_pct, rapid_hard_cap_pct + 1.0)
                if pnl_pct >= rapid_hard_cap_pct:
                    context_payload[rapid_state_key] = rapid_state
                    return {
                        "config": config,
                        "decision": {
                            "action": "close",
                            "reason": f"Rapid hard-cap take profit ({pnl_pct:.1f}% >= {rapid_hard_cap_pct:.1f}%)",
                            "close_price": current_price,
                        },
                    }

                reversal_signal = drawdown_from_peak_pct >= (dynamic_backside_pct * 0.55) and (
                    (momentum_short_pct is not None and momentum_short_pct <= -0.25)
                    or (flow_imbalance is not None and flow_imbalance <= -0.2)
                )
                if drawdown_from_peak_pct >= dynamic_backside_pct or reversal_signal:
                    context_payload[rapid_state_key] = rapid_state
                    return {
                        "config": config,
                        "decision": {
                            "action": "close",
                            "reason": (
                                "Adaptive backside peak exit "
                                f"(drawdown={drawdown_from_peak_pct:.2f}% "
                                f"threshold={dynamic_backside_pct:.2f}%, "
                                f"flow={flow_imbalance if flow_imbalance is not None else 0.0:.2f}, "
                                f"momentum={momentum_short_pct if momentum_short_pct is not None else 0.0:.2f}%)"
                            ),
                            "close_price": current_price,
                        },
                    }

            context_payload[rapid_state_key] = rapid_state
            if bool(rapid_state.get("stalled")):
                breakeven_floor = entry_price * (1.0 + (breakeven_buffer_pct / 100.0))
                if current_price >= (breakeven_floor - 1e-9):
                    return {
                        "config": config,
                        "decision": {
                            "action": "close",
                            "reason": (
                                "Rapid window stalled without upside; exiting at breakeven+ "
                                f"({current_price:.4f} >= {breakeven_floor:.4f})"
                            ),
                            "close_price": current_price,
                        },
                    }

            underwater_state_key = "_crypto_hf_loss_recovery_state"
            if to_bool(config.get("underwater_rebound_exit_enabled"), True):
                raw_underwater_state = context_payload.get(underwater_state_key)
                underwater_state = dict(raw_underwater_state) if isinstance(raw_underwater_state, dict) else {}
                underwater_dwell_minutes = max(0.0, to_float(config.get("underwater_dwell_minutes"), 0.0))
                underwater_recovery_ratio_min = clamp(
                    to_float(config.get("underwater_recovery_ratio_min"), 0.35),
                    0.0,
                    1.0,
                )
                underwater_rebound_pct_min = max(0.0, to_float(config.get("underwater_rebound_pct_min"), 1.2))
                underwater_exit_fade_pct = max(0.0, to_float(config.get("underwater_exit_fade_pct"), 0.45))
                underwater_timeout_minutes = max(0.0, to_float(config.get("underwater_timeout_minutes"), 0.0))
                underwater_timeout_loss_pct = max(0.0, to_float(config.get("underwater_timeout_loss_pct"), 0.0))

                if current_price < (entry_price - 1e-9):
                    lowest_seen = self._float(getattr(position, "lowest_price", None))
                    prior_trough = self._float(underwater_state.get("trough_price"))
                    trough_price = current_price
                    if lowest_seen is not None and lowest_seen > 0.0:
                        trough_price = min(trough_price, lowest_seen)
                    if prior_trough is not None and prior_trough > 0.0:
                        trough_price = min(trough_price, prior_trough)

                    underwater_since_minutes = self._float(underwater_state.get("underwater_since_age_minutes"))
                    if underwater_since_minutes is None and age_minutes is not None:
                        underwater_since_minutes = age_minutes
                    if (
                        age_minutes is not None
                        and underwater_since_minutes is not None
                        and underwater_since_minutes > age_minutes
                    ):
                        underwater_since_minutes = age_minutes

                    dwell_elapsed_minutes = None
                    if age_minutes is not None and underwater_since_minutes is not None:
                        dwell_elapsed_minutes = max(0.0, age_minutes - underwater_since_minutes)

                    rebound_distance = max(0.0, current_price - trough_price)
                    drawdown_distance = max(0.0, entry_price - trough_price)
                    recovery_ratio = 0.0
                    if drawdown_distance > 0.0:
                        recovery_ratio = clamp(rebound_distance / drawdown_distance, 0.0, 2.5)
                    rebound_pct = 0.0
                    if trough_price > 0.0:
                        rebound_pct = max(0.0, ((current_price - trough_price) / trough_price) * 100.0)

                    rebound_peak_price = current_price
                    previous_rebound_peak = self._float(underwater_state.get("rebound_peak_price"))
                    if previous_rebound_peak is not None and previous_rebound_peak > 0.0:
                        rebound_peak_price = max(rebound_peak_price, previous_rebound_peak)
                    fade_from_rebound_peak_pct = 0.0
                    if rebound_peak_price > 0.0:
                        fade_from_rebound_peak_pct = max(
                            0.0,
                            ((rebound_peak_price - current_price) / rebound_peak_price) * 100.0,
                        )

                    reversal_pressure = (
                        (momentum_short_pct is not None and momentum_short_pct <= -0.20)
                        or (flow_imbalance is not None and flow_imbalance <= -0.15)
                        or (drawdown_from_peak_pct >= max(0.30, dynamic_backside_pct * 0.45))
                    )
                    rebound_fading = fade_from_rebound_peak_pct >= underwater_exit_fade_pct
                    soft_fade = fade_from_rebound_peak_pct >= (underwater_exit_fade_pct * 0.60)
                    dwell_ready = (
                        dwell_elapsed_minutes is not None and dwell_elapsed_minutes >= underwater_dwell_minutes
                    )
                    recovery_ready = (
                        rebound_pct >= underwater_rebound_pct_min and recovery_ratio >= underwater_recovery_ratio_min
                    )

                    underwater_state["underwater_since_age_minutes"] = underwater_since_minutes
                    underwater_state["dwell_elapsed_minutes"] = dwell_elapsed_minutes
                    underwater_state["trough_price"] = trough_price
                    underwater_state["rebound_peak_price"] = rebound_peak_price
                    underwater_state["rebound_pct"] = rebound_pct
                    underwater_state["recovery_ratio"] = recovery_ratio
                    underwater_state["fade_from_rebound_peak_pct"] = fade_from_rebound_peak_pct
                    underwater_state["reversal_pressure"] = reversal_pressure
                    underwater_state["timeout_loss_pct"] = underwater_timeout_loss_pct

                    if (
                        exit_headroom_ratio is not None
                        and recovery_ready
                        and soft_fade
                        and exit_headroom_ratio <= warn_headroom_ratio * 1.1
                    ):
                        context_payload[underwater_state_key] = underwater_state
                        return {
                            "config": config,
                            "decision": {
                                "action": "close",
                                "reason": (
                                    "Underwater executable salvage exit "
                                    f"(headroom={exit_headroom_ratio:.2f}x, "
                                    f"recovery={recovery_ratio:.2f}, fade={fade_from_rebound_peak_pct:.2f}%)"
                                ),
                                "close_price": current_price,
                            },
                        }

                    if dwell_ready and recovery_ready and (rebound_fading or (soft_fade and reversal_pressure)):
                        context_payload[underwater_state_key] = underwater_state
                        return {
                            "config": config,
                            "decision": {
                                "action": "close",
                                "reason": (
                                    "Underwater rebound salvage exit "
                                    f"(dwell={dwell_elapsed_minutes if dwell_elapsed_minutes is not None else 0.0:.2f}m, "
                                    f"recovery={recovery_ratio:.2f}, rebound={rebound_pct:.2f}%, "
                                    f"fade={fade_from_rebound_peak_pct:.2f}%)"
                                ),
                                "close_price": current_price,
                            },
                        }

                    if (
                        underwater_timeout_minutes > 0.0
                        and underwater_timeout_loss_pct > 0.0
                        and dwell_elapsed_minutes is not None
                        and dwell_elapsed_minutes >= underwater_timeout_minutes
                        and pnl_pct <= -abs(underwater_timeout_loss_pct)
                    ):
                        context_payload[underwater_state_key] = underwater_state
                        return {
                            "config": config,
                            "decision": {
                                "action": "close",
                                "reason": (
                                    "Underwater timeout stop "
                                    f"(dwell={dwell_elapsed_minutes:.2f}m >= {underwater_timeout_minutes:.2f}m, "
                                    f"loss={pnl_pct:.2f}% <= -{underwater_timeout_loss_pct:.2f}%)"
                                ),
                                "close_price": current_price,
                            },
                        }
                    context_payload[underwater_state_key] = underwater_state
                else:
                    context_payload.pop(underwater_state_key, None)
            else:
                context_payload.pop(underwater_state_key, None)

        stop_loss_policy = str(config.get("stop_loss_policy") or "near_close_only").strip().lower()
        if stop_loss_policy == "volatility_adaptive":
            spread = self._float(context_payload.get("spread"))
            if spread is None:
                spread = self._float(state.get("spread"))
            liquidity = self._float(context_payload.get("liquidity"))
            if liquidity is None:
                liquidity = self._float(state.get("liquidity"))
            seconds_left = self._float(state.get("seconds_left"))
            timeframe_seconds = self._timeframe_seconds(timeframe)

            volatility_factor = 1.0
            if spread is not None:
                volatility_factor += min(0.8, max(0.0, (spread * 100.0) / 6.0))
            if liquidity is not None and liquidity > 0.0:
                volatility_factor += min(0.7, 6000.0 / max(liquidity, 1.0))
            if seconds_left is not None and seconds_left >= 0.0:
                close_ratio = 1.0 - clamp(seconds_left / float(max(1, timeframe_seconds)), 0.0, 1.0)
                volatility_factor += close_ratio * 0.4

            base_stop_loss_pct = max(0.5, to_float(config.get("stop_loss_pct"), 5.0))
            adaptive_stop_loss_pct = clamp(
                base_stop_loss_pct * volatility_factor,
                max(0.5, base_stop_loss_pct * 0.8),
                max(1.0, base_stop_loss_pct * 1.9),
            )
            config["stop_loss_pct"] = adaptive_stop_loss_pct

            trailing_activation = max(0.1, to_float(config.get("trailing_stop_activation_profit_pct"), 4.0))
            config["trailing_stop_activation_profit_pct"] = clamp(
                trailing_activation / max(1.0, volatility_factor * 0.85),
                0.1,
                100.0,
            )

        return {"config": config, "decision": None}

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        if market_state.get("is_resolved"):
            return self.default_exit_check(position, market_state)
        evaluation = self._evaluate_local_exit(position, market_state)
        config = evaluation.get("config")
        if isinstance(config, dict):
            position.config = config
        decision = evaluation.get("decision")
        if isinstance(decision, dict) and str(decision.get("action") or "").strip().lower() == "close":
            close_price = self._float(decision.get("close_price"))
            return ExitDecision(
                "close",
                str(decision.get("reason") or "High-frequency exit"),
                close_price=close_price,
            )
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

        raw_token_ids = d.get("clob_token_ids") or []
        clob_token_ids = [str(t).strip() for t in raw_token_ids if str(t).strip() and len(str(t).strip()) > 20]

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
            clob_token_ids=clob_token_ids,
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
            is_live = bool(market.get("is_live")) if isinstance(market.get("is_live"), bool) else (seconds_left > 0.0)
            is_current = bool(market.get("is_current")) if isinstance(market.get("is_current"), bool) else is_live
            start_time = str(market.get("start_time") or "").strip() or None
            end_time = str(market.get("end_time") or "").strip() or None
            live_market_fetched_at = (
                str(market.get("fetched_at") or market.get("snapshot_fetched_at") or "").strip() or None
            )
            market_data_age_ms = self._float(market.get("market_data_age_ms"))
            if market_data_age_ms is None and live_market_fetched_at:
                parsed_fetched_at = _parse_datetime_utc(live_market_fetched_at)
                if parsed_fetched_at is not None:
                    market_data_age_ms = max(
                        0.0,
                        (datetime.now(timezone.utc) - parsed_fetched_at.astimezone(timezone.utc)).total_seconds()
                        * 1000.0,
                    )

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

            token_idx = 0 if direction == "buy_yes" else 1
            token_ids = typed_market.clob_token_ids or []
            position_token_id = token_ids[token_idx] if len(token_ids) > token_idx else None

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
                        "token_id": position_token_id,
                        # Carry crypto-specific context through positions payload
                        "_crypto_context": {
                            "signal_version": "crypto_worker_v2",
                            "signal_family": "crypto_multistrategy",
                            "strategy_origin": "crypto_worker",
                            "selected_direction": direction,
                            "asset": asset,
                            "timeframe": timeframe,
                            "regime": regime,
                            "start_time": start_time,
                            "end_time": end_time,
                            "seconds_left": float(seconds_left),
                            "is_live": is_live,
                            "is_current": is_current,
                            "oracle_available": has_oracle,
                            "oracle_age_seconds": self._float(market.get("oracle_age_seconds")),
                            "oracle_updated_at_ms": self._float(market.get("oracle_updated_at_ms")),
                            "dominant_strategy": dominant_strategy,
                            "spread": spread,
                            "spread_widening_bps": self._float(market.get("spread_widening_bps")),
                            "orderbook_imbalance": self._float(
                                _first_present(
                                    market.get("orderbook_imbalance"),
                                    market.get("book_imbalance"),
                                    market.get("imbalance"),
                                )
                            ),
                            "liquidity": liquidity,
                            "volume": self._float(market.get("volume")) or 0.0,
                            "price_to_beat": price_to_beat,
                            "oracle_price": oracle_price,
                            "execution_penalty_percent": round(execution_penalty, 6),
                            "live_market_fetched_at": live_market_fetched_at,
                            "market_data_age_ms": market_data_age_ms,
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
                    "start_time": start_time,
                    "end_time": end_time,
                    "seconds_left": float(seconds_left),
                    "is_live": is_live,
                    "is_current": is_current,
                    "selected_direction": direction,
                    "oracle_available": has_oracle,
                    "oracle_age_seconds": self._float(market.get("oracle_age_seconds")),
                    "oracle_updated_at_ms": self._float(market.get("oracle_updated_at_ms")),
                    "dominant_strategy": dominant_strategy,
                    "spread": spread,
                    "spread_widening_bps": self._float(market.get("spread_widening_bps")),
                    "orderbook_imbalance": self._float(
                        _first_present(
                            market.get("orderbook_imbalance"),
                            market.get("book_imbalance"),
                            market.get("imbalance"),
                        )
                    ),
                    "liquidity": liquidity,
                    "volume": self._float(market.get("volume")) or 0.0,
                    "price_to_beat": price_to_beat,
                    "oracle_price": oracle_price,
                    "execution_penalty_percent": round(execution_penalty, 6),
                    "live_market_fetched_at": live_market_fetched_at,
                    "market_data_age_ms": market_data_age_ms,
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
