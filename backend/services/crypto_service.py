"""Standalone crypto market service.

Completely independent from the scanner/strategy pipeline.  Fetches live
Polymarket 15-minute crypto markets directly from the Gamma series API
and returns structured data for the frontend Crypto tab.

This service always returns the 4 live markets (BTC, ETH, SOL, XRP) with
real-time pricing, regardless of whether any arbitrage opportunity exists.

Also runs a dedicated fast-scan loop (every 3-5s) for crypto strategy
evaluation, completely independent of the main scanner.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Optional

import httpx

from config import settings as _cfg
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class CryptoMarket:
    """A live crypto 15-minute market with real-time data."""

    __slots__ = (
        "id",
        "condition_id",
        "slug",
        "question",
        "asset",
        "timeframe",
        "start_time",
        "end_time",
        "up_price",
        "down_price",
        "best_bid",
        "best_ask",
        "spread",
        "liquidity",
        "volume",
        "volume_24h",
        "series_volume_24h",
        "series_liquidity",
        "last_trade_price",
        "clob_token_ids",
        "fees_enabled",
        "event_slug",
        "event_title",
        "is_current",        # True = currently live, False = upcoming
        "upcoming_markets",  # list of upcoming market dicts for this asset
    )

    def __init__(self, **kwargs):
        for k in self.__slots__:
            setattr(self, k, kwargs.get(k))

    def to_dict(self) -> dict:
        end = None
        start = None
        seconds_left = None

        if self.end_time:
            try:
                end = datetime.fromisoformat(str(self.end_time).replace("Z", "+00:00"))
                seconds_left = max(0, (end.timestamp() - time.time()))
            except (ValueError, AttributeError):
                pass

        if self.start_time:
            try:
                start = datetime.fromisoformat(str(self.start_time).replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        is_live = (
            start is not None
            and end is not None
            and start.timestamp() <= time.time() < end.timestamp()
        )

        combined = None
        if self.up_price is not None and self.down_price is not None:
            combined = self.up_price + self.down_price

        return {
            "id": self.id,
            "condition_id": self.condition_id,
            "slug": self.slug,
            "question": self.question,
            "asset": self.asset,
            "timeframe": self.timeframe,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "seconds_left": round(seconds_left) if seconds_left is not None else None,
            "is_live": is_live,
            "is_current": bool(self.is_current),
            "up_price": self.up_price,
            "down_price": self.down_price,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "spread": self.spread,
            "combined": combined,
            "liquidity": self.liquidity,
            "volume": self.volume,
            "volume_24h": self.volume_24h,
            "series_volume_24h": self.series_volume_24h or 0,
            "series_liquidity": self.series_liquidity or 0,
            "last_trade_price": self.last_trade_price,
            "clob_token_ids": self.clob_token_ids or [],
            "fees_enabled": self.fees_enabled,
            "event_slug": self.event_slug,
            "event_title": self.event_title,
            "upcoming_markets": self.upcoming_markets or [],
        }


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


def _get_series_configs() -> list[tuple[str, str, str]]:
    """Read series configs from settings (DB-persisted, editable in Settings UI)."""
    return [
        (_cfg.BTC_ETH_HF_SERIES_BTC_15M, "BTC", "15min"),
        (_cfg.BTC_ETH_HF_SERIES_ETH_15M, "ETH", "15min"),
        (_cfg.BTC_ETH_HF_SERIES_SOL_15M, "SOL", "15min"),
        (_cfg.BTC_ETH_HF_SERIES_XRP_15M, "XRP", "15min"),
    ]


def _parse_float(val, default=None) -> Optional[float]:
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _parse_json_list(val) -> list:
    import json

    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def _parse_outcome_prices(market_data: dict) -> tuple[Optional[float], Optional[float]]:
    """Extract Up/Down market prices from a Gamma market response.

    outcomePrices contains the canonical [up_price, down_price] as displayed
    on Polymarket.  bestBid/bestAsk are CLOB order-book levels for the Up
    token only -- used for spread info but NOT as display prices.
    """
    outcome_prices = _parse_json_list(market_data.get("outcomePrices"))
    outcomes = _parse_json_list(market_data.get("outcomes"))

    if outcome_prices and len(outcome_prices) >= 2:
        up_idx = None
        down_idx = None
        for i, label in enumerate(outcomes):
            lbl = str(label).lower()
            if lbl in ("up", "yes"):
                up_idx = i
            elif lbl in ("down", "no"):
                down_idx = i

        if up_idx is not None and down_idx is not None:
            return _parse_float(outcome_prices[up_idx]), _parse_float(
                outcome_prices[down_idx]
            )
        return _parse_float(outcome_prices[0]), _parse_float(outcome_prices[1])

    # Fallback: derive from bestBid/bestAsk midpoint
    best_bid = _parse_float(market_data.get("bestBid"))
    best_ask = _parse_float(market_data.get("bestAsk"))
    if best_bid is not None and best_ask is not None:
        up_mid = (best_bid + best_ask) / 2.0
        return up_mid, 1.0 - up_mid

    return None, None


class CryptoService:
    """Fetches and caches live crypto markets from the Polymarket Gamma API.

    Completely independent of the scanner and strategy pipeline.
    """

    def __init__(self, gamma_url: str = "", ttl_seconds: float = 5.0):
        self._gamma_url = gamma_url or _cfg.GAMMA_API_URL
        self._ttl = ttl_seconds
        self._cache: list[CryptoMarket] = []
        self._last_fetch: float = 0.0
        self._fast_scan_running = False
        # Price-to-beat tracking: {market_slug: oracle_price_at_start}
        self._price_to_beat: dict[str, float] = {}

    @property
    def is_stale(self) -> bool:
        return (time.monotonic() - self._last_fetch) > self._ttl

    def get_live_markets(self, force_refresh: bool = False) -> list[CryptoMarket]:
        """Return cached live crypto markets, refreshing from Gamma when needed."""
        if force_refresh or self.is_stale:
            try:
                fetched = self._fetch_all()
                if fetched is not None:
                    self._cache = fetched
                    self._last_fetch = time.monotonic()
            except Exception as e:
                logger.warning(f"CryptoService fetch failed: {e}")
        return self._cache

    def _fetch_all(self) -> list[CryptoMarket]:
        """Fetch live + upcoming markets for all configured series.

        Returns one CryptoMarket per asset for the currently-live market,
        with upcoming markets attached as nested data.  Series-level 24h
        volume and liquidity are included from the series metadata.
        """
        series = _get_series_configs()
        all_markets: list[CryptoMarket] = []

        with httpx.Client(timeout=10.0) as client:
            for series_id, asset, timeframe in series:
                try:
                    resp = client.get(
                        f"{self._gamma_url}/events",
                        params={
                            "series_id": series_id,
                            "active": "true",
                            "closed": "false",
                            "limit": 8,
                        },
                    )
                    if resp.status_code != 200:
                        logger.debug(
                            f"CryptoService: series_id={series_id} returned {resp.status_code}"
                        )
                        continue

                    events = resp.json()
                    if not isinstance(events, list):
                        continue

                    # Extract series-level stats (24h volume, liquidity)
                    series_vol_24h = 0.0
                    series_liq = 0.0
                    if events:
                        series_data = (events[0].get("series") or [{}])
                        if series_data and isinstance(series_data, list):
                            s = series_data[0] if series_data else {}
                            series_vol_24h = _parse_float(s.get("volume24hr")) or 0.0
                            series_liq = _parse_float(s.get("liquidity")) or 0.0

                    # Sort events by end time, pick live + upcoming
                    sorted_events = self._sort_events_by_time(events)
                    if not sorted_events:
                        continue

                    # First event = currently live (or soonest upcoming)
                    current_event = sorted_events[0]
                    upcoming_events = sorted_events[1:4]  # Next 3 upcoming

                    # Build the primary (current) market
                    mkt_list = current_event.get("markets", [])
                    if not mkt_list:
                        continue
                    mkt = mkt_list[0]
                    up_price, down_price = _parse_outcome_prices(mkt)
                    clob_ids = _parse_json_list(mkt.get("clobTokenIds"))

                    # Build upcoming market summaries
                    upcoming = []
                    for evt in upcoming_events:
                        emkts = evt.get("markets", [])
                        if not emkts:
                            continue
                        em = emkts[0]
                        e_up, e_down = _parse_outcome_prices(em)
                        upcoming.append({
                            "id": str(em.get("id", "")),
                            "slug": em.get("slug", ""),
                            "event_title": evt.get("title", ""),
                            "start_time": evt.get("startTime") or em.get("eventStartTime"),
                            "end_time": em.get("endDate"),
                            "up_price": e_up,
                            "down_price": e_down,
                            "best_bid": _parse_float(em.get("bestBid")),
                            "best_ask": _parse_float(em.get("bestAsk")),
                            "liquidity": _parse_float(em.get("liquidityNum"))
                            or _parse_float(em.get("liquidity"))
                            or 0.0,
                            "volume": _parse_float(em.get("volumeNum"))
                            or _parse_float(em.get("volume"))
                            or 0.0,
                        })

                    cm = CryptoMarket(
                        id=str(mkt.get("id", "")),
                        condition_id=mkt.get("conditionId", mkt.get("condition_id", "")),
                        slug=mkt.get("slug", ""),
                        question=mkt.get("question", ""),
                        asset=asset,
                        timeframe=timeframe,
                        start_time=current_event.get("startTime")
                        or mkt.get("eventStartTime"),
                        end_time=mkt.get("endDate"),
                        up_price=up_price,
                        down_price=down_price,
                        best_bid=_parse_float(mkt.get("bestBid")),
                        best_ask=_parse_float(mkt.get("bestAsk")),
                        spread=_parse_float(mkt.get("spread")),
                        liquidity=_parse_float(mkt.get("liquidityNum"))
                        or _parse_float(mkt.get("liquidity"))
                        or 0.0,
                        volume=_parse_float(mkt.get("volumeNum"))
                        or _parse_float(mkt.get("volume"))
                        or 0.0,
                        volume_24h=_parse_float(mkt.get("volume24hr")) or 0.0,
                        series_volume_24h=series_vol_24h,
                        series_liquidity=series_liq,
                        last_trade_price=_parse_float(mkt.get("lastTradePrice")),
                        clob_token_ids=clob_ids,
                        fees_enabled=mkt.get("feesEnabled", False),
                        event_slug=current_event.get("slug", ""),
                        event_title=current_event.get("title", ""),
                        is_current=True,
                        upcoming_markets=upcoming,
                    )
                    all_markets.append(cm)

                    time.sleep(0.03)  # Rate limit
                except Exception as e:
                    logger.debug(f"CryptoService: series_id={series_id} failed: {e}")

        if all_markets:
            assets = ", ".join(m.asset for m in all_markets)
            logger.debug(
                f"CryptoService: fetched {len(all_markets)} live markets ({assets})"
            )
        return all_markets

    @staticmethod
    def _sort_events_by_time(events: list[dict]) -> list[dict]:
        """Sort events by end time, filtering out closed/resolved ones."""
        now_ms = time.time() * 1000
        valid = []
        for evt in events:
            if evt.get("closed"):
                continue
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
                continue
            valid.append((end_ms, evt))
        valid.sort(key=lambda x: x[0])
        return [evt for _, evt in valid]

    @staticmethod
    def _pick_best_event(events: list[dict]) -> Optional[dict]:
        """Pick the currently-live event, or the soonest upcoming one."""
        now_ms = time.time() * 1000
        live = []
        upcoming = []

        for evt in events:
            if evt.get("closed"):
                continue
            end_str = evt.get("endDate")
            start_str = evt.get("startTime") or evt.get("startDate")
            if not end_str:
                continue
            try:
                end_ms = (
                    datetime.fromisoformat(end_str.replace("Z", "+00:00")).timestamp()
                    * 1000
                )
            except (ValueError, AttributeError):
                continue
            if end_ms <= now_ms:
                continue

            start_ms = None
            if start_str:
                try:
                    start_ms = (
                        datetime.fromisoformat(
                            start_str.replace("Z", "+00:00")
                        ).timestamp()
                        * 1000
                    )
                except (ValueError, AttributeError):
                    pass

            if start_ms is not None and start_ms <= now_ms:
                live.append((end_ms, evt))
            else:
                upcoming.append((end_ms, evt))

        live.sort(key=lambda x: x[0])
        upcoming.sort(key=lambda x: x[0])

        if live:
            return live[0][1]
        if upcoming:
            return upcoming[0][1]
        return None


    # ------------------------------------------------------------------
    # Fast scan loop (independent of main scanner)
    # ------------------------------------------------------------------

    async def start_fast_scan(self, interval_seconds: float = 2.0) -> None:
        """Start the independent fast-scan loop for crypto markets.

        Runs every 2 seconds:
        1. Refreshes market data if stale (Gamma API, 8s TTL)
        2. Broadcasts live market data to all connected frontends via WebSocket
        """
        self._fast_scan_running = True
        logger.info(
            f"CryptoService: starting fast scan loop (every {interval_seconds:.1f}s)"
        )

        while self._fast_scan_running:
            try:
                # Broadcast current market state to frontends via WebSocket
                await self._broadcast_markets()
            except Exception as e:
                logger.debug(f"CryptoService: fast scan error: {e}")
            await asyncio.sleep(interval_seconds)

    def stop_fast_scan(self) -> None:
        self._fast_scan_running = False

    def _update_price_to_beat(self, markets: list[CryptoMarket]) -> None:
        """Look up the price-to-beat for each market from the Chainlink history.

        The Chainlink feed maintains a rolling 25-minute price history.
        For each market, we look up the Chainlink price at the exact
        ``eventStartTime`` from this history.  This works even if the app
        started mid-window, because the history buffer has past prices.
        """
        try:
            from services.chainlink_feed import get_chainlink_feed
            feed = get_chainlink_feed()
        except Exception:
            return

        for m in markets:
            slug = m.slug
            if not slug:
                continue
            # Already found for this slug?
            if slug in self._price_to_beat and self._price_to_beat[slug] is not None:
                continue

            if not m.start_time:
                continue
            try:
                start_ts = datetime.fromisoformat(
                    str(m.start_time).replace("Z", "+00:00")
                ).timestamp()
            except (ValueError, AttributeError):
                continue

            now = time.time()
            # Only look up if the market has started
            if now < start_ts:
                continue

            # Look up the Chainlink price at eventStartTime from history
            ptb = feed.get_price_at_time(m.asset, start_ts)
            if ptb is not None:
                self._price_to_beat[slug] = ptb
                elapsed = now - start_ts
                logger.info(
                    f"CryptoService: price to beat for {m.asset} ({slug}): "
                    f"${ptb:,.2f} (from history, {elapsed:.0f}s after start)"
                )
            else:
                # No history available -- try current price if within 10s of start
                elapsed = now - start_ts
                if elapsed <= 10:
                    oracle = feed.get_price(m.asset)
                    if oracle and oracle.price:
                        self._price_to_beat[slug] = oracle.price
                        logger.info(
                            f"CryptoService: price to beat for {m.asset} ({slug}): "
                            f"${oracle.price:,.2f} (live, {elapsed:.1f}s after start)"
                        )

        # Clean up old entries
        active_slugs = {m.slug for m in markets}
        for slug in list(self._price_to_beat.keys()):
            if slug not in active_slugs:
                del self._price_to_beat[slug]

    async def _broadcast_markets(self) -> None:
        """Push live crypto market data to all connected frontends via WS.

        Overlays real-time CLOB WebSocket prices on top of cached Gamma data
        so the frontend always sees the freshest available prices.
        """
        try:
            from api.websocket import broadcast_crypto_markets
            from services.chainlink_feed import get_chainlink_feed

            markets = self.get_live_markets()
            if not markets:
                return

            # Update price-to-beat tracking
            self._update_price_to_beat(markets)

            # Get real-time prices: try CLOB WS cache first, then HTTP CLOB API
            ws_prices: dict[str, float] = {}

            # Layer 1: CLOB WebSocket price cache (sub-second)
            try:
                from services.ws_feeds import get_feed_manager
                feed_mgr = get_feed_manager()
                if feed_mgr._started:
                    for m in markets:
                        for i, token_id in enumerate(m.clob_token_ids or []):
                            if token_id and len(token_id) > 20:
                                if feed_mgr.price_cache.is_fresh(token_id):
                                    mid = feed_mgr.price_cache.get_mid(token_id)
                                    if mid is not None:
                                        ws_prices[token_id] = mid
            except Exception:
                pass

            # Layer 2: CLOB HTTP API for tokens not in WS cache
            missing_tokens = []
            for m in markets:
                for token_id in (m.clob_token_ids or []):
                    if token_id and len(token_id) > 20 and token_id not in ws_prices:
                        missing_tokens.append(token_id)

            if missing_tokens:
                try:
                    clob_url = _cfg.CLOB_API_URL
                    async with __import__("httpx").AsyncClient(timeout=3.0) as hc:
                        for token_id in missing_tokens[:8]:  # Cap to avoid slowdown
                            try:
                                resp = await hc.get(
                                    f"{clob_url}/price",
                                    params={"token_id": token_id, "side": "buy"},
                                )
                                if resp.status_code == 200:
                                    data = resp.json()
                                    p = _parse_float(data.get("price"))
                                    if p is not None:
                                        ws_prices[token_id] = p
                            except Exception:
                                pass
                except Exception:
                    pass

            feed = get_chainlink_feed()
            result = []
            for m in markets:
                d = m.to_dict()

                # Overlay CLOB WS prices (real-time) over Gamma prices (8s cache)
                # Token 0 = Up/Yes, Token 1 = Down/No
                tokens = m.clob_token_ids or []
                if len(tokens) >= 2:
                    ws_up = ws_prices.get(tokens[0])
                    ws_down = ws_prices.get(tokens[1])
                    if ws_up is not None and ws_down is not None:
                        d["up_price"] = ws_up
                        d["down_price"] = ws_down
                        d["combined"] = ws_up + ws_down
                    elif ws_up is not None:
                        d["up_price"] = ws_up
                        d["down_price"] = 1.0 - ws_up
                        d["combined"] = 1.0
                    elif ws_down is not None:
                        d["down_price"] = ws_down
                        d["up_price"] = 1.0 - ws_down
                        d["combined"] = 1.0

                # Attach oracle price
                oracle = feed.get_price(m.asset)
                if oracle:
                    d["oracle_price"] = oracle.price
                    d["oracle_updated_at_ms"] = oracle.updated_at_ms
                    d["oracle_age_seconds"] = round(
                        (time.time() * 1000 - oracle.updated_at_ms) / 1000, 1
                    ) if oracle.updated_at_ms else None
                else:
                    d["oracle_price"] = None
                    d["oracle_updated_at_ms"] = None
                    d["oracle_age_seconds"] = None

                # Attach price to beat for this market
                d["price_to_beat"] = self._price_to_beat.get(m.slug)

                # Attach recent oracle price history for sparkline chart
                history = feed._history.get(m.asset) if hasattr(feed, '_history') else None
                if history and len(history) > 0:
                    pts = list(history)
                    # Sample to ~80 points for smooth chart without flooding WS
                    if len(pts) > 80:
                        step = max(1, len(pts) // 80)
                        pts = pts[::step]
                    # Always include the very latest point
                    last = list(history)[-1]
                    if pts[-1] != last:
                        pts.append(last)
                    d["oracle_history"] = [
                        {"t": t, "p": round(p, 2)} for t, p in pts
                    ]
                else:
                    d["oracle_history"] = []

                result.append(d)

            await broadcast_crypto_markets(result)
        except Exception as e:
            logger.debug(f"CryptoService: broadcast failed: {e}")

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[CryptoService] = None


def get_crypto_service() -> CryptoService:
    global _instance
    if _instance is None:
        _instance = CryptoService()
    return _instance
