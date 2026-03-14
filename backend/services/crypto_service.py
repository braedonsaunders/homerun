"""Standalone crypto market service.

Completely independent from the scanner/strategy pipeline.  Fetches live
Polymarket 5-minute and 15-minute crypto markets directly from the Gamma series API
and returns structured data for the frontend Crypto tab.

This service always returns live markets (BTC, ETH, SOL, XRP) across configured
timeframes with
real-time pricing, regardless of whether any arbitrage opportunity exists.

Also runs a dedicated fast-scan loop (every 2-5s) for crypto strategy
evaluation, completely independent of the main scanner.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional

import httpx

from config import settings as _cfg
from utils.logger import get_logger

logger = get_logger(__name__)

_GAMMA_FETCH_TIMEOUT_SECONDS = 4.0
_CLOB_FETCH_TIMEOUT_SECONDS = 2.0
_CRYPTO_FETCH_MAX_WORKERS = 16
_CRYPTO_PRICE_TO_BEAT_API_URL = "https://polymarket.com/api/crypto/crypto-price"
_CRYPTO_PRICE_TO_BEAT_TIMEOUT_SECONDS = 2.0
_BINANCE_KLINES_API_URL = "https://api.binance.com/api/v3/klines"
# Price-to-beat is static for the market window; unresolved retries are coarse.
_PRICE_TO_BEAT_RETRY_SECONDS = 30.0
_PRICE_TO_BEAT_429_RETRY_SECONDS = 300.0
_PRICE_TO_BEAT_API_GLOBAL_COOLDOWN_SECONDS = 60.0
_PRICE_TO_BEAT_DELAYED_HISTORY_MAX_SECONDS = 300.0

_shared_sync_client: httpx.Client | None = None


def _get_shared_sync_client() -> httpx.Client:
    global _shared_sync_client
    if _shared_sync_client is None or _shared_sync_client.is_closed:
        _shared_sync_client = httpx.Client(timeout=_GAMMA_FETCH_TIMEOUT_SECONDS)
    return _shared_sync_client

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class CryptoMarket:
    """A live crypto market with real-time data."""

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
        "up_token_index",
        "down_token_index",
        "fees_enabled",
        "event_slug",
        "event_title",
        "price_to_beat",
        "is_current",  # True = currently live, False = upcoming
        "upcoming_markets",  # list of upcoming market dicts for this asset
    )

    def __init__(self, **kwargs):
        for k in self.__slots__:
            setattr(self, k, kwargs.get(k))

    def to_dict(self) -> dict:
        end = None
        start = None
        seconds_left = None
        now_ts = datetime.now(timezone.utc).timestamp()

        if self.end_time:
            try:
                end = datetime.fromisoformat(str(self.end_time).replace("Z", "+00:00"))
                seconds_left = max(0, (end.timestamp() - now_ts))
            except (ValueError, AttributeError):
                pass

        if self.start_time:
            try:
                start = datetime.fromisoformat(str(self.start_time).replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        is_live = start is not None and end is not None and start.timestamp() <= now_ts < end.timestamp()

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
            "up_token_index": self.up_token_index,
            "down_token_index": self.down_token_index,
            "fees_enabled": self.fees_enabled,
            "event_slug": self.event_slug,
            "event_title": self.event_title,
            "price_to_beat": self.price_to_beat,
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


def _coerce_probability(val) -> Optional[float]:
    parsed = _parse_float(val)
    if parsed is None:
        return None
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def _timeframe_to_crypto_price_variant(timeframe: object) -> Optional[str]:
    raw = str(timeframe or "").strip().lower()
    if not raw:
        return None
    if "4hr" in raw or "4h" in raw or "four" in raw:
        return "fourhour"
    if "1hr" in raw or "1h" in raw or "hour" in raw:
        return "hourly"
    if "15" in raw:
        return "fifteen"
    if "5" in raw:
        return "fiveminute"
    return None


def _resolve_binary_outcome_indexes(outcomes: list[object]) -> tuple[int, int]:
    """Return (up_idx, down_idx) for binary Up/Down or Yes/No markets."""
    up_idx = None
    down_idx = None
    for i, label in enumerate(outcomes):
        lbl = str(label or "").strip().lower()
        if lbl in ("up", "yes"):
            up_idx = i
        elif lbl in ("down", "no"):
            down_idx = i

    if up_idx is None:
        up_idx = 0
    if down_idx is None:
        down_idx = 1 if up_idx != 1 else 0
    return up_idx, down_idx


def _parse_outcome_prices(market_data: dict) -> tuple[Optional[float], Optional[float]]:
    """Extract Up/Down market prices from a Gamma market response.

    outcomePrices contains the canonical [up_price, down_price] as displayed
    on Polymarket.  bestBid/bestAsk are CLOB order-book levels for the Up
    token only -- used for spread info but NOT as display prices.
    """
    outcome_prices = _parse_json_list(market_data.get("outcomePrices"))
    outcomes = _parse_json_list(market_data.get("outcomes"))

    if outcome_prices and len(outcome_prices) >= 2:
        up_idx, down_idx = _resolve_binary_outcome_indexes(outcomes)
        if up_idx < len(outcome_prices) and down_idx < len(outcome_prices):
            return _coerce_probability(outcome_prices[up_idx]), _coerce_probability(outcome_prices[down_idx])
        return _coerce_probability(outcome_prices[0]), _coerce_probability(outcome_prices[1])

    # Fallback: derive from bestBid/bestAsk midpoint
    best_bid = _parse_float(market_data.get("bestBid"))
    best_ask = _parse_float(market_data.get("bestAsk"))
    if best_bid is not None and best_ask is not None:
        up_mid = (best_bid + best_ask) / 2.0
        return _coerce_probability(up_mid), _coerce_probability(1.0 - up_mid)

    return None, None


def _oracle_snapshot_payload(oracle: object | None) -> dict[str, float | str | None]:
    """Normalize oracle snapshot payload for API responses."""
    try:
        from services.chainlink_feed import OraclePrice
    except Exception:

        class OraclePrice:
            pass

    if not isinstance(oracle, OraclePrice):
        return {
            "price": None,
            "source": None,
            "updated_at_ms": None,
            "age_seconds": None,
        }

    updated_ms = getattr(oracle, "updated_at_ms", None)
    age_seconds = round((time.time() * 1000 - int(updated_ms)) / 1000, 1) if updated_ms else None

    return {
        "price": float(getattr(oracle, "price")),
        "source": str(getattr(oracle, "source", "")) or None,
        "updated_at_ms": int(updated_ms),
        "age_seconds": age_seconds,
    }


class CryptoService:
    """Fetches and caches live crypto markets from the Polymarket Gamma API.

    Completely independent of the scanner and strategy pipeline.
    """

    def __init__(self, gamma_url: str = "", ttl_seconds: float = 2.0):
        self._gamma_url = gamma_url or _cfg.GAMMA_API_URL
        self._ttl = ttl_seconds
        self._cache: list[CryptoMarket] = []
        self._last_fetch: float = 0.0
        self._fast_scan_running = False
        # Price-to-beat tracking: {market_slug: oracle_price_at_start}
        self._price_to_beat: dict[str, float] = {}
        self._price_to_beat_retry_after: dict[str, float] = {}
        self._price_to_beat_api_cooldown_until: float = 0.0
        # Track seen market IDs to detect rotations.
        self._prev_market_ids: set[str] = set()

    @property
    def is_stale(self) -> bool:
        return (time.monotonic() - self._last_fetch) > self._ttl

    def get_live_markets(self, force_refresh: bool = False) -> list[CryptoMarket]:
        """Return cached live crypto markets, refreshing from Gamma when needed."""
        if force_refresh or self.is_stale:
            try:
                fetched = self._fetch_all()
                if fetched is not None:
                    new_ids = {str(m.id) for m in fetched if m.id}
                    # Log market rotations — new IDs that weren't in previous set.
                    if self._prev_market_ids:
                        appeared = new_ids - self._prev_market_ids
                        disappeared = self._prev_market_ids - new_ids
                        if appeared:
                            new_descs = [f"{m.asset} {m.timeframe}" for m in fetched if str(m.id) in appeared]
                            logger.info(
                                "Market rotation: %d new market(s) appeared: %s",
                                len(appeared),
                                ", ".join(new_descs[:6]),
                            )
                        if disappeared:
                            logger.debug("Market rotation: %d market(s) expired", len(disappeared))
                    self._prev_market_ids = new_ids
                    self._cache = fetched
                    self._last_fetch = time.monotonic()
            except Exception as e:
                logger.warning(f"CryptoService fetch failed: {e}")
        if self._cache:
            try:
                self._update_price_to_beat(self._cache)
            except Exception as exc:
                logger.warning("CryptoService price-to-beat refresh failed", exc_info=exc)
        return self._cache

    def _fetch_clob_midpoint(self, client: httpx.Client, token_id: str) -> Optional[float]:
        """Fetch token midpoint from CLOB and normalize to [0, 1]."""
        token = str(token_id or "").strip()
        if not token:
            return None
        try:
            resp = client.get(
                f"{_cfg.CLOB_API_URL}/midpoint",
                params={"token_id": token},
            )
            if resp.status_code != 200:
                return None
            payload = resp.json()
            if not isinstance(payload, dict):
                return None
            return _coerce_probability(payload.get("mid"))
        except Exception:
            return None

    def _fetch_clob_price(self, client: httpx.Client, token_id: str, side: str) -> Optional[float]:
        """Fetch token best price from CLOB and normalize to [0, 1]."""
        token = str(token_id or "").strip()
        side_norm = str(side or "").strip().lower()
        if not token or side_norm not in ("buy", "sell"):
            return None
        try:
            resp = client.get(
                f"{_cfg.CLOB_API_URL}/price",
                params={"token_id": token, "side": side_norm},
            )
            if resp.status_code != 200:
                return None
            payload = resp.json()
            if not isinstance(payload, dict):
                return None
            return _coerce_probability(payload.get("price"))
        except Exception:
            return None

    def _fetch_series_market(
        self,
        series_id: str,
        asset: str,
        timeframe: str,
        now_iso: str,
    ) -> Optional[CryptoMarket]:
        """Fetch one series row (market discovery only — single Gamma HTTP call).

        Real-time prices come from the WS feed overlay in crypto_worker, NOT
        from per-token CLOB HTTP calls here.  This keeps each series fetch to
        a single ~200ms Gamma API call instead of 5+ sequential HTTP calls.
        """
        if not str(series_id or "").strip():
            return None

        try:
            client = _get_shared_sync_client()
            resp = client.get(
                f"{self._gamma_url}/events",
                params={
                    "series_id": series_id,
                    "active": "true",
                    "closed": "false",
                    "end_date_min": now_iso,
                    "order": "endDate",
                    "ascending": "true",
                    "limit": 8,
                },
            )
            if resp.status_code != 200:
                logger.debug(
                    "CryptoService: series_id=%s returned %s",
                    series_id,
                    resp.status_code,
                )
                return None

            events = resp.json()
            if not isinstance(events, list):
                return None

            series_vol_24h = 0.0
            series_liq = 0.0
            if events:
                series_data = events[0].get("series") or [{}]
                if series_data and isinstance(series_data, list):
                    s = series_data[0] if series_data else {}
                    series_vol_24h = _parse_float(s.get("volume24hr")) or 0.0
                    series_liq = _parse_float(s.get("liquidity")) or 0.0

            sorted_events = self._sort_events_by_time(events)
            if not sorted_events:
                return None

            current_event = sorted_events[0]
            upcoming_events = sorted_events[1:4]

            mkt_list = current_event.get("markets", [])
            if not mkt_list:
                return None
            mkt = mkt_list[0]
            up_price, down_price = _parse_outcome_prices(mkt)
            outcomes = _parse_json_list(mkt.get("outcomes"))
            up_idx, down_idx = _resolve_binary_outcome_indexes(outcomes)
            clob_ids = [
                str(token).strip() for token in _parse_json_list(mkt.get("clobTokenIds")) if str(token).strip()
            ]

            best_bid = _coerce_probability(mkt.get("bestBid"))
            best_ask = _coerce_probability(mkt.get("bestAsk"))

            upcoming = []
            for evt in upcoming_events:
                emkts = evt.get("markets", [])
                if not emkts:
                    continue
                em = emkts[0]
                e_up, e_down = _parse_outcome_prices(em)
                upcoming.append(
                    {
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
                        "volume": _parse_float(em.get("volumeNum")) or _parse_float(em.get("volume")) or 0.0,
                    }
                )

            return CryptoMarket(
                id=str(mkt.get("id", "")),
                condition_id=mkt.get("conditionId", mkt.get("condition_id", "")),
                slug=mkt.get("slug", ""),
                question=mkt.get("question", ""),
                asset=asset,
                timeframe=timeframe,
                start_time=current_event.get("startTime") or mkt.get("eventStartTime"),
                end_time=mkt.get("endDate"),
                up_price=up_price,
                down_price=down_price,
                best_bid=best_bid,
                best_ask=best_ask,
                spread=_parse_float(mkt.get("spread")),
                liquidity=_parse_float(mkt.get("liquidityNum")) or _parse_float(mkt.get("liquidity")) or 0.0,
                volume=_parse_float(mkt.get("volumeNum")) or _parse_float(mkt.get("volume")) or 0.0,
                volume_24h=_parse_float(mkt.get("volume24hr")) or 0.0,
                series_volume_24h=series_vol_24h,
                series_liquidity=series_liq,
                last_trade_price=_parse_float(mkt.get("lastTradePrice")),
                clob_token_ids=clob_ids,
                up_token_index=up_idx,
                down_token_index=down_idx,
                fees_enabled=mkt.get("feesEnabled", False),
                event_slug=current_event.get("slug", ""),
                event_title=current_event.get("title", ""),
                is_current=True,
                upcoming_markets=upcoming,
            )
        except Exception as e:
            logger.debug("CryptoService: series_id=%s failed: %s", series_id, e)
            return None

    def _fetch_all(self) -> list[CryptoMarket]:
        """Fetch live + upcoming markets for all configured series.

        Fetches each configured series in parallel so worker cycles stay close
        to the configured interval even when individual upstream calls are slow.
        """
        series = _get_series_configs()
        configured_series = [
            (idx, str(series_id), asset, timeframe)
            for idx, (series_id, asset, timeframe) in enumerate(series)
            if str(series_id or "").strip()
        ]
        if not configured_series:
            return []

        now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

        max_workers = max(
            1,
            min(_CRYPTO_FETCH_MAX_WORKERS, len(configured_series)),
        )

        fetched_indexed: list[tuple[int, CryptoMarket]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {
                pool.submit(
                    self._fetch_series_market,
                    series_id,
                    asset,
                    timeframe,
                    now_iso,
                ): idx
                for idx, series_id, asset, timeframe in configured_series
            }

            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    market = future.result()
                except Exception as exc:
                    logger.debug(
                        "CryptoService: parallel fetch failed for index=%s: %s",
                        idx,
                        exc,
                    )
                    continue
                if market is not None:
                    fetched_indexed.append((idx, market))

        fetched_indexed.sort(key=lambda item: item[0])
        all_markets = [market for _, market in fetched_indexed]

        if all_markets:
            assets = ", ".join(f"{m.asset} {m.timeframe}" for m in all_markets)
            logger.debug(
                "CryptoService: fetched %s live markets (%s)",
                len(all_markets),
                assets,
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
                end_ms = datetime.fromisoformat(end_str.replace("Z", "+00:00")).timestamp() * 1000
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
                end_ms = datetime.fromisoformat(end_str.replace("Z", "+00:00")).timestamp() * 1000
            except (ValueError, AttributeError):
                continue
            if end_ms <= now_ms:
                continue

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
        logger.info(f"CryptoService: starting fast scan loop (every {interval_seconds:.1f}s)")

        while self._fast_scan_running:
            try:
                # Broadcast current market state to frontends via WebSocket
                await self._broadcast_markets()
            except Exception as e:
                logger.debug(f"CryptoService: fast scan error: {e}")
            await asyncio.sleep(interval_seconds)

    def stop_fast_scan(self) -> None:
        self._fast_scan_running = False

    def _fetch_price_to_beat_from_crypto_api(
        self,
        client: httpx.Client,
        market: CryptoMarket,
        *,
        event_start_time: str,
    ) -> tuple[Optional[float], Optional[int]]:
        symbol = str(market.asset or "").strip().upper()
        variant = _timeframe_to_crypto_price_variant(market.timeframe)
        if not symbol or not variant or not event_start_time:
            return None, None

        try:
            resp = client.get(
                _CRYPTO_PRICE_TO_BEAT_API_URL,
                params={
                    "symbol": symbol,
                    "variant": variant,
                    "eventStartTime": event_start_time,
                },
            )
        except Exception:
            return None, None

        if resp.status_code != 200:
            return None, int(resp.status_code)

        try:
            payload = resp.json()
        except Exception:
            return None, int(resp.status_code)
        if not isinstance(payload, dict):
            return None, int(resp.status_code)

        open_price = _parse_float(payload.get("openPrice"))
        if open_price is None or open_price <= 0:
            return None, int(resp.status_code)
        return open_price, int(resp.status_code)

    def _fetch_price_to_beat_from_binance_kline(
        self,
        client: httpx.Client,
        market: CryptoMarket,
        *,
        start_timestamp_ms: int,
    ) -> Optional[float]:
        asset = str(market.asset or "").strip().upper()
        symbol_by_asset = {
            "BTC": "BTCUSDT",
            "ETH": "ETHUSDT",
            "SOL": "SOLUSDT",
            "XRP": "XRPUSDT",
        }
        symbol = symbol_by_asset.get(asset)
        if symbol is None:
            return None

        start_ms = int(max(0, start_timestamp_ms))
        end_ms = start_ms + 60_000

        try:
            resp = client.get(
                _BINANCE_KLINES_API_URL,
                params={
                    "symbol": symbol,
                    "interval": "1m",
                    "startTime": start_ms,
                    "endTime": end_ms,
                    "limit": 1,
                },
            )
        except Exception:
            return None

        if resp.status_code != 200:
            return None

        try:
            payload = resp.json()
        except Exception:
            return None
        if not isinstance(payload, list) or not payload:
            return None

        first = payload[0]
        if not isinstance(first, list) or len(first) < 2:
            return None

        open_price = _parse_float(first[1])
        if open_price is None or open_price <= 0:
            return None
        return open_price

    def _update_price_to_beat(self, markets: list[CryptoMarket]) -> None:
        """Look up the price-to-beat for each market.

        Priority:
        1) Polymarket crypto-price API (openPrice at eventStartTime) for
           cold-start accuracy even after extended downtime.
        2) Chainlink rolling history lookup.
        3) Live oracle fallback within 10s of event start.
        """
        feed = None
        try:
            from services.chainlink_feed import get_chainlink_feed

            feed = get_chainlink_feed()
        except Exception:
            pass

        api_client = _get_shared_sync_client()

        for m in markets:
            slug = m.slug
            if not slug:
                continue
            if slug in self._price_to_beat and self._price_to_beat[slug] is not None:
                continue

            if not m.start_time:
                continue

            try:
                start_dt = datetime.fromisoformat(str(m.start_time).replace("Z", "+00:00"))
                if start_dt.tzinfo is None:
                    start_dt = start_dt.replace(tzinfo=timezone.utc)
            except (ValueError, AttributeError):
                continue

            start_ts = start_dt.timestamp()
            start_iso = start_dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

            now = time.time()
            if now < start_ts:
                continue

            elapsed = now - start_ts
            retry_after = float(self._price_to_beat_retry_after.get(slug) or 0.0)
            if now < retry_after:
                continue

            if now >= self._price_to_beat_api_cooldown_until:
                ptb_api, api_status = self._fetch_price_to_beat_from_crypto_api(
                    api_client,
                    m,
                    event_start_time=start_iso,
                )
                if ptb_api is not None:
                    self._price_to_beat[slug] = ptb_api
                    self._price_to_beat_retry_after.pop(slug, None)
                    logger.info(
                        f"CryptoService: price to beat for {m.asset} ({slug}): "
                        f"${ptb_api:,.2f} (api, {elapsed:.0f}s after start)"
                    )
                    continue
                if api_status == 429:
                    previous_cooldown = self._price_to_beat_api_cooldown_until
                    self._price_to_beat_api_cooldown_until = max(
                        self._price_to_beat_api_cooldown_until,
                        now + _PRICE_TO_BEAT_API_GLOBAL_COOLDOWN_SECONDS,
                    )
                    self._price_to_beat_retry_after[slug] = max(
                        float(self._price_to_beat_retry_after.get(slug) or 0.0),
                        now + _PRICE_TO_BEAT_429_RETRY_SECONDS,
                    )
                    if self._price_to_beat_api_cooldown_until > previous_cooldown:
                        logger.warning(
                            "CryptoService: crypto-price API rate limited; backing off",
                            slug=slug,
                            asset=m.asset,
                            retry_after_seconds=round(
                                self._price_to_beat_api_cooldown_until - now,
                                1,
                            ),
                        )
                elif api_status is not None and api_status >= 500:
                    self._price_to_beat_retry_after[slug] = max(
                        float(self._price_to_beat_retry_after.get(slug) or 0.0),
                        now + _PRICE_TO_BEAT_RETRY_SECONDS,
                    )
                elif api_status is not None:
                    self._price_to_beat_retry_after[slug] = max(
                        float(self._price_to_beat_retry_after.get(slug) or 0.0),
                        now + (_PRICE_TO_BEAT_RETRY_SECONDS / 2.0),
                    )

            if feed is None:
                self._price_to_beat_retry_after[slug] = max(
                    float(self._price_to_beat_retry_after.get(slug) or 0.0),
                    now + _PRICE_TO_BEAT_RETRY_SECONDS,
                )
                continue

            ptb = feed.get_price_at_time(m.asset, start_ts)
            if ptb is None and hasattr(feed, "get_price_at_or_after_time"):
                ptb = feed.get_price_at_or_after_time(
                    m.asset,
                    start_ts,
                    max_delay_seconds=_PRICE_TO_BEAT_DELAYED_HISTORY_MAX_SECONDS,
                )
            if ptb is not None:
                self._price_to_beat[slug] = ptb
                self._price_to_beat_retry_after.pop(slug, None)
                logger.info(
                    f"CryptoService: price to beat for {m.asset} ({slug}): "
                    f"${ptb:,.2f} (from history, {elapsed:.0f}s after start)"
                )
                continue

            if elapsed <= 10:
                oracle = feed.get_price(m.asset)
                if oracle and oracle.price:
                    self._price_to_beat[slug] = oracle.price
                    self._price_to_beat_retry_after.pop(slug, None)
                    logger.info(
                        f"CryptoService: price to beat for {m.asset} ({slug}): "
                        f"${oracle.price:,.2f} (live, {elapsed:.1f}s after start)"
                    )
                    continue

            binance_open = self._fetch_price_to_beat_from_binance_kline(
                api_client,
                m,
                start_timestamp_ms=int(start_ts * 1000.0),
            )
            if binance_open is not None:
                self._price_to_beat[slug] = binance_open
                self._price_to_beat_retry_after.pop(slug, None)
                logger.info(
                    f"CryptoService: price to beat for {m.asset} ({slug}): "
                    f"${binance_open:,.2f} (binance-1m-open, {elapsed:.0f}s after start)"
                )
                continue

            self._price_to_beat_retry_after[slug] = max(
                float(self._price_to_beat_retry_after.get(slug) or 0.0),
                now + _PRICE_TO_BEAT_RETRY_SECONDS,
            )

        # Clean up old entries
        active_slugs = {m.slug for m in markets}
        for slug in list(self._price_to_beat.keys()):
            if slug not in active_slugs:
                del self._price_to_beat[slug]
        for slug in list(self._price_to_beat_retry_after.keys()):
            if slug not in active_slugs or slug in self._price_to_beat:
                del self._price_to_beat_retry_after[slug]
        for market in markets:
            market.price_to_beat = self._price_to_beat.get(market.slug)

    async def _broadcast_markets(self) -> None:
        """Push live crypto market data to all connected frontends via WS.

        Overlays real-time CLOB WebSocket prices on top of cached Gamma data
        so the frontend always sees the freshest available prices.
        """
        try:
            from api.websocket import broadcast_crypto_markets
            from services.chainlink_feed import get_chainlink_feed

            markets = await asyncio.to_thread(self.get_live_markets)
            if not markets:
                return

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
                                if feed_mgr.cache.is_fresh(token_id):
                                    mid = feed_mgr.cache.get_mid_price(token_id)
                                    if mid is not None:
                                        ws_prices[token_id] = mid
            except Exception as exc:
                logger.debug("CryptoService: WS price cache read failed: %s", exc)

            # Layer 2: CLOB HTTP API for tokens not in WS cache
            missing_tokens = []
            for m in markets:
                for token_id in m.clob_token_ids or []:
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
                tokens = m.clob_token_ids or []
                up_idx = int(m.up_token_index) if m.up_token_index is not None else 0
                down_idx = int(m.down_token_index) if m.down_token_index is not None else 1
                up_token = tokens[up_idx] if up_idx < len(tokens) else None
                down_token = tokens[down_idx] if down_idx < len(tokens) else None
                if up_token and down_token:
                    ws_up = ws_prices.get(up_token)
                    ws_down = ws_prices.get(down_token)
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
                    d["oracle_source"] = getattr(oracle, "source", None)
                    d["oracle_updated_at_ms"] = oracle.updated_at_ms
                    d["oracle_age_seconds"] = (
                        round((time.time() * 1000 - oracle.updated_at_ms) / 1000, 1) if oracle.updated_at_ms else None
                    )
                else:
                    d["oracle_price"] = None
                    d["oracle_updated_at_ms"] = None
                    d["oracle_age_seconds"] = None

                # Attach price to beat for this market
                d["price_to_beat"] = self._price_to_beat.get(m.slug)

                # Attach source-specific oracle snapshots for debugging and comparison.
                source_snapshots: dict[str, dict[str, float | str | None]] = {}
                for src, snap in feed.get_prices_by_source(m.asset).items():
                    source_snapshots[str(src)] = _oracle_snapshot_payload(snap)
                d["oracle_prices_by_source"] = source_snapshots

                # Attach recent oracle price history for sparkline chart
                history = feed._history.get(m.asset) if hasattr(feed, "_history") else None
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
                    d["oracle_history"] = [{"t": t, "p": round(p, 2)} for t, p in pts]
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
