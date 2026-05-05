"""Async client for the polybacktest.com REST API (v2).

Polybacktest is a paid SaaS that captures sub-second Polymarket Up/Down
prediction-market book snapshots plus Binance spot/futures reference
prices.  It is **not** a Polymarket wrapper — it stores its own
~8 Hz captures.  We import their data on demand into our local
``market_microstructure_snapshots`` table where the existing backtest
engine, fill model, and Data Lab UI can consume it without any
provider-specific code paths downstream.

API shape (verified live against api.polybacktest.com on 2026-05-05):
  * Base URL:           https://api.polybacktest.com
  * Auth:               ``X-API-Key: <api_key>`` request header
  * Rate limit:         per-tier sliding-window + token-bucket burst
                        (429 + ``Retry-After`` on overage)
  * Coins supported:    btc / eth / sol  (one per request)
  * List markets:       ``GET /v2/markets?coin={coin}&limit=&offset=``
  * Get market:         ``GET /v2/markets/{market_id}?coin={coin}``
  * Get snapshots:      ``GET /v2/markets/{market_id}/snapshots`` with
                        params: ``coin``, ``start_time``, ``end_time``,
                        ``limit`` (max 1000), ``offset``, and
                        ``include_orderbook=true`` (REQUIRED to get
                        full book depth — defaults to false → BBO only).
  * Pagination:         offset-based, total returned in response.

Response shapes:
  * Market: ``market_id`` (str), ``slug``, ``market_type`` (5m/15m/1h/4h/24h),
    ``start_time`` / ``end_time`` (ISO 8601), ``btc_price_start``,
    ``btc_price_end``, ``winner`` (Up/Down/null), ``final_volume``,
    ``final_liquidity``, ``condition_id``, ``clob_token_up``,
    ``clob_token_down``.  We synthesize a human ``title`` on top.
  * Snapshot:   ``id``, ``time`` (ISO 8601), ``market_id``,
    ``{coin}_price`` (e.g. ``btc_price``),
    ``price_up`` / ``price_down`` (0-1 range, best mid),
    ``orderbook_up`` / ``orderbook_down``, each with
    ``bids: [{price, size}]`` and ``asks: [{price, size}]``
    (sorted best-first, ~15 levels per side).

Defensive concerns:
  * **No API key** is treated as a normal "provider not configured"
    state.  Callers receive a typed ``PolybacktestNotConfiguredError``
    so the route layer can render a UI hint instead of a 500.
  * **429 / 5xx** retried with exponential backoff (max 5 attempts)
    plus the ``Retry-After`` hint when the server provides one.
  * **Hard concurrency cap** via an internal asyncio.Semaphore (8) — well
    below polybacktest's 10-parallel batch ceiling.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from utils.rate_limiter import TokenBucket

logger = logging.getLogger(__name__)


_DEFAULT_BASE_URL = "https://api.polybacktest.com"

# Concurrency cap (in-flight requests).  Below polybacktest's documented
# 10-parallel batch ceiling so a single import job never starves other
# callers.
_MAX_CONCURRENT_REQUESTS = 8

# Token-bucket request rate (requests / second).  Generous default —
# the server tells us the real limit via ``X-RateLimit-*`` headers and
# 429s; this is just so we don't fire 200 requests in <1s during a big
# import.  Pro tier docs imply ~10 RPS sustained is safe.
_DEFAULT_RATE_PER_SECOND = 8.0
_DEFAULT_BURST = 16

# Retry configuration.  5 attempts with exponential backoff covers
# transient hiccups without the worker hanging forever on a permanent
# auth or 4xx error.
_MAX_RETRIES = 5
_BASE_DELAY = 1.0
_MAX_DELAY = 30.0


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PolybacktestError(RuntimeError):
    """Base for all polybacktest client errors."""


class PolybacktestNotConfiguredError(PolybacktestError):
    """Raised when no API key is configured.  Callers should surface a
    UI hint to add the key in Settings → Data Sources → Providers
    rather than treating this as a 500.
    """


class PolybacktestAuthError(PolybacktestError):
    """401 / 403 from the API — bad or expired key."""


class PolybacktestRateLimitError(PolybacktestError):
    """All retries exhausted while still receiving 429."""


class PolybacktestUpstreamError(PolybacktestError):
    """5xx after all retries exhausted, or unrecognizable response."""


# ---------------------------------------------------------------------------
# Data shapes — mirror polybacktest's verified response keys, with one
# computed convenience: ``title`` is synthesized from market_type +
# start_time + winner so the UI never has to render an opaque ID.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolybacktestMarket:
    """A polybacktest Up/Down prediction market."""

    market_id: str
    coin: str
    slug: Optional[str]
    title: str  # synthesized — never None
    market_type: Optional[str]  # '5m' | '15m' | '1h' | '4h' | '24h'
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    winner: Optional[str]  # 'Up' | 'Down' | None
    final_volume: Optional[float]
    final_liquidity: Optional[float]
    coin_price_start: Optional[float]
    coin_price_end: Optional[float]
    condition_id: Optional[str]
    clob_token_up: Optional[str]
    clob_token_down: Optional[str]
    raw: dict[str, Any] = field(repr=False)


@dataclass(frozen=True)
class PolybacktestSnapshot:
    """One side of one snapshot (UP or DOWN) with its full order book."""

    market_id: str
    coin: str
    side: str  # 'up' | 'down'
    snapshot_id: Optional[str]
    observed_at_ms: int
    sequence: Optional[int]
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]
    best_bid: Optional[float]
    best_ask: Optional[float]
    coin_price: Optional[float]  # spot reference price (e.g. BTC USD)
    price_up: Optional[float]
    price_down: Optional[float]
    raw: dict[str, Any] = field(repr=False)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class PolybacktestClient:
    """Thin async REST wrapper.

    Construct once per import job (cheap — no network on init) and
    ``await client.close()`` when done so the underlying httpx client
    releases its socket pool cleanly.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        timeout_seconds: float = 60.0,
        rate_per_second: float = _DEFAULT_RATE_PER_SECOND,
        burst: int = _DEFAULT_BURST,
        max_concurrent: int = _MAX_CONCURRENT_REQUESTS,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        if not api_key:
            raise PolybacktestNotConfiguredError(
                "Polybacktest API key is empty.  Set it in Settings → Data Providers."
            )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/") or _DEFAULT_BASE_URL
        self._client: Optional[httpx.AsyncClient] = None
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._bucket = TokenBucket(
            capacity=float(max(1, burst)),
            tokens=float(max(1, burst)),
            refill_rate=float(max(0.1, rate_per_second)),
        )
        self._concurrency = asyncio.Semaphore(max(1, max_concurrent))
        # Counters bumped on every successful or rate-limited response —
        # surfaced through ``stats()`` so the worker can persist them
        # back to the import job for observability.
        self._call_count = 0
        self._rate_limited_count = 0
        self._bytes_in = 0

    # ── lifecycle ──────────────────────────────────────────────────────

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout_seconds),
                headers={
                    # Polybacktest expects X-API-Key (NOT Authorization: Bearer).
                    # Verified live 2026-05-05 — Bearer is silently accepted on
                    # some legacy v1/v3 paths but v2 endpoints reject it.
                    "X-API-Key": self._api_key,
                    "Accept": "application/json",
                    "User-Agent": "homerun-polybacktest-importer/2.0",
                },
                limits=httpx.Limits(
                    max_connections=_MAX_CONCURRENT_REQUESTS * 2,
                    max_keepalive_connections=_MAX_CONCURRENT_REQUESTS,
                ),
            )
        return self._client

    async def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None and not client.is_closed:
            try:
                await client.aclose()
            except Exception:
                logger.debug("polybacktest client close failed", exc_info=True)

    async def __aenter__(self) -> "PolybacktestClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # ── observability ──────────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        return {
            "api_calls": self._call_count,
            "rate_limited_count": self._rate_limited_count,
            "bytes_downloaded": self._bytes_in,
        }

    # ── core request loop ──────────────────────────────────────────────

    async def _wait_for_token(self) -> None:
        # Block until the local token bucket has a token available.
        while True:
            self._bucket.refill()
            if self._bucket.tokens >= 1:
                self._bucket.tokens -= 1
                return
            wait = self._bucket.wait_time(1)
            await asyncio.sleep(max(0.01, wait))

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Issue one request with rate-limit + retry handling.

        Returns the parsed JSON body on success.  Raises a typed
        :class:`PolybacktestError` subclass on failure.
        """
        client = self._get_client()
        attempt = 0
        last_exc: Optional[BaseException] = None

        # Drop None-valued params.
        clean_params: Optional[dict[str, Any]] = None
        if params is not None:
            clean_params = {k: v for k, v in params.items() if v is not None}
            if not clean_params:
                clean_params = None

        while attempt < self._max_retries:
            attempt += 1
            await self._wait_for_token()
            async with self._concurrency:
                try:
                    response = await client.request(method, path, params=clean_params)
                except httpx.RequestError as exc:
                    last_exc = exc
                    delay = min(_MAX_DELAY, _BASE_DELAY * (2 ** (attempt - 1)))
                    logger.warning(
                        "polybacktest %s %s network error, retry %d/%d in %.1fs: %s",
                        method,
                        path,
                        attempt,
                        self._max_retries,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
                    continue

                self._call_count += 1
                self._bytes_in += len(response.content or b"")

                if response.status_code == 429:
                    self._rate_limited_count += 1
                    retry_after = response.headers.get("Retry-After")
                    delay = _BASE_DELAY * (2 ** (attempt - 1))
                    try:
                        if retry_after is not None:
                            delay = max(delay, float(retry_after))
                    except (TypeError, ValueError):
                        pass
                    delay = min(_MAX_DELAY, delay)
                    if attempt >= self._max_retries:
                        raise PolybacktestRateLimitError(
                            f"polybacktest rate-limited after {attempt} attempts on {path}"
                        )
                    logger.info(
                        "polybacktest 429 on %s, sleeping %.1fs (attempt %d/%d)",
                        path,
                        delay,
                        attempt,
                        self._max_retries,
                    )
                    await asyncio.sleep(delay)
                    continue

                if response.status_code in (401, 403):
                    raise PolybacktestAuthError(
                        f"polybacktest auth failed ({response.status_code}) on {path} — "
                        "verify the API key in Settings → Data Providers"
                    )

                if 500 <= response.status_code < 600:
                    delay = min(_MAX_DELAY, _BASE_DELAY * (2 ** (attempt - 1)))
                    if attempt >= self._max_retries:
                        raise PolybacktestUpstreamError(
                            f"polybacktest {response.status_code} after {attempt} attempts on {path}"
                        )
                    logger.warning(
                        "polybacktest %d on %s, retry %d/%d in %.1fs",
                        response.status_code,
                        path,
                        attempt,
                        self._max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                if response.status_code >= 400:
                    detail = ""
                    try:
                        body = response.json()
                        if isinstance(body, dict):
                            detail = (
                                body.get("error")
                                or body.get("message")
                                or body.get("detail")
                                or ""
                            )
                    except Exception:
                        detail = (response.text or "").strip()[:300]
                    raise PolybacktestError(
                        f"polybacktest {response.status_code} on {path}: {detail or response.text[:200]}"
                    )

                # 2xx — parse JSON.
                try:
                    return response.json()
                except ValueError as exc:
                    raise PolybacktestUpstreamError(
                        f"polybacktest returned non-JSON response on {path}: {exc}"
                    ) from exc

        if last_exc is not None:
            raise PolybacktestUpstreamError(
                f"polybacktest network failure after {self._max_retries} attempts on {path}: {last_exc}"
            ) from last_exc
        raise PolybacktestUpstreamError(
            f"polybacktest exhausted retries on {path} with no recognizable response"
        )

    # ── endpoints ──────────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """Auth-free ping; useful for the Settings page status pill."""
        client = self._get_client()
        try:
            response = await client.get("/health", timeout=10.0)
            ok = 200 <= response.status_code < 300
            return {
                "ok": bool(ok),
                "status_code": int(response.status_code),
                "elapsed_ms": int(response.elapsed.total_seconds() * 1000)
                if response.elapsed
                else None,
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    async def list_markets(
        self,
        coin: str,
        *,
        offset: int = 0,
        limit: int = 100,
        search: Optional[str] = None,
        market_type: Optional[str] = None,
        resolved: Optional[bool] = None,
    ) -> tuple[list[PolybacktestMarket], int]:
        """Page through markets for a coin.

        Returns ``(markets, total)`` so the caller can paginate against
        the total count.  Pagination is offset-based on polybacktest v2.
        """
        coin_norm = (coin or "").strip().lower()
        if coin_norm not in _SUPPORTED_COINS:
            raise PolybacktestError(
                f"Unsupported coin '{coin}' (supported: {sorted(_SUPPORTED_COINS)})"
            )
        params: dict[str, Any] = {
            "coin": coin_norm,
            "limit": int(limit),
            "offset": int(offset),
        }
        if search:
            params["search"] = search
        if market_type:
            params["market_type"] = market_type
        if resolved is not None:
            params["resolved"] = "true" if resolved else "false"
        body = await self._request("GET", "/v2/markets", params=params)
        items = body.get("markets") or body.get("data") or body.get("items") or []
        markets: list[PolybacktestMarket] = []
        for raw in items if isinstance(items, list) else []:
            if not isinstance(raw, dict):
                continue
            market = _parse_market(coin_norm, raw)
            if market is not None:
                markets.append(market)
        total = int(body.get("total") or len(markets))
        return markets, total

    async def get_market(self, coin: str, market_id: str) -> PolybacktestMarket:
        coin_norm = (coin or "").strip().lower()
        body = await self._request(
            "GET",
            f"/v2/markets/{market_id}",
            params={"coin": coin_norm},
        )
        # Accept either ``{market: {...}}`` or the bare market object.
        raw = body.get("market") if isinstance(body, dict) and "market" in body else body
        if not isinstance(raw, dict):
            raise PolybacktestError(
                f"polybacktest returned no market payload for {market_id}"
            )
        market = _parse_market(coin_norm, raw)
        if market is None:
            raise PolybacktestError(
                f"polybacktest market {market_id} payload missing required fields"
            )
        return market

    async def get_snapshots(
        self,
        coin: str,
        market_id: str,
        *,
        start_ms: int,
        end_ms: int,
        offset: int = 0,
        limit: int = 1000,
        include_orderbook: bool = True,
    ) -> tuple[list[PolybacktestSnapshot], int]:
        """Fetch a page of book snapshots for a market.

        Returns ``(snapshots, total)``.  Each polybacktest snapshot contains
        BOTH UP and DOWN order books — we flatten that to two
        :class:`PolybacktestSnapshot` records (one per side) with the same
        observed_at and snapshot_id, so the importer writes them to
        separate synthetic ``token_id`` rows in microstructure.

        ``include_orderbook=True`` (the default for our use case) is
        REQUIRED to get full L2 depth — polybacktest's default of ``false``
        returns only best-bid/ask + summary metadata.
        """
        coin_norm = (coin or "").strip().lower()
        # Convert ms-epoch to ISO 8601 — polybacktest v2 accepts both,
        # but ISO is unambiguous across server timezones.
        start_iso = _ms_to_iso(start_ms)
        end_iso = _ms_to_iso(end_ms)
        params: dict[str, Any] = {
            "coin": coin_norm,
            "limit": min(1000, max(1, int(limit))),
            "offset": max(0, int(offset)),
            "start_time": start_iso,
            "end_time": end_iso,
            "include_orderbook": "true" if include_orderbook else "false",
        }
        body = await self._request(
            "GET",
            f"/v2/markets/{market_id}/snapshots",
            params=params,
        )
        items = body.get("snapshots") or body.get("data") or body.get("items") or []
        snapshots: list[PolybacktestSnapshot] = []
        for raw in items if isinstance(items, list) else []:
            if not isinstance(raw, dict):
                continue
            snapshots.extend(_parse_snapshot(coin_norm, market_id, raw))
        total = int(body.get("total") or len(snapshots))
        return snapshots, total


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def _parse_market(coin: str, raw: dict[str, Any]) -> Optional[PolybacktestMarket]:
    market_id = str(
        raw.get("market_id") or raw.get("id") or raw.get("slug") or ""
    ).strip()
    if not market_id:
        return None
    market_type = str(raw.get("market_type") or "").strip() or None
    start_time = _coerce_ts(raw.get("start_time") or raw.get("starts_at"))
    end_time = _coerce_ts(raw.get("end_time") or raw.get("ends_at"))
    winner = str(raw.get("winner") or "").strip() or None
    coin_lower = coin.lower()
    coin_price_start = _coerce_float(raw.get(f"{coin_lower}_price_start"))
    coin_price_end = _coerce_float(raw.get(f"{coin_lower}_price_end"))
    title = _synthesize_title(
        coin=coin,
        market_type=market_type,
        start_time=start_time,
        end_time=end_time,
        winner=winner,
        coin_price_start=coin_price_start,
        coin_price_end=coin_price_end,
    )
    return PolybacktestMarket(
        market_id=market_id,
        coin=coin,
        slug=str(raw.get("slug") or "") or None,
        title=title,
        market_type=market_type,
        start_time=start_time,
        end_time=end_time,
        winner=winner,
        final_volume=_coerce_float(raw.get("final_volume")),
        final_liquidity=_coerce_float(raw.get("final_liquidity")),
        coin_price_start=coin_price_start,
        coin_price_end=coin_price_end,
        condition_id=str(raw.get("condition_id") or "") or None,
        clob_token_up=str(raw.get("clob_token_up") or "") or None,
        clob_token_down=str(raw.get("clob_token_down") or "") or None,
        raw=dict(raw),
    )


def _parse_snapshot(
    coin: str,
    market_id: str,
    raw: dict[str, Any],
) -> list[PolybacktestSnapshot]:
    """Split one polybacktest snapshot into one record per side (UP/DOWN)."""
    observed_at = _coerce_ts(raw.get("time") or raw.get("timestamp"))
    if observed_at is None:
        return []
    observed_at_ms = int(observed_at.timestamp() * 1000)
    snap_id = str(raw.get("id") or "") or None
    coin_price = _coerce_float(raw.get(f"{coin.lower()}_price"))
    price_up = _coerce_float(raw.get("price_up"))
    price_down = _coerce_float(raw.get("price_down"))

    out: list[PolybacktestSnapshot] = []
    for side, ob_key in (("up", "orderbook_up"), ("down", "orderbook_down")):
        ob = raw.get(ob_key)
        bids: list[tuple[float, float]] = []
        asks: list[tuple[float, float]] = []
        if isinstance(ob, dict):
            bids = _parse_levels(ob.get("bids"))
            asks = _parse_levels(ob.get("asks"))
        # Even when the orderbook is absent (include_orderbook=false),
        # we still emit a top-of-book record using price_up / price_down
        # so downstream consumers always see something.
        best_bid = _first_price(bids) if bids else None
        best_ask = _first_price(asks) if asks else None
        if best_bid is None and best_ask is None:
            # Synthesize from price_up / price_down when we have no book.
            mid = price_up if side == "up" else price_down
            if mid is not None:
                best_bid = mid
                best_ask = mid

        out.append(
            PolybacktestSnapshot(
                market_id=str(raw.get("market_id") or market_id),
                coin=coin,
                side=side,
                snapshot_id=snap_id,
                observed_at_ms=observed_at_ms,
                sequence=None,  # polybacktest doesn't expose a sequence number
                bids=bids,
                asks=asks,
                best_bid=best_bid,
                best_ask=best_ask,
                coin_price=coin_price,
                price_up=price_up,
                price_down=price_down,
                raw={"orderbook": ob, "source_snapshot": raw},
            )
        )
    return out


def _synthesize_title(
    *,
    coin: str,
    market_type: Optional[str],
    start_time: Optional[datetime],
    end_time: Optional[datetime],
    winner: Optional[str],
    coin_price_start: Optional[float],
    coin_price_end: Optional[float],
) -> str:
    """Build a human-readable market title from the available fields."""
    coin_label = coin.upper()
    horizon = market_type or "?"
    when = (
        start_time.strftime("%Y-%m-%d %H:%M UTC")
        if start_time
        else "?"
    )
    base = f"{coin_label} Up/Down · {horizon} · {when}"
    if winner and coin_price_start is not None and coin_price_end is not None:
        return (
            f"{base} (settled {winner.upper()}: "
            f"${coin_price_start:,.2f} → ${coin_price_end:,.2f})"
        )
    if winner:
        return f"{base} (settled {winner.upper()})"
    if coin_price_start is not None:
        return f"{base} (open ${coin_price_start:,.2f})"
    return base


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SUPPORTED_COINS = frozenset({"btc", "eth", "sol"})


def supported_coins() -> tuple[str, ...]:
    """Coins polybacktest exposes through its v2 API."""
    return tuple(sorted(_SUPPORTED_COINS))


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN
        return None
    return result


def _coerce_ts(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        v = float(value)
        if v > 1e12:
            v = v / 1000.0
        try:
            return datetime.fromtimestamp(v, tz=timezone.utc)
        except (TypeError, ValueError, OSError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _ms_to_iso(ms: int) -> str:
    return (
        datetime.fromtimestamp(int(ms) / 1000.0, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _parse_levels(raw: Any) -> list[tuple[float, float]]:
    """Normalize a polybacktest depth payload to ``[(price, size)]``."""
    if not isinstance(raw, list):
        return []
    out: list[tuple[float, float]] = []
    for level in raw:
        if isinstance(level, dict):
            price = _coerce_float(level.get("price") or level.get("p"))
            size = _coerce_float(level.get("size") or level.get("amount") or level.get("q"))
        elif isinstance(level, (list, tuple)) and len(level) >= 2:
            price = _coerce_float(level[0])
            size = _coerce_float(level[1])
        else:
            continue
        if price is None or size is None:
            continue
        out.append((price, size))
    return out


def _first_price(levels: list[tuple[float, float]]) -> Optional[float]:
    if not levels:
        return None
    return levels[0][0]


# ---------------------------------------------------------------------------
# Factory — builds a client from AppSettings (decrypts secret).  This
# is the single entry point used by routes + the import worker so the
# config-resolution path lives in one spot.
# ---------------------------------------------------------------------------


async def build_client_from_settings() -> PolybacktestClient:
    """Construct a configured client by reading AppSettings.

    Raises :class:`PolybacktestNotConfiguredError` when the operator
    hasn't set the API key — callers should treat that as a 412
    "configure the provider in Settings" UX, not a 500.
    """
    from sqlalchemy import select
    from models.database import AppSettings, AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        row = (await session.execute(select(AppSettings))).scalar_one_or_none()
    api_key = decrypt_or_passthrough(getattr(row, "polybacktest_api_key", None) if row else None)
    base_url = (
        (getattr(row, "polybacktest_base_url", None) or "").strip() if row else ""
    ) or _DEFAULT_BASE_URL
    if not api_key:
        raise PolybacktestNotConfiguredError(
            "Polybacktest API key is not configured.  Add it in Settings → Data Providers."
        )
    return PolybacktestClient(api_key=api_key, base_url=base_url)


def decrypt_or_passthrough(value: Optional[str]) -> Optional[str]:
    """Decrypt an AppSettings secret column, falling back to plaintext."""
    if not value:
        return None
    try:
        from utils.secrets import decrypt_secret

        decrypted = decrypt_secret(value)
        return decrypted if decrypted is not None else value
    except Exception:
        return value


__all__ = [
    "PolybacktestClient",
    "PolybacktestMarket",
    "PolybacktestSnapshot",
    "PolybacktestError",
    "PolybacktestAuthError",
    "PolybacktestNotConfiguredError",
    "PolybacktestRateLimitError",
    "PolybacktestUpstreamError",
    "build_client_from_settings",
    "supported_coins",
]
