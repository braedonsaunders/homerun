"""Reconstruct ``CRYPTO_UPDATE`` events from imported provider book parquet.

Why this exists
===============
Event-driven crypto strategies (``subscriptions=[EventType.CRYPTO_UPDATE]``)
read market state from ``crypto.update.dispatch`` bus envelopes that the live
``market_runtime`` emits.  In backtest those envelopes are replayed from the
recorded-event bus — but only for windows the live dispatcher actually
recorded.  Operator-imported historical book data (e.g. the ``polybacktest``
provider, landing as canonical ``SNAPSHOT_SCHEMA`` parquet +
``ProviderDataset(storage_type='parquet')`` rows) has NO recorded dispatch
events, so an event-driven crypto strategy could never be backtested against
imported windows.

This module closes that gap *at backtest time* by reconstructing the same
``DataEvent(event_type=CRYPTO_UPDATE, ...)`` shape from the canonical book
parquet plus the self-describing ``ProviderDataset.payload_json`` metadata
(condition_id, clob_token_up/down, market start/end).  It is the inverse of
:class:`services.backtest.bus_book_replay.BusBookReplay` (which reconstructs
book snapshots *from* dispatch parquet) and follows the same blessed pattern:
canonical parquet is the source of truth; the event stream is derived.

Nothing is persisted — events are materialised on demand and binned into the
backtester's tick grid.  The recorded-event bus stays pure (live-only),
avoiding retention/pruning contention.
"""
from __future__ import annotations

import bisect
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import unquote, urlparse

logger = logging.getLogger(__name__)

# Providers whose imported parquet we synthesize crypto_update events from.
# polybacktest is the first; the list is open for future providers that land
# canonical up/down book parquet with the same self-describing payload.
DEFAULT_SYNTH_PROVIDERS: tuple[str, ...] = ("polybacktest",)


def _uri_to_path(uri: str) -> Path:
    """file:///C:/foo -> C:/foo (Windows) ; file:///foo -> /foo (POSIX)."""
    parsed = urlparse(uri)
    path = unquote(parsed.path)
    if len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return Path(path)


def _clamp01(p: Any) -> Optional[float]:
    if p is None:
        return None
    try:
        p = float(p)
    except (TypeError, ValueError):
        return None
    if p <= 0.0 or p >= 1.0:
        return None
    return p


class _BookSeries:
    """Sorted (observed_at_us, best_bid, best_ask) lookup for one token."""

    __slots__ = ("_us", "_bid", "_ask")

    def __init__(self) -> None:
        self._us: list[int] = []
        self._bid: list[Optional[float]] = []
        self._ask: list[Optional[float]] = []

    def add_file(self, path: Path, *, start_us: int, end_us: int) -> None:
        import pyarrow.parquet as pq

        try:
            t = pq.read_table(str(path), columns=["observed_at_us", "best_bid", "best_ask"])
        except Exception as exc:  # noqa: BLE001
            logger.debug("crypto_update_synth: unreadable %s: %s", path, exc)
            return
        obs = t.column("observed_at_us").to_pylist()
        bid = t.column("best_bid").to_pylist()
        ask = t.column("best_ask").to_pylist()
        # Pad a little so as-of lookups near the window edge still resolve.
        lo = start_us - 60_000_000
        hi = end_us + 60_000_000
        for o, b, a in zip(obs, bid, ask):
            if o is None or o < lo or o > hi:
                continue
            self._us.append(int(o))
            self._bid.append(b)
            self._ask.append(a)

    def finalize(self) -> None:
        if not self._us:
            return
        order = sorted(range(len(self._us)), key=lambda i: self._us[i])
        self._us = [self._us[i] for i in order]
        self._bid = [self._bid[i] for i in order]
        self._ask = [self._ask[i] for i in order]

    def as_of(self, ts_us: int, *, max_staleness_us: int = 30_000_000) -> Optional[tuple[float, float]]:
        if not self._us:
            return None
        i = bisect.bisect_right(self._us, ts_us) - 1
        if i < 0:
            return None
        if ts_us - self._us[i] > max_staleness_us:
            return None
        b, a = self._bid[i], self._ask[i]
        if b is None or a is None:
            return None
        return (float(b), float(a))

    def __bool__(self) -> bool:
        return bool(self._us)


class _SynthMarket:
    """One importable market: metadata + UP/DOWN book series."""

    __slots__ = (
        "market_id", "condition_id", "slug", "title", "coin", "timeframe",
        "start_us", "end_us", "up_token", "down_token", "price_to_beat",
        "up_series", "down_series",
    )

    def __init__(self, **kw: Any) -> None:
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def market_key(self) -> str:
        return str(self.condition_id or self.market_id or self.slug or "")

    def alive_at(self, ts_us: int) -> bool:
        if self.start_us is not None and ts_us < self.start_us:
            return False
        if self.end_us is not None and ts_us >= self.end_us:
            return False
        return True

    def to_market_dict(self, tick: datetime) -> Optional[dict[str, Any]]:
        ts_us = int(tick.timestamp() * 1_000_000)
        up_book = self.up_series.as_of(ts_us) if self.up_series else None
        down_book = self.down_series.as_of(ts_us) if self.down_series else None
        if up_book is None and down_book is None:
            return None
        up_bid = up_ask = down_bid = down_ask = None
        if up_book is not None:
            up_bid, up_ask = up_book
        if down_book is not None:
            down_bid, down_ask = down_book
        # Derive a missing side from the binary complement P(down)=1-P(up).
        if up_book is None and down_book is not None:
            up_bid = _clamp01(1.0 - down_ask) if down_ask is not None else None
            up_ask = _clamp01(1.0 - down_bid) if down_bid is not None else None
        if down_book is None and up_book is not None:
            down_bid = _clamp01(1.0 - up_ask) if up_ask is not None else None
            down_ask = _clamp01(1.0 - up_bid) if up_bid is not None else None

        def _mid(b: Optional[float], a: Optional[float]) -> Optional[float]:
            if b is not None and a is not None:
                return round((b + a) / 2.0, 6)
            return b if b is not None else a

        up_price = _mid(up_bid, up_ask)
        down_price = _mid(down_bid, down_ask)
        if up_price is None and down_price is None:
            return None

        seconds_left = None
        is_live = True
        if self.end_us is not None:
            seconds_left = max(0, int((self.end_us - ts_us) / 1_000_000))
        if self.start_us is not None and self.end_us is not None:
            is_live = self.start_us <= ts_us < self.end_us

        spread = None
        if up_bid is not None and up_ask is not None:
            spread = round(up_ask - up_bid, 6)

        return {
            "id": str(self.market_id or ""),
            "condition_id": self.condition_id,
            "slug": self.slug,
            "question": self.title or self.slug,
            "asset": (self.coin or "").upper(),
            "timeframe": self.timeframe,
            "start_time": _iso_us(self.start_us),
            "end_time": _iso_us(self.end_us),
            "seconds_left": seconds_left,
            "is_live": is_live,
            "is_current": True,
            "up_price": up_price,
            "down_price": down_price,
            "best_bid": up_bid,
            "best_ask": up_ask,
            "spread": spread,
            "clob_token_ids": [self.up_token or "", self.down_token or ""],
            "up_token_index": 0,
            "down_token_index": 1,
            "price_to_beat": self.price_to_beat,
            "fees_enabled": True,
            "event_slug": self.slug,
            "event_title": self.title,
            "liquidity": 0.0,
            "volume": 0.0,
            "_synthetic_source": "crypto_update_synthesizer",
        }


def _iso_us(us: Optional[int]) -> Optional[str]:
    if us is None:
        return None
    return datetime.fromtimestamp(us / 1_000_000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


async def _load_synth_markets(
    *,
    start: datetime,
    end: datetime,
    providers: tuple[str, ...],
    token_scope: Optional[set[str]],
) -> list[_SynthMarket]:
    """Load every importable market whose ProviderDataset window overlaps
    [start, end] and whose payload is self-describing (schema snapshots_v2).
    """
    from sqlalchemy import select
    from models.database import AsyncSessionLocal, ProviderDataset

    start_naive = start.astimezone(timezone.utc).replace(tzinfo=None) if start.tzinfo else start
    end_naive = end.astimezone(timezone.utc).replace(tzinfo=None) if end.tzinfo else end

    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(ProviderDataset).where(
                    ProviderDataset.provider.in_(list(providers)),
                    ProviderDataset.storage_type == "parquet",
                    ProviderDataset.start_ts <= end_naive,
                    ProviderDataset.end_ts >= start_naive,
                )
            )
        ).scalars().all()

    markets: list[_SynthMarket] = []
    for r in rows:
        payload = r.payload_json or {}
        up_tok = payload.get("clob_token_up")
        down_tok = payload.get("clob_token_down")
        token_ids = list(r.token_ids_json or [])
        # Fall back to token_ids_json positions when payload lacks explicit
        # up/down (older snapshots_v1 rows).  Order is not guaranteed there,
        # so we only trust it as a last resort.
        if not up_tok and len(token_ids) >= 1:
            up_tok = token_ids[0]
        if not down_tok and len(token_ids) >= 2:
            down_tok = token_ids[1]
        up_tok = str(up_tok) if up_tok else None
        down_tok = str(down_tok) if down_tok else None
        if not up_tok and not down_tok:
            continue
        if token_scope is not None:
            if (up_tok not in token_scope) and (down_tok not in token_scope):
                continue
        if not r.storage_uri or not r.storage_uri.startswith("file://"):
            continue
        window_dir = _uri_to_path(r.storage_uri)

        m = _SynthMarket(
            market_id=payload.get("market_id") or r.external_id,
            condition_id=payload.get("condition_id"),
            slug=payload.get("slug") or r.external_slug,
            title=payload.get("title") or r.title,
            coin=payload.get("coin") or r.coin,
            timeframe=payload.get("market_type"),
            start_us=_parse_iso_us(payload.get("market_start_time")),
            end_us=_parse_iso_us(payload.get("market_end_time")),
            up_token=up_tok,
            down_token=down_tok,
            price_to_beat=payload.get("coin_price_start"),
            up_series=_BookSeries(),
            down_series=_BookSeries(),
        )
        # Fall back to the dataset's data span when the market window times
        # weren't captured (older imports) so alive_at / seconds_left still work.
        if m.start_us is None and r.start_ts is not None:
            m.start_us = int(r.start_ts.replace(tzinfo=timezone.utc).timestamp() * 1_000_000)
        if m.end_us is None and r.end_ts is not None:
            m.end_us = int(r.end_ts.replace(tzinfo=timezone.utc).timestamp() * 1_000_000)

        win_start_us = int(start.timestamp() * 1_000_000)
        win_end_us = int(end.timestamp() * 1_000_000)
        from services.external_data.parquet_schema import _safe_segment

        for tok, series in ((up_tok, m.up_series), (down_tok, m.down_series)):
            if not tok:
                continue
            for cand in (
                window_dir / f"snapshots__{_safe_segment(tok)}.parquet",
                window_dir / f"snapshots__{tok}.parquet",
            ):
                if cand.exists():
                    series.add_file(cand, start_us=win_start_us, end_us=win_end_us)
                    break
            series.finalize()
        if not m.up_series and not m.down_series:
            continue
        markets.append(m)
    return markets


def _parse_iso_us(s: Any) -> Optional[int]:
    if not s or not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000)


async def synthesize_crypto_update_events(
    *,
    ticks: list[datetime],
    exclude_market_keys: Optional[set[str]] = None,
    providers: tuple[str, ...] = DEFAULT_SYNTH_PROVIDERS,
    token_scope: Optional[Iterable[str]] = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Build one ``DataEvent(CRYPTO_UPDATE)`` per tick that has >=1 alive
    importable market, reconstructed from imported book parquet.

    Returns ``(events, stats)``.  Each event's ``payload['markets']`` carries
    the same per-market dict shape the live ``crypto.update.dispatch`` emits,
    so the backtester's ``event_kind == 'crypto_update'`` tick loop consumes
    them unchanged.

    ``exclude_market_keys`` lets the caller skip markets that already have
    real recorded dispatch coverage in the window (gap-fill semantics — real
    recorded data stays authoritative).
    """
    from services.data_events import DataEvent, EventType

    if not ticks:
        return [], {"markets": 0, "events": 0, "skipped_excluded": 0}

    start = ticks[0]
    end = ticks[-1]
    scope = {str(t) for t in token_scope} if token_scope is not None else None
    exclude = exclude_market_keys or set()

    markets = await _load_synth_markets(
        start=start, end=end, providers=providers, token_scope=scope,
    )

    skipped = 0
    active: list[_SynthMarket] = []
    for m in markets:
        if m.market_key() in exclude:
            skipped += 1
            continue
        active.append(m)

    events: list[Any] = []
    for tick in ticks:
        ts_us = int(tick.timestamp() * 1_000_000)
        market_dicts: list[dict[str, Any]] = []
        for m in active:
            if not m.alive_at(ts_us):
                continue
            md = m.to_market_dict(tick)
            if md is not None:
                market_dicts.append(md)
        if not market_dicts:
            continue
        events.append(
            DataEvent(
                event_type=EventType.CRYPTO_UPDATE,
                source="crypto_update_synthesizer",
                timestamp=tick,
                payload={
                    "markets": market_dicts,
                    "trigger": "crypto_update_synthesizer",
                    "event_source": "imported_parquet",
                },
            )
        )

    stats = {
        "markets_loaded": len(markets),
        "markets_active": len(active),
        "skipped_excluded": skipped,
        "events": len(events),
    }
    return events, stats


__all__ = ["synthesize_crypto_update_events", "DEFAULT_SYNTH_PROVIDERS"]
