"""crypto_update event projection over the canonical book plane.

Event-driven crypto strategies read market state from ``crypto.update.dispatch``
events the live dispatcher emits. For windows that were only *imported* (e.g.
polybacktest book parquet) those events were never recorded, so the strategy
could not be backtested against them. This module reconstructs the same
``DataEvent(CRYPTO_UPDATE)`` shape from the canonical book plane (via
:class:`MarketDataView`) plus the self-describing ``ProviderDataset`` metadata.

It is the inverse of ``BusBookReplay`` (which derives books from dispatch
events) and supersedes the earlier standalone ``crypto_update_synthesizer``
bolt-on: prices now come from the unified view's point-in-time book access
(one book-reading path), not a private parquet reader. Nothing is persisted;
events are materialised on demand for replay.

Gap-fill semantics: callers pass ``exclude_market_keys`` for markets that
already have *recorded* dispatch coverage, so recorded data stays authoritative
and the projection only fills the rest.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import unquote, urlparse

from services.marketdata.view import MarketDataView

logger = logging.getLogger(__name__)

DEFAULT_PROJECTION_PROVIDERS: tuple[str, ...] = ("polybacktest",)
DEFAULT_CADENCE_SECONDS: float = 2.0
# Treat a top-of-book as absent if the freshest snapshot is older than this.
DEFAULT_MAX_STALENESS_SECONDS: float = 30.0


def _uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    path = unquote(parsed.path)
    if len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return Path(path)


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


def _clamp01(p: Any) -> Optional[float]:
    if p is None:
        return None
    try:
        p = float(p)
    except (TypeError, ValueError):
        return None
    return p if 0.0 < p < 1.0 else None


class ProjectedMarket:
    """Metadata for one importable market the projection can reconstruct."""

    __slots__ = (
        "market_id", "condition_id", "slug", "title", "coin", "timeframe",
        "start_us", "end_us", "up_token", "down_token", "price_to_beat",
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

    @property
    def tokens(self) -> tuple[str, ...]:
        return tuple(t for t in (self.up_token, self.down_token) if t)


async def load_projected_markets(
    *,
    start: datetime,
    end: datetime,
    providers: Iterable[str] = DEFAULT_PROJECTION_PROVIDERS,
    token_scope: Optional[set[str]] = None,
) -> list[ProjectedMarket]:
    """Load importable-market metadata whose dataset window overlaps
    ``[start, end]`` from the self-describing ``ProviderDataset`` rows."""
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

    markets: list[ProjectedMarket] = []
    for r in rows:
        payload = r.payload_json or {}
        up_tok = payload.get("clob_token_up")
        down_tok = payload.get("clob_token_down")
        token_ids = list(r.token_ids_json or [])
        # snapshots_v1 fallback: positions in token_ids_json (order not
        # guaranteed — last resort only).
        if not up_tok and len(token_ids) >= 1:
            up_tok = token_ids[0]
        if not down_tok and len(token_ids) >= 2:
            down_tok = token_ids[1]
        up_tok = str(up_tok) if up_tok else None
        down_tok = str(down_tok) if down_tok else None
        if not up_tok and not down_tok:
            continue
        if token_scope is not None and (up_tok not in token_scope) and (down_tok not in token_scope):
            continue

        start_us = _parse_iso_us(payload.get("market_start_time"))
        end_us = _parse_iso_us(payload.get("market_end_time"))
        if start_us is None and r.start_ts is not None:
            start_us = int(r.start_ts.replace(tzinfo=timezone.utc).timestamp() * 1_000_000)
        if end_us is None and r.end_ts is not None:
            end_us = int(r.end_ts.replace(tzinfo=timezone.utc).timestamp() * 1_000_000)

        markets.append(ProjectedMarket(
            market_id=payload.get("market_id") or r.external_id,
            condition_id=payload.get("condition_id"),
            slug=payload.get("slug") or r.external_slug,
            title=payload.get("title") or r.title,
            coin=payload.get("coin") or r.coin,
            timeframe=payload.get("market_type"),
            start_us=start_us,
            end_us=end_us,
            up_token=up_tok,
            down_token=down_tok,
            price_to_beat=payload.get("coin_price_start"),
        ))
    return markets


def _market_dict(
    m: ProjectedMarket,
    *,
    tick: datetime,
    up_bid: Optional[float],
    up_ask: Optional[float],
    down_bid: Optional[float],
    down_ask: Optional[float],
) -> Optional[dict[str, Any]]:
    # Derive a missing side via the binary complement P(down)=1-P(up).
    if up_bid is None and up_ask is None and (down_bid is not None or down_ask is not None):
        up_bid = _clamp01(1.0 - down_ask) if down_ask is not None else None
        up_ask = _clamp01(1.0 - down_bid) if down_bid is not None else None
    if down_bid is None and down_ask is None and (up_bid is not None or up_ask is not None):
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

    ts_us = int(tick.timestamp() * 1_000_000)
    seconds_left = max(0, int((m.end_us - ts_us) / 1_000_000)) if m.end_us is not None else None
    is_live = (m.start_us is None or m.start_us <= ts_us) and (m.end_us is None or ts_us < m.end_us)
    spread = round(up_ask - up_bid, 6) if (up_bid is not None and up_ask is not None) else None

    return {
        "id": str(m.market_id or ""),
        "condition_id": m.condition_id,
        "slug": m.slug,
        "question": m.title or m.slug,
        "asset": (m.coin or "").upper(),
        "timeframe": m.timeframe,
        "start_time": _iso_us(m.start_us),
        "end_time": _iso_us(m.end_us),
        "seconds_left": seconds_left,
        "is_live": is_live,
        "is_current": True,
        "up_price": up_price,
        "down_price": down_price,
        "best_bid": up_bid,
        "best_ask": up_ask,
        "spread": spread,
        "clob_token_ids": [m.up_token or "", m.down_token or ""],
        "up_token_index": 0,
        "down_token_index": 1,
        "price_to_beat": m.price_to_beat,
        "fees_enabled": True,
        "event_slug": m.slug,
        "event_title": m.title,
        "liquidity": 0.0,
        "volume": 0.0,
        "_synthetic_source": "marketdata.projection",
    }


def _iso_us(us: Optional[int]) -> Optional[str]:
    if us is None:
        return None
    return datetime.fromtimestamp(us / 1_000_000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


async def project_crypto_update_events(
    *,
    start: datetime,
    end: datetime,
    cadence_seconds: float = DEFAULT_CADENCE_SECONDS,
    providers: Iterable[str] = DEFAULT_PROJECTION_PROVIDERS,
    token_scope: Optional[Iterable[str]] = None,
    exclude_market_keys: Optional[set[str]] = None,
    max_staleness_seconds: float = DEFAULT_MAX_STALENESS_SECONDS,
    view: Optional[MarketDataView] = None,
    markets: Optional[list[ProjectedMarket]] = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Reconstruct ``DataEvent(CRYPTO_UPDATE)`` events from imported book parquet.

    Emits one event per cadence tick that has >=1 alive market, each carrying
    the per-market dict shape the live dispatch produces (so the backtest
    ``crypto_update`` tick loop consumes them unchanged). Prices come from the
    unified :class:`MarketDataView`'s point-in-time ``book_at``.

    ``markets`` / ``view`` may be supplied pre-built (skips the DB metadata
    load / parquet view build) — used by tests and by callers that already
    hold a view over the universe.

    Returns ``(events, stats)``.
    """
    from services.data_events import DataEvent, EventType

    scope = {str(t) for t in token_scope} if token_scope is not None else None
    exclude = exclude_market_keys or set()

    if markets is None:
        markets = await load_projected_markets(
            start=start, end=end, providers=tuple(providers), token_scope=scope,
        )
    active = [m for m in markets if m.market_key() not in exclude and m.tokens]
    if not active:
        return [], {"markets_loaded": len(markets), "markets_active": 0, "events": 0}

    # Build (or reuse) a view over every token the active markets touch.
    if view is None:
        all_tokens = sorted({t for m in active for t in m.tokens})
        view = await MarketDataView.build(
            token_ids=all_tokens, start=start, end=end, providers=tuple(providers),
        )

    cadence_us = max(1, int(cadence_seconds * 1_000_000))
    start_us = int(start.astimezone(timezone.utc).timestamp() * 1_000_000) if start.tzinfo else int(start.timestamp() * 1_000_000)
    end_us = int(end.astimezone(timezone.utc).timestamp() * 1_000_000) if end.tzinfo else int(end.timestamp() * 1_000_000)

    events: list[Any] = []
    n_market_dicts = 0
    tick_us = start_us
    while tick_us <= end_us:
        tick = datetime.fromtimestamp(tick_us / 1_000_000, tz=timezone.utc)
        market_dicts: list[dict[str, Any]] = []
        for m in active:
            if not m.alive_at(tick_us):
                continue
            up_snap = await view.book_at(m.up_token, tick, max_staleness_seconds=max_staleness_seconds) if m.up_token else None
            down_snap = await view.book_at(m.down_token, tick, max_staleness_seconds=max_staleness_seconds) if m.down_token else None
            up_bid = up_snap.bids[0].price if (up_snap and up_snap.bids) else None
            up_ask = up_snap.asks[0].price if (up_snap and up_snap.asks) else None
            down_bid = down_snap.bids[0].price if (down_snap and down_snap.bids) else None
            down_ask = down_snap.asks[0].price if (down_snap and down_snap.asks) else None
            md = _market_dict(m, tick=tick, up_bid=up_bid, up_ask=up_ask, down_bid=down_bid, down_ask=down_ask)
            if md is not None:
                market_dicts.append(md)
        if market_dicts:
            n_market_dicts += len(market_dicts)
            events.append(DataEvent(
                event_type=EventType.CRYPTO_UPDATE,
                source="marketdata.projection",
                timestamp=tick,
                payload={
                    "markets": market_dicts,
                    "trigger": "marketdata.projection",
                    "event_source": "imported_parquet",
                },
            ))
        tick_us += cadence_us

    stats = {
        "markets_loaded": len(markets),
        "markets_active": len(active),
        "events": len(events),
        "market_dicts": n_market_dicts,
        "cadence_seconds": cadence_seconds,
    }
    return events, stats


__all__ = [
    "ProjectedMarket",
    "load_projected_markets",
    "project_crypto_update_events",
    "DEFAULT_PROJECTION_PROVIDERS",
    "DEFAULT_CADENCE_SECONDS",
]
