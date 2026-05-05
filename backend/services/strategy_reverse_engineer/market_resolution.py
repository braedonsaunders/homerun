"""Market resolution lookup — winner / settle prices per market.

The reverse-engineer analytics engine needs to know whether each
wallet trade's outcome won or lost in order to compute realized P/L,
the two-leg (paired vs directional) decomposition, and dominant-side
win rates by skew bucket.

Polymarket's ``/trades`` endpoint does NOT include market resolution
info on each trade row — we have to look it up by ``conditionId``
or ``event_slug``.  Two source paths:

  1. **Polybacktest** (preferred for crypto Up/Down markets) — the
     market metadata we already cache in ``ProviderDataset.payload_json``
     contains ``winner: 'Up' | 'Down' | None``, plus ``btc_price_start``
     / ``btc_price_end``.  Free for already-imported datasets; one
     ``/v2/markets/by-slug`` call per missing slug otherwise.

  2. **Polymarket Gamma API** (fallback for non-crypto markets) —
     ``/markets/{condition_id}`` returns ``winner_outcome_index``,
     ``resolved_at``, etc.  Only used when polybacktest doesn't cover
     the market.

This module exposes a single async helper that takes a list of
normalized wallet trades and returns a ``{slug or condition_id: Resolution}``
map.  It is async-batch friendly (uses ``asyncio.gather`` with a
sensible concurrency cap) and silently skips markets it can't resolve
— the analytics engine treats unresolved trades as P/L-uncertain
rather than as confirmed losses, so a missing resolution doesn't
poison the headline numbers.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy import select

from models.database import AsyncSessionLocal, ProviderDataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketResolution:
    """Per-market resolution snapshot."""

    market_id: Optional[str]            # polymarket conditionId
    event_slug: Optional[str]           # polymarket slug
    winner_outcome: Optional[str]       # 'YES' | 'NO' | 'UP' | 'DOWN' | None
    coin_price_start: Optional[float]
    coin_price_end: Optional[float]
    final_volume_usdc: Optional[float]
    final_liquidity_usdc: Optional[float]
    resolved_at: Optional[str]          # ISO-8601 string when known
    source: str                         # 'polybacktest' | 'polymarket'

    def did_outcome_win(self, outcome: str) -> Optional[bool]:
        """True/False if the given outcome was the winner; None if unresolved.

        Outcome can be 'YES'/'NO'/'UP'/'DOWN' or any case variant —
        ``Up`` from polymarket trades and ``UP`` from our normalizer
        both work.
        """
        if self.winner_outcome is None:
            return None
        return self.winner_outcome.upper() == (outcome or "").upper()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def resolve_markets_for_trades(
    trades: list[dict[str, Any]],
    *,
    use_polybacktest: bool = True,
    use_polymarket: bool = True,
    concurrency: int = 8,
) -> dict[str, MarketResolution]:
    """Build a {slug_or_market_id: MarketResolution} lookup.

    Resolution priority per market:
      1. Already-imported polybacktest datasets (zero network calls)
      2. Live polybacktest /v2/markets/by-slug (one call per slug)
      3. Polymarket Gamma /markets/{conditionId} (one call per market)

    Returns whichever the lookup found; unresolved markets are simply
    absent from the dict.
    """
    if not trades:
        return {}

    # Index trades by slug + market_id so we know what to look up.
    slugs_needed: set[str] = set()
    market_ids_needed: set[str] = set()
    for t in trades:
        if t.get("event_slug"):
            slugs_needed.add(str(t["event_slug"]))
        if t.get("market_id"):
            market_ids_needed.add(str(t["market_id"]))

    resolutions: dict[str, MarketResolution] = {}

    # ── Step 1: free, in-process polybacktest catalog ────────────────
    if use_polybacktest and slugs_needed:
        cached = await _resolve_from_provider_catalog(slugs_needed)
        resolutions.update(cached)

    # ── Step 2: live polybacktest by-slug (+ list pages) ─────────────
    if use_polybacktest:
        unresolved_slugs = slugs_needed - set(resolutions.keys())
        if unresolved_slugs:
            try:
                live = await _resolve_from_polybacktest_live(
                    unresolved_slugs, concurrency=concurrency
                )
                resolutions.update(live)
            except Exception as exc:
                logger.info(
                    "polybacktest live resolution failed (continuing with polymarket fallback): %s",
                    exc,
                )

    # ── Step 3: polymarket gamma by conditionId (fallback) ───────────
    if use_polymarket:
        # For markets we still haven't resolved, try by conditionId.
        slug_to_conditions = {
            str(t["event_slug"]): str(t["market_id"])
            for t in trades
            if t.get("event_slug") and t.get("market_id")
        }
        unresolved_conditions: set[str] = set()
        for slug, cond in slug_to_conditions.items():
            if slug not in resolutions and cond not in resolutions:
                unresolved_conditions.add(cond)
        if unresolved_conditions:
            try:
                pm = await _resolve_from_polymarket(
                    unresolved_conditions, concurrency=concurrency
                )
                resolutions.update(pm)
            except Exception as exc:
                logger.info("polymarket gamma resolution failed: %s", exc)

    return resolutions


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


async def _resolve_from_provider_catalog(
    slugs_needed: set[str],
) -> dict[str, MarketResolution]:
    """Read polybacktest market metadata that's already cached locally.

    Every imported polybacktest dataset stashes the original market raw
    response in ``ProviderDataset.payload_json``.  Mining that gives us
    winners + prices for free.
    """
    if not slugs_needed:
        return {}
    out: dict[str, MarketResolution] = {}
    async with AsyncSessionLocal() as session:
        stmt = select(ProviderDataset).where(
            ProviderDataset.provider == "polybacktest",
            ProviderDataset.external_slug.in_(list(slugs_needed)),
        )
        rows = list((await session.execute(stmt)).scalars().all())
    for row in rows:
        payload = row.payload_json or {}
        winner_raw = payload.get("winner")
        if isinstance(winner_raw, str) and winner_raw.strip():
            winner = winner_raw.strip().upper()  # 'UP' | 'DOWN'
        else:
            winner = None
        slug = (row.external_slug or "").strip()
        if not slug:
            continue
        out[slug] = MarketResolution(
            market_id=str(payload.get("condition_id") or "") or None,
            event_slug=slug,
            winner_outcome=winner,
            coin_price_start=_safe_float(payload.get("btc_price_start")
                or payload.get("eth_price_start")
                or payload.get("sol_price_start")),
            coin_price_end=_safe_float(payload.get("btc_price_end")
                or payload.get("eth_price_end")
                or payload.get("sol_price_end")),
            final_volume_usdc=_safe_float(payload.get("final_volume")),
            final_liquidity_usdc=_safe_float(payload.get("final_liquidity")),
            resolved_at=str(payload.get("resolved_at") or "") or None,
            source="polybacktest",
        )
    return out


async def _resolve_from_polybacktest_live(
    slugs: set[str],
    *,
    concurrency: int,
) -> dict[str, MarketResolution]:
    """Page through polybacktest /v2/markets to resolve slugs not yet imported.

    Strategy: bulk-fetch markets in 100-row pages and check each page's
    slugs against our needed set.  The wallet's slugs encode an epoch
    timestamp, so we sort our needed set by epoch (newest first) to
    minimize the number of pages we have to scan — polybacktest returns
    markets in start_time DESC order, matching the wallet's natural
    distribution.
    """
    from services.external_data.polybacktest_client import (
        PolybacktestNotConfiguredError,
        build_client_from_settings,
    )

    out: dict[str, MarketResolution] = {}
    try:
        client = await build_client_from_settings()
    except PolybacktestNotConfiguredError:
        return out

    try:
        # Bucket by coin so we page each coin's market list independently.
        # (Polybacktest's list endpoint is per-coin.)
        slugs_by_coin: dict[str, set[str]] = {}
        for s in slugs:
            coin = _coin_from_slug(s)
            if coin:
                slugs_by_coin.setdefault(coin, set()).add(s)

        for coin, want in slugs_by_coin.items():
            offset = 0
            page_size = 100
            scanned = 0
            # Hard cap: scan at most 10K markets per coin so we don't
            # spend forever paging through ancient history when the wallet
            # only touched recent ones.
            while want and scanned < 10_000:
                try:
                    markets, total = await client.list_markets(
                        coin, offset=offset, limit=page_size
                    )
                except Exception as exc:
                    logger.warning(
                        "polybacktest list_markets %s offset=%d failed: %s",
                        coin, offset, exc,
                    )
                    break
                if not markets:
                    break
                for m in markets:
                    if m.slug in want:
                        out[m.slug] = MarketResolution(
                            market_id=str(m.condition_id or "") or None,
                            event_slug=m.slug,
                            winner_outcome=(m.winner.upper() if m.winner else None),
                            coin_price_start=m.coin_price_start,
                            coin_price_end=m.coin_price_end,
                            final_volume_usdc=m.final_volume,
                            final_liquidity_usdc=m.final_liquidity,
                            resolved_at=None,
                            source="polybacktest",
                        )
                        want.discard(m.slug)
                if total and offset + len(markets) >= total:
                    break
                offset += len(markets)
                scanned += len(markets)
    finally:
        await client.close()

    return out


async def _resolve_from_polymarket(
    condition_ids: set[str],
    *,
    concurrency: int,
) -> dict[str, MarketResolution]:
    """Fetch markets one-by-one from Polymarket's gamma-api.

    The gamma endpoint shape: ``GET https://gamma-api.polymarket.com/markets/{conditionId}``.
    Returns ``{conditions[], outcomes[], outcomePrices[], resolvedBy, ...}``.
    The resolved winner can be derived from ``outcomes[i]`` whose
    ``outcomePrices[i] == "1"`` after resolution.
    """
    if not condition_ids:
        return {}

    import httpx

    out: dict[str, MarketResolution] = {}
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(cid: str, client: httpx.AsyncClient) -> None:
        async with sem:
            try:
                r = await client.get(
                    f"https://gamma-api.polymarket.com/markets/{cid}",
                    timeout=15.0,
                )
                if r.status_code != 200:
                    return
                m = r.json()
            except Exception:
                return
            if not isinstance(m, dict):
                return
            # Determine winner from outcomePrices.
            winner_outcome: Optional[str] = None
            outcomes = m.get("outcomes") or []
            prices = m.get("outcomePrices") or []
            try:
                if isinstance(outcomes, str):
                    import json as _json
                    outcomes = _json.loads(outcomes)
                if isinstance(prices, str):
                    import json as _json
                    prices = _json.loads(prices)
            except Exception:
                outcomes, prices = [], []
            if (
                isinstance(outcomes, list)
                and isinstance(prices, list)
                and len(outcomes) == len(prices)
            ):
                for outcome, price in zip(outcomes, prices):
                    try:
                        if float(price) >= 0.999:
                            winner_outcome = str(outcome).upper()
                            break
                    except (TypeError, ValueError):
                        continue
            slug = str(m.get("slug") or m.get("eventSlug") or "")
            out[cid] = MarketResolution(
                market_id=cid,
                event_slug=slug or None,
                winner_outcome=winner_outcome,
                coin_price_start=None,
                coin_price_end=None,
                final_volume_usdc=_safe_float(m.get("volume")),
                final_liquidity_usdc=_safe_float(m.get("liquidity")),
                resolved_at=(m.get("resolvedAt") or m.get("endDate") or None),
                source="polymarket",
            )

    async with httpx.AsyncClient() as client:
        await asyncio.gather(*[_one(cid, client) for cid in condition_ids])

    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        result = float(v)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN
        return None
    return result


def _coin_from_slug(slug: str) -> Optional[str]:
    """Extract coin from a polybacktest-format slug.

    Examples:
      ``btc-updown-5m-1777943100`` → 'btc'
      ``eth-updown-15m-1777920000`` → 'eth'
      ``sol-updown-1h-1777800000``  → 'sol'

    Returns None for non-crypto / unknown formats.
    """
    if not isinstance(slug, str):
        return None
    head = slug.split("-", 1)[0].strip().lower()
    return head if head in {"btc", "eth", "sol"} else None
