"""Golden-source settlement store: read/populate market resolutions.

Persists per-market resolution truth in the ``market_settlements`` table
(keyed by ``condition_id``) and turns it into the engine's
``token_id -> TokenSettlement`` map.

Population is OFFLINE and write-through:
  * import-time capture of the polybacktest winner (known at import), and
  * the resolution resolver
    (``services.strategy_reverse_engineer.market_resolution``) as a
    backfill for anything not captured at import.

Reads happen at SETTLEMENT time only, so nothing here is ever exposed to a
strategy's decision inputs — no look-ahead.  The store records the winning
TOKEN id (not just an outcome label) so the read path is a direct token-id
equality check, immune to the mislabeled "Yes"/"No" outcome strings that
crypto Up/Down markets carry in the catalog.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Optional, Sequence

from services.backtest.settlement import TokenSettlement

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenMarketMeta:
    """What the backtester knows LOCALLY about a traded token's market
    (from the AS-OF catalog / projected-market metadata).  Carries no
    winner — that comes from the store."""

    token_id: str
    condition_id: Optional[str]
    slug: Optional[str] = None
    resolution_time: Optional[datetime] = None


@dataclass(frozen=True)
class MarketResolveHint:
    """Per-market info the resolver-populate path needs to translate a
    winning OUTCOME label into a winning TOKEN id."""

    condition_id: str
    slug: Optional[str] = None
    token_ids: tuple[str, ...] = ()
    up_token: Optional[str] = None
    down_token: Optional[str] = None
    token_outcomes: dict[str, str] = field(default_factory=dict)  # token_id -> label
    resolution_time: Optional[datetime] = None


@dataclass(frozen=True)
class SettlementRecord:
    """In-memory view of a ``market_settlements`` row."""

    condition_id: str
    slug: Optional[str]
    winning_token_id: Optional[str]
    winning_outcome: Optional[str]
    token_ids: tuple[str, ...]
    resolution_time: Optional[datetime]
    coin_price_start: Optional[float]
    coin_price_end: Optional[float]
    resolved: bool
    source: str


# ── Pure logic (no I/O) ────────────────────────────────────────────────────


def build_token_settlements(
    token_meta: Sequence[TokenMarketMeta],
    records: dict[str, SettlementRecord],
) -> dict[str, TokenSettlement]:
    """PURE: combine local token->market metadata with resolved settlement
    records into the engine's ``token_id -> TokenSettlement`` map.

      * Resolved record with a winning token -> 1.0 (winner) / 0.0 (loser).
      * Record present, winner unknown        -> settle_price None (engine
                                                 surfaces is_resolved instead
                                                 of auto-redeeming).
      * No record / no condition_id           -> token omitted (engine marks
                                                 to mid — honest, not settled).

    ``resolution_time`` prefers the store's value and falls back to the
    local metadata's (catalog end_date / projected end_us), so the engine
    can time the sweep / flag resolution even before a winner is sourced.
    """
    out: dict[str, TokenSettlement] = {}
    for meta in token_meta:
        cond = (meta.condition_id or "").strip()
        rec = records.get(cond) if cond else None
        if rec is None:
            continue  # no settlement data -> honest mark-to-mid
        res_time = rec.resolution_time or meta.resolution_time
        if rec.resolved and rec.winning_token_id:
            # Guard against cross-market contamination when the token set
            # is known: never settle a token that isn't in this market.
            if rec.token_ids and meta.token_id not in rec.token_ids:
                continue
            settle_price: Optional[float] = 1.0 if meta.token_id == rec.winning_token_id else 0.0
        else:
            settle_price = None  # known to resolve, winner not sourced
        if settle_price is None and res_time is None:
            continue  # nothing actionable for this token
        out[meta.token_id] = TokenSettlement(
            token_id=meta.token_id,
            settle_price=settle_price,
            resolution_time=res_time,
            winning_outcome=rec.winning_outcome,
            condition_id=cond or None,
            source=rec.source,
        )
    return out


def _winning_token_from_resolution(
    hint: MarketResolveHint, resolution: Any
) -> tuple[Optional[str], Optional[str]]:
    """PURE: translate a resolver ``MarketResolution`` into
    ``(winning_token_id, winning_outcome_label)``.

    Crypto Up/Down maps via the explicit up/down tokens; everything else
    matches the winner label against per-token outcome labels.  Returns
    ``(None, label)`` when the winner can't be mapped to a token.
    """
    label = getattr(resolution, "winner_outcome", None)
    w = (label or "").strip().upper()
    if not w:
        return None, None
    if w == "UP" and hint.up_token:
        return hint.up_token, label
    if w == "DOWN" and hint.down_token:
        return hint.down_token, label
    for tid, tok_label in (hint.token_outcomes or {}).items():
        if (tok_label or "").strip().upper() == w:
            return tid, label
    return None, label


# ── DB I/O ─────────────────────────────────────────────────────────────────


def _row_to_record(row: Any) -> SettlementRecord:
    return SettlementRecord(
        condition_id=str(row.condition_id),
        slug=row.slug,
        winning_token_id=row.winning_token_id,
        winning_outcome=row.winning_outcome,
        token_ids=tuple(str(t) for t in (row.token_ids_json or [])),
        resolution_time=row.resolution_time,
        coin_price_start=row.coin_price_start,
        coin_price_end=row.coin_price_end,
        resolved=bool(row.resolved),
        source=str(row.source or ""),
    )


async def get_settlement_records(
    condition_ids: Iterable[str],
) -> dict[str, SettlementRecord]:
    """Read settlement rows for the given condition_ids (offline)."""
    cond_list = sorted({str(c).strip() for c in condition_ids if c})
    if not cond_list:
        return {}
    from sqlalchemy import select

    from models.database import BacktestAsyncSessionLocal, MarketSettlement

    out: dict[str, SettlementRecord] = {}
    async with BacktestAsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(MarketSettlement).where(
                    MarketSettlement.condition_id.in_(cond_list)
                )
            )
        ).scalars().all()
    for row in rows:
        out[str(row.condition_id)] = _row_to_record(row)
    return out


async def upsert_settlement(record: SettlementRecord) -> None:
    """Insert or update one settlement row (read-then-write, dialect-portable)."""
    from sqlalchemy import select

    from models.database import BacktestAsyncSessionLocal, MarketSettlement

    async with BacktestAsyncSessionLocal() as session:
        existing = (
            await session.execute(
                select(MarketSettlement).where(
                    MarketSettlement.condition_id == record.condition_id
                )
            )
        ).scalar_one_or_none()
        if existing is None:
            existing = MarketSettlement(condition_id=record.condition_id)
            session.add(existing)
        existing.slug = record.slug
        existing.winning_token_id = record.winning_token_id
        existing.winning_outcome = record.winning_outcome
        existing.token_ids_json = list(record.token_ids)
        existing.coin_price_start = record.coin_price_start
        existing.coin_price_end = record.coin_price_end
        existing.resolution_time = record.resolution_time
        existing.resolved = record.resolved
        existing.source = record.source
        await session.commit()


# ── Orchestration ──────────────────────────────────────────────────────────


async def _resolve_and_store(
    hints: Sequence[MarketResolveHint],
) -> dict[str, SettlementRecord]:
    """Backfill missing winners via the resolution resolver, persisting any
    that resolve (write-through).  Best-effort: failures leave a market
    unresolved (retried on a later run) rather than poisoning the result.
    """
    from services.strategy_reverse_engineer.market_resolution import (
        resolve_markets_for_trades,
    )

    trades = [
        {"event_slug": h.slug, "market_id": h.condition_id}
        for h in hints
        if (h.slug or h.condition_id)
    ]
    try:
        resolutions = await resolve_markets_for_trades(trades)
    except Exception:
        logger.warning("settlement: resolver backfill failed", exc_info=True)
        resolutions = {}

    out: dict[str, SettlementRecord] = {}
    for h in hints:
        mr = resolutions.get(h.slug or "") or resolutions.get(h.condition_id or "")
        win_tok: Optional[str] = None
        win_label: Optional[str] = None
        coin_start = coin_end = None
        source = ""
        if mr is not None:
            win_tok, win_label = _winning_token_from_resolution(h, mr)
            coin_start = getattr(mr, "coin_price_start", None)
            coin_end = getattr(mr, "coin_price_end", None)
            source = getattr(mr, "source", None) or "resolver"
        rec = SettlementRecord(
            condition_id=h.condition_id,
            slug=h.slug,
            winning_token_id=win_tok,
            winning_outcome=win_label,
            token_ids=tuple(h.token_ids),
            resolution_time=h.resolution_time,
            coin_price_start=coin_start,
            coin_price_end=coin_end,
            resolved=win_tok is not None,
            source=source,
        )
        if rec.resolved:
            try:
                await upsert_settlement(rec)
            except Exception:
                logger.warning(
                    "settlement: upsert failed for %s", h.condition_id, exc_info=True
                )
        out[h.condition_id] = rec
    return out


async def populate_settlements(
    hints: Sequence[MarketResolveHint],
) -> dict[str, SettlementRecord]:
    """Resolve winners for the given markets and write the resolved ones
    through to the store.  Public entry point for the offline backfill
    (``scripts/backfill_market_settlements.py``)."""
    if not hints:
        return {}
    return await _resolve_and_store(hints)


async def load_token_settlements(
    token_meta: Sequence[TokenMarketMeta],
    *,
    hints: Optional[Sequence[MarketResolveHint]] = None,
    allow_network: bool = True,
) -> dict[str, TokenSettlement]:
    """Main entry for the backtester: build the ``token_id ->
    TokenSettlement`` map from the offline store, backfilling missing
    winners via the resolver (write-through) when ``allow_network``.
    """
    cond_ids = {m.condition_id for m in token_meta if m.condition_id}
    records = await get_settlement_records(cond_ids)
    if allow_network and hints:
        unresolved = [
            h
            for h in hints
            if h.condition_id in cond_ids
            and (h.condition_id not in records or not records[h.condition_id].resolved)
        ]
        if unresolved:
            records.update(await _resolve_and_store(unresolved))
    return build_token_settlements(token_meta, records)


__all__ = [
    "TokenMarketMeta",
    "MarketResolveHint",
    "SettlementRecord",
    "build_token_settlements",
    "get_settlement_records",
    "upsert_settlement",
    "populate_settlements",
    "load_token_settlements",
]
