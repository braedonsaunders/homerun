"""Resolve outcome-token siblings for a given Polymarket token_id.

Polymarket markets come in two structural shapes that matter here:

  * **Binary markets**: a single ``Market`` with two ``clob_token_ids``
    (Yes / No, Up / Down, etc).  Token outcomes typically labeled "Yes"
    / "No" but NOT always — sports markets use team names, crypto
    markets use Up/Down, etc.  The resolver does not assume label.

  * **Multi-outcome markets**: a single ``Market`` with N
    ``clob_token_ids`` (3-way election, n-way bracket, etc).  Each
    token's price is in [0, 1] and the n outcomes sum to ~1.0.

  * **Negrisk grouped markets**: an ``Event`` with multiple sibling
    binary ``Market``s, each with its own pair of outcome tokens, but
    structurally tied (one and only one of the parent events resolves
    YES across the group).  We DO NOT net across the negrisk group
    here — that's a separate enforcement layer — but we DO mark each
    member market's tokens as siblings within their own market.

The resolver builds a token_id → market_id index and a market_id →
sibling_token_ids index from ``MarketCatalog``.  Lazy-loaded on first
call, refreshed every ``_REFRESH_SECONDS`` to pick up new markets.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from sqlalchemy import select

from models.database import BacktestAsyncSessionLocal, MarketCatalog
from utils.logger import get_logger


logger = get_logger("outcome_resolver")


_REFRESH_SECONDS = 300.0  # rebuild the index every 5 min


@dataclass
class MarketRecord:
    """Minimal market metadata for outcome-aware netting."""

    market_id: str
    condition_id: str
    question: str = ""
    slug: str = ""
    event_slug: str = ""
    neg_risk: bool = False
    token_ids: tuple[str, ...] = ()
    outcomes: tuple[str, ...] = ()  # outcome label per token, parallel to token_ids
    outcome_count: int = 0  # len(token_ids), 2 = binary, 3+ = multi-outcome


@dataclass
class _ResolverIndex:
    by_token: dict[str, MarketRecord] = field(default_factory=dict)
    by_market: dict[str, MarketRecord] = field(default_factory=dict)
    built_at: float = 0.0


class OutcomeResolver:
    """Singleton service. ``await resolver.siblings(token_id)`` returns
    the tuple of sibling token ids for the same market (excluding the
    queried token).  Returns an empty tuple if the token isn't in any
    cached market — callers should treat that as "no netting available"
    rather than fail.
    """

    def __init__(self) -> None:
        self._index = _ResolverIndex()
        self._lock = asyncio.Lock()

    async def _ensure_index(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and self._index.built_at and (now - self._index.built_at) < _REFRESH_SECONDS:
            return
        async with self._lock:
            # Re-check inside the lock — another coroutine may have just
            # rebuilt while we were waiting.
            if not force and self._index.built_at and (now - self._index.built_at) < _REFRESH_SECONDS:
                return
            try:
                await self._rebuild_index()
            except Exception as exc:
                logger.warning("OutcomeResolver index rebuild failed", exc_info=exc)

    async def _rebuild_index(self) -> None:
        by_token: dict[str, MarketRecord] = {}
        by_market: dict[str, MarketRecord] = {}
        async with BacktestAsyncSessionLocal() as session:
            stmt = select(MarketCatalog.markets_json).order_by(MarketCatalog.updated_at.desc()).limit(1)
            result = await session.execute(stmt)
            row = result.first()
            if row is None or not row[0]:
                self._index = _ResolverIndex(built_at=time.time())
                return
            markets_blob: Any = row[0]
            if isinstance(markets_blob, str):
                try:
                    markets_blob = json.loads(markets_blob)
                except json.JSONDecodeError:
                    self._index = _ResolverIndex(built_at=time.time())
                    return
            if not isinstance(markets_blob, list):
                self._index = _ResolverIndex(built_at=time.time())
                return
            for m in markets_blob:
                if not isinstance(m, dict):
                    continue
                token_ids_raw = m.get("clob_token_ids") or []
                token_ids = tuple(str(t).strip() for t in token_ids_raw if t)
                if len(token_ids) < 2:
                    continue
                outcomes_raw = []
                for tok in m.get("tokens") or []:
                    if isinstance(tok, dict):
                        outcomes_raw.append(str(tok.get("outcome") or ""))
                outcomes = tuple(outcomes_raw[: len(token_ids)])
                record = MarketRecord(
                    market_id=str(m.get("id") or m.get("condition_id") or ""),
                    condition_id=str(m.get("condition_id") or ""),
                    question=str(m.get("question") or ""),
                    slug=str(m.get("slug") or ""),
                    event_slug=str(m.get("event_slug") or ""),
                    neg_risk=bool(m.get("neg_risk") or False),
                    token_ids=token_ids,
                    outcomes=outcomes,
                    outcome_count=len(token_ids),
                )
                if not record.market_id:
                    continue
                by_market[record.market_id] = record
                for tid in token_ids:
                    by_token[tid.lower()] = record
        self._index = _ResolverIndex(by_token=by_token, by_market=by_market, built_at=time.time())
        logger.info(
            "OutcomeResolver rebuilt",
            tokens_indexed=len(by_token),
            markets_indexed=len(by_market),
        )

    async def market_for_token(self, token_id: str) -> Optional[MarketRecord]:
        await self._ensure_index()
        return self._index.by_token.get(str(token_id or "").strip().lower())

    async def siblings(self, token_id: str) -> tuple[str, ...]:
        record = await self.market_for_token(token_id)
        if record is None:
            return ()
        target = str(token_id or "").strip().lower()
        return tuple(t for t in record.token_ids if t.lower() != target)

    async def market_record(self, market_id: str) -> Optional[MarketRecord]:
        await self._ensure_index()
        return self._index.by_market.get(str(market_id or "").strip())

    async def index_stats(self) -> dict[str, Any]:
        await self._ensure_index()
        binary = 0
        multi = 0
        for r in self._index.by_market.values():
            if r.outcome_count == 2:
                binary += 1
            elif r.outcome_count > 2:
                multi += 1
        return {
            "tokens_indexed": len(self._index.by_token),
            "markets_indexed": len(self._index.by_market),
            "binary_markets": binary,
            "multi_outcome_markets": multi,
            "built_at_epoch": self._index.built_at,
        }


outcome_resolver = OutcomeResolver()


def get_outcome_resolver() -> OutcomeResolver:
    return outcome_resolver
