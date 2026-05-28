"""Bus-backed book replay — turns recorded-event-bus ``crypto.update.dispatch``
parquet into a backtest ``BookSource`` (same protocol as ``BookReplay`` /
``ParquetBookReplay``).

Why this exists
===============
We are moving market-data recording OFF Postgres and onto the unified
recorded-event-bus parquet plane (bounded + UI-managed via Data Lab →
Topics).  The fill engine consumes a stream of :class:`BookSnapshot`
objects; this adapter materialises them from the bus's crypto dispatch
envelopes so crypto backtests read the SAME archived data the live
strategies saw — no SQL ``market_microstructure_snapshots`` dependency.

Each ``crypto.update.dispatch`` envelope payload carries ``markets: [...]``
with, per market: ``clob_token_ids`` (``[up_token, down_token]`` ordered by
``up_token_index``/``down_token_index``), ``best_bid``/``best_ask`` (top of
book for the UP outcome), and ``up_price``/``down_price``.  Polymarket binary
identity ``P(down) = 1 - P(up)`` lets us derive the DOWN token's book from
the UP book: ``down_bid = 1 - up_ask``, ``down_ask = 1 - up_bid``.

This is **top-of-book** fidelity (one level/side) — appropriate for the
thin crypto up/down markets and matches what the dispatch archives.  Full
L2 depth is not in this stream.
"""
from __future__ import annotations

import bisect
import glob
import json
import logging
import os
from datetime import datetime, timezone
from typing import AsyncIterator, Iterable, Optional

import pyarrow.parquet as pq

from services.backtest.book_replay import BookSnapshot, PriceLevel
from services.external_data.parquet_schema import parquet_roots

logger = logging.getLogger(__name__)

_TOPIC = "crypto.update.dispatch"


def _clamp_price(p: float) -> Optional[float]:
    if p is None:
        return None
    p = float(p)
    if p <= 0.0 or p >= 1.0:
        return None
    return p


def _book_for_outcome(up_bid: Optional[float], up_ask: Optional[float], *, is_up: bool) -> tuple[tuple[PriceLevel, ...], tuple[PriceLevel, ...]]:
    """Return (bids, asks) for the up or down token from the up-side BBO."""
    if is_up:
        bid, ask = up_bid, up_ask
    else:
        # Binary complement: down = 1 - up (bid/ask swap).
        bid = (1.0 - up_ask) if up_ask is not None else None
        ask = (1.0 - up_bid) if up_bid is not None else None
    bids = (PriceLevel(price=bid, size=0.0),) if bid is not None else ()
    asks = (PriceLevel(price=ask, size=0.0),) if ask is not None else ()
    return bids, asks


class BusBookReplay:
    """``_BookSource`` reading crypto book state from bus dispatch parquet.

    Streams BookSnapshots in global chronological order across all tokens
    in the requested window; ``snapshot_at`` does an as-of (≤ ts) lookup.
    """

    def __init__(self, *, token_ids: Iterable[str], start: datetime, end: datetime, roots: Optional[list] = None):
        self._want = {str(t) for t in token_ids if t}
        self._start_us = _to_us(start)
        self._end_us = _to_us(end)
        self.truncated = False
        # per token: parallel sorted arrays of (obs_us) and BookSnapshot
        self._obs: dict[str, list[int]] = {}
        self._snaps: dict[str, list[BookSnapshot]] = {}
        self._roots = roots
        self._load()

    def _topic_files(self) -> list[str]:
        out: list[str] = []
        for root in (self._roots or parquet_roots()):
            base = os.path.join(str(root), "recorded_event_bus", _TOPIC)
            out.extend(glob.glob(os.path.join(base, "*", "*", "*.parquet")))
        return out

    def _load(self) -> None:
        rows: dict[str, list[tuple[int, BookSnapshot]]] = {}
        for f in self._topic_files():
            try:
                # Cheap row-group prune: skip files entirely outside the window
                # by min/max of observed_at_us if available; else read columns.
                d = pq.read_table(f, columns=["observed_at_us", "payload_json"]).to_pydict()
            except Exception:
                logger.warning("BusBookReplay: unreadable %s", f)
                self.truncated = True
                continue
            for obs_us, payload in zip(d["observed_at_us"], d["payload_json"]):
                if obs_us is None or obs_us < self._start_us or obs_us > self._end_us:
                    continue
                try:
                    p = json.loads(payload) if isinstance(payload, str) else payload
                except Exception:
                    continue
                for m in (p.get("markets") or []):
                    if not isinstance(m, dict):
                        continue
                    toks = m.get("clob_token_ids") or []
                    if not isinstance(toks, list) or len(toks) < 2:
                        continue
                    up_idx = int(m.get("up_token_index") or 0)
                    dn_idx = int(m.get("down_token_index") or (1 - up_idx))
                    up_tok = str(toks[up_idx]) if up_idx < len(toks) else None
                    dn_tok = str(toks[dn_idx]) if dn_idx < len(toks) else None
                    up_bid = _clamp_price(m.get("best_bid"))
                    up_ask = _clamp_price(m.get("best_ask"))
                    # Fall back to up_price as a tight synthetic BBO when the
                    # book side is missing (some dispatch rows carry mid only).
                    if up_bid is None and up_ask is None:
                        mid = _clamp_price(m.get("up_price"))
                        if mid is not None:
                            up_bid = max(0.001, mid - 0.005)
                            up_ask = min(0.999, mid + 0.005)
                    for tok, is_up in ((up_tok, True), (dn_tok, False)):
                        if not tok or tok not in self._want:
                            continue
                        bids, asks = _book_for_outcome(up_bid, up_ask, is_up=is_up)
                        if not bids and not asks:
                            continue
                        snap = BookSnapshot(
                            token_id=tok,
                            observed_at=datetime.fromtimestamp(int(obs_us) / 1e6, tz=timezone.utc),
                            bids=bids, asks=asks,
                        )
                        rows.setdefault(tok, []).append((int(obs_us), snap))
        for tok, lst in rows.items():
            lst.sort(key=lambda r: r[0])
            self._obs[tok] = [r[0] for r in lst]
            self._snaps[tok] = [r[1] for r in lst]

    @property
    def loaded_tokens(self) -> int:
        return len(self._snaps)

    def total_snapshots(self) -> int:
        return sum(len(v) for v in self._snaps.values())

    async def iter_snapshots(self) -> AsyncIterator[BookSnapshot]:
        # Global chronological merge across tokens.
        merged: list[tuple[int, BookSnapshot]] = []
        for tok, obs in self._obs.items():
            snaps = self._snaps[tok]
            merged.extend(zip(obs, snaps))
        merged.sort(key=lambda r: r[0])
        for _, snap in merged:
            yield snap

    async def snapshot_at(self, *, token_id: str, ts: datetime) -> Optional[BookSnapshot]:
        obs = self._obs.get(str(token_id))
        if not obs:
            return None
        i = bisect.bisect_right(obs, _to_us(ts)) - 1
        if i < 0:
            return None
        return self._snaps[str(token_id)][i]


def _to_us(ts: datetime) -> int:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return int(ts.astimezone(timezone.utc).timestamp() * 1_000_000)
