"""Manual REST backfill into MarketMicrostructureSnapshot.

Closes the historical-coverage gap that the proactive WS subscription
service can't help with: even a perfectly-subscribed recorder only
has data going back to when it started.  For tokens whose strategy
fires before the WS recorder caught them, we backfill from
Polymarket's REST ``/prices-history`` endpoint.

Important caveats — please surface in the UI:

  * Polymarket REST does NOT serve full L2 book history.  The richest
    historical surface is mid-price time series at fidelity ranging
    from 1 minute to 1 hour.  We synthesize a single-level book by
    centering best_bid / best_ask on the mid with a small epsilon
    spread; sizes are zero (we have no depth information).
  * Each synthesized row carries ``payload_json["synthetic"] = True``
    and ``payload_json["source"] = "rest_backfill"`` so the
    backtester / Cox PH trainer can choose to filter or downweight
    them.
  * Trade-tape backfill is NOT included — Polymarket's trade history
    REST surface is wallet-scoped, not market-scoped, so we can't
    reconstruct the global print tape.  Live trade prints (recorded
    via WebSocket) remain the only authoritative source.

Scope options (the ``BackfillScope`` enum):

  * ``token`` — operator-supplied list of clob_token_ids
  * ``strategy`` — every distinct token referenced by an
    OpportunityHistory row for ``strategy_slug`` in the time window
  * ``session`` — every target_token_id of a recording session
  * ``catalog_top_liquid`` — top N most-liquid markets (matches the
    proactive subscription cap so backfill mirrors live coverage)

The service runs synchronously and returns a per-token result
report so the operator can see exactly what was filled vs what
failed (rate-limited / no data / market closed / etc.).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from sqlalchemy import and_, func, select

from models.database import (
    AsyncSessionLocal,
    MarketMicrostructureSnapshot,
    OpportunityHistory,
    RecordingSession,
)

logger = logging.getLogger("recorder_backfill_service")


BackfillScope = Literal["token", "strategy", "session", "catalog_top_liquid"]


# Mid → (bid, ask) reconstruction: half this in either direction.
# Default 50 bps total spread is a reasonable mid-tier prediction
# market spread — operators can override per-call.
_DEFAULT_SYNTHETIC_SPREAD_BPS = 50.0


@dataclass
class BackfillTokenResult:
    token_id: str
    rows_inserted: int = 0
    skipped_existing: int = 0
    points_fetched: int = 0
    error: str | None = None


@dataclass
class BackfillResult:
    job_id: str
    scope: BackfillScope
    started_at: str
    completed_at: str
    duration_seconds: float
    target_token_count: int
    tokens_with_data: int = 0
    tokens_with_errors: int = 0
    rows_inserted_total: int = 0
    points_fetched_total: int = 0
    skipped_existing_total: int = 0
    interval: str = "1h"
    fidelity_minutes: int | None = None
    start: str | None = None
    end: str | None = None
    synthetic_spread_bps: float = _DEFAULT_SYNTHETIC_SPREAD_BPS
    per_token: list[BackfillTokenResult] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "scope": self.scope,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "target_token_count": self.target_token_count,
            "tokens_with_data": self.tokens_with_data,
            "tokens_with_errors": self.tokens_with_errors,
            "rows_inserted_total": self.rows_inserted_total,
            "points_fetched_total": self.points_fetched_total,
            "skipped_existing_total": self.skipped_existing_total,
            "interval": self.interval,
            "fidelity_minutes": self.fidelity_minutes,
            "start": self.start,
            "end": self.end,
            "synthetic_spread_bps": self.synthetic_spread_bps,
            "per_token": [t.__dict__ for t in self.per_token],
            "error": self.error,
        }


# ── Token resolution per scope ─────────────────────────────────────────


async def _resolve_strategy_tokens(
    *, strategy_slug: str, start_dt: datetime, end_dt: datetime,
) -> list[str]:
    """Distinct token_ids referenced by ``strategy_slug``'s opportunity
    history rows in the requested window.
    """
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(
            select(OpportunityHistory.positions_data).where(
                OpportunityHistory.strategy_type == strategy_slug,
                OpportunityHistory.detected_at >= start_dt,
                OpportunityHistory.detected_at <= end_dt,
            )
        )).all()
    out: set[str] = set()
    for (pd,) in rows:
        if not isinstance(pd, dict):
            continue
        for p in pd.get("positions_to_take") or []:
            if isinstance(p, dict):
                tok = str(p.get("token_id") or "").strip()
                if tok:
                    out.add(tok)
    return sorted(out)


async def _resolve_session_tokens(*, session_id: str) -> list[str]:
    async with AsyncSessionLocal() as session:
        row = (await session.execute(
            select(RecordingSession).where(RecordingSession.id == session_id)
        )).scalar_one_or_none()
    if row is None:
        raise ValueError(f"Recording session '{session_id}' not found")
    return list(row.target_token_ids_json or [])


def _resolve_catalog_top_liquid(*, max_tokens: int, min_liquidity_usd: float) -> list[str]:
    """Pull the N most-liquid catalog tokens.  Mirrors the proactive
    subscription policy so backfill scope matches live coverage."""
    from services.shared_state import _read_market_catalog_file

    catalog = _read_market_catalog_file()
    if catalog is None:
        return []
    _events, markets, _meta = catalog
    by_token: dict[str, float] = {}
    for m in markets:
        if not isinstance(m, dict):
            continue
        if m.get("closed") or m.get("archived") or m.get("resolved"):
            continue
        if m.get("active") is False or m.get("accepting_orders") is False:
            continue
        try:
            liq = float(m.get("liquidity") or 0.0)
        except (TypeError, ValueError):
            liq = 0.0
        if liq < min_liquidity_usd:
            continue
        raw = m.get("clob_token_ids") or []
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                raw = []
        for t in raw or []:
            ts = str(t or "").strip()
            if not ts:
                continue
            if ts not in by_token or liq > by_token[ts]:
                by_token[ts] = liq
    ranked = sorted(by_token.items(), key=lambda kv: kv[1], reverse=True)
    return [t for t, _ in ranked[:max_tokens]]


async def _resolve_tokens(
    *,
    scope: BackfillScope,
    target_values: list[str] | None,
    strategy_slug: str | None,
    session_id: str | None,
    start_dt: datetime,
    end_dt: datetime,
    catalog_max_tokens: int,
    catalog_min_liquidity_usd: float,
) -> list[str]:
    if scope == "token":
        return [str(t).strip() for t in (target_values or []) if str(t).strip()]
    if scope == "strategy":
        if not strategy_slug:
            raise ValueError("strategy scope requires strategy_slug")
        return await _resolve_strategy_tokens(
            strategy_slug=strategy_slug, start_dt=start_dt, end_dt=end_dt,
        )
    if scope == "session":
        if not session_id:
            raise ValueError("session scope requires session_id")
        return await _resolve_session_tokens(session_id=session_id)
    if scope == "catalog_top_liquid":
        return _resolve_catalog_top_liquid(
            max_tokens=catalog_max_tokens,
            min_liquidity_usd=catalog_min_liquidity_usd,
        )
    raise ValueError(f"Unknown scope: {scope}")


# ── REST fetch + synth ─────────────────────────────────────────────────


async def _existing_observed_set(
    *, token_id: str, start_dt: datetime, end_dt: datetime,
) -> set[int]:
    """Return the set of integer-second observed_at timestamps that
    already have a book snapshot for this token.  Used to dedupe so
    we don't insert synthetic rows on top of real WS-recorded ones.
    """
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(
            select(MarketMicrostructureSnapshot.observed_at).where(
                MarketMicrostructureSnapshot.token_id == token_id,
                MarketMicrostructureSnapshot.observed_at >= start_dt,
                MarketMicrostructureSnapshot.observed_at <= end_dt,
                MarketMicrostructureSnapshot.snapshot_type == "book",
            )
        )).all()
    return {int(r[0].timestamp()) for r in rows if r[0] is not None}


async def _backfill_one_token(
    *,
    token_id: str,
    start_dt: datetime,
    end_dt: datetime,
    interval: str,
    fidelity_minutes: int | None,
    synthetic_spread_bps: float,
) -> BackfillTokenResult:
    """Fetch prices-history for one token and write synthetic
    book-snapshot rows.  De-dupes against existing rows at the same
    second so re-running the backfill is idempotent and never
    overwrites real WS captures.
    """
    res = BackfillTokenResult(token_id=token_id)
    try:
        from services.polymarket import polymarket_client

        client = polymarket_client
        kwargs: dict[str, Any] = {
            "token_id": token_id,
            "start_ts": int(start_dt.timestamp()),
            "end_ts": int(end_dt.timestamp()),
        }
        if interval:
            kwargs["interval"] = interval
        if fidelity_minutes is not None:
            kwargs["fidelity"] = int(fidelity_minutes)
        history = await client.get_prices_history(**kwargs)
        res.points_fetched = len(history)
        if not history:
            return res

        # Deduplicate against existing rows.
        existing_secs = await _existing_observed_set(
            token_id=token_id, start_dt=start_dt, end_dt=end_dt,
        )

        rows: list[MarketMicrostructureSnapshot] = []
        half_spread = (synthetic_spread_bps / 2.0) / 10_000.0
        for pt in history:
            ts_ms = float(pt.get("t") or 0.0)
            mid = float(pt.get("p") or 0.0)
            if ts_ms <= 0 or mid <= 0:
                continue
            ts_sec = int(ts_ms // 1000)
            if ts_sec in existing_secs:
                res.skipped_existing += 1
                continue
            existing_secs.add(ts_sec)  # so points within the same fidelity bucket dedupe
            observed_at = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
            # Single-level synthetic book.  Sizes are 0 because we
            # have no depth info — backtest matchers + Cox PH trainer
            # should treat synthetic rows as "no L2" via the
            # payload_json["synthetic"] flag.
            best_bid = max(0.001, mid * (1.0 - half_spread))
            best_ask = min(0.999, mid * (1.0 + half_spread))
            if best_bid >= best_ask:  # rare floating-point edge
                best_bid = max(0.001, mid - 0.001)
                best_ask = min(0.999, mid + 0.001)
            spread_bps = ((best_ask - best_bid) / mid) * 10_000.0 if mid > 0 else None
            rows.append(MarketMicrostructureSnapshot(
                id=uuid.uuid4().hex,
                provider="polymarket",
                token_id=token_id,
                snapshot_type="book",
                observed_at=observed_at,
                exchange_ts_ms=int(ts_ms),
                sequence=None,
                best_bid=float(best_bid),
                best_ask=float(best_ask),
                spread_bps=spread_bps,
                bids_json=[{"price": float(best_bid), "size": 0.0}],
                asks_json=[{"price": float(best_ask), "size": 0.0}],
                payload_json={
                    "synthetic": True,
                    "source": "rest_backfill",
                    "interval": interval,
                    "fidelity_minutes": fidelity_minutes,
                    "synthetic_spread_bps": synthetic_spread_bps,
                },
                created_at=datetime.now(timezone.utc),
            ))
        if rows:
            async with AsyncSessionLocal() as session:
                session.add_all(rows)
                await session.commit()
            res.rows_inserted = len(rows)
    except Exception as exc:
        res.error = f"{type(exc).__name__}: {exc}"
        logger.warning("Backfill for %s failed: %s", token_id, exc)
    return res


# ── Public entrypoint ──────────────────────────────────────────────────


async def run_backfill(
    *,
    scope: BackfillScope,
    target_values: list[str] | None = None,
    strategy_slug: str | None = None,
    session_id: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    interval: str = "1h",
    fidelity_minutes: int | None = None,
    synthetic_spread_bps: float = _DEFAULT_SYNTHETIC_SPREAD_BPS,
    catalog_max_tokens: int = 2000,
    catalog_min_liquidity_usd: float = 100.0,
    concurrency: int = 5,
    max_tokens: int = 5000,
) -> BackfillResult:
    """Run a one-shot REST backfill for the given scope.

    Synchronous from the caller's perspective — returns a full report
    when done.  Cap ``max_tokens`` (5000 default) protects against a
    pathological scope blowing up Polymarket's rate limit.
    """
    started = time.monotonic()
    job_id = uuid.uuid4().hex[:16]
    started_at = datetime.now(timezone.utc).isoformat()
    end_dt = end or datetime.now(timezone.utc)
    start_dt = start or (end_dt - timedelta(days=7))
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)

    result = BackfillResult(
        job_id=job_id,
        scope=scope,
        started_at=started_at,
        completed_at=started_at,
        duration_seconds=0.0,
        target_token_count=0,
        interval=interval,
        fidelity_minutes=fidelity_minutes,
        start=start_dt.isoformat(),
        end=end_dt.isoformat(),
        synthetic_spread_bps=synthetic_spread_bps,
    )

    try:
        tokens = await _resolve_tokens(
            scope=scope,
            target_values=target_values,
            strategy_slug=strategy_slug,
            session_id=session_id,
            start_dt=start_dt,
            end_dt=end_dt,
            catalog_max_tokens=catalog_max_tokens,
            catalog_min_liquidity_usd=catalog_min_liquidity_usd,
        )
    except Exception as exc:
        result.error = f"resolve failed: {exc}"
        result.completed_at = datetime.now(timezone.utc).isoformat()
        result.duration_seconds = time.monotonic() - started
        return result

    if len(tokens) > max_tokens:
        result.error = (
            f"scope yielded {len(tokens)} tokens; capped at {max_tokens}. "
            "Tighten the scope or raise max_tokens."
        )
        tokens = tokens[:max_tokens]
    result.target_token_count = len(tokens)
    if not tokens:
        result.completed_at = datetime.now(timezone.utc).isoformat()
        result.duration_seconds = time.monotonic() - started
        return result

    semaphore = asyncio.Semaphore(max(1, int(concurrency)))

    async def _one(tok: str) -> BackfillTokenResult:
        async with semaphore:
            return await _backfill_one_token(
                token_id=tok,
                start_dt=start_dt,
                end_dt=end_dt,
                interval=interval,
                fidelity_minutes=fidelity_minutes,
                synthetic_spread_bps=synthetic_spread_bps,
            )

    per_token = await asyncio.gather(*[_one(t) for t in tokens])
    result.per_token = list(per_token)
    result.tokens_with_data = sum(1 for r in per_token if r.rows_inserted > 0)
    result.tokens_with_errors = sum(1 for r in per_token if r.error)
    result.rows_inserted_total = sum(r.rows_inserted for r in per_token)
    result.points_fetched_total = sum(r.points_fetched for r in per_token)
    result.skipped_existing_total = sum(r.skipped_existing for r in per_token)
    result.completed_at = datetime.now(timezone.utc).isoformat()
    result.duration_seconds = time.monotonic() - started

    logger.info(
        "Backfill %s done — scope=%s tokens=%d rows_inserted=%d errors=%d duration=%.1fs",
        job_id, scope, result.target_token_count,
        result.rows_inserted_total, result.tokens_with_errors,
        result.duration_seconds,
    )
    return result
