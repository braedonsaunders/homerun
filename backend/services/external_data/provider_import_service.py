"""Provider data import service.

Coordinates the lifecycle of a ``ProviderImportJob``:

  1. Read the job row, mark it ``running``.
  2. Resolve provider-specific config (API key, base URL).
  3. Iterate over the requested markets / time windows, paginate the
     provider's API, transform each page into ``MarketMicrostructureSnapshot``
     rows, batch-insert with a unique-by-natural-key skip-if-exists policy.
  4. Upsert a ``ProviderDataset`` catalog entry per (provider, market_id).
  5. Mark the job ``completed`` (or ``failed`` with the error and the
     partial counters preserved).

Idempotent: rerunning the same job over the same window simply skips
rows that already exist (natural key is ``(provider, token_id, observed_at,
sequence|null)``).  Snapshots have a stable ID derived from that natural
key so SQL-level ``ON CONFLICT DO NOTHING`` is the cleanest behavior on
Postgres; on SQLite (test) the inserter pre-checks the key.

Currently implements only ``polybacktest``.  Adding a new provider is a
matter of writing a sibling ``_run_<provider>_import`` function and
wiring it into ``_DISPATCH``.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from sqlalchemy import and_, delete as sa_delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from models.database import (
    AsyncSessionLocal,
    MarketMicrostructureSnapshot,
    ProviderDataset,
    ProviderImportJob,
)
from services.external_data.polybacktest_client import (
    PolybacktestAuthError,
    PolybacktestError,
    PolybacktestNotConfiguredError,
    PolybacktestSnapshot,
    build_client_from_settings,
    supported_coins,
)
from utils.utcnow import utcnow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------


PROVIDER_POLYBACKTEST = "polybacktest"

# Page sizes — tuned for the typical Pro-tier rate limit.  Polybacktest
# v2 caps snapshots at 1000 per page; we use the max to minimize API
# calls per import.  Trades are not exposed for prediction markets
# (they live only on the spot/futures reference endpoints, which we
# don't currently import).
_SNAPSHOT_PAGE_LIMIT = 1000
_INSERT_BATCH_SIZE = 250


# ---------------------------------------------------------------------------
# Job creation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CreatePolybacktestJobSpec:
    """Operator-supplied import request."""

    coin: str
    market_ids: list[str]
    start_ms: int
    end_ms: int
    # Always pulls the full L2 order book — polybacktest's
    # ``include_orderbook=true`` flag is set unconditionally so we
    # capture every level of depth (15 per side) on every snapshot.
    # The ``include_trades`` knob is gone: polybacktest does not expose
    # a trades endpoint for prediction markets (they only have it for
    # the spot/futures crypto reference feeds).


async def enqueue_polybacktest_import(spec: CreatePolybacktestJobSpec) -> ProviderImportJob:
    """Validate the request and persist a queued job row.

    Returns the persisted row so the caller (the API route) can echo
    back its ID.
    """
    coin = (spec.coin or "").strip().lower()
    if coin not in supported_coins():
        raise ValueError(
            f"Unsupported coin '{spec.coin}'.  Polybacktest exposes: {supported_coins()}"
        )
    if not spec.market_ids:
        raise ValueError("At least one market_id is required")
    if int(spec.end_ms) <= int(spec.start_ms):
        raise ValueError("end_ms must be greater than start_ms")
    job_id = f"prov-{int(time.time() * 1000)}-{_short_hash(spec.market_ids)}"
    payload = {
        "provider": PROVIDER_POLYBACKTEST,
        "coin": coin,
        "market_ids": [str(m) for m in spec.market_ids],
        "start_ms": int(spec.start_ms),
        "end_ms": int(spec.end_ms),
    }
    async with AsyncSessionLocal() as session:
        row = ProviderImportJob(
            id=job_id,
            provider=PROVIDER_POLYBACKTEST,
            status="queued",
            progress=0.0,
            payload_json=payload,
            message="Waiting for import worker",
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return row


# ---------------------------------------------------------------------------
# Job execution — picked up by the worker
# ---------------------------------------------------------------------------


async def claim_next_queued_job() -> Optional[ProviderImportJob]:
    """Atomically claim the oldest queued job (FIFO).

    Uses a single UPDATE ... WHERE id = (SELECT ... LIMIT 1 FOR UPDATE
    SKIP LOCKED) on Postgres so multiple worker processes can poll
    safely.  On SQLite (tests) we fall back to a simple SELECT + UPDATE.
    """
    async with AsyncSessionLocal() as session:
        # Try the Postgres-only path first.
        try:
            stmt = (
                update(ProviderImportJob)
                .where(
                    ProviderImportJob.id.in_(
                        select(ProviderImportJob.id)
                        .where(ProviderImportJob.status == "queued")
                        .order_by(ProviderImportJob.created_at.asc())
                        .limit(1)
                        .with_for_update(skip_locked=True)
                    )
                )
                .values(
                    status="running",
                    started_at=utcnow(),
                    message="Worker claimed job",
                )
                .returning(ProviderImportJob)
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            await session.commit()
            return row
        except Exception:
            await session.rollback()

        # Fallback (SQLite tests).
        candidate = (
            await session.execute(
                select(ProviderImportJob)
                .where(ProviderImportJob.status == "queued")
                .order_by(ProviderImportJob.created_at.asc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if candidate is None:
            return None
        candidate.status = "running"
        candidate.started_at = utcnow()
        candidate.message = "Worker claimed job"
        await session.commit()
        await session.refresh(candidate)
        return candidate


async def run_job(job_id: str) -> dict[str, Any]:
    """Execute a single import job by ID.

    Returns a summary dict with totals; also persists the result on the
    job row.  Raises only on programming errors — provider/network
    failures are caught and the job is marked ``failed`` with the
    error preserved.
    """
    async with AsyncSessionLocal() as session:
        job = (
            await session.execute(select(ProviderImportJob).where(ProviderImportJob.id == job_id))
        ).scalar_one_or_none()
        if job is None:
            raise ValueError(f"Provider import job '{job_id}' not found")
        provider = job.provider
        payload = dict(job.payload_json or {})

    handler = _DISPATCH.get(provider)
    if handler is None:
        await _mark_failed(job_id, f"No handler registered for provider '{provider}'")
        raise ValueError(f"Unknown provider '{provider}'")

    try:
        summary = await handler(job_id, payload)
    except PolybacktestNotConfiguredError as exc:
        await _mark_failed(job_id, str(exc))
        return {"job_id": job_id, "status": "failed", "error": str(exc)}
    except PolybacktestAuthError as exc:
        await _mark_failed(job_id, str(exc))
        return {"job_id": job_id, "status": "failed", "error": str(exc)}
    except PolybacktestError as exc:
        await _mark_failed(job_id, f"polybacktest error: {exc}")
        return {"job_id": job_id, "status": "failed", "error": str(exc)}
    except Exception as exc:
        logger.exception("provider import job %s crashed", job_id)
        await _mark_failed(job_id, f"unexpected: {exc}")
        return {"job_id": job_id, "status": "failed", "error": str(exc)}

    return summary


async def _mark_failed(job_id: str, error: str) -> None:
    async with AsyncSessionLocal() as session:
        row = (
            await session.execute(
                select(ProviderImportJob).where(ProviderImportJob.id == job_id)
            )
        ).scalar_one_or_none()
        if row is None:
            return
        row.status = "failed"
        row.error = error
        row.message = error[:200]
        row.finished_at = utcnow()
        await session.commit()


async def cancel_job(job_id: str) -> bool:
    """Mark a queued / running job as cancelled.

    Currently this is cooperative — a running job checks the row's
    ``status`` between pages and stops if it sees ``cancelled``.
    Returns ``True`` if the job was found and updated.
    """
    async with AsyncSessionLocal() as session:
        row = (
            await session.execute(select(ProviderImportJob).where(ProviderImportJob.id == job_id))
        ).scalar_one_or_none()
        if row is None:
            return False
        if row.status not in {"queued", "running"}:
            return False
        row.status = "cancelled"
        row.message = "Cancelled by operator"
        row.finished_at = utcnow()
        await session.commit()
        return True


# ---------------------------------------------------------------------------
# Polybacktest implementation
# ---------------------------------------------------------------------------


async def _run_polybacktest_import(job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    coin = str(payload.get("coin") or "").strip().lower()
    market_ids = list(payload.get("market_ids") or [])
    start_ms = int(payload.get("start_ms") or 0)
    end_ms = int(payload.get("end_ms") or 0)
    if not market_ids:
        raise PolybacktestError("No market_ids in payload")
    if end_ms <= start_ms:
        raise PolybacktestError("end_ms must be > start_ms")

    client = await build_client_from_settings()
    snapshots_inserted = 0
    snapshots_fetched = 0
    rows_per_market: dict[str, dict[str, Any]] = {}

    try:
        for index, market_id in enumerate(market_ids):
            if await _is_cancelled(job_id):
                logger.info("provider import %s cancelled mid-run", job_id)
                break
            await _set_progress(
                job_id,
                progress=float(index) / max(1, len(market_ids)),
                message=f"Fetching market {market_id} ({index + 1}/{len(market_ids)})",
            )

            try:
                market = await client.get_market(coin, market_id)
            except PolybacktestError as exc:
                logger.warning(
                    "polybacktest get_market %s failed: %s", market_id, exc
                )
                rows_per_market[market_id] = {
                    "snapshots_inserted": 0,
                    "error": str(exc),
                }
                continue

            # Fetch snapshots (paged via offset).  Polybacktest v2 returns
            # ``total`` in every response so we know exactly when to stop.
            per_market_inserted = 0
            per_market_fetched = 0
            offset = 0
            total_avail: Optional[int] = None
            while True:
                if await _is_cancelled(job_id):
                    break
                try:
                    snaps, total_avail = await client.get_snapshots(
                        coin,
                        market_id,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        offset=offset,
                        limit=_SNAPSHOT_PAGE_LIMIT,
                        include_orderbook=True,
                    )
                except PolybacktestError as exc:
                    logger.warning(
                        "polybacktest get_snapshots %s failed: %s", market_id, exc
                    )
                    rows_per_market.setdefault(
                        market_id,
                        {"snapshots_inserted": per_market_inserted},
                    )["error"] = str(exc)
                    break

                # Each polybacktest snapshot row produces TWO records here
                # (one per side: up/down).  Page size is the snapshot
                # COUNT, not the record count, so we advance by half.
                row_count = len(snaps) // 2 if snaps else 0
                per_market_fetched += row_count
                snapshots_fetched += row_count

                if snaps:
                    inserted = await _insert_snapshots(
                        provider=PROVIDER_POLYBACKTEST,
                        snapshots=snaps,
                        market=market,
                    )
                    per_market_inserted += inserted
                    snapshots_inserted += inserted

                await _set_counters(
                    job_id,
                    snapshots_fetched=snapshots_fetched,
                    snapshots_inserted=snapshots_inserted,
                    trades_fetched=0,
                    api_calls=client.stats()["api_calls"],
                    bytes_downloaded=client.stats()["bytes_downloaded"],
                )

                # Advance the page cursor.  We've consumed ``row_count``
                # snapshot rows from the upstream — bump offset by the
                # page size we requested (unique snapshot rows).
                if not snaps or row_count == 0:
                    break
                offset += row_count
                if total_avail is not None and offset >= total_avail:
                    break

            rows_per_market[market_id] = {
                "snapshots_fetched": per_market_fetched,
                "snapshots_inserted": per_market_inserted,
                "total_available": total_avail,
            }

            # Catalog upsert.
            await _upsert_provider_dataset(
                provider=PROVIDER_POLYBACKTEST,
                coin=coin,
                external_id=market_id,
                external_slug=market.slug,
                title=market.title,
                token_ids=_token_ids_for_market(coin, market_id),
                start_ts=_ms_to_utc(start_ms),
                end_ts=_ms_to_utc(end_ms),
                snapshot_count=per_market_inserted,
                trade_count=0,
                payload=market.raw,
                last_import_job_id=job_id,
            )
    finally:
        await client.close()

    summary = {
        "job_id": job_id,
        "provider": PROVIDER_POLYBACKTEST,
        "coin": coin,
        "market_ids": market_ids,
        "snapshots_fetched": snapshots_fetched,
        "snapshots_inserted": snapshots_inserted,
        "trades_fetched": 0,
        "api_calls": client.stats()["api_calls"],
        "bytes_downloaded": client.stats()["bytes_downloaded"],
        "rate_limited_count": client.stats()["rate_limited_count"],
        "per_market": rows_per_market,
    }

    final_status = "cancelled" if await _is_cancelled(job_id) else "completed"
    await _mark_done(job_id, status=final_status, result=summary)
    return summary


_DISPATCH: dict[str, Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]] = {
    PROVIDER_POLYBACKTEST: _run_polybacktest_import,
}


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _token_ids_for_market(coin: str, market_id: str) -> list[str]:
    """Synthetic token IDs we write to ``MarketMicrostructureSnapshot``.

    Polymarket Up/Down has two outcomes per market — we expand them
    here so the backtest engine can iterate either side independently.
    """
    return [
        f"polybacktest:{coin}:{market_id}:up",
        f"polybacktest:{coin}:{market_id}:down",
    ]


def _token_id_for_side(coin: str, market_id: str, side: str) -> str:
    return f"polybacktest:{coin}:{market_id}:{side}"


def _ms_to_utc(ms: int) -> datetime:
    return datetime.fromtimestamp(int(ms) / 1000.0, tz=timezone.utc)


def _snapshot_id(
    token_id: str,
    observed_at: datetime,
    discriminator: Optional[str] = None,
    side: Optional[str] = None,
) -> str:
    """Stable ID derived from the natural key.

    Lets us safely re-run an import — duplicate rows are rejected at
    insert time via ON CONFLICT DO NOTHING (or pre-checked on SQLite).
    The discriminator slot accepts any provider-supplied uniqueness hint
    (snapshot_id, sequence number, etc.); ``side`` discriminates UP/DOWN
    when polybacktest emits a single snapshot row containing both books.
    """
    parts = [token_id, str(int(observed_at.timestamp() * 1000))]
    if side:
        parts.append(side)
    parts.append(str(discriminator) if discriminator is not None else "x")
    natural = "|".join(parts)
    return "pbt-" + hashlib.sha1(natural.encode("utf-8")).hexdigest()[:24]


def _serialize_levels(levels: list[tuple[float, float]]) -> list[list[float]]:
    return [[float(p), float(s)] for p, s in levels]


def _spread_bps(best_bid: Optional[float], best_ask: Optional[float]) -> Optional[float]:
    if best_bid is None or best_ask is None:
        return None
    if best_bid <= 0 or best_ask <= 0:
        return None
    mid = (best_bid + best_ask) / 2.0
    if mid <= 0:
        return None
    return (best_ask - best_bid) / mid * 10_000.0


async def _insert_snapshots(
    *,
    provider: str,
    snapshots: list[PolybacktestSnapshot],
    market: Any,
) -> int:
    """Batch-insert book snapshots, skipping rows that already exist.

    Returns the count of NEW rows inserted (not the page size).
    """
    if not snapshots:
        return 0

    rows: list[dict[str, Any]] = []
    for snap in snapshots:
        token_id = _token_id_for_side(snap.coin, snap.market_id, snap.side)
        observed_at = _ms_to_utc(snap.observed_at_ms)
        # Snapshot ID uses the polybacktest snapshot_id when present
        # (it's already globally unique per side+time).  Falls back to
        # the synthetic natural-key hash so re-running an import is
        # idempotent regardless of API response shape changes.
        sid_key = snap.snapshot_id or str(snap.sequence) if snap.snapshot_id else None
        rows.append(
            {
                "id": _snapshot_id(token_id, observed_at, sid_key, snap.side),
                "provider": provider,
                "token_id": token_id,
                "snapshot_type": "book",
                "observed_at": observed_at,
                "exchange_ts_ms": snap.observed_at_ms,
                "sequence": snap.sequence,
                "best_bid": snap.best_bid,
                "best_ask": snap.best_ask,
                "spread_bps": _spread_bps(snap.best_bid, snap.best_ask),
                "bids_json": _serialize_levels(snap.bids),
                "asks_json": _serialize_levels(snap.asks),
                "trade_price": None,
                "trade_size": None,
                "trade_side": None,
                "payload_json": {
                    "provider": provider,
                    "coin": snap.coin,
                    "market_id": snap.market_id,
                    "side": snap.side,
                    "external_market_slug": getattr(market, "slug", None),
                    "external_market_title": getattr(market, "title", None),
                    # Spot reference price + cross-side prices captured
                    # alongside this snapshot — useful for the agent +
                    # backtest engine when reasoning about the market's
                    # state without a separate fetch.
                    "spot_price": snap.coin_price,
                    "price_up": snap.price_up,
                    "price_down": snap.price_down,
                    "depth_levels_bids": len(snap.bids),
                    "depth_levels_asks": len(snap.asks),
                    "polybacktest_snapshot_id": snap.snapshot_id,
                },
            }
        )

    return await _bulk_insert_microstructure(rows)


async def _bulk_insert_microstructure(rows: list[dict[str, Any]]) -> int:
    """Insert with ``ON CONFLICT DO NOTHING`` semantics (Postgres).

    Falls back to a per-row INSERT-or-skip on non-Postgres engines.
    Returns the number of newly inserted rows (not counting skips).
    """
    if not rows:
        return 0
    inserted_total = 0
    async with AsyncSessionLocal() as session:
        # Preferred path: Postgres ``INSERT ... ON CONFLICT (id) DO NOTHING``.
        try:
            for batch_start in range(0, len(rows), _INSERT_BATCH_SIZE):
                batch = rows[batch_start : batch_start + _INSERT_BATCH_SIZE]
                stmt = pg_insert(MarketMicrostructureSnapshot).values(batch)
                stmt = stmt.on_conflict_do_nothing(index_elements=["id"])
                result = await session.execute(stmt)
                # rowcount from ON CONFLICT only counts inserted rows on
                # Postgres ≥9.5 — same engine we ship on, so this is
                # accurate.  Negative rowcount → fall back to length.
                rc = getattr(result, "rowcount", None)
                inserted_total += rc if rc is not None and rc >= 0 else len(batch)
            await session.commit()
            return inserted_total
        except Exception as exc:
            await session.rollback()
            logger.debug("bulk insert via pg_insert failed, falling back: %s", exc)

        # Fallback: per-row check, used for SQLite-backed unit tests.
        for row in rows:
            existing = (
                await session.execute(
                    select(MarketMicrostructureSnapshot.id).where(
                        MarketMicrostructureSnapshot.id == row["id"]
                    )
                )
            ).first()
            if existing is not None:
                continue
            session.add(MarketMicrostructureSnapshot(**row))
            inserted_total += 1
        await session.commit()
        return inserted_total


async def _upsert_provider_dataset(
    *,
    provider: str,
    coin: Optional[str],
    external_id: str,
    external_slug: Optional[str],
    title: Optional[str],
    token_ids: list[str],
    start_ts: Optional[datetime],
    end_ts: Optional[datetime],
    snapshot_count: int,
    trade_count: int,
    payload: Optional[dict[str, Any]],
    last_import_job_id: Optional[str],
) -> None:
    """Insert-or-update the catalog row for this dataset.

    Counts are recomputed from the current import (so rerunning a
    partial window doesn't double-count).
    """
    async with AsyncSessionLocal() as session:
        existing = (
            await session.execute(
                select(ProviderDataset).where(
                    and_(
                        ProviderDataset.provider == provider,
                        ProviderDataset.external_id == str(external_id),
                    )
                )
            )
        ).scalar_one_or_none()

        # Recompute the actual stored counts from microstructure to
        # keep the catalog accurate even after multiple partial imports.
        snap_count = await _count_microstructure_rows(
            session, provider=provider, token_ids=token_ids, snapshot_type="book"
        )
        trade_count_actual = await _count_microstructure_rows(
            session, provider=provider, token_ids=token_ids, snapshot_type="trade"
        )

        if existing is None:
            row = ProviderDataset(
                id=f"pds-{int(time.time() * 1000)}-{_short_hash([provider, external_id])}",
                provider=provider,
                coin=coin,
                external_id=str(external_id),
                external_slug=external_slug,
                title=title,
                asset_class="prediction",
                token_ids_json=token_ids,
                start_ts=start_ts,
                end_ts=end_ts,
                snapshot_count=int(snap_count),
                trade_count=int(trade_count_actual),
                last_imported_at=utcnow(),
                last_import_job_id=last_import_job_id,
                payload_json=payload or {},
            )
            session.add(row)
        else:
            existing.coin = coin or existing.coin
            existing.external_slug = external_slug or existing.external_slug
            existing.title = title or existing.title
            existing.token_ids_json = token_ids
            # Widen the window if this import extended either edge.
            if start_ts is not None:
                existing.start_ts = (
                    min(existing.start_ts, start_ts) if existing.start_ts else start_ts
                )
            if end_ts is not None:
                existing.end_ts = (
                    max(existing.end_ts, end_ts) if existing.end_ts else end_ts
                )
            existing.snapshot_count = int(snap_count)
            existing.trade_count = int(trade_count_actual)
            existing.last_imported_at = utcnow()
            existing.last_import_job_id = last_import_job_id or existing.last_import_job_id
            existing.payload_json = payload or existing.payload_json
        await session.commit()


async def _count_microstructure_rows(
    session: Any,
    *,
    provider: str,
    token_ids: list[str],
    snapshot_type: str,
) -> int:
    if not token_ids:
        return 0
    stmt = select(func.count(MarketMicrostructureSnapshot.id)).where(
        and_(
            MarketMicrostructureSnapshot.provider == provider,
            MarketMicrostructureSnapshot.token_id.in_(token_ids),
            MarketMicrostructureSnapshot.snapshot_type == snapshot_type,
        )
    )
    return int((await session.execute(stmt)).scalar_one() or 0)


# ---------------------------------------------------------------------------
# Cooperative cancel + progress
# ---------------------------------------------------------------------------


async def _is_cancelled(job_id: str) -> bool:
    async with AsyncSessionLocal() as session:
        status = (
            await session.execute(
                select(ProviderImportJob.status).where(ProviderImportJob.id == job_id)
            )
        ).scalar_one_or_none()
        return status == "cancelled"


async def _set_progress(job_id: str, *, progress: float, message: str) -> None:
    async with AsyncSessionLocal() as session:
        await session.execute(
            update(ProviderImportJob)
            .where(ProviderImportJob.id == job_id)
            .values(progress=max(0.0, min(1.0, progress)), message=message[:200])
        )
        await session.commit()


async def _set_counters(
    job_id: str,
    *,
    snapshots_fetched: int,
    snapshots_inserted: int,
    trades_fetched: int,
    api_calls: int,
    bytes_downloaded: int,
) -> None:
    async with AsyncSessionLocal() as session:
        await session.execute(
            update(ProviderImportJob)
            .where(ProviderImportJob.id == job_id)
            .values(
                snapshots_fetched=int(snapshots_fetched),
                snapshots_inserted=int(snapshots_inserted),
                trades_fetched=int(trades_fetched),
                api_calls=int(api_calls),
                bytes_downloaded=int(bytes_downloaded),
            )
        )
        await session.commit()


async def _mark_done(job_id: str, *, status: str, result: dict[str, Any]) -> None:
    async with AsyncSessionLocal() as session:
        row = (
            await session.execute(
                select(ProviderImportJob).where(ProviderImportJob.id == job_id)
            )
        ).scalar_one_or_none()
        if row is None:
            return
        row.status = status
        row.progress = 1.0 if status == "completed" else row.progress
        row.message = "Done" if status == "completed" else row.message
        row.result_json = result
        row.finished_at = utcnow()
        await session.commit()


# ---------------------------------------------------------------------------
# Catalog read helpers (used by the API layer)
# ---------------------------------------------------------------------------


async def list_provider_datasets(
    *,
    provider: Optional[str] = None,
    coin: Optional[str] = None,
    limit: int = 200,
) -> list[ProviderDataset]:
    async with AsyncSessionLocal() as session:
        stmt = select(ProviderDataset)
        if provider:
            stmt = stmt.where(ProviderDataset.provider == provider)
        if coin:
            stmt = stmt.where(ProviderDataset.coin == coin)
        stmt = stmt.order_by(ProviderDataset.updated_at.desc()).limit(int(limit))
        return list((await session.execute(stmt)).scalars().all())


async def get_provider_dataset(dataset_id: str) -> Optional[ProviderDataset]:
    async with AsyncSessionLocal() as session:
        return (
            await session.execute(
                select(ProviderDataset).where(ProviderDataset.id == dataset_id)
            )
        ).scalar_one_or_none()


async def delete_provider_dataset(dataset_id: str) -> bool:
    """Delete a catalog entry **and** its underlying microstructure rows.

    Returns ``True`` if the dataset existed.
    """
    async with AsyncSessionLocal() as session:
        row = (
            await session.execute(
                select(ProviderDataset).where(ProviderDataset.id == dataset_id)
            )
        ).scalar_one_or_none()
        if row is None:
            return False
        token_ids = list(row.token_ids_json or [])
        if token_ids:
            await session.execute(
                sa_delete(MarketMicrostructureSnapshot).where(
                    and_(
                        MarketMicrostructureSnapshot.provider == row.provider,
                        MarketMicrostructureSnapshot.token_id.in_(token_ids),
                    )
                )
            )
        await session.execute(
            sa_delete(ProviderDataset).where(ProviderDataset.id == dataset_id)
        )
        await session.commit()
        return True


async def resolve_dataset_scope(dataset_ids: list[str]) -> Optional[dict[str, Any]]:
    """Convert a set of provider_dataset IDs into the (token_ids, start, end)
    scope tuple the unified backtester expects.

    Returns ``None`` when no IDs resolve.  When multiple datasets are
    selected the window is the **union** (min start, max end) and
    token_ids is the concatenation.
    """
    if not dataset_ids:
        return None
    async with AsyncSessionLocal() as session:
        rows = list(
            (
                await session.execute(
                    select(ProviderDataset).where(
                        ProviderDataset.id.in_([str(x) for x in dataset_ids])
                    )
                )
            )
            .scalars()
            .all()
        )
    if not rows:
        return None
    token_ids: list[str] = []
    starts: list[datetime] = []
    ends: list[datetime] = []
    labels: list[str] = []
    for row in rows:
        for tid in row.token_ids_json or []:
            if tid not in token_ids:
                token_ids.append(str(tid))
        if row.start_ts is not None:
            starts.append(row.start_ts)
        if row.end_ts is not None:
            ends.append(row.end_ts)
        labels.append(row.title or row.external_slug or row.external_id)
    return {
        "dataset_ids": [r.id for r in rows],
        "labels": labels,
        "token_ids": token_ids,
        "start": min(starts) if starts else None,
        "end": max(ends) if ends else None,
    }


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def _short_hash(parts: Any) -> str:
    return hashlib.sha1(repr(parts).encode("utf-8")).hexdigest()[:8]


__all__ = [
    "PROVIDER_POLYBACKTEST",
    "CreatePolybacktestJobSpec",
    "enqueue_polybacktest_import",
    "claim_next_queued_job",
    "run_job",
    "cancel_job",
    "list_provider_datasets",
    "get_provider_dataset",
    "delete_provider_dataset",
    "resolve_dataset_scope",
]
