"""Provider data import service.

Coordinates the lifecycle of a ``ProviderImportJob``:

  1. Read the job row, mark it ``running``.
  2. Resolve provider-specific config (API key, base URL).
  3. Iterate over the requested markets / time windows, paginate the
     provider's API, accumulate each market's snapshots and write them to
     canonical ``SNAPSHOT_SCHEMA`` parquet (Up + Down files in one window
     directory) — the SAME parquet schema + layout every other ingest
     source uses.  No book data is written to Postgres.
  4. Register a ``ProviderDataset`` catalog entry per (provider, market_id)
     with ``storage_type='parquet'`` so the unified backtester's
     ``find_parquet_coverage()`` routes replays at the files directly.
  5. Mark the job ``completed`` (or ``failed`` with the error and the
     partial counters preserved).

Idempotent: rerunning the same job over the same window recomputes the
market-wide span and overwrites the canonical parquet files in place, and
upserts the catalog row keyed on ``(provider, external_id)``.

Currently implements only ``polybacktest``.  Adding a new provider is a
matter of writing a sibling ``_run_<provider>_import`` function and
wiring it into ``_DISPATCH``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from sqlalchemy import and_, delete as sa_delete, select, update

from models.database import (
    AsyncSessionLocal,
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

# Canonical parquet storage_type — matches what telonex_import_service and
# parquet_scanner write, so the backtester's ``find_parquet_coverage()``
# exact-match filter (storage_type == 'parquet') discovers polybacktest
# imports the same way. Keeping every provider on the SAME parquet schema +
# catalog convention is the whole point of the unified ingest path: no
# provider writes book snapshots to Postgres on the hot or batch path.
_STORAGE_TYPE_PARQUET = "parquet"

# Page sizes — tuned for the typical Pro-tier rate limit.  Polybacktest
# v2 caps snapshots at 1000 per page; we use the max to minimize API
# calls per import.  Trades are not exposed for prediction markets
# (they live only on the spot/futures reference endpoints, which we
# don't currently import).
_SNAPSHOT_PAGE_LIMIT = 1000


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

    # Pre-fetch the market metadata via list_markets and index by id.  The
    # per-market ``/v2/markets/{id}`` endpoint has been returning 500 for
    # months for every request, even with paid credits — but list_markets
    # serves the same payload (clob_token_up/down, slug, title, …) at
    # 1000/page.  Paginating list_markets once up-front lets the inner loop
    # resolve every market_id without a single get_market round-trip.
    market_index: dict[str, Any] = {}
    try:
        wanted = set(str(x) for x in market_ids)
        offset = 0
        # Bound the prefetch to a few pages — operator-supplied market_ids
        # are typically <= 100 markets, and the most-recent pages cover
        # current/recent trading activity.  Stop once we've found them all
        # or paginated 5000 markets.
        for _ in range(5):
            page, total = await client.list_markets(coin, offset=offset, limit=1000)
            for m in page:
                mid = str(getattr(m, "market_id", "") or "")
                if mid in wanted:
                    market_index[mid] = m
            if not page or len(market_index) >= len(wanted) or offset + len(page) >= int(total or 0):
                break
            offset += len(page)
        logger.info(
            "polybacktest import: pre-fetched %d/%d market metadata records via list_markets",
            len(market_index),
            len(wanted),
        )
    except Exception as exc:  # noqa: BLE001 — fall back to per-market get_market below
        logger.warning("polybacktest list_markets pre-fetch failed: %s", exc)

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

            # Prefer the cached list_markets payload; fall back to get_market
            # only when the cache missed (e.g. very new market not in the
            # most-recent pages).  get_market is still upstream-broken (500),
            # so an explicit empty market_index entry just means "skip this id"
            # rather than wasting the whole window on retries.
            market = market_index.get(str(market_id))
            if market is None:
                try:
                    market = await client.get_market(coin, market_id)
                except PolybacktestError as exc:
                    logger.warning(
                        "polybacktest market lookup %s failed (list_markets miss + get_market upstream broken): %s",
                        market_id, exc,
                    )
                    rows_per_market[market_id] = {
                        "snapshots_inserted": 0,
                        "error": f"market metadata unavailable: {exc}",
                    }
                    continue

            # Fetch snapshots (paged via offset).  Polybacktest v2 returns
            # ``total`` in every response so we know exactly when to stop.
            # We accumulate the whole market's snapshots in memory and write
            # them to canonical parquet once at the end (one window dir per
            # market, Up + Down files side-by-side) rather than batch-inserting
            # to Postgres per page — keeping the import path entirely off the
            # DB hot path and on the same parquet schema as every other source.
            per_market_inserted = 0
            per_market_fetched = 0
            offset = 0
            total_avail: Optional[int] = None
            market_snaps: list[PolybacktestSnapshot] = []
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
                    market_snaps.extend(snaps)

                await _set_counters(
                    job_id,
                    snapshots_fetched=snapshots_fetched,
                    # ``inserted`` finalises after the parquet write below; keep
                    # the progress counter at the running total committed so far.
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

            # Write the market's accumulated snapshots to canonical parquet
            # off the event loop, then register the parquet-backed dataset.
            outputs = await asyncio.to_thread(
                _write_polybacktest_parquet,
                snapshots=market_snaps,
                market=market,
            )
            per_market_inserted = sum(o[3] for o in outputs)
            snapshots_inserted += per_market_inserted
            await _set_counters(
                job_id,
                snapshots_fetched=snapshots_fetched,
                snapshots_inserted=snapshots_inserted,
                trades_fetched=0,
                api_calls=client.stats()["api_calls"],
                bytes_downloaded=client.stats()["bytes_downloaded"],
            )

            rows_per_market[market_id] = {
                "snapshots_fetched": per_market_fetched,
                "snapshots_inserted": per_market_inserted,
                "total_available": total_avail,
            }

            if outputs:
                await _register_polybacktest_parquet_dataset(
                    coin=coin,
                    market_id=market_id,
                    market=market,
                    outputs=outputs,
                    requested_start=_ms_to_utc(start_ms),
                    requested_end=_ms_to_utc(end_ms),
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


def _token_id_for_side(coin: str, market_id: str, side: str) -> str:
    """Synthetic fallback token id (only when polybacktest didn't expose
    the real Polymarket clob_token_ids — see ``_token_id_for_snap``)."""
    return f"polybacktest:{coin}:{market_id}:{side}"


def _ms_to_utc(ms: int) -> datetime:
    return datetime.fromtimestamp(int(ms) / 1000.0, tz=timezone.utc)


def _spread_bps(best_bid: Optional[float], best_ask: Optional[float]) -> Optional[float]:
    if best_bid is None or best_ask is None:
        return None
    if best_bid <= 0 or best_ask <= 0:
        return None
    mid = (best_bid + best_ask) / 2.0
    if mid <= 0:
        return None
    return (best_ask - best_bid) / mid * 10_000.0


def _token_id_for_snap(snap: PolybacktestSnapshot, market: Any) -> str:
    """Resolve the canonical token id we write a snapshot under.

    Prefer the REAL Polymarket clob_token_ids exposed by polybacktest's
    market metadata over the synthetic ``polybacktest:btc:X:up`` slugs:
    polybacktest captures Polymarket markets, so the snapshots ARE
    Polymarket book state.  Storing them under synthetic IDs meant crypto
    strategies (which query coverage by their real opportunity-history
    clob_token_ids) silently saw zero polybacktest coverage.  Fall back to
    the synthetic ID only when clob_token_{up,down} is missing (rare).
    """
    real_up = getattr(market, "clob_token_up", None)
    real_down = getattr(market, "clob_token_down", None)
    if snap.side == "up" and real_up:
        return str(real_up)
    if snap.side == "down" and real_down:
        return str(real_down)
    return _token_id_for_side(snap.coin, snap.market_id, snap.side)


def _write_polybacktest_parquet(
    *,
    snapshots: list[PolybacktestSnapshot],
    market: Any,
) -> list[tuple[str, Path, str, int, datetime, datetime]]:
    """Write a market's book snapshots to canonical SNAPSHOT_SCHEMA parquet.

    Groups snapshots by real Polymarket clob_token_id (one file per side)
    and writes them all into a SINGLE window directory keyed off the
    market-wide span, mirroring the telonex canonical layout so the
    backtester's ``find_parquet_coverage()`` discovers Up + Down side-by-
    side.  Synchronous (PyArrow) — call via ``asyncio.to_thread``.

    Returns one ``(coin, file_path, token_id, n_rows, span_start, span_end)``
    tuple per token written (empty if no snapshots).
    """
    if not snapshots:
        return []

    import pyarrow as pa
    import pyarrow.parquet as pq

    from services.external_data.parquet_schema import SNAPSHOT_SCHEMA, parquet_path_for

    by_token: dict[str, list[PolybacktestSnapshot]] = {}
    for snap in snapshots:
        by_token.setdefault(_token_id_for_snap(snap, market), []).append(snap)

    # One shared window span for the whole market so Up + Down land in the
    # same window_dir (a single ProviderDataset.storage_uri points at it).
    all_ms = [int(s.observed_at_ms) for s in snapshots]
    span_start = _ms_to_utc(min(all_ms)).replace(microsecond=0)
    span_end = _ms_to_utc(max(all_ms)).replace(microsecond=0)
    if span_end <= span_start:
        span_end = span_start + timedelta(seconds=1)

    coin = (snapshots[0].coin or "_").lower()
    outputs: list[tuple[str, Path, str, int, datetime, datetime]] = []

    for token_id, snaps in by_token.items():
        snaps.sort(key=lambda s: s.observed_at_ms)
        n = len(snaps)
        observed_us = [int(s.observed_at_ms) * 1000 for s in snaps]
        bids_price = [[float(p) for p, _s in s.bids] for s in snaps]
        bids_size = [[float(sz) for _p, sz in s.bids] for s in snaps]
        asks_price = [[float(p) for p, _s in s.asks] for s in snaps]
        asks_size = [[float(sz) for _p, sz in s.asks] for s in snaps]

        table = pa.table(
            {
                "token_id":       pa.array([token_id] * n, pa.string()),
                "observed_at_us": pa.array(observed_us, pa.int64()),
                "sequence":       pa.array([s.sequence for s in snaps], pa.int64()),
                "best_bid":       pa.array([s.best_bid for s in snaps], pa.float64()),
                "best_ask":       pa.array([s.best_ask for s in snaps], pa.float64()),
                "spread_bps":     pa.array(
                    [_spread_bps(s.best_bid, s.best_ask) for s in snaps], pa.float64()
                ),
                "bids_price":     pa.array(bids_price, pa.list_(pa.float64())),
                "bids_size":      pa.array(bids_size, pa.list_(pa.float64())),
                "asks_price":     pa.array(asks_price, pa.list_(pa.float64())),
                "asks_size":      pa.array(asks_size, pa.list_(pa.float64())),
                "trade_price":    pa.array([None] * n, pa.float64()),
                "trade_size":     pa.array([None] * n, pa.float64()),
                "trade_side":     pa.array([None] * n, pa.string()),
            },
            schema=SNAPSHOT_SCHEMA,
        )

        dest = parquet_path_for(
            provider=PROVIDER_POLYBACKTEST,
            coin=coin,
            token_id=token_id,
            start=span_start,
            end=span_end,
            kind="snapshots",
        )
        dest.window_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, str(dest.file_path), compression="snappy")
        outputs.append((coin, dest.file_path, token_id, n, span_start, span_end))

    return outputs


def _file_uri(path: Path) -> str:
    """Cross-platform ``file://`` URI for a filesystem path."""
    return path.resolve().as_uri()


async def _register_polybacktest_parquet_dataset(
    *,
    coin: str,
    market_id: str,
    market: Any,
    outputs: list[tuple[str, Path, str, int, datetime, datetime]],
    requested_start: datetime,
    requested_end: datetime,
    last_import_job_id: Optional[str],
) -> None:
    """Insert-or-update the ``ProviderDataset`` catalog row for a parquet
    import.  Mirrors telonex's canonical registration:

      • ``storage_type='parquet'`` so ``find_parquet_coverage()`` (which
        exact-matches that value) routes the backtester at the parquet
        files instead of a SQL replay.
      • ``storage_uri`` points at the shared window directory holding the
        Up + Down snapshot files.
      • ``token_ids_json`` holds the REAL Polymarket clob_token_ids the
        files are keyed under, so per-token routing matches live opps.
      • ``start_ts/end_ts`` reflect the ACTUAL data span (not the requested
        calendar window) so coverage overlap is precise.
    """
    if not outputs:
        return

    window_dir = outputs[0][1].parent
    storage_uri = _file_uri(window_dir)

    token_ids: list[str] = []
    seen: set[str] = set()
    for _coin, _path, tok, _n, _s, _e in outputs:
        if tok not in seen:
            token_ids.append(tok)
            seen.add(tok)

    span_start = min(o[4] for o in outputs)
    span_end = max(o[5] for o in outputs)
    total_rows = sum(o[3] for o in outputs)

    # Capture the market metadata the backtest-time crypto_update projection
    # needs to reconstruct dispatch events from these book parquet files
    # (see services.marketdata.projection).  Without these the dataset isn't
    # self-describing: the projection can't tell which token is UP vs DOWN,
    # derive seconds_left, or label the market.
    def _iso(dt: Any) -> Optional[str]:
        if isinstance(dt, datetime):
            return dt.astimezone(timezone.utc).isoformat()
        return None

    payload: dict[str, Any] = {
        "coin": coin,
        "market_id": market_id,
        "slug": getattr(market, "slug", None),
        "title": getattr(market, "title", None),
        "condition_id": getattr(market, "condition_id", None),
        "clob_token_up": getattr(market, "clob_token_up", None),
        "clob_token_down": getattr(market, "clob_token_down", None),
        "market_type": getattr(market, "market_type", None),
        "market_start_time": _iso(getattr(market, "start_time", None)),
        "market_end_time": _iso(getattr(market, "end_time", None)),
        "coin_price_start": getattr(market, "coin_price_start", None),
        "coin_price_end": getattr(market, "coin_price_end", None),
        "requested_start": requested_start.isoformat(),
        "requested_end": requested_end.isoformat(),
        "canonical": True,
        "schema_version": "snapshots_v2",
    }

    external_id = str(market_id)
    dataset_id = "polybacktest:" + hashlib.sha1(
        f"{PROVIDER_POLYBACKTEST}|{external_id}".encode("utf-8")
    ).hexdigest()[:16]

    async with AsyncSessionLocal() as session:
        existing = (
            await session.execute(
                select(ProviderDataset).where(
                    and_(
                        ProviderDataset.provider == PROVIDER_POLYBACKTEST,
                        ProviderDataset.external_id == external_id,
                    )
                )
            )
        ).scalar_one_or_none()

        if existing is None:
            row = ProviderDataset(
                id=dataset_id,
                provider=PROVIDER_POLYBACKTEST,
                coin=coin,
                external_id=external_id,
                external_slug=getattr(market, "slug", None),
                title=getattr(market, "title", None),
                asset_class="prediction",
                token_ids_json=token_ids,
                storage_type=_STORAGE_TYPE_PARQUET,
                storage_uri=storage_uri,
                start_ts=span_start,
                end_ts=span_end,
                snapshot_count=int(total_rows),
                trade_count=0,
                last_imported_at=utcnow(),
                last_import_job_id=last_import_job_id,
                payload_json=payload,
            )
            session.add(row)
        else:
            existing.coin = coin or existing.coin
            existing.external_slug = getattr(market, "slug", None) or existing.external_slug
            existing.title = getattr(market, "title", None) or existing.title
            existing.asset_class = "prediction"
            existing.token_ids_json = token_ids
            existing.storage_type = _STORAGE_TYPE_PARQUET
            existing.storage_uri = storage_uri
            existing.start_ts = min(existing.start_ts, span_start) if existing.start_ts else span_start
            existing.end_ts = max(existing.end_ts, span_end) if existing.end_ts else span_end
            existing.snapshot_count = int(total_rows)
            existing.trade_count = 0
            existing.last_imported_at = utcnow()
            existing.last_import_job_id = last_import_job_id or existing.last_import_job_id
            existing.payload_json = payload
        await session.commit()


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
    """Delete a catalog entry **and** its underlying storage.

    Removes the on-disk window directory the canonical parquet files live in,
    then deletes the catalog row.  Returns ``True`` if the dataset existed.
    """
    async with AsyncSessionLocal() as session:
        row = (
            await session.execute(
                select(ProviderDataset).where(ProviderDataset.id == dataset_id)
            )
        ).scalar_one_or_none()
        if row is None:
            return False

        if row.storage_uri:
            # Remove the on-disk window directory. Best-effort — a missing /
            # locked dir shouldn't block catalog cleanup.
            try:
                from services.marketdata.coverage import _uri_to_path

                window_dir = Path(_uri_to_path(row.storage_uri))
                if window_dir.exists() and window_dir.is_dir():
                    import shutil

                    await asyncio.to_thread(shutil.rmtree, window_dir, True)
            except Exception:
                logger.exception(
                    "delete_provider_dataset: failed to remove parquet dir for %s",
                    dataset_id,
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
