"""
Shared state for scanner/trader workers.
Small operational snapshots live in DB. The scanner market catalog keeps DB metadata
plus an atomic local payload file so large catalogs do not monopolize Postgres.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Optional

from sqlalchemy import Text, cast, func, or_, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    AsyncSessionLocal,
    ScannerControl,
    ScannerMarketHistory,
    ScannerSnapshot,
    ScannerRun,
    OpportunityState,
    ScannerSloIncident,
    release_conn,
)
from models.opportunity import Opportunity, OpportunityFilter
from services.event_bus import event_bus
from services.market_tradability import get_market_tradability_map
from utils.converters import format_iso_utc_z, normalize_market_id, parse_iso_datetime
from utils.retry import commit_with_retry as _shared_commit_with_retry
from utils.utcnow import utcnow

SQL_IN_CLAUSE_CHUNK_SIZE = 900


def _chunked_in(column, values, chunk_size: int = SQL_IN_CLAUSE_CHUNK_SIZE):
    values = list(values)
    if len(values) <= chunk_size:
        return column.in_(values)
    clauses = []
    for i in range(0, len(values), chunk_size):
        clauses.append(column.in_(values[i : i + chunk_size]))
    return or_(*clauses)


def _chunked_values(values: set[str] | list[str], chunk_size: int = SQL_IN_CLAUSE_CHUNK_SIZE) -> list[list[str]]:
    ordered = list(values)
    return [ordered[i : i + chunk_size] for i in range(0, len(ordered), chunk_size)]


logger = logging.getLogger(__name__)

SNAPSHOT_ID = "latest"
TRADERS_SNAPSHOT_ID = "traders_latest"
CONTROL_ID = "default"

# In-memory targeted condition IDs for the next scan request.
# Set by the evaluate endpoint, consumed and cleared by the scanner worker.
_pending_targeted_condition_ids: list[str] = []
_pending_targeted_condition_ids_lock = Lock()
_scanner_projection_lock = Lock()
_scanner_projection_task: asyncio.Task | None = None
_scanner_projection_pending: dict[str, Any] | None = None
_SCANNER_STATE_PROJECTION_TIMEOUT_SECONDS = 10.0  # was 15s; cycle 8 of the
# perf-harness loop hit a 181s zombie-backend incident on
# opportunity_state UPDATE.  SET LOCAL statement_timeout (8s) should
# have aborted that server-side, but evidently didn't apply (now
# logged via _apply_local_db_timeouts).  Tightening the asyncio
# wait_for to 10s caps the worst-case clock-time-to-cancel even when
# the per-statement timeout misfires.  10s = 8s statement budget +
# 2s margin for asyncpg cancel propagation.
_NONCRITICAL_LOCK_TIMEOUT_MS = 1000
_SCANNER_SNAPSHOT_WRITE_STATEMENT_TIMEOUT_MS = 4000
_SCANNER_ACTIVITY_WRITE_STATEMENT_TIMEOUT_MS = 2500
_TRADERS_SNAPSHOT_WRITE_STATEMENT_TIMEOUT_MS = 4000
_MARKET_CATALOG_WRITE_STATEMENT_TIMEOUT_MS = 8000
_SCANNER_STATE_PROJECTION_STATEMENT_TIMEOUT_MS = 8000

# Process-local TTL cache for the market catalog payload.  The catalog can
# be tens of megabytes; reading it repeatedly costs noticeable I/O and JSON
# decoding time even when it no longer comes from Postgres TOAST storage.
# The catalog updates at roughly the scanner cadence (~1 write/minute), so
# a 5s TTL keeps readers within one cycle of fresh data while collapsing
# redundant reads into a single fetch.
#
# The cache is per-process; cross-process invalidation relies solely
# on the TTL.  ``write_market_catalog`` clears it eagerly so the
# writer's own process sees its own write immediately.
_MARKET_CATALOG_CACHE_TTL_SECONDS = 5.0
_market_catalog_cache: dict[str, Any] = {}
_market_catalog_cache_lock = asyncio.Lock()


def _market_catalog_cache_invalidate() -> None:
    _market_catalog_cache.clear()


async def _commit_with_retry(session: AsyncSession) -> None:
    try:
        await _shared_commit_with_retry(session)
    except DBAPIError:
        raise


async def _publish_opportunity_runtime_events(event_messages: list[dict[str, Any]]) -> None:
    if not event_messages:
        return
    try:
        await event_bus.publish(
            "opportunity_events",
            {
                "events": event_messages,
            },
        )
        for event in event_messages:
            opportunity_payload = event.get("opportunity") if isinstance(event, dict) else {}
            if not isinstance(opportunity_payload, dict):
                opportunity_payload = {}
            await event_bus.publish(
                "opportunity_update",
                {
                    "id": event.get("id"),
                    "stable_id": event.get("stable_id"),
                    "run_id": event.get("run_id"),
                    "event_type": event.get("event_type"),
                    "revision": int(opportunity_payload.get("revision") or 0),
                    "opportunity": opportunity_payload,
                    "created_at": event.get("created_at"),
                },
            )
    except Exception:
        pass


async def _apply_local_db_timeouts(
    session: AsyncSession,
    *,
    statement_timeout_ms: int,
    lock_timeout_ms: int = _NONCRITICAL_LOCK_TIMEOUT_MS,
) -> None:
    statement_timeout = max(250, int(statement_timeout_ms))
    lock_timeout = max(250, int(lock_timeout_ms))
    # Cycle 8 of the perf-harness loop hit a zombie-backend incident
    # where an opportunity_state UPDATE ran for 181s before the pool
    # watchdog force-killed the backend.  The SET LOCAL timeouts here
    # MUST apply for postgres to abort blocked queries server-side; if
    # they fail silently the only recovery is the 180s watchdog.
    # Promote both calls from silent except-pass to logged warnings so
    # we can diagnose if the SET LOCAL ever fails on production.
    try:
        await session.execute(text(f"SET LOCAL statement_timeout = '{statement_timeout}ms'"))
    except Exception as exc:
        logger.warning(
            "SET LOCAL statement_timeout=%dms failed; backend will rely on outer wait_for",
            statement_timeout,
            exc_info=exc,
        )
    try:
        await session.execute(text(f"SET LOCAL lock_timeout = '{lock_timeout}ms'"))
    except Exception as exc:
        logger.warning(
            "SET LOCAL lock_timeout=%dms failed; row-lock waits will fall back to outer wait_for",
            lock_timeout,
            exc_info=exc,
        )


async def _project_scanner_state(
    opportunities: list[Opportunity],
    status: dict[str, Any],
    completed_at: datetime,
) -> None:
    try:
        payload, skipped = await asyncio.to_thread(_serialize_opportunity_payload, opportunities)
        if skipped:
            logger.warning(
                "Skipped %d/%d opportunities while projecting scanner state",
                skipped,
                len(opportunities),
            )
    except Exception:
        logger.exception("Scanner opportunity-state serialization failed")
        return

    async def _run_projection() -> list[dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            await _apply_local_db_timeouts(
                session,
                statement_timeout_ms=_SCANNER_STATE_PROJECTION_STATEMENT_TIMEOUT_MS,
            )
            event_messages = await _persist_incremental_state(session, payload, status, completed_at)
            await _commit_with_retry(session)
            return event_messages

    try:
        event_messages = await asyncio.wait_for(
            _run_projection(),
            timeout=_SCANNER_STATE_PROJECTION_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning("Scanner opportunity-state projection timed out; dropped batch count=%s", len(payload))
        return
    except Exception:
        logger.exception("Scanner opportunity-state projection failed")
        return

    await _publish_opportunity_runtime_events(event_messages)


async def _run_scanner_state_projection_loop() -> None:
    global _scanner_projection_pending
    global _scanner_projection_task

    try:
        while True:
            with _scanner_projection_lock:
                pending = _scanner_projection_pending
                _scanner_projection_pending = None
            if pending is None:
                return
            await _project_scanner_state(
                pending["opportunities"],
                pending["status"],
                pending["completed_at"],
            )
    finally:
        with _scanner_projection_lock:
            _scanner_projection_task = None
            should_restart = _scanner_projection_pending is not None
        if should_restart:
            _schedule_scanner_state_projection(
                opportunities=[],
                status={},
                completed_at=utcnow(),
                reuse_pending=True,
            )


def _schedule_scanner_state_projection(
    *,
    opportunities: list[Opportunity],
    status: dict[str, Any],
    completed_at: datetime,
    reuse_pending: bool = False,
) -> None:
    global _scanner_projection_pending
    global _scanner_projection_task

    with _scanner_projection_lock:
        if not reuse_pending:
            _scanner_projection_pending = {
                "opportunities": list(opportunities),
                "status": {"current_activity": status.get("current_activity")},
                "completed_at": completed_at,
            }
        if _scanner_projection_task is None or _scanner_projection_task.done():
            _scanner_projection_task = asyncio.create_task(
                _run_scanner_state_projection_loop(),
                name="scanner-state-projection",
            )


def _normalize_history_points(points: Any) -> list[dict[str, Any]]:
    if not isinstance(points, list):
        return []
    normalized = [dict(point) for point in points if isinstance(point, dict)]
    return normalized if len(normalized) >= 2 else []


async def upsert_scanner_market_history(
    session: AsyncSession,
    market_history: dict[str, list[dict[str, Any]]],
) -> int:
    rows: list[dict[str, Any]] = []
    now = utcnow()
    for raw_market_id, raw_points in market_history.items():
        market_id = normalize_market_id(raw_market_id)
        points = _normalize_history_points(raw_points)
        if not market_id or not points:
            continue
        rows.append(
            {
                "market_id": market_id,
                "updated_at": now,
                "points_json": points,
            }
        )
    if not rows:
        return 0
    batch_size = 25
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        insert_stmt = pg_insert(ScannerMarketHistory).values(chunk)
        await session.execute(
            insert_stmt.on_conflict_do_update(
                index_elements=[ScannerMarketHistory.market_id],
                set_={
                    "updated_at": insert_stmt.excluded.updated_at,
                    "points_json": insert_stmt.excluded.points_json,
                },
            )
        )
    return len(rows)


async def read_scanner_market_history(
    session: AsyncSession,
    *,
    market_ids: set[str] | list[str] | tuple[str, ...],
) -> dict[str, list[dict[str, Any]]]:
    """Read per-market history points for the given ``market_ids``.

    ``market_ids`` is required.  An unfiltered scan pulls hundreds of MB
    of ``points_json`` (47k rows × ~15 KB JSON), and asyncpg decodes
    every row's JSON synchronously on the event loop — a 20 s+ loop
    block that tears asyncpg cursors and saturates async session
    flushes.  Callers must scope to the markets they actually need.

    The JSON column is read as text (``cast(... AS text)``) so asyncpg
    skips its on-loop ``json.loads``; the actual parse happens in a
    worker thread via ``asyncio.to_thread``.
    """
    normalized_market_ids = sorted(
        {
            normalize_market_id(market_id)
            for market_id in (market_ids or [])
            if normalize_market_id(market_id)
        }
    )
    if not normalized_market_ids:
        # Empty filter ⇒ no rows.  Refusing to scan the entire table
        # protects the event loop from the 20 s+ block above.
        return {}

    points_text = cast(ScannerMarketHistory.points_json, Text).label("points_json_text")
    stmt = (
        select(ScannerMarketHistory.market_id, points_text)
        .where(_chunked_in(ScannerMarketHistory.market_id, normalized_market_ids))
    )
    result = await session.execute(stmt)
    raw_rows = [(raw_market_id, raw_points_text) for raw_market_id, raw_points_text in result.all()]

    import json as _json

    def _decode_rows() -> dict[str, list[dict[str, Any]]]:
        decoded: dict[str, list[dict[str, Any]]] = {}
        for raw_market_id, raw_points_text in raw_rows:
            market_id = normalize_market_id(raw_market_id)
            if not market_id or not raw_points_text:
                continue
            try:
                parsed = _json.loads(raw_points_text)
            except Exception:
                continue
            points = _normalize_history_points(parsed)
            if points:
                decoded[market_id] = points
        return decoded

    async with release_conn(session):
        return await asyncio.to_thread(_decode_rows)


def _normalize_weather_edge_title(title: str) -> str:
    prefix = "weather edge:"
    return title[len(prefix) :].lstrip() if title.lower().startswith(prefix) else title


def _serialize_opportunity_payload(opportunities: list[Opportunity]) -> tuple[list[dict[str, Any]], int]:
    payload: list[dict[str, Any]] = []
    skipped = 0
    for opportunity in opportunities:
        try:
            if hasattr(opportunity, "model_dump"):
                payload.append(opportunity.model_dump(mode="json"))
            else:
                payload.append(Opportunity.model_validate(opportunity).model_dump(mode="json"))
        except Exception as exc:
            skipped += 1
            logger.debug("Skip unserializable opportunity payload row: %s", exc)
    return payload, skipped


def _market_history_ids_from_payloads(payloads: list[dict[str, Any]]) -> set[str]:
    market_ids: set[str] = set()
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for market in payload.get("markets") or []:
            if not isinstance(market, dict):
                continue
            for raw_market_id in (
                market.get("id"),
                market.get("condition_id"),
                market.get("conditionId"),
            ):
                market_id = normalize_market_id(raw_market_id)
                if market_id:
                    market_ids.add(market_id)
    return market_ids


async def write_scanner_snapshot(
    session: AsyncSession,
    opportunities: list[Opportunity],
    status: dict[str, Any],
) -> None:
    """Write scanner status/counts and schedule normalized active-state projection."""
    await _apply_local_db_timeouts(
        session,
        statement_timeout_ms=_SCANNER_SNAPSHOT_WRITE_STATEMENT_TIMEOUT_MS,
    )
    last_scan = status.get("last_scan")
    if isinstance(last_scan, str):
        try:
            last_scan = parse_iso_datetime(last_scan, naive=True)
        except Exception as e:
            logger.warning("Invalid last_scan timestamp in snapshot status: %s", e)
            last_scan = utcnow()
    elif last_scan is None:
        last_scan = utcnow()

    result = await session.execute(select(ScannerSnapshot).where(ScannerSnapshot.id == SNAPSHOT_ID))
    row = result.scalar_one_or_none()
    if row is None:
        row = ScannerSnapshot(id=SNAPSHOT_ID)
        session.add(row)
    strategy_diagnostics = status.get("strategy_diagnostics")
    strategy_diagnostics = strategy_diagnostics if isinstance(strategy_diagnostics, dict) else {}
    raw_detected_count = 0
    execution_eligible_count = 0
    for diag in strategy_diagnostics.values():
        if not isinstance(diag, dict):
            continue
        raw_detected_count += int(diag.get("raw_detected_count") or 0)
        execution_eligible_count += int(diag.get("execution_eligible_count") or 0)
    displayable_count = len(opportunities)
    row.updated_at = utcnow()
    row.last_scan_at = last_scan
    row.opportunities_json = []
    row.raw_detected_count = int(raw_detected_count)
    row.displayable_count = int(displayable_count)
    row.execution_eligible_count = int(execution_eligible_count)
    row.opportunities_count = int(displayable_count)
    row.running = status.get("running", True)
    row.enabled = status.get("enabled", True)
    row.current_activity = status.get("current_activity")
    row.interval_seconds = status.get("interval_seconds", 60)
    row.strategies_json = status.get("strategies", [])
    row.strategy_diagnostics_json = status.get("strategy_diagnostics", {})
    row.tiered_scanning_json = status.get("tiered_scanning")
    row.ws_feeds_json = status.get("ws_feeds")
    await _commit_with_retry(session)
    _schedule_scanner_state_projection(opportunities=opportunities, status=status, completed_at=last_scan)

    async def _publish_snapshot_events() -> None:
        try:
            scanner_status = {
                "running": status.get("running", True),
                "enabled": status.get("enabled", True),
                "interval_seconds": status.get("interval_seconds", 60),
                "last_scan": status.get("last_scan"),
                "last_fast_scan": status.get("last_fast_scan"),
                "last_heavy_scan": status.get("last_heavy_scan"),
                "opportunities_count": row.opportunities_count,
                "current_activity": status.get("current_activity"),
                "lane_watchdogs": status.get("lane_watchdogs"),
                "strategies": status.get("strategies", []),
                "strategy_diagnostics": status.get("strategy_diagnostics", {}),
                "tiered_scanning": status.get("tiered_scanning"),
                "ws_feeds": status.get("ws_feeds"),
            }
            await event_bus.publish("scanner_status", scanner_status)
            await event_bus.publish(
                "scanner_activity",
                {"activity": status.get("current_activity") or "Idle"},
            )
            await event_bus.publish(
                "opportunities_update",
                {
                    "count": row.opportunities_count,
                    "source": "scanner_snapshot_write",
                },
            )
        except Exception:
            pass

    asyncio.create_task(_publish_snapshot_events())


# ---------------------------------------------------------------------------
# Market catalog persistence (events + markets from upstream APIs)
# ---------------------------------------------------------------------------

CATALOG_ID = "latest"
_MARKET_CATALOG_FILE_VERSION = 1
_MARKET_CATALOG_FILE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "cache" / "market_catalog_latest.json"
)


def _catalog_file_updated_at(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _write_market_catalog_file(
    events_payload: list[Any],
    markets_payload: list[Any],
    *,
    updated_at: datetime,
    duration_seconds: float,
    error: str | None,
) -> None:
    path = _MARKET_CATALOG_FILE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "version": _MARKET_CATALOG_FILE_VERSION,
                    "updated_at": _catalog_file_updated_at(updated_at),
                    "event_count": len(events_payload),
                    "market_count": len(markets_payload),
                    "fetch_duration_seconds": duration_seconds,
                    "error": error,
                    "events": events_payload,
                    "markets": markets_payload,
                },
                handle,
                ensure_ascii=True,
                separators=(",", ":"),
                default=str,
            )
        # Windows raises ``PermissionError`` (WinError 5) on
        # ``os.replace`` when ANY other process has an open handle on
        # the destination file — including a concurrent reader that's
        # mid-``json.load``.  The API + multiple workers all read this
        # catalog blob, so collisions are routine, not exceptional.
        # Retry with backoff; reader handles complete in milliseconds
        # so a few hundred ms is enough to win the race.
        last_exc: OSError | None = None
        for attempt in range(8):
            try:
                os.replace(tmp_path, path)
                last_exc = None
                break
            except PermissionError as exc:
                last_exc = exc
                time.sleep(0.05 * (1 << min(attempt, 4)))  # 50ms..800ms
        if last_exc is not None:
            raise last_exc
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _read_market_catalog_file() -> tuple[list[Any], list[Any], dict[str, Any]] | None:
    path = _MARKET_CATALOG_FILE_PATH
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    events_payload = payload.get("events")
    markets_payload = payload.get("markets")
    if not isinstance(events_payload, list) or not isinstance(markets_payload, list):
        return None
    try:
        event_count = int(payload.get("event_count") or len(events_payload))
        market_count = int(payload.get("market_count") or len(markets_payload))
    except (TypeError, ValueError):
        return None
    metadata = {
        "updated_at": parse_iso_datetime(str(payload.get("updated_at") or ""), naive=True)
        if payload.get("updated_at")
        else None,
        "event_count": event_count,
        "market_count": market_count,
        "fetch_duration_seconds": payload.get("fetch_duration_seconds"),
        "error": payload.get("error"),
    }
    return events_payload, markets_payload, metadata


def _catalog_file_matches_metadata(
    events_payload: list[Any],
    markets_payload: list[Any],
    file_metadata: dict[str, Any],
    metadata: dict[str, Any],
) -> bool:
    event_count = metadata.get("event_count")
    market_count = metadata.get("market_count")
    try:
        if event_count is not None and int(event_count) != len(events_payload):
            return False
        if market_count is not None and int(market_count) != len(markets_payload):
            return False
    except (TypeError, ValueError):
        return False
    db_updated_at = metadata.get("updated_at")
    if isinstance(db_updated_at, datetime):
        file_updated_at = file_metadata.get("updated_at")
        if not isinstance(file_updated_at, datetime):
            return False
        if file_updated_at.tzinfo is not None:
            file_updated_at = file_updated_at.astimezone(timezone.utc).replace(tzinfo=None)
        if db_updated_at.tzinfo is not None:
            db_updated_at = db_updated_at.astimezone(timezone.utc).replace(tzinfo=None)
        if abs((file_updated_at - db_updated_at).total_seconds()) > 0.001:
            return False
    return True


def _catalog_file_is_newer(file_metadata: dict[str, Any], db_metadata: dict[str, Any]) -> bool:
    file_updated_at = file_metadata.get("updated_at")
    db_updated_at = db_metadata.get("updated_at")
    if not isinstance(file_updated_at, datetime) or not isinstance(db_updated_at, datetime):
        return False
    if file_updated_at.tzinfo is not None:
        file_updated_at = file_updated_at.astimezone(timezone.utc).replace(tzinfo=None)
    if db_updated_at.tzinfo is not None:
        db_updated_at = db_updated_at.astimezone(timezone.utc).replace(tzinfo=None)
    return file_updated_at >= db_updated_at


def relink_event_markets(events: list, markets: list) -> None:
    if not events or not markets:
        return

    markets_by_slug: dict[str, list] = {}
    for market in markets:
        slug = str(getattr(market, "event_slug", "") or "").strip()
        if not slug:
            continue
        markets_by_slug.setdefault(slug, []).append(market)

    for event in events:
        key = str(getattr(event, "slug", "") or getattr(event, "id", "") or "").strip()
        if not key:
            continue
        event.markets = list(markets_by_slug.get(key, []))


async def write_market_catalog(
    session: AsyncSession,
    events: list,
    markets: list,
    duration_seconds: float = 0.0,
    error: str | None = None,
) -> None:
    """Persist catalog payload to the local cache file and scalar metadata to DB."""
    from models.database import MarketCatalog

    def _serialize_catalog_payloads() -> tuple[list[Any], list[Any]]:
        events_payload: list[Any] = []
        for event in events:
            try:
                if hasattr(event, "model_dump"):
                    payload = event.model_dump(mode="json", exclude={"markets"})
                elif isinstance(event, dict):
                    payload = dict(event)
                    payload.pop("markets", None)
                else:
                    payload = event
                events_payload.append(payload)
            except Exception:
                pass

        markets_payload: list[Any] = []
        for market in markets:
            try:
                markets_payload.append(market.model_dump(mode="json") if hasattr(market, "model_dump") else market)
            except Exception:
                pass
        return events_payload, markets_payload

    events_payload, markets_payload = await asyncio.to_thread(_serialize_catalog_payloads)
    updated_at = utcnow()
    should_replace_catalog = error is None or bool(events_payload) or bool(markets_payload)
    if should_replace_catalog:
        await asyncio.to_thread(
            _write_market_catalog_file,
            events_payload,
            markets_payload,
            updated_at=updated_at,
            duration_seconds=duration_seconds,
            error=error,
        )

    await _apply_local_db_timeouts(
        session,
        statement_timeout_ms=_MARKET_CATALOG_WRITE_STATEMENT_TIMEOUT_MS,
    )

    values = {
        "updated_at": updated_at,
        "error": error,
    }
    if should_replace_catalog:
        values.update(
            {
                "events_json": [],
                "markets_json": [],
                "event_count": len(events_payload),
                "market_count": len(markets_payload),
                "fetch_duration_seconds": duration_seconds,
            }
        )

    update_result = await session.execute(
        update(MarketCatalog).where(MarketCatalog.id == CATALOG_ID).values(**values)
    )
    if int(update_result.rowcount or 0) == 0:
        insert_values = {
            "events_json": [],
            "markets_json": [],
            "event_count": len(events_payload) if should_replace_catalog else 0,
            "market_count": len(markets_payload) if should_replace_catalog else 0,
            "fetch_duration_seconds": duration_seconds if should_replace_catalog else None,
            **values,
        }
        session.add(MarketCatalog(id=CATALOG_ID, **insert_values))
    await _commit_with_retry(session)
    # Drop the read-side cache so this process's next read sees the
    # write it just made.  Other processes pick it up via TTL.
    _market_catalog_cache_invalidate()


async def _fetch_market_catalog_metadata(session: AsyncSession) -> dict[str, Any]:
    """Read just the small scalar columns from market_catalog (no JSON)."""
    from models.database import MarketCatalog

    stmt = select(
        MarketCatalog.updated_at,
        MarketCatalog.event_count,
        MarketCatalog.market_count,
        MarketCatalog.fetch_duration_seconds,
        MarketCatalog.error,
    ).where(MarketCatalog.id == CATALOG_ID)
    row = (await session.execute(stmt)).one_or_none()
    try:
        if session.in_transaction():
            await session.rollback()
    except Exception:
        pass
    if row is None:
        return {"updated_at": None, "error": None}
    rm = row._mapping
    return {
        "updated_at": rm.get("updated_at"),
        "event_count": rm.get("event_count"),
        "market_count": rm.get("market_count"),
        "fetch_duration_seconds": rm.get("fetch_duration_seconds"),
        "error": rm.get("error"),
    }


async def _fetch_market_catalog_full(
    session: AsyncSession,
) -> tuple[str, str, dict[str, Any]]:
    """Fetch the catalog row with the heavy JSON columns as raw text.

    asyncpg decodes ``json`` columns by calling ``json.loads`` on the
    event loop the moment the row is received.  For the ``markets_json``
    blob (tens of MB) that is hundreds of ms of pure-Python parse on
    the hot path — long enough to starve WS heartbeats and tear
    asyncpg cursors mid-flight (``cannot switch to state X``).  Casting
    to ``text`` keeps the bytes opaque on transfer; the actual
    ``json.loads`` then runs in ``_deserialize_payload`` (already
    wrapped in ``asyncio.to_thread``).
    """
    from models.database import MarketCatalog

    events_text_col = cast(MarketCatalog.events_json, Text).label("events_json_text")
    markets_text_col = cast(MarketCatalog.markets_json, Text).label("markets_json_text")
    stmt = select(
        MarketCatalog.updated_at,
        MarketCatalog.event_count,
        MarketCatalog.market_count,
        MarketCatalog.fetch_duration_seconds,
        MarketCatalog.error,
        events_text_col,
        markets_text_col,
    ).where(MarketCatalog.id == CATALOG_ID)
    # The JSON columns can be tens of megabytes; extend the timeout
    # so this query isn't killed by the default global 60s.
    await session.execute(text("SET LOCAL statement_timeout = '120s'"))
    row = (await session.execute(stmt)).one_or_none()
    try:
        if session.in_transaction():
            await session.rollback()
    except Exception:
        pass
    if row is None:
        return "", "", {"updated_at": None, "error": None}
    rm = row._mapping
    events_text_value = rm.get("events_json_text") or ""
    markets_text_value = rm.get("markets_json_text") or ""
    metadata = {
        "updated_at": rm.get("updated_at"),
        "event_count": rm.get("event_count"),
        "market_count": rm.get("market_count"),
        "fetch_duration_seconds": rm.get("fetch_duration_seconds"),
        "error": rm.get("error"),
    }
    return events_text_value, markets_text_value, metadata


def _decode_catalog_text(events_text: str, markets_text: str) -> tuple[list[Any], list[Any]]:
    def _decode_text(text_value: str) -> list[Any]:
        if not text_value:
            return []
        try:
            decoded = json.loads(text_value)
        except Exception:
            return []
        return decoded if isinstance(decoded, list) else []

    return _decode_text(events_text), _decode_text(markets_text)


async def _fetch_market_catalog_payloads(
    session: AsyncSession,
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    metadata = await _fetch_market_catalog_metadata(session)
    metadata_has_catalog_row = "event_count" in metadata or "market_count" in metadata
    if metadata_has_catalog_row:
        async with release_conn(session):
            file_payload = await asyncio.to_thread(_read_market_catalog_file)
        if file_payload is not None:
            events_payload, markets_payload, file_metadata = file_payload
            metadata_matches = _catalog_file_matches_metadata(
                events_payload,
                markets_payload,
                file_metadata,
                metadata,
            )
            file_is_newer = _catalog_file_is_newer(file_metadata, metadata)
            if metadata_matches or file_is_newer:
                if metadata_matches:
                    merged_metadata = dict(file_metadata)
                    merged_metadata.update({key: value for key, value in metadata.items() if value is not None})
                else:
                    merged_metadata = dict(file_metadata)
                return events_payload, markets_payload, merged_metadata

    events_text, markets_text, metadata = await _fetch_market_catalog_full(session)
    async with release_conn(session):
        events_payload, markets_payload = await asyncio.to_thread(
            _decode_catalog_text,
            events_text,
            markets_text,
        )
    return events_payload, markets_payload, metadata


async def read_market_catalog(
    session: AsyncSession,
    *,
    include_events: bool = True,
    include_markets: bool = True,
    validate: bool = True,
) -> tuple[list, list, dict[str, Any]]:
    """Read persisted market catalog. Returns (events, markets, metadata).

    Backed by a process-local TTL cache (~5s) for the JSON-heavy paths so
    readers within a single scanner cycle share one payload decode. The
    metadata-only path bypasses the cache because it only reads small DB
    scalar columns and is already cheap.
    """
    from models.market import Event, Market

    # Metadata-only callers never touch the JSON columns; serve them
    # directly without consulting or warming the cache.
    if not include_events and not include_markets:
        metadata = await _fetch_market_catalog_metadata(session)
        return [], [], metadata

    cache = _market_catalog_cache
    loaded_at = cache.get("loaded_at_mono")
    fresh = (
        loaded_at is not None
        and (time.monotonic() - loaded_at) < _MARKET_CATALOG_CACHE_TTL_SECONDS
    )
    if not fresh:
        async with _market_catalog_cache_lock:
            # Another coroutine may have populated the cache while we
            # were waiting on the lock.
            loaded_at = cache.get("loaded_at_mono")
            fresh = (
                loaded_at is not None
                and (time.monotonic() - loaded_at) < _MARKET_CATALOG_CACHE_TTL_SECONDS
            )
            if not fresh:
                events_payload, markets_payload, metadata = await _fetch_market_catalog_payloads(session)
                cache["events_payload"] = events_payload
                cache["markets_payload"] = markets_payload
                cache["metadata"] = metadata
                cache["loaded_at_mono"] = time.monotonic()

    cached_events_payload: list[Any] = list(cache.get("events_payload") or []) if include_events else []
    cached_markets_payload: list[Any] = list(cache.get("markets_payload") or []) if include_markets else []
    metadata = dict(cache.get("metadata") or {"updated_at": None, "error": None})

    if not validate:
        return cached_events_payload, cached_markets_payload, metadata

    def _deserialize_payload() -> tuple[list, list]:
        events: list[Any] = []
        if include_events:
            for d in cached_events_payload:
                try:
                    events.append(Event.model_validate(d))
                except Exception:
                    pass
        markets: list[Any] = []
        if include_markets:
            for d in cached_markets_payload:
                try:
                    markets.append(Market.model_validate(d))
                except Exception:
                    pass
        return events, markets

    async with release_conn(session):
        events, markets = await asyncio.to_thread(_deserialize_payload)
    if include_events and include_markets and events and markets:
        relink_event_markets(events, markets)
    return events, markets, metadata


async def write_traders_snapshot(
    session: AsyncSession,
    opportunities: list[Opportunity],
    status: dict[str, Any],
) -> None:
    """Write trader opportunities into dedicated snapshot storage."""
    last_scan = status.get("last_scan")
    if isinstance(last_scan, str):
        try:
            last_scan = parse_iso_datetime(last_scan, naive=True)
        except Exception:
            last_scan = utcnow()
    elif last_scan is None:
        last_scan = utcnow()

    def _serialize_traders_payload() -> tuple[list[dict[str, Any]], int]:
        out: list[dict[str, Any]] = []
        skip = 0
        for o in opportunities:
            try:
                item = (
                    o.model_dump(mode="json")
                    if hasattr(o, "model_dump")
                    else Opportunity.model_validate(o).model_dump(mode="json")
                )
                if isinstance(item.get("strategy_context"), dict):
                    item["strategy_context"]["source_key"] = "traders"
                else:
                    item["strategy_context"] = {"source_key": "traders"}
                out.append(item)
            except Exception:
                skip += 1
        return out, skip

    payload, skipped = await asyncio.to_thread(_serialize_traders_payload)

    await _apply_local_db_timeouts(
        session,
        statement_timeout_ms=_TRADERS_SNAPSHOT_WRITE_STATEMENT_TIMEOUT_MS,
    )

    result = await session.execute(select(ScannerSnapshot).where(ScannerSnapshot.id == TRADERS_SNAPSHOT_ID))
    row = result.scalar_one_or_none()
    if row is None:
        row = ScannerSnapshot(id=TRADERS_SNAPSHOT_ID)
        session.add(row)

    row.updated_at = utcnow()
    row.last_scan_at = last_scan
    row.opportunities_json = payload
    row.running = status.get("running", True)
    row.enabled = status.get("enabled", True)
    row.current_activity = status.get("current_activity")
    row.interval_seconds = status.get("interval_seconds", 60)
    row.strategies_json = status.get("strategies", [])
    row.tiered_scanning_json = status.get("tiered_scanning")
    row.ws_feeds_json = status.get("ws_feeds")
    await _commit_with_retry(session)

    logger.info(
        "Wrote traders snapshot: opportunities=%s skipped=%s running=%s enabled=%s",
        len(payload),
        skipped,
        bool(row.running),
        bool(row.enabled),
    )
    try:
        await event_bus.publish(
            "opportunities_update",
            {
                "count": len(payload),
                "source": "traders_snapshot_write",
            },
        )
    except Exception:
        pass


async def _persist_incremental_state(
    session: AsyncSession,
    payload: list[dict[str, Any]],
    status: dict[str, Any],
    completed_at: datetime,
) -> list[dict[str, Any]]:
    """Persist per-run + per-opportunity incremental state/event records."""
    # Determine scan mode for observability.
    scan_mode = "full"
    activity = (status.get("current_activity") or "").lower()
    if "fast scan" in activity:
        scan_mode = "fast"
    elif "requested" in activity:
        scan_mode = "manual"

    run = ScannerRun(
        id=uuid.uuid4().hex[:16],
        scan_mode=scan_mode,
        success=not str(status.get("current_activity", "")).lower().startswith("last scan error"),
        opportunity_count=len(payload),
        started_at=completed_at,
        completed_at=completed_at,
    )
    session.add(run)
    event_messages: list[dict[str, Any]] = []

    # Build stable_id -> payload map; stable_id is the lifecycle key.
    current_map: dict[str, dict[str, Any]] = {}
    for item in payload:
        stable_id = str(item.get("stable_id") or item.get("id") or "").strip()
        if stable_id:
            current_map[stable_id] = item

    current_ids = set(current_map.keys())

    async def _load_rows_for_ids(stable_ids: set[str]) -> dict[str, OpportunityState]:
        if not stable_ids:
            return {}
        rows_by_id: dict[str, OpportunityState] = {}
        for stable_id_chunk in _chunked_values(stable_ids):
            existing_rows = (
                (
                    await session.execute(
                        select(OpportunityState).where(OpportunityState.stable_id.in_(stable_id_chunk))
                    )
                )
                .scalars()
                .all()
            )
            for row in existing_rows:
                stable_id = str(row.stable_id or "").strip()
                if stable_id:
                    rows_by_id[stable_id] = row
        return rows_by_id

    existing_by_id = await _load_rows_for_ids(current_ids)
    active_ids = {
        str(stable_id or "").strip()
        for stable_id in (
            (
                await session.execute(
                    select(OpportunityState.stable_id).where(OpportunityState.is_active == True)  # noqa: E712
                )
            )
            .scalars()
            .all()
        )
        if str(stable_id or "").strip()
    }

    # Upsert current opportunities and emit detected/updated/reactivated events.
    for stable_id, item in current_map.items():
        incoming_item = item if isinstance(item, dict) else dict(item)
        row = existing_by_id.get(stable_id)
        if row is None:
            incoming_item["revision"] = 1
            incoming_item["last_updated_at"] = format_iso_utc_z(completed_at)
            row = OpportunityState(
                stable_id=stable_id,
                opportunity_json=incoming_item,
                first_seen_at=completed_at,
                last_seen_at=completed_at,
                last_updated_at=completed_at,
                is_active=True,
                last_run_id=run.id,
            )
            session.add(row)
            event_messages.append(
                {
                    "id": uuid.uuid4().hex[:16],
                    "stable_id": stable_id,
                    "run_id": run.id,
                    "event_type": "detected",
                    "opportunity": incoming_item,
                    "created_at": format_iso_utc_z(completed_at),
                }
            )
            continue

        was_active = bool(row.is_active)
        previous_payload = row.opportunity_json if isinstance(row.opportunity_json, dict) else {}
        previous_revision = int(previous_payload.get("revision") or 0)
        changed = row.opportunity_json != incoming_item
        if changed:
            incoming_item["revision"] = max(1, previous_revision + 1)
        else:
            incoming_item["revision"] = max(1, previous_revision)

        previous_last_updated = previous_payload.get("last_updated_at")
        if changed or not was_active:
            incoming_item["last_updated_at"] = format_iso_utc_z(completed_at)
            row.last_updated_at = completed_at
        elif previous_last_updated:
            incoming_item["last_updated_at"] = previous_last_updated
        else:
            incoming_item["last_updated_at"] = format_iso_utc_z(completed_at)
            row.last_updated_at = completed_at

        # Lock-contention fix: only mutate the ORM row when something
        # genuinely changed.  The previous unconditional reassignments
        # (``row.opportunity_json = ...; row.last_seen_at = ...; ...``)
        # marked SQLAlchemy's instance as dirty for EVERY row in the
        # current scan, even when ``changed=False``.  With 1000+
        # opportunities per scan, the resulting commit emitted 1000+
        # row UPDATEs in a single transaction, holding 1000+ row locks
        # for the duration.  Concurrent transactions queued behind
        # them, observed in production as repeated:
        #    LOCK CONTENTION ... wait=Lock/transactionid query='UPDATE
        #    opportunity_state SET opportunity_json=$1::JSON, ...'
        # followed by ``Connection invalidated after 35.5s checked
        # out`` and ``Scanner opportunity-state projection failed``.
        # The expiration logic uses ``current_ids`` set membership,
        # not ``last_seen_at`` freshness, so skipping the UPDATE on
        # unchanged active rows is safe.
        if changed or not was_active:
            row.opportunity_json = incoming_item
            row.last_seen_at = completed_at
            row.last_run_id = run.id
            row.is_active = True

        if not was_active:
            event_type = "reactivated"
        elif changed:
            event_type = "updated"
        else:
            event_type = None

        if event_type:
            event_messages.append(
                {
                    "id": uuid.uuid4().hex[:16],
                    "stable_id": stable_id,
                    "run_id": run.id,
                    "event_type": event_type,
                    "opportunity": incoming_item,
                    "created_at": format_iso_utc_z(completed_at),
                }
            )

    # Any previously active row missing from current payload is now expired.
    expired_ids = active_ids - current_ids
    if expired_ids:
        expired_rows_by_id = await _load_rows_for_ids(expired_ids)
        for stable_id in expired_ids:
            row = expired_rows_by_id.get(stable_id)
            if row is None or not row.is_active:
                continue
            row.is_active = False
            row.last_seen_at = completed_at
            row.last_updated_at = completed_at
            row.last_run_id = run.id
            expired_payload = row.opportunity_json if isinstance(row.opportunity_json, dict) else {}
            event_messages.append(
                {
                    "id": uuid.uuid4().hex[:16],
                    "stable_id": stable_id,
                    "run_id": run.id,
                    "event_type": "expired",
                    "opportunity": expired_payload,
                    "created_at": format_iso_utc_z(completed_at),
                }
            )

    return event_messages


async def update_scanner_activity(session: AsyncSession, activity: str) -> None:
    """Update only current_activity in the snapshot (worker calls during scan for live status)."""
    await _apply_local_db_timeouts(
        session,
        statement_timeout_ms=_SCANNER_ACTIVITY_WRITE_STATEMENT_TIMEOUT_MS,
    )
    result = await session.execute(select(ScannerSnapshot).where(ScannerSnapshot.id == SNAPSHOT_ID))
    row = result.scalar_one_or_none()
    if row is None:
        row = ScannerSnapshot(
            id=SNAPSHOT_ID,
            current_activity=activity,
            running=True,
            enabled=True,
            interval_seconds=60,
            opportunities_json=[],
        )
        session.add(row)
    else:
        if row.current_activity == activity:
            return
        row.current_activity = activity
        row.updated_at = utcnow()
    await _commit_with_retry(session)

    # Publish activity change event.
    try:
        await event_bus.publish("scanner_activity", {"activity": activity})
    except Exception:
        pass  # fire-and-forget


async def read_active_opportunity_payloads(session: AsyncSession) -> list[dict[str, Any]]:
    """Read all active opportunity payloads as dicts.

    Casts ``opportunity_json`` to text in the SELECT so asyncpg does
    not ``json.loads`` each row on the event loop — a 224 MB table
    parsed on the loop is a multi-second loop block that tears asyncpg
    cursors and saturates async session flushes.  The actual JSON
    parse runs in ``asyncio.to_thread``.
    """
    payload_text_col = cast(OpportunityState.opportunity_json, Text).label("opportunity_json_text")
    stmt = (
        select(payload_text_col)
        .where(OpportunityState.is_active == True)  # noqa: E712
        .order_by(OpportunityState.last_updated_at.desc(), OpportunityState.stable_id.asc())
    )
    result = await session.execute(stmt)
    raw_texts = [text_value for text_value in result.scalars().all() if text_value]

    import json as _json

    def _decode_rows() -> list[dict[str, Any]]:
        decoded: list[dict[str, Any]] = []
        for text_value in raw_texts:
            try:
                obj = _json.loads(text_value)
            except Exception:
                continue
            if isinstance(obj, dict):
                decoded.append(obj)
        return decoded

    async with release_conn(session):
        return await asyncio.to_thread(_decode_rows)


# Back-compat alias for callers that imported the private name.
_read_active_market_opportunity_payloads = read_active_opportunity_payloads


async def read_scanner_snapshot(
    session: AsyncSession,
) -> tuple[list[Opportunity], dict[str, Any]]:
    """Read current status row plus normalized active market opportunities."""
    result = await session.execute(
        select(
            ScannerSnapshot.running,
            ScannerSnapshot.enabled,
            ScannerSnapshot.interval_seconds,
            ScannerSnapshot.last_scan_at,
            ScannerSnapshot.current_activity,
            ScannerSnapshot.strategies_json,
            ScannerSnapshot.strategy_diagnostics_json,
            ScannerSnapshot.tiered_scanning_json,
            ScannerSnapshot.ws_feeds_json,
            ScannerSnapshot.opportunities_count,
        ).where(ScannerSnapshot.id == SNAPSHOT_ID)
    )
    row = result.one_or_none()
    if row is None:
        return [], _default_status()

    raw_opps = await _read_active_market_opportunity_payloads(session)
    market_history = await read_scanner_market_history(
        session,
        market_ids=_market_history_ids_from_payloads(raw_opps),
    )
    try:
        if session.in_transaction():
            await session.rollback()
    except Exception:
        pass

    def _deserialize_opportunities() -> list[Opportunity]:
        out: list[Opportunity] = []
        for d in raw_opps:
            try:
                opp = Opportunity.model_validate(d)
                for market in opp.markets:
                    candidates = (
                        normalize_market_id(market.get("id", "")),
                        normalize_market_id(market.get("condition_id", "")),
                        normalize_market_id(market.get("conditionId", "")),
                    )
                    for candidate in candidates:
                        if not candidate:
                            continue
                        history = market_history.get(candidate)
                        if isinstance(history, list) and len(history) >= 2:
                            market["price_history"] = history
                            break
                out.append(opp)
            except Exception:
                pass
        return out

    async with release_conn(session):
        opportunities = await asyncio.to_thread(_deserialize_opportunities)
    opportunities.sort(key=lambda opp: float(getattr(opp, "roi_percent", 0.0) or 0.0), reverse=True)

    tiered = row.tiered_scanning_json if isinstance(row.tiered_scanning_json, dict) else {}
    status = {
        "running": row.running,
        "enabled": row.enabled,
        "interval_seconds": row.interval_seconds,
        "last_scan": format_iso_utc_z(row.last_scan_at),
        "last_fast_scan": tiered.get("last_fast_scan"),
        "last_heavy_scan": tiered.get("last_heavy_scan") or tiered.get("last_full_snapshot_strategy_scan"),
        "opportunities_count": len(opportunities),
        "current_activity": row.current_activity,
        "lane_watchdogs": tiered.get("lane_watchdogs"),
        "strategies": row.strategies_json or [],
        "strategy_diagnostics": row.strategy_diagnostics_json if isinstance(row.strategy_diagnostics_json, dict) else {},
        "tiered_scanning": row.tiered_scanning_json,
        "ws_feeds": row.ws_feeds_json,
    }
    return opportunities, status


async def read_scanner_status(
    session: AsyncSession,
    *,
    include_opportunity_count: bool = True,
    include_slo_metrics: bool = False,
) -> dict[str, Any]:
    """Read scanner status without deserializing opportunity payloads."""
    result = await session.execute(
        select(
            ScannerSnapshot.running,
            ScannerSnapshot.enabled,
            ScannerSnapshot.interval_seconds,
            ScannerSnapshot.last_scan_at,
            ScannerSnapshot.current_activity,
            ScannerSnapshot.strategies_json,
            ScannerSnapshot.strategy_diagnostics_json,
            ScannerSnapshot.tiered_scanning_json,
            ScannerSnapshot.ws_feeds_json,
            ScannerSnapshot.raw_detected_count,
            ScannerSnapshot.displayable_count,
            ScannerSnapshot.execution_eligible_count,
            ScannerSnapshot.opportunities_count,
        ).where(ScannerSnapshot.id == SNAPSHOT_ID)
    )
    row = result.one_or_none()
    if row is None:
        return _default_status()

    opportunities_count = 0
    if include_opportunity_count:
        opportunities_count = int(
            (
                await session.execute(
                    select(func.count())
                    .select_from(OpportunityState)
                    .where(OpportunityState.is_active == True)  # noqa: E712
                )
            ).scalar_one()
            or 0
        )

    tiered = row.tiered_scanning_json if isinstance(row.tiered_scanning_json, dict) else {}
    status = {
        "running": bool(row.running),
        "enabled": bool(row.enabled),
        "interval_seconds": int(row.interval_seconds or 60),
        "last_scan": format_iso_utc_z(row.last_scan_at),
        "last_fast_scan": tiered.get("last_fast_scan"),
        "last_heavy_scan": tiered.get("last_heavy_scan") or tiered.get("last_full_snapshot_strategy_scan"),
        "opportunities_count": opportunities_count,
        "current_activity": row.current_activity,
        "lane_watchdogs": tiered.get("lane_watchdogs"),
        "strategies": row.strategies_json or [],
        "strategy_diagnostics": row.strategy_diagnostics_json if isinstance(row.strategy_diagnostics_json, dict) else {},
        "tiered_scanning": row.tiered_scanning_json,
        "ws_feeds": row.ws_feeds_json,
    }
    if not include_slo_metrics:
        return status

    def _age_seconds(dt: datetime | None, now_dt: datetime) -> float | None:
        if dt is None:
            return None
        if dt.tzinfo is None:
            aware = dt.replace(tzinfo=timezone.utc)
        else:
            aware = dt.astimezone(timezone.utc)
        return max(0.0, (now_dt - aware).total_seconds())

    now_dt = utcnow()
    last_fast_scan_at = None
    raw_last_fast_scan = tiered.get("last_fast_scan")
    if isinstance(raw_last_fast_scan, str):
        try:
            parsed = parse_iso_datetime(raw_last_fast_scan, naive=True)
            last_fast_scan_at = parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
        except Exception:
            last_fast_scan_at = None
    status["last_fast_scan_age_seconds"] = _age_seconds(last_fast_scan_at, now_dt)

    now_naive = now_dt.astimezone(timezone.utc).replace(tzinfo=None)
    metrics_row = (
        await session.execute(
            select(
                func.percentile_cont(0.95).within_group(
                    func.extract(
                        "epoch",
                        now_naive - func.coalesce(OpportunityState.last_updated_at, OpportunityState.last_seen_at),
                    )
                ),
                func.percentile_cont(0.95).within_group(
                    func.extract("epoch", now_naive - OpportunityState.last_seen_at)
                ),
            ).where(OpportunityState.is_active == True)  # noqa: E712
        )
    ).one()
    price_age_p95 = metrics_row[0]
    detected_age_p95 = metrics_row[1]
    status["opportunity_price_age_p95"] = round(float(price_age_p95), 3) if price_age_p95 is not None else None
    status["opportunity_last_detected_age_p95"] = (
        round(float(detected_age_p95), 3) if detected_age_p95 is not None else None
    )
    status["coverage_ratio"] = tiered.get("full_snapshot_coverage_ratio")
    status["full_coverage_completion_time"] = tiered.get("full_coverage_completion_time")
    return status


async def read_traders_snapshot(
    session: AsyncSession,
) -> tuple[list[Opportunity], dict[str, Any]]:
    """Read latest trader opportunities and status from dedicated snapshot row."""
    result = await session.execute(
        select(
            ScannerSnapshot.running,
            ScannerSnapshot.enabled,
            ScannerSnapshot.interval_seconds,
            ScannerSnapshot.last_scan_at,
            ScannerSnapshot.current_activity,
            ScannerSnapshot.strategies_json,
            ScannerSnapshot.tiered_scanning_json,
            ScannerSnapshot.ws_feeds_json,
            ScannerSnapshot.opportunities_json,
        ).where(ScannerSnapshot.id == TRADERS_SNAPSHOT_ID)
    )
    row = result.one_or_none()
    if row is None:
        return [], _default_status()

    raw_opps = list(row.opportunities_json or [])
    market_history = await read_scanner_market_history(
        session,
        market_ids=_market_history_ids_from_payloads(raw_opps),
    )

    opportunities: list[Opportunity] = []
    for d in raw_opps:
        try:
            opp = Opportunity.model_validate(d)
            if isinstance(market_history, dict):
                for market in opp.markets:
                    if not isinstance(market, dict):
                        continue
                    candidates = {
                        normalize_market_id(market.get("id", "")),
                        normalize_market_id(market.get("condition_id", "")),
                    }
                    for candidate in candidates:
                        if not candidate:
                            continue
                        history = market_history.get(candidate)
                        if isinstance(history, list):
                            market["price_history"] = history
                            break
            if isinstance(opp.strategy_context, dict):
                opp.strategy_context["source_key"] = "traders"
            else:
                opp.strategy_context = {"source_key": "traders"}
            opportunities.append(opp)
        except Exception as e:
            logger.debug("Skip invalid trader opportunity row: %s", e)

    status = {
        "running": row.running,
        "enabled": row.enabled,
        "interval_seconds": row.interval_seconds,
        "last_scan": format_iso_utc_z(row.last_scan_at),
        "opportunities_count": len(opportunities),
        "current_activity": row.current_activity,
        "strategies": row.strategies_json or [],
        "tiered_scanning": row.tiered_scanning_json,
        "ws_feeds": row.ws_feeds_json,
    }
    return opportunities, status


def _default_status() -> dict[str, Any]:
    return {
        "running": False,
        "enabled": True,
        "interval_seconds": 60,
        "last_scan": None,
        "last_fast_scan": None,
        "last_heavy_scan": None,
        "opportunities_count": 0,
        "current_activity": "Waiting for scanner worker.",
        "lane_watchdogs": None,
        "strategies": [],
        "strategy_diagnostics": {},
        "tiered_scanning": None,
        "ws_feeds": None,
    }


def _stable_id_from_opportunity_id(opportunity_id: Optional[str]) -> Optional[str]:
    """Best-effort stable_id extraction from <stable_id>_<timestamp> IDs."""
    if not opportunity_id:
        return None
    text = str(opportunity_id).strip()
    if not text:
        return None
    parts = text.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return text


async def update_opportunity_ai_analysis_in_snapshot(
    session: AsyncSession,
    opportunity_id: str,
    stable_id: Optional[str],
    ai_analysis: dict[str, Any],
) -> bool:
    """Persist ai_analysis into opportunity_state for one opportunity."""
    sid = (stable_id or "").strip() or _stable_id_from_opportunity_id(opportunity_id)
    oid = (opportunity_id or "").strip()
    if not oid and not sid:
        return False

    updated = False

    if sid:
        state_row = await session.get(OpportunityState, sid)
        if state_row is not None and isinstance(state_row.opportunity_json, dict):
            patched_state = dict(state_row.opportunity_json)
            patched_state["ai_analysis"] = ai_analysis
            state_row.opportunity_json = patched_state
            state_row.last_seen_at = utcnow()
            updated = True

    if updated:
        await _commit_with_retry(session)
    return updated


async def get_opportunities_from_db(
    session: AsyncSession,
    filter: Optional[OpportunityFilter] = None,
    source: str = "markets",
) -> list[Opportunity]:
    """Get current opportunities from DB with optional filter (API use)."""
    source_key = str(source or "markets").strip().lower()
    if source_key == "traders":
        opportunities, _ = await read_traders_snapshot(session)
    elif source_key == "all":
        market_opps, _ = await read_scanner_snapshot(session)
        trader_opps, _ = await read_traders_snapshot(session)
        opportunities = list(market_opps) + list(trader_opps)
    else:
        opportunities, _ = await read_scanner_snapshot(session)

    # Release the DB connection before price overlays/history fetches,
    # which may perform network or cache I/O.
    try:
        if session is not None and hasattr(session, "in_transaction") and session.in_transaction():
            await session.rollback()
    except Exception:
        pass

    for opp in opportunities:
        opp.title = _normalize_weather_edge_title(opp.title)

    market_ids = sorted(
        {
            normalize_market_id(market_id)
            for opp in opportunities
            for market_id in [
                str((market or {}).get("id") or "").strip()
                for market in list(getattr(opp, "markets", []) or [])
                if isinstance(market, dict)
            ]
            if normalize_market_id(market_id)
        }
    )
    if market_ids:
        try:
            tradability = await get_market_tradability_map(market_ids)
        except Exception:
            tradability = {}
        if tradability:
            filtered_opportunities: list[Opportunity] = []
            for opp in opportunities:
                opp_market_ids = [
                    normalize_market_id(str((market or {}).get("id") or "").strip())
                    for market in list(getattr(opp, "markets", []) or [])
                    if isinstance(market, dict)
                ]
                if any(
                    market_id and tradability.get(market_id, True) is False
                    for market_id in opp_market_ids
                ):
                    continue
                filtered_opportunities.append(opp)
            opportunities = filtered_opportunities

    if opportunities:
        try:
            from services.scanner import scanner as market_scanner

            opportunities = await market_scanner.refresh_opportunity_prices(
                opportunities,
                drop_stale=False,
            )
        except Exception:
            pass

    if opportunities:
        history_candidates: list[Opportunity] = []
        seen_ids: set[str] = set()
        for opp in opportunities:
            has_missing_market_history = False
            for market in opp.markets:
                history = market.get("price_history")
                if not isinstance(history, list) or len(history) < 2:
                    has_missing_market_history = True
                    break
            if not has_missing_market_history:
                continue
            candidate_id = str(getattr(opp, "stable_id", "") or getattr(opp, "id", "") or "").strip()
            if candidate_id and candidate_id in seen_ids:
                continue
            if candidate_id:
                seen_ids.add(candidate_id)
            history_candidates.append(opp)

        if history_candidates:
            try:
                from services.scanner import scanner as market_scanner

                await market_scanner.attach_price_history_to_opportunities(
                    history_candidates,
                    timeout_seconds=0.0,
                )
            except Exception:
                pass

    if not filter:
        return opportunities
    if filter.min_profit > 0:
        opportunities = [o for o in opportunities if o.roi_percent >= filter.min_profit * 100]
    if filter.max_risk < 1.0:
        opportunities = [o for o in opportunities if o.risk_score <= filter.max_risk]
    if filter.strategies:
        opportunities = [o for o in opportunities if o.strategy in filter.strategies]
    if filter.min_liquidity > 0:
        opportunities = [o for o in opportunities if o.min_liquidity >= filter.min_liquidity]
    if filter.category:
        cl = filter.category.lower()
        opportunities = [o for o in opportunities if o.category and o.category.lower() == cl]
    return opportunities


async def get_scanner_status_from_db(session: AsyncSession) -> dict[str, Any]:
    """Get scanner status from DB (API use)."""
    return await read_scanner_status(session)


# ---------- Scanner control (API writes, worker reads) ----------


async def read_scanner_control(session: AsyncSession) -> dict[str, Any]:
    """Read scanner control row. Returns dict with is_enabled, is_paused, scan_interval_seconds, requested_scan_at."""
    result = await session.execute(select(ScannerControl).where(ScannerControl.id == CONTROL_ID))
    row = result.scalar_one_or_none()
    if row is None:
        return {
            "is_enabled": True,
            "is_paused": False,
            "scan_interval_seconds": 60,
            "requested_scan_at": None,
            "heavy_lane_forced_degraded": False,
            "heavy_lane_degraded_reason": None,
            "heavy_lane_degraded_until": None,
        }
    return {
        "is_enabled": row.is_enabled,
        "is_paused": row.is_paused,
        "scan_interval_seconds": row.scan_interval_seconds,
        "requested_scan_at": row.requested_scan_at,
        "heavy_lane_forced_degraded": bool(row.heavy_lane_forced_degraded),
        "heavy_lane_degraded_reason": row.heavy_lane_degraded_reason,
        "heavy_lane_degraded_until": row.heavy_lane_degraded_until,
    }


async def ensure_scanner_control(session: AsyncSession) -> ScannerControl:
    """Ensure scanner_control row exists; return it."""
    result = await session.execute(select(ScannerControl).where(ScannerControl.id == CONTROL_ID))
    row = result.scalar_one_or_none()
    if row is None:
        row = ScannerControl(id=CONTROL_ID)
        session.add(row)
        await _commit_with_retry(session)
        await session.refresh(row)
    return row


async def set_scanner_paused(session: AsyncSession, paused: bool) -> None:
    """Set scanner pause state (API: pause/resume)."""
    row = await ensure_scanner_control(session)
    row.is_paused = paused
    row.updated_at = utcnow()
    await _commit_with_retry(session)


async def set_scanner_interval(session: AsyncSession, interval_seconds: int) -> None:
    """Set scan interval (API)."""
    row = await ensure_scanner_control(session)
    row.scan_interval_seconds = max(10, min(3600, interval_seconds))
    row.updated_at = utcnow()
    await _commit_with_retry(session)


async def set_scanner_heavy_lane_degraded(
    session: AsyncSession,
    *,
    enabled: bool,
    reason: str | None = None,
    duration_seconds: int | None = None,
) -> None:
    row = await ensure_scanner_control(session)
    now = utcnow()
    row.heavy_lane_forced_degraded = bool(enabled)
    row.heavy_lane_degraded_reason = str(reason or "").strip()[:1000] or None
    if enabled and duration_seconds is not None and int(duration_seconds) > 0:
        row.heavy_lane_degraded_until = now + timedelta(seconds=max(1, int(duration_seconds)))
    elif enabled:
        row.heavy_lane_degraded_until = None
    else:
        row.heavy_lane_degraded_until = None
        row.heavy_lane_degraded_reason = None
    row.updated_at = now
    await _commit_with_retry(session)


async def clear_scanner_heavy_lane_degrade_if_expired(session: AsyncSession) -> bool:
    row = await ensure_scanner_control(session)
    if not bool(row.heavy_lane_forced_degraded):
        return False
    degraded_until = row.heavy_lane_degraded_until
    if degraded_until is None:
        return False
    now = utcnow()
    if degraded_until > now:
        return False
    row.heavy_lane_forced_degraded = False
    row.heavy_lane_degraded_until = None
    row.heavy_lane_degraded_reason = None
    row.updated_at = now
    await _commit_with_retry(session)
    return True


async def request_one_scan(
    session: AsyncSession,
    condition_ids: list[str] | None = None,
) -> None:
    """Set requested_scan_at so worker runs one scan on next loop (API: scan now).

    If *condition_ids* is provided, the scan will prioritise those markets
    instead of running a full untargeted scan.
    """
    global _pending_targeted_condition_ids
    row = await ensure_scanner_control(session)
    row.requested_scan_at = utcnow()
    if condition_ids:
        with _pending_targeted_condition_ids_lock:
            _pending_targeted_condition_ids = list(condition_ids)
    await _commit_with_retry(session)


def pop_targeted_condition_ids() -> list[str]:
    """Return and clear any pending targeted condition IDs for the next scan."""
    global _pending_targeted_condition_ids
    with _pending_targeted_condition_ids_lock:
        ids = list(_pending_targeted_condition_ids)
        _pending_targeted_condition_ids = []
    return ids


async def clear_scan_request(session: AsyncSession) -> None:
    """Clear requested_scan_at after worker has run (worker calls this)."""
    result = await session.execute(select(ScannerControl).where(ScannerControl.id == CONTROL_ID))
    row = result.scalar_one_or_none()
    if row and row.requested_scan_at is not None:
        row.requested_scan_at = None
        await _commit_with_retry(session)


async def upsert_scanner_slo_incident(
    session: AsyncSession,
    *,
    metric: str,
    breached: bool,
    observed_value: float | None,
    threshold_value: float | None,
    severity: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metric_key = str(metric or "").strip().lower()
    if not metric_key:
        return {"action": "noop"}

    now = utcnow()
    open_row = (
        (
            await session.execute(
                select(ScannerSloIncident).where(
                    ScannerSloIncident.metric == metric_key,
                    ScannerSloIncident.status == "open",
                )
            )
        )
        .scalars()
        .first()
    )

    observed = float(observed_value) if observed_value is not None else None
    threshold = float(threshold_value) if threshold_value is not None else None
    normalized_details = dict(details or {})
    normalized_severity = str(severity or "warning").strip().lower() or "warning"

    if breached:
        if open_row is None:
            row = ScannerSloIncident(
                id=uuid.uuid4().hex[:16],
                metric=metric_key,
                severity=normalized_severity,
                status="open",
                threshold_value=threshold,
                observed_value=observed,
                details_json=normalized_details,
                opened_at=now,
                last_seen_at=now,
                resolved_at=None,
            )
            session.add(row)
            await _commit_with_retry(session)
            return {"action": "opened", "incident_id": row.id, "metric": metric_key}

        open_row.severity = normalized_severity
        open_row.threshold_value = threshold
        open_row.observed_value = observed
        open_row.details_json = normalized_details
        open_row.last_seen_at = now
        await _commit_with_retry(session)
        return {"action": "updated", "incident_id": open_row.id, "metric": metric_key}

    if open_row is None:
        return {"action": "noop"}

    open_row.status = "resolved"
    open_row.resolved_at = now
    open_row.last_seen_at = now
    open_row.observed_value = observed
    open_row.threshold_value = threshold
    open_row.details_json = normalized_details
    await _commit_with_retry(session)
    return {"action": "resolved", "incident_id": open_row.id, "metric": metric_key}


async def list_open_scanner_slo_incidents(session: AsyncSession) -> list[dict[str, Any]]:
    rows = (
        (
            await session.execute(
                select(ScannerSloIncident)
                .where(ScannerSloIncident.status == "open")
                .order_by(ScannerSloIncident.opened_at.asc(), ScannerSloIncident.metric.asc())
            )
        )
        .scalars()
        .all()
    )
    return [
        {
            "id": str(row.id),
            "metric": str(row.metric),
            "severity": str(row.severity or "warning"),
            "status": str(row.status or "open"),
            "threshold_value": float(row.threshold_value) if row.threshold_value is not None else None,
            "observed_value": float(row.observed_value) if row.observed_value is not None else None,
            "details": dict(row.details_json or {}),
            "opened_at": format_iso_utc_z(row.opened_at),
            "last_seen_at": format_iso_utc_z(row.last_seen_at),
            "resolved_at": format_iso_utc_z(row.resolved_at),
        }
        for row in rows
    ]


async def clear_opportunities_in_snapshot(session: AsyncSession) -> int:
    """Clear opportunities in snapshot (API: clear all). Returns count cleared."""
    opportunities, status = await read_scanner_snapshot(session)
    count = len(opportunities)
    await write_scanner_snapshot(session, [], {**status, "opportunities_count": 0})
    return count


def _remove_expired_opportunities(
    opportunities: list[Opportunity],
) -> list[Opportunity]:
    """Drop opportunities whose resolution date has passed."""
    from datetime import timezone

    now = datetime.now(timezone.utc)
    out = []
    for o in opportunities:
        if o.resolution_date is None:
            out.append(o)
            continue
        rd = o.resolution_date if o.resolution_date.tzinfo else o.resolution_date.replace(tzinfo=timezone.utc)
        if rd > now:
            out.append(o)
    return out


def _remove_old_opportunities(
    opportunities: list[Opportunity],
    max_age_minutes: int,
) -> list[Opportunity]:
    """Drop opportunities older than max_age_minutes."""
    from datetime import timedelta, timezone

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)

    def ok(o: Opportunity) -> bool:
        d = o.last_detected_at or o.last_seen_at or o.detected_at
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d >= cutoff

    return [o for o in opportunities if ok(o)]


async def cleanup_snapshot_opportunities(
    session: AsyncSession,
    remove_expired: bool = True,
    max_age_minutes: Optional[int] = None,
) -> dict[str, int]:
    """Remove expired/old opportunities from snapshot; return counts."""
    opportunities, status = await read_scanner_snapshot(session)
    expired_removed, old_removed = 0, 0
    if remove_expired:
        before = len(opportunities)
        opportunities = _remove_expired_opportunities(opportunities)
        expired_removed = before - len(opportunities)
    if max_age_minutes:
        before = len(opportunities)
        opportunities = _remove_old_opportunities(opportunities, max_age_minutes)
        old_removed = before - len(opportunities)
    status["opportunities_count"] = len(opportunities)
    await write_scanner_snapshot(session, opportunities, status)
    return {"expired_removed": expired_removed, "old_removed": old_removed, "remaining_count": len(opportunities)}
