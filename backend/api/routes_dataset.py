"""Read-only dataset browser API.

Powers Research -> Data Lab in the UI and the ``query_dataset``
agent tool.  Exposes paginated, filtered reads of five core tables
through a single uniform shape so the UI can render any of them
with the same table component.

Datasets:
    book_snapshot            ── L2 book + trade snapshots (PARQUET-backed,
                                read per-token via the unified market-data
                                layer; no SQL table)
    opportunity_history      ── strategy detect() outputs
    trader_order             ── live + shadow order ledger
    backtest_run             ── persisted BacktestStudio runs

Per-dataset spec is declared in ``_DATASETS`` below.  SQL-backed specs set
``model`` + default sort + filters; parquet-backed specs set
``source='parquet'`` and read the canonical parquet plane through
``services.marketdata`` so the browser shows the exact point-in-time book the
backtester replays.  The generic routes branch on ``source`` so the frontend
renders either kind identically.

CSV export streams the same query results through ``StreamingResponse``
so a 100k-row download doesn't blow up Python heap.
"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel as _PydBaseModel, Field
from sqlalchemy import and_, bindparam, func, select, text

from models.database import (
    AsyncSessionLocal,
    BacktestRun,
    OpportunityHistory,
    ProviderDataset,
    TraderOrder,
)


logger = logging.getLogger("routes.dataset")
router = APIRouter(prefix="/dataset", tags=["Dataset"])


# ── Dataset specs ───────────────────────────────────────────────────────
#
# Each dataset declares: the ORM model, the default ORDER BY column
# (descending), the friendly label, the columns to expose, and which
# columns are filterable + how (eq / time-range / enum / contains).


@dataclass(frozen=True)
class ColumnSpec:
    key: str
    label: str
    type: str  # 'string' | 'int' | 'float' | 'datetime' | 'json' | 'enum'
    sortable: bool = True
    default_visible: bool = True
    enum_values: tuple[str, ...] | None = None
    description: str = ""


@dataclass(frozen=True)
class FilterSpec:
    key: str
    column: str  # ORM attribute name on the model
    label: str
    kind: str  # 'eq' | 'contains' | 'time_range_start' | 'time_range_end' | 'enum_in'
    description: str = ""


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    label: str
    description: str
    default_sort: str
    default_sort_dir: str  # 'asc' | 'desc'
    columns: tuple[ColumnSpec, ...]
    filters: tuple[FilterSpec, ...]
    # SQL-backed datasets set ``model``; parquet-backed datasets (book
    # snapshots read from the canonical parquet plane via the unified
    # market-data layer) set ``source='parquet'`` + ``parquet_kind`` and
    # leave ``model`` None. The generic browser routes on ``source`` so the
    # frontend renders either identically.
    model: type | None = None
    source: str = "sql"  # 'sql' | 'parquet'
    parquet_kind: str | None = None  # 'snapshots' when source == 'parquet'


_DATASETS: dict[str, DatasetSpec] = {
    "book_snapshot": DatasetSpec(
        name="book_snapshot",
        label="Book snapshots (parquet)",
        description=(
            "L2 book + trade-print snapshots read from the canonical parquet "
            "plane via the unified market-data layer (point-in-time). Select a "
            "token to browse its recorded book over a time window."
        ),
        source="parquet",
        parquet_kind="snapshots",
        default_sort="observed_at",
        default_sort_dir="desc",
        columns=(
            ColumnSpec("token_id", "Token", "string"),
            ColumnSpec("observed_at", "Observed", "datetime"),
            ColumnSpec("sequence", "Seq", "int", default_visible=False),
            ColumnSpec("best_bid", "Bid", "float"),
            ColumnSpec("best_ask", "Ask", "float"),
            ColumnSpec("spread_bps", "Spread (bps)", "float"),
            ColumnSpec("bids_json", "Bids (L2)", "json", sortable=False, default_visible=False),
            ColumnSpec("asks_json", "Asks (L2)", "json", sortable=False, default_visible=False),
            ColumnSpec("trade_price", "Trade px", "float"),
            ColumnSpec("trade_size", "Trade size", "float"),
            ColumnSpec("trade_side", "Trade side", "enum", enum_values=("BUY", "SELL")),
        ),
        filters=(
            FilterSpec("token_id", "token_id", "Token ID (required)", "eq"),
            FilterSpec("start", "observed_at", "From", "time_range_start"),
            FilterSpec("end", "observed_at", "To", "time_range_end"),
        ),
    ),
    "opportunity_history": DatasetSpec(
        name="opportunity_history",
        label="Opportunity history",
        description=(
            "Outputs of strategy.detect() across all loaded strategies. "
            "Powers the unified backtest engine's intent generator."
        ),
        model=OpportunityHistory,
        default_sort="detected_at",
        default_sort_dir="desc",
        columns=(
            ColumnSpec("id", "ID", "string", default_visible=False),
            ColumnSpec("strategy_type", "Strategy", "string"),
            ColumnSpec("event_id", "Event", "string"),
            ColumnSpec("title", "Title", "string"),
            ColumnSpec("total_cost", "Cost ($)", "float"),
            ColumnSpec("expected_roi", "Expected ROI", "float"),
            ColumnSpec("risk_score", "Risk", "float"),
            ColumnSpec("detected_at", "Detected", "datetime"),
            ColumnSpec("expired_at", "Expired", "datetime", default_visible=False),
            ColumnSpec("resolution_date", "Resolved", "datetime", default_visible=False),
            ColumnSpec("was_profitable", "Profitable", "string", default_visible=False),
            ColumnSpec("actual_roi", "Actual ROI", "float", default_visible=False),
            ColumnSpec("positions_data", "Positions", "json", sortable=False, default_visible=False),
        ),
        filters=(
            FilterSpec("strategy_type", "strategy_type", "Strategy", "eq"),
            FilterSpec("title_contains", "title", "Title contains", "contains"),
            FilterSpec("start", "detected_at", "From", "time_range_start"),
            FilterSpec("end", "detected_at", "To", "time_range_end"),
        ),
    ),
    "trader_order": DatasetSpec(
        name="trader_order",
        label="Trader orders",
        description=(
            "Live + shadow + paper order ledger.  Filter by mode, status, "
            "or strategy.  Source of realized PnL for the drift monitor."
        ),
        model=TraderOrder,
        default_sort="created_at",
        default_sort_dir="desc",
        columns=(
            ColumnSpec("id", "ID", "string", default_visible=False),
            ColumnSpec("trader_id", "Trader", "string"),
            ColumnSpec("strategy_key", "Strategy", "string"),
            ColumnSpec("mode", "Mode", "enum", enum_values=("live", "shadow", "paper")),
            ColumnSpec("status", "Status", "string"),
            ColumnSpec("market_question", "Market", "string"),
            ColumnSpec("direction", "Side", "string"),
            ColumnSpec("notional_usd", "Notional ($)", "float"),
            ColumnSpec("entry_price", "Entry px", "float"),
            ColumnSpec("effective_price", "Eff px", "float"),
            ColumnSpec("actual_profit", "PnL ($)", "float"),
            ColumnSpec("provider_order_id", "Provider id", "string", default_visible=False),
            ColumnSpec("source", "Source", "string", default_visible=False),
            ColumnSpec("created_at", "Created", "datetime"),
            ColumnSpec("executed_at", "Executed", "datetime"),
            ColumnSpec("updated_at", "Updated", "datetime", default_visible=False),
            ColumnSpec("payload_json", "Payload", "json", sortable=False, default_visible=False),
        ),
        filters=(
            FilterSpec("trader_id", "trader_id", "Trader ID", "eq"),
            FilterSpec("strategy_key", "strategy_key", "Strategy", "eq"),
            FilterSpec("mode", "mode", "Mode", "enum_in"),
            FilterSpec("status_in", "status", "Status", "enum_in"),
            FilterSpec("market_id", "market_id", "Market ID", "eq"),
            FilterSpec("start", "created_at", "From", "time_range_start"),
            FilterSpec("end", "created_at", "To", "time_range_end"),
        ),
    ),
    "provider_dataset": DatasetSpec(
        name="provider_dataset",
        label="Imported provider datasets",
        description=(
            "Catalog of historical datasets imported on demand from "
            "external vendors (polybacktest, etc.).  Snapshot rows live "
            "in microstructure_snapshot keyed by provider; this index "
            "powers the Backtest Studio dataset picker."
        ),
        model=ProviderDataset,
        default_sort="updated_at",
        default_sort_dir="desc",
        columns=(
            ColumnSpec("id", "ID", "string"),
            ColumnSpec("provider", "Provider", "string"),
            ColumnSpec("coin", "Coin", "string"),
            ColumnSpec("external_id", "External ID", "string"),
            ColumnSpec("external_slug", "Slug", "string", default_visible=False),
            ColumnSpec("title", "Title", "string"),
            ColumnSpec("asset_class", "Asset class", "string"),
            ColumnSpec("token_ids_json", "Token IDs", "json", sortable=False, default_visible=False),
            ColumnSpec("start_ts", "Window start", "datetime"),
            ColumnSpec("end_ts", "Window end", "datetime"),
            ColumnSpec("snapshot_count", "Snapshots", "int"),
            ColumnSpec("trade_count", "Trades", "int"),
            ColumnSpec("last_imported_at", "Last imported", "datetime"),
            ColumnSpec("last_import_job_id", "Last job", "string", default_visible=False),
            ColumnSpec("payload_json", "Payload", "json", sortable=False, default_visible=False),
            ColumnSpec("created_at", "Created", "datetime", default_visible=False),
            ColumnSpec("updated_at", "Updated", "datetime", default_visible=False),
        ),
        filters=(
            FilterSpec("provider", "provider", "Provider", "eq"),
            FilterSpec("coin", "coin", "Coin", "eq"),
            FilterSpec("title_contains", "title", "Title contains", "contains"),
            FilterSpec("start", "updated_at", "Updated from", "time_range_start"),
            FilterSpec("end", "updated_at", "Updated to", "time_range_end"),
        ),
    ),
    "backtest_run": DatasetSpec(
        name="backtest_run",
        label="Backtest runs",
        description=(
            "Persisted BacktestStudio runs.  Click a row in Data Lab to "
            "inspect the full augmented result_json payload."
        ),
        model=BacktestRun,
        default_sort="started_at",
        default_sort_dir="desc",
        columns=(
            ColumnSpec("id", "Run ID", "string"),
            ColumnSpec("strategy_slug", "Strategy", "string"),
            ColumnSpec("strategy_name", "Name", "string"),
            ColumnSpec("started_at", "Started", "datetime"),
            ColumnSpec("completed_at", "Completed", "datetime"),
            ColumnSpec("total_time_ms", "Time (ms)", "float"),
            ColumnSpec("status", "Status", "enum", enum_values=("ok", "failed")),
            ColumnSpec("trade_count", "Trades", "int"),
            ColumnSpec("total_return_pct", "Return %", "float"),
            ColumnSpec("sparkline_pct_json", "Sparkline", "json", sortable=False, default_visible=False),
            ColumnSpec("result_json", "Full result", "json", sortable=False, default_visible=False),
            ColumnSpec("created_at", "Created", "datetime", default_visible=False),
        ),
        filters=(
            FilterSpec("strategy_slug", "strategy_slug", "Strategy", "eq"),
            FilterSpec("status_in", "status", "Status", "enum_in"),
            FilterSpec("start", "started_at", "From", "time_range_start"),
            FilterSpec("end", "started_at", "To", "time_range_end"),
        ),
    ),
}


# ── Query construction ──────────────────────────────────────────────────


def _resolve_dataset(name: str) -> DatasetSpec:
    spec = _DATASETS.get(name)
    if spec is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown dataset '{name}'. Available: {sorted(_DATASETS)}",
        )
    return spec


def _parse_iso(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        v = str(value).strip()
        if not v:
            return None
        # Tolerate a trailing 'Z' (UTC) which fromisoformat doesn't accept
        # in older 3.10 stdlib; .replace covers both 3.10 and newer.
        v = v.replace("Z", "+00:00")
        dt = datetime.fromisoformat(v)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (TypeError, ValueError):
        return None


def _apply_filters(stmt, spec: DatasetSpec, params: dict[str, Any]):
    """Apply each filter that has a non-empty value in ``params``."""
    conds = []
    for f in spec.filters:
        raw = params.get(f.key)
        if raw is None or raw == "" or raw == []:
            continue
        col = getattr(spec.model, f.column, None)
        if col is None:
            continue
        if f.kind == "eq":
            conds.append(col == str(raw).strip())
        elif f.kind == "contains":
            text = str(raw).strip()
            if text:
                conds.append(col.ilike(f"%{text}%"))
        elif f.kind == "enum_in":
            values = raw if isinstance(raw, list) else [s for s in str(raw).split(",") if s]
            values = [str(v).strip() for v in values if str(v).strip()]
            if values:
                conds.append(col.in_(values))
        elif f.kind == "time_range_start":
            ts = _parse_iso(str(raw))
            if ts is not None:
                conds.append(col >= ts)
        elif f.kind == "time_range_end":
            ts = _parse_iso(str(raw))
            if ts is not None:
                conds.append(col <= ts)
    if conds:
        stmt = stmt.where(and_(*conds))
    return stmt


def _row_to_dict(row: Any, spec: DatasetSpec) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in spec.columns:
        v = getattr(row, col.key, None)
        if v is None:
            out[col.key] = None
            continue
        if col.type == "datetime":
            out[col.key] = v.isoformat() if hasattr(v, "isoformat") else str(v)
        elif col.type == "json":
            out[col.key] = v
        elif col.type in {"int", "float"}:
            try:
                out[col.key] = float(v) if col.type == "float" else int(v)
            except (TypeError, ValueError):
                out[col.key] = None
        else:
            out[col.key] = str(v)
    return out


def _serialize_columns(spec: DatasetSpec) -> list[dict[str, Any]]:
    return [
        {
            "key": c.key,
            "label": c.label,
            "type": c.type,
            "sortable": c.sortable,
            "default_visible": c.default_visible,
            "enum_values": list(c.enum_values) if c.enum_values else None,
            "description": c.description,
        }
        for c in spec.columns
    ]


def _serialize_filters(spec: DatasetSpec) -> list[dict[str, Any]]:
    return [
        {
            "key": f.key,
            "column": f.column,
            "label": f.label,
            "kind": f.kind,
            "description": f.description,
        }
        for f in spec.filters
    ]


# ── Parquet-backed query path ─────────────────────────────────────────────
#
# Book snapshots live in the canonical parquet plane (no SQL table). The
# parquet datasets read through the unified market-data layer so the browser
# shows the same point-in-time book the backtester replays. A token_id filter
# is required (you browse one token's recorded book over a window) — listing
# every token's rows would mean scanning every parquet file.


def _book_snapshot_to_row(token_id: str, ts_us: int, snap: Any) -> dict[str, Any]:
    obs = datetime.fromtimestamp(ts_us / 1_000_000, tz=timezone.utc)
    return {
        "token_id": token_id,
        "observed_at": obs.isoformat(),
        "sequence": snap.sequence,
        "best_bid": snap.bids[0].price if snap.bids else None,
        "best_ask": snap.asks[0].price if snap.asks else None,
        "spread_bps": snap.spread_bps,
        "bids_json": [[lvl.price, lvl.size] for lvl in snap.bids[:15]],
        "asks_json": [[lvl.price, lvl.size] for lvl in snap.asks[:15]],
        "trade_price": snap.trade_price,
        "trade_size": snap.trade_size,
        "trade_side": snap.trade_side,
    }


async def _query_parquet_book(
    spec: DatasetSpec,
    params: dict[str, Any],
    *,
    limit: int,
    offset: int,
    order_dir: str,
) -> dict[str, Any]:
    """Page a single token's canonical book snapshots from parquet.

    Returns ``{"rows": [...], "total": int, "note": str|None}`` — same row
    dicts the SQL path produces, so the frontend renders them identically.
    """
    token_id = str(params.get("token_id") or "").strip()
    if not token_id:
        return {"rows": [], "total": 0, "note": "Select a token_id to browse parquet book snapshots."}
    start = _parse_iso(params.get("start")) or datetime(2000, 1, 1, tzinfo=timezone.utc)
    end = _parse_iso(params.get("end")) or datetime.now(timezone.utc)

    from services.marketdata.book import load_book_series, us_from_dt
    from services.marketdata.coverage import resolve_coverage

    cov = await resolve_coverage(token_ids=[token_id], start=start, end=end)
    files = cov.files_for(token_id)
    if not files:
        return {"rows": [], "total": 0, "note": "No parquet coverage for this token in the window."}
    series, _n = load_book_series(token_id, files, start_us=us_from_dt(start), end_us=us_from_dt(end))
    entries = list(series.iter_range(us_from_dt(start), us_from_dt(end)))  # ascending (ts_us, snap)
    if order_dir == "desc":
        entries.reverse()
    total = len(entries)
    page = entries[int(offset): int(offset) + int(limit)]
    rows = [_book_snapshot_to_row(token_id, ts, snap) for ts, snap in page]
    return {"rows": rows, "total": total, "note": None}


# ── Routes ──────────────────────────────────────────────────────────────


# Tables small enough that an exact COUNT(*) is cheap are counted live so the
# UI badge increases on every refresh. Anything past this threshold (per the
# planner estimate) falls back to the estimate to avoid a multi-million-row
# seq-scan tripping statement_timeout. The live count is itself wrapped in a
# short SET LOCAL statement_timeout so a surprise large table can't stall the
# whole listing.
_LIVE_COUNT_MAX_ESTIMATE = 2_000_000
_LIVE_COUNT_TIMEOUT_MS = 1500


async def _live_row_count(table_name: str) -> int | None:
    """Exact COUNT(*) for one table, guarded by a per-statement timeout.

    Returns the count on success, or ``None`` if it timed out / errored so the
    caller can fall back to the planner estimate. Runs in its own transaction
    that is always rolled back, so a timeout never poisons later queries.
    """
    try:
        async with AsyncSessionLocal() as session:
            try:
                await session.execute(
                    text(f"SET LOCAL statement_timeout = {int(_LIVE_COUNT_TIMEOUT_MS)}")
                )
                # ident already validated: table_name comes from ORM metadata.
                result = await session.execute(
                    text(f'SELECT COUNT(*) AS n FROM "{table_name}"')
                )
                return int(result.scalar_one())
            finally:
                await session.rollback()
    except Exception:
        # Timeout (QueryCanceled) or any other error -> signal fallback.
        return None


@router.get("")
async def list_datasets() -> dict[str, Any]:
    """List every available dataset with its row count and metadata.

    ``row_count`` is an exact ``COUNT(*)`` for tables whose planner estimate is
    under ``_LIVE_COUNT_MAX_ESTIMATE`` (so actively-written datasets' counts
    increase on refresh), falling back to the planner estimate from
    ``pg_class.reltuples`` / ``pg_stat_user_tables.n_live_tup`` for very large
    tables (e.g. wallet_monitor_events) where an exact count would
    seq-scan the heap and trip the statement_timeout.
    """
    table_names = [
        spec.model.__tablename__
        for spec in _DATASETS.values()
        if spec.source == "sql" and spec.model is not None
    ]
    estimates: dict[str, int] = {}
    async with AsyncSessionLocal() as session:
        try:
            stmt = text(
                """
                SELECT c.relname,
                       GREATEST(
                         COALESCE(s.n_live_tup, 0),
                         COALESCE(NULLIF(c.reltuples, -1)::bigint, 0)
                       )::bigint AS row_estimate
                FROM pg_class c
                LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
                WHERE c.relkind = 'r'
                  AND c.relname IN :names
                """
            ).bindparams(bindparam("names", expanding=True))
            rows = (await session.execute(stmt, {"names": table_names})).all()
            for r in rows:
                estimates[r.relname] = int(r.row_estimate or 0)
        except Exception:
            logger.exception("dataset.list: failed to read row estimates")

    # Live-count the small/medium tables concurrently; fall back to estimate.
    live_targets = [
        t for t in table_names
        if estimates.get(t, 0) <= _LIVE_COUNT_MAX_ESTIMATE
    ]
    live_counts: dict[str, int] = {}
    if live_targets:
        results = await asyncio.gather(
            *(_live_row_count(t) for t in live_targets),
            return_exceptions=True,
        )
        for table_name, res in zip(live_targets, results):
            if isinstance(res, int):
                live_counts[table_name] = res

    out: list[dict[str, Any]] = []
    for spec in _DATASETS.values():
        if spec.source == "parquet" or spec.model is None:
            # Parquet datasets are browsed per-token; a global row count would
            # mean scanning every file, so we don't surface one.
            row_count, row_count_exact = None, False
        else:
            table_name = spec.model.__tablename__
            row_count = live_counts.get(table_name, estimates.get(table_name, 0))
            row_count_exact = table_name in live_counts
        out.append({
            "name": spec.name,
            "label": spec.label,
            "description": spec.description,
            "row_count": row_count,
            "row_count_exact": row_count_exact,
            "source": spec.source,
            "default_sort": spec.default_sort,
            "default_sort_dir": spec.default_sort_dir,
            "columns": _serialize_columns(spec),
            "filters": _serialize_filters(spec),
        })
    return {"datasets": out}


@router.get("/{name}")
async def query_dataset(
    name: str,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    order_by: str | None = None,
    order_dir: str = Query(default="desc", pattern="^(asc|desc)$"),
    # Filter params — passthrough; query() picks them out by spec.
    token_id: str | None = None,
    strategy_type: str | None = None,
    strategy_key: str | None = None,
    strategy_slug: str | None = None,
    snapshot_type: str | None = None,
    event_type: str | None = None,
    side: str | None = None,
    mode: str | None = None,
    status_in: str | None = None,
    trader_id: str | None = None,
    market_id: str | None = None,
    title_contains: str | None = None,
    provider: str | None = None,
    coin: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    """Paginated, filtered read of one dataset.

    Returns a uniform shape:
        {
            "dataset": str,
            "total": int,
            "limit": int, "offset": int,
            "order_by": str, "order_dir": str,
            "columns": [...],
            "rows": [{...row dict keyed by column.key...}],
        }
    """
    spec = _resolve_dataset(name)
    sort_col_name = order_by or spec.default_sort
    params = {
        "token_id": token_id,
        "strategy_type": strategy_type,
        "strategy_key": strategy_key,
        "strategy_slug": strategy_slug,
        "snapshot_type": snapshot_type,
        "event_type": event_type,
        "side": side,
        "mode": mode,
        "status_in": status_in,
        "trader_id": trader_id,
        "market_id": market_id,
        "title_contains": title_contains,
        "provider": provider,
        "coin": coin,
        "start": start,
        "end": end,
    }

    # Parquet-backed datasets read through the unified market-data layer.
    if spec.source == "parquet":
        result = await _query_parquet_book(
            spec, params, limit=limit, offset=offset, order_dir=order_dir
        )
        out = {
            "dataset": spec.name,
            "label": spec.label,
            "total": int(result["total"]),
            "limit": int(limit),
            "offset": int(offset),
            "order_by": sort_col_name,
            "order_dir": order_dir,
            "columns": _serialize_columns(spec),
            "filters": _serialize_filters(spec),
            "rows": result["rows"],
        }
        if result.get("note"):
            out["note"] = result["note"]
        return out

    sort_attr = getattr(spec.model, sort_col_name, None)
    if sort_attr is None:
        raise HTTPException(status_code=400, detail=f"Unknown column '{sort_col_name}'")

    async with AsyncSessionLocal() as session:
        # COUNT — applied with the same filters
        count_stmt = _apply_filters(select(func.count(spec.model.id)), spec, params)
        try:
            total = (await session.execute(count_stmt)).scalar_one()
        except Exception as exc:
            logger.exception("Dataset count failed")
            raise HTTPException(status_code=500, detail=f"count failed: {exc}") from exc

        # PAGE
        stmt = select(spec.model)
        stmt = _apply_filters(stmt, spec, params)
        stmt = stmt.order_by(sort_attr.desc() if order_dir == "desc" else sort_attr.asc())
        stmt = stmt.limit(int(limit)).offset(int(offset))
        try:
            rows = (await session.execute(stmt)).scalars().all()
        except Exception as exc:
            logger.exception("Dataset query failed")
            raise HTTPException(status_code=500, detail=f"query failed: {exc}") from exc

    return {
        "dataset": spec.name,
        "label": spec.label,
        "total": int(total or 0),
        "limit": int(limit),
        "offset": int(offset),
        "order_by": sort_col_name,
        "order_dir": order_dir,
        "columns": _serialize_columns(spec),
        "filters": _serialize_filters(spec),
        "rows": [_row_to_dict(r, spec) for r in rows],
    }


@router.get("/quality/book-snapshot")
async def book_snapshot_quality(
    token_id: str,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    """Parquet-native data-quality report for one token's recorded book.

    Reads the SAME canonical parquet the backtester replays (coverage +
    crossed-book + gaps + staleness), so the report reflects exactly what a
    backtest would see. (Two-segment path so it never collides with the
    ``/{name}`` dataset route.)
    """
    tok = (token_id or "").strip()
    if not tok:
        raise HTTPException(status_code=400, detail="token_id is required")
    from services.marketdata.quality import assess_book_quality

    s = _parse_iso(start) or datetime(2000, 1, 1, tzinfo=timezone.utc)
    e = _parse_iso(end) or datetime.now(timezone.utc)
    return await assess_book_quality(token_id=tok, start=s, end=e)


@router.get("/{name}/csv")
async def export_dataset_csv(
    name: str,
    columns: str | None = None,  # comma-separated subset
    order_by: str | None = None,
    order_dir: str = Query(default="desc", pattern="^(asc|desc)$"),
    max_rows: int = Query(default=50_000, ge=1, le=500_000),
    token_id: str | None = None,
    strategy_type: str | None = None,
    strategy_key: str | None = None,
    strategy_slug: str | None = None,
    snapshot_type: str | None = None,
    event_type: str | None = None,
    side: str | None = None,
    mode: str | None = None,
    status_in: str | None = None,
    trader_id: str | None = None,
    market_id: str | None = None,
    title_contains: str | None = None,
    provider: str | None = None,
    coin: str | None = None,
    start: str | None = None,
    end: str | None = None,
):
    """Stream a CSV of the filtered query.

    Caps at ``max_rows`` (default 50k, 500k absolute max).  JSON
    columns are serialized inline as JSON strings — Excel-friendly.
    """
    spec = _resolve_dataset(name)
    if spec.source == "parquet":
        # Parquet datasets are browsed per-token via the table view; CSV
        # streaming over the canonical parquet plane isn't wired here.
        raise HTTPException(
            status_code=400,
            detail="CSV export is not available for parquet-backed datasets; use the table view.",
        )
    sort_col_name = order_by or spec.default_sort
    sort_attr = getattr(spec.model, sort_col_name, None)
    if sort_attr is None:
        raise HTTPException(status_code=400, detail=f"Unknown column '{sort_col_name}'")

    params = {
        "token_id": token_id,
        "strategy_type": strategy_type,
        "strategy_key": strategy_key,
        "strategy_slug": strategy_slug,
        "snapshot_type": snapshot_type,
        "event_type": event_type,
        "side": side,
        "mode": mode,
        "status_in": status_in,
        "trader_id": trader_id,
        "market_id": market_id,
        "title_contains": title_contains,
        "provider": provider,
        "coin": coin,
        "start": start,
        "end": end,
    }

    chosen_keys: list[str]
    if columns:
        wanted = {c.strip() for c in columns.split(",") if c.strip()}
        chosen_keys = [c.key for c in spec.columns if c.key in wanted]
    else:
        chosen_keys = [c.key for c in spec.columns if c.default_visible]
    if not chosen_keys:
        chosen_keys = [c.key for c in spec.columns]

    async def _generate() -> Iterable[str]:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(chosen_keys)
        yield buf.getvalue()
        buf.seek(0)
        buf.truncate(0)

        async with AsyncSessionLocal() as session:
            stmt = select(spec.model)
            stmt = _apply_filters(stmt, spec, params)
            stmt = stmt.order_by(sort_attr.desc() if order_dir == "desc" else sort_attr.asc())
            stmt = stmt.limit(int(max_rows))
            # Execute in one go (reasonable for max_rows<=500k).
            result = await session.execute(stmt)
            for row in result.scalars():
                values = []
                for k in chosen_keys:
                    v = getattr(row, k, None)
                    if v is None:
                        values.append("")
                    elif hasattr(v, "isoformat"):
                        values.append(v.isoformat())
                    elif isinstance(v, (dict, list)):
                        values.append(json.dumps(v, default=str, separators=(",", ":")))
                    else:
                        values.append(str(v))
                writer.writerow(values)
                yield buf.getvalue()
                buf.seek(0)
                buf.truncate(0)

    filename = f"{spec.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        _generate(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Recording sessions (on-demand captures) ────────────────────────────


class RecordingSessionCreate(_PydBaseModel):
    name: str
    description: str | None = None
    platform: str = "polymarket"
    target_kind: str = "token"  # token | condition | event
    target_values: list[str]
    capture_types: list[str] | None = None  # subset of book/trade/delta
    tick_interval_ms: int = 500
    retention_days: int | None = None
    scheduled_start_at: datetime | None = None
    scheduled_end_at: datetime | None = None
    max_duration_seconds: int | None = None
    config: dict[str, Any] | None = None


def _serialize_session(row: Any) -> dict[str, Any]:
    return {
        "id": row.id,
        "name": row.name,
        "description": row.description,
        "status": row.status,
        "platform": row.platform,
        "target_kind": row.target_kind,
        "target_values": list(row.target_values_json or []),
        "target_token_ids": list(row.target_token_ids_json or []),
        "capture_types": list(row.capture_types_json or []),
        "tick_interval_ms": row.tick_interval_ms,
        "retention_days": row.retention_days,
        "scheduled_start_at": row.scheduled_start_at.isoformat() if row.scheduled_start_at else None,
        "scheduled_end_at": row.scheduled_end_at.isoformat() if row.scheduled_end_at else None,
        "max_duration_seconds": row.max_duration_seconds,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "ended_at": row.ended_at.isoformat() if row.ended_at else None,
        "rows_captured": int(row.rows_captured or 0),
        "last_capture_at": row.last_capture_at.isoformat() if row.last_capture_at else None,
        "error": row.error,
        "config": row.config_json,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


@router.get("/sessions")
async def list_recording_sessions(
    statuses: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    """List recording sessions, optionally filtered by status (csv)."""
    from services.recording_session_service import list_sessions

    parsed = [s.strip() for s in (statuses or "").split(",") if s.strip()] or None
    rows = await list_sessions(statuses=parsed, limit=limit)
    return {"sessions": [_serialize_session(r) for r in rows]}


@router.post("/sessions")
async def create_recording_session(body: RecordingSessionCreate) -> dict[str, Any]:
    """Create a new on-demand recording session.

    If ``scheduled_start_at`` is set the session lands in ``scheduled``
    and the manager loop will activate it at that time.  Otherwise the
    session lands in ``pending`` — call POST /sessions/{id}/start to
    activate immediately.
    """
    from services.recording_session_service import SessionSpec, create_session

    spec = SessionSpec(
        name=body.name,
        description=body.description,
        platform=body.platform,
        target_kind=body.target_kind,
        target_values=body.target_values,
        capture_types=body.capture_types,
        tick_interval_ms=body.tick_interval_ms,
        retention_days=body.retention_days,
        scheduled_start_at=body.scheduled_start_at,
        scheduled_end_at=body.scheduled_end_at,
        max_duration_seconds=body.max_duration_seconds,
        config=body.config,
    )
    row = await create_session(spec)
    return _serialize_session(row)


@router.get("/sessions/{session_id}")
async def get_recording_session(session_id: str) -> dict[str, Any]:
    from services.recording_session_service import get_session

    row = await get_session(session_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return _serialize_session(row)


@router.post("/sessions/{session_id}/start")
async def start_recording_session(session_id: str) -> dict[str, Any]:
    from services.recording_session_service import start_session

    try:
        row = await start_session(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _serialize_session(row)


@router.post("/sessions/{session_id}/stop")
async def stop_recording_session(session_id: str) -> dict[str, Any]:
    from services.recording_session_service import stop_session

    try:
        row = await stop_session(session_id, status="completed")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _serialize_session(row)


@router.post("/sessions/{session_id}/cancel")
async def cancel_recording_session(session_id: str) -> dict[str, Any]:
    from services.recording_session_service import stop_session

    try:
        row = await stop_session(session_id, status="cancelled")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _serialize_session(row)


@router.delete("/sessions/{session_id}")
async def delete_recording_session(session_id: str) -> dict[str, Any]:
    from services.recording_session_service import delete_session

    deleted = await delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return {"deleted": True, "id": session_id}




def _scan_recent_parquet_actuals(minutes: int) -> dict[str, Any]:
    """Sync scan of the parquet data plane for recent recording activity.

    Recording moved off Postgres onto parquet, so counting SQL rows always
    returns 0 now.  Instead we walk every configured parquet root, find
    snapshot/delta files written within the last ``minutes`` (cheap mtime
    stat), and sum row counts straight from the parquet footers
    (``read_metadata().num_rows`` — no data-page scan).  This is the
    cross-process truth: it reflects whatever the trading worker is writing,
    regardless of which process serves this endpoint.
    """
    import time as _time

    import pyarrow.parquet as _pq

    from services.external_data.parquet_schema import parquet_roots

    cutoff = _time.time() - max(1, int(minutes)) * 60
    book_rows = 0
    delta_rows = 0
    tokens: set[str] = set()
    newest_mtime = 0.0
    providers: set[str] = set()

    seen_roots: set[str] = set()
    for root in parquet_roots():
        key = str(root)
        if key in seen_roots or not root.exists():
            continue
        seen_roots.add(key)
        for fp in root.rglob("*.parquet"):
            name = fp.name
            if name.startswith("snapshots__"):
                kind = "book"
            elif name.startswith("deltas__"):
                kind = "delta"
            else:
                continue
            try:
                mt = fp.stat().st_mtime
            except OSError:
                continue
            if mt < cutoff:
                continue
            try:
                n = int(_pq.read_metadata(str(fp)).num_rows)
            except Exception:
                continue
            if kind == "book":
                book_rows += n
            else:
                delta_rows += n
            tokens.add(name.split("__", 1)[1].rsplit(".", 1)[0])
            newest_mtime = max(newest_mtime, mt)
            # provider = first path segment under the root
            try:
                providers.add(fp.relative_to(root).parts[0])
            except Exception:
                pass

    span = max(int(minutes) * 60, 1)
    return {
        "window_minutes": int(minutes),
        "distinct_tokens": len(tokens),
        "book_rows": book_rows,
        # In the parquet model trade prints are delta events (event_type=
        # 'trade'); ``trade_rows`` carries the delta-row count (trades+cancels).
        "trade_rows": delta_rows,
        "book_rows_per_sec": round(book_rows / span, 3),
        "actively_recording": (book_rows > 0 or delta_rows > 0),
        "newest_age_seconds": (round(_time.time() - newest_mtime, 1) if newest_mtime else None),
        "providers": sorted(providers),
        "source": "parquet",
        "note": "parquet-derived (cross-process); per-process counters above reflect only the API process.",
    }


async def _recent_ingestion_actuals(minutes: int = 10) -> dict[str, Any]:
    """Cross-process recording truth, derived from the parquet data plane.

    See ``_scan_recent_parquet_actuals``.  Runs the (filesystem) scan off the
    event loop so a large parquet tree can't block the API.
    """
    try:
        return await asyncio.to_thread(_scan_recent_parquet_actuals, minutes)
    except Exception as exc:
        return {"error": str(exc), "actively_recording": None}


class _RecordingToggle(_PydBaseModel):
    enabled: bool = Field(..., description="Global recording master switch")


@router.get("/recorder/recording")
async def get_recording_state() -> dict[str, Any]:
    """Current state of the global recording master switch + live activity.

    ``enabled`` is the persisted ``app_settings.recording_enabled`` flag;
    ``actual_recording`` is the parquet-derived truth of what's being
    written right now, so the UI can show both intent and reality.
    """
    from services.recording_control import get_recording_enabled

    enabled = await get_recording_enabled()
    return {
        "enabled": enabled,
        "actual_recording": await _recent_ingestion_actuals(),
    }


@router.put("/recorder/recording")
async def set_recording_state(req: _RecordingToggle) -> dict[str, Any]:
    """Turn ALL market-data recording on/off.

    When off: the live book/delta ingestor drops its flush batches, the
    proactive subscription loop subscribes nothing, and the
    crypto.update.dispatch bus tee is skipped.  Takes effect within a few
    seconds (short-TTL cache) across processes — no restart required.
    """
    from services.recording_control import set_recording_enabled

    enabled = await set_recording_enabled(bool(req.enabled))
    logger.info("recording master switch set to enabled=%s via API", enabled)
    return {"enabled": enabled}


class _RecorderConfigUpdate(_PydBaseModel):
    """Partial recorder-config update.  Every field is optional; only the
    supplied keys are changed.  Ranges are validated here so a bad value is
    rejected with 400 before it reaches the store."""

    depth_levels: int | None = Field(
        None, ge=1, le=25, description="L2 levels per side persisted (1..25)"
    )
    max_tokens: int | None = Field(
        None, ge=0, description="REST-baseline breadth cap (markets snapshotted for carry-forward)"
    )
    ws_max_tokens: int | None = Field(
        None,
        ge=0,
        description="Live WS tick-fidelity cap — only the top N liquidity-ranked markets get a WS subscription; the tail is covered by the REST baseline. Bounds delta volume so recording can't starve the orchestrator.",
    )
    min_liquidity_usd: float | None = Field(
        None, ge=0.0, description="Liquidity floor (USD) before the cap"
    )
    capture_books: bool | None = Field(None, description="Record L2 book snapshots + deltas")
    capture_trades: bool | None = Field(None, description="Record trade prints")
    capture_catalog: bool | None = Field(
        None, description="Tee the scanner catalog into the bus each scan"
    )
    capture_crypto_dispatch: bool | None = Field(
        None,
        description="Tee every live CRYPTO_UPDATE dispatch into the bus (strict-fidelity crypto replay)",
    )
    book_retention_days: int | None = Field(
        None, ge=1, le=365, description="Days of recorded book parquet to retain (disk budget)"
    )
    book_max_bytes: int | None = Field(
        None, ge=1024 * 1024 * 1024,
        description="Max on-disk bytes for recorded book parquet (>=1 GB); the denser REST-baseline recording needs headroom",
    )
    disk_guard_enabled: bool | None = Field(
        None,
        description="Free-DISK guard: pause recording writes (and force-prune oldest windows) when total free disk drops below the threshold, so recording can never fill the drive to 0 bytes and crash the host. Independent of the size caps above.",
    )
    disk_guard_min_free_gb: int | None = Field(
        None, ge=0, le=1000,
        description="Free-disk headroom (GB) below which the guard pauses writes and force-prunes",
    )


@router.get("/recorder/config")
async def get_recorder_config_state() -> dict[str, Any]:
    """Operator recording configuration (depth / coverage / capture toggles).

    Returns the full persisted config — depth_levels, max_tokens,
    min_liquidity_usd, capture_books, capture_trades, capture_catalog,
    capture_crypto_dispatch — every
    key always present (service defaults fill any unset).  Read live by the
    ingestor (depth + capture knobs) and the proactive subscription loop
    (max_tokens + liquidity floor); changes take effect within seconds, no
    restart.
    """
    from services.recording_control import get_recorder_config_persisted

    cfg = await get_recorder_config_persisted()
    # Attach the live free-disk-guard status (current free space, whether the
    # guard is active right now, and when it last tripped) so the UI can surface
    # "when it kicks in" alongside the editable enable/threshold knobs.
    try:
        from services.disk_guard import status as _disk_guard_status
        from services.external_data.parquet_schema import parquet_root

        cfg["disk_guard_status"] = await _disk_guard_status(parquet_root())
    except Exception:
        cfg["disk_guard_status"] = None
    return cfg


@router.put("/recorder/config")
async def set_recorder_config_state(req: _RecorderConfigUpdate) -> dict[str, Any]:
    """Update one or more recorder-config knobs.  Accepts a partial body; only
    the supplied keys change.  Ranges are validated (depth 1..25, non-negative
    cap/liquidity).  Returns the full updated config."""
    from services.recording_control import set_recorder_config

    # Only forward keys the caller actually set (exclude_unset) so an omitted
    # field preserves its stored value rather than being reset.
    partial = req.model_dump(exclude_unset=True, exclude_none=True)
    updated = await set_recorder_config(**partial)
    logger.info("recorder config updated keys=%s via API", sorted(partial.keys()))
    return updated


@router.get("/recorder/proactive-subscription")
async def proactive_subscription_status() -> dict[str, Any]:
    """Status of the proactive recorder subscription loop.

    Surfaces the funnel: catalog markets → candidates → above-liquidity
    floor → top-N target → subscribed.  When the gap between target and
    subscribed is large, the WS feed is rejecting subscriptions
    (connection issue) or the feed manager isn't initialized.
    """
    try:
        from services.recorder_subscription_service import get_status

        status = get_status()
        status["actual_recording"] = await _recent_ingestion_actuals()
        return status
    except Exception as exc:
        logger.exception("proactive_subscription_status failed")
        return {"error": str(exc)}


@router.get("/recorder/microstructure")
async def microstructure_recorder_status() -> dict[str, Any]:
    """Live status of the unified market-data ingestor.

    The previous standalone ``MicrostructureRecorder`` was merged into
    ``LiveMarketDataIngestor`` (see services/market_data_ingestor.py).
    The endpoint path is preserved for backwards compatibility with
    the Data Lab UI, but now reads from the unified ingestor's stats.

    The ingestor is in-process and always-on whenever the WebSocket
    feed is alive — there is no separate recorder process to start /
    stop / configure.  Persistence runs off the hot path so it cannot
    impact the orchestrator's sub-second decision loop.

    Returns:
      running: bool — heuristic ("any tokens tracked in this process?")
      tokens_tracked: int
      accepted_books, total_attempts, accept_rate
      rejects_by_reason: per-reason counters
      sequence_gaps_observed: forward jumps > 1
      snapshot_queue_dropped, delta_queue_dropped: backpressure drops
      flush_latency_ms_p50/p95: persistence-task latency
    """
    try:
        from services.market_data_ingestor import get_market_data_ingestor

        ing = get_market_data_ingestor()
        stats = ing.get_data_quality_stats()
        # "Running" heuristic — ingestor is in-process and active
        # whenever the WS feed is alive; proxy on whether any token
        # has been seen.
        stats["running"] = bool(stats.get("tokens_tracked"))
        # Per-process counters above only see the API process; attach the
        # shared-DB truth so the UI shows what the worker is really recording.
        stats["actual_recording"] = await _recent_ingestion_actuals()
        if stats["actual_recording"].get("actively_recording"):
            stats["running"] = True
        return stats
    except Exception as exc:
        logger.exception("microstructure_recorder_status failed")
        return {"error": str(exc), "running": False}


@router.get("/storage/summary")
async def storage_summary() -> dict[str, Any]:
    """Per-table storage usage: row count, on-disk bytes (Postgres
    ``pg_total_relation_size``), and oldest/newest timestamp.

    Powers the Data Lab "Recording" panel so the operator can see
    how much of the disk each dataset is consuming and decide what
    to prune.  Falls back gracefully on engines that don't expose
    pg_total_relation_size — size will be reported as null.
    """
    from sqlalchemy import text

    out: list[dict[str, Any]] = []
    async with AsyncSessionLocal() as session:
        for spec in _DATASETS.values():
            if spec.source == "parquet" or spec.model is None:
                # Parquet-backed datasets have no SQL table; their bytes are
                # reported in the ``parquet`` section below.
                continue
            row_count = 0
            try:
                row_count = int(
                    (await session.execute(select(func.count(spec.model.id)))).scalar_one() or 0
                )
            except Exception:
                pass

            # Oldest + newest timestamp on the natural sort column.
            oldest_iso = None
            newest_iso = None
            sort_col = getattr(spec.model, spec.default_sort, None)
            if sort_col is not None:
                try:
                    oldest = (await session.execute(select(func.min(sort_col)))).scalar_one()
                    newest = (await session.execute(select(func.max(sort_col)))).scalar_one()
                    if oldest is not None and hasattr(oldest, "isoformat"):
                        oldest_iso = oldest.isoformat()
                    if newest is not None and hasattr(newest, "isoformat"):
                        newest_iso = newest.isoformat()
                except Exception:
                    pass

            # Disk size (Postgres-specific).  Returns None on other
            # engines or when the function isn't available.
            size_bytes: int | None = None
            try:
                tbl = spec.model.__tablename__
                stmt = text(f"SELECT pg_total_relation_size('public.{tbl}')")
                size_bytes = int(
                    (await session.execute(stmt)).scalar_one() or 0
                )
            except Exception:
                size_bytes = None

            out.append({
                "name": spec.name,
                "label": spec.label,
                "table_name": spec.model.__tablename__,
                "row_count": row_count,
                "size_bytes": size_bytes,
                "oldest_at": oldest_iso,
                "newest_at": newest_iso,
            })

    total_rows = sum(r["row_count"] for r in out)
    total_bytes = sum((r["size_bytes"] or 0) for r in out) if all(r["size_bytes"] is not None for r in out) else None

    # Parquet data plane — the storage that moved OFF Postgres.  Without
    # this the Storage view (SQL tables only) diverges from the Topics
    # view (parquet bytes).  Walk every configured parquet root and group
    # bytes by top-level provider so the operator sees one reconciled
    # picture: SQL tables + parquet providers + a grand total.
    parquet_providers: list[dict[str, Any]] = []
    parquet_total_bytes = 0
    try:
        from services.external_data.parquet_schema import parquet_roots
        seen: set[str] = set()
        for root in parquet_roots():
            if not root.exists():
                continue
            for prov_dir in sorted(root.iterdir()):
                if not prov_dir.is_dir() or prov_dir.name in seen:
                    continue
                seen.add(prov_dir.name)
                pb = 0
                pf = 0
                for fp in prov_dir.rglob("*.parquet"):
                    try:
                        pb += fp.stat().st_size
                        pf += 1
                    except OSError:
                        continue
                parquet_total_bytes += pb
                parquet_providers.append({
                    "provider": prov_dir.name,
                    "size_bytes": pb,
                    "file_count": pf,
                })
    except Exception as exc:
        logger.warning("storage_summary: parquet enumeration failed: %s", exc)

    parquet_providers.sort(key=lambda p: p["size_bytes"], reverse=True)
    grand_total_bytes = (total_bytes or 0) + parquet_total_bytes
    return {
        "tables": out,
        "total_rows": total_rows,
        "total_bytes": total_bytes,
        "parquet": {
            "providers": parquet_providers,
            "total_bytes": parquet_total_bytes,
        },
        "grand_total_bytes": grand_total_bytes,
    }


@router.get("/{name}/distinct/{column}")
async def dataset_distinct_values(
    name: str,
    column: str,
    limit: int = Query(default=200, ge=1, le=2000),
) -> dict[str, Any]:
    """Distinct values for a column — feeds the filter dropdowns
    (e.g., 'all known strategy_slugs' for the strategy filter).
    """
    spec = _resolve_dataset(name)
    if spec.source == "parquet" or spec.model is None:
        # Parquet datasets have no SQL column to DISTINCT over (token_id is
        # free-text; the only enum filter carries static values).
        return {"column": column, "values": []}
    col = getattr(spec.model, column, None)
    if col is None:
        raise HTTPException(status_code=400, detail=f"Unknown column '{column}'")
    async with AsyncSessionLocal() as session:
        stmt = select(col).distinct().limit(int(limit))
        rows = (await session.execute(stmt)).scalars().all()
    return {"column": column, "values": [v for v in rows if v is not None]}


# ── Recorded-token picker (parquet per-token datasets) ───────────────────
#
# Parquet book datasets are per-token: a read needs a token_id, but a raw
# 77-digit CLOB id is not something an operator has memorised.  The
# /{name}/recorded-tokens endpoint lists the most-recently-recorded tokens
# (recency-ranked, straight from the provider_datasets window index) with
# best-effort market labels, so the browser offers a searchable pick-list
# instead of an empty free-text field.

_TOKEN_LABEL_CACHE: dict[str, str] = {}
_TOKEN_LABEL_CACHE_TS: float = 0.0
_TOKEN_LABEL_TTL_SECONDS = 300.0


def _market_token_labels() -> dict[str, str]:
    """``token_id -> "<question> · <outcome>"`` from the market catalog file.

    Cached for ``_TOKEN_LABEL_TTL_SECONDS`` — the catalog is large and changes
    slowly, so it is rebuilt at most once per 5 min.  Best-effort: tokens absent
    from the catalog simply get no label (the picker falls back to the id).
    """
    global _TOKEN_LABEL_CACHE, _TOKEN_LABEL_CACHE_TS
    now = time.monotonic()
    if _TOKEN_LABEL_CACHE and (now - _TOKEN_LABEL_CACHE_TS) < _TOKEN_LABEL_TTL_SECONDS:
        return _TOKEN_LABEL_CACHE
    labels: dict[str, str] = {}
    try:
        from services.shared_state import _read_market_catalog_file

        cat = _read_market_catalog_file()
    except Exception:  # noqa: BLE001
        cat = None
    if cat:
        _events, markets, _meta = cat
        for m in markets or []:
            if not isinstance(m, dict):
                continue
            question = str(m.get("question") or m.get("title") or "").strip()
            raw = m.get("clob_token_ids") or []
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except Exception:
                    raw = []
            outcomes = m.get("outcomes") or []
            if isinstance(outcomes, str):
                try:
                    outcomes = json.loads(outcomes)
                except Exception:
                    outcomes = []
            for idx, tok in enumerate(raw or []):
                ts = str(tok).strip()
                if not ts:
                    continue
                outcome = str(outcomes[idx]).strip() if idx < len(outcomes) else ""
                labels[ts] = f"{question} · {outcome}" if (question and outcome) else question
    _TOKEN_LABEL_CACHE = labels
    _TOKEN_LABEL_CACHE_TS = now
    return labels


@router.get("/{name}/recorded-tokens")
async def dataset_recorded_tokens(
    name: str,
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict[str, Any]:
    """Recently-recorded tokens for a parquet per-token dataset, recency-ranked
    with best-effort market labels.  Feeds the Data Lab token picker so browsing
    book snapshots doesn't require pasting a raw token_id.  Empty for SQL
    datasets (their filters use the standard distinct-values dropdowns)."""
    spec = _resolve_dataset(name)
    if spec.source != "parquet":
        return {"tokens": []}
    # Unnest the token_ids of the 30 newest recorded windows server-side, then
    # rank distinct tokens by most-recent appearance — returns only ``limit``
    # rows regardless of how broad each window's token set is.
    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                text(
                    "SELECT token, max(updated_at) AS last_at FROM ("
                    "  SELECT json_array_elements_text(t.token_ids_json) AS token, t.updated_at"
                    "  FROM ("
                    "    SELECT token_ids_json, updated_at FROM provider_datasets"
                    "    WHERE token_ids_json IS NOT NULL"
                    "      AND json_typeof(token_ids_json) = 'array'"
                    "    ORDER BY updated_at DESC LIMIT 30"
                    "  ) t"
                    ") sub GROUP BY token ORDER BY last_at DESC LIMIT :lim"
                ),
                {"lim": int(limit)},
            )
        ).all()
    labels = _market_token_labels()
    tokens = [
        {
            "token_id": r.token,
            "label": labels.get(r.token) or None,
            "last_recorded_at": (r.last_at.isoformat() if r.last_at else None),
        }
        for r in rows
    ]
    return {"tokens": tokens}
