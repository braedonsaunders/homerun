"""Read-only dataset browser API.

Powers Research -> Data Lab in the UI and the ``query_dataset``
agent tool.  Exposes paginated, filtered reads of five core tables
through a single uniform shape so the UI can render any of them
with the same table component.

Datasets:
    microstructure_snapshot  ── L2 book + trade snapshots
    book_delta_event         ── trade-vs-cancel decomposed events
    opportunity_history      ── strategy detect() outputs
    trader_order             ── live + shadow order ledger
    backtest_run             ── persisted BacktestStudio runs

Per-dataset spec is declared in ``_DATASETS`` below and includes the
ORM model, default sort column, allowed filter columns + types, and
the columns to surface in the default UI view.  Adding a sixth
dataset is a matter of one entry in that table.

CSV export streams the same query results through ``StreamingResponse``
so a 100k-row download doesn't blow up Python heap.
"""
from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    AsyncSessionLocal,
    BacktestRun,
    BookDeltaEvent,
    MarketMicrostructureSnapshot,
    OpportunityHistory,
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
    model: type
    default_sort: str
    default_sort_dir: str  # 'asc' | 'desc'
    columns: tuple[ColumnSpec, ...]
    filters: tuple[FilterSpec, ...]


_DATASETS: dict[str, DatasetSpec] = {
    "microstructure_snapshot": DatasetSpec(
        name="microstructure_snapshot",
        label="Microstructure snapshots",
        description=(
            "L2 book + trade-print snapshots from the WebSocket recorder. "
            "Filter by token, time range, or snapshot type."
        ),
        model=MarketMicrostructureSnapshot,
        default_sort="observed_at",
        default_sort_dir="desc",
        columns=(
            ColumnSpec("id", "ID", "string", default_visible=False),
            ColumnSpec("provider", "Provider", "string", default_visible=False),
            ColumnSpec("token_id", "Token", "string"),
            ColumnSpec("snapshot_type", "Type", "enum", enum_values=("book", "trade")),
            ColumnSpec("observed_at", "Observed", "datetime"),
            ColumnSpec("exchange_ts_ms", "Exchange ts (ms)", "int", default_visible=False),
            ColumnSpec("sequence", "Seq", "int"),
            ColumnSpec("best_bid", "Bid", "float"),
            ColumnSpec("best_ask", "Ask", "float"),
            ColumnSpec("spread_bps", "Spread (bps)", "float"),
            ColumnSpec("bids_json", "Bids (L2)", "json", sortable=False, default_visible=False),
            ColumnSpec("asks_json", "Asks (L2)", "json", sortable=False, default_visible=False),
            ColumnSpec("trade_price", "Trade px", "float"),
            ColumnSpec("trade_size", "Trade size", "float"),
            ColumnSpec("trade_side", "Trade side", "enum", enum_values=("BUY", "SELL")),
            ColumnSpec("payload_json", "Payload", "json", sortable=False, default_visible=False),
            ColumnSpec("created_at", "Created", "datetime", default_visible=False),
        ),
        filters=(
            FilterSpec("token_id", "token_id", "Token ID", "eq"),
            FilterSpec("snapshot_type", "snapshot_type", "Type", "enum_in"),
            FilterSpec("start", "observed_at", "From", "time_range_start"),
            FilterSpec("end", "observed_at", "To", "time_range_end"),
            FilterSpec("provider", "provider", "Provider", "eq"),
        ),
    ),
    "book_delta_event": DatasetSpec(
        name="book_delta_event",
        label="Book delta events",
        description=(
            "Decomposed trade-vs-cancel events from the book-delta "
            "decomposer.  Powers the Cox PH fill model training set."
        ),
        model=BookDeltaEvent,
        default_sort="observed_at",
        default_sort_dir="desc",
        columns=(
            ColumnSpec("id", "ID", "string", default_visible=False),
            ColumnSpec("provider", "Provider", "string", default_visible=False),
            ColumnSpec("token_id", "Token", "string"),
            ColumnSpec("event_type", "Event", "enum", enum_values=("trade", "cancel")),
            ColumnSpec("side", "Side", "enum", enum_values=("bid", "ask")),
            ColumnSpec("observed_at", "Observed", "datetime"),
            ColumnSpec("exchange_ts_ms", "Exchange ts", "int", default_visible=False),
            ColumnSpec("sequence", "Seq", "int", default_visible=False),
            ColumnSpec("price", "Price", "float"),
            ColumnSpec("trade_size", "Trade size", "float"),
            ColumnSpec("cancel_size", "Cancel size", "float"),
            ColumnSpec("queue_depth_before", "Depth before", "float"),
            ColumnSpec("queue_depth_after", "Depth after", "float"),
            ColumnSpec("spread_bps_at_event", "Spread (bps)", "float"),
            ColumnSpec("payload_json", "Payload", "json", sortable=False, default_visible=False),
            ColumnSpec("created_at", "Created", "datetime", default_visible=False),
        ),
        filters=(
            FilterSpec("token_id", "token_id", "Token ID", "eq"),
            FilterSpec("event_type", "event_type", "Event type", "enum_in"),
            FilterSpec("side", "side", "Side", "enum_in"),
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


# ── Routes ──────────────────────────────────────────────────────────────


@router.get("")
async def list_datasets() -> dict[str, Any]:
    """List every available dataset with its row count and metadata."""
    out: list[dict[str, Any]] = []
    async with AsyncSessionLocal() as session:
        for spec in _DATASETS.values():
            try:
                total = (await session.execute(select(func.count(spec.model.id)))).scalar_one()
            except Exception:
                total = 0
            out.append({
                "name": spec.name,
                "label": spec.label,
                "description": spec.description,
                "row_count": int(total or 0),
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
        "start": start,
        "end": end,
    }

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
    start: str | None = None,
    end: str | None = None,
):
    """Stream a CSV of the filtered query.

    Caps at ``max_rows`` (default 50k, 500k absolute max).  JSON
    columns are serialized inline as JSON strings — Excel-friendly.
    """
    spec = _resolve_dataset(name)
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
    from sqlalchemy.engine import Inspector

    out: list[dict[str, Any]] = []
    async with AsyncSessionLocal() as session:
        for spec in _DATASETS.values():
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
    return {
        "tables": out,
        "total_rows": total_rows,
        "total_bytes": total_bytes,
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
    col = getattr(spec.model, column, None)
    if col is None:
        raise HTTPException(status_code=400, detail=f"Unknown column '{column}'")
    async with AsyncSessionLocal() as session:
        stmt = select(col).distinct().limit(int(limit))
        rows = (await session.execute(stmt)).scalars().all()
    return {"column": column, "values": [v for v in rows if v is not None]}
