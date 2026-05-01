"""Data Lab tools — first-party agent access to recorded datasets.

Mirrors the ``Research -> Data Lab`` UI: the agent can list available
datasets, query rows with the same filter vocabulary the operator
sees, and inspect schemas.  Writes are NOT supported — this is a
read-only window into recorded state.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


_AVAILABLE_DATASETS = [
    "microstructure_snapshot",
    "book_delta_event",
    "opportunity_history",
    "trader_order",
    "backtest_run",
]


def build_tools() -> list:
    from services.ai.agent import AgentTool

    return [
        AgentTool(
            name="list_datasets",
            description=(
                "List the read-only datasets the agent can browse via "
                "query_dataset.  Returns each dataset's name, label, row "
                "count, available columns, and supported filter keys.  "
                "Always call this first when answering a 'what data do "
                "we have?' question — schema is the truth, not memory."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            handler=_list_datasets,
            max_calls=5,
            category="data",
        ),
        AgentTool(
            name="query_dataset",
            description=(
                "Read paginated rows from one dataset.  Available datasets: "
                + ", ".join(_AVAILABLE_DATASETS)
                + ".  Filter vocabulary depends on the dataset (see "
                "list_datasets); common filters are token_id, "
                "strategy_type / strategy_key / strategy_slug, mode "
                "(live|shadow|paper), event_type (trade|cancel), and "
                "time-range start/end as ISO timestamps.  Result is "
                "JSON-encoded; rows are bounded by limit (default 50, "
                "max 200 to keep responses readable for the agent)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "enum": list(_AVAILABLE_DATASETS),
                        "description": "Which dataset to query.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max rows to return (default 50, max 200).",
                        "default": 50,
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Skip the first N matching rows.",
                        "default": 0,
                    },
                    "order_by": {
                        "type": "string",
                        "description": (
                            "Column to sort by.  Defaults to the dataset's "
                            "natural timestamp column (observed_at / "
                            "detected_at / created_at / started_at)."
                        ),
                    },
                    "order_dir": {
                        "type": "string",
                        "enum": ["asc", "desc"],
                        "description": "Sort direction (default desc).",
                        "default": "desc",
                    },
                    "filters": {
                        "type": "object",
                        "description": (
                            "Map of filter_key -> value.  Time ranges use "
                            "'start' and 'end' with ISO timestamps."
                        ),
                    },
                },
                "required": ["dataset"],
            },
            handler=_query_dataset,
            max_calls=10,
            category="data",
        ),
    ]


async def _list_datasets(_args: dict[str, Any]) -> dict[str, Any]:
    """Return summary info for every dataset (count, columns, filters)."""
    try:
        from sqlalchemy import func, select

        from models.database import AsyncSessionLocal
        from api.routes_dataset import _DATASETS, _serialize_columns, _serialize_filters

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
    except Exception as exc:
        logger.exception("list_datasets failed")
        return {"error": str(exc)}


async def _query_dataset(args: dict[str, Any]) -> dict[str, Any]:
    """Run a paginated query against one dataset and return rows."""
    try:
        dataset = str(args.get("dataset") or "").strip()
        if dataset not in _AVAILABLE_DATASETS:
            return {
                "error": f"Unknown dataset '{dataset}'. Use one of: {_AVAILABLE_DATASETS}",
            }
        limit = max(1, min(int(args.get("limit") or 50), 200))
        offset = max(0, int(args.get("offset") or 0))
        order_by = args.get("order_by")
        order_dir = str(args.get("order_dir") or "desc").lower()
        if order_dir not in {"asc", "desc"}:
            order_dir = "desc"
        filters = args.get("filters") or {}
        if not isinstance(filters, dict):
            filters = {}

        from sqlalchemy import func, select

        from models.database import AsyncSessionLocal
        from api.routes_dataset import (
            _DATASETS,
            _apply_filters,
            _row_to_dict,
            _serialize_columns,
        )

        spec = _DATASETS[dataset]
        sort_col_name = order_by or spec.default_sort
        sort_attr = getattr(spec.model, sort_col_name, None)
        if sort_attr is None:
            return {"error": f"Unknown column '{sort_col_name}' on '{dataset}'"}

        # Map filters keyed by the spec's filter keys.  Pass through any
        # key the spec recognizes; ignore the rest.
        recognized = {f.key for f in spec.filters}
        params = {k: v for k, v in filters.items() if k in recognized}

        async with AsyncSessionLocal() as session:
            count_stmt = _apply_filters(select(func.count(spec.model.id)), spec, params)
            total = (await session.execute(count_stmt)).scalar_one() or 0
            stmt = select(spec.model)
            stmt = _apply_filters(stmt, spec, params)
            stmt = stmt.order_by(sort_attr.desc() if order_dir == "desc" else sort_attr.asc())
            stmt = stmt.limit(limit).offset(offset)
            rows = (await session.execute(stmt)).scalars().all()

        return {
            "dataset": dataset,
            "total": int(total),
            "limit": limit,
            "offset": offset,
            "order_by": sort_col_name,
            "order_dir": order_dir,
            "applied_filters": params,
            "ignored_filters": sorted(set(filters.keys()) - recognized) if filters else [],
            "columns": _serialize_columns(spec),
            "rows": [_row_to_dict(r, spec) for r in rows],
        }
    except Exception as exc:
        logger.exception("query_dataset failed")
        return {"error": str(exc)}
