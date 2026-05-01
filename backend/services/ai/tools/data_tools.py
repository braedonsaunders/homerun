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


_RECORDING_TARGET_KINDS = ["token", "condition", "event"]
_RECORDING_CAPTURE_TYPES = ["book", "trade", "delta"]


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
        AgentTool(
            name="list_recording_sessions",
            description=(
                "List on-demand recording sessions (the operator-defined "
                "scoped market-data captures) with their current status, "
                "target tokens, capture types, and rows captured.  Use "
                "this when answering 'do we have recorded data for X?' "
                "or before triggering session-scoped backtest / training."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "statuses": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional status filter — any of pending, "
                            "scheduled, running, paused, completed, "
                            "failed, cancelled."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max sessions to return (default 50, max 200).",
                        "default": 50,
                    },
                },
                "required": [],
            },
            handler=_list_recording_sessions,
            max_calls=10,
            category="data",
        ),
        AgentTool(
            name="create_recording_session",
            description=(
                "Create a new on-demand recording session.  Pick markets "
                "(by token_id, condition_id, or event slug), capture types "
                "(book/trade/delta), tick interval, and an optional time "
                "window.  Session lands in 'pending' (or 'scheduled' if "
                "scheduled_start_at is set).  Call start_recording_session "
                "to activate immediately."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Human-readable session name.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional notes about why this session was created.",
                    },
                    "target_kind": {
                        "type": "string",
                        "enum": _RECORDING_TARGET_KINDS,
                        "description": (
                            "How target_values should be interpreted: "
                            "'token' = raw clob_token_id (one per outcome), "
                            "'condition' = market condition_id (expands to "
                            "all outcome tokens), 'event' = event slug "
                            "(expands to all markets in the event)."
                        ),
                    },
                    "target_values": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of target IDs / slugs per target_kind.",
                    },
                    "capture_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": _RECORDING_CAPTURE_TYPES},
                        "description": (
                            "Which event streams to capture.  book = L2 "
                            "snapshots, trade = print tape, delta = "
                            "trade-vs-cancel decomposed events."
                        ),
                    },
                    "tick_interval_ms": {
                        "type": "integer",
                        "description": "Min interval between book captures in ms (default 500).",
                    },
                    "max_duration_seconds": {
                        "type": "integer",
                        "description": "Auto-stop after this many seconds (optional).",
                    },
                    "scheduled_start_at": {
                        "type": "string",
                        "description": "ISO timestamp; if set, session runs at that time.",
                    },
                    "scheduled_end_at": {
                        "type": "string",
                        "description": "ISO timestamp; if set, session auto-stops then.",
                    },
                },
                "required": ["name", "target_values"],
            },
            handler=_create_recording_session,
            max_calls=5,
            category="data",
        ),
        AgentTool(
            name="start_recording_session",
            description=(
                "Activate a pending or scheduled recording session.  Once "
                "running it captures rows continuously for its target "
                "tokens until stopped manually or auto-stopped at "
                "scheduled_end_at / max_duration_seconds."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                },
                "required": ["session_id"],
            },
            handler=_start_recording_session,
            max_calls=5,
            category="data",
        ),
        AgentTool(
            name="stop_recording_session",
            description=(
                "Stop a running recording session and mark it completed.  "
                "Captured rows remain queryable via query_dataset and the "
                "session is consumable by run_backtest_on_session and "
                "train_adapter_on_session."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "cancel": {
                        "type": "boolean",
                        "description": "When true, mark cancelled instead of completed.",
                    },
                },
                "required": ["session_id"],
            },
            handler=_stop_recording_session,
            max_calls=5,
            category="data",
        ),
        AgentTool(
            name="run_backtest_on_session",
            description=(
                "Run the unified backtester on a specific recording "
                "session — replays the strategy against the exact "
                "(target_token_ids, started_at..ended_at) slice the "
                "session captured.  Returns the same augmented result "
                "as POST /backtest/run with a recording_session metadata "
                "field attached."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "strategy_id": {
                        "type": "string",
                        "description": "Strategy slug or id to backtest.",
                    },
                    "initial_capital_usd": {"type": "number", "default": 1000.0},
                },
                "required": ["session_id", "strategy_id"],
            },
            handler=_run_backtest_on_session,
            max_calls=3,
            category="data",
        ),
        AgentTool(
            name="train_adapter_on_session",
            description=(
                "Train an ML adapter on the data captured during a "
                "specific recording session — overrides the rolling "
                "training_window_days lookback with the session's exact "
                "time window so the adapter sees only that slice.  The "
                "session id is preserved in the adapter's training "
                "manifest for later attribution."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "base_model_id": {"type": "string"},
                    "adapter_kind": {
                        "type": "string",
                        "description": "e.g. 'platt_scaler', 'isotonic'.",
                    },
                    "task_key": {
                        "type": "string",
                        "default": "crypto_directional",
                    },
                    "name": {"type": "string"},
                    "holdout_days": {"type": "integer", "default": 7},
                },
                "required": ["session_id", "base_model_id", "adapter_kind"],
            },
            handler=_train_adapter_on_session,
            max_calls=3,
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


# ─── Recording session tools ───────────────────────────────────────────


def _serialize_session_brief(row: Any) -> dict[str, Any]:
    return {
        "id": row.id,
        "name": row.name,
        "status": row.status,
        "target_kind": row.target_kind,
        "target_values": list(row.target_values_json or []),
        "target_token_count": len(row.target_token_ids_json or []),
        "capture_types": list(row.capture_types_json or []),
        "tick_interval_ms": row.tick_interval_ms,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "ended_at": row.ended_at.isoformat() if row.ended_at else None,
        "rows_captured": int(row.rows_captured or 0),
        "scheduled_start_at": row.scheduled_start_at.isoformat() if row.scheduled_start_at else None,
        "scheduled_end_at": row.scheduled_end_at.isoformat() if row.scheduled_end_at else None,
        "error": row.error,
    }


async def _list_recording_sessions(args: dict[str, Any]) -> dict[str, Any]:
    try:
        from services.recording_session_service import list_sessions

        statuses = args.get("statuses")
        if isinstance(statuses, str):
            statuses = [s.strip() for s in statuses.split(",") if s.strip()]
        if statuses is not None and not isinstance(statuses, list):
            statuses = None
        limit = max(1, min(int(args.get("limit") or 50), 200))
        rows = await list_sessions(statuses=statuses, limit=limit)
        return {"sessions": [_serialize_session_brief(r) for r in rows], "count": len(rows)}
    except Exception as exc:
        logger.exception("list_recording_sessions failed")
        return {"error": str(exc)}


async def _create_recording_session(args: dict[str, Any]) -> dict[str, Any]:
    try:
        from datetime import datetime as _dt

        from services.recording_session_service import SessionSpec, create_session

        def _parse_iso(v: Any):
            if not v:
                return None
            try:
                return _dt.fromisoformat(str(v).replace("Z", "+00:00"))
            except (ValueError, TypeError):
                return None

        spec = SessionSpec(
            name=str(args.get("name") or "").strip() or "session",
            description=args.get("description"),
            platform=str(args.get("platform") or "polymarket"),
            target_kind=str(args.get("target_kind") or "token"),
            target_values=list(args.get("target_values") or []),
            capture_types=list(args.get("capture_types") or ["book", "trade"]),
            tick_interval_ms=int(args.get("tick_interval_ms") or 500),
            retention_days=args.get("retention_days"),
            scheduled_start_at=_parse_iso(args.get("scheduled_start_at")),
            scheduled_end_at=_parse_iso(args.get("scheduled_end_at")),
            max_duration_seconds=args.get("max_duration_seconds"),
            config=args.get("config"),
        )
        row = await create_session(spec)
        return {"session": _serialize_session_brief(row)}
    except Exception as exc:
        logger.exception("create_recording_session failed")
        return {"error": str(exc)}


async def _start_recording_session(args: dict[str, Any]) -> dict[str, Any]:
    try:
        from services.recording_session_service import start_session

        sid = str(args.get("session_id") or "").strip()
        if not sid:
            return {"error": "session_id is required"}
        row = await start_session(sid)
        return {"session": _serialize_session_brief(row)}
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        logger.exception("start_recording_session failed")
        return {"error": str(exc)}


async def _stop_recording_session(args: dict[str, Any]) -> dict[str, Any]:
    try:
        from services.recording_session_service import stop_session

        sid = str(args.get("session_id") or "").strip()
        if not sid:
            return {"error": "session_id is required"}
        target_status = "cancelled" if args.get("cancel") else "completed"
        row = await stop_session(sid, status=target_status)
        return {"session": _serialize_session_brief(row)}
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        logger.exception("stop_recording_session failed")
        return {"error": str(exc)}


async def _run_backtest_on_session(args: dict[str, Any]) -> dict[str, Any]:
    """Run the unified backtester scoped to a recording session."""
    try:
        from sqlalchemy import or_, select

        from models.database import AsyncSessionLocal, Strategy
        from services.backtest.unified_runner import run_unified_backtest
        from services.recording_session_service import session_backtest_scope

        sid = str(args.get("session_id") or "").strip()
        strategy_id = str(args.get("strategy_id") or "").strip()
        if not sid or not strategy_id:
            return {"error": "session_id and strategy_id are both required"}

        scope = await session_backtest_scope(sid)
        if scope is None:
            return {"error": f"Session '{sid}' not found or has no captured data"}

        async with AsyncSessionLocal() as session:
            row = (
                await session.execute(
                    select(Strategy).where(or_(Strategy.id == strategy_id, Strategy.slug == strategy_id))
                )
            ).scalar_one_or_none()
            if row is None:
                return {"error": f"Strategy '{strategy_id}' not found"}
            source = str(row.source_code or "")
            slug = str(row.slug or strategy_id)
            cfg = dict(row.config or {})

        if not source.strip():
            return {"error": f"Strategy '{slug}' has no source_code"}

        result = await run_unified_backtest(
            source_code=source,
            slug=slug,
            config=cfg,
            token_ids=scope["token_ids"],
            start=scope["start"],
            end=scope["end"],
            initial_capital_usd=float(args.get("initial_capital_usd") or 1000.0),
        )
        execution = result.get("execution") or {}
        sharpe = execution.get("sharpe") or {}
        return {
            "session_id": scope["session_id"],
            "session_name": scope["session_name"],
            "strategy_slug": slug,
            "run_id": result.get("run_id"),
            "trade_count": execution.get("trade_count"),
            "total_return_pct": execution.get("total_return_pct"),
            "sharpe": sharpe.get("value") if isinstance(sharpe, dict) else sharpe,
            "max_drawdown_pct": execution.get("max_drawdown_pct"),
            "fees_paid_usd": execution.get("fees_paid_usd"),
            "runtime_error": execution.get("runtime_error"),
        }
    except Exception as exc:
        logger.exception("run_backtest_on_session failed")
        return {"error": str(exc)}


async def _train_adapter_on_session(args: dict[str, Any]) -> dict[str, Any]:
    """Train an ML adapter on data captured during a session."""
    try:
        from services.machine_learning_sdk import get_machine_learning_sdk

        sid = str(args.get("session_id") or "").strip()
        base_model_id = str(args.get("base_model_id") or "").strip()
        adapter_kind = str(args.get("adapter_kind") or "").strip()
        if not sid or not base_model_id or not adapter_kind:
            return {
                "error": "session_id, base_model_id, and adapter_kind are required",
            }

        payload = {
            "task_key": str(args.get("task_key") or "crypto_directional"),
            "base_model_id": base_model_id,
            "adapter_kind": adapter_kind,
            "name": args.get("name") or f"{adapter_kind}_session_{sid[:8]}",
            "holdout_days": int(args.get("holdout_days") or 7),
            "recording_session_id": sid,
            # When session-scoped the lookback is moot, but the SDK
            # validator still requires a value in [7, 365].
            "training_window_days": 90,
        }
        return await get_machine_learning_sdk().start_adapter_training(payload)
    except Exception as exc:
        logger.exception("train_adapter_on_session failed")
        return {"error": str(exc)}
