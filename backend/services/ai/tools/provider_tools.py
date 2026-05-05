"""Agent tools for provider data import (polybacktest etc.).

Lets external MCP clients (Claude Code, Cursor, Continue) drive the
same import workflow the human operator uses in Data Lab → Providers.

Tool list:
  * ``polybacktest_status``           — is the API key configured?
  * ``polybacktest_list_markets``     — search/list markets by coin
  * ``provider_import_polybacktest``  — enqueue an import job
  * ``provider_import_status``        — read job status by ID
  * ``provider_import_list``          — list recent jobs
  * ``provider_dataset_list``         — browse imported datasets
  * ``provider_dataset_delete``       — purge an imported dataset
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from services.ai.agent import AgentTool

logger = logging.getLogger(__name__)


def build_tools() -> list[AgentTool]:
    return [
        AgentTool(
            name="polybacktest_status",
            description=(
                "Check whether polybacktest.com is configured + healthy.  "
                "Returns the API key configured flag and (if configured) a "
                "lightweight ping result."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            handler=_polybacktest_status,
            max_calls=4,
            category="data",
        ),
        AgentTool(
            name="polybacktest_list_markets",
            description=(
                "Search/list markets on polybacktest.com for a given coin "
                "(btc | eth | sol).  Cursor-paginated."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "coin": {"type": "string", "default": "btc"},
                    "search": {"type": "string"},
                    "cursor": {"type": "string"},
                    "limit": {"type": "integer", "default": 50, "minimum": 1, "maximum": 200},
                },
                "required": [],
            },
            handler=_polybacktest_list_markets,
            max_calls=10,
            category="data",
        ),
        AgentTool(
            name="provider_import_polybacktest",
            description=(
                "Enqueue an asynchronous import of polybacktest historical "
                "data into our local Data Lab.  The discovery-plane worker "
                "picks the job up within a few seconds.  Returns the job_id "
                "you can poll with provider_import_status."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "coin": {"type": "string", "default": "btc"},
                    "market_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "start": {
                        "type": "string",
                        "description": "ISO-8601 datetime (inclusive)",
                    },
                    "end": {
                        "type": "string",
                        "description": "ISO-8601 datetime (exclusive)",
                    },
                    "include_trades": {"type": "boolean", "default": True},
                },
                "required": ["coin", "market_ids", "start", "end"],
            },
            handler=_polybacktest_import,
            max_calls=10,
            category="data",
        ),
        AgentTool(
            name="provider_import_status",
            description="Read status / progress / result for a provider import job by ID.",
            parameters={
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
            handler=_provider_import_status,
            max_calls=20,
            category="data",
        ),
        AgentTool(
            name="provider_import_list",
            description="List recent provider import jobs.",
            parameters={
                "type": "object",
                "properties": {
                    "provider": {"type": "string"},
                    "status": {"type": "string"},
                    "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 200},
                },
                "required": [],
            },
            handler=_provider_import_list,
            max_calls=10,
            category="data",
        ),
        AgentTool(
            name="provider_dataset_list",
            description=(
                "List imported provider datasets — these are the entries the "
                "Backtest Studio dataset picker shows.  Filter by provider "
                "or coin."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "provider": {"type": "string"},
                    "coin": {"type": "string"},
                    "limit": {"type": "integer", "default": 50, "minimum": 1, "maximum": 500},
                },
                "required": [],
            },
            handler=_provider_dataset_list,
            max_calls=10,
            category="data",
        ),
        AgentTool(
            name="provider_dataset_delete",
            description=(
                "Delete an imported provider dataset and its underlying "
                "microstructure rows.  Destructive — only call when the "
                "operator has explicitly authorized cleanup."
            ),
            parameters={
                "type": "object",
                "properties": {"dataset_id": {"type": "string"}},
                "required": ["dataset_id"],
            },
            handler=_provider_dataset_delete,
            max_calls=5,
            category="data",
        ),
    ]


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def _polybacktest_status(_args: dict) -> dict:
    from services.external_data.polybacktest_client import (
        PolybacktestNotConfiguredError,
        build_client_from_settings,
    )

    try:
        client = await build_client_from_settings()
    except PolybacktestNotConfiguredError as exc:
        return {"configured": False, "ok": False, "error": str(exc)}
    except Exception as exc:
        return {"configured": False, "ok": False, "error": str(exc)}
    try:
        result = await client.health()
        result["configured"] = True
        return result
    finally:
        await client.close()


async def _polybacktest_list_markets(args: dict) -> dict:
    from services.external_data.polybacktest_client import (
        PolybacktestError,
        PolybacktestNotConfiguredError,
        build_client_from_settings,
    )

    try:
        client = await build_client_from_settings()
    except PolybacktestNotConfiguredError as exc:
        return {"error": str(exc)}
    try:
        markets, next_cursor = await client.list_markets(
            coin=str(args.get("coin") or "btc"),
            cursor=args.get("cursor"),
            search=args.get("search"),
            limit=int(args.get("limit") or 50),
        )
    except PolybacktestError as exc:
        return {"error": str(exc)}
    finally:
        await client.close()
    return {
        "next_cursor": next_cursor,
        "markets": [
            {
                "market_id": m.market_id,
                "slug": m.slug,
                "title": m.title,
                "start_ts": m.start_ts,
                "end_ts": m.end_ts,
                "status": m.status,
            }
            for m in markets
        ],
    }


async def _polybacktest_import(args: dict) -> dict:
    from services.external_data.provider_import_service import (
        CreatePolybacktestJobSpec,
        enqueue_polybacktest_import,
    )

    start = _parse_iso(args.get("start"))
    end = _parse_iso(args.get("end"))
    if start is None or end is None:
        return {"error": "start and end must be ISO-8601 datetimes"}
    spec = CreatePolybacktestJobSpec(
        coin=str(args.get("coin") or "btc"),
        market_ids=list(args.get("market_ids") or []),
        start_ms=int(start.timestamp() * 1000),
        end_ms=int(end.timestamp() * 1000),
        include_trades=bool(args.get("include_trades", True)),
    )
    try:
        job = await enqueue_polybacktest_import(spec)
    except ValueError as exc:
        return {"error": str(exc)}
    return _job_dict(job)


async def _provider_import_status(args: dict) -> dict:
    from sqlalchemy import select

    from models.database import AsyncSessionLocal, ProviderImportJob

    job_id = str(args.get("job_id") or "").strip()
    if not job_id:
        return {"error": "job_id is required"}
    async with AsyncSessionLocal() as session:
        row = (
            await session.execute(
                select(ProviderImportJob).where(ProviderImportJob.id == job_id)
            )
        ).scalar_one_or_none()
    if row is None:
        return {"error": f"job '{job_id}' not found"}
    return _job_dict(row)


async def _provider_import_list(args: dict) -> dict:
    from sqlalchemy import select

    from models.database import AsyncSessionLocal, ProviderImportJob

    limit = int(args.get("limit") or 20)
    provider = args.get("provider")
    status = args.get("status")
    async with AsyncSessionLocal() as session:
        stmt = select(ProviderImportJob)
        if provider:
            stmt = stmt.where(ProviderImportJob.provider == provider)
        if status:
            stmt = stmt.where(ProviderImportJob.status == status)
        stmt = stmt.order_by(ProviderImportJob.created_at.desc()).limit(limit)
        rows = list((await session.execute(stmt)).scalars().all())
    return {"jobs": [_job_dict(r) for r in rows]}


async def _provider_dataset_list(args: dict) -> dict:
    from services.external_data.provider_import_service import list_provider_datasets

    rows = await list_provider_datasets(
        provider=args.get("provider"),
        coin=args.get("coin"),
        limit=int(args.get("limit") or 50),
    )
    return {
        "datasets": [
            {
                "id": r.id,
                "provider": r.provider,
                "coin": r.coin,
                "external_id": r.external_id,
                "title": r.title,
                "token_ids": list(r.token_ids_json or []),
                "snapshot_count": int(r.snapshot_count or 0),
                "trade_count": int(r.trade_count or 0),
                "start_ts": r.start_ts.isoformat() if r.start_ts else None,
                "end_ts": r.end_ts.isoformat() if r.end_ts else None,
                "last_imported_at": r.last_imported_at.isoformat() if r.last_imported_at else None,
            }
            for r in rows
        ]
    }


async def _provider_dataset_delete(args: dict) -> dict:
    from services.external_data.provider_import_service import delete_provider_dataset

    dataset_id = str(args.get("dataset_id") or "").strip()
    if not dataset_id:
        return {"error": "dataset_id is required"}
    ok = await delete_provider_dataset(dataset_id)
    return {"deleted": bool(ok), "dataset_id": dataset_id}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _job_dict(row: Any) -> dict[str, Any]:
    return {
        "id": row.id,
        "provider": row.provider,
        "status": row.status,
        "progress": float(row.progress or 0.0),
        "message": row.message,
        "snapshots_fetched": int(row.snapshots_fetched or 0),
        "snapshots_inserted": int(row.snapshots_inserted or 0),
        "trades_fetched": int(row.trades_fetched or 0),
        "api_calls": int(row.api_calls or 0),
        "error": row.error,
        "result": row.result_json,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
    }


def _parse_iso(value: Any):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
