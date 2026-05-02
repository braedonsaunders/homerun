"""HTTP API for the global search subsystem.

Endpoints (all prefixed ``/api/`` by ``main.app.include_router``):

* ``GET  /search/global``   — primary user-facing search; returns ranked,
                              grouped, snippet-highlighted results.
* ``GET  /search/recent``   — recent successful queries from the
                              telemetry log; powers "recent searches"
                              in the Cmd+K UI.
* ``GET  /search/stats``    — per-entity-type counts of indexed rows.
* ``POST /search/reindex``  — admin: trigger a synchronous reindex of
                              every entity type or a specific one.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import AsyncSessionLocal, get_db_session
from services.search import COLLECTORS, log_query, reindex_all, reindex_one, run_query
from services.search.service import fetch_index_stats, fetch_recent_queries
from utils.logger import get_logger

logger = get_logger("routes.search")

router = APIRouter()


@router.get("/search/global")
async def search_global(
    session: AsyncSession = Depends(get_db_session),
    q: str = Query(..., min_length=1, description="Search query (websearch syntax allowed)"),
    limit: int = Query(30, ge=1, le=100),
    types: Optional[str] = Query(
        None,
        description="Comma-separated list of entity types to restrict to (e.g. 'market,trader')",
    ),
) -> dict[str, Any]:
    """Run a unified ranked search across every indexed entity type."""
    type_list: Optional[list[str]] = None
    if types:
        type_list = [t.strip() for t in types.split(",") if t.strip()]
        unknown = [t for t in type_list if t not in COLLECTORS]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown entity types: {sorted(unknown)}",
            )

    response = await run_query(
        session,
        query=q,
        limit=limit,
        entity_types=type_list,
    )

    # Telemetry — write to a fresh session so the caller's session
    # state isn't perturbed (and a logging failure can't kill the
    # query).  Best-effort.
    try:
        top_type: Optional[str] = None
        if response["results"]:
            top_type = response["results"][0]["entity_type"]
        async with AsyncSessionLocal() as log_session:
            await log_query(
                log_session,
                query=q,
                result_count=int(response["total"]),
                top_entity_type=top_type,
                latency_ms=float(response["latency_ms"]),
            )
    except Exception as exc:
        logger.debug("search telemetry write failed (non-fatal): %s", exc)

    return response


@router.get("/search/recent")
async def search_recent(
    session: AsyncSession = Depends(get_db_session),
    limit: int = Query(10, ge=1, le=50),
) -> dict[str, Any]:
    """Recent (deduplicated) non-empty queries for the UI's recents list."""
    items = await fetch_recent_queries(session, limit=limit)
    return {"queries": items, "total": len(items)}


@router.get("/search/stats")
async def search_stats(
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Per-entity-type counts in ``search_index``."""
    return await fetch_index_stats(session)


class ReindexRequest(BaseModel):
    entity_type: Optional[str] = None


@router.post("/search/reindex")
async def search_reindex(
    payload: Optional[ReindexRequest] = None,
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Synchronously reindex one or all entity types.

    Used as an admin escape hatch when an operator wants the search
    index up-to-the-millisecond after creating / editing a strategy
    or trader rather than waiting for the worker tick.  Safe to call
    while the worker is running — the upsert pattern means concurrent
    writes converge on the same end-state.
    """
    entity_type = (payload.entity_type if payload else None) or None
    if entity_type:
        if entity_type not in COLLECTORS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown entity_type {entity_type!r}; available: {sorted(COLLECTORS)}",
            )
        result = await reindex_one(session, entity_type)
        return {"ok": bool(result.get("ok")), "entity_type": entity_type, "result": result}

    summary = await reindex_all(session)
    return {"ok": bool(summary.get("ok")), "summary": summary}


@router.get("/search/types")
async def search_types() -> dict[str, Any]:
    """List the entity types the search index covers — useful for the UI's filter chips."""
    return {"types": sorted(COLLECTORS.keys())}
