"""Topic catalog API — operator-facing CRUD over the recorded-event
bus's topic registry.

Routes mounted at ``/api/topics``.  Powers:
  * Data Lab → Topics panel (the canonical "what data is in the system"
    view, replacing the per-source lists scattered across data_sources
    / provider_datasets / catalog UI).
  * Backtest Studio data picker (select bus topics to replay).
  * Strategy editor "subscriptions" autocomplete.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from services.recorded_event_bus import (
    register_topic, get_topic, list_topics, delete_topic,
)
from services.recorded_event_bus.envelope import EnvelopeValidationError, parse_topic

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/topics", tags=["Topic Catalog"])


class TopicResponse(BaseModel):
    slug: str
    title: str
    description: Optional[str] = None
    storage_kind: str
    storage_uri: Optional[str] = None
    schema_version: int
    retention_days: Optional[int] = None
    max_bytes: Optional[int] = None
    enabled: bool
    is_replayable: bool
    publishers: list[str]
    subscribers: list[str]
    last_published_at: Optional[datetime] = None
    last_replayed_at: Optional[datetime] = None
    event_count: int
    bytes_on_disk: int


def _to_response(spec) -> TopicResponse:
    return TopicResponse(
        slug=spec.slug,
        title=spec.title,
        description=spec.description,
        storage_kind=spec.storage_kind,
        storage_uri=spec.storage_uri,
        schema_version=spec.schema_version,
        retention_days=spec.retention_days,
        max_bytes=spec.max_bytes,
        enabled=spec.enabled,
        is_replayable=spec.is_replayable,
        publishers=list(spec.publishers),
        subscribers=list(spec.subscribers),
        last_published_at=spec.last_published_at,
        last_replayed_at=spec.last_replayed_at,
        event_count=spec.event_count,
        bytes_on_disk=spec.bytes_on_disk,
    )


@router.get("", response_model=list[TopicResponse])
async def list_all_topics(
    storage_kind: Optional[str] = Query(
        None, description="filter by storage_kind ('parquet'|'sql_table'|'memory')"
    ),
    enabled_only: bool = Query(False),
    replayable_only: bool = Query(False),
) -> list[TopicResponse]:
    specs = await list_topics(
        storage_kind=storage_kind,
        enabled_only=enabled_only,
        replayable_only=replayable_only,
    )
    return [_to_response(s) for s in specs]


@router.get("/{slug:path}", response_model=TopicResponse)
async def get_one_topic(slug: str) -> TopicResponse:
    spec = await get_topic(slug)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"topic {slug!r} not registered")
    return _to_response(spec)


class TopicCreateRequest(BaseModel):
    slug: str
    title: str
    storage_kind: str  # 'parquet' | 'sql_table' | 'memory'
    storage_uri: Optional[str] = None
    description: Optional[str] = None
    payload_schema: Optional[dict[str, Any]] = None
    schema_version: int = 1
    retention_days: Optional[int] = None
    publishers: list[str] = Field(default_factory=list)
    subscribers: list[str] = Field(default_factory=list)
    enabled: bool = True
    is_replayable: bool = True


@router.post("", response_model=TopicResponse)
async def create_topic(req: TopicCreateRequest) -> TopicResponse:
    try:
        parse_topic(req.slug)
    except EnvelopeValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        spec = await register_topic(
            slug=req.slug,
            title=req.title,
            description=req.description,
            storage_kind=req.storage_kind,
            storage_uri=req.storage_uri,
            payload_schema=req.payload_schema,
            schema_version=req.schema_version,
            retention_days=req.retention_days,
            publishers=req.publishers,
            subscribers=req.subscribers,
            enabled=req.enabled,
            is_replayable=req.is_replayable,
            upsert=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _to_response(spec)


class TopicPatchRequest(BaseModel):
    """Partial update.  Fields left None are not changed.  Used by the
    Data Lab UI to flip ``enabled``, set ``retention_days``, or set
    ``max_bytes`` without re-specifying the whole topic."""
    enabled: Optional[bool] = None
    is_replayable: Optional[bool] = None
    retention_days: Optional[int] = Field(None, ge=0)
    max_bytes: Optional[int] = Field(None, ge=0)
    description: Optional[str] = None
    # 0 / "" → null on the way through (UI uses 0 / empty for "no cap")
    # When ``max_bytes`` or ``retention_days`` is 0, we treat as "clear
    # the cap" (set to NULL).


@router.patch("/{slug:path}", response_model=TopicResponse)
async def patch_topic(slug: str, req: TopicPatchRequest) -> TopicResponse:
    """Per-topic settings update.

    Operator use cases this enables from the Data Lab UI:
      * Toggle a topic on/off (``enabled=False`` makes ``bus.publish``
        fail-closed for it; replay still works for historical data).
      * Set / clear a retention age cap (in days).
      * Set / clear a hard size cap (in bytes).
      * Mark a topic non-replayable (e.g. operator decided this
        topic's data is too sensitive for backtest replay).

    Fields left None aren't touched.  Set ``retention_days=0`` or
    ``max_bytes=0`` to explicitly clear an existing cap (the patch
    treats 0 as null).
    """
    from sqlalchemy import select
    from models.database import AsyncSessionLocal, TopicCatalog
    from services.recorded_event_bus.catalog import _invalidate_cache

    async with AsyncSessionLocal() as session:
        row = (
            await session.execute(
                select(TopicCatalog).where(TopicCatalog.slug == slug)
            )
        ).scalar_one_or_none()
        if row is None:
            raise HTTPException(status_code=404, detail=f"topic {slug!r} not registered")
        if req.enabled is not None:
            row.enabled = bool(req.enabled)
        if req.is_replayable is not None:
            row.is_replayable = bool(req.is_replayable)
        if req.retention_days is not None:
            row.retention_days = int(req.retention_days) if req.retention_days > 0 else None
        if req.max_bytes is not None:
            row.max_bytes = int(req.max_bytes) if req.max_bytes > 0 else None
        if req.description is not None:
            row.description = req.description
        await session.commit()
        _invalidate_cache(slug)
        from services.recorded_event_bus.catalog import TopicSpec
        spec = TopicSpec.from_row(row)
    return _to_response(spec)


@router.delete("/{slug:path}")
async def remove_topic(slug: str) -> dict[str, Any]:
    deleted = await delete_topic(slug)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"topic {slug!r} not registered")
    return {"deleted": True, "slug": slug}


# ─── Global rotation controls ────────────────────────────────────────


class BusSettingsResponse(BaseModel):
    pruner_enabled: bool
    global_max_bytes: Optional[int] = None
    # Convenience: total bytes across all parquet/external_parquet topics
    total_bytes_on_disk: int
    n_parquet_topics: int


class BusSettingsRequest(BaseModel):
    pruner_enabled: Optional[bool] = None
    global_max_bytes: Optional[int] = Field(None, ge=0)


@router.get("/settings/rotation", response_model=BusSettingsResponse)
async def get_bus_rotation_settings() -> BusSettingsResponse:
    """Operator dashboard for the bus's rotation policy."""
    from sqlalchemy import select, func
    from models.database import AsyncSessionLocal, AppSettings, TopicCatalog

    async with AsyncSessionLocal() as session:
        settings = (await session.execute(select(AppSettings).limit(1))).scalar_one_or_none()
        agg = (await session.execute(
            select(
                func.coalesce(func.sum(TopicCatalog.bytes_on_disk), 0),
                func.count(),
            ).where(TopicCatalog.storage_kind.in_(("parquet", "external_parquet")))
        )).first()
    return BusSettingsResponse(
        pruner_enabled=bool(settings.recorded_event_bus_pruner_enabled) if settings else True,
        global_max_bytes=settings.recorded_event_bus_global_max_bytes if settings else None,
        total_bytes_on_disk=int(agg[0] or 0),
        n_parquet_topics=int(agg[1] or 0),
    )


@router.patch("/settings/rotation", response_model=BusSettingsResponse)
async def update_bus_rotation_settings(req: BusSettingsRequest) -> BusSettingsResponse:
    """Update global rotation controls.  Takes effect at the next
    pruner pass (max 5 min lag)."""
    from sqlalchemy import select
    from models.database import AsyncSessionLocal, AppSettings

    async with AsyncSessionLocal() as session:
        settings = (await session.execute(select(AppSettings).limit(1))).scalar_one_or_none()
        if settings is None:
            settings = AppSettings()
            session.add(settings)
        if req.pruner_enabled is not None:
            settings.recorded_event_bus_pruner_enabled = bool(req.pruner_enabled)
        if req.global_max_bytes is not None:
            settings.recorded_event_bus_global_max_bytes = (
                int(req.global_max_bytes) if req.global_max_bytes > 0 else None
            )
        await session.commit()
    return await get_bus_rotation_settings()


@router.post("/settings/prune-now")
async def trigger_prune_now() -> dict[str, Any]:
    """Run one pruning pass synchronously and return the report.
    Useful for "stop logging now, I need disk space" operator moments."""
    from services.recorded_event_bus.pruner import prune_once
    report = await prune_once()
    return report


class TopicReplayPreviewRequest(BaseModel):
    topics: list[str]
    start_us: int
    end_us: int
    limit: int = 25
    time_field: str = "observed_at_us"


@router.post("/replay/preview")
async def replay_preview(req: TopicReplayPreviewRequest) -> dict[str, Any]:
    """Stream the first ``limit`` envelopes for the requested topics
    + window.  Used by the Backtest Studio data picker to show
    operators "what data is actually in this window before they
    commit to a 1-hour backtest run."
    """
    from services.recorded_event_bus import bus, ReplayWindow
    import services.recorded_event_bus.storage  # noqa: F401 — attach

    try:
        win = ReplayWindow(
            start_us=req.start_us,
            end_us=req.end_us,
            topics=tuple(req.topics),
            time_field=req.time_field,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    samples: list[dict[str, Any]] = []
    n_total = 0
    try:
        async for ev in bus.replay(win):
            n_total += 1
            if len(samples) < req.limit:
                samples.append({
                    "topic": ev.topic,
                    "entity_id": ev.entity_id,
                    "observed_at_us": ev.observed_at_us,
                    "ingested_at_us": ev.ingested_at_us,
                    "source": ev.source,
                    "sequence": ev.sequence,
                    "schema_version": ev.schema_version,
                    "payload": dict(ev.payload),
                })
            if n_total >= req.limit * 4:
                # Don't try to count beyond a small bound — large windows
                # could be millions of events.  The "n_total" we return
                # is best-effort an upper bound for the UI's "..." marker.
                break
    except Exception as exc:
        logger.exception("replay preview failed")
        raise HTTPException(status_code=500, detail=f"replay failed: {exc}") from exc

    return {
        "n_seen": n_total,
        "truncated": n_total >= req.limit * 4,
        "samples": samples,
    }
