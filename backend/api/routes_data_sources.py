"""Data Source API routes (unified DB-managed source registry)."""

from __future__ import annotations

import re
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    AsyncSessionLocal,
    DataSource,
    DataSourceRecord,
    DataSourceRun,
    DataSourceTombstone,
    get_db_session,
)
from services.data_source_catalog import ensure_all_data_sources_seeded
from services.data_source_loader import (
    DATA_SOURCE_TEMPLATE,
    DataSourceValidationError,
    data_source_loader,
    validate_data_source_source,
)
from services.data_source_runner import run_data_source
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/data-sources", tags=["Data Sources"])

_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]{1,48}[a-z0-9]$")


def _validate_slug(slug: str) -> str:
    value = str(slug or "").strip().lower()
    if not _SLUG_RE.match(value):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid slug '{value}'. Must be 3-50 chars, start with a letter, "
                "use lowercase letters/numbers/underscores, and end with letter/number."
            ),
        )
    return value


class DataSourceCreateRequest(BaseModel):
    slug: str = Field(..., min_length=3, max_length=128)
    source_key: str = Field(default="custom", min_length=2, max_length=64)
    source_kind: str = Field(default="python", min_length=2, max_length=32)
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=500)
    source_code: str = Field(..., min_length=10)
    config: dict = Field(default_factory=dict)
    config_schema: dict = Field(default_factory=dict)
    enabled: bool = True


class DataSourceUpdateRequest(BaseModel):
    slug: Optional[str] = Field(None, min_length=3, max_length=128)
    source_key: Optional[str] = Field(None, min_length=2, max_length=64)
    source_kind: Optional[str] = Field(None, min_length=2, max_length=32)
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    source_code: Optional[str] = Field(None, min_length=10)
    config: Optional[dict] = None
    config_schema: Optional[dict] = None
    enabled: Optional[bool] = None
    unlock_system: bool = False


class DataSourceValidateRequest(BaseModel):
    source_code: str = Field(..., min_length=10)
    class_name: Optional[str] = None


class DataSourceRunRequest(BaseModel):
    max_records: int = Field(default=500, ge=1, le=5000)


def _source_to_dict(row: DataSource) -> dict:
    validation = validate_data_source_source(str(row.source_code or ""), class_name=row.class_name)
    runtime = data_source_loader.get_runtime(str(row.slug or "").strip().lower())
    runtime_payload = None
    if runtime is not None:
        runtime_payload = {
            "slug": runtime.slug,
            "class_name": runtime.class_name,
            "name": runtime.name,
            "description": runtime.description,
            "loaded_at": runtime.loaded_at.isoformat(),
            "source_hash": runtime.source_hash,
            "run_count": runtime.run_count,
            "error_count": runtime.error_count,
            "last_run": runtime.last_run.isoformat() if runtime.last_run else None,
            "last_error": runtime.last_error,
        }

    return {
        "id": row.id,
        "slug": row.slug,
        "source_key": row.source_key,
        "source_kind": row.source_kind,
        "name": row.name,
        "description": row.description,
        "source_code": row.source_code,
        "class_name": row.class_name,
        "is_system": bool(row.is_system),
        "enabled": bool(row.enabled),
        "status": row.status,
        "error_message": row.error_message,
        "version": int(row.version or 1),
        "config": dict(row.config or {}),
        "config_schema": dict(row.config_schema or {}),
        "sort_order": int(row.sort_order or 0),
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        "capabilities": validation.get("capabilities") or {},
        "runtime": runtime_payload,
    }


@router.get("/template")
async def get_data_source_template():
    return {
        "template": DATA_SOURCE_TEMPLATE,
        "instructions": (
            "Create a class extending BaseDataSource and implement fetch() or fetch_async(). "
            "Return normalized dict records and optionally transform each record."
        ),
        "available_imports": [
            "services.data_source_sdk (BaseDataSource, DataSourceSDK)",
            "services.strategy_sdk (StrategySDK)",
            "services.news.*",
            "services.world_intelligence.*",
            "models.*",
            "config (settings)",
            "math, statistics, collections, datetime, re, json, random, asyncio, pathlib",
            "httpx",
            "numpy, scipy, pandas (if installed)",
        ],
    }


@router.get("/docs")
async def get_data_source_docs():
    return {
        "title": "Data Source Developer Reference",
        "version": "1.0",
        "overview": {
            "summary": (
                "Data Sources are DB-managed Python classes that fetch/transform raw data into "
                "normalized records. Strategies can read these records through DataSourceSDK."
            ),
            "lifecycle": [
                "1. Source runs fetch()/fetch_async() and returns list[dict].",
                "2. Optional transform(item) normalizes each record.",
                "3. Runner upserts into data_source_records and writes data_source_runs history.",
                "4. Strategies query records via DataSourceSDK.get_records().",
            ],
        },
        "required_methods": {
            "fetch": "fetch(self) -> list[dict]",
            "fetch_async": "async fetch_async(self) -> list[dict]",
            "note": "Implement one of fetch/fetch_async.",
        },
        "optional_methods": {
            "transform": "transform(self, item: dict) -> dict",
            "configure": "configure(self, config: dict) -> None",
        },
        "record_shape": {
            "external_id": "str | optional",
            "title": "str | optional",
            "summary": "str | optional",
            "category": "str | optional",
            "source": "str | optional",
            "url": "str | optional",
            "observed_at": "datetime|ISO string | optional",
            "payload": "dict | optional",
            "tags": "list[str] | optional",
            "geotagged": "bool | optional",
            "latitude": "float | optional",
            "longitude": "float | optional",
            "country_iso3": "ISO3 code | optional",
        },
        "api_endpoints": {
            "GET /data-sources": "List sources",
            "POST /data-sources": "Create source",
            "PUT /data-sources/{id}": "Update source",
            "DELETE /data-sources/{id}": "Delete source",
            "POST /data-sources/validate": "Validate source code",
            "POST /data-sources/{id}/reload": "Compile/reload source runtime",
            "POST /data-sources/{id}/run": "Run source and ingest records",
            "GET /data-sources/{id}/runs": "Run history",
            "GET /data-sources/{id}/records": "Ingested records",
        },
    }


@router.post("/validate")
async def validate_data_source(req: DataSourceValidateRequest):
    result = validate_data_source_source(req.source_code, class_name=req.class_name)
    return {
        "valid": bool(result.get("valid", False)),
        "class_name": result.get("class_name"),
        "source_name": result.get("source_name"),
        "source_description": result.get("source_description"),
        "capabilities": result.get("capabilities", {}),
        "errors": result.get("errors", []),
        "warnings": result.get("warnings", []),
    }


@router.get("")
async def list_data_sources(
    source_key: Optional[str] = Query(default=None),
    enabled: Optional[bool] = Query(default=None),
):
    async with AsyncSessionLocal() as session:
        await ensure_all_data_sources_seeded(session)

        query = select(DataSource).order_by(
            DataSource.is_system.desc(),
            DataSource.sort_order.asc(),
            DataSource.name.asc(),
        )
        if source_key:
            query = query.where(DataSource.source_key == str(source_key).strip().lower())
        if enabled is not None:
            query = query.where(DataSource.enabled == bool(enabled))

        rows = (await session.execute(query)).scalars().all()
        items = [_source_to_dict(row) for row in rows]

    return {"items": items, "total": len(items)}


@router.get("/{source_id}")
async def get_data_source(source_id: str, session: AsyncSession = Depends(get_db_session)):
    await ensure_all_data_sources_seeded(session)

    row = await session.get(DataSource, source_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Data source not found")

    return _source_to_dict(row)


@router.post("")
async def create_data_source(req: DataSourceCreateRequest):
    slug = _validate_slug(req.slug)
    source_key = str(req.source_key or "custom").strip().lower()
    source_kind = str(req.source_kind or "python").strip().lower()

    validation = validate_data_source_source(req.source_code)
    if not validation["valid"]:
        raise HTTPException(
            status_code=400,
            detail={"message": "Data source validation failed", "errors": validation["errors"]},
        )

    source_name = (req.name or validation.get("source_name") or slug.replace("_", " ").title()).strip()
    source_description = req.description if req.description is not None else validation.get("source_description")
    class_name = validation.get("class_name")

    source_id = uuid.uuid4().hex
    status = "unloaded"
    error_message = None

    async with AsyncSessionLocal() as session:
        existing = await session.execute(select(DataSource).where(DataSource.slug == slug))
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=409, detail=f"A data source with slug '{slug}' already exists.")

        if req.enabled:
            try:
                data_source_loader.load(slug, req.source_code, req.config or None, class_name=class_name)
                status = "loaded"
            except DataSourceValidationError as exc:
                status = "error"
                error_message = str(exc)

        row = DataSource(
            id=source_id,
            slug=slug,
            source_key=source_key,
            source_kind=source_kind,
            name=source_name,
            description=source_description,
            source_code=req.source_code,
            class_name=class_name,
            is_system=False,
            enabled=req.enabled,
            status=status,
            error_message=error_message,
            config=req.config or {},
            config_schema=req.config_schema or {},
            version=1,
            sort_order=0,
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return _source_to_dict(row)


@router.put("/{source_id}")
async def update_data_source(source_id: str, req: DataSourceUpdateRequest):
    async with AsyncSessionLocal() as session:
        row = await session.get(DataSource, source_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Data source not found")

        if bool(row.is_system) and not req.unlock_system:
            raise HTTPException(
                status_code=403,
                detail="System data sources are read-only. Set unlock_system=true for admin override.",
            )

        original_slug = row.slug
        code_changed = False
        slug_changed = False

        if req.slug is not None:
            next_slug = _validate_slug(req.slug)
            if next_slug != row.slug:
                existing_slug = await session.execute(
                    select(DataSource.id).where(
                        DataSource.slug == next_slug,
                        DataSource.id != row.id,
                    )
                )
                if existing_slug.scalar_one_or_none():
                    raise HTTPException(status_code=409, detail=f"Slug '{next_slug}' already exists.")
                row.slug = next_slug
                slug_changed = True

        if req.source_code is not None and req.source_code != row.source_code:
            validation = validate_data_source_source(req.source_code)
            if not validation["valid"]:
                raise HTTPException(
                    status_code=400,
                    detail={"message": "Validation failed", "errors": validation["errors"]},
                )
            row.source_code = req.source_code
            row.class_name = validation.get("class_name")
            if req.name is None and validation.get("source_name"):
                row.name = validation.get("source_name")
            if req.description is None and validation.get("source_description"):
                row.description = validation.get("source_description")
            row.version = int(row.version or 1) + 1
            code_changed = True

        if req.config is not None:
            row.config = req.config
            code_changed = True
        if req.config_schema is not None:
            row.config_schema = req.config_schema

        if req.source_key is not None:
            row.source_key = str(req.source_key or "custom").strip().lower()
        if req.source_kind is not None:
            row.source_kind = str(req.source_kind or "python").strip().lower()
        if req.name is not None:
            row.name = req.name
        if req.description is not None:
            row.description = req.description

        enabled_changed = False
        if req.enabled is not None and req.enabled != row.enabled:
            row.enabled = req.enabled
            enabled_changed = True

        if enabled_changed or code_changed or slug_changed:
            if slug_changed:
                data_source_loader.unload(original_slug)
            if row.enabled:
                try:
                    data_source_loader.load(
                        row.slug,
                        row.source_code,
                        row.config or None,
                        class_name=row.class_name,
                    )
                    row.status = "loaded"
                    row.error_message = None
                except DataSourceValidationError as exc:
                    row.status = "error"
                    row.error_message = str(exc)
            else:
                data_source_loader.unload(row.slug)
                row.status = "unloaded"
                row.error_message = None

        await session.commit()
        await session.refresh(row)
        return _source_to_dict(row)


@router.delete("/{source_id}")
async def delete_data_source(source_id: str):
    async with AsyncSessionLocal() as session:
        row = await session.get(DataSource, source_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Data source not found")

        slug = str(row.slug or "").strip().lower()
        if bool(row.is_system) and slug:
            existing_tombstone = await session.get(DataSourceTombstone, slug)
            if existing_tombstone is None:
                session.add(DataSourceTombstone(slug=slug, reason="deleted_via_api"))

        data_source_loader.unload(slug)
        await session.delete(row)
        await session.commit()

    return {"status": "deleted", "id": source_id}


@router.post("/{source_id}/reload")
async def reload_data_source(source_id: str, session: AsyncSession = Depends(get_db_session)):
    row = await session.get(DataSource, source_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Data source not found")

    slug = str(row.slug or "").strip().lower()
    if not bool(row.enabled):
        data_source_loader.unload(slug)
        row.status = "unloaded"
        row.error_message = None
        await session.commit()
        return {"status": "unloaded", "message": "Source is disabled", "runtime": None}

    try:
        runtime = data_source_loader.load(
            slug=slug,
            source_code=str(row.source_code or ""),
            config=dict(row.config or {}),
            class_name=row.class_name,
        )
        row.status = "loaded"
        row.error_message = None
        row.class_name = runtime.class_name
        await session.commit()
        return {
            "status": "loaded",
            "message": f"Reloaded {slug}",
            "runtime": {
                "slug": runtime.slug,
                "class_name": runtime.class_name,
                "name": runtime.name,
                "description": runtime.description,
                "loaded_at": runtime.loaded_at.isoformat(),
                "source_hash": runtime.source_hash,
                "run_count": runtime.run_count,
                "error_count": runtime.error_count,
                "last_run": runtime.last_run.isoformat() if runtime.last_run else None,
                "last_error": runtime.last_error,
            },
        }
    except DataSourceValidationError as exc:
        row.status = "error"
        row.error_message = str(exc)
        await session.commit()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{source_id}/run")
async def run_data_source_now(
    source_id: str,
    req: DataSourceRunRequest,
    session: AsyncSession = Depends(get_db_session),
):
    row = await session.get(DataSource, source_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Data source not found")

    try:
        result = await run_data_source(
            session,
            row,
            max_records=req.max_records,
            commit=True,
        )
    except (ValueError, DataSourceValidationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return result


@router.get("/{source_id}/runs")
async def list_data_source_runs(
    source_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_db_session),
):
    row = await session.get(DataSource, source_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Data source not found")

    query = (
        select(DataSourceRun)
        .where(DataSourceRun.data_source_id == row.id)
        .order_by(DataSourceRun.started_at.desc())
        .limit(int(limit))
    )
    runs = (await session.execute(query)).scalars().all()

    return {
        "source_id": row.id,
        "source_slug": row.slug,
        "runs": [
            {
                "id": run.id,
                "status": run.status,
                "fetched_count": int(run.fetched_count or 0),
                "transformed_count": int(run.transformed_count or 0),
                "upserted_count": int(run.upserted_count or 0),
                "skipped_count": int(run.skipped_count or 0),
                "error_message": run.error_message,
                "metadata": dict(run.metadata_json or {}),
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "duration_ms": run.duration_ms,
            }
            for run in runs
        ],
    }


@router.get("/{source_id}/records")
async def list_data_source_records(
    source_id: str,
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    geotagged: Optional[bool] = Query(default=None),
    session: AsyncSession = Depends(get_db_session),
):
    row = await session.get(DataSource, source_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Data source not found")

    query = select(DataSourceRecord).where(DataSourceRecord.data_source_id == row.id)
    if geotagged is not None:
        query = query.where(DataSourceRecord.geotagged == bool(geotagged))

    total = int((await session.execute(select(func.count()).select_from(query.subquery()))).scalar() or 0)
    records = (
        (
            await session.execute(
                query.order_by(
                    DataSourceRecord.observed_at.desc(),
                    DataSourceRecord.ingested_at.desc(),
                )
                .offset(int(offset))
                .limit(int(limit))
            )
        )
        .scalars()
        .all()
    )

    return {
        "source_id": row.id,
        "source_slug": row.slug,
        "total": total,
        "offset": int(offset),
        "limit": int(limit),
        "records": [
            {
                "id": record.id,
                "external_id": record.external_id,
                "title": record.title,
                "summary": record.summary,
                "category": record.category,
                "source": record.source,
                "url": record.url,
                "geotagged": bool(record.geotagged),
                "country_iso3": record.country_iso3,
                "latitude": record.latitude,
                "longitude": record.longitude,
                "observed_at": record.observed_at.isoformat() if record.observed_at else None,
                "ingested_at": record.ingested_at.isoformat() if record.ingested_at else None,
                "payload": dict(record.payload_json or {}),
                "transformed": dict(record.transformed_json or {}),
                "tags": list(record.tags_json or []),
            }
            for record in records
        ],
    }
