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
from services.data_source_catalog import ensure_all_data_sources_seeded, list_prebuilt_data_source_presets
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
    presets = [
        {
            "id": "python_custom",
            "slug_prefix": "custom_source",
            "name": "Custom Python Source",
            "description": "Full custom Python source with BaseDataSource",
            "source_key": "custom",
            "source_kind": "python",
            "source_code": DATA_SOURCE_TEMPLATE,
            "config": {},
            "config_schema": {"param_fields": []},
            "is_system_seed": False,
        },
        *list_prebuilt_data_source_presets(),
    ]
    return {
        "template": DATA_SOURCE_TEMPLATE,
        "default_preset": "python_custom",
        "presets": presets,
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
        "version": "2.0",
        "overview": {
            "summary": (
                "Data Sources are DB-managed Python classes that ingest external signals, normalize them "
                "into a shared record contract, and persist them for strategy consumption. They are designed "
                "for deterministic ingestion and repeatable strategy access."
            ),
            "concepts": {
                "source": "A Python class extending BaseDataSource, stored in data_sources.",
                "run": "One execution of a source (fetch -> transform -> upsert), stored in data_source_runs.",
                "record": "A normalized output row from a source, stored in data_source_records.",
                "source_key": (
                    "Domain bucket for UI and strategy routing (events, stories, weather, crypto, traders, custom)."
                ),
                "source_kind": "Implementation kind (python, rss, gdelt, events, stories, bridge).",
            },
            "three_stage_lifecycle": {
                "description": (
                    "Each source execution has three stages. The runner always records run metrics, even if "
                    "no rows are produced."
                ),
                "stages": [
                    {
                        "stage": "FETCH",
                        "method": "fetch() or fetch_async()",
                        "purpose": "Collect raw upstream payloads and return list[dict].",
                        "output": "Unnormalized records.",
                    },
                    {
                        "stage": "TRANSFORM",
                        "method": "transform(item)",
                        "purpose": "Normalize each fetched item into canonical record shape.",
                        "output": "Normalized record dict per item.",
                    },
                    {
                        "stage": "UPSERT",
                        "method": "runner-managed",
                        "purpose": "Write rows into data_source_records and update run counters/history.",
                        "output": "fetched/transformed/upserted/skipped counters + status.",
                    },
                ],
            },
        },
        "quick_start": [
            "1. GET /data-sources/template and choose a preset.",
            "2. Implement class attributes (name, description, default_config).",
            "3. Implement fetch() or fetch_async() returning list[dict].",
            "4. Optional: implement transform(item) to normalize each row.",
            "5. POST /data-sources/validate with source_code.",
            "6. POST /data-sources to save with enabled=true.",
            "7. POST /data-sources/{id}/run to ingest records immediately.",
            "8. GET /data-sources/{id}/records to verify normalized output.",
        ],
        "base_data_source": {
            "import": "from services.data_source_sdk import BaseDataSource",
            "class_attributes": {
                "name": {
                    "type": "str",
                    "required": True,
                    "description": "Human-readable source label shown in the Data -> Sources UI.",
                },
                "description": {
                    "type": "str",
                    "required": False,
                    "description": "Short description of the upstream feed and normalization intent.",
                },
                "default_config": {
                    "type": "dict",
                    "required": False,
                    "description": (
                        "Default runtime config merged with user overrides. Access merged values via self.config."
                    ),
                },
            },
            "methods": {
                "configure": {
                    "signature": "configure(self, config: dict | None) -> None",
                    "required": False,
                    "description": (
                        "Called by loader after instantiation. Default behavior merges default_config with config."
                    ),
                },
                "fetch": {
                    "signature": "fetch(self) -> list[dict[str, Any]]",
                    "required": False,
                    "description": "Sync ingestion path for CPU-bound or simple local parsing.",
                },
                "fetch_async": {
                    "signature": "async fetch_async(self) -> list[dict[str, Any]]",
                    "required": False,
                    "description": (
                        "Async ingestion path for network I/O. If not overridden, default calls fetch() in a thread."
                    ),
                },
                "transform": {
                    "signature": "transform(self, item: dict[str, Any]) -> dict[str, Any]",
                    "required": False,
                    "description": "Normalize one fetched item. Default returns item unchanged.",
                },
            },
            "method_selection": {
                "note": "Implement at least one of fetch() or fetch_async().",
                "runner_behavior": (
                    "Runner prefers fetch_async() when available; otherwise it executes fetch()."
                ),
            },
        },
        "record_contract": {
            "description": (
                "Transformed records are stored as data_source_records. Any field can be omitted, but using the "
                "canonical keys below enables UI rendering, geo filters, and strategy-side filtering."
            ),
            "fields": {
                "external_id": "str | optional stable upstream identifier for dedupe and updates",
                "title": "str | optional headline/title shown in previews",
                "summary": "str | optional short text summary",
                "category": "str | optional lowercase category (conflict, macro, weather, crypto, etc.)",
                "source": "str | optional upstream provider name",
                "url": "str | optional canonical source URL",
                "observed_at": "datetime | ISO8601 string | optional event timestamp",
                "payload": "dict | optional raw or lightly-normalized payload",
                "tags": "list[str] | optional keyword tags for downstream filtering",
                "geotagged": "bool | optional defaults false",
                "latitude": "float | optional if geotagged",
                "longitude": "float | optional if geotagged",
                "country_iso3": "str | optional ISO3 country code",
            },
            "normalization_rules": [
                "Use UTC timestamps for observed_at.",
                "Use lowercase categories for stable filtering.",
                "Set geotagged=true only when latitude and longitude are both valid.",
                "Keep payload JSON-serializable (dict/list/scalars only).",
            ],
        },
        "runner_and_storage": {
            "run_status_values": {
                "success": "Source run completed and persisted records (or produced zero valid records).",
                "error": "Source run failed due to validation, runtime, or upstream error.",
                "skipped": "Run intentionally skipped by scheduler/guard logic.",
            },
            "run_metrics": {
                "fetched_count": "Rows returned by fetch/fetch_async before transform filtering.",
                "transformed_count": "Rows produced after transform() normalization.",
                "upserted_count": "Rows inserted/updated in data_source_records.",
                "skipped_count": "Rows dropped due to invalid shape or dedupe guards.",
                "duration_ms": "Total runtime in milliseconds.",
            },
            "storage_tables": {
                "data_sources": "Source registry metadata and Python source code.",
                "data_source_runs": "Execution history for every run.",
                "data_source_records": "Latest normalized records for strategy/UI consumption.",
            },
        },
        "sdk_reference": {
            "import": "from services.data_source_sdk import DataSourceSDK",
            "read_methods": {
                "get_records": {
                    "signature": (
                        "await DataSourceSDK.get_records(source_slug=None, source_slugs=None, limit=200, "
                        "geotagged=None, category=None, since=None)"
                    ),
                    "description": "Read normalized records with optional source/category/time filters.",
                },
                "get_latest_record": {
                    "signature": "await DataSourceSDK.get_latest_record(source_slug, external_id=None)",
                    "description": "Read most recent record for one source (optionally one external_id).",
                },
                "get_recent_runs": {
                    "signature": "await DataSourceSDK.get_recent_runs(source_slug, limit=20)",
                    "description": "Read recent run history for observability/health checks.",
                },
            },
            "management_methods": {
                "list_sources": {
                    "signature": "await DataSourceSDK.list_sources(enabled_only=True, source_key=None, include_code=False)",
                    "description": "Enumerate registered sources and runtime metadata.",
                },
                "get_source": {
                    "signature": "await DataSourceSDK.get_source(source_slug, include_code=True)",
                    "description": "Read one source definition by slug.",
                },
                "validate_source": {
                    "signature": "DataSourceSDK.validate_source(source_code, class_name=None)",
                    "description": "Static validation before create/update.",
                },
                "create_source": {
                    "signature": "await DataSourceSDK.create_source(slug=..., source_code=..., ...)",
                    "description": "Create and optionally load a new source.",
                },
                "update_source": {
                    "signature": "await DataSourceSDK.update_source(source_slug, ...)",
                    "description": "Patch source metadata/code/config and reload runtime.",
                },
                "delete_source": {
                    "signature": "await DataSourceSDK.delete_source(source_slug, unlock_system=False, ...)",
                    "description": "Delete source by slug (system sources require unlock_system=True).",
                },
                "reload_source": {
                    "signature": "await DataSourceSDK.reload_source(source_slug)",
                    "description": "Recompile/reload source code into runtime loader.",
                },
                "run_source": {
                    "signature": "await DataSourceSDK.run_source(source_slug, max_records=500)",
                    "description": "Execute ingestion immediately and return run summary.",
                },
            },
            "strategy_sdk_bridge": {
                "description": (
                    "Strategy code can call StrategySDK wrappers instead of importing DataSourceSDK directly."
                ),
                "methods": {
                    "StrategySDK.get_data_records": "Wrapper around DataSourceSDK.get_records()",
                    "StrategySDK.get_latest_data_record": "Wrapper around DataSourceSDK.get_latest_record()",
                    "StrategySDK.run_data_source": "Wrapper around DataSourceSDK.run_source()",
                    "StrategySDK.list_data_sources": "Wrapper around DataSourceSDK.list_sources()",
                    "StrategySDK.get_data_source": "Wrapper around DataSourceSDK.get_source()",
                    "StrategySDK.get_data_source_runs": "Wrapper around DataSourceSDK.get_recent_runs()",
                },
            },
        },
        "imports": {
            "app_modules": {
                "services.data_source_sdk": "BaseDataSource and DataSourceSDK",
                "services.strategy_sdk": "Read/use data sources from strategy runtime",
                "services.news.*": "News ingestion helpers",
                "services.world_intelligence.*": "World intelligence helpers",
                "models.*": "Core ORM/Pydantic models",
                "config": "Application settings",
            },
            "standard_library": [
                "math",
                "statistics",
                "collections",
                "datetime",
                "time",
                "re",
                "json",
                "random",
                "asyncio",
                "pathlib",
                "itertools",
                "functools",
                "typing",
            ],
            "third_party": {
                "httpx": "Async-friendly HTTP client for upstream APIs",
                "numpy": "Numerical transformations",
                "scipy": "Scientific/statistical transforms",
                "pandas": "Tabular parsing/cleanup (when installed)",
            },
        },
        "validation": {
            "endpoint": "POST /data-sources/validate",
            "description": "Validates source code without saving.",
            "checks_performed": [
                "1. Python syntax and AST parse.",
                "2. Import allow/block checks.",
                "3. BaseDataSource subclass detection.",
                "4. fetch/fetch_async capability detection.",
                "5. Metadata extraction (name/description/class_name).",
            ],
            "response": {
                "valid": "bool",
                "class_name": "str | null",
                "source_name": "str | null",
                "source_description": "str | null",
                "capabilities": {
                    "has_fetch": "bool",
                    "has_fetch_async": "bool",
                    "has_transform": "bool",
                },
                "errors": "list[str]",
                "warnings": "list[str]",
            },
        },
        "api_endpoints": {
            "sources": {
                "GET /data-sources": "List all sources (?source_key, ?enabled filters)",
                "GET /data-sources/{id}": "Get one source by ID",
                "POST /data-sources": "Create a source",
                "PUT /data-sources/{id}": "Update a source",
                "DELETE /data-sources/{id}": "Delete a source",
            },
            "runtime": {
                "POST /data-sources/validate": "Validate source code",
                "POST /data-sources/{id}/reload": "Compile/reload source runtime",
                "POST /data-sources/{id}/run": "Execute source ingestion now",
            },
            "observability": {
                "GET /data-sources/{id}/runs": "Read run history",
                "GET /data-sources/{id}/records": "Read ingested records",
                "GET /data-sources/template": "Starter template + presets",
                "GET /data-sources/docs": "This documentation",
            },
        },
        "examples": {
            "minimal_sync_source": {
                "description": "Minimal synchronous source with static output.",
                "source_code": (
                    "from services.data_source_sdk import BaseDataSource\n\n"
                    "class MinimalSource(BaseDataSource):\n"
                    "    name = 'Minimal Source'\n"
                    "    description = 'Returns one normalized row'\n\n"
                    "    def fetch(self):\n"
                    "        return [\n"
                    "            {\n"
                    "                'external_id': 'sample-1',\n"
                    "                'title': 'Sample headline',\n"
                    "                'summary': 'Normalized sample payload',\n"
                    "                'category': 'custom',\n"
                    "                'source': 'local',\n"
                    "                'payload': {'raw': True},\n"
                    "                'tags': ['sample'],\n"
                    "            }\n"
                    "        ]\n"
                ),
            },
            "async_http_source": {
                "description": "Async fetch with transform normalization and geotagging.",
                "source_code": (
                    "import httpx\n"
                    "from services.data_source_sdk import BaseDataSource\n\n"
                    "class ApiFeedSource(BaseDataSource):\n"
                    "    name = 'API Feed Source'\n"
                    "    description = 'Fetches upstream feed and normalizes rows'\n"
                    "    default_config = {'url': 'https://example.com/feed', 'limit': 100}\n\n"
                    "    async def fetch_async(self):\n"
                    "        async with httpx.AsyncClient(timeout=15.0) as client:\n"
                    "            response = await client.get(self.config['url'])\n"
                    "            response.raise_for_status()\n"
                    "            data = response.json()\n"
                    "        return list(data.get('items', []))[: int(self.config.get('limit', 100))]\n\n"
                    "    def transform(self, item):\n"
                    "        lat = item.get('lat')\n"
                    "        lon = item.get('lon')\n"
                    "        has_geo = lat is not None and lon is not None\n"
                    "        return {\n"
                    "            'external_id': str(item.get('id') or ''),\n"
                    "            'title': item.get('title'),\n"
                    "            'summary': item.get('summary'),\n"
                    "            'category': 'events',\n"
                    "            'source': item.get('provider') or 'api',\n"
                    "            'url': item.get('url'),\n"
                    "            'observed_at': item.get('timestamp'),\n"
                    "            'payload': item,\n"
                    "            'tags': item.get('tags') or [],\n"
                    "            'geotagged': has_geo,\n"
                    "            'latitude': lat,\n"
                    "            'longitude': lon,\n"
                    "            'country_iso3': item.get('iso3'),\n"
                    "        }\n"
                ),
            },
            "strategy_consumer": {
                "description": "Strategy-side usage: read records and trigger source run.",
                "source_code": (
                    "from services.strategy_sdk import StrategySDK\n\n"
                    "async def load_recent_conflict_records():\n"
                    "    records = await StrategySDK.get_data_records(\n"
                    "        source_slug='events_gdelt_tensions',\n"
                    "        category='conflict',\n"
                    "        geotagged=True,\n"
                    "        limit=50,\n"
                    "    )\n"
                    "    return records\n\n"
                    "async def refresh_and_read():\n"
                    "    await StrategySDK.run_data_source('events_gdelt_tensions', max_records=100)\n"
                    "    return await StrategySDK.get_latest_data_record('events_gdelt_tensions')\n"
                ),
            },
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
