"""Unified Strategy API Routes

Serves both opportunity-detection (StrategyPlugin) and trader-execution
(TraderStrategyDefinition) strategies through a single /strategies endpoint.

The older /plugins and /trader-strategies endpoints remain functional as
backward-compatible aliases.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    AsyncSessionLocal,
    StrategyPlugin,
    StrategyPluginTombstone,
    TraderStrategyDefinition,
    get_db_session,
)
from services.opportunity_strategy_catalog import (
    SYSTEM_OPPORTUNITY_STRATEGY_SEEDS,
    ensure_system_opportunity_strategies_seeded,
)
from services.plugin_loader import (
    PLUGIN_TEMPLATE,
    PluginValidationError,
    plugin_loader,
    validate_plugin_source,
)
from services.trader_orchestrator.strategy_catalog import (
    ensure_system_trader_strategies_seeded,
)
from services.trader_orchestrator.strategy_db_loader import (
    serialize_trader_strategy_definition,
    strategy_db_loader,
    validate_trader_strategy_source,
)
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/strategy-manager", tags=["Strategies (Unified)"])

# ---------------------------------------------------------------------------
# Slug / key validation
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]{1,48}[a-z0-9]$")


def _validate_slug(slug: str) -> str:
    slug = slug.strip().lower()
    if not _SLUG_RE.match(slug):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid slug '{slug}'. Must be 3-50 chars, start with a letter, "
                f"use only lowercase letters/numbers/underscores, end with letter or number."
            ),
        )
    return slug


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class UnifiedStrategyCreateRequest(BaseModel):
    """Create a strategy in the appropriate table based on its source code."""

    slug: str = Field(..., min_length=3, max_length=128, description="Unique identifier (slug or strategy_key)")
    source_key: str = Field(default="scanner", min_length=2, max_length=64)
    name: Optional[str] = Field(None, min_length=1, max_length=200, description="Display name / label")
    description: Optional[str] = Field(None, max_length=500)
    source_code: str = Field(..., min_length=10)
    class_name: Optional[str] = Field(None, min_length=1, max_length=200)
    config: dict = Field(default_factory=dict, description="Config overrides or default_params_json")
    config_schema: dict = Field(default_factory=dict, description="Param schema (for trader strategies)")
    aliases: list[str] = Field(default_factory=list)
    enabled: bool = True


class UnifiedStrategyUpdateRequest(BaseModel):
    """Partial update for a unified strategy."""

    slug: Optional[str] = Field(None, min_length=3, max_length=128)
    source_key: Optional[str] = Field(None, min_length=2, max_length=64)
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    source_code: Optional[str] = Field(None, min_length=10)
    class_name: Optional[str] = Field(None, min_length=1, max_length=200)
    config: Optional[dict] = None
    config_schema: Optional[dict] = None
    aliases: Optional[list[str]] = None
    enabled: Optional[bool] = None
    unlock_system: bool = False


class UnifiedValidateRequest(BaseModel):
    source_code: str = Field(..., min_length=10)
    class_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers — extract config_schema for opportunity plugins
# ---------------------------------------------------------------------------


def _extract_config_schema_for_plugin(p: StrategyPlugin) -> Optional[dict]:
    if bool(p.is_system):
        for seed in SYSTEM_OPPORTUNITY_STRATEGY_SEEDS:
            if seed.slug == p.slug and seed.config_schema:
                return seed.config_schema
    cfg = p.config or {}
    if isinstance(cfg, dict) and "_schema" in cfg:
        return cfg["_schema"]
    for seed in SYSTEM_OPPORTUNITY_STRATEGY_SEEDS:
        if seed.slug == p.slug and seed.config_schema:
            return seed.config_schema
    return None


# ---------------------------------------------------------------------------
# Helpers — detect capabilities from source code
# ---------------------------------------------------------------------------


def _detect_capabilities(source_code: str) -> dict:
    """Inspect source code to determine which methods are implemented."""
    has_detect = bool(re.search(r"\bdef detect\s*\(", source_code))
    has_detect_async = bool(re.search(r"\basync\s+def detect_async\s*\(", source_code))
    has_evaluate = bool(re.search(r"\bdef evaluate\s*\(", source_code))
    has_should_exit = bool(re.search(r"\bdef should_exit\s*\(", source_code))
    return {
        "has_detect": has_detect,
        "has_detect_async": has_detect_async,
        "has_evaluate": has_evaluate,
        "has_should_exit": has_should_exit,
    }


def _infer_strategy_type(capabilities: dict) -> str:
    """Infer whether this is a detect, execute, or unified strategy."""
    has_any_detect = capabilities.get("has_detect") or capabilities.get("has_detect_async")
    has_evaluate = capabilities.get("has_evaluate")
    if has_any_detect and has_evaluate:
        return "unified"
    if has_evaluate:
        return "execute"
    return "detect"


# ---------------------------------------------------------------------------
# Serialisation — unified response from either table
# ---------------------------------------------------------------------------


def _plugin_to_unified(p: StrategyPlugin) -> dict:
    """Convert a StrategyPlugin ORM row to the unified response dict."""
    runtime = plugin_loader.get_status(p.slug)
    config = dict(p.config or {})
    config.pop("_schema", None)
    capabilities = _detect_capabilities(p.source_code or "")
    return {
        "id": p.id,
        "slug": p.slug,
        "source_key": p.source_key or "scanner",
        "name": p.name,
        "description": p.description,
        "source_code": p.source_code,
        "class_name": p.class_name,
        "is_system": bool(p.is_system),
        "enabled": bool(p.enabled),
        "status": p.status,
        "error_message": p.error_message,
        "version": int(p.version or 1),
        "config": config,
        "config_schema": _extract_config_schema_for_plugin(p),
        "strategy_type": _infer_strategy_type(capabilities),
        "capabilities": capabilities,
        "aliases": [],
        "sort_order": p.sort_order,
        "created_at": p.created_at.isoformat() if p.created_at else None,
        "updated_at": p.updated_at.isoformat() if p.updated_at else None,
        "table_source": "strategy_plugins",
        "runtime": runtime,
    }


def _trader_def_to_unified(row: TraderStrategyDefinition) -> dict:
    """Convert a TraderStrategyDefinition ORM row to the unified response dict."""
    runtime = strategy_db_loader.get_runtime_status(str(row.strategy_key or ""))
    capabilities = _detect_capabilities(row.source_code or "")
    return {
        "id": row.id,
        "slug": row.strategy_key,
        "source_key": row.source_key,
        "name": row.label,
        "description": row.description,
        "source_code": row.source_code,
        "class_name": row.class_name,
        "is_system": bool(row.is_system),
        "enabled": bool(row.enabled),
        "status": row.status,
        "error_message": row.error_message,
        "version": int(row.version or 1),
        "config": dict(row.default_params_json or {}),
        "config_schema": dict(row.param_schema_json or {}),
        "strategy_type": _infer_strategy_type(capabilities),
        "capabilities": capabilities,
        "aliases": list(row.aliases_json or []),
        "sort_order": 0,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        "table_source": "trader_strategy_definitions",
        "runtime": runtime,
    }


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


async def _find_in_plugins(strategy_id: str, session: AsyncSession) -> Optional[StrategyPlugin]:
    result = await session.execute(
        select(StrategyPlugin).where(StrategyPlugin.id == strategy_id)
    )
    return result.scalar_one_or_none()


async def _find_in_trader_defs(strategy_id: str, session: AsyncSession) -> Optional[TraderStrategyDefinition]:
    return await session.get(TraderStrategyDefinition, strategy_id)


async def _find_strategy(strategy_id: str, session: AsyncSession):
    """Find a strategy in either table, returning (row, table_source)."""
    plugin = await _find_in_plugins(strategy_id, session)
    if plugin is not None:
        return plugin, "strategy_plugins"
    trader_def = await _find_in_trader_defs(strategy_id, session)
    if trader_def is not None:
        return trader_def, "trader_strategy_definitions"
    return None, None


# ==================== ENDPOINTS ====================


@router.get("/template")
async def get_unified_template():
    """Return the unified strategy template (opportunity detection)."""
    return {
        "template": PLUGIN_TEMPLATE,
        "instructions": (
            "Create a class that extends BaseStrategy and implements detect() or "
            "detect_async(). For execution strategies, extend BaseTraderStrategy "
            "and implement evaluate(signal, context). Unified strategies can "
            "implement both detect/detect_async and evaluate/should_exit."
        ),
        "available_imports": [
            "models (Market, Event, ArbitrageOpportunity, StrategyType)",
            "services.strategies.base (BaseStrategy)",
            "services.trader_orchestrator.strategies.base (BaseTraderStrategy, StrategyDecision, DecisionCheck)",
            "services.strategies.* (built-in strategy modules)",
            "services.news.* (news strategy helpers)",
            "services.optimization.*",
            "services.ws_feeds",
            "services.chainlink_feed",
            "services.fee_model (fee_model)",
            "services.ai (get_llm_manager, LLMMessage, LLMResponse)",
            "services.strategy_sdk (StrategySDK)",
            "config (settings)",
            "math, statistics, collections, datetime, re, json, random, threading, asyncio, calendar, pathlib, etc.",
            "httpx",
            "numpy, scipy (if installed)",
        ],
    }


@router.get("/docs")
async def get_unified_docs():
    """Comprehensive merged documentation for both strategy types."""
    return {
        "overview": {
            "title": "Unified Strategy API Reference",
            "description": (
                "This API manages two families of strategy: opportunity-detection "
                "strategies (StrategyPlugin) that run inside the scanner, and "
                "trader-execution strategies (TraderStrategyDefinition) that the "
                "orchestrator evaluates for each trade signal. A unified strategy "
                "may implement both detect and evaluate methods."
            ),
        },
        "strategy_types": {
            "detect": (
                "Opportunity-detection strategies extend BaseStrategy and implement "
                "detect() or detect_async(). They run every scan cycle and return "
                "ArbitrageOpportunity objects. Stored in strategy_plugins."
            ),
            "execute": (
                "Trader-execution strategies extend BaseTraderStrategy and implement "
                "evaluate(signal, context). They decide whether a detected signal "
                "should be traded, at what size, and with what confidence. Stored in "
                "trader_strategy_definitions."
            ),
            "unified": (
                "A unified strategy implements both detect/detect_async and evaluate. "
                "It is stored in strategy_plugins (detect is primary) and can also "
                "act as an execution strategy."
            ),
        },
        "detect_strategy": {
            "base_class": "BaseStrategy",
            "required_method": "detect(events, markets, prices) -> list[ArbitrageOpportunity]",
            "async_variant": "detect_async(events, markets, prices) -> list[ArbitrageOpportunity]",
            "parameters": {
                "events": "list[Event] — All active Polymarket events",
                "markets": "list[Market] — All active markets across events",
                "prices": "dict[str, dict] — Live CLOB mid-prices { token_id: { mid, best_bid, best_ask } }",
            },
            "config_system": (
                "Define default_config on your class. Access merged config at runtime "
                "via self.config. Users override individual values in the UI."
            ),
        },
        "execute_strategy": {
            "base_class": "BaseTraderStrategy",
            "required_method": "evaluate(signal, context) -> StrategyDecision",
            "signal_fields": {
                "source": "str — Signal source (scanner, crypto, news, weather, traders)",
                "direction": "str — BUY or SELL",
                "edge_percent": "float — Estimated edge percentage",
                "confidence": "float — Confidence score (0-1)",
                "entry_price": "float — Suggested entry price",
                "liquidity": "float — Market liquidity in USD",
                "payload_json": "dict — Source-specific extra data",
            },
            "context_fields": {
                "params": "dict — Strategy parameters (defaults merged with UI overrides)",
                "trader": "ORM row — The trader being evaluated for",
                "mode": "str — 'paper' or 'live'",
            },
            "decision_fields": {
                "decision": "str — selected | skipped | blocked | failed",
                "reason": "str — Human-readable explanation",
                "score": "float — Ranking score",
                "size_usd": "float | None — Trade size in USD",
                "checks": "list[DecisionCheck] — Gate results",
            },
        },
        "allowed_imports_detect": [
            "models, services.strategies.base, services.strategies.*",
            "services.news.*, services.optimization.*, services.ws_feeds",
            "services.chainlink_feed, services.fee_model, services.ai",
            "services.strategy_sdk, services.weather.signal_engine",
            "config, httpx, asyncio, math, statistics, collections",
            "datetime, re, json, random, numpy, scipy",
        ],
        "allowed_imports_execute": [
            "services.trader_orchestrator.strategies.base",
            "math, statistics, collections, datetime, json, re",
            "typing, dataclasses, random",
        ],
        "blocked_imports": [
            "os, sys, subprocess, shutil — No filesystem or process access",
            "socket, http, urllib, requests, aiohttp — Use httpx or services layer",
            "pickle, marshal — No serialization",
            "multiprocessing — No process-based concurrency",
        ],
        "endpoints": {
            "GET /strategy-manager": "List all strategies from both tables with type/source_key/enabled filters",
            "GET /strategy-manager/template": "Starter template for writing a new strategy",
            "GET /strategy-manager/docs": "This documentation",
            "GET /strategy-manager/{id}": "Get one strategy by ID (searches both tables)",
            "POST /strategy-manager": "Create — auto-detects type from source code",
            "PUT /strategy-manager/{id}": "Update (partial)",
            "DELETE /strategy-manager/{id}": "Delete with tombstone for system strategies",
            "POST /strategy-manager/{id}/reload": "Force reload from stored source",
            "POST /strategy-manager/validate": "Validate source code without saving",
        },
    }


@router.post("/validate")
async def validate_unified_source(req: UnifiedValidateRequest):
    """Validate strategy source code without saving.

    Runs both opportunity-plugin and trader-strategy validators.
    """
    plugin_result = validate_plugin_source(req.source_code)
    trader_result = validate_trader_strategy_source(req.source_code, req.class_name)
    capabilities = _detect_capabilities(req.source_code)
    inferred_type = _infer_strategy_type(capabilities)

    # Choose the primary validation result based on inferred type
    if inferred_type == "execute":
        primary = trader_result
    else:
        primary = plugin_result

    return {
        "valid": primary.get("valid", False),
        "inferred_type": inferred_type,
        "capabilities": capabilities,
        "class_name": primary.get("class_name"),
        "strategy_name": plugin_result.get("strategy_name"),
        "strategy_description": plugin_result.get("strategy_description"),
        "errors": primary.get("errors", []),
        "warnings": primary.get("warnings", []),
        "plugin_validation": plugin_result,
        "trader_validation": trader_result,
    }


@router.get("")
async def list_strategies(
    type: Optional[str] = Query(
        default=None,
        description="Filter by strategy type: detect, execute, unified, all",
    ),
    source_key: Optional[str] = Query(default=None, description="Filter by source_key"),
    enabled: Optional[bool] = Query(default=None, description="Filter by enabled status"),
):
    """List ALL strategies from both tables in a unified format."""
    items: list[dict] = []

    async with AsyncSessionLocal() as session:
        # Seed system strategies to ensure they exist
        await ensure_system_opportunity_strategies_seeded(session)
        await ensure_system_trader_strategies_seeded(session)

        # --- Query StrategyPlugin (detect / unified) ---
        plugin_query = select(StrategyPlugin).order_by(
            StrategyPlugin.is_system.desc(),
            StrategyPlugin.sort_order.asc(),
            StrategyPlugin.name.asc(),
        )
        if source_key:
            plugin_query = plugin_query.where(
                StrategyPlugin.source_key == source_key.strip().lower()
            )
        if enabled is not None:
            plugin_query = plugin_query.where(StrategyPlugin.enabled == bool(enabled))

        plugins = (await session.execute(plugin_query)).scalars().all()
        for p in plugins:
            items.append(_plugin_to_unified(p))

        # --- Query TraderStrategyDefinition (execute) ---
        trader_query = select(TraderStrategyDefinition).order_by(
            TraderStrategyDefinition.source_key.asc(),
            TraderStrategyDefinition.strategy_key.asc(),
        )
        if source_key:
            trader_query = trader_query.where(
                TraderStrategyDefinition.source_key == source_key.strip().lower()
            )
        if enabled is not None:
            trader_query = trader_query.where(
                TraderStrategyDefinition.enabled == bool(enabled)
            )

        trader_rows = (await session.execute(trader_query)).scalars().all()
        for row in trader_rows:
            items.append(_trader_def_to_unified(row))

    # Apply type filter after both queries (capabilities require source inspection)
    if type and type != "all":
        items = [s for s in items if s["strategy_type"] == type]

    return {"items": items, "total": len(items)}


@router.get("/{strategy_id}")
async def get_strategy(strategy_id: str, session: AsyncSession = Depends(get_db_session)):
    """Get a single strategy by ID (searches both tables)."""
    await ensure_system_opportunity_strategies_seeded(session)
    await ensure_system_trader_strategies_seeded(session)

    row, table = await _find_strategy(strategy_id, session)
    if row is None:
        raise HTTPException(status_code=404, detail="Strategy not found")

    if table == "strategy_plugins":
        return _plugin_to_unified(row)
    return _trader_def_to_unified(row)


@router.post("")
async def create_strategy(req: UnifiedStrategyCreateRequest):
    """Create a strategy, auto-detecting the target table from source code."""
    slug = _validate_slug(req.slug)
    source_key = str(req.source_key or "scanner").strip().lower()

    capabilities = _detect_capabilities(req.source_code)
    inferred_type = _infer_strategy_type(capabilities)

    # Route to the appropriate table
    if inferred_type == "execute":
        return await _create_trader_strategy(req, slug, source_key, capabilities)
    else:
        # "detect" or "unified" -> strategy_plugins (detect is primary)
        return await _create_plugin_strategy(req, slug, source_key, capabilities)


async def _create_plugin_strategy(
    req: UnifiedStrategyCreateRequest,
    slug: str,
    source_key: str,
    capabilities: dict,
) -> dict:
    """Insert into strategy_plugins."""
    async with AsyncSessionLocal() as session:
        existing = await session.execute(
            select(StrategyPlugin).where(StrategyPlugin.slug == slug)
        )
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=409, detail=f"A strategy with slug '{slug}' already exists.")

    validation = validate_plugin_source(req.source_code)
    if not validation["valid"]:
        raise HTTPException(
            status_code=400,
            detail={"message": "Strategy validation failed", "errors": validation["errors"]},
        )

    strategy_name = (
        req.name or validation["strategy_name"] or slug.replace("_", " ").title()
    ).strip()
    strategy_description = req.description if req.description is not None else validation["strategy_description"]
    class_name = validation["class_name"]

    plugin_id = str(uuid.uuid4())
    status = "unloaded"
    error_message = None

    if req.enabled:
        try:
            plugin_loader.load_plugin(slug, req.source_code, req.config or None)
            status = "loaded"
        except PluginValidationError as e:
            status = "error"
            error_message = str(e)

    async with AsyncSessionLocal() as session:
        plugin = StrategyPlugin(
            id=plugin_id,
            slug=slug,
            source_key=source_key,
            name=strategy_name,
            description=strategy_description,
            source_code=req.source_code,
            class_name=class_name,
            is_system=False,
            enabled=req.enabled,
            status=status,
            error_message=error_message,
            config=req.config or {},
            version=1,
            sort_order=0,
        )
        session.add(plugin)
        await session.commit()
        await session.refresh(plugin)
        return _plugin_to_unified(plugin)


async def _create_trader_strategy(
    req: UnifiedStrategyCreateRequest,
    slug: str,
    source_key: str,
    capabilities: dict,
) -> dict:
    """Insert into trader_strategy_definitions."""
    async with AsyncSessionLocal() as session:
        from sqlalchemy import func

        existing = await session.execute(
            select(TraderStrategyDefinition.id).where(
                func.lower(TraderStrategyDefinition.strategy_key) == slug.lower()
            )
        )
        if existing.scalar_one_or_none() is not None:
            raise HTTPException(
                status_code=409,
                detail=f"A strategy with key '{slug}' already exists.",
            )

        class_name_to_use = req.class_name or slug.replace("_", " ").title().replace(" ", "")
        validation = validate_trader_strategy_source(req.source_code, class_name_to_use)
        if not validation.get("valid"):
            raise HTTPException(
                status_code=422,
                detail={"message": "Strategy validation failed", "errors": validation.get("errors", [])},
            )
        class_name_to_use = str(validation.get("class_name") or class_name_to_use).strip()

        row = TraderStrategyDefinition(
            id=uuid.uuid4().hex,
            strategy_key=slug,
            source_key=source_key,
            label=str(req.name or slug.replace("_", " ").title()).strip(),
            description=req.description,
            class_name=class_name_to_use,
            source_code=req.source_code,
            default_params_json=dict(req.config or {}),
            param_schema_json=dict(req.config_schema or {}),
            aliases_json=list(req.aliases or []),
            is_system=False,
            enabled=bool(req.enabled),
            status="unloaded",
            error_message=None,
            version=1,
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)

        if row.enabled:
            await strategy_db_loader.reload_strategy(row.strategy_key, session=session)
            row = await session.get(TraderStrategyDefinition, row.id)

        return _trader_def_to_unified(row)


@router.put("/{strategy_id}")
async def update_strategy(strategy_id: str, req: UnifiedStrategyUpdateRequest):
    """Update a strategy in whichever table it belongs to."""
    async with AsyncSessionLocal() as session:
        row, table = await _find_strategy(strategy_id, session)
        if row is None:
            raise HTTPException(status_code=404, detail="Strategy not found")

        if table == "strategy_plugins":
            return await _update_plugin(row, req, session)
        else:
            return await _update_trader_def(row, req, session)


async def _update_plugin(
    plugin: StrategyPlugin,
    req: UnifiedStrategyUpdateRequest,
    session: AsyncSession,
) -> dict:
    """Update a StrategyPlugin row."""
    if bool(plugin.is_system) and not req.unlock_system:
        raise HTTPException(
            status_code=403,
            detail="System strategies are read-only. Set unlock_system=true for admin override.",
        )

    original_slug = plugin.slug
    code_changed = False
    slug_changed = False

    if req.slug is not None:
        next_slug = _validate_slug(req.slug)
        if next_slug != plugin.slug:
            existing_slug = await session.execute(
                select(StrategyPlugin.id).where(
                    StrategyPlugin.slug == next_slug,
                    StrategyPlugin.id != plugin.id,
                )
            )
            if existing_slug.scalar_one_or_none():
                raise HTTPException(status_code=409, detail=f"Slug '{next_slug}' already exists.")
            plugin.slug = next_slug
            slug_changed = True

    if req.source_code is not None and req.source_code != plugin.source_code:
        validation = validate_plugin_source(req.source_code)
        if not validation["valid"]:
            raise HTTPException(
                status_code=400,
                detail={"message": "Validation failed", "errors": validation["errors"]},
            )
        plugin.source_code = req.source_code
        plugin.class_name = validation["class_name"]
        if req.name is None and validation["strategy_name"]:
            plugin.name = validation["strategy_name"]
        if req.description is None and validation["strategy_description"]:
            plugin.description = validation["strategy_description"]
        plugin.version += 1
        code_changed = True

    if req.config is not None:
        plugin.config = req.config
        code_changed = True

    if req.source_key is not None:
        plugin.source_key = str(req.source_key or "scanner").strip().lower()
    if req.name is not None:
        plugin.name = req.name
    if req.description is not None:
        plugin.description = req.description

    enabled_changed = False
    if req.enabled is not None and req.enabled != plugin.enabled:
        plugin.enabled = req.enabled
        enabled_changed = True

    if enabled_changed or code_changed or slug_changed:
        if slug_changed:
            plugin_loader.unload_plugin(original_slug)
        if plugin.enabled:
            try:
                plugin_loader.load_plugin(plugin.slug, plugin.source_code, plugin.config or None)
                plugin.status = "loaded"
                plugin.error_message = None
            except PluginValidationError as e:
                plugin.status = "error"
                plugin.error_message = str(e)
        else:
            plugin_loader.unload_plugin(plugin.slug)
            plugin.status = "unloaded"
            plugin.error_message = None

    await session.commit()
    await session.refresh(plugin)
    return _plugin_to_unified(plugin)


async def _update_trader_def(
    row: TraderStrategyDefinition,
    req: UnifiedStrategyUpdateRequest,
    session: AsyncSession,
) -> dict:
    """Update a TraderStrategyDefinition row."""
    if bool(row.is_system) and not req.unlock_system:
        raise HTTPException(
            status_code=403,
            detail="System strategies are read-only. Clone first or set unlock_system=true.",
        )

    from sqlalchemy import func

    next_key = str(req.slug or row.strategy_key).strip().lower()
    if next_key != str(row.strategy_key or "").strip().lower():
        exists = await session.execute(
            select(TraderStrategyDefinition.id).where(
                func.lower(TraderStrategyDefinition.strategy_key) == next_key,
                TraderStrategyDefinition.id != row.id,
            )
        )
        if exists.scalar_one_or_none() is not None:
            raise HTTPException(status_code=409, detail=f"strategy_key '{next_key}' already exists.")
        row.strategy_key = next_key

    next_class_name = str(req.class_name or row.class_name or "").strip()
    next_source_code = str(req.source_code or row.source_code or "")
    if req.class_name is not None or req.source_code is not None:
        validation = validate_trader_strategy_source(next_source_code, next_class_name)
        if not validation.get("valid"):
            raise HTTPException(status_code=422, detail={"errors": validation.get("errors", [])})
        next_class_name = str(validation.get("class_name") or next_class_name).strip()

    if req.source_key is not None:
        row.source_key = str(req.source_key).strip().lower()
    if req.name is not None:
        row.label = str(req.name).strip() or row.label
    if req.description is not None:
        row.description = req.description
    row.class_name = next_class_name or row.class_name
    if req.source_code is not None:
        row.source_code = next_source_code
    if req.config is not None:
        row.default_params_json = dict(req.config)
    if req.config_schema is not None:
        row.param_schema_json = dict(req.config_schema)
    if req.aliases is not None:
        row.aliases_json = list(req.aliases)
    if req.enabled is not None:
        row.enabled = bool(req.enabled)
    row.version = int(row.version or 1) + 1
    row.status = "unloaded"
    row.error_message = None

    await session.commit()
    await session.refresh(row)

    await strategy_db_loader.reload_strategy(row.strategy_key, session=session)
    row = await session.get(TraderStrategyDefinition, row.id)
    return _trader_def_to_unified(row)


@router.delete("/{strategy_id}")
async def delete_strategy(strategy_id: str):
    """Delete a strategy (with tombstone for system opportunity strategies)."""
    async with AsyncSessionLocal() as session:
        row, table = await _find_strategy(strategy_id, session)
        if row is None:
            raise HTTPException(status_code=404, detail="Strategy not found")

        if table == "strategy_plugins":
            plugin: StrategyPlugin = row
            if bool(plugin.is_system):
                tombstone = await session.get(StrategyPluginTombstone, plugin.slug)
                if tombstone is None:
                    session.add(
                        StrategyPluginTombstone(
                            slug=plugin.slug,
                            deleted_at=datetime.utcnow(),
                            reason="user_deleted_system_strategy",
                        )
                    )
                else:
                    tombstone.deleted_at = datetime.utcnow()
                    tombstone.reason = "user_deleted_system_strategy"
            plugin_loader.unload_plugin(plugin.slug)
            await session.delete(plugin)
        else:
            trader_def: TraderStrategyDefinition = row
            key = str(trader_def.strategy_key or "").strip().lower()
            # Remove from loader caches (no public unload method exists)
            strategy_db_loader._loaded.pop(key, None)
            strategy_db_loader._errors.pop(key, None)
            strategy_db_loader._reset_modules_for_key(key)
            # Remove any aliases pointing to this key
            for alias, target in list(strategy_db_loader._aliases.items()):
                if target == key:
                    strategy_db_loader._aliases.pop(alias, None)
            await session.delete(trader_def)

        await session.commit()

    return {"status": "success", "message": "Strategy deleted"}


@router.post("/{strategy_id}/reload")
async def reload_strategy(strategy_id: str):
    """Force reload a strategy from its stored source code."""
    async with AsyncSessionLocal() as session:
        row, table = await _find_strategy(strategy_id, session)
        if row is None:
            raise HTTPException(status_code=404, detail="Strategy not found")

        if table == "strategy_plugins":
            plugin: StrategyPlugin = row
            if not plugin.enabled:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot reload a disabled strategy. Enable it first.",
                )
            try:
                plugin_loader.load_plugin(plugin.slug, plugin.source_code, plugin.config or None)
                plugin.status = "loaded"
                plugin.error_message = None
                await session.commit()
                return {
                    "status": "success",
                    "message": f"Strategy '{plugin.slug}' reloaded",
                    "runtime": plugin_loader.get_status(plugin.slug),
                }
            except PluginValidationError as e:
                plugin.status = "error"
                plugin.error_message = str(e)
                await session.commit()
                raise HTTPException(
                    status_code=400,
                    detail={"message": f"Reload failed for '{plugin.slug}'", "error": str(e)},
                )
        else:
            trader_def: TraderStrategyDefinition = row
            result = await strategy_db_loader.reload_strategy(
                trader_def.strategy_key, session=session,
            )
            refreshed = await session.get(TraderStrategyDefinition, strategy_id)
            return {
                "status": "success",
                "reload": result,
                "strategy": _trader_def_to_unified(refreshed),
            }
