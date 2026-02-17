"""Unified Strategy API Routes

All strategies (detection and execution) live in a single `strategies` table.
This router provides CRUD, validation, reload, template, and docs endpoints.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    AsyncSessionLocal,
    Strategy,
    StrategyPluginTombstone,
    get_db_session,
)
from services.opportunity_strategy_catalog import (
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
    """Create a strategy in the strategies table."""

    slug: str = Field(..., min_length=3, max_length=128, description="Unique identifier")
    source_key: str = Field(default="scanner", min_length=2, max_length=64)
    name: Optional[str] = Field(None, min_length=1, max_length=200, description="Display name")
    description: Optional[str] = Field(None, max_length=500)
    source_code: str = Field(..., min_length=10)
    class_name: Optional[str] = Field(None, min_length=1, max_length=200)
    config: dict = Field(default_factory=dict, description="Config / default params")
    config_schema: dict = Field(default_factory=dict, description="Param schema for UI form")
    aliases: list[str] = Field(default_factory=list)
    enabled: bool = True


class UnifiedStrategyUpdateRequest(BaseModel):
    """Partial update for a strategy."""

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
# Serialisation — unified response from Strategy table
# ---------------------------------------------------------------------------


def _strategy_to_dict(row: Strategy) -> dict:
    """Convert a Strategy ORM row to the API response dict."""
    capabilities = _detect_capabilities(row.source_code or "")
    return {
        "id": row.id,
        "slug": row.slug,
        "source_key": row.source_key or "scanner",
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
        "strategy_type": _infer_strategy_type(capabilities),
        "capabilities": capabilities,
        "aliases": list(row.aliases or []),
        "sort_order": row.sort_order or 0,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        "runtime": None,  # Will be populated by strategy_loader later
    }


# ==================== ENDPOINTS ====================


@router.get("/template")
async def get_unified_template():
    """Return the unified strategy template."""
    return {
        "template": PLUGIN_TEMPLATE,
        "instructions": (
            "Create a class that extends BaseStrategy and implements detect() or "
            "detect_async() for opportunity detection. For execution strategies, "
            "implement evaluate(signal, context). Unified strategies can implement "
            "both detect/detect_async and evaluate/should_exit."
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
    """Comprehensive documentation for the unified strategy system."""
    return {
        "overview": {
            "title": "Unified Strategy API Reference",
            "description": (
                "All strategies live in a single `strategies` table. Each strategy "
                "is a complete Python class with optional detect(), evaluate(), and "
                "should_exit() methods. Strategies are classified as 'detect', "
                "'execute', or 'unified' based on which methods they implement."
            ),
        },
        "strategy_types": {
            "detect": (
                "Opportunity-detection strategies extend BaseStrategy and implement "
                "detect() or detect_async(). They run every scan cycle and return "
                "ArbitrageOpportunity objects."
            ),
            "execute": (
                "Trader-execution strategies extend BaseTraderStrategy and implement "
                "evaluate(signal, context). They decide whether a detected signal "
                "should be traded, at what size, and with what confidence."
            ),
            "unified": (
                "A unified strategy implements both detect/detect_async and evaluate. "
                "It handles end-to-end: finding opportunities and deciding how to trade them."
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
            "GET /strategies": "List all strategies with type/source_key/enabled filters",
            "GET /strategies/template": "Starter template for writing a new strategy",
            "GET /strategies/docs": "This documentation",
            "GET /strategies/{id}": "Get one strategy by ID",
            "POST /strategies": "Create a new strategy",
            "PUT /strategies/{id}": "Update (partial)",
            "DELETE /strategies/{id}": "Delete with tombstone for system strategies",
            "POST /strategies/{id}/reload": "Force reload from stored source",
            "POST /strategies/validate": "Validate source code without saving",
        },
    }


@router.post("/validate")
async def validate_unified_source(req: UnifiedValidateRequest):
    """Validate strategy source code without saving."""
    plugin_result = validate_plugin_source(req.source_code)
    capabilities = _detect_capabilities(req.source_code)
    inferred_type = _infer_strategy_type(capabilities)

    return {
        "valid": plugin_result.get("valid", False),
        "inferred_type": inferred_type,
        "capabilities": capabilities,
        "class_name": plugin_result.get("class_name"),
        "strategy_name": plugin_result.get("strategy_name"),
        "strategy_description": plugin_result.get("strategy_description"),
        "errors": plugin_result.get("errors", []),
        "warnings": plugin_result.get("warnings", []),
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
    """List all strategies from the unified strategies table."""
    async with AsyncSessionLocal() as session:
        # Seed system strategies to ensure they exist
        await ensure_system_opportunity_strategies_seeded(session)
        await ensure_system_trader_strategies_seeded(session)

        query = select(Strategy).order_by(
            Strategy.is_system.desc(),
            Strategy.sort_order.asc(),
            Strategy.name.asc(),
        )
        if source_key:
            query = query.where(Strategy.source_key == source_key.strip().lower())
        if enabled is not None:
            query = query.where(Strategy.enabled == bool(enabled))

        rows = (await session.execute(query)).scalars().all()
        items = [_strategy_to_dict(row) for row in rows]

    # Apply type filter after query (capabilities require source inspection)
    if type and type != "all":
        items = [s for s in items if s["strategy_type"] == type]

    return {"items": items, "total": len(items)}


@router.get("/{strategy_id}")
async def get_strategy(strategy_id: str, session: AsyncSession = Depends(get_db_session)):
    """Get a single strategy by ID."""
    await ensure_system_opportunity_strategies_seeded(session)
    await ensure_system_trader_strategies_seeded(session)

    row = await session.get(Strategy, strategy_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Strategy not found")

    return _strategy_to_dict(row)


@router.post("")
async def create_strategy(req: UnifiedStrategyCreateRequest):
    """Create a new strategy."""
    slug = _validate_slug(req.slug)
    source_key = str(req.source_key or "scanner").strip().lower()

    # Validate source code
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
    class_name = req.class_name or validation["class_name"]

    strategy_id = uuid.uuid4().hex
    status = "unloaded"
    error_message = None

    async with AsyncSessionLocal() as session:
        existing = await session.execute(
            select(Strategy).where(Strategy.slug == slug)
        )
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=409, detail=f"A strategy with slug '{slug}' already exists.")

        if req.enabled:
            try:
                plugin_loader.load_plugin(slug, req.source_code, req.config or None)
                status = "loaded"
            except PluginValidationError as e:
                status = "error"
                error_message = str(e)

        row = Strategy(
            id=strategy_id,
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
            config_schema=req.config_schema or {},
            aliases=req.aliases or [],
            version=1,
            sort_order=0,
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return _strategy_to_dict(row)


@router.put("/{strategy_id}")
async def update_strategy(strategy_id: str, req: UnifiedStrategyUpdateRequest):
    """Update a strategy."""
    async with AsyncSessionLocal() as session:
        row = await session.get(Strategy, strategy_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Strategy not found")

        if bool(row.is_system) and not req.unlock_system:
            raise HTTPException(
                status_code=403,
                detail="System strategies are read-only. Set unlock_system=true for admin override.",
            )

        original_slug = row.slug
        code_changed = False
        slug_changed = False

        if req.slug is not None:
            next_slug = _validate_slug(req.slug)
            if next_slug != row.slug:
                existing_slug = await session.execute(
                    select(Strategy.id).where(
                        Strategy.slug == next_slug,
                        Strategy.id != row.id,
                    )
                )
                if existing_slug.scalar_one_or_none():
                    raise HTTPException(status_code=409, detail=f"Slug '{next_slug}' already exists.")
                row.slug = next_slug
                slug_changed = True

        if req.source_code is not None and req.source_code != row.source_code:
            validation = validate_plugin_source(req.source_code)
            if not validation["valid"]:
                raise HTTPException(
                    status_code=400,
                    detail={"message": "Validation failed", "errors": validation["errors"]},
                )
            row.source_code = req.source_code
            row.class_name = validation["class_name"]
            if req.name is None and validation["strategy_name"]:
                row.name = validation["strategy_name"]
            if req.description is None and validation["strategy_description"]:
                row.description = validation["strategy_description"]
            row.version = int(row.version or 1) + 1
            code_changed = True

        if req.config is not None:
            row.config = req.config
            code_changed = True
        if req.config_schema is not None:
            row.config_schema = req.config_schema
        if req.aliases is not None:
            row.aliases = list(req.aliases)

        if req.source_key is not None:
            row.source_key = str(req.source_key or "scanner").strip().lower()
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
                plugin_loader.unload_plugin(original_slug)
            if row.enabled:
                try:
                    plugin_loader.load_plugin(row.slug, row.source_code, row.config or None)
                    row.status = "loaded"
                    row.error_message = None
                except PluginValidationError as e:
                    row.status = "error"
                    row.error_message = str(e)
            else:
                plugin_loader.unload_plugin(row.slug)
                row.status = "unloaded"
                row.error_message = None

        await session.commit()
        await session.refresh(row)
        return _strategy_to_dict(row)


@router.delete("/{strategy_id}")
async def delete_strategy(strategy_id: str):
    """Delete a strategy (with tombstone for system strategies)."""
    async with AsyncSessionLocal() as session:
        row = await session.get(Strategy, strategy_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Strategy not found")

        if bool(row.is_system):
            tombstone = await session.get(StrategyPluginTombstone, row.slug)
            if tombstone is None:
                session.add(
                    StrategyPluginTombstone(
                        slug=row.slug,
                        deleted_at=datetime.utcnow(),
                        reason="user_deleted_system_strategy",
                    )
                )
            else:
                tombstone.deleted_at = datetime.utcnow()
                tombstone.reason = "user_deleted_system_strategy"

        plugin_loader.unload_plugin(row.slug)
        await session.delete(row)
        await session.commit()

    return {"status": "success", "message": "Strategy deleted"}


@router.post("/{strategy_id}/reload")
async def reload_strategy(strategy_id: str):
    """Force reload a strategy from its stored source code."""
    async with AsyncSessionLocal() as session:
        row = await session.get(Strategy, strategy_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Strategy not found")

        if not row.enabled:
            raise HTTPException(
                status_code=400,
                detail="Cannot reload a disabled strategy. Enable it first.",
            )

        try:
            plugin_loader.load_plugin(row.slug, row.source_code, row.config or None)
            row.status = "loaded"
            row.error_message = None
            await session.commit()
            return {
                "status": "success",
                "message": f"Strategy '{row.slug}' reloaded",
                "runtime": plugin_loader.get_status(row.slug),
            }
        except PluginValidationError as e:
            row.status = "error"
            row.error_message = str(e)
            await session.commit()
            raise HTTPException(
                status_code=400,
                detail={"message": f"Reload failed for '{row.slug}'", "error": str(e)},
            )
