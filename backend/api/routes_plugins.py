"""
Plugin API Routes

Full CRUD for code-based strategy plugins. Each plugin is a Python file
defining a BaseStrategy subclass with custom detection logic — a real strategy,
not just a grouping of existing ones.
"""

import re
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from sqlalchemy import select
from models.database import AsyncSessionLocal, StrategyPlugin
from services.plugin_loader import (
    plugin_loader,
    validate_plugin_source,
    PLUGIN_TEMPLATE,
    PluginValidationError,
)
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/plugins", tags=["Plugins"])


# ==================== SLUG VALIDATION ====================

_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]{1,48}[a-z0-9]$")


def _validate_slug(slug: str) -> str:
    """Validate and normalize a plugin slug."""
    slug = slug.strip().lower()
    if not _SLUG_RE.match(slug):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid slug '{slug}'. Slugs must be 3-50 characters, "
                f"start with a letter, use only lowercase letters, numbers, "
                f"and underscores, and end with a letter or number."
            ),
        )
    return slug


# ==================== REQUEST/RESPONSE MODELS ====================


class PluginCreateRequest(BaseModel):
    """Request to create a new plugin."""

    slug: str = Field(..., min_length=3, max_length=50, description="Unique slug identifier")
    source_code: str = Field(..., min_length=10, description="Python source code")
    config: dict = Field(default_factory=dict, description="Config overrides for the plugin")
    enabled: bool = True


class PluginUpdateRequest(BaseModel):
    """Request to update a plugin (partial)."""

    source_code: Optional[str] = Field(None, min_length=10)
    config: Optional[dict] = None
    enabled: Optional[bool] = None
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)


class PluginValidateRequest(BaseModel):
    """Request to validate plugin source code without saving."""

    source_code: str = Field(..., min_length=10)


class PluginResponse(BaseModel):
    """Plugin response model."""

    id: str
    slug: str
    name: str
    description: Optional[str]
    source_code: str
    class_name: Optional[str]
    enabled: bool
    status: str  # unloaded, loaded, error
    error_message: Optional[str]
    config: dict
    version: int
    sort_order: int
    created_at: Optional[str]
    updated_at: Optional[str]
    # Runtime info (from loader, if loaded)
    runtime: Optional[dict] = None


def _plugin_to_response(p: StrategyPlugin) -> PluginResponse:
    """Convert a DB StrategyPlugin to a PluginResponse."""
    runtime = plugin_loader.get_status(p.slug)
    return PluginResponse(
        id=p.id,
        slug=p.slug,
        name=p.name,
        description=p.description,
        source_code=p.source_code,
        class_name=p.class_name,
        enabled=p.enabled,
        status=p.status,
        error_message=p.error_message,
        config=p.config or {},
        version=p.version,
        sort_order=p.sort_order,
        created_at=p.created_at.isoformat() if p.created_at else None,
        updated_at=p.updated_at.isoformat() if p.updated_at else None,
        runtime=runtime,
    )


# ==================== ENDPOINTS ====================


@router.get("/template")
async def get_plugin_template():
    """Get the starter template for writing a new plugin."""
    return {
        "template": PLUGIN_TEMPLATE,
        "instructions": (
            "Create a class that extends BaseStrategy and implements detect(). "
            "Your detect() method receives events, markets, and live prices on "
            "every scan cycle. Use self.create_opportunity() to build opportunities "
            "with automatic fee calculation, hard filters, and risk scoring. "
            "Access your plugin's config via self.config."
        ),
        "available_imports": [
            "models (Market, Event, ArbitrageOpportunity, StrategyType)",
            "services.strategies.base (BaseStrategy)",
            "services.fee_model (fee_model)",
            "config (settings)",
            "math, statistics, collections, datetime, re, json, random, etc.",
            "numpy, scipy (if installed)",
        ],
    }


@router.get("/docs")
async def get_plugin_docs():
    """Get comprehensive API documentation for plugin authors."""
    return {
        "overview": {
            "title": "Plugin API Reference",
            "description": (
                "Each plugin is a Python class that extends BaseStrategy. "
                "Your class must implement detect() which is called every scan cycle "
                "with the full set of active markets, events, and live prices. "
                "Return a list of ArbitrageOpportunity objects for any opportunities found."
            ),
        },
        "class_structure": {
            "description": "Your plugin class must extend BaseStrategy",
            "required_attributes": {
                "name": "str — Human-readable name shown in the UI",
                "description": "str — Short description shown in the strategy list",
            },
            "optional_attributes": {
                "default_config": "dict — Default config values (users can override in the UI)",
            },
            "auto_set_attributes": {
                "strategy_type": "str — Automatically set to your plugin's slug by the loader",
            },
            "inherited_attributes": {
                "self.fee": "float — Current Polymarket fee rate (e.g. 0.02 = 2%)",
                "self.min_profit": "float — Minimum profit threshold from app settings",
                "self.config": "dict — Your default_config merged with user overrides from the UI",
            },
        },
        "detect_method": {
            "signature": "def detect(self, events: list[Event], markets: list[Market], prices: dict[str, dict]) -> list[ArbitrageOpportunity]",
            "description": "Called every scan cycle. Return detected opportunities.",
            "parameters": {
                "events": {
                    "type": "list[Event]",
                    "description": "All active Polymarket events",
                    "fields": {
                        "id": "str — Event ID",
                        "slug": "str — URL slug",
                        "title": "str — Event title",
                        "description": "str — Event description",
                        "category": "str | None — Category (e.g. 'politics', 'sports', 'crypto')",
                        "markets": "list[Market] — Markets belonging to this event",
                        "neg_risk": "bool — Whether this is a NegRisk event",
                        "active": "bool — Whether the event is active",
                        "closed": "bool — Whether the event is closed",
                    },
                },
                "markets": {
                    "type": "list[Market]",
                    "description": "All active markets across all events (flattened). This is the primary input.",
                    "fields": {
                        "id": "str — Market ID",
                        "condition_id": "str — Condition ID for the CLOB",
                        "question": "str — The market question (e.g. 'Will BTC go up in the next 15 minutes?')",
                        "slug": "str — URL slug",
                        "tokens": "list[Token] — YES/NO tokens with token_id, outcome, price",
                        "clob_token_ids": "list[str] — [yes_token_id, no_token_id] for CLOB lookups",
                        "outcome_prices": "list[float] — [yes_price, no_price] from the API",
                        "active": "bool — Whether the market is active",
                        "closed": "bool — Whether the market is closed/resolved",
                        "neg_risk": "bool — Whether this is a NegRisk market",
                        "volume": "float — Total trading volume in USD",
                        "liquidity": "float — Current liquidity in USD",
                        "end_date": "datetime | None — When the market resolves",
                        "platform": "str — 'polymarket' or 'kalshi'",
                        "yes_price": "float (property) — Shortcut for outcome_prices[0]",
                        "no_price": "float (property) — Shortcut for outcome_prices[1]",
                    },
                },
                "prices": {
                    "type": "dict[str, dict]",
                    "description": "Live CLOB mid-prices keyed by token ID. Use these for the most current prices.",
                    "structure": "{ token_id: { 'mid': float, 'best_bid': float, 'best_ask': float } }",
                    "usage": (
                        "Look up live prices using market.clob_token_ids: "
                        "prices[market.clob_token_ids[0]] gives the YES token's live price data."
                    ),
                },
            },
            "returns": "list[ArbitrageOpportunity] — Use self.create_opportunity() to build these",
        },
        "create_opportunity_method": {
            "signature": (
                "self.create_opportunity(title, description, total_cost, markets, positions, "
                "event=None, expected_payout=1.0, is_guaranteed=True, "
                "vwap_total_cost=None, spread_bps=None, fill_probability=None)"
            ),
            "description": (
                "Builds an ArbitrageOpportunity with automatic fee calculation, risk scoring, "
                "and hard rejection filters. Returns None if the opportunity doesn't meet "
                "minimum thresholds (ROI, liquidity, position size, etc.). Use this instead of "
                "constructing ArbitrageOpportunity directly."
            ),
            "parameters": {
                "title": "str — Short title for the opportunity (shown in the UI)",
                "description": "str — Detailed description of what was detected",
                "total_cost": "float — Total cost to enter the position (e.g. YES + NO combined price)",
                "markets": "list[Market] — The market(s) involved in this opportunity",
                "positions": (
                    "list[dict] — Positions to take. Each dict should have: "
                    "{'action': 'BUY', 'outcome': 'YES'|'NO', 'price': float, 'token_id': str}"
                ),
                "event": "Event | None — The parent event (optional, used for category/metadata)",
                "expected_payout": "float — Expected payout if the trade succeeds (default $1.00)",
                "is_guaranteed": (
                    "bool — True for structural arbitrage (guaranteed profit), "
                    "False for directional/statistical bets"
                ),
                "vwap_total_cost": "float | None — VWAP-adjusted realistic cost from order book analysis",
                "spread_bps": "float | None — Actual spread in basis points",
                "fill_probability": "float | None — Probability that all legs fill (0-1)",
            },
            "hard_filters_applied": [
                "ROI must exceed MIN_PROFIT_THRESHOLD",
                "ROI must be below MAX_PLAUSIBLE_ROI (for guaranteed strategies)",
                "Min liquidity per market must exceed MIN_LIQUIDITY_HARD",
                "Max position size must exceed MIN_POSITION_SIZE",
                "Absolute profit on max position must exceed MIN_ABSOLUTE_PROFIT",
                "Annualized ROI must exceed MIN_ANNUALIZED_ROI (if resolution date known)",
                "Resolution must be within MAX_RESOLUTION_MONTHS",
                "Total legs must be <= MAX_TRADE_LEGS",
            ],
        },
        "config_system": {
            "description": (
                "Define a default_config dict on your class with default values. "
                "Users can override individual values via the plugin config in the UI. "
                "Access merged config at runtime via self.config."
            ),
            "example": (
                "class MyStrategy(BaseStrategy):\n"
                "    default_config = {\n"
                "        'min_spread': 0.03,\n"
                "        'target_categories': ['crypto', 'politics'],\n"
                "        'max_legs': 4,\n"
                "    }\n\n"
                "    def detect(self, events, markets, prices):\n"
                "        threshold = self.config.get('min_spread', 0.03)\n"
                "        categories = self.config.get('target_categories', [])"
            ),
        },
        "risk_scoring": {
            "description": (
                "self.calculate_risk_score(markets, resolution_date) returns (score, factors). "
                "This is called automatically by create_opportunity(), but you can also call "
                "it yourself for custom risk analysis."
            ),
            "risk_factors_considered": [
                "Time to resolution (very short < 2d, short < 7d, long > 180d)",
                "Liquidity level (low < $1,000, moderate < $5,000)",
                "Number of legs (slippage compounds per leg)",
                "Multi-leg execution risk (partial fill probability)",
            ],
        },
        "common_patterns": {
            "get_live_prices": (
                "yes_price = market.yes_price\n"
                "no_price = market.no_price\n"
                "if market.clob_token_ids and len(market.clob_token_ids) > 0:\n"
                "    token = market.clob_token_ids[0]\n"
                "    if token in prices:\n"
                "        yes_price = prices[token].get('mid', yes_price)\n"
                "if market.clob_token_ids and len(market.clob_token_ids) > 1:\n"
                "    token = market.clob_token_ids[1]\n"
                "    if token in prices:\n"
                "        no_price = prices[token].get('mid', no_price)"
            ),
            "filter_markets": (
                "for market in markets:\n"
                "    if market.closed or not market.active:\n"
                "        continue\n"
                "    if len(market.outcome_prices) != 2:\n"
                "        continue  # Skip non-binary markets"
            ),
            "build_positions": (
                "positions = [\n"
                "    {'action': 'BUY', 'outcome': 'YES', 'price': yes_price,\n"
                "     'token_id': market.clob_token_ids[0] if market.clob_token_ids else None},\n"
                "    {'action': 'BUY', 'outcome': 'NO', 'price': no_price,\n"
                "     'token_id': market.clob_token_ids[1] if len(market.clob_token_ids) > 1 else None},\n"
                "]"
            ),
            "find_event_for_market": (
                "# Match a market back to its parent event\n"
                "event = next((e for e in events if any(m.id == market.id for m in e.markets)), None)"
            ),
        },
        "allowed_imports": [
            {"module": "models", "items": "Market, Event, ArbitrageOpportunity, StrategyType"},
            {"module": "services.strategies.base", "items": "BaseStrategy"},
            {"module": "services.fee_model", "items": "fee_model"},
            {"module": "config", "items": "settings (app configuration)"},
            {"module": "math", "items": "Standard math functions"},
            {"module": "statistics", "items": "Statistical functions (mean, stdev, etc.)"},
            {"module": "collections", "items": "defaultdict, Counter, deque, etc."},
            {"module": "datetime", "items": "datetime, timedelta, timezone"},
            {"module": "re", "items": "Regular expressions"},
            {"module": "json", "items": "JSON parsing"},
            {"module": "random", "items": "Random number generation"},
            {"module": "itertools", "items": "Itertools combinatorics"},
            {"module": "functools", "items": "Functional programming utilities"},
            {"module": "dataclasses", "items": "Dataclass decorators"},
            {"module": "enum", "items": "Enum definitions"},
            {"module": "typing", "items": "Type hints"},
            {"module": "numpy", "items": "Numerical computing (if installed)"},
            {"module": "scipy", "items": "Scientific computing (if installed)"},
        ],
        "blocked_imports": [
            "os, sys, subprocess, shutil — No filesystem or process access",
            "socket, http, urllib, requests, httpx — No network access",
            "pickle, marshal — No serialization",
            "threading, multiprocessing, asyncio — No concurrency primitives",
            "importlib, inspect, ast — No code introspection",
        ],
    }


@router.post("/validate")
async def validate_plugin(req: PluginValidateRequest):
    """Validate plugin source code without saving. Returns validation results."""
    result = validate_plugin_source(req.source_code)
    return result


@router.get("", response_model=list[PluginResponse])
async def list_plugins():
    """List all strategy plugins with their current status."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(StrategyPlugin).order_by(
                StrategyPlugin.sort_order.asc(), StrategyPlugin.name.asc()
            )
        )
        plugins = result.scalars().all()
        return [_plugin_to_response(p) for p in plugins]


@router.post("", response_model=PluginResponse)
async def create_plugin(req: PluginCreateRequest):
    """Create a new strategy plugin from source code."""
    slug = _validate_slug(req.slug)

    # Check slug uniqueness
    async with AsyncSessionLocal() as session:
        existing = await session.execute(
            select(StrategyPlugin).where(StrategyPlugin.slug == slug)
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=409,
                detail=f"A plugin with slug '{slug}' already exists.",
            )

    # Validate the source code
    validation = validate_plugin_source(req.source_code)
    if not validation["valid"]:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Plugin validation failed",
                "errors": validation["errors"],
            },
        )

    # Extract metadata from the source
    strategy_name = validation["strategy_name"] or slug.replace("_", " ").title()
    strategy_description = validation["strategy_description"]
    class_name = validation["class_name"]

    plugin_id = str(uuid.uuid4())
    status = "unloaded"
    error_message = None

    # Try to load it if enabled
    if req.enabled:
        try:
            plugin_loader.load_plugin(slug, req.source_code, req.config or None)
            status = "loaded"
        except PluginValidationError as e:
            status = "error"
            error_message = str(e)

    # Save to database
    async with AsyncSessionLocal() as session:
        plugin = StrategyPlugin(
            id=plugin_id,
            slug=slug,
            name=strategy_name,
            description=strategy_description,
            source_code=req.source_code,
            class_name=class_name,
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

    return _plugin_to_response(plugin)


@router.get("/{plugin_id}", response_model=PluginResponse)
async def get_plugin(plugin_id: str):
    """Get a single plugin by ID."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(StrategyPlugin).where(StrategyPlugin.id == plugin_id)
        )
        plugin = result.scalar_one_or_none()
        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not found")
        return _plugin_to_response(plugin)


@router.put("/{plugin_id}", response_model=PluginResponse)
async def update_plugin(plugin_id: str, req: PluginUpdateRequest):
    """Update a plugin. Source code changes trigger re-validation and reload."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(StrategyPlugin).where(StrategyPlugin.id == plugin_id)
        )
        plugin = result.scalar_one_or_none()
        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not found")

        code_changed = False

        if req.source_code is not None and req.source_code != plugin.source_code:
            # Validate new source code
            validation = validate_plugin_source(req.source_code)
            if not validation["valid"]:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "Plugin validation failed",
                        "errors": validation["errors"],
                    },
                )
            plugin.source_code = req.source_code
            plugin.class_name = validation["class_name"]
            # Update name/description from code if not explicitly overridden
            if req.name is None and validation["strategy_name"]:
                plugin.name = validation["strategy_name"]
            if req.description is None and validation["strategy_description"]:
                plugin.description = validation["strategy_description"]
            plugin.version += 1
            code_changed = True

        if req.config is not None:
            plugin.config = req.config
            code_changed = True  # Need reload for config changes too

        if req.name is not None:
            plugin.name = req.name
        if req.description is not None:
            plugin.description = req.description

        enabled_changed = False
        if req.enabled is not None and req.enabled != plugin.enabled:
            plugin.enabled = req.enabled
            enabled_changed = True

        # Handle loading/unloading
        if enabled_changed or code_changed:
            if plugin.enabled:
                try:
                    plugin_loader.load_plugin(
                        plugin.slug, plugin.source_code, plugin.config or None
                    )
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

    return _plugin_to_response(plugin)


@router.delete("/{plugin_id}")
async def delete_plugin(plugin_id: str):
    """Delete a plugin and unload it from the runtime."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(StrategyPlugin).where(StrategyPlugin.id == plugin_id)
        )
        plugin = result.scalar_one_or_none()
        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not found")

        # Unload from runtime
        plugin_loader.unload_plugin(plugin.slug)

        await session.delete(plugin)
        await session.commit()

    return {"status": "success", "message": "Plugin deleted"}


@router.post("/{plugin_id}/reload")
async def reload_plugin(plugin_id: str):
    """Force reload a plugin from its stored source code."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(StrategyPlugin).where(StrategyPlugin.id == plugin_id)
        )
        plugin = result.scalar_one_or_none()
        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not found")

        if not plugin.enabled:
            raise HTTPException(
                status_code=400,
                detail="Cannot reload a disabled plugin. Enable it first.",
            )

        try:
            plugin_loader.load_plugin(
                plugin.slug, plugin.source_code, plugin.config or None
            )
            plugin.status = "loaded"
            plugin.error_message = None
            await session.commit()

            return {
                "status": "success",
                "message": f"Plugin '{plugin.slug}' reloaded successfully",
                "runtime": plugin_loader.get_status(plugin.slug),
            }
        except PluginValidationError as e:
            plugin.status = "error"
            plugin.error_message = str(e)
            await session.commit()

            raise HTTPException(
                status_code=400,
                detail={
                    "message": f"Failed to reload plugin '{plugin.slug}'",
                    "error": str(e),
                },
            )
