"""
Plugin API Routes  (DEPRECATED — backward compatibility alias)

These /plugins endpoints are preserved for backward compatibility.
New clients should use the unified /strategies API (routes_strategies.py)
which serves both opportunity-detection and trader-execution strategies
through a single interface.

Full CRUD for code-based strategy plugins. Each plugin is a Python file
defining a BaseStrategy subclass with custom detection logic — a real strategy,
not just a grouping of existing ones.
"""

import re
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from sqlalchemy import select
from models.database import AsyncSessionLocal, Strategy, StrategyPluginTombstone
from services.opportunity_strategy_catalog import ensure_system_opportunity_strategies_seeded
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
    source_key: str = Field(default="scanner", min_length=2, max_length=64, description="Owning source key")
    source_code: str = Field(..., min_length=10, description="Python source code")
    config: dict = Field(default_factory=dict, description="Config overrides for the plugin")
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    enabled: bool = True


class PluginUpdateRequest(BaseModel):
    """Request to update a plugin (partial)."""

    slug: Optional[str] = Field(None, min_length=3, max_length=50)
    source_code: Optional[str] = Field(None, min_length=10)
    config: Optional[dict] = None
    enabled: Optional[bool] = None
    source_key: Optional[str] = Field(default=None, min_length=2, max_length=64)
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)


class PluginValidateRequest(BaseModel):
    """Request to validate plugin source code without saving."""

    source_code: str = Field(..., min_length=10)


class PluginResponse(BaseModel):
    """Plugin response model."""

    id: str
    slug: str
    source_key: str
    name: str
    description: Optional[str]
    source_code: str
    class_name: Optional[str]
    is_system: bool
    enabled: bool
    status: str  # unloaded, loaded, error
    error_message: Optional[str]
    config: dict
    config_schema: Optional[dict] = None  # param_fields for dynamic config UI
    version: int
    sort_order: int
    created_at: Optional[str]
    updated_at: Optional[str]
    # Runtime info (from loader, if loaded)
    runtime: Optional[dict] = None


def _extract_config_schema(p: Strategy) -> Optional[dict]:
    """Extract config_schema from the config._schema key or seed catalog."""
    from services.opportunity_strategy_catalog import SYSTEM_OPPORTUNITY_STRATEGY_SEEDS

    # System rows should track the latest seed schema so UI param sections
    # stay aligned with shipped strategy capabilities.
    if bool(p.is_system):
        for seed in SYSTEM_OPPORTUNITY_STRATEGY_SEEDS:
            if seed.slug == p.slug and seed.config_schema:
                return seed.config_schema

    cfg = p.config or {}
    if isinstance(cfg, dict) and "_schema" in cfg:
        return cfg["_schema"]
    # Fall back to seed catalog lookup for non-system rows that mirror seed slugs.
    for seed in SYSTEM_OPPORTUNITY_STRATEGY_SEEDS:
        if seed.slug == p.slug and seed.config_schema:
            return seed.config_schema
    return None


def _plugin_to_response(p: Strategy) -> PluginResponse:
    """Convert a DB Strategy to a PluginResponse."""
    runtime = plugin_loader.get_status(p.slug)
    # Strip internal _schema key from config before returning
    config = dict(p.config or {})
    config.pop("_schema", None)
    return PluginResponse(
        id=p.id,
        slug=p.slug,
        source_key=p.source_key or "scanner",
        name=p.name,
        description=p.description,
        source_code=p.source_code,
        class_name=p.class_name,
        is_system=bool(p.is_system),
        enabled=p.enabled,
        status=p.status,
        error_message=p.error_message,
        config=config,
        config_schema=_extract_config_schema(p),
        version=p.version,
        sort_order=p.sort_order,
        created_at=p.created_at.isoformat() if p.created_at else None,
        updated_at=p.updated_at.isoformat() if p.updated_at else None,
        runtime=runtime,
    )


# ==================== ENDPOINTS ====================


@router.get("/template")
async def get_plugin_template():
    """Get the starter template for writing a new opportunity strategy."""
    return {
        "template": PLUGIN_TEMPLATE,
        "instructions": (
            "Create a class that extends BaseStrategy and implements detect() or "
            "detect_async(). Use detect() for synchronous strategies or "
            "detect_async() (preferred) for strategies needing async I/O like "
            "LLM calls or HTTP requests. Both receive events, markets, and live "
            "prices on every scan cycle. Use self.create_opportunity() to build "
            "opportunities with automatic fee calculation, hard filters, and "
            "risk scoring. Access your plugin's config via self.config."
        ),
        "available_imports": [
            "models (Market, Event, ArbitrageOpportunity, StrategyType)",
            "services.strategies.base (BaseStrategy)",
            "services.strategies.* (built-in strategy modules)",
            "services.news.* (news strategy helpers)",
            "services.optimization.*",
            "services.ws_feeds",
            "services.chainlink_feed",
            "services.fee_model (fee_model)",
            "config (settings)",
            "math, statistics, collections, datetime, re, json, random, threading, asyncio, calendar, pathlib, etc.",
            "httpx",
            "numpy, scipy (if installed)",
        ],
    }


@router.get("/docs")
async def get_plugin_docs():
    """Get comprehensive API documentation for opportunity strategy authors."""
    return {
        "overview": {
            "title": "Opportunity Strategy API Reference",
            "description": (
                "Each strategy is a Python class that extends BaseStrategy. "
                "Your class must implement detect() or detect_async(). "
                "detect() is synchronous and runs in a thread-pool executor. "
                "detect_async() is async and preferred for I/O-bound strategies "
                "(LLM calls, HTTP requests, database queries). If both are defined, "
                "detect_async() takes priority. Both are called every scan cycle "
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
                "strategy_type": "str — Set to the DB strategy key/slug by the loader",
            },
            "inherited_attributes": {
                "self.fee": "float — Current Polymarket fee rate (e.g. 0.02 = 2%)",
                "self.min_profit": "float — Minimum profit threshold from app settings",
                "self.config": "dict — Your default_config merged with user overrides from the UI",
            },
        },
        "detect_method": {
            "signature": "def detect(self, events: list[Event], markets: list[Market], prices: dict[str, dict]) -> list[ArbitrageOpportunity]",
            "description": (
                "Synchronous detection method. Called every scan cycle in a thread-pool "
                "executor. Use this for CPU-bound strategies that do not need async I/O."
            ),
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
        "detect_async_method": {
            "signature": "async def detect_async(self, events: list[Event], markets: list[Market], prices: dict[str, dict]) -> list[ArbitrageOpportunity]",
            "description": (
                "Async detection method (preferred for I/O-bound strategies). "
                "Awaited directly on the event loop by the scanner. Use this when your "
                "strategy needs to make LLM calls (services.ai), HTTP requests (httpx), "
                "database queries, or any other async I/O. Same parameters and return "
                "type as detect(). If both detect() and detect_async() are defined, "
                "detect_async() takes priority at runtime."
            ),
            "parameters": "Same as detect() above",
            "returns": "list[ArbitrageOpportunity] — Use self.create_opportunity() to build these",
            "example": (
                "async def detect_async(self, events, markets, prices):\n"
                "    opportunities = []\n"
                "    for market in markets:\n"
                "        if market.closed or not market.active:\n"
                "            continue\n"
                "        # Make async LLM call\n"
                "        from services.ai import get_llm_manager\n"
                "        from services.ai.llm_provider import LLMMessage\n"
                "        manager = get_llm_manager()\n"
                "        if not manager.is_available():\n"
                "            continue\n"
                "        response = await manager.chat(\n"
                "            messages=[LLMMessage(role='user', content=f'Analyze: {market.question}')],\n"
                "            model='gpt-4o-mini',\n"
                "            purpose='custom_strategy',\n"
                "        )\n"
                "        # ... process response and create opportunities\n"
                "    return opportunities"
            ),
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
                    "bool — True for structural arbitrage (guaranteed profit), False for directional/statistical bets"
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
            {"module": "services.strategies", "items": "Built-in strategy modules"},
            {"module": "services.weather.signal_engine", "items": (
                "Weather utility functions: clamp01, temp_range_probability, ensemble_bucket_probability, "
                "compute_consensus, compute_model_agreement, compute_confidence, normalize_weights, logit"
            )},
            {"module": "services.news", "items": "News ingestion/matching components"},
            {"module": "services.optimization", "items": "Optimization strategy helpers"},
            {"module": "services.ws_feeds", "items": "Realtime feed helpers"},
            {"module": "services.chainlink_feed", "items": "Chainlink-derived helpers"},
            {"module": "services.fee_model", "items": "fee_model"},
            {"module": "services.ai", "items": "LLM provider (get_llm_manager, LLMMessage, LLMResponse)"},
            {"module": "services.strategy_sdk", "items": "High-level strategy helpers (StrategySDK)"},
            {"module": "config", "items": "settings (app configuration)"},
            {"module": "httpx", "items": "HTTP client"},
            {"module": "asyncio", "items": "Async orchestration"},
            {"module": "threading", "items": "Thread locks/primitives"},
            {"module": "concurrent.futures", "items": "ThreadPoolExecutor"},
            {"module": "calendar", "items": "Calendar math/helpers"},
            {"module": "pathlib", "items": "Path utilities"},
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
            "socket, http, urllib, requests, aiohttp — Unsupported network/process primitives",
            "pickle, marshal — No serialization",
            "multiprocessing — No process-based concurrency",
            "importlib, inspect, ast — No code introspection",
        ],
        "ai_integration": {
            "description": (
                "Strategy plugins can make LLM calls using the AI provider system. "
                "This supports OpenAI, Anthropic, Google (Gemini), xAI (Grok), DeepSeek, Ollama, and LM Studio. "
                "API keys are managed in the app settings — your strategy code does not need to handle keys."
            ),
            "usage": (
                "from services.ai import get_llm_manager\n"
                "from services.ai.llm_provider import LLMMessage\n\n"
                "async def _ask_llm(self, prompt: str) -> str:\n"
                "    manager = get_llm_manager()\n"
                "    if not manager.is_available():\n"
                "        return ''\n"
                "    response = await manager.chat(\n"
                "        messages=[LLMMessage(role='user', content=prompt)],\n"
                "        model='gpt-4o-mini',\n"
                "        purpose='custom_strategy',\n"
                "    )\n"
                "    return response.content"
            ),
            "structured_output": (
                "from services.ai import get_llm_manager\n"
                "from services.ai.llm_provider import LLMMessage\n\n"
                "schema = {'type': 'object', 'properties': {'score': {'type': 'number'}, 'reasoning': {'type': 'string'}}}\n"
                "result = await manager.structured_output(\n"
                "    messages=[LLMMessage(role='system', content='You are a market analyst.'),\n"
                "             LLMMessage(role='user', content='Rate this opportunity...')],\n"
                "    schema=schema,\n"
                "    model='gpt-4o-mini',\n"
                "    purpose='custom_strategy',\n"
                ")\n"
                "score = result.get('score', 0)"
            ),
            "available_classes": {
                "LLMMessage": "LLMMessage(role, content) — 'system', 'user', or 'assistant'",
                "LLMResponse": "LLMResponse(content, tool_calls, usage, model, provider, latency_ms)",
                "ToolDefinition": "ToolDefinition(name, description, parameters) — for function calling",
                "ToolCall": "ToolCall(id, name, arguments) — parsed tool call from response",
            },
            "notes": [
                "LLM calls are async — implement detect_async() to use them directly with await",
                "If using sync detect(), wrap async calls with asyncio.get_event_loop().run_until_complete()",
                "Prefer detect_async() over detect() when making LLM or HTTP calls",
                "Calls respect the global pause state — they are blocked when the system is paused",
                "Usage is logged automatically to the LLMUsageLog table with cost tracking",
                "Always set purpose='custom_strategy' for usage attribution",
            ],
        },
        "strategy_sdk": {
            "description": (
                "The strategy SDK module provides high-level helpers that wrap common operations. "
                "Import it with: from services.strategy_sdk import StrategySDK"
            ),
            "methods": {
                "get_live_price(market, prices)": "Get the best available mid price for a market",
                "get_spread_bps(market, prices)": "Get the bid-ask spread in basis points",
                "ask_llm(prompt, model, purpose)": "Send a simple prompt to an LLM and get the text response",
                "ask_llm_json(prompt, schema, model, purpose)": "Get structured JSON from an LLM",
                "get_chainlink_price(asset)": "Get the latest Chainlink oracle price for BTC/ETH/SOL/XRP",
                "get_news_for_market(market)": "Get recent news articles matched to a market",
                "calculate_fees(cost, payout, n_legs)": "Calculate comprehensive fees for a trade",
            },
        },
        "weather_strategies": {
            "description": (
                "Weather strategies use detect_from_intents() instead of detect(). "
                "The weather workflow passes enriched intents with raw forecast data, "
                "ensemble members, and sibling market information. Each strategy computes "
                "its own direction by comparing model probability to market price."
            ),
            "detect_from_intents_method": {
                "signature": "def detect_from_intents(self, intents: list[dict], markets: list[Market], events: list[Event]) -> list[ArbitrageOpportunity]",
                "description": "Called with enriched weather intents. Each strategy owns its direction logic.",
            },
            "enriched_intent_fields": {
                "market_id": "str — Target market condition ID",
                "market_question": "str — Human-readable market question",
                "event_slug": "str — Parent event slug (for grouping sibling buckets)",
                "yes_price": "float — Current YES price on market",
                "no_price": "float — Current NO price on market",
                "liquidity": "float — Market liquidity in USD",
                "clob_token_ids": "list[str] — [yes_token_id, no_token_id]",
                "location": "str — City/location (e.g., 'London', 'New York')",
                "metric": "str — Weather metric (e.g., 'temp_max_range')",
                "operator": "str — Contract operator ('between', 'gt', 'lt')",
                "bucket_low_c": "float | None — Low end of temperature bucket in Celsius",
                "bucket_high_c": "float | None — High end of temperature bucket in Celsius",
                "threshold_c": "float | None — Single threshold for above/below contracts",
                "target_time": "str — ISO datetime of forecast target",
                "source_values_c": "dict[str, float] — Per-source temperature values (e.g., {'open_meteo:gfs_seamless': 7.2})",
                "source_probabilities": "dict[str, float] — Per-source probabilities",
                "source_weights": "dict[str, float] — Normalized source weights",
                "consensus_value_c": "float | None — Weighted consensus temperature in Celsius",
                "consensus_probability": "float | None — Weighted consensus probability",
                "source_spread_c": "float | None — Max temperature spread across sources",
                "source_count": "int — Number of forecast sources",
                "model_agreement": "float — Cross-model agreement (0-1)",
                "ensemble_members": "list[float] | None — GFS ensemble member values (up to 31 members)",
                "ensemble_daily_max": "list[float] | None — Per-member daily max temperatures",
                "sibling_markets": (
                    "list[dict] — Other temperature buckets in the same event. "
                    "Each has: market_id, market_question, yes_price, no_price, "
                    "bucket_low_c, bucket_high_c, liquidity, clob_token_ids"
                ),
            },
            "signal_engine_utilities": {
                "description": "Import from services.weather.signal_engine for common calculations",
                "functions": {
                    "temp_range_probability(value_c, low_c, high_c, scale_c=2.0)": (
                        "Probability a temperature falls in [low, high] using logistic CDF. "
                        "Lower scale_c = sharper discrimination between buckets."
                    ),
                    "ensemble_bucket_probability(ensemble_values, low_c, high_c)": (
                        "Count fraction of ensemble members in a temperature bucket. "
                        "Monte Carlo approach — no parametric assumptions."
                    ),
                    "compute_consensus(source_probs, source_weights)": (
                        "Weighted consensus probability from multiple forecast sources."
                    ),
                    "compute_model_agreement(source_probs)": (
                        "Agreement metric: 1.0 - max spread across sources."
                    ),
                    "compute_confidence(agreement, consensus_yes, source_count, source_spread_c)": (
                        "Blended confidence score from agreement, separation, and source depth."
                    ),
                    "clamp01(value)": "Clamp float to [0, 1].",
                },
            },
            "built_in_weather_strategies": {
                "weather_edge": (
                    "Baseline consensus strategy. Compares model probability to market price "
                    "using sigmoid CDF with configurable scale. Uses deterministic forecasts."
                ),
                "weather_ensemble_edge": (
                    "Ensemble Monte Carlo. Counts fraction of 31 GFS ensemble members in each "
                    "bucket. Falls back to sigmoid when ensemble data unavailable."
                ),
                "weather_distribution": (
                    "Full distribution comparison. Builds probability mass across ALL temperature "
                    "buckets in an event, normalizes to sum=1, buys the most underpriced bucket."
                ),
                "weather_conservative_no": (
                    "Conservative NO-betting. Only bets NO on buckets far from consensus forecast. "
                    "High win rate, lower per-trade profit. Never buys YES."
                ),
                "weather_bucket_edge": (
                    "Per-bucket edge with tunable sigmoid. Like weather_edge but with tighter "
                    "default sigmoid scale (1.5 vs 2.0) for sharper bucket discrimination."
                ),
            },
            "direction_logic_pattern": (
                "# The KEY pattern: compare model_prob to market price, NOT to 0.5\n"
                "model_prob = temp_range_probability(consensus_c, bucket_low, bucket_high, scale)\n"
                "if model_prob > yes_price:\n"
                "    direction = 'buy_yes'  # Market underprices this bucket\n"
                "    edge = (model_prob - yes_price) * 100\n"
                "else:\n"
                "    direction = 'buy_no'   # Market overprices this bucket\n"
                "    edge = ((1 - model_prob) - no_price) * 100"
            ),
        },
        "cookbook": {
            "description": "Common patterns and recipes for strategy development",
            "recipes": {
                "iterate_markets_with_live_prices": (
                    "for market in markets:\n"
                    "    if market.closed or not market.active:\n"
                    "        continue\n"
                    "    if len(market.outcome_prices) != 2:\n"
                    "        continue\n"
                    "    yes_id = market.clob_token_ids[0] if market.clob_token_ids else None\n"
                    "    no_id = market.clob_token_ids[1] if len(market.clob_token_ids or []) > 1 else None\n"
                    "    yes_price = prices.get(yes_id, {}).get('mid', market.yes_price) if yes_id else market.yes_price\n"
                    "    no_price = prices.get(no_id, {}).get('mid', market.no_price) if no_id else market.no_price"
                ),
                "group_markets_by_event": (
                    "from collections import defaultdict\n"
                    "event_markets = defaultdict(list)\n"
                    "for event in events:\n"
                    "    for market in event.markets:\n"
                    "        if market.active and not market.closed:\n"
                    "            event_markets[event.id].append(market)"
                ),
                "check_spread_before_trading": (
                    "from services.ws_feeds import get_feed_manager\n"
                    "feed = get_feed_manager()\n"
                    "spread_bps = feed.cache.get_spread_bps(token_id)\n"
                    "if spread_bps is not None and spread_bps > 200:  # 2% spread\n"
                    "    continue  # Skip wide-spread markets"
                ),
                "use_chainlink_oracle_prices": (
                    "from services.chainlink_feed import get_chainlink_feed\n"
                    "feed = get_chainlink_feed()\n"
                    "btc = feed.get_price('BTC')\n"
                    "if btc:\n"
                    "    current_btc_price = btc.price  # e.g. 97234.56"
                ),
                "filter_by_category": (
                    "crypto_events = [e for e in events if e.category == 'crypto']\n"
                    "politics_markets = [m for m in markets if any(\n"
                    "    e.category == 'politics' for e in events if any(em.id == m.id for em in e.markets)\n"
                    ")]"
                ),
                "create_multi_leg_opportunity": (
                    "positions = []\n"
                    "total_cost = 0.0\n"
                    "involved_markets = []\n"
                    "for market, side, price in legs:\n"
                    "    token_idx = 0 if side == 'YES' else 1\n"
                    "    token_id = market.clob_token_ids[token_idx] if len(market.clob_token_ids) > token_idx else None\n"
                    "    positions.append({'action': 'BUY', 'outcome': side, 'price': price, 'token_id': token_id})\n"
                    "    total_cost += price\n"
                    "    involved_markets.append(market)\n"
                    "opp = self.create_opportunity(\n"
                    "    title='Multi-leg edge',\n"
                    "    description='Cross-market structural opportunity',\n"
                    "    total_cost=total_cost,\n"
                    "    markets=involved_markets,\n"
                    "    positions=positions,\n"
                    "    expected_payout=1.0,\n"
                    "    is_guaranteed=True,\n"
                    ")"
                ),
            },
        },
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
        await ensure_system_opportunity_strategies_seeded(session)
        result = await session.execute(
            select(Strategy).order_by(
                Strategy.is_system.desc(),
                Strategy.sort_order.asc(),
                Strategy.name.asc(),
            )
        )
        plugins = result.scalars().all()
        return [_plugin_to_response(p) for p in plugins]


@router.post("", response_model=PluginResponse)
async def create_plugin(req: PluginCreateRequest):
    """Create a new strategy plugin from source code."""
    slug = _validate_slug(req.slug)
    source_key = str(req.source_key or "scanner").strip().lower()

    # Check slug uniqueness
    async with AsyncSessionLocal() as session:
        existing = await session.execute(select(Strategy).where(Strategy.slug == slug))
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
    strategy_name = (req.name or validation["strategy_name"] or slug.replace("_", " ").title()).strip()
    strategy_description = req.description if req.description is not None else validation["strategy_description"]
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
        plugin = Strategy(
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

    return _plugin_to_response(plugin)


@router.get("/{plugin_id}", response_model=PluginResponse)
async def get_plugin(plugin_id: str):
    """Get a single plugin by ID."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Strategy).where(Strategy.id == plugin_id))
        plugin = result.scalar_one_or_none()
        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not found")
        return _plugin_to_response(plugin)


@router.put("/{plugin_id}", response_model=PluginResponse)
async def update_plugin(plugin_id: str, req: PluginUpdateRequest):
    """Update a plugin. Source code changes trigger re-validation and reload."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Strategy).where(Strategy.id == plugin_id))
        plugin = result.scalar_one_or_none()
        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not found")

        original_slug = plugin.slug
        code_changed = False

        slug_changed = False
        if req.slug is not None:
            next_slug = _validate_slug(req.slug)
            if next_slug != plugin.slug:
                existing_slug = await session.execute(
                    select(Strategy.id).where(
                        Strategy.slug == next_slug,
                        Strategy.id != plugin.id,
                    )
                )
                if existing_slug.scalar_one_or_none():
                    raise HTTPException(
                        status_code=409,
                        detail=f"A strategy with slug '{next_slug}' already exists.",
                    )
                plugin.slug = next_slug
                slug_changed = True

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

        # Handle loading/unloading
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

    return _plugin_to_response(plugin)


@router.delete("/{plugin_id}")
async def delete_plugin(plugin_id: str):
    """Delete a plugin and unload it from the runtime."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Strategy).where(Strategy.id == plugin_id))
        plugin = result.scalar_one_or_none()
        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not found")

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

        # Unload from runtime
        plugin_loader.unload_plugin(plugin.slug)

        await session.delete(plugin)
        await session.commit()

    return {"status": "success", "message": "Plugin deleted"}


@router.post("/{plugin_id}/reload")
async def reload_plugin(plugin_id: str):
    """Force reload a plugin from its stored source code."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Strategy).where(Strategy.id == plugin_id))
        plugin = result.scalar_one_or_none()
        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not found")

        if not plugin.enabled:
            raise HTTPException(
                status_code=400,
                detail="Cannot reload a disabled plugin. Enable it first.",
            )

        try:
            plugin_loader.load_plugin(plugin.slug, plugin.source_code, plugin.config or None)
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
