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
    StrategyTombstone,
    get_db_session,
)
from services.opportunity_strategy_catalog import (
    ensure_system_opportunity_strategies_seeded,
    ensure_system_trader_strategies_seeded,
)
from services.strategy_loader import (
    STRATEGY_TEMPLATE as PLUGIN_TEMPLATE,
    StrategyValidationError as PluginValidationError,
    strategy_loader as plugin_loader,
    validate_strategy_source as validate_plugin_source,
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
    config: dict = Field(default_factory=dict, description="Config / default params")
    config_schema: dict = Field(default_factory=dict, description="Param schema for UI form")
    enabled: bool = True


class UnifiedStrategyUpdateRequest(BaseModel):
    """Partial update for a strategy."""

    slug: Optional[str] = Field(None, min_length=3, max_length=128)
    source_key: Optional[str] = Field(None, min_length=2, max_length=64)
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    source_code: Optional[str] = Field(None, min_length=10)
    config: Optional[dict] = None
    config_schema: Optional[dict] = None
    enabled: Optional[bool] = None
    unlock_system: bool = False


class UnifiedValidateRequest(BaseModel):
    source_code: str = Field(..., min_length=10)
    class_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers — detect capabilities from source code
# ---------------------------------------------------------------------------


def _detect_capabilities(source_code: str) -> dict:
    """Inspect source code to determine which lifecycle methods are present.

    All strategy business logic must be in the user-editable source code,
    so a simple regex scan of the source is sufficient.  BaseStrategy
    provides default evaluate/should_exit, so any subclass inherits those.
    """
    has_detect = bool(re.search(r"\bdef detect\s*\(", source_code))
    has_detect_async = bool(re.search(r"\basync\s+def detect_async\s*\(", source_code))
    has_evaluate = bool(re.search(r"\bdef evaluate\s*\(", source_code))
    has_should_exit = bool(re.search(r"\bdef should_exit\s*\(", source_code))

    # BaseStrategy provides default evaluate() and should_exit() for ALL
    # strategies.  Any class extending BaseStrategy has working defaults.
    extends_base = bool(re.search(r"\bBaseStrategy\b", source_code))
    if extends_base:
        has_evaluate = True
        has_should_exit = True

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
        "aliases": [],
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
            "services.trader_orchestrator.strategies.base (BaseStrategy, StrategyDecision, DecisionCheck)",
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
        "title": "Strategy Developer Reference",
        "version": "2.0",

        # ── Section 1: Overview ──────────────────────────────────────
        "overview": {
            "summary": (
                "Strategies are the core decision-making units. Every strategy is a "
                "Python class stored in the database that extends BaseStrategy. "
                "A single strategy can own the ENTIRE lifecycle of a trade — from "
                "finding the opportunity, to deciding whether to execute, to managing "
                "the open position and deciding when to exit."
            ),
            "three_phase_lifecycle": {
                "description": (
                    "Every strategy participates in up to three phases. You must "
                    "implement at least one of detect() or evaluate(). All other "
                    "methods have sensible defaults."
                ),
                "phases": [
                    {
                        "phase": "DETECT",
                        "method": "detect(events, markets, prices) -> list[ArbitrageOpportunity]",
                        "async_method": "detect_async(events, markets, prices) -> list[ArbitrageOpportunity]",
                        "caller": "Scanner service — runs every scan cycle (~30s)",
                        "purpose": "Find trading opportunities from live market data",
                        "default_behavior": "Returns empty list (no opportunities)",
                    },
                    {
                        "phase": "EVALUATE",
                        "method": "evaluate(signal, context) -> StrategyDecision",
                        "caller": "Orchestrator — when a pending signal is ready for execution",
                        "purpose": "Gate execution: decide whether to trade a signal right now",
                        "default_behavior": (
                            "Passthrough — checks min_edge_percent and min_confidence "
                            "from config, sizes position using base_size_usd, returns 'selected'"
                        ),
                    },
                    {
                        "phase": "EXIT",
                        "method": "should_exit(position, market_state) -> ExitDecision",
                        "caller": "Position lifecycle — runs every cycle for open positions",
                        "purpose": "Decide whether to close, hold, or reduce an open position",
                        "default_behavior": (
                            "Delegates to default_exit_check() which applies standard "
                            "take-profit, stop-loss, trailing-stop, and max-hold from config"
                        ),
                    },
                ],
            },
            "strategy_types": {
                "detect": "Implements detect() or detect_async() — finds opportunities",
                "execute": "Implements evaluate() — gates trade execution",
                "unified": "Implements both detect and evaluate — full lifecycle ownership",
                "note": "Type is auto-inferred from which methods your class implements.",
            },
        },

        # ── Section 2: BaseStrategy Interface ────────────────────────
        "base_strategy": {
            "import": "from services.strategies.base import BaseStrategy, StrategyDecision, ExitDecision, DecisionCheck",
            "class_attributes": {
                "name": {
                    "type": "str",
                    "required": True,
                    "description": "Human-readable strategy name (shown in UI)",
                },
                "description": {
                    "type": "str",
                    "required": True,
                    "description": "What this strategy does (shown in strategy list)",
                },
                "default_config": {
                    "type": "dict",
                    "required": False,
                    "description": (
                        "Default configuration values. Users can override these in the UI. "
                        "Access at runtime via self.config (merged defaults + user overrides)."
                    ),
                },
            },
            "built_in_properties": {
                "self.config": "dict — Merged default_config + user overrides (set by configure())",
                "self.fee": "float — Platform fee rate (from settings.POLYMARKET_FEE)",
                "self.min_profit": "float — Min profit threshold (from settings.MIN_PROFIT_THRESHOLD)",
            },
            "helper_methods": {
                "create_opportunity()": {
                    "signature": (
                        "self.create_opportunity(title, description, total_cost, markets, "
                        "positions, event=None, expected_payout=1.0, is_guaranteed=True, "
                        "vwap_total_cost=None, spread_bps=None, fill_probability=None, "
                        "min_liquidity_hard=None, min_position_size=None, min_absolute_profit=None) "
                        "-> Optional[ArbitrageOpportunity]"
                    ),
                    "description": (
                        "Creates an ArbitrageOpportunity after applying hard rejection filters: "
                        "ROI threshold, min liquidity, min position size, min absolute profit, "
                        "annualized ROI, max resolution timeframe. Returns None if any filter rejects."
                    ),
                },
                "calculate_risk_score()": {
                    "signature": "self.calculate_risk_score(markets, resolution_date=None) -> tuple[float, list[str]]",
                    "description": "Multi-factor risk score (0-1) with human-readable risk factors.",
                },
                "default_exit_check()": {
                    "signature": "self.default_exit_check(position, market_state) -> ExitDecision",
                    "description": (
                        "Standard TP/SL/trailing/max-hold exit logic using config params. "
                        "Call this as a fallback in your custom should_exit()."
                    ),
                    "config_params": {
                        "take_profit_pct": "Close when PnL% >= this value",
                        "stop_loss_pct": "Close when PnL% <= -this value",
                        "trailing_stop_pct": "Close when price drops this % from highest",
                        "max_hold_minutes": "Close after this many minutes",
                        "min_hold_minutes": "Don't exit before this many minutes",
                        "resolve_only": "If true, only exit on market resolution",
                        "close_on_inactive_market": "Close if market becomes untradeable",
                    },
                },
                "configure()": {
                    "signature": "self.configure(config: dict) -> None",
                    "description": (
                        "Called by the loader after instantiation. Merges default_config "
                        "with user overrides and sets self.config. You do NOT call this yourself."
                    ),
                },
            },
        },

        # ── Section 3: DETECT Phase ──────────────────────────────────
        "detect_phase": {
            "methods": {
                "sync": {
                    "signature": "detect(self, events: list[Event], markets: list[Market], prices: dict[str, dict]) -> list[ArbitrageOpportunity]",
                    "when_to_use": "CPU-bound strategies with no async I/O needed",
                },
                "async": {
                    "signature": "async detect_async(self, events: list[Event], markets: list[Market], prices: dict[str, dict]) -> list[ArbitrageOpportunity]",
                    "when_to_use": "Strategies that need await — LLM calls (services.ai), HTTP requests (httpx), DB queries",
                    "note": "If detect_async() exists, the scanner calls it instead of detect()",
                },
            },
            "parameters": {
                "events": {
                    "type": "list[Event]",
                    "description": "All active Polymarket events",
                    "useful_fields": "event.id, event.title, event.slug, event.category, event.end_date",
                },
                "markets": {
                    "type": "list[Market]",
                    "description": "All active markets across all events",
                    "useful_fields": (
                        "market.id, market.slug, market.question, market.yes_price, "
                        "market.no_price, market.outcome_prices, market.liquidity, "
                        "market.volume, market.tokens, market.active, market.closed, "
                        "market.neg_risk, market.end_date, market.event_slug"
                    ),
                },
                "prices": {
                    "type": "dict[str, dict]",
                    "description": "Live CLOB prices keyed by token_id",
                    "structure": "{ token_id: { 'mid': float, 'best_bid': float, 'best_ask': float } }",
                },
            },
            "return_value": {
                "type": "list[ArbitrageOpportunity]",
                "tip": (
                    "Use self.create_opportunity() to build these. It handles ROI calculation, "
                    "fee modeling, risk scoring, and hard rejection filters automatically."
                ),
                "strategy_context": (
                    "Set opp.strategy_context = {...} to pass data from detect() to evaluate(). "
                    "This dict is serialized onto the TradeSignal and available in evaluate() "
                    "via signal.strategy_context."
                ),
            },
        },

        # ── Section 4: EVALUATE Phase ────────────────────────────────
        "evaluate_phase": {
            "method": "evaluate(self, signal, context) -> StrategyDecision",
            "when_called": (
                "The orchestrator calls this when a pending TradeSignal from your strategy "
                "is ready for execution. This is your chance to apply real-time gating: "
                "re-check live prices, enforce risk limits, size the position, etc."
            ),
            "signal_object": {
                "description": "TradeSignal ORM row — the opportunity your detect() found",
                "fields": {
                    "signal.source": "str — Data source (scanner, crypto, news, weather, traders)",
                    "signal.direction": "str — 'BUY' or 'SELL'",
                    "signal.edge_percent": "float — Estimated edge at detection time",
                    "signal.confidence": "float — Confidence score (0-1 or 0-100, auto-normalized)",
                    "signal.entry_price": "float — Suggested entry price",
                    "signal.liquidity": "float — Market liquidity at detection time (USD)",
                    "signal.payload_json": "dict — Source-specific extra data",
                    "signal.strategy_context": "dict — Data you set on the opportunity in detect()",
                    "signal.market_slug": "str — Market identifier",
                    "signal.condition_id": "str — Token/condition being traded",
                },
            },
            "context_object": {
                "description": "Dict with runtime context for the evaluation",
                "fields": {
                    "context['params']": "dict — Strategy config (merged default_config + user overrides)",
                    "context['trader']": "object — Trader ORM row (has .mode, .budget, etc.)",
                    "context['mode']": "str — 'paper' or 'live'",
                    "context['live_market']": "dict — Live CLOB prices if available",
                    "context['source_config']": "dict — Source configuration from trader settings",
                },
            },
            "return_value": {
                "type": "StrategyDecision",
                "constructor": "StrategyDecision(decision, reason, score=None, size_usd=None, checks=[], payload={})",
                "decision_values": {
                    "selected": "Execute this trade — must set size_usd",
                    "skipped": "Conditions not met right now (may retry later)",
                    "blocked": "Hard rejection — this signal should not be traded",
                    "failed": "Error during evaluation",
                },
                "checks_field": {
                    "type": "list[DecisionCheck]",
                    "constructor": "DecisionCheck(key, label, passed, score=None, detail=None, payload={})",
                    "purpose": "Individual gate results shown in the UI. Each check should represent one condition.",
                    "example": 'DecisionCheck("edge", "Edge threshold", edge >= 3.0, score=edge, detail=f"min=3.0")',
                },
            },
        },

        # ── Section 5: EXIT Phase ────────────────────────────────────
        "exit_phase": {
            "method": "should_exit(self, position, market_state) -> ExitDecision",
            "when_called": (
                "The position lifecycle calls this every cycle for each open position "
                "that was opened by your strategy. Override to implement custom exit logic "
                "(re-check forecasts, monitor correlated markets, decay-based exits, etc.)."
            ),
            "position_object": {
                "description": "Position with open trade data",
                "fields": {
                    "position.entry_price": "float — Price at entry",
                    "position.current_price": "float — Latest price",
                    "position.highest_price": "float — Highest price since entry",
                    "position.lowest_price": "float — Lowest price since entry",
                    "position.age_minutes": "float — Minutes since position was opened",
                    "position.pnl_percent": "float — Current PnL percentage",
                    "position.strategy_context": "dict — Data from detect() via the signal",
                    "position.config": "dict — Strategy params at time of entry",
                },
            },
            "market_state_object": {
                "description": "Current state of the market this position is in",
                "fields": {
                    "market_state['current_price']": "float — Latest price",
                    "market_state['market_tradable']": "bool — Whether market is still tradeable",
                    "market_state['is_resolved']": "bool — Whether market has resolved",
                    "market_state['winning_outcome']": "str | None — Winning outcome if resolved",
                },
            },
            "return_value": {
                "type": "ExitDecision",
                "constructor": "ExitDecision(action, reason, close_price=None, reduce_fraction=None, payload={})",
                "action_values": {
                    "close": "Close the entire position at close_price",
                    "hold": "Keep the position open",
                    "reduce": "Partially exit — set reduce_fraction (0-1) for the portion to close",
                },
                "tip": (
                    "Call self.default_exit_check(position, market_state) as a fallback "
                    "after your custom checks. It handles TP/SL/trailing/max-hold/resolution."
                ),
            },
        },

        # ── Section 6: Config Schema ─────────────────────────────────
        "config_schema": {
            "description": (
                "The config_schema defines what parameters appear in the strategy settings UI. "
                "It maps to the 'Config' section in the strategy flyout. Each param_field "
                "becomes an input control in the UI."
            ),
            "format": {
                "param_fields": [
                    {
                        "key": "min_edge_percent",
                        "label": "Min Edge (%)",
                        "type": "number",
                        "min": 0,
                        "max": 100,
                    },
                    {
                        "key": "min_confidence",
                        "label": "Min Confidence",
                        "type": "number",
                        "min": 0,
                        "max": 1,
                    },
                    {
                        "key": "base_size_usd",
                        "label": "Base Size (USD)",
                        "type": "number",
                        "min": 1,
                        "max": 10000,
                    },
                    {
                        "key": "cooldown_minutes",
                        "label": "Cooldown (min)",
                        "type": "integer",
                        "min": 0,
                    },
                ],
            },
            "field_types": {
                "number": "Float input with optional min/max bounds",
                "integer": "Whole number input with optional min/max bounds",
            },
            "how_it_works": (
                "1. Define default_config on your strategy class with default values. "
                "2. Set config_schema.param_fields to describe each param for the UI. "
                "3. The keys in param_fields must match keys in default_config. "
                "4. At runtime, user overrides are merged with defaults into self.config. "
                "5. In evaluate(), access via context['params'] which is the same merged config."
            ),
        },

        # ── Section 7: Available Imports ──────────────────────────────
        "imports": {
            "description": (
                "Strategies run in a sandboxed environment. Only approved imports are allowed. "
                "Import validation happens at save time via AST analysis — no code is executed."
            ),
            "app_modules": {
                "models": "Market, Event, ArbitrageOpportunity — core data types",
                "services.strategies.base": "BaseStrategy, StrategyDecision, ExitDecision, DecisionCheck",
                "services.ai": "LLM integration — call AI models from your strategy",
                "services.news": "News analysis services",
                "services.weather": "Weather signal engine",
                "services.optimization": "Parameter optimization utilities",
                "services.ws_feeds": "WebSocket market data feeds",
                "services.chainlink_feed": "Chainlink oracle price feeds",
                "services.fee_model": "Fee calculation model",
                "services.strategy_sdk": "Strategy development utilities",
                "config": "Application settings (settings object)",
                "utils": "Shared utility functions",
            },
            "standard_library": [
                "math", "statistics", "collections", "datetime", "time",
                "re", "json", "random", "asyncio", "threading",
                "itertools", "functools", "operator", "copy",
                "decimal", "fractions", "calendar",
                "dataclasses", "typing", "abc", "enum",
                "hashlib", "hmac", "base64", "uuid",
                "urllib.parse", "logging", "bisect", "heapq",
                "textwrap", "string", "concurrent", "pathlib",
            ],
            "third_party": {
                "httpx": "HTTP client — use for external API calls (async-friendly)",
                "numpy": "Numerical computing",
                "scipy": "Scientific computing and statistics",
            },
            "blocked": {
                "description": "These are blocked for security. Use the approved alternatives.",
                "filesystem": "os, sys, subprocess, shutil, io, tempfile, glob — no filesystem access",
                "network_raw": "socket, http, urllib (except urllib.parse), requests, aiohttp — use httpx instead",
                "serialization": "pickle, shelve, marshal — no arbitrary deserialization",
                "execution": "exec, eval, compile, __import__, open, input — no dynamic code execution",
                "introspection": "ast, dis, inspect, importlib, builtins — no runtime introspection",
                "process": "multiprocessing, signal — no process control",
            },
        },

        # ── Section 8: Complete Examples ──────────────────────────────
        "examples": {
            "minimal_detect_only": {
                "description": "Simplest possible strategy — detect only, uses default evaluate/exit",
                "source_code": (
                    '"""\n'
                    "Strategy: Simple Spread Finder\n"
                    '"""\n'
                    "from models import Market, Event, ArbitrageOpportunity\n"
                    "from services.strategies.base import BaseStrategy\n\n"
                    "class SimpleSpreadFinder(BaseStrategy):\n"
                    '    name = "Simple Spread Finder"\n'
                    '    description = "Finds binary markets where YES + NO < $1"\n\n'
                    "    default_config = {\n"
                    '        "min_spread_pct": 2.0,\n'
                    "    }\n\n"
                    "    def detect(self, events, markets, prices):\n"
                    "        opportunities = []\n"
                    "        for market in markets:\n"
                    "            if market.closed or not market.active:\n"
                    "                continue\n"
                    "            total = market.yes_price + market.no_price\n"
                    "            if total < 1.0:\n"
                    "                spread = (1.0 - total) / total * 100\n"
                    "                if spread >= self.config.get('min_spread_pct', 2.0):\n"
                    "                    opp = self.create_opportunity(\n"
                    '                        title=f"Spread on {market.question[:60]}",\n'
                    '                        description=f"{spread:.1f}% spread",\n'
                    "                        total_cost=total,\n"
                    "                        markets=[market],\n"
                    "                        positions=[\n"
                    '                            {"token_id": market.tokens[0].token_id, "side": "BUY", "price": market.yes_price},\n'
                    '                            {"token_id": market.tokens[1].token_id, "side": "BUY", "price": market.no_price},\n'
                    "                        ],\n"
                    "                        event=next((e for e in events if e.slug == market.event_slug), None),\n"
                    "                    )\n"
                    "                    if opp:\n"
                    "                        opportunities.append(opp)\n"
                    "        return opportunities\n"
                ),
            },
            "full_unified_strategy": {
                "description": "Complete strategy with custom detect, evaluate, and exit logic",
                "source_code": (
                    '"""\n'
                    "Strategy: Momentum Edge\n\n"
                    "Detects directional momentum, gates on live price confirmation,\n"
                    "exits on momentum reversal or standard TP/SL.\n"
                    '"""\n'
                    "from models import Market, Event, ArbitrageOpportunity\n"
                    "from services.strategies.base import BaseStrategy, StrategyDecision, ExitDecision, DecisionCheck\n\n"
                    "class MomentumEdge(BaseStrategy):\n"
                    '    name = "Momentum Edge"\n'
                    '    description = "Trades directional momentum with reversal-based exits"\n\n'
                    "    default_config = {\n"
                    '        "momentum_threshold": 0.05,\n'
                    '        "min_edge_percent": 2.0,\n'
                    '        "min_confidence": 0.5,\n'
                    '        "base_size_usd": 30.0,\n'
                    '        "max_size_usd": 200.0,\n'
                    '        "take_profit_pct": 20.0,\n'
                    '        "stop_loss_pct": 10.0,\n'
                    '        "reversal_threshold": 0.03,\n'
                    "    }\n\n"
                    "    def detect(self, events, markets, prices):\n"
                    "        opportunities = []\n"
                    "        threshold = self.config.get('momentum_threshold', 0.05)\n"
                    "        for market in markets:\n"
                    "            if market.closed or not market.active:\n"
                    "                continue\n"
                    "            # Check for price momentum via CLOB data\n"
                    "            for token in (market.tokens or []):\n"
                    "                price_data = prices.get(token.token_id)\n"
                    "                if not price_data:\n"
                    "                    continue\n"
                    "                mid = price_data.get('mid', 0)\n"
                    "                spread = price_data.get('best_ask', 0) - price_data.get('best_bid', 0)\n"
                    "                # Detect momentum: price far from 0.50 with tight spread\n"
                    "                if mid > 0 and abs(mid - 0.5) > threshold and spread < 0.05:\n"
                    "                    direction = 'BUY' if mid > 0.5 else 'SELL'\n"
                    "                    edge = abs(mid - 0.5) * 100\n"
                    "                    opp = self.create_opportunity(\n"
                    "                        title=f'Momentum {direction} on {market.question[:50]}',\n"
                    "                        description=f'{edge:.1f}% momentum edge',\n"
                    "                        total_cost=mid if direction == 'BUY' else (1 - mid),\n"
                    "                        markets=[market],\n"
                    "                        positions=[{'token_id': token.token_id, 'side': direction, 'price': mid}],\n"
                    "                        event=next((e for e in events if e.slug == market.event_slug), None),\n"
                    "                        is_guaranteed=False,\n"
                    "                    )\n"
                    "                    if opp:\n"
                    "                        opp.strategy_context = {'entry_mid': mid, 'direction': direction}\n"
                    "                        opportunities.append(opp)\n"
                    "        return opportunities\n\n"
                    "    def evaluate(self, signal, context):\n"
                    "        params = context.get('params') or {}\n"
                    "        edge = float(getattr(signal, 'edge_percent', 0) or 0)\n"
                    "        confidence = float(getattr(signal, 'confidence', 0) or 0)\n"
                    "        if confidence > 1.0:\n"
                    "            confidence /= 100.0\n\n"
                    "        min_edge = float(params.get('min_edge_percent', 2.0))\n"
                    "        min_conf = float(params.get('min_confidence', 0.5))\n"
                    "        base_size = float(params.get('base_size_usd', 30.0))\n"
                    "        max_size = float(params.get('max_size_usd', 200.0))\n\n"
                    "        checks = [\n"
                    "            DecisionCheck('edge', 'Edge threshold', edge >= min_edge, score=edge, detail=f'min={min_edge}'),\n"
                    "            DecisionCheck('confidence', 'Confidence', confidence >= min_conf, score=confidence, detail=f'min={min_conf}'),\n"
                    "        ]\n\n"
                    "        if not all(c.passed for c in checks):\n"
                    "            failed = [c.key for c in checks if not c.passed]\n"
                    "            return StrategyDecision('skipped', f'Failed: {failed}', checks=checks)\n\n"
                    "        size = min(base_size * (1 + edge / 50), max_size)\n"
                    "        return StrategyDecision('selected', 'Momentum confirmed', score=edge * confidence, size_usd=size, checks=checks)\n\n"
                    "    def should_exit(self, position, market_state):\n"
                    "        config = getattr(position, 'config', None) or {}\n"
                    "        ctx = getattr(position, 'strategy_context', None) or {}\n"
                    "        current = market_state.get('current_price')\n"
                    "        entry_mid = ctx.get('entry_mid', 0)\n"
                    "        direction = ctx.get('direction', 'BUY')\n\n"
                    "        if current is not None and entry_mid > 0:\n"
                    "            reversal_threshold = float(config.get('reversal_threshold', 0.03))\n"
                    "            if direction == 'BUY' and current < entry_mid - reversal_threshold:\n"
                    "                return ExitDecision('close', f'Momentum reversed (price dropped to {current:.3f})', close_price=current)\n"
                    "            if direction == 'SELL' and current > entry_mid + reversal_threshold:\n"
                    "                return ExitDecision('close', f'Momentum reversed (price rose to {current:.3f})', close_price=current)\n\n"
                    "        # Fall back to standard TP/SL/trailing\n"
                    "        return self.default_exit_check(position, market_state)\n"
                ),
            },
            "async_with_ai": {
                "description": "Async strategy using LLM and HTTP for detection",
                "source_code": (
                    '"""\n'
                    "Strategy: AI News Scanner\n\n"
                    "Uses LLM to analyze market questions and recent news.\n"
                    '"""\n'
                    "import httpx\n"
                    "from models import Market, Event, ArbitrageOpportunity\n"
                    "from services.strategies.base import BaseStrategy\n"
                    "from services.ai import ai_service\n\n"
                    "class AINewsScanner(BaseStrategy):\n"
                    '    name = "AI News Scanner"\n'
                    '    description = "LLM-powered opportunity detection from market analysis"\n\n'
                    "    default_config = {\n"
                    '        "min_edge_percent": 5.0,\n'
                    '        "max_markets_per_scan": 10,\n'
                    "    }\n\n"
                    "    async def detect_async(self, events, markets, prices):\n"
                    "        # Use detect_async for strategies that need await\n"
                    "        opportunities = []\n"
                    "        limit = int(self.config.get('max_markets_per_scan', 10))\n"
                    "        candidates = [m for m in markets if m.active and not m.closed][:limit]\n"
                    "        for market in candidates:\n"
                    "            # Example: use AI to analyze market question\n"
                    "            # analysis = await ai_service.analyze(market.question)\n"
                    "            # Example: fetch external data with httpx\n"
                    "            # async with httpx.AsyncClient() as client:\n"
                    "            #     resp = await client.get('https://api.example.com/data')\n"
                    "            pass  # Your async logic here\n"
                    "        return opportunities\n"
                ),
            },
        },

        # ── Section 9: Backtesting ───────────────────────────────────
        "backtesting": {
            "description": (
                "Test your strategy code against real data without saving. "
                "Three backtest modes match the three lifecycle phases."
            ),
            "modes": {
                "detect": {
                    "endpoint": "POST /validation/code-backtest",
                    "what_it_does": (
                        "Compiles your source code, runs detect() against the current "
                        "live market snapshot, and returns what opportunities it finds right now."
                    ),
                    "returns": "List of opportunities with ROI, risk score, markets, positions",
                },
                "evaluate": {
                    "endpoint": "POST /validation/code-backtest/evaluate",
                    "what_it_does": (
                        "Compiles your source code, fetches recent trade signals from the DB, "
                        "runs evaluate() on each, and shows which would be selected vs skipped."
                    ),
                    "returns": "List of decisions with checks, scores, and reasons for each signal",
                },
                "exit": {
                    "endpoint": "POST /validation/code-backtest/exit",
                    "what_it_does": (
                        "Compiles your source code, fetches current open positions, "
                        "runs should_exit() on each, and shows which would be closed vs held."
                    ),
                    "returns": "List of exit decisions with action (close/hold/reduce) and reason",
                },
            },
            "request_body": {
                "source_code": "str — Your strategy Python source code",
                "slug": "str — Strategy slug (used to find related signals/positions)",
                "config": "dict | null — Config overrides (merged with default_config)",
            },
        },

        # ── Section 10: Validation ───────────────────────────────────
        "validation": {
            "endpoint": "POST /strategy-manager/validate",
            "description": (
                "Validates strategy source code without saving. Checks syntax, "
                "import safety, blocked calls, and extracts class metadata."
            ),
            "checks_performed": [
                "1. Python syntax (AST parse)",
                "2. Import safety — all imports checked against allow/block lists",
                "3. Blocked calls — exec(), eval(), compile(), __import__(), open(), input()",
                "4. Strategy class found — must extend BaseStrategy",
                "5. At least one method — detect(), detect_async(), or evaluate() required",
                "6. Metadata extraction — name and description from class attributes",
            ],
            "response": {
                "valid": "bool — Whether the source code passes all checks",
                "class_name": "str — Auto-detected class name (e.g., 'MyCustomStrategy')",
                "strategy_name": "str — Value of the name attribute",
                "strategy_description": "str — Value of the description attribute",
                "capabilities": {
                    "has_detect": "bool",
                    "has_detect_async": "bool",
                    "has_evaluate": "bool",
                    "has_should_exit": "bool",
                },
                "errors": "list[str] — Validation errors if not valid",
            },
        },

        # ── Section 11: API Endpoints ────────────────────────────────
        "endpoints": {
            "strategies": {
                "GET /strategy-manager": "List all strategies. Filters: ?type=detect|execute|unified, ?source_key=scanner|crypto, ?enabled=true",
                "GET /strategy-manager/template": "Get starter template source code",
                "GET /strategy-manager/docs": "This documentation",
                "GET /strategy-manager/{id}": "Get one strategy by ID",
                "POST /strategy-manager": "Create a new strategy (source_code required, class_name auto-detected)",
                "PUT /strategy-manager/{id}": "Update strategy (partial — only send fields to change)",
                "DELETE /strategy-manager/{id}": "Delete strategy (system strategies get tombstoned to prevent re-seeding)",
                "POST /strategy-manager/{id}/reload": "Force recompile and reload from stored source code",
                "POST /strategy-manager/{id}/reset-to-factory": "Reset a system strategy to its original seed values",
            },
            "validation": {
                "POST /strategy-manager/validate": "Validate source code without saving",
                "POST /validation/code-backtest": "Run detect() backtest against live market data",
                "POST /validation/code-backtest/evaluate": "Run evaluate() backtest against recent signals",
                "POST /validation/code-backtest/exit": "Run should_exit() backtest against open positions",
            },
        },

        # ── Section 12: Quick Start ──────────────────────────────────
        "quick_start": [
            "1. GET /strategy-manager/template → copy the starter template",
            "2. Edit the class: set name, description, default_config",
            "3. Implement detect() to find opportunities from events/markets/prices",
            "4. POST /strategy-manager/validate with your source_code → check for errors",
            "5. POST /validation/code-backtest with your source_code → see what it finds",
            "6. POST /strategy-manager to save it (set source_key, enabled=true)",
            "7. Optionally implement evaluate() for custom execution gating",
            "8. Optionally implement should_exit() for custom exit logic",
            "9. Use the Strategies page in the UI to monitor, configure, and backtest",
        ],
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

    strategy_name = (req.name or validation["strategy_name"] or slug.replace("_", " ").title()).strip()
    strategy_description = req.description if req.description is not None else validation["strategy_description"]
    class_name = validation["class_name"]

    strategy_id = uuid.uuid4().hex
    status = "unloaded"
    error_message = None

    async with AsyncSessionLocal() as session:
        existing = await session.execute(select(Strategy).where(Strategy.slug == slug))
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
            aliases=[],
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
            tombstone = await session.get(StrategyTombstone, row.slug)
            if tombstone is None:
                session.add(
                    StrategyTombstone(
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


@router.post("/{strategy_id}/reset-to-factory")
async def reset_strategy_to_factory_endpoint(strategy_id: str):
    """Reset a system strategy to its original factory seed definition.

    This restores the strategy's source code, config, config_schema, and
    description to the values shipped with the application. Only works
    for system strategies.
    """
    from services.opportunity_strategy_catalog import reset_strategy_to_factory

    async with AsyncSessionLocal() as session:
        row = await session.get(Strategy, strategy_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Strategy not found")
        if not row.is_system:
            raise HTTPException(
                status_code=400,
                detail="Only system strategies can be reset to factory defaults.",
            )

        result = await reset_strategy_to_factory(session, row.slug)

        # Reload into the unified loader after reset
        if result.get("status") in ("reset", "created"):
            try:
                plugin_loader.load_plugin(row.slug, row.source_code, row.config or None)
            except Exception:
                pass

        return result
