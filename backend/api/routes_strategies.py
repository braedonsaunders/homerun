"""Unified Strategy API Routes

All strategies (detection and execution) live in a single `strategies` table.
This router provides CRUD, validation, reload, template, and docs endpoints.
"""

from __future__ import annotations

from functools import lru_cache
import re
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    AsyncSessionLocal,
    Strategy,
    get_db_session,
)
from services.opportunity_strategy_catalog import (
    build_system_opportunity_strategy_rows,
    ensure_system_opportunity_strategies_seeded,
)
from services.strategy_experiments import (
    create_strategy_experiment,
    list_strategy_experiment_assignments,
    list_strategy_experiments,
    promote_strategy_experiment,
    set_strategy_experiment_status,
    serialize_strategy_experiment,
)
from services.strategy_loader import (
    MULTI_WINDOW_STRATEGY_TEMPLATE,
    STRATEGY_TEMPLATE,
    StrategyValidationError,
    strategy_loader,
    validate_strategy_source,
)
from services.strategy_runtime import refresh_strategy_runtime_if_needed
from services.strategy_versioning import (
    create_strategy_version_snapshot,
    ensure_strategy_version_seeded,
    list_strategy_versions,
    normalize_strategy_version,
    restore_strategy_from_snapshot,
    serialize_strategy_version,
)
from services.strategy_sdk import StrategySDK
from services.strategies.news_edge import (
    news_edge_config_schema,
    news_edge_defaults,
    validate_news_edge_config,
)
from services.strategies.traders_copy_trade import (
    traders_copy_trade_config_schema,
    traders_copy_trade_defaults,
    validate_traders_copy_trade_config,
)
from services.strategy_runtime import bump_strategy_runtime_revisions
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/strategy-manager", tags=["Strategies (Unified)"])

# ---------------------------------------------------------------------------
# Slug / key validation
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]{1,48}[a-z0-9]$")


def _get_crypto_scope_attr(attr_name: str) -> dict:
    """Read a module-level dict from services.strategy_helpers.crypto_scope.

    Used by the unified-docs response to surface the crypto-strategy scope
    defaults / schema without strategies importing the module directly.
    """
    try:
        from services.strategy_helpers import crypto_scope
    except Exception:
        return {}
    val = getattr(crypto_scope, attr_name, None)
    if isinstance(val, dict):
        return dict(val)
    return {}


def _get_crypto_scope_fn(fn_name: str) -> dict:
    """Call a function on services.strategy_helpers.crypto_scope by name."""
    try:
        from services.strategy_helpers import crypto_scope
    except Exception:
        return {}
    fn = getattr(crypto_scope, fn_name, None)
    if callable(fn):
        try:
            return fn()
        except Exception:
            return {}
    return {}


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


def _normalize_strategy_config_for_source(
    source_key: str,
    config: Optional[dict],
    *,
    strategy_slug: Optional[str] = None,
) -> dict:
    normalized_source_key = str(source_key or "scanner").strip().lower()
    normalized_slug = str(strategy_slug or "").strip().lower()
    payload = dict(config or {})
    if normalized_source_key == "traders":
        if normalized_slug == "traders_copy_trade":
            payload = validate_traders_copy_trade_config(payload)
        else:
            payload = StrategySDK.validate_trader_filter_config(payload)
    elif normalized_source_key == "news":
        payload = validate_news_edge_config(payload)
    return StrategySDK.normalize_strategy_retention_config(payload)


def _dedupe_param_fields(raw_fields: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[str] = set()
    for field in raw_fields:
        if not isinstance(field, dict):
            continue
        key = str(field.get("key") or "").strip()
        if not key or key in seen:
            continue
        deduped.append(dict(field))
        seen.add(key)
    return deduped


def _merge_config_schemas(base_schema: dict, extra_schema: dict) -> dict:
    merged = dict(base_schema or {})
    base_fields = _dedupe_param_fields(list(merged.get("param_fields") or []))
    existing_keys = {str(field.get("key") or "").strip() for field in base_fields if isinstance(field, dict)}
    for field in list((extra_schema or {}).get("param_fields") or []):
        if not isinstance(field, dict):
            continue
        key = str(field.get("key") or "").strip()
        if not key or key in existing_keys:
            continue
        base_fields.append(dict(field))
        existing_keys.add(key)
    merged["param_fields"] = base_fields
    return merged


def _default_config_schema_for_source(source_key: str) -> dict:
    normalized_source_key = str(source_key or "scanner").strip().lower()
    if normalized_source_key == "traders":
        base_schema = StrategySDK.trader_filter_config_schema()
    elif normalized_source_key == "news":
        base_schema = news_edge_config_schema()
    else:
        base_schema = {}
    return _merge_config_schemas(base_schema, StrategySDK.strategy_retention_config_schema())


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


class StrategyVersionRestoreRequest(BaseModel):
    reason: Optional[str] = Field(default="manual_restore", max_length=300)
    created_by: Optional[str] = Field(default=None, max_length=120)


class StrategyExperimentCreateRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=200)
    source_key: str = Field(..., min_length=2, max_length=64)
    strategy_key: str = Field(..., min_length=2, max_length=128)
    control_version: int = Field(..., ge=1, le=10000)
    candidate_version: int = Field(..., ge=1, le=10000)
    candidate_allocation_pct: float = Field(default=50.0, ge=0.1, le=99.9)
    scope: dict = Field(default_factory=dict)
    notes: Optional[str] = Field(default=None, max_length=2000)
    created_by: Optional[str] = Field(default=None, max_length=120)


class StrategyExperimentStatusRequest(BaseModel):
    status: str = Field(..., min_length=3, max_length=32)


class StrategyExperimentPromoteRequest(BaseModel):
    promoted_version: Optional[int] = Field(default=None, ge=1, le=10000)
    notes: Optional[str] = Field(default=None, max_length=2000)


# ---------------------------------------------------------------------------
# Helpers — detect capabilities from source code
# ---------------------------------------------------------------------------


def _detect_capabilities(source_code: str) -> dict:
    """Detect strategy capabilities from source code using AST analysis.

    Delegates to strategy_loader's AST-based helpers for accurate detection
    instead of fragile regex matching. BaseStrategy provides default
    on_event(), evaluate(), and should_exit() for all subclasses, so those
    flags are set to True whenever the class extends BaseStrategy.
    """
    from services.strategy_loader import _find_strategy_class, _extract_class_capabilities

    # Fallback result if AST parsing fails
    _fallback = {
        "has_detect": False,
        "has_detect_async": False,
        "has_on_event": False,
        "has_evaluate": False,
        "has_should_exit": False,
    }

    try:
        import ast as _ast

        tree = _ast.parse(source_code)
        class_name = _find_strategy_class(tree)
        if not class_name:
            return _fallback

        caps = _extract_class_capabilities(tree, class_name)

        # BaseStrategy provides default on_event, evaluate(), and should_exit()
        # for ALL strategies. Any class extending BaseStrategy has working defaults.
        extends_base = bool(re.search(r"\bBaseStrategy\b", source_code))
        if extends_base:
            caps["has_on_event"] = True
            caps["has_evaluate"] = True
            caps["has_should_exit"] = True

        return caps
    except Exception:
        return _fallback


def _infer_strategy_type(capabilities: dict) -> str:
    """Infer whether this is a detect, execute, or unified strategy."""
    has_any_detect = capabilities.get("has_detect") or capabilities.get("has_detect_async")
    has_evaluate = capabilities.get("has_evaluate")
    has_should_exit = capabilities.get("has_should_exit")
    if has_any_detect and has_evaluate:
        return "unified"
    if has_evaluate or has_should_exit:
        return "execute"
    return "detect"


@lru_cache(maxsize=1)
def _system_seed_default_config_map() -> dict[str, dict]:
    rows = build_system_opportunity_strategy_rows()
    mapping: dict[str, dict] = {}
    for row in rows:
        slug = str(row.get("slug") or "").strip().lower()
        config = row.get("config")
        if not slug or not isinstance(config, dict):
            continue
        mapping[slug] = dict(config)
    return mapping


def _resolved_strategy_config(row: Strategy) -> dict:
    source_key = str(row.source_key or "scanner").strip().lower()
    slug = str(row.slug or "").strip().lower()
    defaults: dict = {}

    if slug:
        loaded = strategy_loader.get_strategy(slug)
        instance = getattr(loaded, "instance", None) if loaded is not None else None
        if instance is not None:
            configured = getattr(instance, "config", None)
            if isinstance(configured, dict):
                defaults = dict(configured)
            else:
                declared = getattr(instance, "default_config", None)
                if isinstance(declared, dict):
                    defaults = dict(declared)

    if not defaults and bool(row.is_system):
        seed_defaults = _system_seed_default_config_map().get(slug)
        if isinstance(seed_defaults, dict):
            defaults = dict(seed_defaults)

    overrides = dict(row.config or {})
    merged = {**defaults, **overrides}
    return _normalize_strategy_config_for_source(source_key, merged, strategy_slug=slug)


# ---------------------------------------------------------------------------
# Serialisation — unified response from Strategy table
# ---------------------------------------------------------------------------


def _strategy_to_dict(row: Strategy) -> dict:
    """Convert a Strategy ORM row to the API response dict."""
    capabilities = _detect_capabilities(row.source_code or "")
    source_key = row.source_key or "scanner"
    normalized_config = _resolved_strategy_config(row)
    normalized_schema = _merge_config_schemas(
        dict(row.config_schema or {}),
        StrategySDK.strategy_retention_config_schema(),
    )
    return {
        "id": row.id,
        "slug": row.slug,
        "source_key": source_key,
        "name": row.name,
        "description": row.description,
        "source_code": row.source_code,
        "class_name": row.class_name,
        "is_system": bool(row.is_system),
        "enabled": bool(row.enabled),
        "status": row.status,
        "error_message": row.error_message,
        "version": int(row.version or 1),
        "config": normalized_config,
        "config_schema": normalized_schema,
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
    """Return the unified strategy template + curated examples."""
    return {
        "template": STRATEGY_TEMPLATE,
        "examples": [
            {
                "slug": "basic",
                "label": "Basic strategy",
                "description": "Minimal detect/evaluate/should_exit skeleton.",
                "source": STRATEGY_TEMPLATE,
            },
            {
                "slug": "compound_movement",
                "label": "Compound Movement (multi-timeframe)",
                "description": (
                    "5m / 15m / 1h windows on a shared price stream, fires only "
                    "on candle closes when enough timeframes agree. Demonstrates "
                    "StrategySDK.MultiWindow, on_timeframe_close(), and "
                    "StrategySDK.PersistentState."
                ),
                "source": MULTI_WINDOW_STRATEGY_TEMPLATE,
            },
        ],
        "instructions": (
            "Create a class that extends BaseStrategy and implements detect() or "
            "detect_async() for opportunity detection. For execution strategies, "
            "implement evaluate(signal, context). Unified strategies can implement "
            "both detect/detect_async and evaluate/should_exit. Exit-only/manage-only "
            "strategies should set allow_new_entries = False and implement should_exit()."
        ),
        "available_imports": [
            "models (Market, Event, Opportunity) — use Opportunity; ArbitrageOpportunity is removed",
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
            "services.data_source_sdk (DataSourceSDK: list/get/validate/create/update/delete/reload/run/read)",
            "services.traders_sdk (TradersSDK: firehose/strategy/confluence/pool/tracked/groups/tags)",
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
                        "method": "detect(events, markets, prices) -> list[Opportunity]",
                        "async_method": "detect_async(events, markets, prices) -> list[Opportunity]",
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
                            "from config, sizes position using trader risk limits, returns 'selected'"
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
                "accepted_signal_strategy_types": {
                    "type": "list[str]",
                    "required": False,
                    "description": (
                        "Optional evaluate() input allowlist by strategy_type. "
                        "Use this when one strategy evaluates feeder signals emitted by another."
                    ),
                },
                "allow_new_entries": {
                    "type": "bool",
                    "required": False,
                    "description": (
                        "Optional strategy-level entry gate. "
                        "Set False for manage-existing-only bots that should run "
                        "should_exit() without opening new positions."
                    ),
                },
                "timeframe_close_intervals": {
                    "type": "list[str]",
                    "required": False,
                    "description": (
                        "Opt into candle-close callbacks. Set to canonical timeframes "
                        "('5m','15m','1h','4h' — aliases like '5min','1hr','60m','240m' "
                        "also accepted). The default on_event(MARKET_DATA_REFRESH) will "
                        "fire on_timeframe_close(timeframe, boundary_ts, events, markets, prices) "
                        "once per crossing. Empty list = hook never fires."
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
                        "-> Opportunity"
                    ),
                    "description": (
                        "Always returns an Opportunity. Hard rejection filters run in "
                        "QualityFilterPipeline after detection."
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
                    "signature": "detect(self, events: list[Event], markets: list[Market], prices: dict[str, dict]) -> list[Opportunity]",
                    "when_to_use": "CPU-bound strategies with no async I/O needed",
                },
                "async": {
                    "signature": "async detect_async(self, events: list[Event], markets: list[Market], prices: dict[str, dict]) -> list[Opportunity]",
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
                "type": "list[Opportunity]",
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
                    "context['mode']": "str — 'shadow' or 'live'",
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
                "constructor": (
                    "ExitDecision(action, reason, close_price=None, reduce_fraction=None, "
                    "exit_policy=None, payload={})"
                ),
                "action_values": {
                    "close": "Close the entire position at close_price",
                    "hold": "Keep the position open",
                    "reduce": "Partially exit — set reduce_fraction (0-1) for the portion to close",
                },
                "tip": (
                    "Call self.default_exit_check(position, market_state) as a fallback "
                    "after your custom checks. It handles TP/SL/trailing/max-hold/resolution. "
                    "Set exit_policy on the ExitDecision (or declare exit_policies on the "
                    "class) to break the exit into a ladder of child orders — see the "
                    "Advanced Exit Execution section below."
                ),
            },
        },
        # ── Section 5a: Advanced Exit Execution (laddered/chunked) ───
        "advanced_exits": {
            "summary": (
                "Beyond a single sell at the close_price, strategies can request a "
                "laddered exit: the orchestrator splits the position into many small "
                "child orders across several price levels, escalates resting orders "
                "to marketable IOC if they don't fill in time, and reprices on mid "
                "drift. Use this for fast-drawdown stop-losses where one large sell "
                "would walk the book unfavorably. The system stays backwards-compatible: "
                "strategies that don't declare a policy keep the legacy single-order path."
            ),
            "how_to_attach": {
                "description": (
                    "Two ways to attach a policy. Class-level ``exit_policies`` is the "
                    "common case; per-decision override is for runtime adaptation."
                ),
                "class_attribute_example": (
                    "from services.strategy_sdk import StrategySDK\n"
                    "from services.strategies.base import BaseStrategy\n\n"
                    "class MyStrategy(BaseStrategy):\n"
                    "    strategy_type = 'my_strategy'\n"
                    "    exit_policies = {\n"
                    "        'stop_loss': StrategySDK.build_ladder_exit_policy(\n"
                    "            levels=10, step_ticks=1, offset_ticks=3,\n"
                    "            chunk_size=2, distribution='back_loaded',\n"
                    "            escalation_after_seconds=5,\n"
                    "            escalation_action='marketable_ioc',\n"
                    "            reprice_on_mid_drift_bps=80,\n"
                    "        ),\n"
                    "        'take_profit': None,  # legacy single-order\n"
                    "        '*': StrategySDK.build_chunked_exit_policy(chunk_size=5),\n"
                    "    }\n"
                ),
                "per_decision_override_example": (
                    "def should_exit(self, position, market_state):\n"
                    "    if rapid_drop_detected(position):\n"
                    "        policy = StrategySDK.build_ladder_exit_policy(\n"
                    "            levels=10, step_ticks=1, offset_ticks=3, chunk_size=2,\n"
                    "        )\n"
                    "        return ExitDecision('close', 'rapid drop',\n"
                    "                            close_price=position.current_price,\n"
                    "                            exit_policy=policy)\n"
                    "    return self.default_exit_check(position, market_state)\n"
                ),
                "trigger_keys": (
                    "Keys in the exit_policies dict are matched against the close_trigger "
                    "the orchestrator emits: 'stop_loss', 'take_profit', 'trailing_stop', "
                    "'max_hold', 'market_inactive'. Use '*' as a wildcard fallback. The "
                    "per-decision exit_policy override always wins."
                ),
            },
            "exit_policy_fields": {
                "ladder": (
                    "LadderSpec(levels, step_ticks, offset_ticks, distribution) — describes "
                    "the price ladder. levels = number of rungs; step_ticks = ticks between "
                    "rungs (1 = 1¢ on Polymarket); offset_ticks = ticks between trigger and "
                    "the inside-most rung (set > 0 to make the ladder marketable on submit); "
                    "distribution = 'uniform' | 'front_loaded' (more size at inside) | "
                    "'back_loaded' (more size at outside, chris's pattern)."
                ),
                "chunk_size": (
                    "float | None — contracts per child order. When set, the position is "
                    "split into ``ceil(target / chunk_size)`` chunks; the planner auto-bumps "
                    "the per-chunk size at low-priced rungs to satisfy the $1 min-notional "
                    "floor. Without a chunk_size, the planner emits one child per ladder rung."
                ),
                "max_chunks": (
                    "int = 50 — safety cap on the total number of child orders for a single "
                    "exit. Prevents runaway plans on tiny chunk_size values."
                ),
                "order_type_mix": (
                    "list[(tif, weight)] | None — e.g. [('IOC', 0.3), ('GTC', 0.7)] sends "
                    "30% of children as IOC takers (no slippage tolerance, but instant), "
                    "70% as resting GTC limits. The planner places aggressive types on the "
                    "inside-most rungs."
                ),
                "escalation": (
                    "EscalationSpec(after_seconds, action, widen_bps, max_escalations) — "
                    "if a child rests unfilled for ``after_seconds``, take ``action``. "
                    "marketable_ioc = cancel and resubmit as IOC at live mid; widen_bps = "
                    "move the limit ``widen_bps`` toward the inside; abort = cancel."
                ),
                "reprice_on_mid_drift_bps": (
                    "float | None — when |child.price - current_mid| / current_mid * 10000 "
                    "exceeds this value, cancel and re-quote the child to keep the ladder "
                    "aligned with the book. Honors min_reprice_interval_seconds for a "
                    "cancel-storm guard."
                ),
                "min_chunk_notional_usd": (
                    "float = 1.0 — Polymarket's $1 min notional. The planner enforces this "
                    "and auto-bumps chunk size on low-priced rungs to stay venue-legal."
                ),
                "min_reprice_interval_seconds": (
                    "float = 1.0 — minimum wall-clock gap between cancel/replace cycles on "
                    "a single child. Prevents reprice storms when the mid is jittering."
                ),
            },
            "sdk_helpers": {
                "build_ladder_exit_policy": (
                    "StrategySDK.build_ladder_exit_policy(levels=5, step_ticks=1, "
                    "offset_ticks=0, chunk_size=None, distribution='uniform', "
                    "escalation_after_seconds=5.0, escalation_action='marketable_ioc', "
                    "reprice_on_mid_drift_bps=None, order_type_mix=None, ...) -> ExitPolicy"
                ),
                "build_chunked_exit_policy": (
                    "StrategySDK.build_chunked_exit_policy(chunk_size, max_chunks=50, "
                    "escalation_after_seconds=None, ...) -> ExitPolicy   # chunk-only, "
                    "no laddering — useful when book depth is the constraint but the price "
                    "level is correct."
                ),
            },
            "imports": (
                "from services.strategies.base import ExitPolicy, LadderSpec, EscalationSpec\n"
                "from services.strategy_sdk import StrategySDK"
            ),
            "child_order_lifecycle": {
                "summary": (
                    "Each child order has an independent lifecycle. The orchestrator "
                    "tracks fills, escalations, and reprices in pending_live_exit['children'] "
                    "on the row payload."
                ),
                "states": {
                    "planned": "Created by the planner; not yet submitted to the venue.",
                    "submitted": "Resting on the venue (provider order id assigned).",
                    "partial": "Partially filled — still working.",
                    "filled": "Terminal — fully filled.",
                    "cancelled": "Terminal — cancelled (e.g., abort escalation).",
                    "failed": "Terminal — venue rejected; not retried.",
                    "escalated": "Cancelled and rebuilt under the escalation action.",
                },
            },
            "polymarket_notes": (
                "Tick = 1¢ ($0.01). Min notional = $1. Ladders below $0.10 are heavily "
                "constrained — at $0.05 a single $1 chunk is 20 contracts, so ladders "
                "with many rungs collapse. The planner is conservative: when no laddered "
                "plan satisfies min-notional, it falls back to a single sell at the trigger."
            ),
            "tip": (
                "Start with the chris pattern for stop-losses on prediction markets that "
                "can drop several cents in seconds: levels=10, step_ticks=1, offset_ticks=3, "
                "chunk_size=2, distribution='back_loaded', escalation_after_seconds=5 "
                "to marketable_ioc. Leave take_profit on the legacy single-order path."
            ),
        },
        # ── Section 5b: Composable Evaluate Pipeline ────────────────
        "composable_evaluate": {
            "description": (
                "Strategies can opt into a declarative scoring/sizing pipeline "
                "by setting scoring_weights on the class. When set, evaluate() "
                "uses custom_checks() + compute_score() + compute_size() hooks "
                "instead of the base passthrough."
            ),
            "scoring_weights": {
                "type": "ScoringWeights dataclass",
                "import": "from services.strategies.base import ScoringWeights",
                "formula": (
                    "(edge * edge_weight) + (confidence * confidence_weight) "
                    "- (risk_score * risk_penalty) + (market_count * market_count_bonus) "
                    "+ (liquidity_score * liquidity_weight) + structural_bonus (if guaranteed)"
                ),
                "fields": {
                    "edge_weight": "float = 0.55 — Weight for edge percentage",
                    "confidence_weight": "float = 30.0 — Weight for confidence score",
                    "risk_penalty": "float = 8.0 — Penalty per risk score unit",
                    "liquidity_weight": "float = 0.0 — Weight for liquidity (0 = disabled)",
                    "liquidity_divisor": "float = 5000.0 — Liquidity normalization divisor",
                    "market_count_bonus": "float = 0.0 — Bonus per additional market",
                    "structural_bonus": "float = 0.0 — Bonus for guaranteed (structural) arb",
                },
            },
            "sizing_config": {
                "type": "SizingConfig dataclass",
                "import": "from services.strategies.base import SizingConfig",
                "formula": (
                    "base_size * (1 + edge/base_divisor) * (confidence_offset + confidence) * market_scale * risk_scale"
                ),
                "fields": {
                    "base_divisor": "float = 100.0 — Edge normalization divisor",
                    "confidence_offset": "float = 0.75 — Minimum confidence multiplier",
                    "risk_scale_factor": "float = 0.35 — How much risk reduces size",
                    "risk_floor": "float = 0.55 — Minimum risk scale (never below this)",
                    "market_scale_factor": "float = 0.08 — Size bump per additional market",
                    "market_scale_cap": "int = 4 — Max markets for scaling bonus",
                },
            },
            "custom_checks_override": {
                "signature": ("custom_checks(self, signal, context, params, payload) -> list[DecisionCheck]"),
                "description": (
                    "Override to add strategy-specific checks beyond the standard pipeline. "
                    "Called after the standard edge/confidence/risk checks. Return additional "
                    "DecisionCheck objects that must all pass for the signal to be selected."
                ),
                "example": (
                    "def custom_checks(self, signal, context, params, payload):\n"
                    "    liquidity = float(payload.get('liquidity', 0) or 0)\n"
                    "    return [\n"
                    "        DecisionCheck('liquidity', 'Min liquidity', liquidity >= 1000,\n"
                    "                      score=liquidity, detail=f'min=1000'),\n"
                    "    ]"
                ),
            },
            "how_to_opt_in": (
                "Set scoring_weights = ScoringWeights() on your class to use defaults, "
                "or ScoringWeights(edge_weight=0.8, ...) for custom weights. "
                "Optionally set sizing_config = SizingConfig(...) for custom sizing. "
                "Override compute_score() or compute_size() for fully custom logic."
            ),
        },
        # ── Section 5c: Event Subscriptions ─────────────────────────
        "event_subscriptions": {
            "description": (
                "Strategies subscribe to scanner/worker data events. The scanner now "
                "runs a hybrid loop: periodic full reconciliation plus reactive "
                "market_data_refresh micro-batches triggered by WebSocket price moves."
            ),
            "how_to_subscribe": (
                "Set subscriptions = [EventType.MARKET_DATA_REFRESH] (or other EventType constants) "
                "on your class. Implement on_event(self, event: DataEvent) -> list[Opportunity]. "
                "Scanner strategies run in realtime against incremental market updates by default."
            ),
            "data_event_types": {
                "price_change": {
                    "description": "Low-level token price update from WS feed",
                    "payload_fields": "token_id, old_price, new_price, source",
                },
                "market_data_refresh": {
                    "description": (
                        "Scanner strategy batch event. Emitted as full reconciliation and as "
                        "reactive/timer fast-scan batches."
                    ),
                    "scan_modes": [
                        "full_reconcile",
                        "fast_timer",
                        "realtime_reactive",
                    ],
                    "payload_fields": (
                        "markets, events, prices, scan_mode, changed_token_ids, changed_market_ids, affected_market_ids"
                    ),
                },
                "market_resolved": {
                    "description": "A market outcome was determined",
                    "payload_fields": "market_id, winning_outcome",
                },
                "crypto_update": {
                    "description": "Crypto market data from crypto worker",
                    "payload_fields": "payload (crypto-specific data)",
                },
                "weather_update": {
                    "description": "Weather forecast data from weather worker",
                    "payload_fields": "payload (forecast data)",
                },
                "trader_activity": {
                    "description": "Smart wallet / copy trading signal from traders worker",
                    "payload_fields": "payload (wallet activity data)",
                },
                "news_event": {
                    "description": "Breaking news signal from news worker",
                    "payload_fields": "payload (news data)",
                },
            },
            "data_event_structure": {
                "type": "DataEvent (frozen dataclass)",
                "import": "from services.data_events import DataEvent",
                "fields": {
                    "event_type": "str — One of the event type keys above",
                    "source": "str — Which worker/service emitted the event",
                    "timestamp": "datetime — When the event occurred",
                    "market_id": "str | None — Market this event relates to",
                    "token_id": "str | None — Token this event relates to",
                    "payload": "dict — Event-type-specific data",
                    "old_price": "float | None — Previous price (for price_change)",
                    "new_price": "float | None — New price (for price_change)",
                    "markets": "list | None — Full market list (for market_data_refresh)",
                    "events": "list | None — Full event list (for market_data_refresh)",
                    "prices": "dict | None — Full price dict (for market_data_refresh)",
                    "scan_mode": "str | None — full_reconcile | fast_timer | realtime_reactive",
                    "changed_token_ids": "list[str] | None — Tokens that moved in reactive batches",
                    "changed_market_ids": "list[str] | None — Markets whose prices changed",
                    "affected_market_ids": "list[str] | None — Markets passed into this strategy batch",
                },
            },
            "on_event_method": {
                "signature": "async on_event(self, event: DataEvent) -> list[Opportunity]",
                "description": (
                    "Called by the event dispatcher when a subscribed event fires. "
                    "Return a list of detected opportunities (may be empty). "
                    "Default: no-op (returns empty list)."
                ),
            },
        },
        # ── Section 5d: Quality Filter Pipeline ─────────────────────
        "quality_filter": {
            "description": (
                "After strategies detect opportunities and deduplication runs, "
                "the QualityFilterPipeline evaluates every opportunity with a "
                "full audit trail. Each filter produces a FilterResult with "
                "threshold vs actual value, so you can see exactly why an "
                "opportunity was accepted or rejected."
            ),
            "import": "from services.quality_filter import QualityFilterPipeline, QualityReport, FilterResult",
            "pipeline_class": {
                "method": "evaluate(opp) -> QualityReport",
                "description": (
                    "Runs all quality filters on an Opportunity. "
                    "Returns a QualityReport with pass/fail and full filter breakdown."
                ),
            },
            "quality_report": {
                "fields": {
                    "opportunity_id": "str — stable_id or id of the opportunity",
                    "passed": "bool — True if all filters passed",
                    "filters": "list[FilterResult] — Individual filter results",
                    "rejection_reasons": "list[str] — Human-readable reasons for failed filters (property)",
                },
            },
            "filter_result": {
                "fields": {
                    "filter_name": "str — Machine-readable filter identifier",
                    "passed": "bool — Whether this filter passed",
                    "reason": "str — Human-readable explanation",
                    "threshold": "Any — The threshold value used for comparison",
                    "actual_value": "Any — The actual value from the opportunity",
                },
            },
            "filters_applied": [
                "min_roi — ROI >= MIN_PROFIT_THRESHOLD",
                "directional_roi_cap — Directional ROI <= 120%",
                "plausible_roi — Guaranteed ROI <= MAX_PLAUSIBLE_ROI",
                "max_legs — Number of markets <= MAX_TRADE_LEGS",
                "leg_liquidity — Total liquidity >= MIN_LIQUIDITY_PER_LEG * num_legs",
                "min_liquidity — Min market liquidity >= MIN_LIQUIDITY_HARD",
                "min_position_size — Max position >= MIN_POSITION_SIZE",
                "min_absolute_profit — Absolute profit >= MIN_ABSOLUTE_PROFIT",
                "resolution_timeframe — Days to resolution <= MAX_RESOLUTION_MONTHS * 30",
                "annualized_roi — Annualized ROI >= MIN_ANNUALIZED_ROI",
            ],
        },
        # ── Section 5e: Platform Hooks ──────────────────────────────
        "platform_hooks": {
            "description": (
                "The platform may override strategy decisions for safety reasons "
                "(trading window restrictions, risk limits, stacking guards, size caps). "
                "Strategies can override these hooks to observe when overrides happen."
            ),
            "on_blocked": {
                "signature": "on_blocked(self, signal, reason: str, context: dict) -> None",
                "description": (
                    "Called when the platform blocks a signal from this strategy. "
                    "Override to log, alert, or adjust strategy behavior when blocked. "
                    "Default: no-op."
                ),
                "called_when": [
                    "Trading window — signal arrives outside configured UTC trading hours",
                    "Risk manager veto — daily loss, exposure, or position limits exceeded",
                    "Stacking guard — market already has an open position (allow_averaging=false)",
                ],
            },
            "on_size_capped": {
                "signature": "on_size_capped(self, original_size: float, capped_size: float, reason: str) -> None",
                "description": (
                    "Called when the platform caps this strategy's position size. "
                    "Override to track how often sizing gets overridden. "
                    "Default: no-op."
                ),
                "called_when": [
                    "Size exceeds max_trade_notional_usd — capped to the limit before risk evaluation",
                ],
            },
            "on_timeframe_close": {
                "signature": (
                    "on_timeframe_close(self, timeframe: str, boundary_ts: datetime, "
                    "events, markets, prices) -> list[Opportunity] | Awaitable[...]"
                ),
                "description": (
                    "Fires whenever a wall-clock boundary for an opted-in timeframe is "
                    "crossed. Opt in by setting class attribute timeframe_close_intervals "
                    "(e.g. ['5m','15m','1h','4h']). Use this for compound-movement / "
                    "candle-close strategies that should act once per closed window "
                    "rather than on every tick. Returned opportunities are merged with "
                    "those returned from detect()/detect_async() in the same refresh."
                ),
                "called_when": [
                    "A wall-clock boundary is crossed since this strategy last saw a "
                    "MARKET_DATA_REFRESH (boundaries are unix-epoch aligned).",
                    "Skipped cycles still produce exactly one call per crossing — the "
                    "hook does not replay older boundaries.",
                ],
                "default_behavior": "No-op — returns []. Override to act on candle closes.",
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
        # ── Section 6b: StrategySDK Reference ───────────────────────
        "strategy_sdk": {
            "summary": (
                "StrategySDK is the stable helper API for strategy authors. "
                "Business logic still lives in your editable strategy class: "
                "detect(), evaluate(), should_exit(), and optional hooks."
            ),
            "business_logic_contract": [
                (
                    "Runtime authority is the strategy source_code stored in the DB row. "
                    "If you edit Python logic, runtime behavior changes after reload."
                ),
                (
                    "StrategySDK helpers expose data access, config defaults/schemas, "
                    "normalization, and utility math; they do not replace your strategy logic."
                ),
                (
                    "Hard platform controls (risk manager, trading window, quality pipeline) "
                    "can still gate execution after your strategy decision."
                ),
            ],
            "signal_routing_controls": {
                "accepted_signal_strategy_types": (
                    "Optional class attribute (list[str]): additional strategy_type values "
                    "your evaluate() should accept from the same source."
                ),
                "allow_new_entries": (
                    "Optional class attribute (bool): set False to disable new entries "
                    "and run this strategy in manage-existing-only mode."
                ),
                "strategy_params.accepted_signal_strategy_types": (
                    "Runtime override for routing allowlist; list or comma-separated string."
                ),
                "strategy_params.enable_live_market_context": (
                    "Runtime override for live context enrichment; true/false."
                ),
                "strategy_params.allow_new_entries": (
                    "Runtime override for entry gating; true/false."
                ),
                "strategy_params.disable_new_entries": (
                    "Runtime override alias; true disables new entries."
                ),
            },
            "configuration_helpers": {
                "StrategySDK.trader_filter_defaults()": "Tracked-trader filtering defaults",
                "StrategySDK.trader_filter_config_schema()": "Schema for tracked-trader filters",
                "StrategySDK.trader_scope_defaults()": "Default wallet scope (tracked/pool/individual/group)",
                "StrategySDK.trader_scope_fields_schema()": "Schema for wallet-scope object fields",
                "StrategySDK.trader_runtime_defaults()": "Default runtime metadata/schedule values",
                "StrategySDK.trader_runtime_fields_schema()": "Schema for runtime metadata fields",
                "StrategySDK.trader_risk_defaults()": "Default trader risk controls",
                "StrategySDK.trader_risk_fields_schema()": "Schema for trader risk controls",
                "StrategySDK.trader_opportunity_filter_defaults()": "Default trader opportunity filters",
                "StrategySDK.trader_opportunity_filter_config_schema()": "Schema for trader opportunity filters",
                "traders_copy_trade_defaults()": "Default explicit traders copy-trade strategy params",
                "traders_copy_trade_config_schema()": "Schema for explicit traders copy-trade params",
                "StrategySDK.pool_eligibility_defaults()": "Default smart-pool selection thresholds",
                "StrategySDK.pool_eligibility_config_schema()": "Schema for pool eligibility tuning",
                "news_edge_defaults()": "Default news strategy filters",
                "news_edge_config_schema()": "Schema for news strategy filters",
                "StrategySDK.strategy_retention_config_schema()": "Schema for max_opportunities and retention_window",
            },
            "validation_helpers": {
                "StrategySDK.validate_trader_filter_config(config)": "Normalize and clamp trader filter config",
                "StrategySDK.validate_trader_scope_config(config)": "Normalize trader wallet scope config",
                "StrategySDK.validate_trader_runtime_metadata(config)": "Normalize schedule/tags/runtime metadata",
                "StrategySDK.validate_trader_risk_config(config)": "Normalize trader risk limits",
                "StrategySDK.validate_trader_opportunity_filter_config(config)": "Normalize trader opportunity filters",
                "validate_traders_copy_trade_config(config)": "Normalize explicit traders copy-trade params",
                "StrategySDK.validate_pool_eligibility_config(config)": "Normalize smart-pool eligibility params",
                "validate_news_edge_config(config)": "Normalize news filter params",
                "StrategySDK.normalize_strategy_retention_config(config)": "Normalize retention aliases to retention_max_age_minutes",
                "StrategySDK.normalize_reverse_intent(value, ...)": "Normalize stop-and-reverse payload for should_exit()",
                "StrategySDK.parse_duration_minutes(value)": "Parse durations like 15m, 2h, 3d into minutes",
            },
            "price_window_helpers": {
                "StrategySDK.PriceWindow(window_seconds=...)": (
                    "Rolling per-stream price window. Maintain a "
                    "dict[token_id, PriceWindow] for one window per outcome / "
                    "feed. Methods: record(price, ts_ms), log_return(seconds_ago), "
                    "stddev(), realized_volatility_bps_per_sec(), distance_bps()."
                ),
                "StrategySDK.MultiWindow(lookbacks={...})": (
                    "Fan one price stream into N rolling lookbacks at different "
                    "sizes — the canonical primitive for compound-movement / "
                    "multi-timeframe-confirmation strategies. "
                    "lookbacks={'5m':300,'15m':900,'1h':3600,'4h':14400}. "
                    "Methods: record(price), log_returns() -> dict, "
                    "all_agree(direction, min_return), aligned_count(...), "
                    "realized_volatility_bps_per_sec()."
                ),
                "StrategySDK.timeframes_agree(returns_by_label, direction, min_count, min_return)": (
                    "Module helper. Returns True iff at least min_count labels "
                    "agree on direction. Accepts the dict shape MultiWindow.log_returns() emits."
                ),
                "StrategySDK.weighted_signal(returns_by_label, weights)": (
                    "Module helper. Weighted average of per-label log returns; "
                    "renormalises over labels that contributed (so partial data "
                    "still produces a meaningful signal)."
                ),
                "BaseStrategy.timeframe_close_intervals = ['5m','15m','1h','4h']": (
                    "Class attribute. Opt into wall-clock candle-close callbacks: "
                    "the default on_event(MARKET_DATA_REFRESH) will additionally "
                    "fire on_timeframe_close(timeframe, boundary_ts, events, markets, prices) "
                    "exactly once per crossing. Boundaries are unix-epoch aligned "
                    "so multiple workers / restarts agree on close times."
                ),
            },
            "persistent_state_helper": {
                "summary": (
                    "BaseStrategy.state is in-memory only — lost on worker restart. "
                    "StrategySDK.PersistentState is the durable counterpart, backed by "
                    "the strategy_persistent_state table."
                ),
                "StrategySDK.PersistentState(strategy_slug)": (
                    "Construct a per-strategy key/value cache. "
                    "Pass strategy_slug=self.strategy_type when instantiating from a strategy."
                ),
                "await state.load()": (
                    "Hydrate the cache from the DB. Call once after instantiation."
                ),
                "state.get(key, default=None)": "Read from cache (deep copy returned).",
                "state.set(key, value)": (
                    "Update cache + mark dirty. value must be JSON-serialisable."
                ),
                "state.delete(key)": "Remove from cache + queue DB delete.",
                "state.dirty": "True when there are unflushed writes.",
                "await state.flush()": "Persist dirty entries. No-op when clean.",
            },
            "market_and_execution_helpers": {
                "StrategySDK.opposite_direction(direction)": "Map buy_yes <-> buy_no",
                "StrategySDK.build_reverse_intent(...)": "Build validated reverse intent payload for ExitDecision.payload",
                "StrategySDK.get_live_price(market, prices, side='YES')": "Resolve best available live price",
                "StrategySDK.get_spread_bps(market, prices, side='YES')": "Bid-ask spread in basis points",
                "StrategySDK.get_ws_mid_price(token_id)": "Live WebSocket mid price",
                "StrategySDK.get_ws_spread_bps(token_id)": "Live WebSocket spread in bps",
                "StrategySDK.get_chainlink_price(asset)": "Latest oracle price",
                "StrategySDK.calculate_fees(total_cost, expected_payout, n_legs)": "Comprehensive fee estimate",
                "StrategySDK.resolve_position_sizing(...)": "One-call sizing output with tradeability gate",
                "StrategySDK.get_order_book_depth(...)": "VWAP/slippage/fill-probability estimate",
                "StrategySDK.get_book_levels(...)": "Raw orderbook levels",
                "StrategySDK.get_price_history(token_id, max_snapshots)": "Recent price snapshots",
                "StrategySDK.get_price_change(token_id, lookback_seconds)": "Lookback price delta summary",
                "StrategySDK.get_recent_trades(token_id, max_trades)": "Recent trade tape",
                "StrategySDK.get_trade_volume(token_id, lookback_seconds)": "Buy/sell volume summary",
                "StrategySDK.get_buy_sell_imbalance(token_id, lookback_seconds)": "Order-flow imbalance in [-1, 1]",
            },
            "crypto_helpers": {
                # Surface backed by services/strategy_helpers/crypto_strategy_utils.py.
                # Strategies access these via the StrategySDK.crypto namespace —
                # no direct import of the helpers module needed.
                "StrategySDK.crypto.pick_oracle_source(row, prefer=...)": "Pick the freshest oracle source from a payload row, with binance_direct preferred by default",
                "StrategySDK.crypto.extract_oracle_status(live_market=..., payload=..., now_ms=...)": "Layered oracle status extraction (price, age, source, freshness flags) from live + payload candidates",
                "StrategySDK.crypto.parse_oracle_point(raw, source_hint=..., now_ms=...)": "Normalize a single by-source oracle entry into {source, price, updated_at_ms, age_ms}",
                "StrategySDK.crypto.normalize_oracle_source(value)": "Canonicalize labels to chainlink / binance_direct / binance / lowercase passthrough",
                "StrategySDK.crypto.resolve_oracle_availability(price=, price_to_beat=, age_ms=, updated_at_ms=)": "Compute freshness + directional availability flags from oracle components",
                "StrategySDK.crypto.to_epoch_ms(value)": "Coerce numeric timestamp to epoch ms (seconds auto-detected by magnitude)",
                "StrategySDK.crypto.compute_age_ms(age_ms=, age_seconds=, updated_at_ms=, now_ms=)": "Resolve oracle age in ms from any of three input shapes",
                "StrategySDK.crypto.normalize_timeframe(value)": "Canonicalize 5m/15m/1h/4h variants",
                "StrategySDK.crypto.timeframe_seconds(value)": "Window length in seconds for a Polymarket crypto timeframe",
                "StrategySDK.crypto.default_min_seconds_left_for_entry(timeframe)": "Per-timeframe minimum seconds-left runway for a fresh entry",
                "StrategySDK.crypto.default_max_market_data_age_ms(timeframe)": "Per-timeframe market-data freshness cap",
                "StrategySDK.crypto.default_max_oracle_age_ms(timeframe)": "Per-timeframe oracle staleness cap",
                "StrategySDK.crypto.build_binary_crypto_market(row)": "Construct a typed Market from a crypto_update worker row",
                "StrategySDK.crypto.CryptoMarketFetcher(gamma_url=, ttl_seconds=)": "TTL-cached Gamma fetcher for live + upcoming crypto Up/Down markets across all configured series",
                "StrategySDK.crypto.get_crypto_market_fetcher()": "Process-wide CryptoMarketFetcher singleton — auto-subscribes discovered tokens to the WS feed",
                "StrategySDK.crypto.get_crypto_series()": "List of (series_id, asset, timeframe) tuples for every configured Polymarket crypto series",
                "StrategySDK.crypto.crypto_direction_allowed(params, regime, active_mode, direction, timeframe, seconds_left)": "Decide whether a direction (buy_yes/buy_no) is allowed for the given mode + regime; returns (allowed, detail)",
                "StrategySDK.crypto.crypto_should_flatten_resolution_risk(params, timeframe, seconds_left, pnl_percent, exit_headroom_ratio, take_profit_armed)": "Decide whether to force-flatten an open position because resolution is too close to risk an in-flight fill",
                "StrategySDK.crypto.crypto_param_value(config, base_key, timeframe)": "Resolve a config value with timeframe-suffix override (e.g. take_profit_pct_5m)",
                "StrategySDK.crypto.SubStrategy": "Enum of canonical sub-strategy modes (MAKER_QUOTE / DIRECTIONAL_EDGE / CONVERGENCE)",
                "StrategySDK.crypto.SUB_STRATEGY_ALIASES": "Map of human-readable aliases to SubStrategy enum values (e.g. \"maker\", \"passive_quote\" → MAKER_QUOTE)",
                "StrategySDK.crypto.normalize_sub_strategy(value)": "Canonicalize a sub-strategy token to a SubStrategy enum value; None for unknown tokens",
                "StrategySDK.crypto.resolve_enabled_sub_strategies(config)": "Return the set of SubStrategy values enabled in config['enabled_sub_strategies']",
                "StrategySDK.crypto.resolve_enabled_active_modes(config)": "Return the set of active_mode strings (dispatch keys) implied by enabled_sub_strategies",
                "StrategySDK.crypto.CryptoCandidate(market, asset, timeframe, yes_price, no_price, oracle_price, price_to_beat)": "Dataclass for a binary crypto market identified for sub-strategy scoring",
                "StrategySDK.crypto.SubStrategyScore(strategy, score, reason, params)": "Score + metadata returned by sub-strategy scoring functions",
                "StrategySDK.crypto.CRYPTO_STRATEGY_MODES": "Set of canonical mode strings: auto / directional / maker_quote / convergence",
                "StrategySDK.crypto.CRYPTO_REGIMES": "Set of canonical regime strings: opening / mid / closing",
                "StrategySDK.crypto.normalize_strategy_mode(value)": "Canonicalize a mode string; falls back to 'auto'",
                "StrategySDK.crypto.normalize_regime(value)": "Canonicalize a regime string; falls back to 'mid'",
                "StrategySDK.crypto.normalize_crypto_asset(value)": "Canonicalize asset symbol (XBT → BTC, otherwise uppercased trim)",
                "StrategySDK.crypto.coerce_float(value, default, lo, hi)": "Coerce to float clamped to [lo, hi]; NaN/inf → default",
                "StrategySDK.crypto.as_list(value)": "Coerce list/tuple/set or comma-string to a list",
                "StrategySDK.crypto.normalize_scope(value, normalizer)": "Apply normalizer to each item, dedupe, drop empties",
                "StrategySDK.crypto.normalize_regime_scope(value)": "Return the set of canonical regimes implied by value",
                "StrategySDK.crypto.seconds_left_from_row(row, fallback_seconds=...)": "Time-to-resolution from a worker row",
                "StrategySDK.crypto.spread_pct_from_row(row)": "Bid-ask spread as a fraction from a worker row",
                "StrategySDK.crypto.market_ml_probability_yes(row)": "Extract clipped ML-predicted YES probability from a row",
                "StrategySDK.crypto.history_cancel_peak(history_tail)": "Peak cancel rate observed in a maker history tail",
                "StrategySDK.crypto.taker_fee_pct(entry_price)": "Polymarket taker fee as a fraction of price",
                "StrategySDK.crypto.fee_aware_min_edge_pct(price, multiplier=2.0)": "Minimum edge-percent threshold needed to clear taker fees by multiplier",
                "StrategySDK.crypto.first_present(*values)": "First non-None value (skips None only — preserves 0/empty/False)",
                "StrategySDK.crypto.normalize_ratio(value)": "Coerce to a [0, 1] ratio, or None",
                "StrategySDK.crypto.normalize_signed_ratio(value)": "Coerce to a [-1, +1] ratio, or None",
                "StrategySDK.crypto.bounded_sigmoid(z)": "Sigmoid clamped to safe range",
                "StrategySDK.crypto.parse_datetime_utc(value)": "Best-effort datetime parsing, always tz-aware UTC",
            },
            "llm_and_news_helpers": {
                "StrategySDK.ask_llm(...)": "Text LLM call with strategy-safe fallback",
                "StrategySDK.ask_llm_json(...)": "Structured JSON LLM call",
                "StrategySDK.get_recent_news(query, max_articles)": "Recent news search",
                "StrategySDK.get_news_for_market(market, max_articles)": "Semantically matched market news",
            },
            "trader_data_helpers": {
                "StrategySDK.get_trader_firehose_signals(...)": "Raw tracked-trader firehose rows",
                "StrategySDK.get_trader_strategy_signals(...)": "Rows after strategy filtering",
                "StrategySDK.get_trader_confluence_signals(...)": "Confluence detector outputs",
                "StrategySDK.get_pooled_traders(...)": "Current smart-pool wallets",
                "StrategySDK.get_tracked_traders(...)": "Tracked wallets and optional activity",
                "StrategySDK.get_trader_groups(...)": "Trader groups and optional members",
                "StrategySDK.get_trader_tags()": "Tag definitions and wallet counts",
                "StrategySDK.get_traders_by_tag(tag_name, limit)": "Wallets for a tag",
            },
            "news_edge_defaults": news_edge_defaults(),
            "news_edge_schema": news_edge_config_schema(),
            "traders_copy_trade_defaults": traders_copy_trade_defaults(),
            "traders_copy_trade_schema": traders_copy_trade_config_schema(),
        },
        # ── Section 7: Available Imports ──────────────────────────────
        "imports": {
            "description": (
                "Strategies run in a sandboxed environment. Only approved imports are allowed. "
                "Import validation happens at save time via AST analysis — no code is executed."
            ),
            "app_modules": {
                "models": "Market, Event, Opportunity — core data types (ArbitrageOpportunity is removed)",
                "services.strategies.base": (
                    "BaseStrategy, StrategyDecision, ExitDecision, DecisionCheck, "
                    "ExitPolicy, LadderSpec, EscalationSpec, ScaleOutConfig, ScaleOutTarget"
                ),
                "services.ai": "LLM integration — call AI models from your strategy",
                "services.news": "News analysis services",
                "services.weather": "Weather signal engine",
                "services.optimization": "Parameter optimization utilities",
                "services.ws_feeds": "WebSocket market data feeds",
                "services.chainlink_feed": "Chainlink oracle price feeds",
                "services.fee_model": "Fee calculation model",
                "services.strategy_sdk": (
                    "Strategy utilities and the canonical entry point for all helpers. "
                    "StrategySDK.* covers orderbook, sizing, news, AI, fees, traders, and more. "
                    "StrategySDK.crypto.* re-exports services.strategy_helpers.crypto_strategy_utils — "
                    "use this for crypto oracle/timeframe/fee helpers without a direct import."
                ),
                "services.data_source_sdk": (
                    "Full source SDK: list/get/validate/create/update/delete/reload/run and record access"
                ),
                "config": "Application settings (settings object)",
                "utils": "Shared utility functions",
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
                "threading",
                "itertools",
                "functools",
                "operator",
                "copy",
                "decimal",
                "fractions",
                "calendar",
                "dataclasses",
                "typing",
                "abc",
                "enum",
                "hashlib",
                "hmac",
                "base64",
                "uuid",
                "urllib.parse",
                "logging",
                "bisect",
                "heapq",
                "textwrap",
                "string",
                "concurrent",
                "pathlib",
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
        # ── Section 7b: Data Source SDK ─────────────────────────────
        "data_source_sdk": {
            "description": (
                "Strategies can consume and operate on DB-backed data sources during detect/evaluate/exit. "
                "Use DataSourceSDK directly, or StrategySDK wrappers when you want soft-failure handling."
            ),
            "imports": {
                "direct": "from services.data_source_sdk import DataSourceSDK",
                "wrapped": "from services.strategy_sdk import StrategySDK",
            },
            "when_to_use": {
                "detect": "Pull supplemental signals (news/events/weather/crypto) before creating opportunities.",
                "evaluate": "Re-check latest data before selecting or blocking a signal.",
                "exit": "Close/reduce positions when external sources indicate regime change.",
            },
            "read_methods": {
                "DataSourceSDK.get_records": {
                    "signature": (
                        "await DataSourceSDK.get_records(source_slug=None, source_slugs=None, "
                        "limit=200, geotagged=None, category=None, since=None)"
                    ),
                    "description": "Read normalized records with source/category/time filters.",
                },
                "DataSourceSDK.get_latest_record": {
                    "signature": "await DataSourceSDK.get_latest_record(source_slug, external_id=None)",
                    "description": "Read newest record for a source (optionally one upstream id).",
                },
                "DataSourceSDK.get_recent_runs": {
                    "signature": "await DataSourceSDK.get_recent_runs(source_slug, limit=20)",
                    "description": "Read run history for source-health aware gating.",
                },
            },
            "management_methods": {
                "DataSourceSDK.run_source": {
                    "signature": "await DataSourceSDK.run_source(source_slug, max_records=500)",
                    "description": "Trigger on-demand ingestion during strategy execution.",
                },
                "DataSourceSDK.list_sources": {
                    "signature": "await DataSourceSDK.list_sources(enabled_only=True, source_key=None, include_code=False)",
                    "description": "Discover available sources and runtime state.",
                },
                "DataSourceSDK.get_source": {
                    "signature": "await DataSourceSDK.get_source(source_slug, include_code=True)",
                    "description": "Inspect one source definition by slug.",
                },
                "DataSourceSDK.validate_source": {
                    "signature": "DataSourceSDK.validate_source(source_code, class_name=None)",
                    "description": "Validate generated source code before create/update.",
                },
                "DataSourceSDK.create_source": {
                    "signature": "await DataSourceSDK.create_source(slug=..., source_code=..., ...)",
                    "description": "Create new source definitions programmatically.",
                },
                "DataSourceSDK.update_source": {
                    "signature": "await DataSourceSDK.update_source(source_slug, ...)",
                    "description": "Update metadata/code/config and reload runtime.",
                },
                "DataSourceSDK.delete_source": {
                    "signature": "await DataSourceSDK.delete_source(source_slug, unlock_system=False, ...)",
                    "description": "Delete a source (system sources require unlock_system=True).",
                },
                "DataSourceSDK.reload_source": {
                    "signature": "await DataSourceSDK.reload_source(source_slug)",
                    "description": "Recompile/reload runtime without changing source code.",
                },
            },
            "strategy_sdk_wrappers": {
                "StrategySDK.get_data_records": "Wrapper for DataSourceSDK.get_records()",
                "StrategySDK.get_latest_data_record": "Wrapper for DataSourceSDK.get_latest_record()",
                "StrategySDK.run_data_source": "Wrapper for DataSourceSDK.run_source()",
                "StrategySDK.list_data_sources": "Wrapper for DataSourceSDK.list_sources()",
                "StrategySDK.get_data_source": "Wrapper for DataSourceSDK.get_source()",
                "StrategySDK.validate_data_source": "Wrapper for DataSourceSDK.validate_source()",
                "StrategySDK.create_data_source": "Wrapper for DataSourceSDK.create_source()",
                "StrategySDK.update_data_source": "Wrapper for DataSourceSDK.update_source()",
                "StrategySDK.delete_data_source": "Wrapper for DataSourceSDK.delete_source()",
                "StrategySDK.reload_data_source": "Wrapper for DataSourceSDK.reload_source()",
                "StrategySDK.get_data_source_runs": "Wrapper for DataSourceSDK.get_recent_runs()",
            },
            "examples": {
                "read_records": (
                    "records = await DataSourceSDK.get_records(\n"
                    "    source_slug='events_gdelt_tensions',\n"
                    "    category='conflict',\n"
                    "    geotagged=True,\n"
                    "    limit=100,\n"
                    ")\n"
                    "if not records:\n"
                    "    return StrategyDecision('skipped', 'No recent conflict records')"
                ),
                "run_then_read": (
                    "await StrategySDK.run_data_source('events_gdelt_tensions', max_records=200)\n"
                    "latest = await StrategySDK.get_latest_data_record('events_gdelt_tensions')\n"
                    "if latest and latest.get('category') == 'conflict':\n"
                    "    ...  # feed into detect/evaluate logic"
                ),
            },
            "guidance": [
                "Prefer source_slug constants; avoid hard-coding IDs.",
                "Filter records by category/since/geotagged to keep evaluation deterministic.",
                "Use run_source sparingly inside hot loops; it performs real ingestion work.",
                "Use StrategySDK wrappers when failures should degrade gracefully.",
            ],
        },
        # ── Section 7c: Trader Data SDK ────────────────────────────
        "trader_data_sdk": {
            "description": (
                "Strategies can query trader intelligence datasets in a first-class way "
                "via StrategySDK (firehose rows, strategy-filtered rows, confluence, "
                "pooled/tracked traders, groups, and tags)."
            ),
            "imports": {
                "wrapped": "from services.strategy_sdk import StrategySDK",
                "direct": "from services.traders_sdk import TradersSDK",
                "advanced_raw": (
                    "from services.trader_data_access import get_trader_firehose_signals, "
                    "get_strategy_filtered_trader_signals, get_trader_confluence_signals, "
                    "get_pooled_traders, get_tracked_traders, get_trader_groups, "
                    "get_trader_tags, get_traders_by_tag"
                ),
            },
            "datasets": {
                "firehose": "Raw tracked-trader firehose rows with canonical source_flags/source_breakdown.",
                "strategy_filtered": "Rows after traders_confluence strategy gates (tradeable/actionable signals).",
                "confluence": "Active confluence detector outputs by strength/tier.",
                "pool": "Smart-pool membership rows (tier, scores, pool flags, tags).",
                "tracked": "Tracked wallets with PnL stats and optional recent-activity enrichment.",
                "groups": "Active trader groups with optional member payloads.",
                "tags": "Tag definitions and wallet counts, plus wallets per tag.",
            },
            "strategy_sdk_methods": {
                "StrategySDK.get_trader_firehose_signals": (
                    "await StrategySDK.get_trader_firehose_signals(limit=250, "
                    "include_filtered=False, include_source_context=True)"
                ),
                "StrategySDK.get_trader_strategy_signals": (
                    "await StrategySDK.get_trader_strategy_signals(limit=50, include_filtered=False)"
                ),
                "StrategySDK.get_trader_confluence_signals": (
                    "await StrategySDK.get_trader_confluence_signals(min_strength=0.0, min_tier='WATCH', limit=50)"
                ),
                "StrategySDK.get_pooled_traders": (
                    "await StrategySDK.get_pooled_traders(limit=200, tier=None, "
                    "include_blacklisted=True, tracked_only=False)"
                ),
                "StrategySDK.get_tracked_traders": (
                    "await StrategySDK.get_tracked_traders(limit=200, include_recent_activity=False, activity_hours=24)"
                ),
                "StrategySDK.get_trader_groups": (
                    "await StrategySDK.get_trader_groups(include_members=False, member_limit=25)"
                ),
                "StrategySDK.get_trader_tags": "await StrategySDK.get_trader_tags()",
                "StrategySDK.get_traders_by_tag": "await StrategySDK.get_traders_by_tag(tag_name, limit=100)",
            },
            "traders_sdk_methods": {
                "TradersSDK.get_firehose_signals": (
                    "await TradersSDK.get_firehose_signals(limit=250, "
                    "include_filtered=False, include_source_context=True)"
                ),
                "TradersSDK.get_strategy_filtered_signals": (
                    "await TradersSDK.get_strategy_filtered_signals(limit=50, include_filtered=False)"
                ),
                "TradersSDK.get_confluence_signals": (
                    "await TradersSDK.get_confluence_signals(min_strength=0.0, min_tier='WATCH', limit=50)"
                ),
                "TradersSDK.get_pooled_traders": (
                    "await TradersSDK.get_pooled_traders(limit=200, tier=None, "
                    "include_blacklisted=True, tracked_only=False)"
                ),
                "TradersSDK.get_tracked_traders": (
                    "await TradersSDK.get_tracked_traders(limit=200, include_recent_activity=False, activity_hours=24)"
                ),
                "TradersSDK.get_groups": "await TradersSDK.get_groups(include_members=False, member_limit=25)",
                "TradersSDK.get_tags": "await TradersSDK.get_tags()",
                "TradersSDK.get_traders_by_tag": "await TradersSDK.get_traders_by_tag(tag_name, limit=100)",
            },
            "examples": {
                "signal_screening": (
                    "signals = await StrategySDK.get_trader_strategy_signals(limit=200)\n"
                    "for signal in signals:\n"
                    "    if float(signal.get('firehose_confidence') or 0.0) < 0.6:\n"
                    "        continue\n"
                    "    # build opportunities from high-confidence rows"
                ),
                "tag_driven_universe": (
                    "whales = await StrategySDK.get_traders_by_tag('whale', limit=200)\n"
                    "addresses = {str(w.get('address') or '').lower() for w in whales}\n"
                    "firehose = await StrategySDK.get_trader_firehose_signals(limit=500)\n"
                    "relevant = [\n"
                    "    row for row in firehose\n"
                    "    if any(str(w).lower() in addresses for w in (row.get('wallets') or []))\n"
                    "]"
                ),
            },
            "guidance": [
                "Prefer StrategySDK wrappers in strategy source for runtime-safe failure handling.",
                "Use strategy-filtered rows when you want parity with traders_confluence execution gates.",
                "Use raw firehose rows when building custom gating logic and include source context.",
                "Treat source_flags.qualified as the canonical pooled/tracked/group provenance gate.",
            ],
        },
        # ── Section 8: Complete Examples ──────────────────────────────
        "examples": {
            "minimal_detect_only": {
                "description": "Simplest possible strategy — detect only, uses default evaluate/exit",
                "source_code": (
                    '"""\n'
                    "Strategy: Simple Spread Finder\n"
                    '"""\n'
                    "from models import Market, Event, Opportunity\n"
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
                    "from models import Market, Event, Opportunity\n"
                    "from services.strategies.base import BaseStrategy, StrategyDecision, ExitDecision, DecisionCheck\n\n"
                    "class MomentumEdge(BaseStrategy):\n"
                    '    name = "Momentum Edge"\n'
                    '    description = "Trades directional momentum with reversal-based exits"\n\n'
                    "    default_config = {\n"
                    '        "momentum_threshold": 0.05,\n'
                    '        "min_edge_percent": 2.0,\n'
                    '        "min_confidence": 0.5,\n'
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
                    "        from services.strategies.base import _trader_size_limits\n"
                    "        min_edge = float(params.get('min_edge_percent', 2.0))\n"
                    "        min_conf = float(params.get('min_confidence', 0.5))\n"
                    "        base_size, max_size = _trader_size_limits(context)\n\n"
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
                    "from models import Market, Event, Opportunity\n"
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
                "use_ohlc_replay": "bool — Detect mode only; replay OHLC snapshots when live detect finds nothing",
                "replay_lookback_hours": "int — Detect mode replay window in hours",
                "replay_timeframe": "str — Detect mode replay cadence (for example 15m, 30m, 1h)",
                "replay_max_markets": "int — Detect mode cap on replayed markets",
                "replay_max_steps": "int — Detect mode cap on replayed snapshots",
                "max_opportunities": "int — Detect mode cap on returned opportunities",
                "max_signals": "int — Evaluate mode cap on recent signals scored",
                "max_positions": "int — Exit mode cap on open positions inspected",
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
                "GET /strategy-manager": "List all strategies. Filters: ?type=detect|execute|unified, ?source_key=scanner|crypto|news|weather|traders|manual, ?enabled=true",
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
            "2. Import from models using Opportunity (not ArbitrageOpportunity)",
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
    plugin_result = validate_strategy_source(req.source_code, class_name=req.class_name)
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

    await refresh_strategy_runtime_if_needed(source_keys=None, force=False)

    async with AsyncSessionLocal() as session:

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


@router.get("/experiments")
async def get_strategy_experiments(
    source_key: Optional[str] = Query(default=None),
    strategy_key: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
):
    async with AsyncSessionLocal() as session:
        rows = await list_strategy_experiments(
            session,
            source_key=source_key,
            strategy_key=strategy_key,
            status=status,
            limit=limit,
        )
    items = [serialize_strategy_experiment(row) for row in rows]
    return {"items": items, "total": len(items)}


@router.post("/experiments")
async def create_strategy_experiment_endpoint(request: StrategyExperimentCreateRequest):
    async with AsyncSessionLocal() as session:
        try:
            row = await create_strategy_experiment(
                session,
                name=request.name,
                source_key=request.source_key,
                strategy_key=request.strategy_key,
                control_version=request.control_version,
                candidate_version=request.candidate_version,
                candidate_allocation_pct=request.candidate_allocation_pct,
                scope=request.scope,
                notes=request.notes,
                created_by=request.created_by,
                commit=True,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
    return serialize_strategy_experiment(row)


@router.post("/experiments/{experiment_id}/status")
async def set_strategy_experiment_status_endpoint(
    experiment_id: str,
    request: StrategyExperimentStatusRequest,
):
    async with AsyncSessionLocal() as session:
        try:
            row = await set_strategy_experiment_status(
                session,
                experiment_id=experiment_id,
                status=request.status,
                commit=True,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
    if row is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return serialize_strategy_experiment(row)


@router.post("/experiments/{experiment_id}/promote")
async def promote_strategy_experiment_endpoint(
    experiment_id: str,
    request: StrategyExperimentPromoteRequest,
):
    async with AsyncSessionLocal() as session:
        try:
            row = await promote_strategy_experiment(
                session,
                experiment_id=experiment_id,
                promoted_version=request.promoted_version,
                notes=request.notes,
                commit=True,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
    if row is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return serialize_strategy_experiment(row)


@router.get("/experiments/{experiment_id}/assignments")
async def get_strategy_experiment_assignments_endpoint(
    experiment_id: str,
    limit: int = Query(default=200, ge=1, le=2000),
):
    async with AsyncSessionLocal() as session:
        rows = await list_strategy_experiment_assignments(
            session,
            experiment_id=experiment_id,
            limit=limit,
        )
    return {"items": rows, "total": len(rows)}


@router.get("/{strategy_id}")
async def get_strategy(strategy_id: str, session: AsyncSession = Depends(get_db_session)):
    """Get a single strategy by ID."""
    await ensure_system_opportunity_strategies_seeded(session)

    row = await session.get(Strategy, strategy_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Strategy not found")

    return _strategy_to_dict(row)


@router.get("/{strategy_id}/versions")
async def get_strategy_versions_endpoint(
    strategy_id: str,
    limit: int = Query(default=200, ge=1, le=1000),
    include_source: bool = Query(default=False),
):
    async with AsyncSessionLocal() as session:
        row = await session.get(Strategy, strategy_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Strategy not found")
        await ensure_strategy_version_seeded(
            session,
            strategy=row,
            reason="seed_on_versions_list",
            created_by="strategy_manager",
            commit=True,
        )
        version_rows = await list_strategy_versions(
            session,
            strategy_id=strategy_id,
            limit=limit,
        )
    return {
        "strategy_id": strategy_id,
        "current_version": int(row.version or 1),
        "items": [
            serialize_strategy_version(version_row, include_source=include_source) for version_row in version_rows
        ],
        "total": len(version_rows),
    }


@router.get("/{strategy_id}/versions/{version}")
async def get_strategy_version_endpoint(
    strategy_id: str,
    version: str,
):
    requested_version = normalize_strategy_version(version)
    async with AsyncSessionLocal() as session:
        row = await session.get(Strategy, strategy_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Strategy not found")
        await ensure_strategy_version_seeded(
            session,
            strategy=row,
            reason="seed_on_version_read",
            created_by="strategy_manager",
            commit=True,
        )
        available = await list_strategy_versions(session, strategy_id=strategy_id, limit=2000)
        if requested_version is None:
            requested_version = int(row.version or 1)
        match = next((item for item in available if int(item.version or 0) == int(requested_version or 0)), None)
        if match is None:
            raise HTTPException(status_code=404, detail=f"Version v{int(requested_version or 0)} not found")
        return serialize_strategy_version(match, include_source=True)


@router.post("/{strategy_id}/versions/{version}/restore")
async def restore_strategy_version_endpoint(
    strategy_id: str,
    version: str,
    request: StrategyVersionRestoreRequest | None = None,
):
    request_payload = request or StrategyVersionRestoreRequest()
    requested_version = normalize_strategy_version(version)
    if requested_version is None:
        raise HTTPException(status_code=422, detail="Specify an explicit version number (e.g. '3' or 'v3').")

    async with AsyncSessionLocal() as session:
        row = await session.get(Strategy, strategy_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Strategy not found")

        await ensure_strategy_version_seeded(
            session,
            strategy=row,
            reason="seed_on_restore",
            created_by="strategy_manager",
            commit=False,
        )
        available = await list_strategy_versions(session, strategy_id=strategy_id, limit=2000)
        snapshot = next((item for item in available if int(item.version or 0) == int(requested_version)), None)
        if snapshot is None:
            raise HTTPException(status_code=404, detail=f"Version v{int(requested_version)} not found")

        restored = await restore_strategy_from_snapshot(
            session,
            strategy=row,
            snapshot=snapshot,
            reason=str(request_payload.reason or "manual_restore").strip() or "manual_restore",
            created_by=(str(request_payload.created_by or "").strip() or None),
            commit=False,
        )

        if row.enabled:
            try:
                strategy_loader.load(row.slug, row.source_code, row.config or None)
                row.status = "loaded"
                row.error_message = None
            except StrategyValidationError as exc:
                row.status = "error"
                row.error_message = str(exc)
        else:
            strategy_loader.unload(row.slug)
            row.status = "unloaded"
            row.error_message = None

        await bump_strategy_runtime_revisions(
            session,
            source_keys=[str(row.source_key or "").strip().lower()],
            commit=False,
        )
        await session.commit()
        await session.refresh(row)
        return {
            "status": "restored",
            "strategy": _strategy_to_dict(row),
            "restored_snapshot": serialize_strategy_version(restored, include_source=False),
            "source_snapshot": serialize_strategy_version(snapshot, include_source=False),
        }


@router.post("")
async def create_strategy(req: UnifiedStrategyCreateRequest):
    """Create a new strategy."""
    slug = _validate_slug(req.slug)
    source_key = str(req.source_key or "scanner").strip().lower()
    normalized_config = _normalize_strategy_config_for_source(source_key, req.config, strategy_slug=slug)
    normalized_schema = _merge_config_schemas(
        req.config_schema or _default_config_schema_for_source(source_key),
        StrategySDK.strategy_retention_config_schema(),
    )

    # Validate source code
    validation = validate_strategy_source(req.source_code)
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
                strategy_loader.load(slug, req.source_code, normalized_config or None)
                status = "loaded"
            except StrategyValidationError as e:
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
            config=normalized_config,
            config_schema=normalized_schema,
            aliases=[],
            version=1,
            sort_order=0,
        )
        session.add(row)
        await session.flush()
        await create_strategy_version_snapshot(
            session,
            strategy=row,
            reason="strategy_created",
            created_by="strategy_manager",
            forced_version=1,
            parent_version=None,
            commit=False,
        )
        await bump_strategy_runtime_revisions(
            session,
            source_keys=[source_key],
            commit=False,
        )
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
        original_source_key = str(row.source_key or "").strip().lower()

        # is_system is an informational flag (UI label / sort hint) only —
        # users can edit any strategy regardless. The historical 403 gate
        # was removed; ``unlock_system`` on the request is ignored.

        original_slug = row.slug
        slug_changed = False
        snapshot_fields_changed: set[str] = set()
        reload_reasons: set[str] = set()
        prior_version = int(row.version or 1)
        next_source_key = str(req.source_key or row.source_key or "scanner").strip().lower()

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
                snapshot_fields_changed.add("slug")
                reload_reasons.add("slug")

        if req.source_code is not None and req.source_code != row.source_code:
            validation = validate_strategy_source(req.source_code)
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
            snapshot_fields_changed.add("source_code")
            reload_reasons.add("source_code")

        if req.config is not None:
            normalized_config = _normalize_strategy_config_for_source(
                next_source_key,
                req.config,
                strategy_slug=row.slug,
            )
            if normalized_config != (row.config or {}):
                row.config = normalized_config
                snapshot_fields_changed.add("config")
                reload_reasons.add("config")
        if req.config_schema is not None:
            merged_schema = _merge_config_schemas(
                req.config_schema,
                StrategySDK.strategy_retention_config_schema(),
            )
            if merged_schema != (row.config_schema or {}):
                row.config_schema = merged_schema
                snapshot_fields_changed.add("config_schema")

        if req.source_key is not None:
            if next_source_key != str(row.source_key or "").strip().lower():
                row.source_key = next_source_key
                snapshot_fields_changed.add("source_key")
            if req.config is None:
                normalized_existing_config = _normalize_strategy_config_for_source(
                    next_source_key,
                    row.config,
                    strategy_slug=row.slug,
                )
                if normalized_existing_config != (row.config or {}):
                    row.config = normalized_existing_config
                    snapshot_fields_changed.add("config")
                    reload_reasons.add("config")
            if not row.config_schema:
                row.config_schema = _default_config_schema_for_source(next_source_key)
                snapshot_fields_changed.add("config_schema")
        if req.name is not None:
            next_name = str(req.name).strip()
            if next_name != str(row.name or "").strip():
                row.name = next_name
                snapshot_fields_changed.add("name")
        if req.description is not None:
            next_description = req.description
            if next_description != row.description:
                row.description = next_description
                snapshot_fields_changed.add("description")

        if req.enabled is not None and req.enabled != row.enabled:
            row.enabled = req.enabled
            snapshot_fields_changed.add("enabled")
            reload_reasons.add("enabled")

        if reload_reasons == {"config"} and row.enabled:
            try:
                reconfigured = strategy_loader.reconfigure_loaded(
                    row.slug,
                    row.source_code,
                    row.config or None,
                )
                if not reconfigured:
                    strategy_loader.load(row.slug, row.source_code, row.config or None)
                row.status = "loaded"
                row.error_message = None
            except StrategyValidationError as e:
                row.status = "error"
                row.error_message = str(e)
        elif reload_reasons:
            if slug_changed:
                strategy_loader.unload(original_slug)
            if row.enabled:
                try:
                    strategy_loader.load(row.slug, row.source_code, row.config or None)
                    row.status = "loaded"
                    row.error_message = None
                except StrategyValidationError as e:
                    row.status = "error"
                    row.error_message = str(e)
            else:
                strategy_loader.unload(row.slug)
                row.status = "unloaded"
                row.error_message = None

        if snapshot_fields_changed:
            await create_strategy_version_snapshot(
                session,
                strategy=row,
                reason=f"strategy_updated:{','.join(sorted(snapshot_fields_changed))}",
                created_by="strategy_manager",
                parent_version=prior_version,
                commit=False,
            )

        await bump_strategy_runtime_revisions(
            session,
            source_keys=[
                original_source_key,
                str(row.source_key or "").strip().lower(),
            ],
            commit=False,
        )
        await session.commit()
        await session.refresh(row)
        return _strategy_to_dict(row)


@router.delete("/{strategy_id}")
async def delete_strategy(strategy_id: str):
    """Delete a strategy.

    Any strategy can be deleted regardless of ``is_system``. No tombstone
    is written — once deleted, the row is gone. To restore a shipped
    template, hit ``POST /{id}/reset-to-factory`` if the slug still has
    a registered seed, or recreate manually.
    """
    async with AsyncSessionLocal() as session:
        row = await session.get(Strategy, strategy_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Strategy not found")

        strategy_loader.unload(row.slug)
        await session.delete(row)
        await bump_strategy_runtime_revisions(
            session,
            source_keys=[str(row.source_key or "").strip().lower()],
            commit=False,
        )
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
            strategy_loader.load(row.slug, row.source_code, row.config or None)
            row.status = "loaded"
            row.error_message = None
            await bump_strategy_runtime_revisions(
                session,
                source_keys=[str(row.source_key or "").strip().lower()],
                commit=False,
            )
            await session.commit()
            return {
                "status": "success",
                "message": f"Strategy '{row.slug}' reloaded",
                "runtime": strategy_loader.get_status(row.slug),
            }
        except StrategyValidationError as e:
            row.status = "error"
            row.error_message = str(e)
            await session.commit()
            raise HTTPException(
                status_code=400,
                detail={"message": f"Reload failed for '{row.slug}'", "error": str(e)},
            )


@router.post("/{strategy_id}/reset-to-factory")
async def reset_strategy_to_factory_endpoint(strategy_id: str):
    """Reset a strategy to its original shipped seed definition.

    Restores ``source_code``, ``config``, ``config_schema``, ``description``,
    etc. to the values shipped with the application. Works for any
    strategy whose slug has a registered seed in
    ``SYSTEM_OPPORTUNITY_STRATEGY_SEEDS`` — ``is_system`` is no longer
    a requirement. Returns 400 when the slug has no seed registered
    (i.e. user-authored strategies have nothing to reset to).
    """
    from services.opportunity_strategy_catalog import reset_strategy_to_factory

    async with AsyncSessionLocal() as session:
        row = await session.get(Strategy, strategy_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Strategy not found")

        result = await reset_strategy_to_factory(session, row.slug)
        if result.get("status") == "not_found":
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Strategy '{row.slug}' has no shipped seed to reset to. "
                    "Reset-to-factory only applies to slugs registered in "
                    "the strategy catalog."
                ),
            )

        # Reload into the unified loader after reset
        if result.get("status") in ("reset", "created"):
            await session.refresh(row)
            try:
                strategy_loader.load(row.slug, row.source_code, row.config or None)
            except Exception:
                pass
            await bump_strategy_runtime_revisions(
                session,
                source_keys=[str(row.source_key or "").strip().lower()],
                commit=True,
            )

        return result
