"""API routes for DB-native trader strategy definitions.

DEPRECATED — backward compatibility alias.

These /trader-strategies endpoints are preserved for backward compatibility.
New clients should use the unified /strategies API (routes_strategies.py)
which serves both opportunity-detection and trader-execution strategies
through a single interface.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import Strategy, get_db_session
from services.trader_orchestrator.strategy_catalog import (
    ensure_system_trader_strategies_seeded,
)
from services.trader_orchestrator.strategy_db_loader import (
    serialize_trader_strategy_definition,
    strategy_db_loader,
    validate_trader_strategy_source,
)

router = APIRouter(prefix="/trader-strategies", tags=["Trader Strategies"])

TRADER_STRATEGY_TEMPLATE = """\"\"\"Custom trader strategy template.\"\"\"

from services.trader_orchestrator.strategies.base import BaseTraderStrategy, StrategyDecision, DecisionCheck


class CustomTraderStrategy(BaseTraderStrategy):
    key = "custom_trader_strategy"

    def evaluate(self, signal, context):
        checks = [
            DecisionCheck("example", "Example gate", True, detail="replace with your logic"),
        ]
        return StrategyDecision(
            decision="skipped",
            reason="Template strategy",
            score=0.0,
            checks=checks,
            payload={},
        )
"""


class TraderStrategyCreateRequest(BaseModel):
    strategy_key: str = Field(min_length=2, max_length=128)
    source_key: str = Field(min_length=2, max_length=64)
    label: str = Field(min_length=1, max_length=200)
    description: Optional[str] = None
    class_name: str = Field(min_length=1, max_length=200)
    source_code: str = Field(min_length=10)
    default_params_json: dict[str, Any] = Field(default_factory=dict)
    param_schema_json: dict[str, Any] = Field(default_factory=dict)
    aliases_json: list[str] = Field(default_factory=list)
    enabled: bool = True


class TraderStrategyUpdateRequest(BaseModel):
    strategy_key: Optional[str] = Field(default=None, min_length=2, max_length=128)
    source_key: Optional[str] = Field(default=None, min_length=2, max_length=64)
    label: Optional[str] = Field(default=None, min_length=1, max_length=200)
    description: Optional[str] = None
    class_name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    source_code: Optional[str] = Field(default=None, min_length=10)
    default_params_json: Optional[dict[str, Any]] = None
    param_schema_json: Optional[dict[str, Any]] = None
    aliases_json: Optional[list[str]] = None
    enabled: Optional[bool] = None
    unlock_system: bool = False


class TraderStrategyValidateRequest(BaseModel):
    source_code: Optional[str] = None
    class_name: Optional[str] = None


class TraderStrategyCloneRequest(BaseModel):
    strategy_key: Optional[str] = Field(default=None, min_length=2, max_length=128)
    label: Optional[str] = Field(default=None, min_length=1, max_length=200)
    enabled: bool = True


def _normalize_key(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_aliases(value: Any) -> list[str]:
    aliases = value if isinstance(value, list) else []
    out: list[str] = []
    seen: set[str] = set()
    for raw in aliases:
        item = _normalize_key(raw)
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _assert_editable(row: Strategy, unlock_system: bool) -> None:
    if bool(row.is_system) and not unlock_system:
        raise HTTPException(
            status_code=403,
            detail=(
                "System strategies are read-only. Clone strategy first or set unlock_system=true "
                "for explicit admin override."
            ),
        )


async def _ensure_unique_strategy_key(
    session: AsyncSession,
    strategy_key: str,
    *,
    current_id: str | None = None,
) -> None:
    query = select(Strategy.id).where(
        func.lower(Strategy.slug) == strategy_key.lower()
    )
    if current_id:
        query = query.where(Strategy.id != current_id)
    exists = (await session.execute(query)).scalar_one_or_none()
    if exists is not None:
        raise HTTPException(status_code=409, detail=f"strategy_key '{strategy_key}' already exists")


@router.get("")
async def list_trader_strategies(
    source_key: Optional[str] = Query(default=None),
    enabled: Optional[bool] = Query(default=None),
    status: Optional[str] = Query(default=None),
    session: AsyncSession = Depends(get_db_session),
):
    await ensure_system_trader_strategies_seeded(session)
    query = select(Strategy).order_by(
        Strategy.source_key.asc(),
        Strategy.slug.asc(),
    )
    if source_key:
        query = query.where(Strategy.source_key == _normalize_key(source_key))
    if enabled is not None:
        query = query.where(Strategy.enabled == bool(enabled))
    if status:
        query = query.where(Strategy.status == _normalize_key(status))

    rows = list((await session.execute(query)).scalars().all())
    return {"items": [serialize_trader_strategy_definition(row) for row in rows]}


@router.get("/template")
async def get_trader_strategy_template():
    return {
        "template": TRADER_STRATEGY_TEMPLATE,
        "instructions": (
            "Create a class that extends BaseTraderStrategy and implements evaluate(signal, context). "
            "Return StrategyDecision with decision/reason/score/checks/payload."
        ),
        "available_imports": [
            "services.trader_orchestrator.strategies.base (BaseTraderStrategy, StrategyDecision, DecisionCheck)",
            "typing, dataclasses, math, statistics, datetime, collections",
        ],
    }


@router.get("/docs")
async def get_trader_strategy_docs():
    return {
        "overview": {
            "title": "Trader Strategy API Reference",
            "description": (
                "Trader strategies are DB-hosted executable classes evaluated by the orchestrator "
                "for each trade signal. They decide whether a detected signal should be traded, "
                "at what size, and with what confidence."
            ),
        },
        "class_structure": {
            "required_base_class": "BaseTraderStrategy",
            "required_method": "evaluate(self, signal, context) -> StrategyDecision",
            "required_attributes": {
                "key": "str — Unique strategy key (e.g. 'my_custom_strategy')",
            },
            "imports": (
                "from services.trader_orchestrator.strategies.base import "
                "BaseTraderStrategy, StrategyDecision, DecisionCheck"
            ),
        },
        "evaluate_method": {
            "signature": "def evaluate(self, signal, context) -> StrategyDecision",
            "description": "Called for each pending trade signal. Return a decision.",
            "parameters": {
                "signal": {
                    "type": "TradeSignal ORM row",
                    "fields": {
                        "source": "str — Signal source (e.g. 'scanner', 'crypto', 'news', 'weather', 'traders')",
                        "signal_type": "str — Type of signal (e.g. 'scanner_opportunity', 'crypto_worker_multistrat')",
                        "direction": "str — 'BUY' or 'SELL'",
                        "edge_percent": "float — Estimated edge as a percentage",
                        "confidence": "float — Confidence score (0-1)",
                        "entry_price": "float — Suggested entry price",
                        "liquidity": "float — Market liquidity in USD",
                        "payload_json": "dict — Source-specific payload with extra data",
                        "created_at": "datetime — When the signal was emitted",
                        "market_id": "str — The market this signal targets",
                    },
                },
                "context": {
                    "type": "dict",
                    "fields": {
                        "params": "dict — Strategy parameters (defaults merged with user overrides from UI)",
                        "trader": "ORM row — The trader this strategy is evaluating for",
                        "mode": "str — 'paper' or 'live'",
                        "live_market": "dict | None — Live market context when in live mode",
                        "source_config": "dict — The source configuration for this signal type",
                    },
                },
            },
            "returns": {
                "type": "StrategyDecision",
                "fields": {
                    "decision": "str — 'selected' | 'skipped' | 'blocked' | 'failed'",
                    "reason": "str — Human-readable explanation",
                    "score": "float — Numeric ranking score (higher = stronger signal)",
                    "size_usd": "float | None — Trade size in USD (when selected)",
                    "checks": "list[DecisionCheck] — Gate results explaining the decision",
                    "payload": "dict — Arbitrary diagnostic data",
                },
            },
        },
        "decision_checks": {
            "description": (
                "Attach DecisionCheck rows to explain each gate/filter in the decision. "
                "These are shown in the UI as a checklist for transparency."
            ),
            "signature": "DecisionCheck(key, label, passed, score=None, detail=None, payload=None)",
            "example": (
                "checks = [\n"
                "    DecisionCheck('edge_gate', 'Minimum edge', signal.edge_percent >= min_edge,\n"
                "                  score=signal.edge_percent, detail=f'{signal.edge_percent:.1f}% vs {min_edge}% min'),\n"
                "    DecisionCheck('confidence_gate', 'Confidence threshold', signal.confidence >= 0.6,\n"
                "                  score=signal.confidence, detail=f'{signal.confidence:.0%} confidence'),\n"
                "    DecisionCheck('liquidity_gate', 'Minimum liquidity', signal.liquidity >= 1000,\n"
                "                  detail=f'${signal.liquidity:,.0f} available'),\n"
                "]"
            ),
        },
        "sizing": {
            "description": (
                "Set size_usd on your StrategyDecision to control trade sizing. "
                "The risk manager may further cap this based on portfolio limits."
            ),
            "example": (
                "base_size = context['params'].get('base_size_usd', 25.0)\n"
                "size = base_size * signal.confidence * min(signal.edge_percent / 5.0, 2.0)\n"
                "return StrategyDecision(\n"
                "    decision='selected',\n"
                "    reason=f'Edge {signal.edge_percent:.1f}% with {signal.confidence:.0%} confidence',\n"
                "    score=signal.edge_percent * signal.confidence,\n"
                "    size_usd=round(size, 2),\n"
                "    checks=checks,\n"
                ")"
            ),
        },
        "allowed_imports": [
            {"module": "services.trader_orchestrator.strategies.base", "items": "BaseTraderStrategy, StrategyDecision, DecisionCheck"},
            {"module": "services.trader_orchestrator.strategies", "items": "Built-in trader strategy modules"},
            {"module": "math", "items": "Standard math functions"},
            {"module": "statistics", "items": "Statistical functions"},
            {"module": "collections", "items": "defaultdict, Counter, etc."},
            {"module": "datetime", "items": "datetime, timedelta"},
            {"module": "json", "items": "JSON parsing"},
            {"module": "re", "items": "Regular expressions"},
            {"module": "typing", "items": "Type hints"},
            {"module": "dataclasses", "items": "Dataclass decorators"},
            {"module": "random", "items": "Random number generation"},
        ],
        "blocked_imports": [
            "os, sys, subprocess — No filesystem or process access",
            "httpx, requests, aiohttp — No direct HTTP (use services layer)",
            "numpy, scipy — Not available in trader strategies",
            "asyncio, threading — No async/concurrent operations",
            "pickle, marshal — No serialization",
        ],
        "safety": {
            "description": "Source code is validated with AST restrictions before compile/load.",
            "notes": [
                "Unsafe imports are blocked at load time via AST inspection",
                "Blocked function calls: exec(), eval(), compile(), __import__(), open(), input()",
                "Invalid strategies are marked with error status and skipped",
            ],
        },
        "cookbook": {
            "description": "Common patterns for trader strategy development",
            "recipes": {
                "basic_edge_confidence_strategy": (
                    "def evaluate(self, signal, context):\n"
                    "    params = context['params']\n"
                    "    min_edge = params.get('min_edge', 3.0)\n"
                    "    min_confidence = params.get('min_confidence', 0.6)\n"
                    "    checks = [\n"
                    "        DecisionCheck('edge', 'Min edge', signal.edge_percent >= min_edge,\n"
                    "                      detail=f'{signal.edge_percent:.1f}%'),\n"
                    "        DecisionCheck('confidence', 'Min confidence', signal.confidence >= min_confidence,\n"
                    "                      detail=f'{signal.confidence:.0%}'),\n"
                    "    ]\n"
                    "    if all(c.passed for c in checks):\n"
                    "        return StrategyDecision('selected', 'Passed all gates',\n"
                    "                                signal.edge_percent * signal.confidence,\n"
                    "                                checks=checks, size_usd=25.0)\n"
                    "    return StrategyDecision('skipped', 'Failed gates', 0, checks=checks)"
                ),
                "source_specific_logic": (
                    "def evaluate(self, signal, context):\n"
                    "    source = signal.source\n"
                    "    payload = signal.payload_json or {}\n"
                    "    if source == 'scanner':\n"
                    "        roi = payload.get('roi_percent', 0)\n"
                    "        guaranteed = payload.get('is_guaranteed', False)\n"
                    "        # Structural arb: lower threshold\n"
                    "        min_edge = 1.5 if guaranteed else 5.0\n"
                    "    elif source == 'crypto':\n"
                    "        mode = payload.get('mode', 'directional')\n"
                    "        min_edge = 2.0 if mode == 'pure_arb' else 4.0\n"
                    "    else:\n"
                    "        min_edge = 5.0"
                ),
                "mode_aware_sizing": (
                    "mode = context.get('mode', 'paper')\n"
                    "base_size = context['params'].get('base_size_usd', 25.0)\n"
                    "if mode == 'live':\n"
                    "    # More conservative in live mode\n"
                    "    base_size *= 0.5\n"
                    "size = base_size * min(signal.confidence, 1.0)"
                ),
            },
        },
    }


@router.get("/{strategy_id}")
async def get_trader_strategy(strategy_id: str, session: AsyncSession = Depends(get_db_session)):
    await ensure_system_trader_strategies_seeded(session)
    row = await session.get(Strategy, strategy_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Strategy definition not found")
    return serialize_trader_strategy_definition(row)


@router.post("")
async def create_trader_strategy(
    request: TraderStrategyCreateRequest,
    session: AsyncSession = Depends(get_db_session),
):
    strategy_key = _normalize_key(request.strategy_key)
    source_key = _normalize_key(request.source_key)

    await _ensure_unique_strategy_key(session, strategy_key)

    validation = validate_trader_strategy_source(request.source_code, request.class_name)
    if not validation.get("valid"):
        raise HTTPException(status_code=422, detail={"errors": validation.get("errors", [])})

    row = Strategy(
        id=uuid.uuid4().hex,
        slug=strategy_key,
        source_key=source_key,
        name=str(request.label).strip(),
        description=request.description,
        class_name=str(validation.get("class_name") or request.class_name).strip(),
        source_code=request.source_code,
        config=dict(request.default_params_json or {}),
        config_schema=dict(request.param_schema_json or {}),
        aliases=_normalize_aliases(request.aliases_json),
        is_system=False,
        enabled=bool(request.enabled),
        status="unloaded",
        error_message=None,
        version=1,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)

    if row.enabled:
        await strategy_db_loader.reload_strategy(row.slug, session=session)
        row = await session.get(Strategy, row.id)

    return serialize_trader_strategy_definition(row)


@router.put("/{strategy_id}")
async def update_trader_strategy(
    strategy_id: str,
    request: TraderStrategyUpdateRequest,
    session: AsyncSession = Depends(get_db_session),
):
    row = await session.get(Strategy, strategy_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Strategy definition not found")

    _assert_editable(row, bool(request.unlock_system))

    updates = request.model_dump(exclude_unset=True)
    updates.pop("unlock_system", None)

    next_strategy_key = _normalize_key(updates.get("strategy_key", row.slug))
    if next_strategy_key != _normalize_key(row.slug):
        await _ensure_unique_strategy_key(session, next_strategy_key, current_id=row.id)

    next_class_name = str(updates.get("class_name", row.class_name) or "").strip()
    next_source_code = str(updates.get("source_code", row.source_code) or "")
    if "class_name" in updates or "source_code" in updates:
        validation = validate_trader_strategy_source(next_source_code, next_class_name)
        if not validation.get("valid"):
            raise HTTPException(status_code=422, detail={"errors": validation.get("errors", [])})
        next_class_name = str(validation.get("class_name") or next_class_name).strip()

    row.slug = next_strategy_key
    if "source_key" in updates:
        row.source_key = _normalize_key(updates.get("source_key"))
    if "label" in updates:
        row.name = str(updates.get("label") or "").strip() or row.name
    if "description" in updates:
        row.description = updates.get("description")
    row.class_name = next_class_name or row.class_name
    row.source_code = next_source_code or row.source_code
    if "default_params_json" in updates:
        row.config = dict(updates.get("default_params_json") or {})
    if "param_schema_json" in updates:
        row.config_schema = dict(updates.get("param_schema_json") or {})
    if "aliases_json" in updates:
        row.aliases = _normalize_aliases(updates.get("aliases_json"))
    if "enabled" in updates:
        row.enabled = bool(updates.get("enabled"))
    row.version = int(row.version or 1) + 1
    row.status = "unloaded"
    row.error_message = None

    await session.commit()
    await session.refresh(row)

    await strategy_db_loader.reload_strategy(row.slug, session=session)
    row = await session.get(Strategy, row.id)
    return serialize_trader_strategy_definition(row)


@router.post("/{strategy_id}/validate")
async def validate_trader_strategy(
    strategy_id: str,
    request: TraderStrategyValidateRequest,
    session: AsyncSession = Depends(get_db_session),
):
    row = await session.get(Strategy, strategy_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Strategy definition not found")

    source_code = request.source_code or row.source_code
    class_name = request.class_name or row.class_name
    validation = validate_trader_strategy_source(source_code, class_name)
    return validation


@router.post("/{strategy_id}/reload")
async def reload_trader_strategy(
    strategy_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    row = await session.get(Strategy, strategy_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Strategy definition not found")

    result = await strategy_db_loader.reload_strategy(row.slug, session=session)
    refreshed = await session.get(Strategy, strategy_id)
    return {
        "status": "ok",
        "reload": result,
        "strategy": serialize_trader_strategy_definition(refreshed),
    }


@router.post("/{strategy_id}/clone")
async def clone_trader_strategy(
    strategy_id: str,
    request: TraderStrategyCloneRequest,
    session: AsyncSession = Depends(get_db_session),
):
    row = await session.get(Strategy, strategy_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Strategy definition not found")

    suffix = uuid.uuid4().hex[:8]
    strategy_key = _normalize_key(request.strategy_key or f"{row.slug}_clone_{suffix}")
    await _ensure_unique_strategy_key(session, strategy_key)

    clone = Strategy(
        id=uuid.uuid4().hex,
        slug=strategy_key,
        source_key=str(row.source_key or "").strip().lower(),
        name=str(request.label or f"{row.name} (Clone)").strip(),
        description=row.description,
        class_name=row.class_name,
        source_code=row.source_code,
        config=dict(row.config or {}),
        config_schema=dict(row.config_schema or {}),
        aliases=[],
        is_system=False,
        enabled=bool(request.enabled),
        status="unloaded",
        error_message=None,
        version=1,
    )
    session.add(clone)
    await session.commit()
    await session.refresh(clone)

    if clone.enabled:
        await strategy_db_loader.reload_strategy(clone.slug, session=session)
        clone = await session.get(Strategy, clone.id)

    return serialize_trader_strategy_definition(clone)
