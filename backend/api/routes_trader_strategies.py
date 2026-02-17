"""API routes for DB-native trader strategy definitions."""

from __future__ import annotations

import uuid
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import TraderStrategyDefinition, get_db_session
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


def _assert_editable(row: TraderStrategyDefinition, unlock_system: bool) -> None:
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
    query = select(TraderStrategyDefinition.id).where(
        func.lower(TraderStrategyDefinition.strategy_key) == strategy_key.lower()
    )
    if current_id:
        query = query.where(TraderStrategyDefinition.id != current_id)
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
    query = select(TraderStrategyDefinition).order_by(
        TraderStrategyDefinition.source_key.asc(),
        TraderStrategyDefinition.strategy_key.asc(),
    )
    if source_key:
        query = query.where(TraderStrategyDefinition.source_key == _normalize_key(source_key))
    if enabled is not None:
        query = query.where(TraderStrategyDefinition.enabled == bool(enabled))
    if status:
        query = query.where(TraderStrategyDefinition.status == _normalize_key(status))

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
                "for each trade signal."
            ),
        },
        "class_structure": {
            "required_base_class": "BaseTraderStrategy",
            "required_method": "evaluate(self, signal, context) -> StrategyDecision",
            "required_attributes": {
                "key": "Unique strategy key string",
            },
        },
        "evaluate_method": {
            "signature": "def evaluate(self, signal, context) -> StrategyDecision",
            "parameters": {
                "signal": "TradeSignal ORM row with source/market/edge/confidence payload",
                "context": "Dict containing trader, source config, strategy params, and runtime hints",
            },
            "returns": "StrategyDecision(decision='approved|blocked|skipped', reason, score, checks, payload)",
        },
        "decision_checks": {
            "description": "Attach DecisionCheck rows to explain each gate in the final decision.",
            "signature": "DecisionCheck(key, label, passed, detail=None, value=None)",
        },
        "safety": {
            "description": "Source code is validated with AST restrictions before compile/load.",
            "notes": [
                "Unsafe imports are blocked.",
                "Invalid strategies are marked error and blocked per strategy key.",
            ],
        },
    }


@router.get("/{strategy_id}")
async def get_trader_strategy(strategy_id: str, session: AsyncSession = Depends(get_db_session)):
    await ensure_system_trader_strategies_seeded(session)
    row = await session.get(TraderStrategyDefinition, strategy_id)
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

    row = TraderStrategyDefinition(
        id=uuid.uuid4().hex,
        strategy_key=strategy_key,
        source_key=source_key,
        label=str(request.label).strip(),
        description=request.description,
        class_name=str(validation.get("class_name") or request.class_name).strip(),
        source_code=request.source_code,
        default_params_json=dict(request.default_params_json or {}),
        param_schema_json=dict(request.param_schema_json or {}),
        aliases_json=_normalize_aliases(request.aliases_json),
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
        await strategy_db_loader.reload_strategy(row.strategy_key, session=session)
        row = await session.get(TraderStrategyDefinition, row.id)

    return serialize_trader_strategy_definition(row)


@router.put("/{strategy_id}")
async def update_trader_strategy(
    strategy_id: str,
    request: TraderStrategyUpdateRequest,
    session: AsyncSession = Depends(get_db_session),
):
    row = await session.get(TraderStrategyDefinition, strategy_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Strategy definition not found")

    _assert_editable(row, bool(request.unlock_system))

    updates = request.model_dump(exclude_unset=True)
    updates.pop("unlock_system", None)

    next_strategy_key = _normalize_key(updates.get("strategy_key", row.strategy_key))
    if next_strategy_key != _normalize_key(row.strategy_key):
        await _ensure_unique_strategy_key(session, next_strategy_key, current_id=row.id)

    next_class_name = str(updates.get("class_name", row.class_name) or "").strip()
    next_source_code = str(updates.get("source_code", row.source_code) or "")
    if "class_name" in updates or "source_code" in updates:
        validation = validate_trader_strategy_source(next_source_code, next_class_name)
        if not validation.get("valid"):
            raise HTTPException(status_code=422, detail={"errors": validation.get("errors", [])})
        next_class_name = str(validation.get("class_name") or next_class_name).strip()

    row.strategy_key = next_strategy_key
    if "source_key" in updates:
        row.source_key = _normalize_key(updates.get("source_key"))
    if "label" in updates:
        row.label = str(updates.get("label") or "").strip() or row.label
    if "description" in updates:
        row.description = updates.get("description")
    row.class_name = next_class_name or row.class_name
    row.source_code = next_source_code or row.source_code
    if "default_params_json" in updates:
        row.default_params_json = dict(updates.get("default_params_json") or {})
    if "param_schema_json" in updates:
        row.param_schema_json = dict(updates.get("param_schema_json") or {})
    if "aliases_json" in updates:
        row.aliases_json = _normalize_aliases(updates.get("aliases_json"))
    if "enabled" in updates:
        row.enabled = bool(updates.get("enabled"))
    row.version = int(row.version or 1) + 1
    row.status = "unloaded"
    row.error_message = None

    await session.commit()
    await session.refresh(row)

    await strategy_db_loader.reload_strategy(row.strategy_key, session=session)
    row = await session.get(TraderStrategyDefinition, row.id)
    return serialize_trader_strategy_definition(row)


@router.post("/{strategy_id}/validate")
async def validate_trader_strategy(
    strategy_id: str,
    request: TraderStrategyValidateRequest,
    session: AsyncSession = Depends(get_db_session),
):
    row = await session.get(TraderStrategyDefinition, strategy_id)
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
    row = await session.get(TraderStrategyDefinition, strategy_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Strategy definition not found")

    result = await strategy_db_loader.reload_strategy(row.strategy_key, session=session)
    refreshed = await session.get(TraderStrategyDefinition, strategy_id)
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
    row = await session.get(TraderStrategyDefinition, strategy_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Strategy definition not found")

    suffix = uuid.uuid4().hex[:8]
    strategy_key = _normalize_key(request.strategy_key or f"{row.strategy_key}_clone_{suffix}")
    await _ensure_unique_strategy_key(session, strategy_key)

    clone = TraderStrategyDefinition(
        id=uuid.uuid4().hex,
        strategy_key=strategy_key,
        source_key=str(row.source_key or "").strip().lower(),
        label=str(request.label or f"{row.label} (Clone)").strip(),
        description=row.description,
        class_name=row.class_name,
        source_code=row.source_code,
        default_params_json=dict(row.default_params_json or {}),
        param_schema_json=dict(row.param_schema_json or {}),
        aliases_json=[],
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
        await strategy_db_loader.reload_strategy(clone.strategy_key, session=session)
        clone = await session.get(TraderStrategyDefinition, clone.id)

    return serialize_trader_strategy_definition(clone)
