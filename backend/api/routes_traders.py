"""API routes for trader CRUD and trader-level runtime surfaces."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from models.database import MarketCatalog, ScannerSnapshot, get_db_session
from services.live_price_snapshot import normalize_binary_price_history
from services.pause_state import global_pause_state
from services.strategy_tune_agent import run_strategy_tune_agent
from services.trader_orchestrator.position_lifecycle import (
    reconcile_live_positions,
    reconcile_paper_positions,
    reconcile_shadow_positions,
)
from services.trader_orchestrator.session_engine import ExecutionSessionEngine
from services.trader_orchestrator_state import (
    adopt_live_wallet_position,
    cleanup_trader_open_orders,
    create_config_revision,
    create_trader,
    create_trader_event,
    create_trader_from_template,
    delete_trader,
    get_trader,
    get_trader_decision_detail,
    get_open_order_summary_for_trader,
    get_open_position_summary_for_trader,
    get_serialized_execution_session_detail,
    list_live_wallet_positions_for_trader,
    list_serialized_trader_decisions,
    list_serialized_trader_events,
    list_serialized_execution_sessions,
    list_serialized_trader_orders,
    list_trader_templates,
    list_traders,
    read_orchestrator_control,
    request_trader_run,
    set_trader_paused,
    sync_trader_position_inventory,
    update_trader,
)
from utils.converters import normalize_market_id, to_iso
from utils.market_urls import infer_market_platform

router = APIRouter(prefix="/traders", tags=["Traders"])
_LOSS_STREAK_RESET_AT_KEY = "loss_streak_reset_at"
_LOSS_STREAK_RESET_REASON_KEY = "loss_streak_reset_reason"


class TraderSourceConfigRequest(BaseModel):
    source_key: str
    strategy_key: str
    strategy_version: int | str | None = None
    strategy_params: dict[str, Any] = Field(default_factory=dict)


class TraderRequest(BaseModel):
    name: str
    description: Optional[str] = None
    mode: Optional[str] = None
    copy_from_trader_id: Optional[str] = None
    source_configs: list[TraderSourceConfigRequest] = Field(default_factory=list)
    interval_seconds: int = Field(default=60, ge=1, le=86400)
    risk_limits: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_enabled: bool = True
    is_paused: bool = False
    requested_by: Optional[str] = None
    reason: Optional[str] = None


class TraderPatchRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    mode: Optional[str] = None
    source_configs: Optional[list[TraderSourceConfigRequest]] = None
    interval_seconds: Optional[int] = Field(default=None, ge=1, le=86400)
    risk_limits: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None
    is_enabled: Optional[bool] = None
    is_paused: Optional[bool] = None
    requested_by: Optional[str] = None
    reason: Optional[str] = None


class TraderTemplateCreateRequest(BaseModel):
    template_id: str
    overrides: dict[str, Any] = Field(default_factory=dict)
    requested_by: Optional[str] = None


class TraderDeleteAction(str, Enum):
    block = "block"
    disable = "disable"
    force_delete = "force_delete"


class TraderPositionCleanupScope(str, Enum):
    paper = "paper"
    shadow = "shadow"
    live = "live"
    all = "all"


class TraderPositionCleanupMethod(str, Enum):
    mark_to_market = "mark_to_market"
    cancel = "cancel"


class TraderPositionCleanupRequest(BaseModel):
    scope: TraderPositionCleanupScope = TraderPositionCleanupScope.paper
    method: TraderPositionCleanupMethod = TraderPositionCleanupMethod.mark_to_market
    max_age_hours: Optional[int] = Field(default=None, ge=1, le=24 * 365)
    dry_run: bool = False
    target_status: str = Field(default="cancelled", min_length=1, max_length=64)
    reason: Optional[str] = None
    confirm_live: bool = False


class TraderLiveWalletPositionAdoptRequest(BaseModel):
    token_id: str = Field(..., min_length=1, max_length=256)
    reason: Optional[str] = None
    requested_by: Optional[str] = None


class TraderExecutionSessionControlRequest(BaseModel):
    reason: Optional[str] = None


class TraderTuneAgentRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=12000)
    model: Optional[str] = Field(default=None, max_length=200)
    max_iterations: int = Field(default=12, ge=1, le=24)
    monitor_job_id: Optional[str] = Field(default=None, max_length=120)


def _collect_market_aliases(raw_market: Any) -> list[str]:
    if not isinstance(raw_market, dict):
        return []
    aliases: list[str] = []
    for candidate in (
        raw_market.get("id"),
        raw_market.get("market_id"),
        raw_market.get("condition_id"),
        raw_market.get("conditionId"),
        raw_market.get("slug"),
        raw_market.get("market_slug"),
        raw_market.get("marketSlug"),
        raw_market.get("event_slug"),
        raw_market.get("eventSlug"),
        raw_market.get("event_ticker"),
        raw_market.get("eventTicker"),
        raw_market.get("ticker"),
        raw_market.get("token_id"),
        raw_market.get("tokenId"),
        raw_market.get("selected_token_id"),
        raw_market.get("selectedTokenId"),
        raw_market.get("asset_id"),
        raw_market.get("assetId"),
        raw_market.get("yes_token_id"),
        raw_market.get("no_token_id"),
    ):
        normalized = normalize_market_id(candidate)
        if normalized and normalized not in aliases:
            aliases.append(normalized)

    for key in ("clob_token_ids", "clobTokenIds", "token_ids", "tokenIds"):
        values = raw_market.get(key)
        if not isinstance(values, list):
            continue
        for candidate in values:
            normalized = normalize_market_id(candidate)
            if normalized and normalized not in aliases:
                aliases.append(normalized)

    tokens = raw_market.get("tokens")
    if isinstance(tokens, list):
        for token in tokens:
            if not isinstance(token, dict):
                continue
            for candidate in (
                token.get("token_id"),
                token.get("tokenId"),
                token.get("asset_id"),
                token.get("assetId"),
                token.get("id"),
            ):
                normalized = normalize_market_id(candidate)
                if normalized and normalized not in aliases:
                    aliases.append(normalized)
    return aliases


def _merge_normalized_binary_history(
    existing: list[dict[str, float]],
    incoming: list[dict[str, float]],
    limit: int,
) -> list[dict[str, float]]:
    if not existing:
        return incoming[-limit:]
    if not incoming:
        return existing[-limit:]

    merged_by_ts: dict[int, dict[str, float]] = {}
    for point in existing:
        try:
            ts_ms = int(float(point.get("t", 0)))
        except Exception:
            continue
        if ts_ms <= 0:
            continue
        merged_by_ts[ts_ms] = point

    for point in incoming:
        try:
            ts_ms = int(float(point.get("t", 0)))
        except Exception:
            continue
        if ts_ms <= 0:
            continue
        merged_by_ts[ts_ms] = point

    merged = [merged_by_ts[key] for key in sorted(merged_by_ts.keys())]
    return merged[-limit:]


def _bind_market_payload_history(
    raw_market: Any,
    normalized_history: dict[str, list[dict[str, float]]],
    alias_to_history_key: dict[str, str],
    limit: int,
    *,
    keep_existing_aliases: bool = False,
    now_ms: int | None = None,
    window_seconds: int | None = None,
    max_points: int | None = None,
) -> None:
    if not isinstance(raw_market, dict):
        return

    aliases = _collect_market_aliases(raw_market)
    if not aliases:
        return

    history_key = next((alias for alias in aliases if alias in normalized_history), None)
    normalize_kwargs: dict[str, Any] = {}
    if now_ms is not None:
        normalize_kwargs["now_ms"] = int(now_ms)
    if window_seconds is not None:
        normalize_kwargs["window_seconds"] = max(60, int(window_seconds))
    if max_points is not None:
        normalize_kwargs["max_points"] = max(2, int(max_points))
    normalized_points = normalize_binary_price_history(raw_market.get("price_history"), **normalize_kwargs)
    if len(normalized_points) >= 2:
        resolved_history_key = history_key or aliases[0]
        merged = _merge_normalized_binary_history(
            normalized_history.get(resolved_history_key, []),
            normalized_points,
            limit,
        )
        if len(merged) >= 2:
            normalized_history[resolved_history_key] = merged
            history_key = resolved_history_key

    if history_key:
        for alias in aliases:
            if keep_existing_aliases and alias in alias_to_history_key:
                continue
            alias_to_history_key[alias] = history_key


def _assert_not_globally_paused() -> None:
    if global_pause_state.is_paused:
        raise HTTPException(
            status_code=409,
            detail="Global pause is active. Use /workers/resume-all first.",
        )


def _apply_loss_streak_reset_metadata(
    metadata: dict[str, Any],
    *,
    reason: str,
) -> tuple[dict[str, Any], str]:
    stamped = dict(metadata or {})
    reset_at = to_iso(datetime.now(timezone.utc))
    if reset_at:
        stamped[_LOSS_STREAK_RESET_AT_KEY] = reset_at
    stamped[_LOSS_STREAK_RESET_REASON_KEY] = reason
    return stamped, reset_at or ""


@router.get("")
async def get_all_traders(
    mode: Optional[str] = Query(default=None),
    session: AsyncSession = Depends(get_db_session),
):
    mode_key = str(mode or "").strip().lower()
    if mode_key and mode_key not in {"paper", "shadow", "live"}:
        raise HTTPException(status_code=422, detail="mode must be 'paper', 'shadow', or 'live'")
    return {"traders": await list_traders(session, mode=mode_key or None)}


@router.get("/templates")
async def get_templates():
    return {"templates": list_trader_templates()}


@router.post("/from-template")
async def create_from_template(
    request: TraderTemplateCreateRequest,
    session: AsyncSession = Depends(get_db_session),
):
    try:
        trader = await create_trader_from_template(
            session,
            template_id=request.template_id,
            overrides=request.overrides,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    await create_trader_event(
        session,
        trader_id=trader["id"],
        event_type="trader_created",
        source="operator",
        operator=request.requested_by,
        message="Trader created from template",
        payload={"template_id": request.template_id},
    )
    return trader


@router.post("")
async def create_trader_route(
    request: TraderRequest,
    session: AsyncSession = Depends(get_db_session),
):
    payload = request.model_dump(exclude_unset=True, exclude={"requested_by", "reason"})
    try:
        trader = await create_trader(session, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    await create_config_revision(
        session,
        trader_id=trader["id"],
        operator=request.requested_by,
        reason=request.reason or "trader_create",
        orchestrator_before={},
        orchestrator_after={},
        trader_before={},
        trader_after=trader,
    )
    copy_from_trader_id = str(request.copy_from_trader_id or "").strip() or None
    event_payload: dict[str, Any] = {"trader": trader}
    event_message = "Trader created"
    if copy_from_trader_id:
        event_payload["copy_from_trader_id"] = copy_from_trader_id
        event_message = "Trader created from existing trader settings"
    await create_trader_event(
        session,
        trader_id=trader["id"],
        event_type="trader_created",
        source="operator",
        operator=request.requested_by,
        message=event_message,
        payload=event_payload,
    )
    return trader


@router.get("/orders/all")
async def get_all_trader_orders_all(
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=1000, ge=1, le=5000),
    session: AsyncSession = Depends(get_db_session),
):
    return {
        "orders": await list_serialized_trader_orders(
            session,
            trader_id=None,
            status=status,
            limit=limit,
        )
    }


@router.get("/market-history")
async def get_trader_market_history(
    market_ids: str = Query(default=""),
    limit: int = Query(default=120, ge=2, le=5000),
    session: AsyncSession = Depends(get_db_session),
):
    requested_ids: list[str] = []
    seen: set[str] = set()
    for raw in str(market_ids or "").split(","):
        normalized = normalize_market_id(raw)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        requested_ids.append(normalized)

    if not requested_ids:
        return {"histories": {}, "updated_at": None}

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    scanner_window_seconds = max(60, int(getattr(settings, "SCANNER_SPARKLINE_WINDOW_HOURS", 24)) * 3600)
    scanner_max_points = max(2, int(getattr(settings, "SCANNER_SPARKLINE_MAX_POINTS", 960)))
    normalize_max_points = max(limit, scanner_max_points)

    row = (await session.execute(select(ScannerSnapshot).where(ScannerSnapshot.id == "latest"))).scalar_one_or_none()
    history_map = row.market_history_json if row is not None and isinstance(row.market_history_json, dict) else {}
    opportunities = row.opportunities_json if row is not None and isinstance(row.opportunities_json, list) else []
    market_catalog_row = (
        await session.execute(select(MarketCatalog).where(MarketCatalog.id == "latest"))
    ).scalar_one_or_none()
    market_catalog = (
        market_catalog_row.markets_json
        if market_catalog_row is not None and isinstance(market_catalog_row.markets_json, list)
        else []
    )

    normalized_history: dict[str, list[dict[str, float]]] = {}
    for raw_market_id, raw_points in history_map.items():
        normalized_market_id = normalize_market_id(raw_market_id)
        if not normalized_market_id:
            continue
        normalized_points = normalize_binary_price_history(
            raw_points,
            now_ms=now_ms,
            window_seconds=scanner_window_seconds,
            max_points=normalize_max_points,
        )
        if len(normalized_points) >= 2:
            normalized_history[normalized_market_id] = _merge_normalized_binary_history(
                normalized_history.get(normalized_market_id, []),
                normalized_points,
                limit,
            )

    alias_to_history_key: dict[str, str] = {}
    catalog_by_alias: dict[str, dict[str, Any]] = {}
    for raw_opportunity in opportunities:
        if not isinstance(raw_opportunity, dict):
            continue
        markets = raw_opportunity.get("markets")
        if not isinstance(markets, list):
            continue
        for raw_market in markets:
            _bind_market_payload_history(
                raw_market,
                normalized_history,
                alias_to_history_key,
                limit,
                now_ms=now_ms,
                window_seconds=scanner_window_seconds,
                max_points=normalize_max_points,
            )

    for raw_market in market_catalog:
        if not isinstance(raw_market, dict):
            continue
        _bind_market_payload_history(
            raw_market,
            normalized_history,
            alias_to_history_key,
            limit,
            keep_existing_aliases=True,
            now_ms=now_ms,
            window_seconds=scanner_window_seconds,
            max_points=normalize_max_points,
        )
        aliases = _collect_market_aliases(raw_market)
        for alias in aliases:
            if alias not in catalog_by_alias:
                catalog_by_alias[alias] = raw_market

    unresolved_catalog_ids = [
        market_id
        for market_id in requested_ids
        if market_id not in catalog_by_alias
    ]
    if unresolved_catalog_ids:
        try:
            from services.polymarket import polymarket_client
        except Exception:
            polymarket_client = None  # type: ignore[assignment]
        if polymarket_client is not None:
            for market_id in unresolved_catalog_ids:
                try:
                    if market_id.startswith("0x"):
                        resolved_market = await polymarket_client.get_market_by_condition_id(market_id)
                    else:
                        resolved_market = await polymarket_client.get_market_by_token_id(market_id)
                except Exception:
                    resolved_market = None
                if not isinstance(resolved_market, dict):
                    continue
                _bind_market_payload_history(
                    resolved_market,
                    normalized_history,
                    alias_to_history_key,
                    limit,
                    keep_existing_aliases=True,
                    now_ms=now_ms,
                    window_seconds=scanner_window_seconds,
                    max_points=normalize_max_points,
                )
                for alias in _collect_market_aliases(resolved_market):
                    if alias not in catalog_by_alias:
                        catalog_by_alias[alias] = resolved_market

    backfill_market_ids = list(requested_ids)

    if backfill_market_ids:
        try:
            from models.opportunity import Opportunity
            from services.scanner import scanner as market_scanner
        except Exception:
            market_scanner = None
            Opportunity = None  # type: ignore[assignment]

        if market_scanner is not None and Opportunity is not None:
            backfill_targets: list[Any] = []
            for market_id in backfill_market_ids:
                catalog_market = catalog_by_alias.get(market_id)
                market_payload: dict[str, Any] = {
                    "id": market_id,
                }
                if isinstance(catalog_market, dict):
                    for key in (
                        "market_id",
                        "condition_id",
                        "conditionId",
                        "slug",
                        "market_slug",
                        "marketSlug",
                        "event_slug",
                        "eventSlug",
                        "event_ticker",
                        "eventTicker",
                        "ticker",
                        "yes_price",
                        "no_price",
                        "clob_token_ids",
                        "clobTokenIds",
                        "token_ids",
                        "tokenIds",
                        "tokens",
                    ):
                        value = catalog_market.get(key)
                        if value is not None:
                            market_payload[key] = value

                market_payload["platform"] = infer_market_platform(market_payload)
                backfill_targets.append(
                    Opportunity(
                        strategy="trader_history",
                        title=str(market_payload.get("question") or market_id),
                        description="Trader modal market history hydration",
                        total_cost=0.0,
                        expected_payout=1.0,
                        gross_profit=0.0,
                        fee=0.0,
                        net_profit=0.0,
                        roi_percent=0.0,
                        risk_score=0.5,
                        confidence=0.5,
                        markets=[market_payload],
                        min_liquidity=0.0,
                        max_position_size=0.0,
                        positions_to_take=[],
                    )
                )

            if backfill_targets:
                try:
                    backfill_attempt_map = getattr(market_scanner, "_market_history_backfill_attempt_ms", None)
                    if isinstance(backfill_attempt_map, dict):
                        for market_id in backfill_market_ids:
                            backfill_attempt_map.pop(str(market_id), None)
                    await market_scanner.attach_price_history_to_opportunities(
                        backfill_targets,
                        now=datetime.now(timezone.utc),
                        timeout_seconds=8.0,
                        block_for_backfill=True,
                    )
                    for opportunity in backfill_targets:
                        for raw_market in opportunity.markets:
                            _bind_market_payload_history(
                                raw_market,
                                normalized_history,
                                alias_to_history_key,
                                limit,
                                now_ms=now_ms,
                                window_seconds=scanner_window_seconds,
                                max_points=normalize_max_points,
                            )
                except Exception:
                    pass

    histories: dict[str, list[dict[str, float]]] = {}
    for market_id in requested_ids:
        resolved_key = alias_to_history_key.get(market_id, market_id)
        histories[market_id] = normalized_history.get(resolved_key, [])

    return {
        "histories": histories,
        "updated_at": row.updated_at.isoformat() if row is not None and row.updated_at is not None else None,
    }


@router.get("/{trader_id}/positions/live-wallet")
async def get_trader_live_wallet_positions(
    trader_id: str,
    include_managed: bool = Query(default=True),
    session: AsyncSession = Depends(get_db_session),
):
    trader = await get_trader(session, trader_id)
    if trader is None:
        raise HTTPException(status_code=404, detail="Trader not found")
    try:
        return await list_live_wallet_positions_for_trader(
            session,
            trader_id=trader_id,
            include_managed=bool(include_managed),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/{trader_id}/positions/live-wallet/adopt")
async def adopt_trader_live_wallet_position(
    trader_id: str,
    request: TraderLiveWalletPositionAdoptRequest,
    session: AsyncSession = Depends(get_db_session),
):
    trader = await get_trader(session, trader_id)
    if trader is None:
        raise HTTPException(status_code=404, detail="Trader not found")

    try:
        result = await adopt_live_wallet_position(
            session,
            trader_id=trader_id,
            token_id=request.token_id,
            reason=request.reason,
        )
    except ValueError as exc:
        message = str(exc)
        if "already managed" in message:
            raise HTTPException(status_code=409, detail=message)
        if "not found in execution wallet positions" in message:
            raise HTTPException(status_code=404, detail=message)
        if "Execution wallet address is not configured" in message:
            raise HTTPException(status_code=409, detail=message)
        raise HTTPException(status_code=422, detail=message)

    await create_trader_event(
        session,
        trader_id=trader_id,
        event_type="trader_live_position_adopted",
        severity="warn",
        source="operator",
        operator=request.requested_by,
        message="Live wallet position adopted for trader management",
        payload={
            "token_id": result.get("token_id"),
            "market_id": result.get("market_id"),
            "direction": result.get("direction"),
            "wallet_address": result.get("wallet_address"),
            "order_id": ((result.get("order") or {}).get("id") if isinstance(result.get("order"), dict) else None),
            "reason": request.reason,
        },
    )
    return result


@router.get("/{trader_id}")
async def get_trader_route(trader_id: str, session: AsyncSession = Depends(get_db_session)):
    trader = await get_trader(session, trader_id)
    if trader is None:
        raise HTTPException(status_code=404, detail="Trader not found")
    return trader


@router.put("/{trader_id}")
async def update_trader_route(
    trader_id: str,
    request: TraderPatchRequest,
    session: AsyncSession = Depends(get_db_session),
):
    before = await get_trader(session, trader_id)
    if before is None:
        raise HTTPException(status_code=404, detail="Trader not found")

    payload = request.model_dump(exclude_none=True, exclude={"requested_by", "reason"})
    was_running = bool(before.get("is_enabled", True)) and not bool(before.get("is_paused", False))
    will_be_enabled = bool(payload.get("is_enabled", before.get("is_enabled", True)))
    will_be_paused = bool(payload.get("is_paused", before.get("is_paused", False)))
    will_be_running = will_be_enabled and not will_be_paused
    loss_streak_reset_at = ""
    if will_be_running and not was_running:
        merged_metadata = dict(before.get("metadata") or {})
        incoming_metadata = payload.get("metadata")
        if isinstance(incoming_metadata, dict):
            merged_metadata.update(incoming_metadata)
        merged_metadata, loss_streak_reset_at = _apply_loss_streak_reset_metadata(
            merged_metadata,
            reason="operator_resume",
        )
        payload["metadata"] = merged_metadata

    try:
        after = await update_trader(session, trader_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if after is None:
        raise HTTPException(status_code=404, detail="Trader not found")

    await create_config_revision(
        session,
        trader_id=trader_id,
        operator=request.requested_by,
        reason=request.reason or "trader_update",
        orchestrator_before={},
        orchestrator_after={},
        trader_before=before,
        trader_after=after,
    )
    event_payload: dict[str, Any] = {"changes": payload}
    if loss_streak_reset_at:
        event_payload["loss_streak_reset_at"] = loss_streak_reset_at
    await create_trader_event(
        session,
        trader_id=trader_id,
        event_type="trader_updated",
        source="operator",
        operator=request.requested_by,
        message="Trader updated",
        payload=event_payload,
    )
    return after


@router.delete("/{trader_id}")
async def delete_trader_route(
    trader_id: str,
    action: TraderDeleteAction = Query(default=TraderDeleteAction.block),
    session: AsyncSession = Depends(get_db_session),
):
    trader = await get_trader(session, trader_id)
    if trader is None:
        raise HTTPException(status_code=404, detail="Trader not found")

    await sync_trader_position_inventory(session, trader_id=trader_id)
    open_summary = await get_open_position_summary_for_trader(session, trader_id)
    open_order_summary = await get_open_order_summary_for_trader(session, trader_id)
    open_live_positions = int(open_summary.get("live", 0))
    open_shadow_positions = int(open_summary.get("shadow", 0))
    open_paper_positions = int(open_summary.get("paper", 0))
    open_other_positions = int(open_summary.get("other", 0))
    open_total_positions = int(open_summary.get("total", 0))
    open_live_orders = int(open_order_summary.get("live", 0))
    open_shadow_orders = int(open_order_summary.get("shadow", 0))
    open_paper_orders = int(open_order_summary.get("paper", 0))
    open_other_orders = int(open_order_summary.get("other", 0))
    open_total_orders = int(open_order_summary.get("total", 0))

    if action == TraderDeleteAction.disable:
        updated = await update_trader(
            session,
            trader_id,
            {
                "is_enabled": False,
                "is_paused": True,
            },
        )
        if updated is None:
            raise HTTPException(status_code=404, detail="Trader not found")
        await create_trader_event(
            session,
            trader_id=trader_id,
            event_type="trader_delete_requested",
            severity="warn",
            source="operator",
            message="Trader disabled instead of deleted",
            payload={
                "open_live_positions": open_live_positions,
                "open_shadow_positions": open_shadow_positions,
                "open_paper_positions": open_paper_positions,
                "open_other_positions": open_other_positions,
                "open_live_orders": open_live_orders,
                "open_shadow_orders": open_shadow_orders,
                "open_paper_orders": open_paper_orders,
                "open_other_orders": open_other_orders,
            },
        )
        return {
            "status": "disabled",
            "trader_id": trader_id,
            "open_live_positions": open_live_positions,
            "open_shadow_positions": open_shadow_positions,
            "open_paper_positions": open_paper_positions,
            "open_other_positions": open_other_positions,
            "open_live_orders": open_live_orders,
            "open_shadow_orders": open_shadow_orders,
            "open_paper_orders": open_paper_orders,
            "open_other_orders": open_other_orders,
            "message": "Trader disabled and paused. Resolve non-paper exposure before permanent deletion.",
        }

    if (
        open_live_positions > 0
        or open_shadow_positions > 0
        or open_other_positions > 0
        or open_live_orders > 0
        or open_shadow_orders > 0
        or open_other_orders > 0
    ) and action != TraderDeleteAction.force_delete:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "open_live_exposure",
                "message": (
                    "Trader has non-paper exposure. Choose disable to pause safely, "
                    "or action=force_delete to permanently delete now."
                ),
                "trader_id": trader_id,
                "open_live_positions": open_live_positions,
                "open_shadow_positions": open_shadow_positions,
                "open_paper_positions": open_paper_positions,
                "open_other_positions": open_other_positions,
                "open_live_orders": open_live_orders,
                "open_shadow_orders": open_shadow_orders,
                "open_paper_orders": open_paper_orders,
                "open_other_orders": open_other_orders,
                "suggested_action": TraderDeleteAction.force_delete.value,
                "safe_action": TraderDeleteAction.disable.value,
            },
        )

    try:
        ok = await delete_trader(
            session,
            trader_id,
            force=(action == TraderDeleteAction.force_delete),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    if not ok:
        raise HTTPException(status_code=404, detail="Trader not found")
    await create_trader_event(
        session,
        trader_id=None,
        event_type="trader_deleted",
        severity="warn" if action == TraderDeleteAction.force_delete else "info",
        source="operator",
        message="Trader deleted",
        payload={
            "deleted_trader_id": trader_id,
            "action": action.value,
            "open_live_positions_at_delete": open_live_positions,
            "open_shadow_positions_at_delete": open_shadow_positions,
            "open_paper_positions_at_delete": open_paper_positions,
            "open_other_positions_at_delete": open_other_positions,
            "open_live_orders_at_delete": open_live_orders,
            "open_shadow_orders_at_delete": open_shadow_orders,
            "open_paper_orders_at_delete": open_paper_orders,
            "open_other_orders_at_delete": open_other_orders,
        },
    )
    return {
        "status": "deleted",
        "trader_id": trader_id,
        "action": action.value,
        "open_live_positions": open_live_positions,
        "open_shadow_positions": open_shadow_positions,
        "open_paper_positions": open_paper_positions,
        "open_other_positions": open_other_positions,
        "open_live_orders": open_live_orders,
        "open_shadow_orders": open_shadow_orders,
        "open_paper_orders": open_paper_orders,
        "open_other_orders": open_other_orders,
        "open_total_positions": open_total_positions,
        "open_total_orders": open_total_orders,
    }


@router.post("/{trader_id}/start")
async def start_trader(trader_id: str, session: AsyncSession = Depends(get_db_session)):
    _assert_not_globally_paused()
    before = await get_trader(session, trader_id)
    if before is None:
        raise HTTPException(status_code=404, detail="Trader not found")
    metadata, loss_streak_reset_at = _apply_loss_streak_reset_metadata(
        dict(before.get("metadata") or {}),
        reason="operator_start",
    )
    trader = await update_trader(
        session,
        trader_id,
        {
            "is_enabled": True,
            "is_paused": False,
            "metadata": metadata,
        },
    )
    if trader is None:
        raise HTTPException(status_code=404, detail="Trader not found")

    control = await read_orchestrator_control(session)
    mode = str(control.get("mode") or "paper").strip().lower()
    await sync_trader_position_inventory(
        session,
        trader_id=trader_id,
        mode=mode if mode in {"paper", "shadow", "live"} else None,
    )
    open_summary = await get_open_position_summary_for_trader(session, trader_id)
    open_live = int(open_summary.get("live", 0))
    open_shadow = int(open_summary.get("shadow", 0))
    open_paper = int(open_summary.get("paper", 0))
    if mode == "live" and open_live > 0:
        await create_trader_event(
            session,
            trader_id=trader_id,
            event_type="trader_start_warning",
            severity="warn",
            source="operator",
            message="Trader started with existing live open positions",
            payload={
                "open_live_positions": open_live,
                "open_paper_positions": open_paper,
            },
        )
    elif mode == "shadow" and open_shadow > 0:
        await create_trader_event(
            session,
            trader_id=trader_id,
            event_type="trader_start_notice",
            source="operator",
            message="Trader started with existing shadow open positions",
            payload={"open_shadow_positions": open_shadow},
        )
    elif mode == "paper" and open_paper > 0:
        await create_trader_event(
            session,
            trader_id=trader_id,
            event_type="trader_start_notice",
            source="operator",
            message="Trader started with existing paper open positions",
            payload={"open_paper_positions": open_paper},
        )

    await create_trader_event(
        session,
        trader_id=trader_id,
        event_type="trader_started",
        source="operator",
        message="Trader resumed",
        payload={_LOSS_STREAK_RESET_AT_KEY: loss_streak_reset_at} if loss_streak_reset_at else {},
    )
    return trader


@router.post("/{trader_id}/pause")
async def pause_trader(trader_id: str, session: AsyncSession = Depends(get_db_session)):
    trader = await set_trader_paused(session, trader_id, True)
    if trader is None:
        raise HTTPException(status_code=404, detail="Trader not found")
    await create_trader_event(
        session,
        trader_id=trader_id,
        event_type="trader_paused",
        source="operator",
        message="Trader paused",
    )
    return trader


@router.post("/{trader_id}/positions/cleanup")
async def cleanup_trader_positions(
    trader_id: str,
    request: TraderPositionCleanupRequest,
    session: AsyncSession = Depends(get_db_session),
):
    trader = await get_trader(session, trader_id)
    if trader is None:
        raise HTTPException(status_code=404, detail="Trader not found")

    if request.scope in {TraderPositionCleanupScope.live, TraderPositionCleanupScope.all} and not request.confirm_live:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "confirm_live_required",
                "message": "Live cleanup requires explicit confirmation. Re-submit with confirm_live=true.",
                "scope": request.scope.value,
            },
        )

    if request.method == TraderPositionCleanupMethod.mark_to_market and request.scope not in {
        TraderPositionCleanupScope.paper,
        TraderPositionCleanupScope.shadow,
        TraderPositionCleanupScope.live,
    }:
        raise HTTPException(
            status_code=422,
            detail="mark_to_market cleanup supports paper, shadow, and live scopes",
        )

    if request.method == TraderPositionCleanupMethod.cancel:
        try:
            result = await cleanup_trader_open_orders(
                session,
                trader_id=trader_id,
                scope=request.scope.value,
                max_age_hours=request.max_age_hours,
                dry_run=request.dry_run,
                target_status=request.target_status,
                reason=request.reason,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
    else:
        if request.scope == TraderPositionCleanupScope.live:
            lifecycle_result = await reconcile_live_positions(
                session,
                trader_id=trader_id,
                trader_params={},
                dry_run=request.dry_run,
                force_mark_to_market=True,
                max_age_hours=request.max_age_hours,
                reason=str(request.reason or "manual_mark_to_market_cleanup"),
            )
            if not request.dry_run:
                await sync_trader_position_inventory(session, trader_id=trader_id, mode="live")
        elif request.scope == TraderPositionCleanupScope.shadow:
            lifecycle_result = await reconcile_shadow_positions(
                session,
                trader_id=trader_id,
                trader_params={},
                dry_run=request.dry_run,
                force_mark_to_market=True,
                max_age_hours=request.max_age_hours,
                reason=str(request.reason or "manual_mark_to_market_cleanup"),
            )
            if not request.dry_run:
                await sync_trader_position_inventory(session, trader_id=trader_id, mode="shadow")
        else:
            lifecycle_result = await reconcile_paper_positions(
                session,
                trader_id=trader_id,
                trader_params={},
                dry_run=request.dry_run,
                force_mark_to_market=True,
                max_age_hours=request.max_age_hours,
                reason=str(request.reason or "manual_mark_to_market_cleanup"),
            )
            if not request.dry_run:
                await sync_trader_position_inventory(session, trader_id=trader_id, mode="paper")
        result = {
            "trader_id": trader_id,
            "scope": request.scope.value,
            "method": request.method.value,
            "dry_run": request.dry_run,
            "max_age_hours": request.max_age_hours,
            "matched": int(lifecycle_result.get("matched", 0)),
            "would_close": int(lifecycle_result.get("would_close", 0)),
            "updated": int(lifecycle_result.get("closed", 0)),
            "skipped": int(lifecycle_result.get("skipped", 0)),
            "held": int(lifecycle_result.get("held", 0)),
            "total_realized_pnl": float(lifecycle_result.get("total_realized_pnl", 0.0)),
            "by_status": lifecycle_result.get("by_status", {}),
            "skipped_reasons": lifecycle_result.get("skipped_reasons", {}),
            "details": lifecycle_result.get("details", []),
        }

    await create_trader_event(
        session,
        trader_id=trader_id,
        event_type="trader_positions_cleanup",
        severity=(
            "warn" if request.scope in {TraderPositionCleanupScope.live, TraderPositionCleanupScope.all} else "info"
        ),
        source="operator",
        message="Trader position cleanup executed" if not request.dry_run else "Trader position cleanup dry-run",
        payload=result,
    )
    return result


@router.post("/{trader_id}/run-once")
async def run_once(trader_id: str, session: AsyncSession = Depends(get_db_session)):
    _assert_not_globally_paused()
    trader = await request_trader_run(session, trader_id)
    if trader is None:
        raise HTTPException(status_code=404, detail="Trader not found")
    await create_trader_event(
        session,
        trader_id=trader_id,
        event_type="run_once_requested",
        source="operator",
        message="Run-once requested",
    )
    return trader


@router.get("/orders")
async def get_all_trader_orders(
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=2000, ge=1, le=5000),
    session: AsyncSession = Depends(get_db_session),
):
    return {
        "orders": await list_serialized_trader_orders(
            session,
            trader_id=None,
            status=status,
            limit=limit,
        )
    }


@router.get("/{trader_id}/decisions")
async def get_trader_decisions(
    trader_id: str,
    decision: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    session: AsyncSession = Depends(get_db_session),
):
    return {
        "decisions": await list_serialized_trader_decisions(
            session,
            trader_id=trader_id,
            decision=decision,
            limit=limit,
        )
    }


@router.get("/{trader_id}/orders")
async def get_trader_orders(
    trader_id: str,
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=5000),
    session: AsyncSession = Depends(get_db_session),
):
    return {
        "orders": await list_serialized_trader_orders(
            session,
            trader_id=trader_id,
            status=status,
            limit=limit,
        )
    }


@router.get("/{trader_id}/events")
async def get_events(
    trader_id: str,
    cursor: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=500),
    types: Optional[str] = Query(default=None),
    session: AsyncSession = Depends(get_db_session),
):
    event_types = [item.strip() for item in (types or "").split(",") if item.strip()]
    events, next_cursor = await list_serialized_trader_events(
        session,
        trader_id=trader_id,
        limit=limit,
        cursor=cursor,
        event_types=event_types or None,
    )
    return {"events": events, "next_cursor": next_cursor}


@router.get("/{trader_id}/execution-sessions")
async def get_execution_sessions(
    trader_id: str,
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    session: AsyncSession = Depends(get_db_session),
):
    trader = await get_trader(session, trader_id)
    if trader is None:
        raise HTTPException(status_code=404, detail="Trader not found")
    return {
        "sessions": await list_serialized_execution_sessions(
            session,
            trader_id=trader_id,
            status=status,
            limit=limit,
        )
    }


@router.get("/execution-sessions/{session_id}")
async def get_execution_session(
    session_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    detail = await get_serialized_execution_session_detail(session, session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Execution session not found")
    return detail


@router.post("/execution-sessions/{session_id}/cancel")
async def cancel_execution_session(
    session_id: str,
    request: TraderExecutionSessionControlRequest = TraderExecutionSessionControlRequest(),
    session: AsyncSession = Depends(get_db_session),
):
    engine = ExecutionSessionEngine(session)
    reason = str(request.reason or "manual_cancel").strip()
    ok = await engine.cancel_session(session_id=session_id, reason=reason)
    if not ok:
        raise HTTPException(status_code=404, detail="Execution session not found")
    return {"status": "cancelled", "session_id": session_id}


@router.post("/execution-sessions/{session_id}/pause")
async def pause_execution_session(
    session_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    engine = ExecutionSessionEngine(session)
    ok = await engine.pause_session(session_id=session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Execution session not found")
    return {"status": "paused", "session_id": session_id}


@router.post("/execution-sessions/{session_id}/resume")
async def resume_execution_session(
    session_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    engine = ExecutionSessionEngine(session)
    ok = await engine.resume_session(session_id=session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Execution session not found")
    return {"status": "working", "session_id": session_id}


@router.get("/decisions/{decision_id}")
async def get_decision_detail(
    decision_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    detail = await get_trader_decision_detail(session, decision_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Decision not found")
    return detail


@router.post("/{trader_id}/tune/iterate")
async def run_trader_tune_iteration(
    trader_id: str,
    request: TraderTuneAgentRequest,
    session: AsyncSession = Depends(get_db_session),
):
    trader = await get_trader(session, trader_id)
    if trader is None:
        raise HTTPException(status_code=404, detail="Trader not found")

    try:
        result = await run_strategy_tune_agent(
            trader_id=trader_id,
            prompt=request.prompt,
            model=request.model,
            max_iterations=request.max_iterations,
            monitor_job_id=request.monitor_job_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return result
