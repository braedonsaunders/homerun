"""API routes for the independent weather workflow pipeline."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    AsyncSessionLocal,
    SimulationTrade,
    WeatherTradeIntent,
    get_db_session,
)
from services.live_price_snapshot import (
    append_live_binary_price_point,
    get_live_mid_prices,
    normalize_binary_price_history,
)
from services.pause_state import global_pause_state
from services.signal_bus import emit_weather_intent_signals
from services.weather import shared_state
from utils.market_urls import serialize_opportunity_with_links
from services.weather.workflow_orchestrator import weather_workflow_orchestrator

router = APIRouter()


def _extract_binary_market_tokens(market: dict) -> tuple[Optional[str], Optional[str]]:
    token_ids = (
        market.get("clob_token_ids")
        or market.get("token_ids")
        or market.get("market_token_ids")
    )
    if not isinstance(token_ids, list):
        return None, None

    cleaned = [str(token_id).strip().lower() for token_id in token_ids if str(token_id or "").strip()]
    if len(cleaned) < 2:
        return None, None
    return cleaned[0], cleaned[1]


def _normalize_weather_price_histories(opportunities: list[dict]) -> None:
    for opp in opportunities:
        markets = opp.get("markets")
        if not isinstance(markets, list):
            continue
        for market in markets:
            if not isinstance(market, dict):
                continue
            normalized = normalize_binary_price_history(market.get("price_history"))
            if not normalized:
                continue
            market["price_history"] = normalized
            last = normalized[-1]
            yes_price = last.get("yes")
            no_price = last.get("no")
            if market.get("yes_price") is None and yes_price is not None:
                market["yes_price"] = yes_price
            if market.get("current_yes_price") is None and yes_price is not None:
                market["current_yes_price"] = yes_price
            if market.get("no_price") is None and no_price is not None:
                market["no_price"] = no_price
            if market.get("current_no_price") is None and no_price is not None:
                market["current_no_price"] = no_price


async def _attach_live_mid_prices_to_weather_payload(
    opportunities: list[dict],
) -> None:
    if not opportunities:
        return
    _normalize_weather_price_histories(opportunities)

    market_refs: list[tuple[dict, str, str]] = []
    unique_tokens: list[str] = []
    seen_tokens: set[str] = set()

    for opp in opportunities:
        markets = opp.get("markets")
        if not isinstance(markets, list):
            continue
        for market in markets:
            if not isinstance(market, dict):
                continue
            yes_token, no_token = _extract_binary_market_tokens(market)
            if not yes_token or not no_token:
                continue
            market_refs.append((market, yes_token, no_token))
            for token_id in (yes_token, no_token):
                if token_id in seen_tokens:
                    continue
                seen_tokens.add(token_id)
                unique_tokens.append(token_id)

    if not market_refs:
        return

    live_prices = await get_live_mid_prices(unique_tokens)
    if not live_prices:
        return

    for market, yes_token, no_token in market_refs:
        yes_price = live_prices.get(yes_token)
        no_price = live_prices.get(no_token)
        if yes_price is None and no_price is not None and 0.0 <= no_price <= 1.0:
            yes_price = float(1.0 - no_price)
        if no_price is None and yes_price is not None and 0.0 <= yes_price <= 1.0:
            no_price = float(1.0 - yes_price)
        if yes_price is None or no_price is None:
            continue

        market["yes_price"] = yes_price
        market["no_price"] = no_price
        market["current_yes_price"] = yes_price
        market["current_no_price"] = no_price
        market["outcome_prices"] = [yes_price, no_price]
        market["price_history"] = append_live_binary_price_point(
            market.get("price_history"),
            yes_price=yes_price,
            no_price=no_price,
        )


class WeatherWorkflowSettingsRequest(BaseModel):
    enabled: Optional[bool] = None
    auto_run: Optional[bool] = None
    scan_interval_seconds: Optional[int] = Field(None, ge=300, le=86400)
    entry_max_price: Optional[float] = Field(None, ge=0.01, le=0.99)
    take_profit_price: Optional[float] = Field(None, ge=0.01, le=0.99)
    stop_loss_pct: Optional[float] = Field(None, ge=0, le=100)
    min_edge_percent: Optional[float] = Field(None, ge=0, le=100)
    min_confidence: Optional[float] = Field(None, ge=0, le=1)
    min_model_agreement: Optional[float] = Field(None, ge=0, le=1)
    min_liquidity: Optional[float] = Field(None, ge=0)
    max_markets_per_scan: Optional[int] = Field(None, ge=10, le=5000)
    orchestrator_enabled: Optional[bool] = None
    orchestrator_min_edge: Optional[float] = Field(None, ge=0, le=100)
    orchestrator_max_age_minutes: Optional[int] = Field(None, ge=1, le=1440)
    default_size_usd: Optional[float] = Field(None, ge=1, le=1000)
    max_size_usd: Optional[float] = Field(None, ge=1, le=5000)
    model: Optional[str] = None
    temperature_unit: Optional[str] = Field(None, pattern=r"^(F|C)$")


async def _build_status_payload(session: AsyncSession) -> dict:
    status = await shared_state.get_weather_status_from_db(session)
    intents = await shared_state.list_weather_intents(session, status_filter="pending", limit=1000)
    status["pending_intents"] = len(intents)

    control = await shared_state.read_weather_control(session)
    status["paused"] = bool(control.get("is_paused", False))
    status["requested_scan_at"] = (
        control.get("requested_scan_at").isoformat() if control.get("requested_scan_at") else None
    )
    return status


@router.get("/weather-workflow/status")
async def get_weather_workflow_status(session: AsyncSession = Depends(get_db_session)):
    return await _build_status_payload(session)


@router.post("/weather-workflow/run")
async def run_weather_workflow_once(session: AsyncSession = Depends(get_db_session)):
    if global_pause_state.is_paused:
        raise HTTPException(
            status_code=409,
            detail="Global pause is active. Resume all workers before refreshing weather workflow.",
        )
    try:
        wf_settings = await shared_state.get_weather_settings(session)
        result = await weather_workflow_orchestrator.run_cycle(session)
        pending_rows = await shared_state.list_weather_intents(session, status_filter="pending", limit=2000)
        emitted = await emit_weather_intent_signals(
            session,
            pending_rows,
            max_age_minutes=int(
                max(
                    1,
                    wf_settings.get("orchestrator_max_age_minutes", 240) or 240,
                )
            ),
        )
        # Clear any previously queued manual run requests to avoid duplicate
        # immediate reruns by the background weather worker loop.
        await shared_state.clear_weather_scan_request(session)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Weather refresh failed: {exc}") from exc

    return {
        "status": "completed",
        "message": "Weather workflow refreshed.",
        "cycle": result,
        "signals_emitted": int(emitted),
        **await _build_status_payload(session),
    }


@router.post("/weather-workflow/start")
async def start_weather_workflow(session: AsyncSession = Depends(get_db_session)):
    if global_pause_state.is_paused:
        raise HTTPException(
            status_code=409,
            detail="Global pause is active. Use /workers/resume-all first.",
        )
    await shared_state.set_weather_paused(session, False)
    return {
        "status": "started",
        **await _build_status_payload(session),
    }


@router.post("/weather-workflow/pause")
async def pause_weather_workflow(session: AsyncSession = Depends(get_db_session)):
    await shared_state.set_weather_paused(session, True)
    return {
        "status": "paused",
        **await _build_status_payload(session),
    }


@router.post("/weather-workflow/interval")
async def set_weather_workflow_interval(
    interval_seconds: int = Query(..., ge=300, le=86400),
    session: AsyncSession = Depends(get_db_session),
):
    await shared_state.set_weather_interval(session, interval_seconds)
    # Keep settings and control aligned.
    await shared_state.update_weather_settings(session, {"scan_interval_seconds": interval_seconds})
    return {
        "status": "updated",
        **await _build_status_payload(session),
    }


@router.get("/weather-workflow/opportunities")
async def get_weather_opportunities(
    session: AsyncSession = Depends(get_db_session),
    min_edge: Optional[float] = Query(None, ge=0),
    direction: Optional[str] = Query(None),
    max_entry: Optional[float] = Query(None, ge=0.01, le=0.99),
    location: Optional[str] = Query(None),
    target_date: Optional[date] = None,
    include_report_only: bool = Query(False),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    opps = await shared_state.get_weather_opportunities_from_db(
        session,
        min_edge_percent=min_edge,
        direction=direction,
        max_entry_price=max_entry,
        location_query=location,
        target_date=target_date,
        require_tradable_markets=True,
        exclude_near_resolution=True,
        include_report_only=include_report_only,
    )

    total = len(opps)
    opps = opps[offset : offset + limit]
    serialized = [serialize_opportunity_with_links(o) for o in opps]
    await _attach_live_mid_prices_to_weather_payload(serialized)
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "opportunities": serialized,
    }


@router.get("/weather-workflow/opportunity-dates")
async def get_weather_opportunity_dates(
    session: AsyncSession = Depends(get_db_session),
    min_edge: Optional[float] = Query(None, ge=0),
    direction: Optional[str] = Query(None),
    max_entry: Optional[float] = Query(None, ge=0.01, le=0.99),
    location: Optional[str] = Query(None),
    include_report_only: bool = Query(False),
):
    date_counts = await shared_state.get_weather_target_date_counts_from_db(
        session,
        min_edge_percent=min_edge,
        direction=direction,
        max_entry_price=max_entry,
        location_query=location,
        require_tradable_markets=True,
        exclude_near_resolution=True,
        include_report_only=include_report_only,
    )
    return {"total_dates": len(date_counts), "dates": date_counts}


@router.get("/weather-workflow/intents")
async def get_weather_intents(
    session: AsyncSession = Depends(get_db_session),
    status_filter: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    rows = await shared_state.list_weather_intents(session, status_filter=status_filter, limit=limit)
    intents = [
        {
            "id": r.id,
            "market_id": r.market_id,
            "market_question": r.market_question,
            "direction": r.direction,
            "entry_price": r.entry_price,
            "take_profit_price": r.take_profit_price,
            "stop_loss_pct": r.stop_loss_pct,
            "model_probability": r.model_probability,
            "edge_percent": r.edge_percent,
            "confidence": r.confidence,
            "model_agreement": r.model_agreement,
            "suggested_size_usd": r.suggested_size_usd,
            "metadata": r.metadata_json,
            "status": r.status,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "consumed_at": r.consumed_at.isoformat() if r.consumed_at else None,
        }
        for r in rows
    ]
    return {"total": len(intents), "intents": intents}


@router.post("/weather-workflow/intents/{intent_id}/skip")
async def skip_weather_intent(
    intent_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    ok = await shared_state.mark_weather_intent(session, intent_id, "skipped")
    if not ok:
        raise HTTPException(status_code=404, detail="Intent not found")
    return {"status": "skipped", "intent_id": intent_id}


@router.get("/weather-workflow/settings")
async def get_weather_workflow_settings(
    session: AsyncSession = Depends(get_db_session),
):
    return await shared_state.get_weather_settings(session)


@router.put("/weather-workflow/settings")
async def update_weather_workflow_settings(
    request: WeatherWorkflowSettingsRequest,
    session: AsyncSession = Depends(get_db_session),
):
    updates = request.model_dump(exclude_unset=True)
    settings = await shared_state.update_weather_settings(session, updates)

    if "scan_interval_seconds" in updates:
        await shared_state.set_weather_interval(session, int(updates["scan_interval_seconds"]))

    return {"status": "success", "settings": settings}


@router.get("/weather-workflow/performance")
async def get_weather_workflow_performance(
    lookback_days: int = Query(90, ge=1, le=3650),
):
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    async with AsyncSessionLocal() as session:
        # Simulation trades executed via weather strategy.
        trades_result = await session.execute(
            select(SimulationTrade).where(
                SimulationTrade.strategy_type == "weather_edge",
                SimulationTrade.executed_at >= cutoff,
            )
        )
        trades = list(trades_result.scalars().all())

        win_count = 0
        loss_count = 0
        resolved = 0
        pnl = 0.0
        for t in trades:
            if t.actual_pnl is not None:
                resolved += 1
                pnl += float(t.actual_pnl)
                if float(t.actual_pnl) > 0:
                    win_count += 1
                elif float(t.actual_pnl) < 0:
                    loss_count += 1

        intents_total = (
            await session.execute(select(func.count()).select_from(WeatherTradeIntent))
        ).scalar_one_or_none() or 0

        pending_intents = (
            await session.execute(
                select(func.count()).select_from(WeatherTradeIntent).where(WeatherTradeIntent.status == "pending")
            )
        ).scalar_one_or_none() or 0

        executed_intents = (
            await session.execute(
                select(func.count()).select_from(WeatherTradeIntent).where(WeatherTradeIntent.status == "executed")
            )
        ).scalar_one_or_none() or 0

    return {
        "lookback_days": lookback_days,
        "trades_total": len(trades),
        "trades_resolved": resolved,
        "wins": win_count,
        "losses": loss_count,
        "win_rate": (win_count / resolved) if resolved > 0 else 0.0,
        "total_pnl": pnl,
        "intents_total": intents_total,
        "pending_intents": pending_intents,
        "executed_intents": executed_intents,
    }
