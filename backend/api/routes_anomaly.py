from datetime import datetime
from utils.utcnow import utcfromtimestamp

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from pydantic import BaseModel, field_validator

from services.anomaly_detector import anomaly_detector, AnomalyType, Severity
from services.polymarket import polymarket_client
from utils.validation import validate_eth_address

anomaly_router = APIRouter()


def _normalize_timestamp(trade: dict) -> str:
    """Normalize trade timestamps to ISO format, handling Unix seconds, ms, and strings."""
    ts = (
        trade.get("match_time")
        or trade.get("timestamp")
        or trade.get("time")
        or trade.get("created_at")
        or trade.get("createdAt")
    )
    if not ts:
        return ""
    try:
        if isinstance(ts, (int, float)):
            # Unix timestamps < 1e12 are in seconds; >= 1e12 are milliseconds
            if ts > 1e12:
                ts = ts / 1000
            return utcfromtimestamp(ts).isoformat() + "Z"
        if isinstance(ts, str):
            if "T" in ts or "-" in ts:
                return ts
            # Numeric string
            numeric = float(ts)
            if numeric > 1e12:
                numeric = numeric / 1000
            return utcfromtimestamp(numeric).isoformat() + "Z"
    except (ValueError, TypeError, OSError):
        pass
    return ""


class AnalyzeWalletRequest(BaseModel):
    address: str

    @field_validator("address")
    @classmethod
    def validate_wallet(cls, v: str) -> str:
        return validate_eth_address(v)


class FindProfitableRequest(BaseModel):
    min_trades: int = 50
    min_win_rate: float = 0.6
    min_pnl: float = 1000.0
    max_anomaly_score: float = 0.5


# ==================== WALLET ANALYSIS ====================


@anomaly_router.post("/analyze")
async def analyze_wallet(request: AnalyzeWalletRequest):
    """
    Analyze a wallet for trading patterns and anomalies.

    Returns comprehensive analysis including:
    - Trading statistics (win rate, ROI, PnL)
    - Detected strategies being used
    - Anomaly detection results
    - Recommendation for whether to copy
    """
    analysis = await anomaly_detector.analyze_wallet(request.address)

    return {
        "wallet": analysis.address,
        "stats": {
            "total_trades": analysis.total_trades,
            "win_rate": analysis.win_rate,
            "total_pnl": analysis.total_pnl,
            "avg_roi": analysis.avg_roi,
            "max_roi": analysis.max_roi,
            "avg_hold_time_hours": analysis.avg_hold_time_hours,
            "trade_frequency_per_day": analysis.trade_frequency_per_day,
            "markets_traded": analysis.markets_traded,
        },
        "strategies_detected": analysis.strategies_detected,
        "anomaly_score": analysis.anomaly_score,
        "anomalies": analysis.anomalies,
        "is_profitable_pattern": analysis.is_profitable_pattern,
        "recommendation": analysis.recommendation,
    }


@anomaly_router.get("/analyze/{wallet_address}")
async def analyze_wallet_get(wallet_address: str):
    """Analyze a wallet (GET method for convenience)"""
    try:
        address = validate_eth_address(wallet_address)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    analysis = await anomaly_detector.analyze_wallet(address)

    return {
        "wallet": analysis.address,
        "stats": {
            "total_trades": analysis.total_trades,
            "win_rate": analysis.win_rate,
            "total_pnl": analysis.total_pnl,
            "avg_roi": analysis.avg_roi,
            "max_roi": analysis.max_roi,
            "avg_hold_time_hours": analysis.avg_hold_time_hours,
            "trade_frequency_per_day": analysis.trade_frequency_per_day,
            "markets_traded": analysis.markets_traded,
        },
        "strategies_detected": analysis.strategies_detected,
        "anomaly_score": analysis.anomaly_score,
        "anomalies": analysis.anomalies,
        "is_profitable_pattern": analysis.is_profitable_pattern,
        "recommendation": analysis.recommendation,
    }


# ==================== FIND PROFITABLE WALLETS ====================


@anomaly_router.post("/find-profitable")
async def find_profitable_wallets(request: FindProfitableRequest):
    """
    Find wallets with profitable patterns that aren't suspicious.

    This is used to discover wallets worth copying.
    Filters out wallets with high anomaly scores (suspicious activity).
    """
    wallets = await anomaly_detector.find_profitable_wallets(
        min_trades=request.min_trades,
        min_win_rate=request.min_win_rate,
        min_pnl=request.min_pnl,
        max_anomaly_score=request.max_anomaly_score,
    )

    return {
        "count": len(wallets),
        "wallets": [
            {
                "address": w.address,
                "win_rate": w.win_rate,
                "total_pnl": w.total_pnl,
                "avg_roi": w.avg_roi,
                "strategies": w.strategies_detected,
                "anomaly_score": w.anomaly_score,
                "recommendation": w.recommendation,
            }
            for w in wallets
        ],
    }


# ==================== ANOMALIES ====================


@anomaly_router.get("/anomalies")
async def get_anomalies(
    severity: Optional[str] = Query(
        default=None, description="Filter by severity: low, medium, high, critical"
    ),
    anomaly_type: Optional[str] = Query(
        default=None, description="Filter by anomaly type"
    ),
    limit: int = Query(default=100, ge=1, le=500),
):
    """Get detected anomalies"""
    # Validate severity
    if severity and severity not in [s.value for s in Severity]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid severity. Must be one of: {[s.value for s in Severity]}",
        )

    # Validate anomaly type
    if anomaly_type and anomaly_type not in [t.value for t in AnomalyType]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid anomaly type. Must be one of: {[t.value for t in AnomalyType]}",
        )

    anomalies = await anomaly_detector.get_anomalies(
        severity=severity, anomaly_type=anomaly_type, limit=limit
    )

    return {"count": len(anomalies), "anomalies": anomalies}


@anomaly_router.get("/anomaly-types")
async def get_anomaly_types():
    """Get all anomaly types and their descriptions"""
    return {
        "types": [
            {
                "type": t.value,
                "category": "statistical"
                if t.value
                in [
                    "impossible_win_rate",
                    "unusual_roi",
                    "perfect_timing",
                    "statistically_impossible",
                ]
                else "pattern"
                if t.value in ["front_running", "wash_trading", "coordinated_trading"]
                else "behavioral",
            }
            for t in AnomalyType
        ],
        "severities": [s.value for s in Severity],
    }


# ==================== QUICK CHECKS ====================


@anomaly_router.get("/check/{wallet_address}")
async def quick_check_wallet(wallet_address: str):
    """
    Quick check if a wallet is suspicious.

    Returns a simple pass/fail with basic stats.
    Use /analyze for full analysis.
    """
    try:
        address = validate_eth_address(wallet_address)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    analysis = await anomaly_detector.analyze_wallet(address)

    is_suspicious = analysis.anomaly_score > 0.5
    critical_anomalies = [a for a in analysis.anomalies if a["severity"] == "critical"]

    return {
        "wallet": address,
        "is_suspicious": is_suspicious,
        "anomaly_score": analysis.anomaly_score,
        "critical_anomalies": len(critical_anomalies),
        "win_rate": analysis.win_rate,
        "total_pnl": analysis.total_pnl,
        "verdict": "AVOID" if is_suspicious else "OK",
        "summary": analysis.recommendation,
    }


# ==================== WALLET TRADES & POSITIONS ====================


@anomaly_router.get("/wallet/{wallet_address}/trades")
async def get_wallet_trades(
    wallet_address: str, limit: int = Query(default=100, ge=1, le=500)
):
    """
    Get individual trades for a wallet.

    Returns raw trade data with calculated cost per trade.
    """
    try:
        address = validate_eth_address(wallet_address)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    trades = await polymarket_client.get_wallet_trades(address, limit=limit)

    # Enrich trades with market titles from Gamma API / cache
    trades = await polymarket_client.enrich_trades_with_market_info(trades)

    # Enrich trades with calculated fields
    enriched_trades = []
    for trade in trades:
        size = float(trade.get("size", 0) or trade.get("amount", 0) or 0)
        price = float(trade.get("price", 0) or 0)
        side = (trade.get("side", "") or "").upper()
        cost = size * price

        enriched_trades.append(
            {
                "id": trade.get("id", trade.get("transactionHash", "")),
                "market": trade.get(
                    "market",
                    trade.get(
                        "conditionId",
                        trade.get("condition_id", trade.get("asset", "")),
                    ),
                ),
                "market_slug": trade.get("market_slug", trade.get("slug", "")),
                "market_title": trade.get("market_title", trade.get("title", "")),
                "event_slug": trade.get(
                    "event_slug", trade.get("eventSlug", "")
                ),
                "outcome": trade.get("outcome", trade.get("outcome_index", "")),
                "side": side,
                "size": size,
                "price": price,
                "cost": cost,
                "timestamp": _normalize_timestamp(trade),
                "transaction_hash": trade.get(
                    "transactionHash", trade.get("transaction_hash", "")
                ),
            }
        )

    return {"wallet": address, "total": len(enriched_trades), "trades": enriched_trades}


@anomaly_router.get("/wallet/{wallet_address}/positions")
async def get_wallet_positions(wallet_address: str):
    """
    Get current open positions for a wallet with real-time market prices.
    """
    try:
        address = validate_eth_address(wallet_address)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Use enriched positions with current market prices from CLOB API
    positions = await polymarket_client.get_wallet_positions_with_prices(address)

    # Resolve market titles for positions missing them
    # Collect condition IDs that need lookup
    import asyncio

    condition_ids_to_lookup = set()
    for pos in positions:
        title = pos.get("title", "")
        if not title:
            cid = (
                pos.get("conditionId", "")
                or pos.get("condition_id", "")
                or pos.get("market", "")
            )
            if cid:
                condition_ids_to_lookup.add(cid)

    # Batch lookup market titles
    if condition_ids_to_lookup:
        semaphore = asyncio.Semaphore(5)

        async def lookup(cid: str):
            async with semaphore:
                await polymarket_client.get_market_by_condition_id(cid)

        await asyncio.gather(*[lookup(cid) for cid in condition_ids_to_lookup])

    # Enrich positions with calculated fields
    enriched_positions = []
    total_value = 0.0
    total_unrealized_pnl = 0.0

    for pos in positions:
        size = float(pos.get("size", 0) or 0)
        avg_price = float(pos.get("avgPrice", pos.get("avg_price", 0)) or 0)
        current_price = float(
            pos.get("currentPrice", pos.get("curPrice", pos.get("price", 0))) or 0
        )

        # Use API-provided values when available, fallback to manual calculation
        current_value = float(pos.get("currentValue", pos.get("current_value", 0)) or 0)
        cost_basis = float(pos.get("initialValue", pos.get("initial_value", 0)) or 0)

        if current_value == 0 and cost_basis == 0:
            cost_basis = size * avg_price
            current_value = size * current_price

        unrealized_pnl = current_value - cost_basis
        roi = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0

        total_value += current_value
        total_unrealized_pnl += unrealized_pnl

        # Resolve market title from API data or Gamma cache
        condition_id = (
            pos.get("conditionId", "")
            or pos.get("condition_id", "")
            or pos.get("market", "")
            or pos.get("asset", "")
        )
        title = pos.get("title", "")
        if not title and condition_id:
            market_info = polymarket_client._market_cache.get(condition_id)
            if market_info:
                title = market_info.get("groupItemTitle") or market_info.get(
                    "question", ""
                )
        market_slug = pos.get("market_slug", pos.get("slug", ""))
        event_slug = pos.get("event_slug", pos.get("eventSlug", ""))
        if (not market_slug or not event_slug) and condition_id:
            market_info = polymarket_client._market_cache.get(condition_id)
            if market_info:
                if not market_slug:
                    market_slug = market_info.get("slug", "")
                if not event_slug:
                    event_slug = market_info.get("event_slug", "")

        enriched_positions.append(
            {
                "market": condition_id,
                "title": title,
                "market_slug": market_slug,
                "event_slug": event_slug,
                "outcome": pos.get("outcome", pos.get("outcome_index", "")),
                "size": size,
                "avg_price": avg_price,
                "current_price": current_price,
                "cost_basis": cost_basis,
                "current_value": current_value,
                "unrealized_pnl": unrealized_pnl,
                "roi_percent": roi,
            }
        )

    return {
        "wallet": address,
        "total_positions": len(enriched_positions),
        "total_value": total_value,
        "total_unrealized_pnl": total_unrealized_pnl,
        "positions": enriched_positions,
    }


@anomaly_router.get("/wallet/{wallet_address}/summary")
async def get_wallet_summary(wallet_address: str):
    """
    Get a comprehensive summary of a wallet including trades, positions, and analysis.
    """
    try:
        address = validate_eth_address(wallet_address)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Fetch all data in parallel - use enriched positions with current prices
    trades = await polymarket_client.get_wallet_trades(address, limit=500)
    positions = await polymarket_client.get_wallet_positions_with_prices(address)

    # Calculate summary stats
    total_invested = 0.0
    total_returned = 0.0
    buys = 0
    sells = 0

    for trade in trades:
        size = float(trade.get("size", 0) or trade.get("amount", 0) or 0)
        price = float(trade.get("price", 0) or 0)
        side = (trade.get("side", "") or "").upper()
        cost = size * price

        if side == "BUY":
            total_invested += cost
            buys += 1
        elif side == "SELL":
            total_returned += cost
            sells += 1

    # Calculate position values - use API-provided values with fallback
    position_value = 0.0
    position_cost_basis = 0.0
    for pos in positions:
        cv = float(pos.get("currentValue", pos.get("current_value", 0)) or 0)
        iv = float(pos.get("initialValue", pos.get("initial_value", 0)) or 0)

        if cv == 0 and iv == 0:
            size = float(pos.get("size", 0) or 0)
            avg_price = float(pos.get("avgPrice", pos.get("avg_price", 0)) or 0)
            current_price = float(
                pos.get("currentPrice", pos.get("curPrice", pos.get("price", 0))) or 0
            )
            cv = size * current_price
            iv = size * avg_price

        position_value += cv
        position_cost_basis += iv

    realized_pnl = total_returned - total_invested
    unrealized_pnl = position_value - position_cost_basis
    total_pnl = realized_pnl + unrealized_pnl
    roi = (total_pnl / total_invested * 100) if total_invested > 0 else 0

    return {
        "wallet": address,
        "summary": {
            "total_trades": len(trades),
            "buys": buys,
            "sells": sells,
            "open_positions": len(positions),
            "total_invested": total_invested,
            "total_returned": total_returned,
            "position_value": position_value,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": total_pnl,
            "roi_percent": roi,
        },
    }
