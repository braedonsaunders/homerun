from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import datetime

from models import ArbitrageOpportunity, StrategyType, OpportunityFilter
from services import scanner, wallet_tracker, polymarket_client

router = APIRouter()


# ==================== OPPORTUNITIES ====================

@router.get("/opportunities", response_model=list[ArbitrageOpportunity])
async def get_opportunities(
    min_profit: float = Query(0.0, description="Minimum profit percentage"),
    max_risk: float = Query(1.0, description="Maximum risk score (0-1)"),
    strategy: Optional[StrategyType] = Query(None, description="Filter by strategy type"),
    min_liquidity: float = Query(0.0, description="Minimum liquidity in USD"),
    search: Optional[str] = Query(None, description="Search query for market titles"),
    limit: int = Query(50, description="Maximum results to return"),
    offset: int = Query(0, description="Number of results to skip")
):
    """Get current arbitrage opportunities"""
    filter = OpportunityFilter(
        min_profit=min_profit / 100,  # Convert from percentage
        max_risk=max_risk,
        strategies=[strategy] if strategy else [],
        min_liquidity=min_liquidity
    )

    opportunities = scanner.get_opportunities(filter)

    # Apply search filter if provided
    if search:
        search_lower = search.lower()
        opportunities = [
            opp for opp in opportunities
            if search_lower in opp.title.lower()
            or (opp.event_title and search_lower in opp.event_title.lower())
            or any(search_lower in m.get("question", "").lower() for m in opp.markets)
        ]

    # Apply pagination
    total = len(opportunities)
    paginated = opportunities[offset:offset + limit]

    # Return as response with total count header
    from fastapi.responses import JSONResponse
    response = JSONResponse(content=[o.model_dump() for o in paginated])
    response.headers["X-Total-Count"] = str(total)
    return response


@router.get("/opportunities/{opportunity_id}", response_model=ArbitrageOpportunity)
async def get_opportunity(opportunity_id: str):
    """Get a specific opportunity by ID"""
    opportunities = scanner.get_opportunities()
    for opp in opportunities:
        if opp.id == opportunity_id:
            return opp
    raise HTTPException(status_code=404, detail="Opportunity not found")


@router.post("/scan")
async def trigger_scan():
    """Manually trigger a new scan"""
    try:
        opportunities = await scanner.scan_once()
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "opportunities_found": len(opportunities),
            "top_opportunities": [
                {
                    "id": o.id,
                    "strategy": o.strategy,
                    "title": o.title,
                    "roi_percent": o.roi_percent
                }
                for o in opportunities[:10]
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== SCANNER STATUS ====================

@router.get("/scanner/status")
async def get_scanner_status():
    """Get scanner status"""
    return scanner.get_status()


@router.post("/scanner/start")
async def start_scanner():
    """Start/resume the scanner"""
    await scanner.start()
    return {"status": "started", **scanner.get_status()}


@router.post("/scanner/pause")
async def pause_scanner():
    """Pause the scanner (keeps loop running but doesn't scan)"""
    await scanner.pause()
    return {"status": "paused", **scanner.get_status()}


@router.post("/scanner/interval")
async def set_scanner_interval(interval_seconds: int = Query(..., ge=10, le=3600)):
    """Set the scan interval (10-3600 seconds)"""
    await scanner.set_interval(interval_seconds)
    return {"status": "updated", **scanner.get_status()}


# ==================== WALLETS ====================

@router.get("/wallets")
async def get_tracked_wallets():
    """Get all tracked wallets"""
    return wallet_tracker.get_all_wallets()


@router.post("/wallets")
async def add_wallet(address: str, label: Optional[str] = None):
    """Add a wallet to track"""
    await wallet_tracker.add_wallet(address, label)
    return {"status": "success", "address": address, "label": label}


@router.delete("/wallets/{address}")
async def remove_wallet(address: str):
    """Remove a tracked wallet"""
    wallet_tracker.remove_wallet(address)
    return {"status": "success", "address": address}


@router.get("/wallets/{address}")
async def get_wallet_info(address: str):
    """Get info for a specific wallet"""
    info = wallet_tracker.get_wallet_info(address)
    if not info:
        raise HTTPException(status_code=404, detail="Wallet not found")
    return info


@router.get("/wallets/{address}/positions")
async def get_wallet_positions(address: str):
    """Get current positions for a wallet"""
    try:
        positions = await polymarket_client.get_wallet_positions(address)
        return {"address": address, "positions": positions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/wallets/{address}/trades")
async def get_wallet_trades(address: str, limit: int = 100):
    """Get recent trades for a wallet"""
    try:
        trades = await polymarket_client.get_wallet_trades(address, limit)
        return {"address": address, "trades": trades}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== MARKETS ====================

@router.get("/markets")
async def get_markets(
    active: bool = True,
    limit: int = 100,
    offset: int = 0
):
    """Get markets from Polymarket"""
    try:
        markets = await polymarket_client.get_markets(
            active=active,
            limit=limit,
            offset=offset
        )
        return [m.model_dump() for m in markets]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events")
async def get_events(
    closed: bool = False,
    limit: int = 100,
    offset: int = 0
):
    """Get events from Polymarket"""
    try:
        events = await polymarket_client.get_events(
            closed=closed,
            limit=limit,
            offset=offset
        )
        return [e.model_dump() for e in events]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== STRATEGY INFO ====================

@router.get("/strategies")
async def get_strategies():
    """Get information about available strategies"""
    return [
        {
            "type": s.strategy_type.value,
            "name": s.name,
            "description": s.description
        }
        for s in scanner.strategies
    ]
