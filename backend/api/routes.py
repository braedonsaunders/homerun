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
    category: Optional[str] = Query(None, description="Filter by category (e.g., politics, sports, crypto)"),
    limit: int = Query(50, description="Maximum results to return"),
    offset: int = Query(0, description="Number of results to skip")
):
    """Get current arbitrage opportunities"""
    filter = OpportunityFilter(
        min_profit=min_profit / 100,  # Convert from percentage
        max_risk=max_risk,
        strategies=[strategy] if strategy else [],
        min_liquidity=min_liquidity,
        category=category
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
    # Use mode='json' to properly serialize datetime objects
    response = JSONResponse(content=[o.model_dump(mode='json') for o in paginated])
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


# ==================== OPPORTUNITIES CLEANUP ====================

@router.delete("/opportunities")
async def clear_opportunities():
    """
    Clear all opportunities from memory.

    This removes all detected arbitrage opportunities.
    They will be repopulated on the next scan.
    """
    count = scanner.clear_opportunities()
    return {
        "status": "success",
        "cleared_count": count,
        "message": f"Cleared {count} opportunities. Next scan will repopulate."
    }


@router.post("/opportunities/cleanup")
async def cleanup_opportunities(
    remove_expired: bool = Query(True, description="Remove opportunities past resolution date"),
    max_age_minutes: Optional[int] = Query(None, ge=1, le=1440, description="Remove opportunities older than X minutes")
):
    """
    Clean up stale opportunities.

    - remove_expired: Remove opportunities whose resolution date has passed
    - max_age_minutes: Remove opportunities detected more than X minutes ago
    """
    results = {}

    if remove_expired:
        expired_count = scanner.remove_expired_opportunities()
        results["expired_removed"] = expired_count

    if max_age_minutes:
        old_count = scanner.remove_old_opportunities(max_age_minutes)
        results["old_removed"] = old_count

    results["remaining_count"] = len(scanner.get_opportunities())

    return {
        "status": "success",
        **results
    }


# ==================== WALLETS ====================

@router.get("/wallets")
async def get_tracked_wallets():
    """Get all tracked wallets"""
    return await wallet_tracker.get_all_wallets()


@router.post("/wallets")
async def add_wallet(address: str, label: Optional[str] = None):
    """Add a wallet to track"""
    await wallet_tracker.add_wallet(address, label)
    return {"status": "success", "address": address, "label": label}


@router.delete("/wallets/{address}")
async def remove_wallet(address: str):
    """Remove a tracked wallet"""
    await wallet_tracker.remove_wallet(address)
    return {"status": "success", "address": address}


@router.get("/wallets/{address}")
async def get_wallet_info(address: str):
    """Get info for a specific wallet"""
    info = await wallet_tracker.get_wallet_info(address)
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


@router.get("/wallets/recent-trades/all")
async def get_all_recent_trades(
    limit: int = Query(50, ge=1, le=200, description="Maximum trades to return"),
    hours: int = Query(24, ge=1, le=168, description="Only show trades from last N hours")
):
    """
    Get recent trades from all tracked wallets, aggregated and sorted by timestamp.

    This provides a feed of the most recent/timely trading opportunities based on
    what wallets you're tracking are doing.
    """
    try:
        from datetime import timedelta

        wallets = await wallet_tracker.get_all_wallets()
        all_trades = []

        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        for wallet in wallets:
            wallet_address = wallet.get("address", "")
            wallet_label = wallet.get("label", wallet_address[:10] + "...")
            recent_trades = wallet.get("recent_trades", [])

            for trade in recent_trades:
                # Parse timestamp and filter by cutoff
                trade_time_str = trade.get("timestamp") or trade.get("time") or trade.get("created_at")
                if trade_time_str:
                    try:
                        # Handle different timestamp formats
                        if "T" in str(trade_time_str):
                            trade_time = datetime.fromisoformat(trade_time_str.replace("Z", "+00:00").replace("+00:00", ""))
                        else:
                            trade_time = datetime.fromtimestamp(float(trade_time_str))

                        if trade_time < cutoff_time:
                            continue
                    except (ValueError, TypeError):
                        pass

                # Add wallet info to the trade
                enriched_trade = {
                    **trade,
                    "wallet_address": wallet_address,
                    "wallet_label": wallet_label,
                }
                all_trades.append(enriched_trade)

        # Sort by timestamp (most recent first)
        def get_sort_key(t):
            ts = t.get("timestamp") or t.get("time") or t.get("created_at") or ""
            if isinstance(ts, str) and ts:
                try:
                    if "T" in ts:
                        return datetime.fromisoformat(ts.replace("Z", "+00:00").replace("+00:00", ""))
                    return datetime.fromtimestamp(float(ts))
                except (ValueError, TypeError):
                    pass
            return datetime.min

        all_trades.sort(key=get_sort_key, reverse=True)

        # Limit results
        all_trades = all_trades[:limit]

        return {
            "trades": all_trades,
            "total": len(all_trades),
            "tracked_wallets": len(wallets),
            "hours_window": hours
        }
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


# ==================== TRADER DISCOVERY ====================

@router.get("/discover/leaderboard")
async def get_leaderboard(
    limit: int = Query(50, ge=1, le=50),
    time_period: str = Query("ALL", description="Time period: DAY, WEEK, MONTH, or ALL"),
    order_by: str = Query("PNL", description="Sort by: PNL (profit) or VOL (volume)"),
    category: str = Query("OVERALL", description="Category: OVERALL, POLITICS, SPORTS, CRYPTO, CULTURE, WEATHER, ECONOMICS, TECH, FINANCE")
):
    """
    Get Polymarket leaderboard - top traders by profit or volume.

    Filters:
    - time_period: DAY (24h), WEEK (7d), MONTH (30d), or ALL (all time)
    - order_by: PNL (profit/loss) or VOL (trading volume)
    - category: Filter by market category
    """
    try:
        leaderboard = await polymarket_client.get_leaderboard(
            limit=limit,
            time_period=time_period,
            order_by=order_by,
            category=category
        )
        return leaderboard
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/discover/top-traders")
async def discover_top_traders(
    limit: int = Query(50, ge=1, le=50),
    min_trades: int = Query(10, ge=1),
    time_period: str = Query("ALL", description="Time period: DAY, WEEK, MONTH, or ALL"),
    order_by: str = Query("PNL", description="Sort by: PNL or VOL"),
    category: str = Query("OVERALL", description="Market category filter")
):
    """
    Discover top traders by analyzing recent trade activity.
    Returns wallets sorted by trading volume or profit.
    """
    try:
        traders = await polymarket_client.get_top_traders_from_trades(
            limit=limit,
            min_trades=min_trades,
            time_period=time_period,
            order_by=order_by,
            category=category
        )
        return traders
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/discover/by-win-rate")
async def discover_by_win_rate(
    min_win_rate: float = Query(70.0, ge=0, le=100, description="Minimum win rate percentage (0-100)"),
    min_trades: int = Query(10, ge=1, description="Minimum number of trades"),
    limit: int = Query(50, ge=1, le=200, description="Max results to return"),
    time_period: str = Query("ALL", description="Time period: DAY, WEEK, MONTH, or ALL"),
    category: str = Query("OVERALL", description="Market category filter"),
    min_volume: float = Query(0, ge=0, description="Minimum trading volume (0 = no minimum)"),
    max_volume: float = Query(0, ge=0, description="Maximum trading volume (0 = no maximum)"),
    scan_count: int = Query(100, ge=10, le=500, description="Number of traders to scan from leaderboard")
):
    """
    Discover traders with high win rates.

    This endpoint fetches traders from the leaderboard, calculates their actual
    win rate by analyzing trade history, and returns only those meeting the threshold.

    Filters:
    - min_win_rate: Filter by minimum win rate (e.g., 99 for 99%+ win rate)
    - min_trades: Minimum closed trades to qualify
    - min_volume/max_volume: Filter by trading volume
    - scan_count: How many traders to analyze (more = slower but finds more results)

    Note: Higher scan_count values will take longer but may find more high win-rate traders.
    """
    try:
        traders = await polymarket_client.discover_by_win_rate(
            min_win_rate=min_win_rate,
            min_trades=min_trades,
            limit=limit,
            time_period=time_period,
            category=category,
            min_volume=min_volume,
            max_volume=max_volume,
            scan_count=scan_count
        )
        return traders
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/discover/wallet/{address}/win-rate")
async def get_wallet_win_rate(address: str):
    """
    Calculate win rate for a specific wallet.
    Analyzes trade history to determine wins vs losses.
    """
    try:
        win_rate_data = await polymarket_client.calculate_wallet_win_rate(address)
        return win_rate_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/discover/wallet/{address}")
async def analyze_wallet_pnl(address: str):
    """
    Analyze a wallet's profit and loss, trade history, and patterns.
    """
    try:
        pnl = await polymarket_client.get_wallet_pnl(address)
        return pnl
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/discover/analyze-and-track")
async def analyze_and_track_wallet(
    address: str,
    label: Optional[str] = None,
    auto_copy: bool = False,
    simulation_account_id: Optional[str] = None
):
    """
    Analyze a wallet and optionally add it for tracking/copy trading.

    - Fetches wallet's PnL and trade history
    - Adds to tracked wallets
    - Optionally sets up copy trading in paper mode
    """
    try:
        # Analyze the wallet
        pnl = await polymarket_client.get_wallet_pnl(address)

        # Add to tracking
        wallet_label = label or f"Discovered ({pnl.get('roi_percent', 0):.1f}% ROI)"
        await wallet_tracker.add_wallet(address, wallet_label)

        result = {
            "status": "success",
            "wallet": address,
            "label": wallet_label,
            "analysis": pnl,
            "tracking": True,
            "copy_trading": False
        }

        # Optionally set up copy trading
        if auto_copy and simulation_account_id:
            from services.copy_trader import copy_trader
            from services.simulation import simulation_service

            # Verify account exists
            account = await simulation_service.get_account(simulation_account_id)
            if account:
                await copy_trader.add_copy_config(
                    source_wallet=address,
                    account_id=simulation_account_id,
                    min_roi_threshold=2.0,
                    max_position_size=100.0
                )
                result["copy_trading"] = True
                result["copy_account_id"] = simulation_account_id

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
