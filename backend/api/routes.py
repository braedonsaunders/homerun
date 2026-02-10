import asyncio

from fastapi import APIRouter, HTTPException, Query, Response
from typing import Optional
from datetime import datetime, timezone

from sqlalchemy import select
from models import ArbitrageOpportunity, StrategyType, OpportunityFilter
from models.database import AsyncSessionLocal, StrategyPlugin
from services import scanner, wallet_tracker, polymarket_client
from services.kalshi_client import kalshi_client
from services.plugin_loader import plugin_loader

router = APIRouter()


async def _resolve_strategy_to_filter(strategy_param: Optional[str]) -> list[str]:
    """Resolve strategy param to list of strategy type strings.

    Accepts:
        - Built-in strategy type: "basic", "negrisk", etc.
        - Plugin slug: "plugin_<slug>" -> resolves to [slug]
    """
    if not strategy_param:
        return []
    strategy_param = strategy_param.strip().lower()

    # Plugin strategy: "plugin_<slug>"
    if strategy_param.startswith("plugin_"):
        slug = strategy_param[7:]  # len("plugin_")
        # Verify plugin exists and is enabled
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(StrategyPlugin).where(
                    StrategyPlugin.slug == slug, StrategyPlugin.enabled == True
                )
            )
            plugin = result.scalar_one_or_none()
        if plugin:
            return [slug]
        return []

    # Built-in strategy type
    try:
        return [StrategyType(strategy_param).value]
    except ValueError:
        # Could be a plugin slug used directly (without prefix)
        if plugin_loader.get_plugin(strategy_param):
            return [strategy_param]
        return []


# ==================== OPPORTUNITIES ====================


@router.get("/opportunities")
async def get_opportunities(
    response: Response,
    min_profit: float = Query(0.0, description="Minimum profit percentage"),
    max_risk: float = Query(1.0, description="Maximum risk score (0-1)"),
    strategy: Optional[str] = Query(
        None,
        description="Filter by strategy type (e.g. basic, negrisk) or plugin (plugin_<id>)",
    ),
    min_liquidity: float = Query(0.0, description="Minimum liquidity in USD"),
    search: Optional[str] = Query(None, description="Search query for market titles"),
    category: Optional[str] = Query(
        None, description="Filter by category (e.g., politics, sports, crypto)"
    ),
    sort_by: Optional[str] = Query(
        None,
        description="Sort field: ai_score (default), roi, profit, liquidity, risk",
    ),
    sort_dir: Optional[str] = Query("desc", description="Sort direction: asc or desc"),
    limit: int = Query(50, description="Maximum results to return"),
    offset: int = Query(0, description="Number of results to skip"),
):
    """Get current arbitrage opportunities"""
    strategies = await _resolve_strategy_to_filter(strategy)
    filter = OpportunityFilter(
        min_profit=min_profit / 100,  # Convert from percentage
        max_risk=max_risk,
        strategies=strategies,
        min_liquidity=min_liquidity,
        category=category,
    )

    opportunities = scanner.get_opportunities(filter)

    # Apply search filter if provided
    if search:
        search_lower = search.lower()
        opportunities = [
            opp
            for opp in opportunities
            if search_lower in opp.title.lower()
            or (opp.event_title and search_lower in opp.event_title.lower())
            or any(search_lower in m.get("question", "").lower() for m in opp.markets)
        ]

    # Server-side rejection: filter out STRONG SKIP opportunities
    # These have been analyzed by AI and determined to be trash
    # (false positives, guaranteed losses, etc.)
    opportunities = [
        opp
        for opp in opportunities
        if not (
            opp.ai_analysis is not None
            and opp.ai_analysis.recommendation == "strong_skip"
        )
    ]

    # Sort opportunities — uses inline ai_analysis (no DB queries needed)
    reverse = sort_dir != "asc"

    # Default to ai_score sorting (LLM buy rating) when no sort specified
    effective_sort = sort_by or "ai_score"

    if effective_sort == "ai_score":
        # Scored opportunities first, sorted by overall_score, unscored last (by ROI)
        opportunities.sort(
            key=lambda o: (
                o.ai_analysis is not None and o.ai_analysis.recommendation != "pending",
                o.ai_analysis.overall_score if o.ai_analysis else 0.0,
                o.roi_percent,
            ),
            reverse=reverse,
        )
    elif effective_sort == "profit":
        opportunities.sort(key=lambda o: o.net_profit, reverse=reverse)
    elif effective_sort == "liquidity":
        opportunities.sort(key=lambda o: o.min_liquidity, reverse=reverse)
    elif effective_sort == "risk":
        opportunities.sort(key=lambda o: o.risk_score, reverse=reverse)
    elif effective_sort == "roi":
        # ROI sort, but deprioritize AI skip recommendations
        opportunities.sort(
            key=lambda o: (
                o.ai_analysis is not None
                and o.ai_analysis.recommendation in ("skip", "strong_skip"),
                -o.roi_percent if reverse else o.roi_percent,
            ),
        )

    # Apply pagination
    total = len(opportunities)
    paginated = opportunities[offset : offset + limit]

    # Set total count header and return serialised list directly.
    # Using Response injection (not JSONResponse) lets FastAPI handle
    # content-negotiation and CORS headers correctly.
    response.headers["X-Total-Count"] = str(total)
    return [o.model_dump(mode="json") for o in paginated]


@router.get("/opportunities/search-polymarket")
async def search_polymarket_opportunities(
    q: str = Query(
        ..., min_length=1, description="Search query for Polymarket and Kalshi markets"
    ),
    limit: int = Query(50, ge=1, le=100, description="Maximum results to return"),
):
    """
    Fast keyword search across Polymarket and Kalshi.

    Returns matching markets as lightweight opportunity-shaped objects
    so the frontend can display them immediately.  Skips the expensive
    price-fetching and full arbitrage-detection pipeline to keep the
    response time under a few seconds.
    """
    import asyncio

    try:
        # Search Polymarket markets directly (fastest) and Kalshi concurrently
        poly_task = polymarket_client.search_markets(q, limit=limit)
        kalshi_task = kalshi_client.search_events(q, limit=limit)
        poly_markets, kalshi_events = await asyncio.gather(
            poly_task, kalshi_task, return_exceptions=True
        )

        # Handle errors gracefully
        if isinstance(poly_markets, BaseException):
            poly_markets = []
        if isinstance(kalshi_events, BaseException):
            kalshi_events = []

        # Collect Kalshi markets from events
        kalshi_markets = []
        for event in (kalshi_events or []):
            kalshi_markets.extend(event.markets)

        all_markets = list(poly_markets) + kalshi_markets

        # Filter out expired markets
        now = datetime.now(timezone.utc)
        all_markets = [
            m for m in all_markets
            if m.end_date is None or m.end_date > now
        ]

        if not all_markets:
            from fastapi.responses import JSONResponse

            response = JSONResponse(content=[])
            response.headers["X-Total-Count"] = "0"
            return response

        # Build lightweight opportunity objects from matched markets
        # so the frontend can render them with the existing UI.
        results: list[dict] = []
        seen: set[str] = set()
        for market in all_markets:
            mid = market.condition_id or market.question[:80]
            if mid in seen:
                continue
            seen.add(mid)

            platform = getattr(market, "platform", "polymarket") or "polymarket"

            # Best-effort price from the market's own outcome data
            yes_price = market.yes_price
            no_price = market.no_price

            slug = market.slug or ""
            category = ""

            results.append({
                "id": f"search-{mid}",
                "stable_id": f"search-{mid}",
                "title": market.question,
                "description": f"{platform.title()} market — Yes {yes_price:.0%} / No {no_price:.0%}",
                "event_title": market.question,
                "strategy": "search",
                "total_cost": 0.0,
                "expected_payout": 0.0,
                "gross_profit": 0.0,
                "fee": 0.0,
                "net_profit": 0.0,
                "roi_percent": 0.0,
                "risk_score": 0.0,
                "risk_factors": [],
                "min_liquidity": float(market.volume or 0),
                "max_position_size": 0.0,
                "category": category,
                "detected_at": datetime.utcnow().isoformat(),
                "expires_at": market.end_date.isoformat() if market.end_date else None,
                "resolution_date": market.end_date.isoformat() if market.end_date else None,
                "platform": platform,
                "positions_to_take": [],
                "markets": [
                    {
                        "id": market.condition_id or "",
                        "question": market.question,
                        "slug": slug,
                        "event_slug": "",
                        "platform": platform,
                        "yes_price": yes_price,
                        "no_price": no_price,
                        "volume": float(market.volume or 0),
                    }
                ],
                "ai_analysis": None,
            })

        results = results[:limit]

        from fastapi.responses import JSONResponse

        response = JSONResponse(content=results)
        response.headers["X-Total-Count"] = str(len(results))
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/opportunities/counts")
async def get_opportunity_counts(
    min_profit: float = Query(0.0, description="Minimum profit percentage"),
    max_risk: float = Query(1.0, description="Maximum risk score (0-1)"),
    min_liquidity: float = Query(0.0, description="Minimum liquidity in USD"),
    search: Optional[str] = Query(None, description="Search query for market titles"),
):
    """Get counts of opportunities grouped by strategy and category.

    Applies base filters (profit, risk, liquidity, search) but NOT strategy/category
    filters, so the UI can show how many opportunities each filter value would match.
    """
    filter = OpportunityFilter(
        min_profit=min_profit / 100,
        max_risk=max_risk,
        min_liquidity=min_liquidity,
    )

    opportunities = scanner.get_opportunities(filter)

    # Apply search filter if provided
    if search:
        search_lower = search.lower()
        opportunities = [
            opp
            for opp in opportunities
            if search_lower in opp.title.lower()
            or (opp.event_title and search_lower in opp.event_title.lower())
            or any(search_lower in m.get("question", "").lower() for m in opp.markets)
        ]

    # Count by strategy
    strategy_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    for opp in opportunities:
        s = opp.strategy
        strategy_counts[s] = strategy_counts.get(s, 0) + 1
        if opp.category:
            cat = opp.category.lower()
            category_counts[cat] = category_counts.get(cat, 0) + 1

    return {"strategies": strategy_counts, "categories": category_counts}


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
    """Manually trigger a new scan (non-blocking).

    Kicks off scan_once() as a background task and returns immediately
    so the UI stays responsive.  The frontend already polls
    /scanner/status and /opportunities on intervals, so results
    appear automatically once the scan completes.
    """
    asyncio.create_task(scanner.scan_once())
    return {
        "status": "started",
        "timestamp": datetime.utcnow().isoformat(),
        "message": "Scan started in background",
    }


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
        "message": f"Cleared {count} opportunities. Next scan will repopulate.",
    }


@router.post("/opportunities/cleanup")
async def cleanup_opportunities(
    remove_expired: bool = Query(
        True, description="Remove opportunities past resolution date"
    ),
    max_age_minutes: Optional[int] = Query(
        None, ge=1, le=1440, description="Remove opportunities older than X minutes"
    ),
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

    return {"status": "success", **results}


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
async def get_wallet_positions(address: str, include_prices: bool = True):
    """Get current positions for a wallet with optional current market prices"""
    try:
        if include_prices:
            positions = await polymarket_client.get_wallet_positions_with_prices(
                address
            )
        else:
            positions = await polymarket_client.get_wallet_positions(address)
        return {"address": address, "positions": positions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/wallets/{address}/profile")
async def get_wallet_profile(address: str):
    """Get profile information for a wallet (username, etc.)"""
    try:
        profile = await polymarket_client.get_user_profile(address)
        return profile
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
    limit: int = Query(100, ge=1, le=500, description="Maximum trades to return"),
    hours: int = Query(
        24, ge=1, le=168, description="Only show trades from last N hours"
    ),
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
            wallet_username = wallet.get("username") or ""
            recent_trades = wallet.get("recent_trades", [])

            for trade in recent_trades:
                # Parse timestamp - check multiple field names
                trade_time_str = (
                    trade.get("match_time")
                    or trade.get("timestamp")
                    or trade.get("time")
                    or trade.get("created_at")
                    or trade.get("createdAt")
                )
                if trade_time_str:
                    try:
                        if isinstance(trade_time_str, (int, float)):
                            trade_time = datetime.fromtimestamp(trade_time_str)
                        elif "T" in str(trade_time_str) or "-" in str(trade_time_str):
                            trade_time = datetime.fromisoformat(
                                str(trade_time_str)
                                .replace("Z", "+00:00")
                                .replace("+00:00", "")
                            )
                        else:
                            trade_time = datetime.fromtimestamp(float(trade_time_str))

                        if trade_time < cutoff_time:
                            continue
                    except (ValueError, TypeError, OSError):
                        pass

                # Add wallet info to the trade
                enriched_trade = {
                    **trade,
                    "wallet_address": wallet_address,
                    "wallet_label": wallet_label,
                    "wallet_username": wallet_username,
                }
                all_trades.append(enriched_trade)

        # Enrich trades with market names and normalized timestamps
        from services.polymarket import polymarket_client

        all_trades = await polymarket_client.enrich_trades_with_market_info(all_trades)

        # Sort by normalized timestamp (most recent first)
        def get_sort_key(t):
            ts = t.get("timestamp_iso", "")
            if ts:
                try:
                    return datetime.fromisoformat(
                        ts.replace("Z", "+00:00").replace("+00:00", "")
                    )
                except (ValueError, TypeError):
                    pass
            # Fallback to raw timestamp fields
            raw = (
                t.get("match_time")
                or t.get("timestamp")
                or t.get("time")
                or t.get("created_at")
                or ""
            )
            if isinstance(raw, str) and raw:
                try:
                    if "T" in raw or "-" in raw:
                        return datetime.fromisoformat(
                            raw.replace("Z", "+00:00").replace("+00:00", "")
                        )
                    return datetime.fromtimestamp(float(raw))
                except (ValueError, TypeError, OSError):
                    pass
            elif isinstance(raw, (int, float)):
                try:
                    return datetime.fromtimestamp(raw)
                except (ValueError, OSError):
                    pass
            return datetime.min

        all_trades.sort(key=get_sort_key, reverse=True)

        # Limit results
        all_trades = all_trades[:limit]

        return {
            "trades": all_trades,
            "total": len(all_trades),
            "tracked_wallets": len(wallets),
            "hours_window": hours,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== MARKETS ====================


@router.get("/markets")
async def get_markets(active: bool = True, limit: int = 100, offset: int = 0):
    """Get markets from Polymarket"""
    try:
        markets = await polymarket_client.get_markets(
            active=active, limit=limit, offset=offset
        )
        return [m.model_dump() for m in markets]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events")
async def get_events(closed: bool = False, limit: int = 100, offset: int = 0):
    """Get events from Polymarket"""
    try:
        events = await polymarket_client.get_events(
            closed=closed, limit=limit, offset=offset
        )
        return [e.model_dump() for e in events]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== STRATEGY INFO ====================


@router.get("/strategies")
async def get_strategies():
    """Get information about available strategies and plugins."""
    builtin = [
        {
            "type": s.strategy_type if isinstance(s.strategy_type, str) else s.strategy_type.value,
            "name": s.name,
            "description": s.description,
            "is_plugin": False,
        }
        for s in scanner.strategies
    ]

    # Append enabled plugins as real strategies
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(StrategyPlugin)
            .where(StrategyPlugin.enabled == True)
            .order_by(StrategyPlugin.sort_order.asc(), StrategyPlugin.name.asc())
        )
        plugins = result.scalars().all()

    plugin_entries = [
        {
            "type": f"plugin_{p.slug}",
            "name": p.name,
            "description": p.description or f"Plugin strategy: {p.slug}",
            "is_plugin": True,
            "plugin_id": p.id,
            "plugin_slug": p.slug,
            "status": p.status,
        }
        for p in plugins
    ]

    return builtin + plugin_entries


# ==================== TRADER DISCOVERY ====================


@router.get("/discover/leaderboard")
async def get_leaderboard(
    limit: int = Query(50, ge=1, le=50),
    time_period: str = Query(
        "ALL", description="Time period: DAY, WEEK, MONTH, or ALL"
    ),
    order_by: str = Query("PNL", description="Sort by: PNL (profit) or VOL (volume)"),
    category: str = Query(
        "OVERALL",
        description="Category: OVERALL, POLITICS, SPORTS, CRYPTO, CULTURE, WEATHER, ECONOMICS, TECH, FINANCE",
    ),
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
            limit=limit, time_period=time_period, order_by=order_by, category=category
        )
        return leaderboard
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/discover/top-traders")
async def discover_top_traders(
    limit: int = Query(50, ge=1, le=50),
    min_trades: int = Query(10, ge=1),
    time_period: str = Query(
        "ALL", description="Time period: DAY, WEEK, MONTH, or ALL"
    ),
    order_by: str = Query("PNL", description="Sort by: PNL or VOL"),
    category: str = Query("OVERALL", description="Market category filter"),
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
            category=category,
        )
        return traders
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/discover/by-win-rate")
async def discover_by_win_rate(
    min_win_rate: float = Query(
        70.0, ge=0, le=100, description="Minimum win rate percentage (0-100)"
    ),
    min_trades: int = Query(10, ge=1, description="Minimum number of closed positions"),
    limit: int = Query(100, ge=1, le=500, description="Max results to return"),
    time_period: str = Query(
        "ALL", description="Time period: DAY, WEEK, MONTH, or ALL"
    ),
    category: str = Query("OVERALL", description="Market category filter"),
    min_volume: float = Query(
        0, ge=0, description="Minimum trading volume (0 = no minimum)"
    ),
    max_volume: float = Query(
        0, ge=0, description="Maximum trading volume (0 = no maximum)"
    ),
    scan_count: int = Query(
        200,
        ge=10,
        le=1050,
        description="Number of traders to scan per leaderboard sort (searches both PNL and VOL)",
    ),
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
            scan_count=scan_count,
        )
        return traders
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/discover/wallet/{address}/win-rate")
async def get_wallet_win_rate(
    address: str,
    time_period: str = Query(
        "ALL", description="Time period: DAY, WEEK, MONTH, or ALL"
    ),
):
    """
    Calculate win rate for a specific wallet.
    Uses closed-positions data (same method as Discover page) for consistency.
    Falls back to full trade analysis if closed-positions data is unavailable.
    """
    try:
        # Primary: Use fast method (closed-positions) for consistency with discover page
        fast_result = await polymarket_client.calculate_win_rate_fast(
            address, min_positions=1
        )
        if fast_result:
            return {
                "address": address,
                "win_rate": fast_result["win_rate"],
                "wins": fast_result["wins"],
                "losses": fast_result["losses"],
                "total_markets": fast_result["closed_positions"],
                "trade_count": fast_result["closed_positions"],
            }

        # Fallback: Full trade analysis if no closed positions found
        win_rate_data = await polymarket_client.calculate_wallet_win_rate(
            address, time_period=time_period
        )
        return win_rate_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/discover/wallet/{address}")
async def analyze_wallet_pnl(
    address: str,
    time_period: str = Query(
        "ALL", description="Time period: DAY, WEEK, MONTH, or ALL"
    ),
):
    """
    Analyze a wallet's profit and loss, trade history, and patterns.
    """
    try:
        pnl = await polymarket_client.get_wallet_pnl(address, time_period=time_period)
        return pnl
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/discover/analyze-and-track")
async def analyze_and_track_wallet(
    address: str,
    label: Optional[str] = None,
    auto_copy: bool = False,
    simulation_account_id: Optional[str] = None,
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
            "copy_trading": False,
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
                    max_position_size=100.0,
                )
                result["copy_trading"] = True
                result["copy_account_id"] = simulation_account_id

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
