import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from typing import Optional
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy import select
from models import ArbitrageOpportunity, StrategyType, OpportunityFilter
from models.database import AsyncSessionLocal, StrategyPlugin, get_db_session
from services import scanner, wallet_tracker, polymarket_client
from services.wallet_discovery import wallet_discovery
from services.kalshi_client import kalshi_client
from services.plugin_loader import plugin_loader
from services import shared_state
from utils.logger import get_logger

router = APIRouter()
logger = get_logger("routes")

DISCOVER_TIME_TO_WINDOW = {
    "DAY": "1d",
    "WEEK": "7d",
    "MONTH": "30d",
    "ALL": None,
}


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
    session: AsyncSession = Depends(get_db_session),
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
    exclude_strategy: Optional[str] = Query(
        None,
        description="Exclude a strategy type from results (e.g. btc_eth_highfreq)",
    ),
    limit: int = Query(50, description="Maximum results to return"),
    offset: int = Query(0, description="Number of results to skip"),
):
    """Get current arbitrage opportunities (from DB snapshot)."""
    strategies = await _resolve_strategy_to_filter(strategy)
    filter = OpportunityFilter(
        min_profit=min_profit / 100,  # Convert from percentage
        max_risk=max_risk,
        strategies=strategies,
        min_liquidity=min_liquidity,
        category=category,
    )

    opportunities = await shared_state.get_opportunities_from_db(session, filter)

    # Exclude a specific strategy if requested
    if exclude_strategy:
        opportunities = [
            opp for opp in opportunities if opp.strategy != exclude_strategy
        ]

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

        # Relevance filter: only keep markets where the query actually
        # appears in the question text.  The Gamma API _q search and
        # slug_contains both return markets from matching *events* (e.g.
        # searching "trump" returns every market in a Trump-tagged event,
        # including GTA VI markets).
        q_lower = q.lower()
        q_words = q_lower.split()
        all_markets = [
            m for m in all_markets
            if any(w in m.question.lower() for w in q_words)
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
            event_slug = getattr(market, "event_slug", "") or ""
            category = ""
            volume = float(market.volume or 0)
            liquidity = float(getattr(market, "liquidity", 0) or market.volume or 0)

            results.append({
                "id": f"search-{mid}",
                "stable_id": f"search-{mid}",
                "title": market.question,
                "description": f"{platform.title()} market — Yes {yes_price:.0%} / No {no_price:.0%}",
                "event_title": market.question,
                "event_slug": event_slug,
                "strategy": "search",
                "total_cost": 0.0,
                "expected_payout": 0.0,
                "gross_profit": 0.0,
                "fee": 0.0,
                "net_profit": 0.0,
                "roi_percent": 0.0,
                "risk_score": 0.0,
                "risk_factors": [],
                "min_liquidity": liquidity,
                "volume": volume,
                "max_position_size": 0.0,
                "category": category,
                "detected_at": datetime.now(timezone.utc).isoformat(),
                "expires_at": market.end_date.isoformat() if market.end_date else None,
                "resolution_date": market.end_date.isoformat() if market.end_date else None,
                "platform": platform,
                "positions_to_take": [],
                "markets": [
                    {
                        "id": market.condition_id or "",
                        "question": market.question,
                        "slug": slug,
                        "event_slug": event_slug,
                        "platform": platform,
                        "yes_price": yes_price,
                        "no_price": no_price,
                        "volume": volume,
                        "liquidity": liquidity,
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


@router.post("/opportunities/search-polymarket/evaluate")
async def evaluate_search_markets(
    body: dict = {},
    session: AsyncSession = Depends(get_db_session),
):
    """
    Trigger strategy evaluation on search result markets.

    Requests a scan from the scanner worker (writes to DB); results
    will appear once the worker runs and updates the snapshot.
    """
    condition_ids = body.get("condition_ids", [])

    await shared_state.request_one_scan(session)

    return {
        "status": "evaluating",
        "count": len(condition_ids),
        "message": "Strategy scan requested. Detected opportunities will appear in the Markets tab.",
    }


@router.get("/opportunities/counts")
async def get_opportunity_counts(
    session: AsyncSession = Depends(get_db_session),
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

    opportunities = await shared_state.get_opportunities_from_db(session, filter)

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
async def get_opportunity(
    opportunity_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    """Get a specific opportunity by ID"""
    opportunities = await shared_state.get_opportunities_from_db(session, None)
    for opp in opportunities:
        if opp.id == opportunity_id:
            return opp
    raise HTTPException(status_code=404, detail="Opportunity not found")


@router.post("/scan")
async def trigger_scan(session: AsyncSession = Depends(get_db_session)):
    """Manually request one scan. Scanner worker will run a scan on its next loop."""
    await shared_state.request_one_scan(session)
    return {
        "status": "started",
        "timestamp": utcnow().isoformat(),
        "message": "Scan requested; scanner worker will run on next cycle.",
    }


# ==================== SCANNER STATUS ====================


@router.get("/scanner/status")
async def get_scanner_status(session: AsyncSession = Depends(get_db_session)):
    """Get scanner status (from DB snapshot)."""
    return await shared_state.get_scanner_status_from_db(session)


@router.post("/scanner/start")
async def start_scanner(session: AsyncSession = Depends(get_db_session)):
    """Start/resume the scanner (worker reads control from DB)."""
    await shared_state.set_scanner_paused(session, False)
    return {"status": "started", **await shared_state.get_scanner_status_from_db(session)}


@router.post("/scanner/pause")
async def pause_scanner(session: AsyncSession = Depends(get_db_session)):
    """Pause the scanner (worker skips scans when paused)."""
    await shared_state.set_scanner_paused(session, True)
    return {"status": "paused", **await shared_state.get_scanner_status_from_db(session)}


@router.post("/scanner/interval")
async def set_scanner_interval(
    session: AsyncSession = Depends(get_db_session),
    interval_seconds: int = Query(..., ge=10, le=3600),
):
    """Set the scan interval (10-3600 seconds)."""
    await shared_state.set_scanner_interval(session, interval_seconds)
    return {"status": "updated", **await shared_state.get_scanner_status_from_db(session)}


# ==================== OPPORTUNITIES CLEANUP ====================


@router.delete("/opportunities")
async def clear_opportunities(session: AsyncSession = Depends(get_db_session)):
    """
    Clear all opportunities in the snapshot. Repopulated on next scanner run.
    """
    count = await shared_state.clear_opportunities_in_snapshot(session)
    return {
        "status": "success",
        "cleared_count": count,
        "message": f"Cleared {count} opportunities. Next scan will repopulate.",
    }


@router.post("/opportunities/cleanup")
async def cleanup_opportunities(
    session: AsyncSession = Depends(get_db_session),
    remove_expired: bool = Query(
        True, description="Remove opportunities past resolution date"
    ),
    max_age_minutes: Optional[int] = Query(
        None, ge=1, le=1440, description="Remove opportunities older than X minutes"
    ),
):
    """
    Clean up stale opportunities in the snapshot.
    """
    results = await shared_state.cleanup_snapshot_opportunities(
        session, remove_expired=remove_expired, max_age_minutes=max_age_minutes
    )
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
from utils.utcnow import utcnow

        wallets = await wallet_tracker.get_all_wallets()
        all_trades = []

        cutoff_time = utcnow() - timedelta(hours=hours)

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

    synthetic = [
        {
            "type": StrategyType.NEWS_EDGE.value,
            "name": "News Edge",
            "description": "News workflow informational edge intents consumed by auto trader",
            "is_plugin": False,
        },
        {
            "type": StrategyType.WEATHER_EDGE.value,
            "name": "Weather Edge",
            "description": "Weather workflow forecast-consensus intents consumed by auto trader",
            "is_plugin": False,
        },
    ]

    return builtin + synthetic + plugin_entries


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
    """Legacy compatibility adapter backed by /api/discovery data."""
    try:
        time_period = time_period.upper()
        order_by = order_by.upper()
        window_key = DISCOVER_TIME_TO_WINDOW.get(time_period)
        if time_period not in DISCOVER_TIME_TO_WINDOW:
            raise HTTPException(
                status_code=400, detail="Invalid time_period. Use DAY/WEEK/MONTH/ALL"
            )

        sort_by = "total_pnl" if order_by == "PNL" else "total_returned"
        data = await wallet_discovery.get_leaderboard(
            limit=limit,
            offset=0,
            min_trades=0,
            min_pnl=0.0,
            sort_by=sort_by,
            sort_dir="desc",
            window_key=window_key,
            active_within_hours=24 if time_period == "DAY" else None,
        )

        wallets = data.get("wallets", [])
        # Legacy response shape from Polymarket leaderboard.
        return [
            {
                "proxyWallet": w.get("address"),
                "userName": w.get("username"),
                "pnl": w.get("period_pnl", w.get("total_pnl", 0.0)),
                "vol": w.get("total_returned", 0.0),
                "rank": w.get("rank_position"),
                "winRate": (w.get("period_win_rate", w.get("win_rate", 0.0)) or 0.0)
                * 100.0,
                "deprecated": True,
                "deprecation_note": "/api/discover/* now proxies /api/discovery/*",
            }
            for w in wallets
        ]
    except HTTPException:
        raise
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
    """Legacy compatibility adapter backed by /api/discovery leaderboard."""
    try:
        time_period = time_period.upper()
        order_by = order_by.upper()
        window_key = DISCOVER_TIME_TO_WINDOW.get(time_period)
        if time_period not in DISCOVER_TIME_TO_WINDOW:
            raise HTTPException(
                status_code=400, detail="Invalid time_period. Use DAY/WEEK/MONTH/ALL"
            )

        sort_by = "total_pnl" if order_by == "PNL" else "total_returned"
        data = await wallet_discovery.get_leaderboard(
            limit=limit,
            offset=0,
            min_trades=min_trades,
            min_pnl=0.0,
            sort_by=sort_by,
            sort_dir="desc",
            window_key=window_key,
        )
        wallets = data.get("wallets", [])

        return [
            {
                "address": w.get("address"),
                "username": w.get("username"),
                "trades": w.get("period_trades", w.get("total_trades", 0)),
                "volume": w.get("total_returned", 0.0),
                "pnl": w.get("period_pnl", w.get("total_pnl", 0.0)),
                "rank": w.get("rank_position"),
                "buys": w.get("wins", 0),
                "sells": w.get("losses", 0),
                "win_rate": (w.get("period_win_rate", w.get("win_rate", 0.0)) or 0.0)
                * 100.0,
                "wins": w.get("wins", 0),
                "losses": w.get("losses", 0),
                "total_markets": w.get("unique_markets", 0),
                "trade_count": w.get("period_trades", w.get("total_trades", 0)),
                "deprecated": True,
            }
            for w in wallets
        ]
    except HTTPException:
        raise
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
    """Legacy compatibility adapter backed by /api/discovery leaderboard."""
    try:
        window_key = DISCOVER_TIME_TO_WINDOW.get(time_period.upper())
        if time_period.upper() not in DISCOVER_TIME_TO_WINDOW:
            raise HTTPException(
                status_code=400, detail="Invalid time_period. Use DAY/WEEK/MONTH/ALL"
            )

        # We over-fetch so win-rate/volume filters still return enough rows.
        data = await wallet_discovery.get_leaderboard(
            limit=min(max(scan_count, limit * 4), 500),
            offset=0,
            min_trades=min_trades,
            min_pnl=0.0,
            sort_by="win_rate",
            sort_dir="desc",
            window_key=window_key,
        )
        wallets = data.get("wallets", [])

        output = []
        for w in wallets:
            wr = (w.get("period_win_rate", w.get("win_rate", 0.0)) or 0.0) * 100.0
            volume = float(w.get("total_returned", 0.0) or 0.0)
            if wr < min_win_rate:
                continue
            if min_volume > 0 and volume < min_volume:
                continue
            if max_volume > 0 and volume > max_volume:
                continue
            output.append(
                {
                    "address": w.get("address"),
                    "username": w.get("username"),
                    "volume": volume,
                    "pnl": w.get("period_pnl", w.get("total_pnl", 0.0)),
                    "rank": w.get("rank_position"),
                    "win_rate": wr,
                    "wins": w.get("wins", 0),
                    "losses": w.get("losses", 0),
                    "total_markets": w.get("unique_markets", 0),
                    "trade_count": w.get("period_trades", w.get("total_trades", 0)),
                    "deprecated": True,
                }
            )
            if len(output) >= limit:
                break
        return output
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/discover/wallet/{address}/win-rate")
async def get_wallet_win_rate(
    address: str,
    time_period: str = Query(
        "ALL", description="Time period: DAY, WEEK, MONTH, or ALL"
    ),
):
    """Legacy compatibility adapter backed by discovered wallet profile."""
    try:
        profile = await wallet_discovery.get_wallet_profile(address.lower())
        if profile:
            key = DISCOVER_TIME_TO_WINDOW.get(time_period.upper())
            if key:
                win_rate = (profile.get("rolling_win_rate", {}) or {}).get(key, 0.0)
                trade_count = (profile.get("rolling_trade_count", {}) or {}).get(key, 0)
            else:
                win_rate = profile.get("win_rate", 0.0)
                trade_count = profile.get("total_trades", 0)

            wins = profile.get("wins", 0)
            losses = profile.get("losses", 0)
            if key and trade_count > 0 and wins + losses > 0:
                # Approximate period wins/losses from period win rate and trade count.
                wins = int(round((win_rate or 0.0) * trade_count))
                losses = max(int(trade_count) - wins, 0)

            return {
                "address": address,
                "win_rate": (win_rate or 0.0) * 100.0,
                "wins": wins,
                "losses": losses,
                "total_markets": profile.get("unique_markets", 0),
                "trade_count": trade_count,
                "deprecated": True,
            }

        # Fallback for unknown wallets.
        fast_result = await polymarket_client.calculate_win_rate_fast(address, min_positions=1)
        if fast_result:
            return {
                "address": address,
                "win_rate": fast_result["win_rate"],
                "wins": fast_result["wins"],
                "losses": fast_result["losses"],
                "total_markets": fast_result["closed_positions"],
                "trade_count": fast_result["closed_positions"],
                "deprecated": True,
            }
        return {
            "address": address,
            "win_rate": 0.0,
            "wins": 0,
            "losses": 0,
            "total_markets": 0,
            "trade_count": 0,
            "deprecated": True,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/discover/wallet/{address}")
async def analyze_wallet_pnl(
    address: str,
    time_period: str = Query(
        "ALL", description="Time period: DAY, WEEK, MONTH, or ALL"
    ),
):
    """Legacy compatibility adapter backed by discovered wallet profile."""
    try:
        profile = await wallet_discovery.get_wallet_profile(address.lower())
        if profile:
            key = DISCOVER_TIME_TO_WINDOW.get(time_period.upper())
            if key:
                pnl = (profile.get("rolling_pnl", {}) or {}).get(key, 0.0)
                trade_count = (profile.get("rolling_trade_count", {}) or {}).get(key, 0)
            else:
                pnl = profile.get("total_pnl", 0.0)
                trade_count = profile.get("total_trades", 0)

            total_invested = float(profile.get("total_invested", 0.0) or 0.0)
            roi_percent = (pnl / total_invested * 100.0) if total_invested > 0 else 0.0
            return {
                "address": address,
                "total_trades": trade_count,
                "open_positions": profile.get("open_positions", 0),
                "total_invested": total_invested,
                "total_returned": profile.get("total_returned", 0.0),
                "position_value": 0.0,
                "realized_pnl": profile.get("realized_pnl", 0.0),
                "unrealized_pnl": profile.get("unrealized_pnl", 0.0),
                "total_pnl": pnl,
                "roi_percent": roi_percent,
                "deprecated": True,
            }

        # Fallback for non-discovered wallets.
        pnl = await polymarket_client.get_wallet_pnl(address, time_period=time_period)
        pnl["deprecated"] = True
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
        profile = await wallet_discovery.get_wallet_profile(address.lower())
        if profile is None:
            analysis = await wallet_discovery.analyze_wallet(address.lower())
            if analysis is not None:
                await wallet_discovery._upsert_wallet(analysis)
                await wallet_discovery.refresh_leaderboard()
                profile = await wallet_discovery.get_wallet_profile(address.lower())

        analysis_payload = (
            {
                "address": address,
                "total_trades": profile.get("total_trades", 0),
                "open_positions": profile.get("open_positions", 0),
                "total_invested": profile.get("total_invested", 0.0),
                "total_returned": profile.get("total_returned", 0.0),
                "position_value": 0.0,
                "realized_pnl": profile.get("realized_pnl", 0.0),
                "unrealized_pnl": profile.get("unrealized_pnl", 0.0),
                "total_pnl": profile.get("total_pnl", 0.0),
                "roi_percent": (
                    (profile.get("total_pnl", 0.0) / profile.get("total_invested", 1.0))
                    * 100.0
                    if (profile.get("total_invested", 0.0) or 0.0) > 0
                    else 0.0
                ),
            }
            if profile
            else await polymarket_client.get_wallet_pnl(address)
        )

        # Add to tracking
        wallet_label = label or f"Discovered ({analysis_payload.get('roi_percent', 0):.1f}% ROI)"
        await wallet_tracker.add_wallet(address, wallet_label)

        result = {
            "status": "success",
            "wallet": address,
            "label": wallet_label,
            "analysis": analysis_payload,
            "tracking": True,
            "copy_trading": False,
            "deprecated": True,
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
