import asyncio

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from services.wallet_discovery import wallet_discovery
from services.wallet_intelligence import wallet_intelligence
from utils.validation import validate_eth_address

discovery_router = APIRouter()


# ==================== LEADERBOARD ====================


@discovery_router.get("/leaderboard")
async def get_leaderboard(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    min_trades: int = Query(default=0, ge=0),
    min_pnl: float = Query(default=0.0),
    sort_by: str = Query(
        default="rank_score",
        description="rank_score, total_pnl, win_rate, sharpe_ratio, profit_factor",
    ),
    sort_dir: str = Query(default="desc", description="asc or desc"),
    tags: Optional[str] = Query(default=None, description="Comma-separated tag filter"),
    recommendation: Optional[str] = Query(
        default=None, description="copy_candidate, monitor, avoid"
    ),
):
    """
    Get the wallet leaderboard with comprehensive filters and sorting.

    Returns ranked wallets with trading stats, tags, and recommendations.
    Supports pagination, multi-field sorting, and tag-based filtering.
    """
    try:
        # Validate sort_by
        valid_sort_fields = [
            "rank_score",
            "total_pnl",
            "win_rate",
            "sharpe_ratio",
            "profit_factor",
            "total_trades",
            "avg_roi",
            "sortino_ratio",
            "trades_per_day",
        ]
        if sort_by not in valid_sort_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid sort_by. Must be one of: {valid_sort_fields}",
            )

        # Validate sort_dir
        if sort_dir not in ("asc", "desc"):
            raise HTTPException(
                status_code=400,
                detail="Invalid sort_dir. Must be 'asc' or 'desc'",
            )

        # Validate recommendation
        valid_recommendations = ["copy_candidate", "monitor", "avoid"]
        if recommendation and recommendation not in valid_recommendations:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid recommendation. Must be one of: {valid_recommendations}",
            )

        # Parse comma-separated tags
        tag_list = tags.split(",") if tags else None

        result = await wallet_discovery.get_leaderboard(
            limit=limit,
            offset=offset,
            min_trades=min_trades,
            min_pnl=min_pnl,
            sort_by=sort_by,
            sort_dir=sort_dir,
            tags=tag_list,
            recommendation=recommendation,
        )

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.get("/leaderboard/stats")
async def get_discovery_stats():
    """
    Get discovery engine statistics.

    Returns total wallets analyzed, last run timestamp, coverage metrics,
    and engine health information.
    """
    try:
        stats = await wallet_discovery.get_discovery_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== WALLET PROFILES ====================


@discovery_router.get("/wallet/{wallet_address}/profile")
async def get_wallet_profile(wallet_address: str):
    """
    Get comprehensive wallet profile with all metrics, tags, cluster info, and rolling windows.

    Returns detailed analysis including trading statistics, detected strategies,
    tag classifications, entity cluster membership, and performance over
    multiple time windows.
    """
    try:
        address = validate_eth_address(wallet_address)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        profile = await wallet_discovery.get_wallet_profile(address)
        if not profile:
            raise HTTPException(status_code=404, detail="Wallet profile not found")
        return profile
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== DISCOVERY CONTROL ====================


@discovery_router.post("/run")
async def trigger_discovery(
    max_markets: int = Query(default=50, ge=1, le=500),
    max_wallets_per_market: int = Query(default=30, ge=1, le=200),
):
    """
    Trigger a manual discovery run.

    Scans active markets for profitable wallets, analyzes their trading
    patterns, and updates the leaderboard. Returns immediately while
    the discovery runs in the background.
    """
    try:
        asyncio.create_task(
            wallet_discovery.run_discovery(
                max_markets=max_markets,
                max_wallets_per_market=max_wallets_per_market,
            )
        )

        return {
            "status": "started",
            "message": "Discovery run started in background",
            "params": {
                "max_markets": max_markets,
                "max_wallets_per_market": max_wallets_per_market,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.post("/refresh-leaderboard")
async def trigger_refresh():
    """
    Force a leaderboard rank recalculation.

    Recomputes rank scores for all tracked wallets using the latest
    metrics without running a full discovery scan.
    """
    try:
        result = await wallet_discovery.refresh_leaderboard()
        return {
            "status": "success",
            "message": "Leaderboard refreshed",
            "result": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== CONFLUENCE SIGNALS ====================


@discovery_router.get("/confluence")
async def get_confluence_signals(
    min_strength: float = Query(default=0.0, ge=0.0, le=1.0),
    limit: int = Query(default=50, ge=1, le=200),
):
    """
    Get active confluence signals.

    Identifies markets where multiple top-ranked wallets are converging
    on the same position. Higher strength indicates stronger agreement
    among skilled traders.
    """
    try:
        signals = await wallet_intelligence.confluence.get_active_signals(
            min_strength=min_strength,
            limit=limit,
        )
        return {"signals": signals}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.post("/confluence/scan")
async def trigger_confluence_scan():
    """
    Trigger a manual confluence scan.

    Analyzes current positions of top wallets to detect convergence
    patterns across active markets.
    """
    try:
        result = await wallet_intelligence.confluence.scan_for_confluence()
        return {
            "status": "success",
            "message": "Confluence scan completed",
            "result": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ENTITY CLUSTERS ====================


@discovery_router.get("/clusters")
async def get_clusters(
    min_wallets: int = Query(default=2, ge=2, le=100),
):
    """
    Get wallet clusters (groups of wallets belonging to the same entity).

    Uses on-chain analysis to identify wallets that are likely controlled
    by the same person or organization based on funding patterns,
    coordinated trading, and timing analysis.
    """
    try:
        clusters = await wallet_intelligence.clusterer.get_clusters(
            min_wallets=min_wallets,
        )
        return {"clusters": clusters}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.get("/clusters/{cluster_id}")
async def get_cluster_detail(cluster_id: str):
    """
    Get detailed info about a specific cluster and its member wallets.

    Returns all wallets in the cluster, evidence linking them, shared
    trading patterns, and aggregate performance metrics.
    """
    try:
        detail = await wallet_intelligence.clusterer.get_cluster_detail(
            cluster_id=cluster_id,
        )
        if not detail:
            raise HTTPException(status_code=404, detail="Cluster not found")
        return detail
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== TAGS ====================


@discovery_router.get("/tags")
async def get_all_tags():
    """
    Get all tag definitions with wallet counts.

    Tags classify wallets by behavior (e.g., whale, sniper, market_maker,
    arbitrageur). Each tag includes a description and the number of
    wallets currently carrying that tag.
    """
    try:
        tags = await wallet_intelligence.tagger.get_all_tags()
        return {"tags": tags}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.get("/tags/{tag_name}/wallets")
async def get_wallets_by_tag(
    tag_name: str,
    limit: int = Query(default=100, ge=1, le=500),
):
    """
    Get wallets with a specific tag.

    Returns a list of wallets classified under the given tag,
    sorted by rank score.
    """
    try:
        wallets = await wallet_intelligence.tagger.get_wallets_by_tag(
            tag_name=tag_name,
            limit=limit,
        )
        return {"wallets": wallets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== CROSS-PLATFORM ====================


@discovery_router.get("/cross-platform")
async def get_cross_platform_entities(
    limit: int = Query(default=50, ge=1, le=200),
):
    """
    Get entities tracked across Polymarket and Kalshi.

    Identifies traders operating on multiple prediction market platforms,
    enabling detection of cross-platform arbitrage strategies and
    providing a more complete view of trader behavior.
    """
    try:
        entities = await wallet_intelligence.cross_platform.get_cross_platform_entities(
            limit=limit,
        )
        return entities
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.get("/cross-platform/arb-activity")
async def get_cross_platform_arb():
    """
    Get recent cross-platform arbitrage activity.

    Returns instances where entities are exploiting price differences
    between Polymarket and Kalshi on the same underlying events.
    """
    try:
        activity = (
            await wallet_intelligence.cross_platform.get_cross_platform_arb_activity()
        )
        return activity
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
