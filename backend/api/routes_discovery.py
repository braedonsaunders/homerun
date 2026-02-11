from collections import defaultdict
from datetime import datetime, timedelta
from utils.utcnow import utcnow
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    DiscoveredWallet,
    TraderGroup,
    TraderGroupMember,
    WalletCluster,
    get_db_session,
)
from services import discovery_shared_state
from services.insider_detector import insider_detector
from services.wallet_discovery import wallet_discovery
from services.wallet_intelligence import wallet_intelligence
from services.smart_wallet_pool import smart_wallet_pool
from services.wallet_tracker import wallet_tracker
from services.worker_state import read_worker_snapshot
from utils.validation import validate_eth_address

# Maps time_period query values to rolling window keys stored in the DB
TIME_PERIOD_TO_WINDOW_KEY = {
    "24h": "1d",
    "7d": "7d",
    "30d": "30d",
    "90d": "90d",
}

discovery_router = APIRouter()


GROUP_SOURCE_TYPES = {
    "manual",
    "suggested_cluster",
    "suggested_tag",
    "suggested_pool",
}


async def _build_discovery_status(session: AsyncSession) -> dict:
    stats = await wallet_discovery.get_discovery_stats()
    worker_status = await discovery_shared_state.get_discovery_status_from_db(session)

    stats["is_running"] = bool(worker_status.get("running", stats.get("is_running", False)))
    stats["last_run_at"] = worker_status.get("last_run_at") or stats.get("last_run_at")
    stats["wallets_discovered_last_run"] = int(
        worker_status.get(
            "wallets_discovered_last_run",
            stats.get("wallets_discovered_last_run", 0),
        )
    )
    stats["wallets_analyzed_last_run"] = int(
        worker_status.get(
            "wallets_analyzed_last_run",
            stats.get("wallets_analyzed_last_run", 0),
        )
    )
    stats["current_activity"] = worker_status.get("current_activity")
    stats["interval_minutes"] = int(
        worker_status.get("run_interval_minutes", 60)
    )
    stats["paused"] = bool(worker_status.get("paused", False))
    stats["requested_run_at"] = worker_status.get("requested_run_at")
    return stats


class CreateTraderGroupRequest(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    description: Optional[str] = Field(default=None, max_length=500)
    wallet_addresses: list[str] = Field(default_factory=list)
    source_type: str = Field(default="manual")
    suggestion_key: Optional[str] = Field(default=None, max_length=200)
    criteria: dict = Field(default_factory=dict)
    auto_track_members: bool = True
    source_label: str = Field(default="manual", max_length=40)


class UpdateTraderGroupMembersRequest(BaseModel):
    wallet_addresses: list[str] = Field(default_factory=list)
    add_to_tracking: bool = True
    source_label: str = Field(default="manual", max_length=40)


def _normalize_wallet_addresses(addresses: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in addresses:
        if not raw:
            continue
        addr = validate_eth_address(str(raw).strip()).lower()
        if addr in seen:
            continue
        normalized.append(addr)
        seen.add(addr)
    return normalized


def _suggestion_id(kind: str, key: str) -> str:
    safe_key = "".join(ch if ch.isalnum() else "_" for ch in key.lower())
    return f"{kind}:{safe_key}"


def _first_valid_trade_time(trade: dict) -> Optional[datetime]:
    for field in ("timestamp_iso", "match_time", "timestamp", "time", "created_at"):
        raw = trade.get(field)
        if raw is None:
            continue
        try:
            if isinstance(raw, (int, float)):
                return datetime.fromtimestamp(raw)
            text = str(raw).strip()
            if not text:
                continue
            if "T" in text or "-" in text:
                return datetime.fromisoformat(
                    text.replace("Z", "+00:00").replace("+00:00", "")
                )
            return datetime.fromtimestamp(float(text))
        except Exception:
            continue
    return None


async def _track_wallet_addresses(
    addresses: list[str],
    label: str,
    fetch_initial: bool = False,
) -> int:
    tracked = 0
    for address in addresses:
        try:
            await wallet_tracker.add_wallet(
                address=address,
                label=label,
                fetch_initial=fetch_initial,
            )
            tracked += 1
        except Exception:
            continue
    return tracked


async def _fetch_group_payload(
    session: AsyncSession,
    include_members: bool = False,
    member_limit: int = 25,
) -> list[dict]:
    group_result = await session.execute(
        select(TraderGroup)
        .where(TraderGroup.is_active == True)  # noqa: E712
        .order_by(TraderGroup.created_at.asc())
    )
    groups = list(group_result.scalars().all())
    if not groups:
        return []

    group_ids = [g.id for g in groups]
    member_result = await session.execute(
        select(TraderGroupMember)
        .where(TraderGroupMember.group_id.in_(group_ids))
        .order_by(TraderGroupMember.added_at.asc())
    )
    members = list(member_result.scalars().all())

    members_by_group: dict[str, list[TraderGroupMember]] = defaultdict(list)
    all_addresses: set[str] = set()
    for member in members:
        members_by_group[member.group_id].append(member)
        all_addresses.add(member.wallet_address.lower())

    profile_map: dict[str, DiscoveredWallet] = {}
    if all_addresses:
        profile_rows = await session.execute(
            select(DiscoveredWallet).where(
                DiscoveredWallet.address.in_(list(all_addresses))
            )
        )
        for wallet in profile_rows.scalars().all():
            profile_map[wallet.address.lower()] = wallet

    payload: list[dict] = []
    for group in groups:
        group_members = members_by_group.get(group.id, [])
        item = {
            "id": group.id,
            "name": group.name,
            "description": group.description,
            "source_type": group.source_type,
            "suggestion_key": group.suggestion_key,
            "criteria": group.criteria or {},
            "auto_track_members": bool(group.auto_track_members),
            "member_count": len(group_members),
            "created_at": group.created_at.isoformat() if group.created_at else None,
            "updated_at": group.updated_at.isoformat() if group.updated_at else None,
        }
        if include_members:
            member_payload = []
            for member in group_members[:member_limit]:
                profile = profile_map.get(member.wallet_address.lower())
                member_payload.append(
                    {
                        "id": member.id,
                        "wallet_address": member.wallet_address,
                        "source": member.source,
                        "confidence": member.confidence,
                        "notes": member.notes,
                        "added_at": member.added_at.isoformat()
                        if member.added_at
                        else None,
                        "username": profile.username if profile else None,
                        "composite_score": (
                            profile.composite_score if profile else None
                        ),
                        "quality_score": profile.quality_score if profile else None,
                        "activity_score": profile.activity_score if profile else None,
                        "pool_tier": profile.pool_tier if profile else None,
                    }
                )
            item["members"] = member_payload
        payload.append(item)

    return payload


# ==================== LEADERBOARD ====================


@discovery_router.get("/leaderboard")
async def get_leaderboard(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    min_trades: int = Query(default=0, ge=0),
    min_pnl: float = Query(default=0.0),
    insider_only: bool = Query(default=False),
    min_insider_score: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    sort_by: str = Query(
        default="rank_score",
        description="rank_score, total_pnl, win_rate, sharpe_ratio, profit_factor, insider_score",
    ),
    sort_dir: str = Query(default="desc", description="asc or desc"),
    tags: Optional[str] = Query(default=None, description="Comma-separated tag filter"),
    recommendation: Optional[str] = Query(
        default=None, description="copy_candidate, monitor, avoid"
    ),
    time_period: Optional[str] = Query(
        default=None,
        description="Time period filter: 24h, 7d, 30d, 90d, or all (default all)",
    ),
    active_within_hours: Optional[int] = Query(
        default=None,
        ge=1,
        le=720,
        description="Only include wallets with activity within last N hours",
    ),
    min_activity_score: Optional[float] = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum activity score filter",
    ),
    pool_only: bool = Query(
        default=False,
        description="Only include wallets currently in the smart top pool",
    ),
    tier: Optional[str] = Query(
        default=None,
        description="Pool tier filter: core, rising, standby",
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
            "composite_score",
            "quality_score",
            "activity_score",
            "stability_score",
            "last_trade_at",
            "total_pnl",
            "total_returned",
            "win_rate",
            "sharpe_ratio",
            "profit_factor",
            "total_trades",
            "avg_roi",
            "sortino_ratio",
            "trades_per_day",
            "insider_score",
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

        # Map time_period to rolling window key
        window_key = None
        if time_period and time_period != "all":
            window_key = TIME_PERIOD_TO_WINDOW_KEY.get(time_period)
            if window_key is None:
                valid_periods = list(TIME_PERIOD_TO_WINDOW_KEY.keys()) + ["all"]
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid time_period. Must be one of: {valid_periods}",
                )

        # Parse comma-separated tags
        tag_list = tags.split(",") if tags else None

        result = await wallet_discovery.get_leaderboard(
            limit=limit,
            offset=offset,
            min_trades=min_trades,
            min_pnl=min_pnl,
            insider_only=insider_only,
            min_insider_score=min_insider_score,
            sort_by=sort_by,
            sort_dir=sort_dir,
            tags=tag_list,
            recommendation=recommendation,
            window_key=window_key,
            active_within_hours=active_within_hours,
            min_activity_score=min_activity_score,
            pool_only=pool_only,
            tier=tier,
        )

        # When no discovered wallets yet, fall back to live Polymarket leaderboard so UI shows traders
        if (result.get("total") or 0) == 0 and offset == 0:
            try:
                from services.polymarket import polymarket_client
                raw = await polymarket_client.get_leaderboard(
                    limit=min(limit, 50),
                    time_period="ALL",
                    order_by="PNL",
                    category="OVERALL",
                )
                if raw:
                    wallets = [
                        {
                            "address": e.get("proxyWallet", ""),
                            "username": e.get("userName"),
                            "total_pnl": float(e.get("pnl") or 0),
                            "vol": float(e.get("vol") or 0),
                            "rank_position": int(e.get("rank", 0)) if str(e.get("rank", "")).isdigit() else None,
                            "tags": [],
                            "from_polymarket_live": True,
                        }
                        for e in raw
                        if e.get("proxyWallet")
                    ]
                    return {"wallets": wallets, "total": len(wallets), "from_polymarket_live": True}
            except Exception:
                pass

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.get("/leaderboard/stats")
async def get_discovery_stats(session: AsyncSession = Depends(get_db_session)):
    """
    Get discovery engine statistics.

    Returns total wallets analyzed, last run timestamp, coverage metrics,
    and engine health information.
    """
    try:
        return await _build_discovery_status(session)
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
    session: AsyncSession = Depends(get_db_session),
):
    """
    Queue a one-time discovery run for the discovery worker.
    """
    try:
        await discovery_shared_state.request_one_discovery_run(session)
        return {
            "status": "queued",
            "message": "Discovery run requested; worker will execute on next cycle.",
            **await discovery_shared_state.get_discovery_status_from_db(session),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.post("/start")
async def start_discovery_worker(session: AsyncSession = Depends(get_db_session)):
    """Resume automatic discovery worker cycles."""
    await discovery_shared_state.set_discovery_paused(session, False)
    return {
        "status": "started",
        **await discovery_shared_state.get_discovery_status_from_db(session),
    }


@discovery_router.post("/pause")
async def pause_discovery_worker(session: AsyncSession = Depends(get_db_session)):
    """Pause automatic discovery worker cycles."""
    await discovery_shared_state.set_discovery_paused(session, True)
    return {
        "status": "paused",
        **await discovery_shared_state.get_discovery_status_from_db(session),
    }


@discovery_router.post("/interval")
async def set_discovery_interval(
    interval_minutes: int = Query(..., ge=5, le=1440),
    session: AsyncSession = Depends(get_db_session),
):
    """Set discovery worker cadence in minutes."""
    await discovery_shared_state.set_discovery_interval(session, interval_minutes)
    return {
        "status": "updated",
        **await discovery_shared_state.get_discovery_status_from_db(session),
    }


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
    min_tier: str = Query(default="WATCH", description="WATCH, HIGH, EXTREME"),
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
            min_tier=min_tier,
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


@discovery_router.get("/pool/stats")
async def get_smart_pool_stats(session: AsyncSession = Depends(get_db_session)):
    """Get near-real-time smart wallet pool health metrics."""
    try:
        worker = await read_worker_snapshot(session, "tracked_traders")
        stats = worker.get("stats") or {}
        if isinstance(stats, dict) and stats:
            return stats.get("pool_stats") or stats
        return await smart_wallet_pool.get_pool_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.get("/traders")
async def get_traders_overview(
    tracked_limit: int = Query(default=200, ge=1, le=500),
    confluence_limit: int = Query(default=50, ge=1, le=200),
    min_tier: str = Query(default="WATCH", description="WATCH, HIGH, EXTREME"),
    hours: int = Query(
        default=24,
        ge=1,
        le=168,
        description="Tracked trade activity window in hours",
    ),
    session: AsyncSession = Depends(get_db_session),
):
    """Broader traders view: tracked traders + discovery confluence + groups."""
    try:
        tracked_wallets = await wallet_tracker.get_all_wallets()
        cutoff = utcnow() - timedelta(hours=hours)

        tracked_rows: list[dict] = []
        total_recent_trades = 0
        for wallet in tracked_wallets:
            trades = wallet.get("recent_trades") or []
            latest_dt: Optional[datetime] = None
            recent_trades = 0
            for trade in trades:
                trade_dt = _first_valid_trade_time(trade)
                if trade_dt:
                    if latest_dt is None or trade_dt > latest_dt:
                        latest_dt = trade_dt
                    if trade_dt >= cutoff:
                        recent_trades += 1

            total_recent_trades += recent_trades
            tracked_rows.append(
                {
                    "address": wallet.get("address"),
                    "label": wallet.get("label"),
                    "username": wallet.get("username"),
                    "recent_trade_count": recent_trades,
                    "latest_trade_at": latest_dt.isoformat() if latest_dt else None,
                    "open_positions": len(wallet.get("positions") or []),
                }
            )

        tracked_rows.sort(
            key=lambda row: (
                row["recent_trade_count"],
                row.get("latest_trade_at") or "",
            ),
            reverse=True,
        )

        confluence_signals = await smart_wallet_pool.get_tracked_trader_opportunities(
            limit=confluence_limit,
            min_tier=min_tier,
        )

        groups = await _fetch_group_payload(session=session, include_members=False)

        return {
            "tracked": {
                "wallets": tracked_rows[:tracked_limit],
                "total_wallets": len(tracked_rows),
                "hours_window": hours,
                "recent_trade_count": total_recent_trades,
            },
            "groups": {
                "items": groups,
                "total_groups": len(groups),
                "total_members": sum(g.get("member_count", 0) for g in groups),
            },
            "confluence": {
                "signals": confluence_signals,
                "total_signals": len(confluence_signals),
                "min_tier": min_tier.upper(),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.get("/opportunities/tracked-traders")
async def get_tracked_trader_opportunities(
    limit: int = Query(default=50, ge=1, le=200),
    min_tier: str = Query(default="WATCH", description="WATCH, HIGH, EXTREME"),
):
    """Legacy alias for confluence signals powering the Traders surface."""
    try:
        opportunities = await smart_wallet_pool.get_tracked_trader_opportunities(
            limit=limit,
            min_tier=min_tier,
        )
        return {"opportunities": opportunities, "total": len(opportunities)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.get("/opportunities/insider")
async def get_insider_opportunities(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    direction: Optional[str] = Query(default=None, description="buy_yes or buy_no"),
    max_age_minutes: int = Query(default=180, ge=1, le=1440),
):
    """List pending/submitted insider opportunities built from flagged-wallet behavior."""
    if direction and direction not in {"buy_yes", "buy_no"}:
        raise HTTPException(status_code=400, detail="direction must be buy_yes or buy_no")
    try:
        return await insider_detector.list_opportunities(
            limit=limit,
            offset=offset,
            min_confidence=min_confidence,
            direction=direction,
            max_age_minutes=max_age_minutes,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.get("/insider/intents")
async def get_insider_intents(
    status_filter: Optional[str] = Query(
        default=None,
        description="pending, submitted, executed, skipped, expired",
    ),
    limit: int = Query(default=100, ge=1, le=1000),
):
    """List insider trade intents for the autotrader pipeline."""
    if status_filter and status_filter not in {
        "pending",
        "submitted",
        "executed",
        "skipped",
        "expired",
    }:
        raise HTTPException(
            status_code=400,
            detail="Invalid status_filter. Must be pending|submitted|executed|skipped|expired",
        )
    try:
        rows = await insider_detector.list_intents(status_filter=status_filter, limit=limit)
        intents = [
            {
                "id": row.id,
                "signal_key": row.signal_key,
                "market_id": row.market_id,
                "market_question": row.market_question,
                "direction": row.direction,
                "entry_price": row.entry_price,
                "edge_percent": row.edge_percent,
                "confidence": row.confidence,
                "insider_score": row.insider_score,
                "wallet_addresses": row.wallet_addresses_json or [],
                "suggested_size_usd": row.suggested_size_usd,
                "metadata": row.metadata_json or {},
                "status": row.status,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "consumed_at": row.consumed_at.isoformat() if row.consumed_at else None,
            }
            for row in rows
        ]
        return {"total": len(intents), "intents": intents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== TRADER GROUPS ====================


@discovery_router.get("/groups")
async def get_trader_groups(
    include_members: bool = Query(default=False),
    member_limit: int = Query(default=25, ge=1, le=200),
    session: AsyncSession = Depends(get_db_session),
):
    """List tracked trader groups."""
    try:
        groups = await _fetch_group_payload(
            session=session,
            include_members=include_members,
            member_limit=member_limit,
        )
        return {"groups": groups, "total": len(groups)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.post("/groups")
async def create_trader_group(
    payload: CreateTraderGroupRequest,
    session: AsyncSession = Depends(get_db_session),
):
    """Create a trader group and optionally add all members to tracking."""
    try:
        name = payload.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Group name is required")

        source_type = payload.source_type.strip().lower()
        if source_type not in GROUP_SOURCE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid source_type. Must be one of: {sorted(GROUP_SOURCE_TYPES)}",
            )

        existing = await session.execute(
            select(TraderGroup).where(func.lower(TraderGroup.name) == name.lower())
        )
        if existing.scalars().first():
            raise HTTPException(
                status_code=409,
                detail=f"Trader group '{name}' already exists",
            )

        try:
            addresses = _normalize_wallet_addresses(payload.wallet_addresses)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        group = TraderGroup(
            id=str(uuid.uuid4()),
            name=name,
            description=(payload.description or None),
            source_type=source_type,
            suggestion_key=payload.suggestion_key,
            criteria=payload.criteria or {},
            auto_track_members=bool(payload.auto_track_members),
            is_active=True,
        )
        session.add(group)
        for address in addresses:
            session.add(
                TraderGroupMember(
                    id=str(uuid.uuid4()),
                    group_id=group.id,
                    wallet_address=address,
                    source=payload.source_label,
                )
            )
        await session.commit()

        tracked_count = 0
        if payload.auto_track_members and addresses:
            tracked_count = await _track_wallet_addresses(
                addresses=addresses,
                label=f"Group: {name}",
                fetch_initial=False,
            )

        groups = await _fetch_group_payload(
            session=session,
            include_members=True,
            member_limit=200,
        )
        created = next((g for g in groups if g.get("id") == group.id), None)

        return {
            "status": "success",
            "group": created,
            "tracked_members": tracked_count,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.post("/groups/{group_id}/members")
async def add_group_members(
    group_id: str,
    payload: UpdateTraderGroupMembersRequest,
    session: AsyncSession = Depends(get_db_session),
):
    """Add wallets to an existing trader group."""
    try:
        group = await session.get(TraderGroup, group_id)
        if not group or not group.is_active:
            raise HTTPException(status_code=404, detail="Trader group not found")

        try:
            addresses = _normalize_wallet_addresses(payload.wallet_addresses)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if not addresses:
            return {"status": "success", "added_members": 0, "tracked_members": 0}

        existing_members = await session.execute(
            select(TraderGroupMember.wallet_address).where(
                TraderGroupMember.group_id == group_id
            )
        )
        existing_addresses = {
            str(address).lower()
            for address in existing_members.scalars().all()
            if isinstance(address, str)
        }

        to_add = [addr for addr in addresses if addr not in existing_addresses]
        for address in to_add:
            session.add(
                TraderGroupMember(
                    id=str(uuid.uuid4()),
                    group_id=group_id,
                    wallet_address=address,
                    source=payload.source_label,
                )
            )
        await session.commit()

        tracked_count = 0
        if payload.add_to_tracking and to_add:
            tracked_count = await _track_wallet_addresses(
                addresses=to_add,
                label=f"Group: {group.name}",
                fetch_initial=False,
            )

        groups = await _fetch_group_payload(
            session=session,
            include_members=True,
            member_limit=200,
        )
        updated = next((g for g in groups if g.get("id") == group_id), None)

        return {
            "status": "success",
            "added_members": len(to_add),
            "tracked_members": tracked_count,
            "group": updated,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.delete("/groups/{group_id}/members/{wallet_address}")
async def remove_group_member(
    group_id: str,
    wallet_address: str,
    session: AsyncSession = Depends(get_db_session),
):
    """Remove a wallet from a trader group."""
    try:
        try:
            address = validate_eth_address(wallet_address).lower()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        member_result = await session.execute(
            select(TraderGroupMember).where(
                TraderGroupMember.group_id == group_id,
                TraderGroupMember.wallet_address == address,
            )
        )
        member = member_result.scalars().first()
        if not member:
            raise HTTPException(status_code=404, detail="Group member not found")

        await session.delete(member)
        await session.commit()
        return {
            "status": "success",
            "group_id": group_id,
            "wallet_address": address,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.delete("/groups/{group_id}")
async def delete_trader_group(
    group_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    """Delete a trader group and all its memberships."""
    try:
        group = await session.get(TraderGroup, group_id)
        if not group:
            raise HTTPException(status_code=404, detail="Trader group not found")

        members_result = await session.execute(
            select(TraderGroupMember).where(TraderGroupMember.group_id == group_id)
        )
        for member in members_result.scalars().all():
            await session.delete(member)
        await session.delete(group)
        await session.commit()
        return {"status": "success", "group_id": group_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.post("/groups/{group_id}/track")
async def track_group_members(
    group_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    """Ensure all wallets in a trader group are tracked by the wallet monitor."""
    try:
        group = await session.get(TraderGroup, group_id)
        if not group or not group.is_active:
            raise HTTPException(status_code=404, detail="Trader group not found")

        member_result = await session.execute(
            select(TraderGroupMember.wallet_address).where(
                TraderGroupMember.group_id == group_id
            )
        )
        addresses = [addr for addr in member_result.scalars().all() if isinstance(addr, str)]

        tracked_count = await _track_wallet_addresses(
            addresses=addresses,
            label=f"Group: {group.name}",
            fetch_initial=False,
        )
        return {
            "status": "success",
            "group_id": group_id,
            "tracked_members": tracked_count,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.get("/groups/suggestions")
async def get_group_suggestions(
    min_group_size: int = Query(default=3, ge=2, le=100),
    max_suggestions: int = Query(default=12, ge=1, le=100),
    min_composite_score: float = Query(default=0.55, ge=0.0, le=1.0),
    session: AsyncSession = Depends(get_db_session),
):
    """Suggest high-quality trader groups from discovery clusters/tags/pool tiers."""
    try:
        suggestions: list[dict] = []

        existing_result = await session.execute(select(TraderGroup.name))
        existing_names = {
            str(name).strip().lower()
            for name in existing_result.scalars().all()
            if isinstance(name, str)
        }

        # 1) Cluster-driven suggestions (entity-linked wallets)
        cluster_result = await session.execute(
            select(WalletCluster)
            .where(WalletCluster.total_wallets >= min_group_size)
            .order_by(WalletCluster.combined_pnl.desc())
            .limit(max_suggestions * 2)
        )
        for cluster in cluster_result.scalars().all():
            wallet_rows = await session.execute(
                select(DiscoveredWallet)
                .where(
                    DiscoveredWallet.cluster_id == cluster.id,
                    DiscoveredWallet.composite_score >= min_composite_score,
                )
                .order_by(DiscoveredWallet.composite_score.desc())
                .limit(200)
            )
            wallets = list(wallet_rows.scalars().all())
            if len(wallets) < min_group_size:
                continue

            addresses = [w.address.lower() for w in wallets]
            name = (cluster.label or f"Cluster {cluster.id[:8]}").strip()
            avg_comp = sum((w.composite_score or 0.0) for w in wallets) / len(wallets)
            suggestions.append(
                {
                    "id": _suggestion_id("cluster", cluster.id),
                    "kind": "cluster",
                    "name": name,
                    "description": (
                        f"Entity-linked wallets discovered via clustering "
                        f"(confidence {round((cluster.confidence or 0.0) * 100)}%)."
                    ),
                    "wallet_count": len(addresses),
                    "wallet_addresses": addresses,
                    "avg_composite_score": round(avg_comp, 4),
                    "sample_wallets": [
                        {
                            "address": w.address,
                            "username": w.username,
                            "composite_score": w.composite_score or 0.0,
                            "pool_tier": w.pool_tier,
                        }
                        for w in wallets[:8]
                    ],
                    "criteria": {
                        "cluster_id": cluster.id,
                        "confidence": cluster.confidence or 0.0,
                    },
                    "already_exists": name.lower() in existing_names,
                }
            )

        # 2) Pool tier suggestions (core/rising/standby quality cohorts)
        for tier in ("core", "rising", "standby"):
            tier_rows = await session.execute(
                select(DiscoveredWallet)
                .where(
                    DiscoveredWallet.in_top_pool == True,  # noqa: E712
                    func.lower(DiscoveredWallet.pool_tier) == tier,
                    DiscoveredWallet.composite_score >= min_composite_score,
                )
                .order_by(
                    DiscoveredWallet.composite_score.desc(),
                    DiscoveredWallet.last_trade_at.desc(),
                )
                .limit(200)
            )
            wallets = list(tier_rows.scalars().all())
            if len(wallets) < min_group_size:
                continue

            name = f"{tier.title()} Pool Traders"
            addresses = [w.address.lower() for w in wallets]
            avg_comp = sum((w.composite_score or 0.0) for w in wallets) / len(wallets)
            suggestions.append(
                {
                    "id": _suggestion_id("pool", tier),
                    "kind": "pool_tier",
                    "name": name,
                    "description": (
                        f"Auto-group from discovery smart pool tier '{tier}'."
                    ),
                    "wallet_count": len(addresses),
                    "wallet_addresses": addresses,
                    "avg_composite_score": round(avg_comp, 4),
                    "sample_wallets": [
                        {
                            "address": w.address,
                            "username": w.username,
                            "composite_score": w.composite_score or 0.0,
                            "pool_tier": w.pool_tier,
                        }
                        for w in wallets[:8]
                    ],
                    "criteria": {"pool_tier": tier},
                    "already_exists": name.lower() in existing_names,
                }
            )

        # 3) Tag-driven suggestions from high-quality discovered wallets
        tag_candidate_rows = await session.execute(
            select(DiscoveredWallet)
            .where(DiscoveredWallet.composite_score >= min_composite_score)
            .order_by(DiscoveredWallet.composite_score.desc())
            .limit(1500)
        )
        wallets = list(tag_candidate_rows.scalars().all())
        wallets_by_tag: dict[str, list[DiscoveredWallet]] = defaultdict(list)
        for wallet in wallets:
            for tag in wallet.tags or []:
                if isinstance(tag, str) and tag.strip():
                    wallets_by_tag[tag.strip().lower()].append(wallet)

        for tag, tagged_wallets in wallets_by_tag.items():
            unique_wallets = {w.address.lower(): w for w in tagged_wallets}
            rows = sorted(
                unique_wallets.values(),
                key=lambda w: (w.composite_score or 0.0),
                reverse=True,
            )
            if len(rows) < min_group_size:
                continue

            display_tag = tag.replace("_", " ").title()
            name = f"{display_tag} Traders"
            addresses = [w.address.lower() for w in rows[:200]]
            avg_comp = sum((w.composite_score or 0.0) for w in rows[:200]) / len(
                addresses
            )
            suggestions.append(
                {
                    "id": _suggestion_id("tag", tag),
                    "kind": "tag",
                    "name": name,
                    "description": (
                        f"High-quality discovered traders sharing tag '{tag}'."
                    ),
                    "wallet_count": len(addresses),
                    "wallet_addresses": addresses,
                    "avg_composite_score": round(avg_comp, 4),
                    "sample_wallets": [
                        {
                            "address": w.address,
                            "username": w.username,
                            "composite_score": w.composite_score or 0.0,
                            "pool_tier": w.pool_tier,
                        }
                        for w in rows[:8]
                    ],
                    "criteria": {"tag": tag},
                    "already_exists": name.lower() in existing_names,
                }
            )

        suggestions.sort(
            key=lambda s: (s["wallet_count"], s.get("avg_composite_score") or 0.0),
            reverse=True,
        )

        # De-duplicate by suggestion id and trim.
        deduped: dict[str, dict] = {}
        for suggestion in suggestions:
            sid = suggestion["id"]
            if sid in deduped:
                continue
            deduped[sid] = suggestion
            if len(deduped) >= max_suggestions:
                break

        return {"suggestions": list(deduped.values()), "total": len(deduped)}
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
