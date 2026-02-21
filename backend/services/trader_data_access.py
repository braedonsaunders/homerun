"""First-class trader data access for strategies, routes, and workers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import desc, func, select

from models.database import (
    AsyncSessionLocal,
    DiscoveredWallet,
    TrackedWallet,
    TraderGroup,
    TraderGroupMember,
)
from services.smart_wallet_pool import (
    POOL_FLAG_BLACKLISTED,
    POOL_FLAG_MANUAL_EXCLUDE,
    POOL_FLAG_MANUAL_INCLUDE,
    smart_wallet_pool,
)
from services.wallet_intelligence import wallet_intelligence
from services.wallet_tracker import wallet_tracker
from utils.utcnow import utcfromtimestamp, utcnow

SQL_IN_CHUNK_SIZE = 2000


def _iter_chunks(values: list[str], chunk_size: int = SQL_IN_CHUNK_SIZE):
    for i in range(0, len(values), chunk_size):
        chunk = values[i : i + chunk_size]
        if chunk:
            yield chunk


def _normalize_wallet_address(value: object) -> Optional[str]:
    text = str(value or "").strip().lower()
    if not text or not text.startswith("0x"):
        return None
    return text


def _coerce_source_flags(raw: object) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def _pool_flags(source_flags: dict[str, Any]) -> dict[str, bool]:
    return {
        "manual_include": bool(source_flags.get(POOL_FLAG_MANUAL_INCLUDE)),
        "manual_exclude": bool(source_flags.get(POOL_FLAG_MANUAL_EXCLUDE)),
        "blacklisted": bool(source_flags.get(POOL_FLAG_BLACKLISTED)),
    }


def _parse_trade_time(value: object) -> Optional[datetime]:
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value

    if value is None:
        return None

    if isinstance(value, (int, float)):
        try:
            return utcfromtimestamp(float(value))
        except Exception:
            return None

    text = str(value).strip()
    if not text:
        return None

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    except Exception:
        try:
            return utcfromtimestamp(float(text))
        except Exception:
            return None


def _recent_trade_activity(wallet: dict[str, Any], cutoff: datetime) -> tuple[int, Optional[datetime], int]:
    trades = wallet.get("recent_trades") if isinstance(wallet, dict) else []
    recent_count = 0
    latest: Optional[datetime] = None

    if isinstance(trades, list):
        for trade in trades:
            if not isinstance(trade, dict):
                continue
            trade_dt: Optional[datetime] = None
            for field in ("timestamp_iso", "match_time", "timestamp", "time", "created_at"):
                raw = trade.get(field)
                trade_dt = _parse_trade_time(raw)
                if trade_dt is not None:
                    break
            if trade_dt is None:
                continue
            if latest is None or trade_dt > latest:
                latest = trade_dt
            if trade_dt >= cutoff:
                recent_count += 1

    open_positions = wallet.get("positions") if isinstance(wallet, dict) else []
    open_count = len(open_positions) if isinstance(open_positions, list) else 0
    return recent_count, latest, open_count


async def annotate_trader_signal_source_context(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Annotate signal rows with canonical source flags and source breakdown."""
    if not rows:
        return rows

    all_addresses: set[str] = set()
    wallets_per_row: list[list[str]] = []

    for row in rows:
        row_wallets: list[str] = []

        wallets = row.get("wallets") if isinstance(row, dict) else None
        if isinstance(wallets, list):
            for raw in wallets:
                addr = None
                if isinstance(raw, dict):
                    addr = _normalize_wallet_address(raw.get("address"))
                else:
                    addr = _normalize_wallet_address(raw)
                if addr is None:
                    continue
                row_wallets.append(addr)
                all_addresses.add(addr)

        top_wallets = row.get("top_wallets") if isinstance(row, dict) else None
        if isinstance(top_wallets, list):
            for raw in top_wallets:
                addr = _normalize_wallet_address(raw.get("address") if isinstance(raw, dict) else raw)
                if addr is None:
                    continue
                row_wallets.append(addr)
                all_addresses.add(addr)

        top_wallet = row.get("top_wallet") if isinstance(row, dict) else None
        if isinstance(top_wallet, dict):
            addr = _normalize_wallet_address(top_wallet.get("address"))
            if addr is not None:
                row_wallets.append(addr)
                all_addresses.add(addr)

        wallet_addresses = row.get("wallet_addresses") if isinstance(row, dict) else None
        if isinstance(wallet_addresses, list):
            for raw in wallet_addresses:
                addr = _normalize_wallet_address(raw)
                if addr is None:
                    continue
                row_wallets.append(addr)
                all_addresses.add(addr)

        wallets_per_row.append(sorted(set(row_wallets)))

    pool_addresses: set[str] = set()
    tracked_addresses: set[str] = set()
    group_ids_by_address: dict[str, set[str]] = {}

    if all_addresses:
        addr_list = list(all_addresses)
        async with AsyncSessionLocal() as session:
            for addr_chunk in _iter_chunks(addr_list):
                discovered_rows = await session.execute(
                    select(DiscoveredWallet.address, DiscoveredWallet.in_top_pool).where(
                        func.lower(DiscoveredWallet.address).in_(addr_chunk)
                    )
                )
                for address, in_top_pool in discovered_rows.all():
                    normalized = _normalize_wallet_address(address)
                    if normalized and bool(in_top_pool):
                        pool_addresses.add(normalized)

                tracked_rows = await session.execute(
                    select(TrackedWallet.address).where(func.lower(TrackedWallet.address).in_(addr_chunk))
                )
                for (address,) in tracked_rows.all():
                    normalized = _normalize_wallet_address(address)
                    if normalized:
                        tracked_addresses.add(normalized)

                group_rows = await session.execute(
                    select(TraderGroupMember.wallet_address, TraderGroupMember.group_id)
                    .join(TraderGroup, TraderGroupMember.group_id == TraderGroup.id)
                    .where(
                        TraderGroup.is_active == True,  # noqa: E712
                        func.lower(TraderGroupMember.wallet_address).in_(addr_chunk),
                    )
                )
                for address, group_id in group_rows.all():
                    normalized = _normalize_wallet_address(address)
                    if normalized is None:
                        continue
                    bucket = group_ids_by_address.setdefault(normalized, set())
                    if group_id:
                        bucket.add(str(group_id))

    for row, wallet_addresses in zip(rows, wallets_per_row):
        pool_wallets = sum(1 for addr in wallet_addresses if addr in pool_addresses)
        tracked_wallets = sum(1 for addr in wallet_addresses if addr in tracked_addresses)
        group_wallets = sum(1 for addr in wallet_addresses if group_ids_by_address.get(addr))
        matched_group_ids = sorted({gid for addr in wallet_addresses for gid in group_ids_by_address.get(addr, set())})

        row["source_flags"] = {
            "from_pool": pool_wallets > 0,
            "from_tracked_traders": tracked_wallets > 0,
            "from_trader_groups": group_wallets > 0,
            "qualified": bool(pool_wallets or tracked_wallets or group_wallets),
        }
        row["source_breakdown"] = {
            "wallets_considered": len(wallet_addresses),
            "pool_wallets": pool_wallets,
            "tracked_wallets": tracked_wallets,
            "group_wallets": group_wallets,
            "group_count": len(matched_group_ids),
            "group_ids": matched_group_ids,
        }

    return rows


async def get_trader_firehose_signals(
    *,
    limit: int = 250,
    include_filtered: bool = False,
    include_source_context: bool = True,
) -> list[dict[str, Any]]:
    """Return firehose rows from smart-wallet confluence storage."""
    safe_limit = max(1, int(limit))
    rows = await smart_wallet_pool.get_tracked_trader_firehose_signals(
        limit=safe_limit,
        include_filtered=bool(include_filtered),
    )
    payload = [dict(row) for row in rows if isinstance(row, dict)]
    if include_source_context:
        await annotate_trader_signal_source_context(payload)
    return payload[:safe_limit]


async def get_strategy_filtered_trader_signals(
    *,
    limit: int = 50,
    include_filtered: bool = False,
) -> list[dict[str, Any]]:
    """Return strategy-filtered trader signals from the traders pipeline."""
    from services.traders_firehose_pipeline import get_strategy_filtered_trader_opportunities

    return await get_strategy_filtered_trader_opportunities(
        limit=max(1, int(limit)),
        include_filtered=bool(include_filtered),
    )


async def get_trader_confluence_signals(
    *,
    min_strength: float = 0.0,
    min_tier: str = "WATCH",
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return active confluence signals from the confluence detector."""
    safe_limit = max(1, int(limit))
    return await wallet_intelligence.confluence.get_active_signals(
        min_strength=float(min_strength),
        min_tier=str(min_tier or "WATCH"),
        limit=safe_limit,
    )


async def get_pooled_traders(
    *,
    limit: int = 200,
    tier: str | None = None,
    include_blacklisted: bool = True,
    tracked_only: bool = False,
) -> list[dict[str, Any]]:
    """Return discovered wallets currently inside the smart pool."""
    safe_limit = max(1, int(limit))
    tier_filter = str(tier or "").strip().lower()

    async with AsyncSessionLocal() as session:
        tracked_result = await session.execute(select(TrackedWallet.address))
        tracked_addresses = {str(row[0]).strip().lower() for row in tracked_result.all() if row[0]}

        query = (
            select(DiscoveredWallet)
            .where(DiscoveredWallet.in_top_pool == True)  # noqa: E712
            .order_by(
                desc(DiscoveredWallet.composite_score),
                desc(DiscoveredWallet.rank_score),
                desc(DiscoveredWallet.total_pnl),
            )
        )
        if tier_filter:
            query = query.where(func.lower(func.coalesce(DiscoveredWallet.pool_tier, "")) == tier_filter)

        rows = (await session.execute(query)).scalars().all()

    output: list[dict[str, Any]] = []
    for row in rows:
        source_flags = _coerce_source_flags(row.source_flags)
        flags = _pool_flags(source_flags)
        address = str(row.address or "").strip().lower()
        is_tracked = address in tracked_addresses

        if tracked_only and not is_tracked:
            continue
        if not include_blacklisted and flags["blacklisted"]:
            continue

        output.append(
            {
                "address": row.address,
                "username": row.username,
                "in_top_pool": bool(row.in_top_pool),
                "pool_tier": row.pool_tier,
                "pool_membership_reason": row.pool_membership_reason,
                "tracked_wallet": is_tracked,
                "pool_flags": flags,
                "rank_score": row.rank_score or 0.0,
                "composite_score": row.composite_score or 0.0,
                "quality_score": row.quality_score or 0.0,
                "activity_score": row.activity_score or 0.0,
                "stability_score": row.stability_score or 0.0,
                "insider_score": row.insider_score or 0.0,
                "recommendation": row.recommendation,
                "tags": list(row.tags or []),
                "source_flags": source_flags,
                "last_trade_at": row.last_trade_at.isoformat() if row.last_trade_at else None,
                "trades_1h": int(row.trades_1h or 0),
                "trades_24h": int(row.trades_24h or 0),
                "total_trades": int(row.total_trades or 0),
                "total_pnl": float(row.total_pnl or 0.0),
                "win_rate": float(row.win_rate or 0.0),
            }
        )
        if len(output) >= safe_limit:
            break

    return output


async def get_tracked_traders(
    *,
    limit: int = 200,
    include_recent_activity: bool = False,
    activity_hours: int = 24,
) -> list[dict[str, Any]]:
    """Return tracked trader wallets with optional activity enrichment."""
    safe_limit = max(1, int(limit))
    hours_window = max(1, int(activity_hours))

    async with AsyncSessionLocal() as session:
        tracked_rows = (
            (
                await session.execute(
                    select(TrackedWallet).order_by(
                        desc(TrackedWallet.last_trade_at),
                        desc(TrackedWallet.total_trades),
                        TrackedWallet.address.asc(),
                    )
                )
            )
            .scalars()
            .all()
        )

        addresses = [str(row.address).strip().lower() for row in tracked_rows if row.address]
        discovered_map: dict[str, DiscoveredWallet] = {}
        if addresses:
            for addr_chunk in _iter_chunks(addresses):
                discovered_rows = await session.execute(
                    select(DiscoveredWallet).where(func.lower(DiscoveredWallet.address).in_(addr_chunk))
                )
                for wallet in discovered_rows.scalars().all():
                    discovered_map[str(wallet.address).strip().lower()] = wallet

    activity_map: dict[str, dict[str, Any]] = {}
    if include_recent_activity:
        cutoff = utcnow() - timedelta(hours=hours_window)
        cached_wallets = await wallet_tracker.get_all_wallets()
        for wallet in cached_wallets:
            if not isinstance(wallet, dict):
                continue
            address = _normalize_wallet_address(wallet.get("address"))
            if address is None:
                continue
            recent_count, latest_trade_at, open_positions = _recent_trade_activity(wallet, cutoff)
            activity_map[address] = {
                "recent_trade_count": recent_count,
                "latest_trade_at": latest_trade_at.isoformat() if latest_trade_at else None,
                "open_positions": open_positions,
                "hours_window": hours_window,
            }

    output: list[dict[str, Any]] = []
    for row in tracked_rows:
        address = str(row.address or "").strip().lower()
        discovered = discovered_map.get(address)
        source_flags = _coerce_source_flags(discovered.source_flags) if discovered else {}

        payload = {
            "address": row.address,
            "label": row.label,
            "total_trades": int(row.total_trades or 0),
            "win_rate": float(row.win_rate or 0.0),
            "total_pnl": float(row.total_pnl or 0.0),
            "avg_roi": float(row.avg_roi or 0.0),
            "last_trade_at": row.last_trade_at.isoformat() if row.last_trade_at else None,
            "anomaly_score": float(row.anomaly_score or 0.0),
            "is_flagged": bool(row.is_flagged),
            "flag_reasons": list(row.flag_reasons or []),
            "last_analyzed_at": row.last_analyzed_at.isoformat() if row.last_analyzed_at else None,
            "analysis_data": dict(row.analysis_data or {}),
            "discovery": {
                "username": discovered.username if discovered else None,
                "rank_score": float(discovered.rank_score or 0.0) if discovered else 0.0,
                "composite_score": float(discovered.composite_score or 0.0) if discovered else 0.0,
                "quality_score": float(discovered.quality_score or 0.0) if discovered else 0.0,
                "activity_score": float(discovered.activity_score or 0.0) if discovered else 0.0,
                "in_top_pool": bool(discovered.in_top_pool) if discovered else False,
                "pool_tier": discovered.pool_tier if discovered else None,
                "recommendation": discovered.recommendation if discovered else None,
                "tags": list(discovered.tags or []) if discovered else [],
                "source_flags": source_flags,
            },
        }
        if include_recent_activity:
            payload["activity"] = activity_map.get(
                address,
                {
                    "recent_trade_count": 0,
                    "latest_trade_at": None,
                    "open_positions": 0,
                    "hours_window": hours_window,
                },
            )

        output.append(payload)
        if len(output) >= safe_limit:
            break

    return output


async def get_trader_groups(
    *,
    include_members: bool = False,
    member_limit: int = 25,
) -> list[dict[str, Any]]:
    """Return active trader groups with optional member payloads."""
    safe_member_limit = max(1, int(member_limit))

    async with AsyncSessionLocal() as session:
        group_rows = await session.execute(
            select(TraderGroup)
            .where(TraderGroup.is_active == True)  # noqa: E712
            .order_by(TraderGroup.created_at.asc())
        )
        groups = list(group_rows.scalars().all())
        if not groups:
            return []

        group_ids = [str(group.id) for group in groups]
        member_rows = await session.execute(
            select(TraderGroupMember)
            .where(TraderGroupMember.group_id.in_(group_ids))
            .order_by(TraderGroupMember.added_at.asc())
        )
        members = list(member_rows.scalars().all())

        members_by_group: dict[str, list[TraderGroupMember]] = {}
        member_addresses: set[str] = set()
        for member in members:
            members_by_group.setdefault(str(member.group_id), []).append(member)
            normalized = _normalize_wallet_address(member.wallet_address)
            if normalized:
                member_addresses.add(normalized)

        profiles_by_address: dict[str, DiscoveredWallet] = {}
        if member_addresses:
            for addr_chunk in _iter_chunks(list(member_addresses)):
                profiles = await session.execute(
                    select(DiscoveredWallet).where(func.lower(DiscoveredWallet.address).in_(addr_chunk))
                )
                for profile in profiles.scalars().all():
                    profiles_by_address[str(profile.address).strip().lower()] = profile

    payload: list[dict[str, Any]] = []
    for group in groups:
        group_id = str(group.id)
        group_members = members_by_group.get(group_id, [])
        item = {
            "id": group.id,
            "name": group.name,
            "description": group.description,
            "source_type": group.source_type,
            "suggestion_key": group.suggestion_key,
            "criteria": dict(group.criteria or {}),
            "auto_track_members": bool(group.auto_track_members),
            "member_count": len(group_members),
            "created_at": group.created_at.isoformat() if group.created_at else None,
            "updated_at": group.updated_at.isoformat() if group.updated_at else None,
        }

        if include_members:
            member_payload: list[dict[str, Any]] = []
            for member in group_members[:safe_member_limit]:
                normalized_address = _normalize_wallet_address(member.wallet_address)
                profile = profiles_by_address.get(normalized_address or "")
                member_payload.append(
                    {
                        "id": member.id,
                        "wallet_address": member.wallet_address,
                        "source": member.source,
                        "confidence": member.confidence,
                        "notes": member.notes,
                        "added_at": member.added_at.isoformat() if member.added_at else None,
                        "username": profile.username if profile else None,
                        "composite_score": profile.composite_score if profile else None,
                        "quality_score": profile.quality_score if profile else None,
                        "activity_score": profile.activity_score if profile else None,
                        "pool_tier": profile.pool_tier if profile else None,
                        "tags": list(profile.tags or []) if profile else [],
                    }
                )
            item["members"] = member_payload

        payload.append(item)

    return payload


async def get_trader_tags() -> list[dict[str, Any]]:
    """Return tag definitions with wallet counts."""
    return await wallet_intelligence.tagger.get_all_tags()


async def get_traders_by_tag(
    tag_name: str,
    *,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return wallets carrying a given trader tag."""
    return await wallet_intelligence.tagger.get_wallets_by_tag(
        tag_name=str(tag_name or "").strip().lower(),
        limit=max(1, int(limit)),
    )


__all__ = [
    "annotate_trader_signal_source_context",
    "get_pooled_traders",
    "get_strategy_filtered_trader_signals",
    "get_tracked_traders",
    "get_trader_confluence_signals",
    "get_trader_firehose_signals",
    "get_trader_groups",
    "get_trader_tags",
    "get_traders_by_tag",
]
