from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import and_, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import Strategy, StrategyVersion
from utils.utcnow import utcnow


@dataclass
class ResolvedStrategyVersion:
    strategy: Strategy
    version_row: StrategyVersion
    latest_version: int
    requested_version: int | None


def _normalize_strategy_key(value: Any) -> str:
    return str(value or "").strip().lower()


def normalize_strategy_version(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("strategy_version must be an integer or 'latest'")
    if isinstance(value, (int, float)):
        parsed = int(value)
        if parsed <= 0:
            raise ValueError("strategy_version must be >= 1")
        return parsed
    text = str(value or "").strip().lower()
    if not text or text == "latest":
        return None
    if text.startswith("v"):
        text = text[1:]
    if not text.isdigit():
        raise ValueError("strategy_version must be an integer or 'latest'")
    parsed = int(text)
    if parsed <= 0:
        raise ValueError("strategy_version must be >= 1")
    return parsed


def serialize_strategy_version(row: StrategyVersion, *, include_source: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": row.id,
        "strategy_id": row.strategy_id,
        "strategy_slug": row.strategy_slug,
        "source_key": row.source_key,
        "version": int(row.version or 1),
        "is_latest": bool(row.is_latest),
        "name": row.name,
        "description": row.description,
        "class_name": row.class_name,
        "config": dict(row.config or {}),
        "config_schema": dict(row.config_schema or {}),
        "aliases": list(row.aliases or []),
        "enabled": bool(row.enabled),
        "is_system": bool(row.is_system),
        "sort_order": int(row.sort_order or 0),
        "parent_version": row.parent_version,
        "created_by": row.created_by,
        "reason": row.reason,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }
    if include_source:
        payload["source_code"] = row.source_code
    return payload


async def list_strategy_versions(
    session: AsyncSession,
    *,
    strategy_id: str,
    limit: int = 200,
) -> list[StrategyVersion]:
    normalized_limit = max(1, min(int(limit or 200), 1000))
    rows = (
        (
            await session.execute(
                select(StrategyVersion)
                .where(StrategyVersion.strategy_id == strategy_id)
                .order_by(StrategyVersion.version.desc())
                .limit(normalized_limit)
            )
        )
        .scalars()
        .all()
    )
    return list(rows)


async def get_strategy_by_id_or_slug(
    session: AsyncSession,
    *,
    strategy_id: str | None = None,
    strategy_key: str | None = None,
) -> Strategy | None:
    if strategy_id:
        row = await session.get(Strategy, strategy_id)
        if row is not None:
            return row
    key = _normalize_strategy_key(strategy_key)
    if not key:
        return None
    return (
        (
            await session.execute(
                select(Strategy)
                .where(func.lower(func.coalesce(Strategy.slug, "")) == key)
                .limit(1)
            )
        )
        .scalars()
        .first()
    )


async def _latest_version_row_for_strategy(
    session: AsyncSession,
    *,
    strategy_id: str,
) -> StrategyVersion | None:
    return (
        (
            await session.execute(
                select(StrategyVersion)
                .where(StrategyVersion.strategy_id == strategy_id)
                .order_by(StrategyVersion.is_latest.desc(), StrategyVersion.version.desc())
                .limit(1)
            )
        )
        .scalars()
        .first()
    )


async def ensure_strategy_version_seeded(
    session: AsyncSession,
    *,
    strategy: Strategy,
    reason: str = "seed_missing_snapshot",
    created_by: str | None = None,
    commit: bool = False,
) -> StrategyVersion:
    latest = await _latest_version_row_for_strategy(session, strategy_id=str(strategy.id))
    if latest is not None:
        return latest
    strategy_version = int(strategy.version or 1)
    seeded = await create_strategy_version_snapshot(
        session,
        strategy=strategy,
        forced_version=max(1, strategy_version),
        parent_version=None,
        reason=reason,
        created_by=created_by,
        commit=commit,
    )
    return seeded


async def create_strategy_version_snapshot(
    session: AsyncSession,
    *,
    strategy: Strategy,
    reason: str,
    created_by: str | None = None,
    parent_version: int | None = None,
    forced_version: int | None = None,
    commit: bool = False,
) -> StrategyVersion:
    strategy_id = str(strategy.id or "").strip()
    if not strategy_id:
        raise ValueError("Strategy id is required to create a version snapshot")

    current_max = (
        await session.execute(
            select(func.max(StrategyVersion.version)).where(StrategyVersion.strategy_id == strategy_id)
        )
    ).scalar_one_or_none()
    next_version = int(forced_version or (int(current_max or 0) + 1))
    if next_version <= 0:
        raise ValueError("Version number must be >= 1")

    existing_version = (
        (
            await session.execute(
                select(StrategyVersion)
                .where(
                    and_(
                        StrategyVersion.strategy_id == strategy_id,
                        StrategyVersion.version == next_version,
                    )
                )
                .limit(1)
            )
        )
        .scalars()
        .first()
    )
    if existing_version is not None:
        await session.execute(
            update(StrategyVersion)
            .where(StrategyVersion.strategy_id == strategy_id)
            .values(is_latest=(StrategyVersion.version == next_version))
        )
        strategy.version = next_version
        strategy.updated_at = utcnow()
        await session.flush()
        if commit:
            await session.commit()
            await session.refresh(existing_version)
            await session.refresh(strategy)
        return existing_version

    await session.execute(
        update(StrategyVersion)
        .where(StrategyVersion.strategy_id == strategy_id)
        .values(is_latest=False)
    )

    snapshot = StrategyVersion(
        id=uuid.uuid4().hex,
        strategy_id=strategy_id,
        strategy_slug=str(strategy.slug or "").strip().lower(),
        source_key=str(strategy.source_key or "scanner").strip().lower() or "scanner",
        version=next_version,
        is_latest=True,
        name=str(strategy.name or strategy.slug or "strategy").strip(),
        description=strategy.description,
        source_code=str(strategy.source_code or ""),
        class_name=str(strategy.class_name or "").strip() or None,
        config=dict(strategy.config or {}),
        config_schema=dict(strategy.config_schema or {}),
        aliases=list(strategy.aliases or []),
        enabled=bool(strategy.enabled),
        is_system=bool(strategy.is_system),
        sort_order=int(strategy.sort_order or 0),
        parent_version=parent_version,
        created_by=(str(created_by or "").strip() or None),
        reason=str(reason or "manual_snapshot").strip() or "manual_snapshot",
        created_at=utcnow(),
    )
    session.add(snapshot)
    strategy.version = next_version
    strategy.updated_at = utcnow()
    await session.flush()

    if commit:
        await session.commit()
        await session.refresh(snapshot)
        await session.refresh(strategy)

    return snapshot


async def resolve_strategy_version(
    session: AsyncSession,
    *,
    strategy_key: str,
    requested_version: int | None,
) -> ResolvedStrategyVersion:
    normalized_key = _normalize_strategy_key(strategy_key)
    if not normalized_key:
        raise ValueError("strategy_key is required")

    strategy_row = await get_strategy_by_id_or_slug(session, strategy_key=normalized_key)
    if strategy_row is None:
        raise ValueError(f"Strategy '{normalized_key}' was not found")

    latest_row = await ensure_strategy_version_seeded(
        session,
        strategy=strategy_row,
        reason="seed_during_resolution",
        created_by="runtime",
        commit=False,
    )
    latest_version = int(latest_row.version or strategy_row.version or 1)

    if requested_version is None:
        return ResolvedStrategyVersion(
            strategy=strategy_row,
            version_row=latest_row,
            latest_version=latest_version,
            requested_version=None,
        )

    version_row = (
        (
            await session.execute(
                select(StrategyVersion)
                .where(
                    and_(
                        StrategyVersion.strategy_id == strategy_row.id,
                        StrategyVersion.version == int(requested_version),
                    )
                )
                .limit(1)
            )
        )
        .scalars()
        .first()
    )
    if version_row is None:
        raise ValueError(
            f"Strategy '{normalized_key}' does not have version {int(requested_version)}"
        )

    return ResolvedStrategyVersion(
        strategy=strategy_row,
        version_row=version_row,
        latest_version=latest_version,
        requested_version=int(requested_version),
    )


async def strategy_versions_by_slug(
    session: AsyncSession,
    *,
    source_key: str | None = None,
) -> dict[str, list[int]]:
    query = select(StrategyVersion.strategy_slug, StrategyVersion.version)
    if source_key is not None:
        normalized_source = str(source_key or "").strip().lower()
        query = query.where(func.lower(func.coalesce(StrategyVersion.source_key, "")) == normalized_source)
    rows = (await session.execute(query.order_by(StrategyVersion.strategy_slug.asc(), StrategyVersion.version.desc()))).all()
    out: dict[str, list[int]] = {}
    for slug, version in rows:
        key = _normalize_strategy_key(slug)
        if not key:
            continue
        parsed = int(version or 0)
        if parsed <= 0:
            continue
        out.setdefault(key, []).append(parsed)
    return out


async def restore_strategy_from_snapshot(
    session: AsyncSession,
    *,
    strategy: Strategy,
    snapshot: StrategyVersion,
    reason: str,
    created_by: str | None = None,
    commit: bool = False,
) -> StrategyVersion:
    strategy.source_key = str(snapshot.source_key or strategy.source_key or "scanner").strip().lower() or "scanner"
    strategy.name = str(snapshot.name or strategy.name or strategy.slug or "strategy").strip()
    strategy.description = snapshot.description
    strategy.source_code = str(snapshot.source_code or "")
    strategy.class_name = str(snapshot.class_name or "").strip() or None
    strategy.config = dict(snapshot.config or {})
    strategy.config_schema = dict(snapshot.config_schema or {})
    strategy.aliases = list(snapshot.aliases or [])
    strategy.enabled = bool(snapshot.enabled)
    strategy.is_system = bool(snapshot.is_system)
    strategy.sort_order = int(snapshot.sort_order or 0)
    strategy.updated_at = utcnow()

    restored = await create_strategy_version_snapshot(
        session,
        strategy=strategy,
        reason=reason,
        created_by=created_by,
        parent_version=int(snapshot.version or 1),
        commit=commit,
    )
    return restored
