from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import AsyncSessionLocal, Strategy, StrategyRuntimeRevision
from services.opportunity_strategy_catalog import ensure_all_strategies_seeded
from services.strategy_loader import strategy_loader
from utils.utcnow import utcnow

GLOBAL_STRATEGY_SCOPE = "__all__"
_last_loaded_revision_stamps: dict[str, str] = {}


def _normalize_source_keys(source_keys: list[str] | None) -> list[str]:
    return sorted(
        {str(source_key or "").strip().lower() for source_key in (source_keys or []) if str(source_key or "").strip()}
    )


def _scope_cache_key(source_keys: list[str] | None) -> str:
    normalized = _normalize_source_keys(source_keys)
    if not normalized:
        return GLOBAL_STRATEGY_SCOPE
    return "|".join(normalized)


async def _read_revision_map(
    session: AsyncSession,
    scopes: list[str],
) -> dict[str, int]:
    if not scopes:
        return {}
    rows = (
        (await session.execute(select(StrategyRuntimeRevision).where(StrategyRuntimeRevision.scope.in_(scopes))))
        .scalars()
        .all()
    )
    return {str(row.scope): int(row.revision or 0) for row in rows}


async def read_strategy_revision_stamp(
    session: AsyncSession,
    *,
    source_keys: list[str] | None = None,
) -> str:
    normalized = _normalize_source_keys(source_keys)
    if not normalized:
        revisions = await _read_revision_map(session, [GLOBAL_STRATEGY_SCOPE])
        return str(int(revisions.get(GLOBAL_STRATEGY_SCOPE, 0)))

    revisions = await _read_revision_map(session, normalized)
    return "|".join(f"{source}:{int(revisions.get(source, 0))}" for source in normalized)


async def bump_strategy_runtime_revisions(
    session: AsyncSession,
    *,
    source_keys: list[str] | None = None,
    commit: bool = False,
) -> dict[str, int]:
    normalized_sources = _normalize_source_keys(source_keys)
    targets = [GLOBAL_STRATEGY_SCOPE, *normalized_sources]
    now = utcnow()

    out: dict[str, int] = {}
    for scope in targets:
        row = await session.get(StrategyRuntimeRevision, scope)
        if row is None:
            row = StrategyRuntimeRevision(
                scope=scope,
                revision=1,
                updated_at=now,
            )
            session.add(row)
            out[scope] = 1
            continue

        row.revision = int(row.revision or 0) + 1
        row.updated_at = now
        out[scope] = int(row.revision)

    if commit:
        await session.commit()
    return out


async def refresh_strategy_runtime_if_needed(
    session: AsyncSession,
    *,
    source_keys: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    normalized_sources = _normalize_source_keys(source_keys)
    cache_key = _scope_cache_key(source_keys)
    revision_stamp = await read_strategy_revision_stamp(
        session,
        source_keys=source_keys,
    )
    effective_stamp = revision_stamp
    previous_stamp = _last_loaded_revision_stamps.get(cache_key)

    scope_fully_loaded = True
    if normalized_sources:
        enabled_rows = (
            (
                await session.execute(
                    select(Strategy.slug).where(
                        Strategy.enabled == True,  # noqa: E712
                        func.lower(func.coalesce(Strategy.source_key, "")).in_(tuple(normalized_sources)),
                    )
                )
            )
            .scalars()
            .all()
        )
        for slug in enabled_rows:
            normalized_slug = str(slug or "").strip().lower()
            if not normalized_slug:
                continue
            if strategy_loader.get_strategy(normalized_slug) is None:
                scope_fully_loaded = False
                break

    if not force and previous_stamp == effective_stamp and scope_fully_loaded:
        return {
            "refreshed": False,
            "revision_stamp": revision_stamp,
            "loaded": sorted(strategy_loader._loaded.keys()),
            "errors": dict(strategy_loader._errors),
        }

    loaded = await strategy_loader.refresh_all_from_db(
        session=session,
        source_keys=normalized_sources or None,
        # In a shared in-process runtime, source-scoped refreshes must not
        # unload strategies owned by other workers.
        prune_unlisted=not bool(normalized_sources),
    )
    _last_loaded_revision_stamps[cache_key] = effective_stamp

    return {
        "refreshed": True,
        "revision_stamp": revision_stamp,
        "loaded": loaded.get("loaded", []),
        "errors": loaded.get("errors", {}),
    }


async def preload_strategy_runtime(*, session: AsyncSession | None = None) -> dict[str, Any]:
    async def _run(target_session: AsyncSession) -> dict[str, Any]:
        seeded = await ensure_all_strategies_seeded(target_session)
        loaded = await refresh_strategy_runtime_if_needed(
            target_session,
            force=True,
        )
        loaded_keys = list(loaded.get("loaded", []))
        errors = dict(loaded.get("errors", {}))
        return {
            "seeded": int(seeded.get("seeded", 0) or 0),
            "loaded": loaded_keys,
            "errors": errors,
            "loaded_count": len(loaded_keys),
            "error_count": len(errors),
            "refreshed": bool(loaded.get("refreshed", False)),
            "revision_stamp": loaded.get("revision_stamp"),
        }

    if session is not None:
        return await _run(session)

    async with AsyncSessionLocal() as local_session:
        return await _run(local_session)
