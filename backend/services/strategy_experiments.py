from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import and_, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import Strategy, StrategyExperiment, StrategyExperimentAssignment, StrategyVersion, Trader
from services.strategy_versioning import normalize_strategy_version
from utils.utcnow import utcnow

_EXPERIMENT_STATUSES = {"active", "paused", "completed", "archived"}


# ─── R11-C: get_active_strategy_experiment TTL cache ────────────────────
#
# ``get_active_strategy_experiment`` runs a filtered SELECT with three
# ``lower(coalesce())`` comparisons + ORDER BY created_at DESC on every
# trader cycle.  Soak instrumentation showed ``prs_get_experiment =
# 609 ms`` under pool pressure.
#
# Experiments change at operator-save cadence (minutes to hours), not
# per cycle.  The vast majority of (source_key, strategy_key) pairs
# have NO active experiment, so we cache negative hits as well
# (experiment_id=None).  The cached path collapses to either a no-op
# (negative cache) or a single ``session.get(StrategyExperiment, id)``
# PK lookup that hits the identity map when the session has already
# loaded it.
#
# Invalidation: ``invalidate_experiment_cache()`` must be called from
# every write path that creates, status-changes, or promotes a
# StrategyExperiment row.  TTL acts as a safety net; operators can
# tolerate at most ``_EXPERIMENT_CACHE_TTL_SECONDS`` of staleness
# before a newly-activated experiment takes effect.
_EXPERIMENT_CACHE_TTL_SECONDS: float = 60.0


@dataclass
class _CachedExperiment:
    experiment_id: str | None  # None = negative cache (no active experiment)
    expires_at: float


_experiment_cache: dict[tuple[str, str], _CachedExperiment] = {}


def invalidate_experiment_cache(
    source_key: str | None = None,
    strategy_key: str | None = None,
) -> None:
    """Drop cached active-experiment pointers.

    If neither key is supplied, the entire cache is cleared.  If only
    one is supplied, every entry matching that dimension is dropped.
    Callers that touch a StrategyExperiment row should invalidate both
    the (source_key, strategy_key) of the affected experiment.
    """
    global _experiment_cache
    if source_key is None and strategy_key is None:
        _experiment_cache.clear()
        return
    normalized_source = _normalize_source_key(source_key) if source_key is not None else None
    normalized_strategy = _normalize_strategy_key(strategy_key) if strategy_key is not None else None
    _experiment_cache = {
        k: v
        for k, v in _experiment_cache.items()
        if (normalized_source is not None and k[0] != normalized_source)
        or (normalized_strategy is not None and k[1] != normalized_strategy)
    }


def _normalize_source_key(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_strategy_key(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    if status not in _EXPERIMENT_STATUSES:
        raise ValueError("status must be one of: active, paused, completed, archived")
    return status


def _normalize_allocation_pct(value: Any) -> float:
    try:
        parsed = float(value)
    except Exception as exc:
        raise ValueError("candidate_allocation_pct must be a number") from exc
    if parsed <= 0.0 or parsed >= 100.0:
        raise ValueError("candidate_allocation_pct must be between 0 and 100")
    return parsed


def serialize_strategy_experiment(row: StrategyExperiment) -> dict[str, Any]:
    return {
        "id": row.id,
        "name": row.name,
        "source_key": row.source_key,
        "strategy_key": row.strategy_key,
        "control_version": int(row.control_version or 1),
        "candidate_version": int(row.candidate_version or 1),
        "candidate_allocation_pct": float(row.candidate_allocation_pct or 50.0),
        "scope": dict(row.scope_json or {}),
        "status": str(row.status or "active").strip().lower() or "active",
        "created_by": row.created_by,
        "notes": row.notes,
        "metadata": dict(row.metadata_json or {}),
        "promoted_version": int(row.promoted_version) if row.promoted_version is not None else None,
        "ended_at": row.ended_at.isoformat() if row.ended_at else None,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def serialize_strategy_experiment_assignment(row: StrategyExperimentAssignment) -> dict[str, Any]:
    return {
        "id": row.id,
        "experiment_id": row.experiment_id,
        "trader_id": row.trader_id,
        "signal_id": row.signal_id,
        "decision_id": row.decision_id,
        "order_id": row.order_id,
        "source_key": row.source_key,
        "strategy_key": row.strategy_key,
        "strategy_version": int(row.strategy_version or 1),
        "assignment_group": row.assignment_group,
        "payload": dict(row.payload_json or {}),
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


async def _get_strategy_row(session: AsyncSession, *, strategy_key: str) -> Strategy | None:
    normalized_strategy = _normalize_strategy_key(strategy_key)
    if not normalized_strategy:
        return None
    return (
        (
            await session.execute(
                select(Strategy).where(func.lower(func.coalesce(Strategy.slug, "")) == normalized_strategy).limit(1)
            )
        )
        .scalars()
        .first()
    )


async def _strategy_has_version(
    session: AsyncSession,
    *,
    strategy_key: str,
    version: int,
) -> bool:
    strategy_row = await _get_strategy_row(session, strategy_key=strategy_key)
    if strategy_row is None:
        return False
    if int(strategy_row.version or 1) == int(version):
        return True
    match = (
        (
            await session.execute(
                select(StrategyVersion.id)
                .where(
                    and_(
                        StrategyVersion.strategy_id == strategy_row.id,
                        StrategyVersion.version == int(version),
                    )
                )
                .limit(1)
            )
        )
        .scalars()
        .first()
    )
    return match is not None


async def create_strategy_experiment(
    session: AsyncSession,
    *,
    name: str,
    source_key: str,
    strategy_key: str,
    control_version: int,
    candidate_version: int,
    candidate_allocation_pct: float = 50.0,
    scope: dict[str, Any] | None = None,
    notes: str | None = None,
    created_by: str | None = None,
    metadata: dict[str, Any] | None = None,
    commit: bool = True,
) -> StrategyExperiment:
    normalized_source = _normalize_source_key(source_key)
    normalized_strategy = _normalize_strategy_key(strategy_key)
    if not normalized_source:
        raise ValueError("source_key is required")
    if not normalized_strategy:
        raise ValueError("strategy_key is required")
    normalized_name = str(name or "").strip()
    if len(normalized_name) < 2:
        raise ValueError("name must be at least 2 characters")

    control = normalize_strategy_version(control_version)
    candidate = normalize_strategy_version(candidate_version)
    if control is None or candidate is None:
        raise ValueError("control_version and candidate_version must be explicit integers")
    if int(control) == int(candidate):
        raise ValueError("control_version and candidate_version must be different")

    if not await _strategy_has_version(session, strategy_key=normalized_strategy, version=int(control)):
        raise ValueError(f"Strategy '{normalized_strategy}' does not have version v{int(control)}")
    if not await _strategy_has_version(session, strategy_key=normalized_strategy, version=int(candidate)):
        raise ValueError(f"Strategy '{normalized_strategy}' does not have version v{int(candidate)}")

    allocation_pct = _normalize_allocation_pct(candidate_allocation_pct)
    now = utcnow()
    row = StrategyExperiment(
        id=uuid.uuid4().hex,
        name=normalized_name,
        source_key=normalized_source,
        strategy_key=normalized_strategy,
        control_version=int(control),
        candidate_version=int(candidate),
        candidate_allocation_pct=allocation_pct,
        scope_json=dict(scope or {}),
        status="active",
        created_by=(str(created_by or "").strip() or None),
        notes=notes,
        metadata_json=dict(metadata or {}),
        promoted_version=None,
        ended_at=None,
        created_at=now,
        updated_at=now,
    )
    session.add(row)
    await session.flush()
    if commit:
        await session.commit()
        await session.refresh(row)
    # R11-C: new experiment (active) — drop negative/stale cache entries.
    try:
        invalidate_experiment_cache(
            source_key=normalized_source,
            strategy_key=normalized_strategy,
        )
    except Exception:
        pass
    return row


async def list_strategy_experiments(
    session: AsyncSession,
    *,
    source_key: str | None = None,
    strategy_key: str | None = None,
    status: str | None = None,
    limit: int = 200,
) -> list[StrategyExperiment]:
    query = select(StrategyExperiment).order_by(StrategyExperiment.created_at.desc())
    if source_key is not None:
        query = query.where(
            func.lower(func.coalesce(StrategyExperiment.source_key, "")) == _normalize_source_key(source_key)
        )
    if strategy_key is not None:
        query = query.where(
            func.lower(func.coalesce(StrategyExperiment.strategy_key, "")) == _normalize_strategy_key(strategy_key)
        )
    if status is not None:
        query = query.where(func.lower(func.coalesce(StrategyExperiment.status, "")) == _normalize_status(status))
    query = query.limit(max(1, min(int(limit or 200), 2000)))
    rows = (await session.execute(query)).scalars().all()
    return list(rows)


async def get_strategy_experiment(session: AsyncSession, *, experiment_id: str) -> StrategyExperiment | None:
    return await session.get(StrategyExperiment, str(experiment_id or "").strip())


async def set_strategy_experiment_status(
    session: AsyncSession,
    *,
    experiment_id: str,
    status: str,
    commit: bool = True,
) -> StrategyExperiment | None:
    row = await get_strategy_experiment(session, experiment_id=experiment_id)
    if row is None:
        return None
    normalized_status = _normalize_status(status)
    now = utcnow()
    row.status = normalized_status
    if normalized_status in {"completed", "archived"} and row.ended_at is None:
        row.ended_at = now
    if normalized_status in {"active", "paused"}:
        row.ended_at = None
    row.updated_at = now
    await session.flush()
    if commit:
        await session.commit()
        await session.refresh(row)
    # R11-C: status transition flips (possibly) the active experiment
    # for this (source_key, strategy_key) — drop the cache entry so
    # the next resolve re-reads.
    try:
        invalidate_experiment_cache(
            source_key=str(row.source_key or ""),
            strategy_key=str(row.strategy_key or ""),
        )
    except Exception:
        pass
    return row


def _set_source_strategy_version_on_configs(
    *,
    source_configs: Any,
    source_key: str,
    strategy_key: str,
    strategy_version: int,
) -> tuple[list[dict[str, Any]], bool]:
    raw_configs = source_configs if isinstance(source_configs, list) else []
    normalized_source = _normalize_source_key(source_key)
    normalized_strategy = _normalize_strategy_key(strategy_key)
    updated = False
    next_configs: list[dict[str, Any]] = []
    for raw_item in raw_configs:
        item = dict(raw_item) if isinstance(raw_item, dict) else {}
        item_source = _normalize_source_key(item.get("source_key"))
        item_strategy = _normalize_strategy_key(item.get("strategy_key"))
        if item_source == normalized_source and item_strategy == normalized_strategy:
            if item.get("strategy_version") != int(strategy_version):
                item["strategy_version"] = int(strategy_version)
                updated = True
        next_configs.append(item)
    return next_configs, updated


async def promote_strategy_experiment(
    session: AsyncSession,
    *,
    experiment_id: str,
    promoted_version: int | None = None,
    notes: str | None = None,
    commit: bool = True,
) -> StrategyExperiment | None:
    row = await get_strategy_experiment(session, experiment_id=experiment_id)
    if row is None:
        return None

    target_version = normalize_strategy_version(promoted_version)
    if target_version is None:
        target_version = int(row.candidate_version or 1)
    if not await _strategy_has_version(
        session,
        strategy_key=str(row.strategy_key or ""),
        version=int(target_version),
    ):
        raise ValueError(f"Strategy '{str(row.strategy_key or '')}' does not have version v{int(target_version)}")

    applied_count = 0
    traders = (await session.execute(select(Trader))).scalars().all()
    for trader in traders:
        next_configs, changed = _set_source_strategy_version_on_configs(
            source_configs=trader.source_configs_json,
            source_key=row.source_key,
            strategy_key=row.strategy_key,
            strategy_version=int(target_version),
        )
        if not changed:
            continue
        trader.source_configs_json = next_configs
        trader.updated_at = utcnow()
        applied_count += 1

    metadata = dict(row.metadata_json or {})
    metadata["promotion"] = {
        "promoted_version": int(target_version),
        "applied_traders": applied_count,
        "promoted_at": utcnow().isoformat(),
        "notes": notes,
    }
    row.metadata_json = metadata
    row.promoted_version = int(target_version)
    row.status = "completed"
    if notes:
        row.notes = notes
    now = utcnow()
    row.ended_at = now
    row.updated_at = now
    await session.flush()
    if commit:
        await session.commit()
        await session.refresh(row)
    # R11-C: promotion flips status=completed — the cached "active"
    # pointer for this (source_key, strategy_key) is now stale.
    try:
        invalidate_experiment_cache(
            source_key=str(row.source_key or ""),
            strategy_key=str(row.strategy_key or ""),
        )
    except Exception:
        pass
    return row


async def list_strategy_experiment_assignments(
    session: AsyncSession,
    *,
    experiment_id: str,
    limit: int = 200,
) -> list[dict[str, Any]]:
    rows = (
        (
            await session.execute(
                select(StrategyExperimentAssignment)
                .where(StrategyExperimentAssignment.experiment_id == str(experiment_id or "").strip())
                .order_by(StrategyExperimentAssignment.created_at.desc())
                .limit(max(1, min(int(limit or 200), 2000)))
            )
        )
        .scalars()
        .all()
    )
    return [serialize_strategy_experiment_assignment(row) for row in rows]


async def get_active_strategy_experiment(
    session: AsyncSession,
    *,
    source_key: str,
    strategy_key: str,
) -> StrategyExperiment | None:
    # R11-C: most (source_key, strategy_key) pairs have NO active
    # experiment — instrument showed ``prs_get_experiment = 609 ms``
    # per cycle on that negative path.  Cache both positive and
    # negative results for ``_EXPERIMENT_CACHE_TTL_SECONDS`` so the
    # per-cycle cost drops to either zero SQL (negative hit) or a PK
    # ``session.get`` that usually hits the identity map.
    normalized_source = _normalize_source_key(source_key)
    normalized_strategy = _normalize_strategy_key(strategy_key)
    cache_key = (normalized_source, normalized_strategy)
    now_mono = time.monotonic()
    cached = _experiment_cache.get(cache_key)
    if cached is not None and cached.expires_at > now_mono:
        if cached.experiment_id is None:
            return None
        row = await session.get(StrategyExperiment, cached.experiment_id)
        # Validate the cached row is still "active".  If an operator
        # paused/completed/archived it within the TTL window we need
        # to fall through to the slow path so a different active
        # experiment (if any) is picked up.
        if row is not None and str(row.status or "").strip().lower() == "active":
            return row
        _experiment_cache.pop(cache_key, None)

    row = (
        (
            await session.execute(
                select(StrategyExperiment)
                .where(
                    and_(
                        func.lower(func.coalesce(StrategyExperiment.source_key, "")) == normalized_source,
                        func.lower(func.coalesce(StrategyExperiment.strategy_key, "")) == normalized_strategy,
                        func.lower(func.coalesce(StrategyExperiment.status, "")) == "active",
                    )
                )
                .order_by(StrategyExperiment.created_at.desc())
                .limit(1)
            )
        )
        .scalars()
        .first()
    )
    _experiment_cache[cache_key] = _CachedExperiment(
        experiment_id=str(row.id) if row is not None else None,
        expires_at=now_mono + _EXPERIMENT_CACHE_TTL_SECONDS,
    )
    return row


def _assignment_sample(*, experiment_id: str, trader_id: str, signal_id: str) -> float:
    key = f"{experiment_id}:{trader_id}:{signal_id}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    basis_points = int(digest[:8], 16) % 10000
    return float(basis_points) / 100.0


def resolve_experiment_assignment(
    *,
    experiment: StrategyExperiment,
    trader_id: str,
    signal_id: str,
) -> tuple[str, int, float]:
    sample = _assignment_sample(
        experiment_id=str(experiment.id or ""),
        trader_id=str(trader_id or ""),
        signal_id=str(signal_id or ""),
    )
    threshold = float(experiment.candidate_allocation_pct or 50.0)
    if sample < threshold:
        return "candidate", int(experiment.candidate_version or 1), sample
    return "control", int(experiment.control_version or 1), sample


async def upsert_strategy_experiment_assignment(
    session: AsyncSession,
    *,
    experiment_id: str,
    trader_id: str | None,
    signal_id: str | None,
    source_key: str,
    strategy_key: str,
    strategy_version: int,
    assignment_group: str,
    decision_id: str | None = None,
    order_id: str | None = None,
    payload: dict[str, Any] | None = None,
    commit: bool = False,
) -> StrategyExperimentAssignment:
    normalized_experiment_id = str(experiment_id or "").strip()
    normalized_trader_id = str(trader_id or "").strip() or None
    normalized_signal_id = str(signal_id or "").strip() or None
    now = utcnow()
    normalized_source = _normalize_source_key(source_key)
    normalized_strategy = _normalize_strategy_key(strategy_key)
    normalized_assignment_group = str(assignment_group or "control").strip().lower() or "control"
    normalized_decision_id = str(decision_id or "").strip() or None
    normalized_order_id = str(order_id or "").strip() or None
    normalized_payload = dict(payload or {})

    # Nullable unique-key fields do not participate in PostgreSQL ON CONFLICT matching.
    # Preserve prior semantics for nullable-key callers.
    if normalized_trader_id is None or normalized_signal_id is None:
        existing = (
            (
                await session.execute(
                    select(StrategyExperimentAssignment)
                    .where(
                        and_(
                            StrategyExperimentAssignment.experiment_id == normalized_experiment_id,
                            StrategyExperimentAssignment.trader_id == normalized_trader_id,
                            StrategyExperimentAssignment.signal_id == normalized_signal_id,
                        )
                    )
                    .limit(1)
                )
            )
            .scalars()
            .first()
        )
        if existing is not None:
            existing.source_key = normalized_source
            existing.strategy_key = normalized_strategy
            existing.strategy_version = int(strategy_version)
            existing.assignment_group = normalized_assignment_group
            existing.decision_id = normalized_decision_id or existing.decision_id
            existing.order_id = normalized_order_id or existing.order_id
            if normalized_payload:
                merged = dict(existing.payload_json or {})
                merged.update(normalized_payload)
                existing.payload_json = merged
            existing.updated_at = now
            await session.flush()
            if commit:
                await session.commit()
                await session.refresh(existing)
            return existing

        row = StrategyExperimentAssignment(
            id=uuid.uuid4().hex,
            experiment_id=normalized_experiment_id,
            trader_id=normalized_trader_id,
            signal_id=normalized_signal_id,
            decision_id=normalized_decision_id,
            order_id=normalized_order_id,
            source_key=normalized_source,
            strategy_key=normalized_strategy,
            strategy_version=int(strategy_version),
            assignment_group=normalized_assignment_group,
            payload_json=normalized_payload,
            created_at=now,
            updated_at=now,
        )
        session.add(row)
        await session.flush()
        if commit:
            await session.commit()
            await session.refresh(row)
        return row

    insert_stmt = pg_insert(StrategyExperimentAssignment).values(
        id=uuid.uuid4().hex,
        experiment_id=normalized_experiment_id,
        trader_id=normalized_trader_id,
        signal_id=normalized_signal_id,
        decision_id=normalized_decision_id,
        order_id=normalized_order_id,
        source_key=normalized_source,
        strategy_key=normalized_strategy,
        strategy_version=int(strategy_version),
        assignment_group=normalized_assignment_group,
        payload_json=normalized_payload,
        created_at=now,
        updated_at=now,
    )
    set_values: dict[str, Any] = {
        "source_key": insert_stmt.excluded.source_key,
        "strategy_key": insert_stmt.excluded.strategy_key,
        "strategy_version": insert_stmt.excluded.strategy_version,
        "assignment_group": insert_stmt.excluded.assignment_group,
        "decision_id": func.coalesce(insert_stmt.excluded.decision_id, StrategyExperimentAssignment.decision_id),
        "order_id": func.coalesce(insert_stmt.excluded.order_id, StrategyExperimentAssignment.order_id),
        "updated_at": now,
    }
    if normalized_payload:
        set_values["payload_json"] = insert_stmt.excluded.payload_json
    else:
        set_values["payload_json"] = StrategyExperimentAssignment.payload_json

    row = (
        await session.execute(
            insert_stmt.on_conflict_do_update(
                index_elements=[
                    StrategyExperimentAssignment.experiment_id,
                    StrategyExperimentAssignment.trader_id,
                    StrategyExperimentAssignment.signal_id,
                ],
                set_=set_values,
            ).returning(StrategyExperimentAssignment)
        )
    ).scalar_one()
    if commit:
        await session.commit()
        await session.refresh(row)
    return row

