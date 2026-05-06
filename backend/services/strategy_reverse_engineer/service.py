"""Job lifecycle for strategy reverse-engineering.

Thin layer over the database tables — the heavy lifting lives in
:mod:`agent`.  Used by:
  * the API route layer (POST /strategy-reverse-engineer/jobs)
  * the worker loop (workers/strategy_reverse_engineer_worker.py)
  * the AI tool wrapper (services/ai/tools/...)
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Optional

from sqlalchemy import select, update

from models.database import (
    AsyncSessionLocal,
    Strategy,
    StrategyReverseEngineerIteration,
    StrategyReverseEngineerJob,
)
from utils.utcnow import utcnow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


async def enqueue_job(
    *,
    wallet_address: str,
    label: Optional[str] = None,
    report_mode: str = "report",
    data_source_kind: str = "auto",
    recording_session_ids: Optional[list[str]] = None,
    provider_dataset_ids: Optional[list[str]] = None,
    llm_model: Optional[str] = None,
    max_iterations: Optional[int] = None,
    target_score: Optional[float] = None,
    max_cost_usd: Optional[float] = None,
    max_wallet_trades: Optional[int] = None,
) -> StrategyReverseEngineerJob:
    """Persist a queued reverse-engineer job; the worker picks it up."""
    address = (wallet_address or "").strip().lower()
    if not address:
        raise ValueError("wallet_address is required")

    if data_source_kind not in {"auto", "recording_session", "provider_dataset", "live"}:
        raise ValueError(
            f"Unknown data_source_kind '{data_source_kind}' "
            "(allowed: auto | recording_session | provider_dataset | live)"
        )
    if report_mode not in {"report", "strategy_seed"}:
        raise ValueError(
            f"Unknown report_mode '{report_mode}' (allowed: report | strategy_seed)"
        )

    # Resolve operator-tunable defaults from AppSettings (no hardcoded
    # values baked into this module per the no-hardcoded-defaults policy).
    defaults = await _load_defaults()
    job_id = f"re-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
    async with AsyncSessionLocal() as session:
        row = StrategyReverseEngineerJob(
            id=job_id,
            wallet_address=address,
            label=(label or "").strip() or None,
            report_mode=report_mode,
            data_source_kind=data_source_kind,
            recording_session_ids_json=list(recording_session_ids or []) or None,
            provider_dataset_ids_json=list(provider_dataset_ids or []) or None,
            llm_model=(llm_model or defaults.get("llm_model")),
            max_iterations=int(
                max_iterations
                if max_iterations is not None
                else defaults.get("max_iterations") or 10
            ),
            target_score=float(
                target_score
                if target_score is not None
                else defaults.get("target_score") or 0.7
            ),
            max_cost_usd=(
                float(max_cost_usd)
                if max_cost_usd is not None
                else defaults.get("max_cost_usd")
            ),
            max_wallet_trades=int(
                max_wallet_trades
                if max_wallet_trades is not None
                else defaults.get("max_wallet_trades") or 50_000
            ),
            status="queued",
            progress=0.0,
            current_iteration=0,
            activity="Queued — waiting for reverse-engineer worker",
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return row


async def _load_defaults() -> dict[str, Any]:
    """Load operator-tunable defaults for new reverse-engineer jobs.

    The model selection lives in the canonical
    ``llm_model_assignments['strategy_reverse_engineer']`` JSON entry —
    same place every other per-purpose model override lives — so the
    AI → Models view manages it uniformly.  All other knobs use the
    dedicated columns added in migration 202605040001.
    """
    from models.database import AppSettings

    async with AsyncSessionLocal() as session:
        row = (await session.execute(select(AppSettings))).scalar_one_or_none()
    if row is None:
        return {}
    assignments = getattr(row, "llm_model_assignments", None) or {}
    llm_model = None
    if isinstance(assignments, dict):
        llm_model = assignments.get("strategy_reverse_engineer")
    return {
        "llm_model": llm_model,
        "max_iterations": getattr(row, "reverse_engineer_max_iterations", None),
        "target_score": getattr(row, "reverse_engineer_target_score", None),
        "max_cost_usd": getattr(row, "reverse_engineer_max_cost_usd", None),
        "max_wallet_trades": getattr(row, "reverse_engineer_max_wallet_trades", None),
    }


# ---------------------------------------------------------------------------
# Worker dispatch
# ---------------------------------------------------------------------------


async def claim_next_queued_job() -> Optional[StrategyReverseEngineerJob]:
    """Atomic FIFO claim — same pattern as the provider import worker."""
    async with AsyncSessionLocal() as session:
        try:
            stmt = (
                update(StrategyReverseEngineerJob)
                .where(
                    StrategyReverseEngineerJob.id.in_(
                        select(StrategyReverseEngineerJob.id)
                        .where(StrategyReverseEngineerJob.status == "queued")
                        .order_by(StrategyReverseEngineerJob.created_at.asc())
                        .limit(1)
                        .with_for_update(skip_locked=True)
                    )
                )
                .values(
                    status="running",
                    started_at=utcnow(),
                    activity="Worker claimed job",
                )
                .returning(StrategyReverseEngineerJob)
            )
            row = (await session.execute(stmt)).scalar_one_or_none()
            await session.commit()
            return row
        except Exception:
            await session.rollback()

        candidate = (
            await session.execute(
                select(StrategyReverseEngineerJob)
                .where(StrategyReverseEngineerJob.status == "queued")
                .order_by(StrategyReverseEngineerJob.created_at.asc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if candidate is None:
            return None
        candidate.status = "running"
        candidate.started_at = utcnow()
        candidate.activity = "Worker claimed job"
        await session.commit()
        await session.refresh(candidate)
        return candidate


async def run_job(job_id: str) -> dict[str, Any]:
    """Execute the job's runner end-to-end.

    Routes by ``report_mode``:
      * 'report'        → analytical report runner (deterministic
        analytics + section-by-section LLM narrative)
      * 'strategy_seed' → LLM agent loop that synthesizes BaseStrategy
        Python and iteratively backtests it
    """
    job = await get_job(job_id)
    mode = (getattr(job, "report_mode", None) or "report") if job else "report"

    if mode == "report":
        from services.strategy_reverse_engineer.analytical_report_runner import (
            run_analytical_report,
        )
        runner = run_analytical_report
    else:
        from services.strategy_reverse_engineer.agent import run_reverse_engineer_agent
        runner = run_reverse_engineer_agent

    try:
        return await runner(job_id)
    except Exception as exc:
        logger.exception("reverse-engineer job %s crashed", job_id)
        async with AsyncSessionLocal() as session:
            row = (
                await session.execute(
                    select(StrategyReverseEngineerJob).where(
                        StrategyReverseEngineerJob.id == job_id
                    )
                )
            ).scalar_one_or_none()
            if row is not None:
                row.status = "failed"
                row.error = str(exc)
                row.activity = str(exc)[:200]
                row.finished_at = utcnow()
                await session.commit()
        return {"job_id": job_id, "status": "failed", "error": str(exc)}


async def cancel_job(job_id: str) -> bool:
    async with AsyncSessionLocal() as session:
        row = (
            await session.execute(
                select(StrategyReverseEngineerJob).where(
                    StrategyReverseEngineerJob.id == job_id
                )
            )
        ).scalar_one_or_none()
        if row is None:
            return False
        if row.status not in {"queued", "running", "profiling", "importing_data"}:
            return False
        row.status = "cancelled"
        row.activity = "Cancelled by operator"
        row.finished_at = utcnow()
        await session.commit()
        return True


async def delete_job(job_id: str) -> bool:
    """Delete a reverse-engineer job + all its iteration rows.

    Allowed in any status — operator can wipe a stuck/half-broken job
    cleanly without leaving dependent iteration rows orphaned.  Returns
    False when the job_id doesn't exist (404 surface).
    """
    from sqlalchemy import delete as sa_delete
    from models.database import StrategyReverseEngineerIteration

    async with AsyncSessionLocal() as session:
        # Iterations first to satisfy any FK ordering — though the
        # schema doesn't enforce a hard FK, deleting children first
        # is the institutional safe-default.
        await session.execute(
            sa_delete(StrategyReverseEngineerIteration).where(
                StrategyReverseEngineerIteration.job_id == job_id
            )
        )
        result = await session.execute(
            sa_delete(StrategyReverseEngineerJob).where(
                StrategyReverseEngineerJob.id == job_id
            )
        )
        deleted = (result.rowcount or 0) > 0
        await session.commit()
        return deleted


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


async def list_jobs(
    *,
    wallet_address: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
) -> list[StrategyReverseEngineerJob]:
    async with AsyncSessionLocal() as session:
        stmt = select(StrategyReverseEngineerJob)
        if wallet_address:
            stmt = stmt.where(
                StrategyReverseEngineerJob.wallet_address == wallet_address.lower()
            )
        if status:
            stmt = stmt.where(StrategyReverseEngineerJob.status == status)
        stmt = stmt.order_by(StrategyReverseEngineerJob.created_at.desc()).limit(int(limit))
        return list((await session.execute(stmt)).scalars().all())


async def get_job(job_id: str) -> Optional[StrategyReverseEngineerJob]:
    async with AsyncSessionLocal() as session:
        return (
            await session.execute(
                select(StrategyReverseEngineerJob).where(
                    StrategyReverseEngineerJob.id == job_id
                )
            )
        ).scalar_one_or_none()


async def list_iterations(job_id: str) -> list[StrategyReverseEngineerIteration]:
    async with AsyncSessionLocal() as session:
        stmt = (
            select(StrategyReverseEngineerIteration)
            .where(StrategyReverseEngineerIteration.job_id == job_id)
            .order_by(StrategyReverseEngineerIteration.iteration.asc())
        )
        return list((await session.execute(stmt)).scalars().all())


async def get_iteration(iteration_id: str) -> Optional[StrategyReverseEngineerIteration]:
    async with AsyncSessionLocal() as session:
        return (
            await session.execute(
                select(StrategyReverseEngineerIteration).where(
                    StrategyReverseEngineerIteration.id == iteration_id
                )
            )
        ).scalar_one_or_none()


# ---------------------------------------------------------------------------
# Promote to strategy library
# ---------------------------------------------------------------------------


async def promote_to_strategy_library(
    job_id: str,
    *,
    name: str,
    slug: str,
    description: Optional[str] = None,
    enabled: bool = False,
) -> dict[str, Any]:
    """Persist the winning strategy as a Strategy row.

    Returns the new strategy's id + slug.  Disabled by default — the
    operator should review and enable manually before promoting to live.
    """
    job = await get_job(job_id)
    if job is None:
        raise ValueError(f"job '{job_id}' not found")
    if not job.best_strategy_code:
        raise ValueError("job has no winning strategy yet — finalize the agent first")

    slug_clean = (slug or "").strip().lower().replace(" ", "_")
    if not slug_clean:
        raise ValueError("slug is required")
    name_clean = (name or "").strip() or slug_clean

    async with AsyncSessionLocal() as session:
        existing = (
            await session.execute(select(Strategy).where(Strategy.slug == slug_clean))
        ).scalar_one_or_none()
        if existing is not None:
            raise ValueError(f"strategy slug '{slug_clean}' already exists")

        strategy_id = f"strat-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"
        row = Strategy(
            id=strategy_id,
            slug=slug_clean,
            name=name_clean,
            description=description or f"Reverse-engineered from wallet {job.wallet_address}",
            source_code=job.best_strategy_code,
            source_key="manual",
            enabled=bool(enabled),
        )
        session.add(row)
        await session.execute(
            update(StrategyReverseEngineerJob)
            .where(StrategyReverseEngineerJob.id == job_id)
            .values(promoted_strategy_id=strategy_id)
        )
        await session.commit()

    return {
        "strategy_id": strategy_id,
        "slug": slug_clean,
        "name": name_clean,
        "enabled": bool(enabled),
    }


# ---------------------------------------------------------------------------
# Serialization for API
# ---------------------------------------------------------------------------


def serialize_job(row: StrategyReverseEngineerJob) -> dict[str, Any]:
    return {
        "id": row.id,
        "wallet_address": row.wallet_address,
        "label": row.label,
        "report_mode": getattr(row, "report_mode", "report") or "report",
        "data_source_kind": row.data_source_kind,
        "recording_session_ids": list(row.recording_session_ids_json or []),
        "provider_dataset_ids": list(row.provider_dataset_ids_json or []),
        "llm_model": row.llm_model,
        "max_iterations": int(row.max_iterations or 0),
        "target_score": float(row.target_score or 0.0),
        "max_cost_usd": float(row.max_cost_usd) if row.max_cost_usd is not None else None,
        "max_wallet_trades": int(row.max_wallet_trades or 0),
        "status": row.status,
        "progress": float(row.progress or 0.0),
        "current_iteration": int(row.current_iteration or 0),
        "activity": row.activity,
        "error": row.error,
        "wallet_profile": row.wallet_profile_json,
        "wallet_trade_count": int(row.wallet_trade_count or 0),
        "wallet_window_start": row.wallet_window_start.isoformat() if row.wallet_window_start else None,
        "wallet_window_end": row.wallet_window_end.isoformat() if row.wallet_window_end else None,
        "best_iteration_id": row.best_iteration_id,
        "best_score": float(row.best_score) if row.best_score is not None else None,
        "best_strategy_class": row.best_strategy_class,
        "best_strategy_code": row.best_strategy_code,
        "best_backtest_run_id": row.best_backtest_run_id,
        "total_input_tokens": int(row.total_input_tokens or 0),
        "total_output_tokens": int(row.total_output_tokens or 0),
        "total_cost_usd": float(row.total_cost_usd or 0.0),
        "promoted_strategy_id": row.promoted_strategy_id,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
    }


def serialize_iteration(row: StrategyReverseEngineerIteration) -> dict[str, Any]:
    return {
        "id": row.id,
        "job_id": row.job_id,
        "iteration": int(row.iteration or 0),
        "status": row.status,
        "strategy_class": row.strategy_class,
        "strategy_code": row.strategy_code,
        "backtest_run_id": row.backtest_run_id,
        "score": float(row.score) if row.score is not None else None,
        "score_breakdown": row.score_breakdown_json,
        "divergence_summary": row.divergence_summary,
        "llm_critique": row.llm_critique,
        "notes": row.notes,
        "error": row.error,
        "input_tokens": int(row.input_tokens or 0),
        "output_tokens": int(row.output_tokens or 0),
        "cost_usd": float(row.cost_usd or 0.0),
        "duration_ms": float(row.duration_ms or 0.0),
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "completed_at": row.completed_at.isoformat() if row.completed_at else None,
    }
