"""Strategy reverse-engineer API.

Routes mounted at ``/api/strategy-reverse-engineer``.  Powers:
  * Strategy Research → Reverse Engineer tab (full UI)
  * Wallet Analysis Panel → "Reverse Engineer Strategy" action link
  * MCP / agent tool surface (see services/ai/tools/...)

The actual agent loop runs on the discovery plane via
``workers/strategy_reverse_engineer_worker.py``.  These routes are
thin wrappers over ``services/strategy_reverse_engineer/service.py``
plus the PDF report renderer.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field, field_validator

from services.strategy_reverse_engineer import service as re_service
from utils.validation import validate_eth_address

logger = logging.getLogger("routes.strategy_reverse_engineer")
router = APIRouter(
    prefix="/strategy-reverse-engineer",
    tags=["Strategy Reverse-Engineer"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class CreateJobRequest(BaseModel):
    wallet_address: str
    label: Optional[str] = None
    # 'report' (default) → analytical multi-section PDF report;
    # 'strategy_seed'   → LLM agent loop synthesizes BaseStrategy code.
    report_mode: str = Field(default="report")
    data_source_kind: str = Field(default="auto")
    recording_session_ids: Optional[list[str]] = None
    provider_dataset_ids: Optional[list[str]] = None
    llm_model: Optional[str] = None
    max_iterations: Optional[int] = Field(default=None, ge=1, le=100)
    target_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_cost_usd: Optional[float] = Field(default=None, ge=0.0)
    max_wallet_trades: Optional[int] = Field(default=None, ge=10, le=250_000)

    @field_validator("wallet_address")
    @classmethod
    def _check_wallet(cls, v: str) -> str:
        return validate_eth_address(v)


class PromoteRequest(BaseModel):
    name: str
    slug: str
    description: Optional[str] = None
    enabled: bool = False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/jobs")
async def create_job(req: CreateJobRequest) -> dict[str, Any]:
    """Enqueue a new reverse-engineer job.

    The discovery-plane worker picks it up within a few seconds.  Poll
    ``GET /strategy-reverse-engineer/jobs/{job_id}`` for status +
    progress.
    """
    try:
        job = await re_service.enqueue_job(
            wallet_address=req.wallet_address,
            label=req.label,
            report_mode=req.report_mode,
            data_source_kind=req.data_source_kind,
            recording_session_ids=req.recording_session_ids,
            provider_dataset_ids=req.provider_dataset_ids,
            llm_model=req.llm_model,
            max_iterations=req.max_iterations,
            target_score=req.target_score,
            max_cost_usd=req.max_cost_usd,
            max_wallet_trades=req.max_wallet_trades,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return re_service.serialize_job(job)


@router.get("/jobs")
async def list_jobs(
    wallet_address: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    rows = await re_service.list_jobs(
        wallet_address=wallet_address.lower() if wallet_address else None,
        status=status,
        limit=limit,
    )
    return {"jobs": [re_service.serialize_job(r) for r in rows]}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    row = await re_service.get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return re_service.serialize_job(row)


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict[str, Any]:
    ok = await re_service.cancel_job(job_id)
    if not ok:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found or no longer cancellable",
        )
    return {"cancelled": True, "id": job_id}


@router.get("/jobs/{job_id}/iterations")
async def list_iterations(job_id: str) -> dict[str, Any]:
    row = await re_service.get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    rows = await re_service.list_iterations(job_id)
    return {
        "job_id": job_id,
        "iterations": [re_service.serialize_iteration(r) for r in rows],
    }


@router.get("/jobs/{job_id}/iterations/{iteration_id}")
async def get_iteration(job_id: str, iteration_id: str) -> dict[str, Any]:
    row = await re_service.get_iteration(iteration_id)
    if row is None or row.job_id != job_id:
        raise HTTPException(status_code=404, detail=f"Iteration '{iteration_id}' not found for job '{job_id}'")
    return re_service.serialize_iteration(row)


@router.post("/jobs/{job_id}/promote")
async def promote_to_strategy(job_id: str, req: PromoteRequest) -> dict[str, Any]:
    """Persist the winning strategy as a Strategy row in the library."""
    try:
        return await re_service.promote_to_strategy_library(
            job_id,
            name=req.name,
            slug=req.slug,
            description=req.description,
            enabled=req.enabled,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/jobs/{job_id}/report.pdf")
async def get_pdf_report(job_id: str) -> Response:
    """Render the executive PDF report for a finished job.

    Routes by ``report_mode``:
      * 'report'        → multi-section analytical report
      * 'strategy_seed' → legacy iteration-history report
    """
    row = await re_service.get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    mode = (getattr(row, "report_mode", None) or "report").lower()
    try:
        from services.reports.wallet_strategy_report import (
            ReportRenderError,
            render_analytical_report,
            render_wallet_strategy_report,
        )

        if mode == "report":
            pdf_bytes = await _render_analytical(row, render_analytical_report)
        else:
            iterations = await re_service.list_iterations(job_id)
            pdf_bytes = render_wallet_strategy_report(job=row, iterations=iterations)
    except ReportRenderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("PDF report render failed for job %s", job_id)
        raise HTTPException(status_code=500, detail=f"PDF render failed: {exc}") from exc
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="reverse_engineer_{job_id}.pdf"',
        },
    )


async def _render_analytical(row: Any, render_fn) -> bytes:
    """Reconstruct the WalletAnalytics + sections payload from the job row."""
    import json
    from types import SimpleNamespace

    profile = row.wallet_profile_json or {}
    raw_payload: dict[str, Any] = {}
    if row.best_strategy_code:
        try:
            raw_payload = json.loads(row.best_strategy_code)
        except (TypeError, ValueError):
            raw_payload = {}
    sections_dict = (raw_payload or {}).get("sections") or {}
    spotlight = (raw_payload or {}).get("spotlight")

    # Walk the analytics dict and re-coerce nested dataclass-shaped dicts
    # back into objects with attribute access (the Jinja template uses
    # dotted paths like ``analytics.headline.total_trades``).
    analytics = _dict_to_namespace(profile)
    sections = _dict_to_namespace(sections_dict)
    return render_fn(analytics=analytics, sections=sections, spotlight=spotlight)


def _dict_to_namespace(obj: Any) -> Any:
    """Recursive dict → SimpleNamespace conversion (lists pass through)."""
    from types import SimpleNamespace

    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_dict_to_namespace(x) for x in obj]
    return obj
