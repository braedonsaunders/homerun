"""Admin / observability endpoints.

Phase 4 (gate-pipeline observability) adds:

* ``GET  /api/admin/gate-stats`` — return the live :class:`GateMetrics`
  snapshot (per-gate p50/p95/p99 latency, rejection rate, budget
  violations).  Stats are process-local and in-memory only: restart
  resets them, and a multi-process deploy will surface only the
  responding worker's stats.

* ``POST /api/admin/gate-stats/reset`` — clear stats.  Optional
  ``?gate=<name>`` query parameter resets just one gate; without it,
  all gates are cleared.  Useful when changing config and wanting a
  fresh observation window.

These endpoints are intentionally read-only on state outside of the
``GateMetrics`` singleton — no DB writes, no settings mutations.  They
exist so the operator UI can surface gate health without scraping
logs.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

from services.trader_orchestrator.gate_pipeline import (
    GateMetricsSnapshot,
    get_gate_metrics,
)
from utils.logger import get_logger
from utils.utcnow import utcnow

logger = get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["Admin"])


class GateStatsEntry(BaseModel):
    """One gate's stats as exposed via the admin API."""

    name: str
    cost_class: str
    runs: int
    rejects: int
    reject_rate: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    budget_ms: Optional[float] = None
    budget_violations: int
    auto_demoted: bool


class GateStatsResponse(BaseModel):
    """Top-level response shape for the gate-stats endpoint."""

    snapshot_at: str
    gates: list[GateStatsEntry]


def _snapshot_to_entry(snap: GateMetricsSnapshot) -> GateStatsEntry:
    return GateStatsEntry(
        name=snap.name,
        cost_class=snap.cost_class,
        runs=snap.runs,
        rejects=snap.rejects,
        reject_rate=snap.reject_rate,
        p50_ms=snap.p50_ms,
        p95_ms=snap.p95_ms,
        p99_ms=snap.p99_ms,
        max_ms=snap.max_ms,
        budget_ms=snap.budget_ms,
        budget_violations=snap.budget_violations,
        auto_demoted=snap.auto_demoted,
    )


@router.get("/gate-stats", response_model=GateStatsResponse)
async def get_gate_stats() -> GateStatsResponse:
    """Return the current per-gate metrics snapshot.

    Sort order: runs descending, so the gates being hit most often
    appear first.  This makes the response useful as a quick "what's
    the orchestrator doing right now" view without UI-side sorting.
    """
    metrics = get_gate_metrics()
    snaps = metrics.snapshot()
    snaps.sort(key=lambda s: s.runs, reverse=True)
    return GateStatsResponse(
        snapshot_at=utcnow().isoformat().replace("+00:00", "Z"),
        gates=[_snapshot_to_entry(s) for s in snaps],
    )


@router.post("/gate-stats/reset")
async def reset_gate_stats(
    gate: Optional[str] = Query(
        default=None,
        description="Reset only this gate by name.  Omit to reset every gate.",
    ),
) -> dict[str, object]:
    """Reset the gate metrics.

    Returns the gate name (or ``"*"``) and a count of gates that
    existed before the reset, so the operator UI can confirm the reset
    happened against the same window they were observing.
    """
    metrics = get_gate_metrics()
    before = len(metrics.snapshot())
    target = (gate or "").strip() or None
    metrics.reset(target)
    logger.info(
        "Gate metrics reset",
        gate=target or "*",
        gates_before=before,
    )
    return {
        "status": "ok",
        "reset_gate": target or "*",
        "gates_before_reset": before,
        "reset_at": utcnow().isoformat().replace("+00:00", "Z"),
    }


__all__ = ["router"]
