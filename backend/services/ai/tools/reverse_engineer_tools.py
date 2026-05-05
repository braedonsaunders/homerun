"""Agent tools for the strategy reverse-engineer workflow.

Lets external MCP clients trigger and observe the same end-to-end
pipeline the human operator drives in Strategy Research → Reverse
Engineer.  These are *outer* tools — they orchestrate jobs.  The agent
*inside* a reverse-engineer job has its own (richer) tool registry
(see services/strategy_reverse_engineer/tools.py).

Tool list:
  * ``strategy_reverse_engineer_start``   — enqueue a new job
  * ``strategy_reverse_engineer_status``  — read job + best result
  * ``strategy_reverse_engineer_iterations`` — list per-iteration audit
  * ``strategy_reverse_engineer_cancel``  — cancel an in-flight job
  * ``strategy_reverse_engineer_promote`` — promote winner to library
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from services.ai.agent import AgentTool

logger = logging.getLogger(__name__)


def build_tools() -> list[AgentTool]:
    return [
        AgentTool(
            name="strategy_reverse_engineer_start",
            description=(
                "Enqueue a new strategy reverse-engineer job for a wallet.  "
                "Returns the job_id immediately; the agent loop runs "
                "asynchronously on the discovery plane.  Use "
                "strategy_reverse_engineer_status to track progress and "
                "strategy_reverse_engineer_iterations to see what it tried."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "wallet_address": {"type": "string"},
                    "label": {"type": "string"},
                    "data_source_kind": {
                        "type": "string",
                        "enum": ["auto", "recording_session", "provider_dataset", "live"],
                        "default": "auto",
                    },
                    "recording_session_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "provider_dataset_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "llm_model": {"type": "string"},
                    "max_iterations": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                    },
                    "target_score": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "max_cost_usd": {"type": "number", "minimum": 0.0},
                    "max_wallet_trades": {
                        "type": "integer",
                        "minimum": 10,
                        "maximum": 10000,
                    },
                },
                "required": ["wallet_address"],
            },
            handler=_start,
            max_calls=5,
            category="strategies",
        ),
        AgentTool(
            name="strategy_reverse_engineer_status",
            description=(
                "Read the current state of a reverse-engineer job — status, "
                "progress, best score, best strategy code, total cost."
            ),
            parameters={
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
            handler=_status,
            max_calls=20,
            category="strategies",
        ),
        AgentTool(
            name="strategy_reverse_engineer_iterations",
            description=(
                "List every iteration of a reverse-engineer job with the "
                "candidate strategy class, score breakdown, and divergence "
                "summary.  Useful for understanding why the agent picked "
                "the strategy it did."
            ),
            parameters={
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
            handler=_iterations,
            max_calls=10,
            category="strategies",
        ),
        AgentTool(
            name="strategy_reverse_engineer_cancel",
            description="Cancel an in-flight reverse-engineer job.",
            parameters={
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
            handler=_cancel,
            max_calls=5,
            category="strategies",
        ),
        AgentTool(
            name="strategy_reverse_engineer_promote",
            description=(
                "Promote a finished reverse-engineer job's winning strategy "
                "into the strategy library as a new (disabled-by-default) "
                "Strategy row.  Operator must explicitly enable it before "
                "live trading."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "name": {"type": "string"},
                    "slug": {"type": "string"},
                    "description": {"type": "string"},
                    "enabled": {"type": "boolean", "default": False},
                },
                "required": ["job_id", "name", "slug"],
            },
            handler=_promote,
            max_calls=5,
            category="strategies",
        ),
    ]


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def _start(args: dict) -> dict:
    from services.strategy_reverse_engineer import service as re_service
    from utils.validation import validate_eth_address

    try:
        wallet = validate_eth_address(str(args.get("wallet_address") or ""))
    except ValueError as exc:
        return {"error": str(exc)}
    try:
        job = await re_service.enqueue_job(
            wallet_address=wallet,
            label=args.get("label"),
            data_source_kind=str(args.get("data_source_kind") or "auto"),
            recording_session_ids=args.get("recording_session_ids"),
            provider_dataset_ids=args.get("provider_dataset_ids"),
            llm_model=args.get("llm_model"),
            max_iterations=_int_or_none(args.get("max_iterations")),
            target_score=_float_or_none(args.get("target_score")),
            max_cost_usd=_float_or_none(args.get("max_cost_usd")),
            max_wallet_trades=_int_or_none(args.get("max_wallet_trades")),
        )
    except ValueError as exc:
        return {"error": str(exc)}
    return re_service.serialize_job(job)


async def _status(args: dict) -> dict:
    from services.strategy_reverse_engineer import service as re_service

    job_id = str(args.get("job_id") or "").strip()
    if not job_id:
        return {"error": "job_id is required"}
    row = await re_service.get_job(job_id)
    if row is None:
        return {"error": f"job '{job_id}' not found"}
    return re_service.serialize_job(row)


async def _iterations(args: dict) -> dict:
    from services.strategy_reverse_engineer import service as re_service

    job_id = str(args.get("job_id") or "").strip()
    if not job_id:
        return {"error": "job_id is required"}
    rows = await re_service.list_iterations(job_id)
    return {
        "job_id": job_id,
        "iterations": [re_service.serialize_iteration(r) for r in rows],
    }


async def _cancel(args: dict) -> dict:
    from services.strategy_reverse_engineer import service as re_service

    job_id = str(args.get("job_id") or "").strip()
    if not job_id:
        return {"error": "job_id is required"}
    ok = await re_service.cancel_job(job_id)
    return {"cancelled": bool(ok), "job_id": job_id}


async def _promote(args: dict) -> dict:
    from services.strategy_reverse_engineer import service as re_service

    job_id = str(args.get("job_id") or "").strip()
    name = str(args.get("name") or "").strip()
    slug = str(args.get("slug") or "").strip()
    if not (job_id and name and slug):
        return {"error": "job_id, name, slug are required"}
    try:
        return await re_service.promote_to_strategy_library(
            job_id,
            name=name,
            slug=slug,
            description=args.get("description"),
            enabled=bool(args.get("enabled", False)),
        )
    except ValueError as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
