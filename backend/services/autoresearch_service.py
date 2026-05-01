"""Autoresearch — Karpathy-inspired continuous optimization loop.

Supports two modes:
    - "params": Tune numeric parameters via walk-forward backtest
    - "code": Evolve strategy source code via strategy backtester

Reuses:
    - param_optimizer._replay_opportunities()      for param evaluation
    - backtest.unified_runner.run_unified_backtest() for code evaluation
    - strategy_versioning                           for code version management
    - strategy_tune_agent tools                     for context gathering
"""

from __future__ import annotations

import asyncio
import difflib
import json
import time
import uuid
from typing import Any, AsyncGenerator, Optional

from sqlalchemy import desc, select, update

from models.database import (
    AppSettings,
    AsyncSessionLocal,
    AutoresearchExperiment,
    AutoresearchIteration,
    CortexMemory,
    OpportunityHistory,
    Strategy,
    Trader,
)
from services.ai.agent import Agent, AgentEventType, AgentTool, run_agent_to_completion
from services.param_optimizer import (
    DEFAULT_PARAM_SPECS,
    BacktestResult,
    TradingParameters,
    _replay_opportunities,
    _split_walk_forward,
)
from services.strategy_tune_agent import (
    _apply_tuned_param_updates,
    _extract_json_payload,
    _summarize_trader_context,
    _tool_get_strategy_context,
    _tool_get_trader_context,
)
from utils.logger import get_logger
from utils.utcnow import utcnow

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Composite scoring (matches param_optimizer formula)
# ---------------------------------------------------------------------------


def _composite_score(result: BacktestResult) -> float:
    """Compute the composite optimization score from a BacktestResult."""
    return round(
        result.sharpe_ratio * 0.4
        + result.win_rate * 20.0 * 0.2
        + result.total_profit * 0.3
        - result.max_drawdown * 0.1,
        4,
    )


# ---------------------------------------------------------------------------
# Autoresearch settings helpers
# ---------------------------------------------------------------------------


async def load_autoresearch_settings() -> dict[str, Any]:
    """Load autoresearch_* columns from AppSettings."""
    async with AsyncSessionLocal() as session:
        row = (await session.execute(select(AppSettings).where(AppSettings.id == "default"))).scalar_one_or_none()
        if row is None:
            return _defaults()
        return {
            "model": getattr(row, "autoresearch_model", None),
            "max_iterations": getattr(row, "autoresearch_max_iterations", 50) or 50,
            "interval_seconds": getattr(row, "autoresearch_interval_seconds", 600) or 600,
            "temperature": getattr(row, "autoresearch_temperature", 0.2) or 0.2,
            "mandate": getattr(row, "autoresearch_mandate", None),
            "auto_apply": bool(getattr(row, "autoresearch_auto_apply", True)),
            "walk_forward_windows": getattr(row, "autoresearch_walk_forward_windows", 5) or 5,
            "train_ratio": getattr(row, "autoresearch_train_ratio", 0.7) or 0.7,
            "mode": getattr(row, "autoresearch_mode", "params") or "params",
        }


def _defaults() -> dict[str, Any]:
    return {
        "model": None,
        "max_iterations": 50,
        "interval_seconds": 600,
        "temperature": 0.2,
        "mandate": None,
        "auto_apply": True,
        "walk_forward_windows": 5,
        "train_ratio": 0.7,
        "mode": "params",
    }


async def save_autoresearch_settings(updates: dict[str, Any]) -> dict[str, Any]:
    """Persist autoresearch settings to AppSettings."""
    col_map = {
        "model": "autoresearch_model",
        "max_iterations": "autoresearch_max_iterations",
        "interval_seconds": "autoresearch_interval_seconds",
        "temperature": "autoresearch_temperature",
        "mandate": "autoresearch_mandate",
        "auto_apply": "autoresearch_auto_apply",
        "walk_forward_windows": "autoresearch_walk_forward_windows",
        "train_ratio": "autoresearch_train_ratio",
        "mode": "autoresearch_mode",
    }
    async with AsyncSessionLocal() as session:
        row = (await session.execute(select(AppSettings).where(AppSettings.id == "default"))).scalar_one_or_none()
        if row is None:
            return _defaults()
        for key, value in updates.items():
            col = col_map.get(key)
            if col and value is not None:
                setattr(row, col, value)
        await session.commit()
    return await load_autoresearch_settings()


async def _run_agent_to_completion_streaming(
    *,
    system_prompt: str,
    query: str,
    tools: list[AgentTool],
    model: str | None,
    max_iterations: int,
    session_type: str,
    iteration_num: int,
) -> AsyncGenerator[dict[str, Any], None]:
    """Run an agent and forward progress events before yielding the final result."""
    agent = Agent(
        system_prompt=system_prompt,
        tools=tools,
        model=model,
        max_iterations=max_iterations,
        session_type=session_type,
    )

    result: dict[str, Any] | None = None
    async for agent_event in agent.run(query):
        if agent_event.type == AgentEventType.THINKING:
            content = str(agent_event.data.get("content") or "").strip()
            if content:
                yield {
                    "event": "agent_progress",
                    "data": {
                        "iteration": iteration_num,
                        "phase": "thinking",
                        "message": content[:1000],
                    },
                }
        elif agent_event.type == AgentEventType.TOOL_START:
            tool_name = str(agent_event.data.get("tool") or "tool")
            yield {
                "event": "agent_progress",
                "data": {
                    "iteration": iteration_num,
                    "phase": "tool_start",
                    "tool": tool_name,
                    "message": f"Using {tool_name}",
                },
            }
        elif agent_event.type == AgentEventType.TOOL_END:
            tool_name = str(agent_event.data.get("tool") or "tool")
            yield {
                "event": "agent_progress",
                "data": {
                    "iteration": iteration_num,
                    "phase": "tool_end",
                    "tool": tool_name,
                    "message": f"Finished {tool_name}",
                },
            }
        elif agent_event.type == AgentEventType.TOOL_ERROR:
            tool_name = str(agent_event.data.get("tool") or "tool")
            error = str(agent_event.data.get("error") or "tool error")
            yield {
                "event": "agent_progress",
                "data": {
                    "iteration": iteration_num,
                    "phase": "tool_error",
                    "tool": tool_name,
                    "message": f"{tool_name}: {error}"[:1000],
                },
            }
        elif agent_event.type == AgentEventType.DONE:
            result = agent_event.data
        elif agent_event.type == AgentEventType.ERROR:
            raise RuntimeError(str(agent_event.data.get("error") or "Agent failed"))

    if result is None:
        raise RuntimeError("Agent completed without producing a result")

    yield {"event": "agent_done", "data": result}


# ---------------------------------------------------------------------------
# Opportunity history loader (mirrors param_optimizer)
# ---------------------------------------------------------------------------


async def _load_opportunity_history() -> list[dict]:
    """Load all resolved opportunity history records sorted by detected_at."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(OpportunityHistory).order_by(OpportunityHistory.detected_at.asc())
        )
        rows = result.scalars().all()
        return [
            {
                "id": r.id,
                "strategy_type": r.strategy_type,
                "event_id": r.event_id,
                "title": r.title,
                "total_cost": r.total_cost,
                "expected_roi": r.expected_roi,
                "risk_score": r.risk_score,
                "positions_data": r.positions_data,
                "detected_at": r.detected_at,
                "expired_at": r.expired_at,
                "resolution_date": r.resolution_date,
                "was_profitable": r.was_profitable,
                "actual_roi": r.actual_roi,
            }
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Evaluate a param set using walk-forward
# ---------------------------------------------------------------------------


def _evaluate_params_sync(
    opportunities: list[dict],
    params: TradingParameters,
    walk_forward: bool = True,
    n_windows: int = 5,
    train_ratio: float = 0.7,
) -> tuple[float, dict]:
    """Run backtest with optional walk-forward. Returns (score, result_dict)."""
    if walk_forward and len(opportunities) >= 10:
        folds = _split_walk_forward(opportunities, n_windows=n_windows, train_ratio=train_ratio)
        if folds:
            test_results: list[BacktestResult] = []
            for _train, test in folds:
                r = _replay_opportunities(test, params)
                test_results.append(r)
            # Average test results across folds
            avg = BacktestResult(
                total_profit=round(sum(r.total_profit for r in test_results) / len(test_results), 4),
                total_cost=round(sum(r.total_cost for r in test_results) / len(test_results), 4),
                num_trades=sum(r.num_trades for r in test_results) // len(test_results),
                num_wins=sum(r.num_wins for r in test_results) // len(test_results),
                num_losses=sum(r.num_losses for r in test_results) // len(test_results),
                win_rate=round(sum(r.win_rate for r in test_results) / len(test_results), 4),
                max_drawdown=round(max(r.max_drawdown for r in test_results), 4),
                sharpe_ratio=round(sum(r.sharpe_ratio for r in test_results) / len(test_results), 4),
                avg_roi=round(sum(r.avg_roi for r in test_results) / len(test_results), 4),
                profit_factor=round(sum(r.profit_factor for r in test_results) / len(test_results), 4),
            )
            return _composite_score(avg), avg.to_dict()

    # Fallback: single backtest over all data
    result = _replay_opportunities(opportunities, params)
    return _composite_score(result), result.to_dict()


# ---------------------------------------------------------------------------
# Agent tools for the autoresearch loop
# ---------------------------------------------------------------------------


def _build_autoresearch_tools(trader_id: str, experiment_id: str) -> list[AgentTool]:
    """Build the tool set for the autoresearch proposal agent."""

    async def _get_iteration_history(args: dict) -> dict:
        limit = max(1, min(int(args.get("limit") or 15), 50))
        async with AsyncSessionLocal() as session:
            rows = (
                await session.execute(
                    select(AutoresearchIteration)
                    .where(AutoresearchIteration.experiment_id == experiment_id)
                    .order_by(desc(AutoresearchIteration.iteration_number))
                    .limit(limit)
                )
            ).scalars().all()
        return {
            "iterations": [
                {
                    "iteration": r.iteration_number,
                    "score": r.new_score,
                    "delta": r.score_delta,
                    "decision": r.decision,
                    "reasoning": (r.reasoning or "")[:300],
                    "changed_params": r.changed_params_json,
                }
                for r in rows
            ]
        }

    async def _get_cortex_memories(args: dict) -> dict:
        query = str(args.get("query") or "strategy optimization").strip()
        limit = max(1, min(int(args.get("limit") or 10), 20))
        async with AsyncSessionLocal() as session:
            stmt = (
                select(CortexMemory)
                .where(CortexMemory.expired == False)  # noqa: E712
                .order_by(CortexMemory.importance.desc())
                .limit(limit)
            )
            rows = (await session.execute(stmt)).scalars().all()
        # Simple keyword filtering
        query_lower = query.lower()
        filtered = [
            {"id": r.id, "category": r.category, "content": r.content, "importance": r.importance}
            for r in rows
            if query_lower in (r.content or "").lower() or not query
        ]
        return {"memories": filtered[:limit]}

    async def _get_param_specs(_args: dict) -> dict:
        return {
            "param_specs": [
                {
                    "name": s.name,
                    "current_value": s.current_value,
                    "min_bound": s.min_bound,
                    "max_bound": s.max_bound,
                    "step": s.step,
                    "category": s.category,
                    "description": s.description,
                }
                for s in DEFAULT_PARAM_SPECS
            ]
        }

    return [
        AgentTool(
            name="get_trader_context",
            description="Fetch trader runtime context, recent decisions, orders, and signals.",
            parameters={
                "type": "object",
                "properties": {
                    "trader_id": {"type": "string"},
                    "decision_limit": {"type": "integer"},
                    "order_limit": {"type": "integer"},
                    "signal_limit": {"type": "integer"},
                },
                "required": ["trader_id"],
            },
            handler=_tool_get_trader_context,
            max_calls=3,
            category="autoresearch",
        ),
        AgentTool(
            name="get_strategy_context",
            description="Fetch strategy source/config and version history.",
            parameters={
                "type": "object",
                "properties": {
                    "strategy_key": {"type": "string"},
                    "version": {"type": ["string", "integer", "null"]},
                    "include_source": {"type": "boolean"},
                    "limit": {"type": "integer"},
                },
                "required": ["strategy_key"],
            },
            handler=_tool_get_strategy_context,
            max_calls=4,
            category="autoresearch",
        ),
        AgentTool(
            name="get_iteration_history",
            description="Get recent iteration results from this autoresearch experiment — scores, decisions, reasoning, and param changes.",
            parameters={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max iterations to return (default 15, max 50)."},
                },
                "required": [],
            },
            handler=_get_iteration_history,
            max_calls=3,
            category="autoresearch",
        ),
        AgentTool(
            name="get_cortex_memories",
            description="Search Cortex fleet commander memories for strategy lessons, rules, and observations.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keyword query to filter memories."},
                    "limit": {"type": "integer", "description": "Max results (default 10)."},
                },
                "required": [],
            },
            handler=_get_cortex_memories,
            max_calls=3,
            category="autoresearch",
        ),
        AgentTool(
            name="get_param_specs",
            description="Get all tunable parameter specifications with names, bounds, step sizes, categories, and descriptions.",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=_get_param_specs,
            max_calls=2,
            category="autoresearch",
        ),
    ]


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------


def _build_system_prompt(
    trader_summary: dict,
    current_params: dict,
    baseline_score: float,
    iteration_num: int,
    mandate: str | None,
) -> str:
    """Build the system prompt for the autoresearch proposal agent."""

    param_summary = json.dumps(current_params, indent=2, default=str)
    trader_json = json.dumps(trader_summary, indent=2, default=str)

    mandate_section = ""
    if mandate:
        mandate_section = f"""
## Custom Constraints
{mandate}
"""

    return f"""\
You are the Homerun Autoresearch agent — a continuous parameter optimization loop
for prediction market trading bots. Your job is to propose parameter changes that
improve risk-adjusted returns as measured by the composite score.

## Current State
- **Iteration**: {iteration_num}
- **Baseline composite score**: {baseline_score}
- **Trader summary**: {trader_json}
- **Current parameters**: {param_summary}

## Composite Score Formula
score = sharpe_ratio * 0.4 + win_rate * 20.0 * 0.2 + total_profit * 0.3 - max_drawdown * 0.1

Higher is better. Your goal is to maximise this score via parameter changes.

## Instructions
1. Use tools to gather context: trader performance, strategy details, past iterations, Cortex memories, param specs.
2. Analyze what's working and what isn't — look at win rate, drawdown, profit trends.
3. Propose a SMALL, focused set of parameter changes (1-4 params per iteration).
4. Explain your reasoning: what you expect to improve and why.
5. Stay within param bounds (use get_param_specs to verify).
6. Be conservative — small incremental improvements are better than wild swings.
7. Learn from past iterations — if tightening a param didn't help, try something else.
{mandate_section}
## Output Format
Return STRICT JSON (no markdown fences, no extra text) with these keys:
{{
  "proposed_changes": {{"param_name": new_value, ...}},
  "reasoning": "Why these changes should improve the score...",
  "confidence": 0.0-1.0
}}

If you believe no changes will improve the score, return:
{{
  "proposed_changes": {{}},
  "reasoning": "Explanation of why current params are optimal...",
  "confidence": 0.0
}}
"""


# ---------------------------------------------------------------------------
# AutoresearchService singleton
# ---------------------------------------------------------------------------


class AutoresearchService:
    """Manages autoresearch experiment lifecycle and loop execution."""

    _instance: Optional["AutoresearchService"] = None
    _active_experiments: dict[str, str]  # trader_id -> experiment_id

    def __new__(cls) -> "AutoresearchService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._active_experiments = {}
        self._stop_flags: dict[str, asyncio.Event] = {}  # trader_id -> stop event
        logger.info("AutoresearchService initialised")

    # ------------------------------------------------------------------
    # Status & history
    # ------------------------------------------------------------------

    async def get_experiment_status(self, trader_id: str) -> dict[str, Any]:
        """Get the latest experiment status for a trader."""
        async with AsyncSessionLocal() as session:
            row = (
                await session.execute(
                    select(AutoresearchExperiment)
                    .where(AutoresearchExperiment.trader_id == trader_id)
                    .order_by(desc(AutoresearchExperiment.created_at))
                    .limit(1)
                )
            ).scalar_one_or_none()

        if row is None:
            return {
                "experiment_id": None,
                "trader_id": trader_id,
                "status": "idle",
                "mode": "params",
                "iteration_count": 0,
                "best_score": 0.0,
                "baseline_score": 0.0,
                "kept_count": 0,
                "reverted_count": 0,
                "started_at": None,
                "name": None,
            }

        return {
            "experiment_id": row.id,
            "trader_id": row.trader_id,
            "name": row.name,
            "status": "running" if trader_id in self._active_experiments else row.status,
            "mode": getattr(row, "mode", "params") or "params",
            "strategy_id": getattr(row, "strategy_id", None),
            "best_version": getattr(row, "best_version", None),
            "iteration_count": row.iteration_count,
            "best_score": row.best_score,
            "baseline_score": row.baseline_score,
            "best_params": row.best_params_json,
            "kept_count": row.kept_count,
            "reverted_count": row.reverted_count,
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "finished_at": row.finished_at.isoformat() if row.finished_at else None,
            "settings": row.settings_json,
        }

    async def get_experiment_history(
        self,
        trader_id: str,
        experiment_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return iteration log rows for an experiment."""
        async with AsyncSessionLocal() as session:
            if experiment_id:
                exp_id = experiment_id
            else:
                exp_row = (
                    await session.execute(
                        select(AutoresearchExperiment)
                        .where(AutoresearchExperiment.trader_id == trader_id)
                        .order_by(desc(AutoresearchExperiment.created_at))
                        .limit(1)
                    )
                ).scalar_one_or_none()
                if exp_row is None:
                    return []
                exp_id = exp_row.id

            rows = (
                await session.execute(
                    select(AutoresearchIteration)
                    .where(AutoresearchIteration.experiment_id == exp_id)
                    .order_by(desc(AutoresearchIteration.iteration_number))
                    .limit(limit)
                )
            ).scalars().all()

        return [
            {
                "id": r.id,
                "iteration_number": r.iteration_number,
                "baseline_score": r.baseline_score,
                "new_score": r.new_score,
                "score_delta": r.score_delta,
                "decision": r.decision,
                "reasoning": r.reasoning,
                "changed_params": r.changed_params_json,
                "backtest_result": r.backtest_result_json,
                "source_diff": r.source_diff,
                "validation_result": r.validation_result_json,
                "duration_seconds": r.duration_seconds,
                "tokens_used": r.tokens_used,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]

    async def list_experiments(self, trader_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """List all experiments for a trader."""
        async with AsyncSessionLocal() as session:
            rows = (
                await session.execute(
                    select(AutoresearchExperiment)
                    .where(AutoresearchExperiment.trader_id == trader_id)
                    .order_by(desc(AutoresearchExperiment.created_at))
                    .limit(limit)
                )
            ).scalars().all()

        return [
            {
                "experiment_id": r.id,
                "name": r.name,
                "status": r.status,
                "iteration_count": r.iteration_count,
                "best_score": r.best_score,
                "baseline_score": r.baseline_score,
                "kept_count": r.kept_count,
                "reverted_count": r.reverted_count,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "finished_at": r.finished_at.isoformat() if r.finished_at else None,
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Stop experiment
    # ------------------------------------------------------------------

    async def stop_experiment(self, trader_id: str) -> dict[str, Any]:
        """Signal a running experiment to stop."""
        return await self._stop_session(trader_id)

    async def _stop_session(self, session_key: str) -> dict[str, Any]:
        """Stop the experiment registered under ``session_key``.

        ``session_key`` is either a trader id (legacy bot-scoped path)
        or ``strategy:<id>`` (new strategy-scoped path). Both forms
        resolve to the same active-experiment + stop-flag dictionaries.
        """
        stop_event = self._stop_flags.get(session_key)
        if stop_event:
            stop_event.set()

        exp_id = self._active_experiments.pop(session_key, None)
        if exp_id:
            async with AsyncSessionLocal() as session:
                await session.execute(
                    update(AutoresearchExperiment)
                    .where(AutoresearchExperiment.id == exp_id)
                    .values(status="paused", finished_at=utcnow(), updated_at=utcnow())
                )
                await session.commit()

        self._stop_flags.pop(session_key, None)
        return {"stopped": True, "experiment_id": exp_id}

    # ── Strategy-scoped helpers ────────────────────────────────────────
    # These mirror the trader-scoped status/history/list/stop methods
    # but key off ``strategy_id`` instead. Code experiments live here
    # because they evolve a strategy's source against the backtest data
    # plane and don't need a bot context at all.

    async def get_strategy_experiment_status(self, strategy_id: str) -> dict[str, Any]:
        """Get the latest experiment status for a strategy (code-only)."""
        session_key = f"strategy:{strategy_id}"
        async with AsyncSessionLocal() as session:
            row = (
                await session.execute(
                    select(AutoresearchExperiment)
                    .where(
                        AutoresearchExperiment.strategy_id == strategy_id,
                        AutoresearchExperiment.mode == "code",
                    )
                    .order_by(desc(AutoresearchExperiment.created_at))
                    .limit(1)
                )
            ).scalar_one_or_none()

        if row is None:
            return {
                "experiment_id": None,
                "strategy_id": strategy_id,
                "status": "idle",
                "mode": "code",
                "iteration_count": 0,
                "best_score": 0.0,
                "baseline_score": 0.0,
                "kept_count": 0,
                "reverted_count": 0,
                "started_at": None,
                "name": None,
            }

        return {
            "experiment_id": row.id,
            "strategy_id": row.strategy_id,
            "trader_id": row.trader_id,
            "name": row.name,
            "status": "running" if session_key in self._active_experiments else row.status,
            "mode": getattr(row, "mode", "code") or "code",
            "best_version": getattr(row, "best_version", None),
            "iteration_count": row.iteration_count,
            "best_score": row.best_score,
            "baseline_score": row.baseline_score,
            "kept_count": row.kept_count,
            "reverted_count": row.reverted_count,
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "finished_at": row.finished_at.isoformat() if row.finished_at else None,
            "settings": row.settings_json,
        }

    async def get_strategy_experiment_history(
        self,
        strategy_id: str,
        experiment_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Iteration log rows for a strategy's code experiment."""
        async with AsyncSessionLocal() as session:
            if experiment_id:
                exp_id = experiment_id
            else:
                exp_row = (
                    await session.execute(
                        select(AutoresearchExperiment)
                        .where(
                            AutoresearchExperiment.strategy_id == strategy_id,
                            AutoresearchExperiment.mode == "code",
                        )
                        .order_by(desc(AutoresearchExperiment.created_at))
                        .limit(1)
                    )
                ).scalar_one_or_none()
                if exp_row is None:
                    return []
                exp_id = exp_row.id

            rows = (
                await session.execute(
                    select(AutoresearchIteration)
                    .where(AutoresearchIteration.experiment_id == exp_id)
                    .order_by(desc(AutoresearchIteration.iteration_number))
                    .limit(limit)
                )
            ).scalars().all()

        return [
            {
                "iteration_number": r.iteration_number,
                "decision": r.decision,
                "new_score": r.new_score,
                "score_delta": r.score_delta,
                "duration_seconds": r.duration_seconds,
                "reasoning": r.reasoning,
                "changed_params": r.changed_params_json,
                "validation_passed": r.validation_passed,
                "validation_result": r.validation_result_json,
                "backtest_result": r.backtest_result_json,
                "source_diff": r.source_diff,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]

    async def stop_strategy_experiment(self, strategy_id: str) -> dict[str, Any]:
        """Stop the strategy-scoped code experiment, if any."""
        return await self._stop_session(f"strategy:{strategy_id}")

    # ------------------------------------------------------------------
    # Core autoresearch loop (async generator for SSE streaming)
    # ------------------------------------------------------------------

    async def run_experiment_stream(
        self,
        trader_id: str,
        settings_override: dict[str, Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Run the autoresearch loop, yielding progress events for SSE.

        Each yield is a dict with 'event' and 'data' keys.
        """
        # Prevent duplicate experiments for the same trader
        if trader_id in self._active_experiments:
            yield {"event": "error", "data": {"error": "An experiment is already running for this trader"}}
            return

        # Load settings
        settings = await load_autoresearch_settings()
        if settings_override:
            for k, v in settings_override.items():
                if v is not None and k in settings:
                    settings[k] = v

        mode = settings.get("mode", "params")

        # Branch to code evolution if mode is "code"
        if mode == "code":
            strategy_id = (settings_override or {}).get("strategy_id")
            if not strategy_id:
                yield {"event": "error", "data": {"error": "strategy_id is required for code evolution mode"}}
                return
            async for event in self.run_code_evolution_stream(trader_id, strategy_id, settings_override):
                yield event
            return

        max_iterations = int(settings.get("max_iterations", 50))
        walk_forward = True
        n_windows = int(settings.get("walk_forward_windows", 5))
        train_ratio = float(settings.get("train_ratio", 0.7))
        auto_apply = bool(settings.get("auto_apply", True))
        model = settings.get("model")
        mandate = settings.get("mandate")

        # Validate trader exists
        async with AsyncSessionLocal() as session:
            trader_row = await session.get(Trader, trader_id)
            if trader_row is None:
                yield {"event": "error", "data": {"error": f"Trader '{trader_id}' not found"}}
                return
            trader_name = str(trader_row.name or trader_id[:8])

        # Load opportunity history
        opportunities = await _load_opportunity_history()
        if not opportunities:
            yield {"event": "error", "data": {"error": "No opportunity history data available for backtesting"}}
            return

        # Create experiment record
        experiment_id = str(uuid.uuid4())
        now = utcnow()
        name = f"auto-{now.strftime('%Y%m%d-%H%M')}"

        # Compute baseline score with current params
        current_params = TradingParameters()
        baseline_score, baseline_result = _evaluate_params_sync(
            opportunities, current_params, walk_forward, n_windows, train_ratio
        )

        async with AsyncSessionLocal() as session:
            session.add(AutoresearchExperiment(
                id=experiment_id,
                trader_id=trader_id,
                name=name,
                status="running",
                baseline_score=baseline_score,
                best_score=baseline_score,
                best_params_json=current_params.to_dict(),
                iteration_count=0,
                kept_count=0,
                reverted_count=0,
                settings_json=settings,
                started_at=now,
                created_at=now,
                updated_at=now,
            ))
            await session.commit()

        self._active_experiments[trader_id] = experiment_id
        stop_event = asyncio.Event()
        self._stop_flags[trader_id] = stop_event

        yield {
            "event": "experiment_start",
            "data": {
                "experiment_id": experiment_id,
                "name": name,
                "baseline_score": baseline_score,
                "baseline_result": baseline_result,
                "max_iterations": max_iterations,
            },
        }

        best_score = baseline_score
        best_params = current_params.to_dict()
        current_baseline = baseline_score
        kept_count = 0
        reverted_count = 0

        try:
            for iteration_num in range(1, max_iterations + 1):
                if stop_event.is_set():
                    break

                iter_start = time.monotonic()

                yield {
                    "event": "iteration_start",
                    "data": {"iteration": iteration_num, "total": max_iterations, "baseline_score": current_baseline},
                }

                # 1. Get trader context for the LLM
                try:
                    trader_context = await _tool_get_trader_context({"trader_id": trader_id})
                    trader_summary = _summarize_trader_context(trader_context)
                except Exception as exc:
                    trader_summary = {"error": str(exc)}

                # 2. Build system prompt and run LLM agent
                system_prompt = _build_system_prompt(
                    trader_summary=trader_summary,
                    current_params=best_params,
                    baseline_score=current_baseline,
                    iteration_num=iteration_num,
                    mandate=mandate,
                )

                tools = _build_autoresearch_tools(trader_id, experiment_id)

                proposed_changes = {}
                reasoning = ""
                confidence = 0.0
                tokens_used = 0

                try:
                    result = await run_agent_to_completion(
                        system_prompt=system_prompt,
                        query=f"Propose parameter changes for iteration {iteration_num} of the autoresearch loop for trader '{trader_name}'. Current baseline score: {current_baseline}. Analyze context and propose improvements.",
                        tools=tools,
                        model=model,
                        max_iterations=8,
                        session_type="autoresearch",
                    )

                    answer = result.get("result", {}).get("answer", "")
                    tokens_used = result.get("result", {}).get("tokens_used", 0) or 0

                    parsed = _extract_json_payload(answer)
                    if parsed:
                        proposed_changes = dict(parsed.get("proposed_changes") or {})
                        reasoning = str(parsed.get("reasoning") or "")
                        confidence = float(parsed.get("confidence") or 0.0)
                    else:
                        reasoning = f"Could not parse JSON from LLM response: {answer[:200]}"

                except Exception as exc:
                    reasoning = f"LLM agent error: {exc}"
                    logger.warning("Autoresearch iteration %d LLM error: %s", iteration_num, exc)

                yield {
                    "event": "proposal",
                    "data": {
                        "iteration": iteration_num,
                        "proposed_changes": proposed_changes,
                        "reasoning": reasoning[:500],
                        "confidence": confidence,
                    },
                }

                # 3. Evaluate proposed changes
                decision = "reverted"
                new_score = current_baseline
                score_delta = 0.0
                backtest_result_dict: dict = {}

                if proposed_changes:
                    # Merge proposed changes into current best params
                    test_params_dict = dict(best_params)
                    changed_params = {}
                    for k, v in proposed_changes.items():
                        if k in test_params_dict:
                            old_val = test_params_dict[k]
                            try:
                                new_val = type(old_val)(v) if old_val is not None else float(v)
                            except (ValueError, TypeError):
                                continue
                            # Validate bounds
                            spec = next((s for s in DEFAULT_PARAM_SPECS if s.name == k), None)
                            if spec:
                                new_val = max(spec.min_bound, min(spec.max_bound, new_val))
                            if new_val != old_val:
                                test_params_dict[k] = new_val
                                changed_params[k] = new_val

                    if changed_params:
                        test_params = TradingParameters.from_dict(test_params_dict)
                        new_score, backtest_result_dict = _evaluate_params_sync(
                            opportunities, test_params, walk_forward, n_windows, train_ratio
                        )
                        score_delta = round(new_score - current_baseline, 4)

                        if new_score > current_baseline:
                            decision = "kept"
                            best_params = test_params_dict
                            current_baseline = new_score
                            if new_score > best_score:
                                best_score = new_score
                            kept_count += 1

                            # Auto-apply to trader if enabled
                            if auto_apply:
                                try:
                                    # Build updates in the format _apply_tuned_param_updates expects
                                    source_configs = trader_context.get("trader", {}).get("source_configs", [])
                                    if source_configs:
                                        updates = [
                                            {
                                                "source_key": cfg.get("source_key", ""),
                                                "strategy_key": cfg.get("strategy_key"),
                                                "params_patch": changed_params,
                                                "reason": f"autoresearch iteration {iteration_num}",
                                            }
                                            for cfg in source_configs
                                        ]
                                        await _apply_tuned_param_updates(trader_id=trader_id, updates=updates)
                                except Exception as exc:
                                    logger.warning("Failed to auto-apply params: %s", exc)
                        else:
                            reverted_count += 1
                    else:
                        reasoning = "No valid param changes proposed (all out of bounds or unchanged)"
                        reverted_count += 1
                else:
                    reasoning = reasoning or "Agent proposed no changes"
                    reverted_count += 1

                duration = round(time.monotonic() - iter_start, 2)

                # 4. Persist iteration
                async with AsyncSessionLocal() as session:
                    session.add(AutoresearchIteration(
                        id=str(uuid.uuid4()),
                        experiment_id=experiment_id,
                        iteration_number=iteration_num,
                        proposed_params_json=proposed_changes,
                        baseline_score=round(current_baseline - score_delta if decision == "kept" else current_baseline, 4),
                        new_score=round(new_score, 4),
                        score_delta=score_delta,
                        decision=decision,
                        reasoning=reasoning[:2000],
                        backtest_result_json=backtest_result_dict,
                        changed_params_json=proposed_changes if decision == "kept" else None,
                        duration_seconds=duration,
                        tokens_used=tokens_used,
                        created_at=utcnow(),
                    ))
                    # Update experiment
                    await session.execute(
                        update(AutoresearchExperiment)
                        .where(AutoresearchExperiment.id == experiment_id)
                        .values(
                            iteration_count=iteration_num,
                            kept_count=kept_count,
                            reverted_count=reverted_count,
                            best_score=best_score,
                            best_params_json=best_params,
                            updated_at=utcnow(),
                        )
                    )
                    await session.commit()

                yield {
                    "event": "decision",
                    "data": {
                        "iteration": iteration_num,
                        "decision": decision,
                        "new_score": round(new_score, 4),
                        "baseline_score": round(current_baseline - score_delta if decision == "kept" else current_baseline, 4),
                        "score_delta": score_delta,
                        "best_score": round(best_score, 4),
                        "changed_params": proposed_changes if decision == "kept" else None,
                        "reasoning": reasoning[:300],
                        "duration_seconds": duration,
                        "kept_count": kept_count,
                        "reverted_count": reverted_count,
                    },
                }

                # Small delay between iterations to avoid overwhelming LLM API
                await asyncio.sleep(1)

        except Exception as exc:
            logger.exception("Autoresearch loop error: %s", exc)
            yield {"event": "error", "data": {"error": str(exc)}}
        finally:
            # Mark experiment as completed
            final_status = "paused" if stop_event.is_set() else "completed"
            async with AsyncSessionLocal() as session:
                await session.execute(
                    update(AutoresearchExperiment)
                    .where(AutoresearchExperiment.id == experiment_id)
                    .values(
                        status=final_status,
                        finished_at=utcnow(),
                        updated_at=utcnow(),
                    )
                )
                await session.commit()

            self._active_experiments.pop(trader_id, None)
            self._stop_flags.pop(trader_id, None)

        yield {
            "event": "done",
            "data": {
                "experiment_id": experiment_id,
                "total_iterations": kept_count + reverted_count,
                "best_score": round(best_score, 4),
                "baseline_score": round(baseline_score, 4),
                "improvement": round(best_score - baseline_score, 4),
                "kept_count": kept_count,
                "reverted_count": reverted_count,
                "best_params": best_params,
            },
        }

    # ------------------------------------------------------------------
    # Fleet-wide status (for Cortex integration)
    # ------------------------------------------------------------------

    async def get_fleet_autoresearch_status(self, trader_id: str | None = None) -> list[dict]:
        """Get active/recent autoresearch experiments across the fleet."""
        async with AsyncSessionLocal() as session:
            stmt = (
                select(AutoresearchExperiment)
                .order_by(desc(AutoresearchExperiment.updated_at))
                .limit(20)
            )
            if trader_id:
                stmt = stmt.where(AutoresearchExperiment.trader_id == trader_id)

            rows = (await session.execute(stmt)).scalars().all()

        results = []
        for r in rows:
            # Get last 3 iterations
            async with AsyncSessionLocal() as session:
                iters = (
                    await session.execute(
                        select(AutoresearchIteration)
                        .where(AutoresearchIteration.experiment_id == r.id)
                        .order_by(desc(AutoresearchIteration.iteration_number))
                        .limit(3)
                    )
                ).scalars().all()

            results.append({
                "experiment_id": r.id,
                "trader_id": r.trader_id,
                "name": r.name,
                "status": "running" if r.trader_id in self._active_experiments else r.status,
                "mode": getattr(r, "mode", "params") or "params",
                "strategy_id": getattr(r, "strategy_id", None),
                "best_version": getattr(r, "best_version", None),
                "iteration_count": r.iteration_count,
                "best_score": r.best_score,
                "baseline_score": r.baseline_score,
                "kept_count": r.kept_count,
                "reverted_count": r.reverted_count,
                "recent_iterations": [
                    {
                        "iteration": i.iteration_number,
                        "score": i.new_score,
                        "delta": i.score_delta,
                        "decision": i.decision,
                    }
                    for i in iters
                ],
            })

        return results


    # ------------------------------------------------------------------
    # Code evolution loop
    # ------------------------------------------------------------------

    async def run_code_evolution_stream(
        self,
        trader_id: str | None,
        strategy_id: str,
        settings_override: dict[str, Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Run strategy code evolution loop, yielding SSE events.

        ``trader_id`` is optional. When omitted, the experiment is
        strategy-scoped — it operates purely on the backtest data plane
        and the active-experiments lock is keyed by ``strategy:<id>``
        instead of the trader id. Trader-scoped runs preserve legacy
        behavior so the existing per-bot Tune flows are unaffected.
        """
        from services.backtest.unified_runner import run_unified_backtest
        from services.strategy_loader import validate_strategy_source, ALLOWED_IMPORT_PREFIXES
        from services.strategy_versioning import create_strategy_version_snapshot
        from services.strategy_runtime import bump_strategy_runtime_revisions

        # Session key — either the trader id (legacy) or a strategy
        # pseudo-key (new path). Used for the active-experiments and
        # stop-flag dictionaries.
        session_key = trader_id or f"strategy:{strategy_id}"

        if session_key in self._active_experiments:
            scope = "trader" if trader_id else "strategy"
            yield {"event": "error", "data": {"error": f"An experiment is already running for this {scope}"}}
            return

        settings = await load_autoresearch_settings()
        if settings_override:
            for k, v in settings_override.items():
                if v is not None and k in settings:
                    settings[k] = v

        max_iterations = int(settings.get("max_iterations", 50))
        auto_apply = bool(settings.get("auto_apply", True))
        model = settings.get("model")
        mandate = settings.get("mandate")

        # Load strategy
        async with AsyncSessionLocal() as session:
            strategy_row = await session.get(Strategy, strategy_id)
            if strategy_row is None:
                yield {"event": "error", "data": {"error": f"Strategy '{strategy_id}' not found"}}
                return
            strategy_slug = str(strategy_row.slug or "")
            strategy_source = str(strategy_row.source_code or "")
            strategy_class_name = str(strategy_row.class_name or "")
            strategy_config = dict(strategy_row.config or {})
            strategy_source_key = str(strategy_row.source_key or "scanner")
            baseline_version_num = int(strategy_row.version or 1)

        if not strategy_source.strip():
            yield {"event": "error", "data": {"error": "Strategy has no source code"}}
            return

        # Validate trader if provided (legacy bot-scoped path).
        # Strategy-scoped runs skip this entirely.
        if trader_id is not None:
            async with AsyncSessionLocal() as session:
                trader_row = await session.get(Trader, trader_id)
                if trader_row is None:
                    yield {"event": "error", "data": {"error": f"Trader '{trader_id}' not found"}}
                    return

        # Baseline backtest — UNIFIED runner (Cox-aware fill simulation,
        # ensemble bands, deflated Sharpe, regime decomposition).  This
        # is the same pipeline BacktestStudio shows, so the score the
        # LLM optimizes against is the score a human would see.
        try:
            baseline_result = await run_unified_backtest(
                source_code=strategy_source,
                slug=strategy_slug,
                config=strategy_config,
            )
            baseline_score = _code_evolution_score(baseline_result)
        except Exception as exc:
            yield {"event": "error", "data": {"error": f"Baseline backtest failed: {exc}"}}
            return

        # Create experiment
        experiment_id = str(uuid.uuid4())
        now = utcnow()
        name = f"code-{strategy_slug}-{now.strftime('%Y%m%d-%H%M')}"

        async with AsyncSessionLocal() as session:
            session.add(AutoresearchExperiment(
                id=experiment_id,
                trader_id=trader_id,
                name=name,
                status="running",
                mode="code",
                strategy_id=strategy_id,
                baseline_score=baseline_score,
                best_score=baseline_score,
                best_source_code=strategy_source,
                best_version=baseline_version_num,
                iteration_count=0,
                settings_json=settings,
                started_at=now,
                created_at=now,
                updated_at=now,
            ))
            await session.commit()

        self._active_experiments[session_key] = experiment_id
        stop_event = asyncio.Event()
        self._stop_flags[session_key] = stop_event

        yield {
            "event": "experiment_start",
            "data": {
                "experiment_id": experiment_id,
                "name": name,
                "mode": "code",
                "strategy_slug": strategy_slug,
                "baseline_score": baseline_score,
                "baseline_version": baseline_version_num,
                "max_iterations": max_iterations,
                "scope": "trader" if trader_id else "strategy",
            },
        }

        best_score = baseline_score
        best_source = strategy_source
        best_version_num = baseline_version_num
        current_source = strategy_source
        kept_count = 0
        reverted_count = 0
        last_backtest_summary: dict | None = None

        try:
            for iteration_num in range(1, max_iterations + 1):
                if stop_event.is_set():
                    break

                iter_start = time.monotonic()

                yield {
                    "event": "iteration_start",
                    "data": {"iteration": iteration_num, "total": max_iterations, "baseline_score": best_score},
                }

                # Build context. Strategy-scoped runs have no trader, so
                # we skip the per-bot context fetch.
                try:
                    if trader_id:
                        trader_context = await _tool_get_trader_context({"trader_id": trader_id})
                        trader_summary = _summarize_trader_context(trader_context)
                    else:
                        trader_summary = {}
                except Exception:
                    trader_summary = {}

                # Build system prompt
                system_prompt = _build_code_evolution_system_prompt(
                    strategy_source=current_source,
                    strategy_slug=strategy_slug,
                    class_name=strategy_class_name,
                    trader_summary=trader_summary,
                    baseline_score=best_score,
                    iteration_num=iteration_num,
                    mandate=mandate,
                    last_backtest_summary=last_backtest_summary,
                    allowed_imports=ALLOWED_IMPORT_PREFIXES,
                )

                tools = _build_code_evolution_tools(trader_id, experiment_id, strategy_id)

                # Run LLM agent
                proposed_source = ""
                reasoning = ""
                confidence = 0.0
                tokens_used = 0

                try:
                    result = None
                    async for agent_event in _run_agent_to_completion_streaming(
                        system_prompt=system_prompt,
                        query=f"Propose code improvements for strategy '{strategy_slug}' (iteration {iteration_num}). Current score: {best_score:.4f}. Use propose_code_change to submit your modified source code.",
                        tools=tools,
                        model=model,
                        max_iterations=10,
                        session_type="autoresearch_code",
                        iteration_num=iteration_num,
                    ):
                        if agent_event["event"] == "agent_done":
                            result = agent_event["data"]
                        else:
                            yield agent_event
                    if result is None:
                        raise RuntimeError("Agent completed without producing a result")
                    answer = result.get("result", {}).get("answer", "")
                    tokens_used = result.get("result", {}).get("tokens_used", 0) or 0

                    # Check if propose_code_change was called (stored in tool results)
                    # The tool handler stores the proposal in a closure variable
                    proposal = _code_proposal_store.get(experiment_id)
                    if proposal:
                        proposed_source = proposal.get("source_code", "")
                        reasoning = proposal.get("reasoning", "")
                        confidence = proposal.get("confidence", 0.0)
                        _code_proposal_store.pop(experiment_id, None)
                    else:
                        # Try extracting from the answer text
                        parsed = _extract_json_payload(answer)
                        if parsed and parsed.get("source_code"):
                            proposed_source = str(parsed["source_code"])
                            reasoning = str(parsed.get("reasoning", ""))
                            confidence = float(parsed.get("confidence", 0.0))
                        else:
                            reasoning = f"Agent did not propose code changes: {answer[:300]}"

                except Exception as exc:
                    reasoning = f"LLM agent error: {exc}"
                    logger.warning("Code evolution iteration %d error: %s", iteration_num, exc)

                yield {
                    "event": "proposal",
                    "data": {
                        "iteration": iteration_num,
                        "has_code": bool(proposed_source),
                        "reasoning": reasoning[:500],
                        "confidence": confidence,
                    },
                }

                # Evaluate proposed code
                decision = "reverted"
                new_score = best_score
                score_delta = 0.0
                source_diff = ""
                validation_result: dict = {}
                backtest_result_dict: dict = {}

                if proposed_source and proposed_source.strip() != current_source.strip():
                    # Validate
                    validation_result = validate_strategy_source(proposed_source, strategy_class_name)

                    if not validation_result.get("valid"):
                        reasoning = f"Validation failed: {validation_result.get('errors', [])}"
                        reverted_count += 1
                    elif len(proposed_source) > 50_000:
                        reasoning = "Source code exceeds 50KB limit"
                        reverted_count += 1
                    else:
                        # Compute diff
                        diff_lines = list(difflib.unified_diff(
                            current_source.splitlines(keepends=True),
                            proposed_source.splitlines(keepends=True),
                            fromfile="current",
                            tofile="proposed",
                            lineterm="",
                        ))
                        source_diff = "\n".join(diff_lines)

                        if not diff_lines:
                            reasoning = "No meaningful code changes detected"
                            reverted_count += 1
                        else:
                            # Backtest new code — UNIFIED runner.
                            try:
                                bt_result = await run_unified_backtest(
                                    source_code=proposed_source,
                                    slug=strategy_slug,
                                    config=strategy_config,
                                )
                                new_score = _code_evolution_score(bt_result)
                                score_delta = round(new_score - best_score, 4)
                                backtest_result_dict = bt_result
                                last_backtest_summary = _summarize_unified_for_prompt(bt_result)

                                bt_success = bool((bt_result.get("execution") or {}).get("success"))
                                if new_score > best_score and bt_success:
                                    decision = "kept"
                                    current_source = proposed_source
                                    best_score = new_score
                                    best_source = proposed_source
                                    kept_count += 1

                                    # Create strategy version
                                    new_version_num = baseline_version_num
                                    if auto_apply:
                                        try:
                                            async with AsyncSessionLocal() as session:
                                                strat = await session.get(Strategy, strategy_id)
                                                if strat:
                                                    strat.source_code = proposed_source
                                                    version_snapshot = await create_strategy_version_snapshot(
                                                        session,
                                                        strategy=strat,
                                                        reason="autoresearch_code_evolution",
                                                        created_by="autoresearch",
                                                        parent_version=best_version_num,
                                                        commit=True,
                                                    )
                                                    new_version_num = version_snapshot.version
                                                    best_version_num = new_version_num
                                            await bump_strategy_runtime_revisions(
                                                source_key=strategy_source_key
                                            )
                                        except Exception as exc:
                                            logger.warning("Failed to create strategy version: %s", exc)
                                else:
                                    reverted_count += 1

                            except Exception as exc:
                                reasoning = f"Backtest error: {exc}"
                                reverted_count += 1
                else:
                    if not proposed_source:
                        reasoning = reasoning or "No code proposed"
                    else:
                        reasoning = "Proposed code is identical to current"
                    reverted_count += 1

                duration = round(time.monotonic() - iter_start, 2)

                # Persist iteration
                async with AsyncSessionLocal() as session:
                    session.add(AutoresearchIteration(
                        id=str(uuid.uuid4()),
                        experiment_id=experiment_id,
                        iteration_number=iteration_num,
                        proposed_params_json=None,
                        baseline_score=round(best_score - score_delta if decision == "kept" else best_score, 4),
                        new_score=round(new_score, 4),
                        score_delta=score_delta,
                        decision=decision,
                        reasoning=reasoning[:2000],
                        backtest_result_json=backtest_result_dict or None,
                        changed_params_json=None,
                        source_code_snapshot=proposed_source[:50_000] if proposed_source else None,
                        source_diff=source_diff[:20_000] if source_diff else None,
                        validation_result_json=validation_result or None,
                        duration_seconds=duration,
                        tokens_used=tokens_used,
                        created_at=utcnow(),
                    ))
                    await session.execute(
                        update(AutoresearchExperiment)
                        .where(AutoresearchExperiment.id == experiment_id)
                        .values(
                            iteration_count=iteration_num,
                            kept_count=kept_count,
                            reverted_count=reverted_count,
                            best_score=best_score,
                            best_source_code=best_source,
                            best_version=best_version_num,
                            updated_at=utcnow(),
                        )
                    )
                    await session.commit()

                yield {
                    "event": "decision",
                    "data": {
                        "iteration": iteration_num,
                        "decision": decision,
                        "new_score": round(new_score, 4),
                        "score_delta": score_delta,
                        "best_score": round(best_score, 4),
                        "source_diff": source_diff[:20_000] if source_diff else None,
                        "source_diff_lines": len(source_diff.splitlines()) if source_diff else 0,
                        "validation_passed": validation_result.get("valid", False) if validation_result else None,
                        "validation_result": validation_result or None,
                        "backtest_result": backtest_result_dict or None,
                        "reasoning": reasoning[:300],
                        "duration_seconds": duration,
                        "kept_count": kept_count,
                        "reverted_count": reverted_count,
                    },
                }

                await asyncio.sleep(1)

        except Exception as exc:
            logger.exception("Code evolution loop error: %s", exc)
            yield {"event": "error", "data": {"error": str(exc)}}
        finally:
            final_status = "paused" if stop_event.is_set() else "completed"
            async with AsyncSessionLocal() as session:
                await session.execute(
                    update(AutoresearchExperiment)
                    .where(AutoresearchExperiment.id == experiment_id)
                    .values(status=final_status, finished_at=utcnow(), updated_at=utcnow())
                )
                await session.commit()
            self._active_experiments.pop(session_key, None)
            self._stop_flags.pop(session_key, None)

        yield {
            "event": "done",
            "data": {
                "experiment_id": experiment_id,
                "mode": "code",
                "strategy_slug": strategy_slug,
                "total_iterations": kept_count + reverted_count,
                "best_score": round(best_score, 4),
                "baseline_score": round(baseline_score, 4),
                "improvement": round(best_score - baseline_score, 4),
                "baseline_version": baseline_version_num if baseline_version_num != best_version_num else None,
                "best_version": best_version_num,
                "can_create_ab_experiment": best_version_num != baseline_version_num,
                "kept_count": kept_count,
                "reverted_count": reverted_count,
            },
        }


# ---------------------------------------------------------------------------
# Code evolution scoring
# ---------------------------------------------------------------------------


def _code_evolution_score(
    result: dict,
    *,
    walk_forward_stability_pct: float | None = None,
) -> float:
    """Risk-adjusted iteration score for the unified backtest result.

    Returns a single float that the autoresearch loop compares against
    the running best to decide keep-vs-revert.  The current shape is
    intentionally trivial — it just returns the observed annualized
    Sharpe — because the full risk-adjusted formula and the walk-forward
    gate are being staged across separate commits.  Commit 2 swaps in
    the institutional formula; commit 4 wires in walk-forward stability.
    """
    if not isinstance(result, dict):
        return -1.0
    execution = result.get("execution") or {}
    if not execution.get("success"):
        return -1.0
    sharpe = execution.get("sharpe") or {}
    if isinstance(sharpe, dict):
        v = sharpe.get("value")
    else:
        v = sharpe
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _summarize_unified_for_prompt(result: dict) -> dict:
    """Compact summary of a unified backtest result for the LLM prompt.

    Commit 3 enriches this with Cox C-index, ensemble band, regime
    breakdown, etc.; here we just keep the prompt readable by stripping
    the noisy nested arrays.
    """
    if not isinstance(result, dict):
        return {}
    execution = result.get("execution") or {}
    sharpe = execution.get("sharpe") or {}
    return {
        "success": execution.get("success"),
        "trade_count": execution.get("trade_count"),
        "total_return_pct": execution.get("total_return_pct"),
        "sharpe": sharpe.get("value") if isinstance(sharpe, dict) else sharpe,
        "max_drawdown_pct": execution.get("max_drawdown_pct"),
        "validation_errors": execution.get("validation_errors"),
        "runtime_error": execution.get("runtime_error"),
    }


# ---------------------------------------------------------------------------
# Code evolution agent tools
# ---------------------------------------------------------------------------

# Global store for code proposals (tool handler → loop communication)
_code_proposal_store: dict[str, dict] = {}


def _build_code_evolution_tools(
    trader_id: str | None, experiment_id: str, strategy_id: str
) -> list[AgentTool]:
    """Build tool set for the code evolution proposal agent.

    ``trader_id`` is optional — strategy-scoped code experiments don't
    have a bot context. The tools defined here only consult the strategy
    and the iteration history, never the trader, so passing ``None`` is
    safe.
    """

    async def _get_strategy_source(_args: dict) -> dict:
        async with AsyncSessionLocal() as session:
            row = await session.get(Strategy, strategy_id)
            if row is None:
                return {"error": "Strategy not found"}
            return {
                "slug": row.slug,
                "name": row.name,
                "class_name": row.class_name,
                "source_key": row.source_key,
                "version": row.version,
                "enabled": row.enabled,
                "source_code": row.source_code,
                "config": dict(row.config or {}),
            }

    async def _get_strategy_template_tool(_args: dict) -> dict:
        from services.strategy_loader import STRATEGY_TEMPLATE, ALLOWED_IMPORT_PREFIXES
        return {
            "template": STRATEGY_TEMPLATE,
            "allowed_imports": sorted(ALLOWED_IMPORT_PREFIXES),
        }

    async def _get_backtest_results(args: dict) -> dict:
        """Get most recent backtest or iteration results."""
        async with AsyncSessionLocal() as session:
            row = (
                await session.execute(
                    select(AutoresearchIteration)
                    .where(AutoresearchIteration.experiment_id == experiment_id)
                    .order_by(desc(AutoresearchIteration.iteration_number))
                    .limit(1)
                )
            ).scalar_one_or_none()
            if row and row.backtest_result_json:
                return {"backtest": row.backtest_result_json, "iteration": row.iteration_number, "score": row.new_score}
        return {"backtest": None, "message": "No backtest results yet (baseline will be computed)"}

    async def _get_iteration_history_code(args: dict) -> dict:
        limit = max(1, min(int(args.get("limit") or 10), 30))
        async with AsyncSessionLocal() as session:
            rows = (
                await session.execute(
                    select(AutoresearchIteration)
                    .where(AutoresearchIteration.experiment_id == experiment_id)
                    .order_by(desc(AutoresearchIteration.iteration_number))
                    .limit(limit)
                )
            ).scalars().all()
        return {
            "iterations": [
                {
                    "iteration": r.iteration_number,
                    "score": r.new_score,
                    "delta": r.score_delta,
                    "decision": r.decision,
                    "reasoning": (r.reasoning or "")[:300],
                    "source_diff": (r.source_diff or "")[:1000],
                    "validation_passed": (r.validation_result_json or {}).get("valid"),
                }
                for r in rows
            ]
        }

    async def _get_cortex_memories_code(args: dict) -> dict:
        query = str(args.get("query") or "").strip()
        limit = max(1, min(int(args.get("limit") or 10), 20))
        async with AsyncSessionLocal() as session:
            stmt = (
                select(CortexMemory)
                .where(CortexMemory.expired == False)  # noqa: E712
                .order_by(CortexMemory.importance.desc())
                .limit(limit)
            )
            rows = (await session.execute(stmt)).scalars().all()
        query_lower = query.lower()
        filtered = [
            {"id": r.id, "category": r.category, "content": r.content, "importance": r.importance}
            for r in rows
            if not query or query_lower in (r.content or "").lower()
        ]
        return {"memories": filtered[:limit]}

    async def _propose_code_change(args: dict) -> dict:
        source_code = str(args.get("source_code") or "").strip()
        reasoning = str(args.get("reasoning") or "").strip()
        confidence_val = float(args.get("confidence") or 0.5)
        if not source_code:
            return {"error": "source_code is required"}
        # Pre-validate
        from services.strategy_loader import validate_strategy_source
        validation = validate_strategy_source(source_code)
        if not validation.get("valid"):
            return {"error": "Validation failed", "errors": validation.get("errors", []), "warnings": validation.get("warnings", [])}
        # Store for the loop to pick up
        _code_proposal_store[experiment_id] = {
            "source_code": source_code,
            "reasoning": reasoning,
            "confidence": confidence_val,
        }
        return {"status": "ok", "message": "Code proposal submitted for evaluation", "class_name": validation.get("class_name")}

    return [
        AgentTool(
            name="get_strategy_source",
            description="Get the current strategy source code, class name, config, and version.",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=_get_strategy_source,
            max_calls=3,
            category="autoresearch",
        ),
        AgentTool(
            name="get_strategy_template",
            description="Get the BaseStrategy template and list of allowed imports. Use this to understand the strategy API contract.",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=_get_strategy_template_tool,
            max_calls=2,
            category="autoresearch",
        ),
        AgentTool(
            name="get_backtest_results",
            description="Get the most recent backtest result for this strategy (from baseline or last iteration).",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=_get_backtest_results,
            max_calls=3,
            category="autoresearch",
        ),
        AgentTool(
            name="get_iteration_history",
            description="Get recent iteration results including source diffs, scores, and validation status.",
            parameters={
                "type": "object",
                "properties": {"limit": {"type": "integer", "description": "Max iterations (default 10)."}},
                "required": [],
            },
            handler=_get_iteration_history_code,
            max_calls=3,
            category="autoresearch",
        ),
        AgentTool(
            name="get_cortex_memories",
            description="Search Cortex memories for strategy lessons and observations.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": [],
            },
            handler=_get_cortex_memories_code,
            max_calls=2,
            category="autoresearch",
        ),
        AgentTool(
            name="get_trader_context",
            description="Fetch trader runtime context, recent decisions, orders, and signals.",
            parameters={
                "type": "object",
                "properties": {
                    "trader_id": {"type": "string"},
                    "decision_limit": {"type": "integer"},
                    "order_limit": {"type": "integer"},
                },
                "required": ["trader_id"],
            },
            handler=_tool_get_trader_context,
            max_calls=2,
            category="autoresearch",
        ),
        AgentTool(
            name="propose_code_change",
            description=(
                "Submit modified strategy source code for evaluation. "
                "Provide the COMPLETE modified source code (not just a diff). "
                "The code will be validated for syntax and import safety, then backtested."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source_code": {"type": "string", "description": "The complete modified strategy Python source code."},
                    "reasoning": {"type": "string", "description": "Why these changes should improve performance."},
                    "confidence": {"type": "number", "description": "Confidence level 0.0-1.0."},
                },
                "required": ["source_code", "reasoning"],
            },
            handler=_propose_code_change,
            max_calls=1,
            category="autoresearch",
        ),
    ]


# ---------------------------------------------------------------------------
# Code evolution system prompt
# ---------------------------------------------------------------------------


def _build_code_evolution_system_prompt(
    strategy_source: str,
    strategy_slug: str,
    class_name: str,
    trader_summary: dict,
    baseline_score: float,
    iteration_num: int,
    mandate: str | None,
    last_backtest_summary: dict | None,
    allowed_imports: set | None = None,
) -> str:
    """Build system prompt for the code evolution agent."""
    trader_json = json.dumps(trader_summary, indent=2, default=str) if trader_summary else "{}"
    backtest_json = json.dumps(last_backtest_summary, indent=2, default=str) if last_backtest_summary else "No backtest data yet"
    imports_list = ", ".join(sorted(allowed_imports or [])) or "standard library modules"

    mandate_section = ""
    if mandate:
        mandate_section = f"\n## Custom Constraints\n{mandate}\n"

    return f"""\
You are the Homerun Code Evolution agent — you improve prediction market trading
strategy source code through iterative modifications and backtesting.

## Current State
- **Strategy**: {strategy_slug} (class: {class_name})
- **Iteration**: {iteration_num}
- **Baseline score**: {baseline_score:.4f}
- **Trader summary**: {trader_json}
- **Last backtest**: {backtest_json}

## Scoring Formula
Score = opportunity_count * 0.3 + avg_roi * 0.3 + quality_pass_rate * 0.25 + diversity * 0.15
Higher is better. Your goal is to maximize this score via code changes.

## Current Strategy Source Code
```python
{strategy_source}
```

## Instructions
1. Use `get_strategy_template` to understand the BaseStrategy API contract.
2. Use `get_iteration_history` to see what changes have been tried before.
3. Use `get_backtest_results` to understand current detection performance.
4. Use `get_cortex_memories` for strategy-related lessons.
5. Analyze the strategy code and identify specific improvements:
   - Better detection logic in detect() / detect_async()
   - Improved filtering or scoring in evaluate()
   - Smarter exit conditions in should_exit()
   - Better use of market data, prices, or event information
6. Use `propose_code_change` to submit the COMPLETE modified source code.

## Rules
- Submit the COMPLETE source code, not just a diff.
- Maintain the same class name: `{class_name}`
- Must extend BaseStrategy and implement detect/detect_async.
- Only use allowed imports: {imports_list}
- Be CONSERVATIVE — make targeted changes to specific methods, don't rewrite everything.
- Each iteration should change 1-3 things. Small incremental improvements.
- Learn from past iterations — don't repeat changes that were reverted.
- Preserve any working detection logic; add to it rather than replacing it.
{mandate_section}"""


# Singleton instance
autoresearch_service = AutoresearchService()
