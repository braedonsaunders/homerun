"""Iteration / advanced backtest tools.

Adds the *missing* tools that the MCP surface needs but the existing
registry didn't already cover:

  * ``start_param_iteration``     — kick off strategy_params autoresearch
  * ``get_iteration_status``      — poll current best_score / kept / etc.
  * ``stop_iteration``            — halt
  * ``run_walk_forward``          — overfit check on a candidate config
  * ``get_drift_report``          — backtest-vs-live divergence
  * ``get_recent_opportunities``  — what a strategy fired on lately
  * ``get_backtest_run``          — fetch by run_id (LRU cache)
  * ``list_backtest_runs``        — recent runs

These coexist with the existing tools in ``strategy_tools.py``,
``data_tools.py``, etc.  Registered the same way (``build_tools()``);
the central registry in ``services/ai/tools/__init__.py`` picks them
up after we add this module to its import list.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


def build_tools() -> list:
    from services.ai.agent import AgentTool

    return [
        AgentTool(
            name="start_param_iteration",
            description=(
                "Kick off a strategy-scoped LLM-driven param iteration. The agent "
                "proposes overrides against the strategy's declared param_schema, "
                "each candidate is evaluated via the unified backtest engine "
                "(Cox fills + walk-forward gate + deflated Sharpe), and kept "
                "iterations are persisted to Strategy.config when auto_apply=true. "
                "Stops on max_iterations, max_no_improvement, or when "
                "best_score >= target_score. Returns immediately with experiment_id; "
                "poll get_iteration_status to track progress."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "strategy_id": {
                        "type": "string",
                        "description": "Strategy UUID or slug",
                    },
                    "target_score": {
                        "type": "number",
                        "description": "Stop early when best_score >= this value (composite of Sharpe×DSR×WF − DD penalty)",
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "Hard upper bound on iterations (default 50)",
                        "default": 50,
                    },
                    "max_no_improvement": {
                        "type": "integer",
                        "description": "Stop after N non-improving iterations (default 10)",
                        "default": 10,
                    },
                    "mandate": {
                        "type": "string",
                        "description": "Free-form constraint passed to the LLM (e.g. 'minimize drawdown')",
                    },
                    "auto_apply": {
                        "type": "boolean",
                        "description": "Persist kept overrides to Strategy.config (default true)",
                        "default": True,
                    },
                    "model": {
                        "type": "string",
                        "description": "LLM model override (uses platform default when omitted)",
                    },
                },
                "required": ["strategy_id"],
            },
            handler=_start_param_iteration,
            max_calls=2,
            category="iteration",
        ),
        AgentTool(
            name="get_iteration_status",
            description=(
                "Poll the latest strategy-scoped param iteration session for a "
                "strategy. Returns current status, iteration_count, best_score, "
                "baseline_score, kept/reverted counts, and the most recent "
                "iteration log entries."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "strategy_id": {
                        "type": "string",
                        "description": "Strategy UUID or slug",
                    },
                    "history_limit": {
                        "type": "integer",
                        "description": "Recent iteration rows to include (default 10)",
                        "default": 10,
                    },
                },
                "required": ["strategy_id"],
            },
            handler=_get_iteration_status,
            max_calls=50,
            category="iteration",
        ),
        AgentTool(
            name="stop_iteration",
            description=(
                "Halt the running strategy-scoped param iteration session for "
                "a strategy. No-op when nothing is active."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "strategy_id": {
                        "type": "string",
                        "description": "Strategy UUID or slug",
                    },
                },
                "required": ["strategy_id"],
            },
            handler=_stop_iteration,
            max_calls=5,
            category="iteration",
        ),
        AgentTool(
            name="run_walk_forward",
            description=(
                "Walk-forward overfit check on a strategy. Splits the window "
                "into n_folds chronological train/test pairs and runs the "
                "unified backtest on each. Use BEFORE committing param changes "
                "from start_param_iteration to verify the gains generalize."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "strategy_id": {
                        "type": "string",
                        "description": "Strategy UUID or slug",
                    },
                    "config_overrides": {
                        "type": "object",
                        "description": "Per-key overrides on top of strategy's current config",
                    },
                    "n_folds": {
                        "type": "integer",
                        "description": "Number of folds (default 5)",
                        "default": 5,
                    },
                    "train_ratio": {
                        "type": "number",
                        "description": "Train slice of each fold (default 0.7)",
                        "default": 0.7,
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["anchored", "rolling"],
                        "default": "anchored",
                    },
                    "window_days": {
                        "type": "number",
                        "description": "Total backtest window (default 14)",
                        "default": 14.0,
                    },
                },
                "required": ["strategy_id"],
            },
            handler=_run_walk_forward,
            max_calls=5,
            category="backtest",
        ),
        AgentTool(
            name="get_drift_report",
            description=(
                "Live-vs-backtest divergence report for active strategies. "
                "Surfaces strategies whose live performance has materially "
                "diverged from their most recent backtest — the prime suspect "
                "for stale params or fill-model drift."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "window_days": {
                        "type": "integer",
                        "description": "Lookback window (default 30)",
                        "default": 30,
                    },
                },
                "required": [],
            },
            handler=_get_drift_report,
            max_calls=5,
            category="diagnostics",
        ),
        AgentTool(
            name="get_recent_opportunities",
            description=(
                "Recent opportunities a strategy fired on (from "
                "OpportunityHistory). Useful for 'what did the strategy detect "
                "yesterday' before proposing param changes."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "strategy_id": {
                        "type": "string",
                        "description": "Strategy UUID or slug (omit for all strategies)",
                    },
                    "hours": {
                        "type": "integer",
                        "description": "Lookback window in hours (default 24)",
                        "default": 24,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max rows (default 50)",
                        "default": 50,
                    },
                },
                "required": [],
            },
            handler=_get_recent_opportunities,
            max_calls=10,
            category="diagnostics",
        ),
        AgentTool(
            name="get_backtest_run",
            description=(
                "Fetch a previously-run unified backtest by run_id. "
                "Process-local LRU caches the last 32 runs."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run ID returned by run_strategy_backtest",
                    },
                },
                "required": ["run_id"],
            },
            handler=_get_backtest_run,
            max_calls=10,
            category="backtest",
        ),
        AgentTool(
            name="list_backtest_runs",
            description=(
                "List recent unified backtest runs (process-local LRU, "
                "max 32 deep). Filter by strategy slug."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "slug": {
                        "type": "string",
                        "description": "Strategy slug filter (omit for all)",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 32,
                    },
                },
                "required": [],
            },
            handler=_list_backtest_runs,
            max_calls=5,
            category="backtest",
        ),
        AgentTool(
            name="run_backtest_with_overrides",
            description=(
                "Run the unified backtest for a strategy with per-run "
                "config overrides. Same Cox-aware engine as "
                "run_strategy_backtest but accepts a ``config_overrides`` "
                "dict that gets merged on top of the strategy's current "
                "config — the same way the BacktestStudio dynamic-params "
                "panel does. Use to validate a proposed param set before "
                "committing it via update_strategy_config."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "strategy_id": {
                        "type": "string",
                        "description": "Strategy UUID or slug",
                    },
                    "config_overrides": {
                        "type": "object",
                        "description": "Param overrides on top of current config",
                    },
                    "window_days": {
                        "type": "number",
                        "default": 7.0,
                    },
                    "initial_capital_usd": {
                        "type": "number",
                        "default": 1000.0,
                    },
                    "seed": {
                        "type": "integer",
                        "description": "RNG seed for the matcher (omit for random)",
                    },
                },
                "required": ["strategy_id"],
            },
            handler=_run_backtest_with_overrides,
            max_calls=5,
            category="backtest",
        ),
    ]


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


async def _resolve_strategy_id(strategy_id_or_slug: str) -> str | None:
    """Accept either a UUID or a slug and return the canonical UUID."""
    from models.database import AsyncSessionLocal, Strategy
    from sqlalchemy import or_, select

    s = str(strategy_id_or_slug or "").strip()
    if not s:
        return None
    async with AsyncSessionLocal() as session:
        row = (
            await session.execute(
                select(Strategy).where(or_(Strategy.id == s, Strategy.slug == s))
            )
        ).scalar_one_or_none()
    return row.id if row else None


async def _start_param_iteration(args: dict) -> dict:
    try:
        from services.autoresearch_service import autoresearch_service

        sid = await _resolve_strategy_id(args.get("strategy_id"))
        if sid is None:
            return {"error": f"Strategy '{args.get('strategy_id')}' not found"}

        settings_override: dict[str, Any] = {
            "max_iterations": int(args.get("max_iterations", 50) or 50),
            "max_no_improvement": int(args.get("max_no_improvement", 10) or 10),
            "auto_apply": bool(args.get("auto_apply", True)),
        }
        if args.get("target_score") is not None:
            try:
                settings_override["target_score"] = float(args.get("target_score"))
            except (TypeError, ValueError):
                pass
        if args.get("mandate"):
            settings_override["mandate"] = str(args.get("mandate"))
        if args.get("model"):
            settings_override["model"] = str(args.get("model"))

        # Drain the async generator in a background task — the loop
        # runs for minutes-to-hours; the agent polls via
        # ``get_iteration_status``.
        experiment_id_holder: dict[str, Any] = {}
        started_event = asyncio.Event()
        error_holder: dict[str, str] = {}

        async def _drive_loop() -> None:
            try:
                async for evt in autoresearch_service.run_strategy_params_stream(
                    strategy_id=sid,
                    settings_override=settings_override,
                ):
                    event_type = evt.get("event")
                    if event_type == "experiment_start":
                        data = evt.get("data") or {}
                        experiment_id_holder["id"] = data.get("experiment_id")
                        started_event.set()
                    elif event_type == "error" and "id" not in experiment_id_holder:
                        error_holder["error"] = str(
                            (evt.get("data") or {}).get("error", "unknown")
                        )
                        started_event.set()
            except Exception as exc:
                error_holder["error"] = str(exc)
                started_event.set()

        asyncio.create_task(_drive_loop(), name=f"mcp-param-iter-{sid}")

        try:
            await asyncio.wait_for(started_event.wait(), timeout=120.0)
        except asyncio.TimeoutError:
            return {
                "started": False,
                "error": "Timed out waiting for baseline backtest (>120s)",
            }

        if "error" in error_holder:
            return {"started": False, "error": error_holder["error"]}
        return {
            "started": True,
            "experiment_id": experiment_id_holder.get("id"),
            "strategy_id": sid,
            "mandate": args.get("mandate"),
            "target_score": args.get("target_score"),
            "max_iterations": settings_override["max_iterations"],
            "max_no_improvement": settings_override["max_no_improvement"],
            "auto_apply": settings_override["auto_apply"],
        }
    except Exception as exc:
        logger.exception("start_param_iteration failed")
        return {"error": str(exc)}


async def _get_iteration_status(args: dict) -> dict:
    try:
        from services.autoresearch_service import autoresearch_service

        sid = await _resolve_strategy_id(args.get("strategy_id"))
        if sid is None:
            return {"error": f"Strategy '{args.get('strategy_id')}' not found"}

        history_limit = int(args.get("history_limit", 10) or 10)
        status = await autoresearch_service.get_strategy_params_experiment_status(sid)
        iterations = await autoresearch_service.get_strategy_params_experiment_history(
            sid, experiment_id=status.get("experiment_id"), limit=history_limit
        )
        out = dict(status)
        out["iterations"] = iterations
        return out
    except Exception as exc:
        logger.exception("get_iteration_status failed")
        return {"error": str(exc)}


async def _stop_iteration(args: dict) -> dict:
    try:
        from services.autoresearch_service import autoresearch_service

        sid = await _resolve_strategy_id(args.get("strategy_id"))
        if sid is None:
            return {"error": f"Strategy '{args.get('strategy_id')}' not found"}
        return await autoresearch_service.stop_strategy_params_experiment(sid)
    except Exception as exc:
        logger.exception("stop_iteration failed")
        return {"error": str(exc)}


async def _run_walk_forward(args: dict) -> dict:
    try:
        from models.database import AsyncSessionLocal, Strategy
        from sqlalchemy import or_, select
        from services.backtest.walk_forward import run_walk_forward as _wf

        sid_in = str(args.get("strategy_id") or "").strip()
        if not sid_in:
            return {"error": "strategy_id is required"}

        async with AsyncSessionLocal() as session:
            row = (
                await session.execute(
                    select(Strategy).where(or_(Strategy.id == sid_in, Strategy.slug == sid_in))
                )
            ).scalar_one_or_none()
            if row is None:
                return {"error": f"Strategy '{sid_in}' not found"}
            source_code = str(row.source_code or "")
            slug = str(row.slug or sid_in)
            base_config = dict(row.config or {})

        if not source_code.strip():
            return {"error": "Strategy has no source code"}

        merged_config = dict(base_config)
        overrides = args.get("config_overrides")
        if isinstance(overrides, dict):
            merged_config.update(overrides)

        end = datetime.now(timezone.utc)
        end_ts = end - timedelta(seconds=0)
        start_ts = end_ts - timedelta(days=max(1.0, float(args.get("window_days", 14) or 14)))

        result = await _wf(
            source_code=source_code,
            slug=slug,
            config=merged_config,
            start=start_ts,
            end=end_ts,
            mode=str(args.get("mode") or "anchored"),
            n_folds=int(args.get("n_folds", 5) or 5),
            train_ratio=float(args.get("train_ratio", 0.7) or 0.7),
            concurrency=2,
        )
        if hasattr(result, "to_dict"):
            return result.to_dict()
        if isinstance(result, dict):
            return result
        return {"result": str(result)}
    except Exception as exc:
        logger.exception("run_walk_forward failed")
        return {"error": str(exc)}


async def _get_drift_report(args: dict) -> dict:
    """Live-vs-backtest divergence — same code path /api/backtest/drift uses."""
    try:
        from services.backtest.drift import compute_drift

        window = max(1, min(180, int(args.get("window_days", 30) or 30)))
        result = await compute_drift(window_days=window)
        if hasattr(result, "to_dict"):
            return result.to_dict()
        if isinstance(result, dict):
            return result
        return {"strategies": list(result or []), "flagged_count": 0}
    except Exception as exc:
        logger.exception("get_drift_report failed")
        return {"error": str(exc), "strategies": []}


async def _get_recent_opportunities(args: dict) -> dict:
    try:
        from models.database import AsyncSessionLocal, OpportunityHistory, Strategy
        from sqlalchemy import desc, or_, select

        cutoff = datetime.now(timezone.utc) - timedelta(
            hours=max(1, int(args.get("hours", 24) or 24))
        )
        capped = max(1, min(500, int(args.get("limit", 50) or 50)))

        target_slug = None
        sid_in = str(args.get("strategy_id") or "").strip()
        if sid_in:
            async with AsyncSessionLocal() as session:
                row = (
                    await session.execute(
                        select(Strategy).where(
                            or_(Strategy.id == sid_in, Strategy.slug == sid_in)
                        )
                    )
                ).scalar_one_or_none()
                if row is None:
                    return {"error": f"Strategy '{sid_in}' not found", "items": []}
                target_slug = (row.slug or "").strip().lower() or None

        async with AsyncSessionLocal() as session:
            stmt = select(OpportunityHistory).where(OpportunityHistory.detected_at >= cutoff)
            if target_slug:
                stmt = stmt.where(OpportunityHistory.strategy_type == target_slug)
            stmt = stmt.order_by(desc(OpportunityHistory.detected_at)).limit(capped)
            rows = (await session.execute(stmt)).scalars().all()

        items: list[dict] = []
        for r in rows:
            pdata = r.positions_data if isinstance(r.positions_data, dict) else {}
            ptt = pdata.get("positions_to_take") if isinstance(pdata, dict) else None
            first_pos = ptt[0] if isinstance(ptt, list) and ptt and isinstance(ptt[0], dict) else {}
            items.append({
                "id": r.id,
                "strategy_type": r.strategy_type,
                "title": (r.title or "")[:140],
                "detected_at": r.detected_at.isoformat() if r.detected_at else None,
                "expected_roi": float(r.expected_roi or 0.0),
                "risk_score": float(r.risk_score or 0.0),
                "total_cost": float(r.total_cost or 0.0),
                "market_question": str(first_pos.get("market_question") or "")[:140],
                "market_id": str(first_pos.get("market_id") or ""),
                "token_id": str(first_pos.get("token_id") or ""),
                "side": str(first_pos.get("action") or first_pos.get("side") or ""),
                "price": first_pos.get("price"),
                "notional_usd": first_pos.get("notional_usd"),
            })
        return {
            "items": items,
            "total": len(items),
            "slug_filter": target_slug,
            "since": cutoff.isoformat(),
        }
    except Exception as exc:
        logger.exception("get_recent_opportunities failed")
        return {"error": str(exc), "items": []}


async def _get_backtest_run(args: dict) -> dict:
    try:
        from services.backtest.unified_runner import get_recent_run

        run_id = str(args.get("run_id") or "").strip()
        if not run_id:
            return {"error": "run_id is required"}
        result = get_recent_run(run_id)
        if not result:
            return {"error": f"Run '{run_id}' not in cache (LRU last 32)"}
        return _summarize_backtest(result)
    except Exception as exc:
        logger.exception("get_backtest_run failed")
        return {"error": str(exc)}


async def _list_backtest_runs(args: dict) -> dict:
    try:
        from services.backtest.unified_runner import list_recent_runs

        capped = max(1, min(200, int(args.get("limit", 32) or 32)))
        items = await list_recent_runs(limit=capped)
        slug_filter = (args.get("slug") or "").strip().lower() or None
        if slug_filter and items:
            items = [
                it for it in items
                if str(it.get("slug") or "").strip().lower() == slug_filter
            ]
        return {"items": items or [], "total": len(items or [])}
    except Exception as exc:
        logger.exception("list_backtest_runs failed")
        return {"error": str(exc), "items": []}


async def _run_backtest_with_overrides(args: dict) -> dict:
    try:
        from models.database import AsyncSessionLocal, Strategy
        from sqlalchemy import or_, select
        from services.backtest.unified_runner import run_unified_backtest

        sid_in = str(args.get("strategy_id") or "").strip()
        if not sid_in:
            return {"error": "strategy_id is required"}

        async with AsyncSessionLocal() as session:
            row = (
                await session.execute(
                    select(Strategy).where(or_(Strategy.id == sid_in, Strategy.slug == sid_in))
                )
            ).scalar_one_or_none()
            if row is None:
                return {"error": f"Strategy '{sid_in}' not found"}
            source_code = str(row.source_code or "")
            slug = str(row.slug or sid_in)
            base_config = dict(row.config or {})

        if not source_code.strip():
            return {"error": "Strategy has no source code"}

        merged_config = dict(base_config)
        overrides = args.get("config_overrides")
        if isinstance(overrides, dict):
            merged_config.update(overrides)

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=max(0.5, float(args.get("window_days", 7) or 7)))

        result = await run_unified_backtest(
            source_code=source_code,
            slug=slug,
            config=merged_config,
            start=start,
            end=end,
            initial_capital_usd=float(args.get("initial_capital_usd", 1000) or 1000),
            seed=int(args["seed"]) if args.get("seed") is not None else None,
        )
        return _summarize_backtest(result)
    except Exception as exc:
        logger.exception("run_backtest_with_overrides failed")
        return {"error": str(exc)}


def _summarize_backtest(result: Any) -> dict:
    """Compact summary of a unified backtest result for tool output."""
    if hasattr(result, "to_dict"):
        d = result.to_dict()
    elif isinstance(result, dict):
        d = result
    else:
        return {"raw": str(result)}
    execution = d.get("execution") or {}
    return {
        "run_id": d.get("run_id"),
        "strategy_slug": d.get("strategy_slug"),
        "started_at": d.get("started_at"),
        "execution_success": bool(execution.get("success")),
        "n_trades": execution.get("trade_count"),
        "n_fills": execution.get("total_fills"),
        "total_return_pct": execution.get("total_return_pct"),
        "annualized_return_pct": execution.get("annualized_return_pct"),
        "sharpe": execution.get("sharpe"),
        "sortino": execution.get("sortino"),
        "calmar": execution.get("calmar"),
        "max_drawdown_pct": execution.get("max_drawdown_pct"),
        "hit_rate": execution.get("hit_rate"),
        "profit_factor": execution.get("profit_factor"),
        "expectancy_usd": execution.get("expectancy_usd"),
        "fees_paid_usd": execution.get("fees_paid_usd"),
        "rejected_orders": execution.get("rejected_orders"),
        "cancelled_orders": execution.get("cancelled_orders"),
        "validation_warnings": execution.get("validation_warnings") or [],
        "deflated_sharpe": d.get("deflated_sharpe"),
        "walk_forward": d.get("walk_forward"),
    }
