"""Strategy management tools — list, inspect, tune, backtest, create."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def build_tools() -> list:
    from services.ai.agent import AgentTool

    return [
        AgentTool(
            name="list_strategies",
            description=(
                "List all loaded strategies with their current status — active, paused, "
                "errored. Includes key config like max_positions, filters, and runtime stats."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            handler=_list_strategies,
            max_calls=10,
            category="strategies",
        ),
        AgentTool(
            name="get_strategy_details",
            description=(
                "Get full details for a specific strategy including its configuration, "
                "source code, runtime status, and recent performance metrics."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "strategy_id": {
                        "type": "string",
                        "description": "Strategy ID or slug",
                    },
                },
                "required": ["strategy_id"],
            },
            handler=_get_strategy_details,
            max_calls=10,
            category="strategies",
        ),
        AgentTool(
            name="get_strategy_performance",
            description=(
                "Get performance metrics for a strategy — trades made, win rate, "
                "P&L, average hold time, and recent decisions."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "strategy_id": {
                        "type": "string",
                        "description": "Strategy ID or slug",
                    },
                },
                "required": ["strategy_id"],
            },
            handler=_get_strategy_performance,
            max_calls=10,
            category="strategies",
        ),
        AgentTool(
            name="update_strategy_config",
            description=(
                "Update configuration parameters for a strategy. Pass a JSON object "
                "of config keys to change (e.g. max_positions, min_edge, filters). "
                "The strategy will be hot-reloaded with new config."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "strategy_id": {
                        "type": "string",
                        "description": "Strategy ID or slug",
                    },
                    "config_updates": {
                        "type": "object",
                        "description": "Key-value pairs of config to update",
                    },
                },
                "required": ["strategy_id", "config_updates"],
            },
            handler=_update_strategy_config,
            max_calls=10,
            category="strategies",
        ),
        AgentTool(
            name="validate_strategy_code",
            description=(
                "Validate strategy source code without deploying it. Checks syntax, "
                "required methods, import safety, and config schema. Returns any "
                "errors or warnings found."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source_code": {
                        "type": "string",
                        "description": "Python source code of the strategy",
                    },
                    "class_name": {
                        "type": "string",
                        "description": "Expected class name in the source",
                    },
                },
                "required": ["source_code"],
            },
            handler=_validate_strategy_code,
            max_calls=10,
            category="strategies",
        ),
        AgentTool(
            name="run_strategy_backtest",
            description=(
                "Run a backtest for a strategy over historical data. Returns simulated "
                "trades, P&L curve, win rate, Sharpe ratio, and max drawdown."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "strategy_id": {
                        "type": "string",
                        "description": "Strategy ID or slug to backtest",
                    },
                    "lookback_days": {
                        "type": "integer",
                        "description": "Days of history to backtest over (default 30)",
                        "default": 30,
                    },
                },
                "required": ["strategy_id"],
            },
            handler=_run_strategy_backtest,
            max_calls=5,
            category="strategies",
        ),
        AgentTool(
            name="create_strategy",
            description=(
                "Create a new strategy from source code and configuration. The strategy "
                "will be validated, saved to the database, and loaded into the runtime."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Human-readable strategy name",
                    },
                    "slug": {
                        "type": "string",
                        "description": "Unique slug identifier (lowercase, underscores)",
                    },
                    "source_code": {
                        "type": "string",
                        "description": "Python source code",
                    },
                    "config": {
                        "type": "object",
                        "description": "Initial configuration",
                        "default": {},
                    },
                },
                "required": ["name", "slug", "source_code"],
            },
            handler=_create_strategy,
            max_calls=10,
            category="strategies",
        ),
    ]


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


async def _list_strategies(args: dict) -> dict:
    try:
        from models.database import AsyncSessionLocal, Strategy
        from sqlalchemy import select

        async with AsyncSessionLocal() as session:
            result = await session.execute(select(Strategy))
            rows = result.scalars().all()

        strategies = []
        for r in rows:
            strategies.append({
                "id": r.id,
                "name": getattr(r, "name", None) or r.id,
                "slug": getattr(r, "slug", None),
                "enabled": getattr(r, "enabled", True),
                "source_key": getattr(r, "source_key", None),
                "config": getattr(r, "config", {}),
                "created_at": r.created_at.isoformat() if getattr(r, "created_at", None) else None,
            })

        # Also get runtime statuses
        try:
            from services.strategy_loader import strategy_loader
            loader = strategy_loader
            if loader:
                for s in strategies:
                    slug = s.get("slug") or s.get("id")
                    status = loader.get_runtime_status(slug)
                    if status:
                        s["runtime_status"] = status
        except Exception:
            pass

        return {"strategies": strategies, "count": len(strategies)}
    except Exception as exc:
        logger.error("list_strategies failed: %s", exc)
        return {"error": str(exc)}


async def _get_strategy_details(args: dict) -> dict:
    try:
        from models.database import AsyncSessionLocal, Strategy
        from sqlalchemy import select

        sid = args["strategy_id"]

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Strategy).where(
                    (Strategy.id == sid) | (Strategy.slug == sid)
                )
            )
            row = result.scalar_one_or_none()

        if not row:
            return {"error": f"Strategy not found: {sid}"}

        detail = {
            "id": row.id,
            "name": getattr(row, "name", None),
            "slug": getattr(row, "slug", None),
            "enabled": getattr(row, "enabled", True),
            "source_code": getattr(row, "source_code", None),
            "config": getattr(row, "config", {}),
            "source_key": getattr(row, "source_key", None),
            "created_at": row.created_at.isoformat() if getattr(row, "created_at", None) else None,
            "updated_at": row.updated_at.isoformat() if getattr(row, "updated_at", None) else None,
        }

        # Runtime status
        try:
            from services.strategy_loader import strategy_loader
            loader = strategy_loader
            if loader:
                slug = detail.get("slug") or detail.get("id")
                detail["runtime_status"] = loader.get_runtime_status(slug)
        except Exception:
            pass

        return detail
    except Exception as exc:
        logger.error("get_strategy_details failed: %s", exc)
        return {"error": str(exc)}


async def _get_strategy_performance(args: dict) -> dict:
    try:
        from models.database import AsyncSessionLocal, TraderOrder
        from sqlalchemy import func, select
        from services.trader_orchestrator_state import (
            REALIZED_ORDER_STATUSES,
            REALIZED_WIN_ORDER_STATUSES,
            REALIZED_LOSS_ORDER_STATUSES,
        )

        sid = args["strategy_id"]

        # Normalize status values for case-insensitive matching
        realized_statuses = tuple(REALIZED_ORDER_STATUSES)
        win_statuses = tuple(REALIZED_WIN_ORDER_STATUSES)
        loss_statuses = tuple(REALIZED_LOSS_ORDER_STATUSES)

        status_col = func.lower(func.trim(func.coalesce(TraderOrder.status, "")))

        async with AsyncSessionLocal() as session:
            # Count only RESOLVED trades (wins + losses) — excludes cancelled,
            # failed, submitted, open, etc. that never actually entered/settled.
            resolved_result = await session.execute(
                select(func.count(TraderOrder.id)).where(
                    TraderOrder.strategy_key == sid,
                    status_col.in_(realized_statuses),
                )
            )
            resolved_count = resolved_result.scalar() or 0

            # Win count — only resolved wins
            win_result = await session.execute(
                select(func.count(TraderOrder.id)).where(
                    TraderOrder.strategy_key == sid,
                    status_col.in_(win_statuses),
                )
            )
            wins = win_result.scalar() or 0

            # Loss count
            loss_result = await session.execute(
                select(func.count(TraderOrder.id)).where(
                    TraderOrder.strategy_key == sid,
                    status_col.in_(loss_statuses),
                )
            )
            losses = loss_result.scalar() or 0

            # Total orders (all statuses) for context
            total_result = await session.execute(
                select(func.count(TraderOrder.id)).where(
                    TraderOrder.strategy_key == sid,
                )
            )
            total_orders = total_result.scalar() or 0

            # Realized P&L — only from resolved orders
            total_pnl_result = await session.execute(
                select(func.sum(TraderOrder.actual_profit)).where(
                    TraderOrder.strategy_key == sid,
                    status_col.in_(realized_statuses),
                    TraderOrder.actual_profit.isnot(None),
                )
            )
            total_pnl = round(float(total_pnl_result.scalar() or 0), 4)

            # Status breakdown for transparency
            status_breakdown_result = await session.execute(
                select(
                    TraderOrder.status,
                    func.count(TraderOrder.id),
                )
                .where(TraderOrder.strategy_key == sid)
                .group_by(TraderOrder.status)
            )
            status_breakdown = {
                str(row[0] or "unknown"): row[1]
                for row in status_breakdown_result.all()
            }

            # Get recent orders
            result2 = await session.execute(
                select(TraderOrder)
                .where(TraderOrder.strategy_key == sid)
                .order_by(TraderOrder.created_at.desc())
                .limit(10)
            )
            recent = result2.scalars().all()

        recent_trades = []
        for r in recent:
            recent_trades.append({
                "id": r.id,
                "market_question": r.market_question or r.market_id,
                "direction": r.direction,
                "mode": r.mode,
                "status": r.status,
                "notional_usd": float(r.notional_usd) if r.notional_usd else None,
                "entry_price": float(r.entry_price) if r.entry_price else None,
                "actual_profit": float(r.actual_profit) if r.actual_profit else None,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            })

        return {
            "strategy_id": sid,
            "resolved_trades": resolved_count,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / resolved_count, 4) if resolved_count > 0 else 0.0,
            "total_pnl_usd": total_pnl,
            "total_orders_all_statuses": total_orders,
            "status_breakdown": status_breakdown,
            "recent_trades": recent_trades,
        }
    except Exception as exc:
        logger.error("get_strategy_performance failed: %s", exc)
        return {"error": str(exc)}


async def _update_strategy_config(args: dict) -> dict:
    try:
        from models.database import AsyncSessionLocal, Strategy
        from sqlalchemy import select

        sid = args["strategy_id"]
        updates = args["config_updates"]

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Strategy).where(
                    (Strategy.id == sid) | (Strategy.slug == sid)
                )
            )
            row = result.scalar_one_or_none()
            if not row:
                return {"error": f"Strategy not found: {sid}"}

            current_config = dict(row.config or {})
            current_config.update(updates)
            row.config = current_config
            await session.commit()

            strategy_slug = getattr(row, "slug", None) or row.id

        # Hot-reload
        try:
            from services.strategy_loader import strategy_loader
            loader = strategy_loader
            if loader:
                await loader.reload_from_db(strategy_slug)
        except Exception as e:
            logger.warning("Hot-reload after config update failed: %s", e)

        return {
            "strategy_id": sid,
            "updated_keys": list(updates.keys()),
            "new_config": current_config,
        }
    except Exception as exc:
        logger.error("update_strategy_config failed: %s", exc)
        return {"error": str(exc)}


async def _validate_strategy_code(args: dict) -> dict:
    try:
        from services.strategy_loader import strategy_loader

        source_code = args["source_code"]
        class_name = args.get("class_name")

        loader = strategy_loader
        if loader is None:
            # Fallback: basic AST validation
            import ast
            try:
                ast.parse(source_code)
                return {"valid": True, "errors": [], "warnings": ["StrategyLoader not available for full validation"]}
            except SyntaxError as se:
                return {"valid": False, "errors": [f"Syntax error: {se}"]}

        result = loader.validate_strategy_source(source_code, class_name)
        return result if isinstance(result, dict) else {"valid": True, "result": result}
    except Exception as exc:
        logger.error("validate_strategy_code failed: %s", exc)
        return {"error": str(exc)}


async def _run_strategy_backtest(args: dict) -> dict:
    """Run a unified backtest for a strategy by ID or slug.

    Looks up the strategy's current source_code + config, then dispatches
    to the same Cox-aware execution-realistic engine BacktestStudio uses.
    Returns a compact summary so the LLM gets the headline numbers
    without drowning in the full nested result.
    """
    try:
        from datetime import datetime, timedelta, timezone
        from sqlalchemy import or_, select

        from models.database import AsyncSessionLocal, Strategy
        from services.backtest.unified_runner import run_unified_backtest

        sid = str(args.get("strategy_id") or "").strip()
        if not sid:
            return {"error": "strategy_id is required"}
        lookback = int(args.get("lookback_days") or 30)
        lookback = max(1, min(180, lookback))

        async with AsyncSessionLocal() as session:
            row = (
                await session.execute(
                    select(Strategy).where(or_(Strategy.id == sid, Strategy.slug == sid))
                )
            ).scalar_one_or_none()
            if row is None:
                return {"error": f"Strategy '{sid}' not found"}
            source = str(row.source_code or "")
            slug = str(row.slug or sid)
            cfg = dict(row.config or {})

        if not source.strip():
            return {"error": f"Strategy '{slug}' has no source_code"}

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback)

        result = await run_unified_backtest(
            source_code=source,
            slug=slug,
            config=cfg,
            start=start,
            end=end,
        )

        execution = result.get("execution") or {}
        sharpe = execution.get("sharpe") or {}
        deflated = result.get("deflated_sharpe") or {}
        return {
            "run_id": result.get("run_id"),
            "strategy_slug": slug,
            "lookback_days": lookback,
            "success": execution.get("success"),
            "trade_count": execution.get("trade_count"),
            "total_return_pct": execution.get("total_return_pct"),
            "sharpe": sharpe.get("value") if isinstance(sharpe, dict) else sharpe,
            "max_drawdown_pct": execution.get("max_drawdown_pct"),
            "deflated_sharpe_probability": deflated.get("deflated_sharpe"),
            "fees_paid_usd": execution.get("fees_paid_usd"),
            "runtime_error": execution.get("runtime_error"),
        }
    except Exception as exc:
        logger.error("run_strategy_backtest failed: %s", exc)
        return {"error": str(exc)}


async def _create_strategy(args: dict) -> dict:
    try:
        from models.database import AsyncSessionLocal, Strategy
        import uuid

        name = args["name"]
        slug = args["slug"]
        source_code = args["source_code"]
        config = args.get("config", {})

        # Validate first
        validation = await _validate_strategy_code({"source_code": source_code})
        if isinstance(validation, dict) and not validation.get("valid", True):
            return {"error": "Validation failed", "details": validation}

        async with AsyncSessionLocal() as session:
            strategy = Strategy(
                id=str(uuid.uuid4()),
                name=name,
                slug=slug,
                source_code=source_code,
                config=config,
                enabled=False,  # Start disabled for safety
            )
            session.add(strategy)
            await session.commit()

        return {
            "created": True,
            "id": strategy.id,
            "slug": slug,
            "name": name,
            "enabled": False,
            "note": "Strategy created in disabled state. Enable it via the UI or update_strategy_config when ready.",
        }
    except Exception as exc:
        logger.error("create_strategy failed: %s", exc)
        return {"error": str(exc)}
