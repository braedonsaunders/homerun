"""Agent tools the strategy reverse-engineer LLM loop can invoke.

Mirrors the *kind* of tools Claude Code itself uses to solve a problem:

  * **dataset_query / dataset_sample / dataset_distinct**  — read the
    captured market data the same way the operator would in Data Lab.
  * **wallet_trades_window**                               — fetch a
    slice of the wallet's actual trades for inspection.
  * **strategy_sdk_reference**                             — read the
    BaseStrategy contract + scoring helpers it must conform to.
  * **list_existing_strategies / get_strategy_source**     — let the
    agent learn from canonical implementations.
  * **submit_strategy_candidate**                          — the only
    side-effecting tool: persists a new iteration row, runs a
    backtest, scores it against the wallet, returns the breakdown.
  * **finalize_best**                                      — the agent
    calls this when satisfied (or when timing out).  Returns the
    final job summary.

All handlers are pure async functions; they accept a closure-captured
``context`` (job_id + wallet trades + dataset scope) and the
``arguments`` dict the LLM produced.

Designed so that an external MCP client could drive the same loop —
the tool definitions are reused by ``services/ai/tools/strategy_reverse_engineer_tools.py``
(see Wave 8).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from services.ai.agent import AgentTool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-job context — closed over by every tool handler so the LLM never
# has to pass job_id / wallet_address by hand.
# ---------------------------------------------------------------------------


@dataclass
class ReverseEngineerContext:
    job_id: str
    wallet_address: str
    wallet_trades: list[dict[str, Any]]   # normalized dicts (see wallet_profile)
    dataset_scope: dict[str, Any]         # {"token_ids": [...], "start": dt, "end": dt, "labels": [...]}
    on_iteration_submitted: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
    """Callback the agent invokes via ``submit_strategy_candidate``.

    Returns the iteration result dict (score breakdown + backtest_run_id)
    so the LLM can immediately reason about it.
    """
    on_finalize: Callable[[Optional[str]], Awaitable[dict[str, Any]]]
    """Callback the agent calls via ``finalize_best``."""
    on_scope_changed: Optional[
        Callable[[dict[str, Any]], Awaitable[None]]
    ] = None
    """Optional hook the agent invokes after a polybacktest_import call
    succeeds.  Implementations re-resolve the dataset scope (which now
    includes the freshly imported datasets) and re-attach wallet-trade
    aliases so subsequent submit_strategy_candidate calls + scoring
    matches benefit immediately.
    """


# ---------------------------------------------------------------------------
# Tool builders
# ---------------------------------------------------------------------------


def build_tools(ctx: ReverseEngineerContext) -> list[AgentTool]:
    """Return the full ``[AgentTool]`` list for this job.

    Closures bind ``ctx`` so handlers can access wallet trades / dataset
    scope / submit-iteration callback without round-tripping through
    arguments.
    """
    return [
        AgentTool(
            name="describe_objective",
            description=(
                "Recall the high-level objective and the resources you have "
                "available.  Call this first — it lays out the wallet, the "
                "dataset scope, and the rules the final strategy must follow."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            handler=_handler_describe(ctx),
            max_calls=2,
            category="reverse_engineer",
        ),
        AgentTool(
            name="strategy_sdk_reference",
            description=(
                "Return the canonical BaseStrategy contract + scoring helpers + "
                "an example minimal strategy.  Use this to understand the exact "
                "shape your candidate code must follow."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            handler=_handler_sdk_reference(ctx),
            max_calls=3,
            category="reverse_engineer",
        ),
        AgentTool(
            name="list_existing_strategies",
            description=(
                "List the strategies already loaded in the registry.  Useful "
                "for picking an analogous example to learn from."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source_key": {
                        "type": "string",
                        "description": "Filter by source bucket (scanner / traders / crypto / news / weather)",
                    },
                },
                "required": [],
            },
            handler=_handler_list_strategies(ctx),
            max_calls=3,
            category="reverse_engineer",
        ),
        AgentTool(
            name="get_strategy_source",
            description=(
                "Read the full source code of a registered strategy by slug or id.  "
                "Use this after list_existing_strategies to study a working example."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "strategy_id": {"type": "string"},
                },
                "required": ["strategy_id"],
            },
            handler=_handler_get_strategy_source(ctx),
            max_calls=8,
            category="reverse_engineer",
        ),
        AgentTool(
            name="dataset_query",
            description=(
                "Paginated read of any Data Lab dataset (microstructure_snapshot, "
                "book_delta_event, opportunity_history, trader_order, backtest_run, "
                "provider_dataset).  Returns columns + rows in the same shape the "
                "Data Lab UI receives.  Use this to inspect the actual market data "
                "you'll be backtesting against."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Dataset name (see /api/dataset list)",
                    },
                    "limit": {"type": "integer", "default": 50, "minimum": 1, "maximum": 500},
                    "offset": {"type": "integer", "default": 0, "minimum": 0},
                    "filters": {
                        "type": "object",
                        "description": "Filter values keyed by filter.key — e.g. {'token_id': '...', 'start': '2026-05-01T00:00:00Z'}",
                    },
                },
                "required": ["dataset"],
            },
            handler=_handler_dataset_query(ctx),
            max_calls=15,
            category="reverse_engineer",
        ),
        AgentTool(
            name="dataset_sample_token",
            description=(
                "Sample N evenly-spaced book snapshots for a single token across "
                "the dataset window.  Returns best_bid/ask, spread, top depth — "
                "enough to reason about the market microstructure without dumping "
                "thousands of rows.  Defaults to the dataset scope's tokens."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "token_id": {"type": "string"},
                    "n": {"type": "integer", "default": 10, "minimum": 1, "maximum": 100},
                },
                "required": ["token_id"],
            },
            handler=_handler_sample_token(ctx),
            max_calls=10,
            category="reverse_engineer",
        ),
        AgentTool(
            name="wallet_trades_window",
            description=(
                "Read a slice of the wallet's actual trades, ordered by timestamp.  "
                "Use this to inspect what the trader actually did at specific times."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 25, "minimum": 1, "maximum": 200},
                    "offset": {"type": "integer", "default": 0, "minimum": 0},
                    "market_id": {"type": "string"},
                },
                "required": [],
            },
            handler=_handler_wallet_trades(ctx),
            max_calls=10,
            category="reverse_engineer",
        ),
        AgentTool(
            name="wallet_profile_summary",
            description=(
                "Return the dense numeric profile of the wallet — trade counts, "
                "side/outcome distributions, hour-of-day cadence, top markets, "
                "and a curated sample of trades.  Built once at job start; this "
                "reads the cached profile."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            handler=_handler_wallet_profile(ctx),
            max_calls=3,
            category="reverse_engineer",
        ),
        AgentTool(
            name="submit_strategy_candidate",
            description=(
                "Persist a candidate strategy as a new iteration, run it through "
                "the unified backtester scoped to this job's dataset, score the "
                "fills against the wallet's actual trades, and return the full "
                "score breakdown so you can revise.  This is the one tool that "
                "advances the iteration counter — every call counts against your "
                "max_iterations budget."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "strategy_class": {
                        "type": "string",
                        "description": "Class name to instantiate (e.g. 'WalletMimicStrategy')",
                    },
                    "source_code": {
                        "type": "string",
                        "description": "Full Python source — must define the class and inherit from services.strategies.base.BaseStrategy",
                    },
                    "notes": {
                        "type": "string",
                        "description": "One-line summary of what changed vs the previous iteration",
                    },
                },
                "required": ["strategy_class", "source_code"],
            },
            handler=_handler_submit_candidate(ctx),
            max_calls=50,  # bounded by max_iterations elsewhere
            category="reverse_engineer",
        ),
        AgentTool(
            name="wallet_market_coverage",
            description=(
                "Report which markets the wallet traded that we DO have "
                "data for vs which we don't.  Lets you decide whether to "
                "import additional polybacktest data before submitting "
                "more candidates.  Returns: covered_markets (with trade "
                "counts), uncovered_markets (with the slug + trade count "
                "needed to import them), and a rough coverage_pct."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "limit_uncovered": {
                        "type": "integer",
                        "default": 25,
                        "minimum": 1,
                        "maximum": 200,
                    },
                },
                "required": [],
            },
            handler=_handler_wallet_market_coverage(ctx),
            max_calls=6,
            category="reverse_engineer",
        ),
        AgentTool(
            name="polybacktest_find_markets",
            description=(
                "Look up polybacktest market_ids for a list of polymarket "
                "event slugs (e.g. 'btc-updown-5m-1777943100').  Use this "
                "after wallet_market_coverage to find the polybacktest "
                "market_ids you need to import.  Returns the resolved "
                "market_ids ready to feed into polybacktest_import."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "slugs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 25,
                    },
                    "coin": {
                        "type": "string",
                        "default": "btc",
                        "enum": ["btc", "eth", "sol"],
                    },
                },
                "required": ["slugs"],
            },
            handler=_handler_polybacktest_find_markets(ctx),
            max_calls=6,
            category="reverse_engineer",
        ),
        AgentTool(
            name="polybacktest_import",
            description=(
                "Import polybacktest data for a list of market_ids "
                "(BLOCKING — waits for the import job to complete).  Each "
                "market produces ~5400 microstructure rows (15-level L2 "
                "depth × 2 sides).  After completion the agent's "
                "dataset_scope is automatically refreshed to include the "
                "new datasets, and wallet trades get alias_market_ids "
                "matching the new tokens — so the very next "
                "submit_strategy_candidate call sees the new data.  "
                "Limit to ~10 markets per call to stay under polybacktest's "
                "rate limits."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "market_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 25,
                    },
                    "coin": {
                        "type": "string",
                        "default": "btc",
                        "enum": ["btc", "eth", "sol"],
                    },
                    "start_iso": {
                        "type": "string",
                        "description": "ISO-8601 datetime (defaults to wallet window start)",
                    },
                    "end_iso": {
                        "type": "string",
                        "description": "ISO-8601 datetime (defaults to wallet window end)",
                    },
                },
                "required": ["market_ids"],
            },
            handler=_handler_polybacktest_import(ctx),
            max_calls=8,
            category="reverse_engineer",
        ),
        AgentTool(
            name="finalize_best",
            description=(
                "Stop iterating and finalize the best result so far as the job's "
                "winning strategy.  Call this when (a) the score has plateaued, "
                "(b) you can't see how to improve further, or (c) you've hit the "
                "iteration budget.  Optionally pass an explicit iteration_id to "
                "promote a specific iteration that wasn't the highest-scored."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "iteration_id": {
                        "type": "string",
                        "description": "Optional override — defaults to the best-scored iteration",
                    },
                    "summary": {
                        "type": "string",
                        "description": "One-paragraph executive summary of the inferred strategy",
                    },
                },
                "required": [],
            },
            handler=_handler_finalize(ctx),
            max_calls=2,
            category="reverse_engineer",
        ),
    ]


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _handler_describe(ctx: ReverseEngineerContext):
    async def _h(_args: dict) -> dict:
        scope_tokens = list(ctx.dataset_scope.get("token_ids") or [])
        sample_tokens = scope_tokens[:6]
        # When the scope is a provider dataset (polybacktest), the
        # backtest engine reads its book replay from rows keyed on the
        # synthetic ``polybacktest:<coin>:<market_id>:<side>`` tokens.
        # The strategy MUST emit Opportunity instances against those
        # token_ids — anything else produces zero fills.  Make this
        # crystal clear so the first iteration doesn't waste budget.
        critical_token_hint = (
            "CRITICAL: when emitting Opportunity instances, set "
            "Opportunity.token_id to one of the dataset_scope.token_ids "
            "values listed below.  The backtest engine's BookReplay only "
            "has data for those token_ids; opportunities emitted against "
            "Polymarket conditionIds, slugs, asset CLOB tokens, or any "
            "other identifier will produce zero fills and a score of 0.0."
        ) if scope_tokens else (
            "Dataset scope is auto/wallet-window — the engine will read "
            "any microstructure rows we have for the wallet's trade "
            "window.  Use the wallet trade's conditionId as Opportunity.token_id."
        )
        return {
            "wallet_address": ctx.wallet_address,
            "wallet_trade_count": len(ctx.wallet_trades),
            "dataset_scope": {
                "labels": ctx.dataset_scope.get("labels"),
                "token_ids": scope_tokens,
                "token_id_count": len(scope_tokens),
                "sample_token_ids": sample_tokens,
                "start": _iso(ctx.dataset_scope.get("start")),
                "end": _iso(ctx.dataset_scope.get("end")),
            },
            "objective": (
                "Reverse-engineer this wallet's trading strategy by writing a "
                "Python class that subclasses services.strategies.base.BaseStrategy "
                "and reproduces the wallet's behavior on the supplied dataset.  "
                "Iterate via submit_strategy_candidate; finalize when the score "
                "plateaus.  The composite score weights: 45% trade overlap, "
                "20% side agreement, 20% PnL correlation, 15% frequency match."
            ),
            "critical_token_id_hint": critical_token_hint,
            "rules": [
                "Pure-Python only — no imports beyond stdlib + services.strategies.base / services.strategy_sdk.",
                "No file I/O, no network calls, no os.system.",
                "Strategy must define detect()/detect_async() returning a list of Opportunity instances.",
                "Opportunity.token_id MUST be drawn from dataset_scope.token_ids — see critical_token_id_hint.",
                "Use submit_strategy_candidate sparingly — every call burns iteration budget and tokens.",
            ],
        }

    return _h


def _handler_sdk_reference(_ctx: ReverseEngineerContext):
    async def _h(_args: dict) -> dict:
        from pathlib import Path

        try:
            base_path = Path(__file__).resolve().parents[1] / "strategies" / "base.py"
            base_text = base_path.read_text(encoding="utf-8")
        except Exception as exc:
            base_text = f"<failed to read base.py: {exc}>"
        try:
            sdk_path = Path(__file__).resolve().parents[1] / "strategy_sdk.py"
            sdk_text = sdk_path.read_text(encoding="utf-8")
        except Exception as exc:
            sdk_text = f"<failed to read strategy_sdk.py: {exc}>"
        # Truncate aggressively — the LLM only needs the public contract,
        # not every helper method body.
        return {
            "base_strategy_excerpt": _public_excerpt(base_text, max_chars=6000),
            "strategy_sdk_excerpt": _public_excerpt(sdk_text, max_chars=4000),
            "minimal_example": _MINIMAL_STRATEGY_EXAMPLE,
        }

    return _h


def _handler_list_strategies(_ctx: ReverseEngineerContext):
    async def _h(args: dict) -> dict:
        from sqlalchemy import select

        from models.database import AsyncSessionLocal, Strategy

        source_key = (args.get("source_key") or "").strip() or None

        async with AsyncSessionLocal() as session:
            stmt = select(Strategy)
            if source_key:
                stmt = stmt.where(Strategy.source_key == source_key)
            stmt = stmt.order_by(Strategy.name.asc()).limit(50)
            rows = list((await session.execute(stmt)).scalars().all())
        return {
            "strategies": [
                {
                    "id": str(r.id),
                    "slug": getattr(r, "slug", None),
                    "name": getattr(r, "name", None),
                    "source_key": getattr(r, "source_key", None),
                    "enabled": getattr(r, "enabled", None),
                }
                for r in rows
            ]
        }

    return _h


def _handler_get_strategy_source(_ctx: ReverseEngineerContext):
    async def _h(args: dict) -> dict:
        from sqlalchemy import or_, select

        from models.database import AsyncSessionLocal, Strategy

        sid = str(args.get("strategy_id") or "").strip()
        if not sid:
            return {"error": "strategy_id is required"}
        async with AsyncSessionLocal() as session:
            stmt = select(Strategy).where(or_(Strategy.id == sid, Strategy.slug == sid))
            row = (await session.execute(stmt)).scalar_one_or_none()
        if row is None:
            return {"error": f"strategy '{sid}' not found"}
        source = getattr(row, "source_code", None) or ""
        return {
            "id": str(row.id),
            "slug": getattr(row, "slug", None),
            "name": getattr(row, "name", None),
            "source_code": source[:20_000],  # hard cap to keep prompts sane
            "truncated": len(source) > 20_000,
        }

    return _h


def _handler_dataset_query(ctx: ReverseEngineerContext):
    async def _h(args: dict) -> dict:
        # Dispatches into the dataset router's underlying handler
        # without going through HTTP — keeps the agent tool fast.
        from api.routes_dataset import _DATASETS, _apply_filters, _resolve_dataset, _row_to_dict, _serialize_columns
        from sqlalchemy import select, func

        from models.database import AsyncSessionLocal

        dataset_name = str(args.get("dataset") or "").strip()
        if not dataset_name:
            return {"error": "dataset is required"}
        try:
            spec = _resolve_dataset(dataset_name)
        except Exception as exc:
            return {"error": f"unknown dataset '{dataset_name}': {exc}", "available": list(_DATASETS)}

        limit = int(args.get("limit") or 50)
        offset = int(args.get("offset") or 0)
        filters = args.get("filters") or {}

        # If the agent didn't supply token_id and we're querying
        # microstructure_snapshot, scope it to this job's dataset
        # tokens by default — saves the agent a step and prevents
        # accidentally querying unrelated wallets' data.
        if dataset_name in ("microstructure_snapshot", "book_delta_event") and "token_id" not in filters:
            tokens = ctx.dataset_scope.get("token_ids") or []
            if tokens:
                filters = dict(filters)
                # Filter only takes single eq — pick the first as a hint.
                filters["token_id"] = str(tokens[0])

        # Apply default time bounds when missing (and the filter set
        # supports them) so the agent doesn't pull historic noise.
        for key, value in [
            ("start", _iso(ctx.dataset_scope.get("start"))),
            ("end", _iso(ctx.dataset_scope.get("end"))),
        ]:
            if value and key not in filters:
                filters = dict(filters)
                filters[key] = value

        async with AsyncSessionLocal() as session:
            try:
                count_stmt = _apply_filters(select(func.count(spec.model.id)), spec, filters)
                total = (await session.execute(count_stmt)).scalar_one()

                stmt = select(spec.model)
                stmt = _apply_filters(stmt, spec, filters)
                sort_attr = getattr(spec.model, spec.default_sort, None)
                if sort_attr is not None:
                    stmt = stmt.order_by(
                        sort_attr.desc() if spec.default_sort_dir == "desc" else sort_attr.asc()
                    )
                stmt = stmt.limit(min(500, max(1, int(limit)))).offset(max(0, int(offset)))
                rows = (await session.execute(stmt)).scalars().all()
            except Exception as exc:
                return {"error": str(exc)}

        return {
            "dataset": spec.name,
            "total": int(total or 0),
            "limit": int(limit),
            "offset": int(offset),
            "applied_filters": filters,
            "columns": _serialize_columns(spec),
            "rows": [_row_to_dict(r, spec) for r in rows],
        }

    return _h


def _handler_sample_token(ctx: ReverseEngineerContext):
    async def _h(args: dict) -> dict:
        from sqlalchemy import asc, select

        from models.database import AsyncSessionLocal, MarketMicrostructureSnapshot

        token_id = str(args.get("token_id") or "").strip()
        n = max(1, min(100, int(args.get("n") or 10)))
        if not token_id:
            return {"error": "token_id is required"}
        start = ctx.dataset_scope.get("start")
        end = ctx.dataset_scope.get("end")

        async with AsyncSessionLocal() as session:
            stmt = (
                select(MarketMicrostructureSnapshot)
                .where(MarketMicrostructureSnapshot.token_id == token_id)
                .where(MarketMicrostructureSnapshot.snapshot_type == "book")
            )
            if start is not None:
                stmt = stmt.where(MarketMicrostructureSnapshot.observed_at >= start)
            if end is not None:
                stmt = stmt.where(MarketMicrostructureSnapshot.observed_at <= end)
            stmt = stmt.order_by(asc(MarketMicrostructureSnapshot.observed_at))
            rows = list((await session.execute(stmt)).scalars().all())

        if not rows:
            return {"token_id": token_id, "samples": [], "note": "no rows in window"}

        if len(rows) <= n:
            chosen = rows
        else:
            step = len(rows) / float(n)
            chosen = [rows[int(i * step)] for i in range(n)]

        return {
            "token_id": token_id,
            "row_count": len(rows),
            "samples": [
                {
                    "observed_at": r.observed_at.isoformat() if r.observed_at else None,
                    "best_bid": r.best_bid,
                    "best_ask": r.best_ask,
                    "spread_bps": r.spread_bps,
                    "bids_top3": (r.bids_json or [])[:3],
                    "asks_top3": (r.asks_json or [])[:3],
                }
                for r in chosen
            ],
        }

    return _h


def _handler_wallet_trades(ctx: ReverseEngineerContext):
    async def _h(args: dict) -> dict:
        limit = max(1, min(200, int(args.get("limit") or 25)))
        offset = max(0, int(args.get("offset") or 0))
        market_id = (args.get("market_id") or "").strip() or None

        trades = ctx.wallet_trades
        if market_id:
            trades = [t for t in trades if (t.get("market_id") or "") == market_id]
        slice_ = trades[offset : offset + limit]
        return {
            "wallet_address": ctx.wallet_address,
            "total_matching": len(trades),
            "limit": limit,
            "offset": offset,
            "trades": [
                {
                    "timestamp": t["timestamp"].isoformat() if isinstance(t["timestamp"], datetime) else t["timestamp"],
                    "market_id": t.get("market_id"),
                    "market_title": t.get("market_title"),
                    "side": t.get("side"),
                    "outcome": t.get("outcome"),
                    "price": t.get("price"),
                    "size": t.get("size"),
                    "notional_usd": t.get("notional_usd"),
                }
                for t in slice_
            ],
        }

    return _h


def _handler_wallet_profile(ctx: ReverseEngineerContext):
    """Returns the precomputed profile (passed in via context's job state)."""

    async def _h(_args: dict) -> dict:
        from sqlalchemy import select

        from models.database import (
            AsyncSessionLocal,
            StrategyReverseEngineerJob,
        )

        async with AsyncSessionLocal() as session:
            row = (
                await session.execute(
                    select(StrategyReverseEngineerJob).where(
                        StrategyReverseEngineerJob.id == ctx.job_id
                    )
                )
            ).scalar_one_or_none()
        if row is None:
            return {"error": "job not found"}
        return row.wallet_profile_json or {"note": "profile not yet computed"}

    return _h


def _handler_submit_candidate(ctx: ReverseEngineerContext):
    async def _h(args: dict) -> dict:
        cls = str(args.get("strategy_class") or "").strip()
        src = str(args.get("source_code") or "")
        notes = (args.get("notes") or "").strip() or None
        if not cls:
            return {"error": "strategy_class is required"}
        if len(src) < 50:
            return {"error": "source_code too short — provide a complete class definition"}
        return await ctx.on_iteration_submitted(
            {
                "strategy_class": cls,
                "source_code": src,
                "notes": notes,
            }
        )

    return _h


def _handler_finalize(ctx: ReverseEngineerContext):
    async def _h(args: dict) -> dict:
        iteration_id = (args.get("iteration_id") or "").strip() or None
        return await ctx.on_finalize(iteration_id)

    return _h


def _handler_wallet_market_coverage(ctx: ReverseEngineerContext):
    """Surface which wallet trades have backtest data vs which don't."""

    async def _h(args: dict) -> dict:
        from collections import Counter

        limit = int(args.get("limit_uncovered") or 25)
        # A wallet trade is "covered" iff it carries at least one
        # alias_market_id (set by _attach_provider_aliases when an
        # imported polybacktest market matches the trade's slug).
        by_slug_total: Counter[str] = Counter()
        by_slug_covered: Counter[str] = Counter()
        by_slug_titles: dict[str, str] = {}
        for t in ctx.wallet_trades:
            slug = t.get("event_slug")
            if not slug:
                continue
            by_slug_total[slug] += 1
            title = t.get("market_title")
            if title and slug not in by_slug_titles:
                by_slug_titles[slug] = title
            if t.get("alias_market_ids"):
                by_slug_covered[slug] += 1

        covered_slugs = sorted(
            (s for s in by_slug_total if by_slug_covered.get(s, 0) > 0),
            key=lambda s: -by_slug_total[s],
        )
        uncovered_slugs = sorted(
            (s for s in by_slug_total if by_slug_covered.get(s, 0) == 0),
            key=lambda s: -by_slug_total[s],
        )
        total_trades = sum(by_slug_total.values())
        covered_trades = sum(by_slug_covered.values())

        return {
            "wallet_trade_count": len(ctx.wallet_trades),
            "trades_with_market_data": covered_trades,
            "trades_without_market_data": total_trades - covered_trades,
            "coverage_pct": (
                round(covered_trades / total_trades, 3) if total_trades else 0.0
            ),
            "covered_market_count": len(covered_slugs),
            "uncovered_market_count": len(uncovered_slugs),
            "covered_top": [
                {
                    "slug": s,
                    "trade_count": int(by_slug_total[s]),
                    "title": by_slug_titles.get(s),
                }
                for s in covered_slugs[:limit]
            ],
            "uncovered_top": [
                {
                    "slug": s,
                    "trade_count": int(by_slug_total[s]),
                    "title": by_slug_titles.get(s),
                }
                for s in uncovered_slugs[:limit]
            ],
            "hint": (
                "To import the uncovered_top markets, call "
                "polybacktest_find_markets with their slugs to resolve "
                "polybacktest market_ids, then polybacktest_import to "
                "pull the data.  After import succeeds the dataset_scope "
                "auto-refreshes so the next submit_strategy_candidate "
                "call sees the new data."
            ),
        }

    return _h


def _handler_polybacktest_find_markets(_ctx: ReverseEngineerContext):
    """Resolve polymarket event slugs to polybacktest market_ids."""

    async def _h(args: dict) -> dict:
        slugs = list(args.get("slugs") or [])
        if not slugs:
            return {"error": "slugs required"}
        coin = str(args.get("coin") or "btc").strip().lower()
        wanted = set(str(s).strip() for s in slugs if s)
        if not wanted:
            return {"error": "no usable slugs"}
        try:
            from services.external_data.polybacktest_client import (
                PolybacktestNotConfiguredError,
                build_client_from_settings,
            )

            client = await build_client_from_settings()
        except PolybacktestNotConfiguredError as exc:
            return {"error": str(exc)}
        try:
            resolved: dict[str, str] = {}
            offset = 0
            page_size = 50
            # Polybacktest doesn't have a by-slug bulk lookup — we page
            # through recent markets until every requested slug is found
            # or we hit a hard cap of 200 markets scanned.
            while wanted - set(resolved.keys()) and offset < 200:
                markets, total = await client.list_markets(coin, offset=offset, limit=page_size)
                for m in markets:
                    if m.slug in wanted and m.slug not in resolved:
                        resolved[m.slug] = m.market_id
                if not markets:
                    break
                offset += len(markets)
                if total and offset >= total:
                    break
        finally:
            await client.close()
        unresolved = sorted(wanted - set(resolved.keys()))
        return {
            "resolved": [
                {"slug": slug, "market_id": mid} for slug, mid in resolved.items()
            ],
            "unresolved": unresolved,
            "hint": (
                "Pass the resolved market_ids straight to polybacktest_import."
                if resolved
                else "No matches — your polybacktest plan may not cover these markets, or they're outside the recent window we scanned."
            ),
        }

    return _h


def _handler_polybacktest_import(ctx: ReverseEngineerContext):
    """Run a polybacktest import + refresh the agent's dataset scope."""

    async def _h(args: dict) -> dict:
        market_ids = list(args.get("market_ids") or [])
        if not market_ids:
            return {"error": "market_ids required"}
        coin = str(args.get("coin") or "btc").strip().lower()
        start_iso = args.get("start_iso")
        end_iso = args.get("end_iso")

        # Default to the wallet's trade window if no explicit window given.
        scope_start = ctx.dataset_scope.get("start")
        scope_end = ctx.dataset_scope.get("end")
        if start_iso:
            start_dt = _parse_iso_dt(start_iso)
        else:
            start_dt = scope_start
        if end_iso:
            end_dt = _parse_iso_dt(end_iso)
        else:
            end_dt = scope_end
        if start_dt is None or end_dt is None:
            return {"error": "Could not derive a window — pass start_iso + end_iso explicitly"}

        try:
            from services.external_data.provider_import_service import (
                CreatePolybacktestJobSpec,
                enqueue_polybacktest_import,
                run_job,
            )

            spec = CreatePolybacktestJobSpec(
                coin=coin,
                market_ids=[str(m) for m in market_ids],
                start_ms=int(start_dt.timestamp() * 1000),
                end_ms=int(end_dt.timestamp() * 1000),
            )
            job = await enqueue_polybacktest_import(spec)
        except ValueError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"failed to enqueue import: {exc}"}

        # Run the import inline so the LLM sees the result on this turn
        # rather than having to poll a separate status tool.  Each market
        # is ~3-10s of polybacktest API time so 10 markets fits inside
        # the agent's per-tool latency budget.
        try:
            summary = await run_job(job.id)
        except Exception as exc:
            return {"error": f"import job crashed: {exc}", "job_id": job.id}

        # Refresh the agent's dataset_scope + wallet alias map so the
        # very next submit_strategy_candidate sees the new data.
        scope_update = {}
        if ctx.on_scope_changed is not None:
            try:
                await ctx.on_scope_changed({})
                scope_update = {
                    "scope_refreshed": True,
                    "new_token_count": len(ctx.dataset_scope.get("token_ids") or []),
                }
            except Exception as exc:
                scope_update = {"scope_refreshed": False, "scope_refresh_error": str(exc)}
        return {
            "job_id": job.id,
            "status": summary.get("status", "completed") if isinstance(summary, dict) else "completed",
            "snapshots_inserted": summary.get("snapshots_inserted") if isinstance(summary, dict) else None,
            "api_calls": summary.get("api_calls") if isinstance(summary, dict) else None,
            "rate_limited_count": summary.get("rate_limited_count") if isinstance(summary, dict) else None,
            "per_market": summary.get("per_market") if isinstance(summary, dict) else None,
            **scope_update,
        }

    return _h


def _parse_iso_dt(value: Any):
    """Parse ISO-8601 string → aware datetime, or return the value if already datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return (value if value.tzinfo else value.replace(tzinfo=timezone.utc)).isoformat()
    return str(value)


def _public_excerpt(text: str, *, max_chars: int) -> str:
    """Trim a Python source file to the bits that look like the public API.

    Keeps every ``class`` / ``def`` / ``@dataclass`` declaration plus
    the docstring directly under it.  Drops large method bodies which
    are rarely necessary for the LLM to reason about the contract.
    """
    if len(text) <= max_chars:
        return text
    lines = text.splitlines()
    keep: list[str] = []
    in_keep = False
    indent_keep = 0
    for line in lines:
        stripped = line.lstrip()
        if (
            stripped.startswith("class ")
            or stripped.startswith("def ")
            or stripped.startswith("async def ")
            or stripped.startswith("@dataclass")
            or stripped.startswith("@abstractmethod")
        ):
            keep.append(line)
            in_keep = True
            indent_keep = len(line) - len(stripped)
            continue
        if in_keep:
            current_indent = len(line) - len(line.lstrip())
            if not stripped:
                keep.append(line)
                continue
            if current_indent > indent_keep and (
                stripped.startswith('"""')
                or stripped.startswith("'''")
                or stripped.startswith('#')
            ):
                keep.append(line)
                continue
            if current_indent <= indent_keep:
                keep.append(line)
                in_keep = False
                continue
            # method body — skip
    excerpt = "\n".join(keep)
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars] + "\n... [truncated]"
    return excerpt


_MINIMAL_STRATEGY_EXAMPLE = '''
"""Minimal example a candidate strategy MUST conform to."""
from __future__ import annotations
from typing import Any
from services.strategies.base import BaseStrategy


class ExampleMimicStrategy(BaseStrategy):
    strategy_type = "wallet_mimic_example"
    name = "Example Wallet Mimic"
    description = "Buy YES when best_bid drops below 0.40 and best_ask above 0.42"

    async def detect_async(
        self,
        events: list,
        markets: list,
        prices: dict[str, dict],
    ) -> list[Any]:
        # `prices` is keyed by token_id and contains best_bid / best_ask /
        # observed_at populated by the backtest engine\'s book replay.
        opportunities = []
        for token_id, price in (prices or {}).items():
            best_bid = price.get("best_bid")
            best_ask = price.get("best_ask")
            if best_bid is None or best_ask is None:
                continue
            if best_bid < 0.40 and best_ask > 0.42:
                opportunities.append(self.build_opportunity(
                    token_id=token_id,
                    side="BUY",
                    price=best_ask,
                    size_usd=25.0,
                    reason="bid below 0.40 and ask above 0.42",
                ))
        return opportunities
'''
