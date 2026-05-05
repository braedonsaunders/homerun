"""LLM agent loop for strategy reverse-engineering.

Wraps the existing :class:`services.ai.agent.Agent` ReAct loop with the
reverse-engineer-specific tool registry, system prompt, and the
"submit candidate → backtest → score → record iteration" callback that
the tools call back into.

Public surface:
  * ``run_reverse_engineer_agent(job_id)`` — drives the loop end-to-end.

The function is async and side-effecting: it persists iteration rows,
updates the job's progress, and writes the final winning strategy back
to the job before returning.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select, update

from models.database import (
    AsyncSessionLocal,
    StrategyReverseEngineerIteration,
    StrategyReverseEngineerJob,
)
from services.ai.agent import Agent, AgentEventType
from services.strategy_reverse_engineer.scoring import (
    ScoreBreakdown,
    score_backtest_against_wallet,
)
from services.strategy_reverse_engineer.tools import (
    ReverseEngineerContext,
    build_tools,
)
from utils.utcnow import utcnow

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """\
You are an expert quantitative-strategy reverse-engineering agent.

Your job is to write a Python ``BaseStrategy`` subclass that reproduces a
Polymarket wallet's trading behavior on a supplied historical dataset.
You operate exactly like a human quant researcher with a Jupyter notebook
and full read access to the market data — you may inspect the data, read
existing strategies for inspiration, write candidate code, run a backtest,
inspect the score breakdown, and revise.

Workflow you SHOULD follow:

  1. Call ``describe_objective`` to learn the wallet, dataset, and rules.
  2. Call ``wallet_profile_summary`` to read the dense numeric profile.
  3. Call ``wallet_market_coverage`` to see which markets the wallet
     traded that we DO have backtest data for vs which we don't.
  4. If coverage is low (< 50%), AGENTICALLY pull more data:
       * ``polybacktest_find_markets`` resolves polymarket event-slugs
         (e.g. 'btc-updown-5m-1777943100') to polybacktest market_ids.
       * ``polybacktest_import`` then pulls those markets' full L2 books
         (15 levels per side) into our local microstructure store and
         AUTO-REFRESHES your dataset_scope + wallet alias map.  No need
         to ask the operator to import — it's just a tool call.
  5. Use ``wallet_trades_window`` and ``dataset_sample_token`` to
     spot-check specific trades against the market state at that moment.
  6. Call ``strategy_sdk_reference`` to learn the BaseStrategy contract.
  7. Optionally call ``list_existing_strategies`` + ``get_strategy_source``
     to study an analogous example.
  8. Synthesize a candidate strategy and call ``submit_strategy_candidate``.
  9. Read the score breakdown.  Diagnose what's wrong:
       * trade_overlap_pct low?     → your detect logic is missing trades or
                                      firing on the wrong markets.
       * side_agreement_pct low?    → your side / outcome inference is off.
       * pnl_correlation low?       → you're trading at the wrong times.
       * frequency_match low?       → you're over- or under-trading.
       * timing_mae_seconds high?   → entry triggers fire too early or late.
       * matched=0 with non-zero backtest_trade_count?
                                    → wrong token_ids — see
                                      describe_objective.critical_token_id_hint.
 10. Revise and submit again.  Repeat until composite score plateaus or
     you reach the iteration budget.
 11. Call ``finalize_best`` to lock in the winning strategy.

Hard rules for the strategy code:
  * Subclass ``services.strategies.base.BaseStrategy``.
  * Define ``strategy_type``, ``name``, and either ``detect`` or
    ``detect_async``.  Prefer ``detect_async`` when you need any I/O.
  * Pure Python — no ``import os``, no file I/O, no network calls.
  * Use the helpers in ``services.strategy_sdk`` for book / depth /
    rolling window access.  Do not invent ``import`` statements that
    don't exist in the SDK.
  * Return ``Opportunity`` instances via ``self.build_opportunity(...)``.

Token economy:
  * Tool results are truncated at ~8 KB.  When you need more, narrow your
    query (smaller limit, narrower time window) instead of repeating it.
  * Every ``submit_strategy_candidate`` call is expensive — think before
    you submit.

When in doubt, prefer fewer rules + clear thresholds over many rules.
A strategy that captures 60% of behavior cleanly is better than one
that captures 80% via opaque heuristics.
"""


async def run_reverse_engineer_agent(job_id: str) -> dict[str, Any]:
    """Execute the reverse-engineer loop for one job.

    Reads the job row, builds the per-job context, kicks off the agent,
    and returns a summary dict.  Side effects: iteration rows persisted,
    job progress + best result fields updated continuously.
    """
    job, wallet_trades, dataset_scope = await _bootstrap(job_id)

    state: dict[str, Any] = {
        "iteration_index": 0,  # bumped by submit_strategy_candidate
        "best_iteration_id": None,
        "best_score": None,
        "best_strategy_code": None,
        "best_strategy_class": None,
        "best_backtest_run_id": None,
        "max_iterations": int(job.max_iterations or 10),
        "target_score": float(job.target_score or 0.7),
        "max_cost_usd": float(job.max_cost_usd) if job.max_cost_usd is not None else None,
        "model": job.llm_model,
        "stop_reason": None,
    }

    async def on_iteration_submitted(payload: dict[str, Any]) -> dict[str, Any]:
        return await _process_iteration(
            job_id=job_id,
            wallet_trades=ctx.wallet_trades,
            dataset_scope=ctx.dataset_scope,
            payload=payload,
            state=state,
        )

    async def on_finalize(iteration_id: Optional[str]) -> dict[str, Any]:
        return await _finalize_job(job_id=job_id, override_iteration_id=iteration_id, state=state)

    async def on_scope_changed(_args: dict[str, Any]) -> None:
        """Re-resolve dataset scope + wallet alias map after a polybacktest_import.

        Invoked by the inner ``polybacktest_import`` tool once a new
        dataset is in the catalog.  Mutates ctx in place so the next
        backtest sees the new token_ids and the next scoring run can
        match fills against the wallet trades that just acquired
        provider aliases.
        """
        new_scope = await _resolve_dataset_scope(
            job, ctx.dataset_scope.get("start"), ctx.dataset_scope.get("end")
        )
        # Rebuild the wallet trades' alias maps against the fresh scope.
        # We start from the trade dicts WITHOUT existing aliases so a
        # second import doesn't accumulate stale ones.
        stripped = [
            {k: v for k, v in t.items() if k != "alias_market_ids"}
            for t in ctx.wallet_trades
        ]
        rebuilt = await _attach_provider_aliases(
            wallet_trades=stripped, scope=new_scope
        )
        ctx.dataset_scope = new_scope
        ctx.wallet_trades = rebuilt

    ctx = ReverseEngineerContext(
        job_id=job_id,
        wallet_address=job.wallet_address,
        wallet_trades=wallet_trades,
        dataset_scope=dataset_scope,
        on_iteration_submitted=on_iteration_submitted,
        on_finalize=on_finalize,
        on_scope_changed=on_scope_changed,
    )
    tools = build_tools(ctx)

    # Resolve model: explicit job override > AppSettings reverse_engineer_default_model
    # > AppSettings ai_default_model > LLMManager default.  No hardcoded
    # value baked into this module per the no-hardcoded-defaults policy.
    model = state["model"] or await _resolve_default_model()

    agent = Agent(
        system_prompt=_SYSTEM_PROMPT,
        tools=tools,
        model=model,
        max_iterations=max(state["max_iterations"] * 4, 30),  # outer ReAct loop budget
        session_type="strategy_reverse_engineer",
        temperature=0.2,
    )

    user_query = (
        f"Reverse-engineer the trading strategy of wallet {job.wallet_address}.\n"
        f"You have up to {state['max_iterations']} candidate submissions.  Target a "
        f"composite score >= {state['target_score']:.2f}.  Begin by orienting yourself "
        f"with describe_objective + wallet_profile_summary."
    )

    await _set_status(job_id, status="running", activity="Agent loop started")
    final_result: Optional[dict[str, Any]] = None
    try:
        async for event in agent.run(query=user_query):
            if event.type == AgentEventType.DONE:
                final_result = event.data.get("result", {})
            elif event.type == AgentEventType.ERROR:
                err = event.data.get("error", "unknown error")
                await _mark_failed(job_id, err)
                return {"job_id": job_id, "status": "failed", "error": err}
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("reverse-engineer agent crashed for job %s", job_id)
        await _mark_failed(job_id, str(exc))
        return {"job_id": job_id, "status": "failed", "error": str(exc)}

    # If the agent never explicitly called finalize_best, do it now —
    # promotes whatever the highest-scored iteration was.
    if state["best_iteration_id"] is not None:
        await _finalize_job(job_id=job_id, override_iteration_id=None, state=state)

    return {
        "job_id": job_id,
        "status": "completed",
        "best_score": state["best_score"],
        "best_iteration_id": state["best_iteration_id"],
        "agent_answer": (final_result or {}).get("answer"),
    }


# ---------------------------------------------------------------------------
# Iteration pipeline
# ---------------------------------------------------------------------------


async def _process_iteration(
    *,
    job_id: str,
    wallet_trades: list[dict[str, Any]],
    dataset_scope: dict[str, Any],
    payload: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    """One iteration: persist row, run backtest, score, update best, return result."""
    state["iteration_index"] += 1
    iter_index = state["iteration_index"]
    iter_id = f"reit-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"

    src = str(payload.get("source_code") or "")
    cls = str(payload.get("strategy_class") or "")
    notes = payload.get("notes")

    started_at = utcnow()
    async with AsyncSessionLocal() as session:
        row = StrategyReverseEngineerIteration(
            id=iter_id,
            job_id=job_id,
            iteration=iter_index,
            status="running",
            strategy_code=src,
            strategy_class=cls,
            notes=notes,
        )
        session.add(row)
        await session.commit()

    await _update_job(
        job_id,
        progress=min(1.0, iter_index / max(1, state["max_iterations"])),
        current_iteration=iter_index,
        activity=f"Backtesting iteration {iter_index}: {cls}",
    )

    backtest_run_id: Optional[str] = None
    score_breakdown: Optional[ScoreBreakdown] = None
    error: Optional[str] = None
    try:
        from services.backtest.unified_runner import run_unified_backtest

        result = await run_unified_backtest(
            source_code=src,
            slug=f"_re_{job_id}_{iter_index}",
            config=None,
            token_ids=list(dataset_scope.get("token_ids") or []) or None,
            start=dataset_scope.get("start"),
            end=dataset_scope.get("end"),
            initial_capital_usd=1000.0,
        )
        backtest_run_id = (
            (result or {}).get("run_id")
            or (result or {}).get("id")
        )
        score_breakdown = score_backtest_against_wallet(
            backtest_result=result or {},
            wallet_trades=wallet_trades,
        )
    except Exception as exc:
        logger.exception("backtest failed for iteration %s of job %s", iter_id, job_id)
        error = str(exc)

    duration_ms = (utcnow() - started_at).total_seconds() * 1000.0

    async with AsyncSessionLocal() as session:
        row = (
            await session.execute(
                select(StrategyReverseEngineerIteration).where(
                    StrategyReverseEngineerIteration.id == iter_id
                )
            )
        ).scalar_one()
        row.completed_at = utcnow()
        row.duration_ms = duration_ms
        if error:
            row.status = "failed"
            row.error = error
        else:
            row.status = "completed"
            row.backtest_run_id = backtest_run_id
            if score_breakdown is not None:
                row.score = float(score_breakdown.composite)
                row.score_breakdown_json = score_breakdown.to_dict()
                row.divergence_summary = _summarize_divergence(score_breakdown)
        await session.commit()

    response: dict[str, Any] = {
        "iteration": iter_index,
        "iteration_id": iter_id,
        "duration_ms": int(duration_ms),
        "backtest_run_id": backtest_run_id,
        "status": "failed" if error else "completed",
    }
    if error:
        response["error"] = error
        return response

    breakdown_dict = score_breakdown.to_dict() if score_breakdown else {}
    response["score_breakdown"] = breakdown_dict
    response["composite_score"] = breakdown_dict.get("composite")
    response["divergence_summary"] = _summarize_divergence(score_breakdown) if score_breakdown else None

    if score_breakdown is not None:
        if state["best_score"] is None or score_breakdown.composite > state["best_score"]:
            state["best_iteration_id"] = iter_id
            state["best_score"] = float(score_breakdown.composite)
            state["best_strategy_code"] = src
            state["best_strategy_class"] = cls
            state["best_backtest_run_id"] = backtest_run_id
            await _update_job(
                job_id,
                best_iteration_id=iter_id,
                best_score=float(score_breakdown.composite),
                best_strategy_code=src,
                best_strategy_class=cls,
                best_backtest_run_id=backtest_run_id,
                activity=f"New best score: {score_breakdown.composite:.3f}",
            )
        if score_breakdown.composite >= state["target_score"]:
            response["target_reached"] = True
        if iter_index >= state["max_iterations"]:
            response["budget_exhausted"] = True

    return response


def _summarize_divergence(breakdown: ScoreBreakdown) -> str:
    parts = [
        f"composite={breakdown.composite:.3f}",
        f"overlap={breakdown.trade_overlap_pct:.3f}",
        f"side_agree={breakdown.side_agreement_pct:.3f}",
        f"pnl_corr={breakdown.pnl_correlation:.3f}",
        f"freq_match={breakdown.frequency_match:.3f}",
    ]
    if breakdown.timing_mae_seconds is not None:
        parts.append(f"timing_mae_s={breakdown.timing_mae_seconds:.0f}")
    parts.append(
        f"matched={breakdown.matched_trades}/{breakdown.backtest_trade_count} bt vs {breakdown.wallet_trade_count} wallet"
    )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Bootstrap + finalization
# ---------------------------------------------------------------------------


async def _bootstrap(job_id: str) -> tuple[StrategyReverseEngineerJob, list[dict[str, Any]], dict[str, Any]]:
    """Load the job + wallet trades + resolved dataset scope.

    Wallet profile is computed (and persisted on the job) here if it
    isn't already.  Dataset scope is resolved based on
    ``data_source_kind`` — defaulting to "auto" which picks the best
    available source given the wallet's market footprint.

    Also enriches each normalized wallet trade with a list of
    ``alias_market_ids`` so scoring can match backtest fills emitted
    against synthetic provider tokens (e.g. ``polybacktest:btc:2151787:up``)
    back to the corresponding wallet trade keyed on Polymarket
    ``conditionId`` + ``event_slug``.  This is the bridge that makes a
    polybacktest-scoped backtest's fills comparable to the wallet's
    real Polymarket trades.
    """
    from services.strategy_reverse_engineer.wallet_profile import (
        fetch_and_profile_wallet,
        profile_trades,
    )

    async with AsyncSessionLocal() as session:
        job = (
            await session.execute(
                select(StrategyReverseEngineerJob).where(
                    StrategyReverseEngineerJob.id == job_id
                )
            )
        ).scalar_one_or_none()
        if job is None:
            raise ValueError(f"reverse-engineer job '{job_id}' not found")

    await _update_job(job_id, status="profiling", activity="Fetching wallet trades")

    await fetch_and_profile_wallet(
        job.wallet_address,
        max_trades=int(job.max_wallet_trades or 50_000),
    )

    from services.polymarket import polymarket_client

    raw = await polymarket_client.get_wallet_trades_paginated(
        job.wallet_address,
        max_trades=int(job.max_wallet_trades or 50_000),
        page_size=500,
    )
    rebuilt = profile_trades(address=job.wallet_address, raw_trades=raw or [])
    full_normalized_trades = _coerce_normalized_trades(raw or [])

    window_start = _parse_iso(rebuilt.get("window_start"))
    window_end = _parse_iso(rebuilt.get("window_end"))

    await _update_job(
        job_id,
        wallet_profile_json=rebuilt,
        wallet_trade_count=int(rebuilt.get("fetched_count", 0)),
        wallet_window_start=window_start,
        wallet_window_end=window_end,
        activity="Resolving dataset scope",
    )

    dataset_scope = await _resolve_dataset_scope(job, window_start, window_end)

    # Enrich wallet trades with provider-token aliases so scoring can
    # match a backtest fill keyed on a synthetic ``polybacktest:*``
    # token back to the wallet trade keyed on Polymarket conditionId.
    # No-op for non-provider scopes (auto / recording_session).
    full_normalized_trades = await _attach_provider_aliases(
        wallet_trades=full_normalized_trades,
        scope=dataset_scope,
    )

    return job, full_normalized_trades, dataset_scope


async def _attach_provider_aliases(
    *,
    wallet_trades: list[dict[str, Any]],
    scope: dict[str, Any],
) -> list[dict[str, Any]]:
    """Add a ``alias_market_ids`` set to each wallet trade.

    Looks up every ProviderDataset whose ``token_ids`` overlap the
    scope, builds a slug → token_id_prefix map, then for each wallet
    trade tags the polybacktest token_ids that correspond to it (one
    per outcome side).  Scoring uses this set as a fallback alias
    when matching backtest fills to wallet trades.
    """
    if not wallet_trades:
        return wallet_trades
    scope_token_ids = scope.get("token_ids") or []
    if not scope_token_ids:
        return wallet_trades

    from models.database import ProviderDataset

    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(ProviderDataset).where(
                    ProviderDataset.provider == "polybacktest"
                )
            )
        ).scalars().all()

    # Build slug → list[token_id] map (one slug → up + down tokens).
    slug_to_tokens: dict[str, list[str]] = {}
    for row in rows:
        slug = (row.external_slug or "").strip()
        if not slug:
            continue
        slug_to_tokens.setdefault(slug, []).extend(list(row.token_ids_json or []))

    if not slug_to_tokens:
        return wallet_trades

    enriched: list[dict[str, Any]] = []
    for trade in wallet_trades:
        slug = (trade.get("event_slug") or "").strip()
        outcome = (trade.get("outcome") or "").strip().lower()
        aliases: set[str] = set()
        if slug and slug in slug_to_tokens:
            for tok in slug_to_tokens[slug]:
                # Tag the side that matches the trade outcome (up/down)
                # — the other side is irrelevant for matching this trade.
                if outcome and tok.endswith(f":{outcome}"):
                    aliases.add(tok)
            # If outcome wasn't specified, tag both sides.
            if not outcome:
                aliases.update(slug_to_tokens[slug])
        enriched.append({**trade, "alias_market_ids": list(aliases)})
    return enriched


def _coerce_normalized_trades(raw_trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert raw Polymarket trade dicts → the shape scoring expects.

    Reuses :func:`wallet_profile._normalize_trade` semantics but keeps
    every trade (no curation / sampling).
    """
    from services.strategy_reverse_engineer.wallet_profile import _normalize_trade

    out: list[dict[str, Any]] = []
    for raw in raw_trades or []:
        norm = _normalize_trade(raw)
        if norm is not None:
            out.append(norm)
    out.sort(key=lambda t: t["timestamp"])
    return out


async def _resolve_dataset_scope(
    job: StrategyReverseEngineerJob,
    window_start: Optional[datetime],
    window_end: Optional[datetime],
) -> dict[str, Any]:
    """Pick the dataset the agent will operate against.

    Priority:
      1. Explicit ``provider_dataset_ids`` on the job → use those.
      2. Explicit ``recording_session_ids`` on the job → use the union.
      3. ``data_source_kind == "auto"`` → fall back to the wallet's own
         time window with no token scope (engine reads any data we have
         in microstructure for that period).  This is the "best effort
         with coverage gap" mode the user requested for non-crypto
         wallets.
    """
    if job.provider_dataset_ids_json:
        from services.external_data.provider_import_service import resolve_dataset_scope

        scope = await resolve_dataset_scope(list(job.provider_dataset_ids_json))
        if scope:
            return {
                "kind": "provider_dataset",
                "labels": scope["labels"],
                "token_ids": scope["token_ids"],
                "start": scope["start"],
                "end": scope["end"],
            }

    if job.recording_session_ids_json:
        from services.recording_session_service import session_backtest_scope

        token_ids: list[str] = []
        starts: list[datetime] = []
        ends: list[datetime] = []
        labels: list[str] = []
        for sid in job.recording_session_ids_json:
            scope = await session_backtest_scope(str(sid))
            if not scope:
                continue
            for tid in scope.get("token_ids") or []:
                if tid not in token_ids:
                    token_ids.append(str(tid))
            if scope.get("start"):
                starts.append(scope["start"])
            if scope.get("end"):
                ends.append(scope["end"])
            labels.append(scope.get("session_name") or sid)
        if token_ids:
            return {
                "kind": "recording_session",
                "labels": labels,
                "token_ids": token_ids,
                "start": min(starts) if starts else None,
                "end": max(ends) if ends else None,
            }

    # Auto fallback — we don't know which markets to scope to, so give
    # the agent the wallet's own trade window and let the dataset_query
    # tool handle drilling down.
    return {
        "kind": "auto",
        "labels": ["auto (wallet trade window)"],
        "token_ids": [],
        "start": window_start,
        "end": window_end,
    }


async def _finalize_job(
    *,
    job_id: str,
    override_iteration_id: Optional[str],
    state: dict[str, Any],
) -> dict[str, Any]:
    chosen_iter_id = override_iteration_id or state.get("best_iteration_id")
    if not chosen_iter_id:
        await _mark_done(job_id, "completed", activity="No successful iteration; nothing to promote.")
        return {
            "job_id": job_id,
            "status": "completed",
            "promoted_iteration_id": None,
            "message": "No successful iteration to finalize",
        }

    async with AsyncSessionLocal() as session:
        chosen = (
            await session.execute(
                select(StrategyReverseEngineerIteration).where(
                    StrategyReverseEngineerIteration.id == chosen_iter_id
                )
            )
        ).scalar_one_or_none()
        if chosen is None:
            await _mark_done(job_id, "failed", activity=f"override iteration {chosen_iter_id} not found")
            return {"job_id": job_id, "status": "failed", "error": "override iteration not found"}

        await session.execute(
            update(StrategyReverseEngineerJob)
            .where(StrategyReverseEngineerJob.id == job_id)
            .values(
                best_iteration_id=chosen.id,
                best_score=chosen.score,
                best_strategy_code=chosen.strategy_code,
                best_strategy_class=chosen.strategy_class,
                best_backtest_run_id=chosen.backtest_run_id,
            )
        )
        await session.commit()

    await _mark_done(job_id, "completed", activity=f"Finalized iteration {chosen.iteration}")
    return {
        "job_id": job_id,
        "status": "completed",
        "promoted_iteration_id": chosen_iter_id,
        "best_score": chosen.score,
    }


# ---------------------------------------------------------------------------
# Job-row helpers
# ---------------------------------------------------------------------------


async def _resolve_default_model() -> Optional[str]:
    """Return the operator-configured default reverse-engineer model.

    Falls through to:
      AppSettings.llm_model_assignments['strategy_reverse_engineer'] →
      AppSettings.ai_default_model →
      LLMManager defaults.

    Per the no-hardcoded-defaults policy we never bake a model name
    into this module.  The per-purpose model assignments live in the
    same JSON column as every other LLM purpose (chat, news_analysis,
    etc.) so the AI tab → Models view manages them uniformly.
    """
    from models.database import AppSettings

    async with AsyncSessionLocal() as session:
        row = (await session.execute(select(AppSettings))).scalar_one_or_none()
    if row is None:
        return None
    assignments = getattr(row, "llm_model_assignments", None) or {}
    candidate = None
    if isinstance(assignments, dict):
        candidate = assignments.get("strategy_reverse_engineer")
    candidate = candidate or getattr(row, "ai_default_model", None)
    return candidate or None


async def _set_status(job_id: str, *, status: str, activity: str) -> None:
    await _update_job(job_id, status=status, activity=activity)


async def _update_job(job_id: str, **fields: Any) -> None:
    async with AsyncSessionLocal() as session:
        row = (
            await session.execute(
                select(StrategyReverseEngineerJob).where(
                    StrategyReverseEngineerJob.id == job_id
                )
            )
        ).scalar_one_or_none()
        if row is None:
            return
        for key, value in fields.items():
            setattr(row, key, value)
        await session.commit()


async def _mark_failed(job_id: str, error: str) -> None:
    await _update_job(
        job_id,
        status="failed",
        error=error,
        activity=error[:200],
        finished_at=utcnow(),
    )


async def _mark_done(job_id: str, status: str, *, activity: str) -> None:
    await _update_job(
        job_id,
        status=status,
        progress=1.0,
        activity=activity,
        finished_at=utcnow(),
    )


def _parse_iso(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
