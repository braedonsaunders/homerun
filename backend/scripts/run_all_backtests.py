"""Run every enabled strategy against the bus-aware backtest path
over a wide window, write progressive results to JSON.

Each strategy result is flushed to disk immediately so the operator
can monitor progress and so an interrupt loses at most the in-flight
strategy.  Output: ``scripts/_all_backtests_results.json``.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure ``backend/`` is on sys.path so ``import models.database`` works
# regardless of where this script is invoked from.  ``__file__`` here
# is ``backend/scripts/run_all_backtests.py``; parent = scripts/;
# parent.parent = backend/.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

# Wide window: covers polybacktest backfill (May 4-6) + live days
# up to today.  No dataset pinning → universe = full opp_history.
WIN_START = datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc)
WIN_END = datetime(2026, 5, 11, 0, 0, tzinfo=timezone.utc)
OUT_PATH = Path(__file__).resolve().parent / "_all_backtests_results.json"


def _f(v: object, d: float = 0.0) -> float:
    if v is None:
        return d
    if isinstance(v, dict):
        return float(v.get("value", v.get("mean", d)) or d)
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return d


async def run_one(slug: str, src: str) -> dict:
    from services.backtest.unified_runner import run_unified_backtest

    t0 = time.time()
    try:
        out = await run_unified_backtest(
            source_code=src,
            slug=slug,
            start=WIN_START,
            end=WIN_END,
            initial_capital_usd=10_000.0,
            n_trials=1,
        )
    except Exception as e:
        return {
            "slug": slug,
            "fatal": f"{type(e).__name__}: {str(e)[:200]}",
            "wall_s": time.time() - t0,
        }
    exc = out.get("execution") or {}
    return {
        "slug": slug,
        "fatal": None,
        "success": exc.get("success"),
        "discovery_mode": exc.get("discovery_mode"),
        "replay_source": exc.get("replay_source") or "",
        "n_snapshots": exc.get("n_snapshots") or 0,
        "n_intents": exc.get("n_intents") or 0,
        "trade_count": exc.get("trade_count") or 0,
        "total_fills": exc.get("total_fills") or 0,
        "rejected": exc.get("rejected_orders") or 0,
        "total_return_pct": _f(exc.get("total_return_pct")),
        "final_equity_usd": _f(exc.get("final_equity_usd"), 10_000.0),
        "sharpe": _f(exc.get("sharpe")),
        "max_drawdown_pct": _f(exc.get("max_drawdown_pct")),
        "fees_paid_usd": _f(exc.get("fees_paid_usd")),
        "wall_s": time.time() - t0,
        "warnings": (exc.get("validation_warnings") or [])[:6],
        "runtime_error": (exc.get("runtime_error") or "")[:200],
    }


async def main() -> None:
    from models.database import AsyncSessionLocal
    from sqlalchemy import text

    async with AsyncSessionLocal() as s:
        rows = (
            await s.execute(
                text(
                    "SELECT slug, source_code FROM strategies WHERE enabled=true ORDER BY slug"
                )
            )
        ).all()

    print(
        f"Window: {WIN_START} → {WIN_END}  ({(WIN_END - WIN_START).days} days)  "
        f"strategies: {len(rows)}",
        flush=True,
    )
    print("=" * 100, flush=True)

    results: list[dict] = []
    OUT_PATH.write_text("[]", encoding="utf-8")

    for r in rows:
        res = await run_one(r.slug, r.source_code)
        results.append(res)
        # Tag
        if res.get("fatal"):
            tag = "FATAL"
        elif not res.get("success"):
            tag = "FAIL"
        elif res["total_fills"] > 0:
            tag = "FILL"
        elif res["n_intents"] > 1:
            tag = "INT"
        else:
            tag = "OK"
        if res.get("fatal"):
            print(
                f'  [{tag:5s}] {r.slug:34s}  {res["fatal"][:80]}  ({res["wall_s"]:>5.1f}s)',
                flush=True,
            )
        else:
            ret = res["total_return_pct"]
            mark = "+" if ret > 0 else ("-" if ret < 0 else " ")
            print(
                f'  [{tag:5s}] {r.slug:34s}  ints={res["n_intents"]:>3} '
                f'fills={res["total_fills"]:>3}  {mark}ret={ret:>7.3f}%  '
                f'sharpe={res["sharpe"]:>6.2f}  mdd={res["max_drawdown_pct"]:>6.2f}%  '
                f'({res["wall_s"]:>5.1f}s)',
                flush=True,
            )
        # Flush results to disk after each strategy.  Operator can read
        # this file from another shell to see live progress.
        OUT_PATH.write_text(
            json.dumps(
                {
                    "window_start": WIN_START.isoformat(),
                    "window_end": WIN_END.isoformat(),
                    "completed": len(results),
                    "remaining": len(rows) - len(results),
                    "results": results,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    # Final summary
    print(flush=True)
    print("=" * 132, flush=True)
    hdr = (
        f'{"strategy":34s}  {"discovery":22s}  {"snaps":>7s} {"int":>4s} {"fill":>4s}  '
        f'{"return%":>8s} {"sharpe":>7s} {"mdd%":>7s} {"final$":>10s}  status'
    )
    print(hdr, flush=True)
    print("=" * 132, flush=True)
    n_ok = n_fill = n_traded = n_fail = 0
    for r in results:
        if r.get("fatal") or not r.get("success"):
            print(
                f'  {r["slug"]:34s}  {(r.get("fatal") or r.get("runtime_error",""))[:90]}',
                flush=True,
            )
            n_fail += 1
            continue
        n_ok += 1
        if r["total_fills"] > 0:
            n_fill += 1
        if r["n_intents"] > 1:
            n_traded += 1
        disc = (r["discovery_mode"] or "")[:22]
        status = (
            "fill"
            if r["total_fills"] > 0
            else ("intent" if r["n_intents"] > 1 else "no-trade")
        )
        print(
            f'  {r["slug"]:34s}  {disc:22s}  {r["n_snapshots"]:>7d} '
            f'{r["n_intents"]:>4d} {r["total_fills"]:>4d}  '
            f'{r["total_return_pct"]:>8.3f} {r["sharpe"]:>7.3f} '
            f'{r["max_drawdown_pct"]:>7.2f} {r["final_equity_usd"]:>10.2f}  {status}',
            flush=True,
        )
    print("=" * 132, flush=True)
    print(
        f"  TOTAL: {len(results)}  loaded_ok: {n_ok}  intents>1: {n_traded}  "
        f"fills>0: {n_fill}  failed: {n_fail}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
