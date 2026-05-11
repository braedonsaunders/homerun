"""Second-pass: run each previously-zero-fill strategy against the
window where its natural data actually exists.

Strategies fall into buckets:

  * Crypto family (btc_eth_*, crypto_*) — replay over the last ~40min
    where the live crypto.update.dispatch tap captured 2k events.
  * Wallet/trader (traders_*) — replay over recent wallet_monitor_events.
  * Book-driven with no recent opps (basic, market_making,
    prob_surface_arb, etc.) — try a much wider 30d window so a
    diversity of market types surface.
  * Cross-platform / sports / weather — acknowledge: needs data we
    don't have in our archive.

Writes results to ``scripts/_targeted_backtests_results.json``.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]


# ── Per-strategy window selection ────────────────────────────────────


# Late-window centred on crypto.update tap range observed earlier.
NOW = datetime.now(timezone.utc)
CRYPTO_TAP_START = NOW - timedelta(hours=1)
CRYPTO_TAP_END = NOW

WIDE_30D_START = NOW - timedelta(days=30)
WIDE_30D_END = NOW

RECENT_24H_START = NOW - timedelta(days=2)
RECENT_24H_END = NOW

# Strategies to re-run with their tuned window.
# Format: (slug, (start, end), why)
TARGETS: list[tuple[str, tuple[datetime, datetime], str]] = [
    # Crypto family — uses crypto.update bus tap (forward-only).
    ("btc_eth_convergence",        (CRYPTO_TAP_START, CRYPTO_TAP_END),  "needs crypto.update events; tap captured 2k events in last 40min"),
    ("btc_eth_directional_edge",   (CRYPTO_TAP_START, CRYPTO_TAP_END),  "needs crypto.update events; tap-recent window"),
    ("btc_eth_maker_quote",        (CRYPTO_TAP_START, CRYPTO_TAP_END),  "needs crypto.update events; tap-recent window"),

    # Trader-driven — uses wallet_monitor_events (recent 24h)
    ("traders_confluence",         (RECENT_24H_START, RECENT_24H_END),  "needs trader_activity; recent wallet events available"),
    ("traders_copy_trade",         (RECENT_24H_START, RECENT_24H_END),  "needs wallet_trade; 2.9M events available"),

    # Book-driven with no opps in 7d window — try 30d
    ("basic",                      (WIDE_30D_START, WIDE_30D_END),      "broaden window for arb opportunities"),
    ("market_making",              (WIDE_30D_START, WIDE_30D_END),      "broaden window for thin-book opportunities"),
    ("prob_surface_arb",           (WIDE_30D_START, WIDE_30D_END),      "broaden window"),
    ("negrisk",                    (WIDE_30D_START, WIDE_30D_END),      "broaden window — needs multi-outcome markets"),
    ("cross_platform",             (WIDE_30D_START, WIDE_30D_END),      "broaden window — needs polymarket+kalshi pairs"),
    ("holding_reward_yield",       (WIDE_30D_START, WIDE_30D_END),      "broaden window — needs reward markets"),
    ("sports_overreaction_fader",  (WIDE_30D_START, WIDE_30D_END),      "broaden window — needs sports markets"),

    # Forward-only architecture gaps — no historical data; honest test
    # is "does the path actually fire when data exists" so we point
    # them at the tap window or the most recent data we have.
    ("news_edge",                  (CRYPTO_TAP_START, CRYPTO_TAP_END),  "needs news.update events (no historical data); confirms path"),
    ("vpin_toxicity",              (CRYPTO_TAP_START, CRYPTO_TAP_END),  "needs polymarket.trade.execution (no historical data); confirms path"),

    # manual_wallet_position — operator-driven only, not auto-detect.
    # No second pass needed.
]

# Opt-in JSON dump via ``--json <path>``.
_json_path: Path | None = None
if "--json" in sys.argv:
    _i = sys.argv.index("--json")
    if _i + 1 < len(sys.argv):
        _json_path = Path(sys.argv[_i + 1]).expanduser().resolve()


def _f(v, d=0.0):
    if v is None:
        return d
    if isinstance(v, dict):
        return float(v.get("value", v.get("mean", d)) or d)
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


async def run_one(slug, src, start, end):
    from services.backtest.unified_runner import run_unified_backtest
    t0 = time.time()
    try:
        out = await run_unified_backtest(
            source_code=src, slug=slug,
            start=start, end=end,
            initial_capital_usd=10_000.0, n_trials=1,
        )
    except Exception as e:
        return {"slug": slug, "fatal": f"{type(e).__name__}: {str(e)[:200]}", "wall_s": time.time()-t0}
    exc = out.get("execution") or {}
    return {
        "slug": slug,
        "fatal": None,
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
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
        "wall_s": time.time()-t0,
        "warnings": (exc.get("validation_warnings") or [])[:5],
    }


async def main():
    from models.database import AsyncSessionLocal
    from sqlalchemy import text

    async with AsyncSessionLocal() as s:
        rows = (await s.execute(text(
            "SELECT slug, source_code FROM strategies WHERE enabled=true ORDER BY slug"
        ))).all()
    src_by_slug = {r.slug: r.source_code for r in rows}

    print(f"Targeted second pass — {len(TARGETS)} strategies\n", flush=True)
    print("=" * 110, flush=True)
    results = []
    if _json_path is not None:
        _json_path.write_text("[]", encoding="utf-8")
    for slug, (start, end), why in TARGETS:
        src = src_by_slug.get(slug)
        if src is None:
            print(f"  [SKIP] {slug:34s}  not in DB", flush=True)
            continue
        print(f"  → {slug:34s}  window={start.isoformat()[:19]} → {end.isoformat()[:19]}", flush=True)
        print(f"     reason: {why}", flush=True)
        res = await run_one(slug, src, start, end)
        results.append(res)
        if _json_path is not None:
            _json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        if res.get("fatal"):
            print(f'     FATAL: {res["fatal"][:120]}  ({res["wall_s"]:.1f}s)\n', flush=True)
            continue
        ret = res["total_return_pct"]
        mark = "+" if ret > 0 else ("-" if ret < 0 else " ")
        tag = "FILL" if res["total_fills"] > 0 else ("INT" if res["n_intents"] > 1 else "no-trade")
        print(f'     [{tag:5s}] ints={res["n_intents"]:>4} fills={res["total_fills"]:>4} '
              f'snaps={res["n_snapshots"]:>7} {mark}ret={ret:>7.3f}% sharpe={res["sharpe"]:>6.2f} '
              f'({res["wall_s"]:.1f}s)', flush=True)
        if res["total_fills"] == 0 and res.get("warnings"):
            print(f'     warnings: {res["warnings"]}', flush=True)
        print(flush=True)

    print("=" * 110, flush=True)
    n_fill = sum(1 for r in results if r.get("total_fills", 0) > 0)
    n_int = sum(1 for r in results if r.get("n_intents", 0) > 1 and r.get("total_fills", 0) == 0)
    n_zero = sum(1 for r in results if r.get("n_intents", 0) <= 1)
    print(f"Targeted: {len(results)} runs  fills>0: {n_fill}  intents-only: {n_int}  no-trade: {n_zero}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
