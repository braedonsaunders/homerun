"""Iterative strategy refinement harness — detect-mode dry runs.

Drives /api/validation/code-backtest (the detect() lifecycle-hook
dry-run endpoint) against the current source of negrisk.py or
stat_arb.py and prints a compact metrics table — opportunity count,
ROI distribution, subtype breakdown — so we can compare iterations.

This is NOT a real backtest: no fills, no PnL, no Sharpe.  For the
real thing use the BacktestStudio UI or POST /backtest/run, both of
which exercise the unified Cox-aware execution-realistic engine.
This script is the right tool when the question is "does my new
detect() logic find more/better opportunities right now?".

Usage:
    python scripts/strategy_iter.py negrisk
    python scripts/strategy_iter.py stat_arb --hours 48 --steps 96
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "backend" / "services" / "strategies"
API = "http://127.0.0.1:8000/api/validation/code-backtest"


def post(url: str, body: dict, timeout: float = 600.0) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}", "body": e.read().decode("utf-8")[:2000]}
    except Exception as e:
        return {"error": type(e).__name__, "body": str(e)[:500]}


def summarize(opps: list[dict]) -> dict:
    if not opps:
        return {"count": 0}
    rois = [float(o.get("roi_percent") or 0.0) for o in opps]
    risks = [float(o.get("risk_score") or 0.0) for o in opps]
    titles = [str(o.get("title") or "") for o in opps]
    sub = Counter()
    for t in titles:
        if t.startswith("NegRisk Short"): sub["negrisk_short"] += 1
        elif t.startswith("NegRisk:"):     sub["negrisk_long"]  += 1
        elif t.startswith("Multi-Outcome Short"): sub["multi_outcome_short"] += 1
        elif t.startswith("Multi-Outcome"):       sub["multi_outcome_long"] += 1
        elif t.startswith("Stat Arb"):     sub["stat_arb"] += 1
        elif t.startswith("Certainty Shock"): sub["certainty_shock"] += 1
        elif t.startswith("Temporal Decay"): sub["temporal_decay"] += 1
        else: sub["other"] += 1

    # categorize markets: sports / crypto / other
    sports_kw = ("vs.", " vs ", "win on ", "draw", "set 1", "game 1", "moneyline",
                 "raptors", "cavaliers", "lakers", "celtics", "knicks",
                 "lol:", "ufc", "atp", "wta", "nba", "nfl", "mlb", "nhl",
                 "premier league", "la liga", "kbo", "lpl", "bundesliga",
                 "champions league", "serie a", "ncaa", "valorant",
                 "counter-strike", "esports", "soccer", "football",
                 "tennis", "match", "fc ", " fc")
    market_breakdown = Counter()
    for t in titles:
        tl = t.lower()
        if any(k in tl for k in sports_kw):
            market_breakdown["sports"] += 1
        elif any(k in tl for k in ("bitcoin", "ethereum", "btc", "eth", "doge", "xrp", "solana", "crypto")):
            market_breakdown["crypto"] += 1
        else:
            market_breakdown["other"] += 1

    return {
        "count": len(opps),
        "roi_mean": round(statistics.mean(rois), 3),
        "roi_median": round(statistics.median(rois), 3),
        "roi_min": round(min(rois), 3),
        "roi_max": round(max(rois), 3),
        "roi_p90": round(sorted(rois)[int(0.9 * len(rois)) - 1] if len(rois) > 1 else rois[0], 3),
        "risk_mean": round(statistics.mean(risks), 3),
        "subtype_counts": dict(sub),
        "market_breakdown": dict(market_breakdown),
    }


def _fixup_imports(source: str) -> str:
    # The strategy_loader AST validator rejects relative imports.
    # Rewrite `from .base import X` -> `from services.strategies.base import X`
    # and `from .reversion_helpers ...` -> `from services.strategies.reversion_helpers ...`
    import re
    source = re.sub(
        r"^from \.(\w+) import",
        r"from services.strategies.\1 import",
        source,
        flags=re.MULTILINE,
    )
    return source


def run(strategy: str, *, hours: int, steps: int, max_markets: int, label: str = "current",
        source_override: str | None = None) -> dict:
    src_path = SRC / f"{strategy}.py"
    source = _fixup_imports(source_override if source_override is not None else src_path.read_text(encoding="utf-8"))

    body = {
        "source_code": source,
        "slug": f"_iter_{strategy}_{label}",
        "use_ohlc_replay": True,
        "replay_lookback_hours": hours,
        "replay_timeframe": "30m",
        "replay_max_markets": max_markets,
        "replay_max_steps": steps,
        "max_opportunities": 500,
    }
    t0 = time.time()
    print(f"[{strategy}/{label}] running backtest hours={hours} steps={steps} max_markets={max_markets} ...", flush=True)
    res = post(API, body, timeout=900)
    dt = time.time() - t0
    if "error" in res:
        print(f"  ERROR: {res['error']} {res.get('body','')[:300]}", flush=True)
        return res

    opps = res.get("opportunities") or []
    summary = summarize(opps)
    qr = res.get("quality_reports") or []
    qf_pass = sum(1 for r in qr if r.get("passed"))
    summary["qf_passed"] = qf_pass
    summary["qf_total"] = len(qr)
    summary["data_source"] = res.get("data_source", "?")
    summary["replay_steps"] = res.get("replay_steps", 0)
    summary["replay_markets"] = res.get("replay_markets", 0)
    summary["wall_seconds"] = round(dt, 1)
    summary["validation_warnings"] = res.get("validation_warnings", [])[:3]
    summary["runtime_error"] = res.get("runtime_error")

    print(json.dumps(summary, indent=2))
    return {"summary": summary, "opportunities_sample": opps[:5]}


def diff_run(strategy: str, *, hours: int, steps: int, max_markets: int,
             baseline_source: str, candidate_source: str) -> dict:
    """Run baseline + candidate back-to-back and emit a diff table."""
    print(f"\n{'='*72}\nA/B run: {strategy}  hours={hours} steps={steps} max_markets={max_markets}\n{'='*72}")
    base = run(strategy, hours=hours, steps=steps, max_markets=max_markets,
               label="baseline", source_override=baseline_source)
    cand = run(strategy, hours=hours, steps=steps, max_markets=max_markets,
               label="candidate", source_override=candidate_source)
    bs = base.get("summary", {}) if isinstance(base, dict) else {}
    cs = cand.get("summary", {}) if isinstance(cand, dict) else {}
    print(f"\n{'METRIC':<25} {'BASELINE':>15} {'CANDIDATE':>15} {'DELTA':>15}")
    for key in ("count", "qf_passed", "roi_mean", "roi_median", "roi_max",
                "risk_mean", "wall_seconds"):
        b = bs.get(key, 0); c = cs.get(key, 0)
        try:
            delta = round(float(c) - float(b), 3)
        except Exception:
            delta = "?"
        print(f"{key:<25} {b!s:>15} {c!s:>15} {delta!s:>15}")
    print(f"\nbaseline subtypes:  {bs.get('subtype_counts', {})}")
    print(f"candidate subtypes: {cs.get('subtype_counts', {})}")
    print(f"baseline markets:   {bs.get('market_breakdown', {})}")
    print(f"candidate markets:  {cs.get('market_breakdown', {})}")
    return {"baseline": base, "candidate": cand}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("strategy", choices=["negrisk", "stat_arb", "temporal_decay", "certainty_shock"])
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--steps", type=int, default=72)
    ap.add_argument("--max-markets", type=int, default=120)
    ap.add_argument("--label", default="current")
    ap.add_argument("--save", default=None, help="path to save full result JSON")
    ap.add_argument("--diff-against", default=None,
                    help="path to baseline source file for A/B diff")
    args = ap.parse_args()

    src_path = SRC / f"{args.strategy}.py"
    candidate_source = src_path.read_text(encoding="utf-8")

    if args.diff_against:
        baseline_source = Path(args.diff_against).read_text(encoding="utf-8")
        result = diff_run(args.strategy, hours=args.hours, steps=args.steps,
                          max_markets=args.max_markets,
                          baseline_source=baseline_source,
                          candidate_source=candidate_source)
    else:
        result = run(args.strategy, hours=args.hours, steps=args.steps,
                     max_markets=args.max_markets, label=args.label)
    if args.save:
        Path(args.save).write_text(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
