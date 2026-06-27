"""Book-validated opportunity hunt.

The scanner ranks opportunities on mid/last prices, which systematically
overstates edge — most "arbs" evaporate at the executable ask. This tool
re-prices the top opportunities against the LIVE order book
(GET /api/markets/{token_id}/orderbook) and reports only edge that actually
survives at executable prices.

For binary markets with two CLOB tokens it computes the true CTF arb:
    buy YES@ask + buy NO@ask  ->  redeem $1  =>  edge = 1 - (yes_ask + no_ask)

Usage: python book_validated_hunt.py [--min-liquidity 3000] [--limit 60] [--min-edge 1.0]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8000"


def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


def book(token_id: str):
    try:
        return get(f"/api/markets/{token_id}/orderbook?depth=5")
    except Exception as exc:
        return {"error": str(exc)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-liquidity", type=float, default=3000.0)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--min-edge", type=float, default=1.0, help="min executable edge %% to report")
    args = ap.parse_args(argv)

    opps = get(
        f"/api/opportunities?source=markets&sort_by=roi&sort_dir=desc"
        f"&limit={args.limit}&min_liquidity={args.min_liquidity}"
    )
    real = []
    checked = 0
    for o in opps:
        mkts = o.get("markets") or []
        if not mkts:
            continue
        m = mkts[0]
        toks = m.get("clob_token_ids") or []
        if len(toks) != 2:
            continue
        yb, nb = book(toks[0]), book(toks[1])
        if "error" in yb or "error" in nb:
            continue
        ya, na = yb.get("best_ask"), nb.get("best_ask")
        if ya is None or na is None:
            continue
        checked += 1
        total = ya + na
        edge_pct = (1.0 - total) * 100.0
        # min executable depth across both legs (USD)
        depth = min(yb.get("ask_depth_usd") or 0.0, nb.get("ask_depth_usd") or 0.0)
        if edge_pct >= args.min_edge:
            real.append({
                "title": (o.get("title") or "")[:70],
                "scanner_roi": round(o.get("roi_percent") or 0.0, 1),
                "yes_ask": ya, "no_ask": na, "sum": round(total, 4),
                "executable_edge_pct": round(edge_pct, 2),
                "depth_usd": round(depth, 0),
                "yes_token": toks[0], "no_token": toks[1],
                "market_id": m.get("id"), "question": m.get("question"),
                "resolves": o.get("resolution_date"),
            })
    real.sort(key=lambda x: -x["executable_edge_pct"])
    print(json.dumps({
        "candidates_with_two_tokens_checked": checked,
        "real_arbs_found": len(real),
        "min_edge_pct": args.min_edge,
        "results": real,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
