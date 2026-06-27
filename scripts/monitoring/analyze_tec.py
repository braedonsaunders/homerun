"""Dissect tail_end_carry realized PnL to find the profitable regime.

Pulls full trader-order history and aggregates realized actual_profit across
the dimensions that distinguish a winning configuration from a bleeding one:
strategy_version, close_reason/trigger, direction, entry-price band, market
category, and resolution-held vs stopped-out.
"""
from __future__ import annotations

import json
import re
import urllib.request
from collections import defaultdict

BASE = "http://127.0.0.1:8000"


def fetch_all():
    out, offset = [], 0
    while True:
        url = f"{BASE}/api/traders/orders/all?status=all&limit=5000&since_seconds=0&offset={offset}"
        with urllib.request.urlopen(url, timeout=60) as r:
            page = json.loads(r.read().decode("utf-8")).get("orders", [])
        out.extend(page)
        if len(page) < 5000:
            break
        offset += 5000
    return out


def category(q: str) -> str:
    q = (q or "").lower()
    if any(k in q for k in ["btc", "bitcoin", "ethereum", "eth", "solana", "xrp", "doge", "crypto"]):
        return "crypto"
    if any(k in q for k in ["temperature", "°c", "°f", "celsius", "fahrenheit", "weather", "highest temp", "lowest temp"]):
        return "weather"
    if any(k in q for k in ["tweets", "mrbeast", "video", "youtube"]):
        return "social"
    if any(k in q for k in [" vs ", " vs.", "ufc", "nba", "nfl", "mlb", "goals", "pole", "t20", "handicap",
                            "set ", "match", "fight", "tennis", "soccer", "league", "counter-strike", "esports",
                            "grand prix", "o/u", "over/under", "corner", "score"]):
        return "sports"
    return "other"


def price_band(p):
    if p is None:
        return "?"
    if p < 0.85:
        return "<0.85"
    if p < 0.90:
        return "0.85-0.90"
    if p < 0.95:
        return "0.90-0.95"
    return ">=0.95"


def agg():
    return {"n": 0, "pnl": 0.0, "w": 0, "l": 0, "win_sum": 0.0, "loss_sum": 0.0}


def add(d, key, p):
    a = d[key]
    a["n"] += 1
    a["pnl"] += p
    if p > 0:
        a["w"] += 1; a["win_sum"] += p
    elif p < 0:
        a["l"] += 1; a["loss_sum"] += p


def show(title, d, top=None):
    rows = sorted(d.items(), key=lambda x: x[1]["pnl"], reverse=True)
    if top:
        rows = rows[:top]
    print(f"\n=== {title} ===")
    print(f"{'bucket':32} {'n':>4} {'PnL':>9} {'avg':>7} {'win%':>5} {'avgW':>7} {'avgL':>7}")
    for k, a in rows:
        wr = 100.0 * a["w"] / a["n"] if a["n"] else 0
        avg = a["pnl"] / a["n"] if a["n"] else 0
        avgw = a["win_sum"] / a["w"] if a["w"] else 0
        avgl = a["loss_sum"] / a["l"] if a["l"] else 0
        print(f"{str(k)[:32]:32} {a['n']:>4} {a['pnl']:>9.2f} {avg:>7.3f} {wr:>5.0f} {avgw:>7.3f} {avgl:>7.3f}")


def main():
    orders = fetch_all()
    tec = [o for o in orders if o.get("strategy_key") == "tail_end_carry" and o.get("actual_profit") is not None]
    total = sum(float(o["actual_profit"]) for o in tec)
    print(f"tail_end_carry realized trades: {len(tec)}  total realized PnL: {total:.2f}")

    by_ver, by_reason, by_trig, by_dir, by_band, by_cat, by_resmode = (
        defaultdict(agg) for _ in range(7))
    for o in tec:
        p = float(o["actual_profit"])
        add(by_ver, o.get("strategy_version"), p)
        add(by_reason, o.get("close_reason") or "(none)", p)
        add(by_trig, o.get("close_trigger") or "(none)", p)
        add(by_dir, o.get("direction"), p)
        add(by_band, price_band(o.get("entry_price")), p)
        add(by_cat, category(o.get("market_question")), p)
        st = str(o.get("status") or "")
        mode = "resolution_held" if st.startswith("resolved") else ("stopped_early" if st.startswith("closed") else st)
        add(by_resmode, mode, p)

    show("by strategy_version", by_ver)
    show("by close_reason", by_reason)
    show("by close_trigger", by_trig)
    show("by direction", by_dir)
    show("by entry_price band", by_band)
    show("by market category", by_cat)
    show("resolution-held vs stopped-early", by_resmode)


if __name__ == "__main__":
    main()
