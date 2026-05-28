"""Recompute resolved P&L for rows produced by the curPrice=0 phantom-win bug.

Background: Fix LL (commit) corrected two related bugs in
``position_lifecycle.py``:

  1. ``_live_trading_position_to_wallet_position`` was nullifying
     ``curPrice`` whenever ``current_price <= 0.0``, masking losing
     post-resolution positions.
  2. ``_extract_wallet_settlement_price`` then fell into a
     ``counts_as_open == False → return 1.0`` fallback, flipping every
     masked loser into a phantom $1.00 winner.

This script fixes the historical rows the bug already produced.  For
each resolved_* row whose ``payload_json.position_close.price_source``
is ``wallet_redeemable_mark``, we look up the corresponding Polymarket
position by token_id and:

  * If the position is actually a win (curPrice >= 0.999) → leave the
    row alone.
  * If the position is actually a loss (curPrice <= 0.001 or the token
    isn't in the wallet at all) → recompute close_price=0,
    realized_pnl=-cost_basis, status=resolved_loss, actual_profit=
    -cost_basis.  Stamp ``payload.position_close.recompute_reason``
    so the audit trail reflects the operator-driven correction.

Usage (from repo root):

    python scripts/maintenance/rerecompute_phantom_resolved_pnl.py \
        --trader-id fb1e2fc1e6bb47fbb5dd199dafc671d2 [--dry-run]

The wallet address comes from the trader's recorded
``execution_wallet_address`` on its first matching row.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _fetch_wallet_positions(wallet: str) -> dict[str, dict[str, Any]]:
    """Fetch open + closed positions for the wallet, keyed by asset."""
    by_asset: dict[str, dict[str, Any]] = {}
    for endpoint in ("positions", "closed-positions"):
        url = f"https://data-api.polymarket.com/{endpoint}?user={wallet}&limit=500"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "homerun-maintenance/1.0 (recompute-phantom-pnl)",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
        for entry in data:
            asset = str(entry.get("asset") or "").strip()
            if not asset:
                continue
            # Closed-positions takes priority — settled values are final.
            if endpoint == "closed-positions" or asset not in by_asset:
                by_asset[asset] = entry
    return by_asset


async def _main(trader_id: str, dry_run: bool) -> int:
    from sqlalchemy import select
    from models.database import AsyncSessionLocal, TraderOrder  # noqa: E402

    async with AsyncSessionLocal() as session:
        wallet_row = (
            await session.execute(
                select(TraderOrder.execution_wallet_address)
                .where(TraderOrder.trader_id == trader_id)
                .where(TraderOrder.execution_wallet_address.is_not(None))
                .limit(1)
            )
        ).scalar_one_or_none()
        if not wallet_row:
            print(f"No execution_wallet_address for trader {trader_id}")
            return 1
        wallet = str(wallet_row).strip().lower()
        print(f"Wallet: {wallet}")

        rows = list(
            (
                await session.execute(
                    select(TraderOrder)
                    .where(TraderOrder.trader_id == trader_id)
                    .where(TraderOrder.mode == "live")
                    .where(TraderOrder.status.in_(("resolved_win", "resolved_loss")))
                )
            )
            .scalars()
            .all()
        )

        candidates: list[TraderOrder] = []
        for row in rows:
            payload = row.payload_json if isinstance(row.payload_json, dict) else {}
            close = payload.get("position_close") or {}
            if str(close.get("price_source") or "") != "wallet_redeemable_mark":
                continue
            candidates.append(row)
        print(f"Found {len(candidates)} resolved rows with price_source=wallet_redeemable_mark")

    by_asset = _fetch_wallet_positions(wallet)
    print(f"Polymarket returned {len(by_asset)} positions for wallet")

    corrections: list[dict[str, Any]] = []
    for row in candidates:
        payload = row.payload_json if isinstance(row.payload_json, dict) else {}
        leg = payload.get("leg") or {}
        token_id = str(leg.get("token_id") or "").strip()
        snap = (payload.get("provider_reconciliation") or {}).get("snapshot") or {}
        filled_size = float(snap.get("filled_size") or 0.0)
        filled_notional = float(snap.get("filled_notional_usd") or 0.0)
        cost_basis = filled_notional or float(row.notional_usd or 0.0)
        if filled_size <= 0.0 or cost_basis <= 0.0:
            continue
        polymarket_entry = by_asset.get(token_id)
        if polymarket_entry is None:
            # Token isn't in the wallet's positions / closed-positions
            # API.  The data-api endpoints have an effective backlog
            # window — older resolved positions age out — so absence
            # is NOT evidence of a loss.  Skip; we cannot prove or
            # disprove the recorded status.  (A future iteration can
            # use the Subgraph or direct CTF queries to verify older
            # resolutions; for now, conservative skip.)
            continue
        cur_price = float(polymarket_entry.get("curPrice") or 0.0)
        # Decide the truth from the on-chain settlement price:
        #   cur_price >= 0.999 → definitively won, position pays $1
        #   cur_price <= 0.001 → definitively lost, position pays $0
        # Anything in between is an unsettled / partially-settled
        # state we shouldn't overwrite.
        if cur_price >= 0.999:
            actually_won = True
        elif cur_price <= 0.001:
            actually_won = False
        else:
            continue  # Indeterminate — don't touch.
        was_recorded_won = str(row.status) == "resolved_win"
        if actually_won == was_recorded_won:
            continue  # Already correct.
        proceeds = filled_size * (1.0 if actually_won else 0.0)
        new_pnl = proceeds - cost_basis
        new_status = "resolved_win" if actually_won else "resolved_loss"
        corrections.append(
            {
                "id": str(row.id),
                "market": row.market_question,
                "old_status": row.status,
                "new_status": new_status,
                "old_pnl": float(row.actual_profit or 0.0),
                "new_pnl": new_pnl,
                "cost_basis": cost_basis,
                "filled_size": filled_size,
                "polymarket_curPrice": cur_price,
                "polymarket_outcome": (polymarket_entry or {}).get("outcome"),
            }
        )

    print(f"\n{len(corrections)} rows need correction.\n")
    if not corrections:
        return 0

    print(f"{'id'[:36]:>36} | {'old':>8} | {'new':>8} | {'curPx':>5} | market")
    print("-" * 110)
    for c in corrections:
        print(
            f"{c['id'][:36]:>36} | "
            f"{c['old_pnl']:>+8.2f} | "
            f"{c['new_pnl']:>+8.2f} | "
            f"{c['polymarket_curPrice']:>5.2f} | "
            f"{(c['market'] or '')[:60]}"
        )
    delta = sum(c["new_pnl"] - c["old_pnl"] for c in corrections)
    print(f"\nNet P&L delta if applied: {delta:+.2f}")

    if dry_run:
        print("\n[dry-run] Not writing.")
        return 0

    async with AsyncSessionLocal() as session:
        for c in corrections:
            row = await session.get(TraderOrder, c["id"])
            if row is None:
                continue
            payload = dict(row.payload_json or {})
            close = dict(payload.get("position_close") or {})
            close["close_price"] = (1.0 if c["new_status"] == "resolved_win" else 0.0)
            close["realized_pnl"] = c["new_pnl"]
            close["recompute_reason"] = (
                "Fix LL: original wallet_redeemable_mark close_price was a "
                "phantom $1 produced by the curPrice=0 nullification bug; "
                "recomputed against on-chain curPrice."
            )
            payload["position_close"] = close
            row.payload_json = payload
            row.status = c["new_status"]
            row.actual_profit = c["new_pnl"]
        await session.commit()
        print(f"\nWrote {len(corrections)} corrections.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trader-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return asyncio.run(_main(args.trader_id, args.dry_run))


if __name__ == "__main__":
    sys.exit(main())
