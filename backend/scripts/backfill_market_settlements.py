"""Offline backfill for the golden settlement store.

Populates ``market_settlements`` with resolved winners for every market in
a window, so subsequent backtests settle held positions at $1/$0 fully
OFFLINE — reproducible, no per-run network, no decision-time look-ahead.
Re-runnable: resolved rows are upserted; markets that don't resolve yet are
left for a later pass.

Usage (from backend/):
    python -m scripts.backfill_market_settlements --days 30
    python -m scripts.backfill_market_settlements --start 2026-05-01 --end 2026-06-01
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


async def _run(start: datetime, end: datetime) -> int:
    from services.backtest.settlement_store import populate_settlements
    from services.strategy_backtester import collect_settlement_metadata

    _token_meta, hints = await collect_settlement_metadata(
        start_dt=start, end_dt=end, token_scope=None
    )
    hint_list = list(hints.values())
    print(
        f"Collected {len(hint_list)} markets in [{start.isoformat()}, "
        f"{end.isoformat()}]; resolving winners..."
    )
    if not hint_list:
        print("Nothing to resolve.")
        return 0
    records = await populate_settlements(hint_list)
    resolved = sum(1 for r in records.values() if r.resolved)
    print(f"Resolved + stored {resolved}/{len(records)} market winners.")
    return resolved


def _parse_dt(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=None, help="window length ending now")
    ap.add_argument("--start", type=str, default=None, help="ISO start (UTC)")
    ap.add_argument("--end", type=str, default=None, help="ISO end (UTC)")
    args = ap.parse_args()

    end = _parse_dt(args.end) if args.end else datetime.now(timezone.utc)
    if args.start:
        start = _parse_dt(args.start)
    else:
        start = end - timedelta(days=args.days or 30)
    asyncio.run(_run(start, end))


if __name__ == "__main__":
    main()
