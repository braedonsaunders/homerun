"""One-shot dataset backfill for backtest evaluation.

Fills ``market_microstructure_snapshots`` with historical book data for
the strategies you want to backtest.  Two phases:

  1. **Polymarket REST** (``recorder_backfill_service.run_backfill``)
     -- synthesizes a single-level book from /prices-history mids for
     tail_end_carry's full opportunity-token universe.  No API key
     needed; rate-limited to Polymarket's CLOB defaults.

  2. **Polybacktest provider** (``provider_import_service``) -- pulls
     full-depth L2 sub-second snapshots for BTC/ETH/SOL Up/Down
     markets that crypto strategies operate on.  Requires the
     polybacktest_api_key in AppSettings (already configured).

Usage::

    python -m scripts.backfill_strategy_data \\
      --tail-end-days 7 \\
      --crypto-coins btc eth sol \\
      --crypto-hours 48 \\
      --tail-end-fidelity 1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


logger = logging.getLogger("backfill_strategy_data")


async def backfill_tail_end_carry(
    *,
    days: int,
    fidelity_minutes: int,
    concurrency: int,
    max_tokens: int,
) -> dict:
    """REST backfill for tail_end_carry's full opp-token universe."""
    from services.recorder_backfill_service import run_backfill

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    print("\n=== tail_end_carry backfill ===")
    print(f"  window: {start.isoformat()} -> {end.isoformat()} ({days}d)")
    print(f"  fidelity: {fidelity_minutes}m  concurrency: {concurrency}  cap: {max_tokens}")
    print("  source: Polymarket REST /prices-history (synthetic single-level book)")
    t0 = time.monotonic()

    result = await run_backfill(
        scope="strategy",
        strategy_slug="tail_end_carry",
        start=start,
        end=end,
        interval="1m",                 # high-frequency mid samples
        fidelity_minutes=fidelity_minutes,
        concurrency=concurrency,
        max_tokens=max_tokens,
    )
    elapsed = time.monotonic() - t0
    summary = {
        "tokens_targeted": result.target_token_count,
        "tokens_with_data": result.tokens_with_data,
        "tokens_with_errors": result.tokens_with_errors,
        "rows_inserted": result.rows_inserted_total,
        "points_fetched": result.points_fetched_total,
        "skipped_existing": result.skipped_existing_total,
        "wall_seconds": round(elapsed, 1),
        "error": result.error,
    }
    print(f"  completed in {elapsed:.1f}s")
    for k, v in summary.items():
        print(f"    {k:24s}: {v}")
    return summary


async def backfill_crypto_polybacktest(
    *,
    coins: list[str],
    hours: int,
    market_limit_per_coin: int,
) -> dict:
    """Polybacktest L2 import for crypto Up/Down markets.

    Walks recent BTC/ETH/SOL markets (those that resolved within the
    last ``hours``) and queues a polybacktest import job per coin --
    pulls full L2 snapshots into ``market_microstructure_snapshots``.
    This is the high-fidelity path: sub-second captures, full depth,
    NOT synthetic.

    The job is created via ``enqueue_polybacktest_import`` then
    immediately driven to completion via ``run_job`` -- no separate
    worker required for the one-shot script flow.
    """
    from services.external_data.polybacktest_client import (
        PolybacktestNotConfiguredError,
        build_client_from_settings,
    )
    from services.external_data.provider_import_service import (
        CreatePolybacktestJobSpec,
        enqueue_polybacktest_import,
        run_job,
    )

    print("\n=== crypto backfill (polybacktest) ===")
    print(f"  coins: {coins}  hours: {hours}")
    print("  source: polybacktest.com /v2/markets (full L2)")

    try:
        client = await build_client_from_settings()
    except PolybacktestNotConfiguredError as exc:
        print(f"  SKIPPED: {exc}")
        return {"skipped": True, "reason": str(exc)}

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    out: dict = {"per_coin": {}}
    try:
        for coin in coins:
            t0 = time.monotonic()
            try:
                # Discover markets that overlap our window.  Polybacktest
                # caps ``limit`` at 100 per page, so we paginate until
                # we have ``market_limit_per_coin`` total OR exhaust
                # the catalog (offset >= total).  Default sort is most-
                # recent first, which matches our "want fresh markets"
                # backfill goal.
                _PAGE = 100
                all_markets: list = []
                total = 0
                offset = 0
                while len(all_markets) < market_limit_per_coin:
                    page_limit = min(_PAGE, market_limit_per_coin - len(all_markets))
                    page, total = await client.list_markets(
                        coin=coin,
                        limit=page_limit,
                        offset=offset,
                    )
                    if not page:
                        break
                    all_markets.extend(page)
                    offset += len(page)
                    if offset >= total:
                        break
                # Filter to markets whose [start_time, end_time] overlap
                # our window -- no point importing a market that doesn't
                # have any snapshots in our range.
                in_window: list[str] = []
                for m in all_markets:
                    m_start_ms = int(m.start_time.timestamp() * 1000) if m.start_time else 0
                    m_end_ms = int(m.end_time.timestamp() * 1000) if m.end_time else end_ms
                    # overlap: m_start <= end AND m_end >= start
                    if m_start_ms <= end_ms and m_end_ms >= start_ms:
                        in_window.append(m.market_id)
                print(f"  {coin}: discovered {len(all_markets)}/{total} markets, "
                      f"{len(in_window)} overlap window")
                if not in_window:
                    out["per_coin"][coin] = {
                        "markets_in_window": 0,
                        "snapshots_imported": 0,
                        "wall_seconds": round(time.monotonic() - t0, 1),
                    }
                    continue

                # Queue + run inline.
                spec = CreatePolybacktestJobSpec(
                    coin=coin,
                    market_ids=in_window,
                    start_ms=start_ms,
                    end_ms=end_ms,
                )
                job = await enqueue_polybacktest_import(spec)
                print(f"  {coin}: enqueued job {job.id} ({len(in_window)} markets), running...")
                summary = await run_job(job.id)
                elapsed = time.monotonic() - t0
                out["per_coin"][coin] = {
                    "job_id": job.id,
                    "markets_in_window": len(in_window),
                    "snapshots_inserted": summary.get("snapshots_inserted", summary.get("rows_inserted")),
                    "skipped_existing": summary.get("skipped_existing"),
                    "wall_seconds": round(elapsed, 1),
                }
                print(f"  {coin}: {out['per_coin'][coin]}")
            except Exception as exc:
                out["per_coin"][coin] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"  {coin}: FAILED -- {exc}")
    finally:
        await client.close()
    return out


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tail-end-days", type=int, default=7,
                   help="window for tail_end_carry backfill (default 7)")
    p.add_argument("--tail-end-fidelity", type=int, default=1,
                   help="prices-history fidelity in minutes (default 1)")
    p.add_argument("--tail-end-concurrency", type=int, default=8,
                   help="parallel fetches (default 8)")
    p.add_argument("--tail-end-max-tokens", type=int, default=2000,
                   help="hard cap on token count (default 2000)")
    p.add_argument("--crypto-coins", nargs="+", default=["btc", "eth", "sol"],
                   help="coins to import via polybacktest")
    p.add_argument("--crypto-hours", type=int, default=48,
                   help="window for crypto import (default 48h)")
    p.add_argument("--crypto-markets-per-coin", type=int, default=80,
                   help="cap on markets discovered per coin (default 80)")
    p.add_argument("--skip-tail-end", action="store_true")
    p.add_argument("--skip-crypto", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    report: dict = {"started_at": datetime.now(timezone.utc).isoformat()}

    if not args.skip_tail_end:
        report["tail_end_carry"] = await backfill_tail_end_carry(
            days=args.tail_end_days,
            fidelity_minutes=args.tail_end_fidelity,
            concurrency=args.tail_end_concurrency,
            max_tokens=args.tail_end_max_tokens,
        )

    if not args.skip_crypto:
        report["crypto"] = await backfill_crypto_polybacktest(
            coins=[c.lower() for c in args.crypto_coins],
            hours=args.crypto_hours,
            market_limit_per_coin=args.crypto_markets_per_coin,
        )

    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    print("\n=== summary ===")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
