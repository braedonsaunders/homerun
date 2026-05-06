"""Worker loop that drains queued ``BacktestRun`` rows.

Lives on the ``discovery`` plane alongside other long-running data /
ML jobs (provider import, strategy reverse-engineer).  Crucially NOT
on the trading plane — backtests are CPU-heavy (1M-snapshot replays
chew GIL for minutes) and cannot be allowed to block the live
orchestrator.  Running them in this dedicated worker process is the
"backtest can never fuck the orchestrator" guarantee.

Polls the queue every ``HOMERUN_BACKTEST_POLL_INTERVAL_SECONDS``
(default 3s) and runs at most one job at a time per worker process.
That's deliberate: each backtest already saturates a single core
inside the engine's matching loop, and the DB write path on
completion bursts hard for a few hundred ms.  Running multiple in
parallel on one process would just contend.

To scale up, run multiple discovery-plane processes — each one polls
independently and the ``FOR UPDATE SKIP LOCKED`` claim makes them
mutually exclusive.
"""
from __future__ import annotations

import asyncio
import logging
import os

from services.backtest.job_runner import (
    claim_next_queued_run,
    run_job,
)


logger = logging.getLogger("workers.backtest")


def _poll_interval_seconds() -> float:
    raw = os.environ.get("HOMERUN_BACKTEST_POLL_INTERVAL_SECONDS", "3")
    try:
        parsed = float(raw)
    except Exception:
        parsed = 3.0
    return max(0.5, parsed)


async def start_loop() -> None:
    interval = _poll_interval_seconds()
    logger.info("Backtest worker starting (poll=%.1fs)", interval)

    while True:
        try:
            run = await claim_next_queued_run()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Backtest worker: queue claim failed")
            await asyncio.sleep(interval)
            continue

        if run is None:
            await asyncio.sleep(interval)
            continue

        logger.info(
            "Backtest worker: starting run=%s slug=%s",
            run.id,
            run.strategy_slug,
        )
        try:
            summary = await run_job(run.id)
            logger.info(
                "Backtest worker: completed run=%s status=%s trades=%s return_pct=%s elapsed_ms=%.0f",
                run.id,
                summary.get("status"),
                summary.get("trade_count"),
                summary.get("total_return_pct"),
                float(summary.get("elapsed_ms") or 0.0),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Backtest worker: run %s crashed unexpectedly", run.id)
        # Loop back immediately to drain the queue when there is more work.
