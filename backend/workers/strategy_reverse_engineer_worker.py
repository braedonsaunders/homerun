"""Worker loop that drains queued ``StrategyReverseEngineerJob`` rows.

Same plane as ``provider_import_worker`` (discovery) since both are
operator-initiated, long-running, REST/LLM I/O-bound jobs.

Process at most one job at a time per worker — each job's agent loop
already issues many concurrent LLM/backtest calls, and stacking
multiple jobs would just trigger LLM provider rate limits.
"""
from __future__ import annotations

import asyncio
import logging
import os

from services.strategy_reverse_engineer.service import (
    claim_next_queued_job,
    run_job,
)

logger = logging.getLogger("workers.strategy_reverse_engineer")


def _poll_interval_seconds() -> float:
    raw = os.environ.get("HOMERUN_REVERSE_ENGINEER_POLL_INTERVAL_SECONDS", "5")
    try:
        parsed = float(raw)
    except Exception:
        parsed = 5.0
    return max(1.0, parsed)


async def start_loop() -> None:
    interval = _poll_interval_seconds()
    logger.info("Strategy reverse-engineer worker starting (poll=%.1fs)", interval)

    while True:
        try:
            job = await claim_next_queued_job()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("reverse-engineer: queue claim failed")
            await asyncio.sleep(interval)
            continue

        if job is None:
            await asyncio.sleep(interval)
            continue

        logger.info(
            "reverse-engineer: starting job=%s wallet=%s model=%s",
            job.id,
            job.wallet_address,
            job.llm_model,
        )
        try:
            summary = await run_job(job.id)
            logger.info(
                "reverse-engineer: completed job=%s status=%s best_score=%s",
                job.id,
                summary.get("status"),
                summary.get("best_score"),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("reverse-engineer: job %s crashed", job.id)
