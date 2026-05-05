"""Worker loop that drains queued ``ProviderImportJob`` rows.

Lives on the ``discovery`` plane alongside other data-ingest work
(``discovery_worker``, ``tracked_traders_worker``).  Trading plane is
reserved for live execution; news plane for ML-heavy semantic search.
Provider import is bounded REST I/O — perfect fit for discovery.

Polls the queue every ``HOMERUN_PROVIDER_IMPORT_POLL_INTERVAL_SECONDS``
(default 5s) and runs at most one job at a time per worker process.
That's deliberate: each job already fans out internally (token-bucket
+ semaphore inside the polybacktest client), so running multiple jobs
in parallel would just trigger 429s.
"""
from __future__ import annotations

import asyncio
import logging
import os

from services.external_data.provider_import_service import (
    claim_next_queued_job,
    run_job,
)


logger = logging.getLogger("workers.provider_import")


def _poll_interval_seconds() -> float:
    raw = os.environ.get("HOMERUN_PROVIDER_IMPORT_POLL_INTERVAL_SECONDS", "5")
    try:
        parsed = float(raw)
    except Exception:
        parsed = 5.0
    return max(1.0, parsed)


async def start_loop() -> None:
    interval = _poll_interval_seconds()
    logger.info("Provider import worker starting (poll=%.1fs)", interval)

    while True:
        try:
            job = await claim_next_queued_job()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Provider import: queue claim failed")
            await asyncio.sleep(interval)
            continue

        if job is None:
            await asyncio.sleep(interval)
            continue

        logger.info(
            "Provider import: starting job=%s provider=%s payload=%s",
            job.id,
            job.provider,
            (job.payload_json or {}),
        )
        try:
            summary = await run_job(job.id)
            logger.info(
                "Provider import: completed job=%s status=%s snapshots_inserted=%s",
                job.id,
                summary.get("status", "completed"),
                summary.get("snapshots_inserted"),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Provider import: job %s crashed", job.id)
        # Loop back immediately to drain the queue when there is more work.
