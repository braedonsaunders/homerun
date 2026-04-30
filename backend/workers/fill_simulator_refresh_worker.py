"""Keep the fill-simulator caches warm.

The order hot path reads measured latency (from
``services/execution_latency_metrics.py`` rolling window),
empirical constants (from ``services/fill_simulator/empirical_
constants.py`` derived from book_delta_events), and the active
Cox model (from ``services/fill_simulator/cox_inference.py``).  All
three are cached in process; this worker refreshes them on a
background loop so the hot path never has to wait on a DB query.

Lives on the trading plane because all reads come from there.
Cheap loop: a few SELECTs + one in-process snapshot every 60 s.
"""
from __future__ import annotations

import asyncio
import logging

from services.fill_simulator.cox_inference import load_active_fill_model
from services.fill_simulator.empirical_constants import refresh_if_stale
from services.fill_simulator.latency import measured_latency_async


logger = logging.getLogger("workers.fill_simulator_refresh")


async def start_loop() -> None:
    logger.info("Fill simulator cache refresher starting")
    while True:
        try:
            await measured_latency_async()
            await refresh_if_stale()
            await load_active_fill_model(strata_key="pooled")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Fill simulator refresh failed")
        await asyncio.sleep(60)
