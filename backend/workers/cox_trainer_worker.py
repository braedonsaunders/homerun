"""Periodic trainer for the Cox PH fill probability model.

Runs once per ``train_interval_seconds`` (default 6h) on the news plane
— the trainer touches lifelines + pandas + scipy and would otherwise
add noticeable RAM to the trading-plane worker host.  The trader hot
path consumes the model via ``services.fill_simulator.cox_inference``,
which only loads the persisted ``fill_probability_models.active=True``
row — so the trainer's dependency stack stays out of the live order
path entirely.

Failure mode: if a single training cycle raises, we log and back off
for ``train_interval_seconds`` rather than tight-looping.  An empty
training set is logged at INFO and skipped silently.
"""
from __future__ import annotations

import asyncio
import logging
import os

from models.database import AsyncSessionLocal


logger = logging.getLogger("workers.cox_trainer")


def _train_interval_seconds() -> int:
    raw = os.environ.get("HOMERUN_COX_TRAIN_INTERVAL_SECONDS", "21600")
    try:
        parsed = int(raw)
    except Exception:
        parsed = 21600
    return max(900, parsed)  # floor: 15 minutes


def _train_window_days() -> int:
    raw = os.environ.get("HOMERUN_COX_TRAIN_WINDOW_DAYS", "30")
    try:
        parsed = int(raw)
    except Exception:
        parsed = 30
    return max(1, parsed)


async def start_loop() -> None:
    interval = _train_interval_seconds()
    window = _train_window_days()
    logger.info(
        "Cox trainer loop starting (interval=%ds, window=%dd)",
        interval,
        window,
    )

    # Defer the heavy import until the loop actually runs.  pandas +
    # scipy + lifelines is ~150MB resident; only worth paying once
    # we're in the long-lived loop, not at module-import time.
    from services.fill_simulator.cox_trainer import train_and_persist

    while True:
        try:
            async with AsyncSessionLocal() as session:
                results = await train_and_persist(
                    session,
                    window_days=window,
                )
            for r in results:
                logger.info(
                    "Cox trainer cycle: family=%s strata=%s n_events=%d c_index=%s notes=%s",
                    r.family,
                    r.strata_key,
                    r.n_events,
                    f"{r.concordance_index:.3f}" if r.concordance_index is not None else "n/a",
                    r.notes,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Cox trainer cycle failed")
        await asyncio.sleep(interval)
