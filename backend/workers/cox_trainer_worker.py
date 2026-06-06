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

    # Lightweight freshness probe — uses only models.database (already
    # imported) so a fresh-model boot pays NEITHER the ~150MB
    # pandas/scipy/lifelines import NOR the fit.  The heavy
    # ``train_and_persist`` import is deferred until a refit is actually due.
    from datetime import datetime, timezone

    from sqlalchemy import select

    from models.database import FillProbabilityModel

    async def _seconds_since_last_train() -> float | None:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(FillProbabilityModel.trained_at)
                .order_by(FillProbabilityModel.trained_at.desc())
                .limit(1)
            )
            latest = result.scalar_one_or_none()
        if latest is None:
            return None
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - latest).total_seconds())

    while True:
        try:
            age = await _seconds_since_last_train()
            # Training is a periodic batch job (default 6h).  Skip the refit
            # when the persisted model is still fresh — this keeps the
            # multi-second Cox fit (and its heavy import) off every process
            # restart instead of stalling the DB on each live boot.
            if age is not None and age < interval:
                remaining = max(60.0, float(interval) - age)
                logger.info(
                    "Cox trainer: active model fresh (age=%.0fs < %ds); "
                    "skipping retrain, next check in %.0fs",
                    age,
                    interval,
                    remaining,
                )
                await asyncio.sleep(remaining)
                continue

            # Only now pay the heavy lifelines/pandas/scipy import.
            from services.fill_simulator.cox_trainer import train_and_persist

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
