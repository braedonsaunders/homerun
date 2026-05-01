"""Redeemer worker: scheduled dry-runs plus explicit on-demand CTF redemption."""

from __future__ import annotations

import asyncio
from typing import Any

from sqlalchemy.exc import OperationalError

from models.database import AsyncSessionLocal, init_database, recover_pool
from services.ctf_execution import ctf_execution_service
from services.live_execution_service import live_execution_service
from services.polymarket_collateral import collateral_registry
from services.worker_state import (
    _is_retryable_db_error,
    clear_worker_run_request,
    read_worker_control,
    write_worker_snapshot,
)
from utils.logger import get_logger
from utils.utcnow import utcnow

logger = get_logger("redeemer_worker")

WORKER_NAME = "redeemer"
DEFAULT_INTERVAL_SECONDS = 120
_IDLE_SLEEP_SECONDS = 2
_MAX_CONSECUTIVE_DB_FAILURES = 3
# How long we stay in the hard-stop state after a boot-invariant
# failure before re-attempting verification.  Operator-actionable
# failures (e.g. NegRiskAdapter redeployment) typically need a code
# update, but a transient RPC failure shouldn't lock us out forever —
# 15 minutes is a balance between "loud failure" and "self-heal".
_BOOT_INVARIANT_RETRY_SECONDS = 900.0
# Dry-run scans (the default scheduled cycle) only do view calls and finish
# in seconds, but a real-mode run has to walk every resolved condition with
# multiple chain RPCs (payoutDenominator + per-token balance + redeem) and
# can comfortably exceed 90s on slower providers.  Give real cycles a much
# wider budget; dry-runs still complete inside their normal envelope.
_REDEEM_CYCLE_TIMEOUT_SECONDS = 90.0
_REDEEM_REAL_CYCLE_TIMEOUT_SECONDS = 240.0


async def _verify_boot_invariants() -> str | None:
    """One-shot boot check before the redeemer enters its loop.

    Asserts every registered NegRiskAdapter's on-chain ``col``/``wcol``
    match the static registry. Returns ``None`` on success or an
    error string on failure — the loop refuses to redeem until a
    subsequent retry succeeds (handles transient RPC outages while
    still failing loudly on genuine adapter drift).
    """
    try:
        w3 = await ctf_execution_service._get_web3()
    except Exception as exc:
        return f"rpc_unavailable_for_invariant_check:{exc}"
    try:
        await collateral_registry.verify_invariants(w3)
    except Exception as exc:
        return str(exc)
    return None


async def _run_redeem_cycle(*, dry_run: bool) -> dict[str, Any]:
    if not await live_execution_service.ensure_initialized():
        return {
            "wallet_address": "",
            "positions_scanned": 0,
            "conditions_checked": 0,
            "resolved_conditions": 0,
            "redeemed": 0,
            "failed": 0,
            "dry_run": bool(dry_run),
            "errors": ["trading_service_not_initialized"],
        }
    return await ctf_execution_service.redeem_resolved_wallet_positions(dry_run=dry_run)


async def run_worker_loop() -> None:
    interval_seconds = DEFAULT_INTERVAL_SECONDS
    is_enabled = True
    is_paused = False
    requested_run = False
    consecutive_db_failures = 0

    while True:
        try:
            async with AsyncSessionLocal() as session:
                control = await read_worker_control(
                    session,
                    WORKER_NAME,
                    default_interval=DEFAULT_INTERVAL_SECONDS,
                )
                interval_seconds = max(5, int(control.get("interval_seconds") or DEFAULT_INTERVAL_SECONDS))
                is_enabled = bool(control.get("is_enabled", True))
                is_paused = bool(control.get("is_paused", False))
                requested_run = bool(control.get("requested_run_at") is not None)
                if requested_run:
                    await clear_worker_run_request(session, WORKER_NAME)

                if not is_enabled or is_paused:
                    await write_worker_snapshot(
                        session,
                        WORKER_NAME,
                        running=False,
                        enabled=bool(is_enabled and not is_paused),
                        current_activity="Paused" if is_paused else "Disabled",
                        interval_seconds=interval_seconds,
                        stats={
                            "requested_run": bool(requested_run),
                        },
                    )

            if not is_enabled or is_paused:
                await asyncio.sleep(max(_IDLE_SLEEP_SECONDS, interval_seconds))
                continue

            # Boot-time invariant check: assert every registered
            # NegRiskAdapter's on-chain ``col``/``wcol`` match our
            # static expectations before we will touch any redemption.
            # A violation hard-stops the worker for
            # ``_BOOT_INVARIANT_RETRY_SECONDS``, surfacing
            # ``collateral_invariants_violated:...`` to the operator
            # dashboard. Once verification succeeds, ``invariants_verified``
            # latches True for the remainder of the process lifetime —
            # we never re-check the gate during normal cycles.
            if not collateral_registry.invariants_verified():
                invariant_failure = await _verify_boot_invariants()
                if invariant_failure is not None:
                    async with AsyncSessionLocal() as session:
                        await write_worker_snapshot(
                            session,
                            WORKER_NAME,
                            running=False,
                            enabled=True,
                            current_activity="Boot invariant failed — refusing to redeem",
                            interval_seconds=interval_seconds,
                            last_error=f"collateral_invariants_violated:{invariant_failure}",
                            stats={
                                "invariant_failure": invariant_failure,
                                "retry_in_seconds": _BOOT_INVARIANT_RETRY_SECONDS,
                            },
                        )
                    logger.error(
                        "Redeemer refusing to start: collateral invariants violated: %s",
                        invariant_failure,
                    )
                    await asyncio.sleep(_BOOT_INVARIANT_RETRY_SECONDS)
                    continue

            cycle_reason = "requested" if requested_run else "scheduled"
            cycle_timeout = (
                _REDEEM_REAL_CYCLE_TIMEOUT_SECONDS
                if requested_run
                else _REDEEM_CYCLE_TIMEOUT_SECONDS
            )
            summary = await asyncio.wait_for(
                _run_redeem_cycle(dry_run=not requested_run),
                timeout=cycle_timeout,
            )

            async with AsyncSessionLocal() as session:
                await write_worker_snapshot(
                    session,
                    WORKER_NAME,
                    running=True,
                    enabled=True,
                    current_activity=(
                        f"{'DryRun' if bool(summary.get('dry_run')) else 'Redeem'}="
                        f"{int(summary.get('redeemed', 0))}, "
                        f"resolved={int(summary.get('resolved_conditions', 0))}, "
                        f"failed={int(summary.get('failed', 0))}"
                    ),
                    interval_seconds=interval_seconds,
                    last_run_at=utcnow(),
                    stats={
                        **summary,
                        "cycle_reason": cycle_reason,
                    },
                )

            consecutive_db_failures = 0
            await asyncio.sleep(interval_seconds)
        except asyncio.TimeoutError:
            logger.warning(
                "Redeemer worker cycle timed out after %.0fs",
                _REDEEM_CYCLE_TIMEOUT_SECONDS,
            )
            try:
                async with AsyncSessionLocal() as session:
                    await write_worker_snapshot(
                        session,
                        WORKER_NAME,
                        running=True,
                        enabled=bool(is_enabled and not is_paused),
                        current_activity="Redeemer cycle timed out",
                        interval_seconds=interval_seconds,
                        stats={
                            "requested_run": bool(requested_run),
                            "timed_out": True,
                        },
                    )
            except Exception:
                pass
            await asyncio.sleep(_IDLE_SLEEP_SECONDS)
        except OperationalError as exc:
            if not _is_retryable_db_error(exc):
                raise
            consecutive_db_failures += 1
            logger.warning(
                "Redeemer worker cycle hit transient DB error (failure=%d)",
                consecutive_db_failures,
                exc_info=exc,
            )
            if consecutive_db_failures >= _MAX_CONSECUTIVE_DB_FAILURES:
                await recover_pool()
                consecutive_db_failures = 0
                logger.warning("Recovered DB connection pool after redeemer worker disconnects")
            await asyncio.sleep(_IDLE_SLEEP_SECONDS)
        except Exception as exc:
            logger.exception("Redeemer worker cycle failed: %s", exc)
            try:
                async with AsyncSessionLocal() as session:
                    await write_worker_snapshot(
                        session,
                        WORKER_NAME,
                        running=False,
                        enabled=True,
                        current_activity="Worker error",
                        interval_seconds=interval_seconds,
                        last_error=str(exc),
                    )
            except Exception:
                pass
            await asyncio.sleep(_IDLE_SLEEP_SECONDS)


async def start_loop() -> None:
    try:
        await run_worker_loop()
    except asyncio.CancelledError:
        logger.info("Redeemer worker shutting down")


async def main() -> None:
    await init_database()
    try:
        await run_worker_loop()
    except asyncio.CancelledError:
        logger.info("Redeemer worker shutting down")


if __name__ == "__main__":
    asyncio.run(main())
