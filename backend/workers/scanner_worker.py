"""
Scanner worker: runs scan loop and writes results to DB (scanner_snapshot).
API and other workers read opportunities/status from DB only.
Run from repo root: cd backend && python -m workers.scanner_worker
Or from backend: python -m workers.scanner_worker
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Optional

# Native numerical libs (OpenMP/BLAS/FAISS) can segfault under high thread
# contention in long-running workers on macOS; pin conservative defaults.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NEWS_FAISS_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("EMBEDDING_DEVICE", "cpu")

from utils.utcnow import utcnow
from config import settings, apply_runtime_settings_overrides
from models.database import AsyncSessionLocal
from services.scanner import scanner
from services.strategy_signal_bridge import bridge_opportunities_to_signals
from services.strategy_runtime import refresh_strategy_runtime_if_needed
from services.shared_state import (
    clear_scan_request,
    pop_targeted_condition_ids,
    read_scanner_control,
    read_scanner_snapshot,
    update_scanner_activity,
    write_scanner_snapshot,
)
from services.worker_state import write_worker_snapshot
from utils.logger import setup_logging

setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"), json_format=False)
logger = logging.getLogger("scanner_worker")


def _is_upstream_resolution_error(exc: Exception) -> bool:
    """Classify DNS/name-resolution upstream failures from HTTP client errors."""
    text = str(exc).lower()
    markers = (
        "nodename nor servname provided",
        "name or service not known",
        "temporary failure in name resolution",
    )
    return any(marker in text for marker in markers)


def _parse_iso_utc(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO timestamp into a UTC-aware datetime."""
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _build_market_history_payload(opps: list) -> Optional[dict[str, list[dict[str, float]]]]:
    """Build a non-empty market history payload for scanner snapshot writes."""
    market_history = scanner.get_broad_market_history(max_markets=500)

    # Ensure all opportunity markets are included even if they didn't make
    # the broad recency cap.
    opp_history = scanner.get_market_history_for_opportunities(opps)
    for market_id, points in opp_history.items():
        if market_id and market_id not in market_history and isinstance(points, list) and len(points) >= 2:
            market_history[market_id] = points

    # Fallback: preserve inline history that already exists on opportunities.
    for opp in opps:
        markets = getattr(opp, "markets", None)
        if not isinstance(markets, list):
            continue
        for market in markets:
            if not isinstance(market, dict):
                continue
            market_id = str(market.get("id", "") or "").strip()
            if not market_id or market_id in market_history:
                continue
            points = market.get("price_history")
            if not isinstance(points, list) or len(points) < 2:
                continue
            market_history[market_id] = points

    return market_history if market_history else None


async def _catalog_refresh_loop() -> None:
    """Background loop that refreshes the market catalog independently.

    Runs on its own timer (FULL_SCAN_INTERVAL_SECONDS) with its own timeout.
    The scan loop never depends on this completing — it uses the cached catalog.
    """
    while True:
        interval = max(60, settings.FULL_SCAN_INTERVAL_SECONDS)
        timeout = min(300, interval * 2.5)
        try:
            await asyncio.wait_for(scanner.refresh_catalog(), timeout=timeout)
        except asyncio.CancelledError:
            return
        except asyncio.TimeoutError:
            logger.warning("Catalog refresh timed out after %.0fs", timeout)
        except Exception as e:
            logger.warning("Catalog refresh failed: %s", e)
        await asyncio.sleep(interval)


async def _one_shot_catalog_refresh() -> None:
    """Fire-and-forget catalog refresh for manual scan requests."""
    try:
        await asyncio.wait_for(scanner.refresh_catalog(), timeout=180)
    except Exception as e:
        logger.warning("Manual catalog refresh failed (scan_fast will use cached catalog): %s", e)


async def _hydrate_scanner_pool_from_snapshot() -> int:
    """Restore scanner in-memory pool from DB snapshot on worker startup."""
    try:
        async with AsyncSessionLocal() as session:
            existing_opps, existing_status = await read_scanner_snapshot(session)
    except Exception as e:
        logger.warning("Scanner startup hydration skipped (snapshot read failed): %s", e)
        return 0

    now = datetime.now(timezone.utc)
    restored = []
    dropped_expired = 0
    for opp in existing_opps:
        rd = opp.resolution_date
        if rd is not None:
            if rd.tzinfo is None:
                rd = rd.replace(tzinfo=timezone.utc)
            else:
                rd = rd.astimezone(timezone.utc)
            if rd <= now:
                dropped_expired += 1
                continue
        restored.append(opp)

    scanner._opportunities = restored
    scanner._market_price_history = {}
    scanner._remember_market_tokens_from_opportunities(restored)
    now_ms = int(now.timestamp() * 1000)
    restored_history_markets = 0
    for opp in restored:
        for market in opp.markets:
            market_id = str(market.get("id", "") or "")
            if not market_id:
                continue
            history = market.get("price_history")
            if not isinstance(history, list) or not history:
                continue
            merged_len = scanner._merge_market_history_points(market_id, history, now_ms)
            if merged_len >= 2:
                restored_history_markets += 1
    if isinstance(existing_status, dict):
        parsed_last_scan = _parse_iso_utc(existing_status.get("last_scan"))
        if parsed_last_scan is not None:
            scanner._last_scan = parsed_last_scan

    if restored or dropped_expired:
        logger.info(
            "Hydrated scanner pool from DB snapshot: restored=%d dropped_expired=%d history_markets=%d",
            len(restored),
            dropped_expired,
            restored_history_markets,
        )
    return len(restored)


async def _reattach_inline_ai_from_snapshot(opps: list) -> int:
    """Reattach inline ai_analysis from the latest snapshot by stable_id.

    Manual `/ai/judge/opportunity` writes ai_analysis into scanner_snapshot
    immediately. The scanner worker keeps its own in-memory pool and can
    overwrite that snapshot on the next cycle; this restores inline AI fields
    before writing so card-level analysis does not disappear after refetch.
    """
    if not opps:
        return 0

    try:
        async with AsyncSessionLocal() as session:
            snapshot_opps, _ = await read_scanner_snapshot(session)
    except Exception:
        return 0

    ai_by_stable: dict[str, object] = {}
    for snapshot_opp in snapshot_opps:
        stable_id = getattr(snapshot_opp, "stable_id", None)
        analysis = getattr(snapshot_opp, "ai_analysis", None)
        if stable_id and analysis is not None:
            ai_by_stable[stable_id] = analysis

    if not ai_by_stable:
        return 0

    attached = 0
    for opp in opps:
        if getattr(opp, "ai_analysis", None) is not None:
            continue
        stable_id = getattr(opp, "stable_id", None)
        if not stable_id:
            continue
        existing = ai_by_stable.get(stable_id)
        if existing is None:
            continue
        opp.ai_analysis = existing
        attached += 1

    return attached


async def _run_scan_loop() -> None:
    """Load scanner, then loop: read control -> scan -> write snapshot -> sleep."""
    await scanner.load_settings()
    await scanner.load_plugins(source_keys=["scanner"])
    restored_count = await _hydrate_scanner_pool_from_snapshot()
    scanner._running = True
    scanner._enabled = True

    # Push live activity to DB so API/UI can show "Scanning Polymarket...", etc.
    async def _on_activity(activity: str) -> None:
        try:
            async with AsyncSessionLocal() as session:
                await update_scanner_activity(session, activity)
        except Exception as e:
            logger.debug("Activity update failed: %s", e)

    scanner.add_activity_callback(_on_activity)

    async def _ensure_ws_feeds_running() -> None:
        return

    async def _stop_ws_feeds() -> None:
        return

    # Scanner worker consumes Redis-backed live prices and cross-process
    # PRICE_CHANGE events; crypto worker owns WS feed ingestion.
    logger.info("Scanner worker initialized with Redis-driven price ingestion")

    # Opportunity recorder hooks into scanner callbacks in this process.
    try:
        from services.opportunity_recorder import opportunity_recorder

        await opportunity_recorder.start()
        logger.info("Opportunity recorder started")
    except Exception as e:
        logger.warning("Opportunity recorder start failed (non-critical): %s", e)

    logger.info("Scanner worker started (interval from DB)")

    # Hydrate market catalog from DB so scan_fast works immediately
    catalog_count = await scanner._hydrate_catalog_from_db()
    if catalog_count:
        logger.info("Hydrated market catalog from DB: %d markets", catalog_count)

    # Write initial status so API doesn't show "Waiting for scanner worker" before first scan
    try:
        current_opps = scanner.get_opportunities()
        current_count = len(current_opps)
        async with AsyncSessionLocal() as session:
            if restored_count > 0:
                activity = f"Scanner resumed with {restored_count} restored opportunities; next scan pending."
            else:
                activity = "Scanner started; first scan pending."
            startup_status = scanner.get_status()
            startup_status["running"] = True
            startup_status["enabled"] = True
            startup_status["interval_seconds"] = int(startup_status.get("interval_seconds") or 60)
            startup_status["current_activity"] = activity
            startup_status["opportunities_count"] = current_count
            await write_scanner_snapshot(session, current_opps, startup_status)
            await update_scanner_activity(session, activity)
            await write_worker_snapshot(
                session,
                "scanner",
                running=True,
                enabled=True,
                current_activity=activity,
                interval_seconds=60,
                last_run_at=None,
                last_error=None,
                stats={
                    "opportunities_count": current_count,
                    "signals_emitted_last_run": 0,
                },
            )
    except Exception:
        pass

    # Start background catalog refresh loop as an independent asyncio task.
    # This runs on its own timer and never blocks the scan loop.
    catalog_task = asyncio.create_task(_catalog_refresh_loop())

    stale_scan_streak = 0
    # Keep watchdog above fast-scan cadence while honoring configured ceiling.
    scan_watchdog_seconds = max(
        30,
        int(settings.SCAN_WATCHDOG_SECONDS),
        int(settings.FAST_SCAN_INTERVAL_SECONDS) * 3,
    )
    full_snapshot_watchdog_seconds = max(
        30,
        int(getattr(settings, "SCANNER_FULL_SNAPSHOT_WATCHDOG_SECONDS", 180) or 180),
    )
    heavy_scan_task: asyncio.Task | None = None
    pending_heavy_targeted_ids: list[str] | None = None

    async def _run_full_snapshot_cycle(
        *,
        reason: str,
        targeted_ids: list[str] | None,
        force: bool,
    ) -> None:
        try:
            await asyncio.wait_for(
                scanner.scan_full_snapshot_strategies(
                    reason=reason,
                    targeted_condition_ids=targeted_ids,
                    force=force,
                ),
                timeout=full_snapshot_watchdog_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Full-snapshot scan timed out after %ss",
                full_snapshot_watchdog_seconds,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Full-snapshot scan cycle failed")

    try:
        while True:
            async with AsyncSessionLocal() as session:
                control = await read_scanner_control(session)
                try:
                    # Rehydrate DB-backed runtime settings each loop using the
                    # global precedence chain: DB override > env > defaults.
                    await apply_runtime_settings_overrides()
                except Exception as exc:
                    logger.warning("Failed to refresh scanner runtime settings from DB: %s", exc)
                try:
                    await refresh_strategy_runtime_if_needed(
                        session,
                        source_keys=["scanner"],
                    )
                except Exception as exc:
                    logger.warning("Scanner strategy refresh check failed: %s", exc)

            interval = max(10, min(3600, control["scan_interval_seconds"] or 60))
            paused = control.get("is_paused", False)
            requested = control.get("requested_scan_at")

            if paused and not requested:
                if heavy_scan_task is not None and not heavy_scan_task.done():
                    heavy_scan_task.cancel()
                    try:
                        await heavy_scan_task
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        pass
                    heavy_scan_task = None
                pending_heavy_targeted_ids = None
                await _stop_ws_feeds()
                try:
                    async with AsyncSessionLocal() as session:
                        await write_worker_snapshot(
                            session,
                            "scanner",
                            running=True,
                            enabled=False,
                            current_activity="Paused",
                            interval_seconds=interval,
                            last_run_at=None,
                            last_error=None,
                            stats={"opportunities_count": 0, "signals_emitted_last_run": 0},
                        )
                except Exception:
                    pass
                await asyncio.sleep(min(10, interval))
                continue

            await _ensure_ws_feeds_running()

            # If a manual scan was requested, trigger an immediate catalog refresh
            # (non-blocking) so the next scan cycle uses the freshest data.
            targeted_ids = pop_targeted_condition_ids() if requested else []
            if requested:
                asyncio.create_task(_one_shot_catalog_refresh())
                pending_heavy_targeted_ids = list(targeted_ids)

            # Always use scan_fast — catalog refresh runs independently.
            scan_error: Exception | None = None
            try:
                reactive_tokens = await scanner.consume_reactive_tokens()
                await asyncio.wait_for(
                    scanner.scan_fast(
                        reactive_token_ids=reactive_tokens,
                        targeted_condition_ids=targeted_ids or None,
                    ),
                    timeout=scan_watchdog_seconds,
                )
                stale_scan_streak = 0
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError as exc:
                stale_scan_streak += 1
                logger.warning(
                    "Scanner scan cycle timed out after %ss (streak=%d)",
                    scan_watchdog_seconds,
                    stale_scan_streak,
                )
                # NOTE: we no longer clear _cached_markets on timeout streaks.
                # The catalog is persisted to DB and refreshed independently —
                # clearing the in-memory cache would cause a degenerate loop.
                if stale_scan_streak >= 5:
                    try:
                        await _stop_ws_feeds()
                        await _ensure_ws_feeds_running()
                    except Exception as fb_exc:
                        logger.warning("Scanner stale timeout heartbeat restart failed: %s", fb_exc)
                    stale_scan_streak = 0
                scan_error = exc
            except Exception as e:
                stale_scan_streak = 0
                scan_error = e

            if scan_error is not None:
                network_resolution_error = _is_upstream_resolution_error(scan_error)
                timeout_error = isinstance(scan_error, asyncio.TimeoutError)
                if timeout_error:
                    logger.warning("Scanner scan timeout handling after %.1fs: %s", scan_watchdog_seconds, scan_error)
                elif network_resolution_error:
                    logger.warning(
                        "Scanner upstream DNS/network resolution failed; retaining previous snapshot where possible: %s",
                        scan_error,
                    )
                else:
                    logger.exception("Scan failed: %s", scan_error)

                try:
                    prev_opps = scanner.get_opportunities()
                except Exception:
                    prev_opps = []

                existing_last_scan = None
                async with AsyncSessionLocal() as session:
                    try:
                        existing_opps, existing_status = await read_scanner_snapshot(session)
                    except Exception:
                        existing_opps, existing_status = [], {}
                if not prev_opps and existing_opps:
                    prev_opps = existing_opps
                if isinstance(existing_status, dict):
                    existing_last_scan = existing_status.get("last_scan")

                if timeout_error and prev_opps:
                    try:
                        attached = await scanner.attach_price_history_to_opportunities(
                            prev_opps,
                            timeout_seconds=0.0,
                        )
                        if attached > 0:
                            logger.info(
                                "Timeout recovery hydrated sparkline history for %d markets",
                                attached,
                            )
                    except Exception as history_exc:
                        logger.debug("Timeout recovery sparkline hydration skipped: %s", history_exc)

                if timeout_error:
                    activity = "Scanner cycle timeout; retaining last known snapshot."
                elif network_resolution_error:
                    activity = "Upstream market API DNS/network error; retaining last known snapshot."
                else:
                    activity = f"Last scan error: {scan_error}"

                async with AsyncSessionLocal() as session:
                    await write_scanner_snapshot(
                        session,
                        prev_opps,
                        {
                            "running": True,
                            "enabled": not paused,
                            "interval_seconds": interval,
                            "last_scan": existing_last_scan,
                            "current_activity": activity,
                            "strategies": [],
                        },
                        market_history=_build_market_history_payload(prev_opps),
                    )
                    await write_worker_snapshot(
                        session,
                        "scanner",
                        running=True,
                        enabled=not paused,
                        current_activity=activity,
                        interval_seconds=interval,
                        last_run_at=utcnow(),
                        last_error=str(scan_error),
                        stats={
                            "opportunities_count": len(prev_opps),
                            "signals_emitted_last_run": 0,
                        },
                    )
                sleep_seconds = settings.FAST_SCAN_INTERVAL_SECONDS if settings.TIERED_SCANNING_ENABLED else interval
                await asyncio.sleep(max(1, sleep_seconds))
                continue

            force_heavy_scan = pending_heavy_targeted_ids is not None
            targeted_for_heavy = list(pending_heavy_targeted_ids or [])
            if heavy_scan_task is None or heavy_scan_task.done():
                if heavy_scan_task is not None and heavy_scan_task.done():
                    try:
                        await heavy_scan_task
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        pass
                    heavy_scan_task = None
                run_heavy_now = force_heavy_scan or scanner._full_snapshot_strategy_due(datetime.now(timezone.utc))
                if run_heavy_now:
                    pending_heavy_targeted_ids = None
                    heavy_scan_task = asyncio.create_task(
                        _run_full_snapshot_cycle(
                            reason="manual_request" if force_heavy_scan else "scheduled",
                            targeted_ids=targeted_for_heavy if targeted_for_heavy else None,
                            force=force_heavy_scan,
                        ),
                        name="scanner-full-snapshot-lane",
                    )

            # Persist snapshot. Post-scan failures must not crash the worker loop.
            try:
                opps = scanner.get_opportunities()
            except Exception as e:
                logger.exception("Failed to fetch opportunities after scan: %s", e)
                opps = []

            # Reattach inline analysis previously persisted by API/manual judging.
            # This prevents worker snapshot writes from clobbering card AI data.
            if opps and any(getattr(o, "ai_analysis", None) is None for o in opps):
                try:
                    attached_inline = await _reattach_inline_ai_from_snapshot(opps)
                    if attached_inline:
                        logger.debug(
                            "Reattached %d inline AI analyses from snapshot",
                            attached_inline,
                        )
                except Exception as e:
                    logger.debug("Inline AI snapshot reattach skipped: %s", e)

            # API-triggered manual AI analysis runs in the API process and writes
            # OpportunityJudgment rows. Re-attach those judgments here so scanner
            # worker memory doesn't wipe them on the next snapshot write.
            if opps and any(getattr(o, "ai_analysis", None) is None for o in opps):
                try:
                    await asyncio.wait_for(scanner._attach_ai_judgments(opps), timeout=6)
                except asyncio.TimeoutError:
                    logger.debug("Timed out reattaching AI judgments in worker loop")
                except Exception as e:
                    logger.debug("AI judgment reattach skipped: %s", e)

            try:
                status = scanner.get_status()
            except Exception as e:
                logger.exception("Failed to build scanner status after scan: %s", e)
                last_scan = None
                try:
                    if scanner.last_scan:
                        ls = scanner.last_scan
                        if ls.tzinfo is None:
                            ls = ls.replace(tzinfo=timezone.utc)
                        else:
                            ls = ls.astimezone(timezone.utc)
                        last_scan = ls.replace(tzinfo=None).isoformat() + "Z"
                except Exception:
                    pass
                status = {
                    "running": True,
                    "enabled": not paused,
                    "interval_seconds": interval,
                    "last_scan": last_scan,
                    "current_activity": getattr(scanner, "_current_activity", None),
                    "strategies": [],
                }

            try:
                async with AsyncSessionLocal() as session:
                    # Re-read AI analysis fields from the current snapshot right before
                    # writing, inside the same session/transaction. This closes the
                    # TOCTOU window for manual AI judgments.
                    if opps:
                        try:
                            snapshot_opps, _ = await read_scanner_snapshot(session)
                            ai_by_stable: dict[str, object] = {}
                            for snap_opp in snapshot_opps:
                                stable_id = getattr(snap_opp, "stable_id", None)
                                analysis = getattr(snap_opp, "ai_analysis", None)
                                if stable_id and analysis is not None:
                                    ai_by_stable[stable_id] = analysis
                            if ai_by_stable:
                                merged = 0
                                for opp in opps:
                                    if getattr(opp, "ai_analysis", None) is not None:
                                        continue
                                    sid = getattr(opp, "stable_id", None)
                                    if sid and sid in ai_by_stable:
                                        opp.ai_analysis = ai_by_stable[sid]
                                        merged += 1
                                if merged:
                                    logger.debug(
                                        "Pre-write AI merge: attached %d analyses",
                                        merged,
                                    )
                        except Exception as e:
                            logger.debug("Pre-write AI merge skipped: %s", e)

                    market_history = _build_market_history_payload(opps)
                    await write_scanner_snapshot(session, opps, status, market_history=market_history)
                    emitted = await bridge_opportunities_to_signals(
                        session,
                        opps,
                        source="scanner",
                        quality_reports=scanner.quality_reports,
                        sweep_missing=True,
                    )
                    await clear_scan_request(session)
                    await write_worker_snapshot(
                        session,
                        "scanner",
                        running=True,
                        enabled=not paused,
                        current_activity="Idle - waiting for next scanner cycle.",
                        interval_seconds=interval,
                        last_run_at=utcnow(),
                        last_error=None,
                        stats={
                            "opportunities_count": len(opps),
                            "signals_emitted_last_run": int(emitted),
                            "full_snapshot_running": bool(
                                (status.get("tiered_scanning") or {}).get("full_snapshot_strategy_running", False)
                            ),
                        },
                    )
                logger.debug("Wrote snapshot: %d opportunities", len(opps))
            except Exception as e:
                logger.exception("Failed to persist scanner snapshot: %s", e)

            sleep_seconds = (
                settings.FAST_SCAN_INTERVAL_SECONDS if settings.TIERED_SCANNING_ENABLED and not requested else interval
            )
            sleep_seconds = max(1, sleep_seconds)

            # Reactive wake-up path: wait for significant WS price changes
            # (debounced in scanner/feed manager) or timeout fallback.
            if settings.TIERED_SCANNING_ENABLED and not requested:
                try:
                    scanner._register_reactive_scanning()
                    if scanner._pending_reactive_tokens:
                        scanner._reactive_trigger.set()
                    else:
                        scanner._reactive_trigger.clear()
                    await asyncio.wait_for(
                        scanner._reactive_trigger.wait(),
                        timeout=sleep_seconds,
                    )
                except asyncio.TimeoutError:
                    pass
                except Exception:
                    await asyncio.sleep(sleep_seconds)
            else:
                await asyncio.sleep(sleep_seconds)
    finally:
        catalog_task.cancel()
        try:
            await catalog_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        if heavy_scan_task is not None and not heavy_scan_task.done():
            heavy_scan_task.cancel()
            try:
                await heavy_scan_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass


async def start_loop() -> None:
    """Run the scanner worker loop (called from API process lifespan)."""
    try:
        await apply_runtime_settings_overrides()
    except Exception as exc:
        logger.warning("Initial scanner runtime settings refresh failed: %s", exc)
    try:
        await _run_scan_loop()
    except asyncio.CancelledError:
        logger.info("Scanner worker shutting down")
    finally:
        scanner._running = False
        try:
            from services.ws_feeds import get_feed_manager

            await get_feed_manager().stop()
        except Exception:
            pass
