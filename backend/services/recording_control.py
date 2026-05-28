"""Global recording master switch.

A single source of truth for "is market-data recording currently enabled?"
backed by ``app_settings.recording_enabled``.  When the operator flips the
switch off, ALL recording stops writing to disk:

  * the live book/delta ingestor drops its flush batches
    (``services.market_data_ingestor._flush_batch``),
  * the proactive subscription loop unsubscribes and selects no tokens
    (``services.recorder_subscription_service``),
  * the crypto.update.dispatch bus tee is skipped
    (``services.market_runtime._publish_crypto_update_to_bus``).

The flag is read with a short TTL cache so hot-path callers (the ingestor
flush loop runs continuously) don't hit the DB on every batch, yet the
toggle still takes effect within a few seconds — no app restart required.

The setter (``set_recording_enabled``) is what the API handler calls; it
persists to the DB and primes the in-process cache so the change is visible
immediately in whichever process served the request.  Other processes pick
it up within ``_TTL`` seconds via their own refreshes.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from sqlalchemy import select

from models.database import AppSettings, AsyncSessionLocal

logger = logging.getLogger(__name__)

# Cache TTL (seconds).  Small enough that the toggle feels immediate, large
# enough that a continuous flush loop adds at most one tiny indexed read per
# this many seconds.
_TTL = 3.0

_cache_value: bool = True
_cache_ts: float = 0.0


async def is_recording_enabled() -> bool:
    """True if recording is enabled.  Refreshes from the DB at most once per
    ``_TTL`` seconds; on any error returns the last known value (fail-open:
    recording stays on so a transient DB blip can't silently drop data)."""
    global _cache_value, _cache_ts
    now = time.monotonic()
    if (now - _cache_ts) < _TTL and _cache_ts > 0.0:
        return _cache_value
    try:
        async with AsyncSessionLocal() as session:
            row = (await session.execute(select(AppSettings).limit(1))).scalar_one_or_none()
        # Null row (fresh install) or column default -> enabled.
        _cache_value = True if row is None else (getattr(row, "recording_enabled", True) is not False)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("recording_control: refresh failed, keeping cached=%s: %s", _cache_value, exc)
    _cache_ts = now
    return _cache_value


def is_recording_enabled_cached() -> bool:
    """Synchronous last-known value (no DB I/O).  For callers that can't
    await — returns the value from the most recent async refresh."""
    return _cache_value


def prime_cache(value: bool) -> None:
    """Seed the in-process cache (e.g. right after a setter) so the new value
    is visible immediately without waiting for the next TTL refresh."""
    global _cache_value, _cache_ts
    _cache_value = bool(value)
    _cache_ts = time.monotonic()


async def set_recording_enabled(enabled: bool) -> bool:
    """Persist the master switch and prime the local cache.  Returns the
    stored value.  Upserts the singleton AppSettings row if absent."""
    enabled = bool(enabled)
    async with AsyncSessionLocal() as session:
        row = (await session.execute(select(AppSettings).limit(1))).scalar_one_or_none()
        if row is None:
            row = AppSettings(recording_enabled=enabled)
            session.add(row)
        else:
            row.recording_enabled = enabled
        await session.commit()
    prime_cache(enabled)
    logger.info("recording_control: recording_enabled set to %s", enabled)
    return enabled


async def get_recording_enabled() -> bool:
    """Authoritative current value, bypassing the TTL cache (for the API GET
    so the UI always reflects the persisted state)."""
    async with AsyncSessionLocal() as session:
        row = (await session.execute(select(AppSettings).limit(1))).scalar_one_or_none()
    value = True if row is None else (getattr(row, "recording_enabled", True) is not False)
    prime_cache(value)
    return value
