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


# ── Recorder configuration (depth / coverage / capture toggles) ────────────
#
# Operator-tunable recording knobs, persisted as a single JSON blob in
# ``app_settings.recorder_config_json`` so new knobs can be added without a
# schema migration.  Read live with the same short-TTL cache as the master
# switch so hot-path callers (the ingestor's level-truncation, the proactive
# subscription loop) pick up changes within ``_CONFIG_TTL`` seconds without
# hitting the DB on every snapshot / tick.
#
# Defaults mirror the historical hardcoded constants:
#   * depth_levels      -> market_data_ingestor._MAX_LEVELS_PER_SIDE (25)
#   * max_tokens        -> recorder_subscription_service._DEFAULT_MAX_TOKENS (8000)
#   * min_liquidity_usd -> recorder_subscription_service._MIN_LIQUIDITY_USD (10.0)
#   * capture_books / capture_trades / capture_catalog default True.

# A touch longer than the master-switch TTL: these knobs change rarely and
# the depth value is read on the per-snapshot hot path.
_CONFIG_TTL = 5.0

_DEPTH_MIN = 1
_DEPTH_MAX = 25

# Authoritative key set + defaults.  Any persisted blob is filtered/merged
# against this so a stray key can never leak into the config and a missing
# key always falls back to the default.
_CONFIG_DEFAULTS: dict[str, object] = {
    "depth_levels": 25,
    "max_tokens": 8000,
    "min_liquidity_usd": 10.0,
    "capture_books": True,
    "capture_trades": True,
    "capture_catalog": True,
}

_config_cache: dict[str, object] = dict(_CONFIG_DEFAULTS)
_config_cache_ts: float = 0.0


def _clamp_depth(value: object) -> int:
    try:
        iv = int(value)
    except (TypeError, ValueError):
        iv = int(_CONFIG_DEFAULTS["depth_levels"])  # type: ignore[arg-type]
    return max(_DEPTH_MIN, min(_DEPTH_MAX, iv))


def _coerce_config(raw: object) -> dict[str, object]:
    """Merge a persisted/partial blob onto the defaults, coercing types and
    clamping ranges.  Unknown keys are dropped; missing keys take defaults."""
    out: dict[str, object] = dict(_CONFIG_DEFAULTS)
    if not isinstance(raw, dict):
        return out
    if "depth_levels" in raw:
        out["depth_levels"] = _clamp_depth(raw.get("depth_levels"))
    if "max_tokens" in raw:
        try:
            out["max_tokens"] = max(0, int(raw.get("max_tokens")))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
    if "min_liquidity_usd" in raw:
        try:
            out["min_liquidity_usd"] = max(0.0, float(raw.get("min_liquidity_usd")))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
    for key in ("capture_books", "capture_trades", "capture_catalog"):
        if key in raw:
            out[key] = bool(raw.get(key))
    return out


def prime_config_cache(config: dict[str, object]) -> None:
    """Seed the in-process config cache (e.g. right after a setter) so the new
    value is visible immediately without waiting for the next TTL refresh."""
    global _config_cache, _config_cache_ts
    _config_cache = _coerce_config(config)
    _config_cache_ts = time.monotonic()


async def get_recorder_config() -> dict[str, object]:
    """Current recorder config, refreshed from the DB at most once per
    ``_CONFIG_TTL`` seconds.  On any error returns the last known value
    (fail-soft: keeps the previous config rather than reverting to defaults
    on a transient DB blip).  Always a full dict with every key present."""
    global _config_cache, _config_cache_ts
    now = time.monotonic()
    if (now - _config_cache_ts) < _CONFIG_TTL and _config_cache_ts > 0.0:
        return dict(_config_cache)
    try:
        async with AsyncSessionLocal() as session:
            row = (await session.execute(select(AppSettings).limit(1))).scalar_one_or_none()
        raw = None if row is None else getattr(row, "recorder_config_json", None)
        _config_cache = _coerce_config(raw)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("recording_control: config refresh failed, keeping cached: %s", exc)
    _config_cache_ts = now
    return dict(_config_cache)


def get_recorder_config_cached() -> dict[str, object]:
    """Synchronous last-known config (no DB I/O).  For callers that can't
    await — returns the value from the most recent async refresh."""
    return dict(_config_cache)


async def set_recorder_config(**partial: object) -> dict[str, object]:
    """Persist a partial recorder-config update and prime the local cache.

    Only the keys present in ``partial`` are changed; the rest are preserved
    from the currently-stored blob.  Unknown keys are ignored, values are
    coerced/clamped.  Returns the full merged config.  Upserts the singleton
    AppSettings row if absent.
    """
    # Filter to known keys before merging so a stray kwarg can't be stored.
    incoming = {k: v for k, v in partial.items() if k in _CONFIG_DEFAULTS}
    async with AsyncSessionLocal() as session:
        row = (await session.execute(select(AppSettings).limit(1))).scalar_one_or_none()
        existing = _coerce_config(getattr(row, "recorder_config_json", None) if row is not None else None)
        merged = _coerce_config({**existing, **incoming})
        if row is None:
            row = AppSettings(recorder_config_json=merged)
            session.add(row)
        else:
            row.recorder_config_json = merged
        await session.commit()
    prime_config_cache(merged)
    logger.info("recording_control: recorder_config updated keys=%s", sorted(incoming.keys()))
    return merged


async def get_recorder_config_persisted() -> dict[str, object]:
    """Authoritative current config, bypassing the TTL cache (for the API GET
    so the UI always reflects the persisted state)."""
    async with AsyncSessionLocal() as session:
        row = (await session.execute(select(AppSettings).limit(1))).scalar_one_or_none()
    config = _coerce_config(getattr(row, "recorder_config_json", None) if row is not None else None)
    prime_config_cache(config)
    return config
