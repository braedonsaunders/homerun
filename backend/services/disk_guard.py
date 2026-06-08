"""Free-disk guard for the parquet recorders.

The per-topic / size caps (``book_max_bytes``,
``recorded_event_bus_global_max_bytes``) bound the APP's own on-disk footprint.
They do NOT protect against the DISK itself filling: on a host whose drive is
already near-full from unrelated data, the recorder's normal write/prune cycle
can still drive total free space to 0 between prune passes and crash the OS.
(Observed: the recorder wrote the host to 0 bytes free and crashed Windows.)

This guard closes that gap.  Before each flush the writer asks whether total
free disk on the parquet root has dropped below ``disk_guard_min_free_gb``.
If so it PAUSES writes (the caller drops the in-memory batch and force-prunes)
until space recovers.  It is independent of the size caps and never deletes
data on its own — it only signals; the caller decides how to shed.

Settings are read live (short-TTL cache via
``recording_control.get_recorder_config`` — no DB hit on the hot path):
  * ``disk_guard_enabled``      — master toggle (default True)
  * ``disk_guard_min_free_gb``  — headroom below which writes pause (default 10)

The last time the guard tripped is published to Redis (best-effort, namespaced
``disk_guard:last_trip``) so the operator UI can show "when it last kicked in"
across processes.  Fail-OPEN everywhere: any guard error returns "not blocked"
so a transient fault can never silently halt recording.
"""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from typing import Any, Optional

from services.recording_control import get_recorder_config
from utils.logger import get_logger

logger = get_logger("disk_guard")

_TRIP_REDIS_KEY = "disk_guard:last_trip"
_LOG_THROTTLE_SECONDS = 30.0
_TRIP_PUBLISH_THROTTLE_SECONDS = 15.0

_last_log_mono: float = 0.0
_last_trip_publish_mono: float = 0.0
_last_trip_local: Optional[dict[str, Any]] = None


def _free_gb(root: Any) -> float:
    return shutil.disk_usage(str(root)).free / (1024.0 ** 3)


async def _read_config() -> tuple[bool, float]:
    cfg = await get_recorder_config()
    enabled = bool(cfg.get("disk_guard_enabled", True))
    min_gb = float(cfg.get("disk_guard_min_free_gb", 10) or 0)
    return enabled, min_gb


async def evaluate(root: Any) -> tuple[bool, float, float]:
    """Return ``(blocked, free_gb, min_gb)`` for the parquet root.

    ``blocked`` is True only when the guard is enabled AND total free disk has
    dropped below the configured headroom.  Fail-open: any error -> not blocked.
    """
    try:
        enabled, min_gb = await _read_config()
    except Exception:
        return (False, -1.0, 0.0)
    try:
        free_gb = _free_gb(root)
    except Exception:
        return (False, -1.0, min_gb)
    if not enabled or min_gb <= 0:
        return (False, free_gb, min_gb)
    blocked = free_gb < min_gb
    if blocked:
        await _on_trip(free_gb, min_gb)
    return (blocked, free_gb, min_gb)


async def _on_trip(free_gb: float, min_gb: float) -> None:
    global _last_log_mono, _last_trip_publish_mono, _last_trip_local
    now_mono = time.monotonic()
    payload = {
        "at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "free_gb": round(free_gb, 2),
        "min_gb": round(min_gb, 2),
    }
    _last_trip_local = payload
    if now_mono - _last_log_mono >= _LOG_THROTTLE_SECONDS:
        _last_log_mono = now_mono
        logger.warning(
            "DISK GUARD tripped — pausing parquet writes + force-pruning",
            free_gb=round(free_gb, 2),
            min_free_gb=round(min_gb, 2),
        )
    if now_mono - _last_trip_publish_mono >= _TRIP_PUBLISH_THROTTLE_SECONDS:
        _last_trip_publish_mono = now_mono
        try:
            from services import redis_client

            client = redis_client.get_client_or_none()
            if client is not None:
                await client.set(redis_client.namespaced(_TRIP_REDIS_KEY), json.dumps(payload))
        except Exception:
            pass


async def get_last_trip() -> Optional[dict[str, Any]]:
    """Most recent trip record (cross-process via Redis; falls back to the
    in-process value).  None if the guard has never tripped."""
    try:
        from services import redis_client

        client = redis_client.get_client_or_none()
        if client is not None:
            raw = await client.get(redis_client.namespaced(_TRIP_REDIS_KEY))
            if raw:
                if isinstance(raw, (bytes, bytearray)):
                    raw = raw.decode("utf-8", "replace")
                return json.loads(raw)
    except Exception:
        pass
    return _last_trip_local


async def status(root: Any) -> dict[str, Any]:
    """Operator status for the API/UI: enabled, threshold, live free space,
    whether the guard is active right now, and when it last tripped."""
    try:
        enabled, min_gb = await _read_config()
    except Exception:
        enabled, min_gb = True, 10.0
    try:
        free_gb = round(_free_gb(root), 2)
    except Exception:
        free_gb = -1.0
    active = bool(enabled and min_gb > 0 and free_gb >= 0 and free_gb < min_gb)
    return {
        "disk_guard_enabled": enabled,
        "disk_guard_min_free_gb": int(min_gb),
        "free_gb": free_gb,
        "active": active,
        "last_trip": await get_last_trip(),
    }
