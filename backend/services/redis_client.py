"""Async Redis client with lifecycle, health, and soft-fail semantics.

Single-pool entrypoint for every consumer in the app: streams (XADD/XREAD
with consumer groups), pub/sub channels, ephemeral caches.  Modeled on the
Postgres pool pattern in ``models/database.py`` — explicit ``start()`` /
``shutdown()`` lifecycle, thread-safe singleton, health probe, and
graceful-degradation accessors so a Redis outage **never** wedges a hot
path.

Soft-fail contract
------------------
Redis is treated as a *cache + bus*, not a system of record.  Every
producer must work when ``get_client_or_none()`` returns ``None``:

* publish-side: silently drop the message (consumers re-derive on next
  cycle from Postgres / REST seed)
* read-side: fall back to the in-memory singleton or DB query

The hot path (trader orchestrator, fast trader runtime) MUST NOT block on
Redis I/O — every helper here is bounded by ``REDIS_SOCKET_TIMEOUT_SECONDS``
and any failure flips the pool to "unhealthy" and disables further calls
until the background reconnect loop restores it.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import redis.asyncio as aioredis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import RedisError

from config import settings
from utils.logger import get_logger

logger = get_logger("redis_client")

_HEALTH_PROBE_KEY_SUFFIX = ":__health__"


@dataclass
class _RedisState:
    pool: Optional[ConnectionPool] = None
    client: Optional[aioredis.Redis] = None
    healthy: bool = False
    last_error: Optional[str] = None
    last_error_at: float = 0.0
    last_ok_at: float = 0.0
    started: bool = False
    health_task: Optional[asyncio.Task] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Consecutive probe failures.  We require ``REDIS_PROBE_FAILURE_THRESHOLD``
    # consecutive failures before flipping ``healthy`` to ``False`` so that
    # an isolated transport hiccup (notably the WSL2↔Windows NAT vSwitch
    # stalling for a second or two on this dev box) does not flap the
    # health state and disable every Redis-using caller.  Reset on the
    # next successful probe.
    consecutive_failures: int = 0


_state = _RedisState()


def _ns(key: str) -> str:
    """Apply the configured namespace prefix to a key."""
    prefix = (getattr(settings, "REDIS_NAMESPACE", "homerun") or "").strip()
    if not prefix:
        return key
    if key.startswith(prefix + ":"):
        return key
    return f"{prefix}:{key}"


def namespaced(key: str) -> str:
    """Public namespace helper for callers building stream/channel keys."""
    return _ns(key)


def is_enabled() -> bool:
    return bool(getattr(settings, "REDIS_ENABLED", True))


def is_healthy() -> bool:
    """True iff the pool is up AND the most recent probe succeeded."""
    return _state.started and _state.healthy and _state.client is not None


def get_client_or_none() -> Optional[aioredis.Redis]:
    """Return the live client, or None if Redis is disabled / unhealthy.

    Callers MUST handle the None case — see soft-fail contract at the top
    of this module.  Use this rather than reaching for ``_state`` directly
    so the healthy check and the disabled-flag check stay in one place.
    """
    if not is_enabled():
        return None
    if not _state.healthy:
        return None
    return _state.client


def get_last_error() -> Optional[str]:
    return _state.last_error


def status_snapshot() -> dict[str, Any]:
    """Lightweight status dict for health endpoints / GUI display."""
    now = time.time()
    return {
        "enabled": is_enabled(),
        "started": _state.started,
        "healthy": _state.healthy,
        "last_error": _state.last_error,
        "last_error_age_seconds": (
            round(now - _state.last_error_at, 3) if _state.last_error_at else None
        ),
        "last_ok_age_seconds": (
            round(now - _state.last_ok_at, 3) if _state.last_ok_at else None
        ),
        "url": _redact_url(getattr(settings, "REDIS_URL", "")),
    }


def _redact_url(url: str) -> str:
    if not url:
        return ""
    # redis://[:password@]host:port/db — strip a password if present.
    try:
        if "@" in url and "://" in url:
            scheme, rest = url.split("://", 1)
            creds, host = rest.split("@", 1)
            if ":" in creds:
                user, _ = creds.split(":", 1)
                creds = f"{user}:***"
            else:
                creds = "***"
            return f"{scheme}://{creds}@{host}"
    except Exception:
        pass
    return url


async def start() -> bool:
    """Initialize the connection pool and run the first health probe.

    Returns True if Redis is reachable.  Returns False (and leaves the
    pool installed in unhealthy state) if Redis is unreachable — the
    background health loop will keep retrying.  When ``REDIS_ENABLED`` is
    False this is a no-op that returns False.
    """
    async with _state.lock:
        if _state.started:
            return _state.healthy

        if not is_enabled():
            logger.info("Redis disabled by configuration; skipping pool start")
            _state.started = True
            return False

        url = (getattr(settings, "REDIS_URL", "") or "").strip()
        if not url:
            logger.warning("REDIS_URL is empty; not starting Redis pool")
            _state.started = True
            _state.last_error = "REDIS_URL empty"
            _state.last_error_at = time.time()
            return False

        try:
            _state.pool = ConnectionPool.from_url(
                url,
                max_connections=int(getattr(settings, "REDIS_MAX_CONNECTIONS", 64)),
                socket_timeout=float(getattr(settings, "REDIS_SOCKET_TIMEOUT_SECONDS", 1.5)),
                socket_connect_timeout=float(
                    getattr(settings, "REDIS_CONNECT_TIMEOUT_SECONDS", 2.0)
                ),
                health_check_interval=float(
                    getattr(settings, "REDIS_HEALTH_CHECK_INTERVAL_SECONDS", 15.0)
                ),
                decode_responses=True,
            )
            _state.client = aioredis.Redis(connection_pool=_state.pool)
        except Exception as exc:
            logger.error("Failed to construct Redis pool: %s", exc)
            _state.last_error = f"pool_init: {exc}"
            _state.last_error_at = time.time()
            _state.started = True
            return False

        ok = await _probe_once()
        _state.started = True

        if _state.health_task is None or _state.health_task.done():
            try:
                loop = asyncio.get_running_loop()
                _state.health_task = loop.create_task(
                    _health_probe_loop(),
                    name="redis_health_probe",
                )
            except RuntimeError:
                # No running loop — caller will start one later.  The
                # next start() invocation will pick up the probe loop.
                pass

        if ok:
            logger.info(
                "Redis pool ready at %s (max_conn=%d)",
                _redact_url(url),
                int(getattr(settings, "REDIS_MAX_CONNECTIONS", 64)),
            )
        else:
            logger.warning(
                "Redis pool started but unhealthy at %s: %s",
                _redact_url(url),
                _state.last_error,
            )
        return ok


async def shutdown() -> None:
    """Cancel the health loop and close the connection pool."""
    async with _state.lock:
        task = _state.health_task
        _state.health_task = None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        client = _state.client
        pool = _state.pool
        _state.client = None
        _state.pool = None
        _state.healthy = False
        _state.started = False

        if client is not None:
            try:
                await client.aclose()
            except Exception:
                pass
        if pool is not None:
            try:
                await pool.aclose()
            except Exception:
                pass


async def _probe_once() -> bool:
    """Run a single PING + SET/GET round-trip; update health state.

    SET and GET are pipelined into one round-trip so the probe makes 2
    RTTs (PING, then SET+GET) instead of 3.  On this dev box the probe
    travels Windows host → WSL2 NAT vSwitch → Redis-in-WSL, where each
    RTT can stall 8-50ms (sometimes seconds) under Hyper-V scheduler
    pressure; cutting the RTT count keeps the probe under budget.

    A single failure does NOT flip ``_state.healthy`` to ``False`` —
    we wait for ``REDIS_PROBE_FAILURE_THRESHOLD`` consecutive failures
    before declaring unhealthy, to absorb isolated transport hiccups.
    """
    client = _state.client
    if client is None:
        _state.healthy = False
        _state.consecutive_failures = 0
        return False

    probe_key = _ns(_HEALTH_PROBE_KEY_SUFFIX)
    # Probe timeout is intentionally generous (4s, vs the 1.5s socket
    # timeout for hot-path ops).  The probe runs every 15s on a periodic
    # background loop — when the worker's asyncio loop is briefly stalled
    # (e.g. a slow DB cycle), we don't want a flapping "health probe
    # failed" warning.  The probe is a *liveness* check, not a latency
    # gate; ops on the hot path use the tighter socket_timeout directly.
    timeout = max(4.0, float(getattr(settings, "REDIS_SOCKET_TIMEOUT_SECONDS", 1.5)) * 2.5)
    failure_threshold = max(
        1, int(getattr(settings, "REDIS_PROBE_FAILURE_THRESHOLD", 2))
    )
    try:
        async with asyncio.timeout(timeout + 0.5):
            pong = await client.ping()
            if pong is True or pong == b"PONG" or pong == "PONG":
                # Light read/write to confirm the chosen DB is accepting
                # writes (PING is answered even by a read-only replica).
                # Pipeline both operations into a single network round-
                # trip — the previous implementation awaited SET and GET
                # sequentially, which on a slow vSwitch stacked the
                # latency.  ``transaction=False`` skips MULTI/EXEC; we
                # don't need atomicity, just the round-trip merge.
                async with client.pipeline(transaction=False) as pipe:
                    pipe.set(probe_key, "1", ex=30)
                    pipe.get(probe_key)
                    await pipe.execute()
            else:
                raise RedisError(f"unexpected PING reply: {pong!r}")
    except (asyncio.TimeoutError, RedisError, OSError) as exc:
        # asyncio.TimeoutError() formats to an empty string — include the
        # type name so the operator sees "TimeoutError" rather than a
        # blank "Redis health probe failed: " in the log.
        exc_text = str(exc) or repr(exc) or type(exc).__name__
        _state.last_error = f"probe: {type(exc).__name__}: {exc_text}"
        _state.last_error_at = time.time()
        _state.consecutive_failures += 1
        # Only flip healthy=False once we've seen ``failure_threshold``
        # consecutive misses — absorbs single transport hiccups.
        if _state.consecutive_failures >= failure_threshold:
            was_healthy = _state.healthy
            _state.healthy = False
            if was_healthy:
                logger.warning(
                    "Redis health probe failed: %s: %s (timeout=%.2fs, %d consecutive)",
                    type(exc).__name__,
                    exc_text,
                    timeout,
                    _state.consecutive_failures,
                )
        return False
    except Exception as exc:  # defensive — never let probe crash the loop
        _state.healthy = False
        _state.consecutive_failures += 1
        exc_text = str(exc) or repr(exc) or type(exc).__name__
        _state.last_error = f"probe_unexpected: {type(exc).__name__}: {exc_text}"
        _state.last_error_at = time.time()
        logger.exception("Redis health probe raised unexpected error")
        return False

    became_healthy = not _state.healthy
    _state.healthy = True
    _state.consecutive_failures = 0
    _state.last_ok_at = time.time()
    if became_healthy:
        logger.info("Redis health restored")
    return True


async def _health_probe_loop() -> None:
    """Background liveness loop — flips state.healthy on transitions."""
    interval = float(getattr(settings, "REDIS_HEALTH_CHECK_INTERVAL_SECONDS", 15.0))
    interval = max(interval, 1.0)
    try:
        while True:
            await asyncio.sleep(interval)
            if not _state.started or _state.client is None:
                return
            await _probe_once()
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Redis health probe loop crashed; will not auto-restart")


# ---------------------------------------------------------------------------
# Convenience helpers — every helper is soft-fail.
# ---------------------------------------------------------------------------


async def safe_set(key: str, value: str, ex: Optional[int] = None) -> bool:
    """SET with TTL; returns True on success, False on any failure (soft fail)."""
    client = get_client_or_none()
    if client is None:
        return False
    try:
        await client.set(_ns(key), value, ex=ex)
        return True
    except (RedisError, asyncio.TimeoutError, OSError) as exc:
        _record_op_failure("set", exc)
        return False


async def safe_get(key: str) -> Optional[str]:
    """GET; returns None if Redis is unavailable or key missing."""
    client = get_client_or_none()
    if client is None:
        return None
    try:
        return await client.get(_ns(key))
    except (RedisError, asyncio.TimeoutError, OSError) as exc:
        _record_op_failure("get", exc)
        return None


async def safe_publish(channel: str, message: str) -> bool:
    """PUBLISH; returns True on success, False on any failure."""
    client = get_client_or_none()
    if client is None:
        return False
    try:
        await client.publish(_ns(channel), message)
        return True
    except (RedisError, asyncio.TimeoutError, OSError) as exc:
        _record_op_failure("publish", exc)
        return False


async def safe_xadd(
    stream: str,
    fields: dict[str, Any],
    *,
    maxlen: Optional[int] = None,
    approximate: bool = True,
) -> Optional[str]:
    """XADD with optional MAXLEN ~; returns the entry ID or None on failure."""
    client = get_client_or_none()
    if client is None:
        return None
    cap = maxlen if maxlen is not None else int(
        getattr(settings, "REDIS_STREAM_MAXLEN_DEFAULT", 10000)
    )
    try:
        # redis-py expects str/bytes/numbers in field values; coerce safely.
        coerced = {k: _coerce_field(v) for k, v in fields.items()}
        return await client.xadd(
            _ns(stream),
            coerced,
            maxlen=cap if cap and cap > 0 else None,
            approximate=approximate,
        )
    except (RedisError, asyncio.TimeoutError, OSError) as exc:
        _record_op_failure("xadd", exc)
        return None


async def safe_xrevrange(
    stream: str,
    *,
    count: int = 500,
) -> list[tuple[str, dict[str, Any]]]:
    """XREVRANGE (newest-first) over a stream; returns ``[(id, fields), ...]``.

    Soft-fails to an empty list on any error or when Redis is
    unavailable — callers treat "no history" as a benign empty result.
    Field keys/values are decoded to ``str`` regardless of the client's
    ``decode_responses`` setting.
    """
    client = get_client_or_none()
    if client is None:
        return []
    try:
        raw = await client.xrevrange(_ns(stream), count=max(1, int(count)))
    except (RedisError, asyncio.TimeoutError, OSError) as exc:
        _record_op_failure("xrevrange", exc)
        return []
    out: list[tuple[str, dict[str, Any]]] = []
    for entry_id, fields in raw or []:
        sid = entry_id.decode() if isinstance(entry_id, bytes) else str(entry_id)
        decoded: dict[str, Any] = {}
        for k, v in (fields or {}).items():
            key = k.decode() if isinstance(k, bytes) else str(k)
            val = v.decode() if isinstance(v, bytes) else v
            decoded[key] = val
        out.append((sid, decoded))
    return out


def _coerce_field(value: Any) -> Any:
    if isinstance(value, (str, bytes, int, float)):
        return value
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _record_op_failure(op: str, exc: BaseException) -> None:
    _state.last_error = f"{op}: {exc}"
    _state.last_error_at = time.time()
    # We don't flip healthy=False here; the dedicated probe owns transitions.
    # A single op failure can be transient (e.g. command-level timeout).
    logger.debug("Redis %s failed: %s", op, exc)
