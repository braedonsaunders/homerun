"""Per-trader daily-notional spend accumulator for the max_daily_spend_usd risk knob.

State lives in Redis with key ``trader_daily_spend:{trader_id}:{YYYY-MM-DD}``
(UTC). 48-hour TTL handles cross-midnight replays. Soft-fail when Redis is
unavailable: the gate returns ``allowed`` rather than blocking trading, but
emits a warning marker on the result so operators can investigate.

This is intentionally a thin counter, not a system of record. The trade-off:
if Redis is down or unhealthy, the per-trader spend cap is unenforceable for
that period — same trade-off the rest of the codebase makes for Redis-backed
state. Risk-conscious operators should monitor Redis health.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from redis.exceptions import RedisError

from services import redis_client
from utils.logger import get_logger

logger = get_logger("daily_spend_tracker")

_KEY_PREFIX = "trader_daily_spend"
_TTL_SECONDS = 48 * 3600  # Two days — covers cross-midnight reads + replays.


def _utc_date_str(now: datetime | None = None) -> str:
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d")


def _key(trader_id: str, date_str: str | None = None) -> str:
    date_part = date_str or _utc_date_str()
    return f"{_KEY_PREFIX}:{trader_id}:{date_part}"


async def get_daily_spend_usd(trader_id: str) -> float:
    """Return today's accumulated spend (USD) for ``trader_id``.

    Returns 0.0 when the key is missing or Redis is unavailable.
    """
    if not trader_id:
        return 0.0
    client = redis_client.get_client_or_none()
    if client is None:
        return 0.0
    try:
        raw = await client.get(redis_client.namespaced(_key(str(trader_id))))
    except (RedisError, asyncio.TimeoutError, OSError) as exc:
        logger.warning("daily_spend_tracker.get failed for trader=%s: %s", trader_id, exc)
        return 0.0
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


async def record_spend_usd(trader_id: str, notional_usd: float) -> float | None:
    """Atomically increment today's spend counter; returns the new total.

    Returns None when the increment was a no-op (Redis unhealthy, invalid
    inputs). Sets the 48h TTL on the first write of each UTC day.
    """
    if not trader_id:
        return None
    try:
        delta = float(notional_usd)
    except (TypeError, ValueError):
        return None
    if delta <= 0.0:
        return None
    client = redis_client.get_client_or_none()
    if client is None:
        return None
    key = redis_client.namespaced(_key(str(trader_id)))
    try:
        new_total = await client.incrbyfloat(key, delta)
        # EXPIRE is idempotent — safe to call every increment. Keeps the
        # TTL fresh in case the key already existed without a TTL set.
        await client.expire(key, _TTL_SECONDS)
    except (RedisError, asyncio.TimeoutError, OSError) as exc:
        logger.warning(
            "daily_spend_tracker.record failed for trader=%s delta=%.4f: %s",
            trader_id,
            delta,
            exc,
        )
        return None
    try:
        return float(new_total)
    except (TypeError, ValueError):
        return None


async def check_daily_spend_cap(
    trader_id: str,
    proposed_notional_usd: float,
    cap_usd: float | None,
) -> dict[str, Any]:
    """Pre-trade check: would this order push today's spend past the cap?

    Returns a dict suitable for embedding in a rejection payload:
    ``{
        "allowed": bool,
        "current_spend_usd": float,
        "projected_spend_usd": float,
        "cap_usd": float | None,
        "soft_fail": bool,  # True when Redis was unavailable
    }``

    A None or non-positive cap is a no-op (knob off) — always ``allowed``.
    Pure read; does NOT increment the counter. Call ``record_spend_usd``
    AFTER a successful fill to advance the counter.
    """
    result: dict[str, Any] = {
        "allowed": True,
        "current_spend_usd": 0.0,
        "projected_spend_usd": 0.0,
        "cap_usd": None,
        "soft_fail": False,
    }
    if cap_usd is None:
        return result
    try:
        cap_value = float(cap_usd)
    except (TypeError, ValueError):
        return result
    if cap_value <= 0.0:
        return result
    result["cap_usd"] = cap_value
    try:
        proposed = float(proposed_notional_usd)
    except (TypeError, ValueError):
        proposed = 0.0
    proposed = max(0.0, proposed)
    # When Redis is unhealthy we can't measure current spend; soft-fail
    # (allow) so we don't wedge trading on a transient Redis outage.
    if not redis_client.is_healthy():
        result["soft_fail"] = True
        result["projected_spend_usd"] = proposed
        return result
    current = await get_daily_spend_usd(trader_id)
    projected = current + proposed
    result["current_spend_usd"] = current
    result["projected_spend_usd"] = projected
    if projected > cap_value:
        result["allowed"] = False
    return result
