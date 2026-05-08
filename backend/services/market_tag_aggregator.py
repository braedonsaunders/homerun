"""Tag aggregator hook for the Polymarket / Kalshi ingest pipeline.

Every ingest cycle, ``record_tags_from_markets`` is called once with the
**raw** events + markets list (before any filter is applied) and upserts
every distinct tag string into ``market_tags_seen``. The
``Settings → Scanner`` tag chooser queries that table for tags seen in
the last 24 hours; without this hook the chooser would have no data and
the operator could not whitelist anything.

Failure isolation
-----------------
Every public coroutine in this module is responsible for committing
its own transaction. Callers wrap calls in ``try/except`` because tag
aggregation must never block the ingest hot path — a transient DB
failure here cannot be allowed to silently empty the trading universe.

The runtime kill-switch is ``settings.MARKET_TAG_AGGREGATOR_ENABLED``.
Retention is governed by ``settings.MARKET_TAG_RETENTION_DAYS`` and
the periodic prune is scheduled by ``settings.MARKET_TAG_PRUNE_INTERVAL_SECONDS``.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from utils.logger import get_logger


logger = get_logger(__name__)


def _normalize_tag(value: object) -> str | None:
    """Lowercase, strip, drop empties. Mirrors the chooser's contract."""
    if value is None:
        return None
    text_value = str(value).strip().lower()
    if not text_value:
        return None
    return text_value


def _collect_tags(events: Iterable[Any], markets: Iterable[Any]) -> set[str]:
    """Return the union of normalised tag strings across raw markets and events."""
    tags: set[str] = set()
    for market in markets:
        for raw in list(getattr(market, "tags", None) or []):
            normalised = _normalize_tag(raw)
            if normalised:
                tags.add(normalised)
    for event in events:
        for raw in list(getattr(event, "tags", None) or []):
            normalised = _normalize_tag(raw)
            if normalised:
                tags.add(normalised)
    return tags


async def record_tags_from_markets(
    session: AsyncSession,
    events: Iterable[Any],
    markets: Iterable[Any],
) -> int:
    """Upsert every distinct tag observed on the raw stream.

    Returns the number of distinct tags written/updated. The caller is
    responsible for catching exceptions — aggregator failure must never
    block ingest. The runtime kill-switch is
    ``settings.MARKET_TAG_AGGREGATOR_ENABLED``.

    The upsert uses a single ``INSERT ... ON CONFLICT DO UPDATE``
    statement with all tags bound as parameters; this keeps the call
    O(1) round trips even when the universe contains hundreds of
    distinct tags.
    """
    if not bool(getattr(settings, "MARKET_TAG_AGGREGATOR_ENABLED", True)):
        return 0

    tags = _collect_tags(events, markets)
    if not tags:
        return 0

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    payload = [{"tag": tag, "now": now} for tag in tags]

    stmt = text(
        """
        INSERT INTO market_tags_seen (tag, first_seen, last_seen, occurrences)
        VALUES (:tag, :now, :now, 1)
        ON CONFLICT (tag) DO UPDATE SET
            last_seen = EXCLUDED.last_seen,
            occurrences = market_tags_seen.occurrences + 1
        """
    )
    await session.execute(stmt, payload)
    await session.commit()
    return len(tags)


async def prune_stale_tags(
    session: AsyncSession,
    *,
    max_age_days: int | None = None,
) -> int:
    """Delete rows whose ``last_seen`` is older than ``max_age_days``.

    Defaults to ``settings.MARKET_TAG_RETENTION_DAYS``. Pass
    ``max_age_days <= 0`` (or set the config to 0) to disable pruning.

    Returns the number of rows deleted. Non-load-bearing — the table
    stays small even without this; the prune just keeps the chooser's
    bottom long-tail clean.
    """
    if max_age_days is None:
        max_age_days = int(getattr(settings, "MARKET_TAG_RETENTION_DAYS", 7) or 0)
    if max_age_days <= 0:
        return 0

    stmt = text(
        """
        DELETE FROM market_tags_seen
        WHERE last_seen < NOW() - make_interval(days => :days)
        """
    )
    result = await session.execute(stmt, {"days": int(max_age_days)})
    await session.commit()
    deleted = int(result.rowcount or 0)
    if deleted:
        logger.info(
            "market_tag_aggregator pruned %d stale rows (retention=%d days)",
            deleted,
            max_age_days,
        )
    return deleted
