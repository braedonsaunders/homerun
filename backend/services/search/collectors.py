"""Per-entity-type collectors for the global search index.

Each collector is an async function ``(session) -> list[dict]`` that
returns one dict per searchable row using a normalized shape::

    {
        "entity_type": "market",
        "entity_id":   "<stable id>",
        "title":       "...",
        "subtitle":    "...",   # optional
        "body":        "...",   # optional, longer prose
        "category":    "...",   # optional
        "tags":        [...],
        "metadata":    {...},   # type-specific extras for the UI
        "liquidity":   123.0,   # optional, used in ranking
        "volume":      456.0,   # optional
        "recency":     <datetime utc-aware>,  # optional
    }

The :func:`reindex_one` driver upserts every row a collector returns
into ``search_index`` and deletes any rows of that ``entity_type``
that the collector didn't yield this run — so collector output is the
source of truth on every refresh.

Adding a new entity type is intentionally one-stop: write a collector
function, add it to ``COLLECTORS``, and the worker / API / UI all
pick it up automatically.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    DataSource,
    DetectedAnomaly,
    NewsArticleCache,
    OpportunityHistory,
    ResearchSession,
    Strategy,
    Trader,
    TrackedWallet,
)
from services import shared_state
from utils.logger import get_logger
from utils.utcnow import utcnow

logger = get_logger("search.collectors")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate(value: Any, length: int) -> str | None:
    if value is None:
        return None
    text_value = str(value).strip()
    if not text_value:
        return None
    if len(text_value) <= length:
        return text_value
    return text_value[: length - 1] + "…"


def _to_naive_utc(dt: Any) -> datetime | None:
    """Coerce arbitrary datetime values to naive UTC for the DB column."""
    if dt is None:
        return None
    if not isinstance(dt, datetime):
        return None
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_tags(values: list[Any]) -> list[str]:
    """Drop empty / None / duplicate tag entries while preserving order."""
    out: list[str] = []
    seen: set[str] = set()
    for v in values or []:
        if v is None:
            continue
        s = str(v).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


# Process-level cache for the opportunity snapshot load.  The market /
# event / category collectors all read from the same source, so we
# cache the result for the duration of a single reindex pass (which
# completes in well under 5 s).  A subsequent worker tick re-loads.
_OPP_CACHE: dict[str, Any] = {"data": None, "at": 0.0}
_OPP_CACHE_TTL_SECONDS = 5.0


async def _load_opps(session: AsyncSession) -> list[Any]:
    """Load opportunities for search-index population.

    The search collectors only need the SNAPSHOT-time fields baked into
    the opportunity row (``markets[*].yes_price``, ``liquidity``,
    ``question``, ``category``, …).  ``shared_state.
    get_opportunities_from_db`` does much more than that: live price
    refresh against Polymarket HTTP, ``get_market_tradability_map``
    (its own DB transaction — caught at 4.88s in the production log),
    and ``attach_price_history_to_opportunities`` (per-market HTTP
    fan-out).  None of that work changes the search index — the
    collectors read from the static market dicts only.

    Running the full enrichment from inside the search reindex was
    the dominant contributor to the ``Search reindex hard-timeout
    for market after 25.0s; deferring to next cycle`` warning that
    fired 91× in the latest production capture.  The search-index
    reindex tick is every 120s; one cycle's worth of stale prices in
    a metadata field used for ranking is acceptable.  The live UI
    that needs fresh prices reads from the opportunity API directly,
    not from the search index.

    This loader therefore reads ONLY from the scanner / traders
    snapshots and skips every enrichment hop.  Net effect: the
    ``market`` collector's wall-clock drops from "regularly >25s" to
    a single DB read of two snapshot rows.
    """
    now = time.monotonic()
    cached = _OPP_CACHE.get("data")
    if cached is not None and (now - float(_OPP_CACHE.get("at") or 0.0)) < _OPP_CACHE_TTL_SECONDS:
        return cached
    try:
        market_opps, _ = await shared_state.read_scanner_snapshot(session)
        trader_opps, _ = await shared_state.read_traders_snapshot(session)
        data = list(market_opps) + list(trader_opps)
    except Exception as exc:
        logger.warning("opportunity snapshot load failed: %s", exc)
        data = []
    _OPP_CACHE["data"] = data
    _OPP_CACHE["at"] = now
    return data


# ---------------------------------------------------------------------------
# Collectors — one per entity_type
# ---------------------------------------------------------------------------


async def collect_markets(session: AsyncSession) -> list[dict[str, Any]]:
    """Markets — extracted from the live opportunity snapshots.

    Markets are not a first-class table (they live as JSON inside
    Polymarket / Kalshi snapshots), so we materialize one search-index
    row per unique market id we see in the current opportunity set.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    opportunities = await _load_opps(session)

    for opp in opportunities:
        event_title = getattr(opp, "event_title", None)
        category = getattr(opp, "category", None)
        for market in getattr(opp, "markets", None) or []:
            if not isinstance(market, dict):
                continue
            mid = str(market.get("id") or market.get("condition_id") or "").strip()
            if not mid or mid in seen:
                continue
            seen.add(mid)
            question = market.get("question") or market.get("title") or ""
            if not question:
                continue
            yes_price = _as_float(market.get("yes_price"))
            no_price = _as_float(market.get("no_price"))
            liquidity = _as_float(market.get("liquidity"))
            volume = _as_float(market.get("volume"))
            slug = market.get("slug") or market.get("event_slug")
            tags = _clean_tags([category, market.get("outcome")])
            out.append(
                {
                    "entity_type": "market",
                    "entity_id": mid,
                    "title": _truncate(question, 500) or question,
                    "subtitle": _truncate(event_title, 240),
                    "body": _truncate(market.get("description"), 4000),
                    "category": _truncate(category, 120),
                    "tags": tags,
                    "metadata": {
                        "market_id": mid,
                        "question": question,
                        "yes_price": yes_price,
                        "no_price": no_price,
                        "liquidity": liquidity,
                        "volume": volume,
                        "event_title": event_title,
                        "category": category,
                        "slug": slug,
                    },
                    "liquidity": liquidity,
                    "volume": volume,
                    "recency": _to_naive_utc(getattr(opp, "last_detected_at", None)),
                }
            )
    return out


async def collect_events(session: AsyncSession) -> list[dict[str, Any]]:
    """Distinct events derived from the opportunity snapshot."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    opportunities = await _load_opps(session)

    for opp in opportunities:
        event_title = getattr(opp, "event_title", None)
        if not event_title:
            continue
        event_id = (
            getattr(opp, "event_id", None)
            or getattr(opp, "event_slug", None)
            or event_title
        )
        key = str(event_id).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        category = getattr(opp, "category", None)
        # Accumulate liquidity across constituent markets so popular
        # events float to the top.
        total_liquidity = 0.0
        for market in getattr(opp, "markets", None) or []:
            if isinstance(market, dict):
                total_liquidity += _as_float(market.get("liquidity")) or 0.0
        market_count = len(list(getattr(opp, "markets", None) or []))
        out.append(
            {
                "entity_type": "event",
                "entity_id": key,
                "title": _truncate(event_title, 500) or event_title,
                "subtitle": _truncate(category, 120),
                "body": None,
                "category": _truncate(category, 120),
                "tags": _clean_tags([category]),
                "metadata": {
                    "event_id": key,
                    "event_title": event_title,
                    "category": category,
                    "market_count": market_count,
                },
                "liquidity": total_liquidity if total_liquidity > 0 else None,
                "volume": None,
                "recency": _to_naive_utc(getattr(opp, "last_detected_at", None)),
            }
        )
    return out


async def collect_categories(session: AsyncSession) -> list[dict[str, Any]]:
    """Distinct categories — quick jumps to the category-filtered view."""
    out: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    opportunities = await _load_opps(session)
    for opp in opportunities:
        category = getattr(opp, "category", None)
        if category:
            key = str(category).strip()
            if key:
                counts[key] = counts.get(key, 0) + 1
    for category, n in counts.items():
        out.append(
            {
                "entity_type": "category",
                "entity_id": category,
                "title": category,
                "subtitle": f"{n} active opportunities",
                "body": None,
                "category": category,
                "tags": [],
                "metadata": {"category": category, "opportunity_count": n},
                "liquidity": None,
                "volume": None,
                "recency": None,
            }
        )
    return out


async def collect_strategies(session: AsyncSession) -> list[dict[str, Any]]:
    rs = await session.execute(select(Strategy))
    out: list[dict[str, Any]] = []
    for s in rs.scalars():
        out.append(
            {
                "entity_type": "strategy",
                "entity_id": s.id,
                "title": s.name or s.slug,
                "subtitle": _truncate(s.description, 240),
                "body": _truncate(s.description, 4000),
                "category": s.source_key,
                "tags": _clean_tags(
                    [
                        *(s.aliases or []),
                        s.source_key,
                        s.status,
                        "system" if s.is_system else "user",
                    ]
                ),
                "metadata": {
                    "strategy_id": s.id,
                    "slug": s.slug,
                    "source_key": s.source_key,
                    "enabled": bool(s.enabled),
                    "status": s.status,
                    "is_system": bool(s.is_system),
                },
                "liquidity": None,
                "volume": None,
                "recency": _to_naive_utc(s.updated_at),
            }
        )
    return out


async def collect_data_sources(session: AsyncSession) -> list[dict[str, Any]]:
    rs = await session.execute(select(DataSource))
    out: list[dict[str, Any]] = []
    for d in rs.scalars():
        out.append(
            {
                "entity_type": "data_source",
                "entity_id": d.id,
                "title": d.name or d.slug,
                "subtitle": _truncate(d.description, 240),
                "body": _truncate(d.description, 4000),
                "category": d.source_kind,
                "tags": _clean_tags([d.source_key, d.source_kind, d.status]),
                "metadata": {
                    "data_source_id": d.id,
                    "slug": d.slug,
                    "source_key": d.source_key,
                    "source_kind": d.source_kind,
                    "enabled": bool(d.enabled),
                    "status": d.status,
                },
                "liquidity": None,
                "volume": None,
                "recency": _to_naive_utc(d.updated_at),
            }
        )
    return out


async def collect_traders(session: AsyncSession) -> list[dict[str, Any]]:
    rs = await session.execute(select(Trader))
    out: list[dict[str, Any]] = []
    for t in rs.scalars():
        out.append(
            {
                "entity_type": "trader",
                "entity_id": t.id,
                "title": t.name,
                "subtitle": _truncate(t.description, 240),
                "body": _truncate(t.description, 4000),
                "category": t.mode,
                "tags": _clean_tags([t.mode, t.latency_class]),
                "metadata": {
                    "trader_id": t.id,
                    "name": t.name,
                    "mode": t.mode,
                    "is_enabled": bool(t.is_enabled),
                    "is_paused": bool(t.is_paused),
                },
                "liquidity": None,
                "volume": None,
                "recency": _to_naive_utc(t.updated_at or t.last_run_at),
            }
        )
    return out


async def collect_wallets(session: AsyncSession) -> list[dict[str, Any]]:
    rs = await session.execute(select(TrackedWallet))
    out: list[dict[str, Any]] = []
    for w in rs.scalars():
        title = w.label or w.address
        # Subtitle gives the short-form address when label is the title,
        # or the full address when label is empty.
        subtitle = w.address if w.label else None
        # Volume signal: total trades count, used to push prolific
        # wallets up the ranking.
        volume = float(w.total_trades or 0)
        out.append(
            {
                "entity_type": "wallet",
                "entity_id": w.address,
                "title": _truncate(title, 200) or title,
                "subtitle": _truncate(subtitle, 200),
                "body": None,
                "category": "tracked",
                "tags": _clean_tags(["flagged"] if w.is_flagged else []),
                "metadata": {
                    "address": w.address,
                    "label": w.label,
                    "total_trades": int(w.total_trades or 0),
                    "win_rate": _as_float(w.win_rate),
                    "total_pnl": _as_float(w.total_pnl),
                    "is_flagged": bool(w.is_flagged),
                },
                "liquidity": None,
                "volume": volume if volume > 0 else None,
                "recency": _to_naive_utc(w.last_trade_at or w.added_at),
            }
        )
    return out


# Recent N news articles, alerts, research sessions — older content
# stays in the database but searching the entire history hurts both
# ranking quality (drowns recent signal in noise) and index size.
NEWS_RECENT_LIMIT = 5_000
ALERTS_RECENT_LIMIT = 2_000
RESEARCH_RECENT_LIMIT = 1_000
OPPORTUNITY_HISTORY_LIMIT = 2_000


async def collect_news(session: AsyncSession) -> list[dict[str, Any]]:
    rs = await session.execute(
        select(NewsArticleCache)
        .order_by(NewsArticleCache.fetched_at.desc())
        .limit(NEWS_RECENT_LIMIT)
    )
    out: list[dict[str, Any]] = []
    for n in rs.scalars():
        out.append(
            {
                "entity_type": "news",
                "entity_id": n.article_id,
                "title": _truncate(n.title, 500) or n.title,
                "subtitle": _truncate(n.source or n.feed_source, 200),
                "body": _truncate(n.summary, 4000),
                "category": n.category,
                "tags": _clean_tags([n.source, n.feed_source]),
                "metadata": {
                    "article_id": n.article_id,
                    "url": n.url,
                    "source": n.source,
                    "feed_source": n.feed_source,
                    "category": n.category,
                    "published": n.published.isoformat() if n.published else None,
                },
                "liquidity": None,
                "volume": None,
                "recency": _to_naive_utc(n.published or n.fetched_at),
            }
        )
    return out


async def collect_alerts(session: AsyncSession) -> list[dict[str, Any]]:
    rs = await session.execute(
        select(DetectedAnomaly)
        .order_by(DetectedAnomaly.detected_at.desc())
        .limit(ALERTS_RECENT_LIMIT)
    )
    out: list[dict[str, Any]] = []
    severity_volume = {"critical": 1.0, "high": 0.7, "medium": 0.4, "low": 0.1}
    for a in rs.scalars():
        title = a.anomaly_type or "Anomaly"
        out.append(
            {
                "entity_type": "alert",
                "entity_id": a.id,
                "title": title,
                "subtitle": _truncate(a.description, 240),
                "body": _truncate(a.description, 4000),
                "category": a.severity,
                "tags": _clean_tags(
                    [a.severity, "resolved" if a.is_resolved else "open"]
                ),
                "metadata": {
                    "alert_id": a.id,
                    "anomaly_type": a.anomaly_type,
                    "severity": a.severity,
                    "is_resolved": bool(a.is_resolved),
                    "wallet_address": a.wallet_address,
                    "market_id": a.market_id,
                },
                "liquidity": None,
                "volume": severity_volume.get(str(a.severity or "").lower()),
                "recency": _to_naive_utc(a.detected_at),
            }
        )
    return out


async def collect_research(session: AsyncSession) -> list[dict[str, Any]]:
    rs = await session.execute(
        select(ResearchSession)
        .order_by(ResearchSession.started_at.desc())
        .limit(RESEARCH_RECENT_LIMIT)
    )
    out: list[dict[str, Any]] = []
    for r in rs.scalars():
        out.append(
            {
                "entity_type": "research",
                "entity_id": r.id,
                "title": _truncate(r.query, 500) or "Research session",
                "subtitle": r.session_type,
                "body": None,
                "category": r.session_type,
                "tags": _clean_tags([r.session_type, r.status]),
                "metadata": {
                    "research_id": r.id,
                    "session_type": r.session_type,
                    "status": r.status,
                    "opportunity_id": r.opportunity_id,
                    "market_id": r.market_id,
                },
                "liquidity": None,
                "volume": None,
                "recency": _to_naive_utc(r.completed_at or r.started_at),
            }
        )
    return out


async def collect_opportunities_history(session: AsyncSession) -> list[dict[str, Any]]:
    """Recent OpportunityHistory rows — historical record of what we caught."""
    rs = await session.execute(
        select(OpportunityHistory)
        .order_by(OpportunityHistory.detected_at.desc())
        .limit(OPPORTUNITY_HISTORY_LIMIT)
    )
    out: list[dict[str, Any]] = []
    for o in rs.scalars():
        out.append(
            {
                "entity_type": "opportunity",
                "entity_id": o.id,
                "title": _truncate(o.title, 500) or "Opportunity",
                "subtitle": o.strategy_type,
                "body": None,
                "category": o.strategy_type,
                "tags": _clean_tags([o.strategy_type]),
                "metadata": {
                    "opportunity_id": o.id,
                    "strategy_type": o.strategy_type,
                    "expected_roi": _as_float(o.expected_roi),
                    "risk_score": _as_float(o.risk_score),
                    "was_profitable": o.was_profitable,
                    "actual_roi": _as_float(o.actual_roi),
                },
                "liquidity": None,
                "volume": _as_float(o.total_cost),
                "recency": _to_naive_utc(o.detected_at),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Driver: upsert + sweep-delete per entity type
# ---------------------------------------------------------------------------


CollectorFn = Callable[[AsyncSession], Awaitable[list[dict[str, Any]]]]

COLLECTORS: dict[str, CollectorFn] = {
    "market": collect_markets,
    "event": collect_events,
    "category": collect_categories,
    "strategy": collect_strategies,
    "data_source": collect_data_sources,
    "trader": collect_traders,
    "wallet": collect_wallets,
    "news": collect_news,
    "alert": collect_alerts,
    "research": collect_research,
    "opportunity": collect_opportunities_history,
}


# Single statement that upserts one row.  We rely on the Postgres
# ``GENERATED ALWAYS AS STORED`` tsvector to keep ``tsv`` in sync.
_UPSERT_SQL = text(
    """
    INSERT INTO search_index
        (entity_type, entity_id, title, subtitle, body, category,
         tags, metadata_json, liquidity, volume, recency, updated_at)
    VALUES
        (:entity_type, :entity_id, :title, :subtitle, :body, :category,
         CAST(:tags AS jsonb), CAST(:metadata_json AS jsonb),
         :liquidity, :volume, :recency, :updated_at)
    ON CONFLICT (entity_type, entity_id) DO UPDATE SET
        title         = EXCLUDED.title,
        subtitle      = EXCLUDED.subtitle,
        body          = EXCLUDED.body,
        category      = EXCLUDED.category,
        tags          = EXCLUDED.tags,
        metadata_json = EXCLUDED.metadata_json,
        liquidity     = EXCLUDED.liquidity,
        volume        = EXCLUDED.volume,
        recency       = EXCLUDED.recency,
        updated_at    = EXCLUDED.updated_at
    """
)


_DELETE_STALE_SQL = text(
    """
    DELETE FROM search_index
    WHERE entity_type = :entity_type
      AND updated_at < :cutoff
    """
)


async def reindex_one(
    session: AsyncSession, entity_type: str
) -> dict[str, Any]:
    """Run one collector and reconcile its rows in ``search_index``.

    Returns a stats dict describing the run.
    """
    collector = COLLECTORS.get(entity_type)
    if collector is None:
        raise KeyError(f"Unknown entity_type {entity_type!r}")

    started = utcnow()
    cutoff_naive = started.replace(tzinfo=None)

    try:
        rows = await collector(session)
    except Exception as exc:
        logger.exception("collector failed: entity_type=%s err=%s", entity_type, exc)
        return {
            "entity_type": entity_type,
            "ok": False,
            "error": str(exc),
            "upserted": 0,
            "deleted": 0,
        }

    # Batch upserts via SQLAlchemy executemany.  Per-row round-trips
    # were the dominant cost when the opportunity collector returned
    # 2k rows (~70 s of network latency for what should be a sub-
    # second insert).  Chunking keeps memory bounded while amortizing
    # the protocol overhead.
    params_list: list[dict[str, Any]] = [
        {
            "entity_type": entity_type,
            "entity_id": str(row["entity_id"])[:255],
            "title": row.get("title") or "",
            "subtitle": row.get("subtitle"),
            "body": row.get("body"),
            "category": row.get("category"),
            "tags": json.dumps(row.get("tags") or []),
            "metadata_json": json.dumps(row.get("metadata") or {}),
            "liquidity": row.get("liquidity"),
            "volume": row.get("volume"),
            "recency": _to_naive_utc(row.get("recency")),
            "updated_at": cutoff_naive,
        }
        for row in rows
    ]

    UPSERT_CHUNK = 500
    upserted = 0
    for i in range(0, len(params_list), UPSERT_CHUNK):
        chunk = params_list[i : i + UPSERT_CHUNK]
        try:
            await session.execute(_UPSERT_SQL, chunk)
            upserted += len(chunk)
        except Exception as exc:
            # Lock-timeout on the chunk → roll back and retry row-by-
            # row so a single bad payload doesn't cost us the whole
            # batch.  Skipped rows are logged.
            logger.warning(
                "search upsert chunk failed (%d rows) — falling back row-by-row: %s",
                len(chunk),
                exc,
            )
            try:
                await session.rollback()
            except Exception:
                pass
            for params in chunk:
                try:
                    await session.execute(_UPSERT_SQL, params)
                    upserted += 1
                except Exception as row_exc:
                    logger.warning(
                        "search upsert failed: entity_type=%s id=%s err=%s",
                        entity_type,
                        params.get("entity_id"),
                        row_exc,
                    )

    # Sweep-delete: anything for this type whose updated_at is older
    # than this run's cutoff has clearly disappeared from the source.
    deleted = 0
    try:
        result = await session.execute(
            _DELETE_STALE_SQL,
            {"entity_type": entity_type, "cutoff": cutoff_naive},
        )
        deleted = int(result.rowcount or 0)
    except Exception as exc:
        logger.warning("search sweep-delete failed: entity_type=%s err=%s", entity_type, exc)

    await session.commit()

    duration_ms = (utcnow() - started).total_seconds() * 1000.0
    return {
        "entity_type": entity_type,
        "ok": True,
        "upserted": upserted,
        "deleted": deleted,
        "duration_ms": round(duration_ms, 1),
    }


async def reindex_all(session: AsyncSession) -> dict[str, Any]:
    """Reindex every registered entity type."""
    started = utcnow()
    per_type: list[dict[str, Any]] = []
    for entity_type in COLLECTORS:
        per_type.append(await reindex_one(session, entity_type))
    duration_ms = (utcnow() - started).total_seconds() * 1000.0
    total_upserted = sum(int(r.get("upserted") or 0) for r in per_type)
    total_deleted = sum(int(r.get("deleted") or 0) for r in per_type)
    return {
        "ok": all(r.get("ok") for r in per_type),
        "per_type": per_type,
        "total_upserted": total_upserted,
        "total_deleted": total_deleted,
        "duration_ms": round(duration_ms, 1),
        "started_at": started.isoformat(),
    }
