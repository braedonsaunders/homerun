"""Ranked global-search query engine.

The single public entry point :func:`run_query` runs one SQL statement
that combines:

* full-text matching via the generated ``tsvector`` (``@@`` with
  ``websearch_to_tsquery`` so users can paste quoted phrases / OR / -)
* fuzzy matching via ``pg_trgm`` similarity on ``title`` (catches
  typos and partial words the FTS lexer can't normalize)
* finance-aware ranking: liquidity + volume + recency blended into a
  single composite score
* server-side highlighting via ``ts_headline`` so the UI gets a
  pre-rendered ``<mark>match</mark>`` snippet without a second round trip

A single call returns up to ``limit`` rows ranked by the composite
score, and (optionally) restricted to a subset of entity types.
"""

from __future__ import annotations

import html
import time
from typing import Any, Iterable, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from utils.logger import get_logger
from utils.utcnow import utcnow

logger = get_logger("search.service")


# Sentinel tokens for ``ts_headline``.  We can't ask Postgres to emit
# literal ``<mark>`` because that defeats HTML escaping on the way out
# (a market titled ``Will <script>alert(1)</script>?`` would XSS the
# command bar).  Instead we tell ts_headline to wrap matches in these
# control-char sentinels, HTML-escape the snippet in Python, then swap
# the sentinels back to real <mark> tags.  The 0x01 byte is invalid in
# UTF-8 text columns, so it cannot appear in user content.
_HEADLINE_OPEN = "\x01HRMK\x01"
_HEADLINE_CLOSE = "\x01EHMK\x01"


# Composite ranking weights.  Tuned for a financial workflow where the
# user almost always wants the *liquid, recent, name-matching* result
# first.  Documented here rather than hidden in the SQL string so the
# tradeoffs are reviewable.
RANK_WEIGHT_FTS = 1.0          # ts_rank_cd of the generated tsvector
RANK_WEIGHT_TRIGRAM = 0.6      # similarity(title, q) — catches typos
RANK_WEIGHT_LIQUIDITY = 0.35   # log-bounded liquidity boost
RANK_WEIGHT_VOLUME = 0.20      # log-bounded volume boost
RANK_WEIGHT_RECENCY_DAY = 0.25 # recency boost: <24h
RANK_WEIGHT_RECENCY_WEEK = 0.10 # recency boost: <7d

# Trigram similarity floor.  ``%`` operator defaults to 0.3
# (configurable per-session via ``SET pg_trgm.similarity_threshold``);
# we set it explicitly to keep behavior portable across environments.
TRIGRAM_SIMILARITY_THRESHOLD = 0.25

# Max results returned even if the caller asks for more — guard
# against runaway requests.
ABSOLUTE_LIMIT = 100


_HEADLINE_OPTS = (
    f"StartSel={_HEADLINE_OPEN}, StopSel={_HEADLINE_CLOSE}, "
    "MaxFragments=2, MaxWords=14, MinWords=3, ShortWord=2, HighlightAll=false"
)


_SEARCH_SQL = text(
    f"""
    WITH q AS (
        SELECT
            websearch_to_tsquery('english', :raw_query) AS tsq,
            :raw_query                                  AS raw
    )
    SELECT
        si.entity_type,
        si.entity_id,
        si.title,
        si.subtitle,
        si.category,
        si.tags,
        si.metadata_json,
        si.liquidity,
        si.volume,
        si.recency,
        ts_rank_cd(si.tsv, q.tsq, 32)            AS fts_rank,
        similarity(si.title, q.raw)              AS trgm_sim,
        ts_headline(
            'english',
            coalesce(si.title, '') ||
                CASE WHEN si.subtitle IS NULL OR si.subtitle = '' THEN ''
                     ELSE ' — ' || si.subtitle END,
            q.tsq,
            :headline_opts
        ) AS snippet,
        (
            (ts_rank_cd(si.tsv, q.tsq, 32)                * {RANK_WEIGHT_FTS})
          + (coalesce(similarity(si.title, q.raw), 0)     * {RANK_WEIGHT_TRIGRAM})
          + (LEAST(coalesce(si.liquidity, 0) / 100000.0, 1.0) * {RANK_WEIGHT_LIQUIDITY})
          + (LEAST(coalesce(si.volume, 0)    / 1000000.0, 1.0) * {RANK_WEIGHT_VOLUME})
          + CASE
                WHEN si.recency IS NULL THEN 0
                WHEN si.recency > (now() AT TIME ZONE 'utc') - INTERVAL '1 day'  THEN {RANK_WEIGHT_RECENCY_DAY}
                WHEN si.recency > (now() AT TIME ZONE 'utc') - INTERVAL '7 days' THEN {RANK_WEIGHT_RECENCY_WEEK}
                ELSE 0
            END
        ) AS score
    FROM search_index si, q
    WHERE
        (
            si.tsv @@ q.tsq
            OR si.title % q.raw
        )
        AND (
            CAST(:type_filter AS TEXT) IS NULL
            OR si.entity_type = ANY(string_to_array(CAST(:type_filter AS TEXT), ','))
        )
    ORDER BY score DESC, si.recency DESC NULLS LAST, si.entity_type, si.entity_id
    LIMIT :limit
    """
)


async def run_query(
    session: AsyncSession,
    *,
    query: str,
    limit: int = 30,
    entity_types: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    """Execute a single ranked global-search query.

    Returns a dict with::

        {
            "query": <echo>,
            "results": [...],
            "total": <int>,
            "groups": {<entity_type>: [...]},
            "latency_ms": <float>,
        }
    """
    raw = (query or "").strip()
    if not raw:
        return {
            "query": "",
            "results": [],
            "total": 0,
            "groups": {},
            "latency_ms": 0.0,
        }

    capped_limit = max(1, min(int(limit or 30), ABSOLUTE_LIMIT))
    type_filter: Optional[str] = None
    if entity_types:
        cleaned = sorted({str(t).strip() for t in entity_types if str(t).strip()})
        if cleaned:
            type_filter = ",".join(cleaned)

    # Tune the per-session trigram threshold so the ``%`` operator
    # behaves identically regardless of server defaults.  Postgres
    # rejects bound parameters in ``SET`` statements (it parses SET
    # before it knows about prepared-statement params), so we inline
    # the constant — safe because ``TRIGRAM_SIMILARITY_THRESHOLD`` is
    # a hardcoded float, never user input.
    started = time.monotonic()
    await session.execute(
        text(f"SET LOCAL pg_trgm.similarity_threshold = {TRIGRAM_SIMILARITY_THRESHOLD}")
    )

    try:
        rs = await session.execute(
            _SEARCH_SQL,
            {
                "raw_query": raw,
                "limit": capped_limit,
                "type_filter": type_filter,
                "headline_opts": _HEADLINE_OPTS,
            },
        )
        rows = rs.mappings().all()
    except Exception as exc:
        logger.warning("Global search query failed: %s", exc)
        rows = []

    latency_ms = (time.monotonic() - started) * 1000.0

    results: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        # Render the snippet: HTML-escape first to neutralize any
        # ``<script>`` etc. embedded in user-supplied titles, *then*
        # swap the sentinel tokens for real <mark>/</mark> tags.  The
        # frontend renders the result via dangerouslySetInnerHTML.
        raw_snippet = row["snippet"] or ""
        snippet = (
            html.escape(raw_snippet)
            .replace(_HEADLINE_OPEN, "<mark>")
            .replace(_HEADLINE_CLOSE, "</mark>")
        )
        item = {
            "entity_type": row["entity_type"],
            "entity_id": row["entity_id"],
            "title": row["title"],
            "subtitle": row["subtitle"],
            "category": row["category"],
            "tags": list(row["tags"] or []),
            "metadata": dict(row["metadata_json"] or {}),
            "liquidity": float(row["liquidity"]) if row["liquidity"] is not None else None,
            "volume": float(row["volume"]) if row["volume"] is not None else None,
            "recency": (
                row["recency"].isoformat() if row["recency"] is not None else None
            ),
            "snippet": snippet,
            "score": float(row["score"] or 0.0),
            "fts_rank": float(row["fts_rank"] or 0.0),
            "trigram_similarity": float(row["trgm_sim"] or 0.0),
        }
        results.append(item)
        groups.setdefault(item["entity_type"], []).append(item)

    return {
        "query": raw,
        "results": results,
        "total": len(results),
        "groups": groups,
        "latency_ms": round(latency_ms, 2),
    }


async def log_query(
    session: AsyncSession,
    *,
    query: str,
    result_count: int,
    top_entity_type: Optional[str],
    latency_ms: float,
) -> None:
    """Persist a single search invocation to the telemetry table.

    Failures here are non-fatal — the search response is the user-
    facing concern; logging is best-effort.
    """
    try:
        await session.execute(
            text(
                "INSERT INTO search_query_log "
                "(query, result_count, top_entity_type, latency_ms, created_at) "
                "VALUES (:q, :rc, :tet, :lat, :at)"
            ),
            {
                "q": (query or "")[:500],
                "rc": int(result_count or 0),
                "tet": top_entity_type,
                "lat": float(latency_ms or 0.0),
                "at": utcnow().replace(tzinfo=None),
            },
        )
        await session.commit()
    except Exception as exc:
        logger.debug("search_query_log insert failed (non-fatal): %s", exc)
        try:
            await session.rollback()
        except Exception:
            pass


async def fetch_recent_queries(
    session: AsyncSession, *, limit: int = 10
) -> list[dict[str, Any]]:
    """Return the most recent successful (non-zero-result) queries.

    Used to power "recent searches" in the UI.  Deduplicates so the
    same query isn't shown three times in a row.
    """
    rs = await session.execute(
        text(
            """
            SELECT query, MAX(created_at) AS last_at, MAX(result_count) AS last_count
            FROM search_query_log
            WHERE result_count > 0
              AND created_at > (now() AT TIME ZONE 'utc') - INTERVAL '14 days'
            GROUP BY query
            ORDER BY last_at DESC
            LIMIT :limit
            """
        ),
        {"limit": max(1, min(int(limit or 10), 50))},
    )
    rows = rs.mappings().all()
    return [
        {
            "query": row["query"],
            "last_at": row["last_at"].isoformat() if row["last_at"] is not None else None,
            "result_count": int(row["last_count"] or 0),
        }
        for row in rows
    ]


async def fetch_index_stats(session: AsyncSession) -> dict[str, Any]:
    """Aggregate per-entity-type counts in the search index.

    Useful for an admin / debug panel and for the ``/search/reindex``
    response payload.
    """
    rs = await session.execute(
        text(
            """
            SELECT entity_type, COUNT(*) AS n, MAX(updated_at) AS last_updated
            FROM search_index
            GROUP BY entity_type
            ORDER BY n DESC
            """
        )
    )
    rows = rs.mappings().all()
    total = sum(int(r["n"] or 0) for r in rows)
    return {
        "total": total,
        "by_type": [
            {
                "entity_type": r["entity_type"],
                "count": int(r["n"] or 0),
                "last_updated": (
                    r["last_updated"].isoformat() if r["last_updated"] is not None else None
                ),
            }
            for r in rows
        ],
    }
