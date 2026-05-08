"""Retention sweep for orphaned `trade_signals` skeleton rows.

Plan 0010 added a publish-side skeleton-INSERT pass to
`intent_runtime.publish_opportunities` that synchronously commits a
``(source, dedupe_key)`` placeholder row in `trade_signals` BEFORE
the projection loop enriches it with `payload_json`,
`runtime_sequence`, the strategy's intended `expires_at`, etc.  That
closed the in-process publish→consume FK race for `traders` source.

Plan 0011 hardens that mechanism's failure mode.  If the rest of
`publish_opportunities` dies between the skeleton commit and the
projection-loop UPSERT (process kill, connection drop, unhandled
exception, mid-call `docker compose restart`), the skeleton row
stays in `trade_signals` with ``payload_json IS NULL`` and
``runtime_sequence IS NULL``.  In the steady state the next
genuine publish for the same dedupe_key adopts the stuck row's id
(skeleton-INSERT's ON CONFLICT DO NOTHING + re-SELECT path), so the
system self-heals — but a dedupe_key that never republishes leaves
its skeleton in the table forever.  The terminal-row pruner keys on
``expires_at < now()``; pre-plan-0011 those rows had
``expires_at IS NULL`` and were invisible to it.

This module owns the cleanup loop: a periodic DELETE of skeleton
rows older than ``max_age_seconds`` (default 1 hour) on the
`discovery` plane.  It is intentionally narrow:

1. We DELETE outright rather than mark expired — the rows never
   carried any consumer-visible state, so there is nothing to
   audit.  ``trader_signal_consumption`` cannot reference them
   (publish failed before any consumer saw the id).
2. We bound `max_age_seconds` to ``>= 60`` defensively to avoid
   racing the projection loop in dev / under heavy load.  The
   production default of 3600 s is generous; the projection loop
   commits within ~500 ms in steady state.
3. The retention sweep lives on the discovery plane (not trading)
   because the trading plane already runs the terminal-row pruner
   and we want the orphan-deletion path off the trader-cycle's
   10 s budget.

Verification queries (operator-facing):

```sql
-- Steady-state stuck-skeleton count.  Should be 0.
select count(*) from trade_signals
where payload_json is null
  and runtime_sequence is null
  and status = 'pending'
  and created_at < now() - interval '1 minute';

-- Last 24 h of pruned skeletons (correlate with publish failures).
select created_at::date, count(*) deleted
from worker_snapshot
where worker_name = 'skeleton-signal-retention'
group by 1 order by 1 desc;
```
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


_MIN_MAX_AGE_SECONDS = 60


async def prune_stuck_skeletons(
    session: AsyncSession,
    *,
    max_age_seconds: int,
) -> int:
    """Delete `trade_signals` skeleton rows older than `max_age_seconds`.

    A skeleton row is identified by the publish-time committed shape:

    - ``payload_json IS NULL`` (projection never enriched it),
    - ``runtime_sequence IS NULL`` (no projection-loop sequence
      assigned),
    - ``status = 'pending'`` (publish-time default; never advanced
      by any consumer),
    - ``created_at < now() - interval '<max_age_seconds> seconds'``.

    `max_age_seconds` is bounded to ``>= 60`` to avoid racing the
    projection loop in dev / under heavy load.

    Returns the number of rows deleted (0 in steady state).
    """
    bounded_max_age = max(_MIN_MAX_AGE_SECONDS, int(max_age_seconds))
    result = await session.execute(
        text(
            """
            DELETE FROM trade_signals
            WHERE payload_json IS NULL
              AND runtime_sequence IS NULL
              AND status = 'pending'
              AND created_at < now() - make_interval(secs => :max_age_seconds)
            RETURNING id
            """
        ),
        {"max_age_seconds": bounded_max_age},
    )
    deleted_ids = result.fetchall()
    await session.commit()
    return len(deleted_ids)
