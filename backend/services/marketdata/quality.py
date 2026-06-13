"""Parquet-native data-quality assessment for the unified market-data layer.

Reads the SAME canonical parquet the backtester replays (via the coverage
resolver), so the quality report reflects exactly what a backtest would see —
no SQL side-channel that disagrees with execution. Surfaces the institutional
data-quality signals an operator needs before trusting a window:

  * coverage:    row count + observed time span vs the requested window
  * crossed:     snapshots where best_bid >= best_ask (bad book)
  * gaps:        longest silence between consecutive observations
  * staleness:   age of the last observation at the window's end

Cheap: reads only the four needed columns; no full-depth ladder scan.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _us(dt: datetime) -> int:
    aware = dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    return int(aware.astimezone(timezone.utc).timestamp() * 1_000_000)


async def assess_book_quality(
    *,
    token_id: str,
    start: datetime,
    end: datetime,
    gap_threshold_seconds: float = 5.0,
) -> dict[str, Any]:
    """Assess canonical book-snapshot quality for one token over [start, end].

    Returns a JSON-friendly dict (``covered`` False with a reason when the
    token has no parquet in the window).
    """
    import pyarrow.parquet as pq

    from services.marketdata.coverage import resolve_coverage

    cov = await resolve_coverage(token_ids=[token_id], start=start, end=end)
    files = cov.files_for(token_id)
    if not files:
        return {"token_id": token_id, "covered": False, "reason": "no parquet coverage in window"}

    start_us, end_us = _us(start), _us(end)
    obs: list[int] = []
    crossed = 0
    null_book = 0
    token = str(token_id)
    for fp in files:
        try:
            t = pq.read_table(
                str(fp),
                columns=["token_id", "observed_at_us", "best_bid", "best_ask"],
                filters=[("token_id", "=", token)],
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("assess_book_quality: unreadable %s: %s", fp, exc)
            continue
        tok = t.column("token_id").to_pylist()
        o = t.column("observed_at_us").to_pylist()
        bb = t.column("best_bid").to_pylist()
        ba = t.column("best_ask").to_pylist()
        for i in range(len(o)):
            if str(tok[i] or "") != token:
                continue
            ts = o[i]
            if ts is None or ts < start_us or ts > end_us:
                continue
            obs.append(int(ts))
            b, a = bb[i], ba[i]
            if b is None or a is None:
                null_book += 1
            elif float(b) >= float(a):
                crossed += 1

    if not obs:
        return {"token_id": token_id, "covered": False, "reason": "no rows in window"}

    obs.sort()
    n = len(obs)
    first_us, last_us = obs[0], obs[-1]
    # Longest silence between consecutive observations.
    max_gap_us = 0
    for i in range(1, n):
        g = obs[i] - obs[i - 1]
        if g > max_gap_us:
            max_gap_us = g
    window_secs = max(1e-9, (end_us - start_us) / 1_000_000)
    span_secs = (last_us - first_us) / 1_000_000

    return {
        "token_id": token_id,
        "covered": True,
        "files": len(files),
        "rows": n,
        "first_observed": datetime.fromtimestamp(first_us / 1e6, tz=timezone.utc).isoformat(),
        "last_observed": datetime.fromtimestamp(last_us / 1e6, tz=timezone.utc).isoformat(),
        "observed_span_seconds": round(span_secs, 3),
        "window_seconds": round(window_secs, 3),
        "rows_per_minute": round(n / (window_secs / 60.0), 2),
        "crossed_book_count": crossed,
        "null_book_count": null_book,
        "max_gap_seconds": round(max_gap_us / 1_000_000, 3),
        "has_large_gap": (max_gap_us / 1_000_000) > gap_threshold_seconds,
        "staleness_at_window_end_seconds": round((end_us - last_us) / 1_000_000, 3),
    }


async def assess_universe_quality(
    *,
    token_ids: list[str],
    start: datetime,
    end: datetime,
    gap_threshold_seconds: float = 5.0,
    max_tokens: int = 200,
) -> dict[str, Any]:
    """Assess quality across a token universe (capped) + an aggregate roll-up."""
    toks = [str(t) for t in token_ids][:max_tokens]
    per_token = []
    for tok in toks:
        per_token.append(await assess_book_quality(
            token_id=tok, start=start, end=end, gap_threshold_seconds=gap_threshold_seconds,
        ))
    covered = [q for q in per_token if q.get("covered")]
    return {
        "requested_tokens": len(toks),
        "covered_tokens": len(covered),
        "total_rows": sum(int(q.get("rows", 0)) for q in covered),
        "total_crossed_book": sum(int(q.get("crossed_book_count", 0)) for q in covered),
        "tokens_with_large_gap": sum(1 for q in covered if q.get("has_large_gap")),
        "per_token": per_token,
    }


__all__ = ["assess_book_quality", "assess_universe_quality"]
