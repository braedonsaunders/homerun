"""Lightweight per-trade latency trace for the crypto trading hot path.

Companion to ``tools/crypto_latency_harness.py``.  This module exposes
two primitives:

  * ``record_wire_event(source, asset, ts_ms)`` — called by every WS
    feed handler (Binance, Chainlink, Polymarket book) on each message
    arrival.  Stores the most-recent wire timestamp per (source, asset)
    in a process-local dict.  Sub-microsecond cost; all bookkeeping is
    wrapped in try/except so a recording bug can never raise into a
    live trade hot path.

  * ``emit_trace(stages, **fields)`` — called by ``place_order`` once
    per completed order (success OR failure).  Logs a structured
    ``crypto_latency_trace`` line that the harness aggregator parses.

Two helpers feed those primitives:

  * ``freshest_wire_ts_ms()`` — the most recent wire ts across all
    sources and assets.  Used by ``place_order`` as a conservative
    ``t0_wire`` baseline when the per-asset attribution isn't
    available.

  * ``wire_ts_for_asset(asset)`` — the most recent wire ts for a
    specific asset (e.g. "BTC", "ETH"), looking across all WS sources.
    Used when the Opportunity carries an explicit asset tag.

Design constraints
==================

1. **Never raise**.  Every public function is wrapped in try/except
   that returns a sentinel (None, 0, no-op) on failure.  The hot path
   is a live trading service; an instrumentation bug must NOT propagate.
2. **Sub-microsecond per call**.  All recording uses simple dict
   writes — no locks, no DB, no async.  CPython's dict assignment is
   atomic under the GIL, so there are no race conditions for the
   "latest" semantics this module needs.
3. **Lossy-OK**.  If two WS messages arrive in the same nanosecond
   for the same (source, asset), one wins; we don't care which.
4. **Bounded memory**.  The dicts are keyed by ``(source, asset)``
   tuples; cardinality is small (a handful of sources × ~20 assets).
   No bounded-LRU is needed.

Trace emission format (parsed by ``tools/crypto_latency_harness.py``)::

    crypto_latency_trace signal_id=<id> token_id=<id> lane=<lane>
        strategy=<name> source=<crypto|...> wire_ts_ms=<int>
        breakdown_ms=<dict>

The ``breakdown_ms`` dict carries per-stage durations in ms, e.g.::

    {
        "wire_to_release": 47.3,
        "release_to_submit": 12.1,
        "submit_to_post_order": 41.0,
        "post_order_to_ack": 116.4,
        "ack_to_persist": 8.2,
        "wire_to_ack": 224.0,
    }

Stage names are intentionally mirrored from the harness regex —
adding a stage here only requires updating the emitter; the
aggregator picks it up automatically into the ``trace.<stage>``
bag.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from utils.logger import get_logger

_logger = get_logger("crypto_latency_trace")


# ---------------------------------------------------------------------------
# Per-source / per-asset wire timestamp tracking
# ---------------------------------------------------------------------------

# (source, asset) -> latest ts_ms seen.  Source is one of:
#   - "binance"     : Binance Spot WS (binance_feed.py)
#   - "chainlink"   : Chainlink RTDS WS (chainlink_feed.py)
#   - "polymarket"  : Polymarket CLOB book delta WS (ws_feeds.py)
# Asset is the canonical upper-case ticker for crypto sources ("BTC",
# "ETH", ...) or the Polymarket asset_id / token_id for the Polymarket
# source.
_wire_ts_by_source_asset: dict[tuple[str, str], int] = {}

# Per-source freshest ts (cached for fast freshest_wire_ts_ms() lookup
# without a dict scan on every call).  Equal to
# max(_wire_ts_by_source_asset[source, *]).
_wire_ts_by_source: dict[str, int] = {}


def record_wire_event(source: str, asset: str, ts_ms: Optional[int] = None) -> None:
    """Record the arrival of a WS message for ``(source, asset)``.

    Called from every WS feed handler.  ``ts_ms`` defaults to current
    wall-clock; pass an explicit value when the feed already has the
    arrival timestamp (e.g. ``binance_feed`` already computes
    ``now_ms = int(time.time() * 1000)`` once per message).

    No-op on bad input.  Never raises.
    """
    try:
        if not source or not asset:
            return
        ts = int(ts_ms) if ts_ms is not None else int(time.time() * 1000)
        if ts <= 0:
            return
        # Asset normalization: crypto tickers are upper-cased so different
        # casings don't fragment the dict; Polymarket token_ids stay as-is.
        asset_key = asset.upper() if source in ("binance", "chainlink") else asset
        key = (source, asset_key)
        _wire_ts_by_source_asset[key] = ts
        # Keep the per-source cache monotonically nondecreasing — only
        # promote when we see a fresher ts than the cached max.  Any
        # later asset for the same source updates only if it's also
        # fresher than the existing cache (the common case in
        # steady-state).
        prev = _wire_ts_by_source.get(source, 0)
        if ts > prev:
            _wire_ts_by_source[source] = ts
    except Exception:
        # Instrumentation failure must never crash the WS handler.
        return


def freshest_wire_ts_ms() -> Optional[int]:
    """Most recent wire ts across ALL sources and assets.

    Returns ``None`` if no wire events have been recorded yet (e.g.
    during the first few ms of process startup before any feed has
    delivered a message).
    """
    try:
        if not _wire_ts_by_source:
            return None
        return max(_wire_ts_by_source.values())
    except Exception:
        return None


def wire_ts_for_asset(asset: str) -> Optional[int]:
    """Most recent wire ts for a specific asset across all WS sources.

    Useful when the Opportunity carries an explicit asset tag (e.g.
    ``crypto_underlying_asset = "BTC"``); this returns ``max(binance_ts,
    chainlink_ts)`` for that asset, ignoring whichever source happened
    to publish first.
    """
    try:
        if not asset:
            return None
        key = asset.upper()
        candidates: list[int] = []
        for source in ("binance", "chainlink"):
            ts = _wire_ts_by_source_asset.get((source, key))
            if ts is not None:
                candidates.append(ts)
        return max(candidates) if candidates else None
    except Exception:
        return None


def wire_ts_for_token(token_id: str) -> Optional[int]:
    """Most recent Polymarket book-update wire ts for a specific token.

    Used by the Polymarket book-side path (the inverse direction:
    "when did we last see the book the strategy is reacting to?").
    """
    try:
        if not token_id:
            return None
        return _wire_ts_by_source_asset.get(("polymarket", str(token_id)))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Trace emission
# ---------------------------------------------------------------------------


def emit_trace(
    *,
    signal_id: Optional[str] = None,
    token_id: Optional[str] = None,
    lane: Optional[str] = None,
    strategy: Optional[str] = None,
    source: Optional[str] = None,
    wire_ts_ms: Optional[int] = None,
    breakdown_ms: Optional[dict[str, float]] = None,
    status: Optional[str] = None,
    **extra: Any,
) -> None:
    """Emit a single ``crypto_latency_trace`` log line.

    ``breakdown_ms`` should be a dict of stage_name -> duration_ms.
    The harness parser tolerates missing fields, so callers can pass
    only the stages they have timing for.

    The log shape is intentionally simple and grep-friendly — the
    harness's ``CRYPTO_TRACE_RE`` regex looks for the literal
    ``crypto_latency_trace`` token followed by ``key=value`` pairs.

    Never raises.
    """
    try:
        # Build the message payload.  Logger formats key=value naturally
        # via the ``data`` extras hook.
        kv: dict[str, Any] = {}
        if signal_id is not None:
            kv["signal_id"] = signal_id
        if token_id is not None:
            kv["token_id"] = token_id
        if lane is not None:
            kv["lane"] = lane
        if strategy is not None:
            kv["strategy"] = strategy
        if source is not None:
            kv["source"] = source
        if wire_ts_ms is not None:
            kv["wire_ts_ms"] = int(wire_ts_ms)
        if status is not None:
            kv["status"] = status
        # Round breakdown values to 1dp for log compactness.
        if isinstance(breakdown_ms, dict) and breakdown_ms:
            kv["breakdown_ms"] = {
                str(k): round(float(v), 1)
                for k, v in breakdown_ms.items()
                if v is not None
            }
        kv.update({k: v for k, v in extra.items() if v is not None})
        _logger.info("crypto_latency_trace", **kv)
    except Exception:
        # Instrumentation failure must never propagate.
        return


# ---------------------------------------------------------------------------
# Convenience: convert a monotonic-clock-derived ms duration to a dict slot
# ---------------------------------------------------------------------------


def stage_duration_ms(started_mono: float, ended_mono: Optional[float] = None) -> float:
    """Convert two ``time.monotonic()`` readings to a millisecond delta.

    Convenience for callers that already use ``time.monotonic()`` for
    per-stage timing (e.g. ``live_execution_service``).  Returns 0.0 on
    bad input rather than negative values.
    """
    try:
        end = ended_mono if ended_mono is not None else time.monotonic()
        delta = (end - started_mono) * 1000.0
        return round(delta, 1) if delta >= 0 else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Diagnostics (used by introspection endpoints, not the hot path)
# ---------------------------------------------------------------------------


def snapshot() -> dict[str, Any]:
    """Return the current wire-ts dict for diagnostic dumps.

    Not called from the hot path — only from periodic health-check
    handlers or test fixtures that want to verify recording is working.
    """
    try:
        now_ms = int(time.time() * 1000)
        return {
            "now_ms": now_ms,
            "by_source": dict(_wire_ts_by_source),
            "by_source_asset": {
                f"{src}:{asset}": ts
                for (src, asset), ts in _wire_ts_by_source_asset.items()
            },
            "freshest_age_ms": (
                now_ms - max(_wire_ts_by_source.values())
                if _wire_ts_by_source
                else None
            ),
        }
    except Exception:
        return {}
