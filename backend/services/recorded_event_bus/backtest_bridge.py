"""Backtest replay bridge — the seam where the discovery-replay loop
meets the recorded-event bus.

Today's discovery replay (strategy_backtester._replay_discover_opportunities)
hand-loads each event kind from its own SQL table:
  * wallet_trade  → wallet_monitor_events (the one wired kind)
  * everything else → not implemented

The bus is the institutional-grade upgrade: discovery replay says
"give me events for these topics in this window," the bus heap-merges
across storage backends, and the loop dispatches them to the strategy
the same way live does.

This module gives the runner one entry point:

  events = await replay_events_for_strategy(
      strategy=strategy,
      slug=slug,
      start_dt=..., end_dt=...,
      entity_filter=...,  # per-topic asset/wallet/token filter
  )

It resolves the strategy's declared subscriptions to bus topics
(via the compat shim in batch G), pulls envelopes from the bus,
and projects them back to the legacy event shape every discovery
replay codepath already understands.  That projection is what lets
the bus light up incrementally — strategies subscribe to topics,
the bridge takes care of "what shape does the existing
event_dispatcher hand them in live?"  Backtest delivery matches live.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Iterable, Mapping, Optional

from services.recorded_event_bus.bus import bus, ReplayWindow
from services.recorded_event_bus.envelope import RecordedEvent
from services.recorded_event_bus.catalog import get_topic, TopicSpec

logger = logging.getLogger(__name__)


# ── Strategy subscription → topic resolution ─────────────────────────


# Compat map: legacy strategy ``subscriptions = ["crypto_update"]`` /
# ``subscriptions = ["market_data_refresh"]`` etc. to bus topic slugs.
# Strategies that already use dotted topic slugs (``"crypto.update.dispatch"``)
# pass through unchanged.
_LEGACY_SUBSCRIPTION_MAP: dict[str, tuple[str, ...]] = {
    "crypto_update":     ("crypto.update.dispatch",),
    "news_update":       ("news.update",),
    "weather_update":    ("weather.update",),
    "trader_activity":   ("trader.activity",),
    "wallet_trade":      ("wallet.trade",),
    "trade_execution":   ("polymarket.trade.execution",),
    # ``market_data_refresh`` is the catch-all live event for "look at
    # the current book" — strategies that subscribe to it are book-
    # driven, which the existing discovery replay handles via
    # ``_build_per_tick_prices_grid`` (not the bus).  Return empty so
    # that path keeps owning book replay.  The bus topic
    # ``polymarket.book.snapshot`` exists for explicit subscribers and
    # is also queried by the grid builder, so this isn't a duplicate
    # — just a separation of concerns: legacy book strategies get the
    # tick-grid; explicit topic subscribers get raw envelopes.
    "market_data_refresh": (),
}


def resolve_subscriptions_to_topics(subscriptions: Iterable[Any]) -> tuple[str, ...]:
    """Given a strategy's ``subscriptions`` field, return the set of
    bus topics the bridge should replay for it.

    Accepts:
      * Dotted topic slugs (``"crypto.update.dispatch"``) — pass through.
      * Legacy event type strings (``"crypto_update"``, ``"wallet_trade"``,
        ``"market_data_refresh"``) — looked up in the compat map.
      * EventType enum members — coerced to their value string then
        treated as a legacy event type string.
    """
    out: list[str] = []
    for sub in subscriptions or []:
        if hasattr(sub, "value"):
            sub = sub.value
        s = str(sub or "").strip().lower()
        if not s:
            continue
        if "." in s:
            out.append(s)  # already a topic slug
            continue
        topics = _LEGACY_SUBSCRIPTION_MAP.get(s)
        if topics is None:
            logger.debug(
                "backtest_bridge: unknown legacy subscription %r — no bus topic",
                s,
            )
            continue
        out.extend(topics)
    # Dedup while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for t in out:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return tuple(deduped)


async def replay_events_for_strategy(
    *,
    strategy: Any,
    start_dt: datetime,
    end_dt: datetime,
    entity_filter: Optional[Mapping[str, Iterable[str]]] = None,
    time_field: str = "observed_at_us",
) -> AsyncIterator[RecordedEvent]:
    """Yield RecordedEvent envelopes the strategy would have seen live.

    Resolves ``strategy.subscriptions`` to bus topics via
    :func:`resolve_subscriptions_to_topics`, filters by the catalog
    to drop unregistered / disabled / non-replayable topics, then
    streams envelopes via ``bus.replay``.

    The caller (discovery replay) is responsible for the
    legacy-event-shape projection — this bridge gives back the
    canonical envelope, and the caller decides how to feed it to
    ``strategy.detect_async`` (most strategies want
    ``(events, markets, prices)`` triples, which the discovery
    replay already constructs).
    """
    subs = getattr(strategy, "subscriptions", None) or []
    candidate_topics = resolve_subscriptions_to_topics(subs)
    if not candidate_topics:
        return

    # Filter to topics actually present in the catalog + replayable.
    topics: list[str] = []
    for t in candidate_topics:
        spec = await get_topic(t)
        if spec is None:
            logger.info(
                "backtest_bridge: strategy %r subscribes to %r — not in catalog, skipped",
                getattr(strategy, "name", "?"), t,
            )
            continue
        if not spec.is_replayable or not spec.enabled:
            logger.info(
                "backtest_bridge: topic %r not replayable/disabled — skipped",
                t,
            )
            continue
        topics.append(t)

    if not topics:
        return

    # Entity filter: re-key from caller's flat mapping to the per-topic
    # frozenset form ReplayWindow expects.
    ef: Optional[dict[str, frozenset[str]]] = None
    if entity_filter is not None:
        ef = {
            topic: frozenset(str(e) for e in entities)
            for topic, entities in entity_filter.items()
            if topic in topics
        } or None

    win = ReplayWindow(
        start_us=int(start_dt.replace(tzinfo=timezone.utc).timestamp() * 1_000_000)
            if start_dt.tzinfo is None
            else int(start_dt.timestamp() * 1_000_000),
        end_us=int(end_dt.replace(tzinfo=timezone.utc).timestamp() * 1_000_000)
            if end_dt.tzinfo is None
            else int(end_dt.timestamp() * 1_000_000),
        topics=tuple(topics),
        entity_filter=ef,
        time_field=time_field,
    )
    async for ev in bus.replay(win):
        yield ev
