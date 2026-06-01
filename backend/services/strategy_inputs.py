"""Canonical reconstructors for the inputs a strategy's
``detect(events, markets, prices)`` receives.

Today this module owns the ONE faithful ``dict -> Market`` / ``dict -> Event``
rehydration used when replaying recorded or projected market snapshots, so the
backtest stops carrying divergent reconstructors.

This module also owns the canonical ``DataEvent`` builders
(:func:`build_market_data_refresh_inputs` / :func:`build_crypto_update_inputs`)
that the LIVE scanner + crypto dispatch and the backtest projection construct
their events through, so a strategy's ``(events, markets, prices)`` are wired in
ONE place across live and backtest.  (The backtest's recorded-event-bus replay
sites are deliberately NOT routed through these builders — they reconstruct from
a recorded envelope whose payload is *already* in the builder's canonical shape,
because the live producer used the builder when it recorded it; re-imposing the
builder there would only risk byte drift.)

Note on the THIRD dict->Market path: the live crypto path uses
``btc_eth_convergence._market_from_crypto_dict`` for crypto-worker dicts
(``up_price``/``down_price``/``end_time``).  It is intentionally NOT folded into
``hydrate_market``: the backtest now dispatches crypto strategies through their
own ``on_event`` (the live entry point), so that reconstructor runs identically
live and in backtest — a private live-path detail, no longer a
backtest-vs-live divergence.

Dependency-light on purpose (imports only ``models.market`` + stdlib; the
``services.data_events`` import is lazy, inside the builders) so scanner /
backtester / projection can all import it without the existing circular-import
dance.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from models.market import Event, Market


def hydrate_market(m: Any) -> Optional[Market]:
    """Rebuild a ``Market`` from a recorded/replayed market dict, faithfully.

    Recorded catalog-snapshot markets are ``Market.model_dump(mode="json")``
    outputs — snake_case field names only — so the FAITHFUL inverse is
    ``model_validate``.  ``from_gamma_response`` parses the Polymarket *gamma
    API* shape (camelCase: ``endDate``, ``clobTokenIds`` …) and silently drops
    fields whose names don't match — notably ``end_date``, which scanner-tick
    strategies (e.g. tail_end_carry) read for days-to-resolution — so it must
    only be used for genuine gamma payloads (incl. the marketdata projection,
    which deliberately emits camelCase so from_gamma_response picks up the
    date/tokens).  Route by shape; fall back to from_gamma_response when
    model_validate can't produce a usable Market."""
    if not isinstance(m, dict):
        return None
    looks_gamma = "clobTokenIds" in m or "endDate" in m or "conditionId" in m
    if not looks_gamma:
        try:
            mk = Market.model_validate(m)
            if getattr(mk, "clob_token_ids", None):
                return mk
        except Exception:
            pass
    try:
        return Market.from_gamma_response(m)
    except Exception:
        return None


def hydrate_markets(markets: list[Any]) -> list[Market]:
    """Rebuild a list of ``Market`` objects, dropping any that fail to hydrate."""
    out: list[Market] = []
    for m in markets or []:
        mk = hydrate_market(m)
        if mk is not None:
            out.append(mk)
    return out


def hydrate_events(raw_events: list[Any]) -> list[Event]:
    """Rebuild ``Event`` models from raw event dicts (gamma shape); pass through
    any already-built ``Event`` objects.  Mirrors what the live
    MARKET_DATA_REFRESH carries in ``DataEvent.events``."""
    out: list[Event] = []
    for raw in (raw_events or []):
        if isinstance(raw, dict):
            try:
                out.append(Event.from_gamma_response(raw))
            except Exception:
                continue
        else:
            out.append(raw)
    return out


# ---------------------------------------------------------------------------
# Canonical DataEvent builders
#
# The ONE place the live scanner / live crypto dispatch / backtest projection
# construct the ``DataEvent`` a strategy's ``on_event`` (and the backtest
# discovery loop) receives.  Centralising the ``event_type`` + field wiring +
# defaults here is what makes "backtest feeds a strategy byte-identical
# (events, markets, prices) it sees LIVE" enforceable: the golden
# characterization tests pin that each call site's DataEvent is unchanged.
#
# ``services.data_events`` is imported lazily (inside each builder) so this
# module stays importable from scanner / market_runtime / backtester /
# projection without the circular-import dance.
# ---------------------------------------------------------------------------


def build_crypto_update_inputs(
    *,
    markets: list,
    trigger: Any,
    timestamp: datetime,
    source: str = "market_runtime",
    extra_payload: Optional[dict] = None,
) -> "DataEvent":
    """Build the canonical ``DataEvent(CRYPTO_UPDATE)`` shared by the live
    market_runtime dispatch and the imported-parquet projection.

    The crypto payload is ``{"markets": markets, "trigger": str(trigger),
    **(extra_payload or {})}`` — markets and trigger first (so the key order
    matches the live literal byte-for-byte), then any source-specific extras
    (the projection adds ``event_source``).  ``timestamp`` must be a tz-aware
    UTC datetime (``DataEvent.__post_init__`` enforces this).
    """
    from services.data_events import DataEvent, EventType

    payload: dict[str, Any] = {"markets": markets, "trigger": str(trigger)}
    if extra_payload:
        payload.update(extra_payload)
    return DataEvent(
        event_type=EventType.CRYPTO_UPDATE,
        source=source,
        timestamp=timestamp,
        payload=payload,
    )


def build_market_data_refresh_inputs(
    *,
    source: str,
    timestamp: datetime,
    payload: dict,
    markets: Optional[list] = None,
    events: Optional[list] = None,
    prices: Optional[dict] = None,
    scan_mode: Optional[str] = None,
    changed_token_ids: Optional[list] = None,
    changed_market_ids: Optional[list] = None,
    affected_market_ids: Optional[list] = None,
) -> "DataEvent":
    """Build the canonical ``DataEvent(MARKET_DATA_REFRESH)`` shared by the live
    scanner (fast + full-snapshot lanes) and the imported-parquet projection.

    The MARKET_DATA_REFRESH ``payload`` key set is **site-specific and disjoint**
    (fast scan vs full snapshot vs projection carry different diagnostic keys),
    so the caller passes the fully-built ``payload`` dict in whole — the builder
    centralises only the ``event_type`` + structured-field wiring (markets /
    events / prices / scan_mode / the *_market_ids) and the dataclass defaults,
    which is where live ↔ backtest used to drift.  Unset structured fields keep
    their ``DataEvent`` defaults (``None``), so a call that passes only markets +
    prices (the projection) is byte-identical to the current inline literal.
    ``timestamp`` must be a tz-aware UTC datetime.
    """
    from services.data_events import DataEvent, EventType

    return DataEvent(
        event_type=EventType.MARKET_DATA_REFRESH,
        source=source,
        timestamp=timestamp,
        payload=payload,
        markets=markets,
        events=events,
        prices=prices,
        scan_mode=scan_mode,
        changed_token_ids=changed_token_ids,
        changed_market_ids=changed_market_ids,
        affected_market_ids=affected_market_ids,
    )
