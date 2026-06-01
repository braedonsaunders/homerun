"""Canonical reconstructors for the inputs a strategy's
``detect(events, markets, prices)`` receives.

Today this module owns the ONE faithful ``dict -> Market`` / ``dict -> Event``
rehydration used when replaying recorded or projected market snapshots, so the
backtest stops carrying divergent reconstructors.

Convergence note (deferred, golden-test-gated live phase): the LIVE scanner
builds ``Market`` objects directly (never serialized), and the crypto path uses
``btc_eth_convergence._market_from_crypto_dict`` — a THIRD dict->Market path.
Folding those onto ``hydrate_market`` requires changing live dispatch, so it is
scheduled separately; this module is the shared seam they will converge onto.

Dependency-light on purpose (imports only ``models.market``) so scanner /
backtester / projection can all import it without the existing circular-import
dance.
"""
from __future__ import annotations

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
