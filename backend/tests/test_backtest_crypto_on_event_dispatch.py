"""Backtest≡live dispatch parity for CRYPTO_UPDATE strategies.

The governing invariant: a strategy must run the SAME detect code in backtest as
live.  Live dispatches CRYPTO_UPDATE through ``strategy.on_event(event)`` (the
base ``on_event`` returns ``[]`` for CRYPTO_UPDATE, so a crypto strategy MUST
override it); the backtest historically only ever called ``detect()``.  That
caused two divergences these tests pin the fix for:

  * Bug 1 — ``btc_eth_convergence`` (+ ``btc_eth_directional_edge`` /
    ``btc_eth_maker_quote``) override BOTH ``on_event`` (the live path) and
    ``detect`` (a different path that also did a live Gamma fetch = lookahead).
    Backtest ran ``detect`` → different code than live.
  * Bug 2 — ``crypto_5m_midcycle`` / ``crypto_entropy_maker`` /
    ``crypto_spike_reversion`` / ``crypto_distance_edge`` override ``on_event``
    but have no real ``detect`` → were never replayed at all (the
    "nothing to replay" guard skipped some of them entirely).

Fix: ``_replay_discover_opportunities`` dispatches a crypto strategy through
``on_event`` iff it overrides ``on_event`` (``_has_custom_on_event``); otherwise
it keeps the existing ``detect`` path (the bonereaper-style detect_async-only
shape — kept byte-identical).  The rule keys off mechanism, not strategy names,
so it covers any user-authored crypto strategy.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from services.data_events import DataEvent, EventType
from services.strategies.base import BaseStrategy
import services.strategy_backtester as strategy_backtester


def _crypto_opp(token_id: str, src: str) -> SimpleNamespace:
    """An Opportunity-shaped object the discovery loop's ``_opp_to_positions_data``
    accepts; ``src`` tags which dispatch path produced it."""
    return SimpleNamespace(
        title=f"{src}_hit {token_id}",
        event_id=token_id,
        positions_to_take=[
            {"token_id": token_id, "side": "BUY", "price": 0.5, "size": 10.0}
        ],
        total_cost=5.0,
        expected_roi=1.0,
        risk_score=0.0,
    )


def _crypto_event(markets: list[dict], ts: datetime) -> DataEvent:
    """The DataEvent(CRYPTO_UPDATE) the live dispatch / recorded-bus replay
    forwards — markets live in ``payload["markets"]``."""
    return DataEvent(
        event_type=EventType.CRYPTO_UPDATE,
        source="unit",
        timestamp=ts,
        payload={"markets": markets, "trigger": "unit"},
    )


class _OnEventWithDetectCryptoStrategy(BaseStrategy):
    """Bug-1 shape: overrides BOTH on_event (the live crypto entry) and detect
    (which, live, is NEVER called for CRYPTO_UPDATE).  Backtest must dispatch via
    on_event — if it calls detect, the WRONG_PATH marker surfaces and the test
    fails loudly."""

    strategy_type = "unit_crypto_on_event_and_detect"
    name = "on_event + detect"
    description = "unit"
    subscriptions = [EventType.CRYPTO_UPDATE]

    def detect(self, events, markets, prices):
        return [_crypto_opp("WRONG_PATH", "detect")]

    async def on_event(self, event):
        if event.event_type != EventType.CRYPTO_UPDATE:
            return []
        return [_crypto_opp(str(m.get("id")), "on_event") for m in (event.payload.get("markets") or [])]


class _OnEventNoDetectCryptoStrategy(BaseStrategy):
    """Bug-2 shape: overrides on_event, has NO detect at all.  Must pass the
    'nothing to replay' guard AND fire via on_event."""

    strategy_type = "unit_crypto_on_event_no_detect"
    name = "on_event only"
    description = "unit"
    subscriptions = [EventType.CRYPTO_UPDATE]

    async def on_event(self, event):
        if event.event_type != EventType.CRYPTO_UPDATE:
            return []
        return [_crypto_opp(str(m.get("id")), "on_event") for m in (event.payload.get("markets") or [])]


class _DetectAsyncOnlyCryptoStrategy(BaseStrategy):
    """Safety shape (bonereaper): subscribes to CRYPTO_UPDATE but overrides ONLY
    detect_async — NOT on_event.  Must KEEP the detect path (the base on_event
    returns [] for CRYPTO_UPDATE, so rerouting it there would zero it out)."""

    strategy_type = "unit_crypto_detect_async_only"
    name = "detect_async only"
    description = "unit"
    subscriptions = [EventType.CRYPTO_UPDATE]

    def __init__(self) -> None:
        super().__init__()
        self.detect_async_calls = 0

    async def detect_async(self, events, markets, prices):
        self.detect_async_calls += 1
        # Read straight off the forwarded DataEvents so the test doesn't depend
        # on dict→Market hydration of the (token-less) unit market dicts.
        out: list[SimpleNamespace] = []
        for ev in events or []:
            payload = getattr(ev, "payload", {}) or {}
            for m in (payload.get("markets") or []):
                out.append(_crypto_opp(str(m.get("id")), "detect_async"))
        return out


class _PlainScannerStrategy(BaseStrategy):
    """A scanner strategy (overrides detect only, no on_event) — the negative
    case for the on_event-override detector."""

    strategy_type = "unit_plain_scanner"
    name = "plain detect"
    description = "unit"

    def detect(self, events, markets, prices):
        return []


async def _run_crypto_replay(monkeypatch, strategy, events_per_tick: dict[int, list[list[dict]]]):
    """Drive ``_replay_discover_opportunities`` for a CRYPTO_UPDATE strategy with
    the recorded-bus replay + catalog + price-grid stubbed.  ``events_per_tick``
    maps a tick index → a list of market-dict lists; each inner list becomes one
    DataEvent(CRYPTO_UPDATE) binned into that tick."""
    start = datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=1)

    monkeypatch.setattr(
        "services.shared_state._read_market_catalog_file",
        lambda: ([], [], {}),
    )

    async def _stub_grid(*, token_ids, ticks, start_dt, end_dt):
        return {}

    monkeypatch.setattr(strategy_backtester, "_build_per_tick_prices_grid", _stub_grid)

    async def _stub_bus(*, strategy, start_dt, ticks, actual_interval, n_ticks,
                        events_by_tick, candidate_token_ids, catalog_markets):
        n = 0
        for tick_i, market_lists in events_per_tick.items():
            idx = min(n_ticks - 1, tick_i)
            for markets in market_lists:
                events_by_tick[idx].append(_crypto_event(markets, ticks[idx]))
                n += 1
        return n

    monkeypatch.setattr(strategy_backtester, "_replay_bus_events_into_tick_grid", _stub_bus)

    return await strategy_backtester._replay_discover_opportunities(
        strategy=strategy,
        slug=strategy.strategy_type,
        start_dt=start,
        end_dt=end,
        sample_interval_seconds=600,
        max_ticks=12,
    )


# ── _has_custom_on_event ─────────────────────────────────────────────────────


def test_has_custom_on_event_detects_override():
    assert strategy_backtester._has_custom_on_event(_OnEventNoDetectCryptoStrategy()) is True
    assert strategy_backtester._has_custom_on_event(_OnEventWithDetectCryptoStrategy()) is True
    # detect_async-only (bonereaper-like) and plain-scanner do NOT override on_event.
    assert strategy_backtester._has_custom_on_event(_DetectAsyncOnlyCryptoStrategy()) is False
    assert strategy_backtester._has_custom_on_event(_PlainScannerStrategy()) is False


def test_production_crypto_strategies_are_covered():
    """Coverage guard for the real strategies the unit shapes above model: every
    production CRYPTO_UPDATE strategy must override ``on_event`` (so the backtest
    dispatches it via on_event, matching live).  The fix keys off mechanism, so a
    new user crypto strategy is covered automatically — this just pins that the
    known ones don't regress (e.g. a refactor dropping the on_event override
    would silently send the strategy back to the divergent detect() path)."""
    import importlib

    cases = [
        ("btc_eth_convergence", "BtcEthConvergenceStrategy"),
        ("btc_eth_directional_edge", "BtcEthDirectionalEdgeStrategy"),
        ("btc_eth_maker_quote", "BtcEthMakerQuoteStrategy"),
        ("crypto_5m_midcycle", "Crypto5mMidcycleStrategy"),
        ("crypto_entropy_maker", "CryptoEntropyMakerStrategy"),
        ("crypto_spike_reversion", "CryptoSpikeReversionStrategy"),
        ("crypto_distance_edge", "CryptoDistanceEdgeStrategy"),
    ]
    for mod, cls in cases:
        strategy_cls = getattr(importlib.import_module(f"services.strategies.{mod}"), cls)
        subs = [str(s) for s in (getattr(strategy_cls, "subscriptions", []) or [])]
        assert EventType.CRYPTO_UPDATE in subs, f"{cls} no longer subscribes to CRYPTO_UPDATE"
        assert strategy_cls.on_event is not BaseStrategy.on_event, (
            f"{cls} no longer overrides on_event — backtest would fall back to detect() "
            f"and silently diverge from live"
        )


# ── _run_on_event_once ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_on_event_once_dispatches_each_event_and_aggregates():
    strat = _OnEventNoDetectCryptoStrategy()
    ts = datetime(2026, 5, 1, 0, 10, 0, tzinfo=timezone.utc)
    events = [_crypto_event([{"id": "A"}], ts), _crypto_event([{"id": "B"}], ts)]
    opps = await strategy_backtester._run_on_event_once(
        strat, events, timeout_seconds=8.0, now_us=int(ts.timestamp() * 1_000_000)
    )
    assert {o.event_id for o in opps} == {"A", "B"}


# ── discovery-loop dispatch parity ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_crypto_dispatches_via_on_event_not_detect(monkeypatch):
    """Bug 1: a crypto strategy overriding BOTH on_event and detect runs
    on_event in backtest (matching live) — NOT detect."""
    strat = _OnEventWithDetectCryptoStrategy()
    opps = await _run_crypto_replay(monkeypatch, strat, {1: [[{"id": "M1"}]]})

    assert len(opps) > 0
    ids = {o.event_id for o in opps}
    assert "M1" in ids
    assert "WRONG_PATH" not in ids, "detect() ran for a crypto strategy — backtest≠live"


@pytest.mark.asyncio
async def test_crypto_on_event_no_detect_now_runs(monkeypatch):
    """Bug 2 + Finding 3: a crypto strategy with on_event and NO detect must pass
    the 'nothing to replay' guard and fire via on_event (it was silently skipped
    before the guard admitted on_event-driven strategies)."""
    strat = _OnEventNoDetectCryptoStrategy()
    opps = await _run_crypto_replay(monkeypatch, strat, {1: [[{"id": "M2"}]]})

    assert len(opps) > 0
    assert {o.event_id for o in opps} == {"M2"}


@pytest.mark.asyncio
async def test_crypto_detect_async_only_keeps_detect_path(monkeypatch):
    """Safety: a CRYPTO_UPDATE strategy overriding ONLY detect_async (no on_event
    — the bonereaper shape) keeps the detect path, NOT rerouted to on_event."""
    strat = _DetectAsyncOnlyCryptoStrategy()
    opps = await _run_crypto_replay(monkeypatch, strat, {1: [[{"id": "M3"}]]})

    assert strat.detect_async_calls > 0, "detect_async path was bypassed — bonereaper would break"
    assert {o.event_id for o in opps} == {"M3"}


@pytest.mark.asyncio
async def test_crypto_on_event_aggregates_multiple_dispatches_per_tick(monkeypatch):
    """Multiple recorded dispatches binned into one tick are EACH dispatched
    through on_event (live processes one dispatch per event; stateful strategies
    must see every dispatch) and their opportunities aggregate."""
    strat = _OnEventNoDetectCryptoStrategy()
    opps = await _run_crypto_replay(monkeypatch, strat, {1: [[{"id": "E1"}], [{"id": "E2"}]]})

    assert {o.event_id for o in opps} == {"E1", "E2"}


# ── universal detect(): base on_event routes CRYPTO_UPDATE -> detect() (LIVE) ──


@pytest.mark.asyncio
async def test_base_on_event_routes_crypto_update_to_detect():
    """Universal detect(): a detect-only crypto strategy (subscribes CRYPTO_UPDATE,
    overrides detect_async, NO on_event) now fires via the BASE on_event for
    CRYPTO_UPDATE — so the documented detect() primary works for crypto LIVE,
    matching the backtest (which already runs its detect path). Previously base
    on_event returned [] for CRYPTO_UPDATE so such a strategy never fired live."""
    strat = _DetectAsyncOnlyCryptoStrategy()
    ts = datetime(2026, 5, 1, 0, 10, 0, tzinfo=timezone.utc)
    # event[s] arg: the strategy iterates the CRYPTO_UPDATE DataEvent stream.
    opps = await strat.on_event(_crypto_event([{"id": "Z1"}], ts))
    assert strat.detect_async_calls > 0, "base on_event did not route CRYPTO_UPDATE to detect"
    assert {o.event_id for o in opps} == {"Z1"}


@pytest.mark.asyncio
async def test_base_on_event_crypto_without_detect_returns_empty():
    """A crypto strategy with neither detect nor on_event override still yields []
    (base detect() returns []) — the universal routing doesn't fabricate opps."""

    class _BareCrypto(BaseStrategy):
        strategy_type = "unit_bare_crypto"
        name = "bare"
        description = "unit"
        subscriptions = [EventType.CRYPTO_UPDATE]

    ts = datetime(2026, 5, 1, 0, 10, 0, tzinfo=timezone.utc)
    opps = await _BareCrypto().on_event(_crypto_event([{"id": "Q"}], ts))
    assert opps == []


@pytest.mark.asyncio
async def test_base_on_event_crypto_override_still_takes_precedence():
    """A strategy that OVERRIDES on_event keeps its own routing (the universal
    detect path is the BASE only) — existing crypto strategies are unaffected."""
    strat = _OnEventWithDetectCryptoStrategy()  # on_event -> on_event_hit; detect -> WRONG_PATH
    ts = datetime(2026, 5, 1, 0, 10, 0, tzinfo=timezone.utc)
    opps = await strat.on_event(_crypto_event([{"id": "M9"}], ts))
    ids = {o.event_id for o in opps}
    assert ids == {"M9"}
    assert all(o.title.startswith("on_event_hit") for o in opps)


# ── SDK guardrail: divergent-detection footgun (overrides both) ───────────────


def test_base_introspection_helpers_are_canonical():
    """The strategy-introspection helpers live canonically in base; the backtester
    delegates to them. Spot-check both surfaces agree."""
    from services.strategies import base as _base

    assert _base._has_custom_on_event(_OnEventNoDetectCryptoStrategy()) is True
    assert _base._has_custom_detect_plain(_PlainScannerStrategy()) is True
    assert _base._has_custom_detect_async(_DetectAsyncOnlyCryptoStrategy()) is True
    assert _base._has_custom_on_event(_PlainScannerStrategy()) is False
    # backtester delegates to base — same answers
    assert strategy_backtester._has_custom_on_event(_OnEventNoDetectCryptoStrategy()) is True
    assert strategy_backtester._has_custom_detect_plain(_PlainScannerStrategy()) is True


def test_divergent_detection_warning_flags_only_both_overrides():
    """The SDK guardrail warns ONLY when a strategy overrides BOTH on_event and a
    detect* method (the footgun where detect() silently becomes dead code)."""
    from services.strategies.base import divergent_detection_warning

    # both on_event + detect → footgun → warning
    warn = divergent_detection_warning(_OnEventWithDetectCryptoStrategy())
    assert warn is not None and "on_event" in warn and "detect()" in warn
    # single entry points → no warning
    assert divergent_detection_warning(_OnEventNoDetectCryptoStrategy()) is None  # on_event only
    assert divergent_detection_warning(_DetectAsyncOnlyCryptoStrategy()) is None  # detect_async only
    assert divergent_detection_warning(_PlainScannerStrategy()) is None  # plain detect only
