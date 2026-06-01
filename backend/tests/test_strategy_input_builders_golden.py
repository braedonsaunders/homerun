"""Golden characterization tests for the canonical DataEvent builders.

Goal-A gate for the backtest≡live fidelity work: every site that constructs a
``DataEvent(MARKET_DATA_REFRESH)`` / ``DataEvent(CRYPTO_UPDATE)`` is being
routed through ``services.strategy_inputs.build_market_data_refresh_inputs`` /
``build_crypto_update_inputs``.  These tests pin that the builder reproduces
each call site's *current inline literal* **byte-for-byte** — the inline literal
is copied verbatim into each test and compared with ``==`` (``DataEvent`` is a
frozen dataclass, so ``==`` compares all 15 fields).  A fixed tz-aware UTC
timestamp is used so ``__post_init__``'s UTC normalization is identical on both
sides.

If a future edit changes a builder's defaults, payload wiring, or field
plumbing, the corresponding golden here fails — that is the regression guard
that lets the live-dispatch routing land safely.
"""
from __future__ import annotations

from datetime import datetime, timezone

from services.data_events import DataEvent, EventType
from services.strategy_inputs import (
    build_crypto_update_inputs,
    build_market_data_refresh_inputs,
)

# Fixed truth-time so both sides normalize identically in __post_init__.
TS = datetime(2026, 5, 28, 23, 14, 0, tzinfo=timezone.utc)


# ── CRYPTO_UPDATE ──────────────────────────────────────────────────────────


def test_g1_crypto_update_live_market_runtime():
    """LIVE market_runtime._run_opportunity_dispatch_loop (market_runtime.py
    ~1763): payload == {"markets": copied_for_event, "trigger": str(trigger)},
    source "market_runtime"."""
    markets = [{"condition_id": "0xabc", "up_price": 0.55, "down_price": 0.45}]
    trigger = "binance_tick"

    old = DataEvent(
        event_type=EventType.CRYPTO_UPDATE,
        source="market_runtime",
        timestamp=TS,
        payload={"markets": markets, "trigger": str(trigger)},
    )
    new = build_crypto_update_inputs(
        markets=markets, trigger=trigger, source="market_runtime", timestamp=TS
    )

    assert new == old
    assert list(new.payload) == ["markets", "trigger"]
    assert new.payload["trigger"] == "binance_tick"  # str()-coerced


def test_g2_crypto_update_projection():
    """PROJECTION project_crypto_update_events (projection.py ~557): payload ==
    {"markets": market_dicts, "trigger": "marketdata.projection",
    "event_source": "imported_parquet"}."""
    market_dicts = [{"id": "m1", "up_price": 0.6, "down_price": 0.4}]

    old = DataEvent(
        event_type=EventType.CRYPTO_UPDATE,
        source="marketdata.projection",
        timestamp=TS,
        payload={
            "markets": market_dicts,
            "trigger": "marketdata.projection",
            "event_source": "imported_parquet",
        },
    )
    new = build_crypto_update_inputs(
        markets=market_dicts,
        trigger="marketdata.projection",
        source="marketdata.projection",
        timestamp=TS,
        extra_payload={"event_source": "imported_parquet"},
    )

    assert new == old
    # Key ORDER, not just content — extras append after markets/trigger.
    assert list(new.payload) == ["markets", "trigger", "event_source"]


def test_g6_crypto_update_defaults_and_unset_fields():
    """No extra_payload → payload is exactly {markets, trigger}; every unset
    DataEvent field keeps its dataclass default (matches the live literal)."""
    markets = [{"id": "m1"}]
    new = build_crypto_update_inputs(markets=markets, trigger="t", timestamp=TS)

    assert new.payload == {"markets": markets, "trigger": "t"}
    # Structured fields the crypto literal never sets stay at their defaults.
    assert new.markets is None
    assert new.events is None
    assert new.prices is None
    assert new.scan_mode is None
    assert new.market_id is None
    assert new.token_id is None
    assert new.old_price is None
    assert new.new_price is None
    assert new.changed_token_ids is None
    assert new.changed_market_ids is None
    assert new.affected_market_ids is None


# ── MARKET_DATA_REFRESH ─────────────────────────────────────────────────────


def test_g3_market_data_refresh_live_fast_scan():
    """LIVE scanner fast lane (scanner.py ~3949): incremental batch, populated
    changed_token_ids (reactive mode)."""
    markets = [{"id": "m1"}, {"id": "m2"}]
    events = [{"id": "e1"}]
    prices = {"t1": {"bid": 0.5, "ask": 0.51}}

    payload = {
        "scan_mode": "realtime_reactive",
        "strategy_batch": "incremental",
        "changed_token_count": 2,
        "changed_market_count": 1,
        "affected_market_count": 2,
    }
    old = DataEvent(
        event_type=EventType.MARKET_DATA_REFRESH,
        source="scanner_fast_reactive",
        timestamp=TS,
        payload=payload,
        markets=markets,
        events=events,
        prices=prices,
        scan_mode="realtime_reactive",
        changed_token_ids=["t1", "t2"],
        changed_market_ids=["m1"],
        affected_market_ids=["m1", "m2"],
    )
    new = build_market_data_refresh_inputs(
        source="scanner_fast_reactive",
        timestamp=TS,
        payload=payload,
        markets=markets,
        events=events,
        prices=prices,
        scan_mode="realtime_reactive",
        changed_token_ids=["t1", "t2"],
        changed_market_ids=["m1"],
        affected_market_ids=["m1", "m2"],
    )
    assert new == old


def test_g4_market_data_refresh_live_full_snapshot():
    """LIVE scanner full-snapshot lane (scanner.py ~4343): changed_token_ids is
    NOT passed (defaults None), scan_mode "full_snapshot_heavy", 9-key payload."""
    markets = [{"id": "m1"}]
    events = [{"id": "e1"}]
    prices = {"t1": {"bid": 0.3}}
    market_ids = ["m1"]

    payload = {
        "scan_mode": "full_snapshot_heavy",
        "strategy_batch": "full_snapshot",
        "reason": "scheduled",
        "targeted": False,
        "chunk_start": 0,
        "chunk_end": 1,
        "chunk_size": 1,
        "total_market_count": 1,
        "affected_market_count": 1,
    }
    old = DataEvent(
        event_type=EventType.MARKET_DATA_REFRESH,
        source="scanner_full_snapshot",
        timestamp=TS,
        payload=payload,
        markets=markets,
        events=events,
        prices=prices,
        scan_mode="full_snapshot_heavy",
        changed_market_ids=market_ids,
        affected_market_ids=market_ids,
    )
    new = build_market_data_refresh_inputs(
        source="scanner_full_snapshot",
        timestamp=TS,
        payload=payload,
        markets=markets,
        events=events,
        prices=prices,
        scan_mode="full_snapshot_heavy",
        changed_market_ids=market_ids,
        affected_market_ids=market_ids,
    )
    assert new == old
    assert new.changed_token_ids is None  # full lane never sets it


def test_g5_market_data_refresh_projection():
    """PROJECTION project_market_data_refresh_events (projection.py ~462): only
    markets + prices are set; events/scan_mode/changed_*/affected_* default None."""
    market_dicts = [{"id": "m1", "clobTokenIds": ["t1", "t2"]}]
    prices_map = {"t1": {"bid": 0.4, "ask": 0.42, "best_bid": 0.4, "best_ask": 0.42}}

    payload = {
        "markets": market_dicts,
        "prices": prices_map,
        "updated_at": TS.isoformat().replace("+00:00", "Z"),
        "trigger": "marketdata.projection",
        "event_source": "imported_parquet",
    }
    old = DataEvent(
        event_type=EventType.MARKET_DATA_REFRESH,
        source="marketdata.projection",
        timestamp=TS,
        payload=payload,
        markets=market_dicts,
        prices=prices_map,
    )
    new = build_market_data_refresh_inputs(
        source="marketdata.projection",
        timestamp=TS,
        payload=payload,
        markets=market_dicts,
        prices=prices_map,
    )
    assert new == old
    assert new.events is None
    assert new.scan_mode is None
    assert new.changed_market_ids is None
    assert new.affected_market_ids is None
