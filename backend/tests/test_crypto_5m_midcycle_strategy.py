"""Decision-gate tests for the crypto_5m_midcycle strategy.

Each test exercises one filter in ``Crypto5mMidcycleStrategy._evaluate_market``
in isolation — wrong timeframe, asset not in config, before midcycle,
distance below threshold, oracle stale, VWAP out of band — plus the
happy path where every gate passes and an Opportunity is emitted.

Tests seed the WS-fed PriceCache directly with synthetic order books
(see ``test_strategy_sdk_orderbook.py`` for the same pattern) so the
``StrategySDK.get_order_book_depth`` call in the strategy returns
deterministic results without any real Polymarket connectivity.
"""

from __future__ import annotations

import pytest

from services.optimization.vwap import OrderBookLevel
from services.strategies.crypto_5m_midcycle import (
    Crypto5mMidcycleStrategy,
    crypto_5m_midcycle_config_schema,
)
from services.ws_feeds import FeedManager, get_feed_manager


# A 5-minute cycle ending at this fixed UTC timestamp (epoch millis).
# All time math in the tests is anchored here for determinism.
END_MS = 2_000_000_000_000  # arbitrary far-future ms
CYCLE_MS = 300_000
START_MS = END_MS - CYCLE_MS
MIDCYCLE_MS = START_MS + 150_000  # 150s into cycle

# CLOB token IDs are 50+ char hex strings on Polymarket — match length.
YES_TOKEN = "0x" + "a" * 60
NO_TOKEN = "0x" + "b" * 60


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_cache():
    FeedManager.reset_instance()
    yield get_feed_manager().cache
    FeedManager.reset_instance()


@pytest.fixture
def strategy():
    s = Crypto5mMidcycleStrategy()
    s.configure({})  # use defaults
    return s


def _seed_book(cache, token_id: str, *, ask_price: float, ask_size: float = 1000.0) -> None:
    """Seed a simple book where buys at ``ask_price`` are guaranteed."""
    cache.update(
        token_id,
        bids=[OrderBookLevel(price=max(0.0, ask_price - 0.005), size=1000.0)],
        asks=[OrderBookLevel(price=ask_price, size=ask_size)],
    )


def _build_market_dict(
    *,
    asset: str = "SOL",
    timeframe: str = "5min",
    end_ms: int = END_MS,
    reference: float = 100.0,
    # Default 20 bps so happy-path tests clear the 2026-05-05-bumped
    # ``min_distance_bps`` floor of 15 bps. Tests that target the
    # distance gate itself override this explicitly.
    spot: float = 100.20,
    oracle_age_ms: float = 200.0,
    oracle_source: str = "chainlink",
    yes_token: str = YES_TOKEN,
    no_token: str = NO_TOKEN,
) -> dict:
    """Build a crypto-worker-shaped market dict for input to on_event."""
    from datetime import datetime, timezone

    end_iso = datetime.fromtimestamp(end_ms / 1000.0, tz=timezone.utc).isoformat()
    return {
        "condition_id": "0xfake_market_id",
        "id": "0xfake_market_id",
        "slug": f"{asset.lower()}-up-or-down-{timeframe}",
        "question": f"{asset} up or down?",
        "asset": asset,
        "timeframe": timeframe,
        "up_price": 0.55,
        "down_price": 0.45,
        "liquidity": 5000.0,
        "clob_token_ids": [yes_token, no_token],
        "end_time": end_iso,
        "price_to_beat": reference,
        "oracle_price": spot,
        "oracle_prices_by_source": {
            oracle_source: {
                "price": spot,
                "age_ms": oracle_age_ms,
            },
        },
    }


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------


def test_config_schema_exposes_all_user_knobs():
    schema = crypto_5m_midcycle_config_schema()
    keys = {f["key"] for f in schema["param_fields"]}
    assert {
        "enabled",
        "assets",
        "min_distance_bps",
        "max_entry_price",
        "min_entry_price",
        "bet_size_usd",
        "midcycle_seconds",
        "min_seconds_to_resolution",
        "max_oracle_age_ms",
    } <= keys


def test_configure_normalizes_assets_to_canonical_names():
    s = Crypto5mMidcycleStrategy()
    s.configure({"assets": "  sol , XBT , eth ,unknown_coin"})
    # XBT → BTC; unknown filtered; case + whitespace normalized
    assert s.config["assets"] == ["SOL", "BTC", "ETH"]


def test_configure_drops_unknown_assets():
    s = Crypto5mMidcycleStrategy()
    s.configure({"assets": ["DOGE", "SHIB", "SOL"]})
    assert s.config["assets"] == ["SOL"]


def test_default_config_ships_sol_and_xrp_only():
    """Per the report, BTC and ETH midcycle were unprofitable."""
    s = Crypto5mMidcycleStrategy()
    s.configure({})
    assert s.config["assets"] == ["SOL", "XRP"]


# ---------------------------------------------------------------------------
# Filter: timeframe
# ---------------------------------------------------------------------------


def test_skipped_when_timeframe_is_not_5min(strategy, fresh_cache):
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.55)
    market = _build_market_dict(timeframe="15min")
    assert strategy._evaluate_market(market, now_ms=MIDCYCLE_MS) is None


# ---------------------------------------------------------------------------
# Filter: asset enable list
# ---------------------------------------------------------------------------


def test_skipped_when_asset_not_in_enabled_list(strategy, fresh_cache):
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.55)
    # SOL+XRP enabled by default; BTC is NOT.
    market = _build_market_dict(asset="BTC")
    assert strategy._evaluate_market(market, now_ms=MIDCYCLE_MS) is None


def test_fires_for_btc_when_user_enables_it(fresh_cache):
    s = Crypto5mMidcycleStrategy()
    s.configure({"assets": ["BTC", "SOL", "XRP"]})
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.55)
    market = _build_market_dict(asset="BTC")
    opp = s._evaluate_market(market, now_ms=MIDCYCLE_MS)
    assert opp is not None


# ---------------------------------------------------------------------------
# Filter: midcycle milestone
# ---------------------------------------------------------------------------


def test_skipped_before_midcycle(strategy, fresh_cache):
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.55)
    market = _build_market_dict()
    # 1s before the midcycle mark
    assert strategy._evaluate_market(market, now_ms=MIDCYCLE_MS - 1_000) is None


def test_fires_at_midcycle(strategy, fresh_cache):
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.55)
    market = _build_market_dict()
    opp = strategy._evaluate_market(market, now_ms=MIDCYCLE_MS)
    assert opp is not None


def test_idempotent_within_same_cycle(strategy, fresh_cache):
    """The CycleTracker fires the milestone once; second call returns None."""
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.55)
    market = _build_market_dict()
    first = strategy._evaluate_market(market, now_ms=MIDCYCLE_MS)
    second = strategy._evaluate_market(market, now_ms=MIDCYCLE_MS + 5_000)
    assert first is not None
    assert second is None


def test_fires_again_on_next_cycle(strategy, fresh_cache):
    """When end_ts_ms changes (new cycle), fired-set resets automatically."""
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.55)
    market_a = _build_market_dict(end_ms=END_MS)
    market_b = _build_market_dict(end_ms=END_MS + CYCLE_MS)  # next cycle
    first = strategy._evaluate_market(market_a, now_ms=MIDCYCLE_MS)
    second = strategy._evaluate_market(
        market_b, now_ms=MIDCYCLE_MS + CYCLE_MS
    )
    assert first is not None
    assert second is not None


# ---------------------------------------------------------------------------
# Filter: minimum distance from reference
# ---------------------------------------------------------------------------


def test_skipped_when_distance_below_threshold(strategy, fresh_cache):
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.55)
    # spot 100.10 vs ref 100.0 = 10 bps; default threshold is 15 bps.
    market = _build_market_dict(reference=100.0, spot=100.10)
    assert strategy._evaluate_market(market, now_ms=MIDCYCLE_MS) is None


def test_fires_when_distance_at_threshold(strategy, fresh_cache):
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.55)
    # spot 100.15 vs ref 100.0 = 15 bps exactly.
    market = _build_market_dict(reference=100.0, spot=100.15)
    opp = strategy._evaluate_market(market, now_ms=MIDCYCLE_MS)
    assert opp is not None


def test_user_can_tighten_distance_threshold(fresh_cache):
    s = Crypto5mMidcycleStrategy()
    s.configure({"min_distance_bps": 30.0})
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.55)
    # 15 bps move — below the user's tightened 30-bp threshold
    market = _build_market_dict(reference=100.0, spot=100.15)
    assert s._evaluate_market(market, now_ms=MIDCYCLE_MS) is None


def test_user_can_loosen_distance_threshold(fresh_cache):
    """User can revert to the pre-2026-05-05 5-bp threshold via UI."""
    s = Crypto5mMidcycleStrategy()
    s.configure({"min_distance_bps": 5.0})
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.55)
    market = _build_market_dict(reference=100.0, spot=100.06)  # 6 bps
    opp = s._evaluate_market(market, now_ms=MIDCYCLE_MS)
    assert opp is not None


# ---------------------------------------------------------------------------
# Filter: side selection
# ---------------------------------------------------------------------------


def test_picks_yes_when_spot_above_reference(strategy, fresh_cache):
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.55)
    # 20 bps above (clears 15-bp default threshold).
    market = _build_market_dict(reference=100.0, spot=100.20)
    opp = strategy._evaluate_market(market, now_ms=MIDCYCLE_MS)
    assert opp is not None
    assert opp.strategy_context["side"] == "YES"


def test_picks_no_when_spot_below_reference(strategy, fresh_cache):
    _seed_book(fresh_cache, NO_TOKEN, ask_price=0.55)
    # 20 bps below (clears 15-bp default threshold).
    market = _build_market_dict(reference=100.0, spot=99.80)
    opp = strategy._evaluate_market(market, now_ms=MIDCYCLE_MS)
    assert opp is not None
    assert opp.strategy_context["side"] == "NO"


# ---------------------------------------------------------------------------
# Filter: oracle freshness
# ---------------------------------------------------------------------------


def test_skipped_when_oracle_too_stale(strategy, fresh_cache):
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.55)
    market = _build_market_dict(oracle_age_ms=10_000)  # 10s, default cap is 5s
    assert strategy._evaluate_market(market, now_ms=MIDCYCLE_MS) is None


def test_skipped_when_oracle_source_is_not_chainlink(strategy, fresh_cache):
    """Polymarket resolves on Chainlink — strategy must reject other sources."""
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.55)
    market = _build_market_dict(oracle_source="binance_direct")
    assert strategy._evaluate_market(market, now_ms=MIDCYCLE_MS) is None


# ---------------------------------------------------------------------------
# Filter: VWAP entry price
# ---------------------------------------------------------------------------


def test_skipped_when_vwap_above_max_entry_price(strategy, fresh_cache):
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.85)  # default cap is 0.70
    market = _build_market_dict()
    assert strategy._evaluate_market(market, now_ms=MIDCYCLE_MS) is None


def test_skipped_when_vwap_below_min_entry_price(strategy, fresh_cache):
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.02)  # default floor is 0.05
    market = _build_market_dict()
    assert strategy._evaluate_market(market, now_ms=MIDCYCLE_MS) is None


def test_skipped_when_book_unavailable(strategy, fresh_cache):
    # No seeding — book is empty → get_order_book_depth returns None
    market = _build_market_dict()
    assert strategy._evaluate_market(market, now_ms=MIDCYCLE_MS) is None


def test_user_can_raise_max_entry_price(fresh_cache):
    s = Crypto5mMidcycleStrategy()
    s.configure({"max_entry_price": 0.95})
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.85)
    market = _build_market_dict()
    opp = s._evaluate_market(market, now_ms=MIDCYCLE_MS)
    assert opp is not None


# ---------------------------------------------------------------------------
# Filter: min seconds to resolution
# ---------------------------------------------------------------------------


def test_skipped_when_too_close_to_resolution(strategy, fresh_cache):
    """Belt-and-suspenders against firing the milestone late."""
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.55)
    market = _build_market_dict()
    # Crossing midcycle for the first time, but with only 30s left.
    # default min_seconds_to_resolution = 90.0
    now_ms = END_MS - 30_000
    assert strategy._evaluate_market(market, now_ms=now_ms) is None


# ---------------------------------------------------------------------------
# Master switch
# ---------------------------------------------------------------------------


def test_disabled_strategy_emits_nothing(fresh_cache):
    s = Crypto5mMidcycleStrategy()
    s.configure({"enabled": False})
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.55)
    market = _build_market_dict()
    # Master switch is checked at on_event level; _evaluate_market itself
    # doesn't gate on it, so verify via on_event.
    import asyncio
    from services.data_events import DataEvent
    from datetime import datetime, timezone

    event = DataEvent(
        event_type="crypto_update",
        source="test",
        timestamp=datetime.now(timezone.utc),
        payload={"markets": [market]},
    )
    result = asyncio.run(s.on_event(event))
    assert result == []


# ---------------------------------------------------------------------------
# Happy path — full opportunity payload
# ---------------------------------------------------------------------------


def test_happy_path_opportunity_carries_full_context(strategy, fresh_cache):
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.55)
    # 20 bps above ref — passes the 2026-05-05-bumped 15-bp threshold.
    market = _build_market_dict(reference=100.0, spot=100.20, oracle_age_ms=250.0)

    opp = strategy._evaluate_market(market, now_ms=MIDCYCLE_MS)
    assert opp is not None

    ctx = opp.strategy_context
    assert ctx["strategy"] == "crypto_5m_midcycle"
    assert ctx["asset"] == "SOL"
    assert ctx["timeframe"] == "5min"
    assert ctx["side"] == "YES"
    assert ctx["reference_price"] == pytest.approx(100.0)
    assert ctx["spot_price"] == pytest.approx(100.20)
    assert ctx["distance_bps"] == pytest.approx(20.0, rel=1e-3)
    assert ctx["oracle_source"] == "chainlink"
    assert ctx["bet_size_usd"] == pytest.approx(15.0)
