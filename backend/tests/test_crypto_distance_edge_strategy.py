"""Decision-gate tests for the crypto_distance_edge strategy.

Each test exercises one filter in ``CryptoDistanceEdgeStrategy._evaluate_market``
in isolation — wrong timeframe, asset not in config, distance below the
smallest tier, cost below the tier floor / above the ceiling, oracle
stale, one-emit-per-cycle — plus the happy path where every gate passes
and an Opportunity is emitted.

Tests seed the WS-fed PriceCache directly with synthetic order books (see
``test_crypto_5m_midcycle_strategy.py`` for the same pattern) so the
``StrategySDK.get_order_book_depth`` call returns deterministic results
without any real Polymarket connectivity.

The default BTC tiers are: $150→90¢, $200→85¢, $300→80¢. Tests anchor a
BTC reference of $100,000 and move spot in dollars to land in specific
tiers.
"""

from __future__ import annotations

import pytest

from services.optimization.vwap import OrderBookLevel
from services.strategies.crypto_distance_edge import (
    CryptoDistanceEdgeStrategy,
    crypto_distance_edge_config_schema,
)
from services.ws_feeds import FeedManager, get_feed_manager


# A 15-minute cycle ending at this fixed UTC timestamp (epoch millis).
END_MS = 2_000_000_000_000  # arbitrary far-future ms
CYCLE_MS = 900_000
# Comfortably mid-cycle, well above the 10s min-seconds-to-resolution.
NOW_MS = END_MS - 300_000

# CLOB token IDs are 50+ char hex strings on Polymarket — match length.
YES_TOKEN = "0x" + "a" * 60
NO_TOKEN = "0x" + "b" * 60

BTC_REF = 100_000.0


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
    s = CryptoDistanceEdgeStrategy()
    s.configure({})  # use defaults (BTC only)
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
    asset: str = "BTC",
    timeframe: str = "15min",
    end_ms: int = END_MS,
    reference: float = BTC_REF,
    spot: float = BTC_REF + 250.0,  # $250 above → clears $200 tier (85¢)
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
        "up_price": 0.88,
        "down_price": 0.12,
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
    schema = crypto_distance_edge_config_schema()
    keys = {f["key"] for f in schema["param_fields"]}
    assert {
        "enabled",
        "assets",
        "distance_cost_tiers",
        "max_cost_cents",
        "bet_size_usd",
        "min_seconds_to_resolution",
        "max_oracle_age_ms",
        "edge_uplift_pct",
    } <= keys


def test_default_config_ships_btc_only():
    """Distance tiers are in USD, calibrated for BTC's price scale."""
    s = CryptoDistanceEdgeStrategy()
    s.configure({})
    assert s.config["assets"] == ["BTC"]


def test_configure_sorts_tiers_ascending_by_distance():
    s = CryptoDistanceEdgeStrategy()
    s.configure({
        "distance_cost_tiers": [
            {"distance_usd": 300.0, "min_cost_cents": 80.0},
            {"distance_usd": 150.0, "min_cost_cents": 90.0},
        ]
    })
    distances = [t["distance_usd"] for t in s.config["distance_cost_tiers"]]
    assert distances == [150.0, 300.0]


def test_configure_drops_malformed_tiers_and_falls_back():
    s = CryptoDistanceEdgeStrategy()
    s.configure({"distance_cost_tiers": ["nonsense", {"distance_usd": "x"}]})
    # All entries invalid → fall back to the shipped defaults.
    assert s.config["distance_cost_tiers"][0]["distance_usd"] == 150.0


# ---------------------------------------------------------------------------
# Filter: timeframe
# ---------------------------------------------------------------------------


def test_skipped_when_timeframe_is_not_15min(strategy, fresh_cache):
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.88)
    market = _build_market_dict(timeframe="5min")
    assert strategy._evaluate_market(market, now_ms=NOW_MS) is None


# ---------------------------------------------------------------------------
# Filter: asset enable list
# ---------------------------------------------------------------------------


def test_skipped_when_asset_not_in_enabled_list(strategy, fresh_cache):
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.88)
    market = _build_market_dict(asset="ETH")  # default enables BTC only
    assert strategy._evaluate_market(market, now_ms=NOW_MS) is None


# ---------------------------------------------------------------------------
# Filter: distance tier selection
# ---------------------------------------------------------------------------


def test_skipped_when_distance_below_smallest_tier(strategy, fresh_cache):
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.88)
    # $100 above ref — below the $150 smallest tier.
    market = _build_market_dict(spot=BTC_REF + 100.0)
    assert strategy._evaluate_market(market, now_ms=NOW_MS) is None


def test_fires_at_smallest_tier_with_sufficient_cost(strategy, fresh_cache):
    # $150 distance → 90¢ floor. Seed cost at 90¢.
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.90)
    market = _build_market_dict(spot=BTC_REF + 150.0)
    opp = strategy._evaluate_market(market, now_ms=NOW_MS)
    assert opp is not None
    assert opp.strategy_context["tier_distance_usd"] == pytest.approx(150.0)


def test_selects_highest_cleared_tier(strategy, fresh_cache):
    # $250 distance clears $150 and $200 but not $300 → $200 tier (85¢).
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.86)
    market = _build_market_dict(spot=BTC_REF + 250.0)
    opp = strategy._evaluate_market(market, now_ms=NOW_MS)
    assert opp is not None
    assert opp.strategy_context["tier_distance_usd"] == pytest.approx(200.0)
    assert opp.strategy_context["tier_min_cost_cents"] == pytest.approx(85.0)


def test_largest_distance_uses_loosest_cost_floor(strategy, fresh_cache):
    # $350 distance → $300 tier → 80¢ floor. 82¢ clears it.
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.82)
    market = _build_market_dict(spot=BTC_REF + 350.0)
    opp = strategy._evaluate_market(market, now_ms=NOW_MS)
    assert opp is not None
    assert opp.strategy_context["tier_min_cost_cents"] == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# Filter: cost floor / ceiling
# ---------------------------------------------------------------------------


def test_skipped_when_cost_below_tier_floor(strategy, fresh_cache):
    # $250 distance → 85¢ floor; 84¢ is below it.
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.84)
    market = _build_market_dict(spot=BTC_REF + 250.0)
    assert strategy._evaluate_market(market, now_ms=NOW_MS) is None


def test_skipped_when_cost_above_ceiling(fresh_cache):
    s = CryptoDistanceEdgeStrategy()
    s.configure({"max_cost_cents": 95.0})
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.97)  # 97¢ > 95¢ ceiling
    market = _build_market_dict(spot=BTC_REF + 350.0)  # clears $300/80¢ floor
    assert s._evaluate_market(market, now_ms=NOW_MS) is None


# ---------------------------------------------------------------------------
# Filter: side selection
# ---------------------------------------------------------------------------


def test_picks_no_when_spot_below_reference(strategy, fresh_cache):
    # $250 below ref → favored side is NO/DOWN. Seed the NO book.
    _seed_book(fresh_cache, NO_TOKEN, ask_price=0.86)
    market = _build_market_dict(spot=BTC_REF - 250.0)
    opp = strategy._evaluate_market(market, now_ms=NOW_MS)
    assert opp is not None
    assert opp.strategy_context["side"] == "NO"


# ---------------------------------------------------------------------------
# Filter: oracle freshness
# ---------------------------------------------------------------------------


def test_skipped_when_oracle_too_stale(strategy, fresh_cache):
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.86)
    market = _build_market_dict(oracle_age_ms=10_000)  # default cap is 5s
    assert strategy._evaluate_market(market, now_ms=NOW_MS) is None


def test_skipped_when_oracle_source_not_chainlink(strategy, fresh_cache):
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.86)
    market = _build_market_dict(oracle_source="binance_direct")
    assert strategy._evaluate_market(market, now_ms=NOW_MS) is None


# ---------------------------------------------------------------------------
# Filter: min seconds to resolution
# ---------------------------------------------------------------------------


def test_skipped_when_too_close_to_resolution(strategy, fresh_cache):
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.86)
    market = _build_market_dict()
    now_ms = END_MS - 5_000  # only 5s left; default floor is 10s
    assert strategy._evaluate_market(market, now_ms=now_ms) is None


# ---------------------------------------------------------------------------
# Filter: one emit per cycle
# ---------------------------------------------------------------------------


def test_idempotent_within_same_cycle(strategy, fresh_cache):
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.86)
    market = _build_market_dict(spot=BTC_REF + 250.0)
    first = strategy._evaluate_market(market, now_ms=NOW_MS)
    second = strategy._evaluate_market(market, now_ms=NOW_MS + 5_000)
    assert first is not None
    assert second is None


def test_fires_again_on_next_cycle(strategy, fresh_cache):
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.86)
    market_a = _build_market_dict(end_ms=END_MS, spot=BTC_REF + 250.0)
    market_b = _build_market_dict(end_ms=END_MS + CYCLE_MS, spot=BTC_REF + 250.0)
    first = strategy._evaluate_market(market_a, now_ms=NOW_MS)
    second = strategy._evaluate_market(market_b, now_ms=NOW_MS + CYCLE_MS)
    assert first is not None
    assert second is not None


# ---------------------------------------------------------------------------
# Master switch
# ---------------------------------------------------------------------------


def test_disabled_strategy_emits_nothing(fresh_cache):
    import asyncio
    from datetime import datetime, timezone
    from services.data_events import DataEvent

    s = CryptoDistanceEdgeStrategy()
    s.configure({"enabled": False})
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.86)
    market = _build_market_dict(spot=BTC_REF + 250.0)

    event = DataEvent(
        event_type="crypto_update",
        source="test",
        timestamp=datetime.now(timezone.utc),
        payload={"markets": [market]},
    )
    assert asyncio.run(s.on_event(event)) == []


# ---------------------------------------------------------------------------
# Happy path — full opportunity payload
# ---------------------------------------------------------------------------


def test_happy_path_opportunity_carries_full_context(strategy, fresh_cache):
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.86)
    market = _build_market_dict(spot=BTC_REF + 250.0, oracle_age_ms=250.0)

    opp = strategy._evaluate_market(market, now_ms=NOW_MS)
    assert opp is not None

    ctx = opp.strategy_context
    assert ctx["strategy"] == "crypto_distance_edge"
    assert ctx["asset"] == "BTC"
    assert ctx["timeframe"] == "15min"
    assert ctx["side"] == "YES"
    assert ctx["reference_price"] == pytest.approx(BTC_REF)
    assert ctx["distance_usd"] == pytest.approx(250.0)
    assert ctx["tier_distance_usd"] == pytest.approx(200.0)
    assert ctx["tier_min_cost_cents"] == pytest.approx(85.0)
    assert ctx["cost_cents"] == pytest.approx(86.0, abs=0.5)
    assert ctx["oracle_source"] == "chainlink"
    # Fees are always modeled — never assume fee-free.
    assert ctx["taker_fee_frac"] > 0.0


def test_edge_uplift_lifts_confidence_above_implied(fresh_cache):
    s = CryptoDistanceEdgeStrategy()
    s.configure({"edge_uplift_pct": 5.0})
    _seed_book(fresh_cache, YES_TOKEN, ask_price=0.86)
    market = _build_market_dict(spot=BTC_REF + 250.0)
    opp = s._evaluate_market(market, now_ms=NOW_MS)
    assert opp is not None
    # win prob = implied 0.86 + 0.05 uplift = 0.91
    assert opp.strategy_context["win_prob_estimate"] == pytest.approx(0.91, abs=1e-6)
