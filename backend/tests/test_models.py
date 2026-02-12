"""Tests for Pydantic models: Market, Event, Token, ArbitrageOpportunity, OpportunityFilter, StrategyType."""

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import json
from datetime import datetime

from models.market import Market, Event, Token
from models.opportunity import (
    ArbitrageOpportunity,
    StrategyType,
    MispricingType,
    OpportunityFilter,
)


# ============================================================================
# Market.from_gamma_response
# ============================================================================


class TestMarketFromGammaResponse:
    """Tests for Market.from_gamma_response."""

    def test_valid_data(self, raw_market_response):
        """Parsing a well-formed Gamma response yields correct fields."""
        market = Market.from_gamma_response(raw_market_response)

        assert market.id == "123456"
        assert market.condition_id == "0xabc123"
        assert market.question == "Will BTC exceed $100k by end of 2025?"
        assert market.slug == "will-btc-exceed-100k-2025"
        assert market.active is True
        assert market.closed is False
        assert market.neg_risk is False
        assert market.volume == 12345.67
        assert market.liquidity == 5000.0
        assert market.clob_token_ids == ["token_yes_1", "token_no_1"]
        assert market.outcome_prices == [0.65, 0.35]

    def test_tokens_built_from_clob_and_prices(self, raw_market_response):
        """Tokens list is constructed from clobTokenIds and outcomePrices."""
        market = Market.from_gamma_response(raw_market_response)

        assert len(market.tokens) == 2
        assert market.tokens[0].token_id == "token_yes_1"
        assert market.tokens[0].outcome == "Yes"
        assert market.tokens[0].price == 0.65
        assert market.tokens[1].token_id == "token_no_1"
        assert market.tokens[1].outcome == "No"
        assert market.tokens[1].price == 0.35

    def test_missing_fields_uses_defaults(self):
        """Missing optional fields fall back to defaults gracefully."""
        data = {
            "id": "42",
            "question": "Minimal market",
            "slug": "minimal",
        }
        market = Market.from_gamma_response(data)

        assert market.id == "42"
        assert market.condition_id == ""
        assert market.question == "Minimal market"
        assert market.slug == "minimal"
        assert market.active is True
        assert market.closed is False
        assert market.neg_risk is False
        assert market.volume == 0.0
        assert market.liquidity == 0.0
        assert market.clob_token_ids == []
        assert market.outcome_prices == []
        assert market.tokens == []

    def test_missing_id_defaults_to_empty(self):
        """Totally empty dict produces a market with empty strings."""
        market = Market.from_gamma_response({})
        assert market.id == ""
        assert market.question == ""

    def test_malformed_json_in_clobTokenIds(self):
        """Malformed JSON in clobTokenIds is handled without raising."""
        data = {
            "id": "bad_json",
            "question": "Bad json test",
            "slug": "bad-json",
            "clobTokenIds": "this is not valid json [[[",
            "outcomePrices": json.dumps(["0.5", "0.5"]),
        }
        market = Market.from_gamma_response(data)

        # clobTokenIds failed to parse, so it should be empty
        assert market.clob_token_ids == []
        # Tokens list should also be empty (built from clob_token_ids)
        assert market.tokens == []
        # outcomePrices should still parse fine
        assert market.outcome_prices == [0.5, 0.5]

    def test_malformed_json_in_outcomePrices(self):
        """Malformed JSON in outcomePrices is handled without raising."""
        data = {
            "id": "bad_prices",
            "question": "Bad prices",
            "slug": "bad-prices",
            "clobTokenIds": json.dumps(["tok1", "tok2"]),
            "outcomePrices": "NOT JSON",
        }
        market = Market.from_gamma_response(data)

        assert market.clob_token_ids == ["tok1", "tok2"]
        assert market.outcome_prices == []
        # Tokens are built but with price 0.0 since no outcome prices
        assert len(market.tokens) == 2
        assert market.tokens[0].price == 0.0
        assert market.tokens[1].price == 0.0

    def test_clob_token_ids_accepts_list_payload(self):
        """Gamma sometimes returns clobTokenIds as a native array."""
        data = {
            "id": "list_tokens",
            "question": "List token payload",
            "slug": "list-token-payload",
            "clobTokenIds": ["yes_tok", "no_tok"],
            "outcomePrices": json.dumps(["0.5", "0.5"]),
        }
        market = Market.from_gamma_response(data)
        assert market.clob_token_ids == ["yes_tok", "no_tok"]
        assert len(market.tokens) == 2

    def test_outcome_prices_accepts_list_payload(self):
        """Gamma sometimes returns outcomePrices as a native array."""
        data = {
            "id": "list_prices",
            "question": "List price payload",
            "slug": "list-price-payload",
            "clobTokenIds": json.dumps(["yes_tok", "no_tok"]),
            "outcomePrices": [0.42, 0.58],
        }
        market = Market.from_gamma_response(data)
        assert market.outcome_prices == [0.42, 0.58]
        assert market.tokens[0].price == 0.42
        assert market.tokens[1].price == 0.58

    def test_condition_id_fallback_to_conditionId(self):
        """condition_id is read from conditionId key as fallback."""
        data = {
            "id": "1",
            "conditionId": "0xfallback",
            "question": "Fallback test",
            "slug": "fallback",
        }
        market = Market.from_gamma_response(data)
        assert market.condition_id == "0xfallback"

    def test_neg_risk_alias(self):
        """negRisk and neg_risk keys are both supported."""
        data_camel = {"id": "1", "question": "q", "slug": "s", "negRisk": True}
        data_snake = {"id": "2", "question": "q", "slug": "s", "neg_risk": True}

        assert Market.from_gamma_response(data_camel).neg_risk is True
        assert Market.from_gamma_response(data_snake).neg_risk is True

    def test_volume_none_coerced_to_zero(self):
        """volume=None should not crash and should coerce to 0."""
        data = {"id": "1", "question": "q", "slug": "s", "volume": None}
        market = Market.from_gamma_response(data)
        assert market.volume == 0.0

    def test_liquidity_none_coerced_to_zero(self):
        """liquidity=None should not crash and should coerce to 0."""
        data = {"id": "1", "question": "q", "slug": "s", "liquidity": None}
        market = Market.from_gamma_response(data)
        assert market.liquidity == 0.0


# ============================================================================
# Market.yes_price / Market.no_price properties
# ============================================================================


class TestMarketPriceProperties:
    """Tests for Market.yes_price and Market.no_price properties."""

    def test_yes_price_returns_first_outcome(self, sample_market):
        assert sample_market.yes_price == 0.65

    def test_no_price_returns_second_outcome(self, sample_market):
        assert sample_market.no_price == 0.35

    def test_yes_price_empty_prices_returns_zero(self):
        market = Market(id="x", condition_id="x", question="q", slug="s")
        assert market.yes_price == 0.0

    def test_no_price_single_price_returns_zero(self):
        market = Market(
            id="x",
            condition_id="x",
            question="q",
            slug="s",
            outcome_prices=[0.7],
        )
        assert market.no_price == 0.0

    def test_yes_price_single_price_still_works(self):
        market = Market(
            id="x",
            condition_id="x",
            question="q",
            slug="s",
            outcome_prices=[0.7],
        )
        assert market.yes_price == 0.7


# ============================================================================
# Event.from_gamma_response
# ============================================================================


class TestEventFromGammaResponse:
    """Tests for Event.from_gamma_response."""

    def test_valid_data(self, raw_event_response):
        event = Event.from_gamma_response(raw_event_response)

        assert event.id == "evt_001"
        assert event.slug == "btc-price-events"
        assert event.title == "Bitcoin Price Predictions"
        assert event.description == "All markets about BTC price targets."
        assert event.category == "Crypto"
        assert event.neg_risk is False
        assert event.active is True
        assert event.closed is False

    def test_nested_market_parsing(self, raw_event_response):
        """Nested markets dicts are parsed into Market objects."""
        event = Event.from_gamma_response(raw_event_response)

        assert len(event.markets) == 2
        assert event.markets[0].id == "123456"
        assert event.markets[1].id == "789"
        assert event.markets[1].neg_risk is True

    def test_category_from_category_string(self):
        """Category field as a plain string is used directly."""
        data = {
            "id": "e1",
            "slug": "s",
            "title": "T",
            "category": "Sports",
        }
        event = Event.from_gamma_response(data)
        assert event.category == "Sports"

    def test_category_from_category_dict(self):
        """Category as a dict {label, name} extracts label first."""
        data = {
            "id": "e2",
            "slug": "s",
            "title": "T",
            "category": {"id": 1, "label": "Politics", "name": "politics"},
        }
        event = Event.from_gamma_response(data)
        assert event.category == "Politics"

    def test_category_from_category_dict_fallback_to_name(self):
        """If dict has no label, falls back to name."""
        data = {
            "id": "e3",
            "slug": "s",
            "title": "T",
            "category": {"id": 1, "name": "tech"},
        }
        event = Event.from_gamma_response(data)
        assert event.category == "tech"

    def test_category_from_tags_list_of_strings(self):
        """tags as a list of strings, first tag used as category."""
        data = {
            "id": "e4",
            "slug": "s",
            "title": "T",
            "tags": ["Entertainment", "Music"],
        }
        event = Event.from_gamma_response(data)
        assert event.category == "Entertainment"

    def test_category_from_tags_list_of_dicts(self):
        """tags as a list of dicts, first tag label used."""
        data = {
            "id": "e5",
            "slug": "s",
            "title": "T",
            "tags": [{"label": "Finance", "name": "finance"}],
        }
        event = Event.from_gamma_response(data)
        assert event.category == "Finance"

    def test_category_from_tags_string(self):
        """tags as a plain string is used directly."""
        data = {
            "id": "e6",
            "slug": "s",
            "title": "T",
            "tags": "Weather",
        }
        event = Event.from_gamma_response(data)
        assert event.category == "Weather"

    def test_category_none_when_no_category_or_tags(self):
        """No category or tags results in None."""
        data = {"id": "e7", "slug": "s", "title": "T"}
        event = Event.from_gamma_response(data)
        assert event.category is None

    def test_category_field_preferred_over_tags(self):
        """When both category and tags exist, category field wins."""
        data = {
            "id": "e8",
            "slug": "s",
            "title": "T",
            "category": "Primary",
            "tags": ["Secondary"],
        }
        event = Event.from_gamma_response(data)
        assert event.category == "Primary"

    def test_bad_nested_market_is_skipped(self):
        """A malformed nested market dict is silently skipped."""
        data = {
            "id": "e9",
            "slug": "s",
            "title": "T",
            "markets": [
                {"id": "good1", "question": "Q?", "slug": "q"},
                "not_a_dict",  # This will cause an exception and be skipped
            ],
        }
        event = Event.from_gamma_response(data)
        # At least the first valid market is parsed
        assert len(event.markets) >= 1

    def test_empty_markets_list(self):
        data = {"id": "e10", "slug": "s", "title": "T", "markets": []}
        event = Event.from_gamma_response(data)
        assert event.markets == []


# ============================================================================
# Token model
# ============================================================================


class TestToken:
    """Tests for the Token model."""

    def test_creation(self, sample_token):
        assert sample_token.token_id == "tok_123"
        assert sample_token.outcome == "Yes"
        assert sample_token.price == 0.65

    def test_default_price(self):
        token = Token(token_id="t", outcome="No")
        assert token.price == 0.0

    def test_serialization_roundtrip(self, sample_token):
        data = sample_token.model_dump()
        restored = Token(**data)
        assert restored.token_id == sample_token.token_id
        assert restored.outcome == sample_token.outcome
        assert restored.price == sample_token.price


# ============================================================================
# ArbitrageOpportunity
# ============================================================================


class TestArbitrageOpportunity:
    """Tests for ArbitrageOpportunity model."""

    def test_id_generation_from_strategy_and_markets(self):
        """ID is auto-generated from strategy + market IDs + timestamp."""
        opp = ArbitrageOpportunity(
            strategy=StrategyType.BASIC,
            title="Test",
            description="Desc",
            total_cost=0.95,
            gross_profit=0.05,
            fee=0.02,
            net_profit=0.03,
            roi_percent=3.16,
            markets=[{"id": "abcdefghijklmnop"}, {"id": "1234567890"}],
        )
        assert opp.id.startswith("basic_")
        # Should contain truncated market IDs
        assert "abcdefgh" in opp.id
        assert "12345678" in opp.id

    def test_id_generation_empty_markets(self):
        """ID with no markets still generates without error."""
        opp = ArbitrageOpportunity(
            strategy=StrategyType.NEGRISK,
            title="Test",
            description="Desc",
            total_cost=0.9,
            gross_profit=0.1,
            fee=0.02,
            net_profit=0.08,
            roi_percent=8.89,
            markets=[],
        )
        assert opp.id.startswith("negrisk_")

    def test_explicit_id_not_overwritten(self):
        """If an explicit id is provided, it is kept."""
        opp = ArbitrageOpportunity(
            id="custom_id_123",
            strategy=StrategyType.BASIC,
            title="Test",
            description="Desc",
            total_cost=0.95,
            gross_profit=0.05,
            fee=0.02,
            net_profit=0.03,
            roi_percent=3.16,
            markets=[],
        )
        assert opp.id == "custom_id_123"

    def test_all_fields_populated(self, sample_opportunity):
        """All fields of a fully-populated opportunity are accessible."""
        opp = sample_opportunity
        assert opp.strategy == StrategyType.BASIC
        assert opp.title == "Basic Arb: Will BTC exceed $100k..."
        assert opp.total_cost == 0.96
        assert opp.expected_payout == 1.0
        assert opp.gross_profit == 0.04
        assert opp.fee == 0.02
        assert opp.net_profit == 0.02
        assert opp.roi_percent == 2.08
        assert opp.risk_score == 0.3
        assert len(opp.risk_factors) == 1
        assert len(opp.markets) == 1
        assert opp.event_id == "evt_001"
        assert opp.event_title == "Bitcoin Price Predictions"
        assert opp.category == "Crypto"
        assert opp.min_liquidity == 5000.0
        assert opp.max_position_size == 500.0
        assert opp.mispricing_type == MispricingType.WITHIN_MARKET
        assert isinstance(opp.detected_at, datetime)

    def test_default_values(self):
        """Default values for optional fields are sensible."""
        opp = ArbitrageOpportunity(
            strategy=StrategyType.BASIC,
            title="T",
            description="D",
            total_cost=0.9,
            gross_profit=0.1,
            fee=0.02,
            net_profit=0.08,
            roi_percent=8.89,
        )
        assert opp.expected_payout == 1.0
        assert opp.risk_score == 0.5
        assert opp.risk_factors == []
        assert opp.markets == []
        assert opp.event_id is None
        assert opp.event_title is None
        assert opp.category is None
        assert opp.min_liquidity == 0.0
        assert opp.max_position_size == 0.0
        assert opp.resolution_date is None
        assert opp.mispricing_type is None
        assert opp.guaranteed_profit is None
        assert opp.capture_ratio is None
        assert opp.positions_to_take == []

    def test_mispricing_type_values(self):
        """MispricingType enum has the three expected values."""
        assert MispricingType.WITHIN_MARKET.value == "within_market"
        assert MispricingType.CROSS_MARKET.value == "cross_market"
        assert MispricingType.SETTLEMENT_LAG.value == "settlement_lag"


# ============================================================================
# StrategyType enum
# ============================================================================


class TestStrategyType:
    """Tests for StrategyType enum values."""

    def test_all_strategy_types_present(self):
        expected = {
            "basic",
            "mutually_exclusive",
            "contradiction",
            "negrisk",
            "must_happen",
            "miracle",
            "combinatorial",
            "settlement_lag",
            "cross_platform",
        }
        actual = {s.value for s in StrategyType}
        assert actual == expected

    def test_string_enum_comparison(self):
        assert StrategyType.BASIC == "basic"
        assert StrategyType.NEGRISK == "negrisk"
        assert StrategyType.COMBINATORIAL == "combinatorial"
        assert StrategyType.SETTLEMENT_LAG == "settlement_lag"

    def test_enum_from_value(self):
        assert StrategyType("basic") == StrategyType.BASIC
        assert StrategyType("miracle") == StrategyType.MIRACLE


# ============================================================================
# OpportunityFilter
# ============================================================================


class TestOpportunityFilter:
    """Tests for OpportunityFilter model."""

    def test_default_values(self):
        f = OpportunityFilter()
        assert f.min_profit == 0.0
        assert f.max_risk == 1.0
        assert f.strategies == []
        assert f.min_liquidity == 0.0
        assert f.category is None

    def test_custom_values(self):
        f = OpportunityFilter(
            min_profit=0.05,
            max_risk=0.5,
            strategies=[StrategyType.BASIC, StrategyType.NEGRISK],
            min_liquidity=1000.0,
            category="Crypto",
        )
        assert f.min_profit == 0.05
        assert f.max_risk == 0.5
        assert len(f.strategies) == 2
        assert f.min_liquidity == 1000.0
        assert f.category == "Crypto"

    def test_filtering_logic_in_scanner_is_consistent(self):
        """OpportunityFilter is a data container; actual filtering is in scanner.
        This test validates the filter can be constructed with all parameters.
        """
        f = OpportunityFilter(
            min_profit=0.01,
            max_risk=0.9,
            strategies=[StrategyType.MIRACLE],
            min_liquidity=500.0,
            category="Science",
        )
        assert f.model_dump() == {
            "min_profit": 0.01,
            "max_risk": 0.9,
            "strategies": [StrategyType.MIRACLE],
            "min_liquidity": 500.0,
            "category": "Science",
        }
