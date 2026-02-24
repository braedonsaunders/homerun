from services.strategies.traders_confluence import TradersConfluenceStrategy


def _base_signal(**overrides):
    signal = {
        "id": "signal-1",
        "market_id": "0xmarket",
        "market_question": "Jazz vs Rockets",
        "market_slug": "jazz-vs-rockets",
        "signal_type": "multi_wallet_buy",
        "side": "all",
        "outcome": "YES",
        "tier": "high",
        "strength": 0.72,
        "wallet_count": 3,
        "cluster_adjusted_wallet_count": 3,
        "edge_percent": 6.2,
        "avg_entry_price": 0.42,
        "yes_price": 0.42,
        "no_price": 0.58,
        "market_liquidity": 50000.0,
        "market_volume_24h": 100000.0,
        "is_active": True,
        "is_tradeable": True,
        "wallets": ["0x1", "0x2", "0x3"],
        "source_flags": {"qualified": True, "from_pool": True},
    }
    signal.update(overrides)
    return signal


def test_explicit_yes_outcome_overrides_ambiguous_side():
    strategy = TradersConfluenceStrategy()
    opportunities = strategy.build_opportunities_from_firehose(
        [
            _base_signal(
                id="signal-yes",
                side="all",
                outcome="YES",
                signal_type="multi_wallet_sell",
            )
        ]
    )

    assert len(opportunities) == 1
    position = opportunities[0].positions_to_take[0]
    assert position["outcome"] == "YES"
    assert position["direction"] == "buy_yes"


def test_explicit_no_outcome_overrides_buy_side_hint():
    strategy = TradersConfluenceStrategy()
    opportunities = strategy.build_opportunities_from_firehose(
        [
            _base_signal(
                id="signal-no",
                side="buy",
                outcome="NO",
                signal_type="multi_wallet_buy",
                avg_entry_price=0.36,
                yes_price=0.64,
                no_price=0.36,
            )
        ]
    )

    assert len(opportunities) == 1
    position = opportunities[0].positions_to_take[0]
    assert position["outcome"] == "NO"
    assert position["direction"] == "buy_no"
