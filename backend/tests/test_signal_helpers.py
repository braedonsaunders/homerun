from types import SimpleNamespace

from utils.signal_helpers import signal_payload, signal_strategy_context, weather_signal_context


def test_signal_payload_merges_strategy_context_and_firehose_fields():
    signal = SimpleNamespace(
        payload_json={
            "strategy_context": {
                "confluence_strength": 0.63,
                "firehose": {
                    "strength": 0.63,
                    "conviction_score": 63.0,
                    "signal_type": "multi_wallet_buy",
                },
            }
        },
        strategy_context_json={
            "source_key": "traders",
            "strategy_slug": "traders_confluence",
        },
    )

    payload = signal_payload(signal)

    assert payload["source_key"] == "traders"
    assert payload["strategy_slug"] == "traders_confluence"
    assert payload["strength"] == 0.63
    assert payload["conviction_score"] == 63.0


def test_signal_payload_extracts_crypto_context_and_market_dimensions():
    signal = SimpleNamespace(
        payload_json={
            "markets": [
                {
                    "slug": "btc-updown-5m-1771709700",
                    "question": "Bitcoin Up or Down - 5m",
                }
            ],
            "positions_to_take": [
                {
                    "action": "BUY",
                    "outcome": "YES",
                    "price": 0.41,
                    "_crypto_context": {
                        "strategy_origin": "crypto_worker",
                        "selected_direction": "buy_yes",
                    },
                }
            ],
        },
        strategy_context_json={},
    )

    payload = signal_payload(signal)

    assert payload["strategy_origin"] == "crypto_worker"
    assert payload["asset"] == "BTC"
    assert payload["timeframe"] == "5m"


def test_weather_signal_context_reads_strategy_context_weather():
    signal = SimpleNamespace(
        payload_json={},
        strategy_context_json={
            "source_key": "weather",
            "strategy_slug": "weather_distribution",
            "weather": {
                "agreement": 0.74,
                "source_count": 31,
                "source_spread_c": 1.5,
            },
        },
    )

    weather = weather_signal_context(signal)

    assert weather["agreement"] == 0.74
    assert weather["source_count"] == 31
    assert weather["source_spread_c"] == 1.5


def test_weather_signal_context_does_not_read_position_payload_metadata():
    signal = SimpleNamespace(
        payload_json={
            "positions_to_take": [
                {
                    "_weather_distribution": {
                        "consensus_temp_c": 11.2,
                        "market_implied_temp_c": 9.1,
                        "model_agreement": 0.67,
                        "source_count": 3,
                        "source_spread_c": 1.8,
                    }
                }
            ]
        },
        strategy_context_json={},
    )

    weather = weather_signal_context(signal)

    assert weather == {}


def test_signal_strategy_context_falls_back_to_payload_strategy_context():
    signal = SimpleNamespace(
        payload_json={
            "strategy_context": {
                "source_key": "weather",
                "weather": {
                    "model_agreement": 0.73,
                    "source_count": 2,
                    "source_spread_c": 2.2,
                },
            }
        },
        strategy_context_json=None,
    )

    context = signal_strategy_context(signal)
    weather = weather_signal_context(signal)

    assert context["source_key"] == "weather"
    assert weather["model_agreement"] == 0.73
    assert weather["source_count"] == 2
    assert weather["source_spread_c"] == 2.2
