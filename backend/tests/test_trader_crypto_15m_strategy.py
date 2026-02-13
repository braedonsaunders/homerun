import sys
from pathlib import Path
from types import SimpleNamespace

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.trader_orchestrator.strategies.crypto_15m import Crypto15mStrategy


def _base_payload() -> dict:
    return {
        "strategy_origin": "crypto_worker",
        "regime": "mid",
        "dominant_strategy": "directional",
        "component_edges": {
            "buy_yes": {"directional": 7.0, "pure_arb": 1.2, "rebalance": 0.8},
            "buy_no": {"directional": 0.2, "pure_arb": 1.2, "rebalance": 2.0},
        },
        "net_edges": {"buy_yes": 6.1, "buy_no": 0.6},
    }


def test_crypto_15m_strategy_normalizes_percent_min_confidence():
    strategy = Crypto15mStrategy()
    signal = SimpleNamespace(
        source="crypto",
        signal_type="crypto_worker_multistrat",
        edge_percent=6.1,
        confidence=0.62,
        direction="buy_yes",
        payload_json=_base_payload(),
    )

    decision = strategy.evaluate(
        signal,
        {
            "params": {
                "min_edge_percent": 3.0,
                "min_confidence": 45,
                "base_size_usd": 25.0,
            }
        },
    )

    assert decision.decision == "selected"
    assert decision.payload["required_confidence"] == 0.45


def test_crypto_15m_strategy_rejects_non_worker_crypto_signals():
    strategy = Crypto15mStrategy()
    signal = SimpleNamespace(
        source="crypto",
        signal_type="crypto_5m_15m_hf",
        edge_percent=9.0,
        confidence=0.8,
        direction="buy_yes",
        payload_json={"regime": "mid"},
    )

    decision = strategy.evaluate(signal, {"params": {"min_edge_percent": 2.0, "min_confidence": 0.3}})

    assert decision.decision == "skipped"
    origin_check = next(check for check in decision.checks if check.key == "signal_origin")
    assert origin_check.passed is False


def test_crypto_15m_strategy_supports_explicit_mode_selection():
    strategy = Crypto15mStrategy()
    payload = _base_payload()
    payload["dominant_strategy"] = "pure_arb"
    payload["component_edges"]["buy_yes"]["directional"] = 0.4
    payload["component_edges"]["buy_yes"]["pure_arb"] = 4.6
    payload["net_edges"]["buy_yes"] = 3.0
    signal = SimpleNamespace(
        source="crypto",
        signal_type="crypto_worker_multistrat",
        edge_percent=3.0,
        confidence=0.55,
        direction="buy_yes",
        payload_json=payload,
    )

    decision = strategy.evaluate(
        signal,
        {
            "params": {
                "strategy_mode": "pure_arb",
                "min_edge_percent": 2.5,
                "min_confidence": 0.45,
                "base_size_usd": 20.0,
            }
        },
    )

    assert decision.decision == "selected"
    assert decision.payload["active_mode"] == "pure_arb"


def test_crypto_15m_strategy_filters_by_target_asset():
    strategy = Crypto15mStrategy()
    payload = _base_payload()
    payload["asset"] = "BTC"
    payload["timeframe"] = "15min"
    signal = SimpleNamespace(
        source="crypto",
        signal_type="crypto_worker_multistrat",
        edge_percent=6.1,
        confidence=0.62,
        direction="buy_yes",
        payload_json=payload,
    )

    decision = strategy.evaluate(
        signal,
        {
            "params": {
                "min_edge_percent": 3.0,
                "min_confidence": 0.45,
                "target_assets": ["ETH"],
            }
        },
    )

    assert decision.decision == "skipped"
    asset_check = next(check for check in decision.checks if check.key == "asset_scope")
    assert asset_check.passed is False


def test_crypto_15m_strategy_filters_by_target_timeframe():
    strategy = Crypto15mStrategy()
    payload = _base_payload()
    payload["asset"] = "ETH"
    payload["timeframe"] = "15min"
    signal = SimpleNamespace(
        source="crypto",
        signal_type="crypto_worker_multistrat",
        edge_percent=6.1,
        confidence=0.62,
        direction="buy_yes",
        payload_json=payload,
    )

    decision = strategy.evaluate(
        signal,
        {
            "params": {
                "min_edge_percent": 3.0,
                "min_confidence": 0.45,
                "target_timeframes": ["1h"],
            }
        },
    )

    assert decision.decision == "skipped"
    timeframe_check = next(check for check in decision.checks if check.key == "timeframe_scope")
    assert timeframe_check.passed is False


def test_crypto_15m_strategy_allows_matching_target_scope():
    strategy = Crypto15mStrategy()
    payload = _base_payload()
    payload["asset"] = "XBT"
    payload["timeframe"] = "15min"
    signal = SimpleNamespace(
        source="crypto",
        signal_type="crypto_worker_multistrat",
        edge_percent=6.1,
        confidence=0.62,
        direction="buy_yes",
        payload_json=payload,
    )

    decision = strategy.evaluate(
        signal,
        {
            "params": {
                "min_edge_percent": 3.0,
                "min_confidence": 0.45,
                "target_assets": ["btc"],
                "target_timeframes": ["15m"],
            }
        },
    )

    assert decision.decision == "selected"
