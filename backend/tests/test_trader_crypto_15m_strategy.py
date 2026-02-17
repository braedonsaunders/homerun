import sys
from pathlib import Path
from types import SimpleNamespace

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.strategies.btc_eth_highfreq import BtcEthHighFreqStrategy


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


def test_crypto_strategy_normalizes_percent_min_confidence():
    strategy = BtcEthHighFreqStrategy()
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
    # Confidence 45 (percent) should be normalized to 0.45
    conf_check = next(c for c in decision.checks if c.key == "confidence")
    assert conf_check.passed is True


def test_crypto_strategy_rejects_non_worker_crypto_signals():
    strategy = BtcEthHighFreqStrategy()
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


def test_crypto_strategy_selects_valid_signal():
    strategy = BtcEthHighFreqStrategy()
    payload = _base_payload()
    signal = SimpleNamespace(
        source="crypto",
        signal_type="crypto_worker_multistrat",
        edge_percent=6.1,
        confidence=0.55,
        direction="buy_yes",
        payload_json=payload,
    )

    decision = strategy.evaluate(
        signal,
        {
            "params": {
                "min_edge_percent": 2.5,
                "min_confidence": 0.45,
                "base_size_usd": 20.0,
            }
        },
    )

    assert decision.decision == "selected"
    assert decision.size_usd is not None and decision.size_usd > 0


def test_crypto_strategy_rejects_low_edge():
    strategy = BtcEthHighFreqStrategy()
    signal = SimpleNamespace(
        source="crypto",
        signal_type="crypto_worker_multistrat",
        edge_percent=1.0,
        confidence=0.62,
        direction="buy_yes",
        payload_json=_base_payload(),
    )

    decision = strategy.evaluate(
        signal,
        {"params": {"min_edge_percent": 3.0, "min_confidence": 0.45}},
    )

    assert decision.decision == "skipped"
    edge_check = next(c for c in decision.checks if c.key == "edge")
    assert edge_check.passed is False


def test_crypto_strategy_rejects_low_confidence():
    strategy = BtcEthHighFreqStrategy()
    signal = SimpleNamespace(
        source="crypto",
        signal_type="crypto_worker_multistrat",
        edge_percent=6.1,
        confidence=0.20,
        direction="buy_yes",
        payload_json=_base_payload(),
    )

    decision = strategy.evaluate(
        signal,
        {"params": {"min_edge_percent": 3.0, "min_confidence": 0.45}},
    )

    assert decision.decision == "skipped"
    conf_check = next(c for c in decision.checks if c.key == "confidence")
    assert conf_check.passed is False


def test_crypto_strategy_rejects_non_crypto_source():
    strategy = BtcEthHighFreqStrategy()
    signal = SimpleNamespace(
        source="scanner",
        signal_type="crypto_worker_multistrat",
        edge_percent=6.1,
        confidence=0.62,
        direction="buy_yes",
        payload_json=_base_payload(),
    )

    decision = strategy.evaluate(
        signal,
        {"params": {"min_edge_percent": 3.0, "min_confidence": 0.45}},
    )

    assert decision.decision == "skipped"
    source_check = next(c for c in decision.checks if c.key == "source")
    assert source_check.passed is False


def test_crypto_strategy_regime_closing_adjusts_thresholds():
    strategy = BtcEthHighFreqStrategy()
    payload = _base_payload()
    payload["regime"] = "closing"
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
        {"params": {"min_edge_percent": 3.0, "min_confidence": 0.45, "base_size_usd": 25.0}},
    )

    # Closing regime reduces required thresholds (factor 0.9/0.95) so should still select
    assert decision.decision == "selected"
    # Closing regime increases size factor by 1.1
    assert decision.size_usd is not None and decision.size_usd > 0


def test_crypto_strategy_sizes_within_max():
    strategy = BtcEthHighFreqStrategy()
    signal = SimpleNamespace(
        source="crypto",
        signal_type="crypto_worker_multistrat",
        edge_percent=50.0,
        confidence=0.99,
        direction="buy_yes",
        payload_json=_base_payload(),
    )

    decision = strategy.evaluate(
        signal,
        {"params": {"min_edge_percent": 3.0, "min_confidence": 0.45, "base_size_usd": 25.0, "max_size_usd": 50.0}},
    )

    assert decision.decision == "selected"
    assert decision.size_usd <= 50.0
