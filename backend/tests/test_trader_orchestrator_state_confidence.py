import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.trader_orchestrator_state import _normalize_trader_payload


def test_normalize_trader_payload_converts_percent_min_confidence():
    payload = {
        "name": "Crypto HF Trader",
        "strategy_key": "crypto_15m",
        "sources": ["crypto"],
        "params": {"min_confidence": 45, "min_edge_percent": 3.0},
    }
    normalized = _normalize_trader_payload(payload)
    assert normalized["params"]["min_confidence"] == 0.45


def test_normalize_trader_payload_preserves_fraction_min_confidence():
    payload = {
        "name": "Crypto HF Trader",
        "strategy_key": "crypto_15m",
        "sources": ["crypto"],
        "params": {"min_confidence": 0.45},
    }
    normalized = _normalize_trader_payload(payload)
    assert normalized["params"]["min_confidence"] == 0.45
