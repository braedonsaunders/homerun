import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.autotrader_state import normalize_trading_domains


def test_normalize_trading_domains_defaults_when_empty():
    assert normalize_trading_domains([]) == ["event_markets", "crypto"]


def test_normalize_trading_domains_filters_unknown_values():
    assert normalize_trading_domains(["event_markets", "unknown", "crypto", "crypto"]) == [
        "event_markets",
        "crypto",
    ]

