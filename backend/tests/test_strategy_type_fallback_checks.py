from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.strategies.flash_crash_reversion import FlashCrashReversionStrategy
from services.strategies.tail_end_carry import TailEndCarryStrategy
from services.strategies.vpin_toxicity import VPINToxicityStrategy


def test_tail_end_carry_uses_signal_strategy_type_when_payload_omits_strategy():
    strategy = TailEndCarryStrategy()
    signal = SimpleNamespace(source="scanner", strategy_type="tail_end_carry", entry_price=0.92)
    payload = {
        "resolution_date": (datetime.now(timezone.utc) + timedelta(days=2)).isoformat().replace("+00:00", "Z"),
    }

    checks = strategy.custom_checks(signal, {}, {}, payload)
    strategy_check = next(check for check in checks if check.key == "strategy")
    assert strategy_check.passed is True


def test_flash_crash_reversion_uses_signal_strategy_type_when_payload_omits_strategy():
    strategy = FlashCrashReversionStrategy()
    signal = SimpleNamespace(
        source="scanner",
        direction="buy_yes",
        strategy_type="flash_crash_reversion",
        liquidity=5000.0,
    )

    checks = strategy.custom_checks(
        signal,
        {"live_market": {}},
        {"require_crash_alignment": False},
        {},
    )
    strategy_check = next(check for check in checks if check.key == "strategy")
    assert strategy_check.passed is True


def test_vpin_toxicity_uses_signal_strategy_type_when_payload_omits_strategy():
    strategy = VPINToxicityStrategy()
    signal = SimpleNamespace(strategy_type="vpin_toxicity")

    checks = strategy.custom_checks(signal, {}, {}, {})
    strategy_check = next(check for check in checks if check.key == "strategy")
    assert strategy_check.passed is True
