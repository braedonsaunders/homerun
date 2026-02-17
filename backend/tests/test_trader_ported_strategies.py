from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.strategies.crypto_spike_reversion import CryptoSpikeReversionStrategy
from services.strategies.flash_crash_reversion import FlashCrashReversionStrategy
from services.strategies.tail_end_carry import TailEndCarryStrategy


def _signal(**kwargs):
    base = {
        "source": "scanner",
        "signal_type": "scanner_opportunity",
        "direction": "buy_yes",
        "edge_percent": 5.0,
        "confidence": 0.6,
        "liquidity": 8000.0,
        "entry_price": 0.5,
        "payload_json": {},
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def _live_context(*, move_5m: float, move_30m: float, move_2h: float) -> dict:
    return {
        "history_summary": {
            "move_5m": {"percent": move_5m},
            "move_30m": {"percent": move_30m},
            "move_2h": {"percent": move_2h},
        }
    }


def test_flash_reversion_selects_when_alignment_and_risk_pass() -> None:
    strategy = FlashCrashReversionStrategy()
    signal = _signal(
        payload_json={
            "strategy": "flash_crash_reversion",
            "risk_score": 0.56,
        },
    )

    decision = strategy.evaluate(
        signal,
        {
            "params": {
                "min_edge_percent": 3.0,
                "min_confidence": 0.4,
                "max_risk_score": 0.8,
                "sizing_policy": "kelly",
            },
            "live_market": _live_context(move_5m=-2.6, move_30m=-1.3, move_2h=-4.0),
        },
    )

    assert decision.decision == "selected"
    assert decision.size_usd is not None and decision.size_usd > 0
    assert decision.payload.get("sizing", {}).get("policy") == "kelly"


def test_flash_reversion_skips_when_move_not_aligned() -> None:
    strategy = FlashCrashReversionStrategy()
    signal = _signal(
        direction="buy_yes",
        payload_json={
            "strategy": "flash_crash_reversion",
            "risk_score": 0.5,
        },
    )

    decision = strategy.evaluate(
        signal,
        {
            "params": {"require_crash_alignment": True, "min_abs_move_5m": 1.5},
            "live_market": _live_context(move_5m=0.4, move_30m=0.1, move_2h=1.0),
        },
    )

    assert decision.decision == "skipped"
    check = next(item for item in decision.checks if item.key == "alignment_5m")
    assert check.passed is False


def test_tail_carry_selects_for_near_expiry_window() -> None:
    strategy = TailEndCarryStrategy()
    resolution_date = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat().replace("+00:00", "Z")
    signal = _signal(
        edge_percent=2.3,
        confidence=0.52,
        entry_price=0.93,
        payload_json={
            "strategy": "tail_end_carry",
            "risk_score": 0.58,
            "resolution_date": resolution_date,
        },
    )

    decision = strategy.evaluate(
        signal,
        {
            "params": {
                "min_edge_percent": 1.5,
                "min_confidence": 0.35,
                "max_risk_score": 0.8,
                "min_entry_price": 0.85,
                "max_entry_price": 0.985,
            }
        },
    )

    assert decision.decision == "selected"
    assert decision.size_usd is not None and decision.size_usd > 0


def test_crypto_spike_reversion_selects_on_down_spike_for_buy_yes() -> None:
    strategy = CryptoSpikeReversionStrategy()
    signal = _signal(
        source="crypto",
        signal_type="crypto_worker_multistrat",
        direction="buy_yes",
        edge_percent=4.1,
        confidence=0.63,
        liquidity=14000.0,
        entry_price=0.47,
        payload_json={
            "strategy_origin": "crypto_worker",
            "model_prob_yes": 0.58,
            "model_prob_no": 0.42,
        },
    )

    decision = strategy.evaluate(
        signal,
        {
            "params": {
                "min_edge_percent": 2.8,
                "min_confidence": 0.44,
                "min_abs_move_5m": 1.8,
                "max_abs_move_2h": 14.0,
            },
            "live_market": _live_context(move_5m=-3.2, move_30m=-1.8, move_2h=-5.2),
        },
    )

    assert decision.decision == "selected"
    assert decision.payload.get("sizing", {}).get("policy") in {"kelly", "linear", "adaptive", "fixed"}


def test_crypto_spike_reversion_rejects_non_worker_signals() -> None:
    strategy = CryptoSpikeReversionStrategy()
    signal = _signal(
        source="crypto",
        signal_type="legacy_crypto_signal",
        payload_json={"strategy_origin": "legacy"},
    )

    decision = strategy.evaluate(
        signal,
        {
            "live_market": _live_context(move_5m=-2.0, move_30m=-1.0, move_2h=-4.0),
        },
    )

    assert decision.decision == "skipped"
    origin_check = next(check for check in decision.checks if check.key == "origin")
    assert origin_check.passed is False
