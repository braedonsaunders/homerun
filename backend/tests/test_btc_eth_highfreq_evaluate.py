from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.strategies.btc_eth_highfreq import BtcEthHighFreqStrategy


def _signal(*, created_at: datetime, updated_at: datetime | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id="sig-1",
        source="crypto",
        signal_type="crypto_worker_v2",
        direction="buy_yes",
        edge_percent=8.0,
        confidence=0.8,
        created_at=created_at,
        updated_at=updated_at or created_at,
        payload_json={
            "strategy_origin": "crypto_worker",
            "asset": "BTC",
            "timeframe": "5m",
            "regime": "opening",
            "dominant_strategy": "directional",
            "is_live": True,
            "is_current": True,
            "seconds_left": 280,
            "model_prob_yes": 0.58,
            "model_prob_no": 0.42,
            "up_price": 0.49,
            "down_price": 0.51,
        },
    )


def _check_by_key(decision, key: str):
    for check in decision.checks:
        if str(getattr(check, "key", "")) == key:
            return check
    raise AssertionError(f"missing decision check: {key}")


def test_evaluate_uses_live_context_for_entry_window_gate():
    strategy = BtcEthHighFreqStrategy()
    signal = _signal(created_at=datetime.now(timezone.utc))

    decision = strategy.evaluate(
        signal,
        {
            "params": {
                "min_edge_percent": 1.0,
                "min_confidence": 0.2,
                "direction_guardrail_enabled": False,
                "min_seconds_left_for_entry_5m": 60,
            },
            "live_market": {
                "available": True,
                "is_live": True,
                "is_current": True,
                "seconds_left": 20,
            },
        },
    )

    assert decision.decision == "skipped"
    entry_window = _check_by_key(decision, "entry_window")
    assert entry_window.passed is False
    assert "required>=60.0" in str(entry_window.detail)
    assert decision.payload["seconds_left"] == 20


def test_evaluate_blocks_stale_signal_age():
    strategy = BtcEthHighFreqStrategy()
    signal = _signal(created_at=datetime.now(timezone.utc) - timedelta(seconds=120))

    decision = strategy.evaluate(
        signal,
        {
            "params": {
                "min_edge_percent": 1.0,
                "min_confidence": 0.2,
                "direction_guardrail_enabled": False,
                "max_signal_age_seconds": 15,
            },
            "live_market": {
                "available": True,
                "is_live": True,
                "is_current": True,
                "seconds_left": 240,
            },
        },
    )

    assert decision.decision == "skipped"
    freshness = _check_by_key(decision, "signal_freshness")
    assert freshness.passed is False
    assert "max=15.0s" in str(freshness.detail)


def test_evaluate_uses_signal_updated_at_for_freshness():
    strategy = BtcEthHighFreqStrategy()
    signal = _signal(
        created_at=datetime.now(timezone.utc) - timedelta(seconds=180),
        updated_at=datetime.now(timezone.utc) - timedelta(seconds=3),
    )

    decision = strategy.evaluate(
        signal,
        {
            "params": {
                "min_edge_percent": 1.0,
                "min_confidence": 0.2,
                "direction_guardrail_enabled": False,
                "max_signal_age_seconds": 35,
            },
            "live_market": {
                "available": True,
                "is_live": True,
                "is_current": True,
                "seconds_left": 240,
            },
        },
    )

    freshness = _check_by_key(decision, "signal_freshness")
    assert freshness.passed is True
    assert decision.payload["signal_age_seconds"] is not None
    assert decision.payload["signal_age_seconds"] < 35


def test_evaluate_blocks_stale_live_market_context():
    strategy = BtcEthHighFreqStrategy()
    signal = _signal(created_at=datetime.now(timezone.utc))

    decision = strategy.evaluate(
        signal,
        {
            "params": {
                "min_edge_percent": 1.0,
                "min_confidence": 0.2,
                "direction_guardrail_enabled": False,
                "max_live_context_age_seconds": 2.0,
            },
            "live_market": {
                "available": True,
                "is_live": True,
                "is_current": True,
                "seconds_left": 240,
                "fetched_at": (datetime.now(timezone.utc) - timedelta(seconds=15)).isoformat(),
            },
        },
    )

    assert decision.decision == "skipped"
    freshness = _check_by_key(decision, "live_context_freshness")
    assert freshness.passed is False
    assert "max=2.0s" in str(freshness.detail)


def test_evaluate_blocks_stale_oracle_when_directional_requires_it():
    strategy = BtcEthHighFreqStrategy()
    signal = _signal(created_at=datetime.now(timezone.utc))
    signal.payload_json["oracle_available"] = True
    signal.payload_json["oracle_age_seconds"] = 45.0

    decision = strategy.evaluate(
        signal,
        {
            "params": {
                "min_edge_percent": 1.0,
                "min_confidence": 0.2,
                "direction_guardrail_enabled": False,
                "max_oracle_age_seconds": 10.0,
                "require_oracle_for_directional": True,
            },
            "live_market": {
                "available": True,
                "is_live": True,
                "is_current": True,
                "seconds_left": 240,
            },
        },
    )

    assert decision.decision == "skipped"
    oracle_freshness = _check_by_key(decision, "oracle_freshness")
    assert oracle_freshness.passed is False
    assert "required=True" in str(oracle_freshness.detail)


def test_evaluate_blocks_when_spread_exceeds_cap():
    strategy = BtcEthHighFreqStrategy()
    signal = _signal(created_at=datetime.now(timezone.utc))
    signal.payload_json["spread"] = 0.12

    decision = strategy.evaluate(
        signal,
        {
            "params": {
                "min_edge_percent": 1.0,
                "min_confidence": 0.2,
                "direction_guardrail_enabled": False,
                "max_spread_pct": 0.05,
            },
            "live_market": {
                "available": True,
                "is_live": True,
                "is_current": True,
                "seconds_left": 240,
            },
        },
    )

    assert decision.decision == "skipped"
    spread = _check_by_key(decision, "spread")
    assert spread.passed is False


def test_evaluate_blocks_when_liquidity_below_threshold():
    strategy = BtcEthHighFreqStrategy()
    signal = _signal(created_at=datetime.now(timezone.utc))
    signal.payload_json["liquidity"] = 100.0

    decision = strategy.evaluate(
        signal,
        {
            "params": {
                "min_edge_percent": 1.0,
                "min_confidence": 0.2,
                "direction_guardrail_enabled": False,
                "min_liquidity_usd": 500.0,
            },
            "live_market": {
                "available": True,
                "is_live": True,
                "is_current": True,
                "seconds_left": 240,
            },
        },
    )

    assert decision.decision == "skipped"
    liquidity = _check_by_key(decision, "liquidity")
    assert liquidity.passed is False


def test_evaluate_enforces_reentry_cooldown_by_default():
    strategy = BtcEthHighFreqStrategy()
    signal = _signal(created_at=datetime.now(timezone.utc))

    decision = strategy.evaluate(
        signal,
        {
            "params": {
                "min_edge_percent": 1.0,
                "min_confidence": 0.2,
                "direction_guardrail_enabled": False,
            },
            "live_market": {
                "available": True,
                "is_live": True,
                "is_current": True,
                "seconds_left": 240,
            },
        },
    )

    cooldown = _check_by_key(decision, "reentry_cooldown")
    assert cooldown.passed is True
    assert "required_ms=75000" in str(cooldown.detail)
    assert decision.payload["reentry_cooldown_seconds_per_market"] == 75.0
    assert decision.payload["force_disable_reentry_cooldown"] is False


def test_evaluate_blocks_4h_timeframe_by_hardening_allowlist():
    strategy = BtcEthHighFreqStrategy()
    signal = _signal(created_at=datetime.now(timezone.utc))
    signal.payload_json["timeframe"] = "4h"

    decision = strategy.evaluate(
        signal,
        {
            "params": {
                "min_edge_percent": 1.0,
                "min_confidence": 0.2,
                "direction_guardrail_enabled": False,
            },
            "live_market": {
                "available": True,
                "is_live": True,
                "is_current": True,
                "seconds_left": 14000,
            },
        },
    )

    assert decision.decision == "skipped"
    timeframe_hardening = _check_by_key(decision, "timeframe_hardening")
    assert timeframe_hardening.passed is False
    detail = str(timeframe_hardening.detail)
    assert "allowed=" in detail
    assert "5m" in detail and "15m" in detail and "1h" in detail


def test_evaluate_blocks_entry_above_hardened_price_ceiling():
    strategy = BtcEthHighFreqStrategy()
    signal = _signal(created_at=datetime.now(timezone.utc))
    signal.entry_price = 0.82

    decision = strategy.evaluate(
        signal,
        {
            "params": {
                "min_edge_percent": 1.0,
                "min_confidence": 0.2,
                "direction_guardrail_enabled": False,
            },
            "live_market": {
                "available": True,
                "is_live": True,
                "is_current": True,
                "seconds_left": 260,
            },
        },
    )

    assert decision.decision == "skipped"
    entry_ceiling = _check_by_key(decision, "entry_price_ceiling")
    assert entry_ceiling.passed is False
    assert "ceiling=" in str(entry_ceiling.detail)


def test_evaluate_blocks_rebalance_low_entry_price_floor():
    strategy = BtcEthHighFreqStrategy()
    signal = _signal(created_at=datetime.now(timezone.utc))
    signal.entry_price = 0.02
    signal.payload_json["dominant_strategy"] = "rebalance"
    signal.payload_json["regime"] = "closing"
    signal.payload_json["up_price"] = 0.02
    signal.payload_json["down_price"] = 0.98

    decision = strategy.evaluate(
        signal,
        {
            "params": {
                "min_edge_percent": 1.0,
                "min_confidence": 0.2,
                "direction_guardrail_enabled": False,
                "rebalance_min_entry_price_floor": 0.08,
            },
            "live_market": {
                "available": True,
                "is_live": True,
                "is_current": True,
                "seconds_left": 240,
            },
        },
    )

    assert decision.decision == "skipped"
    entry_floor = _check_by_key(decision, "entry_price_floor")
    assert entry_floor.passed is False
    assert decision.payload["entry_price"] == 0.02
    assert decision.payload["entry_price_floor"] >= 0.08


def test_evaluate_blocks_when_estimated_size_fails_exitability_guard():
    strategy = BtcEthHighFreqStrategy()
    signal = _signal(created_at=datetime.now(timezone.utc))

    decision = strategy.evaluate(
        signal,
        {
            "params": {
                "min_edge_percent": 1.0,
                "min_confidence": 0.2,
                "direction_guardrail_enabled": False,
                "base_size_usd": 1.0,
                "max_size_usd": 1.0,
                "entry_executable_exit_ratio_floor": 0.30,
            },
            "live_market": {
                "available": True,
                "is_live": True,
                "is_current": True,
                "seconds_left": 240,
            },
        },
    )

    assert decision.decision == "skipped"
    exitability = _check_by_key(decision, "entry_exitability")
    assert exitability.passed is False
    assert decision.payload["entry_exitability_estimated_size_usd"] == 1.0
    assert decision.payload["entry_exitability_required_size_usd"] > 3.0
