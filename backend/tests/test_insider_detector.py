import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.insider_detector import (
    InsiderDetectorService,
    _floor_to_minutes,
    _intent_signal_key,
)


def _event(side: str, price: float, market_id: str = "m1"):
    return SimpleNamespace(
        side=side,
        price=price,
        market_id=market_id,
        notional=100.0,
        traded_at=datetime(2026, 2, 10, 12, 0, 0),
        wallet_address="0xabc",
    )


def test_classification_thresholds():
    svc = InsiderDetectorService()
    assert (
        svc._classify_wallet(
            insider_score=0.74,
            insider_confidence=0.65,
            sample_size=30,
        )
        == "flagged_insider"
    )
    assert (
        svc._classify_wallet(
            insider_score=0.61,
            insider_confidence=0.52,
            sample_size=17,
        )
        == "watch_insider"
    )
    assert (
        svc._classify_wallet(
            insider_score=0.80,
            insider_confidence=0.40,
            sample_size=40,
        )
        == "none"
    )


def test_confidence_penalty_clamps():
    svc = InsiderDetectorService()
    assert svc._confidence_penalty(0.0) == 0.0
    assert 0.0 < svc._confidence_penalty(0.35) < 1.0
    assert svc._confidence_penalty(1.0) == 1.0


def test_brier_component_prefers_calibrated_entries():
    svc = InsiderDetectorService()
    win_rate = 0.70
    calibrated = [
        _event("BUY", 0.68),
        _event("BUY", 0.72),
        _event("YES", 0.69),
    ]
    miscalibrated = [
        _event("BUY", 0.20),
        _event("BUY", 0.25),
        _event("YES", 0.15),
    ]
    calibrated_score = svc._brier_component(win_rate=win_rate, events=calibrated)
    miscalibrated_score = svc._brier_component(win_rate=win_rate, events=miscalibrated)
    assert calibrated_score is not None
    assert miscalibrated_score is not None
    assert calibrated_score > miscalibrated_score


def test_intent_signal_key_stable_for_same_input():
    bucket = _floor_to_minutes(datetime(2026, 2, 10, 12, 34, 8), 10)
    key_a = _intent_signal_key("m1", "buy_yes", bucket, ["0x1", "0x2"])
    key_b = _intent_signal_key("m1", "buy_yes", bucket, ["0x2", "0x1"])
    assert key_a == key_b


def test_intent_signal_key_changes_for_wallet_cohort():
    bucket = _floor_to_minutes(datetime(2026, 2, 10, 12, 34, 8), 10)
    key_a = _intent_signal_key("m1", "buy_yes", bucket, ["0x1", "0x2"])
    key_b = _intent_signal_key("m1", "buy_yes", bucket, ["0x1", "0x3"])
    assert key_a != key_b


def test_funding_overlap_proxy_component_uses_sync_and_cluster():
    svc = InsiderDetectorService()
    weak = svc._funding_overlap_proxy_component(
        cluster_correlation_component=0.2,
        cluster_size=1,
    )
    strong = svc._funding_overlap_proxy_component(
        cluster_correlation_component=0.8,
        cluster_size=5,
    )
    assert weak is not None
    assert strong is not None
    assert strong > weak
