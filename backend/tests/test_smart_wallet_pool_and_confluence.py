"""Unit tests for smart wallet pool scoring/churn and confluence thresholds."""

import sys
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timedelta
from typing import Optional

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.smart_wallet_pool import (  # noqa: E402
    SmartWalletPoolService,
    TARGET_POOL_SIZE,
    MAX_HOURLY_REPLACEMENT_RATE,
)
from services.wallet_intelligence import ConfluenceDetector  # noqa: E402


def _wallet(
    *,
    rank_score: float,
    win_rate: float,
    sharpe_ratio: Optional[float],
    profit_factor: Optional[float],
    total_pnl: float,
    max_drawdown: Optional[float] = None,
    roi_std: float = 0.0,
    anomaly_score: float = 0.0,
    cluster_id: Optional[str] = None,
    is_profitable: bool = False,
):
    return SimpleNamespace(
        rank_score=rank_score,
        win_rate=win_rate,
        sharpe_ratio=sharpe_ratio,
        profit_factor=profit_factor,
        total_pnl=total_pnl,
        max_drawdown=max_drawdown,
        roi_std=roi_std,
        anomaly_score=anomaly_score,
        cluster_id=cluster_id,
        is_profitable=is_profitable,
    )


class TestSmartWalletPoolScoring:
    def test_quality_score_monotonic_for_better_wallet(self):
        svc = SmartWalletPoolService()
        weak = _wallet(
            rank_score=0.20,
            win_rate=0.45,
            sharpe_ratio=0.4,
            profit_factor=1.1,
            total_pnl=500.0,
        )
        strong = _wallet(
            rank_score=0.90,
            win_rate=0.72,
            sharpe_ratio=2.4,
            profit_factor=4.2,
            total_pnl=25000.0,
        )

        assert svc._score_quality(strong) > svc._score_quality(weak)

    def test_activity_score_decays_with_recency(self):
        svc = SmartWalletPoolService()
        now = datetime.utcnow()

        fresh_score = svc._score_activity(
            trades_1h=6,
            trades_24h=40,
            last_trade_at=now - timedelta(minutes=2),
            now=now,
        )
        stale_score = svc._score_activity(
            trades_1h=0,
            trades_24h=1,
            last_trade_at=now - timedelta(hours=96),
            now=now,
        )

        assert 0.0 <= fresh_score <= 1.0
        assert 0.0 <= stale_score <= 1.0
        assert fresh_score > stale_score


class TestSmartWalletPoolChurnGuard:
    def test_replacements_capped_when_score_delta_is_small(self):
        svc = SmartWalletPoolService()
        current = [f"cur_{i}" for i in range(TARGET_POOL_SIZE)]
        desired = [f"new_{i}" for i in range(TARGET_POOL_SIZE)]

        scores = {address: 0.50 for address in current}
        scores.update({address: 0.53 for address in desired})

        final_pool, churn_rate = svc._apply_churn_guard(
            desired=desired,
            current=current,
            scores=scores,
        )

        replacements = len(set(final_pool) - set(current))
        cap = int(TARGET_POOL_SIZE * MAX_HOURLY_REPLACEMENT_RATE)

        assert replacements <= cap
        assert churn_rate <= cap / TARGET_POOL_SIZE

    def test_replacements_can_exceed_cap_when_score_delta_is_large(self):
        svc = SmartWalletPoolService()
        current = [f"cur_{i}" for i in range(TARGET_POOL_SIZE)]
        desired = [f"new_{i}" for i in range(TARGET_POOL_SIZE)]

        scores = {address: 0.20 for address in current}
        scores.update({address: 0.95 for address in desired})

        final_pool, _ = svc._apply_churn_guard(
            desired=desired,
            current=current,
            scores=scores,
        )

        replacements = len(set(final_pool) - set(current))
        cap = int(TARGET_POOL_SIZE * MAX_HOURLY_REPLACEMENT_RATE)
        assert replacements > cap


class TestConfluenceDetectorThresholds:
    def test_tier_thresholds_follow_watch_high_extreme(self):
        detector = ConfluenceDetector()
        assert detector._tier_for_count(5) == "WATCH"
        assert detector._tier_for_count(9) == "WATCH"
        assert detector._tier_for_count(10) == "HIGH"
        assert detector._tier_for_count(14) == "HIGH"
        assert detector._tier_for_count(15) == "EXTREME"

    def test_conviction_score_clamped_and_directional(self):
        detector = ConfluenceDetector()

        best_case = detector._conviction_score(
            adjusted_wallet_count=30,
            weighted_wallet_score=1.0,
            timing_tightness=1.0,
            net_notional=1_000_000_000.0,
            conflicting_notional=0.0,
            market_liquidity=1_000_000_000.0,
            market_volume_24h=1_000_000_000.0,
            anomaly_avg=0.0,
            unique_wallet_count=30,
        )
        worst_case = detector._conviction_score(
            adjusted_wallet_count=0,
            weighted_wallet_score=0.0,
            timing_tightness=0.0,
            net_notional=0.0,
            conflicting_notional=10_000.0,
            market_liquidity=0.0,
            market_volume_24h=0.0,
            anomaly_avg=1.0,
            unique_wallet_count=10,
        )

        assert 0.0 <= best_case <= 100.0
        assert 0.0 <= worst_case <= 100.0
        assert best_case > worst_case
