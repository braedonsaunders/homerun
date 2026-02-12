"""Unit tests for smart wallet pool scoring/churn and confluence thresholds."""

import sys
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timedelta
from typing import Optional

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from utils.utcnow import utcnow
from services.smart_wallet_pool import (  # noqa: E402
    SmartWalletPoolService,
    TARGET_POOL_SIZE,
    MAX_HOURLY_REPLACEMENT_RATE,
    MAX_POOL_SIZE,
    POOL_FLAG_BLACKLISTED,
    POOL_FLAG_MANUAL_EXCLUDE,
    POOL_FLAG_MANUAL_INCLUDE,
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
        now = utcnow()

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

    def test_source_confidence_rewards_broader_signal_coverage(self):
        svc = SmartWalletPoolService()
        light = SimpleNamespace(source_flags={"leaderboard": True})
        broad = SimpleNamespace(
            source_flags={
                "leaderboard": True,
                "leaderboard_pnl": True,
                "wallet_trades": True,
                "market_trades": True,
                "activity": True,
            }
        )
        assert svc._score_source_confidence(broad) > svc._score_source_confidence(light)

    def test_quality_gate_blockers_cover_unanalyzed_low_quality_wallets(self):
        svc = SmartWalletPoolService()
        wallet = SimpleNamespace(
            last_analyzed_at=None,
            recommendation="unanalyzed",
            total_trades=12,
            anomaly_score=0.74,
            total_pnl=-150.0,
        )
        blockers = svc._eligibility_blockers(wallet)
        codes = {b.get("code") for b in blockers}
        assert "not_analyzed" in codes
        assert "recommendation_blocked" in codes
        assert "insufficient_trades" in codes
        assert "anomaly_too_high" in codes
        assert "non_positive_pnl" in codes

    def test_activity_score_downweights_unverified_metrics_profiles(self):
        svc = SmartWalletPoolService()
        now = utcnow()
        verified_wallet = SimpleNamespace(
            metrics_source_version="accuracy_v2_closed_positions",
            last_analyzed_at=now - timedelta(hours=2),
        )
        legacy_wallet = SimpleNamespace(
            metrics_source_version=None,
            last_analyzed_at=now - timedelta(hours=2),
        )
        verified = svc._score_activity(
            trades_1h=4,
            trades_24h=18,
            last_trade_at=now - timedelta(minutes=10),
            now=now,
            wallet=verified_wallet,
        )
        legacy = svc._score_activity(
            trades_1h=4,
            trades_24h=18,
            last_trade_at=now - timedelta(minutes=10),
            now=now,
            wallet=legacy_wallet,
        )
        assert verified > legacy

    def test_selection_reasons_include_manual_churn_and_insider_alignment(self):
        svc = SmartWalletPoolService()
        now = utcnow()
        wallet = SimpleNamespace(
            source_flags={POOL_FLAG_MANUAL_INCLUDE: True},
            quality_score=0.8,
            trades_1h=0,
            trades_24h=4,
            last_trade_at=now - timedelta(hours=2),
            cluster_id=None,
        )
        reasons = svc._derive_selection_reasons(
            wallet=wallet,
            address="wallet_a",
            selection_score=0.86,
            insider_score=0.8,
            cutoff_72h=now - timedelta(hours=72),
            desired_addresses=set(),
            current_addresses={"wallet_a"},
            cluster_count=0,
            cluster_cap=10,
        )
        codes = {r.get("code") for r in reasons}
        assert "manual_include" in codes
        assert "churn_guard_retained" in codes
        assert "insider_alignment" in codes


class TestSmartWalletPoolChurnGuard:
    def test_quality_only_mode_allows_pool_shrink_below_minimum(self):
        svc = SmartWalletPoolService()
        current = [f"cur_{i}" for i in range(200)]
        desired = [f"elite_{i}" for i in range(30)]
        scores = {address: 0.5 for address in current + desired}

        final_pool, _ = svc._apply_churn_guard(
            desired=desired,
            current=current,
            scores=scores,
            quality_only_mode=True,
        )

        assert len(final_pool) == len(desired)
        assert set(final_pool) == set(desired)

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

    def test_manual_include_enforcement_adds_missing_addresses(self):
        svc = SmartWalletPoolService()
        base_pool = [f"cur_{i}" for i in range(10)]
        manual = ["wallet_manual_1", "wallet_manual_2"]
        scores = {address: 0.5 for address in base_pool + manual}
        eligible = set(base_pool + manual)

        final = svc._enforce_manual_includes(
            final_pool=base_pool,
            manual_includes=manual,
            scores=scores,
            eligible_addresses=eligible,
        )

        assert "wallet_manual_1" in final
        assert "wallet_manual_2" in final
        assert len(final) <= MAX_POOL_SIZE

    def test_pool_block_flags_take_precedence(self):
        svc = SmartWalletPoolService()
        wallet = SimpleNamespace(
            source_flags={
                POOL_FLAG_MANUAL_INCLUDE: True,
                POOL_FLAG_MANUAL_EXCLUDE: True,
                POOL_FLAG_BLACKLISTED: False,
            }
        )
        assert svc._is_pool_manually_included(wallet) is True
        assert svc._is_pool_manually_excluded(wallet) is True
        assert svc._is_pool_blocked(wallet) is True

    def test_cluster_diversity_limits_single_cluster_domination(self):
        svc = SmartWalletPoolService()
        crowded = [
            SimpleNamespace(address=f"cluster_{i}", cluster_id="entity_a")
            for i in range(80)
        ]
        independent = [
            SimpleNamespace(address=f"ind_{i}", cluster_id=None)
            for i in range(120)
        ]
        ranked = crowded + independent
        target_size = 100
        diversified = svc._rank_with_cluster_diversity(ranked, target_size=target_size)
        cap = max(3, int(target_size * 0.08))
        crowded_count = sum(1 for w in diversified if getattr(w, "cluster_id", None) == "entity_a")
        assert crowded_count <= cap


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
