from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api import routes_discovery
from utils.utcnow import utcnow


class _ScalarResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _RowsResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


def _make_session(
    wallets,
    *,
    tracked_rows=None,
    activity_rows=None,
    cached_market_rows=None,
    supplemental_cached_market_rows=None,
):
    # Query order inside get_pool_members:
    # 1) select(DiscoveredWallet)
    # 2) select(TrackedWallet.address, TrackedWallet.label)
    # 3) wallet_activity_rollups aggregation (for primary market category)
    # 4) cached_markets direct condition_id lookup
    # 5) cached_markets supplemental token lookup
    execute = AsyncMock(
        side_effect=[
            _ScalarResult(wallets),
            _RowsResult(tracked_rows or []),
            _RowsResult(activity_rows or []),
            _RowsResult(cached_market_rows or []),
            _RowsResult(supplemental_cached_market_rows or []),
        ]
    )
    return SimpleNamespace(execute=execute)


def _wallet(
    *,
    address: str,
    win_rate: float,
    total_pnl: float,
    total_trades: int,
    rank_score: float = 0.7,
    composite_score: float = 0.7,
    quality_score: float = 0.7,
    activity_score: float = 0.7,
    in_top_pool: bool = True,
    pool_tier: str | None = "core",
    pool_membership_reason: str | None = "core_quality_gate",
    source_flags: dict | None = None,
):
    return SimpleNamespace(
        address=address,
        username=None,
        in_top_pool=in_top_pool,
        pool_tier=pool_tier,
        pool_membership_reason=pool_membership_reason,
        rank_score=rank_score,
        composite_score=composite_score,
        quality_score=quality_score,
        activity_score=activity_score,
        stability_score=0.6,
        trades_1h=0,
        trades_24h=0,
        last_trade_at=utcnow(),
        total_trades=total_trades,
        total_pnl=total_pnl,
        win_rate=win_rate,
        tags=[],
        strategies_detected=[],
        cluster_id=None,
        source_flags=source_flags or {},
        last_analyzed_at=utcnow(),
    )


def _activity_row(*, wallet_address: str, market_id: str, trade_count: int, total_notional: float):
    return SimpleNamespace(
        wallet_address=wallet_address,
        market_id=market_id,
        trade_count=trade_count,
        total_notional=total_notional,
    )


def _cached_market_row(*, condition_id: str, category: str | None, extra_data: dict | None = None):
    return SimpleNamespace(
        condition_id=condition_id,
        category=category,
        extra_data=extra_data if isinstance(extra_data, dict) else None,
    )


async def _call_pool_members(session, **overrides):
    params = {
        "limit": 300,
        "offset": 0,
        "pool_only": True,
        "include_blacklisted": True,
        "tier": None,
        "search": "",
        "min_win_rate": 0.0,
        "sort_by": "composite_score",
        "sort_dir": "desc",
        "session": session,
    }
    params.update(overrides)
    return await routes_discovery.get_pool_members(**params)


@pytest.mark.asyncio
async def test_pool_members_sorts_by_win_rate_desc_with_mixed_scales():
    wallets = [
        _wallet(address="0xaaa", win_rate=0.62, total_pnl=1000.0, total_trades=100),
        _wallet(address="0xbbb", win_rate=78.0, total_pnl=1500.0, total_trades=120),  # legacy percent
        _wallet(address="0xccc", win_rate=0.71, total_pnl=1200.0, total_trades=90),
    ]
    session = _make_session(wallets)

    out = await _call_pool_members(
        session,
        sort_by="win_rate",
        sort_dir="desc",
    )

    # 78.0 should normalize to 0.78 and rank first.
    assert [m["address"] for m in out["members"]] == ["0xbbb", "0xccc", "0xaaa"]
    assert out["members"][0]["win_rate"] == pytest.approx(0.78, abs=1e-6)


@pytest.mark.asyncio
async def test_pool_members_min_win_rate_accepts_percent_or_ratio_inputs():
    wallets = [
        _wallet(address="0xaaa", win_rate=0.68, total_pnl=1000.0, total_trades=100),
        _wallet(address="0xbbb", win_rate=0.71, total_pnl=1200.0, total_trades=110),
        _wallet(address="0xccc", win_rate=72.0, total_pnl=1300.0, total_trades=130),  # legacy percent
    ]

    out_pct = await _call_pool_members(
        _make_session(wallets),
        min_win_rate=70.0,
        sort_by="win_rate",
        sort_dir="desc",
    )
    out_ratio = await _call_pool_members(
        _make_session(wallets),
        min_win_rate=0.70,
        sort_by="win_rate",
        sort_dir="desc",
    )

    assert [m["address"] for m in out_pct["members"]] == ["0xccc", "0xbbb"]
    assert [m["address"] for m in out_ratio["members"]] == ["0xccc", "0xbbb"]


@pytest.mark.asyncio
async def test_pool_members_sorts_by_total_pnl_and_total_trades():
    wallets = [
        _wallet(address="0xaaa", win_rate=0.65, total_pnl=5000.0, total_trades=20),
        _wallet(address="0xbbb", win_rate=0.75, total_pnl=1500.0, total_trades=200),
        _wallet(address="0xccc", win_rate=0.70, total_pnl=9900.0, total_trades=80),
    ]

    by_pnl_desc = await _call_pool_members(
        _make_session(wallets),
        sort_by="total_pnl",
        sort_dir="desc",
    )
    assert [m["address"] for m in by_pnl_desc["members"]] == ["0xccc", "0xaaa", "0xbbb"]

    by_trades_asc = await _call_pool_members(
        _make_session(wallets),
        sort_by="total_trades",
        sort_dir="asc",
    )
    assert [m["address"] for m in by_trades_asc["members"]] == ["0xaaa", "0xccc", "0xbbb"]


@pytest.mark.asyncio
async def test_pool_members_invalid_sort_by_returns_400():
    with pytest.raises(HTTPException) as excinfo:
        await _call_pool_members(
            _make_session([]),
            sort_by="not_a_field",
        )
    assert excinfo.value.status_code == 400


@pytest.mark.asyncio
async def test_pool_members_reconciles_stale_blocked_metadata_for_active_pool_rows():
    stale_meta_wallet = _wallet(
        address="0xabc",
        win_rate=0.74,
        total_pnl=3200.0,
        total_trades=120,
        in_top_pool=True,
        source_flags={
            "pool_selection_meta": {
                "eligibility_status": "blocked",
                "eligibility_blockers": [
                    {"code": "recommendation_blocked", "label": "Recommendation blocked"},
                ],
                "reasons": [
                    {"code": "recommendation_blocked", "label": "Recommendation blocked"},
                ],
            }
        },
    )
    out = await _call_pool_members(_make_session([stale_meta_wallet]))

    assert len(out["members"]) == 1
    row = out["members"][0]
    assert row["in_top_pool"] is True
    assert row["eligibility_status"] == "eligible"
    assert row["eligibility_blockers"] == []
    assert row["selection_reasons"] == []


@pytest.mark.asyncio
async def test_pool_members_keeps_blocked_metadata_for_non_pool_rows():
    blocked_meta_wallet = _wallet(
        address="0xdef",
        win_rate=0.48,
        total_pnl=-250.0,
        total_trades=12,
        in_top_pool=False,
        pool_tier=None,
        pool_membership_reason="recommendation_blocked",
        source_flags={
            "pool_selection_meta": {
                "eligibility_status": "blocked",
                "eligibility_blockers": [
                    {"code": "recommendation_blocked", "label": "Recommendation blocked"},
                ],
                "reasons": [
                    {"code": "recommendation_blocked", "label": "Recommendation blocked"},
                ],
            }
        },
    )
    out = await _call_pool_members(_make_session([blocked_meta_wallet]), pool_only=False)

    assert len(out["members"]) == 1
    row = out["members"][0]
    assert row["in_top_pool"] is False
    assert row["eligibility_status"] == "blocked"
    assert row["eligibility_blockers"] == [{"code": "recommendation_blocked", "label": "Recommendation blocked"}]
    assert row["selection_reasons"] == [{"code": "recommendation_blocked", "label": "Recommendation blocked"}]


@pytest.mark.asyncio
async def test_pool_members_primary_market_category_uses_notional_weight():
    wallet = _wallet(
        address="0xabc",
        win_rate=0.72,
        total_pnl=2100.0,
        total_trades=140,
    )
    session = _make_session(
        [wallet],
        activity_rows=[
            _activity_row(
                wallet_address="0xabc",
                market_id="0xmarket_crypto",
                trade_count=5,
                total_notional=1400.0,
            ),
            _activity_row(
                wallet_address="0xabc",
                market_id="0xmarket_sports",
                trade_count=16,
                total_notional=600.0,
            ),
        ],
        cached_market_rows=[
            _cached_market_row(condition_id="0xmarket_crypto", category="Crypto"),
            _cached_market_row(condition_id="0xmarket_sports", category="Sports"),
        ],
    )

    out = await _call_pool_members(session)
    row = out["members"][0]
    assert row["primary_market_category"] == "crypto"
    assert row["primary_market_category_basis"] == "notional"
    assert row["primary_market_category_share"] == pytest.approx(0.70, abs=1e-6)


@pytest.mark.asyncio
async def test_pool_members_primary_market_category_falls_back_to_trade_count_when_notional_absent():
    wallet = _wallet(
        address="0xdef",
        win_rate=0.66,
        total_pnl=900.0,
        total_trades=90,
    )
    session = _make_session(
        [wallet],
        activity_rows=[
            _activity_row(
                wallet_address="0xdef",
                market_id="0xmarket_politics",
                trade_count=3,
                total_notional=0.0,
            ),
            _activity_row(
                wallet_address="0xdef",
                market_id="0xmarket_crypto",
                trade_count=9,
                total_notional=0.0,
            ),
        ],
        cached_market_rows=[
            _cached_market_row(condition_id="0xmarket_politics", category="Politics"),
            _cached_market_row(condition_id="0xmarket_crypto", category="Crypto"),
        ],
    )

    out = await _call_pool_members(session)
    row = out["members"][0]
    assert row["primary_market_category"] == "crypto"
    assert row["primary_market_category_basis"] == "trade_count"
    assert row["primary_market_category_share"] == pytest.approx(0.75, abs=1e-6)


@pytest.mark.asyncio
async def test_pool_members_primary_market_category_falls_back_to_flags_when_activity_missing():
    wallet = _wallet(
        address="0x123",
        win_rate=0.61,
        total_pnl=300.0,
        total_trades=40,
        source_flags={
            "leaderboard_category_politics": True,
            "leaderboard_category_crypto": True,
        },
    )

    out = await _call_pool_members(_make_session([wallet]))
    row = out["members"][0]
    assert row["primary_market_category"] == "crypto"
    assert row["primary_market_category_basis"] == "flags"
    assert row["primary_market_category_share"] is None
