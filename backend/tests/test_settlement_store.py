"""Pure-logic tests for the golden settlement store mapping.

The DB / resolver I/O is integration-tested elsewhere; here we lock the
pure translation from (local token metadata + resolved records) to the
engine's token_id -> TokenSettlement map, and the winner-label -> token
mapping.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.backtest.settlement_store import (
    MarketResolveHint,
    SettlementRecord,
    TokenMarketMeta,
    _winning_token_from_resolution,
    build_token_settlements,
)


RT = datetime(2026, 1, 1, 12, 5, 0, tzinfo=timezone.utc)


def _rec(cond, *, winner=None, token_ids=(), resolved=False, rt=None, outcome=None, source="test"):
    return SettlementRecord(
        condition_id=cond, slug=None, winning_token_id=winner,
        winning_outcome=outcome, token_ids=tuple(token_ids), resolution_time=rt,
        coin_price_start=None, coin_price_end=None, resolved=resolved, source=source,
    )


def test_winner_settles_one_loser_zero():
    meta = [
        TokenMarketMeta("tA", "C1", resolution_time=RT),
        TokenMarketMeta("tB", "C1", resolution_time=RT),
    ]
    recs = {"C1": _rec("C1", winner="tA", token_ids=("tA", "tB"), resolved=True, rt=RT, outcome="Up")}
    out = build_token_settlements(meta, recs)
    assert out["tA"].settle_price == 1.0
    assert out["tB"].settle_price == 0.0
    assert out["tA"].resolution_time == RT
    assert out["tA"].winning_outcome == "Up"
    assert out["tA"].won is True
    assert out["tB"].won is False


def test_no_record_is_omitted():
    out = build_token_settlements([TokenMarketMeta("tA", "C1")], {})
    assert "tA" not in out


def test_no_condition_id_is_omitted():
    out = build_token_settlements(
        [TokenMarketMeta("tA", None, resolution_time=RT)],
        {"C1": _rec("C1", winner="tA", token_ids=("tA",), resolved=True, rt=RT)},
    )
    assert "tA" not in out


def test_resolved_unknown_winner_yields_none_price():
    meta = [TokenMarketMeta("tA", "C1", resolution_time=RT)]
    recs = {"C1": _rec("C1", resolved=False, rt=RT)}  # known to resolve, no winner
    out = build_token_settlements(meta, recs)
    assert out["tA"].settle_price is None
    assert out["tA"].resolution_time == RT
    assert out["tA"].won is None


def test_resolution_time_falls_back_to_local_meta():
    meta = [TokenMarketMeta("tA", "C1", resolution_time=RT)]
    recs = {"C1": _rec("C1", winner="tA", token_ids=("tA",), resolved=True, rt=None, outcome="Up")}
    out = build_token_settlements(meta, recs)
    assert out["tA"].resolution_time == RT  # store had None -> local used


def test_foreign_token_not_settled():
    # token tX claims condition C1 but isn't in C1's known token set.
    meta = [TokenMarketMeta("tX", "C1", resolution_time=RT)]
    recs = {"C1": _rec("C1", winner="tA", token_ids=("tA", "tB"), resolved=True, rt=RT)}
    out = build_token_settlements(meta, recs)
    assert "tX" not in out


def test_winning_token_from_resolution_crypto_updown():
    class _MR:
        winner_outcome = "Up"
        coin_price_start = 1.0
        coin_price_end = 2.0
        source = "polybacktest"

    h = MarketResolveHint(condition_id="C1", slug="s", token_ids=("tU", "tD"), up_token="tU", down_token="tD")
    tok, label = _winning_token_from_resolution(h, _MR())
    assert tok == "tU"
    assert label == "Up"


def test_winning_token_from_resolution_yes_no_label():
    class _MR:
        winner_outcome = "No"
        source = "gamma"

    h = MarketResolveHint(
        condition_id="C1", slug="s", token_ids=("tYes", "tNo"),
        token_outcomes={"tYes": "Yes", "tNo": "No"},
    )
    tok, label = _winning_token_from_resolution(h, _MR())
    assert tok == "tNo"


def test_winning_token_unmappable_returns_none():
    class _MR:
        winner_outcome = "Maybe"
        source = "gamma"

    h = MarketResolveHint(condition_id="C1", slug="s", token_ids=("tYes", "tNo"),
                          token_outcomes={"tYes": "Yes", "tNo": "No"})
    tok, label = _winning_token_from_resolution(h, _MR())
    assert tok is None
    assert label == "Maybe"
