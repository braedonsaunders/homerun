"""Canonical crypto dict->Market reconstruction + equivalence guard.

There used to be SIX copies of "crypto-worker dict -> Market": three
``_market_from_crypto_dict`` (btc_eth_convergence / maker_quote /
directional_edge) plus mirrors in crypto_5m_midcycle and crypto_distance_edge.
They were consolidated onto the canonical
``crypto_strategy_utils.build_binary_crypto_market``.

This pins:
  1. The canonical reconstructor's CONTRACT (field mapping for a real-shaped
     dispatch dict, and None on the unusable rows the call sites reject anyway).
  2. EQUIVALENCE with the legacy ``_market_from_crypto_dict`` logic (inlined here
     as ``_legacy_reconstruct``) for every real-shaped dict — so the
     consolidation could not change live behavior. The legacy and canonical
     paths differ ONLY on rows that carry no usable price/id (legacy built a
     Market with ``[0.0, 0.0]`` prices; canonical returns None) or a non-Z
     end_time — rows the strategies' own gates reject either way.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from models.market import Market
from services.strategies.crypto_strategy_utils import build_binary_crypto_market


def _legacy_reconstruct(d: dict) -> Market:
    """Verbatim copy of the (now-removed) ``_market_from_crypto_dict`` logic the
    five strategies duplicated — the reference the consolidation must match."""
    market_id = str(d.get("condition_id") or d.get("id") or "")
    up_price = float(d.get("up_price") or 0.0)
    down_price = float(d.get("down_price") or 0.0)
    liquidity = max(0.0, float(d.get("liquidity") or 0.0))
    slug = d.get("slug") or market_id
    question = d.get("question") or slug
    end_date = None
    end_time_raw = d.get("end_time")
    if isinstance(end_time_raw, str) and end_time_raw.strip():
        try:
            end_date = datetime.fromisoformat(end_time_raw.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass
    raw_token_ids = d.get("clob_token_ids") or []
    clob_token_ids = [str(t).strip() for t in raw_token_ids if str(t).strip() and len(str(t).strip()) > 20]
    return Market(
        id=market_id,
        condition_id=market_id,
        question=question,
        slug=slug,
        outcome_prices=[up_price, down_price],
        liquidity=liquidity,
        end_date=end_date,
        platform="polymarket",
        clob_token_ids=clob_token_ids,
    )


def _key_fields(m: Optional[Market]) -> Any:
    if m is None:
        return None
    return (
        m.id,
        m.condition_id,
        m.question,
        m.slug,
        [float(p) for p in (m.outcome_prices or [])],
        float(m.liquidity or 0.0),
        # normalize both to aware-UTC for comparison (the only end_date delta is
        # tz-awareness on a tz-less string, which doesn't occur on real Z-stamped
        # dispatch end_time)
        (m.end_date.astimezone(timezone.utc) if (m.end_date and m.end_date.tzinfo) else m.end_date),
        [str(t) for t in (m.clob_token_ids or [])],
        m.platform,
    )


# A real-shaped crypto.update.dispatch market dict (what the worker emits).
_VALID_DICTS = [
    {
        "condition_id": "0xcond1",
        "id": "m1",
        "slug": "bitcoin-up-or-down-15m",
        "question": "Bitcoin Up or Down?",
        "up_price": 0.55,
        "down_price": 0.45,
        "liquidity": 1234.5,
        "end_time": "2026-05-28T23:14:00Z",
        "clob_token_ids": ["a" * 30, "b" * 30],
        "up_token_index": 0,
        "down_token_index": 1,
        "is_live": True,
        "seconds_left": 420,
    },
    {  # no liquidity, extreme prices, condition_id only
        "condition_id": "0xcond2",
        "up_price": 0.99,
        "down_price": 0.01,
        "end_time": "2026-05-28T23:30:00Z",
        "clob_token_ids": ["c" * 30, "d" * 30],
    },
    {  # short/invalid token ids get filtered (len <= 20)
        "condition_id": "0xcond3",
        "up_price": 0.5,
        "down_price": 0.5,
        "clob_token_ids": ["short", "e" * 30],
    },
]


def test_canonical_matches_legacy_on_real_shaped_dicts():
    """build_binary_crypto_market produces a Market byte-equal (on every field
    the strategies read) to the legacy _market_from_crypto_dict for every
    real-shaped dispatch dict — the consolidation is behavior-preserving."""
    for d in _VALID_DICTS:
        canonical = build_binary_crypto_market(d)
        legacy = _legacy_reconstruct(d)
        assert canonical is not None, f"canonical returned None for valid dict {d}"
        assert _key_fields(canonical) == _key_fields(legacy), f"mismatch for {d}"


def test_canonical_contract_fields():
    """Pin the field mapping for a representative dict."""
    m = build_binary_crypto_market(_VALID_DICTS[0])
    assert m is not None
    assert m.id == "0xcond1" and m.condition_id == "0xcond1"
    assert m.slug == "bitcoin-up-or-down-15m"
    assert [float(p) for p in m.outcome_prices] == [0.55, 0.45]
    assert float(m.liquidity) == 1234.5
    assert m.end_date is not None
    assert [str(t) for t in m.clob_token_ids] == ["a" * 30, "b" * 30]
    assert m.platform == "polymarket"


def test_canonical_returns_none_on_unusable_rows():
    """The only behavioral delta vs legacy: rows with no usable price or id —
    which the strategies' own gates reject anyway — yield None (legacy built a
    [0,0]-price Market). Call sites guard this with `if market is None: continue`."""
    assert build_binary_crypto_market({"condition_id": "x"}) is None  # no prices
    assert build_binary_crypto_market({"up_price": 0.5, "down_price": 0.5}) is None  # no id/slug
    assert build_binary_crypto_market({"condition_id": "x", "up_price": 0.5}) is None  # half price
