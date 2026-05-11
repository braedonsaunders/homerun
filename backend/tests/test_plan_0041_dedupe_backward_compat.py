"""Plan 0041 regression: dedupe-key backward compatibility.

The Plan 0041 audit (Bug 1) found that an early implementation passed
``intended_trader_id`` to ``make_dedupe_key`` as the 4th positional
arg unconditionally. ``make_dedupe_key`` packs parts via
``"|".join(str(p or "") for p in parts)`` — a trailing ``None`` adds an
empty trailing component, shifting the hash for every legacy
(singleton-emitted) signal. The first redeploy after that change would
orphan every in-flight ``trade_signals`` row.

These tests lock in the contract: ``_opportunity_dedupe_key`` MUST
produce the same hash as the pre-Plan-0041 3-arg
``make_dedupe_key(stable_id, strategy, market_id)`` form when
``intended_trader_id`` is missing/None/empty-string, and a different
hash when it's actually populated.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
SERVICES_ROOT = BACKEND_ROOT / "services"
if str(SERVICES_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICES_ROOT))

from services.intent_runtime import _opportunity_dedupe_key
from services.signal_bus import make_dedupe_key


def _opp(
    *,
    stable_id: str = "sid-1",
    strategy: str = "strat",
    intended_trader_id: object = None,
) -> SimpleNamespace:
    """Minimal opportunity stub — only the three attributes the helper reads."""
    return SimpleNamespace(
        stable_id=stable_id,
        strategy=strategy,
        intended_trader_id=intended_trader_id,
    )


def test_unscoped_opportunity_matches_legacy_three_arg_dedupe_key():
    """When ``intended_trader_id`` is None, the dedupe key MUST equal
    the pre-Plan-0041 3-arg form. Without this guarantee every existing
    ``trade_signals`` row gets orphaned on the deploy that lands Plan
    0041."""
    market_id = "market-xyz"
    opp = _opp(intended_trader_id=None)

    actual = _opportunity_dedupe_key(opp, market_id)
    expected_legacy = make_dedupe_key("sid-1", "strat", market_id)

    assert actual == expected_legacy


def test_empty_string_intended_trader_id_is_treated_as_unscoped():
    """``intended_trader_id=""`` is not a real scope — opportunities
    that fell through the singleton fallback path land with empty
    strings and must still hash identically to legacy."""
    opp = _opp(intended_trader_id="")

    assert _opportunity_dedupe_key(opp, "m1") == make_dedupe_key("sid-1", "strat", "m1")


def test_whitespace_intended_trader_id_is_treated_as_unscoped():
    """Defensive: a string of only whitespace must not silently become
    a unique scope key — that would split a singleton emission's rows
    across two dedupe entries by accident."""
    opp = _opp(intended_trader_id="   ")

    assert _opportunity_dedupe_key(opp, "m1") == make_dedupe_key("sid-1", "strat", "m1")


def test_per_trader_scope_produces_distinct_hash():
    """When a real trader id is set, the dedupe key MUST differ from
    both the legacy form AND any other trader's form, so each
    per-trader clone owns its own ``trade_signals`` row."""
    market_id = "m1"
    opp_unscoped = _opp(intended_trader_id=None)
    opp_trader_a = _opp(intended_trader_id="trader-a")
    opp_trader_b = _opp(intended_trader_id="trader-b")

    key_unscoped = _opportunity_dedupe_key(opp_unscoped, market_id)
    key_a = _opportunity_dedupe_key(opp_trader_a, market_id)
    key_b = _opportunity_dedupe_key(opp_trader_b, market_id)

    assert key_a != key_unscoped
    assert key_b != key_unscoped
    assert key_a != key_b


def test_per_trader_scope_matches_explicit_four_arg_call():
    """Round-trip: the helper's 4-arg form ``make_dedupe_key(...)``
    must produce identical hashes when called directly with the
    truthy trader id — that's the only form per-trader rows are
    looked up under."""
    market_id = "m1"
    opp = _opp(intended_trader_id="trader-a")

    expected = make_dedupe_key("sid-1", "strat", market_id, "trader-a")
    assert _opportunity_dedupe_key(opp, market_id) == expected


def test_helper_strips_whitespace_around_real_trader_id():
    """Whitespace on a real id (e.g. trader name with stray padding
    from JSON) must be normalized so the hash is stable."""
    market_id = "m1"
    padded = _opp(intended_trader_id="  trader-a  ")
    clean = _opp(intended_trader_id="trader-a")

    assert _opportunity_dedupe_key(padded, market_id) == _opportunity_dedupe_key(clean, market_id)


def test_make_dedupe_key_trailing_none_collision_documented():
    """Lock in the gotcha that motivated the fix: passing ``None`` as
    the 4th arg to ``make_dedupe_key`` produces a DIFFERENT hash from
    the 3-arg form. Future readers should not reintroduce the
    unconditional 4-arg call site."""
    legacy = make_dedupe_key("sid", "strat", "m1")
    naive = make_dedupe_key("sid", "strat", "m1", None)

    assert legacy != naive, (
        "make_dedupe_key trailing-None should differ from 3-arg form; "
        "if this assert ever flips, the helper changed semantics and "
        "_opportunity_dedupe_key can be simplified."
    )
