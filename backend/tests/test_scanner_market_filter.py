"""Unit tests for the tag-whitelist filter on the scanner ingest path.

These are pure-Python tests against ``ArbitrageScanner._apply_market_tag_whitelist``
— no Postgres required. The DB-touching paths (``_load_market_filter_tags``
and the aggregator hook) are covered by integration tests that need the
shared test DB harness.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.scanner import ArbitrageScanner  # noqa: E402


def _market(
    market_id: str,
    *,
    tags: list[str] | None = None,
    event_slug: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=market_id,
        tags=list(tags or []),
        event_slug=event_slug,
    )


def _event(
    slug: str,
    *,
    tags: list[str] | None = None,
    markets: list[SimpleNamespace] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"evt-{slug}",
        slug=slug,
        tags=list(tags or []),
        markets=list(markets or []),
    )


def test_empty_whitelist_returns_inputs_unchanged():
    m1 = _market("m1", tags=["crypto"], event_slug="ev1")
    m2 = _market("m2", tags=["politics"], event_slug="ev2")
    e1 = _event("ev1", markets=[m1])
    e2 = _event("ev2", markets=[m2])

    events, markets = ArbitrageScanner._apply_market_tag_whitelist(
        [e1, e2], [m1, m2], frozenset()
    )
    assert events == [e1, e2]
    assert markets == [m1, m2]


def test_whitelist_keeps_markets_with_matching_market_tag():
    m1 = _market("m1", tags=["crypto"], event_slug="ev1")
    m2 = _market("m2", tags=["politics"], event_slug="ev2")
    e1 = _event("ev1", markets=[m1])
    e2 = _event("ev2", markets=[m2])

    events, markets = ArbitrageScanner._apply_market_tag_whitelist(
        [e1, e2], [m1, m2], frozenset({"crypto"})
    )
    assert markets == [m1]
    assert events == [e1]
    assert e1.markets == [m1]


def test_whitelist_keeps_markets_via_event_tag_union():
    """Markets without their own tag still pass if their event has it."""
    m1 = _market("m1", tags=[], event_slug="sports-1")
    e1 = _event("sports-1", tags=["sports", "nba"], markets=[m1])

    events, markets = ArbitrageScanner._apply_market_tag_whitelist(
        [e1], [m1], frozenset({"nba"})
    )
    assert markets == [m1]
    assert events == [e1]


def test_whitelist_drops_markets_with_no_intersection():
    m1 = _market("m1", tags=["sports"], event_slug="sports-1")
    e1 = _event("sports-1", tags=["nba"], markets=[m1])

    events, markets = ArbitrageScanner._apply_market_tag_whitelist(
        [e1], [m1], frozenset({"crypto"})
    )
    assert markets == []
    assert events == []


def test_whitelist_is_case_insensitive():
    m1 = _market("m1", tags=["Crypto"], event_slug="ev1")
    e1 = _event("ev1", markets=[m1])

    events, markets = ArbitrageScanner._apply_market_tag_whitelist(
        [e1], [m1], frozenset({"crypto"})
    )
    assert markets == [m1]
    assert events == [e1]


def test_whitelist_ignores_non_string_tags_and_whitespace():
    """Defensive: malformed tags on raw rows must not poison the filter."""
    m1 = SimpleNamespace(
        id="m1",
        tags=[None, 123, "  Crypto  ", "", "  "],
        event_slug="ev1",
    )
    e1 = _event("ev1", markets=[m1])

    events, markets = ArbitrageScanner._apply_market_tag_whitelist(
        [e1], [m1], frozenset({"crypto"})
    )
    assert markets == [m1]
    assert events == [e1]


def test_whitelist_drops_event_when_all_children_filtered_out():
    m1 = _market("m1", tags=["crypto"], event_slug="multi")
    m2 = _market("m2", tags=["politics"], event_slug="multi")
    e1 = _event("multi", markets=[m1, m2])

    events, markets = ArbitrageScanner._apply_market_tag_whitelist(
        [e1], [m1, m2], frozenset({"sports"})
    )
    assert markets == []
    assert events == []


def test_whitelist_keeps_partial_event_children():
    m1 = _market("m1", tags=["crypto"], event_slug="multi")
    m2 = _market("m2", tags=["politics"], event_slug="multi")
    e1 = _event("multi", markets=[m1, m2])

    events, markets = ArbitrageScanner._apply_market_tag_whitelist(
        [e1], [m1, m2], frozenset({"crypto"})
    )
    assert markets == [m1]
    assert events == [e1]
    assert e1.markets == [m1]


def test_whitelist_or_logic_multiple_tags():
    m1 = _market("m1", tags=["crypto"], event_slug="ev1")
    m2 = _market("m2", tags=["politics"], event_slug="ev2")
    m3 = _market("m3", tags=["sports"], event_slug="ev3")
    e1 = _event("ev1", markets=[m1])
    e2 = _event("ev2", markets=[m2])
    e3 = _event("ev3", markets=[m3])

    events, markets = ArbitrageScanner._apply_market_tag_whitelist(
        [e1, e2, e3], [m1, m2, m3], frozenset({"crypto", "politics"})
    )
    assert {m.id for m in markets} == {"m1", "m2"}
    assert {e.slug for e in events} == {"ev1", "ev2"}


def test_whitelist_zero_match_logs_warning(caplog):
    """When the whitelist is non-empty but matches nothing, an operator
    warning is emitted so the misconfiguration becomes visible."""
    import logging

    m1 = _market("m1", tags=["crypto"], event_slug="ev1")
    e1 = _event("ev1", markets=[m1])

    with caplog.at_level(logging.WARNING):
        events, markets = ArbitrageScanner._apply_market_tag_whitelist(
            [e1], [m1], frozenset({"sports"})
        )
    assert markets == []
    assert events == []
    assert any(
        "matched zero markets" in record.getMessage()
        for record in caplog.records
    )


def test_whitelist_drop_emits_info_log_with_reason(caplog):
    """Non-empty filter that drops some markets logs the reason code."""
    import logging

    m1 = _market("m1", tags=["crypto"], event_slug="ev1")
    m2 = _market("m2", tags=["politics"], event_slug="ev2")
    e1 = _event("ev1", markets=[m1])
    e2 = _event("ev2", markets=[m2])

    with caplog.at_level(logging.INFO):
        events, markets = ArbitrageScanner._apply_market_tag_whitelist(
            [e1, e2], [m1, m2], frozenset({"crypto"})
        )
    assert markets == [m1]
    assert any(
        "market_filter_tags_no_match" in record.getMessage()
        for record in caplog.records
    )
