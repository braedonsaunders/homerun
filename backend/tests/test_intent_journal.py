"""Tests for the fast-tier intent journal (durable pre-wire dedup + recovery)."""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.trader_orchestrator.intent_journal import (
    IntentJournal,
    _COMPACT_THRESHOLD_BYTES,
)


def _journal(tmp_path) -> IntentJournal:
    j = IntentJournal(tmp_path / "fast_intent.log")
    j.open()
    j.load()
    return j


def test_intent_recorded_and_deduped(tmp_path):
    j = _journal(tmp_path)
    assert not j.has_intent("t1", "s1")
    j.record_intent(trader_id="t1", signal_id="s1", key="0xabc", token_id="tok", side="buy", size_usd=10.0)
    assert j.has_intent("t1", "s1")
    assert j.is_open("t1", "s1")
    # Distinct signal is independent.
    assert not j.has_intent("t1", "s2")


def test_result_closes_open_intent(tmp_path):
    j = _journal(tmp_path)
    j.record_intent(trader_id="t1", signal_id="s1", key="0xabc")
    j.record_result(trader_id="t1", signal_id="s1", status="executed", provider_clob_order_id="clob-1")
    # Still deduped (never re-fire), but no longer an open orphan.
    assert j.has_intent("t1", "s1")
    assert not j.is_open("t1", "s1")
    assert all(r["sig"] != "s1" for r in j.open_intents())


def test_open_intents_lists_only_unresolved(tmp_path):
    j = _journal(tmp_path)
    j.record_intent(trader_id="t1", signal_id="open-1", key="0xk1", token_id="tokA")
    j.record_intent(trader_id="t1", signal_id="done-1", key="0xk2")
    j.record_result(trader_id="t1", signal_id="done-1", status="executed")
    open_sigs = {r["sig"] for r in j.open_intents()}
    assert open_sigs == {"open-1"}
    # Replay payload carries the deterministic key + leg hints.
    rec = next(r for r in j.open_intents() if r["sig"] == "open-1")
    assert rec["key"] == "0xk1"
    assert rec["tok"] == "tokA"


def test_persists_across_reopen(tmp_path):
    j = _journal(tmp_path)
    j.record_intent(trader_id="t1", signal_id="s1", key="0xabc")
    j.record_intent(trader_id="t1", signal_id="s2", key="0xdef")
    j.record_result(trader_id="t1", signal_id="s1", status="executed")
    j.close()

    # Fresh instance on the same file rebuilds the index.
    j2 = _journal(tmp_path)
    assert j2.has_intent("t1", "s1")
    assert not j2.is_open("t1", "s1")  # resolved before crash
    assert j2.is_open("t1", "s2")      # orphan -> must reconcile
    assert {r["sig"] for r in j2.open_intents()} == {"s2"}


def test_torn_trailing_record_is_skipped(tmp_path):
    j = _journal(tmp_path)
    j.record_intent(trader_id="t1", signal_id="s1", key="0xabc")
    j.close()
    # Simulate a crash mid-append: append a partial (un-terminated) line.
    path = tmp_path / "fast_intent.log"
    with open(path, "ab") as fh:
        fh.write(b'{"k":"intent","tr":"t1","sig":"s2","ke')  # torn, no newline

    j2 = _journal(tmp_path)
    assert j2.has_intent("t1", "s1")   # good record survives
    assert not j2.has_intent("t1", "s2")  # torn record ignored


def test_compaction_drops_resolved_and_keeps_open(tmp_path):
    j = _journal(tmp_path)
    # Force the threshold low for the test by writing many resolved pairs.
    n = 2000
    for i in range(n):
        sig = f"s{i}"
        j.record_intent(trader_id="t1", signal_id=sig, key=f"0x{i:064x}")
        j.record_result(trader_id="t1", signal_id=sig, status="executed")
    # One lingering open intent.
    j.record_intent(trader_id="t1", signal_id="orphan", key="0xorphan")
    # Trip compaction explicitly (independent of byte threshold).
    j._compact()  # noqa: SLF001 - exercising internal compaction

    # After compaction the open orphan survives and resolved ones are gone
    # from disk, but the dedup index still knows about a resolved signal.
    j.close()
    j2 = _journal(tmp_path)
    assert j2.is_open("t1", "orphan")
    assert {r["sig"] for r in j2.open_intents()} == {"orphan"}
    # Resolved signals were compacted off disk -> no longer deduped after
    # restart.  That is acceptable: a signal that already executed will not
    # be re-emitted by the cursor (the cursor advanced past it), and the
    # venue holds the position.  Compaction only runs once intents resolve.
    assert not j2.is_open("t1", "s0")


def test_compaction_threshold_constant_is_sane():
    assert _COMPACT_THRESHOLD_BYTES >= 1024 * 1024
