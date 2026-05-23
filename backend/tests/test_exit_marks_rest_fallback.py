"""Regression tests for live exit-mark pricing (_collect_live_exit_marks).

Incident 2026-05-22 (order ec5d6e23): a live buy_no position rode
0.90 -> 0 over ~2h while our mark stayed frozen at entry (0.935).  The
WS book went stale during the illiquid sell-off, and the CLOB
/midpoints REST fallback that was *supposed* to re-price stale-WS
tokens silently discarded its result due to an indentation bug — the
merge loop was nested under the ``elif`` branch that reset ``batch`` to
``{}``.  So ``clob_mid_prices`` was never populated, the exit evaluator
ran on the frozen mark, and every stop saw 0.935 and never fired.

These tests pin the contract:
  * stale WS + allow_rest -> REST midpoints ARE merged (the bug fix),
  * allow_rest=False (orchestrator hot path) -> NO REST call,
  * fresh WS -> WS prices win, REST is not consulted.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import services.ws_feeds as ws_feeds
from services.trader_orchestrator import position_lifecycle


class _FreshCache:
    """WS cache that returns a fresh mid for every token."""

    def __init__(self, mid: float = 0.85, age_s: float = 1.0):
        self._mid = mid
        self._age = age_s

    def get_mid_price(self, token_id):  # noqa: ARG002
        return self._mid

    def staleness(self, token_id):  # noqa: ARG002
        return self._age


def _install(monkeypatch, *, feed_manager, batch, rest_calls):
    monkeypatch.setattr(ws_feeds, "get_feed_manager", lambda: feed_manager)

    @contextlib.asynccontextmanager
    async def _fake_release_conn(_session):
        yield

    monkeypatch.setattr(position_lifecycle, "release_conn", _fake_release_conn)

    async def _fake_batch(token_ids):
        rest_calls.append(list(token_ids))
        return batch

    monkeypatch.setattr(
        position_lifecycle.polymarket_client, "get_midpoints_batch", _fake_batch
    )


@pytest.mark.asyncio
async def test_stale_ws_falls_back_to_rest_midpoints(monkeypatch):
    # WS feed not started -> yields no prices -> all tokens unresolved.
    rest_calls: list = []
    _install(
        monkeypatch,
        feed_manager=SimpleNamespace(_started=False, cache=None),
        batch={"tok1": 0.42, "tok2": -1.0, "tok3": 0.0, "tok4": None},
        rest_calls=rest_calls,
    )

    ws, clob = await position_lifecycle._collect_live_exit_marks(
        token_ids=["tok1", "tok2", "tok3", "tok4"],
        session=object(),
        allow_rest=True,
    )

    assert ws == {}
    # tok2 (negative) and tok4 (None) are dropped; 0.0 is a valid mark.
    assert clob == {"tok1": 0.42, "tok3": 0.0}
    assert rest_calls == [["tok1", "tok2", "tok3", "tok4"]]


@pytest.mark.asyncio
async def test_hot_path_gate_blocks_rest(monkeypatch):
    rest_calls: list = []
    _install(
        monkeypatch,
        feed_manager=SimpleNamespace(_started=False, cache=None),
        batch={"tok1": 0.42},
        rest_calls=rest_calls,
    )

    ws, clob = await position_lifecycle._collect_live_exit_marks(
        token_ids=["tok1"],
        session=object(),
        allow_rest=False,
    )

    assert ws == {}
    assert clob == {}
    assert rest_calls == []  # no REST on the hot path


@pytest.mark.asyncio
async def test_fresh_ws_wins_and_skips_rest(monkeypatch):
    rest_calls: list = []
    _install(
        monkeypatch,
        feed_manager=SimpleNamespace(_started=True, cache=_FreshCache(mid=0.91, age_s=1.0)),
        batch={"tok1": 0.42},
        rest_calls=rest_calls,
    )

    ws, clob = await position_lifecycle._collect_live_exit_marks(
        token_ids=["tok1"],
        session=object(),
        allow_rest=True,
    )

    assert ws == {"tok1": 0.91}
    assert clob == {}
    assert rest_calls == []  # WS was fresh, REST never consulted


@pytest.mark.asyncio
async def test_empty_token_list_is_noop(monkeypatch):
    rest_calls: list = []
    _install(
        monkeypatch,
        feed_manager=SimpleNamespace(_started=True, cache=_FreshCache()),
        batch={},
        rest_calls=rest_calls,
    )
    ws, clob = await position_lifecycle._collect_live_exit_marks(
        token_ids=[], session=object(), allow_rest=True
    )
    assert ws == {} and clob == {}
    assert rest_calls == []
