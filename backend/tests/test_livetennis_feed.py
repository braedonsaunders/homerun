"""Tests for the Live Tennis match-state feed (services/livetennis_feed.py).

Mirrors ``test_chainlink_direct_feed.py``: the credential latch (disabled when
no key), the auth-failure latch, normalization of a raw match record, the
break-point derivation contract, and the free-key 429 -> degraded behavior.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.livetennis_feed import (
    LiveTennisFeed,
    TennisMatchState,
    _derive_break_point,
    _market_key,
    _normalize_match,
)
from utils.feed_availability import FeedStatus


def _match(
    match_id="12345",
    p1="Carlos Alcaraz",
    p2="Jannik Sinner",
    status="live",
    server=1,
    points=("40", "30"),
    sets=(["6", "4"],),
    games=([3, 2],),
):
    return {
        "id": match_id,
        "status": status,
        "players": {"p1": {"name": p1}, "p2": {"name": p2}},
        "score": {
            "sets": list(sets),
            "games": list(games),
            "points": list(points),
            "server": server,
        },
    }


def _envelope(*matches):
    return {"data": list(matches)}


def _response(status_code, json_body=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=json_body if json_body is not None else {})
    return resp


# ---------------------------------------------------------------------------
# Pure helpers: normalization + break-point + market key
# ---------------------------------------------------------------------------


def test_market_key_is_order_independent_and_surname_based():
    assert _market_key("Carlos Alcaraz", "J. Sinner") == "alcaraz-sinner"
    # Reversed order + different formatting keys identically.
    assert _market_key("Jannik Sinner", "Carlos Alcaraz") == "alcaraz-sinner"


def test_normalize_match_extracts_state():
    state = _normalize_match(_match())
    assert isinstance(state, TennisMatchState)
    assert state.match_id == "12345"
    assert state.player1 == "Carlos Alcaraz"
    assert state.player2 == "Jannik Sinner"
    assert state.status == "live"
    assert state.market_key == "alcaraz-sinner"
    assert state.server == 1
    assert state.sets == [["6", "4"]]
    assert state.source == "livetennis"
    assert state.received_at_ms > 0


def test_normalize_match_rejects_incomplete_rows():
    assert _normalize_match({"status": "live"}) is None  # no id
    assert _normalize_match({"id": "x", "players": {}}) is None  # no players
    assert _normalize_match("not a dict") is None
    # Server outside {1, 2} normalizes to None rather than leaking a bad value.
    state = _normalize_match(_match(server=0))
    assert state.server is None


def test_break_point_receiver_forty_vs_server_below():
    # Server (p1) at 30, receiver (p2) at 40 -> break point for the receiver.
    assert _derive_break_point(1, ["30", "40"]) is True
    # Server at 40 too -> deuce, not a break point.
    assert _derive_break_point(1, ["40", "40"]) is False


def test_break_point_receiver_advantage():
    # Receiver (p2) holds advantage.
    assert _derive_break_point(1, ["40", "AD"]) is True
    # Server holds advantage -> not a break point.
    assert _derive_break_point(1, ["AD", "40"]) is False


def test_break_point_undefined_cases():
    # Null server -> undefined.
    assert _derive_break_point(None, ["40", "AD"]) is None
    # Tiebreak integer points are outside the game-point vocabulary -> undefined.
    assert _derive_break_point(1, ["6", "5"]) is None
    # Missing / short points -> undefined.
    assert _derive_break_point(1, []) is None


def test_normalize_flows_break_point_from_server_two():
    # p2 serving (server=2), p1 the receiver at AD -> break point.
    state = _normalize_match(_match(server=2, points=("AD", "40")))
    assert state.break_point is True


# ---------------------------------------------------------------------------
# Credential latch — the graceful-degrade-on-no-key path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feed_disabled_when_key_missing():
    feed = LiveTennisFeed(get_api_key=lambda: "")
    await feed.start()
    assert feed.started is False
    assert feed.status == FeedStatus.DISABLED
    assert feed.disabled_reason == "missing_credentials"


@pytest.mark.asyncio
async def test_feed_latches_on_persistent_auth_failure():
    feed = LiveTennisFeed(get_api_key=lambda: "twjp_key")
    client = AsyncMock()
    client.get.return_value = _response(401)

    await feed._poll_live(client)
    assert feed._availability.is_disabled is True
    assert feed.disabled_reason == "http_401"
    assert feed.status == FeedStatus.DISABLED

    # rearm() clears the latch and the degraded flag.
    feed.rearm()
    assert feed._availability.is_disabled is False


# ---------------------------------------------------------------------------
# Happy path + free-key 429 graceful degradation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_poll_live_ingests_and_fires_callback():
    feed = LiveTennisFeed(get_api_key=lambda: "twjp_key")
    captured = []
    feed.on_update(lambda state: captured.append(state))

    client = AsyncMock()
    client.get.return_value = _response(200, _envelope(_match()))

    await feed._poll_live(client)

    assert len(captured) == 1
    assert captured[0].match_id == "12345"
    assert captured[0].break_point is False  # 40 vs 30, server ahead
    assert "12345" in feed.get_all_matches()
    assert feed.find_by_market_key("alcaraz-sinner")[0].match_id == "12345"
    assert feed.degraded is False


@pytest.mark.asyncio
async def test_free_key_429_degrades_gracefully():
    """A 429 on the live endpoint must NOT latch disabled — it degrades to the
    slow, fixtures-first cadence and keeps serving what it can."""
    feed = LiveTennisFeed(get_api_key=lambda: "twjp_key")

    client = AsyncMock()
    client.get.return_value = _response(429)

    await feed._poll_live(client)
    assert feed.degraded is True
    assert feed._availability.is_disabled is False  # not disabled — just slowed

    # While degraded, fixtures still populate the cache (the cheap fallback).
    client.get.return_value = _response(200, _envelope(_match(status="upcoming")))
    await feed._poll_fixtures(client)
    assert "12345" in feed.get_all_matches()

    # A later healthy live read clears the degraded flag.
    client.get.return_value = _response(200, _envelope(_match()))
    await feed._poll_live(client)
    assert feed.degraded is False


@pytest.mark.asyncio
async def test_malformed_rows_do_not_break_a_poll():
    feed = LiveTennisFeed(get_api_key=lambda: "twjp_key")
    captured = []
    feed.on_update(lambda state: captured.append(state))

    client = AsyncMock()
    client.get.return_value = _response(
        200,
        {"data": ["garbage", {"id": "no-players"}, _match(match_id="999")]},
    )

    await feed._poll_live(client)
    # Only the one well-formed row survives.
    assert [s.match_id for s in captured] == ["999"]


@pytest.mark.asyncio
async def test_singleton_requires_key_getter_on_first_call():
    import services.livetennis_feed as mod

    mod._instance = None
    with pytest.raises(RuntimeError):
        mod.get_livetennis_feed()

    feed = mod.get_livetennis_feed(get_api_key=lambda: "twjp_key")
    # Subsequent calls return the same instance and ignore the argument.
    assert mod.get_livetennis_feed(get_api_key=lambda: "other") is feed
    mod._instance = None
