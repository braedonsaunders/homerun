"""Live tennis match-state feed (Live Tennis API).

Polls the Live Tennis API's live-scores + fixtures endpoints and normalizes
per-match state (set/game/point score, who is serving, break-point flag)
keyed for matching against Polymarket tennis market slugs/titles.

Why this exists
---------------

``services/strategies/sports_overreaction_fader.py`` already targets tennis
markets (its ``SPORT_KEYWORDS`` includes ``tennis``/``atp``/``wta``), but it
works from price action alone. There is no data source feeding actual match
state, so the strategy can't tell "price moved because a break just happened"
from "price moved on noise". This feed supplies that state.

Structurally it mirrors the pollable-REST feed shape of
:mod:`services.chainlink_direct_feed`: a reconnect/poll loop, an ``on_update``
callback per normalized record, a process-wide singleton, and a
:class:`~utils.feed_availability.FeedAvailability` latch so a feed with no API
key never enters its poll loop.

Graceful degradation on a free key
----------------------------------

The Live Tennis API free tier is 30 req/min / 100 req/day — enough to develop,
test, and run slow-cadence checks, but not continuous in-play polling across
many live matches. The feed degrades rather than failing:

- **No API key** -> latched ``DISABLED`` (never polls). Testable one-shot.
- **HTTP 401/403** (bad/revoked key) -> latched ``DISABLED``.
- **HTTP 429** (quota/rate exhausted) -> ``degraded`` mode: back off to a
  slower cadence and fall back to the cheaper fixtures endpoint for upcoming-
  match context, retrying the live endpoint on the slow cadence. Status
  reports ``PARTIAL`` while degraded. The feed keeps serving what it can
  instead of latching off or hammering a 429.

Vendor disclosure: this feed is authored by the operator of the Live Tennis
API (https://livetennisapi.com). Free key signup:
https://livetennisapi.com/subscribe/free  Docs: https://docs.livetennisapi.com

Reference: https://docs.livetennisapi.com
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import httpx

# --- Self-contained example shims -------------------------------------------
# In the homerun codebase, a real feed would import these from utils.* (this is
# modeled on services/chainlink_direct_feed.py). To keep this docs example
# standalone — runnable with only httpx — minimal, faithful equivalents of the
# feed-availability latch and the logger are inlined below. Swap them for the
# project's own utils.feed_availability / utils.logger when wiring into core.
import logging
from enum import Enum


class FeedStatus(str, Enum):
    UNINITIALIZED = "uninitialized"
    CONNECTING = "connecting"
    HEALTHY = "healthy"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class FeedAvailability:
    """Credential latch: once creds are missing or auth fails, stays disabled
    until explicitly re-armed. Mirrors utils.feed_availability.FeedAvailability."""

    has_credentials: Callable[[], bool]
    name: str = "feed"
    _disabled: bool = False
    _disabled_reason: Optional[str] = None

    def check(self) -> bool:
        if self._disabled:
            return False
        if not self.has_credentials():
            self._disabled = True
            self._disabled_reason = "missing_credentials"
            return False
        return True

    def latch_auth_failure(self, reason: str = "auth_failed") -> None:
        self._disabled = True
        self._disabled_reason = reason

    def rearm(self) -> None:
        self._disabled = False
        self._disabled_reason = None

    @property
    def is_disabled(self) -> bool:
        return self._disabled

    @property
    def disabled_reason(self) -> Optional[str]:
        return self._disabled_reason


class _StructLogger:
    """Minimal structlog-style adapter: accepts arbitrary keyword context and
    appends it to the message, so this example runs on the stdlib logging module
    without the project's structlog dependency."""

    def __init__(self, name: str) -> None:
        self._log = logging.getLogger(name)

    def _emit(self, level: int, event: str, *args: object, **kw: object) -> None:
        if args:
            try:
                event = event % args
            except Exception:
                event = " ".join([event, *(str(a) for a in args)])
        if kw:
            event = event + " " + " ".join(f"{k}={v}" for k, v in kw.items())
        self._log.log(level, event)

    def debug(self, event: str, *args: object, **kw: object) -> None:
        self._emit(logging.DEBUG, event, *args, **kw)

    def info(self, event: str, *args: object, **kw: object) -> None:
        self._emit(logging.INFO, event, *args, **kw)

    def warning(self, event: str, *args: object, **kw: object) -> None:
        self._emit(logging.WARNING, event, *args, **kw)

    def error(self, event: str, *args: object, **kw: object) -> None:
        self._emit(logging.ERROR, event, *args, **kw)

    def exception(self, event: str, *args: object, **kw: object) -> None:
        self._log.exception(event % args if args else event)


def get_logger(name: str) -> "_StructLogger":
    return _StructLogger(name)


logger = get_logger(__name__)
# ---------------------------------------------------------------------------


LIVETENNIS_API_BASE = "https://api.livetennisapi.com/api/public/v1"

# Slow enough to be considerate of the shared upstream, fast enough to catch a
# break within one game. Free-tier callers should raise this (or rely on the
# 429-driven degrade below) to stay under 100 req/day.
POLL_INTERVAL_SECONDS = 20.0
# Cadence used after a 429 — slow spot-checks that keep the daily quota alive.
DEGRADED_POLL_INTERVAL_SECONDS = 120.0
REQUEST_TIMEOUT_SECONDS = 10.0

# The four "game point" tokens tennis scoring uses outside a tiebreak. Inside a
# tiebreak points are plain integers ("6", "7", ...) — break-point derivation is
# undefined there, so any token outside this set makes _derive_break_point
# return None (matches the API's own break-point contract).
_POINT_VOCAB = frozenset({"0", "15", "30", "40", "AD", "A"})


def _surname(name: str) -> str:
    """Last whitespace-delimited token of a player name, lowercased.

    Player-name-to-market matching keys off surnames because Polymarket tennis
    slugs/titles carry surnames far more reliably than full names.
    """
    tokens = str(name or "").strip().split()
    return tokens[-1].lower() if tokens else ""


def _market_key(p1_name: str, p2_name: str) -> str:
    """Order-independent surname key, e.g. ("Carlos Alcaraz", "J. Sinner")
    -> ``"alcaraz-sinner"``.

    Sorted so the same pairing keys identically regardless of which player the
    upstream lists first vs. how a market slug orders them. Consumers still get
    the raw names on the record for fuzzier matching.
    """
    a, b = _surname(p1_name), _surname(p2_name)
    return "-".join(sorted(k for k in (a, b) if k))


def _derive_break_point(server: Optional[int], points: object) -> Optional[bool]:
    """Return True if the receiver holds a break point, False if not, or None
    when it cannot be determined.

    Contract (mirrors the Live Tennis API): break point iff the receiver is at
    ``AD``, or the receiver is at ``40`` while the server is at ``0``/``15``/
    ``30``. Never inside a tiebreak, and undefined when the server or points are
    unknown.
    """
    if server not in (1, 2):
        return None
    if not isinstance(points, (list, tuple)) or len(points) < 2:
        return None
    server_idx = 0 if server == 1 else 1
    receiver_idx = 1 - server_idx
    server_pts = str(points[server_idx]).upper()
    receiver_pts = str(points[receiver_idx]).upper()
    # A non-standard token on either side (tiebreak integers, empty, "40A",
    # ...) means standard break-point logic doesn't apply -> undefined.
    if server_pts not in _POINT_VOCAB or receiver_pts not in _POINT_VOCAB:
        return None
    if receiver_pts in ("AD", "A"):
        return True
    if receiver_pts == "40" and server_pts in ("0", "15", "30"):
        return True
    return False


@dataclass
class TennisMatchState:
    """Normalized per-match state fired to ``on_update`` and cached in the feed.

    ``break_point`` is tri-state: ``True``/``False`` when derivable, ``None``
    when undefined (no server, missing points, or a tiebreak).
    """

    match_id: str
    player1: str
    player2: str
    status: str
    market_key: str
    sets: list = field(default_factory=list)
    games: list = field(default_factory=list)
    points: list = field(default_factory=list)
    server: Optional[int] = None
    break_point: Optional[bool] = None
    received_at_ms: int = 0
    source: str = "livetennis"


def _normalize_match(raw: object) -> Optional[TennisMatchState]:
    """Turn one raw match record from the API into a :class:`TennisMatchState`.

    Returns ``None`` for anything that isn't a usable match object (missing id
    or players) so a single malformed row can't poison a poll.
    """
    if not isinstance(raw, dict):
        return None
    match_id = raw.get("id")
    if match_id is None:
        return None

    players = raw.get("players") if isinstance(raw.get("players"), dict) else {}
    p1 = players.get("p1") if isinstance(players.get("p1"), dict) else {}
    p2 = players.get("p2") if isinstance(players.get("p2"), dict) else {}
    p1_name = str(p1.get("name") or "").strip()
    p2_name = str(p2.get("name") or "").strip()
    if not p1_name or not p2_name:
        return None

    score = raw.get("score") if isinstance(raw.get("score"), dict) else {}
    sets = score.get("sets") if isinstance(score.get("sets"), list) else []
    games = score.get("games") if isinstance(score.get("games"), list) else []
    points = score.get("points") if isinstance(score.get("points"), list) else []

    server_raw = score.get("server")
    server = server_raw if server_raw in (1, 2) else None

    return TennisMatchState(
        match_id=str(match_id),
        player1=p1_name,
        player2=p2_name,
        status=str(raw.get("status") or "").strip().lower(),
        market_key=_market_key(p1_name, p2_name),
        sets=list(sets),
        games=list(games),
        points=list(points),
        server=server,
        break_point=_derive_break_point(server, points),
        received_at_ms=int(time.time() * 1000),
    )


class LiveTennisFeed:
    """Pollable Live Tennis API client for live match state.

    Wire it with a callback and a key getter::

        feed = get_livetennis_feed(get_api_key=lambda: settings.LIVETENNIS_API_KEY or "")
        feed.on_update(lambda state: strategy.observe_tennis(state))
        await feed.start()
        ...
        await feed.stop()

    ``start()`` is a no-op that latches ``DISABLED`` when no key is present, so
    it is always safe to call.
    """

    def __init__(
        self,
        get_api_key: Callable[[], str],
        api_base: str = LIVETENNIS_API_BASE,
        poll_interval: float = POLL_INTERVAL_SECONDS,
        degraded_poll_interval: float = DEGRADED_POLL_INTERVAL_SECONDS,
    ) -> None:
        self._get_api_key = get_api_key
        self._api_base = api_base.rstrip("/")
        self._poll_interval = poll_interval
        self._degraded_poll_interval = degraded_poll_interval
        self._task: Optional[asyncio.Task] = None
        self._stopped = False
        self._on_update: Optional[Callable[[TennisMatchState], None]] = None
        self._availability = FeedAvailability(
            has_credentials=lambda: bool(self._get_api_key()),
            name="livetennis",
        )
        self._latest: dict[str, TennisMatchState] = {}
        self._degraded = False

    @property
    def status(self) -> FeedStatus:
        if self._availability.is_disabled:
            return FeedStatus.DISABLED
        if not self.started:
            return FeedStatus.UNINITIALIZED
        if self._degraded:
            return FeedStatus.PARTIAL
        return FeedStatus.HEALTHY if self._latest else FeedStatus.CONNECTING

    @property
    def started(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def degraded(self) -> bool:
        """True once a 429 has forced the slow, fixtures-only cadence."""
        return self._degraded

    @property
    def disabled_reason(self) -> Optional[str]:
        return self._availability.disabled_reason

    def get_match(self, match_id: str) -> Optional[TennisMatchState]:
        return self._latest.get(str(match_id))

    def get_all_matches(self) -> dict[str, TennisMatchState]:
        return dict(self._latest)

    def find_by_market_key(self, market_key: str) -> list[TennisMatchState]:
        """All cached matches whose surname key equals ``market_key``."""
        key = str(market_key or "").lower()
        return [m for m in self._latest.values() if m.market_key == key]

    def on_update(self, callback: Callable[[TennisMatchState], None]) -> None:
        self._on_update = callback

    def rearm(self) -> None:
        """Clear the disabled latch and the degraded flag — call after the API
        key is updated."""
        self._availability.rearm()
        self._degraded = False

    async def start(self) -> None:
        """Begin polling. No-op (latched ``DISABLED``) when no key is present."""
        if self._task and not self._task.done():
            return
        if not self._availability.check():
            logger.info(
                "LiveTennisFeed disabled — no API key. Set a Live Tennis API "
                "key to enable (free: https://livetennisapi.com/subscribe/free)."
            )
            return
        self._stopped = False
        self._task = asyncio.create_task(self._run_loop())
        logger.info("LiveTennisFeed: started")

    async def stop(self) -> None:
        self._stopped = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("LiveTennisFeed: stopped")

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self._get_api_key(), "Accept": "application/json"}

    async def _run_loop(self) -> None:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            while not self._stopped:
                if self._availability.is_disabled:
                    return
                try:
                    if self._degraded:
                        # Degraded: keep upcoming context warm via the cheaper
                        # fixtures endpoint, then spot-check live to see if the
                        # quota window has recovered.
                        await self._poll_fixtures(client)
                        await self._poll_live(client)
                    else:
                        await self._poll_live(client)
                except asyncio.CancelledError:
                    return
                except Exception as exc:  # transient — retry next tick
                    logger.debug("LiveTennisFeed poll error", error=str(exc))
                interval = (
                    self._degraded_poll_interval if self._degraded else self._poll_interval
                )
                try:
                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    return

    async def _poll_live(self, client: httpx.AsyncClient) -> None:
        """Poll live matches. Sets ``degraded`` on 429; latches disabled on
        401/403; clears ``degraded`` on a healthy live read."""
        data = await self._get(client, "/matches", params={"status": "live"})
        if data is None:
            return
        # A successful live read means the quota window is open again.
        self._degraded = False
        self._ingest(data)

    async def _poll_fixtures(self, client: httpx.AsyncClient) -> None:
        """Poll upcoming fixtures — the cheap fallback used while degraded."""
        data = await self._get(client, "/fixtures")
        if data is None:
            return
        self._ingest(data)

    async def _get(
        self,
        client: httpx.AsyncClient,
        path: str,
        params: Optional[dict] = None,
    ) -> Optional[list]:
        """GET one endpoint and return its ``data`` list, or ``None`` when the
        caller should skip this tick (transient error, auth latch, or 429)."""
        api_key = self._get_api_key()
        if not api_key:
            self._availability.latch_auth_failure(reason="missing_credentials")
            logger.warning("LiveTennisFeed: key cleared mid-flight, latching disabled")
            return None

        url = f"{self._api_base}{path}"
        try:
            resp = await client.get(url, headers=self._headers(), params=params)
        except httpx.HTTPError:
            return None  # transient — retry next tick

        if resp.status_code in (401, 403):
            # Bad/revoked key — latch disabled instead of storming the upstream.
            # Operator rotates the key and calls rearm().
            self._availability.latch_auth_failure(reason=f"http_{resp.status_code}")
            logger.error(
                "LiveTennisFeed: auth rejected (%s), latching disabled. "
                "Rotate the Live Tennis API key and call rearm().",
                resp.status_code,
            )
            return None

        if resp.status_code == 429:
            # Quota/rate exhausted (free tier is 100/day). Degrade to the slow,
            # fixtures-first cadence rather than latching off or hammering 429.
            if not self._degraded:
                logger.warning(
                    "LiveTennisFeed: 429 rate/quota limit — degrading to slow "
                    "fixtures-only cadence (%.0fs). Raise the tier for "
                    "continuous in-play polling.",
                    self._degraded_poll_interval,
                )
            self._degraded = True
            return None

        if resp.status_code != 200:
            return None

        try:
            body = resp.json()
        except ValueError:
            return None
        data = body.get("data") if isinstance(body, dict) else None
        return data if isinstance(data, list) else []

    def _ingest(self, rows: list) -> None:
        for raw in rows:
            state = _normalize_match(raw)
            if state is None:
                continue
            self._latest[state.match_id] = state
            if self._on_update is not None:
                try:
                    self._on_update(state)
                except Exception:
                    logger.exception("LiveTennisFeed on_update callback raised")


# ---------------------------------------------------------------------------
# Singleton wiring
# ---------------------------------------------------------------------------

_instance: Optional[LiveTennisFeed] = None


def get_livetennis_feed(
    get_api_key: Optional[Callable[[], str]] = None,
) -> LiveTennisFeed:
    """Return the process-wide LiveTennisFeed.

    First call must supply an API-key getter. Subsequent calls return the same
    instance and ignore the argument.
    """
    global _instance
    if _instance is None:
        if get_api_key is None:
            raise RuntimeError(
                "LiveTennisFeed not yet initialized; pass get_api_key on first call"
            )
        _instance = LiveTennisFeed(get_api_key=get_api_key)
    return _instance
