"""Decision-parity harness — the determinism ruler.

Two questions this answers:

  1. **Is replay deterministic?**  Replay the same window through the
     same strategy twice and assert byte-identical decisions.  This is
     the validator Phases 1 (clock) and 3 (ordering/RNG) are measured
     against — it runs today against recorded data, no live decisions
     required.

  2. **Does replay match what the bot decided live?**  Diff the
     re-derived decisions against the ``strategy.decision`` topic that
     ``signal_bus`` tees on every live emission.  This proves the
     strategy is a pure function of its recorded inputs.  It needs
     accumulated live decisions in the window, so it only becomes
     meaningful once the decision tee has been running live.

Re-derivation reuses the *real* backtester replay
(``strategy_backtester._replay_discover_opportunities``) so the harness
can't drift from how backtests actually run.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


# ── Decision identity ────────────────────────────────────────────────
#
# A decision's *identity* is the tuple of fields a faithful replay must
# reproduce.  We compare on identity, not on volatile bookkeeping
# (timestamps to the microsecond, roster hashes, sequence numbers).


@dataclass(frozen=True)
class DecisionIdentity:
    strategy_type: Optional[str]
    token_id: Optional[str]
    direction: Optional[str]
    # Prices/edge are floats; round so 1e-12 FP noise doesn't read as a
    # divergence.  Identity is about "same decision", not bit-equality of
    # a float that two code paths computed slightly differently.
    entry_price: Optional[float]
    edge_bucket: Optional[float]

    @classmethod
    def from_position(
        cls, *, strategy_type: str, position: dict[str, Any], opp_edge: float | None
    ) -> "DecisionIdentity":
        def _f(v: Any) -> Optional[float]:
            try:
                return round(float(v), 4)
            except (TypeError, ValueError):
                return None

        token = (
            position.get("token_id")
            or position.get("clob_token_id")
            or position.get("market_id")
        )
        direction = (
            position.get("direction")
            or position.get("side")
            or position.get("action")
        )
        return cls(
            strategy_type=str(strategy_type) if strategy_type else None,
            token_id=str(token) if token else None,
            direction=str(direction).lower() if direction else None,
            entry_price=_f(position.get("price") or position.get("entry_price")),
            edge_bucket=_f(opp_edge),
        )

    @classmethod
    def from_recorded_payload(cls, payload: dict[str, Any]) -> "DecisionIdentity":
        def _f(v: Any) -> Optional[float]:
            try:
                return round(float(v), 4)
            except (TypeError, ValueError):
                return None

        # The recorded decision keys on market_id (condition_id) as entity;
        # token-level identity isn't always carried, so we compare on what
        # both sides reliably have.  edge_percent is recorded as a percent.
        return cls(
            strategy_type=(
                str(payload.get("strategy_type")) if payload.get("strategy_type") else None
            ),
            token_id=str(payload.get("market_id")) if payload.get("market_id") else None,
            direction=(
                str(payload.get("direction")).lower() if payload.get("direction") else None
            ),
            entry_price=_f(payload.get("entry_price")),
            edge_bucket=_f(payload.get("edge_percent")),
        )


@dataclass
class ParityReport:
    strategy_slug: str
    window_start: datetime
    window_end: datetime
    replayed_count: int = 0
    recorded_count: int = 0
    matched: int = 0
    missing_from_replay: list[DecisionIdentity] = field(default_factory=list)
    extra_in_replay: list[DecisionIdentity] = field(default_factory=list)
    deterministic: Optional[bool] = None
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        det_ok = self.deterministic is not False
        return det_ok and not self.missing_from_replay and not self.extra_in_replay

    def summary(self) -> str:
        return (
            f"[parity {self.strategy_slug} {self.window_start:%Y-%m-%d %H:%M}"
            f"..{self.window_end:%H:%M}] "
            f"replayed={self.replayed_count} recorded={self.recorded_count} "
            f"matched={self.matched} missing={len(self.missing_from_replay)} "
            f"extra={len(self.extra_in_replay)} "
            f"deterministic={self.deterministic} ok={self.ok}"
        )


# ── Re-derivation (the real replay path) ─────────────────────────────


async def replay_decision_identities(
    *,
    strategy: Any,
    slug: str,
    start: datetime,
    end: datetime,
    sample_interval_seconds: int = 1800,
    max_ticks: int = 96,
    token_ids: Optional[list[str]] = None,
    warmup_seconds: int = 0,
) -> list[DecisionIdentity]:
    """Re-derive the strategy's decisions for the window by replaying
    recorded inputs through it, via the same discovery replay the
    backtester uses.  Returns a sorted list of decision identities.

    Runs under PersistentState replay isolation so the strategy can't
    read or write the live ``strategy_persistent_state`` table; combined
    with ``warmup_seconds`` pre-roll, durable + rolling state is rebuilt
    from the recorded stream rather than the live DB."""
    from services.strategy_backtester import _replay_discover_opportunities
    from services.strategy_helpers.persistent_state import (
        enter_replay_isolation,
        exit_replay_isolation,
    )

    _iso = enter_replay_isolation()
    try:
        opps = await _replay_discover_opportunities(
            strategy=strategy,
            slug=slug,
            start_dt=start,
            end_dt=end,
            sample_interval_seconds=sample_interval_seconds,
            max_ticks=max_ticks,
            candidate_token_ids=token_ids,
            warmup_seconds=warmup_seconds,
        )
    finally:
        exit_replay_isolation(_iso)
    identities: list[DecisionIdentity] = []
    for opp in opps or []:
        pdata = getattr(opp, "positions_data", None) or {}
        opp_edge = pdata.get("expected_roi")
        for pos in pdata.get("positions_to_take") or []:
            if not isinstance(pos, dict):
                continue
            identities.append(
                DecisionIdentity.from_position(
                    strategy_type=getattr(opp, "strategy_type", slug),
                    position=pos,
                    opp_edge=opp_edge,
                )
            )
    return sorted(identities, key=_identity_sort_key)


async def recorded_decision_identities(
    *,
    start: datetime,
    end: datetime,
    strategy_type: Optional[str] = None,
) -> list[DecisionIdentity]:
    """Load the decisions the bot actually emitted live in the window
    from the ``strategy.decision`` topic."""
    import services.recorded_event_bus.storage  # noqa: F401  attach storage
    from services.recorded_event_bus.bus import bus, ReplayWindow
    from services.recorded_event_bus.decision_recorder import STRATEGY_DECISION_TOPIC

    def _us(dt: datetime) -> int:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1_000_000)

    win = ReplayWindow(
        start_us=_us(start),
        end_us=_us(end),
        topics=(STRATEGY_DECISION_TOPIC,),
    )
    out: list[DecisionIdentity] = []
    async for ev in bus.replay(win):
        payload = dict(ev.payload)
        if strategy_type and str(payload.get("strategy_type")) != str(strategy_type):
            continue
        out.append(DecisionIdentity.from_recorded_payload(payload))
    return sorted(out, key=_identity_sort_key)


def _identity_sort_key(d: DecisionIdentity) -> tuple:
    return (
        d.strategy_type or "",
        d.token_id or "",
        d.direction or "",
        d.entry_price if d.entry_price is not None else -1.0,
        d.edge_bucket if d.edge_bucket is not None else -1.0,
    )


# ── Diff + determinism ───────────────────────────────────────────────


def diff_identities(
    replayed: Iterable[DecisionIdentity],
    recorded: Iterable[DecisionIdentity],
) -> tuple[int, list[DecisionIdentity], list[DecisionIdentity]]:
    """Multiset diff.  Returns (matched, missing_from_replay,
    extra_in_replay).  Multiset so duplicate decisions are counted, not
    collapsed."""
    from collections import Counter

    rc = Counter(replayed)
    dc = Counter(recorded)
    matched = sum((rc & dc).values())
    missing = list((dc - rc).elements())  # recorded but not replayed
    extra = list((rc - dc).elements())    # replayed but not recorded
    return matched, sorted(missing, key=_identity_sort_key), sorted(extra, key=_identity_sort_key)


async def assert_replay_deterministic(
    *,
    strategy_factory,
    slug: str,
    start: datetime,
    end: datetime,
    **replay_kwargs: Any,
) -> tuple[bool, list[DecisionIdentity], list[DecisionIdentity]]:
    """Run the re-derivation twice (fresh strategy instance each time) and
    confirm identical output.  ``strategy_factory`` is a zero-arg callable
    returning a fresh strategy instance so per-run in-memory state can't
    leak between the two passes.

    Returns (is_deterministic, run_a, run_b)."""
    a = await replay_decision_identities(
        strategy=strategy_factory(), slug=slug, start=start, end=end, **replay_kwargs
    )
    b = await replay_decision_identities(
        strategy=strategy_factory(), slug=slug, start=start, end=end, **replay_kwargs
    )
    return (a == b), a, b


# ── Built-in strategy loader ─────────────────────────────────────────


def load_builtin_strategy_factory(slug: str):
    """Return a zero-arg factory that builds a fresh instance of the
    built-in strategy whose ``strategy_type`` matches ``slug``.

    Raises LookupError if no built-in strategy advertises that slug, or
    if it can't be instantiated with no arguments (pass your own factory
    in that case)."""
    import services.strategies  # noqa: F401  ensure subclasses are imported
    from services.strategies.base import BaseStrategy

    def _all_subclasses(cls) -> set:
        out = set()
        for sub in cls.__subclasses__():
            out.add(sub)
            out |= _all_subclasses(sub)
        return out

    match = None
    for cls in _all_subclasses(BaseStrategy):
        if str(getattr(cls, "strategy_type", "") or "") == slug:
            match = cls
            break
    if match is None:
        raise LookupError(f"no built-in strategy with strategy_type={slug!r}")

    def _factory():
        return match()

    # Fail fast if it can't be built no-arg, so callers learn now.
    try:
        _factory()
    except Exception as exc:  # noqa: BLE001
        raise LookupError(
            f"strategy {slug!r} ({match.__name__}) cannot be instantiated "
            f"with no arguments: {exc}. Pass a custom factory."
        ) from exc
    return _factory


# ── Top-level entry point ────────────────────────────────────────────


async def run_parity_report(
    *,
    slug: str,
    start: datetime,
    end: datetime,
    strategy_factory=None,
    check_determinism: bool = True,
    sample_interval_seconds: int = 1800,
    max_ticks: int = 96,
    token_ids: Optional[list[str]] = None,
    warmup_seconds: int = 0,
) -> ParityReport:
    """Full parity check for one strategy + window.  Builds the report,
    runs the determinism check, re-derives decisions, and diffs against
    recorded live decisions."""
    report = ParityReport(strategy_slug=slug, window_start=start, window_end=end)

    if strategy_factory is None:
        strategy_factory = load_builtin_strategy_factory(slug)

    replay_kwargs = dict(
        sample_interval_seconds=sample_interval_seconds,
        max_ticks=max_ticks,
        token_ids=token_ids,
        warmup_seconds=warmup_seconds,
    )

    if check_determinism:
        det, run_a, run_b = await assert_replay_deterministic(
            strategy_factory=strategy_factory, slug=slug, start=start, end=end, **replay_kwargs
        )
        report.deterministic = det
        replayed = run_a
        if not det:
            report.notes.append(
                f"NON-DETERMINISTIC: run A had {len(run_a)} decisions, "
                f"run B had {len(run_b)} — replay is not reproducible."
            )
    else:
        replayed = await replay_decision_identities(
            strategy=strategy_factory(), slug=slug, start=start, end=end, **replay_kwargs
        )

    recorded = await recorded_decision_identities(start=start, end=end, strategy_type=slug)

    report.replayed_count = len(replayed)
    report.recorded_count = len(recorded)
    matched, missing, extra = diff_identities(replayed, recorded)
    report.matched = matched
    report.missing_from_replay = missing
    report.extra_in_replay = extra
    if report.recorded_count == 0:
        report.notes.append(
            "no recorded live decisions in window — replay-vs-live diff is "
            "vacuous until the decision tee has run live over this window."
        )
    return report
