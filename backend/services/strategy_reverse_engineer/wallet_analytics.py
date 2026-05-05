"""Deterministic wallet analytics engine.

Pure-Python statistical decomposition of a wallet's trade history into
the tables the report writer feeds to the LLM section-by-section.

Design philosophy (cribbed from polyresearchrobotics' reports):
  * **Math here, prose elsewhere.**  Every number a report cites comes
    from this module, never from the LLM.  The LLM's only job is to
    write narrative *grounded in* these numbers.
  * **Total transparency.**  Every aggregate is computable from the
    raw inputs alone — no hidden state, no provider-specific shortcuts,
    no implicit weighting.  A reader who wants to verify a claim can
    re-run the math here and get the same number.
  * **Two-leg P/L decomposition.**  The crucial mechanic for any
    market-making-shape strategy: split realized P/L into
    ``paired_shares × (1 − paired_cost)`` (the spread / market-making
    leg) and ``excess_shares × outcome`` (the directional leg).  These
    legs frequently point in *opposite* directions and conflating them
    obscures the actual edge source.
  * **No LLM, no I/O.**  The engine is a pure function of
    (normalized_trades, market_resolutions).  Trivially testable.

The module exports a single ``WalletAnalytics`` dataclass — the
report writer pulls section-specific subsets from it.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from services.strategy_reverse_engineer.market_resolution import MarketResolution

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclasses — these flow straight into the report writer's
# Jinja templates; keep field names and units stable.
# ---------------------------------------------------------------------------


@dataclass
class HeadlineMetrics:
    total_trades: int = 0
    buy_trades: int = 0
    sell_trades: int = 0
    markets_touched: int = 0
    total_usdc_deployed: float = 0.0
    active_days: int = 0
    calendar_days: int = 0
    inactive_days: int = 0
    win_rate: Optional[float] = None        # 0-1, of resolved trades
    realized_pl_usdc: Optional[float] = None
    roi_on_deployed: Optional[float] = None  # 0-1
    avg_trades_per_active_day: float = 0.0
    avg_fills_per_market: float = 0.0
    window_start: Optional[str] = None       # ISO
    window_end: Optional[str] = None
    resolved_trade_count: int = 0
    unresolved_trade_count: int = 0


@dataclass
class TradeSizeStats:
    """Distribution of per-fill USDC notional."""
    count: int = 0
    min_usdc: Optional[float] = None
    median_usdc: Optional[float] = None
    mean_usdc: Optional[float] = None
    p90_usdc: Optional[float] = None
    p95_usdc: Optional[float] = None
    p99_usdc: Optional[float] = None
    max_usdc: Optional[float] = None
    top5pct_share_of_capital: Optional[float] = None
    top1pct_share_of_capital: Optional[float] = None


@dataclass
class CadenceStats:
    """Inter-fill timing within the same (market, outcome) tuple."""
    sample_size: int = 0
    median_seconds: Optional[float] = None
    mean_seconds: Optional[float] = None
    pct_under_1s: Optional[float] = None
    pct_under_10s: Optional[float] = None
    pct_under_60s: Optional[float] = None


@dataclass
class DailyRow:
    date: str
    trades: int
    usdc: float
    pl_usdc: Optional[float]
    cum_pl_usdc: Optional[float]


@dataclass
class HourRow:
    hour_utc: int
    trades: int
    wins: int
    win_rate: Optional[float]
    pl_usdc: Optional[float]


@dataclass
class DayOfWeekRow:
    dow: int                  # 1=Mon .. 7=Sun
    label: str                # 'Monday' etc.
    trades: int
    pl_usdc: Optional[float]


@dataclass
class PriceBucketRow:
    band_label: str           # '$0.00–$0.10'
    band_low: float
    band_high: float
    trades: int
    wins: int
    win_rate: Optional[float]
    usdc_deployed: float
    pl_usdc: Optional[float]
    roi: Optional[float]


@dataclass
class DominanceRow:
    """Skew bucket: markets where dominant_side / underdog_side ratio
    falls into ``[low, high)``.  Captures the polyresearchrobotics
    'skew vs. resolution' table verbatim.
    """
    band_label: str
    ratio_low: float
    ratio_high: float           # inf for the open-ended top bucket
    markets: int
    dom_side_wins: int
    dom_side_win_rate: Optional[float]
    avg_paired_cost: Optional[float]
    total_pl_usdc: Optional[float]
    avg_pl_per_market: Optional[float]


@dataclass
class FilterRow:
    """A counterfactual filter: 'what would P/L look like if we'd only
    taken trades matching this rule?'.  ROI lift vs. the unfiltered
    baseline tells the operator which filter is highest-value.
    """
    name: str
    description: str
    trades: int
    usdc_deployed: float
    pl_usdc: Optional[float]
    win_rate: Optional[float]
    roi: Optional[float]
    roi_lift_vs_baseline: Optional[float]   # in absolute ROI percentage points


@dataclass
class TopMarketRow:
    market_label: str
    market_id: Optional[str]
    event_slug: Optional[str]
    trades: int
    usdc_deployed: float
    pl_usdc: Optional[float]


@dataclass
class TwoLegDecomposition:
    """Spread leg vs directional leg P/L."""
    paired_shares: float = 0.0
    median_paired_cost: Optional[float] = None
    mean_paired_cost: Optional[float] = None
    spread_leg_pl_usdc: float = 0.0          # negative when paired_cost > 1
    excess_shares: float = 0.0               # sum across markets of |yes - no|
    directional_leg_pl_usdc: float = 0.0
    one_sided_market_pl_usdc: float = 0.0    # markets with only one outcome bought
    realized_pl_usdc: float = 0.0            # sum of all three
    hedge_tax_usdc: float = 0.0              # USDC spent on losing-side hedges in dom-won markets
    both_sides_market_count: int = 0
    one_sided_market_count: int = 0
    both_sides_participation_rate: Optional[float] = None


@dataclass
class StrategyArchetypeMatch:
    archetype: str
    match_strength: str                      # 'None' | 'Costume' | 'Weak' | 'Strong'
    evidence: str


@dataclass
class TimingStats:
    """Within-window timing — only meaningful when markets have a known
    start_time and end_time (e.g. crypto Up/Down 5m).  Buckets by
    seconds-remaining when the trade was placed.
    """
    bucket_label: str
    seconds_low: int
    seconds_high: int
    trades: int
    usdc_deployed: float
    pl_usdc: Optional[float]
    win_rate: Optional[float]
    roi: Optional[float]


@dataclass
class RollingWindowStats:
    window_days: int
    windows_total: int
    windows_positive: int
    min_pl_usdc: Optional[float]
    max_pl_usdc: Optional[float]


@dataclass
class WalletAnalytics:
    """The full statistical report — every table the report writer needs."""

    address: str
    headline: HeadlineMetrics = field(default_factory=HeadlineMetrics)
    trade_size: TradeSizeStats = field(default_factory=TradeSizeStats)
    cadence: CadenceStats = field(default_factory=CadenceStats)
    side_split: dict[str, int] = field(default_factory=dict)
    outcome_split: dict[str, int] = field(default_factory=dict)
    daily_rows: list[DailyRow] = field(default_factory=list)
    hourly_rows: list[HourRow] = field(default_factory=list)
    dow_rows: list[DayOfWeekRow] = field(default_factory=list)
    price_buckets: list[PriceBucketRow] = field(default_factory=list)
    dominance_buckets: list[DominanceRow] = field(default_factory=list)
    paired_cost_bands: list[PriceBucketRow] = field(default_factory=list)
    filter_ledger: list[FilterRow] = field(default_factory=list)
    top_by_volume: list[TopMarketRow] = field(default_factory=list)
    top_winning: list[TopMarketRow] = field(default_factory=list)
    top_losing: list[TopMarketRow] = field(default_factory=list)
    two_leg: TwoLegDecomposition = field(default_factory=TwoLegDecomposition)
    archetypes: list[StrategyArchetypeMatch] = field(default_factory=list)
    within_window_timing: list[TimingStats] = field(default_factory=list)
    rolling_7d: Optional[RollingWindowStats] = None
    rolling_15d: Optional[RollingWindowStats] = None
    spotlight_market: Optional[dict[str, Any]] = None   # populated by render_spotlight_market()
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Render to a JSON-safe dict for storage / API surfaces."""
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def analyze(
    *,
    address: str,
    trades: list[dict[str, Any]],
    resolutions: dict[str, MarketResolution],
) -> WalletAnalytics:
    """Compute all tables for a wallet.

    Args:
        address:       wallet address (lowercase 0x...)
        trades:        normalized wallet trades — each dict needs at
                       minimum: timestamp (datetime aware), price (float),
                       size (float, shares), notional_usd, side ('BUY'/'SELL'),
                       outcome ('UP'/'DOWN'/'YES'/'NO'), market_id, event_slug.
        resolutions:   {slug_or_market_id: MarketResolution}.  Both keys
                       are checked when looking up resolution per trade.

    Returns:
        WalletAnalytics — pass to ``report_writer`` to draft narrative
        sections, or to the PDF template directly to render the
        deterministic tables.
    """
    out = WalletAnalytics(address=address)
    if not trades:
        return out

    # Augment each trade in-place with a resolved=True/False flag and
    # outcome_won True/False/None.  Done once here so downstream
    # bucketing functions can use a stable shape.
    enriched = _enrich_with_resolution(trades, resolutions)

    out.headline = _compute_headline(enriched)
    out.trade_size = _compute_trade_size(enriched)
    out.cadence = _compute_cadence(enriched)
    out.side_split = _count_field(enriched, "side")
    out.outcome_split = _count_field(enriched, "outcome")
    out.daily_rows = _compute_daily_rows(enriched)
    out.hourly_rows = _compute_hourly_rows(enriched)
    out.dow_rows = _compute_dow_rows(enriched)
    out.price_buckets = _compute_price_buckets(enriched)
    out.two_leg, by_market = _compute_two_leg_decomposition(enriched)
    out.dominance_buckets = _compute_dominance_buckets(by_market)
    out.paired_cost_bands = _compute_paired_cost_bands(by_market)
    out.top_by_volume = _compute_top_by_volume(by_market, n=10)
    out.top_winning = _compute_top_winning(by_market, n=10)
    out.top_losing = _compute_top_losing(by_market, n=10)
    out.archetypes = _identify_archetypes(out)
    out.within_window_timing = _compute_within_window_timing(enriched)
    out.rolling_7d = _compute_rolling_window(out.daily_rows, window_days=7)
    out.rolling_15d = _compute_rolling_window(out.daily_rows, window_days=15)
    out.filter_ledger = _compute_filter_ledger(enriched, by_market)

    return out


def render_spotlight_market(
    *,
    analytics: WalletAnalytics,
    trades: list[dict[str, Any]],
    market_id: Optional[str] = None,
    event_slug: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Build the per-market trade-by-trade walkthrough table.

    Caller picks a market explicitly OR we pick the highest-trade-count
    market with a known winner (most illustrative).  Returns a dict
    suitable for the PDF template's spotlight section, or None when
    no eligible market exists.
    """
    if not trades:
        return None

    # Choose the spotlight market.
    if market_id is None and event_slug is None:
        # Pick the resolved market with the most trades.
        per_market = defaultdict(list)
        for t in trades:
            key = t.get("market_id") or t.get("event_slug") or ""
            if key:
                per_market[key].append(t)
        # Prefer ones with a non-None winner
        best_key, best_trades = None, []
        best_score = -1
        for k, ts in per_market.items():
            sample = ts[0]
            res = sample.get("_resolution")
            if res and res.winner_outcome:
                score = len(ts)
                if score > best_score:
                    best_score = score
                    best_key, best_trades = k, ts
        if not best_key:
            return None
        chosen = best_trades
    else:
        chosen = [
            t for t in trades
            if (market_id and str(t.get("market_id")) == market_id)
            or (event_slug and str(t.get("event_slug")) == event_slug)
        ]
        if not chosen:
            return None

    chosen = sorted(chosen, key=lambda t: t["timestamp"])
    sample = chosen[0]
    resolution: Optional[MarketResolution] = sample.get("_resolution")
    market_title = sample.get("market_title") or sample.get("event_slug") or "?"

    # Compute running P/L by simulating fill-by-fill cash flow.
    rows: list[dict[str, Any]] = []
    yes_shares = no_shares = 0.0
    yes_usdc = no_usdc = 0.0
    cum_pl = 0.0
    for t in chosen:
        side = (t.get("outcome") or "").upper()
        size = float(t.get("size") or 0)
        usdc = float(t.get("notional_usd") or 0)
        if side in ("UP", "YES"):
            yes_shares += size
            yes_usdc += usdc
        elif side in ("DOWN", "NO"):
            no_shares += size
            no_usdc += usdc
        # Per-fill realized P/L = - usdc cost; share payoff happens at expiry.
        # For a running view we attribute the fill's settlement value at the
        # *known* winner — gives the operator an "if this had been the
        # final state, what would PnL be" running estimate.
        winner = (resolution.winner_outcome.upper() if resolution and resolution.winner_outcome else None)
        # If this fill is on the eventual winning side, expected payoff = size * 1
        # If on losing side, expected payoff = 0
        fill_payoff = 0.0
        if winner is not None:
            fill_won = (side in ("UP", "YES") and winner in ("UP", "YES")) or (
                side in ("DOWN", "NO") and winner in ("DOWN", "NO")
            )
            fill_payoff = float(size) if fill_won else 0.0
        fill_pl = fill_payoff - usdc
        cum_pl += fill_pl
        sec_to_close: Optional[int] = None
        if resolution and resolution.event_slug:
            try:
                end_ts = _slug_to_window_end(resolution.event_slug)
                if end_ts:
                    sec_to_close = max(0, int((end_ts - t["timestamp"]).total_seconds()))
            except Exception:
                pass
        rows.append({
            "timestamp": t["timestamp"].isoformat() if hasattr(t["timestamp"], "isoformat") else str(t["timestamp"]),
            "sec_to_close": sec_to_close,
            "side_label": f"{(t.get('side') or 'BUY').upper()} {side}",
            "price": float(t.get("price") or 0),
            "shares": float(size),
            "usdc": -float(usdc),
            "fill_pl": float(fill_pl),
            "running_pl": float(cum_pl),
        })

    return {
        "market_title": market_title,
        "market_id": sample.get("market_id"),
        "event_slug": sample.get("event_slug"),
        "winner": (resolution.winner_outcome if resolution else None),
        "yes_shares": yes_shares,
        "no_shares": no_shares,
        "yes_usdc": yes_usdc,
        "no_usdc": no_usdc,
        "dominance_ratio": (
            max(yes_shares, no_shares) / max(min(yes_shares, no_shares), 1e-9)
            if min(yes_shares, no_shares) > 0
            else None
        ),
        "final_pl_usdc": cum_pl,
        "fill_count": len(chosen),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Per-trade enrichment + lookups
# ---------------------------------------------------------------------------


def _enrich_with_resolution(
    trades: list[dict[str, Any]],
    resolutions: dict[str, MarketResolution],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for t in trades:
        copy = dict(t)
        slug = t.get("event_slug")
        cid = t.get("market_id")
        res: Optional[MarketResolution] = None
        if slug and slug in resolutions:
            res = resolutions[slug]
        elif cid and cid in resolutions:
            res = resolutions[cid]
        copy["_resolution"] = res
        if res is not None:
            outcome_won = res.did_outcome_win(t.get("outcome") or "")
            copy["_outcome_won"] = outcome_won
            copy["_resolved"] = True
        else:
            copy["_outcome_won"] = None
            copy["_resolved"] = False
        enriched.append(copy)
    return enriched


# ---------------------------------------------------------------------------
# Headline + trade-size + cadence
# ---------------------------------------------------------------------------


def _compute_headline(trades: list[dict[str, Any]]) -> HeadlineMetrics:
    h = HeadlineMetrics()
    h.total_trades = len(trades)
    h.buy_trades = sum(1 for t in trades if (t.get("side") or "").upper() == "BUY")
    h.sell_trades = sum(1 for t in trades if (t.get("side") or "").upper() == "SELL")
    h.markets_touched = len({t.get("market_id") or t.get("event_slug") for t in trades if t.get("market_id") or t.get("event_slug")})
    h.total_usdc_deployed = sum(float(t.get("notional_usd") or 0) for t in trades)

    # Day boundaries
    days = sorted({t["timestamp"].date() for t in trades})
    if days:
        h.window_start = trades[0]["timestamp"].isoformat() if trades else None
        h.window_end = trades[-1]["timestamp"].isoformat() if trades else None
        # Re-derive from min/max because trade order isn't guaranteed.
        all_ts = [t["timestamp"] for t in trades]
        h.window_start = min(all_ts).isoformat()
        h.window_end = max(all_ts).isoformat()
        h.calendar_days = (max(days) - min(days)).days + 1
        h.active_days = len(days)
        h.inactive_days = h.calendar_days - h.active_days

    # Resolved subset stats
    resolved = [t for t in trades if t.get("_resolved")]
    h.resolved_trade_count = len(resolved)
    h.unresolved_trade_count = h.total_trades - h.resolved_trade_count
    if resolved:
        wins = sum(1 for t in resolved if t.get("_outcome_won"))
        h.win_rate = wins / len(resolved)
        # Realized P/L per trade = (size if won else 0) - usdc
        pl = 0.0
        for t in resolved:
            usdc = float(t.get("notional_usd") or 0)
            size = float(t.get("size") or 0)
            payoff = size if t.get("_outcome_won") else 0.0
            pl += payoff - usdc
        h.realized_pl_usdc = pl
        if h.total_usdc_deployed > 0:
            h.roi_on_deployed = pl / h.total_usdc_deployed

    if h.active_days > 0:
        h.avg_trades_per_active_day = h.total_trades / h.active_days
    if h.markets_touched > 0:
        h.avg_fills_per_market = h.total_trades / h.markets_touched

    return h


def _compute_trade_size(trades: list[dict[str, Any]]) -> TradeSizeStats:
    notional = sorted(float(t.get("notional_usd") or 0) for t in trades)
    notional = [n for n in notional if n > 0]
    if not notional:
        return TradeSizeStats()
    total = sum(notional)
    return TradeSizeStats(
        count=len(notional),
        min_usdc=notional[0],
        median_usdc=statistics.median(notional),
        mean_usdc=statistics.fmean(notional),
        p90_usdc=_pct(notional, 0.90),
        p95_usdc=_pct(notional, 0.95),
        p99_usdc=_pct(notional, 0.99),
        max_usdc=notional[-1],
        top5pct_share_of_capital=(
            sum(notional[-int(len(notional) * 0.05):]) / total
            if total > 0 and len(notional) >= 20 else None
        ),
        top1pct_share_of_capital=(
            sum(notional[-int(len(notional) * 0.01):]) / total
            if total > 0 and len(notional) >= 100 else None
        ),
    )


def _compute_cadence(trades: list[dict[str, Any]]) -> CadenceStats:
    """Inter-fill gaps within the same (market, outcome)."""
    by_key: dict[tuple, list[datetime]] = defaultdict(list)
    for t in trades:
        key = (t.get("market_id") or t.get("event_slug"), t.get("outcome"))
        if key[0]:
            by_key[key].append(t["timestamp"])

    gaps_seconds: list[float] = []
    for ts_list in by_key.values():
        ts_list.sort()
        for i in range(1, len(ts_list)):
            gap = (ts_list[i] - ts_list[i - 1]).total_seconds()
            if gap > 0:
                gaps_seconds.append(gap)

    if not gaps_seconds:
        return CadenceStats()
    n = len(gaps_seconds)
    return CadenceStats(
        sample_size=n,
        median_seconds=statistics.median(gaps_seconds),
        mean_seconds=statistics.fmean(gaps_seconds),
        pct_under_1s=sum(1 for g in gaps_seconds if g < 1.0) / n,
        pct_under_10s=sum(1 for g in gaps_seconds if g < 10.0) / n,
        pct_under_60s=sum(1 for g in gaps_seconds if g < 60.0) / n,
    )


def _count_field(trades: list[dict[str, Any]], field_name: str) -> dict[str, int]:
    c: Counter[str] = Counter()
    for t in trades:
        val = t.get(field_name)
        if val:
            c[str(val).upper()] += 1
    return dict(c)


# ---------------------------------------------------------------------------
# Daily / Hourly / DOW
# ---------------------------------------------------------------------------


def _compute_daily_rows(trades: list[dict[str, Any]]) -> list[DailyRow]:
    by_day_trades: dict[Any, int] = defaultdict(int)
    by_day_usdc: dict[Any, float] = defaultdict(float)
    by_day_pl: dict[Any, float] = defaultdict(float)
    by_day_resolved_count: dict[Any, int] = defaultdict(int)
    days = set()
    for t in trades:
        d = t["timestamp"].date()
        days.add(d)
        by_day_trades[d] += 1
        by_day_usdc[d] += float(t.get("notional_usd") or 0)
        if t.get("_resolved"):
            by_day_resolved_count[d] += 1
            payoff = float(t.get("size") or 0) if t.get("_outcome_won") else 0.0
            by_day_pl[d] += payoff - float(t.get("notional_usd") or 0)

    if not days:
        return []
    # Fill calendar gaps so the rolling-window analyzer sees inactive days.
    full_days = sorted(days)
    start, end = full_days[0], full_days[-1]
    cur = start
    out: list[DailyRow] = []
    cum = 0.0 if any(by_day_resolved_count.values()) else None
    while cur <= end:
        n_resolved = by_day_resolved_count.get(cur, 0)
        pl = by_day_pl.get(cur, 0.0) if n_resolved > 0 else None
        if pl is not None:
            cum = (cum or 0.0) + pl
        out.append(DailyRow(
            date=cur.isoformat(),
            trades=by_day_trades.get(cur, 0),
            usdc=by_day_usdc.get(cur, 0.0),
            pl_usdc=pl,
            cum_pl_usdc=cum,
        ))
        cur = cur + timedelta(days=1)
    return out


def _compute_hourly_rows(trades: list[dict[str, Any]]) -> list[HourRow]:
    by_hour_trades: dict[int, int] = defaultdict(int)
    by_hour_wins: dict[int, int] = defaultdict(int)
    by_hour_resolved: dict[int, int] = defaultdict(int)
    by_hour_pl: dict[int, float] = defaultdict(float)
    for t in trades:
        h = t["timestamp"].hour
        by_hour_trades[h] += 1
        if t.get("_resolved"):
            by_hour_resolved[h] += 1
            if t.get("_outcome_won"):
                by_hour_wins[h] += 1
            payoff = float(t.get("size") or 0) if t.get("_outcome_won") else 0.0
            by_hour_pl[h] += payoff - float(t.get("notional_usd") or 0)
    out: list[HourRow] = []
    for h in range(24):
        n_resolved = by_hour_resolved.get(h, 0)
        out.append(HourRow(
            hour_utc=h,
            trades=by_hour_trades.get(h, 0),
            wins=by_hour_wins.get(h, 0),
            win_rate=(by_hour_wins[h] / n_resolved) if n_resolved else None,
            pl_usdc=(by_hour_pl[h] if n_resolved else None),
        ))
    return out


def _compute_dow_rows(trades: list[dict[str, Any]]) -> list[DayOfWeekRow]:
    labels = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    by_dow_trades: dict[int, int] = defaultdict(int)
    by_dow_pl: dict[int, float] = defaultdict(float)
    by_dow_resolved: dict[int, int] = defaultdict(int)
    for t in trades:
        d = t["timestamp"].isoweekday()
        by_dow_trades[d] += 1
        if t.get("_resolved"):
            by_dow_resolved[d] += 1
            payoff = float(t.get("size") or 0) if t.get("_outcome_won") else 0.0
            by_dow_pl[d] += payoff - float(t.get("notional_usd") or 0)
    return [
        DayOfWeekRow(
            dow=d,
            label=labels[d - 1],
            trades=by_dow_trades.get(d, 0),
            pl_usdc=(by_dow_pl[d] if by_dow_resolved.get(d, 0) else None),
        )
        for d in range(1, 8)
    ]


# ---------------------------------------------------------------------------
# Price buckets (10-band entry-price histogram)
# ---------------------------------------------------------------------------


def _compute_price_buckets(trades: list[dict[str, Any]]) -> list[PriceBucketRow]:
    bands = [(i * 0.1, (i + 1) * 0.1) for i in range(10)]
    out: list[PriceBucketRow] = []
    for low, high in bands:
        bucket = [
            t for t in trades
            if t.get("price") is not None and low <= float(t["price"]) < high
        ]
        n = len(bucket)
        usdc = sum(float(t.get("notional_usd") or 0) for t in bucket)
        resolved = [t for t in bucket if t.get("_resolved")]
        wins = sum(1 for t in resolved if t.get("_outcome_won"))
        pl = (
            sum(
                (float(t.get("size") or 0) if t.get("_outcome_won") else 0.0)
                - float(t.get("notional_usd") or 0)
                for t in resolved
            )
            if resolved else None
        )
        out.append(PriceBucketRow(
            band_label=f"${low:.2f}–${high:.2f}",
            band_low=low,
            band_high=high,
            trades=n,
            wins=wins,
            win_rate=(wins / len(resolved)) if resolved else None,
            usdc_deployed=usdc,
            pl_usdc=pl,
            roi=(pl / usdc) if pl is not None and usdc > 0 else None,
        ))
    return out


# ---------------------------------------------------------------------------
# Two-leg P/L decomposition (the killer table)
# ---------------------------------------------------------------------------


@dataclass
class _MarketAggregate:
    """Internal per-market accumulator used by the decomposition + ranking."""
    market_id: Optional[str]
    event_slug: Optional[str]
    market_title: Optional[str]
    yes_shares: float = 0.0
    no_shares: float = 0.0
    yes_usdc: float = 0.0
    no_usdc: float = 0.0
    trade_count: int = 0
    resolution: Optional[MarketResolution] = None

    @property
    def paired_shares(self) -> float:
        return min(self.yes_shares, self.no_shares)

    @property
    def excess_shares(self) -> float:
        return abs(self.yes_shares - self.no_shares)

    @property
    def dominant_side(self) -> Optional[str]:
        if self.yes_shares == 0 and self.no_shares == 0:
            return None
        return "UP" if self.yes_shares >= self.no_shares else "DOWN"

    @property
    def underdog_side(self) -> Optional[str]:
        ds = self.dominant_side
        if ds is None:
            return None
        return "DOWN" if ds == "UP" else "UP"

    @property
    def dominance_ratio(self) -> Optional[float]:
        smaller = min(self.yes_shares, self.no_shares)
        if smaller <= 0:
            return None
        return max(self.yes_shares, self.no_shares) / smaller

    @property
    def is_both_sided(self) -> bool:
        return self.yes_shares > 0 and self.no_shares > 0

    @property
    def total_usdc(self) -> float:
        return self.yes_usdc + self.no_usdc

    @property
    def paired_cost_per_share(self) -> Optional[float]:
        """Average cost per paired share-pair (a guaranteed-$1 outcome).

        For markets where ``paired_shares > 0``, the paired portion of
        the position is the smaller-side share count.  The cost of
        owning ``paired_shares`` of UP and ``paired_shares`` of DOWN is
        the sum of the two side VWAPs *for those paired shares*.

        We approximate with the side VWAPs across all fills on each
        side (close to true-paired-cost when fills are roughly uniform
        within the pair window).  Returns None when either side is
        empty.
        """
        if self.paired_shares <= 0:
            return None
        yes_vwap = self.yes_usdc / self.yes_shares if self.yes_shares > 0 else 0.0
        no_vwap = self.no_usdc / self.no_shares if self.no_shares > 0 else 0.0
        return yes_vwap + no_vwap

    def realized_pl(self) -> Optional[float]:
        """Total realized P/L on this market.

        Each share pays $1 if its outcome won, else $0.  Total payoff
        minus total cost.  Returns None when resolution is unknown.
        """
        if not self.resolution or not self.resolution.winner_outcome:
            return None
        winner = self.resolution.winner_outcome.upper()
        payoff = 0.0
        if winner in ("UP", "YES"):
            payoff = self.yes_shares
        elif winner in ("DOWN", "NO"):
            payoff = self.no_shares
        return payoff - self.total_usdc

    def two_leg_split(self) -> tuple[Optional[float], Optional[float]]:
        """(spread_leg_pl, directional_leg_pl) — both signed USDC.

        Returns (None, None) when resolution is unknown or only one side
        was bought.  For one-sided markets the two-leg split is
        meaningless; the caller surfaces those P/Ls in the
        ``one_sided_market_pl_usdc`` bucket instead.
        """
        if not self.resolution or not self.resolution.winner_outcome:
            return None, None
        if not self.is_both_sided:
            return None, None
        paired = self.paired_shares
        cost = self.paired_cost_per_share or 0.0
        # Spread leg: paired_shares × (1 − paired_cost_per_share).  When
        # cost > 1 this is negative (the maker leg lost money).
        spread_pl = paired * (1.0 - cost)
        # Directional leg: excess shares only matter if they're on the
        # winning side — that's the "extra" exposure beyond the paired
        # portion that converts to a $1 payoff.  If on losing side, the
        # excess is sunk cost (already reflected in spread_pl via the
        # paired_cost > 1 component).
        winner = self.resolution.winner_outcome.upper()
        excess = self.excess_shares
        excess_side = self.dominant_side
        if excess_side == "UP" and winner in ("UP", "YES"):
            # Excess shares pay $1 each; cost of those excess shares is
            # already partially booked via the side's average.
            excess_cost = (excess / max(self.yes_shares, 1e-9)) * self.yes_usdc
            directional_pl = (excess * 1.0) - excess_cost
        elif excess_side == "DOWN" and winner in ("DOWN", "NO"):
            excess_cost = (excess / max(self.no_shares, 1e-9)) * self.no_usdc
            directional_pl = (excess * 1.0) - excess_cost
        else:
            # Excess on losing side: pay nothing, but the cost is
            # already in the side's VWAP cost basis we used above.
            if excess_side == "UP":
                excess_cost = (excess / max(self.yes_shares, 1e-9)) * self.yes_usdc
            else:
                excess_cost = (excess / max(self.no_shares, 1e-9)) * self.no_usdc
            directional_pl = -excess_cost
        return spread_pl, directional_pl


def _aggregate_by_market(
    trades: list[dict[str, Any]],
) -> dict[str, _MarketAggregate]:
    out: dict[str, _MarketAggregate] = {}
    for t in trades:
        key = t.get("market_id") or t.get("event_slug") or "?"
        agg = out.get(key)
        if agg is None:
            agg = _MarketAggregate(
                market_id=t.get("market_id"),
                event_slug=t.get("event_slug"),
                market_title=t.get("market_title"),
                resolution=t.get("_resolution"),
            )
            out[key] = agg
        outcome = (t.get("outcome") or "").upper()
        size = float(t.get("size") or 0)
        usdc = float(t.get("notional_usd") or 0)
        if outcome in ("UP", "YES"):
            agg.yes_shares += size
            agg.yes_usdc += usdc
        elif outcome in ("DOWN", "NO"):
            agg.no_shares += size
            agg.no_usdc += usdc
        agg.trade_count += 1
    return out


def _compute_two_leg_decomposition(
    trades: list[dict[str, Any]],
) -> tuple[TwoLegDecomposition, dict[str, _MarketAggregate]]:
    by_market = _aggregate_by_market(trades)
    spread_total = 0.0
    directional_total = 0.0
    one_sided_total = 0.0
    paired_share_total = 0.0
    excess_share_total = 0.0
    paired_costs: list[float] = []
    hedge_tax = 0.0
    both_sides_count = 0
    one_sided_count = 0

    for agg in by_market.values():
        if agg.is_both_sided:
            both_sides_count += 1
            paired_share_total += agg.paired_shares
            excess_share_total += agg.excess_shares
            cost = agg.paired_cost_per_share
            if cost is not None:
                paired_costs.append(cost)
            spread_pl, dir_pl = agg.two_leg_split()
            if spread_pl is not None:
                spread_total += spread_pl
            if dir_pl is not None:
                directional_total += dir_pl
            # Hedge tax = USDC spent on the underdog side in markets
            # where the dominant side won.
            if agg.resolution and agg.resolution.winner_outcome:
                winner = agg.resolution.winner_outcome.upper()
                ds = agg.dominant_side
                if (
                    (ds == "UP" and winner in ("UP", "YES"))
                    or (ds == "DOWN" and winner in ("DOWN", "NO"))
                ):
                    underdog_usdc = agg.no_usdc if ds == "UP" else agg.yes_usdc
                    hedge_tax += underdog_usdc
        else:
            one_sided_count += 1
            pl = agg.realized_pl()
            if pl is not None:
                one_sided_total += pl

    return (
        TwoLegDecomposition(
            paired_shares=paired_share_total,
            median_paired_cost=statistics.median(paired_costs) if paired_costs else None,
            mean_paired_cost=statistics.fmean(paired_costs) if paired_costs else None,
            spread_leg_pl_usdc=spread_total,
            excess_shares=excess_share_total,
            directional_leg_pl_usdc=directional_total,
            one_sided_market_pl_usdc=one_sided_total,
            realized_pl_usdc=spread_total + directional_total + one_sided_total,
            hedge_tax_usdc=hedge_tax,
            both_sides_market_count=both_sides_count,
            one_sided_market_count=one_sided_count,
            both_sides_participation_rate=(
                both_sides_count / (both_sides_count + one_sided_count)
                if (both_sides_count + one_sided_count) > 0
                else None
            ),
        ),
        by_market,
    )


# ---------------------------------------------------------------------------
# Dominance (skew) buckets — the headline alpha-source table
# ---------------------------------------------------------------------------


def _compute_dominance_buckets(
    by_market: dict[str, _MarketAggregate],
) -> list[DominanceRow]:
    bands = [
        ("1.0–1.5×", 1.0, 1.5),
        ("1.5–2.0×", 1.5, 2.0),
        ("2.0–3.0×", 2.0, 3.0),
        ("3.0–5.0×", 3.0, 5.0),
        ("≥ 5.0×", 5.0, math.inf),
    ]
    out: list[DominanceRow] = []
    for label, low, high in bands:
        in_band = [
            agg for agg in by_market.values()
            if agg.dominance_ratio is not None
            and low <= agg.dominance_ratio < high
            and agg.resolution is not None
            and agg.resolution.winner_outcome is not None
        ]
        n = len(in_band)
        if n == 0:
            out.append(DominanceRow(
                band_label=label, ratio_low=low, ratio_high=high,
                markets=0, dom_side_wins=0,
                dom_side_win_rate=None, avg_paired_cost=None,
                total_pl_usdc=None, avg_pl_per_market=None,
            ))
            continue
        wins = 0
        pls: list[float] = []
        costs: list[float] = []
        for agg in in_band:
            ds = agg.dominant_side
            winner = agg.resolution.winner_outcome.upper()
            if (ds == "UP" and winner in ("UP", "YES")) or (ds == "DOWN" and winner in ("DOWN", "NO")):
                wins += 1
            pl = agg.realized_pl()
            if pl is not None:
                pls.append(pl)
            c = agg.paired_cost_per_share
            if c is not None:
                costs.append(c)
        out.append(DominanceRow(
            band_label=label, ratio_low=low, ratio_high=high,
            markets=n, dom_side_wins=wins,
            dom_side_win_rate=(wins / n if n else None),
            avg_paired_cost=(statistics.fmean(costs) if costs else None),
            total_pl_usdc=(sum(pls) if pls else None),
            avg_pl_per_market=(statistics.fmean(pls) if pls else None),
        ))
    return out


def _compute_paired_cost_bands(
    by_market: dict[str, _MarketAggregate],
) -> list[PriceBucketRow]:
    """Bucket markets by paired_cost_per_share (efficiency of MM leg)."""
    bands = [
        ("< 0.97", 0.0, 0.97),
        ("0.97–1.00", 0.97, 1.00),
        ("1.00–1.02", 1.00, 1.02),
        ("1.02–1.05", 1.02, 1.05),
        ("1.05–1.10", 1.05, 1.10),
        ("≥ 1.10", 1.10, math.inf),
    ]
    out: list[PriceBucketRow] = []
    for label, low, high in bands:
        in_band = [
            agg for agg in by_market.values()
            if agg.is_both_sided
            and (agg.paired_cost_per_share or 0) >= low
            and (agg.paired_cost_per_share or 0) < high
            and agg.resolution and agg.resolution.winner_outcome
        ]
        n = len(in_band)
        usdc = sum(agg.total_usdc for agg in in_band)
        pls = [agg.realized_pl() for agg in in_band if agg.realized_pl() is not None]
        wins = sum(
            1 for agg in in_band
            if (
                (agg.dominant_side == "UP" and agg.resolution.winner_outcome.upper() in ("UP", "YES"))
                or (agg.dominant_side == "DOWN" and agg.resolution.winner_outcome.upper() in ("DOWN", "NO"))
            )
        )
        pl_total = sum(pls) if pls else None
        out.append(PriceBucketRow(
            band_label=label,
            band_low=low,
            band_high=high if high != math.inf else 9.99,
            trades=sum(agg.trade_count for agg in in_band),
            wins=wins,
            win_rate=(wins / n) if n else None,
            usdc_deployed=usdc,
            pl_usdc=pl_total,
            roi=(pl_total / usdc) if pl_total is not None and usdc > 0 else None,
        ))
    return out


# ---------------------------------------------------------------------------
# Rankings
# ---------------------------------------------------------------------------


def _compute_top_by_volume(
    by_market: dict[str, _MarketAggregate], n: int,
) -> list[TopMarketRow]:
    rows = sorted(by_market.values(), key=lambda a: a.trade_count, reverse=True)[:n]
    return [_top_row(a) for a in rows]


def _compute_top_winning(
    by_market: dict[str, _MarketAggregate], n: int,
) -> list[TopMarketRow]:
    rows: list[tuple[_MarketAggregate, float]] = []
    for a in by_market.values():
        pl = a.realized_pl()
        if pl is not None:
            rows.append((a, pl))
    rows.sort(key=lambda r: r[1], reverse=True)
    return [_top_row(a, pl) for a, pl in rows[:n]]


def _compute_top_losing(
    by_market: dict[str, _MarketAggregate], n: int,
) -> list[TopMarketRow]:
    rows: list[tuple[_MarketAggregate, float]] = []
    for a in by_market.values():
        pl = a.realized_pl()
        if pl is not None:
            rows.append((a, pl))
    rows.sort(key=lambda r: r[1])
    return [_top_row(a, pl) for a, pl in rows[:n]]


def _top_row(agg: _MarketAggregate, pl: Optional[float] = None) -> TopMarketRow:
    return TopMarketRow(
        market_label=agg.market_title or agg.event_slug or agg.market_id or "?",
        market_id=agg.market_id,
        event_slug=agg.event_slug,
        trades=agg.trade_count,
        usdc_deployed=agg.total_usdc,
        pl_usdc=(pl if pl is not None else agg.realized_pl()),
    )


# ---------------------------------------------------------------------------
# Strategy archetype identification
# ---------------------------------------------------------------------------


def _identify_archetypes(out: WalletAnalytics) -> list[StrategyArchetypeMatch]:
    matches: list[StrategyArchetypeMatch] = []

    h, t = out.headline, out.two_leg

    # A. Both-sides spread capture / market-making
    bs_rate = t.both_sides_participation_rate or 0.0
    if bs_rate >= 0.85:
        if (t.median_paired_cost or 0) > 1.0:
            label = "Costume"
            evidence = (
                f"{bs_rate*100:.1f}% both-sides participation but spread leg loses "
                f"${-t.spread_leg_pl_usdc:,.0f} (paired cost {t.median_paired_cost:.4f} > 1.00)"
            )
        else:
            label = "Strong"
            evidence = (
                f"{bs_rate*100:.1f}% both-sides participation, paired cost "
                f"{t.median_paired_cost:.4f}; spread leg P/L ${t.spread_leg_pl_usdc:,.0f}"
            )
        matches.append(StrategyArchetypeMatch(
            archetype="A. Both-Sides Spread Capture / Market Making",
            match_strength=label,
            evidence=evidence,
        ))

    # B. Directional betting (reactive)
    if out.dominance_buckets:
        top = out.dominance_buckets[-1]
        if top.markets > 0 and (top.dom_side_win_rate or 0) >= 0.95:
            matches.append(StrategyArchetypeMatch(
                archetype="B. Directional Betting (reactive)",
                match_strength="Strong",
                evidence=(
                    f"Dominance {top.band_label}: {(top.dom_side_win_rate*100):.1f}% "
                    f"dom-side win rate over {top.markets} markets"
                ),
            ))

    # C. Stale price / latency arbitrage — look at within-window timing.
    if out.within_window_timing:
        last = out.within_window_timing[-1]   # final-30s bucket
        baseline_roi = (h.roi_on_deployed or 0) * 100
        if last.roi is not None and last.roi * 100 > baseline_roi + 1.0:
            matches.append(StrategyArchetypeMatch(
                archetype="C. Stale Price / Latency Arbitrage",
                match_strength="Strong" if last.roi * 100 > baseline_roi + 2.0 else "Weak",
                evidence=(
                    f"Final-30s window has +{last.roi*100:.2f}% ROI vs. "
                    f"+{baseline_roi:.2f}% baseline"
                ),
            ))

    # D + E + F: deliberately not auto-firing — those need cross-wallet data
    # the analytics module doesn't have.  Surface as 'None' so the report
    # writer renders the archetype matrix with NULLs filled in.

    if not matches:
        matches.append(StrategyArchetypeMatch(
            archetype="(unclassified)",
            match_strength="None",
            evidence="No archetype heuristic crossed its threshold.",
        ))
    return matches


# ---------------------------------------------------------------------------
# Within-window timing (only useful for crypto Up/Down markets)
# ---------------------------------------------------------------------------


def _compute_within_window_timing(
    trades: list[dict[str, Any]],
) -> list[TimingStats]:
    """Bucket trades by seconds-remaining at the time of the trade.

    Only meaningful for time-bounded markets (5m crypto Up/Down).  We
    derive end_time from the slug for slugs of the form
    ``btc-updown-{horizon}-{epoch}`` (where ``epoch`` is the start
    timestamp); for other slugs we skip the trade.
    """
    bands = [
        ("4–5 min (open)", 240, 300),
        ("3–4 min", 180, 240),
        ("2–3 min", 120, 180),
        ("90–120 s", 90, 120),
        ("60–90 s", 60, 90),
        ("30–60 s", 30, 60),
        ("0–30 s", 0, 30),
    ]
    counters: dict[str, dict[str, Any]] = {label: {"trades": 0, "usdc": 0.0, "pl": 0.0, "wins": 0, "resolved": 0} for label, _, _ in bands}

    for t in trades:
        slug = t.get("event_slug") or ""
        end_time = _slug_to_window_end(slug)
        if end_time is None:
            continue
        sec_remaining = max(0, int((end_time - t["timestamp"]).total_seconds()))
        for label, lo, hi in bands:
            if lo <= sec_remaining < hi:
                bucket = counters[label]
                bucket["trades"] += 1
                bucket["usdc"] += float(t.get("notional_usd") or 0)
                if t.get("_resolved"):
                    bucket["resolved"] += 1
                    payoff = float(t.get("size") or 0) if t.get("_outcome_won") else 0.0
                    bucket["pl"] += payoff - float(t.get("notional_usd") or 0)
                    if t.get("_outcome_won"):
                        bucket["wins"] += 1
                break
    out: list[TimingStats] = []
    for label, lo, hi in bands:
        c = counters[label]
        n_resolved = c["resolved"]
        usdc = c["usdc"]
        out.append(TimingStats(
            bucket_label=label,
            seconds_low=lo,
            seconds_high=hi,
            trades=c["trades"],
            usdc_deployed=usdc,
            pl_usdc=(c["pl"] if n_resolved else None),
            win_rate=(c["wins"] / n_resolved) if n_resolved else None,
            roi=(c["pl"] / usdc) if n_resolved and usdc > 0 else None,
        ))
    return out


def _slug_to_window_end(slug: str) -> Optional[datetime]:
    """Parse ``btc-updown-5m-1777943100`` → datetime(end of window).

    Returns None for non-crypto slugs.  The trailing integer is the
    market's start_time as Unix seconds.  Horizon is the third token
    (``5m``, ``15m``, ``1h``, ``4h``, ``24h``).
    """
    parts = (slug or "").split("-")
    if len(parts) < 4:
        return None
    horizon = parts[-2].lower()
    try:
        start_epoch = int(parts[-1])
    except ValueError:
        return None
    horizon_seconds_map = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14_400, "24h": 86_400}
    horizon_s = horizon_seconds_map.get(horizon)
    if not horizon_s:
        return None
    start_dt = datetime.fromtimestamp(start_epoch, tz=timezone.utc)
    return start_dt + timedelta(seconds=horizon_s)


# ---------------------------------------------------------------------------
# Rolling P/L windows
# ---------------------------------------------------------------------------


def _compute_rolling_window(
    daily_rows: list[DailyRow], window_days: int,
) -> RollingWindowStats:
    if not daily_rows:
        return RollingWindowStats(
            window_days=window_days, windows_total=0, windows_positive=0,
            min_pl_usdc=None, max_pl_usdc=None,
        )
    pls: list[float] = []
    for i in range(0, max(1, len(daily_rows) - window_days + 1)):
        slice_rows = daily_rows[i : i + window_days]
        # Only count windows that have at least one resolved day
        win_pl = sum(r.pl_usdc for r in slice_rows if r.pl_usdc is not None)
        if any(r.pl_usdc is not None for r in slice_rows):
            pls.append(win_pl)
    return RollingWindowStats(
        window_days=window_days,
        windows_total=len(pls),
        windows_positive=sum(1 for p in pls if p > 0),
        min_pl_usdc=(min(pls) if pls else None),
        max_pl_usdc=(max(pls) if pls else None),
    )


# ---------------------------------------------------------------------------
# Filter ledger
# ---------------------------------------------------------------------------


def _compute_filter_ledger(
    trades: list[dict[str, Any]],
    by_market: dict[str, _MarketAggregate],
) -> list[FilterRow]:
    """Counterfactual filter experiments — 'what if we'd only taken trades matching X?'."""
    baseline = _filter_summary("Unfiltered (baseline)", "Every trade in the wallet's history.", trades, baseline_roi=None)
    out: list[FilterRow] = [baseline]
    base_roi = baseline.roi or 0.0

    def _flt(name: str, desc: str, predicate) -> FilterRow:
        sub = [t for t in trades if predicate(t)]
        return _filter_summary(name, desc, sub, baseline_roi=base_roi)

    out.extend([
        _flt("Price 0.30–0.70", "Drop tails — trade only the fair-value middle band",
             lambda t: 0.30 <= float(t.get("price") or 0) <= 0.70),
        _flt("Price 0.40–0.60", "Tightest middle band",
             lambda t: 0.40 <= float(t.get("price") or 0) <= 0.60),
        _flt("Price < 0.30 (cheap-side hedges)", "Lottery-ticket band — buys at deep discount",
             lambda t: 0 < float(t.get("price") or 0) < 0.30),
        _flt("Price > 0.70 (high-conviction loads)", "Late-window high-confidence buys",
             lambda t: float(t.get("price") or 0) > 0.70),
        _flt("BUY only", "Drop SELL fills (most strategies are buy-and-hold here)",
             lambda t: (t.get("side") or "").upper() == "BUY"),
    ])

    # Dominance filter (per-market): keep trades from markets where dominance ≥ 2× and on the dominant side.
    def _dom_filter(threshold: float, dom_only: bool):
        def _p(t: dict[str, Any]) -> bool:
            key = t.get("market_id") or t.get("event_slug") or "?"
            agg = by_market.get(key)
            if agg is None or agg.dominance_ratio is None or agg.dominance_ratio < threshold:
                return False
            if not dom_only:
                return True
            outcome = (t.get("outcome") or "").upper()
            ds = agg.dominant_side
            return (
                (ds == "UP" and outcome in ("UP", "YES"))
                or (ds == "DOWN" and outcome in ("DOWN", "NO"))
            )
        return _p

    out.extend([
        _flt("Dominance ≥ 2×, dominant side only",
             "Keep only the dominant-side trades in markets that ended ≥ 2× skewed",
             _dom_filter(2.0, dom_only=True)),
        _flt("Dominance ≥ 3×, dominant side only", "Tighter skew threshold",
             _dom_filter(3.0, dom_only=True)),
        _flt("Dominance ≥ 5×, dominant side only", "Tightest skew — near-certain markets",
             _dom_filter(5.0, dom_only=True)),
    ])

    # Underdog blocking — keep all trades EXCEPT underdog-side fills in markets that ended ≥ 2× skewed.
    def _underdog_block(t: dict[str, Any]) -> bool:
        key = t.get("market_id") or t.get("event_slug") or "?"
        agg = by_market.get(key)
        if agg is None or agg.dominance_ratio is None or agg.dominance_ratio < 2.0:
            return True   # keep all trades in low-skew markets
        outcome = (t.get("outcome") or "").upper()
        ds = agg.dominant_side
        is_underdog = (
            (ds == "UP" and outcome in ("DOWN", "NO"))
            or (ds == "DOWN" and outcome in ("UP", "YES"))
        )
        return not is_underdog
    out.append(_flt(
        "Underdog blocking @ ≥ 2× skew",
        "Operationally clean filter — block underdog fills only in markets already developed ≥ 2× skew",
        _underdog_block,
    ))
    return out


def _filter_summary(
    name: str,
    description: str,
    trades: list[dict[str, Any]],
    baseline_roi: Optional[float],
) -> FilterRow:
    n = len(trades)
    usdc = sum(float(t.get("notional_usd") or 0) for t in trades)
    resolved = [t for t in trades if t.get("_resolved")]
    wins = sum(1 for t in resolved if t.get("_outcome_won"))
    pl = (
        sum(
            (float(t.get("size") or 0) if t.get("_outcome_won") else 0.0)
            - float(t.get("notional_usd") or 0)
            for t in resolved
        )
        if resolved else None
    )
    roi = (pl / usdc) if pl is not None and usdc > 0 else None
    lift = (
        (roi - baseline_roi) if (roi is not None and baseline_roi is not None) else None
    )
    return FilterRow(
        name=name,
        description=description,
        trades=n,
        usdc_deployed=usdc,
        pl_usdc=pl,
        win_rate=(wins / len(resolved)) if resolved else None,
        roi=roi,
        roi_lift_vs_baseline=lift,
    )


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _pct(sorted_values: list[float], q: float) -> Optional[float]:
    if not sorted_values:
        return None
    idx = max(0, min(len(sorted_values) - 1, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[idx]


__all__ = [
    "WalletAnalytics",
    "HeadlineMetrics",
    "TwoLegDecomposition",
    "DominanceRow",
    "FilterRow",
    "PriceBucketRow",
    "DailyRow",
    "HourRow",
    "DayOfWeekRow",
    "TopMarketRow",
    "StrategyArchetypeMatch",
    "TimingStats",
    "RollingWindowStats",
    "TradeSizeStats",
    "CadenceStats",
    "analyze",
    "render_spotlight_market",
]
