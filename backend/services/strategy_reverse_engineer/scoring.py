"""Score a backtest result against the wallet's actual trades.

The composite score the reverse-engineer agent optimizes is a convex
combination of:

  * **trade_overlap_pct**       — fraction of backtest fills whose
    (market_id, timestamp ± window) match an actual wallet trade.  The
    most direct signal of "does this strategy reproduce what the trader
    actually did".
  * **side_agreement**          — of overlapping trades, fraction where
    the backtest fill matches the wallet's side (BUY YES vs SELL YES,
    etc.).  Catches strategies that fire at the right time but trade
    the wrong outcome.
  * **pnl_correlation**         — Pearson correlation of cumulative PnL
    series sampled at common timestamps.  When the wallet's PnL is
    flat (test wallet, single-trade wallet, etc.) we fall back to the
    other components.
  * **frequency_match**         — penalty for over-trading or
    under-trading; ratio of |backtest_count - wallet_count| /
    max(backtest_count, wallet_count).
  * **timing_mae_seconds**      — mean absolute error in entry time
    (seconds) for matched trades; surfaced for the LLM critique but
    contributes a soft penalty to the score.

All weights are exposed on the public ``score()`` API so the operator
can tune them via the UI without code changes.
"""
from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Default weights — sum to 1.0 by design so the headline score lands
# in [0, 1].  ``timing_mae_seconds`` is converted to a multiplier
# instead of a weighted addend.
_DEFAULT_WEIGHTS = {
    "trade_overlap": 0.45,
    "side_agreement": 0.20,
    "pnl_correlation": 0.20,
    "frequency_match": 0.15,
}

# Match window: a backtest fill matches a wallet trade when they're on
# the same market and within this many seconds.  300s is generous
# enough to absorb the latency between a "best-execution" backtest and
# Polymarket's real-time confirmation.
_DEFAULT_MATCH_WINDOW_SECONDS = 300


@dataclass
class ScoreBreakdown:
    """All score components — surfaced to the LLM critique + UI."""

    composite: float
    trade_overlap_pct: float
    side_agreement_pct: float
    pnl_correlation: float
    frequency_match: float
    timing_mae_seconds: Optional[float]
    matched_trades: int
    backtest_trade_count: int
    wallet_trade_count: int
    weights: dict[str, float]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "composite": float(self.composite),
            "trade_overlap_pct": float(self.trade_overlap_pct),
            "side_agreement_pct": float(self.side_agreement_pct),
            "pnl_correlation": float(self.pnl_correlation),
            "frequency_match": float(self.frequency_match),
            "timing_mae_seconds": (
                float(self.timing_mae_seconds)
                if self.timing_mae_seconds is not None
                else None
            ),
            "matched_trades": int(self.matched_trades),
            "backtest_trade_count": int(self.backtest_trade_count),
            "wallet_trade_count": int(self.wallet_trade_count),
            "weights": dict(self.weights),
            "notes": list(self.notes),
        }


def score_backtest_against_wallet(
    *,
    backtest_result: dict[str, Any],
    wallet_trades: list[dict[str, Any]],
    weights: Optional[dict[str, float]] = None,
    match_window_seconds: int = _DEFAULT_MATCH_WINDOW_SECONDS,
) -> ScoreBreakdown:
    """Compute the composite score + breakdown.

    ``backtest_result`` is the dict produced by ``run_unified_backtest``
    (the unified runner output).  ``wallet_trades`` is the **normalized**
    trade list from :mod:`wallet_profile` — every entry has at minimum
    ``timestamp`` (datetime), ``market_id``, ``side``, ``outcome``,
    ``price``, ``size``.
    """
    w = {**_DEFAULT_WEIGHTS, **(weights or {})}
    notes: list[str] = []

    bt_fills = _extract_backtest_fills(backtest_result)
    notes.append(f"backtest fills: {len(bt_fills)}")
    notes.append(f"wallet trades: {len(wallet_trades)}")

    if not wallet_trades:
        return ScoreBreakdown(
            composite=0.0,
            trade_overlap_pct=0.0,
            side_agreement_pct=0.0,
            pnl_correlation=0.0,
            frequency_match=0.0,
            timing_mae_seconds=None,
            matched_trades=0,
            backtest_trade_count=len(bt_fills),
            wallet_trade_count=0,
            weights=w,
            notes=notes + ["wallet has no trades to compare against"],
        )

    matches = _match_trades(
        backtest_fills=bt_fills,
        wallet_trades=wallet_trades,
        match_window_seconds=int(match_window_seconds),
    )

    n_match = len(matches)
    n_bt = len(bt_fills)
    n_wallet = len(wallet_trades)

    overlap_pct = (n_match / n_bt) if n_bt else 0.0
    overlap_pct = min(1.0, max(0.0, overlap_pct))

    side_agree = 0
    timing_errors: list[float] = []
    for m in matches:
        bt = m["backtest"]
        wt = m["wallet"]
        if _sides_agree(bt, wt):
            side_agree += 1
        timing_errors.append(abs((bt["timestamp"] - wt["timestamp"]).total_seconds()))
    side_pct = (side_agree / n_match) if n_match else 0.0
    timing_mae = statistics.fmean(timing_errors) if timing_errors else None

    # PnL correlation needs aligned series.  When a side is flat we
    # treat correlation as 0.0 — better than NaN, doesn't push the
    # score around dishonestly.
    pnl_corr = _pnl_correlation(
        backtest_pnl_series=_extract_backtest_pnl_series(backtest_result),
        wallet_trades=wallet_trades,
    )

    freq_match = 1.0 - (
        abs(n_bt - n_wallet) / max(1, max(n_bt, n_wallet))
    )
    freq_match = max(0.0, min(1.0, freq_match))

    composite = (
        w["trade_overlap"] * overlap_pct
        + w["side_agreement"] * side_pct
        + w["pnl_correlation"] * max(0.0, pnl_corr)  # negative corr scored as 0
        + w["frequency_match"] * freq_match
    )

    # Soft timing penalty — anything > 1h average error costs up to 10%.
    if timing_mae is not None and n_match > 0:
        timing_penalty = min(0.10, max(0.0, (timing_mae - 60.0) / 36_000.0))
        composite *= 1.0 - timing_penalty
        notes.append(f"timing penalty applied: {timing_penalty:.4f}")

    composite = max(0.0, min(1.0, float(composite)))

    return ScoreBreakdown(
        composite=composite,
        trade_overlap_pct=float(overlap_pct),
        side_agreement_pct=float(side_pct),
        pnl_correlation=float(pnl_corr),
        frequency_match=float(freq_match),
        timing_mae_seconds=timing_mae,
        matched_trades=n_match,
        backtest_trade_count=n_bt,
        wallet_trade_count=n_wallet,
        weights=w,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Backtest result extraction
# ---------------------------------------------------------------------------


def _extract_backtest_fills(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize the various places fills can appear in a backtest result.

    The unified runner output contains ``execution_backtest`` (or
    occasionally just ``trades`` at the top level depending on the
    runner branch).  We dig for the first non-empty list.
    """
    candidates: list[Any] = []
    if isinstance(result, dict):
        candidates.append(result.get("trades"))
        candidates.append(result.get("fills"))
        for key in ("execution_backtest", "execution"):
            eb = result.get(key)
            if isinstance(eb, dict):
                candidates.append(eb.get("trades"))
                candidates.append(eb.get("fills"))
                # The unified runner emits ``fills_sample`` (the canonical
                # fills list with cap=200 for UI rendering — same shape).
                candidates.append(eb.get("fills_sample"))
    raw = next((c for c in candidates if isinstance(c, list) and c), [])

    out: list[dict[str, Any]] = []
    for entry in raw or []:
        if not isinstance(entry, dict):
            continue
        ts = _coerce_ts(
            entry.get("timestamp")
            or entry.get("ts")
            or entry.get("filled_at")
            or entry.get("placed_at")
            or entry.get("occurred_at")  # unified-runner fill shape
            or entry.get("time")
        )
        if ts is None:
            continue
        out.append(
            {
                "timestamp": ts,
                "market_id": _coerce_market_id(entry),
                "token_id": entry.get("token_id"),
                "side": (str(entry.get("side") or "").strip().upper()) or None,
                "outcome": (str(entry.get("outcome") or "").strip().upper()) or None,
                "price": _coerce_float(entry.get("price") or entry.get("fill_price")),
                "size": _coerce_float(entry.get("size") or entry.get("filled_size") or entry.get("qty")),
                "raw": entry,
            }
        )
    return out


def _extract_backtest_pnl_series(result: dict[str, Any]) -> list[tuple[datetime, float]]:
    """Best-effort extraction of (timestamp, cumulative_pnl) tuples."""
    series: list[tuple[datetime, float]] = []
    if not isinstance(result, dict):
        return series

    candidates: list[Any] = []
    candidates.append(result.get("equity_curve"))
    candidates.append(result.get("equity_curve_sample"))
    # The unified runner emits the curve under ``execution`` as
    # ``equity_curve_sample`` (rows of ``{at, equity_usd}``), so look there too
    # — otherwise pnl_correlation is silently 0 for every unified-runner result.
    for key in ("execution_backtest", "execution"):
        eb = result.get(key)
        if isinstance(eb, dict):
            candidates.append(eb.get("equity_curve"))
            candidates.append(eb.get("equity_history"))
            candidates.append(eb.get("equity_curve_sample"))
    raw = next((c for c in candidates if isinstance(c, list) and c), [])

    for entry in raw or []:
        if isinstance(entry, dict):
            ts = _coerce_ts(
                entry.get("timestamp") or entry.get("ts") or entry.get("time") or entry.get("at")
            )
            value = _coerce_float(
                entry.get("cumulative_pnl")
                or entry.get("pnl")
                or entry.get("equity")
                or entry.get("value")
                or entry.get("equity_usd")
            )
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            ts = _coerce_ts(entry[0])
            value = _coerce_float(entry[1])
        else:
            continue
        if ts is None or value is None:
            continue
        series.append((ts, float(value)))

    series.sort(key=lambda x: x[0])
    return series


def _coerce_market_id(entry: dict[str, Any]) -> Optional[str]:
    candidate = (
        entry.get("market_id")
        or entry.get("condition_id")
        or entry.get("conditionId")
        or entry.get("market")
    )
    if candidate is None:
        # Fallback: derive market_id from synthetic polybacktest token_id.
        token_id = str(entry.get("token_id") or "")
        if token_id.startswith("polybacktest:"):
            parts = token_id.split(":")
            if len(parts) >= 4:
                return f"polybacktest:{parts[1]}:{parts[2]}"
    if not candidate:
        return None
    return str(candidate)


def _coerce_ts(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        if isinstance(value, (int, float)):
            v = float(value)
            if v > 1e12:
                v = v / 1000.0
            return datetime.fromtimestamp(v, tz=timezone.utc)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if "T" in text or "-" in text:
                if text.endswith("Z"):
                    text = text[:-1] + "+00:00"
                dt = datetime.fromisoformat(text)
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            v = float(text)
            if v > 1e12:
                v = v / 1000.0
            return datetime.fromtimestamp(v, tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:
        return None
    return result


# ---------------------------------------------------------------------------
# Trade matching
# ---------------------------------------------------------------------------


def _match_trades(
    *,
    backtest_fills: list[dict[str, Any]],
    wallet_trades: list[dict[str, Any]],
    match_window_seconds: int,
) -> list[dict[str, Any]]:
    """Greedy time-window matcher with provider-token alias support.

    For each backtest fill (in time order) pick the closest unmatched
    wallet trade on the same market within the window.  A "same market"
    match accepts:

      * Exact equality on ``market_id`` (the common case), OR
      * The fill's market_id appears in the wallet trade's
        ``alias_market_ids`` list (set by the agent's bootstrap when
        a polybacktest provider dataset is in scope — it tags each
        wallet trade with the corresponding ``polybacktest:btc:<id>:<side>``
        token so fills emitted against synthetic provider tokens still
        match the wallet's real Polymarket trades).

    Greedy is fine here — perfect bipartite matching would be overkill
    for a score that's already a soft signal to the LLM.
    """
    if not backtest_fills or not wallet_trades:
        return []
    used = set()
    by_market: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for idx, t in enumerate(wallet_trades):
        mid = t.get("market_id") or "?"
        by_market.setdefault(mid, []).append((idx, t))
        # Index aliases too so fills emitted against synthetic
        # provider tokens land in the same lookup table.
        for alias in t.get("alias_market_ids") or []:
            by_market.setdefault(str(alias), []).append((idx, t))

    window = timedelta(seconds=int(max(0, match_window_seconds)))
    matches: list[dict[str, Any]] = []
    for bt in sorted(backtest_fills, key=lambda x: x["timestamp"]):
        candidates = by_market.get(bt.get("market_id") or "?")
        if not candidates:
            continue
        best_idx: Optional[int] = None
        best_dt: Optional[timedelta] = None
        for idx, wt in candidates:
            if idx in used:
                continue
            dt = abs(bt["timestamp"] - wt["timestamp"])
            if dt > window:
                continue
            if best_dt is None or dt < best_dt:
                best_idx = idx
                best_dt = dt
        if best_idx is not None:
            used.add(best_idx)
            matches.append({"backtest": bt, "wallet": wallet_trades[best_idx]})
    return matches


def _sides_agree(bt: dict[str, Any], wt: dict[str, Any]) -> bool:
    bt_side = (bt.get("side") or "").upper()
    wt_side = (wt.get("side") or "").upper()
    if not bt_side or not wt_side:
        # Fall back to outcome agreement (YES/NO, UP/DOWN) when side
        # isn't tagged.  Worse signal but better than 0 credit.
        bt_outcome = (bt.get("outcome") or "").upper()
        wt_outcome = (wt.get("outcome") or "").upper()
        if bt_outcome and wt_outcome:
            return bt_outcome == wt_outcome
        return False
    return bt_side == wt_side


# ---------------------------------------------------------------------------
# PnL correlation
# ---------------------------------------------------------------------------


def _pnl_correlation(
    *,
    backtest_pnl_series: list[tuple[datetime, float]],
    wallet_trades: list[dict[str, Any]],
) -> float:
    """Pearson correlation of cumulative PnL sampled at common buckets.

    We don't have a definitive wallet PnL stream; build one by
    cumulating each trade's signed notional (rough proxy: BUY = +,
    SELL = -).  When the data is too thin for a meaningful corr we
    return 0.0.
    """
    if not backtest_pnl_series or len(backtest_pnl_series) < 5 or not wallet_trades:
        return 0.0

    wallet_series = _wallet_proxy_pnl_series(wallet_trades)
    if not wallet_series or len(wallet_series) < 5:
        return 0.0

    # Resample both onto a grid sized to the window.  A fixed hourly grid made
    # pnl_correlation undefined for any intraday backtest (a 4h window has only
    # ~4 hourly buckets, below the 5-bucket floor → always 0).  Aim for ~200
    # buckets, clamped to [30s, 1h], so intraday windows resample finely enough
    # to correlate while multi-day windows stay coarse.
    span_s = max(
        (backtest_pnl_series[-1][0] - backtest_pnl_series[0][0]).total_seconds(),
        (wallet_series[-1][0] - wallet_series[0][0]).total_seconds(),
        1.0,
    )
    granularity = int(min(3600, max(30, span_s / 200)))
    bt_resampled = _resample_to_grid(backtest_pnl_series, granularity_seconds=granularity)
    wallet_resampled = _resample_to_grid(wallet_series, granularity_seconds=granularity)

    common_times = sorted(set(bt_resampled) & set(wallet_resampled))
    if len(common_times) < 5:
        return 0.0

    xs = [bt_resampled[t] for t in common_times]
    ys = [wallet_resampled[t] for t in common_times]

    try:
        return float(_pearson(xs, ys))
    except Exception:
        return 0.0


def _wallet_proxy_pnl_series(
    wallet_trades: list[dict[str, Any]],
) -> list[tuple[datetime, float]]:
    cum = 0.0
    out: list[tuple[datetime, float]] = []
    for t in sorted(wallet_trades, key=lambda x: x["timestamp"]):
        notional = t.get("notional_usd")
        if notional is None:
            continue
        side = (t.get("side") or "").upper()
        signed = float(notional) * (1.0 if side == "BUY" else -1.0 if side == "SELL" else 0.0)
        cum += signed
        out.append((t["timestamp"], cum))
    return out


def _resample_to_grid(
    series: list[tuple[datetime, float]],
    *,
    granularity_seconds: int,
) -> dict[datetime, float]:
    """Forward-fill onto a fixed-stride grid; returns {bucket_ts: value}."""
    if not series:
        return {}
    out: dict[datetime, float] = {}
    series = sorted(series, key=lambda x: x[0])
    last_value = series[0][1]
    cursor = _floor_to_grid(series[0][0], granularity_seconds)
    end = series[-1][0]
    idx = 0
    while cursor <= end:
        while idx < len(series) and series[idx][0] <= cursor:
            last_value = series[idx][1]
            idx += 1
        out[cursor] = float(last_value)
        cursor = cursor + timedelta(seconds=granularity_seconds)
    return out


def _floor_to_grid(ts: datetime, granularity_seconds: int) -> datetime:
    epoch = int(ts.timestamp())
    floored = epoch - (epoch % max(1, granularity_seconds))
    return datetime.fromtimestamp(floored, tz=timezone.utc)


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n != len(ys) or n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = math.sqrt(sum((xs[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ys[i] - my) ** 2 for i in range(n)))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)
