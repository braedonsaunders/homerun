# SEED TEMPLATE — not the live strategy. The DB `strategies.source_code` is the
# runtime master (loaded by the orchestrator + backtester); edits here only seed
# fresh installs / reset-to-factory. See services/opportunity_strategy_catalog.py.
"""Crypto digital sigma edge — vol-normalized fair value for up/down cycles.

Every Polymarket crypto up/down cycle is a cash-or-nothing digital option on
the Chainlink oracle: it pays $1 when the oracle ends the cycle on your side
of the strike (``price_to_beat``).  This strategy prices that digital
directly:

  1. Estimate realized per-second volatility from the dispatcher's recorded
     ``oracle_history`` (sum of squared log returns over elapsed time — the
     canonical realized-variance estimator), over a rolling lookback.
  2. Fair value of the UP side is ``prob_above(spot, strike, sigma,
     seconds_left)`` under zero-drift GBM (StrategySDK digital math).
  3. Compare fair value to the EXECUTABLE price on each side — the recorded
     top-of-book from the event payload itself (UP at ``best_ask``; DOWN via
     complement parity ``1 - best_bid`` plus a slippage buffer, since the
     dispatch carries only the UP token's book).
  4. Take whichever side offers ``fair - cost`` above the fee-adjusted edge
     floor.  Both sides are tradeable: late-cycle favorites the market
     under-prices AND overpriced favorites whose tail the market ignores.

What makes it different from the other crypto strategies: none of them
estimate volatility.  ``btc_eth_directional_edge`` keys on raw oracle-vs-
strike divergence, ``crypto_distance_edge`` on fixed dollar distances,
``crypto_5m_midcycle`` on a fixed bps move at one milestone.  A $40 BTC move
with 60s left is decisive in a quiet regime and noise in a violent one — the
same distance can be a buy or a pass depending on sigma.  Pricing the digital
normalizes every (distance, time-left, regime) triple into one number that is
comparable across assets and timeframes, which is also why one config covers
5m and 15m cycles without per-asset distance tables.

Backtest fidelity note: entry gating reads ONLY event-payload fields
(oracle_history, oracle_prices_by_source, best_bid/best_ask, price_to_beat,
seconds_left) — all recorded on the bus and byte-identical in replay.  It
deliberately does NOT gate on ``StrategySDK.get_order_book_depth`` (live WS
cache): that cache is empty during replay, so a depth gate would make the
strategy silently un-backtestable.  Execution realism (real L2 ladders,
queue position, impact) is the matching engine's job at fill time.

Resolution truth source: Polymarket resolves these cycles against Chainlink
Data Streams, so the oracle preference defaults to ``chainlink`` — the same
series the resolution reads, NOT Binance spot.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from utils.utcnow import utcnow  # replay-clock-aware "now" (honors backtest sim time)
from models import Opportunity
from services.data_events import DataEvent
from services.strategies._firehose import (
    GateResult,
    MURMUR,
    WHISPER,
    emit_emit_nowait,
    emit_evaluation_nowait,
)
from services.strategies.base import BaseStrategy
from services.strategy_helpers.crypto_strategy_utils import (
    build_binary_crypto_market,
    pick_oracle_source,
)
from services.strategy_sdk import StrategySDK
from utils.converters import to_float
from utils.logger import get_logger

logger = get_logger(__name__)

_ALL_ASSETS: tuple[str, ...] = ("BTC", "ETH", "SOL", "XRP")

_TIMEFRAME_SECONDS: dict[str, float] = {
    "5m": 300.0,
    "15m": 900.0,
    "1h": 3600.0,
    "4h": 14400.0,
}


DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "assets": ["BTC", "ETH", "SOL", "XRP"],
    # Cycle cadences to trade.  5m/15m resolve fast enough that the
    # zero-drift GBM assumption holds and capital recycles quickly.
    "timeframes": ["5m", "15m"],
    # Sigma estimation: realized variance over this much oracle history.
    "vol_lookback_seconds": 900.0,
    # Reject sigma built from fewer than this many usable intervals, or
    # from history spanning less than this many seconds — a thin sample
    # saturates the digital to 0/1 and manufactures fake edge.
    "min_vol_intervals": 12,
    "min_history_span_seconds": 240.0,
    # Per-second vol sanity rails (unitless return/sec).  Outside them the
    # estimate is degenerate (flat feed) or violent (oracle gap regime).
    "vol_floor": 0.000002,
    "vol_cap": 0.005,
    # Oracle: must match the resolution series and be fresh at decision.
    "oracle_source_preference": "chainlink",
    "max_oracle_age_ms": 4000,
    # Entry timing: skip the cycle-open noise and the terminal race.
    "min_cycle_fraction": 0.15,
    "max_cycle_fraction": 0.85,
    "min_seconds_left": 45.0,
    # Edge: fair-minus-cost must clear fees plus a margin of model error.
    # fee_buffer approximates Polymarket taker fees on fee-enabled crypto
    # cycles; min_edge is the post-fee floor that must remain.
    "fee_buffer": 0.015,
    "min_edge": 0.045,
    # DOWN-side executable cost is inferred from the UP book by complement
    # parity (the dispatch only carries the UP token's book); pad it so
    # vig/thin DOWN books don't flatter the edge.
    "down_side_slippage_buffer": 0.01,
    # Executable-price rails.
    "min_entry_price": 0.05,
    "max_entry_price": 0.92,
    "max_spread": 0.08,
    "min_liquidity": 500.0,
    # Skip oracle jump regimes where trailing realized vol lags reality.
    "max_recent_move_zscore": 3.5,
    "bet_size_usd": 15.0,
}


def _normalize_asset(value: Any) -> str:
    asset = str(value or "").strip().upper()
    if asset == "XBT":
        return "BTC"
    return asset


def _normalize_timeframe(value: Any) -> str:
    tf = str(value or "").strip().lower().replace(" ", "")
    aliases = {
        "5m": "5m", "5min": "5m", "5-minute": "5m", "5minute": "5m",
        "15m": "15m", "15min": "15m", "15-minute": "15m", "15minute": "15m",
        "1h": "1h", "1hr": "1h", "60m": "1h", "hourly": "1h",
        "4h": "4h", "4hr": "4h", "240m": "4h",
    }
    return aliases.get(tf, tf)


def _realized_vol_per_sec(
    history: Any,
    *,
    now_ms: float,
    lookback_seconds: float,
    min_intervals: int,
    min_span_seconds: float,
) -> tuple[Optional[float], int, float]:
    """Realized per-second volatility from ``[{"t": ms, "p": price}, ...]``.

    sigma^2 = sum(log-return^2) / sum(dt) over the lookback — the canonical
    realized-variance-per-unit-time estimator, robust to the dispatcher's
    irregular ~7-8s sampling.  Intervals longer than 60s are gaps (feed
    outage), not information, and are excluded.

    Returns (sigma_per_sec | None, usable_intervals, span_seconds).
    """
    if not isinstance(history, list) or len(history) < 2:
        return None, 0, 0.0
    cutoff_ms = now_ms - lookback_seconds * 1000.0
    pts: list[tuple[float, float]] = []
    for h in history:
        if not isinstance(h, dict):
            continue
        t = to_float(h.get("t"), None)
        p = to_float(h.get("p"), None)
        if t is None or p is None or p <= 0.0 or t < cutoff_ms or t > now_ms + 1000.0:
            continue
        pts.append((t, p))
    if len(pts) < min_intervals + 1:
        return None, max(0, len(pts) - 1), 0.0
    pts.sort(key=lambda x: x[0])
    span_seconds = (pts[-1][0] - pts[0][0]) / 1000.0
    sum_r2 = 0.0
    sum_dt = 0.0
    n = 0
    for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
        dt = (t1 - t0) / 1000.0
        if dt <= 0.0 or dt > 60.0:
            continue
        r = math.log(p1 / p0)
        sum_r2 += r * r
        sum_dt += dt
        n += 1
    if n < min_intervals or sum_dt <= 0.0 or span_seconds < min_span_seconds:
        return None, n, span_seconds
    return math.sqrt(sum_r2 / sum_dt), n, span_seconds


class CryptoDigitalSigmaEdgeStrategy(BaseStrategy):
    """Vol-normalized digital fair value vs executable book, both sides."""

    strategy_type = "crypto_digital_sigma_edge"
    name = "Crypto Digital Sigma Edge"
    description = (
        "Prices each crypto up/down cycle as a cash-or-nothing digital using "
        "realized oracle volatility, then buys whichever side the book sells "
        "below fair value by more than fees — favorites and underdogs alike."
    )
    source_key = "crypto"
    market_categories = ["crypto"]
    requires_historical_prices = False
    subscriptions = ["crypto_update"]
    supports_entry_take_profit_exit = False
    default_open_order_timeout_seconds = 20.0

    default_config = dict(DEFAULT_CONFIG)

    def __init__(self) -> None:
        super().__init__()
        self.min_profit = 0.0
        self.fee = 0.0
        # One entry per market per cycle: each cycle is its own condition_id,
        # so a plain seen-set self-cleans as cycles roll over.  Bounded sweep
        # keeps long-running processes flat.
        self._entered_market_ids: set[str] = set()

    def configure(self, config: dict) -> None:
        merged = dict(self.default_config)
        if config:
            merged.update(config)
        raw_assets = merged.get("assets") or []
        if isinstance(raw_assets, str):
            raw_assets = [a.strip() for a in raw_assets.split(",")]
        assets: list[str] = []
        seen: set[str] = set()
        for a in raw_assets:
            n = _normalize_asset(a)
            if n and n in _ALL_ASSETS and n not in seen:
                seen.add(n)
                assets.append(n)
        merged["assets"] = assets
        raw_tfs = merged.get("timeframes") or []
        if isinstance(raw_tfs, str):
            raw_tfs = [t.strip() for t in raw_tfs.split(",")]
        tfs: list[str] = []
        seen_tf: set[str] = set()
        for t in raw_tfs:
            n = _normalize_timeframe(t)
            if n in _TIMEFRAME_SECONDS and n not in seen_tf:
                seen_tf.add(n)
                tfs.append(n)
        merged["timeframes"] = tfs
        self.config = merged

    async def on_event(self, event: DataEvent) -> list[Opportunity]:
        if event.event_type != "crypto_update":
            return []
        if not self.config.get("enabled", True):
            return []
        markets = event.payload.get("markets") or []
        if not markets:
            return []
        if len(self._entered_market_ids) > 4096:
            self._entered_market_ids.clear()
        now_ms = utcnow().timestamp() * 1000.0
        out: list[Opportunity] = []
        for market in markets:
            if not isinstance(market, dict):
                continue
            opp = self._evaluate_market(market, now_ms=now_ms)
            if opp is not None:
                out.append(opp)
        return out

    def _evaluate_market(
        self, market: dict[str, Any], *, now_ms: float
    ) -> Optional[Opportunity]:
        cfg = self.config
        gates: list[GateResult] = []

        def _reject(verbosity: str = WHISPER) -> None:
            emit_evaluation_nowait(
                strategy_slug=self.strategy_type,
                market=market,
                gates=gates,
                outcome="rejected",
                verbosity=verbosity,
            )

        timeframe = _normalize_timeframe(market.get("timeframe"))
        tf_ok = timeframe in (cfg.get("timeframes") or [])
        gates.append(GateResult(
            "timeframe", "Cycle timeframe enabled", tf_ok,
            detail=f"timeframe={timeframe or '?'}",
        ))
        if not tf_ok:
            _reject()
            return None

        market_id = str(market.get("condition_id") or market.get("id") or "")
        gates.append(GateResult(
            "market_id", "Market id present", bool(market_id),
            detail="condition_id or id required",
        ))
        if not market_id:
            _reject()
            return None
        already = market_id in self._entered_market_ids
        gates.append(GateResult(
            "one_entry_per_cycle", "No prior entry this cycle", not already,
            detail=f"market_id={market_id[:18]}",
        ))
        if already:
            _reject()
            return None

        asset = _normalize_asset(
            market.get("asset") or market.get("symbol") or market.get("coin")
        )
        asset_ok = bool(asset) and asset in (cfg.get("assets") or [])
        gates.append(GateResult(
            "asset_enabled", "Asset in config list", asset_ok,
            detail=f"asset={asset or '?'}",
        ))
        if not asset_ok:
            _reject()
            return None

        end_ms = StrategySDK._coerce_end_ts_ms(market)
        gates.append(GateResult(
            "end_timestamp", "Cycle end timestamp parseable", end_ms is not None,
            detail=f"end_ts_ms={end_ms}",
        ))
        if end_ms is None:
            _reject()
            return None
        seconds_left = (end_ms - now_ms) / 1000.0
        cycle_seconds = to_float(market.get("timeframe_seconds"), None) or _TIMEFRAME_SECONDS[timeframe]
        elapsed_fraction = 1.0 - (seconds_left / cycle_seconds) if cycle_seconds > 0 else 1.0
        min_frac = float(cfg.get("min_cycle_fraction", 0.15))
        max_frac = float(cfg.get("max_cycle_fraction", 0.85))
        min_left = float(cfg.get("min_seconds_left", 45.0))
        timing_ok = (
            seconds_left >= min_left and min_frac <= elapsed_fraction <= max_frac
        )
        gates.append(GateResult(
            "entry_window", "Inside entry window", timing_ok,
            score=elapsed_fraction,
            detail=(
                f"elapsed={elapsed_fraction:.2f} of cycle "
                f"(band [{min_frac:.2f},{max_frac:.2f}]) left={seconds_left:.0f}s"
            ),
        ))
        if not timing_ok:
            _reject()
            return None

        strike = to_float(market.get("price_to_beat"), None)
        strike_ok = strike is not None and strike > 0.0
        gates.append(GateResult(
            "strike", "Strike (price_to_beat) present", strike_ok,
            score=strike, detail=f"price_to_beat={strike}",
        ))
        if not strike_ok:
            _reject()
            return None

        oracle = pick_oracle_source(
            market,
            prefer=str(cfg.get("oracle_source_preference", "chainlink")),
            max_age_ms=float(cfg.get("max_oracle_age_ms", 4000)),
            now_ms=now_ms,
        )
        oracle_ok = (
            oracle is not None
            and str(oracle.get("source", "")).lower()
            == str(cfg.get("oracle_source_preference", "chainlink")).lower()
        )
        gates.append(GateResult(
            "fresh_oracle", "Fresh resolution-source oracle", oracle_ok,
            score=float(oracle.get("age_ms", 0.0)) if oracle else None,
            detail=f"source={(oracle or {}).get('source') or 'none'}",
        ))
        if not oracle_ok:
            _reject(MURMUR)
            return None
        spot = float(oracle["price"])
        if spot <= 0.0:
            gates.append(GateResult("spot", "Spot > 0", False, detail=f"spot={spot}"))
            _reject(MURMUR)
            return None

        zscore = to_float(market.get("recent_move_zscore"), 0.0) or 0.0
        z_cap = float(cfg.get("max_recent_move_zscore", 3.5))
        z_ok = abs(zscore) <= z_cap
        gates.append(GateResult(
            "jump_regime", "Not in oracle jump regime", z_ok,
            score=zscore, detail=f"|z|={abs(zscore):.2f} cap={z_cap:.2f}",
        ))
        if not z_ok:
            _reject(MURMUR)
            return None

        sigma, n_intervals, span_s = _realized_vol_per_sec(
            market.get("oracle_history"),
            now_ms=now_ms,
            lookback_seconds=float(cfg.get("vol_lookback_seconds", 900.0)),
            min_intervals=int(cfg.get("min_vol_intervals", 12)),
            min_span_seconds=float(cfg.get("min_history_span_seconds", 240.0)),
        )
        sigma_ok = sigma is not None
        gates.append(GateResult(
            "realized_vol", "Realized vol estimable", sigma_ok,
            score=sigma,
            detail=f"sigma={sigma if sigma is not None else '?'} n={n_intervals} span={span_s:.0f}s",
        ))
        if not sigma_ok:
            _reject(MURMUR)
            return None
        vol_floor = float(cfg.get("vol_floor", 0.000002))
        vol_cap = float(cfg.get("vol_cap", 0.005))
        rails_ok = vol_floor <= sigma <= vol_cap
        gates.append(GateResult(
            "vol_rails", "Sigma within sanity rails", rails_ok,
            score=sigma, detail=f"sigma={sigma:.3e} rails=[{vol_floor:.1e},{vol_cap:.1e}]",
        ))
        if not rails_ok:
            _reject(MURMUR)
            return None

        p_up = StrategySDK.prob_above(spot, strike, sigma, seconds_left)
        gates.append(GateResult(
            "fair_value", "Digital fair value computed", p_up is not None,
            score=p_up, detail=f"p_up={p_up}",
        ))
        if p_up is None:
            _reject(MURMUR)
            return None

        best_bid = to_float(market.get("best_bid"), None)
        best_ask = to_float(market.get("best_ask"), None)
        spread = to_float(market.get("spread"), None)
        if spread is None and best_bid is not None and best_ask is not None:
            spread = best_ask - best_bid
        book_ok = (
            best_bid is not None and best_ask is not None
            and 0.0 < best_bid < 1.0 and 0.0 < best_ask < 1.0
            and best_ask > best_bid - 1e-9
        )
        gates.append(GateResult(
            "payload_book", "Recorded top-of-book present", book_ok,
            detail=f"bid={best_bid} ask={best_ask}",
        ))
        if not book_ok:
            _reject(MURMUR)
            return None
        max_spread = float(cfg.get("max_spread", 0.08))
        spread_ok = spread is not None and spread <= max_spread
        gates.append(GateResult(
            "spread", "UP-book spread acceptable", spread_ok,
            score=spread, detail=f"spread={spread} max={max_spread}",
        ))
        if not spread_ok:
            _reject(MURMUR)
            return None
        liquidity = to_float(market.get("liquidity"), 0.0) or 0.0
        min_liq = float(cfg.get("min_liquidity", 500.0))
        liq_ok = liquidity >= min_liq
        gates.append(GateResult(
            "liquidity", "Market liquidity floor", liq_ok,
            score=liquidity, detail=f"liquidity={liquidity:.0f} min={min_liq:.0f}",
        ))
        if not liq_ok:
            _reject(MURMUR)
            return None

        fee_buffer = float(cfg.get("fee_buffer", 0.015))
        down_buffer = float(cfg.get("down_side_slippage_buffer", 0.01))
        cost_up = best_ask
        cost_down = (1.0 - best_bid) + down_buffer
        edge_up = p_up - cost_up - fee_buffer
        edge_down = (1.0 - p_up) - cost_down - fee_buffer
        if edge_up >= edge_down:
            side, fair, cost, edge = "UP", p_up, cost_up, edge_up
        else:
            side, fair, cost, edge = "DOWN", 1.0 - p_up, cost_down, edge_down
        min_edge = float(cfg.get("min_edge", 0.045))
        edge_ok = edge >= min_edge
        gates.append(GateResult(
            "net_edge", "Fair-minus-cost clears edge floor", edge_ok,
            score=edge,
            detail=(
                f"side={side} fair={fair:.3f} cost={cost:.3f} "
                f"edge={edge:+.3f} floor={min_edge:.3f}"
            ),
        ))
        if not edge_ok:
            _reject(MURMUR)
            return None

        min_entry = float(cfg.get("min_entry_price", 0.05))
        max_entry = float(cfg.get("max_entry_price", 0.92))
        entry_ok = min_entry <= cost <= max_entry
        gates.append(GateResult(
            "entry_price", "Entry price within rails", entry_ok,
            score=cost, detail=f"cost={cost:.3f} rails=[{min_entry:.2f},{max_entry:.2f}]",
        ))
        if not entry_ok:
            _reject(MURMUR)
            return None

        typed_market = build_binary_crypto_market(market)
        if typed_market is None or not typed_market.clob_token_ids:
            gates.append(GateResult(
                "clob_tokens", "CLOB token ids present", False,
                detail="build_binary_crypto_market failed",
            ))
            _reject(MURMUR)
            return None
        up_idx = int(to_float(market.get("up_token_index"), 0) or 0)
        down_idx = int(to_float(market.get("down_token_index"), 1) or 1)
        try:
            token_id = typed_market.clob_token_ids[up_idx if side == "UP" else down_idx]
        except IndexError:
            _reject(MURMUR)
            return None

        outcome = "YES" if side == "UP" else "NO"
        bet_size_usd = float(cfg.get("bet_size_usd", 15.0))
        slug = str(market.get("slug") or market_id)
        title = f"Digital sigma edge: {slug} {side}"
        description = (
            f"{asset} {timeframe} | fair={fair:.3f} vs cost={cost:.3f} "
            f"(edge {edge:+.3f} after {fee_buffer:.3f} fees) | "
            f"sigma={sigma:.2e}/s n={n_intervals} | left={seconds_left:.0f}s"
        )
        opp = self.create_opportunity(
            title=title,
            description=description,
            total_cost=cost,
            expected_payout=fair,
            markets=[typed_market],
            positions=[
                {
                    "action": "BUY",
                    "outcome": outcome,
                    "price": cost,
                    "token_id": token_id,
                    "_sigma_edge_context": {
                        "asset": asset,
                        "timeframe": timeframe,
                        "side": side,
                        "strike": strike,
                        "spot": spot,
                        "sigma_per_sec": sigma,
                        "vol_intervals": n_intervals,
                        "fair_value": fair,
                        "executable_cost": cost,
                        "net_edge": edge,
                        "seconds_left": seconds_left,
                        "elapsed_fraction": elapsed_fraction,
                        "oracle_source": oracle.get("source"),
                        "oracle_age_ms": float(oracle.get("age_ms") or 0.0),
                        "bet_size_usd": bet_size_usd,
                    },
                }
            ],
            is_guaranteed=False,
            skip_fee_model=True,
            custom_roi_percent=(edge / cost) * 100.0 if cost > 0 else 0.0,
            custom_risk_score=1.0 - fair,
            confidence=fair,
        )
        if opp is None:
            emit_evaluation_nowait(
                strategy_slug=self.strategy_type,
                market=market,
                gates=gates,
                outcome="rejected",
                verbosity=MURMUR,
                extra={"reason": "create_opportunity returned None"},
            )
            return None

        self._entered_market_ids.add(market_id)
        emit_emit_nowait(
            strategy_slug=self.strategy_type,
            market=market,
            detail=(
                f"{asset} {timeframe} {side} • fair={fair:.3f} cost={cost:.3f} "
                f"edge={edge:+.3f} • sigma={sigma:.2e}/s • left={seconds_left:.0f}s"
            ),
            extra={
                "side": side,
                "asset": asset,
                "fair_value": fair,
                "cost": cost,
                "net_edge": edge,
            },
        )
        emit_evaluation_nowait(
            strategy_slug=self.strategy_type,
            market=market,
            gates=gates,
            outcome="emitted",
            verbosity=WHISPER,
        )
        opp.risk_factors = [
            f"Binary cycle resolution risk ({asset} {timeframe})",
            f"Fair value {fair:.3f} from realized sigma {sigma:.2e}/s over {span_s:.0f}s",
            f"Zero-drift GBM assumption over {seconds_left:.0f}s horizon",
            f"DOWN-side cost inferred by complement parity" if side == "DOWN"
            else f"Entry at recorded best ask {cost:.3f}",
        ]
        opp.strategy_context = {
            "source_key": "crypto",
            "strategy": self.strategy_type,
            "asset": asset,
            "timeframe": timeframe,
            "side": side,
            "strike": strike,
            "spot": spot,
            "sigma_per_sec": sigma,
            "fair_value": fair,
            "executable_cost": cost,
            "net_edge": edge,
            "seconds_left": seconds_left,
            "bet_size_usd": bet_size_usd,
            "win_prob_estimate": fair,
        }
        return opp
