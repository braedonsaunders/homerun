from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


_MIN_PRICE = 0.0001
_MAX_PRICE = 0.9999


@dataclass(frozen=True)
class ExecutionEstimatorConfig:
    fee_bps: float = 0.0
    latency_ms: float = 350.0
    time_in_force_seconds: float = 6.0
    displayed_depth_factor: float = 0.88
    min_depth_factor: float = 0.20
    max_book_age_ms: float = 10_000.0
    stale_depth_decay: float = 0.55
    maker_queue_ahead_fraction: float = 0.65
    maker_trade_flow_multiplier: float = 1.20
    adverse_selection_multiplier: float = 0.70
    recent_trade_lookback_seconds: float = 30.0


@dataclass(frozen=True)
class ExecutionEstimate:
    side: str
    order_type: str
    status: str
    reason: str
    requested_shares: float
    filled_shares: float
    remaining_shares: float
    requested_notional_usd: float
    filled_notional_usd: float
    average_price: float | None
    economic_price: float | None
    limit_price: float | None
    fees_usd: float
    slippage_bps: float
    price_impact_bps: float
    adverse_selection_bps: float
    adverse_selection_cost_usd: float
    latency_ms: float
    fill_probability: float
    queue_ahead_shares: float
    levels_consumed: int
    book_depth_factor: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def fill_ratio(self) -> float:
        if self.requested_shares <= 0:
            return 0.0
        return min(1.0, max(0.0, self.filled_shares / self.requested_shares))

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "order_type": self.order_type,
            "status": self.status,
            "reason": self.reason,
            "requested_shares": self.requested_shares,
            "filled_shares": self.filled_shares,
            "remaining_shares": self.remaining_shares,
            "requested_notional_usd": self.requested_notional_usd,
            "filled_notional_usd": self.filled_notional_usd,
            "average_price": self.average_price,
            "economic_price": self.economic_price,
            "limit_price": self.limit_price,
            "fees_usd": self.fees_usd,
            "slippage_bps": self.slippage_bps,
            "price_impact_bps": self.price_impact_bps,
            "adverse_selection_bps": self.adverse_selection_bps,
            "adverse_selection_cost_usd": self.adverse_selection_cost_usd,
            "latency_ms": self.latency_ms,
            "fill_probability": self.fill_probability,
            "fill_ratio": self.fill_ratio,
            "queue_ahead_shares": self.queue_ahead_shares,
            "levels_consumed": self.levels_consumed,
            "book_depth_factor": self.book_depth_factor,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class MultiLegExecutionEstimate:
    leg_estimates: list[ExecutionEstimate]
    all_filled: bool
    joint_fill_probability: float
    filled_notional_usd: float
    fees_usd: float
    adverse_selection_cost_usd: float
    orphan_leg_notional_usd: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "leg_estimates": [estimate.to_dict() for estimate in self.leg_estimates],
            "all_filled": self.all_filled,
            "joint_fill_probability": self.joint_fill_probability,
            "filled_notional_usd": self.filled_notional_usd,
            "fees_usd": self.fees_usd,
            "adverse_selection_cost_usd": self.adverse_selection_cost_usd,
            "orphan_leg_notional_usd": self.orphan_leg_notional_usd,
            "diagnostics": dict(self.diagnostics),
        }


class ExecutionEstimator:
    def estimate_order(
        self,
        *,
        order_book: Any,
        side: str,
        size_usd: float | None = None,
        size_shares: float | None = None,
        limit_price: float | None = None,
        order_type: str = "limit",
        recent_trades: list[Any] | None = None,
        book_age_ms: float | None = None,
        config: ExecutionEstimatorConfig | None = None,
    ) -> ExecutionEstimate:
        cfg = config or ExecutionEstimatorConfig()
        side_key = self._normalize_side(side)
        order_type_key = str(order_type or "limit").strip().lower()
        limit = self._normalize_price(limit_price)
        trades = list(recent_trades or [])
        mid = self._mid_price(order_book)
        best_bid = self._best_price(order_book, "bid")
        best_ask = self._best_price(order_book, "ask")
        reference_price = self._reference_price(
            side=side_key,
            limit_price=limit,
            best_bid=best_bid,
            best_ask=best_ask,
            mid=mid,
        )
        requested_shares = self._requested_shares(size_usd, size_shares, reference_price)
        explicit_size_usd = self._coerce_float(size_usd, 0.0)
        requested_notional = (
            explicit_size_usd
            if explicit_size_usd > 0
            else requested_shares * reference_price if reference_price > 0 else 0.0
        )
        if requested_shares <= 0 or reference_price <= 0:
            return self._empty_estimate(
                side=side_key,
                order_type=order_type_key,
                reason="invalid_size_or_price",
                requested_shares=max(0.0, requested_shares),
                requested_notional_usd=max(0.0, requested_notional),
                limit_price=limit,
                latency_ms=cfg.latency_ms,
            )

        if self._is_marketable(side_key, limit, best_bid, best_ask) or order_type_key in {"market", "taker", "taker_limit"}:
            return self._estimate_taker(
                order_book=order_book,
                side=side_key,
                requested_shares=requested_shares,
                requested_notional=requested_notional,
                spend_budget_usd=explicit_size_usd if explicit_size_usd > 0 else None,
                limit_price=limit,
                recent_trades=trades,
                book_age_ms=book_age_ms,
                config=cfg,
            )

        return self._estimate_maker(
            order_book=order_book,
            side=side_key,
            requested_shares=requested_shares,
            requested_notional=requested_notional,
            limit_price=limit,
            recent_trades=trades,
            book_age_ms=book_age_ms,
            config=cfg,
        )

    def estimate_multileg(self, leg_estimates: list[ExecutionEstimate]) -> MultiLegExecutionEstimate:
        joint_probability = 1.0
        filled_notional = 0.0
        fees = 0.0
        adverse_cost = 0.0
        orphan_notional = 0.0
        filled_legs = 0
        for estimate in leg_estimates:
            joint_probability *= max(0.0, min(1.0, estimate.fill_probability))
            filled_notional += estimate.filled_notional_usd
            fees += estimate.fees_usd
            adverse_cost += estimate.adverse_selection_cost_usd
            if estimate.filled_shares > 0:
                filled_legs += 1
        all_filled = bool(leg_estimates) and filled_legs == len(leg_estimates)
        if filled_legs and not all_filled:
            orphan_notional = sum(estimate.filled_notional_usd for estimate in leg_estimates if estimate.filled_shares > 0)
        return MultiLegExecutionEstimate(
            leg_estimates=list(leg_estimates),
            all_filled=all_filled,
            joint_fill_probability=joint_probability,
            filled_notional_usd=filled_notional,
            fees_usd=fees,
            adverse_selection_cost_usd=adverse_cost,
            orphan_leg_notional_usd=orphan_notional,
            diagnostics={
                "leg_count": len(leg_estimates),
                "filled_leg_count": filled_legs,
                "partial_fill_count": sum(1 for estimate in leg_estimates if 0 < estimate.fill_ratio < 1),
            },
        )

    def _estimate_taker(
        self,
        *,
        order_book: Any,
        side: str,
        requested_shares: float,
        requested_notional: float,
        spend_budget_usd: float | None,
        limit_price: float | None,
        recent_trades: list[Any],
        book_age_ms: float | None,
        config: ExecutionEstimatorConfig,
    ) -> ExecutionEstimate:
        levels = self._levels(order_book, "asks" if side == "BUY" else "bids")
        if not levels:
            return self._empty_estimate(
                side=side,
                order_type="taker",
                reason="empty_opposite_book",
                requested_shares=requested_shares,
                requested_notional_usd=requested_notional,
                limit_price=limit_price,
                latency_ms=config.latency_ms,
            )

        depth_factor = self._book_depth_factor(book_age_ms, config)
        remaining = requested_shares
        remaining_budget = spend_budget_usd if side == "BUY" and spend_budget_usd is not None else None
        filled = 0.0
        notional = 0.0
        levels_consumed = 0
        worst_price = limit_price
        for price, displayed_size in levels:
            if side == "BUY" and worst_price is not None and price > worst_price:
                break
            if side == "SELL" and worst_price is not None and price < worst_price:
                break
            executable_size = max(0.0, displayed_size * depth_factor)
            if executable_size <= 0:
                continue
            if remaining_budget is not None:
                if remaining_budget <= 0:
                    break
                take = min(remaining, executable_size, remaining_budget / price)
            else:
                take = min(remaining, executable_size)
            if take <= 0:
                break
            filled += take
            take_notional = take * price
            notional += take_notional
            remaining -= take
            if remaining_budget is not None:
                remaining_budget -= take_notional
            levels_consumed += 1
            if remaining <= 0 or (remaining_budget is not None and remaining_budget <= 0):
                break

        if filled <= 0:
            return self._empty_estimate(
                side=side,
                order_type="taker",
                reason="limit_price_not_executable",
                requested_shares=requested_shares,
                requested_notional_usd=requested_notional,
                limit_price=limit_price,
                latency_ms=config.latency_ms,
                book_depth_factor=depth_factor,
            )

        average_price = notional / filled
        mid = self._mid_price(order_book) or average_price
        best_opposite = levels[0][0]
        slippage_bps = self._slippage_bps(side, average_price, mid)
        impact_bps = self._slippage_bps(side, average_price, best_opposite)
        adverse_bps = self._adverse_selection_bps(
            side=side,
            order_type="taker",
            order_book=order_book,
            recent_trades=recent_trades,
            config=config,
        )
        adverse_cost = filled * average_price * adverse_bps / 10_000.0
        economic_price = self._apply_adverse_price(side, average_price, adverse_bps)
        fees = notional * max(0.0, config.fee_bps) / 10_000.0
        if spend_budget_usd is not None and spend_budget_usd > 0:
            fill_ratio = notional / spend_budget_usd
        else:
            fill_ratio = filled / requested_shares if requested_shares > 0 else 0.0
        status = "filled" if fill_ratio >= 0.999999 else "partial"
        return ExecutionEstimate(
            side=side,
            order_type="taker",
            status=status,
            reason="opposite_book_consumed" if status == "filled" else "insufficient_executable_depth",
            requested_shares=requested_shares,
            filled_shares=filled,
            remaining_shares=max(0.0, requested_shares - filled),
            requested_notional_usd=requested_notional,
            filled_notional_usd=notional,
            average_price=average_price,
            economic_price=economic_price,
            limit_price=limit_price,
            fees_usd=fees,
            slippage_bps=slippage_bps,
            price_impact_bps=impact_bps,
            adverse_selection_bps=adverse_bps,
            adverse_selection_cost_usd=adverse_cost,
            latency_ms=config.latency_ms,
            fill_probability=min(1.0, max(0.0, fill_ratio)),
            queue_ahead_shares=0.0,
            levels_consumed=levels_consumed,
            book_depth_factor=depth_factor,
            diagnostics={
                "best_opposite_price": best_opposite,
                "mid_price": mid,
                "book_age_ms": book_age_ms,
                "recent_trade_stats": self._trade_stats(recent_trades, config.recent_trade_lookback_seconds),
            },
        )

    def _estimate_maker(
        self,
        *,
        order_book: Any,
        side: str,
        requested_shares: float,
        requested_notional: float,
        limit_price: float | None,
        recent_trades: list[Any],
        book_age_ms: float | None,
        config: ExecutionEstimatorConfig,
    ) -> ExecutionEstimate:
        if limit_price is None:
            return self._empty_estimate(
                side=side,
                order_type="maker",
                reason="maker_order_requires_limit_price",
                requested_shares=requested_shares,
                requested_notional_usd=requested_notional,
                limit_price=None,
                latency_ms=config.latency_ms,
            )

        # Queue-ahead computation: delegated to the shared utility so
        # backtest matcher + live forecasting can't drift apart.  The
        # diagnostic ``better_queue`` / ``same_level_size`` values are
        # still computed locally for the diagnostics dict consumers
        # below.
        own_levels = self._levels(order_book, "bids" if side == "BUY" else "asks")
        better_queue = 0.0
        same_level_size = 0.0
        for price, size in own_levels:
            if side == "BUY":
                if price > limit_price:
                    better_queue += size
                elif abs(price - limit_price) <= 1e-9:
                    same_level_size += size
            else:
                if price < limit_price:
                    better_queue += size
                elif abs(price - limit_price) <= 1e-9:
                    same_level_size += size
        depth_factor = self._book_depth_factor(book_age_ms, config)
        from services.optimization.queue_ahead import compute_queue_ahead_shares
        queue_ahead = compute_queue_ahead_shares(
            bids=order_book.get("bids") if isinstance(order_book, dict) else getattr(order_book, "bids", None),
            asks=order_book.get("asks") if isinstance(order_book, dict) else getattr(order_book, "asks", None),
            side=side,
            limit_price=limit_price,
            maker_queue_ahead_fraction=config.maker_queue_ahead_fraction,
            depth_factor=depth_factor,
        )
        stats = self._trade_stats(recent_trades, config.recent_trade_lookback_seconds)
        opposing_shares = stats["sell_shares"] if side == "BUY" else stats["buy_shares"]
        lookback = max(1.0, stats["lookback_seconds"])
        expected_flow = opposing_shares / lookback * max(0.0, config.time_in_force_seconds)
        expected_flow *= max(0.0, config.maker_trade_flow_multiplier)
        expected_flow *= depth_factor
        fillable_after_queue = max(0.0, expected_flow - queue_ahead)
        filled = min(requested_shares, fillable_after_queue)
        notional = filled * limit_price
        flow_ratio = expected_flow / (queue_ahead + requested_shares) if (queue_ahead + requested_shares) > 0 else 0.0
        fill_probability = min(1.0, max(0.0, flow_ratio))

        if filled <= 0:
            return self._empty_estimate(
                side=side,
                order_type="maker",
                reason="queue_not_reached_by_trade_flow",
                requested_shares=requested_shares,
                requested_notional_usd=requested_notional,
                limit_price=limit_price,
                latency_ms=config.latency_ms,
                fill_probability=fill_probability,
                queue_ahead_shares=queue_ahead,
                book_depth_factor=depth_factor,
                diagnostics={
                    "expected_opposing_flow_shares": expected_flow,
                    "same_level_size": same_level_size,
                    "better_queue_shares": better_queue,
                    "recent_trade_stats": stats,
                    "book_age_ms": book_age_ms,
                },
            )

        average_price = limit_price
        mid = self._mid_price(order_book) or average_price
        adverse_bps = self._adverse_selection_bps(
            side=side,
            order_type="maker",
            order_book=order_book,
            recent_trades=recent_trades,
            config=config,
        )
        adverse_cost = filled * average_price * adverse_bps / 10_000.0
        fees = notional * max(0.0, config.fee_bps) / 10_000.0
        fill_ratio = filled / requested_shares if requested_shares > 0 else 0.0
        status = "filled" if fill_ratio >= 0.999999 else "partial"
        return ExecutionEstimate(
            side=side,
            order_type="maker",
            status=status,
            reason="queue_reached_by_trade_flow" if status == "filled" else "partial_queue_fill",
            requested_shares=requested_shares,
            filled_shares=filled,
            remaining_shares=max(0.0, requested_shares - filled),
            requested_notional_usd=requested_notional,
            filled_notional_usd=notional,
            average_price=average_price,
            economic_price=self._apply_adverse_price(side, average_price, adverse_bps),
            limit_price=limit_price,
            fees_usd=fees,
            slippage_bps=self._slippage_bps(side, average_price, mid),
            price_impact_bps=0.0,
            adverse_selection_bps=adverse_bps,
            adverse_selection_cost_usd=adverse_cost,
            latency_ms=config.latency_ms,
            fill_probability=max(fill_probability, min(1.0, fill_ratio)),
            queue_ahead_shares=queue_ahead,
            levels_consumed=0,
            book_depth_factor=depth_factor,
            diagnostics={
                "expected_opposing_flow_shares": expected_flow,
                "same_level_size": same_level_size,
                "better_queue_shares": better_queue,
                "mid_price": mid,
                "recent_trade_stats": stats,
                "book_age_ms": book_age_ms,
            },
        )

    @staticmethod
    def _normalize_side(side: str) -> str:
        return "SELL" if str(side or "").strip().upper() == "SELL" else "BUY"

    @staticmethod
    def _normalize_price(value: float | None) -> float | None:
        if value is None:
            return None
        try:
            parsed = float(value)
        except Exception:
            return None
        if parsed <= 0 or parsed > 1.0:
            return None
        return max(_MIN_PRICE, min(_MAX_PRICE, parsed))

    @staticmethod
    def _coerce_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _levels(self, order_book: Any, side_name: str) -> list[tuple[float, float]]:
        raw = getattr(order_book, side_name, None)
        if raw is None and isinstance(order_book, dict):
            raw = order_book.get(side_name)
        rows: list[tuple[float, float]] = []
        for level in list(raw or []):
            if isinstance(level, dict):
                price = self._coerce_float(level.get("price"), 0.0)
                size = self._coerce_float(level.get("size"), 0.0)
            else:
                price = self._coerce_float(getattr(level, "price", 0.0), 0.0)
                size = self._coerce_float(getattr(level, "size", 0.0), 0.0)
            if price > 0 and size > 0:
                rows.append((max(_MIN_PRICE, min(_MAX_PRICE, price)), size))
        rows.sort(key=lambda row: row[0], reverse=side_name == "bids")
        return rows

    def _best_price(self, order_book: Any, side_name: str) -> float:
        levels = self._levels(order_book, "bids" if side_name == "bid" else "asks")
        return levels[0][0] if levels else 0.0

    def _mid_price(self, order_book: Any) -> float:
        best_bid = self._best_price(order_book, "bid")
        best_ask = self._best_price(order_book, "ask")
        if best_bid > 0 and best_ask > 0:
            return (best_bid + best_ask) / 2.0
        return best_bid or best_ask

    @staticmethod
    def _reference_price(
        *,
        side: str,
        limit_price: float | None,
        best_bid: float,
        best_ask: float,
        mid: float,
    ) -> float:
        if limit_price is not None and limit_price > 0:
            return limit_price
        if side == "BUY" and best_ask > 0:
            return best_ask
        if side == "SELL" and best_bid > 0:
            return best_bid
        return mid

    @staticmethod
    def _requested_shares(size_usd: float | None, size_shares: float | None, reference_price: float) -> float:
        shares = ExecutionEstimator._coerce_float(size_shares, 0.0)
        if shares > 0:
            return shares
        usd = ExecutionEstimator._coerce_float(size_usd, 0.0)
        if usd <= 0 or reference_price <= 0:
            return 0.0
        return usd / reference_price

    @staticmethod
    def _is_marketable(side: str, limit_price: float | None, best_bid: float, best_ask: float) -> bool:
        if limit_price is None:
            return True
        if side == "BUY":
            return best_ask > 0 and best_ask <= limit_price
        return best_bid > 0 and best_bid >= limit_price

    @staticmethod
    def _book_depth_factor(book_age_ms: float | None, config: ExecutionEstimatorConfig) -> float:
        base = max(0.0, min(1.0, config.displayed_depth_factor))
        if book_age_ms is None or config.max_book_age_ms <= 0 or config.stale_depth_decay <= 0:
            return max(config.min_depth_factor, base)
        age_ratio = max(0.0, float(book_age_ms)) / float(config.max_book_age_ms)
        stale_multiplier = max(config.min_depth_factor, 1.0 - min(1.0, age_ratio) * config.stale_depth_decay)
        return max(config.min_depth_factor, min(1.0, base * stale_multiplier))

    def _trade_stats(self, recent_trades: list[Any], lookback_seconds: float) -> dict[str, float]:
        if not recent_trades:
            return {
                "buy_shares": 0.0,
                "sell_shares": 0.0,
                "buy_notional": 0.0,
                "sell_notional": 0.0,
                "imbalance": 0.0,
                "trade_count": 0.0,
                "lookback_seconds": max(1.0, lookback_seconds),
            }
        latest_ts = max(self._coerce_float(getattr(trade, "timestamp", None), self._coerce_float(trade.get("timestamp") if isinstance(trade, dict) else None, 0.0)) for trade in recent_trades)
        cutoff = latest_ts - max(1.0, lookback_seconds) if latest_ts > 0 else None
        buy_shares = 0.0
        sell_shares = 0.0
        buy_notional = 0.0
        sell_notional = 0.0
        count = 0
        first_ts = latest_ts
        for trade in recent_trades:
            if isinstance(trade, dict):
                ts = self._coerce_float(trade.get("timestamp"), 0.0)
                price = self._coerce_float(trade.get("price"), 0.0)
                size = self._coerce_float(trade.get("size"), 0.0)
                side = str(trade.get("side") or "").strip().upper()
            else:
                ts = self._coerce_float(getattr(trade, "timestamp", 0.0), 0.0)
                price = self._coerce_float(getattr(trade, "price", 0.0), 0.0)
                size = self._coerce_float(getattr(trade, "size", 0.0), 0.0)
                side = str(getattr(trade, "side", "") or "").strip().upper()
            if cutoff is not None and ts and ts < cutoff:
                continue
            if price <= 0 or size <= 0:
                continue
            count += 1
            if ts > 0:
                first_ts = min(first_ts, ts)
            if side == "SELL":
                sell_shares += size
                sell_notional += size * price
            else:
                buy_shares += size
                buy_notional += size * price
        total_notional = buy_notional + sell_notional
        imbalance = (buy_notional - sell_notional) / total_notional if total_notional > 0 else 0.0
        observed_window = max(1.0, latest_ts - first_ts) if latest_ts > 0 and first_ts > 0 else max(1.0, lookback_seconds)
        return {
            "buy_shares": buy_shares,
            "sell_shares": sell_shares,
            "buy_notional": buy_notional,
            "sell_notional": sell_notional,
            "imbalance": imbalance,
            "trade_count": float(count),
            "lookback_seconds": observed_window,
        }

    def _adverse_selection_bps(
        self,
        *,
        side: str,
        order_type: str,
        order_book: Any,
        recent_trades: list[Any],
        config: ExecutionEstimatorConfig,
    ) -> float:
        stats = self._trade_stats(recent_trades, config.recent_trade_lookback_seconds)
        spread_bps = self._spread_bps(order_book)
        imbalance = stats["imbalance"]
        if order_type == "maker":
            pressure = -imbalance if side == "BUY" else imbalance
            multiplier = config.adverse_selection_multiplier * 1.35
        else:
            pressure = imbalance if side == "BUY" else -imbalance
            multiplier = config.adverse_selection_multiplier
        trade_count_component = min(1.0, stats["trade_count"] / 20.0)
        toxic_pressure = max(0.0, pressure)
        return max(0.0, toxic_pressure * max(10.0, spread_bps) * multiplier * (0.65 + 0.35 * trade_count_component))

    def _spread_bps(self, order_book: Any) -> float:
        bid = self._best_price(order_book, "bid")
        ask = self._best_price(order_book, "ask")
        if bid <= 0 or ask <= 0:
            return 0.0
        mid = (bid + ask) / 2.0
        if mid <= 0:
            return 0.0
        return max(0.0, (ask - bid) / mid * 10_000.0)

    @staticmethod
    def _slippage_bps(side: str, average_price: float, reference_price: float) -> float:
        if reference_price <= 0 or average_price <= 0:
            return 0.0
        if side == "BUY":
            return (average_price - reference_price) / reference_price * 10_000.0
        return (reference_price - average_price) / reference_price * 10_000.0

    @staticmethod
    def _apply_adverse_price(side: str, average_price: float, adverse_bps: float) -> float:
        adjustment = average_price * max(0.0, adverse_bps) / 10_000.0
        if side == "BUY":
            return max(_MIN_PRICE, min(_MAX_PRICE, average_price + adjustment))
        return max(_MIN_PRICE, min(_MAX_PRICE, average_price - adjustment))

    @staticmethod
    def _empty_estimate(
        *,
        side: str,
        order_type: str,
        reason: str,
        requested_shares: float,
        requested_notional_usd: float,
        limit_price: float | None,
        latency_ms: float,
        fill_probability: float = 0.0,
        queue_ahead_shares: float = 0.0,
        book_depth_factor: float = 0.0,
        diagnostics: dict[str, Any] | None = None,
    ) -> ExecutionEstimate:
        return ExecutionEstimate(
            side=side,
            order_type=order_type,
            status="unfilled",
            reason=reason,
            requested_shares=requested_shares,
            filled_shares=0.0,
            remaining_shares=requested_shares,
            requested_notional_usd=requested_notional_usd,
            filled_notional_usd=0.0,
            average_price=None,
            economic_price=None,
            limit_price=limit_price,
            fees_usd=0.0,
            slippage_bps=0.0,
            price_impact_bps=0.0,
            adverse_selection_bps=0.0,
            adverse_selection_cost_usd=0.0,
            latency_ms=latency_ms,
            fill_probability=max(0.0, min(1.0, fill_probability)),
            queue_ahead_shares=queue_ahead_shares,
            levels_consumed=0,
            book_depth_factor=book_depth_factor,
            diagnostics=dict(diagnostics or {}),
        )


execution_estimator = ExecutionEstimator()
