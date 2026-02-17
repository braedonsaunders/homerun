from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    ExecutionSimEvent,
    ExecutionSimRun,
    TradeSignalEmission,
)
from services.simulation.fill_models import FillConfig, FillModel
from services.simulation.historical_data_provider import HistoricalDataProvider
from services.strategy_loader import strategy_loader as strategy_db_loader


class ExecutionSimulator:
    """Replay trader strategy execution against historical signal and market data."""

    def __init__(self) -> None:
        self._provider = HistoricalDataProvider()

    @staticmethod
    def _to_utc(dt_value: Any) -> datetime | None:
        if dt_value is None:
            return None
        if isinstance(dt_value, datetime):
            if dt_value.tzinfo is None:
                return dt_value.replace(tzinfo=timezone.utc)
            return dt_value.astimezone(timezone.utc)
        text = str(dt_value).strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            return None

    @staticmethod
    def _to_epoch_ms(dt_value: datetime | None) -> int | None:
        if dt_value is None:
            return None
        if dt_value.tzinfo is None:
            dt_value = dt_value.replace(tzinfo=timezone.utc)
        return int(dt_value.timestamp() * 1000)

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _normalize_provider(value: Any) -> str:
        provider = str(value or "polymarket").strip().lower()
        return provider if provider in {"polymarket", "kalshi"} else "polymarket"

    async def _fetch_candles(
        self,
        *,
        provider: str,
        market_ref: str,
        start_ms: int | None,
        end_ms: int | None,
        timeframe: str,
        direction: str,
    ) -> list[dict[str, Any]]:
        if provider == "kalshi":
            return await self._provider.get_kalshi_candles(
                market_ticker=market_ref,
                start_ts=start_ms,
                end_ts=end_ms,
                timeframe=timeframe,
                outcome=direction,
            )
        return await self._provider.get_polymarket_candles(
            token_id=market_ref,
            start_ts=start_ms,
            end_ts=end_ms,
            timeframe=timeframe,
            outcome=direction,
        )

    @staticmethod
    def _find_candle_for_ts(candles: list[dict[str, Any]], ts_ms: int | None) -> dict[str, Any] | None:
        if not candles:
            return None
        if ts_ms is None:
            return candles[-1]
        candidate = None
        for candle in candles:
            candle_ts = int(candle.get("t") or 0)
            if candle_ts <= ts_ms:
                candidate = candle
            else:
                break
        return candidate or candles[0]

    async def _append_event(
        self,
        session: AsyncSession,
        *,
        run_id: str,
        sequence: int,
        event_type: str,
        event_at: datetime,
        signal_id: str | None,
        market_id: str | None,
        direction: str | None,
        price: float | None,
        quantity: float | None,
        notional_usd: float | None,
        fees_usd: float | None,
        slippage_bps: float | None,
        realized_pnl_usd: float | None,
        unrealized_pnl_usd: float | None,
        payload: dict[str, Any],
    ) -> int:
        session.add(
            ExecutionSimEvent(
                id=uuid.uuid4().hex,
                run_id=run_id,
                sequence=sequence,
                event_type=event_type,
                event_at=event_at.replace(tzinfo=None),
                signal_id=signal_id,
                market_id=market_id,
                direction=direction,
                price=price,
                quantity=quantity,
                notional_usd=notional_usd,
                fees_usd=fees_usd,
                slippage_bps=slippage_bps,
                realized_pnl_usd=realized_pnl_usd,
                unrealized_pnl_usd=unrealized_pnl_usd,
                payload_json=payload,
                created_at=datetime.utcnow(),
            )
        )
        return sequence + 1

    async def run(
        self,
        session: AsyncSession,
        *,
        run_row: ExecutionSimRun,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        strategy_key = str(payload.get("strategy_key") or run_row.strategy_key or "").strip().lower()
        source_key = str(payload.get("source_key") or run_row.source_key or "").strip().lower()
        provider = self._normalize_provider(payload.get("market_provider"))
        timeframe = str(payload.get("timeframe") or "15m").strip().lower()
        strategy_params = dict(payload.get("strategy_params") or {})
        market_ref_override = str(payload.get("market_ref") or "").strip()
        market_id_filter = str(payload.get("market_id") or "").strip()

        start_dt = self._to_utc(payload.get("start_at") or run_row.requested_start_at)
        end_dt = self._to_utc(payload.get("end_at") or run_row.requested_end_at)
        start_ms = self._to_epoch_ms(start_dt)
        end_ms = self._to_epoch_ms(end_dt)

        fill_config = FillConfig(
            slippage_bps=self._to_float(payload.get("slippage_bps"), 5.0),
            fee_bps=self._to_float(payload.get("fee_bps"), 200.0),
        )
        default_notional = max(1.0, self._to_float(payload.get("default_notional_usd"), 50.0))

        run_row.status = "running"
        run_row.started_at = datetime.utcnow()
        run_row.error_message = None
        run_row.params_json = dict(payload)
        await session.flush()

        reload_result = await strategy_db_loader.reload_from_db(strategy_key, session=session)
        strategy = strategy_db_loader.get_instance(strategy_key)
        if strategy is None:
            reason = f"strategy_unavailable:{strategy_key}"
            run_row.status = "failed"
            run_row.error_message = reason
            run_row.finished_at = datetime.utcnow()
            run_row.summary_json = {
                "status": "failed",
                "reason": reason,
                "reload": reload_result,
            }
            await session.flush()
            return dict(run_row.summary_json)

        query = select(TradeSignalEmission).where(TradeSignalEmission.source == source_key)
        if start_dt is not None:
            query = query.where(TradeSignalEmission.created_at >= start_dt.replace(tzinfo=None))
        if end_dt is not None:
            query = query.where(TradeSignalEmission.created_at <= end_dt.replace(tzinfo=None))
        if market_id_filter:
            query = query.where(TradeSignalEmission.market_id == market_id_filter)

        query = query.order_by(TradeSignalEmission.created_at.asc(), TradeSignalEmission.id.asc())
        emissions = list((await session.execute(query)).scalars().all())

        sequence = 1
        evaluated = 0
        selected = 0
        filled = 0
        skipped = 0
        total_fees = 0.0
        total_realized = 0.0
        open_positions: list[dict[str, Any]] = []
        candle_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}

        for emission in emissions:
            if str(emission.status or "").strip().lower() not in {"pending", "selected", "submitted", "executed"}:
                continue
            signal_payload = dict(emission.payload_json or {})
            signal_obj = SimpleNamespace(
                id=str(emission.signal_id or ""),
                source=str(emission.source or ""),
                signal_type=str(emission.signal_type or ""),
                market_id=str(emission.market_id or ""),
                market_question=(
                    signal_payload.get("market_question") or signal_payload.get("question") or emission.market_id
                ),
                direction=str(emission.direction or signal_payload.get("direction") or "buy_yes"),
                entry_price=emission.entry_price,
                edge_percent=emission.edge_percent,
                confidence=emission.confidence,
                payload_json=signal_payload,
            )

            decision = strategy.evaluate(
                signal_obj,
                {
                    "params": strategy_params,
                    "mode": "execution_simulation",
                    "source_config": {
                        "source_key": source_key,
                        "strategy_key": strategy_key,
                        "strategy_params": strategy_params,
                    },
                },
            )
            evaluated += 1
            decision_payload = {
                "decision": decision.decision,
                "reason": decision.reason,
                "score": decision.score,
                "size_usd": decision.size_usd,
                "checks": [
                    {
                        "key": check.key,
                        "passed": check.passed,
                        "score": check.score,
                        "detail": check.detail,
                    }
                    for check in (decision.checks or [])
                ],
            }
            sequence = await self._append_event(
                session,
                run_id=run_row.id,
                sequence=sequence,
                event_type="signal_evaluated",
                event_at=(emission.created_at or datetime.utcnow()).replace(tzinfo=timezone.utc),
                signal_id=emission.signal_id,
                market_id=emission.market_id,
                direction=signal_obj.direction,
                price=emission.entry_price,
                quantity=None,
                notional_usd=None,
                fees_usd=None,
                slippage_bps=None,
                realized_pnl_usd=None,
                unrealized_pnl_usd=None,
                payload=decision_payload,
            )

            if decision.decision != "selected":
                skipped += 1
                continue

            selected += 1
            notional_usd = max(1.0, self._to_float(decision.size_usd, default_notional))
            market_ref = market_ref_override or str(emission.market_id or "")
            if not market_ref:
                skipped += 1
                continue

            direction = str(signal_obj.direction or "buy_yes")
            cache_key = (market_ref, direction)
            candles = candle_cache.get(cache_key)
            if candles is None:
                candles = await self._fetch_candles(
                    provider=provider,
                    market_ref=market_ref,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    timeframe=timeframe,
                    direction=direction,
                )
                candle_cache[cache_key] = candles

            candle = self._find_candle_for_ts(candles, self._to_epoch_ms(self._to_utc(emission.created_at)))
            if candle is None:
                skipped += 1
                continue

            target_price = self._to_float(emission.entry_price, self._to_float(candle.get("close"), 0.5))
            fill = FillModel.intrabar_touch_fill(
                direction=direction,
                target_price=target_price,
                notional_usd=notional_usd,
                candle=candle,
                config=fill_config,
            )
            if not fill.get("filled"):
                skipped += 1
                sequence = await self._append_event(
                    session,
                    run_id=run_row.id,
                    sequence=sequence,
                    event_type="order_unfilled",
                    event_at=datetime.fromtimestamp(int(candle.get("t") or 0) / 1000, tz=timezone.utc),
                    signal_id=emission.signal_id,
                    market_id=emission.market_id,
                    direction=direction,
                    price=target_price,
                    quantity=0.0,
                    notional_usd=notional_usd,
                    fees_usd=0.0,
                    slippage_bps=float(fill_config.slippage_bps),
                    realized_pnl_usd=0.0,
                    unrealized_pnl_usd=0.0,
                    payload={"reason": "intrabar_touch_not_reached"},
                )
                continue

            filled += 1
            fees_usd = float(fill.get("fees_usd") or 0.0)
            total_fees += fees_usd
            fill_price = float(fill.get("fill_price") or target_price)
            quantity = float(fill.get("quantity") or 0.0)
            open_positions.append(
                {
                    "signal_id": emission.signal_id,
                    "market_id": emission.market_id,
                    "market_ref": market_ref,
                    "direction": direction,
                    "entry_price": fill_price,
                    "quantity": quantity,
                    "fees_usd": fees_usd,
                    "opened_at_ms": int(fill.get("event_ts_ms") or candle.get("t") or 0),
                }
            )
            sequence = await self._append_event(
                session,
                run_id=run_row.id,
                sequence=sequence,
                event_type="order_filled",
                event_at=datetime.fromtimestamp(
                    int(fill.get("event_ts_ms") or candle.get("t") or 0) / 1000, tz=timezone.utc
                ),
                signal_id=emission.signal_id,
                market_id=emission.market_id,
                direction=direction,
                price=fill_price,
                quantity=quantity,
                notional_usd=notional_usd,
                fees_usd=fees_usd,
                slippage_bps=float(fill.get("slippage_bps") or fill_config.slippage_bps),
                realized_pnl_usd=None,
                unrealized_pnl_usd=None,
                payload={"target_price": target_price},
            )

        for position in open_positions:
            market_ref = str(position.get("market_ref") or "")
            direction = str(position.get("direction") or "buy_yes")
            candles = candle_cache.get((market_ref, direction), [])
            if not candles:
                continue
            final_candle = candles[-1]
            final_price = self._to_float(final_candle.get("close"), self._to_float(position.get("entry_price"), 0.5))
            close = FillModel.mark_to_market_close(
                direction=direction,
                entry_price=self._to_float(position.get("entry_price"), 0.5),
                quantity=self._to_float(position.get("quantity"), 0.0),
                last_price=final_price,
                config=fill_config,
            )
            total_realized += float(close.get("realized_pnl_usd") or 0.0)
            sequence = await self._append_event(
                session,
                run_id=run_row.id,
                sequence=sequence,
                event_type="position_mtm_close",
                event_at=datetime.fromtimestamp(int(final_candle.get("t") or 0) / 1000, tz=timezone.utc),
                signal_id=position.get("signal_id"),
                market_id=position.get("market_id"),
                direction=direction,
                price=float(close.get("exit_price") or final_price),
                quantity=float(close.get("quantity") or 0.0),
                notional_usd=float(close.get("quantity") or 0.0) * float(close.get("entry_price") or 0.0),
                fees_usd=float(close.get("fees_usd") or 0.0),
                slippage_bps=float(fill_config.slippage_bps),
                realized_pnl_usd=float(close.get("realized_pnl_usd") or 0.0),
                unrealized_pnl_usd=0.0,
                payload={
                    "entry_price": close.get("entry_price"),
                    "exit_price": close.get("exit_price"),
                },
            )

        summary = {
            "status": "completed",
            "strategy_key": strategy_key,
            "source_key": source_key,
            "provider": provider,
            "timeframe": timeframe,
            "emissions_sampled": len(emissions),
            "signals_evaluated": evaluated,
            "signals_selected": selected,
            "orders_filled": filled,
            "orders_skipped": skipped,
            "total_fees_usd": total_fees,
            "total_realized_pnl_usd": total_realized,
            "event_count": sequence - 1,
        }

        run_row.status = "completed"
        run_row.finished_at = datetime.utcnow()
        run_row.summary_json = summary
        run_row.error_message = None
        await session.flush()
        return summary


execution_simulator = ExecutionSimulator()
