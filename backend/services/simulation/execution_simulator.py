from __future__ import annotations

import hashlib
import json
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
from services.simulation.historical_data_provider import HistoricalDataProvider
from services.strategy_loader import strategy_loader as strategy_db_loader
from services.trader_orchestrator.order_manager import submit_execution_leg


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

    @staticmethod
    def _normalize_direction(value: Any) -> str:
        normalized = str(value or "").strip().lower()
        if normalized in {"buy_no", "no"}:
            return "buy_no"
        if normalized in {"sell_yes", "short_yes"}:
            return "sell_yes"
        if normalized in {"sell_no", "short_no"}:
            return "sell_no"
        return "buy_yes"

    @staticmethod
    def _direction_to_outcome(direction: str) -> str:
        direction_key = str(direction or "").strip().lower()
        return "no" if "no" in direction_key else "yes"

    @staticmethod
    def _stable_hash(payload: Any) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    async def _fetch_points(
        self,
        *,
        provider: str,
        market_ref: str,
        start_ms: int | None,
        end_ms: int | None,
        timeframe: str,
        outcome: str,
    ) -> list[dict[str, Any]]:
        if provider == "kalshi":
            return await self._provider.get_kalshi_points(
                market_ticker=market_ref,
                start_ts=start_ms,
                end_ts=end_ms,
                timeframe=timeframe,
                outcome=outcome,
            )
        return await self._provider.get_polymarket_points(
            token_id=market_ref,
            start_ts=start_ms,
            end_ts=end_ms,
            timeframe=timeframe,
            outcome=outcome,
        )

    @staticmethod
    def _find_point_for_ts(points: list[dict[str, Any]], ts_ms: int | None) -> tuple[int, dict[str, Any]] | None:
        if not points:
            return None
        if ts_ms is None:
            return len(points) - 1, points[-1]
        candidate_idx = 0
        for idx, point in enumerate(points):
            point_ts = int(point.get("t") or 0)
            if point_ts <= ts_ms:
                candidate_idx = idx
            else:
                break
        return candidate_idx, points[candidate_idx]

    @classmethod
    def _history_summary_for_point(cls, points: list[dict[str, Any]], *, anchor_index: int) -> dict[str, Any]:
        if not points:
            return {}
        anchor_index = max(0, min(anchor_index, len(points) - 1))
        anchor = points[anchor_index]
        anchor_ts = int(anchor.get("t") or 0)
        anchor_price = max(0.0001, min(0.9999, cls._to_float(anchor.get("p"), 0.5)))

        def _move_percent(window_minutes: int) -> float:
            cutoff_ts = anchor_ts - (window_minutes * 60_000)
            baseline = anchor_price
            for idx in range(anchor_index, -1, -1):
                row_ts = int(points[idx].get("t") or 0)
                baseline = max(0.0001, min(0.9999, cls._to_float(points[idx].get("p"), baseline)))
                if row_ts <= cutoff_ts:
                    break
            return ((anchor_price - baseline) / baseline) * 100.0 if baseline > 0 else 0.0

        return {
            "move_5m": {"percent": _move_percent(5)},
            "move_30m": {"percent": _move_percent(30)},
            "move_2h": {"percent": _move_percent(120)},
        }

    @classmethod
    def _mark_to_market_close(
        cls,
        *,
        entry_price: float,
        quantity: float,
        mark_price: float,
        fee_bps: float,
    ) -> dict[str, float]:
        qty = max(0.0, cls._to_float(quantity, 0.0))
        entry = max(0.0001, min(0.9999, cls._to_float(entry_price, 0.5)))
        mark = max(0.0001, min(0.9999, cls._to_float(mark_price, entry)))
        notional = qty * mark
        fees = notional * (max(0.0, cls._to_float(fee_bps, 0.0)) / 10000.0)
        pnl = ((mark - entry) * qty) - fees
        return {
            "entry_price": entry,
            "exit_price": mark,
            "quantity": qty,
            "fees_usd": fees,
            "notional_usd": notional,
            "realized_pnl_usd": pnl,
        }

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
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
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
        input_strategy_params = dict(payload.get("strategy_params") or {})
        market_ref_override = str(payload.get("market_ref") or "").strip()
        market_id_filter = str(payload.get("market_id") or "").strip()
        requested_seed = str(payload.get("run_seed") or getattr(run_row, "run_seed", "") or "").strip()

        start_dt = self._to_utc(payload.get("start_at") or run_row.requested_start_at)
        end_dt = self._to_utc(payload.get("end_at") or run_row.requested_end_at)
        start_ms = self._to_epoch_ms(start_dt)
        end_ms = self._to_epoch_ms(end_dt)
        run_seed = (
            requested_seed
            or hashlib.sha256(
                f"{run_row.id}|{strategy_key}|{source_key}|{start_ms}|{end_ms}".encode("utf-8")
            ).hexdigest()[:16]
        )
        fee_bps = max(0.0, self._to_float(payload.get("fee_bps"), 200.0))
        fee_rate = fee_bps / 10000.0

        default_notional = max(1.0, self._to_float(payload.get("default_notional_usd"), 50.0))
        config_payload = {
            "strategy_key": strategy_key,
            "source_key": source_key,
            "market_provider": provider,
            "market_ref": market_ref_override or None,
            "market_id_filter": market_id_filter or None,
            "timeframe": timeframe,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "strategy_params": input_strategy_params,
            "market_scope": dict(payload.get("market_scope") or {}),
            "default_notional_usd": default_notional,
            "slippage_bps": self._to_float(payload.get("slippage_bps"), 5.0),
            "fee_bps": fee_bps,
        }
        config_hash = self._stable_hash(config_payload)

        run_row.status = "running"
        run_row.started_at = datetime.now(timezone.utc).replace(tzinfo=None)
        run_row.error_message = None
        run_row.run_seed = run_seed
        run_row.config_hash = config_hash
        run_row.dataset_hash = None
        run_row.code_sha = None
        run_row.params_json = {
            **dict(payload),
            "run_manifest": {
                "run_seed": run_seed,
                "config_hash": config_hash,
                "dataset_hash": None,
                "code_sha": None,
            },
        }
        await session.flush()

        reload_result = await strategy_db_loader.reload_from_db(strategy_key, session=session)
        strategy = strategy_db_loader.get_instance(strategy_key)
        runtime_status = strategy_db_loader.get_runtime_status(strategy_key) or {}
        code_sha = str(runtime_status.get("source_hash") or reload_result.get("source_hash") or "").strip()
        if strategy is None:
            reason = f"strategy_unavailable:{strategy_key}"
            dataset_hash = self._stable_hash(
                {
                    "provider": provider,
                    "source_key": source_key,
                    "strategy_key": strategy_key,
                    "reason": reason,
                }
            )
            run_manifest = {
                "run_seed": run_seed,
                "dataset_hash": dataset_hash,
                "config_hash": config_hash,
                "code_sha": code_sha or None,
            }
            run_row.status = "failed"
            run_row.error_message = reason
            run_row.dataset_hash = dataset_hash
            run_row.code_sha = code_sha or None
            run_row.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
            run_row.summary_json = {
                "status": "failed",
                "reason": reason,
                "reload": reload_result,
                "run_manifest": run_manifest,
            }
            run_row.params_json = {
                **dict(payload),
                "run_manifest": run_manifest,
            }
            await session.flush()
            return dict(run_row.summary_json)
        run_row.code_sha = code_sha or None

        strategy_defaults: dict[str, Any] = {}
        loaded_strategy_config = getattr(strategy, "config", None)
        if isinstance(loaded_strategy_config, dict):
            strategy_defaults = dict(loaded_strategy_config)
        else:
            loaded_default_config = getattr(strategy, "default_config", None)
            if isinstance(loaded_default_config, dict):
                strategy_defaults = dict(loaded_default_config)
        strategy_params = {**strategy_defaults, **input_strategy_params}

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
        point_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
        point_series_hashes: dict[str, str] = {}
        emission_fingerprints: list[dict[str, Any]] = []

        for emission in emissions:
            if str(emission.status or "").strip().lower() not in {"pending", "selected", "submitted", "executed"}:
                continue
            signal_payload = dict(emission.payload_json or {})
            emission_ts = self._to_utc(emission.created_at)
            emission_ts_ms = self._to_epoch_ms(emission_ts)
            direction = self._normalize_direction(emission.direction or signal_payload.get("direction") or "buy_yes")
            outcome = self._direction_to_outcome(direction)
            emission_fingerprints.append(
                {
                    "id": str(emission.id),
                    "signal_id": str(emission.signal_id or ""),
                    "market_id": str(emission.market_id or ""),
                    "direction": direction,
                    "entry_price": self._to_float(emission.entry_price, 0.0),
                    "status": str(emission.status or ""),
                    "created_at_ms": emission_ts_ms,
                }
            )
            signal_obj = SimpleNamespace(
                id=str(emission.signal_id or ""),
                source=str(emission.source or ""),
                signal_type=str(emission.signal_type or ""),
                market_id=str(emission.market_id or ""),
                market_question=(
                    signal_payload.get("market_question") or signal_payload.get("question") or emission.market_id
                ),
                direction=direction,
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
                event_at=(emission.created_at or datetime.now(timezone.utc)).replace(tzinfo=timezone.utc),
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

            cache_key = (market_ref, outcome)
            points = point_cache.get(cache_key)
            if points is None:
                points = await self._fetch_points(
                    provider=provider,
                    market_ref=market_ref,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    timeframe=timeframe,
                    outcome=outcome,
                )
                point_cache[cache_key] = points
                point_series_hashes[f"{market_ref}:{outcome}"] = self._stable_hash(
                    [{"t": int(row.get("t") or 0), "p": round(self._to_float(row.get("p"), 0.0), 6)} for row in points]
                )

            point_match = self._find_point_for_ts(points, emission_ts_ms)
            if point_match is None:
                skipped += 1
                continue
            point_idx, replay_point = point_match

            replay_price = max(0.0001, min(0.9999, self._to_float(replay_point.get("p"), 0.5)))
            target_price = max(0.0001, min(0.9999, self._to_float(emission.entry_price, replay_price)))
            entry_delta_pct = ((replay_price - target_price) / target_price * 100.0) if target_price > 0 else 0.0
            history_summary = self._history_summary_for_point(points, anchor_index=point_idx)
            live_context = {
                "live_selected_price": replay_price,
                "entry_price_delta_pct": entry_delta_pct,
                "history_summary": history_summary,
                "liquidity": self._to_float(signal_payload.get("liquidity"), self._to_float(emission.liquidity, 0.0)),
            }
            execution_payload = dict(signal_payload)
            execution_payload["live_market"] = live_context
            execution_signal = SimpleNamespace(
                id=str(signal_obj.id or emission.id),
                source=signal_obj.source,
                signal_type=signal_obj.signal_type,
                market_id=market_ref,
                market_question=signal_obj.market_question,
                direction=direction,
                entry_price=target_price,
                edge_percent=signal_obj.edge_percent,
                confidence=signal_obj.confidence,
                payload_json=execution_payload,
                live_context=live_context,
                liquidity=live_context.get("liquidity"),
            )
            leg = {
                "leg_id": f"leg_{str(emission.signal_id or emission.id)}",
                "market_id": market_ref,
                "market_question": signal_obj.market_question,
                "side": "buy",
                "outcome": outcome,
                "limit_price": target_price,
                "price_policy": "maker_limit",
                "time_in_force": "GTC",
            }
            submission = await submit_execution_leg(
                mode="paper",
                signal=execution_signal,
                leg=leg,
                notional_usd=notional_usd,
            )
            normalized_status = str(submission.status or "").strip().lower()
            paper_payload = submission.payload.get("paper_simulation") if isinstance(submission.payload, dict) else {}
            slippage_bps = self._to_float((paper_payload or {}).get("slippage_bps"), 0.0)
            if normalized_status not in {"executed", "open", "submitted"}:
                skipped += 1
                sequence = await self._append_event(
                    session,
                    run_id=run_row.id,
                    sequence=sequence,
                    event_type="order_unfilled",
                    event_at=datetime.fromtimestamp(int(replay_point.get("t") or 0) / 1000, tz=timezone.utc),
                    signal_id=emission.signal_id,
                    market_id=emission.market_id,
                    direction=direction,
                    price=target_price,
                    quantity=0.0,
                    notional_usd=notional_usd,
                    fees_usd=0.0,
                    slippage_bps=slippage_bps,
                    realized_pnl_usd=0.0,
                    unrealized_pnl_usd=0.0,
                    payload={
                        "reason": str(submission.error_message or "paper_execution_unfilled"),
                        "status": normalized_status,
                        "execution_payload": submission.payload,
                        "replay_point": replay_point,
                    },
                )
                continue

            filled += 1
            fill_price = max(0.0001, min(0.9999, self._to_float(submission.effective_price, target_price)))
            filled_notional = max(0.0, self._to_float(submission.notional_usd, notional_usd))
            quantity = max(
                0.0, self._to_float(submission.shares, filled_notional / fill_price if fill_price > 0 else 0.0)
            )
            entry_fees = filled_notional * fee_rate
            total_fees += entry_fees
            open_positions.append(
                {
                    "signal_id": emission.signal_id,
                    "market_id": emission.market_id,
                    "market_ref": market_ref,
                    "direction": direction,
                    "outcome": outcome,
                    "entry_price": fill_price,
                    "quantity": quantity,
                    "entry_notional_usd": filled_notional,
                    "opened_at_ms": int(replay_point.get("t") or 0),
                }
            )
            sequence = await self._append_event(
                session,
                run_id=run_row.id,
                sequence=sequence,
                event_type="order_filled",
                event_at=datetime.fromtimestamp(int(replay_point.get("t") or 0) / 1000, tz=timezone.utc),
                signal_id=emission.signal_id,
                market_id=emission.market_id,
                direction=direction,
                price=fill_price,
                quantity=quantity,
                notional_usd=filled_notional,
                fees_usd=entry_fees,
                slippage_bps=slippage_bps,
                realized_pnl_usd=None,
                unrealized_pnl_usd=None,
                payload={
                    "target_price": target_price,
                    "execution_payload": submission.payload,
                    "replay_point": replay_point,
                },
            )

        for position in open_positions:
            market_ref = str(position.get("market_ref") or "")
            direction = self._normalize_direction(position.get("direction") or "buy_yes")
            outcome = str(position.get("outcome") or self._direction_to_outcome(direction))
            points = point_cache.get((market_ref, outcome), [])
            if not points:
                continue
            final_point = points[-1]
            final_price = self._to_float(final_point.get("p"), self._to_float(position.get("entry_price"), 0.5))
            close = self._mark_to_market_close(
                entry_price=self._to_float(position.get("entry_price"), 0.5),
                quantity=self._to_float(position.get("quantity"), 0.0),
                mark_price=final_price,
                fee_bps=fee_bps,
            )
            total_fees += float(close.get("fees_usd") or 0.0)
            total_realized += float(close.get("realized_pnl_usd") or 0.0)
            sequence = await self._append_event(
                session,
                run_id=run_row.id,
                sequence=sequence,
                event_type="position_mtm_close",
                event_at=datetime.fromtimestamp(int(final_point.get("t") or 0) / 1000, tz=timezone.utc),
                signal_id=position.get("signal_id"),
                market_id=position.get("market_id"),
                direction=direction,
                price=float(close.get("exit_price") or final_price),
                quantity=float(close.get("quantity") or 0.0),
                notional_usd=float(close.get("notional_usd") or 0.0),
                fees_usd=float(close.get("fees_usd") or 0.0),
                slippage_bps=0.0,
                realized_pnl_usd=float(close.get("realized_pnl_usd") or 0.0),
                unrealized_pnl_usd=0.0,
                payload={
                    "entry_price": close.get("entry_price"),
                    "exit_price": close.get("exit_price"),
                    "replay_point": final_point,
                },
            )

        dataset_hash = self._stable_hash(
            {
                "provider": provider,
                "timeframe": timeframe,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "emissions": emission_fingerprints,
                "point_series": point_series_hashes,
            }
        )
        run_manifest = {
            "run_seed": run_seed,
            "dataset_hash": dataset_hash,
            "config_hash": config_hash,
            "code_sha": code_sha or None,
        }
        summary = {
            "status": "completed",
            "strategy_key": strategy_key,
            "source_key": source_key,
            "provider": provider,
            "replay_granularity": "points",
            "timeframe": timeframe,
            "emissions_sampled": len(emissions),
            "signals_evaluated": evaluated,
            "signals_selected": selected,
            "orders_filled": filled,
            "orders_skipped": skipped,
            "total_fees_usd": total_fees,
            "total_realized_pnl_usd": total_realized,
            "event_count": sequence - 1,
            "replay_series_count": len(point_series_hashes),
            "run_manifest": run_manifest,
        }

        run_row.status = "completed"
        run_row.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
        run_row.summary_json = summary
        run_row.dataset_hash = dataset_hash
        run_row.config_hash = config_hash
        run_row.run_seed = run_seed
        run_row.code_sha = code_sha or None
        run_row.params_json = {
            **dict(payload),
            "strategy_params": input_strategy_params,
            "effective_strategy_params": strategy_params,
            "run_manifest": run_manifest,
        }
        run_row.error_message = None
        await session.flush()
        return summary


execution_simulator = ExecutionSimulator()
