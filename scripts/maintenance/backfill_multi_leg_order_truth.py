from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import (  # noqa: E402
    AsyncSessionLocal,
    LiveTradingOrder,
    LiveTradingPosition,
    TradeSignal,
    TraderOrder,
)
from services.trader_orchestrator_state import (  # noqa: E402
    _build_live_order_authority_payload,
    _sync_order_runtime_payload,
    build_trader_order_row,
)
from utils.converters import safe_float  # noqa: E402
from utils.utcnow import utcnow  # noqa: E402


@dataclass(frozen=True)
class LegTruth:
    token_id: str
    market_id: str
    market_question: str
    direction: str
    outcome: str
    condition_id: str | None
    yes_token_id: str | None
    no_token_id: str | None


def _as_utc(value: datetime | None) -> datetime:
    if value is None:
        return utcnow()
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _extract_order_token_id(row: TraderOrder) -> str:
    payload = row.payload_json if isinstance(row.payload_json, dict) else {}
    token_id = str(payload.get("token_id") or "").strip()
    if token_id:
        return token_id
    reconciliation = payload.get("provider_reconciliation")
    if isinstance(reconciliation, dict):
        snapshot = reconciliation.get("snapshot")
        if isinstance(snapshot, dict):
            token_id = str(snapshot.get("asset_id") or snapshot.get("token_id") or "").strip()
            if token_id:
                return token_id
    return ""


def _normalize_market_text(value: Any) -> str:
    return str(value or "").strip()


def _build_leg_truths(signal: TradeSignal) -> dict[str, LegTruth]:
    payload = signal.payload_json if isinstance(signal.payload_json, dict) else {}
    markets = payload.get("markets")
    positions = payload.get("positions_to_take")
    execution_plan = payload.get("execution_plan")
    legs = execution_plan.get("legs") if isinstance(execution_plan, dict) else []

    market_lookup: dict[str, dict[str, Any]] = {}
    if isinstance(markets, list):
        for market in markets:
            if not isinstance(market, dict):
                continue
            token_ids = market.get("clob_token_ids")
            if not isinstance(token_ids, list):
                continue
            question = _normalize_market_text(market.get("question") or market.get("market_question"))
            condition_id = _normalize_market_text(market.get("condition_id")) or None
            yes_token_id = _normalize_market_text(token_ids[0]) if len(token_ids) > 0 else None
            no_token_id = _normalize_market_text(token_ids[1]) if len(token_ids) > 1 else None
            for token in token_ids:
                token_id = _normalize_market_text(token)
                if not token_id:
                    continue
                market_lookup[token_id] = {
                    "question": question,
                    "condition_id": condition_id,
                    "yes_token_id": yes_token_id,
                    "no_token_id": no_token_id,
                }

    position_lookup: dict[str, dict[str, Any]] = {}
    if isinstance(positions, list):
        for position in positions:
            if not isinstance(position, dict):
                continue
            token_id = _normalize_market_text(position.get("token_id"))
            if token_id:
                position_lookup[token_id] = position

    truths: dict[str, LegTruth] = {}
    if not isinstance(legs, list):
        return truths

    for leg in legs:
        if not isinstance(leg, dict):
            continue
        token_id = _normalize_market_text(leg.get("token_id"))
        if not token_id:
            continue
        market_info = market_lookup.get(token_id, {})
        position_info = position_lookup.get(token_id, {})
        market_id = _normalize_market_text(leg.get("market_id"))
        position_market = _normalize_market_text(position_info.get("market"))
        market_question = _normalize_market_text(leg.get("market_question"))
        if not market_question:
            market_question = _normalize_market_text(market_info.get("question")) or position_market
        if not market_id:
            market_id = market_question or position_market
        side = _normalize_market_text(leg.get("side")).lower() or "buy"
        outcome = _normalize_market_text(leg.get("outcome")).lower()
        direction = f"{side}_{outcome}" if side and outcome else side
        truths[token_id] = LegTruth(
            token_id=token_id,
            market_id=market_id or market_question,
            market_question=market_question or market_id,
            direction=direction,
            outcome=outcome,
            condition_id=market_info.get("condition_id"),
            yes_token_id=market_info.get("yes_token_id"),
            no_token_id=market_info.get("no_token_id"),
        )

    return truths


def _update_payload_truth(
    payload: dict[str, Any],
    truth: LegTruth,
    *,
    selected_price: float | None,
    preserve_live_market: bool,
) -> dict[str, Any]:
    next_payload = dict(payload)
    next_payload["market_id"] = truth.market_id
    next_payload["market_question"] = truth.market_question
    next_payload["direction"] = truth.direction
    next_payload["token_id"] = truth.token_id

    base_live_market = next_payload.get("live_market") if preserve_live_market else None
    live_market = dict(base_live_market) if isinstance(base_live_market, dict) else {}
    live_market["market_id"] = truth.market_id
    live_market["market_question"] = truth.market_question
    live_market["selected_token_id"] = truth.token_id
    if truth.outcome:
        live_market["selected_outcome"] = truth.outcome
    if truth.condition_id:
        live_market["condition_id"] = truth.condition_id
    if truth.yes_token_id:
        live_market["yes_token_id"] = truth.yes_token_id
    if truth.no_token_id:
        live_market["no_token_id"] = truth.no_token_id
    if selected_price is not None and selected_price > 0.0:
        live_market["live_selected_price"] = float(selected_price)
    next_payload["live_market"] = live_market
    return next_payload


def _select_template_order(rows: list[TraderOrder], target_created_at: datetime | None) -> TraderOrder | None:
    if not rows:
        return None
    target_ts = _as_utc(target_created_at)
    return min(rows, key=lambda row: abs((_as_utc(row.created_at) - target_ts).total_seconds()))


async def _run(apply_changes: bool) -> dict[str, Any]:
    async with AsyncSessionLocal() as session:
        signal_ids: set[str] = set()

        trader_signal_rows = (
            await session.execute(select(TraderOrder.signal_id).where(TraderOrder.signal_id.is_not(None)))
        ).scalars().all()
        signal_ids.update(str(signal_id or "").strip() for signal_id in trader_signal_rows if str(signal_id or "").strip())

        live_signal_rows = (
            await session.execute(select(LiveTradingOrder.opportunity_id).where(LiveTradingOrder.opportunity_id.is_not(None)))
        ).scalars().all()
        signal_ids.update(str(signal_id or "").strip() for signal_id in live_signal_rows if str(signal_id or "").strip())

        signals = list((await session.execute(select(TradeSignal).where(TradeSignal.id.in_(sorted(signal_ids))))).scalars().all())
        signals_by_id = {str(signal.id): signal for signal in signals}

        multi_signal_ids: list[str] = []
        leg_truths_by_signal: dict[str, dict[str, LegTruth]] = {}
        for signal_id, signal in signals_by_id.items():
            truths = _build_leg_truths(signal)
            if len(truths) <= 1:
                continue
            multi_signal_ids.append(signal_id)
            leg_truths_by_signal[signal_id] = truths

        if not multi_signal_ids:
            return {
                "signals_scanned": 0,
                "trader_orders_updated": 0,
                "live_orders_updated": 0,
                "trader_orders_created": 0,
                "missing_rows_skipped": 0,
                "applied": apply_changes,
            }

        trader_orders = list(
            (
                await session.execute(
                    select(TraderOrder)
                    .where(TraderOrder.signal_id.in_(multi_signal_ids))
                    .order_by(TraderOrder.created_at.asc(), TraderOrder.id.asc())
                )
            )
            .scalars()
            .all()
        )
        live_orders = list(
            (
                await session.execute(
                    select(LiveTradingOrder)
                    .where(LiveTradingOrder.opportunity_id.in_(multi_signal_ids))
                    .order_by(LiveTradingOrder.created_at.asc(), LiveTradingOrder.id.asc())
                )
            )
            .scalars()
            .all()
        )
        position_rows = list((await session.execute(select(LiveTradingPosition))).scalars().all())
        positions_by_token = {
            str(row.token_id or "").strip(): row for row in position_rows if str(row.token_id or "").strip()
        }

        orders_by_signal: dict[str, list[TraderOrder]] = {}
        orders_by_signal_token: dict[tuple[str, str], TraderOrder] = {}
        for row in trader_orders:
            signal_id = str(row.signal_id or "").strip()
            token_id = _extract_order_token_id(row)
            if not signal_id:
                continue
            orders_by_signal.setdefault(signal_id, []).append(row)
            if token_id:
                orders_by_signal_token[(signal_id, token_id)] = row

        live_orders_by_signal: dict[str, list[LiveTradingOrder]] = {}
        for row in live_orders:
            signal_id = str(row.opportunity_id or "").strip()
            if signal_id:
                live_orders_by_signal.setdefault(signal_id, []).append(row)

        now = utcnow()
        trader_orders_updated = 0
        live_orders_updated = 0
        trader_orders_created = 0
        missing_rows_skipped = 0

        for signal_id in multi_signal_ids:
            signal = signals_by_id[signal_id]
            truths = leg_truths_by_signal[signal_id]
            signal_orders = orders_by_signal.get(signal_id, [])

            for order in signal_orders:
                token_id = _extract_order_token_id(order)
                truth = truths.get(token_id)
                if truth is None:
                    continue

                payload = dict(order.payload_json or {})
                selected_price = safe_float(order.effective_price, safe_float(order.entry_price, None))
                next_payload = _update_payload_truth(payload, truth, selected_price=selected_price, preserve_live_market=True)

                current_question = _normalize_market_text(order.market_question)
                current_market_id = _normalize_market_text(order.market_id)
                changed = False

                if current_question != truth.market_question:
                    order.market_question = truth.market_question
                    changed = True
                if current_market_id in {"", current_question} and current_market_id != truth.market_id:
                    order.market_id = truth.market_id
                    changed = True
                if _normalize_market_text(order.direction).lower() != truth.direction:
                    order.direction = truth.direction
                    changed = True
                if next_payload != payload:
                    order.payload_json = _sync_order_runtime_payload(
                        payload=next_payload,
                        status=order.status,
                        now=now,
                    )
                    changed = True
                if changed:
                    order.updated_at = now
                    trader_orders_updated += 1

            for live_row in live_orders_by_signal.get(signal_id, []):
                token_id = _normalize_market_text(live_row.token_id)
                truth = truths.get(token_id)
                if truth is None:
                    continue
                if _normalize_market_text(live_row.market_question) == truth.market_question:
                    continue
                live_row.market_question = truth.market_question
                live_row.updated_at = now
                live_orders_updated += 1

            signal_orders = orders_by_signal.get(signal_id, [])
            signal_order_tokens = {_extract_order_token_id(row) for row in signal_orders}
            for live_row in live_orders_by_signal.get(signal_id, []):
                token_id = _normalize_market_text(live_row.token_id)
                if not token_id or token_id in signal_order_tokens:
                    continue
                truth = truths.get(token_id)
                if truth is None:
                    missing_rows_skipped += 1
                    continue

                sibling_orders = [row for row in signal_orders if _extract_order_token_id(row) != token_id]
                template = _select_template_order(sibling_orders, live_row.created_at)
                if template is None:
                    missing_rows_skipped += 1
                    continue

                authority_payload = _build_live_order_authority_payload(
                    live_row=live_row,
                    position_row=positions_by_token.get(token_id),
                    now=now,
                )
                selected_price = safe_float(
                    authority_payload.get("entry_price"),
                    safe_float(live_row.average_fill_price, safe_float(live_row.price, None)),
                )
                payload = _update_payload_truth(
                    authority_payload["payload_json"],
                    truth,
                    selected_price=selected_price,
                    preserve_live_market=False,
                )
                row = build_trader_order_row(
                    trader_id=str(template.trader_id),
                    signal=signal,
                    decision_id=template.decision_id,
                    strategy_key=template.strategy_key,
                    strategy_version=template.strategy_version,
                    mode=str(template.mode),
                    status=str(template.status),
                    notional_usd=float(authority_payload["order_notional_usd"]),
                    effective_price=selected_price,
                    reason=str(template.reason or "Recovered from live venue authority"),
                    payload=payload,
                    error_message=None,
                    trace_id=template.trace_id,
                    created_at=_as_utc(live_row.created_at),
                    market_id=truth.market_id,
                    market_question=truth.market_question,
                    direction=truth.direction,
                    entry_price=selected_price,
                    edge_percent=template.edge_percent,
                    confidence=template.confidence,
                )
                row.event_id = template.event_id
                session.add(row)
                signal_orders.append(row)
                signal_order_tokens.add(token_id)
                orders_by_signal_token[(signal_id, token_id)] = row
                trader_orders_created += 1

        if apply_changes:
            await session.commit()
        else:
            await session.rollback()

        return {
            "signals_scanned": len(multi_signal_ids),
            "trader_orders_updated": trader_orders_updated,
            "live_orders_updated": live_orders_updated,
            "trader_orders_created": trader_orders_created,
            "missing_rows_skipped": missing_rows_skipped,
            "applied": apply_changes,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair multi-leg live order truth for historical rows.")
    parser.add_argument("--apply", action="store_true", help="Persist changes instead of running a dry run.")
    args = parser.parse_args()
    summary = asyncio.run(_run(apply_changes=args.apply))
    print(summary)


if __name__ == "__main__":
    main()
