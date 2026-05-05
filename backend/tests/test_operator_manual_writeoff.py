"""Tests for the operator manual-writeoff path.

This is the institutional escape hatch — the only code path that can
write a non-NULL ``actual_profit`` outside of the verifier's
wallet-activity match.  Every rule in ``manual_writeoff_order`` is a
financial-correctness invariant; we test each in isolation.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import select

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import (
    Base,
    Trader,
    TraderOrder,
    TraderOrderVerificationEvent,
)
from services.operator_writeoff import (
    ManualWriteoffRejected,
    ManualWriteoffReversalRejected,
    WalletAbsentResolutionReversalRejected,
    manual_writeoff_order,
    reverse_manual_writeoff_order,
    reverse_wallet_absent_resolution,
)
from tests.postgres_test_db import build_postgres_session_factory
from utils.utcnow import utcnow


async def _seed(
    session,
    trader_id: str,
    order_id: str,
    *,
    pending_exit_status: str = "blocked_persistent_timeout",
    lifecycle_status: str = "executed",
    # Default to a real venue-rejection error so the new
    # venue-rejection gate doesn't reject every pre-existing test.
    # Tests that explicitly want to exercise the gate override this.
    last_error: str = "Order rejected by CLOB: orderbook does not exist for this market",
):
    session.add(
        Trader(
            id=trader_id,
            name=f"trader-{trader_id}",
            mode="live",
            source_configs_json=[],
            risk_limits_json={},
        )
    )
    now = utcnow()
    pending_exit_payload: dict = {
        "status": pending_exit_status,
        "consecutive_blocked_failure_count": 3,
    }
    if last_error is not None:
        pending_exit_payload["last_error"] = last_error
    session.add(
        TraderOrder(
            id=order_id,
            trader_id=trader_id,
            source="scanner",
            market_id="market-1",
            direction="buy_yes",
            mode="live",
            status=lifecycle_status,
            notional_usd=7.56,
            entry_price=0.87,
            effective_price=0.87,
            payload_json={
                "selected_token_id": "token-1",
                "pending_live_exit": pending_exit_payload,
            },
            created_at=now - timedelta(hours=24),
            updated_at=now - timedelta(minutes=1),
            executed_at=now - timedelta(hours=24),
        )
    )
    await session.commit()


@pytest.mark.asyncio
async def test_manual_writeoff_happy_path_writes_pnl_and_audit_event(tmp_path):
    """A valid request should: set actual_profit, transition status,
    emit a TraderOrderVerificationEvent with operator + reason."""
    engine, session_factory = await build_postgres_session_factory(
        Base, "manual_writeoff_happy"
    )
    try:
        async with session_factory() as session:
            await _seed(session, "trader-1", "order-happy")

        async with session_factory() as session:
            result = await manual_writeoff_order(
                session,
                order_id="order-happy",
                realized_pnl=-7.56,  # full loss
                reason="market unresolved 30 days post-end-date; assuming total loss",
                operator_id="ops@homerun",
            )
            await session.commit()

        async with session_factory() as session:
            row = await session.get(TraderOrder, "order-happy")
            assert row is not None
            assert row.actual_profit == pytest.approx(-7.56)
            assert row.status == "closed_loss"
            assert row.verification_status == "manual_writeoff"
            assert row.verification_source == "operator"
            payload = row.payload_json or {}
            close = payload.get("position_close") or {}
            assert close.get("close_trigger") == "manual_writeoff"
            assert close.get("operator_id") == "ops@homerun"
            assert close.get("realized_pnl") == pytest.approx(-7.56)

            events = (
                await session.execute(
                    select(TraderOrderVerificationEvent).where(
                        TraderOrderVerificationEvent.trader_order_id == "order-happy"
                    )
                )
            ).scalars().all()
            assert any(e.event_type == "manual_writeoff" for e in events)
            evt = next(e for e in events if e.event_type == "manual_writeoff")
            assert evt.source == "operator"
            assert "market unresolved" in (evt.reason or "")
            assert evt.payload_json.get("operator_id") == "ops@homerun"
            assert evt.payload_json.get("realized_pnl") == pytest.approx(-7.56)

        assert result["next_status"] == "closed_loss"
        assert result["operator_id"] == "ops@homerun"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_manual_writeoff_rejects_already_closed_order(tmp_path):
    """An already-closed order cannot be manually written off — that
    would let an operator overwrite verified P&L."""
    engine, session_factory = await build_postgres_session_factory(
        Base, "manual_writeoff_rejects_closed"
    )
    try:
        async with session_factory() as session:
            await _seed(session, "trader-c", "order-closed", lifecycle_status="closed_win")

        async with session_factory() as session:
            with pytest.raises(ManualWriteoffRejected, match="already terminal"):
                await manual_writeoff_order(
                    session,
                    order_id="order-closed",
                    realized_pnl=0.0,
                    reason="trying to overwrite a winner",
                    operator_id="ops",
                )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_manual_writeoff_rejects_order_without_pending_exit(tmp_path):
    """An order the lifecycle has NOT yet attempted to close cannot
    be written off — the operator path is for stuck orders only."""
    engine, session_factory = await build_postgres_session_factory(
        Base, "manual_writeoff_rejects_no_pending_exit"
    )
    try:
        async with session_factory() as session:
            session.add(
                Trader(
                    id="trader-np",
                    name="t-np",
                    mode="live",
                    source_configs_json=[],
                    risk_limits_json={},
                )
            )
            now = utcnow()
            session.add(
                TraderOrder(
                    id="order-no-pending",
                    trader_id="trader-np",
                    source="scanner",
                    market_id="market-x",
                    direction="buy_yes",
                    mode="live",
                    status="executed",
                    notional_usd=10.0,
                    payload_json={"selected_token_id": "tok"},  # no pending_live_exit
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()

        async with session_factory() as session:
            with pytest.raises(ManualWriteoffRejected, match="no pending_live_exit"):
                await manual_writeoff_order(
                    session,
                    order_id="order-no-pending",
                    realized_pnl=-5.0,
                    reason="should fail",
                    operator_id="ops",
                )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_manual_writeoff_rejects_non_blocked_pending_exit(tmp_path):
    """An order whose pending_exit is just 'failed' (still retrying)
    cannot be written off — only blocked-terminal states qualify."""
    engine, session_factory = await build_postgres_session_factory(
        Base, "manual_writeoff_rejects_non_blocked"
    )
    try:
        async with session_factory() as session:
            await _seed(session, "trader-f", "order-failed", pending_exit_status="failed")

        async with session_factory() as session:
            with pytest.raises(ManualWriteoffRejected, match="not a blocked-terminal state"):
                await manual_writeoff_order(
                    session,
                    order_id="order-failed",
                    realized_pnl=-5.0,
                    reason="should fail",
                    operator_id="ops",
                )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_manual_writeoff_requires_non_empty_reason(tmp_path):
    engine, session_factory = await build_postgres_session_factory(
        Base, "manual_writeoff_requires_reason"
    )
    try:
        async with session_factory() as session:
            await _seed(session, "trader-r", "order-reason")

        async with session_factory() as session:
            with pytest.raises(ManualWriteoffRejected, match="reason is required"):
                await manual_writeoff_order(
                    session,
                    order_id="order-reason",
                    realized_pnl=0.0,
                    reason="   ",  # whitespace only
                    operator_id="ops",
                )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_manual_writeoff_requires_operator_id(tmp_path):
    engine, session_factory = await build_postgres_session_factory(
        Base, "manual_writeoff_requires_operator"
    )
    try:
        async with session_factory() as session:
            await _seed(session, "trader-o", "order-op")

        async with session_factory() as session:
            with pytest.raises(ManualWriteoffRejected, match="operator_id is required"):
                await manual_writeoff_order(
                    session,
                    order_id="order-op",
                    realized_pnl=0.0,
                    reason="valid reason",
                    operator_id="",
                )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_manual_writeoff_rejects_nan_pnl(tmp_path):
    engine, session_factory = await build_postgres_session_factory(
        Base, "manual_writeoff_rejects_nan"
    )
    try:
        async with session_factory() as session:
            await _seed(session, "trader-n", "order-nan")

        async with session_factory() as session:
            with pytest.raises(ManualWriteoffRejected, match="finite number"):
                await manual_writeoff_order(
                    session,
                    order_id="order-nan",
                    realized_pnl=float("nan"),
                    reason="valid reason",
                    operator_id="ops",
                )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_unverified_actual_profit_persists_as_baseline(tmp_path):
    """Fix KK: lifecycle-computed ``actual_profit`` must persist alongside
    an unverified ``verification_status`` (e.g. ``venue_fill``).

    Previously a DB-layer guard silently NULL'd the value unless
    ``verification_status`` was on the verifier whitelist.  That guard
    was removed because, when the verifier failed to complete, the UI
    showed $0 P&L for resolved-but-not-yet-verified positions and
    operators reasonably interpreted that as a 100% loss.  See the
    comment block above ``TraderOrder.actual_profit`` in
    ``models/database.py`` for the full architectural rationale.
    """
    engine, session_factory = await build_postgres_session_factory(
        Base, "manual_writeoff_unverified_baseline_persists"
    )
    try:
        async with session_factory() as session:
            await _seed(session, "trader-g", "order-guard", lifecycle_status="executed")

        async with session_factory() as session:
            row = await session.get(TraderOrder, "order-guard")
            assert row is not None
            # Simulate a lifecycle resolution write: actual_profit is
            # computed from the bot's recorded fill + close price, with
            # verification_status carrying the "not yet verified by the
            # on-chain trade matcher" flag.  Both must persist so the
            # UI can render real numbers immediately.
            row.actual_profit = -123.45
            row.verification_status = "venue_fill"
            await session.commit()

            persisted = await session.get(TraderOrder, "order-guard")
            assert persisted.actual_profit == pytest.approx(-123.45), (
                "Lifecycle-computed actual_profit must persist as the "
                "baseline value before the verifier overwrites it."
            )
            assert persisted.verification_status == "venue_fill"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_db_guard_allows_actual_profit_when_status_is_manual_writeoff(tmp_path):
    """The new whitelisted status must let actual_profit persist."""
    engine, session_factory = await build_postgres_session_factory(
        Base, "manual_writeoff_db_guard_allows_new_status"
    )
    try:
        async with session_factory() as session:
            await _seed(session, "trader-w", "order-mw", lifecycle_status="executed")

        async with session_factory() as session:
            row = await session.get(TraderOrder, "order-mw")
            assert row is not None
            row.actual_profit = -7.56
            row.verification_status = "manual_writeoff"
            await session.commit()

            persisted = await session.get(TraderOrder, "order-mw")
            assert persisted.actual_profit == pytest.approx(-7.56)
            assert persisted.verification_status == "manual_writeoff"
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# Venue-rejection gate (post-2026-04-28 incident hardening)
# ---------------------------------------------------------------------------
#
# After the 2026-04-28 incident — in which client-side timeouts caused
# by our own DB pressure were misclassified as venue rejections by the
# bulk writeoff script, costing ~$200 of unrealised gain — the
# manual_writeoff_order function REFUSES to fire on rows whose
# pending_live_exit.last_error doesn't match a known venue-rejection
# pattern.  Operators can override with an explicit rationale; the
# override is captured immutably in the audit event payload.


@pytest.mark.asyncio
async def test_manual_writeoff_rejects_client_side_timeout_without_override(tmp_path):
    """A row whose only failure was our own TimeoutError cannot be
    bulk-written-off.  This is the 2026-04-28 incident regression test."""
    engine, session_factory = await build_postgres_session_factory(
        Base, "manual_writeoff_rejects_timeout"
    )
    try:
        async with session_factory() as session:
            await _seed(
                session,
                "trader-tt",
                "order-timeout",
                last_error="TimeoutError",
            )

        async with session_factory() as session:
            with pytest.raises(ManualWriteoffRejected, match="venue-rejection pattern"):
                await manual_writeoff_order(
                    session,
                    order_id="order-timeout",
                    realized_pnl=-7.56,
                    reason="bulk script attempting to writeoff a timeout",
                    operator_id="bulk-script",
                )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_manual_writeoff_override_path_requires_rationale(tmp_path):
    """Override flag without rationale is itself rejected."""
    engine, session_factory = await build_postgres_session_factory(
        Base, "manual_writeoff_override_needs_rationale"
    )
    try:
        async with session_factory() as session:
            await _seed(
                session,
                "trader-or",
                "order-no-rationale",
                last_error="ConnectionError",
            )

        async with session_factory() as session:
            with pytest.raises(ManualWriteoffRejected, match="override_rationale"):
                await manual_writeoff_order(
                    session,
                    order_id="order-no-rationale",
                    realized_pnl=-7.56,
                    reason="override without rationale",
                    operator_id="ops",
                    override_venue_rejection_check=True,
                    override_rationale=None,
                )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_manual_writeoff_override_with_rationale_captures_audit(tmp_path):
    """When the operator legitimately overrides (e.g. they've verified
    venue state out-of-band), the writeoff fires AND the override
    rationale + the original client-side error are recorded
    immutably."""
    engine, session_factory = await build_postgres_session_factory(
        Base, "manual_writeoff_override_audited"
    )
    try:
        async with session_factory() as session:
            await _seed(
                session,
                "trader-ov",
                "order-override",
                last_error="TimeoutError",
            )

        async with session_factory() as session:
            await manual_writeoff_order(
                session,
                order_id="order-override",
                realized_pnl=-7.56,
                reason="market verified resolved out-of-band",
                operator_id="ops",
                override_venue_rejection_check=True,
                override_rationale="Confirmed via Polymarket UI — market resolved 12h ago",
            )
            await session.commit()

        async with session_factory() as session:
            events = (
                await session.execute(
                    select(TraderOrderVerificationEvent).where(
                        TraderOrderVerificationEvent.trader_order_id == "order-override"
                    )
                )
            ).scalars().all()
            evt = next(e for e in events if e.event_type == "manual_writeoff")
            assert evt.payload_json.get("override_venue_rejection_check") is True
            rationale = evt.payload_json.get("override_rationale")
            assert "Confirmed" in rationale
            evidence = evt.payload_json.get("venue_rejection_evidence")
            assert evidence is not None
            assert evidence.get("matched") is False
            assert "TimeoutError" in (evidence.get("last_error_excerpt") or "")
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_manual_writeoff_recognises_venue_rejection_markers(tmp_path):
    """Several known venue-rejection messages must pass the gate
    without an override."""
    engine, session_factory = await build_postgres_session_factory(
        Base, "manual_writeoff_recognises_markers"
    )
    try:
        for idx, marker in enumerate([
            "Order rejected: orderbook does not exist for this market",
            "Polymarket: market not tradable",
            "Insufficient allowance to settle the SELL",
            "REST 400: market is closed",
        ]):
            order_id = f"order-marker-{idx}"
            trader_id = f"trader-m-{idx}"
            async with session_factory() as session:
                await _seed(session, trader_id, order_id, last_error=marker)

            async with session_factory() as session:
                result = await manual_writeoff_order(
                    session,
                    order_id=order_id,
                    realized_pnl=-1.0,
                    reason=f"marker test {idx}",
                    operator_id="ops",
                )
                await session.commit()
                assert result["next_status"] == "closed_loss"
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# Reverse manual writeoff
# ---------------------------------------------------------------------------


async def _seed_written_off(
    session,
    trader_id: str,
    order_id: str,
    *,
    chain_balance_shares: float = 130.28,
    realized_pnl: float = -7.56,
):
    """Seed a row already in the closed_loss / manual_writeoff state, as
    if a prior writeoff had fired against it."""
    await _seed(session, trader_id, order_id, last_error="orderbook does not exist")
    async with session.begin():
        row = await session.get(TraderOrder, order_id)
        assert row is not None
        row.actual_profit = realized_pnl
        row.status = "closed_loss"
        row.verification_status = "manual_writeoff"
        row.verification_source = "operator"
        payload = dict(row.payload_json or {})
        payload["pending_live_exit"] = {
            **(payload.get("pending_live_exit") or {}),
            "status": "superseded_manual_writeoff",
            "resolved_at": utcnow().isoformat(),
        }
        payload["position_close"] = {
            "close_price": None,
            "price_source": "manual_writeoff",
            "close_trigger": "manual_writeoff",
            "realized_pnl": realized_pnl,
            "closed_at": utcnow().isoformat(),
            "reason": "test seed",
            "operator_id": "test-bulk",
        }
        row.payload_json = payload
        # Synthesise the audit event the bulk script would have produced,
        # including the chain_balance_shares fragment in the reason text
        # (that's what the reversal's optional sanity check looks for).
        session.add(
            TraderOrderVerificationEvent(
                id=f"ev-{order_id}",
                trader_order_id=order_id,
                verification_status="manual_writeoff",
                source="operator",
                event_type="manual_writeoff",
                reason=(
                    f"Operator manual write-off | classification=operator_intervention "
                    f"| chain_balance_shares={chain_balance_shares} | market_resolved=False"
                ),
                payload_json={
                    "operator_id": "test-bulk",
                    "realized_pnl": realized_pnl,
                },
                created_at=utcnow(),
            )
        )


@pytest.mark.asyncio
async def test_reverse_writeoff_restores_active_state_and_audits(tmp_path):
    """Happy path: a row that was incorrectly written off can be
    restored to status=executed / verification_status=wallet_position
    with actual_profit cleared, and the reversal is captured in an
    immutable manual_writeoff_reversed verification event."""
    engine, session_factory = await build_postgres_session_factory(
        Base, "reverse_writeoff_happy"
    )
    try:
        async with session_factory() as session:
            await _seed_written_off(session, "trader-rev", "order-rev")

        async with session_factory() as session:
            result = await reverse_manual_writeoff_order(
                session,
                order_id="order-rev",
                reason="DB pressure caused timeouts; wallet still holds shares; reversing",
                operator_id="ops@homerun",
                expected_chain_balance_shares=130.0,
            )
            await session.commit()

        async with session_factory() as session:
            row = await session.get(TraderOrder, "order-rev")
            assert row is not None
            assert row.status == "executed"
            assert row.actual_profit is None
            assert row.verification_status == "wallet_position"
            payload = row.payload_json or {}
            assert "position_close" not in payload
            pe = payload.get("pending_live_exit") or {}
            assert pe.get("status") == "reopened_after_writeoff_reversal"
            assert pe.get("consecutive_timeout_count") == 0
            assert pe.get("retry_count") == 0
            assert pe.get("last_error") is None

            events = (
                await session.execute(
                    select(TraderOrderVerificationEvent).where(
                        TraderOrderVerificationEvent.trader_order_id == "order-rev"
                    )
                )
            ).scalars().all()
            reversal = [e for e in events if e.event_type == "manual_writeoff_reversed"]
            assert len(reversal) == 1
            evt = reversal[0]
            assert evt.source == "operator_reversal"
            payload = evt.payload_json or {}
            assert payload.get("operator_id") == "ops@homerun"
            assert "DB pressure" in (evt.reason or "")
            assert payload.get("prior_actual_profit") == pytest.approx(-7.56)
            assert payload.get("prior_lifecycle_status") == "closed_loss"
            assert "prior_writeoff_event_id" in payload

        assert result["next_status"] == "executed"
        assert result["verification_status"] == "wallet_position"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_reverse_writeoff_rejects_non_writeoff_rows(tmp_path):
    """A row that was closed by normal lifecycle (e.g. wallet_activity
    verifier) cannot be reversed via this path — that would let an
    operator overwrite a real venue-verified close."""
    engine, session_factory = await build_postgres_session_factory(
        Base, "reverse_writeoff_rejects_non_writeoff"
    )
    try:
        async with session_factory() as session:
            await _seed(session, "trader-nw", "order-real-close")
            row = await session.get(TraderOrder, "order-real-close")
            row.status = "closed_loss"
            row.verification_status = "wallet_activity"  # NOT manual_writeoff
            row.actual_profit = -3.21
            await session.commit()

        async with session_factory() as session:
            with pytest.raises(ManualWriteoffReversalRejected, match="manual_writeoff"):
                await reverse_manual_writeoff_order(
                    session,
                    order_id="order-real-close",
                    reason="trying to overwrite a real close",
                    operator_id="ops",
                )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_reverse_writeoff_chain_balance_sanity_check(tmp_path):
    """When ``expected_chain_balance_shares`` is provided, the reversal
    aborts unless the original writeoff event recorded at least that
    many shares — so a reversal driven by 'wallet still holds shares'
    cannot fire against a row whose wallet was actually empty at
    writeoff time."""
    engine, session_factory = await build_postgres_session_factory(
        Base, "reverse_writeoff_chain_check"
    )
    try:
        async with session_factory() as session:
            await _seed_written_off(
                session, "trader-cb", "order-cb", chain_balance_shares=5.0
            )

        async with session_factory() as session:
            with pytest.raises(ManualWriteoffReversalRejected, match="sanity check failed"):
                await reverse_manual_writeoff_order(
                    session,
                    order_id="order-cb",
                    reason="checking against impossibly high balance",
                    operator_id="ops",
                    expected_chain_balance_shares=100.0,
                )

        # Below the recorded balance: passes.
        async with session_factory() as session:
            result = await reverse_manual_writeoff_order(
                session,
                order_id="order-cb",
                reason="balance is 5 — reversing",
                operator_id="ops",
                expected_chain_balance_shares=4.0,
            )
            await session.commit()
            assert result["next_status"] == "executed"
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# Reverse wallet-absent resolution
# ---------------------------------------------------------------------------


async def _seed_wallet_absent_resolved(
    session,
    trader_id: str,
    order_id: str,
    *,
    pe_status: str = "superseded_wallet_absent_no_provider",
):
    """Seed a row in the resolved/wallet-absent state — what the
    lifecycle's wallet-absent close path leaves behind when it fires
    (and what the 2026-04-28 incident incorrectly fired against three
    positions)."""
    await _seed(session, trader_id, order_id, last_error="TimeoutError")
    async with session.begin():
        row = await session.get(TraderOrder, order_id)
        assert row is not None
        row.status = "resolved"
        row.actual_profit = None
        payload = dict(row.payload_json or {})
        payload["pending_live_exit"] = {
            **(payload.get("pending_live_exit") or {}),
            "status": pe_status,
            "resolved_at": utcnow().isoformat(),
            "retry_count": 18,
            "last_error": "persistent_timeout_auto_recovery_after_1926s (prior: TimeoutError)",
        }
        payload["wallet_absent_resolution"] = {
            "resolved_at": utcnow().isoformat(),
            "reason": "stale_pending_exit_no_provider_and_wallet_absent",
            "retry_count": 18,
            "last_error": "TimeoutError",
        }
        row.payload_json = payload


@pytest.mark.asyncio
async def test_reverse_wallet_absent_resolution_happy_path(tmp_path):
    engine, session_factory = await build_postgres_session_factory(
        Base, "reverse_wallet_absent_happy"
    )
    try:
        async with session_factory() as session:
            await _seed_wallet_absent_resolved(session, "trader-wa", "order-wa")

        async with session_factory() as session:
            result = await reverse_wallet_absent_resolution(
                session,
                order_id="order-wa",
                reason="DB-pressure timeouts + Polymarket /positions blip caused false wallet-absent",
                operator_id="ops@homerun",
            )
            await session.commit()

        async with session_factory() as session:
            row = await session.get(TraderOrder, "order-wa")
            assert row is not None
            assert row.status == "executed"
            assert row.actual_profit is None
            assert row.verification_status == "wallet_position"
            payload = row.payload_json or {}
            assert "wallet_absent_resolution" not in payload
            pe = payload.get("pending_live_exit") or {}
            assert pe.get("status") == "reopened_after_wallet_absent_reversal"
            assert pe.get("retry_count") == 0
            assert pe.get("last_error") is None

            events = (
                await session.execute(
                    select(TraderOrderVerificationEvent).where(
                        TraderOrderVerificationEvent.trader_order_id == "order-wa"
                    )
                )
            ).scalars().all()
            reversal = [e for e in events if e.event_type == "wallet_absent_resolution_reversed"]
            assert len(reversal) == 1
            evt = reversal[0]
            payload = evt.payload_json or {}
            assert payload.get("operator_id") == "ops@homerun"
            assert payload.get("prior_lifecycle_status") == "resolved"
            assert payload.get("prior_pending_exit", {}).get("status") == "superseded_wallet_absent_no_provider"
            assert payload.get("prior_wallet_absent_resolution") is not None

        assert result["next_status"] == "executed"
        assert result["prior_pending_exit_status"] == "superseded_wallet_absent_no_provider"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_reverse_wallet_absent_rejects_non_resolved_rows(tmp_path):
    engine, session_factory = await build_postgres_session_factory(
        Base, "reverse_wallet_absent_rejects_non_resolved"
    )
    try:
        async with session_factory() as session:
            await _seed(session, "trader-x", "order-still-active")  # status=executed

        async with session_factory() as session:
            with pytest.raises(WalletAbsentResolutionReversalRejected, match="not 'resolved'"):
                await reverse_wallet_absent_resolution(
                    session,
                    order_id="order-still-active",
                    reason="should fail",
                    operator_id="ops",
                )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_reverse_wallet_absent_rejects_non_wallet_absent_pe_status(tmp_path):
    """A row that's resolved for a non-wallet-absent reason (real
    settlement, terminal close) cannot be reversed via this path."""
    engine, session_factory = await build_postgres_session_factory(
        Base, "reverse_wallet_absent_rejects_other_pe"
    )
    try:
        async with session_factory() as session:
            # Seed resolved with a non-wallet-absent pending_exit status.
            await _seed(session, "trader-y", "order-real-resolve")
            row = await session.get(TraderOrder, "order-real-resolve")
            row.status = "resolved"
            payload = dict(row.payload_json or {})
            payload["pending_live_exit"] = {"status": "superseded_settlement"}
            row.payload_json = payload
            await session.commit()

        async with session_factory() as session:
            with pytest.raises(
                WalletAbsentResolutionReversalRejected,
                match="wallet-absent-resolution",
            ):
                await reverse_wallet_absent_resolution(
                    session,
                    order_id="order-real-resolve",
                    reason="should fail",
                    operator_id="ops",
                )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_reverse_writeoff_requires_reason_and_operator(tmp_path):
    engine, session_factory = await build_postgres_session_factory(
        Base, "reverse_writeoff_requires_inputs"
    )
    try:
        async with session_factory() as session:
            await _seed_written_off(session, "trader-rq", "order-rq")

        async with session_factory() as session:
            with pytest.raises(ManualWriteoffReversalRejected, match="reason is required"):
                await reverse_manual_writeoff_order(
                    session,
                    order_id="order-rq",
                    reason="   ",
                    operator_id="ops",
                )

        async with session_factory() as session:
            with pytest.raises(ManualWriteoffReversalRejected, match="operator_id is required"):
                await reverse_manual_writeoff_order(
                    session,
                    order_id="order-rq",
                    reason="valid",
                    operator_id="",
                )
    finally:
        await engine.dispose()
