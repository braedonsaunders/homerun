"""Tests for the StrategySDK partial-exit (action="reduce") consumer plumbing.

These exercise the strategy-agnostic helpers in ``position_lifecycle`` that
honor an ``ExitDecision(action="reduce", reduce_fraction=X)`` by:

  * submitting a venue sell sized by the fraction (live mode)
  * realizing a slice of P&L and shrinking residual notional (shadow mode)
  * persisting a bounded ``partial_exit_history`` audit trail
  * leaving the order's status open so subsequent ticks manage the remainder

Pure unit tests on the helper surface — no DB, no event loop wiring, no
strategy code — so the contract is locked down independently of any
specific strategy that emits reduces.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.trader_orchestrator.position_lifecycle import (  # noqa: E402
    _coerce_reduce_fraction,
    _record_partial_exit,
    _submit_live_partial_exit,
    _submit_shadow_partial_exit,
)


# ---------------------------------------------------------------------------
# _coerce_reduce_fraction — input validation
# ---------------------------------------------------------------------------


class TestCoerceReduceFraction:
    def test_valid_fraction(self):
        d = SimpleNamespace(action="reduce", reduce_fraction=0.25)
        assert _coerce_reduce_fraction(d) == pytest.approx(0.25)

    def test_close_action_returns_none(self):
        d = SimpleNamespace(action="close", reduce_fraction=0.5)
        assert _coerce_reduce_fraction(d) is None

    def test_hold_action_returns_none(self):
        d = SimpleNamespace(action="hold", reduce_fraction=0.5)
        assert _coerce_reduce_fraction(d) is None

    def test_zero_fraction_returns_none(self):
        d = SimpleNamespace(action="reduce", reduce_fraction=0.0)
        assert _coerce_reduce_fraction(d) is None

    def test_one_fraction_returns_none(self):
        # frac == 1.0 is a full close, not a reduce; caller should escalate.
        d = SimpleNamespace(action="reduce", reduce_fraction=1.0)
        assert _coerce_reduce_fraction(d) is None

    def test_negative_fraction_returns_none(self):
        d = SimpleNamespace(action="reduce", reduce_fraction=-0.1)
        assert _coerce_reduce_fraction(d) is None

    def test_missing_fraction_returns_none(self):
        d = SimpleNamespace(action="reduce", reduce_fraction=None)
        assert _coerce_reduce_fraction(d) is None

    def test_malformed_fraction_returns_none(self):
        d = SimpleNamespace(action="reduce", reduce_fraction="not-a-number")
        assert _coerce_reduce_fraction(d) is None

    def test_none_decision_returns_none(self):
        assert _coerce_reduce_fraction(None) is None


# ---------------------------------------------------------------------------
# _record_partial_exit — bounded audit history
# ---------------------------------------------------------------------------


class TestRecordPartialExit:
    def test_appends_entry_to_empty_history(self):
        payload: dict = {}
        now = datetime(2026, 5, 9, 12, 0, 0, tzinfo=timezone.utc)
        _record_partial_exit(
            payload,
            reason="scale-out tier 1",
            reduce_fraction=0.25,
            exit_size=100.0,
            close_price=0.96,
            price_source="ws",
            now=now,
        )
        assert len(payload["partial_exit_history"]) == 1
        entry = payload["partial_exit_history"][0]
        assert entry["reason"] == "scale-out tier 1"
        assert entry["reduce_fraction"] == 0.25
        assert entry["exit_size"] == 100.0
        assert entry["close_price"] == 0.96
        assert entry["price_source"] == "ws"

    def test_extra_fields_merge(self):
        payload: dict = {}
        now = datetime(2026, 5, 9, 12, 0, 0, tzinfo=timezone.utc)
        _record_partial_exit(
            payload,
            reason="tier",
            reduce_fraction=0.5,
            exit_size=50.0,
            close_price=0.97,
            price_source="redis",
            now=now,
            extra={"status": "submitted", "exit_order_id": "abc-123"},
        )
        entry = payload["partial_exit_history"][0]
        assert entry["status"] == "submitted"
        assert entry["exit_order_id"] == "abc-123"

    def test_extra_cannot_overwrite_core_fields(self):
        """Critical: status fields can be overridden, but core audit values can't be smuggled."""
        payload: dict = {}
        now = datetime(2026, 5, 9, 12, 0, 0, tzinfo=timezone.utc)
        _record_partial_exit(
            payload,
            reason="tier",
            reduce_fraction=0.25,
            exit_size=100.0,
            close_price=0.96,
            price_source="ws",
            now=now,
            extra={"reduce_fraction": 999.0, "exit_size": 0.0},
        )
        entry = payload["partial_exit_history"][0]
        assert entry["reduce_fraction"] == 0.25  # not 999.0
        assert entry["exit_size"] == 100.0  # not 0.0

    def test_history_capped_at_16_entries(self):
        payload: dict = {}
        now = datetime(2026, 5, 9, 12, 0, 0, tzinfo=timezone.utc)
        for i in range(20):
            _record_partial_exit(
                payload,
                reason=f"tier-{i}",
                reduce_fraction=0.25,
                exit_size=10.0,
                close_price=0.95,
                price_source="ws",
                now=now,
            )
        assert len(payload["partial_exit_history"]) == 16
        # Oldest entries dropped — first kept entry should be the 5th submission
        assert payload["partial_exit_history"][0]["reason"] == "tier-4"
        assert payload["partial_exit_history"][-1]["reason"] == "tier-19"

    def test_invalid_existing_history_replaced(self):
        payload = {"partial_exit_history": "not-a-list"}
        now = datetime(2026, 5, 9, 12, 0, 0, tzinfo=timezone.utc)
        _record_partial_exit(
            payload,
            reason="tier",
            reduce_fraction=0.25,
            exit_size=10.0,
            close_price=0.95,
            price_source="ws",
            now=now,
        )
        assert isinstance(payload["partial_exit_history"], list)
        assert len(payload["partial_exit_history"]) == 1


# ---------------------------------------------------------------------------
# _submit_shadow_partial_exit — pure-payload accounting
# ---------------------------------------------------------------------------


class TestShadowPartialExit:
    @pytest.mark.asyncio
    async def test_records_realized_pnl_slice(self):
        payload = {
            # Provide a non-empty payload so reduce_fraction has context
        }
        decision = SimpleNamespace(
            action="reduce", reduce_fraction=0.25, reason="scale-out tier 1"
        )
        now = datetime(2026, 5, 9, 12, 0, 0, tzinfo=timezone.utc)
        ok = await _submit_shadow_partial_exit(
            payload=payload,
            decision=decision,
            close_price=0.96,
            price_source="ws",
            notional_usd=100.0,
            entry_price=0.88,
            now=now,
        )
        assert ok is True
        # 25% of 100 USD notional = 25 USD sliced
        # quantity = 25 / 0.88 = 28.4090...
        # proceeds = 28.4090... * 0.96 = 27.27...
        # realized = 27.27... - 25.0 = 2.27...
        history = payload["partial_realized_pnl_history"]
        assert len(history) == 1
        assert history[0]["realized_pnl"] == pytest.approx(2.272727, rel=1e-3)
        assert history[0]["sliced_notional_usd"] == pytest.approx(25.0)
        assert history[0]["fraction"] == pytest.approx(0.25)
        # Residual notional shrunk
        assert payload["effective_notional_usd"] == pytest.approx(75.0)
        # Audit row recorded
        assert payload["partial_exit_history"][0]["mode"] == "shadow"

    @pytest.mark.asyncio
    async def test_no_op_for_close_decision(self):
        payload: dict = {}
        decision = SimpleNamespace(action="close", reduce_fraction=None)
        ok = await _submit_shadow_partial_exit(
            payload=payload,
            decision=decision,
            close_price=0.95,
            price_source="ws",
            notional_usd=100.0,
            entry_price=0.88,
            now=datetime.now(timezone.utc),
        )
        assert ok is False
        assert "partial_realized_pnl_history" not in payload
        assert "partial_exit_history" not in payload

    @pytest.mark.asyncio
    async def test_no_op_for_zero_entry_price(self):
        payload: dict = {}
        decision = SimpleNamespace(action="reduce", reduce_fraction=0.5)
        ok = await _submit_shadow_partial_exit(
            payload=payload,
            decision=decision,
            close_price=0.95,
            price_source="ws",
            notional_usd=100.0,
            entry_price=0.0,  # invalid
            now=datetime.now(timezone.utc),
        )
        assert ok is False

    @pytest.mark.asyncio
    async def test_no_op_for_no_close_price(self):
        payload: dict = {}
        decision = SimpleNamespace(action="reduce", reduce_fraction=0.5)
        ok = await _submit_shadow_partial_exit(
            payload=payload,
            decision=decision,
            close_price=None,
            price_source="ws",
            notional_usd=100.0,
            entry_price=0.88,
            now=datetime.now(timezone.utc),
        )
        assert ok is False

    @pytest.mark.asyncio
    async def test_residual_notional_floor_at_zero(self):
        """Two consecutive 60% slices shouldn't make residual notional go negative."""
        payload = {"effective_notional_usd": 100.0}
        decision = SimpleNamespace(action="reduce", reduce_fraction=0.6, reason="t")
        now = datetime.now(timezone.utc)
        await _submit_shadow_partial_exit(
            payload=payload,
            decision=decision,
            close_price=0.95,
            price_source="ws",
            notional_usd=100.0,
            entry_price=0.88,
            now=now,
        )
        # First slice: residual = 40.0
        assert payload["effective_notional_usd"] == pytest.approx(40.0)
        # Now run again from the residual base
        await _submit_shadow_partial_exit(
            payload=payload,
            decision=decision,
            close_price=0.96,
            price_source="ws",
            notional_usd=40.0,
            entry_price=0.88,
            now=now,
        )
        # 60% of 40 = 24 sliced; residual = 16.0
        assert payload["effective_notional_usd"] == pytest.approx(16.0)


# ---------------------------------------------------------------------------
# _submit_live_partial_exit — venue submission with mocked execute_live_order
# ---------------------------------------------------------------------------


class TestLivePartialExit:
    @pytest.fixture
    def order_row(self):
        row = MagicMock()
        row.id = "order-123"
        return row

    @pytest.fixture
    def base_payload(self):
        return {
            "yes_token_id": "tok-yes-123",
            "selected_token_id": "tok-yes-123",
            "side": "BUY",
        }

    @pytest.fixture
    def base_session(self):
        return MagicMock()

    @pytest.mark.asyncio
    async def test_submits_partial_size_to_venue(self, order_row, base_payload, base_session):
        decision = SimpleNamespace(
            action="reduce", reduce_fraction=0.25, reason="scale-out tier 1"
        )
        exec_result = SimpleNamespace(
            status="submitted",
            order_id="exec-abc",
            payload={"clob_order_id": "clob-xyz"},
            error_message=None,
        )

        with patch(
            "services.live_execution_adapter.execute_live_order", new_callable=AsyncMock
        ) as mock_exec, patch(
            "services.trader_orchestrator.position_lifecycle._prepare_sell_allowance_bounded",
            new_callable=AsyncMock,
        ), patch(
            "services.trader_orchestrator.position_lifecycle.release_conn"
        ) as mock_release:
            mock_release.return_value.__aenter__ = AsyncMock(return_value=None)
            mock_release.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_exec.return_value = exec_result

            submitted, error = await _submit_live_partial_exit(
                session=base_session,
                row=order_row,
                payload=base_payload,
                decision=decision,
                close_price=0.96,
                price_source="ws",
                filled_size=100.0,  # entry filled 100 shares
                notional_usd=88.0,
                entry_price=0.88,
                params={"min_order_size_usd": 1.0},
                now=datetime.now(timezone.utc),
            )

        assert submitted is True
        assert error is None
        # The submitted size must be 25% of 100.0 = 25.0
        called_kwargs = mock_exec.call_args.kwargs
        assert called_kwargs["size"] == pytest.approx(25.0)
        assert called_kwargs["side"] == "SELL"
        assert called_kwargs["token_id"] == "tok-yes-123"
        # History recorded
        assert len(base_payload["partial_exit_history"]) == 1
        entry = base_payload["partial_exit_history"][0]
        assert entry["status"] == "submitted"
        assert entry["exit_order_id"] == "exec-abc"
        assert entry["provider_clob_order_id"] == "clob-xyz"

    @pytest.mark.asyncio
    async def test_no_token_id_short_circuits(self, order_row, base_session):
        payload = {}  # no token_id keys
        decision = SimpleNamespace(action="reduce", reduce_fraction=0.5, reason="t")
        submitted, error = await _submit_live_partial_exit(
            session=base_session,
            row=order_row,
            payload=payload,
            decision=decision,
            close_price=0.95,
            price_source="ws",
            filled_size=100.0,
            notional_usd=88.0,
            entry_price=0.88,
            params={},
            now=datetime.now(timezone.utc),
        )
        assert submitted is False
        assert error == "missing_token_id"

    @pytest.mark.asyncio
    async def test_invalid_fraction_no_ops(self, order_row, base_payload, base_session):
        decision = SimpleNamespace(action="reduce", reduce_fraction=None, reason="t")
        submitted, error = await _submit_live_partial_exit(
            session=base_session,
            row=order_row,
            payload=base_payload,
            decision=decision,
            close_price=0.95,
            price_source="ws",
            filled_size=100.0,
            notional_usd=88.0,
            entry_price=0.88,
            params={},
            now=datetime.now(timezone.utc),
        )
        assert submitted is False
        assert error == "invalid_reduce_fraction"

    @pytest.mark.asyncio
    async def test_below_min_notional_short_circuits(self, order_row, base_payload, base_session):
        """A slice that can't clear the venue floor must NOT submit — prevent over-fragmentation.

        Pin ``_effective_exit_min_order_size_usd`` to 1.0 so this test doesn't
        couple to that helper's current 0.01 stub return value.
        """
        decision = SimpleNamespace(action="reduce", reduce_fraction=0.01, reason="t")
        with patch(
            "services.live_execution_adapter.execute_live_order", new_callable=AsyncMock
        ) as mock_exec, patch(
            "services.trader_orchestrator.position_lifecycle._effective_exit_min_order_size_usd",
            return_value=1.0,
        ):
            submitted, error = await _submit_live_partial_exit(
                session=base_session,
                row=order_row,
                payload=base_payload,
                decision=decision,
                close_price=0.5,
                price_source="ws",
                filled_size=10.0,  # 1% of 10 = 0.1 size at $0.50 = $0.05 notional
                notional_usd=5.0,
                entry_price=0.5,
                params={"min_order_size_usd": 1.0},
                now=datetime.now(timezone.utc),
            )
            mock_exec.assert_not_called()
        assert submitted is False
        assert error == "partial_below_min_notional"

    @pytest.mark.asyncio
    async def test_exec_failure_recorded_in_history(self, order_row, base_payload, base_session):
        decision = SimpleNamespace(action="reduce", reduce_fraction=0.25, reason="t")
        exec_result = SimpleNamespace(
            status="rejected",
            order_id=None,
            payload={},
            error_message="venue rejected",
        )
        with patch(
            "services.live_execution_adapter.execute_live_order", new_callable=AsyncMock
        ) as mock_exec, patch(
            "services.trader_orchestrator.position_lifecycle._prepare_sell_allowance_bounded",
            new_callable=AsyncMock,
        ), patch(
            "services.trader_orchestrator.position_lifecycle.release_conn"
        ) as mock_release:
            mock_release.return_value.__aenter__ = AsyncMock(return_value=None)
            mock_release.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_exec.return_value = exec_result

            submitted, error = await _submit_live_partial_exit(
                session=base_session,
                row=order_row,
                payload=base_payload,
                decision=decision,
                close_price=0.95,
                price_source="ws",
                filled_size=100.0,
                notional_usd=88.0,
                entry_price=0.88,
                params={"min_order_size_usd": 1.0},
                now=datetime.now(timezone.utc),
            )
        assert submitted is False
        assert error == "venue rejected"
        # Failure still audited
        assert len(base_payload["partial_exit_history"]) == 1
        assert base_payload["partial_exit_history"][0]["status"] == "failed"
