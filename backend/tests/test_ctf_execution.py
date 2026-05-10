import pytest
from unittest.mock import AsyncMock

from services.ctf_execution import CTFExecutionService


class _TxHash:
    def __init__(self, value: str) -> None:
        self._value = value

    def hex(self) -> str:
        return self._value


class _Receipt:
    status = 1


class _Signer:
    def __init__(self, txs: list[dict]) -> None:
        self._txs = txs

    def sign_transaction(self, tx: dict, _private_key: str):
        self._txs.append(dict(tx))
        return type("Signed", (), {"raw_transaction": b"rawtx"})()


class _EthEOAStub:
    def __init__(self, nonce: int, gas_price: int) -> None:
        self._nonce = nonce
        self.gas_price = gas_price
        self.account = _Signer([])
        self.nonce_calls: list[tuple[str, str | None]] = []

    def get_transaction_count(self, address: str, block_identifier: str | None = None) -> int:
        self.nonce_calls.append((address, block_identifier))
        return self._nonce

    def send_raw_transaction(self, _raw: bytes):
        return _TxHash("0xdeadbeef")

    def wait_for_transaction_receipt(self, _tx_hash, _timeout: int):
        return _Receipt()


class _Web3EOAStub:
    def __init__(self, nonce: int = 11, gas_price: int = 100):
        self.eth = _EthEOAStub(nonce=nonce, gas_price=gas_price)


class _CallResult:
    def __init__(self, value):
        self._value = value

    def call(self):
        return self._value


class _SafeExecBuilder:
    def __init__(self, built_txs: list[dict]) -> None:
        self._built_txs = built_txs

    def build_transaction(self, tx: dict) -> dict:
        built = dict(tx)
        self._built_txs.append(built)
        return built


class _SafeFunctionsStub:
    def __init__(self, built_txs: list[dict]) -> None:
        self._built_txs = built_txs

    def nonce(self):
        return _CallResult(7)

    def getTransactionHash(self, *_args):
        return _CallResult(b"\x11" * 32)

    def execTransaction(self, *_args):
        return _SafeExecBuilder(self._built_txs)


class _SafeContractStub:
    def __init__(self, built_txs: list[dict]) -> None:
        self.functions = _SafeFunctionsStub(built_txs)


class _EthSafeStub(_EthEOAStub):
    def __init__(self, nonce: int, gas_price: int, built_txs: list[dict]) -> None:
        super().__init__(nonce=nonce, gas_price=gas_price)
        self._built_txs = built_txs

    def contract(self, address: str, abi):
        _ = address, abi
        return _SafeContractStub(self._built_txs)


class _Web3SafeStub:
    def __init__(self, nonce: int = 13, gas_price: int = 120):
        self._built_txs: list[dict] = []
        self.eth = _EthSafeStub(nonce=nonce, gas_price=gas_price, built_txs=self._built_txs)


@pytest.mark.asyncio
async def test_send_eoa_call_uses_pending_nonce():
    service = CTFExecutionService()
    w3 = _Web3EOAStub(nonce=11, gas_price=100)

    tx_hash = await service._send_eoa_call(
        w3=w3,
        from_address="0xsender",
        private_key="0xabc123",
        to_address="0xcontract",
        data=b"\x01\x02",
        gas_limit=21000,
    )

    assert tx_hash == "0xdeadbeef"
    assert ("0xsender", "pending") in w3.eth.nonce_calls
    assert w3.eth.account._txs
    assert w3.eth.account._txs[0]["nonce"] == 11


@pytest.mark.asyncio
async def test_send_safe_call_uses_pending_owner_nonce(monkeypatch):
    service = CTFExecutionService()
    w3 = _Web3SafeStub(nonce=13, gas_price=120)
    monkeypatch.setattr(service, "_safe_signature", AsyncMock(return_value=b"sig"))

    tx_hash = await service._send_safe_call(
        w3=w3,
        safe_address="0xsafe",
        owner_eoa="0xowner",
        private_key="0xabc123",
        to_address="0xcontract",
        data=b"\x03\x04",
        gas_limit=250000,
    )

    assert tx_hash == "0xdeadbeef"
    assert ("0xowner", "pending") in w3.eth.nonce_calls
    assert w3._built_txs
    assert w3._built_txs[0]["nonce"] == 13


# ── Redeemer guard math ─────────────────────────────────────────────
#
# These tests cover the pure-function payout math used by the redeemer
# guard. They lock in the world-class invariant that the bot never
# auto-burns gas on $0-payout redemptions, and that fractional / scalar
# resolutions allocate proceeds proportionally to the slot numerators.


def test_redeemer_payout_winning_yes_position_returns_full_balance():
    # Binary YES win: numerator[0]=1, numerator[1]=0, denominator=1.
    # Wallet holds 100 shares of slot 0 → payout = 100 * 1/1 = 100.
    breakdown = CTFExecutionService.compute_condition_payout_breakdown(
        denominator=1,
        outcome_balances={0: 100.0},
        outcome_numerators={0: 1, 1: 0},
    )
    assert breakdown["expected_payout_usd"] == pytest.approx(100.0)
    assert breakdown["winning_shares"] == pytest.approx(100.0)
    assert breakdown["losing_shares"] == pytest.approx(0.0)


def test_redeemer_payout_losing_no_position_returns_zero():
    # Binary YES win, wallet held NO (slot 1). Payout = 0; this is the
    # case the guard MUST catch to avoid burning gas to redeem dust.
    breakdown = CTFExecutionService.compute_condition_payout_breakdown(
        denominator=1,
        outcome_balances={1: 50.0},
        outcome_numerators={0: 1, 1: 0},
    )
    assert breakdown["expected_payout_usd"] == 0.0
    assert breakdown["winning_shares"] == 0.0
    assert breakdown["losing_shares"] == pytest.approx(50.0)


def test_redeemer_payout_split_position_winning_and_losing_legs():
    # Wallet held both YES and NO (combined-position remnant).
    # YES wins: payout = balance[0] only.
    breakdown = CTFExecutionService.compute_condition_payout_breakdown(
        denominator=1,
        outcome_balances={0: 30.0, 1: 70.0},
        outcome_numerators={0: 1, 1: 0},
    )
    assert breakdown["expected_payout_usd"] == pytest.approx(30.0)
    assert breakdown["winning_shares"] == pytest.approx(30.0)
    assert breakdown["losing_shares"] == pytest.approx(70.0)


def test_redeemer_payout_scalar_resolution_proportional_to_numerator():
    # Scalar / fractional resolution: numerators sum to denominator but
    # neither is the full payout. Slot 0 paid 0.4, slot 1 paid 0.6.
    # Holding 50 of slot 0 + 50 of slot 1 => 50*0.4 + 50*0.6 = 50.
    breakdown = CTFExecutionService.compute_condition_payout_breakdown(
        denominator=10,
        outcome_balances={0: 50.0, 1: 50.0},
        outcome_numerators={0: 4, 1: 6},
    )
    assert breakdown["expected_payout_usd"] == pytest.approx(50.0)
    # Both slots have positive numerator → both count as "winning".
    assert breakdown["winning_shares"] == pytest.approx(100.0)
    assert breakdown["losing_shares"] == pytest.approx(0.0)


def test_redeemer_payout_unresolved_market_returns_zero_payout():
    # Denominator <= 0 means the condition isn't resolved yet — math is
    # undefined; we report zero payout so the guard skips redemption.
    breakdown = CTFExecutionService.compute_condition_payout_breakdown(
        denominator=0,
        outcome_balances={0: 100.0},
        outcome_numerators={0: 1, 1: 0},
    )
    assert breakdown["expected_payout_usd"] == 0.0
    assert breakdown["total_shares"] == pytest.approx(100.0)
    assert breakdown["losing_shares"] == pytest.approx(100.0)


def test_redeemer_payout_missing_numerator_treated_as_zero():
    # If we couldn't read a slot numerator (RPC failure → defaulted to
    # 0), we should under-redeem rather than over-estimate. Held slot
    # missing from the numerator dict counts as a losing balance.
    breakdown = CTFExecutionService.compute_condition_payout_breakdown(
        denominator=1,
        outcome_balances={0: 100.0, 5: 25.0},  # slot 5 unknown
        outcome_numerators={0: 1, 1: 0},
    )
    assert breakdown["expected_payout_usd"] == pytest.approx(100.0)
    assert breakdown["winning_shares"] == pytest.approx(100.0)
    assert breakdown["losing_shares"] == pytest.approx(25.0)


# ── Collateral-aware redemption dispatch ────────────────────────────
#
# The auto-redeemer must dispatch to vanilla CTF or NegRiskAdapter
# based on the chain-derived collateral kind.  Calling vanilla on a
# NegRisk position (or vice versa) silently no-ops on chain — we lock
# the routing in with these tests so that bug class can never recur.

from services.polymarket_collateral import (
    CollateralKind,
    CollateralMatch,
    CollateralToken,
    EXPECTED_NEGRISK_ADAPTERS,
    PUSD_ADDRESS,
    USDC_E_ADDRESS,
)


class _RedeemTracker:
    """Captures the path taken by ``_redeem_via_*`` helpers."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def vanilla(self, **kwargs):
        self.calls.append({"path": "vanilla", **kwargs})

    def negrisk(self, **kwargs):
        self.calls.append({"path": "negrisk", **kwargs})


def _make_match_vanilla() -> CollateralMatch:
    return CollateralMatch(
        collateral=CollateralToken(
            name="pUSD",
            address=PUSD_ADDRESS,
            kind=CollateralKind.VANILLA,
        ),
        outcome_slot=0,
        index_set=1,
    )


def _make_match_negrisk() -> CollateralMatch:
    adapter = EXPECTED_NEGRISK_ADAPTERS[0]
    return CollateralMatch(
        collateral=CollateralToken(
            name=f"WCOL ({adapter.name})",
            address=adapter.wrapped_collateral,
            kind=CollateralKind.NEGRISK_WRAPPED,
            adapter=adapter,
        ),
        outcome_slot=0,
        index_set=1,
    )


@pytest.mark.asyncio
async def test_redeem_vanilla_path_targets_ctf_with_correct_collateral(monkeypatch):
    """Vanilla redemption must call CTF.redeemPositions with the
    inferred collateral as the first arg — NOT the legacy hardcoded
    USDC.e — so post-pUSD-migration positions redeem correctly."""
    from services import ctf_execution as mod

    service = CTFExecutionService()
    captured: dict = {}

    async def _fake_execute(**kwargs):
        captured.update(kwargs)
        return mod.CTFExecutionResult(
            status="executed",
            action=kwargs["action"],
            tx_hash="0xfeed",
            error_message=None,
            payload={},
        )

    monkeypatch.setattr(service, "_execute_contract_call", _fake_execute)

    class _StubFunc:
        def _encode_transaction_data(self):
            return b"\xab"

    class _StubFunctions:
        def redeemPositions(self, *_args):
            return _StubFunc()

    class _StubContract:
        def __init__(self):
            self.functions = _StubFunctions()

    class _StubW3:
        @staticmethod
        def to_checksum_address(a):
            return a

        class eth:
            @staticmethod
            def contract(*, address, abi):
                return _StubContract()

    result = await service._redeem_via_vanilla_ctf(
        w3=_StubW3,
        collateral_address=PUSD_ADDRESS,
        condition_id="0x" + "ab" * 32,
        index_sets=[1, 2],
    )

    assert result.status == "executed"
    assert captured["contract_address"] == service.CTF_ADDRESS
    assert captured["action"] == "redeem"
    assert result.payload["redemption_path"] == "ctf_vanilla"
    assert result.payload["collateral_address"] == PUSD_ADDRESS


@pytest.mark.asyncio
async def test_redeem_negrisk_path_requires_adapter_approval_then_routes(monkeypatch):
    """NegRisk redemption must (1) ensure setApprovalForAll(adapter)
    on the CTF and (2) call NegRiskAdapter.redeemPositions, never the
    raw CTF — calling raw CTF would target a position the wallet
    doesn't hold and silently no-op."""
    from services import ctf_execution as mod

    service = CTFExecutionService()
    flow: list[str] = []

    async def _fake_approval(*, operator_address, action):
        flow.append(f"approve:{operator_address}:{action}")
        return mod.CTFExecutionResult(
            status="executed",
            action=action,
            tx_hash="0xappr",
            error_message=None,
            payload={"already_approved": False, "operator": operator_address},
        )

    captured: dict = {}

    async def _fake_execute(**kwargs):
        flow.append(f"execute:{kwargs['contract_address']}:{kwargs['action']}")
        captured.update(kwargs)
        return mod.CTFExecutionResult(
            status="executed",
            action=kwargs["action"],
            tx_hash="0xredeem",
            error_message=None,
            payload={},
        )

    monkeypatch.setattr(service, "_ensure_ctf_operator_approval", _fake_approval)
    monkeypatch.setattr(service, "_execute_contract_call", _fake_execute)

    class _StubFunc:
        def _encode_transaction_data(self):
            return b"\xcd"

    class _StubFunctions:
        def redeemPositions(self, *_args):
            return _StubFunc()

    class _StubContract:
        def __init__(self):
            self.functions = _StubFunctions()

    class _StubW3:
        @staticmethod
        def to_checksum_address(a):
            return a

        class eth:
            @staticmethod
            def contract(*, address, abi):
                return _StubContract()

    adapter_address = EXPECTED_NEGRISK_ADAPTERS[0].address
    result = await service._redeem_via_negrisk_adapter(
        w3=_StubW3,
        adapter_address=adapter_address,
        condition_id="0x" + "cd" * 32,
        amounts_yes_no_base_units=(11_000_000, 0),
    )

    # 1. Approval call happens first.
    # 2. Redeem call happens second, against the adapter (not the CTF).
    assert flow[0].startswith(f"approve:{adapter_address}:approve_negrisk_adapter")
    assert flow[1].startswith(f"execute:{adapter_address}:redeem")
    assert result.status == "executed"
    assert result.payload["redemption_path"] == "negrisk_adapter"
    assert result.payload["amounts_yes_no"] == [11_000_000, 0]
    assert captured["contract_address"] == adapter_address


@pytest.mark.asyncio
async def test_redeem_negrisk_aborts_when_approval_fails(monkeypatch):
    """If ``setApprovalForAll`` reverts, we must NOT submit the
    redemption — the adapter would revert pulling positions and we'd
    burn gas for nothing."""
    from services import ctf_execution as mod

    service = CTFExecutionService()

    async def _fail_approval(*, operator_address, action):
        return mod.CTFExecutionResult(
            status="failed",
            action=action,
            tx_hash=None,
            error_message="revert",
            payload={"operator": operator_address},
        )

    submitted = False

    async def _fake_execute(**kwargs):
        nonlocal submitted
        submitted = True
        return mod.CTFExecutionResult(
            status="executed",
            action=kwargs["action"],
            tx_hash="0x",
            error_message=None,
            payload={},
        )

    monkeypatch.setattr(service, "_ensure_ctf_operator_approval", _fail_approval)
    monkeypatch.setattr(service, "_execute_contract_call", _fake_execute)

    result = await service._redeem_via_negrisk_adapter(
        w3=None,  # never reached
        adapter_address=EXPECTED_NEGRISK_ADAPTERS[0].address,
        condition_id="0x" + "ef" * 32,
        amounts_yes_no_base_units=(0, 0),
    )

    assert result.status == "failed"
    assert submitted is False


@pytest.mark.asyncio
async def test_redeem_resolved_wallet_positions_skips_unknown_collateral(monkeypatch):
    """A resolved condition whose collateral can't be inferred must NOT
    trigger a redemption — silent no-op redemption is the bug we're
    fixing. Surface as ``skipped_unknown_collateral``."""
    from services import ctf_execution as mod
    from services.polymarket_collateral import collateral_registry

    service = CTFExecutionService()

    # Force live_execution_service to claim the wallet.
    monkeypatch.setattr(
        mod.live_execution_service,
        "get_execution_wallet_address",
        lambda: "0x1ba7fdb0b103d7b805f7c0c097c32ed5a8ac0bae",
    )

    # One position to scan.
    async def _fake_positions(_wallet):
        return [
            {
                "conditionId": "0xfeedfeedfeedfeedfeedfeedfeedfeedfeedfeedfeedfeedfeedfeedfeedfeed",
                "asset": "12345678901234567890",
                "outcomeIndex": 0,
            }
        ]

    monkeypatch.setattr(mod.polymarket_client, "get_wallet_positions", _fake_positions)

    # Stub the registry so verify_invariants succeeds and inference returns None.
    async def _verify_ok(_w3):
        return None

    async def _infer_none(_w3, *, condition_id, token_id, outcome_index_hint=None):
        return None

    monkeypatch.setattr(collateral_registry, "verify_invariants", _verify_ok)
    monkeypatch.setattr(collateral_registry, "infer", _infer_none)

    # Stub w3 + CTF to report the condition resolved (denominator > 0).
    class _Funcs:
        def payoutDenominator(self, _cond):
            return _CallResult(1)

        def payoutNumerators(self, _cond, _slot):
            return _CallResult(1)

        def balanceOf(self, _wallet, _token):
            return _CallResult(11_000_000)

    class _Contract:
        functions = _Funcs()

    class _W3:
        @staticmethod
        def to_checksum_address(a):
            return a

        class eth:
            @staticmethod
            def contract(*, address, abi):
                return _Contract()

    async def _fake_w3():
        return _W3

    monkeypatch.setattr(service, "_get_web3", _fake_w3)
    monkeypatch.setattr(service, "_gas_price_gwei", AsyncMock(return_value=10.0))

    redeemed_calls = []

    async def _should_not_be_called(*args, **kwargs):
        redeemed_calls.append((args, kwargs))
        raise AssertionError("must not redeem with unknown collateral")

    monkeypatch.setattr(service, "_redeem_via_vanilla_ctf", _should_not_be_called)
    monkeypatch.setattr(service, "_redeem_via_negrisk_adapter", _should_not_be_called)

    summary = await service.redeem_resolved_wallet_positions(dry_run=False)

    assert summary["resolved_conditions"] == 1
    assert summary["skipped_unknown_collateral"] == 1
    assert summary["redeemed"] == 0
    assert any("unknown_collateral" in e for e in summary["errors"])
    assert redeemed_calls == []


@pytest.mark.asyncio
async def test_redeem_resolved_wallet_positions_aborts_when_invariant_fails(monkeypatch):
    """Boot invariant violation (e.g. NegRiskAdapter redeployed) must
    refuse the entire cycle, not silently proceed with vanilla-only
    redemptions."""
    from services import ctf_execution as mod
    from services.polymarket_collateral import collateral_registry

    service = CTFExecutionService()

    monkeypatch.setattr(
        mod.live_execution_service,
        "get_execution_wallet_address",
        lambda: "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
    )

    async def _one_pos(_wallet):
        return [
            {
                "conditionId": "0x" + "ab" * 32,
                "asset": "1",
                "outcomeIndex": 0,
            }
        ]

    monkeypatch.setattr(mod.polymarket_client, "get_wallet_positions", _one_pos)

    async def _verify_fails(_w3):
        raise RuntimeError("wcol() invariant violated: ...")

    monkeypatch.setattr(collateral_registry, "verify_invariants", _verify_fails)

    class _W3:
        @staticmethod
        def to_checksum_address(a):
            return a

        class eth:
            @staticmethod
            def contract(*, address, abi):
                return None

    async def _fake_w3():
        return _W3

    monkeypatch.setattr(service, "_get_web3", _fake_w3)

    summary = await service.redeem_resolved_wallet_positions(dry_run=False)
    assert summary["redeemed"] == 0
    assert summary["resolved_conditions"] == 0
    assert any("collateral_invariants_violated" in e for e in summary["errors"])


def test_default_collateral_resolution_falls_back_to_pusd(monkeypatch):
    """The split/merge default must follow the operator setting — and
    fall back to pUSD (the post-2026-04 canonical) when missing."""
    from services import ctf_execution as mod

    monkeypatch.setattr(mod.settings, "POLYMARKET_DEFAULT_COLLATERAL", "", raising=False)
    assert mod._resolve_default_collateral_address() == PUSD_ADDRESS

    monkeypatch.setattr(mod.settings, "POLYMARKET_DEFAULT_COLLATERAL", "usdc.e", raising=False)
    assert mod._resolve_default_collateral_address() == USDC_E_ADDRESS

    monkeypatch.setattr(mod.settings, "POLYMARKET_DEFAULT_COLLATERAL", "PUSD", raising=False)
    assert mod._resolve_default_collateral_address() == PUSD_ADDRESS

    monkeypatch.setattr(mod.settings, "POLYMARKET_DEFAULT_COLLATERAL", "garbage_value", raising=False)
    # Unknown values fall through to pUSD rather than crashing.
    assert mod._resolve_default_collateral_address() == PUSD_ADDRESS


# ── CLOB V2 exchange-operator approvals ─────────────────────────────
#
# Polymarket cut over from CLOB V1 (single 0x4bFb…982E exchange) to
# V2 (CtfExchangeV2 + NegRiskCtfExchangeV2) on 2026-04-28. The CTF
# must have ``setApprovalForAll`` granted to **both** V2 operators —
# normal markets execute on CtfExchangeV2 and negrisk markets execute
# on NegRiskCtfExchangeV2; missing either operator silently breaks
# half of the market universe. These tests lock in the
# ``ensure_exchange_approval`` contract end-to-end.


def test_v2_exchange_addresses_match_sdk():
    """Class attrs must equal what ``py_clob_client_v2`` ships with so
    we don't drift from the live exchange contracts."""
    from py_clob_client_v2.config import get_contract_config

    cfg = get_contract_config(137)
    assert (
        CTFExecutionService.POLYMARKET_EXCHANGE_V2.lower()
        == cfg.exchange_v2.lower()
    )
    assert (
        CTFExecutionService.POLYMARKET_NEG_RISK_EXCHANGE_V2.lower()
        == cfg.neg_risk_exchange_v2.lower()
    )


@pytest.mark.asyncio
async def test_ensure_exchange_approval_calls_both_v2_operators(monkeypatch):
    """Both CtfExchangeV2 and NegRiskCtfExchangeV2 must be approved.
    A first-call short-circuit on the negrisk operator would silently
    leave negrisk markets unable to execute."""
    from services import ctf_execution as mod

    service = CTFExecutionService()
    invocations: list[str] = []

    async def _approve(*, operator_address, action):
        invocations.append(operator_address)
        return mod.CTFExecutionResult(
            status="executed",
            action=action,
            tx_hash=None,  # already approved
            error_message=None,
            payload={"already_approved": True, "operator": operator_address},
        )

    monkeypatch.setattr(service, "_ensure_ctf_operator_approval", _approve)

    result = await service.ensure_exchange_approval()

    assert result.status == "executed"
    assert invocations == [
        service.POLYMARKET_EXCHANGE_V2,
        service.POLYMARKET_NEG_RISK_EXCHANGE_V2,
    ]
    operators = [a["operator"] for a in result.payload["approvals"]]
    assert operators == [
        service.POLYMARKET_EXCHANGE_V2,
        service.POLYMARKET_NEG_RISK_EXCHANGE_V2,
    ]
    assert all(a["already_approved"] for a in result.payload["approvals"])
    assert result.tx_hash is None  # nothing actually submitted


@pytest.mark.asyncio
async def test_ensure_exchange_approval_aggregates_mixed_already_approved(monkeypatch):
    """One operator already approved, the other needs a fresh tx —
    the aggregate result must still be ``executed`` and the payload
    must record the per-operator state (``already_approved`` +
    ``tx_hash``) so an operator can audit which calls actually hit
    the chain."""
    from services import ctf_execution as mod

    service = CTFExecutionService()

    async def _approve(*, operator_address, action):
        already = operator_address == service.POLYMARKET_EXCHANGE_V2
        return mod.CTFExecutionResult(
            status="executed",
            action=action,
            tx_hash=None if already else "0xfreshtx",
            error_message=None,
            payload={"already_approved": already, "operator": operator_address},
        )

    monkeypatch.setattr(service, "_ensure_ctf_operator_approval", _approve)

    result = await service.ensure_exchange_approval()

    assert result.status == "executed"
    assert result.tx_hash == "0xfreshtx"  # propagates the fresh leg's hash
    by_op = {a["operator"]: a for a in result.payload["approvals"]}
    assert by_op[service.POLYMARKET_EXCHANGE_V2]["already_approved"] is True
    assert by_op[service.POLYMARKET_NEG_RISK_EXCHANGE_V2]["already_approved"] is False
    assert by_op[service.POLYMARKET_NEG_RISK_EXCHANGE_V2]["tx_hash"] == "0xfreshtx"


@pytest.mark.asyncio
async def test_ensure_exchange_approval_aborts_on_first_failure(monkeypatch):
    """If the first operator's approval reverts, we must stop — no
    point submitting against the second operator if the first one
    failed for a reason that will likely repeat (e.g. nonce race or
    insufficient gas)."""
    from services import ctf_execution as mod

    service = CTFExecutionService()
    invocations: list[str] = []

    async def _approve(*, operator_address, action):
        invocations.append(operator_address)
        return mod.CTFExecutionResult(
            status="failed",
            action=action,
            tx_hash=None,
            error_message="revert",
            payload={"operator": operator_address},
        )

    monkeypatch.setattr(service, "_ensure_ctf_operator_approval", _approve)

    result = await service.ensure_exchange_approval()

    assert result.status == "failed"
    assert result.error_message == "revert"
    # Stopped after the first operator — never tried negrisk.
    assert invocations == [service.POLYMARKET_EXCHANGE_V2]
    assert result.payload["operator"] == service.POLYMARKET_EXCHANGE_V2
