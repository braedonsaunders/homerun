"""Unit tests for the on-chain-derived collateral registry + inference.

The fixture matrix below is grounded in real Polymarket positions
observed on Polygon mainnet at block 86,236,467 (2026-04-30):

  * 3 vanilla USDC.e positions
  * 2 NegRisk-wrapped (WCOL) positions

For each, the wallet's ERC-1155 ``token_id``, the ``conditionId``, and
the resolved ``(collateral, slot)`` were captured directly from the CTF
``getPositionId``/``getCollectionId`` view functions. Tests mock those
view functions to return the same values, so the inference logic is
validated against ground truth without requiring an RPC.
"""

from __future__ import annotations

from typing import Any

import pytest

from services.polymarket_collateral import (
    CTF_ADDRESS,
    CollateralKind,
    CollateralRegistry,
    EXPECTED_NEGRISK_ADAPTERS,
    PUSD_ADDRESS,
    USDC_E_ADDRESS,
    USDC_NATIVE_ADDRESS,
)


# ── On-chain ground truth ───────────────────────────────────────────
# Each entry: (token_id, condition_id, expected_collateral_address, expected_slot)
USDC_E_POSITIONS = [
    (
        70875671319261145927353065712261110016720360955180326005988675454606060933537,
        "0x30406d9360ac4012d816108d4de9439ad9f6cb74ae5fc40fff6ede41016489ef",
        USDC_E_ADDRESS,
        1,
    ),
    (
        80144472161204662432632515787916731037645908021386629011174636846200963880418,
        "0x158307f1246f889c2ac2154d08c9815646842b085587b4d627c0ae936ce3c04d",
        USDC_E_ADDRESS,
        1,
    ),
    (
        90454427592177420217450605116320154354170745162774911478995578795874158973530,
        "0xbb5b5f4e6b3d851252b9d1226f514f32b6b042e62c4c0dbcc94e4638fca6ce9d",
        USDC_E_ADDRESS,
        0,
    ),
]

WCOL_ADDRESS = EXPECTED_NEGRISK_ADAPTERS[0].wrapped_collateral

NEGRISK_POSITIONS = [
    (
        94015476063235308933398522356313024719791890660976438762504144906595074123774,
        "0x251f1431c02356f7539f668cde61b9788faff667cd19223172ce38a8e8c03cc5",
        WCOL_ADDRESS,
        0,
    ),
    (
        80481301277088253580485665638115936980291630797650188377453077017936047830306,
        "0x581d855fa1794d9304b4429029917753d3ecae9ba2dd7ca684e8658fe13f4f45",
        WCOL_ADDRESS,
        0,
    ),
]


# ── Mock contract scaffolding ────────────────────────────────────────


class _Call:
    def __init__(self, value: Any) -> None:
        self._value = value

    def call(self) -> Any:
        return self._value


class _CTFFunctions:
    """Mocks CTF.getCollectionId, getPositionId, getOutcomeSlotCount.

    Built from a fixture map of (condition_id_lower, slot, collateral_lower)
    → expected_position_id. The collectionId values returned are
    synthetic but deterministic, since the inference logic only uses
    them as opaque inputs to getPositionId.
    """

    def __init__(
        self,
        position_map: dict[tuple[str, int, str], int],
        slot_count: int = 2,
    ) -> None:
        self._position_map = position_map
        self._slot_count = slot_count

    def getOutcomeSlotCount(self, condition_id: str) -> _Call:
        return _Call(self._slot_count)

    def getCollectionId(
        self, parent: str, condition_id: str, index_set: int
    ) -> _Call:
        # Synthetic deterministic collection id encoding the (condition, indexSet).
        cond = condition_id.lower().removeprefix("0x").rjust(64, "0")
        cid_bytes = bytes.fromhex(cond)[:24] + index_set.to_bytes(8, "big")
        return _Call(cid_bytes)

    def getPositionId(self, collateral: str, collection_id: bytes) -> _Call:
        # Reverse the indexSet from collection_id (last 8 bytes).
        index_set = int.from_bytes(collection_id[-8:], "big")
        # Reverse the condition (first 24 bytes prefix).
        cond_prefix = collection_id[:24].hex()
        # Find the (slot, condition) match
        slot = 0 if index_set == 1 else 1 if index_set == 2 else -1
        # Find any condition whose first 24 bytes match cond_prefix.
        for (cond_lower, s, collat_lower), pid in self._position_map.items():
            if s != slot:
                continue
            if collat_lower != collateral.lower():
                continue
            cond_full = cond_lower.removeprefix("0x").rjust(64, "0")
            if cond_full[:48] == cond_prefix:  # 24 bytes = 48 hex chars
                return _Call(pid)
        return _Call(0)

    def col(self) -> _Call:
        # Used by the NegRiskAdapter mock; routed via the same _Functions
        # facade below for simplicity. Override per-contract.
        raise AttributeError("col() not on CTF mock")

    def wcol(self) -> _Call:
        raise AttributeError("wcol() not on CTF mock")


class _AdapterFunctions:
    def __init__(self, col: str, wcol: str) -> None:
        self._col = col
        self._wcol = wcol

    def col(self) -> _Call:
        return _Call(self._col)

    def wcol(self) -> _Call:
        return _Call(self._wcol)


class _MockContract:
    def __init__(self, functions: Any) -> None:
        self.functions = functions


class _MockEth:
    def __init__(self, contracts_by_addr: dict[str, _MockContract]) -> None:
        self._by_addr = {a.lower(): c for a, c in contracts_by_addr.items()}

    def contract(self, *, address: str, abi: Any) -> _MockContract:
        c = self._by_addr.get(address.lower())
        if c is None:
            raise KeyError(f"no mock contract registered for {address}")
        return c


class _MockW3:
    def __init__(self, contracts_by_addr: dict[str, _MockContract]) -> None:
        self.eth = _MockEth(contracts_by_addr)

    @staticmethod
    def to_checksum_address(addr: str) -> str:
        # Tests don't depend on real EIP-55 checksum; just normalize case.
        return addr if addr.startswith("0x") else f"0x{addr}"


def _build_w3_with_full_fixture() -> _MockW3:
    # Build the position map for both vanilla USDC.e and WCOL positions.
    pmap: dict[tuple[str, int, str], int] = {}
    for token_id, cond, collat, slot in USDC_E_POSITIONS:
        pmap[(cond.lower(), slot, collat.lower())] = token_id
    for token_id, cond, collat, slot in NEGRISK_POSITIONS:
        pmap[(cond.lower(), slot, collat.lower())] = token_id
    ctf = _MockContract(_CTFFunctions(pmap))
    adapter = _MockContract(
        _AdapterFunctions(
            col=EXPECTED_NEGRISK_ADAPTERS[0].parent_collateral,
            wcol=EXPECTED_NEGRISK_ADAPTERS[0].wrapped_collateral,
        )
    )
    return _MockW3({CTF_ADDRESS: ctf, EXPECTED_NEGRISK_ADAPTERS[0].address: adapter})


# ── Tests ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_verify_invariants_succeeds_when_chain_matches_registry():
    registry = CollateralRegistry()
    w3 = _build_w3_with_full_fixture()
    await registry.verify_invariants(w3)
    assert registry.invariants_verified()
    # WCOL is now part of the registry as a NEGRISK_WRAPPED candidate.
    wrapped = registry.negrisk_wrapped()
    assert len(wrapped) == 1
    assert wrapped[0].kind == CollateralKind.NEGRISK_WRAPPED
    assert wrapped[0].address.lower() == WCOL_ADDRESS.lower()
    assert wrapped[0].adapter is not None
    assert wrapped[0].adapter.address.lower() == EXPECTED_NEGRISK_ADAPTERS[0].address.lower()


@pytest.mark.asyncio
async def test_verify_invariants_idempotent():
    registry = CollateralRegistry()
    w3 = _build_w3_with_full_fixture()
    await registry.verify_invariants(w3)
    # Replace adapter with one that would FAIL on a second read; if
    # verify_invariants is idempotent, it shouldn't read again.
    failing = _MockContract(
        _AdapterFunctions(col="0x0000000000000000000000000000000000000000", wcol="0x0")
    )
    w3.eth._by_addr[EXPECTED_NEGRISK_ADAPTERS[0].address.lower()] = failing
    await registry.verify_invariants(w3)  # must not raise


@pytest.mark.asyncio
async def test_verify_invariants_refuses_to_start_when_wcol_drifts():
    """If Polymarket redeploys NegRiskAdapter, wcol() will differ — we
    must refuse to start rather than silently mis-route redemptions."""
    registry = CollateralRegistry()
    bad_wcol = "0xdeadbeef00000000000000000000000000000000"
    adapter = _MockContract(
        _AdapterFunctions(
            col=EXPECTED_NEGRISK_ADAPTERS[0].parent_collateral,
            wcol=bad_wcol,
        )
    )
    w3 = _MockW3({EXPECTED_NEGRISK_ADAPTERS[0].address: adapter})
    with pytest.raises(RuntimeError, match="wcol\\(\\) invariant violated"):
        await registry.verify_invariants(w3)
    assert not registry.invariants_verified()


@pytest.mark.asyncio
async def test_verify_invariants_refuses_to_start_when_col_drifts():
    """A col() change is equally critical — the adapter would unwrap to
    a different parent stablecoin than the wallet expected."""
    registry = CollateralRegistry()
    bad_col = "0xdeadbeef00000000000000000000000000000001"
    adapter = _MockContract(
        _AdapterFunctions(
            col=bad_col,
            wcol=EXPECTED_NEGRISK_ADAPTERS[0].wrapped_collateral,
        )
    )
    w3 = _MockW3({EXPECTED_NEGRISK_ADAPTERS[0].address: adapter})
    with pytest.raises(RuntimeError, match="col\\(\\) invariant violated"):
        await registry.verify_invariants(w3)


@pytest.mark.asyncio
@pytest.mark.parametrize("token_id, condition_id, expected_collateral, expected_slot", USDC_E_POSITIONS)
async def test_infer_resolves_vanilla_usdc_e_positions_from_chain(
    token_id, condition_id, expected_collateral, expected_slot
):
    registry = CollateralRegistry()
    w3 = _build_w3_with_full_fixture()
    match = await registry.infer(
        w3,
        condition_id=condition_id,
        token_id=token_id,
        outcome_index_hint=expected_slot,
    )
    assert match is not None
    assert match.collateral.address.lower() == expected_collateral.lower()
    assert match.collateral.kind == CollateralKind.VANILLA
    assert match.outcome_slot == expected_slot
    assert match.index_set == (1 << expected_slot)


@pytest.mark.asyncio
@pytest.mark.parametrize("token_id, condition_id, expected_collateral, expected_slot", NEGRISK_POSITIONS)
async def test_infer_resolves_negrisk_wcol_positions_from_chain(
    token_id, condition_id, expected_collateral, expected_slot
):
    registry = CollateralRegistry()
    w3 = _build_w3_with_full_fixture()
    match = await registry.infer(
        w3,
        condition_id=condition_id,
        token_id=token_id,
        outcome_index_hint=expected_slot,
    )
    assert match is not None
    assert match.collateral.address.lower() == expected_collateral.lower()
    assert match.collateral.kind == CollateralKind.NEGRISK_WRAPPED
    assert match.collateral.adapter is not None
    assert match.collateral.adapter.address.lower() == EXPECTED_NEGRISK_ADAPTERS[0].address.lower()
    assert match.outcome_slot == expected_slot


@pytest.mark.asyncio
async def test_infer_returns_none_for_unknown_collateral():
    registry = CollateralRegistry()
    w3 = _build_w3_with_full_fixture()
    # A token_id that doesn't exist in the position map.
    match = await registry.infer(
        w3,
        condition_id="0x0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        token_id=99999999999999999999999999999999999999999999999,
    )
    assert match is None


@pytest.mark.asyncio
async def test_infer_works_without_outcome_index_hint():
    """Slot probing must succeed even when the data API outcome_index
    hint is missing (e.g. older positions, partial API responses)."""
    registry = CollateralRegistry()
    w3 = _build_w3_with_full_fixture()
    token_id, cond, collat, slot = USDC_E_POSITIONS[0]
    match = await registry.infer(w3, condition_id=cond, token_id=token_id)
    assert match is not None
    assert match.collateral.address.lower() == collat.lower()
    assert match.outcome_slot == slot


@pytest.mark.asyncio
async def test_infer_caches_per_condition_and_token():
    registry = CollateralRegistry()
    w3 = _build_w3_with_full_fixture()
    token_id, cond, _collat, slot = USDC_E_POSITIONS[0]

    # First inference reads the chain.
    first = await registry.infer(w3, condition_id=cond, token_id=token_id, outcome_index_hint=slot)
    assert first is not None

    # Replace the CTF mock with one that would fail; if cached, no read.
    class _FailingFunctions:
        def getOutcomeSlotCount(self, *_):
            raise RuntimeError("should not be called: result must be cached")

        def getCollectionId(self, *_):
            raise RuntimeError("should not be called: result must be cached")

        def getPositionId(self, *_):
            raise RuntimeError("should not be called: result must be cached")

    w3.eth._by_addr[CTF_ADDRESS.lower()] = _MockContract(_FailingFunctions())

    second = await registry.infer(w3, condition_id=cond, token_id=token_id, outcome_index_hint=slot)
    assert second == first


@pytest.mark.asyncio
async def test_registry_reports_all_candidates_post_invariants():
    registry = CollateralRegistry()
    # Pre-verify: vanilla candidates only.
    vanilla_addrs = {c.address.lower() for c in registry.vanilla()}
    assert PUSD_ADDRESS.lower() in vanilla_addrs
    assert USDC_E_ADDRESS.lower() in vanilla_addrs
    assert USDC_NATIVE_ADDRESS.lower() in vanilla_addrs
    assert registry.negrisk_wrapped() == ()

    # Post-verify: WCOL appears as well.
    w3 = _build_w3_with_full_fixture()
    await registry.verify_invariants(w3)
    all_addrs = {c.address.lower() for c in registry.all_candidates()}
    assert WCOL_ADDRESS.lower() in all_addrs


def test_normalize_condition_id_validates_hex_and_length():
    registry = CollateralRegistry()
    # 32-byte hex → normalized 0x-lowercase.
    text = registry._normalize_condition_id(
        "0x" + "ABCD" * 16
    )
    assert text == "0x" + "abcd" * 16
    assert len(text) == 66

    with pytest.raises(ValueError):
        registry._normalize_condition_id("0xshort")
    with pytest.raises(ValueError):
        registry._normalize_condition_id("not-hex-content-at-allllllllllllllllllllllllllllllllllllllllllllllll")


def test_by_address_lookup_is_case_insensitive():
    registry = CollateralRegistry()
    found = registry.by_address(PUSD_ADDRESS.upper())
    assert found is not None
    assert found.kind == CollateralKind.VANILLA
    assert found.name == "pUSD"
