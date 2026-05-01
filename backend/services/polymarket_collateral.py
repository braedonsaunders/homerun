"""On-chain-derived collateral inference for Polymarket CTF positions.

Polymarket positions are ERC-1155 tokens whose ID is deterministically
derived from ``(collateralToken, conditionId, indexSet)`` via the Gnosis
ConditionalTokensFramework's ``getCollectionId`` and ``getPositionId``.

A wallet's open positions can sit on any of:

  * **USDC.e** (legacy bridged USDC) — vanilla CTF.
  * **pUSD** (Polymarket's canonical stablecoin post-2026-04 migration) — vanilla CTF.
  * **USDC native** (Polygon-native USDC, occasional) — vanilla CTF.
  * **WCOL** (NegRiskAdapter wrapped collateral) — NegRisk markets ONLY.

Different collateral → different position ID, even for the same condition
and outcome slot. Calling ``CTF.redeemPositions`` with the wrong
collateral arg silently no-ops (the position lookup misses), leaving the
wallet with un-redeemable resolved shares.

Rather than guess — or hard-code a single collateral that fails
half-quiet for the rest — we resolve the collateral from the chain's own
math:

  1. For each registered candidate, compute the expected position ID via
     the CTF's own ``getPositionId(collateral, getCollectionId(0, condition, indexSet))``
     view functions. We deliberately do **not** re-implement keccak in
     Python: encoding bugs are silent and would mis-route redemptions.
     The chain is the spec.
  2. The unique match against the wallet's held token ID proves both the
     collateral and the outcome slot.
  3. Cache the answer per ``(conditionId, tokenId)`` — both immutable.

NegRiskAdapter routing
----------------------

A NegRiskAdapter wraps a parent ERC-20 (e.g. USDC.e) into a per-adapter
``WrappedCollateral`` ERC-20. Inner CTF conditions are minted against the
wrapped token, so the position IDs use ``wcol()`` as the collateral. The
adapter's ``col``/``wcol`` are ``immutable`` — a pUSD-backed NegRisk
adapter requires a fresh deployment, and a wallet may simultaneously
hold positions across multiple adapters mid-migration. The registry
therefore tracks adapters as a list; each registered entry's
``col()``/``wcol()`` are asserted at boot and the WCOL is registered as
its own ``CollateralToken`` linked back to the originating adapter.

Boot invariants
---------------

``verify_invariants(w3)`` asserts that every registered NegRiskAdapter's
on-chain ``col()`` and ``wcol()`` match the values we statically expect.
Drift means Polymarket has redeployed an adapter (or the registry is out
of date); we **refuse to start** rather than silently mis-route
redemptions. Update the registry only after manual verification of any
newly-deployed adapter.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Canonical Polymarket contract addresses on Polygon mainnet (chain 137).
# Source: https://docs.polymarket.com/contracts (verified 2026-04-30).
# ─────────────────────────────────────────────────────────────────────
CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

# Vanilla-CTF collateral tokens we know how to redeem against directly.
USDC_E_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"      # legacy bridged USDC
PUSD_ADDRESS = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB"        # current canonical
USDC_NATIVE_ADDRESS = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"  # Polygon-native USDC


class CollateralKind(str, Enum):
    """How a position must be redeemed."""

    VANILLA = "vanilla"
    NEGRISK_WRAPPED = "negrisk_wrapped"


@dataclass(frozen=True)
class NegRiskAdapterInfo:
    """One deployed NegRiskAdapter, with its wrapped + parent collateral.

    All addresses stored lower-cased; checksum is applied at call sites.
    """

    name: str
    address: str
    parent_collateral: str   # immutable col()
    wrapped_collateral: str  # immutable wcol()


@dataclass(frozen=True)
class CollateralToken:
    """A token that may serve as the ``collateralToken`` for a CTF position."""

    name: str
    address: str
    kind: CollateralKind
    # Populated for NEGRISK_WRAPPED kind only — points to the adapter
    # that minted the wrapper, so the redeemer can route through it.
    adapter: NegRiskAdapterInfo | None = None


@dataclass(frozen=True)
class CollateralMatch:
    """The unique on-chain answer for a held ``(conditionId, tokenId)``."""

    collateral: CollateralToken
    outcome_slot: int  # 0-indexed
    index_set: int     # 1 << outcome_slot for atomic outcomes


# ─────────────────────────────────────────────────────────────────────
# Statically-known NegRiskAdapter deployments. New adapters (e.g. a
# pUSD-backed one once Polymarket deploys it) MUST be added here only
# after manual verification of the on-chain col/wcol values, because the
# boot invariant compares against these constants.
# ─────────────────────────────────────────────────────────────────────
EXPECTED_NEGRISK_ADAPTERS: tuple[NegRiskAdapterInfo, ...] = (
    NegRiskAdapterInfo(
        name="NegRiskAdapter (USDC.e)",
        address="0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296",
        parent_collateral=USDC_E_ADDRESS,
        wrapped_collateral="0x3a3bd7bb9528e159577f7c2e685cc81a765002e2",
    ),
)


_CTF_HELPERS_ABI = (
    {
        "name": "getCollectionId",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId", "type": "bytes32"},
            {"name": "indexSet", "type": "uint256"},
        ],
        "outputs": [{"type": "bytes32"}],
    },
    {
        "name": "getPositionId",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "collateralToken", "type": "address"},
            {"name": "collectionId", "type": "bytes32"},
        ],
        "outputs": [{"type": "uint256"}],
    },
    {
        "name": "getOutcomeSlotCount",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "conditionId", "type": "bytes32"}],
        "outputs": [{"type": "uint256"}],
    },
)

_NEGRISK_ADAPTER_ABI = (
    {
        "name": "wcol",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"type": "address"}],
    },
    {
        "name": "col",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"type": "address"}],
    },
)


_ZERO_BYTES32 = "0x" + "00" * 32


class CollateralRegistry:
    """Process-wide registry of known collaterals plus chain inference.

    The registry is populated lazily: vanilla collaterals are known
    statically, and NegRisk adapters' wrapped collaterals are registered
    once ``verify_invariants(w3)`` confirms the on-chain ``col``/``wcol``
    match our expectations. Inference results are cached per
    ``(condition_id, token_id)`` for the lifetime of the process — both
    are immutable on-chain identifiers, so a stale cache would only be a
    problem if the registry itself changes mid-process (and that is
    explicitly rejected by ``verify_invariants``).
    """

    def __init__(self) -> None:
        self._vanilla: tuple[CollateralToken, ...] = (
            CollateralToken("pUSD", PUSD_ADDRESS, CollateralKind.VANILLA),
            CollateralToken("USDC.e", USDC_E_ADDRESS, CollateralKind.VANILLA),
            CollateralToken("USDC native", USDC_NATIVE_ADDRESS, CollateralKind.VANILLA),
        )
        self._negrisk: tuple[CollateralToken, ...] = ()
        self._invariants_verified: bool = False
        self._invariants_lock = threading.Lock()
        self._inference_cache: dict[tuple[str, int], CollateralMatch | None] = {}
        self._inference_lock = asyncio.Lock()

    # ── Public iteration ───────────────────────────────────────────
    def all_candidates(self) -> tuple[CollateralToken, ...]:
        return self._vanilla + self._negrisk

    def vanilla(self) -> tuple[CollateralToken, ...]:
        return self._vanilla

    def negrisk_wrapped(self) -> tuple[CollateralToken, ...]:
        return self._negrisk

    def by_address(self, address: str) -> CollateralToken | None:
        target = (address or "").lower()
        for c in self.all_candidates():
            if c.address.lower() == target:
                return c
        return None

    # ── Boot-time invariants ──────────────────────────────────────
    async def verify_invariants(self, w3: Any) -> None:
        """Verify every registered NegRiskAdapter's on-chain col/wcol match expectations.

        Idempotent across the process lifetime: subsequent calls after
        the first successful verification are no-ops.

        Raises ``RuntimeError`` if any adapter's on-chain values diverge
        from the registry — this is a hard failure mode by design. A
        silent divergence would mis-route redemptions; we refuse to
        proceed and require manual review of the registry.
        """
        with self._invariants_lock:
            if self._invariants_verified:
                return

        verified: list[CollateralToken] = []
        for adapter in EXPECTED_NEGRISK_ADAPTERS:
            adapter_addr = w3.to_checksum_address(adapter.address)
            contract = w3.eth.contract(address=adapter_addr, abi=list(_NEGRISK_ADAPTER_ABI))
            try:
                on_chain_col = await asyncio.to_thread(
                    lambda: contract.functions.col().call()
                )
                on_chain_wcol = await asyncio.to_thread(
                    lambda: contract.functions.wcol().call()
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to read NegRiskAdapter immutables during boot invariant check; "
                    f"adapter={adapter.name} address={adapter.address}: {exc}"
                ) from exc

            on_chain_col_lower = str(on_chain_col).lower()
            on_chain_wcol_lower = str(on_chain_wcol).lower()
            if on_chain_col_lower != adapter.parent_collateral.lower():
                raise RuntimeError(
                    f"NegRiskAdapter col() invariant violated for {adapter.name}: "
                    f"on-chain={on_chain_col} expected={adapter.parent_collateral}. "
                    "Polymarket appears to have redeployed the adapter or the registry is "
                    "out of date. Refusing to start to prevent silently mis-routed "
                    "redemptions. Update EXPECTED_NEGRISK_ADAPTERS after manual review."
                )
            if on_chain_wcol_lower != adapter.wrapped_collateral.lower():
                raise RuntimeError(
                    f"NegRiskAdapter wcol() invariant violated for {adapter.name}: "
                    f"on-chain={on_chain_wcol} expected={adapter.wrapped_collateral}. "
                    "Polymarket appears to have redeployed the adapter or the registry is "
                    "out of date. Refusing to start to prevent silently mis-routed "
                    "redemptions. Update EXPECTED_NEGRISK_ADAPTERS after manual review."
                )
            verified.append(
                CollateralToken(
                    name=f"WCOL ({adapter.name})",
                    address=adapter.wrapped_collateral,
                    kind=CollateralKind.NEGRISK_WRAPPED,
                    adapter=adapter,
                )
            )

        with self._invariants_lock:
            if not self._invariants_verified:
                self._negrisk = tuple(verified)
                self._invariants_verified = True
                # Inference cache is keyed by token IDs so adding new
                # candidates strictly widens the search space; stale
                # negative entries (None) would be incorrect though, so
                # we drop them.
                self._inference_cache = {
                    k: v for k, v in self._inference_cache.items() if v is not None
                }

        logger.info(
            "Polymarket collateral invariants verified",
            adapters=[a.name for a in EXPECTED_NEGRISK_ADAPTERS],
            vanilla=[c.name for c in self._vanilla],
        )

    def invariants_verified(self) -> bool:
        with self._invariants_lock:
            return self._invariants_verified

    # ── Inference ────────────────────────────────────────────────
    async def infer(
        self,
        w3: Any,
        *,
        condition_id: str,
        token_id: int,
        outcome_index_hint: int | None = None,
    ) -> CollateralMatch | None:
        """Resolve the collateral + slot for a held position via on-chain views.

        Returns ``None`` when no registered candidate produces the held
        ``token_id``. The caller MUST treat that as "do not redeem" — a
        guess would either no-op (gas burn) or, worse, target a position
        the wallet doesn't actually own.

        Boot invariants must pass for NegRisk inference to work; this
        method calls ``verify_invariants`` lazily on first use.
        """
        await self.verify_invariants(w3)

        cond_lower = self._normalize_condition_id(condition_id)
        cache_key = (cond_lower, int(token_id))

        async with self._inference_lock:
            if cache_key in self._inference_cache:
                return self._inference_cache[cache_key]

        match = await self._infer_uncached(
            w3,
            condition_id=cond_lower,
            token_id=int(token_id),
            outcome_index_hint=outcome_index_hint,
        )

        async with self._inference_lock:
            self._inference_cache[cache_key] = match
        return match

    async def _infer_uncached(
        self,
        w3: Any,
        *,
        condition_id: str,
        token_id: int,
        outcome_index_hint: int | None,
    ) -> CollateralMatch | None:
        ctf = w3.eth.contract(
            address=w3.to_checksum_address(CTF_ADDRESS),
            abi=list(_CTF_HELPERS_ABI),
        )

        # Slot range determination. Unprepared conditions revert on
        # getOutcomeSlotCount; treat that as "not on this CTF" and skip
        # inference.
        try:
            slot_count = int(
                await asyncio.to_thread(
                    lambda: ctf.functions.getOutcomeSlotCount(condition_id).call()
                )
            )
        except Exception as exc:
            logger.warning(
                "getOutcomeSlotCount failed during collateral inference",
                condition_id=condition_id,
                error=str(exc),
            )
            return None

        if slot_count < 2:
            return None

        # Probe the hinted slot first for the common-case fast path,
        # then the rest in ascending order.
        slot_order: list[int] = []
        if outcome_index_hint is not None and 0 <= int(outcome_index_hint) < slot_count:
            slot_order.append(int(outcome_index_hint))
        for s in range(slot_count):
            if s not in slot_order:
                slot_order.append(s)

        candidates = self.all_candidates()

        for slot in slot_order:
            index_set = 1 << slot
            try:
                collection_id = await asyncio.to_thread(
                    lambda i=index_set: ctf.functions.getCollectionId(
                        _ZERO_BYTES32, condition_id, i
                    ).call()
                )
            except Exception as exc:
                logger.warning(
                    "getCollectionId failed; continuing with other slots",
                    condition_id=condition_id,
                    slot=slot,
                    error=str(exc),
                )
                continue

            for collateral in candidates:
                try:
                    pid = int(
                        await asyncio.to_thread(
                            lambda c=collateral, cid=collection_id: ctf.functions.getPositionId(
                                w3.to_checksum_address(c.address), cid
                            ).call()
                        )
                    )
                except Exception as exc:
                    logger.warning(
                        "getPositionId failed; continuing with other candidates",
                        condition_id=condition_id,
                        slot=slot,
                        collateral=collateral.name,
                        error=str(exc),
                    )
                    continue
                if pid == token_id:
                    return CollateralMatch(
                        collateral=collateral,
                        outcome_slot=slot,
                        index_set=index_set,
                    )

        return None

    # ── Helpers ───────────────────────────────────────────────────
    @staticmethod
    def _normalize_condition_id(raw: str) -> str:
        text = (raw or "").strip().lower()
        if not text.startswith("0x"):
            text = "0x" + text
        if len(text) != 66:
            raise ValueError(f"condition_id must be a 32-byte hex string, got {raw!r}")
        # Fail loudly on non-hex content.
        int(text[2:], 16)
        return text

    # ── Test support ──────────────────────────────────────────────
    def _reset_for_tests(self) -> None:
        """Reset registry + cache. Not for production use."""
        with self._invariants_lock:
            self._invariants_verified = False
            self._negrisk = ()
        self._inference_cache.clear()


collateral_registry = CollateralRegistry()
