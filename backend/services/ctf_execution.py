from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

from config import settings
from services.polymarket import polymarket_client
from services.live_execution_service import live_execution_service
from services.polymarket_collateral import (
    CTF_ADDRESS,
    CollateralKind,
    CollateralToken,
    PUSD_ADDRESS,
    USDC_E_ADDRESS,
    USDC_NATIVE_ADDRESS,
    collateral_registry,
)
from utils.converters import safe_float
from utils.logger import get_logger

logger = get_logger(__name__)

_USDC_DECIMALS = 6
_MAX_UINT256 = 2**256 - 1
_ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
_ZERO_BYTES32 = "0x" + "00" * 32
_MIN_APPROVAL_BUFFER_BASE = 10_000_000  # 10 USDC (6 decimals)
_REDEEMER_DRY_RUN_CONDITION_BUDGET = 25
_REDEEMER_REAL_CONDITION_BUDGET = 50
_NONCE_RACE_ERROR_MARKERS = (
    "replacement transaction underpriced",
    "transaction underpriced",
    "nonce too low",
)


_VALID_DEFAULT_COLLATERALS: dict[str, str] = {
    "pusd": PUSD_ADDRESS,
    "usdc.e": USDC_E_ADDRESS,
    "usdc_native": USDC_NATIVE_ADDRESS,
}


def _resolve_default_collateral_address() -> str:
    """Return the address of the operator-configured default collateral.

    Used by ``split_position``/``merge_positions`` when the caller does
    not specify a collateral explicitly. The default tracks Polymarket's
    current canonical (pUSD post-2026-04 migration) but is operator-
    overridable via the ``POLYMARKET_DEFAULT_COLLATERAL`` setting so
    legacy USDC.e-only deployments stay supported without code changes.
    """
    raw = str(getattr(settings, "POLYMARKET_DEFAULT_COLLATERAL", "") or "").strip().lower()
    if not raw:
        return PUSD_ADDRESS
    return _VALID_DEFAULT_COLLATERALS.get(raw, PUSD_ADDRESS)


@dataclass
class CTFExecutionResult:
    status: str
    action: str
    tx_hash: str | None
    error_message: str | None
    payload: dict[str, Any]


class CTFExecutionService:
    # Source of truth for canonical contract addresses lives in
    # ``services.polymarket_collateral``. Re-exposed as class attrs so
    # call sites and tests can patch them without having to re-import.
    CTF_ADDRESS = CTF_ADDRESS
    CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"

    _CTF_ABI = [
        {
            "name": "splitPosition",
            "type": "function",
            "inputs": [
                {"name": "collateralToken", "type": "address"},
                {"name": "parentCollectionId", "type": "bytes32"},
                {"name": "conditionId", "type": "bytes32"},
                {"name": "partition", "type": "uint256[]"},
                {"name": "amount", "type": "uint256"},
            ],
            "outputs": [],
            "stateMutability": "nonpayable",
        },
        {
            "name": "mergePositions",
            "type": "function",
            "inputs": [
                {"name": "collateralToken", "type": "address"},
                {"name": "parentCollectionId", "type": "bytes32"},
                {"name": "conditionId", "type": "bytes32"},
                {"name": "partition", "type": "uint256[]"},
                {"name": "amount", "type": "uint256"},
            ],
            "outputs": [],
            "stateMutability": "nonpayable",
        },
        {
            "name": "redeemPositions",
            "type": "function",
            "inputs": [
                {"name": "collateralToken", "type": "address"},
                {"name": "parentCollectionId", "type": "bytes32"},
                {"name": "conditionId", "type": "bytes32"},
                {"name": "indexSets", "type": "uint256[]"},
            ],
            "outputs": [],
            "stateMutability": "nonpayable",
        },
        {
            "name": "balanceOf",
            "type": "function",
            "inputs": [
                {"name": "account", "type": "address"},
                {"name": "id", "type": "uint256"},
            ],
            "outputs": [{"name": "", "type": "uint256"}],
            "stateMutability": "view",
        },
        {
            "name": "payoutDenominator",
            "type": "function",
            "inputs": [{"name": "conditionId", "type": "bytes32"}],
            "outputs": [{"name": "", "type": "uint256"}],
            "stateMutability": "view",
        },
        {
            # CTF stores per-outcome-slot payout numerators after resolution.
            # For binary markets (slots 0/1), one will be 1 and the other 0
            # (or scaled so the pair sums to denominator). For scalar
            # markets the slots can carry fractional weights summing to
            # the denominator. Used by the redeemer's expected-payout
            # guard to skip $0-return redemptions that would only burn gas.
            "name": "payoutNumerators",
            "type": "function",
            "inputs": [
                {"name": "conditionId", "type": "bytes32"},
                {"name": "outcomeSlot", "type": "uint256"},
            ],
            "outputs": [{"name": "", "type": "uint256"}],
            "stateMutability": "view",
        },
        {
            "name": "isApprovedForAll",
            "type": "function",
            "inputs": [
                {"name": "account", "type": "address"},
                {"name": "operator", "type": "address"},
            ],
            "outputs": [{"name": "", "type": "bool"}],
            "stateMutability": "view",
        },
        {
            "name": "setApprovalForAll",
            "type": "function",
            "inputs": [
                {"name": "operator", "type": "address"},
                {"name": "approved", "type": "bool"},
            ],
            "outputs": [],
            "stateMutability": "nonpayable",
        },
    ]

    _ERC20_ABI = [
        {
            "name": "allowance",
            "type": "function",
            "inputs": [
                {"name": "owner", "type": "address"},
                {"name": "spender", "type": "address"},
            ],
            "outputs": [{"name": "", "type": "uint256"}],
            "stateMutability": "view",
        },
        {
            "name": "approve",
            "type": "function",
            "inputs": [
                {"name": "spender", "type": "address"},
                {"name": "amount", "type": "uint256"},
            ],
            "outputs": [{"name": "", "type": "bool"}],
            "stateMutability": "nonpayable",
        },
    ]

    # NegRiskAdapter.redeemPositions(bytes32 conditionId, uint256[] amounts).
    # Source verified against
    # https://github.com/Polymarket/neg-risk-ctf-adapter/blob/main/src/NegRiskAdapter.sol
    # — the adapter requires ``setApprovalForAll(adapter, true)`` on the
    # CTF (it pulls the wallet's WCOL positions via ``safeBatchTransferFrom``),
    # then redeems against the inner CTF and unwraps WCOL → parent
    # collateral back to the caller.  ``amounts`` is length-2:
    # ``[yes_amount, no_amount]``.
    _NEG_RISK_ADAPTER_ABI = [
        {
            "name": "redeemPositions",
            "type": "function",
            "inputs": [
                {"name": "_conditionId", "type": "bytes32"},
                {"name": "_amounts", "type": "uint256[]"},
            ],
            "outputs": [],
            "stateMutability": "nonpayable",
        },
    ]

    _SAFE_ABI = [
        {
            "name": "nonce",
            "type": "function",
            "inputs": [],
            "outputs": [{"name": "", "type": "uint256"}],
            "stateMutability": "view",
        },
        {
            "name": "getTransactionHash",
            "type": "function",
            "inputs": [
                {"name": "to", "type": "address"},
                {"name": "value", "type": "uint256"},
                {"name": "data", "type": "bytes"},
                {"name": "operation", "type": "uint8"},
                {"name": "safeTxGas", "type": "uint256"},
                {"name": "baseGas", "type": "uint256"},
                {"name": "gasPrice", "type": "uint256"},
                {"name": "gasToken", "type": "address"},
                {"name": "refundReceiver", "type": "address"},
                {"name": "_nonce", "type": "uint256"},
            ],
            "outputs": [{"name": "", "type": "bytes32"}],
            "stateMutability": "view",
        },
        {
            "name": "execTransaction",
            "type": "function",
            "inputs": [
                {"name": "to", "type": "address"},
                {"name": "value", "type": "uint256"},
                {"name": "data", "type": "bytes"},
                {"name": "operation", "type": "uint8"},
                {"name": "safeTxGas", "type": "uint256"},
                {"name": "baseGas", "type": "uint256"},
                {"name": "gasPrice", "type": "uint256"},
                {"name": "gasToken", "type": "address"},
                {"name": "refundReceiver", "type": "address"},
                {"name": "signatures", "type": "bytes"},
            ],
            "outputs": [{"name": "", "type": "bool"}],
            "stateMutability": "payable",
        },
    ]

    def __init__(self) -> None:
        self._tx_lock = asyncio.Lock()
        self._redeemer_condition_cursor = 0

    def _rpc_candidates(self) -> list[str]:
        candidates: list[str] = []
        for raw in (
            settings.POLYGON_RPC_URL,
            "https://polygon-rpc.com",
            "https://rpc-mainnet.matic.quiknode.pro",
            "https://polygon.gateway.tenderly.co",
        ):
            url = str(raw or "").strip()
            if not url or url in candidates:
                continue
            candidates.append(url)
        return candidates

    async def _get_web3(self):
        from web3 import Web3

        last_error: Exception | None = None
        for rpc_url in self._rpc_candidates():
            try:
                candidate = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 12}))
                await asyncio.to_thread(lambda: candidate.eth.block_number)
                return candidate
            except Exception as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise RuntimeError(f"All Polygon RPC providers failed: {last_error}")
        raise RuntimeError("No Polygon RPC provider configured")

    async def _resolve_wallet_context(self) -> dict[str, str]:
        from eth_account import Account

        private_key, _, _, _, _ = await live_execution_service._resolve_polymarket_credentials()
        if not private_key:
            raise RuntimeError("Missing Polymarket private key for CTF execution")

        eoa = str(getattr(live_execution_service, "_eoa_address", "") or "").strip()
        if not eoa:
            eoa = Account.from_key(private_key).address

        execution_wallet = str(live_execution_service.get_execution_wallet_address() or "").strip()
        if not execution_wallet:
            execution_wallet = eoa

        return {
            "private_key": private_key,
            "eoa_address": eoa,
            "execution_wallet": execution_wallet,
        }

    async def _safe_signature(self, tx_hash_bytes: bytes, private_key: str) -> bytes:
        from eth_keys import keys

        def _sign_digest() -> bytes:
            normalized_key = str(private_key or "").strip()
            if normalized_key.startswith("0x"):
                normalized_key = normalized_key[2:]
            signer = keys.PrivateKey(bytes.fromhex(normalized_key))
            signature = signer.sign_msg_hash(tx_hash_bytes)
            v = int(signature.v) + 27
            return signature.r.to_bytes(32, "big") + signature.s.to_bytes(32, "big") + bytes([v])

        return await asyncio.to_thread(_sign_digest)

    async def _is_safe_wallet(self, w3, wallet_address: str) -> bool:
        try:
            checksum_wallet = w3.to_checksum_address(wallet_address)
            code = await asyncio.to_thread(lambda: w3.eth.get_code(checksum_wallet))
            if not code:
                return False
            safe = w3.eth.contract(address=checksum_wallet, abi=self._SAFE_ABI)
            await asyncio.to_thread(lambda: safe.functions.nonce().call())
            return True
        except Exception:
            return False

    async def _next_sender_nonce(self, w3, address: str) -> int:
        try:
            return int(await asyncio.to_thread(lambda: w3.eth.get_transaction_count(address, "pending")))
        except TypeError:
            return int(await asyncio.to_thread(lambda: w3.eth.get_transaction_count(address)))

    async def _send_eoa_call(
        self,
        *,
        w3,
        from_address: str,
        private_key: str,
        to_address: str,
        data: bytes,
        gas_limit: int,
    ) -> str:
        chain_id = int(getattr(settings, "CHAIN_ID", 137) or 137)

        async with self._tx_lock:
            nonce = await self._next_sender_nonce(w3, from_address)
            gas_price = await asyncio.to_thread(lambda: w3.eth.gas_price)
            tx = {
                "from": from_address,
                "to": to_address,
                "value": 0,
                "data": data,
                "nonce": nonce,
                "gas": int(gas_limit),
                "gasPrice": gas_price,
                "chainId": chain_id,
            }
            signed = await asyncio.to_thread(w3.eth.account.sign_transaction, tx, private_key)
            tx_hash = await asyncio.to_thread(w3.eth.send_raw_transaction, signed.raw_transaction)
            receipt = await asyncio.to_thread(w3.eth.wait_for_transaction_receipt, tx_hash, 120)

        if int(getattr(receipt, "status", 0) or 0) != 1:
            raise RuntimeError("On-chain transaction reverted")
        return tx_hash.hex()

    async def _send_safe_call(
        self,
        *,
        w3,
        safe_address: str,
        owner_eoa: str,
        private_key: str,
        to_address: str,
        data: bytes,
        gas_limit: int,
    ) -> str:
        chain_id = int(getattr(settings, "CHAIN_ID", 137) or 137)
        safe = w3.eth.contract(address=safe_address, abi=self._SAFE_ABI)

        async with self._tx_lock:
            safe_nonce = await asyncio.to_thread(lambda: safe.functions.nonce().call())
            safe_tx_hash = await asyncio.to_thread(
                lambda: safe.functions.getTransactionHash(
                    to_address,
                    0,
                    data,
                    0,
                    0,
                    0,
                    0,
                    _ZERO_ADDRESS,
                    _ZERO_ADDRESS,
                    safe_nonce,
                ).call()
            )
            signature = await self._safe_signature(safe_tx_hash, private_key)

            owner_nonce = await self._next_sender_nonce(w3, owner_eoa)
            gas_price = await asyncio.to_thread(lambda: w3.eth.gas_price)
            tx = safe.functions.execTransaction(
                to_address,
                0,
                data,
                0,
                0,
                0,
                0,
                _ZERO_ADDRESS,
                _ZERO_ADDRESS,
                signature,
            ).build_transaction(
                {
                    "from": owner_eoa,
                    "nonce": owner_nonce,
                    "gas": int(gas_limit),
                    "gasPrice": gas_price,
                    "chainId": chain_id,
                }
            )
            signed = await asyncio.to_thread(w3.eth.account.sign_transaction, tx, private_key)
            tx_hash = await asyncio.to_thread(w3.eth.send_raw_transaction, signed.raw_transaction)
            receipt = await asyncio.to_thread(w3.eth.wait_for_transaction_receipt, tx_hash, 120)

        if int(getattr(receipt, "status", 0) or 0) != 1:
            raise RuntimeError("Safe transaction reverted")
        return tx_hash.hex()

    async def _execute_contract_call(
        self,
        *,
        contract_address: str,
        data: bytes,
        gas_limit: int,
        action: str,
    ) -> CTFExecutionResult:
        try:
            wallet_ctx = await self._resolve_wallet_context()
            w3 = await self._get_web3()
            execution_wallet = w3.to_checksum_address(wallet_ctx["execution_wallet"])
            eoa = w3.to_checksum_address(wallet_ctx["eoa_address"])
            to_address = w3.to_checksum_address(contract_address)

            if await self._is_safe_wallet(w3, execution_wallet):
                tx_hash = await self._send_safe_call(
                    w3=w3,
                    safe_address=execution_wallet,
                    owner_eoa=eoa,
                    private_key=wallet_ctx["private_key"],
                    to_address=to_address,
                    data=data,
                    gas_limit=gas_limit,
                )
            else:
                tx_hash = await self._send_eoa_call(
                    w3=w3,
                    from_address=eoa,
                    private_key=wallet_ctx["private_key"],
                    to_address=to_address,
                    data=data,
                    gas_limit=gas_limit,
                )
            return CTFExecutionResult(
                status="executed",
                action=action,
                tx_hash=tx_hash,
                error_message=None,
                payload={
                    "execution_wallet": wallet_ctx["execution_wallet"],
                    "eoa_address": wallet_ctx["eoa_address"],
                    "contract_address": contract_address,
                },
            )
        except Exception as exc:
            error_message = str(exc)
            normalized_error = error_message.lower()
            if "insufficient funds for gas" in normalized_error:
                logger.warning(
                    "CTF contract call skipped due to insufficient native gas",
                    action=action,
                    error=error_message,
                )
            elif any(marker in normalized_error for marker in _NONCE_RACE_ERROR_MARKERS):
                logger.warning(
                    "CTF contract call deferred due to nonce/gas pricing race",
                    action=action,
                    error=error_message,
                )
            else:
                logger.error("CTF contract call failed", action=action, exc_info=exc)
            return CTFExecutionResult(
                status="failed",
                action=action,
                tx_hash=None,
                error_message=error_message,
                payload={"contract_address": contract_address},
            )

    def _normalize_condition_id(self, raw: Any) -> str:
        text = str(raw or "").strip().lower()
        if not text.startswith("0x"):
            text = f"0x{text}"
        if len(text) != 66:
            raise ValueError("condition_id must be a 32-byte hex string")
        int(text[2:], 16)
        return text

    def _to_base_units(self, amount: float) -> int:
        normalized = max(0.0, float(amount))
        return int(round(normalized * (10**_USDC_DECIMALS)))

    def _parse_token_id_uint256(self, token_id: Any) -> Optional[int]:
        text = str(token_id or "").strip().lower()
        if not text:
            return None
        try:
            if text.startswith("0x"):
                return int(text, 16)
            return int(text)
        except Exception:
            return None

    async def ensure_collateral_approval(
        self,
        *,
        collateral_address: str,
        min_amount_base_units: int = 0,
    ) -> CTFExecutionResult:
        """Ensure CTF has ERC-20 ``approve`` allowance to pull the given collateral.

        Required before splitting collateral into a complete set of
        outcomes. Uses ``_MAX_UINT256`` to amortize the approval cost
        across many splits; the floor of ``_MIN_APPROVAL_BUFFER_BASE``
        ensures we top-up rather than incrementally approve.
        """
        action_label = f"approve_collateral:{(collateral_address or '').lower()}"
        try:
            wallet_ctx = await self._resolve_wallet_context()
            w3 = await self._get_web3()
            owner = w3.to_checksum_address(wallet_ctx["execution_wallet"])
            collateral_checksum = w3.to_checksum_address(collateral_address)
            ctf_address = w3.to_checksum_address(self.CTF_ADDRESS)
            erc20 = w3.eth.contract(address=collateral_checksum, abi=self._ERC20_ABI)
            allowance = await asyncio.to_thread(
                lambda: erc20.functions.allowance(owner, ctf_address).call()
            )
            required = max(int(min_amount_base_units), _MIN_APPROVAL_BUFFER_BASE)
            if int(allowance or 0) >= required:
                return CTFExecutionResult(
                    status="executed",
                    action=action_label,
                    tx_hash=None,
                    error_message=None,
                    payload={
                        "collateral_address": collateral_address,
                        "allowance": int(allowance),
                        "required": required,
                        "already_approved": True,
                    },
                )

            data = erc20.functions.approve(ctf_address, _MAX_UINT256)._encode_transaction_data()
            result = await self._execute_contract_call(
                contract_address=collateral_address,
                data=data,
                gas_limit=140_000,
                action=action_label,
            )
            if result.status == "executed":
                result.payload.update(
                    {
                        "collateral_address": collateral_address,
                        "required": required,
                        "already_approved": False,
                    }
                )
            return result
        except Exception as exc:
            logger.error(
                "Collateral approval check failed",
                collateral_address=collateral_address,
                exc_info=exc,
            )
            return CTFExecutionResult(
                status="failed",
                action=action_label,
                tx_hash=None,
                error_message=str(exc),
                payload={"collateral_address": collateral_address},
            )

    async def _ensure_ctf_operator_approval(
        self,
        *,
        operator_address: str,
        action: str,
    ) -> CTFExecutionResult:
        """Idempotent ``setApprovalForAll(operator, true)`` on the CTF.

        Required for any contract that needs to pull the wallet's
        ERC-1155 positions on its behalf — the CTF Exchange (existing
        order matching path) and any NegRiskAdapter we redeem through
        (it pulls WCOL positions in its ``redeemPositions``).
        """
        try:
            wallet_ctx = await self._resolve_wallet_context()
            w3 = await self._get_web3()
            owner = w3.to_checksum_address(wallet_ctx["execution_wallet"])
            ctf_address = w3.to_checksum_address(self.CTF_ADDRESS)
            operator_checksum = w3.to_checksum_address(operator_address)
            ctf = w3.eth.contract(address=ctf_address, abi=self._CTF_ABI)
            approved = await asyncio.to_thread(
                lambda: ctf.functions.isApprovedForAll(owner, operator_checksum).call()
            )
            if bool(approved):
                return CTFExecutionResult(
                    status="executed",
                    action=action,
                    tx_hash=None,
                    error_message=None,
                    payload={"already_approved": True, "operator": operator_address},
                )

            data = ctf.functions.setApprovalForAll(operator_checksum, True)._encode_transaction_data()
            result = await self._execute_contract_call(
                contract_address=self.CTF_ADDRESS,
                data=data,
                gas_limit=170_000,
                action=action,
            )
            if result.status == "executed":
                result.payload.update({"already_approved": False, "operator": operator_address})
            return result
        except Exception as exc:
            logger.error(
                "CTF operator approval failed",
                operator=operator_address,
                action=action,
                exc_info=exc,
            )
            return CTFExecutionResult(
                status="failed",
                action=action,
                tx_hash=None,
                error_message=str(exc),
                payload={"operator": operator_address},
            )

    async def ensure_exchange_approval(self) -> CTFExecutionResult:
        return await self._ensure_ctf_operator_approval(
            operator_address=self.CTF_EXCHANGE,
            action="approve_exchange",
        )

    async def get_native_gas_affordability(self, *, gas_limit: int) -> dict[str, Any]:
        try:
            wallet_ctx = await self._resolve_wallet_context()
            w3 = await self._get_web3()
            eoa_address = w3.to_checksum_address(wallet_ctx["eoa_address"])
            balance_wei = int(await asyncio.to_thread(lambda: w3.eth.get_balance(eoa_address)) or 0)
            gas_price_wei = int(await asyncio.to_thread(lambda: w3.eth.gas_price) or 0)
            required_wei = max(0, int(gas_limit)) * max(0, gas_price_wei)
            return {
                "affordable": balance_wei >= required_wei,
                "balance_wei": balance_wei,
                "required_wei": required_wei,
                "gas_price_wei": gas_price_wei,
                "wallet_address": eoa_address,
            }
        except Exception as exc:
            return {
                "affordable": False,
                "balance_wei": 0,
                "required_wei": 0,
                "gas_price_wei": 0,
                "wallet_address": "",
                "error": str(exc),
            }

    async def split_position(
        self,
        *,
        condition_id: str,
        amount_usd: float,
        collateral_address: str | None = None,
    ) -> CTFExecutionResult:
        """Split parent collateral into a complete set of binary outcome shares.

        ``collateral_address`` defaults to the operator-configured
        ``POLYMARKET_DEFAULT_COLLATERAL`` (currently pUSD post-2026-04
        migration). Pass an explicit address to override — e.g. when
        operating against a legacy USDC.e-collateralized condition.
        """
        normalized_condition_id = self._normalize_condition_id(condition_id)
        amount = max(0.0, safe_float(amount_usd, 0.0) or 0.0)
        if amount <= 0.0:
            return CTFExecutionResult(
                status="failed",
                action="split",
                tx_hash=None,
                error_message="amount_usd must be greater than zero",
                payload={"condition_id": normalized_condition_id},
            )

        collateral = (collateral_address or _resolve_default_collateral_address()).strip()

        amount_base = self._to_base_units(amount)
        approval = await self.ensure_collateral_approval(
            collateral_address=collateral,
            min_amount_base_units=amount_base,
        )
        if approval.status != "executed":
            return approval

        exchange_approval = await self.ensure_exchange_approval()
        if exchange_approval.status != "executed":
            return exchange_approval

        w3 = await self._get_web3()
        ctf = w3.eth.contract(address=w3.to_checksum_address(self.CTF_ADDRESS), abi=self._CTF_ABI)
        data = ctf.functions.splitPosition(
            w3.to_checksum_address(collateral),
            _ZERO_BYTES32,
            normalized_condition_id,
            [1, 2],
            amount_base,
        )._encode_transaction_data()
        result = await self._execute_contract_call(
            contract_address=self.CTF_ADDRESS,
            data=data,
            gas_limit=320_000,
            action="split",
        )
        result.payload.update(
            {
                "condition_id": normalized_condition_id,
                "collateral_address": collateral,
                "amount_usd": amount,
                "amount_base_units": amount_base,
                "shares_per_side": amount,
            }
        )
        return result

    async def merge_positions(
        self,
        *,
        condition_id: str,
        shares_per_side: float,
        collateral_address: str | None = None,
    ) -> CTFExecutionResult:
        """Merge a complete set of binary outcomes back to parent collateral.

        See ``split_position`` for the ``collateral_address`` contract.
        """
        normalized_condition_id = self._normalize_condition_id(condition_id)
        shares = max(0.0, safe_float(shares_per_side, 0.0) or 0.0)
        if shares <= 0.0:
            return CTFExecutionResult(
                status="failed",
                action="merge",
                tx_hash=None,
                error_message="shares_per_side must be greater than zero",
                payload={"condition_id": normalized_condition_id},
            )

        collateral = (collateral_address or _resolve_default_collateral_address()).strip()

        amount_base = self._to_base_units(shares)
        w3 = await self._get_web3()
        ctf = w3.eth.contract(address=w3.to_checksum_address(self.CTF_ADDRESS), abi=self._CTF_ABI)
        data = ctf.functions.mergePositions(
            w3.to_checksum_address(collateral),
            _ZERO_BYTES32,
            normalized_condition_id,
            [1, 2],
            amount_base,
        )._encode_transaction_data()
        result = await self._execute_contract_call(
            contract_address=self.CTF_ADDRESS,
            data=data,
            gas_limit=280_000,
            action="merge",
        )
        result.payload.update(
            {
                "condition_id": normalized_condition_id,
                "collateral_address": collateral,
                "shares_per_side": shares,
                "amount_base_units": amount_base,
            }
        )
        return result

    # ── Redemption (auto + manual) ────────────────────────────────
    #
    # Redemption goes through one of two contract paths based on the
    # collateral that minted the position:
    #
    #   * Vanilla (USDC.e, pUSD, USDC native): call CTF.redeemPositions
    #     with the correct ``collateralToken`` arg.  No approval needed —
    #     the CTF burns the wallet's own ERC-1155 positions in place.
    #
    #   * NegRisk-wrapped (WCOL): call ``NegRiskAdapter.redeemPositions``
    #     which pulls the wallet's WCOL positions via
    #     ``ctf.safeBatchTransferFrom`` (so the adapter must have
    #     ``setApprovalForAll`` from the wallet on the CTF), redeems
    #     against the inner CTF, and unwraps WCOL → parent collateral.
    #     The amounts arg is fixed length-2 ``[yes, no]`` per the source.
    #
    # Calling vanilla with the wrong collateral, or vanilla on a NegRisk
    # position, silently no-ops and leaves resolved shares unredeemable.
    # Inference via the collateral registry is therefore mandatory.

    async def _redeem_via_vanilla_ctf(
        self,
        *,
        w3,
        collateral_address: str,
        condition_id: str,
        index_sets: list[int],
    ) -> CTFExecutionResult:
        ctf = w3.eth.contract(
            address=w3.to_checksum_address(self.CTF_ADDRESS),
            abi=self._CTF_ABI,
        )
        data = ctf.functions.redeemPositions(
            w3.to_checksum_address(collateral_address),
            _ZERO_BYTES32,
            condition_id,
            index_sets,
        )._encode_transaction_data()
        result = await self._execute_contract_call(
            contract_address=self.CTF_ADDRESS,
            data=data,
            gas_limit=260_000,
            action="redeem",
        )
        result.payload.update(
            {
                "condition_id": condition_id,
                "collateral_address": collateral_address,
                "index_sets": list(index_sets),
                "redemption_path": "ctf_vanilla",
            }
        )
        return result

    async def _redeem_via_negrisk_adapter(
        self,
        *,
        w3,
        adapter_address: str,
        condition_id: str,
        amounts_yes_no_base_units: tuple[int, int],
    ) -> CTFExecutionResult:
        # The adapter pulls our WCOL positions via safeBatchTransferFrom,
        # so it must have setApprovalForAll on the CTF first.
        approval = await self._ensure_ctf_operator_approval(
            operator_address=adapter_address,
            action="approve_negrisk_adapter",
        )
        if approval.status != "executed":
            return approval

        adapter = w3.eth.contract(
            address=w3.to_checksum_address(adapter_address),
            abi=self._NEG_RISK_ADAPTER_ABI,
        )
        amounts = [int(amounts_yes_no_base_units[0]), int(amounts_yes_no_base_units[1])]
        data = adapter.functions.redeemPositions(
            condition_id,
            amounts,
        )._encode_transaction_data()
        # NegRisk redeem does more than vanilla CTF (transfer batch +
        # inner redeem + unwrap), so its gas envelope is wider.
        result = await self._execute_contract_call(
            contract_address=adapter_address,
            data=data,
            gas_limit=400_000,
            action="redeem",
        )
        result.payload.update(
            {
                "condition_id": condition_id,
                "adapter_address": adapter_address,
                "amounts_yes_no": amounts,
                "redemption_path": "negrisk_adapter",
            }
        )
        return result

    async def _resolve_balances_for_condition(
        self,
        *,
        w3,
        wallet_checksum: str,
        condition_id: str,
        candidates: tuple[CollateralToken, ...],
    ) -> tuple[CollateralToken | None, dict[int, int]]:
        """Probe ``balanceOf`` on every (candidate collateral × slot) for a binary condition.

        Returns the unique collateral that holds a non-zero balance plus
        the per-slot raw balances. Used by the operator-driven
        ``redeem_positions`` path where the caller knows the condition
        but not the collateral. Two view calls per candidate × 2 slots
        is dirt cheap and gives provably-correct dispatch.
        """
        ctf = w3.eth.contract(
            address=w3.to_checksum_address(self.CTF_ADDRESS),
            abi=self._CTF_ABI,
        )

        # Build collection IDs for the binary partition once.
        slot_collection_ids: dict[int, bytes] = {}
        for slot in (0, 1):
            try:
                slot_collection_ids[slot] = await asyncio.to_thread(
                    lambda s=slot: ctf.functions.getCollectionId(
                        _ZERO_BYTES32, condition_id, 1 << s
                    ).call()
                )
            except Exception as exc:
                logger.warning(
                    "getCollectionId failed during redeem balance probe",
                    condition_id=condition_id,
                    slot=slot,
                    error=str(exc),
                )
                return None, {}

        for candidate in candidates:
            balances: dict[int, int] = {}
            for slot, collection_id in slot_collection_ids.items():
                try:
                    pid = int(
                        await asyncio.to_thread(
                            lambda c=candidate, cid=collection_id: ctf.functions.getPositionId(
                                w3.to_checksum_address(c.address), cid
                            ).call()
                        )
                    )
                    bal = int(
                        await asyncio.to_thread(
                            lambda token=pid: ctf.functions.balanceOf(
                                wallet_checksum, token
                            ).call()
                        )
                    )
                except Exception as exc:
                    logger.warning(
                        "balanceOf probe failed during redeem discovery",
                        condition_id=condition_id,
                        slot=slot,
                        collateral=candidate.name,
                        error=str(exc),
                    )
                    bal = 0
                balances[slot] = bal
            if any(b > 0 for b in balances.values()):
                return candidate, balances

        return None, {}

    async def redeem_positions(
        self,
        *,
        condition_id: str,
        index_sets: list[int] | None = None,
    ) -> CTFExecutionResult:
        """Redeem all of the wallet's holdings on a condition.

        Discovers the collateral by probing CTF ``balanceOf`` against
        every registered candidate, then dispatches:

          * Vanilla → ``CTF.redeemPositions(collateral, 0x0, conditionId, indexSets)``
          * NegRisk → ``NegRiskAdapter.redeemPositions(conditionId, [yes, no])``

        Returns a ``failed`` result with structured detail if no
        collateral candidate produces non-zero shares — never silently
        attempts a redemption that would no-op.
        """
        normalized_condition_id = self._normalize_condition_id(condition_id)
        normalized_sets = [int(value) for value in (index_sets or [1, 2]) if int(value) > 0]
        if not normalized_sets:
            normalized_sets = [1, 2]

        w3 = await self._get_web3()

        # Boot invariants must pass before we touch NegRisk routing.
        # verify_invariants is idempotent and cheap on the hot path.
        try:
            await collateral_registry.verify_invariants(w3)
        except Exception as exc:
            return CTFExecutionResult(
                status="failed",
                action="redeem",
                tx_hash=None,
                error_message=f"collateral_invariants_violated:{exc}",
                payload={"condition_id": normalized_condition_id},
            )

        wallet_ctx = await self._resolve_wallet_context()
        wallet_checksum = w3.to_checksum_address(wallet_ctx["execution_wallet"])

        match_collateral, balances = await self._resolve_balances_for_condition(
            w3=w3,
            wallet_checksum=wallet_checksum,
            condition_id=normalized_condition_id,
            candidates=collateral_registry.all_candidates(),
        )

        if match_collateral is None or not any(b > 0 for b in balances.values()):
            return CTFExecutionResult(
                status="failed",
                action="redeem",
                tx_hash=None,
                error_message="no_redeemable_balance_for_any_known_collateral",
                payload={
                    "condition_id": normalized_condition_id,
                    "candidates_tried": [c.name for c in collateral_registry.all_candidates()],
                },
            )

        if match_collateral.kind == CollateralKind.VANILLA:
            return await self._redeem_via_vanilla_ctf(
                w3=w3,
                collateral_address=match_collateral.address,
                condition_id=normalized_condition_id,
                index_sets=normalized_sets,
            )

        # NEGRISK_WRAPPED — pass the wallet's per-slot balances as
        # [yes (slot 0), no (slot 1)]. The adapter requires both, even if
        # one is zero.
        adapter = match_collateral.adapter
        if adapter is None:
            return CTFExecutionResult(
                status="failed",
                action="redeem",
                tx_hash=None,
                error_message="negrisk_collateral_missing_adapter_metadata",
                payload={
                    "condition_id": normalized_condition_id,
                    "collateral": match_collateral.name,
                },
            )
        return await self._redeem_via_negrisk_adapter(
            w3=w3,
            adapter_address=adapter.address,
            condition_id=normalized_condition_id,
            amounts_yes_no_base_units=(int(balances.get(0, 0)), int(balances.get(1, 0))),
        )

    @staticmethod
    def compute_condition_payout_breakdown(
        *,
        denominator: int,
        outcome_balances: dict[int, float],
        outcome_numerators: dict[int, int],
    ) -> dict[str, float]:
        """Pure-function payout math used by the redeemer guard.

        Given the on-chain ``denominator`` and per-outcome-slot
        ``numerators`` from ``payoutNumerators(conditionId, slot)``,
        plus the wallet's per-slot ``balances`` (already in human shares,
        NOT raw uint256), returns a breakdown of expected USDC payout if
        we redeem now.

        Math (binary or scalar): for each held outcome slot,
        ``payout_per_share = numerator[slot] / denominator``, and the
        condition payout is the sum of ``balance[slot] * payout_per_share``
        across every slot the wallet holds.

        For a fully-losing binary position, the held slot has
        ``numerator = 0``, so ``expected_payout_usd = 0`` and the redeemer
        will skip unless the operator force-redeems. Extracted to a static
        method so the math has unit tests independent of web3/RPC mocks.
        """
        denominator_f = float(denominator or 0)
        if denominator_f <= 0:
            return {
                "expected_payout_usd": 0.0,
                "total_shares": float(sum(outcome_balances.values()) or 0.0),
                "winning_shares": 0.0,
                "losing_shares": float(sum(outcome_balances.values()) or 0.0),
            }
        expected_payout_usd = 0.0
        winning_shares = 0.0
        losing_shares = 0.0
        for slot, balance in outcome_balances.items():
            numerator = float(outcome_numerators.get(int(slot), 0) or 0)
            if numerator > 0:
                expected_payout_usd += float(balance) * (numerator / denominator_f)
                winning_shares += float(balance)
            else:
                losing_shares += float(balance)
        return {
            "expected_payout_usd": expected_payout_usd,
            "total_shares": winning_shares + losing_shares,
            "winning_shares": winning_shares,
            "losing_shares": losing_shares,
        }

    async def fetch_position_chain_status(
        self,
        *,
        wallet_address: str,
        token_id: str,
        condition_id: str,
        outcome_index: int,
    ) -> dict[str, Any]:
        """Read-only on-chain truth for a single conditional-token holding.

        Returns a structured dict with the wallet's CTF balance, the
        market's resolution state, and the deterministic redeemable
        payout if redemption fired right now.

        This is the institutional-grade truth primitive: it talks to
        Polygon mainnet directly (NOT Polymarket's data API), so the
        answer is exactly what an auditor would compute from on-chain
        evidence.  Callers (the stuck-position monitor, the
        manual-writeoff API, the verifier's resolution path) should
        prefer this over polymarket_client when they need certainty
        rather than UI-grade freshness.

        Returned shape::

          {
            "wallet_address":        str (lowercase),
            "token_id":              str,
            "condition_id":          str (0x-prefixed bytes32),
            "outcome_index":         int,
            "wallet_balance_shares": float,   # ERC1155 balanceOf / 1e6
            "market_resolved":       bool,    # payoutDenominator > 0
            "payout_denominator":    int,
            "payout_numerator":      int,     # for THIS outcome slot
            "winning":               bool|None,  # numerator > 0
            "expected_payout_usdc":  float,   # balance * num/den
            "block_number":          int,     # for audit traceability
            "error":                 str|None,
          }

        ALL numerical values come from chain reads, NOT data-API.  No
        caching at this layer — the caller is responsible for not
        spamming this method.
        """
        normalized_wallet = (wallet_address or "").strip().lower()
        normalized_token = (token_id or "").strip()
        normalized_condition = (condition_id or "").strip().lower()
        if not normalized_wallet or not normalized_token or not normalized_condition:
            return {
                "wallet_address": normalized_wallet,
                "token_id": normalized_token,
                "condition_id": normalized_condition,
                "outcome_index": int(outcome_index),
                "wallet_balance_shares": 0.0,
                "market_resolved": False,
                "payout_denominator": 0,
                "payout_numerator": 0,
                "winning": None,
                "expected_payout_usdc": 0.0,
                "block_number": 0,
                "error": "missing_required_input",
            }
        if not normalized_condition.startswith("0x") or len(normalized_condition) != 66:
            return {
                "wallet_address": normalized_wallet,
                "token_id": normalized_token,
                "condition_id": normalized_condition,
                "outcome_index": int(outcome_index),
                "wallet_balance_shares": 0.0,
                "market_resolved": False,
                "payout_denominator": 0,
                "payout_numerator": 0,
                "winning": None,
                "expected_payout_usdc": 0.0,
                "block_number": 0,
                "error": "invalid_condition_id_format",
            }
        try:
            token_id_uint = int(normalized_token)
        except (TypeError, ValueError):
            return {
                "wallet_address": normalized_wallet,
                "token_id": normalized_token,
                "condition_id": normalized_condition,
                "outcome_index": int(outcome_index),
                "wallet_balance_shares": 0.0,
                "market_resolved": False,
                "payout_denominator": 0,
                "payout_numerator": 0,
                "winning": None,
                "expected_payout_usdc": 0.0,
                "block_number": 0,
                "error": "invalid_token_id_format",
            }

        try:
            w3 = await self._get_web3()
        except Exception as exc:
            return {
                "wallet_address": normalized_wallet,
                "token_id": normalized_token,
                "condition_id": normalized_condition,
                "outcome_index": int(outcome_index),
                "wallet_balance_shares": 0.0,
                "market_resolved": False,
                "payout_denominator": 0,
                "payout_numerator": 0,
                "winning": None,
                "expected_payout_usdc": 0.0,
                "block_number": 0,
                "error": f"rpc_unavailable:{exc}",
            }

        ctf = w3.eth.contract(
            address=w3.to_checksum_address(self.CTF_ADDRESS),
            abi=self._CTF_ABI,
        )
        checksum_wallet = w3.to_checksum_address(normalized_wallet)

        try:
            block_number, raw_balance, denominator, numerator = await asyncio.gather(
                asyncio.to_thread(lambda: int(w3.eth.block_number)),
                asyncio.to_thread(
                    lambda: int(ctf.functions.balanceOf(checksum_wallet, token_id_uint).call())
                ),
                asyncio.to_thread(
                    lambda: int(ctf.functions.payoutDenominator(normalized_condition).call())
                ),
                asyncio.to_thread(
                    lambda: int(
                        ctf.functions.payoutNumerators(
                            normalized_condition, int(outcome_index)
                        ).call()
                    )
                ),
            )
        except Exception as exc:
            return {
                "wallet_address": normalized_wallet,
                "token_id": normalized_token,
                "condition_id": normalized_condition,
                "outcome_index": int(outcome_index),
                "wallet_balance_shares": 0.0,
                "market_resolved": False,
                "payout_denominator": 0,
                "payout_numerator": 0,
                "winning": None,
                "expected_payout_usdc": 0.0,
                "block_number": 0,
                "error": f"rpc_call_failed:{type(exc).__name__}:{exc}",
            }

        balance_shares = float(raw_balance) / float(10**_USDC_DECIMALS)
        market_resolved = denominator > 0
        winning: bool | None
        if not market_resolved:
            winning = None
            expected_payout = 0.0
        else:
            winning = numerator > 0
            if denominator > 0 and balance_shares > 0:
                expected_payout = balance_shares * (float(numerator) / float(denominator))
            else:
                expected_payout = 0.0

        return {
            "wallet_address": normalized_wallet,
            "token_id": normalized_token,
            "condition_id": normalized_condition,
            "outcome_index": int(outcome_index),
            "wallet_balance_shares": balance_shares,
            "market_resolved": market_resolved,
            "payout_denominator": denominator,
            "payout_numerator": numerator,
            "winning": winning,
            "expected_payout_usdc": expected_payout,
            "block_number": block_number,
            "error": None,
        }

    async def _gas_price_gwei(self, w3) -> float:
        # Lightweight TTL cache so a busy redeemer doesn't hammer the RPC.
        # Class-level so cache survives across calls within the process.
        import time as _time

        cache = getattr(self, "_gas_price_cache", None)
        ttl_seconds = 30.0
        now = _time.monotonic()
        if isinstance(cache, tuple) and (now - cache[0]) < ttl_seconds:
            return float(cache[1])
        try:
            wei = await asyncio.to_thread(lambda: w3.eth.gas_price)
        except Exception:
            return 0.0
        gwei = float(wei or 0) / 1e9
        self._gas_price_cache = (now, gwei)
        return gwei

    async def redeem_resolved_wallet_positions(
        self,
        *,
        wallet_address: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        execution_wallet = (
            str(wallet_address or live_execution_service.get_execution_wallet_address() or "").strip().lower()
        )
        if not execution_wallet:
            return {
                "wallet_address": "",
                "positions_scanned": 0,
                "conditions_checked": 0,
                "resolved_conditions": 0,
                "redeemable_value_usd": 0.0,
                "redeemed": 0,
                "skipped_low_payout": 0,
                "skipped_high_gas": 0,
                "skipped_unknown_collateral": 0,
                "failed": 0,
                "dry_run": bool(dry_run),
                "errors": ["missing_execution_wallet"],
            }

        positions = await polymarket_client.get_wallet_positions(execution_wallet)
        if not positions:
            return {
                "wallet_address": execution_wallet,
                "positions_scanned": 0,
                "conditions_checked": 0,
                "resolved_conditions": 0,
                "redeemable_value_usd": 0.0,
                "redeemed": 0,
                "skipped_low_payout": 0,
                "skipped_high_gas": 0,
                "skipped_unknown_collateral": 0,
                "failed": 0,
                "dry_run": bool(dry_run),
                "errors": [],
            }

        # Group positions by conditionId, keeping per-token outcomeIndex so
        # the payout math knows which slot each token represents.
        @dataclass
        class _HeldOutcome:
            token_id_uint: int
            outcome_index: int

        grouped: dict[str, list[_HeldOutcome]] = {}
        for row in positions:
            condition_id_raw = row.get("conditionId") or row.get("condition_id") or row.get("market")
            token_id_raw = row.get("asset") or row.get("asset_id") or row.get("token_id") or row.get("tokenId")
            outcome_index_raw = row.get("outcomeIndex")
            try:
                condition_id = self._normalize_condition_id(condition_id_raw)
            except Exception:
                continue
            token_id_uint = self._parse_token_id_uint256(token_id_raw)
            if token_id_uint is None:
                continue
            try:
                outcome_index = int(outcome_index_raw) if outcome_index_raw is not None else 0
            except (TypeError, ValueError):
                outcome_index = 0
            grouped.setdefault(condition_id, []).append(
                _HeldOutcome(token_id_uint=token_id_uint, outcome_index=outcome_index)
            )

        if not grouped:
            return {
                "wallet_address": execution_wallet,
                "positions_scanned": len(positions),
                "conditions_checked": 0,
                "resolved_conditions": 0,
                "redeemable_value_usd": 0.0,
                "redeemed": 0,
                "skipped_low_payout": 0,
                "skipped_high_gas": 0,
                "skipped_unknown_collateral": 0,
                "failed": 0,
                "dry_run": bool(dry_run),
                "errors": [],
            }

        w3 = await self._get_web3()

        # Boot invariants must pass before we can dispatch NegRisk
        # redemptions; verify_invariants is idempotent and refuses to
        # succeed if the on-chain adapter state has drifted from the
        # registry. Failure here aborts the cycle loudly — the redeemer
        # worker surfaces the error to operator dashboards rather than
        # silently mis-routing redemptions.
        try:
            await collateral_registry.verify_invariants(w3)
        except Exception as exc:
            return {
                "wallet_address": execution_wallet,
                "positions_scanned": len(positions),
                "conditions_checked": 0,
                "resolved_conditions": 0,
                "redeemable_value_usd": 0.0,
                "redeemed": 0,
                "skipped_low_payout": 0,
                "skipped_high_gas": 0,
                "skipped_unknown_collateral": 0,
                "failed": 0,
                "dry_run": bool(dry_run),
                "errors": [f"collateral_invariants_violated:{exc}"],
            }

        ctf = w3.eth.contract(address=w3.to_checksum_address(self.CTF_ADDRESS), abi=self._CTF_ABI)
        checksum_wallet = w3.to_checksum_address(execution_wallet)

        # Operator-tunable policy (config + DB-overridable; see
        # config.Settings + alembic 202604280001).
        min_payout_usd = float(getattr(settings, "REDEEMER_MIN_PAYOUT_USD", 0.10) or 0.0)
        max_gas_price_gwei = float(getattr(settings, "REDEEMER_MAX_GAS_PRICE_GWEI", 200.0) or 0.0)
        force_losers = bool(getattr(settings, "REDEEMER_FORCE_INCLUDING_LOSERS", False))

        # Snapshot gas price once per cycle (cached) — defers cleanup
        # to a cheaper window when network is hot, without aborting the
        # whole cycle (we can still redeem high-value winners).
        gas_price_gwei = await self._gas_price_gwei(w3) if max_gas_price_gwei > 0 else 0.0
        gas_too_hot = max_gas_price_gwei > 0 and gas_price_gwei > max_gas_price_gwei

        redeemed = 0
        skipped_low_payout = 0
        skipped_high_gas = 0
        skipped_unknown_collateral = 0
        failed = 0
        resolved = 0
        redeemable_value_usd = 0.0
        errors: list[str] = []
        grouped_items = list(grouped.items())
        condition_budget = (
            _REDEEMER_DRY_RUN_CONDITION_BUDGET
            if dry_run
            else _REDEEMER_REAL_CONDITION_BUDGET
        )
        if len(grouped_items) > condition_budget:
            start = self._redeemer_condition_cursor % len(grouped_items)
            rotated = grouped_items[start:] + grouped_items[:start]
            grouped_items = rotated[:condition_budget]
            self._redeemer_condition_cursor = (start + len(grouped_items)) % len(grouped)

        for condition_id, holdings in grouped_items:
            try:
                denominator = await asyncio.to_thread(
                    lambda: ctf.functions.payoutDenominator(condition_id).call()
                )
                if int(denominator or 0) <= 0:
                    continue
                resolved += 1

                # Infer the collateral that backs this condition's
                # positions from on-chain math. All holdings under one
                # condition share a collateral by construction — we
                # infer once using the first held token + outcomeIndex
                # hint and trust the chain-derived slot for the rest.
                # Inference is cached, so re-asking on the next cycle
                # is free.
                first = holdings[0]
                try:
                    match = await collateral_registry.infer(
                        w3,
                        condition_id=condition_id,
                        token_id=first.token_id_uint,
                        outcome_index_hint=first.outcome_index,
                    )
                except Exception as exc:
                    failed += 1
                    errors.append(f"collateral_inference_error:{condition_id}:{exc}")
                    continue
                if match is None:
                    skipped_unknown_collateral += 1
                    errors.append(f"unknown_collateral:{condition_id}:tokens={len(holdings)}")
                    logger.warning(
                        "Skipping resolved condition with unknown collateral",
                        condition_id=condition_id,
                        wallet=execution_wallet,
                        token_id=str(first.token_id_uint),
                        candidates=[c.name for c in collateral_registry.all_candidates()],
                    )
                    continue

                # Read per-slot numerators (only the slots we hold) and
                # per-token balances. Use the chain-derived slot from
                # inference for the first held token; for the rest, keep
                # the data-API outcome_index — it is just a hint and we
                # always go through balanceOf(token_id) for the actual
                # value, so a mis-labelled outcome_index is at worst a
                # cosmetic key in the breakdown (won't change payout).
                slots_held: set[int] = {match.outcome_slot}
                for held in holdings[1:]:
                    slots_held.add(int(held.outcome_index))

                outcome_numerators: dict[int, int] = {}
                for slot in sorted(slots_held):
                    try:
                        n = await asyncio.to_thread(
                            lambda s=slot: ctf.functions.payoutNumerators(condition_id, s).call()
                        )
                        outcome_numerators[slot] = int(n or 0)
                    except Exception as exc:
                        # Treat unreadable slot as zero-payout; we'd rather
                        # under-redeem than burn gas on a bad estimate.
                        outcome_numerators[slot] = 0
                        errors.append(f"numerator_read_failed:{condition_id}:slot={slot}:{exc}")

                # Per-slot raw balances (chain truth, not data-API mark).
                # Track raw uint256 balances for NegRisk amounts arg, and
                # human shares for the payout breakdown.
                outcome_raw_balances: dict[int, int] = {}
                outcome_balances: dict[int, float] = {}
                for idx, held in enumerate(holdings):
                    held_slot = match.outcome_slot if idx == 0 else int(held.outcome_index)
                    raw_balance = int(
                        await asyncio.to_thread(
                            lambda token=held.token_id_uint: ctf.functions.balanceOf(
                                checksum_wallet, token
                            ).call()
                        )
                        or 0
                    )
                    outcome_raw_balances[held_slot] = (
                        outcome_raw_balances.get(held_slot, 0) + raw_balance
                    )
                    shares = float(raw_balance) / (10**_USDC_DECIMALS)
                    outcome_balances[held_slot] = (
                        outcome_balances.get(held_slot, 0.0) + shares
                    )

                breakdown = self.compute_condition_payout_breakdown(
                    denominator=int(denominator),
                    outcome_balances=outcome_balances,
                    outcome_numerators=outcome_numerators,
                )
                expected_payout_usd = breakdown["expected_payout_usd"]
                total_shares = breakdown["total_shares"]
                redeemable_value_usd += expected_payout_usd

                # Wallets without dust shouldn't be on the list; if all
                # balances are zero we already redeemed at some point.
                if total_shares <= 1e-6:
                    continue

                # Decision matrix
                will_redeem = False
                skip_reason: str | None = None
                if expected_payout_usd >= min_payout_usd:
                    if gas_too_hot:
                        will_redeem = False
                        skip_reason = (
                            f"gas_price_too_high gwei={gas_price_gwei:.2f} ceiling={max_gas_price_gwei:.2f}"
                        )
                        skipped_high_gas += 1
                    else:
                        will_redeem = True
                elif force_losers:
                    if gas_too_hot:
                        will_redeem = False
                        skip_reason = (
                            f"gas_price_too_high (force-losers cycle) gwei={gas_price_gwei:.2f} "
                            f"ceiling={max_gas_price_gwei:.2f}"
                        )
                        skipped_high_gas += 1
                    else:
                        will_redeem = True  # dust-cleanup mode, gas burn accepted
                else:
                    will_redeem = False
                    skip_reason = (
                        f"expected_payout_below_floor "
                        f"payout=${expected_payout_usd:.4f} floor=${min_payout_usd:.4f} "
                        f"shares_winning={breakdown['winning_shares']:.4f} "
                        f"shares_losing={breakdown['losing_shares']:.4f}"
                    )
                    skipped_low_payout += 1

                logger.info(
                    "redeemer.decision condition=%s collateral=%s payout_usd=%.4f total_shares=%.4f "
                    "winning_shares=%.4f losing_shares=%.4f redeem=%s reason=%s gas_gwei=%.2f dry_run=%s",
                    condition_id,
                    match.collateral.name,
                    expected_payout_usd,
                    total_shares,
                    breakdown["winning_shares"],
                    breakdown["losing_shares"],
                    bool(will_redeem),
                    skip_reason or "above_floor",
                    gas_price_gwei,
                    bool(dry_run),
                )

                if not will_redeem or dry_run:
                    if dry_run and will_redeem:
                        # In dry-run we still want the "would have redeemed"
                        # counter so operators see what a requested run will do.
                        redeemed += 1
                    continue

                # Dispatch redemption to the contract path that matches
                # the inferred collateral.
                if match.collateral.kind == CollateralKind.VANILLA:
                    redeem_result = await self._redeem_via_vanilla_ctf(
                        w3=w3,
                        collateral_address=match.collateral.address,
                        condition_id=condition_id,
                        index_sets=[1, 2],
                    )
                else:  # CollateralKind.NEGRISK_WRAPPED
                    adapter = match.collateral.adapter
                    if adapter is None:
                        failed += 1
                        errors.append(
                            f"negrisk_collateral_missing_adapter:{condition_id}"
                        )
                        continue
                    redeem_result = await self._redeem_via_negrisk_adapter(
                        w3=w3,
                        adapter_address=adapter.address,
                        condition_id=condition_id,
                        amounts_yes_no_base_units=(
                            int(outcome_raw_balances.get(0, 0)),
                            int(outcome_raw_balances.get(1, 0)),
                        ),
                    )

                if redeem_result.status == "executed":
                    redeemed += 1
                else:
                    failed += 1
                    errors.append(
                        str(redeem_result.error_message or f"redeem_failed:{condition_id}")
                    )
            except Exception as exc:
                failed += 1
                errors.append(str(exc))

        return {
            "wallet_address": execution_wallet,
            "positions_scanned": len(positions),
            "conditions_checked": len(grouped_items),
            "conditions_total": len(grouped),
            "resolved_conditions": resolved,
            "redeemable_value_usd": round(redeemable_value_usd, 4),
            "redeemed": redeemed,
            "skipped_low_payout": skipped_low_payout,
            "skipped_high_gas": skipped_high_gas,
            "skipped_unknown_collateral": skipped_unknown_collateral,
            "failed": failed,
            "gas_price_gwei": round(gas_price_gwei, 2),
            "policy": {
                "min_payout_usd": min_payout_usd,
                "max_gas_price_gwei": max_gas_price_gwei,
                "force_including_losers": force_losers,
            },
            "dry_run": bool(dry_run),
            "errors": errors,
        }


ctf_execution_service = CTFExecutionService()
