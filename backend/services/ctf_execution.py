from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

from config import settings
from services.polymarket import polymarket_client
from services.live_execution_service import live_execution_service
from utils.converters import safe_float
from utils.logger import get_logger

logger = get_logger(__name__)

_USDC_DECIMALS = 6
_MAX_UINT256 = 2**256 - 1
_ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
_MIN_APPROVAL_BUFFER_BASE = 10_000_000  # 10 USDC (6 decimals)


@dataclass
class CTFExecutionResult:
    status: str
    action: str
    tx_hash: str | None
    error_message: str | None
    payload: dict[str, Any]


class CTFExecutionService:
    CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
    USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
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
            nonce = await asyncio.to_thread(lambda: w3.eth.get_transaction_count(from_address))
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

            owner_nonce = await asyncio.to_thread(lambda: w3.eth.get_transaction_count(owner_eoa))
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
            logger.error("CTF contract call failed", action=action, exc_info=exc)
            return CTFExecutionResult(
                status="failed",
                action=action,
                tx_hash=None,
                error_message=str(exc),
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

    async def ensure_usdc_approval(self, *, min_amount_base_units: int = 0) -> CTFExecutionResult:
        try:
            wallet_ctx = await self._resolve_wallet_context()
            w3 = await self._get_web3()
            owner = w3.to_checksum_address(wallet_ctx["execution_wallet"])
            usdc_address = w3.to_checksum_address(self.USDC_ADDRESS)
            ctf_address = w3.to_checksum_address(self.CTF_ADDRESS)
            usdc = w3.eth.contract(address=usdc_address, abi=self._ERC20_ABI)
            allowance = await asyncio.to_thread(lambda: usdc.functions.allowance(owner, ctf_address).call())
            required = max(int(min_amount_base_units), _MIN_APPROVAL_BUFFER_BASE)
            if int(allowance or 0) >= required:
                return CTFExecutionResult(
                    status="executed",
                    action="approve_usdc",
                    tx_hash=None,
                    error_message=None,
                    payload={"allowance": int(allowance), "required": required, "already_approved": True},
                )

            data = usdc.functions.approve(ctf_address, _MAX_UINT256)._encode_transaction_data()
            result = await self._execute_contract_call(
                contract_address=self.USDC_ADDRESS,
                data=data,
                gas_limit=140_000,
                action="approve_usdc",
            )
            if result.status == "executed":
                result.payload.update({"required": required, "already_approved": False})
            return result
        except Exception as exc:
            logger.error("USDC approval check failed", exc_info=exc)
            return CTFExecutionResult(
                status="failed",
                action="approve_usdc",
                tx_hash=None,
                error_message=str(exc),
                payload={},
            )

    async def ensure_exchange_approval(self) -> CTFExecutionResult:
        try:
            wallet_ctx = await self._resolve_wallet_context()
            w3 = await self._get_web3()
            owner = w3.to_checksum_address(wallet_ctx["execution_wallet"])
            ctf_address = w3.to_checksum_address(self.CTF_ADDRESS)
            exchange_address = w3.to_checksum_address(self.CTF_EXCHANGE)
            ctf = w3.eth.contract(address=ctf_address, abi=self._CTF_ABI)
            approved = await asyncio.to_thread(lambda: ctf.functions.isApprovedForAll(owner, exchange_address).call())
            if bool(approved):
                return CTFExecutionResult(
                    status="executed",
                    action="approve_exchange",
                    tx_hash=None,
                    error_message=None,
                    payload={"already_approved": True},
                )

            data = ctf.functions.setApprovalForAll(exchange_address, True)._encode_transaction_data()
            result = await self._execute_contract_call(
                contract_address=self.CTF_ADDRESS,
                data=data,
                gas_limit=170_000,
                action="approve_exchange",
            )
            if result.status == "executed":
                result.payload.update({"already_approved": False})
            return result
        except Exception as exc:
            logger.error("Exchange approval check failed", exc_info=exc)
            return CTFExecutionResult(
                status="failed",
                action="approve_exchange",
                tx_hash=None,
                error_message=str(exc),
                payload={},
            )

    async def split_position(self, *, condition_id: str, amount_usd: float) -> CTFExecutionResult:
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

        amount_base = self._to_base_units(amount)
        approval = await self.ensure_usdc_approval(min_amount_base_units=amount_base)
        if approval.status != "executed":
            return approval

        exchange_approval = await self.ensure_exchange_approval()
        if exchange_approval.status != "executed":
            return exchange_approval

        w3 = await self._get_web3()
        ctf = w3.eth.contract(address=w3.to_checksum_address(self.CTF_ADDRESS), abi=self._CTF_ABI)
        data = ctf.functions.splitPosition(
            self.USDC_ADDRESS,
            "0x" + ("00" * 32),
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
                "amount_usd": amount,
                "amount_base_units": amount_base,
                "shares_per_side": amount,
            }
        )
        return result

    async def merge_positions(self, *, condition_id: str, shares_per_side: float) -> CTFExecutionResult:
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

        amount_base = self._to_base_units(shares)
        w3 = await self._get_web3()
        ctf = w3.eth.contract(address=w3.to_checksum_address(self.CTF_ADDRESS), abi=self._CTF_ABI)
        data = ctf.functions.mergePositions(
            self.USDC_ADDRESS,
            "0x" + ("00" * 32),
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
                "shares_per_side": shares,
                "amount_base_units": amount_base,
            }
        )
        return result

    async def redeem_positions(
        self,
        *,
        condition_id: str,
        index_sets: list[int] | None = None,
    ) -> CTFExecutionResult:
        normalized_condition_id = self._normalize_condition_id(condition_id)
        normalized_sets = [int(value) for value in (index_sets or [1, 2]) if int(value) > 0]
        if not normalized_sets:
            normalized_sets = [1, 2]

        w3 = await self._get_web3()
        ctf = w3.eth.contract(address=w3.to_checksum_address(self.CTF_ADDRESS), abi=self._CTF_ABI)
        data = ctf.functions.redeemPositions(
            self.USDC_ADDRESS,
            "0x" + ("00" * 32),
            normalized_condition_id,
            normalized_sets,
        )._encode_transaction_data()
        result = await self._execute_contract_call(
            contract_address=self.CTF_ADDRESS,
            data=data,
            gas_limit=260_000,
            action="redeem",
        )
        result.payload.update(
            {
                "condition_id": normalized_condition_id,
                "index_sets": normalized_sets,
            }
        )
        return result

    async def redeem_resolved_wallet_positions(
        self,
        *,
        wallet_address: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        execution_wallet = str(wallet_address or live_execution_service.get_execution_wallet_address() or "").strip().lower()
        if not execution_wallet:
            return {
                "wallet_address": "",
                "positions_scanned": 0,
                "conditions_checked": 0,
                "resolved_conditions": 0,
                "redeemed": 0,
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
                "redeemed": 0,
                "failed": 0,
                "dry_run": bool(dry_run),
                "errors": [],
            }

        grouped: dict[str, set[int]] = {}
        for row in positions:
            condition_id_raw = row.get("conditionId") or row.get("condition_id") or row.get("market")
            token_id_raw = row.get("asset") or row.get("asset_id") or row.get("token_id") or row.get("tokenId")
            try:
                condition_id = self._normalize_condition_id(condition_id_raw)
            except Exception:
                continue
            token_id_uint = self._parse_token_id_uint256(token_id_raw)
            if token_id_uint is None:
                continue
            grouped.setdefault(condition_id, set()).add(token_id_uint)

        if not grouped:
            return {
                "wallet_address": execution_wallet,
                "positions_scanned": len(positions),
                "conditions_checked": 0,
                "resolved_conditions": 0,
                "redeemed": 0,
                "failed": 0,
                "dry_run": bool(dry_run),
                "errors": [],
            }

        w3 = await self._get_web3()
        ctf = w3.eth.contract(address=w3.to_checksum_address(self.CTF_ADDRESS), abi=self._CTF_ABI)
        checksum_wallet = w3.to_checksum_address(execution_wallet)

        redeemed = 0
        failed = 0
        resolved = 0
        errors: list[str] = []

        for condition_id, token_ids in grouped.items():
            try:
                denominator = await asyncio.to_thread(lambda: ctf.functions.payoutDenominator(condition_id).call())
                if int(denominator or 0) <= 0:
                    continue
                resolved += 1

                total_shares = 0.0
                for token_id in token_ids:
                    raw_balance = await asyncio.to_thread(
                        lambda token=token_id: ctf.functions.balanceOf(checksum_wallet, token).call()
                    )
                    total_shares += float(raw_balance or 0) / (10**_USDC_DECIMALS)

                if total_shares <= 1e-6:
                    continue

                if dry_run:
                    redeemed += 1
                    continue

                redeem_result = await self.redeem_positions(condition_id=condition_id, index_sets=[1, 2])
                if redeem_result.status == "executed":
                    redeemed += 1
                else:
                    failed += 1
                    errors.append(str(redeem_result.error_message or f"redeem_failed:{condition_id}"))
            except Exception as exc:
                failed += 1
                errors.append(str(exc))

        return {
            "wallet_address": execution_wallet,
            "positions_scanned": len(positions),
            "conditions_checked": len(grouped),
            "resolved_conditions": resolved,
            "redeemed": redeemed,
            "failed": failed,
            "dry_run": bool(dry_run),
            "errors": errors,
        }


ctf_execution_service = CTFExecutionService()
