"""EIP-712 signer for kalshi_predictit_arb_source.py (live mode only).

Kept out of the data-source file on purpose: homerun's data-source sandbox
blocks ``os`` and ``eth_account`` imports, so signing lives in host code. Attach
an instance to a loaded source to opt in to the paid live feed::

    runtime.instance.payment_signer = EthAccountSigner()  # reads X402_WALLET_KEY

Requires ``eth-account`` (already in ``backend/requirements-trading.txt``).
Use a dedicated low-balance wallet: each call pays up to ``max_usd_per_call``.
The key is never logged or stored on the source.
"""

from __future__ import annotations

import os
from typing import Any


class EthAccountSigner:
    def __init__(self, private_key: str | bytes | None = None) -> None:
        from eth_account import Account

        key = private_key or os.environ.get("X402_WALLET_KEY")
        if not key:
            raise ValueError("No key: pass private_key or set X402_WALLET_KEY")
        self._account = Account.from_key(key)

    @property
    def address(self) -> str:
        return self._account.address

    def sign_typed_data(self, typed_data: dict[str, Any]) -> str:
        from eth_account.messages import encode_typed_data

        signed = self._account.sign_message(encode_typed_data(full_message=typed_data))
        # hexbytes>=1.0 .hex() drops the 0x prefix
        return "0x" + bytes(signed.signature).hex()

    def __repr__(self) -> str:
        return f"EthAccountSigner(address={self.address})"
