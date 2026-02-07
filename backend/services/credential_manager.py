"""
EIP-712 Credential Auto-Generation

Auto-derives Polymarket CLOB API credentials from a wallet private key.
Eliminates the need for users to manually generate and copy API keys
from the Polymarket settings page.

Flow:
1. Sign EIP-712 ClobAuth message with wallet private key
2. POST to /auth/api-key to obtain API credentials
3. Store credentials in SQL DB for persistence across restarts
4. Fall back to /auth/derive-api-key for credential recovery
"""

import uuid
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
from sqlalchemy import Column, String, DateTime, Boolean, select
from models.database import Base, AsyncSessionLocal
from utils.logger import get_logger

logger = get_logger("credential_manager")

CLOB_AUTH_DOMAIN = {
    "name": "ClobAuthDomain",
    "version": "1",
    "chainId": 137,
}

CLOB_AUTH_TYPES = {
    "ClobAuth": [
        {"name": "address", "type": "address"},
        {"name": "timestamp", "type": "string"},
        {"name": "nonce", "type": "uint256"},
        {"name": "message", "type": "string"},
    ]
}

CLOB_AUTH_MESSAGE = "This message attests that I control the given wallet"


class StoredCredential(Base):
    """CLOB API credentials stored in database."""

    __tablename__ = "stored_credentials"

    id = Column(String, primary_key=True)
    wallet_address = Column(String, nullable=False, unique=True, index=True)
    api_key = Column(String, nullable=False)
    api_secret = Column(String, nullable=False)
    api_passphrase = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)
    last_verified_at = Column(DateTime, nullable=True)


@dataclass
class ClobCredentials:
    api_key: str
    api_secret: str
    api_passphrase: str
    wallet_address: str


class CredentialManager:
    def __init__(self):
        self._cached_creds: dict[str, ClobCredentials] = {}  # address -> creds

    def _get_wallet_address(self, private_key: str) -> str:
        """Derive wallet address from private key."""
        try:
            from eth_account import Account

            account = Account.from_key(private_key)
            return account.address
        except Exception as e:
            logger.error("Failed to derive wallet address", error=str(e))
            raise

    def _build_clob_auth_message(self, address: str) -> dict:
        """Build the EIP-712 structured data for ClobAuth."""
        timestamp = str(int(time.time()))
        nonce = 0  # First-time nonce
        return {
            "address": address,
            "timestamp": timestamp,
            "nonce": nonce,
            "message": CLOB_AUTH_MESSAGE,
        }

    def _sign_clob_auth(self, private_key: str, address: str) -> tuple[str, str, int]:
        """Sign EIP-712 ClobAuth message. Returns (signature, timestamp, nonce)."""
        try:
            from eth_account import Account
            from eth_account.messages import encode_structured_data

            timestamp = str(int(time.time()))
            nonce = 0

            structured_data = {
                "types": {
                    "EIP712Domain": [
                        {"name": "name", "type": "string"},
                        {"name": "version", "type": "string"},
                        {"name": "chainId", "type": "uint256"},
                    ],
                    **CLOB_AUTH_TYPES,
                },
                "primaryType": "ClobAuth",
                "domain": CLOB_AUTH_DOMAIN,
                "message": {
                    "address": address,
                    "timestamp": timestamp,
                    "nonce": nonce,
                    "message": CLOB_AUTH_MESSAGE,
                },
            }

            encoded = encode_structured_data(structured_data)
            signed = Account.sign_message(encoded, private_key=private_key)
            return signed.signature.hex(), timestamp, nonce
        except ImportError:
            raise ImportError("eth_account required: pip install eth-account")
        except Exception as e:
            logger.error("EIP-712 signing failed", error=str(e))
            raise

    async def generate_credentials(
        self, private_key: str, clob_url: str = "https://clob.polymarket.com"
    ) -> ClobCredentials:
        """
        Generate CLOB API credentials from a private key.

        1. Signs EIP-712 ClobAuth message
        2. POSTs to /auth/api-key with L1 auth headers
        3. Stores credentials in DB
        4. Returns ClobCredentials
        """
        address = self._get_wallet_address(private_key)

        # Check DB cache first
        cached = await self._load_from_db(address)
        if cached:
            logger.info("Loaded stored credentials", address=address)
            return cached

        # Check in-memory cache
        if address.lower() in self._cached_creds:
            return self._cached_creds[address.lower()]

        # Generate new credentials via EIP-712
        signature, timestamp, nonce = self._sign_clob_auth(private_key, address)

        import httpx

        headers = {
            "POLY_ADDRESS": address,
            "POLY_SIGNATURE": signature,
            "POLY_TIMESTAMP": timestamp,
            "POLY_NONCE": str(nonce),
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Try to create new API key
            try:
                resp = await client.post(
                    f"{clob_url}/auth/api-key",
                    headers=headers,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    creds = ClobCredentials(
                        api_key=data["apiKey"],
                        api_secret=data["secret"],
                        api_passphrase=data["passphrase"],
                        wallet_address=address,
                    )
                    await self._save_to_db(creds)
                    self._cached_creds[address.lower()] = creds
                    logger.info("Generated new CLOB credentials", address=address)
                    return creds
            except Exception as e:
                logger.warning("Failed to create API key, trying derive", error=str(e))

            # Fallback: derive existing credentials
            try:
                resp = await client.get(
                    f"{clob_url}/auth/derive-api-key",
                    headers=headers,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    creds = ClobCredentials(
                        api_key=data["apiKey"],
                        api_secret=data["secret"],
                        api_passphrase=data["passphrase"],
                        wallet_address=address,
                    )
                    await self._save_to_db(creds)
                    self._cached_creds[address.lower()] = creds
                    logger.info("Derived existing CLOB credentials", address=address)
                    return creds
            except Exception as e:
                logger.error("Failed to derive API key", error=str(e))
                raise RuntimeError(
                    f"Could not generate or derive CLOB credentials: {e}"
                )

        raise RuntimeError("Failed to obtain CLOB API credentials")

    async def get_or_generate(
        self, private_key: str, clob_url: str = "https://clob.polymarket.com"
    ) -> ClobCredentials:
        """Get cached credentials or generate new ones."""
        address = self._get_wallet_address(private_key)

        # Memory cache
        if address.lower() in self._cached_creds:
            return self._cached_creds[address.lower()]

        # DB cache
        cached = await self._load_from_db(address)
        if cached:
            self._cached_creds[address.lower()] = cached
            return cached

        # Generate
        return await self.generate_credentials(private_key, clob_url)

    async def _load_from_db(self, address: str) -> Optional[ClobCredentials]:
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(StoredCredential).where(
                        StoredCredential.wallet_address == address,
                        StoredCredential.is_active.is_(True),
                    )
                )
                row = result.scalar_one_or_none()
                if row:
                    row.last_used_at = datetime.utcnow()
                    await session.commit()
                    return ClobCredentials(
                        api_key=row.api_key,
                        api_secret=row.api_secret,
                        api_passphrase=row.api_passphrase,
                        wallet_address=row.wallet_address,
                    )
        except Exception as e:
            logger.error("Failed to load credentials from DB", error=str(e))
        return None

    async def _save_to_db(self, creds: ClobCredentials):
        try:
            async with AsyncSessionLocal() as session:
                # Deactivate old creds for this address
                result = await session.execute(
                    select(StoredCredential).where(
                        StoredCredential.wallet_address == creds.wallet_address
                    )
                )
                for old in result.scalars().all():
                    old.is_active = False

                session.add(
                    StoredCredential(
                        id=str(uuid.uuid4()),
                        wallet_address=creds.wallet_address,
                        api_key=creds.api_key,
                        api_secret=creds.api_secret,
                        api_passphrase=creds.api_passphrase,
                    )
                )
                await session.commit()
        except Exception as e:
            logger.error("Failed to save credentials to DB", error=str(e))

    async def invalidate(self, address: str):
        """Invalidate cached credentials for an address."""
        self._cached_creds.pop(address.lower(), None)
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(StoredCredential).where(
                        StoredCredential.wallet_address == address
                    )
                )
                for cred in result.scalars().all():
                    cred.is_active = False
                await session.commit()
        except Exception as e:
            logger.error("Failed to invalidate credentials", error=str(e))


credential_manager = CredentialManager()
