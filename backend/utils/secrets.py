"""Helpers for encrypting/decrypting secrets stored in the database."""

from __future__ import annotations

import base64
import hashlib
import os
from typing import Optional

from utils.logger import get_logger

logger = get_logger("secrets")

_ENC_PREFIX = "enc:v1:"
_FERNET_CACHE = None


def _derive_fernet_key(raw_key: str) -> bytes:
    """Derive a Fernet-compatible key from arbitrary input."""
    # Fernet expects 32-byte URL-safe base64 data.
    digest = hashlib.sha256(raw_key.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def _get_fernet():
    """Return initialized Fernet instance or None if key/dependency missing."""
    global _FERNET_CACHE
    if _FERNET_CACHE is not None:
        return _FERNET_CACHE

    secret_key = os.getenv("APP_SECRETS_KEY")
    if not secret_key:
        _FERNET_CACHE = False
        return None

    try:
        from cryptography.fernet import Fernet

        _FERNET_CACHE = Fernet(_derive_fernet_key(secret_key))
        return _FERNET_CACHE
    except Exception as exc:
        logger.warning("Secret crypto unavailable; DB secrets stay plaintext", error=str(exc))
        _FERNET_CACHE = False
        return None


def is_encrypted(value: Optional[str]) -> bool:
    return bool(value and value.startswith(_ENC_PREFIX))


def encrypt_secret(value: Optional[str]) -> Optional[str]:
    """Encrypt a plaintext secret value. Returns original when crypto unavailable."""
    if value is None or value == "":
        return None
    if is_encrypted(value):
        return value
    fernet = _get_fernet()
    if not fernet:
        return value
    token = fernet.encrypt(value.encode("utf-8")).decode("utf-8")
    return _ENC_PREFIX + token


def decrypt_secret(value: Optional[str]) -> Optional[str]:
    """Decrypt a stored secret value. Plaintext values are returned unchanged."""
    if value is None or value == "":
        return None
    if not is_encrypted(value):
        return value
    fernet = _get_fernet()
    if not fernet:
        logger.warning("Encrypted secret cannot be decrypted without APP_SECRETS_KEY")
        return None
    token = value[len(_ENC_PREFIX) :]
    try:
        return fernet.decrypt(token.encode("utf-8")).decode("utf-8")
    except Exception as exc:
        logger.warning("Failed to decrypt stored secret", error=str(exc))
        return None
