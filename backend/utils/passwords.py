from __future__ import annotations

import base64
import hashlib
import hmac
import os

SCHEME = "scrypt"
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 64
SALT_LEN_BYTES = 16


def hash_password(password: str) -> str:
    normalized = str(password or "")
    if not normalized:
        raise ValueError("Password cannot be empty.")
    salt = os.urandom(SALT_LEN_BYTES)
    digest = hashlib.scrypt(
        normalized.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=SCRYPT_DKLEN,
    )
    salt_b64 = base64.urlsafe_b64encode(salt).decode("ascii")
    digest_b64 = base64.urlsafe_b64encode(digest).decode("ascii")
    return f"{SCHEME}${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt_b64}${digest_b64}"


def verify_password(password: str, encoded_hash: str | None) -> bool:
    if not encoded_hash:
        return False
    parts = str(encoded_hash).split("$")
    if len(parts) != 6:
        return False
    scheme, n_raw, r_raw, p_raw, salt_b64, digest_b64 = parts
    if scheme != SCHEME:
        return False
    try:
        n = int(n_raw)
        r = int(r_raw)
        p = int(p_raw)
        salt = base64.urlsafe_b64decode(salt_b64.encode("ascii"))
        expected = base64.urlsafe_b64decode(digest_b64.encode("ascii"))
    except Exception:
        return False
    try:
        candidate = hashlib.scrypt(
            str(password or "").encode("utf-8"),
            salt=salt,
            n=n,
            r=r,
            p=p,
            dklen=len(expected),
        )
    except Exception:
        return False
    return hmac.compare_digest(candidate, expected)
