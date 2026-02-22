from __future__ import annotations

import base64
import hashlib
import hmac
import os

SCHEME_SCRYPT = "scrypt"
SCHEME_PBKDF2 = "pbkdf2_sha256"
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 64
PBKDF2_ITERATIONS = 600_000
PBKDF2_DKLEN = 32
SALT_LEN_BYTES = 16


def _decode_b64(value: str) -> bytes:
    try:
        return base64.urlsafe_b64decode(value.encode("ascii"))
    except Exception:
        return b""


def _encode_b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii")


def _hash_with_scrypt(password: str) -> str:
    scrypt_fn = getattr(hashlib, "scrypt", None)
    if not callable(scrypt_fn):
        raise RuntimeError("hashlib.scrypt is unavailable")
    salt = os.urandom(SALT_LEN_BYTES)
    digest = scrypt_fn(
        password.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=SCRYPT_DKLEN,
    )
    return f"{SCHEME_SCRYPT}${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${_encode_b64(salt)}${_encode_b64(digest)}"


def _hash_with_pbkdf2(password: str) -> str:
    salt = os.urandom(SALT_LEN_BYTES)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
        dklen=PBKDF2_DKLEN,
    )
    return f"{SCHEME_PBKDF2}${PBKDF2_ITERATIONS}${_encode_b64(salt)}${_encode_b64(digest)}"


def hash_password(password: str) -> str:
    normalized = str(password or "")
    if not normalized:
        raise ValueError("Password cannot be empty.")
    try:
        return _hash_with_scrypt(normalized)
    except Exception:
        return _hash_with_pbkdf2(normalized)


def verify_password(password: str, encoded_hash: str | None) -> bool:
    if not encoded_hash:
        return False
    parts = str(encoded_hash).split("$")
    if not parts:
        return False
    scheme = parts[0]

    if scheme == SCHEME_SCRYPT:
        if len(parts) != 6:
            return False
        _, n_raw, r_raw, p_raw, salt_b64, digest_b64 = parts
        try:
            n = int(n_raw)
            r = int(r_raw)
            p = int(p_raw)
            salt = _decode_b64(salt_b64)
            expected = _decode_b64(digest_b64)
            if not salt or not expected:
                return False
            scrypt_fn = getattr(hashlib, "scrypt", None)
            if not callable(scrypt_fn):
                return False
            candidate = scrypt_fn(
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

    if scheme == SCHEME_PBKDF2:
        if len(parts) != 4:
            return False
        _, iterations_raw, salt_b64, digest_b64 = parts
        try:
            iterations = int(iterations_raw)
            if iterations <= 0:
                return False
            salt = _decode_b64(salt_b64)
            expected = _decode_b64(digest_b64)
            if not salt or not expected:
                return False
            candidate = hashlib.pbkdf2_hmac(
                "sha256",
                str(password or "").encode("utf-8"),
                salt,
                iterations,
                dklen=len(expected),
            )
        except Exception:
            return False
        return hmac.compare_digest(candidate, expected)

    return False
