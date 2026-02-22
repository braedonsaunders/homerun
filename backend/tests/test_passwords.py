import base64
import hashlib
import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from utils.passwords import verify_password, hash_password


def test_hash_password_falls_back_when_scrypt_unavailable(monkeypatch):
    monkeypatch.setattr(hashlib, "scrypt", None, raising=False)

    encoded = hash_password("my-password")

    assert encoded.startswith("pbkdf2_sha256$")
    assert verify_password("my-password", encoded) is True
    assert verify_password("wrong-password", encoded) is False


def test_verify_password_supports_scrypt_hashes_when_available():
    scrypt_fn = getattr(hashlib, "scrypt", None)
    if not callable(scrypt_fn):
        return

    salt = os.urandom(16)
    digest = scrypt_fn(
        b"my-password",
        salt=salt,
        n=2**14,
        r=8,
        p=1,
        dklen=64,
    )
    encoded = "scrypt$16384$8$1$%s$%s" % (
        base64.urlsafe_b64encode(salt).decode("ascii"),
        base64.urlsafe_b64encode(digest).decode("ascii"),
    )

    assert verify_password("my-password", encoded) is True
    assert verify_password("wrong-password", encoded) is False


def test_hash_password_rejects_empty_input():
    try:
        hash_password("")
    except ValueError as exc:
        assert str(exc) == "Password cannot be empty."
        return
    raise AssertionError("Expected ValueError for empty password")
