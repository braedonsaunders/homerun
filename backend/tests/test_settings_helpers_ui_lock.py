import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api.settings_helpers import apply_update_request
from models.database import AppSettings
from utils.passwords import verify_password


def _make_request(ui_lock_section):
    return SimpleNamespace(
        polymarket=None,
        kalshi=None,
        llm=None,
        notifications=None,
        scanner=None,
        trading=None,
        maintenance=None,
        discovery=None,
        search_filters=None,
        trading_proxy=None,
        events=None,
        ui_lock=ui_lock_section,
    )


def test_apply_update_request_ui_lock_with_password_without_scrypt(monkeypatch):
    monkeypatch.setattr(hashlib, "scrypt", None, raising=False)

    settings = AppSettings(id="default")
    request = _make_request(
        SimpleNamespace(
            enabled=True,
            idle_timeout_minutes=30,
            clear_password=False,
            password=" local-pass ",
        )
    )

    flags = apply_update_request(settings, request)

    assert settings.ui_lock_enabled is True
    assert settings.ui_lock_idle_timeout_minutes == 30
    assert settings.ui_lock_password_hash.startswith("pbkdf2_sha256$")
    assert verify_password("local-pass", settings.ui_lock_password_hash) is True
    assert flags["needs_ui_lock_reload"] is True
    assert flags["reset_ui_lock_sessions"] is True
