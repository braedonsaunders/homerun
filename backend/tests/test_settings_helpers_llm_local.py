import sys
from pathlib import Path
from types import SimpleNamespace

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api.settings_helpers import apply_update_request, llm_payload
from models.database import AppSettings
from utils.secrets import decrypt_secret


def _make_empty_request_with_llm(llm_section):
    return SimpleNamespace(
        polymarket=None,
        kalshi=None,
        llm=llm_section,
        notifications=None,
        scanner=None,
        trading=None,
        maintenance=None,
        trading_proxy=None,
        search_filters=None,
    )


def test_llm_payload_includes_local_provider_fields():
    settings = AppSettings(id="default")
    settings.llm_provider = "ollama"
    settings.ollama_api_key = "ollama-secret"
    settings.ollama_base_url = "http://localhost:11434"
    settings.lmstudio_api_key = "lmstudio-secret"
    settings.lmstudio_base_url = "http://localhost:1234/v1"

    payload = llm_payload(settings)

    assert payload["provider"] == "ollama"
    assert payload["ollama_base_url"] == "http://localhost:11434"
    assert payload["lmstudio_base_url"] == "http://localhost:1234/v1"
    assert payload["ollama_api_key"] is not None
    assert payload["lmstudio_api_key"] is not None


def test_apply_update_request_sets_and_normalizes_local_llm_fields():
    settings = AppSettings(id="default")
    llm_section = SimpleNamespace(
        provider="ollama",
        openai_api_key=None,
        anthropic_api_key=None,
        google_api_key=None,
        xai_api_key=None,
        deepseek_api_key=None,
        ollama_api_key="ollama-secret",
        ollama_base_url=" http://localhost:11434 ",
        lmstudio_api_key="lmstudio-secret",
        lmstudio_base_url=" http://localhost:1234/v1 ",
        model="llama3.2:latest",
        max_monthly_spend=25.0,
    )
    request = _make_empty_request_with_llm(llm_section)

    flags = apply_update_request(settings, request)

    assert settings.llm_provider == "ollama"
    assert decrypt_secret(settings.ollama_api_key) == "ollama-secret"
    assert decrypt_secret(settings.lmstudio_api_key) == "lmstudio-secret"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.lmstudio_base_url == "http://localhost:1234/v1"
    assert settings.llm_model == "llama3.2:latest"
    assert settings.ai_default_model == "llama3.2:latest"
    assert settings.ai_max_monthly_spend == 25.0
    assert flags["needs_llm_reinit"] is True
