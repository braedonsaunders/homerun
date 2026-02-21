import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import config


def test_database_url_default_is_postgres_asyncpg():
    default_database_url = config.Settings.model_fields["DATABASE_URL"].default
    assert str(default_database_url).startswith("postgresql+asyncpg://")


def test_normalize_database_url_trims_quotes_whitespace_and_trailing_slash():
    normalized = config.Settings._normalize_database_url(
        "  'postgresql+asyncpg://homerun:homerun@127.0.0.1:5432/homerun/'  "
    )
    assert normalized == "postgresql+asyncpg://homerun:homerun@127.0.0.1:5432/homerun"


def test_normalize_database_url_preserves_none_and_empty():
    assert config.Settings._normalize_database_url(None) is None
    assert config.Settings._normalize_database_url("   ") == ""
