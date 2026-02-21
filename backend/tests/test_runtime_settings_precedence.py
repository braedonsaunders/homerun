import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import config
from models import database


def test_resolve_runtime_override_prefers_db_non_null():
    assert config._resolve_runtime_override(42, 10, 5) == 42


def test_resolve_runtime_override_falls_back_to_current_then_default():
    assert config._resolve_runtime_override(None, 10, 5) == 10
    assert config._resolve_runtime_override(None, None, 5) == 5


@pytest.mark.asyncio
async def test_apply_runtime_settings_overrides_runs_events_then_filters(monkeypatch):
    call_order: list[str] = []

    async def _events() -> None:
        call_order.append("events")

    async def _filters() -> None:
        call_order.append("filters")

    monkeypatch.setattr(config, "apply_events_settings", _events)
    monkeypatch.setattr(config, "apply_search_filters", _filters)

    await config.apply_runtime_settings_overrides()
    assert call_order == ["events", "filters"]


def test_postgres_engine_pool_tuning_is_explicit():
    db_url = str(database.settings.DATABASE_URL or "").strip().lower()
    if db_url.startswith("postgresql"):
        assert database._engine_kw["pool_size"] == max(1, int(database.settings.DATABASE_POOL_SIZE))
        assert database._engine_kw["max_overflow"] == max(0, int(database.settings.DATABASE_MAX_OVERFLOW))
        assert database._engine_kw["pool_timeout"] == max(1, int(database.settings.DATABASE_POOL_TIMEOUT_SECONDS))
        assert database._engine_kw["pool_recycle"] == max(30, int(database.settings.DATABASE_POOL_RECYCLE_SECONDS))
        assert database._engine_kw["pool_use_lifo"] is True
    else:
        assert "pool_size" not in database._engine_kw
