import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import config


def test_detect_project_root_backend_only_layout(tmp_path):
    backend_dir = tmp_path / "app"
    backend_dir.mkdir(parents=True, exist_ok=True)
    (backend_dir / "config.py").write_text("# stub\n", encoding="utf-8")

    resolved = config._detect_project_root(backend_dir.resolve())
    assert resolved == backend_dir.resolve()


def test_normalize_database_url_bootstraps_legacy_backend_db(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    backend_dir = project_root / "backend"
    data_dir = project_root / "data"
    backend_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    legacy_db = backend_dir / "arbitrage.db"
    legacy_bytes = b"legacy-db-bytes"
    legacy_db.write_bytes(legacy_bytes)

    monkeypatch.setattr(config, "_PROJECT_ROOT", project_root.resolve())
    monkeypatch.setattr(config, "_LEGACY_DB_PATH", legacy_db.resolve())

    normalized = config.Settings._normalize_database_url(
        "sqlite+aiosqlite:///./data/arbitrage.db"
    )
    expected_path = (project_root / "data" / "arbitrage.db").resolve()

    assert normalized == f"sqlite+aiosqlite:///{expected_path}"
    assert expected_path.read_bytes() == legacy_bytes


def test_normalize_database_url_does_not_overwrite_existing_target(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    backend_dir = project_root / "backend"
    data_dir = project_root / "data"
    backend_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    legacy_db = backend_dir / "arbitrage.db"
    legacy_db.write_bytes(b"legacy")

    target_db = data_dir / "arbitrage.db"
    target_db.write_bytes(b"current")

    monkeypatch.setattr(config, "_PROJECT_ROOT", project_root.resolve())
    monkeypatch.setattr(config, "_LEGACY_DB_PATH", legacy_db.resolve())

    normalized = config.Settings._normalize_database_url(
        "sqlite+aiosqlite:///./data/arbitrage.db"
    )

    assert normalized == f"sqlite+aiosqlite:///{target_db.resolve()}"
    assert target_db.read_bytes() == b"current"
