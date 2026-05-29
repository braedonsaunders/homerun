"""Tests for dataset pins (pruner-protection for running backtests)."""
from __future__ import annotations

import time

import services.marketdata.pins as pins


def _use_tmp_root(monkeypatch, tmp_path):
    monkeypatch.setattr(pins, "parquet_root", lambda: tmp_path)


def test_pin_release_roundtrip(monkeypatch, tmp_path):
    _use_tmp_root(monkeypatch, tmp_path)
    win = tmp_path / "live_ingestor" / "_" / "20260101T000000__20260101T001500"
    win.mkdir(parents=True)
    f = win / "snapshots__T.parquet"
    f.write_bytes(b"x")

    assert pins.active_pinned_paths() == set()
    pins.pin_paths("hashA", [win])
    active = pins.active_pinned_paths()
    assert str(win).replace("\\", "/") in active
    # a pinned dir protects descendants
    assert pins.is_path_pinned(f)
    assert pins.is_path_pinned(win)
    assert not pins.is_path_pinned(tmp_path / "other")

    pins.release_pin("hashA")
    assert pins.active_pinned_paths() == set()
    assert not pins.is_path_pinned(f)


def test_expired_pin_is_ignored_and_gced(monkeypatch, tmp_path):
    _use_tmp_root(monkeypatch, tmp_path)
    win = tmp_path / "live_ingestor" / "_" / "w"
    win.mkdir(parents=True)
    pins.pin_paths("expls", [win], ttl_seconds=1)
    # force expiry
    pin_file = tmp_path / ".pins" / "expls.json"
    assert pin_file.exists()
    time.sleep(1.1)
    assert pins.active_pinned_paths() == set()  # expired -> not active
    assert not pin_file.exists()  # ... and GC'd


def test_no_pins_dir_is_empty(monkeypatch, tmp_path):
    _use_tmp_root(monkeypatch, tmp_path)
    assert pins.active_pinned_paths() == set()
    assert pins.is_path_pinned(tmp_path / "anything") is False
