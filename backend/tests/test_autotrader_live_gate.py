import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from workers.autotrader_worker import _source_live_enabled


def test_source_live_enabled_defaults_true():
    assert _source_live_enabled("live", {}) is True


def test_source_live_enabled_respects_metadata_gate():
    assert _source_live_enabled("live", {"metadata": {"live_enabled": False}}) is False
    assert _source_live_enabled("live", {"metadata": {"live_enabled": True}}) is True


def test_source_live_enabled_ignored_outside_live():
    assert _source_live_enabled("paper", {"metadata": {"live_enabled": False}}) is True
