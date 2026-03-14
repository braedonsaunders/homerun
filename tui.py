#!/usr/bin/env python3
"""Homerun TUI - Beautiful terminal interface for the Homerun trading platform."""
from __future__ import annotations

import collections
import ctypes
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

# ---------------------------------------------------------------------------
# Auto-install TUI prerequisites on the host machine.
# This runs every launch and is a no-op when packages are already present.
# ---------------------------------------------------------------------------
_TUI_PREREQS = ["textual>=8.0.0,<9.0", "rich>=14.0.0,<15.0.0"]

def _ensure_prereqs() -> None:
    needs_install = False
    try:
        import textual
        import rich  # noqa: F401
        _tv = tuple(int(x) for x in textual.__version__.split(".")[:2])
        if _tv < (8, 0) or _tv >= (9, 0):
            needs_install = True
    except ImportError:
        needs_install = True
    if needs_install:
        print("Installing TUI dependencies...", flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", *_TUI_PREREQS],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        # Re-launch so the freshly installed packages are loaded cleanly
        # instead of the stale modules cached in sys.modules.
        sys.exit(subprocess.call([sys.executable] + sys.argv))

_ensure_prereqs()

from textual import on, work
from textual.app import App, ComposeResult
from textual.events import MouseDown, Resize
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Log,
    Static,
    TabbedContent,
    TabPane,
)

# ---------------------------------------------------------------------------
# Windows Job Object – ensures ALL child processes die when the TUI exits,
# regardless of how it exits (graceful quit, terminal close, crash, etc.).
# ---------------------------------------------------------------------------
_win_job_handle = None

# PIDs of direct child processes (backend, frontend) for SIGBREAK cleanup.
_child_pids: set = set()

if sys.platform == "win32":
    try:
        import ctypes.wintypes

        _kernel32 = ctypes.windll.kernel32

        # Set proper restype/argtypes for 64-bit handle safety.
        _HANDLE = ctypes.wintypes.HANDLE
        _BOOL = ctypes.wintypes.BOOL
        _DWORD = ctypes.wintypes.DWORD
        _LPVOID = ctypes.wintypes.LPVOID

        _kernel32.CreateJobObjectW.restype = _HANDLE
        _kernel32.CreateJobObjectW.argtypes = [_LPVOID, ctypes.wintypes.LPCWSTR]
        _kernel32.SetInformationJobObject.restype = _BOOL
        _kernel32.SetInformationJobObject.argtypes = [_HANDLE, ctypes.c_int, _LPVOID, _DWORD]
        _kernel32.AssignProcessToJobObject.restype = _BOOL
        _kernel32.AssignProcessToJobObject.argtypes = [_HANDLE, _HANDLE]
        _kernel32.OpenProcess.restype = _HANDLE
        _kernel32.OpenProcess.argtypes = [_DWORD, _BOOL, _DWORD]
        _kernel32.CloseHandle.restype = _BOOL
        _kernel32.CloseHandle.argtypes = [_HANDLE]

        # Job Object constants
        _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        _JOB_OBJECT_EXTENDED_LIMIT_INFO_CLASS = 9

        class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", ctypes.wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", ctypes.wintypes.DWORD),
                ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                ("PriorityClass", ctypes.wintypes.DWORD),
                ("SchedulingClass", ctypes.wintypes.DWORD),
            ]

        class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", ctypes.c_byte * 48),  # IO_COUNTERS
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        _win_job_handle = _kernel32.CreateJobObjectW(None, None)
        if _win_job_handle:
            info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            _kernel32.SetInformationJobObject(
                _win_job_handle,
                _JOB_OBJECT_EXTENDED_LIMIT_INFO_CLASS,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
    except Exception:
        _win_job_handle = None


def _assign_to_job(proc: subprocess.Popen) -> None:
    """Assign a subprocess to the Windows Job Object so it dies when we die."""
    _child_pids.add(proc.pid)
    if _win_job_handle is None or sys.platform != "win32":
        return
    try:
        handle = _kernel32.OpenProcess(0x1FFFFF, False, proc.pid)  # PROCESS_ALL_ACCESS
        if handle:
            _kernel32.AssignProcessToJobObject(_win_job_handle, handle)
            _kernel32.CloseHandle(handle)
    except Exception:
        pass


def _terminate_process_tree(root_pid: int) -> None:
    """Kill *root_pid* and every descendant using Win32 API only.

    No subprocess spawning — safe to call from a SIGBREAK signal handler
    where the console is being torn down and new processes can't start.
    """
    if sys.platform != "win32":
        return
    try:
        TH32CS_SNAPPROCESS = 0x00000002
        PROCESS_TERMINATE = 0x0001

        class PROCESSENTRY32(ctypes.Structure):
            _fields_ = [
                ("dwSize", _DWORD),
                ("cntUsage", _DWORD),
                ("th32ProcessID", _DWORD),
                ("th32DefaultHeapID", ctypes.c_void_p),
                ("th32ModuleID", _DWORD),
                ("cntThreads", _DWORD),
                ("th32ParentProcessID", _DWORD),
                ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", _DWORD),
                ("szExeFile", ctypes.c_char * 260),
            ]

        _kernel32.CreateToolhelp32Snapshot.restype = _HANDLE
        _kernel32.CreateToolhelp32Snapshot.argtypes = [_DWORD, _DWORD]
        _kernel32.Process32First.restype = _BOOL
        _kernel32.Process32First.argtypes = [_HANDLE, ctypes.POINTER(PROCESSENTRY32)]
        _kernel32.Process32Next.restype = _BOOL
        _kernel32.Process32Next.argtypes = [_HANDLE, ctypes.POINTER(PROCESSENTRY32)]
        _kernel32.TerminateProcess.restype = _BOOL
        _kernel32.TerminateProcess.argtypes = [_HANDLE, ctypes.c_uint]

        snap = _kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        INVALID = ctypes.wintypes.HANDLE(-1).value  # INVALID_HANDLE_VALUE
        if snap is None or snap == INVALID:
            return

        # Build parent → [children] map from snapshot.
        children: dict[int, list[int]] = {}
        entry = PROCESSENTRY32()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32)
        if _kernel32.Process32First(snap, ctypes.byref(entry)):
            while True:
                children.setdefault(entry.th32ParentProcessID, []).append(
                    entry.th32ProcessID
                )
                if not _kernel32.Process32Next(snap, ctypes.byref(entry)):
                    break
        _kernel32.CloseHandle(snap)

        # BFS to collect all descendants.
        to_kill: list[int] = []
        queue = [root_pid]
        while queue:
            pid = queue.pop(0)
            to_kill.append(pid)
            queue.extend(children.get(pid, []))

        # Terminate leaves first, then parents.
        for pid in reversed(to_kill):
            try:
                h = _kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
                if h:
                    _kernel32.TerminateProcess(h, 1)
                    _kernel32.CloseHandle(h)
            except Exception:
                pass
    except Exception:
        pass


def _sigbreak_kill_children() -> None:
    """Kill all tracked child process trees via Win32 API.

    Uses TerminateProcess (direct kernel call) instead of taskkill.exe
    (subprocess) because during console teardown new processes may fail
    to start.  Also closes the Job Object handle as a safety net.
    """
    for pid in _child_pids:
        _terminate_process_tree(pid)
    # Close Job Object handle — triggers KILL_ON_JOB_CLOSE for anything
    # the tree-walk missed (e.g. processes that escaped the parent tree).
    if _win_job_handle:
        try:
            _kernel32.CloseHandle(_win_job_handle)
        except Exception:
            pass


def _run_osascript(script: str) -> bool:
    if shutil.which("osascript") is None:
        return False
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=2,
        )
        return result.returncode == 0
    except Exception:
        return False


def _resize_macos_terminal_window_to_quarter_screen() -> bool:
    if sys.platform != "darwin":
        return False

    terminal_script = """
tell application "Terminal"
    if (count of windows) = 0 then return
    set screenBounds to bounds of window of desktop of application "Finder"
    set screenLeft to item 1 of screenBounds
    set screenTop to item 2 of screenBounds
    set screenRight to item 3 of screenBounds
    set screenBottom to item 4 of screenBounds
    set targetLeft to screenLeft + 40
    set targetTop to screenTop + 40
    set targetRight to targetLeft + ((screenRight - screenLeft) div 2)
    set targetBottom to targetTop + ((screenBottom - screenTop) div 2)
    set bounds of front window to {targetLeft, targetTop, targetRight, targetBottom}
end tell
""".strip()

    iterm_script = """
tell application "iTerm2"
    if (count of windows) = 0 then return
    set screenBounds to bounds of window of desktop of application "Finder"
    set screenLeft to item 1 of screenBounds
    set screenTop to item 2 of screenBounds
    set screenRight to item 3 of screenBounds
    set screenBottom to item 4 of screenBounds
    set targetLeft to screenLeft + 40
    set targetTop to screenTop + 40
    set targetRight to targetLeft + ((screenRight - screenLeft) div 2)
    set targetBottom to targetTop + ((screenBottom - screenTop) div 2)
    tell current window
        set bounds to {targetLeft, targetTop, targetRight, targetBottom}
    end tell
end tell
""".strip()

    term_program = os.environ.get("TERM_PROGRAM", "")
    scripts: list[str] = []
    if term_program == "Apple_Terminal":
        scripts = [terminal_script, iterm_script]
    elif term_program == "iTerm.app":
        scripts = [iterm_script, terminal_script]
    else:
        scripts = [terminal_script, iterm_script]

    for script in scripts:
        if _run_osascript(script):
            return True
    return False


def _resize_windows_terminal_window_to_quarter_screen() -> bool:
    if sys.platform != "win32":
        return False
    try:
        import ctypes.wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        class RECT(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long),
            ]

        class MONITORINFO(ctypes.Structure):
            _fields_ = [
                ("cbSize", ctypes.wintypes.DWORD),
                ("rcMonitor", RECT),
                ("rcWork", RECT),
                ("dwFlags", ctypes.wintypes.DWORD),
            ]

        MONITOR_DEFAULTTONEAREST = 2

        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            hwnd = kernel32.GetConsoleWindow()
        if not hwnd:
            return False

        monitor = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
        if not monitor:
            return False

        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        if not user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
            return False

        work = info.rcWork
        work_width = max(1, work.right - work.left)
        work_height = max(1, work.bottom - work.top)
        target_width = max(1, work_width // 2)
        target_height = max(1, work_height // 2)
        target_left = work.left + 40
        target_top = work.top + 40

        max_left = work.right - target_width
        max_top = work.bottom - target_height
        if target_left > max_left:
            target_left = max_left
        if target_top > max_top:
            target_top = max_top
        if target_left < work.left:
            target_left = work.left
        if target_top < work.top:
            target_top = work.top

        return bool(user32.MoveWindow(hwnd, target_left, target_top, target_width, target_height, True))
    except Exception:
        return False


def _nudge_windows_terminal_window() -> bool:
    if sys.platform != "win32":
        return False
    try:
        import ctypes.wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        class RECT(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long),
            ]

        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            hwnd = kernel32.GetConsoleWindow()
        if not hwnd or user32.IsZoomed(hwnd):
            return False

        rect = RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return False

        left = int(rect.left)
        top = int(rect.top)
        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
        if width < 32 or height < 16:
            return False

        if not user32.MoveWindow(hwnd, left, top, width + 1, height, True):
            return False
        time.sleep(0.03)
        return bool(user32.MoveWindow(hwnd, left, top, width, height, True))
    except Exception:
        return False


def _target_terminal_size() -> tuple[int, int]:
    try:
        current = shutil.get_terminal_size(fallback=(80, 24))
    except Exception:
        current = os.terminal_size((80, 24))
    target_cols = max(current.columns, MIN_TERMINAL_COLUMNS)
    target_lines = max(current.lines, MIN_TERMINAL_LINES)
    return target_cols, target_lines


def _resize_windows_console(cols: int, lines: int) -> bool:
    if sys.platform != "win32":
        return False
    try:
        result = subprocess.run(
            ["cmd", "/c", f"mode con: cols={cols} lines={lines}"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        return result.returncode == 0
    except Exception:
        return False


def _resize_posix_terminal(cols: int, lines: int) -> bool:
    if sys.platform == "win32":
        return False
    if not sys.stdout.isatty():
        return False
    try:
        sys.stdout.write(f"\x1b[8;{lines};{cols}t")
        sys.stdout.flush()
        return True
    except Exception:
        return False


def _ensure_startup_terminal_size() -> None:
    if not sys.stdout.isatty():
        return
    if sys.platform == "darwin" and _resize_macos_terminal_window_to_quarter_screen():
        return
    if sys.platform == "win32":
        return
    cols, lines = _target_terminal_size()
    _resize_posix_terminal(cols, lines)


_WINDOWS_CONSOLE_INPUT_MODE: int | None = None
STD_INPUT_HANDLE = -10
ENABLE_QUICK_EDIT_MODE = 0x0040
ENABLE_EXTENDED_FLAGS = 0x0080
ENABLE_MOUSE_INPUT = 0x0010


def _configure_windows_console_for_tui() -> None:
    global _WINDOWS_CONSOLE_INPUT_MODE
    if sys.platform != "win32" or not sys.stdout.isatty():
        return
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(STD_INPUT_HANDLE)
        if handle in (0, -1):
            return
        mode = ctypes.c_uint()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return
        _WINDOWS_CONSOLE_INPUT_MODE = int(mode.value)
        new_mode = (_WINDOWS_CONSOLE_INPUT_MODE | ENABLE_EXTENDED_FLAGS | ENABLE_MOUSE_INPUT) & ~ENABLE_QUICK_EDIT_MODE
        kernel32.SetConsoleMode(handle, new_mode)
    except Exception:
        return


def _restore_windows_console_mode() -> None:
    if sys.platform != "win32" or _WINDOWS_CONSOLE_INPUT_MODE is None:
        return
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(STD_INPUT_HANDLE)
        if handle in (0, -1):
            return
        kernel32.SetConsoleMode(handle, _WINDOWS_CONSOLE_INPUT_MODE)
    except Exception:
        return


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BACKEND_PORT = 8000
FRONTEND_PORT = 3000
_WORKER_PLANES = (
    ("WORKERS", "all"),
)
_WORKER_SOURCE_TAG_BY_PLANE = {plane_name: source_tag for source_tag, plane_name in _WORKER_PLANES}
_WORKER_PLANE_BY_NAME: dict[str, str] = {
    "scanner": "all",
    "scanner_slo": "all",
    "crypto": "all",
    "trader_orchestrator": "all",
    "tracked_traders": "all",
    "discovery": "all",
    "weather": "all",
    "news": "all",
    "events": "all",
    "trader_reconciliation": "all",
    "redeemer": "all",
}
# Use 127.0.0.1 instead of localhost; on Windows, localhost can resolve to
# ::1 (IPv6) first while uvicorn only binds 0.0.0.0 (IPv4), causing timeouts.
HEALTH_URL = f"http://127.0.0.1:{BACKEND_PORT}/health/tui"
PROJECT_ROOT = Path(__file__).parent.resolve()
BACKEND_DIR = PROJECT_ROOT / "backend"

LOG_MAX_LINES = 5000
LOG_TRIM_BATCH = 1000  # Remove this many lines when cap is hit
LOG_FLUSH_MS = 500  # Flush buffered lines every N ms
HEALTH_POLL_INTERVAL_SECONDS = 3.0
HEALTH_REQUEST_TIMEOUT_SECONDS = 2.5
TUI_STARTUP_DEBUG_PATH = PROJECT_ROOT / "data" / "tui_startup_debug.jsonl"
TUI_STARTUP_DEBUG_WINDOW_SECONDS = 90.0
HEALTH_FAILURES_BEFORE_OFFLINE = 2
HEALTH_OFFLINE_GRACE_SECONDS = 10.0

# Log level ordering for filter comparison
LOG_LEVEL_ORDER = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
OSC_ESCAPE_RE = re.compile(r"\x1B\].*?(?:\x07|\x1B\\)")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

WORKER_STATUS_ORDER: list[tuple[str, str]] = [
    ("scanner", "SCANNER"),
    ("scanner_slo", "SCANNER SLO"),
    ("discovery", "DISCOVERY"),
    ("weather", "WEATHER"),
    ("news", "NEWS"),
    ("crypto", "CRYPTO"),
    ("tracked_traders", "TRACKED"),
    ("trader_orchestrator", "ORCHESTRATOR"),
    ("trader_reconciliation", "RECONCILIATION"),
    ("redeemer", "REDEEMER"),
    ("events", "EVENTS"),
]

WORKER_TAG_TO_NAME: dict[str, str] = {
    "SCANNER": "scanner",
    "SCANNER SLO": "scanner_slo",
    "DISCOVERY": "discovery",
    "WEATHER": "weather",
    "NEWS": "news",
    "CRYPTO": "crypto",
    "TRACKED": "tracked_traders",
    "ORCHESTRATOR": "trader_orchestrator",
    "RECONCILIATION": "trader_reconciliation",
    "REDEEMER": "redeemer",
    "EVENTS": "events",
}

WORKER_BACKEND_HINTS: tuple[tuple[str, str], ...] = (
    # Workers run in-process inside the API lifespan (Phase 4).
    # Backend log lines are matched to worker panels via these hints.
    ("scanner", "scanner"),
    ("scanner_slo", "scanner_slo"),
    ("discovery", "discovery"),
    ("weather", "weather"),
    ("news_worker", "news"),
    ("crypto_worker", "crypto"),
    ("tracked_traders", "tracked_traders"),
    ("orchestrator", "trader_orchestrator"),
    ("reconciliation", "trader_reconciliation"),
    ("redeemer_worker", "redeemer"),
    ("events_worker", "events"),
)

SOURCE_FILTER_BY_BUTTON: dict[str, str] = {
    "src-all": "all",
    "src-backend": "backend",
    "src-workers": "workers",
    "src-frontend": "frontend",
    "src-system": "system",
}

SOURCE_FILTER_LABELS: dict[str, str] = {
    "all": "All Sources",
    "backend": "Backend",
    "workers": "Workers",
    "frontend": "Frontend",
    "system": "System",
}

LEVEL_FILTER_BY_BUTTON: dict[str, str] = {
    "lvl-all": "all",
    "lvl-error": "error",
    "lvl-warning": "warning",
    "lvl-info": "info",
    "lvl-debug": "debug",
}

LEVEL_FILTER_LABELS: dict[str, str] = {
    "all": "All Levels",
    "error": "Error+",
    "warning": "Warning+",
    "info": "Info+",
    "debug": "Debug+",
}

WORKER_FILTER_ORDER: tuple[str, ...] = (
    "all",
    "none",
    *[name for name, _ in WORKER_STATUS_ORDER],
)
WORKER_FILTER_LABELS: dict[str, str] = {
    "all": "All Workers",
    "none": "No Worker",
    **{name: label for name, label in WORKER_STATUS_ORDER},
}

WORKER_MINI_LOG_LINES = 2
WORKER_PANEL_HORIZONTAL_CHROME = 6
WORKER_PANEL_FALLBACK_CONTENT_WIDTH = 24
WORKER_PANEL_HEIGHT = 7
WORKER_GRID_GUTTER = 1
HOME_NON_GRID_HEIGHT = 20
MIN_TERMINAL_COLUMNS = 80
MIN_TERMINAL_LINES = 24

LOGO = r"""
 _   _  ___  __  __ _____ ____  _   _ _   _
| | | |/ _ \|  \/  | ____|  _ \| | | | \ | |
| |_| | | | | |\/| |  _| | |_) | | | |  \| |
|  _  | |_| | |  | | |___|  _ <| |_| | |\  |
|_| |_|\___/|_|  |_|_____|_| \_\\___/|_| \_|
""".strip(
    "\n"
)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
CSS = """
Screen {
    background: #08111a;
    color: #d8e5ef;
}

#logo {
    color: #58f1c1;
    text-style: bold;
    text-align: left;
    padding: 0 0 0 1;
}

#subtitle {
    color: #87a9bf;
    text-align: left;
    padding: 0 0 0 1;
}

/* ---- Home hero ---- */
#hero-row {
    layout: horizontal;
    margin: 1 1 0 1;
    height: 11;
}

#brand-panel {
    width: 2fr;
    background: #0e1a28;
    border: round #28445f;
    margin: 0 1 0 0;
}

#action-bar {
    layout: horizontal;
    height: 3;
    padding: 0 1;
    margin: 0 0 0 0;
}

#action-bar Button {
    margin: 0 1 0 1;
    min-width: 14;
}

#platform-panel {
    width: 1fr;
    background: #112333;
    border: round #325b77;
    padding: 0 1;
}

#platform-title {
    color: #b6d3e6;
    text-style: bold;
    padding: 0 0 0 0;
}

.platform-item {
    color: #9ec4da;
    height: 1;
    padding: 0 0 0 0;
}

.platform-url {
    color: #7ebee6;
    height: 1;
    padding: 0 0 0 0;
}

/* ---- Runtime metrics bar ---- */
#metrics-bar {
    layout: horizontal;
    height: 1;
    margin: 1 2 0 2;
    color: #8cb3c8;
}

.metric-item {
    width: auto;
    margin: 0 2 0 0;
}

.metric-value {
    color: #58f1c1;
    text-style: bold;
}

.metric-label {
    color: #6e94ab;
}

/* ---- Worker command center ---- */
#workers-title {
    color: #b5d4e7;
    text-style: bold;
    margin: 1 2 0 2;
    height: 1;
}

#workers-grid {
    layout: grid;
    grid-size: 4;
    grid-gutter: 1;
    padding: 0 1;
    margin: 0 1 1 1;
    height: 1fr;
    overflow-y: auto;
    overflow-x: hidden;
}

.worker-panel {
    height: 7;
    background: #0f1d2c;
    border: round #2b4961;
    padding: 0 1;
}

.worker-panel-title {
    color: #d2e8f7;
    text-style: bold;
    height: 1;
    text-wrap: nowrap;
    text-overflow: ellipsis;
}

.worker-panel-status {
    text-style: bold;
    height: 1;
}

.worker-panel-meta {
    color: #89acc3;
    height: 1;
    text-wrap: nowrap;
    text-overflow: ellipsis;
}

.worker-panel-logs {
    color: #9ab4c6;
    height: 3;
    padding: 0 0 0 0;
    text-wrap: nowrap;
    text-overflow: ellipsis;
}

.status-on {
    color: #55f0b8;
}

.status-off {
    color: #ff6a6a;
}

.status-warn {
    color: #ffd166;
}

.status-idle {
    color: #8faec0;
}

/* ---- Uptime ---- */
#uptime-bar {
    height: 1;
    margin: 0 2 1 2;
    color: #7f98ac;
    text-align: center;
    text-wrap: nowrap;
    text-overflow: ellipsis;
}

/* ---- Logs pane ---- */
#log-pane {
    height: 1fr;
}

#log-header {
    layout: horizontal;
    height: 1;
    padding: 0 1;
    background: #0f1d2c;
    margin: 0 1 0 1;
}

#log-header-left {
    width: 1fr;
}

#log-header-right {
    width: auto;
    min-width: 40;
    text-align: right;
}

#log-controls {
    layout: vertical;
    height: auto;
    max-height: 8;
    padding: 0 1;
    margin: 0 1 0 1;
}

.log-controls-row {
    layout: horizontal;
    height: 3;
    margin: 0;
}

#log-controls Button {
    min-width: 10;
    margin: 0 1 0 0;
}

#log-worker-prev-btn,
#log-worker-next-btn {
    width: 5;
    min-width: 5;
    margin: 0 1 0 0;
}

#log-worker-filter-label {
    width: 26;
    min-width: 18;
    padding: 1 1 0 1;
    margin: 0 1 0 0;
    border: round #2b4961;
}

#log-search-input {
    width: 1fr;
    min-width: 24;
    margin: 0 1 0 0;
}

#log-bottom-btn {
    display: none;
}

#log-bottom-btn.visible {
    display: block;
}

#log-output {
    margin: 0 1;
    border: round #2b4961;
    scrollbar-size: 1 1;
}

/* ---- Tabs ---- */
TabbedContent {
    height: 1fr;
}

TabPane {
    padding: 0;
}


#home {
    layout: vertical;
    overflow-y: hidden;
}

/* ---- Light mode ---- */
Screen.light-mode {
    background: #f0f2f5;
    color: #1a1a2e;
}

Screen.light-mode #logo {
    color: #0d7a52;
}

Screen.light-mode #subtitle {
    color: #5a6e7f;
}

Screen.light-mode #brand-panel {
    background: #e4e8ec;
    border: round #b0bec5;
}

Screen.light-mode #platform-panel {
    background: #dce3e8;
    border: round #90a4ae;
}

Screen.light-mode #platform-title {
    color: #37474f;
}

Screen.light-mode .platform-item {
    color: #455a64;
}

Screen.light-mode .platform-url {
    color: #1565c0;
}

Screen.light-mode .worker-panel {
    background: #e8ecf0;
    border: round #b0bec5;
}

Screen.light-mode .worker-panel-title {
    color: #263238;
}

Screen.light-mode .worker-panel-meta {
    color: #546e7a;
}

Screen.light-mode .worker-panel-logs {
    color: #607d8b;
}

Screen.light-mode #workers-title {
    color: #37474f;
}

Screen.light-mode .metric-value {
    color: #0d7a52;
}

Screen.light-mode .metric-label {
    color: #607d8b;
}

Screen.light-mode #metrics-bar {
    color: #546e7a;
}

Screen.light-mode #uptime-bar {
    color: #607d8b;
}

Screen.light-mode #log-output {
    border: round #b0bec5;
}

Screen.light-mode #log-header {
    background: #e4e8ec;
}

Screen.light-mode .status-on {
    color: #0d7a52;
}

Screen.light-mode .status-off {
    color: #c62828;
}

Screen.light-mode .status-warn {
    color: #e65100;
}

Screen.light-mode .status-idle {
    color: #78909c;
}
"""


# ---------------------------------------------------------------------------
# Helper to kill processes on a port
# ---------------------------------------------------------------------------
def kill_port(port: int) -> None:
    """Kill any process listening on the given port."""
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in result.stdout.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    pid = int(parts[-1])
                    if pid == os.getpid() or pid == 0:
                        continue
                    try:
                        subprocess.run(
                            ["taskkill", "/F", "/PID", str(pid)],
                            capture_output=True,
                            timeout=5,
                        )
                    except Exception:
                        pass
            time.sleep(0.5)
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, OSError):
            pass
    else:
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            pids = result.stdout.strip()
            if pids:
                for pid_str in pids.split("\n"):
                    pid = int(pid_str.strip())
                    if pid == os.getpid():
                        continue
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                time.sleep(0.5)
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, OSError):
            try:
                subprocess.run(
                    ["fuser", "-k", f"{port}/tcp"],
                    capture_output=True,
                    timeout=5,
                )
                time.sleep(0.5)
            except Exception:
                pass


def kill_legacy_worker_processes() -> None:
    """Stop detached legacy worker processes from pre in-process runs."""
    if sys.platform == "win32":
        backend_dir = str(BACKEND_DIR)
        project_root = str(PROJECT_ROOT)
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_Process "
                    "| Where-Object { $_.Name -eq 'python.exe' -or $_.Name -eq 'pythonw.exe' } "
                    "| Select-Object ProcessId,CommandLine "
                    "| ConvertTo-Json -Compress",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return

            data = json.loads(result.stdout)
            if isinstance(data, dict):
                data = [data]

            for entry in data:
                pid = entry.get("ProcessId")
                cmd = str(entry.get("CommandLine") or "")
                if not pid or pid == os.getpid():
                    continue
                if backend_dir not in cmd and project_root not in cmd:
                    continue
                is_homerun = False
                if "workers.host" in cmd:
                    is_homerun = True
                elif "workers.runner" in cmd or ("workers." in cmd and "_worker" in cmd):
                    is_homerun = True
                elif "uvicorn" in cmd and "main:app" in cmd:
                    is_homerun = True
                if not is_homerun:
                    continue
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        capture_output=True,
                        timeout=5,
                    )
                except Exception:
                    pass
        except Exception:
            pass
        time.sleep(0.25)
        return
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,command="],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return

    for line in result.stdout.splitlines():
        entry = line.strip()
        if not entry:
            continue
        parts = entry.split(None, 1)
        if len(parts) != 2:
            continue
        pid_text, command = parts
        if " -m workers." not in command:
            continue
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except OSError:
            continue

    time.sleep(0.25)


class WorkerPanel(Static):
    """Compact worker telemetry panel with mini logs."""

    def __init__(self, worker_name: str, label: str) -> None:
        super().__init__("", id=f"worker-{worker_name}", classes="worker-panel")
        self.worker_name = worker_name
        self._label = label
        self.border_title = label
        self._status = "OFFLINE"
        self._status_class = "status-off"
        self._meta = "No telemetry yet"
        self._logs = ["waiting for worker events", "--"]
        self._refresh_content()

    def _status_markup(self) -> str:
        color = {
            "status-on": "#55f0b8",
            "status-off": "#ff6b6b",
            "status-warn": "#ffbf69",
            "status-idle": "#8aa7ba",
        }.get(self._status_class, "#d2e8f7")
        return f"[{color} bold]{self._status}[/]"

    def _refresh_content(self) -> None:
        log_lines = self._logs[-WORKER_MINI_LOG_LINES:]
        while len(log_lines) < WORKER_MINI_LOG_LINES:
            log_lines.append("--")
        body = "\n".join(
            [
                self._status_markup(),
                f"[#89acc3]{self._meta}[/]",
                f"[#9ab4c6]  {log_lines[0]}[/]",
                f"[#9ab4c6]  {log_lines[1]}[/]",
            ]
        )
        self.update(body)

    def update_state(self, status: str, status_class: str, meta: str) -> None:
        self._status = status
        self._status_class = status_class
        self._meta = meta
        self._refresh_content()

    def update_logs(self, lines: list[str]) -> None:
        self._logs = lines[-WORKER_MINI_LOG_LINES:] if lines else ["waiting for worker events", "--"]
        self._refresh_content()


# ---------------------------------------------------------------------------
# Plain-text log formatter (no Rich markup)
# ---------------------------------------------------------------------------
def format_log_line(line: str, tag: str) -> tuple[str, str]:
    """Format a raw log line into readable plain text for the log viewer.

    Returns (formatted_text, level).
    """
    line = OSC_ESCAPE_RE.sub("", line)
    line = CONTROL_CHAR_RE.sub("", ANSI_ESCAPE_RE.sub("", line.replace("\r", "")))
    prefix = f"[{tag}]"

    # Try to parse structured JSON log lines from backend
    if line.startswith("{"):
        try:
            data = json.loads(line)
            level = data.get("level", "INFO").upper()
            msg = data.get("message", line)
            logger_name = data.get("logger", "")
            func = data.get("function", "")
            ts = data.get("timestamp", "")
            extra = data.get("data")

            ts_short = ts[11:19] if len(ts) >= 19 else ts
            parts = f"{prefix} {ts_short} {level:<8s} {logger_name}.{func}  {msg}"
            if extra and isinstance(extra, dict):
                kv = " ".join(f"{k}={v}" for k, v in extra.items())
                parts += f"  {kv}"
            return (parts, level)
        except (json.JSONDecodeError, KeyError):
            pass

    # Detect level from plain text
    upper = line.upper()
    if any(kw in upper for kw in ("ERROR", "FATAL", "CRITICAL", "TRACEBACK", "EXCEPTION")):
        level = "ERROR"
    elif "WARN" in upper:
        level = "WARNING"
    elif "DEBUG" in upper:
        level = "DEBUG"
    else:
        level = "INFO"

    return (f"{prefix} {line}", level)


# ---------------------------------------------------------------------------
# Main TUI App
# ---------------------------------------------------------------------------
class HomerunApp(App):
    """Homerun Trading Platform TUI."""

    TITLE = "HOMERUN"
    SUB_TITLE = "Autonomous Prediction Market Trading Platform"
    CSS = CSS
    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("h", "show_tab('home')", "Home", show=True, priority=True),
        Binding("l", "show_tab('logs')", "Logs", show=True, priority=True),
        Binding("d", "toggle_dark", "Dark/Light", show=True, priority=True),
        Binding("slash", "command_palette", "Search", show=True, priority=True),
        Binding("r", "do_restart", "Restart", show=True, priority=True),
        Binding("u", "do_update", "Update", show=False, priority=True),
        Binding("question_mark", "show_help", "Help", show=True, priority=True),
        Binding("ctrl+c", "copy_to_clip", "Copy", show=False, priority=True),
    ]

    # Process handles.
    backend_proc: Optional[subprocess.Popen] = None
    worker_procs: dict[str, subprocess.Popen] = {}
    frontend_proc: Optional[subprocess.Popen] = None

    # State
    start_time: float = 0.0
    backend_healthy: bool = False
    health_data: dict = {}
    health_poll_count: int = 0
    _is_light_mode: bool = False
    _startup_layout_settle_deadline: float = 0.0
    _last_known_viewport_size: tuple[int, int] = (0, 0)
    _startup_layout_timer = None
    _startup_services_started: bool = False
    _startup_debug_enabled: bool = False
    _startup_debug_started_at: float = 0.0

    def __init__(self) -> None:
        super().__init__()
        self.worker_procs = {}
        # Thread-safe log line buffer: worker threads append here,
        # a periodic timer flushes into the log view on the main thread.
        self._log_buf: collections.deque[tuple[str, str, str, str]] = collections.deque(
            maxlen=2000
        )
        self._log_lock = threading.Lock()
        # Master list of all log entries for rebuilding filtered views.
        self._log_entries: list[tuple[str, str, str, str]] = []  # (text, source, level, worker)
        self._log_line_count = 0
        # Track whether user is scrolled to the bottom (auto-follow mode).
        self._log_follow = True
        # Filter state
        self._source_filter = "all"  # "all", "backend", "workers", "frontend", "system"
        self._level_filter = "all"  # "all", "debug", "info", "warning", "error"
        self._worker_filter = "all"  # "all", "none", "scanner", ...
        self._search_filter = ""
        # Shutdown flag so reader threads can exit
        self._shutting_down = False
        # Guard against starting frontend twice (race between worker thread and @work)
        self._frontend_starting = False
        # Prevent concurrent restart/update operations.
        self._service_op_in_progress = False
        # Worker telemetry buffers for Home mini logs.
        self._worker_logs: dict[str, collections.deque[str]] = {
            name: collections.deque(maxlen=12) for name, _ in WORKER_STATUS_ORDER
        }
        self._worker_event_buf: collections.deque[tuple[str, str]] = collections.deque(
            maxlen=2000
        )
        self._worker_state_cache: dict[str, dict] = {}
        self._worker_last_state: dict[str, str] = {}
        self._worker_last_activity: dict[str, str] = {}
        self._worker_last_error: dict[str, str] = {}
        self._health_poll_lock = threading.Lock()
        self._health_poll_inflight = False
        self._worker_supervisor_lock = threading.Lock()
        self._worker_supervisor_inflight = False
        self._health_last_success_monotonic = 0.0
        self._health_consecutive_failures = 0
        self._startup_debug_enabled = os.environ.get("HOMERUN_TUI_DEBUG_STARTUP") == "1"
        self._startup_debug_started_at = 0.0
        self._last_button_action_at: dict[str, float] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="home"):
            with TabPane("  Home  ", id="home"):
                yield from self._compose_home()
            with TabPane("  Logs  ", id="logs"):
                yield from self._compose_logs()
        yield Footer()

    # ---- Home tab layout ----
    def _compose_home(self) -> ComposeResult:
        with Horizontal(id="hero-row"):
            with Vertical(id="brand-panel"):
                yield Static(LOGO, id="logo")
                yield Static(
                    "Autonomous Prediction Market Trading Platform",
                    id="subtitle",
                )
                with Horizontal(id="action-bar"):
                    yield Button("Restart", id="restart-btn", variant="warning")
                    yield Button("Update", id="update-btn", variant="success")
            with Vertical(id="platform-panel"):
                yield Static("Platform", id="platform-title")
                yield Static("[red]\u25cf[/] BACKEND   OFFLINE", id="svc-backend", classes="platform-item")
                yield Static("[red]\u25cf[/] DATABASE  OFFLINE", id="svc-database", classes="platform-item")
                yield Static("[red]\u25cf[/] FRONTEND  OFFLINE", id="svc-frontend", classes="platform-item")
                yield Static("[red]\u25cf[/] WS FEEDS  OFFLINE", id="svc-wsfeeds", classes="platform-item")
                yield Static(f"Dashboard  http://localhost:{FRONTEND_PORT}", classes="platform-url")
                yield Static(f"API        http://localhost:{BACKEND_PORT}", classes="platform-url")
                yield Static(f"Docs       http://localhost:{BACKEND_PORT}/docs", classes="platform-url")

        # Runtime metrics bar
        with Horizontal(id="metrics-bar"):
            yield Static("Uptime [bold]--:--:--[/]", id="metric-uptime", classes="metric-item")
            yield Static(f"Workers [bold]0/{len(WORKER_STATUS_ORDER)}[/]", id="metric-workers", classes="metric-item")
            yield Static("Health [bold red]OFFLINE[/]", id="metric-health", classes="metric-item")
            yield Static("Polls [bold]0[/]", id="metric-polls", classes="metric-item")
            yield Static("Logs [bold]0[/]", id="metric-logs", classes="metric-item")

        yield Static("Worker Command Center", id="workers-title")
        with Container(id="workers-grid"):
            for worker_name, worker_label in WORKER_STATUS_ORDER:
                yield WorkerPanel(worker_name, worker_label)

        # Shortcuts bar
        yield Static("", id="uptime-bar")

    # ---- Logs tab layout ----
    def _compose_logs(self) -> ComposeResult:
        with Vertical(id="log-pane"):
            with Horizontal(id="log-header"):
                yield Static(
                    "[bold]LOGS[/]",
                    id="log-header-left",
                )
                yield Static(
                    "[bold green]FOLLOWING[/]  0/0 lines",
                    id="log-header-right",
                )
            with Vertical(id="log-controls"):
                with Horizontal(classes="log-controls-row"):
                    yield Button("All", id="src-all", variant="primary")
                    yield Button("Backend", id="src-backend", variant="default")
                    yield Button("Workers", id="src-workers", variant="default")
                    yield Button("Frontend", id="src-frontend", variant="default")
                    yield Button("System", id="src-system", variant="default")
                    yield Button("Any", id="lvl-all", variant="primary")
                    yield Button("Error+", id="lvl-error", variant="default")
                    yield Button("Warn+", id="lvl-warning", variant="default")
                    yield Button("Info+", id="lvl-info", variant="default")
                    yield Button("Debug+", id="lvl-debug", variant="default")
                with Horizontal(classes="log-controls-row"):
                    yield Button("<", id="log-worker-prev-btn", variant="default")
                    yield Static("Worker: All Workers", id="log-worker-filter-label")
                    yield Button(">", id="log-worker-next-btn", variant="default")
                    yield Input(
                        placeholder="Search logs (message, symbol, error...)",
                        id="log-search-input",
                    )
                    yield Button("Pause", id="log-follow-toggle-btn", variant="default")
                    yield Button("Clear", id="log-clear-btn", variant="warning")
                    yield Button("Copy", id="log-copy-btn", variant="success")
                    yield Button("↓ Bottom", id="log-bottom-btn", variant="primary")
            yield Log(id="log-output", highlight=False, max_lines=LOG_MAX_LINES, auto_scroll=True)

    # ---- Lifecycle ----
    def on_mount(self) -> None:
        self.start_time = time.time()
        self._init_startup_debug_trace()
        self._startup_layout_settle_deadline = time.monotonic() + 3.0
        self._last_known_viewport_size = (int(self.size.width), int(self.size.height))
        self._trace_startup_debug("on_mount:start")
        # Remove "active" press animation delay so UI clicks feel immediate.
        for button in self.query(Button):
            button.active_effect_duration = 0.0
        self._sync_filter_button_variants()
        self._update_worker_filter_display()
        self._update_log_header()
        self._apply_responsive_layout()
        self.set_timer(0.0, self._finalize_initial_layout)
        self.set_timer(0.2, self._finalize_initial_layout)
        self.set_timer(1.0, self._finalize_delayed_startup_layout)
        self.set_timer(2.0, self._finalize_delayed_startup_layout)
        self.set_timer(0.5, self._start_services_after_initial_paint)
        self._startup_layout_timer = self.set_interval(0.1, self._settle_startup_layout)
        self._poll_health()
        self._update_uptime()
        # Flush log buffer periodically (batched writes for performance)
        self.set_interval(LOG_FLUSH_MS / 1000.0, self._flush_log_buffer)
        # Check scroll state less frequently (not on_idle which fires constantly)
        self.set_interval(0.5, self._check_scroll_follow)

    def _init_startup_debug_trace(self) -> None:
        if not self._startup_debug_enabled:
            return
        self._startup_debug_started_at = time.monotonic()
        try:
            TUI_STARTUP_DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)
            if TUI_STARTUP_DEBUG_PATH.exists():
                TUI_STARTUP_DEBUG_PATH.unlink()
        except Exception:
            pass
        self._trace_startup_debug("trace_initialized")

    def _trace_startup_debug(self, event: str, **extra: object) -> None:
        if not self._startup_debug_enabled:
            return
        if self._startup_debug_started_at and (time.monotonic() - self._startup_debug_started_at) > TUI_STARTUP_DEBUG_WINDOW_SECONDS:
            return
        payload: dict[str, object] = {
            "event": event,
            "ts": round(time.time(), 3),
            "mono": round(time.monotonic(), 3),
        }
        try:
            payload["app_size"] = {
                "width": int(self.size.width),
                "height": int(self.size.height),
            }
        except Exception:
            payload["app_size"] = None
        try:
            workers_grid = self.query_one("#workers-grid", Container)
            payload["workers_grid"] = {
                "width": int(workers_grid.size.width),
                "height": int(workers_grid.size.height),
                "grid_columns": getattr(workers_grid.styles, "grid_size_columns", None),
            }
        except Exception:
            payload["workers_grid"] = None
        worker_widths: dict[str, int | None] = {}
        for worker_name, _worker_label in WORKER_STATUS_ORDER[:4]:
            try:
                panel = self.query_one(f"#worker-{worker_name}", WorkerPanel)
                worker_widths[worker_name] = int(panel.size.width)
            except Exception:
                worker_widths[worker_name] = None
        payload["sample_worker_widths"] = worker_widths
        if extra:
            payload.update(extra)
        try:
            with TUI_STARTUP_DEBUG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        except Exception:
            pass

    def _apply_responsive_layout(self, viewport_width: int | None = None) -> None:
        try:
            width = int(viewport_width) if isinstance(viewport_width, int) and viewport_width > 0 else int(self.size.width)
        except Exception:
            return

        if width >= 160:
            grid_cols = 5
        elif width >= 120:
            grid_cols = 4
        elif width >= 88:
            grid_cols = 3
        elif width >= 60:
            grid_cols = 2
        else:
            grid_cols = 1

        try:
            workers_grid = self.query_one("#workers-grid", Container)
            workers_grid.styles.grid_size_columns = grid_cols
        except Exception:
            pass
        self._trace_startup_debug("apply_responsive_layout", viewport_width=viewport_width, resolved_width=width, grid_cols=grid_cols)
        self._rerender_all_worker_panels()

    def _finalize_initial_layout(self) -> None:
        try:
            self.screen.clear_cached_dimensions()
        except Exception:
            pass
        self._apply_responsive_layout()
        self.refresh(repaint=True, layout=True)
        try:
            self.screen.refresh(repaint=True, layout=True)
        except Exception:
            pass
        self._trace_startup_debug("finalize_initial_layout")

    def _finalize_delayed_startup_layout(self) -> None:
        self._finalize_initial_layout()
        self._rerender_all_worker_panels()

    def _start_services_after_initial_paint(self) -> None:
        if self._startup_services_started:
            return
        self._startup_services_started = True
        self._trace_startup_debug("start_services_after_initial_paint")
        self._start_services()

    def _settle_startup_layout(self) -> None:
        if time.monotonic() >= self._startup_layout_settle_deadline:
            try:
                if self._startup_layout_timer is not None:
                    self._startup_layout_timer.stop()
            except Exception:
                pass
            self._startup_layout_timer = None
            return
        current_size = (int(self.size.width), int(self.size.height))
        if current_size != self._last_known_viewport_size:
            self._last_known_viewport_size = current_size
            self._trace_startup_debug("settle_layout:size_changed", current_size={"width": current_size[0], "height": current_size[1]})
            self._finalize_initial_layout()
            return
        for worker_name, _worker_label in WORKER_STATUS_ORDER:
            if self._get_worker_content_width(worker_name) <= 0:
                self._trace_startup_debug("settle_layout:bad_worker_width", worker_name=worker_name)
                self._finalize_initial_layout()
                return

    def _rerender_all_worker_panels(self) -> None:
        for worker_name, _worker_label in WORKER_STATUS_ORDER:
            self._render_worker_panel(worker_name)

    def on_resize(self, event: Resize) -> None:
        viewport_width = getattr(event, "width", None)
        viewport_height = getattr(getattr(event, "size", None), "height", None)
        if not isinstance(viewport_width, int) or viewport_width <= 0:
            maybe_size = getattr(event, "size", None)
            viewport_width = getattr(maybe_size, "width", None)
            viewport_height = getattr(maybe_size, "height", None)
        self._trace_startup_debug(
            "on_resize",
            event_width=viewport_width,
            event_height=viewport_height,
            pixel_size=getattr(event, "pixel_size", None),
        )
        if isinstance(viewport_width, int) and isinstance(viewport_height, int):
            current_size = (viewport_width, viewport_height)
            if current_size == self._last_known_viewport_size:
                return
            self._last_known_viewport_size = current_size
        self._apply_responsive_layout(viewport_width if isinstance(viewport_width, int) else None)
        self.refresh(layout=True)
        self.set_timer(0.0, self._rerender_all_worker_panels)

    def action_show_tab(self, tab: str) -> None:
        self.query_one(TabbedContent).active = tab

    def action_toggle_dark(self) -> None:
        self._is_light_mode = not self._is_light_mode
        if self._is_light_mode:
            self.screen.add_class("light-mode")
        else:
            self.screen.remove_class("light-mode")
        mode = "Light" if self._is_light_mode else "Dark"
        self.notify(f"Switched to {mode} mode", timeout=2)

    def action_do_restart(self) -> None:
        self._restart_services()

    def action_do_update(self) -> None:
        self._update_and_restart()

    def action_show_help(self) -> None:
        shortcuts = (
            "[bold]Keyboard Shortcuts[/]\n"
            "\n"
            "  [bold]h[/]       Switch to Home tab\n"
            "  [bold]l[/]       Switch to Logs tab\n"
            "  [bold]d[/]       Toggle dark/light mode\n"
            "  [bold]/[/]       Open command palette (search)\n"
            "  [bold]r[/]       Restart all services\n"
            "  [bold]u[/]       Update & restart (git pull)\n"
            "  [bold]?[/]       Show this help\n"
            "  [bold]Ctrl+C[/]  Copy logs to clipboard\n"
            "  [bold]q[/]       Quit"
        )
        self.notify(shortcuts, timeout=8)

    # ---- Copy action ----
    def _do_copy(self) -> None:
        """Copy selected text (or all text) from the log viewer to system clipboard."""
        try:
            log = self.query_one("#log-output", Log)
            text = "\n".join(log.lines)
            if not text:
                return
            # Use system clipboard via subprocess (reliable across terminals)
            copied = False
            if sys.platform == "darwin":
                try:
                    subprocess.run(
                        ["pbcopy"],
                        input=text.encode("utf-8"),
                        check=True,
                        timeout=3,
                    )
                    copied = True
                except Exception:
                    pass
            else:
                for cmd in (
                    ["xclip", "-selection", "clipboard"],
                    ["xsel", "--clipboard", "--input"],
                ):
                    try:
                        subprocess.run(
                            cmd,
                            input=text.encode("utf-8"),
                            check=True,
                            timeout=3,
                        )
                        copied = True
                        break
                    except Exception:
                        pass
            if not copied:
                # Fall back to Textual's OSC 52 (terminal-dependent)
                self.copy_to_clipboard(text)
            self.notify("Copied to clipboard", timeout=2)
        except Exception:
            pass

    def action_copy_to_clip(self) -> None:
        self._do_copy()

    # ---- Log filter helpers ----
    def _source_bucket(self, source: str) -> str:
        source_upper = source.upper()
        if source_upper == "FRONTEND":
            return "frontend"
        if source_upper == "SYSTEM":
            return "system"
        if source_upper in WORKER_TAG_TO_NAME or source_upper.startswith("WORKER-"):
            return "workers"
        return "backend"

    def _matches_filter(self, text: str, source: str, level: str, worker_name: str) -> bool:
        """Check if a log entry matches the current filters."""
        if self._source_filter != "all":
            if self._source_bucket(source) != self._source_filter:
                return False
        if self._level_filter != "all":
            min_level = LOG_LEVEL_ORDER.get(self._level_filter.upper(), 0)
            entry_level = LOG_LEVEL_ORDER.get(level.upper(), 1)
            if entry_level < min_level:
                return False
        if self._worker_filter == "none":
            if worker_name:
                return False
        elif self._worker_filter != "all":
            if worker_name != self._worker_filter:
                return False
        if self._search_filter:
            if self._search_filter not in text.lower():
                return False
        return True

    def _worker_filter_text(self) -> str:
        label = WORKER_FILTER_LABELS.get(self._worker_filter)
        if label:
            return label
        return self._worker_filter.replace("_", " ").upper()

    def _cycle_worker_filter(self, direction: int) -> None:
        try:
            idx = WORKER_FILTER_ORDER.index(self._worker_filter)
        except ValueError:
            idx = 0
        self._worker_filter = WORKER_FILTER_ORDER[
            (idx + direction) % len(WORKER_FILTER_ORDER)
        ]
        self._update_worker_filter_display()
        self._rebuild_log_view()

    def _update_worker_filter_display(self) -> None:
        try:
            self.query_one("#log-worker-filter-label", Static).update(
                f"Worker: {self._worker_filter_text()}"
            )
        except Exception:
            pass

    def _format_filter_summary(self) -> str:
        source = SOURCE_FILTER_LABELS.get(self._source_filter, self._source_filter)
        level = LEVEL_FILTER_LABELS.get(self._level_filter, self._level_filter)
        worker = self._worker_filter_text()
        summary = f"source={source}  level={level}  worker={worker}"
        if self._search_filter:
            query = self._search_filter
            if len(query) > 40:
                query = query[:37] + "..."
            summary += f"  search='{query}'"
        return summary

    def _sync_filter_button_variants(self) -> None:
        for button_id, value in SOURCE_FILTER_BY_BUTTON.items():
            try:
                button = self.query_one(f"#{button_id}", Button)
                button.variant = "primary" if self._source_filter == value else "default"
            except Exception:
                pass
        for button_id, value in LEVEL_FILTER_BY_BUTTON.items():
            try:
                button = self.query_one(f"#{button_id}", Button)
                button.variant = "primary" if self._level_filter == value else "default"
            except Exception:
                pass

    def _rebuild_log_view(self) -> None:
        """Rebuild the log content from master entries based on current filters."""
        try:
            log = self.query_one("#log-output", Log)
        except Exception:
            return

        matching = [
            text
            for text, source, level, worker_name in self._log_entries
            if self._matches_filter(text, source, level, worker_name)
        ]

        log.clear()

        if matching:
            log.auto_scroll = bool(self._log_follow)
            log.write_lines(matching)

        self._log_line_count = len(log.lines)
        if self._log_follow:
            log.scroll_end(animate=False)
        self._update_log_header()

    # ---- Log buffer: thread-safe batched writes ----
    def _enqueue_log(self, text: str, source: str = "BACKEND", level: str = "INFO") -> None:
        """Called from worker threads. Appends to buffer; main-thread timer flushes."""
        worker_name = self._infer_worker_from_log(source, text) or ""
        with self._log_lock:
            self._log_buf.append((text, source, level, worker_name))
            if worker_name:
                self._worker_event_buf.append((worker_name, text))

    def _flush_log_buffer(self) -> None:
        """Called on the main thread by a periodic timer. Flushes pending lines
        into the log view in one batch for performance."""
        with self._log_lock:
            if not self._log_buf:
                return
            entries = list(self._log_buf)
            self._log_buf.clear()

        # Store in master list
        self._log_entries.extend(entries)

        # Trim master list if it exceeds the cap
        if len(self._log_entries) > LOG_MAX_LINES:
            self._log_entries = self._log_entries[-(LOG_MAX_LINES - LOG_TRIM_BATCH) :]

        # Filter new entries for current display
        matching = [
            text
            for text, source, level, worker_name in entries
            if self._matches_filter(text, source, level, worker_name)
        ]

        if not matching:
            self._flush_worker_event_buffer()
            return

        try:
            log = self.query_one("#log-output", Log)
        except Exception:
            self._flush_worker_event_buffer()
            return

        at_bottom = self._log_follow
        saved_scroll_y = log.scroll_y
        log.auto_scroll = bool(self._log_follow)
        log.write_lines(matching)
        self._log_line_count = len(log.lines)

        if at_bottom:
            log.scroll_end(animate=False)
        else:
            log.scroll_to(y=max(0, saved_scroll_y), animate=False)

        # Update header with line count & follow state
        self._update_log_header()
        self._flush_worker_event_buffer()

    def _flush_worker_event_buffer(self) -> None:
        with self._log_lock:
            if not self._worker_event_buf:
                return
            events = list(self._worker_event_buf)
            self._worker_event_buf.clear()

        touched: set[str] = set()
        for worker_name, raw_line in events:
            self._append_worker_log(worker_name, raw_line, update=False)
            touched.add(worker_name)
        for worker_name in touched:
            self._render_worker_panel(worker_name)

    def _update_log_header(self) -> None:
        follow_label = (
            "[bold green]FOLLOWING[/]"
            if self._log_follow
            else "[bold yellow]PAUSED[/]"
        )
        try:
            self.query_one("#log-header-left", Static).update(
                f"[bold]LOGS[/]  [dim]{self._format_filter_summary()}[/]"
            )
        except Exception:
            pass
        try:
            self.query_one("#log-header-right", Static).update(
                f"{follow_label}  {self._log_line_count:,}/{len(self._log_entries):,} lines"
            )
        except Exception:
            pass
        try:
            follow_button = self.query_one("#log-follow-toggle-btn", Button)
            if self._log_follow:
                follow_button.label = "Pause"
                follow_button.variant = "default"
            else:
                follow_button.label = "Resume"
                follow_button.variant = "primary"
        except Exception:
            pass
        # Show/hide the snap-to-bottom button
        try:
            btn = self.query_one("#log-bottom-btn", Button)
            if self._log_follow:
                btn.remove_class("visible")
            else:
                btn.add_class("visible")
        except Exception:
            pass

    def _check_scroll_follow(self) -> None:
        """Periodic check: detect when the user scrolls to/from the bottom."""
        try:
            log = self.query_one("#log-output", Log)
            if log.max_scroll_y <= 0:
                return
            at_bottom = log.scroll_y >= (log.max_scroll_y - 2)
            if at_bottom != self._log_follow:
                self._log_follow = at_bottom
                self._update_log_header()
        except Exception:
            pass

    def _infer_worker_from_log(self, source: str, text: str) -> Optional[str]:
        source_upper = source.upper()
        if source_upper.startswith("WORKER-"):
            return None
        direct = WORKER_TAG_TO_NAME.get(source_upper)
        if direct:
            return direct
        if source_upper not in {"BACKEND", "WORKERS"}:
            return None
        lowered = text.lower()
        for hint, worker_name in WORKER_BACKEND_HINTS:
            if hint in lowered:
                return worker_name
        return None

    def _append_worker_log(self, worker_name: str, text: str, update: bool = True) -> None:
        logs = self._worker_logs.get(worker_name)
        if logs is None:
            return
        line = self._normalize_worker_log_line(text)
        if not line:
            return
        if logs and logs[-1] == line:
            return
        logs.append(line)
        if update:
            self._render_worker_panel(worker_name)

    def _truncate_worker_text(self, text: str, max_width: int) -> str:
        cleaned = " ".join(text.strip().split())
        if not cleaned:
            return ""
        if max_width <= 0:
            max_width = WORKER_PANEL_FALLBACK_CONTENT_WIDTH
        if len(cleaned) <= max_width:
            return cleaned
        if max_width <= 3:
            return cleaned[:max_width]
        return cleaned[: max_width - 3].rstrip() + "..."

    def _get_worker_content_width(self, worker_name: str, panel: WorkerPanel | None = None) -> int:
        try:
            target = panel or self.query_one(f"#worker-{worker_name}", WorkerPanel)
            width = int(target.size.width)
        except Exception:
            width = 0
        if width <= WORKER_PANEL_HORIZONTAL_CHROME:
            return WORKER_PANEL_FALLBACK_CONTENT_WIDTH
        return width - WORKER_PANEL_HORIZONTAL_CHROME

    def _normalize_worker_log_line(self, text: str, max_width: int | None = None) -> str:
        line = OSC_ESCAPE_RE.sub("", text.replace("\r", "")).strip()
        line = CONTROL_CHAR_RE.sub("", ANSI_ESCAPE_RE.sub("", line))
        if line.startswith("[") and "] " in line:
            line = line.split("] ", 1)[1]
        return self._truncate_worker_text(line, max_width or 0)

    # ---- Button handlers ----
    def _should_handle_button_action(self, btn_id: Optional[str], now: float | None = None) -> bool:
        if not btn_id:
            return False
        current = time.monotonic() if now is None else now
        previous = self._last_button_action_at.get(btn_id, 0.0)
        if current - previous < 0.35:
            return False
        self._last_button_action_at[btn_id] = current
        return True

    def _handle_button_action(self, btn_id: Optional[str]) -> None:
        if not btn_id:
            return

        if btn_id == "log-worker-prev-btn":
            self._cycle_worker_filter(-1)
            return

        if btn_id == "log-worker-next-btn":
            self._cycle_worker_filter(1)
            return

        # Source filter buttons
        source_filter = SOURCE_FILTER_BY_BUTTON.get(btn_id)
        if source_filter is not None:
            self._source_filter = source_filter
            self._sync_filter_button_variants()
            self._rebuild_log_view()
            return

        # Level filter buttons
        level_filter = LEVEL_FILTER_BY_BUTTON.get(btn_id)
        if level_filter is not None:
            self._level_filter = level_filter
            self._sync_filter_button_variants()
            self._rebuild_log_view()
            return

        if btn_id == "log-clear-btn":
            self._log_entries.clear()
            try:
                self.query_one("#log-output", Log).clear()
            except Exception:
                pass
            self._log_line_count = 0
            self._log_follow = True
            self._update_log_header()
            self.notify("Logs cleared", timeout=2)
            return

        if btn_id == "log-copy-btn":
            self._do_copy()
            return

        if btn_id == "log-follow-toggle-btn":
            self._log_follow = not self._log_follow
            if self._log_follow:
                try:
                    log = self.query_one("#log-output", Log)
                    log.auto_scroll = True
                    log.scroll_end(animate=False)
                except Exception:
                    pass
            self._update_log_header()
            return

        if btn_id == "log-bottom-btn":
            try:
                log = self.query_one("#log-output", Log)
                log.auto_scroll = True
                log.scroll_end(animate=False)
                self._log_follow = True
                self._update_log_header()
            except Exception:
                pass
            return

        if btn_id == "restart-btn":
            self._restart_services()
            return

        if btn_id == "update-btn":
            self._update_and_restart()
            return

    @on(MouseDown, "Button")
    def _on_button_mouse_down(self, event: MouseDown) -> None:
        button = event.widget if isinstance(event.widget, Button) else None
        if button is None or button.disabled:
            return
        if getattr(event, "button", 1) != 1:
            return
        if not self._should_handle_button_action(button.id):
            return
        self._handle_button_action(button.id)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if not self._should_handle_button_action(event.button.id):
            return
        self._handle_button_action(event.button.id)

    @on(Input.Changed, "#log-search-input")
    def _on_search_changed(self, event: Input.Changed) -> None:
        self._search_filter = event.value.strip().lower()
        self._rebuild_log_view()

    def _set_action_buttons_enabled(self, enabled: bool) -> None:
        for button_id in ("restart-btn", "update-btn"):
            try:
                self.query_one(f"#{button_id}", Button).disabled = not enabled
            except Exception:
                pass

    def _reset_worker_telemetry(self) -> None:
        with self._log_lock:
            self._worker_event_buf.clear()
        self._worker_last_state.clear()
        self._worker_last_activity.clear()
        self._worker_last_error.clear()
        for worker_name, _worker_label in WORKER_STATUS_ORDER:
            self._worker_logs[worker_name].clear()
            self._worker_state_cache[worker_name] = {}
            self._render_worker_panel(worker_name)

    @work(thread=True)
    def _restart_services(self) -> None:
        if self._service_op_in_progress:
            self.call_from_thread(
                self.notify, "A service operation is already running.", severity="warning"
            )
            return
        self._service_op_in_progress = True
        self.call_from_thread(self._set_action_buttons_enabled, False)
        self.call_from_thread(
            self.notify, "Restarting backend/frontend services...", timeout=2
        )
        self._log_activity("[yellow]Restart requested[/]")
        self._enqueue_log(">>> Restarting services...", source="SYSTEM", level="INFO")
        self._frontend_starting = False
        self._kill_children()
        kill_port(BACKEND_PORT)
        kill_port(FRONTEND_PORT)
        self._start_services()
        self._service_op_in_progress = False
        self.call_from_thread(self._set_action_buttons_enabled, True)

    @work(thread=True)
    def _update_and_restart(self) -> None:
        if self._service_op_in_progress:
            self.call_from_thread(
                self.notify, "A service operation is already running.", severity="warning"
            )
            return
        self._service_op_in_progress = True
        self.call_from_thread(self._set_action_buttons_enabled, False)
        self.call_from_thread(self.notify, "Updating project...", timeout=2)
        self._log_activity("[yellow]Update requested[/]")
        self._enqueue_log(">>> Running git pull --ff-only...", source="SYSTEM", level="INFO")

        if shutil.which("git") is None:
            self._enqueue_log("ERROR: git is not available on PATH.", source="SYSTEM", level="ERROR")
            self.call_from_thread(
                self.notify, "git is not available on PATH.", severity="error", timeout=4
            )
            self._service_op_in_progress = False
            self.call_from_thread(self._set_action_buttons_enabled, True)
            return

        try:
            pull_result = subprocess.run(
                ["git", "pull", "--ff-only"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=300,
            )
            for line in (pull_result.stdout or "").splitlines():
                self._enqueue_log(line, source="SYSTEM", level="INFO")
            for line in (pull_result.stderr or "").splitlines():
                self._enqueue_log(line, source="SYSTEM", level="WARNING")
            if pull_result.returncode != 0:
                self._enqueue_log(
                    "ERROR: git pull failed; aborting update/restart.",
                    source="SYSTEM",
                    level="ERROR",
                )
                self.call_from_thread(
                    self.notify, "Update failed: git pull error.", severity="error", timeout=4
                )
                self._service_op_in_progress = False
                self.call_from_thread(self._set_action_buttons_enabled, True)
                return
        except Exception as exc:
            self._enqueue_log(
                f"ERROR: git pull failed with exception: {exc}",
                source="SYSTEM",
                level="ERROR",
            )
            self.call_from_thread(
                self.notify, "Update failed while running git pull.", severity="error", timeout=4
            )
            self._service_op_in_progress = False
            self.call_from_thread(self._set_action_buttons_enabled, True)
            return

        setup_cmd: list[str]
        if sys.platform == "win32":
            setup_cmd = [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(PROJECT_ROOT / "scripts" / "infra" / "setup.ps1"),
            ]
        else:
            setup_cmd = ["bash", str(PROJECT_ROOT / "scripts" / "infra" / "setup.sh")]

        self._enqueue_log(">>> Running setup script...", source="SYSTEM", level="INFO")
        try:
            setup_proc = subprocess.Popen(
                setup_cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert setup_proc.stdout is not None
            for line in setup_proc.stdout:
                msg = line.rstrip()
                if msg:
                    self._enqueue_log(msg, source="SYSTEM", level="INFO")
            code = setup_proc.wait(timeout=900)
            if code != 0:
                self._enqueue_log(
                    "ERROR: setup failed; services were not restarted.",
                    source="SYSTEM",
                    level="ERROR",
                )
                self.call_from_thread(
                    self.notify, "Update failed during setup.", severity="error", timeout=4
                )
                self._service_op_in_progress = False
                self.call_from_thread(self._set_action_buttons_enabled, True)
                return
        except Exception as exc:
            self._enqueue_log(
                f"ERROR: setup failed with exception: {exc}",
                source="SYSTEM",
                level="ERROR",
            )
            self.call_from_thread(
                self.notify, "Update failed while running setup.", severity="error", timeout=4
            )
            self._service_op_in_progress = False
            self.call_from_thread(self._set_action_buttons_enabled, True)
            return

        self._frontend_starting = False
        self._kill_children()
        kill_port(BACKEND_PORT)
        kill_port(FRONTEND_PORT)
        self._start_services()
        self._enqueue_log(
            ">>> Update complete. Services restarting...",
            source="SYSTEM",
            level="INFO",
        )
        self.call_from_thread(
            self.notify, "Update complete. Restarting services...", timeout=3
        )
        self._service_op_in_progress = False
        self.call_from_thread(self._set_action_buttons_enabled, True)

    def _frontend_alive(self) -> bool:
        return self.frontend_proc is not None and self.frontend_proc.poll() is None

    def _venv_python_path(self) -> Path:
        if sys.platform == "win32":
            return BACKEND_DIR / "venv" / "Scripts" / "python.exe"
        return BACKEND_DIR / "venv" / "bin" / "python"

    def _build_runtime_env(self, *, process_role: str) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if sys.platform == "win32":
            venv_bin = str(BACKEND_DIR / "venv" / "Scripts")
        else:
            venv_bin = str(BACKEND_DIR / "venv" / "bin")
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
        env["VIRTUAL_ENV"] = str(BACKEND_DIR / "venv")
        env["HOMERUN_PROCESS_ROLE"] = process_role
        env.setdefault("LOG_LEVEL", "INFO")
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")
        env.setdefault("NEWS_FAISS_THREADS", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("EMBEDDING_DEVICE", "cpu")
        return env

    def _runtime_database_url_path(self) -> Path:
        return BACKEND_DIR / ".runtime" / "database_url"

    def _resolve_runtime_database_url(self) -> str:
        raw_value = str(os.environ.get("DATABASE_URL", "")).lstrip("\ufeff").strip().strip('"').strip("'")
        if raw_value:
            return raw_value.rstrip("/")
        try:
            raw_value = self._runtime_database_url_path().read_text(encoding="utf-8")
        except OSError:
            raw_value = ""
        raw_value = str(raw_value).lstrip("\ufeff").strip().strip('"').strip("'")
        if raw_value:
            return raw_value.rstrip("/")
        return "postgresql+asyncpg://homerun:homerun@127.0.0.1:5432/homerun"

    def _write_runtime_database_url(self, database_url: str) -> None:
        runtime_path = self._runtime_database_url_path()
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path.write_text(database_url.rstrip("/") + "\n", encoding="utf-8")

    def _database_target_from_url(self, database_url: str) -> dict[str, str | int]:
        parsed = urlsplit(database_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 5432
        database_name = parsed.path.lstrip("/") or "homerun"
        username = parsed.username or "homerun"
        password = parsed.password or "homerun"
        return {
            "host": host,
            "port": port,
            "database": database_name,
            "username": username,
            "password": password,
        }

    def _is_local_database_target(self, database_url: str) -> bool:
        host = str(self._database_target_from_url(database_url)["host"]).lower()
        return host in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}

    def _probe_database_ready(
        self,
        venv_python: Path,
        database_url: str,
        *,
        retries: int,
        retry_delay_seconds: float,
    ) -> tuple[bool, str]:
        probe_env = self._build_runtime_env(process_role="api")
        probe_env["DATABASE_URL"] = database_url
        result = subprocess.run(
            [
                str(venv_python),
                str(PROJECT_ROOT / "scripts" / "infra" / "ensure_postgres_ready.py"),
                "--database-url",
                database_url,
                "--retries",
                str(retries),
                "--retry-delay-seconds",
                str(retry_delay_seconds),
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            env=probe_env,
            timeout=max(15, int(retries * max(retry_delay_seconds, 0.25) * 4)),
        )
        output = "\n".join(
            part.strip()
            for part in ((result.stdout or "").strip(), (result.stderr or "").strip())
            if part and part.strip()
        ).strip()
        return result.returncode == 0, output

    def _start_local_postgres_runtime(self, database_url: str) -> tuple[bool, str]:
        target = self._database_target_from_url(database_url)
        docker_bin = shutil.which("docker") or "docker"
        compose_file = PROJECT_ROOT / "scripts" / "infra" / "docker-compose.infra.yml"
        runtime_env = os.environ.copy()
        runtime_env["POSTGRES_HOST"] = str(target["host"])
        runtime_env["POSTGRES_PORT"] = str(target["port"])
        runtime_env["POSTGRES_DB"] = str(target["database"])
        runtime_env["POSTGRES_USER"] = str(target["username"])
        runtime_env["POSTGRES_PASSWORD"] = str(target["password"])
        runtime_env.setdefault("POSTGRES_CONTAINER_NAME", "homerun-postgres")
        runtime_env.setdefault("POSTGRES_IMAGE", "postgres:16-alpine")

        commands: list[list[str]] = [
            [docker_bin, "compose", "-f", str(compose_file), "up", "-d", "postgres"],
            [docker_bin, "start", runtime_env["POSTGRES_CONTAINER_NAME"]],
            [
                docker_bin,
                "run",
                "--name",
                runtime_env["POSTGRES_CONTAINER_NAME"],
                "--detach",
                "--publish",
                f"{runtime_env['POSTGRES_HOST']}:{runtime_env['POSTGRES_PORT']}:5432",
                "--env",
                f"POSTGRES_DB={runtime_env['POSTGRES_DB']}",
                "--env",
                f"POSTGRES_USER={runtime_env['POSTGRES_USER']}",
                "--env",
                f"POSTGRES_PASSWORD={runtime_env['POSTGRES_PASSWORD']}",
                "--volume",
                "homerun-postgres-data:/var/lib/postgresql/data",
                runtime_env["POSTGRES_IMAGE"],
                "postgres",
                "-c",
                "max_connections=200",
            ],
        ]

        outputs: list[str] = []
        for command in commands:
            try:
                result = subprocess.run(
                    command,
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    env=runtime_env,
                    timeout=120,
                )
            except Exception as exc:
                outputs.append(f"{' '.join(command[:3])}: {exc}")
                continue
            combined = "\n".join(
                part.strip()
                for part in ((result.stdout or "").strip(), (result.stderr or "").strip())
                if part and part.strip()
            ).strip()
            if result.returncode == 0:
                return True, combined
            outputs.append(combined or f"{' '.join(command)} exited with code {result.returncode}")
        return False, "\n".join(entry for entry in outputs if entry)

    def _ensure_database_ready(self, venv_python: Path) -> tuple[bool, str]:
        database_url = self._resolve_runtime_database_url()
        self._write_runtime_database_url(database_url)

        ready, output = self._probe_database_ready(
            venv_python,
            database_url,
            retries=2,
            retry_delay_seconds=0.25,
        )
        if ready:
            return True, database_url

        if not self._is_local_database_target(database_url):
            if output:
                self._enqueue_log(output, source="SYSTEM", level="ERROR")
            return False, database_url

        self._enqueue_log(
            ">>> Starting Postgres runtime...",
            source="SYSTEM",
            level="INFO",
        )
        started, start_output = self._start_local_postgres_runtime(database_url)
        if start_output:
            self._enqueue_log(start_output, source="SYSTEM", level="INFO" if started else "ERROR")
        if not started:
            return False, database_url

        ready, output = self._probe_database_ready(
            venv_python,
            database_url,
            retries=240,
            retry_delay_seconds=0.5,
        )
        if output:
            self._enqueue_log(output, source="SYSTEM", level="INFO" if ready else "ERROR")
        return ready, database_url

    def _spawn_worker_plane(self, plane_name: str, *, reason: str) -> subprocess.Popen:
        source_tag = _WORKER_SOURCE_TAG_BY_PLANE[plane_name]
        worker_proc = subprocess.Popen(
            [
                str(self._venv_python_path()),
                "-m",
                "workers.host",
                plane_name,
            ],
            cwd=str(BACKEND_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=self._build_runtime_env(process_role="worker"),
        )
        _assign_to_job(worker_proc)
        self.worker_procs[plane_name] = worker_proc
        self._enqueue_log(
            f">>> Starting {plane_name} worker plane ({reason})...",
            source=source_tag,
            level="INFO",
        )
        self._start_stream_thread(worker_proc, source_tag)
        return worker_proc

    def _ensure_worker_planes_alive(self) -> None:
        if self.backend_proc is None or self.backend_proc.poll() is not None:
            return
        for _source_tag, plane_name in _WORKER_PLANES:
            proc = self.worker_procs.get(plane_name)
            if proc is not None and proc.poll() is None:
                continue
            if proc is not None:
                self._enqueue_log(
                    f"Worker plane {plane_name} exited with code {proc.poll()}, restarting.",
                    source=_WORKER_SOURCE_TAG_BY_PLANE[plane_name],
                    level="WARNING",
                )
            try:
                self._spawn_worker_plane(plane_name, reason="health-supervisor")
            except Exception as exc:
                self._enqueue_log(
                    f"FATAL: Failed to restart {plane_name} worker plane: {exc}",
                    source=_WORKER_SOURCE_TAG_BY_PLANE[plane_name],
                    level="ERROR",
                )

    def _ensure_worker_planes_alive_threadsafe(self) -> None:
        with self._worker_supervisor_lock:
            if self._worker_supervisor_inflight:
                return
            self._worker_supervisor_inflight = True
        try:
            self._ensure_worker_planes_alive()
        except Exception as exc:
            self._enqueue_log(
                f"Worker plane supervision failed: {exc}",
                source="SYSTEM",
                level="ERROR",
            )
        finally:
            with self._worker_supervisor_lock:
                self._worker_supervisor_inflight = False

    def _apply_health_safe(self, data: dict) -> None:
        try:
            self._apply_health(data)
        except Exception as exc:
            self._enqueue_log(
                f"TUI health apply failed: {exc}",
                source="SYSTEM",
                level="ERROR",
            )

    def _request_frontend_start(self, reason: str) -> None:
        if self._frontend_starting or self._frontend_alive():
            return
        self._frontend_starting = True
        self._enqueue_log(
            f">>> Triggering frontend start ({reason})...",
            source="FRONTEND",
            level="INFO",
        )
        self._start_frontend()

    # ---- Start backend & frontend as subprocesses ----
    @work(thread=True)
    def _start_services(self) -> None:
        """Start backend and frontend subprocesses."""
        self._log_activity("[bold cyan]Starting services...[/]")
        self.call_from_thread(self._reset_worker_telemetry)

        # Clean up detached worker processes from older launch modes.
        kill_legacy_worker_processes()

        # Kill stale processes
        kill_port(BACKEND_PORT)
        kill_port(FRONTEND_PORT)

        # Activate venv and start backend
        venv_python = self._venv_python_path()
        if not venv_python.exists():
            setup_cmd = ".\\scripts\\infra\\setup.ps1" if sys.platform == "win32" else "./scripts/infra/setup.sh"
            self._enqueue_log(
                f"ERROR: Virtual environment not found. Run {setup_cmd} first.",
                source="BACKEND",
                level="ERROR",
            )
            return

        self._enqueue_log(
            ">>> Ensuring database runtime is ready...",
            source="SYSTEM",
            level="INFO",
        )
        database_ready, database_url = self._ensure_database_ready(venv_python)
        if not database_ready:
            self._enqueue_log(
                f"FATAL: Database is not reachable at {database_url}.",
                source="BACKEND",
                level="ERROR",
            )
            return

        env = self._build_runtime_env(process_role="api")
        env["DATABASE_URL"] = database_url

        self._enqueue_log(
            ">>> Starting backend (uvicorn)...", source="BACKEND", level="INFO"
        )
        self._log_activity("[cyan]Backend starting...[/]")
        try:
            self.backend_proc = subprocess.Popen(
                [
                    str(venv_python),
                    "-m",
                    "uvicorn",
                    "main:app",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(BACKEND_PORT),
                    "--log-level",
                    "info",
                ],
                cwd=str(BACKEND_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
            )
            _assign_to_job(self.backend_proc)
        except Exception as e:
            self._enqueue_log(
                f"FATAL: Failed to start backend: {e}",
                source="BACKEND",
                level="ERROR",
            )
            return

        self._start_stream_thread(self.backend_proc, "BACKEND")

        self.worker_procs = {}

    def _read_network_access_setting(self) -> bool:
        """Read allow_network_access from backend settings API."""
        import urllib.request
        import urllib.error

        url = f"http://127.0.0.1:{BACKEND_PORT}/api/settings"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                # AllSettings wraps inside {"data": ...} when using unwrapApiData
                payload = data.get("data", data)
                network = payload.get("network", {})
                return bool(network.get("allow_network_access", False))
        except Exception:
            return False

    @work(thread=True)
    def _start_frontend(self) -> None:
        """Start frontend after backend is healthy."""
        env = os.environ.copy()
        env["BROWSER"] = "none"  # Don't auto-open browser
        env["FORCE_COLOR"] = "0"

        if self._read_network_access_setting():
            env["VITE_HOST"] = "0.0.0.0"
            self._enqueue_log(
                "Network access enabled — binding frontend to 0.0.0.0",
                source="FRONTEND",
                level="INFO",
            )

        self._enqueue_log(
            ">>> Starting frontend (npm run dev)...", source="FRONTEND", level="INFO"
        )
        self._log_activity("[cyan]Frontend starting...[/]")
        try:
            # On Windows, npm is a .cmd script; resolve it via shutil.which so
            # we can avoid shell=True (which wraps in cmd.exe and breaks pipe
            # handling and process-tree cleanup).
            npm_bin = shutil.which("npm") or "npm"
            self.frontend_proc = subprocess.Popen(
                [npm_bin, "run", "dev"],
                cwd=str(PROJECT_ROOT / "frontend"),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
            )
            _assign_to_job(self.frontend_proc)
        except Exception as e:
            self._enqueue_log(
                f"FATAL: Failed to start frontend: {e}",
                source="FRONTEND",
                level="ERROR",
            )
            self._frontend_starting = False
            return

        self._start_stream_thread(self.frontend_proc, "FRONTEND")

    def _stream_output(self, proc: subprocess.Popen, tag: str) -> None:
        """Read process stdout line-by-line and enqueue for batched display."""
        try:
            stdout = proc.stdout
            if stdout is None:
                return
            for raw_line in iter(stdout.readline, b""):
                if self._shutting_down:
                    break
                if proc.poll() is not None and not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue

                formatted, level = format_log_line(line, tag)
                self._enqueue_log(formatted, source=tag, level=level)

                # If backend started, kick off frontend (once only)
                if tag == "BACKEND":
                    if "Application startup complete" in line or "Uvicorn running" in line:
                        self._log_activity("[bold green]Backend is ready![/]")
                        self._request_frontend_start("backend startup log")
        except Exception:
            pass
        finally:
            if tag == "FRONTEND":
                self._frontend_starting = False
            if not self._shutting_down:
                self._enqueue_log(f"[{tag}] Process exited", source=tag, level="INFO")

    def _start_stream_thread(self, proc: subprocess.Popen, tag: str) -> None:
        thread = threading.Thread(
            target=self._stream_output,
            args=(proc, tag),
            name=f"{tag.lower()}-log-stream",
            daemon=True,
        )
        thread.start()

    # ---- Activity hooks (reserved for future system feed) ----
    def _log_activity(self, text: str) -> None:
        """Thread-safe activity hook."""
        self.call_from_thread(self._do_log_activity, text)

    def _do_log_activity(self, text: str) -> None:
        _ = text

    # ---- Periodic health polling ----
    def _poll_health(self) -> None:
        """Set up a periodic timer to poll /health/tui."""
        self.set_interval(HEALTH_POLL_INTERVAL_SECONDS, self._fetch_health)

    @work(thread=True)
    def _fetch_health(self) -> None:
        """Fetch health data from backend API."""
        import urllib.request
        import urllib.error

        with self._health_poll_lock:
            if self._health_poll_inflight:
                return
            self._health_poll_inflight = True

        try:
            req = urllib.request.Request(HEALTH_URL, method="GET")
            with urllib.request.urlopen(req, timeout=HEALTH_REQUEST_TIMEOUT_SECONDS) as resp:
                data = json.loads(resp.read().decode())
                self._ensure_worker_planes_alive_threadsafe()
                self.call_from_thread(self._apply_health_safe, data)
        except (urllib.error.URLError, Exception):
            self.call_from_thread(self._apply_health_offline)
        finally:
            with self._health_poll_lock:
                self._health_poll_inflight = False

    def _apply_health(self, data: dict) -> None:
        """Update all dashboard widgets from health data."""
        self.backend_healthy = True
        self.health_data = data
        self.health_poll_count += 1
        self._health_consecutive_failures = 0
        self._health_last_success_monotonic = time.monotonic()
        services = data.get("services", {})
        workers = data.get("workers")
        if workers is None:
            workers = services.get("workers", {})

        # --- Platform status ---
        ws = services.get("ws_feeds", {})
        database_healthy = self._resolve_database_health(data, services)

        self._update_platform_item("svc-backend", "BACKEND", True)
        self._update_platform_item("svc-database", "DATABASE", database_healthy)
        if not self._frontend_alive():
            self._request_frontend_start("backend health ready")
        frontend_alive = self._frontend_alive()
        self._update_platform_item("svc-frontend", "FRONTEND", frontend_alive)
        ws_started = bool(ws.get("started", False)) if isinstance(ws, dict) else False
        ws_available = (
            bool(ws.get("websockets_available", True)) if isinstance(ws, dict) else False
        )
        ws_healthy = bool(ws.get("healthy", False)) or (ws_started and ws_available)
        self._update_platform_item("svc-wsfeeds", "WS FEEDS", ws_healthy)

        # --- Worker command center ---
        self._update_worker_panels(workers, services)

        # --- Runtime metrics ---
        self._update_runtime_metrics()

    def _apply_health_offline(self) -> None:
        """Mark backend as offline after bounded health poll failure debounce."""
        self._health_consecutive_failures += 1
        if self._health_last_success_monotonic > 0:
            if self._health_consecutive_failures < HEALTH_FAILURES_BEFORE_OFFLINE:
                self._update_runtime_metrics()
                return
            age_since_success = time.monotonic() - self._health_last_success_monotonic
            if age_since_success < HEALTH_OFFLINE_GRACE_SECONDS:
                self._update_runtime_metrics()
                return
        self.backend_healthy = False
        self._update_platform_item("svc-backend", "BACKEND", False)
        self._update_platform_item("svc-database", "DATABASE", False)
        # Frontend and WS feeds depend on backend, mark offline
        frontend_alive = self.frontend_proc is not None and self.frontend_proc.poll() is None
        self._update_platform_item("svc-frontend", "FRONTEND", frontend_alive)
        self._update_platform_item("svc-wsfeeds", "WS FEEDS", False)
        # Workers are separate processes; check them individually instead of
        # blanket-marking everything offline when only the backend is down.
        self._update_workers_from_processes()
        self._update_runtime_metrics()

    def _update_platform_item(self, widget_id: str, label: str, is_on: bool) -> None:
        state = "ONLINE" if is_on else "OFFLINE"
        dot = "[green]\u25cf[/]" if is_on else "[red]\u25cf[/]"
        try:
            self.query_one(f"#{widget_id}", Static).update(f"{dot} {label:<8} {state}")
        except Exception:
            pass

    def _resolve_database_health(self, data: dict, services: dict) -> bool:
        checks = data.get("checks")
        if isinstance(checks, dict) and "database" in checks:
            return bool(checks.get("database"))
        db_status = services.get("database")
        if isinstance(db_status, bool):
            return db_status
        if isinstance(db_status, dict):
            for key in ("healthy", "connected", "ok", "running"):
                if key in db_status:
                    return bool(db_status.get(key))
        return True

    def _normalize_workers_payload(self, workers: object, services: dict) -> dict[str, dict]:
        by_name: dict[str, dict] = {}
        if isinstance(workers, list):
            by_name = {
                str(item.get("worker_name")): item
                for item in workers
                if isinstance(item, dict) and item.get("worker_name")
            }
        elif isinstance(workers, dict):
            by_name = {
                str(name): item
                for name, item in workers.items()
                if isinstance(item, dict)
            }
        for worker_name, _ in WORKER_STATUS_ORDER:
            if worker_name not in by_name:
                by_name[worker_name] = self._fallback_worker_snapshot(worker_name, services)
        return by_name

    def _fallback_worker_snapshot(self, worker_name: str, services: dict) -> dict:
        """Build a minimal worker snapshot from the services section of /health.

        Workers run in-process inside the backend, so there are no separate
        OS processes to check.  We infer running state from the services
        sub-keys that the health endpoint already exposes.
        """
        running = False
        if worker_name == "scanner":
            running = bool(services.get("scanner", {}).get("running", False))
        elif worker_name == "discovery":
            running = bool(services.get("wallet_discovery", {}).get("running", False))
        elif worker_name == "news":
            running = bool(services.get("news_workflow", {}).get("running", False))
        elif worker_name == "trader_orchestrator":
            running = bool(services.get("trader_orchestrator", {}).get("running", False))
        elif worker_name == "trader_reconciliation":
            running = bool(services.get("trader_reconciliation", {}).get("running", False))
        else:
            # crypto, weather, tracked_traders, events — no dedicated
            # services key; they report via worker_snapshots which the main
            # workers dict already covers.  If we got here, health didn't include
            # a snapshot yet (worker hasn't completed its first cycle).
            running = False
        return {
            "running": running,
            "enabled": True,
            "interval_seconds": None,
            "last_run_at": None,
            "lag_seconds": None,
            "current_activity": None,
            "last_error": None,
        }

    def _update_worker_panels(self, workers: object, services: dict) -> None:
        by_name = self._normalize_workers_payload(workers, services)
        for worker_name, _worker_label in WORKER_STATUS_ORDER:
            snapshot = by_name.get(worker_name, {})
            if not isinstance(snapshot, dict):
                snapshot = {}
            self._worker_state_cache[worker_name] = snapshot
            self._emit_worker_snapshot_events(worker_name, snapshot)
            self._render_worker_panel(worker_name)

    def _emit_worker_snapshot_events(self, worker_name: str, snapshot: dict) -> None:
        state, _status_class = self._resolve_worker_state(snapshot)
        prev_state = self._worker_last_state.get(worker_name)
        if prev_state is not None and prev_state != state:
            self._append_worker_log(worker_name, f"state changed to {state}", update=False)
        self._worker_last_state[worker_name] = state

        activity = snapshot.get("current_activity")
        if isinstance(activity, str):
            activity = activity.strip()
        else:
            activity = ""
        if activity and activity != self._worker_last_activity.get(worker_name):
            self._worker_last_activity[worker_name] = activity
            self._append_worker_log(worker_name, activity, update=False)

        last_error = snapshot.get("last_error")
        if isinstance(last_error, str):
            last_error = last_error.strip()
        else:
            last_error = ""
        if last_error and last_error != self._worker_last_error.get(worker_name):
            self._worker_last_error[worker_name] = last_error
            self._append_worker_log(worker_name, f"ERROR: {last_error}", update=False)

    def _render_worker_panel(self, worker_name: str) -> None:
        snapshot = self._worker_state_cache.get(worker_name, {})
        status, status_class = self._resolve_worker_state(snapshot)
        try:
            panel = self.query_one(f"#worker-{worker_name}", WorkerPanel)
            content_width = self._get_worker_content_width(worker_name, panel)
            meta = self._truncate_worker_text(self._format_worker_meta(snapshot), content_width)
            lines = [
                self._normalize_worker_log_line(line, content_width)
                for line in self._worker_logs.get(worker_name, [])
            ]
            lines = [line for line in lines if line]
            panel.update_state(status, status_class, meta)
            panel.update_logs(lines)
        except Exception:
            pass

    def _set_workers_offline(self) -> None:
        for worker_name, _worker_label in WORKER_STATUS_ORDER:
            self._worker_state_cache[worker_name] = {}
            self._render_worker_panel(worker_name)

    def _update_workers_from_processes(self) -> None:
        """Fallback worker state when the backend health endpoint is unreachable."""
        for worker_name, _worker_label in WORKER_STATUS_ORDER:
            plane_name = _WORKER_PLANE_BY_NAME.get(worker_name)
            proc = self.worker_procs.get(plane_name or "")
            running = proc is not None and proc.poll() is None
            self._worker_state_cache[worker_name] = {
                "running": running,
                "enabled": running,
                "interval_seconds": None,
                "last_run_at": None,
                "lag_seconds": None,
                "current_activity": "Process running" if running else None,
                "last_error": None,
            }
            self._render_worker_panel(worker_name)

    def _resolve_worker_state(self, snapshot: dict) -> tuple[str, str]:
        if not snapshot:
            return ("OFFLINE", "status-off")
        activity = str(snapshot.get("current_activity") or "").strip().lower()
        active_hint = (
            bool(activity)
            and not activity.startswith("idle")
            and not activity.startswith("paused")
            and not activity.startswith("disabled")
            and not activity.startswith("stopped")
            and not activity.startswith("waiting")
            and not activity.startswith("blocked")
        )
        if bool(snapshot.get("running", False)) or (bool(snapshot.get("enabled", True)) and active_hint):
            return ("RUNNING", "status-on")
        enabled = snapshot.get("enabled")
        if enabled is False:
            return ("PAUSED", "status-warn")
        return ("IDLE", "status-idle")

    def _format_worker_meta(self, snapshot: dict) -> str:
        if not snapshot:
            return "No telemetry yet"
        parts: list[str] = []
        stats = snapshot.get("stats")

        if isinstance(stats, dict):
            memory_mb: Optional[float] = None
            memory_raw = stats.get("memory_mb")
            if isinstance(memory_raw, (int, float)) and float(memory_raw) > 0:
                memory_mb = float(memory_raw)
            else:
                rss_bytes = stats.get("rss_bytes")
                if isinstance(rss_bytes, (int, float)) and float(rss_bytes) > 0:
                    memory_mb = float(rss_bytes) / (1024 * 1024)

            if memory_mb is not None:
                parts.append(f"mem {memory_mb:.1f} MB")

            pid = stats.get("pid")
            if isinstance(pid, int) and pid > 0:
                parts.append(f"pid {pid}")
            elif isinstance(pid, str) and pid.isdigit():
                parts.append(f"pid {pid}")

        interval_seconds = snapshot.get("interval_seconds")
        if isinstance(interval_seconds, (int, float)) and interval_seconds > 0:
            interval_int = int(interval_seconds)
            if interval_int >= 60 and interval_int % 60 == 0:
                parts.append(f"every {interval_int // 60}m")
            else:
                parts.append(f"every {interval_int}s")

        lag_seconds = snapshot.get("lag_seconds")
        if isinstance(lag_seconds, (int, float)):
            parts.append(f"lag {float(lag_seconds):.1f}s")

        last_run = snapshot.get("last_run_at")
        if isinstance(last_run, str) and last_run:
            relative = self._format_relative_age(last_run)
            if relative:
                parts.append(f"last {relative}")

        if not parts:
            return "No telemetry yet"
        return " | ".join(parts)

    def _format_relative_age(self, iso_text: str) -> Optional[str]:
        try:
            normalized = iso_text.strip()
            if normalized.endswith("Z"):
                normalized = normalized[:-1] + "+00:00"
            ts = datetime.fromisoformat(normalized)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = ts.astimezone(timezone.utc)
            delta = int((datetime.now(timezone.utc) - ts).total_seconds())
            if delta < 0:
                delta = 0
            if delta < 60:
                return f"{delta}s ago"
            if delta < 3600:
                return f"{delta // 60}m ago"
            if delta < 86400:
                return f"{delta // 3600}h ago"
            return f"{delta // 86400}d ago"
        except Exception:
            return None

    # ---- Runtime metrics ----
    def _update_runtime_metrics(self) -> None:
        """Update the runtime metrics bar on the home page."""
        elapsed = time.time() - self.start_time
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)

        # Count running workers
        running = 0
        total = len(WORKER_STATUS_ORDER)
        for worker_name, _ in WORKER_STATUS_ORDER:
            snapshot = self._worker_state_cache.get(worker_name, {})
            if bool(snapshot.get("running", False)):
                running += 1

        health_text = "[bold green]OK[/]" if self.backend_healthy else "[bold red]OFFLINE[/]"

        try:
            self.query_one("#metric-uptime", Static).update(
                f"Uptime [bold]{h:02d}:{m:02d}:{s:02d}[/]"
            )
        except Exception:
            pass
        try:
            self.query_one("#metric-workers", Static).update(
                f"Workers [bold]{running}/{total}[/]"
            )
        except Exception:
            pass
        try:
            self.query_one("#metric-health", Static).update(
                f"Health {health_text}"
            )
        except Exception:
            pass
        try:
            self.query_one("#metric-polls", Static).update(
                f"Polls [bold]{self.health_poll_count}[/]"
            )
        except Exception:
            pass
        try:
            self.query_one("#metric-logs", Static).update(
                f"Logs [bold]{len(self._log_entries):,}[/]"
            )
        except Exception:
            pass

    # ---- Uptime ticker ----
    def _update_uptime(self) -> None:
        self.set_interval(1.0, self._tick_uptime)

    def _tick_uptime(self) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            self.query_one("#uptime-bar", Static).update(
                f"{ts}  |  [bold]h[/]=Home  [bold]l[/]=Logs  [bold]d[/]=Theme  "
                f"[bold]/[/]=Search  [bold]r[/]=Restart  [bold]?[/]=Help  [bold]q[/]=Quit"
            )
        except Exception:
            pass
        self._update_runtime_metrics()

    # ---- Cleanup ----
    def on_unmount(self) -> None:
        self._shutting_down = True
        self._kill_children()

    async def action_quit(self) -> None:
        self._shutting_down = True
        self.notify("Shutting down...", severity="warning", timeout=10)
        self._kill_children()
        self.exit()

    def _kill_children(self) -> None:
        """Kill child processes and close their pipes to unblock reader threads.

        Uses _terminate_process_tree (Win32 TerminateProcess) which is
        instantaneous and doesn't spawn subprocesses.  Falls back to
        _kill_orphaned_workers() for anything that escaped the tree.
        """
        procs = [
            p
            for p in (
                self.backend_proc,
                self.frontend_proc,
                *self.worker_procs.values(),
            )
            if p and p.poll() is None
        ]
        for proc in procs:
            try:
                if sys.platform == "win32":
                    _terminate_process_tree(proc.pid)
                else:
                    proc.kill()
            except Exception:
                pass
        # Close stdout pipes to unblock readline() in reader threads
        for proc in procs:
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass
        # Brief wait for processes to actually die (SIGKILL is fast)
        for proc in procs:
            try:
                proc.wait(timeout=1)
            except Exception:
                pass
        self.worker_procs = {}
        # Sweep for orphaned worker processes that escaped the process tree kill.
        # The backend spawns workers via asyncio.create_subprocess_exec; if the
        # backend dies before taskkill /T runs, those children become orphans.
        self._kill_orphaned_workers()

    def _kill_orphaned_workers(self) -> None:
        """Kill orphaned Homerun Python worker processes by command-line pattern."""
        if sys.platform != "win32":
            return
        my_pid = os.getpid()
        backend_dir = str(BACKEND_DIR)
        project_root = str(PROJECT_ROOT)
        try:
            import ctypes
            import ctypes.wintypes

            # Use WMI via PowerShell for reliable command-line inspection.
            # This is the same approach as the launcher's Cleanup-StaleHomerunProcesses.
            result = subprocess.run(
                [
                    "powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" "
                    "| Select-Object ProcessId,CommandLine "
                    "| ConvertTo-Json -Compress",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return

            import json as _json
            data = _json.loads(result.stdout)
            if isinstance(data, dict):
                data = [data]

            for entry in data:
                pid = entry.get("ProcessId")
                cmd = entry.get("CommandLine") or ""
                if not pid or pid == my_pid:
                    continue

                is_homerun = False
                if "workers.host" in cmd:
                    is_homerun = True
                elif "workers.runner" in cmd or "workers." in cmd and "_worker" in cmd:
                    is_homerun = True
                elif "uvicorn" in cmd and "main:app" in cmd:
                    is_homerun = True

                if not is_homerun:
                    continue
                if backend_dir not in cmd and project_root not in cmd:
                    continue

                try:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        capture_output=True,
                        timeout=5,
                    )
                except Exception:
                    pass
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    # Verify venv exists
    venv_dir = BACKEND_DIR / "venv"
    if not venv_dir.exists():
        setup_cmd = ".\\scripts\\infra\\setup.ps1" if sys.platform == "win32" else "./scripts/infra/setup.sh"
        print(f"Setup not complete. Run {setup_cmd} first.")
        sys.exit(1)

    # On Windows, closing the terminal window sends SIGBREAK.  Kill all child
    # process trees explicitly (taskkill /F /T) so workers don't survive.
    if sys.platform == "win32":
        def _sigbreak_handler(signum, frame):
            _sigbreak_kill_children()
            os._exit(0)
        signal.signal(signal.SIGBREAK, _sigbreak_handler)

    _ensure_startup_terminal_size()
    _configure_windows_console_for_tui()

    app = HomerunApp()
    exit_code = 0

    try:
        app.run()
    except BaseException:
        exit_code = 1
        raise
    finally:
        _restore_windows_console_mode()
        # Kill any remaining child process trees, then close the Job Object handle
        # (triggers KILL_ON_JOB_CLOSE as a safety net).
        _sigbreak_kill_children()

        # Force-exit to avoid hanging on background thread joins.
        # Textual worker threads (subprocess readers) may still be blocked
        # on I/O; Python's atexit handler would wait for them indefinitely.
        os._exit(exit_code)


if __name__ == "__main__":
    main()
