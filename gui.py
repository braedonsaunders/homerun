#!/usr/bin/env python3
"""Homerun GUI – Desktop launcher for the Homerun trading platform.

Native tkinter desktop launcher.  Zero extra dependencies – tkinter ships
with every standard Python install.
"""
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
import traceback
import tkinter as tk
import webbrowser
import tkinter.font as tkfont
import tkinter.ttk as ttk
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

# ---------------------------------------------------------------------------
# Windows Job Object – ensures ALL child processes die when the GUI exits
# ---------------------------------------------------------------------------
_win_job_handle = None
_child_pids: set = set()
_single_instance_handle = None

if sys.platform == "win32":
    try:
        import ctypes.wintypes

        _kernel32 = ctypes.windll.kernel32
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
        _kernel32.CreateMutexW.restype = _HANDLE
        _kernel32.CreateMutexW.argtypes = [_LPVOID, _BOOL, ctypes.wintypes.LPCWSTR]
        _kernel32.GetLastError.restype = _DWORD

        _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        _JOB_OBJECT_EXTENDED_LIMIT_INFO_CLASS = 9
        _ERROR_ALREADY_EXISTS = 183

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
                ("IoInfo", ctypes.c_byte * 48),
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


def _acquire_single_instance_guard() -> bool:
    global _single_instance_handle
    if sys.platform != "win32":
        return True
    if _single_instance_handle:
        return True
    try:
        mutex = _kernel32.CreateMutexW(None, False, "Local\\HomerunDesktopLauncher")
        if not mutex:
            return True
        already_running = int(_kernel32.GetLastError()) == _ERROR_ALREADY_EXISTS
        if already_running:
            _kernel32.CloseHandle(mutex)
            return False
        _single_instance_handle = mutex
        return True
    except Exception:
        return True


def _notify_existing_instance() -> None:
    message = "Homerun is already running. Close the existing window before launching another copy."
    if sys.platform == "win32":
        try:
            ctypes.windll.user32.MessageBoxW(None, message, "Homerun", 0x00000040)
            return
        except Exception:
            pass
    print(message)


def _assign_to_job(proc: subprocess.Popen) -> None:
    _child_pids.add(proc.pid)
    if _win_job_handle is None or sys.platform != "win32":
        return
    try:
        handle = _kernel32.OpenProcess(0x1FFFFF, False, proc.pid)
        if handle:
            _kernel32.AssignProcessToJobObject(_win_job_handle, handle)
            _kernel32.CloseHandle(handle)
    except Exception:
        pass


def _terminate_process_tree(root_pid: int) -> None:
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
        INVALID = ctypes.wintypes.HANDLE(-1).value
        if snap is None or snap == INVALID:
            return

        children: dict[int, list[int]] = {}
        entry = PROCESSENTRY32()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32)
        if _kernel32.Process32First(snap, ctypes.byref(entry)):
            while True:
                children.setdefault(entry.th32ParentProcessID, []).append(entry.th32ProcessID)
                if not _kernel32.Process32Next(snap, ctypes.byref(entry)):
                    break
        _kernel32.CloseHandle(snap)

        to_kill: list[int] = []
        queue = [root_pid]
        while queue:
            pid = queue.pop(0)
            to_kill.append(pid)
            queue.extend(children.get(pid, []))

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
    for pid in _child_pids:
        _terminate_process_tree(pid)
    if _win_job_handle:
        try:
            _kernel32.CloseHandle(_win_job_handle)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BACKEND_PORT = 8000
FRONTEND_PORT = 3000
# Plane separation: trading workers run in one process, news/weather
# (ML-heavy: sentence-transformers, FAISS) in another, and wallet
# discovery / smart-pool work in a third.  Discovery is isolated
# because its Polymarket REST fan-out (5h soak: ``wallet_discovery
# ._worker x12``, ``smart_wallet_pool._scan_worker x8``, 90+ in-flight
# ``try_connect``) was producing 5-8 s event-loop stalls in the
# trading plane and starving the wallet-state reseeder.  See the
# ``discovery`` plane comment in ``backend/workers/host.py``.
_WORKER_PLANES = (
    ("WORKERS", "trading"),
    ("NEWS", "news"),
    ("DISCOVERY", "discovery"),
)
_WORKER_SOURCE_TAG_BY_PLANE = {pn: st for st, pn in _WORKER_PLANES}
_WORKER_PLANE_BY_NAME: dict[str, str] = {
    "scanner": "trading", "scanner_slo": "trading", "crypto": "trading",
    "trader_orchestrator": "trading",
    "events": "trading",
    "trader_reconciliation": "trading", "redeemer": "trading",
    "tracked_traders": "discovery", "discovery": "discovery",
    "weather": "news", "news": "news",
}

HEALTH_URL = f"http://127.0.0.1:{BACKEND_PORT}/health/gui"
PROJECT_ROOT = Path(__file__).parent.resolve()
BACKEND_DIR = PROJECT_ROOT / "backend"

LOG_MAX_LINES = 5000
LOG_TRIM_BATCH = 1000
LOG_FLUSH_MS = 500
HEALTH_POLL_INTERVAL_MS = 3000
HEALTH_REQUEST_TIMEOUT_SECONDS = 2.5
HEALTH_FAILURES_BEFORE_OFFLINE = 2
HEALTH_OFFLINE_GRACE_SECONDS = 10.0

LOG_LEVEL_ORDER = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
OSC_ESCAPE_RE = re.compile(r"\x1B\].*?(?:\x07|\x1B\\)")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

WORKER_STATUS_ORDER: list[tuple[str, str]] = [
    ("scanner", "SCANNER"), ("scanner_slo", "SCANNER SLO"),
    ("discovery", "DISCOVERY"), ("weather", "WEATHER"),
    ("news", "NEWS"), ("crypto", "CRYPTO"),
    ("tracked_traders", "TRACKED"), ("trader_orchestrator", "ORCHESTRATOR"),
    ("trader_reconciliation", "RECONCILIATION"), ("redeemer", "REDEEMER"),
    ("events", "EVENTS"),
]

WORKER_TAG_TO_NAME: dict[str, str] = {
    "SCANNER": "scanner", "SCANNER SLO": "scanner_slo",
    "DISCOVERY": "discovery", "WEATHER": "weather",
    "NEWS": "news", "CRYPTO": "crypto",
    "TRACKED": "tracked_traders", "ORCHESTRATOR": "trader_orchestrator",
    "RECONCILIATION": "trader_reconciliation", "REDEEMER": "redeemer",
    "EVENTS": "events",
}

WORKER_BACKEND_HINTS: tuple[tuple[str, str], ...] = (
    ("scanner", "scanner"), ("scanner_slo", "scanner_slo"),
    ("discovery", "discovery"), ("weather", "weather"),
    ("news_worker", "news"), ("crypto_worker", "crypto"),
    ("tracked_traders", "tracked_traders"),
    ("orchestrator", "trader_orchestrator"),
    ("reconciliation", "trader_reconciliation"),
    ("redeemer_worker", "redeemer"), ("events_worker", "events"),
)

SOURCE_FILTER_LABELS = {
    "all": "All Sources", "backend": "Backend", "workers": "Workers",
    "frontend": "Frontend", "system": "System",
}

LEVEL_FILTER_LABELS = {
    "all": "All Levels", "error": "Error+", "warning": "Warning+",
    "info": "Info+", "debug": "Debug+",
}

WORKER_FILTER_ORDER: tuple[str, ...] = (
    "all", "none", *[name for name, _ in WORKER_STATUS_ORDER],
)
WORKER_FILTER_LABELS: dict[str, str] = {
    "all": "All Workers", "none": "No Worker",
    **{name: label for name, label in WORKER_STATUS_ORDER},
}

# ---------------------------------------------------------------------------
# Colour palette (dark theme)
# ---------------------------------------------------------------------------
BG = "#08111a"
BG_PANEL = "#0e1a28"
BG_WORKER = "#0f1d2c"
BG_INPUT = "#112333"
FG = "#d8e5ef"
FG_DIM = "#6e94ab"
FG_MID = "#8cb3c8"
FG_ACCENT = "#58f1c1"
GREEN = "#55f0b8"
RED = "#ff6b6b"
YELLOW = "#ffbf69"
BLUE_DIM = "#7ebee6"
BORDER = "#28445f"
BORDER_LIGHT = "#2b4961"


# ---------------------------------------------------------------------------
# Windows subprocess launch options
# ---------------------------------------------------------------------------
def _windows_subprocess_kwargs() -> dict:
    if sys.platform != "win32":
        return {}

    kwargs: dict[str, object] = {}
    create_no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if create_no_window:
        kwargs["creationflags"] = create_no_window

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0
    kwargs["startupinfo"] = startupinfo
    return kwargs


# ---------------------------------------------------------------------------
# Port killing
# ---------------------------------------------------------------------------
def kill_port(port: int) -> None:
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True, text=True, timeout=5, **_windows_subprocess_kwargs(),
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
                            **_windows_subprocess_kwargs(),
                        )
                    except Exception:
                        pass
            time.sleep(0.5)
        except Exception:
            pass
    else:
        try:
            result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True, timeout=5)
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
        except Exception:
            try:
                subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True, timeout=5)
                time.sleep(0.5)
            except Exception:
                pass


def _list_windows_python_processes() -> list[dict]:
    if sys.platform != "win32":
        return []
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process "
             "| Where-Object { $_.Name -eq 'python.exe' -or $_.Name -eq 'pythonw.exe' } "
             "| Select-Object ProcessId,ParentProcessId,CommandLine "
             "| ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=10, **_windows_subprocess_kwargs(),
        )
    except Exception:
        return []
    if result.returncode != 0 or not result.stdout.strip():
        return []
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    return []


def _current_windows_python_lineage(process_entries: list[dict]) -> set[int]:
    lineage = {os.getpid()}
    parent_by_pid: dict[int, int] = {}
    for entry in process_entries:
        try:
            pid = int(entry.get("ProcessId") or 0)
            parent_pid = int(entry.get("ParentProcessId") or 0)
        except (TypeError, ValueError):
            continue
        if pid > 0:
            parent_by_pid[pid] = parent_pid
    cursor = parent_by_pid.get(os.getpid(), 0)
    while cursor and cursor not in lineage:
        lineage.add(cursor)
        cursor = parent_by_pid.get(cursor, 0)
    return lineage


def kill_legacy_worker_processes() -> None:
    if sys.platform == "win32":
        backend_dir = str(BACKEND_DIR)
        project_root = str(PROJECT_ROOT)
        gui_bootstrap_name = "_gui_bootstrap.py"
        try:
            data = _list_windows_python_processes()
            if not data:
                return
            protected_pids = _current_windows_python_lineage(data)
            for entry in data:
                pid = entry.get("ProcessId")
                cmd = str(entry.get("CommandLine") or "")
                try:
                    pid_int = int(pid or 0)
                except (TypeError, ValueError):
                    continue
                if not pid_int or pid_int in protected_pids:
                    continue
                is_homerun = False
                if "workers.host" in cmd:
                    is_homerun = True
                elif "workers.runner" in cmd or ("workers." in cmd and "_worker" in cmd):
                    is_homerun = True
                elif "uvicorn" in cmd and "main:app" in cmd:
                    is_homerun = True
                elif "gui.py" in cmd or gui_bootstrap_name in cmd:
                    is_homerun = True
                if not is_homerun:
                    continue
                if (
                    "workers.host" not in cmd
                    and "workers.runner" not in cmd
                    and "_worker" not in cmd
                    and "gui.py" not in cmd
                    and gui_bootstrap_name not in cmd
                    and backend_dir not in cmd
                    and project_root not in cmd
                ):
                    continue
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(pid_int)],
                        capture_output=True,
                        timeout=5,
                        **_windows_subprocess_kwargs(),
                    )
                except Exception:
                    pass
        except Exception:
            pass
        time.sleep(0.25)
    else:
        try:
            result = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=5)
        except Exception:
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
            except (ProcessLookupError, OSError):
                continue
        time.sleep(0.25)


# ---------------------------------------------------------------------------
# Plain-text log formatter
# ---------------------------------------------------------------------------
def format_log_line(line: str, tag: str) -> tuple[str, str]:
    line = OSC_ESCAPE_RE.sub("", line)
    line = CONTROL_CHAR_RE.sub("", ANSI_ESCAPE_RE.sub("", line.replace("\r", "")))
    prefix = f"[{tag}]"

    if line.startswith("{"):
        try:
            data = json.loads(line)
            level = data.get("level", "INFO").upper()
            msg = data.get("message", line)
            logger_name = data.get("logger", "")
            func = data.get("function", "")
            ts = data.get("timestamp", "")
            ts_short = ""
            if ts:
                try:
                    ts_short = ts.split("T")[1].split(".")[0] if "T" in ts else ts[-8:]
                except Exception:
                    ts_short = ts[:19]
            parts = [prefix]
            if ts_short:
                parts.append(ts_short)
            parts.append(f"[{level}]")
            if logger_name:
                parts.append(logger_name)
            if func:
                parts.append(f"{func}()")
            parts.append(msg)
            # Include extra_data key=value pairs (e.g. strategy, event_type)
            extra = data.get("data")
            if isinstance(extra, dict):
                for k, v in extra.items():
                    if v is not None:
                        parts.append(f"{k}={v}")
            # Append exception traceback (last line only) so errors are visible
            exc_text = data.get("exception", "")
            if exc_text:
                # Take the last non-empty line of the traceback (the actual error)
                exc_lines = [ln for ln in exc_text.strip().splitlines() if ln.strip()]
                if exc_lines:
                    parts.append(f"|| {exc_lines[-1].strip()}")
                    level = "ERROR"
            return (" ".join(parts), level)
        except (json.JSONDecodeError, KeyError):
            pass

    level = "INFO"
    line_upper = line.upper()
    if "ERROR" in line_upper or "TRACEBACK" in line_upper or "EXCEPTION" in line_upper:
        level = "ERROR"
    elif "WARNING" in line_upper or "WARN" in line_upper:
        level = "WARNING"
    elif "DEBUG" in line_upper:
        level = "DEBUG"

    return (f"{prefix} {line}", level)


# ---------------------------------------------------------------------------
# Main GUI Application
# ---------------------------------------------------------------------------
class HomerunApp:
    """Homerun Trading Platform – tkinter desktop launcher."""

    def __init__(self) -> None:
        # Process handles
        self.backend_proc: Optional[subprocess.Popen] = None
        self.worker_procs: dict[str, subprocess.Popen] = {}
        self.frontend_proc: Optional[subprocess.Popen] = None

        # State
        self.start_time = time.time()
        self.backend_healthy = False
        self.health_data: dict = {}
        self.health_poll_count = 0
        self._shutting_down = False
        self._frontend_starting = False
        self._service_op_in_progress = False

        # Thread-safe log buffer
        self._log_lock = threading.Lock()
        self._log_buf: collections.deque[tuple[str, str, str, str]] = collections.deque(maxlen=2000)
        self._log_entries: list[tuple[str, str, str, str]] = []
        self._log_line_count = 0

        # Filters
        self._source_filter = "all"
        self._level_filter = "all"
        self._worker_filter = "all"
        self._search_filter = ""

        # Worker telemetry
        self._worker_logs: dict[str, collections.deque[str]] = {
            name: collections.deque(maxlen=12) for name, _ in WORKER_STATUS_ORDER
        }
        self._worker_state_cache: dict[str, dict] = {}
        self._worker_last_state: dict[str, str] = {}
        self._worker_last_activity: dict[str, str] = {}
        self._worker_last_error: dict[str, str] = {}

        # Health poll guards
        self._health_poll_lock = threading.Lock()
        self._health_poll_inflight = False
        self._worker_supervisor_lock = threading.Lock()
        self._worker_supervisor_inflight = False
        self._health_last_success_monotonic = 0.0
        self._health_consecutive_failures = 0

        # Build the window
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.root = tk.Tk()
        self.root.report_callback_exception = self._report_tk_exception
        self.root.title("HOMERUN – Autonomous Prediction Market Trading Platform")
        self.root.configure(bg=BG)
        self.root.geometry("1200x820")
        self.root.minsize(800, 500)
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)

        # High-DPI awareness on Windows
        if sys.platform == "win32":
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                pass

        # Fonts
        self._font = tkfont.Font(family="Consolas" if sys.platform == "win32" else "Menlo", size=10)
        self._font_bold = tkfont.Font(family=self._font.cget("family"), size=10, weight="bold")
        self._font_small = tkfont.Font(family=self._font.cget("family"), size=9)
        self._font_title = tkfont.Font(family=self._font.cget("family"), size=12, weight="bold")
        self._font_logo = tkfont.Font(family=self._font.cget("family"), size=9, weight="bold")

        # Notebook (tabs)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG_PANEL, foreground=FG_DIM,
                         padding=[12, 4], font=self._font_bold)
        style.map("TNotebook.Tab",
                  background=[("selected", BG_WORKER)],
                  foreground=[("selected", FG)])

        self._notebook = ttk.Notebook(self.root)
        self._notebook.pack(fill=tk.BOTH, expand=True)

        # ---- Home tab ----
        home_frame = tk.Frame(self._notebook, bg=BG)
        self._notebook.add(home_frame, text="  Home  ")
        self._build_home_tab(home_frame)

        # ---- Logs tab ----
        logs_frame = tk.Frame(self._notebook, bg=BG)
        self._notebook.add(logs_frame, text="  Logs  ")
        self._build_logs_tab(logs_frame)

    def _build_home_tab(self, parent: tk.Frame) -> None:
        # Top row: logo + platform status
        top = tk.Frame(parent, bg=BG)
        top.pack(fill=tk.X, padx=10, pady=(10, 0))

        # Brand panel
        brand = tk.Frame(top, bg=BG_PANEL, highlightbackground=BORDER, highlightthickness=1)
        brand.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        logo_text = (
            "██   ██  ██████  ███    ███ ███████ ██████  ██    ██ ███    ██\n"
            "██   ██ ██    ██ ████  ████ ██      ██   ██ ██    ██ ████   ██\n"
            "███████ ██    ██ ██ ████ ██ █████   ██████  ██    ██ ██ ██  ██\n"
            "██   ██ ██    ██ ██  ██  ██ ██      ██   ██ ██    ██ ██  ██ ██\n"
            "██   ██  ██████  ██      ██ ███████ ██   ██  ██████  ██   ████"
        )
        tk.Label(brand, text=logo_text, font=self._font_logo, fg=GREEN, bg=BG_PANEL,
                 justify=tk.LEFT, anchor="w").pack(padx=10, pady=(8, 2), anchor="w")
        tk.Label(brand, text="Autonomous Prediction Market Trading Platform",
                 font=self._font_small, fg=FG_DIM, bg=BG_PANEL, anchor="w").pack(padx=10, anchor="w")

        # Action buttons
        btn_frame = tk.Frame(brand, bg=BG_PANEL)
        btn_frame.pack(padx=10, pady=(6, 8), anchor="w")
        self._open_btn = tk.Button(
            btn_frame, text="Open", font=self._font_bold, bg="#1a4a7a", fg=FG,
            activebackground="#2a6aaa", activeforeground=FG, relief=tk.FLAT,
            padx=12, pady=2, command=self._open_in_browser,
        )
        self._open_btn.pack(side=tk.LEFT, padx=(0, 6))
        self._restart_btn = tk.Button(
            btn_frame, text="Restart", font=self._font_bold, bg="#664400", fg=FG,
            activebackground="#886600", activeforeground=FG, relief=tk.FLAT,
            padx=12, pady=2, command=self._restart_services,
        )
        self._restart_btn.pack(side=tk.LEFT, padx=(0, 6))
        self._update_btn = tk.Button(
            btn_frame, text="Update", font=self._font_bold, bg="#1a5c32", fg=FG,
            activebackground="#2a8c4a", activeforeground=FG, relief=tk.FLAT,
            padx=12, pady=2, command=self._update_and_restart,
        )
        self._update_btn.pack(side=tk.LEFT)

        # Platform panel
        platform = tk.Frame(top, bg=BG_INPUT, highlightbackground=BORDER, highlightthickness=1)
        platform.pack(side=tk.LEFT, fill=tk.BOTH, padx=(5, 0))

        tk.Label(platform, text="Platform", font=self._font_bold, fg=FG, bg=BG_INPUT,
                 anchor="w").pack(padx=8, pady=(6, 2), anchor="w")

        self._svc_labels: dict[str, tk.Label] = {}
        for svc_id, svc_name in [
            ("svc-backend", "BACKEND"), ("svc-database", "DATABASE"),
            ("svc-redis", "REDIS"),
            ("svc-frontend", "FRONTEND"), ("svc-wsfeeds", "WS FEEDS"),
        ]:
            lbl = tk.Label(platform, text=f"● {svc_name:<8} OFFLINE", font=self._font,
                           fg=RED, bg=BG_INPUT, anchor="w")
            lbl.pack(padx=8, anchor="w")
            self._svc_labels[svc_id] = lbl

        tk.Label(platform, text=f"Dashboard  http://localhost:{FRONTEND_PORT}",
                 font=self._font_small, fg=BLUE_DIM, bg=BG_INPUT, anchor="w").pack(padx=8, anchor="w")
        tk.Label(platform, text=f"API        http://localhost:{BACKEND_PORT}",
                 font=self._font_small, fg=BLUE_DIM, bg=BG_INPUT, anchor="w").pack(padx=8, anchor="w")
        tk.Label(platform, text=f"Docs       http://localhost:{BACKEND_PORT}/docs",
                 font=self._font_small, fg=BLUE_DIM, bg=BG_INPUT, anchor="w").pack(padx=8, pady=(0, 6), anchor="w")

        # Metrics bar
        metrics = tk.Frame(parent, bg=BG)
        metrics.pack(fill=tk.X, padx=12, pady=(8, 0))

        self._metric_labels: dict[str, tk.Label] = {}
        for key, text in [
            ("uptime", "Uptime --:--:--"), ("workers", f"Workers 0/{len(WORKER_STATUS_ORDER)}"),
            ("health", "Health OFFLINE"), ("polls", "Polls 0"), ("logs", "Logs 0"),
        ]:
            lbl = tk.Label(metrics, text=text, font=self._font, fg=FG_MID, bg=BG, anchor="w")
            lbl.pack(side=tk.LEFT, padx=(0, 20))
            self._metric_labels[key] = lbl

        # Worker Command Center title
        tk.Label(parent, text="Worker Command Center", font=self._font_bold, fg=FG,
                 bg=BG, anchor="w").pack(padx=12, pady=(10, 4), anchor="w")

        # Worker grid
        worker_outer = tk.Frame(parent, bg=BG)
        worker_outer.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 4))

        self._worker_panels: dict[str, tk.Label] = {}
        COLS = 4
        for i, (worker_name, worker_label) in enumerate(WORKER_STATUS_ORDER):
            row, col = divmod(i, COLS)
            cell = tk.Frame(worker_outer, bg=BG_WORKER, highlightbackground=BORDER_LIGHT,
                            highlightthickness=1)
            cell.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")

            tk.Label(cell, text=worker_label, font=self._font_bold, fg=FG, bg=BG_WORKER,
                     anchor="w").pack(padx=6, pady=(4, 0), anchor="w")

            info_lbl = tk.Label(cell, text="● OFFLINE\nNo telemetry yet\n--\n--",
                                font=self._font_small, fg=FG_DIM, bg=BG_WORKER,
                                justify=tk.LEFT, anchor="w")
            info_lbl.pack(padx=6, pady=(0, 4), anchor="w", fill=tk.X)
            self._worker_panels[worker_name] = info_lbl

        for col in range(COLS):
            worker_outer.columnconfigure(col, weight=1, uniform="wk")
        for row in range((len(WORKER_STATUS_ORDER) + COLS - 1) // COLS):
            worker_outer.rowconfigure(row, weight=1)

        # Status bar
        self._status_bar = tk.Label(parent, text="", font=self._font_small, fg=FG_DIM, bg=BG, anchor="center")
        self._status_bar.pack(fill=tk.X, padx=10, pady=(0, 6))

    def _build_logs_tab(self, parent: tk.Frame) -> None:
        # Header
        header = tk.Frame(parent, bg=BG_PANEL)
        header.pack(fill=tk.X, padx=10, pady=(8, 0))

        self._log_header_left = tk.Label(header, text="LOGS", font=self._font_bold,
                                          fg=FG, bg=BG_PANEL, anchor="w")
        self._log_header_left.pack(side=tk.LEFT, padx=6)

        self._log_header_right = tk.Label(header, text="FOLLOWING  0/0 lines",
                                           font=self._font, fg=GREEN, bg=BG_PANEL, anchor="e")
        self._log_header_right.pack(side=tk.RIGHT, padx=6)

        # Filter controls
        controls = tk.Frame(parent, bg=BG)
        controls.pack(fill=tk.X, padx=10, pady=(6, 0))

        # Source filters
        src_frame = tk.Frame(controls, bg=BG)
        src_frame.pack(side=tk.LEFT, padx=(0, 8))
        self._source_btns: dict[str, tk.Button] = {}
        for key, label in [("all", "All"), ("backend", "Backend"), ("workers", "Workers"),
                            ("frontend", "Frontend"), ("system", "System")]:
            btn = tk.Button(
                src_frame, text=label, font=self._font_small,
                bg=BG_WORKER if key != "all" else FG_ACCENT,
                fg=FG if key != "all" else BG,
                activebackground=FG_ACCENT, activeforeground=BG,
                relief=tk.FLAT, padx=6, pady=1,
                command=lambda k=key: self._set_source_filter(k),
            )
            btn.pack(side=tk.LEFT, padx=1)
            self._source_btns[key] = btn

        # Level filters
        lvl_frame = tk.Frame(controls, bg=BG)
        lvl_frame.pack(side=tk.LEFT, padx=(0, 8))
        self._level_btns: dict[str, tk.Button] = {}
        for key, label in [("all", "Any"), ("error", "Error+"), ("warning", "Warn+"),
                            ("info", "Info+"), ("debug", "Debug+")]:
            btn = tk.Button(
                lvl_frame, text=label, font=self._font_small,
                bg=BG_WORKER if key != "all" else FG_ACCENT,
                fg=FG if key != "all" else BG,
                activebackground=FG_ACCENT, activeforeground=BG,
                relief=tk.FLAT, padx=6, pady=1,
                command=lambda k=key: self._set_level_filter(k),
            )
            btn.pack(side=tk.LEFT, padx=1)
            self._level_btns[key] = btn

        # Worker filter
        wk_frame = tk.Frame(controls, bg=BG)
        wk_frame.pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(wk_frame, text="<", font=self._font_small, bg=BG_WORKER, fg=FG,
                  relief=tk.FLAT, padx=4, command=lambda: self._cycle_worker_filter(-1)).pack(side=tk.LEFT)
        self._worker_filter_label = tk.Label(wk_frame, text="Worker: All Workers",
                                              font=self._font_small, fg=FG, bg=BG_INPUT,
                                              width=20, anchor="center",
                                              highlightbackground=BORDER_LIGHT, highlightthickness=1)
        self._worker_filter_label.pack(side=tk.LEFT, padx=2)
        tk.Button(wk_frame, text=">", font=self._font_small, bg=BG_WORKER, fg=FG,
                  relief=tk.FLAT, padx=4, command=lambda: self._cycle_worker_filter(1)).pack(side=tk.LEFT)

        # Search
        self._search_var = tk.StringVar()
        self._search_var.trace_add("write", lambda *_: self._on_search_changed())
        search_entry = tk.Entry(controls, textvariable=self._search_var, font=self._font_small,
                                bg=BG_INPUT, fg=FG, insertbackground=FG,
                                highlightbackground=BORDER_LIGHT, highlightthickness=1, relief=tk.FLAT)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        # Action buttons
        act_frame = tk.Frame(controls, bg=BG)
        act_frame.pack(side=tk.LEFT)
        tk.Button(act_frame, text="Clear", font=self._font_small, bg="#664400", fg=FG,
                  relief=tk.FLAT, padx=6, command=self._clear_logs).pack(side=tk.LEFT, padx=1)
        tk.Button(act_frame, text="Copy", font=self._font_small, bg="#1a5c32", fg=FG,
                  relief=tk.FLAT, padx=6, command=self._copy_logs).pack(side=tk.LEFT, padx=1)

        # Log text widget
        log_frame = tk.Frame(parent, bg=BORDER_LIGHT)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 8))

        self._log_text = tk.Text(
            log_frame, bg=BG_WORKER, fg=FG, font=self._font_small,
            wrap=tk.NONE, state=tk.DISABLED, relief=tk.FLAT,
            insertbackground=FG, selectbackground="#2a4a6a", selectforeground=FG,
            borderwidth=0, padx=4, pady=4,
        )
        scrollbar_y = tk.Scrollbar(log_frame, command=self._log_text.yview)
        scrollbar_x = tk.Scrollbar(log_frame, orient=tk.HORIZONTAL, command=self._log_text.xview)
        self._log_text.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Text tags for log levels
        self._log_text.tag_configure("ERROR", foreground=RED)
        self._log_text.tag_configure("WARNING", foreground=YELLOW)
        self._log_text.tag_configure("DEBUG", foreground=FG_DIM)
        self._log_text.tag_configure("INFO", foreground=FG)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self) -> None:
        # Start services in background thread
        threading.Thread(target=self._start_services, daemon=True).start()

        # Schedule periodic tasks
        self.root.after(LOG_FLUSH_MS, self._flush_log_buffer_loop)
        self.root.after(HEALTH_POLL_INTERVAL_MS, self._health_poll_loop)
        self.root.after(1000, self._tick_uptime_loop)

        self.root.mainloop()

    # ------------------------------------------------------------------
    # Periodic loops (scheduled on main thread via root.after)
    # ------------------------------------------------------------------
    def _flush_log_buffer_loop(self) -> None:
        if self._shutting_down:
            return
        self._flush_log_buffer()
        self.root.after(LOG_FLUSH_MS, self._flush_log_buffer_loop)

    def _health_poll_loop(self) -> None:
        if self._shutting_down:
            return
        threading.Thread(target=self._fetch_health, daemon=True).start()
        self.root.after(HEALTH_POLL_INTERVAL_MS, self._health_poll_loop)

    def _tick_uptime_loop(self) -> None:
        if self._shutting_down:
            return
        self._update_uptime()
        self.root.after(1000, self._tick_uptime_loop)

    # ------------------------------------------------------------------
    # Log buffer (thread-safe batched writes)
    # ------------------------------------------------------------------
    def _enqueue_log(self, text: str, source: str = "BACKEND", level: str = "INFO") -> None:
        worker_name = self._infer_worker_from_log(source, text) or ""
        with self._log_lock:
            self._log_buf.append((text, source, level, worker_name))

    def _flush_log_buffer(self) -> None:
        with self._log_lock:
            if not self._log_buf:
                return
            entries = list(self._log_buf)
            self._log_buf.clear()

        self._log_entries.extend(entries)

        if len(self._log_entries) > LOG_MAX_LINES:
            self._log_entries = self._log_entries[-(LOG_MAX_LINES - LOG_TRIM_BATCH):]

        matching = [
            (text, level)
            for text, source, level, worker_name in entries
            if self._matches_filter(text, source, level, worker_name)
        ]

        # Update worker mini-logs
        for text, source, level, worker_name in entries:
            if worker_name:
                self._append_worker_log(worker_name, text, update=False)
        # Re-render touched worker panels
        touched = {wn for _, _, _, wn in entries if wn}
        for wn in touched:
            self._render_worker_panel(wn)

        if not matching:
            return

        try:
            self._log_text.configure(state=tk.NORMAL)
            for text, level in matching:
                tag = level if level in ("ERROR", "WARNING", "DEBUG") else "INFO"
                self._log_text.insert(tk.END, text + "\n", tag)
            # Trim display
            line_count = int(self._log_text.index("end-1c").split(".")[0])
            if line_count > LOG_MAX_LINES:
                self._log_text.delete("1.0", f"{line_count - LOG_MAX_LINES + LOG_TRIM_BATCH}.0")
            self._log_text.configure(state=tk.DISABLED)
            self._log_text.see(tk.END)
            self._log_line_count = int(self._log_text.index("end-1c").split(".")[0]) - 1
        except tk.TclError:
            pass

        self._update_log_header()

    def _rebuild_log_view(self) -> None:
        matching = [
            (text, level)
            for text, source, level, worker_name in self._log_entries
            if self._matches_filter(text, source, level, worker_name)
        ]
        try:
            self._log_text.configure(state=tk.NORMAL)
            self._log_text.delete("1.0", tk.END)
            for text, level in matching:
                tag = level if level in ("ERROR", "WARNING", "DEBUG") else "INFO"
                self._log_text.insert(tk.END, text + "\n", tag)
            self._log_text.configure(state=tk.DISABLED)
            self._log_text.see(tk.END)
            self._log_line_count = len(matching)
        except tk.TclError:
            pass
        self._update_log_header()

    # ------------------------------------------------------------------
    # Log filters
    # ------------------------------------------------------------------
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

    def _set_source_filter(self, key: str) -> None:
        self._source_filter = key
        for k, btn in self._source_btns.items():
            btn.configure(bg=FG_ACCENT if k == key else BG_WORKER,
                         fg=BG if k == key else FG)
        self._rebuild_log_view()

    def _set_level_filter(self, key: str) -> None:
        self._level_filter = key
        for k, btn in self._level_btns.items():
            btn.configure(bg=FG_ACCENT if k == key else BG_WORKER,
                         fg=BG if k == key else FG)
        self._rebuild_log_view()

    def _cycle_worker_filter(self, direction: int) -> None:
        try:
            idx = list(WORKER_FILTER_ORDER).index(self._worker_filter)
        except ValueError:
            idx = 0
        self._worker_filter = WORKER_FILTER_ORDER[(idx + direction) % len(WORKER_FILTER_ORDER)]
        label = WORKER_FILTER_LABELS.get(self._worker_filter, self._worker_filter.replace("_", " ").upper())
        self._worker_filter_label.configure(text=f"Worker: {label}")
        self._rebuild_log_view()

    def _on_search_changed(self) -> None:
        self._search_filter = self._search_var.get().strip().lower()
        self._rebuild_log_view()

    def _clear_logs(self) -> None:
        self._log_entries.clear()
        try:
            self._log_text.configure(state=tk.NORMAL)
            self._log_text.delete("1.0", tk.END)
            self._log_text.configure(state=tk.DISABLED)
        except tk.TclError:
            pass
        self._log_line_count = 0
        self._update_log_header()

    def _copy_logs(self) -> None:
        try:
            text = self._log_text.get("1.0", tk.END).strip()
            if text:
                self.root.clipboard_clear()
                self.root.clipboard_append(text)
        except tk.TclError:
            pass

    # ------------------------------------------------------------------
    # Worker inference & mini logs
    # ------------------------------------------------------------------
    def _infer_worker_from_log(self, source: str, text: str) -> Optional[str]:
        source_upper = source.upper()
        if source_upper.startswith("WORKER-"):
            return None
        direct = WORKER_TAG_TO_NAME.get(source_upper)
        if direct:
            return direct
        if source_upper not in {"BACKEND", "WORKERS", "NEWS"}:
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

    def _normalize_worker_log_line(self, text: str) -> str:
        line = OSC_ESCAPE_RE.sub("", text.replace("\r", "")).strip()
        line = CONTROL_CHAR_RE.sub("", ANSI_ESCAPE_RE.sub("", line))
        # Parse JSON log lines to extract readable message + exception info
        if line.startswith("{"):
            formatted, _ = format_log_line(line, "")
            # strip the empty "[] " prefix from format_log_line
            formatted = formatted.lstrip("[] ")
            return formatted[:200] if formatted else ""
        if line.startswith("[") and "] " in line:
            line = line.split("] ", 1)[1]
        return " ".join(line.strip().split())[:200]

    # ------------------------------------------------------------------
    # Worker panel rendering
    # ------------------------------------------------------------------
    def _render_worker_panel(self, worker_name: str) -> None:
        snapshot = self._worker_state_cache.get(worker_name, {})
        status, status_class = self._resolve_worker_state(snapshot)
        meta = self._format_worker_meta(snapshot)
        logs = list(self._worker_logs.get(worker_name, []))
        log_lines = logs[-2:] if logs else ["--", "--"]
        while len(log_lines) < 2:
            log_lines.append("--")

        color = {
            "status-on": GREEN, "status-off": RED,
            "status-warn": YELLOW, "status-idle": FG_DIM,
        }.get(status_class, FG_DIM)

        display = f"● {status}\n{meta}\n{log_lines[0][:120]}\n{log_lines[1][:120]}"
        panel = self._worker_panels.get(worker_name)
        if panel:
            try:
                panel.configure(text=display, fg=color)
            except tk.TclError:
                pass

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
        if snapshot.get("enabled") is False:
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
                parts.append(f"mem {memory_mb:.1f}MB")
            pid = stats.get("pid")
            if isinstance(pid, int) and pid > 0:
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

        return " | ".join(parts) if parts else "No telemetry yet"

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

    # ------------------------------------------------------------------
    # UI updates
    # ------------------------------------------------------------------
    def _update_log_header(self) -> None:
        summary_parts = [f"source={SOURCE_FILTER_LABELS.get(self._source_filter, self._source_filter)}"]
        summary_parts.append(f"level={LEVEL_FILTER_LABELS.get(self._level_filter, self._level_filter)}")
        worker_label = WORKER_FILTER_LABELS.get(self._worker_filter, self._worker_filter)
        summary_parts.append(f"worker={worker_label}")
        if self._search_filter:
            q = self._search_filter[:37] + "..." if len(self._search_filter) > 40 else self._search_filter
            summary_parts.append(f"search='{q}'")
        try:
            self._log_header_left.configure(text=f"LOGS  {' '.join(summary_parts)}")
            self._log_header_right.configure(
                text=f"FOLLOWING  {self._log_line_count:,}/{len(self._log_entries):,} lines"
            )
        except tk.TclError:
            pass

    def _update_platform_item(self, widget_id: str, label: str, is_on: bool) -> None:
        state = "ONLINE" if is_on else "OFFLINE"
        color = GREEN if is_on else RED
        try:
            self._svc_labels[widget_id].configure(text=f"● {label:<8} {state}", fg=color)
        except (KeyError, tk.TclError):
            pass

    def _update_runtime_metrics(self) -> None:
        elapsed = time.time() - self.start_time
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)

        running = sum(
            1 for wn, _ in WORKER_STATUS_ORDER
            if bool(self._worker_state_cache.get(wn, {}).get("running", False))
        )
        total = len(WORKER_STATUS_ORDER)
        health_text = "OK" if self.backend_healthy else "OFFLINE"

        try:
            self._metric_labels["uptime"].configure(text=f"Uptime {h:02d}:{m:02d}:{s:02d}")
            self._metric_labels["workers"].configure(text=f"Workers {running}/{total}")
            self._metric_labels["health"].configure(
                text=f"Health {health_text}",
                fg=GREEN if self.backend_healthy else RED,
            )
            self._metric_labels["polls"].configure(text=f"Polls {self.health_poll_count}")
            self._metric_labels["logs"].configure(text=f"Logs {len(self._log_entries):,}")
        except tk.TclError:
            pass

    def _update_uptime(self) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            self._status_bar.configure(text=f"{ts}  |  Restart / Update buttons above  |  Logs tab for details")
        except tk.TclError:
            pass
        self._update_runtime_metrics()

    # ------------------------------------------------------------------
    # Service management
    # ------------------------------------------------------------------
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
        return {
            "host": parsed.hostname or "127.0.0.1",
            "port": parsed.port or 5432,
            "database": parsed.path.lstrip("/") or "homerun",
            "username": parsed.username or "homerun",
            "password": parsed.password or "homerun",
        }

    def _is_local_database_target(self, database_url: str) -> bool:
        host = str(self._database_target_from_url(database_url)["host"]).lower()
        return host in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}

    # ------------------------------------------------------------------
    # Redis lifecycle (parallels Postgres helpers above).  Local-only by
    # default; the GUI starts the docker-compose ``redis`` service before
    # spawning the backend so worker boot can rely on a healthy bus.
    # ------------------------------------------------------------------
    def _resolve_runtime_redis_url(self) -> str:
        raw_value = str(os.environ.get("REDIS_URL", "")).lstrip("﻿").strip().strip('"').strip("'")
        if raw_value:
            return raw_value.rstrip("/")
        return "redis://127.0.0.1:6379/0"

    def _redis_target_from_url(self, redis_url: str) -> dict[str, str | int]:
        parsed = urlsplit(redis_url)
        return {
            "host": parsed.hostname or "127.0.0.1",
            "port": int(parsed.port or 6379),
            "db": (parsed.path.lstrip("/") or "0"),
            "password": parsed.password or "",
        }

    def _is_local_redis_target(self, redis_url: str) -> bool:
        host = str(self._redis_target_from_url(redis_url)["host"]).lower()
        return host in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}

    def _probe_redis_ready(
        self,
        venv_python: Path,
        redis_url: str,
        *,
        retries: int,
        retry_delay_seconds: float,
    ) -> tuple[bool, str]:
        probe_env = self._build_runtime_env(process_role="api")
        probe_env["REDIS_URL"] = redis_url
        result = subprocess.run(
            [
                str(venv_python),
                str(PROJECT_ROOT / "scripts" / "infra" / "ensure_redis_ready.py"),
                "--redis-url",
                redis_url,
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
            **_windows_subprocess_kwargs(),
        )
        output = "\n".join(
            part.strip()
            for part in ((result.stdout or "").strip(), (result.stderr or "").strip())
            if part and part.strip()
        ).strip()
        return result.returncode == 0, output

    def _start_local_redis_runtime(self, redis_url: str) -> tuple[bool, str]:
        target = self._redis_target_from_url(redis_url)
        docker_bin = shutil.which("docker") or "docker"
        compose_file = PROJECT_ROOT / "scripts" / "infra" / "docker-compose.infra.yml"
        runtime_env = os.environ.copy()
        runtime_env["REDIS_HOST"] = str(target["host"])
        runtime_env["REDIS_PORT"] = str(target["port"])
        runtime_env.setdefault("REDIS_CONTAINER_NAME", "homerun-redis")
        runtime_env.setdefault("REDIS_IMAGE", "redis:7-alpine")

        # Prefer compose (writes the same flags / healthcheck as our YAML);
        # fall back to ``docker start`` (existing container) and a bare
        # ``docker run`` so a partial install — e.g. the user blew away the
        # compose project but the named container is still on disk — still
        # boots.  Mirror the postgres pattern verbatim.
        commands: list[list[str]] = [
            [docker_bin, "compose", "-f", str(compose_file), "up", "-d", "redis"],
            [docker_bin, "start", runtime_env["REDIS_CONTAINER_NAME"]],
            [
                docker_bin,
                "run",
                "--name",
                runtime_env["REDIS_CONTAINER_NAME"],
                "--detach",
                "--publish",
                f"{runtime_env['REDIS_HOST']}:{runtime_env['REDIS_PORT']}:6379",
                runtime_env["REDIS_IMAGE"],
                "redis-server",
                "--save",
                "",
                "--appendonly",
                "no",
                "--maxmemory",
                "1gb",
                "--maxmemory-policy",
                "volatile-lru",
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
                    **_windows_subprocess_kwargs(),
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

    def _ensure_redis_ready(self, venv_python: Path) -> tuple[bool, str]:
        """Ensure the Redis runtime is reachable.  Soft dependency: a
        failure here is logged but does not abort startup — every consumer
        treats Redis as a cache + bus with an in-memory fallback."""
        redis_url = self._resolve_runtime_redis_url()

        ready, output = self._probe_redis_ready(
            venv_python, redis_url, retries=2, retry_delay_seconds=0.25
        )
        if ready:
            return True, redis_url

        if not self._is_local_redis_target(redis_url):
            if output:
                self._enqueue_log(output, source="SYSTEM", level="WARNING")
            return False, redis_url

        self._enqueue_log(">>> Starting Redis runtime...", source="SYSTEM", level="INFO")
        started, start_output = self._start_local_redis_runtime(redis_url)
        if start_output:
            self._enqueue_log(
                start_output,
                source="SYSTEM",
                level="INFO" if started else "WARNING",
            )
        if not started:
            return False, redis_url

        ready, output = self._probe_redis_ready(
            venv_python, redis_url, retries=120, retry_delay_seconds=0.5
        )
        if output:
            self._enqueue_log(
                output, source="SYSTEM", level="INFO" if ready else "WARNING"
            )
        return ready, redis_url

    def _probe_database_ready(self, venv_python: Path, database_url: str, *, retries: int, retry_delay_seconds: float) -> tuple[bool, str]:
        probe_env = self._build_runtime_env(process_role="api")
        probe_env["DATABASE_URL"] = database_url
        result = subprocess.run(
            [str(venv_python), str(PROJECT_ROOT / "scripts" / "infra" / "ensure_postgres_ready.py"),
             "--database-url", database_url, "--retries", str(retries),
             "--retry-delay-seconds", str(retry_delay_seconds)],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True, env=probe_env,
            timeout=max(15, int(retries * max(retry_delay_seconds, 0.25) * 4)),
            **_windows_subprocess_kwargs(),
        )
        output = "\n".join(
            part.strip() for part in ((result.stdout or "").strip(), (result.stderr or "").strip())
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
            [docker_bin, "run", "--name", runtime_env["POSTGRES_CONTAINER_NAME"],
             "--detach", "--publish",
             f"{runtime_env['POSTGRES_HOST']}:{runtime_env['POSTGRES_PORT']}:5432",
             "--env", f"POSTGRES_DB={runtime_env['POSTGRES_DB']}",
             "--env", f"POSTGRES_USER={runtime_env['POSTGRES_USER']}",
             "--env", f"POSTGRES_PASSWORD={runtime_env['POSTGRES_PASSWORD']}",
             "--volume", "homerun-postgres-data:/var/lib/postgresql/data",
             runtime_env["POSTGRES_IMAGE"], "postgres", "-c", "max_connections=200"],
        ]

        outputs: list[str] = []
        for command in commands:
            try:
                result = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True,
                                        text=True, env=runtime_env, timeout=120, **_windows_subprocess_kwargs())
            except Exception as exc:
                outputs.append(f"{' '.join(command[:3])}: {exc}")
                continue
            combined = "\n".join(
                part.strip() for part in ((result.stdout or "").strip(), (result.stderr or "").strip())
                if part and part.strip()
            ).strip()
            if result.returncode == 0:
                return True, combined
            outputs.append(combined or f"{' '.join(command)} exited with code {result.returncode}")
        return False, "\n".join(entry for entry in outputs if entry)

    def _ensure_database_ready(self, venv_python: Path) -> tuple[bool, str]:
        database_url = self._resolve_runtime_database_url()
        self._write_runtime_database_url(database_url)

        ready, output = self._probe_database_ready(venv_python, database_url, retries=2, retry_delay_seconds=0.25)
        if ready:
            return True, database_url

        if not self._is_local_database_target(database_url):
            if output:
                self._enqueue_log(output, source="SYSTEM", level="ERROR")
            return False, database_url

        self._enqueue_log(">>> Starting Postgres runtime...", source="SYSTEM", level="INFO")
        started, start_output = self._start_local_postgres_runtime(database_url)
        if start_output:
            self._enqueue_log(start_output, source="SYSTEM", level="INFO" if started else "ERROR")
        if not started:
            return False, database_url

        ready, output = self._probe_database_ready(venv_python, database_url, retries=240, retry_delay_seconds=0.5)
        if output:
            self._enqueue_log(output, source="SYSTEM", level="INFO" if ready else "ERROR")
        return ready, database_url

    def _spawn_worker_plane(self, plane_name: str, *, reason: str) -> subprocess.Popen:
        source_tag = _WORKER_SOURCE_TAG_BY_PLANE[plane_name]
        env = self._build_runtime_env(process_role="worker")
        # Debug log-file teeing: when ``HOMERUN_DEBUG=1`` is set
        # (typically by the ``-Debug`` flag on the launcher), the
        # worker also writes its full JSON log stream to a per-plane
        # file under ``tools/.harness/<plane>.log``.  ``perf_harness
        # .py --tail`` consumes those files for offline analysis.
        # The stdout pipe to the GUI is unaffected — file is
        # additive, not a replacement.
        if env.get("HOMERUN_DEBUG", "").strip() in {"1", "true", "TRUE", "yes"}:
            harness_dir = PROJECT_ROOT / "tools" / ".harness"
            try:
                harness_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            log_path = harness_dir / f"{plane_name}.log"
            env[f"HOMERUN_DEBUG_LOG_FILE_{plane_name.upper()}"] = str(log_path)
            # Also clear the file at spawn time so each launch has a
            # clean tail boundary — important so the harness doesn't
            # double-count lines from a prior run.
            try:
                log_path.write_text("", encoding="utf-8")
            except Exception:
                pass
        worker_proc = subprocess.Popen(
            [str(self._venv_python_path()), "-m", "workers.host", plane_name],
            cwd=str(BACKEND_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env,
            **_windows_subprocess_kwargs(),
        )
        _assign_to_job(worker_proc)
        self.worker_procs[plane_name] = worker_proc
        self._enqueue_log(f">>> Starting {plane_name} worker plane ({reason})...", source=source_tag, level="INFO")
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
                    source=_WORKER_SOURCE_TAG_BY_PLANE[plane_name], level="WARNING",
                )
            try:
                self._spawn_worker_plane(plane_name, reason="health-supervisor")
            except Exception as exc:
                self._enqueue_log(f"FATAL: Failed to restart {plane_name} worker plane: {exc}",
                                  source=_WORKER_SOURCE_TAG_BY_PLANE[plane_name], level="ERROR")

    def _ensure_worker_planes_alive_threadsafe(self) -> None:
        with self._worker_supervisor_lock:
            if self._worker_supervisor_inflight:
                return
            self._worker_supervisor_inflight = True
        try:
            self._ensure_worker_planes_alive()
        except Exception as exc:
            self._enqueue_log(f"Worker plane supervision failed: {exc}", source="SYSTEM", level="ERROR")
        finally:
            with self._worker_supervisor_lock:
                self._worker_supervisor_inflight = False

    def _start_services(self) -> None:
        self._enqueue_log(">>> Starting services...", source="SYSTEM", level="INFO")
        self._reset_worker_telemetry()
        kill_legacy_worker_processes()
        kill_port(BACKEND_PORT)
        kill_port(FRONTEND_PORT)

        venv_python = self._venv_python_path()
        if not venv_python.exists():
            setup_cmd = ".\\scripts\\infra\\setup.ps1" if sys.platform == "win32" else "./scripts/infra/setup.sh"
            self._enqueue_log(f"ERROR: Virtual environment not found. Run {setup_cmd} first.", source="BACKEND", level="ERROR")
            return

        self._enqueue_log(">>> Ensuring database runtime is ready...", source="SYSTEM", level="INFO")
        database_ready, database_url = self._ensure_database_ready(venv_python)
        if not database_ready:
            self._enqueue_log(f"FATAL: Database is not reachable at {database_url}.", source="BACKEND", level="ERROR")
            return

        # Redis is a soft dependency — log + continue if unavailable.  The
        # backend's redis_client treats a missing pool as "use in-memory
        # fallback" everywhere, so failure here degrades freshness across
        # processes but does not stop trading.  Propagate via os.environ
        # so every subsequently-spawned worker / backend inherits it via
        # ``_build_runtime_env`` (which does ``os.environ.copy()``).
        self._enqueue_log(">>> Ensuring Redis runtime is ready...", source="SYSTEM", level="INFO")
        redis_ready, redis_url = self._ensure_redis_ready(venv_python)
        os.environ["REDIS_URL"] = redis_url
        os.environ["REDIS_ENABLED"] = "true" if redis_ready else "false"
        if not redis_ready:
            self._enqueue_log(
                f"WARNING: Redis is not reachable at {redis_url}; running with in-memory fallback.",
                source="SYSTEM",
                level="WARNING",
            )

        env = self._build_runtime_env(process_role="api")
        env["DATABASE_URL"] = database_url
        env["REDIS_URL"] = redis_url
        env["REDIS_ENABLED"] = "true" if redis_ready else "false"

        self._enqueue_log(">>> Starting backend (uvicorn)...", source="BACKEND", level="INFO")
        try:
            self.backend_proc = subprocess.Popen(
                [str(venv_python), "-m", "uvicorn", "main:app", "--host", "0.0.0.0",
                 "--port", str(BACKEND_PORT), "--log-level", "info"],
                cwd=str(BACKEND_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                **_windows_subprocess_kwargs(),
            )
            _assign_to_job(self.backend_proc)
        except Exception as e:
            self._enqueue_log(f"FATAL: Failed to start backend: {e}", source="BACKEND", level="ERROR")
            return

        self._start_stream_thread(self.backend_proc, "BACKEND")
        self.worker_procs = {}

    def _start_frontend(self) -> None:
        env = os.environ.copy()
        env["BROWSER"] = "none"
        env["FORCE_COLOR"] = "0"

        if self._read_network_access_setting():
            env["VITE_HOST"] = "0.0.0.0"
            self._enqueue_log("Network access enabled — binding frontend to 0.0.0.0", source="FRONTEND", level="INFO")

        self._enqueue_log(">>> Starting frontend (npm run dev)...", source="FRONTEND", level="INFO")
        try:
            npm_bin = shutil.which("npm") or "npm"
            self.frontend_proc = subprocess.Popen(
                [npm_bin, "run", "dev"], cwd=str(PROJECT_ROOT / "frontend"),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                **_windows_subprocess_kwargs(),
            )
            _assign_to_job(self.frontend_proc)
        except Exception as e:
            self._enqueue_log(f"FATAL: Failed to start frontend: {e}", source="FRONTEND", level="ERROR")
            self._frontend_starting = False
            return

        self._start_stream_thread(self.frontend_proc, "FRONTEND")

    def _read_network_access_setting(self) -> bool:
        import urllib.request
        import urllib.error
        url = f"http://127.0.0.1:{BACKEND_PORT}/api/settings"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                payload = data.get("data", data)
                network = payload.get("network", {})
                return bool(network.get("allow_network_access", False))
        except Exception:
            return False

    def _stream_output(self, proc: subprocess.Popen, tag: str) -> None:
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

                if tag == "BACKEND":
                    if "Application startup complete" in line or "Uvicorn running" in line:
                        self._request_frontend_start("backend startup log")
        except Exception:
            pass
        finally:
            if tag == "FRONTEND":
                self._frontend_starting = False
            if not self._shutting_down:
                self._enqueue_log(f"[{tag}] Process exited", source=tag, level="INFO")

    def _start_stream_thread(self, proc: subprocess.Popen, tag: str) -> None:
        thread = threading.Thread(target=self._stream_output, args=(proc, tag),
                                  name=f"{tag.lower()}-log-stream", daemon=True)
        thread.start()

    def _request_frontend_start(self, reason: str) -> None:
        if self._frontend_starting or self._frontend_alive():
            return
        self._frontend_starting = True
        self._enqueue_log(f">>> Triggering frontend start ({reason})...", source="FRONTEND", level="INFO")
        threading.Thread(target=self._start_frontend, daemon=True).start()

    def _frontend_alive(self) -> bool:
        return self.frontend_proc is not None and self.frontend_proc.poll() is None

    def _open_in_browser(self) -> None:
        webbrowser.open(f"http://localhost:{FRONTEND_PORT}")

    def _restart_services(self) -> None:
        if self._service_op_in_progress:
            return
        threading.Thread(target=self._do_restart, daemon=True).start()

    def _do_restart(self) -> None:
        self._service_op_in_progress = True
        try:
            self.root.after(0, lambda: self._restart_btn.configure(state=tk.DISABLED))
            self.root.after(0, lambda: self._update_btn.configure(state=tk.DISABLED))
        except Exception:
            pass
        self._enqueue_log(">>> Restarting services...", source="SYSTEM", level="INFO")
        self._frontend_starting = False
        self._kill_children()
        kill_port(BACKEND_PORT)
        kill_port(FRONTEND_PORT)
        self._start_services()
        self._service_op_in_progress = False
        try:
            self.root.after(0, lambda: self._restart_btn.configure(state=tk.NORMAL))
            self.root.after(0, lambda: self._update_btn.configure(state=tk.NORMAL))
        except Exception:
            pass

    def _update_and_restart(self) -> None:
        if self._service_op_in_progress:
            return
        threading.Thread(target=self._do_update_and_restart, daemon=True).start()

    def _do_update_and_restart(self) -> None:
        self._service_op_in_progress = True
        try:
            self.root.after(0, lambda: self._restart_btn.configure(state=tk.DISABLED))
            self.root.after(0, lambda: self._update_btn.configure(state=tk.DISABLED))
        except Exception:
            pass
        self._enqueue_log(">>> Running git pull --ff-only...", source="SYSTEM", level="INFO")

        if shutil.which("git") is None:
            self._enqueue_log("ERROR: git is not available on PATH.", source="SYSTEM", level="ERROR")
            self._service_op_in_progress = False
            return

        try:
            pull_result = subprocess.run(
                ["git", "pull", "--ff-only"], cwd=str(PROJECT_ROOT),
                capture_output=True, text=True, timeout=300, **_windows_subprocess_kwargs(),
            )
            for line in (pull_result.stdout or "").splitlines():
                self._enqueue_log(line, source="SYSTEM", level="INFO")
            for line in (pull_result.stderr or "").splitlines():
                self._enqueue_log(line, source="SYSTEM", level="WARNING")
            if pull_result.returncode != 0:
                self._enqueue_log("ERROR: git pull failed; aborting update/restart.", source="SYSTEM", level="ERROR")
                self._service_op_in_progress = False
                return
        except Exception as exc:
            self._enqueue_log(f"ERROR: git pull failed with exception: {exc}", source="SYSTEM", level="ERROR")
            self._service_op_in_progress = False
            return

        setup_cmd: list[str]
        if sys.platform == "win32":
            setup_cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                         "-File", str(PROJECT_ROOT / "scripts" / "infra" / "setup.ps1")]
        else:
            setup_cmd = ["bash", str(PROJECT_ROOT / "scripts" / "infra" / "setup.sh")]

        self._enqueue_log(">>> Running setup script...", source="SYSTEM", level="INFO")
        try:
            setup_proc = subprocess.Popen(setup_cmd, cwd=str(PROJECT_ROOT),
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                          **_windows_subprocess_kwargs())
            assert setup_proc.stdout is not None
            for line in setup_proc.stdout:
                msg = line.rstrip()
                if msg:
                    self._enqueue_log(msg, source="SYSTEM", level="INFO")
            code = setup_proc.wait(timeout=900)
            if code != 0:
                self._enqueue_log("ERROR: setup failed; services were not restarted.", source="SYSTEM", level="ERROR")
                self._service_op_in_progress = False
                return
        except Exception as exc:
            self._enqueue_log(f"ERROR: setup failed with exception: {exc}", source="SYSTEM", level="ERROR")
            self._service_op_in_progress = False
            return

        self._frontend_starting = False
        self._kill_children()
        kill_port(BACKEND_PORT)
        kill_port(FRONTEND_PORT)
        self._start_services()
        self._enqueue_log(">>> Update complete. Services restarting...", source="SYSTEM", level="INFO")
        self._service_op_in_progress = False
        try:
            self.root.after(0, lambda: self._restart_btn.configure(state=tk.NORMAL))
            self.root.after(0, lambda: self._update_btn.configure(state=tk.NORMAL))
        except Exception:
            pass

    def _reset_worker_telemetry(self) -> None:
        with self._log_lock:
            pass  # _worker_event_buf removed in GUI version
        self._worker_last_state.clear()
        self._worker_last_activity.clear()
        self._worker_last_error.clear()
        for worker_name, _ in WORKER_STATUS_ORDER:
            self._worker_logs[worker_name].clear()
            self._worker_state_cache[worker_name] = {}
            self._render_worker_panel(worker_name)

    # ------------------------------------------------------------------
    # Health polling
    # ------------------------------------------------------------------
    def _fetch_health(self) -> None:
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
                self.root.after(0, self._apply_health_safe, data)
        except Exception:
            self.root.after(0, self._apply_health_offline)
        finally:
            with self._health_poll_lock:
                self._health_poll_inflight = False

    def _apply_health_safe(self, data: dict) -> None:
        try:
            self._apply_health(data)
        except Exception as exc:
            self._enqueue_log(f"GUI health apply failed: {exc}", source="SYSTEM", level="ERROR")

    def _apply_health(self, data: dict) -> None:
        self.backend_healthy = True
        self.health_data = data
        self.health_poll_count += 1
        self._health_consecutive_failures = 0
        self._health_last_success_monotonic = time.monotonic()
        services = data.get("services", {})
        workers = data.get("workers")
        if workers is None:
            workers = services.get("workers", {})

        ws = services.get("ws_feeds", {})
        database_healthy = self._resolve_database_health(data, services)
        redis_healthy = self._resolve_redis_health(data)

        self._update_platform_item("svc-backend", "BACKEND", True)
        self._update_platform_item("svc-database", "DATABASE", database_healthy)
        self._update_platform_item("svc-redis", "REDIS", redis_healthy)
        if not self._frontend_alive():
            self._request_frontend_start("backend health ready")
        self._update_platform_item("svc-frontend", "FRONTEND", self._frontend_alive())
        ws_started = bool(ws.get("started", False)) if isinstance(ws, dict) else False
        ws_available = bool(ws.get("websockets_available", True)) if isinstance(ws, dict) else False
        ws_healthy = bool(ws.get("healthy", False)) or (ws_started and ws_available)
        self._update_platform_item("svc-wsfeeds", "WS FEEDS", ws_healthy)

        self._update_worker_panels(workers, services)
        self._update_runtime_metrics()

    def _apply_health_offline(self) -> None:
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
        self._update_platform_item("svc-redis", "REDIS", False)
        self._update_platform_item("svc-frontend", "FRONTEND", self._frontend_alive())
        self._update_platform_item("svc-wsfeeds", "WS FEEDS", False)
        self._update_workers_from_processes()
        self._update_runtime_metrics()

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

    def _resolve_redis_health(self, data: dict) -> bool:
        """Read Redis health from the /health/gui payload.

        Backend reports Redis under both ``checks.redis`` (boolean roll-up
        used by the dashboard) and ``redis`` (full status snapshot).
        Prefer the snapshot when present so we can tell "disabled by
        config" from "configured but unreachable" in future surfaces;
        for the ON/OFFLINE pill we just need the boolean.
        """
        checks = data.get("checks")
        if isinstance(checks, dict) and "redis" in checks:
            return bool(checks.get("redis"))
        snapshot = data.get("redis")
        if isinstance(snapshot, dict):
            if "healthy" in snapshot:
                if not bool(snapshot.get("enabled", True)):
                    # Disabled by config — surface as ONLINE so the GUI
                    # doesn't flap a healthy box.  The user can flip
                    # REDIS_ENABLED=true to re-enable.
                    return True
                return bool(snapshot.get("healthy"))
        return False

    def _normalize_workers_payload(self, workers: object, services: dict) -> dict[str, dict]:
        by_name: dict[str, dict] = {}
        if isinstance(workers, list):
            by_name = {str(item.get("worker_name")): item for item in workers
                       if isinstance(item, dict) and item.get("worker_name")}
        elif isinstance(workers, dict):
            by_name = {str(name): item for name, item in workers.items() if isinstance(item, dict)}
        for worker_name, _ in WORKER_STATUS_ORDER:
            if worker_name not in by_name:
                by_name[worker_name] = self._fallback_worker_snapshot(worker_name, services)
        return by_name

    def _fallback_worker_snapshot(self, worker_name: str, services: dict) -> dict:
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
        return {
            "running": running, "enabled": True, "interval_seconds": None,
            "last_run_at": None, "lag_seconds": None, "current_activity": None, "last_error": None,
        }

    def _update_worker_panels(self, workers: object, services: dict) -> None:
        by_name = self._normalize_workers_payload(workers, services)
        for worker_name, _ in WORKER_STATUS_ORDER:
            snapshot = by_name.get(worker_name, {})
            if not isinstance(snapshot, dict):
                snapshot = {}
            self._worker_state_cache[worker_name] = snapshot
            self._emit_worker_snapshot_events(worker_name, snapshot)
            self._render_worker_panel(worker_name)

    def _emit_worker_snapshot_events(self, worker_name: str, snapshot: dict) -> None:
        state, _ = self._resolve_worker_state(snapshot)
        prev_state = self._worker_last_state.get(worker_name)
        if prev_state is not None and prev_state != state:
            self._append_worker_log(worker_name, f"state changed to {state}", update=False)
        self._worker_last_state[worker_name] = state

        activity = snapshot.get("current_activity")
        activity = activity.strip() if isinstance(activity, str) else ""
        if activity and activity != self._worker_last_activity.get(worker_name):
            self._worker_last_activity[worker_name] = activity
            self._append_worker_log(worker_name, activity, update=False)

        last_error = snapshot.get("last_error")
        last_error = last_error.strip() if isinstance(last_error, str) else ""
        if last_error and last_error != self._worker_last_error.get(worker_name):
            self._worker_last_error[worker_name] = last_error
            self._append_worker_log(worker_name, f"ERROR: {last_error}", update=False)

    def _update_workers_from_processes(self) -> None:
        for worker_name, _ in WORKER_STATUS_ORDER:
            plane_name = _WORKER_PLANE_BY_NAME.get(worker_name)
            proc = self.worker_procs.get(plane_name or "")
            running = proc is not None and proc.poll() is None
            self._worker_state_cache[worker_name] = {
                "running": running, "enabled": running, "interval_seconds": None,
                "last_run_at": None, "lag_seconds": None,
                "current_activity": "Process running" if running else None, "last_error": None,
            }
            self._render_worker_panel(worker_name)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def _kill_children(self) -> None:
        procs = [p for p in (self.backend_proc, self.frontend_proc, *self.worker_procs.values())
                 if p and p.poll() is None]
        for proc in procs:
            try:
                if sys.platform == "win32":
                    _terminate_process_tree(proc.pid)
                else:
                    proc.kill()
            except Exception:
                pass
        for proc in procs:
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass
        for proc in procs:
            try:
                proc.wait(timeout=1)
            except Exception:
                pass
        self.worker_procs = {}
        self._kill_orphaned_workers()

    def _kill_orphaned_workers(self) -> None:
        if sys.platform != "win32":
            return
        backend_dir = str(BACKEND_DIR)
        project_root = str(PROJECT_ROOT)
        gui_bootstrap_name = "_gui_bootstrap.py"
        try:
            data = _list_windows_python_processes()
            if not data:
                return
            protected_pids = _current_windows_python_lineage(data)
            for entry in data:
                pid = entry.get("ProcessId")
                cmd = entry.get("CommandLine") or ""
                try:
                    pid_int = int(pid or 0)
                except (TypeError, ValueError):
                    continue
                if not pid_int or pid_int in protected_pids:
                    continue
                is_homerun = False
                if "workers.host" in cmd:
                    is_homerun = True
                elif "workers.runner" in cmd or ("workers." in cmd and "_worker" in cmd):
                    is_homerun = True
                elif "uvicorn" in cmd and "main:app" in cmd:
                    is_homerun = True
                elif "gui.py" in cmd or gui_bootstrap_name in cmd:
                    is_homerun = True
                if not is_homerun:
                    continue
                if (
                    "workers.host" not in cmd
                    and "workers.runner" not in cmd
                    and "_worker" not in cmd
                    and "gui.py" not in cmd
                    and gui_bootstrap_name not in cmd
                    and backend_dir not in cmd
                    and project_root not in cmd
                ):
                    continue
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(pid_int)],
                        capture_output=True,
                        timeout=5,
                        **_windows_subprocess_kwargs(),
                    )
                except Exception:
                    pass
        except Exception:
            pass

    def _report_tk_exception(self, exc_type, exc_value, exc_traceback) -> None:
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)

    def _on_quit(self) -> None:
        self._shutting_down = True
        self._kill_children()
        try:
            self.root.destroy()
        except Exception:
            pass
        os._exit(0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    venv_dir = BACKEND_DIR / "venv"
    if not venv_dir.exists():
        setup_cmd = ".\\scripts\\infra\\setup.ps1" if sys.platform == "win32" else "./scripts/infra/setup.sh"
        print(f"Setup not complete. Run {setup_cmd} first.")
        sys.exit(1)
    if not _acquire_single_instance_guard():
        _notify_existing_instance()
        sys.exit(0)

    if sys.platform == "win32":
        def _sigbreak_handler(signum, frame):
            _sigbreak_kill_children()
            os._exit(1)
        try:
            signal.signal(signal.SIGBREAK, _sigbreak_handler)
        except (AttributeError, OSError):
            pass

    def _unhandled_exception_hook(exc_type, exc_value, exc_traceback):
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)

    def _thread_exception_hook(args: threading.ExceptHookArgs) -> None:
        traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback, file=sys.stderr)

    sys.excepthook = _unhandled_exception_hook
    threading.excepthook = _thread_exception_hook

    app = HomerunApp()
    app.run()


if __name__ == "__main__":
    main()
