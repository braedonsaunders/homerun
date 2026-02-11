#!/usr/bin/env python3
"""Homerun TUI - Beautiful terminal interface for the Homerun trading platform."""
from __future__ import annotations

import collections
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Button,
    Footer,
    Header,
    Label,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BACKEND_PORT = 8000
FRONTEND_PORT = 3000
# Use 127.0.0.1 instead of localhost; on Windows, localhost can resolve to
# ::1 (IPv6) first while uvicorn only binds 0.0.0.0 (IPv4), causing timeouts.
HEALTH_URL = f"http://127.0.0.1:{BACKEND_PORT}/health/detailed"
PROJECT_ROOT = Path(__file__).parent.resolve()
BACKEND_DIR = PROJECT_ROOT / "backend"

LOG_MAX_LINES = 5000
LOG_TRIM_BATCH = 1000  # Remove this many lines when cap is hit
LOG_FLUSH_MS = 500  # Flush buffered lines every N ms

# Log level ordering for filter comparison
LOG_LEVEL_ORDER = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}

WORKER_STATUS_ORDER: list[tuple[str, str]] = [
    ("scanner", "SCANNER"),
    ("discovery", "DISCOVERY"),
    ("weather", "WEATHER"),
    ("news", "NEWS"),
    ("crypto", "CRYPTO"),
    ("tracked_traders", "TRACKED"),
    ("autotrader", "AUTOTRADER"),
    ("world_intelligence", "WORLD INTEL"),
]

WORKER_TAG_TO_NAME: dict[str, str] = {
    "SCANNER": "scanner",
    "DISCOVERY": "discovery",
    "WEATHER": "weather",
    "NEWS": "news",
    "CRYPTO": "crypto",
    "TRACKED": "tracked_traders",
    "AUTOTRADER": "autotrader",
    "WORLDINTEL": "world_intelligence",
}

WORKER_BACKEND_HINTS: tuple[tuple[str, str], ...] = (
    ("scanner_worker", "scanner"),
    ("discovery_worker", "discovery"),
    ("weather_worker", "weather"),
    ("news_worker", "news"),
    ("crypto_worker", "crypto"),
    ("tracked_traders_worker", "tracked_traders"),
    ("autotrader_worker", "autotrader"),
    ("world_intelligence_worker", "world_intelligence"),
)

WORKER_MINI_LOG_LINES = 2
WORKER_MINI_LOG_WIDTH = 84

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
}

#workers-grid {
    layout: grid;
    grid-size: 4;
    grid-gutter: 1;
    padding: 0 1;
    margin: 0 1 1 1;
    height: auto;
}

.worker-panel {
    height: 8;
    background: #0f1d2c;
    border: round #2b4961;
    padding: 0 1;
}

.worker-panel-title {
    color: #d2e8f7;
    text-style: bold;
    height: 1;
}

.worker-panel-status {
    text-style: bold;
    height: 1;
}

.worker-panel-meta {
    color: #89acc3;
    height: 1;
}

.worker-panel-logs {
    color: #9ab4c6;
    height: 3;
    padding: 0 0 0 0;
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
    layout: horizontal;
    height: auto;
    max-height: 5;
    padding: 0 1;
    margin: 0 1 0 1;
}

#log-controls Button {
    min-width: 12;
    margin: 0 1 0 0;
}

#log-level-select {
    max-width: 22;
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


class WorkerPanel(Static):
    """Compact worker telemetry panel with mini logs."""

    def __init__(self, worker_name: str, label: str) -> None:
        super().__init__(id=f"worker-{worker_name}", classes="worker-panel")
        self.worker_name = worker_name
        self._label = label

    def compose(self) -> ComposeResult:
        yield Label(self._label, classes="worker-panel-title")
        yield Label("OFFLINE", id=f"{self.id}-status", classes="worker-panel-status status-off")
        yield Label("No telemetry yet", id=f"{self.id}-meta", classes="worker-panel-meta")
        yield Static(
            "  waiting for worker events\n  --",
            id=f"{self.id}-logs",
            classes="worker-panel-logs",
        )

    def update_state(self, status: str, status_class: str, meta: str) -> None:
        try:
            status_label = self.query_one(f"#{self.id}-status", Label)
            status_label.update(status)
            status_label.remove_class("status-on")
            status_label.remove_class("status-off")
            status_label.remove_class("status-warn")
            status_label.remove_class("status-idle")
            status_label.add_class(status_class)
            self.query_one(f"#{self.id}-meta", Label).update(meta)
        except Exception:
            pass

    def update_logs(self, lines: list[str]) -> None:
        if lines:
            body = "\n".join(f"  {line}" for line in lines[-WORKER_MINI_LOG_LINES:])
        else:
            body = "  waiting for worker events\n  --"
        try:
            self.query_one(f"#{self.id}-logs", Static).update(body)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Plain-text log formatter (no Rich markup -- TextArea is plain text)
# ---------------------------------------------------------------------------
def format_log_line(line: str, tag: str) -> tuple[str, str]:
    """Format a raw log line into readable plain text for the log viewer.

    Returns (formatted_text, level).
    """
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

    # Process handles
    scanner_worker_proc: Optional[subprocess.Popen] = None
    discovery_worker_proc: Optional[subprocess.Popen] = None
    weather_worker_proc: Optional[subprocess.Popen] = None
    news_worker_proc: Optional[subprocess.Popen] = None
    crypto_worker_proc: Optional[subprocess.Popen] = None
    tracked_worker_proc: Optional[subprocess.Popen] = None
    autotrader_worker_proc: Optional[subprocess.Popen] = None
    backend_proc: Optional[subprocess.Popen] = None
    frontend_proc: Optional[subprocess.Popen] = None

    # State
    start_time: float = 0.0
    backend_healthy: bool = False
    health_data: dict = {}
    health_poll_count: int = 0
    _is_light_mode: bool = False

    def __init__(self) -> None:
        super().__init__()
        # Thread-safe log line buffer: worker threads append here,
        # a periodic timer flushes into the TextArea on the main thread.
        self._log_buf: collections.deque[tuple[str, str, str]] = collections.deque(
            maxlen=2000
        )
        self._log_lock = threading.Lock()
        # Master list of all log entries for rebuilding filtered views.
        self._log_entries: list[tuple[str, str, str]] = []  # (text, source, level)
        self._log_line_count = 0
        # Track whether user is scrolled to the bottom (auto-follow mode).
        self._log_follow = True
        # Filter state
        self._source_filter = "all"  # "all", "backend", "frontend"
        self._level_filter = "all"  # "all", "debug", "info", "warning", "error"
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
                yield Static("[red]\u25cf[/] FRONTEND  OFFLINE", id="svc-frontend", classes="platform-item")
                yield Static("[red]\u25cf[/] WS FEEDS  OFFLINE", id="svc-wsfeeds", classes="platform-item")
                yield Static(f"Dashboard  http://localhost:{FRONTEND_PORT}", classes="platform-url")
                yield Static(f"API        http://localhost:{BACKEND_PORT}", classes="platform-url")
                yield Static(f"Docs       http://localhost:{BACKEND_PORT}/docs", classes="platform-url")

        # Runtime metrics bar
        with Horizontal(id="metrics-bar"):
            yield Static("Uptime [bold]--:--:--[/]", id="metric-uptime", classes="metric-item")
            yield Static("Workers [bold]0/7[/]", id="metric-workers", classes="metric-item")
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
                    "[bold]LOGS[/]  [dim]Ctrl+C or Copy button to copy[/]",
                    id="log-header-left",
                )
                yield Static(
                    "[bold green]FOLLOWING[/]  0 lines",
                    id="log-header-right",
                )
            with Horizontal(id="log-controls"):
                yield Button("All", id="src-all", variant="primary")
                yield Button("Backend", id="src-backend", variant="default")
                yield Button("Frontend", id="src-frontend", variant="default")
                yield Select(
                    [
                        ("All Levels", "all"),
                        ("Error", "error"),
                        ("Warning", "warning"),
                        ("Info", "info"),
                        ("Debug", "debug"),
                    ],
                    value="all",
                    id="log-level-select",
                    allow_blank=False,
                )
                yield Button("Clear", id="log-clear-btn", variant="warning")
                yield Button("Copy", id="log-copy-btn", variant="success")
                yield Button("↓ Bottom", id="log-bottom-btn", variant="primary")
            yield TextArea(
                "",
                id="log-output",
                read_only=True,
                show_line_numbers=True,
                language=None,
                soft_wrap=True,
            )

    # ---- Lifecycle ----
    def on_mount(self) -> None:
        self.start_time = time.time()
        self._start_services()
        self._poll_health()
        self._update_uptime()
        # Flush log buffer periodically (batched writes for performance)
        self.set_interval(LOG_FLUSH_MS / 1000.0, self._flush_log_buffer)
        # Check scroll state less frequently (not on_idle which fires constantly)
        self.set_interval(0.5, self._check_scroll_follow)

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
            ta = self.query_one("#log-output", TextArea)
            text = ta.selected_text
            if not text:
                text = ta.text
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
    def _matches_filter(self, source: str, level: str) -> bool:
        """Check if a log entry matches the current source and level filters."""
        if self._source_filter != "all":
            if source.lower() != self._source_filter:
                return False
        if self._level_filter != "all":
            min_level = LOG_LEVEL_ORDER.get(self._level_filter.upper(), 0)
            entry_level = LOG_LEVEL_ORDER.get(level.upper(), 1)
            if entry_level < min_level:
                return False
        return True

    def _rebuild_log_view(self) -> None:
        """Rebuild the TextArea content from master entries based on current filters."""
        try:
            ta = self.query_one("#log-output", TextArea)
        except Exception:
            return

        matching = [
            text
            for text, source, level in self._log_entries
            if self._matches_filter(source, level)
        ]

        # Clear the TextArea
        end = ta.document.end
        if end != (0, 0):
            ta.delete((0, 0), end)

        # Insert filtered content
        if matching:
            content = "\n".join(matching)
            ta.insert(content, location=(0, 0))

        self._log_line_count = len(matching)
        self._log_follow = True
        ta.scroll_end(animate=False)
        self._update_log_header()

    # ---- Log buffer: thread-safe batched writes ----
    def _enqueue_log(self, text: str, source: str = "BACKEND", level: str = "INFO") -> None:
        """Called from worker threads. Appends to buffer; main-thread timer flushes."""
        with self._log_lock:
            self._log_buf.append((text, source, level))
            worker_name = self._infer_worker_from_log(source, text)
            if worker_name:
                self._worker_event_buf.append((worker_name, text))

    def _flush_log_buffer(self) -> None:
        """Called on the main thread by a periodic timer. Flushes pending lines
        into the TextArea in one batch for performance."""
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
            for text, source, level in entries
            if self._matches_filter(source, level)
        ]

        if not matching:
            self._flush_worker_event_buffer()
            return

        try:
            ta = self.query_one("#log-output", TextArea)
        except Exception:
            self._flush_worker_event_buffer()
            return

        # Snapshot scroll state and selection BEFORE any mutation.
        at_bottom = self._log_follow
        saved_scroll_y = ta.scroll_y
        saved_selection = ta.selection

        # Build the chunk to insert
        chunk = "\n".join(matching)
        if self._log_line_count > 0:
            chunk = "\n" + chunk

        # Insert at the end of the document
        end = ta.document.end
        ta.insert(chunk, location=end)
        self._log_line_count += len(matching)

        # Restore selection after insert -- insert at end moved cursor
        # but didn't change line numbers for existing text.
        ta.selection = saved_selection

        # Trim from the top if we've exceeded the cap
        trimmed = 0
        if self._log_line_count > LOG_MAX_LINES:
            trim = self._log_line_count - (LOG_MAX_LINES - LOG_TRIM_BATCH)
            actual_lines = ta.document.line_count
            trim = min(trim, actual_lines - 1)
            if trim > 0:
                ta.delete((0, 0), (trim, 0), maintain_selection_offset=True)
                trimmed = trim
                self._log_line_count = ta.document.line_count

        if at_bottom:
            ta.scroll_end(animate=False)
        else:
            # Force-restore the viewport to where the user was reading.
            restored = max(0, saved_scroll_y - trimmed)
            ta.scroll_to(y=restored, animate=False)

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
            self.query_one("#log-header-right", Static).update(
                f"{follow_label}  {self._log_line_count:,} lines"
            )
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
            ta = self.query_one("#log-output", TextArea)
            if ta.max_scroll_y <= 0:
                return
            at_bottom = ta.scroll_y >= (ta.max_scroll_y - 2)
            if at_bottom != self._log_follow:
                self._log_follow = at_bottom
                self._update_log_header()
        except Exception:
            pass

    def _infer_worker_from_log(self, source: str, text: str) -> Optional[str]:
        source_upper = source.upper()
        direct = WORKER_TAG_TO_NAME.get(source_upper)
        if direct:
            return direct
        if source_upper != "BACKEND":
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
        line = text.strip()
        if line.startswith("[") and "] " in line:
            line = line.split("] ", 1)[1]
        line = " ".join(line.split())
        if len(line) > WORKER_MINI_LOG_WIDTH:
            line = line[: WORKER_MINI_LOG_WIDTH - 3].rstrip() + "..."
        return line

    # ---- Button / Select handlers ----
    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id

        # Source filter buttons
        if btn_id in ("src-all", "src-backend", "src-frontend"):
            self._source_filter = btn_id.replace("src-", "")
            for sid in ("src-all", "src-backend", "src-frontend"):
                try:
                    btn = self.query_one(f"#{sid}", Button)
                    btn.variant = "primary" if sid == btn_id else "default"
                except Exception:
                    pass
            self._rebuild_log_view()

        elif btn_id == "log-clear-btn":
            self._log_entries.clear()
            try:
                ta = self.query_one("#log-output", TextArea)
                end = ta.document.end
                if end != (0, 0):
                    ta.delete((0, 0), end)
            except Exception:
                pass
            self._log_line_count = 0
            self._update_log_header()
            self.notify("Logs cleared", timeout=2)

        elif btn_id == "log-copy-btn":
            self._do_copy()

        elif btn_id == "log-bottom-btn":
            try:
                ta = self.query_one("#log-output", TextArea)
                ta.scroll_end(animate=False)
                self._log_follow = True
                self._update_log_header()
            except Exception:
                pass

        elif btn_id == "restart-btn":
            self._restart_services()

        elif btn_id == "update-btn":
            self._update_and_restart()

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
                str(PROJECT_ROOT / "setup.ps1"),
            ]
        else:
            setup_cmd = ["bash", str(PROJECT_ROOT / "setup.sh")]

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

    @on(Select.Changed, "#log-level-select")
    def _on_level_changed(self, event: Select.Changed) -> None:
        self._level_filter = str(event.value)
        self._rebuild_log_view()

    def _spawn_worker_process(
        self,
        venv_python: Path,
        env: dict[str, str],
        module_name: str,
        tag: str,
        label: str,
    ) -> Optional[subprocess.Popen]:
        self._enqueue_log(
            f">>> Starting {label} worker...", source="BACKEND", level="INFO"
        )
        try:
            proc = subprocess.Popen(
                [
                    str(venv_python),
                    "-m",
                    module_name,
                ],
                cwd=str(BACKEND_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
            )
            worker_stream = threading.Thread(
                target=self._stream_output,
                args=(proc, tag),
                daemon=True,
                name=f"{tag.lower()}-stream",
            )
            worker_stream.start()
            return proc
        except Exception as exc:
            self._enqueue_log(
                f"{label.capitalize()} worker failed to start (non-fatal): {exc}",
                source="BACKEND",
                level="WARNING",
            )
            return None

    # ---- Start backend & frontend as subprocesses ----
    @work(thread=True)
    def _start_services(self) -> None:
        self._log_activity("[bold cyan]Starting services...[/]")
        self.call_from_thread(self._reset_worker_telemetry)

        # Kill stale processes
        kill_port(BACKEND_PORT)
        kill_port(FRONTEND_PORT)

        # Activate venv and start backend
        if sys.platform == "win32":
            venv_python = BACKEND_DIR / "venv" / "Scripts" / "python.exe"
        else:
            venv_python = BACKEND_DIR / "venv" / "bin" / "python"
        if not venv_python.exists():
            setup_cmd = ".\\setup.ps1" if sys.platform == "win32" else "./setup.sh"
            self._enqueue_log(
                f"ERROR: Virtual environment not found. Run {setup_cmd} first.",
                source="BACKEND",
                level="ERROR",
            )
            return

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Add venv bin to PATH so uvicorn is found
        if sys.platform == "win32":
            venv_bin = str(BACKEND_DIR / "venv" / "Scripts")
        else:
            venv_bin = str(BACKEND_DIR / "venv" / "bin")
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
        env["VIRTUAL_ENV"] = str(BACKEND_DIR / "venv")
        # Default to INFO; the TUI level filter can show DEBUG if needed
        env.setdefault("LOG_LEVEL", "INFO")
        # Keep native ML/linear algebra threading conservative for stability.
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")
        env.setdefault("NEWS_FAISS_THREADS", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("EMBEDDING_DEVICE", "cpu")

        worker_specs: list[tuple[str, str, str, str]] = [
            ("scanner_worker_proc", "workers.scanner_worker", "SCANNER", "scanner"),
            ("discovery_worker_proc", "workers.discovery_worker", "DISCOVERY", "discovery"),
            ("weather_worker_proc", "workers.weather_worker", "WEATHER", "weather"),
            ("news_worker_proc", "workers.news_worker", "NEWS", "news"),
            ("crypto_worker_proc", "workers.crypto_worker", "CRYPTO", "crypto"),
            ("tracked_worker_proc", "workers.tracked_traders_worker", "TRACKED", "tracked-traders"),
            ("autotrader_worker_proc", "workers.autotrader_worker", "AUTOTRADER", "autotrader"),
            ("world_intel_worker_proc", "workers.world_intelligence_worker", "WORLDINTEL", "world-intelligence"),
        ]
        for attr, module_name, tag, label in worker_specs:
            setattr(
                self,
                attr,
                self._spawn_worker_process(
                    venv_python,
                    env,
                    module_name,
                    tag,
                    label,
                ),
            )

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
        except Exception as e:
            self._enqueue_log(
                f"FATAL: Failed to start backend: {e}",
                source="BACKEND",
                level="ERROR",
            )
            return

        # Stream backend output in a thread
        self._stream_output(self.backend_proc, "BACKEND")

    @work(thread=True)
    def _start_frontend(self) -> None:
        """Start frontend after backend is healthy."""
        env = os.environ.copy()
        env["BROWSER"] = "none"  # Don't auto-open browser
        env["FORCE_COLOR"] = "0"

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
        except Exception as e:
            self._enqueue_log(
                f"FATAL: Failed to start frontend: {e}",
                source="FRONTEND",
                level="ERROR",
            )
            return

        self._stream_output(self.frontend_proc, "FRONTEND")

    def _stream_output(self, proc: subprocess.Popen, tag: str) -> None:
        """Read process stdout line-by-line and enqueue for batched display."""
        try:
            for raw_line in iter(proc.stdout.readline, b""):
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
                if tag == "BACKEND" and not self._frontend_starting:
                    if "Application startup complete" in line or "Uvicorn running" in line:
                        self._frontend_starting = True
                        self._log_activity("[bold green]Backend is ready![/]")
                        self._start_frontend()
        except Exception:
            pass
        finally:
            if not self._shutting_down:
                self._enqueue_log(f"[{tag}] Process exited", source=tag, level="INFO")

    # ---- Activity hooks (reserved for future system feed) ----
    def _log_activity(self, text: str) -> None:
        """Thread-safe activity hook."""
        self.call_from_thread(self._do_log_activity, text)

    def _do_log_activity(self, text: str) -> None:
        _ = text

    # ---- Periodic health polling ----
    def _poll_health(self) -> None:
        """Set up a periodic timer to poll /health/detailed."""
        self.set_interval(3.0, self._fetch_health)

    @work(thread=True)
    def _fetch_health(self) -> None:
        """Fetch health data from backend API."""
        import urllib.request
        import urllib.error

        try:
            req = urllib.request.Request(HEALTH_URL, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                self.call_from_thread(self._apply_health, data)
        except (urllib.error.URLError, Exception):
            self.call_from_thread(self._apply_health_offline)

    def _apply_health(self, data: dict) -> None:
        """Update all dashboard widgets from health data."""
        self.backend_healthy = True
        self.health_data = data
        self.health_poll_count += 1
        services = data.get("services", {})
        workers = data.get("workers")
        if workers is None:
            workers = services.get("workers", {})

        # --- Platform status ---
        ws = services.get("ws_feeds", {})

        self._update_platform_item("svc-backend", "BACKEND", True)
        frontend_alive = self.frontend_proc is not None and self.frontend_proc.poll() is None
        self._update_platform_item("svc-frontend", "FRONTEND", frontend_alive)
        ws_healthy = ws.get("healthy", False) if isinstance(ws, dict) else False
        self._update_platform_item("svc-wsfeeds", "WS FEEDS", ws_healthy)

        # --- Worker command center ---
        self._update_worker_panels(workers, services)

        # --- Runtime metrics ---
        self._update_runtime_metrics()

    def _apply_health_offline(self) -> None:
        """Mark backend as offline; check worker/frontend processes directly."""
        self.backend_healthy = False
        self._update_platform_item("svc-backend", "BACKEND", False)
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
        running = False
        if worker_name == "scanner":
            running = bool(services.get("scanner", {}).get("running", False))
        elif worker_name == "discovery":
            running = bool(services.get("wallet_discovery", {}).get("running", False))
        elif worker_name == "weather":
            running = bool(
                self.weather_worker_proc is not None and self.weather_worker_proc.poll() is None
            )
        elif worker_name == "news":
            running = bool(services.get("news_workflow", {}).get("running", False))
        elif worker_name == "autotrader":
            running = bool(services.get("auto_trader", {}).get("running", False))
        elif worker_name == "crypto":
            running = bool(
                self.crypto_worker_proc is not None and self.crypto_worker_proc.poll() is None
            )
        elif worker_name == "tracked_traders":
            running = bool(
                self.tracked_worker_proc is not None and self.tracked_worker_proc.poll() is None
            )
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
        meta = self._format_worker_meta(snapshot)
        lines = list(self._worker_logs.get(worker_name, []))
        try:
            panel = self.query_one(f"#worker-{worker_name}", WorkerPanel)
            panel.update_state(status, status_class, meta)
            panel.update_logs(lines)
        except Exception:
            pass

    def _set_workers_offline(self) -> None:
        for worker_name, _worker_label in WORKER_STATUS_ORDER:
            self._worker_state_cache[worker_name] = {}
            self._render_worker_panel(worker_name)

    # Map worker names to their Popen attribute on self
    _WORKER_PROC_ATTR: dict[str, str] = {
        "scanner": "scanner_worker_proc",
        "discovery": "discovery_worker_proc",
        "weather": "weather_worker_proc",
        "news": "news_worker_proc",
        "crypto": "crypto_worker_proc",
        "tracked_traders": "tracked_worker_proc",
        "autotrader": "autotrader_worker_proc",
    }

    def _update_workers_from_processes(self) -> None:
        """Update worker panels based on whether their OS process is still alive.

        This is used when the backend health endpoint is unreachable so we
        cannot get rich telemetry, but the worker subprocesses may still be
        running fine (they are independent of the API server).
        """
        for worker_name, _worker_label in WORKER_STATUS_ORDER:
            attr = self._WORKER_PROC_ATTR.get(worker_name)
            proc: Optional[subprocess.Popen] = getattr(self, attr, None) if attr else None
            alive = proc is not None and proc.poll() is None
            snapshot: dict = {"running": alive, "enabled": True}
            self._worker_state_cache[worker_name] = snapshot
            self._render_worker_panel(worker_name)

    def _resolve_worker_state(self, snapshot: dict) -> tuple[str, str]:
        if not snapshot:
            return ("OFFLINE", "status-off")
        if bool(snapshot.get("running", False)):
            return ("RUNNING", "status-on")
        enabled = snapshot.get("enabled")
        if enabled is False:
            return ("PAUSED", "status-warn")
        return ("IDLE", "status-idle")

    def _format_worker_meta(self, snapshot: dict) -> str:
        if not snapshot:
            return "No telemetry yet"
        parts: list[str] = []

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

    def action_quit(self) -> None:
        self._shutting_down = True
        self.notify("Shutting down...", severity="warning", timeout=10)
        self._kill_children()
        self.exit()

    def _kill_children(self) -> None:
        """Kill child processes and close their pipes to unblock reader threads."""
        procs = [
            p
            for p in (
                self.scanner_worker_proc,
                self.discovery_worker_proc,
                self.weather_worker_proc,
                self.news_worker_proc,
                self.crypto_worker_proc,
                self.tracked_worker_proc,
                self.autotrader_worker_proc,
                self.backend_proc,
                self.frontend_proc,
            )
            if p and p.poll() is None
        ]
        # Kill all child processes (non-blocking).
        # On Windows, proc.kill() only kills the direct child; use taskkill /T
        # to terminate the entire process tree (npm → node, etc.).
        for proc in procs:
            try:
                if sys.platform == "win32":
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        capture_output=True,
                        timeout=5,
                    )
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    # On Windows Python 3.10+ the default ProactorEventLoop can crash during
    # asyncio runner shutdown.  SelectorEventLoop avoids this and is all the
    # TUI needs (no subprocess pipes managed via IOCP).
    if sys.platform == "win32":
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Verify venv exists
    venv_dir = BACKEND_DIR / "venv"
    if not venv_dir.exists():
        setup_cmd = ".\\setup.ps1" if sys.platform == "win32" else "./setup.sh"
        print(f"Setup not complete. Run {setup_cmd} first.")
        sys.exit(1)

    app = HomerunApp()
    app.run()

    # Force-exit to avoid hanging on background thread joins.
    # Textual worker threads (subprocess readers) may still be blocked
    # on I/O; Python's atexit handler would wait for them indefinitely.
    os._exit(0)


if __name__ == "__main__":
    main()
