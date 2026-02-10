#!/usr/bin/env python3
"""Homerun TUI - Beautiful terminal interface for the Homerun trading platform."""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.reactive import reactive
from textual.widgets import (
    Footer,
    Header,
    Label,
    Static,
    TabbedContent,
    TabPane,
    RichLog,
    Rule,
    Sparkline,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BACKEND_PORT = 8000
FRONTEND_PORT = 3000
HEALTH_URL = f"http://localhost:{BACKEND_PORT}/health/detailed"
PROJECT_ROOT = Path(__file__).parent.resolve()
BACKEND_DIR = PROJECT_ROOT / "backend"

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
    background: $surface;
}

#logo {
    color: #00ff88;
    text-style: bold;
    text-align: center;
    padding: 1 0;
}

#subtitle {
    color: #888888;
    text-align: center;
    padding: 0 0 1 0;
}

/* ---- Service status bar ---- */
#status-bar {
    layout: horizontal;
    height: 3;
    padding: 0 2;
    background: $boost;
    margin: 0 1 1 1;
}

.status-item {
    width: 1fr;
    content-align: center middle;
}

.status-label {
    text-style: bold;
}

.status-on {
    color: #00ff88;
}

.status-off {
    color: #ff4444;
}

.status-warn {
    color: #ffaa00;
}

/* ---- URL bar ---- */
#url-bar {
    layout: horizontal;
    height: 3;
    padding: 0 2;
    background: $boost;
    margin: 0 1 1 1;
}

.url-item {
    width: 1fr;
    content-align: center middle;
    color: #66ccff;
}

/* ---- Stats grid ---- */
#stats-grid {
    layout: grid;
    grid-size: 4 2;
    grid-gutter: 1;
    padding: 0 1;
    margin: 0 1;
    height: auto;
}

.stat-card {
    height: 5;
    background: $boost;
    border: round $primary-background;
    padding: 0 1;
    content-align: center middle;
}

.stat-value {
    text-style: bold;
    color: #00ff88;
    text-align: center;
    width: 100%;
}

.stat-title {
    color: #888888;
    text-align: center;
    width: 100%;
}

/* ---- Config section ---- */
#config-section {
    layout: grid;
    grid-size: 3;
    grid-gutter: 1;
    padding: 0 1;
    margin: 1 1;
    height: auto;
}

.config-card {
    height: 5;
    background: $boost;
    border: round $primary-background;
    padding: 0 1;
    content-align: center middle;
}

.config-value {
    text-style: bold;
    color: #ffaa00;
    text-align: center;
    width: 100%;
}

.config-label {
    color: #888888;
    text-align: center;
    width: 100%;
}

/* ---- Sparkline section ---- */
#sparkline-section {
    height: 5;
    margin: 0 2;
    padding: 0 1;
}

#sparkline-label {
    color: #888888;
    text-align: left;
    padding: 0 0;
}

#opp-sparkline {
    height: 3;
    color: #00ff88;
}

/* ---- Activity feed ---- */
#activity-section {
    margin: 0 2;
    height: 8;
    background: $boost;
    border: round $primary-background;
    padding: 0 1;
}

#activity-title {
    color: #888888;
    text-style: bold;
    padding: 0 0;
}

/* ---- Uptime ---- */
#uptime-bar {
    height: 1;
    margin: 0 2 1 2;
    color: #666666;
    text-align: center;
}

/* ---- Logs pane ---- */
#log-pane {
    height: 1fr;
}

#log-controls {
    layout: horizontal;
    height: 3;
    padding: 0 1;
    background: $boost;
    margin: 0 1 1 1;
}

.log-filter-btn {
    margin: 0 1;
    min-width: 12;
}

#log-output {
    margin: 0 1;
    border: round $primary-background;
    scrollbar-size: 1 1;
}

/* ---- Tabs ---- */
TabbedContent {
    height: 1fr;
}

TabPane {
    padding: 0;
}
"""


# ---------------------------------------------------------------------------
# Helper to kill processes on a port
# ---------------------------------------------------------------------------
def kill_port(port: int) -> None:
    """Kill any process listening on the given port."""
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


# ---------------------------------------------------------------------------
# Stat card widget
# ---------------------------------------------------------------------------
class StatCard(Static):
    """A small card showing a single metric."""

    def __init__(self, title: str, value: str = "--", card_id: str = "") -> None:
        super().__init__(id=card_id)
        self._title = title
        self._value = value

    def compose(self) -> ComposeResult:
        yield Label(self._value, classes="stat-value", id=f"{self.id}-val")
        yield Label(self._title, classes="stat-title")

    def update_value(self, value: str) -> None:
        self._value = value
        try:
            self.query_one(f"#{self.id}-val", Label).update(value)
        except Exception:
            pass


class ConfigCard(Static):
    """A config display card."""

    def __init__(self, label: str, value: str = "--", card_id: str = "") -> None:
        super().__init__(id=card_id)
        self._label = label
        self._value = value

    def compose(self) -> ComposeResult:
        yield Label(self._value, classes="config-value", id=f"{self.id}-val")
        yield Label(self._label, classes="config-label")

    def update_value(self, value: str) -> None:
        self._value = value
        try:
            self.query_one(f"#{self.id}-val", Label).update(value)
        except Exception:
            pass


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
        Binding("h", "show_tab('home')", "Home", show=True),
        Binding("l", "show_tab('logs')", "Logs", show=True),
        Binding("d", "toggle_dark", "Dark/Light"),
    ]

    # Process handles
    backend_proc: subprocess.Popen | None = None
    frontend_proc: subprocess.Popen | None = None

    # State
    start_time: float = 0.0
    opp_history: list[float] = []
    backend_healthy: bool = False
    health_data: dict = {}

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
        yield Static(LOGO, id="logo")
        yield Static(
            "Autonomous Prediction Market Trading Platform",
            id="subtitle",
        )

        # Service status bar
        with Horizontal(id="status-bar"):
            yield Static(
                "[@click=app.noop]BACKEND[/]  [bold red]OFF[/]",
                id="svc-backend",
                classes="status-item",
            )
            yield Static(
                "[@click=app.noop]FRONTEND[/]  [bold red]OFF[/]",
                id="svc-frontend",
                classes="status-item",
            )
            yield Static(
                "[@click=app.noop]SCANNER[/]  [bold red]OFF[/]",
                id="svc-scanner",
                classes="status-item",
            )
            yield Static(
                "[@click=app.noop]AUTO TRADER[/]  [bold red]OFF[/]",
                id="svc-autotrader",
                classes="status-item",
            )
            yield Static(
                "[@click=app.noop]WS FEEDS[/]  [bold red]OFF[/]",
                id="svc-wsfeeds",
                classes="status-item",
            )

        # URLs
        with Horizontal(id="url-bar"):
            yield Static(
                f"Dashboard  http://localhost:{FRONTEND_PORT}",
                classes="url-item",
            )
            yield Static(
                f"API  http://localhost:{BACKEND_PORT}",
                classes="url-item",
            )
            yield Static(
                f"Docs  http://localhost:{BACKEND_PORT}/docs",
                classes="url-item",
            )

        # Stats grid
        with Container(id="stats-grid"):
            yield StatCard("Opportunities", "--", card_id="stat-opps")
            yield StatCard("Tracked Wallets", "--", card_id="stat-wallets")
            yield StatCard("Auto Trades", "--", card_id="stat-trades")
            yield StatCard("Total Profit", "--", card_id="stat-profit")
            yield StatCard("Copy Configs", "--", card_id="stat-copy")
            yield StatCard("Trader Mode", "--", card_id="stat-mode")
            yield StatCard("Wallets Found", "--", card_id="stat-discovered")
            yield StatCard("Rate Limits", "--", card_id="stat-ratelimits")

        # Sparkline: opportunity history
        yield Static("  Opportunities (last 60 polls)", id="sparkline-label")
        yield Sparkline([], id="opp-sparkline")

        # Config section
        with Container(id="config-section"):
            yield ConfigCard("Scan Interval", "--", card_id="cfg-interval")
            yield ConfigCard("Min Profit", "--", card_id="cfg-profit")
            yield ConfigCard("Max Markets", "--", card_id="cfg-markets")

        # Activity feed
        with Vertical(id="activity-section"):
            yield Static("  Recent Activity", id="activity-title")
            yield RichLog(id="activity-log", markup=True, max_lines=50)

        # Uptime
        yield Static("Uptime: starting...", id="uptime-bar")

    # ---- Logs tab layout ----
    def _compose_logs(self) -> ComposeResult:
        with Vertical(id="log-pane"):
            with Horizontal(id="log-controls"):
                yield Static(
                    "[bold]VERBOSE LOGS[/]  [dim]Backend + Frontend combined[/]",
                )
            yield RichLog(
                id="log-output",
                highlight=True,
                markup=True,
                max_lines=10000,
                wrap=True,
            )

    # ---- Lifecycle ----
    def on_mount(self) -> None:
        self.start_time = time.time()
        self.opp_history = []
        self._start_services()
        self._poll_health()
        self._update_uptime()

    def action_show_tab(self, tab: str) -> None:
        self.query_one(TabbedContent).active = tab

    # ---- Start backend & frontend as subprocesses ----
    @work(thread=True)
    def _start_services(self) -> None:
        log = self._get_log_widget
        self._log_activity("[bold cyan]Starting services...[/]")

        # Kill stale processes
        kill_port(BACKEND_PORT)
        kill_port(FRONTEND_PORT)

        # Activate venv and start backend
        venv_python = BACKEND_DIR / "venv" / "bin" / "python"
        if not venv_python.exists():
            self._write_log(
                "[bold red]ERROR:[/] Virtual environment not found. Run ./setup.sh first."
            )
            return

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Add venv bin to PATH so uvicorn is found
        venv_bin = str(BACKEND_DIR / "venv" / "bin")
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
        env["VIRTUAL_ENV"] = str(BACKEND_DIR / "venv")
        # Ensure LOG_LEVEL is DEBUG for verbose logs
        env["LOG_LEVEL"] = "DEBUG"

        self._write_log("[bold cyan]>>> Starting backend (uvicorn)...[/]")
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
                    "debug",
                ],
                cwd=str(BACKEND_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                bufsize=1,
            )
        except Exception as e:
            self._write_log(f"[bold red]Failed to start backend: {e}[/]")
            return

        # Stream backend output in a thread
        self._stream_output(self.backend_proc, "BACKEND")

    @work(thread=True)
    def _start_frontend(self) -> None:
        """Start frontend after backend is healthy."""
        env = os.environ.copy()
        env["BROWSER"] = "none"  # Don't auto-open browser
        env["FORCE_COLOR"] = "0"

        self._write_log("[bold cyan]>>> Starting frontend (npm run dev)...[/]")
        self._log_activity("[cyan]Frontend starting...[/]")
        try:
            self.frontend_proc = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=str(PROJECT_ROOT / "frontend"),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                bufsize=1,
            )
        except Exception as e:
            self._write_log(f"[bold red]Failed to start frontend: {e}[/]")
            return

        self._stream_output(self.frontend_proc, "FRONTEND")

    def _stream_output(self, proc: subprocess.Popen, tag: str) -> None:
        """Read process stdout line-by-line and write to log."""
        color = "cyan" if tag == "BACKEND" else "magenta"
        try:
            for raw_line in iter(proc.stdout.readline, b""):
                if proc.poll() is not None and not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue

                styled = self._style_log_line(line, tag, color)
                self._write_log(styled)

                # If backend started, kick off frontend
                if tag == "BACKEND" and not self.frontend_proc:
                    if "Application startup complete" in line or "Uvicorn running" in line:
                        self._log_activity("[bold green]Backend is ready![/]")
                        self._start_frontend()
        except Exception:
            pass
        finally:
            self._write_log(f"[bold {color}][{tag}][/] [dim]Process exited[/]")

    def _style_log_line(self, line: str, tag: str, color: str) -> str:
        """Apply rich markup to a log line based on content."""
        prefix = f"[bold {color}][{tag}][/] "

        # Try to parse JSON log lines from backend
        if line.startswith("{"):
            try:
                data = json.loads(line)
                level = data.get("level", "INFO")
                msg = data.get("message", line)
                logger_name = data.get("logger", "")
                module = data.get("module", "")
                func = data.get("function", "")
                ts = data.get("timestamp", "")
                extra = data.get("data", {})

                # Color by level
                level_colors = {
                    "DEBUG": "dim white",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red bold",
                    "CRITICAL": "red bold reverse",
                }
                lc = level_colors.get(level, "white")

                ts_short = ts[11:19] if len(ts) >= 19 else ts
                parts = f"{prefix}[dim]{ts_short}[/] [{lc}]{level:8s}[/] [bold]{logger_name}[/].[dim]{func}[/] {msg}"
                if extra:
                    extra_str = " ".join(
                        f"[dim]{k}[/]=[italic]{v}[/]" for k, v in extra.items()
                    )
                    parts += f"  {extra_str}"
                return parts
            except (json.JSONDecodeError, KeyError):
                pass

        # Highlight keywords in plain-text lines
        lower = line.lower()
        if "error" in lower or "traceback" in lower or "exception" in lower:
            return f"{prefix}[red]{line}[/]"
        elif "warning" in lower or "warn" in lower:
            return f"{prefix}[yellow]{line}[/]"
        elif "started" in lower or "ready" in lower or "healthy" in lower:
            return f"{prefix}[green]{line}[/]"
        else:
            return f"{prefix}{line}"

    # ---- Log helpers ----
    def _write_log(self, text: str) -> None:
        """Thread-safe write to the log widget."""
        self.call_from_thread(self._do_write_log, text)

    def _do_write_log(self, text: str) -> None:
        try:
            log_widget = self.query_one("#log-output", RichLog)
            log_widget.write(text)
        except Exception:
            pass

    def _log_activity(self, text: str) -> None:
        """Thread-safe write to the activity feed."""
        self.call_from_thread(self._do_log_activity, text)

    def _do_log_activity(self, text: str) -> None:
        try:
            ts = datetime.now().strftime("%H:%M:%S")
            activity = self.query_one("#activity-log", RichLog)
            activity.write(f"[dim]{ts}[/]  {text}")
        except Exception:
            pass

    @property
    def _get_log_widget(self):
        try:
            return self.query_one("#log-output", RichLog)
        except Exception:
            return None

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
        services = data.get("services", {})
        config = data.get("config", {})
        rate_limits = data.get("rate_limits", {})

        # --- Service status bar ---
        def svc_badge(running: bool) -> str:
            return "[bold green]ON[/]" if running else "[bold red]OFF[/]"

        scanner = services.get("scanner", {})
        auto = services.get("auto_trader", {})
        ws = services.get("ws_feeds", {})

        self._update_status("svc-backend", "BACKEND", True)
        # Frontend: we check if the process is alive
        frontend_alive = self.frontend_proc is not None and self.frontend_proc.poll() is None
        self._update_status("svc-frontend", "FRONTEND", frontend_alive)
        self._update_status("svc-scanner", "SCANNER", scanner.get("running", False))
        self._update_status("svc-autotrader", "AUTO TRADER", auto.get("running", False))
        ws_healthy = ws.get("healthy", False) if isinstance(ws, dict) else False
        self._update_status("svc-wsfeeds", "WS FEEDS", ws_healthy)

        # --- Stats ---
        opp_count = scanner.get("opportunities_count", 0)
        self._update_stat("stat-opps", str(opp_count))
        self.opp_history.append(float(opp_count))
        if len(self.opp_history) > 60:
            self.opp_history = self.opp_history[-60:]

        wallets = services.get("wallet_tracker", {})
        self._update_stat("stat-wallets", str(wallets.get("tracked_wallets", 0)))

        auto_stats = auto.get("stats", {})
        if isinstance(auto_stats, dict):
            self._update_stat("stat-trades", str(auto_stats.get("total_trades", 0)))
            profit = auto_stats.get("total_profit", 0)
            self._update_stat("stat-profit", f"${profit:,.2f}" if profit else "$0.00")
        else:
            self._update_stat("stat-trades", "0")
            self._update_stat("stat-profit", "$0.00")

        copy_cfg = services.get("copy_trader", {})
        self._update_stat("stat-copy", str(copy_cfg.get("active_configs", 0)))

        mode = auto.get("mode", "paper")
        mode_colors = {"paper": "cyan", "live": "green", "shadow": "yellow", "mock": "dim"}
        mc = mode_colors.get(mode, "white")
        try:
            w = self.query_one("#stat-mode-val", Label)
            w.update(f"[{mc}]{mode.upper()}[/]")
        except Exception:
            self._update_stat("stat-mode", mode.upper())

        discovery = services.get("wallet_discovery", {})
        self._update_stat(
            "stat-discovered", str(discovery.get("wallets_discovered", 0))
        )

        # Rate limits
        if isinstance(rate_limits, dict):
            remaining = rate_limits.get("remaining", "?")
            limit = rate_limits.get("limit", "?")
            self._update_stat("stat-ratelimits", f"{remaining}/{limit}")
        else:
            self._update_stat("stat-ratelimits", "OK")

        # --- Config ---
        self._update_config("cfg-interval", f"{config.get('scan_interval', '?')}s")
        self._update_config(
            "cfg-profit",
            f"{config.get('min_profit_threshold', 0) * 100:.1f}%"
            if isinstance(config.get("min_profit_threshold"), (int, float))
            else "?",
        )
        self._update_config("cfg-markets", str(config.get("max_markets", "?")))

        # --- Sparkline ---
        try:
            spark = self.query_one("#opp-sparkline", Sparkline)
            spark.data = self.opp_history
        except Exception:
            pass

    def _apply_health_offline(self) -> None:
        """Mark backend as offline."""
        if self.backend_healthy:
            self.backend_healthy = False
            self._update_status("svc-backend", "BACKEND", False)
            self._update_status("svc-scanner", "SCANNER", False)
            self._update_status("svc-autotrader", "AUTO TRADER", False)
            self._update_status("svc-wsfeeds", "WS FEEDS", False)

    def _update_status(self, widget_id: str, label: str, is_on: bool) -> None:
        badge = "[bold green]ON[/]" if is_on else "[bold red]OFF[/]"
        dot = "[green]\u25cf[/]" if is_on else "[red]\u25cf[/]"
        try:
            self.query_one(f"#{widget_id}", Static).update(
                f"{dot} {label}  {badge}"
            )
        except Exception:
            pass

    def _update_stat(self, card_id: str, value: str) -> None:
        try:
            self.query_one(f"#{card_id}", StatCard).update_value(value)
        except Exception:
            pass

    def _update_config(self, card_id: str, value: str) -> None:
        try:
            self.query_one(f"#{card_id}", ConfigCard).update_value(value)
        except Exception:
            pass

    # ---- Uptime ticker ----
    def _update_uptime(self) -> None:
        self.set_interval(1.0, self._tick_uptime)

    def _tick_uptime(self) -> None:
        elapsed = time.time() - self.start_time
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            self.query_one("#uptime-bar", Static).update(
                f"Uptime: {h:02d}:{m:02d}:{s:02d}  |  {ts}  |  Press [bold]h[/]=Home  [bold]l[/]=Logs  [bold]q[/]=Quit"
            )
        except Exception:
            pass

    # ---- Cleanup ----
    def on_unmount(self) -> None:
        self._kill_children()

    def action_quit(self) -> None:
        self._kill_children()
        self.exit()

    def _kill_children(self) -> None:
        for proc in (self.backend_proc, self.frontend_proc):
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        # Clean up ports
        kill_port(BACKEND_PORT)
        kill_port(FRONTEND_PORT)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    # Verify venv exists
    venv_dir = BACKEND_DIR / "venv"
    if not venv_dir.exists():
        print("Setup not complete. Run ./setup.sh first.")
        sys.exit(1)

    app = HomerunApp()
    app.run()


if __name__ == "__main__":
    main()
