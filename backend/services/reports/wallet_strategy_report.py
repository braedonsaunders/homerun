"""Executive PDF report for a finished strategy reverse-engineer job.

Pipeline:  Jinja2 HTML template  →  WeasyPrint  →  bytes (application/pdf).

WeasyPrint is the cleanest "HTML+CSS to PDF" library that works
identically on Windows, macOS, Linux, and inside our Docker image.  It
has a mandatory native dependency on Pango/Cairo/GdkPixbuf:

  * Linux (Docker / dev):  ``apt-get install libpango-1.0-0 libpangoft2-1.0-0
                            libharfbuzz0b libfreetype6 libffi7
                            libgdk-pixbuf-2.0-0`` — already added to the
                            backend Dockerfile.
  * macOS:                  ``brew install pango`` — handled in
                            scripts/launchers/install-deps.sh.
  * Windows:                GTK runtime (MSYS2 mingw64).  The launcher
                            ``scripts/launchers/Homerun.bat`` warns and
                            opens the install instructions when WeasyPrint
                            init fails.

The report DOES NOT crash the API when WeasyPrint cannot be imported —
the route catches :class:`ReportRenderError` and returns a 503 with a
hint pointing the operator at the platform-specific install steps.
"""
from __future__ import annotations

import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from utils.utcnow import utcnow

logger = logging.getLogger(__name__)


_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_TEMPLATE_NAME = "wallet_strategy_report.html.j2"
_ANALYTICAL_TEMPLATE_NAME = "wallet_strategy_analytical_report.html.j2"

# Common locations for the GTK 3 runtime on Windows.  Checked in order;
# the first existing one is added to the DLL search path so WeasyPrint's
# ctypes loader can find libgobject/libpango/libcairo without the user
# having to edit PATH manually.
_WINDOWS_GTK_BIN_CANDIDATES = (
    r"C:\Program Files\GTK3-Runtime Win64\bin",
    r"C:\Program Files (x86)\GTK3-Runtime Win64\bin",
    r"C:\msys64\mingw64\bin",
)


class ReportRenderError(RuntimeError):
    """Raised when WeasyPrint can't be loaded or rendering fails."""


def _ensure_windows_gtk_runtime_on_path() -> None:
    """Add the GTK runtime's bin/ directory to the DLL search path.

    No-op on non-Windows.  Idempotent — safe to call repeatedly.
    Silent when no candidate path exists; the subsequent WeasyPrint
    import will then fail with a clear ``ReportRenderError`` pointing
    at the install instructions.
    """
    import os
    for candidate in _WINDOWS_GTK_BIN_CANDIDATES:
        if os.path.isdir(candidate):
            try:
                os.add_dll_directory(candidate)
            except (AttributeError, OSError):
                pass
            current_path = os.environ.get("PATH", "")
            if candidate not in current_path:
                os.environ["PATH"] = candidate + os.pathsep + current_path
            return


def render_wallet_strategy_report(
    *,
    job: Any,
    iterations: list[Any],
) -> bytes:
    """Render the executive PDF and return its bytes.

    ``job`` is a ``StrategyReverseEngineerJob`` row; ``iterations`` is
    the ordered list of ``StrategyReverseEngineerIteration`` rows.
    """
    # Windows: WeasyPrint relies on the GTK 3 runtime DLLs
    # (libgobject, libpango, libcairo, etc.).  Tobias Schönberg's
    # installer drops them at ``C:\Program Files\GTK3-Runtime Win64\bin``
    # but Python won't find them unless that directory is on the DLL
    # search path.  Add it pre-import so a fresh ``winget install``
    # produces a working PDF endpoint without the operator setting PATH.
    if platform.system() == "Windows":
        _ensure_windows_gtk_runtime_on_path()

    try:
        # Defer the heavy import until first use — keeps API cold-start
        # cheap and avoids a hard dependency for installs that never
        # generate a report.
        from weasyprint import HTML  # type: ignore[import-not-found]
    except (ImportError, OSError) as exc:
        raise ReportRenderError(
            f"PDF rendering unavailable: {exc}.\n\n"
            "Install WeasyPrint and its native deps:\n"
            f"  • Detected OS: {platform.system()}\n"
            "  • Linux/Docker: apt-get install libpango-1.0-0 libpangoft2-1.0-0 "
            "libharfbuzz0b libfreetype6 libffi-dev libgdk-pixbuf-2.0-0\n"
            "  • macOS:        brew install pango cairo libffi gdk-pixbuf\n"
            "  • Windows:      winget install --id tschoonj.GTKForWindows\n"
            "Then `pip install weasyprint`."
        ) from exc

    env = _build_jinja_env()
    template = env.get_template(_TEMPLATE_NAME)
    html_text = template.render(
        job=_job_view(job),
        iterations=[_iteration_view(it) for it in iterations],
        generated_at=utcnow(),
    )

    try:
        pdf_bytes = HTML(string=html_text, base_url=str(_TEMPLATE_DIR)).write_pdf()
    except Exception as exc:
        raise ReportRenderError(f"WeasyPrint render failed: {exc}") from exc
    if not pdf_bytes:
        raise ReportRenderError("WeasyPrint produced an empty PDF")
    return pdf_bytes


# ---------------------------------------------------------------------------
# View-model adaptors — keep the Jinja template free of
# database-row-shaped attribute access; the PDF source is a small
# stable dict.
# ---------------------------------------------------------------------------


def _job_view(job: Any) -> dict[str, Any]:
    return {
        "id": job.id,
        "wallet_address": job.wallet_address,
        "label": job.label or "(unlabeled)",
        "status": job.status,
        "best_score": (
            float(job.best_score) if job.best_score is not None else None
        ),
        "max_iterations": int(job.max_iterations or 0),
        "current_iteration": int(job.current_iteration or 0),
        "best_strategy_class": job.best_strategy_class,
        "best_strategy_code": job.best_strategy_code,
        "best_backtest_run_id": job.best_backtest_run_id,
        "wallet_trade_count": int(job.wallet_trade_count or 0),
        "wallet_window_start": _iso(job.wallet_window_start),
        "wallet_window_end": _iso(job.wallet_window_end),
        "data_source_kind": job.data_source_kind,
        "provider_dataset_ids": list(job.provider_dataset_ids_json or []),
        "recording_session_ids": list(job.recording_session_ids_json or []),
        "wallet_profile": job.wallet_profile_json or {},
        "llm_model": job.llm_model,
        "total_input_tokens": int(job.total_input_tokens or 0),
        "total_output_tokens": int(job.total_output_tokens or 0),
        "total_cost_usd": float(job.total_cost_usd or 0.0),
        "promoted_strategy_id": job.promoted_strategy_id,
        "created_at": _iso(job.created_at),
        "started_at": _iso(job.started_at),
        "finished_at": _iso(job.finished_at),
    }


def _iteration_view(it: Any) -> dict[str, Any]:
    return {
        "id": it.id,
        "iteration": int(it.iteration or 0),
        "status": it.status,
        "strategy_class": it.strategy_class,
        "score": float(it.score) if it.score is not None else None,
        "score_breakdown": it.score_breakdown_json or {},
        "divergence_summary": it.divergence_summary,
        "notes": it.notes,
        "duration_ms": float(it.duration_ms or 0.0),
        "error": it.error,
        "completed_at": _iso(it.completed_at),
    }


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return (value if value.tzinfo else value.replace(tzinfo=timezone.utc)).isoformat()
    return str(value)


def _pretty_json(value: Any) -> str:
    try:
        return json.dumps(value, indent=2, default=str)
    except Exception:
        return str(value)


def _pct_filter(value: Any, decimals: int = 1) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except Exception:
        return str(value)


def _money_filter(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"${float(value):,.2f}"
    except Exception:
        return str(value)


def _humanize_dt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        try:
            text = value.strip()
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            value = datetime.fromisoformat(text)
        except Exception:
            return value
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M UTC")
    return str(value)


def _money_signed_filter(value: Any) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except Exception:
        return str(value)
    sign = "+" if v >= 0 else "−"
    return f"{sign}${abs(v):,.2f}"


def _money_compact_filter(value: Any) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except Exception:
        return str(value)
    abs_v = abs(v)
    if abs_v >= 1_000_000:
        return f"${v / 1_000_000:.2f}M"
    if abs_v >= 1_000:
        return f"${v / 1_000:.1f}K"
    return f"${v:.2f}"


def _pct_signed_filter(value: Any, decimals: int = 2) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except Exception:
        return str(value)
    sign = "+" if v >= 0 else "−"
    return f"{sign}{abs(v) * 100:.{decimals}f}%"


def _pp_signed_filter(value: Any, decimals: int = 2) -> str:
    """Percentage points (already in 0-1 range, display as +X.XXpp)."""
    if value is None:
        return "—"
    try:
        v = float(value)
    except Exception:
        return str(value)
    sign = "+" if v >= 0 else "−"
    return f"{sign}{abs(v) * 100:.{decimals}f}pp"


def _int_thousands_filter(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{int(value):,}"
    except Exception:
        return str(value)


def _build_jinja_env():
    """Centralized Jinja env so the legacy + analytical templates share filters."""
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "htm", "xml", "j2"]),
    )
    env.filters["pretty_json"] = _pretty_json
    env.filters["pct"] = _pct_filter
    env.filters["money"] = _money_filter
    env.filters["humanize_dt"] = _humanize_dt
    env.filters["money_signed"] = _money_signed_filter
    env.filters["money_compact"] = _money_compact_filter
    env.filters["pct_signed"] = _pct_signed_filter
    env.filters["pp_signed"] = _pp_signed_filter
    env.filters["int_thousands"] = _int_thousands_filter
    return env


def render_analytical_report(
    *,
    analytics: Any,
    sections: Any,
    spotlight: Optional[dict[str, Any]] = None,
) -> bytes:
    """Render the analytical (multi-section) wallet report PDF.

    Inputs:
      analytics: WalletAnalytics dataclass (services.strategy_reverse_engineer.wallet_analytics)
      sections:  ReportSections dataclass (services.strategy_reverse_engineer.report_writer)
      spotlight: optional spotlight market dict (render_spotlight_market output)
    """
    if platform.system() == "Windows":
        _ensure_windows_gtk_runtime_on_path()

    try:
        from weasyprint import HTML  # type: ignore[import-not-found]
    except (ImportError, OSError) as exc:
        raise ReportRenderError(
            f"PDF rendering unavailable: {exc}.  Install GTK runtime + weasyprint."
        ) from exc

    env = _build_jinja_env()
    template = env.get_template(_ANALYTICAL_TEMPLATE_NAME)
    html_text = template.render(
        analytics=analytics,
        sections=sections,
        spotlight=spotlight,
        generated_at=utcnow(),
    )

    try:
        pdf_bytes = HTML(string=html_text, base_url=str(_TEMPLATE_DIR)).write_pdf()
    except Exception as exc:
        raise ReportRenderError(f"WeasyPrint render failed: {exc}") from exc
    if not pdf_bytes:
        raise ReportRenderError("WeasyPrint produced an empty PDF")
    return pdf_bytes
