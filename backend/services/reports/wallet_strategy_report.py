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


# ---------------------------------------------------------------------------
# Backtest run report — separate template, same WeasyPrint pipeline.
# ---------------------------------------------------------------------------

_BACKTEST_TEMPLATE_NAME = "backtest_run_report.html.j2"


def render_backtest_run_report(
    *,
    run_row: Any,
    result: dict[str, Any],
    triangulation: Optional[dict[str, Any]] = None,
    portfolio_correlation: Optional[dict[str, Any]] = None,
    drift: Optional[dict[str, Any]] = None,
) -> bytes:
    """Render an executive PDF for a single completed backtest run.

    ``run_row`` is the ORM ``BacktestRun`` row (provides id, strategy
    metadata, started_at, status, etc.); ``result`` is the parsed
    ``result_json`` blob (the rich UnifiedBacktestResult shape the
    studio's Inspect tab consumes).

    ``triangulation``, ``portfolio_correlation``, ``drift`` are the
    three live cross-strategy/live-vs-backtest queries that the studio
    fetches separately from the run row.  They're optional — when the
    operator generates the PDF without those fetched, the
    corresponding sections are simply omitted.  When provided, the
    PDF is a 1:1 mirror of the studio Inspect view.
    """
    if platform.system() == "Windows":
        _ensure_windows_gtk_runtime_on_path()

    try:
        from weasyprint import HTML  # type: ignore[import-not-found]
    except (ImportError, OSError) as exc:
        raise ReportRenderError(
            f"PDF rendering unavailable: {exc}.\n\n"
            "Install WeasyPrint + GTK runtime — same as the reverse-engineer report.\n"
            f"  • Detected OS: {platform.system()}\n"
            "  • Linux/Docker: apt-get install libpango-1.0-0 libpangoft2-1.0-0 "
            "libharfbuzz0b libfreetype6 libffi-dev libgdk-pixbuf-2.0-0\n"
            "  • macOS:        brew install pango cairo libffi gdk-pixbuf\n"
            "  • Windows:      winget install --id tschoonj.GTKForWindows\n"
            "Then `pip install weasyprint`."
        ) from exc

    env = _build_jinja_env()
    template = env.get_template(_BACKTEST_TEMPLATE_NAME)
    html_text = template.render(
        run=_backtest_run_view(run_row, result),
        triangulation=_triangulation_view(triangulation),
        portfolio=_portfolio_correlation_view(portfolio_correlation),
        drift=_drift_view(drift),
        generated_at=utcnow(),
    )

    try:
        pdf_bytes = HTML(string=html_text, base_url=str(_TEMPLATE_DIR)).write_pdf()
    except Exception as exc:
        raise ReportRenderError(f"WeasyPrint render failed: {exc}") from exc
    if not pdf_bytes:
        raise ReportRenderError("WeasyPrint produced an empty PDF")
    return pdf_bytes


def _triangulation_view(payload: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Adapt /backtest/triangulation response into a flat view.  The
    studio's Robustness section reads this verbatim; we keep the
    structure simple so the template stays declarative."""
    if not payload or not isinstance(payload, dict):
        return None
    modes = payload.get("modes") or {}
    shadow = modes.get("shadow") or {}
    live = modes.get("live") or {}
    return {
        "shadow_realized_pnl_usd": shadow.get("realized_pnl_usd"),
        "shadow_orders": shadow.get("orders"),
        "shadow_filled": shadow.get("filled"),
        "live_realized_pnl_usd": live.get("realized_pnl_usd"),
        "live_orders": live.get("orders"),
        "live_filled": live.get("filled"),
        "window_days": payload.get("window_days"),
    }


def _portfolio_correlation_view(payload: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Adapt /backtest/portfolio-correlation response.  Returns None
    when fewer than 2 strategies — there's nothing to show."""
    if not payload or not isinstance(payload, dict):
        return None
    strategies = payload.get("strategies") or []
    if len(strategies) < 2:
        return None
    return {
        "strategies": list(strategies),
        "matrix": payload.get("correlation_matrix") or [],
        "summary": payload.get("summary") or {},
        "window_days": payload.get("window_days"),
    }


def _drift_view(payload: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Adapt /backtest/drift response.  Returns None when no
    strategies tracked."""
    if not payload or not isinstance(payload, dict):
        return None
    strategies = payload.get("strategies") or []
    if not strategies:
        return None
    return {
        "strategies": list(strategies),
        "summary": payload.get("summary") or {},
        "window_days": payload.get("window_days"),
    }


def _backtest_run_view(run_row: Any, result: dict[str, Any]) -> dict[str, Any]:
    """Flatten the run row + result blob into a single template-friendly
    dict.  All defensive: missing nested keys collapse to None / 0 / []
    so the template never KeyErrors on partial payloads.  This view is
    the COMPLETE 1:1 mirror of every UI section that's backed by
    result_json — regime, ensemble bands, counterfactuals, data
    quality, fill model, latency, decomposition, empirical constants,
    full partial-fill aggregates, full outcome netting.  Sections
    backed by live cross-strategy queries (triangulation, portfolio
    correlation, drift) are passed through separately to the
    renderer."""
    exec_d = result.get("execution") or {}
    coverage = result.get("data_coverage") or {}
    deflated = result.get("deflated_sharpe") or {}
    partial = result.get("partial_fills") or {}
    outcome = result.get("outcome_netting") or {}
    tom = result.get("trade_order_monte_carlo") or {}
    regime = result.get("regime_breakdown") or {}
    ensemble = result.get("ensemble_band") or []
    counterfactuals = result.get("counterfactuals") or []
    data_quality = result.get("data_quality") or {}
    fill_model = result.get("fill_model") or {}
    latency = result.get("latency") or {}
    decomposition = result.get("decomposition") or {}
    empirical = result.get("empirical_constants") or {}

    def _ci(m: Any) -> dict[str, Any]:
        if not isinstance(m, dict):
            return {"value": None, "ci_low": None, "ci_high": None}
        return {
            "value": m.get("value"),
            "ci_low": m.get("ci_low"),
            "ci_high": m.get("ci_high"),
        }

    initial_cap = float(exec_d.get("initial_capital_usd") or 0.0)
    final_eq = float(exec_d.get("final_equity_usd") or 0.0)
    net_pnl = final_eq - initial_cap

    # Build a downsampled equity curve for the embedded SVG (cap at
    # 240 points so the inline SVG stays compact in the PDF).
    raw_curve = exec_d.get("equity_curve_sample") or []
    if isinstance(raw_curve, list) and len(raw_curve) > 240:
        step = max(1, len(raw_curve) // 240)
        sampled = raw_curve[::step]
        # Always include the true final point (matches the studio fix).
        if sampled and raw_curve and sampled[-1] is not raw_curve[-1]:
            sampled.append(raw_curve[-1])
    else:
        sampled = list(raw_curve or [])

    # Cap ensemble + counterfactual sample lists so the PDF stays
    # compact even when the operator ran with sample_size=64.
    ensemble_capped = ensemble[:16] if isinstance(ensemble, list) else []
    counterfactuals_capped = counterfactuals[:16] if isinstance(counterfactuals, list) else []

    # Hazard ratios (Cox PH coefficients) — top 8 by |log(hr)| so the
    # most influential covariates lead the table.  Mirrors the
    # studio's UI sort.
    coefs = fill_model.get("coefficients") or {}
    hazard_ratios: list[dict[str, Any]] = []
    if isinstance(coefs, dict) and coefs:
        import math
        items = [(name, float(hr)) for name, hr in coefs.items() if hr and hr > 0]
        items.sort(key=lambda kv: abs(math.log(kv[1])) if kv[1] > 0 else 0, reverse=True)
        hazard_ratios = [{"covariate": n, "hr": hr} for n, hr in items[:8]]

    return {
        "id": run_row.id,
        "strategy_slug": run_row.strategy_slug,
        "strategy_name": run_row.strategy_name,
        "status": run_row.status,
        "started_at": _iso(run_row.started_at),
        "completed_at": _iso(run_row.completed_at),
        "total_time_ms": float(run_row.total_time_ms or 0.0),
        "trade_count": int(run_row.trade_count or 0),
        "total_return_pct": float(run_row.total_return_pct or 0.0),
        # ── Headline KPIs ──
        "initial_capital_usd": initial_cap,
        "final_equity_usd": final_eq,
        "net_pnl_usd": net_pnl,
        # ── Risk-adjusted (with bootstrap CIs) ──
        "sharpe": _ci(exec_d.get("sharpe")),
        "sortino": _ci(exec_d.get("sortino")),
        "calmar": _ci(exec_d.get("calmar")),
        "hit_rate": _ci(exec_d.get("hit_rate")),
        "profit_factor": _ci(exec_d.get("profit_factor")),
        "expectancy_usd": _ci(exec_d.get("expectancy_usd")),
        # ── Tail risk ──
        "expected_shortfall_5pct": _ci(exec_d.get("expected_shortfall_5pct")),
        "expected_shortfall_1pct": _ci(exec_d.get("expected_shortfall_1pct")),
        "tail_ratio": _ci(exec_d.get("tail_ratio")),
        "gain_to_pain": _ci(exec_d.get("gain_to_pain")),
        # ── Trade economics + drawdown ──
        "max_drawdown_pct": float(exec_d.get("max_drawdown_pct") or 0.0),
        "max_drawdown_usd": float(exec_d.get("max_drawdown_usd") or 0.0),
        "drawdown_duration_seconds": float(exec_d.get("drawdown_duration_seconds") or 0.0),
        "avg_win_usd": float(exec_d.get("avg_win_usd") or 0.0),
        "avg_loss_usd": float(exec_d.get("avg_loss_usd") or 0.0),
        "fees_paid_usd": float(exec_d.get("fees_paid_usd") or 0.0),
        "fees_per_fill_usd": float(exec_d.get("fees_per_fill_usd") or 0.0),
        "fees_resolution_usd": float(exec_d.get("fees_resolution_usd") or 0.0),
        "total_fills": int(exec_d.get("total_fills") or 0),
        "rejected_orders": int(exec_d.get("rejected_orders") or 0),
        "cancelled_orders": int(exec_d.get("cancelled_orders") or 0),
        "replay_source": exec_d.get("replay_source"),
        "discovery_mode": exec_d.get("discovery_mode"),
        "runtime_error": exec_d.get("runtime_error"),
        # ── Equity curve sample for embedded SVG ──
        "equity_curve_sample": sampled,
        # ── Coverage / data fidelity banner ──
        "fidelity_rating": coverage.get("fidelity_rating"),
        "tokens_with_snapshots": coverage.get("tokens_with_snapshots"),
        "opp_tokens": coverage.get("opp_tokens"),
        "median_snaps_per_token_per_hour": coverage.get("median_snaps_per_token_per_hour"),
        "fidelity_recommendation": coverage.get("recommended_action"),
        # ── Deflated Sharpe (López de Prado) ──
        "probabilistic_sharpe": deflated.get("probabilistic_sharpe"),
        "deflated_sharpe_value": deflated.get("deflated_sharpe"),
        "n_trials": deflated.get("n_trials"),
        "observed_sharpe": deflated.get("observed_sharpe"),
        "sr_zero": deflated.get("sr_zero"),
        # ── Trade-order Monte Carlo (full block) ──
        "tom_skipped_reason": tom.get("skipped_reason"),
        "tom_realized_sharpe": tom.get("realized_sharpe"),
        "tom_n_trades": tom.get("n_trades"),
        "tom_n_resamples": tom.get("n_resamples"),
        "tom_distribution": tom.get("sharpe_distribution") or {},
        "tom_observed_vs_distribution": tom.get("observed_vs_distribution") or {},
        "tom_position_pct": (tom.get("observed_vs_distribution") or {}).get("position_pct"),
        "tom_interpretation": (tom.get("observed_vs_distribution") or {}).get("interpretation"),
        # ── Regime decomposition (4 quadrants) ──
        "regime_by_hour": regime.get("by_hour") or [],
        "regime_by_dow": regime.get("by_dow") or [],
        "regime_by_ttr": regime.get("by_ttr") or [],
        "regime_by_size": regime.get("by_size") or [],
        # ── Ensemble bands (sample of fills with p10/p50/p90) ──
        "ensemble_band": ensemble_capped,
        "ensemble_total": len(ensemble) if isinstance(ensemble, list) else 0,
        # ── Counterfactual replays (sample list) ──
        "counterfactuals": counterfactuals_capped,
        "counterfactuals_total": len(counterfactuals) if isinstance(counterfactuals, list) else 0,
        # ── Partial-fill aggregates (full block) ──
        "instant_fill_rate": partial.get("instant_fill_rate"),
        "n_orders": partial.get("n_orders"),
        "n_instant_fills": partial.get("n_instant_fills"),
        "mean_children_per_order": partial.get("mean_children_per_order"),
        "max_children_per_order": partial.get("max_children_per_order"),
        "mean_intra_order_seconds": partial.get("mean_intra_order_seconds"),
        "mean_vwap_dispersion_bps": partial.get("mean_vwap_dispersion_bps"),
        "child_count_distribution": partial.get("child_count_distribution") or [],
        # ── Data quality (acceptance, gaps, drops, rejects breakdown) ──
        "dq_accept_rate": data_quality.get("accept_rate"),
        "dq_accepted_books": data_quality.get("accepted_books"),
        "dq_total_attempts": data_quality.get("total_attempts"),
        "dq_sequence_gaps": data_quality.get("sequence_gaps_observed"),
        "dq_tokens_tracked": data_quality.get("tokens_tracked"),
        "dq_queue_dropped": data_quality.get("queue_dropped"),
        "dq_rejects_by_reason": data_quality.get("rejects_by_reason") or {},
        # ── Outcome netting (full block) ──
        "gross_exposure_usd": outcome.get("gross_exposure_usd"),
        "net_exposure_usd": outcome.get("net_exposure_usd"),
        "capital_efficiency_pct": outcome.get("capital_efficiency_pct"),
        "rebate_estimate_usd": outcome.get("rebate_estimate_usd"),
        "locked_capital_usd": outcome.get("locked_capital_usd"),
        "open_positions": outcome.get("open_positions"),
        "outcome_groups": outcome.get("outcome_groups") or {},
        "avg_lockup_seconds": outcome.get("avg_lockup_seconds"),
        "max_lockup_seconds": outcome.get("max_lockup_seconds"),
        # ── Fill model (Cox PH snapshot) ──
        "fill_model_loaded": bool(fill_model.get("loaded")),
        "fill_model_family": fill_model.get("family"),
        "fill_model_n_events": fill_model.get("n_events"),
        "concordance_index": fill_model.get("concordance_index"),
        "hazard_ratios": hazard_ratios,
        "calibration_bins": fill_model.get("calibration_bins") or [],
        # ── Latency distribution ──
        "latency_p50_ms": latency.get("p50_ms"),
        "latency_p95_ms": latency.get("p95_ms"),
        "latency_p99_ms": latency.get("p99_ms"),
        "latency_sample_count": latency.get("sample_count"),
        "latency_pessimistic_ms": latency.get("pessimistic_ms"),
        "latency_realistic_ms": latency.get("realistic_ms"),
        "latency_optimistic_ms": latency.get("optimistic_ms"),
        # ── Trade-vs-cancel decomposition ──
        "decomp_trade_count": decomposition.get("trade_count"),
        "decomp_cancel_count": decomposition.get("cancel_count"),
        "decomp_trade_count_pct": decomposition.get("trade_count_pct"),
        "decomp_window_hours": decomposition.get("window_hours"),
        # ── Empirical constants ──
        "empirical_measured": bool(empirical.get("measured")),
        "empirical_sample_count": empirical.get("sample_count"),
        "empirical_values": empirical.get("values") or {},
    }
