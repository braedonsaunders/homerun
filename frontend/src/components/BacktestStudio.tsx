/**
 * BacktestStudio — institutional-grade backtest workbench.
 *
 * The single backtest surface in the Research subview.  Pulls every
 * advanced datum the platform has — Cox PH fill model state,
 * ensemble PnL bands, empirical constants, latency distribution,
 * trade-vs-cancel decomposition, counterfactual replay diagnostics,
 * triangulation against shadow + live PnL — into one coherent
 * multi-pane workbench.
 *
 * Layout: 3-pane split — left rail (run controls + history),
 * center (results + equity curve + trades), right rail
 * (microstructure / fill model state).  Fonts and color tokens
 * follow the FillModelPanel conventions so the two surfaces feel
 * like one product.
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Flame,
  Layers3,
  LineChart as LineChartIcon,
  Loader2,
  Play,
  RotateCcw,
  Sliders,
  Sparkles,
  Square,
  Target,
  TrendingDown,
  TrendingUp,
  Wand2,
  Zap,
} from 'lucide-react'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { ScrollArea } from './ui/scroll-area'
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs'
import { cn } from '../lib/utils'
import StrategyConfigForm from './StrategyConfigForm'
import {
  groupStrategyParamFields,
  type StrategyParamGroup,
} from '../lib/strategyParams'
import {
  streamStrategyParamsAutoresearchExperiment,
  stopStrategyParamsAutoresearchExperiment,
  type StrategyParamsStartBody,
} from '../services/apiIntelligence'
import {
  type BacktestRunSummary,
  type CPCVResult,
  type MonteCarloLatencyResult,
  type PortfolioCorrelationResult,
  type UnifiedBacktestResult,
  type WalkForwardResult,
  type BacktestRunStatus,
  cancelBacktestRun,
  enqueueBacktest,
  getBacktestRun,
  getBacktestRunStatus,
  getDriftMonitor,
  getPortfolioCorrelation,
  listBacktestRuns,
  runCPCV,
  runMonteCarloLatency,
  runWalkForward,
} from '../services/apiBacktest'
import {
  getActiveFillModel,
  getDecompositionSummary,
  getEmpiricalConstants,
  getLatencyDistribution,
  getTriangulation,
} from '../services/apiFillModel'
import {
  listProviderDatasets,
  type ProviderDataset,
} from '../services/apiProviders'

interface BacktestStudioProps {
  initialSourceCode?: string
  initialSlug?: string
  // Strategy database UUID — required by the param-iteration endpoint
  // ``/autoresearch/strategy/{id}/params/stream`` which keys on the
  // Strategy primary key.  When omitted, the "Iterate params" button
  // is hidden (the studio degrades gracefully to manual backtests).
  initialStrategyId?: string
  initialConfig?: Record<string, unknown>
  // Param schema (``{ param_fields: [...] }``) declared by the
  // strategy.  Drives the dynamic "Strategy parameters" panel in
  // the left rail so the operator can override the strategy's
  // declared knobs FOR THIS RUN — same machinery the bot
  // orchestrator's tune subtab uses.  Optional: when absent (or
  // when the strategy declares no fields) the panel is hidden and
  // the run uses ``initialConfig`` verbatim.
  initialParamSchema?: { param_fields?: Array<Record<string, unknown>> } | null
  // Human-readable strategy label shown in the "loaded" indicator so
  // the operator can verify which strategy is staged when switching
  // between them in the parent dropdown.  Optional — falls back to
  // the slug.
  strategyLabel?: string | null
}

// Backend sentinel for "ratio with zero denominator" (e.g., gain-to-
// pain with no losses).  metrics.py:_NO_DENOM_SENTINEL = 1_000_000.
// Render as ∞ in the UI so it's not mistaken for an absurd magnitude.
const NO_DENOM_SENTINEL = 1_000_000

function fmtNum(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(Number(value))) return '—'
  if (Math.abs(Number(value)) >= NO_DENOM_SENTINEL) return '∞'
  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  })
}

function fmtUsd(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(Number(value))) return '—'
  const v = Number(value)
  const abs = Math.abs(v)
  const formatted =
    abs >= 1_000_000
      ? `$${(v / 1_000_000).toFixed(2)}M`
      : abs >= 1_000
        ? `$${(v / 1_000).toFixed(2)}K`
        : `$${v.toFixed(2)}`
  return formatted
}

function fmtPct(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(Number(value))) return '—'
  return `${Number(value).toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  })}%`
}

function fmtMs(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(Number(value))) return '—'
  const v = Math.round(Number(value))
  return v >= 1000 ? `${(v / 1000).toFixed(2)}s` : `${v}ms`
}

function StatTile({
  label,
  value,
  hint,
  tone = 'neutral',
  icon: Icon,
}: {
  label: string
  value: string
  hint?: string
  tone?: 'good' | 'warn' | 'bad' | 'neutral'
  icon?: typeof TrendingUp
}) {
  return (
    <div
      className={cn(
        'rounded-md border px-3 py-2 leading-tight',
        tone === 'good' && 'border-emerald-500/30 bg-emerald-500/5',
        tone === 'warn' && 'border-amber-500/30 bg-amber-500/5',
        tone === 'bad' && 'border-red-500/30 bg-red-500/5',
        tone === 'neutral' && 'border-border/50 bg-card/40',
      )}
    >
      <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-muted-foreground">
        {Icon ? <Icon className="h-3 w-3" /> : null}
        {label}
      </div>
      <div
        className={cn(
          'mt-0.5 text-base font-semibold tabular-nums',
          tone === 'good' && 'text-emerald-300',
          tone === 'warn' && 'text-amber-300',
          tone === 'bad' && 'text-red-300',
        )}
      >
        {value}
      </div>
      {hint ? <div className="mt-0.5 text-[10px] text-muted-foreground">{hint}</div> : null}
    </div>
  )
}

function MetricRow({ label, m, tone }: { label: string; m: { value: number; ci_low: number | null; ci_high: number | null } | null | undefined; tone?: 'good' | 'warn' | 'bad' | 'neutral' }) {
  const v = m?.value ?? null
  const lo = m?.ci_low ?? null
  const hi = m?.ci_high ?? null
  return (
    <div className="grid grid-cols-[120px,80px,1fr] items-center gap-2 py-1 text-xs">
      <div className="text-muted-foreground">{label}</div>
      <div className={cn('font-mono tabular-nums', tone === 'good' && 'text-emerald-300', tone === 'bad' && 'text-red-300')}>
        {fmtNum(v, 3)}
      </div>
      {lo != null && hi != null ? (
        <div className="text-[10px] text-muted-foreground">
          ci [{fmtNum(lo, 3)}, {fmtNum(hi, 3)}]
        </div>
      ) : (
        <div className="text-[10px] text-muted-foreground">no ci</div>
      )}
    </div>
  )
}

function EquityCurveChart({ points }: { points: Array<{ timestamp?: string; equity_usd?: number }> }) {
  const { t } = useTranslation()
  if (!points || points.length < 2) {
    return (
      <div className="rounded-md border border-dashed border-border/50 bg-card/30 px-3 py-6 text-center text-xs text-muted-foreground">
        {t('backtestStudio.equityCurveEmpty')}
      </div>
    )
  }
  const w = 560
  const h = 140
  const xs = points.map((_, i) => (i / (points.length - 1)) * (w - 16) + 8)
  const equities = points.map((p) => Number(p.equity_usd ?? 0))
  const maxE = Math.max(...equities)
  const minE = Math.min(...equities)
  const range = Math.max(1e-6, maxE - minE)
  const ys = equities.map((e) => h - 10 - ((e - minE) / range) * (h - 24))
  const path = xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${ys[i].toFixed(1)}`).join(' ')
  // baseline (initial capital reference)
  const initial = equities[0]
  const yBaseline = h - 10 - ((initial - minE) / range) * (h - 24)
  const ending = equities[equities.length - 1]
  const isUp = ending >= initial
  return (
    <div className="rounded-md border border-border/50 bg-card/40 p-2">
      <div className="flex items-center justify-between text-[10px] text-muted-foreground">
        <span>{`${t('backtestStudio.equityLabel')} ${fmtUsd(initial)} → ${fmtUsd(ending)}`}</span>
        <span>{t('backtestStudio.samplesCount', { n: points.length })}</span>
      </div>
      <svg width={w} height={h} className="mt-1">
        <line x1={8} y1={yBaseline} x2={w - 8} y2={yBaseline} stroke="rgb(120,120,120)" strokeOpacity={0.35} strokeDasharray="3,3" strokeWidth={0.5} />
        <path d={path} fill="none" stroke={isUp ? 'hsl(150, 80%, 55%)' : 'hsl(0, 80%, 60%)'} strokeWidth={1.5} />
      </svg>
    </div>
  )
}

function CorrelationHeatmap({ result }: { result: PortfolioCorrelationResult }) {
  const { t } = useTranslation()
  const { strategies, correlation_matrix, summary } = result
  if (!strategies || strategies.length === 0) {
    return (
      <div className="text-[11px] text-muted-foreground italic">
        {t('backtestStudio.correlationNoStrategies', { days: result.window_days })}
      </div>
    )
  }
  const cellSize = Math.min(36, Math.max(20, Math.floor(280 / strategies.length)))
  const labelMaxLen = 12
  const truncate = (s: string) => (s.length > labelMaxLen ? s.slice(0, labelMaxLen - 1) + '…' : s)
  const colorFor = (r: number): string => {
    // Diverging palette: -1 blue, 0 neutral, +1 red.  Reds are the
    // worry colour because high cross-correlation = concentrated risk.
    if (r >= 0.7) return 'rgba(239, 68, 68, 0.8)'
    if (r >= 0.4) return 'rgba(249, 115, 22, 0.6)'
    if (r >= 0.1) return 'rgba(245, 158, 11, 0.4)'
    if (r >= -0.1) return 'rgba(120, 120, 120, 0.25)'
    if (r >= -0.4) return 'rgba(34, 197, 94, 0.4)'
    if (r >= -0.7) return 'rgba(16, 185, 129, 0.6)'
    return 'rgba(34, 197, 94, 0.85)'
  }
  return (
    <div>
      <div className="grid grid-cols-3 gap-2">
        <StatTile
          label={t('backtestStudio.diversification')}
          value={`${(summary.diversification_ratio * 100).toFixed(0)}%`}
          hint={t('backtestStudio.diversificationHint')}
          tone={summary.diversification_ratio >= 0.7 ? 'good' : summary.diversification_ratio >= 0.4 ? 'warn' : 'bad'}
        />
        <StatTile
          label={t('backtestStudio.meanAbsRho')}
          value={summary.mean_abs_pairwise_correlation.toFixed(2)}
          hint={`${t('backtestStudio.min')} ${summary.min_pairwise_correlation.toFixed(2)} · ${t('backtestStudio.max')} ${summary.max_pairwise_correlation.toFixed(2)}`}
          tone={summary.mean_abs_pairwise_correlation >= 0.5 ? 'bad' : summary.mean_abs_pairwise_correlation >= 0.3 ? 'warn' : 'good'}
        />
        <StatTile
          label={t('backtestStudio.strategiesLabel')}
          value={`${summary.n_strategies}`}
          hint={t('backtestStudio.daysOfPnl', { n: summary.n_days })}
        />
      </div>
      <div className="mt-3 overflow-x-auto">
        <table className="border-collapse">
          <thead>
            <tr>
              <th />
              {strategies.map((s) => (
                <th
                  key={s}
                  title={s}
                  style={{ width: cellSize, height: cellSize }}
                  className="text-[8px] text-muted-foreground"
                >
                  <div className="origin-bottom-left -translate-y-1 translate-x-2 -rotate-45 whitespace-nowrap font-mono">
                    {truncate(s)}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {strategies.map((rowStrat, i) => (
              <tr key={rowStrat}>
                <td
                  title={rowStrat}
                  className="pr-1 text-right text-[9px] font-mono text-muted-foreground"
                  style={{ height: cellSize }}
                >
                  {truncate(rowStrat)}
                </td>
                {strategies.map((_, j) => {
                  const r = correlation_matrix[i]?.[j] ?? 0
                  return (
                    <td
                      key={`${i}-${j}`}
                      title={`${rowStrat} ↔ ${strategies[j]}: ρ = ${r.toFixed(2)}`}
                      style={{
                        width: cellSize,
                        height: cellSize,
                        backgroundColor: colorFor(r),
                        border: '1px solid rgba(120,120,120,0.15)',
                      }}
                      className="text-center font-mono text-[8px] text-foreground/80"
                    >
                      {r.toFixed(2)}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-2 text-[10px] text-muted-foreground">
        {t('backtestStudio.correlationLegend')}
      </div>
    </div>
  )
}


function RegimeBlock({
  title,
  rows,
}: {
  title: string
  rows: Array<{ bucket: string; n: number; wins: number; total_pnl_usd: number; win_rate: number; mean_pnl_usd: number }>
}) {
  const { t } = useTranslation()
  const maxN = rows.reduce((m, r) => Math.max(m, r.n), 1)
  return (
    <div className="rounded-sm border border-border/40 bg-background/40 p-2">
      <div className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground">{title}</div>
      {rows.length === 0 ? (
        <div className="text-[10px] text-muted-foreground italic">{t('backtestStudio.noData')}</div>
      ) : (
        <div className="space-y-0.5">
          {rows
            .slice()
            .sort((a, b) => b.n - a.n)
            .map((r) => {
              const winPct = r.win_rate * 100
              const tone =
                r.n < 3 ? 'text-muted-foreground'
                  : winPct >= 60 ? 'text-emerald-300'
                    : winPct >= 40 ? 'text-amber-300'
                      : 'text-red-300'
              const widthPct = Math.min(100, (r.n / maxN) * 100)
              return (
                <div key={r.bucket} className="grid grid-cols-[60px,1fr,40px] items-center gap-1 text-[10px]">
                  <span className="truncate font-mono">{r.bucket}</span>
                  <div className="relative h-2 rounded-sm bg-muted/30">
                    <div
                      className={cn(
                        'absolute inset-y-0 left-0 rounded-sm',
                        winPct >= 60 ? 'bg-emerald-500/50' : winPct >= 40 ? 'bg-amber-500/50' : 'bg-red-500/50',
                      )}
                      style={{ width: `${widthPct}%` }}
                    />
                  </div>
                  <span className={cn('text-right font-mono tabular-nums', tone)}>
                    {r.n < 1 ? '—' : `${winPct.toFixed(0)}%`}
                  </span>
                </div>
              )
            })}
        </div>
      )}
    </div>
  )
}


function CalibrationPlot({ bins }: { bins: Array<{ predicted_mean: number; observed_rate: number; n: number }> }) {
  const { t } = useTranslation()
  if (!bins || bins.length === 0) return null
  const w = 220
  const h = 110
  const pad = 8
  const innerW = w - pad * 2
  const innerH = h - pad * 2
  const points = bins
    .slice()
    .sort((a, b) => a.predicted_mean - b.predicted_mean)
  // Diagonal y=x reference (perfect calibration).
  const diag = `M${pad},${h - pad} L${w - pad},${pad}`
  // Observed rate trace.
  const trace = points
    .map((p, i) => {
      const x = pad + Math.max(0, Math.min(1, p.predicted_mean)) * innerW
      const y = h - pad - Math.max(0, Math.min(1, p.observed_rate)) * innerH
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(' ')
  // Per-bin sample-size circles.
  const maxN = Math.max(...points.map((p) => p.n)) || 1
  return (
    <div className="rounded-md border border-border/40 bg-card/40 p-2">
      <svg width={w} height={h}>
        <line x1={pad} y1={pad} x2={pad} y2={h - pad} stroke="rgb(120,120,120)" strokeOpacity={0.3} strokeWidth={0.5} />
        <line x1={pad} y1={h - pad} x2={w - pad} y2={h - pad} stroke="rgb(120,120,120)" strokeOpacity={0.3} strokeWidth={0.5} />
        <path d={diag} fill="none" stroke="rgb(120,120,120)" strokeOpacity={0.4} strokeDasharray="2,2" strokeWidth={0.6} />
        <path d={trace} fill="none" stroke="hsl(160, 80%, 55%)" strokeWidth={1.5} />
        {points.map((p, i) => {
          const x = pad + Math.max(0, Math.min(1, p.predicted_mean)) * innerW
          const y = h - pad - Math.max(0, Math.min(1, p.observed_rate)) * innerH
          const r = 2 + (p.n / maxN) * 3
          return (
            <circle
              key={i}
              cx={x}
              cy={y}
              r={r}
              fill="hsl(160, 80%, 55%)"
              fillOpacity={0.7}
              stroke="hsl(160, 80%, 65%)"
              strokeWidth={0.5}
            />
          )
        })}
      </svg>
      <div className="flex items-center justify-between text-[9px] text-muted-foreground">
        <span>{t('backtestStudio.calibration0')}</span>
        <span>{t('backtestStudio.calibrationDiag')}</span>
        <span>1</span>
      </div>
    </div>
  )
}


function HazardBar({ label, hr }: { label: string; hr: number }) {
  const clamped = Math.max(0.2, Math.min(2.5, hr))
  const isPos = clamped >= 1.0
  const widthPct = Math.min(100, Math.abs(Math.log2(clamped)) * 100)
  return (
    <div className="grid grid-cols-[140px,1fr,60px] items-center gap-2 py-0.5 text-[11px]">
      <div className="truncate text-muted-foreground">{label}</div>
      <div className="relative h-2.5 rounded-sm bg-muted/30">
        <div className="absolute inset-y-0 left-1/2 w-px bg-border/70" />
        <div
          className={cn('absolute inset-y-0 rounded-sm', isPos ? 'left-1/2 bg-emerald-500/60' : 'right-1/2 bg-red-500/60')}
          style={{ width: `${widthPct / 2}%` }}
        />
      </div>
      <div className={cn('text-right font-mono tabular-nums', isPos ? 'text-emerald-300' : 'text-red-300')}>{hr.toFixed(2)}×</div>
    </div>
  )
}

/**
 * Replay-source pill.  Tiny indicator that shows which book-replay
 * data source the matching engine ran against:
 *
 *   - snapshots     → BookReplay over market_microstructure_snapshots
 *   - deltas        → BookDeltaReplay over book_delta_events (live-parity)
 *   - deltas+anchor → BookDeltaReplay seeded from mms anchors
 *
 * Color codes: deltas* paths get an emerald tint (live-parity), the
 * pure-snapshots path is neutral.  Both light + dark colorways
 * preserve AA contrast.
 */
/**
 * Skeleton shown in the center pane while a backtest is running.
 *
 * Replaces the previous-run's KPIs / charts / tables with shimmer
 * placeholders so the operator never sees stale data alongside an
 * in-flight mutation.  Mirrors the layout of the real Performance
 * tab (4 KPI tiles + a wide chart-equivalent block + a metrics
 * grid) so the visual transition is smooth — when the real result
 * arrives, the layout doesn't jump.
 *
 * The pulse animation is deliberately slow (1.6s) and the contrast
 * is subtle.  Don't make it loud — backtests can take 30s-2min and
 * a hyperactive shimmer becomes irritating.
 */
function RunningBacktestSkeleton({
  variant,
  caption,
  status,
  onCancel,
}: {
  variant: 'running' | 'loading'
  caption: string
  /** Live status from the worker poll, when available.  When set, the
   *  banner renders a real progress bar + activity message. */
  status?: BacktestRunStatus | null
  /** Operator cancel handler.  Renders a stop button when set. */
  onCancel?: () => void
}) {
  const { t } = useTranslation()
  // Real progress comes from the worker's debounced writes to the
  // BacktestRun row.  ``progress`` is 0-1 when an estimate is set;
  // otherwise we fall back to an indeterminate-style snapshots-
  // processed counter from ``message``.
  const pct = status ? Math.round(Math.min(100, Math.max(0, status.progress * 100))) : null
  const determinate = pct !== null && status && (status.snapshots_total_estimate ?? 0) > 0
  return (
    <div className="space-y-3">
      <div
        className={cn(
          'rounded-md border px-3 py-2.5 text-[11px]',
          variant === 'running'
            ? 'border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-500/40 dark:bg-amber-500/5 dark:text-amber-200'
            : 'border-border/40 bg-card/40 text-muted-foreground',
        )}
      >
        <div className="flex items-center gap-2">
          <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" />
          <span className="flex-1">{status?.message || caption}</span>
          {status && pct !== null ? (
            <span className="font-mono tabular-nums text-[10px] opacity-80">
              {determinate ? `${pct}%` : t('backtestStudio.snapsCount', { n: (status.snapshots_processed || 0).toLocaleString() })}
            </span>
          ) : null}
          {onCancel && status && ['queued', 'running'].includes(status.status) && !status.cancel_requested ? (
            <button
              type="button"
              onClick={onCancel}
              className="rounded-sm border border-amber-400 bg-white/60 px-1.5 py-0.5 text-[10px] font-medium text-amber-900 hover:bg-white/80 dark:border-amber-500/40 dark:bg-amber-500/10 dark:text-amber-200 dark:hover:bg-amber-500/20"
            >
              {t('backtestStudio.cancel')}
            </button>
          ) : null}
          {status?.cancel_requested ? (
            <span className="rounded-sm bg-amber-500/20 px-1.5 py-0.5 text-[10px] text-amber-900 dark:text-amber-200">
              {t('backtestStudio.cancelling')}
            </span>
          ) : null}
        </div>
        {/* Progress bar — only renders when we have a determinate
            progress estimate.  Indeterminate mode just shows the
            snapshot counter above. */}
        {determinate ? (
          <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-amber-200/50 dark:bg-amber-900/40">
            <div
              className="h-full rounded-full bg-amber-500 transition-all dark:bg-amber-400"
              style={{ width: `${pct}%` }}
            />
          </div>
        ) : status && variant === 'running' ? (
          // Indeterminate mode — animated bar that pulses without
          // implying a percentage we don't have.
          <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-amber-200/50 dark:bg-amber-900/40">
            <div
              className="h-full w-1/3 rounded-full bg-amber-500 animate-pulse dark:bg-amber-400"
              style={{ animationDuration: '1.2s' }}
            />
          </div>
        ) : null}
      </div>

      {/* Mirror the 4-tile KPI grid */}
      <div className="grid grid-cols-4 gap-2">
        {[0, 1, 2, 3].map((i) => (
          <div
            key={i}
            className="rounded-md border border-border/40 bg-card/40 px-3 py-2.5"
          >
            <div className="h-2 w-12 rounded bg-muted/60 animate-pulse" />
            <div className="mt-2 h-6 w-20 rounded bg-muted/80 animate-pulse" style={{ animationDuration: '1.6s' }} />
            <div className="mt-1.5 h-2 w-16 rounded bg-muted/40 animate-pulse" />
          </div>
        ))}
      </div>

      {/* Mirror the secondary-metrics two-column block */}
      <div className="grid grid-cols-2 gap-2">
        {[0, 1].map((col) => (
          <div key={col} className="rounded-md border border-border/50 bg-card/40 p-3">
            <div className="mb-2 h-3 w-32 rounded bg-muted/60 animate-pulse" />
            {[0, 1, 2, 3, 4].map((row) => (
              <div key={row} className="flex items-center justify-between py-1">
                <div className="h-2.5 w-20 rounded bg-muted/50 animate-pulse" />
                <div className="h-2.5 w-16 rounded bg-muted/70 animate-pulse" />
              </div>
            ))}
          </div>
        ))}
      </div>

      {/* Mirror the equity-curve / chart block */}
      <div className="rounded-md border border-border/40 bg-card/40 p-3">
        <div className="mb-2 h-3 w-24 rounded bg-muted/60 animate-pulse" />
        <div className="h-32 rounded bg-muted/30 animate-pulse" style={{ animationDuration: '2.0s' }} />
      </div>
    </div>
  )
}

/**
 * Discovery-mode pill.  Shows which path the strategy's opportunities
 * came from on this run:
 *   - hybrid                — live cache + replay-discovery merged
 *   - historical_synthesis  — pure replay-discovery (no live opps)
 *   - live_opps             — pure cache (legacy path; usually means
 *                             ``discover_from_history`` was disabled)
 *
 * "Hybrid" and "historical_synthesis" are emerald — they mean the
 * strategy actually ran discovery against historical data.  Pure
 * "live_opps" is amber to signal "this is fill-counterfactual mode,
 * not full backtest".
 */
function DiscoveryModePill({ mode }: { mode?: string }) {
  const { t } = useTranslation()
  if (!mode) return null
  const label =
    mode === 'historical_synthesis'
      ? t('backtestStudio.discoveryReplayDetect')
      : mode === 'hybrid'
      ? t('backtestStudio.discoveryLiveReplay')
      : mode === 'live_opps'
      ? t('backtestStudio.discoveryLiveCacheOnly')
      : mode
  const live_only = mode === 'live_opps'
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-sm px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide',
        live_only
          ? 'bg-amber-100 text-amber-800 ring-1 ring-amber-300 dark:bg-amber-500/15 dark:text-amber-200 dark:ring-amber-500/30'
          : 'bg-emerald-100 text-emerald-800 ring-1 ring-emerald-300 dark:bg-emerald-500/15 dark:text-emerald-200 dark:ring-emerald-500/30',
      )}
      title={
        mode === 'hybrid'
          ? t('backtestStudio.discoveryHybridTip')
          : mode === 'historical_synthesis'
          ? t('backtestStudio.discoveryHistoricalTip')
          : t('backtestStudio.discoveryLiveOppsTip')
      }
    >
      {t('backtestStudio.discoveryPrefix')} {label}
    </span>
  )
}

function ReplaySourcePill({ source }: { source?: string }) {
  const { t } = useTranslation()
  if (!source) return null
  const isDelta = source.startsWith('deltas')
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-sm px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide',
        isDelta
          ? 'bg-emerald-100 text-emerald-800 ring-1 ring-emerald-300 dark:bg-emerald-500/15 dark:text-emerald-200 dark:ring-emerald-500/30'
          : 'bg-muted text-muted-foreground',
      )}
      title={isDelta ? t('backtestStudio.replayDeltaTip') : t('backtestStudio.replaySnapshotsTip')}
    >
      {t('backtestStudio.replayPrefix')} {source}
    </span>
  )
}

/**
 * Data-coverage / fidelity banner.
 *
 * Renders ABOVE the trade-count KPI tiles when a run completes so the
 * operator immediately sees whether 0 trades is a strategy result or
 * a data-coverage artifact.  Color coding (paired light + dark for
 * AA contrast on both):
 *   - high   → emerald, single line summary
 *   - medium → amber, recommendation
 *   - low/none → red, recommendation + explicit deltas-vs-snapshots split
 *
 * The banner reflects the live-parity delta-replay path: when the
 * engine ran on book_delta_events (deltas*), fidelity is implicitly
 * high regardless of the snapshot table — it's the SAME data the live
 * system uses.
 */
function DataCoverageBanner({
  coverage,
  replaySource,
  discoveryMode,
}: {
  coverage?: UnifiedBacktestResult['data_coverage']
  replaySource?: string
  discoveryMode?: string
}) {
  const { t } = useTranslation()
  if (!coverage || !coverage.fidelity_rating) return null
  const rating = coverage.fidelity_rating
  const median = coverage.median_snaps_per_token_per_hour ?? 0
  const deltasMedian = coverage.median_deltas_per_token_per_hour ?? 0
  const tokensWithDeltas = coverage.tokens_with_deltas ?? 0
  const tokensWithSnaps = coverage.tokens_with_snapshots ?? 0
  const oppTokens = coverage.opp_tokens ?? 0
  const rec = coverage.recommended_action || ''
  const ranOnDeltas = replaySource?.startsWith('deltas') ?? false

  // Delta-replay path → live-parity, always green regardless of the
  // snapshot-table coverage rating.
  if (ranOnDeltas) {
    return (
      <div className="flex items-center justify-between rounded-md border border-emerald-300 bg-emerald-50 px-3 py-1.5 text-[11px] text-emerald-900 dark:border-emerald-500/30 dark:bg-emerald-500/5 dark:text-emerald-200">
        <div>
          <span className="font-semibold">{t('backtestStudio.liveParityReplay')}</span>
          <span className="ml-2 text-emerald-800/90 dark:text-emerald-300/80">
            {t('backtestStudio.liveParityDetails', { withDeltas: tokensWithDeltas, total: oppTokens, deltasMedian: deltasMedian.toFixed(1) })}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <DiscoveryModePill mode={discoveryMode} />
          <ReplaySourcePill source={replaySource} />
        </div>
      </div>
    )
  }

  if (rating === 'high') {
    return (
      <div className="flex items-center justify-between rounded-md border border-emerald-300 bg-emerald-50 px-3 py-1.5 text-[11px] text-emerald-900 dark:border-emerald-500/30 dark:bg-emerald-500/5 dark:text-emerald-200">
        <div>
          <span className="font-semibold">{t('backtestStudio.fidelityHigh')}</span>
          <span className="ml-2 text-emerald-800/90 dark:text-emerald-300/80">
            {t('backtestStudio.fidelityHighDetails', { median: median.toFixed(1), withSnaps: tokensWithSnaps, total: oppTokens })}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <DiscoveryModePill mode={discoveryMode} />
          <ReplaySourcePill source={replaySource} />
        </div>
      </div>
    )
  }

  if (rating === 'medium') {
    return (
      <div className="rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-900 dark:border-amber-500/40 dark:bg-amber-500/10 dark:text-amber-200">
        <div className="flex items-center justify-between">
          <span className="font-semibold">{t('backtestStudio.fidelityMedium')}</span>
          <div className="flex items-center gap-1.5">
          <DiscoveryModePill mode={discoveryMode} />
          <ReplaySourcePill source={replaySource} />
        </div>
        </div>
        <div className="mt-1 text-amber-900/90 dark:text-amber-300/90">
          {t('backtestStudio.fidelityMediumDetails', { median: median.toFixed(1), withSnaps: tokensWithSnaps, total: oppTokens })}
        </div>
        {rec ? <div className="mt-1 text-amber-900/80 dark:text-amber-200/80">{rec}</div> : null}
      </div>
    )
  }

  // low / none — most actionable case.  This is what shows when "0
  // trades" is a coverage problem.
  return (
    <div className="rounded-md border border-red-300 bg-red-50 px-3 py-2 text-xs text-red-900 dark:border-red-500/50 dark:bg-red-500/10 dark:text-red-100">
      <div className="flex items-center justify-between">
        <span className="font-semibold text-red-900 dark:text-red-200">
          {t('backtestStudio.fidelityLowHeader', { rating: rating.toUpperCase() })}
        </span>
        <div className="flex items-center gap-1.5">
          <DiscoveryModePill mode={discoveryMode} />
          <ReplaySourcePill source={replaySource} />
        </div>
      </div>
      <div className="mt-1 grid grid-cols-2 gap-2 text-[11px] text-red-900/90 dark:text-red-100/90">
        <div className="rounded-sm bg-red-100 px-2 py-1 dark:bg-red-500/10">
          <div className="text-red-800/80 dark:text-red-200/70">{t('backtestStudio.fidelitySnapshotsLabel')}</div>
          <div className="font-mono">
            {t('backtestStudio.fidelityTokensHr', { withTokens: tokensWithSnaps, total: oppTokens, median: median.toFixed(1) })}
          </div>
        </div>
        <div className="rounded-sm bg-red-100 px-2 py-1 dark:bg-red-500/10">
          <div className="text-red-800/80 dark:text-red-200/70">{t('backtestStudio.fidelityDeltasLabel')}</div>
          <div className="font-mono">
            {t('backtestStudio.fidelityTokensHr', { withTokens: tokensWithDeltas, total: oppTokens, median: deltasMedian.toFixed(1) })}
          </div>
        </div>
      </div>
      {rec ? <div className="mt-2 text-red-900/90 dark:text-red-100/90">{rec}</div> : null}
    </div>
  )
}

function EnsembleBand({ band }: { band: UnifiedBacktestResult['ensemble_band'] }) {
  const { t } = useTranslation()
  if (!band || band.length === 0) {
    return (
      <div className="text-xs text-muted-foreground italic">
        {t('backtestStudio.ensembleEmpty')}
      </div>
    )
  }
  return (
    <div className="space-y-1">
      {band.map((b, i) => (
        <div key={`${b.fill_id || i}-${i}`} className="rounded-sm border border-border/40 bg-background/40 px-2 py-1.5">
          <div className="flex items-center justify-between text-[10px] text-muted-foreground">
            <span>{t('backtestStudio.ensembleFillNumber', { n: i + 1 })}</span>
            {b.cox_loaded ? <Badge className="bg-emerald-500/10 text-emerald-300 text-[9px]">{t('backtestStudio.ensembleCox')}</Badge> : <Badge variant="outline" className="text-[9px]">{t('backtestStudio.ensembleHeuristic')}</Badge>}
          </div>
          <div className="mt-1 grid grid-cols-3 gap-1 text-[11px]">
            <div className="rounded-sm bg-red-500/5 px-1.5 py-0.5 text-red-300">
              p10 {fmtNum(b.pessimistic * 100, 1)}%
            </div>
            <div className="rounded-sm bg-amber-500/5 px-1.5 py-0.5 text-amber-300">
              p50 {fmtNum(b.realistic * 100, 1)}%
            </div>
            <div className="rounded-sm bg-emerald-500/5 px-1.5 py-0.5 text-emerald-300">
              p90 {fmtNum(b.optimistic * 100, 1)}%
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

function CounterfactualList({ rows }: { rows: UnifiedBacktestResult['counterfactuals'] }) {
  const { t } = useTranslation()
  if (!rows || rows.length === 0) {
    return (
      <div className="text-xs text-muted-foreground italic">
        {t('backtestStudio.counterfactualEmpty')}
      </div>
    )
  }
  return (
    <div className="space-y-1">
      {rows.map((row, i) => {
        const r = row.result
        const fillRatio =
          row.fill.size > 0 ? Math.min(1, Math.max(0, r.filled_shares / row.fill.size)) : 0
        const tone = r.expired ? 'warn' : fillRatio >= 0.99 ? 'good' : 'neutral'
        return (
          <div
            key={i}
            className={cn(
              'rounded-sm border bg-background/40 px-2 py-1.5 text-[11px]',
              tone === 'good' && 'border-emerald-500/30',
              tone === 'warn' && 'border-amber-500/30',
              tone === 'neutral' && 'border-border/40',
            )}
          >
            <div className="flex items-center justify-between">
              <span className="font-mono">
                {row.fill.side.toUpperCase()} ${fmtNum(row.fill.price, 4)} × {fmtNum(row.fill.size, 1)}
              </span>
              <span className={cn('text-[10px]', tone === 'good' && 'text-emerald-300', tone === 'warn' && 'text-amber-300')}>
                {r.expired ? t('backtestStudio.counterfactualExpired') : t('backtestStudio.counterfactualFilled', { pct: (fillRatio * 100).toFixed(0) })}
              </span>
            </div>
            <div className="mt-0.5 flex flex-wrap gap-2 text-[10px] text-muted-foreground">
              <span>{t('backtestStudio.counterfactualQueue', { n: fmtNum(r.final_queue_ahead, 0) })}</span>
              <span>{t('backtestStudio.counterfactualTradesAhead', { n: fmtNum(r.trades_ahead_observed, 0) })}</span>
              <span>{t('backtestStudio.counterfactualCancelsAhead', { n: fmtNum(r.cancels_ahead_observed, 0) })}</span>
              {r.time_to_fill_seconds != null ? <span>{t('backtestStudio.counterfactualTtf', { n: fmtNum(r.time_to_fill_seconds, 1) })}</span> : null}
            </div>
          </div>
        )
      })}
    </div>
  )
}

function Sparkline({ values, isUp }: { values: number[]; isUp: boolean }) {
  if (!values || values.length < 2) {
    // Render an empty placeholder strip so the row layout stays
    // consistent across runs that didn't produce an equity curve.
    return <div className="h-4 w-full opacity-30" />
  }
  const w = 80
  const h = 16
  const xs = values.map((_, i) => (i / (values.length - 1)) * (w - 2) + 1)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = Math.max(1e-6, max - min)
  const ys = values.map((v) => h - 2 - ((v - min) / range) * (h - 4))
  const path = xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${ys[i].toFixed(1)}`).join(' ')
  // Baseline reference: the "zero-drift" line at the start equity.
  const baselineY = h - 2 - ((0 - min) / range) * (h - 4)
  const stroke = isUp ? 'hsl(150, 80%, 60%)' : 'hsl(0, 80%, 65%)'
  const lastX = xs[xs.length - 1]
  const lastY = ys[ys.length - 1]
  return (
    <svg width={w} height={h} className="block">
      {Number.isFinite(baselineY) ? (
        <line
          x1={1}
          y1={baselineY}
          x2={w - 1}
          y2={baselineY}
          stroke="rgb(120,120,120)"
          strokeOpacity={0.25}
          strokeDasharray="2,2"
          strokeWidth={0.5}
        />
      ) : null}
      <path d={path} fill="none" stroke={stroke} strokeWidth={1.2} strokeLinejoin="round" />
      <circle cx={lastX} cy={lastY} r={1.5} fill={stroke} />
    </svg>
  )
}


function RunHistory({
  runs,
  activeId,
  onSelect,
}: {
  runs: BacktestRunSummary[]
  activeId: string | null
  onSelect: (run: BacktestRunSummary) => void
}) {
  const { t } = useTranslation()
  if (runs.length === 0) {
    return (
      <div className="px-3 py-3 text-[11px] text-muted-foreground italic" dangerouslySetInnerHTML={{ __html: t('backtestStudio.runHistoryEmpty') }} />
    )
  }
  return (
    <div className="space-y-0.5 px-2 py-2">
      {runs.map((run) => {
        const active = run.run_id === activeId
        const tone = run.status === 'failed' ? 'bad' : run.total_return_pct >= 0 ? 'good' : 'bad'
        return (
          <button
            key={run.run_id}
            onClick={() => onSelect(run)}
            className={cn(
              'block w-full rounded-sm border px-2 py-1.5 text-left text-[11px] transition-colors',
              active
                ? 'border-amber-500/40 bg-amber-500/5'
                : 'border-border/30 bg-card/40 hover:border-border/60 hover:bg-card/60',
            )}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="font-mono text-muted-foreground shrink-0">
                {run.run_id.slice(0, 6)}
              </span>
              <Sparkline
                values={run.sparkline_pct ?? []}
                isUp={run.total_return_pct >= 0}
              />
              <span
                className={cn(
                  'shrink-0 tabular-nums',
                  tone === 'good' && 'text-emerald-300',
                  tone === 'bad' && 'text-red-300',
                )}
              >
                {fmtPct(run.total_return_pct, 1)}
              </span>
            </div>
            <div className="truncate text-[10px] text-muted-foreground">
              {run.strategy_name || run.strategy_slug || t('backtestStudio.unknownStrategy')}
            </div>
            <div className="flex items-center justify-between text-[10px] text-muted-foreground">
              <span>
                {t('backtestStudio.tradesCountShort', { n: run.trade_count })} · {fmtMs(run.total_time_ms)}
              </span>
              <span>{new Date(run.started_at).toLocaleTimeString()}</span>
            </div>
          </button>
        )
      })}
    </div>
  )
}

export default function BacktestStudio({
  initialSourceCode,
  initialSlug,
  initialStrategyId,
  initialConfig,
  initialParamSchema,
  strategyLabel,
}: BacktestStudioProps) {
  const { t } = useTranslation()
  const queryClient = useQueryClient()

  // Run controls.  Internal state is initialized from props once at
  // mount.  Parent prop changes (i.e. operator switches strategies in
  // the dropdown above) flow into the state via the useEffect below.
  const [sourceCode, setSourceCode] = useState<string>(initialSourceCode || '')
  const [slug, setSlug] = useState<string>(initialSlug || '_backtest_unified')
  // Remember the previous slug so we can show a brief "loaded
  // <slug>" highlight when the strategy actually changes.
  const [justLoadedSlug, setJustLoadedSlug] = useState<string | null>(null)

  // Per-run strategy-parameter overrides.  Initialized from the
  // strategy's declared default_config (``initialConfig``) and
  // edited in-place via the dynamic "Strategy parameters" panel.
  // The panel shows the SAME field schema the bot orchestrator's
  // tune subtab renders — every dynamic knob declared in
  // ``param_fields`` is editable for this run.  When the operator
  // changes strategies via the parent dropdown, overrides reset
  // back to the new strategy's defaults (see useEffect below).
  const [paramOverrides, setParamOverrides] = useState<Record<string, unknown>>(
    () => ({ ...(initialConfig || {}) })
  )
  // Whether the dynamic-params panel is expanded.  Defaults to
  // collapsed so the rail stays compact for operators who just
  // want to run the strategy with declared defaults.
  // Left-rail subtab.  Three tabs split the rail by the kind of
  // decision the operator is making: configure the run (Setup), tune
  // the strategy (Parameters), pick a prior result (Runs).  Replaces
  // the old single-column scroll where opening the params panel
  // would push the Run button below the fold.
  const [leftTab, setLeftTab] = useState<'setup' | 'parameters' | 'runs'>('setup')
  // Track which group (Signal / Entry / Sizing / Exit / Risk /
  // Advanced / etc.) is active in the inner tabs.  Reset when
  // the strategy changes.
  const [paramGroupTab, setParamGroupTab] = useState<string>('')

  useEffect(() => {
    const nextSource = initialSourceCode || ''
    const nextSlug = initialSlug || '_backtest_unified'
    setSourceCode(nextSource)
    setSlug(nextSlug)
    // Reset param overrides to the new strategy's declared defaults
    // when the parent swaps the strategy.  Otherwise overrides set
    // for ``stat_arb`` would carry into a backtest of
    // ``tail_end_carry``, which would silently re-key by name and
    // blow up the run.
    setParamOverrides({ ...(initialConfig || {}) })
    setParamGroupTab('')
    if (nextSlug && nextSlug !== '_backtest_unified') {
      setJustLoadedSlug(nextSlug)
      const handle = window.setTimeout(() => setJustLoadedSlug(null), 1600)
      return () => window.clearTimeout(handle)
    }
    return undefined
  }, [initialSourceCode, initialSlug, initialConfig])

  // Build the field-group structure the StrategyConfigForm consumes.
  // Empty groups list ⇒ panel renders the empty-state message.
  // Computed every render — cheap (the schema is small) and always
  // tracks the active strategy.
  const paramFieldGroups: StrategyParamGroup[] = useMemo(() => {
    const fields = Array.isArray(initialParamSchema?.param_fields)
      ? (initialParamSchema!.param_fields as Array<Record<string, unknown>>)
      : []
    return groupStrategyParamFields(fields)
  }, [initialParamSchema])

  // Sync the active group tab to the first available group when
  // the schema (or its grouping) changes.
  useEffect(() => {
    if (paramFieldGroups.length === 0) return
    if (paramFieldGroups.some((g) => g.key === paramGroupTab)) return
    setParamGroupTab(paramFieldGroups[0].key)
  }, [paramFieldGroups, paramGroupTab])

  // Detect when the operator has actually moved a knob away from
  // the strategy's declared default — drives the "modified" badge
  // + the Reset button's disabled state.
  const paramsDirty = useMemo(() => {
    const base = initialConfig || {}
    const baseKeys = Object.keys(base)
    const overrideKeys = Object.keys(paramOverrides)
    if (baseKeys.length !== overrideKeys.length) return true
    for (const key of baseKeys) {
      if (JSON.stringify(base[key]) !== JSON.stringify(paramOverrides[key])) return true
    }
    return false
  }, [initialConfig, paramOverrides])

  const handleResetParams = () => {
    setParamOverrides({ ...(initialConfig || {}) })
  }

  // ───────── Param iteration (LLM-driven autoresearch loop) ─────────
  // Streams iterations of the unified backtest with the LLM proposing
  // overrides against the strategy's declared param_schema.  Drives
  // the same autoresearch service the /api/autoresearch/strategy/{id}
  // /params/stream SSE endpoint exposes, displayed inline below the
  // params panel so the operator never leaves the studio.
  type IterationDecision = {
    iteration: number
    decision: string
    new_score?: number
    score_delta?: number
    best_score?: number
    changed_params?: Record<string, unknown> | null
    reasoning?: string
    duration_seconds?: number
    no_improve_streak?: number
  }
  type IterationProposal = {
    iteration: number
    proposed_changes: Record<string, unknown>
    reasoning: string
    confidence: number
  }
  const [iterRunning, setIterRunning] = useState(false)
  const [iterStarted, setIterStarted] = useState(false)
  // Progressive-disclosure for the footer iterate config expander.
  // Closed by default — clicking the footer "Iterate" button toggles
  // it open to reveal target_score / max_iters / no-improve /
  // mandate / auto-apply, and a "Start iteration" commit button.
  // Auto-opens on first hover / first time a strategy with params is
  // selected so the operator finds it; auto-closes when a run
  // starts so the live status pill replaces it.
  const [iterConfigOpen, setIterConfigOpen] = useState(false)
  const [iterError, setIterError] = useState<string | null>(null)
  const [iterBaselineScore, setIterBaselineScore] = useState<number | null>(null)
  const [iterBestScore, setIterBestScore] = useState<number | null>(null)
  const [iterIteration, setIterIteration] = useState(0)
  const [iterMaxIterations, setIterMaxIterations] = useState(50)
  const [iterTargetScore, setIterTargetScore] = useState<string>('')
  const [iterMaxNoImprove, setIterMaxNoImprove] = useState<string>('10')
  const [iterMandate, setIterMandate] = useState<string>('')
  const [iterAutoApply, setIterAutoApply] = useState<boolean>(false)
  const [iterDecisions, setIterDecisions] = useState<IterationDecision[]>([])
  const [iterLastProposal, setIterLastProposal] = useState<IterationProposal | null>(null)
  const [iterEarlyStopReason, setIterEarlyStopReason] = useState<string | null>(null)
  const [iterDoneSummary, setIterDoneSummary] = useState<{
    total_iterations: number
    best_score: number
    baseline_score: number
    improvement: number
    target_reached: boolean
    early_stop_reason: string | null
  } | null>(null)
  const iterAbortRef = useRef<AbortController | null>(null)

  const iterAvailable = paramFieldGroups.length > 0 && Boolean(initialStrategyId)

  const handleStartIteration = () => {
    if (!initialStrategyId) {
      setIterError(t('backtestStudio.iteratePickStrategy'))
      return
    }
    // Reset run state.
    setIterDecisions([])
    setIterLastProposal(null)
    setIterError(null)
    setIterBaselineScore(null)
    setIterBestScore(null)
    setIterIteration(0)
    setIterEarlyStopReason(null)
    setIterDoneSummary(null)
    setIterRunning(true)
    setIterStarted(true)
    // Collapse the config expander so the live status row + log
    // card become the focal point during the run.  The user can
    // re-open it after Stop or completion to tweak + relaunch.
    setIterConfigOpen(false)

    const body: StrategyParamsStartBody = {
      max_iterations: Math.max(1, Math.min(500, parseInt(String(iterMaxIterations), 10) || 50)),
      auto_apply: iterAutoApply,
    }
    const target = parseFloat(iterTargetScore)
    if (Number.isFinite(target)) body.target_score = target
    const noImp = parseInt(iterMaxNoImprove, 10)
    if (Number.isFinite(noImp) && noImp > 0) body.max_no_improvement = noImp
    if (iterMandate.trim()) body.mandate = iterMandate.trim()

    const ctrl = new AbortController()
    iterAbortRef.current = ctrl

    streamStrategyParamsAutoresearchExperiment(
      initialStrategyId,
      (evt) => {
        const data = (evt.data || {}) as Record<string, unknown>
        switch (evt.event) {
          case 'experiment_start':
            setIterBaselineScore(typeof data.baseline_score === 'number' ? data.baseline_score : null)
            setIterBestScore(typeof data.baseline_score === 'number' ? data.baseline_score : null)
            break
          case 'iteration_start':
            setIterIteration(typeof data.iteration === 'number' ? data.iteration : 0)
            break
          case 'proposal':
            setIterLastProposal({
              iteration: Number(data.iteration ?? 0),
              proposed_changes: (data.proposed_changes as Record<string, unknown>) || {},
              reasoning: String(data.reasoning || ''),
              confidence: Number(data.confidence ?? 0),
            })
            break
          case 'decision': {
            const dec: IterationDecision = {
              iteration: Number(data.iteration ?? 0),
              decision: String(data.decision || 'reverted'),
              new_score: typeof data.new_score === 'number' ? data.new_score : undefined,
              score_delta: typeof data.score_delta === 'number' ? data.score_delta : undefined,
              best_score: typeof data.best_score === 'number' ? data.best_score : undefined,
              changed_params: (data.changed_params as Record<string, unknown> | null) ?? null,
              reasoning: String(data.reasoning || ''),
              duration_seconds: typeof data.duration_seconds === 'number' ? data.duration_seconds : undefined,
              no_improve_streak: typeof data.no_improve_streak === 'number' ? data.no_improve_streak : undefined,
            }
            setIterDecisions((prev) => [dec, ...prev].slice(0, 100))
            if (typeof dec.best_score === 'number') setIterBestScore(dec.best_score)
            // If auto_apply landed a kept change AND we read it back
            // into our local override panel, the user sees the new
            // values immediately.
            if (dec.decision === 'kept' && iterAutoApply && dec.changed_params) {
              setParamOverrides((prev) => ({ ...prev, ...dec.changed_params! }))
            }
            break
          }
          case 'done':
            setIterDoneSummary({
              total_iterations: Number(data.total_iterations ?? 0),
              best_score: Number(data.best_score ?? 0),
              baseline_score: Number(data.baseline_score ?? 0),
              improvement: Number(data.improvement ?? 0),
              target_reached: Boolean(data.target_reached),
              early_stop_reason: (data.early_stop_reason as string | null) ?? null,
            })
            setIterEarlyStopReason((data.early_stop_reason as string | null) ?? null)
            setIterRunning(false)
            iterAbortRef.current = null
            break
          case 'error':
            setIterError(String(data.error || t('autoresearch.unknownError')))
            setIterRunning(false)
            iterAbortRef.current = null
            break
        }
      },
      () => {
        setIterRunning(false)
        iterAbortRef.current = null
      },
      (err) => {
        setIterError(err)
        setIterRunning(false)
        iterAbortRef.current = null
      },
      ctrl.signal,
      body,
    )
  }

  const handleStopIteration = () => {
    iterAbortRef.current?.abort()
    iterAbortRef.current = null
    if (initialStrategyId) {
      // Server-side stop signal — the loop checks this between
      // iterations so the in-flight backtest still completes.
      stopStrategyParamsAutoresearchExperiment(initialStrategyId).catch(() => undefined)
    }
    setIterRunning(false)
  }
  const [initialCapital, setInitialCapital] = useState<string>('1000')
  const [submitP50, setSubmitP50] = useState<string>('')
  const [submitP95, setSubmitP95] = useState<string>('')
  const [seed, setSeed] = useState<string>('')
  const [impactBps, setImpactBps] = useState<string>('')
  const [makerRebateBps, setMakerRebateBps] = useState<string>('')
  // Default window: 1 day for fast iteration.  Earlier default of 7d
  // routinely produced 1.5M-snapshot replays that took 30s+ wall-clock
  // on the API thread (now mitigated by the worker process, but still
  // a slow first-run experience).  Operators tune up to 7/30 via
  // preset chips.
  const [windowDays, setWindowDays] = useState<string>('1')

  // Imported provider dataset(s) the operator picked from Data Lab →
  // Providers.  When non-empty the backend resolves these into the
  // (token_ids, start, end) scope for the run; window/days/start/end
  // controls above are ignored.  Mutually exclusive with session_id
  // (recording session wins on the backend if both are present).
  const [providerDatasetIds, setProviderDatasetIds] = useState<string[]>([])
  // Center-pane subtab.  Three coherent groupings + a small status
  // ribbon-equivalent so the workbench doesn't scroll-and-pray.
  const [centerTab, setCenterTab] = useState<'performance' | 'fill_quality' | 'robustness' | 'portfolio'>('performance')

  // Active run.  State survives navigation across the app via
  // localStorage: the run_id is the canonical pointer (the result
  // payload is fetched on mount via getBacktestRun).  This is the
  // institutional pattern — ID in storage, fresh fetch on mount,
  // never trust a stale serialized payload that may be out of sync
  // with the backend.
  const [activeRun, setActiveRun] = useState<UnifiedBacktestResult | null>(null)
  // Set when the user clicks Run.  Persists immediately (BEFORE the
  // run finishes) so that if the user navigates away mid-flight, we
  // can find the row in the recent-runs list once it lands.  Cleared
  // on success or after a 5-min timeout.
  type PendingRun = { startedAt: number; strategySlug: string }

  const runsQuery = useQuery({
    queryKey: ['backtest', 'runs'],
    queryFn: listBacktestRuns,
    refetchInterval: 5000,
  })

  // ── Cross-navigation run persistence ────────────────────────────────
  //
  // Two storage keys:
  //
  //   hr_backtest_active_run_id      string | null
  //     The completed run currently displayed in the studio.  Set on
  //     run-mutation success and on Recent-Runs row click.  Cleared
  //     when the user explicitly resets.
  //
  //   hr_backtest_pending_run        { startedAt, strategySlug } | null
  //     Marker for an in-flight run.  Set the moment the user clicks
  //     Run (BEFORE the backend finishes), so navigating away mid-
  //     flight doesn't lose the run.  When the user navigates BACK,
  //     we scan the recent-runs list for a row whose started_at >=
  //     this marker AND whose strategy_slug matches AND completed_at
  //     is non-null — that's our run.  Marker times out after 5 min.
  //
  // On mount, if either key has a value, restore the run.  This is
  // why the studio survives tab navigation, page reload, and even
  // browser restart.

  const ACTIVE_RUN_ID_KEY = 'hr_backtest_active_run_id'
  const PENDING_RUN_KEY = 'hr_backtest_pending_run'
  const PENDING_RUN_TIMEOUT_MS = 5 * 60_000

  const persistActiveRunId = (id: string | null) => {
    try {
      if (id) localStorage.setItem(ACTIVE_RUN_ID_KEY, id)
      else localStorage.removeItem(ACTIVE_RUN_ID_KEY)
    } catch {
      /* quota / disabled storage — silent */
    }
  }
  const persistPendingRun = (pending: PendingRun | null) => {
    try {
      if (pending) localStorage.setItem(PENDING_RUN_KEY, JSON.stringify(pending))
      else localStorage.removeItem(PENDING_RUN_KEY)
    } catch {
      /* silent */
    }
  }
  const readPendingRun = (): PendingRun | null => {
    try {
      const raw = localStorage.getItem(PENDING_RUN_KEY)
      if (!raw) return null
      const parsed = JSON.parse(raw)
      if (!parsed || typeof parsed.startedAt !== 'number') return null
      // Expire stale markers.
      if (Date.now() - parsed.startedAt > PENDING_RUN_TIMEOUT_MS) {
        localStorage.removeItem(PENDING_RUN_KEY)
        return null
      }
      return parsed as PendingRun
    } catch {
      return null
    }
  }
  const readActiveRunId = (): string | null => {
    try {
      return localStorage.getItem(ACTIVE_RUN_ID_KEY)
    } catch {
      return null
    }
  }

  // Live state — independent of any single run.  Drives the top
  // ribbon so the workbench shows what the platform is doing RIGHT
  // NOW, not just what the most-recently-loaded run captured.
  const liveFillModelQuery = useQuery({
    queryKey: ['fill-model', 'active', 'pooled'],
    queryFn: () => getActiveFillModel('pooled'),
    refetchInterval: 30_000,
  })
  const liveLatencyQuery = useQuery({
    queryKey: ['fill-model', 'latency'],
    queryFn: getLatencyDistribution,
    refetchInterval: 15_000,
  })
  const liveConstantsQuery = useQuery({
    queryKey: ['fill-model', 'empirical-constants'],
    queryFn: getEmpiricalConstants,
    refetchInterval: 30_000,
  })
  const liveDecompositionQuery = useQuery({
    queryKey: ['fill-model', 'decomposition'],
    queryFn: () => getDecompositionSummary(24),
    refetchInterval: 30_000,
  })

  // Triangulation: backtest vs shadow vs live PnL for THIS strategy
  // over the last 30 days.  Big divergence between any two means the
  // fill model is the prime suspect.  Only loaded when we have a slug.
  //
  // Follow the *active run's* strategy_slug, not the static
  // ``initialSlug`` prop.  Otherwise clicking a recent run in the
  // sidebar (different strategy from whatever the studio mounted
  // with) keeps querying triangulation against the original slug
  // and the panel reads "0 live trades" even when 6,000+ exist for
  // the run's actual strategy.
  const triangSlug = activeRun?.strategy_slug || initialSlug || ''
  const triangulationQuery = useQuery({
    queryKey: ['triangulation', triangSlug],
    queryFn: () => getTriangulation(triangSlug, 30),
    enabled: Boolean(triangSlug && triangSlug !== '_backtest_unified' && triangSlug !== '_research'),
    refetchInterval: 60_000,
  })

  // Portfolio correlation: pulls live cross-strategy PnL correlation
  // matrix over the last 30 days.  Auto-refresh every 5 minutes.
  const portfolioCorrelationQuery = useQuery({
    queryKey: ['portfolio-correlation', 30],
    queryFn: () => getPortfolioCorrelation(30, 5),
    refetchInterval: 300_000,
  })

  // Live-vs-backtest drift monitor: surfaces strategies whose live
  // performance has materially diverged from their most recent
  // backtest.  Refreshes every 5 minutes.
  const driftQuery = useQuery({
    queryKey: ['backtest-drift', 30],
    queryFn: () => getDriftMonitor(30),
    refetchInterval: 300_000,
  })

  // CPCV: opt-in, user-triggered.  Heavy (15+ backtest runs).
  const [cpcvResult, setCpcvResult] = useState<CPCVResult | null>(null)
  const [cpcvNFolds, setCpcvNFolds] = useState<number>(6)
  const [cpcvKTest, setCpcvKTest] = useState<number>(2)
  const cpcvMutation = useMutation({
    mutationFn: runCPCV,
    onSuccess: (data) => setCpcvResult(data),
  })

  // Latency Monte Carlo: opt-in, user-triggered.
  const [latencyMcResult, setLatencyMcResult] = useState<MonteCarloLatencyResult | null>(null)
  const latencyMcMutation = useMutation({
    mutationFn: runMonteCarloLatency,
    onSuccess: (data) => setLatencyMcResult(data),
  })

  // Walk-forward: not auto-run.  User triggers via the panel button.
  const [walkForwardResult, setWalkForwardResult] = useState<WalkForwardResult | null>(null)
  const [walkForwardMode, setWalkForwardMode] = useState<'anchored' | 'rolling'>('anchored')
  const [walkForwardFolds, setWalkForwardFolds] = useState<number>(6)
  const walkForwardMutation = useMutation({
    mutationFn: runWalkForward,
    onSuccess: (data) => setWalkForwardResult(data),
  })

  const handleWalkForward = () => {
    if (!sourceCode.trim() || sourceCode.trim().length < 10) return
    // Default test window: last 14 days.
    const end = new Date()
    const start = new Date(end.getTime() - 14 * 24 * 60 * 60 * 1000)
    walkForwardMutation.mutate({
      source_code: sourceCode,
      slug: slug,
      // ``paramOverrides`` is initialized from ``initialConfig`` and
      // edited in-place by the dynamic Strategy parameters panel; it
      // is the single source of truth for run-time strategy config.
      config: paramOverrides,
      start: start.toISOString(),
      end: end.toISOString(),
      initial_capital_usd: parseFloat(initialCapital) || 1000,
      mode: walkForwardMode,
      n_folds: walkForwardFolds,
      train_ratio: 0.5,
      seed: seed ? parseInt(seed, 10) : undefined,
      concurrency: 2,
    })
  }

  const loadRunMutation = useMutation({
    mutationFn: getBacktestRun,
    onSuccess: (data) => {
      setActiveRun(data)
      persistActiveRunId(data.run_id)
    },
  })

  // ── Async-by-default run flow ─────────────────────────────────────
  //
  // The new pipeline:
  //
  //   click Run → POST /backtest/runs/enqueue → returns run_id with
  //   status='queued'.  The backtest worker process picks it up off
  //   the discovery plane and chews on it; the API process is never
  //   blocked.
  //
  //   pendingRunId tracks the in-flight run.  A useQuery polls
  //   /backtest/runs/{id}/status every 1.5s while the run is alive,
  //   feeding the progress bar.
  //
  //   When status flips to 'completed', we fetch the full result
  //   blob via getBacktestRun and promote it to activeRun.  On
  //   'failed' / 'cancelled', surface the error.

  const PENDING_RUN_ID_KEY = 'hr_backtest_pending_run_id'
  const readPendingRunId = (): string | null => {
    try {
      return localStorage.getItem(PENDING_RUN_ID_KEY)
    } catch {
      return null
    }
  }
  const writePendingRunId = (id: string | null) => {
    try {
      if (id) localStorage.setItem(PENDING_RUN_ID_KEY, id)
      else localStorage.removeItem(PENDING_RUN_ID_KEY)
    } catch {
      /* silent */
    }
  }
  const [pendingRunId, setPendingRunIdState] = useState<string | null>(
    () => readPendingRunId(),
  )
  const setPendingRunId = (id: string | null) => {
    writePendingRunId(id)
    setPendingRunIdState(id)
  }

  const runMutation = useMutation({
    mutationFn: enqueueBacktest,
    onMutate: (vars) => {
      // Legacy marker for slow-restoration paths (kept for back-compat
      // with any tab that might land on a pre-pendingRunId build).
      persistPendingRun({
        startedAt: Date.now(),
        strategySlug: vars.slug || initialSlug || '_backtest_unified',
      })
    },
    onSuccess: (data) => {
      setPendingRunId(data.run_id)
      persistPendingRun(null)
      queryClient.invalidateQueries({ queryKey: ['backtest', 'runs'] })
    },
    onError: () => {
      persistPendingRun(null)
    },
  })

  // Poll the status of the in-flight run.  Stops polling automatically
  // when the run reaches a terminal state.
  const runStatusQuery = useQuery<BacktestRunStatus>({
    queryKey: ['backtest', 'run-status', pendingRunId],
    queryFn: () => getBacktestRunStatus(pendingRunId!),
    enabled: !!pendingRunId,
    // 1.5s while running; the engine writes progress at ~1s cadence
    // so this keeps the UI fresh without slamming the DB.
    refetchInterval: (q) => {
      const data = q.state.data as BacktestRunStatus | undefined
      if (!data) return 1500
      if (['queued', 'running'].includes(data.status)) return 1500
      // Terminal state — stop polling.
      return false
    },
  })

  // When the polled status flips to a terminal state, promote / clear.
  useEffect(() => {
    const data = runStatusQuery.data
    if (!data || !pendingRunId) return
    if (data.status === 'completed' || data.status === 'ok') {
      // Fetch the full result blob and promote.
      loadRunMutation.mutate(pendingRunId)
      setPendingRunId(null)
      queryClient.invalidateQueries({ queryKey: ['backtest', 'runs'] })
    } else if (data.status === 'failed' || data.status === 'cancelled') {
      // Surface the error / cancel state by promoting the row's
      // result_json (which carries the traceback for failures); the
      // UI's error banner in the header will render it.
      loadRunMutation.mutate(pendingRunId)
      setPendingRunId(null)
      queryClient.invalidateQueries({ queryKey: ['backtest', 'runs'] })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runStatusQuery.data?.status])

  const cancelMutation = useMutation({
    mutationFn: cancelBacktestRun,
  })

  // ── Restore on mount ────────────────────────────────────────────────
  //
  // Two paths:
  //
  // 1. activeRunId in localStorage → fetch the completed run.  This
  //    is the common case: user backtested, navigated away, came
  //    back.  The fetch happens immediately and the studio is fully
  //    populated within ~100ms.
  //
  // 2. pendingRun marker in localStorage → the user navigated away
  //    while the backtest was still running on the backend.  We
  //    don't know the run_id yet.  Watch the recent-runs list for a
  //    matching just-completed row; once seen, promote it.
  //
  // Both effects run only when activeRun is null — clicking a row in
  // the Recent Runs sidebar already handles promotion via
  // loadRunMutation.

  useEffect(() => {
    if (activeRun) return
    const stored = readActiveRunId()
    if (!stored) return
    // Fire-and-forget — onSuccess sets activeRun.  If the run was
    // deleted server-side, the 4xx surfaces in mutation.error and
    // the user sees an empty state; clear stale id so the next mount
    // doesn't keep re-fetching a 404.
    loadRunMutation.mutate(stored, {
      onError: () => persistActiveRunId(null),
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (activeRun) return
    const pending = readPendingRun()
    if (!pending) return
    const runs = runsQuery.data || []
    // Match: started after the marker (with 30s clock-skew slack),
    // strategy slug matches, AND the run has completed (completed_at
    // is set).  Pick the most recent match if multiple.
    const candidate = runs
      .filter((r) => {
        if (!r.completed_at) return false
        if (r.strategy_slug && pending.strategySlug && r.strategy_slug !== pending.strategySlug)
          return false
        const startedMs = new Date(r.started_at).getTime()
        return startedMs >= pending.startedAt - 30_000
      })
      .sort((a, b) => new Date(b.started_at).getTime() - new Date(a.started_at).getTime())[0]
    if (candidate) {
      persistPendingRun(null)
      persistActiveRunId(candidate.run_id)
      loadRunMutation.mutate(candidate.run_id)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runsQuery.data, activeRun])

  const errorMessage = useMemo(() => {
    const err = runMutation.error as { response?: { data?: { detail?: string } }; message?: string } | undefined
    if (!err) return null
    return err.response?.data?.detail || err.message || t('autoresearch.unknownError')
  }, [runMutation.error, t])

  const handleRun = () => {
    if (!sourceCode.trim() || sourceCode.trim().length < 10) return
    // Window is "last N days" measured from now.  Skip when the
    // operator left it blank or set to a non-positive number, in
    // which case the backend's 7d default applies.
    const days = parseFloat(windowDays || '0')
    let startIso: string | undefined
    let endIso: string | undefined
    if (Number.isFinite(days) && days > 0) {
      const end = new Date()
      const start = new Date(end.getTime() - days * 24 * 60 * 60 * 1000)
      startIso = start.toISOString()
      endIso = end.toISOString()
    }
    runMutation.mutate({
      source_code: sourceCode,
      slug,
      // ``paramOverrides`` carries any in-rail edits to the strategy's
      // declared knobs (default-merged at mount in initialConfig).  It
      // lands on ``run_execution_backtest(config=...)`` which feeds it
      // into the StrategyLoader, which calls strategy.configure() —
      // the same path the live trader takes when a tune-subtab edit
      // is saved.  Every gate in apply_platform_decision_gates +
      // strategy.evaluate() reads from strategy.config so overrides
      // genuinely drive the backtest run.
      config: paramOverrides,
      initial_capital_usd: parseFloat(initialCapital) || 1000,
      start: startIso,
      end: endIso,
      // Provider datasets win over the (start, end) controls — the
      // backend resolves them into the union of (token_ids, window).
      provider_dataset_ids: providerDatasetIds.length > 0 ? providerDatasetIds : undefined,
      submit_p50_ms: submitP50 ? parseFloat(submitP50) : undefined,
      submit_p95_ms: submitP95 ? parseFloat(submitP95) : undefined,
      seed: seed ? parseInt(seed, 10) : undefined,
      counterfactual_sample_size: 8,
      ensemble_sample_size: 8,
      impact_strength_bps: impactBps ? parseFloat(impactBps) : undefined,
      maker_rebate_bps: makerRebateBps ? parseFloat(makerRebateBps) : undefined,
    })
  }

  const exec = activeRun?.execution
  // Prefer the run's snapshot when a run is loaded (it's the model
  // that ACTUALLY drove that backtest).  Fall back to the live
  // platform state so the workbench is informative even pre-run.
  const liveFillModel = liveFillModelQuery.data
  const liveLatency = liveLatencyQuery.data
  const liveConstants = liveConstantsQuery.data
  const liveDecomp = liveDecompositionQuery.data
  const fillModel = activeRun?.fill_model ?? (liveFillModel ? {
    loaded: true,
    family: liveFillModel.family,
    strata_key: liveFillModel.strata_key,
    n_events: liveFillModel.n_events,
    concordance_index: liveFillModel.concordance_index,
    coefficients: liveFillModel.coefficients,
    feature_means: liveFillModel.feature_means,
    feature_stds: liveFillModel.feature_stds,
    notes: liveFillModel.notes,
  } : { loaded: false })
  const constants = activeRun?.empirical_constants ?? (liveConstants
    ? {
        measured: liveConstants.measured,
        sample_count: liveConstants.sample_count,
        measured_at_epoch: liveConstants.measured_at_epoch,
        notes: liveConstants.notes,
        values: liveConstants.values,
      }
    : null)
  const latency = activeRun?.latency ?? (liveLatency ?? null)
  const decomp = activeRun?.decomposition ?? (liveDecomp ?? null)

  const totalReturnTone =
    exec && exec.total_return_pct >= 5
      ? 'good'
      : exec && exec.total_return_pct >= 0
        ? 'neutral'
        : 'bad'
  const sharpeTone =
    exec?.sharpe?.value != null
      ? exec.sharpe.value >= 1.5
        ? 'good'
        : exec.sharpe.value >= 0.5
          ? 'warn'
          : 'bad'
      : 'neutral'
  const ddTone =
    exec && Math.abs(exec.max_drawdown_pct) <= 10
      ? 'good'
      : exec && Math.abs(exec.max_drawdown_pct) <= 25
        ? 'warn'
        : 'bad'

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* TOP STRIP — global ribbon, ALWAYS visible regardless of run state */}
      <div className="border-b border-border/50 bg-card/30 px-3 py-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-amber-300" />
            <div>
              <div className="text-sm font-semibold">{t('backtestStudio.studioTitle')}</div>
              <div className="text-[11px] text-muted-foreground">
                {t('backtestStudio.studioSubtitle')}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {fillModel?.loaded ? (
              <Badge
                className={cn(
                  'text-[10px]',
                  fillModel.family === 'cox_ph'
                    ? 'bg-emerald-500/10 text-emerald-300'
                    : 'bg-amber-500/10 text-amber-300',
                )}
              >
                {t('backtestStudio.fillModelEvents', { family: fillModel.family, n: fillModel.n_events?.toLocaleString() })}
              </Badge>
            ) : (
              <Badge variant="outline" className="text-[10px]">
                {t('backtestStudio.noFillModelLoaded')}
              </Badge>
            )}
            {latency ? (
              <Badge variant="outline" className="text-[10px] font-mono">
                {t('backtestStudio.latencyP50P95', { p50: Math.round(latency.p50_ms), p95: Math.round(latency.p95_ms) })}
              </Badge>
            ) : null}
            {constants?.measured ? (
              <Badge className="bg-emerald-500/10 text-emerald-300 text-[10px]">
                {t('backtestStudio.empiricalConstantsLive')}
              </Badge>
            ) : (
              <Badge variant="outline" className="text-[10px]">
                {t('backtestStudio.empiricalConstantsDefault')}
              </Badge>
            )}
          </div>
        </div>
      </div>

      {/* MAIN — 3-pane workbench */}
      <div className="flex flex-1 min-h-0">
        {/* LEFT RAIL — controls + history.  Vertical layout:
              [strategy source pill]      ← always visible (identity)
              [subtab bar]                ← Setup | Parameters | Runs
              [active subtab content]     ← scrolls independently
              [Run backtest button]       ← always visible (action)
            Each subtab's body is its own ScrollArea so a tall
            Parameters panel doesn't push the Run button off-screen
            (the bug that triggered this refactor — the rail used to
            be one flat scroll and the params + iterate UI got cut). */}
        <div className="flex w-[340px] shrink-0 flex-col border-r border-border/50 bg-background/40">
          {/* Strategy identity — small persistent header. */}
          <div className="border-b border-border/50 px-3 py-2">
            <div
              className={cn(
                'flex items-center justify-between gap-2 rounded-sm border px-2 py-1.5 transition-colors duration-700',
                justLoadedSlug
                  ? 'border-amber-400/60 bg-amber-500/10'
                  : 'border-border/40 bg-background/40',
              )}
              title={t('backtestStudio.strategySourceTooltip')}
            >
              <div className="min-w-0 flex-1">
                <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
                  {t('backtestStudio.strategySource')}
                </div>
                <div className="truncate font-mono text-[11px] text-foreground">
                  {strategyLabel || slug || t('backtestStudio.noStrategy')}
                </div>
              </div>
              <div className="shrink-0 text-right">
                <div className="font-mono text-[10px] text-muted-foreground">
                  {sourceCode.length > 0
                    ? t('backtestStudio.charsCount', { n: sourceCode.length.toLocaleString() })
                    : t('backtestStudio.noSource')}
                </div>
                {justLoadedSlug ? (
                  <div className="text-[9px] uppercase tracking-wide text-amber-300">
                    {t('backtestStudio.justLoaded')}
                  </div>
                ) : null}
              </div>
            </div>
          </div>

          {/* Subtab bar.  Three tabs split the rail by the kind of
              decision the operator is making: configure the run
              (Setup), tune the strategy (Parameters), pick a prior
              result to view (Runs).  Counts hint at how much content
              is in each — Parameters shows the dynamic field count
              + a 'modified' badge when overrides differ from
              defaults; Runs shows the cached run count. */}
          <Tabs
            value={leftTab}
            onValueChange={(v) => setLeftTab(v as 'setup' | 'parameters' | 'runs')}
            className="flex flex-1 min-h-0 flex-col"
          >
            <TabsList className="h-8 w-full justify-start gap-0 rounded-none border-b border-border/50 bg-transparent p-0">
              <TabsTrigger
                value="setup"
                className="flex-1 h-8 rounded-none data-[state=active]:bg-background/70 data-[state=active]:shadow-none border-b-2 border-transparent data-[state=active]:border-cyan-500 text-[11px]"
              >
                {t('backtestStudio.tabSetup')}
              </TabsTrigger>
              <TabsTrigger
                value="parameters"
                className="flex-1 h-8 rounded-none data-[state=active]:bg-background/70 data-[state=active]:shadow-none border-b-2 border-transparent data-[state=active]:border-cyan-500 text-[11px] gap-1"
              >
                <span>{t('backtestStudio.tabParameters')}</span>
                {paramFieldGroups.length > 0 ? (
                  <span className="text-[9px] text-muted-foreground">
                    {paramFieldGroups.reduce((sum, g) => sum + g.fields.length, 0)}
                  </span>
                ) : null}
                {paramsDirty ? (
                  <span className="h-1.5 w-1.5 rounded-full bg-amber-400" title={t('backtestStudio.overridesModified')} />
                ) : null}
                {iterRunning ? (
                  <span className="h-1.5 w-1.5 rounded-full bg-cyan-400 animate-pulse" title={t('backtestStudio.iterationRunning')} />
                ) : null}
              </TabsTrigger>
              <TabsTrigger
                value="runs"
                className="flex-1 h-8 rounded-none data-[state=active]:bg-background/70 data-[state=active]:shadow-none border-b-2 border-transparent data-[state=active]:border-cyan-500 text-[11px] gap-1"
              >
                <span>{t('backtestStudio.tabRuns')}</span>
                {runsQuery.data && runsQuery.data.length > 0 ? (
                  <span className="text-[9px] text-muted-foreground">{runsQuery.data.length}</span>
                ) : null}
              </TabsTrigger>
            </TabsList>

            {/* ─────── Setup tab ─────── */}
            <TabsContent value="setup" className="flex-1 min-h-0 mt-0 overflow-hidden">
              <ScrollArea className="h-full">
                <div className="px-3 py-3 space-y-2">
                  {/* Provider dataset picker — when set, the run uses
                      the selected datasets' (token_ids, window) instead
                      of the Window field below.  Imports happen in
                      Data Lab → Providers (polybacktest etc.). */}
                  <ProviderDatasetSelector
                    selected={providerDatasetIds}
                    onChange={setProviderDatasetIds}
                  />

                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <Label className="text-[10px] uppercase tracking-wide text-muted-foreground">
                        {t('backtestStudio.labelCapital')}
                      </Label>
                      <Input
                        value={initialCapital}
                        onChange={(e) => setInitialCapital(e.target.value)}
                        className="h-7 text-xs"
                      />
                    </div>
                    <div>
                      <Label className="text-[10px] uppercase tracking-wide text-muted-foreground">
                        {t('backtestStudio.labelSeed')}
                      </Label>
                      <Input
                        value={seed}
                        onChange={(e) => setSeed(e.target.value)}
                        placeholder={t('backtestStudio.labelSeedAuto')}
                        className="h-7 text-xs"
                      />
                    </div>
                    <div>
                      <Label className="text-[10px] uppercase tracking-wide text-muted-foreground">
                        {t('backtestStudio.labelLatencyP50')}
                      </Label>
                      <Input
                        value={submitP50}
                        onChange={(e) => setSubmitP50(e.target.value)}
                        placeholder={t('backtestStudio.labelMeasured')}
                        className="h-7 text-xs"
                      />
                    </div>
                    <div>
                      <Label className="text-[10px] uppercase tracking-wide text-muted-foreground">
                        {t('backtestStudio.labelLatencyP95')}
                      </Label>
                      <Input
                        value={submitP95}
                        onChange={(e) => setSubmitP95(e.target.value)}
                        placeholder={t('backtestStudio.labelMeasured')}
                        className="h-7 text-xs"
                      />
                    </div>
                    <div>
                      <Label
                        className="text-[10px] uppercase tracking-wide text-muted-foreground"
                        title={t('backtestStudio.windowTooltip')}
                      >
                        {t('backtestStudio.labelWindowDays')}
                      </Label>
                      <Input
                        value={windowDays}
                        onChange={(e) => setWindowDays(e.target.value)}
                        placeholder="1"
                        className="h-7 text-xs"
                      />
                      {/* Preset chips — illuminated defaults that
                          map to wall-clock expectations.  Quick is
                          the recommended first run; Standard for
                          deciding whether to deploy; Thorough for
                          a final pre-production sanity check. */}
                      <div className="mt-1 flex items-center gap-1">
                        {([
                          ['1', t('backtestStudio.presetQuick'), t('backtestStudio.presetQuickEta')],
                          ['7', t('backtestStudio.presetStandard'), t('backtestStudio.presetStandardEta')],
                          ['30', t('backtestStudio.presetThorough'), t('backtestStudio.presetThoroughEta')],
                        ] as const).map(([val, label, eta]) => (
                          <button
                            key={val}
                            type="button"
                            onClick={() => setWindowDays(val)}
                            title={t('backtestStudio.presetTitle', { label, val, eta })}
                            className={cn(
                              'rounded-sm border px-1.5 py-0.5 text-[9px] font-medium transition-colors',
                              windowDays === val
                                ? 'border-violet-500/50 bg-violet-500/10 text-violet-700 dark:text-violet-300'
                                : 'border-border/40 bg-card/40 text-muted-foreground hover:border-border/60 hover:text-foreground',
                            )}
                          >
                            {label}
                            <span className="ml-1 opacity-60">{eta}</span>
                          </button>
                        ))}
                      </div>
                    </div>
                    <div>
                      <Label className="text-[10px] uppercase tracking-wide text-muted-foreground" title={t('backtestStudio.impactTooltip')}>
                        {t('backtestStudio.labelImpactBps')}
                      </Label>
                      <Input
                        value={impactBps}
                        onChange={(e) => setImpactBps(e.target.value)}
                        placeholder={t('backtestStudio.placeholderOff')}
                        className="h-7 text-xs"
                      />
                    </div>
                    <div>
                      <Label className="text-[10px] uppercase tracking-wide text-muted-foreground" title={t('backtestStudio.makerRebateTooltip')}>
                        {t('backtestStudio.labelMakerRebate')}
                      </Label>
                      <Input
                        value={makerRebateBps}
                        onChange={(e) => setMakerRebateBps(e.target.value)}
                        placeholder={t('backtestStudio.placeholderOff')}
                        className="h-7 text-xs"
                      />
                    </div>
                  </div>

                  {/* Hint when the strategy carries dynamic
                      params — point operators at the Parameters tab
                      so they don't miss it. */}
                  {paramFieldGroups.length > 0 ? (
                    <button
                      type="button"
                      onClick={() => setLeftTab('parameters')}
                      className="w-full flex items-center justify-between gap-2 px-2.5 py-1.5 text-[10px] rounded-md border border-cyan-500/30 bg-cyan-500/5 hover:bg-cyan-500/10 transition-colors text-cyan-200"
                    >
                      <span className="flex items-center gap-1.5">
                        <Sliders className="h-3 w-3" />
                        {t('backtestStudio.paramsAvailable', { n: paramFieldGroups.reduce((sum, g) => sum + g.fields.length, 0) })}
                      </span>
                      <span className="text-cyan-300">{t('backtestStudio.goToParameters')}</span>
                    </button>
                  ) : null}
                </div>
              </ScrollArea>
            </TabsContent>

            {/* ─────── Parameters tab ─────── */}
            <TabsContent value="parameters" className="flex-1 min-h-0 mt-0 overflow-hidden">
              {paramFieldGroups.length > 0 ? (
                <ScrollArea className="h-full">
                  <div className="px-3 py-3 space-y-2">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[10px] text-muted-foreground">
                        {t('backtestStudio.paramsHint')}{' '}
                        <code className="font-mono text-foreground">{t('backtestStudio.defaultConfig')}</code>.
                      </span>
                      <Button
                        type="button"
                        size="sm"
                        variant="ghost"
                        disabled={!paramsDirty}
                        onClick={handleResetParams}
                        className="h-6 gap-1 px-1.5 text-[10px]"
                        title={t('backtestStudio.resetTooltip')}
                      >
                        <RotateCcw className="h-3 w-3" />
                        {t('backtestStudio.reset')}
                      </Button>
                    </div>

                    {/* Field-group tabs (Signal / Entry / Sizing /
                        Exit / Risk / Advanced).  Each tab content is
                        scrollable but the whole Parameters tab is
                        also wrapped in a parent ScrollArea, so even
                        a 50-field strategy never overflows the rail.

                        Iteration controls (Start/Stop, target_score,
                        mandate, etc.) live in the footer beside the
                        Run backtest button — no longer inside this
                        tab.  The decision log + scoreboard + summary
                        renders BELOW these field tabs (see
                        "Iteration log" card further down) so the
                        operator can review the optimizer's history
                        without leaving Parameters. */}
                    <Tabs value={paramGroupTab} onValueChange={setParamGroupTab}
                      className="flex min-h-0 flex-col">
                      <div className="overflow-x-auto pb-1">
                        <TabsList className="h-auto w-max min-w-full justify-start gap-1 rounded-md border border-border/50 bg-background/60 p-1">
                          {paramFieldGroups.map((group) => (
                            <TabsTrigger
                              key={group.key}
                              value={group.key}
                              className="h-6 gap-1 px-2 text-[10px]"
                            >
                              <span>{group.label}</span>
                              <span className="text-[9px] text-muted-foreground">
                                {group.fields.length}
                              </span>
                            </TabsTrigger>
                          ))}
                        </TabsList>
                      </div>
                      {paramFieldGroups.map((group) => (
                        <TabsContent
                          key={`panel:${group.key}`}
                          value={group.key}
                          className="mt-0 rounded-md border border-border/50 bg-background/65 p-2"
                        >
                          <StrategyConfigForm
                            schema={{ param_fields: group.fields as any[] }}
                            values={paramOverrides}
                            onChange={(next) => setParamOverrides(next)}
                          />
                        </TabsContent>
                      ))}
                    </Tabs>

                    {/* ───────── Iteration log ─────────
                        Renders only after the operator kicks off an
                        iteration from the footer.  The CONTROLS
                        (target_score, max_iters, mandate, auto-apply,
                        Start/Stop) live in the footer alongside
                        Run backtest — see the sticky footer below.
                        This card is a read-only audit view: full
                        scoreboard, last LLM proposal + reasoning,
                        decision log (kept iterations highlighted),
                        and the done summary with early-stop reason. */}
                    {iterStarted ? (
                      <div className="rounded-md border border-cyan-500/30 bg-cyan-500/5 p-2 space-y-1.5">
                        <div className="flex items-center justify-between gap-2">
                          <span className="flex items-center gap-1.5 text-[10px] font-medium text-cyan-300">
                            <Wand2 className="h-3 w-3" />
                            {t('backtestStudio.iterationLog')}
                          </span>
                          {iterRunning ? (
                            <Badge variant="outline" className="h-4 px-1.5 text-[9px] uppercase tracking-wide text-cyan-300 border-cyan-400/40 animate-pulse">
                              {t('backtestStudio.iterRunningStatus', { cur: iterIteration, max: iterMaxIterations })}
                            </Badge>
                          ) : iterDoneSummary ? (
                            <Badge variant="outline" className="h-4 px-1.5 text-[9px] uppercase tracking-wide text-emerald-300 border-emerald-400/40">
                              {iterDoneSummary.target_reached ? t('backtestStudio.iterDoneTarget') : t('backtestStudio.iterDoneCompleted')}
                            </Badge>
                          ) : null}
                        </div>

                        <div className="grid grid-cols-3 gap-1.5 text-[10px]">
                          <div>
                            <div className="text-muted-foreground">{t('backtestStudio.baseline')}</div>
                            <div className="font-mono">
                              {iterBaselineScore !== null ? iterBaselineScore.toFixed(4) : '—'}
                            </div>
                          </div>
                          <div>
                            <div className="text-muted-foreground">{t('backtestStudio.best')}</div>
                            <div className="font-mono text-emerald-400">
                              {iterBestScore !== null ? iterBestScore.toFixed(4) : '—'}
                            </div>
                          </div>
                          <div>
                            <div className="text-muted-foreground flex items-center gap-1">
                              <Target className="h-2.5 w-2.5" />
                              {t('backtestStudio.target')}
                            </div>
                            <div className="font-mono">
                              {iterTargetScore || '—'}
                            </div>
                          </div>
                        </div>

                        {iterLastProposal ? (
                          <div className="text-[10px] border-t border-border/30 pt-1">
                            <div className="text-muted-foreground">
                              {t('backtestStudio.lastProposal', { n: iterLastProposal.iteration, conf: iterLastProposal.confidence.toFixed(2) })}
                            </div>
                            <div className="text-foreground italic line-clamp-2">
                              {iterLastProposal.reasoning || t('backtestStudio.noReasoning')}
                            </div>
                          </div>
                        ) : null}

                        {iterDecisions.length > 0 ? (
                          <div className="max-h-48 overflow-y-auto border-t border-border/30 pt-1 space-y-0.5">
                            {iterDecisions.map((d, idx) => (
                              <div
                                key={`${d.iteration}-${idx}`}
                                className={cn(
                                  'flex items-start gap-1 text-[10px] font-mono px-1 py-0.5 rounded',
                                  d.decision === 'kept'
                                    ? 'bg-emerald-500/10 text-emerald-200'
                                    : 'text-muted-foreground'
                                )}
                              >
                                <span className="w-8 shrink-0">#{d.iteration}</span>
                                <span className="w-12 shrink-0">{d.decision}</span>
                                <span className="w-14 shrink-0">
                                  {typeof d.new_score === 'number' ? d.new_score.toFixed(4) : '—'}
                                </span>
                                <span className="w-14 shrink-0">
                                  {typeof d.score_delta === 'number'
                                    ? (d.score_delta > 0 ? '+' : '') + d.score_delta.toFixed(4)
                                    : ''}
                                </span>
                                <span className="truncate">
                                  {d.changed_params && Object.keys(d.changed_params).length > 0
                                    ? Object.keys(d.changed_params).slice(0, 3).join(', ')
                                    : (d.reasoning || '').slice(0, 60)}
                                </span>
                              </div>
                            ))}
                          </div>
                        ) : null}

                        {iterDoneSummary ? (
                          <div className="border-t border-border/30 pt-1 text-[10px]">
                            <div className="flex items-center gap-2">
                              <CheckCircle2 className="h-3 w-3 text-emerald-400" />
                              <span className="text-foreground">
                                {t('backtestStudio.iterationsImprovement', { n: iterDoneSummary.total_iterations })}{' '}
                                <span className={iterDoneSummary.improvement > 0 ? 'text-emerald-400' : 'text-muted-foreground'}>
                                  {iterDoneSummary.improvement >= 0 ? '+' : ''}
                                  {iterDoneSummary.improvement.toFixed(4)}
                                </span>
                              </span>
                            </div>
                            {iterEarlyStopReason ? (
                              <div className="text-muted-foreground italic mt-0.5">
                                {t('backtestStudio.earlyStop', { reason: iterEarlyStopReason })}
                              </div>
                            ) : null}
                          </div>
                        ) : null}
                      </div>
                    ) : null}
                  </div>
                </ScrollArea>
              ) : (
                <div className="flex h-full items-center justify-center px-3 text-center">
                  <div className="space-y-1.5">
                    <Sliders className="h-5 w-5 mx-auto text-muted-foreground/50" />
                    <div className="text-[11px] text-muted-foreground">
                      {t('backtestStudio.noDynamicParams')}
                    </div>
                    <div className="text-[10px] text-muted-foreground/70" dangerouslySetInnerHTML={{ __html: t('backtestStudio.noDynamicParamsHint') }} />
                  </div>
                </div>
              )}
            </TabsContent>

            {/* ─────── Runs tab ─────── */}
            <TabsContent value="runs" className="flex-1 min-h-0 mt-0 overflow-hidden">
              <ScrollArea className="h-full">
                <RunHistory
                  runs={runsQuery.data ?? []}
                  activeId={activeRun?.run_id ?? null}
                  onSelect={(run) => loadRunMutation.mutate(run.run_id)}
                />
              </ScrollArea>
            </TabsContent>
          </Tabs>

          {/* Sticky footer — primary actions.  Lives OUTSIDE the
              Tabs container so it stays visible regardless of which
              tab is active.  Two coequal entry points:

                [▶ Run backtest]  [⚡ Iterate ▾]

              The iterate config (target_score, max_iters,
              max_no_improvement, mandate, auto_apply) is hidden
              behind the chevron on the iterate button — clicking
              the button (when not running) toggles the expander
              open above the buttons, where the operator commits
              the run via "Start iteration".

              When iteration is running, the iterate button morphs
              into a red [■ Stop] and a compact live status row
              appears above showing best score / current iter /
              target.  The full decision log lives in the
              Parameters tab as a separate "Iteration log" card. */}
          <div className="border-t border-border/50 px-3 py-2 space-y-1.5 bg-background/60">
            {/* Live status row — only when iteration has been
                kicked off.  Stays visible from any tab so the
                operator can monitor progress without switching. */}
            {iterStarted ? (
              <div className="flex items-center justify-between gap-2 rounded-sm border border-cyan-500/30 bg-cyan-500/5 px-2 py-1 text-[10px]">
                <span className="flex items-center gap-1.5">
                  {iterRunning ? (
                    <span className="h-1.5 w-1.5 rounded-full bg-cyan-400 animate-pulse" />
                  ) : iterDoneSummary?.target_reached ? (
                    <CheckCircle2 className="h-3 w-3 text-emerald-400" />
                  ) : (
                    <CheckCircle2 className="h-3 w-3 text-muted-foreground" />
                  )}
                  <span className="font-mono">
                    {t('backtestStudio.iterStatusBest')} <span className="text-emerald-400">
                      {iterBestScore !== null ? iterBestScore.toFixed(4) : '—'}
                    </span>
                  </span>
                </span>
                <span className="font-mono text-muted-foreground">
                  {iterRunning
                    ? t('backtestStudio.iterStatusIter', { cur: iterIteration, max: iterMaxIterations })
                    : iterDoneSummary
                      ? t('backtestStudio.iterStatusIterFinished', { n: iterDoneSummary.total_iterations, improvement: `${iterDoneSummary.improvement >= 0 ? '+' : ''}${iterDoneSummary.improvement.toFixed(4)}` })
                      : ''}
                </span>
                {iterTargetScore ? (
                  <span className="flex items-center gap-1 text-muted-foreground">
                    <Target className="h-2.5 w-2.5" />
                    <span className="font-mono">{iterTargetScore}</span>
                  </span>
                ) : null}
              </div>
            ) : null}

            {/* Iterate config expander — progressive disclosure.
                Closed by default; opens above the buttons when the
                operator clicks the iterate button (and isn't
                already running).  Inputs map 1:1 onto the
                strategy_params autoresearch service's structured
                stop conditions. */}
            {iterAvailable && iterConfigOpen && !iterRunning ? (
              <div className="rounded-sm border border-cyan-500/30 bg-cyan-500/5 p-2 space-y-1.5">
                <div className="grid grid-cols-2 gap-1.5">
                  <div>
                    <Label className="text-[9px] uppercase tracking-wide text-muted-foreground">
                      {t('backtestStudio.labelTargetScore')}
                    </Label>
                    <Input
                      value={iterTargetScore}
                      onChange={(e) => setIterTargetScore(e.target.value)}
                      placeholder={t('backtestStudio.targetScorePlaceholder')}
                      className="h-6 text-[11px] font-mono"
                      title={t('backtestStudio.targetScoreTooltip')}
                    />
                  </div>
                  <div>
                    <Label className="text-[9px] uppercase tracking-wide text-muted-foreground">
                      {t('backtestStudio.labelMaxIters')}
                    </Label>
                    <Input
                      type="number"
                      min={1}
                      max={500}
                      value={iterMaxIterations}
                      onChange={(e) => setIterMaxIterations(parseInt(e.target.value, 10) || 50)}
                      className="h-6 text-[11px] font-mono"
                    />
                  </div>
                  <div>
                    <Label className="text-[9px] uppercase tracking-wide text-muted-foreground">
                      {t('backtestStudio.labelStopAfterNoImprove')}
                    </Label>
                    <Input
                      value={iterMaxNoImprove}
                      onChange={(e) => setIterMaxNoImprove(e.target.value)}
                      placeholder="10"
                      className="h-6 text-[11px] font-mono"
                      title={t('backtestStudio.stopNoImproveTooltip')}
                    />
                  </div>
                  <div className="flex items-end pb-0.5">
                    <label className="flex items-center gap-1.5 cursor-pointer text-[10px]">
                      <input
                        type="checkbox"
                        checked={iterAutoApply}
                        onChange={(e) => setIterAutoApply(e.target.checked)}
                        className="h-3 w-3"
                      />
                      <span title={t('backtestStudio.autoApplyTooltip')}>
                        {t('backtestStudio.autoApplyKept')}
                      </span>
                    </label>
                  </div>
                </div>
                <div>
                  <Label className="text-[9px] uppercase tracking-wide text-muted-foreground">
                    {t('backtestStudio.labelMandate')}
                  </Label>
                  <Input
                    value={iterMandate}
                    onChange={(e) => setIterMandate(e.target.value)}
                    placeholder={t('backtestStudio.mandatePlaceholder')}
                    className="h-6 text-[11px]"
                  />
                </div>
                <Button
                  size="sm"
                  onClick={handleStartIteration}
                  className="h-7 w-full gap-1 text-[11px] bg-cyan-600 hover:bg-cyan-700 text-white"
                  disabled={!initialStrategyId}
                >
                  <Wand2 className="h-3 w-3" />
                  {iterDoneSummary ? t('backtestStudio.iterateAgain') : t('backtestStudio.startIteration')}
                </Button>
              </div>
            ) : null}

            {/* Two-button row.  ``Run backtest`` is unchanged from the
                old layout; ``Iterate`` is new and either opens the
                config expander or stops the in-flight loop. */}
            <div className="flex items-center gap-1.5">
              <Button
                onClick={handleRun}
                disabled={runMutation.isPending || sourceCode.trim().length < 10}
                // Default ``bg-primary`` reads as near-white in dark
                // mode and overwhelmed the rail; emerald-600 keeps
                // the "primary action" weight (it's clearly louder
                // than the cyan-600 Iterate sibling) while staying
                // legible against the dark background.
                className="flex-1 bg-emerald-600 hover:bg-emerald-700 text-white"
              >
                {runMutation.isPending ? (
                  <Loader2 className="mr-1 h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Play className="mr-1 h-3.5 w-3.5" />
                )}
                {t('backtestStudio.runBacktest')}
              </Button>
              {iterRunning ? (
                <Button
                  variant="destructive"
                  onClick={handleStopIteration}
                  className="flex-1 gap-1"
                  title={t('backtestStudio.stopIterationTooltip')}
                >
                  <Square className="h-3.5 w-3.5" />
                  {t('backtestStudio.stop')}
                </Button>
              ) : (
                <Button
                  onClick={() => setIterConfigOpen((v) => !v)}
                  disabled={!iterAvailable}
                  className={cn(
                    'flex-1 gap-1',
                    iterConfigOpen
                      ? 'bg-cyan-700 hover:bg-cyan-800 text-white'
                      : 'bg-cyan-600 hover:bg-cyan-700 text-white',
                  )}
                  title={
                    !iterAvailable
                      ? (paramFieldGroups.length === 0
                          ? t('backtestStudio.iterateNoParams')
                          : t('backtestStudio.iteratePickStrategy'))
                      : iterConfigOpen
                        ? t('backtestStudio.iterateHide')
                        : t('backtestStudio.iterateOpen')
                  }
                >
                  <Wand2 className="h-3.5 w-3.5" />
                  {t('backtestStudio.iterateButton')}
                  {iterConfigOpen ? '▴' : '▾'}
                </Button>
              )}
            </div>

            {/* Errors from either action surface here. */}
            {errorMessage ? (
              <div className="flex items-start gap-1 text-[10px] text-red-300">
                <AlertTriangle className="h-3 w-3 shrink-0 mt-0.5" />
                <span>{t('backtestStudio.runErrorPrefix', { msg: errorMessage })}</span>
              </div>
            ) : null}
            {iterError ? (
              <div className="flex items-start gap-1 text-[10px] text-red-300">
                <AlertTriangle className="h-3 w-3 shrink-0 mt-0.5" />
                <span>{t('backtestStudio.iterErrorPrefix', { msg: iterError })}</span>
              </div>
            ) : null}
          </div>
        </div>

        {/* CENTER — results */}
        <ScrollArea className="flex-1 min-h-0">
          <div className="space-y-4 p-3">
            {/* CURRENTLY-VIEWING BANNER — clearly indicates which run
                populates the subtabs below.  Shows the strategy slug,
                the run id (short), the return %, and the started_at
                timestamp.  Visible whenever there's an active run.
                Loading state shows a subtle pulse while a run is
                being fetched after a click in the recent-runs list. */}
            {activeRun ? (
              <div className="flex items-center gap-2 rounded-md border border-amber-500/40 bg-amber-500/5 px-3 py-1.5 text-[11px]">
                <Activity className="h-3.5 w-3.5 shrink-0 text-amber-300" />
                <span className="text-muted-foreground">{t('backtestStudio.viewingRun')}</span>
                <span className="font-mono text-amber-200">{activeRun.run_id.slice(0, 8)}</span>
                <span className="text-muted-foreground">·</span>
                <span className="truncate font-medium text-foreground">
                  {activeRun.strategy_name || activeRun.strategy_slug || t('backtestStudio.unknown')}
                </span>
                <span
                  className={cn(
                    'shrink-0 font-mono tabular-nums',
                    (activeRun.execution?.total_return_pct ?? 0) >= 0 ? 'text-emerald-300' : 'text-red-300',
                  )}
                >
                  {fmtPct(activeRun.execution?.total_return_pct, 2)}
                </span>
                <span className="ml-auto shrink-0 text-[10px] text-muted-foreground">
                  {new Date(activeRun.started_at).toLocaleString()}
                </span>
              </div>
            ) : loadRunMutation.isPending ? (
              <div className="flex items-center gap-2 rounded-md border border-border/40 bg-card/40 px-3 py-1.5 text-[11px] text-muted-foreground">
                <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-muted-foreground" />
                {t('backtestStudio.loadingRunData')}
              </div>
            ) : runMutation.isPending || readPendingRun() ? (
              // Run is in flight on the backend.  Either we're holding
              // the in-flight mutation (user stayed in the studio) OR
              // we navigated away and came back and the marker tells
              // us the backend is still chewing.  The runs-list poll
              // (every 5s) will reconcile the row when it lands.
              <div className="flex items-center gap-2 rounded-md border border-amber-500/40 bg-amber-500/5 px-3 py-1.5 text-[11px] text-amber-200 dark:text-amber-300">
                <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" />
                <span className="text-muted-foreground">{t('backtestStudio.backtestRunningBackend')}</span>
              </div>
            ) : null}

            {/* SUBTAB STRIP — always visible.  Performance / Fill
                quality / Robustness require an active run to populate;
                Portfolio is run-independent (live cross-strategy
                correlation) and renders the same regardless. */}
            <div className="flex items-center gap-1 border-b border-border/40 pb-1.5">
              {([
                ['performance', t('backtestStudio.tabPerformance'), TrendingUp],
                ['fill_quality', t('backtestStudio.tabFillQuality'), Zap],
                ['robustness', t('backtestStudio.tabRobustness'), Activity],
                ['portfolio', t('backtestStudio.tabPortfolio'), Layers3],
              ] as Array<[
                'performance' | 'fill_quality' | 'robustness' | 'portfolio',
                string,
                typeof TrendingUp,
              ]>).map(([key, label, Icon]) => {
                const active = centerTab === key
                return (
                  <button
                    key={key}
                    type="button"
                    onClick={() => setCenterTab(key)}
                    className={cn(
                      'flex items-center gap-1.5 rounded-sm px-3 py-1 text-[11px] font-medium transition-colors',
                      active
                        ? 'bg-amber-500/10 text-amber-300 ring-1 ring-amber-500/30'
                        : 'text-muted-foreground hover:bg-muted/30 hover:text-foreground',
                    )}
                  >
                    <Icon className="h-3 w-3" />
                    {label}
                  </button>
                )
              })}
            </div>

            {/* In-flight skeleton.  Takes precedence over both the
                empty-state and the activeRun render so the operator
                never sees stale KPIs / charts overlaid with a tiny
                "Backtest running" pill — the whole pane swaps to a
                shimmer that mirrors the real layout. */}
            {(runMutation.isPending || loadRunMutation.isPending || pendingRunId) && centerTab !== 'portfolio' ? (
              <RunningBacktestSkeleton
                variant={pendingRunId ? 'running' : (runMutation.isPending ? 'running' : 'loading')}
                caption={
                  pendingRunId
                    ? t('backtestStudio.skeletonRunning')
                    : runMutation.isPending
                      ? t('backtestStudio.skeletonEnqueueing')
                      : t('backtestStudio.skeletonLoading')
                }
                status={runStatusQuery.data ?? null}
                onCancel={
                  pendingRunId
                    ? () => {
                        if (!pendingRunId) return
                        cancelMutation.mutate(pendingRunId)
                      }
                    : undefined
                }
              />
            ) : null}

            {/* Empty-state placeholder for run-required tabs when no
                run is loaded.  Only shown when nothing else is in
                flight — otherwise the skeleton above wins. */}
            {!activeRun &&
            !runMutation.isPending &&
            !loadRunMutation.isPending &&
            !pendingRunId &&
            centerTab !== 'portfolio' ? (
              <div className="flex flex-col items-center justify-center gap-2 rounded-md border border-dashed border-border/40 bg-card/20 px-6 py-8 text-center">
                <Flame className="h-7 w-7 text-amber-300/50" />
                <div className="text-sm font-medium">
                  {centerTab === 'performance'
                    ? t('backtestStudio.emptyNoRun')
                    : centerTab === 'fill_quality'
                      ? t('backtestStudio.emptyNoFill')
                      : t('backtestStudio.emptyNoRobustness')}
                </div>
                <div className="max-w-[420px] text-xs text-muted-foreground">
                  {centerTab === 'performance'
                    ? t('backtestStudio.emptyPerformanceHint')
                    : centerTab === 'fill_quality'
                      ? t('backtestStudio.emptyFillHint')
                      : t('backtestStudio.emptyRobustnessHint')}
                </div>
              </div>
            ) : null}

            {/* PORTFOLIO TAB — always renders, never gated on activeRun. */}
            {centerTab === 'portfolio' ? (
              <>
                {portfolioCorrelationQuery.data && portfolioCorrelationQuery.data.strategies.length > 0 ? (
                  <div className="rounded-md border border-border/50 bg-card/40 p-3">
                    <div className="mb-2 flex items-center gap-1.5 text-xs font-medium">
                      <Layers3 className="h-3.5 w-3.5 text-emerald-300" />
                      {t('backtestStudio.portfolioCorrelationTitle')}
                      <span className="ml-auto text-[10px] text-muted-foreground">
                        {t('backtestStudio.portfolioCorrelationSub')}
                      </span>
                    </div>
                    <CorrelationHeatmap result={portfolioCorrelationQuery.data} />
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center gap-2 rounded-md border border-dashed border-border/40 bg-card/20 px-6 py-8 text-center">
                    <Layers3 className="h-7 w-7 text-emerald-300/40" />
                    <div className="text-sm font-medium">{t('backtestStudio.noPortfolioData')}</div>
                    <div className="max-w-[420px] text-xs text-muted-foreground">
                      {t('backtestStudio.noPortfolioDataHint')}
                    </div>
                  </div>
                )}

                {/* OUTCOME NETTING + CAPITAL LOCKUP */}
                {activeRun?.outcome_netting ? (
                  <div className="mt-2 rounded-md border border-border/50 bg-card/40 p-3">
                    <div className="mb-2 flex items-center gap-1.5 text-xs font-medium">
                      <Layers3 className="h-3.5 w-3.5 text-violet-700 dark:text-violet-300" />
                      {t('backtestStudio.outcomeNetting')}
                      <span className="ml-auto text-[10px] text-muted-foreground">
                        {t('backtestStudio.outcomeNettingSub')}
                      </span>
                    </div>
                    <div className="grid grid-cols-4 gap-2">
                      <StatTile
                        label={t('backtestStudio.grossExposure')}
                        value={fmtUsd(activeRun.outcome_netting.gross_exposure_usd)}
                        hint={t('backtestStudio.grossExposureHint')}
                      />
                      <StatTile
                        label={t('backtestStudio.netExposure')}
                        value={fmtUsd(activeRun.outcome_netting.net_exposure_usd)}
                        hint={
                          activeRun.outcome_netting.rebate_estimate_usd > 0
                            ? t('backtestStudio.netExposureRebate', { value: fmtUsd(activeRun.outcome_netting.rebate_estimate_usd) })
                            : t('backtestStudio.netExposureNone')
                        }
                        tone={
                          activeRun.outcome_netting.rebate_estimate_usd > 0 ? 'good' : 'neutral'
                        }
                      />
                      <StatTile
                        label={t('backtestStudio.capitalEfficiency')}
                        value={
                          activeRun.outcome_netting.capital_efficiency_pct != null
                            ? `${fmtNum(activeRun.outcome_netting.capital_efficiency_pct, 1)}%`
                            : '—'
                        }
                        hint={t('backtestStudio.capitalEfficiencyHint')}
                        tone={
                          (activeRun.outcome_netting.capital_efficiency_pct ?? 0) >= 20 ? 'good'
                            : (activeRun.outcome_netting.capital_efficiency_pct ?? 0) >= 5 ? 'warn'
                            : 'neutral'
                        }
                      />
                      <StatTile
                        label={t('backtestStudio.lockedCapital')}
                        value={fmtUsd(activeRun.outcome_netting.locked_capital_usd)}
                        hint={t('backtestStudio.lockedCapitalHint', { n: activeRun.outcome_netting.open_positions })}
                      />
                    </div>
                    <div className="mt-2 grid grid-cols-3 gap-2">
                      <div className="rounded-sm bg-emerald-500/10 px-2 py-1 text-[11px]">
                        <div className="text-[9px] uppercase tracking-wide text-emerald-300">{t('backtestStudio.fullCoverage')}</div>
                        <div className="font-mono tabular-nums text-emerald-200">
                          {activeRun.outcome_netting.outcome_groups.full_coverage}
                        </div>
                        <div className="text-[9px] text-muted-foreground">{t('backtestStudio.fullCoverageHint')}</div>
                      </div>
                      <div className="rounded-sm bg-amber-500/10 px-2 py-1 text-[11px]">
                        <div className="text-[9px] uppercase tracking-wide text-amber-300">{t('backtestStudio.partialOutcome')}</div>
                        <div className="font-mono tabular-nums text-amber-200">
                          {activeRun.outcome_netting.outcome_groups.partial}
                        </div>
                        <div className="text-[9px] text-muted-foreground">{t('backtestStudio.partialOutcomeHint')}</div>
                      </div>
                      <div className="rounded-sm bg-muted/40 px-2 py-1 text-[11px]">
                        <div className="text-[9px] uppercase tracking-wide text-muted-foreground">{t('backtestStudio.singleOutcome')}</div>
                        <div className="font-mono tabular-nums">
                          {activeRun.outcome_netting.outcome_groups.single}
                        </div>
                        <div className="text-[9px] text-muted-foreground">{t('backtestStudio.singleOutcomeHint')}</div>
                      </div>
                    </div>
                    <div className="mt-2 grid grid-cols-2 gap-2 text-[10px]">
                      <div className="rounded-sm bg-muted/30 px-2 py-1">
                        <span className="text-muted-foreground">{t('backtestStudio.avgLockup')} </span>
                        <span className="font-mono tabular-nums">
                          {activeRun.outcome_netting.avg_lockup_seconds != null
                            ? t('backtestStudio.lockupDays', { n: (activeRun.outcome_netting.avg_lockup_seconds / 86400).toFixed(1) })
                            : '—'}
                        </span>
                      </div>
                      <div className="rounded-sm bg-muted/30 px-2 py-1">
                        <span className="text-muted-foreground">{t('backtestStudio.maxLockup')} </span>
                        <span className="font-mono tabular-nums">
                          {activeRun.outcome_netting.max_lockup_seconds != null
                            ? t('backtestStudio.lockupDays', { n: (activeRun.outcome_netting.max_lockup_seconds / 86400).toFixed(1) })
                            : '—'}
                        </span>
                      </div>
                    </div>
                    <div className="mt-2 text-[10px] text-muted-foreground">
                      {t('backtestStudio.outcomeNettingFootnote')}
                    </div>
                  </div>
                ) : null}

                {/* DRIFT MONITOR */}
                {driftQuery.data && driftQuery.data.strategies.length > 0 ? (
                  <div className="mt-2 rounded-md border border-border/50 bg-card/40 p-3">
                    <div className="mb-2 flex items-center gap-1.5 text-xs font-medium">
                      <Activity className="h-3.5 w-3.5 text-rose-300" />
                      {t('backtestStudio.driftTitle', { days: driftQuery.data.window_days })}
                      <span className="ml-auto text-[10px] text-muted-foreground">
                        {t('backtestStudio.driftStrategiesTracked', { n: driftQuery.data.summary.n_strategies })}
                      </span>
                    </div>
                    <div className="grid grid-cols-4 gap-2">
                      {(['stable', 'improved', 'degraded', 'stale'] as const).map((sev) => (
                        <div
                          key={sev}
                          className={cn(
                            'rounded-sm px-2 py-1 text-[11px]',
                            sev === 'stable'
                              ? 'bg-emerald-500/10 text-emerald-200'
                              : sev === 'improved'
                              ? 'bg-sky-500/10 text-sky-200'
                              : sev === 'degraded'
                              ? 'bg-rose-500/15 text-rose-200'
                              : 'bg-muted/40 text-muted-foreground',
                          )}
                        >
                          <div className="text-[9px] uppercase tracking-wide">{t(`backtestStudio.drift${sev.charAt(0).toUpperCase()}${sev.slice(1)}`)}</div>
                          <div className="font-mono tabular-nums text-base">
                            {driftQuery.data.summary.by_severity[sev] ?? 0}
                          </div>
                        </div>
                      ))}
                    </div>
                    {driftQuery.data.summary.worst_offender ? (
                      <div className="mt-2 rounded-sm bg-rose-500/10 px-2 py-1.5 text-[11px] text-rose-100">
                        <span className="font-semibold">{t('backtestStudio.worstOffender')} </span>
                        <span className="font-mono">{driftQuery.data.summary.worst_offender.strategy_slug}</span>
                        <span className="ml-2 text-rose-200/80">{driftQuery.data.summary.worst_offender.reason}</span>
                      </div>
                    ) : null}
                    <div className="mt-2 max-h-[280px] overflow-y-auto">
                      <table className="w-full text-[10px]">
                        <thead className="sticky top-0 bg-card/95 text-muted-foreground">
                          <tr>
                            <th className="text-left">{t('backtestStudio.colStrategy')}</th>
                            <th className="text-right">{t('backtestStudio.colBtSharpe')}</th>
                            <th className="text-right">{t('backtestStudio.colLiveSharpe')}</th>
                            <th className="text-right">{t('backtestStudio.colDelta')}</th>
                            <th className="text-right">{t('backtestStudio.colLivePnl')}</th>
                            <th className="text-right">{t('backtestStudio.colLiveTrades')}</th>
                            <th className="text-left">{t('backtestStudio.colSeverity')}</th>
                          </tr>
                        </thead>
                        <tbody>
                          {driftQuery.data.strategies.map((s) => (
                            <tr key={s.strategy_slug} className="border-t border-border/20">
                              <td className="font-mono">{s.strategy_slug}</td>
                              <td className="text-right font-mono tabular-nums">
                                {s.backtest_sharpe != null ? s.backtest_sharpe.toFixed(2) : '—'}
                              </td>
                              <td className="text-right font-mono tabular-nums">
                                {s.live_sharpe != null ? s.live_sharpe.toFixed(2) : '—'}
                              </td>
                              <td
                                className={cn(
                                  'text-right font-mono tabular-nums',
                                  (s.sharpe_delta ?? 0) < -0.5 ? 'text-rose-300'
                                    : (s.sharpe_delta ?? 0) > 0.5 ? 'text-emerald-300'
                                    : '',
                                )}
                              >
                                {s.sharpe_delta != null ? (s.sharpe_delta >= 0 ? '+' : '') + s.sharpe_delta.toFixed(2) : '—'}
                              </td>
                              <td className="text-right font-mono tabular-nums">{fmtUsd(s.live_total_pnl_usd)}</td>
                              <td className="text-right font-mono tabular-nums">{s.live_trade_count}</td>
                              <td>
                                <span
                                  className={cn(
                                    'rounded-sm px-1.5 py-0.5 text-[9px] uppercase',
                                    s.severity === 'stable' ? 'bg-emerald-500/15 text-emerald-200'
                                      : s.severity === 'improved' ? 'bg-sky-500/15 text-sky-200'
                                      : s.severity === 'degraded' ? 'bg-rose-500/15 text-rose-200'
                                      : 'bg-muted/40 text-muted-foreground',
                                  )}
                                >
                                  {s.severity}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <div className="mt-2 text-[10px] text-muted-foreground">
                      {t('backtestStudio.driftFootnote')}
                    </div>
                  </div>
                ) : null}
              </>
            ) : null}

            {activeRun && !runMutation.isPending && !loadRunMutation.isPending && !pendingRunId ? (
              <>
              {/* HEADLINE KPIS + SECONDARY METRICS — Performance tab. */}
              {centerTab === 'performance' && (
              <>
              {/* Data fidelity / replay-source banner — must appear BEFORE
                  the trade-count headline.  When fidelity is low/none AND
                  the engine ran on snapshots (not deltas), "0 trades" is
                  a data-coverage problem.  When the engine ran on the
                  delta-replay path it's live-parity regardless of mms
                  density — that turns the banner emerald. */}
              <DataCoverageBanner
                coverage={activeRun.data_coverage}
                replaySource={activeRun.execution?.replay_source}
                discoveryMode={activeRun.execution?.discovery_mode}
              />
              <div className="grid grid-cols-4 gap-2">
                <StatTile
                  label={t('backtestStudio.kpiReturn')}
                  value={fmtPct(exec?.total_return_pct, 2)}
                  hint={t('backtestStudio.kpiReturnHint', { value: fmtUsd((exec?.final_equity_usd ?? 0) - (exec?.initial_capital_usd ?? 0)) })}
                  tone={totalReturnTone}
                  icon={exec && exec.total_return_pct >= 0 ? TrendingUp : TrendingDown}
                />
                <StatTile
                  label={t('backtestStudio.kpiSharpe')}
                  value={fmtNum(exec?.sharpe?.value, 2)}
                  hint={
                    exec?.sharpe?.ci_low != null && exec?.sharpe?.ci_high != null
                      ? t('backtestStudio.kpiSharpeCi', { lo: fmtNum(exec.sharpe.ci_low, 2), hi: fmtNum(exec.sharpe.ci_high, 2) })
                      : undefined
                  }
                  tone={sharpeTone}
                  icon={Activity}
                />
                <StatTile
                  label={t('backtestStudio.kpiMaxDrawdown')}
                  value={fmtPct(exec?.max_drawdown_pct, 2)}
                  hint={t('backtestStudio.kpiDrawdownDuration', { value: fmtMs((exec?.drawdown_duration_seconds ?? 0) * 1000) })}
                  tone={ddTone}
                  icon={TrendingDown}
                />
                <StatTile
                  label={t('backtestStudio.kpiTrades')}
                  value={(exec?.trade_count ?? 0).toLocaleString()}
                  hint={t('backtestStudio.kpiTradesHint', { fills: exec?.total_fills ?? 0, cancels: exec?.cancelled_orders ?? 0, rejects: exec?.rejected_orders ?? 0 })}
                  icon={Zap}
                />
              </div>

              {/* SECONDARY METRICS */}
              <div className="grid grid-cols-2 gap-2">
                <div className="rounded-md border border-border/50 bg-card/40 p-3">
                  <div className="mb-1 flex items-center gap-1.5 text-xs font-medium">
                    <Activity className="h-3.5 w-3.5 text-amber-300" />
                    {t('backtestStudio.riskAdjustedTitle')}
                  </div>
                  <MetricRow label={t('backtestStudio.metricSharpe')} m={exec?.sharpe} tone={sharpeTone === 'bad' ? 'bad' : sharpeTone === 'good' ? 'good' : undefined} />
                  <MetricRow label={t('backtestStudio.metricSortino')} m={exec?.sortino} />
                  <MetricRow label={t('backtestStudio.metricCalmar')} m={exec?.calmar} />
                  <MetricRow label={t('backtestStudio.metricHitRate')} m={exec?.hit_rate} />
                  <MetricRow label={t('backtestStudio.metricProfitFactor')} m={exec?.profit_factor} />
                  <MetricRow label={t('backtestStudio.metricExpectancyUsd')} m={exec?.expectancy_usd} />
                  {(exec?.expected_shortfall_5pct || exec?.tail_ratio || exec?.gain_to_pain) ? (
                    <div className="mt-2 border-t border-border/40 pt-2">
                      <div className="mb-1 flex items-center justify-between text-[10px] uppercase tracking-wide text-muted-foreground">
                        <span>{t('backtestStudio.tailRiskTitle')}</span>
                        <span>{t('backtestStudio.tailRiskSub')}</span>
                      </div>
                      <MetricRow label={t('backtestStudio.metricEs5')} m={exec?.expected_shortfall_5pct} tone={(exec?.expected_shortfall_5pct?.value ?? 0) < -0.05 ? 'bad' : undefined} />
                      <MetricRow label={t('backtestStudio.metricEs1')} m={exec?.expected_shortfall_1pct} />
                      <MetricRow
                        label={t('backtestStudio.metricTailRatio')}
                        m={exec?.tail_ratio}
                        tone={
                          (exec?.tail_ratio?.value ?? 0) >= 1.5 ? 'good'
                            : (exec?.tail_ratio?.value ?? 0) < 0.7 ? 'bad'
                            : undefined
                        }
                      />
                      <MetricRow label={t('backtestStudio.metricGainToPain')} m={exec?.gain_to_pain} tone={(exec?.gain_to_pain?.value ?? 0) >= 1.5 ? 'good' : undefined} />
                      <div className="mt-1 text-[10px] text-muted-foreground">
                        {t('backtestStudio.tailRiskFootnote')}
                      </div>
                    </div>
                  ) : null}
                  {activeRun?.deflated_sharpe ? (
                    <div className="mt-2 border-t border-border/40 pt-2">
                      <div className="flex items-center justify-between text-[10px] uppercase tracking-wide text-muted-foreground">
                        <span>{t('backtestStudio.deflatedSharpeTitle')}</span>
                        <span>{activeRun.deflated_sharpe.n_trials === 1 ? t('backtestStudio.deflatedSharpeTrials', { n: activeRun.deflated_sharpe.n_trials }) : t('backtestStudio.deflatedSharpeTrialsPlural', { n: activeRun.deflated_sharpe.n_trials })}</span>
                      </div>
                      <div className="mt-1 grid grid-cols-2 gap-1 text-[11px]">
                        <div className="flex items-center justify-between rounded-sm bg-muted/40 px-1.5 py-0.5">
                          <span className="text-muted-foreground">{t('backtestStudio.deflatedPTrueSr')}</span>
                          <span className={cn(
                            'font-mono tabular-nums',
                            activeRun.deflated_sharpe.probabilistic_sharpe >= 0.95 ? 'text-emerald-300'
                              : activeRun.deflated_sharpe.probabilistic_sharpe >= 0.7 ? 'text-amber-300'
                              : 'text-red-300',
                          )}>
                            {(activeRun.deflated_sharpe.probabilistic_sharpe * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex items-center justify-between rounded-sm bg-muted/40 px-1.5 py-0.5">
                          <span className="text-muted-foreground">{t('backtestStudio.deflatedPSrOverfit')}</span>
                          <span className={cn(
                            'font-mono tabular-nums',
                            activeRun.deflated_sharpe.deflated_sharpe >= 0.95 ? 'text-emerald-300'
                              : activeRun.deflated_sharpe.deflated_sharpe >= 0.7 ? 'text-amber-300'
                              : 'text-red-300',
                          )}>
                            {(activeRun.deflated_sharpe.deflated_sharpe * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                      <div className="mt-1 text-[10px] text-muted-foreground">
                        {t('backtestStudio.deflatedFootnote', {
                          sr0: activeRun.deflated_sharpe.sr_zero.toFixed(2),
                          obs: activeRun.deflated_sharpe.observed_sharpe.toFixed(2),
                          verdict: activeRun.deflated_sharpe.deflated_sharpe < 0.95 && activeRun.deflated_sharpe.n_trials > 1
                            ? t('backtestStudio.deflatedLikelyOverfit')
                            : t('backtestStudio.deflatedOk'),
                        })}
                      </div>
                    </div>
                  ) : null}
                </div>

                <div className="rounded-md border border-border/50 bg-card/40 p-3">
                  <div className="mb-1 flex items-center gap-1.5 text-xs font-medium">
                    <LineChartIcon className="h-3.5 w-3.5 text-emerald-300" />
                    {t('backtestStudio.equityCurveTitle')}
                  </div>
                  <EquityCurveChart points={exec?.equity_curve_sample ?? []} />
                  <div className="mt-2 grid grid-cols-3 gap-2">
                    <StatTile
                      label={t('backtestStudio.avgWin')}
                      value={fmtUsd(exec?.avg_win_usd)}
                      tone="good"
                    />
                    <StatTile
                      label={t('backtestStudio.avgLoss')}
                      value={fmtUsd(exec?.avg_loss_usd)}
                      tone="bad"
                    />
                    <StatTile
                      label={t('backtestStudio.feesPerFill')}
                      value={fmtUsd(exec?.fees_per_fill_usd)}
                      hint={t('backtestStudio.feesTotal', { value: fmtUsd(exec?.fees_paid_usd) })}
                    />
                  </div>
                </div>
              </div>
              </>
              )}

              {/* ENSEMBLE BANDS + COUNTERFACTUALS */}
              {centerTab === 'fill_quality' && (
              <div className="grid grid-cols-2 gap-2">
                <div className="rounded-md border border-border/50 bg-card/40 p-3">
                  <div className="mb-2 flex items-center gap-1.5 text-xs font-medium">
                    <Layers3 className="h-3.5 w-3.5 text-violet-700 dark:text-violet-300" />
                    {t('backtestStudio.ensembleTitle')}
                    <span className="ml-auto text-[10px] text-muted-foreground">
                      {t('backtestStudio.ensembleSampleFills', { n: activeRun.ensemble_band.length })}
                    </span>
                  </div>
                  <EnsembleBand band={activeRun.ensemble_band} />
                </div>

                <div className="rounded-md border border-border/50 bg-card/40 p-3">
                  <div className="mb-2 flex items-center gap-1.5 text-xs font-medium">
                    <Clock className="h-3.5 w-3.5 text-sky-300" />
                    {t('backtestStudio.counterfactualTitle')}
                    <span className="ml-auto text-[10px] text-muted-foreground">
                      {t('backtestStudio.ensembleSampleFills', { n: activeRun.counterfactuals.length })}
                    </span>
                  </div>
                  <CounterfactualList rows={activeRun.counterfactuals} />
                </div>
              </div>
              )}

              {/* TRIANGULATION — backtest vs shadow vs live PnL */}
              {centerTab === 'robustness' && triangulationQuery.data ? (
                <div className="rounded-md border border-border/50 bg-card/40 p-3">
                  <div className="mb-2 flex items-center gap-1.5 text-xs font-medium">
                    <Activity className="h-3.5 w-3.5 text-amber-300" />
                    {t('backtestStudio.triangulationTitle')}
                    <span className="ml-auto text-[10px] text-muted-foreground">
                      {t('backtestStudio.triangulationSub')}
                    </span>
                  </div>
                  <div className="grid grid-cols-3 gap-2">
                    {(() => {
                      const tri = triangulationQuery.data
                      const shadowMode = tri.modes.shadow
                      const liveMode = tri.modes.live
                      const btPnl = exec
                        ? (exec.final_equity_usd ?? 0) - (exec.initial_capital_usd ?? 0)
                        : 0
                      const shadowPnl = shadowMode?.realized_pnl_usd ?? 0
                      const livePnl = liveMode?.realized_pnl_usd ?? 0
                      // Tone the live tile by alignment with backtest.
                      const liveDeltaPct = btPnl !== 0 ? ((livePnl - btPnl) / Math.abs(btPnl)) * 100 : 0
                      const shadowDeltaPct =
                        btPnl !== 0 ? ((shadowPnl - btPnl) / Math.abs(btPnl)) * 100 : 0
                      const divergent =
                        Math.abs(liveDeltaPct) > 30 || Math.abs(shadowDeltaPct) > 30
                      return (
                        <>
                          <StatTile
                            label={t('backtestStudio.tileBacktestPnl')}
                            value={fmtUsd(btPnl)}
                            hint={t('backtestStudio.tileBacktestPnlHint', { trades: exec?.trade_count ?? 0, ret: fmtPct(exec?.total_return_pct, 1) })}
                            tone={btPnl >= 0 ? 'good' : 'bad'}
                            icon={Flame}
                          />
                          <StatTile
                            label={t('backtestStudio.tileShadowPnl')}
                            value={fmtUsd(shadowPnl)}
                            hint={
                              shadowMode
                                ? t('backtestStudio.shadowOrders', { orders: shadowMode.orders, filled: shadowMode.filled, delta: btPnl !== 0 ? t('backtestStudio.deltaPctLabel', { pct: fmtPct(shadowDeltaPct, 0) }) : t('backtestStudio.noBacktest') })
                                : t('backtestStudio.noShadowData')
                            }
                            tone={shadowPnl >= 0 ? 'good' : 'bad'}
                          />
                          <StatTile
                            label={t('backtestStudio.tileLivePnl')}
                            value={fmtUsd(livePnl)}
                            hint={
                              liveMode
                                ? t('backtestStudio.shadowOrders', { orders: liveMode.orders, filled: liveMode.filled, delta: btPnl !== 0 ? t('backtestStudio.deltaPctLabel', { pct: fmtPct(liveDeltaPct, 0) }) : t('backtestStudio.noBacktest') })
                                : t('backtestStudio.noLiveData')
                            }
                            tone={
                              divergent ? 'warn' : livePnl >= 0 ? 'good' : 'bad'
                            }
                          />
                        </>
                      )
                    })()}
                  </div>
                  <div className="mt-2 text-[10px] text-muted-foreground">
                    {t('backtestStudio.triangulationFootnote')}
                  </div>
                </div>
              ) : null}

              {/* PARTIAL FILL AGGREGATES */}
              {centerTab === 'fill_quality' && activeRun?.partial_fills && activeRun.partial_fills.n_orders > 0 ? (
                <div className="rounded-md border border-border/50 bg-card/40 p-3">
                  <div className="mb-2 flex items-center gap-1.5 text-xs font-medium">
                    <Zap className="h-3.5 w-3.5 text-sky-300" />
                    {t('backtestStudio.partialFillTitle')}
                    <span className="ml-auto text-[10px] text-muted-foreground">
                      {t('backtestStudio.partialFillSub')}
                    </span>
                  </div>
                  <div className="grid grid-cols-4 gap-2">
                    <StatTile
                      label={t('backtestStudio.instantFills')}
                      value={`${(activeRun.partial_fills.instant_fill_rate * 100).toFixed(0)}%`}
                      hint={t('backtestStudio.instantFillsHint', { instant: activeRun.partial_fills.n_instant_fills, orders: activeRun.partial_fills.n_orders })}
                      tone={activeRun.partial_fills.instant_fill_rate >= 0.7 ? 'good' : activeRun.partial_fills.instant_fill_rate >= 0.4 ? 'warn' : 'bad'}
                    />
                    <StatTile
                      label={t('backtestStudio.avgChildren')}
                      value={fmtNum(activeRun.partial_fills.mean_children_per_order, 2)}
                      hint={t('backtestStudio.avgChildrenHint', { n: activeRun.partial_fills.max_children_per_order })}
                    />
                    <StatTile
                      label={t('backtestStudio.intraOrderSpan')}
                      value={activeRun.partial_fills.mean_intra_order_seconds > 0 ? fmtMs(activeRun.partial_fills.mean_intra_order_seconds * 1000) : '—'}
                      hint={t('backtestStudio.intraOrderSpanHint')}
                    />
                    <StatTile
                      label={t('backtestStudio.vwapDispersion')}
                      value={t('backtestStudio.vwapDispersionUnit', { n: fmtNum(activeRun.partial_fills.mean_vwap_dispersion_bps, 1) })}
                      hint={t('backtestStudio.vwapDispersionHint')}
                      tone={activeRun.partial_fills.mean_vwap_dispersion_bps > 50 ? 'warn' : 'neutral'}
                    />
                  </div>
                  {activeRun.partial_fills.child_count_distribution.length > 1 ? (
                    <div className="mt-2">
                      <div className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground">
                        {t('backtestStudio.childCountDist')}
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {activeRun.partial_fills.child_count_distribution.map((d) => {
                          const pct = activeRun.partial_fills.n_orders > 0
                            ? (d.n_orders / activeRun.partial_fills.n_orders) * 100
                            : 0
                          return (
                            <div
                              key={d.children}
                              className={cn(
                                'rounded-sm border px-1.5 py-0.5 font-mono text-[10px]',
                                d.children === 1 ? 'border-emerald-500/30 text-emerald-300' :
                                d.children <= 3 ? 'border-amber-500/30 text-amber-300' :
                                'border-red-500/30 text-red-300',
                              )}
                            >
                              {t('backtestStudio.childCountEntry', { children: d.children, orders: d.n_orders, pct: pct.toFixed(0) })}
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  ) : null}
                  <div className="mt-2 text-[10px] text-muted-foreground">
                    {t('backtestStudio.partialFillFootnote')}
                  </div>
                </div>
              ) : null}

              {/* REGIME DECOMPOSITION */}
              {centerTab === 'robustness' && activeRun?.regime_breakdown ? (
                <div className="rounded-md border border-border/50 bg-card/40 p-3">
                  <div className="mb-2 flex items-center gap-1.5 text-xs font-medium">
                    <Layers3 className="h-3.5 w-3.5 text-amber-300" />
                    {t('backtestStudio.regimeTitle')}
                    <span className="ml-auto text-[10px] text-muted-foreground">
                      {t('backtestStudio.regimeSub')}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
                    <RegimeBlock title={t('backtestStudio.regimeHour')} rows={activeRun.regime_breakdown.by_hour} />
                    <RegimeBlock title={t('backtestStudio.regimeDow')} rows={activeRun.regime_breakdown.by_dow} />
                    <RegimeBlock title={t('backtestStudio.regimeTtr')} rows={activeRun.regime_breakdown.by_ttr} />
                    <RegimeBlock title={t('backtestStudio.regimeSize')} rows={activeRun.regime_breakdown.by_size} />
                  </div>
                  <div className="mt-2 text-[10px] text-muted-foreground">
                    {t('backtestStudio.regimeFootnote')}
                  </div>
                </div>
              ) : null}

              {/* WALK-FORWARD ANALYSIS */}
              {centerTab === 'robustness' && (
              <div className="rounded-md border border-border/50 bg-card/40 p-3">
                <div className="mb-2 flex items-center gap-1.5 text-xs font-medium">
                  <Activity className="h-3.5 w-3.5 text-violet-700 dark:text-violet-300" />
                  {t('backtestStudio.walkForwardTitle')}
                  <div className="ml-auto flex items-center gap-1.5">
                    <select
                      value={walkForwardMode}
                      onChange={(e) => setWalkForwardMode(e.target.value as 'anchored' | 'rolling')}
                      className="h-6 rounded-sm border border-border/40 bg-background/60 px-1.5 text-[10px]"
                    >
                      <option value="anchored">{t('backtestStudio.wfModeAnchored')}</option>
                      <option value="rolling">{t('backtestStudio.wfModeRolling')}</option>
                    </select>
                    <select
                      value={walkForwardFolds}
                      onChange={(e) => setWalkForwardFolds(parseInt(e.target.value, 10))}
                      className="h-6 rounded-sm border border-border/40 bg-background/60 px-1.5 text-[10px]"
                    >
                      {[3, 4, 6, 8, 10, 12].map((n) => (
                        <option key={n} value={n}>
                          {t('backtestStudio.wfFolds', { n })}
                        </option>
                      ))}
                    </select>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={handleWalkForward}
                      disabled={walkForwardMutation.isPending || sourceCode.trim().length < 10}
                      className="h-6 text-[10px]"
                    >
                      {walkForwardMutation.isPending ? (
                        <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                      ) : (
                        <Play className="mr-1 h-3 w-3" />
                      )}
                      {t('backtestStudio.wfRun')}
                    </Button>
                  </div>
                </div>
                {walkForwardResult ? (
                  <>
                    <div className="grid grid-cols-4 gap-2">
                      <StatTile
                        label={t('backtestStudio.wfStableFolds')}
                        value={`${walkForwardResult.summary.stable_window_pct.toFixed(0)}%`}
                        hint={t('backtestStudio.wfStableFoldsHint', { ok: walkForwardResult.summary.n_windows_succeeded, total: walkForwardResult.summary.n_windows_run })}
                        tone={walkForwardResult.summary.stable_window_pct >= 70 ? 'good' : walkForwardResult.summary.stable_window_pct >= 50 ? 'warn' : 'bad'}
                      />
                      <StatTile
                        label={t('backtestStudio.wfMeanReturn')}
                        value={fmtPct(walkForwardResult.summary.mean_return_pct, 2)}
                        hint={t('backtestStudio.wfMeanReturnHint', { min: fmtPct(walkForwardResult.summary.min_return_pct, 1), max: fmtPct(walkForwardResult.summary.max_return_pct, 1) })}
                        tone={walkForwardResult.summary.mean_return_pct >= 0 ? 'good' : 'bad'}
                      />
                      <StatTile
                        label={t('backtestStudio.wfMeanSharpe')}
                        value={walkForwardResult.summary.mean_sharpe != null ? fmtNum(walkForwardResult.summary.mean_sharpe, 2) : '—'}
                        hint={
                          walkForwardResult.summary.min_sharpe != null && walkForwardResult.summary.max_sharpe != null
                            ? t('backtestStudio.wfMeanSharpeHint', { min: fmtNum(walkForwardResult.summary.min_sharpe, 2), max: fmtNum(walkForwardResult.summary.max_sharpe, 2) })
                            : undefined
                        }
                        tone={walkForwardResult.summary.mean_sharpe != null && walkForwardResult.summary.mean_sharpe > 1.0 ? 'good' : 'neutral'}
                      />
                      <StatTile
                        label={t('backtestStudio.wfMode')}
                        value={walkForwardResult.mode}
                        hint={t('backtestStudio.wfModeHint', { n: walkForwardResult.n_windows_run })}
                      />
                    </div>
                    <div className="mt-2 space-y-1">
                      {walkForwardResult.windows.map((w) => (
                        <div
                          key={w.index}
                          className={cn(
                            'grid grid-cols-[60px,1fr,80px,80px,60px,80px] items-center gap-2 rounded-sm border px-2 py-1 text-[10px]',
                            !w.success
                              ? 'border-red-500/30 bg-red-500/5'
                              : w.total_return_pct >= 0
                                ? 'border-emerald-500/20 bg-emerald-500/5'
                                : 'border-amber-500/20 bg-amber-500/5',
                          )}
                        >
                          <span className="font-mono text-muted-foreground">{t('backtestStudio.wfFold', { n: w.index })}</span>
                          <span className="truncate text-muted-foreground">
                            {new Date(w.test_start_iso).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit' })} →{' '}
                            {new Date(w.test_end_iso).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit' })}
                          </span>
                          <span
                            className={cn(
                              'text-right font-mono tabular-nums',
                              w.total_return_pct >= 0 ? 'text-emerald-300' : 'text-red-300',
                            )}
                          >
                            {fmtPct(w.total_return_pct, 1)}
                          </span>
                          <span className="text-right font-mono tabular-nums text-muted-foreground">
                            SR {w.sharpe != null ? fmtNum(w.sharpe, 2) : '—'}
                          </span>
                          <span className="text-right font-mono tabular-nums text-muted-foreground">
                            {t('backtestStudio.wfTrades', { n: w.trade_count })}
                          </span>
                          <span className="text-right font-mono tabular-nums text-muted-foreground">
                            {fmtUsd(w.final_equity_usd)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </>
                ) : (
                  <div className="text-[11px] text-muted-foreground italic" dangerouslySetInnerHTML={{ __html: t('backtestStudio.wfEmpty', { folds: walkForwardFolds, mode: walkForwardMode }) }} />
                )}
              </div>
              )}

              {/* TRADE-ORDER MONTE CARLO — auto-populated from active run */}
              {centerTab === 'robustness' && activeRun?.trade_order_monte_carlo ? (
                <div className="rounded-md border border-border/50 bg-card/40 p-3">
                  <div className="mb-2 flex items-center gap-1.5 text-xs font-medium">
                    <Activity className="h-3.5 w-3.5 text-amber-300" />
                    {t('backtestStudio.tomTitle')}
                    <span className="ml-auto text-[10px] text-muted-foreground">
                      {t('backtestStudio.tomSub')}
                    </span>
                  </div>
                  {activeRun.trade_order_monte_carlo.skipped_reason ? (
                    <div className="text-[11px] text-muted-foreground italic">
                      {activeRun.trade_order_monte_carlo.skipped_reason}
                    </div>
                  ) : (
                    <>
                      <div className="grid grid-cols-4 gap-2">
                        <StatTile
                          label={t('backtestStudio.tomRealizedSharpe')}
                          value={fmtNum(activeRun.trade_order_monte_carlo.realized_sharpe, 2)}
                          hint={t('backtestStudio.tomRealizedSharpeHint', { n: activeRun.trade_order_monte_carlo.n_trades })}
                        />
                        <StatTile
                          label={t('backtestStudio.tomShuffleMedian')}
                          value={fmtNum(activeRun.trade_order_monte_carlo.sharpe_distribution.p50 ?? 0, 2)}
                          hint={t('backtestStudio.tomShuffleMedianHint', { p5: fmtNum(activeRun.trade_order_monte_carlo.sharpe_distribution.p5 ?? 0, 2), p95: fmtNum(activeRun.trade_order_monte_carlo.sharpe_distribution.p95 ?? 0, 2) })}
                        />
                        <StatTile
                          label={t('backtestStudio.tomShuffleStdev')}
                          value={fmtNum(activeRun.trade_order_monte_carlo.sharpe_distribution.stdev ?? 0, 2)}
                          hint={t('backtestStudio.tomShuffleStdevHint', { n: activeRun.trade_order_monte_carlo.n_resamples })}
                        />
                        <StatTile
                          label={t('backtestStudio.tomPositionPct')}
                          value={
                            activeRun.trade_order_monte_carlo.observed_vs_distribution
                              ? `${activeRun.trade_order_monte_carlo.observed_vs_distribution.position_pct.toFixed(0)}%`
                              : '—'
                          }
                          hint={
                            activeRun.trade_order_monte_carlo.observed_vs_distribution?.interpretation ?? ''
                          }
                          tone={
                            activeRun.trade_order_monte_carlo.observed_vs_distribution?.interpretation ===
                            'sequence-driven'
                              ? 'warn'
                              : 'good'
                          }
                        />
                      </div>
                      <div className="mt-2 text-[10px] text-muted-foreground">
                        {t('backtestStudio.tomFootnote')}
                      </div>
                    </>
                  )}
                </div>
              ) : null}

              {/* CPCV (Combinatorial Purged Cross-Validation) */}
              {centerTab === 'robustness' && (
                <div className="rounded-md border border-border/50 bg-card/40 p-3">
                  <div className="mb-2 flex items-center gap-1.5 text-xs font-medium">
                    <Activity className="h-3.5 w-3.5 text-emerald-300" />
                    {t('backtestStudio.cpcvTitle')}
                    <div className="ml-auto flex items-center gap-1.5">
                      <Label className="text-[10px] text-muted-foreground">{t('backtestStudio.cpcvNFolds')}</Label>
                      <Input
                        type="number"
                        min={3}
                        max={12}
                        value={cpcvNFolds}
                        onChange={(e) => setCpcvNFolds(parseInt(e.target.value, 10))}
                        className="h-6 w-14 text-[10px]"
                      />
                      <Label className="text-[10px] text-muted-foreground">{t('backtestStudio.cpcvKTest')}</Label>
                      <Input
                        type="number"
                        min={1}
                        max={6}
                        value={cpcvKTest}
                        onChange={(e) => setCpcvKTest(parseInt(e.target.value, 10))}
                        className="h-6 w-12 text-[10px]"
                      />
                      <Button
                        size="sm"
                        className="h-6 text-[10px]"
                        onClick={() => {
                          if (sourceCode.trim().length < 10) return
                          const end = new Date()
                          const start = new Date(end.getTime() - 14 * 24 * 3600_000)
                          cpcvMutation.mutate({
                            source_code: sourceCode,
                            slug: slug || '_backtest_cpcv',
                            start: start.toISOString(),
                            end: end.toISOString(),
                            n_folds: cpcvNFolds,
                            k_test_folds: cpcvKTest,
                            embargo_seconds: 3600,
                          })
                        }}
                        disabled={cpcvMutation.isPending || sourceCode.trim().length < 10}
                      >
                        {cpcvMutation.isPending ? t('backtestStudio.cpcvRunning') : t('backtestStudio.cpcvRun')}
                      </Button>
                    </div>
                  </div>
                  {cpcvResult ? (
                    <>
                      <div className="grid grid-cols-4 gap-2">
                        <StatTile
                          label={t('backtestStudio.cpcvStablePaths')}
                          value={`${cpcvResult.summary.stable_path_pct.toFixed(0)}%`}
                          hint={t('backtestStudio.cpcvStablePathsHint', { ok: cpcvResult.summary.n_paths_succeeded, total: cpcvResult.summary.n_paths_run })}
                          tone={
                            cpcvResult.summary.stable_path_pct >= 70 ? 'good'
                              : cpcvResult.summary.stable_path_pct >= 50 ? 'warn'
                              : 'bad'
                          }
                        />
                        <StatTile
                          label={t('backtestStudio.cpcvSharpeMedian')}
                          value={fmtNum(cpcvResult.summary.sharpe_median ?? 0, 2)}
                          hint={
                            cpcvResult.summary.sharpe_p10 != null && cpcvResult.summary.sharpe_p90 != null
                              ? t('backtestStudio.cpcvSharpeHint', { p10: cpcvResult.summary.sharpe_p10.toFixed(2), p90: cpcvResult.summary.sharpe_p90.toFixed(2) })
                              : ''
                          }
                        />
                        <StatTile
                          label={t('backtestStudio.cpcvMeanReturn')}
                          value={fmtPct(cpcvResult.summary.return_mean_pct ?? 0, 2)}
                          hint={t('backtestStudio.cpcvMeanReturnHint', { min: fmtPct(cpcvResult.summary.return_min_pct ?? 0, 1), max: fmtPct(cpcvResult.summary.return_max_pct ?? 0, 1) })}
                        />
                        <StatTile
                          label={t('backtestStudio.cpcvPbo')}
                          value={cpcvResult.summary.pbo != null ? `${(cpcvResult.summary.pbo * 100).toFixed(0)}%` : '—'}
                          hint={t('backtestStudio.cpcvPboHint')}
                          tone={
                            cpcvResult.summary.pbo == null ? 'neutral'
                              : cpcvResult.summary.pbo > 0.5 ? 'bad'
                              : cpcvResult.summary.pbo > 0.3 ? 'warn'
                              : 'good'
                          }
                        />
                      </div>
                      <div className="mt-2 max-h-[180px] overflow-y-auto">
                        <table className="w-full text-[10px]">
                          <thead className="sticky top-0 bg-card/95 text-muted-foreground">
                            <tr>
                              <th className="text-left">{t('backtestStudio.cpcvColPath')}</th>
                              <th className="text-left">{t('backtestStudio.cpcvColTestFolds')}</th>
                              <th className="text-right">{t('backtestStudio.cpcvColTrades')}</th>
                              <th className="text-right">{t('backtestStudio.cpcvColReturn')}</th>
                              <th className="text-right">{t('backtestStudio.cpcvColSharpe')}</th>
                              <th className="text-right">{t('backtestStudio.cpcvColMaxDd')}</th>
                            </tr>
                          </thead>
                          <tbody>
                            {cpcvResult.paths.map((p) => (
                              <tr key={p.path_index} className="border-t border-border/20">
                                <td className="font-mono">#{p.path_index}</td>
                                <td className="font-mono text-muted-foreground">
                                  {`{${p.test_fold_indices.join(',')}}`}
                                </td>
                                <td className="text-right font-mono tabular-nums">{p.trade_count}</td>
                                <td className={cn('text-right font-mono tabular-nums', p.total_return_pct >= 0 ? 'text-emerald-300' : 'text-rose-300')}>
                                  {fmtPct(p.total_return_pct, 1)}
                                </td>
                                <td className="text-right font-mono tabular-nums">
                                  {p.sharpe != null ? p.sharpe.toFixed(2) : '—'}
                                </td>
                                <td className="text-right font-mono tabular-nums">{fmtPct(p.max_drawdown_pct, 1)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      <div className="mt-2 text-[10px] text-muted-foreground">
                        {t('backtestStudio.cpcvFootnote', { seconds: cpcvResult.embargo_seconds.toFixed(0) })}
                      </div>
                    </>
                  ) : (
                    <div className="text-[11px] text-muted-foreground italic" dangerouslySetInnerHTML={{ __html: t('backtestStudio.cpcvEmpty', { folds: cpcvNFolds, kTest: cpcvKTest, combinations: (() => {
                      const f = (n: number): number => (n <= 1 ? 1 : n * f(n - 1))
                      return Math.round(f(cpcvNFolds) / (f(cpcvKTest) * f(cpcvNFolds - cpcvKTest)))
                    })() }) }} />
                  )}
                </div>
              )}

              {/* LATENCY MONTE CARLO */}
              {centerTab === 'robustness' && (
                <div className="rounded-md border border-border/50 bg-card/40 p-3">
                  <div className="mb-2 flex items-center gap-1.5 text-xs font-medium">
                    <Clock className="h-3.5 w-3.5 text-sky-300" />
                    {t('backtestStudio.latencyMcTitle')}
                    <Button
                      size="sm"
                      className="ml-auto h-6 text-[10px]"
                      onClick={() => {
                        if (sourceCode.trim().length < 10) return
                        const end = new Date()
                        const start = new Date(end.getTime() - 7 * 24 * 3600_000)
                        latencyMcMutation.mutate({
                          source_code: sourceCode,
                          slug: slug || '_backtest_mc_latency',
                          start: start.toISOString(),
                          end: end.toISOString(),
                          multipliers: [0.5, 0.75, 1.0, 1.5, 2.0],
                        })
                      }}
                      disabled={latencyMcMutation.isPending || sourceCode.trim().length < 10}
                    >
                      {latencyMcMutation.isPending ? t('backtestStudio.cpcvRunning') : t('backtestStudio.latencyMcRun')}
                    </Button>
                  </div>
                  {latencyMcResult ? (
                    <>
                      <div className="grid grid-cols-4 gap-2">
                        <StatTile
                          label={t('backtestStudio.latencyMcSharpeBaseline')}
                          value={fmtNum(latencyMcResult.summary.sharpe_at_baseline ?? 0, 2)}
                          hint={t('backtestStudio.latencyMcSharpeBaselineHint')}
                        />
                        <StatTile
                          label={t('backtestStudio.latencyMcSharpeBest')}
                          value={fmtNum(latencyMcResult.summary.sharpe_at_best_latency ?? 0, 2)}
                          hint={t('backtestStudio.latencyMcSharpeBestHint')}
                          tone="good"
                        />
                        <StatTile
                          label={t('backtestStudio.latencyMcSharpeWorst')}
                          value={fmtNum(latencyMcResult.summary.sharpe_at_worst_latency ?? 0, 2)}
                          hint={t('backtestStudio.latencyMcSharpeWorstHint')}
                          tone="warn"
                        />
                        <StatTile
                          label={t('backtestStudio.latencyMcSlope')}
                          value={fmtNum(latencyMcResult.summary.sharpe_slope_per_x_latency ?? 0, 2)}
                          hint={
                            (latencyMcResult.summary.sharpe_slope_per_x_latency ?? 0) < -0.3
                              ? t('backtestStudio.latencyMcLatencySensitive')
                              : t('backtestStudio.latencyMcLatencyRobust')
                          }
                          tone={
                            (latencyMcResult.summary.sharpe_slope_per_x_latency ?? 0) < -0.5 ? 'bad'
                              : (latencyMcResult.summary.sharpe_slope_per_x_latency ?? 0) < -0.2 ? 'warn'
                              : 'good'
                          }
                        />
                      </div>
                      <div className="mt-2 max-h-[140px] overflow-y-auto">
                        <table className="w-full text-[10px]">
                          <thead className="sticky top-0 bg-card/95 text-muted-foreground">
                            <tr>
                              <th className="text-left">{t('backtestStudio.latencyMcColMult')}</th>
                              <th className="text-right">{t('backtestStudio.latencyMcColSubmitP95')}</th>
                              <th className="text-right">{t('backtestStudio.latencyMcColTrades')}</th>
                              <th className="text-right">{t('backtestStudio.latencyMcColReturn')}</th>
                              <th className="text-right">{t('backtestStudio.latencyMcColSharpe')}</th>
                              <th className="text-right">{t('backtestStudio.latencyMcColMaxDd')}</th>
                            </tr>
                          </thead>
                          <tbody>
                            {latencyMcResult.runs.map((r, i) => (
                              <tr key={i} className="border-t border-border/20">
                                <td className="font-mono">{r.p95_multiplier.toFixed(2)}×</td>
                                <td className="text-right font-mono tabular-nums">{r.submit_p95_ms.toFixed(0)} ms</td>
                                <td className="text-right font-mono tabular-nums">{r.trade_count}</td>
                                <td className={cn('text-right font-mono tabular-nums', r.total_return_pct >= 0 ? 'text-emerald-300' : 'text-rose-300')}>
                                  {fmtPct(r.total_return_pct, 1)}
                                </td>
                                <td className="text-right font-mono tabular-nums">
                                  {r.sharpe != null ? r.sharpe.toFixed(2) : '—'}
                                </td>
                                <td className="text-right font-mono tabular-nums">{fmtPct(r.max_drawdown_pct, 1)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      <div className="mt-2 text-[10px] text-muted-foreground">
                        {t('backtestStudio.latencyMcFootnote')}
                      </div>
                    </>
                  ) : (
                    <div className="text-[11px] text-muted-foreground italic" dangerouslySetInnerHTML={{ __html: t('backtestStudio.latencyMcEmpty') }} />
                  )}
                </div>
              )}

              {/* DATA QUALITY (Fill quality subtab) */}
              {centerTab === 'fill_quality' && activeRun?.data_quality ? (
                <div className="rounded-md border border-border/50 bg-card/40 p-3">
                  <div className="mb-2 flex items-center gap-1.5 text-xs font-medium">
                    <AlertTriangle className="h-3.5 w-3.5 text-rose-300" />
                    {t('backtestStudio.dataQualityTitle')}
                    <span className="ml-auto text-[10px] text-muted-foreground">
                      {t('backtestStudio.dataQualitySub')}
                    </span>
                  </div>
                  <div className="grid grid-cols-4 gap-2">
                    <StatTile
                      label={t('backtestStudio.dqAcceptRate')}
                      value={
                        activeRun.data_quality.accept_rate != null
                          ? `${(activeRun.data_quality.accept_rate * 100).toFixed(1)}%`
                          : '—'
                      }
                      hint={t('backtestStudio.dqAcceptRateHint', { accepted: activeRun.data_quality.accepted_books.toLocaleString(), total: activeRun.data_quality.total_attempts.toLocaleString() })}
                      tone={
                        (activeRun.data_quality.accept_rate ?? 1) >= 0.99 ? 'good'
                          : (activeRun.data_quality.accept_rate ?? 1) >= 0.95 ? 'warn'
                          : 'bad'
                      }
                    />
                    <StatTile
                      label={t('backtestStudio.dqSeqGaps')}
                      value={activeRun.data_quality.sequence_gaps_observed.toLocaleString()}
                      hint={t('backtestStudio.dqSeqGapsHint', { n: activeRun.data_quality.tokens_tracked })}
                      tone={activeRun.data_quality.sequence_gaps_observed > 100 ? 'warn' : 'neutral'}
                    />
                    <StatTile
                      label={t('backtestStudio.dqQueueDropped')}
                      value={activeRun.data_quality.queue_dropped.toLocaleString()}
                      hint={t('backtestStudio.dqQueueDroppedHint')}
                      tone={activeRun.data_quality.queue_dropped > 0 ? 'warn' : 'good'}
                    />
                    <StatTile
                      label={t('backtestStudio.dqTotalRejects')}
                      value={Object.values(activeRun.data_quality.rejects_by_reason || {})
                        .reduce((a, b) => a + b, 0)
                        .toLocaleString()}
                      hint={t('backtestStudio.dqTotalRejectsHint')}
                    />
                  </div>
                  {Object.entries(activeRun.data_quality.rejects_by_reason || {}).some(([, n]) => n > 0) ? (
                    <div className="mt-2 grid grid-cols-2 gap-1 text-[10px] md:grid-cols-3">
                      {Object.entries(activeRun.data_quality.rejects_by_reason || {})
                        .filter(([, n]) => n > 0)
                        .map(([reason, n]) => (
                          <div key={reason} className="flex items-center justify-between rounded-sm bg-rose-500/10 px-1.5 py-0.5">
                            <span className="text-rose-200">{reason}</span>
                            <span className="font-mono tabular-nums text-rose-300">{n.toLocaleString()}</span>
                          </div>
                        ))}
                    </div>
                  ) : (
                    <div className="mt-2 rounded-sm bg-emerald-500/10 px-2 py-1 text-[10px] text-emerald-200">
                      {t('backtestStudio.dqNoRejects')}
                    </div>
                  )}
                </div>
              ) : null}

              {/* RUNTIME ERRORS */}
              {exec?.runtime_error ? (
                <div className="rounded-md border border-red-500/30 bg-red-500/5 p-3 text-xs text-red-300">
                  <div className="flex items-center gap-1.5 font-medium">
                    <AlertTriangle className="h-3.5 w-3.5" />
                    {t('backtestStudio.runtimeError')}
                  </div>
                  <pre className="mt-1 whitespace-pre-wrap font-mono text-[10px]">
                    {exec.runtime_error}
                  </pre>
                </div>
              ) : null}
              </>
            ) : null}
          </div>
        </ScrollArea>

        {/* RIGHT RAIL — microstructure / fill model */}
        <div className="flex w-[300px] shrink-0 flex-col border-l border-border/50 bg-background/40">
          <ScrollArea className="flex-1 min-h-0">
            <div className="space-y-3 p-3">
              {/* FILL MODEL */}
              <div className="rounded-md border border-border/50 bg-card/40 p-3">
                <div className="mb-1 flex items-center gap-1.5 text-xs font-medium">
                  <Sparkles className="h-3.5 w-3.5 text-amber-300" />
                  {t('backtestStudio.fillModelTitle')}
                </div>
                {fillModel?.loaded ? (
                  <>
                    <div className="grid grid-cols-2 gap-1.5">
                      <StatTile
                        label={t('backtestStudio.cIndex')}
                        value={fillModel.concordance_index != null ? fmtNum(fillModel.concordance_index, 3) : '—'}
                        tone={
                          fillModel.concordance_index != null
                            ? fillModel.concordance_index > 0.62
                              ? 'good'
                              : fillModel.concordance_index > 0.55
                                ? 'warn'
                                : 'bad'
                            : 'neutral'
                        }
                      />
                      <StatTile
                        label={t('backtestStudio.events')}
                        value={(fillModel.n_events ?? 0).toLocaleString()}
                      />
                    </div>
                    {fillModel.coefficients && Object.keys(fillModel.coefficients).length > 0 ? (
                      <div className="mt-2">
                        <div className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground">
                          {t('backtestStudio.hazardRatios')}
                        </div>
                        <div className="space-y-0">
                          {Object.entries(fillModel.coefficients)
                            .sort((a, b) => Math.abs(Math.log(b[1])) - Math.abs(Math.log(a[1])))
                            .slice(0, 8)
                            .map(([cov, hr]) => (
                              <HazardBar key={cov} label={cov} hr={hr} />
                            ))}
                        </div>
                      </div>
                    ) : (
                      <div className="mt-2 text-[10px] text-muted-foreground italic">
                        {t('backtestStudio.kmBaselineNote')}
                      </div>
                    )}
                    {fillModel.calibration_bins && fillModel.calibration_bins.length >= 3 ? (
                      <div className="mt-2">
                        <div className="mb-1 flex items-center justify-between text-[10px] uppercase tracking-wide text-muted-foreground">
                          <span>{t('backtestStudio.calibrationTitle')}</span>
                          <span>{t('backtestStudio.calibrationBins', { n: fillModel.calibration_bins.length })}</span>
                        </div>
                        <CalibrationPlot bins={fillModel.calibration_bins} />
                      </div>
                    ) : null}
                  </>
                ) : (
                  <div className="text-[11px] text-muted-foreground italic">
                    {t('backtestStudio.noActiveModel')}
                  </div>
                )}
              </div>

              {/* LATENCY */}
              {latency ? (
                <div className="rounded-md border border-border/50 bg-card/40 p-3">
                  <div className="mb-1 flex items-center gap-1.5 text-xs font-medium">
                    <Clock className="h-3.5 w-3.5 text-sky-300" />
                    {latency.sample_count > 0 ? t('backtestStudio.measuredLatency') : t('backtestStudio.latencyDefaults')}
                    {latency.sample_count === 0 ? (
                      <span className="ml-auto rounded bg-amber-500/10 px-1.5 py-0.5 text-[9px] uppercase tracking-wide text-amber-300">
                        {t('backtestStudio.noSamples')}
                      </span>
                    ) : null}
                  </div>
                  <div className={`grid grid-cols-3 gap-1.5 ${latency.sample_count === 0 ? 'opacity-60' : ''}`}>
                    <StatTile label={t('backtestStudio.p50')} value={`${Math.round(latency.p50_ms)}ms`} />
                    <StatTile label={t('backtestStudio.p95')} value={`${Math.round(latency.p95_ms)}ms`} tone={latency.sample_count > 0 && latency.p95_ms > 800 ? 'warn' : 'neutral'} />
                    <StatTile label={t('backtestStudio.p99')} value={`${Math.round(latency.p99_ms)}ms`} tone={latency.sample_count > 0 && latency.p99_ms > 1500 ? 'bad' : 'neutral'} />
                  </div>
                  <div className="mt-1 text-[10px] text-muted-foreground">
                    {latency.sample_count > 0
                      ? t('backtestStudio.latencyDetails', { n: latency.sample_count.toLocaleString(), pess: Math.round(latency.pessimistic_ms), real: Math.round(latency.realistic_ms), opt: Math.round(latency.optimistic_ms) })
                      : t('backtestStudio.latencyDefaultsNote')}
                  </div>
                </div>
              ) : null}

              {/* DECOMPOSITION */}
              {decomp ? (
                <div className="rounded-md border border-border/50 bg-card/40 p-3">
                  <div className="mb-1 flex items-center gap-1.5 text-xs font-medium">
                    <Layers3 className="h-3.5 w-3.5 text-violet-700 dark:text-violet-300" />
                    {t('backtestStudio.tradeVsCancel', { hours: decomp.window_hours })}
                  </div>
                  <div className="grid grid-cols-2 gap-1.5">
                    <StatTile
                      label={t('backtestStudio.trades')}
                      value={decomp.trade_count.toLocaleString()}
                      hint={decomp.trade_count_pct != null ? `${fmtNum(decomp.trade_count_pct, 1)}%` : undefined}
                      tone="good"
                    />
                    <StatTile
                      label={t('backtestStudio.cancels')}
                      value={decomp.cancel_count.toLocaleString()}
                      hint={decomp.trade_count_pct != null ? `${fmtNum(100 - decomp.trade_count_pct, 1)}%` : undefined}
                      tone={decomp.trade_count_pct != null && decomp.trade_count_pct < 30 ? 'warn' : 'neutral'}
                    />
                  </div>
                  <div className="mt-1 text-[10px] text-muted-foreground">
                    {t('backtestStudio.cancelRateNote')}
                  </div>
                </div>
              ) : null}

              {/* EMPIRICAL CONSTANTS */}
              {constants ? (
                <div className="rounded-md border border-border/50 bg-card/40 p-3">
                  <div className="mb-1 flex items-center gap-1.5 text-xs font-medium">
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-300" />
                    {t('backtestStudio.empiricalConstants')}
                    <Badge
                      className={cn(
                        'ml-auto text-[9px]',
                        constants.measured
                          ? 'bg-emerald-500/10 text-emerald-300'
                          : 'bg-amber-500/10 text-amber-300',
                      )}
                    >
                      {constants.measured ? t('backtestStudio.measured') : t('backtestStudio.defaults')}
                    </Badge>
                  </div>
                  <div className="space-y-0.5 text-[11px]">
                    {Object.entries(constants.values).map(([k, v]) => (
                      <div key={k} className="grid grid-cols-[1fr,60px] items-center gap-1">
                        <span className="truncate text-muted-foreground">{k.replace(/_/g, ' ')}</span>
                        <span className="text-right font-mono tabular-nums">{fmtNum(v, 3)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}
            </div>
          </ScrollArea>
        </div>
      </div>
    </div>
  )
}


/**
 * Compact picker for imported provider datasets.
 *
 * Pulls the catalog from /api/providers/datasets, lets the operator
 * multi-select.  When at least one dataset is selected the run uses
 * the union of those datasets' (token_ids, window) instead of the
 * standard Window field — backend resolves on the server side via
 * the `provider_dataset_ids` request field.
 */
function ProviderDatasetSelector({
  selected,
  onChange,
}: {
  selected: string[]
  onChange: (ids: string[]) => void
}) {
  const { t } = useTranslation()
  const datasetsQuery = useQuery({
    queryKey: ['providers', 'datasets', 'backtest-picker'],
    queryFn: () => listProviderDatasets({ limit: 200 }),
    staleTime: 60_000,
  })
  const datasets: ProviderDataset[] = datasetsQuery.data ?? []
  const [open, setOpen] = useState(false)

  const selectedSet = useMemo(() => new Set(selected), [selected])
  const summary = useMemo(() => {
    if (selected.length === 0) return t('backtestStudio.providerDatasetNone')
    if (selected.length === 1) {
      const d = datasets.find((x) => x.id === selected[0])
      return d ? (d.title || d.external_slug || d.external_id) : t('backtestStudio.providerDatasetCount', { n: selected.length })
    }
    return t('backtestStudio.providerDatasetCount', { n: selected.length })
  }, [selected, datasets, t])

  return (
    <div className="rounded-md border border-violet-500/20 bg-violet-500/5 p-2">
      <div className="flex items-center justify-between gap-2">
        <Label className="text-[10px] uppercase tracking-wide text-violet-700 dark:text-violet-300">
          {t('backtestStudio.providerDataset')}
        </Label>
        {selected.length > 0 ? (
          <button
            type="button"
            className="text-[10px] text-muted-foreground hover:text-foreground"
            onClick={() => onChange([])}
          >
            {t('backtestStudio.clearAll')}
          </button>
        ) : null}
      </div>
      <button
        type="button"
        className="mt-1 flex w-full items-center justify-between rounded-sm border border-border/40 bg-background/40 px-2 py-1 text-[11px] hover:bg-background/60"
        onClick={() => setOpen((o) => !o)}
      >
        <span className="truncate">{summary}</span>
        <span className="text-muted-foreground">{open ? '▴' : '▾'}</span>
      </button>
      {open ? (
        <div className="mt-1 max-h-44 overflow-auto rounded-sm border border-border/30 bg-background/40 p-1">
          {datasetsQuery.isLoading ? (
            <div className="px-2 py-1 text-[10px] text-muted-foreground">{t('backtestStudio.providerDatasetLoading')}</div>
          ) : datasets.length === 0 ? (
            <div className="px-2 py-1 text-[10px] text-muted-foreground">
              {t('backtestStudio.providerDatasetEmpty')}
            </div>
          ) : (
            datasets.map((d) => {
              const checked = selectedSet.has(d.id)
              return (
                <label
                  key={d.id}
                  className="flex cursor-pointer items-center gap-1.5 rounded-sm px-1.5 py-1 hover:bg-card/40"
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={(e) => {
                      const next = new Set(selectedSet)
                      if (e.target.checked) next.add(d.id)
                      else next.delete(d.id)
                      onChange(Array.from(next))
                    }}
                    className="h-3 w-3 accent-violet-500"
                  />
                  <span className="flex-1 truncate text-[10.5px]">
                    {d.title || d.external_slug || d.external_id}
                  </span>
                  <span className="ml-1 text-[9px] text-muted-foreground">
                    {(d.coin || '?').toUpperCase()} · {d.snapshot_count.toLocaleString()}
                  </span>
                </label>
              )
            })
          )}
        </div>
      ) : null}
    </div>
  )
}
