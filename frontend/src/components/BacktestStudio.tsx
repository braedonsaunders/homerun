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
import { useEffect, useMemo, useState } from 'react'
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
  Sparkles,
  TrendingDown,
  TrendingUp,
  Zap,
} from 'lucide-react'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { ScrollArea } from './ui/scroll-area'
import { cn } from '../lib/utils'
import {
  type BacktestRunSummary,
  type PortfolioCorrelationResult,
  type UnifiedBacktestResult,
  type WalkForwardResult,
  getBacktestRun,
  getPortfolioCorrelation,
  listBacktestRuns,
  runUnifiedBacktest,
  runWalkForward,
} from '../services/apiBacktest'
import {
  getActiveFillModel,
  getDecompositionSummary,
  getEmpiricalConstants,
  getLatencyDistribution,
  getTriangulation,
} from '../services/apiFillModel'

interface BacktestStudioProps {
  initialSourceCode?: string
  initialSlug?: string
  initialConfig?: Record<string, unknown>
  // Human-readable strategy label shown in the "loaded" indicator so
  // the operator can verify which strategy is staged when switching
  // between them in the parent dropdown.  Optional — falls back to
  // the slug.
  strategyLabel?: string | null
}

function fmtNum(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(Number(value))) return '—'
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
  if (!points || points.length < 2) {
    return (
      <div className="rounded-md border border-dashed border-border/50 bg-card/30 px-3 py-6 text-center text-xs text-muted-foreground">
        Equity curve will appear after the first run.
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
        <span>{`equity ${fmtUsd(initial)} → ${fmtUsd(ending)}`}</span>
        <span>{`${points.length} samples`}</span>
      </div>
      <svg width={w} height={h} className="mt-1">
        <line x1={8} y1={yBaseline} x2={w - 8} y2={yBaseline} stroke="rgb(120,120,120)" strokeOpacity={0.35} strokeDasharray="3,3" strokeWidth={0.5} />
        <path d={path} fill="none" stroke={isUp ? 'hsl(150, 80%, 55%)' : 'hsl(0, 80%, 60%)'} strokeWidth={1.5} />
      </svg>
    </div>
  )
}

function CorrelationHeatmap({ result }: { result: PortfolioCorrelationResult }) {
  const { strategies, correlation_matrix, summary } = result
  if (!strategies || strategies.length === 0) {
    return (
      <div className="text-[11px] text-muted-foreground italic">
        No strategies have ≥ 5 terminal trades in the last {result.window_days} days. Cross-strategy
        correlation needs daily PnL series to compute.
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
          label="Diversification"
          value={`${(summary.diversification_ratio * 100).toFixed(0)}%`}
          hint="1 - mean(|ρ|), higher = better"
          tone={summary.diversification_ratio >= 0.7 ? 'good' : summary.diversification_ratio >= 0.4 ? 'warn' : 'bad'}
        />
        <StatTile
          label="Mean |ρ|"
          value={summary.mean_abs_pairwise_correlation.toFixed(2)}
          hint={`min ${summary.min_pairwise_correlation.toFixed(2)} · max ${summary.max_pairwise_correlation.toFixed(2)}`}
          tone={summary.mean_abs_pairwise_correlation >= 0.5 ? 'bad' : summary.mean_abs_pairwise_correlation >= 0.3 ? 'warn' : 'good'}
        />
        <StatTile
          label="Strategies"
          value={`${summary.n_strategies}`}
          hint={`${summary.n_days} days of PnL`}
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
        Red cells (ρ &gt; 0.4) ⇒ strategies that drew down together. Green (ρ &lt; -0.4) ⇒ natural
        hedge pair. Diversification ratio above 70% = healthy portfolio.
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
  const maxN = rows.reduce((m, r) => Math.max(m, r.n), 1)
  return (
    <div className="rounded-sm border border-border/40 bg-background/40 p-2">
      <div className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground">{title}</div>
      {rows.length === 0 ? (
        <div className="text-[10px] text-muted-foreground italic">no data</div>
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
        <span>0 predicted</span>
        <span>diagonal = perfect</span>
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

function EnsembleBand({ band }: { band: UnifiedBacktestResult['ensemble_band'] }) {
  if (!band || band.length === 0) {
    return (
      <div className="text-xs text-muted-foreground italic">
        No ensemble samples for this run (no fills had captured book context).
      </div>
    )
  }
  return (
    <div className="space-y-1">
      {band.map((b, i) => (
        <div key={`${b.fill_id || i}-${i}`} className="rounded-sm border border-border/40 bg-background/40 px-2 py-1.5">
          <div className="flex items-center justify-between text-[10px] text-muted-foreground">
            <span>fill #{i + 1}</span>
            {b.cox_loaded ? <Badge className="bg-emerald-500/10 text-emerald-300 text-[9px]">cox</Badge> : <Badge variant="outline" className="text-[9px]">heuristic</Badge>}
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
  if (!rows || rows.length === 0) {
    return (
      <div className="text-xs text-muted-foreground italic">
        No counterfactual replays available for this run (no recorded book deltas in the fill window).
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
                {r.expired ? 'expired' : `filled ${(fillRatio * 100).toFixed(0)}%`}
              </span>
            </div>
            <div className="mt-0.5 flex flex-wrap gap-2 text-[10px] text-muted-foreground">
              <span>queue {fmtNum(r.final_queue_ahead, 0)}</span>
              <span>trades-ahead {fmtNum(r.trades_ahead_observed, 0)}</span>
              <span>cancels-ahead {fmtNum(r.cancels_ahead_observed, 0)}</span>
              {r.time_to_fill_seconds != null ? <span>ttf {fmtNum(r.time_to_fill_seconds, 1)}s</span> : null}
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
  if (runs.length === 0) {
    return (
      <div className="px-3 py-3 text-[11px] text-muted-foreground italic">
        No runs yet. Click <strong>Run backtest</strong> to start.
      </div>
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
              {run.strategy_name || run.strategy_slug || 'unknown strategy'}
            </div>
            <div className="flex items-center justify-between text-[10px] text-muted-foreground">
              <span>
                {run.trade_count} trades · {fmtMs(run.total_time_ms)}
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
  initialConfig,
  strategyLabel,
}: BacktestStudioProps) {
  const queryClient = useQueryClient()

  // Run controls.  Internal state is initialized from props once at
  // mount.  Parent prop changes (i.e. operator switches strategies in
  // the dropdown above) flow into the state via the useEffect below.
  const [sourceCode, setSourceCode] = useState<string>(initialSourceCode || '')
  const [slug, setSlug] = useState<string>(initialSlug || '_backtest_unified')
  // Remember the previous slug so we can show a brief "loaded
  // <slug>" highlight when the strategy actually changes.
  const [justLoadedSlug, setJustLoadedSlug] = useState<string | null>(null)
  useEffect(() => {
    const nextSource = initialSourceCode || ''
    const nextSlug = initialSlug || '_backtest_unified'
    setSourceCode(nextSource)
    setSlug(nextSlug)
    if (nextSlug && nextSlug !== '_backtest_unified') {
      setJustLoadedSlug(nextSlug)
      const t = window.setTimeout(() => setJustLoadedSlug(null), 1600)
      return () => window.clearTimeout(t)
    }
    return undefined
  }, [initialSourceCode, initialSlug])
  const [initialCapital, setInitialCapital] = useState<string>('1000')
  const [submitP50, setSubmitP50] = useState<string>('')
  const [submitP95, setSubmitP95] = useState<string>('')
  const [seed, setSeed] = useState<string>('')
  const [impactBps, setImpactBps] = useState<string>('')
  const [makerRebateBps, setMakerRebateBps] = useState<string>('')
  // Date range — defaulted blank so the backend's 7d window applies.
  // Operator can extend (e.g. "30" for 30 days, "0" for "now").
  const [windowDays, setWindowDays] = useState<string>('7')
  // Center-pane subtab.  Three coherent groupings + a small status
  // ribbon-equivalent so the workbench doesn't scroll-and-pray.
  const [centerTab, setCenterTab] = useState<'performance' | 'fill_quality' | 'robustness' | 'portfolio'>('performance')

  // Active run.
  const [activeRun, setActiveRun] = useState<UnifiedBacktestResult | null>(null)

  const runsQuery = useQuery({
    queryKey: ['backtest', 'runs'],
    queryFn: listBacktestRuns,
    refetchInterval: 5000,
  })

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
  const triangulationQuery = useQuery({
    queryKey: ['triangulation', initialSlug],
    queryFn: () => getTriangulation(initialSlug || '', 30),
    enabled: Boolean(initialSlug && initialSlug !== '_backtest_unified' && initialSlug !== '_research'),
    refetchInterval: 60_000,
  })

  // Portfolio correlation: pulls live cross-strategy PnL correlation
  // matrix over the last 30 days.  Auto-refresh every 5 minutes.
  const portfolioCorrelationQuery = useQuery({
    queryKey: ['portfolio-correlation', 30],
    queryFn: () => getPortfolioCorrelation(30, 5),
    refetchInterval: 300_000,
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
      config: initialConfig,
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
    onSuccess: (data) => setActiveRun(data),
  })

  const runMutation = useMutation({
    mutationFn: runUnifiedBacktest,
    onSuccess: (data) => {
      setActiveRun(data)
      queryClient.invalidateQueries({ queryKey: ['backtest', 'runs'] })
    },
  })

  const errorMessage = useMemo(() => {
    const err = runMutation.error as { response?: { data?: { detail?: string } }; message?: string } | undefined
    if (!err) return null
    return err.response?.data?.detail || err.message || 'unknown error'
  }, [runMutation.error])

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
      config: initialConfig,
      initial_capital_usd: parseFloat(initialCapital) || 1000,
      start: startIso,
      end: endIso,
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
              <div className="text-sm font-semibold">Backtest Studio</div>
              <div className="text-[11px] text-muted-foreground">
                L2 replay · Cox PH fill model · ensemble PnL bands · counterfactual queue replay
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
                {fillModel.family} · {fillModel.n_events?.toLocaleString()} events
              </Badge>
            ) : (
              <Badge variant="outline" className="text-[10px]">
                no fill model loaded
              </Badge>
            )}
            {latency ? (
              <Badge variant="outline" className="text-[10px] font-mono">
                p50 {Math.round(latency.p50_ms)}ms · p95 {Math.round(latency.p95_ms)}ms
              </Badge>
            ) : null}
            {constants?.measured ? (
              <Badge className="bg-emerald-500/10 text-emerald-300 text-[10px]">
                empirical constants live
              </Badge>
            ) : (
              <Badge variant="outline" className="text-[10px]">
                empirical constants default
              </Badge>
            )}
          </div>
        </div>
      </div>

      {/* MAIN — 3-pane workbench */}
      <div className="flex flex-1 min-h-0">
        {/* LEFT RAIL — controls + history */}
        <div className="flex w-[320px] shrink-0 flex-col border-r border-border/50 bg-background/40">
          <div className="border-b border-border/50 px-3 py-3 space-y-2">
            <div
              className={cn(
                'flex items-center justify-between gap-2 rounded-sm border px-2 py-1.5 transition-colors duration-700',
                justLoadedSlug
                  ? 'border-amber-400/60 bg-amber-500/10'
                  : 'border-border/40 bg-background/40',
              )}
              title="Source loaded from the selected strategy. Edit it in Code Experiments."
            >
              <div className="min-w-0 flex-1">
                <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
                  Strategy source
                </div>
                <div className="truncate font-mono text-[11px] text-foreground">
                  {strategyLabel || slug || 'no strategy'}
                </div>
              </div>
              <div className="shrink-0 text-right">
                <div className="font-mono text-[10px] text-muted-foreground">
                  {sourceCode.length > 0
                    ? `${sourceCode.length.toLocaleString()} chars`
                    : 'no source'}
                </div>
                {justLoadedSlug ? (
                  <div className="text-[9px] uppercase tracking-wide text-amber-300">
                    just loaded
                  </div>
                ) : null}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <Label className="text-[10px] uppercase tracking-wide text-muted-foreground">
                  Capital
                </Label>
                <Input
                  value={initialCapital}
                  onChange={(e) => setInitialCapital(e.target.value)}
                  className="h-7 text-xs"
                />
              </div>
              <div>
                <Label className="text-[10px] uppercase tracking-wide text-muted-foreground">
                  Seed
                </Label>
                <Input
                  value={seed}
                  onChange={(e) => setSeed(e.target.value)}
                  placeholder="auto"
                  className="h-7 text-xs"
                />
              </div>
              <div>
                <Label className="text-[10px] uppercase tracking-wide text-muted-foreground">
                  Latency p50 (ms)
                </Label>
                <Input
                  value={submitP50}
                  onChange={(e) => setSubmitP50(e.target.value)}
                  placeholder="measured"
                  className="h-7 text-xs"
                />
              </div>
              <div>
                <Label className="text-[10px] uppercase tracking-wide text-muted-foreground">
                  Latency p95 (ms)
                </Label>
                <Input
                  value={submitP95}
                  onChange={(e) => setSubmitP95(e.target.value)}
                  placeholder="measured"
                  className="h-7 text-xs"
                />
              </div>
              <div>
                <Label className="text-[10px] uppercase tracking-wide text-muted-foreground" title="How many days of history to backtest against.  7 days is the default; extend for thicker samples, shorten for fast iteration.">
                  Window (days)
                </Label>
                <Input
                  value={windowDays}
                  onChange={(e) => setWindowDays(e.target.value)}
                  placeholder="7"
                  className="h-7 text-xs"
                />
              </div>
              <div>
                <Label className="text-[10px] uppercase tracking-wide text-muted-foreground" title="Square-root impact: bps adverse adjustment when consuming 100% of side depth. 5-10 = deep crypto books; 25-50 = thin event markets. 0 = disabled.">
                  Impact (bps)
                </Label>
                <Input
                  value={impactBps}
                  onChange={(e) => setImpactBps(e.target.value)}
                  placeholder="0 (off)"
                  className="h-7 text-xs"
                />
              </div>
              <div>
                <Label className="text-[10px] uppercase tracking-wide text-muted-foreground" title="Polymarket maker LP-rewards approximation. ~1-3 bps realistic on top crypto markets; >5 bps optimistic. Only paid on inside-band fills.">
                  Maker rebate (bps)
                </Label>
                <Input
                  value={makerRebateBps}
                  onChange={(e) => setMakerRebateBps(e.target.value)}
                  placeholder="0 (off)"
                  className="h-7 text-xs"
                />
              </div>
            </div>

            <Button
              onClick={handleRun}
              disabled={runMutation.isPending || sourceCode.trim().length < 10}
              className="mt-1 w-full"
            >
              {runMutation.isPending ? (
                <Loader2 className="mr-1 h-3.5 w-3.5 animate-spin" />
              ) : (
                <Play className="mr-1 h-3.5 w-3.5" />
              )}
              Run backtest
            </Button>
            {errorMessage ? (
              <div className="mt-1 flex items-start gap-1 text-[10px] text-red-300">
                <AlertTriangle className="h-3 w-3 shrink-0 mt-0.5" />
                <span>{errorMessage}</span>
              </div>
            ) : null}
          </div>

          <div className="px-3 py-2 text-[11px] uppercase tracking-wide text-muted-foreground">
            Recent runs
          </div>
          <ScrollArea className="flex-1 min-h-0">
            <RunHistory
              runs={runsQuery.data ?? []}
              activeId={activeRun?.run_id ?? null}
              onSelect={(run) => loadRunMutation.mutate(run.run_id)}
            />
          </ScrollArea>
        </div>

        {/* CENTER — results */}
        <ScrollArea className="flex-1 min-h-0">
          <div className="space-y-4 p-3">
            {/* SUBTAB STRIP — always visible.  Performance / Fill
                quality / Robustness require an active run to populate;
                Portfolio is run-independent (live cross-strategy
                correlation) and renders the same regardless. */}
            <div className="flex items-center gap-1 border-b border-border/40 pb-1.5">
              {([
                ['performance', 'Performance', TrendingUp],
                ['fill_quality', 'Fill quality', Zap],
                ['robustness', 'Robustness', Activity],
                ['portfolio', 'Portfolio', Layers3],
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

            {/* Empty-state placeholder for run-required tabs when no
                run is loaded. */}
            {!activeRun && centerTab !== 'portfolio' ? (
              <div className="flex flex-col items-center justify-center gap-2 rounded-md border border-dashed border-border/40 bg-card/20 px-6 py-8 text-center">
                <Flame className="h-7 w-7 text-amber-300/50" />
                <div className="text-sm font-medium">
                  {centerTab === 'performance'
                    ? 'No run loaded'
                    : centerTab === 'fill_quality'
                      ? 'No fill data'
                      : 'No robustness data'}
                </div>
                <div className="max-w-[420px] text-xs text-muted-foreground">
                  {centerTab === 'performance'
                    ? 'Pick a strategy and click Run backtest. Headline KPIs, risk-adjusted metrics with deflated Sharpe, and the equity curve appear here.'
                    : centerTab === 'fill_quality'
                      ? 'Run a backtest to populate ensemble PnL bands, counterfactual queue replay, and partial-fill aggregation against the live trade tape.'
                      : 'Run a backtest to populate triangulation, regime decomposition, and walk-forward cross-validation.'}
                </div>
              </div>
            ) : null}

            {/* PORTFOLIO TAB — always renders, never gated on activeRun. */}
            {centerTab === 'portfolio' ? (
              portfolioCorrelationQuery.data && portfolioCorrelationQuery.data.strategies.length > 0 ? (
                <div className="rounded-md border border-border/50 bg-card/40 p-3">
                  <div className="mb-2 flex items-center gap-1.5 text-xs font-medium">
                    <Layers3 className="h-3.5 w-3.5 text-emerald-300" />
                    Portfolio correlation (live, last 30 days)
                    <span className="ml-auto text-[10px] text-muted-foreground">
                      cross-strategy daily PnL
                    </span>
                  </div>
                  <CorrelationHeatmap result={portfolioCorrelationQuery.data} />
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center gap-2 rounded-md border border-dashed border-border/40 bg-card/20 px-6 py-8 text-center">
                  <Layers3 className="h-7 w-7 text-emerald-300/40" />
                  <div className="text-sm font-medium">No portfolio data yet</div>
                  <div className="max-w-[420px] text-xs text-muted-foreground">
                    The portfolio correlation matrix needs at least 2 strategies with ≥ 5
                    terminal trades each in the last 30 days. As trades resolve, this view
                    populates automatically.
                  </div>
                </div>
              )
            ) : null}

            {activeRun ? (
              <>
              {/* HEADLINE KPIS + SECONDARY METRICS — Performance tab. */}
              {centerTab === 'performance' && (
              <>
              <div className="grid grid-cols-4 gap-2">
                <StatTile
                  label="Return"
                  value={fmtPct(exec?.total_return_pct, 2)}
                  hint={fmtUsd((exec?.final_equity_usd ?? 0) - (exec?.initial_capital_usd ?? 0)) + ' net'}
                  tone={totalReturnTone}
                  icon={exec && exec.total_return_pct >= 0 ? TrendingUp : TrendingDown}
                />
                <StatTile
                  label="Sharpe"
                  value={fmtNum(exec?.sharpe?.value, 2)}
                  hint={
                    exec?.sharpe?.ci_low != null && exec?.sharpe?.ci_high != null
                      ? `ci [${fmtNum(exec.sharpe.ci_low, 2)}, ${fmtNum(exec.sharpe.ci_high, 2)}]`
                      : undefined
                  }
                  tone={sharpeTone}
                  icon={Activity}
                />
                <StatTile
                  label="Max drawdown"
                  value={fmtPct(exec?.max_drawdown_pct, 2)}
                  hint={`${fmtMs((exec?.drawdown_duration_seconds ?? 0) * 1000)} duration`}
                  tone={ddTone}
                  icon={TrendingDown}
                />
                <StatTile
                  label="Trades"
                  value={(exec?.trade_count ?? 0).toLocaleString()}
                  hint={`${exec?.total_fills ?? 0} fills · ${exec?.cancelled_orders ?? 0} cancels · ${exec?.rejected_orders ?? 0} rejects`}
                  icon={Zap}
                />
              </div>

              {/* SECONDARY METRICS */}
              <div className="grid grid-cols-2 gap-2">
                <div className="rounded-md border border-border/50 bg-card/40 p-3">
                  <div className="mb-1 flex items-center gap-1.5 text-xs font-medium">
                    <Activity className="h-3.5 w-3.5 text-amber-300" />
                    Risk-adjusted metrics (bootstrap CIs)
                  </div>
                  <MetricRow label="Sharpe" m={exec?.sharpe} tone={sharpeTone === 'bad' ? 'bad' : sharpeTone === 'good' ? 'good' : undefined} />
                  <MetricRow label="Sortino" m={exec?.sortino} />
                  <MetricRow label="Calmar" m={exec?.calmar} />
                  <MetricRow label="Hit rate" m={exec?.hit_rate} />
                  <MetricRow label="Profit factor" m={exec?.profit_factor} />
                  <MetricRow label="Expectancy ($)" m={exec?.expectancy_usd} />
                  {activeRun?.deflated_sharpe ? (
                    <div className="mt-2 border-t border-border/40 pt-2">
                      <div className="flex items-center justify-between text-[10px] uppercase tracking-wide text-muted-foreground">
                        <span>Deflated Sharpe (López de Prado)</span>
                        <span>{activeRun.deflated_sharpe.n_trials} trial{activeRun.deflated_sharpe.n_trials === 1 ? '' : 's'}</span>
                      </div>
                      <div className="mt-1 grid grid-cols-2 gap-1 text-[11px]">
                        <div className="flex items-center justify-between rounded-sm bg-muted/40 px-1.5 py-0.5">
                          <span className="text-muted-foreground">P(true SR &gt; 0)</span>
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
                          <span className="text-muted-foreground">P(SR &gt; SR₀ overfit)</span>
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
                        SR₀ {activeRun.deflated_sharpe.sr_zero.toFixed(2)} (max-of-N noise floor) ·
                        observed {activeRun.deflated_sharpe.observed_sharpe.toFixed(2)} ·
                        {activeRun.deflated_sharpe.deflated_sharpe < 0.95 && activeRun.deflated_sharpe.n_trials > 1
                          ? ' likely overfit ⇒ run holdout'
                          : ' overfit-aware OK'}
                      </div>
                    </div>
                  ) : null}
                </div>

                <div className="rounded-md border border-border/50 bg-card/40 p-3">
                  <div className="mb-1 flex items-center gap-1.5 text-xs font-medium">
                    <LineChartIcon className="h-3.5 w-3.5 text-emerald-300" />
                    Equity curve
                  </div>
                  <EquityCurveChart points={exec?.equity_curve_sample ?? []} />
                  <div className="mt-2 grid grid-cols-3 gap-2">
                    <StatTile
                      label="Avg win"
                      value={fmtUsd(exec?.avg_win_usd)}
                      tone="good"
                    />
                    <StatTile
                      label="Avg loss"
                      value={fmtUsd(exec?.avg_loss_usd)}
                      tone="bad"
                    />
                    <StatTile
                      label="Fees / fill"
                      value={fmtUsd(exec?.fees_per_fill_usd)}
                      hint={fmtUsd(exec?.fees_paid_usd) + ' total'}
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
                    <Layers3 className="h-3.5 w-3.5 text-violet-300" />
                    Ensemble fill probability (p10 / p50 / p90)
                    <span className="ml-auto text-[10px] text-muted-foreground">
                      {activeRun.ensemble_band.length} sample fills
                    </span>
                  </div>
                  <EnsembleBand band={activeRun.ensemble_band} />
                </div>

                <div className="rounded-md border border-border/50 bg-card/40 p-3">
                  <div className="mb-2 flex items-center gap-1.5 text-xs font-medium">
                    <Clock className="h-3.5 w-3.5 text-sky-300" />
                    Counterfactual queue replay
                    <span className="ml-auto text-[10px] text-muted-foreground">
                      {activeRun.counterfactuals.length} sample fills
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
                    Triangulation — backtest vs shadow vs live (last 30 days)
                    <span className="ml-auto text-[10px] text-muted-foreground">
                      same strategy across regimes
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
                            label="Backtest PnL"
                            value={fmtUsd(btPnl)}
                            hint={`${exec?.trade_count ?? 0} trades · ${fmtPct(exec?.total_return_pct, 1)}`}
                            tone={btPnl >= 0 ? 'good' : 'bad'}
                            icon={Flame}
                          />
                          <StatTile
                            label="Shadow PnL"
                            value={fmtUsd(shadowPnl)}
                            hint={
                              shadowMode
                                ? `${shadowMode.orders} orders · ${shadowMode.filled} filled · ${
                                    btPnl !== 0 ? `Δ ${fmtPct(shadowDeltaPct, 0)}` : 'no backtest'
                                  }`
                                : 'no shadow data'
                            }
                            tone={shadowPnl >= 0 ? 'good' : 'bad'}
                          />
                          <StatTile
                            label="Live PnL"
                            value={fmtUsd(livePnl)}
                            hint={
                              liveMode
                                ? `${liveMode.orders} orders · ${liveMode.filled} filled · ${
                                    btPnl !== 0 ? `Δ ${fmtPct(liveDeltaPct, 0)}` : 'no backtest'
                                  }`
                                : 'no live data'
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
                    Δ &gt; 30% between any two regimes ⇒ the fill model is likely the suspect (latency or queue
                    assumptions). Δ &lt; 10% means the simulator is well-calibrated for this strategy.
                  </div>
                </div>
              ) : null}

              {/* PARTIAL FILL AGGREGATES */}
              {centerTab === 'fill_quality' && activeRun?.partial_fills && activeRun.partial_fills.n_orders > 0 ? (
                <div className="rounded-md border border-border/50 bg-card/40 p-3">
                  <div className="mb-2 flex items-center gap-1.5 text-xs font-medium">
                    <Zap className="h-3.5 w-3.5 text-sky-300" />
                    Partial-fill aggregation
                    <span className="ml-auto text-[10px] text-muted-foreground">
                      child fills per parent order
                    </span>
                  </div>
                  <div className="grid grid-cols-4 gap-2">
                    <StatTile
                      label="Instant fills"
                      value={`${(activeRun.partial_fills.instant_fill_rate * 100).toFixed(0)}%`}
                      hint={`${activeRun.partial_fills.n_instant_fills}/${activeRun.partial_fills.n_orders}`}
                      tone={activeRun.partial_fills.instant_fill_rate >= 0.7 ? 'good' : activeRun.partial_fills.instant_fill_rate >= 0.4 ? 'warn' : 'bad'}
                    />
                    <StatTile
                      label="Avg children"
                      value={fmtNum(activeRun.partial_fills.mean_children_per_order, 2)}
                      hint={`max ${activeRun.partial_fills.max_children_per_order}`}
                    />
                    <StatTile
                      label="Intra-order span"
                      value={activeRun.partial_fills.mean_intra_order_seconds > 0 ? fmtMs(activeRun.partial_fills.mean_intra_order_seconds * 1000) : '—'}
                      hint="mean across partials"
                    />
                    <StatTile
                      label="VWAP dispersion"
                      value={`${fmtNum(activeRun.partial_fills.mean_vwap_dispersion_bps, 1)} bps`}
                      hint="price std / VWAP"
                      tone={activeRun.partial_fills.mean_vwap_dispersion_bps > 50 ? 'warn' : 'neutral'}
                    />
                  </div>
                  {activeRun.partial_fills.child_count_distribution.length > 1 ? (
                    <div className="mt-2">
                      <div className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground">
                        child-count distribution
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
                              {d.children}× → {d.n_orders} ord ({pct.toFixed(0)}%)
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  ) : null}
                  <div className="mt-2 text-[10px] text-muted-foreground">
                    Low instant-fill rate ⇒ orders walk the book or queue-decay before completing.
                    High VWAP dispersion ⇒ price moved during the partial fill (slippage cost beyond
                    the headline number).
                  </div>
                </div>
              ) : null}

              {/* REGIME DECOMPOSITION */}
              {centerTab === 'robustness' && activeRun?.regime_breakdown ? (
                <div className="rounded-md border border-border/50 bg-card/40 p-3">
                  <div className="mb-2 flex items-center gap-1.5 text-xs font-medium">
                    <Layers3 className="h-3.5 w-3.5 text-amber-300" />
                    Regime decomposition
                    <span className="ml-auto text-[10px] text-muted-foreground">
                      win-rate by slice
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
                    <RegimeBlock title="Hour of day" rows={activeRun.regime_breakdown.by_hour} />
                    <RegimeBlock title="Day of week" rows={activeRun.regime_breakdown.by_dow} />
                    <RegimeBlock title="Time to resolution" rows={activeRun.regime_breakdown.by_ttr} />
                    <RegimeBlock title="Order size" rows={activeRun.regime_breakdown.by_size} />
                  </div>
                  <div className="mt-2 text-[10px] text-muted-foreground">
                    Lopsided win-rate across one slice (e.g. 80% on Tue but 30% Mon-Fri-Sat) ⇒ strategy
                    works only in one regime. Healthy strategies have flat-ish bars across all four
                    decompositions.
                  </div>
                </div>
              ) : null}

              {/* WALK-FORWARD ANALYSIS */}
              {centerTab === 'robustness' && (
              <div className="rounded-md border border-border/50 bg-card/40 p-3">
                <div className="mb-2 flex items-center gap-1.5 text-xs font-medium">
                  <Activity className="h-3.5 w-3.5 text-violet-300" />
                  Walk-forward analysis
                  <div className="ml-auto flex items-center gap-1.5">
                    <select
                      value={walkForwardMode}
                      onChange={(e) => setWalkForwardMode(e.target.value as 'anchored' | 'rolling')}
                      className="h-6 rounded-sm border border-border/40 bg-background/60 px-1.5 text-[10px]"
                    >
                      <option value="anchored">anchored</option>
                      <option value="rolling">rolling</option>
                    </select>
                    <select
                      value={walkForwardFolds}
                      onChange={(e) => setWalkForwardFolds(parseInt(e.target.value, 10))}
                      className="h-6 rounded-sm border border-border/40 bg-background/60 px-1.5 text-[10px]"
                    >
                      {[3, 4, 6, 8, 10, 12].map((n) => (
                        <option key={n} value={n}>
                          {n} folds
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
                      Run
                    </Button>
                  </div>
                </div>
                {walkForwardResult ? (
                  <>
                    <div className="grid grid-cols-4 gap-2">
                      <StatTile
                        label="Stable folds"
                        value={`${walkForwardResult.summary.stable_window_pct.toFixed(0)}%`}
                        hint={`${walkForwardResult.summary.n_windows_succeeded}/${walkForwardResult.summary.n_windows_run} succeeded`}
                        tone={walkForwardResult.summary.stable_window_pct >= 70 ? 'good' : walkForwardResult.summary.stable_window_pct >= 50 ? 'warn' : 'bad'}
                      />
                      <StatTile
                        label="Mean return"
                        value={fmtPct(walkForwardResult.summary.mean_return_pct, 2)}
                        hint={`min ${fmtPct(walkForwardResult.summary.min_return_pct, 1)} · max ${fmtPct(walkForwardResult.summary.max_return_pct, 1)}`}
                        tone={walkForwardResult.summary.mean_return_pct >= 0 ? 'good' : 'bad'}
                      />
                      <StatTile
                        label="Mean Sharpe"
                        value={walkForwardResult.summary.mean_sharpe != null ? fmtNum(walkForwardResult.summary.mean_sharpe, 2) : '—'}
                        hint={
                          walkForwardResult.summary.min_sharpe != null && walkForwardResult.summary.max_sharpe != null
                            ? `min ${fmtNum(walkForwardResult.summary.min_sharpe, 2)} · max ${fmtNum(walkForwardResult.summary.max_sharpe, 2)}`
                            : undefined
                        }
                        tone={walkForwardResult.summary.mean_sharpe != null && walkForwardResult.summary.mean_sharpe > 1.0 ? 'good' : 'neutral'}
                      />
                      <StatTile
                        label="Mode"
                        value={walkForwardResult.mode}
                        hint={`${walkForwardResult.n_windows_run} folds`}
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
                          <span className="font-mono text-muted-foreground">fold {w.index}</span>
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
                            {w.trade_count}t
                          </span>
                          <span className="text-right font-mono tabular-nums text-muted-foreground">
                            {fmtUsd(w.final_equity_usd)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </>
                ) : (
                  <div className="text-[11px] text-muted-foreground italic">
                    Click <strong>Run</strong> to split the last 14 days into {walkForwardFolds} {walkForwardMode} folds and
                    backtest each separately. Stable returns across folds = real edge; high variance = overfit.
                  </div>
                )}
              </div>
              )}

              {/* RUNTIME ERRORS */}
              {exec?.runtime_error ? (
                <div className="rounded-md border border-red-500/30 bg-red-500/5 p-3 text-xs text-red-300">
                  <div className="flex items-center gap-1.5 font-medium">
                    <AlertTriangle className="h-3.5 w-3.5" />
                    Runtime error
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
                  Fill probability model
                </div>
                {fillModel?.loaded ? (
                  <>
                    <div className="grid grid-cols-2 gap-1.5">
                      <StatTile
                        label="C-index"
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
                        label="Events"
                        value={(fillModel.n_events ?? 0).toLocaleString()}
                      />
                    </div>
                    {fillModel.coefficients && Object.keys(fillModel.coefficients).length > 0 ? (
                      <div className="mt-2">
                        <div className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground">
                          hazard ratios (per 1 σ)
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
                        KM baseline — no covariate ratios. Cox will train once new orders accumulate
                        with full survival_features.
                      </div>
                    )}
                    {fillModel.calibration_bins && fillModel.calibration_bins.length >= 3 ? (
                      <div className="mt-2">
                        <div className="mb-1 flex items-center justify-between text-[10px] uppercase tracking-wide text-muted-foreground">
                          <span>calibration (predicted vs observed)</span>
                          <span>{fillModel.calibration_bins.length} bins</span>
                        </div>
                        <CalibrationPlot bins={fillModel.calibration_bins} />
                      </div>
                    ) : null}
                  </>
                ) : (
                  <div className="text-[11px] text-muted-foreground italic">
                    No active model. Trigger a retrain in Strategies → Machine Learning → Fill Model.
                  </div>
                )}
              </div>

              {/* LATENCY */}
              {latency ? (
                <div className="rounded-md border border-border/50 bg-card/40 p-3">
                  <div className="mb-1 flex items-center gap-1.5 text-xs font-medium">
                    <Clock className="h-3.5 w-3.5 text-sky-300" />
                    {latency.sample_count > 0 ? 'Measured latency' : 'Latency (defaults)'}
                    {latency.sample_count === 0 ? (
                      <span className="ml-auto rounded bg-amber-500/10 px-1.5 py-0.5 text-[9px] uppercase tracking-wide text-amber-300">
                        no samples
                      </span>
                    ) : null}
                  </div>
                  <div className={`grid grid-cols-3 gap-1.5 ${latency.sample_count === 0 ? 'opacity-60' : ''}`}>
                    <StatTile label="p50" value={`${Math.round(latency.p50_ms)}ms`} />
                    <StatTile label="p95" value={`${Math.round(latency.p95_ms)}ms`} tone={latency.sample_count > 0 && latency.p95_ms > 800 ? 'warn' : 'neutral'} />
                    <StatTile label="p99" value={`${Math.round(latency.p99_ms)}ms`} tone={latency.sample_count > 0 && latency.p99_ms > 1500 ? 'bad' : 'neutral'} />
                  </div>
                  <div className="mt-1 text-[10px] text-muted-foreground">
                    {latency.sample_count > 0
                      ? `${latency.sample_count.toLocaleString()} samples · ensemble uses pessimistic=${Math.round(latency.pessimistic_ms)}ms / realistic=${Math.round(latency.realistic_ms)}ms / optimistic=${Math.round(latency.optimistic_ms)}ms`
                      : `No measured fills yet — values shown are hardcoded fallbacks. Run a strategy long enough to capture submit/cancel timestamps to replace these.`}
                  </div>
                </div>
              ) : null}

              {/* DECOMPOSITION */}
              {decomp ? (
                <div className="rounded-md border border-border/50 bg-card/40 p-3">
                  <div className="mb-1 flex items-center gap-1.5 text-xs font-medium">
                    <Layers3 className="h-3.5 w-3.5 text-violet-300" />
                    Trade vs cancel ({decomp.window_hours}h)
                  </div>
                  <div className="grid grid-cols-2 gap-1.5">
                    <StatTile
                      label="Trades"
                      value={decomp.trade_count.toLocaleString()}
                      hint={decomp.trade_count_pct != null ? `${fmtNum(decomp.trade_count_pct, 1)}%` : undefined}
                      tone="good"
                    />
                    <StatTile
                      label="Cancels"
                      value={decomp.cancel_count.toLocaleString()}
                      hint={decomp.trade_count_pct != null ? `${fmtNum(100 - decomp.trade_count_pct, 1)}%` : undefined}
                      tone={decomp.trade_count_pct != null && decomp.trade_count_pct < 30 ? 'warn' : 'neutral'}
                    />
                  </div>
                  <div className="mt-1 text-[10px] text-muted-foreground">
                    high cancel rate ⇒ spoofy book ⇒ lower effective displayed depth (auto-applied to
                    ensemble)
                  </div>
                </div>
              ) : null}

              {/* EMPIRICAL CONSTANTS */}
              {constants ? (
                <div className="rounded-md border border-border/50 bg-card/40 p-3">
                  <div className="mb-1 flex items-center gap-1.5 text-xs font-medium">
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-300" />
                    Empirical constants
                    <Badge
                      className={cn(
                        'ml-auto text-[9px]',
                        constants.measured
                          ? 'bg-emerald-500/10 text-emerald-300'
                          : 'bg-amber-500/10 text-amber-300',
                      )}
                    >
                      {constants.measured ? 'measured' : 'defaults'}
                    </Badge>
                  </div>
                  <div className="space-y-0.5 text-[11px]">
                    {Object.entries(constants.values).map(([k, v]) => (
                      <div key={k} className="grid grid-cols-[1fr,60px] items-center gap-1">
                        <span className="truncate text-muted-foreground">{k.replaceAll('_', ' ')}</span>
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
