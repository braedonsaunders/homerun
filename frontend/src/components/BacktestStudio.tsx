/**
 * BacktestStudio — institutional-grade backtest workbench.
 *
 * Single-path successor to the legacy StrategyBacktestFlyout (which
 * stays mounted as a fallback for now).  Pulls every advanced datum
 * the platform has — Cox PH fill model state, ensemble PnL bands,
 * empirical constants, latency distribution, trade-vs-cancel
 * decomposition, counterfactual replay diagnostics — into one
 * coherent multi-pane workbench.
 *
 * Layout: 3-pane split — left rail (run controls + history),
 * center (results + equity curve + trades), right rail
 * (microstructure / fill model state).  Fonts and color tokens
 * follow the FillModelPanel conventions so the two surfaces feel
 * like one product.
 */
import { useMemo, useState } from 'react'
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
  type UnifiedBacktestResult,
  getBacktestRun,
  listBacktestRuns,
  runUnifiedBacktest,
} from '../services/apiBacktest'
import {
  getActiveFillModel,
  getDecompositionSummary,
  getEmpiricalConstants,
  getLatencyDistribution,
} from '../services/apiFillModel'

interface BacktestStudioProps {
  initialSourceCode?: string
  initialSlug?: string
  initialConfig?: Record<string, unknown>
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
            <div className="flex items-center justify-between">
              <span className="font-mono text-muted-foreground">
                {run.run_id.slice(0, 6)}
              </span>
              <span
                className={cn(
                  'tabular-nums',
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
}: BacktestStudioProps) {
  const queryClient = useQueryClient()

  // Run controls.
  const [sourceCode, setSourceCode] = useState<string>(initialSourceCode || '')
  const [slug] = useState<string>(initialSlug || '_backtest_unified')
  const [initialCapital, setInitialCapital] = useState<string>('1000')
  const [submitP50, setSubmitP50] = useState<string>('')
  const [submitP95, setSubmitP95] = useState<string>('')
  const [seed, setSeed] = useState<string>('')

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
    runMutation.mutate({
      source_code: sourceCode,
      slug,
      config: initialConfig,
      initial_capital_usd: parseFloat(initialCapital) || 1000,
      submit_p50_ms: submitP50 ? parseFloat(submitP50) : undefined,
      submit_p95_ms: submitP95 ? parseFloat(submitP95) : undefined,
      seed: seed ? parseInt(seed, 10) : undefined,
      counterfactual_sample_size: 8,
      ensemble_sample_size: 8,
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
            <div className="flex items-center justify-between">
              <span className="text-[11px] uppercase tracking-wide text-muted-foreground">
                Strategy source
              </span>
              <span className="text-[10px] text-muted-foreground">
                {sourceCode.length.toLocaleString()} chars
              </span>
            </div>
            <textarea
              value={sourceCode}
              onChange={(e) => setSourceCode(e.target.value)}
              placeholder="# Paste your strategy class source here…"
              className="h-32 w-full rounded-sm border border-border/40 bg-background/60 p-2 font-mono text-[10px] leading-tight text-foreground"
              spellCheck={false}
            />

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
          {!activeRun ? (
            <div className="flex h-full min-h-[400px] flex-col items-center justify-center gap-2 px-6 py-12 text-center">
              <Flame className="h-8 w-8 text-amber-300/50" />
              <div className="text-sm font-medium">No run loaded</div>
              <div className="max-w-[420px] text-xs text-muted-foreground">
                Paste a strategy in the left panel and click <strong>Run backtest</strong>. Results
                will include execution-realistic L2 replay, ensemble PnL bands at p10/p50/p90,
                counterfactual queue replay against the live trade tape, and the active Cox PH fill
                model snapshot.
              </div>
            </div>
          ) : (
            <div className="space-y-4 p-3">
              {/* HEADLINE KPIS */}
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

              {/* ENSEMBLE BANDS + COUNTERFACTUALS */}
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
            </div>
          )}
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
                    Measured latency
                  </div>
                  <div className="grid grid-cols-3 gap-1.5">
                    <StatTile label="p50" value={`${Math.round(latency.p50_ms)}ms`} />
                    <StatTile label="p95" value={`${Math.round(latency.p95_ms)}ms`} tone={latency.p95_ms > 800 ? 'warn' : 'neutral'} />
                    <StatTile label="p99" value={`${Math.round(latency.p99_ms)}ms`} tone={latency.p99_ms > 1500 ? 'bad' : 'neutral'} />
                  </div>
                  <div className="mt-1 text-[10px] text-muted-foreground">
                    {latency.sample_count.toLocaleString()} samples · ensemble uses pessimistic={Math.round(latency.pessimistic_ms)}ms / realistic={Math.round(latency.realistic_ms)}ms / optimistic={Math.round(latency.optimistic_ms)}ms
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
