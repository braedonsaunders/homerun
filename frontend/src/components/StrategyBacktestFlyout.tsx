import { useMemo, useState, type ComponentType } from 'react'
import { useMutation } from '@tanstack/react-query'
import {
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clock,
  Database,
  FlaskConical,
  Gauge,
  Loader2,
  Play,
  ShieldCheck,
  Target,
  TrendingUp,
  XCircle,
} from 'lucide-react'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { ScrollArea } from './ui/scroll-area'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from './ui/sheet'
import { Switch } from './ui/switch'
import { cn } from '../lib/utils'
import { normalizeUtcTimestampsInPlace } from '../lib/timestamps'
import axios from 'axios'

const api = axios.create({ baseURL: '/api', timeout: 120000 })

type BacktestMode = 'detect' | 'evaluate' | 'exit'
type Tone = 'good' | 'warn' | 'bad' | 'neutral'

interface QualityFilter {
  filter_name: string
  passed: boolean
  reason: string
  threshold: number | string | null
  actual_value: number | string | null
}

interface QualityReportData {
  opportunity_id: string
  passed: boolean
  rejection_reasons: string[]
  filters: QualityFilter[]
}

interface DecisionCheck {
  check_key?: string
  check_label?: string
  passed?: boolean
  score?: number | null
}

interface BacktestResult {
  success: boolean
  strategy_slug: string
  strategy_name: string
  class_name: string
  load_time_ms: number
  data_fetch_time_ms: number
  total_time_ms: number
  validation_errors: string[]
  validation_warnings: string[]
  runtime_error: string | null
  runtime_traceback: string | null

  num_events?: number
  num_markets?: number
  num_prices?: number
  data_source?: string
  replay_mode?: string
  replay_steps?: number
  replay_markets?: number
  replay_window_hours?: number
  replay_timeframe?: string
  opportunities?: Array<Record<string, any>>
  num_opportunities?: number
  quality_reports?: QualityReportData[]
  detect_time_ms?: number

  num_signals?: number
  decisions?: Array<Record<string, any>>
  selected?: number
  skipped?: number
  blocked?: number
  evaluate_time_ms?: number

  num_positions?: number
  exit_decisions?: Array<Record<string, any>>
  would_close?: number
  would_reduce?: number
  would_hold?: number
  errors?: number
  exit_time_ms?: number
}

interface OutcomeRow {
  id: string
  primary: string
  secondary?: string
  action: string
  tone: Tone
  headlineMetric: string
  subMetrics: string[]
  reason?: string
  detail: Record<string, any>
}

interface SummaryCard {
  label: string
  value: string
  subtitle: string
  tone: Tone
  icon: ComponentType<{ className?: string }>
}

interface BacktestRunSettings {
  detect: {
    useOhlcReplay: boolean
    replayLookbackHours: string
    replayTimeframe: string
    replayMaxMarkets: string
    replayMaxSteps: string
    maxOpportunities: string
  }
  evaluate: {
    maxSignals: string
  }
  exit: {
    maxPositions: string
  }
}

function toNumber(value: unknown): number {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : 0
}

function fmt(n: number, decimals = 0): string {
  return toNumber(n).toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })
}

function fmtMoney(n: number): string {
  return `$${fmt(n, 2)}`
}

function fmtSignedPct(n: number): string {
  const value = toNumber(n)
  return `${value > 0 ? '+' : ''}${fmt(value, 2)}%`
}

function fmtPrice(n: number): string {
  return `$${fmt(n, 3)}`
}

function fmtMinutes(n: number): string {
  const value = toNumber(n)
  if (value < 60) {
    return `${fmt(value, 0)}m`
  }
  return `${fmt(value / 60, 1)}h`
}

function fmtMs(ms: number): string {
  const value = toNumber(ms)
  if (value < 1000) {
    return `${Math.round(value)}ms`
  }
  return `${(value / 1000).toFixed(1)}s`
}

function percent(part: number, total: number): number {
  if (total <= 0) {
    return 0
  }
  return (part / total) * 100
}

function mean(values: number[]): number {
  if (values.length === 0) {
    return 0
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

function actionTone(action: string): Tone {
  const value = action.toLowerCase()
  if (value === 'selected' || value === 'close' || value === 'quality-pass' || value === 'pass') {
    return 'good'
  }
  if (value === 'blocked' || value === 'error' || value === 'quality-fail' || value === 'fail') {
    return 'bad'
  }
  if (value === 'reduce') {
    return 'warn'
  }
  return 'neutral'
}

function toneClasses(tone: Tone): string {
  if (tone === 'good') {
    return 'border-emerald-500/30 bg-emerald-500/5 text-emerald-300'
  }
  if (tone === 'warn') {
    return 'border-amber-500/30 bg-amber-500/5 text-amber-300'
  }
  if (tone === 'bad') {
    return 'border-red-500/30 bg-red-500/5 text-red-300'
  }
  return 'border-border/40 bg-card/30 text-foreground'
}

function phaseLabel(mode: BacktestMode): string {
  if (mode === 'detect') {
    return 'Detection'
  }
  if (mode === 'evaluate') {
    return 'Evaluation'
  }
  return 'Exit Logic'
}

function phaseDurationMs(mode: BacktestMode, result: BacktestResult): number {
  if (mode === 'detect') {
    return toNumber(result.detect_time_ms)
  }
  if (mode === 'evaluate') {
    return toNumber(result.evaluate_time_ms)
  }
  return toNumber(result.exit_time_ms)
}

function runSampleSize(mode: BacktestMode, result: BacktestResult): number {
  if (mode === 'detect') {
    return toNumber(result.num_opportunities)
  }
  if (mode === 'evaluate') {
    return toNumber(result.num_signals)
  }
  return toNumber(result.num_positions)
}

function completenessScore(mode: BacktestMode, result: BacktestResult): number {
  if (mode === 'detect') {
    const opportunities = result.opportunities || []
    if (opportunities.length === 0) {
      return 0
    }
    let present = 0
    let total = 0
    opportunities.forEach((opp) => {
      total += 3
      if (Number.isFinite(Number(opp.roi_percent ?? opp.roi))) {
        present += 1
      }
      if (Number.isFinite(Number(opp.total_cost))) {
        present += 1
      }
      if (Number.isFinite(Number(opp.risk_score))) {
        present += 1
      }
    })
    return Math.round(percent(present, total))
  }

  if (mode === 'evaluate') {
    const decisions = result.decisions || []
    if (decisions.length === 0) {
      return 0
    }
    let present = 0
    let total = 0
    decisions.forEach((decision) => {
      total += 3
      if (String(decision.decision || '').trim()) {
        present += 1
      }
      if (String(decision.reason || decision.strategy_reason || '').trim()) {
        present += 1
      }
      if (Array.isArray(decision.checks)) {
        present += 1
      }
    })
    return Math.round(percent(present, total))
  }

  const exitDecisions = result.exit_decisions || []
  if (exitDecisions.length === 0) {
    return 0
  }
  let present = 0
  let total = 0
  exitDecisions.forEach((decision) => {
    total += 4
    if (String(decision.action || '').trim()) {
      present += 1
    }
    if (Number.isFinite(Number(decision.entry_price))) {
      present += 1
    }
    if (Number.isFinite(Number(decision.current_price))) {
      present += 1
    }
    if (Number.isFinite(Number(decision.pnl_pct))) {
      present += 1
    }
  })
  return Math.round(percent(present, total))
}

function buildOutcomeRows(mode: BacktestMode, result: BacktestResult | null): OutcomeRow[] {
  if (!result) {
    return []
  }

  if (mode === 'detect') {
    const opportunities = result.opportunities || []
    const qualityReports = result.quality_reports || []
    const qualityById = new Map<string, QualityReportData>()
    qualityReports.forEach((report) => {
      const key = String(report.opportunity_id || '').trim()
      if (key) {
        qualityById.set(key, report)
      }
    })

    return opportunities.map((opp, index) => {
      const oppId = String(opp.id || opp.stable_id || `opp-${index}`)
      const quality = qualityById.get(oppId) || qualityReports[index]
      const roi = toNumber(opp.roi_percent ?? opp.roi)
      const cost = toNumber(opp.total_cost)
      const risk = toNumber(opp.risk_score)
      const markets = Array.isArray(opp.markets) ? opp.markets.length : 0
      const legs = Array.isArray(opp.positions_to_take) ? opp.positions_to_take.length : 0
      const qualityAction = quality ? (quality.passed ? 'quality-pass' : 'quality-fail') : 'candidate'

      return {
        id: oppId,
        primary: String(opp.title || `Opportunity ${index + 1}`),
        secondary: String(opp.strategy || result.strategy_name || '').trim() || undefined,
        action: qualityAction,
        tone: actionTone(qualityAction),
        headlineMetric: `${fmtSignedPct(roi)} ROI`,
        subMetrics: [
          `Cost ${fmtMoney(cost)}`,
          `Risk ${fmt(risk, 2)}`,
          `Markets ${fmt(markets)}`,
          `Legs ${fmt(legs)}`,
        ],
        reason:
          quality && !quality.passed
            ? (quality.rejection_reasons || []).join(' | ')
            : String(opp.description || ''),
        detail: {
          opportunity: opp,
          quality_report: quality || null,
        },
      }
    })
  }

  if (mode === 'evaluate') {
    const decisions = result.decisions || []
    return decisions.map((decision, index) => {
      const checks = Array.isArray(decision.checks) ? (decision.checks as DecisionCheck[]) : []
      const passedChecks = checks.filter((check) => Boolean(check.passed)).length
      const action = String(decision.decision || 'skipped').toLowerCase()
      return {
        id: String(decision.signal_id || `signal-${index}`),
        primary: String(decision.source || decision.strategy_type || `Signal ${index + 1}`),
        secondary: String(decision.signal_id || '').trim() || undefined,
        action,
        tone: actionTone(action),
        headlineMetric: `Size ${fmtMoney(toNumber(decision.size_usd))}`,
        subMetrics: [
          `Checks ${passedChecks}/${checks.length}`,
          `Strategy ${String(decision.strategy_decision || 'n/a')}`,
        ],
        reason: String(decision.reason || decision.strategy_reason || ''),
        detail: {
          decision,
        },
      }
    })
  }

  const exitDecisions = result.exit_decisions || []
  return exitDecisions.map((decision, index) => {
    const action = String(decision.action || 'hold').toLowerCase()
    const reduceFraction = toNumber(decision.reduce_fraction)
    const subMetrics = [
      `Entry ${fmtPrice(toNumber(decision.entry_price))}`,
      `Now ${fmtPrice(toNumber(decision.current_price))}`,
      `Age ${fmtMinutes(toNumber(decision.age_minutes))}`,
      `Notional ${fmtMoney(toNumber(decision.notional_usd))}`,
    ]
    if (action === 'reduce' && reduceFraction > 0) {
      subMetrics.push(`Reduce ${fmt(reduceFraction * 100, 0)}%`)
    }

    return {
      id: String(decision.position_id || `position-${index}`),
      primary: String(decision.market_question || decision.market_id || `Position ${index + 1}`),
      secondary: String(decision.position_id || '').trim() || undefined,
      action,
      tone: actionTone(action),
      headlineMetric: `${fmtSignedPct(toNumber(decision.pnl_pct))} PnL`,
      subMetrics,
      reason: String(decision.reason || ''),
      detail: {
        decision,
      },
    }
  })
}

function buildSummaryCards(mode: BacktestMode, result: BacktestResult, dataCompleteness: number): SummaryCard[] {
  if (mode === 'detect') {
    const opportunities = result.opportunities || []
    const qualityReports = result.quality_reports || []
    const passCount = qualityReports.filter((report) => report.passed).length
    const roiValues = opportunities.map((opp) => toNumber(opp.roi_percent ?? opp.roi))
    const riskValues = opportunities.map((opp) => toNumber(opp.risk_score))

    return [
      {
        label: 'Opportunities',
        value: fmt(toNumber(result.num_opportunities)),
        subtitle: 'Detected opportunities',
        tone: toNumber(result.num_opportunities) > 0 ? 'good' : 'neutral',
        icon: Target,
      },
      {
        label: 'Quality Pass',
        value: qualityReports.length > 0 ? `${fmt(percent(passCount, qualityReports.length), 1)}%` : 'n/a',
        subtitle: `${fmt(passCount)}/${fmt(qualityReports.length)} passed`,
        tone: qualityReports.length > 0 && passCount > 0 ? 'good' : 'warn',
        icon: ShieldCheck,
      },
      {
        label: 'Average ROI',
        value: `${fmt(mean(roiValues), 2)}%`,
        subtitle: 'Across detected opportunities',
        tone: mean(roiValues) > 0 ? 'good' : 'neutral',
        icon: TrendingUp,
      },
      {
        label: 'Average Risk',
        value: fmt(mean(riskValues), 2),
        subtitle: 'Normalized risk score',
        tone: mean(riskValues) > 0.7 ? 'bad' : mean(riskValues) > 0.45 ? 'warn' : 'good',
        icon: Gauge,
      },
      {
        label: 'Replay Coverage',
        value: `${fmt(toNumber(result.replay_steps))} steps`,
        subtitle: `${fmt(toNumber(result.replay_markets))} markets`,
        tone: toNumber(result.replay_steps) > 0 ? 'good' : 'neutral',
        icon: BarChart3,
      },
      {
        label: 'Completeness',
        value: `${fmt(dataCompleteness)}%`,
        subtitle: 'Core fields populated',
        tone: dataCompleteness >= 90 ? 'good' : dataCompleteness >= 70 ? 'warn' : 'bad',
        icon: Database,
      },
    ]
  }

  if (mode === 'evaluate') {
    const decisions = result.decisions || []
    const checks = decisions.flatMap((decision) =>
      Array.isArray(decision.checks) ? (decision.checks as DecisionCheck[]) : []
    )
    const passedChecks = checks.filter((check) => Boolean(check.passed)).length
    const sizeValues = decisions
      .map((decision) => toNumber(decision.size_usd))
      .filter((value) => value > 0)
    const decisionErrors = decisions.filter(
      (decision) => String(decision.decision || '').toLowerCase() === 'error'
    ).length

    return [
      {
        label: 'Signals Tested',
        value: fmt(toNumber(result.num_signals)),
        subtitle: 'Recent emissions evaluated',
        tone: toNumber(result.num_signals) > 0 ? 'good' : 'neutral',
        icon: Target,
      },
      {
        label: 'Selected Rate',
        value: `${fmt(percent(toNumber(result.selected), toNumber(result.num_signals)), 1)}%`,
        subtitle: `${fmt(toNumber(result.selected))} selected`,
        tone: toNumber(result.selected) > 0 ? 'good' : 'neutral',
        icon: CheckCircle2,
      },
      {
        label: 'Blocked Rate',
        value: `${fmt(percent(toNumber(result.blocked), toNumber(result.num_signals)), 1)}%`,
        subtitle: `${fmt(toNumber(result.blocked))} blocked`,
        tone: toNumber(result.blocked) > 0 ? 'warn' : 'good',
        icon: AlertTriangle,
      },
      {
        label: 'Avg Size',
        value: sizeValues.length > 0 ? fmtMoney(mean(sizeValues)) : 'n/a',
        subtitle: 'Suggested position size',
        tone: sizeValues.length > 0 ? 'good' : 'neutral',
        icon: Gauge,
      },
      {
        label: 'Check Pass Rate',
        value: checks.length > 0 ? `${fmt(percent(passedChecks, checks.length), 1)}%` : 'n/a',
        subtitle: `${fmt(passedChecks)}/${fmt(checks.length)} checks passed`,
        tone: checks.length > 0 && passedChecks > 0 ? 'good' : 'warn',
        icon: ShieldCheck,
      },
      {
        label: 'Completeness',
        value: `${fmt(dataCompleteness)}%`,
        subtitle: `${fmt(decisionErrors)} decision errors`,
        tone: dataCompleteness >= 90 ? 'good' : dataCompleteness >= 70 ? 'warn' : 'bad',
        icon: Database,
      },
    ]
  }

  const exits = result.exit_decisions || []
  const pnlValues = exits
    .filter((decision) => String(decision.action || '').toLowerCase() !== 'error')
    .map((decision) => toNumber(decision.pnl_pct))
  const totalNotional = exits.reduce((sum, decision) => sum + toNumber(decision.notional_usd), 0)
  const actionable = toNumber(result.would_close) + toNumber(result.would_reduce)

  return [
    {
      label: 'Positions Tested',
      value: fmt(toNumber(result.num_positions)),
      subtitle: 'Open inventory sampled',
      tone: toNumber(result.num_positions) > 0 ? 'good' : 'neutral',
      icon: Target,
    },
    {
      label: 'Exit Rate',
      value: `${fmt(percent(actionable, toNumber(result.num_positions)), 1)}%`,
      subtitle: `${fmt(actionable)} actionable exits`,
      tone: actionable > 0 ? 'good' : 'neutral',
      icon: TrendingUp,
    },
    {
      label: 'Close / Reduce',
      value: `${fmt(toNumber(result.would_close))}/${fmt(toNumber(result.would_reduce))}`,
      subtitle: 'Action split',
      tone: toNumber(result.would_close) > 0 || toNumber(result.would_reduce) > 0 ? 'good' : 'neutral',
      icon: CheckCircle2,
    },
    {
      label: 'Hold',
      value: fmt(toNumber(result.would_hold)),
      subtitle: 'Held positions',
      tone: toNumber(result.would_hold) > 0 ? 'warn' : 'good',
      icon: Gauge,
    },
    {
      label: 'Average PnL',
      value: `${fmt(mean(pnlValues), 2)}%`,
      subtitle: 'Across non-error exits',
      tone: mean(pnlValues) > 0 ? 'good' : 'neutral',
      icon: BarChart3,
    },
    {
      label: 'Completeness',
      value: `${fmt(dataCompleteness)}%`,
      subtitle: `${fmtMoney(totalNotional)} notional`,
      tone: dataCompleteness >= 90 ? 'good' : dataCompleteness >= 70 ? 'warn' : 'bad',
      icon: Database,
    },
  ]
}

function SummaryMetricCard({ card }: { card: SummaryCard }) {
  return (
    <div className={cn('rounded-md border px-2.5 py-2', toneClasses(card.tone))}>
      <div className="flex items-center gap-1.5 mb-1">
        <card.icon className="w-3 h-3 opacity-70" />
        <span className="text-[9px] uppercase tracking-wider opacity-75">{card.label}</span>
      </div>
      <div className="text-sm font-mono font-semibold">{card.value}</div>
      <p className="text-[9px] opacity-70 mt-0.5">{card.subtitle}</p>
    </div>
  )
}

function OutcomeCard({
  row,
  expanded,
  onToggle,
}: {
  row: OutcomeRow
  expanded: boolean
  onToggle: () => void
}) {
  return (
    <div className="border border-border/40 rounded-lg overflow-hidden">
      <button
        className="w-full text-left px-3 py-2 hover:bg-card/40 transition-colors"
        onClick={onToggle}
      >
        <div className="flex items-start gap-2">
          {expanded ? (
            <ChevronDown className="w-3 h-3 text-muted-foreground shrink-0 mt-0.5" />
          ) : (
            <ChevronRight className="w-3 h-3 text-muted-foreground shrink-0 mt-0.5" />
          )}
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-medium truncate">{row.primary}</span>
              <Badge
                variant="outline"
                className={cn(
                  'text-[8px] h-3.5 uppercase',
                  row.tone === 'good'
                    ? 'border-emerald-500/30 text-emerald-400'
                    : row.tone === 'warn'
                      ? 'border-amber-500/30 text-amber-400'
                      : row.tone === 'bad'
                        ? 'border-red-500/30 text-red-400'
                        : 'border-border/40 text-muted-foreground'
                )}
              >
                {row.action}
              </Badge>
              <span className="ml-auto text-[10px] font-mono font-medium">{row.headlineMetric}</span>
            </div>
            {row.secondary && (
              <p className="text-[10px] text-muted-foreground mt-0.5 truncate">{row.secondary}</p>
            )}
            <div className="mt-1 flex flex-wrap gap-1.5">
              {row.subMetrics.map((metric, index) => (
                <span
                  key={`${row.id}-${index}`}
                  className="text-[9px] px-1.5 py-0.5 rounded bg-muted/40 text-muted-foreground font-mono"
                >
                  {metric}
                </span>
              ))}
            </div>
            {row.reason && <p className="text-[10px] text-muted-foreground mt-1.5 leading-relaxed">{row.reason}</p>}
          </div>
        </div>
      </button>
      {expanded && (
        <div className="px-3 pb-3 border-t border-border/20">
          <pre className="text-[10px] font-mono bg-black/20 rounded p-2 mt-2 overflow-x-auto whitespace-pre">
            {JSON.stringify(row.detail, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}

const MODE_ENDPOINTS: Record<BacktestMode, string> = {
  detect: '/validation/code-backtest',
  evaluate: '/validation/code-backtest/evaluate',
  exit: '/validation/code-backtest/exit',
}

const MODE_LABELS: Record<BacktestMode, { label: string; desc: string; running: string }> = {
  detect: {
    label: 'Detect',
    desc: 'Find opportunities in current and replayed market states',
    running: 'Running detection...',
  },
  evaluate: {
    label: 'Evaluate',
    desc: 'Score recent signals through strategy and platform gates',
    running: 'Evaluating signals...',
  },
  exit: {
    label: 'Exit',
    desc: 'Run should_exit() against live open position inventory',
    running: 'Testing exit logic...',
  },
}

const DETECT_REPLAY_OPTIONS = {
  use_ohlc_replay: true,
  replay_lookback_hours: 24,
  replay_timeframe: '30m',
  replay_max_markets: 80,
  replay_max_steps: 72,
  max_opportunities: 100,
}

const BACKTEST_MODES: BacktestMode[] = ['detect', 'evaluate', 'exit']
const REPLAY_TIMEFRAME_OPTIONS = ['5m', '15m', '30m', '1h', '4h']

function createDefaultBacktestSettings(): BacktestRunSettings {
  return {
    detect: {
      useOhlcReplay: DETECT_REPLAY_OPTIONS.use_ohlc_replay,
      replayLookbackHours: String(DETECT_REPLAY_OPTIONS.replay_lookback_hours),
      replayTimeframe: DETECT_REPLAY_OPTIONS.replay_timeframe,
      replayMaxMarkets: String(DETECT_REPLAY_OPTIONS.replay_max_markets),
      replayMaxSteps: String(DETECT_REPLAY_OPTIONS.replay_max_steps),
      maxOpportunities: String(DETECT_REPLAY_OPTIONS.max_opportunities),
    },
    evaluate: {
      maxSignals: '50',
    },
    exit: {
      maxPositions: '50',
    },
  }
}

function normalizeIntegerSetting(value: string, fallback: number, min: number, max: number): number {
  const parsed = Number.parseInt(String(value || '').trim(), 10)
  if (!Number.isFinite(parsed)) {
    return fallback
  }
  return Math.min(max, Math.max(min, parsed))
}

function buildBacktestPayload(
  mode: BacktestMode,
  sourceCode: string,
  slug: string,
  config: Record<string, any> | undefined,
  settings: BacktestRunSettings
): Record<string, any> {
  return {
    source_code: sourceCode,
    slug,
    config: config || {},
    use_ohlc_replay: mode === 'detect' ? settings.detect.useOhlcReplay : false,
    replay_lookback_hours: normalizeIntegerSetting(settings.detect.replayLookbackHours, 24, 1, 720),
    replay_timeframe: settings.detect.replayTimeframe || DETECT_REPLAY_OPTIONS.replay_timeframe,
    replay_max_markets: normalizeIntegerSetting(settings.detect.replayMaxMarkets, 80, 1, 300),
    replay_max_steps: normalizeIntegerSetting(settings.detect.replayMaxSteps, 72, 1, 500),
    max_opportunities: normalizeIntegerSetting(settings.detect.maxOpportunities, 100, 1, 500),
    max_signals: normalizeIntegerSetting(settings.evaluate.maxSignals, 50, 1, 500),
    max_positions: normalizeIntegerSetting(settings.exit.maxPositions, 50, 1, 500),
  }
}

interface IntegratedBacktestScore {
  overall: number
  readiness: number
  passedPhases: number
  phaseScores: Record<BacktestMode, number>
  phaseCompleteness: Record<BacktestMode, number>
  phaseSamples: Record<BacktestMode, number>
  phasePassed: Record<BacktestMode, boolean>
}

function scoreTone(score: number): Tone {
  if (score >= 85) {
    return 'good'
  }
  if (score >= 65) {
    return 'warn'
  }
  return 'bad'
}

function createRequestFailureResult(mode: BacktestMode, slug: string, error: unknown): BacktestResult {
  const detail =
    axios.isAxiosError(error)
      ? String(error.response?.data?.detail || error.message || 'Request failed')
      : String(error)
  return {
    success: false,
    strategy_slug: slug,
    strategy_name: slug,
    class_name: '',
    load_time_ms: 0,
    data_fetch_time_ms: 0,
    total_time_ms: 0,
    validation_errors: [],
    validation_warnings: [],
    runtime_error: `${phaseLabel(mode)} request failed: ${detail}`,
    runtime_traceback: null,
  }
}

function buildIntegratedScore(results: Partial<Record<BacktestMode, BacktestResult>>): IntegratedBacktestScore {
  const phaseScores: Record<BacktestMode, number> = {
    detect: 0,
    evaluate: 0,
    exit: 0,
  }
  const phaseCompleteness: Record<BacktestMode, number> = {
    detect: 0,
    evaluate: 0,
    exit: 0,
  }
  const phaseSamples: Record<BacktestMode, number> = {
    detect: 0,
    evaluate: 0,
    exit: 0,
  }
  const phasePassed: Record<BacktestMode, boolean> = {
    detect: false,
    evaluate: false,
    exit: false,
  }

  BACKTEST_MODES.forEach((mode) => {
    const result = results[mode]
    if (!result) {
      return
    }

    const completeness = completenessScore(mode, result)
    const samples = runSampleSize(mode, result)
    const runHealthy =
      result.success &&
      !result.runtime_error &&
      Array.isArray(result.validation_errors) &&
      result.validation_errors.length === 0
    const phaseScore = Math.round(completeness * 0.6 + (samples > 0 ? 100 : 0) * 0.2 + (runHealthy ? 100 : 0) * 0.2)

    phaseCompleteness[mode] = completeness
    phaseSamples[mode] = samples
    phaseScores[mode] = phaseScore
    phasePassed[mode] = runHealthy && samples > 0 && completeness >= 70
  })

  const passedPhases = BACKTEST_MODES.filter((mode) => phasePassed[mode]).length
  const readiness = Math.round(percent(passedPhases, BACKTEST_MODES.length))
  const averagePhaseScore = mean(BACKTEST_MODES.map((mode) => phaseScores[mode]))
  const overall = Math.round(averagePhaseScore * (readiness / 100))

  return {
    overall,
    readiness,
    passedPhases,
    phaseScores,
    phaseCompleteness,
    phaseSamples,
    phasePassed,
  }
}

export default function StrategyBacktestFlyout({
  open,
  onOpenChange,
  sourceCode,
  slug,
  config,
  variant,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  sourceCode: string
  slug: string
  config?: Record<string, any>
  variant: 'opportunity' | 'trader'
}) {
  const [phaseResults, setPhaseResults] = useState<Partial<Record<BacktestMode, BacktestResult>>>({})
  const [expandedRowId, setExpandedRowId] = useState<string | null>(null)
  const [runningPhase, setRunningPhase] = useState<BacktestMode | null>(null)
  const [runSettings, setRunSettings] = useState<BacktestRunSettings>(() => createDefaultBacktestSettings())

  const backtestMutation = useMutation({
    mutationFn: async () => {
      const suite: Record<BacktestMode, BacktestResult> = {
        detect: createRequestFailureResult('detect', slug, 'Not run'),
        evaluate: createRequestFailureResult('evaluate', slug, 'Not run'),
        exit: createRequestFailureResult('exit', slug, 'Not run'),
      }

      for (const candidate of BACKTEST_MODES) {
        setRunningPhase(candidate)
        const payload = buildBacktestPayload(candidate, sourceCode, slug, config, runSettings)
        try {
          const { data } = await api.post(MODE_ENDPOINTS[candidate], payload)
          normalizeUtcTimestampsInPlace(data)
          suite[candidate] = data as BacktestResult
        } catch (error) {
          suite[candidate] = createRequestFailureResult(candidate, slug, error)
        }
      }

      return suite
    },
    onMutate: () => {
      setExpandedRowId(null)
      setPhaseResults({})
      setRunningPhase('detect')
    },
    onSuccess: (data) => {
      setExpandedRowId(null)
      setPhaseResults(data)
      setRunningPhase(null)
    },
    onError: () => {
      setExpandedRowId(null)
      setPhaseResults({})
      setRunningPhase(null)
    },
  })

  const integratedScore = useMemo(() => buildIntegratedScore(phaseResults), [phaseResults])
  const overallTone = scoreTone(integratedScore.overall)
  const hasSuiteResults = useMemo(
    () => BACKTEST_MODES.some((candidate) => Boolean(phaseResults[candidate])),
    [phaseResults]
  )
  const totalRuntimeMs = useMemo(
    () => BACKTEST_MODES.reduce((sum, candidate) => sum + toNumber(phaseResults[candidate]?.total_time_ms), 0),
    [phaseResults]
  )

  const phaseViews = useMemo(
    () =>
      BACKTEST_MODES.map((candidate) => {
        const phaseResult = phaseResults[candidate] || null
        const rows = buildOutcomeRows(candidate, phaseResult)
        const actionCounts = (() => {
          const counts = new Map<string, number>()
          rows.forEach((row) => {
            counts.set(row.action, (counts.get(row.action) || 0) + 1)
          })
          return Array.from(counts.entries()).sort((a, b) => b[1] - a[1])
        })()
        const dataCompleteness = phaseResult ? completenessScore(candidate, phaseResult) : 0
        const summaryCards = phaseResult ? buildSummaryCards(candidate, phaseResult, dataCompleteness) : []
        const phaseMs = phaseResult ? phaseDurationMs(candidate, phaseResult) : 0
        const sampleSize = phaseResult ? runSampleSize(candidate, phaseResult) : 0
        const hasWarnings = Boolean(phaseResult && phaseResult.validation_warnings.length > 0)
        return {
          mode: candidate,
          result: phaseResult,
          rows,
          actionCounts,
          dataCompleteness,
          summaryCards,
          phaseMs,
          sampleSize,
          hasWarnings,
        }
      }),
    [phaseResults]
  )

  const runningLabel = runningPhase ? MODE_LABELS[runningPhase].running : 'Running full suite...'

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-[98vw] sm:max-w-4xl xl:max-w-6xl p-0">
        <div className="h-full min-h-0 flex flex-col">
          <div className="border-b border-border px-4 py-3 space-y-3">
            <SheetHeader className="space-y-1 text-left">
              <SheetTitle className="text-base flex items-center gap-2">
                <FlaskConical className="w-4 h-4" />
                Strategy Backtest
                <Badge variant="outline" className="text-[9px] h-4 font-normal">
                  Integrated
                </Badge>
              </SheetTitle>
              <SheetDescription>
                Runs detect, evaluate, and exit as one required suite, shown as stacked sections below.
              </SheetDescription>
            </SheetHeader>

            <div className="rounded-lg border border-border/40 bg-card/30 p-3 space-y-3">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-[11px] font-medium">Run Settings</p>
                  <p className="text-[10px] text-muted-foreground">
                    Tune replay depth and per-phase sample caps before launching the suite.
                  </p>
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="h-7 px-2 text-[10px]"
                  onClick={() => setRunSettings(createDefaultBacktestSettings())}
                  disabled={backtestMutation.isPending}
                >
                  Reset defaults
                </Button>
              </div>

              <div className="grid gap-3 xl:grid-cols-3">
                <div className="rounded-md border border-border/40 bg-background/40 p-3 space-y-3">
                  <div className="flex items-center gap-2">
                    <BarChart3 className="w-3.5 h-3.5 text-cyan-400" />
                    <div>
                      <p className="text-[11px] font-medium">Detect</p>
                      <p className="text-[10px] text-muted-foreground">Replay window and collection limits.</p>
                    </div>
                  </div>

                  <div className="rounded-md border border-border/30 bg-background/70 px-2.5 py-2">
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <Label htmlFor="backtest-use-ohlc-replay" className="text-[11px]">
                          Use OHLC replay
                        </Label>
                        <p className="mt-1 text-[10px] text-muted-foreground">
                          Replays historical snapshots only when live detect returns nothing.
                        </p>
                      </div>
                      <Switch
                        id="backtest-use-ohlc-replay"
                        checked={runSettings.detect.useOhlcReplay}
                        onCheckedChange={(checked) =>
                          setRunSettings((current) => ({
                            ...current,
                            detect: {
                              ...current.detect,
                              useOhlcReplay: checked,
                            },
                          }))
                        }
                        disabled={backtestMutation.isPending}
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-2">
                    <div className="space-y-1">
                      <Label htmlFor="backtest-replay-lookback" className="text-[10px] uppercase tracking-wide text-muted-foreground">
                        Replay Hours
                      </Label>
                      <Input
                        id="backtest-replay-lookback"
                        type="number"
                        min={1}
                        max={720}
                        step={1}
                        className="h-8 text-xs font-mono"
                        value={runSettings.detect.replayLookbackHours}
                        onChange={(event) =>
                          setRunSettings((current) => ({
                            ...current,
                            detect: {
                              ...current.detect,
                              replayLookbackHours: event.target.value,
                            },
                          }))
                        }
                        disabled={backtestMutation.isPending || !runSettings.detect.useOhlcReplay}
                      />
                    </div>
                    <div className="space-y-1">
                      <Label className="text-[10px] uppercase tracking-wide text-muted-foreground">Replay Cadence</Label>
                      <Select
                        value={runSettings.detect.replayTimeframe}
                        onValueChange={(value) =>
                          setRunSettings((current) => ({
                            ...current,
                            detect: {
                              ...current.detect,
                              replayTimeframe: value,
                            },
                          }))
                        }
                        disabled={backtestMutation.isPending || !runSettings.detect.useOhlcReplay}
                      >
                        <SelectTrigger className="h-8 text-xs font-mono">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {REPLAY_TIMEFRAME_OPTIONS.map((timeframe) => (
                            <SelectItem key={timeframe} value={timeframe}>
                              {timeframe}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-1">
                      <Label htmlFor="backtest-replay-markets" className="text-[10px] uppercase tracking-wide text-muted-foreground">
                        Replay Markets
                      </Label>
                      <Input
                        id="backtest-replay-markets"
                        type="number"
                        min={1}
                        max={300}
                        step={1}
                        className="h-8 text-xs font-mono"
                        value={runSettings.detect.replayMaxMarkets}
                        onChange={(event) =>
                          setRunSettings((current) => ({
                            ...current,
                            detect: {
                              ...current.detect,
                              replayMaxMarkets: event.target.value,
                            },
                          }))
                        }
                        disabled={backtestMutation.isPending || !runSettings.detect.useOhlcReplay}
                      />
                    </div>
                    <div className="space-y-1">
                      <Label htmlFor="backtest-replay-steps" className="text-[10px] uppercase tracking-wide text-muted-foreground">
                        Replay Steps
                      </Label>
                      <Input
                        id="backtest-replay-steps"
                        type="number"
                        min={1}
                        max={500}
                        step={1}
                        className="h-8 text-xs font-mono"
                        value={runSettings.detect.replayMaxSteps}
                        onChange={(event) =>
                          setRunSettings((current) => ({
                            ...current,
                            detect: {
                              ...current.detect,
                              replayMaxSteps: event.target.value,
                            },
                          }))
                        }
                        disabled={backtestMutation.isPending || !runSettings.detect.useOhlcReplay}
                      />
                    </div>
                  </div>

                  <div className="space-y-1">
                    <Label htmlFor="backtest-max-opportunities" className="text-[10px] uppercase tracking-wide text-muted-foreground">
                      Max Opportunities Returned
                    </Label>
                    <Input
                      id="backtest-max-opportunities"
                      type="number"
                      min={1}
                      max={500}
                      step={1}
                      className="h-8 text-xs font-mono"
                      value={runSettings.detect.maxOpportunities}
                      onChange={(event) =>
                        setRunSettings((current) => ({
                          ...current,
                          detect: {
                            ...current.detect,
                            maxOpportunities: event.target.value,
                          },
                        }))
                      }
                      disabled={backtestMutation.isPending}
                    />
                  </div>
                </div>

                <div className="rounded-md border border-border/40 bg-background/40 p-3 space-y-3">
                  <div className="flex items-center gap-2">
                    <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                    <div>
                      <p className="text-[11px] font-medium">Evaluate</p>
                      <p className="text-[10px] text-muted-foreground">How many recent signals to score.</p>
                    </div>
                  </div>

                  <div className="space-y-1">
                    <Label htmlFor="backtest-max-signals" className="text-[10px] uppercase tracking-wide text-muted-foreground">
                      Max Signals
                    </Label>
                    <Input
                      id="backtest-max-signals"
                      type="number"
                      min={1}
                      max={500}
                      step={1}
                      className="h-8 text-xs font-mono"
                      value={runSettings.evaluate.maxSignals}
                      onChange={(event) =>
                        setRunSettings((current) => ({
                          ...current,
                          evaluate: {
                            ...current.evaluate,
                            maxSignals: event.target.value,
                          },
                        }))
                      }
                      disabled={backtestMutation.isPending}
                    />
                  </div>

                  <p className="text-[10px] text-muted-foreground">
                    The evaluator scores the newest qualifying signal emissions first, capped at this count.
                  </p>
                </div>

                <div className="rounded-md border border-border/40 bg-background/40 p-3 space-y-3">
                  <div className="flex items-center gap-2">
                    <Gauge className="w-3.5 h-3.5 text-amber-400" />
                    <div>
                      <p className="text-[11px] font-medium">Exit</p>
                      <p className="text-[10px] text-muted-foreground">How much open inventory to inspect.</p>
                    </div>
                  </div>

                  <div className="space-y-1">
                    <Label htmlFor="backtest-max-positions" className="text-[10px] uppercase tracking-wide text-muted-foreground">
                      Max Open Positions
                    </Label>
                    <Input
                      id="backtest-max-positions"
                      type="number"
                      min={1}
                      max={500}
                      step={1}
                      className="h-8 text-xs font-mono"
                      value={runSettings.exit.maxPositions}
                      onChange={(event) =>
                        setRunSettings((current) => ({
                          ...current,
                          exit: {
                            ...current.exit,
                            maxPositions: event.target.value,
                          },
                        }))
                      }
                      disabled={backtestMutation.isPending}
                    />
                  </div>

                  <p className="text-[10px] text-muted-foreground">
                    Exit backtests walk the most recent open positions first, up to this cap.
                  </p>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-1 rounded-md bg-muted/50 p-1">
              {BACKTEST_MODES.map((candidate) => (
                <div
                  key={candidate}
                  className={cn(
                    'rounded px-2 py-1 text-[10px]',
                    hasSuiteResults
                      ? integratedScore.phasePassed[candidate]
                        ? 'border border-emerald-500/30 bg-emerald-500/5 text-emerald-300'
                        : 'border border-red-500/30 bg-red-500/5 text-red-300'
                      : 'border border-border/30 bg-background text-muted-foreground'
                  )}
                >
                  <p className="uppercase tracking-wide opacity-80">{MODE_LABELS[candidate].label}</p>
                  <p className="font-mono">{hasSuiteResults ? `${fmt(integratedScore.phaseScores[candidate])}/100` : 'Pending'}</p>
                </div>
              ))}
            </div>

            <Button
              size="sm"
              className="w-full gap-2"
              onClick={() => backtestMutation.mutate()}
              disabled={backtestMutation.isPending || !sourceCode.trim()}
            >
              {backtestMutation.isPending ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  {runningLabel}
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5" />
                  Run Full Backtest Suite
                </>
              )}
            </Button>
          </div>

          <ScrollArea className="flex-1 min-h-0 px-4 py-3">
            {backtestMutation.isError && (
              <div className="rounded-lg border border-red-500/30 bg-red-500/5 p-3 mb-3">
                <div className="flex items-center gap-2 text-red-400 text-xs font-medium mb-1">
                  <XCircle className="w-3.5 h-3.5" />
                  Request Failed
                </div>
                <p className="text-[11px] text-red-300/80">
                  Failed to run backtest. Check that the backend is running.
                </p>
              </div>
            )}

            {!hasSuiteResults && !backtestMutation.isPending && !backtestMutation.isError && (
              <div className="text-center py-12 space-y-2">
                <FlaskConical className="w-8 h-8 mx-auto text-muted-foreground/30" />
                <p className="text-xs text-muted-foreground">
                  Run a full backtest suite to benchmark this strategy across{' '}
                  {variant === 'opportunity'
                    ? 'live+replay detection, evaluation, and exits'
                    : 'signal gating and open-position exits'}
                  .
                </p>
                <p className="text-[10px] text-muted-foreground/60">
                  The score only passes when all three phases succeed with usable output.
                </p>
              </div>
            )}

            {backtestMutation.isPending && (
              <div className="text-center py-12 space-y-3">
                <Loader2 className="w-8 h-8 mx-auto text-primary animate-spin" />
                <p className="text-xs text-muted-foreground">{runningLabel}</p>
                <p className="text-[10px] text-muted-foreground/60">
                  Running Detect, Evaluate, and Exit in sequence.
                </p>
              </div>
            )}

            {hasSuiteResults && !backtestMutation.isPending && (
              <div className="space-y-3">
                <div
                  className={cn(
                    'rounded-lg border p-3',
                    overallTone === 'good'
                      ? 'border-emerald-500/30 bg-emerald-500/5'
                      : overallTone === 'warn'
                        ? 'border-amber-500/30 bg-amber-500/5'
                        : 'border-red-500/30 bg-red-500/5'
                  )}
                >
                  <div
                    className={cn(
                      'flex items-center gap-2 text-xs font-medium',
                      overallTone === 'good'
                        ? 'text-emerald-400'
                        : overallTone === 'warn'
                          ? 'text-amber-400'
                          : 'text-red-400'
                    )}
                  >
                    {integratedScore.passedPhases === BACKTEST_MODES.length ? (
                      <CheckCircle2 className="w-3.5 h-3.5" />
                    ) : integratedScore.passedPhases > 0 ? (
                      <AlertTriangle className="w-3.5 h-3.5" />
                    ) : (
                      <XCircle className="w-3.5 h-3.5" />
                    )}
                    Integrated score
                    <span className="ml-auto text-base font-mono font-semibold opacity-90">
                      {fmt(integratedScore.overall)}/100
                    </span>
                  </div>
                  <div className="mt-2 grid grid-cols-2 md:grid-cols-3 gap-2 text-[10px]">
                    <div className="rounded border border-border/30 bg-black/10 px-2 py-1.5">
                      <span className="text-muted-foreground">Readiness</span>
                      <p className="font-mono font-medium">{fmt(integratedScore.readiness)}%</p>
                    </div>
                    <div className="rounded border border-border/30 bg-black/10 px-2 py-1.5">
                      <span className="text-muted-foreground">Phase Passes</span>
                      <p className="font-mono font-medium">
                        {fmt(integratedScore.passedPhases)}/{fmt(BACKTEST_MODES.length)}
                      </p>
                    </div>
                    <div className="rounded border border-border/30 bg-black/10 px-2 py-1.5">
                      <span className="text-muted-foreground">Suite Runtime</span>
                      <p className="font-mono font-medium">{fmtMs(totalRuntimeMs)}</p>
                    </div>
                  </div>
                  <div className="mt-2 grid grid-cols-1 md:grid-cols-3 gap-2 text-[10px]">
                    {BACKTEST_MODES.map((candidate) => (
                      <div
                        key={candidate}
                        className={cn(
                          'rounded border px-2 py-1.5',
                          integratedScore.phasePassed[candidate]
                            ? 'border-emerald-500/30 bg-emerald-500/5'
                            : 'border-red-500/30 bg-red-500/5'
                        )}
                      >
                        <div className="flex items-center gap-1.5">
                          <span className="uppercase tracking-wide opacity-75">{MODE_LABELS[candidate].label}</span>
                          <span className="ml-auto font-mono">{fmt(integratedScore.phaseScores[candidate])}/100</span>
                        </div>
                        <div className="mt-1 flex items-center justify-between font-mono">
                          <span>Samples {fmt(integratedScore.phaseSamples[candidate])}</span>
                          <span>Data {fmt(integratedScore.phaseCompleteness[candidate])}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="mt-2 h-1.5 rounded bg-muted/40 overflow-hidden">
                    <div
                      className={cn(
                        'h-full',
                        integratedScore.overall >= 85
                          ? 'bg-emerald-500/80'
                          : integratedScore.overall >= 65
                            ? 'bg-amber-500/80'
                            : 'bg-red-500/80'
                      )}
                      style={{ width: `${Math.min(100, Math.max(0, integratedScore.overall))}%` }}
                    />
                  </div>
                  <p className="mt-1 text-[10px] text-muted-foreground">
                    Pass rule per phase: successful run, non-zero sample, and at least 70% completeness.
                  </p>
                </div>

                {phaseViews.map((view) => (
                  <div
                    key={view.mode}
                    className={cn(
                      'rounded-lg border p-3 space-y-3',
                      view.result && view.result.success
                        ? 'border-border/40 bg-card/30'
                        : 'border-red-500/20 bg-red-500/5'
                    )}
                  >
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="text-[9px] h-4 font-medium uppercase tracking-wide">
                        {phaseLabel(view.mode)}
                      </Badge>
                      <span className="text-[10px] text-muted-foreground">{MODE_LABELS[view.mode].desc}</span>
                    </div>

                    {view.result && (
                      <>
                        <div
                          className={cn(
                            'rounded-lg border p-3',
                            view.result.success
                              ? 'border-emerald-500/30 bg-emerald-500/5'
                              : 'border-red-500/30 bg-red-500/5'
                          )}
                        >
                          <div
                            className={cn(
                              'flex items-center gap-2 text-xs font-medium',
                              view.result.success ? 'text-emerald-400' : 'text-red-400'
                            )}
                          >
                            {view.result.success ? (
                              <CheckCircle2 className="w-3.5 h-3.5" />
                            ) : (
                              <XCircle className="w-3.5 h-3.5" />
                            )}
                            {view.result.success ? `${phaseLabel(view.mode)} completed` : `${phaseLabel(view.mode)} failed`}
                            <span className="ml-auto text-[10px] font-mono opacity-70">{fmtMs(view.result.total_time_ms)}</span>
                          </div>
                          <div className="mt-2 grid grid-cols-2 gap-2 text-[10px]">
                            <div className="rounded border border-border/30 bg-black/10 px-2 py-1.5">
                              <span className="text-muted-foreground">Sample Size</span>
                              <p className="font-mono font-medium">{fmt(view.sampleSize)}</p>
                            </div>
                            <div className="rounded border border-border/30 bg-black/10 px-2 py-1.5">
                              <span className="text-muted-foreground">Data Completeness</span>
                              <p className="font-mono font-medium">{fmt(view.dataCompleteness)}%</p>
                            </div>
                          </div>
                          <div className="mt-2 h-1.5 rounded bg-muted/40 overflow-hidden">
                            <div
                              className={cn(
                                'h-full',
                                view.dataCompleteness >= 90
                                  ? 'bg-emerald-500/80'
                                  : view.dataCompleteness >= 70
                                    ? 'bg-amber-500/80'
                                    : 'bg-red-500/80'
                              )}
                              style={{ width: `${Math.min(100, Math.max(0, view.dataCompleteness))}%` }}
                            />
                          </div>
                        </div>

                        <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-2">
                          {view.summaryCards.map((card) => (
                            <SummaryMetricCard key={`${view.mode}-${card.label}`} card={card} />
                          ))}
                        </div>

                        <div className="rounded-lg border border-border/30 p-3 space-y-1.5">
                          <div className="flex items-center gap-2 text-xs font-medium mb-1.5">
                            <Clock className="w-3.5 h-3.5 text-muted-foreground" />
                            Timing Breakdown
                          </div>
                          <div className="flex items-center justify-between text-[11px]">
                            <span className="text-muted-foreground">Code Load</span>
                            <span className="font-mono">{fmtMs(view.result.load_time_ms)}</span>
                          </div>
                          <div className="flex items-center justify-between text-[11px]">
                            <span className="text-muted-foreground">Data Fetch</span>
                            <span className="font-mono">{fmtMs(view.result.data_fetch_time_ms)}</span>
                          </div>
                          <div className="flex items-center justify-between text-[11px]">
                            <span className="text-muted-foreground">{phaseLabel(view.mode)}</span>
                            <span className="font-mono">{fmtMs(view.phaseMs)}</span>
                          </div>
                          <div className="flex items-center justify-between text-[11px] border-t border-border/20 pt-1.5 mt-1.5">
                            <span className="font-medium">Total</span>
                            <span className="font-mono font-medium">{fmtMs(view.result.total_time_ms)}</span>
                          </div>
                        </div>

                        <div className="rounded-lg border border-border/30 p-3 space-y-2">
                          <div className="flex items-center gap-2 text-xs font-medium">
                            <Database className="w-3.5 h-3.5 text-blue-400" />
                            Outcome Stream
                            <span className="ml-auto text-[10px] text-muted-foreground font-mono">{fmt(view.rows.length)} rows</span>
                          </div>

                          {view.actionCounts.length > 0 && (
                            <div className="flex flex-wrap gap-1.5">
                              {view.actionCounts.map(([action, count]) => (
                                <Badge key={`${view.mode}-${action}`} variant="outline" className="text-[9px] h-4 font-mono">
                                  {action}: {fmt(count)}
                                </Badge>
                              ))}
                            </div>
                          )}

                          {view.rows.length === 0 ? (
                            <div className="text-center py-6 space-y-2">
                              <Target className="w-6 h-6 mx-auto text-muted-foreground/30" />
                              <p className="text-xs text-muted-foreground">No outcomes produced for this run.</p>
                            </div>
                          ) : (
                            <div className="space-y-1.5">
                              {view.rows.map((row) => {
                                const rowKey = `${view.mode}:${row.id}`
                                return (
                                  <OutcomeCard
                                    key={rowKey}
                                    row={row}
                                    expanded={expandedRowId === rowKey}
                                    onToggle={() => setExpandedRowId(expandedRowId === rowKey ? null : rowKey)}
                                  />
                                )
                              })}
                            </div>
                          )}
                        </div>

                        {(view.result.validation_errors.length > 0 || view.hasWarnings || view.result.runtime_error) && (
                          <div className="rounded-lg border border-border/30 p-3 space-y-2">
                            <div className="flex items-center gap-2 text-xs font-medium">
                              <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
                              Diagnostics
                            </div>

                            {view.result.validation_errors.length > 0 && (
                              <div className="rounded border border-red-500/30 bg-red-500/5 p-2 space-y-1">
                                <p className="text-[10px] uppercase tracking-wider text-red-400 font-medium">
                                  Validation Errors
                                </p>
                                {view.result.validation_errors.map((error, index) => (
                                  <p key={index} className="text-[11px] text-red-300/80 font-mono">
                                    {error}
                                  </p>
                                ))}
                              </div>
                            )}

                            {view.hasWarnings && (
                              <div className="rounded border border-amber-500/30 bg-amber-500/5 p-2 space-y-1">
                                <p className="text-[10px] uppercase tracking-wider text-amber-400 font-medium">Warnings</p>
                                {view.result.validation_warnings.map((warning, index) => (
                                  <p key={index} className="text-[11px] text-amber-200/80">
                                    {warning}
                                  </p>
                                ))}
                              </div>
                            )}

                            {view.result.runtime_error && (
                              <div className="rounded border border-red-500/30 bg-red-500/5 p-2 space-y-1">
                                <p className="text-[10px] uppercase tracking-wider text-red-400 font-medium">
                                  Runtime Error
                                </p>
                                <p className="text-[11px] text-red-300/80 font-mono">{view.result.runtime_error}</p>
                                {view.result.runtime_traceback && (
                                  <pre className="text-[10px] text-red-300/60 font-mono bg-black/20 rounded p-2 overflow-x-auto whitespace-pre max-h-48 overflow-y-auto">
                                    {view.result.runtime_traceback}
                                  </pre>
                                )}
                              </div>
                            )}
                          </div>
                        )}
                      </>
                    )}
                  </div>
                ))}
              </div>
            )}
          </ScrollArea>
        </div>
      </SheetContent>
    </Sheet>
  )
}
