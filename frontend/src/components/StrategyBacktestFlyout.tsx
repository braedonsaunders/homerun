import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import {
  FlaskConical,
  Play,
  Clock,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Loader2,
  ChevronDown,
  ChevronRight,
  TrendingUp,
  DollarSign,
  BarChart3,
  Target,
} from 'lucide-react'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { ScrollArea } from './ui/scroll-area'
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from './ui/sheet'
import { cn } from '../lib/utils'
import axios from 'axios'

const api = axios.create({ baseURL: '/api', timeout: 120000 })

// ==================== TYPES ====================

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

interface BacktestResult {
  success: boolean
  strategy_slug: string
  strategy_name: string
  class_name: string
  num_events: number
  num_markets: number
  num_prices: number
  data_source: string
  opportunities: Array<Record<string, any>>
  num_opportunities: number
  quality_reports: QualityReportData[]
  load_time_ms: number
  data_fetch_time_ms: number
  detect_time_ms: number
  total_time_ms: number
  validation_errors: string[]
  validation_warnings: string[]
  runtime_error: string | null
  runtime_traceback: string | null
}

// ==================== HELPERS ====================

function fmt(n: number, decimals = 0): string {
  return n.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })
}

function fmtMs(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

// ==================== OPPORTUNITY CARD ====================

function QualityReportSection({ report }: { report: QualityReportData }) {
  const [filtersOpen, setFiltersOpen] = useState(false)
  const failedCount = report.filters.filter(f => !f.passed).length
  const passedCount = report.filters.filter(f => f.passed).length

  return (
    <div className="border border-border/20 rounded-md overflow-hidden mt-1.5">
      <button
        className="w-full flex items-center gap-2 px-2 py-1.5 text-left hover:bg-card/30 transition-colors"
        onClick={() => setFiltersOpen(!filtersOpen)}
      >
        {filtersOpen ? (
          <ChevronDown className="w-2.5 h-2.5 text-muted-foreground shrink-0" />
        ) : (
          <ChevronRight className="w-2.5 h-2.5 text-muted-foreground shrink-0" />
        )}
        <span className="text-[9px] uppercase tracking-wider text-muted-foreground">Quality Filters</span>
        <div className="ml-auto flex items-center gap-1.5">
          {passedCount > 0 && (
            <span className="text-[9px] text-emerald-400 font-mono">{passedCount} pass</span>
          )}
          {failedCount > 0 && (
            <span className="text-[9px] text-red-400 font-mono">{failedCount} fail</span>
          )}
          <Badge
            variant="outline"
            className={cn(
              'text-[8px] h-3.5',
              report.passed ? 'border-emerald-500/30 text-emerald-400' : 'border-red-500/30 text-red-400'
            )}
          >
            {report.passed ? 'PASS' : 'FAIL'}
          </Badge>
        </div>
      </button>
      {filtersOpen && (
        <div className="px-2 pb-2 space-y-0.5 border-t border-border/10">
          {report.filters.map((f, i) => (
            <div key={i} className="flex items-start gap-1.5 py-0.5">
              {f.passed ? (
                <CheckCircle2 className="w-2.5 h-2.5 text-emerald-400 mt-0.5 shrink-0" />
              ) : (
                <XCircle className="w-2.5 h-2.5 text-red-400 mt-0.5 shrink-0" />
              )}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5">
                  <span className="text-[9px] font-mono text-muted-foreground">{f.filter_name}</span>
                </div>
                <p className="text-[9px] text-muted-foreground/80 leading-tight">{f.reason}</p>
                {f.threshold != null && f.actual_value != null && (
                  <div className="flex gap-2 text-[8px] font-mono mt-0.5">
                    <span className="text-amber-400/70">threshold: {String(typeof f.threshold === 'number' ? f.threshold.toFixed(2) : f.threshold)}</span>
                    <span className={f.passed ? 'text-emerald-400/70' : 'text-red-400/70'}>actual: {String(typeof f.actual_value === 'number' ? f.actual_value.toFixed(2) : f.actual_value)}</span>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function OpportunityCard({ opp, index, qualityReport }: { opp: Record<string, any>; index: number; qualityReport?: QualityReportData }) {
  const [expanded, setExpanded] = useState(false)
  const roi = opp.roi_percent ?? opp.roi ?? 0
  const cost = opp.total_cost ?? 0
  const risk = opp.risk_score ?? 0
  const title = opp.title || `Opportunity #${index + 1}`
  const numMarkets = opp.markets?.length ?? 0
  const positions = opp.positions_to_take ?? []

  return (
    <div className="border border-border/40 rounded-lg overflow-hidden">
      <button
        className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-card/50 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? (
          <ChevronDown className="w-3 h-3 text-muted-foreground shrink-0" />
        ) : (
          <ChevronRight className="w-3 h-3 text-muted-foreground shrink-0" />
        )}
        <span className="text-[11px] font-medium truncate flex-1">{title}</span>
        {qualityReport && (
          <Badge
            variant="outline"
            className={cn(
              'text-[8px] h-3.5 shrink-0 mr-1',
              qualityReport.passed ? 'border-emerald-500/20 text-emerald-400/70' : 'border-red-500/20 text-red-400/70'
            )}
          >
            {qualityReport.passed ? 'QF' : 'QF FAIL'}
          </Badge>
        )}
        <Badge
          variant="outline"
          className={cn(
            'text-[9px] h-4 shrink-0',
            roi > 0 ? 'border-emerald-500/30 text-emerald-400' : 'border-red-500/30 text-red-400'
          )}
        >
          {roi > 0 ? '+' : ''}{fmt(roi, 2)}% ROI
        </Badge>
      </button>
      {expanded && (
        <div className="px-3 pb-3 space-y-2 border-t border-border/20">
          <div className="grid grid-cols-3 gap-2 pt-2">
            <div className="text-center">
              <div className="text-[9px] uppercase tracking-wider text-muted-foreground">Cost</div>
              <div className="text-xs font-mono font-medium">${fmt(cost, 2)}</div>
            </div>
            <div className="text-center">
              <div className="text-[9px] uppercase tracking-wider text-muted-foreground">Risk</div>
              <div className={cn('text-xs font-mono font-medium', risk > 0.7 ? 'text-red-400' : risk > 0.4 ? 'text-amber-400' : 'text-emerald-400')}>
                {fmt(risk, 2)}
              </div>
            </div>
            <div className="text-center">
              <div className="text-[9px] uppercase tracking-wider text-muted-foreground">Markets</div>
              <div className="text-xs font-mono font-medium">{numMarkets}</div>
            </div>
          </div>
          {opp.description && (
            <p className="text-[10px] text-muted-foreground leading-relaxed">{opp.description}</p>
          )}
          {positions.length > 0 && (
            <div className="space-y-1">
              <div className="text-[9px] uppercase tracking-wider text-muted-foreground">Positions</div>
              {positions.map((pos: any, i: number) => (
                <div key={i} className="flex items-center gap-2 text-[10px]">
                  <Badge variant="outline" className={cn('text-[8px] h-3.5', pos.outcome === 'Yes' ? 'text-emerald-400 border-emerald-500/30' : 'text-red-400 border-red-500/30')}>
                    {pos.action} {pos.outcome}
                  </Badge>
                  <span className="text-muted-foreground truncate">{pos.market}</span>
                  <span className="font-mono ml-auto shrink-0">${fmt(pos.price ?? 0, 2)}</span>
                </div>
              ))}
            </div>
          )}
          {qualityReport && (
            <QualityReportSection report={qualityReport} />
          )}
        </div>
      )}
    </div>
  )
}

// ==================== STAT CARD ====================

function Stat({ label, value, icon: Icon, tone = 'neutral' }: {
  label: string
  value: string
  icon: React.ComponentType<{ className?: string }>
  tone?: 'good' | 'warn' | 'bad' | 'neutral'
}) {
  const toneClass =
    tone === 'good' ? 'border-emerald-500/30 bg-emerald-500/5 text-emerald-300' :
    tone === 'warn' ? 'border-amber-500/30 bg-amber-500/5 text-amber-300' :
    tone === 'bad' ? 'border-red-500/30 bg-red-500/5 text-red-300' :
    'border-border/40 bg-card/30 text-foreground'

  return (
    <div className={cn('rounded-md border px-2.5 py-2', toneClass)}>
      <div className="flex items-center gap-1.5 mb-1">
        <Icon className="w-3 h-3 opacity-60" />
        <span className="text-[9px] uppercase tracking-wider opacity-70">{label}</span>
      </div>
      <div className="text-sm font-mono font-semibold">{value}</div>
    </div>
  )
}

// ==================== MAIN FLYOUT ====================

type BacktestMode = 'detect' | 'evaluate' | 'exit'

const MODE_ENDPOINTS: Record<BacktestMode, string> = {
  detect: '/validation/code-backtest',
  evaluate: '/validation/code-backtest/evaluate',
  exit: '/validation/code-backtest/exit',
}

const MODE_LABELS: Record<BacktestMode, { label: string; desc: string; running: string }> = {
  detect: { label: 'Detect', desc: 'Find opportunities in current market data', running: 'Running detection...' },
  evaluate: { label: 'Evaluate', desc: 'Test signal gating against recent trade signals', running: 'Evaluating signals...' },
  exit: { label: 'Exit', desc: 'Test exit logic against current open positions', running: 'Testing exit logic...' },
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
  const [result, setResult] = useState<BacktestResult | null>(null)
  const [mode, setMode] = useState<BacktestMode>('detect')

  const backtestMutation = useMutation({
    mutationFn: async () => {
      const endpoint = MODE_ENDPOINTS[mode]
      const { data } = await api.post(endpoint, {
        source_code: sourceCode,
        slug,
        config: config || {},
      })
      return data as BacktestResult
    },
    onSuccess: (data) => setResult(data),
    onError: () => setResult(null),
  })

  const hasWarnings = result && result.validation_warnings.length > 0

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-full sm:max-w-xl p-0">
        <div className="h-full min-h-0 flex flex-col">
          {/* Header */}
          <div className="border-b border-border px-4 py-3 space-y-3">
            <SheetHeader className="space-y-1 text-left">
              <SheetTitle className="text-base flex items-center gap-2">
                <FlaskConical className="w-4 h-4" />
                Strategy Backtest
                <Badge variant="outline" className="text-[9px] h-4 font-normal">Live</Badge>
              </SheetTitle>
              <SheetDescription>
                {MODE_LABELS[mode].desc}
              </SheetDescription>
            </SheetHeader>

            {/* Mode selector */}
            <div className="flex gap-1 rounded-md bg-muted/50 p-0.5">
              {(['detect', 'evaluate', 'exit'] as BacktestMode[]).map((m) => (
                <button
                  key={m}
                  className={cn(
                    'flex-1 text-[10px] font-medium py-1 px-2 rounded transition-colors',
                    mode === m
                      ? 'bg-background text-foreground shadow-sm'
                      : 'text-muted-foreground hover:text-foreground'
                  )}
                  onClick={() => { setMode(m); setResult(null) }}
                >
                  {MODE_LABELS[m].label}
                </button>
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
                  {MODE_LABELS[mode].running}
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5" />
                  Run {MODE_LABELS[mode].label} Backtest
                </>
              )}
            </Button>
          </div>

          {/* Content */}
          <ScrollArea className="flex-1 min-h-0 px-4 py-3">
            {/* Error from mutation itself */}
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

            {/* No result yet */}
            {!result && !backtestMutation.isPending && !backtestMutation.isError && (
              <div className="text-center py-12 space-y-2">
                <FlaskConical className="w-8 h-8 mx-auto text-muted-foreground/30" />
                <p className="text-xs text-muted-foreground">
                  Click "Run Backtest" to test your strategy against {variant === 'opportunity' ? 'live market data' : 'current signals'}.
                </p>
                <p className="text-[10px] text-muted-foreground/60">
                  Your code will be compiled, loaded, and run against the current market snapshot.
                </p>
              </div>
            )}

            {/* Loading */}
            {backtestMutation.isPending && (
              <div className="text-center py-12 space-y-3">
                <Loader2 className="w-8 h-8 mx-auto text-primary animate-spin" />
                <p className="text-xs text-muted-foreground">
                  Compiling strategy and running detection...
                </p>
                <p className="text-[10px] text-muted-foreground/60">
                  This may take up to 60 seconds for I/O-heavy strategies.
                </p>
              </div>
            )}

            {/* Results */}
            {result && !backtestMutation.isPending && (
              <div className="space-y-3">
                {/* Status banner */}
                {result.success ? (
                  <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-3">
                    <div className="flex items-center gap-2 text-emerald-400 text-xs font-medium">
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      Detection completed successfully
                      <span className="ml-auto text-[10px] text-emerald-400/60 font-mono">
                        {fmtMs(result.total_time_ms)}
                      </span>
                    </div>
                  </div>
                ) : (
                  <div className="rounded-lg border border-red-500/30 bg-red-500/5 p-3">
                    <div className="flex items-center gap-2 text-red-400 text-xs font-medium">
                      <XCircle className="w-3.5 h-3.5" />
                      Backtest failed
                    </div>
                  </div>
                )}

                {/* Summary stats — adaptive to mode */}
                <div className="grid grid-cols-2 gap-2">
                  {mode === 'detect' && (
                    <>
                      <Stat label="Opportunities" value={fmt(result.num_opportunities || 0)} icon={Target} tone={(result.num_opportunities || 0) > 0 ? 'good' : 'neutral'} />
                      <Stat label="Markets Scanned" value={fmt(result.num_markets || 0)} icon={BarChart3} />
                      <Stat label="Events" value={fmt(result.num_events || 0)} icon={TrendingUp} />
                      <Stat label="Data Source" value={result.data_source || 'N/A'} icon={DollarSign} />
                    </>
                  )}
                  {mode === 'evaluate' && (
                    <>
                      <Stat label="Signals Tested" value={fmt((result as any).num_signals || 0)} icon={Target} />
                      <Stat label="Selected" value={fmt((result as any).selected || 0)} icon={CheckCircle2} tone={(result as any).selected > 0 ? 'good' : 'neutral'} />
                      <Stat label="Skipped" value={fmt((result as any).skipped || 0)} icon={XCircle} />
                      <Stat label="Blocked" value={fmt((result as any).blocked || 0)} icon={AlertTriangle} />
                    </>
                  )}
                  {mode === 'exit' && (
                    <>
                      <Stat label="Positions Tested" value={fmt((result as any).num_positions || 0)} icon={Target} />
                      <Stat label="Would Close" value={fmt((result as any).would_close || 0)} icon={XCircle} tone={(result as any).would_close > 0 ? 'good' : 'neutral'} />
                      <Stat label="Would Hold" value={fmt((result as any).would_hold || 0)} icon={CheckCircle2} />
                      <Stat label="" value="" icon={DollarSign} />
                    </>
                  )}
                </div>

                {/* Timing breakdown */}
                <div className="rounded-lg border border-border/30 p-3 space-y-1.5">
                  <div className="flex items-center gap-2 text-xs font-medium mb-2">
                    <Clock className="w-3.5 h-3.5 text-muted-foreground" />
                    Timing Breakdown
                  </div>
                  {[
                    { label: 'Code Load', ms: result.load_time_ms },
                    { label: 'Data Fetch', ms: result.data_fetch_time_ms },
                    { label: 'Detection', ms: result.detect_time_ms },
                  ].map(({ label, ms }) => (
                    <div key={label} className="flex items-center justify-between text-[11px]">
                      <span className="text-muted-foreground">{label}</span>
                      <span className="font-mono text-foreground">{fmtMs(ms)}</span>
                    </div>
                  ))}
                  <div className="flex items-center justify-between text-[11px] border-t border-border/20 pt-1.5 mt-1.5">
                    <span className="font-medium">Total</span>
                    <span className="font-mono font-medium">{fmtMs(result.total_time_ms)}</span>
                  </div>
                </div>

                {/* Validation errors */}
                {result.validation_errors.length > 0 && (
                  <div className="rounded-lg border border-red-500/30 bg-red-500/5 p-3 space-y-2">
                    <div className="flex items-center gap-2 text-red-400 text-xs font-medium">
                      <XCircle className="w-3.5 h-3.5" />
                      Validation Errors
                    </div>
                    {result.validation_errors.map((err, i) => (
                      <p key={i} className="text-[11px] text-red-300/80 font-mono">{err}</p>
                    ))}
                  </div>
                )}

                {/* Validation warnings */}
                {hasWarnings && (
                  <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 space-y-2">
                    <div className="flex items-center gap-2 text-amber-400 text-xs font-medium">
                      <AlertTriangle className="w-3.5 h-3.5" />
                      Warnings
                    </div>
                    {result.validation_warnings.map((warn, i) => (
                      <p key={i} className="text-[11px] text-amber-300/80">{warn}</p>
                    ))}
                  </div>
                )}

                {/* Runtime error */}
                {result.runtime_error && (
                  <div className="rounded-lg border border-red-500/30 bg-red-500/5 p-3 space-y-2">
                    <div className="flex items-center gap-2 text-red-400 text-xs font-medium">
                      <XCircle className="w-3.5 h-3.5" />
                      Runtime Error
                    </div>
                    <p className="text-[11px] text-red-300/80 font-mono">{result.runtime_error}</p>
                    {result.runtime_traceback && (
                      <pre className="text-[10px] text-red-300/60 font-mono bg-black/20 rounded p-2 overflow-x-auto whitespace-pre max-h-48 overflow-y-auto">
                        {result.runtime_traceback}
                      </pre>
                    )}
                  </div>
                )}

                {/* DETECT: Opportunities list */}
                {mode === 'detect' && (result.opportunities || []).length > 0 && (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-xs font-medium">
                      <Target className="w-3.5 h-3.5 text-emerald-400" />
                      Detected Opportunities ({result.num_opportunities})
                    </div>
                    <div className="space-y-1.5">
                      {(result.opportunities || []).map((opp, i) => {
                        const qr = (result.quality_reports || [])[i] || undefined
                        return <OpportunityCard key={opp.id || i} opp={opp} index={i} qualityReport={qr} />
                      })}
                    </div>
                  </div>
                )}

                {/* EVALUATE: Decisions list */}
                {mode === 'evaluate' && ((result as any).decisions || []).length > 0 && (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-xs font-medium">
                      <Target className="w-3.5 h-3.5 text-blue-400" />
                      Evaluate Decisions ({(result as any).decisions?.length || 0})
                    </div>
                    <div className="space-y-1.5">
                      {((result as any).decisions || []).map((d: any, i: number) => (
                        <div key={i} className="border border-border/40 rounded-lg p-2 space-y-1">
                          <div className="flex items-center justify-between">
                            <span className="text-[11px] text-muted-foreground font-mono truncate">{d.source || d.strategy_type || `Signal #${i+1}`}</span>
                            <Badge variant="outline" className={cn('text-[9px] h-4', d.decision === 'selected' ? 'border-emerald-500/30 text-emerald-400' : d.decision === 'blocked' ? 'border-red-500/30 text-red-400' : 'border-amber-500/30 text-amber-400')}>
                              {d.decision}
                            </Badge>
                          </div>
                          {d.reason && <p className="text-[10px] text-muted-foreground">{d.reason}</p>}
                          {d.size_usd && <p className="text-[10px] font-mono text-emerald-400">${d.size_usd.toFixed(2)}</p>}
                          {(d.checks || []).length > 0 && (
                            <div className="space-y-0.5 mt-1">
                              {d.checks.map((c: any, j: number) => (
                                <div key={j} className="flex items-center gap-1 text-[9px]">
                                  {c.passed ? <CheckCircle2 className="w-2.5 h-2.5 text-emerald-400" /> : <XCircle className="w-2.5 h-2.5 text-red-400" />}
                                  <span className="text-muted-foreground">{c.check_label || c.check_key}</span>
                                  {c.score != null && <span className="font-mono ml-auto">{c.score.toFixed(2)}</span>}
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* EXIT: Exit decisions list */}
                {mode === 'exit' && ((result as any).exit_decisions || []).length > 0 && (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-xs font-medium">
                      <Target className="w-3.5 h-3.5 text-orange-400" />
                      Exit Decisions ({(result as any).exit_decisions?.length || 0})
                    </div>
                    <div className="space-y-1.5">
                      {((result as any).exit_decisions || []).map((d: any, i: number) => (
                        <div key={i} className="border border-border/40 rounded-lg p-2 space-y-1">
                          <div className="flex items-center justify-between">
                            <span className="text-[11px] text-muted-foreground font-mono truncate">{d.market_id || `Position #${i+1}`}</span>
                            <Badge variant="outline" className={cn('text-[9px] h-4', d.action === 'close' ? 'border-red-500/30 text-red-400' : 'border-emerald-500/30 text-emerald-400')}>
                              {d.action}
                            </Badge>
                          </div>
                          {d.reason && <p className="text-[10px] text-muted-foreground">{d.reason}</p>}
                          <div className="flex gap-3 text-[10px] font-mono">
                            <span>Entry: ${d.entry_price?.toFixed(3)}</span>
                            <span>Now: ${d.current_price?.toFixed(3)}</span>
                            <span className={d.pnl_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}>{d.pnl_pct >= 0 ? '+' : ''}{d.pnl_pct}%</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* No results for detect mode */}
                {mode === 'detect' && result.success && (result.num_opportunities || 0) === 0 && (
                  <div className="text-center py-6 space-y-2">
                    <Target className="w-6 h-6 mx-auto text-muted-foreground/30" />
                    <p className="text-xs text-muted-foreground">
                      No opportunities detected in the current market snapshot.
                    </p>
                    <p className="text-[10px] text-muted-foreground/60">
                      This is normal — your strategy may have strict filters or current markets may not match its criteria.
                    </p>
                  </div>
                )}

                {/* No results for evaluate mode */}
                {mode === 'evaluate' && result.success && ((result as any).decisions || []).length === 0 && (
                  <div className="text-center py-6 space-y-2">
                    <Target className="w-6 h-6 mx-auto text-muted-foreground/30" />
                    <p className="text-xs text-muted-foreground">No recent signals to evaluate.</p>
                  </div>
                )}

                {/* No results for exit mode */}
                {mode === 'exit' && result.success && ((result as any).exit_decisions || []).length === 0 && (
                  <div className="text-center py-6 space-y-2">
                    <Target className="w-6 h-6 mx-auto text-muted-foreground/30" />
                    <p className="text-xs text-muted-foreground">No open positions to test exit logic against.</p>
                  </div>
                )}
              </div>
            )}
          </ScrollArea>
        </div>
      </SheetContent>
    </Sheet>
  )
}
